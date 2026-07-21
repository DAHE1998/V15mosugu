/*
 * video_cut_v2.c — GPU Segment Renderer
 *
 * 架构：一次 NVDEC 解码 → 逐段创建 NVENC 编码 → flush → 销毁
 * 每段编码器状态绝对干净，杜绝跨段污染导致的画面抽搐。
 *
 * Compile:
 *   gcc -o video_cut video_cut_v2.c \
 *     -I/home/dahe/ffmpeg_dev/usr/include/x86_64-linux-gnu \
 *     -L/usr/lib/x86_64-linux-gnu \
 *     -l:libavformat.so.60 -l:libavcodec.so.60 -l:libavutil.so.58 \
 *     -O2 -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/hwcontext.h>
#include <libavutil/error.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>

#define CHECK(err, msg) do { \
    if ((err) < 0) { \
        char _buf_[AV_ERROR_MAX_STRING_SIZE]; \
        av_strerror(err, _buf_, sizeof(_buf_)); \
        fprintf(stderr, "ERROR: %s: %s\n", msg, _buf_); \
        exit(1); \
    } \
} while (0)

#define FATAL(msg) do { fprintf(stderr, "ERROR: %s\n", msg); exit(1); } while (0)

/* ── shot data ── */
typedef struct { int id, start, end; } Shot;

/* ── JSON parser (same as original) ── */
static char *read_file(const char *path, long *len) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    *len = ftell(fp);
    rewind(fp);
    char *buf = malloc(*len + 1);
    if (!buf) { fclose(fp); return NULL; }
    size_t n = fread(buf, 1, *len, fp);
    fclose(fp);
    if (n != (size_t)*len) { free(buf); return NULL; }
    buf[*len] = '\0';
    return buf;
}

static int extract_str(const char *j, const char *k, char *out, int sz) {
    char s[64];
    snprintf(s, sizeof(s), "\"%s\"", k);
    const char *p = strstr(j, s);
    if (!p) return -1;
    p = strchr(p, ':');
    if (!p) return -1;
    p++;
    while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    if (*p != '"') return -1;
    p++;
    int i = 0;
    while (*p && *p != '"' && i < sz - 1) {
        if (*p == '\\' && *(p+1)) p++;
        out[i++] = *p++;
    }
    out[i] = '\0';
    return 0;
}

static int parse_shots(const char *j, Shot *s, int max) {
    const char *p = strstr(j, "\"shots\"");
    if (!p) return 0;
    p = strchr(p, '[');
    if (!p) return 0;
    p++;
    int n = 0;
    while (*p && *p != ']' && n < max) {
        while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ',')) p++;
        if (*p != '{') break;
        int d = 0;
        const char *e = p;
        while (*e) {
            if (*e == '{') d++;
            if (*e == '}') { d--; if (d == 0) break; }
            e++;
        }
        if (!*e) break;
        int olen = (int)(e - p + 1);
        char obj[4096];
        if (olen >= (int)sizeof(obj)) { p = e + 1; continue; }
        strncpy(obj, p, olen);
        obj[olen] = '\0';

        int id = -1, st = -1, ed = -1;
        char *k;
        k = strstr(obj, "\"id\"");
        if (k) { k = strchr(k, ':'); if (k) id = (int)strtol(k+1, NULL, 10); }
        k = strstr(obj, "\"start\"");
        if (k) { k = strchr(k, ':'); if (k) st = (int)strtol(k+1, NULL, 10); }
        k = strstr(obj, "\"end\"");
        if (k) { k = strchr(k, ':'); if (k) ed = (int)strtol(k+1, NULL, 10); }

        if (id >= 0 && st >= 0 && ed >= 0 && ed >= st)
            s[n++] = (Shot){id, st, ed};
        p = e + 1;
    }
    return n;
}

static int shot_cmp(const void *a, const void *b) {
    return ((const Shot *)a)->start - ((const Shot *)b)->start;
}

/* ── GPU device ── */
static AVBufferRef *create_cuda_device(void) {
    AVBufferRef *dev = NULL;
    CHECK(av_hwdevice_ctx_create(&dev, AV_HWDEVICE_TYPE_CUDA, "0", NULL, 0), "CUDA device");
    return dev;
}

/* ── encoder (per-segment) ── */
static AVCodecContext *open_encoder(int w, int h, AVRational fps,
                                     AVBufferRef *hw_frames, AVBufferRef *hw_dev) {
    const AVCodec *ec = avcodec_find_encoder_by_name("h264_nvenc");
    if (!ec) FATAL("NVENC not found");

    AVCodecContext *enc = avcodec_alloc_context3(ec);
    if (!enc) FATAL("alloc encoder");

    /* Use standard NTSC timebase/framerate, matching Python CLI version */
    AVRational std_fps  = {30000, 1001};
    AVRational std_time = {1001, 30000};

    enc->width          = w;
    enc->height         = h;
    enc->pix_fmt        = AV_PIX_FMT_CUDA;
    enc->time_base      = std_time;
    enc->framerate      = std_fps;
    enc->gop_size       = 30;
    /* Allow B-frames — Python version proves they work correctly */
    enc->hw_frames_ctx  = av_buffer_ref(hw_frames);
    enc->hw_device_ctx  = av_buffer_ref(hw_dev);
    enc->flags         |= AV_CODEC_FLAG_GLOBAL_HEADER;
    enc->profile        = FF_PROFILE_H264_HIGH;

    CHECK(av_opt_set(enc->priv_data, "preset", "p4", 0), "preset");
    CHECK(av_opt_set_int(enc->priv_data, "forced-idr", 1, 0), "forced-idr");
    CHECK(av_opt_set_int(enc->priv_data, "cq", 26, 0), "cq");

    CHECK(avcodec_open2(enc, ec, NULL), "open encoder");
    return enc;
}

/* ── muxer (per-segment) ── */
static void mux_open(AVFormatContext **fmt, AVStream **st,
                     const char *path, AVCodecContext *enc) {
    CHECK(avformat_alloc_output_context2(fmt, NULL, "mp4", path), "alloc muxer");
    *st = avformat_new_stream(*fmt, NULL);
    if (!*st) FATAL("new stream");

    CHECK(avcodec_parameters_from_context((*st)->codecpar, enc), "copy params");
    (*st)->time_base = enc->time_base;

    CHECK(avio_open(&(*fmt)->pb, path, AVIO_FLAG_WRITE), "open file");
    CHECK(avformat_write_header(*fmt, NULL), "write header");
}

static void mux_write_pkt(AVFormatContext *fmt, AVStream *st,
                          AVPacket *pkt, int64_t *pts_counter,
                          int frame_ticks) {
    pkt->stream_index = st->index;
    pkt->pts = (*pts_counter) * frame_ticks;
    /* Let encoder's DTS stand — needed for B-frame reordering */
    pkt->duration = frame_ticks;
    CHECK(av_interleaved_write_frame(fmt, pkt), "write packet");
    (*pts_counter)++;
}

static void mux_close(AVFormatContext *fmt) {
    av_write_trailer(fmt);
    avio_closep(&fmt->pb);
    avformat_free_context(fmt);
}

/* ── drain encoder: pull all ready packets into muxer ── */
static void drain_encoder(AVCodecContext *enc, AVFormatContext *fmt,
                          AVStream *st, int64_t *pts_counter,
                          int frame_ticks) {
    AVPacket *pkt = av_packet_alloc();
    while (1) {
        int ret = avcodec_receive_packet(enc, pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
        CHECK(ret, "receive packet");
        mux_write_pkt(fmt, st, pkt, pts_counter, frame_ticks);
        av_packet_unref(pkt);
    }
    av_packet_free(&pkt);
}

/* ── process one segment: create encoder → encode → flush → destroy ── */
static void process_segment(const Shot *shot, const char *out_dir,
                             AVCodecContext *dec, AVFrame *dec_frame,
                             AVPacket *in_pkt,
                             AVFormatContext *in_fmt, int vid_idx,
                             int w, int h, AVRational fps,
                             AVBufferRef *hw_frames, AVBufferRef *hw_dev,
                             int64_t *fi) {
    char out_path[4096];
    snprintf(out_path, sizeof(out_path), "%s/segment_%04d.mp4",
             out_dir, shot->id);

    /* 1. Create fresh encoder */
    AVCodecContext *enc = open_encoder(w, h, fps, hw_frames, hw_dev);
    printf("  [%3d] %s  (%d-%d, %d frames)\n",
           shot->id, out_path, shot->start, shot->end,
           shot->end - shot->start + 1);

    /* 2. Create muxer */
    AVFormatContext *fmt = NULL;
    AVStream *st = NULL;
    mux_open(&fmt, &st, out_path, enc);
    int64_t pts_counter = 0;
    int frame_ticks = 1001;  /* 1001 ticks at 1/30000 timebase = 1 frame */

    /* 3. Forward to shot start if needed */
    int dec_eof = 0;
    while (*fi < shot->start) {
        int ret = av_read_frame(in_fmt, in_pkt);
        if (ret < 0) {
            av_packet_unref(in_pkt);
            if (!dec_eof) {
                avcodec_send_packet(dec, NULL);  /* flush decoder */
                dec_eof = 1;
                continue;
            }
            break;
        }
        if (in_pkt->stream_index != vid_idx) {
            av_packet_unref(in_pkt);
            continue;
        }
        CHECK(avcodec_send_packet(dec, in_pkt), "send pkt");
        av_packet_unref(in_pkt);

        while (1) {
            ret = avcodec_receive_frame(dec, dec_frame);
            if (ret == AVERROR(EAGAIN)) break;
            if (ret == AVERROR_EOF)   break;
            CHECK(ret, "recv frame");
            av_frame_unref(dec_frame);
            (*fi)++;
            if (*fi >= shot->start) break;
        }
        if (*fi >= shot->start) break;
    }

    /* 4. Encode this segment's frames */
    int produced = 0;
    while (*fi <= shot->end) {
        /* need more decoded frames? */
        if (*fi < shot->start) { (*fi)++; continue; }  /* shouldn't happen */

        /* Read and decode until we have a frame */
        while (1) {
            int ret = avcodec_receive_frame(dec, dec_frame);
            if (ret == 0) break;  /* got a frame */
            if (ret == AVERROR(EAGAIN)) {
                /* Feed more data to decoder */
                ret = av_read_frame(in_fmt, in_pkt);
                if (ret < 0) {
                    av_packet_unref(in_pkt);
                    /* Flush decoder to drain buffered frames */
                    if (!dec_eof) {
                        avcodec_send_packet(dec, NULL);
                        dec_eof = 1;
                        continue;
                    }
                    goto segment_done;  /* decoder truly empty */
                }
                if (in_pkt->stream_index != vid_idx) {
                    av_packet_unref(in_pkt);
                    continue;
                }
                CHECK(avcodec_send_packet(dec, in_pkt), "send pkt");
                av_packet_unref(in_pkt);
                continue;
            }
            if (ret == AVERROR_EOF) goto segment_done;
            CHECK(ret, "recv frame");
        }

        if (*fi >= shot->start && *fi <= shot->end) {
            /* First frame of segment = IDR */
            if (produced == 0)
                dec_frame->pict_type = AV_PICTURE_TYPE_I;
            else
                dec_frame->pict_type = AV_PICTURE_TYPE_NONE;

            CHECK(avcodec_send_frame(enc, dec_frame), "send to encoder");
            produced++;

            /* Pull ready packets */
            drain_encoder(enc, fmt, st, &pts_counter, frame_ticks);
        }

        av_frame_unref(dec_frame);
        (*fi)++;
    }

segment_done:
    /* 5. Flush encoder */
    CHECK(avcodec_send_frame(enc, NULL), "flush encoder");
    drain_encoder(enc, fmt, st, &pts_counter, frame_ticks);

    /* 6. Close muxer, destroy encoder */
    mux_close(fmt);
    avcodec_free_context(&enc);
}

/* ══════════════════════════════════════════════════════════════ */

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <skeleton.json>\n", argv[0]);
        return 1;
    }
    av_log_set_level(AV_LOG_ERROR);

    /* Parse JSON */
    long json_len;
    char *json = read_file(argv[1], &json_len);
    if (!json) FATAL("Cannot read JSON");

    char video_path[4096] = {0};
    if (extract_str(json, "video", video_path, sizeof(video_path)) < 0)
        FATAL("Missing video field");

    Shot shots[65536];
    int n_shots = parse_shots(json, shots, 65536);
    free(json);

    if (n_shots == 0) FATAL("No shots found");
    qsort(shots, n_shots, sizeof(Shot), shot_cmp);

    /* Output directory */
    char out_dir[4096];
    {
        const char *s = strrchr(argv[1], '/');
        if (s) {
            int d = (int)(s - argv[1]);
            strncpy(out_dir, argv[1], d);
            out_dir[d] = '\0';
        } else {
            strcpy(out_dir, ".");
        }
    }

    /* Open input */
    AVFormatContext *in_fmt = NULL;
    CHECK(avformat_open_input(&in_fmt, video_path, NULL, NULL), "open input");
    CHECK(avformat_find_stream_info(in_fmt, NULL), "stream info");

    int vid_idx = av_find_best_stream(in_fmt, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (vid_idx < 0) FATAL("No video stream");

    AVStream *vst = in_fmt->streams[vid_idx];
    int w = vst->codecpar->width;
    int h = vst->codecpar->height;
    enum AVCodecID codec_id = vst->codecpar->codec_id;

    /* FPS */
    AVRational fps = vst->avg_frame_rate;
    if (fps.num <= 0 || fps.den <= 0) fps = vst->r_frame_rate;
    if (fps.num <= 0 || fps.den <= 0) FATAL("Cannot determine FPS");

    printf("Video: %s\n", video_path);
    printf("Size: %dx%d  FPS: %d/%d = %.4f\n", w, h, fps.num, fps.den,
           (double)fps.num / fps.den);
    printf("Shots: %d\n", n_shots);

    /* ── GPU device + frame context ── */
    AVBufferRef *hw_dev = create_cuda_device();

    AVBufferRef *hw_frames = NULL;
    {
        hw_frames = av_hwframe_ctx_alloc(hw_dev);
        if (!hw_frames) FATAL("alloc hw frames");
        AVHWFramesContext *fc = (AVHWFramesContext *)hw_frames->data;
        fc->format    = AV_PIX_FMT_CUDA;
        fc->sw_format = AV_PIX_FMT_NV12;
        fc->width     = w;
        fc->height    = h;
        fc->initial_pool_size = 4;
        CHECK(av_hwframe_ctx_init(hw_frames), "init hw frames");
    }

    /* ── Decoder (NVDEC, single instance) ── */
    const char *dec_name = NULL;
    if (codec_id == AV_CODEC_ID_H264)      dec_name = "h264_cuvid";
    else if (codec_id == AV_CODEC_ID_HEVC) dec_name = "hevc_cuvid";
    else FATAL("Unsupported codec");

    const AVCodec *dec_codec = avcodec_find_decoder_by_name(dec_name);
    if (!dec_codec) FATAL("NVDEC not found");

    AVCodecContext *dec = avcodec_alloc_context3(dec_codec);
    if (!dec) FATAL("alloc decoder");

    CHECK(avcodec_parameters_to_context(dec, vst->codecpar), "dec params");
    dec->hw_device_ctx = av_buffer_ref(hw_dev);
    dec->hw_frames_ctx = av_buffer_ref(hw_frames);
    CHECK(avcodec_open2(dec, dec_codec, NULL), "open decoder");

    /* ── Seek to first frame we need ── */
    int64_t fi = 0;  /* frame index */

    /* ── Process segments ── */
    AVPacket  *in_pkt    = av_packet_alloc();
    AVFrame   *dec_frame = av_frame_alloc();

    for (int si = 0; si < n_shots; si++) {
        process_segment(&shots[si], out_dir,
                        dec, dec_frame, in_pkt,
                        in_fmt, vid_idx,
                        w, h, fps,
                        hw_frames, hw_dev,
                        &fi);
    }

    printf("\nDone: %d segments\n", n_shots);

    /* Cleanup */
    av_frame_free(&dec_frame);
    av_packet_free(&in_pkt);
    avcodec_free_context(&dec);
    avformat_close_input(&in_fmt);
    av_buffer_unref(&hw_frames);
    av_buffer_unref(&hw_dev);

    return 0;
}
