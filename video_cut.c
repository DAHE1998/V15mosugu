/*
 * video_cut.c — GPU 帧范围切割工具 (fixed)
 *
 * 读 skeleton.json，每个 shot 切一个 MP4。
 * 纯帧号 (Frame Index) 驱动，支持 29.79 fps / VFR 视频平滑导出。
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>

#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/hwcontext.h>
#include <libavutil/error.h>
#include <libavutil/opt.h>

#define CHECK(err, msg) do { \
    if ((err) < 0) { \
        char buf[AV_ERROR_MAX_STRING_SIZE]; \
        av_strerror(err, buf, sizeof(buf)); \
        fprintf(stderr, "ERROR: %s: %s\n", msg, buf); \
        exit(1); \
    } \
} while (0)

#define FATAL(msg) do { \
    fprintf(stderr, "ERROR: %s\n", msg); \
    exit(1); \
} while (0)

typedef struct { int id, start, end; } Shot;

/* ── JSON ── */

static char *read_file(const char *path, long *len) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    *len = ftell(fp);
    rewind(fp);
    char *buf = malloc(*len + 1);
    if (!buf) { fclose(fp); return NULL; }
    size_t n = fread(buf, 1, *len, fp);
    fclose(fp); if (n != (size_t)*len) { free(buf); return NULL; }
    buf[*len] = '\0';
    return buf;
}

static int extract_str(const char *j, const char *k, char *out, int sz) {
    char s[64]; snprintf(s, sizeof(s), "\"%s\"", k);
    const char *p = strstr(j, s); if (!p) return -1;
    p = strchr(p, ':'); if (!p) return -1;
    p++; while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    if (*p != '"') return -1; p++;
    int i = 0; while (*p && *p != '"' && i < sz - 1) { if (*p == '\\' && *(p+1)) p++; out[i++] = *p++; }
    out[i] = '\0'; return 0;
}

static int parse_shots(const char *j, Shot *s, int max) {
    const char *p = strstr(j, "\"shots\""); if (!p) return 0;
    p = strchr(p, '['); if (!p) return 0; p++;
    int n = 0;
    while (*p && *p != ']' && n < max) {
        while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ',')) p++;
        if (*p != '{') break;
        int d = 0; const char *e = p;
        while (*e) { if (*e == '{') d++; if (*e == '}') { d--; if (d == 0) break; } e++; }
        if (!*e) break;
        int olen = (int)(e - p + 1); char obj[4096];
        if (olen >= (int)sizeof(obj)) { p = e + 1; continue; }
        strncpy(obj, p, olen); obj[olen] = '\0';
        int id = -1, st = -1, ed = -1;
        char *k; k = strstr(obj, "\"id\""); if (k) { k = strchr(k, ':'); if (k) id = (int)strtol(k+1, NULL, 10); }
        k = strstr(obj, "\"start\""); if (k) { k = strchr(k, ':'); if (k) st = (int)strtol(k+1, NULL, 10); }
        k = strstr(obj, "\"end\""); if (k) { k = strchr(k, ':'); if (k) ed = (int)strtol(k+1, NULL, 10); }
        if (id >= 0 && st >= 0 && ed >= 0 && ed >= st) s[n++] = (Shot){id, st, ed};
        p = e + 1;
    }
    return n;
}

static int shot_cmp(const void *a, const void *b) {
    return ((const Shot *)a)->start - ((const Shot *)b)->start;
}

/* ── Muxer ── */

typedef struct {
    AVFormatContext *fmt;
    AVStream        *st;
    int64_t          pts;
    int              active;
} Muxer;

static Muxer mux_open(const char *path, AVCodecContext *enc) {
    Muxer m = {0};
    CHECK(avformat_alloc_output_context2(&m.fmt, NULL, "mp4", path), "alloc");
    m.st = avformat_new_stream(m.fmt, NULL);
    if (!m.st) FATAL("new stream");
    CHECK(avcodec_parameters_from_context(m.st->codecpar, enc), "params");
    m.st->time_base = enc->time_base;
    CHECK(avio_open(&m.fmt->pb, path, AVIO_FLAG_WRITE), "open");
    CHECK(avformat_write_header(m.fmt, NULL), "header");
    m.active = 1;
    return m;
}

static void mux_write(Muxer *m, AVCodecContext *enc, AVPacket *pkt) {
    if (!m->active) { av_packet_unref(pkt); return; }
    pkt->pts = m->pts; pkt->dts = m->pts; pkt->duration = 1;
    av_packet_rescale_ts(pkt, enc->time_base, m->st->time_base);
    pkt->stream_index = m->st->index;
    CHECK(av_interleaved_write_frame(m->fmt, pkt), "write");
    av_packet_unref(pkt); m->pts++;
}

static void mux_close(Muxer *m) {
    if (!m->active) return;
    av_write_trailer(m->fmt);
    avio_closep(&m->fmt->pb);
    avformat_free_context(m->fmt);
    m->active = 0;
}

/* ── 帧号路由 FIFO ── */
#define ROUTE_CAP 65536
static int route_buf[ROUTE_CAP];
static int route_head = 0, route_tail = 0, route_count = 0;

static void route_push(int seg) {
    if (route_count >= ROUTE_CAP) FATAL("route queue overflow");
    route_buf[route_tail] = seg;
    route_tail = (route_tail + 1) % ROUTE_CAP;
    route_count++;
}

static int route_pop(void) {
    if (route_count == 0) FATAL("route queue underflow");
    int v = route_buf[route_head];
    route_head = (route_head + 1) % ROUTE_CAP;
    route_count--;
    return v;
}

/* ── main ── */

int main(int argc, char *argv[]) {
    if (argc != 2) { fprintf(stderr, "Usage: %s <skeleton.json>\n", argv[0]); return 1; }
    av_log_set_level(AV_LOG_ERROR);

    long len;
    char *json = read_file(argv[1], &len);
    if (!json) FATAL("Cannot read JSON");
    char video[4096] = {0};
    if (extract_str(json, "video", video, sizeof(video)) < 0) FATAL("Missing video");
    Shot shots[65536];
    int n_shots = parse_shots(json, shots, 65536);
    free(json);
    if (n_shots == 0) FATAL("No shots");
    qsort(shots, n_shots, sizeof(Shot), shot_cmp);
    printf("Video: %s\nShots: %d\n", video, n_shots);

    char out_dir[4096];
    { const char *s = strrchr(argv[1], '/');
      if (s) { int d = (int)(s - argv[1]); strncpy(out_dir, argv[1], d); out_dir[d] = '\0'; }
      else strcpy(out_dir, "."); }

    AVFormatContext *in_fmt = NULL;
    CHECK(avformat_open_input(&in_fmt, video, NULL, NULL), "open");
    CHECK(avformat_find_stream_info(in_fmt, NULL), "info");
    int vi = av_find_best_stream(in_fmt, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (vi < 0) FATAL("No video stream");

    AVStream *is = in_fmt->streams[vi];
    int w = is->codecpar->width, h = is->codecpar->height;
    enum AVCodecID cid = is->codecpar->codec_id;

    /* 优化 29.79 fps 等非标/VFR 帧率获取：优先 avg_frame_rate */
    AVRational fps = is->avg_frame_rate;
    if (fps.num <= 0 || fps.den <= 0) fps = is->r_frame_rate;
    if (fps.num <= 0 || fps.den <= 0) FATAL("fps");
    printf("  %dx%d  %d/%d fps\n", w, h, fps.num, fps.den);

    AVBufferRef *hw_dev = NULL;
    CHECK(av_hwdevice_ctx_create(&hw_dev, AV_HWDEVICE_TYPE_CUDA, "0", NULL, 0), "cuda");

    const char *dn = (cid == AV_CODEC_ID_H264) ? "h264_cuvid" : (cid == AV_CODEC_ID_HEVC) ? "hevc_cuvid" : NULL;
    if (!dn) FATAL("Unsupported codec");
    const AVCodec *dc = avcodec_find_decoder_by_name(dn);
    if (!dc) FATAL("NVDEC not found");
    AVCodecContext *dec = avcodec_alloc_context3(dc);
    CHECK(avcodec_parameters_to_context(dec, is->codecpar), "params");
    dec->hw_device_ctx = av_buffer_ref(hw_dev);
    CHECK(avcodec_open2(dec, dc, NULL), "open decoder");

    AVBufferRef *hwf = NULL;
    { hwf = av_hwframe_ctx_alloc(hw_dev); if (!hwf) FATAL("alloc hwf");
      AVHWFramesContext *fc = (AVHWFramesContext *)hwf->data;
      fc->format = AV_PIX_FMT_CUDA; fc->sw_format = AV_PIX_FMT_NV12;
      fc->width = w; fc->height = h; fc->initial_pool_size = 0;
      CHECK(av_hwframe_ctx_init(hwf), "init"); }
    dec->hw_frames_ctx = av_buffer_ref(hwf);

    const char *en = (cid == AV_CODEC_ID_H264) ? "h264_nvenc" : "hevc_nvenc";
    const AVCodec *ec = avcodec_find_encoder_by_name(en);
    if (!ec) FATAL("NVENC not found");
    AVCodecContext *enc = avcodec_alloc_context3(ec);
    enc->width = w; enc->height = h; enc->pix_fmt = AV_PIX_FMT_CUDA;
    enc->time_base = av_inv_q(fps); enc->framerate = fps;
    enc->gop_size = 30;
    enc->max_b_frames = 0;
    enc->hw_frames_ctx = av_buffer_ref(hwf);
    enc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    CHECK(av_opt_set(enc->priv_data, "preset", "p6", 0), "preset");
    CHECK(av_opt_set_int(enc->priv_data, "forced-idr", 1, 0), "forced-idr");
    CHECK(av_opt_set_int(enc->priv_data, "cq", 18, 0), "cq");
    CHECK(avcodec_open2(enc, ec, NULL), "open encoder");

    Muxer *mxs   = calloc(n_shots, sizeof(Muxer));
    int   *sent  = calloc(n_shots, sizeof(int));
    int   *recvd = calloc(n_shots, sizeof(int));
    int   *done  = calloc(n_shots, sizeof(int));
    if (!mxs || !sent || !recvd || !done) FATAL("oom");
    int close_ptr = 0;

#define TRY_CLOSE_READY() \
    while (close_ptr < n_shots && done[close_ptr] && recvd[close_ptr] >= sent[close_ptr]) { \
        if (mxs[close_ptr].active) mux_close(&mxs[close_ptr]); \
        close_ptr++; \
    }

#define DRAIN_READY() do { \
        AVPacket *_op = av_packet_alloc(); \
        while (1) { \
            int _ret = avcodec_receive_packet(enc, _op); \
            if (_ret == AVERROR(EAGAIN) || _ret == AVERROR_EOF) break; \
            CHECK(_ret, "recv pkt"); \
            int _seg = route_pop(); \
            mux_write(&mxs[_seg], enc, _op); \
            recvd[_seg]++; \
        } \
        av_packet_free(&_op); \
        TRY_CLOSE_READY(); \
    } while (0)

    AVPacket *pkt = av_packet_alloc();
    AVFrame  *fr = av_frame_alloc();
    int64_t fi = 0, decoded = 0;
    int si = 0;
    int need_idr = 1;

    char p0[4096]; snprintf(p0, sizeof(p0), "%s/segment_%04d.mp4", out_dir, shots[0].id);
    mxs[0] = mux_open(p0, enc);
    printf("  [%d] %s  (%d-%d)\n", shots[0].id, p0, shots[0].start, shots[0].end);

    while (av_read_frame(in_fmt, pkt) >= 0) {
        if (pkt->stream_index != vi) { av_packet_unref(pkt); continue; }
        CHECK(avcodec_send_packet(dec, pkt), "send pkt"); av_packet_unref(pkt);

        while (1) {
            int ret = avcodec_receive_frame(dec, fr);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
            CHECK(ret, "recv frame"); decoded++;

            if (si < n_shots && fi >= shots[si].start && fi <= shots[si].end) {
                if (!mxs[si].active) {
                    char p[4096]; snprintf(p, sizeof(p), "%s/segment_%04d.mp4", out_dir, shots[si].id);
                    mxs[si] = mux_open(p, enc);
                    printf("  [%d] %s  (%d-%d)\n", shots[si].id, p, shots[si].start, shots[si].end);
                }
                if (need_idr) { fr->pict_type = AV_PICTURE_TYPE_I; need_idr = 0; }
                else fr->pict_type = AV_PICTURE_TYPE_NONE;

                /* 【关键修复】重置送到 NVENC 的帧 pts 为当前段内从 0 开始递增的帧号 */
                fr->pts = sent[si];

                CHECK(avcodec_send_frame(enc, fr), "send enc");
                route_push(si);
                sent[si]++;
            }

            av_frame_unref(fr);
            fi++;

            if (si < n_shots && fi > shots[si].end) {
                done[si] = 1;
                si++;
                need_idr = 1;
            }

            DRAIN_READY();
        }
        if (decoded % 1000 == 0) printf("  %ld decoded\r", decoded), fflush(stdout);
    }

    /* Flush decoder */
    CHECK(avcodec_send_packet(dec, NULL), "flush dec");
    while (avcodec_receive_frame(dec, fr) >= 0) {
        decoded++;
        if (si < n_shots && fi >= shots[si].start && fi <= shots[si].end) {
            if (!mxs[si].active) {
                char p[4096]; snprintf(p, sizeof(p), "%s/segment_%04d.mp4", out_dir, shots[si].id);
                mxs[si] = mux_open(p, enc);
            }
            if (need_idr) { fr->pict_type = AV_PICTURE_TYPE_I; need_idr = 0; }
            else fr->pict_type = AV_PICTURE_TYPE_NONE;

            /* 【关键修复】Flush 阶段同样重置 pts */
            fr->pts = sent[si];

            CHECK(avcodec_send_frame(enc, fr), "send enc");
            route_push(si);
            sent[si]++;
        }
        av_frame_unref(fr); fi++;
        if (si < n_shots && fi > shots[si].end) { done[si] = 1; si++; need_idr = 1; }
        DRAIN_READY();
    }

    for (int k = si; k < n_shots; k++) done[k] = 1;

    /* Flush encoder */
    CHECK(avcodec_send_frame(enc, NULL), "flush enc");
    {
        AVPacket *op = av_packet_alloc();
        while (1) {
            int ret = avcodec_receive_packet(enc, op);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
            CHECK(ret, "recv pkt");
            int seg = route_pop();
            mux_write(&mxs[seg], enc, op);
            recvd[seg]++;
        }
        av_packet_free(&op);
    }
    TRY_CLOSE_READY();
    if (close_ptr != n_shots) FATAL("内部一致性检查失败：仍有 segment 未正确收尾");

    printf("\nDone: %ld decoded, %d segments\n", decoded, n_shots);

    free(mxs); free(sent); free(recvd); free(done);
    av_frame_free(&fr); av_packet_free(&pkt);
    avcodec_free_context(&enc); avcodec_free_context(&dec);
    avformat_close_input(&in_fmt);
    av_buffer_unref(&hwf); av_buffer_unref(&hw_dev);
    return 0;
}
