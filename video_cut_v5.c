/*
 * video_cut_v5.c — GPU Segment Renderer (frame-accurate)
 *
 * NVDEC 顺序解码 → frame_id 计数器 → shot.start 开 encoder → shot.end 关。
 * 零 seek、零 -ss、每段独立 NVENC。帧号驱动，全 GPU 流水线。
 *
 * Compile:
 *   gcc -o video_cut video_cut_v5.c \
 *     -I/home/dahe/ffmpeg_dev/usr/include/x86_64-linux-gnu \
 *     -L/usr/lib/x86_64-linux-gnu \
 *     -l:libavformat.so.60 -l:libavcodec.so.60 -l:libavutil.so.58 -O2
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/opt.h>

#define CHECK(e, m) do { if ((e) < 0) { char _b[256]; av_strerror(e,_b,256); \
    fprintf(stderr,"FATAL %s: %s\n",m,_b); exit(1); } } while(0)

typedef struct { int id, start, end; } Shot;

/* ── JSON parser ── */
static char *rf(const char *path, long *len) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END); *len = ftell(fp); rewind(fp);
    char *buf = malloc(*len + 1);
    if (!buf) { fclose(fp); return NULL; }
    fread(buf, 1, *len, fp); fclose(fp); buf[*len] = 0; return buf;
}
static int exs(const char *j, const char *k, char *out, int sz) {
    char s[64]; snprintf(s, sizeof(s), "\"%s\"", k);
    const char *p = strstr(j, s); if (!p) return -1;
    p = strchr(p, ':'); if (!p) return -1;
    p++; while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    if (*p != '"') return -1; p++;
    int i = 0;
    while (*p && *p != '"' && i < sz - 1) {
        if (*p == '\\' && *(p+1)) p++; out[i++] = *p++;
    }
    out[i] = 0; return 0;
}
static int ps(const char *j, Shot *s, int max) {
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
        strncpy(obj, p, olen); obj[olen] = 0;
        int id = -1, st = -1, ed = -1; char *k;
        k = strstr(obj, "\"id\"");    if (k) { k = strchr(k,':'); if (k) id = (int)strtol(k+1,NULL,10); }
        k = strstr(obj, "\"start\""); if (k) { k = strchr(k,':'); if (k) st = (int)strtol(k+1,NULL,10); }
        k = strstr(obj, "\"end\"");   if (k) { k = strchr(k,':'); if (k) ed = (int)strtol(k+1,NULL,10); }
        if (id >= 0 && st >= 0 && ed >= 0 && ed >= st) s[n++] = (Shot){id, st, ed};
        p = e + 1;
    }
    return n;
}
static int scmp(const void *a, const void *b) {
    return ((const Shot *)a)->start - ((const Shot *)b)->start;
}

int main(int argc, char *argv[]) {
    if (argc != 2) { fprintf(stderr, "Usage: %s <skeleton.json>\n", argv[0]); return 1; }
    av_log_set_level(AV_LOG_ERROR);

    /* ── parse JSON ── */
    long jl; char *js = rf(argv[1], &jl);
    if (!js) { fprintf(stderr, "Cannot read JSON\n"); return 1; }
    char vp[4096] = {0}; exs(js, "video", vp, sizeof(vp));
    Shot *shots = malloc(65536 * sizeof(Shot));
    int n = ps(js, shots, 65536); free(js);
    if (n == 0) { fprintf(stderr, "No shots\n"); return 1; }
    qsort(shots, n, sizeof(Shot), scmp);

    char od[4096];
    { const char *s = strrchr(argv[1], '/');
      if (s) { int d = (int)(s - argv[1]); strncpy(od, argv[1], d); od[d]=0; }
      else strcpy(od, "."); }

    /* ── GPU init (order matters: CUDA before input open) ── */
    AVBufferRef *hd = NULL;
    CHECK(av_hwdevice_ctx_create(&hd, AV_HWDEVICE_TYPE_CUDA, "0", NULL, 0), "CUDA");

    AVBufferRef *hf = av_hwframe_ctx_alloc(hd);
    if (!hf) { fprintf(stderr, "hw_alloc\n"); return 1; }
    AVHWFramesContext *fc = (AVHWFramesContext *)hf->data;
    fc->format    = AV_PIX_FMT_CUDA;
    fc->sw_format = AV_PIX_FMT_NV12;

    /* open input */
    AVFormatContext *ifmt = NULL;
    CHECK(avformat_open_input(&ifmt, vp, NULL, NULL), "open");
    CHECK(avformat_find_stream_info(ifmt, NULL), "info");
    int vi = av_find_best_stream(ifmt, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (vi < 0) { fprintf(stderr, "No video\n"); return 1; }
    AVCodecParameters *par = ifmt->streams[vi]->codecpar;

    /* complete hw_frames */
    fc->width  = par->width;
    fc->height = par->height;
    fc->initial_pool_size = 4;
    CHECK(av_hwframe_ctx_init(hf), "hwframe_init");

    /* NVDEC */
    const char *dn = (par->codec_id == AV_CODEC_ID_H264) ? "h264_cuvid" : "hevc_cuvid";
    const AVCodec *dc = avcodec_find_decoder_by_name(dn);
    if (!dc) { fprintf(stderr, "NVDEC not found\n"); return 1; }
    AVCodecContext *dec = avcodec_alloc_context3(dc);
    CHECK(avcodec_parameters_to_context(dec, par), "params");
    dec->hw_device_ctx = av_buffer_ref(hd);
    dec->hw_frames_ctx = av_buffer_ref(hf);
    CHECK(avcodec_open2(dec, dc, NULL), "open dec");

    printf("Video: %s  %dx%d  Shots: %d  Output: %s/\n\n",
           vp, par->width, par->height, n, od);

    /* ── main: sequential decode → per-shot encode ── */
    AVPacket *ipkt   = av_packet_alloc();
    AVFrame  *ifrm   = av_frame_alloc();
    int64_t fi = 0;            /* global frame index */
    int si = 0, ok = 0, fail = 0, eof = 0;
    while (si < n) {
        Shot *sh = &shots[si];
        int nfr = sh->end - sh->start + 1;

        /* skip to shot start */
        while (fi < sh->start && !eof) {
            int r = avcodec_receive_frame(dec, ifrm);
            if (r == 0) { av_frame_unref(ifrm); fi++; continue; }
            if (r == AVERROR(EAGAIN)) {
                r = av_read_frame(ifmt, ipkt);
                if (r < 0) { av_packet_unref(ipkt); avcodec_send_packet(dec, NULL); eof=1; continue; }
                if (ipkt->stream_index != vi) { av_packet_unref(ipkt); continue; }
                avcodec_send_packet(dec, ipkt); av_packet_unref(ipkt); continue;
            }
            if (r == AVERROR_EOF) eof = 1; break;
        }
        if (eof && fi < sh->start) break;

        /* ── per-shot encoder + muxer ── */
        char op[4096]; snprintf(op, sizeof(op), "%s/segment_%04d.mp4", od, si);

        const AVCodec *ec = avcodec_find_encoder_by_name("h264_nvenc");
        AVCodecContext *enc = avcodec_alloc_context3(ec);
        enc->width = dec->width; enc->height = dec->height;
        enc->pix_fmt = dec->pix_fmt;
        enc->time_base = (AVRational){1001, 30000};
        enc->framerate = (AVRational){30000, 1001};
        enc->gop_size = 30; enc->max_b_frames = 0;
        enc->profile = FF_PROFILE_H264_BASELINE;
        enc->hw_frames_ctx = av_buffer_ref(hf);
        enc->hw_device_ctx  = av_buffer_ref(hd);
        enc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        av_opt_set(enc->priv_data, "preset", "p1", 0);
        av_opt_set_int(enc->priv_data, "cq", 26, 0);
        CHECK(avcodec_open2(enc, ec, NULL), "open enc");

        AVFormatContext *mux = NULL; AVStream *mst = NULL;
        CHECK(avformat_alloc_output_context2(&mux, NULL, "mp4", op), "mux alloc");
        mst = avformat_new_stream(mux, NULL);
        CHECK(avcodec_parameters_from_context(mst->codecpar, enc), "mux params");
        mst->time_base = enc->time_base;
        CHECK(avio_open(&mux->pb, op, AVIO_FLAG_WRITE), "mux open");
        CHECK(avformat_write_header(mux, NULL), "mux hdr");

        /* encode frames in range */
        int prod = 0;
        while (fi <= sh->end && !eof) {
            int r = avcodec_receive_frame(dec, ifrm);
            if (r == 0) {
                if (fi >= sh->start) {
                    ifrm->pts = fi;
                    CHECK(avcodec_send_frame(enc, ifrm), "send");
                    { AVPacket *opkt = av_packet_alloc();
                    while (1) {
                        r = avcodec_receive_packet(enc, opkt);
                        if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) break;
                        CHECK(r, "recv pkt");
                        opkt->stream_index = mst->index;
                        av_packet_rescale_ts(opkt, enc->time_base, mst->time_base);
                        av_interleaved_write_frame(mux, opkt);
                        prod++;
                        av_packet_unref(opkt);
                    }
                    av_packet_free(&opkt); }
                }
                av_frame_unref(ifrm); fi++; continue;
            }
            if (r == AVERROR(EAGAIN)) {
                r = av_read_frame(ifmt, ipkt);
                if (r < 0) { av_packet_unref(ipkt); avcodec_send_packet(dec, NULL); continue; }
                if (ipkt->stream_index != vi) { av_packet_unref(ipkt); continue; }
                avcodec_send_packet(dec, ipkt); av_packet_unref(ipkt); continue;
            }
            if (r == AVERROR_EOF) { break; }
        }

        /* drain decoder tail at EOF */
        if (eof) {
            avcodec_send_packet(dec, NULL);
            while (1) {
                int r = avcodec_receive_frame(dec, ifrm);
                if (r == AVERROR_EOF || r == AVERROR(EAGAIN)) break;
                if (r < 0) break;
                if (fi >= sh->start && fi <= sh->end) {
                    ifrm->pts = fi;
                    CHECK(avcodec_send_frame(enc, ifrm), "send");
                    { AVPacket *opkt = av_packet_alloc();
                    while (1) {
                        r = avcodec_receive_packet(enc, opkt);
                        if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) break;
                        CHECK(r, "recv pkt");
                        opkt->stream_index = mst->index;
                        av_packet_rescale_ts(opkt, enc->time_base, mst->time_base);
                        av_interleaved_write_frame(mux, opkt);
                        prod++;
                        av_packet_unref(opkt);
                    }
                    av_packet_free(&opkt); }
                }
                av_frame_unref(ifrm); fi++;
            }
        }

        /* flush encoder */
        avcodec_send_frame(enc, NULL);
        { AVPacket *opkt = av_packet_alloc();
          while (1) {
              int r = avcodec_receive_packet(enc, opkt);
              if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) break;
              if (r < 0) break;
              opkt->stream_index = mst->index;
              av_packet_rescale_ts(opkt, enc->time_base, mst->time_base);
              av_interleaved_write_frame(mux, opkt);
              prod++;
              av_packet_unref(opkt);
          }
          av_packet_free(&opkt); }

        av_write_trailer(mux);
        avio_closep(&mux->pb);
        avformat_free_context(mux);
        avcodec_free_context(&enc);

        if (prod == nfr) {
            printf("  [%3d] segment_%04d.mp4  %4dfr  OK\n", si, si, prod); ok++;
        } else {
            fprintf(stderr, "  [%3d] MISMATCH got=%d exp=%d\n", si, prod, nfr); fail++;
        }
        fflush(stdout);
        si++;
    }

    av_frame_free(&ifrm); av_packet_free(&ipkt);
    avcodec_free_context(&dec); avformat_close_input(&ifmt);
    av_buffer_unref(&hf); av_buffer_unref(&hd);
    free(shots);

    printf("\nDone: %d OK  %d FAIL  (total %d)\n", ok, fail, n);
    return fail > 0 ? 1 : 0;
}
