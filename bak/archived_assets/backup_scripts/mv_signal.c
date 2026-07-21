/*
 * mv_signal.c — 旁路信号接收器 (sideband signal receiver)
 *
 * 利用 FFmpeg libavformat/libavcodec 解码 H.264，
 * 从 AVFrame side data 提取 Motion Vector 统计，输出到二进制文件。
 *
 * 输出格式 (每帧 20 字节, big-endian):
 *   offset 0   : uint32  frame_number    帧序号 (从 0 递增)
 *   offset 4   : uint16  mv_count        motion vector 总数
 *   offset 6   : uint16  mv_zero_count   magnitude < 1.0 的 MV 数
 *   offset 8   : float   mv_mean_mag     所有 MV 的平均幅度
 *   offset 12  : float   mv_max_mag      最大 MV 幅度
 *   offset 16  : uint32  reserved        对齐 / 扩展
 *
 * 使用:
 *   gcc -O3 mv_signal.c -o mv_signal \
 *       $(pkg-config --cflags --libs libavformat libavcodec libavutil)
 *   ./mv_signal input.mp4 output.signal
 *
 * 依赖:
 *   FFmpeg >= 5.1  (libavformat, libavcodec, libavutil)
 *   H.264 解码器在解码时自动附上 AV_FRAME_DATA_MOTION_VECTORS side data
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/frame.h>
#include <libavutil/motion_vector.h>

/* ─────────────────────────── 输出记录 ─────────────────────────── */

typedef struct {
    uint32_t frame_number;
    uint16_t mv_count;
    uint16_t mv_zero_count;
    float    mv_mean_mag;
    float    mv_max_mag;
    uint32_t reserved;
} __attribute__((packed)) FrameSignal;

/* ─────────────────────────── 回调 ─────────────────────────── */

typedef struct {
    FILE   *fp;           /* 输出文件 */
    int     vstream_idx;  /* 目标视频流索引 */
    int64_t last_pts;     /* 上一帧 PTS (去重) */
    int     frame_count;  /* 已处理帧计数 */
    int     has_error;    /* 是否已出错 */
} CallbackCtx;

static float vec_mag(const AVMotionVector *mv)
{
    float mx = (mv->motion_scale > 0)
        ? (float)mv->motion_x / (float)mv->motion_scale
        : (float)mv->motion_x;
    float my = (mv->motion_scale > 0)
        ? (float)mv->motion_y / (float)mv->motion_scale
        : (float)mv->motion_y;
    return sqrtf(mx * mx + my * my);
}

/**
 * frame_callback — 每 decoded frame 调用一次
 * 提取 MV side data → 计算统计量 → 写入 binary signal
 */
static void frame_callback(AVFrame *frame, CallbackCtx *ctx)
{
    if (ctx->has_error) return;

    /* 用 PTS 去重 (某些解码器会重复输出相同帧) */
    if (frame->pts == ctx->last_pts) return;
    if (frame->pts == AV_NOPTS_VALUE) return;

    /* 从 AVFrame side data 取出 motion vectors */
    AVFrameSideData *sd = av_frame_get_side_data(
        frame, AV_FRAME_DATA_MOTION_VECTORS);
    if (!sd || !sd->data || sd->size <= 0) return;

    AVMotionVector *mvs = (AVMotionVector *)sd->data;
    int n = sd->size / (int)sizeof(AVMotionVector);
    if (n <= 0 || n > 65535) return;

    /* 统计 */
    double sum = 0.0;
    float  mx  = 0.0f;
    int    zc  = 0;

    for (int i = 0; i < n; i++) {
        float m = vec_mag(&mvs[i]);
        sum += m;
        if (m > mx) mx = m;
        if (m < 1.0f) zc++;
    }

    FrameSignal sig;
    memset(&sig, 0, sizeof(sig));
    sig.frame_number = (uint32_t)ctx->frame_count;
    sig.mv_count     = (uint16_t)n;
    sig.mv_zero_count= (uint16_t)zc;
    sig.mv_mean_mag  = (float)(sum / (double)n);
    sig.mv_max_mag   = mx;
    sig.reserved     = 0;

    if (fwrite(&sig, sizeof(FrameSignal), 1, ctx->fp) != 1) {
        fprintf(stderr, "  [warn] write frame %d failed\n", ctx->frame_count);
        ctx->has_error = 1;
        return;
    }

    ctx->frame_count++;
    ctx->last_pts = frame->pts;
}

/* ─────────────────────────── 解码主循环 ─────────────────────────── */

static int decode_video(const char *src_path, CallbackCtx *ctx)
{
    AVFormatContext *fmt  = NULL;
    AVCodecContext  *dec  = NULL;
    const AVCodec   *codec = NULL;
    AVPacket        *pkt  = NULL;
    AVFrame         *frame = NULL;
    int ret;

    /* 1. 打开封装器 */
    ret = avformat_open_input(&fmt, src_path, NULL, NULL);
    if (ret < 0) {
        fprintf(stderr, "[mv_signal] Cannot open %s (ret=%d)\n", src_path, ret);
        return ret;
    }

    ret = avformat_find_stream_info(fmt, NULL);
    if (ret < 0) {
        fprintf(stderr, "[mv_signal] Cannot find stream info\n");
        avformat_close_input(&fmt);
        return ret;
    }

    /* 2. 定位第一个视频流 */
    ctx->vstream_idx = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO,
                                           -1, -1, &codec, 0);
    if (ctx->vstream_idx < 0) {
        fprintf(stderr, "[mv_signal] No video stream\n");
        avformat_close_input(&fmt);
        return ctx->vstream_idx;
    }

    /* 3. 打开解码器 */
    AVStream *st = fmt->streams[ctx->vstream_idx];
    dec = avcodec_alloc_context3(codec);
    if (!dec) {
        fprintf(stderr, "[mv_signal] alloc_context failed\n");
        avformat_close_input(&fmt);
        return AVERROR(ENOMEM);
    }
    avcodec_parameters_to_context(dec, st->codecpar);

    ret = avcodec_open2(dec, codec, NULL);
    if (ret < 0) {
        fprintf(stderr, "[mv_signal] Cannot open decoder (ret=%d)\n", ret);
        avcodec_free_context(&dec);
        avformat_close_input(&fmt);
        return ret;
    }

    /* 开启 motion vector 导出 (H.264 解码器需要此 flag) */
    dec->export_side_data |= AV_CODEC_EXPORT_DATA_MVS;

    frame = av_frame_alloc();
    pkt   = av_packet_alloc();
    if (!frame || !pkt) {
        fprintf(stderr, "[mv_signal] alloc frame/pkt failed\n");
        av_packet_free(&pkt);
        av_frame_free(&frame);
        avcodec_free_context(&dec);
        avformat_close_input(&fmt);
        return AVERROR(ENOMEM);
    }

    ctx->last_pts    = AV_NOPTS_VALUE;
    ctx->frame_count = 0;
    ctx->has_error   = 0;

    fprintf(stderr, "[mv_signal] decoding %s ...\n", src_path);

    /* 4. 主解码循环 */
    while (av_read_frame(fmt, pkt) >= 0) {
        if (pkt->stream_index != ctx->vstream_idx) {
            av_packet_unref(pkt);
            continue;
        }
        ret = avcodec_send_packet(dec, pkt);
        av_packet_unref(pkt);
        if (ret < 0) continue;

        while ((ret = avcodec_receive_frame(dec, frame)) == 0) {
            frame_callback(frame, ctx);
            av_frame_unref(frame);
            if (ctx->has_error) break;
        }
        if (ctx->has_error) break;
    }

    /* 5. 刷新解码器 (draining) */
    if (!ctx->has_error) {
        avcodec_send_packet(dec, NULL);
        while ((ret = avcodec_receive_frame(dec, frame)) == 0) {
            frame_callback(frame, ctx);
            av_frame_unref(frame);
            if (ctx->has_error) break;
        }
    }

    fprintf(stderr, "[mv_signal] decoded %d frames\n", ctx->frame_count);

    av_packet_free(&pkt);
    av_frame_free(&frame);
    avcodec_free_context(&dec);
    avformat_close_input(&fmt);

    return ctx->has_error ? -1 : 0;
}

/* ─────────────────────────── main ─────────────────────────── */

int main(int argc, char *argv[])
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s input.mp4 output.signal\n", argv[0]);
        return 1;
    }

    av_log_set_level(AV_LOG_ERROR);

    CallbackCtx ctx;
    memset(&ctx, 0, sizeof(ctx));

    ctx.fp = fopen(argv[2], "wb");
    if (!ctx.fp) {
        fprintf(stderr, "[mv_signal] Cannot open %s for writing\n", argv[2]);
        return 1;
    }

    int ret = decode_video(argv[1], &ctx);

    long file_size = ftell(ctx.fp);      /* 在 fclose 之前取大小 */
    fclose(ctx.fp);

    if (ret == 0 && file_size > 0) {
        fprintf(stderr, "[mv_signal] signal written: %s  (%ld bytes, %ld frames)\n",
                argv[2], file_size, file_size / (long)sizeof(FrameSignal));
    }

    return ret;
}
