/*
 * change_cuda_filter.c — FFmpeg libavfilter change_cuda plugin (FFmpeg 6.1)
 *
 * This is a minimal filter that demonstrates the CUDA change detection
 * integrated into FFmpeg's filter graph. Uses the activate callback pattern.
 *
 * Build: gcc -shared -fPIC -o libchange_cuda_filter.so change_cuda_filter.c \
 *       $(pkg-config --libs --cflags libavfilter libavcodec libavutil libswscale) \
 *       -lcuda -lcudart
 *
 * Usage: ffmpeg -hwaccel cuda -i input.mp4 -vf "change_cuda" -f null -
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <libavutil/avassert.h>
#include <libavutil/avutil.h>
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/opt.h>

#include <libavfilter/avfilter.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "change_cuda.h"

/* ──────────────────────────────────────────────
 * Filter context
 * ────────────────────────────────────────────── */

typedef struct ChangeCudaContext {
    AVClass *class;

    /* Small Y plane dimensions */
    int small_w;
    int small_h;

    /* CUDA buffers */
    uint8_t *d_resized;
    uint8_t *d_prev_resized;
    float *d_score;
    int frame_count;

    /* Score history */
    float *scores;
    double *timestamps;
    int n_history;
    int history_capacity;

} ChangeCudaContext;

/* ──────────────────────────────────────────────
 * Filter initialization
 * ────────────────────────────────────────────── */

static av_cold int filter_init(AVFilterContext *ctx)
{
    ChangeCudaContext *s = ctx->priv;

    s->small_w = 64;
    s->small_h = 36;

    s->history_capacity = 200000;
    s->scores = av_malloc_array(s->history_capacity, sizeof(float));
    s->timestamps = av_malloc_array(s->history_capacity, sizeof(double));
    if (!s->scores || !s->timestamps) return AVERROR(ENOMEM);

    int small_size = s->small_w * s->small_h;
    cudaError_t err;
    err = cudaMalloc(&s->d_resized, small_size);
    if (err != cudaSuccess) return AVERROR(ENOMEM);
    err = cudaMalloc(&s->d_prev_resized, small_size);
    if (err != cudaSuccess) { cudaFree(s->d_resized); return AVERROR(ENOMEM); }
    err = cudaMalloc(&s->d_score, sizeof(float));
    if (err != cudaSuccess) { cudaFree(s->d_resized); cudaFree(s->d_prev_resized); return AVERROR(ENOMEM); }

    s->frame_count = 0;
    s->n_history = 0;

    change_cuda_init(0);

    fprintf(stderr, "[change_cuda] Initialized: small=%dx%d\n", s->small_w, s->small_h);
    return 0;
}

/* ──────────────────────────────────────────────
 * Activate callback (FFmpeg 6.1 pattern)
 * ────────────────────────────────────────────── */

static int activate(AVFilterContext *ctx)
{
    ChangeCudaContext *s = ctx->priv;
    AVFilterLink *inlink = ctx->inputs[0];
    AVFilterLink *outlink = ctx->outputs[0];
    int ret;

    /* Try to get a frame from input */
    AVFrame *frame = NULL;
    ret = ff_inlink_consume_frame(inlink, &frame);
    if (ret < 0) return ret;
    if (!frame) {
        /* No frame available - check if we need to request one */
        if (ff_inlink_queued_frames(inlink) > 0) {
            ff_filter_set_ready(ctx, 100);
        } else {
            ff_outlink_request_frame(outlink);
        }
        return 0;
    }

    /* Process CUDA frame */
    if (frame->format == AV_PIX_FMT_CUDA && frame->data[0]) {
        uint8_t *d_frame = (uint8_t *)(intptr_t)frame->data[0];
        int stride = frame->linesize[0];
        double pts = frame->pts * av_q2d(inlink->time_base);

        /* Resize current frame's Y plane */
        launch_resize_kernel(d_frame, stride, frame->height, s->d_resized, s->small_w, s->small_h);

        float score = 0.0f;
        if (s->frame_count > 0) {
            /* MAFD */
            cudaMemset(s->d_score, 0, sizeof(float));
            int total = s->small_w * s->small_h;
            int bs = 256, gs = (total + bs - 1) / bs;
            mafd_kernel<<<gs, bs>>>(s->d_resized, s->d_prev_resized, s->d_score, s->small_w, s->small_h);
            cudaDeviceSynchronize();
            cudaMemcpy(&score, s->d_score, sizeof(float), cudaMemcpyDeviceToHost);
            score /= (float)(s->small_w * s->small_h);
            cudaMemcpy(s->d_prev_resized, s->d_resized, s->small_w * s->small_h, cudaMemcpyDeviceToDevice);
        } else {
            cudaMemcpy(s->d_prev_resized, s->d_resized, s->small_w * s->small_h, cudaMemcpyDeviceToDevice);
        }

        /* Store score */
        if (s->n_history < s->history_capacity) {
            s->scores[s->n_history] = score;
            s->timestamps[s->n_history] = pts;
            s->n_history++;
        }

        /* Attach score as frame side data */
        char score_str[64];
        snprintf(score_str, sizeof(score_str), "%f", score);

        if (s->frame_count % 500 == 0) {
            fprintf(stderr, "[change_cuda] frame=%d score=%.4f\n", s->frame_count, score);
        }
        s->frame_count++;
    }

    /* Pass frame through */
    ret = ff_outlink_submit_frame(outlink, frame);
    if (ret < 0) return ret;

    /* Schedule more if input has more frames */
    if (ff_inlink_queued_frames(inlink) > 0) {
        ff_filter_set_ready(ctx, 100);
    }

    return 0;
}

/* ──────────────────────────────────────────────
 * Query formats
 * ────────────────────────────────────────────── */

static int query_formats(AVFilterContext *ctx)
{
    static const enum AVPixelFormat pix_fmts[] = { AV_PIX_FMT_CUDA, AV_PIX_FMT_NONE };
    AVFilterFormats *formats = ff_make_format_list(pix_fmts);
    return ff_set_common_formats(ctx, formats);
}

/* ──────────────────────────────────────────────
 * Filter cleanup
 * ────────────────────────────────────────────── */

static av_cold void filter_uninit(AVFilterContext *ctx)
{
    ChangeCudaContext *s = ctx->priv;

    if (s->d_resized) cudaFree(s->d_resized);
    if (s->d_prev_resized) cudaFree(s->d_prev_resized);
    if (s->d_score) cudaFree(s->d_score);

    change_cuda_cleanup();

    av_freep(&s->scores);
    av_freep(&s->timestamps);

    fprintf(stderr, "[change_cuda] Processed %d frames\n", s->n_history);
}

/* ──────────────────────────────────────────────
 * Configuration
 * ────────────────────────────────────────────── */

static av_cold int filter_config(AVFilterLink *outlink)
{
    return 0;
}

/* ──────────────────────────────────────────────
 * FFmpeg filter registration
 * ────────────────────────────────────────────── */

#define OFFSET(x) offsetof(ChangeCudaContext, x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_FILTERING_PARAM

static const AVOption change_cuda_options[] = {
    { "small_w", "Small Y plane width",  OFFSET(small_w), AV_OPT_TYPE_INT, {.i64 = 64}, 16, 256, FLAGS },
    { "small_h", "Small Y plane height", OFFSET(small_h), AV_OPT_TYPE_INT, {.i64 = 36}, 16, 256, FLAGS },
    { NULL }
};

static const AVClass change_cuda_class = {
    .class_name = "change_cuda",
    .item_name  = av_default_item_name,
    .option     = change_cuda_options,
    .version    = LIBAVUTIL_VERSION_INT,
};

static const AVFilterPad change_cuda_inputs[] = {
    { .name = "default", .type = AVMEDIA_TYPE_VIDEO },
    { NULL }
};

static const AVFilterPad change_cuda_outputs[] = {
    { .name = "default", .type = AVMEDIA_TYPE_VIDEO },
    { NULL }
};

AVFilter ff_change_cuda = {
    .name          = "change_cuda",
    .description   = "CUDA-accelerated change detection filter",
    .priv_size     = sizeof(ChangeCudaContext),
    .priv_class    = &change_cuda_class,
    .init          = filter_init,
    .uninit        = filter_uninit,
    .query_formats = query_formats,
    .inputs        = change_cuda_inputs,
    .outputs       = change_cuda_outputs,
    .activate      = activate,
    .flags         = AVFILTER_FLAG_SUPPORT_TIMELINE_INTERNAL | AVFILTER_FLAG_HWDEVICE,
};
