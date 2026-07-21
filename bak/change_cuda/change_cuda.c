/*
 * change_cuda.c — FFmpeg libavfilter CUDA change detection filter (C only)
 *
 * FFmpeg 6.1 API. CUDA kernels are compiled in change_cuda_kernel.cu.
 *
 * Usage: ffmpeg -hwaccel cuda -i input.mp4 -vf "change_cuda" -f null -
 *
 * Input:  AV_PIX_FMT_CUDA (NV12 CUDA surface)
 * Output: AV_PIX_FMT_CUDA (passthrough)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <libavutil/avassert.h>
#include <libavutil/avstring.h>
#include <libavutil/avutil.h>
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/opt.h>

#include <libavfilter/avfilter.h>
#include <libavfilter/buffersrc.h>
#include <libavfilter/buffersink.h>

#include <libavcodec/avcodec.h>

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

    /* CUDA frame info */
    uint8_t *d_frame_data;
    int frame_stride;

    /* Frame counter */
    int64_t frame_count;

    /* History for downstream event detection */
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

    /* Default small size */
    s->small_w = 64;
    s->small_h = 36;

    /* Allocate history */
    s->history_capacity = 200000;
    s->scores = av_malloc_array(s->history_capacity, sizeof(float));
    s->timestamps = av_malloc_array(s->history_capacity, sizeof(double));
    if (!s->scores || !s->timestamps) {
        return AVERROR(ENOMEM);
    }

    /* Initialize CUDA kernel side */
    change_cuda_init(0);

    s->frame_count = 0;
    s->n_history = 0;

    fprintf(stderr, "[change_cuda] Initialized: small=%dx%d\n",
            s->small_w, s->small_h);

    return 0;
}

/* ──────────────────────────────────────────────
 * Filter frame processing
 * ────────────────────────────────────────────── */

static int filter_frame(AVFilterLink *inlink, AVFrame *frame)
{
    AVFilterContext *ctx = inlink->dst;
    ChangeCudaContext *s = ctx->priv;
    float score = 0.0f;

    /* Get CUDA Y plane pointer and stride */
    /* For AV_PIX_FMT_CUDA frames, data[0] is the CUDA device pointer */
    s->d_frame_data = (uint8_t *)(intptr_t)frame->data[0];
    s->frame_stride = frame->linesize[0];

    if (!s->d_frame_data) {
        fprintf(stderr, "[change_cuda] ERROR: NULL frame data\n");
        return ff_filter_frame(ctx->outputs[0], frame);
    }

    /* Call CUDA kernel via external function */
    change_cuda_process(
        s->d_frame_data,
        s->frame_stride,
        frame->height,
        &score,
        frame->pts * av_q2d(inlink->time_base)
    );

    /* Store score in history */
    if (s->n_history < s->history_capacity) {
        s->scores[s->n_history] = score;
        s->timestamps[s->n_history] = frame->pts * av_q2d(inlink->time_base);
        s->n_history++;
    }

    /* Print progress every 500 frames */
    if (s->frame_count % 500 == 0) {
        fprintf(stderr, "[change_cuda] frame=%ld score=%.4f\n",
                (long)s->frame_count, score);
    }

    s->frame_count++;

    /* Pass frame through */
    return ff_filter_frame(ctx->outputs[0], frame);
}

/* ──────────────────────────────────────────────
 * Filter configuration
 * ────────────────────────────────────────────── */

static av_cold int filter_config(AVFilterLink *outlink)
{
    return 0;
}

/* ──────────────────────────────────────────────
 * Query formats
 * ────────────────────────────────────────────── */

static int query_formats(AVFilterContext *ctx)
{
    static const enum AVPixelFormat pix_fmts[] = {
        AV_PIX_FMT_CUDA,
        AV_PIX_FMT_NONE
    };

    AVFilterFormats *formats = ff_make_format_list(pix_fmts);
    if (!formats)
        return AVERROR(ENOMEM);

    return ff_set_common_formats(ctx, formats);
}

/* ──────────────────────────────────────────────
 * Filter cleanup
 * ────────────────────────────────────────────── */

static av_cold void filter_uninit(AVFilterContext *ctx)
{
    ChangeCudaContext *s = ctx->priv;

    change_cuda_cleanup();

    av_freep(&s->scores);
    av_freep(&s->timestamps);

    if (s->n_history > 0) {
        fprintf(stderr, "[change_cuda] Processed %d frames\n", s->n_history);
        fprintf(stderr, "[change_cuda] Score range: [%.6f, %.6f]\n",
                s->scores[0], s->scores[s->n_history - 1]);
    }
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
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
    },
    { NULL }
};

static const AVFilterPad change_cuda_outputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
    },
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
    .filter_frame  = filter_frame,
    .flags         = AVFILTER_FLAG_SUPPORT_TIMELINE_INTERNAL,
};
