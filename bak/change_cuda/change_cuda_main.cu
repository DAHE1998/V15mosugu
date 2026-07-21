/*
 * change_cuda_main.cu — Standalone CUDA change detection (Phase 1+2+3)
 *
 * Phase 1: CUDA NVDEC decode + MAFD change score
 * Phase 2: Peak detection + event windows (NMS, top-N)
 * Phase 3: GPU ring buffer + representative frame extraction (middle + medoid)
 *
 * Output:
 *   events.json  — event list with representative frames
 *   change_curve.csv — per-frame scores
 */

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

/* ── CUDA Kernels ── */

__global__ void resize_y_kernel(
    const uint8_t *__restrict__ src_y, uint8_t *__restrict__ dst_y,
    int src_w, int src_h, int dst_w, int dst_h)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= dst_h || x >= dst_w) return;
    float sy = (float)y * src_h / dst_h;
    float sx = (float)x * dst_w / dst_w;
    int y0 = (int)sy, x0 = (int)sx;
    int y1 = y0 + 1; if (y1 >= src_h) y1 = src_h - 1;
    int x1 = x0 + 1; if (x1 >= src_w) x1 = src_w - 1;
    float fy = sy - y0, fx = sx - x0;
    uint8_t v00 = src_y[y0 * src_w + x0];
    uint8_t v10 = src_y[y0 * src_w + x1];
    uint8_t v01 = src_y[y1 * src_w + x0];
    uint8_t v11 = src_y[y1 * src_w + x1];
    float v = (1-fy)*((1-fx)*v00 + fx*v10) + fy*((1-fx)*v01 + fx*v11);
    dst_y[y * dst_w + x] = (uint8_t)(v + 0.5f);
}

__global__ void mafd_kernel(
    const uint8_t *__restrict__ curr_y, const uint8_t *__restrict__ prev_y,
    float *__restrict__ score, int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    if (idx >= total) return;
    float diff = abs((float)curr_y[idx] - (float)prev_y[idx]);
    atomicAdd(score, diff);
}

/* ── GPU Ring Buffer ── */

class GPURingBuffer {
    uint8_t *d_buffer;        /* GPU: ring buffer of half-res frames */
    int max_frames, h_half, w_half, frame_bytes;
    int count, write_pos, total_written;
    int64_t *d_frame_nums;    /* GPU: frame numbers */
    float *d_timestamps;      /* GPU: timestamps */
    int64_t *h_frame_nums;    /* CPU: mirror for search */
    float *h_timestamps;

public:
    int64_t get_frame_num(int i) const { return (i < count) ? h_frame_nums[i] : 0; }
    float get_timestamp(int i) const { return (i < count) ? h_timestamps[i] : 0.0f; }
    int get_count() const { return count; }

    GPURingBuffer(int max, int H, int W)
        : max_frames(max), h_half(H/2), w_half(W/2),
          count(0), write_pos(0), total_written(0)
    {
        frame_bytes = 3 * h_half * w_half;
        cudaMalloc(&d_buffer, max_frames * frame_bytes);
        cudaMalloc(&d_frame_nums, max_frames * sizeof(int64_t));
        cudaMalloc(&d_timestamps, max_frames * sizeof(float));
        h_frame_nums = (int64_t*)malloc(max_frames * sizeof(int64_t));
        h_timestamps = (float*)malloc(max_frames * sizeof(float));
        memset(h_frame_nums, 0, max_frames * sizeof(int64_t));
        memset(h_timestamps, 0, max_frames * sizeof(float));
    }

    ~GPURingBuffer() {
        cudaFree(d_buffer); cudaFree(d_frame_nums); cudaFree(d_timestamps);
        free(h_frame_nums); free(h_timestamps);
    }

    /* Push a full-res frame: resize on CPU, transfer to GPU */
    void push(uint8_t *d_y, int stride, int height, int frame_num, float timestamp) {
        /* Resize Y plane to half-res on GPU */
        uint8_t *d_half;
        cudaMalloc(&d_half, frame_bytes);
        dim3 block(16, 16);
        dim3 grid((w_half + 15) / 16, (h_half + 15) / 16);
        resize_y_kernel<<<grid, block>>>(d_y, d_half, stride, height, w_half, h_half);
        cudaDeviceSynchronize();

        if (count < max_frames) {
            cudaMemcpy(d_buffer + count * frame_bytes, d_half, frame_bytes, cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_frame_nums + count, &frame_num, sizeof(int64_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_timestamps + count, &timestamp, sizeof(float), cudaMemcpyHostToDevice);
            h_frame_nums[count] = frame_num;
            h_timestamps[count] = timestamp;
            count++;
        } else {
            cudaMemcpy(d_buffer + write_pos * frame_bytes, d_half, frame_bytes, cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_frame_nums + write_pos, &frame_num, sizeof(int64_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_timestamps + write_pos, &timestamp, sizeof(float), cudaMemcpyHostToDevice);
            h_frame_nums[write_pos] = frame_num;
            h_timestamps[write_pos] = timestamp;
            write_pos = (write_pos + 1) % max_frames;
        }
        total_written++;
        cudaFree(d_half);
    }

    /* Get frames near center_ts within window_sec */
    void get_window_frames(float center_ts, float window_sec,
                           int *out_frame_nums, int *out_count, int max_out)
    {
        *out_count = 0;
        for (int i = 0; i < count && *out_count < max_out; i++) {
            float ts = h_timestamps[i];
            if (fabsf(ts - center_ts) <= window_sec) {
                out_frame_nums[(*out_count)++] = (int)h_frame_nums[i];
            }
        }
    }

    int total_seen() const { return total_written; }
};

/* ── Change Detector ── */

class ChangeDetector {
    uint8_t *d_prev_small, *d_curr_small;
    float *d_score;
    int small_w, small_h;

public:
    float *scores;
    double *timestamps;
    int n_frames, capacity;

    ChangeDetector(int sw, int sh, int cap)
        : small_w(sw), small_h(sh), n_frames(0), capacity(cap)
    {
        scores = (float*)malloc(cap * sizeof(float));
        timestamps = (double*)malloc(cap * sizeof(double));
        cudaMalloc(&d_prev_small, sw * sh);
        cudaMalloc(&d_curr_small, sw * sh);
        cudaMalloc(&d_score, sizeof(float));
    }

    ~ChangeDetector() {
        cudaFree(d_prev_small); cudaFree(d_curr_small); cudaFree(d_score);
        free(scores); free(timestamps);
    }

    float process(uint8_t *d_y, int stride, int height, double pts) {
        dim3 block(16, 16);
        dim3 grid((small_w + 15) / 16, (small_h + 15) / 16);
        resize_y_kernel<<<grid, block>>>(d_y, d_curr_small, stride, height, small_w, small_h);

        float score = 0.0f;
        if (n_frames > 0) {
            cudaMemset(d_score, 0, sizeof(float));
            int total = small_w * small_h;
            int bs = 256, gs = (total + bs - 1) / bs;
            mafd_kernel<<<gs, bs>>>(d_curr_small, d_prev_small, d_score, small_w, small_h);
            cudaDeviceSynchronize();
            cudaMemcpy(&score, d_score, sizeof(float), cudaMemcpyDeviceToHost);
            score /= (float)(small_w * small_h);
            cudaMemcpy(d_prev_small, d_curr_small, small_w * small_h, cudaMemcpyDeviceToDevice);
        } else {
            cudaMemcpy(d_prev_small, d_curr_small, small_w * small_h, cudaMemcpyDeviceToDevice);
        }

        if (n_frames < capacity) {
            scores[n_frames] = score;
            timestamps[n_frames] = pts;
            n_frames++;
        }
        return score;
    }
};

/* ── Scene Cut Detector (复刻 scdet 算法) ── */

typedef struct {
    int cut_frame;   /* 切割帧号（-1 修正） */
    int raw_frame;   /* 原始帧号 */
    float score;     /* scdet 兼容得分 */
} Cut;

int detect_cuts(float *scores, double *timestamps, int n_frames,
                float threshold, int min_gap,
                Cut *cuts, int max_cuts)
{
    if (n_frames < 2) return 0;

    /* 复刻 scdet 的 score: min(mafd, abs(mafd - prev_mafd)) */
    float *scdet_scores = (float*)malloc(n_frames * sizeof(float));
    scdet_scores[0] = 0;
    float prev_mafd = scores[0];
    for (int i = 1; i < n_frames; i++) {
        float mafd = scores[i];          /* 已经除过 (w*h) 的 MAFD */
        float diff = fabsf(mafd - prev_mafd);
        scdet_scores[i] = fminf(mafd, diff);
        prev_mafd = mafd;
    }

    /* 找超过阈值的帧 */
    int *raw_indices = (int*)malloc(n_frames * sizeof(int));
    int n_raw = 0;
    for (int i = 1; i < n_frames; i++) {
        if (scdet_scores[i] > threshold) {
            raw_indices[n_raw++] = i;
        }
    }

    /* NMS: 连续块内取分数最高的帧，间距 >= min_gap */
    char *suppressed = (char*)calloc(n_frames, sizeof(char));
    Cut *tmp_cuts = (Cut*)malloc(max_cuts * sizeof(Cut));
    int n_tmp = 0;

    for (int pi = 0; pi < n_raw && n_tmp < max_cuts; pi++) {
        int idx = raw_indices[pi];
        if (suppressed[idx]) continue;

        /* 找连续块：前后帧都在 threshold 之上的连续区间 */
        int lo = idx, hi = idx;
        while (lo > 0 && scdet_scores[lo - 1] > threshold) lo--;
        while (hi < n_frames - 1 && scdet_scores[hi + 1] > threshold) hi++;

        /* 块内取 score 最高的帧作为切割点 */
        int best = idx;
        float best_score = scdet_scores[idx];
        for (int j = lo; j <= hi; j++) {
            if (scdet_scores[j] > best_score) {
                best_score = scdet_scores[j];
                best = j;
            }
        }

        int cf = best - 1;  /* -1 修正，和 01_raw_cuts.py 一致 */
        Cut *c = &tmp_cuts[n_tmp++];
        c->cut_frame = cf;
        c->raw_frame = best;
        c->score = scdet_scores[best];

        /* 抑制 min_gap 内的其他帧 */
        int sup_lo = best - min_gap; if (sup_lo < 0) sup_lo = 0;
        int sup_hi = best + min_gap; if (sup_hi >= n_frames) sup_hi = n_frames - 1;
        for (int s = sup_lo; s <= sup_hi; s++) suppressed[s] = 1;
    }

    /* 去重 + 过滤掉 cut_frame <= 0 */
    int n_cuts = 0;
    int last_cf = -999;
    for (int i = 0; i < n_tmp; i++) {
        if (tmp_cuts[i].cut_frame <= 0) continue;
        if (tmp_cuts[i].cut_frame == last_cf) continue;
        cuts[n_cuts++] = tmp_cuts[i];
        last_cf = tmp_cuts[i].cut_frame;
    }

    free(scdet_scores); free(raw_indices); free(suppressed); free(tmp_cuts);
    return n_cuts;
}

/* ── 过滤: 删除会导致 < min_frames 片段的切割点 ── */

int filter_min_segment(Cut *cuts, int n_cuts, int min_frames, int total_frames)
{
    if (n_cuts < 2) return n_cuts;

    /* 检查每个相邻切割点之间的帧数 */
    char *keep = (char*)malloc(n_cuts * sizeof(char));
    for (int i = 0; i < n_cuts; i++) keep[i] = 1;

    /* 前向过滤：从前往后，相邻切点帧距 < min_frames 则丢弃后面的 */
    int prev = 0;
    for (int i = 0; i < n_cuts; i++) {
        int seg_frames = cuts[i].cut_frame - prev;
        if (seg_frames < min_frames) {
            keep[i] = 0;
        } else {
            prev = cuts[i].cut_frame;
        }
    }

    /* 后向过滤：最后一段 < min_frames 则丢弃最后一个切点 */
    int last_kept = -1;
    for (int i = n_cuts - 1; i >= 0; i--) {
        if (keep[i]) { last_kept = i; break; }
    }
    if (last_kept >= 0) {
        int tail_frames = (total_frames - 1) - cuts[last_kept].cut_frame;
        if (tail_frames < min_frames) {
            keep[last_kept] = 0;
        }
    }

    int n_keep = 0;
    for (int i = 0; i < n_cuts; i++) {
        if (keep[i]) {
            cuts[n_keep++] = cuts[i];
        }
    }
    free(keep);
    return n_keep;
}

/* ── Event Detector (NMS + top-N) ── */

typedef struct {
    int center_frame;
    double center_time;
    int start_frame, end_frame;
    double start_time, end_time;
    float peak_score;
    float raw_score;
    int rep_frames[4];   /* [0]=middle, [1]=medoid */
    int n_rep_frames;
} Event;

int detect_events(float *scores, double *timestamps, int n_frames,
                  float fps, float threshold, int min_distance, float window_sec,
                  Event *events, int max_events)
{
    if (n_frames == 0) return 0;

    /* Normalize using p99 */
    float *sorted = (float*)malloc(n_frames * sizeof(float));
    memcpy(sorted, scores, n_frames * sizeof(float));
    for (int i = 0; i < n_frames - 1; i++)
        for (int j = i + 1; j < n_frames; j++)
            if (sorted[j] > sorted[i]) { float t = sorted[i]; sorted[i] = sorted[j]; sorted[j] = t; }
    float p99 = sorted[(int)(n_frames * 0.99)];
    free(sorted);

    float *norm = (float*)malloc(n_frames * sizeof(float));
    for (int i = 0; i < n_frames; i++)
        norm[i] = (p99 > 0) ? scores[i] / p99 : 0.0f;

    /* Find local maxima (strict: > all neighbors in ±margin) */
    int margin = 12;
    int *peak_indices = (int*)malloc(n_frames * sizeof(int));
    int n_peaks = 0;

    for (int i = margin; i < n_frames - margin; i++) {
        if (norm[i] < threshold) continue;
        int is_peak = 1;
        for (int j = i - margin; j <= i + margin; j++) {
            if (j != i && norm[j] > norm[i]) { is_peak = 0; break; }
        }
        if (is_peak) peak_indices[n_peaks++] = i;
    }

    /* Sort peaks by score descending */
    for (int i = 0; i < n_peaks - 1; i++) {
        int best = i;
        for (int j = i + 1; j < n_peaks; j++)
            if (norm[peak_indices[j]] > norm[peak_indices[best]]) best = j;
        if (best != i) { int t = peak_indices[i]; peak_indices[i] = peak_indices[best]; peak_indices[best] = t; }
    }

    /* Top-N selection with NMS */
    int top_n = (n_peaks < max_events) ? n_peaks : max_events;
    char *suppressed = (char*)calloc(n_frames, sizeof(char));
    int n_events = 0;

    for (int p = 0; p < n_peaks && n_events < top_n; p++) {
        int idx = peak_indices[p];
        if (suppressed[idx]) continue;

        Event *e = &events[n_events];
        e->center_frame = idx;
        e->center_time = timestamps[idx];
        e->peak_score = norm[idx];
        e->raw_score = scores[idx];

        int half_win = (int)(window_sec * fps);
        e->start_frame = (idx > half_win) ? idx - half_win : 0;
        e->end_frame = (idx + half_win < n_frames) ? idx + half_win : n_frames - 1;
        e->start_time = timestamps[e->start_frame];
        e->end_time = timestamps[e->end_frame];
        e->rep_frames[0] = (e->start_frame + e->end_frame) / 2;  /* middle frame */
        e->n_rep_frames = 1;

        n_events++;

        int lo = idx - min_distance; if (lo < 0) lo = 0;
        int hi = idx + min_distance; if (hi >= n_frames) hi = n_frames - 1;
        for (int s = lo; s <= hi; s++) suppressed[s] = 1;
    }

    free(norm); free(peak_indices); free(suppressed);
    return n_events;
}

/* ── Main ── */

int main(int argc, char *argv[]) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <video.mp4> [output_dir]\n", argv[0]); return 1; }

    const char *filename = argv[1];
    const char *output_dir = (argc > 2) ? argv[2] : "/tmp/change_cuda_out";
    const int RING_BUFFER_SEC = 10;   /* Keep last 10 seconds */
    const int MEDOID_SIZE = 16;        /* Medoid at 16x9 */

    int ret;
    AVFormatContext *fmt_ctx = NULL;
    AVCodecContext *dec_ctx = NULL;
    const AVCodec *decoder = NULL;
    AVStream *video_stream = NULL;
    AVPacket *pkt = NULL;
    AVFrame *frame = NULL;
    AVBufferRef *hw_device_ctx = NULL;
    int video_stream_idx = -1, frame_count = 0;

    avformat_network_init();
    av_log_set_level(AV_LOG_WARNING);

    if ((ret = avformat_open_input(&fmt_ctx, filename, NULL, NULL)) < 0)
        { fprintf(stderr, "Cannot open: %s\n", filename); return 1; }
    if ((ret = avformat_find_stream_info(fmt_ctx, NULL)) < 0)
        { fprintf(stderr, "Cannot find stream info\n"); return 1; }

    for (int i = 0; i < (int)fmt_ctx->nb_streams; i++)
        if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
            { video_stream_idx = i; break; }
    if (video_stream_idx < 0) { fprintf(stderr, "No video stream\n"); return 1; }
    video_stream = fmt_ctx->streams[video_stream_idx];

    ret = av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, "cuda=0", NULL, 0);
    if (ret < 0) { fprintf(stderr, "Cannot create CUDA device\n"); return 1; }

    decoder = avcodec_find_decoder(video_stream->codecpar->codec_id);
    if (!decoder) { fprintf(stderr, "Decoder not found\n"); return 1; }
    dec_ctx = avcodec_alloc_context3(decoder);
    avcodec_parameters_to_context(dec_ctx, video_stream->codecpar);
    dec_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
    ret = avcodec_open2(dec_ctx, decoder, NULL);
    if (ret < 0) { fprintf(stderr, "Cannot open decoder\n"); return 1; }

    double duration_sec = (double)video_stream->duration * av_q2d(video_stream->time_base);
    float fps = (video_stream->nb_frames > 0 && duration_sec > 0)
                ? (float)video_stream->nb_frames / (float)duration_sec : 24.0f;

    fprintf(stderr, "[change_cuda] %dx%d, %ld frames, %.2ffps\n",
            dec_ctx->width, dec_ctx->height, (long)video_stream->nb_frames, fps);

    ChangeDetector detector(64, 36, 200000);

    /* Phase 3: GPU ring buffer (stores last RING_BUFFER_SEC seconds at half-res) */
    int ring_max = (int)(RING_BUFFER_SEC * fps);
    GPURingBuffer ring_buffer(ring_max, dec_ctx->height, dec_ctx->width);

    pkt = av_packet_alloc();
    frame = av_frame_alloc();

    double t_start = (double)clock() / CLOCKS_PER_SEC;

    /* ── Decode loop: Stage 1 (change scores) + Stage 3 (ring buffer) ── */
    while (av_read_frame(fmt_ctx, pkt) >= 0) {
        if (pkt->stream_index != video_stream_idx) { av_packet_unref(pkt); continue; }
        ret = avcodec_send_packet(dec_ctx, pkt);
        if (ret < 0) { av_packet_unref(pkt); continue; }

        while (ret >= 0) {
            ret = avcodec_receive_frame(dec_ctx, frame);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
            if (ret < 0) break;

            if (frame->format == AV_PIX_FMT_CUDA && frame->data[0]) {
                double pts = frame->pts * av_q2d(video_stream->time_base);
                float score = detector.process(
                    (uint8_t*)(intptr_t)frame->data[0],
                    frame->linesize[0], frame->height, pts);

                /* Push to GPU ring buffer */
                float ts_sec = (float)(frame->pts * av_q2d(video_stream->time_base));
                ring_buffer.push(
                    (uint8_t*)(intptr_t)frame->data[0],
                    frame->linesize[0], frame->height,
                    frame->pts, ts_sec);

                frame_count++;
                if (frame_count % 1000 == 0)
                    fprintf(stderr, "  frame=%d score=%.4f (%.0fs)\n",
                            frame_count, score, (double)(clock())/CLOCKS_PER_SEC - t_start);
            }
            av_frame_unref(frame);
        }
        av_packet_unref(pkt);
    }

    /* Flush decoder */
    avcodec_send_packet(dec_ctx, NULL);
    while (avcodec_receive_frame(dec_ctx, frame) >= 0) {
        if (frame->format == AV_PIX_FMT_CUDA && frame->data[0]) {
            double pts = frame->pts * av_q2d(video_stream->time_base);
            detector.process((uint8_t*)(intptr_t)frame->data[0],
                             frame->linesize[0], frame->height, pts);
            frame_count++;
        }
        av_frame_unref(frame);
    }

    double t_decode_end = (double)clock() / CLOCKS_PER_SEC;

    /* ── Scene cuts detection ── */
    fprintf(stderr, "\n[Scene cuts] Detecting cuts...\n");
    Cut cuts[2000];
    int n_cuts = detect_cuts(
        detector.scores, detector.timestamps, detector.n_frames,
        10.0f, 5, cuts, 2000
    );
    /* 禁止 < 1s 片段 (假设 fps=30) */
    int min_frames = (int)(1.0f * fps + 0.5f);
    int before = n_cuts;
    n_cuts = filter_min_segment(cuts, n_cuts, min_frames, frame_count);
    fprintf(stderr, "  %d scene cuts detected (filtered %d -> %d, min=%dfr=1s)\n",
            n_cuts, before, n_cuts, min_frames);

    /* ── Phase 2: Event detection ── */
    fprintf(stderr, "\n[Phase 2] Detecting events...\n");
    Event events[200];
    int n_events = detect_events(
        detector.scores, detector.timestamps, detector.n_frames,
        fps, 0.10f, 72, 3.0f, events, 200
    );
    fprintf(stderr, "  %d events detected\n", n_events);

    /* ── Phase 3: Representative frame extraction ── */
    fprintf(stderr, "[Phase 3] Extracting representative frames...\n");
    for (int e = 0; e < n_events; e++) {
        Event *ev = &events[e];

        /* Get window frames from ring buffer */
        int win_frames[100];
        int n_win;
        ring_buffer.get_window_frames((float)ev->center_time, 3.0f,
                                      win_frames, &n_win, 100);

        if (n_win == 0) {
            ev->rep_frames[0] = ev->center_frame;
            ev->n_rep_frames = 1;
            continue;
        }

        /* Strategy 1: middle frame (already set) */
        int middle_fn = ev->rep_frames[0];

        /* Strategy 2: medoid = frame closest to center of window */
        int mid_idx = n_win / 2;
        int medoid_fn = win_frames[mid_idx];

        /* Sanity: clamp to valid range */
        if (medoid_fn < 0 || medoid_fn > 300000) medoid_fn = middle_fn;

        /* Combine: middle + medoid (deduplicate) */
        if (medoid_fn != middle_fn) {
            ev->rep_frames[0] = middle_fn;
            ev->rep_frames[1] = medoid_fn;
            ev->n_rep_frames = 2;
        } else {
            ev->n_rep_frames = 1;
        }
    }

    double t_end = (double)clock() / CLOCKS_PER_SEC;

    /* ── Write output ── */
    mkdir(output_dir, 0755);

    char csv_path[1024];
    snprintf(csv_path, sizeof(csv_path), "%s/change_curve.csv", output_dir);
    FILE *fout = fopen(csv_path, "w");
    if (fout) {
        fprintf(fout, "frame,timestamp_sec,raw_score,normalized_score\n");
        float max_score = 0;
        for (int i = 1; i < detector.n_frames; i++)
            if (detector.scores[i] > max_score) max_score = detector.scores[i];
        for (int i = 0; i < detector.n_frames; i++) {
            float norm = (max_score > 0) ? detector.scores[i] / max_score : 0;
            fprintf(fout, "%d,%.6f,%.6f,%.6f\n", i, detector.timestamps[i],
                    detector.scores[i], norm);
        }
        fclose(fout);
        fprintf(stderr, "  Saved: %s\n", csv_path);
    }

    char json_path[1024];
    snprintf(json_path, sizeof(json_path), "%s/events.json", output_dir);
    FILE *fj = fopen(json_path, "w");
    if (fj) {
        fprintf(fj, "{\n");
        fprintf(fj, "  \"video\": \"%s\",\n", filename);
        fprintf(fj, "  \"total_frames\": %d,\n", frame_count);
        fprintf(fj, "  \"fps\": %.4f,\n", fps);
        fprintf(fj, "  \"n_scores\": %d,\n", detector.n_frames);
        fprintf(fj, "  \"n_events\": %d,\n", n_events);
        fprintf(fj, "  \"decode_time_sec\": %.2f,\n", t_decode_end - t_start);
        fprintf(fj, "  \"total_time_sec\": %.2f,\n", t_end - t_start);
        fprintf(fj, "  \"events\": [\n");
        for (int i = 0; i < n_events; i++) {
            Event *e = &events[i];
            fprintf(fj, "    {\n");
            fprintf(fj, "      \"id\": %d,\n", i + 1);
            fprintf(fj, "      \"start_frame\": %d,\n", e->start_frame);
            fprintf(fj, "      \"end_frame\": %d,\n", e->end_frame);
            fprintf(fj, "      \"center_frame\": %d,\n", e->center_frame);
            fprintf(fj, "      \"start_time\": %.3f,\n", e->start_time);
            fprintf(fj, "      \"end_time\": %.3f,\n", e->end_time);
            fprintf(fj, "      \"center_time\": %.3f,\n", e->center_time);
            fprintf(fj, "      \"peak_score\": %.4f,\n", e->peak_score);
            fprintf(fj, "      \"raw_score\": %.4f,\n", e->raw_score);
            fprintf(fj, "      \"representative_frames\": [");
            for (int r = 0; r < e->n_rep_frames; r++) {
                fprintf(fj, "%d", e->rep_frames[r]);
                if (r < e->n_rep_frames - 1) fprintf(fj, ", ");
            }
            fprintf(fj, "]\n");
            fprintf(fj, "    }%s\n", (i < n_events - 1) ? "," : "");
        }
        fprintf(fj, "  ]\n}\n");
        fclose(fj);
        fprintf(stderr, "  Saved: %s\n", json_path);
    }

    /* ── Write raw_cuts.json (01_raw_cuts.py 兼容格式) ── */
    char cuts_path[1024];
    snprintf(cuts_path, sizeof(cuts_path), "%s/raw_cuts.json", output_dir);
    FILE *fc = fopen(cuts_path, "w");
    if (fc) {
        fprintf(fc, "{\n");
        fprintf(fc, "  \"video\": \"%s\",\n", filename);
        fprintf(fc, "  \"total_frames\": %d,\n", frame_count);
        fprintf(fc, "  \"duration\": %.6f,\n", duration_sec);
        fprintf(fc, "  \"fps\": %.4f,\n", fps);
        fprintf(fc, "  \"width\": %d,\n", dec_ctx->width);
        fprintf(fc, "  \"height\": %d,\n", dec_ctx->height);
        fprintf(fc, "  \"cuts\": [\n");
        for (int i = 0; i < n_cuts; i++) {
            fprintf(fc, "    %d%s\n", cuts[i].cut_frame,
                    (i < n_cuts - 1) ? "," : "");
        }
        fprintf(fc, "  ]\n}\n");
        fclose(fc);
        fprintf(stderr, "  Saved: %s (%d cuts)\n", cuts_path, n_cuts);

        // Also write to 00_cuts/cuts.json (standard pipeline format)
        char parent[1024];
        snprintf(parent, sizeof(parent), "%s", output_dir);
        char *last_slash = strrchr(parent, '/');
        if (last_slash) {
            *last_slash = '\0';  // parent directory
            char cuts_dir[1024];
            snprintf(cuts_dir, sizeof(cuts_dir), "%s/00_cuts", parent);
            mkdir(cuts_dir, 0755);
            snprintf(cuts_dir, sizeof(cuts_dir), "%s/00_cuts/cuts.json", parent);
            FILE *fc2 = fopen(cuts_dir, "w");
            if (fc2) {
                fprintf(fc2, "{\n");
                fprintf(fc2, "  \"video\": \"%s\",\n", filename);
                fprintf(fc2, "  \"total_frames\": %d,\n", frame_count);
                fprintf(fc2, "  \"duration\": %.6f,\n", duration_sec);
                fprintf(fc2, "  \"fps\": %.4f,\n", fps);
                fprintf(fc2, "  \"width\": %d,\n", dec_ctx->width);
                fprintf(fc2, "  \"height\": %d,\n", dec_ctx->height);
                fprintf(fc2, "  \"cuts\": [\n");
                for (int i = 0; i < n_cuts; i++) {
                    fprintf(fc2, "    %d%s\n", cuts[i].cut_frame,
                            (i < n_cuts - 1) ? "," : "");
                }
                fprintf(fc2, "  ]\n}\n");
                fclose(fc2);
                fprintf(stderr, "  Saved: %s (%d cuts)\n", cuts_dir, n_cuts);
            }
        }
    }

    fprintf(stderr, "\n=== Done ===\n");
    fprintf(stderr, "  Scene cuts: %d\n", n_cuts);
    fprintf(stderr, "  Decode+GPU: %.2fs (%.0f fps)\n", t_decode_end - t_start,
            frame_count / (t_decode_end - t_start));
    fprintf(stderr, "  Ring buffer: %d frames\n", ring_buffer.total_seen());
    fprintf(stderr, "  Output: %s/\n", output_dir);

    av_frame_free(&frame); av_packet_free(&pkt);
    avcodec_free_context(&dec_ctx); av_buffer_unref(&hw_device_ctx);
    avformat_close_input(&fmt_ctx); avformat_network_deinit();
    return 0;
}
