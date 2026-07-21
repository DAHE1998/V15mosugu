#include "change_cuda.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdlib.h>
#include <string.h>

#define MAX_HISTORY 200000

static float d_scores[MAX_HISTORY];
static double d_timestamps[MAX_HISTORY];
static int d_n_frames = 0;
static int small_w_prev = 0, small_h_prev = 0;
static uint8_t *d_curr_small = NULL;
static uint8_t *d_prev_small = NULL;
static float *d_score = NULL;

/* CUDA kernel: bilinear resize of Y plane */
__global__ void resize_y_kernel(
    const uint8_t *__restrict__ src_y,
    uint8_t *__restrict__ dst_y,
    int src_w, int src_h,
    int dst_w, int dst_h
) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= dst_h || x >= dst_w) return;

    float sy = (float)y * src_h / dst_h;
    float sx = (float)x * dst_w / dst_w;

    int y0 = (int)sy;
    int x0 = (int)sx;
    int y1 = y0 + 1;
    if (y1 >= src_h) y1 = src_h - 1;
    int x1 = x0 + 1;
    if (x1 >= src_w) x1 = src_w - 1;

    float fy = sy - y0;
    float fx = sx - x0;

    uint8_t v00 = src_y[y0 * src_w + x0];
    uint8_t v10 = src_y[y0 * src_w + x1];
    uint8_t v01 = src_y[y1 * src_w + x0];
    uint8_t v11 = src_y[y1 * src_w + x1];

    float v = (1 - fy) * ((1 - fx) * v00 + fx * v10)
            + fy      * ((1 - fx) * v01 + fx * v11);

    dst_y[y * dst_w + x] = (uint8_t)(v + 0.5f);
}

/* CUDA kernel: MAFD - Mean Absolute Frame Difference */
__global__ void mafd_kernel(
    const uint8_t *__restrict__ curr_y,
    float *__restrict__ score,
    int width, int height
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    if (idx >= total) return;

    /* Use atomic add for reduction */
    float diff = abs((float)curr_y[idx] - (float)curr_y[total + idx]);
    atomicAdd(score, diff);
}

extern "C" {

int change_cuda_init(int device_id) {
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) return -1;
    return 0;
}

void change_cuda_process(uint8_t *frame_data, int width, int height,
                         float *out_score, double pts) {
    int dst_w = 64;
    int dst_h = 36;

    /* Allocate buffers on first call */
    if (d_curr_small == NULL) {
        cudaMalloc(&d_curr_small, dst_w * dst_h);
        cudaMalloc(&d_prev_small, dst_w * dst_h);
        cudaMalloc(&d_score, sizeof(float));
        small_w_prev = dst_w;
        small_h_prev = dst_h;
    }

    /* Resize current frame */
    dim3 block(16, 16);
    dim3 grid((dst_w + 15) / 16, (dst_h + 15) / 16);
    resize_y_kernel<<<grid, block>>>(frame_data, d_curr_small,
                                      width, height, dst_w, dst_h);

    if (d_n_frames > 0) {
        /* Compute MAFD */
        cudaMemset(d_score, 0, sizeof(float));
        int total = dst_w * dst_h;
        int bs = 256;
        int gs = (total + bs - 1) / bs;
        mafd_kernel<<<gs, bs>>>(d_curr_small, d_score, dst_w, dst_h);
        cudaDeviceSynchronize();

        cudaMemcpy(out_score, d_score, sizeof(float), cudaMemcpyDeviceToHost);
        *out_score /= (float)(dst_w * dst_h);

        /* Swap */
        cudaMemcpy(d_prev_small, d_curr_small, dst_w * dst_h, cudaMemcpyDeviceToDevice);
    } else {
        /* First frame */
        cudaMemcpy(d_prev_small, d_curr_small, dst_w * dst_h, cudaMemcpyDeviceToDevice);
        *out_score = 0.0f;
    }

    /* Store history */
    if (d_n_frames < MAX_HISTORY) {
        d_scores[d_n_frames] = *out_score;
        d_timestamps[d_n_frames] = pts;
        d_n_frames++;
    }
}

void change_cuda_get_history(float *scores, double *timestamps, int *n_frames) {
    *n_frames = d_n_frames;
    if (scores && d_n_frames > 0) {
        memcpy(scores, d_scores, d_n_frames * sizeof(float));
    }
    if (timestamps && d_n_frames > 0) {
        memcpy(timestamps, d_timestamps, d_n_frames * sizeof(double));
    }
}

void change_cuda_cleanup(void) {
    if (d_curr_small) cudaFree(d_curr_small);
    if (d_prev_small) cudaFree(d_prev_small);
    if (d_score) cudaFree(d_score);
    d_curr_small = NULL;
    d_prev_small = NULL;
    d_score = NULL;
    d_n_frames = 0;
    small_w_prev = 0;
    small_h_prev = 0;
}

} /* extern "C" */
