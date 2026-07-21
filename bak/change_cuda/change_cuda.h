#ifndef CHANGE_CUDA_H
#define CHANGE_CUDA_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize the CUDA change detection kernel.
 * Returns 0 on success, negative on error.
 */
int change_cuda_init(int device_id);

/* Process one CUDA frame.
 * frame_data: CUDA device pointer (uint8_t Y plane)
 * width, height: Y plane dimensions (after resize)
 * score: output change score
 * pts: presentation timestamp (seconds)
 */
void change_cuda_process(uint8_t *frame_data, int width, int height,
                         float *score, double pts);

/* Get the full change score history for event detection.
 * scores: output array (caller-allocated, size = n_frames)
 * timestamps: output array (caller-allocated, size = n_frames)
 * n_frames: number of frames processed
 */
void change_cuda_get_history(float *scores, double *timestamps, int *n_frames);

/* Cleanup. */
void change_cuda_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif /* CHANGE_CUDA_H */
