#include "mma_ptx.cuh"
#include <cuda_fp16.h>

/*
 * Stage 03: K-loop accumulation.
 *
 * Delta vs stage 02:
 *   - An inner K-tile loop is added: for each N-tile, the C fragment
 *     accumulator (d_tile) is reused across K-tiles via the "+f" MMA
 *     read-modify-write semantic: d_tile += A_frag * B_frag.
 *   - Kernel signature now takes both K and N as dynamic parameters
 *     (stage 02 only took N; K was fixed at 16).
 *   - The MMA helper is called with k_tile_base to offset A column and
 *     B row indices for each K-tile slice.
 */
__global__ void tensor_core_kernel(const half *A, const half *B, float *C, int K, int N) {
    const int lane = threadIdx.x & 31;
    const int group = lane >> 2;
    const int tid = lane & 3;

    for (int n_tile_base = 0; n_tile_base < N; n_tile_base += 8) {
        // Per-N-tile accumulator: persists across K-tiles, accumulates
        // the full reduction sum for this 16x8 output sub-tile.
        float d_tile[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int k_tile_base = 0; k_tile_base < K; k_tile_base += 16) {
            // Each MMA call loads A[0:16, k_tile_base+0:16] and
            // B[k_tile_base+0:16, n_tile_base+0:8], then accumulates
            // into d_tile: d_tile += A_frag * B_frag.
            mma_sync_m16n8k16_row_col_f16(d_tile, A, B, K, N, k_tile_base, n_tile_base, lane);
        }

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int row = (i < 2) ? group : (group + 8);
            const int col = n_tile_base + tid * 2 + (i & 1);
            C[row * N + col] = d_tile[i];
        }
    }
}
