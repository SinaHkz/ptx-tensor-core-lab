#include "mma_ptx.cuh"
#include <cuda_fp16.h>

/*
 * Stage 02: one warp + parametric N tiling.
 *
 * Delta vs stage 01:
 * - output-N is tiled in a loop (8 columns per iteration),
 * - each iteration issues one m16n8k16 MMA for the current tile_base_col.
 */
__global__ void tensor_core_kernel(const half *A, const half *B, float *C, int n_cols) {
    const int lane = threadIdx.x & 31;
    const int group = lane >> 2;
    const int tid = lane & 3;

    for (int tile_base_col = 0; tile_base_col < n_cols; tile_base_col += 8) {
        float d_tile[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        // Stage-02 update: ldb=n_cols enables B as 16xN instead of fixed 16x16.
        mma_sync_m16n8k16_row_col_f16(d_tile, A, B, 16, n_cols, tile_base_col, lane);

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int row = (i < 2) ? group : (group + 8);
            const int col = tile_base_col + tid * 2 + (i & 1);
            C[row * n_cols + col] = d_tile[i];
        }
    }
}
