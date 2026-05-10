#include "mma_ptx.cuh"
#include <cuda_fp16.h>

__global__ void tensor_core_kernel(const half *A, const half *B, float *C, int K, int N) {
    const int lane = threadIdx.x & 31;
    const int group = lane >> 2;
    const int tid = lane & 3;

    for (int n_tile_base = 0; n_tile_base < N; n_tile_base += 8) {
        float d_tile[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int k_tile_base = 0; k_tile_base < K; k_tile_base += 16) {
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
