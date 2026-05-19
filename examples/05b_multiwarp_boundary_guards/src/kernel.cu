#include "mma_ptx.cuh"
#include <cuda_fp16.h>

namespace {
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;

constexpr int WARPS_PER_BLOCK = 4;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;
constexpr int BLOCK_TILE_N = WARPS_PER_BLOCK * MMA_N;  // 16x32 C tile per block.
}  // namespace

/*
 * Stage 05b: multi-warp block tiling with boundary guards.
 *
 * Delta vs stage 05:
 *   - Adds guarded shared-memory B loads for partial tail tiles.
 *   - Adds warp-participation guard so only valid 8-column subtile owners
 *     execute MMA/store when N is not a multiple of 32.
 *   - Core multi-warp mapping and mma.sync fragment formulas are unchanged.
 */
__global__ void tensor_core_kernel(const half *A, const half *B, float *C, int K, int N) {
    __shared__ half sA[MMA_M][MMA_K];
    __shared__ half sB[MMA_K][BLOCK_TILE_N];

    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;        // 0..3
    const int block_tid = threadIdx.x;           // 0..127
    const int warp_tile_col = warp_id * MMA_N;   // 0, 8, 16, 24

    for (int k_tile_base = 0; k_tile_base < K; k_tile_base += MMA_K) {
        for (int idx = block_tid; idx < MMA_M * MMA_K; idx += THREADS_PER_BLOCK) {
            const int r = idx / MMA_K;
            const int c = idx % MMA_K;
            sA[r][c] = A[r * K + (k_tile_base + c)];
        }
        __syncthreads();

        // Same block-tiling loop as stage 05, now with tail guards.
        for (int n_block_base = 0; n_block_base < N; n_block_base += BLOCK_TILE_N) {
            for (int idx = block_tid; idx < MMA_K * BLOCK_TILE_N; idx += THREADS_PER_BLOCK) {
                const int r = idx / BLOCK_TILE_N;
                const int c = idx % BLOCK_TILE_N;
                const int global_col = n_block_base + c;
                half value = __float2half(0.0f);
                if (global_col < N) {
                    value = B[(k_tile_base + r) * N + global_col];
                }
                sB[r][c] = value;
            }
            __syncthreads();

            if (n_block_base + warp_tile_col < N) {
                float d_tile[4] = {0.0f, 0.0f, 0.0f, 0.0f};

                mma_sync_m16n8k16_row_col_f16_smem(
                    d_tile,
                    &sA[0][0],
                    &sB[0][warp_tile_col],
                    lane,
                    MMA_K,
                    BLOCK_TILE_N);

                const int group = lane >> 2;
                const int tid = lane & 3;

                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    const int row = (i < 2) ? group : (group + 8);
                    const int col = n_block_base + warp_tile_col + tid * 2 + (i & 1);
                    C[row * N + col] += d_tile[i];
                }
            }
            __syncthreads();
        }
    }
}
