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
 * Stage 05: multi-warp block tiling (core mapping, no boundary guards).
 *
 * Delta vs stage 04:
 *   - One block now contains 4 warps; each warp computes one 16x8 tile.
 *   - Shared B tile is widened to 16x32 so all warps can read distinct
 *     N-subtiles from a single cooperative block load.
 *   - Synchronization is block-wide (__syncthreads) instead of warp-only,
 *     because shared-memory handoff now crosses warps.
 */
__global__ void tensor_core_kernel(const half *A, const half *B, float *C, int K, int N) {
    __shared__ half sA[MMA_M][MMA_K];
    __shared__ half sB[MMA_K][BLOCK_TILE_N];

    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;        // 0..3
    const int block_tid = threadIdx.x;           // 0..127
    const int warp_tile_col = warp_id * MMA_N;   // 0, 8, 16, 24

    for (int k_tile_base = 0; k_tile_base < K; k_tile_base += MMA_K) {
        // Block-cooperative A load: 16x16 tile used by every warp in block.
        for (int idx = block_tid; idx < MMA_M * MMA_K; idx += THREADS_PER_BLOCK) {
            const int r = idx / MMA_K;
            const int c = idx % MMA_K;
            sA[r][c] = A[r * K + (k_tile_base + c)];
        }
        __syncthreads();

        // Replaces stage-04 inner N loop (n_tile_base += 8).
        // Here one iteration advances by 32 columns because 4 warps run in
        // parallel, each owning one 8-column MMA tile (4 x 8 = 32).
        // Stage 05 intentionally assumes full 32-column tiles (N % 32 == 0).
        for (int n_block_base = 0; n_block_base < N; n_block_base += BLOCK_TILE_N) {
            // Block-cooperative B load: 16x32 tile feeds all 4 warps.
            for (int idx = block_tid; idx < MMA_K * BLOCK_TILE_N; idx += THREADS_PER_BLOCK) {
                const int r = idx / BLOCK_TILE_N;
                const int c = idx % BLOCK_TILE_N;
                sB[r][c] = B[(k_tile_base + r) * N + (n_block_base + c)];
            }
            __syncthreads();

            // Each warp owns one 16x8 output tile inside this 16x32 block tile.
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
            __syncthreads();
        }
    }
}
