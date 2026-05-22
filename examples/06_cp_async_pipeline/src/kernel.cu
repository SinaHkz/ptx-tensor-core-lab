#include "cp_async_staging.cuh"
#include "mma_ptx.cuh"
#include <cuda_fp16.h>

namespace {
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;

constexpr int WARPS_PER_BLOCK = 4;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;
constexpr int BLOCK_TILE_N = WARPS_PER_BLOCK * MMA_N;  // 16x32 C tile per block.

constexpr int PIPE_STAGES = 2;
}  // namespace

/*
 * Stage 06: cp.async double-buffered pipeline (core version, no boundary guards).
 *
 * Delta vs stage 05:
 *   - Shared-memory A/B staging in the K loop is switched from synchronous
 *     load+barrier to async global->shared copies via cp.async.
 *   - Two shared buffers are used (ping-pong) so tile k+1 is prefetched while
 *     tile k is consumed by mma.sync.
 *   - This core stage keeps full-tile assumptions from stage 05 (N % 32 == 0).
 */
__global__ void tensor_core_kernel(const half *A, const half *B, float *C, int K, int N) {
    __shared__ half sA[PIPE_STAGES][MMA_M][MMA_K];
    __shared__ half sB[PIPE_STAGES][MMA_K][BLOCK_TILE_N];

    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;        // 0..3
    const int block_tid = threadIdx.x;           // 0..127
    const int warp_tile_col = warp_id * MMA_N;   // 0, 8, 16, 24

    const int k_tiles = K / MMA_K;

    // For each 32-column output block tile:
    //   1) warmup prefetch into buffer 0,
    //   2) for each K tile:
    //        prefetch next tile into alternate buffer,
    //        compute current tile from current buffer,
    //        wait before flipping buffers.
    for (int n_block_base = 0; n_block_base < N; n_block_base += BLOCK_TILE_N) {
        // Warmup: stage first K tile into buffer 0.
        stage_k_tile_cp_async_core(
            &sA[0][0][0],
            &sB[0][0][0],
            A,
            B,
            K,
            N,
            /*k_tile_base=*/0,
            n_block_base,
            block_tid,
            MMA_M,
            MMA_K,
            BLOCK_TILE_N,
            THREADS_PER_BLOCK);
        cp_async_commit_group();
        cp_async_wait_group0();
        __syncthreads();

        for (int k_iter = 0; k_iter < k_tiles; k_iter++) {
            const int curr = k_iter & 1;
            const int next = curr ^ 1;
            const int k_tile_base = k_iter * MMA_K;
            const bool has_next = (k_iter + 1) < k_tiles;

            // Overlap path: prefetch K tile (k_iter + 1) while we compute k_iter.
            if (has_next) {
                const int next_k_tile_base = k_tile_base + MMA_K;
                stage_k_tile_cp_async_core(
                    &sA[next][0][0],
                    &sB[next][0][0],
                    A,
                    B,
                    K,
                    N,
                    next_k_tile_base,
                    n_block_base,
                    block_tid,
                    MMA_M,
                    MMA_K,
                    BLOCK_TILE_N,
                    THREADS_PER_BLOCK);
                cp_async_commit_group();
            }

            // Compute one warp-owned 16x8 tile from the current shared buffers.
            float d_tile[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            mma_sync_m16n8k16_row_col_f16_smem(
                d_tile,
                &sA[curr][0][0],
                &sB[curr][0][warp_tile_col],
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

            if (has_next) {
                // Ensure prefetched next buffer is fully visible before next iteration.
                cp_async_wait_group0();
            }
            __syncthreads();
        }
    }
}
