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
 * Stage 06b: cp.async pipeline with boundary guards.
 *
 * Delta vs stage 06:
 *   - B-tile async staging now guards partial N-tail columns and zero-fills
 *     out-of-range shared entries.
 *   - Warp participation is guarded for tail 32-column block tiles.
 *   - Double-buffered cp.async pipeline structure is unchanged.
 */
__global__ void tensor_core_kernel(const half *A, const half *B, float *C, int K, int N) {
    __shared__ half sA[PIPE_STAGES][MMA_M][MMA_K];
    __shared__ half sB[PIPE_STAGES][MMA_K][BLOCK_TILE_N];

    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;        // 0..3
    const int block_tid = threadIdx.x;           // 0..127
    const int warp_tile_col = warp_id * MMA_N;   // 0, 8, 16, 24

    const int k_tiles = K / MMA_K;

    // Same cp.async double-buffer schedule as stage 06, plus tail guards:
    // - guarded B staging in cp_async_staging.cuh,
    // - guarded warp participation below for N-tail block tiles.
    for (int n_block_base = 0; n_block_base < N; n_block_base += BLOCK_TILE_N) {
        stage_k_tile_cp_async_guarded(
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

            // Overlap path: prefetch K tile (k_iter + 1) while computing k_iter.
            if (has_next) {
                const int next_k_tile_base = k_tile_base + MMA_K;
                stage_k_tile_cp_async_guarded(
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

            // Tail-safe warp mask: only warps with an in-range 8-column subtile
            // execute MMA/store on partial N block tiles.
            if (n_block_base + warp_tile_col < N) {
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
            }

            if (has_next) {
                // Ensure prefetched next buffer is fully visible before next iteration.
                cp_async_wait_group0();
            }
            __syncthreads();
        }
    }
}
