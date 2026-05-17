#include "mma_ptx.cuh"
#include <cuda_fp16.h>

/*
 * Stage 04: shared-memory staging.
 *
 * Delta vs stage 03:
 *   - A and B tiles are cooperatively loaded into __shared__ memory before
 *     fragment packing (instead of direct global-memory fragment reads).
 *   - Loop order is swapped: K-tile outer loop, N-tile inner loop.
 *   - Each MMA computes one (k_tile, n_tile) partial that is accumulated
 *     into persistent C via C += d_tile.
 */
__global__ void tensor_core_kernel(const half *A, const half *B, float *C, int K, int N) {
    // Tile sizes match one m16n8k16 operation:
    //   sA: 16x16 half, sB: 16x8 half.
    __shared__ half sA[16][16];
    __shared__ half sB[16][8];

    const int lane = threadIdx.x & 31;
    const int group = lane >> 2;
    const int tid = lane & 3;

    // Stage-04 structure: K outer loop to enable A-tile reuse across N tiles.
    for (int k_tile_base = 0; k_tile_base < K; k_tile_base += 16) {
        // Cooperative A load: 32 lanes x 8 values = full 16x16 tile.
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            // Coalesced mapping: for fixed i, lanes touch consecutive addresses.
            const int idx = i * 32 + lane;
            const int r = idx / 16;
            const int c = idx % 16;
            sA[r][c] = A[r * K + k_tile_base + c];
        }
        __syncwarp();

        // N inner loop: load one B tile per N-slice and issue one MMA.
        for (int n_tile_base = 0; n_tile_base < N; n_tile_base += 8) {
            // Per-call accumulator for one 16x8 output fragment.
            float d_tile[4] = {0.0f, 0.0f, 0.0f, 0.0f};

            // Cooperative B load: 32 lanes x 4 values = full 16x8 tile.
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                // Coalesced mapping: for fixed i, lanes touch consecutive addresses.
                const int idx = i * 32 + lane;
                const int r = idx / 8;
                const int c = idx % 8;
                sB[r][c] = B[(k_tile_base + r) * N + (n_tile_base + c)];
            }
            __syncwarp();

            // Fragment pack now reads from shared-memory tiles.
            mma_sync_m16n8k16_row_col_f16_smem(d_tile, &sA[0][0], &sB[0][0], lane);

            // Accumulate this K-slice partial into global C.
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int row = (i < 2) ? group : (group + 8);
                const int col = n_tile_base + tid * 2 + (i & 1);
                C[row * N + col] += d_tile[i];
            }
            __syncwarp();
        }
    }
}
