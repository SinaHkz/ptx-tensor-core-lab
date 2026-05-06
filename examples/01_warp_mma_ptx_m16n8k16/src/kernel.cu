#include "mma_ptx.cuh"
#include <cuda_fp16.h>

/*
 * One-warp educational Tensor Core kernel.
 *
 * Computes a full 16x16 output tile C from 16x16 A and 16x16 B by issuing two
 * PTX MMA operations of shape m16n8k16:
 *   tile 0 -> output columns [0..7]
 *   tile 1 -> output columns [8..15]
 *
 * PTX formulation used:
 *   mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
 *   D = A * B + C
 *
 * Fragment ownership and lane-to-matrix index formulas are documented in
 * NVIDIA PTX ISA section:
 *   "Matrix Fragments for mma.m16n8k16 with floating point type".
 */
__global__ void tensor_core_kernel(const half *A, const half *B, float *C) {
    // We use one warp (32 lanes). Lane ID inside the warp.
    int lane = threadIdx.x & 31;

    // PTX fragment mapping helpers:
    //   group = lane >> 2    -> selects one of 8 lane-groups
    //   tid   = lane & 3     -> lane position inside 4-lane group
    int group = lane >> 2;
    int tid = lane & 3;

    // First 16x8 output tile (N columns 0..7), initialized with C fragment = 0.
    float d_tile0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    mma_sync_m16n8k16_row_col_f16(d_tile0, A, B, 16, 16, 0, lane);

    // Second 16x8 output tile (N columns 8..15), same K depth (16).
    float d_tile1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    mma_sync_m16n8k16_row_col_f16(d_tile1, A, B, 16, 16, 8, lane);

    /*
     * Store lane-local accumulator fragment (4 fp32 values) back to row-major C.
     *
     * PTX C/D fragment mapping for m16n8k16:
     *   row = group          for i < 2
     *   row = group + 8      for i >= 2
     *   col = tid*2 + (i&1)  for i in {0,1,2,3}
     *
     * We apply this mapping once for tile0 (base col 0) and once for tile1
     * (base col 8) to reconstruct the complete 16x16 output.
     */
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int row = (i < 2) ? group : (group + 8);
        int col_in_tile = tid * 2 + (i & 1);
        C[row * 16 + col_in_tile] = d_tile0[i];
        C[row * 16 + (8 + col_in_tile)] = d_tile1[i];
    }
}
