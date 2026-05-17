#pragma once
#include <cuda_fp16.h>
#include <stdint.h>

__device__ __forceinline__ uint32_t pack_half2(half lo, half hi) {
    const uint16_t lo_bits = reinterpret_cast<const __half_raw&>(lo).x;
    const uint16_t hi_bits = reinterpret_cast<const __half_raw&>(hi).x;
    return static_cast<uint32_t>(lo_bits) | (static_cast<uint32_t>(hi_bits) << 16);
}

/*
 * Stage 04: shared-memory PTX MMA helper.
 *
 * Delta vs stage 03 helper:
 *   - Inputs are shared-memory tile pointers (sA, sB), not global A/B with
 *     k_tile/n_tile offset parameters.
 *   - Lane-to-fragment formulas and mma.sync instruction are unchanged.
 */
__device__ inline void mma_sync_m16n8k16_row_col_f16_smem(
    float *accum4,
    const half *sA,
    const half *sB,
    int lane) {
    const int group = lane >> 2;
    const int tid = lane & 0x3;

    // A fragment from staged 16x16 tile (same PTX per-lane mapping as 01-03).
    half a0 = sA[group * 16 + (tid * 2 + 0)];
    half a1 = sA[group * 16 + (tid * 2 + 1)];
    half a2 = sA[(group + 8) * 16 + (tid * 2 + 0)];
    half a3 = sA[(group + 8) * 16 + (tid * 2 + 1)];
    half a4 = sA[group * 16 + (tid * 2 + 0 + 8)];
    half a5 = sA[group * 16 + (tid * 2 + 1 + 8)];
    half a6 = sA[(group + 8) * 16 + (tid * 2 + 0 + 8)];
    half a7 = sA[(group + 8) * 16 + (tid * 2 + 1 + 8)];

    // B fragment from staged 16x8 tile (same PTX per-lane mapping as 01-03).
    half b0 = sB[(tid * 2 + 0) * 8 + group];
    half b1 = sB[(tid * 2 + 1) * 8 + group];
    half b2 = sB[(tid * 2 + 0 + 8) * 8 + group];
    half b3 = sB[(tid * 2 + 1 + 8) * 8 + group];

    const uint32_t a_reg0 = pack_half2(a0, a1);
    const uint32_t a_reg1 = pack_half2(a2, a3);
    const uint32_t a_reg2 = pack_half2(a4, a5);
    const uint32_t a_reg3 = pack_half2(a6, a7);
    const uint32_t b_reg0 = pack_half2(b0, b1);
    const uint32_t b_reg1 = pack_half2(b2, b3);

    float d0 = accum4[0];
    float d1 = accum4[1];
    float d2 = accum4[2];
    float d3 = accum4[3];

    // D = A * B + D, where D starts from caller-provided accum4.
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a_reg0), "r"(a_reg1), "r"(a_reg2), "r"(a_reg3),
          "r"(b_reg0), "r"(b_reg1));

    accum4[0] = d0;
    accum4[1] = d1;
    accum4[2] = d2;
    accum4[3] = d3;
}
