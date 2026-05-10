#pragma once
#include <cuda_fp16.h>
#include <stdint.h>

__device__ __forceinline__ uint32_t pack_half2(half lo, half hi) {
    const uint16_t lo_bits = reinterpret_cast<const __half_raw&>(lo).x;
    const uint16_t hi_bits = reinterpret_cast<const __half_raw&>(hi).x;
    return static_cast<uint32_t>(lo_bits) | (static_cast<uint32_t>(hi_bits) << 16);
}

__device__ inline void mma_sync_m16n8k16_row_col_f16(
    float *accum4,
    const half *A,
    const half *B,
    int lda,
    int ldb,
    int k_tile_base,
    int n_tile_base_col,
    int lane) {
    const int group = lane >> 2;
    const int tid = lane & 0x3;

    half a0 = A[group * lda + (k_tile_base + tid * 2 + 0)];
    half a1 = A[group * lda + (k_tile_base + tid * 2 + 1)];
    half a2 = A[(group + 8) * lda + (k_tile_base + tid * 2 + 0)];
    half a3 = A[(group + 8) * lda + (k_tile_base + tid * 2 + 1)];
    half a4 = A[group * lda + (k_tile_base + tid * 2 + 0 + 8)];
    half a5 = A[group * lda + (k_tile_base + tid * 2 + 1 + 8)];
    half a6 = A[(group + 8) * lda + (k_tile_base + tid * 2 + 0 + 8)];
    half a7 = A[(group + 8) * lda + (k_tile_base + tid * 2 + 1 + 8)];

    half b0 = B[(k_tile_base + tid * 2 + 0) * ldb + (n_tile_base_col + group)];
    half b1 = B[(k_tile_base + tid * 2 + 1) * ldb + (n_tile_base_col + group)];
    half b2 = B[(k_tile_base + tid * 2 + 0 + 8) * ldb + (n_tile_base_col + group)];
    half b3 = B[(k_tile_base + tid * 2 + 1 + 8) * ldb + (n_tile_base_col + group)];

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
