#pragma once
#include <cuda_fp16.h>
#include <stdint.h>

/*
 * PTX Tensor Core helper for:
 *   mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
 *
 * NVIDIA PTX ISA formulation:
 *   - MMA computes D = A * B + C on warp-distributed fragments.
 *   - Shape m16n8k16 means A is 16x16, B is 16x8, C/D is 16x8.
 *   - For this floating-point form, each lane contributes:
 *       A: 8 fp16 values (packed into 4 x f16x2 registers)
 *       B: 4 fp16 values (packed into 2 x f16x2 registers)
 *       C/D: 4 fp32 accumulator values
 *
 * Mapping used below follows PTX ISA section:
 *   "Matrix Fragments for mma.m16n8k16 with floating point type"
 * and instruction semantics in:
 *   "Multiply-and-Accumulate Instruction: mma".
 */
__device__ __forceinline__ uint32_t pack_half2(half lo, half hi) {
    // PTX expects f16x2 packed in one 32-bit register:
    // bits [15:0]   = first half value
    // bits [31:16]  = second half value
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
    int n_tile_base_col,
    int lane) {
    // lane decomposition used by PTX fragment formulas:
    //   groupID           = lane >> 2
    //   threadID_in_group = lane % 4
    // We keep the exact same meaning under shorter names.
    const int group = lane >> 2; // groupID, range 0..7
    const int tid = lane & 0x3;  // threadID_in_group, range 0..3

    /*
     * Build multiplicand A fragment (a0..a7), where A is interpreted as row-major.
     * For m16n8k16 floating-point fragments in PTX:
     *   row = groupID           for a0,a1,a4,a5
     *   row = groupID + 8       for a2,a3,a6,a7
     *   col = tid*2 + {0,1}     for low-k half
     *   col = tid*2 + {0,1} + 8 for high-k half
     */
    half a0 = A[group * lda + (tid * 2 + 0)];
    half a1 = A[group * lda + (tid * 2 + 1)];
    half a2 = A[(group + 8) * lda + (tid * 2 + 0)];
    half a3 = A[(group + 8) * lda + (tid * 2 + 1)];
    half a4 = A[group * lda + (tid * 2 + 0 + 8)];
    half a5 = A[group * lda + (tid * 2 + 1 + 8)];
    half a6 = A[(group + 8) * lda + (tid * 2 + 0 + 8)];
    half a7 = A[(group + 8) * lda + (tid * 2 + 1 + 8)];

    /*
     * Build multiplicand B fragment (b0..b3), where B is interpreted as col-major
     * by the ".row.col" instruction qualifiers.
     *
     * PTX fragment rules for B in m16n8k16:
     *   row = tid*2 + {0,1}       for b0,b1
     *   row = tid*2 + {0,1} + 8   for b2,b3
     *   col = groupID
     *
     * n_tile_base_col shifts this 8-column output tile in N:
     *   0 -> columns [0..7], 8 -> columns [8..15], etc.
     */
    half b0 = B[(tid * 2 + 0) * ldb + (n_tile_base_col + group)];
    half b1 = B[(tid * 2 + 1) * ldb + (n_tile_base_col + group)];
    half b2 = B[(tid * 2 + 0 + 8) * ldb + (n_tile_base_col + group)];
    half b3 = B[(tid * 2 + 1 + 8) * ldb + (n_tile_base_col + group)];

    // Pack lane fragments exactly as inline PTX operand lists expect:
    //   A -> 4 packed f16x2 registers
    //   B -> 2 packed f16x2 registers
    const uint32_t a_reg0 = pack_half2(a0, a1);
    const uint32_t a_reg1 = pack_half2(a2, a3);
    const uint32_t a_reg2 = pack_half2(a4, a5);
    const uint32_t a_reg3 = pack_half2(a6, a7);
    const uint32_t b_reg0 = pack_half2(b0, b1);
    const uint32_t b_reg1 = pack_half2(b2, b3);

    // Per-lane accumulator fragment c0..c3 (also destination d0..d3).
    // With "+f" constraints below, registers are read-modify-written:
    // d = a*b + d, which is the PTX D = A*B + C formulation.
    float d0 = accum4[0];
    float d1 = accum4[1];
    float d2 = accum4[2];
    float d3 = accum4[3];

    /*
     * .sync    -> warp-synchronous issue; participating lanes rendezvous
     * .aligned -> all lanes in the warp must execute the same mma instruction
     *
     * Operand groups:
     *   {%0..%3}  : D/C fragment (4 fp32 registers)
     *   {%4..%7}  : A fragment   (4 packed f16x2 registers)
     *   {%8..%9}  : B fragment   (2 packed f16x2 registers)
     */
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a_reg0), "r"(a_reg1), "r"(a_reg2), "r"(a_reg3),
          "r"(b_reg0), "r"(b_reg1));

    // Write updated D fragment back to caller-provided accumulator storage.
    accum4[0] = d0;
    accum4[1] = d1;
    accum4[2] = d2;
    accum4[3] = d3;
}
