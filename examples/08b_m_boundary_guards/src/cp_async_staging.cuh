#pragma once

#include <cuda_fp16.h>
#include <stdint.h>

/*
 * cp.async staging glossary (stage 08b M-boundary guards)
 *
 * - One `cp.async ... , 4` instruction copies exactly 4 bytes.
 * - With half data (2 bytes), one copy moves 2 half elements.
 * - We call this 4-byte unit a "word" in this file.
 *
 * Stage-08b delta vs stage 08:
 * - A staging adds per-row bounds guard for partial M tiles.
 * - Out-of-range A rows are zero-filled in shared memory.
 * - B staging/N-tail behavior is unchanged from stage 08.
 */

// Convert a generic pointer to the 32-bit shared-memory address format
// expected by cp.async instructions.
__device__ __forceinline__ uint32_t smem_addr_u32(const void *ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// Issue one 4-byte asynchronous global->shared copy.
// On pre-sm80 targets we fall back to a regular 4-byte load/store.
__device__ __forceinline__ void cp_async_ca_shared_global_4(void *smem_ptr, const void *gmem_ptr) {
#if __CUDA_ARCH__ >= 800
    const uint32_t smem = smem_addr_u32(smem_ptr);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" : : "r"(smem), "l"(gmem_ptr));
#else
    *reinterpret_cast<uint32_t *>(smem_ptr) = *reinterpret_cast<const uint32_t *>(gmem_ptr);
#endif
}

// Close the current batch of issued cp.async operations.
__device__ __forceinline__ void cp_async_commit_group() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" : :);
#endif
}

// Wait until all previously committed async-copy groups are complete.
__device__ __forceinline__ void cp_async_wait_group0() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;\n" : :);
#endif
}

__device__ inline void stage_k_tile_cp_async_guarded(
    half *sA_buf,
    half *sB_buf,
    const half *A,
    const half *B,
    int M,
    int K,
    int N,
    int m_block_base,
    int k_tile_base,
    int n_block_base,
    int block_tid,
    int mma_m,
    int mma_k,
    int block_tile_n,
    int threads_per_block) {
    // Stage one A K-tile and one B block-tile into the selected ping-pong
    // buffer. A and B both support boundary-safe tail handling.

    const int a_words_per_tile = mma_m * (mma_k / 2);
    if (block_tid < a_words_per_tile) {
        const int a_words_per_row = mma_k / 2;
        const int a_row = block_tid / a_words_per_row;
        const int a_col_h2 = block_tid % a_words_per_row;
        const int a_col = a_col_h2 * 2;
        const int global_row = m_block_base + a_row;

        if (global_row < M) {
            cp_async_ca_shared_global_4(
                &sA_buf[a_row * mma_k + a_col],
                &A[global_row * K + (k_tile_base + a_col)]);
        } else {
            *reinterpret_cast<uint32_t *>(&sA_buf[a_row * mma_k + a_col]) = 0u;
        }
    }

    const int b_words_per_tile = mma_k * (block_tile_n / 2);
    for (int b_word = block_tid; b_word < b_words_per_tile; b_word += threads_per_block) {
        const int b_words_per_row = block_tile_n / 2;
        const int b_row = b_word / b_words_per_row;
        const int b_col_h2 = b_word % b_words_per_row;
        const int b_col = b_col_h2 * 2;
        const int global_col = n_block_base + b_col;

        if (global_col + 1 < N) {
            cp_async_ca_shared_global_4(
                &sB_buf[b_row * block_tile_n + b_col],
                &B[(k_tile_base + b_row) * N + global_col]);
        } else {
            *reinterpret_cast<uint32_t *>(&sB_buf[b_row * block_tile_n + b_col]) = 0u;
        }
    }
}
