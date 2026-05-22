# Example 06: Double-Buffered cp.async Pipeline Core (`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`)

This stage keeps Stage 05 multi-warp mapping and introduces a double-buffered `cp.async` pipeline for K-tile staging.

## What Changed

- Matrix dimensions: unchanged from stage 05 default (`A=16x64`, `B=64x32`, `C=16x32`).
- Kernel structure change: loop order becomes `n_block` outer, `k_tile` inner so each N block can stream K tiles through a 2-stage pipeline.
- Staging change: synchronous global->shared loads are replaced by async `cp.async.ca.shared.global` copies.
- Buffering change: shared tiles become ping-pong buffers (`stage 0`/`stage 1`) for both A and B.
- Synchronization change: `cp.async.commit_group` + `cp.async.wait_group 0` are added around tile handoff; `__syncthreads()` remains for block-level visibility.
- Helper change: none in PTX MMA mapping; `mma_ptx.cuh` interface is unchanged.
- Host/config change: stage id and binary updated; this core stage still enforces `N % 32 == 0`.

## Execution path table

Default config: `A=16x64`, `B=64x32` (`K=64`, `N=32`).

| # | k_tile | n_tile | operation | accumulator |
|---|--------|--------|-----------|-------------|
| 1  | 0  | 0  | `warp0: C[:,0:7]   += A[:,0:15]  @ B[0:15,0:7]`   | `C += partial` |
| 2  | 0  | 8  | `warp1: C[:,8:15]  += A[:,0:15]  @ B[0:15,8:15]`  | `C += partial` |
| 3  | 0  | 16 | `warp2: C[:,16:23] += A[:,0:15]  @ B[0:15,16:23]` | `C += partial` |
| 4  | 0  | 24 | `warp3: C[:,24:31] += A[:,0:15]  @ B[0:15,24:31]` | `C += partial` |
| 5  | 16 | 0  | `warp0: C[:,0:7]   += A[:,16:31] @ B[16:31,0:7]`  | `C += partial` |
| 6  | 16 | 8  | `warp1: C[:,8:15]  += A[:,16:31] @ B[16:31,8:15]` | `C += partial` |
| 7  | 16 | 16 | `warp2: C[:,16:23] += A[:,16:31] @ B[16:31,16:23]`| `C += partial` |
| 8  | 16 | 24 | `warp3: C[:,24:31] += A[:,16:31] @ B[16:31,24:31]`| `C += partial` |
| 9  | 32 | 0  | `warp0: C[:,0:7]   += A[:,32:47] @ B[32:47,0:7]`  | `C += partial` |
| 10 | 32 | 8  | `warp1: C[:,8:15]  += A[:,32:47] @ B[32:47,8:15]` | `C += partial` |
| 11 | 32 | 16 | `warp2: C[:,16:23] += A[:,32:47] @ B[32:47,16:23]`| `C += partial` |
| 12 | 32 | 24 | `warp3: C[:,24:31] += A[:,32:47] @ B[32:47,24:31]`| `C += partial` |
| 13 | 48 | 0  | `warp0: C[:,0:7]   += A[:,48:63] @ B[48:63,0:7]`  | `C += partial` |
| 14 | 48 | 8  | `warp1: C[:,8:15]  += A[:,48:63] @ B[48:63,8:15]` | `C += partial` |
| 15 | 48 | 16 | `warp2: C[:,16:23] += A[:,48:63] @ B[48:63,16:23]`| `C += partial` |
| 16 | 48 | 24 | `warp3: C[:,24:31] += A[:,48:63] @ B[48:63,24:31]`| `C += partial` |

Pipeline note (same calls as stage 05): while MMA consumes `k_tile=t` from buffer `curr`, `cp.async` prefetches `k_tile=t+16` into buffer `next`.

### cp.async commit/wait semantics

- PTX ISA reference: <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async>
- `cp.async.commit_group` closes the current batch of previously issued `cp.async` copies, making that batch a waitable group.
- `cp.async.wait_group 0` waits until all previously committed async-copy groups are complete before shared-memory data is consumed by MMA.
- In this kernel:
  - warmup path: `commit_group` then `wait_group 0` ensures buffer `0` is fully staged before first MMA.
  - steady state: we prefetch next tile, `commit_group`, compute current tile, then `wait_group 0` before swapping buffers.

### Copy Granularity And Indexing

- This stage uses `cp.async ... , 4`, so each async copy moves exactly `4` bytes.
- With `half` inputs (`2` bytes each), one copy moves `2` elements (`half2`-sized chunk).
- In code, `*_col_h2` means "column index in 2-half chunks":
  - `a_col_h2` / `b_col_h2` index 4-byte words inside a row.
  - `a_col = a_col_h2 * 2`, `b_col = b_col_h2 * 2` convert to half-element columns.
- Per-tile copy counts:
  - A tile (`16x16` half): `16 * 16 * 2 = 512` bytes -> `512 / 4 = 128` copies.
  - B tile (`16x32` half): `16 * 32 * 2 = 1024` bytes -> `1024 / 4 = 256` copies.
- Thread mapping:
  - A: one 4-byte word per thread (first 128 threads).
  - B: stride loop (`b_word += blockDim.x`) so all 256 words are covered.

### Pipeline Timeline

- Warmup:
  - issue async copies for `k_tile=0` into buffer `0`,
  - `commit_group`,
  - `wait_group 0`,
  - `__syncthreads`.
- Steady state (`k_iter = 0..k_tiles-1`):
  - optionally prefetch `k_iter+1` into alternate buffer and `commit_group`,
  - run MMA + store using current buffer,
  - `wait_group 0` (if next exists),
  - `__syncthreads`,
  - flip `curr/next` via ping-pong index.

## Why

- Isolates async copy and double-buffer pipeline mechanics before adding boundary complexity.
- Keeps MMA mapping unchanged so you can focus on staging timeline (`prefetch/commit/wait`).
- Creates the exact control structure needed for later profiling and overlap analysis.

## PTX Mapping Delta

- `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` is unchanged.
- Per-lane fragment formulas are unchanged.
- Data movement path changes:
  - stage 05: synchronous global load -> shared -> MMA,
  - stage 06: `cp.async` global->shared pipeline with 2 shared buffers -> MMA.

## Run, Files, Constraints

Run:
```bash
cd examples/06_cp_async_pipeline
make all
```

Files:
- `src/main.cu`: host I/O, shape checks, and launch.
- `src/kernel.cu`: pipeline schedule + MMA compute flow.
- `src/cp_async_staging.cuh`: cp.async primitives (`issue/commit/wait`) and the core K-tile staging helper.
- `src/mma_ptx.cuh`: unchanged MMA fragment pack + inline PTX MMA helper.

Constraints:
- `M = 16`
- `K % 16 == 0`
- `N % 32 == 0`
- Requires `sm_80+` for actual `cp.async` execution (default build target: `sm_89`).
