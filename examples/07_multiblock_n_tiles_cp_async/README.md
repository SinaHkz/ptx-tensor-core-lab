# Example 07: Multi-Block N-Tiles with cp.async (`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`)

This stage keeps the guarded double-buffered `cp.async` K pipeline from stage 06b and moves 32-column N-block ownership to `grid.x`.

## What Changed

- Matrix dimensions: unchanged from stage 06b (`A=16x64`, `B=64x40`, `C=16x40` in default config).
- Kernel structure change: removed in-kernel `for (n_block_base ...)` loop; each block now owns one N block tile with `n_block_base = blockIdx.x * 32`.
- Kernel launch change: host computes `grid_x = ceil_div(N, 32)` and launches one block per N block tile.
- Boundary behavior: guarded B staging and guarded warp participation are unchanged from stage 06b (`N % 8 == 0` supported).
- Helper change: none (`stage_k_tile_cp_async_guarded` and MMA helper are reused as-is).
- Config change: example id/stage id and binary path updated to stage `07_multiblock_n_tiles_cp_async`.

## Execution path table

Default config: `A=16x64`, `B=64x40` (`K=64`, `N=40`), so `grid.x=2` (`block_x=0` owns cols `0..31`, `block_x=1` owns cols `32..39`).

| # | block_x | k_tile | n_tile | operation | accumulator |
|---|---------|--------|--------|-----------|-------------|
| 1  | 0 | 0  | 0  | `warp0: C[:,0:7]   += A[:,0:15]  @ B[0:15,0:7]`    | `C += partial` |
| 2  | 0 | 0  | 8  | `warp1: C[:,8:15]  += A[:,0:15]  @ B[0:15,8:15]`   | `C += partial` |
| 3  | 0 | 0  | 16 | `warp2: C[:,16:23] += A[:,0:15]  @ B[0:15,16:23]`  | `C += partial` |
| 4  | 0 | 0  | 24 | `warp3: C[:,24:31] += A[:,0:15]  @ B[0:15,24:31]`  | `C += partial` |
| 5  | 0 | 16 | 0  | `warp0: C[:,0:7]   += A[:,16:31] @ B[16:31,0:7]`   | `C += partial` |
| 6  | 0 | 16 | 8  | `warp1: C[:,8:15]  += A[:,16:31] @ B[16:31,8:15]`  | `C += partial` |
| 7  | 0 | 16 | 16 | `warp2: C[:,16:23] += A[:,16:31] @ B[16:31,16:23]` | `C += partial` |
| 8  | 0 | 16 | 24 | `warp3: C[:,24:31] += A[:,16:31] @ B[16:31,24:31]` | `C += partial` |
| 9  | 0 | 32 | 0  | `warp0: C[:,0:7]   += A[:,32:47] @ B[32:47,0:7]`   | `C += partial` |
| 10 | 0 | 32 | 8  | `warp1: C[:,8:15]  += A[:,32:47] @ B[32:47,8:15]`  | `C += partial` |
| 11 | 0 | 32 | 16 | `warp2: C[:,16:23] += A[:,32:47] @ B[32:47,16:23]` | `C += partial` |
| 12 | 0 | 32 | 24 | `warp3: C[:,24:31] += A[:,32:47] @ B[32:47,24:31]` | `C += partial` |
| 13 | 0 | 48 | 0  | `warp0: C[:,0:7]   += A[:,48:63] @ B[48:63,0:7]`   | `C += partial` |
| 14 | 0 | 48 | 8  | `warp1: C[:,8:15]  += A[:,48:63] @ B[48:63,8:15]`  | `C += partial` |
| 15 | 0 | 48 | 16 | `warp2: C[:,16:23] += A[:,48:63] @ B[48:63,16:23]` | `C += partial` |
| 16 | 0 | 48 | 24 | `warp3: C[:,24:31] += A[:,48:63] @ B[48:63,24:31]` | `C += partial` |
| 17 | 1 | 0  | 32 | `warp0: C[:,32:39] += A[:,0:15]  @ B[0:15,32:39]`  | `C += partial` |
| 18 | 1 | 16 | 32 | `warp0: C[:,32:39] += A[:,16:31] @ B[16:31,32:39]` | `C += partial` |
| 19 | 1 | 32 | 32 | `warp0: C[:,32:39] += A[:,32:47] @ B[32:47,32:39]` | `C += partial` |
| 20 | 1 | 48 | 32 | `warp0: C[:,32:39] += A[:,48:63] @ B[48:63,32:39]` | `C += partial` |

Note: blocks execute independently (and may overlap in time), but ownership is deterministic by `block_x`.

## Why

- Stage 06b established correctness for guarded cp.async staging and N-tail handling.
- This stage isolates the next scalability step: parallelizing across N block tiles by assigning each tile to a distinct CTA via `grid.x`.
- Keeping K-loop math and guard logic unchanged makes the parallel mapping delta easy to review and validate.

## PTX Mapping Delta

- PTX MMA instruction is unchanged: `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`.
- Per-lane A/B fragment formulas are unchanged.
- Ownership/indexing delta vs stage 06b:
  - Before: `n_block_base` came from an in-kernel loop incrementing by `32`.
  - Now: `n_block_base = blockIdx.x * BLOCK_TILE_N`.
- Store-column formula is structurally unchanged: `col = n_block_base + warp_tile_col + tid * 2 + (i & 1)`.

## Run, Files, Constraints

Run:
```bash
cd examples/07_multiblock_n_tiles_cp_async
make all
```

Files:
- `src/main.cu`: host I/O, `grid.x` launch across N block tiles.
- `src/kernel.cu`: one-CTA-per-N-tile ownership with the same guarded cp.async K pipeline.
- `src/cp_async_staging.cuh`: guarded async global->shared tile staging helper.
- `src/mma_ptx.cuh`: MMA PTX helper (unchanged).

Constraints:
- `M = 16`
- `K % 16 == 0`
- `N % 8 == 0`
- Requires `sm_80+` for true `cp.async` behavior (default build target: `sm_89`).
