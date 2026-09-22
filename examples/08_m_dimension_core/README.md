# Example 08: M-Dimension Core (`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`)

This stage extends stage 07 by adding CTA ownership across M tiles via `grid.y` while keeping the guarded cp.async N/K pipeline unchanged.

## What Changed

- Matrix dimension change: default matrix now uses `M=32` (`A=32x64`, `B=64x40`, `C=32x40`) so `grid.y` launches two 16-row M tiles.
- Kernel structure change: each block now owns one `(m_tile, n_tile)` output block with:
  - `m_block_base = blockIdx.y * 16`,
  - `n_block_base = blockIdx.x * 32`.
- Helper function change: A-tile staging now uses `m_block_base` to load the correct 16-row slab of A for each CTA.
- Host-side change: runtime `M` is an explicit CLI argument (`--m`) and is passed to kernel; launch grid becomes `dim3(grid_x, grid_y)`.
- Config change: `example.id`/`learning.stage_id` set to `08_m_dimension_core`; default `matrix.a.rows` and `matrix.c.rows` changed to `32`; binary renamed.

## Execution path table

Default config: `A=32x64`, `B=64x40` (`M=32`, `K=64`, `N=40`), so `grid=(2,2)`.

| # | block (x,y) | k_tile | n_tile | operation | accumulator |
|---|-------------|--------|--------|-----------|-------------|
| 1  | (0,0) | 0  | 0  | `warp0: C[0:15,0:7]   += A[0:15,0:15]   @ B[0:15,0:7]`   | `C += partial` |
| 2  | (0,0) | 0  | 8  | `warp1: C[0:15,8:15]  += A[0:15,0:15]   @ B[0:15,8:15]`  | `C += partial` |
| 3  | (0,0) | 0  | 16 | `warp2: C[0:15,16:23] += A[0:15,0:15]   @ B[0:15,16:23]` | `C += partial` |
| 4  | (0,0) | 0  | 24 | `warp3: C[0:15,24:31] += A[0:15,0:15]   @ B[0:15,24:31]` | `C += partial` |
| 5  | (0,0) | 16 | 0  | `warp0: C[0:15,0:7]   += A[0:15,16:31]  @ B[16:31,0:7]`  | `C += partial` |
| 6  | (0,0) | 16 | 8  | `warp1: C[0:15,8:15]  += A[0:15,16:31]  @ B[16:31,8:15]` | `C += partial` |
| 7  | (0,0) | 16 | 16 | `warp2: C[0:15,16:23] += A[0:15,16:31]  @ B[16:31,16:23]`| `C += partial` |
| 8  | (0,0) | 16 | 24 | `warp3: C[0:15,24:31] += A[0:15,16:31]  @ B[16:31,24:31]`| `C += partial` |
| 9  | (0,0) | 32 | 0  | `warp0: C[0:15,0:7]   += A[0:15,32:47]  @ B[32:47,0:7]`  | `C += partial` |
| 10 | (0,0) | 32 | 8  | `warp1: C[0:15,8:15]  += A[0:15,32:47]  @ B[32:47,8:15]` | `C += partial` |
| 11 | (0,0) | 32 | 16 | `warp2: C[0:15,16:23] += A[0:15,32:47]  @ B[32:47,16:23]`| `C += partial` |
| 12 | (0,0) | 32 | 24 | `warp3: C[0:15,24:31] += A[0:15,32:47]  @ B[32:47,24:31]`| `C += partial` |
| 13 | (0,0) | 48 | 0  | `warp0: C[0:15,0:7]   += A[0:15,48:63]  @ B[48:63,0:7]`  | `C += partial` |
| 14 | (0,0) | 48 | 8  | `warp1: C[0:15,8:15]  += A[0:15,48:63]  @ B[48:63,8:15]` | `C += partial` |
| 15 | (0,0) | 48 | 16 | `warp2: C[0:15,16:23] += A[0:15,48:63]  @ B[48:63,16:23]`| `C += partial` |
| 16 | (0,0) | 48 | 24 | `warp3: C[0:15,24:31] += A[0:15,48:63]  @ B[48:63,24:31]`| `C += partial` |
| 17 | (1,0) | 0  | 32 | `warp0: C[0:15,32:39] += A[0:15,0:15]   @ B[0:15,32:39]` | `C += partial` |
| 18 | (1,0) | 16 | 32 | `warp0: C[0:15,32:39] += A[0:15,16:31]  @ B[16:31,32:39]`| `C += partial` |
| 19 | (1,0) | 32 | 32 | `warp0: C[0:15,32:39] += A[0:15,32:47]  @ B[32:47,32:39]`| `C += partial` |
| 20 | (1,0) | 48 | 32 | `warp0: C[0:15,32:39] += A[0:15,48:63]  @ B[48:63,32:39]`| `C += partial` |
| 21 | (0,1) | 0  | 0  | `warp0: C[16:31,0:7]   += A[16:31,0:15]   @ B[0:15,0:7]`   | `C += partial` |
| 22 | (0,1) | 0  | 8  | `warp1: C[16:31,8:15]  += A[16:31,0:15]   @ B[0:15,8:15]`  | `C += partial` |
| 23 | (0,1) | 0  | 16 | `warp2: C[16:31,16:23] += A[16:31,0:15]   @ B[0:15,16:23]` | `C += partial` |
| 24 | (0,1) | 0  | 24 | `warp3: C[16:31,24:31] += A[16:31,0:15]   @ B[0:15,24:31]` | `C += partial` |
| 25 | (0,1) | 16 | 0  | `warp0: C[16:31,0:7]   += A[16:31,16:31]  @ B[16:31,0:7]`  | `C += partial` |
| 26 | (0,1) | 16 | 8  | `warp1: C[16:31,8:15]  += A[16:31,16:31]  @ B[16:31,8:15]` | `C += partial` |
| 27 | (0,1) | 16 | 16 | `warp2: C[16:31,16:23] += A[16:31,16:31]  @ B[16:31,16:23]`| `C += partial` |
| 28 | (0,1) | 16 | 24 | `warp3: C[16:31,24:31] += A[16:31,16:31]  @ B[16:31,24:31]`| `C += partial` |
| 29 | (0,1) | 32 | 0  | `warp0: C[16:31,0:7]   += A[16:31,32:47]  @ B[32:47,0:7]`  | `C += partial` |
| 30 | (0,1) | 32 | 8  | `warp1: C[16:31,8:15]  += A[16:31,32:47]  @ B[32:47,8:15]` | `C += partial` |
| 31 | (0,1) | 32 | 16 | `warp2: C[16:31,16:23] += A[16:31,32:47]  @ B[32:47,16:23]`| `C += partial` |
| 32 | (0,1) | 32 | 24 | `warp3: C[16:31,24:31] += A[16:31,32:47]  @ B[32:47,24:31]`| `C += partial` |
| 33 | (0,1) | 48 | 0  | `warp0: C[16:31,0:7]   += A[16:31,48:63]  @ B[48:63,0:7]`  | `C += partial` |
| 34 | (0,1) | 48 | 8  | `warp1: C[16:31,8:15]  += A[16:31,48:63]  @ B[48:63,8:15]` | `C += partial` |
| 35 | (0,1) | 48 | 16 | `warp2: C[16:31,16:23] += A[16:31,48:63]  @ B[48:63,16:23]`| `C += partial` |
| 36 | (0,1) | 48 | 24 | `warp3: C[16:31,24:31] += A[16:31,48:63]  @ B[48:63,24:31]`| `C += partial` |
| 37 | (1,1) | 0  | 32 | `warp0: C[16:31,32:39] += A[16:31,0:15]   @ B[0:15,32:39]` | `C += partial` |
| 38 | (1,1) | 16 | 32 | `warp0: C[16:31,32:39] += A[16:31,16:31]  @ B[16:31,32:39]`| `C += partial` |
| 39 | (1,1) | 32 | 32 | `warp0: C[16:31,32:39] += A[16:31,32:47]  @ B[32:47,32:39]`| `C += partial` |
| 40 | (1,1) | 48 | 32 | `warp0: C[16:31,32:39] += A[16:31,48:63]  @ B[48:63,32:39]`| `C += partial` |

## Why

- Stage 07 already parallelized across N via `grid.x`; this stage adds the orthogonal M-axis decomposition.
- Keeping K-loop cp.async scheduling and N-tail behavior unchanged isolates a single new concept: CTA ownership over row tiles.
- This creates the minimal stepping stone before relaxing M-alignment constraints in stage 08b.

## PTX Mapping Delta

- PTX MMA opcode is unchanged: `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`.
- Per-lane fragment formulas are unchanged.
- Indexing ownership delta vs stage 07:
  - New row-base term: `m_block_base = blockIdx.y * 16`.
  - Global row store now uses `global_row = m_block_base + row_local`.
  - A staging global row uses the same `m_block_base` offset.

## Run, Files, Constraints

Run:
```bash
cd examples/08_m_dimension_core
make all
```

Files:
- `src/main.cu`: runtime `--m`, 2D grid launch, host validations.
- `src/kernel.cu`: stage-07 cp.async schedule plus `grid.y` M-tile ownership.
- `src/cp_async_staging.cuh`: A staging with `m_block_base`; guarded B staging unchanged.
- `src/mma_ptx.cuh`: MMA PTX helper (unchanged).

Constraints:
- `M % 16 == 0`
- `K % 16 == 0`
- `N % 8 == 0`
- Requires `sm_80+` for true `cp.async` behavior (default build target: `sm_89`).
