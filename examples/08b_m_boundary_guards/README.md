# Example 08b: M-Boundary Guards (`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`)

This stage keeps stage 08 `grid.y` M-tiling and adds row-boundary handling so non-16-aligned `M` is safe.

## What Changed

- Matrix dimension change: default matrix uses `M=30` (`A=30x64`, `B=64x40`, `C=30x40`) so the second 16-row M tile is partial.
- Kernel structure change: same stage-08 `(blockIdx.x, blockIdx.y)` ownership, plus guarded C store (`if (global_row < M)`).
- Helper function change: A staging now guards row bounds and zero-fills shared entries for out-of-range rows in partial M tiles.
- Host-side change: runtime `--m` remains explicit, but `M % 16 == 0` requirement is removed.
- Config change: `example.id`/`learning.stage_id` set to `08b_m_boundary_guards`; default `matrix.a.rows`/`matrix.c.rows` set to `30`; binary renamed.

## Execution path table

Default config: `A=30x64`, `B=64x40` (`M=30`, `K=64`, `N=40`), so `grid=(2,2)`.

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
| 21 | (0,1) | 0  | 0  | `warp0: C[16:29,0:7]   += A[16:29,0:15]   @ B[0:15,0:7]`   | `C += partial` |
| 22 | (0,1) | 0  | 8  | `warp1: C[16:29,8:15]  += A[16:29,0:15]   @ B[0:15,8:15]`  | `C += partial` |
| 23 | (0,1) | 0  | 16 | `warp2: C[16:29,16:23] += A[16:29,0:15]   @ B[0:15,16:23]` | `C += partial` |
| 24 | (0,1) | 0  | 24 | `warp3: C[16:29,24:31] += A[16:29,0:15]   @ B[0:15,24:31]` | `C += partial` |
| 25 | (0,1) | 16 | 0  | `warp0: C[16:29,0:7]   += A[16:29,16:31]  @ B[16:31,0:7]`  | `C += partial` |
| 26 | (0,1) | 16 | 8  | `warp1: C[16:29,8:15]  += A[16:29,16:31]  @ B[16:31,8:15]` | `C += partial` |
| 27 | (0,1) | 16 | 16 | `warp2: C[16:29,16:23] += A[16:29,16:31]  @ B[16:31,16:23]`| `C += partial` |
| 28 | (0,1) | 16 | 24 | `warp3: C[16:29,24:31] += A[16:29,16:31]  @ B[16:31,24:31]`| `C += partial` |
| 29 | (0,1) | 32 | 0  | `warp0: C[16:29,0:7]   += A[16:29,32:47]  @ B[32:47,0:7]`  | `C += partial` |
| 30 | (0,1) | 32 | 8  | `warp1: C[16:29,8:15]  += A[16:29,32:47]  @ B[32:47,8:15]` | `C += partial` |
| 31 | (0,1) | 32 | 16 | `warp2: C[16:29,16:23] += A[16:29,32:47]  @ B[32:47,16:23]`| `C += partial` |
| 32 | (0,1) | 32 | 24 | `warp3: C[16:29,24:31] += A[16:29,32:47]  @ B[32:47,24:31]`| `C += partial` |
| 33 | (0,1) | 48 | 0  | `warp0: C[16:29,0:7]   += A[16:29,48:63]  @ B[48:63,0:7]`  | `C += partial` |
| 34 | (0,1) | 48 | 8  | `warp1: C[16:29,8:15]  += A[16:29,48:63]  @ B[48:63,8:15]` | `C += partial` |
| 35 | (0,1) | 48 | 16 | `warp2: C[16:29,16:23] += A[16:29,48:63]  @ B[48:63,16:23]`| `C += partial` |
| 36 | (0,1) | 48 | 24 | `warp3: C[16:29,24:31] += A[16:29,48:63]  @ B[48:63,24:31]`| `C += partial` |
| 37 | (1,1) | 0  | 32 | `warp0: C[16:29,32:39] += A[16:29,0:15]   @ B[0:15,32:39]` | `C += partial` |
| 38 | (1,1) | 16 | 32 | `warp0: C[16:29,32:39] += A[16:29,16:31]  @ B[16:31,32:39]`| `C += partial` |
| 39 | (1,1) | 32 | 32 | `warp0: C[16:29,32:39] += A[16:29,32:47]  @ B[32:47,32:39]`| `C += partial` |
| 40 | (1,1) | 48 | 32 | `warp0: C[16:29,32:39] += A[16:29,48:63]  @ B[48:63,32:39]`| `C += partial` |

Note: In block row `y=1`, rows `30:31` are out-of-range and are safely zero-filled/ignored by stage 08b guards.

## Why

- Stage 08 introduced M-axis tiling but still required `M % 16 == 0`.
- This stage isolates the boundary-safety upgrade for real-world row counts that are not tensor-tile aligned.
- It keeps the same PTX math path and cp.async schedule, so the only conceptual delta is data-validity handling.

## PTX Mapping Delta

- PTX MMA opcode is unchanged: `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`.
- Per-lane fragment formulas are unchanged.
- New behavior is guard logic only:
  - A staging adds `if (global_row < M)` else shared zero-fill.
  - C store adds `if (global_row < M)` guard.
  - `n_block_base` and N-tail guard formulas are unchanged from stage 08.

## Run, Files, Constraints

Run:
```bash
cd examples/08b_m_boundary_guards
make all
```

Files:
- `src/main.cu`: runtime `--m`, 2D grid launch with `ceil_div(M,16)`.
- `src/kernel.cu`: stage-08 ownership plus guarded stores on M tails.
- `src/cp_async_staging.cuh`: guarded A staging + guarded B staging.
- `src/mma_ptx.cuh`: MMA PTX helper (unchanged).

Constraints:
- `M > 0` (no `M % 16` requirement)
- `K % 16 == 0`
- `N % 8 == 0`
- Requires `sm_80+` for true `cp.async` behavior (default build target: `sm_89`).
