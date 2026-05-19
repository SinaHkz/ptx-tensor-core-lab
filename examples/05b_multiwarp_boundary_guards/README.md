# Example 05b: Multi-Warp Boundary Guards (`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`)

This stage keeps Stage 05 block-tiling mapping unchanged and adds boundary handling for non-32-aligned `N`.

## What Changed

- Matrix dimension change: default `N` becomes `40` (from `32` in stage 05) so the last 32-column block tile is partial.
- Kernel structure change: same loop structure, but two guards are added:
  - guarded B load (`if (global_col < N)`) to zero-fill out-of-range shared entries,
  - guarded warp participation (`if (n_block_base + warp_tile_col < N)`) so invalid 8-column owners skip MMA/store.
- Helper function change: none (same helper and PTX instruction as stage 05).
- Host-side change: constraints relax from `N % 32 == 0` (stage 05) to `N % 8 == 0`.
- Config change: `example.id`/`learning.stage_id` set to `05b_multiwarp_boundary_guards`; binary renamed; default `B/C` width set to `40`.

## Execution path table

Default config: `A=16x64`, `B=64x40` (`K=64`, `N=40`).

| # | k_tile | n_tile | operation | accumulator |
|---|--------|--------|-----------|-------------|
| 1  | 0  | 0  | `warp0: C[:,0:7]   += A[:,0:15]  @ B[0:15,0:7]`   | `C += partial` |
| 2  | 0  | 8  | `warp1: C[:,8:15]  += A[:,0:15]  @ B[0:15,8:15]`  | `C += partial` |
| 3  | 0  | 16 | `warp2: C[:,16:23] += A[:,0:15]  @ B[0:15,16:23]` | `C += partial` |
| 4  | 0  | 24 | `warp3: C[:,24:31] += A[:,0:15]  @ B[0:15,24:31]` | `C += partial` |
| 5  | 0  | 32 | `warp0: C[:,32:39] += A[:,0:15]  @ B[0:15,32:39]` | `C += partial` |
| 6  | 16 | 0  | `warp0: C[:,0:7]   += A[:,16:31] @ B[16:31,0:7]`  | `C += partial` |
| 7  | 16 | 8  | `warp1: C[:,8:15]  += A[:,16:31] @ B[16:31,8:15]` | `C += partial` |
| 8  | 16 | 16 | `warp2: C[:,16:23] += A[:,16:31] @ B[16:31,16:23]`| `C += partial` |
| 9  | 16 | 24 | `warp3: C[:,24:31] += A[:,16:31] @ B[16:31,24:31]`| `C += partial` |
| 10 | 16 | 32 | `warp0: C[:,32:39] += A[:,16:31] @ B[16:31,32:39]`| `C += partial` |
| 11 | 32 | 0  | `warp0: C[:,0:7]   += A[:,32:47] @ B[32:47,0:7]`  | `C += partial` |
| 12 | 32 | 8  | `warp1: C[:,8:15]  += A[:,32:47] @ B[32:47,8:15]` | `C += partial` |
| 13 | 32 | 16 | `warp2: C[:,16:23] += A[:,32:47] @ B[32:47,16:23]`| `C += partial` |
| 14 | 32 | 24 | `warp3: C[:,24:31] += A[:,32:47] @ B[32:47,24:31]`| `C += partial` |
| 15 | 32 | 32 | `warp0: C[:,32:39] += A[:,32:47] @ B[32:47,32:39]`| `C += partial` |
| 16 | 48 | 0  | `warp0: C[:,0:7]   += A[:,48:63] @ B[48:63,0:7]`  | `C += partial` |
| 17 | 48 | 8  | `warp1: C[:,8:15]  += A[:,48:63] @ B[48:63,8:15]` | `C += partial` |
| 18 | 48 | 16 | `warp2: C[:,16:23] += A[:,48:63] @ B[48:63,16:23]`| `C += partial` |
| 19 | 48 | 24 | `warp3: C[:,24:31] += A[:,48:63] @ B[48:63,24:31]`| `C += partial` |
| 20 | 48 | 32 | `warp0: C[:,32:39] += A[:,48:63] @ B[48:63,32:39]`| `C += partial` |

Note: For each `n_block_base = 32` tail tile, `warp1/2/3` are skipped by guard.

## Why

- Keeps Stage 05 minimal for first-time understanding of block/warp ownership.
- Introduces boundary safety as a focused second step without changing PTX math mapping.
- Enables practical `N` values that are MMA-aligned (`%8`) but not block-tile-aligned (`%32`).

## PTX Mapping Delta

- `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` is unchanged.
- Per-lane fragment formulas are unchanged.
- New behavior is control-flow/data-validity only:
  - masked/zero-filled B entries for out-of-range columns,
  - per-warp MMA/store participation guard on tail block tiles.

## Run, Files, Constraints

Run:
```bash
cd examples/05b_multiwarp_boundary_guards
make all
```

Files:
- `src/main.cu`: host I/O, relaxed `N` constraint (`N % 8 == 0`).
- `src/kernel.cu`: stage 05 mapping plus explicit boundary guards.
- `src/mma_ptx.cuh`: unchanged helper from stage 05.

Constraints:
- `M = 16`
- `K % 16 == 0`
- `N % 8 == 0`
