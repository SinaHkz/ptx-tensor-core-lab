# Example 05: Multi-Warp Block Tiling Core (`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`)

This stage introduces multi-warp block mapping first, without boundary checks, so the ownership/indexing model stays simple.

## What Changed

- Matrix dimension change: default output width is `N=32` so one block computes one full `16x32` C tile.
- Kernel structure change: launch uses `4` warps (`128` threads) per block, and each warp owns one `16x8` subtile (`n_tile={0,8,16,24}`).
- Shared-memory change: `sB` is widened from `16x8` (stage 04) to `16x32` and loaded cooperatively per `(k_tile, n_block)`.
- Synchronization change: `__syncwarp()` is replaced by `__syncthreads()` because shared-memory handoff spans multiple warps.
- Helper change: PTX helper now takes shared leading dimensions so B can be consumed from a wider block tile.
- Host-side change: launch changes from `<<<1,32>>>` to `<<<1,128>>>`.
- Boundary policy change: this stage intentionally has no in-kernel guards; instead host enforces `N % 32 == 0`.

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

## Why

- Keeps stage 05 focused on one idea: warp-to-output-tile mapping at block scope.
- Avoids mixing tail/guard logic into first exposure of multi-warp indexing.
- Sets up a clean follow-up (`05b`) where boundary checks are introduced as the only new concept.

## PTX Mapping Delta

- `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` is unchanged.
- Per-lane fragment formulas are unchanged.
- Mapping delta is base/stride context only:
  - A still uses a `16x16` shared tile (`lda=16`).
  - B now uses warp-specific base `&sB[0][warp_id*8]` inside a `16x32` shared tile (`ldb=32`).

## Run, Files, Constraints

Run:
```bash
cd examples/05_multiwarp_block_tiling
make all
```

Files:
- `src/main.cu`: host I/O, shape checks, `<<<1,128>>>` launch.
- `src/kernel.cu`: multi-warp block tiling without boundary guards.
- `src/mma_ptx.cuh`: PTX helper with shared-memory leading dimensions.

Constraints:
- `M = 16`
- `K % 16 == 0`
- `N % 32 == 0` (full block-tile coverage required in this stage)
