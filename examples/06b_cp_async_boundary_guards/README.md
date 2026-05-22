# Example 06b: cp.async Pipeline with Boundary Guards (`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`)

This stage keeps the Stage 06 double-buffered `cp.async` pipeline and adds tail-safe handling for non-32-aligned `N`.

## What Changed

- Matrix dimension change: default `N` changes from `32` (stage 06) to `40` so the second N block tile is partial.
- Kernel structure change: pipeline structure is unchanged, but guarded B-tile async staging is added.
- Boundary change #1: each 4-byte B `cp.async` copy is guarded with `global_col + 1 < N`; out-of-range shared slots are zero-filled.
- Boundary change #2: warp participation is guarded with `if (n_block_base + warp_tile_col < N)` on tail N block tiles.
- Helper function change: none in `mma_ptx.cuh`.
- Host/config change: constraints relax from `N % 32 == 0` to `N % 8 == 0`; stage id and binary updated.

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

Tail note: for each `n_block_base = 32` pass, only `warp0` participates; `warp1/2/3` are masked by guard.

### cp.async commit/wait semantics

- PTX ISA reference: <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async>
- `cp.async.commit_group` closes the current batch of issued `cp.async` copies so that batch becomes visible to a later wait.
- `cp.async.wait_group 0` blocks until all previously committed async-copy groups are done before MMA reads the shared tiles.
- In this stage, the sequence is unchanged from stage 06:
  - warmup: `commit_group` + `wait_group 0` for the first staged tile.
  - steady state: prefetch next tile, `commit_group`, compute current tile, then `wait_group 0` before buffer flip.

### Copy Granularity And Indexing

- This stage also uses `cp.async ... , 4`, so each async copy moves exactly `4` bytes.
- With `half` inputs (`2` bytes each), one copy moves `2` elements (`half2`-sized chunk).
- In code, `*_col_h2` means "column index in 2-half chunks":
  - `a_col_h2` / `b_col_h2` index 4-byte words inside a row.
  - `a_col = a_col_h2 * 2`, `b_col = b_col_h2 * 2` convert to half-element columns.
- Per-tile copy counts are unchanged from stage 06:
  - A tile (`16x16` half): `128` copies of 4 bytes.
  - B tile (`16x32` half): `256` copies of 4 bytes.

### Guarded Tail Semantics

- Guarded B staging:
  - check `global_col + 1 < N` before issuing `cp.async` for a 4-byte B word.
  - reason: one 4-byte copy writes two half elements (`col`, `col+1`), so both must be in range.
  - else path writes `0` into shared tile (zero-fill).
- Guarded warp participation:
  - `if (n_block_base + warp_tile_col < N)` gates MMA/store on partial N block tiles.
  - prevents out-of-range writes from warps whose 8-column subtile is outside N.

### Pipeline Timeline

- Warmup:
  - issue guarded async copies for `k_tile=0` into buffer `0`,
  - `commit_group`,
  - `wait_group 0`,
  - `__syncthreads`.
- Steady state (`k_iter = 0..k_tiles-1`):
  - optionally prefetch guarded `k_iter+1` tile into alternate buffer and `commit_group`,
  - run guarded MMA/store using current buffer,
  - `wait_group 0` (if next exists),
  - `__syncthreads`,
  - flip `curr/next` via ping-pong index.

## Why

- Keeps stage 06 focused on async pipeline mechanics first, then adds bounds safety as an isolated delta.
- Demonstrates how cp.async staging and warp ownership guards combine for practical non-32-aligned N.
- Preserves identical MMA math mapping, making correctness review easier.

## PTX Mapping Delta

- `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` is unchanged.
- Per-lane fragment formulas are unchanged.
- Delta is control/data validity only:
  - guarded B `cp.async` issue + zero-fill fallback,
  - guarded warp participation on N-tail block tiles.

## Run, Files, Constraints

Run:
```bash
cd examples/06b_cp_async_boundary_guards
make all
```

Files:
- `src/main.cu`: host I/O, relaxed N constraint (`N % 8 == 0`).
- `src/kernel.cu`: stage 06 pipeline schedule + guarded compute flow.
- `src/cp_async_staging.cuh`: cp.async primitives and the guarded K-tile staging helper.
- `src/mma_ptx.cuh`: unchanged MMA helper interface.

Constraints:
- `M = 16`
- `K % 16 == 0`
- `N % 8 == 0`
- Requires `sm_80+` for actual `cp.async` execution (default build target: `sm_89`).
