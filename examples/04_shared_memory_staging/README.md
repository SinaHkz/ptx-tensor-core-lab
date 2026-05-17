# Example 04: Shared Memory Staging (Single Warp, PTX MMA)

This example is stage `04_shared_memory_staging` from the roadmap.

It extends stage 03 by staging A and B tiles in `__shared__` memory before
fragment packing. The same PTX instruction is used:

- `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`

## What Changed

- **Shared memory tiles** - kernel declares `__shared__ half sA[16][16]` and
  `__shared__ half sB[16][8]` matching one `m16n8k16` tile.
- **Cooperative staging** - all 32 lanes cooperatively load global-memory
  values into `sA` and `sB`; fragment pack reads from shared memory.
- **Loop order** - K-tile is outer loop, N-tile is inner loop. `sA` is loaded
  once per K-tile and reused across every N-tile in that K slice.
- **Output accumulation** - each MMA computes one `(k_tile, n_tile)` partial,
  then kernel does `C += d_tile` into global output.
- **Helper change** - PTX helper now takes shared pointers (`sA`, `sB`) and
  `lane`; no global-memory offset parameters.


## Cooperative Load Mapping

Both staging loops now use:

- `idx = i * 32 + lane`

Then:

- A tile: `r = idx / 16`, `c = idx % 16`, load `A[r * K + k_tile_base + c]`
- B tile: `r = idx / 8`, `c = idx % 8`, load `B[(k_tile_base + r) * N + (n_tile_base + c)]`

Memory coalescing is decided per warp instruction. At a fixed loop iteration
`i`, the expression `idx = i * 32 + lane` makes consecutive lanes produce
consecutive `idx` values. That gives each staging step a tighter, more
contiguous address footprint in global memory.

Example for A staging at fixed `i`:

- lane 0 -> idx 0, lane 1 -> idx 1, lane 2 -> idx 2, lane 3 -> idx 3

The total work is unchanged (same 256 A elements and 128 B elements staged per
tile), but the warp-level access pattern is more coalescing-friendly.

### Note on `__syncwarp()`

Even in a single warp, we still synchronize after staging `sA` and `sB`.
Reason: lockstep progress is not a portable correctness guarantee for
shared-memory handoffs (write -> read across lanes), and `__syncwarp()`
provides the required warp-level synchronization and memory-ordering guarantee.

### Execution path for the default config (A=16x64, B=64x16, so K=64, N=16)

With K=64 (4 K-tiles, `k_tile_base = 0, 16, 32, 48`) and N=16 (2 N-tiles,
`n_tile_base = 0, 8`), the kernel issues **8 MMA calls**:

| # | k_tile | n_tile | MMA input slices | accumulator |
|---|--------|--------|------------------|-------------|
| 1 | 0      | 0      | `A[:,0:15]` with `B[0:15,0:7]`   | `C[:,0:7] += A0@B0` |
| 2 | 0      | 8      | `A[:,0:15]` with `B[0:15,8:15]`  | `C[:,8:15] += A0@B1` |
| 3 | 16     | 0      | `A[:,16:31]` with `B[16:31,0:7]` | `C[:,0:7] += A1@B0` |
| 4 | 16     | 8      | `A[:,16:31]` with `B[16:31,8:15]`| `C[:,8:15] += A1@B1` |
| 5 | 32     | 0      | `A[:,32:47]` with `B[32:47,0:7]` | `C[:,0:7] += A2@B0` |
| 6 | 32     | 8      | `A[:,32:47]` with `B[32:47,8:15]`| `C[:,8:15] += A2@B1` |
| 7 | 48     | 0      | `A[:,48:63]` with `B[48:63,0:7]` | `C[:,0:7] += A3@B0` |
| 8 | 48     | 8      | `A[:,48:63]` with `B[48:63,8:15]`| `C[:,8:15] += A3@B1` |

## Why

- Shared staging introduces the memory hierarchy concept needed for later
  block-tiling and async-copy stages.
- K-outer / N-inner ordering creates explicit A-tile reuse across N tiles.
- Fragment mapping and PTX MMA stay unchanged, isolating the memory-path
  change from math-path change.
- Shared staging + cooperative lane mapping improves global-memory behavior
  while still keeping this stage simple (scalar per-lane loads).

## PTX Mapping Delta

- `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` is unchanged.
- Lane-to-fragment formulas are unchanged from stages 01-03.
- Data source changes only:
  - A: global `A[row * K + col]` -> staged tile `sA[r][c]` -> fragment loads.
  - B: global `B[row * N + col]` -> staged tile `sB[r][c]` -> fragment loads.

## Run

```bash
cd examples/04_shared_memory_staging
make all
```

## Files

- `src/main.cu`: loads `A/B`, infers `K/N`, launches kernel.
- `src/kernel.cu`: shared-memory tile staging, MMA calls, `C += d_tile`.
- `src/mma_ptx.cuh`: fragment pack from shared memory + inline PTX MMA.

## Constraints

- M = 16 (fixed by `m16n8k16`)
- K % 16 == 0
- N % 8 == 0
