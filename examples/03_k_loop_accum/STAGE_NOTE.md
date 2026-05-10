# Stage Note: 03_k_loop_accum

## What Changed

- New snapshot example: `examples/03_k_loop_accum`.
- Kernel now loops over K in tiles of 16, accumulating across K-tiles into the same C fragment:
  - Outer loop over N-tiles (from stage 02).
  - Inner loop over K-tiles (new): `for k_tile_base in {0, 16, 32, ...}`.
  - C fragment initialized to zero before K-loop per N-tile.
- `mma_ptx.cuh` helper gained `k_tile_base` parameter to offset A column and B row indices.
- Host infers both K (from A size) and N (from B size), validates alignment.
- `example.json` updated to `A: 16x32`, `B: 32x16` (K=32, 2 K-tiles).

## Why

- Isolates K-tile accumulation as a distinct concept before adding shared memory (stage 04).
- Demonstrates correct fragment accumulation: the `"+f"` constraint in PTX `mma.sync` naturally accumulates `D += A*B` across K tiles.
- Without this stage, the jump from fixed K=16 to shared-memory-staged K would conflate two concepts.

## PTX Mapping Delta

- A fragment column indices shifted by `k_tile_base`: `A[row * lda + (k_tile_base + col_in_tile)]`.
- B fragment row indices shifted by `k_tile_base`: `B[(k_tile_base + row_in_tile) * ldb + col]`.
- Fragment formulas per lane are otherwise unchanged from stage 01/02.
- The `mma.sync` instruction itself is identical; only the input data slices change per K iteration.
