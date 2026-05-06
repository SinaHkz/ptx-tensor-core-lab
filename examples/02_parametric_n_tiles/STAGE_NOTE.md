# Stage Note: 02_parametric_n_tiles

## What Changed

- Added new snapshot example: `examples/02_parametric_n_tiles`.
- Kernel changed from two fixed MMA calls (columns `0..7`, `8..15`) to a loop over N tiles:
  - `for tile_base_col in {0, 8, 16, ...}`
  - one `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` per tile.
- Host side now infers `N` from `B.txt` length (`B` shape `16xN`) and writes `C` as `16xN`.
- Added stage metadata and validation targets in `example.json`:
  - `toolchain.sm`
  - `learning.stage_id`
  - `validation.max_abs`
  - `validation.mae`

## Why

- This stage isolates a single concept: expanding along output-N using repeated `m16n8k16` tiles while keeping warp and K behavior unchanged.
- It preserves the same PTX math path as stage 01 so diffs stay focused on indexing and tiling.

## PTX Mapping Delta

- Fragment formulas per lane are unchanged.
- The only mapping delta is `n_tile_base_col` becoming loop-driven instead of fixed constants (`0`, `8`).
- `B` fragment loads and `C` stores now use `tile_base_col` for each repeated N tile.
