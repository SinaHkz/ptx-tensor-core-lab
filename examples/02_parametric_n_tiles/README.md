# Example 02: Parametric N Tiles (Single Warp, PTX MMA)

This example is stage `02_parametric_n_tiles` from the roadmap.

It keeps the same warp-level PTX instruction from example 01:

- `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`

The change is that output width `N` is no longer hardcoded to `16`. The kernel now
iterates over `N` in `8`-column tiles (`m16n8k16`) while still using a single warp.

Comment policy for this stage (and future stages):

- remove stale comments copied from previous stages,
- keep comments only for the current stage's changes and updates.

## Run

```bash
cd examples/02_parametric_n_tiles
make all
```

## Files

- `src/main.cu`: loads `A` as `16x16`, infers `N` from `B` (`16xN`), runs kernel.
- `src/kernel.cu`: one warp loops over `tile_base_col = 0, 8, 16, ...`.
- `src/mma_ptx.cuh`: PTX fragment packing + inline `mma.sync` helper.

## Constraints

- Stage 02 assumes `K=16` and requires `N % 8 == 0`.
- Default config uses `N=32` (`4` N-tiles).
