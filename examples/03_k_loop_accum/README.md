# Example 03: K-Loop Accumulation (Single Warp, PTX MMA)

This example is stage `03_k_loop_accum` from the roadmap.

It extends stage 02 with a K-tile accumulation loop. The same PTX instruction is used:

- `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`

The reduction dimension K is now split into tiles of 16, with fragment accumulation across tiles.

## Run

```bash
cd examples/03_k_loop_accum
make all
```

## Files

- `src/main.cu`: loads `A` as `16xK`, infers `N` from `B`, runs kernel.
- `src/kernel.cu`: one warp loops over K-tiles inside N-tiles with accumulation.
- `src/mma_ptx.cuh`: PTX fragment packing + inline `mma.sync` helper with `k_tile_base`.

## Constraints

- M = 16 (fixed by `m16n8k16`)
- K % 16 == 0
- N % 8 == 0
