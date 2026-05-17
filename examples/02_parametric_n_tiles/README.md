# Example 02: Parametric N Tiles (Single Warp, PTX MMA)

This example is stage `02_parametric_n_tiles` from the roadmap.

It keeps the same warp-level PTX instruction from example 01:

- `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`

The change is that output width `N` is no longer hardcoded to `16`. The
kernel now iterates over `N` in `8`-column tiles (`m16n8k16`) while still
using a single warp.

---

## What Changed

- **Parametric N** — N is no longer fixed at 16. The kernel uses a loop over
  `tile_base_col` in `{0, 8, 16, ...}` instead of two hard-coded MMA calls
  (columns 0..7, 8..15).
- **Dynamic N inference** — host reads B.txt until EOF, computes
  `N = B_size / K`, validates `N % 8 == 0`.
- **`n_tile_base_col` parameter** — the MMA helper now takes a column offset
  to select the current 8-wide N-tile.
- **`ldb = N`** — the B stride is now dynamic instead of hardcoded `16`.
- **`example.json` metadata** — added `toolchain.sm`, `learning.stage_id`,
  `validation.max_abs`, `validation.mae`.

### Execution path for the default config (A=16×16, B=16×32, so K=16, N=32)

With N=32 (4 N-tiles, n_tile_base = 0, 8, 16, 24) and K=16 (1 K-tile),
the kernel issues **4 MMA calls**:

| # | n_tile | k_tile | operation |
|---|--------|--------|-----------|
| 1 | 0      | 0      | `C[:, 0:7]  += A[:, 0:15] @ B[0:15, 0:7]` |
| 2 | 8      | 0      | `C[:, 8:15] += A[:, 0:15] @ B[0:15, 8:15]` |
| 3 | 16     | 0      | `C[:, 16:23] += A[:, 0:15] @ B[0:15, 16:23]` |
| 4 | 24     | 0      | `C[:, 24:31] += A[:, 0:15] @ B[0:15, 24:31]` |

Each row is one `mma.sync.aligned.m16n8k16` instruction covering a 16×8
output sub-tile. K remains 16 (single K-tile), so no accumulation across
K is needed — each MMA produces the final value for its N-tile directly.

---

## Why

- Isolates parametric N tiling as a distinct concept before adding K-loop
  accumulation (stage 03) and shared memory (stage 04).
- Preserves the same PTX math path as stage 01 so diffs stay focused on
  indexing and tiling.
- Without this stage, the jump from two hard-coded N tiles to a K+N tiled
  design would conflate two independent tiling dimensions.

---

## PTX Mapping Delta

- Fragment formulas per lane are unchanged from stage 01.
- The only mapping delta is `n_tile_base_col` becoming loop-driven instead
  of fixed constants (`0`, `8`).
- B fragment loads and C stores use `tile_base_col` for each repeated N tile.
- Stride `ldb` is now `N` (dynamic) instead of `16` (hardcoded).

---

## Run

```bash
cd examples/02_parametric_n_tiles
make all
```

## Files

- `src/main.cu`: loads `A` as `16×16`, infers `N` from `B` (`16×N`), runs kernel.
- `src/kernel.cu`: one warp loops over `tile_base_col = 0, 8, 16, ...`.
- `src/mma_ptx.cuh`: PTX fragment packing + inline `mma.sync` helper.

## Constraints

- K = 16 (fixed by `m16n8k16`, no K-loop yet — comes in stage 03)
- N % 8 == 0
- Default config: N = 32 (4 N-tiles)
