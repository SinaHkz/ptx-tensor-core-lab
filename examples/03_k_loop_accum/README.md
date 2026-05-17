# Example 03: K-Loop Accumulation (Single Warp, PTX MMA)

This example is stage `03_k_loop_accum` from the roadmap.

It extends stage 02 by making the reduction dimension K variable. The same
PTX instruction is used:

- `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`

---

## What Changed

- **Matrix dimensions** — K is no longer fixed at 16. The default config
  uses **A: 16×32, B: 32×16** (vs previous stages where A was 16×16 and K
  was hard-coded). Any K that is a multiple of 16 is now supported.
- **K-tile loop** — the kernel adds an inner loop over `k_tile_base` in
  `{0, 16, 32, ...}`. Each iteration issues one `m16n8k16` MMA call on a
  16-wide slice of A and B, accumulating into the same C fragment.
- `mma_ptx.cuh` helper gained `k_tile_base` parameter to offset A column
  and B row indices.
- Host infers both K (from A size) and N (from B size), validates alignment.

### Execution path for the default config (A=16×32, B=32×16, so K=32, N=16)

With N=16 (2 N-tiles, n_tile_base = 0, 8) and K=32 (2 K-tiles,
k_tile_base = 0, 16), the kernel issues **4 MMA calls**:

| # | n_tile | k_tile | MMA input slices | accumulator |
|---|--------|--------|------------------|-------------|
| 1 | 0      | 0      | `A[:,0:15]`, `B[0:15,0:7]`    | `d_tile = A₀·B₀` |
| 2 | 0      | 16     | `A[:,16:31]`, `B[16:31,0:7]`  | `d_tile += A₁·B₁` |
|   |        |        | → store to `C[:,0:7]`         | `C[:,0:7] = A₀·B₀ + A₁·B₁` |
| 3 | 8      | 0      | `A[:,0:15]`, `B[0:15,8:15]`   | `d_tile = A₀·B₀` |
| 4 | 8      | 16     | `A[:,16:31]`, `B[16:31,8:15]` | `d_tile += A₁·B₁` |
|   |        |        | → store to `C[:,8:15]`        | `C[:,8:15] = A₀·B₀ + A₁·B₁` |

The `"+f"` constraint in the PTX `mma.sync` makes calls 2 and 4 true
read-modify-writes: `d_tile = A₁·B₁ + d_tile` (where `d_tile` already holds
`A₀·B₀` from the previous call).

---

## Why

- Isolates K-tile accumulation as a distinct concept before adding shared
  memory (stage 04).
- Without this stage, the jump from fixed K=16 to shared-memory-staged K
  would conflate two concepts: K tiling and memory staging.
- Demonstrates correct fragment accumulation through the hardware `D += A*B`
  semantics — the same MMA instruction that computes the product also handles
  the accumulation across K tiles at no extra cost.

---

## PTX Mapping Delta

- A fragment column indices shifted by `k_tile_base`:
  `A[row * lda + (k_tile_base + col_in_tile)]`.
- B fragment row indices shifted by `k_tile_base`:
  `B[(k_tile_base + row_in_tile) * ldb + col]`.
- Fragment formulas per lane are otherwise unchanged from stage 01/02.
- The `mma.sync` instruction is identical; only the input data slices change
  per K iteration.

---

## Run

```bash
cd examples/03_k_loop_accum
make all
```

## Files

- `src/main.cu`: loads `A` as `16×K`, infers `N` from `B`, runs kernel.
- `src/kernel.cu`: one warp, outer N-tile loop, inner K-tile loop with accumulation.
- `src/mma_ptx.cuh`: PTX fragment packing + inline `mma.sync` helper with `k_tile_base`.

## Constraints

- M = 16 (fixed by `m16n8k16`)
- K % 16 == 0
- N % 8 == 0
