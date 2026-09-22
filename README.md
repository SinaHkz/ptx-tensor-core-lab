# PTX Tensor Core Lab

A progressive CUDA C++ and PTX learning lab for understanding how NVIDIA
Tensor Core matrix multiplication kernels are built from the instruction level
up. The repository starts with a single warp issuing an inline PTX
`mma.sync.aligned.m16n8k16` instruction and incrementally adds K/N tiling,
shared-memory staging, multi-warp cooperation, boundary guards,
double-buffered `cp.async`, and two-dimensional multi-CTA decomposition.

The examples use FP16 inputs and FP32 accumulation. Each stage is intentionally
small and documents only the concept introduced at that step, making the
repository useful for studying lane-to-fragment mapping and the evolution of a
Tensor Core GEMM kernel without hiding the mechanics behind a library API.

> This is an educational kernel-development lab, not a replacement for cuBLAS.
> Its current focus is correctness, mapping, and memory-pipeline mechanics—not
> production-level performance or a cuBLAS speedup claim.

## Highlights

- Inline PTX `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`
- Manual warp-lane to MMA-fragment mapping and register packing
- Parametric N tiling and K-loop accumulation
- Cooperative global-to-shared-memory staging
- Four-warp `16x32` CTA output tiles
- Boundary-safe M and N tiles through zero-fill and guarded stores
- Double-buffered `cp.async` global-to-shared-memory pipeline
- Two-dimensional CTA ownership through `grid.x` and `grid.y`
- Deterministic input generation and CPU-reference validation
- Configurable maximum-absolute-error and mean-absolute-error thresholds

## Learning Path

The repository contains 11 runnable examples. Lettered stages isolate boundary
handling from the core mapping or pipeline change introduced immediately before
them.

| Example | Main concept |
|---|---|
| [01 — Warp MMA](examples/01_warp_mma_ptx_m16n8k16/) | One warp computes a `16x16` result using inline PTX `m16n8k16` MMA instructions. |
| [02 — Parametric N tiles](examples/02_parametric_n_tiles/) | A single warp iterates over an output width composed of 8-column MMA tiles. |
| [03 — K-loop accumulation](examples/03_k_loop_accum/) | Multiple 16-wide K tiles accumulate into the output fragments. |
| [04 — Shared-memory staging](examples/04_shared_memory_staging/) | A and B tiles are cooperatively staged in shared memory before MMA. |
| [05 — Multi-warp block tiling](examples/05_multiwarp_block_tiling/) | Four warps cooperate on a `16x32` CTA output tile. |
| [05b — N boundary guards](examples/05b_multiwarp_boundary_guards/) | Partial 32-column block tiles are zero-filled and invalid warp owners are masked. |
| [06 — `cp.async` pipeline](examples/06_cp_async_pipeline/) | A two-stage ping-pong pipeline overlaps asynchronous staging with MMA work. |
| [06b — Guarded `cp.async`](examples/06b_cp_async_boundary_guards/) | The asynchronous pipeline safely handles partial N block tiles. |
| [07 — Multi-block N tiling](examples/07_multiblock_n_tiles_cp_async/) | `grid.x` assigns one 32-column output tile to each CTA. |
| [08 — M-dimension tiling](examples/08_m_dimension_core/) | `grid.y` extends CTA ownership across 16-row M tiles. |
| [08b — M boundary guards](examples/08b_m_boundary_guards/) | Partial M tiles are zero-filled on load and guarded on store. |

## Current Endpoint

Stage 08b computes a row-major matrix product

```text
C[M x N] = A[M x K] x B[K x N]
```

with the following current constraints:

- `M > 0`
- `K` must be a multiple of 16
- `N` must be a multiple of 8
- A CTA computes one `16x32` output tile using four warps
- M-tail and N-block-tail accesses are boundary-safe

The default Stage 08b case is `A[30x64] x B[64x40] -> C[30x40]`. It exercises
both a partial 16-row M tile and a partial 32-column N block tile.

## Requirements

- Linux
- Python 3
- GNU Make
- NVIDIA CUDA Toolkit with `nvcc`
- An NVIDIA GPU supporting the selected target architecture

The Makefiles target `sm_89` by default. The latest pipeline stages use
`cp.async`, which requires `sm_80` or newer. Override the target when building
for another supported GPU, for example:

```bash
make NVCCFLAGS="-arch=sm_80" all
```

## Quick Start

Run the latest boundary-safe M/N example:

```bash
git clone https://github.com/SinaHkz/ptx-tensor-core-lab.git
cd ptx-tensor-core-lab/examples/08b_m_boundary_guards
make all
```

`make all` performs the complete workflow:

1. Builds the CUDA executable.
2. Generates deterministic FP16 input matrices from `example.json`.
3. Runs the Tensor Core kernel.
4. Computes a CPU reference result.
5. Reports maximum absolute error, mean absolute error, RMSE, and the worst
   element-wise mismatches.
6. Returns a non-zero status if a configured validation threshold is exceeded.

Individual targets are also available:

```bash
make help
make build
make gen-input
make run
make compare
make clean
```

## Validation

Every example contains an `example.json` file that defines its matrix shapes,
input seed, paths, target architecture metadata, and numerical tolerances. The
shared scripts under [`scripts/`](scripts/) generate inputs and compare GPU
output against a row-major CPU matrix multiplication reference.

For Stage 08b, validation currently requires:

```json
{
  "max_abs": 0.02,
  "mae": 0.004
}
```

These tolerances account for FP16 input quantization and FP32 Tensor Core
accumulation. A successful run prints `[PASS] Validation thresholds satisfied.`

## Repository Structure

```text
ptx-tensor-core-lab/
|-- examples/
|   |-- 01_warp_mma_ptx_m16n8k16/
|   |-- ...
|   `-- 08b_m_boundary_guards/
|       |-- README.md
|       |-- example.json
|       |-- Makefile
|       `-- src/
|           |-- main.cu
|           |-- kernel.cu
|           |-- mma_ptx.cuh
|           `-- cp_async_staging.cuh
|-- scripts/
|   |-- generate_input.py
|   `-- compare_gpu_output.py
`-- README.md
```

Each example is self-contained and has its own README describing the execution
path, PTX mapping delta, constraints, and the exact concept introduced by that
stage.

## Design Notes

- A and B are stored in row-major memory. The `.row.col` PTX qualifier
  describes MMA fragment interpretation; values are still manually placed in
  the correct per-lane registers.
- The `m16n8k16` instruction produces a `16x8` output tile. Four warps therefore
  cover the four 8-column subtiles of a `16x32` CTA tile.
- Boundary variants deliberately follow their unguarded core stages so mapping
  changes and data-validity logic can be studied independently.
- K tails are not implemented yet; K must remain aligned to the instruction's
  16-element reduction tile.

## Scope and Next Steps

Possible extensions include K-tail handling, register-resident accumulation
across the full K loop, wider asynchronous copies, shared-memory layout and bank
conflict experiments, occupancy analysis, Nsight Compute profiling, and fair
comparisons with cuBLAS or CUTLASS.
