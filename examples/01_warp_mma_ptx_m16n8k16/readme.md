# Example 01 Beginner Guide: Warp-level PTX MMA (m16n8k16)

This guide is dedicated to:

- `examples/01_warp_mma_ptx_m16n8k16/`

It combines quick-start beginner notes and the full beginner technical review for this specific example.

## Quick Start (This Example Only)

1. Move to this example:
   - `cd examples/01_warp_mma_ptx_m16n8k16`
2. Show commands:
   - `make help`
3. Run full flow (build + input generation + run + compare):
   - `make all`

Step-by-step commands are also available:

- `make build`
- `make gen-input`
- `make run`
- `make compare`
- `make clean`

Generated files for this example are stored only here:

- `examples/01_warp_mma_ptx_m16n8k16/inputs/`
- `examples/01_warp_mma_ptx_m16n8k16/outputs/`

## Beginner Glossary

- **Tensor Core**: specialized GPU hardware for fast matrix multiply-accumulate operations.
- **Warp**: a group of 32 threads that execute together.
- **Lane**: a thread's index inside a warp (`0..31`).
- **`laneid`**: PTX special register containing the current lane index.
- **Fragment**: the per-thread slice of a matrix tile assigned to one lane for MMA.
- **MMA tile shape (`m16n8k16`)**:
  - `m`: output rows (`16`)
  - `n`: output cols (`8`)
  - `k`: reduction dimension (`16`)
- **`mma.sync`**: warp-scope matrix multiply-accumulate instruction (`D = A * B + C`).
- **`.row.col`**: layout interpretation for A/B fragments in the instruction.
- **Accumulator**: existing output values (`C`) that get added into new results.
- **Register packing (`f16x2`)**: two `half` values packed into one 32-bit register.
- **`group` / `tid` in this example**:
  - `group = lane >> 2` (8 groups total)
  - `tid = lane & 3` (lane position inside each 4-thread group)

## About This Example's Mapping

This example uses:

- `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`

That means:

- A and B are `fp16`
- accumulation/output is `fp32`
- one MMA call produces a `16x8` output tile

To produce a full `16x16` output, the kernel executes two MMA tiles:

- columns `0..7`
- columns `8..15`

## What This Code Is Doing

At a high level:

1. `src/main.cu` reads `inputs/A.txt` and `inputs/B.txt` as 16x16 matrices (stored row-major).
2. It copies them to the GPU as `half` (`fp16`) values.
3. `src/kernel.cu` launches exactly one warp (`32` threads).
4. That warp executes PTX Tensor Core MMA instructions from `src/mma_ptx.cuh`.
5. The output is written as a full 16x16 `float` (`fp32`) matrix to `outputs/C_gpu.txt`.

The MMA instruction used is:

- `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`

This means:

- Tile shape: `m=16, n=8, k=16`
- Data types: `A=f16`, `B=f16`, accumulate/output in `f32`
- Mathematical form per instruction: `D = A * B + C`

Because the kernel needs a 16x16 final result, it runs two `m16n8k16` tiles:

- First tile for output columns `0..7`
- Second tile for output columns `8..15`

## Warp, Lane, Group: Core Mental Model

Tensor Core `mma.sync` instructions are warp-scope operations. A warp is 32 threads.

In this code (`src/kernel.cu`):

- `lane = threadIdx.x & 31`
- `group = lane >> 2`
- `tid = lane & 3`

Interpretation:

- `lane`: thread index inside warp (`0..31`)
- `group`: one of 8 groups (`0..7`), each group contains 4 lanes
- `tid`: position inside 4-lane group (`0..3`)

Why this matters: the PTX spec defines fragment formulas using `%laneid`. So every lane owns specific matrix elements.

## Fragmentation: What Each Thread Owns

For `mma.m16n8k16` with `f16` inputs and `f32` accumulators:

- Each lane provides:
  - `A` fragment: 8 fp16 values (`a0..a7`) packed into 4 registers (`f16x2`)
  - `B` fragment: 4 fp16 values (`b0..b3`) packed into 2 registers (`f16x2`)
  - `C` fragment: 4 fp32 values (`c0..c3`)
- Each lane receives:
  - `D` fragment: 4 fp32 values (`d0..d3`)

That is why the inline PTX has:

- 4 A registers
- 2 B registers
- 4 accumulator/output registers

### Formula View (From `%laneid`)

For this instruction family, the PTX docs define:

- `groupID = lane >> 2`
- `threadID_in_group = lane % 4`

For A fragment elements `ai` (`i=0..7`) in row-major A:

- Row:
  - `groupID` for `i in {0,1,4,5}`
  - `groupID + 8` for `i in {2,3,6,7}`
- Col:
  - `(threadID_in_group * 2) + (i & 1)` for `i < 4`
  - `(threadID_in_group * 2) + (i & 1) + 8` for `i >= 4`

For B fragment elements `bi` (`i=0..3`) in column-major B:

- Row:
  - `(threadID_in_group * 2) + (i & 1)` for `i < 2`
  - `(threadID_in_group * 2) + (i & 1) + 8` for `i >= 2`
- Col:
  - `groupID`

For C/D fragment elements `ci/di` (`i=0..3`) with fp32 accumulators:

- Row:
  - `groupID` for `i < 2`
  - `groupID + 8` for `i >= 2`
- Col:
  - `(threadID_in_group * 2) + (i & 1)`

## How Spec Formulas Map Into This Code

### A fragment load (`src/mma_ptx.cuh`)

The code computes `a0..a7` using `group` and `tid`:

- Rows are split between `group` and `group + 8`
- Columns use `tid*2 + {0,1}` and then `+8` for the high-k half

This matches PTX fragment rules for `m16n8k16` row-major A layout.

### B fragment load (`src/mma_ptx.cuh`)

The code computes `b0..b3` using:

- Row positions driven by `tid*2 + {0,1}` and then `+8`
- Column position driven by `group` (+ tile base column 0 or 8)

This matches PTX fragment rules for `m16n8k16` column-major B layout in the instruction.

### Packing fp16 pairs

PTX expects packed `f16x2` in 32-bit registers for A/B operands.

`pack_half2()` converts two half values into one `uint32_t`:

- low 16 bits = first half
- high 16 bits = second half

### Running MMA

Inline asm:

- `"+f"` on outputs (`d0..d3`) means read-modify-write (`D = A*B + D`)
- `"r"` on A/B means pass packed 32-bit registers

Important detail:

- `.row.col` in `mma.sync...row.col...` sets logical layout interpretation for A and B fragments.
- It does not auto-load matrices from memory.
- You still place correct values in correct lane registers manually.

## Why `lane`/`group` Store Mapping Works

After MMA, each lane has 4 fp32 outputs for a 16x8 tile.

In `src/kernel.cu`, for each `i in 0..3`:

- `row = group` for `i < 2`, else `group + 8`
- `col_in_tile = tid*2 + (i & 1)`

Then it stores:

- tile0 -> columns `0..7`
- tile1 -> columns `8..15`

This reconstructs a row-major 16x16 output matrix in `C`.

## Important Beginner Pitfalls

### Pitfall A: Treating MMA as single-thread

`mma.sync` is warp-scope. All lanes must participate correctly.

### Pitfall B: Wrong instruction vs data type

This example uses floating-point MMA (`f16,f16 -> f32`), not integer MMA.

### Pitfall C: Wrong tile shape assumption

`m16n8k16` computes 16x8. Full 16x16 requires two tiles.

### Pitfall D: Wrong fragment mapping

Most wrong answers come from lane-to-element mapping mistakes.

### Pitfall E: Wrong asm constraints

Using `"=f"` instead of `"+f"` changes accumulator semantics.

### Pitfall F: PTX MMA vs WMMA confusion

This example is manual PTX MMA (explicit mapping/packing).

### Pitfall G: Performance optimization too early

This example is correctness-first and educational.

## Quick Mapping Cheat Sheet

- Warp size: 32
- Lane: `lane = threadIdx.x & 31`
- Group of 4 lanes: `group = lane >> 2`
- Lane inside group: `tid = lane & 3`
- MMA instruction: `m16n8k16`
- One MMA output shape: `16 x 8`
- Two MMA calls needed for `16 x 16`

## File-by-File Notes

### `src/main.cu`

- Handles I/O and launch.
- Uses one warp launch (`<<<1,32>>>`).
- Reads from `inputs/`, writes to `outputs/`.

### `src/kernel.cu`

- Orchestrates tile-level MMA calls.
- Converts lane-local fragments into global output indices.

### `src/mma_ptx.cuh`

- Performs fragment extraction from A/B.
- Packs fp16 pairs.
- Issues inline PTX MMA instruction.

## What To Learn Next

1. Validate this kernel on deterministic inputs (all ones, identity, small integers).
2. Move A/B into shared memory and load fragments from shared.
3. Add a K-loop (multiple `k` tiles) for larger GEMM.
4. Compare this inline PTX path against `nvcuda::wmma` API.

## Practical Debug Checklist

1. Does instruction type match data types?
2. Are all 32 lanes participating?
3. Are `%laneid` formulas exact for A, B, and C/D?
4. Are f16 values packed in correct low/high 16-bit order?
5. Are asm constraints correct (`"r"` inputs, `"+f"` accumulators)?
6. Is store mapping back to global matrix coordinates correct?
7. Are you comparing the same shape/layout on CPU?

## External References

1. PTX ISA (matrix fragment rules and MMA forms):  
https://docs.nvidia.com/cuda/archive/11.0/parallel-thread-execution/index.html

2. CUDA Programming Guide (warps, lanes, SIMT):  
https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html

3. Inline PTX constraints (`"r"`, `"f"`, `"+f"`):  
https://docs.nvidia.com/cuda/archive/13.0.2/inline-ptx-assembly/index.html

4. WMMA fragment caveat (layout unspecified/architecture dependent):  
https://docs.nvidia.com/cuda/archive/13.1.1/cuda-c-programming-guide/05-appendices/cpp-language-extensions.html
