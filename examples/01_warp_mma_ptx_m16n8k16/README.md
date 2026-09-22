# Example 01: Warp-Level PTX MMA (`m16n8k16`)

This first stage implements a complete `16x16` matrix multiplication with one
CUDA warp and inline PTX Tensor Core instructions. It is the smallest example
in the lab and focuses on the essential mechanics: distributing matrix
fragments across 32 lanes, packing FP16 operands into registers, issuing
`mma.sync`, and mapping the resulting FP32 fragments back to memory.

No shared memory, block tiling, asynchronous copies, or boundary handling is
used yet. Those features are introduced independently in later stages.

[Back to the project roadmap](../../README.md)

## What This Stage Demonstrates

- One CUDA block containing exactly one 32-thread warp
- Manual lane-to-fragment mapping for A, B, and C/D
- Inline PTX `mma.sync` rather than the CUDA WMMA C++ API
- FP16 A/B operands with FP32 accumulation and output
- Two `16x8` MMA operations combined into one `16x16` result
- Deterministic input generation and comparison with a CPU reference

## Operation

The example computes the row-major matrix product

```text
C[16x16] = A[16x16] x B[16x16]
```

using:

```ptx
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
```

The instruction suffix describes both the tile and operand types:

| Component | Meaning |
|---|---|
| `m16n8k16` | A is `16x16`, B is `16x8`, and C/D is `16x8` |
| `.row.col` | A and B use the PTX row/column fragment interpretations |
| first `.f32` | destination D elements are FP32 |
| first and second `.f16` | A and B elements are FP16 |
| final `.f32` | accumulator C elements are FP32 |

One instruction produces only a `16x8` output tile. The kernel therefore
issues the instruction twice:

```text
MMA 0: C[:,  0:8] = A[:, 0:16] x B[0:16,  0:8]
MMA 1: C[:, 8:16] = A[:, 0:16] x B[0:16, 8:16]
```

Both accumulator fragments start at zero, so each operation computes
`D = A x B + 0` for its half of the output.

## Requirements

- Linux
- Python 3
- GNU Make
- NVIDIA CUDA Toolkit with `nvcc`
- An NVIDIA GPU compatible with the default `sm_80` build target

To target a different compatible architecture, override `NVCCFLAGS`:

```bash
make NVCCFLAGS="-arch=sm_89" all
```

## Run the Example

From the repository root:

```bash
cd examples/01_warp_mma_ptx_m16n8k16
make all
```

The full workflow:

1. Compiles the CUDA sources into `tc_mma_ptx`.
2. Generates deterministic `16x16` inputs in `inputs/`.
3. Launches the kernel as `<<<1, 32>>>`.
4. Writes the GPU result to `outputs/C_gpu.txt`.
5. Computes `outputs/C_cpu.txt` and reports numerical differences.

Run individual steps with:

```bash
make help
make build
make gen-input
make run
make compare
make clean
```

## Code Flow

### 1. Host setup — [`src/main.cu`](src/main.cu)

The host code reads A and B as row-major `16x16` matrices, converts their
elements to CUDA `half`, allocates device memory, and launches one 32-thread
block. Because every output element is written by the kernel, the output buffer
does not require initialization.

### 2. Warp orchestration — [`src/kernel.cu`](src/kernel.cu)

The kernel derives each thread's lane identifiers, creates two four-element
FP32 accumulator fragments, and calls the MMA helper for output-column bases
`0` and `8`. It then applies the PTX C/D fragment mapping to store both `16x8`
tiles as one row-major `16x16` matrix.

### 3. Fragment construction and MMA — [`src/mma_ptx.cuh`](src/mma_ptx.cuh)

The helper selects the A and B values owned by each lane, packs pairs of FP16
values into 32-bit registers, issues the inline PTX instruction, and returns
four FP32 results per lane.

## Warp and Lane Mapping

`mma.sync` is a warp-level operation: all 32 lanes collectively supply the
operands and receive the result. For lane `0..31`, the code defines:

```text
group = lane >> 2   // 0..7
tid   = lane & 3    // 0..3 within the group
```

Each lane owns:

| Fragment | Values per lane | Register representation |
|---|---:|---|
| A | 8 FP16 values | 4 packed `f16x2` registers |
| B | 4 FP16 values | 2 packed `f16x2` registers |
| C/D | 4 FP32 values | 4 FP32 registers |

### A fragment

For the lane's A values:

| Elements | Row | Columns |
|---|---|---|
| `a0, a1` | `group` | `2*tid + {0,1}` |
| `a2, a3` | `group + 8` | `2*tid + {0,1}` |
| `a4, a5` | `group` | `2*tid + {8,9}` |
| `a6, a7` | `group + 8` | `2*tid + {8,9}` |

### B fragment

For a `16x8` output tile beginning at `n_tile_base`:

| Elements | Rows | Column |
|---|---|---|
| `b0, b1` | `2*tid + {0,1}` | `n_tile_base + group` |
| `b2, b3` | `2*tid + {8,9}` | `n_tile_base + group` |

B is stored row-major in global memory. The `.col` qualifier describes the
logical fragment interpretation expected by the MMA instruction; it does not
load or transpose memory automatically. The helper explicitly places each
logical B element into the correct lane register.

### C/D fragment and output store

Each lane receives four FP32 values for one `16x8` output tile:

| Elements | Row | Columns within the tile |
|---|---|---|
| `d0, d1` | `group` | `2*tid + {0,1}` |
| `d2, d3` | `group + 8` | `2*tid + {0,1}` |

Applying this mapping once with column base `0` and once with column base `8`
reconstructs all 256 elements of C without overlapping stores.

## Register Packing and Inline PTX

PTX expects each pair of FP16 inputs in a single 32-bit register. The helper's
`pack_half2()` function places the first value in bits `15:0` and the second in
bits `31:16`.

The inline assembly uses:

```cpp
: "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
: "r"(a_reg0), /* ... */, "r"(b_reg1)
```

- `"r"` supplies the packed 32-bit A/B registers.
- `"+f"` marks each FP32 accumulator as both an input and an output, matching
  the `D = A x B + C` read-modify-write behavior.
- `.sync.aligned` requires the participating warp lanes to execute the same MMA
  instruction together; divergent participation is invalid.

## Validation

Input generation is controlled by [`example.json`](example.json). The default
case uses a fixed random seed and values in `[-1, 1]`. The shared comparison
script computes a CPU matrix multiplication reference and reports:

- maximum absolute error
- mean absolute error
- root mean squared error
- the ten largest element-wise mismatches

Because the GPU path converts inputs to FP16 while the reference reads the
generated decimal values, small numerical differences are expected. This
stage's configuration reports the metrics but does not currently define
pass/fail thresholds.

## Common Failure Modes

- Launching fewer than 32 threads or allowing lanes to diverge around
  `mma.sync`
- Assuming one `m16n8k16` instruction produces a `16x16` result
- Confusing row-major memory storage with PTX fragment layout qualifiers
- Reversing the low/high FP16 values in a packed `f16x2` register
- Using output-only (`"=f"`) constraints instead of read/write (`"+f"`)
- Storing D fragments with ordinary row/column indexing instead of the PTX
  lane mapping

## Scope of This Stage

This example is intentionally correctness-first. Every lane loads its fragment
directly from global memory, the dimensions are fixed at `16x16`, and there is
only one warp. It establishes the instruction and mapping model used by all
later stages; it is not intended as a performance benchmark.

Continue with [Example 02: Parametric N Tiles](../02_parametric_n_tiles/) to
replace the two hard-coded output tiles with an N-tile loop.

## References

- [PTX ISA: Warp-level matrix instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Using Inline PTX Assembly in CUDA](https://docs.nvidia.com/cuda/inline-ptx-assembly/)
