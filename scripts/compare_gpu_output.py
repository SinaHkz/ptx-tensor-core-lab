#!/usr/bin/env python3
"""Write CPU A x B result and compare against GPU output using example config."""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXAMPLE = "examples/01_warp_mma_ptx_m16n8k16"
DEFAULT_CONFIG_NAME = "example.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write CPU A x B result to file, then compare GPU vs CPU."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to example JSON config (default: <example-root>/example.json)",
    )
    parser.add_argument(
        "--example-root",
        default=None,
        help="Example root directory (defaults to config directory when --config is set)",
    )
    parser.add_argument("--a", default=None, help="Path to A input file")
    parser.add_argument("--b", default=None, help="Path to B input file")
    parser.add_argument("--gpu", default=None, help="Path to GPU output file")
    parser.add_argument("--cpu-out", default=None, help="Path to write CPU output file")
    parser.add_argument("--m", type=int, default=None, help="Rows of A")
    parser.add_argument("--k", type=int, default=None, help="Cols of A / rows of B")
    parser.add_argument("--n", type=int, default=None, help="Cols of B")
    parser.add_argument("--size", type=int, default=None, help="Legacy square size override (N => m=k=n=N)")
    parser.add_argument("--topk", type=int, default=None, help="How many worst mismatches to print")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_roots(args: argparse.Namespace) -> Tuple[Path, Path]:
    if args.config:
        cfg_path = Path(args.config).resolve()
        example_root = Path(args.example_root).resolve() if args.example_root else cfg_path.parent
    else:
        example_root = (
            Path(args.example_root).resolve()
            if args.example_root
            else (PROJECT_ROOT / DEFAULT_EXAMPLE).resolve()
        )
        cfg_path = example_root / DEFAULT_CONFIG_NAME
    return example_root, cfg_path


def positive_int(name: str, value: int) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def maybe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def resolve_dims_from_matrix_cfg(matrix: Dict[str, Any]) -> Tuple[int, int, int]:
    legacy_m = maybe_int(matrix.get("m"))
    legacy_k = maybe_int(matrix.get("k"))
    legacy_n = maybe_int(matrix.get("n"))

    a = matrix.get("a", {})
    b = matrix.get("b", {})
    c = matrix.get("c", {})

    a_rows = maybe_int(a.get("rows"))
    a_cols = maybe_int(a.get("cols"))
    b_rows = maybe_int(b.get("rows"))
    b_cols = maybe_int(b.get("cols"))
    c_rows = maybe_int(c.get("rows"))
    c_cols = maybe_int(c.get("cols"))

    m = legacy_m if legacy_m is not None else (a_rows if a_rows is not None else (c_rows if c_rows is not None else 16))
    k = legacy_k if legacy_k is not None else (a_cols if a_cols is not None else (b_rows if b_rows is not None else 16))
    n = legacy_n if legacy_n is not None else (b_cols if b_cols is not None else (c_cols if c_cols is not None else 16))

    m = positive_int("m", m)
    k = positive_int("k", k)
    n = positive_int("n", n)

    if a_rows is not None and a_rows != m:
        raise ValueError(f"Dimension mismatch: matrix.a.rows ({a_rows}) != m ({m})")
    if a_cols is not None and a_cols != k:
        raise ValueError(f"Dimension mismatch: matrix.a.cols ({a_cols}) != k ({k})")
    if b_rows is not None and b_rows != k:
        raise ValueError(f"Dimension mismatch: matrix.b.rows ({b_rows}) != k ({k})")
    if b_cols is not None and b_cols != n:
        raise ValueError(f"Dimension mismatch: matrix.b.cols ({b_cols}) != n ({n})")
    if c_rows is not None and c_rows != m:
        raise ValueError(f"Dimension mismatch: matrix.c.rows ({c_rows}) != m ({m})")
    if c_cols is not None and c_cols != n:
        raise ValueError(f"Dimension mismatch: matrix.c.cols ({c_cols}) != n ({n})")

    return m, k, n


def read_floats(path: str) -> List[float]:
    with open(path, "r", encoding="utf-8") as f:
        data = f.read().strip().split()
    return [float(x) for x in data]


def write_floats(path: str, values: List[float]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(" ".join(f"{v:.8f}" for v in values))
        f.write("\n")


def display_path(path: str, example_root: Path) -> str:
    resolved = Path(path).resolve()
    root = example_root.resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return path


def matmul_row_major(a: List[float], b: List[float], m: int, k: int, n: int) -> List[float]:
    c = [0.0] * (m * n)
    for i in range(m):
        for j in range(n):
            s = 0.0
            for t in range(k):
                s += a[i * k + t] * b[t * n + j]
            c[i * n + j] = s
    return c


def compare_vectors(gpu: List[float], ref: List[float]) -> Tuple[float, float, float]:
    count = min(len(gpu), len(ref))
    if count == 0:
        return 0.0, 0.0, 0.0

    max_abs = 0.0
    mse = 0.0
    mae = 0.0
    for i in range(count):
        e = abs(gpu[i] - ref[i])
        mae += e
        mse += e * e
        if e > max_abs:
            max_abs = e
    mae /= count
    rmse = math.sqrt(mse / count)
    return max_abs, mae, rmse


def worst_k(
    gpu: List[float], ref: List[float], n_cols: int, topk: int
) -> List[Tuple[float, int, int, float, float]]:
    count = min(len(gpu), len(ref))
    rows = []
    for idx in range(count):
        err = abs(gpu[idx] - ref[idx])
        r = idx // n_cols
        c = idx % n_cols
        rows.append((err, r, c, gpu[idx], ref[idx]))
    rows.sort(key=lambda x: x[0], reverse=True)
    return rows[:topk]


def resolve_dims(args: argparse.Namespace, cfg: Dict[str, Any]) -> Tuple[int, int, int]:
    if args.size is not None:
        dim = positive_int("size", args.size)
        return dim, dim, dim

    m_cfg, k_cfg, n_cfg = resolve_dims_from_matrix_cfg(cfg.get("matrix", {}))
    m = args.m if args.m is not None else m_cfg
    k = args.k if args.k is not None else k_cfg
    n = args.n if args.n is not None else n_cfg

    return positive_int("m", m), positive_int("k", k), positive_int("n", n)


def resolve_paths(
    args: argparse.Namespace, cfg: Dict[str, Any], example_root: Path
) -> Tuple[str, str, str, str]:
    paths = cfg.get("paths", {})

    a_path = str(Path(args.a)) if args.a else str(example_root / paths.get("input_a", "inputs/A.txt"))
    b_path = str(Path(args.b)) if args.b else str(example_root / paths.get("input_b", "inputs/B.txt"))
    gpu_path = (
        str(Path(args.gpu)) if args.gpu else str(example_root / paths.get("output_gpu", "outputs/C_gpu.txt"))
    )
    cpu_out_path = (
        str(Path(args.cpu_out))
        if args.cpu_out
        else str(example_root / paths.get("output_cpu", "outputs/C_cpu.txt"))
    )
    return a_path, b_path, gpu_path, cpu_out_path


def resolve_topk(args: argparse.Namespace, cfg: Dict[str, Any]) -> int:
    compare_cfg = cfg.get("compare", {})
    topk = args.topk if args.topk is not None else int(compare_cfg.get("topk", 10))
    return positive_int("topk", topk)


def main() -> int:
    args = parse_args()
    example_root, cfg_path = resolve_roots(args)
    cfg = load_config(cfg_path)

    m, k, n = resolve_dims(args, cfg)
    topk = resolve_topk(args, cfg)
    a_path, b_path, gpu_path, cpu_out_path = resolve_paths(args, cfg, example_root)

    a_disp = display_path(a_path, example_root)
    b_disp = display_path(b_path, example_root)
    gpu_disp = display_path(gpu_path, example_root)
    cpu_disp = display_path(cpu_out_path, example_root)

    a = read_floats(a_path)
    b = read_floats(b_path)
    gpu = read_floats(gpu_path)

    expected_a = m * k
    expected_b = k * n
    expected_c = m * n

    if len(a) != expected_a:
        raise ValueError(f"{a_disp} contains {len(a)} values, expected {expected_a} for A({m}x{k}).")
    if len(b) != expected_b:
        raise ValueError(f"{b_disp} contains {len(b)} values, expected {expected_b} for B({k}x{n}).")
    if len(gpu) != expected_c:
        raise ValueError(
            f"{gpu_disp} contains {len(gpu)} values, expected {expected_c} for C({m}x{n})."
        )

    cpu = matmul_row_major(a, b, m, k, n)
    write_floats(cpu_out_path, cpu)
    max_abs, mae, rmse = compare_vectors(gpu, cpu)

    print(f"Compared {len(gpu)} values (shape: {m}x{n}, reduction k={k})")
    print(f"CPU output written to: {cpu_disp}")
    print(f"Max abs error: {max_abs:.8f}")
    print(f"Mean abs error: {mae:.8f}")
    print(f"RMSE: {rmse:.8f}")

    print(f"\nWorst {topk} mismatches:")
    for err, r, c, g, t in worst_k(gpu, cpu, n, topk):
        print(f"  ({r:2d}, {c:2d})  gpu={g: .8f}  cpu={t: .8f}  abs_err={err:.8f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
