#!/usr/bin/env python3

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXAMPLE = "examples/01_warp_mma_ptx_m16n8k16"
DEFAULT_CONFIG_NAME = "example.json"

FLOAT_DTYPES = {"float16", "fp16", "float32", "fp32", "float64", "fp64"}
INT_DTYPES = {"int8", "int16", "int32", "int64"}


@dataclass
class GenerationSettings:
    m: int
    k: int
    n: int
    dtype: str
    distribution: str
    low: float
    high: float
    seed: Optional[int]
    a_path: Path
    b_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate A and B matrix inputs using an example JSON config."
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
    parser.add_argument("--a", default=None, help="Override output path for A")
    parser.add_argument("--b", default=None, help="Override output path for B")
    parser.add_argument("--m", type=int, default=None, help="Rows of A")
    parser.add_argument("--k", type=int, default=None, help="Cols of A / rows of B")
    parser.add_argument("--n", type=int, default=None, help="Cols of B")
    parser.add_argument(
        "--dtype",
        default=None,
        help="Data type for random values (float16/float32/int8/int32/...)",
    )
    parser.add_argument("--distribution", default=None, help="Random distribution (uniform)")
    parser.add_argument("--low", type=float, default=None, help="Minimum random value")
    parser.add_argument("--high", type=float, default=None, help="Maximum random value")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed")
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


def resolve_settings(args: argparse.Namespace, cfg: Dict[str, Any], example_root: Path) -> GenerationSettings:
    matrix = cfg.get("matrix", {})
    gen = cfg.get("input_generation", {})
    paths = cfg.get("paths", {})

    m_cfg, k_cfg, n_cfg = resolve_dims_from_matrix_cfg(matrix)
    m = args.m if args.m is not None else m_cfg
    k = args.k if args.k is not None else k_cfg
    n = args.n if args.n is not None else n_cfg

    m = positive_int("m", m)
    k = positive_int("k", k)
    n = positive_int("n", n)

    dtype = (args.dtype if args.dtype is not None else str(gen.get("dtype", "float32"))).lower()
    if dtype not in FLOAT_DTYPES and dtype not in INT_DTYPES:
        raise ValueError(f"Unsupported dtype: {dtype}")

    distribution = (
        args.distribution if args.distribution is not None else str(gen.get("distribution", "uniform"))
    ).lower()
    if distribution != "uniform":
        raise ValueError(f"Unsupported distribution: {distribution}")

    low = args.low if args.low is not None else float(gen.get("low", -1.0))
    high = args.high if args.high is not None else float(gen.get("high", 1.0))
    if low > high:
        raise ValueError(f"low ({low}) must be <= high ({high})")

    if args.seed is not None:
        seed = args.seed
    else:
        seed_cfg = gen.get("seed", None)
        seed = int(seed_cfg) if seed_cfg is not None else None

    a_rel = paths.get("input_a", "inputs/A.txt")
    b_rel = paths.get("input_b", "inputs/B.txt")
    a_path = Path(args.a) if args.a else (example_root / a_rel)
    b_path = Path(args.b) if args.b else (example_root / b_rel)

    return GenerationSettings(
        m=m,
        k=k,
        n=n,
        dtype=dtype,
        distribution=distribution,
        low=low,
        high=high,
        seed=seed,
        a_path=a_path,
        b_path=b_path,
    )


def sample_value(rng: random.Random, dtype: str, low: float, high: float) -> str:
    if dtype in INT_DTYPES:
        lo_int = int(round(low))
        hi_int = int(round(high))
        if lo_int > hi_int:
            lo_int, hi_int = hi_int, lo_int
        return str(rng.randint(lo_int, hi_int))

    value = rng.uniform(low, high)
    if dtype in {"float16", "fp16"}:
        return f"{value:.6f}"
    if dtype in {"float32", "fp32"}:
        return f"{value:.8f}"
    return f"{value:.12f}"


def write_matrix(path: Path, count: int, rng: random.Random, dtype: str, low: float, high: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for _ in range(count):
            f.write(sample_value(rng, dtype, low, high))
            f.write(" ")
        f.write("\n")


def display_path(path: Path, example_root: Path) -> str:
    resolved = path.resolve()
    root = example_root.resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(path)


def main() -> int:
    args = parse_args()
    example_root, cfg_path = resolve_roots(args)
    cfg = load_config(cfg_path)
    settings = resolve_settings(args, cfg, example_root)

    rng = random.Random(settings.seed)
    a_count = settings.m * settings.k
    b_count = settings.k * settings.n

    write_matrix(settings.a_path, a_count, rng, settings.dtype, settings.low, settings.high)
    write_matrix(settings.b_path, b_count, rng, settings.dtype, settings.low, settings.high)

    print(
        "Generated inputs with "
        f"dtype={settings.dtype}, A={settings.m}x{settings.k}, B={settings.k}x{settings.n}, "
        f"seed={settings.seed if settings.seed is not None else 'none'}"
    )
    print(f"A path: {display_path(settings.a_path, example_root)}")
    print(f"B path: {display_path(settings.b_path, example_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
