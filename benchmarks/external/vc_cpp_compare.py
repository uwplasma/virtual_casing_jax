"""Compare virtual_casing_jax against hiddenSymmetries/virtual-casing.

The external package wraps the BIEST/SCTL implementation used by the upstream
virtual-casing tests, including the W7-X surface benchmark described in
Malhotra et al., Plasma Physics and Controlled Fusion 62, 024004 (2020).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXTERNAL_ROOT = os.environ.get("VIRTUAL_CASING_CPP_ROOT")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _git_commit(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    return result.stdout.strip()


def _unflatten(arr: Any, nt: int, npol: int) -> np.ndarray:
    return np.asarray(arr, dtype=float).reshape((3, nt, npol))


def _flatten(arr: Any) -> np.ndarray:
    return np.asarray(arr, dtype=float).reshape(-1)


def _relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    denom = np.linalg.norm(reference.ravel())
    if denom == 0.0:
        return float(np.linalg.norm(candidate.ravel()))
    return float(np.linalg.norm((candidate - reference).ravel()) / denom)


def _max_relative_to_scale(candidate: np.ndarray, reference: np.ndarray, scale: np.ndarray) -> float:
    denom = float(np.max(np.abs(scale)))
    if denom == 0.0:
        return float(np.max(np.abs(candidate - reference)))
    return float(np.max(np.abs(candidate - reference)) / denom)


def run_compare(args: argparse.Namespace) -> dict[str, Any]:
    external_root = args.external_root.resolve() if args.external_root is not None else None
    if external_root is not None and str(external_root) not in sys.path:
        sys.path.insert(0, str(external_root))

    try:
        import virtual_casing as vc_cpp
    except Exception as exc:
        if args.skip_if_missing:
            return {
                "status": "skipped",
                "reason": f"could not import hiddenSymmetries virtual_casing: {exc}",
                "external_root": str(external_root) if external_root is not None else None,
            }
        raise

    from virtual_casing_jax.virtual_casing import VirtualCasingJAX

    surf_type = getattr(vc_cpp.SurfType, args.surface)
    t0 = time.perf_counter()
    X = vc_cpp.VirtualCasingTestData.surface_coordinates(
        args.nfp,
        args.half_period,
        args.surface_nt,
        args.surface_np,
        surf_type,
    )
    Bext_src, Bint_src = vc_cpp.VirtualCasingTestData.magnetic_field_data(
        args.nfp,
        args.half_period,
        args.surface_nt,
        args.surface_np,
        X,
        args.src_nt,
        args.src_np,
    )
    Bext_true, Bint_true = vc_cpp.VirtualCasingTestData.magnetic_field_data(
        args.nfp,
        args.half_period,
        args.surface_nt,
        args.surface_np,
        X,
        args.trg_nt,
        args.trg_np,
    )
    data_seconds = time.perf_counter() - t0

    Btotal_src = np.asarray(Bext_src) + np.asarray(Bint_src)

    vcasing = vc_cpp.VirtualCasing()
    t0 = time.perf_counter()
    vcasing.setup(
        args.digits,
        args.nfp,
        args.half_period,
        args.surface_nt,
        args.surface_np,
        X,
        args.src_nt,
        args.src_np,
        args.trg_nt,
        args.trg_np,
    )
    Bext_cpp = np.asarray(vcasing.compute_external_B(Btotal_src))
    cpp_seconds = time.perf_counter() - t0

    jax_vc = VirtualCasingJAX()
    t0 = time.perf_counter()
    jax_vc.setup(
        args.digits,
        args.nfp,
        args.half_period,
        args.surface_nt,
        args.surface_np,
        X,
        args.src_nt,
        args.src_np,
        args.trg_nt,
        args.trg_np,
    )
    Bext_jax = _flatten(
        jax_vc.compute_external_B(
            _unflatten(Btotal_src, args.src_nt, args.src_np),
            digits=args.digits,
            chunk_size=args.chunk_size,
            target_chunk_size=args.target_chunk_size,
        )
    )
    jax_seconds = time.perf_counter() - t0

    Bext_true = np.asarray(Bext_true)
    metrics: dict[str, Any] = {
        "status": "completed",
        "surface": args.surface,
        "digits": int(args.digits),
        "nfp": int(args.nfp),
        "half_period": bool(args.half_period),
        "surface_nt": int(args.surface_nt),
        "surface_np": int(args.surface_np),
        "src_nt": int(args.src_nt),
        "src_np": int(args.src_np),
        "trg_nt": int(args.trg_nt),
        "trg_np": int(args.trg_np),
        "data_seconds": float(data_seconds),
        "hidden_symmetries_seconds": float(cpp_seconds),
        "virtual_casing_jax_seconds": float(jax_seconds),
        "hidden_symmetries_vs_true_relative_l2": _relative_l2(Bext_cpp, Bext_true),
        "virtual_casing_jax_vs_hidden_symmetries_relative_l2": _relative_l2(Bext_jax, Bext_cpp),
        "virtual_casing_jax_vs_true_relative_l2": _relative_l2(Bext_jax, Bext_true),
        "hidden_symmetries_vs_true_max_relative_to_src_total": _max_relative_to_scale(
            Bext_cpp, Bext_true, Btotal_src
        ),
        "virtual_casing_jax_vs_hidden_symmetries_max_relative_to_src_total": _max_relative_to_scale(
            Bext_jax, Bext_cpp, Btotal_src
        ),
        "virtual_casing_jax_vs_true_max_relative_to_src_total": _max_relative_to_scale(
            Bext_jax, Bext_true, Btotal_src
        ),
        "virtual_casing_jax_commit": _git_commit(ROOT),
        "hidden_symmetries_virtual_casing_commit": _git_commit(external_root) if external_root is not None else None,
    }
    failures = []
    if metrics["virtual_casing_jax_vs_hidden_symmetries_relative_l2"] > args.max_jax_vs_cpp_relative_l2:
        failures.append(
            "virtual_casing_jax_vs_hidden_symmetries_relative_l2="
            f"{metrics['virtual_casing_jax_vs_hidden_symmetries_relative_l2']:.6g} "
            f"> {args.max_jax_vs_cpp_relative_l2:.6g}"
        )
    if metrics["virtual_casing_jax_vs_true_max_relative_to_src_total"] > args.max_jax_vs_true_max_relative:
        failures.append(
            "virtual_casing_jax_vs_true_max_relative_to_src_total="
            f"{metrics['virtual_casing_jax_vs_true_max_relative_to_src_total']:.6g} "
            f"> {args.max_jax_vs_true_max_relative:.6g}"
        )
    metrics["threshold_failures"] = failures
    metrics["passed_thresholds"] = not failures
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-root", type=Path, default=DEFAULT_EXTERNAL_ROOT)
    parser.add_argument("--surface", default="W7X_")
    parser.add_argument("--digits", type=int, default=3)
    parser.add_argument("--nfp", type=int, default=5)
    parser.add_argument("--half-period", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--surface-nt", type=int, default=10)
    parser.add_argument("--surface-np", type=int, default=12)
    parser.add_argument("--src-nt", type=int, default=18)
    parser.add_argument("--src-np", type=int, default=32)
    parser.add_argument("--trg-nt", type=int, default=12)
    parser.add_argument("--trg-np", type=int, default=16)
    parser.add_argument("--chunk-size", default=1024)
    parser.add_argument("--target-chunk-size", default="auto")
    parser.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "external" / "vc_cpp_compare.json")
    parser.add_argument("--max-jax-vs-cpp-relative-l2", type=float, default=1.0e-2)
    parser.add_argument("--max-jax-vs-true-max-relative", type=float, default=3.0e-2)
    parser.add_argument("--skip-if-missing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metrics = run_compare(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics.get("passed_thresholds", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
