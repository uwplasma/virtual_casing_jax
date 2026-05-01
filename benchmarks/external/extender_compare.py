"""Compare JAX VMEC-extender field samples with STELLOPT/EXTENDER output.

This script is intentionally format-oriented rather than tied to one local
STELLOPT build. It consumes point-field samples exported by EXTENDER and by the
JAX workflow, then reports physics metrics for total, coil-only, and
plasma-only fields plus the decomposition closure ``B_total = B_coils +
B_plasma`` when the inputs provide all three components.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

FIELD_NAMES = ("B_total_xyz", "B_plasma_xyz", "B_coils_xyz")
POINT_ALIASES = ("xyz", "points_xyz", "target_xyz", "targets_xyz")
FIELD_ALIASES = {
    "B_xyz": "B_total_xyz",
    "B_total": "B_total_xyz",
    "total_xyz": "B_total_xyz",
    "B_plasma": "B_plasma_xyz",
    "plasma_xyz": "B_plasma_xyz",
    "B_coils": "B_coils_xyz",
    "B_coil_xyz": "B_coils_xyz",
    "B_coil": "B_coils_xyz",
    "coil_xyz": "B_coils_xyz",
}
CSV_FIELD_COLUMNS = {
    "B_total_xyz": (("b_total_x", "b_total_y", "b_total_z"), ("btot_x", "btot_y", "btot_z"), ("bx", "by", "bz")),
    "B_plasma_xyz": (("b_plasma_x", "b_plasma_y", "b_plasma_z"), ("bplasma_x", "bplasma_y", "bplasma_z")),
    "B_coils_xyz": (
        ("b_coils_x", "b_coils_y", "b_coils_z"),
        ("b_coil_x", "b_coil_y", "b_coil_z"),
        ("bvac_x", "bvac_y", "bvac_z"),
    ),
}


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


def _as_array(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n, 3), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _canonicalize_mapping(data: dict[str, Any]) -> dict[str, np.ndarray]:
    canonical: dict[str, np.ndarray] = {}
    for key in POINT_ALIASES:
        if key in data:
            canonical["xyz"] = _as_array(data[key], key)
            break
    for key, value in data.items():
        name = FIELD_ALIASES.get(key, key)
        if name in FIELD_NAMES:
            canonical[name] = _as_array(value, key)
    if "xyz" not in canonical:
        raise ValueError(f"field samples must include one of {POINT_ALIASES}")
    npts = canonical["xyz"].shape[0]
    for name in FIELD_NAMES:
        if name in canonical and canonical[name].shape[0] != npts:
            raise ValueError(f"{name} has {canonical[name].shape[0]} points but xyz has {npts}")
    return canonical


def _load_json(path: Path) -> dict[str, np.ndarray]:
    raw = json.loads(path.read_text())
    if isinstance(raw, list):
        if not raw:
            raise ValueError("JSON sample list is empty")
        keys = set().union(*(row.keys() for row in raw))
        data = {key: [row[key] for row in raw if key in row] for key in keys}
        return _canonicalize_mapping(data)
    if isinstance(raw, dict):
        return _canonicalize_mapping(raw)
    raise ValueError("JSON samples must be an object of arrays or a list of sample objects")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return _canonicalize_mapping({key: data[key] for key in data.files})


def _columns(rows: list[dict[str, str]], names: tuple[str, str, str]) -> np.ndarray:
    return np.asarray([[float(row[name]) for name in names] for row in rows], dtype=float)


def _first_available_columns(headers: set[str], candidates: tuple[tuple[str, str, str], ...]):
    for names in candidates:
        if all(name in headers for name in names):
            return names
    return None


def _load_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError("CSV sample file is empty")
    headers = set(reader.fieldnames or ())
    point_cols = _first_available_columns(headers, (("x", "y", "z"), ("X", "Y", "Z")))
    if point_cols is None:
        raise ValueError("CSV samples must include x,y,z columns")
    data: dict[str, Any] = {"xyz": _columns(rows, point_cols)}
    for field_name, candidates in CSV_FIELD_COLUMNS.items():
        cols = _first_available_columns(headers, candidates)
        if cols is not None:
            data[field_name] = _columns(rows, cols)
    return _canonicalize_mapping(data)


def load_field_samples(path: Path) -> dict[str, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_json(path)
    if suffix == ".npz":
        return _load_npz(path)
    if suffix in {".csv", ".txt"}:
        return _load_csv(path)
    raise ValueError(f"Unsupported sample format {suffix!r}; expected .json, .npz, or .csv")


def _relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    denom = np.linalg.norm(reference.ravel())
    if denom == 0.0:
        return float(np.linalg.norm(candidate.ravel()))
    return float(np.linalg.norm((candidate - reference).ravel()) / denom)


def _max_relative_to_reference(candidate: np.ndarray, reference: np.ndarray) -> float:
    denom = float(np.max(np.abs(reference)))
    if denom == 0.0:
        return float(np.max(np.abs(candidate)))
    return float(np.max(np.abs(candidate - reference)) / denom)


def _closure_metrics(samples: dict[str, np.ndarray], prefix: str) -> dict[str, float]:
    if not all(name in samples for name in FIELD_NAMES):
        return {}
    residual = samples["B_total_xyz"] - samples["B_plasma_xyz"] - samples["B_coils_xyz"]
    scale = np.linalg.norm(samples["B_total_xyz"].ravel())
    if scale == 0.0:
        rel = float(np.linalg.norm(residual.ravel()))
    else:
        rel = float(np.linalg.norm(residual.ravel()) / scale)
    return {
        f"{prefix}_closure_relative_l2": rel,
        f"{prefix}_closure_max_abs": float(np.max(np.abs(residual))),
    }


def compare_samples(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    *,
    point_atol: float,
    point_rtol: float,
) -> dict[str, Any]:
    if reference["xyz"].shape != candidate["xyz"].shape:
        raise ValueError(f"point shape mismatch: reference {reference['xyz'].shape}, candidate {candidate['xyz'].shape}")
    if not np.allclose(candidate["xyz"], reference["xyz"], rtol=point_rtol, atol=point_atol):
        point_delta = candidate["xyz"] - reference["xyz"]
        raise ValueError(
            "candidate and reference target points differ; "
            f"max_abs_delta={float(np.max(np.abs(point_delta))):.6g}"
        )

    fields = [name for name in FIELD_NAMES if name in reference and name in candidate]
    if not fields:
        raise ValueError(f"no common field components found; expected any of {FIELD_NAMES}")

    metrics: dict[str, Any] = {
        "n_points": int(reference["xyz"].shape[0]),
        "fields_compared": fields,
        "point_max_abs_delta": float(np.max(np.abs(candidate["xyz"] - reference["xyz"]))),
    }
    for name in fields:
        diff = candidate[name] - reference[name]
        metrics[f"{name}_relative_l2"] = _relative_l2(candidate[name], reference[name])
        metrics[f"{name}_max_abs"] = float(np.max(np.abs(diff)))
        metrics[f"{name}_max_relative_to_ref_max"] = _max_relative_to_reference(candidate[name], reference[name])

    metrics.update(_closure_metrics(reference, "reference"))
    metrics.update(_closure_metrics(candidate, "candidate"))
    return metrics


def _threshold_failures(metrics: dict[str, Any], args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    for name in metrics.get("fields_compared", []):
        rel = metrics[f"{name}_relative_l2"]
        max_abs = metrics[f"{name}_max_abs"]
        if rel > args.max_relative_l2:
            failures.append(f"{name}_relative_l2={rel:.6g} > {args.max_relative_l2:.6g}")
        if max_abs > args.max_abs:
            failures.append(f"{name}_max_abs={max_abs:.6g} > {args.max_abs:.6g}")
    for key in ("reference_closure_relative_l2", "candidate_closure_relative_l2"):
        if key in metrics and metrics[key] > args.max_closure_relative_l2:
            failures.append(f"{key}={metrics[key]:.6g} > {args.max_closure_relative_l2:.6g}")
    return failures


def _find_stellopt_root(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"status": "not_requested"}
    root = path.expanduser().resolve()
    executables = []
    if root.exists():
        for candidate in root.rglob("*"):
            if candidate.is_file() and candidate.name.lower() in {"xextender", "extender"}:
                executables.append(str(candidate))
    return {"root": str(root), "exists": root.exists(), "extender_executables": executables}


def run_compare(args: argparse.Namespace) -> dict[str, Any]:
    if args.reference is None or args.candidate is None:
        if args.skip_if_missing:
            return {
                "status": "skipped",
                "reason": "provide --reference and --candidate STELLOPT/EXTENDER field-sample files",
                "stellopt": _find_stellopt_root(args.stellopt_root),
                "virtual_casing_jax_commit": _git_commit(ROOT),
            }
        raise ValueError("--reference and --candidate are required unless --skip-if-missing is enabled")

    reference_path = args.reference.resolve()
    candidate_path = args.candidate.resolve()
    if not reference_path.exists() or not candidate_path.exists():
        if args.skip_if_missing:
            return {
                "status": "skipped",
                "reason": "missing reference or candidate field-sample file",
                "reference": str(reference_path),
                "candidate": str(candidate_path),
                "stellopt": _find_stellopt_root(args.stellopt_root),
                "virtual_casing_jax_commit": _git_commit(ROOT),
            }
        missing = [str(path) for path in (reference_path, candidate_path) if not path.exists()]
        raise FileNotFoundError(f"missing field-sample file(s): {missing}")

    reference = load_field_samples(reference_path)
    candidate = load_field_samples(candidate_path)
    metrics = compare_samples(reference, candidate, point_atol=args.point_atol, point_rtol=args.point_rtol)
    metrics.update(
        {
            "status": "completed",
            "reference": str(reference_path),
            "candidate": str(candidate_path),
            "stellopt": _find_stellopt_root(args.stellopt_root),
            "virtual_casing_jax_commit": _git_commit(ROOT),
        }
    )
    failures = _threshold_failures(metrics, args)
    metrics["threshold_failures"] = failures
    metrics["passed_thresholds"] = not failures
    return metrics


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=None, help="STELLOPT/EXTENDER reference samples")
    parser.add_argument("--candidate", type=Path, default=None, help="JAX VMEC-extender candidate samples")
    parser.add_argument("--stellopt-root", type=Path, default=None, help="Optional STELLOPT checkout for provenance")
    parser.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "external" / "extender_compare.json")
    parser.add_argument("--max-relative-l2", type=float, default=1e-4)
    parser.add_argument("--max-abs", type=float, default=1e-6)
    parser.add_argument("--max-closure-relative-l2", type=float, default=1e-10)
    parser.add_argument("--point-atol", type=float, default=1e-12)
    parser.add_argument("--point-rtol", type=float, default=1e-12)
    parser.add_argument(
        "--skip-if-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write a skipped JSON report instead of failing when reference inputs are absent.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metrics = run_compare(args)
    text = json.dumps(metrics, indent=2, sort_keys=True)
    if args.out is not None:
        _write_json(args.out, metrics)
    print(text)
    if metrics.get("status") == "skipped":
        return 0
    return 0 if metrics.get("passed_thresholds") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
