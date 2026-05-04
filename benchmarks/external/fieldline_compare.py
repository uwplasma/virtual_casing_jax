"""Compare JAX/ESSOS field-line diagnostics with external tracing output.

The harness is format-oriented so it can compare outputs from STELLOPT
FIELDLINES, TORLINES, FLARE, or another tracing code once they have been
exported to simple JSON, NPZ, CSV, or STELLOPT/FIELDLINES HDF5 samples. It
reports Poincare point agreement and connection-length agreement, the two
diagnostics called out in the VMEC-extender benchmark plan.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

XYZ_ALIASES = ("poincare_xyz", "points_xyz", "xyz", "poincare_points_xyz")
RPHIZ_ALIASES = ("poincare_rphiz", "rphiz", "R_phi_Z", "poincare_R_phi_Z")
CONNECTION_ALIASES = ("connection_lengths", "connection_length", "connection_length_m", "lengths")
HIT_ALIASES = ("hit_mask", "wall_hit", "wall_hits", "connected")
ID_ALIASES = ("line_id", "fieldline_id", "seed_id")
SECTION_ALIASES = ("section_phi", "phi_section", "poincare_phi")


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


def _as_points(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        arr = arr.reshape((-1, 3))
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n, 3) or (..., 3), got {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one point")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _as_vector(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _as_bool_vector(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    return arr.astype(bool)


def _first_key(data: dict[str, Any], aliases: tuple[str, ...]) -> str | None:
    for key in aliases:
        if key in data:
            return key
    return None


def _canonicalize_mapping(data: dict[str, Any]) -> dict[str, np.ndarray]:
    canonical: dict[str, np.ndarray] = {}

    key = _first_key(data, XYZ_ALIASES)
    if key is not None:
        canonical["poincare_xyz"] = _as_points(data[key], key)
    key = _first_key(data, RPHIZ_ALIASES)
    if key is not None:
        canonical["poincare_rphiz"] = _as_points(data[key], key)
    key = _first_key(data, CONNECTION_ALIASES)
    if key is not None:
        canonical["connection_lengths"] = _as_vector(data[key], key)
    key = _first_key(data, HIT_ALIASES)
    if key is not None:
        canonical["hit_mask"] = _as_bool_vector(data[key], key)
    key = _first_key(data, ID_ALIASES)
    if key is not None:
        canonical["line_id"] = _as_vector(data[key], key)
    key = _first_key(data, SECTION_ALIASES)
    if key is not None:
        canonical["section_phi"] = _as_vector(data[key], key)

    if not any(name in canonical for name in ("poincare_xyz", "poincare_rphiz", "connection_lengths")):
        raise ValueError("field-line samples must include Poincare points or connection lengths")
    return canonical


def _load_json(path: Path) -> dict[str, np.ndarray]:
    raw = json.loads(path.read_text())
    if isinstance(raw, list):
        if not raw:
            raise ValueError("JSON sample list is empty")
        keys = set.intersection(*(set(row.keys()) for row in raw))
        data = {key: [row[key] for row in raw] for key in keys}
        if {"x", "y", "z"}.issubset(keys):
            data["poincare_xyz"] = [[row["x"], row["y"], row["z"]] for row in raw]
        if {"R", "phi", "Z"}.issubset(keys):
            data["poincare_rphiz"] = [[row["R"], row["phi"], row["Z"]] for row in raw]
        return _canonicalize_mapping(data)
    if isinstance(raw, dict):
        data = {key: value for key, value in raw.items() if key != "metadata"}
        return _canonicalize_mapping(data)
    raise ValueError("JSON samples must be an object of arrays or a list of sample objects")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return _canonicalize_mapping({key: data[key] for key in data.files})


def _column(rows: list[dict[str, str]], name: str) -> np.ndarray:
    return np.asarray([float(row[name]) for row in rows], dtype=float)


def _columns(rows: list[dict[str, str]], names: tuple[str, str, str]) -> np.ndarray:
    return np.asarray([[float(row[name]) for name in names] for row in rows], dtype=float)


def _first_available_columns(headers: set[str], candidates: tuple[tuple[str, str, str], ...]):
    for names in candidates:
        if all(name in headers for name in names):
            return names
    return None


def _first_available_column(headers: set[str], candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in headers:
            return name
    return None


def _load_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError("CSV sample file is empty")
    headers = set(reader.fieldnames or ())
    data: dict[str, Any] = {}

    xyz_cols = _first_available_columns(headers, (("x", "y", "z"), ("X", "Y", "Z")))
    if xyz_cols is not None:
        data["poincare_xyz"] = _columns(rows, xyz_cols)
    rphiz_cols = _first_available_columns(headers, (("R", "phi", "Z"), ("r", "phi", "z")))
    if rphiz_cols is not None:
        data["poincare_rphiz"] = _columns(rows, rphiz_cols)
    conn_col = _first_available_column(headers, CONNECTION_ALIASES)
    if conn_col is not None:
        data["connection_lengths"] = _column(rows, conn_col)
    hit_col = _first_available_column(headers, HIT_ALIASES)
    if hit_col is not None:
        data["hit_mask"] = _column(rows, hit_col)
    line_col = _first_available_column(headers, ID_ALIASES)
    if line_col is not None:
        data["line_id"] = _column(rows, line_col)
    section_col = _first_available_column(headers, SECTION_ALIASES)
    if section_col is not None:
        data["section_phi"] = _column(rows, section_col)
    return _canonicalize_mapping(data)


def _h5_scalar(data: Any, name: str, default: int | None = None) -> int:
    if name not in data:
        if default is None:
            raise ValueError(f"FIELDLINES HDF5 file is missing {name!r}")
        return default
    arr = np.asarray(data[name]).reshape(-1)
    if arr.size == 0:
        if default is None:
            raise ValueError(f"FIELDLINES HDF5 dataset {name!r} is empty")
        return default
    return int(arr[0])


def _load_stellopt_h5(path: Path) -> dict[str, np.ndarray]:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - exercised only without optional dependency
        raise RuntimeError("h5py is required to load STELLOPT/FIELDLINES HDF5 output") from exc

    with h5py.File(path, "r") as f:
        missing = [name for name in ("R_lines", "PHI_lines", "Z_lines") if name not in f]
        if missing:
            raise ValueError(f"FIELDLINES HDF5 file is missing trajectory dataset(s): {missing}")

        R = np.asarray(f["R_lines"], dtype=float)
        phi = np.asarray(f["PHI_lines"], dtype=float)
        Z = np.asarray(f["Z_lines"], dtype=float)
        if R.ndim != 2 or phi.shape != R.shape or Z.shape != R.shape:
            raise ValueError(
                "FIELDLINES trajectory datasets must share shape (nsteps, nlines); "
                f"got R={R.shape}, PHI={phi.shape}, Z={Z.shape}"
            )
        if R.shape[0] == 0 or R.shape[1] == 0:
            raise ValueError("FIELDLINES trajectory datasets must contain at least one step and one line")

        stride = _h5_scalar(f, "npoinc", default=1)
        if stride <= 0:
            raise ValueError(f"FIELDLINES npoinc must be positive, got {stride}")
        sample_indices = np.arange(0, R.shape[0], stride, dtype=int)

        rphiz = np.stack((R[sample_indices], phi[sample_indices], Z[sample_indices]), axis=-1)
        nsections, nlines, _ = rphiz.shape
        data: dict[str, Any] = {
            "poincare_rphiz": rphiz.reshape((-1, 3)),
            "line_id": np.broadcast_to(np.arange(nlines, dtype=float), (nsections, nlines)).reshape(-1),
            "section_phi": phi[sample_indices].reshape(-1),
        }

        if "L_lines" in f:
            lengths = np.asarray(f["L_lines"], dtype=float).reshape(-1)
            if lengths.size not in {nlines, nsections * nlines}:
                raise ValueError(
                    "FIELDLINES L_lines must have one value per line or sampled point; "
                    f"got {lengths.size} values for {nlines} lines and {nsections} sections"
                )
            data["connection_lengths"] = lengths
        if "wall_hits" in f:
            hits = np.asarray(f["wall_hits"]).reshape(-1)
            if hits.size:
                data["hit_mask"] = hits.astype(bool)
    return _canonicalize_mapping(data)


def load_fieldline_samples(path: Path) -> dict[str, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_json(path)
    if suffix == ".npz":
        return _load_npz(path)
    if suffix in {".csv", ".txt"}:
        return _load_csv(path)
    if suffix in {".h5", ".hdf5"}:
        return _load_stellopt_h5(path)
    raise ValueError(f"Unsupported sample format {suffix!r}; expected .json, .npz, .csv, or .h5")


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


def _point_scale(reference: np.ndarray) -> float:
    scale = float(np.sqrt(np.mean(np.sum(reference**2, axis=1))))
    return scale if scale != 0.0 else 1.0


def _ordered_point_metrics(reference: np.ndarray, candidate: np.ndarray, prefix: str) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError(f"{prefix} shape mismatch: reference {reference.shape}, candidate {candidate.shape}")
    diff = candidate - reference
    distances = np.linalg.norm(diff, axis=1)
    rms = float(np.sqrt(np.mean(distances**2)))
    return {
        f"{prefix}_point_relative_l2": _relative_l2(candidate, reference),
        f"{prefix}_point_rms_distance": rms,
        f"{prefix}_point_relative_rms_distance": rms / _point_scale(reference),
        f"{prefix}_point_max_distance": float(np.max(distances)),
        f"{prefix}_point_max_abs_component": float(np.max(np.abs(diff))),
    }


def _label_key(line_id: float, section_phi: float, section_phi_atol: float) -> tuple[float, int | float]:
    line_key = float(np.round(line_id, 12))
    if section_phi_atol <= 0.0:
        return line_key, float(np.round(section_phi, 12))
    return line_key, int(np.rint(section_phi / section_phi_atol))


def _label_index(
    samples: dict[str, np.ndarray],
    point_name: str,
    section_phi_atol: float,
) -> dict[tuple[float, int | float], list[int]]:
    if "line_id" not in samples or "section_phi" not in samples:
        raise ValueError(
            "point_mode='labeled' requires both reference and candidate samples to provide "
            "line_id and section_phi arrays"
        )
    point_count = samples[point_name].shape[0]
    line_id = np.asarray(samples["line_id"], dtype=float).reshape(-1)
    section_phi = np.asarray(samples["section_phi"], dtype=float).reshape(-1)
    if line_id.size != point_count:
        raise ValueError(f"line_id length {line_id.size} does not match {point_name} point count {point_count}")
    if section_phi.size != point_count:
        raise ValueError(
            f"section_phi length {section_phi.size} does not match {point_name} point count {point_count}"
        )

    index: dict[tuple[float, int | float], list[int]] = defaultdict(list)
    for idx, (line, phi) in enumerate(zip(line_id, section_phi, strict=True)):
        index[_label_key(float(line), float(phi), section_phi_atol)].append(idx)
    return dict(index)


def _labeled_point_metrics(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    point_name: str,
    prefix: str,
    section_phi_atol: float,
) -> dict[str, float | int]:
    ref_index = _label_index(reference, point_name, section_phi_atol)
    cand_index = _label_index(candidate, point_name, section_phi_atol)
    ref_labels = set(ref_index)
    cand_labels = set(cand_index)
    common_labels = sorted(ref_labels & cand_labels)
    count_mismatch_labels = [
        label for label in common_labels if len(ref_index[label]) != len(cand_index[label])
    ]
    comparable_labels = [label for label in common_labels if label not in set(count_mismatch_labels)]

    metrics: dict[str, float | int] = {
        f"{prefix}_labeled_reference_label_count": len(ref_labels),
        f"{prefix}_labeled_candidate_label_count": len(cand_labels),
        f"{prefix}_labeled_matched_label_count": len(common_labels),
        f"{prefix}_labeled_missing_candidate_label_count": len(ref_labels - cand_labels),
        f"{prefix}_labeled_extra_candidate_label_count": len(cand_labels - ref_labels),
        f"{prefix}_labeled_count_mismatch_label_count": len(count_mismatch_labels),
    }
    if not comparable_labels:
        raise ValueError(f"{prefix} has no comparable labeled Poincare samples")

    ref_order = np.asarray([idx for label in comparable_labels for idx in ref_index[label]], dtype=int)
    cand_order = np.asarray([idx for label in comparable_labels for idx in cand_index[label]], dtype=int)
    metrics[f"{prefix}_labeled_matched_point_count"] = int(ref_order.size)
    metrics.update(
        _ordered_point_metrics(
            reference[point_name][ref_order],
            candidate[point_name][cand_order],
            f"{prefix}_labeled",
        )
    )
    return metrics


def _cloud_point_metrics(reference: np.ndarray, candidate: np.ndarray, prefix: str) -> dict[str, float]:
    if reference.shape[1] != 3 or candidate.shape[1] != 3:
        raise ValueError(f"{prefix} point clouds must have 3 components")
    distances = np.linalg.norm(candidate[:, None, :] - reference[None, :, :], axis=2)
    candidate_to_reference = np.min(distances, axis=1)
    reference_to_candidate = np.min(distances, axis=0)
    symmetric_rms = float(
        np.sqrt(0.5 * (np.mean(candidate_to_reference**2) + np.mean(reference_to_candidate**2)))
    )
    symmetric_max = float(max(np.max(candidate_to_reference), np.max(reference_to_candidate)))
    return {
        f"{prefix}_cloud_symmetric_rms_distance": symmetric_rms,
        f"{prefix}_cloud_symmetric_relative_rms_distance": symmetric_rms / _point_scale(reference),
        f"{prefix}_cloud_symmetric_max_distance": symmetric_max,
        f"{prefix}_cloud_candidate_to_reference_max_distance": float(np.max(candidate_to_reference)),
        f"{prefix}_cloud_reference_to_candidate_max_distance": float(np.max(reference_to_candidate)),
    }


def _connection_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError(
            f"connection_lengths shape mismatch: reference {reference.shape}, candidate {candidate.shape}"
        )
    diff = candidate - reference
    return {
        "connection_length_relative_l2": _relative_l2(candidate, reference),
        "connection_length_max_abs": float(np.max(np.abs(diff))),
        "connection_length_max_relative_to_ref_max": _max_relative_to_reference(candidate, reference),
    }


def _hit_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError(f"hit_mask shape mismatch: reference {reference.shape}, candidate {candidate.shape}")
    mismatch = reference != candidate
    return {
        "hit_mask_mismatch_fraction": float(np.mean(mismatch)),
        "hit_mask_mismatch_count": int(np.count_nonzero(mismatch)),
    }


def compare_samples(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    *,
    point_mode: str,
    section_phi_atol: float = 1e-10,
) -> dict[str, Any]:
    if point_mode not in {"ordered", "cloud", "labeled"}:
        raise ValueError("point_mode must be 'ordered', 'cloud', or 'labeled'")

    common_points = [name for name in ("poincare_xyz", "poincare_rphiz") if name in reference and name in candidate]
    has_connection = "connection_lengths" in reference and "connection_lengths" in candidate
    has_hits = "hit_mask" in reference and "hit_mask" in candidate
    if not common_points and not has_connection:
        raise ValueError("no common Poincare point coordinates or connection lengths found")

    metrics: dict[str, Any] = {
        "point_mode": point_mode,
        "point_coordinates_compared": common_points,
        "compared_connection_lengths": has_connection,
        "compared_hit_masks": has_hits,
    }
    for name in common_points:
        prefix = name.replace("poincare_", "poincare_")
        metrics[f"{prefix}_reference_count"] = int(reference[name].shape[0])
        metrics[f"{prefix}_candidate_count"] = int(candidate[name].shape[0])
        if point_mode == "ordered":
            metrics.update(_ordered_point_metrics(reference[name], candidate[name], prefix))
        elif point_mode == "cloud":
            metrics.update(_cloud_point_metrics(reference[name], candidate[name], prefix))
        else:
            metrics.update(_labeled_point_metrics(reference, candidate, name, prefix, section_phi_atol))

    if has_connection:
        metrics["connection_length_count"] = int(reference["connection_lengths"].shape[0])
        metrics.update(_connection_metrics(reference["connection_lengths"], candidate["connection_lengths"]))
    if has_hits:
        metrics["hit_mask_count"] = int(reference["hit_mask"].shape[0])
        metrics.update(_hit_metrics(reference["hit_mask"], candidate["hit_mask"]))
    return metrics


def _threshold_failures(metrics: dict[str, Any], args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    for name in metrics.get("point_coordinates_compared", []):
        prefix = name.replace("poincare_", "poincare_")
        if metrics["point_mode"] == "ordered":
            checks = (
                (f"{prefix}_point_relative_l2", args.max_point_relative_l2),
                (f"{prefix}_point_rms_distance", args.max_point_rms_distance),
                (f"{prefix}_point_max_distance", args.max_point_max_distance),
            )
        elif metrics["point_mode"] == "cloud":
            checks = (
                (f"{prefix}_cloud_symmetric_relative_rms_distance", args.max_cloud_relative_rms_distance),
                (f"{prefix}_cloud_symmetric_rms_distance", args.max_cloud_rms_distance),
                (f"{prefix}_cloud_symmetric_max_distance", args.max_cloud_max_distance),
            )
        else:
            for count_key in (
                f"{prefix}_labeled_missing_candidate_label_count",
                f"{prefix}_labeled_extra_candidate_label_count",
                f"{prefix}_labeled_count_mismatch_label_count",
            ):
                if metrics.get(count_key, 0) > 0:
                    failures.append(f"{count_key}={metrics[count_key]} > 0")
            checks = (
                (f"{prefix}_labeled_point_relative_l2", args.max_point_relative_l2),
                (f"{prefix}_labeled_point_rms_distance", args.max_point_rms_distance),
                (f"{prefix}_labeled_point_max_distance", args.max_point_max_distance),
            )
        for key, threshold in checks:
            if threshold is not None and key in metrics and metrics[key] > threshold:
                failures.append(f"{key}={metrics[key]:.6g} > {threshold:.6g}")

    if metrics.get("compared_connection_lengths"):
        checks = (
            ("connection_length_relative_l2", args.max_connection_relative_l2),
            ("connection_length_max_abs", args.max_connection_max_abs),
        )
        for key, threshold in checks:
            if threshold is not None and metrics[key] > threshold:
                failures.append(f"{key}={metrics[key]:.6g} > {threshold:.6g}")

    if metrics.get("compared_hit_masks") and args.max_hit_mismatch_fraction is not None:
        value = metrics["hit_mask_mismatch_fraction"]
        if value > args.max_hit_mismatch_fraction:
            failures.append(f"hit_mask_mismatch_fraction={value:.6g} > {args.max_hit_mismatch_fraction:.6g}")
    return failures


def _find_stellopt_root(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"status": "not_requested"}
    root = path.expanduser().resolve()
    executables = []
    if root.exists():
        for candidate in root.rglob("*"):
            if candidate.is_file() and candidate.name.lower() in {"xfieldlines", "xtorlines", "fieldlines", "torlines"}:
                executables.append(str(candidate))
    return {"root": str(root), "exists": root.exists(), "fieldline_executables": executables}


def run_compare(args: argparse.Namespace) -> dict[str, Any]:
    if args.reference is None or args.candidate is None:
        if args.skip_if_missing:
            return {
                "status": "skipped",
                "reason": "provide --reference and --candidate field-line diagnostic files",
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
                "reason": "missing reference or candidate field-line diagnostic file",
                "reference": str(reference_path),
                "candidate": str(candidate_path),
                "stellopt": _find_stellopt_root(args.stellopt_root),
                "virtual_casing_jax_commit": _git_commit(ROOT),
            }
        missing = [str(path) for path in (reference_path, candidate_path) if not path.exists()]
        raise FileNotFoundError(f"missing field-line diagnostic file(s): {missing}")

    reference = load_fieldline_samples(reference_path)
    candidate = load_fieldline_samples(candidate_path)
    metrics = compare_samples(
        reference,
        candidate,
        point_mode=args.point_mode,
        section_phi_atol=args.section_phi_atol,
    )
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
    parser.add_argument("--reference", type=Path, default=None, help="External FIELDLINES/TORLINES samples")
    parser.add_argument("--candidate", type=Path, default=None, help="JAX/ESSOS VMEC-extender samples")
    parser.add_argument("--stellopt-root", type=Path, default=None, help="Optional STELLOPT checkout for provenance")
    parser.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "external" / "fieldline_compare.json")
    parser.add_argument("--point-mode", choices=("ordered", "cloud", "labeled"), default="ordered")
    parser.add_argument(
        "--section-phi-atol",
        type=float,
        default=1e-10,
        help="Toroidal-angle tolerance used to bin section_phi labels in --point-mode labeled.",
    )
    parser.add_argument("--max-point-relative-l2", type=float, default=1e-3)
    parser.add_argument("--max-point-rms-distance", type=float, default=None)
    parser.add_argument("--max-point-max-distance", type=float, default=None)
    parser.add_argument("--max-cloud-relative-rms-distance", type=float, default=1e-3)
    parser.add_argument("--max-cloud-rms-distance", type=float, default=None)
    parser.add_argument("--max-cloud-max-distance", type=float, default=None)
    parser.add_argument("--max-connection-relative-l2", type=float, default=1e-3)
    parser.add_argument("--max-connection-max-abs", type=float, default=None)
    parser.add_argument("--max-hit-mismatch-fraction", type=float, default=0.0)
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
