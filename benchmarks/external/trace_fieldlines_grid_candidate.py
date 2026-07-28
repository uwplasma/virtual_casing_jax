"""Generate an independent candidate trace from a STELLOPT/FIELDLINES HDF5 grid.

FIELDLINES HDF5 output can contain both the gridded field-line right-hand side
(``B_R`` and ``B_Z`` after FIELDLINES has formed ``R*BR/BPHI`` and
``R*BZ/BPHI``) and the trajectories traced by STELLOPT. This script traces the
stored grid with a small NumPy/SciPy RK4 integrator and writes comparator-ready
NPZ files for a short-horizon candidate-vs-reference benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from scipy.ndimage import map_coordinates


ROOT = Path(__file__).resolve().parents[2]


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


def _require_uniform_axis(axis: np.ndarray, name: str) -> float:
    if axis.ndim != 1 or axis.size < 2:
        raise ValueError(f"{name} must be a one-dimensional axis with at least two points")
    spacing = np.diff(axis)
    if not np.allclose(spacing, spacing[0], rtol=1e-12, atol=1e-14):
        raise ValueError(f"{name} must be uniformly spaced for this benchmark tracer")
    return float(spacing[0])


def _load_h5(path: Path) -> dict[str, np.ndarray]:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("h5py is required to read STELLOPT/FIELDLINES HDF5 output") from exc

    with h5py.File(path, "r") as f:
        required = ("raxis", "phiaxis", "zaxis", "B_R", "B_Z", "R_lines", "PHI_lines", "Z_lines")
        missing = [name for name in required if name not in f]
        if missing:
            raise ValueError(f"FIELDLINES HDF5 file is missing dataset(s): {missing}")
        data = {name: np.asarray(f[name], dtype=float) for name in required}
        data["npoinc"] = np.asarray(f["npoinc"], dtype=int) if "npoinc" in f else np.asarray([1], dtype=int)
    return data


class FieldlinesGridRHS:
    """Interpolated ``dR/dphi`` and ``dZ/dphi`` from a FIELDLINES grid."""

    def __init__(self, data: dict[str, np.ndarray]):
        self.raxis = data["raxis"]
        self.phiaxis = data["phiaxis"]
        self.zaxis = data["zaxis"]
        self.br = data["B_R"]
        self.bz = data["B_Z"]
        if self.br.shape != self.bz.shape:
            raise ValueError(f"B_R and B_Z shape mismatch: {self.br.shape} vs {self.bz.shape}")
        expected = (self.zaxis.size, self.phiaxis.size, self.raxis.size)
        if self.br.shape != expected:
            raise ValueError(
                "FIELDLINES HDF5 arrays are expected in Fortran-written order "
                f"(nz, nphi, nr)={expected}, got {self.br.shape}"
            )
        self.dr = _require_uniform_axis(self.raxis, "raxis")
        self.dphi = _require_uniform_axis(self.phiaxis, "phiaxis")
        self.dz = _require_uniform_axis(self.zaxis, "zaxis")
        self.period = float(self.phiaxis[-1])

    def __call__(self, phi: float, R: float, Z: float) -> tuple[float, float]:
        coords = np.asarray(
            [
                [(Z - self.zaxis[0]) / self.dz],
                [((phi % self.period) - self.phiaxis[0]) / self.dphi],
                [(R - self.raxis[0]) / self.dr],
            ],
            dtype=float,
        )
        dR = float(map_coordinates(self.br, coords, order=1, mode="nearest")[0])
        dZ = float(map_coordinates(self.bz, coords, order=1, mode="nearest")[0])
        return dR, dZ


def _rk4_step(rhs: FieldlinesGridRHS, phi: float, R: float, Z: float, step: float) -> tuple[float, float]:
    k1 = np.asarray(rhs(phi, R, Z))
    k2 = np.asarray(rhs(phi + 0.5 * step, R + 0.5 * step * k1[0], Z + 0.5 * step * k1[1]))
    k3 = np.asarray(rhs(phi + 0.5 * step, R + 0.5 * step * k2[0], Z + 0.5 * step * k2[1]))
    k4 = np.asarray(rhs(phi + step, R + step * k3[0], Z + step * k3[1]))
    R_next, Z_next = np.asarray([R, Z]) + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return float(R_next), float(Z_next)


def _line_ids(values: str | None, nlines: int) -> list[int]:
    if values is None:
        return list(range(nlines))
    ids = [int(value) for value in values.split(",") if value.strip()]
    if not ids:
        raise ValueError("--lines must list at least one line index")
    bad = [line for line in ids if line < 0 or line >= nlines]
    if bad:
        raise ValueError(f"line index out of range for {nlines} lines: {bad}")
    return ids


def _flatten_section_major(samples: np.ndarray) -> np.ndarray:
    return np.transpose(samples, (1, 0, 2)).reshape((-1, 3))


def _samples_from_arrays(rphiz: np.ndarray, lines: list[int]) -> dict[str, np.ndarray]:
    nlines, nsections, _ = rphiz.shape
    return {
        "poincare_rphiz": _flatten_section_major(rphiz),
        "line_id": np.broadcast_to(np.asarray(lines, dtype=float), (nsections, nlines)).reshape(-1),
        "section_phi": np.transpose(rphiz[:, :, 1], (1, 0)).reshape(-1),
    }


def trace_grid_candidate(
    h5_path: Path,
    *,
    lines: list[int],
    nsections: int,
    substeps: int,
    source_label: str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    data = _load_h5(h5_path)
    R_ref = data["R_lines"]
    phi_ref = data["PHI_lines"]
    Z_ref = data["Z_lines"]
    if R_ref.ndim != 2 or phi_ref.shape != R_ref.shape or Z_ref.shape != R_ref.shape:
        raise ValueError("R_lines, PHI_lines, and Z_lines must share shape (nsteps, nlines)")
    stride = int(np.asarray(data["npoinc"]).reshape(-1)[0]) if np.asarray(data["npoinc"]).size else 1
    if stride <= 0:
        raise ValueError(f"npoinc must be positive, got {stride}")
    if nsections <= 0:
        raise ValueError("nsections must be positive")
    if substeps <= 0:
        raise ValueError("substeps must be positive")
    max_index = (nsections - 1) * stride
    if max_index >= R_ref.shape[0]:
        raise ValueError(f"requested {nsections} sections exceeds available trajectory length")

    rhs = FieldlinesGridRHS(data)
    sample_indices = np.arange(0, nsections * stride, stride, dtype=int)
    candidate = np.empty((len(lines), nsections, 3), dtype=float)
    reference = np.empty_like(candidate)

    for line_pos, line in enumerate(lines):
        R = float(R_ref[0, line])
        Z = float(Z_ref[0, line])
        phi = float(phi_ref[0, line])
        if phi < 0.0:
            raise ValueError(f"line {line} has invalid initial phi={phi}")
        candidate[line_pos, 0] = (R, phi, Z)
        reference[line_pos] = np.stack(
            (R_ref[sample_indices, line], phi_ref[sample_indices, line], Z_ref[sample_indices, line]),
            axis=1,
        )
        for step_index in range(1, max_index + 1):
            target_phi = float(phi_ref[step_index, line])
            if target_phi < 0.0:
                raise ValueError(f"line {line} terminated before requested section count")
            total_step = target_phi - phi
            for _ in range(substeps):
                R, Z = _rk4_step(rhs, phi, R, Z, total_step / substeps)
                phi += total_step / substeps
            if step_index % stride == 0:
                candidate[line_pos, step_index // stride] = (R, phi, Z)

    candidate_samples = _samples_from_arrays(candidate, lines)
    reference_samples = _samples_from_arrays(reference, lines)
    candidate_samples["connection_lengths"] = np.zeros(len(lines), dtype=float)
    reference_samples["connection_lengths"] = np.zeros(len(lines), dtype=float)
    metadata = {
        "source": source_label or str(h5_path),
        "lines": lines,
        "nsections": int(nsections),
        "npoinc": int(stride),
        "substeps": int(substeps),
        "method": "RK4 on stored FIELDLINES R*BR/BPHI and R*BZ/BPHI grid",
        "axis_order": "(nz, nphi, nr)",
    }
    return reference_samples, candidate_samples, metadata


def _write_npz(path: Path, samples: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(samples)
    payload["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez(path, **payload)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5", type=Path, help="STELLOPT/FIELDLINES HDF5 output")
    parser.add_argument("--lines", default="0", help="Comma-separated zero-based FIELDLINES line indices")
    parser.add_argument("--nsections", type=int, default=16)
    parser.add_argument("--substeps", type=int, default=1, help="RK4 substeps per stored FIELDLINES step")
    parser.add_argument("--reference-out", type=Path, required=True)
    parser.add_argument("--candidate-out", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--source-label", default=None, help="Portable source label stored in artifacts")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data = _load_h5(args.h5)
    lines = _line_ids(args.lines, data["R_lines"].shape[1])
    reference, candidate, metadata = trace_grid_candidate(
        args.h5,
        lines=lines,
        nsections=int(args.nsections),
        substeps=int(args.substeps),
        source_label=args.source_label,
    )
    metadata["virtual_casing_jax_commit"] = _git_commit(ROOT)
    _write_npz(args.reference_out, reference, metadata | {"kind": "fieldlines_reference_subset"})
    _write_npz(args.candidate_out, candidate, metadata | {"kind": "fieldlines_grid_candidate"})
    report = {
        "status": "completed",
        "reference_out": str(args.reference_out),
        "candidate_out": str(args.candidate_out),
        **metadata,
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
