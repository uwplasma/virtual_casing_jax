"""Trace a FIELDLINES case with an independent Simsopt Biot-Savart field.

This benchmark loader uses a STELLOPT/MAKEGRID ``coils`` file, applies the
same group-current scaling used by ``FIELDLINES/Sources/fieldlines_init_coil.f90``,
and traces the same stored FIELDLINES seed sequence with a small RK4
integrator. The output is comparator-ready NPZ data for short-horizon
FIELDLINES-vs-Simsopt validation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import re
import sys
import tempfile
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from trace_fieldlines_grid_candidate import (  # noqa: E402
    _git_commit,
    _line_ids,
    _load_h5,
    _samples_from_arrays,
    _write_npz,
)


ROOT = Path(__file__).resolve().parents[2]
_EXTCUR_RE = re.compile(r"EXTCUR\(\s*(\d+)\s*\)\s*=\s*([-+0-9.EeDd]+)", re.IGNORECASE)


@dataclass(frozen=True)
class MakegridCoilMetadata:
    group: int
    label: str
    raw_current: float
    point_count: int


def write_sanitized_makegrid_file(source: Path, destination: Path) -> Path:
    """Write a Simsopt-loadable MAKEGRID file with trailing blank lines removed."""

    lines = source.read_text().splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines or lines[-1].strip().lower() != "end":
        raise ValueError(f"{source} does not end with a MAKEGRID 'end' marker")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n")
    return destination


def parse_makegrid_coil_metadata(path: Path) -> list[MakegridCoilMetadata]:
    """Return group labels and raw currents in the same coil order as Simsopt."""

    metadata: list[MakegridCoilMetadata] = []
    current: float | None = None
    point_count = 0
    for line in path.read_text().splitlines()[3:]:
        vals = line.split()
        if len(vals) == 4:
            if point_count == 0:
                current = float(vals[3])
            point_count += 1
        elif len(vals) == 6:
            if current is None or point_count == 0:
                raise ValueError(f"coil terminator encountered before coil points in {path}")
            metadata.append(
                MakegridCoilMetadata(
                    group=int(vals[4]),
                    label=vals[5],
                    raw_current=current,
                    point_count=point_count,
                )
            )
            current = None
            point_count = 0
        elif len(vals) == 1 and vals[0].lower() == "end":
            break
        elif not vals:
            continue
        else:
            raise ValueError(f"unrecognized MAKEGRID line with {len(vals)} fields: {line!r}")
    if not metadata:
        raise ValueError(f"no coils found in {path}")
    return metadata


def parse_fieldlines_extcur(path: Path) -> dict[int, float]:
    """Parse active ``EXTCUR(i)`` assignments from a VMEC/FIELDLINES input file."""

    extcur: dict[int, float] = {}
    for line in path.read_text().splitlines():
        uncommented = line.split("!", 1)[0]
        match = _EXTCUR_RE.search(uncommented)
        if match is None:
            continue
        value = match.group(2).replace("D", "E").replace("d", "e")
        extcur[int(match.group(1))] = float(value)
    return extcur


def scaled_currents_for_fieldlines(
    metadata: list[MakegridCoilMetadata],
    extcur: dict[int, float],
    *,
    current_mode: str = "scaled",
) -> np.ndarray:
    """Apply FIELDLINES group-current semantics to MAKEGRID raw currents."""

    if current_mode not in {"scaled", "raw"}:
        raise ValueError("current_mode must be 'scaled' or 'raw'")
    if current_mode == "raw":
        return np.asarray([coil.raw_current for coil in metadata], dtype=float)

    first_current_by_group: dict[int, float] = {}
    for coil in metadata:
        first_current_by_group.setdefault(coil.group, coil.raw_current)

    currents = []
    for coil in metadata:
        first_current = first_current_by_group[coil.group]
        if first_current == 0.0:
            currents.append(coil.raw_current)
        else:
            currents.append((coil.raw_current / first_current) * extcur.get(coil.group, 0.0))
    return np.asarray(currents, dtype=float)


def _import_simsopt(simsopt_src: Path | None):
    if simsopt_src is not None:
        sys.path.insert(0, str(simsopt_src))
    coil_mod = importlib.import_module("simsopt.field.coil")
    bs_mod = importlib.import_module("simsopt.field.biotsavart")
    return coil_mod, bs_mod


def load_scaled_simsopt_field(
    coils_path: Path,
    input_path: Path,
    *,
    order: int,
    ppp: int,
    current_mode: str,
    simsopt_src: Path | None = None,
):
    """Load a Simsopt Biot-Savart object with FIELDLINES-compatible currents."""

    coil_mod, bs_mod = _import_simsopt(simsopt_src)
    with tempfile.TemporaryDirectory(prefix="fieldlines_simsopt_") as tmpdir:
        sanitized = write_sanitized_makegrid_file(coils_path, Path(tmpdir) / coils_path.name)
        metadata = parse_makegrid_coil_metadata(sanitized)
        extcur = parse_fieldlines_extcur(input_path)
        currents = scaled_currents_for_fieldlines(metadata, extcur, current_mode=current_mode)
        coils = coil_mod.load_coils_from_makegrid_file(str(sanitized), order=order, ppp=ppp)

    if len(coils) != len(currents):
        raise ValueError(f"Simsopt loaded {len(coils)} coils but parsed {len(currents)} current records")
    scaled_coils = [coil_mod.Coil(coil.curve, coil_mod.Current(current)) for coil, current in zip(coils, currents)]
    return bs_mod.BiotSavart(scaled_coils), metadata, extcur, currents


class SimsoptBiotSavartRHS:
    """Cylindrical field-line RHS ``dR/dphi, dZ/dphi`` from Simsopt ``B_xyz``."""

    def __init__(self, biotsavart):
        self.biotsavart = biotsavart

    def __call__(self, phi: float, R: float, Z: float) -> tuple[float, float]:
        point = np.asarray([[R * np.cos(phi), R * np.sin(phi), Z]], dtype=float)
        self.biotsavart.set_points(point)
        B = np.asarray(self.biotsavart.B()[0], dtype=float)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        BR = cos_phi * B[0] + sin_phi * B[1]
        Bphi = -sin_phi * B[0] + cos_phi * B[1]
        if Bphi == 0.0:
            raise ValueError(f"Bphi is zero at R={R}, phi={phi}, Z={Z}")
        return float(R * BR / Bphi), float(R * B[2] / Bphi)


def _rk4_step(rhs: SimsoptBiotSavartRHS, phi: float, R: float, Z: float, step: float) -> tuple[float, float]:
    k1 = np.asarray(rhs(phi, R, Z))
    k2 = np.asarray(rhs(phi + 0.5 * step, R + 0.5 * step * k1[0], Z + 0.5 * step * k1[1]))
    k3 = np.asarray(rhs(phi + 0.5 * step, R + 0.5 * step * k2[0], Z + 0.5 * step * k2[1]))
    k4 = np.asarray(rhs(phi + step, R + step * k3[0], Z + step * k3[1]))
    R_next, Z_next = np.asarray([R, Z]) + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return float(R_next), float(Z_next)


def _group_current_summary(metadata: list[MakegridCoilMetadata], currents: np.ndarray) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for coil, current in zip(metadata, currents):
        key = str(coil.group)
        entry = summary.setdefault(key, {"label": coil.label, "count": 0, "current_min": current, "current_max": current})
        entry["count"] += 1
        entry["current_min"] = float(min(entry["current_min"], current))
        entry["current_max"] = float(max(entry["current_max"], current))
    return summary


def trace_simsopt_candidate(
    h5_path: Path,
    coils_path: Path,
    input_path: Path,
    *,
    lines: list[int],
    nsections: int,
    substeps: int,
    order: int,
    ppp: int,
    current_mode: str = "scaled",
    simsopt_src: Path | None = None,
    source_label: str | None = None,
    coils_label: str | None = None,
    input_label: str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Trace FIELDLINES seeds through a Simsopt coil field."""

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

    biotsavart, metadata, extcur, currents = load_scaled_simsopt_field(
        coils_path,
        input_path,
        order=order,
        ppp=ppp,
        current_mode=current_mode,
        simsopt_src=simsopt_src,
    )
    rhs = SimsoptBiotSavartRHS(biotsavart)
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

    reference_samples = _samples_from_arrays(reference, lines)
    candidate_samples = _samples_from_arrays(candidate, lines)
    reference_samples["connection_lengths"] = np.zeros(len(lines), dtype=float)
    candidate_samples["connection_lengths"] = np.zeros(len(lines), dtype=float)
    benchmark_metadata = {
        "source": source_label or str(h5_path),
        "coils_source": coils_label or str(coils_path),
        "input_source": input_label or str(input_path),
        "lines": lines,
        "nsections": int(nsections),
        "npoinc": int(stride),
        "substeps": int(substeps),
        "method": "RK4 on Simsopt BiotSavart loaded from STELLOPT/MAKEGRID coils",
        "current_mode": current_mode,
        "order": int(order),
        "ppp": int(ppp),
        "ncoils": len(metadata),
        "active_extcur_groups": sorted(extcur),
        "group_currents": _group_current_summary(metadata, currents),
        "simsopt_commit": _git_commit(simsopt_src) if simsopt_src is not None else None,
    }
    return reference_samples, candidate_samples, benchmark_metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5", type=Path, help="STELLOPT/FIELDLINES HDF5 output")
    parser.add_argument("coils", type=Path, help="STELLOPT/MAKEGRID coils file")
    parser.add_argument("input", type=Path, help="FIELDLINES/VMEC input file containing EXTCUR assignments")
    parser.add_argument("--lines", default="0", help="Comma-separated zero-based FIELDLINES line indices")
    parser.add_argument("--nsections", type=int, default=16)
    parser.add_argument("--substeps", type=int, default=2, help="RK4 substeps per stored FIELDLINES step")
    parser.add_argument("--order", type=int, default=20, help="Simsopt Fourier order for MAKEGRID coils")
    parser.add_argument("--ppp", type=int, default=20, help="Simsopt quadrature points per Fourier period")
    parser.add_argument("--current-mode", choices=("scaled", "raw"), default="scaled")
    parser.add_argument("--simsopt-src", type=Path, default=None, help="Optional Simsopt source tree to prepend to sys.path")
    parser.add_argument("--reference-out", type=Path, required=True)
    parser.add_argument("--candidate-out", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--source-label", default=None, help="Portable source label stored in artifacts")
    parser.add_argument("--coils-label", default=None, help="Portable coils source label stored in artifacts")
    parser.add_argument("--input-label", default=None, help="Portable input source label stored in artifacts")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data = _load_h5(args.h5)
    lines = _line_ids(args.lines, data["R_lines"].shape[1])
    reference, candidate, metadata = trace_simsopt_candidate(
        args.h5,
        args.coils,
        args.input,
        lines=lines,
        nsections=int(args.nsections),
        substeps=int(args.substeps),
        order=int(args.order),
        ppp=int(args.ppp),
        current_mode=args.current_mode,
        simsopt_src=args.simsopt_src,
        source_label=args.source_label,
        coils_label=args.coils_label,
        input_label=args.input_label,
    )
    metadata["virtual_casing_jax_commit"] = _git_commit(ROOT)
    _write_npz(args.reference_out, reference, metadata | {"kind": "fieldlines_reference_subset"})
    _write_npz(args.candidate_out, candidate, metadata | {"kind": "fieldlines_simsopt_candidate"})
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
