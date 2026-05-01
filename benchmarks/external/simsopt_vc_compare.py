"""Compare virtual_casing_jax against upstream SIMSOPT virtual casing.

The default case uses the SIMSOPT-derived finite-beta QH VMEC/BNORM assets
shipped with the test suite. The benchmark reports direct SIMSOPT parity and
the BNORM normal-field residual used by the legacy SIMSOPT validation tests.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_WOUT = (
    ROOT
    / "tests"
    / "test_files"
    / "wout_20220102-01-053-003_QH_nfp4_aspect6p5_beta0p05_iteratedWithSfincs_reference.nc"
)
DEFAULT_BNORM = (
    ROOT
    / "tests"
    / "test_files"
    / "bnorm.20220102-01-053-003_QH_nfp4_aspect6p5_beta0p05_iteratedWithSfincs"
)


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


def _package_root(module_file: str | None) -> Path | None:
    if not module_file:
        return None
    path = Path(module_file).resolve()
    for parent in [path, *path.parents]:
        if (parent / ".git").exists():
            return parent
    return None


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


def bnorm_normal_field(bnorm_path: Path, vmec_wout: Any, trgt_phi: np.ndarray, trgt_theta: np.ndarray) -> np.ndarray:
    """Evaluate SIMSOPT/BNORM sine-series normal field on normalized grids."""

    nfp = int(vmec_wout.nfp)
    theta, phi = np.meshgrid(2 * np.pi * trgt_theta, 2 * np.pi * trgt_phi)
    normal_field = np.zeros((len(trgt_phi), len(trgt_theta)))

    with bnorm_path.open() as f:
        for line in f:
            splitline = line.split()
            if len(splitline) != 3:
                continue
            m = int(splitline[0])
            n = int(splitline[1])
            amplitude = float(splitline[2])
            normal_field += amplitude * np.sin(m * theta + n * nfp * phi)

    # This is the normalization used in SIMSOPT's virtual-casing tests. The
    # two-point radial extrapolation matches the boundary-current convention
    # used when generating the bundled BNORM reference file.
    curpol = (2 * np.pi / nfp) * (1.5 * vmec_wout.bsubvmnc[0, -1] - 0.5 * vmec_wout.bsubvmnc[0, -2])
    return normal_field * curpol


def _threshold_failures(metrics: dict[str, Any], args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    if metrics["external_normal_relative_l2"] > args.max_external_normal_relative_l2:
        failures.append(
            "external_normal_relative_l2="
            f"{metrics['external_normal_relative_l2']:.6g} > {args.max_external_normal_relative_l2:.6g}"
        )
    if metrics["external_vector_relative_l2"] > args.max_external_vector_relative_l2:
        failures.append(
            "external_vector_relative_l2="
            f"{metrics['external_vector_relative_l2']:.6g} > {args.max_external_vector_relative_l2:.6g}"
        )
    bnorm_max = metrics.get("jax_bnorm_max_abs")
    if bnorm_max is not None and bnorm_max > args.max_bnorm_max_abs:
        failures.append(f"jax_bnorm_max_abs={bnorm_max:.6g} > {args.max_bnorm_max_abs:.6g}")
    return failures


def _positive_int_or_auto(value: str) -> int | None:
    if value.lower() == "auto":
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer or 'auto'")
    return parsed


def run_compare(args: argparse.Namespace) -> dict[str, Any]:
    from simsopt import __file__ as simsopt_file
    from simsopt.mhd import VirtualCasing as SimsoptVirtualCasing
    from simsopt.mhd import Vmec

    from virtual_casing_jax import VirtualCasing as JaxVirtualCasing

    wout = args.wout.resolve()
    if not wout.exists():
        raise FileNotFoundError(f"Missing VMEC wout file: {wout}")

    common_kwargs = dict(
        src_nphi=args.src_nphi,
        src_ntheta=args.src_ntheta,
        trgt_nphi=args.trgt_nphi,
        trgt_ntheta=args.trgt_ntheta,
        use_stellsym=args.use_stellsym,
        digits=args.digits,
        filename=None,
    )

    t0 = time.perf_counter()
    simsopt_vc = SimsoptVirtualCasing.from_vmec(str(wout), **common_kwargs)
    simsopt_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    jax_vc = JaxVirtualCasing.from_vmec(str(wout), **common_kwargs)
    jax_seconds = time.perf_counter() - t0

    external_normal_diff = jax_vc.B_external_normal - simsopt_vc.B_external_normal
    external_vector_diff = jax_vc.B_external - simsopt_vc.B_external

    metrics: dict[str, Any] = {
        "status": "completed",
        "case": wout.name,
        "wout": str(wout),
        "src_nphi": int(jax_vc.src_nphi),
        "src_ntheta": int(jax_vc.src_ntheta),
        "src_ntheta_requested": "auto" if args.src_ntheta is None else int(args.src_ntheta),
        "trgt_nphi": int(jax_vc.trgt_nphi),
        "trgt_ntheta": int(jax_vc.trgt_ntheta),
        "digits": int(args.digits),
        "use_stellsym": bool(args.use_stellsym),
        "nfp": int(simsopt_vc.nfp),
        "simsopt_seconds": float(simsopt_seconds),
        "virtual_casing_jax_seconds": float(jax_seconds),
        "external_normal_relative_l2": _relative_l2(jax_vc.B_external_normal, simsopt_vc.B_external_normal),
        "external_normal_max_abs": float(np.max(np.abs(external_normal_diff))),
        "external_normal_max_relative_to_ref_max": _max_relative_to_reference(
            jax_vc.B_external_normal, simsopt_vc.B_external_normal
        ),
        "external_vector_relative_l2": _relative_l2(jax_vc.B_external, simsopt_vc.B_external),
        "external_vector_max_abs": float(np.max(np.abs(external_vector_diff))),
        "external_vector_max_relative_to_ref_max": _max_relative_to_reference(jax_vc.B_external, simsopt_vc.B_external),
        "boundary_total_relative_l2": _relative_l2(jax_vc.B_total, simsopt_vc.B_total),
        "geometry_relative_l2": _relative_l2(jax_vc.gamma, simsopt_vc.gamma),
        "virtual_casing_jax_commit": _git_commit(ROOT),
        "simsopt_commit": _git_commit(_package_root(simsopt_file) or Path(".")),
    }

    if args.bnorm is not None:
        bnorm = args.bnorm.resolve()
        if not bnorm.exists():
            raise FileNotFoundError(f"Missing BNORM reference file: {bnorm}")
        vmec = Vmec(str(wout))
        vmec.run()
        bnorm_ref = bnorm_normal_field(bnorm, vmec.wout, simsopt_vc.trgt_phi, simsopt_vc.trgt_theta)
        metrics.update(
            {
                "bnorm": str(bnorm),
                "simsopt_bnorm_relative_l2": _relative_l2(simsopt_vc.B_external_normal, bnorm_ref),
                "simsopt_bnorm_max_abs": float(np.max(np.abs(simsopt_vc.B_external_normal - bnorm_ref))),
                "jax_bnorm_relative_l2": _relative_l2(jax_vc.B_external_normal, bnorm_ref),
                "jax_bnorm_max_abs": float(np.max(np.abs(jax_vc.B_external_normal - bnorm_ref))),
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
    parser.add_argument("--wout", type=Path, default=DEFAULT_WOUT)
    parser.add_argument("--bnorm", type=Path, default=DEFAULT_BNORM)
    parser.add_argument("--src-nphi", type=int, default=25)
    parser.add_argument("--src-ntheta", type=_positive_int_or_auto, default=None, metavar="N|auto")
    parser.add_argument("--trgt-nphi", type=int, default=32)
    parser.add_argument("--trgt-ntheta", type=int, default=32)
    parser.add_argument("--digits", type=int, default=6)
    parser.add_argument("--no-stellsym", dest="use_stellsym", action="store_false")
    parser.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "external" / "simsopt_vc_compare.json")
    parser.add_argument("--max-external-normal-relative-l2", type=float, default=1e-4)
    parser.add_argument("--max-external-vector-relative-l2", type=float, default=5e-5)
    parser.add_argument("--max-bnorm-max-abs", type=float, default=6.5e-3)
    parser.add_argument(
        "--skip-if-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write a skipped JSON report instead of failing when optional external dependencies are missing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        metrics = run_compare(args)
    except ImportError as exc:
        if not args.skip_if_missing:
            raise
        metrics = {"status": "skipped", "reason": f"missing optional dependency: {exc}"}
    text = json.dumps(metrics, indent=2, sort_keys=True)
    if args.out is not None:
        _write_json(args.out, metrics)
    print(text)
    if metrics.get("status") == "skipped":
        return 0
    return 0 if metrics.get("passed_thresholds") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
