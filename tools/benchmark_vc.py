#!/usr/bin/env python3
"""Reproducible correctness and performance benchmarks for VirtualCasingJAX."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax
import numpy as np

from profile_vc import DATA_DIR, get_digits, infer_setup, load_dump, reconstruct_B0
from virtual_casing_jax.virtual_casing import VirtualCasingJAX

QUICK_CASES = [
    dict(case="case_vc", op="B", mode="external", jit=False, quadrature="explicit"),
    dict(case="case_vc", op="B", mode="external", jit=True, quadrature="auto"),
    dict(case="case_vc_int", op="B", mode="internal", jit=False, quadrature="explicit"),
    dict(case="case_vc_int", op="B", mode="internal", jit=True, quadrature="auto"),
    dict(case="case_vc", op="GradB", mode="external", jit=False, quadrature="explicit"),
    dict(case="case_vc", op="GradB", mode="external", jit=True, quadrature="auto"),
    dict(case="case_vc_int", op="GradB", mode="internal", jit=False, quadrature="explicit"),
    dict(case="case_vc_int", op="GradB", mode="internal", jit=True, quadrature="auto"),
    dict(case="case_vc", op="Boff", mode="external", jit=False, quadrature="adaptive"),
    dict(case="case_vc_int", op="Boff", mode="internal", jit=False, quadrature="adaptive"),
    dict(case="case_vc", op="GradBoff", mode="external", jit=False, quadrature="base"),
]

FULL_EXTRA_CASES = [
    dict(case="case_vc_large", op="B", mode="external", jit=True, quadrature="explicit"),
    dict(
        case="case_vc_large",
        op="GradB",
        mode="external",
        jit=True,
        quadrature="explicit",
        scan_targets=True,
    ),
    dict(case="case_vc_w7x_large", op="B", mode="external", jit=True, quadrature="explicit"),
    dict(
        case="case_vc_w7x_large",
        op="GradB",
        mode="external",
        jit=True,
        quadrature="explicit",
        scan_targets=True,
    ),
    dict(case="case_simsopt", op="B", mode="external", jit=True, quadrature="explicit"),
    dict(case="case_simsopt", op="GradB", mode="external", jit=True, quadrature="explicit"),
    dict(case="case_vc_large", op="Boff", mode="external", jit=False, quadrature="adaptive"),
    dict(case="case_vc_large", op="GradBoff", mode="external", jit=False, quadrature="base"),
]


def _kind(op: str) -> str:
    return {
        "B": "computeB",
        "GradB": "computeGradB",
        "Boff": "computeBOff",
        "GradBoff": "computeGradBOff",
    }[op]


def _reference_name(op: str) -> str:
    return {
        "B": "computeB_Bvc",
        "GradB": "computeGradB_gradBvc",
        "Boff": "computeBOff_Bvc",
        "GradBoff": "computeGradBOff_gradBvc",
    }[op]


def _tolerance(case: str, op: str, mode: str) -> float | None:
    if op == "B":
        if mode == "internal":
            return 3e-4 if case == "case_vc_int" else 8e-4
        return {
            "case_vc": 3e-4,
            "case_vc_large": 6e-4,
            "case_simsopt": 6e-4,
            "case_simsopt_large": 8e-4,
            "case_vc_w7x": 8e-4,
            "case_vc_w7x_large": 1.2e-3,
        }.get(case, 1.2e-3)
    if op == "GradB":
        if mode == "internal":
            return None
        return {
            "case_vc": 5e-3,
            "case_vc_large": 7e-3,
            "case_simsopt": 6e-3,
            "case_simsopt_large": 8e-3,
            "case_vc_w7x": 2.5e-2,
            "case_vc_w7x_large": 3e-2,
        }.get(case, 3e-2)
    if op == "Boff":
        return 1.5e-3 if "w7x" in case else 8e-4
    if op == "GradBoff":
        return 1.8e-2 if "w7x" in case else 8e-3
    return None


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _git_state() -> dict:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        ).stdout.strip()
    )
    return {"commit": commit or None, "dirty": dirty}


def _environment() -> dict:
    device = jax.devices()[0]
    return {
        "python": sys.version.split()[0],
        "virtual_casing_jax": importlib.metadata.version("virtual-casing-jax"),
        "jax": importlib.metadata.version("jax"),
        "jaxlib": importlib.metadata.version("jaxlib"),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "backend": jax.default_backend(),
        "device": str(device),
        "x64_enabled": bool(jax.config.x64_enabled),
        "git": _git_state(),
    }


def _dataset_provenance() -> dict:
    manifest = DATA_DIR / "provenance.json"
    if not manifest.exists():
        return {
            "status": "legacy-unverified",
            "message": "No provenance manifest is present; reference commits are not inferred.",
        }
    return {"status": "verified-manifest", "manifest": json.loads(manifest.read_text())}


def _load_case(spec: dict):
    case = spec["case"]
    op = spec["op"]
    kind = _kind(op)
    X, src_nt, src_np, nfp, nfp_eff, half_period, _ = infer_setup(case, kind)
    digits = get_digits(case)

    reference_path = DATA_DIR / f"{case}_{_reference_name(op)}"
    reference = None
    if reference_path.with_suffix(".bin").exists() and not (
        op == "GradB" and spec["mode"] == "internal"
    ):
        reference = load_dump(reference_path)

    if op == "B":
        trg_nt, trg_np = reference.shape[1:]
        quad = load_dump(DATA_DIR / f"{case}_computeB_quad_coord")
        explicit_quad = (quad.shape[1], quad.shape[2])
    elif op == "GradB":
        shape_source = reference
        if shape_source is None:
            shape_source = load_dump(DATA_DIR / f"{case}_computeGradB_gradBvc")
        trg_nt, trg_np = shape_source.shape[2:]
        quad = load_dump(DATA_DIR / f"{case}_computeGradB_quad_coord")
        explicit_quad = (quad.shape[1], quad.shape[2])
    else:
        Xt = load_dump(DATA_DIR / f"{case}_{kind}_Xt")
        b_surface = DATA_DIR / f"{case}_computeB_Bvc"
        if b_surface.with_suffix(".bin").exists():
            surface_shape = load_dump(b_surface).shape
            trg_nt, trg_np = surface_shape[-2:]
        else:
            trg_nt = trg_np = 1
        explicit_quad = (None, None)

    vc = VirtualCasingJAX()
    vc.setup(
        digits,
        nfp,
        half_period,
        src_nt,
        src_np,
        X,
        src_nt,
        src_np,
        trg_nt,
        trg_np,
    )
    B0 = reconstruct_B0(
        case, kind, src_nt, src_np, nfp, nfp_eff, half_period, trg_nt
    )
    return vc, B0, reference, explicit_quad, locals().get("Xt"), digits


def _make_call(spec: dict):
    vc, B0, reference, explicit_quad, Xt, digits = _load_case(spec)
    op = spec["op"]
    mode = spec["mode"]
    use_explicit = spec["quadrature"] == "explicit"
    quad_nt, quad_np = explicit_quad if use_explicit else (None, None)
    common = dict(digits=digits, chunk_size="auto", target_chunk_size="auto")

    if op == "B":
        fn = vc.compute_external_B if mode == "external" else vc.compute_internal_B
        if spec["jit"]:
            fn = (
                vc.compute_external_B_jit
                if mode == "external"
                else vc.compute_internal_B_jit
            )
        return (
            lambda: fn(B0, quad_nt=quad_nt, quad_np=quad_np, **common),
            reference,
        )
    if op == "GradB":
        fn = (
            vc.compute_external_gradB
            if mode == "external"
            else vc.compute_internal_gradB
        )
        if spec["jit"]:
            fn = (
                vc.compute_external_gradB_jit
                if mode == "external"
                else vc.compute_internal_gradB_jit
            )
        return (
            lambda: fn(
                B0,
                quad_nt=quad_nt,
                quad_np=quad_np,
                scan_targets=spec.get("scan_targets", False),
                **common,
            ),
            reference,
        )
    if op == "Boff":
        fn = (
            vc.compute_external_B_offsurf
            if mode == "external"
            else vc.compute_internal_B_offsurf
        )
        return lambda: fn(B0, X_trg=Xt, max_levels=6, **common), reference
    fn = (
        vc.compute_external_gradB_offsurf
        if mode == "external"
        else vc.compute_internal_gradB_offsurf
    )
    return lambda: fn(B0, X_trg=Xt, adaptive=False, **common), reference


def _worker(spec: dict, repeats: int) -> dict:
    call, reference = _make_call(spec)
    start = time.perf_counter()
    output = call()
    jax.block_until_ready(output)
    first_seconds = time.perf_counter() - start

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        output = call()
        jax.block_until_ready(output)
        samples.append(time.perf_counter() - start)

    output = np.asarray(output)
    finite = bool(np.all(np.isfinite(output)))
    metrics = {
        "finite": finite,
        "max_abs_error": None,
        "scaled_max_relative_error": None,
        "normalized_l2_error": None,
    }
    if reference is not None:
        reference = np.asarray(reference)
        error = output - reference
        metrics.update(
            max_abs_error=float(np.max(np.abs(error))),
            scaled_max_relative_error=float(
                np.max(np.abs(error)) / (np.max(np.abs(reference)) + 1e-14)
            ),
            normalized_l2_error=float(
                np.linalg.norm(error) / (np.linalg.norm(reference) + 1e-14)
            ),
        )

    tolerance = _tolerance(spec["case"], spec["op"], spec["mode"])
    correct = finite and (
        reference is None
        or tolerance is None
        or metrics["normalized_l2_error"] < tolerance
    )
    return {
        "spec": spec,
        "correct": bool(correct),
        "tolerance": tolerance,
        "timing": {
            "first_seconds": first_seconds,
            "steady_median_seconds": float(np.median(samples)),
            "steady_p95_seconds": float(np.percentile(samples, 95)),
            "samples_seconds": samples,
        },
        "peak_rss_bytes": _peak_rss_bytes(),
        "metrics": metrics,
        "environment": _environment(),
    }


def _case_id(spec: dict) -> str:
    mode = spec["mode"]
    compiled = "jit" if spec["jit"] else "eager"
    return f"{spec['case']}:{mode}:{spec['op']}:{compiled}:{spec['quadrature']}"


def _run_isolated(spec: dict, repeats: int) -> dict:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-json",
        json.dumps(spec, separators=(",", ":")),
        "--repeats",
        str(repeats),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        env=dict(os.environ),
        check=False,
    )
    if completed.returncode != 0:
        return {
            "spec": spec,
            "correct": False,
            "error": completed.stderr.strip() or completed.stdout.strip(),
        }
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _write_markdown(report: dict, path: Path) -> None:
    lines = [
        f"# VirtualCasingJAX CPU Benchmark — {report['created_utc'][:10]}",
        "",
        f"Suite: `{report['suite']}`  ",
        f"Backend: `{report['backend']}`  ",
        f"GPU coverage: `{report['gpu']['status']}` ({report['gpu']['reason']})  ",
        f"Overall correctness: **{'PASS' if report['correct'] else 'FAIL'}**",
        "",
        "| Case | Correct | First (s) | Median (s) | p95 (s) | Rel. L2 | Peak RSS (MiB) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in report["results"]:
        timing = result.get("timing", {})
        metrics = result.get("metrics", {})
        lines.append(
            "| {case} | {correct} | {first} | {median} | {p95} | {rel} | {rss} |".format(
                case=_case_id(result["spec"]),
                correct="PASS" if result.get("correct") else "FAIL",
                first=f"{timing.get('first_seconds', float('nan')):.6g}",
                median=f"{timing.get('steady_median_seconds', float('nan')):.6g}",
                p95=f"{timing.get('steady_p95_seconds', float('nan')):.6g}",
                rel=(
                    "n/a"
                    if metrics.get("normalized_l2_error") is None
                    else f"{metrics['normalized_l2_error']:.3e}"
                ),
                rss=f"{result.get('peak_rss_bytes', 0) / 2**20:.1f}",
            )
        )
        if result.get("error"):
            lines.extend(["", f"- `{_case_id(result['spec'])}`: {result['error']}"])
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("quick", "full"), default="quick")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmark_results")
    parser.add_argument("--worker-json", default="")
    args = parser.parse_args()

    if args.worker_json:
        print(json.dumps(_worker(json.loads(args.worker_json), args.repeats)))
        return 0

    cases = list(QUICK_CASES)
    if args.suite == "full":
        cases.extend(FULL_EXTRA_CASES)
    results = []
    for index, spec in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {_case_id(spec)}", flush=True)
        results.append(_run_isolated(spec, args.repeats))

    created = datetime.now(timezone.utc)
    report = {
        "schema_version": 1,
        "created_utc": created.isoformat(),
        "suite": args.suite,
        "backend": "cpu" if jax.default_backend() == "cpu" else jax.default_backend(),
        "gpu": (
            {"status": "run", "reason": "benchmark backend is GPU"}
            if jax.default_backend() == "gpu"
            else {
                "status": "skipped",
                "reason": "no GPU backend was selected for this run",
            }
        ),
        "dataset_provenance": _dataset_provenance(),
        "correct": all(result.get("correct", False) for result in results),
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = created.strftime("%Y%m%dT%H%M%SZ") + f"_{args.suite}"
    json_path = args.output_dir / f"{stem}.json"
    md_path = args.output_dir / f"{stem}.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    _write_markdown(report, md_path)
    print(json_path)
    print(md_path)
    return 0 if report["correct"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
