"""Small VMEC-extender benchmark for CI-friendly local performance reports."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import jax.numpy as jnp

from virtual_casing_jax import ExteriorFieldConfig, VirtualCasingExteriorField, surface_field_from_vmec_jax


def _vmec_example_root():
    env_root = os.environ.get("VMEC_JAX_ROOT")
    candidates = []
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend([Path.cwd().parent / "vmec_jax", Path.cwd() / "vmec_jax"])
    for root in candidates:
        if (root / "examples" / "data" / "input.circular_tokamak").exists():
            return root
    return None


def _build_field(digits: int):
    import vmec_jax

    root = _vmec_example_root()
    if root is None:
        raise RuntimeError("Set VMEC_JAX_ROOT or place vmec_jax next to this checkout")
    example = vmec_jax.load_example("circular_tokamak", root=root)
    surface = surface_field_from_vmec_jax(example.state, example.static, example.indata, wout=example.wout)
    field = VirtualCasingExteriorField(
        surface,
        ExteriorFieldConfig(
            digits=digits,
            levels=((13, 13),),
            chunk_size=64,
            target_chunk_size=4,
            dtype="float64",
        ),
    )
    return surface, field


def run_benchmark(digits: int = 3) -> dict:
    t0 = time.perf_counter()
    surface, field = _build_field(digits)
    construct_seconds = time.perf_counter() - t0

    targets = jnp.asarray(
        [
            [2.45, 0.00, 0.00],
            [2.50, 0.05, 0.02],
            [2.55, 0.10, -0.03],
            [2.60, 0.15, 0.04],
            [2.65, 0.20, 0.00],
            [2.70, 0.25, 0.03],
            [2.75, 0.30, -0.02],
            [2.80, 0.35, 0.01],
        ],
        dtype=jnp.float64,
    )

    t0 = time.perf_counter()
    B = field.B_plasma_xyz(targets).block_until_ready()
    eval_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    grid = field.export_rphiz_grid(
        jnp.linspace(2.45, 2.8, 4),
        jnp.linspace(0.0, 0.5, 4),
        jnp.linspace(-0.05, 0.05, 4),
        chunk_size=8,
    )
    jnp.asarray(grid["absB"]).block_until_ready()
    grid_seconds = time.perf_counter() - t0

    Bn = jnp.sum(surface.B_total * surface.normal, axis=0)
    absB = jnp.linalg.norm(surface.B_total, axis=0)
    bdotn_rms = jnp.sqrt(jnp.mean(Bn * Bn)) / jnp.sqrt(jnp.mean(absB * absB))

    return {
        "case": "circular_tokamak",
        "digits": int(digits),
        "source_nphi": int(surface.gamma.shape[1]),
        "source_ntheta": int(surface.gamma.shape[2]),
        "construct_seconds": float(construct_seconds),
        "eval_seconds": float(eval_seconds),
        "eval_points": int(targets.shape[0]),
        "seconds_per_eval_point": float(eval_seconds / targets.shape[0]),
        "grid_seconds": float(grid_seconds),
        "grid_points": int(np.asarray(grid["absB"]).size),
        "seconds_per_grid_point": float(grid_seconds / np.asarray(grid["absB"]).size),
        "B_norm_l2": float(jnp.linalg.norm(B)),
        "B_dot_n_rms_normalized": float(bdotn_rms),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--digits", type=int, default=3)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    result = run_benchmark(digits=args.digits)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
