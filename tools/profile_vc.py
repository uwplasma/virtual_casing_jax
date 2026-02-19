#!/usr/bin/env python3
"""Profiling harness for VirtualCasingJAX using parity datasets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from virtual_casing_jax.surface_ops import rotate_toroidal, complete_vec_field
from virtual_casing_jax.virtual_casing import VirtualCasingJAX


DATA_DIR = Path(__file__).resolve().parents[1] / "tests" / "data"


def load_dump(base_path: Path):
    meta_path = base_path.with_suffix(".json")
    bin_path = base_path.with_suffix(".bin")
    if not meta_path.exists() or not bin_path.exists():
        raise FileNotFoundError(f"Missing dump files for {base_path}")
    with meta_path.open("r") as f:
        meta = json.load(f)
    dtype = np.float32 if meta["dtype"] == "float32" else np.float64
    shape = tuple(meta["shape"])
    data = np.fromfile(bin_path, dtype=dtype)
    return data.reshape(shape, order="C")


def infer_setup(prefix: str, kind: str):
    X = load_dump(DATA_DIR / f"{prefix}_setup_X")
    surf = load_dump(DATA_DIR / f"{prefix}_setup_surface_coord")
    B0_complete = load_dump(DATA_DIR / f"{prefix}_{kind}_B0_complete")

    src_nt = X.shape[1]
    src_np = X.shape[2]
    nfp_eff = B0_complete.shape[1] // src_nt
    half_period = surf.shape[1] == nfp_eff * (src_nt + 1)
    nfp = nfp_eff // 2 if half_period else nfp_eff
    return X, src_nt, src_np, nfp, nfp_eff, half_period, B0_complete


def reconstruct_B0(prefix: str, kind: str, src_nt: int, src_np: int, nfp: int, nfp_eff: int, half_period: bool, trg_nt: int):
    B0_complete_ref = load_dump(DATA_DIR / f"{prefix}_{kind}_B0_complete")
    B0_complete = jnp.asarray(B0_complete_ref)

    dtheta = 0.0
    if half_period:
        dtheta = np.pi * (1.0 / (nfp * trg_nt * 2) - 1.0 / (nfp * src_nt * 2))
        B0_complete = rotate_toroidal(B0_complete, nfp_eff * src_nt, src_np, -dtheta)

    B0 = B0_complete[:, :src_nt, :]
    _ = complete_vec_field(B0, False, half_period, nfp, src_nt, src_np, dtheta)
    return np.asarray(B0)


def get_digits(prefix: str):
    return {
        "case_vc": 5,
        "case_vc_int": 5,
        "case_vc_large": 6,
        "case_vc_w7x": 6,
        "case_vc_w7x_large": 6,
        "case_simsopt": 6,
        "case_simsopt_int": 6,
        "case_simsopt_large": 6,
    }.get(prefix, 6)


def profile(args: argparse.Namespace):
    kind_map = {
        "B": "computeB",
        "GradB": "computeGradB",
        "Boff": "computeBOff",
        "GradBoff": "computeGradBOff",
    }
    kind = kind_map[args.op]

    X, src_nt, src_np, nfp, nfp_eff, half_period, _ = infer_setup(args.case, kind)

    if args.op == "B":
        ref = load_dump(DATA_DIR / f"{args.case}_computeB_Bvc")
        quad = load_dump(DATA_DIR / f"{args.case}_computeB_quad_coord")
        trg_nt, trg_np = ref.shape[1], ref.shape[2]
        quad_nt, quad_np = quad.shape[1], quad.shape[2]
    elif args.op == "GradB":
        ref = load_dump(DATA_DIR / f"{args.case}_computeGradB_gradBvc")
        quad = load_dump(DATA_DIR / f"{args.case}_computeGradB_quad_coord")
        trg_nt, trg_np = ref.shape[2], ref.shape[3]
        quad_nt, quad_np = quad.shape[1], quad.shape[2]
    elif args.op == "Boff":
        Xt = load_dump(DATA_DIR / f"{args.case}_computeBOff_Xt")
        trg_nt, trg_np = None, None
        quad_nt = quad_np = None
    elif args.op == "GradBoff":
        Xt = load_dump(DATA_DIR / f"{args.case}_computeGradBOff_Xt")
        trg_nt, trg_np = None, None
        quad_nt = quad_np = None
    else:
        raise ValueError(f"Unknown op: {args.op}")

    digits = get_digits(args.case)
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
        trg_nt or 1,
        trg_np or 1,
    )

    if args.op in ("B", "GradB"):
        B0 = reconstruct_B0(args.case, kind, src_nt, src_np, nfp, nfp_eff, half_period, trg_nt)
    else:
        B0 = reconstruct_B0(args.case, kind, src_nt, src_np, nfp, nfp_eff, half_period, 1)

    if args.op == "B":
        fn = vc.compute_external_B if args.mode == "external" else vc.compute_internal_B
        if args.jit:
            fn = vc.compute_external_B_jit if args.mode == "external" else vc.compute_internal_B_jit
        call = lambda: fn(
            B0,
            quad_nt=quad_nt,
            quad_np=quad_np,
            digits=digits,
            chunk_size=args.chunk_size,
            target_chunk_size=args.target_chunk_size,
            pou_dtype=args.pou_dtype,
            patch_dtype=args.patch_dtype,
            interp_block_size=args.interp_block_size,
            remat=args.remat,
        )
    elif args.op == "GradB":
        fn = vc.compute_external_gradB if args.mode == "external" else vc.compute_internal_gradB
        if args.jit:
            fn = vc.compute_external_gradB_jit if args.mode == "external" else vc.compute_internal_gradB_jit
        call = lambda: fn(
            B0,
            quad_nt=quad_nt,
            quad_np=quad_np,
            digits=digits,
            chunk_size=args.chunk_size,
            target_chunk_size=args.target_chunk_size,
            pou_dtype=args.pou_dtype,
            patch_dtype=args.patch_dtype,
            interp_block_size=args.interp_block_size,
            remat=args.remat,
        )
    elif args.op == "Boff":
        fn = vc.compute_external_B_offsurf if args.mode == "external" else vc.compute_internal_B_offsurf
        call = lambda: fn(
            B0,
            X_trg=Xt,
            digits=digits,
            max_Nt=args.max_nt,
            max_Np=args.max_np,
            chunk_size=args.chunk_size,
            target_chunk_size=args.target_chunk_size,
        )
    else:
        fn = vc.compute_external_gradB_offsurf if args.mode == "external" else vc.compute_internal_gradB_offsurf
        call = lambda: fn(
            B0,
            X_trg=Xt,
            digits=digits,
            max_Nt=args.max_nt,
            max_Np=args.max_np,
            chunk_size=args.chunk_size,
            target_chunk_size=args.target_chunk_size,
        )

    # Warmup (compile / cache)
    for _ in range(args.warmup):
        out = call()
        jax.block_until_ready(out)

    if args.trace_dir:
        jax.profiler.start_trace(args.trace_dir)

    for _ in range(args.repeat):
        out = call()
        jax.block_until_ready(out)

    if args.trace_dir:
        jax.profiler.stop_trace()


def main():
    parser = argparse.ArgumentParser(description="Profile VirtualCasingJAX on parity datasets")
    parser.add_argument("--case", default="case_vc", help="Dataset prefix (case_vc, case_simsopt, case_vc_w7x)")
    parser.add_argument("--op", choices=["B", "GradB", "Boff", "GradBoff"], default="B")
    parser.add_argument("--mode", choices=["external", "internal"], default="external")
    parser.add_argument("--jit", action="store_true", help="Use JIT wrapper when available")
    parser.add_argument("--trace-dir", default="", help="Directory for JAX profiler trace")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--chunk-size", type=str, default="auto")
    parser.add_argument("--target-chunk-size", type=str, default="auto")
    parser.add_argument("--pou-dtype", type=str, default=None)
    parser.add_argument("--patch-dtype", type=str, default=None)
    parser.add_argument("--interp-block-size", type=str, default="auto")
    parser.add_argument("--remat", dest="remat", action="store_true")
    parser.add_argument("--no-remat", dest="remat", action="store_false")
    parser.add_argument("--max-nt", type=int, default=-1)
    parser.add_argument("--max-np", type=int, default=-1)
    parser.set_defaults(remat=None)
    args = parser.parse_args()

    def _parse_chunk(val):
        if val is None:
            return None
        if isinstance(val, str) and val.lower() == "auto":
            return "auto"
        return int(val)

    args.chunk_size = _parse_chunk(args.chunk_size)
    args.target_chunk_size = _parse_chunk(args.target_chunk_size)
    if args.interp_block_size is None:
        args.interp_block_size = None
    elif isinstance(args.interp_block_size, str) and args.interp_block_size.lower() == "auto":
        args.interp_block_size = "auto"
    else:
        args.interp_block_size = int(args.interp_block_size)

    profile(args)


if __name__ == "__main__":
    main()
