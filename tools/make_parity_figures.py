#!/usr/bin/env python3
"""Generate parity figures for documentation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp

from virtual_casing_jax.virtual_casing import VirtualCasingJAX
from virtual_casing_jax.surface_ops import rotate_toroidal, complete_vec_field

# Allow direct import of dump_io when tools are not a package.
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "tests"))
from dump_io import load_dump  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "tests" / "data"
OUT_DIR = ROOT / "docs" / "_static"


def _infer_setup(prefix: str, kind: str):
    X = load_dump(DATA_DIR / f"{prefix}_setup_X")
    surf = load_dump(DATA_DIR / f"{prefix}_setup_surface_coord")
    B0_complete = load_dump(DATA_DIR / f"{prefix}_{kind}_B0_complete")

    src_nt = X.shape[1]
    src_np = X.shape[2]
    nfp_eff = B0_complete.shape[1] // src_nt
    half_period = surf.shape[1] == nfp_eff * (src_nt + 1)
    nfp = nfp_eff // 2 if half_period else nfp_eff
    return X, src_nt, src_np, nfp, nfp_eff, half_period


def _reconstruct_B0(prefix: str, kind: str, src_nt: int, src_np: int, nfp: int, nfp_eff: int, half_period: bool, trg_nt: int):
    B0_complete_ref = load_dump(DATA_DIR / f"{prefix}_{kind}_B0_complete")
    B0_complete_ref = jnp.asarray(B0_complete_ref)
    B0_complete = B0_complete_ref

    dtheta = 0.0
    if half_period:
        dtheta = np.pi * (1.0 / (nfp * trg_nt * 2) - 1.0 / (nfp * src_nt * 2))
        B0_complete = rotate_toroidal(B0_complete, nfp_eff * src_nt, src_np, -dtheta)

    B0 = B0_complete[:, :src_nt, :]

    B0_re = complete_vec_field(B0, False, half_period, nfp, src_nt, src_np, dtheta)
    num = np.linalg.norm(np.asarray(B0_re) - np.asarray(B0_complete_ref))
    den = np.linalg.norm(np.asarray(B0_complete_ref)) + 1e-14
    if num / den >= 1e-4:
        raise RuntimeError("B0 reconstruction sanity check failed.")

    return np.asarray(B0)


def _plot_parity(ref, val, title: str, out_path: Path):
    ref_flat = ref.reshape(-1)
    val_flat = val.reshape(-1)
    rel = np.abs(val_flat - ref_flat) / (np.abs(ref_flat) + 1e-14)
    rel_l2 = np.linalg.norm(val_flat - ref_flat) / (np.linalg.norm(ref_flat) + 1e-14)

    rng = np.random.default_rng(0)
    sample = min(ref_flat.size, 5000)
    idx = rng.choice(ref_flat.size, size=sample, replace=False)

    lo = np.percentile(ref_flat, 1)
    hi = np.percentile(ref_flat, 99)
    pad = 0.05 * max(1.0, abs(hi - lo))
    lo -= pad
    hi += pad

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax0, ax1 = axes

    ax0.scatter(ref_flat[idx], val_flat[idx], s=6, alpha=0.5, edgecolors="none")
    ax0.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    ax0.set_xlim(lo, hi)
    ax0.set_ylim(lo, hi)
    ax0.set_xlabel("C++ reference")
    ax0.set_ylabel("JAX")
    ax0.set_title("Scatter (sampled)")

    log_rel = np.log10(rel + 1e-16)
    ax1.hist(log_rel, bins=50, color="#4c78a8", alpha=0.85)
    ax1.set_xlabel("log10(relative error)")
    ax1.set_ylabel("count")
    ax1.set_title(f"rel L2 = {rel_l2:.2e}")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _compute_case(prefix: str):
    vc = VirtualCasingJAX()

    # ComputeB parity
    X, src_nt, src_np, nfp, nfp_eff, half_period = _infer_setup(prefix, "computeB")
    Bvc_ref = load_dump(DATA_DIR / f"{prefix}_computeB_Bvc")
    quad_coord = load_dump(DATA_DIR / f"{prefix}_computeB_quad_coord")
    trg_nt = Bvc_ref.shape[1]
    trg_np = Bvc_ref.shape[2]
    quad_nt = quad_coord.shape[1]
    quad_np = quad_coord.shape[2]
    B0 = _reconstruct_B0(prefix, "computeB", src_nt, src_np, nfp, nfp_eff, half_period, trg_nt)

    digits = 5
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

    Bvc = vc.compute_external_B(B0, quad_nt=quad_nt, quad_np=quad_np, digits=digits, chunk_size=1024)
    _plot_parity(
        np.asarray(Bvc_ref),
        np.asarray(Bvc),
        f"ComputeB parity ({prefix})",
        OUT_DIR / f"parity_computeB_{prefix}.png",
    )

    # ComputeGradB parity
    Xg, src_ntg, src_npg, nfpg, nfp_effg, half_periodg = _infer_setup(prefix, "computeGradB")
    gradB_ref = load_dump(DATA_DIR / f"{prefix}_computeGradB_gradBvc")
    quad_coord_g = load_dump(DATA_DIR / f"{prefix}_computeGradB_quad_coord")
    trg_ntg = gradB_ref.shape[2]
    trg_npg = gradB_ref.shape[3]
    quad_ntg = quad_coord_g.shape[1]
    quad_npg = quad_coord_g.shape[2]
    B0g = _reconstruct_B0(prefix, "computeGradB", src_ntg, src_npg, nfpg, nfp_effg, half_periodg, trg_ntg)

    vc.setup(
        digits,
        nfpg,
        half_periodg,
        src_ntg,
        src_npg,
        Xg,
        src_ntg,
        src_npg,
        trg_ntg,
        trg_npg,
    )
    gradB = vc.compute_external_gradB(
        B0g,
        quad_nt=quad_ntg,
        quad_np=quad_npg,
        digits=digits,
        hedgehog_order=8,
        chunk_size=1024,
    )
    _plot_parity(
        np.asarray(gradB_ref),
        np.asarray(gradB),
        f"ComputeGradB parity ({prefix})",
        OUT_DIR / f"parity_computeGradB_{prefix}.png",
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _compute_case("case_simsopt")


if __name__ == "__main__":
    main()
