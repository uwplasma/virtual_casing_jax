from pathlib import Path

import numpy as np

from virtual_casing_jax.virtual_casing import VirtualCasingJAX
from virtual_casing_jax.surface_ops import rotate_toroidal, complete_vec_field

# Allow direct import of dump_io when tests are not a package.
import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).parent))
from dump_io import load_dump  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"


def _infer_setup(prefix: str):
    X = load_dump(DATA_DIR / f"{prefix}_setup_X")
    surf = load_dump(DATA_DIR / f"{prefix}_setup_surface_coord")
    B0_complete = load_dump(DATA_DIR / f"{prefix}_computeBOff_B0_complete")

    src_nt = X.shape[1]
    src_np = X.shape[2]
    nfp_eff = B0_complete.shape[1] // src_nt
    half_period = surf.shape[1] == nfp_eff * (src_nt + 1)
    nfp = nfp_eff // 2 if half_period else nfp_eff
    return X, src_nt, src_np, nfp, nfp_eff, half_period


def _reconstruct_B0(prefix: str, src_nt: int, src_np: int, nfp: int, nfp_eff: int, half_period: bool, trg_nt: int):
    B0_complete_ref = load_dump(DATA_DIR / f"{prefix}_computeBOff_B0_complete")
    B0_complete = B0_complete_ref

    dtheta = 0.0
    if half_period:
        dtheta = np.pi * (1.0 / (nfp * trg_nt * 2) - 1.0 / (nfp * src_nt * 2))
        B0_complete = rotate_toroidal(B0_complete, nfp_eff * src_nt, src_np, -dtheta)

    B0 = B0_complete[:, :src_nt, :]

    B0_re = complete_vec_field(B0, False, half_period, nfp, src_nt, src_np, dtheta)
    num = np.linalg.norm(np.asarray(B0_re) - np.asarray(B0_complete_ref))
    den = np.linalg.norm(np.asarray(B0_complete_ref)) + 1e-14
    assert num / den < 1e-4

    return np.asarray(B0)


def test_gradb_offsurf_adaptive_matches_fd():
    prefix = "case_vc"
    X, src_nt, src_np, nfp, nfp_eff, half_period = _infer_setup(prefix)
    Xt = load_dump(DATA_DIR / f"{prefix}_computeBOff_Xt")
    Bvc_trg = load_dump(DATA_DIR / f"{prefix}_computeB_Bvc")
    trg_nt, trg_np = Bvc_trg.shape[1], Bvc_trg.shape[2]

    B0 = _reconstruct_B0(prefix, src_nt, src_np, nfp, nfp_eff, half_period, trg_nt)

    digits = 5
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

    Xt = np.asarray(Xt)
    gradB = vc.compute_external_gradB_offsurf(
        B0,
        X_trg=Xt,
        digits=digits,
        max_Nt=-1,
        max_Np=-1,
        adaptive=True,
        chunk_size=1024,
    )
    gradB = np.asarray(gradB)

    eps = 1e-5
    ntrg = Xt.shape[1]
    grad_fd = np.zeros_like(gradB)
    for i in range(3):
        Xt_pos = Xt.copy()
        Xt_neg = Xt.copy()
        Xt_pos[i] += eps
        Xt_neg[i] -= eps
        B_pos = vc.compute_external_B_offsurf(
            B0,
            X_trg=Xt_pos,
            digits=digits,
            max_Nt=-1,
            max_Np=-1,
            chunk_size=1024,
        )
        B_neg = vc.compute_external_B_offsurf(
            B0,
            X_trg=Xt_neg,
            digits=digits,
            max_Nt=-1,
            max_Np=-1,
            chunk_size=1024,
        )
        grad_fd[:, i, :] = (np.asarray(B_pos) - np.asarray(B_neg)) / (2.0 * eps)

    rel = np.linalg.norm(gradB - grad_fd) / (np.linalg.norm(grad_fd) + 1e-14)
    assert rel < 2e-2
