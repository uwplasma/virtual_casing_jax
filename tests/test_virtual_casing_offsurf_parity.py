from pathlib import Path

import numpy as np
import jax.numpy as jnp

from virtual_casing_jax.virtual_casing import VirtualCasingJAX
from virtual_casing_jax.surface_ops import rotate_toroidal, complete_vec_field

# Allow direct import of dump_io when tests are not a package.
import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).parent))
from dump_io import load_dump  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"


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
    assert num / den < 1e-4

    return np.asarray(B0)


def _get_trg_shape(prefix: str):
    Bvc = load_dump(DATA_DIR / f"{prefix}_computeB_Bvc")
    return Bvc.shape[1], Bvc.shape[2]


def test_virtual_casing_computeB_offsurf_parity():
    prefix = "case_vc"
    Bvc_ref = load_dump(DATA_DIR / f"{prefix}_computeBOff_Bvc")
    Xt = load_dump(DATA_DIR / f"{prefix}_computeBOff_Xt")

    X, src_nt, src_np, nfp, nfp_eff, half_period = _infer_setup(prefix, "computeBOff")
    trg_nt, trg_np = _get_trg_shape(prefix)
    B0 = _reconstruct_B0(prefix, "computeBOff", src_nt, src_np, nfp, nfp_eff, half_period, trg_nt)

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

    Bvc = vc.compute_external_B_offsurf(
        B0,
        X_trg=Xt,
        digits=digits,
        max_Nt=-1,
        max_Np=-1,
        chunk_size=1024,
    )
    Bvc = np.asarray(Bvc)

    rel = np.linalg.norm(Bvc - Bvc_ref) / (np.linalg.norm(Bvc_ref) + 1e-14)
    assert rel < 5e-4


def test_virtual_casing_computeBint_offsurf_parity():
    prefix = "case_vc_int"
    Bvc_ref = load_dump(DATA_DIR / f"{prefix}_computeBOff_Bvc")
    Xt = load_dump(DATA_DIR / f"{prefix}_computeBOff_Xt")

    X, src_nt, src_np, nfp, nfp_eff, half_period = _infer_setup(prefix, "computeBOff")
    trg_nt, trg_np = _get_trg_shape(prefix)
    B0 = _reconstruct_B0(prefix, "computeBOff", src_nt, src_np, nfp, nfp_eff, half_period, trg_nt)

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

    Bvc = vc.compute_internal_B_offsurf(
        B0,
        X_trg=Xt,
        digits=digits,
        max_Nt=-1,
        max_Np=-1,
        chunk_size=1024,
    )
    Bvc = np.asarray(Bvc)

    rel = np.linalg.norm(Bvc - Bvc_ref) / (np.linalg.norm(Bvc_ref) + 1e-14)
    assert rel < 5e-4


def test_virtual_casing_gradB_offsurf_parity():
    prefix = "case_vc"
    gradB_ref = load_dump(DATA_DIR / f"{prefix}_computeGradBOff_gradBvc")
    Xt = load_dump(DATA_DIR / f"{prefix}_computeGradBOff_Xt")

    X, src_nt, src_np, nfp, nfp_eff, half_period = _infer_setup(prefix, "computeGradBOff")
    trg_nt, trg_np = _get_trg_shape(prefix)
    B0 = _reconstruct_B0(prefix, "computeGradBOff", src_nt, src_np, nfp, nfp_eff, half_period, trg_nt)

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

    gradB = vc.compute_external_gradB_offsurf(
        B0,
        X_trg=Xt,
        digits=digits,
        max_Nt=-1,
        max_Np=-1,
        chunk_size=1024,
    )
    gradB = np.asarray(gradB)

    rel = np.linalg.norm(gradB - gradB_ref) / (np.linalg.norm(gradB_ref) + 1e-14)
    assert rel < 5e-3
