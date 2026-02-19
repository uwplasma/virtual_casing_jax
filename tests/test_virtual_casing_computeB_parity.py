from pathlib import Path

import numpy as np
import jax.numpy as jnp
import pytest

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
    B0_complete = load_dump(DATA_DIR / f"{prefix}_computeB_B0_complete")

    src_nt = X.shape[1]
    src_np = X.shape[2]
    nfp_eff = B0_complete.shape[1] // src_nt
    half_period = surf.shape[1] == nfp_eff * (src_nt + 1)
    nfp = nfp_eff // 2 if half_period else nfp_eff
    return X, src_nt, src_np, nfp, nfp_eff, half_period


def _reconstruct_B0(
    prefix: str,
    src_nt: int,
    src_np: int,
    nfp: int,
    nfp_eff: int,
    half_period: bool,
    trg_nt: int,
    *,
    tol: float = 1e-4,
):
    B0_complete_ref = load_dump(DATA_DIR / f"{prefix}_computeB_B0_complete")
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
    assert num / den < tol

    return np.asarray(B0)


@pytest.mark.parametrize(
    "prefix",
    ["case_vc", "case_vc_large", "case_simsopt", "case_simsopt_large", "case_vc_w7x", "case_vc_w7x_large"],
)
def test_virtual_casing_computeB_parity(prefix):
    if not (DATA_DIR / f"{prefix}_computeB_Bvc.bin").exists():
        pytest.skip("parity dump not available")

    X, src_nt, src_np, nfp, nfp_eff, half_period = _infer_setup(prefix)
    Bvc_ref = load_dump(DATA_DIR / f"{prefix}_computeB_Bvc")
    quad_coord = load_dump(DATA_DIR / f"{prefix}_computeB_quad_coord")

    trg_nt = Bvc_ref.shape[1]
    trg_np = Bvc_ref.shape[2]
    quad_nt = quad_coord.shape[1]
    quad_np = quad_coord.shape[2]

    tol_map = {
        "case_vc": 1e-4,
        "case_vc_large": 1e-4,
        "case_simsopt": 1e-4,
        "case_simsopt_large": 1e-4,
        "case_vc_w7x": 1e-3,
        "case_vc_w7x_large": 1e-3,
    }
    B0 = _reconstruct_B0(
        prefix,
        src_nt,
        src_np,
        nfp,
        nfp_eff,
        half_period,
        trg_nt,
        tol=tol_map[prefix],
    )

    digits_map = {
        "case_vc": 5,
        "case_vc_large": 6,
        "case_simsopt": 6,
        "case_simsopt_large": 6,
        "case_vc_w7x": 6,
        "case_vc_w7x_large": 6,
    }
    digits = digits_map[prefix]
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

    Bvc = vc.compute_external_B(
        B0,
        quad_nt=quad_nt,
        quad_np=quad_np,
        digits=digits,
        chunk_size=1024,
    )
    Bvc = np.asarray(Bvc)

    rel = np.linalg.norm(Bvc - Bvc_ref) / (np.linalg.norm(Bvc_ref) + 1e-14)
    tol_map = {
        "case_vc": 3e-4,
        "case_vc_large": 6e-4,
        "case_simsopt": 6e-4,
        "case_simsopt_large": 8e-4,
        "case_vc_w7x": 8e-4,
        "case_vc_w7x_large": 1.2e-3,
    }
    tol = tol_map[prefix]
    assert rel < tol
