from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from virtual_casing_jax.integrals import field_period_target_coords
from virtual_casing_jax.surface_ops import rotate_toroidal, complete_vec_field
from virtual_casing_jax.virtual_casing import VirtualCasingJAX

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


def _reconstruct_B0(prefix: str, src_nt: int, src_np: int, nfp: int, nfp_eff: int, half_period: bool, trg_nt: int):
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
    assert num / den < 1e-4

    return np.asarray(B0)


@pytest.mark.parametrize("prefix", ["case_vc", "case_simsopt"])
def test_autodiff_gradb_matches_reference(prefix):
    if not (DATA_DIR / f"{prefix}_computeGradB_gradBvc.bin").exists():
        pytest.skip("parity dump not available")

    X, src_nt, src_np, nfp, nfp_eff, half_period = _infer_setup(prefix)
    gradBvc_ref = load_dump(DATA_DIR / f"{prefix}_computeGradB_gradBvc")
    quad_coord = load_dump(DATA_DIR / f"{prefix}_computeB_quad_coord")

    trg_nt = gradBvc_ref.shape[2]
    trg_np = gradBvc_ref.shape[3]
    quad_nt = quad_coord.shape[1]
    quad_np = quad_coord.shape[2]

    B0 = _reconstruct_B0(prefix, src_nt, src_np, nfp, nfp_eff, half_period, trg_nt)
    X_trg = field_period_target_coords(jnp.asarray(quad_coord), trg_nt, trg_np, nfp_eff)

    vc = VirtualCasingJAX()
    vc.setup(
        5,
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

    def f(xtrg):
        return vc.compute_external_B_autodiff(
            B0,
            X_trg=xtrg,
            quad_nt=quad_nt,
            quad_np=quad_np,
            digits=5,
            chunk_size=1024,
            hedgehog_order=8,
        )

    jac = jax.jacobian(f)(X_trg)
    jac = np.asarray(jac)
    diag = np.zeros_like(gradBvc_ref)
    for i in range(trg_nt):
        for j in range(trg_np):
            diag[:, :, i, j] = jac[:, i, j, :, i, j]

    rel = np.linalg.norm(diag - gradBvc_ref) / (np.linalg.norm(gradBvc_ref) + 1e-14)
    tol = 5e-3 if prefix == "case_vc" else 7e-3
    assert rel < tol
