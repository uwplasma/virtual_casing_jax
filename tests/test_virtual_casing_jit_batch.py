from pathlib import Path
import warnings

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


def test_jit_and_batch_wrappers_case_vc():
    prefix = "case_vc"
    X, src_nt, src_np, nfp, nfp_eff, half_period = _infer_setup(prefix)
    Bvc_ref = load_dump(DATA_DIR / f"{prefix}_computeB_Bvc")
    gradB_ref = load_dump(DATA_DIR / f"{prefix}_computeGradB_gradBvc")
    quad_coord = load_dump(DATA_DIR / f"{prefix}_computeB_quad_coord")
    quad_nt = quad_coord.shape[1]
    quad_np = quad_coord.shape[2]
    trg_nt = Bvc_ref.shape[1]
    trg_np = Bvc_ref.shape[2]

    B0 = _reconstruct_B0(prefix, src_nt, src_np, nfp, nfp_eff, half_period, trg_nt)

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

    Bvc = vc.compute_external_B(B0, quad_nt=quad_nt, quad_np=quad_np, digits=5, chunk_size=1024)
    Bvc_jit = vc.compute_external_B_jit(B0, quad_nt=quad_nt, quad_np=quad_np, digits=5, chunk_size=1024)
    np.testing.assert_allclose(np.asarray(Bvc_jit), np.asarray(Bvc), rtol=1e-8, atol=1e-10)
    Bint = vc.compute_internal_B(B0, quad_nt=quad_nt, quad_np=quad_np, digits=5, chunk_size=1024)
    Bint_jit = vc.compute_internal_B_jit(B0, quad_nt=quad_nt, quad_np=quad_np, digits=5, chunk_size=1024)
    np.testing.assert_allclose(np.asarray(Bint_jit), np.asarray(Bint), rtol=1e-8, atol=1e-10)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Some donated buffers were not usable")
        Bvc_jit_donate = vc.compute_external_B_jit(
            B0.copy(),
            quad_nt=quad_nt,
            quad_np=quad_np,
            digits=5,
            chunk_size=1024,
            donate=True,
        )
    np.testing.assert_allclose(np.asarray(Bvc_jit_donate), np.asarray(Bvc), rtol=1e-8, atol=1e-10)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Some donated buffers were not usable")
        Bint_jit_donate = vc.compute_internal_B_jit(
            B0.copy(),
            quad_nt=quad_nt,
            quad_np=quad_np,
            digits=5,
            chunk_size=1024,
            donate=True,
        )
    np.testing.assert_allclose(np.asarray(Bint_jit_donate), np.asarray(Bint), rtol=1e-8, atol=1e-10)

    gradB = vc.compute_external_gradB(B0, quad_nt=quad_nt, quad_np=quad_np, digits=5, hedgehog_order=8, chunk_size=1024)
    gradB_jit = vc.compute_external_gradB_jit(B0, quad_nt=quad_nt, quad_np=quad_np, digits=5, hedgehog_order=8, chunk_size=1024)
    np.testing.assert_allclose(np.asarray(gradB_jit), np.asarray(gradB), rtol=1e-8, atol=1e-10)
    gradBint = vc.compute_internal_gradB(B0, quad_nt=quad_nt, quad_np=quad_np, digits=5, hedgehog_order=8, chunk_size=1024)
    gradBint_jit = vc.compute_internal_gradB_jit(B0, quad_nt=quad_nt, quad_np=quad_np, digits=5, hedgehog_order=8, chunk_size=1024)
    np.testing.assert_allclose(np.asarray(gradBint_jit), np.asarray(gradBint), rtol=1e-8, atol=1e-10)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Some donated buffers were not usable")
        gradB_jit_donate = vc.compute_external_gradB_jit(
            B0.copy(),
            quad_nt=quad_nt,
            quad_np=quad_np,
            digits=5,
            hedgehog_order=8,
            chunk_size=1024,
            donate=True,
        )
    np.testing.assert_allclose(np.asarray(gradB_jit_donate), np.asarray(gradB), rtol=1e-8, atol=1e-10)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Some donated buffers were not usable")
        gradBint_jit_donate = vc.compute_internal_gradB_jit(
            B0.copy(),
            quad_nt=quad_nt,
            quad_np=quad_np,
            digits=5,
            hedgehog_order=8,
            chunk_size=1024,
            donate=True,
        )
    np.testing.assert_allclose(np.asarray(gradBint_jit_donate), np.asarray(gradBint), rtol=1e-8, atol=1e-10)

    B0_batch = np.stack([B0, B0 * 1.001], axis=0)
    Bvc_batch = vc.compute_external_B_batch(B0_batch, quad_nt=quad_nt, quad_np=quad_np, digits=5, chunk_size=1024)
    np.testing.assert_allclose(np.asarray(Bvc_batch[0]), np.asarray(Bvc), rtol=1e-8, atol=1e-10)
    Bint_batch = vc.compute_internal_B_batch(B0_batch, quad_nt=quad_nt, quad_np=quad_np, digits=5, chunk_size=1024)
    np.testing.assert_allclose(np.asarray(Bint_batch[0]), np.asarray(Bint), rtol=1e-8, atol=1e-10)

    gradB_batch = vc.compute_external_gradB_batch(B0_batch, quad_nt=quad_nt, quad_np=quad_np, digits=5, hedgehog_order=8, chunk_size=1024)
    np.testing.assert_allclose(np.asarray(gradB_batch[0]), np.asarray(gradB), rtol=1e-8, atol=1e-10)
    gradBint_batch = vc.compute_internal_gradB_batch(B0_batch, quad_nt=quad_nt, quad_np=quad_np, digits=5, hedgehog_order=8, chunk_size=1024)
    np.testing.assert_allclose(np.asarray(gradBint_batch[0]), np.asarray(gradBint), rtol=1e-8, atol=1e-10)
