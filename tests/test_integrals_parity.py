from pathlib import Path

import numpy as np
import pytest
import jax.numpy as jnp

from virtual_casing_jax.integrals import (
    laplace_fxd_u_eval,
    laplace_fxd_u_eval_vec,
    laplace_fxd_u_eval_singular,
    laplace_fxd_u_eval_vec_singular,
    laplace_fxd2_u_eval_singular,
    laplace_fxd2_u_eval_vec_singular,
    field_period_target_coords,
    computeB_offsurface_baseline,
    computeB_offsurface_adaptive,
)
from virtual_casing_jax.surface_ops import surf_normal_area_elem

# Allow direct import of dump_io when tests are not a package.
import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).parent))
from dump_io import load_dump  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"


def _dump_exists(prefix: str, name: str):
    base = DATA_DIR / f"{prefix}_{name}"
    return base.with_suffix(".bin").exists() and base.with_suffix(".json").exists()


def _load_case(prefix: str):
    X = load_dump(DATA_DIR / f"{prefix}_computeB_quad_coord")
    dX = load_dump(DATA_DIR / f"{prefix}_computeB_dX")
    BdotN = load_dump(DATA_DIR / f"{prefix}_computeB_BdotN")
    J = load_dump(DATA_DIR / f"{prefix}_computeB_J")
    gradG_BdotN = load_dump(DATA_DIR / f"{prefix}_computeB_gradG_BdotN")
    gradG_J = load_dump(DATA_DIR / f"{prefix}_computeB_gradG_J")
    Bvc = load_dump(DATA_DIR / f"{prefix}_computeB_Bvc")
    B_on_trg = load_dump(DATA_DIR / f"{prefix}_computeB_B_on_trg")
    return X, dX, BdotN, J, gradG_BdotN, gradG_J, Bvc, B_on_trg


def _load_offsurf_case(prefix: str):
    X = load_dump(DATA_DIR / f"{prefix}_computeBOff_quad_coord")
    BdotN = load_dump(DATA_DIR / f"{prefix}_computeBOff_BdotN")
    J = load_dump(DATA_DIR / f"{prefix}_computeBOff_J")
    Xt = load_dump(DATA_DIR / f"{prefix}_computeBOff_Xt")
    Bvc = load_dump(DATA_DIR / f"{prefix}_computeBOff_Bvc")
    return X, BdotN, J, Xt, Bvc


def _load_gradb_case(prefix: str):
    X = load_dump(DATA_DIR / f"{prefix}_computeGradB_quad_coord")
    dX = load_dump(DATA_DIR / f"{prefix}_computeGradB_dX")
    BdotN = load_dump(DATA_DIR / f"{prefix}_computeGradB_BdotN")
    J = load_dump(DATA_DIR / f"{prefix}_computeGradB_J")
    gradgradG_BdotN = load_dump(DATA_DIR / f"{prefix}_computeGradB_gradgradG_BdotN")
    gradG_J = load_dump(DATA_DIR / f"{prefix}_computeGradB_gradG_J")
    gradBvc = load_dump(DATA_DIR / f"{prefix}_computeGradB_gradBvc")
    return X, dX, BdotN, J, gradgradG_BdotN, gradG_J, gradBvc


def _infer_nfp(prefix: str):
    if _dump_exists(prefix, "computeGradB_B0_complete") and _dump_exists(prefix, "setup_X"):
        B0 = load_dump(DATA_DIR / f"{prefix}_computeGradB_B0_complete")
        base = load_dump(DATA_DIR / f"{prefix}_setup_X")
        if B0.shape[1] % base.shape[1] != 0:
            raise ValueError("computeGradB_B0_complete must be an integer multiple of setup_X in toroidal dimension")
        return B0.shape[1] // base.shape[1]
    if _dump_exists(prefix, "computeB_B0_complete") and _dump_exists(prefix, "setup_X"):
        B0 = load_dump(DATA_DIR / f"{prefix}_computeB_B0_complete")
        base = load_dump(DATA_DIR / f"{prefix}_setup_X")
        if B0.shape[1] % base.shape[1] != 0:
            raise ValueError("computeB_B0_complete must be an integer multiple of setup_X in toroidal dimension")
        return B0.shape[1] // base.shape[1]
    if _dump_exists(prefix, "setup_surface_coord") and _dump_exists(prefix, "setup_X"):
        surf = load_dump(DATA_DIR / f"{prefix}_setup_surface_coord")
        base = load_dump(DATA_DIR / f"{prefix}_setup_X")
        if surf.shape[1] % base.shape[1] != 0:
            raise ValueError("setup_surface_coord must be an integer multiple of setup_X in toroidal dimension")
        return surf.shape[1] // base.shape[1]
    return 1


@pytest.mark.parametrize("prefix", ["case_vc", "case_simsopt"])
def test_laplace_fxd_u_eval_parity(prefix):
    if not _dump_exists(prefix, "computeB_gradG_BdotN"):
        pytest.skip("parity dump not available")

    X, dX, BdotN, _, gradG_BdotN, _, Bvc, B_on_trg = _load_case(prefix)
    trg_nt = Bvc.shape[1]
    trg_np = Bvc.shape[2]
    nfp = B_on_trg.shape[1] // trg_nt

    X_trg = field_period_target_coords(jnp.asarray(X), trg_nt, trg_np, nfp)
    _, area_elem = surf_normal_area_elem(jnp.asarray(dX), jnp.asarray(X))

    out = laplace_fxd_u_eval(
        jnp.asarray(X),
        X_trg,
        jnp.asarray(BdotN),
        area_elem,
        chunk_size=2048,
    )
    out = np.asarray(out).reshape((3, trg_nt, trg_np))

    ref = gradG_BdotN
    num = np.linalg.norm(out - ref)
    den = np.linalg.norm(ref) + 1e-14
    rel = num / den
    # Baseline direct-sum (no singular correction). Expect coarse agreement only.
    assert rel < 0.25


@pytest.mark.parametrize("prefix", ["case_vc", "case_simsopt"])
def test_laplace_fxd_u_eval_vec_parity(prefix):
    if not _dump_exists(prefix, "computeB_gradG_J"):
        pytest.skip("parity dump not available")

    X, dX, _, J, _, gradG_J, Bvc, B_on_trg = _load_case(prefix)
    trg_nt = Bvc.shape[1]
    trg_np = Bvc.shape[2]
    nfp = B_on_trg.shape[1] // trg_nt

    X_trg = field_period_target_coords(jnp.asarray(X), trg_nt, trg_np, nfp)
    _, area_elem = surf_normal_area_elem(jnp.asarray(dX), jnp.asarray(X))

    out = laplace_fxd_u_eval_vec(
        jnp.asarray(X),
        X_trg,
        jnp.asarray(J),
        area_elem,
        chunk_size=2048,
    )
    out = np.asarray(out).reshape((3, 3, trg_nt, trg_np))

    ref = gradG_J
    num = np.linalg.norm(out - ref)
    den = np.linalg.norm(ref) + 1e-14
    rel = num / den
    assert rel < 0.1


@pytest.mark.parametrize("prefix", ["case_vc", "case_simsopt"])
def test_laplace_fxd_u_eval_singular_parity(prefix):
    if not _dump_exists(prefix, "computeB_gradG_BdotN"):
        pytest.skip("parity dump not available")

    X, dX, BdotN, _, gradG_BdotN, _, Bvc, B_on_trg = _load_case(prefix)
    trg_nt = Bvc.shape[1]
    trg_np = Bvc.shape[2]
    nfp = B_on_trg.shape[1] // trg_nt

    out = laplace_fxd_u_eval_singular(
        jnp.asarray(X),
        jnp.asarray(dX),
        jnp.asarray(BdotN),
        trg_nt,
        trg_np,
        nfp,
        digits=5,
        chunk_size=2048,
    )
    out = np.asarray(out).reshape((3, trg_nt, trg_np))

    ref = gradG_BdotN
    rel = np.linalg.norm(out - ref) / (np.linalg.norm(ref) + 1e-14)
    assert rel < 1.2e-3


def test_laplace_fxd_u_eval_singular_patch_dtype():
    prefix = "case_vc"
    if not _dump_exists(prefix, "computeB_gradG_BdotN"):
        pytest.skip("parity dump not available")

    X, dX, BdotN, _, gradG_BdotN, _, Bvc, B_on_trg = _load_case(prefix)
    trg_nt = Bvc.shape[1]
    trg_np = Bvc.shape[2]
    nfp = B_on_trg.shape[1] // trg_nt

    out = laplace_fxd_u_eval_singular(
        jnp.asarray(X),
        jnp.asarray(dX),
        jnp.asarray(BdotN),
        trg_nt,
        trg_np,
        nfp,
        digits=5,
        chunk_size=2048,
        patch_dtype="float32",
    )
    out = np.asarray(out).reshape((3, trg_nt, trg_np))

    ref = gradG_BdotN
    rel = np.linalg.norm(out - ref) / (np.linalg.norm(ref) + 1e-14)
    assert rel < 2.0e-3


@pytest.mark.parametrize("prefix", ["case_vc", "case_simsopt"])
def test_laplace_fxd_u_eval_vec_singular_parity(prefix):
    if not _dump_exists(prefix, "computeB_gradG_J"):
        pytest.skip("parity dump not available")

    X, dX, _, J, _, gradG_J, Bvc, B_on_trg = _load_case(prefix)
    trg_nt = Bvc.shape[1]
    trg_np = Bvc.shape[2]
    nfp = B_on_trg.shape[1] // trg_nt

    out = laplace_fxd_u_eval_vec_singular(
        jnp.asarray(X),
        jnp.asarray(dX),
        jnp.asarray(J),
        trg_nt,
        trg_np,
        nfp,
        digits=5,
        chunk_size=2048,
    )
    out = np.asarray(out).reshape((3, 3, trg_nt, trg_np))

    ref = gradG_J
    rel = np.linalg.norm(out - ref) / (np.linalg.norm(ref) + 1e-14)
    assert rel < 4e-4


@pytest.mark.parametrize("prefix", ["case_vc", "case_simsopt"])
def test_baseline_computeB_parity(prefix):
    if not _dump_exists(prefix, "computeB_Bvc"):
        pytest.skip("parity dump not available")

    X, dX, BdotN, J, _, _, Bvc_ref, B_on_trg = _load_case(prefix)
    trg_nt = Bvc_ref.shape[1]
    trg_np = Bvc_ref.shape[2]
    nfp = B_on_trg.shape[1] // trg_nt

    X_trg = field_period_target_coords(jnp.asarray(X), trg_nt, trg_np, nfp)
    _, area_elem = surf_normal_area_elem(jnp.asarray(dX), jnp.asarray(X))

    gradG_BdotN = laplace_fxd_u_eval(
        jnp.asarray(X),
        X_trg,
        jnp.asarray(BdotN),
        area_elem,
        chunk_size=2048,
    )
    gradG_BdotN = np.asarray(gradG_BdotN).reshape((3, trg_nt, trg_np))

    gradG_J = laplace_fxd_u_eval_vec(
        jnp.asarray(X),
        X_trg,
        jnp.asarray(J),
        area_elem,
        chunk_size=2048,
    )
    gradG_J = np.asarray(gradG_J).reshape((3, 3, trg_nt, trg_np))

    Bvc = np.zeros_like(Bvc_ref)
    for k in range(3):
        k1 = (k + 1) % 3
        k2 = (k + 2) % 3
        Bvc[k] = gradG_J[k1, k2] - gradG_J[k2, k1]

    B_on = B_on_trg[:, :trg_nt, :]
    Bvc = Bvc + gradG_BdotN + 0.5 * B_on

    rel = np.linalg.norm(Bvc - Bvc_ref) / (np.linalg.norm(Bvc_ref) + 1e-14)
    assert rel < 0.08


@pytest.mark.parametrize("prefix", ["case_vc", "case_simsopt"])
def test_singular_computeB_parity(prefix):
    if not _dump_exists(prefix, "computeB_Bvc"):
        pytest.skip("parity dump not available")

    X, dX, BdotN, J, _, _, Bvc_ref, B_on_trg = _load_case(prefix)
    trg_nt = Bvc_ref.shape[1]
    trg_np = Bvc_ref.shape[2]
    nfp = B_on_trg.shape[1] // trg_nt

    gradG_BdotN = laplace_fxd_u_eval_singular(
        jnp.asarray(X),
        jnp.asarray(dX),
        jnp.asarray(BdotN),
        trg_nt,
        trg_np,
        nfp,
        digits=5,
        chunk_size=2048,
    )
    gradG_BdotN = np.asarray(gradG_BdotN).reshape((3, trg_nt, trg_np))

    gradG_J = laplace_fxd_u_eval_vec_singular(
        jnp.asarray(X),
        jnp.asarray(dX),
        jnp.asarray(J),
        trg_nt,
        trg_np,
        nfp,
        digits=5,
        chunk_size=2048,
    )
    gradG_J = np.asarray(gradG_J).reshape((3, 3, trg_nt, trg_np))

    Bvc = np.zeros_like(Bvc_ref)
    for k in range(3):
        k1 = (k + 1) % 3
        k2 = (k + 2) % 3
        Bvc[k] = gradG_J[k1, k2] - gradG_J[k2, k1]

    B_on = B_on_trg[:, :trg_nt, :]
    Bvc = Bvc + gradG_BdotN + 0.5 * B_on

    rel = np.linalg.norm(Bvc - Bvc_ref) / (np.linalg.norm(Bvc_ref) + 1e-14)
    assert rel < 3e-4


def test_baseline_computeBOff_parity_case_vc():
    prefix = "case_vc"
    if not _dump_exists(prefix, "computeBOff_Bvc"):
        pytest.skip("parity dump not available")

    X, BdotN, J, Xt, Bvc_ref = _load_offsurf_case(prefix)

    Bvc = np.asarray(
        computeB_offsurface_baseline(
            jnp.asarray(X),
            jnp.asarray(BdotN),
            jnp.asarray(J),
            jnp.asarray(Xt),
            upsample_factor=4,
            chunk_size=2048,
            ext=True,
        )
    )

    rel = np.linalg.norm(Bvc - Bvc_ref) / (np.linalg.norm(Bvc_ref) + 1e-14)
    assert rel < 0.15


def test_adaptive_computeBOff_parity_case_vc():
    prefix = "case_vc"
    if not _dump_exists(prefix, "computeBOff_Bvc"):
        pytest.skip("parity dump not available")

    X, BdotN, J, Xt, Bvc_ref = _load_offsurf_case(prefix)

    Bvc = np.asarray(
        computeB_offsurface_adaptive(
            jnp.asarray(X),
            jnp.asarray(BdotN),
            jnp.asarray(J),
            jnp.asarray(Xt),
            digits=5,
            max_Nt=-1,
            max_Np=-1,
            ext=True,
            chunk_size=2048,
        )
    )

    rel = np.linalg.norm(Bvc - Bvc_ref) / (np.linalg.norm(Bvc_ref) + 1e-14)
    assert rel < 1e-2


@pytest.mark.parametrize("prefix", ["case_vc", "case_simsopt"])
def test_laplace_fxd2_u_eval_singular_parity(prefix):
    if not _dump_exists(prefix, "computeGradB_gradgradG_BdotN"):
        pytest.skip("parity dump not available")

    X, dX, BdotN, _, gradgradG_BdotN, _, gradBvc = _load_gradb_case(prefix)
    trg_nt = gradBvc.shape[2]
    trg_np = gradBvc.shape[3]

    nfp = _infer_nfp(prefix)

    out = laplace_fxd2_u_eval_singular(
        jnp.asarray(X),
        jnp.asarray(dX),
        jnp.asarray(BdotN),
        trg_nt,
        trg_np,
        nfp,
        digits=5,
        hedgehog_order=8,
        chunk_size=1024,
    )
    out = np.asarray(out).reshape((3, 3, trg_nt, trg_np))

    rel = np.linalg.norm(out - gradgradG_BdotN) / (np.linalg.norm(gradgradG_BdotN) + 1e-14)
    assert rel < 5e-3


@pytest.mark.parametrize("prefix", ["case_vc", "case_simsopt"])
def test_laplace_fxd2_u_eval_vec_singular_parity(prefix):
    if not _dump_exists(prefix, "computeGradB_gradG_J"):
        pytest.skip("parity dump not available")

    X, dX, _, J, _, gradG_J, gradBvc = _load_gradb_case(prefix)
    trg_nt = gradBvc.shape[2]
    trg_np = gradBvc.shape[3]
    nfp = _infer_nfp(prefix)

    out = laplace_fxd2_u_eval_vec_singular(
        jnp.asarray(X),
        jnp.asarray(dX),
        jnp.asarray(J),
        trg_nt,
        trg_np,
        nfp,
        digits=5,
        hedgehog_order=8,
        chunk_size=1024,
    )
    out = np.asarray(out).reshape((3, 3, 3, trg_nt, trg_np))

    rel = np.linalg.norm(out - gradG_J) / (np.linalg.norm(gradG_J) + 1e-14)
    tol = 5e-3 if prefix == "case_vc" else 8e-3
    assert rel < tol


@pytest.mark.parametrize("prefix", ["case_vc", "case_simsopt"])
def test_computeGradB_parity(prefix):
    if not _dump_exists(prefix, "computeGradB_gradBvc"):
        pytest.skip("parity dump not available")

    X, dX, BdotN, J, _, _, gradBvc_ref = _load_gradb_case(prefix)
    trg_nt = gradBvc_ref.shape[2]
    trg_np = gradBvc_ref.shape[3]
    nfp = _infer_nfp(prefix)

    gradgradG_BdotN = laplace_fxd2_u_eval_singular(
        jnp.asarray(X),
        jnp.asarray(dX),
        jnp.asarray(BdotN),
        trg_nt,
        trg_np,
        nfp,
        digits=5,
        hedgehog_order=8,
        chunk_size=1024,
    )
    gradgradG_BdotN = np.asarray(gradgradG_BdotN).reshape((3, 3, trg_nt, trg_np))

    gradG_J = laplace_fxd2_u_eval_vec_singular(
        jnp.asarray(X),
        jnp.asarray(dX),
        jnp.asarray(J),
        trg_nt,
        trg_np,
        nfp,
        digits=5,
        hedgehog_order=8,
        chunk_size=1024,
    )
    gradG_J = np.asarray(gradG_J).reshape((3, 3, 3, trg_nt, trg_np))

    gradBvc = np.zeros_like(gradBvc_ref)
    for k in range(3):
        k1 = (k + 1) % 3
        k2 = (k + 2) % 3
        gradBvc[k] = gradG_J[k1, k2] - gradG_J[k2, k1]
    gradBvc = gradBvc + gradgradG_BdotN

    rel = np.linalg.norm(gradBvc - gradBvc_ref) / (np.linalg.norm(gradBvc_ref) + 1e-14)
    tol = 5e-3 if prefix == "case_vc" else 6e-3
    assert rel < tol
