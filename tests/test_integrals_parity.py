from pathlib import Path

import numpy as np
import pytest
import jax.numpy as jnp

from virtual_casing_jax.integrals import (
    laplace_fxd_u_eval,
    laplace_fxd_u_eval_vec,
    field_period_target_coords,
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
