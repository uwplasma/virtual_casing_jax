from pathlib import Path

import numpy as np
import pytest
import jax.numpy as jnp

from virtual_casing_jax.surface_ops import grad2d, surf_normal_area_elem, dot_prod, cross_prod

# Allow direct import of dump_io when tests are not a package.
import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).parent))
from dump_io import load_dump  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"


def _dump_exists(prefix: str, name: str):
    base = DATA_DIR / f"{prefix}_{name}"
    return base.with_suffix(".bin").exists() and base.with_suffix(".json").exists()


@pytest.mark.parametrize("prefix", ["case_vc", "case_simsopt"])
def test_grad2d_parity(prefix):
    if not _dump_exists(prefix, "computeB_quad_coord"):
        pytest.skip("parity dump not available")
    X = load_dump(DATA_DIR / f"{prefix}_computeB_quad_coord")
    dX_ref = load_dump(DATA_DIR / f"{prefix}_computeB_dX")
    nt = X.shape[1]
    npol = X.shape[2]
    dX = grad2d(jnp.asarray(X), nt, npol)
    np.testing.assert_allclose(np.asarray(dX), dX_ref, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("prefix", ["case_vc", "case_simsopt"])
def test_normal_parity(prefix):
    if not _dump_exists(prefix, "computeB_normal"):
        pytest.skip("parity dump not available")
    X = load_dump(DATA_DIR / f"{prefix}_computeB_quad_coord")
    dX = load_dump(DATA_DIR / f"{prefix}_computeB_dX")
    normal_ref = load_dump(DATA_DIR / f"{prefix}_computeB_normal")
    normal, area_elem = surf_normal_area_elem(jnp.asarray(dX), jnp.asarray(X))
    np.testing.assert_allclose(np.asarray(normal), normal_ref, rtol=1e-10, atol=1e-12)
    assert np.all(np.asarray(area_elem) > 0.0)


@pytest.mark.parametrize("prefix", ["case_vc", "case_simsopt"])
def test_cross_dot_parity(prefix):
    if not _dump_exists(prefix, "computeB_J"):
        pytest.skip("parity dump not available")
    B = load_dump(DATA_DIR / f"{prefix}_computeB_B_resampled")
    normal = load_dump(DATA_DIR / f"{prefix}_computeB_normal")
    J_ref = load_dump(DATA_DIR / f"{prefix}_computeB_J")
    BdotN_ref = load_dump(DATA_DIR / f"{prefix}_computeB_BdotN")

    J = cross_prod(jnp.asarray(normal), jnp.asarray(B))
    BdotN = dot_prod(jnp.asarray(B), jnp.asarray(normal))

    np.testing.assert_allclose(np.asarray(J), J_ref, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.asarray(BdotN), BdotN_ref, rtol=1e-10, atol=1e-12)
