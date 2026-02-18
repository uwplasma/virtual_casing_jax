import numpy as np
import jax.numpy as jnp

from virtual_casing_jax.surface_ops import complete_vec_field, rotate_toroidal, upsample, resample


def test_rotate_toroidal_identity():
    dof, nt, npol = 3, 4, 5
    X = jnp.arange(dof * nt * npol, dtype=jnp.float64).reshape((dof, nt, npol))
    Y = rotate_toroidal(X, nt, npol, 0.0)
    np.testing.assert_allclose(np.asarray(Y), np.asarray(X))


def test_complete_vec_field_nfp_replication():
    dof, nt, npol = 3, 3, 4
    nfp = 2
    X = jnp.ones((dof, nt, npol), dtype=jnp.float64)
    Y = complete_vec_field(X, is_surf=True, half_period=False, nfp=nfp, nt=nt, npol=npol, dtheta=0.0)
    assert Y.shape == (dof, nfp * nt, npol)
    np.testing.assert_allclose(np.asarray(Y[:, :nt, :]), np.asarray(X))


def test_upsample_resample_identity():
    dof, nt, npol = 2, 6, 6
    X = jnp.arange(dof * nt * npol, dtype=jnp.float64).reshape((dof, nt, npol))
    Y = upsample(X, nt, npol, nt, npol)
    Z = resample(X, nt, npol, nt, npol)
    np.testing.assert_allclose(np.asarray(Y), np.asarray(X), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(Z), np.asarray(X), rtol=1e-12, atol=1e-12)
