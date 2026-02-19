import numpy as np
import jax.numpy as jnp

from virtual_casing_jax.integrals import laplace_fxd_u_eval


def test_target_chunk_blocking_matches():
    rng = np.random.default_rng(0)
    X_src = jnp.asarray(rng.normal(size=(3, 6, 5)))
    X_trg = jnp.asarray(rng.normal(size=(3, 4, 3)))
    density = jnp.asarray(rng.normal(size=(6, 5)))
    area_elem = jnp.asarray(np.abs(rng.normal(size=(6, 5))))

    out_full = laplace_fxd_u_eval(
        X_src,
        X_trg,
        density,
        area_elem,
        chunk_size=4,
        target_chunk_size=None,
    )
    out_blocked = laplace_fxd_u_eval(
        X_src,
        X_trg,
        density,
        area_elem,
        chunk_size=4,
        target_chunk_size=2,
    )

    np.testing.assert_allclose(
        np.asarray(out_full),
        np.asarray(out_blocked),
        rtol=1e-10,
        atol=1e-10,
    )
