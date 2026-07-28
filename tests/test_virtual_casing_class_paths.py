import numpy as np
import pytest
import jax.numpy as jnp

from virtual_casing_jax.virtual_casing import VirtualCasingJAX


def _torus(nt=5, npol=4, R0=2.0, r=0.3):
    phi = jnp.linspace(0.0, 2.0 * jnp.pi, nt, endpoint=False)
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, npol, endpoint=False)
    theta2d, phi2d = jnp.meshgrid(theta, phi)
    return jnp.stack(
        [
            (R0 + r * jnp.cos(theta2d)) * jnp.cos(phi2d),
            (R0 + r * jnp.cos(theta2d)) * jnp.sin(phi2d),
            r * jnp.sin(theta2d),
        ],
        axis=0,
    )


def test_auto_quadrature_cache_and_boundary_jump_identity():
    X = _torus(nt=4, npol=4, r=0.25)
    B0 = X * 0.02 + jnp.array([0.1, -0.02, 0.05])[:, None, None]

    vc = VirtualCasingJAX()
    vc.setup(1, 1, False, 4, 4, X, 4, 4, 4, 4)

    Bext = vc.compute_external_B(
        B0,
        digits=1,
        chunk_size=64,
        target_chunk_size=4,
        pou_dtype=jnp.float64,
        patch_dtype=jnp.float64,
    )
    cached_setup = vc._b_setup
    Bint = vc.compute_internal_B(
        B0,
        digits=1,
        chunk_size=64,
        target_chunk_size=4,
        pou_dtype=jnp.float64,
        patch_dtype=jnp.float64,
    )

    assert cached_setup is vc._b_setup
    assert cached_setup is not None
    assert cached_setup.quad_nt % vc.trg_nt == 0
    assert cached_setup.quad_np % vc.trg_np == 0
    assert len(cached_setup.patch_idx_cache) == 1
    np.testing.assert_allclose(Bext + Bint, B0, rtol=1e-12, atol=1e-12)


def test_on_surface_bad_offsurface_target_rank_rejected_after_setup():
    X = _torus(nt=4, npol=4)
    B0 = X * 0.01 + 0.2

    vc = VirtualCasingJAX()
    vc.setup(2, 1, False, 4, 4, X, 4, 4, 4, 4)

    with pytest.raises(ValueError, match="X_trg must have shape"):
        vc.compute_external_B(
            B0,
            X_trg=jnp.zeros((3,)),
            digits=2,
            quad_nt=4,
            quad_np=4,
        )


def test_offsurface_internal_branches_reshape_and_cancel_external_branch():
    X = _torus()
    B0 = X * 0.03 + 0.08
    X_trg = jnp.array(
        [
            [[2.2], [2.3]],
            [[0.05], [-0.1]],
            [[0.02], [0.03]],
        ]
    )

    vc = VirtualCasingJAX()
    vc.setup(2, 1, False, 5, 4, X, 5, 4, 5, 4)
    X_src, _, _ = vc._offsurface_densities(B0)
    levels = ((int(X_src.shape[1]), int(X_src.shape[2])),)

    Bext = vc.compute_external_B_offsurf(
        B0,
        X_trg=X_trg,
        digits=2,
        max_Nt=13,
        max_Np=13,
        chunk_size=64,
        target_chunk_size=1,
    )
    Bint = vc.compute_internal_B_offsurf(
        B0,
        X_trg=X_trg,
        digits=2,
        max_Nt=13,
        max_Np=13,
        chunk_size=64,
        target_chunk_size=1,
    )
    Bint_jit = vc.compute_internal_B_offsurf_schedule_jit(
        B0,
        X_trg=X_trg,
        levels=levels,
        digits=2,
        chunk_size=64,
        target_chunk_size=1,
    )

    assert Bext.shape == (3, 2, 1)
    assert Bint.shape == (3, 2, 1)
    np.testing.assert_allclose(Bext + Bint, 0.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(Bint_jit, Bint, rtol=1e-11, atol=1e-11)


def test_offsurface_gradB_internal_branches_reshape_and_match_schedule():
    X = _torus()
    B0 = X * 0.03 + 0.08
    X_trg = jnp.array(
        [
            [[2.2], [2.3]],
            [[0.05], [-0.1]],
            [[0.02], [0.03]],
        ]
    )

    vc = VirtualCasingJAX()
    vc.setup(2, 1, False, 5, 4, X, 5, 4, 5, 4)
    X_src, _, _ = vc._offsurface_densities(B0)
    levels = ((int(X_src.shape[1]), int(X_src.shape[2])),)

    grad_ext = vc.compute_external_gradB_offsurf(
        B0,
        X_trg=X_trg,
        digits=2,
        adaptive=False,
        chunk_size=64,
        target_chunk_size=1,
    )
    grad_int = vc.compute_internal_gradB_offsurf(
        B0,
        X_trg=X_trg,
        digits=2,
        adaptive=False,
        chunk_size=64,
        target_chunk_size=1,
    )
    grad_int_jit = vc.compute_internal_gradB_offsurf_schedule_jit(
        B0,
        X_trg=X_trg,
        levels=levels,
        digits=2,
        chunk_size=64,
        target_chunk_size=1,
    )

    assert grad_ext.shape == (3, 3, 2, 1)
    assert grad_int.shape == (3, 3, 2, 1)
    np.testing.assert_allclose(grad_ext + grad_int, 0.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(grad_int_jit, grad_int, rtol=1e-11, atol=1e-11)
