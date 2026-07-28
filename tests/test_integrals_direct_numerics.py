import numpy as np
import pytest
import jax.numpy as jnp

from virtual_casing_jax.integrals import (
    biotsavart_fx_u_eval,
    biotsavart_fxd_u_eval,
    computeB_offsurface_adaptive_schedule,
    computeGradB_offsurface_adaptive_schedule,
    field_period_target_coords,
    laplace_dx_u_eval,
    laplace_fxd2_u_eval,
    laplace_fxd2_u_eval_vec,
    laplace_fxd_u_eval,
    laplace_fxd_u_eval_vec,
)
from virtual_casing_jax.kernels import (
    biotsavart_fx_u,
    biotsavart_fxd_u,
    laplace_dx_u,
    laplace_fxd2_u,
    laplace_fxd_u,
)


def _direct_inputs():
    X_src = jnp.array(
        [
            [[0.0, 0.7, -0.2], [1.1, -0.4, 0.5]],
            [[0.1, -0.3, 0.9], [0.6, 1.2, -0.7]],
            [[-0.5, 0.4, 1.3], [-1.1, 0.8, 0.2]],
        ],
        dtype=jnp.float64,
    )
    X_trg = jnp.array(
        [
            [2.0, -1.5, 0.3, 1.4, -0.8],
            [1.5, 0.8, -1.4, 0.2, -1.1],
            [0.6, -1.2, 1.8, -0.9, 1.0],
        ],
        dtype=jnp.float64,
    )
    density = jnp.array([[0.8, -0.2, 0.5], [1.3, -0.7, 0.4]], dtype=jnp.float64)
    density_vec = jnp.array(
        [
            [[0.3, -0.1, 0.7], [0.2, -0.4, 0.9]],
            [[-0.5, 0.6, 0.1], [0.8, 0.2, -0.3]],
            [[0.4, 0.9, -0.6], [-0.2, 0.5, 0.1]],
        ],
        dtype=jnp.float64,
    )
    normal = jnp.array(
        [
            [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
        ],
        dtype=jnp.float64,
    )
    area_elem = jnp.array([[0.5, 1.2, 0.7], [1.1, 0.9, 0.4]], dtype=jnp.float64)
    return X_src, X_trg, density, density_vec, normal, area_elem


def _flatten_sources(X_src):
    return jnp.asarray(X_src).reshape((3, -1)).T


def _flatten_targets(X_trg):
    return jnp.asarray(X_trg).reshape((3, -1)).T


def _manual_laplace_fxd(X_src, X_trg, density, area_elem):
    weights = jnp.asarray(density).reshape(-1) * jnp.asarray(area_elem).reshape(-1)
    dx = _flatten_targets(X_trg)[:, None, :] - _flatten_sources(X_src)[None, :, :]
    out = jnp.sum(laplace_fxd_u(dx, weights), axis=1)
    return out.T


def _manual_laplace_fxd2(X_src, X_trg, density, area_elem):
    weights = jnp.asarray(density).reshape(-1) * jnp.asarray(area_elem).reshape(-1)
    dx = _flatten_targets(X_trg)[:, None, :] - _flatten_sources(X_src)[None, :, :]
    out = jnp.sum(laplace_fxd2_u(dx, weights), axis=1)
    return out.reshape((out.shape[0], 9)).T


def _manual_biotsavart_fx(X_src, X_trg, density_vec, area_elem):
    weights = jnp.asarray(density_vec).reshape((3, -1)) * jnp.asarray(area_elem).reshape((1, -1))
    dx = _flatten_targets(X_trg)[:, None, :] - _flatten_sources(X_src)[None, :, :]
    fvec = weights.T[None, :, :]
    out = jnp.sum(biotsavart_fx_u(dx, fvec), axis=1)
    return out.T


def _manual_biotsavart_fxd(X_src, X_trg, density_vec, area_elem):
    weights = jnp.asarray(density_vec).reshape((3, -1)) * jnp.asarray(area_elem).reshape((1, -1))
    dx = _flatten_targets(X_trg)[:, None, :] - _flatten_sources(X_src)[None, :, :]
    fvec = weights.T[None, :, :]
    out = jnp.sum(biotsavart_fxd_u(dx, fvec), axis=1)
    return jnp.transpose(out, (1, 2, 0))


def _manual_laplace_dx(X_src, normal, X_trg, density, area_elem):
    weights = jnp.asarray(density).reshape(-1) * jnp.asarray(area_elem).reshape(-1)
    dx = _flatten_targets(X_trg)[:, None, :] - _flatten_sources(X_src)[None, :, :]
    n = jnp.asarray(normal).reshape((3, -1)).T[None, :, :]
    out = jnp.sum(laplace_dx_u(dx, n, weights), axis=1)
    return out.reshape((1, -1))


@pytest.mark.parametrize(
    "eval_fn,manual_fn,args",
    [
        (
            laplace_fxd_u_eval,
            _manual_laplace_fxd,
            ("X_src", "X_trg", "density", "area_elem"),
        ),
        (
            laplace_fxd2_u_eval,
            _manual_laplace_fxd2,
            ("X_src", "X_trg", "density", "area_elem"),
        ),
        (
            biotsavart_fx_u_eval,
            _manual_biotsavart_fx,
            ("X_src", "X_trg", "density_vec", "area_elem"),
        ),
        (
            biotsavart_fxd_u_eval,
            _manual_biotsavart_fxd,
            ("X_src", "X_trg", "density_vec", "area_elem"),
        ),
        (
            laplace_dx_u_eval,
            _manual_laplace_dx,
            ("X_src", "normal", "X_trg", "density", "area_elem"),
        ),
    ],
)
def test_direct_integral_evaluators_match_explicit_kernel_sums(eval_fn, manual_fn, args):
    X_src, X_trg, density, density_vec, normal, area_elem = _direct_inputs()
    values = {
        "X_src": X_src,
        "X_trg": X_trg,
        "density": density,
        "density_vec": density_vec,
        "normal": normal,
        "area_elem": area_elem,
    }
    call_args = [values[name] for name in args]
    expected = manual_fn(*call_args)

    unchunked = eval_fn(*call_args, chunk_size=None, target_chunk_size=None)
    source_chunked = eval_fn(*call_args, chunk_size=4, target_chunk_size=None)
    target_chunked = eval_fn(*call_args, chunk_size=4, target_chunk_size=2)
    target_only_chunked = eval_fn(*call_args, chunk_size=None, target_chunk_size=2)

    np.testing.assert_allclose(unchunked, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(source_chunked, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(target_chunked, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(target_only_chunked, expected, rtol=1e-12, atol=1e-12)


def test_direct_integral_evaluators_restore_target_grid_shape():
    X_src, _, density, density_vec, normal, area_elem = _direct_inputs()
    X_trg_grid = jnp.arange(36.0, dtype=jnp.float64).reshape((3, 3, 4)) / 7.0 + 2.0

    laplace = laplace_fxd_u_eval(X_src, X_trg_grid, density, area_elem, chunk_size=4, target_chunk_size=5)
    biot = biotsavart_fx_u_eval(X_src, X_trg_grid, density_vec, area_elem, chunk_size=4, target_chunk_size=5)
    dlayer = laplace_dx_u_eval(X_src, normal, X_trg_grid, density, area_elem, chunk_size=4, target_chunk_size=5)

    assert laplace.shape == (3, 12)
    assert biot.shape == (3, 12)
    assert dlayer.shape == (1, 12)


def test_laplace_vector_density_wrappers_match_componentwise_evaluations():
    X_src, X_trg, _, density_vec, _, area_elem = _direct_inputs()

    grad = laplace_fxd_u_eval_vec(X_src, X_trg, density_vec, area_elem, chunk_size=4, target_chunk_size=2)
    hess = laplace_fxd2_u_eval_vec(X_src, X_trg, density_vec, area_elem, chunk_size=4, target_chunk_size=2)

    for component in range(3):
        np.testing.assert_allclose(
            grad[component],
            laplace_fxd_u_eval(X_src, X_trg, density_vec[component], area_elem, chunk_size=4, target_chunk_size=2),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            hess[component],
            laplace_fxd2_u_eval(X_src, X_trg, density_vec[component], area_elem, chunk_size=4, target_chunk_size=2),
            rtol=1e-12,
            atol=1e-12,
        )


def test_field_period_target_coords_selects_first_period_and_validates_divisibility():
    X_quad = jnp.arange(3 * 12 * 6, dtype=jnp.float64).reshape((3, 12, 6))

    got = field_period_target_coords(X_quad, trg_nt=3, trg_np=2, nfp=2)

    np.testing.assert_allclose(got, X_quad[:, ::2, ::3][:, :3, :])
    assert got.shape == (3, 3, 2)

    with pytest.raises(ValueError, match="quad_nt"):
        field_period_target_coords(X_quad, trg_nt=4, trg_np=2, nfp=2)

    with pytest.raises(ValueError, match="quad_np"):
        field_period_target_coords(X_quad, trg_nt=3, trg_np=4, nfp=2)


def test_direct_integral_evaluators_validate_shapes_and_source_weights():
    X_src, X_trg, density, density_vec, normal, area_elem = _direct_inputs()

    with pytest.raises(ValueError, match="X_src and X_trg"):
        laplace_fxd_u_eval(X_src[0], X_trg, density, area_elem)

    with pytest.raises(ValueError, match="density/area_elem"):
        laplace_fxd_u_eval(X_src, X_trg, density[:, :-1], area_elem)

    with pytest.raises(ValueError, match="X_src, X_trg, density_vec"):
        biotsavart_fx_u_eval(X_src, X_trg, density_vec[:2], area_elem)

    with pytest.raises(ValueError, match="area_elem"):
        biotsavart_fxd_u_eval(X_src, X_trg, density_vec, area_elem[:, :-1])

    with pytest.raises(ValueError, match="X_src, X_trg, n_src"):
        laplace_dx_u_eval(X_src, normal[:2], X_trg, density, area_elem)


def test_empty_adaptive_schedules_are_rejected_before_resampling():
    X_src, X_trg, density, density_vec, _, _ = _direct_inputs()

    with pytest.raises(ValueError, match="levels"):
        computeB_offsurface_adaptive_schedule(X_src, density, density_vec, X_trg, levels=())

    with pytest.raises(ValueError, match="levels"):
        computeGradB_offsurface_adaptive_schedule(X_src, density, density_vec, X_trg, levels=())
