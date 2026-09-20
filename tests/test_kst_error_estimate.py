"""The a-priori estimate bounds the trapezoid error and sizes a grid.

Scored against the TRUE error of a single level, which is that level differenced
against a far finer one of the SAME source data: differencing two levels of one
source grid cancels the source-data truncation floor and leaves exactly what the
estimate models.  Scoring against an analytic answer instead would compare the
estimate with a floor it does not claim to describe.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from virtual_casing_jax.error_estimate import (
    GridSizeError,
    boundary_series_from_gamma,
    density_magnitude_from_surface,
    kst_error_estimate,
    required_levels,
)

from test_offsurface_level_choice import (  # noqa: E402
    _field, _offsets, _rotating_ellipse, _scale,
)

NFP, NPHI, NTHETA = 3, 32, 32
REFERENCE_LEVEL = (768, 256)


@pytest.fixture(scope="module")
def setup():
    surface = _rotating_ellipse(NPHI, NTHETA, NFP)
    return (surface,
            boundary_series_from_gamma(np.asarray(surface.gamma), NFP),
            density_magnitude_from_surface(surface),
            _field(surface, 12, (REFERENCE_LEVEL,)))


def _derivative(field, points, order):
    function = lambda p: field.B_plasma_xyz(p[None, :])[0]  # noqa: E731
    for _ in range(order):
        function = jax.jacfwd(function)
    return np.asarray(jax.jit(jax.vmap(function))(jnp.asarray(points)))


def _true_error(surface, reference, level, points, order):
    value = _derivative(_field(surface, 12, (level,)), points, order)
    exact = _derivative(reference, points, order)
    return np.sqrt(((value - exact) ** 2).reshape(len(points), -1).sum(1))


def test_the_series_reproduces_the_boundary_at_its_own_nodes():
    surface = _rotating_ellipse(8, 8, NFP)
    series = boundary_series_from_gamma(np.asarray(surface.gamma), NFP)
    theta = np.linspace(0.0, 2 * np.pi, 8, endpoint=False)
    phi = np.linspace(0.0, 2 * np.pi / NFP, 8, endpoint=False)
    mesh_phi, mesh_theta = np.meshgrid(phi, theta, indexing="ij")
    position, _, _ = series.evaluate(mesh_theta.ravel(), mesh_phi.ravel())
    expected = np.asarray(surface.gamma).reshape(3, -1).T
    np.testing.assert_allclose(np.real(position), expected, rtol=0.0, atol=1e-12)
    assert np.max(np.abs(np.imag(position))) < 1e-12


@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_the_estimate_brackets_the_trapezoid_error(setup, order):
    """Conservative, and never optimistic by more than the safety factor covers."""
    surface, series, magnitude, reference = setup
    level = (96, 32)
    points = _offsets(surface, 0.1, count=8)
    true = _true_error(surface, reference, level, points, order)
    estimate = kst_error_estimate(series, points, level, order, magnitude)
    ratio = estimate / true
    # measured medians are 1.3 to 2.2 over four distances and four orders
    assert 1.0 <= np.median(ratio) <= 4.0, np.median(ratio)
    assert ratio.min() >= 0.25, ratio.min()
    assert ratio.max() <= 20.0, ratio.max()


def test_the_estimate_falls_with_distance_and_rises_with_order(setup):
    surface, series, magnitude, _ = setup
    level = (96, 32)
    near = kst_error_estimate(series, _offsets(surface, 0.05, count=6), level, 0, magnitude)
    far = kst_error_estimate(series, _offsets(surface, 0.30, count=6), level, 0, magnitude)
    assert far.max() < near.max()
    points = _offsets(surface, 0.1, count=6)
    by_order = [kst_error_estimate(series, points, level, k, magnitude).max() for k in range(4)]
    assert by_order == sorted(by_order)


def test_the_jacobian_is_part_of_the_density(setup):
    """Passing |n x B| rather than |area_vector x B| is wrong by the Jacobian.

    On this boundary the Jacobian is about 0.39, so dropping it inflates the
    estimate by about 2.5; on a larger surface it would deflate it instead.
    Either way it is a real factor, not a normalisation that cancels.
    """
    surface, series, magnitude, _ = setup
    points = _offsets(surface, 0.1, count=4)
    with_jacobian = kst_error_estimate(series, points, (96, 32), 0, magnitude)
    unit = np.linalg.norm(
        np.cross(np.asarray(surface.normal), np.asarray(surface.B_total), axis=0), axis=0)
    without = kst_error_estimate(series, points, (96, 32), 0, float(unit.mean()))

    jacobian = np.linalg.norm(np.asarray(surface.area_vector), axis=0)
    assert 0.3 < jacobian.mean() < 0.5
    # the estimate is linear in the density, so the two differ by that factor
    ratio = np.median(with_jacobian / without)
    assert 0.5 * jacobian.min() <= ratio <= 2.0 * jacobian.max(), (ratio, jacobian.mean())
    assert ratio < 0.8


def test_required_levels_reaches_the_tolerance_it_reports(setup):
    """The sized grid's ACHIEVED error meets the request, measured after the fact."""
    surface, series, magnitude, reference = setup
    points = _offsets(surface, 0.2, count=6)
    scale = _scale(surface)
    levels, reported = required_levels(
        series, points, digits=6, order=0, density_magnitude=magnitude,
        scale=scale, base_level=(48, 16))
    assert reported <= 1e-6
    achieved = _true_error(surface, reference, levels[0], points, 0).max() / scale
    assert achieved <= 1e-6, (levels, achieved)
    # a single level, so the quadrature is branch-free
    assert len(levels) == 1

    coarser, _ = required_levels(
        series, points, digits=3, order=0, density_magnitude=magnitude,
        scale=scale, base_level=(48, 16))
    assert coarser[0][0] <= levels[0][0]


def test_an_impossible_request_is_refused_not_silently_capped(setup):
    surface, series, magnitude, _ = setup
    points = _offsets(surface, 0.01, count=4)
    with pytest.raises(GridSizeError) as info:
        required_levels(series, points, digits=10, order=3,
                        density_magnitude=magnitude, scale=_scale(surface),
                        base_level=(48, 16), cap=256)
    assert info.value.cap == 256
    assert info.value.achieved > 1e-10
    assert "raise the cap" in str(info.value)
