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
    _field, _offsets, _ring_field, _rotating_ellipse, _scale,
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


# ---------------------------------------------------------------------------
# The readable transcription of the paper, kept as the reference the fast
# implementation is checked against.  It is deliberately a line-by-line
# rendering of Error estimate 4 and eqs. 101-104, one target at a time, so the
# correspondence with arXiv:2012.06870 can be read off it directly.  The shipped
# code is vectorised over targets; this is what says the vectorisation did not
# change the numerics.
# ---------------------------------------------------------------------------


def _reference_root(series, target, theta, phi, direction, guess):
    variable = (phi if direction == "phi" else theta) + 1j * guess
    for _ in range(40):
        if direction == "phi":
            position, _, derivative = series.evaluate(theta, variable)
        else:
            position, derivative, _ = series.evaluate(variable, phi)
        separation = position - target
        value = np.sum(separation * separation)
        slope = 2.0 * np.sum(separation * derivative)
        if slope == 0.0:
            break
        step = value / slope
        variable = variable - step
        if abs(step) < 1e-13:
            break
    if direction == "phi":
        position, _, derivative = series.evaluate(theta, variable)
    else:
        position, derivative, _ = series.evaluate(variable, phi)
    separation = position - target
    denominator = 2.0 * np.sum(separation * derivative)
    factor = np.inf if denominator == 0.0 else 1.0 / denominator
    return variable, factor, float(np.sqrt(np.sum(np.abs(separation) ** 2)))


def _reference_line_integral(count, separation, parallel, transverse, imaginary_part):
    s = np.linspace(-np.pi, np.pi, 401)
    a = ((separation @ separation) + 2.0 * (separation @ transverse) * s
         + (transverse @ transverse) * s**2)
    b = 2.0 * (separation @ parallel) + 2.0 * (transverse @ parallel) * s
    c = parallel @ parallel
    imaginary = np.sqrt(np.maximum(4.0 * a * c - b**2, 0.0)) / (2.0 * c)
    shifted = imaginary_part - imaginary[200] + imaginary
    return float(np.trapezoid(np.exp(-count * shifted), s))


def _reference_estimate(series, targets, level, order, density_magnitude):
    """One target at a time, exactly as the paper writes it."""
    import math

    n_toroidal, n_poloidal = int(level[0]), int(level[1])
    targets = np.asarray(targets, dtype=float)
    theta_nodes = np.linspace(0.0, 2.0 * np.pi, n_poloidal, endpoint=False)
    phi_nodes = np.linspace(0.0, 2.0 * np.pi, n_toroidal, endpoint=False)
    phi_mesh, theta_mesh = np.meshgrid(phi_nodes, theta_nodes, indexing="ij")
    nodes = np.real(series.evaluate(theta_mesh.ravel(), phi_mesh.ravel())[0])

    exponent = 1.5 + order
    constant = (1.0, 3.0, 15.0, 105.0)[order]
    estimates = np.empty(len(targets))
    for index, target in enumerate(targets):
        nearest = int(np.argmin(((nodes - target) ** 2).sum(axis=1)))
        theta_star = float(theta_mesh.ravel()[nearest])
        phi_star = float(phi_mesh.ravel()[nearest])
        position, d_theta, d_phi = series.evaluate(theta_star, phi_star)
        separation = np.real(position) - target
        distance = float(np.linalg.norm(separation))
        magnitude = (density_magnitude(theta_star, phi_star)
                     if callable(density_magnitude) else float(density_magnitude))
        total = 0.0
        for direction, count, parallel, transverse in (
            ("phi", n_toroidal, np.real(d_phi), np.real(d_theta)),
            ("theta", n_poloidal, np.real(d_theta), np.real(d_phi)),
        ):
            guess = distance / max(np.linalg.norm(parallel), 1e-300)
            root, factor, root_distance = _reference_root(
                series, target, theta_star, phi_star, direction, guess)
            numerator = constant * magnitude * root_distance ** (order + 1) / (4.0 * np.pi)
            total += (numerator * abs(factor) ** exponent
                      * 4.0 * np.pi * count ** (exponent - 1.0) / math.gamma(exponent)
                      * _reference_line_integral(
                          count, separation, parallel, transverse, abs(root.imag)))
        estimates[index] = total
    return estimates


@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_the_vectorised_estimate_matches_the_paper_transcription(setup, order):
    """The fast path is the same numerics, not merely the same ballpark."""
    surface, series, magnitude, _ = setup
    points = np.concatenate([_offsets(surface, 0.05, count=6),
                             _offsets(surface, 0.2, count=6)])
    fast = kst_error_estimate(series, points, (96, 32), order, magnitude)
    slow = _reference_estimate(series, points, (96, 32), order, magnitude)
    np.testing.assert_allclose(fast, slow, rtol=1e-12, atol=0.0)


def test_the_vectorised_estimate_is_independent_of_batching(setup):
    """Targets must not influence one another through the vectorisation."""
    surface, series, magnitude, _ = setup
    points = _offsets(surface, 0.15, count=9)
    together = kst_error_estimate(series, points, (96, 32), 1, magnitude)
    apart = np.concatenate([
        kst_error_estimate(series, points[i:i + 1], (96, 32), 1, magnitude)
        for i in range(len(points))])
    np.testing.assert_allclose(together, apart, rtol=1e-12, atol=0.0)


def test_node_positions_by_fft_match_the_direct_series(setup):
    """The FFT shortcut for the level's nodes is the same grid, not an approximation."""
    from virtual_casing_jax.error_estimate import _node_positions

    _, series, _, _ = setup
    for level in ((60, 16), (120, 32)):
        n_toroidal, n_poloidal = level
        theta = np.linspace(0.0, 2.0 * np.pi, n_poloidal, endpoint=False)
        phi = np.linspace(0.0, 2.0 * np.pi, n_toroidal, endpoint=False)
        phi_mesh, theta_mesh = np.meshgrid(phi, theta, indexing="ij")
        direct = np.real(series.evaluate(theta_mesh.ravel(), phi_mesh.ravel())[0])
        np.testing.assert_allclose(_node_positions(series, *level), direct,
                                   rtol=0.0, atol=1e-11)

    # a toroidal count that is not a multiple of nfp has no per-period tiling
    # and must fall back rather than return something plausible but wrong
    n_toroidal, n_poloidal = 61, 16
    theta = np.linspace(0.0, 2.0 * np.pi, n_poloidal, endpoint=False)
    phi = np.linspace(0.0, 2.0 * np.pi, n_toroidal, endpoint=False)
    phi_mesh, theta_mesh = np.meshgrid(phi, theta, indexing="ij")
    direct = np.real(series.evaluate(theta_mesh.ravel(), phi_mesh.ravel())[0])
    np.testing.assert_allclose(_node_positions(series, n_toroidal, n_poloidal),
                               direct, rtol=0.0, atol=1e-11)


def test_evaluate_matches_an_explicit_mode_sum(setup):
    """The separable contraction is the same series, written faster."""
    _, series, _, _ = setup
    theta = np.array([0.3, 1.7, 4.2 + 0.11j])
    phi = np.array([0.2 + 0.05j, 1.1, 2.4])

    position, d_theta, d_phi = series.evaluate(theta, phi)
    for index in range(len(theta)):
        phase = np.exp(1j * (series.m[:, None] * theta[index]
                             + series.n[None, :] * phi[index]))
        R, Z = (series.coefficients * phase).sum(axis=(1, 2))
        R_t, Z_t = (series.coefficients * (1j * series.m)[:, None] * phase).sum(axis=(1, 2))
        R_p, Z_p = (series.coefficients * (1j * series.n)[None, :] * phase).sum(axis=(1, 2))
        c, s = np.cos(phi[index]), np.sin(phi[index])
        np.testing.assert_allclose(position[index], [R * c, R * s, Z], rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(d_theta[index], [R_t * c, R_t * s, Z_t],
                                   rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(d_phi[index], [R_p * c - R * s, R_p * s + R * c, Z_p],
                                   rtol=1e-12, atol=1e-14)

    # derivatives=False returns the position alone, unchanged
    np.testing.assert_allclose(series.evaluate(theta, phi, derivatives=False),
                               position, rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------------
# plan_levels: the grid the estimate picks must deliver, judged by the known
# answer rather than by the estimate that chose it.
# ---------------------------------------------------------------------------


def _identity_error(surface, levels, points, digits=12):
    """Relative error of the exterior identity: outside, B_plasma is the ring field."""
    value = np.asarray(_field(surface, digits, levels).B_plasma_xyz(points))
    exact = _ring_field(points)
    return float(np.max(np.linalg.norm(value - exact, axis=1) / _scale(surface)))


def test_a_planned_grid_meets_the_identity_it_was_sized_for():
    """End-to-end: size for 5 digits, then check the KNOWN answer, not the estimate.

    Letting the estimate certify the grid it chose would be circular; the
    rotating-ellipse oracle has an exact answer outside, so the achieved error
    is measurable independently.
    """
    from virtual_casing_jax.error_estimate import plan_levels

    surface = _rotating_ellipse(24, 24, 3)
    points = _offsets(surface, 0.30, count=12)
    levels, reported = plan_levels(surface, points, digits=5, order=0)

    assert len(levels) == 1, levels
    assert reported <= 1e-5
    assert _identity_error(surface, levels, points) <= 1e-5


def test_planning_from_a_subset_matches_planning_from_all(setup):
    """The worst target sets the grid, so the closest few give the same answer."""
    from virtual_casing_jax.error_estimate import plan_levels

    surface = _rotating_ellipse(24, 24, 3)
    spread = np.concatenate([_offsets(surface, d, count=8)
                             for d in (0.08, 0.15, 0.3, 0.6)])
    everything, _ = plan_levels(surface, spread, digits=5, order=0, subset=None)
    closest, _ = plan_levels(surface, spread, digits=5, order=0, subset=8)
    assert closest == everything, (closest, everything)


def test_a_higher_derivative_order_asks_for_a_finer_grid(setup):
    """Sizing for the field alone does not deliver the same digits for its curvature."""
    from virtual_casing_jax.error_estimate import plan_levels

    surface = _rotating_ellipse(24, 24, 3)
    points = _offsets(surface, 0.25, count=8)
    for_field, _ = plan_levels(surface, points, digits=5, order=0)
    for_curvature, _ = plan_levels(surface, points, digits=5, order=2)
    assert for_curvature[0][0] >= for_field[0][0]
    assert for_curvature[0][1] >= for_field[0][1]


def test_a_planned_level_leaves_no_branch_in_the_schedule():
    """One level means no lax.cond, so the VJP is that grid's and nothing else."""
    from virtual_casing_jax.error_estimate import plan_levels

    surface = _rotating_ellipse(12, 12, 3)
    points = _offsets(surface, 0.35, count=4)
    levels, _ = plan_levels(surface, points, digits=4, order=0)
    field = _field(surface, 12, levels)
    pinned = _field(surface, 12, (levels[0],))

    def total(scale, target):
        return jnp.sum(target.B_plasma_xyz(jnp.asarray(points) * scale))

    np.testing.assert_allclose(float(jax.grad(total)(1.0, field)),
                               float(jax.grad(total)(1.0, pinned)), rtol=1e-12)
    step = 1e-6
    finite = (total(1.0 + step, field) - total(1.0 - step, field)) / (2.0 * step)
    np.testing.assert_allclose(float(jax.grad(total)(1.0, field)), float(finite),
                               rtol=2e-5)


def test_planning_is_opt_in_and_the_accuracy_request_is_mandatory():
    """No default `digits`, and the default schedule is untouched by this module.

    A caller who never plans gets exactly today's behaviour. Giving `digits` a
    default would silently re-decide the accuracy of everyone who adopts
    planning without stating a request; on the nfp 2/3/5 oracles the smallest
    default that never degrades today's achieved error is 5, and it costs about
    1.7x the default schedule on average, so the choice is not free and should
    not be made on a caller's behalf.
    """
    import inspect

    from virtual_casing_jax.error_estimate import plan_levels

    parameter = inspect.signature(plan_levels).parameters["digits"]
    assert parameter.default is inspect.Parameter.empty
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY

    surface = _rotating_ellipse(12, 12, 3)
    with pytest.raises(TypeError):
        plan_levels(surface, _offsets(surface, 0.3, count=2))

    # and the unplanned schedule is still the nfp-sized two-level default
    assert _field(surface, 4).schedule_levels == ((36, 12), (72, 24))


def test_planning_rejects_malformed_targets():
    """The one branch in `plan_levels` its other tests never reach."""
    from virtual_casing_jax.error_estimate import plan_levels

    surface = _rotating_ellipse(12, 12, 3)
    for bad in (np.zeros(3), np.zeros((4, 2)), np.zeros((2, 2, 3))):
        with pytest.raises(ValueError, match=r"targets must have shape"):
            plan_levels(surface, bad, digits=4, order=0)
