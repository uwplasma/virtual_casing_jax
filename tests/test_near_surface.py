"""Fields and derivatives arbitrarily close to the surface, on both sides.

The oracle is the one of ``test_offsurface_level_choice``: a rotating-ellipse
boundary (``nfp = 3``) carrying a filament inside and the z-axis current
outside, so the internal branch is exactly the filament field outside the
surface and minus the z-axis field inside it.  The closed forms are wrong by
orders of magnitude within one node spacing; the routed evaluator must not care, and
must hand the targets the closed forms resolve back to them untouched.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_offsurface_level_choice import (
    A_MINOR,
    DELTA,
    R0,
    _field,
    _ring_field,
    _rotating_ellipse,
    _z_axis_field,
)

NFP, NPHI, NTHETA = 3, 16, 24


def _foot(phi, theta):
    """Surface points and outward unit normals of the rotating ellipse."""
    u = theta - NFP * phi
    R = R0 + A_MINOR * np.cos(theta) + DELTA * np.cos(u)
    Rt, Zt = -A_MINOR * np.sin(theta) - DELTA * np.sin(u), A_MINOR * np.cos(theta) + DELTA * np.cos(u)
    Rp, Zp = NFP * DELTA * np.sin(u), -NFP * DELTA * np.cos(u)
    c, s = np.cos(phi), np.sin(phi)
    point = np.stack([R * c, R * s, A_MINOR * np.sin(theta) + DELTA * np.sin(u)], axis=1)
    normal = np.cross(np.stack([Rt * c, Rt * s, Zt], axis=1),
                      np.stack([Rp * c - R * s, Rp * s + R * c, Zp], axis=1))
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    axis = np.stack([R0 * c, R0 * s, 0 * c], axis=1)
    return point, np.sign(np.sum(normal * (point - axis), axis=1))[:, None] * normal


def _targets(side, distance, shift):
    """Targets above feet ``shift`` node spacings from the nodes (0: on one)."""
    k = np.arange(6)
    point, outward = _foot((2 * np.pi / (NFP * NPHI)) * (3 * k + 1 + shift),
                           (2 * np.pi / NTHETA) * (4 * k + 2 + shift))
    return point + side * distance * A_MINOR * outward


def _oracle(points, side):
    def value(p):
        return _ring_field(p) if side > 0 else -_z_axis_field(p)

    h = 1e-5
    grad = np.stack([(value(points + h * np.eye(3)[j]) - value(points - h * np.eye(3)[j]))
                     / (2 * h) for j in range(3)], axis=-1)
    return value(points), grad


@pytest.fixture(scope="module")
def field():
    return _field(_rotating_ellipse(NPHI, NTHETA, NFP), 6)


@pytest.mark.parametrize("side", (1, -1))
@pytest.mark.parametrize("shift", (0.0, 0.37))
def test_accurate_at_every_distance_where_closed_forms_are_not(field, side, shift):
    near = jax.jit(lambda p: field.B_and_derivatives_accurate_xyz(p, order=1))
    for distance in (0.1, 1e-2, 1e-4):
        points = _targets(side, distance, shift)
        (value, gradient), _ = near(jnp.asarray(points))
        expected, expected_gradient = _oracle(points, side)
        scale = np.max(np.abs(expected))
        assert np.max(np.abs(np.asarray(value) - expected)) <= 5e-4 * scale, distance
        assert A_MINOR * np.max(np.abs(np.asarray(gradient) - expected_gradient)) <= 5e-3 * scale
    closed = np.asarray(field.B_and_derivatives_xyz(points, order=0)[0])
    assert np.max(np.abs(closed - expected)) > 0.1 * scale


def test_far_targets_are_routed_to_the_closed_forms(field):
    points = jnp.asarray(_targets(1, 3.0, 0.37))
    (value, gradient), estimate = jax.jit(
        lambda p: field.B_and_derivatives_accurate_xyz(p, order=1))(points)
    closed = field.B_and_derivatives_xyz(points, order=1)
    np.testing.assert_allclose(value, closed[0], rtol=1e-12, atol=0)
    np.testing.assert_allclose(gradient, closed[1], rtol=1e-12, atol=0)
    assert np.max(estimate) <= 1e-8
    near = field.B_and_derivatives_accurate_xyz(jnp.asarray(_targets(1, 0.1, 0.37)), order=0)
    assert np.min(near[1]) > 1e-8 and not np.allclose(near[0][0], field.B_and_derivatives_xyz(
        jnp.asarray(_targets(1, 0.1, 0.37)), order=0)[0], rtol=1e-3)


def test_estimate_is_a_jit_safe_error_bound(field):
    points = _targets(1, 1e-3, 0.37)
    expected, _ = _oracle(points, 1)
    (value,), estimate = jax.jit(
        lambda p: field.B_and_derivatives_accurate_xyz(p, order=0))(jnp.asarray(points))
    error = np.max(np.abs(np.asarray(value) - expected), axis=1) / np.max(np.abs(expected))
    assert np.all(np.asarray(estimate) >= error)
    coarse = _field(_rotating_ellipse(8, 12, NFP), 6)  # a coarser grid is flagged
    assert np.min(coarse.B_and_derivatives_accurate_xyz(points, order=0)[1]) \
        > 10.0 * np.max(np.asarray(estimate))


def test_derivatives_are_derivatives_of_the_returned_field(field):
    points = jnp.asarray(_targets(-1, 1e-2, 0.37))
    (value, gradient), _ = field.B_and_derivatives_accurate_xyz(points, order=1)
    direction = jnp.asarray(np.random.default_rng(1).standard_normal(points.shape))

    def B(p):
        return field.B_and_derivatives_accurate_xyz(p, order=0)[0][0]

    tangent = jnp.einsum("tij,tj->ti", gradient, direction)
    np.testing.assert_allclose(jax.jvp(B, (points,), (direction,))[1], tangent, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(jax.vjp(B, points)[1](direction)[0],
                               jnp.einsum("tij,ti->tj", gradient, direction), rtol=1e-8, atol=1e-9)
    residual = [float(jnp.max(jnp.abs(B(points + h * direction) - value - h * tangent)))
                for h in (4e-5, 2e-5)]
    assert residual[0] / residual[1] == pytest.approx(4.0, rel=0.1)


def test_target_on_a_node_is_finite_and_continuous(field):
    on_node = jnp.asarray(_targets(1, 0.0, 0.0))
    near = jnp.asarray(_targets(1, 1e-6, 0.0))
    (value, gradient), _ = field.B_and_derivatives_accurate_xyz(on_node, order=1)
    jacobian = jax.jacfwd(lambda p: field.B_and_derivatives_accurate_xyz(p, order=0)[0][0])(on_node)
    assert np.all(np.isfinite(value)) and np.all(np.isfinite(gradient)) and np.all(np.isfinite(jacobian))
    outside = field.B_and_derivatives_accurate_xyz(near, order=0)[0][0]
    assert np.max(np.abs(value - outside)) <= 1e-3 * np.max(np.abs(outside))


def test_branches_and_coefficient_gradient():
    """``external`` is minus ``internal``; the field is linear in the surface data."""
    surface = _rotating_ellipse(NPHI, NTHETA, NFP)
    ring = jnp.asarray(_ring_field(np.asarray(surface.gamma).reshape(3, -1).T).T.reshape(surface.gamma.shape))
    axis = surface.B_total - ring
    points = jnp.asarray(_targets(1, 1e-3, 0.37))

    def near(scale, branch="internal"):
        return _field(replace(surface, B_total=scale * ring + axis), 6) \
            .B_and_derivatives_accurate_xyz(points, order=0, branch=branch)[0][0]

    expected = _ring_field(np.asarray(points))
    assert np.max(np.abs(np.asarray(jax.jacfwd(near)(1.0)) - expected)) <= 5e-4 * np.max(np.abs(expected))
    np.testing.assert_allclose(near(1.0, "external"), -near(1.0), rtol=1e-12, atol=1e-14)
    with pytest.raises(ValueError, match="branch"):
        near(1.0, "outside")


def test_a_single_level_schedule_has_no_error_estimate():
    one_level = _field(_rotating_ellipse(NPHI, NTHETA, NFP), 6, levels=((NFP * NPHI, NTHETA),))
    with pytest.raises(ValueError, match="two levels"):
        one_level.B_and_derivatives_accurate_xyz(_targets(1, 0.1, 0.0))
