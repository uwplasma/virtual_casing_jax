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

from virtual_casing_jax import ExteriorFieldConfig, VirtualCasingExteriorField

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


def _accurate(field, points, order=1, **kwargs):
    return jax.jit(lambda p: field.B_and_derivatives_accurate_xyz(p, order=order, **kwargs))(
        jnp.asarray(points))


@pytest.mark.parametrize("side", (1, -1))
@pytest.mark.parametrize("shift", (0.0, 0.37))
def test_accurate_at_every_distance_where_closed_forms_are_not(field, side, shift):
    for distance in (0.1, 1e-2, 1e-4):
        points = _targets(side, distance, shift)
        (value, gradient), _ = _accurate(field, points)
        expected, expected_gradient = _oracle(points, side)
        scale = np.max(np.abs(expected))
        error = np.max(np.abs(np.asarray(value) - expected), axis=1) / scale
        assert error.max() <= 5e-4, distance
        assert A_MINOR * np.max(np.abs(np.asarray(gradient) - expected_gradient)) <= 5e-3 * scale
    closed = np.asarray(field.B_and_derivatives_xyz(points, order=0)[0])
    assert np.max(np.abs(closed - expected)) > 0.1 * scale  # unlike the closed forms


def test_far_targets_keep_the_closed_forms_and_coarse_data_is_flagged(field):
    points = _targets(1, 3.0, 0.37)
    values, estimate = _accurate(field, points)
    for got, closed in zip(values, field.B_and_derivatives_xyz(points, order=1)):
        np.testing.assert_allclose(got, closed, rtol=1e-12, atol=0)
    assert np.max(estimate) <= 1e-8
    near = _targets(1, 1e-3, 0.37)
    (value,), estimate = _accurate(field, near, 0)
    expected = _oracle(near, 1)[0]  # the estimate bounds the error
    assert np.all(estimate >= np.max(np.abs(value - expected), axis=1) / np.max(np.abs(expected)))
    coarse = _field(_rotating_ellipse(8, 12, NFP), 6)
    assert np.min(coarse.B_and_derivatives_accurate_xyz(near, order=0)[1]) \
        > 10.0 * np.max(estimate)


def test_derivatives_are_derivatives_of_the_returned_field_and_nodes_are_finite(field):
    points = jnp.asarray(_targets(-1, 1e-2, 0.37))
    (_, gradient), _ = _accurate(field, points)
    direction = jnp.asarray(np.random.default_rng(1).standard_normal(points.shape))

    def B(p):
        return field.B_and_derivatives_accurate_xyz(p, order=0)[0][0]

    tangent = jnp.einsum("tij,tj->ti", gradient, direction)
    np.testing.assert_allclose(jax.jvp(B, (points,), (direction,))[1], tangent, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(jax.vjp(B, points)[1](direction)[0],
                               jnp.einsum("tij,ti->tj", gradient, direction), rtol=1e-8, atol=1e-9)
    on_node = jnp.asarray(_targets(1, 0.0, 0.0))
    (node_value, node_gradient), _ = _accurate(field, on_node)
    assert all(np.all(np.isfinite(a)) for a in (node_value, node_gradient, jax.jacfwd(B)(on_node)))
    outside = B(jnp.asarray(_targets(1, 1e-6, 0.0)))
    assert np.max(np.abs(node_value - outside)) <= 1e-3 * np.max(np.abs(outside))


def test_branches_coefficient_gradient_and_single_level():
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
    one_level = _field(surface, 6, levels=((NFP * NPHI, NTHETA),))
    with pytest.raises(ValueError, match="two levels"):
        one_level.B_and_derivatives_accurate_xyz(points)


@pytest.mark.parametrize("side", (1, -1))
def test_the_default_field_methods_are_accurate_next_to_the_surface(side):
    surface = _rotating_ellipse(NPHI, NTHETA, NFP)
    config = ExteriorFieldConfig(digits=6, src_nphi=NPHI, src_ntheta=NTHETA)
    default = VirtualCasingExteriorField(surface, config)
    legacy = VirtualCasingExteriorField(surface, replace(config, near_surface=False))
    points = _targets(side, 1e-3, 0.37)
    expected, expected_gradient = _oracle(points, side)
    scale = np.max(np.abs(expected))
    value, estimate = default.B_plasma_xyz(points, return_estimate=True)
    assert np.max(np.abs(value - expected)) <= 5e-4 * scale and np.max(estimate) < 1e-2
    assert A_MINOR * np.max(np.abs(default.gradB_plasma_xyz(points) - expected_gradient)) <= 5e-3 * scale
    assert np.max(np.abs(legacy.B_plasma_xyz(points) - expected)) > 0.1 * scale
    far = _targets(1, 3.0, 0.37)  # transposed layout and the closed forms agree
    np.testing.assert_allclose(default.B_plasma_xyz(far.T), legacy.B_plasma_xyz(far.T), rtol=1e-6)
    np.testing.assert_allclose(default.gradB_plasma_xyz(far), legacy.gradB_plasma_xyz(far), rtol=1e-6, atol=1e-9)
