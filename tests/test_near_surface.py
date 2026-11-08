"""Fields and derivatives arbitrarily close to the surface, on both sides.

The oracle is the one of ``test_offsurface_level_choice``: a rotating-ellipse
boundary (``nfp = 3``) carrying a filament inside and the z-axis current
outside, so the internal branch is exactly the filament field outside the
surface and minus the z-axis field inside it, and both layer densities are
exercised.  The closed-form path is accurate a few node spacings away and
wrong by orders of magnitude within one; the near-surface path must not care.
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
DISTANCES = (0.1, 1e-2, 1e-4)


def _oracle(points, side):
    """Internal-branch field and its gradient (central differences of the oracle)."""
    def value(p):
        return _ring_field(p) if side > 0 else -_z_axis_field(p)

    step = 1e-5
    grad = np.stack([(value(points + step * np.eye(3)[j]) - value(points - step * np.eye(3)[j]))
                     / (2 * step) for j in range(3)], axis=-1)
    return value(points), grad


def _foot_points(phi, theta):
    """Surface points and outward unit normals of the rotating ellipse at angles."""
    u = theta - NFP * phi
    R = R0 + A_MINOR * np.cos(theta) + DELTA * np.cos(u)
    Rt, Zt = -A_MINOR * np.sin(theta) - DELTA * np.sin(u), A_MINOR * np.cos(theta) + DELTA * np.cos(u)
    Rp, Zp = NFP * DELTA * np.sin(u), -NFP * DELTA * np.cos(u)
    Z = A_MINOR * np.sin(theta) + DELTA * np.sin(u)
    c, s = np.cos(phi), np.sin(phi)
    point = np.stack([R * c, R * s, Z], axis=1)
    normal = np.cross(np.stack([Rt * c, Rt * s, Zt], axis=1),
                      np.stack([Rp * c - R * s, Rp * s + R * c, Zp], axis=1))
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    axis = np.stack([R0 * c, R0 * s, 0 * c], axis=1)
    return point, np.sign(np.sum(normal * (point - axis), axis=1))[:, None] * normal


@pytest.fixture(scope="module")
def setup():
    field = _field(_rotating_ellipse(NPHI, NTHETA, NFP), 6)
    return field, _foot_points


def _targets(setup, side, distance, shift):
    """Targets above foot points ``shift`` node spacings from the nodes (0: on one)."""
    _, foot = setup
    steps = np.arange(6)
    phi = (2.0 * np.pi / (NFP * NPHI)) * (3 * steps + 1 + shift)
    theta = (2.0 * np.pi / NTHETA) * (4 * steps + 2 + shift)
    point, outward = foot(phi, theta)
    return point + side * distance * A_MINOR * outward


@pytest.mark.parametrize("side", (1, -1))
@pytest.mark.parametrize("shift", (0.0, 0.37))
def test_accurate_at_every_distance_where_closed_forms_are_not(setup, side, shift):
    field, _ = setup
    near = jax.jit(lambda p: field.B_and_derivatives_near_surface_xyz(p, order=1))
    for distance in DISTANCES:
        points = _targets(setup, side, distance, shift)
        (value, gradient), _ = near(jnp.asarray(points))
        expected, expected_gradient = _oracle(points, side)
        scale = np.max(np.abs(expected))
        assert np.max(np.abs(np.asarray(value) - expected)) <= 5e-4 * scale, distance
        assert (A_MINOR * np.max(np.abs(np.asarray(gradient) - expected_gradient))
                <= 5e-3 * scale), distance
        if distance == 1e-4:
            closed = np.asarray(field.B_and_derivatives_xyz(points, order=0)[0])
            assert np.max(np.abs(closed - expected)) > 0.1 * scale


def test_estimate_is_a_jit_safe_error_bound(setup):
    field, _ = setup
    points = _targets(setup, 1, 1e-3, 0.37)
    expected, _ = _oracle(points, 1)
    (value,), estimate = jax.jit(
        lambda p: field.B_and_derivatives_near_surface_xyz(p, order=0))(jnp.asarray(points))
    assert estimate.shape == (len(points),)
    error = np.max(np.abs(np.asarray(value) - expected), axis=1) / np.max(np.abs(expected))
    assert np.all(np.asarray(estimate) >= error)
    # a coarser source grid is flagged by a larger estimate
    coarse = _field(_rotating_ellipse(8, 12, NFP), 6)
    assert np.min(np.asarray(coarse.B_and_derivatives_near_surface_xyz(
        points, order=0)[1])) > 10.0 * np.max(np.asarray(estimate))


def test_derivatives_are_derivatives_of_the_returned_field(setup):
    field, _ = setup
    points = jnp.asarray(_targets(setup, -1, 1e-2, 0.37))
    (value, gradient), _ = field.B_and_derivatives_near_surface_xyz(points, order=1)
    direction = jnp.asarray(np.random.default_rng(1).standard_normal(points.shape))

    def B(p):
        return field.B_and_derivatives_near_surface_xyz(p, order=0)[0][0]

    tangent = jnp.einsum("tij,tj->ti", gradient, direction)
    np.testing.assert_allclose(jax.jvp(B, (points,), (direction,))[1], tangent,
                               rtol=1e-8, atol=1e-9)
    cotangent = jax.vjp(B, points)[1](direction)[0]
    np.testing.assert_allclose(cotangent, jnp.einsum("tij,ti->tj", gradient, direction),
                               rtol=1e-8, atol=1e-9)
    residual = [float(jnp.max(jnp.abs(B(points + h * direction) - value - h * tangent)))
                for h in (4e-5, 2e-5)]
    assert residual[0] / residual[1] == pytest.approx(4.0, rel=0.1)


def test_branches_and_coefficient_gradient():
    """``external`` is minus ``internal``; the field is linear in the surface data."""
    surface = _rotating_ellipse(NPHI, NTHETA, NFP)
    g = np.asarray(surface.gamma).reshape(3, -1).T
    ring = jnp.asarray(_ring_field(g).T.reshape(surface.gamma.shape))
    axis = surface.B_total - ring
    points = jnp.asarray(_targets((None, _foot_points), 1, 1e-3, 0.37))

    def near(scale, branch="internal"):
        data = replace(surface, B_total=scale * ring + axis)
        field = VirtualCasingExteriorField(data, ExteriorFieldConfig(
            digits=6, src_nphi=NPHI, src_ntheta=NTHETA))
        return field.B_and_derivatives_near_surface_xyz(points, order=0, branch=branch)[0][0]

    sensitivity = jax.jacfwd(near)(1.0)
    expected = _ring_field(np.asarray(points))
    assert np.max(np.abs(np.asarray(sensitivity) - expected)) <= 5e-4 * np.max(np.abs(expected))
    np.testing.assert_allclose(near(1.0, "external"), -near(1.0), rtol=1e-12, atol=1e-14)
    with pytest.raises(ValueError, match="branch"):
        near(1.0, "outside")
