"""The off-surface schedule chooses its level by a calibrated error estimate.

The oracle is asset-free.  A rotating-ellipse boundary with ``nfp`` periods
carries the sum of two analytic fields: a circular filament on the magnetic axis
(inside the surface) and the straight current on the z axis (outside it).  For
any closed surface enclosing the filament and excluding the z axis, the internal
virtual-casing branch must return the filament field outside and minus the
z-axis field inside, so both layer densities are exercised and the answer is
known exactly.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from virtual_casing_jax import (
    ExteriorFieldConfig,
    VirtualCasingExteriorField,
    VmecSurfaceFieldData,
    default_schedule_levels,
)

R0, A_MINOR, DELTA, B0R0, CURRENT = 1.0, 0.30, 0.08, 1.0, 3.0e5
MU0 = 4.0e-7 * np.pi


def _ring_field(points, segments=2048):
    angle = 2.0 * np.pi * np.arange(segments) / segments
    source = np.stack([R0 * np.cos(angle), R0 * np.sin(angle), np.zeros(segments)], axis=1)
    tangent = np.stack([-np.sin(angle), np.cos(angle), np.zeros(segments)], axis=1)
    separation = np.asarray(points)[:, None, :] - source[None]
    kernel = np.cross(tangent[None], separation) / np.linalg.norm(
        separation, axis=-1, keepdims=True) ** 3
    return MU0 * CURRENT * R0 / (2.0 * segments) * kernel.sum(axis=1)


def _z_axis_field(points):
    x, y = np.asarray(points)[:, 0], np.asarray(points)[:, 1]
    return B0R0 * np.stack([-y / (x * x + y * y), x / (x * x + y * y),
                            np.zeros_like(x)], axis=1)


def _rotating_ellipse(nphi, ntheta, nfp):
    theta = jnp.linspace(0.0, 2 * jnp.pi, ntheta, endpoint=False)
    phi = jnp.linspace(0.0, 2 * jnp.pi / nfp, nphi, endpoint=False)
    ph, th = jnp.meshgrid(phi, theta, indexing="ij")
    u = th - nfp * ph
    R = R0 + A_MINOR * jnp.cos(th) + DELTA * jnp.cos(u)
    Z = A_MINOR * jnp.sin(th) + DELTA * jnp.sin(u)
    Rt = -A_MINOR * jnp.sin(th) - DELTA * jnp.sin(u)
    Zt = A_MINOR * jnp.cos(th) + DELTA * jnp.cos(u)
    Rp, Zp = nfp * DELTA * jnp.sin(u), -nfp * DELTA * jnp.cos(u)
    c, s = jnp.cos(ph), jnp.sin(ph)
    gamma = jnp.stack([R * c, R * s, Z], axis=0)
    e_th = jnp.stack([Rt * c, Rt * s, Zt], axis=0)
    e_ph = jnp.stack([Rp * c - R * s, Rp * s + R * c, Zp], axis=0)
    area = jnp.cross(e_th, e_ph, axis=0)
    points = np.asarray(gamma).reshape(3, -1).T
    B = _z_axis_field(points) + _ring_field(points)
    return VmecSurfaceFieldData(
        gamma=gamma, B_total=jnp.asarray(B.T.reshape(gamma.shape)),
        normal=area / jnp.linalg.norm(area, axis=0), area_vector=area,
        theta=theta, phi=phi, nfp=nfp, stellsym=False, signgs=1,
        source_convention="synthetic")


def _field(surface, digits, levels=None):
    return VirtualCasingExteriorField(
        surface,
        ExteriorFieldConfig(
            digits=digits, levels=levels,
            src_nphi=int(surface.gamma.shape[1]), src_ntheta=int(surface.gamma.shape[2]),
        ),
    )


def _offsets(surface, distance, count=16):
    """Points at a signed ``distance`` along the outward unit normal."""
    g = np.asarray(surface.gamma).reshape(3, -1).T
    n = np.asarray(surface.normal).reshape(3, -1).T
    phi = np.arctan2(g[:, 1], g[:, 0])
    axis = np.stack([R0 * np.cos(phi), R0 * np.sin(phi), np.zeros_like(phi)], axis=1)
    outward = np.sign(np.sum(n * (g - axis), axis=1))[:, None] * n
    index = np.random.default_rng(0).choice(g.shape[0], size=count, replace=False)
    return g[index] + distance * outward[index]


def _spacing(field):
    n_toroidal, n_poloidal = field.schedule_levels[-1]
    return max(2 * np.pi * (R0 + A_MINOR + DELTA) / n_toroidal,
               2 * np.pi * (A_MINOR + DELTA) / n_poloidal)


def _scale(surface):
    return float(np.sqrt(np.mean(np.sum(np.asarray(surface.B_total) ** 2, axis=0))))


def test_default_schedule_counts_the_whole_torus_per_period():
    """A level counts the whole torus while the source grid holds one period."""
    assert default_schedule_levels(32, 32, 5) == ((160, 32), (320, 64))
    assert default_schedule_levels(32, 32, 1) == ((32, 32), (64, 64))
    surface = _rotating_ellipse(8, 8, 3)
    assert _field(surface, 3).schedule_levels == ((24, 8), (48, 16))


# ``nphi`` per period chosen so the finest level counts about 128 points around
# the whole torus whatever ``nfp`` is: the identities are then testable at the
# same physical stand-off, and two grid spacings still fit inside the plasma.
@pytest.mark.parametrize("nfp, nphi", [(2, 32), (3, 22), (5, 13)])
def test_identities_hold_outside_and_inside_at_the_default_schedule(nfp, nphi):
    """Both known answers meet the requested digits beyond two grid spacings."""
    digits = 3
    surface = _rotating_ellipse(nphi, 16, nfp)
    field = _field(surface, digits)
    scale, h, tol = _scale(surface), _spacing(field), 10.0 ** -digits
    # the interior target must clear the surface by 2h and still stay inside
    assert 2.0 * h < A_MINOR - DELTA

    outside = _offsets(surface, 2.5 * h)
    value, estimate = field.B_plasma_xyz(outside, return_estimate=True)
    error = np.linalg.norm(np.asarray(value) - _ring_field(outside), axis=1) / scale
    assert error.max() <= tol, error.max()
    assert np.asarray(estimate).max() <= tol

    inside = _offsets(surface, -2.0 * h)
    value = np.asarray(field.B_plasma_xyz(inside))
    error = np.linalg.norm(value + _z_axis_field(inside), axis=1) / scale
    assert error.max() <= tol, error.max()


def test_the_schedule_does_not_stop_at_an_unresolved_level():
    """The coarse level misses the requested digits; the schedule returns the fine one.

    The double-layer self-test that used to choose the level cannot see this:
    ``min(|1 + U|, |U|)`` tests the quadrature of a unit density only, not how
    well the layer densities themselves are resolved.
    """
    digits = 4
    surface = _rotating_ellipse(13, 16, 5)
    field = _field(surface, digits)
    coarse, fine = field.schedule_levels
    points = _offsets(surface, 2.5 * _spacing(field))
    exact = _ring_field(points)
    scale = _scale(surface)

    def level_error(levels):
        value = np.asarray(_field(surface, digits, levels).B_plasma_xyz(points))
        return float(np.max(np.linalg.norm(value - exact, axis=1) / scale))

    assert level_error((coarse,)) > 10.0**-digits
    assert level_error((fine,)) <= 10.0**-digits
    np.testing.assert_allclose(
        np.asarray(field.B_plasma_xyz(points)),
        np.asarray(_field(surface, digits, (fine,)).B_plasma_xyz(points)),
        rtol=1e-12, atol=1e-14)


def test_two_level_schedule_is_branch_free_and_differentiable():
    """With two levels the result is always the finest, so the VJP is exact."""
    surface = _rotating_ellipse(8, 8, 3)
    field = _field(surface, 3)
    points = jnp.asarray(_offsets(surface, 3.0 * _spacing(field), count=4))

    def total(scale):
        return jnp.sum(field.B_plasma_xyz(points * scale))

    value, pullback = jax.vjp(total, 1.0)
    step = 1e-6
    finite = (total(1.0 + step) - total(1.0 - step)) / (2.0 * step)
    np.testing.assert_allclose(float(pullback(1.0)[0]), float(finite), rtol=2e-5)
    np.testing.assert_allclose(float(value), float(total(1.0)), rtol=1e-12)

    tangent = jax.jvp(total, (1.0,), (1.0,))[1]
    np.testing.assert_allclose(float(tangent), float(pullback(1.0)[0]), rtol=1e-10)


def test_unconverged_targets_report_their_achieved_error():
    """A target the schedule cannot resolve reports it instead of staying silent."""
    surface = _rotating_ellipse(12, 12, 3)
    field = _field(surface, 6)
    near = _offsets(surface, 0.25 * _spacing(field))
    value, estimate = field.B_plasma_xyz(near, return_estimate=True)
    assert np.all(np.asarray(estimate) > 1e-6)
    assert np.all(np.isfinite(np.asarray(value)))
    np.testing.assert_allclose(
        np.asarray(field.B_plasma_error_estimate(near)), np.asarray(estimate),
        rtol=1e-12, atol=0.0)


def test_estimate_is_traceable_and_matches_the_eager_call():
    surface = _rotating_ellipse(8, 8, 2)
    field = _field(surface, 3)
    points = jnp.asarray(_offsets(surface, 3.0 * _spacing(field), count=4))
    eager = np.asarray(field.B_plasma_error_estimate(points))
    traced = np.asarray(jax.jit(field.B_plasma_error_estimate)(points))
    np.testing.assert_allclose(traced, eager, rtol=1e-10, atol=1e-14)


def test_three_level_schedule_selects_per_target():
    """A longer schedule stops per target once the estimate meets the tolerance."""
    digits = 3
    surface = _rotating_ellipse(8, 8, 3)
    base = _field(surface, digits).schedule_levels[0]
    levels = (base, (2 * base[0], 2 * base[1]), (4 * base[0], 4 * base[1]))
    field = _field(surface, digits, levels)
    near = _offsets(surface, 0.5 * _spacing(field), count=8)
    far = _offsets(surface, 4.0 * _spacing(field), count=8)

    value, estimate = field.B_plasma_xyz(np.concatenate([near, far]), return_estimate=True)
    assert np.asarray(value).shape == (16, 3)
    # far targets converge, near ones cannot and say so
    assert np.asarray(estimate)[8:].max() <= 10.0**-digits
    assert np.asarray(estimate)[:8].max() > 10.0**-digits
    # a target already converged at the middle level keeps that level's answer
    two_level = _field(surface, digits, levels[:2]).B_plasma_xyz(far)
    np.testing.assert_allclose(
        np.asarray(field.B_plasma_xyz(far)), np.asarray(two_level),
        rtol=1e-9, atol=1e-12)


def test_gradient_schedule_reports_its_own_estimate():
    """The gradB schedule returns an estimate scaled by its own magnitude."""
    digits = 3
    surface = _rotating_ellipse(8, 8, 3)
    field = _field(surface, digits)
    targets = jnp.asarray(_offsets(surface, 3.0 * _spacing(field), count=4)).T
    gradB, estimate = field._vc.compute_internal_gradB_offsurf_schedule(
        field.B_total, X_trg=targets, levels=field.schedule_levels,
        digits=digits, return_estimate=True)
    assert np.asarray(gradB).shape == (3, 3, 4)
    assert np.asarray(estimate).shape == (4,)
    assert np.all(np.isfinite(np.asarray(estimate)))
    plain = field._vc.compute_internal_gradB_offsurf_schedule(
        field.B_total, X_trg=targets, levels=field.schedule_levels, digits=digits)
    np.testing.assert_allclose(np.asarray(gradB), np.asarray(plain), rtol=1e-12, atol=0.0)


def test_return_estimate_needs_the_jitted_schedule():
    surface = _rotating_ellipse(8, 8, 3)
    field = VirtualCasingExteriorField(
        surface, ExteriorFieldConfig(digits=3, use_jit_schedule=False,
                                     src_nphi=8, src_ntheta=8))
    with pytest.raises(ValueError, match="use_jit_schedule"):
        field.B_plasma_error_estimate(_offsets(surface, 0.2, count=2))
