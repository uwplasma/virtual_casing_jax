"""Scalar-derived densities: valid up to half the Nyquist band, flagged above it.

A boundary field ``s n`` (pure charge) is not band-limited, since the unit normal has an
infinite geometric tail, so the samples alias once ``s`` has modes near the Nyquist limit.  The
contract is that densities derived from a scalar with ``m, k <= Nyquist / 2`` are accurate and that
the returned estimate flags the rest.  The oracle integrates the layer sums of the closed-form
density on a grid far finer than the one the field samples.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_near_surface import NFP, NPHI, NTHETA, _foot
from test_offsurface_level_choice import A_MINOR, DELTA, R0

from virtual_casing_jax import (
    ExteriorFieldConfig,
    VirtualCasingExteriorField,
    VmecSurfaceFieldData,
)


def _tangents(theta, phi):
    u = theta - NFP * phi
    R = R0 + A_MINOR * jnp.cos(theta) + DELTA * jnp.cos(u)
    Rt, Zt = -A_MINOR * jnp.sin(theta) - DELTA * jnp.sin(u), A_MINOR * jnp.cos(theta) + DELTA * jnp.cos(u)
    Rp, Zp = NFP * DELTA * jnp.sin(u), -NFP * DELTA * jnp.cos(u)
    c, s = jnp.cos(phi), jnp.sin(phi)
    point = jnp.stack([R * c, R * s, A_MINOR * jnp.sin(theta) + DELTA * jnp.sin(u)], axis=-1)
    return point, jnp.stack([Rt * c, Rt * s, Zt], axis=-1), jnp.stack([Rp * c - R * s, Rp * s + R * c, Zp], axis=-1)


_SIGN = float(jnp.sign(jnp.sum(jnp.cross(*_tangents(0.0, 0.0)[1:]) * (_tangents(0.0, 0.0)[0] - jnp.array([R0, 0.0, 0.0])))))


def _outward(theta, phi):
    """The outward unit normal and the area element along it."""
    _, e_theta, e_phi = _tangents(theta, phi)
    area = _SIGN * jnp.cross(e_theta, e_phi)
    return area / jnp.linalg.norm(area, axis=-1, keepdims=True), area


def _density(m, k, theta, phi):
    """The charge density ``cos(m theta + k nfp phi) n_hat``."""
    return jnp.cos(m * theta + k * NFP * phi + 0.3)[..., None] * _outward(theta, phi)[0]


def _reference(m, k, targets, n_theta=256, n_phi=768):
    """``B = sum grad G x K - sigma grad G`` of the density, trapezoid over the whole torus."""
    theta = 2 * jnp.pi * jnp.arange(n_theta) / n_theta
    weight = (2 * jnp.pi / n_theta) * (2 * jnp.pi / n_phi)

    def row(total, phi):
        point = _tangents(theta, phi)[0]
        area = _outward(theta, phi)[1] * weight
        B = _density(m, k, theta, phi)
        r = targets[:, None, :] - point[None]
        q3 = 1.0 / (4 * jnp.pi * jnp.sum(r * r, axis=-1) ** 1.5)
        current, charge = jnp.cross(area, B), jnp.sum(area * B, axis=-1)
        # grad G = -q3 r
        field = jnp.sum(q3[..., None] * (-jnp.cross(r, current[None]) + charge[None, :, None] * r), axis=1)
        return total + field, None

    return jax.lax.scan(jax.jit(row), jnp.zeros_like(targets), 2 * jnp.pi * jnp.arange(n_phi) / n_phi)[0]


def _case(m, k):
    theta = jnp.linspace(0.0, 2 * jnp.pi, NTHETA, endpoint=False)
    phi = jnp.linspace(0.0, 2 * jnp.pi / NFP, NPHI, endpoint=False)
    ph, th = jnp.meshgrid(phi, theta, indexing="ij")
    point, area = _tangents(th, ph)[0], _outward(th, ph)[1]
    move = lambda a: jnp.moveaxis(a, -1, 0)
    surface = VmecSurfaceFieldData(
        gamma=move(point), B_total=move(_density(m, k, th, ph)), normal=move(area / jnp.linalg.norm(area, axis=-1, keepdims=True)),
        area_vector=move(area), theta=theta, phi=phi, nfp=NFP, stellsym=False, signgs=1, source_convention="synthetic")
    field = VirtualCasingExteriorField(surface, ExteriorFieldConfig(digits=8))
    targets = jnp.asarray(_foot_targets())
    (value, *_), estimate = field.B_and_derivatives_accurate_xyz(targets, order=0)
    reference = _reference(m, k, targets)
    scale = float(jnp.sqrt(jnp.mean(jnp.sum(surface.B_total**2, axis=0))))
    return float(jnp.max(jnp.abs(value - reference))) / scale, float(jnp.max(estimate))


def _foot_targets(distance=0.1):
    """Targets a fraction of a minor radius outside feet between source nodes."""
    k = np.arange(4)
    point, outward = _foot((2 * np.pi / (NFP * NPHI)) * (3 * k + 1 + 0.37), (2 * np.pi / NTHETA) * (4 * k + 2 + 0.37))
    return point + distance * A_MINOR * outward


@pytest.mark.parametrize("m, k", ((3, 1), (NTHETA // 4, NPHI // 8)))
def test_scalar_derived_densities_up_to_half_the_nyquist_band_are_accurate(m, k):
    error, estimate = _case(m, k)
    assert error <= 1e-6, (error, estimate)


@pytest.mark.parametrize("m, k", ((NTHETA // 2 - 2, NPHI // 3), (NTHETA // 2 - 4, NPHI // 3 + 1)))
def test_scalar_derived_densities_near_the_nyquist_band_are_flagged_by_the_estimate(m, k):
    error, estimate = _case(m, k)
    assert estimate > 1e-8 and estimate >= error, (error, estimate)
