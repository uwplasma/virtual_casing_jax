"""Non-stellarator-symmetric (lasym) virtual casing against an exact MHD equilibrium.

Integer family of Landreman (arXiv:2609.26742) with the symmetry-breaking tau of
Issan et al. (arXiv:2610.07304), mu0 = 1, c = 1, nfp = 2. tau -> -tau is the mirror
image under the rotation M = diag(1, -1, -1): B_{-tau}(M x) = -M B_tau(x). The field
is defined everywhere by one formula, so the plasma field on and outside the boundary
is the Biot-Savart integral of J = curl B over the plasma volume.
"""
import functools

import jax
import jax.numpy as jnp
import pytest
from numpy.polynomial.legendre import leggauss

from virtual_casing_jax import VirtualCasingJAX

jax.config.update("jax_enable_x64", True)

A, B, EDGE, NFP = 1.5**0.5, 0.5**0.5, 1 / 64, 2
M = jnp.diag(jnp.array([1.0, -1.0, -1.0]))


def field(x, tau):
    X, Y, z = x[..., 0], x[..., 1], x[..., 2]
    q = (X / A) ** 2 + (Y / B) ** 2
    f = jnp.sqrt(2 * q - q * q - 4 * z * z - 4 * tau * q * z)
    return jnp.stack(
        (((2 * z + tau * q) * X - A / B * f * Y) / q, ((2 * z + tau * q) * Y + B / A * f * X) / q, 1 - q - 2 * tau * z), -1
    )


def surface(theta, phi, tau, label=EDGE):
    """Flux surface psi = label in physical phi and a poloidal angle theta."""
    w = jnp.sqrt(1 - tau**2)
    beta = 2 * phi + theta
    u, v = -w * w * (A * A - B * B) / 4 + jnp.sqrt(label) * jnp.cos(beta), jnp.sqrt(label) * jnp.sin(beta)
    ell = jnp.sqrt((1 + jnp.sqrt(1 - 4 * (u * u + v * v))) / 2)
    a, c, d, e = A * (ell + u / ell), A * v / ell, B * v / ell, B * (ell - u / ell)
    ct, st = e * jnp.cos(phi) - c * jnp.sin(phi), a * jnp.sin(phi) - d * jnp.cos(phi)
    ct, st = ct / jnp.hypot(ct, st), st / jnp.hypot(ct, st)
    c2, s2 = ct * ct - st * st, 2 * st * ct
    c2, s2 = w * c2 - tau * s2, w * s2 + tau * c2
    return jnp.stack(((a * ct + c * st) / w, (d * ct + e * st) / w, (v * c2 - u * s2 - tau / 2) / (w * w)), -1)


@functools.lru_cache
def _setup(tau, half_period=False, nt=16, npol=32):
    phi = (jnp.arange(nt) + 0.5 * half_period) * 2 * jnp.pi / (NFP * nt * (1 + half_period))
    theta = jnp.arange(npol) * 2 * jnp.pi / npol
    X = surface(theta[None, :], phi[:, None], tau)
    vc = VirtualCasingJAX()
    vc.setup(6, NFP, half_period, nt, npol, jnp.moveaxis(X, -1, 0), nt, npol, nt, npol)
    return vc, jnp.moveaxis(field(X, tau), -1, 0).reshape(3, -1)


def _b_plasma_vc(tau, x_trg, half_period=False):
    vc, B0 = _setup(tau, half_period)
    return vc.compute_internal_B_offsurf(B0, X_trg=x_trg.T, digits=6).T


def _plasma_field_biot_savart(x_trg, tau, nr=12, nt=48, nphi=96):
    """(1/4 pi) int_V J x (x - x') / |x - x'|^3 dV' in (sqrt(psi), theta, phi)."""
    r, wr = leggauss(nr)
    r, wr = (r + 1) * EDGE**0.5 / 2, wr * EDGE**0.5 / 2
    grid = jnp.meshgrid(r, jnp.arange(nt) * 2 * jnp.pi / nt, jnp.arange(nphi) * 2 * jnp.pi / nphi, indexing="ij")
    pts = jnp.stack(grid, -1).reshape(-1, 3)
    chart = lambda p: surface(p[1], p[2], tau, p[0] ** 2)
    x = jax.vmap(chart)(pts)
    jac = jnp.abs(jnp.linalg.det(jax.vmap(jax.jacfwd(chart))(pts)))
    dB = jax.vmap(jax.jacfwd(lambda y: field(y, tau)))(x)
    J = jnp.stack((dB[:, 2, 1] - dB[:, 1, 2], dB[:, 0, 2] - dB[:, 2, 0], dB[:, 1, 0] - dB[:, 0, 1]), -1)
    w = (jac * jnp.repeat(jnp.asarray(wr), nt * nphi) * (2 * jnp.pi) ** 2 / (nt * nphi))[:, None]
    d = x_trg[:, None, :] - x[None]
    return jnp.sum(jnp.cross(w * J, d) / jnp.sum(d * d, -1)[..., None] ** 1.5, 1) / (4 * jnp.pi)


# Off-surface targets in the hole and outside the plasma; all are >= 0.2 from the boundary.
ANGLE = jnp.linspace(0, 2 * jnp.pi, 5, endpoint=False)
X_TRG = jnp.concatenate([jnp.stack((R * jnp.cos(ANGLE), R * jnp.sin(ANGLE), jnp.full(5, -0.3)), -1) for R in (0.4, 1.8)])


@pytest.mark.parametrize("tau", [0.5, -0.5, 0.0])
def test_lasym_plasma_field_matches_exact_volume_current(tau):
    # Spectral in the source grid: 1e-4 (12x24), <1e-5 (16x32), <3e-8 (24x48).
    B_ref = _plasma_field_biot_savart(X_TRG, tau)
    assert jnp.max(jnp.abs(_b_plasma_vc(tau, X_TRG) - B_ref)) < 2e-5 * jnp.max(jnp.abs(B_ref))


def test_mirror_image_boundaries_give_mirror_image_plasma_field():
    B_p, B_m = _b_plasma_vc(0.5, X_TRG), _b_plasma_vc(-0.5, X_TRG @ M)
    assert jnp.max(jnp.abs(B_m @ M + B_p)) < 1e-13 * jnp.max(jnp.abs(B_p))


def test_tau_zero_full_period_matches_stellarator_symmetric_half_period():
    full, half = _b_plasma_vc(0.0, X_TRG), _b_plasma_vc(0.0, X_TRG, half_period=True)
    assert jnp.max(jnp.abs(full - half)) < 1e-5 * jnp.max(jnp.abs(full))
