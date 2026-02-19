"""Kernel functions matching BIEST scaling and conventions."""
from __future__ import annotations

import jax.numpy as jnp

FOUR_PI = 4.0 * jnp.pi


def _safe_rinv(r2, eps=1e-30):
    return jnp.where(r2 > eps, 1.0 / jnp.sqrt(r2), 0.0)


def laplace_fx_u(dx, f):
    """Laplace single-layer potential: f / (4*pi*r).

    dx: (..., 3)
    f: (...,)
    returns (...,)
    """
    r2 = jnp.sum(dx * dx, axis=-1)
    rinv = _safe_rinv(r2)
    return f * rinv / FOUR_PI


def laplace_fxd_u(dx, f):
    """Gradient of Laplace single-layer: -(dx * f) / (4*pi*r^3).

    dx: (..., 3)
    f: (...,)
    returns (..., 3)
    """
    r2 = jnp.sum(dx * dx, axis=-1)
    rinv = _safe_rinv(r2)
    rinv3 = rinv * rinv * rinv
    return -(dx * f[..., None]) * rinv3[..., None] / FOUR_PI


def laplace_fxd2_u(dx, f):
    """Second derivatives of Laplace single-layer.

    Returns a (..., 3, 3) tensor with entries:
    (-delta_ij * r^-3 + 3 r_i r_j r^-5) * f / (4*pi)
    """
    r2 = jnp.sum(dx * dx, axis=-1)
    rinv = _safe_rinv(r2)
    rinv2 = rinv * rinv
    rinv3 = rinv * rinv2
    rinv5 = rinv3 * rinv2

    # Outer product r_i r_j
    r_outer = dx[..., :, None] * dx[..., None, :]
    eye = jnp.eye(3, dtype=dx.dtype)
    u = -eye * rinv3[..., None, None] + 3.0 * r_outer * rinv5[..., None, None]
    return u * f[..., None, None] / FOUR_PI


def laplace_dx_u(dx, n, f):
    """Laplace double-layer kernel: (-(n·dx) * f) / (4*pi*r^3).

    dx: (..., 3)
    n: (..., 3) source normals
    f: (...,)
    returns (...,)
    """
    r2 = jnp.sum(dx * dx, axis=-1)
    rinv = _safe_rinv(r2)
    rinv3 = rinv * rinv * rinv
    ndotr = -jnp.sum(n * dx, axis=-1)
    return f * ndotr * rinv3 / FOUR_PI


def biotsavart_fx_u(dx, fvec):
    """Biot-Savart kernel: (f x dx) / (4*pi*r^3).

    dx: (..., 3)
    fvec: (..., 3)
    returns (..., 3)
    """
    r2 = jnp.sum(dx * dx, axis=-1)
    rinv = _safe_rinv(r2)
    rinv3 = rinv * rinv * rinv
    return jnp.cross(fvec, dx) * rinv3[..., None] / FOUR_PI


def biotsavart_fxd_u(dx, fvec):
    """Derivative of Biot-Savart kernel (matches BIEST FxdU).

    Returns (..., 3, 3) tensor. The explicit formula matches the
    uker_FxdU in BIEST (see kernel.hpp).
    """
    r2 = jnp.sum(dx * dx, axis=-1)
    rinv = _safe_rinv(r2)
    rinv2 = rinv * rinv
    rinv3 = rinv * rinv2
    rinv5 = rinv3 * rinv2

    x = dx[..., 0]
    y = dx[..., 1]
    z = dx[..., 2]

    z0 = jnp.zeros_like(x)
    u00 = z0
    u10 = 3.0 * z * x * rinv5
    u20 = -3.0 * y * x * rinv5

    u01 = z0
    u11 = 3.0 * z * y * rinv5
    u21 = rinv3 - 3.0 * y * y * rinv5

    u02 = z0
    u12 = -rinv3 + 3.0 * z * z * rinv5
    u22 = -3.0 * y * z * rinv5

    u03 = -3.0 * z * x * rinv5
    u13 = z0
    u23 = -rinv3 + 3.0 * x * x * rinv5

    u04 = -3.0 * z * y * rinv5
    u14 = z0
    u24 = 3.0 * x * y * rinv5

    u05 = rinv3 - 3.0 * z * z * rinv5
    u15 = z0
    u25 = 3.0 * x * z * rinv5

    u06 = 3.0 * y * x * rinv5
    u16 = rinv3 - 3.0 * x * x * rinv5
    u26 = z0

    u07 = -rinv3 + 3.0 * y * y * rinv5
    u17 = -3.0 * x * y * rinv5
    u27 = z0

    u08 = 3.0 * y * z * rinv5
    u18 = -3.0 * x * z * rinv5
    u28 = z0

    # u is (3,9) with rows 0,1,2
    u = jnp.stack(
        [
            jnp.stack([u00, u01, u02, u03, u04, u05, u06, u07, u08], axis=-1),
            jnp.stack([u10, u11, u12, u13, u14, u15, u16, u17, u18], axis=-1),
            jnp.stack([u20, u21, u22, u23, u24, u25, u26, u27, u28], axis=-1),
        ],
        axis=-2,
    )

    fx = fvec[..., 0]
    fy = fvec[..., 1]
    fz = fvec[..., 2]

    v0 = -(
        u[..., 0, :] * fx[..., None]
        + u[..., 1, :] * fy[..., None]
        + u[..., 2, :] * fz[..., None]
    )

    # v0 shape (..., 9) -> reshape to (..., 3, 3)
    v = v0.reshape(dx.shape[:-1] + (3, 3))
    return v / FOUR_PI
