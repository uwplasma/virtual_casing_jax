"""Singular quadrature utilities (POU + polar correction)."""
from __future__ import annotations

from dataclasses import dataclass
import functools
import math
import numpy as np
import jax.numpy as jnp


INTERP_ORDER = 12


@dataclass(frozen=True)
class SingularPrecomp:
    patch_dim0: int
    rad_dim: int
    ang_dim: int
    patch_dim: int
    ngrid: int
    npolar: int
    qx: jnp.ndarray
    qw: jnp.ndarray
    Gpou: jnp.ndarray
    Ppou: jnp.ndarray
    I_G2P: jnp.ndarray
    M_G2P: jnp.ndarray
    interp_idx: jnp.ndarray


def _legendre_rule_01(order: int):
    """Gauss-Legendre nodes/weights on [0, 1]."""
    x, w = np.polynomial.legendre.leggauss(order)
    x = 0.5 * (x + 1.0)
    w = 0.5 * w
    return x, w


def _pou_fn(patch_dim: int):
    if patch_dim > 45:
        power = 10
    elif patch_dim > 20:
        power = 8
    else:
        power = 6

    def pou(r):
        if r < 0:
            return 1.0
        return math.exp(-36.0 * (r ** power))

    return pou


def _lagrange_interp(z0, z1, i0, i1):
    h = 1.0 / (INTERP_ORDER - 1)
    p = 1.0
    z0i = i0 * h
    z1i = i1 * h
    for j0 in range(INTERP_ORDER):
        if j0 != i0:
            y0 = j0 * h
            p *= (z0 - y0) / (z0i - y0)
    for j1 in range(INTERP_ORDER):
        if j1 != i1:
            y1 = j1 * h
            p *= (z1 - y1) / (z1i - y1)
    return p


@functools.lru_cache(maxsize=None)
def precompute_singular(patch_dim0: int, rad_dim: int):
    patch_dim = 2 * patch_dim0 + 1
    ang_dim = 2 * rad_dim
    ngrid = patch_dim * patch_dim
    npolar = rad_dim * ang_dim
    patch_rad = (patch_dim - 1) // 2

    qx, qw = _legendre_rule_01(rad_dim)

    pou = _pou_fn(patch_dim)

    # Gpou on grid
    Gpou = np.zeros(ngrid, dtype=np.float64)
    h = 1.0 / patch_rad
    for i in range(patch_dim):
        for j in range(patch_dim):
            dr0 = (i - patch_rad) * h
            dr1 = (j - patch_rad) * h
            r = math.sqrt(dr0 * dr0 + dr1 * dr1)
            Gpou[i * patch_dim + j] = -pou(r)

    # Ppou on polar grid
    Ppou = np.zeros(npolar, dtype=np.float64)
    dt = 2.0 * math.pi / ang_dim
    for i in range(rad_dim):
        for j in range(ang_dim):
            dr = qw[i] * patch_rad
            rdt = qx[i] * patch_rad * dt
            Ppou[i * ang_dim + j] = pou(qx[i]) * dr * rdt

    # Interpolation map
    I_G2P = np.zeros(npolar, dtype=np.int64)
    M_G2P = np.zeros((npolar, INTERP_ORDER, INTERP_ORDER), dtype=np.float64)
    h_ang = 2.0 * math.pi / ang_dim
    h_int = 1.0 / (INTERP_ORDER - 1)
    for i0 in range(rad_dim):
        for i1 in range(ang_dim):
            x0 = 0.5 + 0.5 * qx[i0] * math.cos(h_ang * i1)
            x1 = 0.5 + 0.5 * qx[i0] * math.sin(h_ang * i1)

            y0 = int(x0 * (patch_dim - 1) - (INTERP_ORDER - 1) / 2)
            y1 = int(x1 * (patch_dim - 1) - (INTERP_ORDER - 1) / 2)
            y0 = max(0, min(y0, patch_dim - INTERP_ORDER))
            y1 = max(0, min(y1, patch_dim - INTERP_ORDER))

            z0 = (x0 * (patch_dim - 1) - y0) * h_int
            z1 = (x1 * (patch_dim - 1) - y1) * h_int

            idx = i0 * ang_dim + i1
            I_G2P[idx] = y0 * patch_dim + y1
            for j0 in range(INTERP_ORDER):
                for j1 in range(INTERP_ORDER):
                    M_G2P[idx, j0, j1] = _lagrange_interp(z0, z1, j0, j1)

    # Precompute interpolation indices
    ii = np.arange(INTERP_ORDER)[:, None]
    jj = np.arange(INTERP_ORDER)[None, :]
    interp_idx = I_G2P[:, None, None] + ii * patch_dim + jj

    return SingularPrecomp(
        patch_dim0=patch_dim0,
        rad_dim=rad_dim,
        ang_dim=ang_dim,
        patch_dim=patch_dim,
        ngrid=ngrid,
        npolar=npolar,
        qx=jnp.asarray(qx),
        qw=jnp.asarray(qw),
        Gpou=jnp.asarray(Gpou),
        Ppou=jnp.asarray(Ppou),
        I_G2P=jnp.asarray(I_G2P),
        M_G2P=jnp.asarray(M_G2P),
        interp_idx=jnp.asarray(interp_idx),
    )


def select_patch_dim(digits: int, cond: float):
    p = int(digits * cond * 1.6)
    for thresh in [64, 60, 56, 52, 48, 44, 40, 36, 32, 28, 24, 20, 16, 12, 8]:
        if p >= thresh:
            return thresh
    return 6
