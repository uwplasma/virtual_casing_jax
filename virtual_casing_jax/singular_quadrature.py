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
    hedgehog_order: int
    rad_dim_base: int
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
    hedgehog_wts: jnp.ndarray


def _legpoly_and_deriv(x, degree: int):
    if degree == 0:
        return np.ones_like(x), np.zeros_like(x)
    if degree == 1:
        return x.copy(), np.ones_like(x)
    p0 = np.ones_like(x)
    p1 = x.copy()
    dp0 = np.zeros_like(x)
    dp1 = np.ones_like(x)
    for n in range(2, degree + 1):
        scal0 = -(n - 1) / n
        scal1 = (2 * n - 1) / n
        p = scal1 * x * p1 + scal0 * p0
        dp = scal1 * (p1 + x * dp1) + scal0 * dp0
        p0, p1 = p1, p
        dp0, dp1 = dp1, dp
    return p1, dp1


def _legendre_rule_01(order: int):
    """Gauss-Legendre nodes/weights on [0, 1] matching SCTL."""
    x = np.empty(order, dtype=np.float64)
    for i in range(order):
        x[i] = -(
            1
            - 1.0 / (8 * order * order)
            + 1.0 / (8 * order * order * order)
        ) * math.cos(math.pi * (4 * i + 3) / (4 * order + 2))
    for _ in range(5):
        p, dp = _legpoly_and_deriv(x, order)
        dx = p / dp
        x = x - dx
        if np.max(np.abs(dx)) < np.finfo(np.float64).eps:
            break

    nds = 0.5 * (x + 1.0)
    _, dp = _legpoly_and_deriv(x, order)
    wts = 1.0 / ((1.0 - x * x) * (dp * dp))
    return nds, wts


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
def precompute_singular(
    patch_dim0: int,
    rad_dim: int,
    hedgehog_order: int = 1,
    pou_dtype=None,
):
    if pou_dtype is not None:
        pou_dtype = np.dtype(pou_dtype)
    patch_dim = 2 * patch_dim0 + 1
    rad_dim_base = rad_dim
    rad_dim = rad_dim_base * (3 if hedgehog_order > 1 else 1)
    ang_dim = 2 * rad_dim_base
    ngrid = patch_dim * patch_dim
    npolar = rad_dim * ang_dim
    patch_rad = (patch_dim - 1) // 2

    qx, qw = _legendre_rule_01(rad_dim)
    if hedgehog_order > 1:
        qw = qw * (2.0 * qx)
        qx = qx * qx

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

    # Hedgehog weights
    if hedgehog_order > 1:
        interp_nds = np.arange(1, 17, dtype=np.float64)
        wts = np.zeros(hedgehog_order, dtype=np.float64)
        for k in range(hedgehog_order):
            pn = 1.0
            pd = 1.0
            for i in range(hedgehog_order):
                if i != k:
                    pn *= interp_nds[i]
                    pd *= (interp_nds[i] - interp_nds[k])
            wts[k] = pn / pd
    else:
        wts = np.ones(1, dtype=np.float64)

    def _cast(arr):
        if pou_dtype is None:
            return jnp.asarray(arr)
        return jnp.asarray(arr, dtype=pou_dtype)

    return SingularPrecomp(
        patch_dim0=patch_dim0,
        hedgehog_order=hedgehog_order,
        rad_dim_base=rad_dim_base,
        rad_dim=rad_dim,
        ang_dim=ang_dim,
        patch_dim=patch_dim,
        ngrid=ngrid,
        npolar=npolar,
        qx=jnp.asarray(qx),
        qw=jnp.asarray(qw),
        Gpou=_cast(Gpou),
        Ppou=_cast(Ppou),
        I_G2P=jnp.asarray(I_G2P),
        M_G2P=_cast(M_G2P),
        interp_idx=jnp.asarray(interp_idx),
        hedgehog_wts=_cast(wts),
    )


def select_patch_dim(digits: int, cond: float):
    p = int(digits * cond * 1.6)
    for thresh in [64, 60, 56, 52, 48, 44, 40, 36, 32, 28, 24, 20, 16, 12, 8]:
        if p >= thresh:
            return thresh
    return 6
