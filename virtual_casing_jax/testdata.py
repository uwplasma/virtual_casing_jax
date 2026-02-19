"""Test data utilities mirroring virtual-casing VirtualCasingTestData."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import functools
import importlib.resources as resources
from typing import Iterable, List, Sequence, Tuple

import jax.numpy as jnp
import numpy as np

from .integrals import biotsavart_fx_u_eval, biotsavart_fxd_u_eval
from .surface_ops import complete_vec_field, rotate_toroidal, resample, upsample, grad2d, surf_normal_area_elem
from .w7x_coeffs import RBC as _W7X_RBC, ZBS as _W7X_ZBS, W7X_NFP

TWOPI = 2.0 * jnp.pi


class SurfType(Enum):
    AxisymCircleWide = 0
    AxisymCircleNarrow = 1
    AxisymWide = 2
    AxisymNarrow = 3
    RotatingEllipseWide = 4
    RotatingEllipseNarrow = 5
    Quas3 = 6
    LHD = 7
    W7X = 8
    Stell = 9
    W7X_ = 10
    NoneType = 11


@dataclass
class Drand48:
    """Deterministic drand48-compatible generator."""

    state: int = 0x1234ABCD330E

    def rand(self) -> float:
        self.state = (0x5DEECE66D * self.state + 0xB) & ((1 << 48) - 1)
        return self.state / float(1 << 48)


def _reshape_soa(X, nt: int | None = None, npol: int | None = None):
    X = jnp.asarray(X)
    if X.ndim == 3:
        return X
    if X.ndim == 2 and X.shape[0] == 3:
        if nt is not None and npol is not None and X.shape[1] == nt * npol:
            return X.reshape(3, nt, npol)
        return X
    if X.ndim == 1:
        if nt is None or npol is None:
            return X.reshape(3, -1)
        return X.reshape(3, nt, npol)
    raise ValueError("Expected SoA layout with shape (3, nt, np) or (3, n)")


def _restore_shape(Y, ref_shape):
    if len(ref_shape) == 3:
        return Y.reshape((3,) + ref_shape[1:])
    return Y


def _stell_geom(nt: int, npol: int, R0: float, a: float, b: float, ellipse: bool):
    t = jnp.arange(nt, dtype=jnp.float64)
    p = jnp.arange(npol, dtype=jnp.float64)
    theta = TWOPI * t / float(nt)
    phi = TWOPI * p / float(npol)

    x = jnp.cos(phi)[None, :]
    y = jnp.sin(phi)[None, :]
    if not ellipse:
        x = (x + jnp.exp(jnp.cos(phi))[None, :] - 1.5) / 2.4

    alpha = 1.5 * theta[:, None]
    cos_a = jnp.cos(alpha)
    sin_a = jnp.sin(alpha)

    x1 = cos_a * x + sin_a * y
    y1 = -sin_a * x + cos_a * y
    x1 = x1 * a
    y1 = y1 * b
    x2 = cos_a * x1 - sin_a * y1
    y2 = sin_a * x1 + cos_a * y1

    R = x2 + R0
    cos_t = jnp.cos(theta)[:, None]
    sin_t = jnp.sin(theta)[:, None]

    X = R * cos_t
    Y = R * sin_t
    Z = y2
    return jnp.stack([X, Y, Z], axis=0)


def _fourier_surface(
    nt: int,
    npol: int,
    rcoeff: jnp.ndarray,
    zcoeff: jnp.ndarray,
    nfp_factor: int,
    i_idx: jnp.ndarray | None = None,
    j_idx: jnp.ndarray | None = None,
):
    t = jnp.arange(nt, dtype=jnp.float64)
    p = jnp.arange(npol, dtype=jnp.float64)
    theta = TWOPI * t / float(nt)
    phi = TWOPI * p / float(npol)

    if i_idx is None:
        i = jnp.arange(rcoeff.shape[0], dtype=jnp.float64)
    else:
        i = jnp.asarray(i_idx, dtype=jnp.float64)
    if j_idx is None:
        j = jnp.arange(rcoeff.shape[1], dtype=jnp.float64)
    else:
        j = jnp.asarray(j_idx, dtype=jnp.float64)

    theta4 = theta[:, None, None, None]
    phi4 = phi[None, :, None, None]
    i4 = i[None, None, :, None]
    j4 = j[None, None, None, :]

    phase = j4 * phi4 - float(nfp_factor) * i4 * theta4
    r = jnp.sum(rcoeff[None, None, :, :] * jnp.cos(phase), axis=(-2, -1))
    z = jnp.sum(zcoeff[None, None, :, :] * jnp.sin(phase), axis=(-2, -1))

    cos_t = jnp.cos(theta)[:, None]
    sin_t = jnp.sin(theta)[:, None]
    X = r * cos_t
    Y = r * sin_t
    Z = z
    return jnp.stack([X, Y, Z], axis=0)


@functools.lru_cache(maxsize=None)
def _load_geom_npz(name: str):
    geom_pkg = resources.files("virtual_casing_jax.geom")
    path = geom_pkg.joinpath(f"{name}.npz")
    if not path.exists():
        raise FileNotFoundError(f"Missing bundled geometry asset: {path}")
    with path.open("rb") as f:
        data = np.load(f)
        X = data["X"]
        Y = data["Y"]
        Z = data["Z"]
    return jnp.stack([jnp.asarray(X), jnp.asarray(Y), jnp.asarray(Z)], axis=0)


def _surface_base(nt: int, npol: int, surf_type: SurfType):
    if surf_type == SurfType.AxisymCircleWide:
        return _stell_geom(nt, npol, 2.0, 1.0, 1.0, True)
    if surf_type == SurfType.AxisymCircleNarrow:
        return _stell_geom(nt, npol, 2.0, 0.5, 0.5, True)
    if surf_type == SurfType.AxisymWide:
        return _stell_geom(nt, npol, 2.0, 1.0, 1.0, False)
    if surf_type == SurfType.AxisymNarrow:
        return _stell_geom(nt, npol, 2.0, 0.5, 0.5, False)
    if surf_type == SurfType.RotatingEllipseWide:
        return _stell_geom(nt, npol, 2.0, 0.7, 1.0, True)
    if surf_type == SurfType.RotatingEllipseNarrow:
        return _stell_geom(nt, npol, 2.0, 0.3, 0.55, True)
    if surf_type == SurfType.Stell:
        rcoeff = jnp.array([[10.0, 1.0], [0.0, 0.25]], dtype=jnp.float64)
        zcoeff = jnp.array([[0.0, -1.0], [0.0, 0.25]], dtype=jnp.float64)
        return _fourier_surface(nt, npol, rcoeff, zcoeff, nfp_factor=5)
    if surf_type == SurfType.W7X_:
        rcoeff = jnp.asarray(_W7X_RBC, dtype=jnp.float64)
        zcoeff = jnp.asarray(_W7X_ZBS, dtype=jnp.float64)
        idx = jnp.arange(-10, 11)
        return _fourier_surface(nt, npol, rcoeff, zcoeff, nfp_factor=W7X_NFP, i_idx=idx, j_idx=idx)
    if surf_type == SurfType.Quas3:
        base = _load_geom_npz("Quas3")
        return upsample(base, base.shape[1], base.shape[2], nt, npol)
    if surf_type == SurfType.LHD:
        base = _load_geom_npz("LHD")
        return upsample(base, base.shape[1], base.shape[2], nt, npol)
    if surf_type == SurfType.W7X:
        base = _load_geom_npz("W7X")
        return upsample(base, base.shape[1], base.shape[2], nt, npol)
    if surf_type == SurfType.NoneType:
        return jnp.zeros((3, nt, npol), dtype=jnp.float64)
    raise ValueError(f"Unsupported surface type: {surf_type}")


def surface_coordinates(nfp: int, half_period: bool, nt: int, npol: int, surf_type: SurfType = SurfType.AxisymNarrow):
    nt_full = (2 if half_period else 1) * nt
    X0 = _surface_base(nfp * nt_full, npol, surf_type)
    shift = (jnp.pi / float(nfp * nt * 2)) if half_period else 0.0
    X_rot = rotate_toroidal(X0, nfp * nt_full, npol, shift)
    return X_rot[:, :nt, :]


def _build_surface_resampled(nfp: int, half_period: bool, surf_nt: int, surf_np: int, X):
    X = _reshape_soa(X, surf_nt, surf_np)
    shift = (-jnp.pi / float(nfp * surf_nt * 2)) if half_period else 0.0
    XX = complete_vec_field(X, True, half_period, nfp, surf_nt, surf_np, shift)
    nt_full = nfp * (2 if half_period else 1) * surf_nt
    nt_surf = nfp * (2 if half_period else 1) * (surf_nt + 1)
    X_surf = resample(XX, nt_full, surf_np, nt_surf, surf_np)
    return X_surf


def _build_source_loops(
    nfp: int,
    half_period: bool,
    surf_nt: int,
    surf_np: int,
    X,
    rng: Drand48 | None,
):
    rng = Drand48() if rng is None else rng
    X_surf = _build_surface_resampled(nfp, half_period, surf_nt, surf_np, X)

    Nt = 100
    Np = 100
    S_coord = upsample(X_surf, X_surf.shape[1], surf_np, Nt, Np)
    dX = grad2d(S_coord, Nt, Np)
    normal, _ = surf_normal_area_elem(dX, S_coord)
    S_coord = S_coord + (-2.17) * normal

    coord = jnp.mean(S_coord, axis=2, keepdims=True)

    N = 20000
    X_loop = upsample(coord, Nt, 1, N, 1)
    dX_loop = grad2d(X_loop, N, 1)
    dX_t = dX_loop[0::2]

    source0 = [X_loop.reshape(3, N)]
    density0 = [dX_t.reshape(3, N) * 0.05]

    source1: List[jnp.ndarray] = []
    density1: List[jnp.ndarray] = []

    def cross_norm(a, b):
        c = jnp.array(
            [
                a[1] * b[2] - b[1] * a[2],
                a[2] * b[0] - b[2] * a[0],
                a[0] * b[1] - b[0] * a[1],
            ]
        )
        r = jnp.sqrt(jnp.sum(c * c))
        return c * (1.0 / r)

    for i in range(nfp):
        Nskip = i * N // nfp
        Xc = source0[0][:, Nskip]
        Xn = density0[0][:, Nskip]
        R = jnp.sqrt(jnp.sum(Xc * Xc))

        normal = Xn / jnp.sqrt(jnp.sum(Xn * Xn))
        e0 = jnp.array([rng.rand(), rng.rand(), rng.rand()], dtype=jnp.float64)
        e0 = cross_norm(e0, normal) * R
        e1 = cross_norm(e0, normal) * R

        Nloop = 10000
        t = TWOPI * jnp.arange(Nloop, dtype=jnp.float64) / float(Nloop)
        sin_t = jnp.sin(t)
        cos_t = jnp.cos(t)
        r = Xc[:, None] + e0[:, None] * sin_t[None, :] + e1[:, None] * cos_t[None, :]
        dr = e0[:, None] * cos_t[None, :] - e1[:, None] * sin_t[None, :]

        source1.append(r)
        density1.append(dr)

    return source0, density0, source1, density1


def _eval_biot_savart(X_trg, sources: Sequence[jnp.ndarray], densities: Sequence[jnp.ndarray], chunk_size: int = 1024):
    X_trg = _reshape_soa(X_trg)
    if X_trg.ndim == 3:
        out = jnp.zeros((3, X_trg.shape[1], X_trg.shape[2]), dtype=X_trg.dtype)
    else:
        out = jnp.zeros((3, X_trg.shape[1]), dtype=X_trg.dtype)
    for Xs, Fs in zip(sources, densities):
        Xs_ = _reshape_soa(Xs)
        Fs_ = _reshape_soa(Fs)
        nsrc = Xs_.shape[1] * (Xs_.shape[2] if Xs_.ndim == 3 else 1)
        area = jnp.ones((nsrc,), dtype=X_trg.dtype)
        contrib = biotsavart_fx_u_eval(Xs_, X_trg, Fs_, area, chunk_size=chunk_size)
        if X_trg.ndim == 3:
            contrib = contrib.reshape((3, X_trg.shape[1], X_trg.shape[2]))
        out = out + contrib
    return out


def _eval_biot_savart_grad(X_trg, sources: Sequence[jnp.ndarray], densities: Sequence[jnp.ndarray], chunk_size: int = 512):
    X_trg = _reshape_soa(X_trg)
    if X_trg.ndim == 3:
        out = jnp.zeros((3, 3, X_trg.shape[1], X_trg.shape[2]), dtype=X_trg.dtype)
    else:
        out = jnp.zeros((3, 3, X_trg.shape[1]), dtype=X_trg.dtype)
    for Xs, Fs in zip(sources, densities):
        Xs_ = _reshape_soa(Xs)
        Fs_ = _reshape_soa(Fs)
        nsrc = Xs_.shape[1] * (Xs_.shape[2] if Xs_.ndim == 3 else 1)
        area = jnp.ones((nsrc,), dtype=X_trg.dtype)
        contrib = biotsavart_fxd_u_eval(Xs_, X_trg, Fs_, area, chunk_size=chunk_size)
        if X_trg.ndim == 3:
            contrib = contrib.reshape((3, 3, X_trg.shape[1], X_trg.shape[2]))
        out = out + contrib
    return out


def magnetic_field_data_offsurf(
    nfp: int,
    half_period: bool,
    surf_nt: int,
    surf_np: int,
    X,
    X_trg,
    *,
    rng: Drand48 | None = None,
    chunk_size: int = 1024,
):
    """Generate B field at arbitrary target points from synthetic loops."""
    X_trg_arr = _reshape_soa(X_trg)
    ref_shape = X_trg_arr.shape
    source0, density0, source1, density1 = _build_source_loops(nfp, half_period, surf_nt, surf_np, X, rng)

    Bint = _eval_biot_savart(X_trg_arr, source0, density0, chunk_size=chunk_size)
    Bext = _eval_biot_savart(X_trg_arr, source1, density1, chunk_size=chunk_size)

    if len(ref_shape) == 2:
        return Bext, Bint
    return Bext.reshape(ref_shape), Bint.reshape(ref_shape)


def magnetic_field_data(
    nfp: int,
    half_period: bool,
    surf_nt: int,
    surf_np: int,
    X,
    trg_nt: int,
    trg_np: int,
    *,
    rng: Drand48 | None = None,
):
    """Generate B field data for testing VirtualCasingJAX."""
    X = _reshape_soa(X, surf_nt, surf_np)
    X_surf = _build_surface_resampled(nfp, half_period, surf_nt, surf_np, X)

    trg_nt_full = (2 if half_period else 1) * trg_nt
    shift = (jnp.pi / float(nfp * trg_nt * 2)) if half_period else 0.0
    X_surf_shifted = rotate_toroidal(X_surf, X_surf.shape[1], surf_np, shift)
    X_trg_full = resample(
        X_surf_shifted,
        X_surf.shape[1],
        surf_np,
        nfp * trg_nt_full,
        trg_np,
    )
    X_trg = X_trg_full[:, :trg_nt_full, :]

    Bext_full, Bint_full = magnetic_field_data_offsurf(
        nfp,
        half_period,
        surf_nt,
        surf_np,
        X,
        X_trg,
        rng=rng,
    )

    if half_period:
        Bext = Bext_full[:, :trg_nt, :]
        Bint = Bint_full[:, :trg_nt, :]
    else:
        Bext, Bint = Bext_full, Bint_full
    return Bext, Bint


def magnetic_field_grad_data(
    nfp: int,
    half_period: bool,
    surf_nt: int,
    surf_np: int,
    X,
    trg_nt: int,
    trg_np: int,
    *,
    rng: Drand48 | None = None,
):
    """Generate GradB data for testing VirtualCasingJAX."""
    X = _reshape_soa(X, surf_nt, surf_np)
    X_surf = _build_surface_resampled(nfp, half_period, surf_nt, surf_np, X)

    trg_nt_full = (2 if half_period else 1) * trg_nt
    shift = (jnp.pi / float(nfp * trg_nt * 2)) if half_period else 0.0
    X_surf_shifted = rotate_toroidal(X_surf, X_surf.shape[1], surf_np, shift)
    X_trg_full = resample(
        X_surf_shifted,
        X_surf.shape[1],
        surf_np,
        nfp * trg_nt_full,
        trg_np,
    )
    X_trg = X_trg_full[:, :trg_nt_full, :]

    source0, density0, source1, density1 = _build_source_loops(nfp, half_period, surf_nt, surf_np, X, rng)
    GradBint_full = _eval_biot_savart_grad(X_trg, source0, density0)
    GradBext_full = _eval_biot_savart_grad(X_trg, source1, density1)

    if half_period:
        GradBext = GradBext_full[:, :, :trg_nt, :]
        GradBint = GradBint_full[:, :, :trg_nt, :]
    else:
        GradBext, GradBint = GradBext_full, GradBint_full
    return GradBext, GradBint


class VirtualCasingTestData:
    """JAX mirror of virtual-casing VirtualCasingTestData."""

    @staticmethod
    def surface_coordinates(
        nfp: int,
        half_period: bool,
        nt: int,
        npol: int,
        surf_type: SurfType = SurfType.AxisymNarrow,
    ):
        return surface_coordinates(nfp, half_period, nt, npol, surf_type)

    @staticmethod
    def magnetic_field_data(
        nfp: int,
        half_period: bool,
        nt: int,
        npol: int,
        X,
        trg_nt: int,
        trg_np: int,
    ):
        return magnetic_field_data(nfp, half_period, nt, npol, X, trg_nt, trg_np)

    @staticmethod
    def magnetic_field_grad_data(
        nfp: int,
        half_period: bool,
        nt: int,
        npol: int,
        X,
        trg_nt: int,
        trg_np: int,
    ):
        return magnetic_field_grad_data(nfp, half_period, nt, npol, X, trg_nt, trg_np)

    @staticmethod
    def magnetic_field_data_offsurf(
        nfp: int,
        half_period: bool,
        nt: int,
        npol: int,
        X,
        X_trg,
    ):
        return magnetic_field_data_offsurf(nfp, half_period, nt, npol, X, X_trg)
