"""Functional Virtual Casing API with differentiable geometry inputs."""
from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .utils import autotune_chunk_sizes
from .surface_ops import (
    complete_vec_field,
    resample,
    rotate_toroidal,
    grad2d,
    surf_normal_area_elem,
    dot_prod,
    cross_prod,
)
from .integrals import (
    laplace_fxd_u_eval_singular,
    laplace_fxd_u_eval_vec_singular,
    laplace_fxd2_u_eval_singular,
    laplace_fxd2_u_eval_vec_singular,
    laplace_fxd_u_eval,
    laplace_fxd2_u_eval,
    laplace_fxd2_u_eval_vec,
    biotsavart_fx_u_eval,
    computeB_offsurface_adaptive,
    _offsurface_adapt_grid,
    _build_patch_indices,
    _surface_cond,
    select_patch_dim,
)


@dataclass(frozen=True)
class FunctionalSetup:
    """Static quadrature setup for functional API."""

    nfp: int
    nfp_eff: int
    half_period: bool
    surf_nt: int
    surf_np: int
    src_nt: int
    src_np: int
    trg_nt: int
    trg_np: int
    quad_nt: int
    quad_np: int
    patch_dim0: int
    patch_idx: jnp.ndarray
    orient: float


def _resolve_chunk_sizes(op: str, chunk_size, target_chunk_size, *, nsrc: int, ntrg: int):
    chunk_auto = chunk_size is None or (isinstance(chunk_size, str) and chunk_size.lower() == "auto")
    target_auto = isinstance(target_chunk_size, str) and target_chunk_size.lower() == "auto"

    if chunk_auto:
        src_auto, trg_auto = autotune_chunk_sizes(op, nsrc, ntrg)
        chunk_size = src_auto
        if target_auto:
            target_chunk_size = trg_auto
    else:
        chunk_size = int(chunk_size)
        if target_auto:
            _, trg_auto = autotune_chunk_sizes(op, nsrc, ntrg)
            target_chunk_size = trg_auto

    if target_chunk_size is not None and not isinstance(target_chunk_size, str):
        target_chunk_size = int(target_chunk_size)
    return chunk_size, target_chunk_size


def _resolve_pou_dtype(pou_dtype, value_dtype):
    if pou_dtype is None:
        return None
    if isinstance(pou_dtype, str):
        if pou_dtype.lower() == "auto":
            return jnp.float32 if value_dtype == jnp.float64 else value_dtype
        return jnp.dtype(pou_dtype)
    return jnp.dtype(pou_dtype)


def _resolve_patch_dtype(patch_dtype, value_dtype):
    if patch_dtype is None:
        return None
    if isinstance(patch_dtype, str):
        if patch_dtype.lower() == "auto":
            return jnp.float32 if value_dtype == jnp.float64 else value_dtype
        return jnp.dtype(patch_dtype)
    return jnp.dtype(patch_dtype)


def build_surface_coord(X, nfp: int, half_period: bool, surf_nt: int, surf_np: int, trg_nt: int):
    """Build full-field-period surface coordinates from base grid."""
    X = jnp.asarray(X).reshape((3, surf_nt, surf_np))
    if half_period:
        X0 = complete_vec_field(
            X,
            True,
            half_period,
            nfp,
            surf_nt,
            surf_np,
            -math.pi / (nfp * surf_nt * 2),
        )
        X1 = resample(X0, nfp * 2 * surf_nt, surf_np, nfp * 2 * (surf_nt + 1), surf_np)
        surface_coord = rotate_toroidal(
            X1,
            nfp * 2 * (surf_nt + 1),
            surf_np,
            math.pi / (nfp * trg_nt * 2),
        )
        nfp_eff = nfp * 2
    else:
        surface_coord = complete_vec_field(X, True, half_period, nfp, surf_nt, surf_np, 0.0)
        nfp_eff = nfp
    return surface_coord, int(nfp_eff)


def build_quad_setup(surface_coord, quad_nt: int, quad_np: int, *, orient: float | None = None):
    """Compute quadrature coordinates, derivatives, and normals."""
    surf_nt_full = int(surface_coord.shape[1])
    surf_np_full = int(surface_coord.shape[2])
    quad_coord = resample(surface_coord, surf_nt_full, surf_np_full, quad_nt, quad_np)
    dX = grad2d(quad_coord, quad_nt, quad_np)
    normal, area_elem, orient0 = surf_normal_area_elem(
        dX, quad_coord, return_orientation=True
    )
    if orient is None:
        orient = jax.lax.stop_gradient(orient0)
    else:
        normal = normal * (orient / orient0)
    return quad_coord, dX, normal, area_elem, orient


def build_patch_idx(quad_nt: int, quad_np: int, trg_nt: int, trg_np: int, nfp_eff: int, patch_dim0: int):
    """Build patch indices for singular quadrature."""
    skip_nt = quad_nt // (nfp_eff * trg_nt)
    skip_np = quad_np // trg_np
    t_idx = jnp.arange(trg_nt) * skip_nt
    p_idx = jnp.arange(trg_np) * skip_np
    tt, pp = jnp.meshgrid(t_idx, p_idx, indexing="ij")
    return _build_patch_indices(
        tt.reshape(-1),
        pp.reshape(-1),
        quad_nt,
        quad_np,
        patch_dim0,
    )


def select_patch_dim_from_geom(dX, quad_nt: int, quad_np: int, digits: int):
    """Select patch_dim0 using surface condition (non-differentiable)."""
    cond = _surface_cond(dX, quad_nt, quad_np)
    return select_patch_dim(int(digits), float(cond))


def target_surface_normal(
    X,
    *,
    nfp: int,
    half_period: bool,
    surf_nt: int,
    surf_np: int,
    trg_nt: int,
    trg_np: int,
    orient: float | None = None,
):
    """Return unit normals on the virtual-casing target grid."""
    surface_coord, nfp_eff = build_surface_coord(X, nfp, half_period, surf_nt, surf_np, trg_nt)
    surf_nt_full = int(surface_coord.shape[1])
    surf_np_full = int(surface_coord.shape[2])
    trg_coord = resample(
        surface_coord,
        surf_nt_full,
        surf_np_full,
        nfp_eff * trg_nt,
        trg_np,
    )
    dX = grad2d(trg_coord, nfp_eff * trg_nt, trg_np)
    normal, _, orient0 = surf_normal_area_elem(dX, trg_coord, return_orientation=True)
    if orient is not None:
        normal = normal * (orient / orient0)
    return normal[:, :trg_nt, :]


def prepare_functional_setup(
    X,
    *,
    digits: int,
    nfp: int,
    half_period: bool,
    surf_nt: int,
    surf_np: int,
    src_nt: int,
    src_np: int,
    trg_nt: int,
    trg_np: int,
    quad_nt: int,
    quad_np: int,
    patch_dim0: int | None = None,
    orient: float | None = None,
):
    """Prepare static quadrature setup for functional API.

    This helper is intended to be called outside autodiff; it uses
    non-differentiable logic to choose patch sizes if not provided.
    """
    surface_coord, nfp_eff = build_surface_coord(X, nfp, half_period, surf_nt, surf_np, trg_nt)
    quad_coord, dX, normal, _, orient = build_quad_setup(
        surface_coord, quad_nt, quad_np, orient=orient
    )
    if patch_dim0 is None:
        patch_dim0 = select_patch_dim_from_geom(dX, quad_nt, quad_np, digits)
    patch_idx = build_patch_idx(quad_nt, quad_np, trg_nt, trg_np, nfp_eff, patch_dim0)
    return FunctionalSetup(
        nfp=int(nfp),
        nfp_eff=int(nfp_eff),
        half_period=bool(half_period),
        surf_nt=int(surf_nt),
        surf_np=int(surf_np),
        src_nt=int(src_nt),
        src_np=int(src_np),
        trg_nt=int(trg_nt),
        trg_np=int(trg_np),
        quad_nt=int(quad_nt),
        quad_np=int(quad_np),
        patch_dim0=int(patch_dim0),
        patch_idx=patch_idx,
        orient=float(orient),
    )


def _compute_dtheta(nfp: int, half_period: bool, trg_nt: int, src_nt: int):
    if not half_period:
        return 0.0
    return math.pi * (1.0 / (nfp * trg_nt * 2) - 1.0 / (nfp * src_nt * 2))


def _complete_b0(B0, nfp: int, half_period: bool, src_nt: int, src_np: int, dtheta: float):
    B0 = jnp.asarray(B0).reshape((3, src_nt, src_np))
    return complete_vec_field(B0, False, half_period, nfp, src_nt, src_np, dtheta)


def _compute_B_signed(
    X,
    B0,
    *,
    sign: float,
    digits: int,
    nfp: int,
    half_period: bool,
    surf_nt: int,
    surf_np: int,
    src_nt: int,
    src_np: int,
    trg_nt: int,
    trg_np: int,
    quad_nt: int,
    quad_np: int,
    patch_dim0: int | None = None,
    patch_idx=None,
    orient: float | None = None,
    X_trg=None,
    chunk_size: int | str | None = "auto",
    target_chunk_size: int | str | None = "auto",
    pou_dtype=None,
    patch_dtype=None,
    interp_block_size: int | str | None = "auto",
    remat: bool | None = None,
):
    surface_coord, nfp_eff = build_surface_coord(X, nfp, half_period, surf_nt, surf_np, trg_nt)
    quad_coord, dX, normal, _, orient = build_quad_setup(
        surface_coord, quad_nt, quad_np, orient=orient
    )
    if remat is None:
        remat = False
    value_dtype = jnp.asarray(B0).dtype
    pou_dtype = _resolve_pou_dtype(pou_dtype, value_dtype)
    patch_dtype = _resolve_patch_dtype(patch_dtype, value_dtype)
    nsrc = quad_nt * quad_np
    if X_trg is None:
        ntrg = trg_nt * trg_np
    else:
        X_trg_arr = jnp.asarray(X_trg)
        if X_trg_arr.ndim == 3:
            ntrg = X_trg_arr.shape[1] * X_trg_arr.shape[2]
        elif X_trg_arr.ndim == 2:
            ntrg = X_trg_arr.shape[1]
        else:
            raise ValueError("X_trg must have shape (3, nt, np) or (3, ntrg)")
    chunk_size, target_chunk_size = _resolve_chunk_sizes(
        "b", chunk_size, target_chunk_size, nsrc=nsrc, ntrg=ntrg
    )
    if patch_dim0 is None:
        patch_dim0 = select_patch_dim_from_geom(dX, quad_nt, quad_np, digits)
    if patch_idx is None:
        patch_idx = build_patch_idx(quad_nt, quad_np, trg_nt, trg_np, nfp_eff, patch_dim0)

    dtheta = _compute_dtheta(nfp, half_period, trg_nt, src_nt)
    B0_complete = _complete_b0(B0, nfp, half_period, src_nt, src_np, dtheta)
    B_quad = resample(B0_complete, nfp_eff * src_nt, src_np, quad_nt, quad_np)

    J = cross_prod(normal, B_quad)
    BdotN = dot_prod(B_quad, normal)

    gradG_J = laplace_fxd_u_eval_vec_singular(
        quad_coord,
        dX,
        J,
        trg_nt,
        trg_np,
        nfp_eff,
        X_trg=X_trg,
        digits=digits,
        patch_dim0=patch_dim0,
        chunk_size=chunk_size,
        target_chunk_size=target_chunk_size,
        patch_idx=patch_idx,
        orient=orient,
        pou_dtype=pou_dtype,
        patch_dtype=patch_dtype,
        interp_block_size=interp_block_size,
        remat=remat,
    )
    gradG_J = jnp.asarray(gradG_J).reshape((3, 3, trg_nt, trg_np))

    gradG_BdotN = laplace_fxd_u_eval_singular(
        quad_coord,
        dX,
        BdotN,
        trg_nt,
        trg_np,
        nfp_eff,
        X_trg=X_trg,
        digits=digits,
        patch_dim0=patch_dim0,
        chunk_size=chunk_size,
        target_chunk_size=target_chunk_size,
        patch_idx=patch_idx,
        orient=orient,
        pou_dtype=pou_dtype,
        patch_dtype=patch_dtype,
        interp_block_size=interp_block_size,
        remat=remat,
    )
    gradG_BdotN = jnp.asarray(gradG_BdotN).reshape((3, trg_nt, trg_np))

    B_on_trg = resample(B0_complete, nfp_eff * src_nt, src_np, nfp_eff * trg_nt, trg_np)
    B_on = B_on_trg[:, :trg_nt, :]

    Bvc = jnp.zeros((3, trg_nt, trg_np), dtype=gradG_J.dtype)
    for k in range(3):
        k1 = (k + 1) % 3
        k2 = (k + 2) % 3
        Bvc = Bvc.at[k].set(gradG_J[k1, k2] - gradG_J[k2, k1])

    return sign * (Bvc + gradG_BdotN) + 0.5 * B_on


def compute_external_B_functional(
    X,
    B0,
    *,
    digits: int,
    nfp: int,
    half_period: bool,
    surf_nt: int,
    surf_np: int,
    src_nt: int,
    src_np: int,
    trg_nt: int,
    trg_np: int,
    quad_nt: int,
    quad_np: int,
    patch_dim0: int | None = None,
    patch_idx=None,
    orient: float | None = None,
    X_trg=None,
    chunk_size: int | str | None = "auto",
    target_chunk_size: int | str | None = "auto",
    pou_dtype=None,
    patch_dtype=None,
    interp_block_size: int | str | None = "auto",
    remat: bool | None = None,
):
    """Compute Bext with surface coordinates as differentiable inputs."""
    return _compute_B_signed(
        X,
        B0,
        sign=1.0,
        digits=digits,
        nfp=nfp,
        half_period=half_period,
        surf_nt=surf_nt,
        surf_np=surf_np,
        src_nt=src_nt,
        src_np=src_np,
        trg_nt=trg_nt,
        trg_np=trg_np,
        quad_nt=quad_nt,
        quad_np=quad_np,
        patch_dim0=patch_dim0,
        patch_idx=patch_idx,
        orient=orient,
        X_trg=X_trg,
        chunk_size=chunk_size,
        target_chunk_size=target_chunk_size,
        pou_dtype=pou_dtype,
        patch_dtype=patch_dtype,
        interp_block_size=interp_block_size,
        remat=remat,
    )


def compute_external_B_normal_functional(
    X,
    B0,
    *,
    digits: int,
    nfp: int,
    half_period: bool,
    surf_nt: int,
    surf_np: int,
    src_nt: int,
    src_np: int,
    trg_nt: int,
    trg_np: int,
    quad_nt: int,
    quad_np: int,
    patch_dim0: int | None = None,
    patch_idx=None,
    orient: float | None = None,
    chunk_size: int | str | None = "auto",
    target_chunk_size: int | str | None = "auto",
    pou_dtype=None,
    patch_dtype=None,
    interp_block_size: int | str | None = "auto",
    remat: bool | None = None,
):
    """Compute on-surface Bext dot n with differentiable geometry inputs."""
    Bext = compute_external_B_functional(
        X,
        B0,
        digits=digits,
        nfp=nfp,
        half_period=half_period,
        surf_nt=surf_nt,
        surf_np=surf_np,
        src_nt=src_nt,
        src_np=src_np,
        trg_nt=trg_nt,
        trg_np=trg_np,
        quad_nt=quad_nt,
        quad_np=quad_np,
        patch_dim0=patch_dim0,
        patch_idx=patch_idx,
        orient=orient,
        chunk_size=chunk_size,
        target_chunk_size=target_chunk_size,
        pou_dtype=pou_dtype,
        patch_dtype=patch_dtype,
        interp_block_size=interp_block_size,
        remat=remat,
    )
    normal = target_surface_normal(
        X,
        nfp=nfp,
        half_period=half_period,
        surf_nt=surf_nt,
        surf_np=surf_np,
        trg_nt=trg_nt,
        trg_np=trg_np,
        orient=orient,
    )
    return dot_prod(Bext, normal)


def compute_internal_B_functional(
    X,
    B0,
    *,
    digits: int,
    nfp: int,
    half_period: bool,
    surf_nt: int,
    surf_np: int,
    src_nt: int,
    src_np: int,
    trg_nt: int,
    trg_np: int,
    quad_nt: int,
    quad_np: int,
    patch_dim0: int | None = None,
    patch_idx=None,
    orient: float | None = None,
    X_trg=None,
    chunk_size: int | str | None = "auto",
    target_chunk_size: int | str | None = "auto",
    pou_dtype=None,
    patch_dtype=None,
    interp_block_size: int | str | None = "auto",
    remat: bool | None = None,
):
    """Compute Bint with surface coordinates as differentiable inputs."""
    return _compute_B_signed(
        X,
        B0,
        sign=-1.0,
        digits=digits,
        nfp=nfp,
        half_period=half_period,
        surf_nt=surf_nt,
        surf_np=surf_np,
        src_nt=src_nt,
        src_np=src_np,
        trg_nt=trg_nt,
        trg_np=trg_np,
        quad_nt=quad_nt,
        quad_np=quad_np,
        patch_dim0=patch_dim0,
        patch_idx=patch_idx,
        orient=orient,
        X_trg=X_trg,
        chunk_size=chunk_size,
        target_chunk_size=target_chunk_size,
        pou_dtype=pou_dtype,
        patch_dtype=patch_dtype,
        interp_block_size=interp_block_size,
        remat=remat,
    )


def _compute_gradB_signed(
    X,
    B0,
    *,
    sign: float,
    digits: int,
    nfp: int,
    half_period: bool,
    surf_nt: int,
    surf_np: int,
    src_nt: int,
    src_np: int,
    trg_nt: int,
    trg_np: int,
    quad_nt: int,
    quad_np: int,
    patch_dim0: int | None = None,
    patch_idx=None,
    orient: float | None = None,
    hedgehog_order: int = 8,
    chunk_size: int | str | None = "auto",
    target_chunk_size: int | str | None = "auto",
    pou_dtype=None,
    patch_dtype=None,
    interp_block_size: int | str | None = "auto",
    remat: bool | None = None,
):
    surface_coord, nfp_eff = build_surface_coord(X, nfp, half_period, surf_nt, surf_np, trg_nt)
    quad_coord, dX, normal, _, orient = build_quad_setup(
        surface_coord, quad_nt, quad_np, orient=orient
    )
    if remat is None:
        remat = True
    value_dtype = jnp.asarray(B0).dtype
    pou_dtype = _resolve_pou_dtype(pou_dtype, value_dtype)
    patch_dtype = _resolve_patch_dtype(patch_dtype, value_dtype)
    nsrc = quad_nt * quad_np
    ntrg = trg_nt * trg_np
    chunk_size, target_chunk_size = _resolve_chunk_sizes(
        "gradb", chunk_size, target_chunk_size, nsrc=nsrc, ntrg=ntrg
    )
    if patch_dim0 is None:
        patch_dim0 = select_patch_dim_from_geom(dX, quad_nt, quad_np, digits)
    if patch_idx is None:
        patch_idx = build_patch_idx(quad_nt, quad_np, trg_nt, trg_np, nfp_eff, patch_dim0)

    dtheta = _compute_dtheta(nfp, half_period, trg_nt, src_nt)
    B0_complete = _complete_b0(B0, nfp, half_period, src_nt, src_np, dtheta)
    B_quad = resample(B0_complete, nfp_eff * src_nt, src_np, quad_nt, quad_np)

    J = cross_prod(normal, B_quad)
    BdotN = dot_prod(B_quad, normal)

    gradG_J = laplace_fxd2_u_eval_vec_singular(
        quad_coord,
        dX,
        J,
        trg_nt,
        trg_np,
        nfp_eff,
        digits=digits,
        patch_dim0=patch_dim0,
        hedgehog_order=hedgehog_order,
        chunk_size=chunk_size,
        target_chunk_size=target_chunk_size,
        patch_idx=patch_idx,
        orient=orient,
        pou_dtype=pou_dtype,
        patch_dtype=patch_dtype,
        interp_block_size=interp_block_size,
        remat=remat,
    )
    gradG_J = jnp.asarray(gradG_J).reshape((3, 3, 3, trg_nt, trg_np))

    gradgradG_BdotN = laplace_fxd2_u_eval_singular(
        quad_coord,
        dX,
        BdotN,
        trg_nt,
        trg_np,
        nfp_eff,
        digits=digits,
        patch_dim0=patch_dim0,
        hedgehog_order=hedgehog_order,
        chunk_size=chunk_size,
        target_chunk_size=target_chunk_size,
        patch_idx=patch_idx,
        orient=orient,
        pou_dtype=pou_dtype,
        patch_dtype=patch_dtype,
        interp_block_size=interp_block_size,
        remat=remat,
    )
    gradgradG_BdotN = jnp.asarray(gradgradG_BdotN).reshape((3, 3, trg_nt, trg_np))

    gradBvc = jnp.zeros((3, 3, trg_nt, trg_np), dtype=gradG_J.dtype)
    for k in range(3):
        k1 = (k + 1) % 3
        k2 = (k + 2) % 3
        gradBvc = gradBvc.at[k].set(gradG_J[k1, k2] - gradG_J[k2, k1])

    return (gradBvc + gradgradG_BdotN) * sign


def compute_external_gradB_functional(
    X,
    B0,
    *,
    digits: int,
    nfp: int,
    half_period: bool,
    surf_nt: int,
    surf_np: int,
    src_nt: int,
    src_np: int,
    trg_nt: int,
    trg_np: int,
    quad_nt: int,
    quad_np: int,
    patch_dim0: int | None = None,
    patch_idx=None,
    orient: float | None = None,
    hedgehog_order: int = 8,
    chunk_size: int | str | None = "auto",
    target_chunk_size: int | str | None = "auto",
    pou_dtype=None,
    patch_dtype=None,
    interp_block_size: int | str | None = "auto",
    remat: bool | None = None,
):
    """Compute GradBext with surface coordinates as differentiable inputs."""
    return _compute_gradB_signed(
        X,
        B0,
        sign=1.0,
        digits=digits,
        nfp=nfp,
        half_period=half_period,
        surf_nt=surf_nt,
        surf_np=surf_np,
        src_nt=src_nt,
        src_np=src_np,
        trg_nt=trg_nt,
        trg_np=trg_np,
        quad_nt=quad_nt,
        quad_np=quad_np,
        patch_dim0=patch_dim0,
        patch_idx=patch_idx,
        orient=orient,
        hedgehog_order=hedgehog_order,
        chunk_size=chunk_size,
        target_chunk_size=target_chunk_size,
        pou_dtype=pou_dtype,
        patch_dtype=patch_dtype,
        interp_block_size=interp_block_size,
        remat=remat,
    )


def compute_internal_gradB_functional(
    X,
    B0,
    *,
    digits: int,
    nfp: int,
    half_period: bool,
    surf_nt: int,
    surf_np: int,
    src_nt: int,
    src_np: int,
    trg_nt: int,
    trg_np: int,
    quad_nt: int,
    quad_np: int,
    patch_dim0: int | None = None,
    patch_idx=None,
    orient: float | None = None,
    hedgehog_order: int = 8,
    chunk_size: int | str | None = "auto",
    target_chunk_size: int | str | None = "auto",
    pou_dtype=None,
    patch_dtype=None,
    interp_block_size: int | str | None = "auto",
    remat: bool | None = None,
):
    """Compute GradBint with surface coordinates as differentiable inputs."""
    return _compute_gradB_signed(
        X,
        B0,
        sign=-1.0,
        digits=digits,
        nfp=nfp,
        half_period=half_period,
        surf_nt=surf_nt,
        surf_np=surf_np,
        src_nt=src_nt,
        src_np=src_np,
        trg_nt=trg_nt,
        trg_np=trg_np,
        quad_nt=quad_nt,
        quad_np=quad_np,
        patch_dim0=patch_dim0,
        patch_idx=patch_idx,
        orient=orient,
        hedgehog_order=hedgehog_order,
        chunk_size=chunk_size,
        target_chunk_size=target_chunk_size,
        pou_dtype=pou_dtype,
        patch_dtype=patch_dtype,
        interp_block_size=interp_block_size,
        remat=remat,
    )


def compute_external_B_offsurf_functional(
    X,
    B0,
    *,
    X_trg,
    digits: int,
    nfp: int,
    half_period: bool,
    surf_nt: int,
    surf_np: int,
    src_nt: int,
    src_np: int,
    trg_nt: int,
    trg_np: int,
    max_Nt: int = -1,
    max_Np: int = -1,
    chunk_size: int | str | None = "auto",
    target_chunk_size: int | str | None = "auto",
    adaptive: bool = True,
):
    """Compute off-surface Bext with differentiable geometry inputs."""
    surface_coord, nfp_eff = build_surface_coord(X, nfp, half_period, surf_nt, surf_np, trg_nt)
    surf_nt_full = int(surface_coord.shape[1])
    surf_np_full = int(surface_coord.shape[2])
    patch_dim = 13
    base_nt = max(nfp_eff * src_nt, surf_nt_full, patch_dim)
    base_np = max(src_np, surf_np_full, patch_dim)

    X_src = resample(surface_coord, surf_nt_full, surf_np_full, base_nt, base_np)
    dX = grad2d(X_src, base_nt, base_np)
    normal, _ = surf_normal_area_elem(dX, X_src)

    dtheta = _compute_dtheta(nfp, half_period, trg_nt, src_nt)
    B0_complete = _complete_b0(B0, nfp, half_period, src_nt, src_np, dtheta)
    B_quad = resample(B0_complete, nfp_eff * src_nt, src_np, base_nt, base_np)
    J = cross_prod(normal, B_quad)
    BdotN = dot_prod(B_quad, normal)

    X_trg = jnp.asarray(X_trg)
    nsrc = X_src.shape[1] * X_src.shape[2]
    ntrg = X_trg.shape[1] * X_trg.shape[2] if X_trg.ndim == 3 else X_trg.shape[1]
    chunk_size, target_chunk_size = _resolve_chunk_sizes(
        "boff", chunk_size, target_chunk_size, nsrc=nsrc, ntrg=ntrg
    )
    if adaptive:
        out = computeB_offsurface_adaptive(
            X_src,
            BdotN,
            J,
            X_trg,
            digits=digits,
            max_Nt=max_Nt,
            max_Np=max_Np,
            ext=True,
            chunk_size=chunk_size,
            target_chunk_size=target_chunk_size,
        )
    else:
        area_elem = surf_normal_area_elem(dX, X_src)[1]
        gradG = laplace_fxd_u_eval(
            X_src, X_trg, BdotN, area_elem, chunk_size=chunk_size, target_chunk_size=target_chunk_size
        )
        bs = biotsavart_fx_u_eval(
            X_src, X_trg, J, area_elem, chunk_size=chunk_size, target_chunk_size=target_chunk_size
        )
        out = gradG - bs
    return out


def compute_external_gradB_offsurf_functional(
    X,
    B0,
    *,
    X_trg,
    digits: int,
    nfp: int,
    half_period: bool,
    surf_nt: int,
    surf_np: int,
    src_nt: int,
    src_np: int,
    trg_nt: int,
    trg_np: int,
    max_Nt: int = -1,
    max_Np: int = -1,
    chunk_size: int | str | None = "auto",
    target_chunk_size: int | str | None = "auto",
    adaptive: bool = False,
):
    """Compute off-surface GradBext with differentiable geometry inputs."""
    surface_coord, nfp_eff = build_surface_coord(X, nfp, half_period, surf_nt, surf_np, trg_nt)
    surf_nt_full = int(surface_coord.shape[1])
    surf_np_full = int(surface_coord.shape[2])
    patch_dim = 13
    base_nt = max(nfp_eff * src_nt, surf_nt_full, patch_dim)
    base_np = max(src_np, surf_np_full, patch_dim)

    X_src = resample(surface_coord, surf_nt_full, surf_np_full, base_nt, base_np)
    dX = grad2d(X_src, base_nt, base_np)
    normal, area_elem = surf_normal_area_elem(dX, X_src)

    dtheta = _compute_dtheta(nfp, half_period, trg_nt, src_nt)
    B0_complete = _complete_b0(B0, nfp, half_period, src_nt, src_np, dtheta)
    B_quad = resample(B0_complete, nfp_eff * src_nt, src_np, base_nt, base_np)
    J = cross_prod(normal, B_quad)
    BdotN = dot_prod(B_quad, normal)

    X_trg = jnp.asarray(X_trg)
    X_trg_flat = X_trg.reshape((3, -1)) if X_trg.ndim == 3 else X_trg
    nsrc = X_src.shape[1] * X_src.shape[2]
    ntrg = X_trg_flat.shape[1]
    chunk_size, target_chunk_size = _resolve_chunk_sizes(
        "gradb_off", chunk_size, target_chunk_size, nsrc=nsrc, ntrg=ntrg
    )

    if adaptive:
        X_src, BdotN, J, area_elem = _offsurface_adapt_grid(
            X_src,
            BdotN,
            J,
            X_trg_flat,
            digits=digits,
            max_Nt=max_Nt,
            max_Np=max_Np,
            chunk_size=chunk_size,
            target_chunk_size=target_chunk_size,
        )

    gradG_J = laplace_fxd2_u_eval_vec(
        X_src,
        X_trg_flat,
        J,
        area_elem,
        chunk_size=chunk_size,
        target_chunk_size=target_chunk_size,
    )
    gradG_J = jnp.asarray(gradG_J).reshape((3, 3, 3, X_trg_flat.shape[1]))

    gradgradG_BdotN = laplace_fxd2_u_eval(
        X_src,
        X_trg_flat,
        BdotN,
        area_elem,
        chunk_size=chunk_size,
        target_chunk_size=target_chunk_size,
    )
    gradgradG_BdotN = jnp.asarray(gradgradG_BdotN).reshape((3, 3, X_trg_flat.shape[1]))

    gradB = jnp.zeros((3, 3, X_trg_flat.shape[1]), dtype=gradG_J.dtype)
    for k in range(3):
        k1 = (k + 1) % 3
        k2 = (k + 2) % 3
        gradB = gradB.at[k].set(gradG_J[k1, k2] - gradG_J[k2, k1])

    gradB = gradB + gradgradG_BdotN
    if X_trg.ndim == 3:
        return gradB.reshape((3, 3, X_trg.shape[1], X_trg.shape[2]))
    return gradB
