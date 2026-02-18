"""Boundary integral evaluation (baseline direct-sum)."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .kernels import laplace_fxd_u


def _flatten_soa(x, name: str):
    x = jnp.asarray(x)
    if x.ndim == 3:
        return x.reshape((x.shape[0], -1))
    if x.ndim == 2:
        return x
    raise ValueError(f"{name} must have shape (dof, nt, np) or (dof, n)")


def field_period_target_coords(X_quad, trg_nt: int, trg_np: int, nfp: int):
    """Select target coordinates used by FieldPeriodBIOp.

    X_quad: (3, quad_nt, quad_np) for the full NFP surface.
    Returns X_trg: (3, trg_nt, trg_np) for the first field period.
    """
    X_quad = jnp.asarray(X_quad)
    quad_nt = X_quad.shape[1]
    quad_np = X_quad.shape[2]
    if quad_nt % (nfp * trg_nt) != 0:
        raise ValueError("quad_nt must be divisible by nfp*trg_nt")
    if quad_np % trg_np != 0:
        raise ValueError("quad_np must be divisible by trg_np")
    skip_nt = quad_nt // (nfp * trg_nt)
    skip_np = quad_np // trg_np
    X_trg_full = X_quad[:, ::skip_nt, ::skip_np]
    return X_trg_full[:, :trg_nt, :]


def laplace_fxd_u_eval(X_src, X_trg, density, area_elem, chunk_size: int = 1024):
    """Evaluate Laplace FxdU (grad single-layer) by direct quadrature.

    X_src: (3, nt, np) or (3, nsrc)
    X_trg: (3, nt, np) or (3, ntrg)
    density: (nt, np) or (nsrc,)
    area_elem: (nt, np) or (nsrc,)
    Returns: (3, ntrg) or (3, nt, np) matching X_trg layout.
    """
    X_src = _flatten_soa(X_src, "X_src")
    X_trg = _flatten_soa(X_trg, "X_trg")
    density = jnp.asarray(density).reshape(-1)
    area_elem = jnp.asarray(area_elem).reshape(-1)

    if X_src.shape[0] != 3 or X_trg.shape[0] != 3:
        raise ValueError("X_src and X_trg must be 3D coordinates in SoA layout")

    nsrc = X_src.shape[1]
    ntrg = X_trg.shape[1]
    if density.shape[0] != nsrc or area_elem.shape[0] != nsrc:
        raise ValueError("density/area_elem must match source grid size")

    weights = density * area_elem
    Xs = jnp.transpose(X_src, (1, 0))  # (nsrc, 3)
    Xt = jnp.transpose(X_trg, (1, 0))  # (ntrg, 3)

    if chunk_size is None or chunk_size <= 0:
        dx = Xt[:, None, :] - Xs[None, :, :]
        contrib = laplace_fxd_u(dx, weights)
        out = jnp.sum(contrib, axis=1)
        return jnp.transpose(out, (1, 0))

    pad = (-nsrc) % chunk_size
    if pad:
        Xs = jnp.pad(Xs, ((0, pad), (0, 0)))
        weights = jnp.pad(weights, (0, pad))
    nsrc_pad = Xs.shape[0]
    n_chunks = nsrc_pad // chunk_size

    X_chunks = Xs.reshape((n_chunks, chunk_size, 3))
    w_chunks = weights.reshape((n_chunks, chunk_size))

    def scan_fn(acc, xs):
        Xc, wc = xs
        dx = Xt[:, None, :] - Xc[None, :, :]
        contrib = laplace_fxd_u(dx, wc)
        acc = acc + jnp.sum(contrib, axis=1)
        return acc, None

    init = jnp.zeros((ntrg, 3), dtype=Xs.dtype)
    out, _ = jax.lax.scan(scan_fn, init, (X_chunks, w_chunks))
    return jnp.transpose(out, (1, 0))


def laplace_fxd_u_eval_vec(X_src, X_trg, density_vec, area_elem, chunk_size: int = 1024):
    """Vector-density wrapper for Laplace FxdU.

    density_vec: (3, nt, np) or (3, nsrc)
    Returns: (3, 3, ntrg) with first index over density component.
    """
    density_vec = _flatten_soa(density_vec, "density_vec")
    return jax.vmap(
        lambda dens: laplace_fxd_u_eval(X_src, X_trg, dens, area_elem, chunk_size=chunk_size),
        in_axes=0,
        out_axes=0,
    )(density_vec)
