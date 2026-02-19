"""Boundary integral evaluation (baseline direct-sum)."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .kernels import laplace_fxd_u, biotsavart_fx_u
from .surface_ops import upsample, grad2d, surf_normal_area_elem, normal_orientation
from .singular_quadrature import precompute_singular, select_patch_dim, INTERP_ORDER


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


def biotsavart_fx_u_eval(X_src, X_trg, density_vec, area_elem, chunk_size: int = 1024):
    """Evaluate Biot-Savart FxU by direct quadrature.

    density_vec: (3, nt, np) or (3, nsrc)
    Returns: (3, ntrg) or (3, nt, np) matching X_trg layout.
    """
    X_src = _flatten_soa(X_src, "X_src")
    X_trg = _flatten_soa(X_trg, "X_trg")
    density_vec = _flatten_soa(density_vec, "density_vec")
    area_elem = jnp.asarray(area_elem).reshape(-1)

    if X_src.shape[0] != 3 or X_trg.shape[0] != 3 or density_vec.shape[0] != 3:
        raise ValueError("X_src, X_trg, density_vec must be 3D in SoA layout")

    nsrc = X_src.shape[1]
    ntrg = X_trg.shape[1]
    if area_elem.shape[0] != nsrc:
        raise ValueError("area_elem must match source grid size")

    weights = density_vec * area_elem[None, :]
    Xs = jnp.transpose(X_src, (1, 0))  # (nsrc, 3)
    Xt = jnp.transpose(X_trg, (1, 0))  # (ntrg, 3)

    if chunk_size is None or chunk_size <= 0:
        dx = Xt[:, None, :] - Xs[None, :, :]
        fvec = jnp.transpose(weights, (1, 0))[None, :, :]
        contrib = biotsavart_fx_u(dx, fvec)
        out = jnp.sum(contrib, axis=1)
        return jnp.transpose(out, (1, 0))

    pad = (-nsrc) % chunk_size
    if pad:
        Xs = jnp.pad(Xs, ((0, pad), (0, 0)))
        weights = jnp.pad(weights, ((0, 0), (0, pad)))
    nsrc_pad = Xs.shape[0]
    n_chunks = nsrc_pad // chunk_size

    X_chunks = Xs.reshape((n_chunks, chunk_size, 3))
    W_chunks = weights.reshape((3, n_chunks, chunk_size)).transpose(1, 2, 0)

    def scan_fn(acc, xs):
        Xc, Wc = xs
        dx = Xt[:, None, :] - Xc[None, :, :]
        fvec = Wc[None, :, :]
        contrib = biotsavart_fx_u(dx, fvec)
        acc = acc + jnp.sum(contrib, axis=1)
        return acc, None

    init = jnp.zeros((ntrg, 3), dtype=Xs.dtype)
    out, _ = jax.lax.scan(scan_fn, init, (X_chunks, W_chunks))
    return jnp.transpose(out, (1, 0))


def computeB_offsurface_baseline(
    X_src,
    BdotN,
    J,
    Xt,
    upsample_factor: int = 1,
    chunk_size: int = 1024,
    ext: bool = True,
):
    """Baseline off-surface evaluation using direct quadrature.

    This mirrors ExtVacuumField behavior (no singular correction) with
    optional upsampling for improved accuracy.
    """
    X_src = jnp.asarray(X_src)
    BdotN = jnp.asarray(BdotN)
    J = jnp.asarray(J)

    nt = X_src.shape[1]
    npol = X_src.shape[2]

    if upsample_factor > 1:
        nt1 = nt * upsample_factor
        np1 = npol * upsample_factor
        X_src = upsample(X_src, nt, npol, nt1, np1)
        BdotN = upsample(BdotN[None, ...], nt, npol, nt1, np1)[0]
        J = upsample(J, nt, npol, nt1, np1)

    dX = grad2d(X_src, X_src.shape[1], X_src.shape[2])
    _, area_elem = surf_normal_area_elem(dX, X_src)

    sign = 1.0 if ext else -1.0
    gradG = laplace_fxd_u_eval(X_src, Xt, BdotN * sign, area_elem, chunk_size=chunk_size)
    bs = biotsavart_fx_u_eval(X_src, Xt, J * sign, area_elem, chunk_size=chunk_size)

    # For external fields, B = gradG[BdotN] - BiotSavart[J]
    if ext:
        return gradG - bs
    return gradG + bs


def _surface_cond(dX, nt: int, npol: int):
    dX = jnp.asarray(dX)
    xt = dX[0]
    xp = dX[1]
    yt = dX[2]
    yp = dX[3]
    zt = dX[4]
    zp = dX[5]
    m00 = (xt * xt + yt * yt + zt * zt) / (nt * nt)
    m11 = (xp * xp + yp * yp + zp * zp) / (npol * npol)
    ratio = jnp.sqrt(m00 / m11)
    amin = jnp.min(ratio)
    amax = jnp.max(ratio)
    return jnp.sqrt(amax / amin)


def _build_patch_indices(t_idx, p_idx, nt: int, npol: int, patch_dim0: int):
    patch_dim = 2 * patch_dim0 + 1
    dt = jnp.arange(-patch_dim0, patch_dim0 + 1)
    dp = jnp.arange(-patch_dim0, patch_dim0 + 1)
    tt = (t_idx[:, None, None] + dt[None, :, None]) % nt
    pp = (p_idx[:, None, None] + dp[None, None, :]) % npol
    idx = (tt * npol + pp).reshape((t_idx.shape[0], patch_dim * patch_dim))
    return idx


def _interp_patch(values, precomp):
    # values: (dof, Ngrid)
    dof = values.shape[0]
    idx = precomp.interp_idx.reshape(-1)
    gathered = jnp.take(values, idx, axis=1)
    gathered = gathered.reshape((dof, precomp.npolar, INTERP_ORDER, INTERP_ORDER))
    weights = precomp.M_G2P[None, ...]
    return jnp.sum(gathered * weights, axis=(2, 3))


def laplace_fxd_u_eval_singular(
    X_src,
    dX_src,
    density,
    trg_nt: int,
    trg_np: int,
    nfp: int,
    digits: int = 5,
    patch_dim0: int | None = None,
    rad_dim: int | None = None,
    chunk_size: int = 1024,
):
    """Evaluate Laplace FxdU with singular correction on surface targets."""
    X_src = jnp.asarray(X_src)
    dX_src = jnp.asarray(dX_src)
    density = jnp.asarray(density)

    nt = X_src.shape[1]
    npol = X_src.shape[2]

    X_trg = field_period_target_coords(X_src, trg_nt, trg_np, nfp)
    base = laplace_fxd_u_eval(X_src, X_trg, density, surf_normal_area_elem(dX_src, X_src)[1], chunk_size=chunk_size)

    cond = _surface_cond(dX_src, nt, npol)
    cond_val = float(cond)
    if patch_dim0 is None:
        patch_dim0 = select_patch_dim(digits, cond_val)
    if rad_dim is None:
        rad_dim = int(patch_dim0 * 1.6)

    precomp = precompute_singular(patch_dim0, rad_dim)
    patch_dim = precomp.patch_dim
    ngrid = precomp.ngrid

    skip_nt = nt // (nfp * trg_nt)
    skip_np = npol // trg_np
    t_idx = jnp.arange(trg_nt) * skip_nt
    p_idx = jnp.arange(trg_np) * skip_np
    tt, pp = jnp.meshgrid(t_idx, p_idx, indexing="ij")
    t_flat = tt.reshape(-1)
    p_flat = pp.reshape(-1)

    patch_idx = _build_patch_indices(t_flat, p_flat, nt, npol, patch_dim0)
    X_flat = X_src.reshape((3, -1))
    dX_flat = dX_src.reshape((6, -1))
    dens_flat = density.reshape(-1)

    def gather(values):
        return jax.vmap(lambda idx: values[:, idx])(patch_idx)

    G = gather(X_flat)  # (Ntrg, 3, Ngrid)
    Gg = gather(dX_flat)  # (Ntrg, 6, Ngrid)
    GF = jax.vmap(lambda idx: dens_flat[idx])(patch_idx)  # (Ntrg, Ngrid)

    orient = float(normal_orientation(X_src, surf_normal_area_elem(dX_src, X_src)[0]))
    invNt = 1.0 / nt
    invNp = 1.0 / npol

    def corr_one(Gi, Ggi, GiF, TrgCoord):
        # Gi: (3, Ngrid), Ggi: (6, Ngrid)
        n0 = Ggi[2] * Ggi[5] - Ggi[3] * Ggi[4]
        n1 = Ggi[4] * Ggi[1] - Ggi[5] * Ggi[0]
        n2 = Ggi[0] * Ggi[3] - Ggi[1] * Ggi[2]
        r = jnp.sqrt(n0 * n0 + n1 * n1 + n2 * n2)
        Ga = r * invNt * invNp

        # scale gradients
        Ggs = Ggi.at[0].multiply(invNt)
        Ggs = Ggs.at[2].multiply(invNt)
        Ggs = Ggs.at[4].multiply(invNt)
        Ggs = Ggs.at[1].multiply(invNp)
        Ggs = Ggs.at[3].multiply(invNp)
        Ggs = Ggs.at[5].multiply(invNp)

        # grid kernel
        dx = TrgCoord[None, :] - Gi.T
        MGrid = laplace_fxd_u(dx, jnp.ones((ngrid,)))
        MGrid = MGrid * (Ga * precomp.Gpou)[:, None]

        # polar interpolation
        P = _interp_patch(Gi, precomp)  # (3, Npolar)
        Pg = _interp_patch(Ggs, precomp)  # (6, Npolar)
        n0p = Pg[2] * Pg[5] - Pg[3] * Pg[4]
        n1p = Pg[4] * Pg[1] - Pg[5] * Pg[0]
        n2p = Pg[0] * Pg[3] - Pg[1] * Pg[2]
        rp = jnp.sqrt(n0p * n0p + n1p * n1p + n2p * n2p)

        dxp = TrgCoord[None, :] - P.T
        MPolar = laplace_fxd_u(dxp, jnp.ones((precomp.npolar,)))
        MPolar = MPolar * (rp * precomp.Ppou)[:, None]

        # scatter polar contributions back to grid
        idx = precomp.interp_idx.reshape(-1)
        w = precomp.M_G2P.reshape((precomp.npolar, -1))
        for k in range(3):
            contrib = (MPolar[:, k:k+1] * w).reshape(-1)
            MGrid = MGrid.at[idx, k].add(contrib)

        return jnp.sum(GiF[:, None] * MGrid, axis=0)

    Trg_flat = X_trg.reshape((3, -1)).T
    corr = jax.vmap(corr_one)(G, Gg, GF, Trg_flat)
    corr = corr.T.reshape((3, trg_nt, trg_np))

    base = base.reshape((3, trg_nt, trg_np))
    return base + corr


def laplace_fxd_u_eval_vec_singular(
    X_src,
    dX_src,
    density_vec,
    trg_nt: int,
    trg_np: int,
    nfp: int,
    digits: int = 5,
    patch_dim0: int | None = None,
    rad_dim: int | None = None,
    chunk_size: int = 1024,
):
    density_vec = jnp.asarray(density_vec)
    return jax.vmap(
        lambda dens: laplace_fxd_u_eval_singular(
            X_src,
            dX_src,
            dens,
            trg_nt,
            trg_np,
            nfp,
            digits=digits,
            patch_dim0=patch_dim0,
            rad_dim=rad_dim,
            chunk_size=chunk_size,
        ),
        in_axes=0,
        out_axes=0,
    )(density_vec)
