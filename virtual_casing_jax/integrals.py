"""Boundary integral evaluation (baseline direct-sum)."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .kernels import laplace_fxd_u, laplace_fxd2_u, biotsavart_fx_u, biotsavart_fxd_u, laplace_dx_u
from .surface_ops import upsample, resample, grad2d, surf_normal_area_elem, normal_orientation
from .singular_quadrature import precompute_singular, select_patch_dim, INTERP_ORDER


def _flatten_soa(x, name: str):
    x = jnp.asarray(x)
    if x.ndim == 3:
        return x.reshape((x.shape[0], -1))
    if x.ndim == 2:
        return x
    raise ValueError(f"{name} must have shape (dof, nt, np) or (dof, n)")


def _pad_to_multiple(x, axis: int, chunk: int):
    if chunk is None or chunk <= 0:
        return x, 0
    n = x.shape[axis]
    pad = (-n) % chunk
    if pad:
        pad_width = [(0, 0)] * x.ndim
        pad_width[axis] = (0, pad)
        x = jnp.pad(x, pad_width)
    return x, pad


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


def laplace_fxd_u_eval(
    X_src,
    X_trg,
    density,
    area_elem,
    *,
    chunk_size: int = 1024,
    target_chunk_size: int | None = None,
):
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
        if target_chunk_size is None or target_chunk_size <= 0:
            dx = Xt[:, None, :] - Xs[None, :, :]
            contrib = laplace_fxd_u(dx, weights)
            out = jnp.sum(contrib, axis=1)
            return jnp.transpose(out, (1, 0))
        chunk_size = 0

    # Pad sources to chunk size
    Xs, pad_src = _pad_to_multiple(Xs, 0, chunk_size if chunk_size > 0 else 1)
    if pad_src:
        weights = jnp.pad(weights, (0, pad_src))
    nsrc_pad = Xs.shape[0]
    n_chunks = 1 if chunk_size <= 0 else nsrc_pad // chunk_size
    if chunk_size <= 0:
        X_chunks = Xs.reshape((1, nsrc_pad, 3))
        w_chunks = weights.reshape((1, nsrc_pad))
    else:
        X_chunks = Xs.reshape((n_chunks, chunk_size, 3))
        w_chunks = weights.reshape((n_chunks, chunk_size))

    if target_chunk_size is None or target_chunk_size <= 0:
        def scan_fn(acc, xs):
            Xc, wc = xs
            dx = Xt[:, None, :] - Xc[None, :, :]
            contrib = laplace_fxd_u(dx, wc)
            acc = acc + jnp.sum(contrib, axis=1)
            return acc, None

        init = jnp.zeros((ntrg, 3), dtype=Xs.dtype)
        out, _ = jax.lax.scan(scan_fn, init, (X_chunks, w_chunks))
        return jnp.transpose(out, (1, 0))

    # Target blocking
    Xt, pad_trg = _pad_to_multiple(Xt, 0, target_chunk_size)
    ntrg_pad = Xt.shape[0]
    n_tchunks = ntrg_pad // target_chunk_size
    Xt_chunks = Xt.reshape((n_tchunks, target_chunk_size, 3))

    def eval_chunk(Xt_chunk):
        def scan_fn(acc, xs):
            Xc, wc = xs
            dx = Xt_chunk[:, None, :] - Xc[None, :, :]
            contrib = laplace_fxd_u(dx, wc)
            acc = acc + jnp.sum(contrib, axis=1)
            return acc, None

        init = jnp.zeros((target_chunk_size, 3), dtype=Xs.dtype)
        out, _ = jax.lax.scan(scan_fn, init, (X_chunks, w_chunks))
        return out

    _, outs = jax.lax.scan(lambda c, x: (c, eval_chunk(x)), None, Xt_chunks)
    out = outs.reshape((ntrg_pad, 3))[:ntrg]
    return jnp.transpose(out, (1, 0))


def laplace_fxd2_u_eval(
    X_src,
    X_trg,
    density,
    area_elem,
    *,
    chunk_size: int = 1024,
    target_chunk_size: int | None = None,
):
    """Evaluate Laplace Fxd2U (second derivatives) by direct quadrature."""
    X_src = _flatten_soa(X_src, "X_src")
    X_trg = _flatten_soa(X_trg, "X_trg")
    density = jnp.asarray(density).reshape(-1)
    area_elem = jnp.asarray(area_elem).reshape(-1)

    if X_src.shape[0] != 3 or X_trg.shape[0] != 3:
        raise ValueError("X_src and X_trg must be 3D in SoA layout")

    nsrc = X_src.shape[1]
    ntrg = X_trg.shape[1]
    if density.shape[0] != nsrc or area_elem.shape[0] != nsrc:
        raise ValueError("density/area_elem must match source grid size")

    weights = density * area_elem
    Xs = jnp.transpose(X_src, (1, 0))
    Xt = jnp.transpose(X_trg, (1, 0))

    if chunk_size is None or chunk_size <= 0:
        if target_chunk_size is None or target_chunk_size <= 0:
            dx = Xt[:, None, :] - Xs[None, :, :]
            contrib = laplace_fxd2_u(dx, weights)
            out = jnp.sum(contrib, axis=1)
            out = out.reshape((ntrg, 9))
            return jnp.transpose(out, (1, 0))
        chunk_size = 0

    Xs, pad_src = _pad_to_multiple(Xs, 0, chunk_size if chunk_size > 0 else 1)
    if pad_src:
        weights = jnp.pad(weights, (0, pad_src))
    nsrc_pad = Xs.shape[0]
    n_chunks = 1 if chunk_size <= 0 else nsrc_pad // chunk_size
    if chunk_size <= 0:
        X_chunks = Xs.reshape((1, nsrc_pad, 3))
        W_chunks = weights.reshape((1, nsrc_pad))
    else:
        X_chunks = Xs.reshape((n_chunks, chunk_size, 3))
        W_chunks = weights.reshape((n_chunks, chunk_size))

    if target_chunk_size is None or target_chunk_size <= 0:
        def scan_fn(acc, xs):
            Xc, Wc = xs
            dx = Xt[:, None, :] - Xc[None, :, :]
            contrib = laplace_fxd2_u(dx, Wc)
            acc = acc + jnp.sum(contrib, axis=1)
            return acc, None

        init = jnp.zeros((ntrg, 3, 3), dtype=Xs.dtype)
        out, _ = jax.lax.scan(scan_fn, init, (X_chunks, W_chunks))
        out = out.reshape((ntrg, 9))
        return jnp.transpose(out, (1, 0))

    Xt, pad_trg = _pad_to_multiple(Xt, 0, target_chunk_size)
    ntrg_pad = Xt.shape[0]
    n_tchunks = ntrg_pad // target_chunk_size
    Xt_chunks = Xt.reshape((n_tchunks, target_chunk_size, 3))

    def eval_chunk(Xt_chunk):
        def scan_fn(acc, xs):
            Xc, Wc = xs
            dx = Xt_chunk[:, None, :] - Xc[None, :, :]
            contrib = laplace_fxd2_u(dx, Wc)
            acc = acc + jnp.sum(contrib, axis=1)
            return acc, None

        init = jnp.zeros((target_chunk_size, 3, 3), dtype=Xs.dtype)
        out, _ = jax.lax.scan(scan_fn, init, (X_chunks, W_chunks))
        out = out.reshape((target_chunk_size, 9))
        return out

    _, outs = jax.lax.scan(lambda c, x: (c, eval_chunk(x)), None, Xt_chunks)
    out = outs.reshape((ntrg_pad, 9))[:ntrg]
    return jnp.transpose(out, (1, 0))


def laplace_fxd2_u_eval_vec(
    X_src,
    X_trg,
    density_vec,
    area_elem,
    *,
    chunk_size: int = 1024,
    target_chunk_size: int | None = None,
):
    """Vector-density wrapper for Laplace Fxd2U."""
    density_vec = _flatten_soa(density_vec, "density_vec")
    return jax.vmap(
        lambda dens: laplace_fxd2_u_eval(
            X_src,
            X_trg,
            dens,
            area_elem,
            chunk_size=chunk_size,
            target_chunk_size=target_chunk_size,
        ),
        in_axes=0,
        out_axes=0,
    )(density_vec)


def laplace_fxd_u_eval_vec(
    X_src,
    X_trg,
    density_vec,
    area_elem,
    *,
    chunk_size: int = 1024,
    target_chunk_size: int | None = None,
):
    """Vector-density wrapper for Laplace FxdU.

    density_vec: (3, nt, np) or (3, nsrc)
    Returns: (3, 3, ntrg) with first index over density component.
    """
    density_vec = _flatten_soa(density_vec, "density_vec")
    return jax.vmap(
        lambda dens: laplace_fxd_u_eval(
            X_src,
            X_trg,
            dens,
            area_elem,
            chunk_size=chunk_size,
            target_chunk_size=target_chunk_size,
        ),
        in_axes=0,
        out_axes=0,
    )(density_vec)


def biotsavart_fx_u_eval(
    X_src,
    X_trg,
    density_vec,
    area_elem,
    *,
    chunk_size: int = 1024,
    target_chunk_size: int | None = None,
):
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
        if target_chunk_size is None or target_chunk_size <= 0:
            dx = Xt[:, None, :] - Xs[None, :, :]
            fvec = jnp.transpose(weights, (1, 0))[None, :, :]
            contrib = biotsavart_fx_u(dx, fvec)
            out = jnp.sum(contrib, axis=1)
            return jnp.transpose(out, (1, 0))
        chunk_size = 0

    Xs, pad_src = _pad_to_multiple(Xs, 0, chunk_size if chunk_size > 0 else 1)
    if pad_src:
        weights = jnp.pad(weights, ((0, 0), (0, pad_src)))
    nsrc_pad = Xs.shape[0]
    n_chunks = 1 if chunk_size <= 0 else nsrc_pad // chunk_size
    if chunk_size <= 0:
        X_chunks = Xs.reshape((1, nsrc_pad, 3))
        W_chunks = weights.reshape((3, 1, nsrc_pad)).transpose(1, 2, 0)
    else:
        X_chunks = Xs.reshape((n_chunks, chunk_size, 3))
        W_chunks = weights.reshape((3, n_chunks, chunk_size)).transpose(1, 2, 0)

    if target_chunk_size is None or target_chunk_size <= 0:
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

    Xt, pad_trg = _pad_to_multiple(Xt, 0, target_chunk_size)
    ntrg_pad = Xt.shape[0]
    n_tchunks = ntrg_pad // target_chunk_size
    Xt_chunks = Xt.reshape((n_tchunks, target_chunk_size, 3))

    def eval_chunk(Xt_chunk):
        def scan_fn(acc, xs):
            Xc, Wc = xs
            dx = Xt_chunk[:, None, :] - Xc[None, :, :]
            fvec = Wc[None, :, :]
            contrib = biotsavart_fx_u(dx, fvec)
            acc = acc + jnp.sum(contrib, axis=1)
            return acc, None

        init = jnp.zeros((target_chunk_size, 3), dtype=Xs.dtype)
        out, _ = jax.lax.scan(scan_fn, init, (X_chunks, W_chunks))
        return out

    _, outs = jax.lax.scan(lambda c, x: (c, eval_chunk(x)), None, Xt_chunks)
    out = outs.reshape((ntrg_pad, 3))[:ntrg]
    return jnp.transpose(out, (1, 0))


def biotsavart_fxd_u_eval(
    X_src,
    X_trg,
    density_vec,
    area_elem,
    *,
    chunk_size: int = 512,
    target_chunk_size: int | None = None,
):
    """Evaluate Biot-Savart FxdU by direct quadrature.

    density_vec: (3, nt, np) or (3, nsrc)
    Returns: (3, 3, ntrg) or (3, 3, nt, np) matching X_trg layout.
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
        if target_chunk_size is None or target_chunk_size <= 0:
            dx = Xt[:, None, :] - Xs[None, :, :]
            fvec = jnp.transpose(weights, (1, 0))[None, :, :]
            contrib = biotsavart_fxd_u(dx, fvec)
            out = jnp.sum(contrib, axis=1)
            return jnp.transpose(out, (1, 2, 0))
        chunk_size = 0

    Xs, pad_src = _pad_to_multiple(Xs, 0, chunk_size if chunk_size > 0 else 1)
    if pad_src:
        weights = jnp.pad(weights, ((0, 0), (0, pad_src)))
    nsrc_pad = Xs.shape[0]
    n_chunks = 1 if chunk_size <= 0 else nsrc_pad // chunk_size
    if chunk_size <= 0:
        X_chunks = Xs.reshape((1, nsrc_pad, 3))
        W_chunks = weights.reshape((3, 1, nsrc_pad)).transpose(1, 2, 0)
    else:
        X_chunks = Xs.reshape((n_chunks, chunk_size, 3))
        W_chunks = weights.reshape((3, n_chunks, chunk_size)).transpose(1, 2, 0)

    if target_chunk_size is None or target_chunk_size <= 0:
        def scan_fn(acc, xs):
            Xc, Wc = xs
            dx = Xt[:, None, :] - Xc[None, :, :]
            fvec = Wc[None, :, :]
            contrib = biotsavart_fxd_u(dx, fvec)
            acc = acc + jnp.sum(contrib, axis=1)
            return acc, None

        init = jnp.zeros((ntrg, 3, 3), dtype=Xs.dtype)
        out, _ = jax.lax.scan(scan_fn, init, (X_chunks, W_chunks))
        return jnp.transpose(out, (1, 2, 0))

    Xt, pad_trg = _pad_to_multiple(Xt, 0, target_chunk_size)
    ntrg_pad = Xt.shape[0]
    n_tchunks = ntrg_pad // target_chunk_size
    Xt_chunks = Xt.reshape((n_tchunks, target_chunk_size, 3))

    def eval_chunk(Xt_chunk):
        def scan_fn(acc, xs):
            Xc, Wc = xs
            dx = Xt_chunk[:, None, :] - Xc[None, :, :]
            fvec = Wc[None, :, :]
            contrib = biotsavart_fxd_u(dx, fvec)
            acc = acc + jnp.sum(contrib, axis=1)
            return acc, None

        init = jnp.zeros((target_chunk_size, 3, 3), dtype=Xs.dtype)
        out, _ = jax.lax.scan(scan_fn, init, (X_chunks, W_chunks))
        return out

    _, outs = jax.lax.scan(lambda c, x: (c, eval_chunk(x)), None, Xt_chunks)
    out = outs.reshape((ntrg_pad, 3, 3))[:ntrg]
    return jnp.transpose(out, (1, 2, 0))


def laplace_dx_u_eval(
    X_src,
    n_src,
    X_trg,
    density,
    area_elem,
    *,
    chunk_size: int = 1024,
    target_chunk_size: int | None = None,
):
    """Evaluate Laplace DxU (double-layer) by direct quadrature."""
    X_src = _flatten_soa(X_src, "X_src")
    n_src = _flatten_soa(n_src, "n_src")
    X_trg = _flatten_soa(X_trg, "X_trg")
    density = jnp.asarray(density).reshape(-1)
    area_elem = jnp.asarray(area_elem).reshape(-1)

    if X_src.shape[0] != 3 or X_trg.shape[0] != 3 or n_src.shape[0] != 3:
        raise ValueError("X_src, X_trg, n_src must be 3D in SoA layout")

    nsrc = X_src.shape[1]
    ntrg = X_trg.shape[1]
    if density.shape[0] != nsrc or area_elem.shape[0] != nsrc:
        raise ValueError("density/area_elem must match source grid size")

    weights = density * area_elem
    Xs = jnp.transpose(X_src, (1, 0))
    Ns = jnp.transpose(n_src, (1, 0))
    Xt = jnp.transpose(X_trg, (1, 0))

    if chunk_size is None or chunk_size <= 0:
        if target_chunk_size is None or target_chunk_size <= 0:
            dx = Xt[:, None, :] - Xs[None, :, :]
            n = Ns[None, :, :]
            contrib = laplace_dx_u(dx, n, weights)
            out = jnp.sum(contrib, axis=1)
            return out.reshape((1, ntrg))
        chunk_size = 0

    Xs, pad_src = _pad_to_multiple(Xs, 0, chunk_size if chunk_size > 0 else 1)
    if pad_src:
        Ns = jnp.pad(Ns, ((0, pad_src), (0, 0)))
        weights = jnp.pad(weights, (0, pad_src))
    nsrc_pad = Xs.shape[0]
    n_chunks = 1 if chunk_size <= 0 else nsrc_pad // chunk_size
    if chunk_size <= 0:
        X_chunks = Xs.reshape((1, nsrc_pad, 3))
        N_chunks = Ns.reshape((1, nsrc_pad, 3))
        W_chunks = weights.reshape((1, nsrc_pad))
    else:
        X_chunks = Xs.reshape((n_chunks, chunk_size, 3))
        N_chunks = Ns.reshape((n_chunks, chunk_size, 3))
        W_chunks = weights.reshape((n_chunks, chunk_size))

    if target_chunk_size is None or target_chunk_size <= 0:
        def scan_fn(acc, xs):
            Xc, Nc, Wc = xs
            dx = Xt[:, None, :] - Xc[None, :, :]
            n = Nc[None, :, :]
            contrib = laplace_dx_u(dx, n, Wc)
            acc = acc + jnp.sum(contrib, axis=1)
            return acc, None

        init = jnp.zeros((ntrg,), dtype=Xs.dtype)
        out, _ = jax.lax.scan(scan_fn, init, (X_chunks, N_chunks, W_chunks))
        return out.reshape((1, ntrg))

    Xt, pad_trg = _pad_to_multiple(Xt, 0, target_chunk_size)
    ntrg_pad = Xt.shape[0]
    n_tchunks = ntrg_pad // target_chunk_size
    Xt_chunks = Xt.reshape((n_tchunks, target_chunk_size, 3))

    def eval_chunk(Xt_chunk):
        def scan_fn(acc, xs):
            Xc, Nc, Wc = xs
            dx = Xt_chunk[:, None, :] - Xc[None, :, :]
            n = Nc[None, :, :]
            contrib = laplace_dx_u(dx, n, Wc)
            acc = acc + jnp.sum(contrib, axis=1)
            return acc, None

        init = jnp.zeros((target_chunk_size,), dtype=Xs.dtype)
        out, _ = jax.lax.scan(scan_fn, init, (X_chunks, N_chunks, W_chunks))
        return out

    _, outs = jax.lax.scan(lambda c, x: (c, eval_chunk(x)), None, Xt_chunks)
    out = outs.reshape((ntrg_pad,))[:ntrg]
    return out.reshape((1, ntrg))


def computeB_offsurface_baseline(
    X_src,
    BdotN,
    J,
    Xt,
    upsample_factor: int = 1,
    chunk_size: int = 1024,
    target_chunk_size: int | None = None,
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
    gradG = laplace_fxd_u_eval(
        X_src,
        Xt,
        BdotN,
        area_elem,
        chunk_size=chunk_size,
        target_chunk_size=target_chunk_size,
    )
    bs = biotsavart_fx_u_eval(
        X_src,
        Xt,
        J,
        area_elem,
        chunk_size=chunk_size,
        target_chunk_size=target_chunk_size,
    )
    return sign * (gradG - bs)


def computeB_offsurface_adaptive(
    X_src,
    BdotN,
    J,
    Xt,
    digits: int = 5,
    max_Nt: int = -1,
    max_Np: int = -1,
    ext: bool = True,
    chunk_size: int = 1024,
    target_chunk_size: int | None = None,
):
    """Adaptive off-surface evaluation matching ExtVacuumField logic."""
    X_src, BdotN, J, area_elem = _offsurface_adapt_grid(
        X_src,
        BdotN,
        J,
        Xt,
        digits=digits,
        max_Nt=max_Nt,
        max_Np=max_Np,
        chunk_size=chunk_size,
        target_chunk_size=target_chunk_size,
    )

    sign = 1.0 if ext else -1.0
    gradG = laplace_fxd_u_eval(
        X_src,
        Xt,
        BdotN,
        area_elem,
        chunk_size=chunk_size,
        target_chunk_size=target_chunk_size,
    )
    bs = biotsavart_fx_u_eval(
        X_src,
        Xt,
        J,
        area_elem,
        chunk_size=chunk_size,
        target_chunk_size=target_chunk_size,
    )
    return sign * (gradG - bs)


def computeB_offsurface_adaptive_schedule(
    X_src,
    BdotN,
    J,
    Xt,
    *,
    levels: tuple[tuple[int, int], ...],
    digits: int = 5,
    ext: bool = True,
    chunk_size: int = 1024,
    target_chunk_size: int | None = None,
):
    """JIT-friendly adaptive off-surface evaluation with fixed refinement schedule.

    The refinement schedule is provided as a static tuple of (Nt, Np) pairs.
    Shapes are static per-level, so this function can be JIT-compiled with
    ``levels`` marked static. The method updates the result only while the
    double-layer self-test error exceeds the tolerance.
    """
    X_src = jnp.asarray(X_src)
    BdotN = jnp.asarray(BdotN)
    J = jnp.asarray(J)
    Xt = jnp.asarray(Xt)

    if len(levels) == 0:
        raise ValueError("levels must contain at least one (Nt, Np) pair")

    nt0 = int(X_src.shape[1])
    np0 = int(X_src.shape[2])
    tol = 10.0 ** (-digits)
    sign = 1.0 if ext else -1.0

    def eval_level(nt, npol):
        X_lvl = resample(X_src, nt0, np0, nt, npol)
        BdotN_lvl = resample(BdotN[None, ...], nt0, np0, nt, npol)[0]
        J_lvl = resample(J, nt0, np0, nt, npol)

        dX = grad2d(X_lvl, nt, npol)
        normal, area_elem = surf_normal_area_elem(dX, X_lvl)

        ones = jnp.ones((nt, npol), dtype=X_lvl.dtype)
        U = laplace_dx_u_eval(
            X_lvl,
            normal,
            Xt,
            ones,
            area_elem,
            chunk_size=chunk_size,
            target_chunk_size=target_chunk_size,
        )
        U = jnp.asarray(U).reshape(-1)
        err = jnp.max(jnp.minimum(jnp.abs(1.0 - U), jnp.abs(U)))

        gradG = laplace_fxd_u_eval(
            X_lvl,
            Xt,
            BdotN_lvl,
            area_elem,
            chunk_size=chunk_size,
            target_chunk_size=target_chunk_size,
        )
        bs = biotsavart_fx_u_eval(
            X_lvl,
            Xt,
            J_lvl,
            area_elem,
            chunk_size=chunk_size,
            target_chunk_size=target_chunk_size,
        )
        return sign * (gradG - bs), err

    nt_init, np_init = levels[0]
    B_best, err_best = eval_level(int(nt_init), int(np_init))

    for nt, npol in levels[1:]:
        nt_i = int(nt)
        np_i = int(npol)

        def update(state):
            return eval_level(nt_i, np_i)

        def keep(state):
            return state

        B_best, err_best = jax.lax.cond(
            err_best > tol,
            update,
            keep,
            operand=(B_best, err_best),
        )

    return B_best


def computeGradB_offsurface_adaptive_schedule(
    X_src,
    BdotN,
    J,
    Xt,
    *,
    levels: tuple[tuple[int, int], ...],
    digits: int = 5,
    ext: bool = True,
    chunk_size: int = 1024,
    target_chunk_size: int | None = None,
):
    """JIT-friendly adaptive off-surface GradB evaluation with fixed schedule."""
    X_src = jnp.asarray(X_src)
    BdotN = jnp.asarray(BdotN)
    J = jnp.asarray(J)
    Xt = jnp.asarray(Xt)

    if len(levels) == 0:
        raise ValueError("levels must contain at least one (Nt, Np) pair")

    nt0 = int(X_src.shape[1])
    np0 = int(X_src.shape[2])
    tol = 10.0 ** (-digits)
    sign = 1.0 if ext else -1.0

    def eval_level(nt, npol):
        X_lvl = resample(X_src, nt0, np0, nt, npol)
        BdotN_lvl = resample(BdotN[None, ...], nt0, np0, nt, npol)[0]
        J_lvl = resample(J, nt0, np0, nt, npol)

        dX = grad2d(X_lvl, nt, npol)
        normal, area_elem = surf_normal_area_elem(dX, X_lvl)

        ones = jnp.ones((nt, npol), dtype=X_lvl.dtype)
        U = laplace_dx_u_eval(
            X_lvl,
            normal,
            Xt,
            ones,
            area_elem,
            chunk_size=chunk_size,
            target_chunk_size=target_chunk_size,
        )
        U = jnp.asarray(U).reshape(-1)
        err = jnp.max(jnp.minimum(jnp.abs(1.0 - U), jnp.abs(U)))

        gradG_J = laplace_fxd2_u_eval_vec(
            X_lvl,
            Xt,
            J_lvl,
            area_elem,
            chunk_size=chunk_size,
            target_chunk_size=target_chunk_size,
        )
        gradG_J = jnp.asarray(gradG_J).reshape((3, 3, 3, Xt.shape[1]))

        gradgradG_BdotN = laplace_fxd2_u_eval(
            X_lvl,
            Xt,
            BdotN_lvl,
            area_elem,
            chunk_size=chunk_size,
            target_chunk_size=target_chunk_size,
        )
        gradgradG_BdotN = jnp.asarray(gradgradG_BdotN).reshape((3, 3, Xt.shape[1]))

        gradB = jnp.zeros((3, 3, Xt.shape[1]), dtype=gradG_J.dtype)
        for k in range(3):
            k1 = (k + 1) % 3
            k2 = (k + 2) % 3
            gradB = gradB.at[k].set(gradG_J[k1, k2] - gradG_J[k2, k1])

        gradB = gradB + gradgradG_BdotN
        return gradB * sign, err

    nt_init, np_init = levels[0]
    grad_best, err_best = eval_level(int(nt_init), int(np_init))

    for nt, npol in levels[1:]:
        nt_i = int(nt)
        np_i = int(npol)

        def update(state):
            return eval_level(nt_i, np_i)

        def keep(state):
            return state

        grad_best, err_best = jax.lax.cond(
            err_best > tol,
            update,
            keep,
            operand=(grad_best, err_best),
        )

    return grad_best


def _offsurface_adapt_grid(
    X_src,
    BdotN,
    J,
    Xt,
    *,
    digits: int = 5,
    max_Nt: int = -1,
    max_Np: int = -1,
    chunk_size: int = 1024,
    target_chunk_size: int | None = None,
):
    """Upsample source grid until the double-layer self-test meets tolerance."""
    X_src = jnp.asarray(X_src)
    BdotN = jnp.asarray(BdotN)
    J = jnp.asarray(J)
    Xt = jnp.asarray(Xt)

    nt = X_src.shape[1]
    npol = X_src.shape[2]
    tol = 10.0 ** (-digits)

    while True:
        dX = grad2d(X_src, nt, npol)
        normal, area_elem = surf_normal_area_elem(dX, X_src)

        ones = jnp.ones((nt, npol), dtype=X_src.dtype)
        U = laplace_dx_u_eval(
            X_src,
            normal,
            Xt,
            ones,
            area_elem,
            chunk_size=chunk_size,
            target_chunk_size=target_chunk_size,
        )
        U = jnp.asarray(U).reshape(-1)
        err = jnp.max(jnp.minimum(jnp.abs(1.0 - U), jnp.abs(U)))

        if err <= tol:
            return X_src, BdotN, J, area_elem

        nt2 = nt * 2
        np2 = npol * 2
        if max_Nt > 0:
            nt2 = min(nt2, max_Nt)
        if max_Np > 0:
            np2 = min(np2, max_Np)
        if nt2 == nt and np2 == npol:
            return X_src, BdotN, J, area_elem

        X_src = upsample(X_src, nt, npol, nt2, np2)
        BdotN = upsample(BdotN[None, ...], nt, npol, nt2, np2)[0]
        J = upsample(J, nt, npol, nt2, np2)
        nt, npol = nt2, np2


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
    dt = jnp.arange(-patch_dim0, patch_dim0 + 1, dtype=jnp.int32)
    dp = jnp.arange(-patch_dim0, patch_dim0 + 1, dtype=jnp.int32)
    tt = (t_idx[:, None, None] + dt[None, :, None]) % nt
    pp = (p_idx[:, None, None] + dp[None, None, :]) % npol
    idx = (tt * npol + pp).reshape((t_idx.shape[0], patch_dim * patch_dim)).astype(jnp.int32)
    return idx


def _interp_patch(values, precomp):
    # values: (dof, Ngrid)
    dof = values.shape[0]
    idx = precomp.interp_idx.reshape(-1)
    if values.dtype != precomp.M_G2P.dtype:
        values = values.astype(precomp.M_G2P.dtype)
    gathered = jnp.take(values, idx, axis=1)
    gathered = gathered.reshape((dof, precomp.npolar, INTERP_ORDER, INTERP_ORDER))
    weights = precomp.M_G2P[None, ...]
    return jnp.sum(gathered * weights, axis=(2, 3))


def _interp_patch_blocked(values, precomp, block_size: int):
    # values: (dof, Ngrid)
    dof = values.shape[0]
    if values.dtype != precomp.M_G2P.dtype:
        values = values.astype(precomp.M_G2P.dtype)

    npolar = precomp.npolar
    block_size = int(block_size)
    if block_size <= 0 or block_size >= npolar:
        return _interp_patch(values, precomp)

    idx = precomp.interp_idx.reshape((npolar, INTERP_ORDER, INTERP_ORDER))
    weights = precomp.M_G2P

    pad = (-npolar) % block_size
    if pad:
        idx = jnp.pad(idx, ((0, pad), (0, 0), (0, 0)))
        weights = jnp.pad(weights, ((0, pad), (0, 0), (0, 0)))
    npolar_pad = npolar + pad
    nblocks = npolar_pad // block_size

    idx_blocks = idx.reshape((nblocks, block_size, INTERP_ORDER, INTERP_ORDER))
    w_blocks = weights.reshape((nblocks, block_size, INTERP_ORDER, INTERP_ORDER))

    def scan_fn(carry, xs):
        idx_block, w_block = xs
        gathered = jnp.take(values, idx_block, axis=1)
        block_out = jnp.sum(gathered * w_block[None, ...], axis=(2, 3))
        return carry, block_out

    _, blocks = jax.lax.scan(scan_fn, None, (idx_blocks, w_blocks))
    blocks = jnp.transpose(blocks, (1, 0, 2))
    out = blocks.reshape((dof, npolar_pad))[:, :npolar]
    return out


def _resolve_interp_block_size(interp_block_size, npolar: int, mode: str):
    if interp_block_size is None:
        return None
    if isinstance(interp_block_size, str) and interp_block_size.lower() == "auto":
        if npolar <= 256:
            return None
        if mode == "gradb":
            return 32
        return 64
    return int(interp_block_size)


def laplace_fxd_u_eval_singular(
    X_src,
    dX_src,
    density,
    trg_nt: int,
    trg_np: int,
    nfp: int,
    X_trg=None,
    digits: int = 5,
    patch_dim0: int | None = None,
    rad_dim: int | None = None,
    chunk_size: int = 1024,
    target_chunk_size: int | None = None,
    patch_idx=None,
    orient: float | None = None,
    pou_dtype=None,
    patch_dtype=None,
    interp_block_size: int | str | None = "auto",
    remat: bool = False,
):
    """Evaluate Laplace FxdU with singular correction on surface targets."""
    X_src = jnp.asarray(X_src)
    dX_src = jnp.asarray(dX_src)
    density = jnp.asarray(density)

    nt = X_src.shape[1]
    npol = X_src.shape[2]

    if X_trg is None:
        X_trg = field_period_target_coords(X_src, trg_nt, trg_np, nfp)
    else:
        X_trg = jnp.asarray(X_trg)
    base = laplace_fxd_u_eval(
        X_src,
        X_trg,
        density,
        surf_normal_area_elem(dX_src, X_src)[1],
        chunk_size=chunk_size,
        target_chunk_size=target_chunk_size,
    )

    if patch_dim0 is None:
        cond = _surface_cond(dX_src, nt, npol)
        cond_val = float(cond)
        patch_dim0 = select_patch_dim(digits, cond_val)
    if rad_dim is None:
        rad_dim = int(patch_dim0 * 1.6)

    precomp = precompute_singular(
        patch_dim0,
        rad_dim,
        pou_dtype=pou_dtype,
        patch_dtype=patch_dtype,
    )
    patch_dim = precomp.patch_dim
    ngrid = precomp.ngrid

    skip_nt = nt // (nfp * trg_nt)
    skip_np = npol // trg_np
    t_idx = jnp.arange(trg_nt) * skip_nt
    p_idx = jnp.arange(trg_np) * skip_np
    tt, pp = jnp.meshgrid(t_idx, p_idx, indexing="ij")
    t_flat = tt.reshape(-1)
    p_flat = pp.reshape(-1)

    if patch_idx is None:
        patch_idx = _build_patch_indices(t_flat, p_flat, nt, npol, patch_dim0)
    X_flat = X_src.reshape((3, -1))
    dX_flat = dX_src.reshape((6, -1))
    dens_flat = density.reshape(-1)

    def gather(values, idx):
        return jax.vmap(lambda ii: values[:, ii])(idx)

    if orient is None:
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
        if interp_block_size is None:
            P = _interp_patch(Gi, precomp)  # (3, Npolar)
            Pg = _interp_patch(Ggs, precomp)  # (6, Npolar)
        else:
            P = _interp_patch_blocked(Gi, precomp, interp_block_size)
            Pg = _interp_patch_blocked(Ggs, precomp, interp_block_size)
        if P.dtype != TrgCoord.dtype:
            P = P.astype(TrgCoord.dtype)
        if Pg.dtype != TrgCoord.dtype:
            Pg = Pg.astype(TrgCoord.dtype)
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
    corr_fn = jax.checkpoint(corr_one) if remat else corr_one
    interp_block_size = _resolve_interp_block_size(interp_block_size, precomp.npolar, "b")

    if target_chunk_size is None or target_chunk_size <= 0:
        G = gather(X_flat, patch_idx)  # (Ntrg, 3, Ngrid)
        Gg = gather(dX_flat, patch_idx)  # (Ntrg, 6, Ngrid)
        GF = jax.vmap(lambda idx: dens_flat[idx])(patch_idx)  # (Ntrg, Ngrid)
        corr = jax.vmap(corr_fn)(G, Gg, GF, Trg_flat)
    else:
        ntrg = Trg_flat.shape[0]
        pad = (-ntrg) % target_chunk_size
        if pad:
            patch_idx = jnp.pad(patch_idx, ((0, pad), (0, 0)))
            Trg_flat = jnp.pad(Trg_flat, ((0, pad), (0, 0)))
        ntrg_pad = Trg_flat.shape[0]
        n_chunks = ntrg_pad // target_chunk_size
        patch_chunks = patch_idx.reshape((n_chunks, target_chunk_size, -1))
        trg_chunks = Trg_flat.reshape((n_chunks, target_chunk_size, 3))

        def scan_fn(carry, xs):
            pidx_chunk, trg_chunk = xs
            G = gather(X_flat, pidx_chunk)
            Gg = gather(dX_flat, pidx_chunk)
            GF = jax.vmap(lambda idx: dens_flat[idx])(pidx_chunk)
            corr_chunk = jax.vmap(corr_fn)(G, Gg, GF, trg_chunk)
            return carry, corr_chunk

        _, corr_chunks = jax.lax.scan(scan_fn, None, (patch_chunks, trg_chunks))
        corr = corr_chunks.reshape((ntrg_pad, -1))[:ntrg]

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
    X_trg=None,
    digits: int = 5,
    patch_dim0: int | None = None,
    rad_dim: int | None = None,
    chunk_size: int = 1024,
    target_chunk_size: int | None = None,
    patch_idx=None,
    orient: float | None = None,
    pou_dtype=None,
    patch_dtype=None,
    interp_block_size: int | str | None = "auto",
    remat: bool = False,
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
            X_trg=X_trg,
            digits=digits,
            patch_dim0=patch_dim0,
            rad_dim=rad_dim,
            chunk_size=chunk_size,
            target_chunk_size=target_chunk_size,
            patch_idx=patch_idx,
            orient=orient,
            pou_dtype=pou_dtype,
            patch_dtype=patch_dtype,
            interp_block_size=interp_block_size,
            remat=remat,
        ),
        in_axes=0,
        out_axes=0,
    )(density_vec)


def laplace_dx_u_eval_singular(
    X_src,
    dX_src,
    density,
    trg_nt: int,
    trg_np: int,
    nfp: int,
    X_trg=None,
    digits: int = 5,
    patch_dim0: int | None = None,
    rad_dim: int | None = None,
    chunk_size: int = 1024,
    target_chunk_size: int | None = None,
    patch_idx=None,
    orient: float | None = None,
    pou_dtype=None,
    patch_dtype=None,
    interp_block_size: int | str | None = "auto",
    remat: bool = False,
):
    """Evaluate Laplace DxU with singular correction on surface targets."""
    X_src = jnp.asarray(X_src)
    dX_src = jnp.asarray(dX_src)
    density = jnp.asarray(density)

    nt = X_src.shape[1]
    npol = X_src.shape[2]

    normal, area_elem = surf_normal_area_elem(dX_src, X_src)
    if X_trg is None:
        X_trg = field_period_target_coords(X_src, trg_nt, trg_np, nfp)
    else:
        X_trg = jnp.asarray(X_trg)
    base = laplace_dx_u_eval(
        X_src,
        normal,
        X_trg,
        density,
        area_elem,
        chunk_size=chunk_size,
        target_chunk_size=target_chunk_size,
    )

    if patch_dim0 is None:
        cond = _surface_cond(dX_src, nt, npol)
        cond_val = float(cond)
        patch_dim0 = select_patch_dim(digits, cond_val)
    if rad_dim is None:
        rad_dim = int(patch_dim0 * 1.6)

    precomp = precompute_singular(
        patch_dim0,
        rad_dim,
        pou_dtype=pou_dtype,
        patch_dtype=patch_dtype,
    )
    patch_dim = precomp.patch_dim
    ngrid = precomp.ngrid

    skip_nt = nt // (nfp * trg_nt)
    skip_np = npol // trg_np
    t_idx = jnp.arange(trg_nt) * skip_nt
    p_idx = jnp.arange(trg_np) * skip_np
    tt, pp = jnp.meshgrid(t_idx, p_idx, indexing="ij")
    t_flat = tt.reshape(-1)
    p_flat = pp.reshape(-1)

    if patch_idx is None:
        patch_idx = _build_patch_indices(t_flat, p_flat, nt, npol, patch_dim0)
    X_flat = X_src.reshape((3, -1))
    dX_flat = dX_src.reshape((6, -1))
    dens_flat = density.reshape(-1)

    def gather(values, idx):
        return jax.vmap(lambda ii: values[:, ii])(idx)

    if orient is None:
        orient = float(normal_orientation(X_src, normal))
    invNt = 1.0 / nt
    invNp = 1.0 / npol

    def corr_one(Gi, Ggi, GiF, TrgCoord):
        n0 = Ggi[2] * Ggi[5] - Ggi[3] * Ggi[4]
        n1 = Ggi[4] * Ggi[1] - Ggi[5] * Ggi[0]
        n2 = Ggi[0] * Ggi[3] - Ggi[1] * Ggi[2]
        r = jnp.sqrt(n0 * n0 + n1 * n1 + n2 * n2)
        Ga = r * invNt * invNp
        inv_r = 1.0 / r
        Gn = jnp.stack([n0, n1, n2], axis=0) * inv_r * orient

        # scale gradients
        Ggs = Ggi.at[0].multiply(invNt)
        Ggs = Ggs.at[2].multiply(invNt)
        Ggs = Ggs.at[4].multiply(invNt)
        Ggs = Ggs.at[1].multiply(invNp)
        Ggs = Ggs.at[3].multiply(invNp)
        Ggs = Ggs.at[5].multiply(invNp)

        dx = TrgCoord[None, :] - Gi.T
        MGrid = laplace_dx_u(dx, Gn.T, jnp.ones((ngrid,)))
        MGrid = MGrid * (Ga * precomp.Gpou)
        MGrid = MGrid.reshape((ngrid,))

        if interp_block_size is None:
            P = _interp_patch(Gi, precomp)  # (3, Npolar)
            Pg = _interp_patch(Ggs, precomp)  # (6, Npolar)
        else:
            P = _interp_patch_blocked(Gi, precomp, interp_block_size)
            Pg = _interp_patch_blocked(Ggs, precomp, interp_block_size)
        if P.dtype != TrgCoord.dtype:
            P = P.astype(TrgCoord.dtype)
        if Pg.dtype != TrgCoord.dtype:
            Pg = Pg.astype(TrgCoord.dtype)
        n0p = Pg[2] * Pg[5] - Pg[3] * Pg[4]
        n1p = Pg[4] * Pg[1] - Pg[5] * Pg[0]
        n2p = Pg[0] * Pg[3] - Pg[1] * Pg[2]
        rp = jnp.sqrt(n0p * n0p + n1p * n1p + n2p * n2p)
        inv_rp = 1.0 / rp
        Pn = jnp.stack([n0p, n1p, n2p], axis=0) * inv_rp * orient

        dxp = TrgCoord[None, :] - P.T
        MPolar = laplace_dx_u(dxp, Pn.T, jnp.ones((precomp.npolar,)))
        MPolar = MPolar * (rp * precomp.Ppou)

        idx = precomp.interp_idx.reshape(-1)
        w = precomp.M_G2P.reshape((precomp.npolar, -1))
        contrib = (MPolar[:, None] * w).reshape(-1)
        MGrid = MGrid.at[idx].add(contrib)

        return jnp.sum(GiF * MGrid)

    Trg_flat = X_trg.reshape((3, -1)).T
    corr_fn = jax.checkpoint(corr_one) if remat else corr_one
    interp_block_size = _resolve_interp_block_size(interp_block_size, precomp.npolar, "b")

    if target_chunk_size is None or target_chunk_size <= 0:
        G = gather(X_flat, patch_idx)
        Gg = gather(dX_flat, patch_idx)
        GF = jax.vmap(lambda idx: dens_flat[idx])(patch_idx)
        corr = jax.vmap(corr_fn)(G, Gg, GF, Trg_flat)
    else:
        ntrg = Trg_flat.shape[0]
        pad = (-ntrg) % target_chunk_size
        if pad:
            patch_idx = jnp.pad(patch_idx, ((0, pad), (0, 0)))
            Trg_flat = jnp.pad(Trg_flat, ((0, pad), (0, 0)))
        ntrg_pad = Trg_flat.shape[0]
        n_chunks = ntrg_pad // target_chunk_size
        patch_chunks = patch_idx.reshape((n_chunks, target_chunk_size, -1))
        trg_chunks = Trg_flat.reshape((n_chunks, target_chunk_size, 3))

        def scan_fn(carry, xs):
            pidx_chunk, trg_chunk = xs
            G = gather(X_flat, pidx_chunk)
            Gg = gather(dX_flat, pidx_chunk)
            GF = jax.vmap(lambda idx: dens_flat[idx])(pidx_chunk)
            corr_chunk = jax.vmap(corr_fn)(G, Gg, GF, trg_chunk)
            return carry, corr_chunk

        _, corr_chunks = jax.lax.scan(scan_fn, None, (patch_chunks, trg_chunks))
        corr = corr_chunks.reshape((ntrg_pad,))[:ntrg]

    corr = corr.reshape((1, trg_nt, trg_np))

    base = base.reshape((1, trg_nt, trg_np))
    return base + corr


def laplace_fxd2_u_eval_singular(
    X_src,
    dX_src,
    density,
    trg_nt: int,
    trg_np: int,
    nfp: int,
    X_trg=None,
    digits: int = 5,
    patch_dim0: int | None = None,
    rad_dim: int | None = None,
    hedgehog_order: int = 8,
    chunk_size: int = 1024,
    target_chunk_size: int | None = None,
    patch_idx=None,
    orient: float | None = None,
    pou_dtype=None,
    patch_dtype=None,
    interp_block_size: int | str | None = "auto",
    remat: bool = False,
):
    """Evaluate Laplace Fxd2U with singular correction (Hedgehog)."""
    X_src = jnp.asarray(X_src)
    dX_src = jnp.asarray(dX_src)
    density = jnp.asarray(density)

    nt = X_src.shape[1]
    npol = X_src.shape[2]

    if X_trg is None:
        X_trg = field_period_target_coords(X_src, trg_nt, trg_np, nfp)
    else:
        X_trg = jnp.asarray(X_trg)
    base = laplace_fxd2_u_eval(
        X_src,
        X_trg,
        density,
        surf_normal_area_elem(dX_src, X_src)[1],
        chunk_size=chunk_size,
        target_chunk_size=target_chunk_size,
    )

    if patch_dim0 is None:
        cond = _surface_cond(dX_src, nt, npol)
        cond_val = float(cond)
        patch_dim0 = select_patch_dim(digits, cond_val)
    if rad_dim is None:
        rad_dim = int(patch_dim0 * 1.6)

    precomp = precompute_singular(
        patch_dim0,
        rad_dim,
        hedgehog_order,
        pou_dtype=pou_dtype,
        patch_dtype=patch_dtype,
    )
    patch_dim = precomp.patch_dim
    ngrid = precomp.ngrid

    skip_nt = nt // (nfp * trg_nt)
    skip_np = npol // trg_np
    t_idx = jnp.arange(trg_nt) * skip_nt
    p_idx = jnp.arange(trg_np) * skip_np
    tt, pp = jnp.meshgrid(t_idx, p_idx, indexing="ij")
    t_flat = tt.reshape(-1)
    p_flat = pp.reshape(-1)

    if patch_idx is None:
        patch_idx = _build_patch_indices(t_flat, p_flat, nt, npol, patch_dim0)
    X_flat = X_src.reshape((3, -1))
    dX_flat = dX_src.reshape((6, -1))
    dens_flat = density.reshape(-1)

    def gather(values, idx):
        return jax.vmap(lambda ii: values[:, ii])(idx)

    if orient is None:
        orient = float(normal_orientation(X_src, surf_normal_area_elem(dX_src, X_src)[0]))
    invNt = 1.0 / nt
    invNp = 1.0 / npol

    interp_nds = jnp.arange(1, 17, dtype=X_src.dtype)
    interp_nds = interp_nds[:hedgehog_order]

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
        MGrid = laplace_fxd2_u(dx, jnp.ones((ngrid,)))
        MGrid = MGrid * (Ga * precomp.Gpou)[:, None, None]
        MGrid = MGrid.reshape((ngrid, 9))

        # polar interpolation
        if interp_block_size is None:
            P = _interp_patch(Gi, precomp)  # (3, Npolar)
            Pg = _interp_patch(Ggs, precomp)  # (6, Npolar)
        else:
            P = _interp_patch_blocked(Gi, precomp, interp_block_size)
            Pg = _interp_patch_blocked(Ggs, precomp, interp_block_size)
        if P.dtype != TrgCoord.dtype:
            P = P.astype(TrgCoord.dtype)
        if Pg.dtype != TrgCoord.dtype:
            Pg = Pg.astype(TrgCoord.dtype)
        n0p = Pg[2] * Pg[5] - Pg[3] * Pg[4]
        n1p = Pg[4] * Pg[1] - Pg[5] * Pg[0]
        n2p = Pg[0] * Pg[3] - Pg[1] * Pg[2]
        rp = jnp.sqrt(n0p * n0p + n1p * n1p + n2p * n2p)

        # hedgehog target coordinates
        ntrg0 = Ggi[2, patch_dim0 * patch_dim + patch_dim0] * Ggi[5, patch_dim0 * patch_dim + patch_dim0] - Ggi[3, patch_dim0 * patch_dim + patch_dim0] * Ggi[4, patch_dim0 * patch_dim + patch_dim0]
        ntrg1 = Ggi[4, patch_dim0 * patch_dim + patch_dim0] * Ggi[1, patch_dim0 * patch_dim + patch_dim0] - Ggi[5, patch_dim0 * patch_dim + patch_dim0] * Ggi[0, patch_dim0 * patch_dim + patch_dim0]
        ntrg2 = Ggi[0, patch_dim0 * patch_dim + patch_dim0] * Ggi[3, patch_dim0 * patch_dim + patch_dim0] - Ggi[1, patch_dim0 * patch_dim + patch_dim0] * Ggi[2, patch_dim0 * patch_dim + patch_dim0]
        rtrg = jnp.sqrt(ntrg0 * ntrg0 + ntrg1 * ntrg1 + ntrg2 * ntrg2)
        scal = jnp.sqrt(rtrg * invNt * invNp) * orient / rtrg * (-20.0 / precomp.rad_dim)
        nvec = jnp.array([ntrg0, ntrg1, ntrg2]) * scal
        TrgCoordPolar = TrgCoord[None, :] + interp_nds[:, None] * nvec[None, :]

        dxp = TrgCoordPolar[None, :, :] - P.T[:, None, :]
        MPolar = laplace_fxd2_u(dxp, jnp.ones((precomp.npolar, hedgehog_order)))
        MPolar = MPolar * (rp * precomp.Ppou)[:, None, None, None]
        MPolar = MPolar.reshape((precomp.npolar, hedgehog_order, 9))
        MPolar = jnp.tensordot(MPolar, precomp.hedgehog_wts, axes=(1, 0))

        # scatter polar contributions back to grid
        idx = precomp.interp_idx.reshape(-1)
        w = precomp.M_G2P.reshape((precomp.npolar, -1))
        for k in range(9):
            contrib = (MPolar[:, k:k+1] * w).reshape(-1)
            MGrid = MGrid.at[idx, k].add(contrib)

        return jnp.sum(GiF[:, None] * MGrid, axis=0)

    Trg_flat = X_trg.reshape((3, -1)).T
    corr_fn = jax.checkpoint(corr_one) if remat else corr_one
    interp_block_size = _resolve_interp_block_size(interp_block_size, precomp.npolar, "gradb")

    if target_chunk_size is None or target_chunk_size <= 0:
        G = gather(X_flat, patch_idx)
        Gg = gather(dX_flat, patch_idx)
        GF = jax.vmap(lambda idx: dens_flat[idx])(patch_idx)
        corr = jax.vmap(corr_fn)(G, Gg, GF, Trg_flat)
    else:
        ntrg = Trg_flat.shape[0]
        pad = (-ntrg) % target_chunk_size
        if pad:
            patch_idx = jnp.pad(patch_idx, ((0, pad), (0, 0)))
            Trg_flat = jnp.pad(Trg_flat, ((0, pad), (0, 0)))
        ntrg_pad = Trg_flat.shape[0]
        n_chunks = ntrg_pad // target_chunk_size
        patch_chunks = patch_idx.reshape((n_chunks, target_chunk_size, -1))
        trg_chunks = Trg_flat.reshape((n_chunks, target_chunk_size, 3))

        def scan_fn(carry, xs):
            pidx_chunk, trg_chunk = xs
            G = gather(X_flat, pidx_chunk)
            Gg = gather(dX_flat, pidx_chunk)
            GF = jax.vmap(lambda idx: dens_flat[idx])(pidx_chunk)
            corr_chunk = jax.vmap(corr_fn)(G, Gg, GF, trg_chunk)
            return carry, corr_chunk

        _, corr_chunks = jax.lax.scan(scan_fn, None, (patch_chunks, trg_chunks))
        corr = corr_chunks.reshape((ntrg_pad, -1))[:ntrg]

    corr = corr.T.reshape((9, trg_nt, trg_np))

    base = base.reshape((9, trg_nt, trg_np))
    return base + corr


def laplace_fxd2_u_eval_vec_singular(
    X_src,
    dX_src,
    density_vec,
    trg_nt: int,
    trg_np: int,
    nfp: int,
    X_trg=None,
    digits: int = 5,
    patch_dim0: int | None = None,
    rad_dim: int | None = None,
    hedgehog_order: int = 8,
    chunk_size: int = 1024,
    target_chunk_size: int | None = None,
    patch_idx=None,
    orient: float | None = None,
    pou_dtype=None,
    patch_dtype=None,
    interp_block_size: int | str | None = "auto",
    remat: bool = False,
):
    density_vec = jnp.asarray(density_vec)
    return jax.vmap(
        lambda dens: laplace_fxd2_u_eval_singular(
            X_src,
            dX_src,
            dens,
            trg_nt,
            trg_np,
            nfp,
            X_trg=X_trg,
            digits=digits,
            patch_dim0=patch_dim0,
            rad_dim=rad_dim,
            hedgehog_order=hedgehog_order,
            chunk_size=chunk_size,
            target_chunk_size=target_chunk_size,
            patch_idx=patch_idx,
            orient=orient,
            pou_dtype=pou_dtype,
            patch_dtype=patch_dtype,
            interp_block_size=interp_block_size,
            remat=remat,
        ),
        in_axes=0,
        out_axes=0,
    )(density_vec)
