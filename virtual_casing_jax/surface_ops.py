"""JAX surface operators matching BIEST conventions."""
from __future__ import annotations

import jax
import jax.numpy as jnp

TWOPI = 2.0 * jnp.pi


def _fft_norm_factor(n):
    return 1.0 / jnp.sqrt(jnp.asarray(n, dtype=jnp.float64))


def fft_r2c(x, nt: int, npol: int):
    """Unitary r2c FFT over (nt, npol) axes."""
    x = jnp.asarray(x)
    y = jnp.fft.rfftn(x, axes=(-2, -1))
    return y * _fft_norm_factor(nt * npol)


def fft_c2r(y, nt: int, npol: int):
    """Unitary c2r FFT over (nt, npol) axes."""
    y = jnp.asarray(y)
    x = jnp.fft.irfftn(y, s=(nt, npol), axes=(-2, -1))
    return x * _fft_norm_factor(nt * npol)


def rotate_toroidal(X, nt: int, npol: int, dtheta):
    """Rotate field in toroidal angle by dtheta.

    X shape: (dof, nt, npol)
    """
    if dtheta == 0 or nt == 0:
        return X
    X = jnp.asarray(X)
    coeff = fft_r2c(X, nt, npol)
    m = jnp.fft.fftfreq(nt) * nt
    phase = jnp.exp(1j * m[:, None] * dtheta)
    coeff = coeff * phase[None, :, :]
    return fft_c2r(coeff, nt, npol)


def complete_vec_field(Y, is_surf: bool, half_period: bool, nfp: int, nt: int, npol: int, dtheta: float):
    """Match BIEST SurfaceOp::CompleteVecField.

    Y shape: (dof, nt, npol)
    Returns X shape: (dof, nfp*nt, npol)
    """
    Y = jnp.asarray(Y)
    dof = int(Y.shape[0])

    if half_period:
        # Build doubled toroidal grid with stellarator symmetry.
        t_idx = jnp.arange(nt)
        p_idx = jnp.arange(npol)
        t_mirror = nt - t_idx - 1
        p_mirror = (-p_idx) % npol
        Y_mirror = Y[:, t_mirror[:, None], p_mirror[None, :]]

        if dof == 3:
            sign = 1.0 if is_surf else -1.0
            cos_theta = jnp.cos(TWOPI / nfp)
            sin_theta = jnp.sin(TWOPI / nfp)
            x = Y_mirror[0] * sign
            y = -Y_mirror[1] * sign
            z = -Y_mirror[2] * sign
            x2 = x * cos_theta - y * sin_theta
            y2 = x * sin_theta + y * cos_theta
            z2 = z
            Y_tail = jnp.stack([x2, y2, z2], axis=0)
        else:
            Y_tail = Y_mirror

        Y = jnp.concatenate([Y, Y_tail], axis=1)
        nt = nt * 2
        half_period = False

    # Replicate for NFP field periods.
    if nfp <= 0:
        raise ValueError("nfp must be positive")

    if dof == 3:
        j = jnp.arange(nfp, dtype=Y.dtype)
        cost = jnp.cos(TWOPI * j / nfp)
        sint = jnp.sin(TWOPI * j / nfp)

        x0 = Y[0]
        y0 = Y[1]
        z0 = Y[2]

        x = cost[:, None, None] * x0[None, :, :] - sint[:, None, None] * y0[None, :, :]
        y = sint[:, None, None] * x0[None, :, :] + cost[:, None, None] * y0[None, :, :]
        z = jnp.broadcast_to(z0[None, :, :], x.shape)

        x = x.reshape((nfp * nt, npol))
        y = y.reshape((nfp * nt, npol))
        z = z.reshape((nfp * nt, npol))
        X = jnp.stack([x, y, z], axis=0)
    else:
        X = jnp.tile(Y, (1, nfp, 1))

    if dtheta != 0:
        X = rotate_toroidal(X, nfp * nt, npol, dtheta)
    return X


def upsample(X0, nt0: int, np0: int, nt1: int, np1: int):
    """Upsample using Fourier zero-padding (BIEST SurfaceOp::Upsample)."""
    X0 = jnp.asarray(X0)
    dof = int(X0.shape[0])
    coeff0 = fft_r2c(X0, nt0, np0)

    nt0_ = nt0
    np0_ = np0 // 2 + 1
    nt1_ = nt1
    np1_ = np1 // 2 + 1

    coeff1 = jnp.zeros((dof, nt1_, np1_), dtype=coeff0.dtype)

    scale = jnp.sqrt(jnp.asarray(nt1 * np1, dtype=coeff0.real.dtype)) / jnp.sqrt(jnp.asarray(nt0 * np0, dtype=coeff0.real.dtype))
    ntt = min(nt0_, nt1_)
    npp = min(np0_, np1_)

    # Scaling for Nyquist modes
    def _scale_t_pos(t):
        scale_t = 1.0
        if (nt0 % 2 == 0) and (nt0_ < nt1_) and (t == ntt // 2):
            scale_t = 0.5
        if (nt1 % 2 == 0) and (nt1_ < nt0_) and (t == ntt // 2):
            scale_t = 2.0
        return scale_t

    def _scale_t_neg(t):
        scale_t = 1.0
        if (nt0 % 2 == 0) and (nt0_ < nt1_) and (t == ntt // 2 - 1):
            scale_t = 0.5
        if (nt1 % 2 == 0) and (nt1_ < nt0_) and (t == ntt // 2 - 1):
            scale_t = 2.0
        return scale_t

    def _scale_p(p):
        scale_p = 1.0
        if (np0 % 2 == 0) and (np0_ < np1_) and (p == npp - 1):
            scale_p = 0.5
        if (np1 % 2 == 0) and (np1_ < np0_) and (p == npp - 1):
            scale_p = 2.0
        return scale_p

    t_pos = jnp.arange(0, ntt // 2 + 1)
    t_neg = jnp.arange(0, ntt // 2)
    p_idx = jnp.arange(0, npp)

    scale_t_pos = jnp.vectorize(_scale_t_pos)(t_pos)
    scale_t_neg = jnp.vectorize(_scale_t_neg)(t_neg)
    scale_p = jnp.vectorize(_scale_p)(p_idx)

    # Positive frequencies
    coeff1 = coeff1.at[:, t_pos[:, None], p_idx[None, :]].set(
        coeff0[:, t_pos[:, None], p_idx[None, :]] * (scale * scale_t_pos[:, None] * scale_p[None, :])[None, :, :]
    )

    # Negative frequencies (toroidal)
    if t_neg.size > 0:
        coeff1 = coeff1.at[:, (nt1_ - t_neg - 1)[:, None], p_idx[None, :]].set(
            coeff0[:, (nt0_ - t_neg - 1)[:, None], p_idx[None, :]] * (scale * scale_t_neg[:, None] * scale_p[None, :])[None, :, :]
        )

    X1 = fft_c2r(coeff1, nt1, np1)

    # Floating-point correction for integer upsample ratios
    ut = nt1 // nt0
    up = np1 // np0
    if nt1 == nt0 * ut and np1 == np0 * up:
        t_idx = jnp.arange(nt0)
        p_idx = jnp.arange(np0)
        tt = (t_idx * ut)[:, None]
        pp = (p_idx * up)[None, :]
        X1 = X1.at[:, tt, pp].set(X0[:, t_idx[:, None], p_idx[None, :]])

    return X1


def resample(X0, nt0: int, np0: int, nt1: int, np1: int):
    """Resample using upsample + decimation (BIEST SurfaceOp::Resample)."""
    skip_tor = int(jnp.ceil(nt0 / float(nt1)))
    skip_pol = int(jnp.ceil(np0 / float(np1)))

    X_up = upsample(X0, nt0, np0, nt1 * skip_tor, np1 * skip_pol)
    # Decimate
    X1 = X_up[:, ::skip_tor, ::skip_pol]
    return X1
