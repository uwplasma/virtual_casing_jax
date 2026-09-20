"""Closed-form spatial derivatives of the virtual-casing layer potentials.

The exterior plasma field is

    ``B(x) = sum_k [ grad G(x - y_k) x K_k - sigma_k grad G(x - y_k) ]``,
    ``G = 1 / (4 pi |r|)``,

with ``K = (n x B) dA`` the sheet current and ``sigma = (n . B) dA`` the charge,
both already carrying the quadrature weight.  Every spatial derivative of ``B``
is therefore a contraction of the derivative tensors of ``G``, which are
polynomials in ``r`` over odd powers of ``|r|``:

    ``G_a    = -q3 r_a``
    ``G_ab   = 3 q5 r_a r_b - q3 d_ab``
    ``G_abc  = -15 q7 r_a r_b r_c + 3 q5 (d_ab r_c + d_ac r_b + d_bc r_a)``
    ``G_abcd = 105 q9 r_a r_b r_c r_d - 15 q7 (six d r r terms)
               + 3 q5 (three d d terms)``

with ``q_n = 1 / (4 pi |r|^n)``.

Taking these in closed form rather than by nesting ``jacfwd`` over the field
matters for two reasons.  One pass returns every order up to the one asked for,
instead of one traversal of the source sum per order; and the derivative is the
analytic one rather than a derivative of the quadrature, so it does not inherit
whatever the evaluation did about branches or chunking.

``K`` here is the sheet current ``n x B``.  ``VirtualCasingJAX`` stores the
opposite sign internally (``J = B x n``), so the assembly in
:mod:`virtual_casing_jax.exterior_field` negates it; getting that backwards
reproduces the field's magnitude with a wrong rotational part, which is why the
sign is named in both places.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["MAX_DERIVATIVE_ORDER", "layer_derivatives"]

#: Highest derivative order the closed forms are written out for.
MAX_DERIVATIVE_ORDER = 3

_FOUR_PI = 4.0 * np.pi

_LEVI_CIVITA = np.zeros((3, 3, 3))
_LEVI_CIVITA[0, 1, 2] = _LEVI_CIVITA[1, 2, 0] = _LEVI_CIVITA[2, 0, 1] = 1.0
_LEVI_CIVITA[0, 2, 1] = _LEVI_CIVITA[2, 1, 0] = _LEVI_CIVITA[1, 0, 2] = -1.0
_LEVI_CIVITA = jnp.asarray(_LEVI_CIVITA)
_IDENTITY = jnp.eye(3)


def _pair_block(targets, sources, sheet_current, charge, order):
    """All orders up to ``order`` for one (targets, sources) block."""
    r = targets[:, None, :] - sources[None, :, :]
    inverse = 1.0 / jnp.sum(r * r, axis=-1)
    q3 = inverse * jnp.sqrt(inverse) / _FOUR_PI
    cross = jnp.cross(r, sheet_current[None, :, :])
    s3 = q3 * charge[None, :]

    out = [jnp.einsum("ts,tsi->ti", -q3, cross) + jnp.einsum("ts,tsi->ti", s3, r)]
    if order < 1:
        return tuple(out)

    q5 = q3 * inverse
    s5 = q5 * charge[None, :]
    current3 = jnp.einsum("ts,sb->tb", q3, sheet_current)
    out.append(
        3.0 * jnp.einsum("ts,tsi,tsj->tij", q5, cross, r)
        - jnp.einsum("ijb,tb->tij", _LEVI_CIVITA, current3)
        - 3.0 * jnp.einsum("ts,tsi,tsj->tij", s5, r, r)
        + jnp.sum(s3, axis=1)[:, None, None] * _IDENTITY
    )
    if order < 2:
        return tuple(out)

    q7 = q5 * inverse
    s7 = q7 * charge[None, :]
    current5 = jnp.einsum("ts,sb,tsk->tbk", q5, sheet_current, r)
    cross5 = jnp.einsum("ts,tsi->ti", q5, cross)
    charge5 = jnp.einsum("ts,tsi->ti", s5, r)
    epsilon5 = jnp.einsum("ijb,tbk->tijk", _LEVI_CIVITA, current5)
    out.append(
        -15.0 * jnp.einsum("ts,tsi,tsj,tsk->tijk", q7, cross, r, r)
        + 3.0 * (epsilon5 + jnp.swapaxes(epsilon5, 2, 3)
                 + jnp.einsum("ti,jk->tijk", cross5, _IDENTITY))
        + 15.0 * jnp.einsum("ts,tsi,tsj,tsk->tijk", s7, r, r, r)
        - 3.0 * (jnp.einsum("tk,ij->tijk", charge5, _IDENTITY)
                 + jnp.einsum("tj,ik->tijk", charge5, _IDENTITY)
                 + jnp.einsum("ti,jk->tijk", charge5, _IDENTITY))
    )
    if order < 3:
        return tuple(out)

    q9 = q7 * inverse
    s9 = q9 * charge[None, :]
    current7 = jnp.einsum("ts,sb,tsk,tsl->tbkl", q7, sheet_current, r, r)
    epsilon7 = jnp.einsum("ijb,tbkl->tijkl", _LEVI_CIVITA, current7)
    cross7 = jnp.einsum("ts,tsi,tsl->til", q7, cross, r)
    epsilon5_flat = jnp.einsum(
        "ijb,tb->tij", _LEVI_CIVITA, jnp.einsum("ts,sb->tb", q5, sheet_current))
    charge7 = jnp.einsum("ts,tsi,tsj->tij", s7, r, r)
    charge5_sum = jnp.sum(s5, axis=1)
    delta = _IDENTITY

    current_part = (
        105.0 * jnp.einsum("ts,tsi,tsj,tsk,tsl->tijkl", q9, cross, r, r, r)
        - 15.0 * (epsilon7
                  + jnp.einsum("tikjl->tijkl", epsilon7)
                  + jnp.einsum("tiljk->tijkl", epsilon7)
                  + jnp.einsum("til,jk->tijkl", cross7, delta)
                  + jnp.einsum("tik,jl->tijkl", cross7, delta)
                  + jnp.einsum("tij,kl->tijkl", cross7, delta))
        + 3.0 * (jnp.einsum("tij,kl->tijkl", epsilon5_flat, delta)
                 + jnp.einsum("tik,jl->tijkl", epsilon5_flat, delta)
                 + jnp.einsum("til,jk->tijkl", epsilon5_flat, delta))
    )
    charge_part = (
        105.0 * jnp.einsum("ts,tsi,tsj,tsk,tsl->tijkl", s9, r, r, r, r)
        - 15.0 * (jnp.einsum("tkl,ij->tijkl", charge7, delta)
                  + jnp.einsum("tjl,ik->tijkl", charge7, delta)
                  + jnp.einsum("tjk,il->tijkl", charge7, delta)
                  + jnp.einsum("til,jk->tijkl", charge7, delta)
                  + jnp.einsum("tik,jl->tijkl", charge7, delta)
                  + jnp.einsum("tij,kl->tijkl", charge7, delta))
        + 3.0 * charge5_sum[:, None, None, None, None]
        * (jnp.einsum("ij,kl->ijkl", delta, delta)
           + jnp.einsum("ik,jl->ijkl", delta, delta)
           + jnp.einsum("il,jk->ijkl", delta, delta))[None]
    )
    out.append(current_part - charge_part)
    return tuple(out)


@functools.partial(jax.jit, static_argnames=("order", "source_chunk"))
def layer_derivatives(targets, sources, sheet_current, charge, *, order=0,
                      source_chunk=4096):
    """``B`` and its spatial derivatives up to ``order``, in one pass.

    Parameters
    ----------
    targets:
        ``(n, 3)`` Cartesian evaluation points.
    sources:
        ``(m, 3)`` Cartesian quadrature nodes on the boundary.
    sheet_current:
        ``(m, 3)``, the sheet current ``n x B`` times the quadrature weight.
    charge:
        ``(m,)``, ``(n . B)`` times the quadrature weight.
    order:
        0 for ``B`` alone, up to :data:`MAX_DERIVATIVE_ORDER`.

    Returns
    -------
    A tuple of ``order + 1`` arrays with shapes ``(n, 3)``, ``(n, 3, 3)``,
    ``(n, 3, 3, 3)``, ``(n, 3, 3, 3, 3)``, the ``k``-th being
    ``d^k B_i / dx_j ... `` at each target.

    Sources are scanned in blocks of ``source_chunk`` so the working set stays
    bounded; padding sits far away with zero weight and therefore contributes
    nothing.
    """
    order = int(order)
    if not 0 <= order <= MAX_DERIVATIVE_ORDER:
        raise ValueError(f"order must be 0..{MAX_DERIVATIVE_ORDER}, got {order}")
    targets = jnp.asarray(targets)
    sources = jnp.asarray(sources)
    sheet_current = jnp.asarray(sheet_current)
    charge = jnp.asarray(charge)

    count = sources.shape[0]
    padding = (-count) % source_chunk
    if padding:
        sources = jnp.concatenate([sources, jnp.full((padding, 3), 1.0e6, sources.dtype)])
        sheet_current = jnp.concatenate(
            [sheet_current, jnp.zeros((padding, 3), sheet_current.dtype)])
        charge = jnp.concatenate([charge, jnp.zeros(padding, charge.dtype)])

    blocks = sources.shape[0] // source_chunk
    shaped = (sources.reshape(blocks, source_chunk, 3),
              sheet_current.reshape(blocks, source_chunk, 3),
              charge.reshape(blocks, source_chunk))
    initial = tuple(jnp.zeros((targets.shape[0],) + (3,) * (k + 1), targets.dtype)
                    for k in range(order + 1))

    def accumulate(carry, block):
        values = _pair_block(targets, block[0], block[1], block[2], order)
        return tuple(a + v for a, v in zip(carry, values)), None

    total, _ = jax.lax.scan(accumulate, initial, shaped)
    return total
