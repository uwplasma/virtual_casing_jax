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

__all__ = ["MAX_DERIVATIVE_ORDER", "layer_derivatives", "near_surface_derivatives"]

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


@functools.lru_cache(maxsize=None)
def _harmonic_gradients(degree):
    """Polynomial fields ``grad(phi)`` for harmonic ``phi`` of degree ``1..degree+1``.

    Returns the monomial exponents ``(nm, 3)`` up to ``degree``, the monomial
    coefficients of each field ``(nc, 3, nm)`` and each field's ``phi`` degree.
    """
    def monomials(d):
        return [(a, b, d - a - b) for a in range(d, -1, -1) for b in range(d - a, -1, -1)]

    exps = [m for d in range(degree + 1) for m in monomials(d)]
    where = {m: i for i, m in enumerate(exps)}
    fields, degrees = [], []
    for l in range(1, degree + 2):
        mons = monomials(l)
        col = {m: i for i, m in enumerate(mons)}
        row = {m: i for i, m in enumerate(monomials(l - 2))} if l > 1 else {}
        laplacian = np.zeros((len(row), len(mons)))
        for m in mons:
            for ax in range(3):
                if m[ax] > 1:
                    k = list(m)
                    k[ax] -= 2
                    laplacian[row[tuple(k)], col[m]] += m[ax] * (m[ax] - 1)
        for v in np.linalg.svd(laplacian)[2][len(row):] if row else np.eye(3):
            field = np.zeros((3, len(exps)))
            for m, c in zip(mons, v):
                for ax in range(3):
                    if m[ax]:
                        k = list(m)
                        k[ax] -= 1
                        field[ax, where[tuple(k)]] += c * m[ax]
            fields.append(field)
            degrees.append(l)
    return np.array(exps), np.array(fields), np.array(degrees)


def _monomials(s, exps, degree):
    powers = jnp.cumprod(jnp.concatenate(
        [jnp.ones_like(s)[..., None], jnp.repeat(s[..., None], degree, -1)], -1), -1)
    return (powers[..., 0, exps[:, 0]] * powers[..., 1, exps[:, 1]]
            * powers[..., 2, exps[:, 2]])


def _inside(point, coeff, start, outward):
    """True when ``point`` lies inside the surface.

    The foot point is the Newton iterate, on the Fourier interpolant of the
    nodes (``coeff`` is their ``fft2``), started at the nearest node ``start``
    (angles in radians); the side is the sign of ``point - foot`` against the
    outward normal ``outward`` at that node.
    """
    nt, npol = coeff.shape[1:]
    kt, kp = jnp.fft.fftfreq(nt, 1.0 / nt), jnp.fft.fftfreq(npol, 1.0 / npol)

    def surface(uv):
        return jnp.real(jnp.einsum(
            "cab,a,b->c", coeff, jnp.exp(1j * kt * uv[0]), jnp.exp(1j * kp * uv[1])))

    def residual(uv):
        return jax.jacfwd(surface)(uv).T @ (point - surface(uv))

    uv = start
    for _ in range(3):
        uv = uv - jnp.linalg.solve(jax.jacfwd(residual)(uv), residual(uv))
    tangents = jax.jacfwd(surface)(uv)
    normal = jnp.cross(tangents[:, 0], tangents[:, 1])
    return (point - surface(uv)) @ normal * (normal @ outward) < 0.0


#: Tikhonov weight of the polynomial fit, relative to the field at one
#: surface half-width from the fitted patch.  It keeps the fitted field
#: bounded on the far side of the surface, where it is only extrapolated.
_FIT_RIDGE = 1.0e-9

#: Ratio of the largest relative error of ``B`` to the misfit of the fitted
#: field on the ring of nodes just outside the fit, over the gate cases
#: (axisymmetric and non-axisymmetric tori, ``nfp`` 1-3, 24-96 nodes per
#: period, targets 1e-4 to 0.3 minor radii on both sides) with a margin of
#: 2.5; the gate cases reach 0.02 at most.
_ESTIMATE_FACTOR = 0.05


def near_surface_derivatives(targets, nodes, normal, weight, field, *, order=1,
                             degree=8, patch=3, batch=8, source_chunk=4096):
    """``B`` and its derivatives at targets arbitrarily close to the surface.

    The layer sum is evaluated for the density of ``field - P``, where ``P`` is
    a harmonic polynomial vector field fitted to the surface values of the
    field around the node nearest to each target.  A harmonic polynomial is
    harmonic everywhere, so the virtual-casing integral of its trace is known
    exactly -- zero outside the surface and ``-P(x)`` inside -- and is added
    back, derivatives included.  Next to the target ``field - P`` vanishes to
    high order, which removes the near-singular quadrature error that grows as
    the target approaches the nodes.

    Parameters
    ----------
    targets:
        ``(n, 3)`` points.
    nodes, normal, field:
        ``(nt, np, 3)`` quadrature nodes, outward unit normals and the total
        field at the nodes.
    weight:
        ``(nt, np)`` area element of the quadrature (including ``1/(nt np)``).
    order:
        Highest derivative returned, ``0..MAX_DERIVATIVE_ORDER``.
    degree:
        Degree of the polynomial field.
    patch:
        The fit uses the ``(2 patch + 1)^2`` nodes around the nearest node.
    batch:
        Targets evaluated together; bounds the memory.

    Returns ``(values, estimate)``: the tuple of ``order + 1`` arrays of
    :func:`layer_derivatives` and, per target, an estimate of the relative error
    of ``B``: the misfit of the fitted field on the ring of nodes just outside
    the fitted patch, relative to the RMS field, times a calibrated factor.
    In the gate cases the first derivative, in units of the surface's minor
    radius, was never worse than ``20`` times the estimate.
    """
    nt, npol = nodes.shape[:2]
    exps, table, degrees = (jnp.asarray(a) for a in _harmonic_gradients(degree))
    flat_nodes = nodes.reshape(-1, 3)
    wide = np.arange(-patch - 1, patch + 2)
    offsets = np.stack(np.meshgrid(wide, wide, indexing="ij"), -1).reshape(-1, 2)
    inner = np.abs(offsets).max(axis=1) <= patch
    ring = np.abs(offsets).max(axis=1) == patch + 1
    scale = patch * jnp.median(jnp.linalg.norm(nodes[1:] - nodes[:-1], axis=-1))
    extent = jnp.min(jnp.ptp(flat_nodes, axis=0)) / 2.0
    damping = _FIT_RIDGE * (extent / scale) ** (degrees.astype(nodes.dtype) - 1.0)
    coeff = jnp.fft.fft2(jnp.moveaxis(nodes, -1, 0)) / (nt * npol)
    field_scale = jnp.sqrt(jnp.mean(jnp.sum(field**2, axis=-1)))
    weight = weight.reshape(-1)
    flat_normal = normal.reshape(-1, 3)
    flat_field = field.reshape(-1, 3)

    def one(x):
        nearest = jnp.argmin(jnp.sum((flat_nodes - x) ** 2, axis=-1))
        i0, j0 = nearest // npol, nearest % npol
        around = flat_nodes[((i0 + offsets[:, 0]) % nt) * npol + (j0 + offsets[:, 1]) % npol]
        values = flat_field[((i0 + offsets[:, 0]) % nt) * npol + (j0 + offsets[:, 1]) % npol]
        centre = flat_nodes[nearest]
        # The three constant fields come first.  They are set to the field at
        # the nearest node, so the fit interpolates it exactly: a target at
        # distance d above that node weighs its density residual by 1/d^2 (and
        # the gradient by 1/d^3).
        design = jnp.einsum(
            "cim,pm->pic", table[3:], _monomials((around[inner] - centre) / scale, exps, degree))
        nc = design.shape[-1]
        q, r = jnp.linalg.qr(jnp.concatenate([design.reshape(-1, nc), jnp.diag(damping[3:])]))
        rhs = jnp.concatenate([(values[inner] - flat_field[nearest]).reshape(-1),
                               jnp.zeros(nc, values.dtype)])
        coefficients = jnp.concatenate([
            flat_field[nearest], jax.scipy.linalg.solve_triangular(r, q.T @ rhs)])
        monomial_coefficients = jnp.einsum("c,cim->im", coefficients, table)

        def polynomial(y):
            return _monomials((y - centre) / scale, exps, degree) @ monomial_coefficients.T

        misfit = jnp.max(jnp.abs(values[ring] - polynomial(around[ring]))) / field_scale
        residual = flat_field - polynomial(flat_nodes)
        result = layer_derivatives(
            x[None], flat_nodes,
            jnp.cross(flat_normal, residual) * weight[:, None],
            jnp.sum(flat_normal * residual, axis=-1) * weight,
            order=order, source_chunk=source_chunk)
        start = 2.0 * jnp.pi * jnp.stack((i0, j0)).astype(x.dtype) / jnp.asarray((nt, npol), x.dtype)
        inside = jax.lax.stop_gradient(_inside(x, coeff, start, flat_normal[nearest]))
        restored, derivative = [], polynomial
        for value in result:
            restored.append(value[0] - jnp.where(inside, derivative(x), 0.0))
            derivative = jax.jacfwd(derivative)
        return tuple(restored), _ESTIMATE_FACTOR * misfit

    return jax.lax.map(one, targets, batch_size=batch)
