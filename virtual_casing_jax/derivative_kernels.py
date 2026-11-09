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


@functools.cache
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
    # a point on the surface, to rounding, counts as outside
    tolerance = 16.0 * jnp.finfo(point.dtype).eps * jnp.max(jnp.abs(point))
    return (point - surface(uv)) @ normal * jnp.sign(normal @ outward) < -tolerance


#: Tikhonov weight of the polynomial fit, relative to the field at one
#: surface half-width from the fitted patch.  It keeps the fitted field
#: bounded on the far side of the surface, where it is only extrapolated.
_FIT_RIDGE = 2.0e-10

#: The fit uses the ``(2 * _PATCH + 1)**2`` fine nodes around the nearest
#: coarse node.
_PATCH = 5

#: The fine level replaces the coarse one inside a window of ``_WINDOW`` coarse
#: nodes in each direction around the target, through the smooth bump
#: ``exp(_SHAPE (sqrt(1 - t^2) - 1))``: narrow enough to be a few nodes wide,
#: wide enough to be resolved by the coarse level (aliasing and truncation both
#: stay below 1e-9).
_WINDOW = 10
_SHAPE = 20.0

#: Ratio of the largest relative error of ``B`` to the misfit of the fitted
#: field on the ring of nodes just outside the fit, over the gate cases
#: (non-axisymmetric tori and a rotating ellipse, 16-96 nodes per direction,
#: targets 1e-4 to 1 minor radii on both sides) with a margin of 2.
_ESTIMATE_FACTOR = 0.25


def _horner(coefficients, s, degree):
    """``sum_a coefficients[:, a, b, c] s_x^a s_y^b s_z^c``: vector polynomial by Horner."""
    x, y, z = s[..., 0:1], s[..., 1:2], s[..., 2:3]
    total = 0.0
    for a in range(degree, -1, -1):
        row = 0.0
        for b in range(degree - a, -1, -1):
            line = coefficients[:, a, b, degree - a - b]
            for c in range(degree - a - b - 1, -1, -1):
                line = line * z + coefficients[:, a, b, c]
            row = row * y + line
        total = total * x + row
    return total


def _bump(t):
    """1 at ``t = 0``, about ``2e-9`` at ``|t| = 1`` and 0 beyond."""
    return jnp.where(jnp.abs(t) < 1.0,
                     jnp.exp(_SHAPE * (jnp.sqrt(1.0 - jnp.minimum(t * t, 1.0)) - 1.0)), 0.0)


def near_surface_derivatives(targets, active, coarse, fine, *, order=1, degree=8):
    """``B`` and its derivatives at targets arbitrarily close to the surface.

    The layer sum is evaluated for the density of ``field - P``, where ``P`` is
    a harmonic polynomial vector field fitted to the surface values of the
    field around the node nearest to each target.  A harmonic polynomial is
    harmonic everywhere, so the virtual-casing integral of its trace is known
    exactly -- zero outside the surface and ``-P(x)`` inside -- and is added
    back, derivatives included.  Next to the target ``field - P`` vanishes to
    high order, which removes the near-singular quadrature error that grows as
    the target approaches the nodes.  Within a smooth window around the target
    the sum runs over the ``fine`` level, elsewhere over the ``coarse`` one,
    so the cost is that of the coarse level.

    Parameters
    ----------
    targets, active:
        ``(n, 3)`` points and an ``(n,)`` mask; the sum is only evaluated for
        the active targets (inactive ones return zeros), in chunks sorted so
        that the active ones come first.
    coarse, fine:
        ``(nodes, normal, weight, field)`` at two levels, the fine one a whole
        multiple of the coarse one: ``(nt, np, 3)`` nodes, outward unit normals
        and total field, and the ``(nt, np)`` area element including the
        quadrature weight.
    order:
        Highest derivative returned, ``0..MAX_DERIVATIVE_ORDER``.
    degree:
        Degree of the polynomial field.

    Returns ``(values, estimate)``: the tuple of ``order + 1`` arrays of
    :func:`layer_derivatives` and, per target, an estimate of the relative error
    of ``B``: the misfit of the fitted field on the ring of nodes just outside
    the fitted patch, relative to the RMS field, times a calibrated factor.
    """
    nodes, normal, weight, field = coarse
    fine_nodes, _, _, fine_field = fine
    nt, npol = nodes.shape[:2]
    mt, mp = fine_nodes.shape[:2]
    ratio = mt // nt
    half = min(_WINDOW, nt // 2, npol // 2)
    exps, table, degrees = (jnp.asarray(a) for a in _harmonic_gradients(degree))
    wide = np.arange(-_PATCH - 1, _PATCH + 2)
    offsets = np.stack(np.meshgrid(wide, wide, indexing="ij"), -1).reshape(-1, 2)
    inner = np.abs(offsets).max(axis=1) <= _PATCH
    ring = np.abs(offsets).max(axis=1) == _PATCH + 1
    spacing = [jnp.mean(jnp.linalg.norm(jnp.diff(nodes, axis=a), axis=-1)) for a in (0, 1)]
    scale = _PATCH * jnp.sqrt(spacing[0] * spacing[1]) / ratio
    extent = jnp.min(jnp.ptp(nodes.reshape(-1, 3), axis=0)) / 2.0
    damping = _FIT_RIDGE * (extent / scale) ** (degrees.astype(nodes.dtype) - 1.0)
    coeff = jnp.fft.fft2(jnp.moveaxis(nodes, -1, 0)) / (nt * npol)
    field_scale = jnp.sqrt(jnp.mean(jnp.sum(field**2, axis=-1)))
    flat_nodes, flat_normal = nodes.reshape(-1, 3), normal.reshape(-1, 3)
    cells = np.arange(1 - half * ratio, half * ratio)
    bump = _bump(cells / (half * ratio))
    # the node under the nearest coarse node has a vanishing residual: drop it
    disc = (bump[:, None] * bump[None, :]).at[half * ratio - 1, half * ratio - 1].set(0.0)
    tiny = (1.0e-12 * extent) ** 2

    def one(x):
        nearest = jnp.argmin(jnp.sum((flat_nodes - x) ** 2, axis=-1))
        i0, j0 = nearest // npol, nearest % npol
        rows, cols = (i0 * ratio + offsets[:, 0]) % mt, (j0 * ratio + offsets[:, 1]) % mp
        around, values = fine_nodes[rows, cols], fine_field[rows, cols]
        centre, centre_value = nodes[i0, j0], field[i0, j0]

        def local(y):
            return (y - centre) / scale

        design = jnp.einsum("cim,pm->pic", table[3:], _monomials(local(around[inner]), exps, degree))
        nc = design.shape[-1]
        rhs = jnp.concatenate([(values[inner] - centre_value).reshape(-1), jnp.zeros(nc, values.dtype)])
        matrix = jnp.concatenate([design.reshape(-1, nc), jnp.diag(damping[3:])])
        triangle = jnp.linalg.qr(matrix, mode="r")

        def solve(b):  # semi-normal equations with one refinement step
            return jax.scipy.linalg.solve_triangular(
                triangle, jax.scipy.linalg.solve_triangular(triangle, b, trans=1))

        fitted = solve(matrix.T @ rhs)
        fitted = fitted + solve(matrix.T @ (rhs - matrix @ fitted))
        monomial_coefficients = jnp.einsum(
            "c,cim->im", jnp.concatenate([centre_value, fitted]), table)
        tensor = jnp.zeros((3,) + (degree + 1,) * 3, x.dtype).at[
            :, exps[:, 0], exps[:, 1], exps[:, 2]].set(monomial_coefficients)

        def polynomial(y):
            return _horner(tensor, local(y), degree)

        misfit = jnp.max(jnp.abs(values[ring] - polynomial(around[ring]))) / field_scale
        # coarse nodes weighted by one minus the window, fine nodes by the window
        wrapped = (jnp.arange(nt) - i0 + nt // 2) % nt - nt // 2, (jnp.arange(npol) - j0 + npol // 2) % npol - npol // 2
        outside = 1.0 - _bump(wrapped[0] / half)[:, None] * _bump(wrapped[1] / half)[None, :]
        window = ((i0 * ratio + cells) % mt)[:, None], ((j0 * ratio + cells) % mp)[None, :]
        pts = jnp.concatenate([flat_nodes, fine_nodes[window].reshape(-1, 3)])
        unit = jnp.concatenate([flat_normal, fine[1][window].reshape(-1, 3)])
        total = jnp.concatenate([field.reshape(-1, 3), fine_field[window].reshape(-1, 3)])
        w = jnp.concatenate([(outside * weight).reshape(-1), (disc * fine[2][window]).reshape(-1)])
        # a node on the target contributes nothing; moving it away keeps the sum finite
        coincident = jnp.sum((pts - x) ** 2, axis=-1) < tiny
        pts = jnp.where(coincident[:, None], 1.0e6, pts)
        w = jnp.where(coincident, 0.0, w)
        residual = total - polynomial(pts)
        result = layer_derivatives(
            x[None], pts, jnp.cross(unit, residual) * w[:, None],
            jnp.sum(unit * residual, axis=-1) * w, order=order)
        start = 2.0 * jnp.pi * jnp.stack((i0, j0)).astype(x.dtype) / jnp.asarray((nt, npol), x.dtype)
        inside = jax.lax.stop_gradient(_inside(x, coeff, start, flat_normal[nearest]))
        restored, derivative = [], polynomial
        for value in result:
            restored.append(value[0] - jnp.where(inside, derivative(x), 0.0))
            derivative = jax.jacfwd(derivative)
        return tuple(restored), _ESTIMATE_FACTOR * misfit

    batch = 16 if jax.default_backend() == "cpu" else 64  # chunks the active-first order is cut into
    count = targets.shape[0]
    rank = jnp.argsort(~active)
    pad = (-count) % batch
    sorted_targets = jnp.concatenate([targets[rank], jnp.repeat(targets[rank][-1:], pad, axis=0)])
    flags = jnp.concatenate([active[rank], jnp.zeros(pad, bool)])
    shape = jax.eval_shape(jax.vmap(one), sorted_targets[:batch])
    idle = jax.tree_util.tree_map(lambda s: jnp.zeros(s.shape, s.dtype), shape)

    def chunk(args):
        return jax.lax.cond(jnp.any(args[1]), lambda: jax.vmap(one)(args[0]), lambda: idle)

    out = jax.lax.map(chunk, (sorted_targets.reshape(-1, batch, 3), flags.reshape(-1, batch)))
    restore = jnp.argsort(rank)
    return jax.tree_util.tree_map(lambda a: a.reshape((-1,) + a.shape[2:])[:count][restore], out)
