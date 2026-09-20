"""A-priori trapezoid error estimate for the off-surface layer potentials.

The schedule's estimate is a-posteriori: it evaluates two grids and compares
them.  That is honest but it cannot answer the question a caller actually has,
which is *how fine a grid do I need* for this target, this tolerance and this
derivative order -- and it certifies the field only, while each spatial
derivative multiplies the quadrature error by roughly the grid count.

This module implements the a-priori estimate of af Klinteberg, Sorgentone and
Tornberg, "Quadrature error estimates for layer potentials evaluated near curved
surfaces in three dimensions", arXiv:2012.06870 (Error estimate 4, their
eq. 61, with the surface treatment of their section 6.2).  For the ``n``-point
periodic trapezoid rule applied to ``int f(t) / |gamma(t) - x|^(2p) dt`` it gives

    ``|E| ~ (4 pi n^(p-1) / Gamma(p)) |f(t0)| |G(t0)|^p exp(-n |Im t0|)``

with ``t0`` the complex root of ``R^2(t) = |gamma(t) - x|^2`` nearest the real
axis and ``G = 1 / (2 (gamma(t0) - x) . gamma'(t0))``.  There is no fitted
constant.  A surface estimate takes the grid line through the nearest node,
finds ``t0`` by Newton on the complexified boundary, integrates the
one-dimensional estimate along the other parameter with the combined linear root
model (their eqs. 101-104), repeats with the directions swapped, and adds.

Because it is a-priori it can be inverted for the grid: the error falls like
``exp(-n |Im t0|)``, so the ``n`` that reaches a tolerance follows directly.
:func:`required_levels` does that, which lets a caller fix the level *before*
tracing and hand the schedule a single static grid -- no ``lax.cond``, so every
schedule becomes branch-free and its derivatives are exactly that grid's.

This is host-side planning code, like ``VirtualCasingJAX.plan_precision``: it
runs on concrete geometry, outside any trace, and is written in NumPy.  The
kernel order ``k`` (0 for ``B``, 1 for ``gradB``, ...) enters as ``p = 3/2 + k``
and a numerator ``c_k |K| |r|^(k+1) / (4 pi)`` with ``c_k = 1, 3, 15, 105``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "BoundarySeries",
    "GridSizeError",
    "boundary_series_from_gamma",
    "density_magnitude_from_surface",
    "kst_error_estimate",
    "required_levels",
]

#: Numerator constants of the ``k``-th derivative of the Laplace/Biot-Savart
#: kernel, ``c_k / |r|^(2k+3)`` up to the tensor structure.
_DERIVATIVE_CONSTANTS = (1.0, 3.0, 15.0, 105.0)

#: Samples along the transverse parameter for the combined-root line integral.
_LINE_SAMPLES = 401

#: Newton iterations and tolerance for the complex root.
_NEWTON_STEPS = 40
_NEWTON_TOLERANCE = 1e-13


class GridSizeError(RuntimeError):
    """The requested accuracy needs a grid larger than the cap allows.

    Raised by :func:`required_levels` rather than silently returning the cap:
    a caller that asked for six digits of ``gradgradB`` at a quarter of a minor
    radius needs to be told that the direct quadrature will not deliver it, not
    handed a grid that quietly does not.
    """

    def __init__(self, tolerance, order, cap, achieved, n_toroidal, n_poloidal):
        self.tolerance = float(tolerance)
        self.order = int(order)
        self.cap = int(cap)
        self.achieved = float(achieved)
        super().__init__(
            f"derivative order {int(order)} to {float(tolerance):.1e} needs a source grid "
            f"beyond the cap of {int(cap)}: at {int(n_toroidal)} x {int(n_poloidal)} the "
            f"estimated error is still {float(achieved):.3e}. Move the targets further "
            f"from the surface, ask for fewer digits or a lower order, or raise the cap."
        )


@dataclass(frozen=True)
class BoundarySeries:
    """A trigonometric interpolant of the boundary, evaluable at complex angles.

    ``r_cos``/``z_sin`` etc. are the two-dimensional Fourier coefficients of
    ``R`` and ``Z`` over ``theta in [0, 2 pi)`` and ``phi in [0, 2 pi / nfp)``.
    Cartesian components are not stored: ``x = R cos phi`` is not periodic over
    one field period, while ``R`` and ``Z`` are, so the series is taken in
    ``(R, Z)`` and the Cartesian point is formed afterwards -- which also works
    unchanged when ``phi`` is complex.
    """

    coefficients: np.ndarray  # (2, n_poloidal_modes, n_toroidal_modes), complex
    m: np.ndarray  # poloidal wavenumbers, (n_poloidal_modes,)
    n: np.ndarray  # toroidal wavenumbers including the nfp factor
    nfp: int

    def evaluate(self, theta, phi, derivatives=True):
        """Position and both parameter derivatives; ``theta`` or ``phi`` may be complex.

        The phase separates, ``exp(i(m theta + n phi)) = exp(i m theta)
        exp(i n phi)``, so the mode sum is a batched bilinear form rather than a
        reduction over a ``(targets, 2, m, n)`` temporary.  With
        ``derivatives=False`` only the position is formed, which is all Newton
        needs while it is iterating.
        """
        theta = np.asarray(theta, dtype=complex)
        phi = np.asarray(phi, dtype=complex)
        theta_phase = np.exp(1j * np.multiply.outer(theta, self.m))
        phi_phase = np.exp(1j * np.multiply.outer(phi, self.n))

        def contract(weighted):
            return np.einsum("...m,kmn,...n->...k", theta_phase, weighted, phi_phase,
                             optimize=True)

        rz = contract(self.coefficients)
        R, Z = rz[..., 0], rz[..., 1]
        cos_phi, sin_phi = np.cos(phi), np.sin(phi)
        position = np.stack([R * cos_phi, R * sin_phi, Z], axis=-1)
        if not derivatives:
            return position

        rz_theta = contract(self.coefficients * (1j * self.m)[:, None])
        rz_phi = contract(self.coefficients * (1j * self.n)[None, :])
        R_theta, Z_theta = rz_theta[..., 0], rz_theta[..., 1]
        R_phi, Z_phi = rz_phi[..., 0], rz_phi[..., 1]
        d_theta = np.stack([R_theta * cos_phi, R_theta * sin_phi, Z_theta], axis=-1)
        d_phi = np.stack([R_phi * cos_phi - R * sin_phi,
                          R_phi * sin_phi + R * cos_phi, Z_phi], axis=-1)
        return position, d_theta, d_phi


def boundary_series_from_gamma(gamma, nfp: int) -> BoundarySeries:
    """Fourier-interpolate a ``(3, nphi, ntheta)`` boundary grid of one field period."""
    gamma = np.asarray(gamma)
    if gamma.ndim != 3 or gamma.shape[0] != 3:
        raise ValueError(f"gamma must have shape (3, nphi, ntheta), got {gamma.shape}")
    nphi, ntheta = gamma.shape[1], gamma.shape[2]
    R = np.hypot(gamma[0], gamma[1])
    Z = gamma[2]
    # axes: (phi, theta) -> coefficients indexed (theta mode, phi mode)
    coefficients = np.stack([np.fft.fft2(R).T, np.fft.fft2(Z).T]) / (nphi * ntheta)
    m = np.fft.fftfreq(ntheta, 1.0 / ntheta)
    n = np.fft.fftfreq(nphi, 1.0 / nphi) * float(nfp)
    return BoundarySeries(coefficients=coefficients, m=m, n=n, nfp=int(nfp))


def _complex_roots(series, targets, theta, phi, direction, guess):
    """Newton for the complex root of ``|gamma - x|^2``, all targets at once.

    ``theta``, ``phi`` and ``guess`` are ``(n,)``: one parameter line per
    target, through its own nearest node.  The iteration count is fixed rather
    than broken out of per target, so every target follows the same code path;
    converged entries take steps of order the rounding error and stay put.
    """
    variable = (phi if direction == "phi" else theta) + 1j * guess

    def evaluate(current):
        if direction == "phi":
            position, _, derivative = series.evaluate(theta, current)
        else:
            position, derivative, _ = series.evaluate(current, phi)
        return position, derivative

    for _ in range(_NEWTON_STEPS):
        position, derivative = evaluate(variable)  # noqa: F841 - both used below
        separation = position - targets
        value = np.sum(separation * separation, axis=-1)
        slope = 2.0 * np.sum(separation * derivative, axis=-1)
        stalled = slope == 0.0
        step = np.where(stalled, 0.0, value / np.where(stalled, 1.0, slope))
        variable = variable - step
        if np.all(np.abs(step) < _NEWTON_TOLERANCE):
            break

    position, derivative = evaluate(variable)
    separation = position - targets
    denominator = 2.0 * np.sum(separation * derivative, axis=-1)
    singular = denominator == 0.0
    factor = np.where(singular, np.inf, 1.0 / np.where(singular, 1.0, denominator))
    return variable, factor, np.sqrt(np.sum(np.abs(separation) ** 2, axis=-1))


def _combined_root_line_integrals(count, separation, parallel, transverse, imaginary_part):
    """``int exp(-n Im t0(s)) ds`` per target, combined linear root model (KST eqs. 101-104)."""
    s = np.linspace(-np.pi, np.pi, _LINE_SAMPLES)
    row = s[None, :]

    def dot(u, v):
        return np.sum(u * v, axis=-1)[:, None]

    a = (dot(separation, separation) + 2.0 * dot(separation, transverse) * row
         + dot(transverse, transverse) * row**2)
    b = 2.0 * dot(separation, parallel) + 2.0 * dot(transverse, parallel) * row
    c = dot(parallel, parallel)
    imaginary = np.sqrt(np.maximum(4.0 * a * c - b**2, 0.0)) / (2.0 * c)
    shifted = (imaginary_part[:, None] - imaginary[:, _LINE_SAMPLES // 2, None] + imaginary)
    return np.trapezoid(np.exp(-count * shifted), s, axis=-1)


def _node_positions(series, n_toroidal, n_poloidal):
    """Cartesian source nodes of a level, by FFT upsampling rather than mode sums.

    The nodes are a regular grid, so evaluating the series pointwise at each of
    them costs O(nodes x modes) -- 1.3 s for a 320 x 64 level of a 32 x 32
    boundary, which dominated everything else in the estimate. Zero-padding the
    coefficients and inverting the transform gives the same numbers in
    O(nodes log nodes).

    ``R`` and ``Z`` have period ``2 pi / nfp`` in ``phi``, so the full torus is
    one period tiled ``nfp`` times.  When the toroidal count is not a multiple
    of ``nfp`` that tiling does not exist and the direct evaluation is used.
    """
    nfp = max(int(series.nfp), 1)
    if n_toroidal % nfp:
        theta = np.linspace(0.0, 2.0 * np.pi, n_poloidal, endpoint=False)
        phi = np.linspace(0.0, 2.0 * np.pi, n_toroidal, endpoint=False)
        phi_mesh, theta_mesh = np.meshgrid(phi, theta, indexing="ij")
        return np.real(series.evaluate(theta_mesh.ravel(), phi_mesh.ravel())[0])

    per_period = n_toroidal // nfp
    poloidal_index = (np.rint(series.m).astype(int) % n_poloidal)
    # the stored toroidal wavenumbers carry the nfp factor; the per-period
    # sample index sees only the integer part
    toroidal_index = (np.rint(series.n / nfp).astype(int) % per_period)

    values = []
    for coefficients in series.coefficients:
        padded = np.zeros((n_poloidal, per_period), dtype=complex)
        np.add.at(padded, (poloidal_index[:, None], toroidal_index[None, :]), coefficients)
        values.append(np.real(np.fft.ifft2(padded)) * (n_poloidal * per_period))

    R = np.tile(values[0].T, (nfp, 1))
    Z = np.tile(values[1].T, (nfp, 1))
    phi = np.linspace(0.0, 2.0 * np.pi, n_toroidal, endpoint=False)[:, None]
    return np.stack([(R * np.cos(phi)).ravel(),
                     (R * np.sin(phi)).ravel(), Z.ravel()], axis=-1)


def _nearest_node_indices(nodes, targets, chunk=64):
    """Index of the closest source node to each target, in bounded memory."""
    indices = np.empty(len(targets), dtype=int)
    for begin in range(0, len(targets), chunk):
        block = targets[begin:begin + chunk]
        squared = ((nodes[None, :, :] - block[:, None, :]) ** 2).sum(axis=-1)
        indices[begin:begin + chunk] = np.argmin(squared, axis=1)
    return indices


def kst_error_estimate(series, targets, level, order, density_magnitude):
    """Estimated absolute quadrature error per target, for one grid and one order.

    Parameters
    ----------
    series:
        A :class:`BoundarySeries` for the source boundary.
    targets:
        ``(n, 3)`` Cartesian points, outside or inside the surface.
    level:
        ``(n_toroidal, n_poloidal)`` full-torus source counts, as a schedule
        level.
    order:
        0 for the field, 1 for its gradient, and so on up to 3.
    density_magnitude:
        ``|area_vector x B|`` on the source surface, as a callable
        ``(theta, phi) -> float`` or a constant; see
        :func:`density_magnitude_from_surface`.  This is the density in the
        PARAMETRIZATION, so it carries the surface Jacobian.  The estimate is
        linear in it, and passing the unit-normal ``|n x B|`` instead makes it
        wrong by that Jacobian.

    Returns
    -------
    ``(n,)`` array of estimated absolute errors.
    """
    order = int(order)
    if not 0 <= order < len(_DERIVATIVE_CONSTANTS):
        raise ValueError(f"order must be 0..{len(_DERIVATIVE_CONSTANTS) - 1}, got {order}")
    n_toroidal, n_poloidal = int(level[0]), int(level[1])
    targets = np.asarray(targets, dtype=float)
    if targets.ndim != 2 or targets.shape[1] != 3:
        raise ValueError(f"targets must have shape (n, 3), got {targets.shape}")

    theta_nodes = np.linspace(0.0, 2.0 * np.pi, n_poloidal, endpoint=False)
    phi_nodes = np.linspace(0.0, 2.0 * np.pi, n_toroidal, endpoint=False)
    phi_mesh, theta_mesh = np.meshgrid(phi_nodes, theta_nodes, indexing="ij")
    nodes = _node_positions(series, n_toroidal, n_poloidal)

    exponent = 1.5 + order
    constant = _DERIVATIVE_CONSTANTS[order]
    flat_theta, flat_phi = theta_mesh.ravel(), phi_mesh.ravel()
    nearest = _nearest_node_indices(nodes, targets)
    theta_star, phi_star = flat_theta[nearest], flat_phi[nearest]

    position, d_theta, d_phi = series.evaluate(theta_star, phi_star)
    separation = np.real(position) - targets
    distance = np.linalg.norm(separation, axis=-1)
    if callable(density_magnitude):
        magnitude = np.array([density_magnitude(t, p)
                              for t, p in zip(theta_star, phi_star)])
    else:
        magnitude = np.full(len(targets), float(density_magnitude))

    estimates = np.zeros(len(targets))
    for direction, count, parallel, transverse in (
        ("phi", n_toroidal, np.real(d_phi), np.real(d_theta)),
        ("theta", n_poloidal, np.real(d_theta), np.real(d_phi)),
    ):
        guess = distance / np.maximum(np.linalg.norm(parallel, axis=-1), 1e-300)
        root, factor, root_distance = _complex_roots(
            series, targets, theta_star, phi_star, direction, guess)
        numerator = constant * magnitude * root_distance ** (order + 1) / (4.0 * np.pi)
        estimates += (numerator * np.abs(factor) ** exponent
                      * 4.0 * np.pi * count ** (exponent - 1.0) / math.gamma(exponent)
                      * _combined_root_line_integrals(
                          count, separation, parallel, transverse, np.abs(root.imag)))
    return estimates


def density_magnitude_from_surface(surface_data):
    """``|area_vector x B|`` on the source grid, as a nearest-node lookup.

    The KST numerator is the density in the PARAMETRIZATION, so it carries the
    surface Jacobian: the integral is ``int |K| dS = int |area_vector x B|
    dtheta dphi``.  Passing the unit-normal ``|n x B|`` instead drops that
    factor and makes the estimate wrong by the Jacobian -- on a torus of
    aspect ratio 3 that is a factor of about 2.5.
    """
    area = np.asarray(surface_data.area_vector)
    field = np.asarray(surface_data.B_total)
    magnitude = np.linalg.norm(np.cross(area, field, axis=0), axis=0)
    nphi, ntheta = magnitude.shape
    nfp = max(int(surface_data.nfp), 1)
    period = 2.0 * np.pi / nfp

    def lookup(theta, phi):
        i = int(round((phi % period) / period * nphi)) % nphi
        j = int(round((theta % (2.0 * np.pi)) / (2.0 * np.pi) * ntheta)) % ntheta
        return float(magnitude[i, j])

    return lookup


#: Multiplier applied to the estimate when sizing a grid.  Measured against the
#: true error of single levels on a shaped nfp = 3 boundary, four distances and
#: four derivative orders, the estimate is conservative by a median factor of
#: 1.3 to 2.2 -- but its minimum over targets dips to about 0.3 at the two ends,
#: where the rule is either far from resolved or already at its floor.  Sizing a
#: grid is the one use where under-estimating is not recoverable, so it carries a
#: margin that covers the measured spread.
DEFAULT_SAFETY = 4.0


def required_levels(series, targets, *, digits, order, density_magnitude, scale,
                    base_level, cap=4096, growth=2, safety=DEFAULT_SAFETY):
    """Smallest doubling of ``base_level`` whose estimated error meets ``digits``.

    ``density_magnitude`` is ``|area_vector x B|``, as
    :func:`density_magnitude_from_surface` builds it; ``scale`` is the RMS field
    on the source surface, so the comparison with ``10**-digits`` matches the
    schedule's own relative convention.

    Returns ``(levels, estimate)`` with ``levels`` a one-level schedule: pin it
    before tracing and the quadrature becomes a fixed function of its inputs,
    with no ``lax.cond`` and derivatives that are exactly that grid's.

    Raises :class:`GridSizeError` when the cap is reached first, rather than
    returning the cap and letting the caller believe the tolerance was met.
    """
    tolerance = 10.0 ** (-int(digits))
    n_toroidal, n_poloidal = int(base_level[0]), int(base_level[1])
    scale = float(scale)
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    while True:
        achieved = float(safety) * float(np.max(kst_error_estimate(
            series, targets, (n_toroidal, n_poloidal), order, density_magnitude))) / scale
        if achieved <= tolerance:
            return ((n_toroidal, n_poloidal),), achieved
        if n_toroidal * growth > cap or n_poloidal * growth > cap:
            raise GridSizeError(tolerance, order, cap, achieved, n_toroidal, n_poloidal)
        n_toroidal *= growth
        n_poloidal *= growth
