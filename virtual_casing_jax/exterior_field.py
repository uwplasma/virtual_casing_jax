"""High-level exterior magnetic-field wrapper for virtual casing."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from .virtual_casing import VirtualCasingJAX


Branch = Literal["internal", "external"]


@dataclass(frozen=True)
class VmecSurfaceFieldData:
    """VMEC boundary data on a virtual-casing source grid.

    Arrays use structure-of-arrays layout ``(3, nphi, ntheta)``. ``phi`` is the
    physical toroidal angle, not the VMEC field-period coordinate ``zeta``.
    """

    gamma: jax.Array
    B_total: jax.Array
    normal: jax.Array
    area_vector: jax.Array
    theta: jax.Array
    phi: jax.Array
    nfp: int
    stellsym: bool
    signgs: int
    source_convention: str = "vmex"


@dataclass(frozen=True)
class NearSurfaceTaylorPlan:
    """Reusable interpolation data for first-order near-surface continuation."""

    center_rz: jax.Array
    surface_rz: jax.Array
    B_surface: jax.Array
    gradB_surface: jax.Array
    nfp: int


@dataclass(frozen=True)
class ExteriorFieldConfig:
    """Configuration for fixed-schedule exterior-field evaluation."""

    digits: int = 8
    src_nphi: int = 64
    src_ntheta: int = 64
    levels: tuple[tuple[int, int], ...] | None = None
    chunk_size: int | str = "auto"
    target_chunk_size: int | str = "auto"
    branch: Branch = "internal"
    use_jit_schedule: bool = True
    #: Route targets the closed forms miss through the graded rule of ``B_and_derivatives_accurate_xyz``
    #: (needs a schedule of two or more levels); ``False`` keeps the schedule alone.
    near_surface: bool = True
    #: ``(poloidal, toroidal)`` nodes of the graded rule over the full torus;
    #: ``None`` takes twice the finest schedule level, and at least 256 x 512.
    graded_nodes: tuple[int, int] | None = None
    validate_orientation: bool = True
    dtype: str = "float64"


def _as_dtype(dtype: str):
    dtype = str(dtype).lower()
    if dtype in {"float64", "double"}:
        return jnp.float64
    if dtype in {"float32", "single"}:
        return jnp.float32
    raise ValueError(f"Unsupported dtype {dtype!r}; expected 'float64' or 'float32'")


def _check_soa3(name: str, value):
    arr = jnp.asarray(value)
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ValueError(f"{name} must have shape (3, nphi, ntheta), got {arr.shape}")
    return arr


def _points_to_soa(points):
    """Return ``(3, n)`` points and a restorer for vector outputs."""
    arr = jnp.asarray(points)
    if arr.ndim == 1:
        if arr.shape[0] != 3:
            raise ValueError(f"1D point input must have length 3, got {arr.shape}")
        return arr.reshape((3, 1)), lambda out: out[:, 0]

    # Common user layout: (..., 3)
    if arr.shape[-1] == 3 and not (arr.ndim == 2 and arr.shape[0] == 3 and arr.shape[1] != 3):
        out_shape = arr.shape
        return arr.reshape((-1, 3)).T, lambda out: out.T.reshape(out_shape)

    # VirtualCasingJAX layout: (3, ...)
    if arr.shape[0] == 3:
        trailing = arr.shape[1:]
        return arr.reshape((3, -1)), lambda out: out.reshape((3,) + trailing)

    raise ValueError(f"Point input must have shape (..., 3) or (3, ...), got {arr.shape}")


def _restore_matrix_output(out, points):
    arr = jnp.asarray(points)
    out = jnp.asarray(out)
    if arr.ndim == 1:
        return out[:, :, 0]
    if arr.shape[-1] == 3 and not (arr.ndim == 2 and arr.shape[0] == 3 and arr.shape[1] != 3):
        leading = arr.shape[:-1]
        return jnp.moveaxis(out.reshape((3, 3) + leading), (0, 1), (-2, -1))
    if arr.shape[0] == 3:
        return out.reshape((3, 3) + arr.shape[1:])
    raise ValueError(f"Point input must have shape (..., 3) or (3, ...), got {arr.shape}")


def cyl_to_xyz(R_phi_Z):
    """Convert cylindrical point coordinates ``(R, phi, Z)`` to Cartesian."""
    arr = jnp.asarray(R_phi_Z)
    if arr.shape[-1] == 3:
        R = arr[..., 0]
        phi = arr[..., 1]
        Z = arr[..., 2]
        return jnp.stack((R * jnp.cos(phi), R * jnp.sin(phi), Z), axis=-1)
    if arr.shape[0] == 3:
        R = arr[0]
        phi = arr[1]
        Z = arr[2]
        return jnp.stack((R * jnp.cos(phi), R * jnp.sin(phi), Z), axis=0)
    raise ValueError(f"R_phi_Z must have shape (..., 3) or (3, ...), got {arr.shape}")


def xyz_vec_to_cyl_vec(R_phi_Z, B_xyz):
    """Convert Cartesian vector components to cylindrical components."""
    pts = jnp.asarray(R_phi_Z)
    vec = jnp.asarray(B_xyz)
    if pts.shape[-1] == 3:
        phi = pts[..., 1]
        c = jnp.cos(phi)
        s = jnp.sin(phi)
        bx = vec[..., 0]
        by = vec[..., 1]
        bz = vec[..., 2]
        return jnp.stack((c * bx + s * by, -s * bx + c * by, bz), axis=-1)
    if pts.shape[0] == 3:
        phi = pts[1]
        c = jnp.cos(phi)
        s = jnp.sin(phi)
        bx = vec[0]
        by = vec[1]
        bz = vec[2]
        return jnp.stack((c * bx + s * by, -s * bx + c * by, bz), axis=0)
    raise ValueError(f"R_phi_Z must have shape (..., 3) or (3, ...), got {pts.shape}")


def B_cyl_from_B_xyz(field_fn: Callable, R_phi_Z):
    """Evaluate a Cartesian field callback and return cylindrical components."""
    xyz = cyl_to_xyz(R_phi_Z)
    return xyz_vec_to_cyl_vec(R_phi_Z, field_fn(xyz))


def default_schedule_levels(nphi: int, ntheta: int, nfp: int):
    """Two-level schedule sized from the source grid, in full-torus counts.

    ``surface_data`` is sampled on ONE field period, while a schedule level
    counts the WHOLE torus, so the toroidal entries carry the ``nfp`` factor.
    Without it an nfp = 5 boundary sampled at 32 points per period was resolved
    by a finest level of 64 over the torus -- 13 points per period -- and the
    achieved error one minor radius off the boundary was 2.5e-02 against a
    requested 1e-6.

    Two levels are enough: the coarse level exists only to supply the
    Richardson term of the error estimate, and it costs a quarter of the fine
    one.  A schedule of one level is accepted and reports the double-layer
    quadrature error alone.
    """
    full = max(int(nphi), 1) * max(int(nfp), 1)
    ntheta = max(int(ntheta), 1)
    return ((full, ntheta), (2 * full, 2 * ntheta))


def _align_schedule_levels_to_nfp(levels, nfp: int):
    aligned = []
    nfp = max(int(nfp), 1)
    for nt, npol in levels:
        nt_i = int(nt)
        np_i = int(npol)
        if nt_i <= 0 or np_i <= 0:
            raise ValueError("ExteriorFieldConfig.levels entries must be positive")
        nt_i = nfp * ((nt_i + nfp - 1) // nfp)
        aligned.append((nt_i, np_i))
    return tuple(aligned)


def _periodic_linear(values, coordinate, period):
    count = values.shape[-1]
    scaled = jnp.mod(coordinate, period) * count / period
    lower = jnp.floor(scaled).astype(int) % count
    fraction = scaled - jnp.floor(scaled)
    return (1.0 - fraction) * values[..., lower] + fraction * values[..., (lower + 1) % count]


def _periodic_bilinear(values, first, second, first_period, second_period):
    nfirst, nsecond = values.shape[-2:]
    u = jnp.mod(first, first_period) * nfirst / first_period
    v = jnp.mod(second, second_period) * nsecond / second_period
    i0, j0 = jnp.floor(u).astype(int) % nfirst, jnp.floor(v).astype(int) % nsecond
    du, dv = u - jnp.floor(u), v - jnp.floor(v)
    i1, j1 = (i0 + 1) % nfirst, (j0 + 1) % nsecond
    return ((1.0 - du) * (1.0 - dv) * values[..., i0, j0]
            + du * (1.0 - dv) * values[..., i1, j0]
            + (1.0 - du) * dv * values[..., i0, j1]
            + du * dv * values[..., i1, j1])


def build_near_surface_taylor_plan(surface_data, B_surface, gradB_surface):
    """Build a smooth first-order near-surface continuation plan.

    Vector and gradient components are stored in the local cylindrical basis,
    making their Fourier representation periodic over one field period.
    """
    gamma = _check_soa3("surface_data.gamma", surface_data.gamma)
    B_surface = _check_soa3("B_surface", B_surface)
    gradB_surface = jnp.asarray(gradB_surface)
    if gradB_surface.shape != (3, 3) + gamma.shape[1:]:
        raise ValueError(
            "gradB_surface must have shape (3, 3, nphi, ntheta), got "
            f"{gradB_surface.shape}")

    phi = jnp.asarray(surface_data.phi)
    cphi, sphi = jnp.cos(phi), jnp.sin(phi)
    rotation = jnp.stack((
        jnp.stack((cphi, -sphi, jnp.zeros_like(phi)), axis=1),
        jnp.stack((sphi, cphi, jnp.zeros_like(phi)), axis=1),
        jnp.stack((jnp.zeros_like(phi), jnp.zeros_like(phi), jnp.ones_like(phi)), axis=1),
    ), axis=1)
    B_pt = jnp.moveaxis(B_surface, (0, 1, 2), (2, 0, 1))
    gradB_pt = jnp.moveaxis(gradB_surface, (0, 1, 2, 3), (2, 3, 0, 1))
    B_cyl = jnp.einsum("pia,pti->pta", rotation, B_pt)
    gradB_cyl = jnp.einsum("pia,ptij,pjb->ptab", rotation, gradB_pt, rotation)
    rz = jnp.stack((jnp.hypot(gamma[0], gamma[1]), gamma[2]))
    center_rz = jnp.mean(rz, axis=-1)
    geometric_angle = jnp.mod(jnp.arctan2(
        rz[1] - center_rz[1, :, None], rz[0] - center_rz[0, :, None]),
        2.0 * jnp.pi)
    alpha_grid = jnp.linspace(0.0, 2.0 * jnp.pi, gamma.shape[2], endpoint=False)

    def resample_to_geometric_angle(values):
        values_by_phi = jnp.moveaxis(values, -2, 0)

        def resample_row(alpha, row):
            flat = row.reshape((-1, row.shape[-1]))
            resampled = jax.vmap(
                lambda component: jnp.interp(
                    alpha_grid, alpha, component, period=2.0 * jnp.pi))(flat)
            return resampled.reshape(row.shape)

        return jnp.moveaxis(jax.vmap(resample_row)(
            geometric_angle, values_by_phi), 0, -2)

    return NearSurfaceTaylorPlan(
        center_rz=center_rz,
        surface_rz=resample_to_geometric_angle(rz),
        B_surface=resample_to_geometric_angle(jnp.moveaxis(B_cyl, -1, 0)),
        gradB_surface=resample_to_geometric_angle(
            jnp.moveaxis(gradB_cyl, (-2, -1), (0, 1))),
        nfp=int(surface_data.nfp),
    )


def _near_surface_taylor_B(plan: NearSurfaceTaylorPlan, point):
    x, y, z = jnp.asarray(point)
    radius, phi = jnp.hypot(x, y), jnp.arctan2(y, x)
    period = 2.0 * jnp.pi / plan.nfp
    reduced_phi = jnp.mod(phi, period)
    center = _periodic_linear(plan.center_rz, reduced_phi, period)
    alpha = jnp.arctan2(z - center[1], radius - center[0])
    rz = _periodic_bilinear(
        plan.surface_rz, reduced_phi, alpha, period, 2.0 * jnp.pi)
    B_cyl = _periodic_bilinear(
        plan.B_surface, reduced_phi, alpha, period, 2.0 * jnp.pi)
    gradB_cyl = _periodic_bilinear(
        plan.gradB_surface, reduced_phi, alpha, period, 2.0 * jnp.pi)
    cphi, sphi = jnp.cos(phi), jnp.sin(phi)
    rotation = jnp.asarray(((cphi, -sphi, 0.0), (sphi, cphi, 0.0), (0.0, 0.0, 1.0)))
    surface_point = jnp.asarray((rz[0] * cphi, rz[0] * sphi, rz[1]))
    B_surface = rotation @ B_cyl
    gradB_surface = rotation @ gradB_cyl @ rotation.T
    return B_surface + gradB_surface @ (point - surface_point)


#: Local node spacing at a graded target, as a fraction of its distance.
_GRADING = 8.0
#: The substitution ``u - a sin u`` stays monotone for ``a < 1``.
_MAX_GRADING = 0.9999
#: Targets closer than this fraction of the half-extent are continued from it.
_FLOOR = 1.0e-3
#: Fewest nodes of the default rule: resolving a distance does not depend on the data.
_MIN_NODES = (256, 512)

_ROUTE_FACTOR = 30.0  # closed-form error over the squared level difference, gate cases


def _graded_evaluate(series, phis, thetas):
    """Position, both tangents and the field on the tensor grid ``phis x thetas``.

    Separable: the toroidal phases contract first, then the poloidal ones.
    """
    c, kp, kt = series["coefficients"], series["kp"], series["kt"]
    toroidal = jnp.exp(1j * jnp.outer(phis, kp))
    poloidal = jnp.exp(1j * jnp.outer(thetas, kt))
    half = jnp.einsum("pk,klq->plq", toroidal, c)
    half_phi = jnp.einsum("pk,klq->plq", toroidal * (1j * kp)[None, :], c[..., :3])

    x = jnp.real(jnp.einsum("plq,tl->ptq", half[..., :3], poloidal))
    x_theta = jnp.real(jnp.einsum("plq,tl->ptq", half[..., :3], poloidal * (1j * kt)[None, :]))
    x_phi = jnp.real(jnp.einsum("plq,tl->ptq", half_phi, poloidal))
    B = jnp.real(jnp.einsum("plq,tl->ptq", half[..., 3:], poloidal))
    cosine, sine = jnp.cos(phis)[:, None], jnp.sin(phis)[:, None]

    def cartesian(v):
        return jnp.stack((cosine * v[..., 0] - sine * v[..., 1],
                          sine * v[..., 0] + cosine * v[..., 1], v[..., 2]), axis=-1)

    turn = jnp.stack((-sine * x[..., 0] - cosine * x[..., 1],
                      cosine * x[..., 0] - sine * x[..., 1], jnp.zeros_like(x[..., 0])), axis=-1)
    return cartesian(x), cartesian(x_theta), cartesian(x_phi) + turn, cartesian(B)


def _graded_series(gamma, field, nfp):
    """The boundary and its field as Fourier series over one field period.

    Cylindrical components are periodic over a field period, so the one-period
    samples interpolate spectrally to any angle of the full torus.  Outward is the side
    of ``e_theta x e_phi`` that encloses a positive volume.  A dense cloud of
    the whole torus seeds the nearest-point search.
    """
    nphi, ntheta = gamma.shape[1:]
    phi = jnp.linspace(0.0, 2.0 * jnp.pi / nfp, nphi, endpoint=False)[:, None]
    cosine, sine = jnp.cos(phi), jnp.sin(phi)

    def cylindrical(v):
        return jnp.stack((cosine * v[0] + sine * v[1], -sine * v[0] + cosine * v[1], v[2]), axis=-1)

    coefficients = jnp.fft.fft2(jnp.concatenate((cylindrical(gamma), cylindrical(field)), axis=-1),
                                axes=(0, 1)) / (nphi * ntheta)

    def modes(n):
        """Row picks, weights and wavenumbers, the Nyquist mode split into two halves."""
        k = np.fft.fftfreq(n, 1.0 / n)
        pick = np.r_[0:n, np.flatnonzero(k == -n / 2)]
        return (pick, np.where(np.abs(k[pick]) == n / 2, 0.5, 1.0),
                np.where(np.arange(len(pick)) < n, k[pick], -k[pick]))

    (rows, w_t, k_t), (cols, w_p, k_p) = modes(nphi), modes(ntheta)
    series = dict(coefficients=coefficients[rows][:, cols] * (w_t[:, None] * w_p[None, :])[..., None],
                  kp=jnp.asarray(k_t * nfp), kt=jnp.asarray(k_p))
    # the field's spectrum within the top quarter of either direction: what the samples cannot resolve
    edge = (np.abs(np.fft.fftfreq(nphi, 1.0 / nphi))[:, None] >= 0.375 * nphi) \
        | (np.abs(np.fft.fftfreq(ntheta, 1.0 / ntheta))[None, :] >= 0.375 * ntheta)
    series["tail"] = jnp.max(jnp.where(edge[..., None], jnp.abs(coefficients[..., 3:]), 0.0))
    x, e_theta, e_phi, _ = _graded_evaluate(
        series, phi[:, 0], jnp.linspace(0.0, 2.0 * jnp.pi, ntheta, endpoint=False))
    series["orientation"] = jax.lax.stop_gradient(jnp.sign(jnp.sum(x * jnp.cross(e_theta, e_phi))))
    angles = (jnp.linspace(0.0, 2.0 * jnp.pi, 2 * n, endpoint=False) for n in (ntheta, nphi * nfp))
    cloud_theta, cloud_phi = tuple(angles)
    cloud = _graded_evaluate(series, cloud_phi, cloud_theta)[0].reshape(-1, 3)
    grid_phi, grid_theta = jnp.meshgrid(cloud_phi, cloud_theta, indexing="ij")
    series["cloud"] = jax.lax.stop_gradient(cloud)
    series["cloud_angles"] = jnp.stack((grid_theta.ravel(), grid_phi.ravel()), axis=1)
    series["cloud_spacing"] = float(2.0 * np.pi / (2 * ntheta))
    return series


def _graded_point(series, theta, phi):
    x, e_theta, e_phi, _ = _graded_evaluate(series, jnp.atleast_1d(phi), jnp.atleast_1d(theta))
    return x[0, 0], e_theta[0, 0], e_phi[0, 0]


def _nearest_surface_point(series, point, steps: int = 6):
    """``(theta, phi)`` of the surface point nearest ``point``.

    The nearest cloud sample, then Newton on the squared distance with steps no
    longer than two cloud spacings, keeping the best iterate.
    """
    start = series["cloud_angles"][jnp.argmin(jnp.sum((series["cloud"] - point) ** 2, axis=1))]
    limit = 2.0 * series["cloud_spacing"]

    def distance2(angles):
        return jnp.sum((_graded_point(series, angles[0], angles[1])[0] - point) ** 2)

    def step(carry, _):
        angles, best, best_value = carry
        g, H = jax.grad(distance2)(angles), jax.hessian(distance2)(angles)
        det = H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]
        newton = jnp.array((H[1, 1] * g[0] - H[0, 1] * g[1],
                            H[0, 0] * g[1] - H[1, 0] * g[0])) / jnp.where(det > 0.0, det, 1.0)
        move = jnp.where(det > 0.0, newton, g / (jnp.abs(jnp.trace(H)) + 1e-30))
        move = move * jnp.minimum(1.0, limit / jnp.maximum(jnp.linalg.norm(move), 1e-300))
        angles = angles - move
        value = distance2(angles)
        better = value < best_value
        return (angles, jnp.where(better, angles, best), jnp.where(better, value, best_value)), None

    return jax.lax.scan(step, (start, start, distance2(start)), None, length=steps)[0][1]


def _graded_sources(series, center, distance, nodes):
    """Quadrature nodes and weighted layer densities graded about ``center``.

    ``theta = theta* + u - a sin u`` (likewise ``phi``) with ``u`` on a uniform
    periodic grid is entire and periodic, so the trapezoid rule in ``u`` stays
    spectrally accurate, while the node spacing at the target shrinks to
    ``(1 - a) h``; ``1 - a`` is chosen to make it ``distance / 8``.  Every
    second node of an even ``nodes`` is again such a rule, with four times the
    weights.  Returns ``(n_phi, n_theta)`` grids of positions, weighted sheet
    currents and weighted charges.
    """
    n_theta, n_phi = nodes
    _, e_theta, e_phi = _graded_point(series, center[0], center[1])
    h_theta = jnp.linalg.norm(e_theta) * 2.0 * jnp.pi / n_theta
    h_phi = jnp.linalg.norm(e_phi) * 2.0 * jnp.pi / n_phi
    a_theta = jnp.clip(1.0 - distance / (_GRADING * h_theta), 0.0, _MAX_GRADING)
    a_phi = jnp.clip(1.0 - distance / (_GRADING * h_phi), 0.0, _MAX_GRADING)
    u = 2.0 * jnp.pi * jnp.arange(n_theta) / n_theta
    v = 2.0 * jnp.pi * jnp.arange(n_phi) / n_phi
    x, x_theta, x_phi, B = _graded_evaluate(
        series, center[1] + v - a_phi * jnp.sin(v), center[0] + u - a_theta * jnp.sin(u))
    weight = (((1.0 - a_phi * jnp.cos(v)) * 2.0 * jnp.pi / n_phi)[:, None]
              * ((1.0 - a_theta * jnp.cos(u)) * 2.0 * jnp.pi / n_theta)[None, :])
    area = series["orientation"] * jnp.cross(x_theta, x_phi) * weight[..., None]
    return x, jnp.cross(area, B), jnp.sum(area * B, axis=-1)


def _continue(jets, delta, order):
    """Taylor-continue ``jets`` (``B`` and its derivatives) by ``delta``, to ``order``.

    Also returns the last term of the series for ``B``, which bounds what the
    truncation and the data's own error in the highest derivative leave.
    """
    out, last = [], 0.0
    for j in range(order + 1):
        total = jets[j]
        for k in range(1, len(jets) - j):
            term = jets[j + k]
            for _ in range(k):
                term = term @ delta
            total = total + term / math.factorial(k)
            last = term / math.factorial(k) if j == 0 else last
        out.append(total)
    return tuple(out), jnp.max(jnp.abs(last))


class VirtualCasingExteriorField:
    """JAX-native exterior field from VMEC-surface virtual-casing data.

    ``B_plasma_xyz`` defaults to the ``internal`` off-surface branch because
    VMEC plasma currents are inside the LCFS. The ``external`` branch means
    currents outside the VMEC surface, not targets outside the VMEC surface.
    """

    def __init__(
        self,
        surface_data: VmecSurfaceFieldData,
        config: ExteriorFieldConfig | None = None,
        external_B_fn: Callable | None = None,
        external_gradB_fn: Callable | None = None,
    ):
        self.surface_data = surface_data
        self.config = config or ExteriorFieldConfig()
        if self.config.branch not in ("internal", "external"):
            raise ValueError("ExteriorFieldConfig.branch must be 'internal' or 'external'")

        dtype = _as_dtype(self.config.dtype)
        gamma = _check_soa3("surface_data.gamma", surface_data.gamma).astype(dtype)
        B_total = _check_soa3("surface_data.B_total", surface_data.B_total).astype(dtype)
        normal = _check_soa3("surface_data.normal", surface_data.normal).astype(dtype)
        area_vector = _check_soa3("surface_data.area_vector", surface_data.area_vector).astype(dtype)

        if B_total.shape != gamma.shape or normal.shape != gamma.shape or area_vector.shape != gamma.shape:
            raise ValueError("gamma, B_total, normal, and area_vector must have identical shapes")

        nphi, ntheta = int(gamma.shape[1]), int(gamma.shape[2])
        self.gamma = gamma
        self.B_total = B_total
        self.normal = normal
        self.area_vector = area_vector
        self.external_B_fn = external_B_fn
        self.external_gradB_fn = external_gradB_fn
        self.schedule_levels = _align_schedule_levels_to_nfp(
            self.config.levels
            if self.config.levels is not None
            else default_schedule_levels(nphi, ntheta, int(surface_data.nfp)),
            int(surface_data.nfp),
        )

        self._vc = VirtualCasingJAX()
        self._vc.setup(
            int(self.config.digits),
            int(surface_data.nfp),
            False,
            nphi,
            ntheta,
            gamma,
            nphi,
            ntheta,
            nphi,
            ntheta,
        )
        self._sharded_cache = {}

    def plan_surface_precision(self, *, digits=None, quad_nt=None, quad_np=None):
        """Plan on-surface singular quadrature using this prepared geometry."""
        return self._vc.plan_precision(
            digits=int(self.config.digits if digits is None else digits),
            quad_nt=quad_nt,
            quad_np=quad_np,
        )

    def B_plasma_on_surface(
        self, *, digits=None, chunk_size=None, quad_nt=None, quad_np=None,
        precision=None, target_chunk_size=None,
    ):
        """Return the internal-current plasma field on the source surface.

        This reuses the geometry prepared by the exterior field, avoiding a
        second quadrature setup when boundary diagnostics and off-surface
        evaluations are needed together. Target chunking defaults to the
        configured value unless overridden for this call.
        """
        kwargs = {
            "digits": int(self.config.digits if digits is None else digits),
            "chunk_size": self.config.chunk_size if chunk_size is None else chunk_size,
            "target_chunk_size": (self.config.target_chunk_size
                                  if target_chunk_size is None else target_chunk_size),
        }
        if quad_nt is not None:
            kwargs["quad_nt"] = int(quad_nt)
        if quad_np is not None:
            kwargs["quad_np"] = int(quad_np)
        if precision is not None:
            kwargs["precision"] = precision
        return self._vc.compute_internal_B(self.B_total, **kwargs)

    def plan_near_surface(self, *, digits=None, precision=None, B_surface=None):
        """Prepare a smooth first-order field continuation near the surface.

        The singular on-surface field and gradient are computed once. Reusing
        the returned plan avoids increasingly fine direct quadrature for every
        nearby target in a field-line or grid evaluation.
        """
        digits = int(self.config.digits if digits is None else digits)
        if B_surface is None:
            B_surface = self.B_plasma_on_surface(digits=digits, precision=precision)
        precision_kwargs = {}
        if precision is not None:
            precision_kwargs = dict(
                quad_nt=precision.quad_nt, quad_np=precision.quad_np,
                patch_dim0=precision.patch_dim0, patch_idx=precision.patch_idx)
        gradB_surface = self._vc.compute_internal_gradB(
            self.B_total, digits=digits, chunk_size=self.config.chunk_size,
            target_chunk_size=self.config.target_chunk_size, **precision_kwargs)
        return build_near_surface_taylor_plan(
            self.surface_data, B_surface, gradB_surface)

    def B_plasma_near_surface_xyz(self, xyz, plan: NearSurfaceTaylorPlan):
        """Evaluate a first-order continuation of the plasma field.

        This path is intended for targets close enough to the source surface
        that direct periodic trapezoidal quadrature would require prohibitive
        refinement. Returned vectors are Cartesian and batched over ``xyz``.
        """
        xyz_soa, restore = _points_to_soa(xyz)
        values = jax.vmap(lambda point: _near_surface_taylor_B(plan, point))(xyz_soa.T)
        return restore(values.T)

    def _call_vc_B(self, xyz_soa, branch: Branch, return_estimate: bool = False):
        kwargs = dict(
            X_trg=xyz_soa,
            digits=int(self.config.digits),
            chunk_size=self.config.chunk_size,
            target_chunk_size=self.config.target_chunk_size,
        )
        if self.config.use_jit_schedule:
            kwargs["levels"] = self.schedule_levels
            kwargs["return_estimate"] = return_estimate
            if branch == "internal":
                return self._vc.compute_internal_B_offsurf_schedule(self.B_total, **kwargs)
            return self._vc.compute_external_B_offsurf_schedule(self.B_total, **kwargs)
        if return_estimate:
            raise ValueError("return_estimate requires use_jit_schedule=True")
        if branch == "internal":
            return self._vc.compute_internal_B_offsurf(self.B_total, **kwargs)
        return self._vc.compute_external_B_offsurf(self.B_total, **kwargs)

    def _call_vc_gradB(self, xyz_soa, branch: Branch):
        kwargs = dict(
            X_trg=xyz_soa,
            digits=int(self.config.digits),
            chunk_size=self.config.chunk_size,
            target_chunk_size=self.config.target_chunk_size,
        )
        if self.config.use_jit_schedule:
            kwargs["levels"] = self.schedule_levels
            if branch == "internal":
                return self._vc.compute_internal_gradB_offsurf_schedule(self.B_total, **kwargs)
            return self._vc.compute_external_gradB_offsurf_schedule(self.B_total, **kwargs)
        if branch == "internal":
            return self._vc.compute_internal_gradB_offsurf(self.B_total, **kwargs)
        return self._vc.compute_external_gradB_offsurf(self.B_total, **kwargs)

    def _level_fields(self, level=None):
        """Nodes, unit normal, area element and unit-normal densities of a level.

        Returns ``(nodes, normal, area, charge, current)`` with shapes
        ``(3, nt, np)``, ``(3, nt, np)``, ``(nt, np)``, ``(nt, np)`` and
        ``(3, nt, np)``; ``charge = B . n`` and ``current = B x n`` carry no
        quadrature weight.  The densities are built the way the schedule builds
        them: the field is recovered pointwise on the base grid and
        interpolated, and the products are formed on the level with the level's
        own normal.
        """
        from .integrals import _level_densities
        from .surface_ops import grad2d, resample, surf_normal_area_elem

        level = self.schedule_levels[-1] if level is None else level
        n_toroidal, n_poloidal = int(level[0]), int(level[1])

        def build():
            source, charge_base, current_base = self._vc._offsurface_densities(
                self.B_total, int(self.config.digits))
            nt0, np0 = int(source.shape[1]), int(source.shape[2])
            normal_base = surf_normal_area_elem(grad2d(source, nt0, np0), source)[0]
            nodes = resample(source, nt0, np0, n_toroidal, n_poloidal)
            normal, area = surf_normal_area_elem(grad2d(nodes, n_toroidal, n_poloidal), nodes)
            charge, current = _level_densities(
                charge_base, current_base, normal_base, normal,
                nt0, np0, n_toroidal, n_poloidal)
            return tuple(jnp.asarray(a) for a in (nodes, normal, area, charge, current))

        return self._cached((n_toroidal, n_poloidal), build)

    def _cached(self, key, build):
        """``build()``, remembered unless it holds a tracer, which must not outlive its trace.

        These depend only on the field data and the level; rebuilding them costs
        about one field evaluation.
        """
        def traced(value):
            return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(value))

        cache = self.__dict__.setdefault("_level_source_cache", {})
        if key not in cache or traced(cache[key]):
            with jax.ensure_compile_time_eval():  # concrete even inside a caller's jit
                value = jax.jit(build)()  # compiled: dispatched eagerly these cost seconds
            if traced(value):
                cache.pop(key, None)  # a stale entry from another trace
                return value
            cache[key] = value
        return cache[key]

    def level_sources(self, level=None):
        """Quadrature nodes and layer densities of one schedule level.

        Returns ``(nodes, sheet_current, charge)`` with shapes ``(m, 3)``,
        ``(m, 3)`` and ``(m,)``, the densities already carrying the quadrature
        weight, ready for
        :func:`~virtual_casing_jax.derivative_kernels.layer_derivatives`.

        ``level`` defaults to the finest of the schedule -- the one the
        two-level default returns unconditionally.  These nodes reproduce
        :meth:`B_plasma_xyz` rather than approximating it.
        """
        nodes, _, area, charge, current = self._level_fields(level)
        # the package stores J = B x n; the sheet current is n x B
        weight = area.reshape(-1)
        return (nodes.reshape(3, -1).T,
                -current.reshape(3, -1).T * weight[:, None],
                charge.reshape(-1) * weight)

    def B_and_derivatives_xyz(self, xyz, *, order=0, level=None, source_chunk=4096):
        """``B`` and its spatial derivatives at ``xyz``, from the closed forms.

        One traversal of the source sum returns every order up to ``order``,
        where nesting ``jacfwd`` over :meth:`B_plasma_xyz` costs one traversal
        per order and differentiates the quadrature rather than the field.

        Returns a tuple of ``order + 1`` arrays, the ``k``-th with shape
        ``xyz.shape[:-1] + (3,) * (k + 1)``.
        """
        from .derivative_kernels import layer_derivatives

        points = jnp.asarray(xyz)
        if points.ndim == 1:
            points = points[None, :]
        flat = points.reshape((-1, 3)).astype(self.gamma.dtype)
        nodes, sheet_current, charge = self.level_sources(level)
        values = layer_derivatives(flat, nodes, sheet_current, charge,
                                   order=int(order), source_chunk=int(source_chunk))
        leading = jnp.asarray(xyz).shape[:-1]
        return tuple(v.reshape(leading + v.shape[1:]) for v in values)

    def B_and_derivatives_accurate_xyz(
        self, xyz, *, order=1, tol=1.0e-8, route_order=None, branch: Branch | None = None,
    ):
        """``B`` and its derivatives at any distance from the surface, either side.

        Each target goes to the closed forms of :meth:`B_and_derivatives_xyz`
        when their estimated error is below ``tol`` and otherwise to a rule
        graded about its own nearest surface point (:meth:`_graded`), evaluated
        only where needed; the routing is a stop-gradient ``where``, safe under
        ``jit``.  Errors are scaled, ``L**k max|D**k B - ref| / B*``
        (``L`` the half-extent of the surface, ``B*`` its RMS field), and the
        ``k``-th derivative may be off by ``tol * 100**k``.  The closed-form
        error is estimated as ``30`` times the squared difference between the
        coarsest and the finest level, which covered every gate case; it does not
        see the resolution of the field data.  The routing looks at the
        derivatives up to ``route_order`` (default ``order``); ``route_order=0``
        makes it that of ``B`` alone, whatever ``order`` is, so that nested
        derivatives of the order-0 call differentiate the same map.  The result
        jumps by at most ``tol`` where the routing switches.

        Returns ``(values, estimate)``: the ``order + 1`` arrays and a relative
        error estimate of ``B`` per target (for a graded one, the change from
        halving the rule's nodes, the last Taylor term within ``1e-3`` of the
        surface, or the spectrum of the field data at the edge of what they
        resolve, whichever is largest).  ``jvp``,
        ``grad`` and ``jacfwd`` through the targets and the field data agree with
        the returned derivatives.  A target on the surface returns the outside
        limit.
        """
        branch = self.config.branch if branch is None else branch
        if branch not in ("internal", "external"):
            raise ValueError("branch must be 'internal' or 'external'")
        coarse, fine = self.schedule_levels[0], self.schedule_levels[-1]
        if coarse == fine:
            raise ValueError("the error estimate needs a schedule of at least two levels")
        order = int(order)
        route = order if route_order is None else int(route_order)
        points = jnp.asarray(xyz)
        flat = points.reshape((-1, 3)).astype(self.gamma.dtype)
        nodes = self.level_sources(fine)[0]
        reference = jnp.sqrt(jnp.mean(jnp.sum(self.B_total**2, axis=0)))
        extent = jnp.min(jnp.ptp(nodes, axis=0)) / 2.0
        # a target on a node would make the closed forms infinite and their
        # gradient NaN; they are not used there, so move it
        gap2 = jax.lax.map(lambda x: jnp.min(jnp.sum((nodes - x) ** 2, axis=-1)),
                           flat, batch_size=64)
        on_node = gap2 < (1.0e-12 * extent) ** 2
        safe = jnp.where(on_node[:, None], flat + extent, flat)
        closed = self.B_and_derivatives_xyz(safe, order=order)
        rough = self.B_and_derivatives_xyz(safe, order=order, level=coarse)
        estimates = [_ROUTE_FACTOR * (extent**k * jnp.max(jnp.abs((a - b).reshape(len(flat), -1)), axis=1)
                                      / reference) ** 2 / 100.0**k
                     for k, (a, b) in enumerate(zip(closed[:route + 1], rough[:route + 1]))]
        direct = jnp.max(jnp.stack(estimates), axis=0)
        # ``not (x <= tol)`` sends a NaN estimate to the graded rule
        needs = jax.lax.stop_gradient(~(direct <= tol) | on_node)
        values, graded = self._graded(flat, needs, order, reference, extent)
        sign = 1.0 if branch == "internal" else -1.0
        leading = points.shape[:-1]
        merged = tuple(
            sign * jnp.where(needs.reshape((-1,) + (1,) * (v.ndim - 1)), v, c).reshape(leading + v.shape[1:])
            for v, c in zip(values, closed))
        return merged, jnp.where(needs, graded, direct).reshape(leading)

    def _graded(self, points, active, order, reference, extent):
        """Graded-rule ``B`` and derivatives to ``order`` at the ``active`` targets, and an estimate.

        Each target gets its own quadrature (see :func:`_graded_sources`) about
        its nearest surface point, in the Cartesian layer sums of
        :func:`~virtual_casing_jax.derivative_kernels.layer_derivatives`.  Within
        ``_FLOOR`` of the surface the rule is evaluated at that distance and the
        jets are Taylor-continued along the normal, from the same side; the
        exterior and interior fields are analytic up to the surface, so the
        outside limit is what a target on it returns.  The estimate is the
        difference from the same rule on every second node, or the larger of that
        and the field's spectrum at the edge of what the samples resolve.  The
        grading centre is held fixed under differentiation.
        """
        from .derivative_kernels import MAX_DERIVATIVE_ORDER, layer_derivatives

        fine = self.schedule_levels[-1]
        nodes = self.config.graded_nodes or tuple(
            max(2 * n, m) for n, m in zip((fine[1], fine[0]), _MIN_NODES))
        series = self._cached(("series",), lambda: _graded_series(
            self.gamma, self.B_total, int(self.surface_data.nfp)))
        floor = _FLOOR * extent

        def locate(x):
            """Whether the target is within the floor, its grading centre and distance, and its lift."""
            x = jax.lax.stop_gradient(x)
            angles = _nearest_surface_point(series, x)
            foot, e_theta, e_phi = _graded_point(series, angles[0], angles[1])
            normal = jnp.cross(e_theta, e_phi) * series["orientation"]
            normal = normal / jnp.linalg.norm(normal)
            height = (x - foot) @ normal
            distance = jnp.maximum(jnp.abs(height), floor)
            # a target on the surface, to rounding, counts as outside
            side = jnp.where(height < -1.0e-10 * extent, -distance, distance)
            return jnp.abs(height) < floor, angles, distance, (side - height) * normal

        def rule(n, k, continued):
            def one(x, angles, distance, lift):
                moved = x + lift
                grid, current, charge = _graded_sources(series, angles, distance, n)

                def layer(step, weight, order):
                    return layer_derivatives(
                        moved[None], grid[::step, ::step].reshape(-1, 3),
                        weight * current[::step, ::step].reshape(-1, 3),
                        weight * charge[::step, ::step].reshape(-1), order=order)

                jets = tuple(v[0] for v in layer(1, 1.0, k))
                miss = jnp.max(jnp.abs(jets[0] - layer(2, 4.0, 0)[0][0])) / reference
                if continued:
                    jets, tail = _continue(jets, -lift, order)
                    miss = jnp.maximum(miss, tail / reference)
                return jets, jnp.maximum(miss, series["tail"] / reference)

            return jax.vmap(one)

        def zeros(shapes):
            return jax.tree_util.tree_map(lambda a: jnp.zeros(a.shape, a.dtype), shapes)

        idle = zeros(jax.eval_shape(locate, points[0]))
        tier, angles, distance, lift = jax.lax.map(
            lambda a: jax.lax.cond(a[1], locate, lambda _: idle, a[0]), (points, active))
        # active targets first and those within the floor first, so that a chunk shares one rule
        level = jnp.where(active, 1 + tier, 0)
        rank = jnp.argsort(-level)
        batch = 4 if jax.default_backend() == "cpu" else 32
        padding = (-points.shape[0]) % batch

        def chunks(a):
            a = a[rank]
            return jnp.concatenate((a, jnp.repeat(a[-1:], padding, axis=0))).reshape((-1, batch) + a.shape[1:])

        rules = (lambda *a: zeros(jax.eval_shape(rules[1], *a)),
                 rule(nodes, order, False), rule(nodes, MAX_DERIVATIVE_ORDER, True))
        out = jax.lax.map(
            lambda c: jax.lax.switch(jnp.max(c[0]), rules, *c[1:]),
            tuple(chunks(a) for a in (level, points, angles, distance, lift)))
        return jax.tree_util.tree_map(
            lambda a: a.reshape((-1,) + a.shape[2:])[:points.shape[0]][jnp.argsort(rank)], out)

    def _routed(self):
        """Whether ``B_plasma_xyz`` and ``gradB_plasma_xyz`` use the accurate evaluator."""
        levels = self.schedule_levels
        return self.config.near_surface and self.config.use_jit_schedule and levels[0] != levels[-1]

    def B_plasma_xyz(self, xyz, *, branch: Branch | None = None, return_estimate: bool = False):
        """Return the virtual-casing plasma-current field in Cartesian components.

        With ``return_estimate`` also return the schedule's achieved relative
        error per target, dimensionless and comparable with ``10**-digits``, so
        an unconverged schedule reports what it achieved instead of silently
        handing back its last level.  See :func:`B_plasma_error_estimate`.
        """
        branch = self.config.branch if branch is None else branch
        if branch not in ("internal", "external"):
            raise ValueError("branch must be 'internal' or 'external'")
        xyz_soa, restore = _points_to_soa(xyz)
        xyz_soa = xyz_soa.astype(self.gamma.dtype)
        if self._routed():
            (value,), estimate = self.B_and_derivatives_accurate_xyz(
                xyz_soa.T, order=0, tol=10.0**-self.config.digits, branch=branch)
            return (restore(value.T), estimate) if return_estimate else restore(value.T)
        if not return_estimate:
            return restore(self._call_vc_B(xyz_soa, branch))
        value, estimate = self._call_vc_B(xyz_soa, branch, return_estimate=True)
        return restore(value), estimate

    def B_plasma_error_estimate(self, xyz, *, branch: Branch | None = None):
        """Achieved relative error of :func:`B_plasma_xyz`, per target.

        Compare with ``10**-config.digits``.  Valid on either side of the
        source surface.  A target closer to it than about two source-grid
        spacings reports a large error: the periodic trapezoid rule is
        near-singular there and the direct schedule is not the right tool,
        whatever level it reaches.
        """
        return self.B_plasma_xyz(xyz, branch=branch, return_estimate=True)[1]

    def B_external_xyz(self, xyz):
        """Return the diagnostic external-branch virtual-casing field."""
        return self.B_plasma_xyz(xyz, branch="external")

    def B_xyz(self, xyz):
        """Return coil/external callback plus the plasma virtual-casing field."""
        B = self.B_plasma_xyz(xyz)
        if self.external_B_fn is None:
            return B
        return B + self.external_B_fn(xyz)

    def B_xyz_sharded(self, xyz, *, devices=None):
        """Evaluate a large ``(n, 3)`` target batch across JAX devices.

        Surface data are replicated while the independent target axis is
        partitioned. A single-device process follows :meth:`B_xyz` directly.
        """
        # Stage target coordinates through host memory so systems without
        # GPU peer-to-peer copies still transfer the correct shard to each device.
        points = np.asarray(xyz)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"xyz must have shape (n, 3), got {points.shape}")
        devices = tuple(jax.devices() if devices is None else devices)
        if not devices:
            raise ValueError("devices must contain at least one JAX device")
        if len(devices) == 1 or points.shape[0] < len(devices):
            return self.B_xyz(points)

        original_size = points.shape[0]
        padding = (-original_size) % len(devices)
        if padding:
            points = np.pad(points, ((0, padding), (0, 0)), mode="edge")
        local_size = points.shape[0] // len(devices)
        key = (tuple((device.platform, device.id) for device in devices), local_size)
        compiled = self._sharded_cache.get(key)
        if compiled is None:
            mesh = Mesh(np.asarray(devices, dtype=object), ("targets",))
            compiled = (
                jax.pmap(self.B_xyz, devices=devices),
                NamedSharding(mesh, PartitionSpec("targets", None)),
            )
            self._sharded_cache[key] = compiled
        function, sharding = compiled
        mapped = function(points.reshape((len(devices), local_size, 3)))
        shards = [shard.data.reshape((local_size, 3)) for shard in mapped.addressable_shards]
        result = jax.make_array_from_single_device_arrays(points.shape, sharding, shards)
        return result[:original_size] if padding else result

    def gradB_plasma_xyz(self, xyz, *, branch: Branch | None = None):
        """Return ``dB_i/dx_j`` for the virtual-casing plasma field."""
        branch = self.config.branch if branch is None else branch
        if branch not in ("internal", "external"):
            raise ValueError("branch must be 'internal' or 'external'")
        xyz_soa, _ = _points_to_soa(xyz)
        xyz_soa = xyz_soa.astype(self.gamma.dtype)
        if self._routed():
            gradient = self.B_and_derivatives_accurate_xyz(
                xyz_soa.T, order=1, tol=10.0**-self.config.digits, branch=branch)[0][1]
            return _restore_matrix_output(jnp.moveaxis(gradient, 0, -1), xyz)
        return _restore_matrix_output(self._call_vc_gradB(xyz_soa, branch), xyz)

    def gradB_xyz(self, xyz):
        """Return ``dB_i/dx_j`` for the total field.

        JAX differentiates ``external_B_fn`` point by point when an analytic
        ``external_gradB_fn`` was not supplied.
        """
        gradB = self.gradB_plasma_xyz(xyz)
        if self.external_gradB_fn is not None:
            return gradB + self.external_gradB_fn(xyz)
        if self.external_B_fn is None:
            return gradB

        points_soa, _ = _points_to_soa(xyz)
        external_gradB = jax.vmap(jax.jacfwd(self.external_B_fn))(points_soa.T)
        external_gradB = jnp.moveaxis(external_gradB, (1, 2), (0, 1))
        return gradB + _restore_matrix_output(external_gradB, xyz)

    def B_cyl(self, R_phi_Z):
        """Evaluate the total field and return ``(B_R, B_phi, B_Z)``."""
        return B_cyl_from_B_xyz(self.B_xyz, R_phi_Z)

    def export_rphiz_grid(self, R, phi, Z, *, chunk_size: int | str = "auto"):
        """Evaluate this field on a tensor-product cylindrical grid."""
        from .grid_export import evaluate_on_rphiz_grid

        return evaluate_on_rphiz_grid(self, R, phi, Z, chunk_size=chunk_size)
