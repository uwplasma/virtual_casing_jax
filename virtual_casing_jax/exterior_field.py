"""High-level exterior magnetic-field wrapper for virtual casing."""
from __future__ import annotations

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
    source_convention: str = "vmec_jax"


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
    levels: tuple[tuple[int, int], ...] = ((64, 64), (128, 128), (256, 256))
    chunk_size: int | str = "auto"
    target_chunk_size: int | str = "auto"
    branch: Branch = "internal"
    use_jit_schedule: bool = True
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
            self.config.levels,
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
        precision=None,
    ):
        """Return the internal-current plasma field on the source surface.

        This reuses the geometry prepared by the exterior field, avoiding a
        second quadrature setup when boundary diagnostics and off-surface
        evaluations are needed together.
        """
        kwargs = {
            "digits": int(self.config.digits if digits is None else digits),
            "chunk_size": self.config.chunk_size if chunk_size is None else chunk_size,
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

    def _call_vc_B(self, xyz_soa, branch: Branch):
        kwargs = dict(
            X_trg=xyz_soa,
            digits=int(self.config.digits),
            chunk_size=self.config.chunk_size,
            target_chunk_size=self.config.target_chunk_size,
        )
        if self.config.use_jit_schedule:
            kwargs["levels"] = self.schedule_levels
            if branch == "internal":
                return self._vc.compute_internal_B_offsurf_schedule(self.B_total, **kwargs)
            return self._vc.compute_external_B_offsurf_schedule(self.B_total, **kwargs)
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

    def B_plasma_xyz(self, xyz, *, branch: Branch | None = None):
        """Return the virtual-casing plasma-current field in Cartesian components."""
        branch = self.config.branch if branch is None else branch
        if branch not in ("internal", "external"):
            raise ValueError("branch must be 'internal' or 'external'")
        xyz_soa, restore = _points_to_soa(xyz)
        xyz_soa = xyz_soa.astype(self.gamma.dtype)
        return restore(self._call_vc_B(xyz_soa, branch))

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
            points = jnp.pad(points, ((0, padding), (0, 0)), mode="edge")
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
