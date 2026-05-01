"""High-level exterior magnetic-field wrapper for virtual casing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import jax
import jax.numpy as jnp

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

    def _call_vc_B(self, xyz_soa, branch: Branch):
        kwargs = dict(
            X_trg=xyz_soa,
            digits=int(self.config.digits),
            chunk_size=self.config.chunk_size,
            target_chunk_size=self.config.target_chunk_size,
        )
        if self.config.use_jit_schedule:
            kwargs["levels"] = tuple((int(nt), int(np)) for nt, np in self.config.levels)
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
            kwargs["levels"] = tuple((int(nt), int(np)) for nt, np in self.config.levels)
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

    def gradB_plasma_xyz(self, xyz, *, branch: Branch | None = None):
        """Return ``dB_i/dx_j`` for the virtual-casing plasma field."""
        branch = self.config.branch if branch is None else branch
        if branch not in ("internal", "external"):
            raise ValueError("branch must be 'internal' or 'external'")
        xyz_soa, _ = _points_to_soa(xyz)
        xyz_soa = xyz_soa.astype(self.gamma.dtype)
        return _restore_matrix_output(self._call_vc_gradB(xyz_soa, branch), xyz)

    def gradB_xyz(self, xyz):
        """Return ``dB_i/dx_j`` for the total field when callbacks support it."""
        gradB = self.gradB_plasma_xyz(xyz)
        if self.external_gradB_fn is None:
            return gradB
        return gradB + self.external_gradB_fn(xyz)

    def B_cyl(self, R_phi_Z):
        """Evaluate the total field and return ``(B_R, B_phi, B_Z)``."""
        return B_cyl_from_B_xyz(self.B_xyz, R_phi_Z)

    def export_rphiz_grid(self, R, phi, Z, *, chunk_size: int | str = "auto"):
        """Evaluate this field on a tensor-product cylindrical grid."""
        from .grid_export import evaluate_on_rphiz_grid

        return evaluate_on_rphiz_grid(self, R, phi, Z, chunk_size=chunk_size)
