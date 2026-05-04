"""Cylindrical grid evaluation and lightweight NetCDF export."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import jax.numpy as jnp
from scipy.io import netcdf_file


def _resolve_chunk_size(chunk_size, npts: int) -> int:
    if chunk_size is None or (isinstance(chunk_size, str) and chunk_size.lower() == "auto"):
        return int(npts)
    chunk = int(chunk_size)
    if chunk <= 0:
        return int(npts)
    return chunk


def evaluate_on_rphiz_grid(field, R, phi, Z, *, chunk_size: int | str = "auto") -> dict:
    """Return ``BR``, ``Bphi``, ``BZ``, and ``absB`` on an ``R, phi, Z`` grid."""
    R = jnp.asarray(R)
    phi = jnp.asarray(phi)
    Z = jnp.asarray(Z)
    RR, PP, ZZ = jnp.meshgrid(R, phi, Z, indexing="ij")
    pts = jnp.stack((RR, PP, ZZ), axis=-1)
    flat = pts.reshape((-1, 3))
    npts = int(flat.shape[0])
    chunk = _resolve_chunk_size(chunk_size, npts)

    blocks = []
    for start in range(0, npts, chunk):
        blocks.append(field.B_cyl(flat[start : start + chunk]))
    B = jnp.concatenate(blocks, axis=0).reshape(pts.shape)
    BR = B[..., 0]
    Bphi = B[..., 1]
    BZ = B[..., 2]
    return {
        "R": R,
        "phi": phi,
        "Z": Z,
        "BR": BR,
        "Bphi": Bphi,
        "BZ": BZ,
        "absB": jnp.sqrt(BR * BR + Bphi * Bphi + BZ * BZ),
    }


def _metadata_value(value):
    if isinstance(value, (str, bytes, int, float, np.integer, np.floating)):
        return value
    if isinstance(value, (tuple, list)):
        return ",".join(str(v) for v in value)
    return str(value)


def write_extended_field_netcdf(path, grid_data: Mapping, metadata: Mapping | None = None):
    """Write an extended-field cylindrical grid to a NetCDF file."""
    metadata = {} if metadata is None else dict(metadata)
    arrays = {name: np.asarray(grid_data[name]) for name in ("R", "phi", "Z", "BR", "Bphi", "BZ", "absB")}
    with netcdf_file(path, "w") as nc:
        nc.createDimension("nR", arrays["R"].shape[0])
        nc.createDimension("nphi", arrays["phi"].shape[0])
        nc.createDimension("nZ", arrays["Z"].shape[0])
        for name, dim in (("R", "nR"), ("phi", "nphi"), ("Z", "nZ")):
            var = nc.createVariable(name, "d", (dim,))
            var[:] = arrays[name]
        for name in ("BR", "Bphi", "BZ", "absB"):
            var = nc.createVariable(name, "d", ("nR", "nphi", "nZ"))
            var[:] = arrays[name]
        for key, value in metadata.items():
            setattr(nc, str(key), _metadata_value(value))
    return path


def write_mgrid_like(path, grid_data: Mapping, metadata: Mapping | None = None):
    """Write a lightweight MGRID-like NetCDF file with explicit convention metadata."""
    metadata = {} if metadata is None else dict(metadata)
    metadata.setdefault("format", "mgrid_like")
    metadata.setdefault("coordinate_convention", "(R, phi, Z), physical phi")
    metadata.setdefault("field_components", "BR,Bphi,BZ")
    return write_extended_field_netcdf(path, grid_data, metadata)
