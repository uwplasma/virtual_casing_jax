from pathlib import Path

import numpy as np
import jax.numpy as jnp
from scipy.io import netcdf_file

from virtual_casing_jax import ExteriorFieldConfig, VirtualCasingExteriorField, VmecSurfaceFieldData
from virtual_casing_jax.grid_export import evaluate_on_rphiz_grid, write_extended_field_netcdf, write_mgrid_like


def _surface_data():
    phi = jnp.linspace(0.0, 2.0 * jnp.pi, 8, endpoint=False)
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, 6, endpoint=False)
    theta2d, phi2d = jnp.meshgrid(theta, phi)
    gamma = jnp.stack(
        [
            (2.0 + 0.25 * jnp.cos(theta2d)) * jnp.cos(phi2d),
            (2.0 + 0.25 * jnp.cos(theta2d)) * jnp.sin(phi2d),
            0.25 * jnp.sin(theta2d),
        ],
        axis=0,
    )
    normal = jnp.stack(
        [
            jnp.cos(theta2d) * jnp.cos(phi2d),
            jnp.cos(theta2d) * jnp.sin(phi2d),
            jnp.sin(theta2d),
        ],
        axis=0,
    )
    return VmecSurfaceFieldData(
        gamma=gamma,
        B_total=jnp.zeros_like(gamma),
        normal=normal,
        area_vector=normal,
        theta=theta,
        phi=phi,
        nfp=1,
        stellsym=False,
        signgs=1,
        source_convention="unit-test",
    )


def _constant_cartesian_B(xyz):
    return jnp.broadcast_to(jnp.array([1.0, 2.0, 3.0], dtype=jnp.asarray(xyz).dtype), jnp.asarray(xyz).shape)


def test_evaluate_on_rphiz_grid_shapes_and_components():
    field = VirtualCasingExteriorField(
        _surface_data(),
        ExteriorFieldConfig(digits=3, levels=((13, 13),), chunk_size=64, target_chunk_size=2, dtype="float32"),
        external_B_fn=_constant_cartesian_B,
    )
    grid = evaluate_on_rphiz_grid(
        field,
        jnp.array([2.7, 2.8], dtype=jnp.float32),
        jnp.array([0.0, 0.5 * jnp.pi], dtype=jnp.float32),
        jnp.array([-0.1, 0.1], dtype=jnp.float32),
        chunk_size=3,
    )

    assert grid["BR"].shape == (2, 2, 2)
    np.testing.assert_allclose(grid["BR"][:, 0, :], 1.0, atol=1e-6)
    np.testing.assert_allclose(grid["Bphi"][:, 0, :], 2.0, atol=1e-6)
    np.testing.assert_allclose(grid["BR"][:, 1, :], 2.0, atol=1e-6)
    np.testing.assert_allclose(grid["Bphi"][:, 1, :], -1.0, atol=1e-6)
    np.testing.assert_allclose(grid["BZ"], 3.0, atol=1e-6)


def test_write_extended_field_netcdf_roundtrip(tmp_path: Path):
    grid = {
        "R": np.array([1.0, 2.0]),
        "phi": np.array([0.0]),
        "Z": np.array([-0.5, 0.5]),
        "BR": np.ones((2, 1, 2)),
        "Bphi": 2.0 * np.ones((2, 1, 2)),
        "BZ": 3.0 * np.ones((2, 1, 2)),
        "absB": np.sqrt(14.0) * np.ones((2, 1, 2)),
    }
    path = tmp_path / "extended.nc"
    write_extended_field_netcdf(path, grid, {"nfp": 1, "coordinate_convention": "(R,phi,Z)"})

    with netcdf_file(path, "r", mmap=False) as nc:
        assert nc.dimensions["nR"] == 2
        np.testing.assert_allclose(nc.variables["BZ"].data.copy(), grid["BZ"])
        assert nc.nfp == 1


def test_write_mgrid_like_adds_format_metadata(tmp_path: Path):
    grid = {
        "R": np.array([1.0]),
        "phi": np.array([0.0]),
        "Z": np.array([0.0]),
        "BR": np.zeros((1, 1, 1)),
        "Bphi": np.zeros((1, 1, 1)),
        "BZ": np.zeros((1, 1, 1)),
        "absB": np.zeros((1, 1, 1)),
    }
    path = tmp_path / "mgrid_like.nc"
    write_mgrid_like(path, grid, {"nfp": 1})
    with netcdf_file(path, "r", mmap=False) as nc:
        assert nc.format == b"mgrid_like"
