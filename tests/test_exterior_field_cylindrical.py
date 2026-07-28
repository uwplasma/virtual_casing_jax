import numpy as np
import jax.numpy as jnp

from virtual_casing_jax import (
    B_cyl_from_B_xyz,
    ExteriorFieldConfig,
    VirtualCasingExteriorField,
    VmecSurfaceFieldData,
    cyl_to_xyz,
    xyz_vec_to_cyl_vec,
)


def _zero_surface_data(nphi=8, ntheta=6):
    phi = jnp.linspace(0.0, 2.0 * jnp.pi, nphi, endpoint=False)
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, ntheta, endpoint=False)
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
    area = 0.25 * (2.0 + 0.25 * jnp.cos(theta2d)) * normal
    return VmecSurfaceFieldData(
        gamma=gamma,
        B_total=jnp.zeros_like(gamma),
        normal=normal,
        area_vector=area,
        theta=theta,
        phi=phi,
        nfp=1,
        stellsym=False,
        signgs=1,
        source_convention="unit-test",
    )


def _constant_cartesian_B(xyz):
    return jnp.broadcast_to(jnp.array([1.0, 2.0, 3.0], dtype=jnp.asarray(xyz).dtype), jnp.asarray(xyz).shape)


def test_cylindrical_coordinate_conversions_for_aos_points():
    rphiz = jnp.array([[2.0, 0.5 * jnp.pi, 0.25]])
    xyz = cyl_to_xyz(rphiz)
    np.testing.assert_allclose(xyz, jnp.array([[0.0, 2.0, 0.25]]), atol=1e-7)

    Bcyl = xyz_vec_to_cyl_vec(rphiz, jnp.array([[1.0, 2.0, 3.0]]))
    np.testing.assert_allclose(Bcyl, jnp.array([[2.0, -1.0, 3.0]]), atol=1e-7)


def test_B_cyl_from_B_xyz_callback():
    rphiz = jnp.array([[2.0, 0.5 * jnp.pi, 0.25]])
    got = B_cyl_from_B_xyz(_constant_cartesian_B, rphiz)
    np.testing.assert_allclose(got, jnp.array([[2.0, -1.0, 3.0]]), atol=1e-7)


def test_exterior_field_B_cyl_adds_external_callback():
    data = _zero_surface_data()
    cfg = ExteriorFieldConfig(
        digits=3,
        levels=((13, 13),),
        chunk_size=64,
        target_chunk_size=1,
        dtype="float32",
    )
    field = VirtualCasingExteriorField(data, cfg, external_B_fn=_constant_cartesian_B)

    got = field.B_cyl(jnp.array([[2.8, 0.5 * jnp.pi, 0.1]], dtype=jnp.float32))

    np.testing.assert_allclose(got, jnp.array([[2.0, -1.0, 3.0]], dtype=jnp.float32), atol=1e-6)
