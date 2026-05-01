import numpy as np
import jax.numpy as jnp

from virtual_casing_jax import ExteriorFieldConfig, VirtualCasingExteriorField, VmecSurfaceFieldData
from virtual_casing_jax.virtual_casing import VirtualCasingJAX


def _torus_surface_data(nphi=8, ntheta=6, R0=2.0, r=0.25):
    phi = jnp.linspace(0.0, 2.0 * jnp.pi, nphi, endpoint=False)
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, ntheta, endpoint=False)
    theta2d, phi2d = jnp.meshgrid(theta, phi)
    gamma = jnp.stack(
        [
            (R0 + r * jnp.cos(theta2d)) * jnp.cos(phi2d),
            (R0 + r * jnp.cos(theta2d)) * jnp.sin(phi2d),
            r * jnp.sin(theta2d),
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
    area = r * (R0 + r * jnp.cos(theta2d)) * normal
    B_total = 0.05 * gamma + 0.1
    return VmecSurfaceFieldData(
        gamma=gamma,
        B_total=B_total,
        normal=normal,
        area_vector=area,
        theta=theta,
        phi=phi,
        nfp=1,
        stellsym=False,
        signgs=1,
        source_convention="unit-test",
    )


def _field_period_surface_data(nfp=2, nphi=8, ntheta=6, R0=2.0, r=0.25):
    phi = jnp.linspace(0.0, 2.0 * jnp.pi / nfp, nphi, endpoint=False)
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, ntheta, endpoint=False)
    theta2d, phi2d = jnp.meshgrid(theta, phi)
    gamma = jnp.stack(
        [
            (R0 + r * jnp.cos(theta2d)) * jnp.cos(phi2d),
            (R0 + r * jnp.cos(theta2d)) * jnp.sin(phi2d),
            r * jnp.sin(theta2d),
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
    ephi = jnp.stack([-jnp.sin(phi2d), jnp.cos(phi2d), jnp.zeros_like(phi2d)], axis=0)
    area = r * (R0 + r * jnp.cos(theta2d)) * normal
    return VmecSurfaceFieldData(
        gamma=gamma,
        B_total=0.12 * normal + 0.07 * ephi,
        normal=normal,
        area_vector=area,
        theta=theta,
        phi=phi,
        nfp=nfp,
        stellsym=False,
        signgs=1,
        source_convention="unit-test-field-period",
    )


def _rotate_about_z(points, angle):
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    rot = jnp.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=jnp.asarray(points).dtype,
    )
    return jnp.asarray(points) @ rot.T


def test_exterior_field_defaults_to_internal_branch():
    data = _torus_surface_data()
    cfg = ExteriorFieldConfig(
        digits=3,
        levels=((13, 13),),
        chunk_size=64,
        target_chunk_size=2,
        dtype="float32",
    )
    field = VirtualCasingExteriorField(data, cfg)
    targets = jnp.array([[2.8, 0.0, 0.0], [2.7, 0.2, 0.1]], dtype=jnp.float32)

    got = field.B_plasma_xyz(targets)

    vc = VirtualCasingJAX()
    vc.setup(3, 1, False, data.gamma.shape[1], data.gamma.shape[2], data.gamma, data.gamma.shape[1], data.gamma.shape[2], data.gamma.shape[1], data.gamma.shape[2])
    expected = vc.compute_internal_B_offsurf_schedule(
        data.B_total,
        X_trg=targets.T,
        levels=((13, 13),),
        digits=3,
        chunk_size=64,
        target_chunk_size=2,
    ).T

    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_external_branch_is_opposite_offsurface_sign():
    data = _torus_surface_data()
    cfg = ExteriorFieldConfig(
        digits=3,
        levels=((13, 13),),
        chunk_size=64,
        target_chunk_size=2,
        dtype="float32",
    )
    field = VirtualCasingExteriorField(data, cfg)
    targets = jnp.array([[2.8, 0.0, 0.0], [2.7, 0.2, 0.1]], dtype=jnp.float32)

    B_int = field.B_plasma_xyz(targets)
    B_ext = field.B_external_xyz(targets)

    np.testing.assert_allclose(B_int, -B_ext, rtol=1e-6, atol=1e-6)


def test_exterior_field_respects_field_period_rotation_covariance():
    data = _field_period_surface_data()
    cfg = ExteriorFieldConfig(
        digits=3,
        levels=((13, 13),),
        chunk_size=64,
        target_chunk_size=2,
        dtype="float32",
    )
    field = VirtualCasingExteriorField(data, cfg)
    assert field.schedule_levels == ((14, 13),)

    angle = 2.0 * jnp.pi / data.nfp
    targets = jnp.array([[2.8, 0.15, 0.05], [2.65, 0.30, -0.08]], dtype=jnp.float32)
    rotated_targets = _rotate_about_z(targets, angle)

    B = field.B_plasma_xyz(targets)
    B_rotated_targets = field.B_plasma_xyz(rotated_targets)

    np.testing.assert_allclose(
        B_rotated_targets,
        _rotate_about_z(B, angle),
        rtol=2e-6,
        atol=2e-6,
    )
