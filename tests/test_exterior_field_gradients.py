import numpy as np
import jax
import jax.numpy as jnp

from virtual_casing_jax import ExteriorFieldConfig, VirtualCasingExteriorField, VmecSurfaceFieldData


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


def _linear_external_B(xyz):
    xyz = jnp.asarray(xyz)
    return jnp.stack(
        (
            2.0 * xyz[..., 0] + xyz[..., 1],
            -xyz[..., 0] + 3.0 * xyz[..., 2],
            0.5 * xyz[..., 1] - xyz[..., 2],
        ),
        axis=-1,
    )


def _linear_external_gradB(xyz):
    xyz = jnp.asarray(xyz)
    grad = jnp.array([[2.0, 1.0, 0.0], [-1.0, 0.0, 3.0], [0.0, 0.5, -1.0]], dtype=xyz.dtype)
    if xyz.ndim == 1:
        return grad
    return jnp.broadcast_to(grad, xyz.shape[:-1] + (3, 3))


def test_gradB_xyz_adds_external_callback_gradient_for_zero_plasma_field():
    field = VirtualCasingExteriorField(
        _zero_surface_data(),
        ExteriorFieldConfig(digits=3, levels=((13, 13),), chunk_size=64, target_chunk_size=1, dtype="float32"),
        external_B_fn=_linear_external_B,
        external_gradB_fn=_linear_external_gradB,
    )
    point = jnp.array([2.8, 0.1, 0.2], dtype=jnp.float32)

    np.testing.assert_allclose(field.gradB_xyz(point), _linear_external_gradB(point), rtol=1e-6, atol=1e-6)


def test_B_xyz_jvp_matches_finite_difference_for_target_coordinate():
    field = VirtualCasingExteriorField(
        _zero_surface_data(),
        ExteriorFieldConfig(digits=3, levels=((13, 13),), chunk_size=64, target_chunk_size=1, dtype="float32"),
        external_B_fn=_linear_external_B,
    )
    point = jnp.array([2.8, 0.1, 0.2], dtype=jnp.float32)
    tangent = jnp.array([0.2, -0.3, 0.4], dtype=jnp.float32)

    _, jvp = jax.jvp(field.B_xyz, (point,), (tangent,))
    expected = _linear_external_gradB(point) @ tangent

    np.testing.assert_allclose(jvp, expected, rtol=1e-5, atol=1e-5)
