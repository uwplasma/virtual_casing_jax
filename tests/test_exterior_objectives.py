import numpy as np
import jax
import jax.numpy as jnp

from virtual_casing_jax import (
    VmecSurfaceFieldData,
    boundary_normal_residual,
    exterior_B_magnitude_squared,
    exterior_Bn_squared,
    grid_divergence_penalty,
)


class ConstantField:
    def B_xyz(self, xyz):
        xyz = jnp.asarray(xyz)
        if xyz.ndim > 1 and xyz.shape[0] == 3 and xyz.shape[-1] != 3:
            base_shape = (3,) + (1,) * (xyz.ndim - 1)
            return jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0], dtype=xyz.dtype).reshape(base_shape), xyz.shape)
        return jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0], dtype=xyz.dtype), xyz.shape)


def test_exterior_Bn_squared_is_smooth_and_differentiable():
    field = ConstantField()
    xyz = jnp.array([[2.0, 0.0, 0.0], [2.1, 0.0, 0.0]])
    normals = jnp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    weights = jnp.array([1.0, 3.0])

    def objective(target):
        return exterior_Bn_squared(field, xyz, normals, weights, target_Bn=target)

    assert np.isclose(objective(0.0), 1.0)
    assert np.isclose(jax.grad(objective)(0.0), -2.0)


def test_exterior_B_magnitude_squared_matches_target_absB():
    field = ConstantField()
    xyz = jnp.array([[2.0, 0.0, 0.0], [2.1, 0.0, 0.0]])
    weights = jnp.ones((2,))

    assert np.isclose(exterior_B_magnitude_squared(field, xyz, weights, target_absB=1.0), 0.0)


def test_grid_divergence_penalty_zero_for_constant_cartesian_z_field():
    R = jnp.array([1.0, 1.5, 2.0])
    phi = jnp.linspace(0.0, 2.0 * jnp.pi, 8, endpoint=False)
    Z = jnp.array([-0.5, 0.0, 0.5])
    BR = jnp.zeros((3, 8, 3))
    Bphi = jnp.zeros((3, 8, 3))
    BZ = jnp.ones((3, 8, 3))

    assert np.isclose(grid_divergence_penalty(BR, Bphi, BZ, R, phi, Z), 0.0)


def test_grid_divergence_penalty_handles_singleton_phi_and_z_axes():
    R = jnp.array([1.0, 1.5, 2.0])
    phi = jnp.array([0.0])
    Z = jnp.array([0.0])
    BR = jnp.zeros((3, 1, 1))
    Bphi = 2.0 * jnp.ones((3, 1, 1))
    BZ = 3.0 * jnp.ones((3, 1, 1))
    weights = jnp.array([[[1.0]], [[2.0]], [[3.0]]])

    assert np.isclose(grid_divergence_penalty(BR, Bphi, BZ, R, phi, Z, weights=weights), 0.0)


def test_boundary_normal_residual_projects_field_on_surface_normals():
    phi = jnp.array([0.0])
    theta = jnp.array([0.0, jnp.pi])
    gamma = jnp.zeros((3, 1, 2))
    normal = jnp.array([[[1.0, -1.0]], [[0.0, 0.0]], [[0.0, 0.0]]])
    data = VmecSurfaceFieldData(
        gamma=gamma,
        B_total=gamma,
        normal=normal,
        area_vector=normal,
        theta=theta,
        phi=phi,
        nfp=1,
        stellsym=False,
        signgs=1,
        source_convention="unit-test",
    )

    got = boundary_normal_residual(data, ConstantField())
    np.testing.assert_allclose(got, jnp.array([[1.0, -1.0]]))
