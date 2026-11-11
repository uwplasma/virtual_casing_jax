"""Closed mirror surface (side wall plus end caps): sources inside and outside against closed-form fields."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_circular_loop_oracle import loop_field

from virtual_casing_jax.mirror import MirrorExteriorField, mirror_panels

MU0 = 4e-7 * np.pi
Z0, Z1 = -1.0, 1.0


def dipole(x, centre, m):
    r = x - jnp.asarray(centre)
    d = jnp.linalg.norm(r)
    return MU0 / (4 * jnp.pi) * (3 * r * (r @ m) / d**5 - m / d**3)


def build(radius, internal, external):
    total = lambda x: internal(x) + external(x)
    pos = lambda z, v: jnp.stack((radius(z, v) * jnp.cos(v), radius(z, v) * jnp.sin(v), z))
    cap = lambda z: lambda w, v: total(jnp.stack((w * radius(z, v) * jnp.cos(v), w * radius(z, v) * jnp.sin(v), z + 0 * v)))
    return mirror_panels(radius, Z0, Z1, lambda z, v: total(pos(z, v)), cap(Z0), cap(Z1))


def loops():
    inner = lambda x: loop_field(x, 0.5, 0.1, 1e6)
    outer = lambda x: loop_field(x, 1.5, 0.8, -2e6) + loop_field(x, 1.4, -0.9, -1.5e6)
    return inner, outer


def scale(truth, pts):
    return float(jnp.max(jnp.linalg.norm(jax.vmap(truth)(pts), axis=-1)))


@pytest.fixture(scope="module")
def cylinder():
    inner, outer = loops()
    return MirrorExteriorField(build(lambda z, v: 1.0 + 0 * z + 0 * v, inner, outer)), inner, outer


def test_flux_is_solenoidal(cylinder):
    assert cylinder[0].flux_defect() < 1e-12


def test_omitted_cap_flux_is_detected():
    inner, outer = loops()
    radius = lambda z, v: 1.0 + 0 * z + 0 * v
    total = lambda x: inner(x) + outer(x)
    pos = lambda z, v: jnp.stack((radius(z, v) * jnp.cos(v), radius(z, v) * jnp.sin(v), z))
    zero_cap = lambda w, v: jnp.stack((total(jnp.stack((w * jnp.cos(v), w * jnp.sin(v), Z0 + 0 * v)))[0], 0 * v, 0 * v)) * 0.0
    panels = mirror_panels(radius, Z0, Z1, lambda z, v: total(pos(z, v)), zero_cap, zero_cap)
    assert MirrorExteriorField(panels).flux_defect() > 1e-2


def test_far_and_near_values_and_gradients(cylinder):
    field, inner, _ = cylinder
    far = jnp.array([[1.3, 0.4, 0.9], [0.0, 2.0, -0.3], [0.0, 0.0, 1.6]])
    near = jnp.array([[1.0 + 1e-3, 0, 0.2], [0.4, 0.2, 1 + 1e-3], [1 + 1e-6, 0, 0.2], [0.0, 0.0, 1 + 1e-3], [1 + 1e-3 / 1.4, 0, 1 + 1e-3 / 1.4]])
    for pts, tol0, tol1 in ((far, 1e-9, 1e-8), (near, 1e-6, 1e-6)):
        (v, g), est = field.B_and_derivatives_xyz(pts, order=1)
        s = scale(inner, far)
        assert np.abs(v - jax.vmap(inner)(pts)).max() / s < tol0
        assert np.abs(g - jax.vmap(jax.jacfwd(inner))(pts)).max() / s < tol1
        assert float(est.max()) < 1e-5


def test_inside_branch_returns_external_sources(cylinder):
    field, inner, outer = cylinder
    pts = jnp.array([[0.3, 0.1, 0.2], [0.7, 0.0, 0.5], [0.0, 0.0, 0.9], [1 - 1e-3, 0, 0.2], [0.5, 0, 1 - 1e-3]])
    (v, g), _ = field.B_and_derivatives_xyz(pts, order=1, branch="external")
    s = scale(inner, jnp.array([[1.3, 0.4, 0.9]]))
    assert np.abs(v - jax.vmap(outer)(pts)).max() / s < 1e-6
    assert np.abs(g - jax.vmap(jax.jacfwd(outer))(pts)).max() / s < 1e-6


def test_non_axisymmetric_wall_and_tilted_sources():
    radius = lambda z, v: 1.0 + 0.15 * jnp.cos(2 * v) + 0.1 * jnp.sin(v - 0.3) * jnp.cos(2 * z) + 0 * z
    inner = lambda x: dipole(x, [0.1, -0.1, 0.2], jnp.array([1e5, 2e5, 3e5])) + loop_field(x, 0.3, 0.0, 1e6)
    outer = lambda x: dipole(x, [2.2, 0.5, 0.4], jnp.array([-4e5, 1e5, 2e5])) + dipole(x, [-0.3, 1.9, -1.8], jnp.array([1e5, 3e5, -2e5]))
    field = MirrorExteriorField(build(radius, inner, outer))
    assert field.flux_defect() < 1e-9
    pts = jnp.array([[1.6, 0.2, 0.3], [0.0, -1.7, 0.1], [0.1, 0.2, 1.7], [1.1 + 5e-3, 0.0, -0.4]])
    s = scale(inner, pts[:3])
    (v, g), _ = field.B_and_derivatives_xyz(pts[:3], order=1)
    assert np.abs(v - jax.vmap(inner)(pts[:3])).max() / s < 1e-7
    assert np.abs(g - jax.vmap(jax.jacfwd(inner))(pts[:3])).max() / s < 1e-6
    inside = jnp.array([[0.2, 0.1, 0.0], [0.5, -0.3, 0.4]])
    (v, g), _ = field.B_and_derivatives_xyz(inside, order=1, branch="external")
    assert np.abs(v - jax.vmap(outer)(inside)).max() / s < 1e-7


def test_autodiff_through_the_sources():
    """The gradient with respect to a source parameter equals the closed-form one (the field outside is the source's)."""
    radius = lambda z, v: 1.0 + 0 * z + 0 * v
    target = jnp.array([[1.5, 0.3, 0.4]])

    def value(rc):
        inner = lambda x: loop_field(x, rc, 0.1, 1e6)
        outer = lambda x: loop_field(x, 1.5, 0.8, -2e6)
        return MirrorExteriorField(build(radius, inner, outer), plain_nodes=(24, 64)).B_plasma_xyz(target)[0]

    exact = lambda rc: loop_field(target[0], rc, 0.1, 1e6)
    ad = jax.jacfwd(value)(0.5)
    assert np.abs(ad - jax.jacfwd(exact)(0.5)).max() < 1e-7 * np.abs(ad).max()


def test_bad_arguments(cylinder):
    with pytest.raises(ValueError):
        cylinder[0].B_and_derivatives_xyz(jnp.zeros((1, 3)), branch="x")
    with pytest.raises(ValueError):
        cylinder[0].B_and_derivatives_xyz(jnp.zeros((1, 3)), order=9)
