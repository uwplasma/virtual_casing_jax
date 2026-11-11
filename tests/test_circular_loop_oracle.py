"""Axisymmetric trace against closed-form circular-loop fields (complete elliptic integrals by AGM).

A plasma loop (internal) plus a poloidal-field loop and a toroidal-field term (external) give the surface
trace of a torus; the exterior and interior branches must return the loop fields, including on the axis,
for the field, its gradient, and the derivatives with respect to the loop parameters.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from virtual_casing_jax import (
    ExteriorFieldConfig,
    VirtualCasingExteriorField,
    VmecSurfaceFieldData,
)

MU0 = 4e-7 * np.pi
N = 48


def _ellipk_e(m):
    a, b, c, s, p = jnp.ones_like(m), jnp.sqrt(1.0 - m), jnp.sqrt(m), 0.5 * m, 0.5
    for _ in range(14):
        a, b, c = 0.5 * (a + b), jnp.sqrt(a * b), 0.5 * (a - b)
        p *= 2.0
        s = s + p * c**2
    K = jnp.pi / (2.0 * a)
    return K, K * (1.0 - s)


def loop_field(xyz, Rc, Zc, I):
    """Cartesian field of a loop of radius Rc at height Zc; smooth on the axis (series for R < 1e-6 Rc)."""
    R2, z = xyz[..., 0] ** 2 + xyz[..., 1] ** 2, xyz[..., 2] - Zc
    ax = R2 < (1e-6 * Rc) ** 2
    R = jnp.sqrt(jnp.where(ax, 1.0, R2))
    d2 = (R + Rc) ** 2 + z**2
    K, E = _ellipk_e(4.0 * R * Rc / d2)
    pref, den = MU0 * I / (2.0 * jnp.pi * jnp.sqrt(d2)), (Rc - R) ** 2 + z**2
    Bz = pref * (K + (Rc**2 - R**2 - z**2) / den * E)
    g = pref * z / R**2 * (-K + (Rc**2 + R**2 + z**2) / den * E)
    g = jnp.where(ax, 0.75 * MU0 * I * Rc**2 * z / (Rc**2 + z**2) ** 2.5, g)
    Bz = jnp.where(ax, 0.5 * MU0 * I * Rc**2 / (Rc**2 + z**2) ** 1.5, Bz)
    return jnp.stack([g * xyz[..., 0], g * xyz[..., 1], Bz], -1)


def torus_data(B, R0=3.0, a=1.0):
    th = jnp.linspace(0, 2 * jnp.pi, N, endpoint=False)
    T, P = jnp.meshgrid(th, th)
    gamma = jnp.stack(
        [
            (R0 + a * jnp.cos(T)) * jnp.cos(P),
            (R0 + a * jnp.cos(T)) * jnp.sin(P),
            a * jnp.sin(T),
        ]
    )
    n = jnp.stack([jnp.cos(T) * jnp.cos(P), jnp.cos(T) * jnp.sin(P), jnp.sin(T)])
    return VmecSurfaceFieldData(
        gamma=gamma,
        B_total=jnp.moveaxis(B(jnp.moveaxis(gamma, 0, -1)), -1, 0),
        normal=n,
        area_vector=a * (R0 + a * jnp.cos(T)) * n,
        theta=th,
        phi=th,
        nfp=1,
        stellsym=False,
        signgs=1,
    )


PLASMA, PF, I_TF = (2.6, 0.0, 1.0e6), (5.0, 1.5, -2.0e6), 3.0e6
OUT = jnp.array(
    [
        [5.0, 0, 0.3],
        [0, 4.2, -1.0],
        [2.0, 2.0, 2.0],
        [0, 0, 1.7],
        [0, 0, -2.5],
        [3.0, 0, 1.05],
    ]
)
IN = jnp.array([[3.2, 0, 0.1], [0, 2.5, -0.3], [-3.0, 0.4, 0.5]])


def b_plasma(x):
    return loop_field(x, *PLASMA)


def b_external(x, pf=PF):
    tf = (
        MU0
        * I_TF
        / (2 * jnp.pi)
        * jnp.stack([-x[..., 1], x[..., 0], 0 * x[..., 0]], -1)
        / (x[..., 0] ** 2 + x[..., 1] ** 2)[..., None]
    )
    return loop_field(x, *pf) + tf


def make(B):
    return VirtualCasingExteriorField(
        torus_data(B), ExteriorFieldConfig()
    )


@pytest.fixture(scope="module")
def field():
    return make(lambda x: b_plasma(x) + b_external(x))


def test_loop_oracle_matches_scipy():
    scipy_special = pytest.importorskip("scipy.special")
    m = np.array([1e-9, 0.1, 0.5, 0.9, 0.999999])
    K, E = _ellipk_e(jnp.array(m))
    assert (
        np.max(np.abs(K - scipy_special.ellipk(m))) < 1e-14
        and np.max(np.abs(E - scipy_special.ellipe(m))) < 1e-14
    )


def test_branches_return_loop_fields_including_axis(field):
    scale = float(jnp.max(jnp.linalg.norm(b_plasma(OUT), axis=1)))
    assert (
        float(jnp.max(jnp.abs(field.B_plasma_xyz(OUT) - b_plasma(OUT)))) < 1e-8 * scale
    )
    assert (
        float(
            jnp.max(jnp.abs(field.B_plasma_xyz(IN, branch="external") - b_external(IN)))
        )
        < 1e-8 * scale
    )
    grad = jax.vmap(jax.jacfwd(b_plasma))(OUT)
    assert float(jnp.max(jnp.abs(field.gradB_plasma_xyz(OUT) - grad))) < 1e-6 * scale


def test_loop_parameter_derivatives_through_the_pipeline():
    def inside(p):
        return make(lambda x: b_plasma(x) + b_external(x, p)).B_plasma_xyz(
            IN, branch="external"
        )

    p0 = jnp.array(PF)
    got, want = jax.jacfwd(inside)(p0), jax.jacfwd(lambda p: b_external(IN, p))(p0)
    assert (
        float(jnp.max(jnp.abs(got - want) / jnp.max(jnp.abs(want), axis=(0, 1)))) < 1e-6
    )
