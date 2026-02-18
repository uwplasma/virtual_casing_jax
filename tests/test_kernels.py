import numpy as np
import jax.numpy as jnp

from virtual_casing_jax.kernels import (
    laplace_fx_u,
    laplace_fxd_u,
    laplace_fxd2_u,
    biotsavart_fx_u,
    biotsavart_fxd_u,
)

FOUR_PI = 4.0 * np.pi


def _safe_rinv(r2, eps=1e-30):
    return np.where(r2 > eps, 1.0 / np.sqrt(r2), 0.0)


def test_laplace_fx_u():
    rng = np.random.default_rng(0)
    dx = rng.normal(size=(5, 3))
    f = rng.normal(size=(5,))
    r2 = np.sum(dx * dx, axis=-1)
    rinv = _safe_rinv(r2)
    ref = f * rinv / FOUR_PI

    out = np.asarray(laplace_fx_u(jnp.asarray(dx), jnp.asarray(f)))
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)


def test_laplace_fxd_u():
    rng = np.random.default_rng(1)
    dx = rng.normal(size=(7, 3))
    f = rng.normal(size=(7,))
    r2 = np.sum(dx * dx, axis=-1)
    rinv = _safe_rinv(r2)
    rinv3 = rinv * rinv * rinv
    ref = -(dx * f[:, None]) * rinv3[:, None] / FOUR_PI

    out = np.asarray(laplace_fxd_u(jnp.asarray(dx), jnp.asarray(f)))
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)


def test_laplace_fxd2_u():
    rng = np.random.default_rng(2)
    dx = rng.normal(size=(4, 3))
    f = rng.normal(size=(4,))
    r2 = np.sum(dx * dx, axis=-1)
    rinv = _safe_rinv(r2)
    rinv2 = rinv * rinv
    rinv3 = rinv * rinv2
    rinv5 = rinv3 * rinv2
    eye = np.eye(3)

    ref = np.zeros((4, 3, 3))
    for i in range(4):
        r_outer = np.outer(dx[i], dx[i])
        ref[i] = (-eye * rinv3[i] + 3.0 * r_outer * rinv5[i]) * f[i] / FOUR_PI

    out = np.asarray(laplace_fxd2_u(jnp.asarray(dx), jnp.asarray(f)))
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)


def test_biotsavart_fx_u():
    rng = np.random.default_rng(3)
    dx = rng.normal(size=(6, 3))
    f = rng.normal(size=(6, 3))
    r2 = np.sum(dx * dx, axis=-1)
    rinv = _safe_rinv(r2)
    rinv3 = rinv * rinv * rinv
    ref = np.cross(f, dx) * rinv3[:, None] / FOUR_PI

    out = np.asarray(biotsavart_fx_u(jnp.asarray(dx), jnp.asarray(f)))
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)


def test_biotsavart_fxd_u_against_formula():
    rng = np.random.default_rng(4)
    dx = rng.normal(size=(3, 3))
    f = rng.normal(size=(3, 3))

    out = np.asarray(biotsavart_fxd_u(jnp.asarray(dx), jnp.asarray(f)))

    # Manual formula from BIEST for each sample
    ref = np.zeros_like(out)
    for i in range(dx.shape[0]):
        x, y, z = dx[i]
        r2 = x * x + y * y + z * z
        rinv = _safe_rinv(r2)
        rinv2 = rinv * rinv
        rinv3 = rinv * rinv2
        rinv5 = rinv3 * rinv2

        u = np.zeros((3, 9))
        u[0, 0] = 0
        u[1, 0] = 3 * z * x * rinv5
        u[2, 0] = -3 * y * x * rinv5

        u[0, 1] = 0
        u[1, 1] = 3 * z * y * rinv5
        u[2, 1] = rinv3 - 3 * y * y * rinv5

        u[0, 2] = 0
        u[1, 2] = -rinv3 + 3 * z * z * rinv5
        u[2, 2] = -3 * y * z * rinv5

        u[0, 3] = -3 * z * x * rinv5
        u[1, 3] = 0
        u[2, 3] = -rinv3 + 3 * x * x * rinv5

        u[0, 4] = -3 * z * y * rinv5
        u[1, 4] = 0
        u[2, 4] = 3 * x * y * rinv5

        u[0, 5] = rinv3 - 3 * z * z * rinv5
        u[1, 5] = 0
        u[2, 5] = 3 * x * z * rinv5

        u[0, 6] = 3 * y * x * rinv5
        u[1, 6] = rinv3 - 3 * x * x * rinv5
        u[2, 6] = 0

        u[0, 7] = -rinv3 + 3 * y * y * rinv5
        u[1, 7] = -3 * x * y * rinv5
        u[2, 7] = 0

        u[0, 8] = 3 * y * z * rinv5
        u[1, 8] = -3 * x * z * rinv5
        u[2, 8] = 0

        fx, fy, fz = f[i]
        v = -(u[0] * fx + u[1] * fy + u[2] * fz) / FOUR_PI
        ref[i] = v.reshape((3, 3))

    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)
