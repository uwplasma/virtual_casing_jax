"""Virtual casing must not depend on the direction of the poloidal parameter."""
import jax
import jax.numpy as jnp

from virtual_casing_jax import VirtualCasingJAX

jax.config.update("jax_enable_x64", True)


def _setup(theta_sign, nt=16, npol=16, nfp=2):
    phi = jnp.linspace(0, 2 * jnp.pi / nfp, nt, endpoint=False)[:, None]
    theta = theta_sign * jnp.linspace(0, 2 * jnp.pi, npol, endpoint=False)[None, :]
    R = 1 + 0.3 * jnp.cos(theta) + 0.1 * jnp.cos(theta - nfp * phi)
    Z = 0.3 * jnp.sin(theta) - 0.1 * jnp.sin(theta - nfp * phi)
    X = jnp.stack([R * jnp.cos(phi), R * jnp.sin(phi), Z])
    B = jnp.stack([-jnp.sin(phi) / R, jnp.cos(phi) / R, 0 * R])  # vacuum: B_external = B
    vc = VirtualCasingJAX()
    vc.setup(6, nfp, False, nt, npol, X, nt, npol, nt, npol)
    return vc, B


def test_left_handed_grid_plans_and_evaluates_like_right_handed():
    (right, B_right), (left, B_left) = _setup(1.0), _setup(-1.0)
    plan_right, plan_left = right.plan_precision(digits=6), left.plan_precision(digits=6)
    # A left-handed grid used to fail the singular self-test and inflate the quadrature.
    assert (plan_left.quad_nt, plan_left.quad_np) == (plan_right.quad_nt, plan_right.quad_np)
    for vc, B in ((right, B_right), (left, B_left)):
        B_ext = vc.compute_external_B(B.reshape(3, -1), digits=6, precision=plan_right)
        assert jnp.max(jnp.abs(B_ext.reshape(B.shape) - B)) < 1e-4
