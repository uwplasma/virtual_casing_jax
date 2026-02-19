import numpy as np
import jax.numpy as jnp

from virtual_casing_jax.virtual_casing import VirtualCasingJAX


def _torus(nt, npol, R0=2.0, r=0.3):
    phi = jnp.linspace(0.0, 2.0 * jnp.pi, nt, endpoint=False)
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, npol, endpoint=False)
    theta2d, phi2d = jnp.meshgrid(theta, phi)
    x = (R0 + r * jnp.cos(theta2d)) * jnp.cos(phi2d)
    y = (R0 + r * jnp.cos(theta2d)) * jnp.sin(phi2d)
    z = r * jnp.sin(theta2d)
    return jnp.stack([x, y, z], axis=0)


def test_offsurface_adaptive_schedule_matches_python():
    nfp = 1
    half_period = False
    surf_nt = 6
    surf_np = 5
    src_nt = 6
    src_np = 5
    trg_nt = 6
    trg_np = 5
    digits = 2

    X = _torus(surf_nt, surf_np)
    B0 = X * 0.03 + 0.08

    vc = VirtualCasingJAX()
    vc.setup(digits, nfp, half_period, surf_nt, surf_np, X, src_nt, src_np, trg_nt, trg_np)

    Xt = jnp.array([[2.1], [0.05], [0.02]])

    X_src, _, _ = vc._offsurface_densities(B0)
    nt0 = int(X_src.shape[1])
    np0 = int(X_src.shape[2])
    levels = ((nt0, np0), (nt0 * 2, np0 * 2), (nt0 * 4, np0 * 4), (nt0 * 8, np0 * 8))

    b_py = vc.compute_external_B_offsurf(B0, X_trg=Xt, digits=digits)
    b_sched = vc.compute_external_B_offsurf_schedule(
        B0,
        X_trg=Xt,
        levels=levels,
        digits=digits,
    )

    np.testing.assert_allclose(b_sched, np.asarray(b_py), rtol=5e-6, atol=5e-8)

    grad_py = vc.compute_external_gradB_offsurf(
        B0,
        X_trg=Xt,
        digits=digits,
        adaptive=True,
    )
    grad_sched = vc.compute_external_gradB_offsurf_schedule(
        B0,
        X_trg=Xt,
        levels=levels,
        digits=digits,
    )
    np.testing.assert_allclose(grad_sched, np.asarray(grad_py), rtol=5e-5, atol=5e-7)
