import numpy as np
import jax.numpy as jnp
import pytest

from virtual_casing_jax import OffsurfaceConvergenceError
from virtual_casing_jax.virtual_casing import VirtualCasingJAX


def _torus(nt, npol, R0=2.0, r=0.3):
    phi = jnp.linspace(0.0, 2.0 * jnp.pi, nt, endpoint=False)
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, npol, endpoint=False)
    theta2d, phi2d = jnp.meshgrid(theta, phi)
    x = (R0 + r * jnp.cos(theta2d)) * jnp.cos(phi2d)
    y = (R0 + r * jnp.cos(theta2d)) * jnp.sin(phi2d)
    z = r * jnp.sin(theta2d)
    return jnp.stack([x, y, z], axis=0)


def _converged_reference(vc, B0, Xt, nt0, np0, digits, gradient=False):
    """A single level far finer than any the schedule visits."""
    levels = ((nt0 * 64, np0 * 64),)
    if gradient:
        return np.asarray(vc.compute_external_gradB_offsurf_schedule(
            B0, X_trg=Xt, levels=levels, digits=digits))
    return np.asarray(vc.compute_external_B_offsurf_schedule(
        B0, X_trg=Xt, levels=levels, digits=digits))


def _relative_error(value, reference):
    value = np.asarray(value)
    return float(np.linalg.norm(value - reference) / np.linalg.norm(reference))


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

    # The two paths no longer agree by construction: the schedule selects its
    # level by the calibrated error estimate, the eager path by the
    # double-layer self-test, which stops one level too early here.  Compare
    # each against a converged reference instead of against the other.
    reference = _converged_reference(vc, B0, Xt, nt0, np0, digits)
    assert _relative_error(b_sched, reference) <= 10.0**-digits
    assert _relative_error(b_sched, reference) < _relative_error(b_py, reference)

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
    reference = _converged_reference(vc, B0, Xt, nt0, np0, digits, gradient=True)
    assert _relative_error(grad_sched, reference) <= _relative_error(grad_py, reference)


def test_offsurface_adaptive_schedule_jit_auto():
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
    max_Nt = nt0 * 8
    max_Np = np0 * 8

    b_py = vc.compute_external_B_offsurf(B0, X_trg=Xt, digits=digits, max_Nt=max_Nt, max_Np=max_Np)
    b_jit = vc.compute_external_B_offsurf_schedule_jit(
        B0,
        X_trg=Xt,
        levels="auto",
        digits=digits,
        max_Nt=max_Nt,
        max_Np=max_Np,
        max_levels=4,
    )
    reference = _converged_reference(vc, B0, Xt, nt0, np0, digits)
    assert _relative_error(b_jit, reference) <= 10.0**-digits
    assert _relative_error(b_jit, reference) < _relative_error(b_py, reference)

    grad_py = vc.compute_external_gradB_offsurf(
        B0,
        X_trg=Xt,
        digits=digits,
        max_Nt=max_Nt,
        max_Np=max_Np,
        adaptive=True,
    )
    grad_jit = vc.compute_external_gradB_offsurf_schedule_jit(
        B0,
        X_trg=Xt,
        levels="auto",
        digits=digits,
        max_Nt=max_Nt,
        max_Np=max_Np,
        max_levels=4,
    )
    reference = _converged_reference(vc, B0, Xt, nt0, np0, digits, gradient=True)
    assert _relative_error(grad_jit, reference) <= _relative_error(grad_py, reference)


def test_eager_adaptive_refinement_has_finite_failure_budget():
    X = _torus(6, 5)
    B0 = X * 0.03 + 0.08
    vc = VirtualCasingJAX()
    vc.setup(5, 1, False, 6, 5, X, 6, 5, 6, 5)

    # A target exactly on the boundary makes the off-surface self-test tend to
    # the jump value 1/2, so it cannot satisfy the requested classification.
    X_src, _, _ = vc._offsurface_densities(B0)
    X_on_surface = X_src[:, :1, :1].reshape(3, 1)
    with pytest.raises(OffsurfaceConvergenceError) as excinfo:
        vc.compute_external_B_offsurf(
            B0,
            X_trg=X_on_surface,
            digits=5,
            max_levels=1,
        )

    error = excinfo.value
    assert error.levels == 1
    assert error.nt > 0 and error.np > 0
    assert error.achieved_error > error.tolerance
    assert "final_grid" in str(error)
