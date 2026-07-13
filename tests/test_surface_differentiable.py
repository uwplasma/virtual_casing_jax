"""Differentiability of the on-surface virtual casing in the SURFACE geometry.

``compute_internal_B`` is differentiable in the source field out of the box; the
extra capability tested here is differentiability in the *surface coordinates*,
which needs (a) the adaptive precision (quad sizes + singular-patch dim) frozen to
static values via :meth:`VirtualCasingJAX.plan_precision` / ``PrecisionPlan`` so
it does not concretize traced surface values, and (b) a NaN-safe Laplace kernel
gradient (the self-interaction ``r -> 0`` term).  This is what lets a downstream
code (e.g. vmec-jax) differentiate a free-boundary objective in the plasma
boundary, not just the coils.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from virtual_casing_jax import PrecisionPlan, VirtualCasingJAX, testdata
from virtual_casing_jax.kernels import _safe_rinv, laplace_fxd_u


# ---------------------------------------------------------------------------
# root cause: the Laplace kernels' self-interaction gradient must be finite
# ---------------------------------------------------------------------------


def test_safe_rinv_value_and_gradient_finite_at_zero():
    r2 = jnp.array([0.0, 1e-40, 1e-20, 1.0, 4.0])
    val = np.asarray(_safe_rinv(r2))
    assert val[0] == 0.0 and val[1] == 0.0          # self-term masked to 0
    assert np.allclose(val[3:], [1.0, 0.5])          # 1/sqrt(r2) elsewhere
    g = np.asarray(jax.grad(lambda r: jnp.sum(_safe_rinv(r)))(r2))
    assert np.all(np.isfinite(g))                    # double-where -> no NaN


def test_laplace_fxd_u_gradient_finite_at_self():
    # first source coincides with the target (dx = 0); its gradient must not NaN
    dx = jnp.array([[0.0, 0.0, 0.0], [0.7, 0.1, -0.2], [1.3, 0.0, 0.4]])
    f = jnp.array([1.0, 1.0, 1.0])
    g = jax.grad(lambda d: jnp.sum(laplace_fxd_u(d, f) ** 2))(dx)
    assert np.all(np.isfinite(np.asarray(g)))


# ---------------------------------------------------------------------------
# plan_precision + PrecisionPlan make compute_internal_B differentiable in X
# ---------------------------------------------------------------------------

_NFP, _HP, _NT, _NP, _DIG = 1, False, 8, 8, 3


def _make_surface():
    X = testdata.surface_coordinates(
        _NFP, _HP, _NT, _NP, surf_type=testdata.SurfType.RotatingEllipseNarrow)
    B0 = X * 0.02 + jnp.array([0.1, -0.02, 0.05])[:, None, None]
    return jnp.asarray(X), jnp.asarray(B0)


def _vc(X):
    vc = VirtualCasingJAX()
    vc.setup(_DIG, _NFP, _HP, _NT, _NP, X, _NT, _NP, _NT, _NP)
    return vc


def test_plan_precision_matches_autoselect_value():
    """precision=plan reproduces the auto-selected precision -> identical B."""
    X, B0 = _make_surface()
    plan = _vc(X).plan_precision(digits=_DIG)
    assert isinstance(plan, PrecisionPlan)
    assert plan.quad_nt > 0 and plan.quad_np > 0 and plan.patch_dim0 > 0
    B_auto = np.asarray(_vc(X).compute_internal_B(B0, digits=_DIG))
    B_plan = np.asarray(_vc(X).compute_internal_B(B0, digits=_DIG, precision=plan))
    assert np.allclose(B_auto, B_plan, rtol=0, atol=1e-12)


def test_compute_internal_B_differentiable_in_surface():
    """grad of a functional of compute_internal_B w.r.t. the surface X is finite.

    The source field B0 is held fixed, so the gradient is purely geometric (the
    quadrature points move with X).  Without the precision plan this raises a
    ConcretizationTypeError; without the NaN-safe kernel it returns all-NaN.
    """
    X, B0 = _make_surface()
    plan = _vc(X).plan_precision(digits=_DIG)

    def loss(Xv):
        return jnp.sum(_vc(Xv).compute_internal_B(B0, digits=_DIG, precision=plan) ** 2)

    g = np.asarray(jax.grad(loss)(X))
    assert np.all(np.isfinite(g))
    assert np.linalg.norm(g) > 0.0

    # finite-difference spot check on one surface coordinate
    i = (0, _NT // 2, _NP // 2)
    h = 1e-5
    Xp = X.at[i].add(h)
    Xm = X.at[i].add(-h)
    fd = (float(loss(Xp)) - float(loss(Xm))) / (2 * h)
    assert abs(g[i] - fd) <= 1e-4 * (abs(fd) + 1.0)
