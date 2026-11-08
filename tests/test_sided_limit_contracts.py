"""Sided-limit, principal-value and orientation contracts of the on-surface API.

Conventions pinned here (``B0`` is the total field on the surface):

* ``compute_external_B(B0)`` is the trace of the field of the sources outside
  the surface and ``compute_internal_B(B0)`` that of the sources inside, so
  ``external + internal == B0``.
* With ``PV`` the principal value, ``external = PV + B0/2`` and
  ``internal = B0/2 - PV``; the jump between the one-sided limits is ``B0``.

The oracles are closed-form or direct Biot-Savart sums that do not use the
code under test.
"""

import numpy as np
import pytest
import jax.numpy as jnp

from virtual_casing_jax.virtual_casing import VirtualCasingJAX

R0, A = 2.0, 0.3
N, DIGITS = 12, 3


def _torus(nt, npol, nfp=1, reverse_theta=False):
    phi = np.linspace(0.0, 2.0 * np.pi / nfp, nt, endpoint=False)
    theta = np.linspace(0.0, 2.0 * np.pi, npol, endpoint=False)
    if reverse_theta:
        theta = -theta
    t2, p2 = np.meshgrid(theta, phi)
    return np.stack([(R0 + A * np.cos(t2)) * np.cos(p2),
                     (R0 + A * np.cos(t2)) * np.sin(p2),
                     A * np.sin(t2)])


def _dipole(X, m=(0.0, 0.0, 1.0)):
    """Field of a point dipole at the origin, which lies outside the torus."""
    m = np.asarray(m)[:, None, None]
    rr = np.sqrt((X ** 2).sum(0))
    return 3.0 * (m * X).sum(0) * X / rr ** 5 - m / rr ** 3


def _ring(X, n=4000):
    """Biot-Savart field of a unit current ring on the torus core circle."""
    t = (np.arange(n) + 0.5) * 2.0 * np.pi / n
    src = np.stack([R0 * np.cos(t), R0 * np.sin(t), 0.0 * t])
    dl = np.stack([-np.sin(t), np.cos(t), 0.0 * t]) * R0 * 2.0 * np.pi / n
    pts = X.reshape(3, -1)
    d = pts[:, :, None] - src[:, None, :]
    rr = np.sqrt((d ** 2).sum(0))
    B = (np.cross(dl[:, None, :], d, axis=0) / rr ** 3).sum(-1)
    return B.reshape(X.shape)


def _vc(X, nfp=1, nt=N, npol=N):
    vc = VirtualCasingJAX()
    vc.setup(DIGITS, nfp, False, nt, npol, X, nt, npol, nt, npol)
    return vc


def _close(actual, expected, tol=2e-4):
    scale = np.max(np.abs(expected)) + np.max(np.abs(actual))
    assert np.max(np.abs(np.asarray(actual) - expected)) <= tol * scale


@pytest.fixture(scope="module")
def case():
    X = _torus(N, N)
    outside, inside = _dipole(X), _ring(X)
    B0 = outside + inside
    vc = _vc(X)
    return X, outside, inside, B0, vc, vc.compute_external_B(B0), vc.compute_internal_B(B0)


def test_each_side_recovers_its_own_sources(case):
    _, outside, inside, _, _, ext, internal = case
    _close(ext, outside)
    _close(internal, inside)


def test_jump_equals_field_and_sides_sum_to_b0(case):
    _, outside, inside, B0, _, ext, internal = case
    np.testing.assert_allclose(ext + internal, B0, rtol=1e-12, atol=1e-12)
    # PV = (V+ + V-)/2 and the jump V+ - V- = b: the one-sided limits of the
    # single-layer potential are ext = PV + B0/2 and -internal = PV - B0/2.
    _close(ext - internal, outside - inside)


def test_principal_value_is_half_the_signed_difference(case):
    _, outside, inside, B0, vc, ext, internal = case
    pv = vc.compute_principal_value_B(B0)
    np.testing.assert_allclose(pv, 0.5 * (ext - internal), rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(ext, pv + 0.5 * B0, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(internal, 0.5 * B0 - pv, rtol=1e-11, atol=1e-12)
    _close(pv, 0.5 * (outside - inside))


def test_constant_trace_identity():
    """A uniform field has only outside sources: external is b, internal is 0."""
    X = _torus(N, N)
    c = np.array([0.3, -0.2, 0.7])
    B0 = jnp.broadcast_to(c[:, None, None], X.shape)
    vc = _vc(X)
    _close(vc.compute_external_B(B0), np.asarray(B0))
    assert np.max(np.abs(vc.compute_internal_B(B0))) < 1e-4 * np.max(np.abs(c))
    _close(vc.compute_principal_value_B(B0), 0.5 * np.asarray(B0))


def test_swapped_roles_flip_the_sign_of_pv():
    """A field with only inside sources has PV = -B0/2; outside-only gives +B0/2."""
    X = _torus(N, N)
    vc = _vc(X)
    for B0, sgn in ((_dipole(X), 0.5), (_ring(X), -0.5)):
        _close(vc.compute_principal_value_B(B0), sgn * B0)


def test_reversed_parametrisation_keeps_physical_sides(case):
    """The normal is outward whichever way theta runs, so sides do not swap."""
    X = _torus(N, N, reverse_theta=True)
    B0 = _dipole(X) + _ring(X)
    vc = _vc(X)
    ext = np.asarray(vc.compute_external_B(B0))
    internal = np.asarray(vc.compute_internal_B(B0))
    _close(ext, _dipole(X))
    _close(internal, _ring(X))
    # Same physical field on the reversed grid: equal to the forward result with
    # the poloidal index mapped theta -> -theta.
    _, _, _, _, _, ext0, _ = case
    idx = (-np.arange(N)) % N
    np.testing.assert_allclose(ext[:, :, idx], np.asarray(ext0),
                               atol=1e-4 * np.abs(ext).max())


def test_supplied_normals_do_not_choose_the_side():
    """Orientation comes from the geometry; reversed supplied normals change nothing."""
    from virtual_casing_jax import ExteriorFieldConfig, VirtualCasingExteriorField
    from test_exterior_field_branch_sign import _torus_surface_data
    from dataclasses import replace

    data = _torus_surface_data()
    flipped = replace(data, normal=-data.normal, area_vector=-data.area_vector)
    xyz = jnp.array([[2.1, 0.05, 0.02], [2.2, 0.0, 0.1]])
    for branch in ("internal", "external"):
        cfg = ExteriorFieldConfig(branch=branch)
        a = VirtualCasingExteriorField(data, cfg).B_plasma_xyz(xyz)
        b = VirtualCasingExteriorField(flipped, cfg).B_plasma_xyz(xyz)
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14)


def test_field_period_data_matches_full_torus():
    """nfp=2 data on one period reproduces the matching rows of the full torus."""
    nt_full = 16
    Xf = _torus(nt_full, N)
    Bf = _dipole(Xf) + _ring(Xf)
    vf = _vc(Xf, nt=nt_full)
    Xp = _torus(nt_full // 2, N, nfp=2)
    Bp = Bf[:, : nt_full // 2, :]
    vp = _vc(Xp, nfp=2, nt=nt_full // 2)
    for name in ("compute_external_B", "compute_internal_B", "compute_principal_value_B"):
        full = np.asarray(getattr(vf, name)(Bf))[:, : nt_full // 2, :]
        period = np.asarray(getattr(vp, name)(Bp))
        np.testing.assert_allclose(period, full, rtol=1e-6, atol=1e-8 * np.abs(full).max())
