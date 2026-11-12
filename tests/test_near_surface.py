"""Fields and derivatives arbitrarily close to the surface, on both sides.

The oracle is the one of ``test_offsurface_level_choice``: a rotating-ellipse
boundary (``nfp = 3``) carrying a filament inside and the z-axis current
outside, so the internal branch is exactly the filament field outside the
surface and minus the z-axis field inside it.  The closed forms are wrong by
orders of magnitude within one node spacing; the routed evaluator must not care, and
must hand the targets the closed forms resolve back to them untouched.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_offsurface_level_choice import (
    A_MINOR,
    B0R0,
    CURRENT,
    DELTA,
    MU0,
    R0,
    _field,
    _offsets,
    _ring_field,
    _rotating_ellipse,
    _scale,
    _z_axis_field,
)

from virtual_casing_jax import ExteriorFieldConfig, VirtualCasingExteriorField

NFP, NPHI, NTHETA = 3, 32, 48


def _foot(phi, theta):
    """Surface points and outward unit normals of the rotating ellipse."""
    u = theta - NFP * phi
    R = R0 + A_MINOR * np.cos(theta) + DELTA * np.cos(u)
    Rt, Zt = -A_MINOR * np.sin(theta) - DELTA * np.sin(u), A_MINOR * np.cos(theta) + DELTA * np.cos(u)
    Rp, Zp = NFP * DELTA * np.sin(u), -NFP * DELTA * np.cos(u)
    c, s = np.cos(phi), np.sin(phi)
    point = np.stack([R * c, R * s, A_MINOR * np.sin(theta) + DELTA * np.sin(u)], axis=1)
    normal = np.cross(np.stack([Rt * c, Rt * s, Zt], axis=1),
                      np.stack([Rp * c - R * s, Rp * s + R * c, Zp], axis=1))
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    axis = np.stack([R0 * c, R0 * s, 0 * c], axis=1)
    return point, np.sign(np.sum(normal * (point - axis), axis=1))[:, None] * normal


def _targets(side, distance, shift):
    """Targets above feet ``shift`` node spacings from the nodes (0: on one)."""
    k = np.arange(6)
    point, outward = _foot((2 * np.pi / (NFP * NPHI)) * (3 * k + 1 + shift),
                           (2 * np.pi / NTHETA) * (4 * k + 2 + shift))
    return point + side * distance * A_MINOR * outward


def _oracle(points, side):
    def value(p):
        return _ring_field(p) if side > 0 else -_z_axis_field(p)

    h = 1e-5
    grad = np.stack([(value(points + h * np.eye(3)[j]) - value(points - h * np.eye(3)[j]))
                     / (2 * h) for j in range(3)], axis=-1)
    return value(points), grad


@pytest.fixture(scope="module")
def field():
    return _field(_rotating_ellipse(NPHI, NTHETA, NFP), 6)


def _scale_of(field):
    return _scale(field.surface_data)


def _accurate(field, points, order=1, **kwargs):
    return jax.jit(lambda p: field.B_and_derivatives_accurate_xyz(p, order=order, **kwargs))(
        jnp.asarray(points))


@pytest.mark.parametrize("side", (1, -1))
@pytest.mark.parametrize("shift", (0.0, 0.37))
def test_accurate_at_every_distance_where_closed_forms_are_not(field, side, shift):
    for distance in (0.1, 1e-2, 1e-4):
        points = _targets(side, distance, shift)
        (value, gradient), _ = _accurate(field, points)
        expected, expected_gradient = _oracle(points, side)
        scale = np.max(np.abs(expected))
        error = np.max(np.abs(np.asarray(value) - expected), axis=1) / scale
        assert error.max() <= 1e-8, distance
        assert A_MINOR * np.max(np.abs(np.asarray(gradient) - expected_gradient)) <= 1e-5 * scale
    closed = np.asarray(field.B_and_derivatives_xyz(points, order=0)[0])
    assert np.max(np.abs(closed - expected)) > 0.1 * scale  # unlike the closed forms


def test_far_targets_keep_the_closed_forms_and_coarse_data_is_flagged(field):
    points = _targets(1, 3.0, 0.37)
    values, estimate = _accurate(field, points)
    for got, closed in zip(values, field.B_and_derivatives_xyz(points, order=1)):
        np.testing.assert_allclose(got, closed, rtol=1e-12, atol=0)
    assert np.max(estimate) <= 1e-8
    near = _targets(1, 1e-3, 0.37)
    (value,), estimate = _accurate(field, near, 0)
    expected = _oracle(near, 1)[0]  # the estimate bounds the error
    assert np.all(estimate >= np.max(np.abs(value - expected), axis=1) / np.max(np.abs(expected)))
    coarse = _field(_rotating_ellipse(8, 12, NFP), 6)
    assert np.min(coarse.B_and_derivatives_accurate_xyz(near, order=0)[1]) \
        > 10.0 * np.max(estimate)


def test_derivatives_are_derivatives_of_the_returned_field_and_nodes_are_finite(field):
    points = jnp.asarray(_targets(-1, 1e-2, 0.37))
    (_, gradient), _ = _accurate(field, points)
    direction = jnp.asarray(np.random.default_rng(1).standard_normal(points.shape))

    def B(p):
        return field.B_and_derivatives_accurate_xyz(p, order=0)[0][0]

    tangent = jnp.einsum("tij,tj->ti", gradient, direction)
    np.testing.assert_allclose(jax.jvp(B, (points,), (direction,))[1], tangent, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(jax.vjp(B, points)[1](direction)[0],
                               jnp.einsum("tij,ti->tj", gradient, direction), rtol=1e-8, atol=1e-9)
    on_node = jnp.asarray(_targets(1, 0.0, 0.0))
    (node_value, node_gradient), _ = _accurate(field, on_node)
    assert all(np.all(np.isfinite(a)) for a in (node_value, node_gradient, jax.jacfwd(B)(on_node)))
    outside = B(jnp.asarray(_targets(1, 1e-6, 0.0)))
    assert np.max(np.abs(node_value - outside)) <= 1e-5 * np.max(np.abs(outside))


def test_branches_coefficient_gradient_and_single_level():
    """``external`` is minus ``internal``; the field is linear in the surface data."""
    surface = _rotating_ellipse(NPHI, NTHETA, NFP)
    ring = jnp.asarray(_ring_field(np.asarray(surface.gamma).reshape(3, -1).T).T.reshape(surface.gamma.shape))
    axis = surface.B_total - ring
    points = jnp.asarray(_targets(1, 1e-3, 0.37))

    def near(scale, branch="internal"):
        return _field(replace(surface, B_total=scale * ring + axis), 6) \
            .B_and_derivatives_accurate_xyz(points, order=0, branch=branch)[0][0]

    expected = _ring_field(np.asarray(points))
    assert np.max(np.abs(np.asarray(jax.jacfwd(near)(1.0)) - expected)) <= 1e-8 * np.max(np.abs(expected))
    np.testing.assert_allclose(near(1.0, "external"), -near(1.0), rtol=1e-12, atol=1e-14)
    with pytest.raises(ValueError, match="branch"):
        near(1.0, "outside")
    one_level = _field(surface, 6, levels=((NFP * NPHI, NTHETA),))
    with pytest.raises(ValueError, match="two levels"):
        one_level.B_and_derivatives_accurate_xyz(points)


@pytest.mark.parametrize("side", (1, -1))
def test_the_default_field_methods_are_accurate_next_to_the_surface(side):
    surface = _rotating_ellipse(NPHI, NTHETA, NFP)
    config = ExteriorFieldConfig(digits=6)
    default = VirtualCasingExteriorField(surface, config)
    legacy = VirtualCasingExteriorField(surface, replace(config, near_surface=False))
    points = _targets(side, 1e-3, 0.37)
    expected, expected_gradient = _oracle(points, side)
    scale = np.max(np.abs(expected))
    value, estimate = default.B_plasma_xyz(points, return_estimate=True)
    assert np.max(np.abs(value - expected)) <= 1e-8 * scale and np.max(estimate) < 1e-6
    assert A_MINOR * np.max(np.abs(default.gradB_plasma_xyz(points) - expected_gradient)) <= 1e-5 * scale
    assert np.max(np.abs(legacy.B_plasma_xyz(points) - expected)) > 0.1 * scale
    far = _targets(1, 3.0, 0.37)  # transposed layout and the closed forms agree
    np.testing.assert_allclose(default.B_plasma_xyz(far.T), legacy.B_plasma_xyz(far.T), rtol=1e-6)
    np.testing.assert_allclose(default.gradB_plasma_xyz(far), legacy.gradB_plasma_xyz(far), rtol=1e-6, atol=1e-9)


def test_the_default_never_does_worse_than_the_closed_forms_on_a_thin_torus(monkeypatch):
    """Where the data are too coarse for the near sum, a target stays on the closed forms.

    On a torus of aspect ratio 15 with coarse data the near sum is limited by the
    field data at about 1e-4, while the closed forms are already far better a
    minor radius or so out; routed targets must keep the better of the two.
    """
    import test_offsurface_level_choice as oracle

    monkeypatch.setattr(oracle, "A_MINOR", 0.065)
    monkeypatch.setattr(oracle, "DELTA", 0.015)
    surface = oracle._rotating_ellipse(NPHI, NTHETA, NFP)
    config = ExteriorFieldConfig(digits=8)
    default = VirtualCasingExteriorField(surface, config)
    legacy = VirtualCasingExteriorField(surface, replace(config, near_surface=False))
    for distance in (1.0, 0.7, 0.3, 0.1, 1e-2):
        points = oracle._offsets(surface, distance * oracle.A_MINOR)
        expected = oracle._ring_field(points)
        scale = np.max(np.abs(expected))
        error = lambda field: np.max(np.abs(np.asarray(field.B_plasma_xyz(points)) - expected)) / scale  # noqa: E731
        assert error(default) <= 1.5 * error(legacy) + 1e-12, distance
    far = oracle._offsets(surface, 3.0 * oracle.A_MINOR)
    np.testing.assert_allclose(default.B_plasma_xyz(far), legacy.B_plasma_xyz(far), rtol=1e-6, atol=1e-6)


def test_the_graded_rule_reproduces_the_closed_forms_where_both_resolve(field):
    """Forcing every target through the graded rule checks its signs and kernels to order 3."""
    points = _targets(1, 0.5, 0.37)
    closed = field.B_and_derivatives_xyz(points, order=3)
    explicit = VirtualCasingExteriorField(
        field.surface_data, replace(field.config, graded_nodes=(96, 192)))
    for source in (field, explicit):
        graded, estimate = source.B_and_derivatives_accurate_xyz(points, order=3, tol=-1.0)
        assert np.max(estimate) < 1e-7
        for order, (got, expected) in enumerate(zip(graded, closed)):
            assert np.max(np.abs(got - expected)) <= 1e-6 * 10.0**order * np.max(np.abs(expected))


def test_routing_by_the_field_alone_does_not_depend_on_the_order_asked_for(field):
    """With ``route_order=0`` the jets of every order come from the routing of ``B``."""
    points = _targets(1, 0.5, 0.37)
    (value,), _ = field.B_and_derivatives_accurate_xyz(points, order=0)
    (routed, *_), _ = field.B_and_derivatives_accurate_xyz(points, order=3, route_order=0)
    np.testing.assert_allclose(routed, value, rtol=1e-12, atol=0)


def test_coarse_data_without_axisymmetry_keeps_the_toroidal_modes_it_resolves():
    """One period of data (nfp = 1) from sources of no symmetry: its spectrum is not centred on mode 0."""
    angle = 2.0 * np.pi * np.arange(512) / 512
    plane = np.stack([np.cos(angle), np.cos(0.6) * np.sin(angle), np.sin(0.6) * np.sin(angle)], axis=1)
    source, tangent = np.array([1.9, 0.5, 0.4]) + 0.25 * plane, np.stack(
        [-np.sin(angle), np.cos(0.6) * np.cos(angle), np.sin(0.6) * np.cos(angle)], axis=1)

    def loop(points):  # an off-axis loop outside the surface
        separation = points[:, None, :] - source[None]
        return 0.125 * np.mean(np.cross(tangent[None], separation) / np.linalg.norm(
            separation, axis=-1, keepdims=True) ** 3, axis=1) * MU0 * CURRENT

    surface = _rotating_ellipse(24, 36, 1)
    nodes = np.asarray(surface.gamma).reshape(3, -1).T
    B_total = jnp.asarray((_ring_field(nodes) + loop(nodes)).T.reshape(surface.gamma.shape))
    coarse = _field(replace(surface, B_total=B_total), 8)
    points = _offsets(surface, 0.1 * A_MINOR)
    error = np.max(np.abs(_accurate(coarse, points, 0)[0][0] - _ring_field(points))) / np.max(np.abs(_ring_field(points)))
    assert error <= 4e-4  # 5.3e-4 with the cylindrical window


def _oracle_jets(points, side, order=3):
    """Jets to ``order`` of the filament (outside) or minus the z-axis (inside) field.

    Automatic derivatives of the analytic formulas, which share nothing with the
    layer sums under test.
    """
    angle = 2.0 * jnp.pi * jnp.arange(2048) / 2048
    source = jnp.stack([R0 * jnp.cos(angle), R0 * jnp.sin(angle), 0.0 * angle], axis=1)
    tangent = jnp.stack([-jnp.sin(angle), jnp.cos(angle), 0.0 * angle], axis=1)

    def ring(x):
        r = x - source
        kernel = jnp.cross(tangent, r) / jnp.linalg.norm(r, axis=1, keepdims=True) ** 3
        return MU0 * CURRENT * R0 / (2.0 * len(angle)) * jnp.sum(kernel, axis=0)

    def axis(x):
        return -B0R0 * jnp.array([-x[1], x[0], 0.0]) / (x[0] ** 2 + x[1] ** 2)

    function, jets = (ring if side > 0 else axis), []
    for _ in range(order + 1):
        jets.append(np.asarray(jax.jit(jax.vmap(function))(jnp.asarray(points))))
        function = jax.jacfwd(function)
    return jets


@pytest.fixture(scope="module")
def routed_jets(field):
    return jax.jit(lambda p: field.B_and_derivatives_accurate_xyz(p, order=3)[0])


def _jet_errors(routed_jets, field_scale, points, side):
    """Scaled ``L**k max|D**k B - ref| / B*`` for k = 0..3."""
    got = routed_jets(jnp.asarray(points))
    return [A_MINOR**k * np.max(np.abs(np.asarray(g) - r)) / field_scale
            for k, (g, r) in enumerate(zip(got, _oracle_jets(points, side)))]


@pytest.mark.parametrize("side, distance", ((1, 3.0), (1, 0.5), (1, 1e-2), (-1, 0.5), (-1, 1e-2)))
def test_jets_of_orders_two_and_three_match_the_filament_fields(field, routed_jets, side, distance):
    """The closed forms (far) and the graded rule (near), on both sides, to third order."""
    scale = _scale_of(field)
    errors = _jet_errors(routed_jets, scale, _targets(side, distance, 0.37), side)
    for order, (error, gate) in enumerate(zip(errors, (1e-8, 1e-7, 1e-6, 1e-5))):
        assert error <= gate, (order, error)


@pytest.mark.parametrize("side", (1, -1))
def test_jets_within_the_floor_are_continued_and_the_estimate_covers_them(field, side):
    """Closer than ``_FLOOR`` the jets are Taylor-continued from the floor, the fourth derivative from a second height.

    Orders 0 and 1 keep their gates, order 2 is good to ``3e-5`` and order 3 to a few ``1e-3``; the
    estimate bounds every order, scaled by ``100**-k``.
    """
    points = _targets(side, 1e-4, 0.37)
    jets, estimate = jax.jit(lambda p: field.B_and_derivatives_accurate_xyz(p, order=3))(jnp.asarray(points))
    scale = _scale_of(field)
    errors = [A_MINOR**k * np.max(np.abs(np.asarray(g) - r)) / scale
              for k, (g, r) in enumerate(zip(jets, _oracle_jets(points, side)))]
    assert all(error <= gate for error, gate in zip(errors, (1e-8, 1e-6, 3e-5, 3e-3))), errors
    assert all(error / 100.0**k <= np.max(estimate) for k, error in enumerate(errors)), (errors, estimate)


@pytest.mark.parametrize("side", (1, -1))
@pytest.mark.parametrize("kind", ("constant", "external"))
def test_estimate_bounds_the_error_within_the_floor_where_the_jets_are_known_exactly(kind, side):
    """A constant trace and an external-only trace have exact jets (zero or minus the trace's field).

    The estimate, scaled by ``100**k``, must bound the error of every order at 1e-4 of the minor radius,
    on data the rule resolves only to a few digits.
    """
    constant = np.array([0.3, -0.2, 0.5])
    surface = _rotating_ellipse(24, 24, 1 if kind == "constant" else 3)
    points = np.asarray(surface.gamma).reshape(3, -1).T
    trace = constant[None] + 0 * points if kind == "constant" else _z_axis_field(points)
    field = _field(replace(surface, B_total=jnp.asarray(trace.T.reshape(surface.gamma.shape))), 6)
    if kind == "constant":
        targets = _offsets(surface, side * 1e-4 * A_MINOR, 6)
        exact = [-constant * (side < 0) * np.ones((len(targets), 3))] + [np.zeros((len(targets),) + (3,) * k) for k in (2, 3, 4)]
    else:
        targets = _targets(side, 1e-4, 0.37)
        exact = _oracle_jets(targets, side) if side < 0 else [np.zeros_like(j) for j in _oracle_jets(targets, side)]
    jets, estimate = jax.jit(lambda p: field.B_and_derivatives_accurate_xyz(p, order=3))(jnp.asarray(targets))
    scale = _scale_of(field)
    for k, (got, want) in enumerate(zip(jets, exact)):
        assert A_MINOR**k * np.max(np.abs(np.asarray(got) - want)) / scale / 100.0**k <= np.max(estimate), (k, estimate)
