"""The closed-form derivatives are the same field, and its exact derivatives.

Order 0 must reproduce the schedule's own ``B_plasma_xyz`` on the same level,
which fixes the sign and weight conventions; the higher orders are then checked
against nested ``jacfwd`` of that path, which is the thing they replace.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_offsurface_level_choice import _field, _offsets, _rotating_ellipse

from virtual_casing_jax import layer_derivatives
from virtual_casing_jax.derivative_kernels import MAX_DERIVATIVE_ORDER

NFP, NPHI, NTHETA = 3, 16, 16


@pytest.fixture(scope="module")
def setup():
    surface = _rotating_ellipse(NPHI, NTHETA, NFP)
    field = _field(surface, 6)
    return surface, field, _offsets(surface, 0.35, count=6)


def _nested(field, points, order):
    """``order``-th derivative of the schedule path, one nesting per order."""
    def function(point):
        return field.B_plasma_xyz(point[None, :])[0]

    for _ in range(order):
        function = jax.jacfwd(function)
    return np.asarray(jax.jit(jax.vmap(function))(jnp.asarray(points)))


def test_order_zero_reproduces_the_schedule(setup):
    """Fixes the conventions: sheet current sign, quadrature weight, branch."""
    surface, field, points = setup
    single = _field(surface, 6, (field.schedule_levels[-1],))
    closed = np.asarray(single.B_and_derivatives_xyz(points, order=0)[0])
    expected = np.asarray(single.B_plasma_xyz(points))
    np.testing.assert_allclose(closed, expected, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("order", range(1, MAX_DERIVATIVE_ORDER + 1))
def test_closed_forms_match_nested_jacfwd(setup, order):
    surface, field, points = setup
    single = _field(surface, 6, (field.schedule_levels[-1],))
    closed = np.asarray(single.B_and_derivatives_xyz(points, order=order)[order])
    expected = _nested(single, points, order)
    scale = np.max(np.abs(expected))
    assert np.max(np.abs(closed - expected)) <= 1e-9 * scale, (
        order, np.max(np.abs(closed - expected)) / scale)


def test_one_pass_returns_every_lower_order(setup):
    """Asking for order 3 also gives 0, 1 and 2, identical to asking separately."""
    surface, field, points = setup
    single = _field(surface, 6, (field.schedule_levels[-1],))
    together = single.B_and_derivatives_xyz(points, order=3)
    assert len(together) == 4
    for order in range(4):
        alone = single.B_and_derivatives_xyz(points, order=order)[order]
        np.testing.assert_allclose(np.asarray(together[order]), np.asarray(alone),
                                   rtol=1e-12, atol=0.0)


def test_shapes_follow_the_target_layout(setup):
    surface, field, points = setup
    single = _field(surface, 6, (field.schedule_levels[-1],))
    values = single.B_and_derivatives_xyz(points, order=3)
    for order, value in enumerate(values):
        assert value.shape == (len(points),) + (3,) * (order + 1), (order, value.shape)
    one = single.B_and_derivatives_xyz(points[0], order=1)
    assert one[0].shape == (3,) and one[1].shape == (3, 3)


def test_source_chunking_does_not_change_the_answer(setup):
    surface, field, points = setup
    single = _field(surface, 6, (field.schedule_levels[-1],))
    whole = single.B_and_derivatives_xyz(points, order=2, source_chunk=1 << 20)
    split = single.B_and_derivatives_xyz(points, order=2, source_chunk=512)
    for a, b in zip(whole, split):
        a, b = np.asarray(a), np.asarray(b)
        # chunking reassociates the source sum, so agreement is to rounding on
        # the scale of the tensor, not elementwise on its smallest entries
        assert np.max(np.abs(a - b)) <= 1e-12 * np.max(np.abs(a)), (
            np.max(np.abs(a - b)), np.max(np.abs(a)))


def test_an_unsupported_order_is_refused(setup):
    _, field, points = setup
    nodes, current, charge = field.level_sources()
    with pytest.raises(ValueError, match="order must be"):
        layer_derivatives(jnp.asarray(points), nodes, current, charge,
                          order=MAX_DERIVATIVE_ORDER + 1)


def test_the_sheet_current_sign_is_the_one_that_reproduces_the_field(setup):
    """Negating it keeps the magnitude plausible and the answer wrong."""
    surface, field, points = setup
    single = _field(surface, 6, (field.schedule_levels[-1],))
    nodes, current, charge = single.level_sources()
    right = np.asarray(layer_derivatives(jnp.asarray(points), nodes, current,
                                         charge, order=0)[0])
    wrong = np.asarray(layer_derivatives(jnp.asarray(points), nodes, -current,
                                         charge, order=0)[0])
    expected = np.asarray(single.B_plasma_xyz(points))
    assert np.max(np.abs(right - expected)) <= 1e-12 * np.max(np.abs(expected))
    assert np.max(np.abs(wrong - expected)) > 0.1 * np.max(np.abs(expected))
