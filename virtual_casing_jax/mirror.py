"""Field of the sources inside (or outside) a closed mirror surface: a side wall closed by end caps.

The virtual-casing layer sums of :mod:`~virtual_casing_jax.derivative_kernels` are taken over three
patches ``X(u, v)``, ``u`` in ``[0, 1]`` and ``v`` periodic, with the outward normal along
``X_u x X_v``.  Each patch is integrated by Gauss-Legendre in ``u`` and the trapezoid rule in ``v``;
the rims where patches meet are patch boundaries, so no integrand has a corner inside a patch,
and an axis (a cap centre, where ``X`` does not depend on ``v``) is a regular point of the weight
``|X_u x X_v| -> 0``.  Targets within a few node spacings of a patch use, as the torus path does, a rule
graded about their nearest surface point -- here geometric Gauss-Legendre panels in ``u`` and the
``v - a sin v`` substitution in ``v`` -- and targets within ``_FLOOR`` of the surface are continued
along the normal from that distance with the closed-form jets.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .derivative_kernels import MAX_DERIVATIVE_ORDER, layer_derivatives
from .exterior_field import _GRADING, _MAX_GRADING, _continue, _nearest_surface_point

__all__ = ["MirrorExteriorField", "MirrorPanel", "mirror_panels"]

#: Geometric panels per side of the grading centre; with ratio 2 they span ``2**_LEVELS`` widths.
_LEVELS = 12
#: A target farther than this many plain-rule node spacings from every patch uses the plain rule.
_FAR = 4.0
#: Targets closer than this fraction of the half-extent are continued from it (less than the torus path's
#: ``1e-3``: the rule's error grows as the cube of the inverse distance, the continuation's as the cube of the distance).
_FLOOR = 5.0e-4
_TWO_PI = 2.0 * np.pi


class MirrorPanel(NamedTuple):
    """One patch of the closed surface.

    ``position(u, v)`` and ``field(u, v)`` take scalar ``u`` in ``[0, 1]`` and ``v`` in ``[0, 2 pi)`` and
    return Cartesian ``(3,)`` arrays: the point and the total field ``B`` on the surface.  Both are
    ``2 pi``-periodic in ``v``, and ``X_u x X_v`` points out of the enclosed volume.  ``B`` may differ
    between patches along a rim.
    """

    position: Callable
    field: Callable


def mirror_panels(radius, z_min, z_max, side_field, lower_field, upper_field):
    """Side wall ``r = radius(z, v)`` for ``z_min <= z <= z_max`` and a flat cap at each end.

    A cap is the star-shaped disk ``r <= radius(z_end, v)``.  ``side_field(z, v)`` is the field at the
    wall, and ``lower_field(w, v)``, ``upper_field(w, v)`` the field on the caps at the fraction
    ``w = r / radius(z_end, v)`` of the rim radius.  The caps carry the through-flux ``B . n``.
    """
    def side(u, v):
        z = z_max - (z_max - z_min) * u
        return jnp.stack((radius(z, v) * jnp.cos(v), radius(z, v) * jnp.sin(v), z))

    def cap(z, w, v):
        r = w * radius(z, v)
        return jnp.stack((r * jnp.cos(v), r * jnp.sin(v), z + 0.0 * v))

    return (
        MirrorPanel(lambda u, v: cap(z_min, 1.0 - u, v), lambda u, v: lower_field(1.0 - u, v)),
        MirrorPanel(side, lambda u, v: side_field(z_max - (z_max - z_min) * u, v)),
        MirrorPanel(lambda u, v: cap(z_max, u, v), upper_field),
    )


def _tangents(position, u, v):
    one, zero = jnp.ones_like(u), jnp.zeros_like(u)
    x, x_u = jax.jvp(position, (u, v), (one, zero))
    return x, x_u, jax.jvp(position, (u, v), (zero, one))[1]


def _sources(panel, u, v, weight):
    """Nodes, weighted sheet currents ``n x B dA`` and charges ``n . B dA`` at the flat nodes ``(u, v)``."""
    x, x_u, x_v = jax.vmap(lambda a, b: _tangents(panel.position, a, b))(u, v)
    area = jnp.cross(x_u, x_v) * weight[:, None]
    field = jax.vmap(panel.field)(u, v)
    return x, jnp.cross(area, field), jnp.sum(area * field, axis=-1)


def _tensor(u, w_u, v, w_v):
    return (jnp.repeat(u, v.size), jnp.tile(v, u.size), jnp.repeat(w_u, v.size) * jnp.tile(w_v, u.size))


def _plain(panel, nodes):
    q, n = nodes
    s, w = np.polynomial.legendre.leggauss(q)
    return _sources(panel, *_tensor((s + 1) / 2, w / 2, _TWO_PI * np.arange(n) / n, np.full(n, _TWO_PI / n)))


def _graded_u(centre, width, q):
    """Gauss-Legendre panels on ``[0, 1]`` whose widths double away from ``centre``, the first being ``width``."""
    s, w = np.polynomial.legendre.leggauss(q)

    def side(room):
        edge = jnp.concatenate((jnp.zeros(1), jnp.minimum(width * 2.0 ** jnp.arange(_LEVELS), room).at[-1].set(room)))
        span = jnp.diff(edge)[:, None]
        return (edge[:-1, None] + span * (s + 1) / 2).ravel(), (span * w / 2).ravel()

    (d_right, w_right), (d_left, w_left) = side(1.0 - centre), side(centre)
    return jnp.concatenate((centre + d_right, centre - d_left)), jnp.concatenate((w_right, w_left))


def _graded(panel, foot, distance, nodes):
    """Sources of ``panel`` graded about ``foot = (u*, v*, X_u, X_v)`` for a target at ``distance``."""
    q, n = nodes
    u0, v0, x_u, x_v = foot
    a = jnp.clip(1.0 - distance / (_GRADING * jnp.maximum(jnp.linalg.norm(x_v) * _TWO_PI / n, 1e-300)),
                 0.0, _MAX_GRADING)
    u, w_u = _graded_u(u0, 0.5 * distance / jnp.maximum(jnp.linalg.norm(x_u), 1e-300), q)
    s = _TWO_PI * jnp.arange(n) / n
    return _sources(panel, *_tensor(u, w_u, v0 + s - a * jnp.sin(s), (1.0 - a * jnp.cos(s)) * _TWO_PI / n))


class MirrorExteriorField:
    """Field of the sources inside a closed surface of three :class:`MirrorPanel` patches.

    The total field ``B`` on the surface, including the through-flux ``B . n`` of the caps, determines
    the field ``V`` of the sources inside outside the surface (``branch="internal"``, the default) and,
    with its sign reversed, the field of the sources outside inside it (``branch="external"``); the two
    sum to ``B`` on the surface.  ``V`` is the outside limit on the surface.

    ``nodes = (q, n)``: Gauss-Legendre points per graded panel and ``v`` nodes of the graded rule;
    ``plain_nodes = (q, n)``: points per patch of the plain rule used far from the surface.
    """

    def __init__(self, panels, nodes=(14, 512), plain_nodes=(32, 96), cloud=(48, 96)):
        self.panels, self.nodes, self.plain_nodes, self.cloud = tuple(panels), nodes, plain_nodes, cloud

    def flux_defect(self):
        """``|net flux| / total |flux|`` of ``B`` through the surface: zero for a solenoidal field.

        A cap whose through-flux is omitted or zeroed leaves a defect of order one.
        """
        charge = jnp.concatenate([_plain(p, self.plain_nodes)[2] for p in self.panels])
        return jnp.abs(jnp.sum(charge)) / jnp.sum(jnp.abs(charge))

    def B_and_derivatives_xyz(self, xyz, *, order=1, branch="internal"):
        """``B`` and its spatial derivatives up to ``order`` (at most 3) at ``xyz``, and a relative error estimate.

        Returns ``(values, estimate)`` as :meth:`VirtualCasingExteriorField.B_and_derivatives_accurate_xyz`:
        ``order + 1`` arrays of shape ``xyz.shape[:-1] + (3,) * (k + 1)`` and, per target, the change of
        ``B`` between the rule and a coarser one (for a target within ``_FLOOR`` of the surface, or the last
        Taylor term if larger), relative to the RMS surface field.  Orders 2 and 3 within ``_FLOOR`` of the
        surface are limited by the three jets of the continuation, which the estimate shows.
        """
        if branch not in ("internal", "external"):
            raise ValueError("branch must be 'internal' or 'external'")
        order = int(order)
        if not 0 <= order <= MAX_DERIVATIVE_ORDER:
            raise ValueError(f"order must be 0..{MAX_DERIVATIVE_ORDER}")
        points = jnp.asarray(xyz)
        flat = points.reshape((-1, 3)).astype(float)
        (nu, nt), plain, coarse = self.cloud, self.plain_nodes, (self.plain_nodes[0] - 8, self.plain_nodes[1] // 2)
        grid = (jnp.repeat(jnp.linspace(0.0, 1.0, nu), nt), jnp.tile(_TWO_PI * jnp.arange(nt) / nt, nu))
        clouds = [jax.vmap(p.position)(*grid) for p in self.panels]
        angles = jnp.stack((grid[0], grid[1] / _TWO_PI), axis=1)
        spacing = max(1.0 / (nu - 1), 1.0 / nt)
        extent = jnp.min(jnp.ptp(jnp.concatenate(clouds), axis=0)) / 2.0
        reference = jnp.sqrt(jnp.mean(jnp.concatenate([jnp.sum(jax.vmap(p.field)(*grid) ** 2, axis=-1) for p in self.panels])))
        join = lambda parts: tuple(jnp.concatenate(a) for a in zip(*parts))
        sources = [join([_plain(p, n) for p in self.panels]) for n in (plain, coarse)]
        floor = _FLOOR * extent

        def locate(panel, cloud, x):
            """Nearest point of a patch: ``(u*, v*, X_u, X_v)`` and the distance."""
            x = jax.lax.stop_gradient(x)
            t = _nearest_surface_point(lambda a, b: panel.position(jnp.clip(a, 0.0, 1.0), _TWO_PI * b),
                                       cloud, angles, spacing, x)
            u, v = jnp.clip(t[0], 0.0, 1.0), _TWO_PI * t[1]
            foot = panel.position(u, v)
            _, x_u, x_v = _tangents(panel.position, jnp.clip(u, 1e-6, 1.0 - 1e-6), v)  # not at an axis, where X_v = 0
            return (u, v, x_u, x_v), jnp.linalg.norm(x - foot), x - foot

        def one(x):
            found = [locate(p, c, x) for p, c in zip(self.panels, clouds)]
            nearest = jnp.argmin(jnp.stack([f[1] for f in found]))
            offset = jnp.stack([f[2] for f in found])[nearest]
            x_u, x_v = (jnp.stack([f[0][k] for f in found])[nearest] for k in (2, 3))
            normal = jnp.cross(x_u, x_v) / jnp.linalg.norm(jnp.cross(x_u, x_v))
            height = offset @ normal
            close = jnp.stack([f[1] for f in found]).min() < floor
            side = jnp.where(height < -1.0e-10 * extent, -1.0, 1.0)
            lift = jax.lax.stop_gradient(jnp.where(close, (side * floor - height) * normal, 0.0))
            moved = x + lift
            found = [locate(p, c, moved) for p, c in zip(self.panels, clouds)]
            spans = [jnp.maximum(jnp.linalg.norm(f[0][2]) * jnp.pi / (2 * plain[0]),
                                 jnp.linalg.norm(f[0][3]) * _TWO_PI / plain[1]) for f in found]
            far = jnp.all(jnp.stack([f[1] > _FAR * h for f, h in zip(found, spans)]))
            tier = jnp.where(close, 2, jnp.where(far, 0, 1))

            def plain_rule(_):
                jets = layer_derivatives(moved[None], *sources[0], order=order)
                rough = layer_derivatives(moved[None], *sources[1], order=0)[0]
                return tuple(v[0] for v in jets), jnp.max(jnp.abs(jets[0][0] - rough[0])) / reference

            def graded_rule(k, continued):
                def rule(_):
                    def build(nodes):
                        return join([_graded(p, f[0], f[1], nodes) for p, f in zip(self.panels, found)])

                    q, n = self.nodes
                    jets = tuple(v[0] for v in layer_derivatives(moved[None], *build((q, n)), order=k))
                    rough = layer_derivatives(moved[None], *build((q - 2, n // 2)), order=0)[0][0]
                    miss = jnp.max(jnp.abs(jets[0] - rough)) / reference
                    if continued:
                        jets, left = _continue(jets, -lift, order, extent, miss * reference, floor)
                        miss = jnp.maximum(miss, jnp.max(jnp.stack(
                            [m * extent**j / 100.0**j for j, m in enumerate(left)])) / reference)
                    return jets, miss
                return rule

            return jax.lax.switch(tier, (plain_rule, graded_rule(order, False),
                                         graded_rule(MAX_DERIVATIVE_ORDER, True)), None)

        values, estimate = jax.lax.map(jax.checkpoint(one), flat)
        sign = 1.0 if branch == "internal" else -1.0
        return (tuple(sign * v.reshape(points.shape[:-1] + v.shape[1:]) for v in values),
                estimate.reshape(points.shape[:-1]))

    def B_plasma_xyz(self, xyz, *, branch="internal"):
        """The field ``V`` (or, for ``branch="external"``, ``-V``) at ``xyz``: ``B_and_derivatives_xyz`` at order 0."""
        return self.B_and_derivatives_xyz(xyz, order=0, branch=branch)[0][0]
