Exterior Field API
==================

The high-level exterior-field API is intentionally thin. It keeps the
quadrature and singular-correction logic in ``VirtualCasingJAX`` while adding
VMEC-oriented data containers, branch selection, cylindrical conversion, and
grid export.

User-facing magnetic-field objects, including the SIMSOPT-style stored-point
interface, live in `VMEX <https://github.com/uwplasma/VMEX>`_. This package
keeps explicit-array functions so its quadrature kernels remain stateless and
easy to transform with JAX.

On-surface target chunking
--------------------------

``B_plasma_on_surface()`` uses ``ExteriorFieldConfig.target_chunk_size`` for
on-surface quadrature, as the off-surface evaluators do. A call can override
that memory control without changing the prepared geometry or precision plan::

    B_surface = field.B_plasma_on_surface(precision=plan, target_chunk_size=8)

Branch, side and orientation conventions
----------------------------------------

For a total field ``B0`` on a closed surface, ``external`` is the field of the
sources outside the surface and ``internal`` the field of the sources inside,
so ``external + internal = B0`` at every target.

- On the surface, ``VirtualCasingJAX.compute_external_B`` returns the limit of
  the external field taken from inside the surface, and ``compute_internal_B``
  the limit of the internal field taken from outside. Writing ``PV`` for the
  principal value of the singular integral, ``external = PV + B0/2`` and
  ``internal = B0/2 - PV``. The jump between the two one-sided limits is
  ``B0``. ``compute_principal_value_B`` returns ``PV = (external - internal)/2``
  in a single quadrature pass.
- A constant ``B0`` has only outside sources: ``external = B0``,
  ``internal = 0`` and ``PV = B0/2``.
- Off the surface the jump term is absent. ``B_plasma_xyz`` with
  ``branch="internal"`` (the default) is the field of the internal sources and
  ``branch="external"`` that of the external sources.
- The surface normal is always outward. It is chosen from the geometry
  (the sign of the outward direction at the point of largest ``x``), so
  reversing the poloidal parametrisation does not swap the sides. The
  ``normal`` and ``area_vector`` stored in ``VmecSurfaceFieldData`` are checked
  for shape only; they do not select the orientation. Surfaces for which the
  largest-``x`` heuristic is not the outward direction are not supported.
- On the surface, field values (order 0) and their gradients are available;
  higher derivative orders are evaluated off the surface only
  (``B_and_derivatives_xyz``).

Targets at any distance from the surface
----------------------------------------

The periodic trapezoid rule behind ``B_plasma_xyz`` and
``B_and_derivatives_xyz`` is near-singular within a few node spacings of the
surface: the error is of order one a hundredth of a minor radius away, grows
like ``1/d**k`` for the ``k``-th derivative as the target approaches a node, and
the closed forms carry no failure signal. ``B_and_derivatives_accurate_xyz``
evaluates ``B`` and its derivatives up to order 3 at arbitrary targets, on
either side of the surface, with one call::

    (B, gradB), estimate = field.B_and_derivatives_accurate_xyz(xyz, order=1, tol=1e-8)

Each target is routed, by a stop-gradient ``jnp.where`` that is safe under
``jit``, to the closed forms when their estimated error is below ``tol`` (for
``B``; ``100**k * tol`` for the ``k``-th derivative; scaled by the surface
half-extent ``L`` and the RMS surface field ``B*``), and otherwise to the
near-surface sum, which is evaluated only for the targets that need it. The
closed-form estimate is ``30`` times the squared difference between the
coarsest and the finest schedule level, since the trapezoid error falls
geometrically with the node spacing.

For a routed target a harmonic polynomial field ``P`` is fitted to the surface
field around the nearest node, and the layer sum is taken for the density of
``B - P``. The layer integral of the trace of ``P`` is known exactly -- ``0``
outside the surface and ``-P(x)`` inside it, with the same derivatives -- and is
added back. Within a smooth window of 10 coarse nodes around the target the sum
runs over a level twice as fine, elsewhere over the finest schedule level. A
fixed number of Newton steps on the Fourier interpolant of the surface decides
the side, so the map is ``jit``-, ``vmap``- and AD-compatible: ``jax.jvp`` and
``jax.grad`` through the targets, the surface and the field data agree with the
returned derivatives to rounding. A target exactly on the surface returns the
outside limit.

``estimate`` is an array of dimensionless relative errors of ``B``. For a routed
target it is the misfit of the fitted field one node ring outside the fit, times
a factor calibrated on the gate cases (the error stayed below a fifth of it; the
first derivative, in units of ``L``, below 16 times). It cannot see the
resolution of the field data: with ``n`` nodes per direction the data is only
known to its Fourier interpolation error, ``6e-4``, ``6e-6`` and ``2e-10`` for
``n = 24, 48, 96`` on the gate fixture, and no evaluation of the same data does
better.

On a non-axisymmetric two-loop fixture (targets above 8 random surface points and
2 nodes, on both sides, field grid ``n x n``) the worst scaled errors
``e_k = L**k max|D**k B - ref| / B*`` over ``d = 10**-4 .. 1`` minor radii are:

.. list-table::
   :header-rows: 1

   * - n
     - closed forms, 0.01 a
     - accurate, 1e-4 a to 1 a
     - ``e2``, ``e3`` at 1e-2 a
   * - 24
     - e0 22, e1 4e3
     - e0 1e-4, e1 1.4e-3
     - 0.1, 13
   * - 48
     - e0 5, e1 1e3
     - e0 1.2e-6, e1 1.1e-5
     - 3e-4, 0.07
   * - 96
     - e0 1.2, e1 2.4e2
     - e0 1e-9, e1 3.5e-7
     - 9e-6, 3e-3

At ``n = 96`` the goal ``e0 <= 1e-8``, ``e1 <= 1e-6`` holds at every distance; at
smaller ``n`` the data, not the quadrature, sets the error. Orders 2 and 3 lose
about a digit or two per decade of ``d`` below ``0.1 a``. At the switch (``d`` between
0.3 and 0.5 a at ``n = 96``) the one-sided mismatch between the two evaluations
is ``1.3e-10`` in ``B`` and ``4e-9`` in the gradient; the routed field is
discontinuous by that much, and no smoothness is claimed across it.

Core types
----------

.. automodule:: virtual_casing_jax.exterior_field
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Grid export
-----------

.. automodule:: virtual_casing_jax.grid_export
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Smooth objectives
-----------------

.. automodule:: virtual_casing_jax.objectives
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
