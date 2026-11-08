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

Targets near the surface
------------------------

The periodic trapezoid rule behind ``B_plasma_xyz`` and
``B_and_derivatives_xyz`` is near-singular within a few node spacings of the
surface: the error is of order one a hundredth of a minor radius away, grows
like ``1/d**k`` for the ``k``-th derivative as the target approaches a node, and
the closed forms carry no failure signal. ``B_and_derivatives_near_surface_xyz``
removes it on either side of the surface::

    (B, gradB), estimate = field.B_and_derivatives_near_surface_xyz(xyz, order=1)

For each target a harmonic polynomial field ``P`` is fitted to the surface field
around the nearest node (interpolating it there), and the layer sum is taken
for the density of ``B - P``. The layer integral of the trace of ``P`` is known
exactly -- ``0`` outside the surface and ``-P(x)`` inside it, with the same
derivatives -- and is added back. The fixed iteration count (a Newton step on
the Fourier interpolant of the surface decides the side) and the absence of
Python exceptions keep it ``jit``-, ``vmap``- and AD-compatible. Derivatives of
every order up to 3 cost the same pass; the derivatives are those of the
returned map, so ``jax.jvp``/``jax.grad`` through the targets, the surface and
the field data agree with the returned gradient to rounding.

``estimate`` is an array of dimensionless relative errors of ``B``, comparable
with ``10**-digits``. It is the misfit of the fitted field one node ring outside
the fit times a factor calibrated on the gate cases, which never saw the error
exceed it; the first-derivative error, in minor radii, was at most 20 times
larger. A high estimate asks for a larger ``degree`` or a finer
``level`` (the accuracy follows the node spacing of the level, not that of the
field data: a 24-node field with ``level=(4*nt, 4*np)`` matches a 48-node field
at its default level). The cost, a few milliseconds per target at 2 x 48 nodes
per direction, is dominated by evaluating ``P`` on the nodes and does not depend
on the derivative order; use it for the targets the schedule's error estimate
rejects. A target exactly on the surface returns the outside limit.

On a non-axisymmetric two-loop fixture, the worst scaled errors
``e_k = L**k max|D**k B - ref| / B*`` (``L`` the minor radius; targets above
random surface points and above nodes, on both sides, field grid ``n x n``,
default level) are:

.. list-table::
   :header-rows: 1

   * - n
     - closed forms, 0.01 a
     - near-surface, 1e-4 a to 0.01 a
     - near-surface, 0.3 a
   * - 24
     - e0 22, e1 4e3
     - e0 8e-5, e1 1.4e-3
     - e0 1e-4, e1 1.6e-3
   * - 48
     - e0 5, e1 1e3
     - e0 2e-6, e1 4e-5
     - e0 3e-7, e1 2.5e-6
   * - 96
     - e0 1.2, e1 2.4e2
     - e0 1.4e-8, e1 8e-7
     - e0 7e-11, e1 4e-11

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
