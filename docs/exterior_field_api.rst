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

The periodic trapezoid rule is near-singular within a few node spacings of the
surface: the closed forms of ``B_and_derivatives_xyz`` are off by order one a
hundredth of a minor radius away, and their error grows like ``1/d**k`` for the
``k``-th derivative. By default ``B_plasma_xyz``, ``gradB_plasma_xyz`` (and so
``B_xyz``, ``gradB_xyz``) therefore route each target through
``B_and_derivatives_accurate_xyz``, on either side of the surface, to ``tol =
10**-digits``; ``return_estimate=True`` returns the relative error estimate of
``B``. ``ExteriorFieldConfig(near_surface=False)`` restores the schedule alone,
as does a single-level schedule or ``use_jit_schedule=False``. The call itself
gives derivatives up to order 3::

    (B, gradB), estimate = field.B_and_derivatives_accurate_xyz(xyz, order=1, tol=1e-8)

Each target is routed, by a stop-gradient ``jnp.where`` that is safe under
``jit``, to the closed forms when their estimated error is below ``tol`` (``100**k
* tol`` for the ``k``-th derivative, scaled by the surface half-extent ``L`` and
the RMS surface field ``B*``), and otherwise to the near-surface sum, evaluated
only for those targets. The closed-form estimate is ``30`` times the squared
difference between the coarsest and the finest level.

For a routed target a harmonic polynomial field ``P`` is fitted to the surface
field around the nearest node, and the layer sum is taken for the density of
``B - P``. The layer integral of ``P`` is known exactly (``0`` outside the surface,
``-P(x)`` inside, derivatives included) and is added back. Within a smooth
window of 10 coarse nodes around the target the sum runs over a level twice as
fine, elsewhere over the finest schedule level. The side is decided by a fixed
number of Newton steps on the Fourier interpolant of the surface, so the map is
``jit``-, ``vmap``- and AD-compatible: ``jvp``, ``grad`` and ``jacfwd`` agree with
the returned derivatives to rounding. A target on the surface returns the
outside limit, and the result jumps by at most ``tol`` where the routing switches.

``estimate`` is the misfit of the fitted field one node ring outside the fit,
times a factor calibrated on the gate cases. It cannot see the resolution of the
field data: with ``n`` nodes per direction the data is known to its Fourier
interpolation error, ``6e-4``, ``6e-6`` and ``2e-10`` for ``n = 24, 48, 96`` on
the gate fixture, and no evaluation of the same data does better.

Measured worst scaled errors ``e_k = L**k max|D**k B - ref| / B*`` over
``d = 10**-4 .. 1`` minor radii, on both sides, targets above random points and
nodes, with ``n x n`` field data (closed forms at 0.01 a for comparison):

.. list-table::
   :header-rows: 1

   * - n
     - closed forms, 0.01 a
     - accurate, non-axisymmetric
     - accurate, ``R0/a = 3, 8, 12`` (outside)
   * - 48
     - e0 5, e1 1e3
     - e0 1.2e-6, e1 1.1e-5
     - e0 7e-9, 1e-7, 2e-6; e1 2e-6, 1e-5, 4e-5
   * - 96
     - e0 1.2, e1 2.4e2
     - e0 1e-9, e1 3.5e-7
     - e0 2e-9, 7e-9, 6e-8; e1 3e-6, 1.2e-6, 8e-7

At ``n = 96`` the goal ``e0 <= 1e-8``, ``e1 <= 1e-6`` holds at every distance on
the non-axisymmetric fixture; at smaller ``n`` the data sets the error. Orders 2
and 3 lose a digit or two per decade of ``d`` below ``0.1 a``. On a thin torus
the error has a floor that rises with the aspect ratio (``R0/a = 12``: ``e0``
``6e-8`` at ``n = 96`` and ``144``, ``2e-8`` at ``192``) that finer data does not
remove: the fitted polynomial grows like ``(R/a)**8`` over the torus and the
rounding of its cancelling layer sum is proportional to that. Stronger
regularization lowers the floor but costs the near-surface ``e1``; a decaying
point-source basis, a physical-norm penalty and a size-scaled ridge were tried
and were less accurate. Check ``estimate`` for such surfaces.

Cost: targets the closed forms resolve cost about the same as before (a second
evaluation at the coarsest level gives the estimate); a routed target costs about
1 ms (``B``) to 2 ms (with the gradient) at ``n = 96`` on a laptop CPU.

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
