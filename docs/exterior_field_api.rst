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
the RMS surface field ``B*``), and otherwise to a rule graded about the target, evaluated
only for those targets. The closed-form estimate is ``30`` times the squared
difference between the coarsest and the finest level.

The graded rule is the same layer sum as the closed forms, over better placed
nodes. The field data and the boundary are interpolated spectrally from their
samples (cylindrical components, so one field period suffices). For each target
the nearest boundary point is found by a fixed number of Newton steps, and the
nodes ``theta = theta* + u - a sin u`` (likewise ``phi``), with ``u`` uniform
and ``a`` chosen so that the local spacing is an eighth of the distance, cluster
about it: the substitution is entire and periodic, so the trapezoid rule stays
spectrally accurate while the nodes crowd the singularity. The default node
count is twice that of the finest schedule level and at least ``256 x 512``,
because resolving a distance does not depend on the data
(``ExteriorFieldConfig.graded_nodes`` overrides it).
Within ``1e-3`` it is evaluated at that distance, on the target's side, with
``B`` and its derivatives to order 3, and Taylor-continued along the normal:
each side's field is analytic up to the boundary, so a target on the surface
returns the outside limit. The grading centre is held fixed under
differentiation (the rule's value does not depend on it), so ``jvp``, ``grad``
and ``jacfwd`` through the targets and the field data agree with the returned
derivatives, up to the truncation of the continuation (``1e-8`` of the scale
within ``1e-3``). The result jumps by at most ``tol`` where the routing
switches.

``estimate`` is, for a graded target, the larger of the change from using
every second node, the last Taylor term (within ``1e-3``) and the field's
spectrum in the top quarter of the samples. The last one is what bounds the
resolution of the data: with ``n`` samples per direction the field is known to
its Fourier interpolation error, ``6e-4``, ``6e-6`` and ``2e-10`` for
``n = 24, 48, 96`` on the gate fixture, and no evaluation of the same data does
better.

Measured worst scaled errors ``e_k = L**k max|D**k B - ref| / B*`` for ``k = 0``
(``B``) and ``k = 1`` (``grad B``), over both sides and random targets above
random nodes, against the same rule on four times the nodes:

.. list-table::
   :header-rows: 1

   * - deck
     - quantity
     - d/a = 1
     - d/a = 0.1
     - d/a = 0.01
     - d/a = 0.001
     - d/a = 0

   * - QA deck, 64x64
     - B
     - 7e-15
     - 8e-15
     - 4e-14
     - 9e-12
     - 4e-11

   *
     -
     - grad B
     - 2e-13
     - 1e-13
     - 1e-11
     - 2e-08
     - 4e-08

   * - QA deck, 128x128
     - B
     - 4e-12
     - 3e-15
     - 3e-14
     - 3e-13
     - 6e-13

   *
     -
     - grad B
     - 2e-13
     - 4e-14
     - 3e-12
     - 2e-10
     - 6e-10

   * - QA, R0/a = 15, 64x64
     - B
     - 8e-15
     - 7e-15
     - 1e-13
     - 9e-12
     - 4e-11

   *
     -
     - grad B
     - 1e-13
     - 1e-13
     - 2e-11
     - 2e-08
     - 5e-08

   * - QA, R0/a = 15, 128x128
     - B
     - 2e-13
     - 4e-15
     - 5e-14
     - 5e-13
     - 1e-12

   *
     -
     - grad B
     - 8e-12
     - 9e-14
     - 1e-11
     - 5e-10
     - 1e-09

Data of 24 to 48 samples per direction are known only to their Fourier
interpolation error (``1e-5`` to ``1e-7``); ``estimate`` reports that limit.


Cost: targets the closed forms resolve cost about the same as before (a second
evaluation at the coarsest level gives the estimate); a graded target costs
about 7 ms (``B``) to 9 ms (with the gradient) above ``1e-3`` half-extents, and about 25 ms within it (order-3 jets on at least 131 072 nodes), at 64 samples per direction on a laptop CPU; the closed forms cost about 0.4 ms. On an accelerator the targets of a chunk run together, but the rule is still several times slower than the closed forms. The graded rule is a point-query tool: it runs target by target, so a
large batch of near-surface targets is better served by the closed forms on a
finer data set where that is possible.

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
