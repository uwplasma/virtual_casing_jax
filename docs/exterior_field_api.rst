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
