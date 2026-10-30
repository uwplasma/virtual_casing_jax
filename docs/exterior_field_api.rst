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
