VMEC Extender
=============

``virtual_casing_jax`` provides the plasma-field core for an EXTENDER-like
workflow built from ``vmec_jax`` boundary data.

Physics model
-------------

Let ``Gamma`` be the VMEC last closed flux surface with outward unit normal
``n``. VMEC supplies the boundary position ``x_b(theta, phi)`` and total field
``B_Gamma(theta, phi)``. The virtual-casing densities are

.. math::

   \sigma = B_\Gamma \cdot n,\qquad K = n \times B_\Gamma .

For targets away from the surface, this package follows the documented
off-surface convention:

.. math::

   B_\mathrm{ext}^\mathrm{VC}(x) =
   \nabla G[\sigma](x) - \mathrm{BiotSavart}[K](x),

.. math::

   B_\mathrm{int}^\mathrm{VC}(x) =
   -\nabla G[\sigma](x) + \mathrm{BiotSavart}[K](x).

The plasma currents are inside the VMEC surface, so the exterior plasma field
uses the **internal** branch:

.. math::

   B_\mathrm{out}(x) = B_\mathrm{coils}(x) + B_\mathrm{int}^\mathrm{VC}(x).

The ``external`` branch remains available for diagnostics and for comparison
with coil-normal fields on a matched free-boundary case. It does not mean
"evaluate outside the VMEC surface".

Coordinate conventions
----------------------

``vmec_jax`` evaluates geometry on ``(s, theta, zeta)`` where ``zeta`` spans a
field period. The physical toroidal angle is

.. math::

   \phi = \zeta / \mathrm{NFP}.

``VmecSurfaceFieldData.phi`` stores the physical toroidal angle. All exterior
field targets are Cartesian unless a method explicitly takes ``(R, phi, Z)``.
The exported cylindrical grid uses physical ``phi``.

Fixed-schedule levels
---------------------

The high-level ``VirtualCasingExteriorField`` uses fixed off-surface schedules
for JIT-friendly evaluation. When ``NFP > 1``, the toroidal size of each
schedule level is rounded up to the next multiple of ``NFP`` before evaluation.
This preserves field-period rotational covariance in the replicated source
surface while keeping user-supplied schedules static.

Python workflow
---------------

.. code-block:: python

   import vmec_jax
   from virtual_casing_jax import (
       ExteriorFieldConfig,
       VirtualCasingExteriorField,
       surface_field_from_vmec_jax,
   )

   run = vmec_jax.run_fixed_boundary("input.vmec")
   surface = surface_field_from_vmec_jax(run.state, run.static, run.indata)
   field = VirtualCasingExteriorField(surface, ExteriorFieldConfig(digits=8))

   B_plasma = field.B_plasma_xyz([[1.8, 0.0, 0.0]])
   B_cyl = field.B_cyl([[1.8, 0.0, 0.0]])

Validation checks
-----------------

At minimum, VMEC-extender changes should report:

* surface orientation;
* ``B_Gamma dot n`` mean, RMS, and max values;
* ``B_internal + B_external - B_total`` on the boundary;
* coil plus internal-branch normal-field cancellation for matched
  free-boundary cases;
* field-period and stellarator-symmetry residuals for cases where those
  symmetries are expected.

This workflow is not a self-consistent SOL or edge MHD solver. It does not
replace HINT, SIESTA, PIES, or M3D-C1 when edge currents, pressure relaxation,
islands, stochastic regions, or resistive response must be solved
self-consistently.
