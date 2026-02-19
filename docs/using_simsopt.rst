Using in SIMSOPT
================

This section documents how to use ``virtual_casing_jax`` alongside
SIMSOPT. The package now ships a SIMSOPT-compatible
``VirtualCasing`` class that mirrors
``simsopt.mhd.virtual_casing.VirtualCasing`` but calls the JAX backend.

The JAX-backed class is useful immediately in SIMSOPT scripts while
preserving the original API. It also acts as the drop-in entry point
for the single-stage finite-beta workflows once the SIMSOPT-side
integration is finalized.

SIMSOPT-Compatible VirtualCasing
--------------------------------

Import the class from ``virtual_casing_jax`` and keep the SIMSOPT
workflow unchanged:

.. code-block:: python

   from simsopt.mhd import Vmec
   from virtual_casing_jax import VirtualCasing

   vc = VirtualCasing.from_vmec(
       vmec="wout_example.nc",
       src_nphi=48,
       trgt_nphi=32,
       trgt_ntheta=32,
       filename=None,
   )

   # Identical attributes to the SIMSOPT class:
   Bn = vc.B_external_normal
   vc.save("vcasing_example.nc")

The class implements the same ``from_vmec()``, ``save()``, ``load()``,
and ``plot()`` methods as SIMSOPT. Internally it uses
``VirtualCasingJAX`` to compute the external field, so the output is
ready for parity checks against the C++ backend and for JAX-based
autodiff once connected to differentiable geometry inputs.

Examples
--------

The repository ships direct SIMSOPT examples that mirror the originals,
but swap the import to ``virtual_casing_jax``:

- ``examples/simsopt_stage_two_optimization_finite_beta.py``
- ``examples/simsopt_single_stage_optimization_finite_beta.py``

Bundled Data
------------

To keep the SIMSOPT-flavored scripts self-contained, this repository
ships a small subset of SIMSOPT test assets:

- ``tests/test_files/`` includes the W7-X/QH wout files and BNORM data
  referenced by the SIMSOPT-style tests and stage-two example.
- ``examples/inputs/input.QH_finitebeta`` is the VMEC input used by the
  single-stage finite-beta example.

These files are sourced from SIMSOPT for validation and example runs.
If you have a full SIMSOPT checkout, you can replace them with your own
cases or point to alternative data by editing the example scripts.

Direct JAX Usage
----------------

The snippet below mirrors the SIMSOPT usage pattern but calls the
JAX implementation explicitly:

.. code-block:: python

   from virtual_casing_jax.virtual_casing import VirtualCasingJAX

   vc = VirtualCasingJAX()
   vc.setup(
       digits=5,
       nfp=4,
       half_period=True,
       surf_nt=8,
       surf_np=8,
       X=surface_coords,  # shape (3, surf_nt, surf_np)
       src_nt=8,
       src_np=8,
       trg_nt=6,
       trg_np=6,
   )

   Bext = vc.compute_external_B(B0, quad_nt=quad_nt, quad_np=quad_np)
   GradBext = vc.compute_external_gradB(B0, quad_nt=quad_nt, quad_np=quad_np)

For more advanced workflows (custom quadrature or off-surface
evaluation), use ``VirtualCasingJAX`` directly.
