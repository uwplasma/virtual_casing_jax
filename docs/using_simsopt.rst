Using in SIMSOPT
================

This section documents the intended integration path with SIMSOPT.
At the moment, SIMSOPT still calls the C++ ``virtual_casing`` backend,
and the JAX backend is invoked directly from Python for parity testing.

Planned Integration Path
------------------------

1. Add a SIMSOPT backend switch (or a parallel JAX class) that calls
   ``VirtualCasingJAX`` for on-surface fields and gradients.
2. Replace finite-difference derivatives in
   ``examples/3_Advanced/single_stage_finite_beta.py`` with JAX autodiff
   of ``VirtualCasingJAX.compute_external_B`` or
   ``VirtualCasingJAX.compute_external_gradB``.
3. Verify the full single-stage pipeline against the current
   two-stage reference.

Current Usage (Direct)
----------------------

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

Once SIMSOPT integration begins, this section will be expanded with
full examples and hooks.
