Functional API
==============

The functional API makes the surface coordinates ``X`` a JAX input so shape
derivatives are native and end-to-end differentiable. This is essential for
single-stage optimization pipelines where geometry updates are part of the
computational graph.

Key idea
--------

The class-based ``VirtualCasingJAX`` caches quadrature geometry in a mutable
object. That is great for throughput, but it separates the geometry from the
autodiff graph. The functional API instead rebuilds the geometry from ``X`` on
each call, while keeping all discrete quadrature choices *static* so JAX can
trace and compile.

Core functions live in ``virtual_casing_jax.functional``:

* ``build_surface_coord``: reproduce the BIEST-style full-field-period surface
  coordinates from a single-period input grid.
* ``build_quad_setup``: resample to quadrature grid, compute derivatives and
  normals.
* ``build_patch_idx``: precompute the patch indices for singular quadrature.
* ``compute_external_B_functional`` / ``compute_internal_B_functional``:
  on-surface virtual casing fields with singular correction.
* ``compute_external_gradB_functional`` / ``compute_internal_gradB_functional``:
  on-surface gradients.

Because the patch size and quadrature sizes are *discrete* decisions, they must
remain fixed during differentiation. The helper ``prepare_functional_setup``
can be used to compute those values outside autodiff, and then passed into the
functional calls as static arguments.

Example
-------

.. code-block:: python

    import jax
    import jax.numpy as jnp
    from virtual_casing_jax.functional import (
        prepare_functional_setup,
        compute_external_B_functional,
    )

    # X: surface coordinates (3, surf_nt, surf_np)
    # B0: total field on source grid (3, src_nt, src_np)
    setup = prepare_functional_setup(
        X,
        digits=6,
        nfp=1,
        half_period=False,
        surf_nt=16,
        surf_np=16,
        src_nt=16,
        src_np=16,
        trg_nt=16,
        trg_np=16,
        quad_nt=24,
        quad_np=24,
    )

    def scalar_objective(xsurf):
        b = compute_external_B_functional(
            xsurf,
            B0,
            digits=6,
            nfp=setup.nfp,
            half_period=setup.half_period,
            surf_nt=setup.surf_nt,
            surf_np=setup.surf_np,
            src_nt=setup.src_nt,
            src_np=setup.src_np,
            trg_nt=setup.trg_nt,
            trg_np=setup.trg_np,
            quad_nt=setup.quad_nt,
            quad_np=setup.quad_np,
            patch_dim0=setup.patch_dim0,
            patch_idx=setup.patch_idx,
            orient=setup.orient,
        )
        return jnp.sum(b * b)

    grad_x = jax.grad(scalar_objective)(X)

Guidelines
----------

* Keep ``quad_nt``, ``quad_np``, and ``patch_dim0`` static during differentiation.
  If these change, the discretization changes and autodiff is not meaningful.
* Use ``prepare_functional_setup`` outside the gradient context to select
  appropriate quadrature sizes.
* ``orient`` is treated as a fixed sign (``±1``) computed from the input
  geometry. This is a topological choice that should not be differentiated.
