Off-Surface Evaluation
======================

This section describes the off-surface evaluation paths for ``B`` and
``GradB`` in ``virtual_casing_jax``, including the adaptive refinement
logic and validation strategy. The implementation mirrors the reference
``virtual-casing``/BIEST flow and is designed to be both accurate and
efficient for small to moderate target sets.

Equations
---------

Let ``\Gamma`` be the surface with outward normal ``n``. Define the
surface densities:

.. math::

   \sigma = \mathbf{B}\cdot\mathbf{n}, \qquad
   \mathbf{K} = \mathbf{n}\times\mathbf{B}.

For off-surface targets ``x``, the jump term is absent. The external and
internal fields are

.. math::

   \mathbf{B}_{\mathrm{ext}}(x)
   = \nabla G[\sigma](x) - \mathrm{BiotSavart}[\mathbf{K}](x),

.. math::

   \mathbf{B}_{\mathrm{int}}(x)
   = -\nabla G[\sigma](x) + \mathrm{BiotSavart}[\mathbf{K}](x),

which matches the reference BIEST implementation [MCO2019]_ and
``virtual-casing`` C++ API.

The off-surface gradient is

.. math::

   (\nabla \mathbf{B}_{\mathrm{ext}})_{k i}
   = \varepsilon_{k \ell m}\,\partial_i\partial_\ell G[K_m]
   + \partial_i\partial_k G[\sigma],

and ``\nabla \mathbf{B}_{\mathrm{int}} = - \nabla \mathbf{B}_{\mathrm{ext}}``.

Algorithm
---------

Off-surface evaluation is performed in two stages:

1. **Base resampling**: the surface and field are resampled to a
   quadrature grid that is at least the maximum of:

   - ``NFP * src_nt`` (toroidal modes from the source field)
   - surface resolution from ``setup``
   - a minimum patch size (``13x13``) to match the BIEST off-surface
     singular-correction requirements.

2. **Evaluation**:
   - ``ComputeBOffSurf`` uses a combination of
     ``LaplaceFxdU`` and ``BiotSavartFxU``.
   - ``ComputeGradBOffSurf`` uses ``LaplaceFxd2U`` (second derivatives)
     for both ``\mathbf{K}`` and ``\sigma`` and assembles the curl term.

Adaptive Refinement
-------------------

The off-surface **field** evaluator supports adaptive refinement, matching
the strategy in BIEST:

- Evaluate the Laplace double-layer ``DxU`` on the target set using a
  constant density.
- Define an error metric:

  .. math::

     \epsilon = \max_i \min(|1-U_i|, |U_i|),

  where ``U_i`` is the double-layer potential at target ``i``.
- Refine the source grid by doubling ``(Nt, Np)`` until
  ``\epsilon \le 10^{-digits}`` or the optional ``max_Nt/max_Np`` caps
  are reached.

This logic is implemented in
``virtual_casing_jax.integrals.computeB_offsurface_adaptive`` and
``_offsurface_adapt_grid``, mirroring
``ExtVacuumField::EvalOffSurface`` in the C++ code.

Off-Surface GradB (Base vs Adaptive)
------------------------------------

The reference C++ code evaluates off-surface GradB on the base grid
(no adaptive refinement). The default JAX path follows this behavior to
maintain parity with the C++ dumps.

An **optional adaptive** GradB path is available via
``adaptive=True`` in ``compute_external_gradB_offsurf``. This uses the
same adaptive grid selection as the off-surface field evaluator, and is
validated against a finite-difference gradient of
``compute_external_B_offsurf``.

Validation
----------

Off-surface validation includes:

- Parity tests against the C++ dumps for:
  - ``ComputeBOffSurf`` (external/internal)
  - ``ComputeGradBOffSurf`` (external)
- A finite-difference test comparing adaptive GradB against the
  gradient of ``compute_external_B_offsurf``.

See the tests:

- ``tests/test_virtual_casing_offsurf_parity.py``
- ``tests/test_virtual_casing_gradb_offsurf_fd.py``

Performance Notes
-----------------

- The direct quadrature path is memory-intensive; use ``chunk_size`` to
  control memory.
- Adaptive refinement can upsample aggressively for small surfaces;
  use ``max_Nt`` and ``max_Np`` to cap grid sizes if needed.
- Off-surface GradB is more expensive than off-surface B because it
  evaluates second-derivative kernels.

Implementation Mapping
----------------------

JAX entry points and their reference equivalents:

- ``VirtualCasingJAX.compute_external_B_offsurf`` →
  ``VirtualCasing::ComputeBextOffSurf`` (C++)
- ``VirtualCasingJAX.compute_internal_B_offsurf`` →
  ``VirtualCasing::ComputeBintOffSurf`` (C++)
- ``VirtualCasingJAX.compute_external_gradB_offsurf`` →
  ``VirtualCasing::ComputeGradBOffSurf`` (C++ local parity extension)

All kernel normalizations and sign conventions follow BIEST [BIEST]_ and
the formulas in [MCO2019]_.
