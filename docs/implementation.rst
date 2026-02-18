Implementation
==============

Structure
---------

The code is organized into the following modules:

- ``surface_ops``: FFT-based surface operators.
- ``kernels``: Laplace and Biot-Savart kernels.
- ``integrals``: baseline direct-sum surface quadrature.
- ``singular_quadrature``: POU + polar quadrature.
- ``boundary_integral``: surface integrals with singular correction.
- ``virtual_casing``: high-level API for on-surface and off-surface fields.

Design Decisions
----------------

Static Shapes
^^^^^^^^^^^^^

JAX compiles for fixed shapes. We select quadrature sizes on the Python
side, then use JIT-compiled kernels for the actual evaluation.

Mixed Precision
^^^^^^^^^^^^^^^

Mixed precision is allowed. The default policy is:

- Inputs can be float32.
- Accumulations are promoted to float64 when required for accuracy.

Custom VJPs
^^^^^^^^^^^

Boundary integrals can be memory-intensive. We plan to implement
``custom_vjp`` rules that re-evaluate kernels during the backward pass
instead of storing large intermediates.

Compatibility with Reference Code
---------------------------------

The goal is bitwise-identical results for small test cases, and
numerical parity for larger grids. All critical numerics and
normalizations follow the BIEST implementation.

Boundary Integrals (Baseline)
-----------------------------

The first JAX implementation uses a direct-sum quadrature:

.. math::

   u(x_i) = \\sum_{j} K(x_i, x_j) \\, f_j \\, w_j

where ``w_j`` is the area element returned by ``SurfNormalAreaElem``.
This reproduces the far-field part of BIEST's Nyström evaluation.

The singular correction used by BIEST (partition-of-unity + polar
quadrature) is not yet replicated. The baseline is therefore expected
to deviate near the diagonal, and is used to build the parity harness
and profiling instrumentation before introducing the full correction.

The direct-sum implementation is chunked to limit memory use and is
JIT-compatible using ``jax.lax.scan``.

Off-Surface Baseline
--------------------

For off-surface targets the kernels are non-singular. The current
implementation evaluates:

.. math::

   B(x) = \\nabla G[B\\cdot n](x) - \\text{BiotSavart}[J](x)

with optional Fourier upsampling of the source grid to improve
accuracy. This provides a drop-in baseline for ``ComputeBOffSurf`` and
serves as the reference for future adaptive refinement.
