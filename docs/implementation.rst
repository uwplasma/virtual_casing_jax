Implementation
==============

Structure
---------

The code is organized into the following modules:

- ``surface_ops``: FFT-based surface operators.
- ``kernels``: Laplace and Biot-Savart kernels.
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
