Validation
==========

Validation is performed against:

- The reference ``virtual-casing`` C++ implementation.
- SIMSOPT virtual casing tests and benchmarks.
- Analytical or semi-analytical test cases (e.g., axisymmetric surfaces).

Test Categories
---------------

1. **Surface Operator Parity**
   - FFT-based resampling and derivatives match C++ outputs.
   - Surface normals, dot/cross products, and area elements match dumps.

2. **Kernel Parity**
   - Laplace and Biot-Savart kernels match analytic formulas and reference
     scaling (``1/(4*pi)``).

3. **Boundary Integral Parity**
   - Singular-corrected ``LaplaceFxdU`` matches reference at ~1e-3
     relative error on small grids.
   - Baseline direct-sum ``LaplaceFxdU`` matches reference up to
     singular-correction errors.
   - Double-layer self-test returns ``0.5`` jump for constant density.

4. **End-to-End Virtual Casing Parity**
   - Baseline ``ComputeB`` using direct-sum integrals matches reference
     within coarse tolerances.
   - Singular-corrected ``ComputeB`` achieves ~1e-4 relative error on
     small grids.
   - ``B_external_normal`` matches reference for SIMSOPT and BIEST test cases.
   - Off-surface ``ComputeBOffSurf`` baseline is validated using
     upsampled direct-sum quadrature.

Tolerance Policy
----------------

We use strict tolerances for small grids and looser tolerances for
large-scale problems where floating point and algorithmic noise
are amplified.
