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

2. **Kernel Parity**
   - Laplace and Biot-Savart kernels match reference outputs on random points.

3. **Boundary Integral Parity**
   - Double-layer self-test returns ``0.5`` jump for constant density.

4. **End-to-End Virtual Casing Parity**
   - ``B_external_normal`` matches reference for SIMSOPT and BIEST test cases.

Tolerance Policy
----------------

We use strict tolerances for small grids and looser tolerances for
large-scale problems where floating point and algorithmic noise
are amplified.
