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
   - Singular-corrected ``LaplaceFxd2U`` (Hedgehog order 8) matches
     reference at ~5e-3 relative error on the GradB parity case.
   - Baseline direct-sum ``LaplaceFxdU`` matches reference up to
     singular-correction errors.
   - Double-layer self-test returns ``0.5`` jump for constant density.

4. **End-to-End Virtual Casing Parity**
   - Baseline ``ComputeB`` using direct-sum integrals matches reference
     within coarse tolerances.
   - Singular-corrected ``ComputeB`` achieves ~1e-4 relative error on
     small grids.
   - Internal ``ComputeB`` and ``ComputeGradB`` match reference on the
     internal parity cases (``case_vc_int`` and ``case_simsopt_int``).
   - ``ComputeGradB`` matches reference within ~5e-3 on the GradB parity
     case.
   - ``VirtualCasingJAX.compute_external_gradB`` matches reference on
     the internal test case and SIMSOPT VMEC case.
   - ``VirtualCasingJAX.compute_internal_gradB`` matches reference on
     the internal test case and SIMSOPT VMEC case.
   - ``VirtualCasingJAX.compute_external_B`` matches reference on the
     internal test case and SIMSOPT VMEC case.
   - ``VirtualCasingJAX.compute_internal_B`` matches reference on the
     internal test case and SIMSOPT VMEC case.
   - Autodiff of ``compute_external_B_autodiff`` matches the C++
     ``ComputeGradB`` outputs on both parity cases.
   - ``B_external_normal`` matches reference for SIMSOPT and BIEST test cases.
   - Off-surface ``ComputeBOffSurf`` baseline is validated using
     upsampled direct-sum quadrature.
   - Off-surface adaptive refinement is validated against the reference
     for small target sets.
   - Off-surface ``ComputeGradBOffSurf`` matches reference on the
     internal virtual-casing test case.

Tolerance Policy
----------------

We use strict tolerances for small grids and looser tolerances for
large-scale problems where floating point and algorithmic noise
are amplified.
