Numerics
========

This section describes how the equations are discretized and evaluated.

Grid Conventions
----------------

The surface is parameterized by toroidal angle ``phi`` and poloidal
angle ``theta``. SIMSOPT and the reference BIEST code use normalized
angles of period 1, not ``2*pi``. The grids are uniform:

- ``theta`` in ``[0, 1)`` with ``ntheta`` points.
- ``phi`` in ``[0, 1/nfp)`` with ``nphi`` points for a field period.

For stellarator symmetry (``half_period=True``), the toroidal grid is
shifted by half a grid point. This is essential for spectral accuracy
with uniform weights.

Spectral Resampling
-------------------

The core surface operators are Fourier based:

- ``Upsample``: zero-pad Fourier coefficients.
- ``Resample``: upsample then decimate.
- ``Grad2D``: spectral differentiation in both angles.
- ``RotateToroidal``: phase shift in Fourier space.

The JAX implementation uses unitary FFTs to match the normalization in
SCTL. The reference FFT wrapper scales by ``1/sqrt(N)`` on both forward
and inverse transforms.

Singular Quadrature
-------------------

The boundary integrals involve kernels singular at ``r = r'``. BIEST
uses a high-order scheme:

1. Partition of unity (POU) on a local patch around each target point.
2. The non-singular part is integrated by the trapezoidal rule.
3. The singular part is evaluated in local polar coordinates with a
   high-order radial quadrature.

The scheme is described in detail in Malhotra et al. (2020), and the
implementation in this repository mirrors the reference code in
``biest/singular_correction.hpp``.

Adaptive Quadrature Resolution
------------------------------

The field-period operator chooses a quadrature resolution ``quad_Nt``
``quad_Np`` based on:

- Surface anisotropy estimates.
- A double-layer self-test that checks the expected ``1/2`` jump
  for constant density.

This adaptive selection is replicated in the JAX implementation to
ensure parity.
