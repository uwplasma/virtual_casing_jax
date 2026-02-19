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

Surface Differentiation and Normals
-----------------------------------

Surface derivatives are computed spectrally. For a Fourier mode
``exp(2*pi*i*(m*theta + n*phi))``:

- ``partial_theta`` multiplies the coefficient by ``-2*pi*i*m``.
- ``partial_phi`` multiplies the coefficient by ``-2*pi*i*n``.

The negative sign matches the FFT sign convention used in BIEST's
``Grad2D`` implementation.

Toroidal frequencies use signed indices ``m`` with the Nyquist index
treated as positive, matching the reference implementation:

``m = t`` for ``t <= Nt/2`` and ``m = t - Nt`` otherwise.

Given ``X_theta`` and ``X_phi``, the unit normal and area element are:

.. math::

   n = \\frac{X_{\\theta} \\times X_{\\phi}}
            {\\lVert X_{\\theta} \\times X_{\\phi} \\rVert},
   \\quad
   dA = \\frac{\\lVert X_{\\theta} \\times X_{\\phi} \\rVert}{N},

with ``N = Nt * Np`` grid points. The orientation is chosen so that the
normal component corresponding to the maximum coordinate among ``x,y,z``
is positive, matching BIEST's convention.

Field-Period Target Selection
-----------------------------

The BIEST field-period operator evaluates on a **subset** of the
quadrature grid. Given:

- ``quad_nt`` / ``quad_np``: quadrature grid sizes for the full surface.
- ``trg_nt`` / ``trg_np``: target grid sizes for one field period.
- ``nfp``: number of field periods.

The target indices are selected by uniform strides:

.. math::

   \\text{skip}_t = \\frac{\\text{quad\\_nt}}{n_{fp}\\,\\text{trg\\_nt}}, \\quad
   \\text{skip}_p = \\frac{\\text{quad\\_np}}{\\text{trg\\_np}}

and ``(i,j)`` maps to the quadrature index
``(t,p) = (i * skip_t, j * skip_p)`` with ``i \\in [0, trg_nt)``.
This picks points from the first field period only.

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

Implementation Notes
--------------------

The JAX port implements the partition-of-unity correction for Laplace
FxdU (Hedgehog order 1) and Laplace Fxd2U (Hedgehog order > 1). The patch
size is chosen using the same thresholding rules as BIEST:

.. math::

   \\text{PDIM} = \\lfloor 1.6\\,\\text{digits}\\,\\text{cond} \\rfloor

and then rounded up to the nearest supported template value
``{6, 8, 12, 16, ..., 64}``. The base polar quadrature order is
``RAD_DIM_0 = \\lfloor 1.6\\,\\text{PDIM} \\rfloor``.

For Laplace Fxd2U, BIEST uses Hedgehog quadrature with a tripled radial
resolution while keeping the angular resolution tied to the base order:

.. math::

   \\text{RAD\\_DIM} = 3\\,\\text{RAD\\_DIM}_0, \\quad
   \\text{ANG\\_DIM} = 2\\,\\text{RAD\\_DIM}_0

The JAX implementation mirrors this convention and uses the same
hedgehog extrapolation weights at nodes ``1..16`` when
``HedgehogOrder=8``.

For quadrature selection, the JAX port also implements the
singular-corrected Laplace ``DxU`` (double-layer) operator with
Hedgehog order 1. This matches the BIEST self-test that verifies the
expected ``1/2`` jump for constant density and drives adaptive
quadrature refinement.

Adaptive Quadrature Resolution
------------------------------

The field-period operator chooses a quadrature resolution ``quad_Nt``
``quad_Np`` based on:

- Surface anisotropy estimates.
- A double-layer self-test that checks the expected ``1/2`` jump
  for constant density.

This adaptive selection is replicated in the JAX implementation to
ensure parity.
