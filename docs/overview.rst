Overview
========

Purpose
-------

This project re-implements the virtual casing principle and the associated
high-order singular quadrature schemes used in the BIEST library in pure
Python with JAX acceleration. The goals are:

- Full end-to-end differentiability via JAX autodiff.
- CPU and GPU execution.
- Parity with the reference C++ implementation of ``virtual-casing``.
- A clean, testable, and documented numerical pipeline.

Scope
-----

The v1 target includes:

- On-surface evaluation of the **external and internal** fields via the
  virtual casing principle.
- On-surface gradients (``GradB``) with hyper-singular corrections.
- Off-surface evaluation with adaptive surface upsampling.
- Off-surface gradients (``GradB``) with direct quadrature.
- High-order singular quadrature (partition of unity + polar change of variables).

The implementation mirrors the structure of the reference code:

- Surface operators (Fourier resampling, differentiation, normal vectors).
- Boundary integral operators (Laplace and Biot-Savart kernels).
- High-order singular corrections.
- Field-period symmetry handling.

References
----------

The primary algorithmic reference is:

- D. Malhotra, A. J. Cerfon, M. O'Neil, and E. Toler,
  "Efficient high-order singular quadrature schemes in magnetic fusion",
  Plasma Physics and Controlled Fusion 62, 024004 (2020).
  Preprint: https://arxiv.org/abs/1909.07417

See :doc:`references` for full citations.
