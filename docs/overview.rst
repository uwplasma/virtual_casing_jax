Overview
========

Purpose
-------

This project is a JAX translation of virtual-casing [VC]_, and of the parts
of BIEST [BIEST]_ and SCTL [SCTL]_ it uses, by Dhairya Malhotra and co-authors
(see the ``NOTICE`` file). The goals of the translation are:

- Full end-to-end differentiability via JAX autodiff.
- CPU and GPU execution.
- Parity with the original C++ ``virtual-casing`` code.
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

The implementation follows the structure of the original C++ code:

- Surface operators (Fourier resampling, differentiation, normal vectors).
- Boundary integral operators (Laplace and Biot-Savart kernels).
- High-order singular corrections.
- Field-period symmetry handling.

References
----------

If you use this package, please cite the software it is translated from,
virtual-casing [VC]_ and BIEST [BIEST]_, and the papers describing the method,
[MCO2019]_ and [MCIO2019]_. See :doc:`references` for full citations.
