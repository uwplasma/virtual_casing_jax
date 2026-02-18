Performance
===========

The virtual casing evaluation is dominated by surface integrals with
singular kernels. Performance and memory efficiency are addressed via:

- Source/target blocking (tiling).
- JIT-compiled kernels with static shapes.
- Precomputed quadrature tables and interpolation matrices.
- Optional rematerialization to reduce memory in the backward pass.

CPU and GPU
-----------

The same JAX code runs on CPU and GPU. GPU acceleration is achieved
via large batched kernel evaluations. The code avoids Python loops in
performance-critical paths.

Tips and Tricks
--------------

- Use a fixed set of ``nphi``, ``ntheta`` for JIT reuse.
- Cache POU and interpolation tables for each ``(PATCH_DIM, RAD_DIM)``.
- For parity checks, use float64; for production, use mixed precision
  with float32 inputs.
