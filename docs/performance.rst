Performance
===========

The virtual casing evaluation is dominated by surface integrals with
singular kernels. Performance and memory efficiency are addressed via:

- Source/target blocking (tiling).
- JIT-compiled kernels with static shapes.
- Precomputed quadrature tables and interpolation matrices.
- Optional rematerialization to reduce memory in the backward pass.

Baseline (Direct-Sum) Path
--------------------------

The initial JAX implementation evaluates Laplace FxdU using a direct
quadrature with chunking and ``jax.lax.scan``. This avoids materializing
the full ``N_trg x N_src`` kernel matrix and keeps memory use linear in
the chunk size. The baseline is primarily for correctness and parity
instrumentation; singular corrections will be layered on later.

Singular Correction
-------------------

The singular correction introduces per-target patch work. For now it is
implemented in Python with JAX primitives, which is adequate for parity
tests but not yet optimized. The next step is to batch patches and use
``vmap``/``scan`` to reduce overhead and enable GPU acceleration.

Adaptive Off-Surface
--------------------

Adaptive refinement requires repeated evaluations of a double-layer
test. This is currently a Python loop; performance will improve once
the refinement is JIT-compiled with static shape schedules.

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
  Enable ``jax_enable_x64`` in tests to match the reference C++ results.
