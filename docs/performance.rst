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

Performance Guide
-----------------

Chunk Size
~~~~~~~~~~

The ``chunk_size`` parameter controls the source tiling in direct
quadrature. Larger chunks improve arithmetic intensity but increase
peak memory. For parity tests, ``chunk_size=1024`` is a good balance.
On GPUs, values in the 2k–8k range typically work well.

JIT Caching
~~~~~~~~~~~

The high-level wrappers in ``VirtualCasingJAX`` expose JIT-compiled
variants (``compute_external_B_jit`` and ``compute_external_gradB_jit``).
These cache compiled functions keyed by argument settings. For repeated
evaluations with fixed grid sizes, prefer the JIT variants to amortize
compilation cost.

Batching
~~~~~~~~

Use ``compute_external_B_batch`` or ``compute_external_gradB_batch`` when
evaluating many fields in parallel (e.g., multiple VMEC surfaces or
Monte Carlo samples). These functions use ``vmap`` to avoid Python loops.

Precision Tradeoffs
~~~~~~~~~~~~~~~~~~~

Float64 is recommended for parity with the C++ backend. Mixed precision
with float32 inputs can provide significant speedups, but requires
relaxed tolerances in validation. Keep ``jax_enable_x64`` enabled in CI
to maintain reference accuracy.

Precompute Reuse
~~~~~~~~~~~~~~~~

The polar quadrature tables and interpolation weights are cached via
``precompute_singular``. Patch index maps are cached per quadrature
setup inside ``VirtualCasingJAX`` to avoid recomputing the patch gather
indices on each call.
