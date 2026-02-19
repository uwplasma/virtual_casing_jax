Performance
===========

The virtual casing evaluation is dominated by surface integrals with
singular kernels. Performance and memory efficiency are addressed via:

- 2D blocking over sources *and* targets (tiling).
- JIT-compiled kernels with static shapes.
- Precomputed quadrature tables and interpolation matrices.
- Auto-tuned chunk sizes per operator (B vs GradB) and backend (CPU/GPU).
- Optional rematerialization to reduce memory in the backward pass.
- Mixed precision for POU/patch intermediates with float64 outputs.

Baseline (Direct-Sum) Path
--------------------------

The initial JAX implementation evaluates Laplace FxdU using a direct
quadrature with chunking and ``jax.lax.scan``. This avoids materializing
the full ``N_trg x N_src`` kernel matrix and keeps memory use linear in
the chunk size. The baseline is primarily for correctness and parity
instrumentation; singular corrections will be layered on later.

Target Blocking (2D Tiling)
---------------------------

The direct-sum kernels now support a second tiling dimension over targets.
This avoids large broadcasted temporaries such as ``[ntrg, nsrc, 3]`` when
``ntrg`` is large. The API exposes ``target_chunk_size`` to control the
target tile size. When enabled, each tile performs a source scan using
``jax.lax.scan`` so accumulation happens inside the kernel loop.

Singular Correction
-------------------

The singular correction introduces per-target patch work. For now it is
implemented in Python with JAX primitives, which is adequate for parity
tests but not yet optimized. The next step is to batch patches and use
``vmap``/``scan`` to reduce overhead and enable GPU acceleration.

Rematerialization
-----------------

The GradB singular correction supports optional rematerialization via
``jax.checkpoint`` to trade recomputation for memory. Use ``remat=True``
in GradB paths to reduce the size of saved intermediates during autodiff.

Adaptive Off-Surface
--------------------

Adaptive refinement requires repeated evaluations of a double-layer
test. This is currently a Python loop; performance will improve once
the refinement is JIT-compiled with static shape schedules.

Off-Surface GradB
-----------------

The off-surface gradient evaluates second-derivative kernels and is
more expensive than the field evaluation. The default path matches the
C++ reference and uses the base resampled grid (no adaptive refinement).
Enable ``adaptive=True`` only when additional accuracy is needed, and
use ``max_Nt``/``max_Np`` to cap growth.

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
peak memory. ``target_chunk_size`` provides a second tiling dimension
over targets. For parity tests, ``chunk_size=1024`` and
``target_chunk_size=auto`` is a good balance.

Auto tuning is enabled by passing ``chunk_size="auto"`` and
``target_chunk_size="auto"`` (default in the high-level API). The
heuristics can be overridden via environment variables:

- ``VCJAX_CHUNK_B`` / ``VCJAX_CHUNK_B_SRC`` / ``VCJAX_CHUNK_B_TRG``
- ``VCJAX_CHUNK_BOFF`` / ``VCJAX_CHUNK_BOFF_SRC`` / ``VCJAX_CHUNK_BOFF_TRG``
- ``VCJAX_CHUNK_GRADB`` / ``VCJAX_CHUNK_GRADB_SRC`` / ``VCJAX_CHUNK_GRADB_TRG``

Interpolation Blocking
~~~~~~~~~~~~~~~~~~~~~~

``interp_block_size`` controls blocking of the polar interpolation in the
singular correction. ``interp_block_size="auto"`` (default) uses a block
size of 64 for ``B`` and 32 for ``GradB`` when the polar grid is large,
reducing temporary memory. Set ``interp_block_size=None`` to restore the
full (unblocked) interpolation.

On ``case_vc_large`` (CPU HLO), enabling ``interp_block_size="auto"``
reduces the largest singular-correction temporaries to ~50 MiB (``B``)
and ~76 MiB (``GradB``), compared to >150 MiB without interpolation
blocking even with ``patch_dtype="float32"``.

JIT Caching
~~~~~~~~~~~

The high-level wrappers in ``VirtualCasingJAX`` expose JIT-compiled
variants (``compute_external_B_jit`` and ``compute_external_gradB_jit``).
These cache compiled functions keyed by argument settings. For repeated
evaluations with fixed grid sizes, prefer the JIT variants to amortize
compilation cost.

For long-running loops, pass ``donate=True`` to the JIT wrappers to
allow XLA to reuse the input buffers and reduce peak memory.

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

For singular correction, the POU/polar interpolation tables can be cast
to float32 while keeping the final accumulation in float64. Use
``pou_dtype="auto"`` or ``pou_dtype="float32"`` in high-level calls to
enable this optimization. The interpolation weights and patch gather
values can be cast independently via ``patch_dtype="auto"`` (or
``"float32"``), which reduces the largest temporary tensors while the
final outputs remain in the input precision.

On ``case_vc_large`` (CPU HLO), ``patch_dtype="float32"`` reduces the
largest singular-correction gather from ~304 MiB to ~152 MiB for ``B``,
and from ~457 MiB to ~228 MiB for ``GradB``.

Precompute Reuse
~~~~~~~~~~~~~~~~

The polar quadrature tables and interpolation weights are cached via
``precompute_singular``. Patch index maps are cached per quadrature
setup inside ``VirtualCasingJAX`` to avoid recomputing the patch gather
indices on each call. Patch indices are stored as int32 to reduce memory
traffic during gathers.

Profiling and Diagnostics
-------------------------

JAX Profiler (TensorBoard)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the profiling harness in ``tools/profile_vc.py`` to emit a trace
that can be opened in TensorBoard:

.. code-block:: bash

   JAX_ENABLE_X64=1 python tools/profile_vc.py \
     --case case_vc --op B --jit --repeat 5 \
     --chunk-size auto --target-chunk-size auto \
     --trace-dir /tmp/vc_trace

   tensorboard --logdir /tmp/vc_trace

The trace includes XLA compilation time, kernel launches, and host-to-device
transfer costs. Always call ``jax.block_until_ready`` (handled by the script)
to ensure timings reflect actual execution.

XLA HLO / MLIR Dumps
~~~~~~~~~~~~~~~~~~~~

To inspect XLA lowering and fusion decisions, enable dump flags:

.. code-block:: bash

   XLA_FLAGS=\"--xla_dump_to=/tmp/xla --xla_dump_hlo_as_text\" \\
   JAX_ENABLE_X64=1 python tools/profile_vc.py --case case_vc --op GradB --jit

The dump directory contains HLO modules and MLIR. These are useful for
verifying fusion, identifying large intermediates, and checking precision
lowering.

Kernel-Level GPU Profiling
~~~~~~~~~~~~~~~~~~~~~~~~~~

On NVIDIA GPUs, use ``nsys`` or ``nvprof`` to profile kernel launches:

.. code-block:: bash

   nsys profile -o /tmp/vc_profile \\
     python tools/profile_vc.py --case case_vc --op B --jit --repeat 10

Pair this with the JAX trace to correlate high-level ops with GPU kernels.

Memory Audits
~~~~~~~~~~~~~

Memory usage is dominated by chunked kernel evaluation and intermediate
arrays during singular correction. Use smaller ``chunk_size`` values to
reduce peak memory, and profile with multiple chunk sizes to identify
the best speed/memory tradeoff. For large runs, consider setting:

.. code-block:: bash

   export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

to control the allocator footprint on GPU.
