# virtual_casing_jax

JAX implementation of the virtual casing principle with high-order singular quadrature.

Status: parity-validated B/GradB (on- and off-surface) against the C++
reference for multiple datasets, including W7-X. The JAX port includes
adaptive off-surface evaluation, internal/external variants, and
autodiff-ready wrappers.

Performance features:
- Source/target tiling with auto-tuned chunk sizes.
- Rematerialization hooks for GradB singular correction.
- Mixed-precision POU/patch tables with float64 outputs.
- Bundled Quas3/LHD/W7X geometry assets (converted from SCTL .mat).

Docs
----

Sphinx documentation lives in `docs/` and is configured for ReadTheDocs.
It includes the equations, numerics, implementation details, and validation
strategy. Run locally:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

Profiling
---------

Use the profiling harness to capture JAX traces and inspect performance:

```bash
JAX_ENABLE_X64=1 python tools/profile_vc.py --case case_vc --op B --jit \
  --repeat 5 --trace-dir /tmp/vc_trace

tensorboard --logdir /tmp/vc_trace
```

See `docs/performance.rst` for XLA HLO dumps, GPU kernel profiling, and
memory tuning guidance.
