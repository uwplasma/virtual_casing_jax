# virtual_casing_jax

JAX implementation of the virtual casing principle with high-order singular quadrature.

Status: parity-validated B/GradB (on- and off-surface) against the C++
reference for multiple datasets, including W7-X. The JAX port includes
adaptive off-surface evaluation, internal/external variants, and
autodiff-ready wrappers.

Performance features:
- Source/target tiling with auto-tuned chunk sizes.
- Rematerialization hooks for GradB singular correction.
- Optional target-scan mode to reduce GradB peak memory (`scan_targets`).
- Mixed-precision POU/patch tables with float64 outputs.
- Bundled Quas3/LHD/W7X geometry assets (converted from SCTL .mat).

SIMSOPT compatibility:
The package ships a SIMSOPT-compatible ``VirtualCasing`` class that
mirrors ``simsopt.mhd.virtual_casing.VirtualCasing`` while using the
JAX backend. Import it as ``from virtual_casing_jax import VirtualCasing``.
See `docs/using_simsopt.rst` and the examples in `examples/` for full scripts.

Bundled test data:
To make the SIMSOPT-style examples and tests self-contained, the repo
includes a small subset of SIMSOPT test assets under `tests/test_files/`
and the VMEC input `examples/inputs/input.QH_finitebeta`. These files
originated from the SIMSOPT repository ([SIMSOPT](https://github.com/hiddenSymmetries/simsopt))
and are used only for validation and example runs.

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

For the new tuning knobs:

```bash
JAX_ENABLE_X64=1 XLA_FLAGS="--xla_dump_to=/tmp/vc_xla --xla_dump_hlo_as_text" \
  python tools/profile_vc.py --case case_vc_large --op GradB --jit \
  --chunk-size auto --target-chunk-size auto --pou-dtype float32 --patch-dtype float32 \
  --interp-block-size auto --remat --donate \
  --repeat 2 --trace-dir /tmp/vc_trace_case_vc_large_GradB

tensorboard --logdir /tmp/vc_trace_case_vc_large_GradB
```

This writes JAX traces under `/tmp/vc_trace_*` and HLO dumps under
`/tmp/vc_xla_*`. See `docs/performance.rst` for detailed interpretation.
