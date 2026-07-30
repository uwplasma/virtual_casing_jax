[![CI](https://github.com/uwplasma/virtual_casing_jax/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/uwplasma/virtual_casing_jax/actions/workflows/ci.yml)
[![CI-Large](https://github.com/uwplasma/virtual_casing_jax/actions/workflows/ci-large.yml/badge.svg)](https://github.com/uwplasma/virtual_casing_jax/actions/workflows/ci-large.yml)
[![Coverage](https://codecov.io/gh/uwplasma/virtual_casing_jax/branch/main/graph/badge.svg)](https://codecov.io/gh/uwplasma/virtual_casing_jax)
[![PyPI](https://img.shields.io/pypi/v/virtual-casing-jax.svg)](https://pypi.org/project/virtual-casing-jax/)
[![Python](https://img.shields.io/pypi/pyversions/virtual-casing-jax.svg)](https://pypi.org/project/virtual-casing-jax/)
[![License](https://img.shields.io/github/license/uwplasma/virtual_casing_jax.svg)](LICENSE)

# virtual_casing_jax

`virtual_casing_jax` is a JAX implementation of the virtual casing
principle for computing magnetic-field contributions from plasma currents
using high-order singular quadrature. It is based on the C++ reference
implementation in [`hiddenSymmetries/virtual-casing`](https://github.com/hiddenSymmetries/virtual-casing)
and on the SIMSOPT virtual-casing interface in
[`hiddenSymmetries/simsopt`](https://github.com/hiddenSymmetries/simsopt).

Documentation is available at
[`virtual-casing-jax.readthedocs.io`](https://virtual-casing-jax.readthedocs.io/).

## Installation

Install the latest release from PyPI:

```bash
pip install virtual-casing-jax
```

Or install from a local source checkout:

```bash
git clone https://github.com/uwplasma/virtual_casing_jax.git
cd virtual_casing_jax
pip install -e .
```

## Basic Usage

The SIMSOPT-compatible wrapper can be used as a drop-in virtual-casing
calculation when SIMSOPT is installed:

```python
from virtual_casing_jax import VirtualCasing

vc = VirtualCasing.from_vmec(
    "wout_example.nc",
    src_nphi=32,
    trgt_nphi=32,
    trgt_ntheta=32,
    filename="auto",
)

B_external_normal = vc.B_external_normal
```

For lower-level JAX workflows, use `VirtualCasingJAX` directly after
preparing surface coordinates and magnetic-field arrays:

```python
from virtual_casing_jax import VirtualCasingJAX

vc_jax = VirtualCasingJAX()
vc_jax.setup(digits, nfp, stellsym, Nt, Np, gamma, Nt, Np, Nt, Np)
B_external = vc_jax.compute_external_B(B_total)
```

### Differentiable in the surface geometry

`compute_external_B` and `compute_internal_B` are differentiable in the source
field directly. They are also differentiable in the surface coordinates once
the geometry-dependent precision selection has been frozen:

```python
plan = vc_jax.plan_precision(digits=4)

def loss(surface_coord):
    vc = VirtualCasingJAX()
    vc.setup(digits, nfp, stellsym, Nt, Np, surface_coord, Nt, Np, Nt, Np)
    return objective(vc.compute_internal_B(B_total, precision=plan))

grad = jax.grad(loss)(surface_coord)
```

`precision=plan` reuses concrete quadrature sizes and singular-patch indices,
while the numerical surface geometry remains differentiable. Recreate the plan
when geometry changes are large enough to alter the appropriate quadrature.

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

VMEC Exterior Fields
--------------------

`virtual_casing_jax` can wrap VMEC boundary data as an EXTENDER-like exterior
field. The current downstream integration is
[VMEX](https://github.com/uwplasma/vmex), whose
`vmex.core.freeboundary_diff` module builds `VmecSurfaceFieldData` from a
`wout` file or VMEX state:

```python
from vmex import read_wout
from vmex.core.freeboundary_diff import surface_field_data_from_wout
from virtual_casing_jax import ExteriorFieldConfig, VirtualCasingExteriorField

wout = read_wout("wout_circular_tokamak.nc")
surface = surface_field_data_from_wout(wout, nphi=32, ntheta=32)
field = VirtualCasingExteriorField(surface, ExteriorFieldConfig(digits=8))

B_plasma = field.B_plasma_xyz([[1.8, 0.0, 0.0]])
```

For targets outside the VMEC boundary, the plasma-current contribution uses
the `internal` virtual-casing branch because the plasma currents are inside
the LCFS. The `external` branch means currents outside the VMEC surface, not
targets outside it.

The legacy `surface_field_from_vmec_jax` bridge remains available for the
historical `vmec_jax` package name. This field wrapper is not a self-consistent
SOL or edge-MHD solver.

For synchronized correctness and timing measurements in isolated processes:

```bash
JAX_ENABLE_X64=1 python tools/benchmark_vc.py --suite quick --repeats 5
JAX_ENABLE_X64=1 python tools/benchmark_vc.py --suite full --repeats 5
```

The benchmark writes dated JSON and Markdown reports under
`benchmark_results/`. These machine-specific results are intentionally
untracked.
