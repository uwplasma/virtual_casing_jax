[![CI](https://github.com/uwplasma/virtual_casing_jax/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/uwplasma/virtual_casing_jax/actions/workflows/ci.yml)
[![CI-Large](https://github.com/uwplasma/virtual_casing_jax/actions/workflows/ci-large.yml/badge.svg)](https://github.com/uwplasma/virtual_casing_jax/actions/workflows/ci-large.yml)
[![Coverage](https://codecov.io/gh/uwplasma/virtual_casing_jax/branch/main/graph/badge.svg)](https://codecov.io/gh/uwplasma/virtual_casing_jax)
[![PyPI](https://img.shields.io/pypi/v/virtual-casing-jax.svg)](https://pypi.org/project/virtual-casing-jax/)
[![Python](https://img.shields.io/pypi/pyversions/virtual-casing-jax.svg)](https://pypi.org/project/virtual-casing-jax/)
[![License](https://img.shields.io/github/license/uwplasma/virtual_casing_jax.svg)](LICENSE)

# virtual_casing_jax

`virtual_casing_jax` is a JAX implementation of the virtual casing
principle for computing magnetic-field contributions from plasma currents
using high-order singular quadrature. It is a JAX translation of
[`virtual-casing`](https://github.com/hiddenSymmetries/virtual-casing), and of
the parts of [`BIEST`](https://github.com/dmalhotra/BIEST) and
[`SCTL`](https://github.com/dmalhotra/SCTL) it uses, by Dhairya Malhotra and
co-authors (see [NOTICE](NOTICE)), extended with differentiable and batched
evaluation. Its SIMSOPT-compatible interface follows
[`hiddenSymmetries/simsopt`](https://github.com/hiddenSymmetries/simsopt).

Documentation is available at
[`virtual-casing-jax.readthedocs.io`](https://virtual-casing-jax.readthedocs.io/).

## Installation

Install the latest release from PyPI:

```bash
python -m pip install --upgrade virtual-casing-jax
```

VMEX users should install its free-boundary dependency with the same Python
interpreter that runs VMEX:

```bash
python -m pip install --upgrade "vmex[freeb]"
python -c "from vmex.core.freeboundary_diff import have_virtual_casing_jax; assert have_virtual_casing_jax()"
```

Or install from a local source checkout:

```bash
git clone https://github.com/uwplasma/virtual_casing_jax.git
cd virtual_casing_jax
python -m pip install -e .
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

Reference test data:
The default suite uses generated analytic cases. Upstream C++ and SIMSOPT
parity data are kept outside git so a clone stays small. To run those scheduled
checks locally, download the checksummed release archive and run the marked
tests:

```bash
python tools/fetch_reference_data.py
pytest -m "large or reference"
```

The finite-beta VMEC input in `examples/inputs/` remains small enough to ship
with the source.

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
from vmex.core.virtual_casing import surface_field_data_from_wout
from virtual_casing_jax import ExteriorFieldConfig, VirtualCasingExteriorField

wout = read_wout("wout_circular_tokamak.nc")
surface = surface_field_data_from_wout(wout, nphi=32, ntheta=32)
field = VirtualCasingExteriorField(surface, ExteriorFieldConfig(digits=8))

points = [[1.8, 0.0, 0.0]]
B_plasma = field.B_plasma_xyz(points)

# Reuse accurate on-surface data for many targets close to the LCFS.
near = field.plan_near_surface(digits=4)
B_near = field.B_plasma_near_surface_xyz(points, near)

# Large target batches can use every visible JAX device.
B_total = field.B_xyz_sharded(points)
```

`plan_near_surface` is a first-order local continuation. Bound field-line
traces by distance from the LCFS, and use a converged direct off-surface
schedule before interpreting farther targets or magnetic topology.

The explicit field functions and their derivatives are JAX differentiable.
VMEX owns the user-facing magnetic-field object and SIMSOPT-compatible
stored-point methods.

For targets outside the VMEC boundary, the plasma-current contribution uses
the `internal` virtual-casing branch because the plasma currents are inside
the LCFS. The `external` branch means currents outside the VMEC surface, not
targets outside it.

The `surface_field_from_vmec_jax` bridge was removed with the retired
`vmec_jax` package. Use VMEX (`pip install vmex`), which builds the surface
data through `vmex.core.virtual_casing.surface_field_data_from_*`.
This field wrapper is not a self-consistent SOL or edge-MHD solver.

## Citing

If you use `virtual_casing_jax`, please cite the software it is translated from
and the papers describing the method (also listed in [CITATION.cff](CITATION.cff)):

- D. Malhotra, B. Medasani, M. Landreman, C. Smiet, A. Cerfon, *virtual-casing: the virtual-casing principle on toroidal surfaces*, https://github.com/hiddenSymmetries/virtual-casing, [doi:10.5281/zenodo.23619289](https://doi.org/10.5281/zenodo.23619289).
- D. Malhotra, *BIEST: Boundary Integral Equation Solver for Taylor states*, https://github.com/dmalhotra/BIEST, [doi:10.5281/zenodo.23511944](https://doi.org/10.5281/zenodo.23511944).
- D. Malhotra, A. J. Cerfon, M. O'Neil, E. Toler, "Efficient high-order singular quadrature schemes in magnetic fusion," *Plasma Physics and Controlled Fusion* 62 (2020) 024004, [doi:10.1088/1361-6587/ab57f4](https://doi.org/10.1088/1361-6587/ab57f4).
- D. Malhotra, A. Cerfon, L.-M. Imbert-Gérard, M. O'Neil, "Taylor states in stellarators: A fast high-order boundary integral solver," *Journal of Computational Physics* 397 (2019) 108791, [doi:10.1016/j.jcp.2019.06.067](https://doi.org/10.1016/j.jcp.2019.06.067).

<details>
<summary>BibTeX</summary>

```bibtex
@software{virtualcasing,
  author    = {Malhotra, Dhairya and Medasani, Bharat and Landreman, Matt and Smiet, Chris and Cerfon, Antoine},
  title     = {virtual-casing: the virtual-casing principle on toroidal surfaces},
  url       = {https://github.com/hiddenSymmetries/virtual-casing},
  doi       = {10.5281/zenodo.23619289},
  version   = {0.3.1},
  year      = {2026}
}

@software{biest,
  author    = {Malhotra, Dhairya},
  title     = {{BIEST}: Boundary Integral Equation Solver for {Taylor} states},
  url       = {https://github.com/dmalhotra/BIEST},
  doi       = {10.5281/zenodo.23511944},
  version   = {1.0.0},
  year      = {2026}
}

@article{malhotra2020quadrature,
  author  = {Malhotra, Dhairya and Cerfon, Antoine J. and O'Neil, Michael and Toler, Evan},
  title   = {Efficient high-order singular quadrature schemes in magnetic fusion},
  journal = {Plasma Physics and Controlled Fusion},
  volume  = {62},
  number  = {2},
  pages   = {024004},
  year    = {2020},
  doi     = {10.1088/1361-6587/ab57f4}
}

@article{malhotra2019taylor,
  author  = {Malhotra, Dhairya and Cerfon, Antoine and Imbert-G{\'e}rard, Lise-Marie and O'Neil, Michael},
  title   = {Taylor states in stellarators: A fast high-order boundary integral solver},
  journal = {Journal of Computational Physics},
  volume  = {397},
  pages   = {108791},
  year    = {2019},
  doi     = {10.1016/j.jcp.2019.06.067}
}
```

</details>

## License

`virtual_casing_jax` is licensed under the [Apache License 2.0](LICENSE). It
includes code translated from virtual-casing, BIEST and SCTL; the
[NOTICE](NOTICE) file lists their copyright and authors and must be kept in any
redistribution.
