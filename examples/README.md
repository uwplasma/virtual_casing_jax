# Examples

This folder mirrors the pedagogic cases from the original
`virtual-casing` test suite. Each script is runnable as a standalone
example (use `PYTHONPATH=.` from the repo root).

Examples:
- `w7x_external_B.py`: external-field accuracy on W7-X geometry.
- `w7x_gradB.py`: GradB accuracy on W7-X geometry.
- `testdata_phi_grid.py`: validates the phi grid layout for half-period
  and full-period configurations.
- `rotating_ellipse_external_B.py`: rotating-ellipse analytic reference.
- `simsopt_stage_two_optimization_finite_beta.py`: SIMSOPT stage-2
  finite-beta optimization example using `virtual_casing_jax.VirtualCasing`.
- `simsopt_single_stage_optimization_finite_beta.py`: SIMSOPT single-stage
  finite-beta optimization example using `virtual_casing_jax.VirtualCasing`.

Bundled inputs:
- `tests/test_files/` contains the VMEC/BNORM data required by the SIMSOPT-style
  examples.
- `examples/inputs/input.QH_finitebeta` provides the VMEC input used by the
  single-stage finite-beta example.
