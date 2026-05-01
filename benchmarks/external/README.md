# External Benchmarks

These scripts are reproducible entry points for comparing
`virtual_casing_jax` against external field codes. They are intentionally not
run in default CI.

Initial targets:

- `run_simsopt_vc_compare.sh`: SIMSOPT virtual-casing comparison.
- `run_extender_compare.sh`: STELLOPT/EXTENDER comparison placeholder.
- `run_bmw_compare.sh`: BMW/vector-potential comparison placeholder.

Each benchmark should write a machine-readable JSON report containing input
paths, git commits, grid sizes, error metrics, runtime, and memory where
available.
