# External Benchmarks

These scripts are reproducible entry points for comparing
`virtual_casing_jax` against external field codes. They are intentionally not
run in default CI.

Initial targets:

- `run_simsopt_vc_compare.sh`: executable SIMSOPT virtual-casing comparison.
- `run_extender_compare.sh`: STELLOPT/EXTENDER point-field comparison harness.
- `run_bmw_compare.sh`: BMW/vector-potential comparison placeholder.

Each benchmark should write a machine-readable JSON report containing input
paths, git commits, grid sizes, error metrics, runtime, and memory where
available.

The SIMSOPT comparison runs against the finite-beta QH VMEC/BNORM assets
bundled under `tests/test_files` by default. It reports:

- relative and maximum errors between upstream `simsopt.mhd.VirtualCasing`
  and `virtual_casing_jax.VirtualCasing`;
- the BNORM sine-series normal-field residual using the same current
  normalization as SIMSOPT's validation tests;
- wall-clock timings and source/target grid sizes;
- git commit hashes when the checkouts are available.

The EXTENDER comparison consumes reference and candidate point-field samples in
JSON, NPZ, or CSV format. Provide `--reference` for STELLOPT/EXTENDER output
and `--candidate` for JAX VMEC-extender output. Supported vector components are
`B_total_xyz`, `B_plasma_xyz`, and `B_coils_xyz`; the report includes relative
L2 errors, max absolute errors, target-point agreement, and the decomposition
closure `B_total = B_plasma + B_coils` when all components are present.
