# External Benchmarks

These scripts are reproducible entry points for comparing
`virtual_casing_jax` against external field codes. They are intentionally not
run in default CI.

Initial targets:

- `run_simsopt_vc_compare.sh`: executable SIMSOPT virtual-casing comparison.
- `run_extender_compare.sh`: STELLOPT/EXTENDER point-field comparison harness.
- `run_fieldline_compare.sh`: FIELDLINES/TORLINES/FLARE Poincare and
  connection-length comparison harness.
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
closure `B_total = B_plasma + B_coils` when all components are present. If both
inputs provide `normal_xyz`, the report also includes normal-component parity
and the LCFS-style cancellation metric `rms(B_total dot n) / rms(|B_total|)`.

A deterministic boundary-format example is included so the comparator can be
run without STELLOPT installed:

```bash
benchmarks/external/run_extender_compare.sh \
  --reference benchmarks/external/examples/extender_boundary_reference.json \
  --candidate benchmarks/external/examples/extender_boundary_candidate.json \
  --max-normal-relative-l2 1e-14 \
  --max-normal-abs 1e-14 \
  --max-total-normal-relative-rms 1e-14 \
  --out /tmp/extender_compare_example.json
```

The example is not a replacement for a STELLOPT run. It is a reproducible
contract test for sample layout, vector decomposition closure, and boundary
normal cancellation before exchanging files with an external EXTENDER build.

The FIELDLINES/TORLINES comparison consumes reference and candidate field-line
diagnostics in JSON, NPZ, or CSV format. Provide `poincare_xyz` or
`poincare_rphiz` points and/or `connection_lengths`. The report includes
ordered point errors or unordered point-cloud distances, connection-length
relative L2 errors, and optional wall-hit mask mismatch fractions.

A deterministic field-line example is included:

```bash
benchmarks/external/run_fieldline_compare.sh \
  --reference benchmarks/external/examples/fieldline_reference.json \
  --candidate benchmarks/external/examples/fieldline_candidate.json \
  --max-point-relative-l2 1e-14 \
  --max-connection-relative-l2 1e-14 \
  --max-hit-mismatch-fraction 0 \
  --out /tmp/fieldline_compare_example.json
```

Use `--point-mode cloud` for unordered Poincare point clouds where the external
tracer and ESSOS/JAX export do not preserve the same point ordering.
