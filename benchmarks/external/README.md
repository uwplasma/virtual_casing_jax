# External Benchmarks

These scripts are reproducible entry points for comparing
`virtual_casing_jax` against external field codes. They are intentionally not
run in default CI.

Initial targets:

- `run_simsopt_vc_compare.sh`: executable SIMSOPT virtual-casing comparison.
- `vc_cpp_compare.py`: hiddenSymmetries/virtual-casing + BIEST W7-X
  virtual-casing comparison.
- `run_extender_compare.sh`: STELLOPT/EXTENDER point-field comparison harness.
- `run_fieldline_compare.sh`: FIELDLINES/TORLINES/FLARE Poincare and
  connection-length comparison harness.
- `extract_fieldlines_h5_samples.py`: compact reference artifact extractor for
  real STELLOPT/FIELDLINES HDF5 runs.
- `trace_fieldlines_grid_candidate.py`: independent short-horizon candidate
  tracer for a stored STELLOPT/FIELDLINES HDF5 grid.
- `trace_fieldlines_simsopt_candidate.py`: independent Simsopt Biot-Savart
  tracer for a STELLOPT/FIELDLINES MAKEGRID coil case, including FIELDLINES
  `EXTCUR` group-current scaling.
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

The committed report
`benchmarks/external/reports/simsopt_vc_compare_qh_report.json` was generated
from the bundled finite-beta QH VMEC/BNORM case and passed the default
thresholds:

- external-normal relative L2: `1.688886506194555e-05`;
- external-vector relative L2: `3.082168761484368e-06`;
- JAX BNORM max residual: `0.006059902309512782`.

The hiddenSymmetries/virtual-casing comparison uses the local upstream
`virtual-casing` Python extension, which wraps the BIEST/SCTL implementation
and includes the W7-X benchmark data used by Malhotra et al. The small
committed report
`benchmarks/external/reports/vc_cpp_w7x_small_compare.json` is intentionally
low-resolution so it remains quick and reproducible on a laptop:

```bash
python benchmarks/external/vc_cpp_compare.py \
  --out benchmarks/external/reports/vc_cpp_w7x_small_compare.json
```

This is a literature-anchored numerical benchmark for the virtual-casing
surface integral. It complements the SIMSOPT parity check by exercising an
independent C++/BIEST implementation and an analytic reference field bundled
with that upstream package.

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
diagnostics in JSON, NPZ, CSV, or STELLOPT/FIELDLINES HDF5 format. Provide
`poincare_xyz` or `poincare_rphiz` points and/or `connection_lengths` for the
plain formats. For STELLOPT/FIELDLINES `.h5` output, the loader reads
`R_lines`, `PHI_lines`, `Z_lines`, `npoinc`, and `L_lines`; it samples every
`npoinc` step as a field-period Poincare section and exports ordered
`poincare_rphiz` points. The report includes ordered point errors or unordered
point-cloud distances, connection-length relative L2 errors, and optional
wall-hit mask mismatch fractions.

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
tracer and ESSOS/JAX export do not preserve the same point ordering. Prefer
`--point-mode labeled` when both files provide `line_id` and `section_phi`:
this matches samples by field-line id and toroidal section before computing
point errors, so the comparison is invariant to export order while still
catching missing, extra, or duplicated sections. The `--section-phi-atol`
option controls the section-angle binning tolerance for labeled comparisons.

An installed STELLOPT/FIELDLINES checkout can be run outside CI and compared
directly:

```bash
xfieldlines -vmec NCSX_s1 -coil coils.NCSX -vac
benchmarks/external/run_fieldline_compare.sh \
  --reference fieldlines_NCSX_s1.h5 \
  --candidate vmec_extender_trace_samples.npz \
  --stellopt-root /path/to/STELLOPT \
  --max-point-relative-l2 1e-3 \
  --max-connection-relative-l2 1e-3 \
  --out /tmp/fieldline_compare_ncsx.json
```

A real STELLOPT/FIELDLINES NCSX vacuum run has also been reduced to a compact
committed reference artifact:

- raw run command:
  `xfieldlines -vmec NCSX_s1 -coil coils.NCSX -vac`;
- raw HDF5 schema: `R_lines`, `PHI_lines`, `Z_lines`, and `B_lines` with shape
  `(24001, 2)`, `npoinc=8`, and `L_lines` with two connection lengths;
- committed sample:
  `benchmarks/external/examples/fieldlines_ncsx_s1_reference.npz`, containing
  6002 ordered `poincare_rphiz` points and two `connection_lengths`;
- extraction metadata:
  `benchmarks/external/reports/fieldlines_ncsx_s1_reference_metadata.json`;
- self-check report:
  `benchmarks/external/reports/fieldlines_ncsx_s1_self_compare.json`.
- labeled self-check report:
  `benchmarks/external/reports/fieldlines_ncsx_s1_labeled_compare.json`.

Regenerate the compact artifact from a raw HDF5 file with:

```bash
python benchmarks/external/extract_fieldlines_h5_samples.py \
  fieldlines_NCSX_s1.h5 \
  --out benchmarks/external/examples/fieldlines_ncsx_s1_reference.npz \
  --report benchmarks/external/reports/fieldlines_ncsx_s1_reference_metadata.json \
  --input input.NCSX_s1 \
  --coils coils.NCSX \
  --source-command "xfieldlines -vmec NCSX_s1 -coil coils.NCSX -vac"
```

The raw HDF5 is not committed because it is much larger than the compact
sample. The committed NPZ is the stable reference contract for candidate ESSOS
or JAX field-line exports.

The repository also includes a short-horizon candidate-vs-reference NCSX
artifact generated by tracing the raw FIELDLINES HDF5 grid independently with
RK4. This is not a replacement for an ESSOS VMEC-extender candidate run, but it
does exercise a real external FIELDLINES grid and trajectory rather than a
self-compare:

```bash
python benchmarks/external/trace_fieldlines_grid_candidate.py \
  fieldlines_NCSX_s1.h5 \
  --source-label external-run/fieldlines_NCSX_s1_run/fieldlines_NCSX_s1.h5 \
  --lines 0 \
  --nsections 16 \
  --substeps 1 \
  --reference-out benchmarks/external/examples/fieldlines_ncsx_s1_line0_16_reference.npz \
  --candidate-out benchmarks/external/examples/fieldlines_ncsx_s1_line0_16_grid_candidate.npz \
  --report benchmarks/external/reports/fieldlines_ncsx_s1_grid_candidate_metadata.json

benchmarks/external/run_fieldline_compare.sh \
  --reference benchmarks/external/examples/fieldlines_ncsx_s1_line0_16_reference.npz \
  --candidate benchmarks/external/examples/fieldlines_ncsx_s1_line0_16_grid_candidate.npz \
  --point-mode labeled \
  --max-point-relative-l2 2e-2 \
  --max-point-rms-distance 2e-2 \
  --max-point-max-distance 4e-2 \
  --max-connection-relative-l2 1e-14 \
  --out benchmarks/external/reports/fieldlines_ncsx_s1_grid_candidate_compare.json
```

The committed comparison report matches 16 labeled Poincare sections for line
0. The RMS point distance is `1.598938676560808e-02` m and the max point
distance is `3.4256843828908055e-02` m. The finite tolerance reflects the
intentional difference between FIELDLINES' LSODE/Hermite spline path and this
small independent linear-grid RK4 candidate tracer.

A stronger short-horizon NCSX artifact traces the same line using an
independent Simsopt Biot-Savart field reconstructed from the MAKEGRID
`coils.NCSX` file. The script applies the same scaled-current convention as
STELLOPT `FIELDLINES/Sources/fieldlines_init_coil.f90`: for each current
group, the raw MAKEGRID currents are rescaled by
`(raw_current / first_group_current) * EXTCUR(group)` using active `EXTCUR`
assignments from `input.NCSX_s1`.

```bash
python benchmarks/external/trace_fieldlines_simsopt_candidate.py \
  fieldlines_NCSX_s1.h5 \
  coils.NCSX \
  input.NCSX_s1 \
  --simsopt-src /path/to/simsopt/src \
  --source-label STELLOPT_FIELDLINES_NCSX_s1 \
  --coils-label external-run/fieldlines_NCSX_s1_run/coils.NCSX \
  --input-label external-run/fieldlines_NCSX_s1_run/input.NCSX_s1 \
  --lines 0 \
  --nsections 16 \
  --substeps 2 \
  --order 20 \
  --ppp 20 \
  --reference-out benchmarks/external/examples/fieldlines_ncsx_s1_line0_16_simsopt_reference.npz \
  --candidate-out benchmarks/external/examples/fieldlines_ncsx_s1_line0_16_simsopt_candidate.npz \
  --report benchmarks/external/reports/fieldlines_ncsx_s1_simsopt_candidate_metadata.json

benchmarks/external/run_fieldline_compare.sh \
  --reference benchmarks/external/examples/fieldlines_ncsx_s1_line0_16_simsopt_reference.npz \
  --candidate benchmarks/external/examples/fieldlines_ncsx_s1_line0_16_simsopt_candidate.npz \
  --reference-label benchmarks/external/examples/fieldlines_ncsx_s1_line0_16_simsopt_reference.npz \
  --candidate-label benchmarks/external/examples/fieldlines_ncsx_s1_line0_16_simsopt_candidate.npz \
  --point-mode labeled \
  --max-point-relative-l2 2e-4 \
  --max-point-rms-distance 4e-3 \
  --max-point-max-distance 7e-3 \
  --max-connection-relative-l2 1e-14 \
  --out benchmarks/external/reports/fieldlines_ncsx_s1_simsopt_candidate_compare.json
```

The committed Simsopt comparison report matches all 16 labeled sections. The
relative L2 point error is `1.9164674995888646e-04`, RMS point distance is
`3.544757608425162e-03` m, and max point distance is
`6.7763504965551525e-03` m. This is an external-code cross-check of coil
current conventions and field-line equations; it is still not a VMEC-extender
plasma-field benchmark because no matching NCSX VMEC `wout` for this raw run
is committed.
