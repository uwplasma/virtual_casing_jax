Benchmarks
==========

Benchmark scripts live under ``benchmarks/external``. They are not part of the
default unit-test suite because they may require external codes, larger inputs,
or longer runtimes.

The lightweight in-repository smoke benchmark
``benchmarks/vmec_extender_smoke_benchmark.py`` exercises the ``vmec_jax``
bridge, fixed-schedule off-surface field evaluation, and small cylindrical grid
export on the bundled ``circular_tokamak`` VMEC case. It writes a JSON report
with construction time, target-evaluation time, grid-export time, source-grid
size, and boundary ``B dot n`` residual. This benchmark is intended for
regression tracking and CI smoke validation; it is not a substitute for the
external-code comparisons below.

Planned benchmark families
--------------------------

SIMSOPT virtual casing
   ``benchmarks/external/run_simsopt_vc_compare.sh`` runs an executable
   comparison against upstream ``simsopt.mhd.VirtualCasing`` with matched VMEC
   input, source grid, target grid, stellarator-symmetry setting, and digits.
   The default finite-beta QH case also evaluates the bundled BNORM
   sine-series normal-field reference using the SIMSOPT current normalization.
   The script writes a JSON report with external-normal parity, full-vector
   parity, BNORM residuals, timings, grid sizes, and git commit hashes.

EXTENDER / STELLOPT
   ``benchmarks/external/run_extender_compare.sh`` compares point-field
   samples exported by STELLOPT/EXTENDER with samples from the JAX
   VMEC-extender workflow. The executable harness accepts JSON, NPZ, or CSV
   files with matched target points and any of ``B_total_xyz``,
   ``B_plasma_xyz``, and ``B_coils_xyz``. It reports relative L2 errors,
   maximum absolute errors, target-point agreement, and the decomposition
   closure ``B_total = B_plasma + B_coils`` when all components are present.
   When both files provide ``normal_xyz``, it also reports normal-component
   parity and the LCFS boundary diagnostic
   ``rms(B_total dot n) / rms(|B_total|)``. The repository includes a
   deterministic boundary-format example under
   ``benchmarks/external/examples`` for exercising the file contract and
   normal-cancellation thresholds before exchanging files with STELLOPT. A
   complete benchmark run still requires a matched EXTENDER field-sample export
   for the same VMEC/coils input.

BMW / vector potential
   Compare direct volume-current fields and curl-of-vector-potential grids once
   the BMW prototype is implemented.

FIELDLINES / TORLINES / FLARE
   ``benchmarks/external/run_fieldline_compare.sh`` compares Poincare points
   and connection lengths exported by an external field-line tool with the
   corresponding ESSOS/JAX diagnostics. The harness accepts JSON, NPZ, or CSV
   files containing ``poincare_xyz`` or ``poincare_rphiz`` and/or
   ``connection_lengths``. Ordered comparisons report point relative L2, RMS
   distance, and max distance. Unordered point-cloud comparisons report
   symmetric nearest-neighbor RMS and max distances. Optional ``hit_mask`` data
   reports wall-hit mismatch fractions. A deterministic example under
   ``benchmarks/external/examples`` exercises the file contract without
   requiring STELLOPT to be installed.

Reporting requirements
----------------------

Benchmark reports should include:

* input file paths and git commit hashes;
* source and target grid resolutions;
* virtual-casing digits and schedule levels;
* RMS and max relative errors;
* JIT compile time separated from warm execution time;
* peak memory when available;
* a short note on any unresolved unit or sign convention differences.
