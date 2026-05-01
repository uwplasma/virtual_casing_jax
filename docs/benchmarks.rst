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
   Compare coil-only, plasma-only, and total fields at points and on grids for
   matched VMEC/coils inputs.

BMW / vector potential
   Compare direct volume-current fields and curl-of-vector-potential grids once
   the BMW prototype is implemented.

FIELDLINES / TORLINES / FLARE
   Compare exported-grid tracing results and connection lengths with external
   field-line tools.

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
