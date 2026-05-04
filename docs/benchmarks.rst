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
   The committed finite-beta QH report under
   ``benchmarks/external/reports/simsopt_vc_compare_qh_report.json`` passed
   the default thresholds with external-normal relative L2
   ``1.688886506194555e-05`` and external-vector relative L2
   ``3.082168761484368e-06``.

hiddenSymmetries / BIEST virtual casing
   ``benchmarks/external/vc_cpp_compare.py`` compares the JAX implementation
   against the upstream hiddenSymmetries ``virtual-casing`` Python extension,
   which wraps BIEST/SCTL and ships the W7-X benchmark data used by Malhotra
   et al., Plasma Physics and Controlled Fusion 62, 024004 (2020). The
   committed small W7-X report under
   ``benchmarks/external/reports/vc_cpp_w7x_small_compare.json`` is a
   literature-anchored virtual-casing surface-integral check. It is deliberately
   low-resolution so it remains practical on a laptop; higher-resolution
   versions should be generated outside default CI and tracked as benchmark
   artifacts.

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
   corresponding ESSOS/JAX diagnostics. The harness accepts JSON, NPZ, CSV, or
   STELLOPT/FIELDLINES HDF5 files. Plain files should contain
   ``poincare_xyz`` or ``poincare_rphiz`` and/or ``connection_lengths``.
   FIELDLINES HDF5 files are read directly from ``R_lines``, ``PHI_lines``,
   ``Z_lines``, ``npoinc``, and ``L_lines``; every ``npoinc`` step is treated
   as a field-period Poincare section. Ordered comparisons report point
   relative L2, RMS distance, and max distance. Unordered point-cloud
   comparisons report symmetric nearest-neighbor RMS and max distances.
   Labeled comparisons match by ``line_id`` and ``section_phi`` before
   computing point errors, which is the preferred mode for external-code
   parity once both tracers export the same seed-line and Poincare-section
   labels. Optional ``hit_mask`` data reports wall-hit mismatch fractions. A
   deterministic example under ``benchmarks/external/examples`` exercises the
   file contract without requiring STELLOPT to be installed.

   A real STELLOPT/FIELDLINES NCSX vacuum run has been reduced to
   ``benchmarks/external/examples/fieldlines_ncsx_s1_reference.npz`` with 6002
   ordered Poincare samples and two connection lengths. The raw HDF5 came from
   ``xfieldlines -vmec NCSX_s1 -coil coils.NCSX -vac`` and contained
   ``R_lines``, ``PHI_lines``, ``Z_lines``, and ``B_lines`` arrays with shape
   ``(24001, 2)`` plus ``npoinc=8``. The compact artifact is generated by
   ``benchmarks/external/extract_fieldlines_h5_samples.py`` and validated by
   ``benchmarks/external/reports/fieldlines_ncsx_s1_self_compare.json``.
   A short-horizon candidate-vs-reference artifact is also generated from the
   same raw HDF5 grid by ``benchmarks/external/trace_fieldlines_grid_candidate.py``.
   This independent RK4 tracer uses the stored FIELDLINES
   ``R*BR/BPHI`` and ``R*BZ/BPHI`` grid for line 0 over 16 Poincare sections.
   The committed report
   ``benchmarks/external/reports/fieldlines_ncsx_s1_grid_candidate_compare.json``
   matches all 16 labeled sections with RMS point distance
   ``1.598938676560808e-02`` m and max point distance
   ``3.4256843828908055e-02`` m. The finite tolerance is expected because the
   candidate uses linear grid interpolation and a minimal RK4 path rather than
   FIELDLINES' LSODE/Hermite spline implementation.

   A stronger NCSX short-horizon artifact is generated by
   ``benchmarks/external/trace_fieldlines_simsopt_candidate.py``. It loads the
   same MAKEGRID ``coils.NCSX`` file through Simsopt, applies FIELDLINES'
   scaled ``EXTCUR`` convention
   ``(raw_current / first_group_current) * EXTCUR(group)``, and traces line 0
   for 16 Poincare sections with RK4. The committed report
   ``benchmarks/external/reports/fieldlines_ncsx_s1_simsopt_candidate_compare.json``
   matches all 16 labeled sections with relative L2 point error
   ``1.9164674995888646e-04``, RMS point distance
   ``3.544757608425162e-03`` m, and max point distance
   ``6.7763504965551525e-03`` m. This benchmark validates coil-current
   conventions and field-line equations against FIELDLINES, but it remains a
   vacuum-coil cross-check rather than a VMEC-extender plasma-field benchmark.

Physics/literature benchmark ladder
-----------------------------------

The external artifacts are only one layer of scientific validation. A stronger
Phase 1 benchmark ladder should include:

* virtual-casing jump identities on the boundary:
  ``B_internal + B_external = B_surface`` and the corresponding normal-field
  cancellation;
* analytic rotating-ellipse/current-loop fields, where the reference field is
  available from elliptic-integral Biot-Savart formulas;
* W7-X virtual-casing convergence against the Malhotra et al. BIEST benchmark
  data;
* SIMSOPT finite-beta QH VMEC/BNORM parity with matched source and target
  grids;
* STELLOPT/FIELDLINES Poincare and connection-length parity on NCSX or W7-X
  external traces;
* grid diagnostics in current-free target regions: finite-difference
  ``div B`` convergence, field-period symmetry, stellarator-symmetry parity,
  and curl-free residuals away from plasma and coils.

For downstream scrape-off-layer work, use the stellarator edge/SOL literature
as acceptance guidance rather than claiming predictive turbulence from the
field benchmark alone: connection-length maps, strike-line topology, magnetic
mesh quality, wall-hit statistics, and sensitivity to spatially varying
transport coefficients are the relevant bridge metrics before running a full
``jax_drb`` turbulence campaign.

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
