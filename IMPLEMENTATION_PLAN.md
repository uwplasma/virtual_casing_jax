# VirtualCasingJAX Correctness Remediation and Benchmark Plan

**Date proposed:** July 23, 2026  
**Date implemented:** July 23, 2026  
**Status:** Implemented and validated

## Summary

Implement all confirmed correctness fixes from `REVIEW_REPORT.md` and
Claude's addendum, then validate the module against independent analytic
fields, pinned C++ reference data, eager/JIT consistency, autodiff identities,
and bounded off-surface convergence.

CPU benchmarking is required and GPU benchmarking is optional. Performance
results are recorded, but timing and memory measurements are not CI pass/fail
thresholds.

## Implementation Changes

### Singular kernels and autodiff

- Replace `_safe_rinv` with a masked-safe formulation that never evaluates
  `rsqrt(0)` on the differentiable path.
- Add reverse-mode, VJP, JVP, and higher-order finite-value tests at
  coincident source/target points.
- Forward `patch_dtype` and `interp_block_size` consistently through both the
  primal and JVP paths of `compute_external_B_autodiff`.

### JIT and cache lifecycle

- Split singular precomputation into a bounded host-NumPy cache and uncached
  JAX conversion, preventing tracers from entering global cache state.
- Limit the host singular-precomputation cache to eight configurations.
- Clear every compiled closure in `_jit_cache` after each successful
  `VirtualCasingJAX.setup()` call.
- Resolve automatic quadrature dimensions before constructing JIT closures.
- Use resolved quadrature integers in cache keys and compiled calls.
- Apply the resolved setup behavior to external/internal on-surface B and
  GradB wrappers and batch wrappers.
- Repair `tools/profile_vc.py` so each operation receives only supported
  arguments:
  - `donate` is passed only to JIT wrappers.
  - `scan_targets` is passed only to GradB operations.
- Ensure profiling and benchmarking scripts import the working-tree package
  instead of a previously installed package.

### Internal GradB and target semantics

- Add a static
  `hedgehog_side: Literal["interior", "exterior"] = "interior"` argument to
  the singular second-derivative routines and vector wrapper.
- Retain the interior Hedgehog displacement for external GradB.
- Use the exterior Hedgehog displacement, together with the existing integral
  sign convention, for internal GradB.
- Propagate the distinction through class-based and functional APIs.
- Validate on-surface `X_trg` as a one-to-one perturbation of the configured
  target grid.
- Accept only `(3, trg_nt, trg_np)` or `(3, trg_nt * trg_np)` target layouts.
- Direct arbitrary target sets to the off-surface APIs.
- Document that the B/2 jump term remains anchored to the configured surface
  nodes.

### Precision and adaptive convergence

- Emit one `RuntimeWarning` per process when more than five digits are
  requested while calculations use float32 with JAX x64 disabled.
- Never mutate the global JAX precision configuration.
- Apply the warning to class-based, functional, on-surface, and off-surface
  calculations.
- Add `max_levels=6` to eager adaptive B and adaptive GradB entry points,
  including functional and low-level paths.
- Export `OffsurfaceConvergenceError`.
- Include requested tolerance, achieved error, final `(Nt, Np)`, and attempted
  level count in the convergence exception.
- Raise the exception when eager adaptive refinement exhausts `max_levels` or
  grid caps without convergence.
- Preserve fixed-schedule JIT semantics: schedules always terminate and
  return their best final-level result.

### Reference provenance

- Require explicit virtual-casing and Simsopt commit IDs when generating
  parity datasets.
- Recommend and document these pinned commits:
  - virtual-casing:
    `6a3898add7324125a938fded698ac145479e823e`
  - Simsopt:
    `377cf665158f47a9bed4a8b03a00352457ea27c8`
- Record the generator commit, dirty-diff hash, command, UTC generation time,
  environment, instrumentation patch hash, file sizes, and SHA-256 checksums.
- Record only files actually regenerated during a run.
- List existing, non-regenerated files as unverified legacy files rather than
  assigning inferred provenance.
- Do not use the historical internal-GradB dump as an independent oracle,
  because it encodes the sign-only implementation being corrected.

## Correctness Test Plan

- Retain C++ parity tests for:
  - external and internal B;
  - external GradB;
  - off-surface B and GradB;
  - surface utilities;
  - the Simsopt-compatible adapter.
- Validate corrected internal GradB independently with analytic synthetic loop
  fields:
  - compare internal GradB with its analytic component;
  - compare `GradBext + GradBint` with analytic total GradB;
  - check vacuum-field tensor symmetry;
  - check the zero-trace/divergence identity.
- Compare corrected on-surface internal GradB with analytic field evaluations
  approaching the surface from the exterior side.
- Add regressions for:
  - JIT-first followed by eager evaluation;
  - reusing an object after `setup()` with different geometry;
  - default `quad_nt=None` and `quad_np=None` on all four on-surface JIT
    wrappers;
  - eager/JIT and explicit/automatic quadrature agreement;
  - finite reverse-mode derivatives;
  - JVP/VJP duality;
  - invalid on-surface target counts;
  - x64-disabled warning behavior;
  - deterministic adaptive failure diagnostics;
  - bounded singular-cache size.
- Run fast regressions and small parity cases in ordinary x64 CI.
- Keep large W7-X, Simsopt, and analytic internal-GradB cases in the
  scheduled/manual large-test workflow.
- Gate CI only on correctness, not timing.

## Benchmark and Acceptance Plan

- Add a reproducible benchmark driver with `quick` and `full` suites.
- Run every benchmark case in an isolated subprocess.
- Cover:
  - external and internal B;
  - external and internal on-surface GradB;
  - external and internal off-surface B;
  - external off-surface GradB;
  - eager and supported JIT paths;
  - automatic and stored-reference quadrature sizes;
  - chunked and `scan_targets` GradB;
  - small axisymmetric, large axisymmetric, W7-X, and Simsopt cases.
- Synchronize every timed result with `jax.block_until_ready`.
- Record:
  - first-call compile-plus-execute time;
  - steady-state median and p95 over five repetitions;
  - individual timing samples;
  - peak process RSS;
  - output finiteness;
  - maximum absolute error;
  - scaled maximum relative error;
  - normalized L2 error;
  - Python, package, JAX, JAXlib, and NumPy versions;
  - platform, backend, and device;
  - x64 state;
  - git commit and dirty state;
  - dataset provenance status.
- Produce dated machine-readable JSON and Markdown summaries.
- Keep machine-specific benchmark result directories untracked by default.
- Require CPU quick and full correctness matrices to pass.
- Run the same suite on GPU when available and explicitly mark unavailable GPU
  coverage as skipped.
- Do not introduce hard performance thresholds in CI.

## Documentation Plan

- Update the README and the validation, GradB, off-surface, functional,
  performance, and parity-dataset documentation.
- Explain the internal/external one-sided Hedgehog limits.
- Explain the on-surface `X_trg` contract.
- Document the float32 precision warning.
- Document bounded eager refinement and `OffsurfaceConvergenceError`.
- Document the profiling and benchmark commands and output formats.
- Document provenance generation and legacy-data treatment.
- Leave `REVIEW_REPORT.md` unchanged.

## Implementation Record

The plan was implemented in the working tree. Major implementation locations
include:

- `virtual_casing_jax/kernels.py`
- `virtual_casing_jax/singular_quadrature.py`
- `virtual_casing_jax/integrals.py`
- `virtual_casing_jax/virtual_casing.py`
- `virtual_casing_jax/functional.py`
- `tools/benchmark_vc.py`
- `tools/profile_vc.py`
- `tools/make_parity_dumps.py`
- `tests/test_regression_lifecycle.py`
- `tests/test_virtual_casing_gradbint_parity.py`

### Validation results

- Non-large test suite: **82 passed**
- Large test suite: **19 passed**
- Total: **101 passed**
- Documentation: Sphinx build passed with warnings treated as errors
- Profiling smoke tests: on-surface B and GradB JIT paths passed
- CPU quick benchmark: all 11 cases passed
- CPU full benchmark: all 19 cases passed
- GPU benchmark: skipped because no GPU backend was selected

Independent analytic internal-GradB validation produced:

- internal GradB relative error of approximately **0.4%**;
- total decomposition relative error of approximately **0.75%**;
- final exterior one-sided disagreement below **1%**.

The final full CPU benchmark report is:

- `benchmark_results/20260723T193437Z_full.md`
- `benchmark_results/20260723T193437Z_full.json`

### Intentional limitations

- The existing parity dumps were not regenerated because their installed
  reference binary could not be tied reliably to the pinned source commits.
  They remain labeled `legacy-unverified`; no provenance was fabricated.
- The corrected internal GradB benchmark records finiteness and performance,
  but does not compare against the historical internal-GradB dump. Its
  correctness gate is the independent analytic large test.
- GPU execution remains optional and was not available for this validation
  run.

## Assumptions Applied

- Internal GradB is implemented correctly rather than disabled.
- High-accuracy float32 use warns rather than fails.
- Exhausted eager adaptive refinement raises with diagnostics.
- CPU validation is mandatory and GPU validation is optional.
- Stored C++ arrays validate only supported reference paths.
- Analytic fields, physical identities, and one-sided limits are authoritative
  for internal GradB.

## Upstream 0.0.3 Integration — July 30, 2026

The local implementation was compared with upstream
`uwplasma/virtual_casing_jax` version 0.0.3 at commit
`7e6ec5c5df6f16a79a6ca382f6b67551e59a085d`. The following upstream
capabilities were ported while retaining the correctness remediation above:

- `PrecisionPlan` and fixed-precision differentiation with respect to surface
  geometry;
- stop-gradient handling of the discrete surface-orientation sign;
- the VMEC exterior-field data model and high-level field wrapper;
- Cartesian/cylindrical conversion, cylindrical grid export, and NetCDF
  writers;
- smooth exterior-field objective helpers;
- the legacy `vmec_jax` bridge, with VMEX documented as the supported current
  integration;
- focused numerical tests, API documentation, a current VMEX example, package
  version 0.0.3, and expanded Python/CI coverage.

The upstream sign-only internal-GradB implementation, unbounded/tracer-bearing
cache behavior, inconsistent custom-JVP primal, silent adaptive
nonconvergence, and invalid `GradBext + GradBint == 0` test assumption were not
adopted. A stale manual workflow that depended on the removed
`vmec_jax.load_example` API was also excluded.

Post-integration validation:

- non-large suite: **133 passed, 1 optional test skipped**;
- large analytic/parity suite: **19 passed**;
- documentation: Sphinx passed with warnings treated as errors;
- CPU quick benchmark: **11 of 11 correctness cases passed**;
- current VMEX successfully imports the local `VmecSurfaceFieldData` and
  `VirtualCasingExteriorField` types.
