# Technical Review of `virtual_casing_jax`

**Report date:** July 22, 2026  
**Reviewer:** OpenAI Codex  
**Local revision reviewed:** `cc10e21e66c6568445a594fa0b492a4a01671c9a` (`v0.0.2`)  
**Upstream reference revision:** `6a3898add7324125a938fded698ac145479e823e`  
**Reference repository:** [hiddenSymmetries/virtual-casing](https://github.com/hiddenSymmetries/virtual-casing)

## Executive summary

The eager, float64 **external-field** implementation is a credible and substantially correct JAX port of the reference virtual-casing algorithm for the cases covered by the repository. Surface completion, Fourier resampling, normal construction, Laplace/Biot-Savart kernels, on-surface singular correction, off-surface quadrature, and the SIMSOPT adapter all show strong forward-value parity.

The package as a whole should not yet be considered fully correct. Four high-priority defects affect advertised or public workflows:

1. On-surface reverse-mode geometry differentiation returns NaNs.
2. A JIT-first call can poison the global singular-quadrature cache with tracers.
3. JIT functions can silently reuse geometry from an earlier `setup()` call.
4. On-surface internal `GradB` is implemented as a simple sign flip even though the cited upstream implementation explicitly says the opposite Hedgehog limit is required.

Accordingly, the current version is suitable for carefully controlled eager external-field calculations on validated grids. It should not be relied upon for on-surface reverse-mode shape optimization, repeated JIT setup with changing surfaces, or physical on-surface internal-`GradB` calculations until the issues below are resolved.

## Scope and methodology

The review covered:

- `virtual_casing_jax/virtual_casing.py`
- `virtual_casing_jax/functional.py`
- `virtual_casing_jax/integrals.py`
- `virtual_casing_jax/kernels.py`
- `virtual_casing_jax/singular_quadrature.py`
- `virtual_casing_jax/surface_ops.py`
- `virtual_casing_jax/simsopt_virtual_casing.py`
- Tests, documentation, examples, profiling utilities, and stored parity datasets
- The corresponding upstream C++ API and implementation at the reference revision above

Review activities included:

- Direct comparison of formulas, signs, layouts, grid conventions, and target selection with upstream C++.
- Running the complete local test suite in the existing SIMSOPT environment.
- Focused reproductions for reverse-mode autodiff, first-call JIT behavior, JIT reuse after `setup()`, float32 behavior, and internal-`GradB` semantics.
- Inspection of test coverage and parity-dataset provenance.

The tests were run with Python 3.11.14 and JAX 0.7.2. The repository's `tests/conftest.py` enables JAX x64 for pytest.

## Verification results

All 94 collected tests passed:

- 76 non-large tests
- 10 large core parity tests
- 8 large SIMSOPT/VMEC integration tests

The passing coverage includes:

- Surface operators and geometry assets
- Kernel formulas and normalization
- Singular and non-singular integral operators
- External and stored internal `B` parity
- External and stored internal `GradB` parity
- Off-surface `B` and `GradB`
- JIT and batch wrappers under the test suite's call ordering
- Functional forward-mode JVPs
- SIMSOPT VMEC initialization, BNORM comparison, stellarator symmetry, vacuum behavior, save/load, and plotting

Stored end-to-end parity tolerances are approximately:

- External `B`: relative L2 error from `3e-4` to `1.2e-3`
- External `GradB`: relative L2 error from `5e-3` to `3e-2`

These are meaningful positive results, but the current tests do not exercise the failure modes described below.

## Detailed findings

### F-1 — High: reverse-mode on-surface geometry autodiff returns NaNs

The package documentation advertises end-to-end differentiability and gives a `jax.grad` example for `compute_external_B_functional`. That workflow fails for on-surface evaluation.

The root problem begins in [`kernels.py`](virtual_casing_jax/kernels.py), where `_safe_rinv` is implemented conceptually as:

```python
jnp.where(r2 > eps, 1.0 / jnp.sqrt(r2), 0.0)
```

At a self interaction, the primal result is masked to zero, but automatic differentiation still encounters the undefined derivative of the inactive `1/sqrt(0)` expression. A direct kernel diagnostic produced finite primal values and NaN reverse-mode gradients. Applying `jax.grad` to an on-surface functional objective produced 60 NaNs out of 60 geometry-gradient entries.

Why tests miss it:

- The on-surface functional tests use forward-mode `jax.jvp`, not reverse-mode `jax.grad`.
- The existing reverse-mode functional test is off-surface, where `r != 0` and the kernels are smooth.

Recommendations:

1. Form a safe denominator before applying `rsqrt`, for example by replacing masked `r2` values with a finite constant, and only then mask the result.
2. Define explicit JVP/VJP behavior for excluded self interactions if necessary.
3. Audit all singular-kernel uses, not only `_safe_rinv`, for masked invalid arithmetic.
4. Add reverse-mode regression tests for both `compute_external_B_functional` and the normal-field functional.
5. Require all returned geometry gradients to be finite before comparing them to finite differences.

### F-2 — High: JIT-first invocation can leak tracers into a global cache

[`singular_quadrature.py`](virtual_casing_jax/singular_quadrature.py) decorates `precompute_singular` with an unbounded `functools.lru_cache`. The cached object contains JAX arrays created by `jnp.asarray`.

If a JIT wrapper is the first operation that requests a particular singular table, table construction occurs while JAX is tracing. The cache then retains traced arrays outside the trace. A subsequent eager evaluation fails with `jax.errors.UnexpectedTracerError`.

This was reproduced in a fresh Python process by:

1. Constructing and setting up `VirtualCasingJAX`.
2. Calling `compute_external_B_jit` before any eager singular evaluation.
3. Calling `compute_external_B` afterward.

Why tests miss it:

- `tests/test_virtual_casing_jit_batch.py` calls eager `compute_external_B` before `compute_external_B_jit`, which warms the global cache with ordinary device arrays.

Recommendations:

1. Cache NumPy host tables rather than traced JAX values, or construct/cast the JAX tables explicitly outside all JIT traces.
2. Resolve and close over the complete precomputed table in each JIT wrapper before calling `jax.jit`.
3. Add isolated subprocess tests in which each JIT API is the first package operation.
4. Consider a bounded cache because the present cache has `maxsize=None` and dtype/order combinations can accumulate indefinitely.

### F-3 — High: JIT cache is not invalidated by `setup()`

`VirtualCasingJAX.setup()` resets `_b_setup` and `_grad_setup`, but it does not clear `_jit_cache`. The cached JIT functions close over geometry arrays that were captured during their original trace.

Reproduction:

1. Set up a torus with minor radius `0.3` and evaluate `compute_external_B_jit`.
2. Call `setup()` on the same object with minor radius `0.5`, keeping array shapes and JIT options unchanged.
3. Evaluate `compute_external_B_jit` again.

The second JIT result was exactly equal to the old-surface result (`max(abs(new_cached - old)) == 0`) and disagreed with an eager evaluation using the new surface.

This is a silent correctness failure, not merely a recompilation or performance issue.

Recommendations:

1. Clear `_jit_cache` in every successful `setup()` call.
2. Prefer pure functions whose geometry is an explicit argument, or include a setup-generation token in cache keys.
3. Key caches using resolved quadrature sizes rather than the possibly `None` user arguments.
4. Add regression tests that reuse one object across two geometrically distinct surfaces with identical shapes.

### F-4 — High: on-surface internal `GradB` does not implement the required limit

The local `_compute_gradB_signed` constructs one hypersingular result and returns it multiplied by `sign`. Thus:

```text
compute_internal_gradB(B) == -compute_external_gradB(B)
```

for every input.

The cited upstream code does not expose internal on-surface GradB. Its header comments out `ComputeGradBint` with the explanation that Hedgehog quadrature requires the normal to be flipped. The implementation also asserts the external path. See:

- [Upstream declaration and warning](https://github.com/hiddenSymmetries/virtual-casing/blob/6a3898add7324125a938fded698ac145479e823e/include/virtual-casing.hpp#L182)
- [Upstream GradB implementation](https://github.com/hiddenSymmetries/virtual-casing/blob/6a3898add7324125a938fded698ac145479e823e/include/virtual-casing.txx#L188-L257)

A simple sign change does not select the opposite one-sided Hedgehog limit. It also forces the computed external and internal gradients to sum identically to zero, whereas physical fields satisfy:

```text
GradB_total = GradB_external + GradB_internal
```

and `GradB_total` is not generally zero.

The internal-GradB parity fixtures do not establish correctness against the cited upstream project because upstream has no such public method. The repository's dump-generation tool calls methods that are absent from an unmodified upstream Python extension, indicating that these fixtures came from a local extension or patch that is not identified in the dataset metadata.

Recommendations:

1. Temporarily remove, disable, or explicitly mark on-surface internal GradB as unsupported.
2. Implement the opposite Hedgehog limit by flipping the appropriate target-normal displacement/orientation, following the upstream warning.
3. Validate against independently generated physical internal-current gradients, not only against a patched sign-flip implementation.
4. Add the identity check `GradBext + GradBint ~= GradBtotal` on analytic or synthetic-loop cases.

### F-5 — Medium: `X_trg` semantics in the on-surface API are unsafe

`compute_external_B` and `compute_internal_B` accept `X_trg`, suggesting arbitrary target support. Internally, however, the singular-patch indices, output reshape, and `B/2` jump term remain tied to configured on-surface target-grid topology.

Consequences:

- A different target count leads to shape failures.
- An arbitrary target set can receive a mathematically inappropriate jump term.
- Patch centers are based on parameter-grid indices, not a nearest-point relation for arbitrary coordinates.

Recommendations:

1. Rename this argument to make clear that it represents perturbations of the configured on-surface target grid.
2. Validate an exact shape of `(3, trg_nt, trg_np)` for the singular on-surface path.
3. Direct arbitrary targets to `compute_external_B_offsurf` or `compute_internal_B_offsurf`.

### F-6 — Medium: default adaptive off-surface refinement is unbounded

`_offsurface_adapt_grid` doubles both source-grid dimensions until the double-layer self-test reaches tolerance. The public defaults `max_Nt=-1` and `max_Np=-1` impose no cap.

For a target on the surface, the self-test is not expected to approach the off-surface indicator value used by the stopping criterion. Very near-surface targets can also demand extreme resolution. The loop can therefore grow until memory exhaustion.

Recommendations:

1. Add a `max_levels` limit to the non-scheduled path.
2. Use finite, documented default caps based on initial resolution or memory budget.
3. Detect targets on or numerically indistinguishable from the surface and issue a clear error directing users to the on-surface evaluator.
4. Report non-convergence instead of silently returning a capped result.

### F-7 — Medium: accuracy promises depend on x64, but x64 is not enforced

All pytest parity results run with `jax_enable_x64=True` because `tests/conftest.py` sets it globally. A normal JAX process defaults to x64 disabled, including the reviewed SIMSOPT environment outside pytest.

The public `digits` parameter can therefore request accuracy that the active dtype cannot plausibly deliver. NumPy float64 input supplied by SIMSOPT is silently represented as float32 when x64 is disabled.

A small external-`B` parity case still achieved about `1.98e-4` relative L2 error in float32, so float32 is useful; the concern is that `digits` no longer has the documented meaning, especially for higher requested accuracy and hypersingular GradB.

Recommendations:

1. Check `jax.config.x64_enabled` and the input dtype at public entry points.
2. Warn or reject when requested `digits` exceed a documented float32 capability.
3. State the x64 requirement prominently in basic usage and the SIMSOPT wrapper documentation.
4. Add a separate float32 test matrix with explicit relaxed tolerances.

### F-8 — Low: custom-JVP primal options are inconsistent

In `compute_external_B_autodiff`, the ordinary custom-JVP primal call does not forward all exposed options, notably `patch_dtype` and `interp_block_size`, while the JVP rule's primal calculation does.

This can make a plain call and the primal returned by a transformed `jax.jvp` call differ when non-default options are supplied.

Recommendation: construct one shared options dictionary and use it for both primal evaluations.

### F-9 — Low: parity datasets lack sufficient provenance

The JSON sidecars record only dtype and shape. They do not record:

- Upstream repository URL and commit
- BIEST/SCTL revisions
- Compiler and precision
- Generator command
- Local patches or added C++/Python methods
- Error metric and intended tolerance

This is particularly important for internal GradB and off-surface GradB, which are described in documentation as local parity extensions or rely on APIs absent from upstream.

Recommendation: add a manifest containing immutable source revisions, patch hashes, build settings, generator commands, and per-dataset purpose/tolerances.

## Positive technical observations

The following aspects appear well implemented:

- Structure-of-arrays conventions consistently match the C++ interface.
- Half-period stellarator symmetry and toroidal shifts follow the upstream setup logic.
- Fourier resampling and spectral derivatives match stored reference data.
- Surface normal orientation and area weights are consistent with the parity cases.
- Laplace and Biot-Savart kernels use the expected `1/(4*pi)` normalization.
- External on-surface `B` assembly correctly includes the `B/2` jump term.
- External GradB assembly follows the upstream curl/Hessian structure.
- Source and target blocking substantially reduce direct-sum peak memory.
- Off-surface field and gradient paths are separated appropriately from singular on-surface quadrature.
- The functional API is a useful architectural direction because it exposes geometry as a JAX input rather than hiding it in mutable cached state.
- Documentation is unusually thorough for a young numerical package, even though several claims need narrowing as noted above.

## Recommended remediation order

### Release-blocking

1. Fix differentiably safe zero-distance kernels and add reverse-mode tests.
2. Prevent tracer values from entering global caches.
3. Invalidate all object JIT caches when `setup()` changes state.
4. Disable or correctly implement internal on-surface GradB.

### Next priority

5. Clarify and validate on-surface `X_trg` semantics.
6. Bound adaptive off-surface refinement.
7. Add explicit dtype/x64 policy and tests.
8. Make custom-JVP option forwarding consistent.

### Validation and maintenance

9. Add provenance manifests for stored parity data.
10. Add independent physical validation for internal GradB.
11. Add isolated subprocess tests for JIT-first behavior.
12. Add setup-reuse tests for every cached JIT wrapper.
13. Add convergence studies across `digits`, quadrature resolution, distance to surface, aspect ratio, and float precision—not only fixed parity points.

## Suggested acceptance criteria

A subsequent release should satisfy at least the following:

- `jax.grad` of documented on-surface functional examples is finite and agrees with directional finite differences.
- Every JIT wrapper works as the first package call in a fresh process.
- Reusing one object after `setup()` produces the same result as a newly constructed object for the new geometry.
- Internal GradB either raises `NotImplementedError` or passes independent physical tests and the total-gradient identity.
- Arbitrary on-surface/off-surface target misuse produces explicit validation errors.
- Adaptive off-surface evaluation terminates predictably under all public default settings.
- Test reports clearly distinguish float32 and float64 accuracy.
- Every stored parity dataset is tied to immutable, reproducible source and patch revisions.

## Final assessment

The external eager implementation is a strong foundation and demonstrates real numerical parity with the reference project. The main problems are concentrated in newer JAX-specific extensions—reverse-mode differentiation, mutable JIT caching, and unsupported internal hypersingular limits—rather than in the basic external virtual-casing formula.

With the four release-blocking items corrected and the validation gaps closed, this package could become a reliable JAX replacement for the relevant external-field portion of the SIMSOPT virtual-casing workflow.

---

# Verification addendum

**Addendum date:** July 23, 2026
**Reviewer:** Claude (Fable 5, Anthropic) — independent corroboration of the Codex report above; everything below this horizontal rule is Claude's, everything above is Codex's.
**Local revision reviewed:** `cc10e21e66c6568445a594fa0b492a4a01671c9a` (`v0.0.2`), same as above
**Environment:** `simsopt_env` conda environment, Python 3.11.14, JAX 0.7.2 (same as the original review); reproductions also exercised under JAX 0.10.0

## Verdict on the Codex report

The report is **correct** and its evidence held up under independent re-verification. It is **nearly complete**; one additional high-severity defect was found (C-1 below).

## Verification performed

- Re-read all source files in scope and matched every code-level claim to the actual code.
- Re-ran the reproductions for F-1, F-2, and F-3 from scratch (fresh processes, independently written scripts on a simple torus geometry).
- Fetched the upstream `include/virtual-casing.hpp` and `src/python.cpp` at revision `6a3898add7324125a938fded698ac145479e823e` to check the F-4 upstream claims directly.
- Re-ran the test suite: 94 tests collect (76 non-large + 18 large, matching the report); all 76 non-large tests pass in `simsopt_env` (2m29s). The large tests were not re-run here; the report states they pass.

## Per-finding corroboration

| Finding | Status | Evidence |
| --- | --- | --- |
| F-1 | **Confirmed, reproduced** | `_safe_rinv` (`kernels.py:9`) is the known `jnp.where` autodiff trap. Kernel primal finite, gradient NaN at `dx=0`. `jax.grad` of an on-surface `compute_external_B_functional` objective: 60/60 geometry-gradient entries NaN (matches report). The sole `jax.grad` test (`test_functional_grad_wrt_surface`) is off-surface, as stated. `docs/functional.rst` showcases exactly the broken workflow. |
| F-2 | **Confirmed, reproduced** | Unbounded `lru_cache` at `singular_quadrature.py:107` caches `jnp.asarray` results. Fresh process, JIT-first then eager: `UnexpectedTracerError`, as described. The jit-batch test warms the cache eagerly first, which is why the suite misses it. |
| F-3 | **Confirmed, reproduced** | `setup()` (lines 177–178) never clears `_jit_cache`; jitted lambdas close over `self`. After re-`setup()` with a different minor radius: JIT result bit-identical to the old surface (max diff 0.0), disagrees with eager by ~4.8e-2. Cache keys use raw, possibly-`None` `quad_nt`/`quad_np` (line 831), as noted. |
| F-4 | **Confirmed** | `_compute_gradB_signed` ends in `return gradBvc * sign` (line 453), so internal ≡ −external identically. Upstream header at the cited revision comments out `ComputeGradBint` with "TODO: requires Hedgehog quadrature normal to be flipped"; upstream `src/python.cpp` exposes no internal-gradB binding, yet `tools/make_parity_dumps.py:99` calls `vcasing.compute_internal_gradB(...)` — so the fixtures came from a patched extension, as inferred. The only internal-GradB test compares against that patched dump (circular). Contrast: for `B` the sign flip *is* correct because the `+B/2` jump is added outside the sign (line 733), consistent with `Bint = B − Bext`; GradB has no analogous structure. |
| F-5 | **Confirmed** | `_compute_B_signed` reshapes results to the configured `(3, trg_nt, trg_np)` unconditionally (line 695) and always adds the on-grid `0.5 * B_on` jump term (lines 718–733) regardless of `X_trg`. |
| F-6 | **Confirmed** | `_offsurface_adapt_grid` is `while True` with uncapped `max_Nt=-1`/`max_Np=-1` defaults. For an on-surface target the double-layer self-test gives `U ≈ 0.5`, so `err = min(|1−U|, |U|) ≈ 0.5` can never meet tolerance → unbounded doubling. |
| F-7 | **Confirmed** | x64 is enabled only in `tests/conftest.py`; no enforcement, check, or warning anywhere in package code (only advisory notes in `docs/performance.rst`). The float32 spot-check value was not re-measured. |
| F-8 | **Confirmed** | The custom-JVP primal `_eval` (`virtual_casing.py:1030–1041`) omits `patch_dtype` and `interp_block_size`; the JVP rule's primal (lines 1047–1059) forwards them. Note the method lives in `virtual_casing.py`, not `functional.py`. |
| F-9 | **Confirmed** | JSON sidecars contain literally only `dtype` and `shape` (verified on `case_simsopt_int_computeGradB_J.json`). |

## Additional finding not in the Codex report

### C-1 — High: JIT wrappers crash with default `quad_nt`/`quad_np`

Calling any `*_jit` wrapper without explicit quadrature sizes (the public default, `quad_nt=None, quad_np=None`) raises `jax.errors.ConcretizationTypeError` at trace time.

Mechanism: the wrapper resolves the setup eagerly (e.g. `compute_external_B_jit`, line 814), but the traced lambda re-enters `compute_external_B` → `_compute_B_signed` → `_ensure_b_setup(None, None, digits)` (line 622). Because the user arguments are still `None`, the `quad_nt is None` branch re-runs `_select_quad_sizes` *inside the trace*, and that function calls `float(jnp.min(ratio))` (line 197) on values that are tracers under the JIT trace. The `_ensure_*_setup` early-return never fires because it compares resolved integers against `None`.

Reproduced under both JAX 0.10.0 and confirmed structurally for 0.7.2 (`float()` on a tracer has always raised). This occurs even with the eager-first call ordering that avoids F-2, so it is a distinct defect: a public API crashes under its own default arguments.

Why tests miss it: every JIT-wrapper test passes explicit `quad_nt`/`quad_np`.

Recommendations:

1. Resolve quadrature sizes once, eagerly, in the wrapper, and pass the resolved integers down so the traced path never consults `_select_quad_sizes` (this also implements Codex recommendation F-3.3, keying caches on resolved sizes).
2. Add JIT-wrapper tests that use default arguments.

This belongs with F-2/F-3 in the release-blocking cluster; all three share one root cause — mutable setup state resolved lazily inside traced code — and one fix direction: resolve all setup state eagerly before `jax.jit` tracing.

## Minor notes

- `compute_external_B_autodiff`'s JVP rule uses the *external* on-surface GradB as the Jacobian for target perturbations. This is defensible for the exterior limit but deserves an explicit validation note given the F-4 territory it borders.
- The SIMSOPT adapter (`simsopt_virtual_casing.py`) was re-read and matches the report's positive assessment; nothing new found there.

## Final assessment

The Codex report can be trusted as written: all nine findings are real, the test counts and tolerances are accurate, and the remediation ordering is sound. The release-blocking list should be extended with C-1 (default-argument JIT crash), which is fixed most naturally together with F-2 and F-3.
