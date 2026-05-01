Differentiability
=================

The exterior-field core is written in JAX and is intended for smooth
optimization objectives. Differentiability applies to the numerical field
evaluation and smooth reductions of that field. It does not automatically apply
to discontinuous diagnostics such as hard wall hits, topology changes, or
Poincare point counting.

Differentiable paths
--------------------

The following operations are designed to participate in JAX transforms when
their inputs are JAX arrays and static shapes are fixed:

* off-surface ``B_plasma_xyz``;
* off-surface ``gradB_plasma_xyz`` where supported by the underlying schedule;
* cylindrical coordinate conversion;
* smooth objectives such as normal-field and magnitude mean-square residuals.

For VMEC-coupled differentiation, use the ``vmec_jax`` state/static path and
avoid hidden Python-side resampling. Fixed source grid shapes are part of the
compiled function contract.

Diagnostics only
----------------

The following should be treated as diagnostics unless a smooth surrogate is
explicitly introduced:

* Poincare plot topology;
* wall-hit or loss events;
* hard connection-length cutoffs;
* adaptive Python loops whose grid shapes change dynamically.

Recommended gradient checks
---------------------------

For new public APIs, compare JAX JVP/VJP or ``jax.grad`` results against finite
differences for:

* one boundary Fourier coefficient;
* one coil current or curve parameter;
* one target coordinate;
* one smooth exterior-field objective.

The comparison case should keep all JIT shapes fixed and should use the
internal virtual-casing branch for plasma exterior fields.
