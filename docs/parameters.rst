Parameter reference
===================

Every public argument of the exterior-field API, with its meaning, its default and
what it trades. Defaults are chosen to give the best accuracy at the best speed, so
none of them needs to be set for a first calculation.

``VmecSurfaceFieldData``
------------------------

The boundary data, sampled on **one field period**. Arrays are structure-of-arrays,
shape ``(3, nphi, ntheta)``.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - field
     - meaning
   * - ``gamma``
     - Cartesian boundary points. ``phi`` is the physical toroidal angle (not the VMEC
       field-period coordinate ``zeta = nfp * phi``), sampled uniformly on
       ``[0, 2 pi / nfp)``; ``theta`` uniformly on ``[0, 2 pi)``.
   * - ``B_total``
     - Cartesian components of the *total* field on the boundary (all sources, inside
       and outside). See :doc:`tutorial_density_contract` for what may be sampled.
   * - ``normal``, ``area_vector``
     - Checked for shape only. The outward normal is recomputed from ``gamma``.
   * - ``theta``, ``phi``
     - The one-dimensional angle grids of the samples.
   * - ``nfp``
     - Number of field periods; the data are extended by rotation about the z axis.
   * - ``stellsym``
     - Stored for the caller. The field class does not read it and does not complete a half
       period: pass a full field period.
   * - ``signgs``, ``source_convention``
     - Bookkeeping carried from the equilibrium code.

``ExteriorFieldConfig``
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 20 18 62

   * - field
     - default
     - meaning
   * - ``digits``
     - ``8``
     - Requested accuracy ``10**-digits``, relative to the RMS boundary field. It sets
       the on-surface quadrature, the tolerance ``tol`` used to route each target to the
       closed forms or the graded rule, and the level of the error estimate you compare with.
       Raising it costs more nodes on the surface; the data resolution caps what is reachable
       (see :doc:`tutorial_near_surface`).
   * - ``src_nphi``, ``src_ntheta``
     - ``64``, ``64``
     - Not read: the grid is the shape of ``surface_data.gamma``. Set the resolution by
       sampling the boundary on the grid you want.
   * - ``levels``
     - ``None``
     - Closed-form schedule as ``((n_toroidal, n_poloidal), ...)``, counts around the
       **whole** torus, coarse to fine. ``None`` takes two levels sized from the data:
       ``(nphi * nfp, ntheta)`` and twice that. The coarse level exists for the error estimate;
       the near-surface routing needs at least two distinct levels. Toroidal entries are rounded
       up to a multiple of ``nfp``.
   * - ``chunk_size``
     - ``"auto"``
     - Sources per tile of the direct sums. ``"auto"`` is tuned per operator and backend; an
       integer fixes it. Smaller tiles lower peak memory, larger ones are faster until the
       tile no longer fits in cache or device memory.
   * - ``target_chunk_size``
     - ``"auto"``
     - Targets per tile, likewise. Also used by ``B_plasma_on_surface``.
   * - ``branch``
     - ``"internal"``
     - Which side's sources ``B_plasma_xyz`` returns. ``"internal"`` is the field of the
       sources *inside* the surface (the plasma currents), valid at targets outside it.
       ``"external"`` is the field of the sources *outside* the surface, valid at targets
       inside it (:doc:`tutorial_density_contract`). It does not mean "targets outside".
   * - ``use_jit_schedule``
     - ``True``
     - Use the fixed-shape schedule (jit-compatible, returns the estimate). ``False`` runs the
       older adaptive Python loop; it cannot return an estimate and disables the near-surface route.
   * - ``near_surface``
     - ``True``
     - Send targets whose closed-form estimate exceeds ``10**-digits`` to the graded rule of
       ``B_and_derivatives_accurate_xyz``. ``False`` keeps the closed forms alone, which are
       wrong by order one within a few node spacings of the surface.
   * - ``graded_nodes``
     - ``None``
     - ``(poloidal, toroidal)`` nodes of the graded rule over the whole torus. ``None`` takes twice
       the finest schedule level, at least ``256 x 512``. The distance to resolve, not the data,
       sets the need, so lowering it saves time only for targets that stay far from the surface;
       raising it buys nothing beyond the data limit.
   * - ``validate_orientation``
     - ``True``
     - Not read by the current implementation. The outward direction is always chosen from the
       geometry.
   * - ``dtype``
     - ``"float64"``
     - ``"float64"`` or ``"float32"``. Use float64 (and ``jax_enable_x64``) for ``digits`` above
       about 4; float32 limits every result to single-precision accuracy.

Evaluation methods of ``VirtualCasingExteriorField``
-----------------------------------------------------

``xyz`` is an ``(n, 3)`` array (or ``(..., 3)``) of Cartesian targets.

``B_plasma_xyz(xyz, *, branch=None, return_estimate=False)``
   Field of the sources of ``branch`` (default ``config.branch``). With ``return_estimate=True``
   also the relative error estimate per target, dimensionless and comparable with ``10**-digits``.

``gradB_plasma_xyz(xyz, *, branch=None)``
   ``dB_i/dx_j``, shape ``(n, 3, 3)`` for an ``(n, 3)`` input (element ``[t, i, j]``), routed the
   same way.

``B_xyz``, ``gradB_xyz``
   Plasma field plus the ``external_B_fn(xyz)`` / ``external_gradB_fn(xyz)`` callbacks given to
   the constructor (coils); the gradient of ``external_B_fn`` is taken by ``jacfwd`` when no
   analytic one is given.

``B_cyl(R_phi_Z)``, ``export_rphiz_grid(R, phi, Z, chunk_size="auto")``
   Cylindrical components ``(B_R, B_phi, B_Z)`` and tensor-product grid export.

``B_xyz_sharded(xyz, *, devices=None)``
   ``B_xyz`` with the target axis split over ``devices`` (default: all visible). A single device
   falls back to ``B_xyz``. See :doc:`tutorial_gpu`.

``B_and_derivatives_xyz(xyz, *, order=0, level=None, source_chunk=4096)``
   Closed-form jets up to ``order`` (at most 3) in one traversal of the sources of schedule
   level ``level`` (default the finest). Not accurate near the surface. ``source_chunk`` is the
   source tile of the sum.

``B_and_derivatives_accurate_xyz(xyz, *, order=1, tol=1e-8, route_order=None, branch=None)``
   Jets at any distance from the surface; returns ``(values, estimate)``.

   ``order``
      Highest derivative returned, 0 to 3; ``values[k]`` has shape ``xyz.shape[:-1] + (3,) * (k + 1)``.
   ``tol``
      Scaled error below which the closed forms are accepted. The ``k``-th derivative is accepted
      at ``100**k * tol``. Smaller ``tol`` sends more targets to the slower graded rule.
   ``route_order``
      Derivative orders that take part in the routing decision (default ``order``). ``0`` routes on
      ``B`` alone whatever ``order`` is, so derivatives of the order-0 call differentiate the same map.
   ``branch``
      As above.

``B_plasma_on_surface(*, digits=None, chunk_size=None, quad_nt=None, quad_np=None, precision=None, target_chunk_size=None)``
   The internal-branch field on the surface itself (the limit from outside). ``quad_nt``,
   ``quad_np`` fix the on-surface quadrature grid instead of selecting it from ``digits``;
   ``precision`` is a frozen plan (below).

``plan_surface_precision(*, digits=None, quad_nt=None, quad_np=None)``
   Freezes the geometry-dependent quadrature and singular-patch choices into a ``PrecisionPlan``
   so that on-surface calls can be differentiated with respect to the surface coordinates.

``VirtualCasingJAX``
--------------------

The on-surface operator behind the field class. ``setup(digits, nfp, half_period, surf_nt, surf_np,
X, src_nt, src_np, trg_nt, trg_np)``: requested digits, field periods, whether the data are a half
period (stellarator symmetry), the shape of the surface array ``X`` of shape ``(3, surf_nt, surf_np)``,
and the source and target grids. The ``compute_external_B``, ``compute_internal_B``,
``compute_principal_value_B`` and ``compute_*_gradB`` methods take the total field ``B0`` on the source grid
and the following keywords (others in the API reference):

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - keyword
     - meaning
   * - ``digits``
     - Override the digits of ``setup``.
   * - ``quad_nt``, ``quad_np``
     - Singular-quadrature grid; ``None`` selects it from ``digits`` by a constant-trace test.
   * - ``chunk_size``, ``target_chunk_size``
     - Source and target tiles, ``"auto"`` by default.
   * - ``precision``
     - A ``PrecisionPlan`` from ``plan_precision``; makes the call differentiable in the geometry.
   * - ``pou_dtype``, ``patch_dtype``
     - Precision of the partition-of-unity tables and singular patches (``"float32"`` or ``"auto"``
       halves the largest temporaries; outputs stay in the input precision).
   * - ``remat``
     - Rematerialize the singular correction in reverse mode (less memory, more compute).
   * - ``X_trg``
     - Target nodes for the on-surface operator, shape ``(3, trg_nt, trg_np)``; they must be the
       target grid of ``setup``. Use the off-surface API for arbitrary points.

``MirrorExteriorField`` and ``mirror_panels``
---------------------------------------------

``mirror_panels(radius, z_min, z_max, side_field, lower_field, upper_field)``
   Builds the three patches of a closed mirror surface. ``radius(z, v)`` is the star-shaped wall
   radius at height ``z`` and angle ``v``; ``side_field(z, v)`` the total field on the wall;
   ``lower_field(w, v)``, ``upper_field(w, v)`` the total field on the end caps at the fraction
   ``w = r / radius(z_end, v)`` of the rim radius. All return Cartesian 3-vectors.

``MirrorExteriorField(panels, nodes=(14, 512), plain_nodes=(32, 96), cloud=(48, 96))``
   ``nodes = (q, n)``
      Gauss-Legendre points per graded panel and nodes in ``v`` of the rule graded about a near
      target. Raise for derivative orders 2-3 very close to the surface.
   ``plain_nodes = (q, n)``
      Points per patch (``u`` Gauss-Legendre, ``v`` trapezoid) of the rule used far from the surface.
      Its coarser sibling (``q - 8``, ``n / 2``) supplies the estimate.
   ``cloud = (nu, nv)``
      Points per patch of the cloud that seeds the nearest-point search.

   ``B_and_derivatives_xyz(xyz, *, order=1, branch="internal")`` returns ``(values, estimate)`` with
   ``order`` up to 3; ``B_plasma_xyz(xyz, *, branch="internal")`` the order-0 field;
   ``flux_defect()`` the net flux over the total absolute flux (zero for a solenoidal trace).
   See :doc:`tutorial_mirror`.
