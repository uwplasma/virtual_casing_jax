The density-space contract and sign conventions
===============================================

What may be sampled
-------------------

The data are the *total* field on the surface, sampled as Cartesian components. The code integrates
the surface charge ``sigma = B . n`` and the surface current ``K = B x n``, so the unit normal's own
spectrum enters the integrand. Two cases:

* **Cartesian components of a field** that are band-limited below the Nyquist index of the source grid
  (any smooth field, including one derived from an equilibrium): supported up to the Nyquist index minus
  one, with the accuracy of the requested ``digits``.
* **A trace built from a scalar**, ``B = s n`` (pure charge) or ``B = s e_theta`` (pure current): the
  normal or tangent adds an infinite geometric tail, so the trace is not band-limited whatever ``s`` is.
  Keep the scalar's poloidal wavenumber ``m <= ntheta / 4`` and toroidal wavenumber per period
  ``k <= nphi / 4``, half the Nyquist index. Above that the error grows, and the returned error estimate
  flags it (it reports validity rather than certifying accuracy: it is large, of order ``0.1`` to ``1``,
  for data with energy in the top quarter of the spectrum).

When in doubt, compare ``estimate`` with ``10**-digits``: a target whose estimate exceeds it should not be
trusted, and a finer sampling of the boundary field is the remedy.

Branches and signs
------------------

For a total field ``B0`` on a closed surface, ``external`` is the field of the sources outside the
surface and ``internal`` the field of the sources inside it, so ``external + internal = B0`` at every
point of the surface.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - quantity
     - definition
   * - ``compute_external_B``
     - limit, from inside, of the field of the sources outside; ``PV + B0/2``
   * - ``compute_internal_B``
     - limit, from outside, of the field of the sources inside; ``B0/2 - PV``
   * - ``compute_principal_value_B``
     - ``PV = (external - internal) / 2``, one quadrature pass; the average of the two one-sided limits
       of the *total* field
   * - ``B_plasma_xyz(branch="internal")``
     - off the surface: the field of the sources inside (the plasma) at targets outside; at a target inside
       it returns minus the field of the sources outside
   * - ``B_plasma_xyz(branch="external")``
     - off the surface: the field of the sources outside at targets inside

The jump between the two one-sided limits is ``B0``; off the surface there is no jump term. A constant
``B0`` has only outside sources: ``external = B0``, ``internal = 0``, ``PV = B0/2``.

In the layer-potential convention ``V^{+-} = PV V +- b/2`` (``V^+`` the limit from outside of the field
of the inside sources, ``b`` the trace), ``V^+`` is ``compute_internal_B`` and ``V^-`` is
``-compute_external_B``, so ``PV V = -compute_principal_value_B``. Negate the method's output to compare with
``PV V``.

The surface normal is always outward, chosen from the geometry (the sign of the outward direction at the
point of largest ``x``), so reversing the poloidal parametrisation does not swap the sides. Surfaces
for which that heuristic is not the outward direction are not supported.

On the surface, values and gradients are available; higher derivative orders only off the surface
(``B_and_derivatives_accurate_xyz``), including targets a micrometre away.
