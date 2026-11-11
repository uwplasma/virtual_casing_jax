Mirror and open-ended geometries
================================

``virtual_casing_jax.mirror`` evaluates the field of the sources inside (or outside) a closed surface
that is not a torus: a side wall ``r = r(z, v)`` closed by a flat disk at each end.  The surface is
topologically a sphere, so the same boundary integral as the torus path applies,

.. math::

   \mathbf{V}(x) = \nabla G[\sigma](x) + \mathrm{BiotSavart}[\mathbf{K}](x),
   \qquad \sigma = \mathbf{B}\cdot\mathbf{n}, \quad \mathbf{K} = \mathbf{B}\times\mathbf{n},

where ``B`` is the *total* field on the surface.  ``V`` is the field of the sources inside at points
outside (``branch="internal"``) and, with its sign reversed, the field of the sources outside at points
inside (``branch="external"``); the two sum to ``B`` on the surface.

Conventions
-----------

* The surface is three patches ``X(u, v)`` (lower cap, side wall, upper cap), ``u`` in ``[0, 1]``, ``v``
  periodic, with outward normal along ``X_u x X_v``.  :func:`~virtual_casing_jax.mirror.mirror_panels`
  builds them from a star-shaped radius ``radius(z, v)`` and the traces of ``B`` on the wall and caps.
* The through-flux ``B . n`` of each cap is part of the data.  A cap whose flux is omitted or zeroed gives
  a wrong field; :meth:`~virtual_casing_jax.mirror.MirrorExteriorField.flux_defect` reports the net flux
  relative to the total absolute flux, which is zero (to rounding) for a solenoidal field and of order
  one when a cap flux is missing.
* Rims (where wall and cap meet) are boundaries between patches, so no patch integrand has a corner.
  ``B`` may differ between patches at a rim.  A cap centre is an axis, where ``X_v = 0``; the weight
  ``|X_u x X_v|`` goes to zero there and the point is regular.

Numerics
--------

Each patch uses Gauss-Legendre in ``u`` and the trapezoid rule in ``v``.  A target within a few node
spacings of a patch uses the rule graded about its nearest surface point (geometric Gauss-Legendre panels
in ``u``, the ``v - a sin v`` substitution in ``v``), as the torus path does; a target closer than
``5e-4`` of the half-extent is continued along the normal with the closed-form jets.  Derivatives up to
order three are available and differentiable with respect to the surface data and the sources.

Accuracy (cylinder of radius 1, length 2, circular-loop sources, error relative to the maximum source
field): far from the surface about ``1e-11``; approaching the wall or a cap from outside down to
``1e-12`` of the radius, value error below ``1e-10`` and gradient error below ``1e-7``; approaching a rim
from outside, value error ``1e-7`` and gradient error ``4e-7``.  The returned estimate bounds these in
the ranges above.

Limits
------

* From inside (``branch="external"``), targets closer than ``1e-3`` of the radius to a rim lose accuracy
  quickly (error ``7e-8`` at ``1e-3``, ``4e-4`` at ``1e-4``, order one at ``1e-6``), and the estimate stops tracking the error below ``1e-6``.
  Targets on the cap at the axis within ``1e-9`` are likewise unreliable.  Keep interior targets away
  from the rim by more than ``1e-3`` of the radius.
* Walls that are not star-shaped about the axis, or caps that are not planar disks, are not covered
  by :func:`~virtual_casing_jax.mirror.mirror_panels`; any smooth patches with the contract of
  :class:`~virtual_casing_jax.mirror.MirrorPanel` can be passed directly.
* Only the total field on the surface is used, so any current-carrying or anisotropic-pressure equilibrium
  can supply it.  Anisotropic equilibria are future work and are not validated here.
