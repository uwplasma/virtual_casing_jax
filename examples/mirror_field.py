"""Field of the sources inside a closed mirror surface (a cylinder with flat end caps), with a known answer.

A dipole on the axis is the source inside; another dipole outside the cylinder is the source outside.  The
total field on the wall and caps is the only input.  ``branch="internal"`` returns the inside source's field at
targets outside; ``branch="external"`` the outside source's field at targets inside.

    python examples/mirror_field.py
"""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from virtual_casing_jax.mirror import MirrorExteriorField, mirror_panels

Z0, Z1 = -1.0, 1.0
INSIDE, OUTSIDE = (jnp.array([0.0, 0.0, 0.1]), jnp.array([0.0, 0.0, 1.0e3])), (jnp.array([2.5, 0.0, 0.3]), jnp.array([1.0e3, 0.0, 0.0]))


def dipole(x, source):
    centre, moment = source
    r = x - centre
    d = jnp.linalg.norm(r)
    return 1e-7 * (3 * r * (r @ moment) / d**5 - moment / d**3)


total = lambda x: dipole(x, INSIDE) + dipole(x, OUTSIDE)
radius = lambda z, v: 1.0 + 0 * z + 0 * v
wall = lambda z, v: jnp.stack((radius(z, v) * jnp.cos(v), radius(z, v) * jnp.sin(v), z + 0 * v))
cap = lambda z: lambda w, v: total(jnp.stack((w * radius(z, v) * jnp.cos(v), w * radius(z, v) * jnp.sin(v), z + 0 * v)))
field = MirrorExteriorField(mirror_panels(radius, Z0, Z1, lambda z, v: total(wall(z, v)), cap(Z0), cap(Z1)))
print("net flux / total |flux| (zero for a solenoidal trace):", float(field.flux_defect()))

# targets outside the wall, above a cap, next to the rim, and a micrometre from the wall
outside = jnp.array([[1.5, 0.3, 0.2], [0.2, 0.1, 1.5], [1.0 + 1e-3, 0.0, 1.0 + 1e-3], [1.0 + 1e-6, 0.0, 0.2]])
inside = jnp.array([[0.3, 0.2, 0.1], [0.5, 0.0, 0.6], [1.0 - 1e-3, 0.0, 0.2]])
for name, points, branch, truth in (("outside", outside, "internal", jax.vmap(lambda x: dipole(x, INSIDE))(outside)),
                                    ("inside", inside, "external", jax.vmap(lambda x: dipole(x, OUTSIDE))(inside))):
    (B, gradB), estimate = field.B_and_derivatives_xyz(points, order=1, branch=branch)
    error = jnp.max(jnp.abs(B - truth), axis=1) / jnp.max(jnp.abs(truth))
    print(f"{name}: error {np.asarray(error)}  estimate {np.asarray(estimate)}")
    assert float(error.max()) < 1e-6
