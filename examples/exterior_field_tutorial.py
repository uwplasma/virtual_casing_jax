"""Exterior field of a stellarator-like surface with a known answer: values, error estimate, jets, gradients.

A rotating-ellipse boundary (3 field periods) carries the field of three point dipoles inside it (one per
period) and of a dipole in the hole of the torus, outside it.  The internal branch must return the
first field outside the surface and minus the second one inside it, to ``10**-digits``, at any
distance from the surface.

    python examples/exterior_field_tutorial.py
"""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from virtual_casing_jax import ExteriorFieldConfig, VirtualCasingExteriorField, VmecSurfaceFieldData

NFP, NPHI, NTHETA, R0, A, DELTA = 3, 64, 96, 1.0, 0.30, 0.08
Z_MOMENT = jnp.array([0.0, 0.0, 1.0e3])
ANGLES = 2 * jnp.pi * jnp.arange(NFP) / NFP
INSIDE = jnp.stack([R0 * jnp.cos(ANGLES), R0 * jnp.sin(ANGLES), 0 * ANGLES], axis=1)  # one dipole per period
OUTSIDE = jnp.zeros((1, 3))  # in the hole of the torus


def dipole(x, centres):
    """Field of z-directed point dipoles at ``centres``: periodic under rotation by 2 pi / nfp when they are."""
    r = x[..., None, :] - centres
    d = jnp.linalg.norm(r, axis=-1, keepdims=True)
    return 1e-7 * jnp.sum(3 * r * (r @ Z_MOMENT)[..., None] / d**5 - Z_MOMENT / d**3, axis=-2)


def surface():
    """Boundary samples over one field period and the total field on them, as ``(3, nphi, ntheta)``."""
    theta = jnp.linspace(0.0, 2 * jnp.pi, NTHETA, endpoint=False)
    phi = jnp.linspace(0.0, 2 * jnp.pi / NFP, NPHI, endpoint=False)
    ph, th = jnp.meshgrid(phi, theta, indexing="ij")
    u = th - NFP * ph
    R, Z = R0 + A * jnp.cos(th) + DELTA * jnp.cos(u), A * jnp.sin(th) + DELTA * jnp.sin(u)
    gamma = jnp.stack([R * jnp.cos(ph), R * jnp.sin(ph), Z])
    x = jnp.moveaxis(gamma, 0, -1)
    total = dipole(x, INSIDE) + dipole(x, OUTSIDE)
    return VmecSurfaceFieldData(
        gamma=gamma, B_total=jnp.moveaxis(total, -1, 0), normal=jnp.zeros_like(gamma), area_vector=jnp.zeros_like(gamma),
        theta=theta, phi=phi, nfp=NFP, stellsym=False, signgs=1)


field = VirtualCasingExteriorField(surface(), ExteriorFieldConfig(digits=8, src_nphi=NPHI, src_ntheta=NTHETA))

# targets on both sides of the surface at phi = 0, down to a micrometre from it
out = jnp.array([[R0 + (A + DELTA) + d, 0.0, 0.0] for d in (1.0, 0.1, 1e-3, 1e-6)])
inn = jnp.array([[R0 + (A + DELTA) - d, 0.0, 0.0] for d in (0.3, 0.1, 1e-3, 1e-6)])
for name, points, truth in (("outside", out, dipole(out, INSIDE)), ("inside", inn, -dipole(inn, OUTSIDE))):
    B, estimate = field.B_plasma_xyz(points, return_estimate=True)
    error = jnp.max(jnp.abs(B - truth), axis=1) / jnp.max(jnp.abs(truth))
    print(f"{name}: error {np.asarray(error)}  estimate {np.asarray(estimate)}")
    assert float(error.max()) < 1e-6 and bool(jnp.all(estimate >= 0))

# derivative jets up to order 3 from the same call; the graded rule is used where the closed forms are not accurate
(B, gradB, hessB, thirdB), estimate = field.B_and_derivatives_accurate_xyz(out, order=3, tol=1e-8)
print("jet shapes:", B.shape, gradB.shape, hessB.shape, thirdB.shape)
np.testing.assert_allclose(gradB, jax.vmap(jax.jacfwd(lambda x: dipole(x, INSIDE)))(out), atol=1e-6 * float(jnp.abs(gradB).max()))

# reverse-mode gradients through the targets (the boundary data are differentiable too)
loss = lambda p: jnp.sum(field.B_plasma_xyz(p) ** 2)
g = jax.grad(loss)(out)
step = 1e-6 * jnp.ones_like(out)
fd = (loss(out + step) - loss(out - step)) / (2e-6)
print("directional derivative: jax.grad", float(jnp.sum(g)), "finite differences", float(fd))
assert abs(float(jnp.sum(g)) - float(fd)) < 1e-5 * abs(float(fd))
