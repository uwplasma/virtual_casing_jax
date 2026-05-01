"""Smooth objective helpers for exterior-field workflows."""
from __future__ import annotations

import jax.numpy as jnp


def _weighted_mean_square(values, weights):
    values = jnp.asarray(values)
    weights = jnp.asarray(weights)
    return jnp.sum(weights * values * values) / jnp.maximum(jnp.sum(weights), jnp.asarray(1e-300, dtype=values.dtype))


def exterior_Bn_squared(field, target_xyz, target_normals, weights, target_Bn=0.0):
    """Weighted mean-square normal-field objective on target points."""
    B = field.B_xyz(target_xyz)
    normals = jnp.asarray(target_normals)
    Bn = jnp.sum(B * normals, axis=-1)
    return _weighted_mean_square(Bn - target_Bn, weights)


def exterior_B_magnitude_squared(field, target_xyz, weights, target_absB):
    """Weighted mean-square magnetic-field magnitude objective."""
    B = field.B_xyz(target_xyz)
    absB = jnp.linalg.norm(B, axis=-1)
    return _weighted_mean_square(absB - target_absB, weights)


def _gradient_axis(values, coords, axis: int):
    values = jnp.asarray(values)
    coords = jnp.asarray(coords)
    if values.shape[axis] < 2:
        return jnp.zeros_like(values)
    return jnp.gradient(values, coords, axis=axis)


def grid_divergence_penalty(BR, Bphi, BZ, R, phi, Z, weights=None):
    """Finite-difference cylindrical ``div B`` mean-square diagnostic."""
    BR = jnp.asarray(BR)
    Bphi = jnp.asarray(Bphi)
    BZ = jnp.asarray(BZ)
    R = jnp.asarray(R)
    phi = jnp.asarray(phi)
    Z = jnp.asarray(Z)

    d_RBR_dR = _gradient_axis(R[:, None, None] * BR, R, axis=0)
    if phi.shape[0] > 1:
        dphi = phi[1] - phi[0]
        dBphi_dphi = (jnp.roll(Bphi, -1, axis=1) - jnp.roll(Bphi, 1, axis=1)) / (2.0 * dphi)
    else:
        dBphi_dphi = jnp.zeros_like(Bphi)
    dBZ_dZ = _gradient_axis(BZ, Z, axis=2)
    divB = (d_RBR_dR + dBphi_dphi) / R[:, None, None] + dBZ_dZ
    if weights is None:
        weights = jnp.ones_like(divB)
    return _weighted_mean_square(divB, weights)


def boundary_normal_residual(surface_data, field):
    """Return ``B_total . n`` on the VMEC boundary source grid."""
    B = field.B_xyz(surface_data.gamma)
    return jnp.sum(B * surface_data.normal, axis=0)
