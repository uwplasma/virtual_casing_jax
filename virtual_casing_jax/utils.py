"""Utility helpers for virtual_casing_jax."""
from __future__ import annotations

import jax.numpy as jnp


def soa_to_array(vec, dof: int, nt: int, npol: int):
    """Convert SoA 1D vector to array with shape (dof, nt, npol)."""
    return jnp.asarray(vec).reshape((dof, nt, npol), order="C")


def array_to_soa(arr):
    """Flatten array with shape (dof, nt, npol) to SoA 1D vector."""
    return jnp.asarray(arr).reshape((-1,), order="C")
