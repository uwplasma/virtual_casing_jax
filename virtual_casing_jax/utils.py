"""Utility helpers for virtual_casing_jax."""
from __future__ import annotations

import os
import jax
import jax.numpy as jnp


def soa_to_array(vec, dof: int, nt: int, npol: int):
    """Convert SoA 1D vector to array with shape (dof, nt, npol)."""
    return jnp.asarray(vec).reshape((dof, nt, npol), order="C")


def array_to_soa(arr):
    """Flatten array with shape (dof, nt, npol) to SoA 1D vector."""
    return jnp.asarray(arr).reshape((-1,), order="C")


def _env_int(name: str):
    val = os.getenv(name)
    if val is None or val == "":
        return None
    try:
        return int(val)
    except ValueError:
        return None


def autotune_chunk_sizes(op: str, nsrc: int, ntrg: int, backend: str | None = None):
    """Heuristic chunk sizes based on op type and backend."""
    backend = backend or jax.default_backend()
    op_key = op.lower()

    # Environment overrides
    env_base = f"VCJAX_CHUNK_{op_key.upper()}"
    src_override = _env_int(env_base) or _env_int(f"{env_base}_SRC")
    trg_override = _env_int(f"{env_base}_TRG")

    if backend == "cpu":
        if op_key in ("b", "boff"):
            src = 1024 if nsrc <= 5000 else 512
            trg = 64
        else:
            src = 256 if nsrc >= 2048 else 512
            trg = 32
    else:
        if op_key in ("b", "boff"):
            src = 4096 if nsrc >= 4096 else 1024
            trg = 256
        else:
            src = 1024 if nsrc >= 1024 else 512
            trg = 64

    if src_override is not None:
        src = src_override
    if trg_override is not None:
        trg = trg_override

    if nsrc > 0:
        src = min(src, nsrc)
    if ntrg > 0 and trg is not None:
        if ntrg <= trg:
            trg = None
        else:
            trg = min(trg, ntrg)

    return int(src), (None if trg is None else int(trg))
