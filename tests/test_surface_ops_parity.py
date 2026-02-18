import os
from pathlib import Path

import numpy as np
import pytest
import jax.numpy as jnp

from virtual_casing_jax.surface_ops import upsample, resample
from .dump_io import load_dump

DATA_DIR = Path(__file__).parent / "data"


def _dump_exists(prefix: str, name: str):
    base = DATA_DIR / f"{prefix}_{name}"
    return base.with_suffix(".bin").exists() and base.with_suffix(".json").exists()


@pytest.mark.skipif(not _dump_exists("case_vc", "computeB_B_resampled"), reason="parity dump not available")
def test_resample_parity_case_vc():
    B = load_dump(DATA_DIR / "case_vc_computeB_B_resampled")
    # B is (3, quad_Nt, quad_Np) in SoA layout
    nt = B.shape[1]
    npol = B.shape[2]

    # Resample to same grid must match
    B_jax = resample(jnp.asarray(B), nt, npol, nt, npol)
    np.testing.assert_allclose(np.asarray(B_jax), B, rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(not _dump_exists("case_vc", "computeB_B_resampled"), reason="parity dump not available")
def test_upsample_parity_case_vc():
    B = load_dump(DATA_DIR / "case_vc_computeB_B_resampled")
    nt = B.shape[1]
    npol = B.shape[2]
    B_jax = upsample(jnp.asarray(B), nt, npol, nt, npol)
    np.testing.assert_allclose(np.asarray(B_jax), B, rtol=1e-12, atol=1e-12)
