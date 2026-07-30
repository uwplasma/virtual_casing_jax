from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from virtual_casing_jax.singular_quadrature import (
    _precompute_singular_host,
    precompute_singular,
)
from virtual_casing_jax.virtual_casing import VirtualCasingJAX

from test_virtual_casing_jit_batch import _infer_setup, _reconstruct_B0
from dump_io import load_dump


DATA_DIR = Path(__file__).parent / "data"


def _case_vc():
    X, src_nt, src_np, nfp, nfp_eff, half_period = _infer_setup("case_vc")
    ref = load_dump(DATA_DIR / "case_vc_computeB_Bvc")
    quad = load_dump(DATA_DIR / "case_vc_computeB_quad_coord")
    trg_nt, trg_np = ref.shape[1:]
    B0 = _reconstruct_B0(
        "case_vc", src_nt, src_np, nfp, nfp_eff, half_period, trg_nt
    )
    vc = VirtualCasingJAX()
    vc.setup(
        5,
        nfp,
        half_period,
        src_nt,
        src_np,
        X,
        src_nt,
        src_np,
        trg_nt,
        trg_np,
    )
    return vc, X, B0, (quad.shape[1], quad.shape[2])


def test_all_on_surface_jit_wrappers_accept_automatic_quadrature():
    vc, _, B0, _ = _case_vc()

    calls = (
        (vc.compute_external_B, vc.compute_external_B_jit),
        (vc.compute_internal_B, vc.compute_internal_B_jit),
        (vc.compute_external_gradB, vc.compute_external_gradB_jit),
        (vc.compute_internal_gradB, vc.compute_internal_gradB_jit),
    )
    for eager, compiled in calls:
        expected = eager(B0, chunk_size=1024)
        got = compiled(B0, chunk_size=1024)
        np.testing.assert_allclose(got, expected, rtol=1e-8, atol=1e-10)


def test_jit_first_does_not_leak_tracers_into_eager_calls():
    _precompute_singular_host.cache_clear()
    vc, _, B0, (quad_nt, quad_np) = _case_vc()
    kwargs = dict(quad_nt=quad_nt, quad_np=quad_np, chunk_size=1024)

    compiled = vc.compute_external_B_jit(B0, **kwargs)
    eager = vc.compute_external_B(B0, **kwargs)
    np.testing.assert_allclose(compiled, eager, rtol=1e-8, atol=1e-10)


def test_setup_invalidates_geometry_capturing_jit_closures():
    vc, X, B0, (quad_nt, quad_np) = _case_vc()
    kwargs = dict(quad_nt=quad_nt, quad_np=quad_np, chunk_size=1024)
    old = np.asarray(vc.compute_external_B_jit(B0, **kwargs))

    X_new = np.array(X, copy=True)
    X_new[2] *= 1.2
    vc.setup(5, 1, False, X.shape[1], X.shape[2], X_new, X.shape[1], X.shape[2], 4, 4)
    assert not vc._jit_cache
    eager = np.asarray(vc.compute_external_B(B0, **kwargs))
    compiled = np.asarray(vc.compute_external_B_jit(B0, **kwargs))

    assert np.linalg.norm(compiled - old) > 1e-6
    np.testing.assert_allclose(compiled, eager, rtol=1e-8, atol=1e-10)


def test_on_surface_target_shape_is_explicitly_validated():
    vc, _, B0, (quad_nt, quad_np) = _case_vc()
    with pytest.raises(ValueError, match="use an off-surface API"):
        vc.compute_external_B(
            B0,
            X_trg=np.zeros((3, 3)),
            quad_nt=quad_nt,
            quad_np=quad_np,
        )


def test_host_singular_cache_is_bounded():
    _precompute_singular_host.cache_clear()
    for patch_dim0 in range(6, 16):
        precompute_singular(patch_dim0, 10, hedgehog_order=1)
    assert _precompute_singular_host.cache_info().maxsize == 8
    assert _precompute_singular_host.cache_info().currsize <= 8


def test_float32_high_digits_warn_once_in_fresh_process():
    code = """
import warnings
import jax.numpy as jnp
from virtual_casing_jax.utils import warn_if_x64_needed
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    warn_if_x64_needed(6, jnp.float32)
    warn_if_x64_needed(7, jnp.float32)
messages = [str(item.message) for item in caught]
assert len(messages) == 1, messages
assert "JAX_ENABLE_X64=1" in messages[0]
"""
    env = dict(os.environ)
    env["JAX_ENABLE_X64"] = "0"
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=True,
    )
