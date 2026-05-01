import os
from pathlib import Path

import numpy as np
import pytest
import jax.numpy as jnp

from virtual_casing_jax import ExteriorFieldConfig, VirtualCasingExteriorField, surface_field_from_vmec_jax


def _vmec_example_root():
    env_root = os.environ.get("VMEC_JAX_ROOT")
    candidates = []
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend(
        [
            Path(__file__).resolve().parents[2] / "vmec_jax",
            Path(__file__).resolve().parents[2].parent / "vmec_jax",
        ]
    )
    for root in candidates:
        if (root / "examples" / "data" / "input.circular_tokamak").exists():
            return root
    return None


def test_surface_field_from_vmec_jax_circular_tokamak_smoke():
    vmec_jax = pytest.importorskip("vmec_jax")
    root = _vmec_example_root()
    if root is None:
        pytest.skip("vmec_jax circular_tokamak example data is not available")

    example = vmec_jax.load_example("circular_tokamak", root=root)
    surface = surface_field_from_vmec_jax(example.state, example.static, example.indata, wout=example.wout)

    assert surface.gamma.shape == surface.B_total.shape == surface.normal.shape == surface.area_vector.shape
    assert surface.gamma.shape[0] == 3
    assert surface.phi.shape[0] == surface.gamma.shape[1]
    assert surface.theta.shape[0] == surface.gamma.shape[2]
    assert surface.source_convention == "vmec_jax"

    Bn = jnp.sum(surface.B_total * surface.normal, axis=0)
    absB = jnp.linalg.norm(surface.B_total, axis=0)
    normalized_rms = jnp.sqrt(jnp.mean(Bn * Bn)) / jnp.sqrt(jnp.mean(absB * absB))
    assert float(normalized_rms) < 1e-12

    field = VirtualCasingExteriorField(
        surface,
        ExteriorFieldConfig(digits=3, levels=((13, 13),), chunk_size=64, target_chunk_size=1, dtype="float64"),
    )
    B = field.B_plasma_xyz(np.array([[2.5, 0.0, 0.0]]))
    assert B.shape == (1, 3)
    assert np.all(np.isfinite(B))
