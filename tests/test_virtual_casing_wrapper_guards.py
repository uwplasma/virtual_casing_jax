import numpy as np
import pytest
import jax.numpy as jnp

from virtual_casing_jax.virtual_casing import VirtualCasingJAX


def _torus_surface(nt=4, npol=5, R0=2.0, r=0.25):
    phi = jnp.linspace(0.0, 2.0 * jnp.pi, nt, endpoint=False)
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, npol, endpoint=False)
    phi2d, theta2d = jnp.meshgrid(phi, theta, indexing="ij")
    return jnp.stack(
        (
            (R0 + r * jnp.cos(theta2d)) * jnp.cos(phi2d),
            (R0 + r * jnp.cos(theta2d)) * jnp.sin(phi2d),
            r * jnp.sin(theta2d),
        ),
        axis=0,
    )


def test_setup_records_field_period_metadata_and_resets_quad_setups():
    vc = VirtualCasingJAX()
    vc._b_setup = object()
    vc._grad_setup = object()
    X = _torus_surface(nt=4, npol=5)

    vc.setup(3, 2, False, 4, 5, X, 4, 5, 4, 5)

    assert vc._setup is True
    assert vc.nfp == 2
    assert vc.nfp_eff == 2
    assert vc.surface_coord.shape == (3, 8, 5)
    assert vc._b_setup is None
    assert vc._grad_setup is None


def test_setup_half_period_builds_stellarator_symmetric_full_surface():
    vc = VirtualCasingJAX()
    X = _torus_surface(nt=3, npol=5)

    vc.setup(3, 2, True, 3, 5, X, 3, 5, 3, 5)

    assert vc.half_period is True
    assert vc.nfp_eff == 4
    assert vc.surface_coord.shape == (3, 16, 5)


def test_resolution_helpers_resolve_auto_dtypes_and_schedules():
    vc = VirtualCasingJAX()

    src, trg = vc._resolve_chunk_sizes("b", "auto", "auto", nsrc=18432, ntrg=512)
    assert src == 512
    assert trg == 64

    src, trg = vc._resolve_chunk_sizes("gradb", 128, "auto", nsrc=4096, ntrg=16)
    assert src == 128
    assert trg is None

    assert vc._resolve_pou_dtype("auto", jnp.dtype("float64")) == jnp.float32
    assert vc._resolve_pou_dtype("float64", jnp.dtype("float32")) == jnp.float64
    assert vc._resolve_pou_dtype(None, jnp.dtype("float64")) is None
    assert vc._resolve_patch_dtype("auto", jnp.dtype("float64")) == jnp.float32
    assert vc._resolve_patch_dtype(jnp.float64, jnp.dtype("float32")) == jnp.float64

    explicit = vc._resolve_offsurface_levels(((13, 14), (26, 28)), nt0=13, np0=14, max_Nt=-1, max_Np=-1, max_levels=3)
    assert explicit == ((13, 14), (26, 28))

    auto = vc._resolve_offsurface_levels(None, nt0=13, np0=14, max_Nt=52, max_Np=56, max_levels=3)
    assert auto[0] == (13, 14)
    assert len(auto) <= 3
    assert all(nt <= 52 and npol <= 56 for nt, npol in auto)


def test_public_methods_fail_fast_before_setup_or_without_targets():
    vc = VirtualCasingJAX()
    B0 = jnp.zeros((3, 4, 5))
    X_trg = jnp.zeros((3, 2))

    with pytest.raises(RuntimeError, match="setup"):
        vc.compute_external_B(B0)

    with pytest.raises(RuntimeError, match="setup"):
        vc.compute_internal_gradB(B0)

    with pytest.raises(RuntimeError, match="off-surface"):
        vc.compute_external_B_offsurf(B0, X_trg=X_trg)

    with pytest.raises(RuntimeError, match="off-surface"):
        vc.compute_internal_B_offsurf(B0, X_trg=X_trg)

    with pytest.raises(RuntimeError, match="off-surface"):
        vc.compute_external_B_offsurf_schedule(B0, X_trg=X_trg, levels=((13, 13),))

    with pytest.raises(RuntimeError, match="off-surface"):
        vc.compute_internal_gradB_offsurf_schedule(B0, X_trg=X_trg, levels=((13, 13),))

    with pytest.raises(RuntimeError, match="off-surface"):
        vc.compute_external_gradB_offsurf(B0, X_trg=X_trg)

    with pytest.raises(RuntimeError, match="off-surface"):
        vc.compute_internal_gradB_offsurf(B0, X_trg=X_trg)

    with pytest.raises(ValueError, match="compute_external_B_jit does not support X_trg"):
        vc.compute_external_B_jit(B0, X_trg=X_trg)

    with pytest.raises(ValueError, match="compute_internal_B_jit does not support X_trg"):
        vc.compute_internal_B_jit(B0, X_trg=X_trg)

    with pytest.raises(ValueError, match="compute_external_gradB_jit does not support X_trg"):
        vc.compute_external_gradB_jit(B0, X_trg=X_trg)

    with pytest.raises(ValueError, match="compute_internal_gradB_jit does not support X_trg"):
        vc.compute_internal_gradB_jit(B0, X_trg=X_trg)

    with pytest.raises(ValueError, match="X_trg must be provided"):
        vc.compute_external_B_autodiff(B0, X_trg=None)


def test_batch_methods_route_over_field_and_target_batch_axes():
    vc = VirtualCasingJAX()
    B0_batch = jnp.arange(24.0, dtype=jnp.float64).reshape((2, 3, 4))
    X_trg_batch = 100.0 + B0_batch

    def external_B(b, *, X_trg=None, scale=1.0):
        return scale * b if X_trg is None else scale * b + X_trg

    def internal_B(b, *, X_trg=None, offset=0.0):
        return -b + offset if X_trg is None else -b + X_trg + offset

    def external_gradB(b, *, offset=0.0):
        return b[:, None, :] + offset

    def internal_gradB(b, *, offset=0.0):
        return -b[:, None, :] + offset

    vc.compute_external_B = external_B
    vc.compute_internal_B = internal_B
    vc.compute_external_gradB = external_gradB
    vc.compute_internal_gradB = internal_gradB

    np.testing.assert_allclose(vc.compute_external_B_batch(B0_batch, scale=2.0), 2.0 * B0_batch)
    np.testing.assert_allclose(
        vc.compute_external_B_batch(B0_batch, X_trg=X_trg_batch, scale=2.0),
        2.0 * B0_batch + X_trg_batch,
    )
    np.testing.assert_allclose(vc.compute_internal_B_batch(B0_batch, offset=3.0), -B0_batch + 3.0)
    np.testing.assert_allclose(
        vc.compute_internal_B_batch(B0_batch, X_trg=X_trg_batch, offset=3.0),
        -B0_batch + X_trg_batch + 3.0,
    )
    np.testing.assert_allclose(vc.compute_external_gradB_batch(B0_batch, offset=5.0), B0_batch[:, :, None, :] + 5.0)
    np.testing.assert_allclose(vc.compute_internal_gradB_batch(B0_batch, offset=5.0), -B0_batch[:, :, None, :] + 5.0)
