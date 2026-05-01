import os
import sys
import types
import builtins
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import jax.numpy as jnp

from virtual_casing_jax import ExteriorFieldConfig, VirtualCasingExteriorField, surface_field_from_vmec_jax
from virtual_casing_jax import vmec_jax_bridge


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


def _fake_static_geom(nfp=2, ntheta=5, nzeta=6):
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, ntheta, endpoint=False)
    zeta = jnp.linspace(0.0, 2.0 * jnp.pi, nzeta, endpoint=False)
    theta2d, zeta2d = jnp.meshgrid(theta, zeta, indexing="ij")
    phi2d = zeta2d / nfp
    major_radius = 2.0
    minor_radius = 0.25
    R = major_radius + minor_radius * jnp.cos(theta2d)
    Z = minor_radius * jnp.sin(theta2d)

    geom = SimpleNamespace(
        R=R[None, :, :],
        Z=Z[None, :, :],
        Rs=jnp.cos(theta2d)[None, :, :],
        Zs=jnp.sin(theta2d)[None, :, :],
        Rt=(-minor_radius * jnp.sin(theta2d))[None, :, :],
        Zt=(minor_radius * jnp.cos(theta2d))[None, :, :],
        Rp=jnp.zeros_like(R)[None, :, :],
        Zp=jnp.zeros_like(Z)[None, :, :],
        sqrtg=jnp.ones((1, ntheta, nzeta)),
    )
    B_cart = jnp.stack(
        [
            -jnp.sin(phi2d),
            jnp.cos(phi2d),
            0.1 * jnp.ones_like(phi2d),
        ],
        axis=-1,
    )[None, :, :, :]
    static = SimpleNamespace(
        cfg=SimpleNamespace(nfp=nfp, lasym=False),
        grid=SimpleNamespace(theta=theta, zeta=zeta),
        s=jnp.array([1.0]),
    )
    expected_normal = jnp.stack(
        [
            jnp.cos(theta2d) * jnp.cos(phi2d),
            jnp.cos(theta2d) * jnp.sin(phi2d),
            jnp.sin(theta2d),
        ],
        axis=-1,
    )
    return static, geom, B_cart, expected_normal


def _install_fake_vmec_jax(monkeypatch, geom, B_cart, *, signgs=1, record=None):
    record = {} if record is None else record
    vmec_mod = types.ModuleType("vmec_jax")
    vmec_mod.__path__ = []
    geom_mod = types.ModuleType("vmec_jax.geom")
    field_mod = types.ModuleType("vmec_jax.field")
    energy_mod = types.ModuleType("vmec_jax.energy")

    geom_mod.eval_geom = lambda state, static: geom
    field_mod.signgs_from_sqrtg = lambda sqrtg, axis_index=1: signgs

    def flux_profiles_from_indata(indata, s, signgs):
        record["flux_signgs"] = int(signgs)
        return SimpleNamespace(
            phipf=jnp.ones_like(s),
            chipf=2.0 * jnp.ones_like(s),
            lamscale=3.0 * jnp.ones_like(s),
        )

    def chips_from_wout_chipf(chipf, phipf, iotaf=None, iotas=None):
        record["wout_chipf"] = jnp.asarray(chipf)
        record["wout_phipf"] = jnp.asarray(phipf)
        return 4.0 * jnp.asarray(chipf)

    def lamscale_from_phips(phips, s):
        record["wout_phips"] = jnp.asarray(phips)
        return 5.0 * jnp.ones_like(s)

    def bsup_from_geom(geom, phipf, chipf, nfp, signgs, lamscale):
        record["bsup_nfp"] = int(nfp)
        record["bsup_signgs"] = int(signgs)
        record["bsup_phipf"] = jnp.asarray(phipf)
        record["bsup_chipf"] = jnp.asarray(chipf)
        record["bsup_lamscale"] = jnp.asarray(lamscale)
        shape = geom.R.shape
        return jnp.zeros(shape), jnp.zeros(shape)

    def b_cartesian_from_bsup(geom, bsupu, bsupv, zeta, nfp):
        record["b_nfp"] = int(nfp)
        record["b_zeta"] = jnp.asarray(zeta)
        return B_cart

    energy_mod.flux_profiles_from_indata = flux_profiles_from_indata
    field_mod.bsup_from_geom = bsup_from_geom
    field_mod.b_cartesian_from_bsup = b_cartesian_from_bsup
    field_mod.chips_from_wout_chipf = chips_from_wout_chipf
    field_mod.lamscale_from_phips = lamscale_from_phips

    monkeypatch.setitem(sys.modules, "vmec_jax", vmec_mod)
    monkeypatch.setitem(sys.modules, "vmec_jax.geom", geom_mod)
    monkeypatch.setitem(sys.modules, "vmec_jax.field", field_mod)
    monkeypatch.setitem(sys.modules, "vmec_jax.energy", energy_mod)
    return record


def test_surface_field_from_vmec_jax_fake_vmec_enforces_outward_phi_and_layout(monkeypatch):
    static, geom, B_cart, expected_normal = _fake_static_geom(nfp=3)
    record = _install_fake_vmec_jax(monkeypatch, geom, B_cart, signgs=-1)

    surface = surface_field_from_vmec_jax(object(), static, indata=object(), orientation="auto")

    assert surface.gamma.shape == (3, static.grid.zeta.size, static.grid.theta.size)
    np.testing.assert_allclose(surface.phi, static.grid.zeta / static.cfg.nfp)
    np.testing.assert_allclose(jnp.transpose(surface.B_total, (2, 1, 0)), B_cart[0])
    np.testing.assert_allclose(jnp.transpose(surface.normal, (2, 1, 0)), expected_normal, atol=1e-14)
    assert surface.stellsym is True
    assert surface.signgs == -1
    assert record["flux_signgs"] == -1
    assert record["bsup_nfp"] == 3
    assert record["b_nfp"] == 3


def test_surface_field_from_vmec_jax_resampling_and_orientation_errors(monkeypatch):
    static, geom, B_cart, _ = _fake_static_geom()
    _install_fake_vmec_jax(monkeypatch, geom, B_cart)

    with pytest.raises(ValueError, match="orientation"):
        surface_field_from_vmec_jax(object(), static, indata=object(), orientation="inward")
    with pytest.raises(NotImplementedError, match="src_nphi"):
        surface_field_from_vmec_jax(object(), static, indata=object(), src_nphi=static.grid.zeta.size + 1)
    with pytest.raises(NotImplementedError, match="src_ntheta"):
        surface_field_from_vmec_jax(object(), static, indata=object(), src_ntheta=static.grid.theta.size + 1)


def test_surface_field_from_vmec_jax_wout_flux_path_uses_internal_units(monkeypatch):
    static, geom, B_cart, _ = _fake_static_geom(nfp=2)
    record = _install_fake_vmec_jax(monkeypatch, geom, B_cart, signgs=1)
    wout = SimpleNamespace(
        signgs=-1,
        phipf=jnp.array([2.0 * jnp.pi]),
        chipf=jnp.array([4.0 * jnp.pi]),
        phips=jnp.array([0.0, 1.0]),
        iotaf=None,
        iotas=None,
    )

    surface = surface_field_from_vmec_jax(object(), static, wout=wout, use_stellsym=False)

    scale = 2.0 * np.pi * wout.signgs
    np.testing.assert_allclose(record["wout_phipf"], jnp.asarray(wout.phipf) / scale)
    np.testing.assert_allclose(record["wout_chipf"], jnp.asarray(wout.chipf) / scale)
    np.testing.assert_allclose(record["bsup_chipf"], 4.0 * record["wout_chipf"])
    np.testing.assert_allclose(record["bsup_lamscale"], 5.0 * jnp.ones_like(static.s))
    assert surface.stellsym is False
    assert surface.signgs == -1


def test_require_vmec_jax_reports_optional_dependency(monkeypatch):
    real_import = builtins.__import__

    def missing_vmec_import(name, *args, **kwargs):
        if name == "vmec_jax":
            raise ImportError("forced missing vmec_jax")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "vmec_jax", raising=False)
    monkeypatch.setattr(builtins, "__import__", missing_vmec_import)

    with pytest.raises(RuntimeError, match="requires vmec_jax"):
        vmec_jax_bridge._require_vmec_jax()
