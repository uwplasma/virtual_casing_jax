#!/usr/bin/env python3
"""Generate small parity dumps from the reference virtual-casing implementation."""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np


def _copy_prefix(src_dir: Path, dst_dir: Path, prefix: str):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for path in src_dir.glob(f"{prefix}_*.bin"):
        shutil.copy2(path, dst_dir / path.name)
        meta = path.with_suffix(".json")
        if meta.exists():
            shutil.copy2(meta, dst_dir / meta.name)


def make_virtual_casing_testdata(dst_dir: Path, mode: str):
    import virtual_casing as vc

    nfp = 1
    half_period = False
    nt = 6
    npol = 5
    src_nt = 6
    src_np = 5
    trg_nt = 4
    trg_np = 4
    digits = 5

    X = vc.VirtualCasingTestData.surface_coordinates(nfp, half_period, nt, npol, vc.SurfType.AxisymNarrow)
    Bext, Bint = vc.VirtualCasingTestData.magnetic_field_data(nfp, half_period, nt, npol, X, src_nt, src_np)
    Btotal = (np.array(Bext) + np.array(Bint)).tolist()

    vcasing = vc.VirtualCasing()
    vcasing.setup(digits, nfp, half_period, nt, npol, X, src_nt, src_np, trg_nt, trg_np)

    # Off-surface, 3 target points
    Xt = [2.0, 2.1, 2.2, 0.0, 0.1, 0.2, 0.0, 0.0, 0.0]

    if mode == "ext":
        _ = vcasing.compute_external_B(Btotal)
        _ = vcasing.compute_external_gradB(Btotal)
        _ = vcasing.compute_external_B_offsurf(Btotal, Xt, -1, -1)
        _ = vcasing.compute_external_gradB_offsurf(Btotal, Xt, -1, -1)
    elif mode == "int":
        _ = vcasing.compute_internal_B(Btotal)
        _ = vcasing.compute_internal_gradB(Btotal)
        _ = vcasing.compute_internal_B_offsurf(Btotal, Xt, -1, -1)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def make_simsopt_vmec_case(dst_dir: Path, mode: str):
    try:
        from simsopt.mhd import VirtualCasing, Vmec
    except Exception:
        print("simsopt not available; skipping simsopt parity dumps")
        return
    import virtual_casing as vc_module

    test_dir = Path(__file__).resolve().parents[2] / "simsopt" / "tests" / "test_files"
    wout = test_dir / "wout_20220102-01-053-003_QH_nfp4_aspect6p5_beta0p05_iteratedWithSfincs_reference.nc"
    if not wout.exists():
        print(f"Missing VMEC file: {wout}; skipping simsopt case")
        return

    vmec = Vmec(str(wout))
    vc = VirtualCasing.from_vmec(
        vmec,
        src_nphi=8,
        src_ntheta=8,
        trgt_nphi=6,
        trgt_ntheta=6,
        use_stellsym=True,
        digits=6,
    )

    gamma = vc.gamma
    B_total = vc.B_total
    src_nphi = vc.src_nphi
    src_ntheta = vc.src_ntheta
    trgt_nphi = vc.trgt_nphi
    trgt_ntheta = vc.trgt_ntheta

    gamma1d = np.zeros(src_nphi * src_ntheta * 3)
    B1d = np.zeros(src_nphi * src_ntheta * 3)
    for jxyz in range(3):
        gamma1d[jxyz * src_nphi * src_ntheta:(jxyz + 1) * src_nphi * src_ntheta] = \
            gamma[:, :, jxyz].flatten(order="C")
        B1d[jxyz * src_nphi * src_ntheta:(jxyz + 1) * src_nphi * src_ntheta] = \
            B_total[:, :, jxyz].flatten(order="C")

    vcasing = vc_module.VirtualCasing()
    vcasing.setup(
        6,
        vc.nfp,
        True,
        src_nphi,
        src_ntheta,
        gamma1d,
        src_nphi,
        src_ntheta,
        trgt_nphi,
        trgt_ntheta,
    )
    if mode == "ext":
        _ = vcasing.compute_external_B(B1d)
        _ = vcasing.compute_external_gradB(B1d)
    elif mode == "int":
        _ = vcasing.compute_internal_B(B1d)
        _ = vcasing.compute_internal_gradB(B1d)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main():
    dst_dir = Path(__file__).resolve().parents[1] / "tests" / "data"
    dst_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        os.environ["VC_DUMP_DIR"] = str(tmpdir)

        os.environ["VC_DUMP_PREFIX"] = "case_vc"
        make_virtual_casing_testdata(dst_dir, mode="ext")
        _copy_prefix(tmpdir, dst_dir, "case_vc")

        os.environ["VC_DUMP_PREFIX"] = "case_vc_int"
        make_virtual_casing_testdata(dst_dir, mode="int")
        _copy_prefix(tmpdir, dst_dir, "case_vc_int")

        os.environ["VC_DUMP_PREFIX"] = "case_simsopt"
        make_simsopt_vmec_case(dst_dir, mode="ext")
        _copy_prefix(tmpdir, dst_dir, "case_simsopt")

        os.environ["VC_DUMP_PREFIX"] = "case_simsopt_int"
        make_simsopt_vmec_case(dst_dir, mode="int")
        _copy_prefix(tmpdir, dst_dir, "case_simsopt_int")

    print(f"Parity dumps written to {dst_dir}")


if __name__ == "__main__":
    main()
