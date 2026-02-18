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


def make_virtual_casing_testdata(dst_dir: Path):
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

    # On-surface
    _ = vcasing.compute_external_B(Btotal)

    # Off-surface, 3 target points
    Xt = [2.0, 2.1, 2.2, 0.0, 0.1, 0.2, 0.0, 0.0, 0.0]
    _ = vcasing.compute_external_B_offsurf(Btotal, Xt, -1, -1)

    # GradB
    _ = vcasing.compute_external_gradB(Btotal)


def make_simsopt_vmec_case(dst_dir: Path):
    try:
        from simsopt.mhd import VirtualCasing, Vmec
    except Exception:
        print("simsopt not available; skipping simsopt parity dumps")
        return

    test_dir = Path(__file__).resolve().parents[2] / "simsopt" / "tests" / "test_files"
    wout = test_dir / "wout_20220102-01-053-003_QH_nfp4_aspect6p5_beta0p05_iteratedWithSfincs_reference.nc"
    if not wout.exists():
        print(f"Missing VMEC file: {wout}; skipping simsopt case")
        return

    vmec = Vmec(str(wout))
    _ = VirtualCasing.from_vmec(vmec, src_nphi=8, src_ntheta=8, trgt_nphi=6, trgt_ntheta=6, use_stellsym=True)


def main():
    dst_dir = Path(__file__).resolve().parents[1] / "tests" / "data"
    dst_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        os.environ["VC_DUMP_DIR"] = str(tmpdir)

        os.environ["VC_DUMP_PREFIX"] = "case_vc"
        make_virtual_casing_testdata(dst_dir)
        _copy_prefix(tmpdir, dst_dir, "case_vc")

        os.environ["VC_DUMP_PREFIX"] = "case_simsopt"
        make_simsopt_vmec_case(dst_dir)
        _copy_prefix(tmpdir, dst_dir, "case_simsopt")

    print(f"Parity dumps written to {dst_dir}")


if __name__ == "__main__":
    main()
