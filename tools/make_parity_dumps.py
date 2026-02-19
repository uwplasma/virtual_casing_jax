#!/usr/bin/env python3
"""Generate parity dumps from the reference virtual-casing implementation."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import json


def _copy_prefix(src_dir: Path, dst_dir: Path, prefix: str):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for path in src_dir.glob(f"{prefix}_*.bin"):
        shutil.copy2(path, dst_dir / path.name)
        meta = path.with_suffix(".json")
        if meta.exists():
            shutil.copy2(meta, dst_dir / meta.name)


def _write_dump(dst_dir: Path, name: str, arr: np.ndarray):
    dst_dir.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(arr)
    meta = {
        "dtype": "float32" if arr.dtype == np.float32 else "float64",
        "shape": list(arr.shape),
    }
    bin_path = dst_dir / f"{name}.bin"
    json_path = dst_dir / f"{name}.json"
    arr.astype(arr.dtype).tofile(bin_path)
    with json_path.open("w") as f:
        json.dump(meta, f, indent=2)


def _reset_drand48():
    try:
        import ctypes

        libc = ctypes.CDLL(None)
        libc.srand48.argtypes = [ctypes.c_long]
        libc.srand48.restype = None
        libc.srand48(ctypes.c_long(0x1234ABCD))
    except Exception as exc:
        print(f"Warning: could not reset drand48 state: {exc}")


def _make_xt_points(npts: int):
    if npts == 3:
        xs = np.array([2.0, 2.1, 2.2])
        ys = np.array([0.0, 0.1, 0.2])
        zs = np.array([0.0, 0.0, 0.0])
    else:
        t = np.linspace(0.0, 1.0, npts, endpoint=False)
        xs = 2.0 + 0.1 * t
        ys = 0.15 * np.sin(2 * np.pi * t)
        zs = 0.10 * np.cos(2 * np.pi * t)
    return np.concatenate([xs, ys, zs]).tolist()


def make_virtual_casing_case(
    *,
    mode: str,
    nfp: int,
    half_period: bool,
    nt: int,
    npol: int,
    src_nt: int,
    src_np: int,
    trg_nt: int,
    trg_np: int,
    digits: int,
    surf_type,
    xt: list[float] | None = None,
):
    import virtual_casing as vc

    X = vc.VirtualCasingTestData.surface_coordinates(nfp, half_period, nt, npol, surf_type)
    Bext, Bint = vc.VirtualCasingTestData.magnetic_field_data(nfp, half_period, nt, npol, X, src_nt, src_np)
    Btotal = (np.array(Bext) + np.array(Bint)).tolist()

    vcasing = vc.VirtualCasing()
    vcasing.setup(digits, nfp, half_period, nt, npol, X, src_nt, src_np, trg_nt, trg_np)

    if xt is None:
        xt = _make_xt_points(3)

    if mode == "ext":
        _ = vcasing.compute_external_B(Btotal)
        _ = vcasing.compute_external_gradB(Btotal)
        _ = vcasing.compute_external_B_offsurf(Btotal, xt, -1, -1)
        _ = vcasing.compute_external_gradB_offsurf(Btotal, xt, -1, -1)
    elif mode == "int":
        _ = vcasing.compute_internal_B(Btotal)
        _ = vcasing.compute_internal_gradB(Btotal)
        _ = vcasing.compute_internal_B_offsurf(Btotal, xt, -1, -1)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def make_virtual_casing_testdata(dst_dir: Path, mode: str):
    import virtual_casing as vc

    make_virtual_casing_case(
        mode=mode,
        nfp=1,
        half_period=False,
        nt=6,
        npol=5,
        src_nt=6,
        src_np=5,
        trg_nt=4,
        trg_np=4,
        digits=5,
        surf_type=vc.SurfType.AxisymNarrow,
        xt=_make_xt_points(3),
    )


def make_virtual_casing_testdata_w7x(dst_dir: Path, mode: str):
    import virtual_casing as vc

    make_virtual_casing_case(
        mode=mode,
        nfp=5,
        half_period=True,
        nt=8,
        npol=6,
        src_nt=8,
        src_np=6,
        trg_nt=6,
        trg_np=5,
        digits=6,
        surf_type=vc.SurfType.W7X_,
        xt=_make_xt_points(3),
    )


def make_virtual_casing_testdata_large(dst_dir: Path, mode: str):
    import virtual_casing as vc

    make_virtual_casing_case(
        mode=mode,
        nfp=1,
        half_period=False,
        nt=14,
        npol=10,
        src_nt=14,
        src_np=10,
        trg_nt=10,
        trg_np=8,
        digits=6,
        surf_type=vc.SurfType.AxisymNarrow,
        xt=_make_xt_points(32),
    )


def make_virtual_casing_testdata_w7x_large(dst_dir: Path, mode: str):
    import virtual_casing as vc

    make_virtual_casing_case(
        mode=mode,
        nfp=5,
        half_period=True,
        nt=12,
        npol=10,
        src_nt=12,
        src_np=10,
        trg_nt=8,
        trg_np=8,
        digits=6,
        surf_type=vc.SurfType.W7X_,
        xt=_make_xt_points(32),
    )


def make_virtual_casing_testdata_dumps(dst_dir: Path):
    import virtual_casing as vc

    nfp = 1
    half_period = False
    nt = 6
    npol = 5
    trg_nt = 4
    trg_np = 4

    X = vc.VirtualCasingTestData.surface_coordinates(nfp, half_period, nt, npol, vc.SurfType.AxisymNarrow)

    _reset_drand48()
    Bext, Bint = vc.VirtualCasingTestData.magnetic_field_data(nfp, half_period, nt, npol, X, trg_nt, trg_np)
    Bext = np.array(Bext).reshape(3, trg_nt, trg_np)
    Bint = np.array(Bint).reshape(3, trg_nt, trg_np)
    _write_dump(dst_dir, "case_testdata_axisym_Bext", Bext)
    _write_dump(dst_dir, "case_testdata_axisym_Bint", Bint)

    _reset_drand48()
    GradBext, GradBint = vc.VirtualCasingTestData.magnetic_field_grad_data(
        nfp, half_period, nt, npol, X, trg_nt, trg_np
    )
    GradBext = np.array(GradBext).reshape(3, 3, trg_nt, trg_np)
    GradBint = np.array(GradBint).reshape(3, 3, trg_nt, trg_np)
    _write_dump(dst_dir, "case_testdata_axisym_GradBext", GradBext)
    _write_dump(dst_dir, "case_testdata_axisym_GradBint", GradBint)


def make_simsopt_vmec_case(
    dst_dir: Path,
    mode: str,
    *,
    src_nphi: int = 8,
    src_ntheta: int = 8,
    trgt_nphi: int = 6,
    trgt_ntheta: int = 6,
    digits: int = 6,
    xt: list[float] | None = None,
):
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
        src_nphi=src_nphi,
        src_ntheta=src_ntheta,
        trgt_nphi=trgt_nphi,
        trgt_ntheta=trgt_ntheta,
        use_stellsym=True,
        digits=digits,
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
        digits,
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
        if xt is not None:
            _ = vcasing.compute_external_B_offsurf(B1d, xt, -1, -1)
            _ = vcasing.compute_external_gradB_offsurf(B1d, xt, -1, -1)
    elif mode == "int":
        _ = vcasing.compute_internal_B(B1d)
        _ = vcasing.compute_internal_gradB(B1d)
        if xt is not None:
            _ = vcasing.compute_internal_B_offsurf(B1d, xt, -1, -1)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def make_simsopt_vmec_case_large(dst_dir: Path, mode: str):
    return make_simsopt_vmec_case(
        dst_dir,
        mode,
        src_nphi=12,
        src_ntheta=12,
        trgt_nphi=10,
        trgt_ntheta=10,
        digits=6,
        xt=_make_xt_points(32),
    )


def run_case(case: str, dst_dir: Path):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        os.environ["VC_DUMP_DIR"] = str(tmpdir)

        if case == "case_vc":
            os.environ["VC_DUMP_PREFIX"] = case
            make_virtual_casing_testdata(dst_dir, mode="ext")
            _copy_prefix(tmpdir, dst_dir, case)
        elif case == "case_vc_int":
            os.environ["VC_DUMP_PREFIX"] = case
            make_virtual_casing_testdata(dst_dir, mode="int")
            _copy_prefix(tmpdir, dst_dir, case)
        elif case == "case_vc_large":
            os.environ["VC_DUMP_PREFIX"] = case
            make_virtual_casing_testdata_large(dst_dir, mode="ext")
            _copy_prefix(tmpdir, dst_dir, case)
        elif case == "case_vc_w7x":
            os.environ["VC_DUMP_PREFIX"] = case
            make_virtual_casing_testdata_w7x(dst_dir, mode="ext")
            _copy_prefix(tmpdir, dst_dir, case)
        elif case == "case_vc_w7x_large":
            os.environ["VC_DUMP_PREFIX"] = case
            make_virtual_casing_testdata_w7x_large(dst_dir, mode="ext")
            _copy_prefix(tmpdir, dst_dir, case)
        elif case == "case_simsopt":
            os.environ["VC_DUMP_PREFIX"] = case
            make_simsopt_vmec_case(dst_dir, mode="ext")
            _copy_prefix(tmpdir, dst_dir, case)
        elif case == "case_simsopt_int":
            os.environ["VC_DUMP_PREFIX"] = case
            make_simsopt_vmec_case(dst_dir, mode="int")
            _copy_prefix(tmpdir, dst_dir, case)
        elif case == "case_simsopt_large":
            os.environ["VC_DUMP_PREFIX"] = case
            make_simsopt_vmec_case_large(dst_dir, mode="ext")
            _copy_prefix(tmpdir, dst_dir, case)
        elif case == "case_testdata_axisym":
            make_virtual_casing_testdata_dumps(dst_dir)
        else:
            raise ValueError(f"Unknown case: {case}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="", help="Run a single case (by prefix)")
    parser.add_argument(
        "--subprocess",
        action="store_true",
        help="Run each case in a separate subprocess to avoid VC_DUMP_PREFIX caching",
    )
    args = parser.parse_args()

    dst_dir = Path(__file__).resolve().parents[1] / "tests" / "data"
    dst_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        "case_vc",
        "case_vc_int",
        "case_vc_large",
        "case_simsopt",
        "case_simsopt_int",
        "case_simsopt_large",
        "case_vc_w7x",
        "case_vc_w7x_large",
        "case_testdata_axisym",
    ]

    if args.case:
        run_case(args.case, dst_dir)
        print(f"Parity dumps written to {dst_dir}")
        return

    if args.subprocess:
        for case in cases:
            subprocess.run(
                [sys.executable, __file__, "--case", case],
                check=True,
            )
        print(f"Parity dumps written to {dst_dir}")
        return

    for case in cases:
        run_case(case, dst_dir)

    print(f"Parity dumps written to {dst_dir}")


if __name__ == "__main__":
    main()
