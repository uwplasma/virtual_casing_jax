#!/usr/bin/env python3
"""Convert SCTL .mat geometry files to bundled .npz assets."""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


def read_sctl_matrix(path: Path, dtype=np.float64) -> np.ndarray:
    with path.open("rb") as f:
        header = f.read(16)
        if len(header) != 16:
            raise ValueError(f"Invalid SCTL matrix header in {path}")
        n0, n1 = struct.unpack("QQ", header)
        data = np.fromfile(f, dtype=dtype, count=n0 * n1)
    if data.size != n0 * n1:
        raise ValueError(f"Expected {n0*n1} entries in {path}, got {data.size}")
    return data.reshape((n0, n1))


def convert_surface(name: str, src_dir: Path, dst_dir: Path, scale: float, drop_last: bool):
    X = read_sctl_matrix(src_dir / f"{name}-X.mat")
    Y = read_sctl_matrix(src_dir / f"{name}-Y.mat")
    Z = read_sctl_matrix(src_dir / f"{name}-Z.mat")

    if drop_last:
        X = X[:-1, :-1]
        Y = Y[:-1, :-1]
        Z = Z[:-1, :-1]

    X = X.astype(np.float64) * scale
    Y = Y.astype(np.float64) * scale
    Z = Z.astype(np.float64) * scale

    dst_dir.mkdir(parents=True, exist_ok=True)
    out_path = dst_dir / f"{name}.npz"
    np.savez(out_path, X=X, Y=Y, Z=Z, scale=np.array(scale), drop_last=np.array(int(drop_last)))
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("/Users/rogerio/local/virtual-casing/extern/BIEST/geom"),
        help="Path to geom/*.mat directory",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "virtual_casing_jax" / "geom",
        help="Output directory for .npz files",
    )
    args = parser.parse_args()

    convert_surface("Quas3", args.src, args.dst, scale=0.45, drop_last=True)
    convert_surface("LHD", args.src, args.dst, scale=0.25, drop_last=False)
    convert_surface("W7X", args.src, args.dst, scale=0.45, drop_last=False)

    print(f"Wrote .npz files to {args.dst}")


if __name__ == "__main__":
    main()
