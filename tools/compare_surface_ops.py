#!/usr/bin/env python3
"""Quick comparison harness for surface ops using parity dumps."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import jax.numpy as jnp

from virtual_casing_jax.surface_ops import resample, upsample


def load_dump(base_path: Path):
    meta_path = base_path.with_suffix(".json")
    bin_path = base_path.with_suffix(".bin")
    if not meta_path.exists() or not bin_path.exists():
        raise FileNotFoundError(base_path)
    import json

    with meta_path.open("r") as f:
        meta = json.load(f)
    dtype = np.float32 if meta["dtype"] == "float32" else np.float64
    shape = tuple(meta["shape"])
    data = np.fromfile(bin_path, dtype=dtype)
    return data.reshape(shape, order="C")


def main():
    data_dir = Path(__file__).resolve().parents[1] / "tests" / "data"
    base = data_dir / "case_vc_computeB_B_resampled"
    if not base.with_suffix(".bin").exists():
        raise SystemExit("Missing parity dump: case_vc_computeB_B_resampled")

    B = load_dump(base)
    nt, npol = B.shape[1], B.shape[2]
    Bj = resample(jnp.asarray(B), nt, npol, nt, npol)
    diff = np.max(np.abs(np.asarray(Bj) - B))
    print(f"resample max diff: {diff:.3e}")

    Bj2 = upsample(jnp.asarray(B), nt, npol, nt, npol)
    diff2 = np.max(np.abs(np.asarray(Bj2) - B))
    print(f"upsample max diff: {diff2:.3e}")


if __name__ == "__main__":
    main()
