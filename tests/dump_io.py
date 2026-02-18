"""Helpers for reading parity dump binaries."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def load_dump(base_path: Path):
    """Load <base>.bin and <base>.json into numpy array."""
    meta_path = base_path.with_suffix(".json")
    bin_path = base_path.with_suffix(".bin")
    if not meta_path.exists() or not bin_path.exists():
        raise FileNotFoundError(f"Missing dump files for {base_path}")

    with meta_path.open("r") as f:
        meta = json.load(f)
    dtype = np.float32 if meta["dtype"] == "float32" else np.float64
    shape = tuple(meta["shape"])
    data = np.fromfile(bin_path, dtype=dtype)
    return data.reshape(shape, order="C")
