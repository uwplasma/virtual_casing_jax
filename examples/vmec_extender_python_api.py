"""Minimal VMEC-extender Python API example.

Run from a checkout that also has ``vmec_jax`` importable, for example:

    PYTHONPATH=/path/to/vmec_jax:/path/to/virtual_casing_jax \
      python examples/vmec_extender_python_api.py
"""

from __future__ import annotations

import numpy as np

import vmec_jax
from virtual_casing_jax import (
    ExteriorFieldConfig,
    VirtualCasingExteriorField,
    surface_field_from_vmec_jax,
)


def main():
    example = vmec_jax.load_example("circular_tokamak", root="../vmec_jax")
    surface = surface_field_from_vmec_jax(example.state, example.static, example.indata, wout=example.wout)
    field = VirtualCasingExteriorField(
        surface,
        ExteriorFieldConfig(
            digits=3,
            levels=((13, 13),),
            chunk_size=128,
            target_chunk_size=4,
        ),
    )

    target_xyz = np.array([[2.5, 0.0, 0.0]])
    print("B_plasma_xyz:", np.asarray(field.B_plasma_xyz(target_xyz)))
    print("B_cyl:", np.asarray(field.B_cyl(np.array([[2.5, 0.0, 0.0]]))))


if __name__ == "__main__":
    main()
