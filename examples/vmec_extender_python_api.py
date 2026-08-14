"""Evaluate the virtual-casing plasma field from a VMEX ``wout`` file.

Run from an environment containing VMEX and this package:

    python examples/vmec_extender_python_api.py wout_example.nc
"""

from __future__ import annotations

import argparse

import numpy as np

from vmex import read_wout
from vmex.core.freeboundary_diff import surface_field_data_from_wout
from virtual_casing_jax import ExteriorFieldConfig, VirtualCasingExteriorField


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wout", help="VMEC/VMEX wout NetCDF file")
    parser.add_argument("--nphi", type=int, default=32)
    parser.add_argument("--ntheta", type=int, default=32)
    parser.add_argument("--digits", type=int, default=4)
    parser.add_argument(
        "--target",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(2.5, 0.0, 0.0),
        help="Cartesian target point",
    )
    args = parser.parse_args()

    wout = read_wout(args.wout)
    surface = surface_field_data_from_wout(
        wout,
        nphi=args.nphi,
        ntheta=args.ntheta,
    )
    levels = (
        (args.nphi, args.ntheta),
        (2 * args.nphi, 2 * args.ntheta),
    )
    field = VirtualCasingExteriorField(
        surface,
        ExteriorFieldConfig(
            digits=args.digits,
            levels=levels,
            chunk_size="auto",
            target_chunk_size="auto",
        ),
    )

    target_xyz = np.asarray([args.target])
    print("B_plasma_xyz:", np.asarray(field.B_plasma_xyz(target_xyz)))


if __name__ == "__main__":
    main()
