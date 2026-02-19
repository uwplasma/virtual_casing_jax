#!/usr/bin/env python3
"""
W7-X external-field accuracy example.

This mirrors the test in `virtual-casing/test/test.py` and
Figure 2 of Malhotra et al., PPCF 62, 024004 (2020).
"""
from __future__ import annotations

import numpy as np

from virtual_casing_jax import SurfType, VirtualCasingTestData
from virtual_casing_jax.virtual_casing import VirtualCasingJAX


def run():
    expected_errors = [2e-4, 3e-7, 2e-9, 3e-11]
    nfp = 5
    for j in range(4):
        digits = 3 + 3 * j
        for half_period in (True, False):
            nphi_factor = 1 if half_period else 2
            nphi = nphi_factor * 10
            ntheta = 20
            src_nphi = nphi_factor * 6 * digits
            src_ntheta = 32 * digits
            trg_nphi = nphi_factor * 20
            trg_ntheta = 40

            X = VirtualCasingTestData.surface_coordinates(
                nfp, half_period, nphi, ntheta, SurfType.W7X_
            )

            Bext_trg, Bint_trg = VirtualCasingTestData.magnetic_field_data(
                nfp, half_period, nphi, ntheta, X, trg_nphi, trg_ntheta
            )
            Bext_src, Bint_src = VirtualCasingTestData.magnetic_field_data(
                nfp, half_period, nphi, ntheta, X, src_nphi, src_ntheta
            )
            B_total_src = np.asarray(Bext_src) + np.asarray(Bint_src)

            vc = VirtualCasingJAX()
            vc.setup(
                digits,
                nfp,
                half_period,
                nphi,
                ntheta,
                X,
                src_nphi,
                src_ntheta,
                trg_nphi,
                trg_ntheta,
            )

            Bext = vc.compute_external_B(B_total_src)
            B_err = np.asarray(Bext) - np.asarray(Bext_trg)
            max_val = np.abs(B_total_src).max()
            max_rel_err = np.abs(B_err).max() / max_val
            print(
                f"digits={digits:2d} half_period={half_period} "
                f"max_rel_err={max_rel_err:.3e} "
                f"(target < {expected_errors[j]:.1e})"
            )


if __name__ == "__main__":
    run()
