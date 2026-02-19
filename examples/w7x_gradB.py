#!/usr/bin/env python3
"""
W7-X GradB accuracy example.

This mirrors the GradB test in `virtual-casing/test/test.py`.
"""
from __future__ import annotations

import numpy as np

from virtual_casing_jax import SurfType, VirtualCasingTestData
from virtual_casing_jax.virtual_casing import VirtualCasingJAX


def run():
    expected_errors = [2e-2, 9e-6, 2e-7, 3e-9]
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

            Bext_src, Bint_src = VirtualCasingTestData.magnetic_field_data(
                nfp, half_period, nphi, ntheta, X, src_nphi, src_ntheta
            )
            B_total_src = np.asarray(Bext_src) + np.asarray(Bint_src)

            GradBext_trg, GradBint_trg = VirtualCasingTestData.magnetic_field_grad_data(
                nfp, half_period, nphi, ntheta, X, trg_nphi, trg_ntheta
            )
            GradB_total_trg = np.asarray(GradBext_trg) + np.asarray(GradBint_trg)

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

            GradBext = vc.compute_external_gradB(B_total_src)
            GradB_err = np.asarray(GradBext) - np.asarray(GradBext_trg)
            max_val = np.abs(GradB_total_trg).max()
            max_rel_err = np.abs(GradB_err).max() / max_val
            print(
                f"digits={digits:2d} half_period={half_period} "
                f"max_rel_grad_err={max_rel_err:.3e} "
                f"(target < {expected_errors[j]:.1e})"
            )


if __name__ == "__main__":
    run()
