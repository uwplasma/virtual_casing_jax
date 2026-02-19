#!/usr/bin/env python3
"""
Verify the phi grid layout for VirtualCasingTestData.surface_coordinates.

This mirrors the `test_test_data_phi_grid` check from the original
virtual-casing test suite.
"""
from __future__ import annotations

import numpy as np

from virtual_casing_jax import SurfType, VirtualCasingTestData


def run():
    nfp = 5
    nphi = 30
    ntheta = 25

    for half_period in (True, False):
        X = VirtualCasingTestData.surface_coordinates(
            nfp, half_period, nphi, ntheta, SurfType.W7X_
        )
        X = np.asarray(X)

        # X is SoA: (3, nphi, ntheta)
        x3d = np.zeros((nphi, ntheta, 3))
        for jxyz in range(3):
            x3d[:, :, jxyz] = X[jxyz]

        phi_from_vc = np.arctan2(x3d[:, :, 1], x3d[:, :, 0])

        if half_period:
            phi1d = np.linspace(0, np.pi / nfp, nphi, endpoint=False)
            phi1d += 0.5 * (phi1d[1] - phi1d[0])
        else:
            phi1d = np.linspace(0, 2 * np.pi / nfp, nphi, endpoint=False)
        theta1d = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)

        theta2d, phi2d = np.meshgrid(theta1d, phi1d)

        np.testing.assert_allclose(phi2d, phi_from_vc, atol=1e-14)
        print(f"phi-grid check passed (half_period={half_period})")


if __name__ == "__main__":
    run()
