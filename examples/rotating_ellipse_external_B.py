#!/usr/bin/env python3
"""
Rotating-ellipse external-field accuracy example.

This mirrors `test_rotating_ellipse` from the original virtual-casing test.
It uses an analytic reference field based on elliptic integrals.
"""
from __future__ import annotations

import numpy as np

try:
    from scipy.special import ellipk, ellipe
except Exception as exc:  # pragma: no cover
    raise RuntimeError("This example requires scipy (scipy.special.ellipk/ellipe).") from exc

from virtual_casing_jax.virtual_casing import VirtualCasingJAX


def rotating_ellipse_gamma(nfp: int, half_period: bool, nphi: int, ntheta: int) -> np.ndarray:
    major_radius = 0.7
    minor_radius_a = 0.1
    minor_radius_b = 0.2

    if half_period:
        phi1d = np.linspace(0, np.pi / nfp, nphi, endpoint=False)
        phi1d += 0.5 * (phi1d[1] - phi1d[0])
    else:
        phi1d = np.linspace(0, 2 * np.pi / nfp, nphi, endpoint=False)
    theta1d = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    theta2d, phi2d = np.meshgrid(theta1d, phi1d)

    alpha = theta2d - nfp * phi2d / 2
    u = minor_radius_a * np.cos(alpha)
    v = minor_radius_b * np.sin(alpha)
    cosangle = np.cos(nfp * phi2d / 2)
    sinangle = np.sin(nfp * phi2d / 2)
    r = u * cosangle - v * sinangle + major_radius
    z = u * sinangle + v * cosangle

    x3d = np.zeros((nphi, ntheta, 3))
    x3d[:, :, 0] = r * np.cos(phi2d)
    x3d[:, :, 1] = r * np.sin(phi2d)
    x3d[:, :, 2] = z
    return x3d


def reference_B(nfp: int, half_period: bool, nphi: int, ntheta: int):
    """Return (Bext, Bint) on the rotating-ellipse surface."""
    x3d = rotating_ellipse_gamma(nfp, half_period, nphi, ntheta)

    B_external = np.zeros((nphi, ntheta, 3))
    B_internal = np.zeros((nphi, ntheta, 3))

    r_squared = x3d[:, :, 0] ** 2 + x3d[:, :, 1] ** 2
    B_external[:, :, 0] = -x3d[:, :, 1] / r_squared
    B_external[:, :, 1] = x3d[:, :, 0] / r_squared

    r0 = 0.72
    Inorm = 1.0
    rho = np.sqrt(x3d[:, :, 0] ** 2 + x3d[:, :, 1] ** 2)
    r = np.sqrt(x3d[:, :, 0] ** 2 + x3d[:, :, 1] ** 2 + x3d[:, :, 2] ** 2)
    alpha = np.sqrt(r0 * r0 + r * r - 2 * r0 * rho)
    beta = np.sqrt(r0 * r0 + r * r + 2 * r0 * rho)
    k_squared = 1 - alpha * alpha / (beta * beta)
    ellipek2 = ellipe(k_squared)
    ellipkk2 = ellipk(k_squared)

    B_internal[:, :, 0] = (
        Inorm
        * x3d[:, :, 0]
        * x3d[:, :, 2]
        / (2 * alpha**2 * beta * rho**2 + 1e-31)
        * ((r0**2 + r**2) * ellipek2 - alpha**2 * ellipkk2)
    )
    B_internal[:, :, 1] = (
        Inorm
        * x3d[:, :, 1]
        * x3d[:, :, 2]
        / (2 * alpha**2 * beta * rho**2 + 1e-31)
        * ((r0**2 + r**2) * ellipek2 - alpha**2 * ellipkk2)
    )
    B_internal[:, :, 2] = (
        Inorm
        / (2 * alpha**2 * beta + 1e-31)
        * ((r0**2 - r**2) * ellipek2 + alpha**2 * ellipkk2)
    )
    return B_external, B_internal


def run():
    expected_errors = [2e-4, 2e-8, 1e-9]
    nfp = 4
    for j in range(3):
        digits = 3 + 3 * j
        for half_period in (True, False):
            nphi_factor = 1 if half_period else 2
            nphi = nphi_factor * 10
            ntheta = 20
            src_nphi = nphi_factor * 3 * digits
            src_ntheta = 15 * digits
            trg_nphi = nphi_factor * 19
            trg_ntheta = 29

            X = rotating_ellipse_gamma(nfp, half_period, nphi, ntheta)
            X = np.transpose(X, (2, 0, 1))  # to SoA (3, nphi, ntheta)

            Bext_trg, Bint_trg = reference_B(nfp, half_period, trg_nphi, trg_ntheta)
            Bext_src, Bint_src = reference_B(nfp, half_period, src_nphi, src_ntheta)
            B_total_src = np.transpose(Bext_src + Bint_src, (2, 0, 1))

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
            B_err = np.asarray(Bext) - np.transpose(Bext_trg, (2, 0, 1))
            max_val = np.abs(B_total_src).max()
            max_rel_err = np.abs(B_err).max() / max_val
            print(
                f"digits={digits:2d} half_period={half_period} "
                f"max_rel_err={max_rel_err:.3e} "
                f"(target < {expected_errors[j]:.1e})"
            )


if __name__ == "__main__":
    run()
