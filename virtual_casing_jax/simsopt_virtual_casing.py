"""SIMSOPT-compatible VirtualCasing class backed by virtual_casing_jax."""
from __future__ import annotations

import os
import logging
from datetime import datetime

import numpy as np
from scipy.io import netcdf_file

from .virtual_casing import VirtualCasingJAX

logger = logging.getLogger(__name__)


def _soa_from_3d(arr3d: np.ndarray) -> np.ndarray:
    """Convert (nphi, ntheta, 3) -> (3, nphi, ntheta)."""
    return np.transpose(arr3d, (2, 0, 1))


def _3d_from_soa(arr_soa: np.ndarray) -> np.ndarray:
    """Convert (3, nphi, ntheta) -> (nphi, ntheta, 3)."""
    return np.transpose(arr_soa, (1, 2, 0))


class VirtualCasing:
    r"""
    SIMSOPT-compatible VirtualCasing class backed by JAX.

    This class mirrors ``simsopt.mhd.virtual_casing.VirtualCasing`` so it
    can be imported as:

    ``from virtual_casing_jax import VirtualCasing``
    """

    @classmethod
    def from_vmec(
        cls,
        vmec,
        src_nphi,
        src_ntheta=None,
        trgt_nphi=None,
        trgt_ntheta=None,
        use_stellsym=True,
        digits=6,
        filename="auto",
    ):
        """
        Create a VirtualCasing object from a VMEC equilibrium.

        This routine uses simsopt's VMEC utilities and computes the
        external field using VirtualCasingJAX.
        """
        try:
            from simsopt.mhd.vmec_diagnostics import B_cartesian
            from simsopt.mhd.vmec import Vmec
            from simsopt.geo.surfacerzfourier import SurfaceRZFourier
            from simsopt.geo.surface import best_nphi_over_ntheta
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("simsopt is required for from_vmec().") from exc

        if not isinstance(vmec, Vmec):
            vmec = Vmec(vmec)

        vmec.run()
        nfp = vmec.wout.nfp
        stellsym = (not bool(vmec.wout.lasym)) and use_stellsym
        if vmec.wout.lasym:
            raise RuntimeError("virtual casing presently only works for stellarator symmetry")

        if src_ntheta is None:
            src_ntheta = int(
                (1 + int(stellsym)) * nfp * src_nphi / best_nphi_over_ntheta(vmec.boundary)
            )
            logger.info("new src_ntheta: %s", src_ntheta)

        ran = "half period" if stellsym else "field period"
        surf = SurfaceRZFourier.from_nphi_ntheta(
            mpol=vmec.wout.mpol,
            ntor=vmec.wout.ntor,
            nfp=nfp,
            nphi=src_nphi,
            ntheta=src_ntheta,
            range=ran,
        )
        for jmn in range(vmec.wout.mnmax):
            surf.set_rc(int(vmec.wout.xm[jmn]), int(vmec.wout.xn[jmn] / nfp), vmec.wout.rmnc[jmn, -1])
            surf.set_zs(int(vmec.wout.xm[jmn]), int(vmec.wout.xn[jmn] / nfp), vmec.wout.zmns[jmn, -1])

        Bxyz = B_cartesian(vmec, nphi=src_nphi, ntheta=src_ntheta, range=ran)
        gamma = surf.gamma()

        if trgt_nphi is None:
            trgt_nphi = src_nphi
        if trgt_ntheta is None:
            trgt_ntheta = src_ntheta
        trgt_surf = SurfaceRZFourier.from_nphi_ntheta(
            mpol=vmec.wout.mpol,
            ntor=vmec.wout.ntor,
            nfp=nfp,
            nphi=trgt_nphi,
            ntheta=trgt_ntheta,
            range=ran,
        )
        trgt_surf.x = surf.x

        unit_normal = trgt_surf.unitnormal()

        # Convert to SoA for VirtualCasingJAX
        gamma_soa = _soa_from_3d(gamma)
        B_total_soa = np.asarray(Bxyz)
        B3d = _3d_from_soa(B_total_soa)

        vc_jax = VirtualCasingJAX()
        vc_jax.setup(
            digits,
            nfp,
            stellsym,
            src_nphi,
            src_ntheta,
            gamma_soa,
            src_nphi,
            src_ntheta,
            trgt_nphi,
            trgt_ntheta,
        )
        Bexternal_soa = vc_jax.compute_external_B(B_total_soa, digits=digits)
        Bexternal3d = _3d_from_soa(np.asarray(Bexternal_soa))

        Bexternal_normal = np.sum(Bexternal3d * unit_normal, axis=2)

        vc = cls()
        vc.src_ntheta = src_ntheta
        vc.src_nphi = src_nphi
        vc.src_theta = surf.quadpoints_theta
        vc.src_phi = surf.quadpoints_phi

        vc.trgt_ntheta = trgt_ntheta
        vc.trgt_nphi = trgt_nphi
        vc.trgt_theta = trgt_surf.quadpoints_theta
        vc.trgt_phi = trgt_surf.quadpoints_phi

        vc.nfp = nfp
        vc.B_total = B3d
        vc.gamma = gamma
        vc.unit_normal = unit_normal
        vc.B_external = Bexternal3d
        vc.B_external_normal = Bexternal_normal

        Bexternal_normal_with_last_point = np.hstack((Bexternal_normal, Bexternal_normal[:, [0]]))
        Bexternal_normal_with_last_point = np.vstack(
            (Bexternal_normal_with_last_point, -np.flip(np.flip(Bexternal_normal_with_last_point, axis=0), axis=1)[0])
        )
        flipped_B = -np.flip(np.flip(Bexternal_normal_with_last_point, axis=0), axis=1)
        vc.B_external_normal_extended = np.concatenate(
            [np.concatenate((Bexternal_normal, flipped_B[:-1, :-1])) for _ in range(nfp)]
        )

        if filename is not None:
            if filename == "auto":
                directory, basefile = os.path.split(vmec.output_file)
                filename = os.path.join(directory, "vcasing" + basefile[4:])
                logger.debug("New filename: %s", filename)
            vc.save(filename)

        return vc

    def save(self, filename="vcasing.nc"):
        """Save the results of a virtual casing calculation in a NetCDF file."""
        with netcdf_file(filename, "w") as f:
            f.history = "This file created by virtual_casing_jax on " + datetime.now().strftime(
                "%B %d %Y, %H:%M:%S"
            )
            f.createDimension("src_ntheta", self.src_ntheta)
            f.createDimension("src_nphi", self.src_nphi)
            f.createDimension("trgt_ntheta", self.trgt_ntheta)
            f.createDimension("trgt_nphi", self.trgt_nphi)
            f.createDimension("trgt_nphi_extended", self.trgt_nphi * 2 * self.nfp)
            f.createDimension("xyz", 3)

            src_ntheta = f.createVariable("src_ntheta", "i", tuple())
            src_ntheta.data[()] = self.src_ntheta
            src_ntheta.description = "Number of grid points in poloidal angle theta"
            src_ntheta.units = "Dimensionless"

            trgt_ntheta = f.createVariable("trgt_ntheta", "i", tuple())
            trgt_ntheta.data[()] = self.trgt_ntheta
            trgt_ntheta.description = "Number of grid points in poloidal angle theta for output"
            trgt_ntheta.units = "Dimensionless"

            src_nphi = f.createVariable("src_nphi", "i", tuple())
            src_nphi.data[()] = self.src_nphi
            src_nphi.description = "Number of grid points in toroidal angle phi"
            src_nphi.units = "Dimensionless"

            trgt_nphi = f.createVariable("trgt_nphi", "i", tuple())
            trgt_nphi.data[()] = self.trgt_nphi
            trgt_nphi.description = "Number of grid points in toroidal angle phi for output"
            trgt_nphi.units = "Dimensionless"

            nfp = f.createVariable("nfp", "i", tuple())
            nfp.data[()] = self.nfp
            nfp.description = "Periodicity in toroidal direction"
            nfp.units = "Dimensionless"

            src_theta = f.createVariable("src_theta", "d", ("src_ntheta",))
            src_theta[:] = self.src_theta
            src_theta.description = "Grid points in poloidal angle theta"
            src_theta.units = "Dimensionless"

            trgt_theta = f.createVariable("trgt_theta", "d", ("trgt_ntheta",))
            trgt_theta[:] = self.trgt_theta
            trgt_theta.description = "Grid points in poloidal angle theta for output"
            trgt_theta.units = "Dimensionless"

            src_phi = f.createVariable("src_phi", "d", ("src_nphi",))
            src_phi[:] = self.src_phi
            src_phi.description = "Grid points in toroidal angle phi"
            src_phi.units = "Dimensionless"

            trgt_phi = f.createVariable("trgt_phi", "d", ("trgt_nphi",))
            trgt_phi[:] = self.trgt_phi
            trgt_phi.description = "Grid points in toroidal angle phi for output"
            trgt_phi.units = "Dimensionless"

            gamma = f.createVariable("gamma", "d", ("src_nphi", "src_ntheta", "xyz"))
            gamma[:, :, :] = self.gamma
            gamma.description = "Position vector on the boundary surface"
            gamma.units = "meter"

            unit_normal = f.createVariable("unit_normal", "d", ("trgt_nphi", "trgt_ntheta", "xyz"))
            unit_normal[:, :, :] = self.unit_normal
            unit_normal.description = "Unit-length normal vector on the boundary surface"
            unit_normal.units = "Dimensionless"

            B_total = f.createVariable("B_total", "d", ("src_nphi", "src_ntheta", "xyz"))
            B_total[:, :, :] = self.B_total
            B_total.description = "Total magnetic field vector on the surface"
            B_total.units = "Tesla"

            B_external = f.createVariable("B_external", "d", ("trgt_nphi", "trgt_ntheta", "xyz"))
            B_external[:, :, :] = self.B_external
            B_external.description = "Contribution to the magnetic field due to currents outside"
            B_external.units = "Tesla"

            B_external_normal = f.createVariable("B_external_normal", "d", ("trgt_nphi", "trgt_ntheta"))
            B_external_normal[:, :] = self.B_external_normal
            B_external_normal.description = "Component of B_external normal to the surface"
            B_external_normal.units = "Tesla"

            B_external_normal_extended = f.createVariable(
                "B_external_normal_extended", "d", ("trgt_nphi_extended", "trgt_ntheta")
            )
            B_external_normal_extended[:, :] = self.B_external_normal_extended
            B_external_normal_extended.description = "Extended normal component over full torus"
            B_external_normal_extended.units = "Tesla"

    @classmethod
    def load(cls, filename):
        """Load a virtual casing solution from a NetCDF file."""
        vc = cls()
        with netcdf_file(filename, mmap=False) as f:
            for key, val in f.variables.items():
                vc.__setattr__(key, val[()])
        return vc

    def plot(self, ax=None, show=True):
        """Plot B_external_normal and B_external_normal_extended."""
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = plt.gcf()
        contours = ax.contourf(self.trgt_phi, self.trgt_theta, self.B_external_normal.T, 25)
        ax.set_xlabel(r"$\phi$")
        ax.set_ylabel(r"$\theta$")
        ax.set_title("B_external_normal [Tesla]")
        fig.colorbar(contours)
        fig.tight_layout()

        fig1, ax1 = plt.subplots()
        shape = self.B_external_normal_extended.T.shape
        contours = ax1.contourf(
            np.linspace(0, 1, shape[1]),
            np.linspace(0, 1, shape[0]),
            self.B_external_normal_extended.T,
            25,
        )
        ax1.set_xlabel(r"$\phi$")
        ax1.set_ylabel(r"$\theta$")
        ax1.set_title("B_external_normal_extended [Tesla]")
        fig1.colorbar(contours)
        fig1.tight_layout()

        if show:
            plt.show()
        return ax
