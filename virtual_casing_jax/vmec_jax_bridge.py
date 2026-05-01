"""Bridge helpers from ``vmec_jax`` state/static data to virtual casing."""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from .exterior_field import VmecSurfaceFieldData


def _require_vmec_jax():
    try:
        import vmec_jax  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("surface_field_from_vmec_jax requires vmec_jax to be importable") from exc
    return vmec_jax


def _boundary_basis_from_geom(geom, zeta, nfp: int, s_index: int):
    """Return Cartesian boundary position and covariant basis vectors."""
    zeta = jnp.asarray(zeta)
    phi = zeta / int(nfp)
    cosphi = jnp.cos(phi)[None, :]
    sinphi = jnp.sin(phi)[None, :]

    R = geom.R[s_index]
    Z = geom.Z[s_index]
    gamma = jnp.stack((R * cosphi, R * sinphi, Z), axis=-1)

    e_s = jnp.stack((geom.Rs[s_index] * cosphi, geom.Rs[s_index] * sinphi, geom.Zs[s_index]), axis=-1)
    e_theta = jnp.stack((geom.Rt[s_index] * cosphi, geom.Rt[s_index] * sinphi, geom.Zt[s_index]), axis=-1)
    e_phi = jnp.stack(
        (
            geom.Rp[s_index] * cosphi - R * sinphi,
            geom.Rp[s_index] * sinphi + R * cosphi,
            geom.Zp[s_index],
        ),
        axis=-1,
    )
    return gamma, e_s, e_theta, e_phi


def _field_from_state(vmec_jax, state, static, indata, wout, geom, signgs: int):
    from vmec_jax.field import b_cartesian_from_bsup, bsup_from_geom, chips_from_wout_chipf, lamscale_from_phips

    if indata is not None:
        from vmec_jax.energy import flux_profiles_from_indata

        flux = flux_profiles_from_indata(indata, static.s, signgs=signgs)
        phipf = flux.phipf
        chipf = flux.chipf
        lamscale = flux.lamscale
    elif wout is not None:
        scale = jnp.asarray(2.0 * np.pi * float(signgs), dtype=jnp.asarray(wout.phipf).dtype)
        phipf = jnp.asarray(getattr(wout, "phipf_internal", jnp.asarray(wout.phipf) / scale))
        chipf_raw = getattr(wout, "chipf_internal", None)
        if chipf_raw is None:
            chipf_raw = jnp.asarray(wout.chipf) / scale
        chipf = chips_from_wout_chipf(
            chipf=chipf_raw,
            phipf=phipf,
            iotaf=getattr(wout, "iotaf", None),
            iotas=getattr(wout, "iotas", None),
        )
        lamscale = lamscale_from_phips(wout.phips, static.s)
    else:
        raise ValueError("indata is required unless wout supplies flux profiles")

    bsupu, bsupv = bsup_from_geom(
        geom,
        phipf=phipf,
        chipf=chipf,
        nfp=int(static.cfg.nfp),
        signgs=signgs,
        lamscale=lamscale,
    )
    return b_cartesian_from_bsup(geom, bsupu, bsupv, zeta=static.grid.zeta, nfp=int(static.cfg.nfp))


def surface_field_from_vmec_jax(
    state,
    static,
    indata=None,
    *,
    wout=None,
    s_index: int = -1,
    src_nphi: int | None = None,
    src_ntheta: int | None = None,
    use_stellsym: bool = True,
    orientation: str = "auto",
) -> VmecSurfaceFieldData:
    """Return VMEC boundary geometry and total Cartesian field for virtual casing.

    The returned arrays use ``(3, nphi, ntheta)`` layout and physical toroidal
    angle ``phi = zeta / nfp``. Resampling is intentionally not hidden here:
    ``src_nphi`` and ``src_ntheta`` must match the ``static`` grid unless a
    future ``vmec_jax`` fixed-resolution public grid helper is added.
    """
    if orientation not in {"auto", "outward"}:
        raise ValueError("orientation must be 'auto' or 'outward'")

    vmec_jax = _require_vmec_jax()
    from vmec_jax.field import signgs_from_sqrtg
    from vmec_jax.geom import eval_geom

    geom = eval_geom(state, static)
    nfp = int(static.cfg.nfp)
    theta = jnp.asarray(static.grid.theta)
    zeta = jnp.asarray(static.grid.zeta)
    phi = zeta / nfp

    if src_nphi is not None and int(src_nphi) != int(zeta.shape[0]):
        raise NotImplementedError("src_nphi resampling is not yet implemented; rebuild VMECStatic on that grid")
    if src_ntheta is not None and int(src_ntheta) != int(theta.shape[0]):
        raise NotImplementedError("src_ntheta resampling is not yet implemented; rebuild VMECStatic on that grid")

    signgs = int(getattr(wout, "signgs", signgs_from_sqrtg(np.asarray(geom.sqrtg), axis_index=1)))
    gamma_aos, e_s, e_theta, e_phi = _boundary_basis_from_geom(geom, zeta, nfp, int(s_index))
    area_vector_aos = jnp.cross(e_theta, e_phi, axis=-1)
    area_norm = jnp.linalg.norm(area_vector_aos, axis=-1)
    normal_aos = area_vector_aos / jnp.maximum(area_norm[..., None], jnp.asarray(1e-300, dtype=area_norm.dtype))

    mean_radial = jnp.mean(jnp.sum(e_s * normal_aos, axis=-1))
    if orientation in {"auto", "outward"}:
        flip = jnp.where(mean_radial < 0.0, -1.0, 1.0)
        area_vector_aos = flip * area_vector_aos
        normal_aos = flip * normal_aos

    B_aos = _field_from_state(vmec_jax, state, static, indata, wout, geom, signgs)[int(s_index)]

    # geom arrays are (ntheta, nzeta, 3); virtual_casing_jax uses (3, nphi, ntheta).
    gamma = jnp.transpose(gamma_aos, (2, 1, 0))
    B_total = jnp.transpose(B_aos, (2, 1, 0))
    normal = jnp.transpose(normal_aos, (2, 1, 0))
    area_vector = jnp.transpose(area_vector_aos, (2, 1, 0))

    return VmecSurfaceFieldData(
        gamma=gamma,
        B_total=B_total,
        normal=normal,
        area_vector=area_vector,
        theta=theta,
        phi=phi,
        nfp=nfp,
        stellsym=(not bool(getattr(static.cfg, "lasym", False))) and bool(use_stellsym),
        signgs=signgs,
        source_convention="vmec_jax",
    )
