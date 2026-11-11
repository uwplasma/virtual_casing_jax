"""Regenerate the README figures from the current code.

Every panel evaluates the library against a field known in closed form: a
circular current loop (complete elliptic integrals) inside the surface and a
straight current on the z axis outside it, so the sources inside and outside
are known separately and no panel compares the code with itself.

    python docs/make_readme_figures.py            # every figure
    python docs/make_readme_figures.py near_surface mirror

The speed figure is drawn from ``docs/speed.json``; ``measure_speed`` re-measures it on
the visible devices and rewrites that file (timings are not reproducible bit for
bit, so the plot reads the file).  The committed images are the output of two identical runs,
compressed with ``pngquant --quality 70-95`` and ``optipng``.
"""

import json
import os
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from virtual_casing_jax import ExteriorFieldConfig, VirtualCasingExteriorField, VmecSurfaceFieldData
from virtual_casing_jax.mirror import MirrorExteriorField, mirror_panels

HERE = Path(__file__).resolve().parent
DPI = 110
MU0 = 4e-7 * np.pi
R0, A, DELTA, NFP = 1.0, 0.30, 0.08, 3
CURRENT, B0R0 = 3.0e5, 1.0
C = {"new": "#1f6fb4", "old": "#c8472b", "ref": "#222222", "ext": "#2a9d8f", "int": "#e9a23b", "pv": "#7b5ea7"}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": "white"})


# --- closed-form sources ------------------------------------------------------------------------

def _ellipk_e(m):
    a, b, c, s, p = jnp.ones_like(m), jnp.sqrt(1.0 - m), jnp.sqrt(m), 0.5 * m, 0.5
    for _ in range(14):
        a, b, c = 0.5 * (a + b), jnp.sqrt(a * b), 0.5 * (a - b)
        p *= 2.0
        s = s + p * c**2
    K = jnp.pi / (2.0 * a)
    return K, K * (1.0 - s)


def loop_field(xyz, radius=R0, height=0.0, current=CURRENT):
    """Cartesian field of a circular current loop (away from its own wire); a series on the axis."""
    R2, z = xyz[..., 0] ** 2 + xyz[..., 1] ** 2, xyz[..., 2] - height
    on_axis = R2 < (1e-6 * radius) ** 2
    R = jnp.sqrt(jnp.where(on_axis, 1.0, R2))
    d2 = (R + radius) ** 2 + z**2
    K, E = _ellipk_e(4.0 * R * radius / d2)
    pref, den = MU0 * current / (2.0 * jnp.pi * jnp.sqrt(d2)), (radius - R) ** 2 + z**2
    Bz = pref * (K + (radius**2 - R**2 - z**2) / den * E)
    g = pref * z / R**2 * (-K + (radius**2 + R**2 + z**2) / den * E)
    g = jnp.where(on_axis, 0.75 * MU0 * current * radius**2 * z / (radius**2 + z**2) ** 2.5, g)
    Bz = jnp.where(on_axis, 0.5 * MU0 * current * radius**2 / (radius**2 + z**2) ** 1.5, Bz)
    return jnp.stack([g * xyz[..., 0], g * xyz[..., 1], Bz], -1)


def z_axis_field(xyz):
    r2 = xyz[..., 0] ** 2 + xyz[..., 1] ** 2
    return B0R0 * jnp.stack([-xyz[..., 1] / r2, xyz[..., 0] / r2, 0 * r2], -1)


def inside_sources(xyz):
    return loop_field(xyz)


def outside_sources(xyz):
    return z_axis_field(xyz)


# --- surfaces -----------------------------------------------------------------------------------

def rotating_ellipse(nphi, ntheta, nfp=NFP, a=A, delta=DELTA):
    """Boundary data (one field period) of a rotating ellipse carrying the two sources' total field."""
    theta = jnp.linspace(0.0, 2 * jnp.pi, ntheta, endpoint=False)
    phi = jnp.linspace(0.0, 2 * jnp.pi / nfp, nphi, endpoint=False)
    ph, th = jnp.meshgrid(phi, theta, indexing="ij")
    u = th - nfp * ph
    R = R0 + a * jnp.cos(th) + delta * jnp.cos(u)
    Z = a * jnp.sin(th) + delta * jnp.sin(u)
    Rt, Zt = -a * jnp.sin(th) - delta * jnp.sin(u), a * jnp.cos(th) + delta * jnp.cos(u)
    Rp, Zp = nfp * delta * jnp.sin(u), -nfp * delta * jnp.cos(u)
    c, s = jnp.cos(ph), jnp.sin(ph)
    gamma = jnp.stack([R * c, R * s, Z])
    e_th = jnp.stack([Rt * c, Rt * s, Zt])
    e_ph = jnp.stack([Rp * c - R * s, Rp * s + R * c, Zp])
    area = jnp.cross(e_th, e_ph, axis=0)
    flat = jnp.moveaxis(gamma, 0, -1)
    B = jnp.moveaxis(inside_sources(flat) + outside_sources(flat), -1, 0)
    return VmecSurfaceFieldData(gamma=gamma, B_total=B, normal=area / jnp.linalg.norm(area, axis=0),
                                area_vector=area, theta=theta, phi=phi, nfp=nfp, stellsym=False, signgs=1,
                                source_convention="analytic")


def feet(count, shift=0.37):
    """Surface points and outward normals of the rotating ellipse above feet away from the source nodes."""
    k = np.arange(count)
    phi = (2 * np.pi / (NFP * 32)) * (3 * k + 1 + shift)
    theta = (2 * np.pi / 48) * (4 * k + 2 + shift)
    u = theta - NFP * phi
    R = R0 + A * np.cos(theta) + DELTA * np.cos(u)
    Rt, Zt = -A * np.sin(theta) - DELTA * np.sin(u), A * np.cos(theta) + DELTA * np.cos(u)
    Rp, Zp = NFP * DELTA * np.sin(u), -NFP * DELTA * np.cos(u)
    c, s = np.cos(phi), np.sin(phi)
    point = np.stack([R * c, R * s, A * np.sin(theta) + DELTA * np.sin(u)], 1)
    n = np.cross(np.stack([Rt * c, Rt * s, Zt], 1), np.stack([Rp * c - R * s, Rp * s + R * c, Zp], 1))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    axis = np.stack([R0 * c, R0 * s, 0 * c], 1)
    return point, np.sign(np.sum(n * (point - axis), 1))[:, None] * n


def targets(side, distance, count=8):
    point, outward = feet(count)
    return point + side * distance * A * outward


def exact(points, side):
    """Internal-branch field: the loop outside the surface, minus the z-axis current inside it."""
    p = jnp.asarray(points)
    return np.asarray(inside_sources(p) if side > 0 else -outside_sources(p))


def make_field(nphi=32, ntheta=48, **config):
    return VirtualCasingExteriorField(rotating_ellipse(nphi, ntheta), ExteriorFieldConfig(
        digits=8, src_nphi=nphi, src_ntheta=ntheta, **config))


def save(fig, name):
    path = HERE / f"readme_{name}.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"wrote {path.name}")


def rel(a, b):
    return np.max(np.abs(np.asarray(a) - np.asarray(b)), axis=-1) / np.max(np.abs(b))


# --- figures ------------------------------------------------------------------------------------

def channels():
    """Exterior, interior and principal-value channels on the surface, and the field around it."""
    from virtual_casing_jax import VirtualCasingJAX

    nphi, ntheta = 32, 48
    surf = rotating_ellipse(nphi, ntheta)
    vc = VirtualCasingJAX()
    vc.setup(6, NFP, False, nphi, ntheta, surf.gamma, nphi, ntheta, nphi, ntheta)
    ext, intr = vc.compute_external_B(surf.B_total), vc.compute_internal_B(surf.B_total)
    pv = vc.compute_principal_value_B(surf.B_total)
    gam = np.asarray(surf.gamma)
    truth = {"ext": np.asarray(jnp.moveaxis(outside_sources(jnp.moveaxis(surf.gamma, 0, -1)), -1, 0)),
             "int": np.asarray(jnp.moveaxis(inside_sources(jnp.moveaxis(surf.gamma, 0, -1)), -1, 0))}
    err = max(np.abs(np.asarray(ext) - truth["ext"]).max(), np.abs(np.asarray(intr) - truth["int"]).max()) \
        / np.abs(np.asarray(surf.B_total)).max()
    pv_err = np.abs(np.asarray(pv) - 0.5 * (truth["ext"] - truth["int"])).max() / np.abs(np.asarray(surf.B_total)).max()
    print(f"channels: on-surface errors {err:.1e}, PV {pv_err:.1e}")

    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.0), gridspec_kw={"hspace": 0.38, "wspace": 0.3})
    mag = lambda x: np.linalg.norm(np.asarray(x), axis=0)
    extent = [0, 360 / NFP, 0, 360]
    for ax, (label, data) in zip(axes.ravel(), (("|B| of the sources outside", ext), ("|B| of the sources inside", intr),
                                                ("principal value |ext - int| / 2", pv))):
        im = ax.imshow(mag(data).T, origin="lower", extent=extent, aspect="auto", cmap="viridis")
        ax.set_title(label)
        ax.set_xlabel("toroidal angle (deg)")
        ax.set_ylabel("poloidal angle (deg)")
        fig.colorbar(im, ax=ax, label="T", pad=0.02)
    field = VirtualCasingExteriorField(surf, ExteriorFieldConfig(digits=8, src_nphi=nphi, src_ntheta=ntheta))
    xs, zs = np.linspace(0.45, 1.55, 90), np.linspace(-0.55, 0.55, 80)
    X, Z = np.meshgrid(xs, zs)
    pts = jnp.asarray(np.stack([X.ravel(), 0 * X.ravel(), Z.ravel()], 1))
    out, inn = field.B_plasma_xyz(pts, branch="internal"), field.B_plasma_xyz(pts, branch="external")
    th = np.linspace(0, 2 * np.pi, 400)
    mask = (X - R0) ** 2 + Z**2 < (A + DELTA) ** 2  # at phi = 0 the boundary is a circle
    shown = np.where(mask.ravel(), np.linalg.norm(np.asarray(inn), axis=1), np.linalg.norm(np.asarray(out), axis=1))
    ax = axes[1, 1]
    im = ax.pcolormesh(X, Z, np.log10(shown.reshape(X.shape)), shading="auto", cmap="magma")
    ax.plot(R0 + (A + DELTA) * np.cos(th), (A + DELTA) * np.sin(th), color="white", lw=1.2)
    ax.set_aspect("equal")
    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title(r"$\log_{10}|B|$ at $\phi=0$, both branches")
    fig.colorbar(im, ax=ax, pad=0.02)
    fig.suptitle(f"rotating ellipse, nfp = {NFP}: on-surface channels agree with the exact fields to {err:.0e} of max|B|",
                 fontsize=9.5, y=0.97)
    save(fig, "channels")


def near_surface():
    """Error against distance to the surface, the graded default against the closed-form schedule alone."""
    new, old = make_field(), make_field(near_surface=False)
    dist = np.logspace(-6, 0, 13)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3), sharey=True)
    for ax, side, name in zip(axes, (1, -1), ("outside", "inside")):
        rows = {"new": [], "old": [], "est": []}
        for d in dist:
            p = jnp.asarray(targets(side, d))
            ref = exact(p, side)
            v, est = new.B_plasma_xyz(p, return_estimate=True)
            rows["new"].append(rel(v, ref).max())
            rows["old"].append(rel(old.B_plasma_xyz(p), ref).max())
            rows["est"].append(float(np.max(est)))
        ax.loglog(dist, np.maximum(rows["old"], 1e-17), "o-", color=C["old"], label="closed-form schedule alone")
        ax.loglog(dist, np.maximum(rows["new"], 1e-17), "s-", color=C["new"], label="default (graded near the surface)")
        ax.loglog(dist, rows["est"], "--", color=C["ref"], lw=1, label="returned error estimate")
        ax.axhline(1e-8, color="0.6", lw=0.8, ls=":")
        ax.set_title(f"targets {name} the surface")
        ax.set_xlabel("distance / minor radius")
        ax.set_ylim(1e-16, 3)
        print(f"near_surface {name}: default max {np.max(rows['new']):.1e}, schedule alone max {np.max(rows['old']):.1e}")
    axes[0].set_ylabel(r"$\max|B-B_{exact}|/\max|B|$")
    axes[0].legend(loc="upper right", frameon=False)
    fig.tight_layout()
    save(fig, "near_surface")


def sided_limits():
    """Profile of the field across the surface and the on-surface one-sided limits."""
    from virtual_casing_jax import VirtualCasingJAX

    field = make_field()
    point, outward = feet(1)
    s = np.concatenate([-np.logspace(-1, -7, 25), [0.0], np.logspace(-7, -1, 25)])
    p = jnp.asarray(point + s[:, None] * A * outward)
    v = np.asarray(field.B_plasma_xyz(p))
    n = outward[0]
    truth = np.where((s >= 0)[:, None], np.asarray(inside_sources(p)), -np.asarray(outside_sources(p)))
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    ax = axes[0]
    xs = s
    ax.plot(xs, v @ n * 1e3, "o", ms=3.5, color=C["new"], label="library")
    ax.plot(xs, np.asarray(truth) @ n * 1e3, "-", color=C["ref"], lw=1, label="sources inside (+), minus sources outside (-)")
    b0 = np.asarray(inside_sources(jnp.asarray(point)) + outside_sources(jnp.asarray(point)))[0] @ n * 1e3
    ax.annotate("", xy=(0, (np.asarray(inside_sources(jnp.asarray(point)))[0] @ n) * 1e3),
                xytext=(0, (-np.asarray(outside_sources(jnp.asarray(point)))[0] @ n) * 1e3),
                arrowprops=dict(arrowstyle="<->", color=C["old"]))
    ax.text(3e-4, -8.0, "jump = total B", color=C["old"], fontsize=8, va="center")
    ax.set_xscale("symlog", linthresh=1e-7)
    ax.set_xlim(-0.3, 0.3)
    ax.set_xticks([-1e-1, -1e-3, 0, 1e-3, 1e-1])
    ax.set_xticklabels(["-0.1", "-0.001", "0", "0.001", "0.1"])
    ax.minorticks_off()
    ax.set_xlabel("signed distance / minor radius (normal line)")
    ax.set_ylabel(r"$B\cdot n$ of the internal branch (mT)")
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    ax.set_title("off-surface evaluator across the surface")
    err_line = rel(v, np.asarray(truth)).max()
    print(f"sided_limits: worst profile error {err_line:.1e}")

    ax = axes[1]
    digs = (2, 3, 4, 5, 6, 7, 8)
    nphi, ntheta = 32, 48
    surf = rotating_ellipse(nphi, ntheta)
    flat = jnp.moveaxis(surf.gamma, 0, -1)
    te = np.asarray(jnp.moveaxis(outside_sources(flat), -1, 0))
    ti = np.asarray(jnp.moveaxis(inside_sources(flat), -1, 0))
    scale = np.abs(np.asarray(surf.B_total)).max()
    ee, ei = [], []
    for d in digs:
        vc = VirtualCasingJAX()
        vc.setup(d, NFP, False, nphi, ntheta, surf.gamma, nphi, ntheta, nphi, ntheta)
        ee.append(np.abs(np.asarray(vc.compute_external_B(surf.B_total)) - te).max() / scale)
        ei.append(np.abs(np.asarray(vc.compute_internal_B(surf.B_total)) - ti).max() / scale)
    ax.semilogy(digs, ee, "o-", ms=7, color=C["ext"], label="external (sources outside)")
    ax.semilogy(digs, ei, "s--", ms=3.5, color=C["int"], label="internal (sources inside)")
    ax.semilogy(digs, [10.0**-d for d in digs], ":", color="0.5", label=r"$10^{-\mathrm{digits}}$")
    ax.set_xlabel("requested digits")
    ax.set_ylabel(r"max error / max|B| on the surface")
    ax.set_title("on-surface limits, 32 x 48 samples")
    ax.legend(frameon=False, fontsize=7.5)
    print("sided_limits: on-surface errors", ["%.1e" % e for e in ee], ["%.1e" % e for e in ei])
    fig.tight_layout()
    save(fig, "sided_limits")


def jets():
    """Derivative orders 0-3 against closed-form (automatic) derivatives of the loop field."""
    field = make_field(64, 64)
    dist = np.logspace(-5, 0, 11)
    exact_jets = lambda x: loop_field(x)
    d1 = jax.jacfwd(exact_jets)
    d2, d3 = jax.jacfwd(d1), jax.jacfwd(jax.jacfwd(d1))
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3), sharey=True)
    for ax, order_set, title in ((axes[0], "accurate", "default evaluator"), (axes[1], "closed", "closed forms only")):
        for k, color in zip(range(4), ("#1f6fb4", "#2a9d8f", "#e9a23b", "#c8472b")):
            errs = []
            for d in dist:
                p = jnp.asarray(targets(1, d, 6))
                if order_set == "accurate":
                    got = field.B_and_derivatives_accurate_xyz(p, order=3, tol=1e-8)[0][k]
                else:
                    got = field.B_and_derivatives_xyz(p, order=3)[k]
                ref = [lambda x: loop_field(x), d1, d2, d3][k]
                ref = jax.vmap(ref)(p)
                scale = np.max(np.abs(ref))
                errs.append(np.max(np.abs(np.asarray(got) - np.asarray(ref))) / scale)
            ax.loglog(dist, np.maximum(errs, 1e-17), "o-", ms=3.5, color=color, label=f"order {k}")
            print(f"jets {title} order {k}: max {np.max(errs):.1e} min {np.min(errs):.1e}")
        ax.set_title(title)
        ax.set_xlabel("distance / minor radius (outside)")
        ax.set_ylim(1e-16, 10)
    axes[0].set_ylabel(r"max error / max |$D^kB$|")
    axes[0].legend(frameon=False, ncol=2)
    fig.tight_layout()
    save(fig, "jets")


def gradients():
    """Reverse-mode gradients through the targets, the surface field and the surface shape against finite differences."""
    nphi, ntheta = 32, 48
    surf = rotating_ellipse(nphi, ntheta)
    p0 = jnp.asarray(targets(1, 0.05, 4)[:1] + 0.0)

    def make(gamma, B):
        s = VmecSurfaceFieldData(gamma=gamma, B_total=B, normal=surf.normal, area_vector=surf.area_vector,
                                 theta=surf.theta, phi=surf.phi, nfp=NFP, stellsym=False, signgs=1)
        return VirtualCasingExteriorField(s, ExteriorFieldConfig(digits=8, src_nphi=nphi, src_ntheta=ntheta, validate_orientation=False))

    def loss_target(p):
        return jnp.sum(make(surf.gamma, surf.B_total).B_plasma_xyz(p) ** 2)

    def loss_field(B):
        return jnp.sum(make(surf.gamma, B).B_plasma_xyz(p0) ** 2)

    def loss_shape(gamma):
        return jnp.sum(make(gamma, surf.B_total).B_plasma_xyz(p0) ** 2)

    rng = np.random.default_rng(0)
    cases = []
    for name, f, x in (("target position", loss_target, p0), ("boundary field samples", loss_field, surf.B_total),
                       ("boundary shape", loss_shape, surf.gamma)):
        g = jax.grad(f)(x)
        d = jnp.asarray(rng.standard_normal(x.shape))
        d = d / jnp.linalg.norm(d)
        jvp = float(jnp.vdot(g, d))
        row = []
        for h in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
            h = h * float(jnp.linalg.norm(x) / np.sqrt(x.size)) * 1.0
            fd = (float(f(x + h * d)) - float(f(x - h * d))) / (2 * h)
            row.append((h, abs(fd - jvp) / abs(jvp)))
        cases.append((name, row, jvp))
        print(f"gradients {name}: directional derivative {jvp:.4e}, best FD agreement {min(r[1] for r in row):.1e}")
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    for (name, row, _), color in zip(cases, (C["new"], C["ext"], C["old"])):
        ax.loglog([r[0] for r in row], [r[1] for r in row], "o-", color=color, label=name)
    ax.set_xlabel("finite-difference step h")
    ax.set_ylabel(r"$|\mathrm{FD}-\mathrm{jax.grad}|\,/\,|\mathrm{jax.grad}|$")
    ax.set_title("jax.grad of $|B|^2$ against central differences")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "gradients")


def mirror():
    """Mirror surface: field map around the cylinder and error against distance to wall, cap and rim."""
    z0, z1 = -1.0, 1.0
    inner = lambda x: loop_field(x, 0.5, 0.1, 1e6)
    outer = lambda x: loop_field(x, 1.5, 0.8, -2e6) + loop_field(x, 1.4, -0.9, -1.5e6)
    total = lambda x: inner(x) + outer(x)
    radius = lambda z, v: 1.0 + 0 * z + 0 * v
    pos = lambda z, v: jnp.stack((radius(z, v) * jnp.cos(v), radius(z, v) * jnp.sin(v), z))
    cap = lambda z: lambda w, v: total(jnp.stack((w * radius(z, v) * jnp.cos(v), w * radius(z, v) * jnp.sin(v), z + 0 * v)))
    field = MirrorExteriorField(mirror_panels(radius, z0, z1, lambda z, v: total(pos(z, v)), cap(z0), cap(z1)))
    scale = float(jnp.max(jnp.linalg.norm(jax.vmap(inner)(jnp.array([[1.3, 0.4, 0.9], [0.0, 2.0, -0.3]])), axis=-1)))
    fig = plt.figure(figsize=(7.6, 3.5), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15])
    ax = fig.add_subplot(gs[0])
    xs, zs = np.linspace(0.02, 2.2, 44), np.linspace(-1.8, 1.8, 56)
    X, Z = np.meshgrid(xs, zs)
    pts = jnp.asarray(np.stack([X.ravel(), 0 * X.ravel(), Z.ravel()], 1))
    out_mask = ~((X < 1.0) & (np.abs(Z) < 1.0))
    v, _ = field.B_and_derivatives_xyz(pts, order=0)
    fin, _ = field.B_and_derivatives_xyz(pts, order=0, branch="external")
    val = np.where(out_mask.ravel(), np.linalg.norm(np.asarray(v[0]), axis=1), np.linalg.norm(np.asarray(fin[0]), axis=1))
    im = ax.pcolormesh(X, Z, np.log10(val.reshape(X.shape)), shading="auto", cmap="magma")
    ax.plot([1, 1, 0], [-1, 1, 1], color="white", lw=1.3)
    ax.plot([0, 1], [-1, -1], color="white", lw=1.3)
    ax.set_aspect("equal")
    ax.set_xlabel("r (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(r"$\log_{10}|B|$, both branches")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    ax = fig.add_subplot(gs[1])
    dist = np.logspace(-9, -1, 9)
    kinds = {"wall": lambda d: [1.0 + d, 0.0, 0.2], "cap": lambda d: [0.4, 0.2, 1.0 + d], "rim": lambda d: [1.0 + d / 1.4, 0.0, 1.0 + d / 1.4]}
    for (name, at), color in zip(kinds.items(), (C["new"], C["ext"], C["old"])):
        errs, est = [], []
        for d in dist:
            p = jnp.asarray([at(d)])
            (val,), e = field.B_and_derivatives_xyz(p, order=0)
            errs.append(float(jnp.max(jnp.abs(val[0] - inner(p[0]))) / scale))
            est.append(float(e[0]))
        ax.loglog(dist, np.maximum(errs, 1e-17), "o-", ms=3.5, color=color, label=name)
        print(f"mirror {name}: max error {max(errs):.1e}, max estimate {max(est):.1e}")
    ax.set_xlabel("distance from the surface / radius")
    ax.set_ylabel("error / max source field")
    ax.set_title("error approaching wall, cap and rim")
    ax.set_ylim(1e-14, 1e-3)
    ax.legend(frameon=False)
    save(fig, "mirror")


def topology():
    """Flux and circulation: the net flux through the surface and the circulation that links the sources."""
    field = make_field()
    from virtual_casing_jax.mirror import MirrorExteriorField as M

    z0, z1 = -1.0, 1.0
    total = lambda x: loop_field(x, 0.5, 0.1, 1e6) + loop_field(x, 1.5, 0.8, -2e6)
    radius = lambda z, v: 1.0 + 0 * z + 0 * v
    pos = lambda z, v: jnp.stack((radius(z, v) * jnp.cos(v), radius(z, v) * jnp.sin(v), z))
    cap = lambda z: lambda w, v: total(jnp.stack((w * jnp.cos(v), w * jnp.sin(v), z + 0 * v)))
    zero_cap = lambda w, v: 0 * total(jnp.stack((w * jnp.cos(v), w * jnp.sin(v), z0 + 0 * v)))
    good = M(mirror_panels(radius, z0, z1, lambda z, v: total(pos(z, v)), cap(z0), cap(z1))).flux_defect()
    bad = M(mirror_panels(radius, z0, z1, lambda z, v: total(pos(z, v)), zero_cap, zero_cap)).flux_defect()
    print(f"topology: flux defect with cap flux {float(good):.1e}, without {float(bad):.2f}")

    rho = np.concatenate([np.linspace(0.05, 0.45, 41)])
    ang = np.linspace(0, 2 * np.pi, 256, endpoint=False)
    circ = []
    for r in rho:
        loop = np.stack([R0 + r * np.cos(ang), 0 * ang, r * np.sin(ang)], 1)
        tangent = np.stack([np.sin(ang), 0 * ang, -np.cos(ang)], 1) * r * (2 * np.pi / len(ang))
        circ.append(np.sum(np.asarray(field.B_plasma_xyz(jnp.asarray(loop))) * tangent) / (MU0 * CURRENT))
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), gridspec_kw={"width_ratios": [1.5, 1]}, layout="constrained")
    ax = axes[0]
    ax.plot(rho / A, circ, "o-", ms=3, color=C["new"], label="library, branch \"internal\"")
    ax.axvline((A + DELTA) / A, color="0.6", ls=":", label="surface in this plane")
    ax.set_xlabel("radius of the poloidal loop / minor radius")
    ax.set_ylabel(r"circulation / $\mu_0 I$")
    ax.set_title("circulation around the tube")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax = axes[1]
    ax.bar(["cap flux\nincluded", "cap flux\nomitted"], [max(float(good), 1e-17), float(bad)], color=[C["new"], C["old"]])
    ax.set_yscale("log")
    ax.set_ylim(1e-16, 3)
    ax.set_ylabel("|net flux| / total |flux|")
    ax.set_title("mirror: flux through the caps")
    save(fig, "topology")
    print("topology: circulation / mu0 I at the innermost and outermost loops:", circ[0], circ[-1])


def measure_speed():
    """Time the default evaluator on the first visible device (label it with ``SPEED_LABEL``) and add it to ``speed.json``."""
    nt = 2000
    out = {}
    for name, field_kw in (("graded default", {}), ("closed forms only", {"near_surface": False})):
        field = make_field(64, 64, **field_kw)
        for dist in (1.0, 0.01):
            p = jnp.asarray(targets(1, dist, 8)).repeat(nt // 8, 0)
            fn = jax.jit(lambda x: field.B_plasma_xyz(x))
            fn(p).block_until_ready()
            t = time.perf_counter()
            for _ in range(3):
                fn(p).block_until_ready()
            out[f"{name}, d/a={dist}"] = nt * 3 / (time.perf_counter() - t)
    device = jax.devices()[0]
    path = HERE / "speed.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    label = os.environ.get("SPEED_LABEL") or f"{device.platform}: {device.device_kind}"
    data[label] = {k: round(v, 1) for k, v in out.items()}
    path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    print(json.dumps(data, indent=1))


def speed():
    """Targets per second of the default evaluator, per device, from ``speed.json``."""
    data = json.loads((HERE / "speed.json").read_text())
    keys = ["graded default, d/a=1.0", "graded default, d/a=0.01", "closed forms only, d/a=0.01"]
    labels = ["default, 1 a away", "default, 0.01 a away", "closed forms alone,\n0.01 a (wrong: error ~1)"]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    width = 0.8 / len(data)
    for i, (device, row) in enumerate(sorted(data.items())):
        ax.bar(np.arange(len(keys)) + i * width, [row[k] for k in keys], width, label=device, color=(C["new"], C["ext"], C["old"])[i])
    ax.set_xticks(np.arange(len(keys)) + 0.4 - width / 2)
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.set_ylabel("targets per second (B, 64x64 data)")
    ax.set_ylim(top=1e5)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("B_plasma_xyz throughput by device")
    fig.tight_layout()
    save(fig, "speed")


FIGURES = {f.__name__: f for f in (channels, near_surface, sided_limits, jets, gradients, mirror, topology, speed)}

if __name__ == "__main__":
    args = sys.argv[1:] or list(FIGURES)
    for name in args:
        if name == "measure_speed":
            measure_speed()
        else:
            FIGURES[name]()
