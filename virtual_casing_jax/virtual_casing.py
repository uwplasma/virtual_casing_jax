"""High-level Virtual Casing routines in JAX."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from .utils import autotune_chunk_sizes
from .surface_ops import (
    complete_vec_field,
    resample,
    rotate_toroidal,
    grad2d,
    surf_normal_area_elem,
    dot_prod,
    cross_prod,
)
from .integrals import (
    laplace_fxd_u_eval,
    laplace_fxd_u_eval_singular,
    laplace_fxd_u_eval_vec_singular,
    laplace_fxd2_u_eval,
    laplace_fxd2_u_eval_vec,
    laplace_fxd2_u_eval_singular,
    laplace_fxd2_u_eval_vec_singular,
    laplace_dx_u_eval_singular,
    computeB_offsurface_adaptive,
    computeB_offsurface_adaptive_schedule,
    computeGradB_offsurface_adaptive_schedule,
    _offsurface_adapt_grid,
    _build_patch_indices,
    _surface_cond,
    select_patch_dim,
)


@dataclass
class QuadSetup:
    quad_nt: int
    quad_np: int
    quad_coord: jnp.ndarray
    dX: jnp.ndarray
    normal: jnp.ndarray
    orient: float
    patch_idx_cache: dict[int, jnp.ndarray] = field(default_factory=dict)


class VirtualCasingJAX:
    """JAX mirror of VirtualCasing for external field and GradB."""

    def __init__(self):
        self._setup = False
        self._grad_setup: QuadSetup | None = None
        self._b_setup: QuadSetup | None = None
        self._jit_cache: dict[tuple, callable] = {}

    def _resolve_chunk_sizes(
        self,
        op: str,
        chunk_size,
        target_chunk_size,
        *,
        nsrc: int,
        ntrg: int,
    ):
        """Resolve source/target chunk sizes with optional auto tuning."""
        chunk_auto = chunk_size is None or (isinstance(chunk_size, str) and chunk_size.lower() == "auto")
        target_auto = isinstance(target_chunk_size, str) and target_chunk_size.lower() == "auto"

        if chunk_auto:
            src_auto, trg_auto = autotune_chunk_sizes(op, nsrc, ntrg)
            chunk_size = src_auto
            if target_auto:
                target_chunk_size = trg_auto
        else:
            chunk_size = int(chunk_size)
            if target_auto:
                _, trg_auto = autotune_chunk_sizes(op, nsrc, ntrg)
                target_chunk_size = trg_auto

        if target_chunk_size is not None and not isinstance(target_chunk_size, str):
            target_chunk_size = int(target_chunk_size)

        return chunk_size, target_chunk_size

    def _resolve_pou_dtype(self, pou_dtype, value_dtype):
        if pou_dtype is None:
            return None
        if isinstance(pou_dtype, str):
            if pou_dtype.lower() == "auto":
                return jnp.float32 if value_dtype == jnp.float64 else value_dtype
            return jnp.dtype(pou_dtype)
        return jnp.dtype(pou_dtype)

    def _resolve_patch_dtype(self, patch_dtype, value_dtype):
        if patch_dtype is None:
            return None
        if isinstance(patch_dtype, str):
            if patch_dtype.lower() == "auto":
                return jnp.float32 if value_dtype == jnp.float64 else value_dtype
            return jnp.dtype(patch_dtype)
        return jnp.dtype(patch_dtype)

    def setup(
        self,
        digits: int,
        nfp: int,
        half_period: bool,
        surf_nt: int,
        surf_np: int,
        X,
        src_nt: int,
        src_np: int,
        trg_nt: int,
        trg_np: int,
    ):
        X = jnp.asarray(X).reshape((3, surf_nt, surf_np))

        if half_period:
            X0 = complete_vec_field(
                X,
                True,
                half_period,
                nfp,
                surf_nt,
                surf_np,
                -math.pi / (nfp * surf_nt * 2),
            )
            X1 = resample(X0, nfp * 2 * surf_nt, surf_np, nfp * 2 * (surf_nt + 1), surf_np)
            surface_coord = rotate_toroidal(
                X1,
                nfp * 2 * (surf_nt + 1),
                surf_np,
                math.pi / (nfp * trg_nt * 2),
            )
            nfp_eff = nfp * 2
        else:
            surface_coord = complete_vec_field(
                X,
                True,
                half_period,
                nfp,
                surf_nt,
                surf_np,
                0.0,
            )
            nfp_eff = nfp

        self.digits = int(digits)
        self.nfp = int(nfp)
        self.nfp_eff = int(nfp_eff)
        self.half_period = bool(half_period)
        self.surf_nt = int(surf_nt)
        self.surf_np = int(surf_np)
        self.src_nt = int(src_nt)
        self.src_np = int(src_np)
        self.trg_nt = int(trg_nt)
        self.trg_np = int(trg_np)
        self.surface_coord = surface_coord
        self._setup = True
        self._grad_setup = None
        self._b_setup = None

    def _select_quad_sizes(self, digits: int):
        surf_nt_full = int(self.surface_coord.shape[1])
        surf_np_full = int(self.surface_coord.shape[2])
        src_nt_full = int(self.nfp_eff * self.src_nt)

        dX = grad2d(self.surface_coord, surf_nt_full, surf_np_full)
        dX_np = jnp.asarray(dX)

        xt = dX_np[0]
        xp = dX_np[1]
        yt = dX_np[2]
        yp = dX_np[3]
        zt = dX_np[4]
        zp = dX_np[5]
        m00 = (xt * xt + yt * yt + zt * zt) / (surf_nt_full * surf_nt_full)
        m11 = (xp * xp + yp * yp + zp * zp) / (surf_np_full * surf_np_full)
        ratio = jnp.sqrt(m00 / m11)
        amin = float(jnp.min(ratio))
        amax = float(jnp.max(ratio))

        optim_aspect_ratio = math.sqrt(amin * amax) * surf_nt_full / surf_np_full
        cond = math.sqrt(amax / amin)
        pdim = digits * cond * 1.6

        quad_np = self.trg_np * math.ceil(
            max(surf_np_full, self.src_np, 2 * pdim + 1) / self.trg_np
        )
        quad_nt = self.nfp_eff * self.trg_nt * math.ceil(
            max(max(surf_nt_full, src_nt_full), optim_aspect_ratio * quad_np)
            / (self.nfp_eff * self.trg_nt)
        )

        trg_nt_self = surf_nt_full // self.nfp_eff
        trg_np_self = surf_np_full

        for _ in range(3):
            quad_nt_aligned = math.ceil(quad_nt / surf_nt_full) * surf_nt_full
            quad_np_aligned = math.ceil(quad_np / surf_np_full) * surf_np_full

            X_quad = resample(
                self.surface_coord,
                surf_nt_full,
                surf_np_full,
                quad_nt_aligned,
                quad_np_aligned,
            )
            dX_quad = grad2d(X_quad, quad_nt_aligned, quad_np_aligned)
            ones = jnp.ones((quad_nt_aligned, quad_np_aligned), dtype=X_quad.dtype)
            U = laplace_dx_u_eval_singular(
                X_quad,
                dX_quad,
                ones,
                trg_nt_self,
                trg_np_self,
                self.nfp_eff,
                digits=digits,
                chunk_size=1024,
            )
            err = float(jnp.max(jnp.abs(jnp.asarray(U).reshape(-1) - 0.5)))
            if err <= 0:
                break
            scal = max(1.0, (digits + 1) / (math.log(err) / math.log(0.1)))
            quad_nt = int(scal * quad_nt_aligned)
            quad_np = int(scal * quad_np_aligned)
            if err < 10 ** (-digits) or scal < 1.5:
                break

        quad_np = self.trg_np * round((quad_nt / optim_aspect_ratio) / self.trg_np)
        quad_nt = self.nfp_eff * self.trg_nt * round(
            (optim_aspect_ratio * quad_np) / (self.nfp_eff * self.trg_nt)
        )

        return int(quad_nt), int(quad_np)

    def _build_quad_setup(self, quad_nt: int, quad_np: int):
        surf_nt_full = int(self.surface_coord.shape[1])
        surf_np_full = int(self.surface_coord.shape[2])
        quad_coord = resample(
            self.surface_coord,
            surf_nt_full,
            surf_np_full,
            quad_nt,
            quad_np,
        )
        dX = grad2d(quad_coord, quad_nt, quad_np)
        normal, _, orient = surf_normal_area_elem(dX, quad_coord, return_orientation=True)
        orient = float(orient)
        return QuadSetup(
            quad_nt=quad_nt,
            quad_np=quad_np,
            quad_coord=quad_coord,
            dX=dX,
            normal=normal,
            orient=orient,
        )

    def _get_patch_idx(self, setup: QuadSetup, digits: int):
        cond = float(_surface_cond(setup.dX, setup.quad_nt, setup.quad_np))
        patch_dim0 = select_patch_dim(digits, cond)
        patch_idx = setup.patch_idx_cache.get(patch_dim0)
        if patch_idx is None:
            skip_nt = setup.quad_nt // (self.nfp_eff * self.trg_nt)
            skip_np = setup.quad_np // self.trg_np
            t_idx = jnp.arange(self.trg_nt) * skip_nt
            p_idx = jnp.arange(self.trg_np) * skip_np
            tt, pp = jnp.meshgrid(t_idx, p_idx, indexing="ij")
            patch_idx = _build_patch_indices(
                tt.reshape(-1),
                pp.reshape(-1),
                setup.quad_nt,
                setup.quad_np,
                patch_dim0,
            )
            setup.patch_idx_cache[patch_dim0] = patch_idx
        return patch_dim0, patch_idx

    def _ensure_grad_setup(self, quad_nt: int | None, quad_np: int | None, digits: int):
        if not self._setup:
            raise RuntimeError("VirtualCasingJAX.setup must be called before compute_external_gradB")

        if quad_nt is None or quad_np is None:
            quad_nt_sel, quad_np_sel = self._select_quad_sizes(digits)
            quad_nt = quad_nt_sel
            quad_np = quad_np_sel

        if (
            self._grad_setup is not None
            and self._grad_setup.quad_nt == quad_nt
            and self._grad_setup.quad_np == quad_np
        ):
            return

        self._grad_setup = self._build_quad_setup(quad_nt, quad_np)

    def _ensure_b_setup(self, quad_nt: int | None, quad_np: int | None, digits: int):
        if not self._setup:
            raise RuntimeError("VirtualCasingJAX.setup must be called before compute_external_B")

        if quad_nt is None or quad_np is None:
            quad_nt_sel, quad_np_sel = self._select_quad_sizes(digits)
            quad_nt = quad_nt_sel
            quad_np = quad_np_sel

        if (
            self._b_setup is not None
            and self._b_setup.quad_nt == quad_nt
            and self._b_setup.quad_np == quad_np
        ):
            return

        self._b_setup = self._build_quad_setup(quad_nt, quad_np)

    def _compute_gradB_signed(
        self,
        B0,
        *,
        sign: float,
        quad_nt: int | None = None,
        quad_np: int | None = None,
        digits: int | None = None,
        hedgehog_order: int = 8,
        chunk_size: int | str | None = "auto",
        target_chunk_size: int | str | None = "auto",
        pou_dtype=None,
        patch_dtype=None,
        remat: bool | None = None,
        patch_dim0: int | None = None,
        patch_idx=None,
    ):
        if not self._setup:
            raise RuntimeError("VirtualCasingJAX.setup must be called before compute_gradB")

        digits = self.digits if digits is None else int(digits)
        self._ensure_grad_setup(quad_nt, quad_np, digits)
        assert self._grad_setup is not None

        B0 = jnp.asarray(B0).reshape((3, self.src_nt, self.src_np))

        if remat is None:
            remat = True

        pou_dtype = self._resolve_pou_dtype(pou_dtype, B0.dtype)
        patch_dtype = self._resolve_patch_dtype(patch_dtype, B0.dtype)
        nsrc = self._grad_setup.quad_nt * self._grad_setup.quad_np
        ntrg = self.trg_nt * self.trg_np
        chunk_size, target_chunk_size = self._resolve_chunk_sizes(
            "gradb", chunk_size, target_chunk_size, nsrc=nsrc, ntrg=ntrg
        )

        dtheta = 0.0
        if self.half_period:
            dtheta = math.pi * (
                1.0 / (self.nfp * self.trg_nt * 2) - 1.0 / (self.nfp * self.src_nt * 2)
            )

        B0_complete = complete_vec_field(
            B0,
            False,
            self.half_period,
            self.nfp,
            self.src_nt,
            self.src_np,
            dtheta,
        )
        B_quad = resample(
            B0_complete,
            self.nfp_eff * self.src_nt,
            self.src_np,
            self._grad_setup.quad_nt,
            self._grad_setup.quad_np,
        )

        J = cross_prod(self._grad_setup.normal, B_quad)
        BdotN = dot_prod(B_quad, self._grad_setup.normal)

        if patch_dim0 is None or patch_idx is None:
            patch_dim0, patch_idx = self._get_patch_idx(self._grad_setup, digits)

        gradG_J = laplace_fxd2_u_eval_vec_singular(
            self._grad_setup.quad_coord,
            self._grad_setup.dX,
            J,
            self.trg_nt,
            self.trg_np,
            self.nfp_eff,
            digits=digits,
            patch_dim0=patch_dim0,
            hedgehog_order=hedgehog_order,
            chunk_size=chunk_size,
            target_chunk_size=target_chunk_size,
            patch_idx=patch_idx,
            orient=self._grad_setup.orient,
            pou_dtype=pou_dtype,
            patch_dtype=patch_dtype,
            remat=remat,
        )
        gradG_J = jnp.asarray(gradG_J).reshape((3, 3, 3, self.trg_nt, self.trg_np))

        gradgradG_BdotN = laplace_fxd2_u_eval_singular(
            self._grad_setup.quad_coord,
            self._grad_setup.dX,
            BdotN,
            self.trg_nt,
            self.trg_np,
            self.nfp_eff,
            digits=digits,
            patch_dim0=patch_dim0,
            hedgehog_order=hedgehog_order,
            chunk_size=chunk_size,
            target_chunk_size=target_chunk_size,
            patch_idx=patch_idx,
            orient=self._grad_setup.orient,
            pou_dtype=pou_dtype,
            patch_dtype=patch_dtype,
            remat=remat,
        )
        gradgradG_BdotN = jnp.asarray(gradgradG_BdotN).reshape(
            (3, 3, self.trg_nt, self.trg_np)
        )

        gradBvc = jnp.zeros((3, 3, self.trg_nt, self.trg_np), dtype=gradG_J.dtype)
        for k in range(3):
            k1 = (k + 1) % 3
            k2 = (k + 2) % 3
            gradBvc = gradBvc.at[k].set(gradG_J[k1, k2] - gradG_J[k2, k1])

        gradBvc = gradBvc + gradgradG_BdotN
        return gradBvc * sign

    def compute_external_gradB(
        self,
        B0,
        *,
        quad_nt: int | None = None,
        quad_np: int | None = None,
        digits: int | None = None,
        hedgehog_order: int = 8,
        chunk_size: int | str | None = "auto",
        target_chunk_size: int | str | None = "auto",
        pou_dtype=None,
        patch_dtype=None,
        remat: bool | None = None,
        patch_dim0: int | None = None,
        patch_idx=None,
    ):
        """Compute GradBext from total B on the source grid."""
        return self._compute_gradB_signed(
            B0,
            sign=1.0,
            quad_nt=quad_nt,
            quad_np=quad_np,
            digits=digits,
            hedgehog_order=hedgehog_order,
            chunk_size=chunk_size,
            target_chunk_size=target_chunk_size,
            pou_dtype=pou_dtype,
            patch_dtype=patch_dtype,
            remat=remat,
            patch_dim0=patch_dim0,
            patch_idx=patch_idx,
        )

    def compute_internal_gradB(
        self,
        B0,
        *,
        quad_nt: int | None = None,
        quad_np: int | None = None,
        digits: int | None = None,
        hedgehog_order: int = 8,
        chunk_size: int | str | None = "auto",
        target_chunk_size: int | str | None = "auto",
        pou_dtype=None,
        patch_dtype=None,
        remat: bool | None = None,
        patch_dim0: int | None = None,
        patch_idx=None,
    ):
        """Compute GradBint from total B on the source grid."""
        return self._compute_gradB_signed(
            B0,
            sign=-1.0,
            quad_nt=quad_nt,
            quad_np=quad_np,
            digits=digits,
            hedgehog_order=hedgehog_order,
            chunk_size=chunk_size,
            target_chunk_size=target_chunk_size,
            pou_dtype=pou_dtype,
            patch_dtype=patch_dtype,
            remat=remat,
            patch_dim0=patch_dim0,
            patch_idx=patch_idx,
        )

    def compute_external_gradB_jit(self, B0, **kwargs):
        """JIT-compiled version of compute_external_gradB."""
        if "X_trg" in kwargs and kwargs["X_trg"] is not None:
            raise ValueError("compute_external_gradB_jit does not support X_trg")
        digits = self.digits if kwargs.get("digits") is None else int(kwargs["digits"])
        quad_nt = kwargs.get("quad_nt")
        quad_np = kwargs.get("quad_np")
        self._ensure_grad_setup(quad_nt, quad_np, digits)
        patch_dim0, patch_idx = self._get_patch_idx(self._grad_setup, digits)

        chunk_size = kwargs.get("chunk_size", "auto")
        target_chunk_size = kwargs.get("target_chunk_size", "auto")
        remat = kwargs.get("remat")
        pou_dtype = kwargs.get("pou_dtype")
        patch_dtype = kwargs.get("patch_dtype")
        nsrc = self._grad_setup.quad_nt * self._grad_setup.quad_np
        ntrg = self.trg_nt * self.trg_np
        chunk_size, target_chunk_size = self._resolve_chunk_sizes(
            "gradb", chunk_size, target_chunk_size, nsrc=nsrc, ntrg=ntrg
        )
        pou_dtype = self._resolve_pou_dtype(pou_dtype, jnp.asarray(B0).dtype)
        patch_dtype = self._resolve_patch_dtype(patch_dtype, jnp.asarray(B0).dtype)

        key = (
            "gradB",
            digits,
            quad_nt,
            quad_np,
            kwargs.get("hedgehog_order", 8),
            chunk_size,
            target_chunk_size,
            remat,
            pou_dtype,
            patch_dtype,
        )
        fn = self._jit_cache.get(key)
        if fn is None:
            call_kwargs = dict(kwargs)
            call_kwargs["chunk_size"] = chunk_size
            call_kwargs["target_chunk_size"] = target_chunk_size
            call_kwargs["remat"] = remat
            call_kwargs["pou_dtype"] = pou_dtype
            call_kwargs["patch_dtype"] = patch_dtype
            fn = jax.jit(
                lambda B: self.compute_external_gradB(
                    B,
                    patch_dim0=patch_dim0,
                    patch_idx=patch_idx,
                    **call_kwargs,
                )
            )
            self._jit_cache[key] = fn
        return fn(B0)

    def _compute_B_signed(
        self,
        B0,
        *,
        sign: float,
        X_trg=None,
        quad_nt: int | None = None,
        quad_np: int | None = None,
        digits: int | None = None,
        chunk_size: int | str | None = "auto",
        target_chunk_size: int | str | None = "auto",
        pou_dtype=None,
        patch_dtype=None,
        remat: bool | None = None,
        patch_dim0: int | None = None,
        patch_idx=None,
    ):
        if not self._setup:
            raise RuntimeError("VirtualCasingJAX.setup must be called before compute_B")

        digits = self.digits if digits is None else int(digits)
        self._ensure_b_setup(quad_nt, quad_np, digits)
        assert self._b_setup is not None

        B0 = jnp.asarray(B0).reshape((3, self.src_nt, self.src_np))

        if remat is None:
            remat = False

        pou_dtype = self._resolve_pou_dtype(pou_dtype, B0.dtype)
        patch_dtype = self._resolve_patch_dtype(patch_dtype, B0.dtype)
        nsrc = self._b_setup.quad_nt * self._b_setup.quad_np
        if X_trg is None:
            ntrg = self.trg_nt * self.trg_np
        else:
            X_trg_arr = jnp.asarray(X_trg)
            if X_trg_arr.ndim == 3:
                ntrg = X_trg_arr.shape[1] * X_trg_arr.shape[2]
            elif X_trg_arr.ndim == 2:
                ntrg = X_trg_arr.shape[1]
            else:
                raise ValueError("X_trg must have shape (3, nt, np) or (3, ntrg)")
        chunk_size, target_chunk_size = self._resolve_chunk_sizes(
            "b", chunk_size, target_chunk_size, nsrc=nsrc, ntrg=ntrg
        )

        dtheta = 0.0
        if self.half_period:
            dtheta = math.pi * (
                1.0 / (self.nfp * self.trg_nt * 2) - 1.0 / (self.nfp * self.src_nt * 2)
            )

        B0_complete = complete_vec_field(
            B0,
            False,
            self.half_period,
            self.nfp,
            self.src_nt,
            self.src_np,
            dtheta,
        )
        B_quad = resample(
            B0_complete,
            self.nfp_eff * self.src_nt,
            self.src_np,
            self._b_setup.quad_nt,
            self._b_setup.quad_np,
        )

        J = cross_prod(self._b_setup.normal, B_quad)
        BdotN = dot_prod(B_quad, self._b_setup.normal)

        if patch_dim0 is None or patch_idx is None:
            patch_dim0, patch_idx = self._get_patch_idx(self._b_setup, digits)

        gradG_J = laplace_fxd_u_eval_vec_singular(
            self._b_setup.quad_coord,
            self._b_setup.dX,
            J,
            self.trg_nt,
            self.trg_np,
            self.nfp_eff,
            X_trg=X_trg,
            digits=digits,
            patch_dim0=patch_dim0,
            chunk_size=chunk_size,
            target_chunk_size=target_chunk_size,
            patch_idx=patch_idx,
            orient=self._b_setup.orient,
            pou_dtype=pou_dtype,
            patch_dtype=patch_dtype,
            remat=remat,
        )
        gradG_J = jnp.asarray(gradG_J).reshape((3, 3, self.trg_nt, self.trg_np))

        gradG_BdotN = laplace_fxd_u_eval_singular(
            self._b_setup.quad_coord,
            self._b_setup.dX,
            BdotN,
            self.trg_nt,
            self.trg_np,
            self.nfp_eff,
            X_trg=X_trg,
            digits=digits,
            patch_dim0=patch_dim0,
            chunk_size=chunk_size,
            target_chunk_size=target_chunk_size,
            patch_idx=patch_idx,
            orient=self._b_setup.orient,
            pou_dtype=pou_dtype,
            patch_dtype=patch_dtype,
            remat=remat,
        )
        gradG_BdotN = jnp.asarray(gradG_BdotN).reshape((3, self.trg_nt, self.trg_np))

        B_on_trg = resample(
            B0_complete,
            self.nfp_eff * self.src_nt,
            self.src_np,
            self.nfp_eff * self.trg_nt,
            self.trg_np,
        )
        B_on = B_on_trg[:, : self.trg_nt, :]

        Bvc = jnp.zeros((3, self.trg_nt, self.trg_np), dtype=gradG_J.dtype)
        for k in range(3):
            k1 = (k + 1) % 3
            k2 = (k + 2) % 3
            Bvc = Bvc.at[k].set(gradG_J[k1, k2] - gradG_J[k2, k1])

        Bvc = sign * (Bvc + gradG_BdotN) + 0.5 * B_on
        return Bvc

    def compute_external_B(
        self,
        B0,
        *,
        X_trg=None,
        quad_nt: int | None = None,
        quad_np: int | None = None,
        digits: int | None = None,
        chunk_size: int | str | None = "auto",
        target_chunk_size: int | str | None = "auto",
        pou_dtype=None,
        patch_dtype=None,
        remat: bool | None = None,
        patch_dim0: int | None = None,
        patch_idx=None,
    ):
        """Compute Bext from total B on the source grid."""
        return self._compute_B_signed(
            B0,
            sign=1.0,
            X_trg=X_trg,
            quad_nt=quad_nt,
            quad_np=quad_np,
            digits=digits,
            chunk_size=chunk_size,
            target_chunk_size=target_chunk_size,
            pou_dtype=pou_dtype,
            patch_dtype=patch_dtype,
            remat=remat,
            patch_dim0=patch_dim0,
            patch_idx=patch_idx,
        )

    def compute_internal_B(
        self,
        B0,
        *,
        X_trg=None,
        quad_nt: int | None = None,
        quad_np: int | None = None,
        digits: int | None = None,
        chunk_size: int | str | None = "auto",
        target_chunk_size: int | str | None = "auto",
        pou_dtype=None,
        patch_dtype=None,
        remat: bool | None = None,
        patch_dim0: int | None = None,
        patch_idx=None,
    ):
        """Compute Bint from total B on the source grid."""
        return self._compute_B_signed(
            B0,
            sign=-1.0,
            X_trg=X_trg,
            quad_nt=quad_nt,
            quad_np=quad_np,
            digits=digits,
            chunk_size=chunk_size,
            target_chunk_size=target_chunk_size,
            pou_dtype=pou_dtype,
            patch_dtype=patch_dtype,
            remat=remat,
            patch_dim0=patch_dim0,
            patch_idx=patch_idx,
        )

    def compute_external_B_jit(self, B0, **kwargs):
        """JIT-compiled version of compute_external_B."""
        if "X_trg" in kwargs and kwargs["X_trg"] is not None:
            raise ValueError("compute_external_B_jit does not support X_trg; jit externally if needed")
        digits = self.digits if kwargs.get("digits") is None else int(kwargs["digits"])
        quad_nt = kwargs.get("quad_nt")
        quad_np = kwargs.get("quad_np")
        self._ensure_b_setup(quad_nt, quad_np, digits)
        patch_dim0, patch_idx = self._get_patch_idx(self._b_setup, digits)

        chunk_size = kwargs.get("chunk_size", "auto")
        target_chunk_size = kwargs.get("target_chunk_size", "auto")
        remat = kwargs.get("remat")
        pou_dtype = kwargs.get("pou_dtype")
        patch_dtype = kwargs.get("patch_dtype")
        nsrc = self._b_setup.quad_nt * self._b_setup.quad_np
        ntrg = self.trg_nt * self.trg_np
        chunk_size, target_chunk_size = self._resolve_chunk_sizes(
            "b", chunk_size, target_chunk_size, nsrc=nsrc, ntrg=ntrg
        )
        pou_dtype = self._resolve_pou_dtype(pou_dtype, jnp.asarray(B0).dtype)
        patch_dtype = self._resolve_patch_dtype(patch_dtype, jnp.asarray(B0).dtype)

        key = (
            "B",
            digits,
            quad_nt,
            quad_np,
            chunk_size,
            target_chunk_size,
            remat,
            pou_dtype,
            patch_dtype,
        )
        fn = self._jit_cache.get(key)
        if fn is None:
            call_kwargs = dict(kwargs)
            call_kwargs["chunk_size"] = chunk_size
            call_kwargs["target_chunk_size"] = target_chunk_size
            call_kwargs["remat"] = remat
            call_kwargs["pou_dtype"] = pou_dtype
            call_kwargs["patch_dtype"] = patch_dtype
            fn = jax.jit(
                lambda B: self.compute_external_B(
                    B,
                    patch_dim0=patch_dim0,
                    patch_idx=patch_idx,
                    **call_kwargs,
                )
            )
            self._jit_cache[key] = fn
        return fn(B0)

    def compute_internal_B_jit(self, B0, **kwargs):
        """JIT-compiled version of compute_internal_B."""
        if "X_trg" in kwargs and kwargs["X_trg"] is not None:
            raise ValueError("compute_internal_B_jit does not support X_trg; jit externally if needed")
        digits = self.digits if kwargs.get("digits") is None else int(kwargs["digits"])
        quad_nt = kwargs.get("quad_nt")
        quad_np = kwargs.get("quad_np")
        self._ensure_b_setup(quad_nt, quad_np, digits)
        patch_dim0, patch_idx = self._get_patch_idx(self._b_setup, digits)

        chunk_size = kwargs.get("chunk_size", "auto")
        target_chunk_size = kwargs.get("target_chunk_size", "auto")
        remat = kwargs.get("remat")
        pou_dtype = kwargs.get("pou_dtype")
        patch_dtype = kwargs.get("patch_dtype")
        nsrc = self._b_setup.quad_nt * self._b_setup.quad_np
        ntrg = self.trg_nt * self.trg_np
        chunk_size, target_chunk_size = self._resolve_chunk_sizes(
            "b", chunk_size, target_chunk_size, nsrc=nsrc, ntrg=ntrg
        )
        pou_dtype = self._resolve_pou_dtype(pou_dtype, jnp.asarray(B0).dtype)
        patch_dtype = self._resolve_patch_dtype(patch_dtype, jnp.asarray(B0).dtype)

        key = (
            "Bint",
            digits,
            quad_nt,
            quad_np,
            chunk_size,
            target_chunk_size,
            remat,
            pou_dtype,
            patch_dtype,
        )
        fn = self._jit_cache.get(key)
        if fn is None:
            call_kwargs = dict(kwargs)
            call_kwargs["chunk_size"] = chunk_size
            call_kwargs["target_chunk_size"] = target_chunk_size
            call_kwargs["remat"] = remat
            call_kwargs["pou_dtype"] = pou_dtype
            call_kwargs["patch_dtype"] = patch_dtype
            fn = jax.jit(
                lambda B: self.compute_internal_B(
                    B,
                    patch_dim0=patch_dim0,
                    patch_idx=patch_idx,
                    **call_kwargs,
                )
            )
            self._jit_cache[key] = fn
        return fn(B0)

    def compute_external_B_batch(self, B0_batch, *, X_trg=None, **kwargs):
        """Vectorized compute_external_B over a batch dimension."""
        if X_trg is None:
            return jax.vmap(lambda b: self.compute_external_B(b, **kwargs), in_axes=0)(B0_batch)
        return jax.vmap(lambda b, xt: self.compute_external_B(b, X_trg=xt, **kwargs), in_axes=(0, 0))(B0_batch, X_trg)

    def compute_internal_B_batch(self, B0_batch, *, X_trg=None, **kwargs):
        """Vectorized compute_internal_B over a batch dimension."""
        if X_trg is None:
            return jax.vmap(lambda b: self.compute_internal_B(b, **kwargs), in_axes=0)(B0_batch)
        return jax.vmap(lambda b, xt: self.compute_internal_B(b, X_trg=xt, **kwargs), in_axes=(0, 0))(B0_batch, X_trg)

    def compute_external_gradB_batch(self, B0_batch, **kwargs):
        """Vectorized compute_external_gradB over a batch dimension."""
        return jax.vmap(lambda b: self.compute_external_gradB(b, **kwargs), in_axes=0)(B0_batch)

    def compute_internal_gradB_batch(self, B0_batch, **kwargs):
        """Vectorized compute_internal_gradB over a batch dimension."""
        return jax.vmap(lambda b: self.compute_internal_gradB(b, **kwargs), in_axes=0)(B0_batch)

    def compute_internal_gradB_jit(self, B0, **kwargs):
        """JIT-compiled version of compute_internal_gradB."""
        if "X_trg" in kwargs and kwargs["X_trg"] is not None:
            raise ValueError("compute_internal_gradB_jit does not support X_trg")
        digits = self.digits if kwargs.get("digits") is None else int(kwargs["digits"])
        quad_nt = kwargs.get("quad_nt")
        quad_np = kwargs.get("quad_np")
        self._ensure_grad_setup(quad_nt, quad_np, digits)
        patch_dim0, patch_idx = self._get_patch_idx(self._grad_setup, digits)

        chunk_size = kwargs.get("chunk_size", "auto")
        target_chunk_size = kwargs.get("target_chunk_size", "auto")
        remat = kwargs.get("remat")
        pou_dtype = kwargs.get("pou_dtype")
        patch_dtype = kwargs.get("patch_dtype")
        nsrc = self._grad_setup.quad_nt * self._grad_setup.quad_np
        ntrg = self.trg_nt * self.trg_np
        chunk_size, target_chunk_size = self._resolve_chunk_sizes(
            "gradb", chunk_size, target_chunk_size, nsrc=nsrc, ntrg=ntrg
        )
        pou_dtype = self._resolve_pou_dtype(pou_dtype, jnp.asarray(B0).dtype)
        patch_dtype = self._resolve_patch_dtype(patch_dtype, jnp.asarray(B0).dtype)

        key = (
            "gradBint",
            digits,
            quad_nt,
            quad_np,
            kwargs.get("hedgehog_order", 8),
            chunk_size,
            target_chunk_size,
            remat,
            pou_dtype,
            patch_dtype,
        )
        fn = self._jit_cache.get(key)
        if fn is None:
            call_kwargs = dict(kwargs)
            call_kwargs["chunk_size"] = chunk_size
            call_kwargs["target_chunk_size"] = target_chunk_size
            call_kwargs["remat"] = remat
            call_kwargs["pou_dtype"] = pou_dtype
            call_kwargs["patch_dtype"] = patch_dtype
            fn = jax.jit(
                lambda B: self.compute_internal_gradB(
                    B,
                    patch_dim0=patch_dim0,
                    patch_idx=patch_idx,
                    **call_kwargs,
                )
            )
            self._jit_cache[key] = fn
        return fn(B0)

    def compute_external_B_autodiff(
        self,
        B0,
        *,
        X_trg,
        quad_nt: int | None = None,
        quad_np: int | None = None,
        digits: int | None = None,
        chunk_size: int | str | None = "auto",
        target_chunk_size: int | str | None = "auto",
        pou_dtype=None,
        patch_dtype=None,
        remat: bool | None = None,
        hedgehog_order: int = 8,
    ):
        """Compute Bext with a custom JVP that matches ComputeGradB on-surface."""
        if X_trg is None:
            raise ValueError("X_trg must be provided for autodiff-enabled evaluation")

        B0 = jnp.asarray(B0)
        X_trg = jnp.asarray(X_trg)
        digits = self.digits if digits is None else int(digits)

        @jax.custom_jvp
        def _eval(xtrg):
            return self.compute_external_B(
                B0,
                X_trg=xtrg,
                quad_nt=quad_nt,
                quad_np=quad_np,
                digits=digits,
                chunk_size=chunk_size,
                target_chunk_size=target_chunk_size,
                pou_dtype=pou_dtype,
                remat=remat,
            )

        @_eval.defjvp
        def _eval_jvp(primals, tangents):
            (xtrg,) = primals
            (dxtrg,) = tangents
            b = self.compute_external_B(
                B0,
                X_trg=xtrg,
                quad_nt=quad_nt,
                quad_np=quad_np,
                digits=digits,
                chunk_size=chunk_size,
                target_chunk_size=target_chunk_size,
                pou_dtype=pou_dtype,
                patch_dtype=patch_dtype,
                remat=remat,
            )
            gradb = self.compute_external_gradB(
                B0,
                quad_nt=quad_nt,
                quad_np=quad_np,
                digits=digits,
                hedgehog_order=hedgehog_order,
                chunk_size=chunk_size,
                target_chunk_size=target_chunk_size,
                pou_dtype=pou_dtype,
                patch_dtype=patch_dtype,
                remat=remat,
            )
            db = jnp.einsum("k i t p, i t p -> k t p", gradb, dxtrg)
            return b, db

        return _eval(X_trg)

    def _offsurface_densities(self, B0):
        if not self._setup:
            raise RuntimeError("VirtualCasingJAX.setup must be called before off-surface evaluation")

        B0 = jnp.asarray(B0).reshape((3, self.src_nt, self.src_np))

        surf_nt_full = int(self.surface_coord.shape[1])
        surf_np_full = int(self.surface_coord.shape[2])
        patch_dim = 13  # 2*6+1 to match BIEST off-surface minimum
        base_nt = max(self.nfp_eff * self.src_nt, surf_nt_full, patch_dim)
        base_np = max(self.src_np, surf_np_full, patch_dim)

        X_src = resample(
            self.surface_coord,
            surf_nt_full,
            surf_np_full,
            base_nt,
            base_np,
        )
        dX = grad2d(X_src, base_nt, base_np)
        normal, _ = surf_normal_area_elem(dX, X_src)

        dtheta = 0.0
        if self.half_period:
            dtheta = math.pi * (
                1.0 / (self.nfp * self.trg_nt * 2) - 1.0 / (self.nfp * self.src_nt * 2)
            )

        B0_complete = complete_vec_field(
            B0,
            False,
            self.half_period,
            self.nfp,
            self.src_nt,
            self.src_np,
            dtheta,
        )
        B_quad = resample(
            B0_complete,
            self.nfp_eff * self.src_nt,
            self.src_np,
            base_nt,
            base_np,
        )

        J = cross_prod(normal, B_quad)
        BdotN = dot_prod(B_quad, normal)
        return X_src, BdotN, J

    def compute_external_B_offsurf(
        self,
        B0,
        *,
        X_trg,
        digits: int | None = None,
        max_Nt: int = -1,
        max_Np: int = -1,
        chunk_size: int | str | None = "auto",
        target_chunk_size: int | str | None = "auto",
    ):
        """Compute Bext at off-surface targets using adaptive quadrature."""
        digits = self.digits if digits is None else int(digits)
        X_src, BdotN, J = self._offsurface_densities(B0)
        X_trg = jnp.asarray(X_trg)
        nsrc = X_src.shape[1] * X_src.shape[2]
        ntrg = X_trg.shape[1] * X_trg.shape[2] if X_trg.ndim == 3 else X_trg.shape[1]
        chunk_size, target_chunk_size = self._resolve_chunk_sizes(
            "boff", chunk_size, target_chunk_size, nsrc=nsrc, ntrg=ntrg
        )
        out = computeB_offsurface_adaptive(
            X_src,
            BdotN,
            J,
            X_trg,
            digits=digits,
            max_Nt=max_Nt,
            max_Np=max_Np,
            ext=True,
            chunk_size=chunk_size,
            target_chunk_size=target_chunk_size,
        )
        if X_trg.ndim == 3:
            return jnp.asarray(out).reshape((3, X_trg.shape[1], X_trg.shape[2]))
        return out

    def compute_internal_B_offsurf(
        self,
        B0,
        *,
        X_trg,
        digits: int | None = None,
        max_Nt: int = -1,
        max_Np: int = -1,
        chunk_size: int | str | None = "auto",
        target_chunk_size: int | str | None = "auto",
    ):
        """Compute Bint at off-surface targets using adaptive quadrature."""
        digits = self.digits if digits is None else int(digits)
        X_src, BdotN, J = self._offsurface_densities(B0)
        X_trg = jnp.asarray(X_trg)
        nsrc = X_src.shape[1] * X_src.shape[2]
        ntrg = X_trg.shape[1] * X_trg.shape[2] if X_trg.ndim == 3 else X_trg.shape[1]
        chunk_size, target_chunk_size = self._resolve_chunk_sizes(
            "boff", chunk_size, target_chunk_size, nsrc=nsrc, ntrg=ntrg
        )
        out = computeB_offsurface_adaptive(
            X_src,
            BdotN,
            J,
            X_trg,
            digits=digits,
            max_Nt=max_Nt,
            max_Np=max_Np,
            ext=False,
            chunk_size=chunk_size,
            target_chunk_size=target_chunk_size,
        )
        if X_trg.ndim == 3:
            return jnp.asarray(out).reshape((3, X_trg.shape[1], X_trg.shape[2]))
        return out

    def compute_external_B_offsurf_schedule(
        self,
        B0,
        *,
        X_trg,
        levels: tuple[tuple[int, int], ...],
        digits: int | None = None,
        chunk_size: int | str | None = "auto",
        target_chunk_size: int | str | None = "auto",
    ):
        """Compute Bext off-surface using a fixed adaptive refinement schedule."""
        digits = self.digits if digits is None else int(digits)
        X_src, BdotN, J = self._offsurface_densities(B0)
        X_trg = jnp.asarray(X_trg)
        nsrc = X_src.shape[1] * X_src.shape[2]
        ntrg = X_trg.shape[1] * X_trg.shape[2] if X_trg.ndim == 3 else X_trg.shape[1]
        chunk_size, target_chunk_size = self._resolve_chunk_sizes(
            "boff", chunk_size, target_chunk_size, nsrc=nsrc, ntrg=ntrg
        )
        out = computeB_offsurface_adaptive_schedule(
            X_src,
            BdotN,
            J,
            X_trg,
            levels=levels,
            digits=digits,
            ext=True,
            chunk_size=chunk_size,
            target_chunk_size=target_chunk_size,
        )
        if X_trg.ndim == 3:
            return jnp.asarray(out).reshape((3, X_trg.shape[1], X_trg.shape[2]))
        return out

    def compute_internal_B_offsurf_schedule(
        self,
        B0,
        *,
        X_trg,
        levels: tuple[tuple[int, int], ...],
        digits: int | None = None,
        chunk_size: int | str | None = "auto",
        target_chunk_size: int | str | None = "auto",
    ):
        """Compute Bint off-surface using a fixed adaptive refinement schedule."""
        digits = self.digits if digits is None else int(digits)
        X_src, BdotN, J = self._offsurface_densities(B0)
        X_trg = jnp.asarray(X_trg)
        nsrc = X_src.shape[1] * X_src.shape[2]
        ntrg = X_trg.shape[1] * X_trg.shape[2] if X_trg.ndim == 3 else X_trg.shape[1]
        chunk_size, target_chunk_size = self._resolve_chunk_sizes(
            "boff", chunk_size, target_chunk_size, nsrc=nsrc, ntrg=ntrg
        )
        out = computeB_offsurface_adaptive_schedule(
            X_src,
            BdotN,
            J,
            X_trg,
            levels=levels,
            digits=digits,
            ext=False,
            chunk_size=chunk_size,
            target_chunk_size=target_chunk_size,
        )
        if X_trg.ndim == 3:
            return jnp.asarray(out).reshape((3, X_trg.shape[1], X_trg.shape[2]))
        return out

    def compute_external_gradB_offsurf(
        self,
        B0,
        *,
        X_trg,
        digits: int | None = None,
        max_Nt: int = -1,
        max_Np: int = -1,
        adaptive: bool = False,
        chunk_size: int | str | None = "auto",
        target_chunk_size: int | str | None = "auto",
    ):
        """Compute GradBext at off-surface targets using direct quadrature.

        The off-surface GradB path mirrors the reference implementation and
        currently uses the base resampled grid (no adaptive refinement).
        """
        digits = self.digits if digits is None else int(digits)
        X_trg = jnp.asarray(X_trg)
        X_trg_flat = X_trg.reshape((3, -1)) if X_trg.ndim == 3 else X_trg
        ntrg = X_trg_flat.shape[1]

        X_src, BdotN, J = self._offsurface_densities(B0)
        nsrc = X_src.shape[1] * X_src.shape[2]
        chunk_size, target_chunk_size = self._resolve_chunk_sizes(
            "gradb_off", chunk_size, target_chunk_size, nsrc=nsrc, ntrg=ntrg
        )
        if adaptive:
            X_src, BdotN, J, area_elem = _offsurface_adapt_grid(
                X_src,
                BdotN,
                J,
                X_trg_flat,
                digits=digits,
                max_Nt=max_Nt,
                max_Np=max_Np,
                chunk_size=chunk_size,
                target_chunk_size=target_chunk_size,
            )
        else:
            dX = grad2d(X_src, X_src.shape[1], X_src.shape[2])
            _, area_elem = surf_normal_area_elem(dX, X_src)

        gradG_J = laplace_fxd2_u_eval_vec(
            X_src,
            X_trg_flat,
            J,
            area_elem,
            chunk_size=chunk_size,
            target_chunk_size=target_chunk_size,
        )
        gradG_J = jnp.asarray(gradG_J).reshape((3, 3, 3, ntrg))

        gradgradG_BdotN = laplace_fxd2_u_eval(
            X_src,
            X_trg_flat,
            BdotN,
            area_elem,
            chunk_size=chunk_size,
            target_chunk_size=target_chunk_size,
        )
        gradgradG_BdotN = jnp.asarray(gradgradG_BdotN).reshape((3, 3, ntrg))

        gradB = jnp.zeros((3, 3, ntrg), dtype=gradG_J.dtype)
        for k in range(3):
            k1 = (k + 1) % 3
            k2 = (k + 2) % 3
            gradB = gradB.at[k].set(gradG_J[k1, k2] - gradG_J[k2, k1])
        gradB = gradB + gradgradG_BdotN

        if X_trg.ndim == 3:
            return gradB.reshape((3, 3, X_trg.shape[1], X_trg.shape[2]))
        return gradB

    def compute_internal_gradB_offsurf(
        self,
        B0,
        *,
        X_trg,
        digits: int | None = None,
        max_Nt: int = -1,
        max_Np: int = -1,
        adaptive: bool = False,
        chunk_size: int | str | None = "auto",
        target_chunk_size: int | str | None = "auto",
    ):
        """Compute GradBint at off-surface targets using direct quadrature."""
        gradB = self.compute_external_gradB_offsurf(
            B0,
            X_trg=X_trg,
            digits=digits,
            max_Nt=max_Nt,
            max_Np=max_Np,
            adaptive=adaptive,
            chunk_size=chunk_size,
            target_chunk_size=target_chunk_size,
        )
        return -gradB

    def compute_external_gradB_offsurf_schedule(
        self,
        B0,
        *,
        X_trg,
        levels: tuple[tuple[int, int], ...],
        digits: int | None = None,
        chunk_size: int | str | None = "auto",
        target_chunk_size: int | str | None = "auto",
    ):
        """Compute GradBext off-surface using a fixed adaptive refinement schedule."""
        digits = self.digits if digits is None else int(digits)
        X_src, BdotN, J = self._offsurface_densities(B0)
        X_trg = jnp.asarray(X_trg)
        X_trg_flat = X_trg.reshape((3, -1)) if X_trg.ndim == 3 else X_trg
        nsrc = X_src.shape[1] * X_src.shape[2]
        ntrg = X_trg_flat.shape[1]
        chunk_size, target_chunk_size = self._resolve_chunk_sizes(
            "gradb_off", chunk_size, target_chunk_size, nsrc=nsrc, ntrg=ntrg
        )
        gradB = computeGradB_offsurface_adaptive_schedule(
            X_src,
            BdotN,
            J,
            X_trg_flat,
            levels=levels,
            digits=digits,
            ext=True,
            chunk_size=chunk_size,
            target_chunk_size=target_chunk_size,
        )
        if X_trg.ndim == 3:
            return jnp.asarray(gradB).reshape((3, 3, X_trg.shape[1], X_trg.shape[2]))
        return gradB

    def compute_internal_gradB_offsurf_schedule(
        self,
        B0,
        *,
        X_trg,
        levels: tuple[tuple[int, int], ...],
        digits: int | None = None,
        chunk_size: int | str | None = "auto",
        target_chunk_size: int | str | None = "auto",
    ):
        """Compute GradBint off-surface using a fixed adaptive refinement schedule."""
        digits = self.digits if digits is None else int(digits)
        X_src, BdotN, J = self._offsurface_densities(B0)
        X_trg = jnp.asarray(X_trg)
        X_trg_flat = X_trg.reshape((3, -1)) if X_trg.ndim == 3 else X_trg
        nsrc = X_src.shape[1] * X_src.shape[2]
        ntrg = X_trg_flat.shape[1]
        chunk_size, target_chunk_size = self._resolve_chunk_sizes(
            "gradb_off", chunk_size, target_chunk_size, nsrc=nsrc, ntrg=ntrg
        )
        gradB = computeGradB_offsurface_adaptive_schedule(
            X_src,
            BdotN,
            J,
            X_trg_flat,
            levels=levels,
            digits=digits,
            ext=False,
            chunk_size=chunk_size,
            target_chunk_size=target_chunk_size,
        )
        if X_trg.ndim == 3:
            return jnp.asarray(gradB).reshape((3, 3, X_trg.shape[1], X_trg.shape[2]))
        return gradB
