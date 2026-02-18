"""virtual_casing_jax package."""

from . import surface_ops
from .surface_ops import (
    complete_vec_field,
    rotate_toroidal,
    upsample,
    resample,
    grad2d,
    surf_normal_area_elem,
    dot_prod,
    cross_prod,
)
from .kernels import (
    laplace_fx_u,
    laplace_fxd_u,
    laplace_fxd2_u,
    biotsavart_fx_u,
    biotsavart_fxd_u,
)
from .integrals import (
    laplace_fxd_u_eval,
    laplace_fxd_u_eval_vec,
    field_period_target_coords,
    biotsavart_fx_u_eval,
    computeB_offsurface_baseline,
)

__all__ = [
    "surface_ops",
    "complete_vec_field",
    "rotate_toroidal",
    "upsample",
    "resample",
    "grad2d",
    "surf_normal_area_elem",
    "dot_prod",
    "cross_prod",
    "laplace_fx_u",
    "laplace_fxd_u",
    "laplace_fxd2_u",
    "biotsavart_fx_u",
    "biotsavart_fxd_u",
    "laplace_fxd_u_eval",
    "laplace_fxd_u_eval_vec",
    "field_period_target_coords",
    "biotsavart_fx_u_eval",
    "computeB_offsurface_baseline",
]
