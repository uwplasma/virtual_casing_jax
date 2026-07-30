"""Independent physical validation for the internal on-surface GradB limit.

The historical ``*_int_computeGradB`` dumps were produced with the same
sign-only implementation that this test is intended to catch, so they are not
used as an oracle here.
"""
from __future__ import annotations

import numpy as np
import pytest

from virtual_casing_jax import SurfType, VirtualCasingJAX
from virtual_casing_jax import testdata
from virtual_casing_jax.integrals import field_period_target_coords


@pytest.mark.large
def test_internal_gradb_matches_analytic_loop_field():
    nfp = 1
    half_period = False
    src_nt, src_np = 24, 20
    trg_nt, trg_np = 4, 4

    X = testdata.surface_coordinates(
        nfp, half_period, src_nt, src_np, SurfType.AxisymNarrow
    )
    Bext, Bint = testdata.magnetic_field_data(
        nfp, half_period, src_nt, src_np, X, src_nt, src_np
    )
    grad_ext_ref, grad_int_ref = testdata.magnetic_field_grad_data(
        nfp, half_period, src_nt, src_np, X, trg_nt, trg_np
    )

    vc = VirtualCasingJAX()
    vc.setup(
        6,
        nfp,
        half_period,
        src_nt,
        src_np,
        X,
        src_nt,
        src_np,
        trg_nt,
        trg_np,
    )
    Btotal = Bext + Bint
    grad_ext = np.asarray(vc.compute_external_gradB(Btotal, chunk_size=512))
    grad_int = np.asarray(vc.compute_internal_gradB(Btotal, chunk_size=512))
    grad_ext_ref = np.asarray(grad_ext_ref)
    grad_int_ref = np.asarray(grad_int_ref)

    def relerr(got, ref):
        return np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-14)

    # The internal component is the primary independent oracle.  The sum is
    # more accurately resolved than the much smaller external component.
    assert relerr(grad_int, grad_int_ref) < 1e-2
    assert relerr(grad_ext + grad_int, grad_ext_ref + grad_int_ref) < 1e-2

    antisymmetric = grad_int - np.swapaxes(grad_int, 0, 1)
    trace = np.trace(grad_int, axis1=0, axis2=1)
    scale = np.linalg.norm(grad_int) + 1e-14
    assert np.linalg.norm(antisymmetric) / scale < 1e-2
    assert np.linalg.norm(trace) / scale < 1e-2

    setup = vc._grad_setup
    target = field_period_target_coords(
        setup.quad_coord, trg_nt, trg_np, vc.nfp_eff
    )
    normal = field_period_target_coords(
        setup.normal, trg_nt, trg_np, vc.nfp_eff
    )
    one_sided_errors = []
    for epsilon in (0.02, 0.01, 0.0025):
        _, grad_int_off = testdata.magnetic_field_grad_data_offsurf(
            nfp,
            half_period,
            src_nt,
            src_np,
            X,
            target + epsilon * normal,
        )
        one_sided_errors.append(relerr(grad_int, np.asarray(grad_int_off)))

    assert one_sided_errors[-1] < one_sided_errors[0]
    assert one_sided_errors[-1] < 2e-2
