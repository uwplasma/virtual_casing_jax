import numpy as np
import pytest
import jax
import jax.numpy as jnp

from virtual_casing_jax.functional import (
    build_surface_coord,
    build_quad_setup,
    build_patch_idx,
    prepare_functional_setup,
    select_patch_dim_from_geom,
    target_surface_normal,
    compute_external_B_functional,
    compute_external_B_normal_functional,
    compute_external_B_jvp_columns_functional,
    compute_external_B_normal_jvp_columns_functional,
    compute_internal_B_functional,
    compute_external_gradB_functional,
    compute_internal_gradB_functional,
    compute_external_B_offsurf_functional,
    compute_external_gradB_offsurf_functional,
)
from virtual_casing_jax.virtual_casing import VirtualCasingJAX


def _torus(nt, npol, R0=2.0, r=0.3):
    phi = jnp.linspace(0.0, 2.0 * jnp.pi, nt, endpoint=False)
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, npol, endpoint=False)
    theta2d, phi2d = jnp.meshgrid(theta, phi)
    x = (R0 + r * jnp.cos(theta2d)) * jnp.cos(phi2d)
    y = (R0 + r * jnp.cos(theta2d)) * jnp.sin(phi2d)
    z = r * jnp.sin(theta2d)
    return jnp.stack([x, y, z], axis=0)


def test_functional_matches_class():
    nfp = 1
    half_period = False
    surf_nt = 6
    surf_np = 5
    src_nt = 6
    src_np = 5
    trg_nt = 6
    trg_np = 5
    digits = 4

    X = _torus(surf_nt, surf_np)
    B0 = X * 0.05 + 0.1

    surface_coord, nfp_eff = build_surface_coord(X, nfp, half_period, surf_nt, surf_np, trg_nt)
    quad_nt = surf_nt
    quad_np = surf_np
    quad_coord, dX, normal, _, orient = build_quad_setup(surface_coord, quad_nt, quad_np)
    patch_dim0 = select_patch_dim_from_geom(dX, quad_nt, quad_np, digits)
    patch_idx = build_patch_idx(quad_nt, quad_np, trg_nt, trg_np, nfp_eff, patch_dim0)

    vc = VirtualCasingJAX()
    vc.setup(digits, nfp, half_period, surf_nt, surf_np, X, src_nt, src_np, trg_nt, trg_np)
    bref = vc.compute_external_B(
        B0,
        quad_nt=quad_nt,
        quad_np=quad_np,
        digits=digits,
        patch_dim0=patch_dim0,
        patch_idx=patch_idx,
    )
    gradb_ref = vc.compute_external_gradB(
        B0,
        quad_nt=quad_nt,
        quad_np=quad_np,
        digits=digits,
        patch_dim0=patch_dim0,
        patch_idx=patch_idx,
    )

    bfun = compute_external_B_functional(
        X,
        B0,
        digits=digits,
        nfp=nfp,
        half_period=half_period,
        surf_nt=surf_nt,
        surf_np=surf_np,
        src_nt=src_nt,
        src_np=src_np,
        trg_nt=trg_nt,
        trg_np=trg_np,
        quad_nt=quad_nt,
        quad_np=quad_np,
        patch_dim0=patch_dim0,
        patch_idx=patch_idx,
        orient=orient,
    )
    gradb_fun = compute_external_gradB_functional(
        X,
        B0,
        digits=digits,
        nfp=nfp,
        half_period=half_period,
        surf_nt=surf_nt,
        surf_np=surf_np,
        src_nt=src_nt,
        src_np=src_np,
        trg_nt=trg_nt,
        trg_np=trg_np,
        quad_nt=quad_nt,
        quad_np=quad_np,
        patch_dim0=patch_dim0,
        patch_idx=patch_idx,
        orient=orient,
    )

    np.testing.assert_allclose(bfun, np.asarray(bref), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(gradb_fun, np.asarray(gradb_ref), rtol=1e-12, atol=1e-12)


def test_functional_grad_wrt_surface():
    nfp = 1
    half_period = False
    surf_nt = 5
    surf_np = 4
    src_nt = 5
    src_np = 4
    trg_nt = 5
    trg_np = 4
    digits = 4

    X = _torus(surf_nt, surf_np)
    B0 = X * 0.02 + 0.05

    surface_coord, nfp_eff = build_surface_coord(X, nfp, half_period, surf_nt, surf_np, trg_nt)
    quad_nt = surf_nt
    quad_np = surf_np
    quad_coord, dX, normal, _, orient = build_quad_setup(surface_coord, quad_nt, quad_np)
    patch_dim0 = select_patch_dim_from_geom(dX, quad_nt, quad_np, digits)
    patch_idx = build_patch_idx(quad_nt, quad_np, trg_nt, trg_np, nfp_eff, patch_dim0)

    X_trg = jnp.array(
        [
            [2.2, 2.25, 2.15],
            [0.0, 0.1, -0.1],
            [0.05, -0.02, 0.07],
        ]
    )

    def scalar_fn(xsurf):
        b = compute_external_B_offsurf_functional(
            xsurf,
            B0,
            X_trg=X_trg,
            digits=digits,
            nfp=nfp,
            half_period=half_period,
            surf_nt=surf_nt,
            surf_np=surf_np,
            src_nt=src_nt,
            src_np=src_np,
            trg_nt=trg_nt,
            trg_np=trg_np,
            adaptive=False,
        )
        return jnp.sum(b * b)

    grad = jax.grad(scalar_fn)(X)

    eps = 1e-5
    idx = (0, 0, 0)
    x_plus = X.at[idx].add(eps)
    x_minus = X.at[idx].add(-eps)
    fd = (scalar_fn(x_plus) - scalar_fn(x_minus)) / (2.0 * eps)

    np.testing.assert_allclose(grad[idx], np.asarray(fd), rtol=5e-3, atol=1e-5)


def test_external_B_normal_functional_matches_manual_projection():
    nfp = 1
    half_period = False
    surf_nt = 6
    surf_np = 5
    src_nt = 6
    src_np = 5
    trg_nt = 6
    trg_np = 5
    digits = 4

    X = _torus(surf_nt, surf_np)
    B0 = X * 0.03 + 0.07

    surface_coord, nfp_eff = build_surface_coord(X, nfp, half_period, surf_nt, surf_np, trg_nt)
    quad_nt = surf_nt
    quad_np = surf_np
    quad_coord, dX, normal, _, orient = build_quad_setup(surface_coord, quad_nt, quad_np)
    patch_dim0 = select_patch_dim_from_geom(dX, quad_nt, quad_np, digits)
    patch_idx = build_patch_idx(quad_nt, quad_np, trg_nt, trg_np, nfp_eff, patch_dim0)

    Bext = compute_external_B_functional(
        X,
        B0,
        digits=digits,
        nfp=nfp,
        half_period=half_period,
        surf_nt=surf_nt,
        surf_np=surf_np,
        src_nt=src_nt,
        src_np=src_np,
        trg_nt=trg_nt,
        trg_np=trg_np,
        quad_nt=quad_nt,
        quad_np=quad_np,
        patch_dim0=patch_dim0,
        patch_idx=patch_idx,
        orient=orient,
    )
    n = target_surface_normal(
        X,
        nfp=nfp,
        half_period=half_period,
        surf_nt=surf_nt,
        surf_np=surf_np,
        trg_nt=trg_nt,
        trg_np=trg_np,
        orient=orient,
    )
    Bnormal = compute_external_B_normal_functional(
        X,
        B0,
        digits=digits,
        nfp=nfp,
        half_period=half_period,
        surf_nt=surf_nt,
        surf_np=surf_np,
        src_nt=src_nt,
        src_np=src_np,
        trg_nt=trg_nt,
        trg_np=trg_np,
        quad_nt=quad_nt,
        quad_np=quad_np,
        patch_dim0=patch_dim0,
        patch_idx=patch_idx,
        orient=orient,
    )

    np.testing.assert_allclose(Bnormal, jnp.sum(Bext * n, axis=0), rtol=1e-12, atol=1e-12)


def test_external_B_normal_functional_jvp_wrt_surface():
    nfp = 1
    half_period = False
    surf_nt = 5
    surf_np = 4
    src_nt = 5
    src_np = 4
    trg_nt = 5
    trg_np = 4
    digits = 4

    X = _torus(surf_nt, surf_np)
    B0 = X * 0.02 + 0.05

    surface_coord, nfp_eff = build_surface_coord(X, nfp, half_period, surf_nt, surf_np, trg_nt)
    quad_nt = surf_nt
    quad_np = surf_np
    quad_coord, dX, normal, _, orient = build_quad_setup(surface_coord, quad_nt, quad_np)
    patch_dim0 = select_patch_dim_from_geom(dX, quad_nt, quad_np, digits)
    patch_idx = build_patch_idx(quad_nt, quad_np, trg_nt, trg_np, nfp_eff, patch_dim0)

    def scalar_fn(xsurf):
        bnormal = compute_external_B_normal_functional(
            xsurf,
            B0,
            digits=digits,
            nfp=nfp,
            half_period=half_period,
            surf_nt=surf_nt,
            surf_np=surf_np,
            src_nt=src_nt,
            src_np=src_np,
            trg_nt=trg_nt,
            trg_np=trg_np,
            quad_nt=quad_nt,
            quad_np=quad_np,
            patch_dim0=patch_dim0,
            patch_idx=patch_idx,
            orient=orient,
        )
        return jnp.sum(bnormal * bnormal)

    eps = 1e-5
    idx = (0, 0, 0)
    tangent = jnp.zeros_like(X).at[idx].set(1.0)
    _, jvp = jax.jvp(scalar_fn, (X,), (tangent,))

    x_plus = X.at[idx].add(eps)
    x_minus = X.at[idx].add(-eps)
    fd = (scalar_fn(x_plus) - scalar_fn(x_minus)) / (2.0 * eps)

    np.testing.assert_allclose(jvp, np.asarray(fd), rtol=5e-3, atol=1e-5)


def test_external_B_jvp_columns_match_individual_jvps():
    nfp = 1
    half_period = False
    surf_nt = 5
    surf_np = 4
    src_nt = 5
    src_np = 4
    trg_nt = 5
    trg_np = 4
    digits = 4

    X = _torus(surf_nt, surf_np)
    B0 = X * 0.02 + 0.05

    surface_coord, nfp_eff = build_surface_coord(X, nfp, half_period, surf_nt, surf_np, trg_nt)
    quad_nt = surf_nt
    quad_np = surf_np
    quad_coord, dX, normal, _, orient = build_quad_setup(surface_coord, quad_nt, quad_np)
    patch_dim0 = select_patch_dim_from_geom(dX, quad_nt, quad_np, digits)
    patch_idx = build_patch_idx(quad_nt, quad_np, trg_nt, trg_np, nfp_eff, patch_dim0)

    dX0 = jnp.zeros_like(X).at[0, 0, 0].set(1.0)
    dX1 = 0.1 * X
    dB0 = 0.2 * B0
    dB1 = jnp.zeros_like(B0).at[2, 1, 1].set(-0.3)
    X_tangents = jnp.stack([dX0, dX1])
    B0_tangents = jnp.stack([dB0, dB1])

    kwargs = dict(
        digits=digits,
        nfp=nfp,
        half_period=half_period,
        surf_nt=surf_nt,
        surf_np=surf_np,
        src_nt=src_nt,
        src_np=src_np,
        trg_nt=trg_nt,
        trg_np=trg_np,
        quad_nt=quad_nt,
        quad_np=quad_np,
        patch_dim0=patch_dim0,
        patch_idx=patch_idx,
        orient=orient,
    )

    def external_field(x, b0):
        return compute_external_B_functional(x, b0, **kwargs)

    Bext, columns = compute_external_B_jvp_columns_functional(
        X,
        B0,
        X_tangents,
        B0_tangents,
        **kwargs,
    )
    Bnormal, normal_columns = compute_external_B_normal_jvp_columns_functional(
        X,
        B0,
        X_tangents,
        B0_tangents,
        **kwargs,
    )

    jvp_columns = []
    normal_jvp_columns = []
    for j in range(2):
        Bext_j, dBext_j = jax.jvp(
            external_field,
            (X, B0),
            (X_tangents[j], B0_tangents[j]),
        )
        Bnormal_j, dBnormal_j = jax.jvp(
            lambda x, b0: compute_external_B_normal_functional(x, b0, **kwargs),
            (X, B0),
            (X_tangents[j], B0_tangents[j]),
        )
        np.testing.assert_allclose(Bext, Bext_j, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(Bnormal, Bnormal_j, rtol=1e-12, atol=1e-12)
        jvp_columns.append(dBext_j)
        normal_jvp_columns.append(dBnormal_j)

    np.testing.assert_allclose(
        columns,
        jnp.stack(jvp_columns),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        normal_columns,
        jnp.stack(normal_jvp_columns),
        rtol=1e-12,
        atol=1e-12,
    )


def test_half_period_setup_builds_symmetric_metadata_and_unit_normals():
    nfp = 2
    surf_nt = 4
    surf_np = 5
    trg_nt = 4
    trg_np = 5
    quad_nt = 16
    quad_np = 5

    X = _torus(surf_nt, surf_np)
    setup = prepare_functional_setup(
        X,
        digits=4,
        nfp=nfp,
        half_period=True,
        surf_nt=surf_nt,
        surf_np=surf_np,
        src_nt=surf_nt,
        src_np=surf_np,
        trg_nt=trg_nt,
        trg_np=trg_np,
        quad_nt=quad_nt,
        quad_np=quad_np,
        patch_dim0=5,
    )
    normal = target_surface_normal(
        X,
        nfp=nfp,
        half_period=True,
        surf_nt=surf_nt,
        surf_np=surf_np,
        trg_nt=trg_nt,
        trg_np=trg_np,
    )

    assert setup.nfp == nfp
    assert setup.nfp_eff == 2 * nfp
    assert setup.half_period is True
    assert setup.patch_idx.shape[-1] == (2 * setup.patch_dim0 + 1) ** 2
    np.testing.assert_allclose(jnp.linalg.norm(normal, axis=0), 1.0, rtol=1e-12, atol=1e-12)


def test_internal_external_functional_boundary_identities():
    nfp = 1
    half_period = False
    surf_nt = 6
    surf_np = 5
    src_nt = 6
    src_np = 5
    trg_nt = 6
    trg_np = 5
    digits = 4

    X = _torus(surf_nt, surf_np)
    B0 = X * 0.04 + jnp.array([0.2, -0.1, 0.05])[:, None, None]

    kwargs = dict(
        digits=digits,
        nfp=nfp,
        half_period=half_period,
        surf_nt=surf_nt,
        surf_np=surf_np,
        src_nt=src_nt,
        src_np=src_np,
        trg_nt=trg_nt,
        trg_np=trg_np,
        quad_nt=surf_nt,
        quad_np=surf_np,
        chunk_size=128,
        target_chunk_size=7,
        pou_dtype="auto",
        patch_dtype="float64",
        interp_block_size=8,
        remat=True,
    )

    Bext = compute_external_B_functional(X, B0, **kwargs)
    Bint = compute_internal_B_functional(X, B0, **kwargs)
    grad_ext = compute_external_gradB_functional(X, B0, hedgehog_order=4, **kwargs)
    grad_int = compute_internal_gradB_functional(X, B0, hedgehog_order=4, **kwargs)

    # VCP jump relation on Gamma: external + internal branches recover B on the surface.
    np.testing.assert_allclose(Bext + Bint, B0, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(grad_ext + grad_int, 0.0, rtol=1e-10, atol=1e-10)


def test_external_B_functional_rejects_bad_offsurface_target_rank():
    nfp = 1
    surf_nt = 4
    surf_np = 4
    X = _torus(surf_nt, surf_np)
    B0 = X * 0.01 + 0.2

    with pytest.raises(ValueError, match="X_trg must have shape"):
        compute_external_B_functional(
            X,
            B0,
            X_trg=jnp.zeros((3,)),
            digits=4,
            nfp=nfp,
            half_period=False,
            surf_nt=surf_nt,
            surf_np=surf_np,
            src_nt=surf_nt,
            src_np=surf_np,
            trg_nt=surf_nt,
            trg_np=surf_np,
            quad_nt=surf_nt,
            quad_np=surf_np,
            patch_dim0=3,
        )


def test_functional_offsurface_adaptive_matches_fixed_grid_when_no_refinement():
    nfp = 1
    half_period = False
    surf_nt = 5
    surf_np = 4
    src_nt = 5
    src_np = 4
    trg_nt = 5
    trg_np = 4
    digits = 4

    X = _torus(surf_nt, surf_np)
    B0 = X * 0.02 + jnp.array([0.1, -0.03, 0.04])[:, None, None]
    X_trg = jnp.array(
        [
            [2.5, 2.3, 2.4],
            [0.0, 0.2, -0.15],
            [0.1, -0.05, 0.03],
        ]
    )
    kwargs = dict(
        X_trg=X_trg,
        digits=digits,
        nfp=nfp,
        half_period=half_period,
        surf_nt=surf_nt,
        surf_np=surf_np,
        src_nt=src_nt,
        src_np=src_np,
        trg_nt=trg_nt,
        trg_np=trg_np,
        max_Nt=13,
        max_Np=13,
        chunk_size=64,
        target_chunk_size=2,
    )

    direct = compute_external_B_offsurf_functional(X, B0, adaptive=False, **kwargs)
    adaptive = compute_external_B_offsurf_functional(X, B0, adaptive=True, **kwargs)

    np.testing.assert_allclose(adaptive, direct, rtol=1e-12, atol=1e-12)


def test_functional_offsurface_gradB_matches_target_finite_difference_and_reshapes():
    nfp = 1
    half_period = False
    surf_nt = 5
    surf_np = 4
    src_nt = 5
    src_np = 4
    trg_nt = 5
    trg_np = 4
    digits = 4

    X = _torus(surf_nt, surf_np)
    B0 = X * 0.02 + jnp.array([0.1, -0.03, 0.04])[:, None, None]
    target = jnp.array([[2.5], [0.0], [0.1]])
    kwargs = dict(
        digits=digits,
        nfp=nfp,
        half_period=half_period,
        surf_nt=surf_nt,
        surf_np=surf_np,
        src_nt=src_nt,
        src_np=src_np,
        trg_nt=trg_nt,
        trg_np=trg_np,
        max_Nt=13,
        max_Np=13,
        chunk_size=64,
        target_chunk_size=1,
    )

    grad = compute_external_gradB_offsurf_functional(X, B0, X_trg=target, adaptive=False, **kwargs)
    grad_grid = compute_external_gradB_offsurf_functional(
        X,
        B0,
        X_trg=target.reshape((3, 1, 1)),
        adaptive=True,
        **kwargs,
    )

    eps = 1e-5
    target_plus = target.at[0, 0].add(eps)
    target_minus = target.at[0, 0].add(-eps)
    B_plus = compute_external_B_offsurf_functional(X, B0, X_trg=target_plus, adaptive=False, **kwargs)
    B_minus = compute_external_B_offsurf_functional(X, B0, X_trg=target_minus, adaptive=False, **kwargs)
    fd = (B_plus[:, 0] - B_minus[:, 0]) / (2.0 * eps)

    assert grad.shape == (3, 3, 1)
    assert grad_grid.shape == (3, 3, 1, 1)
    np.testing.assert_allclose(grad_grid[:, :, 0, 0], grad[:, :, 0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(grad[:, 0, 0], fd, rtol=3e-4, atol=3e-6)
