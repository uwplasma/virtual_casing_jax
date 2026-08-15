from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from virtual_casing_jax import (
    ExteriorFieldConfig,
    VirtualCasingExteriorField,
    VmecSurfaceFieldData,
    build_near_surface_taylor_plan,
    cyl_to_xyz,
    xyz_vec_to_cyl_vec,
)


def _surface_data(nfp=1):
    phi = jnp.linspace(0.0, 2.0 * jnp.pi / nfp, 4, endpoint=False)
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, 3, endpoint=False)
    theta2d, phi2d = jnp.meshgrid(theta, phi)
    gamma = jnp.stack(
        (
            (2.0 + 0.2 * jnp.cos(theta2d)) * jnp.cos(phi2d),
            (2.0 + 0.2 * jnp.cos(theta2d)) * jnp.sin(phi2d),
            0.2 * jnp.sin(theta2d),
        ),
        axis=0,
    )
    normal = jnp.stack(
        (
            jnp.cos(theta2d) * jnp.cos(phi2d),
            jnp.cos(theta2d) * jnp.sin(phi2d),
            jnp.sin(theta2d),
        ),
        axis=0,
    )
    return VmecSurfaceFieldData(
        gamma=gamma,
        B_total=0.1 * normal,
        normal=normal,
        area_vector=0.2 * (2.0 + 0.2 * jnp.cos(theta2d)) * normal,
        theta=theta,
        phi=phi,
        nfp=nfp,
        stellsym=False,
        signgs=1,
        source_convention="unit-test-wrapper",
    )


class _RecordingVC:
    def __init__(self):
        self.calls = []

    def compute_internal_B_offsurf_schedule(self, B_total, **kwargs):
        self.calls.append(("internal_B_schedule", kwargs))
        return 2.0 * kwargs["X_trg"] + 0.5

    def plan_precision(self, **kwargs):
        self.calls.append(("plan_precision", kwargs))
        return (kwargs["digits"], kwargs["quad_nt"], kwargs["quad_np"])

    def compute_internal_B(self, B_total, **kwargs):
        self.calls.append(("internal_B_surface", kwargs))
        return 4.0 * B_total

    def compute_internal_gradB(self, B_total, **kwargs):
        self.calls.append(("internal_gradB_surface", kwargs))
        return jnp.zeros((3, 3) + B_total.shape[1:], dtype=B_total.dtype)

    def compute_external_B_offsurf_schedule(self, B_total, **kwargs):
        self.calls.append(("external_B_schedule", kwargs))
        return -2.0 * kwargs["X_trg"] - 0.5

    def compute_internal_B_offsurf(self, B_total, **kwargs):
        self.calls.append(("internal_B_direct", kwargs))
        return 3.0 * kwargs["X_trg"] + 0.25

    def compute_external_B_offsurf(self, B_total, **kwargs):
        self.calls.append(("external_B_direct", kwargs))
        return -3.0 * kwargs["X_trg"] - 0.25

    def compute_internal_gradB_offsurf_schedule(self, B_total, **kwargs):
        self.calls.append(("internal_gradB_schedule", kwargs))
        return self._grad(2.0, kwargs["X_trg"])

    def compute_external_gradB_offsurf_schedule(self, B_total, **kwargs):
        self.calls.append(("external_gradB_schedule", kwargs))
        return self._grad(-2.0, kwargs["X_trg"])

    def compute_internal_gradB_offsurf(self, B_total, **kwargs):
        self.calls.append(("internal_gradB_direct", kwargs))
        return self._grad(3.0, kwargs["X_trg"])

    def compute_external_gradB_offsurf(self, B_total, **kwargs):
        self.calls.append(("external_gradB_direct", kwargs))
        return self._grad(-3.0, kwargs["X_trg"])

    @staticmethod
    def _grad(scale, X_trg):
        base = scale * jnp.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=X_trg.dtype,
        )
        return jnp.broadcast_to(base[:, :, None], (3, 3, X_trg.shape[1]))


def _field(*, config=None, external_B_fn=None, external_gradB_fn=None, nfp=1):
    field = VirtualCasingExteriorField(
        _surface_data(nfp=nfp),
        config or ExteriorFieldConfig(digits=3, levels=((5, 4),), target_chunk_size=2, dtype="float64"),
        external_B_fn=external_B_fn,
        external_gradB_fn=external_gradB_fn,
    )
    recorder = _RecordingVC()
    field._vc = recorder
    return field, recorder


def test_exterior_field_validates_configuration_and_layout_errors():
    data = _surface_data()

    with pytest.raises(ValueError, match="Unsupported dtype"):
        VirtualCasingExteriorField(data, ExteriorFieldConfig(dtype="complex128"))

    with pytest.raises(ValueError, match="branch"):
        VirtualCasingExteriorField(data, ExteriorFieldConfig(branch="vacuum"))

    with pytest.raises(ValueError, match="levels entries must be positive"):
        VirtualCasingExteriorField(data, ExteriorFieldConfig(levels=((0, 4),)))

    with pytest.raises(ValueError, match="shape"):
        bad = VmecSurfaceFieldData(
            gamma=data.gamma[0],
            B_total=data.B_total,
            normal=data.normal,
            area_vector=data.area_vector,
            theta=data.theta,
            phi=data.phi,
            nfp=data.nfp,
            stellsym=data.stellsym,
            signgs=data.signgs,
        )
        VirtualCasingExteriorField(bad, ExteriorFieldConfig(digits=3, levels=((5, 4),)))

    with pytest.raises(ValueError, match="identical shapes"):
        bad = VmecSurfaceFieldData(
            gamma=data.gamma,
            B_total=data.B_total[:, :, :-1],
            normal=data.normal,
            area_vector=data.area_vector,
            theta=data.theta,
            phi=data.phi,
            nfp=data.nfp,
            stellsym=data.stellsym,
            signgs=data.signgs,
        )
        VirtualCasingExteriorField(bad, ExteriorFieldConfig(digits=3, levels=((5, 4),)))


def test_exterior_field_restores_point_layouts_and_routes_scheduled_branches():
    field, recorder = _field(nfp=2)
    assert field.schedule_levels == ((6, 4),)

    point = jnp.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(field.B_plasma_xyz(point), 2.0 * point + 0.5)
    assert recorder.calls[-1][0] == "internal_B_schedule"
    assert recorder.calls[-1][1]["levels"] == ((6, 4),)

    aos = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.testing.assert_allclose(field.B_external_xyz(aos), -2.0 * aos - 0.5)
    assert recorder.calls[-1][0] == "external_B_schedule"

    soa = jnp.arange(12.0).reshape((3, 2, 2))
    got = field.B_plasma_xyz(soa)
    np.testing.assert_allclose(got, 2.0 * soa + 0.5)
    assert got.shape == soa.shape

    with pytest.raises(ValueError, match="1D point input"):
        field.B_plasma_xyz(jnp.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="Point input"):
        field.B_plasma_xyz(jnp.ones((2, 2)))

    with pytest.raises(ValueError, match="branch"):
        field.B_plasma_xyz(point, branch="bad")


def test_sharded_target_api_has_single_device_fallback_and_validation():
    field, _ = _field(external_B_fn=lambda xyz: jnp.ones_like(xyz))
    points = jnp.asarray([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]])
    np.testing.assert_allclose(
        field.B_xyz_sharded(points, devices=[jax.devices()[0]]),
        field.B_xyz(points))
    np.testing.assert_allclose(field.B_xyz_sharded(points), field.B_xyz(points))
    with pytest.raises(ValueError, match="at least one"):
        field.B_xyz_sharded(points, devices=[])


def test_exterior_field_reuses_prepared_geometry_for_surface_quadrature():
    field, recorder = _field()
    plan = field.plan_surface_precision(digits=4, quad_nt=9, quad_np=7)
    assert plan == (4, 9, 7)
    np.testing.assert_allclose(
        field.B_plasma_on_surface(digits=4, precision=plan),
        4.0 * field.B_total,
    )
    assert [call[0] for call in recorder.calls] == [
        "plan_precision", "internal_B_surface"
    ]
    assert recorder.calls[-1][1]["precision"] == plan


def test_near_surface_plan_reuses_precision_for_gradient_quadrature():
    field, recorder = _field()
    precision = SimpleNamespace(
        quad_nt=9, quad_np=7, patch_dim0=4, patch_idx=jnp.arange(3))
    plan = field.plan_near_surface(
        digits=4, precision=precision, B_surface=field.B_total)

    assert plan.gradB_surface.shape == (3, 3) + field.B_total.shape[1:]
    name, kwargs = recorder.calls[-1]
    assert name == "internal_gradB_surface"
    assert kwargs["quad_nt"] == 9 and kwargs["quad_np"] == 7
    assert kwargs["patch_dim0"] == 4
    np.testing.assert_array_equal(kwargs["patch_idx"], precision.patch_idx)


def test_near_surface_plan_projects_smoothly_and_rotates_field_periods():
    data = _surface_data(nfp=2)
    phi = data.phi[:, None]
    B_surface = jnp.broadcast_to(jnp.stack((
        -2.0 * jnp.sin(phi), 2.0 * jnp.cos(phi), jnp.zeros_like(phi))),
        data.gamma.shape)
    gradB_surface = jnp.zeros((3, 3) + data.gamma.shape[1:])
    gradB_surface = gradB_surface.at[0, 0].set(0.3 * jnp.cos(phi) ** 2)
    gradB_surface = gradB_surface.at[0, 1].set(0.3 * jnp.cos(phi) * jnp.sin(phi))
    gradB_surface = gradB_surface.at[1, 0].set(0.3 * jnp.cos(phi) * jnp.sin(phi))
    gradB_surface = gradB_surface.at[1, 1].set(0.3 * jnp.sin(phi) ** 2)
    plan = build_near_surface_taylor_plan(data, B_surface, gradB_surface)
    field, _ = _field(nfp=2)
    target_phi = 1.5 * jnp.pi
    points = jnp.array([[2.25 * jnp.cos(target_phi), 2.25 * jnp.sin(target_phi), 0.0]])

    got = field.B_plasma_near_surface_xyz(points, plan)
    radial = jnp.array([jnp.cos(target_phi), jnp.sin(target_phi), 0.0])
    expected = jnp.array([[-2.0 * jnp.sin(target_phi), 2.0 * jnp.cos(target_phi), 0.0]])
    expected = expected + 0.05 * 0.3 * radial
    np.testing.assert_allclose(got, expected, rtol=1e-11, atol=1e-11)


def test_exterior_field_uses_nonjit_direct_paths_and_external_callbacks():
    def external_B(xyz):
        return jnp.ones_like(xyz)

    cfg = ExteriorFieldConfig(
        digits=3,
        levels=((5, 4),),
        branch="external",
        use_jit_schedule=False,
        target_chunk_size=2,
        dtype="float64",
    )
    field, recorder = _field(config=cfg, external_B_fn=external_B)
    points = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    np.testing.assert_allclose(field.B_plasma_xyz(points), -3.0 * points - 0.25)
    assert recorder.calls[-1][0] == "external_B_direct"
    assert "levels" not in recorder.calls[-1][1]

    np.testing.assert_allclose(field.B_xyz(points), -3.0 * points + 0.75)
    assert recorder.calls[-1][0] == "external_B_direct"


def test_exterior_field_nonjit_internal_and_no_callback_paths():
    cfg = ExteriorFieldConfig(
        digits=3,
        levels=((5, 4),),
        branch="internal",
        use_jit_schedule=False,
        target_chunk_size=2,
        dtype="float64",
    )
    field, recorder = _field(config=cfg)
    points = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    np.testing.assert_allclose(field.B_plasma_xyz(points), 3.0 * points + 0.25)
    assert recorder.calls[-1][0] == "internal_B_direct"

    np.testing.assert_allclose(field.B_xyz(points), 3.0 * points + 0.25)
    assert recorder.calls[-1][0] == "internal_B_direct"

    expected_grad = 3.0 * jnp.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        dtype=jnp.float64,
    )
    np.testing.assert_allclose(
        field.gradB_plasma_xyz(points),
        jnp.broadcast_to(expected_grad, (2, 3, 3)),
    )
    assert recorder.calls[-1][0] == "internal_gradB_direct"

    np.testing.assert_allclose(
        field.gradB_plasma_xyz(points, branch="external"),
        jnp.broadcast_to(-expected_grad, (2, 3, 3)),
    )
    assert recorder.calls[-1][0] == "external_gradB_direct"

    np.testing.assert_allclose(
        field.gradB_xyz(points),
        jnp.broadcast_to(expected_grad, (2, 3, 3)),
    )
    assert recorder.calls[-1][0] == "internal_gradB_direct"


def test_exterior_field_gradB_restores_matrix_layouts_and_adds_external_gradient():
    external_grad = jnp.array(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        dtype=jnp.float64,
    )

    def external_gradB(xyz):
        xyz = jnp.asarray(xyz)
        if xyz.ndim == 1:
            return external_grad.astype(xyz.dtype)
        return jnp.broadcast_to(external_grad.astype(xyz.dtype), xyz.shape[:-1] + (3, 3))

    field, recorder = _field(external_gradB_fn=external_gradB)
    point = jnp.array([1.0, 2.0, 3.0])
    expected_internal = 2.0 * jnp.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        dtype=jnp.float64,
    )

    np.testing.assert_allclose(field.gradB_plasma_xyz(point), expected_internal)
    assert recorder.calls[-1][0] == "internal_gradB_schedule"

    aos = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    got_aos = field.gradB_xyz(aos)
    np.testing.assert_allclose(
        got_aos,
        jnp.broadcast_to(expected_internal + external_grad, (2, 3, 3)),
    )
    assert got_aos.shape == (2, 3, 3)

    soa = jnp.arange(12.0).reshape((3, 2, 2))
    got_soa = field.gradB_plasma_xyz(soa, branch="external")
    np.testing.assert_allclose(got_soa, -expected_internal.reshape((3, 3, 1, 1)) * jnp.ones((1, 1, 2, 2)))
    assert got_soa.shape == (3, 3, 2, 2)

    with pytest.raises(ValueError, match="branch"):
        field.gradB_plasma_xyz(point, branch="bad")


def test_exterior_field_public_grid_export_uses_total_cylindrical_field():
    def external_B(xyz):
        return jnp.ones_like(xyz)

    field, _ = _field(external_B_fn=external_B)
    grid = field.export_rphiz_grid(
        jnp.array([2.0]),
        jnp.array([0.0]),
        jnp.array([0.0, 0.5]),
        chunk_size=1,
    )

    assert grid["BR"].shape == (1, 1, 2)
    assert np.all(np.asarray(grid["absB"]) > 0.0)


def test_cylindrical_conversions_support_soa_layout_and_reject_bad_shapes():
    rphiz = jnp.array(
        [
            [[2.0, 3.0], [4.0, 5.0]],
            [[0.0, 0.5 * jnp.pi], [jnp.pi, 1.5 * jnp.pi]],
            [[0.1, 0.2], [0.3, 0.4]],
        ]
    )
    xyz = cyl_to_xyz(rphiz)
    expected_xyz = jnp.array(
        [
            [[2.0, 0.0], [-4.0, 0.0]],
            [[0.0, 3.0], [0.0, -5.0]],
            [[0.1, 0.2], [0.3, 0.4]],
        ]
    )
    np.testing.assert_allclose(xyz, expected_xyz, atol=1e-12)

    B_xyz = jnp.ones_like(rphiz)
    got = xyz_vec_to_cyl_vec(rphiz, B_xyz)
    np.testing.assert_allclose(got[0, 0, 0], 1.0, atol=1e-12)
    np.testing.assert_allclose(got[1, 0, 1], -1.0, atol=1e-12)

    with pytest.raises(ValueError, match="R_phi_Z"):
        cyl_to_xyz(jnp.ones((2, 2)))

    with pytest.raises(ValueError, match="R_phi_Z"):
        xyz_vec_to_cyl_vec(jnp.ones((2, 2)), jnp.ones((2, 2)))
