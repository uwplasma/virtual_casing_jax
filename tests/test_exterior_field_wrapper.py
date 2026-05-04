import numpy as np
import pytest
import jax.numpy as jnp

from virtual_casing_jax import (
    ExteriorFieldConfig,
    VirtualCasingExteriorField,
    VmecSurfaceFieldData,
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
