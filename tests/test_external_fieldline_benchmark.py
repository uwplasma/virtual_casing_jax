from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "external" / "fieldline_compare.py"
spec = importlib.util.spec_from_file_location("fieldline_compare", BENCHMARK)
fieldline_compare = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(fieldline_compare)

GRID_CANDIDATE = ROOT / "benchmarks" / "external" / "trace_fieldlines_grid_candidate.py"
grid_spec = importlib.util.spec_from_file_location("trace_fieldlines_grid_candidate", GRID_CANDIDATE)
trace_fieldlines_grid_candidate = importlib.util.module_from_spec(grid_spec)
assert grid_spec.loader is not None
grid_spec.loader.exec_module(trace_fieldlines_grid_candidate)

SIMSOPT_CANDIDATE = ROOT / "benchmarks" / "external" / "trace_fieldlines_simsopt_candidate.py"
simsopt_spec = importlib.util.spec_from_file_location("trace_fieldlines_simsopt_candidate", SIMSOPT_CANDIDATE)
trace_fieldlines_simsopt_candidate = importlib.util.module_from_spec(simsopt_spec)
assert simsopt_spec.loader is not None
sys.modules[simsopt_spec.name] = trace_fieldlines_simsopt_candidate
simsopt_spec.loader.exec_module(trace_fieldlines_simsopt_candidate)


def _samples(point_delta: float = 0.0, length_delta: float = 0.0, hit_flip: bool = False):
    points = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    hits = np.asarray([False, False, True, True])
    if hit_flip:
        hits = hits.copy()
        hits[1] = True
    return {
        "poincare_xyz": points + point_delta,
        "connection_lengths": np.asarray([10.0, 11.0, 12.0, 13.0]) + length_delta,
        "hit_mask": hits,
    }


def _labeled_samples(point_delta: float = 0.0):
    samples = _samples(point_delta=point_delta)
    samples["line_id"] = np.asarray([0.0, 1.0, 0.0, 1.0])
    samples["section_phi"] = np.asarray([0.0, 0.0, 0.5, 0.5])
    return samples


def test_ordered_point_and_connection_metrics_report_physics_errors():
    reference = _samples()
    candidate = _samples(point_delta=1e-3, length_delta=1e-2)

    metrics = fieldline_compare.compare_samples(reference, candidate, point_mode="ordered")

    assert metrics["point_coordinates_compared"] == ["poincare_xyz"]
    assert metrics["compared_connection_lengths"] is True
    assert metrics["poincare_xyz_point_relative_l2"] > 0.0
    assert metrics["poincare_xyz_point_rms_distance"] > 0.0
    assert metrics["connection_length_relative_l2"] > 0.0
    assert metrics["hit_mask_mismatch_fraction"] == 0.0


def test_cloud_point_metrics_are_invariant_to_permutation():
    reference = _samples()
    candidate = _samples()
    candidate["poincare_xyz"] = candidate["poincare_xyz"][[2, 0, 3, 1]]

    ordered = fieldline_compare.compare_samples(reference, candidate, point_mode="ordered")
    cloud = fieldline_compare.compare_samples(reference, candidate, point_mode="cloud")

    assert ordered["poincare_xyz_point_relative_l2"] > 0.0
    assert cloud["poincare_xyz_cloud_symmetric_rms_distance"] == 0.0
    assert cloud["poincare_xyz_cloud_symmetric_max_distance"] == 0.0


def test_labeled_point_metrics_match_permuted_fieldline_sections():
    reference = _labeled_samples()
    candidate = _labeled_samples()
    permutation = np.asarray([3, 1, 0, 2])
    for key in ("poincare_xyz", "line_id", "section_phi"):
        candidate[key] = candidate[key][permutation]

    ordered = fieldline_compare.compare_samples(reference, candidate, point_mode="ordered")
    labeled = fieldline_compare.compare_samples(reference, candidate, point_mode="labeled")

    assert ordered["poincare_xyz_point_relative_l2"] > 0.0
    assert labeled["poincare_xyz_labeled_matched_label_count"] == 4
    assert labeled["poincare_xyz_labeled_missing_candidate_label_count"] == 0
    assert labeled["poincare_xyz_labeled_extra_candidate_label_count"] == 0
    assert labeled["poincare_xyz_labeled_count_mismatch_label_count"] == 0
    assert labeled["poincare_xyz_labeled_point_relative_l2"] == 0.0
    assert labeled["poincare_xyz_labeled_point_max_distance"] == 0.0


def test_labeled_point_metrics_report_missing_extra_and_count_mismatch_labels():
    reference = _labeled_samples()
    candidate = _labeled_samples()
    candidate["section_phi"] = candidate["section_phi"].copy()
    candidate["section_phi"][0] = 1.0
    candidate["line_id"] = np.concatenate([candidate["line_id"], [candidate["line_id"][1]]])
    candidate["section_phi"] = np.concatenate([candidate["section_phi"], [candidate["section_phi"][1]]])
    candidate["poincare_xyz"] = np.vstack([candidate["poincare_xyz"], candidate["poincare_xyz"][1]])

    metrics = fieldline_compare.compare_samples(reference, candidate, point_mode="labeled")
    args = argparse.Namespace(
        max_point_relative_l2=1.0,
        max_point_rms_distance=1.0,
        max_point_max_distance=1.0,
        max_cloud_relative_rms_distance=1.0,
        max_cloud_rms_distance=1.0,
        max_cloud_max_distance=1.0,
        max_connection_relative_l2=1.0,
        max_connection_max_abs=None,
        max_hit_mismatch_fraction=1.0,
    )

    failures = fieldline_compare._threshold_failures(metrics, args)

    assert metrics["poincare_xyz_labeled_missing_candidate_label_count"] == 1
    assert metrics["poincare_xyz_labeled_extra_candidate_label_count"] == 1
    assert metrics["poincare_xyz_labeled_count_mismatch_label_count"] == 1
    assert any("missing_candidate_label_count" in failure for failure in failures)
    assert any("extra_candidate_label_count" in failure for failure in failures)
    assert any("count_mismatch_label_count" in failure for failure in failures)


def test_labeled_point_metrics_require_line_and_section_labels():
    with pytest.raises(ValueError, match="requires both"):
        fieldline_compare.compare_samples(_samples(), _samples(), point_mode="labeled")


def test_threshold_failures_cover_poincare_connection_and_wall_hit_metrics():
    metrics = fieldline_compare.compare_samples(
        _samples(),
        _samples(point_delta=1e-2, length_delta=1e-1, hit_flip=True),
        point_mode="ordered",
    )
    args = argparse.Namespace(
        max_point_relative_l2=1e-6,
        max_point_rms_distance=1e-6,
        max_point_max_distance=1e-6,
        max_cloud_relative_rms_distance=1e-6,
        max_cloud_rms_distance=1e-6,
        max_cloud_max_distance=1e-6,
        max_connection_relative_l2=1e-6,
        max_connection_max_abs=1e-6,
        max_hit_mismatch_fraction=0.0,
    )

    failures = fieldline_compare._threshold_failures(metrics, args)

    assert any("poincare_xyz_point_relative_l2" in failure for failure in failures)
    assert any("connection_length_relative_l2" in failure for failure in failures)
    assert any("hit_mask_mismatch_fraction" in failure for failure in failures)


def test_load_fieldline_samples_from_json_npz_and_csv_aliases(tmp_path):
    json_path = tmp_path / "samples.json"
    json_path.write_text(
        json.dumps(
            [
                {"x": 1.0, "y": 0.0, "z": 0.0, "connection_length": 10.0, "wall_hit": 0},
                {"x": 0.0, "y": 1.0, "z": 0.0, "connection_length": 11.0, "wall_hit": 1},
            ]
        )
    )
    loaded_json = fieldline_compare.load_fieldline_samples(json_path)
    np.testing.assert_allclose(loaded_json["poincare_xyz"], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    np.testing.assert_allclose(loaded_json["connection_lengths"], [10.0, 11.0])
    np.testing.assert_array_equal(loaded_json["hit_mask"], [False, True])

    npz_path = tmp_path / "samples.npz"
    np.savez(
        npz_path,
        poincare_R_phi_Z=np.asarray([[1.0, 0.0, 0.0]]),
        lengths=np.asarray([4.0]),
        connected=np.asarray([1]),
    )
    loaded_npz = fieldline_compare.load_fieldline_samples(npz_path)
    np.testing.assert_allclose(loaded_npz["poincare_rphiz"], [[1.0, 0.0, 0.0]])
    np.testing.assert_allclose(loaded_npz["connection_lengths"], [4.0])
    np.testing.assert_array_equal(loaded_npz["hit_mask"], [True])

    csv_path = tmp_path / "samples.csv"
    csv_path.write_text("x,y,z,connection_lengths,hit_mask,line_id,section_phi\n1,0,0,5,0,2,0\n")
    loaded_csv = fieldline_compare.load_fieldline_samples(csv_path)
    np.testing.assert_allclose(loaded_csv["poincare_xyz"], [[1.0, 0.0, 0.0]])
    np.testing.assert_allclose(loaded_csv["connection_lengths"], [5.0])
    np.testing.assert_array_equal(loaded_csv["hit_mask"], [False])
    np.testing.assert_allclose(loaded_csv["line_id"], [2.0])
    np.testing.assert_allclose(loaded_csv["section_phi"], [0.0])


def test_load_stellopt_fieldlines_h5_samples_poincare_sections(tmp_path):
    h5py = pytest.importorskip("h5py")
    h5_path = tmp_path / "fieldlines_synthetic.h5"
    # STELLOPT/FIELDLINES writes trajectories as (nsteps, nlines) and uses
    # npoinc as the number of integration steps per field-period Poincare hit.
    with h5py.File(h5_path, "w") as f:
        f["R_lines"] = np.asarray(
            [
                [1.0, 2.0],
                [1.1, 2.1],
                [1.2, 2.2],
                [1.3, 2.3],
                [1.4, 2.4],
            ]
        )
        f["PHI_lines"] = np.asarray(
            [
                [0.0, 0.0],
                [0.5, 0.5],
                [1.0, 1.0],
                [1.5, 1.5],
                [2.0, 2.0],
            ]
        )
        f["Z_lines"] = np.asarray(
            [
                [0.0, 0.2],
                [0.1, 0.3],
                [0.2, 0.4],
                [0.3, 0.5],
                [0.4, 0.6],
            ]
        )
        f["L_lines"] = np.asarray([12.0, 14.0])
        f["npoinc"] = np.asarray([2], dtype=np.int32)

    loaded = fieldline_compare.load_fieldline_samples(h5_path)

    np.testing.assert_allclose(
        loaded["poincare_rphiz"],
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.2],
            [1.2, 1.0, 0.2],
            [2.2, 1.0, 0.4],
            [1.4, 2.0, 0.4],
            [2.4, 2.0, 0.6],
        ],
    )
    np.testing.assert_allclose(loaded["line_id"], [0, 1, 0, 1, 0, 1])
    np.testing.assert_allclose(loaded["section_phi"], [0, 0, 1, 1, 2, 2])
    np.testing.assert_allclose(loaded["connection_lengths"], [12.0, 14.0])


def test_stellopt_fieldlines_h5_defaults_to_every_step_without_npoinc(tmp_path):
    h5py = pytest.importorskip("h5py")
    h5_path = tmp_path / "fieldlines_no_npoinc.hdf5"
    with h5py.File(h5_path, "w") as f:
        f["R_lines"] = np.asarray([[1.0], [1.1], [1.2]])
        f["PHI_lines"] = np.asarray([[0.0], [0.5], [1.0]])
        f["Z_lines"] = np.asarray([[0.0], [0.1], [0.2]])
        f["wall_hits"] = np.asarray([0, 1, 0], dtype=np.int32)

    loaded = fieldline_compare.load_fieldline_samples(h5_path)

    np.testing.assert_allclose(
        loaded["poincare_rphiz"],
        [[1.0, 0.0, 0.0], [1.1, 0.5, 0.1], [1.2, 1.0, 0.2]],
    )
    np.testing.assert_array_equal(loaded["hit_mask"], [False, True, False])


def test_stellopt_fieldlines_h5_empty_npoinc_uses_every_step(tmp_path):
    h5py = pytest.importorskip("h5py")
    h5_path = tmp_path / "fieldlines_empty_npoinc.h5"
    with h5py.File(h5_path, "w") as f:
        f["R_lines"] = np.asarray([[1.0], [1.2]])
        f["PHI_lines"] = np.asarray([[0.0], [0.6]])
        f["Z_lines"] = np.asarray([[0.0], [0.2]])
        f["npoinc"] = np.asarray([], dtype=np.int32)

    loaded = fieldline_compare.load_fieldline_samples(h5_path)

    np.testing.assert_allclose(loaded["poincare_rphiz"], [[1.0, 0.0, 0.0], [1.2, 0.6, 0.2]])


def test_stellopt_fieldlines_h5_rejects_invalid_trajectory_contracts(tmp_path):
    h5py = pytest.importorskip("h5py")

    missing_path = tmp_path / "missing_z.h5"
    with h5py.File(missing_path, "w") as f:
        f["R_lines"] = np.asarray([[1.0]])
        f["PHI_lines"] = np.asarray([[0.0]])
    with pytest.raises(ValueError, match="missing trajectory"):
        fieldline_compare.load_fieldline_samples(missing_path)

    shape_path = tmp_path / "shape_mismatch.h5"
    with h5py.File(shape_path, "w") as f:
        f["R_lines"] = np.asarray([[1.0, 2.0]])
        f["PHI_lines"] = np.asarray([[0.0]])
        f["Z_lines"] = np.asarray([[0.0, 0.1]])
    with pytest.raises(ValueError, match="share shape"):
        fieldline_compare.load_fieldline_samples(shape_path)

    stride_path = tmp_path / "bad_stride.h5"
    with h5py.File(stride_path, "w") as f:
        f["R_lines"] = np.asarray([[1.0]])
        f["PHI_lines"] = np.asarray([[0.0]])
        f["Z_lines"] = np.asarray([[0.0]])
        f["npoinc"] = np.asarray([0], dtype=np.int32)
    with pytest.raises(ValueError, match="npoinc must be positive"):
        fieldline_compare.load_fieldline_samples(stride_path)

    length_path = tmp_path / "bad_lengths.h5"
    with h5py.File(length_path, "w") as f:
        f["R_lines"] = np.asarray([[1.0, 2.0], [1.1, 2.1]])
        f["PHI_lines"] = np.asarray([[0.0, 0.0], [0.5, 0.5]])
        f["Z_lines"] = np.asarray([[0.0, 0.2], [0.1, 0.3]])
        f["npoinc"] = np.asarray([1], dtype=np.int32)
        f["L_lines"] = np.asarray([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="L_lines"):
        fieldline_compare.load_fieldline_samples(length_path)


def test_stellopt_fieldlines_h5_rejects_empty_lines_and_required_scalar_errors(tmp_path):
    h5py = pytest.importorskip("h5py")

    empty_lines_path = tmp_path / "empty_lines.h5"
    with h5py.File(empty_lines_path, "w") as f:
        f["R_lines"] = np.empty((0, 1))
        f["PHI_lines"] = np.empty((0, 1))
        f["Z_lines"] = np.empty((0, 1))
    with pytest.raises(ValueError, match="at least one step"):
        fieldline_compare.load_fieldline_samples(empty_lines_path)

    with pytest.raises(ValueError, match="missing 'required_scalar'"):
        fieldline_compare._h5_scalar({}, "required_scalar")

    empty_scalar_path = tmp_path / "empty_scalar.h5"
    with h5py.File(empty_scalar_path, "w") as f:
        f["required_scalar"] = np.asarray([], dtype=np.int32)
        with pytest.raises(ValueError, match="dataset 'required_scalar' is empty"):
            fieldline_compare._h5_scalar(f, "required_scalar")


def test_trace_fieldlines_grid_candidate_matches_constant_rhs(tmp_path):
    h5py = pytest.importorskip("h5py")
    h5_path = tmp_path / "constant_rhs_fieldlines.h5"
    raxis = np.linspace(1.0, 2.0, 5)
    phiaxis = np.linspace(0.0, 2.0, 6)
    zaxis = np.linspace(-0.5, 0.5, 5)
    dR_dphi = 0.1
    dZ_dphi = -0.05
    phi_steps = np.linspace(0.0, 1.2, 7)
    with h5py.File(h5_path, "w") as f:
        f["raxis"] = raxis
        f["phiaxis"] = phiaxis
        f["zaxis"] = zaxis
        f["B_R"] = np.full((zaxis.size, phiaxis.size, raxis.size), dR_dphi)
        f["B_Z"] = np.full((zaxis.size, phiaxis.size, raxis.size), dZ_dphi)
        f["R_lines"] = (1.2 + dR_dphi * phi_steps)[:, None]
        f["PHI_lines"] = phi_steps[:, None]
        f["Z_lines"] = (0.1 + dZ_dphi * phi_steps)[:, None]
        f["npoinc"] = np.asarray([2], dtype=np.int32)

    reference, candidate, metadata = trace_fieldlines_grid_candidate.trace_grid_candidate(
        h5_path,
        lines=[0],
        nsections=4,
        substeps=1,
    )

    assert metadata["axis_order"] == "(nz, nphi, nr)"
    np.testing.assert_allclose(candidate["poincare_rphiz"], reference["poincare_rphiz"], atol=1e-14)
    labeled = fieldline_compare.compare_samples(reference, candidate, point_mode="labeled")
    assert labeled["poincare_rphiz_labeled_point_max_distance"] < 1e-14


def test_trace_fieldlines_grid_candidate_rejects_nonuniform_axes_and_terminated_lines(tmp_path):
    h5py = pytest.importorskip("h5py")
    h5_path = tmp_path / "bad_fieldlines.h5"
    with h5py.File(h5_path, "w") as f:
        f["raxis"] = np.asarray([1.0, 1.2, 1.8])
        f["phiaxis"] = np.linspace(0.0, 1.0, 3)
        f["zaxis"] = np.linspace(0.0, 1.0, 3)
        f["B_R"] = np.zeros((3, 3, 3))
        f["B_Z"] = np.zeros((3, 3, 3))
        f["R_lines"] = np.ones((3, 1))
        f["PHI_lines"] = np.asarray([[0.0], [-1.0], [-1.0]])
        f["Z_lines"] = np.zeros((3, 1))
        f["npoinc"] = np.asarray([1], dtype=np.int32)

    with pytest.raises(ValueError, match="raxis must be uniformly spaced"):
        trace_fieldlines_grid_candidate.trace_grid_candidate(
            h5_path,
            lines=[0],
            nsections=2,
            substeps=1,
        )

    with h5py.File(h5_path, "a") as f:
        del f["raxis"]
        f["raxis"] = np.linspace(1.0, 2.0, 3)
    with pytest.raises(ValueError, match="terminated before requested section count"):
        trace_fieldlines_grid_candidate.trace_grid_candidate(
            h5_path,
            lines=[0],
            nsections=2,
            substeps=1,
        )


def test_simsopt_candidate_parses_makegrid_extcur_and_fieldlines_current_scaling(tmp_path):
    coils_path = tmp_path / "coils.synthetic"
    coils_path.write_text(
        "\n".join(
            [
                "periods 1",
                "begin filament",
                "mirror NIL",
                "0 0 0 10",
                "1 0 0 20",
                "0 0 0 0 1 G1",
                "0 1 0 40",
                "0 2 0 80",
                "0 1 0 0 1 G1",
                "0 0 1 5",
                "0 0 2 5",
                "0 0 1 0 2 G2",
                "end",
                "",
            ]
        )
    )
    input_path = tmp_path / "input.synthetic"
    input_path.write_text(
        "\n".join(
            [
                "&INDATA",
                "! EXTCUR(1) = 999.0",
                "EXTCUR(1) = 1.0D2",
                "EXTCUR(1) = 2.0E2",
                "/",
            ]
        )
    )
    sanitized = trace_fieldlines_simsopt_candidate.write_sanitized_makegrid_file(
        coils_path, tmp_path / "coils.sanitized"
    )
    metadata = trace_fieldlines_simsopt_candidate.parse_makegrid_coil_metadata(sanitized)
    extcur = trace_fieldlines_simsopt_candidate.parse_fieldlines_extcur(input_path)

    scaled = trace_fieldlines_simsopt_candidate.scaled_currents_for_fieldlines(metadata, extcur)
    raw = trace_fieldlines_simsopt_candidate.scaled_currents_for_fieldlines(metadata, extcur, current_mode="raw")

    assert sanitized.read_text().splitlines()[-1] == "end"
    assert [(coil.group, coil.label, coil.point_count) for coil in metadata] == [(1, "G1", 2), (1, "G1", 2), (2, "G2", 2)]
    assert extcur == {1: 200.0}
    np.testing.assert_allclose(scaled, [200.0, 800.0, 0.0])
    np.testing.assert_allclose(raw, [10.0, 40.0, 5.0])
    with pytest.raises(ValueError, match="current_mode"):
        trace_fieldlines_simsopt_candidate.scaled_currents_for_fieldlines(metadata, extcur, current_mode="bad")


def test_simsopt_candidate_rejects_bad_makegrid_contracts_and_preserves_zero_first_current(tmp_path):
    missing_end = tmp_path / "missing_end.coils"
    missing_end.write_text("periods 1\nbegin filament\nmirror NIL\n0 0 0 1\n")
    with pytest.raises(ValueError, match="does not end"):
        trace_fieldlines_simsopt_candidate.write_sanitized_makegrid_file(missing_end, tmp_path / "out.coils")

    terminator_before_points = tmp_path / "terminator_first.coils"
    terminator_before_points.write_text(
        "periods 1\nbegin filament\nmirror NIL\n0 0 0 0 1 G1\nend\n"
    )
    with pytest.raises(ValueError, match="before coil points"):
        trace_fieldlines_simsopt_candidate.parse_makegrid_coil_metadata(terminator_before_points)

    bad_line = tmp_path / "bad_line.coils"
    bad_line.write_text("periods 1\nbegin filament\nmirror NIL\n0 0 0 1 2\nend\n")
    with pytest.raises(ValueError, match="unrecognized MAKEGRID line"):
        trace_fieldlines_simsopt_candidate.parse_makegrid_coil_metadata(bad_line)

    no_coils = tmp_path / "no_coils.coils"
    no_coils.write_text("periods 1\nbegin filament\nmirror NIL\nend\n")
    with pytest.raises(ValueError, match="no coils found"):
        trace_fieldlines_simsopt_candidate.parse_makegrid_coil_metadata(no_coils)

    metadata = [
        trace_fieldlines_simsopt_candidate.MakegridCoilMetadata(1, "zero", 0.0, 2),
        trace_fieldlines_simsopt_candidate.MakegridCoilMetadata(1, "zero", 5.0, 2),
    ]
    # FIELDLINES leaves a scaled group unchanged when the first raw group
    # current is zero, since the Fortran branch skips the ratio assignment.
    np.testing.assert_allclose(
        trace_fieldlines_simsopt_candidate.scaled_currents_for_fieldlines(metadata, {1: 12.0}),
        [0.0, 5.0],
    )


class _FakeCylindricalField:
    def __init__(self, dR_dphi: float = 0.2, dZ_dphi: float = -0.1, Bphi: float = 2.0):
        self.dR_dphi = dR_dphi
        self.dZ_dphi = dZ_dphi
        self.Bphi = Bphi
        self.points = None

    def set_points(self, points):
        self.points = np.asarray(points, dtype=float)

    def B(self):
        assert self.points is not None
        values = []
        for x, y, _z in self.points:
            R = np.hypot(x, y)
            phi = np.arctan2(y, x)
            BR = self.dR_dphi * self.Bphi / R
            BZ = self.dZ_dphi * self.Bphi / R
            values.append(
                [
                    np.cos(phi) * BR - np.sin(phi) * self.Bphi,
                    np.sin(phi) * BR + np.cos(phi) * self.Bphi,
                    BZ,
                ]
            )
        return np.asarray(values)


def test_simsopt_candidate_rhs_converts_cartesian_field_to_fieldline_equations():
    rhs = trace_fieldlines_simsopt_candidate.SimsoptBiotSavartRHS(
        _FakeCylindricalField(dR_dphi=0.2, dZ_dphi=-0.1, Bphi=2.0)
    )

    dR, dZ = rhs(np.pi / 4.0, 1.5, 0.25)

    np.testing.assert_allclose([dR, dZ], [0.2, -0.1], atol=1e-14)


def test_simsopt_candidate_rhs_rejects_zero_toroidal_field_component():
    rhs = trace_fieldlines_simsopt_candidate.SimsoptBiotSavartRHS(
        _FakeCylindricalField(dR_dphi=0.0, dZ_dphi=0.0, Bphi=0.0)
    )

    with pytest.raises(ValueError, match="Bphi is zero"):
        rhs(0.0, 1.0, 0.0)


def test_trace_fieldlines_simsopt_candidate_matches_constant_rhs(tmp_path, monkeypatch):
    h5py = pytest.importorskip("h5py")
    h5_path = tmp_path / "fieldlines_constant_simsopt.h5"
    dR_dphi = 0.03
    dZ_dphi = -0.02
    phi_steps = np.linspace(0.0, 1.2, 7)
    with h5py.File(h5_path, "w") as f:
        f["raxis"] = np.linspace(1.0, 2.0, 3)
        f["phiaxis"] = np.linspace(0.0, 1.0, 3)
        f["zaxis"] = np.linspace(-0.5, 0.5, 3)
        f["B_R"] = np.zeros((3, 3, 3))
        f["B_Z"] = np.zeros((3, 3, 3))
        f["R_lines"] = (1.25 + dR_dphi * phi_steps)[:, None]
        f["PHI_lines"] = phi_steps[:, None]
        f["Z_lines"] = (0.1 + dZ_dphi * phi_steps)[:, None]
        f["npoinc"] = np.asarray([2], dtype=np.int32)

    def fake_loader(*_args, **_kwargs):
        metadata = [trace_fieldlines_simsopt_candidate.MakegridCoilMetadata(1, "G1", 1.0, 2)]
        return _FakeCylindricalField(dR_dphi=dR_dphi, dZ_dphi=dZ_dphi), metadata, {1: 1.0}, np.asarray([1.0])

    monkeypatch.setattr(trace_fieldlines_simsopt_candidate, "load_scaled_simsopt_field", fake_loader)
    reference, candidate, metadata = trace_fieldlines_simsopt_candidate.trace_simsopt_candidate(
        h5_path,
        tmp_path / "coils.synthetic",
        tmp_path / "input.synthetic",
        lines=[0],
        nsections=4,
        substeps=1,
        order=3,
        ppp=4,
    )

    assert metadata["method"].startswith("RK4 on Simsopt BiotSavart")
    assert metadata["current_mode"] == "scaled"
    np.testing.assert_allclose(candidate["poincare_rphiz"], reference["poincare_rphiz"], atol=1e-14)
    labeled = fieldline_compare.compare_samples(reference, candidate, point_mode="labeled")
    assert labeled["poincare_rphiz_labeled_point_max_distance"] < 1e-14


def test_trace_fieldlines_simsopt_candidate_validates_fieldline_inputs(tmp_path, monkeypatch):
    h5py = pytest.importorskip("h5py")
    h5_path = tmp_path / "fieldlines_bad_simsopt.h5"
    with h5py.File(h5_path, "w") as f:
        f["raxis"] = np.linspace(1.0, 2.0, 3)
        f["phiaxis"] = np.linspace(0.0, 1.0, 3)
        f["zaxis"] = np.linspace(-0.5, 0.5, 3)
        f["B_R"] = np.zeros((3, 3, 3))
        f["B_Z"] = np.zeros((3, 3, 3))
        f["R_lines"] = np.ones((3, 1))
        f["PHI_lines"] = np.asarray([[0.0], [-1.0], [-1.0]])
        f["Z_lines"] = np.zeros((3, 1))
        f["npoinc"] = np.asarray([1], dtype=np.int32)

    def fake_loader(*_args, **_kwargs):
        metadata = [trace_fieldlines_simsopt_candidate.MakegridCoilMetadata(1, "G1", 1.0, 2)]
        return _FakeCylindricalField(), metadata, {1: 1.0}, np.asarray([1.0])

    monkeypatch.setattr(trace_fieldlines_simsopt_candidate, "load_scaled_simsopt_field", fake_loader)
    base_kwargs = dict(
        h5_path=h5_path,
        coils_path=tmp_path / "coils.synthetic",
        input_path=tmp_path / "input.synthetic",
        lines=[0],
        nsections=2,
        substeps=1,
        order=3,
        ppp=4,
    )

    with pytest.raises(ValueError, match="nsections must be positive"):
        trace_fieldlines_simsopt_candidate.trace_simsopt_candidate(**(base_kwargs | {"nsections": 0}))
    with pytest.raises(ValueError, match="substeps must be positive"):
        trace_fieldlines_simsopt_candidate.trace_simsopt_candidate(**(base_kwargs | {"substeps": 0}))
    with pytest.raises(ValueError, match="terminated before requested section count"):
        trace_fieldlines_simsopt_candidate.trace_simsopt_candidate(**base_kwargs)


def test_trace_fieldlines_simsopt_candidate_main_writes_artifacts(tmp_path, monkeypatch):
    h5py = pytest.importorskip("h5py")
    h5_path = tmp_path / "fieldlines_cli_simsopt.h5"
    with h5py.File(h5_path, "w") as f:
        f["raxis"] = np.linspace(1.0, 2.0, 3)
        f["phiaxis"] = np.linspace(0.0, 1.0, 3)
        f["zaxis"] = np.linspace(-0.5, 0.5, 3)
        f["B_R"] = np.zeros((3, 3, 3))
        f["B_Z"] = np.zeros((3, 3, 3))
        f["R_lines"] = np.ones((1, 1))
        f["PHI_lines"] = np.zeros((1, 1))
        f["Z_lines"] = np.zeros((1, 1))
        f["npoinc"] = np.asarray([1], dtype=np.int32)

    def fake_trace(*_args, **kwargs):
        samples = {
            "poincare_rphiz": np.asarray([[1.0, 0.0, 0.0]]),
            "line_id": np.asarray(kwargs["lines"], dtype=float),
            "section_phi": np.asarray([0.0]),
            "connection_lengths": np.asarray([0.0]),
        }
        return samples, samples, {"method": "fake", "lines": kwargs["lines"]}

    monkeypatch.setattr(trace_fieldlines_simsopt_candidate, "trace_simsopt_candidate", fake_trace)
    reference_out = tmp_path / "reference.npz"
    candidate_out = tmp_path / "candidate.npz"
    report_out = tmp_path / "report.json"

    rc = trace_fieldlines_simsopt_candidate.main(
        [
            str(h5_path),
            str(tmp_path / "coils.synthetic"),
            str(tmp_path / "input.synthetic"),
            "--reference-out",
            str(reference_out),
            "--candidate-out",
            str(candidate_out),
            "--report",
            str(report_out),
        ]
    )
    report = json.loads(report_out.read_text())

    assert rc == 0
    assert reference_out.exists()
    assert candidate_out.exists()
    assert report["status"] == "completed"
    assert report["method"] == "fake"


def test_load_fieldline_samples_rejects_unknown_format(tmp_path):
    with pytest.raises(ValueError, match="Unsupported sample format"):
        fieldline_compare.load_fieldline_samples(tmp_path / "samples.dat")


def test_zero_reference_norm_metrics_fall_back_to_absolute_errors():
    reference = {
        "poincare_xyz": np.zeros((2, 3)),
        "connection_lengths": np.zeros(2),
    }
    candidate = {
        "poincare_xyz": np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
        "connection_lengths": np.asarray([3.0, 4.0]),
    }

    metrics = fieldline_compare.compare_samples(reference, candidate, point_mode="ordered")

    assert metrics["poincare_xyz_point_relative_l2"] == np.sqrt(5.0)
    assert metrics["poincare_xyz_point_relative_rms_distance"] == np.sqrt(2.5)
    assert metrics["connection_length_relative_l2"] == 5.0
    assert metrics["connection_length_max_relative_to_ref_max"] == 4.0


def test_run_compare_skips_with_stellopt_provenance_when_inputs_are_absent(tmp_path):
    args = argparse.Namespace(
        reference=None,
        candidate=None,
        skip_if_missing=True,
        stellopt_root=tmp_path,
        point_mode="ordered",
    )

    metrics = fieldline_compare.run_compare(args)

    assert metrics["status"] == "skipped"
    assert metrics["stellopt"]["exists"] is True


def test_run_compare_writes_completed_report_with_example_metrics(tmp_path):
    reference_path = tmp_path / "reference.json"
    candidate_path = tmp_path / "candidate.json"
    out_path = tmp_path / "metrics.json"
    reference_path.write_text(json.dumps({key: np.asarray(value).tolist() for key, value in _samples().items()}))
    candidate_path.write_text(json.dumps({key: np.asarray(value).tolist() for key, value in _samples().items()}))

    rc = fieldline_compare.main(
        [
            "--reference",
            str(reference_path),
            "--candidate",
            str(candidate_path),
            "--reference-label",
            "external/reference.json",
            "--candidate-label",
            "candidate/jax.json",
            "--out",
            str(out_path),
            "--max-point-relative-l2",
            "1e-14",
            "--max-connection-relative-l2",
            "1e-14",
            "--max-hit-mismatch-fraction",
            "0",
            "--no-skip-if-missing",
        ]
    )
    metrics = json.loads(out_path.read_text())

    assert rc == 0
    assert metrics["status"] == "completed"
    assert metrics["reference"] == "external/reference.json"
    assert metrics["candidate"] == "candidate/jax.json"
    assert metrics["passed_thresholds"] is True
    assert metrics["connection_length_relative_l2"] == 0.0
