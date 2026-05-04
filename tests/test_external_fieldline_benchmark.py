from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

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
    assert metrics["passed_thresholds"] is True
    assert metrics["connection_length_relative_l2"] == 0.0
