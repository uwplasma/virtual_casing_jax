from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "external" / "fieldline_compare.py"
spec = importlib.util.spec_from_file_location("fieldline_compare", BENCHMARK)
fieldline_compare = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(fieldline_compare)


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
