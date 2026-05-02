from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "external" / "extender_compare.py"
spec = importlib.util.spec_from_file_location("extender_compare", BENCHMARK)
extender_compare = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(extender_compare)


def _samples(delta: float = 0.0):
    xyz = np.asarray([[1.0, 0.0, 0.0], [1.2, 0.1, -0.2]])
    coils = np.asarray([[0.0, 1.0, 0.0], [0.1, 0.8, 0.0]])
    plasma = np.asarray([[0.2, 0.0, 0.3], [0.1, 0.1, 0.4]])
    total = coils + plasma + delta
    return {
        "xyz": xyz,
        "B_coils_xyz": coils,
        "B_plasma_xyz": plasma,
        "B_total_xyz": total,
    }


def _boundary_samples(delta_normal: float = 0.0):
    xyz = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    normal = xyz.copy()
    coils = np.asarray(
        [
            [0.15, 1.0, 0.05],
            [-0.9, 0.12, 0.02],
            [-0.14, -1.1, 0.03],
            [0.8, -0.11, 0.02],
        ]
    )
    plasma = np.asarray(
        [
            [-0.15 + delta_normal, 0.2, 0.01],
            [0.1, -0.12 + delta_normal, 0.04],
            [0.14 - delta_normal, -0.1, -0.02],
            [-0.05, 0.11 - delta_normal, 0.01],
        ]
    )
    return {
        "xyz": xyz,
        "normal_xyz": normal,
        "B_coils_xyz": coils,
        "B_plasma_xyz": plasma,
        "B_total_xyz": coils + plasma,
    }


def test_compare_samples_reports_vector_errors_and_closure():
    reference = _samples()
    candidate = _samples()
    candidate["B_plasma_xyz"] = candidate["B_plasma_xyz"] + 1e-3
    candidate["B_total_xyz"] = candidate["B_coils_xyz"] + candidate["B_plasma_xyz"]

    metrics = extender_compare.compare_samples(reference, candidate, point_atol=0.0, point_rtol=0.0)

    assert metrics["n_points"] == 2
    assert metrics["fields_compared"] == ["B_total_xyz", "B_plasma_xyz", "B_coils_xyz"]
    assert metrics["B_plasma_xyz_relative_l2"] > 0.0
    assert metrics["reference_closure_relative_l2"] == 0.0
    assert metrics["candidate_closure_relative_l2"] < 1e-15


def test_compare_samples_reports_normal_projection_and_lcfs_cancellation():
    reference = _boundary_samples()
    candidate = _boundary_samples(delta_normal=1e-3)

    metrics = extender_compare.compare_samples(reference, candidate, point_atol=0.0, point_rtol=0.0)

    assert metrics["normal_max_abs_delta"] == 0.0
    assert metrics["reference_total_normal_relative_rms"] < 1e-15
    assert metrics["candidate_total_normal_relative_rms"] > 0.0
    assert metrics["B_plasma_xyz_normal_relative_l2"] > 0.0
    assert metrics["B_coils_xyz_normal_relative_l2"] == 0.0


def test_threshold_failures_cover_field_error_and_decomposition_closure():
    metrics = extender_compare.compare_samples(_samples(), _samples(delta=1e-2), point_atol=0.0, point_rtol=0.0)
    args = argparse.Namespace(max_relative_l2=1e-6, max_abs=1e-6, max_closure_relative_l2=1e-6)

    failures = extender_compare._threshold_failures(metrics, args)

    assert any("B_total_xyz_relative_l2" in failure for failure in failures)
    assert any("candidate_closure_relative_l2" in failure for failure in failures)


def test_threshold_failures_cover_normal_projection_and_cancellation():
    metrics = extender_compare.compare_samples(
        _boundary_samples(),
        _boundary_samples(delta_normal=1e-2),
        point_atol=0.0,
        point_rtol=0.0,
    )
    args = argparse.Namespace(
        max_relative_l2=1.0,
        max_abs=1.0,
        max_closure_relative_l2=1e-12,
        max_normal_relative_l2=1e-6,
        max_normal_abs=1e-6,
        max_total_normal_relative_rms=1e-6,
    )

    failures = extender_compare._threshold_failures(metrics, args)

    assert any("B_total_xyz_normal_relative_l2" in failure for failure in failures)
    assert any("B_plasma_xyz_normal_max_abs" in failure for failure in failures)
    assert any("candidate_total_normal_relative_rms" in failure for failure in failures)


def test_load_field_samples_from_json_npz_and_csv_aliases(tmp_path):
    json_path = tmp_path / "samples.json"
    json_path.write_text(
        json.dumps(
            {
                "points_xyz": [[1.0, 0.0, 0.0]],
                "unit_normal_xyz": [[1.0, 0.0, 0.0]],
                "B_xyz": [[0.0, 1.0, 0.0]],
                "coil_xyz": [[0.0, 0.5, 0.0]],
            }
        )
    )
    loaded_json = extender_compare.load_field_samples(json_path)
    np.testing.assert_allclose(loaded_json["xyz"], [[1.0, 0.0, 0.0]])
    np.testing.assert_allclose(loaded_json["normal_xyz"], [[1.0, 0.0, 0.0]])
    np.testing.assert_allclose(loaded_json["B_total_xyz"], [[0.0, 1.0, 0.0]])
    np.testing.assert_allclose(loaded_json["B_coils_xyz"], [[0.0, 0.5, 0.0]])

    npz_path = tmp_path / "samples.npz"
    np.savez(
        npz_path,
        target_xyz=np.asarray([[1.0, 0.0, 0.0]]),
        n_xyz=np.asarray([[1.0, 0.0, 0.0]]),
        B_coil=np.asarray([[0.0, 0.5, 0.0]]),
    )
    loaded_npz = extender_compare.load_field_samples(npz_path)
    np.testing.assert_allclose(loaded_npz["xyz"], [[1.0, 0.0, 0.0]])
    np.testing.assert_allclose(loaded_npz["normal_xyz"], [[1.0, 0.0, 0.0]])
    np.testing.assert_allclose(loaded_npz["B_coils_xyz"], [[0.0, 0.5, 0.0]])

    csv_path = tmp_path / "samples.csv"
    csv_path.write_text(
        "x,y,z,nx,ny,nz,bx,by,bz,b_plasma_x,b_plasma_y,b_plasma_z\n"
        "1,0,0,1,0,0,0,1,0,0,0.5,0\n"
    )
    loaded_csv = extender_compare.load_field_samples(csv_path)
    np.testing.assert_allclose(loaded_csv["normal_xyz"], [[1.0, 0.0, 0.0]])
    np.testing.assert_allclose(loaded_csv["B_total_xyz"], [[0.0, 1.0, 0.0]])
    np.testing.assert_allclose(loaded_csv["B_plasma_xyz"], [[0.0, 0.5, 0.0]])


def test_run_compare_skips_with_stellopt_provenance_when_inputs_are_absent(tmp_path):
    args = argparse.Namespace(
        reference=None,
        candidate=None,
        skip_if_missing=True,
        stellopt_root=tmp_path,
        point_atol=1e-12,
        point_rtol=1e-12,
        max_relative_l2=1e-4,
        max_abs=1e-6,
        max_closure_relative_l2=1e-10,
    )

    metrics = extender_compare.run_compare(args)

    assert metrics["status"] == "skipped"
    assert metrics["stellopt"]["exists"] is True


def test_run_compare_writes_completed_report_with_boundary_normal_metrics(tmp_path):
    reference = _boundary_samples()
    candidate = _boundary_samples()
    reference_path = tmp_path / "reference.json"
    candidate_path = tmp_path / "candidate.json"
    out_path = tmp_path / "metrics.json"
    reference_path.write_text(json.dumps({key: value.tolist() for key, value in reference.items()}))
    candidate_path.write_text(json.dumps({key: value.tolist() for key, value in candidate.items()}))

    rc = extender_compare.main(
        [
            "--reference",
            str(reference_path),
            "--candidate",
            str(candidate_path),
            "--out",
            str(out_path),
            "--max-normal-relative-l2",
            "1e-14",
            "--max-normal-abs",
            "1e-14",
            "--max-total-normal-relative-rms",
            "1e-14",
            "--no-skip-if-missing",
        ]
    )
    metrics = json.loads(out_path.read_text())

    assert rc == 0
    assert metrics["status"] == "completed"
    assert metrics["passed_thresholds"] is True
    assert metrics["candidate_total_normal_relative_rms"] < 1e-15
