from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np


BENCHMARK = Path(__file__).resolve().parents[1] / "benchmarks" / "external" / "simsopt_vc_compare.py"
spec = importlib.util.spec_from_file_location("simsopt_vc_compare", BENCHMARK)
simsopt_vc_compare = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(simsopt_vc_compare)


class _FakeWout:
    nfp = 2
    bsubvmnc = np.asarray([[0.0, 1.0, 3.0]])


def test_bnorm_normal_field_uses_simsopt_fourier_and_curpol_convention(tmp_path):
    bnorm = tmp_path / "bnorm.synthetic"
    bnorm.write_text("# m n amplitude\n1 1 0.25\n2 -1 -0.5\n")

    trgt_phi = np.asarray([0.0, 0.125])
    trgt_theta = np.asarray([0.0, 0.25, 0.5])
    got = simsopt_vc_compare.bnorm_normal_field(bnorm, _FakeWout(), trgt_phi, trgt_theta)

    theta, phi = np.meshgrid(2 * np.pi * trgt_theta, 2 * np.pi * trgt_phi)
    curpol = (2 * np.pi / _FakeWout.nfp) * (1.5 * 3.0 - 0.5 * 1.0)
    expected = curpol * (0.25 * np.sin(theta + 2 * phi) - 0.5 * np.sin(2 * theta - 2 * phi))

    np.testing.assert_allclose(got, expected)


def test_relative_l2_uses_reference_norm_and_handles_zero_reference():
    reference = np.asarray([3.0, 4.0])
    candidate = np.asarray([4.0, 6.0])
    assert simsopt_vc_compare._relative_l2(candidate, reference) == np.sqrt(5.0) / 5.0

    zero_reference = np.zeros(2)
    assert simsopt_vc_compare._relative_l2(candidate, zero_reference) == np.linalg.norm(candidate)


def test_threshold_failures_report_physics_metrics():
    args = argparse.Namespace(
        max_external_normal_relative_l2=1e-3,
        max_external_vector_relative_l2=1e-4,
        max_bnorm_max_abs=5e-3,
    )
    metrics = {
        "external_normal_relative_l2": 2e-3,
        "external_vector_relative_l2": 2e-4,
        "jax_bnorm_max_abs": 1e-2,
    }

    failures = simsopt_vc_compare._threshold_failures(metrics, args)

    assert len(failures) == 3
    assert failures[0].startswith("external_normal_relative_l2")
    assert failures[1].startswith("external_vector_relative_l2")
    assert failures[2].startswith("jax_bnorm_max_abs")
