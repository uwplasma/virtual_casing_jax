from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from virtual_casing_jax import SurfType
from virtual_casing_jax import testdata

from dump_io import load_dump  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"
pytestmark = pytest.mark.reference


def _assert_close(got, ref, rtol=1e-10, atol=1e-12, atol_zero=None):
    """Assert closeness with an optional absolute tolerance for expected-zero entries.

    Components whose reference value is (numerically) zero pick up platform- and
    version-dependent floating-point noise from the quadrature, so a relative
    tolerance is meaningless for them.  When ``atol_zero`` is given, entries with
    ``|ref| <= atol_zero`` are compared with that absolute tolerance only, while
    all remaining entries keep the strict relative/absolute tolerances.
    """
    got = np.asarray(got)
    ref = np.asarray(ref)
    if atol_zero is None:
        np.testing.assert_allclose(got, ref, rtol=rtol, atol=atol)
        return
    zero_mask = np.abs(ref) <= atol_zero
    np.testing.assert_allclose(got[zero_mask], ref[zero_mask], rtol=0.0, atol=atol_zero)
    np.testing.assert_allclose(got[~zero_mask], ref[~zero_mask], rtol=rtol, atol=atol)


def test_testdata_surface_coordinates_axisym():
    X_ref = load_dump(DATA_DIR / "case_vc_setup_X")
    X = testdata.surface_coordinates(1, False, 6, 5, SurfType.AxisymNarrow)
    _assert_close(X, X_ref, rtol=1e-12, atol=1e-12)


def test_testdata_surface_coordinates_w7x():
    X_ref = load_dump(DATA_DIR / "case_vc_w7x_setup_X")
    X = testdata.surface_coordinates(5, True, 8, 6, SurfType.W7X_)
    _assert_close(X, X_ref, rtol=1e-10, atol=1e-10)


def test_testdata_magnetic_field_data_axisym():
    X = load_dump(DATA_DIR / "case_vc_setup_X")
    Bext_ref = load_dump(DATA_DIR / "case_testdata_axisym_Bext")
    Bint_ref = load_dump(DATA_DIR / "case_testdata_axisym_Bint")
    Bext, Bint = testdata.magnetic_field_data(1, False, 6, 5, X, 4, 4)
    _assert_close(Bext, Bext_ref, rtol=1e-9, atol=5e-10, atol_zero=2e-9)
    _assert_close(Bint, Bint_ref, rtol=1e-9, atol=5e-10, atol_zero=2e-9)


def test_testdata_magnetic_field_grad_data_axisym():
    X = load_dump(DATA_DIR / "case_vc_setup_X")
    GradBext_ref = load_dump(DATA_DIR / "case_testdata_axisym_GradBext")
    GradBint_ref = load_dump(DATA_DIR / "case_testdata_axisym_GradBint")
    GradBext, GradBint = testdata.magnetic_field_grad_data(1, False, 6, 5, X, 4, 4)
    _assert_close(GradBext, GradBext_ref, rtol=1e-8, atol=5e-10, atol_zero=2e-9)
    _assert_close(GradBint, GradBint_ref, rtol=1e-8, atol=5e-10, atol_zero=2e-9)
