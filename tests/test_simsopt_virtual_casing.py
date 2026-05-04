import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from virtual_casing_jax import VirtualCasing
from virtual_casing_jax.simsopt_virtual_casing import _3d_from_soa, _soa_from_3d

try:  # optional deps
    from mpi4py import MPI  # noqa: F401
except Exception:  # pragma: no cover - optional
    MPI = None

try:  # optional deps
    import vmec as vmec_mod  # noqa: F401
except Exception:  # pragma: no cover - optional
    vmec_mod = None

try:  # optional deps
    import matplotlib  # noqa: F401
except Exception:  # pragma: no cover - optional
    matplotlib = None

try:
    import simsopt  # noqa: F401
    from simsopt.mhd.vmec import Vmec
    HAVE_SIMSOPT = True
except Exception:  # pragma: no cover - optional
    simsopt = None
    Vmec = None
    HAVE_SIMSOPT = False

LOCAL_TEST_DIR = Path(__file__).resolve().parent / "test_files"
SIMSOPT_TEST_DIR = None
if HAVE_SIMSOPT:
    try:
        from simsopt.tests.mhd import TEST_DIR as SIMSOPT_TEST_DIR  # type: ignore
    except Exception:  # pragma: no cover - optional
        pass
    if SIMSOPT_TEST_DIR is None:
        pkg_root = Path(simsopt.__file__).resolve()
        for parent in pkg_root.parents:
            candidate = parent / "tests" / "test_files"
            if candidate.is_dir():
                SIMSOPT_TEST_DIR = candidate
                break


REQUIRES_SIMSOPT = pytest.mark.skipif(
    (not HAVE_SIMSOPT) or (MPI is None) or (vmec_mod is None),
    reason="Need simsopt, mpi4py, and vmec python packages",
)


def _require_test_files(*names: str):
    if LOCAL_TEST_DIR.is_dir():
        base_dir = LOCAL_TEST_DIR
    elif SIMSOPT_TEST_DIR is not None:
        base_dir = Path(SIMSOPT_TEST_DIR)
    else:
        pytest.skip("simsopt test_files not available")
    paths = []
    for name in names:
        path = Path(base_dir) / name
        if not path.exists():
            pytest.skip(f"Missing simsopt test file: {path}")
        paths.append(path)
    return paths if len(paths) > 1 else paths[0]


def _legacy_vmec_from_input_or_skip(input_file):
    try:
        return Vmec(str(input_file))
    except Exception as err:
        pytest.skip(f"legacy VMEC input initialization is unavailable: {err}")


VARIABLES = [
    "src_nphi",
    "src_ntheta",
    "src_phi",
    "src_theta",
    "trgt_nphi",
    "trgt_ntheta",
    "trgt_phi",
    "trgt_theta",
    "gamma",
    "unit_normal",
    "B_total",
    "B_external",
    "B_external_normal",
]


def _small_virtual_casing_result():
    vc = VirtualCasing()
    vc.src_ntheta = 2
    vc.src_nphi = 3
    vc.src_theta = np.array([0.0, 0.5])
    vc.src_phi = np.array([0.0, 1.0 / 3.0, 2.0 / 3.0])
    vc.trgt_ntheta = 2
    vc.trgt_nphi = 3
    vc.trgt_theta = np.array([0.125, 0.625])
    vc.trgt_phi = np.array([0.05, 0.35, 0.65])
    vc.nfp = 2
    vc.gamma = np.arange(18.0).reshape((3, 2, 3))
    vc.unit_normal = np.ones((3, 2, 3))
    vc.B_total = 0.1 * np.arange(18.0).reshape((3, 2, 3))
    vc.B_external = 0.2 * np.arange(18.0).reshape((3, 2, 3))
    vc.B_external_normal = np.array([[1.0, -1.0], [0.5, -0.5], [0.25, -0.25]])
    vc.B_external_normal_extended = np.vstack(
        [vc.B_external_normal, -vc.B_external_normal, vc.B_external_normal, -vc.B_external_normal]
    )
    return vc


def test_simsopt_layout_conversions_round_trip_vector_components():
    aos = np.arange(24.0).reshape((4, 2, 3))

    soa = _soa_from_3d(aos)
    restored = _3d_from_soa(soa)

    assert soa.shape == (3, 4, 2)
    np.testing.assert_allclose(restored, aos)
    np.testing.assert_allclose(soa[:, 1, 0], aos[1, 0, :])


def test_virtual_casing_save_load_roundtrip_without_simsopt_runtime(tmp_path):
    vc1 = _small_virtual_casing_result()
    path = tmp_path / "vcasing.nc"

    vc1.save(path)
    vc2 = VirtualCasing.load(path)

    for variable in VARIABLES + ["B_external_normal_extended"]:
        np.testing.assert_allclose(getattr(vc1, variable), getattr(vc2, variable))
    assert int(vc2.nfp) == vc1.nfp
    assert int(vc2.src_nphi) == vc1.src_nphi
    assert int(vc2.trgt_ntheta) == vc1.trgt_ntheta


@pytest.mark.skipif(matplotlib is None, reason="Need matplotlib")
def test_virtual_casing_plot_uses_existing_axis_and_returns_it():
    vc = _small_virtual_casing_result()
    import matplotlib.pyplot as plt

    _, ax0 = plt.subplots()
    ax1 = vc.plot(ax=ax0, show=False)

    assert ax1 is ax0
    plt.close("all")


@pytest.mark.skipif(matplotlib is None, reason="Need matplotlib")
def test_virtual_casing_plot_allocates_axis_and_honors_show(monkeypatch):
    vc = _small_virtual_casing_result()
    import matplotlib.pyplot as plt

    calls = []
    monkeypatch.setattr(plt, "show", lambda: calls.append("show"))

    ax = vc.plot(show=True)

    assert ax.get_title() == "B_external_normal [Tesla]"
    assert calls == ["show"]
    plt.close("all")


def test_virtual_casing_plot_is_testable_without_matplotlib_dependency(monkeypatch):
    vc = _small_virtual_casing_result()
    pyplot = types.ModuleType("matplotlib.pyplot")
    matplotlib_mod = types.ModuleType("matplotlib")

    class FakeFigure:
        def __init__(self):
            self.colorbars = 0
            self.tight_layouts = 0

        def colorbar(self, contours):
            assert contours == "contours"
            self.colorbars += 1

        def tight_layout(self):
            self.tight_layouts += 1

    class FakeAxis:
        def __init__(self):
            self.xlabel = None
            self.ylabel = None
            self.title = None

        def contourf(self, *args):
            assert len(args) == 4
            return "contours"

        def set_xlabel(self, label):
            self.xlabel = label

        def set_ylabel(self, label):
            self.ylabel = label

        def set_title(self, title):
            self.title = title

    figures = []
    show_calls = []

    def subplots():
        fig = FakeFigure()
        ax = FakeAxis()
        figures.append(fig)
        return fig, ax

    def gcf():
        if not figures:
            figures.append(FakeFigure())
        return figures[-1]

    pyplot.subplots = subplots
    pyplot.gcf = gcf
    pyplot.show = lambda: show_calls.append("show")
    matplotlib_mod.pyplot = pyplot
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib_mod)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)

    allocated_ax = vc.plot(show=True)
    existing_ax = FakeAxis()
    returned_ax = vc.plot(ax=existing_ax, show=False)

    assert allocated_ax.title == "B_external_normal [Tesla]"
    assert existing_ax.title == "B_external_normal [Tesla]"
    assert returned_ax is existing_ax
    assert show_calls == ["show"]
    assert sum(fig.colorbars for fig in figures) == 4
    assert sum(fig.tight_layouts for fig in figures) == 4


def test_from_vmec_reports_missing_simsopt_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "simsopt.mhd.vmec_diagnostics", None)

    with pytest.raises(RuntimeError, match="simsopt is required"):
        VirtualCasing.from_vmec("dummy-wout.nc", src_nphi=4, filename=None)


@pytest.mark.large
@REQUIRES_SIMSOPT
def test_input_file_initialization():
    input_file = _require_test_files("input.li383_low_res")

    vmec_input = _legacy_vmec_from_input_or_skip(input_file)
    VirtualCasing.from_vmec(vmec_input, src_nphi=8, filename=None)


@pytest.mark.large
@REQUIRES_SIMSOPT
def test_wout_initializations():
    wout_file = _require_test_files(
        "wout_20220102-01-053-003_QH_nfp4_aspect6p5_beta0p05_iteratedWithSfincs_reference.nc"
    )

    VirtualCasing.from_vmec(str(wout_file), src_nphi=9, src_ntheta=10, filename=None)

    vmec = Vmec(str(wout_file))
    VirtualCasing.from_vmec(vmec, src_nphi=10, filename=None)


@pytest.mark.large
@REQUIRES_SIMSOPT
@pytest.mark.parametrize("use_stellsym", [True, False])
def test_bnorm_benchmark(use_stellsym):
    wout_file, bnorm_file = _require_test_files(
        "wout_20220102-01-053-003_QH_nfp4_aspect6p5_beta0p05_iteratedWithSfincs_reference.nc",
        "bnorm.20220102-01-053-003_QH_nfp4_aspect6p5_beta0p05_iteratedWithSfincs",
    )

    vmec = Vmec(str(wout_file))
    nphi_fac = 1 if use_stellsym else 2
    vc = VirtualCasing.from_vmec(
        vmec,
        src_nphi=25 * nphi_fac,
        trgt_nphi=32,
        trgt_ntheta=32,
        use_stellsym=use_stellsym,
        filename=None,
    )

    nfp = vmec.wout.nfp
    theta, phi = np.meshgrid(2 * np.pi * vc.trgt_theta, 2 * np.pi * vc.trgt_phi)
    B_external_normal_bnorm = np.zeros((vc.trgt_nphi, vc.trgt_ntheta))

    with open(bnorm_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        splitline = line.split()
        if len(splitline) != 3:
            continue
        m = int(splitline[0])
        n = int(splitline[1])
        amplitude = float(splitline[2])
        B_external_normal_bnorm += amplitude * np.sin(m * theta + n * nfp * phi)

    curpol = (2 * np.pi / nfp) * (1.5 * vmec.wout.bsubvmnc[0, -1] - 0.5 * vmec.wout.bsubvmnc[0, -2])
    B_external_normal_bnorm *= curpol

    np.testing.assert_allclose(B_external_normal_bnorm, vc.B_external_normal, atol=0.0061)


@pytest.mark.large
@REQUIRES_SIMSOPT
def test_save_load(tmp_path):
    wout_file = _require_test_files(
        "wout_20220102-01-053-003_QH_nfp4_aspect6p5_beta0p05_iteratedWithSfincs_reference.nc"
    )
    outfile = tmp_path / "vcasing.nc"

    vc1 = VirtualCasing.from_vmec(
        str(wout_file),
        src_nphi=11,
        src_ntheta=12,
        trgt_nphi=13,
        trgt_ntheta=11,
        filename=str(outfile),
    )
    vc2 = VirtualCasing.load(str(outfile))

    for variable in VARIABLES:
        v1 = getattr(vc1, variable)
        v2 = getattr(vc2, variable)
        np.testing.assert_allclose(v1, v2)


@pytest.mark.large
@REQUIRES_SIMSOPT
@pytest.mark.skipif(matplotlib is None, reason="Need matplotlib")
def test_plot(tmp_path):
    wout_file = _require_test_files(
        "wout_20220102-01-053-003_QH_nfp4_aspect6p5_beta0p05_iteratedWithSfincs_reference.nc"
    )
    vc = VirtualCasing.from_vmec(str(wout_file), src_nphi=8, src_ntheta=9, filename=None)
    vc.plot(show=False)

    import matplotlib.pyplot as plt

    _, ax0 = plt.subplots()
    ax1 = vc.plot(ax=ax0, show=False)
    assert ax1 is ax0


@pytest.mark.large
@REQUIRES_SIMSOPT
def test_vacuum():
    wout_file = _require_test_files("wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc")
    vmec = Vmec(str(wout_file))
    vc = VirtualCasing.from_vmec(vmec, src_nphi=32, filename=None)

    np.testing.assert_allclose(vc.B_external, vc.B_total, atol=0.02)
    np.testing.assert_allclose(vc.B_external_normal, 0, atol=0.001)


@pytest.mark.large
@REQUIRES_SIMSOPT
def test_stellsym():
    wout_file = _require_test_files(
        "wout_20220102-01-053-003_QH_nfp4_aspect6p5_beta0p05_iteratedWithSfincs_reference.nc"
    )
    vmec = Vmec(str(wout_file))
    src_nphi = 48
    src_ntheta = 12
    vc = VirtualCasing.from_vmec(vmec, src_nphi=src_nphi, src_ntheta=src_ntheta, use_stellsym=False, filename=None)

    Bn_flipped = -np.rot90(np.rot90(vc.B_external_normal))
    Bn_flipped = np.roll(np.roll(Bn_flipped, 1, axis=0), 1, axis=1)
    np.testing.assert_allclose(vc.B_external_normal, Bn_flipped, atol=2e-4)

    vc_ss = VirtualCasing.from_vmec(
        vmec,
        src_nphi=src_nphi // 4,
        src_ntheta=src_ntheta,
        use_stellsym=True,
        filename=None,
    )
    idxs = list(range(1, vc.trgt_nphi // 2, 2))
    np.testing.assert_allclose(vc.trgt_phi[idxs], vc_ss.trgt_phi)
    np.testing.assert_allclose(vc.B_external_normal[idxs, :], vc_ss.B_external_normal, atol=1e-5)
