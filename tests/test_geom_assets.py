import numpy as np
import pytest

from virtual_casing_jax.testdata import SurfType, surface_coordinates


@pytest.mark.parametrize("surf_type", [SurfType.Quas3, SurfType.LHD, SurfType.W7X])
def test_geom_assets_load(surf_type):
    X = surface_coordinates(nfp=1, half_period=False, nt=8, npol=6, surf_type=surf_type)
    assert X.shape == (3, 8, 6)
    assert np.all(np.isfinite(np.asarray(X)))
