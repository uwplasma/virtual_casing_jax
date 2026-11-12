"""The tutorial examples run as written and meet their own accuracy assertions."""

import runpy
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.parametrize("name", ["exterior_field_tutorial", "mirror_field"])
def test_example_runs(name):
    runpy.run_path(str(EXAMPLES / f"{name}.py"), run_name="__main__")
