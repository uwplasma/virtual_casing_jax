from __future__ import annotations

from virtual_casing_jax.utils import autotune_chunk_sizes


def test_cpu_b_autotune_keeps_small_target_grid_unblocked():
    src, trg = autotune_chunk_sizes("b", nsrc=18432, ntrg=144, backend="cpu")

    assert src == 512
    assert trg is None


def test_cpu_b_autotune_blocks_larger_target_grid():
    src, trg = autotune_chunk_sizes("b", nsrc=18432, ntrg=512, backend="cpu")

    assert src == 512
    assert trg == 64


def test_cpu_b_autotune_target_env_override_wins(monkeypatch):
    monkeypatch.setenv("VCJAX_CHUNK_B_TRG", "32")

    src, trg = autotune_chunk_sizes("b", nsrc=18432, ntrg=144, backend="cpu")

    assert src == 512
    assert trg == 32
