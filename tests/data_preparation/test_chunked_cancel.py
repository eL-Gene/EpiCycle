"""Cancel-safety + progress tests for the block-parallel tile engine.

Companion to ``test_chunked.py`` (the byte-identity gate).  This file owns the
CANCEL contract of ``chunked.block_parallel_apply`` and its propagation through
``ops.execute`` — the exact runtime behavior whose absence let the prior parallel
build crash on Cancel.

Teeth: the cancel tests assert the run STOPPED EARLY (``tiles_run << total``),
not merely that it raised; and a negative control proves a plain ``Exception``
from an op IS swallowed by the pipeline (so a cancel MUST be ``BaseException``).

Pure — no napari / Qt.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from packages.epicycle_preprocessing import chunked, ops
from packages.epicycle_preprocessing.exceptions import PreprocessingCancelled
from packages.epicycle_preprocessing.recipe import IntensityStep, Recipe


def _vol(shape, seed=0):
    return np.random.RandomState(seed).rand(*shape).astype(np.float32)


# --------------------------------------------------------------------------- #
# PreprocessingCancelled is BaseException (structurally un-swallowable)
# --------------------------------------------------------------------------- #

def test_cancel_is_baseexception_not_exception():
    assert issubclass(PreprocessingCancelled, BaseException)
    assert not issubclass(PreprocessingCancelled, Exception)


# --------------------------------------------------------------------------- #
# block_parallel_apply: cancel stops early + raises (WITH TEETH)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("workers", [1, 4])
def test_cancel_stops_early_and_raises(workers):
    """A ``should_cancel`` that flips after a few tiles → the run raises
    PreprocessingCancelled AND ran far fewer than all tiles (it actually stopped,
    it did not drain every tile then raise — the B2 trap)."""
    vol = _vol((48, 160, 160), seed=3)
    halo = ops._halo_gaussian(1.5)
    # force many small tiles so "stopped early" is unambiguous
    # (48,160,160)/block 16 = 3*10*10 = 300 tiles
    n_tiles = 3 * 10 * 10
    ran = {"n": 0}

    def op(a):
        ran["n"] += 1
        return ops.gaussian(a, sigma=1.5)

    def should_cancel():
        return ran["n"] >= 3

    with pytest.raises(PreprocessingCancelled):
        chunked.block_parallel_apply(
            op, vol, halo, block=16, max_workers=workers, should_cancel=should_cancel
        )
    assert ran["n"] < n_tiles // 4, (
        f"cancel must stop early: ran {ran['n']} of {n_tiles} tiles"
    )


def test_no_cancel_runs_all_tiles_and_is_byte_identical():
    """The negative control for the cancel path: with should_cancel=None the wave
    loop still runs EVERY tile and the result is byte-identical to whole-array."""
    vol = _vol((48, 160, 160), seed=3)
    halo = ops._halo_gaussian(1.5)
    n_tiles = 3 * 10 * 10
    ran = {"n": 0}

    def op(a):
        ran["n"] += 1
        return ops.gaussian(a, sigma=1.5)

    got = chunked.block_parallel_apply(op, vol, halo, block=16, max_workers=4)
    assert ran["n"] == n_tiles, f"expected all {n_tiles} tiles, ran {ran['n']}"
    assert np.array_equal(got, ops.gaussian(vol, sigma=1.5))


def test_cancel_before_any_tile_short_circuits():
    """A should_cancel already True at entry raises before any tile runs."""
    vol = _vol((48, 160, 160), seed=3)
    ran = {"n": 0}

    def op(a):
        ran["n"] += 1
        return a

    with pytest.raises(PreprocessingCancelled):
        chunked.block_parallel_apply(
            op, vol, ops._halo_gaussian(1.5), block=16, should_cancel=lambda: True
        )
    assert ran["n"] == 0


# --------------------------------------------------------------------------- #
# progress callback: monotonic (done, total), reaches total on completion
# --------------------------------------------------------------------------- #

def test_on_progress_reports_monotonic_done_total():
    vol = _vol((48, 160, 160), seed=3)
    events = []
    chunked.block_parallel_apply(
        lambda a: ops.gaussian(a, sigma=1.5),
        vol, ops._halo_gaussian(1.5), block=16, max_workers=4,
        on_progress=lambda done, total: events.append((done, total)),
    )
    assert events, "on_progress must be called at least once"
    totals = {t for _, t in events}
    assert len(totals) == 1, "total must be constant"
    dones = [d for d, _ in events]
    assert dones == sorted(dones), "done must be monotonic non-decreasing"
    assert dones[-1] == events[-1][1], "final done must equal total"


# --------------------------------------------------------------------------- #
# propagation through ops.execute (past the except-Exception guards)
# --------------------------------------------------------------------------- #

def test_cancel_propagates_out_of_execute():
    """A cancel raised inside a chunked op propagates through
    _run_intensity_ops' + execute's `except Exception` guards out of execute."""
    # above the small-array gate + wider than one adaptive block so gaussian
    # genuinely tiles (the whole-array short-circuit only checks cancel up front).
    chans = [_vol((24, 300, 300), seed=10)]   # Y/X > the 256 block floor
    assert chans[0].size >= chunked._SMALL_ARRAY_VOXELS
    recipe = Recipe(
        channels=[[IntensityStep("gaussian", {"sigma": 1.5})]], geometry=None
    )
    ran = {"n": 0}
    _real = ops.gaussian

    def counting_gaussian(a, **kw):
        ran["n"] += 1
        return _real(a, **kw)

    # cancel after a few tiles
    def should_cancel():
        return ran["n"] >= 2

    # patch the registry's bound fn so the counter sees each tile call
    from packages.epicycle_preprocessing import registry
    registry.find("gaussian").fn = counting_gaussian
    try:
        with pytest.raises(PreprocessingCancelled):
            ops.execute(
                recipe, chans, (1.0, 1.0, 1.0), parallel=True,
                should_cancel=should_cancel,
            )
    finally:
        registry.find("gaussian").fn = _real


def test_plain_exception_is_swallowed_proving_baseexception_needed(monkeypatch):
    """NEGATIVE CONTROL: a plain Exception raised from an op IS swallowed by
    _run_intensity_ops (op skipped, pipeline continues) — which is exactly why a
    cancel must be BaseException-derived.  A BaseException from the same op is NOT
    swallowed."""
    from packages.epicycle_preprocessing import registry

    vol = _vol((16, 16, 16), seed=1)
    step = IntensityStep("gaussian", {"sigma": 1.0})

    def _fake_op(raiser):
        return types.SimpleNamespace(
            key="gaussian", implemented=True, kind="intensity",
            chunk_class="adaptive", halo=None, boundary="none", fn=raiser,
        )

    def boom_exc(a, **kw):
        raise ValueError("boom-exception")

    def boom_base(a, **kw):
        raise PreprocessingCancelled("boom-cancel")

    # plain Exception -> swallowed: _run_intensity_ops returns the input unchanged
    monkeypatch.setattr(registry, "find", lambda k: _fake_op(boom_exc))
    out = ops._run_intensity_ops(vol, [step])
    assert np.array_equal(out, vol.astype(np.float32)), "plain Exception must be swallowed"

    # BaseException -> propagates
    monkeypatch.setattr(registry, "find", lambda k: _fake_op(boom_base))
    with pytest.raises(PreprocessingCancelled):
        ops._run_intensity_ops(vol, [step])


# --------------------------------------------------------------------------- #
# block_workers seam: a RAM-tight width still byte-identical
# --------------------------------------------------------------------------- #

def test_execute_block_workers_is_byte_identical():
    """Threading a smaller block-pool width (the RAM lever) must not change the
    result — byte-identical to the default width and to serial."""
    chans = [_vol((56, 140, 140), seed=10)]
    recipe = Recipe(
        channels=[[IntensityStep("gaussian", {"sigma": 1.5}),
                   IntensityStep("background", {"radius": 3})]],
        geometry=None,
    )
    ser, _, _ = ops.execute(recipe, chans, (1.0, 1.0, 1.0), parallel=False)
    par1, _, _ = ops.execute(recipe, chans, (1.0, 1.0, 1.0), parallel=True, block_workers=1)
    par4, _, _ = ops.execute(recipe, chans, (1.0, 1.0, 1.0), parallel=True, block_workers=4)
    np.testing.assert_array_equal(par1[0], ser[0])
    np.testing.assert_array_equal(par4[0], ser[0])


# --------------------------------------------------------------------------- #
# max_local_tile_scratch_bytes — the RAM-sizing helper (adaptive core count)
# --------------------------------------------------------------------------- #

def test_max_local_tile_scratch_bytes_no_local_op_is_zero():
    assert ops.max_local_tile_scratch_bytes({"channels": [[]], "geometry": None}) == 0
    assert ops.max_local_tile_scratch_bytes(None) == 0
    assert ops.max_local_tile_scratch_bytes("garbage") == 0
    # a whole-array-only recipe (CLAHE) has no local op → 0
    clahe = {"channels": [[{"key": "clahe", "params": {}}]], "geometry": None}
    assert ops.max_local_tile_scratch_bytes(clahe) == 0


def test_max_local_tile_scratch_bytes_uses_largest_local_op():
    recipe = {"channels": [[
        {"key": "gaussian", "params": {"sigma": 1.5}},
        {"key": "background", "params": {"radius": 15}},
    ]], "geometry": None}
    got = ops.max_local_tile_scratch_bytes(recipe)
    halo = ops._halo_background(radius=15)          # 30 (bigger than gaussian's)
    edge = chunked._adaptive_block_edge(halo)       # 180
    tile = edge + 2 * halo                          # 240
    assert got == tile ** 3 * 4 * 3                 # (block+2halo)^3 · 4 bytes · 3×


# --------------------------------------------------------------------------- #
# execute emits op + tile progress events (the whole-volume progress feed)
# --------------------------------------------------------------------------- #

def test_execute_emits_op_and_tile_progress_events():
    events = []
    chans = [_vol((56, 160, 160), seed=1)]     # tiles (gaussian block 64 < 160)
    recipe = Recipe(
        channels=[[IntensityStep("gaussian", {"sigma": 1.5})]], geometry=None
    )
    ops.execute(recipe, chans, (1.0, 1.0, 1.0), parallel=True,
                on_progress=lambda e: events.append(e))
    op_events = [e for e in events if e["kind"] == "op"]
    tile_events = [e for e in events if e["kind"] == "tile"]
    # one normalize + one gaussian, monotonic index, total == 2
    assert [e["op"] for e in op_events] == ["normalize", "gaussian"]
    assert [e["index"] for e in op_events] == [1, 2]
    assert all(e["total"] == 2 for e in op_events)
    assert op_events[1]["tiled"] is True and op_events[0]["tiled"] is False
    # the gaussian op is tiled → tile events, the last reaching total
    assert tile_events, "expected tile events for the tiled gaussian op"
    assert tile_events[-1]["done"] == tile_events[-1]["total"]


def test_execute_no_progress_callback_is_noop():
    # on_progress None (preview / default) → no events, byte-identical result
    chans = [_vol((56, 160, 160), seed=2)]
    recipe = Recipe(
        channels=[[IntensityStep("gaussian", {"sigma": 1.5})]], geometry=None
    )
    a, _, _ = ops.execute(recipe, chans, (1.0, 1.0, 1.0), parallel=True)
    b, _, _ = ops.execute(recipe, chans, (1.0, 1.0, 1.0), parallel=False)
    np.testing.assert_array_equal(a[0], b[0])
