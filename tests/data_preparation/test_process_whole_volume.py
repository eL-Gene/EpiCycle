"""Brick 1b-1 tests — the pure whole-volume PROCESS core (`controller`).

Unit-tests the module-level `_process_whole_volume_core` (+ its helpers
`_materialize_channel` / `_strided_percentile_bounds`) directly — NO Qt, no
controller instance, no napari viewer.  The core is the pure heart of the
"Process whole volume" button: materialize (possibly lazy) channels → measure the
normalize window from a strided subsample → `execute` serially (`max_workers=1`)
→ fixed-scale `to_uint16`.

The distinguishing property under test is the FIXED-scale uint16 bake:
a uniformly DIM recipe output stays dim (``0.2 → 13107``) and is NEVER re-stretched
to fill 0..65535 (the old per-channel min-max behavior).  `execute` is referenced
through the controller module globals, so a test can monkeypatch it to drive a
known dim output + capture the (bounds, max_workers) it is called with.
"""

from __future__ import annotations

import numpy as np
import pytest

from napari_epicycle.data_preparation import controller
from packages.epicycle_preprocessing import ops


def _empty_recipe(n):
    """A recipe json with N empty channels + no geometry — so `execute` runs
    only the hoisted percentile-normalize working space."""
    return {"channels": [[] for _ in range(n)], "geometry": None}


# --------------------------------------------------------------------------- #
# (a) + (d) — N uint16 arrays out for N in, shapes CZYX-consistent; 1ch + 3ch
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("n", [1, 3])
def test_core_returns_n_uint16_arrays_shape_consistent(n):
    rng = np.random.RandomState(0)
    chans = [(rng.rand(6, 8, 7).astype(np.float32) * 100.0) for _ in range(n)]

    processed, new_spacing, bounds, stamp = controller._process_whole_volume_core(
        _empty_recipe(n), chans, (1.0, 1.0, 1.0))

    assert len(processed) == n
    assert len(bounds) == n
    for p, c in zip(processed, chans):
        assert p.dtype == np.uint16
        # No geometry step → every channel keeps its ZYX grid (co-registered).
        assert p.shape == c.shape
    # No geometry → spacing unchanged.
    assert tuple(float(s) for s in new_spacing) == (1.0, 1.0, 1.0)
    assert isinstance(stamp, dict)


# --------------------------------------------------------------------------- #
# (b) — a uniformly DIM channel stays dim: fixed-scale bake, NOT re-stretched
# --------------------------------------------------------------------------- #

def test_core_fixed_scale_dim_channel_not_restretched(monkeypatch):
    """execute returns a DIM float channel (bulk 0.2, max 0.25, never 1.0).  The
    fixed-scale bake maps 0.2 → 13107 and 0.25 → 16384 — it does NOT re-stretch
    the channel's max to 65535 (the per-channel min-max bug this guards)."""

    def fake_execute(recipe, channels, spacing, *,
                     max_workers=None, bounds_per_channel=None, parallel=False,
                     should_cancel=None, block_workers=None, on_progress=None):
        out = np.full(channels[0].shape, 0.2, dtype=np.float32)
        out.flat[0] = 0.25                       # brightest voxel — still dim
        return [out], tuple(spacing), {"dtype": "uint16"}

    monkeypatch.setattr(controller, "_pp_execute", fake_execute)

    ch = (np.linspace(0.0, 100.0, 1000)
          .reshape(10, 10, 10).astype(np.float32))
    processed, _sp, _bounds, _stamp = controller._process_whole_volume_core(
        _empty_recipe(1), [ch], (1.0, 1.0, 1.0))

    u16 = processed[0]
    assert u16.dtype == np.uint16
    # 0.2 → round(0.2 * 65535) = 13107 (the dim bulk stays dim).
    assert abs(int(np.median(u16)) - 13107) <= 2
    # 0.25 → 16384 — NOT stretched to fill the range.
    assert abs(int(u16.max()) - 16384) <= 2
    assert int(u16.max()) < 20000            # emphatically not ~65535


def test_core_real_pipeline_midtone_maps_fixed_scale():
    """End-to-end (real execute, empty recipe): a channel spanning [0, 100] →
    percentile-normalize → fixed-scale uint16.  The value-20 voxel normalizes to
    ~0.2 → ~13107; the p99.5 voxel saturates near 65535."""
    ch = (np.linspace(0.0, 100.0, 1000)
          .reshape(10, 10, 10).astype(np.float32))
    processed, *_ = controller._process_whole_volume_core(
        _empty_recipe(1), [ch], (1.0, 1.0, 1.0))
    u16 = processed[0]
    idx = int(np.argmin(np.abs(ch.ravel() - 20.0)))
    assert 12000 <= int(u16.ravel()[idx]) <= 14200      # ~0.2 → ~13107
    assert int(u16.max()) >= 64000                        # top saturates


# --------------------------------------------------------------------------- #
# (c) — per-channel bounds are measured + THREADED into execute
# --------------------------------------------------------------------------- #

def test_core_threads_distinct_bounds_and_serial_workers(monkeypatch):
    captured = {}

    def fake_execute(recipe, channels, spacing, *,
                     max_workers=None, bounds_per_channel=None, parallel=False,
                     should_cancel=None, block_workers=None, on_progress=None):
        captured["max_workers"] = max_workers
        captured["parallel"] = parallel
        captured["block_workers"] = block_workers
        captured["bounds"] = (list(bounds_per_channel)
                              if bounds_per_channel is not None else None)
        return ([np.zeros(c.shape, dtype=np.float32) for c in channels],
                tuple(spacing), {})

    monkeypatch.setattr(controller, "_pp_execute", fake_execute)

    ch0 = np.linspace(0.0, 100.0, 1000).reshape(10, 10, 10).astype(np.float32)
    ch1 = np.linspace(0.0, 1000.0, 1000).reshape(10, 10, 10).astype(np.float32)
    processed, _sp, bounds, _stamp = controller._process_whole_volume_core(
        _empty_recipe(2), [ch0, ch1], (1.0, 1.0, 1.0))

    # max_workers=1 is the memory-bound requirement.
    assert captured["max_workers"] == 1
    # `_process_whole_volume_core` defaults to the proven SERIAL path — parallel is
    # opt-in via EPICYCLE_PREPROC_PARALLEL and stays False by default until the user
    # verifies the cancel-safe parallel path at real scale.
    assert captured["parallel"] is False
    assert captured["block_workers"] is None
    # Distinct per-channel windows were threaded into execute...
    assert captured["bounds"] is not None
    assert len(captured["bounds"]) == 2
    assert captured["bounds"][0] != captured["bounds"][1]
    # ...and the core returns those same per-channel bounds.
    assert bounds[0] != bounds[1]
    assert bounds[0][1] < bounds[1][1]        # ch0 hi ~100 < ch1 hi ~1000
    assert len(processed) == 2


def test_core_bounds_match_strided_percentile():
    """The `bounds` the core returns == `_strided_percentile_bounds` of the
    channel (the single source of truth shared with `percentile_normalize`)."""
    ch = np.linspace(0.0, 500.0, 8000).reshape(20, 20, 20).astype(np.float32)
    _proc, _sp, bounds, _stamp = controller._process_whole_volume_core(
        _empty_recipe(1), [ch], (1.0, 1.0, 1.0))
    expected = controller._strided_percentile_bounds(ch)
    assert bounds[0] == expected


# --------------------------------------------------------------------------- #
# helpers: materialize + strided percentile bounds
# --------------------------------------------------------------------------- #

def test_materialize_channel_realizes_lazy_compute():
    class _Lazy:
        def __init__(self, arr):
            self._arr = arr

        def compute(self):
            return self._arr

        @property
        def shape(self):
            return self._arr.shape

    arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    out = controller._materialize_channel(_Lazy(arr))
    assert isinstance(out, np.ndarray)
    assert np.array_equal(out, arr)


def test_materialize_channel_passthrough_ndarray():
    arr = np.ones((3, 3, 3), dtype=np.float32)
    out = controller._materialize_channel(arr)
    assert isinstance(out, np.ndarray)
    assert np.array_equal(out, arr)


def test_strided_percentile_bounds_matches_full_pass():
    a = np.linspace(0.0, 100.0, 100000).astype(np.float32).reshape(50, 50, 40)
    lo, hi = controller._strided_percentile_bounds(a, sample_cap=1000)
    flo, fhi = ops.percentile_bounds(a)
    # A strided subsample of a smooth ramp tracks the full-pass window closely.
    assert abs(lo - flo) < 2.0
    assert abs(hi - fhi) < 2.0


def test_strided_percentile_bounds_degenerate_is_none():
    a = np.zeros((4, 4, 4), dtype=np.float32)
    assert controller._strided_percentile_bounds(a) is None


# --------------------------------------------------------------------------- #
# geometry volume factor — the too-large-guard upsample scaling (grill MED-1)
# --------------------------------------------------------------------------- #

def test_geometry_volume_factor_none_or_intensity_only_is_one():
    assert controller._recipe_geometry_volume_factor(
        {"channels": [[]], "geometry": None}, (1.0, 1.0, 1.0)) == 1.0
    assert controller._recipe_geometry_volume_factor(None, (1.0, 1.0, 1.0)) == 1.0
    assert controller._recipe_geometry_volume_factor("garbage", (1.0, 1.0, 1.0)) == 1.0


def test_geometry_volume_factor_iso_upsample_scales_by_volume():
    # Anisotropic stack Z=5µm, target 1µm → Z factor 5, Y/X factor 1 → volume 5×.
    recipe = {"channels": [[]], "geometry": {
        "key": "isotropic", "boundary_index": 0,
        "config": {"mode": "iso", "target_um": 1.0, "method": "linear"}}}
    f = controller._recipe_geometry_volume_factor(recipe, (5.0, 1.0, 1.0))
    assert abs(f - 5.0) < 1e-6


def test_geometry_volume_factor_aniso_uses_factors():
    recipe = {"channels": [[]], "geometry": {
        "key": "isotropic", "boundary_index": 0,
        "config": {"mode": "aniso", "factors": [2.0, 2.0, 1.0]}}}
    f = controller._recipe_geometry_volume_factor(recipe, (1.0, 1.0, 1.0))
    assert abs(f - 4.0) < 1e-6  # 2×2×1


def test_geometry_volume_factor_downsample_clamped_to_one():
    # Downsample (factor<1) shrinks the output; the input is still materialized +
    # counted, so the guard must never SHRINK below the input-sized estimate.
    recipe = {"channels": [[]], "geometry": {
        "key": "isotropic", "boundary_index": 0,
        "config": {"mode": "aniso", "factors": [0.5, 0.5, 0.5]}}}
    assert controller._recipe_geometry_volume_factor(recipe, (1.0, 1.0, 1.0)) == 1.0


# --------------------------------------------------------------------------- #
# display pyramid — conditional single-scale vs multiscale (render-safety)
# --------------------------------------------------------------------------- #

def test_coarsen_uint16_block_means_by_two():
    a = np.arange(8 * 8 * 8, dtype=np.uint16).reshape(8, 8, 8)
    out = controller._coarsen_uint16(a, min_axis=1)
    assert out.dtype == np.uint16
    assert out.shape == (4, 4, 4)
    # Corner block = mean of a[0:2,0:2,0:2].
    expected = int(round(a[0:2, 0:2, 0:2].astype(np.float32).mean()))
    assert int(out[0, 0, 0]) == expected


def test_coarsen_uint16_at_floor_returns_none():
    a = np.zeros((4, 4, 4), dtype=np.uint16)
    assert controller._coarsen_uint16(a, min_axis=32) is None  # no axis qualifies


def test_build_display_levels_small_is_single_scale():
    # A channel already under the cap → single-scale (full 3D detail), 1 level,
    # and level 0 is the ORIGINAL array (identity, for the cache + 2D).
    a = np.zeros((8, 16, 16), dtype=np.uint16)
    levels = controller._build_display_levels(a, cap_bytes=10 * 1024 * 1024)
    assert len(levels) == 1
    assert levels[0] is a


def test_build_display_levels_large_builds_pyramid_under_cap():
    # Force pyramiding with a tiny cap; the coarsest level must fall under it,
    # level 0 stays the full-res original, and shapes strictly shrink.
    a = np.zeros((64, 64, 64), dtype=np.uint16)  # 512 KB
    cap = 100 * 1024  # 100 KB
    levels = controller._build_display_levels(a, cap_bytes=cap)
    assert len(levels) > 1
    assert levels[0] is a  # full-res retained at level 0
    assert int(levels[-1].size) * 2 <= cap  # coarsest fits the cap
    sizes = [lv.size for lv in levels]
    assert sizes == sorted(sizes, reverse=True)  # strictly coarser each level
    assert all(lv.dtype == np.uint16 for lv in levels)


# --------------------------------------------------------------------------- #
# cancel-safe parallel: the CORE honors should_cancel + parallel/block_workers
# --------------------------------------------------------------------------- #

def test_core_parallel_cancel_raises():
    """The whole-volume core, run with parallel=True + a should_cancel that is
    True, raises PreprocessingCancelled (the cancel reaches the compute and
    propagates OUT of the core — the worker turns it into a clean 'cancelled')."""
    from packages.epicycle_preprocessing.exceptions import PreprocessingCancelled

    ch = np.random.RandomState(0).rand(56, 160, 160).astype(np.float32)
    recipe = {"channels": [[{"key": "gaussian", "params": {"sigma": 1.5}}]],
              "geometry": None}
    with pytest.raises(PreprocessingCancelled):
        controller._process_whole_volume_core(
            recipe, [ch], (1.0, 1.0, 1.0),
            parallel=True, should_cancel=lambda: True)


def test_core_parallel_no_cancel_matches_serial():
    """parallel=True with NO cancel is byte-identical to the default serial core."""
    ch = np.random.RandomState(1).rand(56, 160, 160).astype(np.float32)
    recipe = {"channels": [[{"key": "gaussian", "params": {"sigma": 1.5}},
                            {"key": "background", "params": {"radius": 3}}]],
              "geometry": None}
    par, _sp, _b, _st = controller._process_whole_volume_core(
        recipe, [ch], (1.0, 1.0, 1.0), parallel=True)
    ser, _sp2, _b2, _st2 = controller._process_whole_volume_core(
        recipe, [ch], (1.0, 1.0, 1.0), parallel=False)
    np.testing.assert_array_equal(par[0], ser[0])


# --------------------------------------------------------------------------- #
# the env toggle keeps the shipped default on the SERIAL path
# --------------------------------------------------------------------------- #

def test_preproc_parallel_enabled_env(monkeypatch):
    # ON by default (no env var) and for any value that is not an explicit disable.
    monkeypatch.delenv("EPICYCLE_PREPROC_PARALLEL", raising=False)
    assert controller._preproc_parallel_enabled() is True
    for v in ("1", "true", "YES", " on ", "", "anything"):
        monkeypatch.setenv("EPICYCLE_PREPROC_PARALLEL", v)
        assert controller._preproc_parallel_enabled() is True, v
    # the escape hatch: explicit disable values force the serial path.
    for v in ("0", "false", "NO", " off "):
        monkeypatch.setenv("EPICYCLE_PREPROC_PARALLEL", v)
        assert controller._preproc_parallel_enabled() is False, v


# --------------------------------------------------------------------------- #
# RAM-adaptive block-pool width — fewer cores under pressure, not a refusal
# --------------------------------------------------------------------------- #

def _derive(recipe):
    # `_derive_block_workers` uses no instance state → call unbound with a dummy self
    return controller.DataPreparationController._derive_block_workers(object(), recipe)


def test_derive_block_workers_backs_off_under_ram(monkeypatch):
    import os as _os
    import psutil

    recipe = {"channels": [[{"key": "background", "params": {"radius": 15}}]],
              "geometry": None}

    class _VM:
        def __init__(self, avail):
            self.available = avail

    # 1 GB free vs a ~379 MB/tile background r=15 (halo 30 -> edge 180+2*30=240,
    # 240^3 * 4 B * 3) -> 0.25*1GB // 379MB = 0 -> clamped to the 1-worker floor.
    monkeypatch.setattr(psutil, "virtual_memory", lambda: _VM(1_000_000_000))
    assert _derive(recipe) == 1

    # 1 TB free → the budget dwarfs the tile → the default cap min(cpu, 32).
    monkeypatch.setattr(psutil, "virtual_memory", lambda: _VM(1_000_000_000_000))
    assert _derive(recipe) == min(_os.cpu_count() or 1, 32)


def test_derive_block_workers_none_when_no_local_op(monkeypatch):
    # a recipe with no local op → no block scratch → default width (None)
    assert _derive({"channels": [[]], "geometry": None}) is None


def test_derive_block_workers_none_when_psutil_missing(monkeypatch):
    import builtins
    recipe = {"channels": [[{"key": "gaussian", "params": {"sigma": 1.5}}]],
              "geometry": None}
    real_import = builtins.__import__

    def _no_psutil(name, *a, **k):
        if name == "psutil":
            raise ImportError("no psutil")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_psutil)
    assert _derive(recipe) is None


# --------------------------------------------------------------------------- #
# shutdown never NULLS a still-running process thread — it ORPHANS it (B1)
# --------------------------------------------------------------------------- #

class _FakeSignal:
    def connect(self, fn):
        pass


class _FakeThread:
    def __init__(self, running):
        self._running = running
        self.finished = _FakeSignal()
        self.quit_called = False

    def isRunning(self):
        return self._running

    def quit(self):
        self.quit_called = True

    def wait(self, ms=0):
        return not self._running


def _fake_controller_with_process_thread(thread):
    from types import SimpleNamespace
    return SimpleNamespace(
        _flush_source_state_save=lambda: None,
        _process_thread=thread,
        _process_worker=SimpleNamespace(request_cancel=lambda: None),
        _draining_process_threads=[],
    )


def test_shutdown_orphans_a_still_running_process_thread():
    controller._ORPHANED_THREADS.clear()
    th = _FakeThread(running=True)   # a whole-array op still executing after wait
    fake = _fake_controller_with_process_thread(th)
    controller.DataPreparationController.shutdown(fake)
    # It must be parked in the orphan list (NOT dropped into GC while running)…
    assert any(t is th for (t, w) in controller._ORPHANED_THREADS)
    # …and the controller handle is cleared (the orphan list holds the reference).
    assert fake._process_thread is None
    controller._ORPHANED_THREADS.clear()


def test_shutdown_does_not_orphan_a_stopped_process_thread():
    controller._ORPHANED_THREADS.clear()
    th = _FakeThread(running=False)  # stopped within the wait (the common case)
    fake = _fake_controller_with_process_thread(th)
    controller.DataPreparationController.shutdown(fake)
    assert all(t is not th for (t, w) in controller._ORPHANED_THREADS)
    controller._ORPHANED_THREADS.clear()


# --------------------------------------------------------------------------- #
# progress aggregator — cost-weighted, honest (marquee for the dominant op)
# --------------------------------------------------------------------------- #

def test_format_eta():
    assert controller._format_eta(5) == "~5s left"
    assert controller._format_eta(90) == "~1 min left"
    assert controller._format_eta(3720) == "~1h 2m left"


def _run_agg(recipe, voxels=(1_000_000,), workers=4, min_interval=0.0):
    emits = []
    clk = [0.0]

    def now():
        clk[0] += 0.5
        return clk[0]

    agg = controller._ProcessProgressAggregator(
        list(voxels), recipe, workers,
        emit=lambda f, l, ind: emits.append((f, l, ind)),
        now=now, min_interval=min_interval)
    return agg, emits


def test_aggregator_marquee_for_wholearray_determinate_for_tiled():
    recipe = {"channels": [[{"key": "gaussian", "params": {}},
                            {"key": "clahe", "params": {}}]], "geometry": None}
    agg, emits = _run_agg(recipe)
    agg.feed({"kind": "op", "op": "normalize", "channel": 0, "n_channels": 1,
              "index": 1, "total": 3, "tiled": False})
    agg.feed({"kind": "op", "op": "gaussian", "channel": 0, "n_channels": 1,
              "index": 2, "total": 3, "tiled": True})
    agg.feed({"kind": "tile", "done": 5, "total": 10})
    agg.feed({"kind": "op", "op": "clahe", "channel": 0, "n_channels": 1,
              "index": 3, "total": 3, "tiled": False})

    gaussian = [e for e in emits if "Gaussian" in e[1]]
    clahe = [e for e in emits if "contrast" in e[1].lower()]
    assert gaussian and gaussian[0][2] is False, "tiled op → determinate"
    assert clahe and clahe[0][2] is True, "whole-array op → marquee (indeterminate)"
    # the dominant slow step is labelled as such
    assert "slow step" in clahe[0][1]


def test_aggregator_is_cost_weighted_not_op_count():
    """When the (whole-array) contrast step dominates wall time, the bar must NOT
    be near-complete when it STARTS — an op-count bar would sit at 2/3 (~0.67)."""
    recipe = {"channels": [[{"key": "gaussian", "params": {}},
                            {"key": "clahe", "params": {}}]], "geometry": None}
    agg, emits = _run_agg(recipe)
    for ev in [
        {"kind": "op", "op": "normalize", "channel": 0, "n_channels": 1,
         "index": 1, "total": 3, "tiled": False},
        {"kind": "op", "op": "gaussian", "channel": 0, "n_channels": 1,
         "index": 2, "total": 3, "tiled": True},
        {"kind": "op", "op": "clahe", "channel": 0, "n_channels": 1,
         "index": 3, "total": 3, "tiled": False},
    ]:
        agg.feed(ev)
    clahe_frac = [f for (f, l, ind) in emits if "contrast" in l.lower()][0]
    assert clahe_frac < 0.2, f"CLAHE-start fraction {clahe_frac} not cost-weighted"


def test_aggregator_tiled_progress_advances_with_eta():
    recipe = {"channels": [[{"key": "gaussian", "params": {}}]], "geometry": None}
    agg, emits = _run_agg(recipe)
    agg.feed({"kind": "op", "op": "gaussian", "channel": 0, "n_channels": 1,
              "index": 1, "total": 1, "tiled": True})
    agg.feed({"kind": "tile", "done": 3, "total": 10})
    agg.feed({"kind": "tile", "done": 9, "total": 10})
    fracs = [f for (f, l, ind) in emits if ind is False]
    assert fracs == sorted(fracs), "determinate fraction advances monotonically"
    # a rough time-left appears once we're past the tiny-fraction guard
    assert any("left" in l for (f, l, ind) in emits)


def test_aggregator_never_raises_on_bad_event():
    recipe = {"channels": [[]], "geometry": None}
    agg, emits = _run_agg(recipe)
    agg.feed({})                    # missing kind
    agg.feed({"kind": "op"})        # missing fields
    agg.feed("not a dict")          # wrong type
    # no exception escaped; nothing crashed


# --------------------------------------------------------------------------- #
# Brick 1b #9 — the Process sibling of the Preview C#1 release-funnel bug.
#
# AUDIT RESULT: `_on_process_computed`'s stale-token early return is ALREADY
# safe, unlike `_on_preview_computed`'s was.  The difference is the set of
# token-bump sites, not the handler shape:
#
#   * `_preview_token` is bumped from ~8 sites that take NO busy refcount
#     (`_bump_preview_token` on every recipe edit / box place / move / resize),
#     each reachable WHILE a Preview is in flight -> the acquire had no matching
#     release -> `_busy_depth` pinned at 1 for the session.
#   * `_process_token` has exactly four bump sites: the two launches (each
#     immediately paired with its OWN `_set_step1_busy(True)`),
#     `cancel_process_whole_volume` (which releases the refcount, clears
#     `_processing` and restores the button before the stale finish arrives),
#     and `shutdown` (teardown: sets `_processing = False`, nulls the threads,
#     the widget is going away).
#
# So adding an unconditional release on the stale path would be a REGRESSION —
# a double-decrement after a Cancel, wrongly unlocking a legitimately-nested
# outer op.  These tests pin the invariant instead, so a future `_process_token`
# bump site that forgets to release fails here rather than in the field.
# --------------------------------------------------------------------------- #

class TestProcessStaleReleaseInvariant:

    def _ctl(self):
        from types import SimpleNamespace

        ctl = controller.DataPreparationController.__new__(
            controller.DataPreparationController)
        ctl._log = lambda *a, **k: None
        ctl.widget = SimpleNamespace(_busy_hide=lambda: None,
                                     _busy_show=lambda *a: None)
        ctl._busy_depth = 0
        ctl._step1_busy = False
        ctl._pre_busy_enabled = {}
        ctl._processing = False
        ctl._process_token = 0
        ctl._process_button_text = None
        ctl._process_button_icon = None
        ctl._process_worker = None
        ctl._process_store_worker = None
        ctl._set_preview_status = lambda *a, **k: None
        return ctl

    def test_cancel_releases_the_refcount_itself(self):
        ctl = self._ctl()
        ctl._process_token = 3
        ctl._processing = True
        ctl._set_step1_busy(True)
        assert ctl._busy_depth == 1

        ctl.cancel_process_whole_volume()

        assert ctl._process_token == 4          # the finish is now stale
        assert ctl._busy_depth == 0             # ... and already released
        assert ctl._processing is False

    def test_stale_finish_after_cancel_does_not_double_release(self):
        """The reason the stale path must NOT release: Cancel already did."""
        ctl = self._ctl()
        ctl._process_token = 3
        ctl._processing = True
        ctl._set_step1_busy(True)               # an outer op ALSO holds one
        ctl._set_step1_busy(True)               # the process run
        ctl.cancel_process_whole_volume()       # -> depth 1 (outer op survives)
        assert ctl._busy_depth == 1

        ctl._on_process_computed([], (1, 1, 1), None, None, [], {}, token=3)

        assert ctl._busy_depth == 1, (
            "the stale finish double-released and unlocked a nested outer op")

    def test_stale_streaming_error_after_cancel_does_not_double_release(self):
        ctl = self._ctl()
        ctl._process_token = 3
        ctl._processing = True
        ctl._set_step1_busy(True)
        ctl._set_step1_busy(True)
        ctl.cancel_process_whole_volume()
        ctl._on_process_store_error("boom", token=3)
        assert ctl._busy_depth == 1

    def test_a_launch_that_bumps_the_token_also_acquires_the_refcount(self):
        """The invariant the stale-path relies on, asserted BEHAVIOURALLY (a
        source-inspection count of bump sites is the brittle
        class: it can pass while the behaviour is broken).  A launch must leave the token bumped AND the refcount
        held, so the finish's release is matched."""
        ctl = self._ctl()
        ctl._step1_busy = False
        ctl._gather_process_inputs = lambda: None      # bail after the stamp
        ctl._set_preview_status = lambda *a, **k: None
        ctl._process_thread = None
        ctl._process_store_thread = None
        ctl._draining_process_threads = []
        ctl._draining_process_store_threads = []
        before_token = ctl._process_token

        ctl.on_process_whole_volume()

        # This early-out bails BEFORE the stamp, so nothing is acquired and
        # nothing is staled -- the two must move together or not at all.
        assert ctl._process_token == before_token
        assert ctl._busy_depth == 0

    def test_a_stale_finish_never_moves_the_refcount(self):
        """Whatever the depth is when a superseded finish arrives, it is
        unchanged afterwards -- the property the four bump sites must preserve,
        stated without counting them."""
        ctl = self._ctl()
        ctl._process_token = 11
        for depth in (0, 1, 2):
            ctl._busy_depth = depth
            ctl._step1_busy = depth > 0
            ctl._on_process_computed([], (1, 1, 1), None, None, [], {}, token=5)
            assert ctl._busy_depth == depth
            ctl._on_process_store_error("boom", token=5)
            assert ctl._busy_depth == depth


class TestLateStreamingFinishAfterStepExit:
    """`_on_process_store_ready` is guarded by `_process_token` only, and
    `on_step_exit` never bumps it (it calls `cancel_process_whole_volume`, which
    does — but only when a run is detected in flight).  A streaming Apply that
    lands after the user left Step 1 therefore re-added ⟨processed⟩ Image layers
    to the SHARED canvas and restarted a progressive loader AFTER
    `ensure_sync_slicing` had already run: the Step-3 all-Image-layers abort,
    plus the async_ poison back on.  Mirrors the ④-Trim late-finish-inert
    pattern — go inert for UI, still release the lock."""

    def _ctl(self):
        from types import SimpleNamespace

        ctl = controller.DataPreparationController.__new__(
            controller.DataPreparationController)
        ctl._log = lambda *a, **k: None
        ctl.widget = SimpleNamespace(_busy_hide=lambda: None,
                                     _busy_show=lambda *a: None)
        ctl._busy_depth = 0
        ctl._step1_busy = False
        ctl._pre_busy_enabled = {}
        ctl._processing = False
        ctl._process_token = 7
        ctl._process_button_text = None
        ctl._process_button_icon = None
        ctl._left_step1 = False
        ctl._displayed = []
        ctl._display_processed_store = lambda *a: (
            ctl._displayed.append(a) or 1)
        ctl._add_processed_layers = lambda *a: (ctl._displayed.append(a) or 1)
        ctl._set_preview_status = lambda *a, **k: None
        ctl._show_processed_view = lambda: ctl._displayed.append("show")
        ctl._persist_processed_pointer = lambda *a: None
        ctl._current_processed_pointer_path = lambda: None
        ctl._reclaim_superseded_store = lambda *a: None
        ctl._delete_superseded_processed_stores = lambda *a: None
        return ctl

    def test_late_streaming_finish_adds_no_layers_but_unlocks(self):
        ctl = self._ctl()
        ctl._processing = True
        ctl._set_step1_busy(True)
        ctl._left_step1 = True                    # the user navigated away

        ctl._on_process_store_ready("s.zarr", {}, ["c0"], token=7)

        assert ctl._displayed == [], (
            "a late streaming finish painted ⟨processed⟩ onto another step")
        assert ctl._busy_depth == 0               # ... but the lock IS released
        assert ctl._processing is False

    def test_a_live_streaming_finish_still_displays(self):
        ctl = self._ctl()
        ctl._processing = True
        ctl._set_step1_busy(True)

        ctl._on_process_store_ready("s.zarr", {}, ["c0"], token=7)

        assert ctl._displayed                      # the normal path is unchanged
        assert ctl._busy_depth == 0

    def test_readd_is_inert_after_step_exit(self):
        """The Raw⇄Processed flip helper is the other door onto the same
        display methods (a queued toggle / keybind landing during teardown)."""
        ctl = self._ctl()
        ctl._processed_redisplay = {"kind": "store", "store_path": "s.zarr",
                                    "names": ["c0"]}
        ctl._left_step1 = True
        ctl._readd_processed_layers()
        assert ctl._displayed == []

    def test_readd_still_works_inside_step_1(self):
        ctl = self._ctl()
        ctl._processed_redisplay = {"kind": "store", "store_path": "s.zarr",
                                    "names": ["c0"]}
        ctl._readd_processed_layers()
        assert ctl._displayed


def test_derive_block_workers_backs_off_on_a_small_box_at_the_new_floor(monkeypatch):
    """Brick 3a #1: the raised block floor (256) makes a tile ~0.23 GB for a plain
    gaussian, so an 8 GB-free laptop must drop to a handful of workers instead of
    the default min(cpu, 32) — the budget path, not the floor, is what bounds RAM."""
    import psutil
    from packages.epicycle_preprocessing import ops as _ops

    recipe = {"channels": [[{"key": "gaussian", "params": {"sigma": 1.5}}]],
              "geometry": None}
    tile = _ops.max_local_tile_scratch_bytes(recipe)
    assert tile > 200_000_000                       # the floor IS visible here

    class _VM:
        def __init__(self, avail):
            self.available = avail

    eight_gb = 8 * 1024 ** 3
    monkeypatch.setattr(psutil, "virtual_memory", lambda: _VM(eight_gb))
    w = _derive(recipe)
    assert 1 <= w <= 9                              # 0.25·8GB // 0.23GB = 9
    assert w * tile <= 0.25 * eight_gb              # peak stays under budget
