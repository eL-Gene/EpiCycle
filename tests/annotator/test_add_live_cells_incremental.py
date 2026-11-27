"""Cells ▸ Click markers on a box, filled click by click.

A Cells box at or under ``ADD_LIVE_CELLS_MAX_VOXELS`` with click markers
and the fill to the walls hands the run a prepared box (the click fill's
stage) that the run keeps up to date and hands back when it ends: the
first click prepares the box, every later click floods only a window
around the change.  The stage is keyed on everything but the clicks; it is
let go when that key changes, at Accept, a Clear, Close box, a patch
switch, the end of the session and when "Update as I click" goes off.  A
Spread pick or a ridge wall never hands it; a run that cannot use it (not
compiled yet, flat spots, a failure) is the full run, unchanged on the
page -- on a live click only on a box at or under
``ADD_LIVE_FULL_RUN_MAX_VOXELS``, as the fixture's is (over it the click
is refused instead).  One console line per run says how it ran.
"""

from __future__ import annotations

import numpy as np
import pytest

from packages.epicycle_annotator import add_preview as AP
from packages.epicycle_annotator.reshape import incremental as INC

from .add_harness import FakeThread
from .test_add_click_markers import h  # noqa: F401
from .test_add_live_cells_box import (
    CELL_A,
    CELL_B,
    CLICKS,
    _active,
    _fire,
    _finish,
    _id_at,
    _spy_launch,
)
from .test_add_live_nuclei_box import LiveAdv, _FakeTimer

#: a third cell of the click fixture's box, clear of the kept object 7
CELL_C = (11, 20, 12)


def _membrane(shape, seed=0):
    """Bright sheets every 8 voxels in y and x over faint noise: closed
    walls, and no two touching voxels of one value between the clips (a
    box with flat spots keeps the full run)."""
    rng = np.random.default_rng(seed)
    img = (0.05 + rng.normal(0, 0.01, shape)).astype(np.float32)
    img = np.clip(img, 0.0, None)
    img[:, ::8, :] = 1.0
    img[:, :, ::8] = 1.0
    return img


def _cells(hh, *, live):
    from napari.layers import Image

    adv = LiveAdv()
    adv.preset = "cells"
    adv.source = CLICKS
    adv.params["dark_outside"] = False
    adv.params["click_radius_um"] = 1.0
    adv.wall = dict(adv.wall, channel="mem")
    hh.adv = adv
    vol = hh.sync_manager.shared_labels["L"]
    hh.viewer.layers.append(Image(_membrane(vol.shape), name="mem"))
    hh.viewer.layers.append(Image(_membrane(vol.shape, seed=1),
                                  name="mem2"))
    hh._add_info = AP.AddBoxInfo(empty_share=1.0)
    hh._add_live_enabled = bool(live)
    return hh


class _Prepares:
    def __init__(self, monkeypatch):
        self.n = 0
        real = INC.prepare

        def spy(*a, **k):
            self.n += 1
            return real(*a, **k)

        monkeypatch.setattr(INC, "prepare", spy)


@pytest.fixture
def ci(h, monkeypatch):  # noqa: F811
    import qtpy.QtCore

    assert INC.warm_kernel(), "numba is needed for the click fill"
    monkeypatch.setattr(AP, "_INCREMENTAL_OFF", False)
    _FakeTimer.made = []
    monkeypatch.setattr(qtpy.QtCore, "QTimer", _FakeTimer)
    hh = _cells(h, live=True)
    hh.prepares = _Prepares(monkeypatch)
    hh.launched = _spy_launch(hh)
    return hh


@pytest.fixture
def co(h, monkeypatch):  # noqa: F811
    import qtpy.QtCore

    assert INC.warm_kernel(), "numba is needed for the click fill"
    monkeypatch.setattr(AP, "_INCREMENTAL_OFF", False)
    _FakeTimer.made = []
    monkeypatch.setattr(qtpy.QtCore, "QTimer", _FakeTimer)
    hh = _cells(h, live=False)
    hh.prepares = _Prepares(monkeypatch)
    hh.launched = _spy_launch(hh)
    return hh


def _click_and_draw(hh, at):
    assert hh._add_marker_click(at)
    _fire(hh)
    _finish(hh)
    assert hh._add_preview_result is not None, hh._add_transient
    return hh._add_preview_res


def _lines(out):
    return [ln for ln in out.splitlines()
            if ln.startswith("[ADD] cells by click:")]


def _full_run(hh):
    """The last launch run again without a stage: the full run."""
    kw = {k: v for k, v in hh.launched[-1][0].items()
          if k not in ("cancel_event", "stage_holder", "stage_key")}
    return AP.run_add_preview(**kw)


# --------------------------------------------------------------------------- #
# the stage, click by click
# --------------------------------------------------------------------------- #
def test_the_first_click_prepares_the_box_and_the_second_reuses_it(ci):
    first = _click_and_draw(ci, CELL_A)
    assert first.by_click and first.click_how == "stage prepared"
    assert ci.prepares.n == 1
    kept = ci._add_cells_stage
    assert kept is not None and kept["key"] == ci._add_cells_stage_key()
    stage = kept["stage"]
    second = _click_and_draw(ci, CELL_B)
    assert second.by_click and second.click_how == "stage reused"
    assert ci.prepares.n == 1
    assert ci._add_cells_stage["stage"] is stage
    assert ci.launched[-1][0]["stage_holder"]["stage"] is stage
    res = ci._add_preview_result
    assert (res == _id_at(ci, CELL_A)).any()
    assert (res == _id_at(ci, CELL_B)).any()


def test_the_click_fill_equals_the_full_run(ci):
    _click_and_draw(ci, CELL_A)
    _click_and_draw(ci, CELL_B)
    res = ci._add_preview_res
    assert res.by_click
    full = _full_run(ci)
    assert not full.by_click and full.ok
    assert np.array_equal(res.result, full.result)
    assert res.ids == full.ids and res.sizes == full.sizes


def test_while_a_run_holds_the_stage_the_page_does_not(ci):
    _click_and_draw(ci, CELL_A)
    assert ci._add_marker_click(CELL_B)
    _fire(ci)
    assert ci._add_cells_stage is None, "the run holds it"
    _finish(ci)
    assert ci._add_cells_stage is not None


def test_a_manual_preview_takes_the_stage_with_the_tick_off(co):
    co._add_click_markers = [{"zyx": CELL_A, "id": 8, "ticked": True}]
    co._preview_add()
    _finish(co)
    assert co._add_preview_res.click_how == "stage prepared"
    co._add_click_markers.append({"zyx": CELL_B, "id": 9, "ticked": True})
    co._preview_add()
    _finish(co)
    res = co._add_preview_res
    assert res.by_click and res.click_how == "stage reused"
    assert co.prepares.n == 1
    assert np.array_equal(res.result, _full_run(co).result)


# --------------------------------------------------------------------------- #
# what prepares the box again
# --------------------------------------------------------------------------- #
def _changes_the_key(hh, change):
    _click_and_draw(hh, CELL_A)
    k0 = hh._add_cells_stage_key()
    assert hh._add_cells_stage is not None
    change(hh)
    assert hh._add_cells_stage_key() != k0
    hh._add_prune_caches()
    assert hh._add_cells_stage is None


def test_another_wall_channel_prepares_again(ci):
    def change(hh):
        hh.adv.wall = dict(hh.adv.wall, channel="mem2")

    _changes_the_key(ci, change)
    _click_and_draw(ci, CELL_B)
    assert ci.prepares.n == 2
    assert ci._add_preview_res.click_how == "stage prepared"


def test_the_wall_preparation_prepares_again(ci):
    def change(hh):
        hh.adv.wall = dict(hh.adv.wall, chain="zcorrect")

    _changes_the_key(ci, change)


def test_the_dark_rule_is_in_the_key(ci, monkeypatch):
    def change(hh):
        hh.adv.params["dark_outside"] = True
        monkeypatch.setattr(type(hh), "_add_cached_dark",
                            lambda self: 0.25, raising=False)

    _changes_the_key(ci, change)
    k1 = ci._add_cells_stage_key()
    monkeypatch.setattr(type(ci), "_add_cached_dark", lambda self: 0.3,
                        raising=False)
    assert ci._add_cells_stage_key() != k1, "the dark threshold's value"


def test_the_outline_is_in_the_key(ci, monkeypatch):
    def change(hh):
        monkeypatch.setattr(type(hh), "_add_outline_key",
                            lambda self, mode=None: ("shrink", "x"),
                            raising=False)

    _changes_the_key(ci, change)


def test_a_box_move_prepares_again(ci):
    def change(hh):
        hh.place_box((slice(2, 18), slice(2, 27), slice(2, 28)))

    _changes_the_key(ci, change)


def test_the_labels_epoch_is_in_the_key(ci):
    def change(hh):
        hh._add_volume_epoch = int(getattr(hh, "_add_volume_epoch", 0)) + 1

    _changes_the_key(ci, change)


def test_the_finish_settings_keep_the_stage(ci):
    _click_and_draw(ci, CELL_A)
    k0 = ci._add_cells_stage_key()
    ci.adv.params["min_size"] = 5
    ci.adv.params["auto_smooth"] = True
    assert ci._add_cells_stage_key() == k0
    ci._add_prune_caches()
    assert ci._add_cells_stage is not None


# --------------------------------------------------------------------------- #
# what lets it go
# --------------------------------------------------------------------------- #
def test_accept_drops_the_stage(ci):
    ci.adv.params["clear_border"] = False
    _click_and_draw(ci, CELL_A)
    assert ci._add_cells_stage is not None
    ci._accept_add()
    assert ci._add_cells_stage is None


def test_a_clear_drops_the_stage(ci):
    _click_and_draw(ci, CELL_A)
    assert ci._add_cells_stage is not None
    ci._add_drop_cells_stage("a Clear wrote the volume")
    assert ci._add_cells_stage is None
    # the Clear's own path names the drop next to its epoch bump
    import inspect

    from napari_epicycle.annotator import curation_3d_widget as CW

    src = inspect.getsource(CW)
    for why in ("a Clear wrote the volume", "a Clear was put back",
                "Accept wrote the volume"):
        assert '_add_drop_cells_stage("{}")'.format(why) in src, why


def test_close_box_and_a_patch_switch_drop_the_stage(ci):
    _click_and_draw(ci, CELL_A)
    ci._add_forget_clears()                    # a patch switch / a close
    assert ci._add_cells_stage is None
    _click_and_draw(ci, CELL_B)
    assert ci._add_cells_stage is not None
    ci._reject_add()                           # Close box
    assert getattr(ci, "_add_cells_stage", None) is None


def test_the_session_end_drops_the_stage(ci):
    _click_and_draw(ci, CELL_A)
    ci._reset_add_state()
    assert ci._add_cells_stage is None


def test_the_tick_off_drops_the_stage(ci):
    _click_and_draw(ci, CELL_A)
    ci._on_add_live_toggled(False)
    assert ci._add_cells_stage is None


def test_the_preset_leaving_cells_drops_the_stage(ci):
    _click_and_draw(ci, CELL_A)
    ci.adv.preset = "nuclei"
    ci._add_prune_caches()
    assert ci._add_cells_stage is None


# --------------------------------------------------------------------------- #
# what never hands a stage
# --------------------------------------------------------------------------- #
def test_spread_never_hands_a_stage(ci):
    ci.adv.params["growth"] = AP.GROW_SPREAD
    assert ci._add_cells_stage_key() is None
    ci._add_click_markers = [{"zyx": CELL_A, "id": 8, "ticked": True}]
    kw, why = ci._add_gather_preview()
    assert why is None, why
    assert "stage_holder" not in kw


@pytest.mark.parametrize("kind", ["frangi", "meijering", "intensity"])
def test_a_wall_kind_the_click_fill_does_not_redo_hands_none(ci, kind):
    ci.adv.wall = dict(ci.adv.wall, wall_kind=kind)
    assert ci._add_cells_stage_key() is None
    ci._add_click_markers = [{"zyx": CELL_A, "id": 8, "ticked": True}]
    kw, why = ci._add_gather_preview()
    assert why is None, why
    assert "stage_holder" not in kw


def test_over_the_live_limit_no_stage(ci, monkeypatch):
    monkeypatch.setattr(AP, "ADD_LIVE_CELLS_MAX_VOXELS", 1000)
    ci._add_click_markers = [{"zyx": CELL_A, "id": 8, "ticked": True}]
    assert ci._add_cells_stage_key() is None
    kw, why = ci._add_gather_preview()
    assert why is None and "stage_holder" not in kw


def test_another_marker_source_hands_none(ci):
    ci.adv.source = ("nuclei_masks", None)
    assert ci._add_cells_stage_key() is None


# --------------------------------------------------------------------------- #
# the bridge falls back: the full run, the page unchanged
# --------------------------------------------------------------------------- #
def test_a_bridge_fallback_is_the_full_run(ci, monkeypatch, capsys):
    _click_and_draw(ci, CELL_A)
    monkeypatch.setattr(AP, "_cells_by_click", lambda *a, **k: None)
    capsys.readouterr()
    res = _click_and_draw(ci, CELL_B)
    assert not res.by_click
    assert res.click_how.startswith("full run")
    full = _full_run(ci)
    assert np.array_equal(res.result, full.result)
    assert np.array_equal(ci._add_preview_result, full.result)
    assert res.ids == full.ids
    assert len(_lines(capsys.readouterr().out)) == 1


def test_until_the_flood_is_compiled_the_full_run_answers(ci, monkeypatch,
                                                          capsys):
    monkeypatch.setattr(INC, "_WARM", False)
    capsys.readouterr()
    res = _click_and_draw(ci, CELL_A)
    assert not res.by_click and ci.prepares.n == 0
    lines = _lines(capsys.readouterr().out)
    assert lines == ["[ADD] cells by click: full run (the flood is not "
                     "compiled yet), window {} voxels, engine {:.0f} ms"
                     .format(res.click_window, res.click_ms)]
    assert res.click_window == int(np.prod(res.result.shape))


def test_the_solver_check_compiles_the_flood(monkeypatch):
    calls = []
    monkeypatch.setattr(INC, "_WARM", False)
    monkeypatch.setattr(INC, "warm_kernel",
                        lambda: calls.append(1) or True)
    assert AP.numba_answer() == (AP.NUMBA_OK, "")
    assert calls == [1]


# --------------------------------------------------------------------------- #
# a stale run keeps the stage as a cache, applies nothing
# --------------------------------------------------------------------------- #
def test_a_stale_run_keeps_the_stage_and_applies_nothing(ci):
    _click_and_draw(ci, CELL_A)
    stage = ci._add_cells_stage["stage"]
    drawn = ci._add_preview_res
    assert ci._add_marker_click(CELL_B)
    _fire(ci)
    th = FakeThread.instances[-1]
    assert ci._add_marker_click(CELL_C)        # the markers changed
    assert ci._add_live_dirty
    th.run()
    assert ci._add_preview_res is drawn or ci._add_preview_res is None
    assert ci._add_cells_stage is not None
    assert ci._add_cells_stage["stage"] is stage, "kept as a cache"
    th.end()
    assert len(FakeThread.instances) >= 2      # the one run owed
    _finish(ci)
    res = ci._add_preview_res
    assert res.by_click and res.click_how == "stage reused"
    assert ci.prepares.n == 1
    for at in (CELL_A, CELL_B, CELL_C):
        assert (ci._add_preview_result == _id_at(ci, at)).any()


def test_a_stale_run_by_the_walls_keeps_no_stage(ci):
    _click_and_draw(ci, CELL_A)
    assert ci._add_marker_click(CELL_B)
    _fire(ci)
    th = FakeThread.instances[-1]
    ci.adv.wall = dict(ci.adv.wall, chain="zcorrect")
    ci._invalidate_add_preview()
    th.run()
    assert ci._add_cells_stage is None
    th.end()


def test_a_stopped_run_keeps_no_stage(ci):
    _click_and_draw(ci, CELL_A)
    ci._add_click_markers.append({"zyx": CELL_B, "id": 99,
                                  "ticked": True})
    ci._add_live_timer.stop()
    ci._preview_add()
    assert ci._add_cells_stage is None
    ci._stop_add_preview()
    th = FakeThread.instances[-1]
    th.run()
    th.end()
    assert ci._add_cells_stage is None


# --------------------------------------------------------------------------- #
# the console: one line per run
# --------------------------------------------------------------------------- #
def test_each_run_prints_one_line(ci, capsys):
    capsys.readouterr()
    first = _click_and_draw(ci, CELL_A)
    lines = _lines(capsys.readouterr().out)
    assert lines == ["[ADD] cells by click: stage prepared, window {} "
                     "voxels, engine {:.0f} ms".format(first.click_window,
                                                       first.click_ms)]
    second = _click_and_draw(ci, CELL_B)
    lines = _lines(capsys.readouterr().out)
    assert len(lines) == 1 and lines[0].startswith(
        "[ADD] cells by click: stage reused, window ")
    assert 0 < second.click_window <= int(np.prod(second.result.shape))
    assert second.click_line() == lines[0]


def test_the_window_is_the_voxels_the_floods_covered(ci, monkeypatch):
    """A step whose window grew, then flooded the whole box, floods more
    voxels in all than the box holds; the console's window is the voxels
    covered (the rounds nest: the last one holds the others)."""
    real = INC.CellsClickFill.step

    def step(self, *a, **k):
        st = real(self, *a, **k)
        st.voxels_flooded = int(np.prod(self.shape)) + 12345
        st.window_voxels = int(np.prod(self.shape))
        return st

    monkeypatch.setattr(INC.CellsClickFill, "step", step)
    res = _click_and_draw(ci, CELL_A)
    assert res.by_click
    assert res.click_window == int(np.prod(res.result.shape))


def test_a_run_without_a_stage_says_why_it_ran_whole(ci, capsys):
    ci.adv.params["growth"] = AP.GROW_SPREAD
    ci._add_live_enabled = False
    ci._add_click_markers = [{"zyx": CELL_A, "id": 8, "ticked": True}]
    capsys.readouterr()
    ci._preview_add()
    _finish(ci)
    lines = _lines(capsys.readouterr().out)
    assert len(lines) == 1 and lines[0].startswith(
        "[ADD] cells by click: full run (Spread), window "), lines
    assert not _active(ci)
