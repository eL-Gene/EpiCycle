"""Cells ▸ Whole patch: the patch fills to the walls with a background marker.

Pinned here (the page half; the engine half is
``test_region_fill_background.py``):

* routing -- a Cells whole patch reaches the engine as
  ``"fill_with_background"`` through BOTH gather functions (the Preview's
  and the box-info worker's), whatever the hidden Grow by says; a box
  reaches it as the Grow by pick (``"compete"`` / ``"flood"``); the growth
  limit falls back to the engine's own default;
* the warm-up rule -- the solver is set up before Spread, and before the
  fill only when the membrane step will run;
* the Advanced card -- on a whole patch Grow by, Wall cost and Growth limit
  are hidden (never grey), Include membrane and the whole-patch line are
  shown; on a box Grow by is back with the box's pick, Wall cost and Growth
  limit and Include membrane with Spread only; the help slot and the
  summary on a whole patch;
* the walls picture -- one slice of the wall picture in the 3D view, at the
  XY view's depth, moving with it, on a real ``ViewerModel``;
* one image channel -- Preview never raises, the whole patch and a box
  return a result, Nuclei masks is grey with its words, a Nuclei-masks pick
  stops before any worker, the whole-patch amber;
* the size ceiling and a parked fill -- refused before any read, and while
  an earlier Add thread still runs;
* the progress bar -- the fill's last step leaves the bar moving;
* the names that retired stay retired.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from packages.epicycle_annotator import add_preview as AP
from packages.epicycle_annotator.reshape import region as R
from packages.epicycle_annotator.reshape import spec as S

from .add_harness import FakeThread, Harness, install_fakes
from .add_widget_fixture import qapp, reset_add, widget  # noqa: F401
from .test_add_outline_wiring import _CellsAdv, _land_info
from .test_reshape_region import _threshold, _tissue

def _cls():
    from napari_epicycle.annotator.curation_3d_widget import (
        Curation3DAnnotator,
    )

    return Curation3DAnnotator


# --------------------------------------------------------------------------- #
# the orchestrator (the Qt-free harness)
# --------------------------------------------------------------------------- #
def _harness(monkeypatch, *, one_channel=False, outline="none"):
    install_fakes(monkeypatch)
    mem, nuc, cells = _tissue(shape=(12, 40, 40))
    ch = {"membrane": mem} if one_channel else {"membrane": mem,
                                                "nuclei": nuc}
    h = Harness(np.zeros(cells.shape, np.int32), images=ch)
    h.adv = _CellsAdv()
    h.adv.params.update(outline=outline, dark_outside=False)
    h.ctx["spacing_source"] = "qc_settings"
    h._add_run_guide = cells
    h.ch = ch
    h._add_dark_cache = {h._add_dark_key(): _threshold(ch)}
    return h


class _Stop(Exception):
    pass


def _capture_engine(monkeypatch):
    """Record the growth rule every engine call is asked for."""
    seen = {"preview": [], "info": []}
    real_seeds = R.find_region_seeds

    def _fill(existing, channels, spec, **kw):
        seen["preview"].append(spec.params["growth"])
        raise _Stop()

    def _seeds(existing, channels, spec, **kw):
        seen["info"].append(spec.params["growth"])
        return real_seeds(existing, channels, spec, **kw)

    monkeypatch.setattr(R, "reshape_empty_region", _fill)
    monkeypatch.setattr(R, "find_region_seeds", _seeds)
    monkeypatch.setattr(AP, "_warm_up", lambda progress: (AP.NUMBA_OK, ""))
    return seen


def _engine_growth(h, seen):
    """The growth rule the box-info worker and the Preview hand the
    engine, through the real gather functions."""
    seen["info"].clear()
    seen["preview"].clear()
    h._add_info = None
    h._request_add_box_info()
    _land_info(h)
    kw, why = h._add_gather_preview()
    assert why is None, why
    with pytest.raises(_Stop):
        AP.run_add_preview(**kw)
    return seen["info"][-1], seen["preview"][-1], kw


@pytest.mark.parametrize("hidden", ["flood", "compete"])
def test_routing(monkeypatch, hidden):
    h = _harness(monkeypatch)
    seen = _capture_engine(monkeypatch)
    h.adv.params["growth"] = hidden
    h._start_add_whole()
    assert h._add_whole_patch
    info, preview, kw = _engine_growth(h, seen)
    assert info == "fill_with_background"
    assert preview == "fill_with_background"
    assert AP.cells_growth(kw["params"]) == "fill_with_background"
    # a box: the Grow by pick, whatever it is
    h.place_box((slice(0, 12), slice(0, 20), slice(0, 20)))
    h._add_whole_patch = False
    info, preview, _kw = _engine_growth(h, seen)
    assert info == hidden and preview == hidden


def test_the_growth_limit_falls_back_to_the_engine_default(monkeypatch):
    h = _harness(monkeypatch)
    _capture_engine(monkeypatch)
    h.adv.params.pop("growth_limit", None)
    h.adv.params["growth"] = "compete"
    h.place_box((slice(0, 12), slice(0, 20), slice(0, 20)))
    h._request_add_box_info()
    _land_info(h)
    kw, why = h._add_gather_preview()
    assert why is None, why
    assert kw["params"]["growth_limit"] == S.DEFAULT_PARAMS["growth_limit"]


def test_cells_growth():
    assert AP.cells_growth({"whole_patch": True}) == AP.GROW_PATCH
    assert AP.GROW_PATCH == "fill_with_background"
    for g in ("flood", "compete"):
        assert AP.cells_growth({"whole_patch": True, "growth": g}) == (
            AP.GROW_PATCH)
        assert AP.cells_growth({"growth": g}) == g
        assert AP.cells_growth({"whole_patch": False, "growth": g}) == g
    assert AP.cells_growth({}) == AP.GROW_FLOOD


@pytest.mark.parametrize("params,warm", [
    ({"growth": "compete"}, True),
    ({"growth": "flood"}, False),
    ({"growth": "flood", "whole_patch": True, "include_membrane": True},
     True),
    # the whole patch stays gated on numba: checked with the tick off too
    ({"growth": "compete", "whole_patch": True, "include_membrane": False},
     True),
])
def test_the_warm_up_rule(monkeypatch, params, warm):
    calls = []

    def _warm(progress):
        calls.append(progress)
        return AP.NUMBA_OK, ""

    def _fill(*_a, **_k):
        raise _Stop()

    monkeypatch.setattr(AP, "_warm_up", _warm)
    monkeypatch.setattr(R, "reshape_empty_region", _fill)
    shape = (4, 8, 8)
    with pytest.raises(_Stop):
        AP.run_add_preview(
            preset="cells",
            roi=tuple(slice(0, n) for n in shape),
            existing_roi=np.zeros(shape, np.int32), next_label=1,
            spacing=(1.0, 1.0, 1.0),
            params=dict(params, source=AP.SOURCE_CLICKS),
            wall_roi=np.zeros(shape, np.float32),
            click_points=[(2, 4, 4)], reserved_ids=[1])
    assert bool(calls) is warm


@pytest.mark.parametrize("membrane", [True, False])
@pytest.mark.parametrize("why", [AP.NUMBA_MISSING_KEY, AP.NUMBA_CACHE_KEY])
def test_a_whole_patch_without_numba_is_refused_in_its_own_words(
        monkeypatch, membrane, why):
    """Whatever the membrane tick, the fill refuses where numba cannot run,
    with the whole patch's sentence (never the Spread one)."""
    ran = []

    def _fill(*_a, **_k):
        ran.append(1)
        raise _Stop()

    monkeypatch.setattr(AP, "_warm_up", lambda progress: (why, "engine"))
    monkeypatch.setattr(R, "reshape_empty_region", _fill)
    shape = (4, 8, 8)
    out = AP.run_add_preview(
        preset="cells", roi=tuple(slice(0, n) for n in shape),
        existing_roi=np.zeros(shape, np.int32), next_label=1,
        spacing=(1.0, 1.0, 1.0),
        params={"growth": "flood", "whole_patch": True,
                "include_membrane": membrane, "source": AP.SOURCE_CLICKS},
        wall_roi=np.zeros(shape, np.float32),
        click_points=[(2, 4, 4)], reserved_ids=[1])
    assert ran == []
    assert out.status == "refused"
    assert out.refusal == AP.whole_needs_numba_sentence(why)
    assert out.refusal != AP.no_numba_sentence(why)


# --------------------------------------------------------------------------- #
# the Advanced card (the real panel)
# --------------------------------------------------------------------------- #
def _cells_page(w, *, whole=False, numba=True):
    reset_add(w)
    w.set_add_preset("cells")
    w.update_add_ui_state(box_active=True, whole_patch=whole,
                          numba_ok=numba, cleared_in_force=False)
    w._add_advanced_toggle.setChecked(True)


def _undo_page(w):
    w._add_advanced_toggle.setChecked(False)
    w.update_add_ui_state(box_active=False, whole_patch=False,
                          numba_ok=True, cleared_in_force=False)
    w.add_grow_combo.setCurrentIndex(0)
    w.add_include_membrane_cb.setChecked(True)
    w.add_dark_outside_cb.setChecked(True)
    w.add_outline_combo.setCurrentIndex(w.add_outline_combo.count() - 1)
    reset_add(w)


def _title(w):
    return str(w._add_advanced_toggle.property("_epicycle_chevron_title"))


def test_the_live_rule_swaps_the_line_and_hides_the_membrane_tick(widget):
    """A Cells whole patch filled as you click (the box rule): the same
    slot says the live words, the membrane tick (never read there) is
    hidden, never grey, and the summary has no membrane part; off the
    rule, today's words byte for byte."""
    w = widget
    _cells_page(w)
    try:
        w.update_add_ui_state(whole_patch=True, live_whole_rule=True)
        assert not w._add_whole_fill_label.isHidden()
        assert w._add_whole_fill_label.text() == AP.ADD_HELP_WHOLE_FILL_LIVE
        assert w.add_include_membrane_cb.isHidden()
        assert w.add_include_membrane_cb.isEnabled()
        assert AP.ADD_SUM_ADVANCED_FILL_LIVE in _title(w)
        assert "membrane" not in _title(w)
        w.update_add_ui_state(live_whole_rule=False)
        assert w._add_whole_fill_label.text() == AP.ADD_HELP_WHOLE_FILL
        assert not w.add_include_membrane_cb.isHidden()
        assert AP.ADD_SUM_ADVANCED_FILL in _title(w)
    finally:
        w.update_add_ui_state(live_whole_rule=False)
        _undo_page(w)


def test_advanced_card(widget):
    w = widget
    _cells_page(w)
    try:
        # the three rows have one wrapper each, inside the Cells card (Wall
        # cost and Growth limit inside Spread's own box); Include membrane
        # left Spread's box for the card itself
        assert w._add_grow_row.parentWidget() is w._add_cells_adv
        for row in (w._add_wall_cost_row, w._add_growth_limit_row):
            assert row.parentWidget() is w._add_spread_adv
        assert w._add_spread_adv.parentWidget() is w._add_cells_adv
        assert w.add_include_membrane_cb.parentWidget() is w._add_cells_adv
        assert w._add_whole_fill_label.parentWidget() is w._add_cells_adv
        assert w._add_cells_adv.layout().indexOf(
            w._add_whole_fill_label) == 0
        assert w._add_whole_fill_label.text() == AP.ADD_HELP_WHOLE_FILL
        # a box with Spread: every growth row
        w.add_grow_combo.setCurrentIndex(AP.GROW_KEYS.index(AP.GROW_SPREAD))
        assert not w._add_grow_row.isHidden()
        assert not w._add_wall_cost_row.isHidden()
        assert not w._add_growth_limit_row.isHidden()
        assert not w.add_include_membrane_cb.isHidden()
        assert w._add_whole_fill_label.isHidden()
        assert "spreading" in _title(w)
        # the whole patch: the three rows hidden, the box's pick kept
        w.update_add_ui_state(whole_patch=True)
        assert not w._add_cells_adv.isHidden()
        assert w._add_grow_row.isHidden()
        assert w._add_wall_cost_row.isHidden()
        assert w._add_growth_limit_row.isHidden()
        assert not w.add_include_membrane_cb.isHidden()
        assert not w._add_whole_fill_label.isHidden()
        assert w.get_add_grow() == AP.GROW_SPREAD
        assert AP.ADD_SUM_ADVANCED_FILL in _title(w)
        assert "spreading" not in _title(w)
        w.add_include_membrane_cb.setChecked(False)
        assert AP.ADD_SUM_ADVANCED_FILL_NO_MEMBRANE in _title(w)
        w.add_include_membrane_cb.setChecked(True)
        # back to a box: Spread is still the pick
        w.update_add_ui_state(whole_patch=False)
        assert not w._add_grow_row.isHidden()
        assert w.get_add_grow() == AP.GROW_SPREAD
        assert w._add_whole_fill_label.isHidden()
        # a box with Fill: Wall cost, Growth limit, Include membrane hidden
        w.add_grow_combo.setCurrentIndex(AP.GROW_KEYS.index(AP.GROW_FLOOD))
        assert not w._add_grow_row.isHidden()
        assert w._add_wall_cost_row.isHidden()
        assert w._add_growth_limit_row.isHidden()
        assert w.add_include_membrane_cb.isHidden()
        # a whole patch never forces Spread: Fill stays the pick, hidden
        w.update_add_ui_state(whole_patch=True)
        assert w.get_add_grow() == AP.GROW_FLOOD
        assert not w.add_include_membrane_cb.isHidden()
        g = w.add_grow_combo
        assert g.itemText(0) == AP.ADD_GROW_FLOOD
        assert g.model().item(0).isEnabled()
        w.update_add_ui_state(whole_patch=False)
        assert w.get_add_grow() == AP.GROW_FLOOD
    finally:
        _undo_page(w)


def test_the_help_slot_on_a_whole_patch(widget):
    w = widget
    # Spread picked on a box with no outline and the tick off: the
    # open-background note is said there ...
    _cells_page(w)
    try:
        w.add_outline_combo.setCurrentIndex(
            AP.OUTLINE_KEYS.index(AP.OUTLINE_NONE))
        w.add_dark_outside_cb.setChecked(False)
        w.add_grow_combo.setCurrentIndex(AP.GROW_KEYS.index(AP.GROW_SPREAD))
        assert AP.ADD_NOTE_SPREAD_OPEN in w._add_advanced_help.text()
        # ... and never on a whole patch, with the same pick kept
        w.update_add_ui_state(whole_patch=True)
        assert w.get_add_grow() == AP.GROW_SPREAD
        assert AP.ADD_NOTE_SPREAD_OPEN not in w._add_advanced_help.text()
    finally:
        _undo_page(w)
    _cells_page(w, whole=True, numba=False)
    try:
        w.add_outline_combo.setCurrentIndex(
            AP.OUTLINE_KEYS.index(AP.OUTLINE_NONE))
        w.add_dark_outside_cb.setChecked(False)
        text = w._add_advanced_help.text()
        assert AP.ADD_NOTE_SPREAD_OPEN not in text
        assert AP.no_numba_sentence(AP.NUMBA_MISSING_KEY) not in text
        assert AP.no_numba_sentence(AP.NUMBA_CACHE_KEY) not in text
        # the same knobs on a box: the reason is said there
        w.update_add_ui_state(whole_patch=False)
        assert AP.ADD_NO_NUMBA in w._add_advanced_help.text()
    finally:
        _undo_page(w)


def test_fill_progress(widget, monkeypatch):
    from napari_epicycle.annotator.reshape_worker import stage_text

    h = _harness(monkeypatch)
    h._start_add_whole()
    h._add_busy = True
    h._on_add_progress("add_markers", 3, 4)
    assert h.adv.pushes[-1]["progress"] == (3, 4)
    h._on_add_progress("add_cut", 4, 4)
    push = h.adv.pushes[-1]
    assert push["progress"] == ()
    assert push["stage_text"] == stage_text("add_cut")
    # the panel: () is the moving look
    reset_add(widget)
    try:
        widget.update_add_ui_state(busy=True, progress=(3, 4))
        assert widget.add_busy_bar.maximum() == 4
        widget.update_add_ui_state(progress=())
        assert widget.add_busy_bar.maximum() == 0
    finally:
        widget.update_add_ui_state(busy=False)


# --------------------------------------------------------------------------- #
# the walls picture: one slice, at the XY view's depth
# --------------------------------------------------------------------------- #
def test_walls_plane(monkeypatch):
    from napari.components import ViewerModel
    from napari.layers import Labels

    h = _harness(monkeypatch)
    shape = h.sync_manager.shared_labels["L"].shape
    h.viewer = ViewerModel()
    main = Labels(np.zeros(shape, np.int32), name="L",
                  scale=(2.0, 0.5, 0.5), translate=(10.0, 3.0, -4.0))
    h.viewer.layers.append(main)
    h._add_main_label_layer = lambda: main
    cls = _cls()
    h._add_temp_layer = types.MethodType(cls._add_temp_layer, h)
    h._remove_temp_layer = types.MethodType(cls._remove_temp_layer, h)
    # the XY side view: the volume at identity scale, sliced on z
    xy = ViewerModel()
    xy.add_image(np.zeros(shape, np.float32))
    h.viewer_model_xy = xy
    h.viewer_model_xz = None
    h.viewer_model_yz = None
    xy.dims.set_point(0, 7)
    n_cbs = len(xy.dims.events.point.callbacks)
    roi = (slice(3, 11), slice(4, 24), slice(6, 30))
    crop_shape = tuple(sl.stop - sl.start for sl in roi)
    res = AP.AddPreviewResult(roi=roi, preset="cells")
    res.elevation = np.random.default_rng(0).random(crop_shape).astype(
        np.float32)
    h._add_preview_res = res
    assert h._show_add_walls() is True
    plane = h._temp_guide_layers["add_walls"]
    assert plane.depiction == "plane"
    assert tuple(plane.plane.normal) == (1.0, 0.0, 0.0)
    pos = tuple(plane.plane.position)
    assert pos[1] == pytest.approx(crop_shape[1] / 2.0)
    assert pos[2] == pytest.approx(crop_shape[2] / 2.0)

    def _world_z():
        return float(plane.data_to_world(tuple(plane.plane.position))[0])

    def _slice_world_z(z):
        return float(main.data_to_world((z, 0.0, 0.0))[0])

    assert pos[0] == pytest.approx(7 - 3)
    assert _world_z() == pytest.approx(_slice_world_z(7))
    assert len(xy.dims.events.point.callbacks) == n_cbs + 1
    # it follows the XY view
    xy.dims.set_point(0, 9)
    assert _world_z() == pytest.approx(_slice_world_z(9))
    # clamped into the crop
    xy.dims.set_point(0, 0)
    assert tuple(plane.plane.position)[0] == pytest.approx(0.0)
    xy.dims.set_point(0, 11)
    assert tuple(plane.plane.position)[0] == pytest.approx(crop_shape[0] - 1)
    # a move in the plane of the view does not touch the depth
    moved = []
    plane.plane.events.position.connect(lambda e: moved.append(e))
    xy.dims.set_point(1, 5)
    assert moved == []
    # gone with the layer, and the handler with it
    h._remove_temp_layer("add_walls")
    assert "add_walls" not in h._temp_guide_layers
    assert len(xy.dims.events.point.callbacks) == n_cbs
    xy.dims.set_point(0, 5)            # nothing left to move


def test_the_walls_handler_is_a_bound_method():
    import inspect

    src = inspect.getsource(_cls()._show_add_walls)
    assert "_on_add_walls_xy_point" in src
    assert "lambda" not in src


# --------------------------------------------------------------------------- #
# one image channel
# --------------------------------------------------------------------------- #
def test_one_channel(monkeypatch):
    h = _harness(monkeypatch, one_channel=True)
    assert h._add_image_channel_names() == ["membrane"]
    # Nuclei masks is grey, in its own words
    entries = h._add_marker_source_entries()
    nuc = [e for e in entries if e[2][0] == "nuclei_masks"]
    assert nuc == [(AP.ADD_ITEM_NUCLEI_GREY, False, ("nuclei_masks", None))]
    assert AP.ADD_ITEM_NUCLEI_GREY == "Nuclei masks (needs a second channel)"
    # a whole patch with another run's markers
    h._start_add_whole()
    _land_info(h)
    kw, why = h._add_gather_preview()
    assert why is None, why
    res = AP.run_add_preview(**kw)
    assert res.status == "ok", (res.refusal, res.messages)
    assert AP.cells_growth(kw["params"]) == "fill_with_background"
    # the whole-patch amber
    assert h._add_cells_note_text() == AP.ADD_AMBER_ONE_CHANNEL_WHOLE
    assert AP.one_channel_note(spread=True, shrink=True, whole=True) == (
        AP.ADD_AMBER_ONE_CHANNEL_WHOLE)
    # a box with the run's markers
    h.place_box((slice(0, 12), slice(0, 24), slice(0, 24)))
    h._add_whole_patch = False
    h._add_info = None
    h._request_add_box_info()
    _land_info(h)
    kw, why = h._add_gather_preview()
    assert why is None, why
    res = AP.run_add_preview(**kw)
    assert res.status == "ok", (res.refusal, res.messages)
    assert h._add_cells_note_text() == AP.one_channel_note(
        spread=False, shrink=False)
    # a box with click markers
    h.adv.source = ("paint", None)
    h._add_click_markers = [(4, 6, 5), (4, 6, 15)]
    h._add_info = None
    h._request_add_box_info()
    _land_info(h)
    kw, why = h._add_gather_preview()
    assert why is None, why
    res = AP.run_add_preview(**kw)
    assert res.status == "ok", (res.refusal, res.messages)


@pytest.mark.parametrize("nuclei,want", [
    ("", AP.ADD_S1_PICK_NUCLEI_CHANNEL),
    ("membrane", AP.ADD_S1_CHANNEL_CLASH),
])
def test_one_channel_nuclei_masks_stops_before_any_worker(monkeypatch,
                                                          nuclei, want):
    h = _harness(monkeypatch, one_channel=True)
    h.adv.source = ("nuclei_masks", None)
    h.adv.nuclei_channel = nuclei
    h.place_box((slice(0, 12), slice(0, 20), slice(0, 20)))
    h._add_info = AP.AddBoxInfo()
    h._add_info.empty_share = 1.0
    n_threads = len(FakeThread.instances)
    assert AP.preview_block(h._add_s1_state()) == want
    h._preview_add()
    assert h._add_transient == want
    assert h._add_worker is None and h._add_thread is None
    assert len(FakeThread.instances) == n_threads


# --------------------------------------------------------------------------- #
# the size ceiling and a parked fill
# --------------------------------------------------------------------------- #
class _NoRead(np.ndarray):
    """A volume that may be held but not read."""

    armed = False

    def __getitem__(self, item):
        if _NoRead.armed:
            raise AssertionError("the volume was read")
        return super().__getitem__(item)


def test_the_whole_too_big_line():
    assert AP.whole_too_big_line(150_000_000) == (
        "This patch is 150.0 million voxels; a whole patch can be filled up "
        "to 100.0 million. Press S, M or L for a box.")
    # one voxel over the ceiling never reads as the ceiling itself
    assert AP.whole_too_big_line(R.WHOLE_FILL_MAX_VOXELS + 1) == (
        "This patch is 100.1 million voxels; a whole patch can be filled up "
        "to 100.0 million. Press S, M or L for a box.")


def test_the_whole_too_big_line_names_any_ceiling_in_millions(monkeypatch):
    monkeypatch.setattr(R, "WHOLE_FILL_MAX_VOXELS", 19_199)
    line = AP.whole_too_big_line(19_200)
    assert "0.019199" not in line
    assert line == AP.ADD_S1_WHOLE_TOO_BIG.format(
        m=AP.millions(19_200), k=AP.millions(19_199))


def test_size_and_parked(monkeypatch):
    h = _harness(monkeypatch)
    h._start_add_whole()
    _land_info(h)
    n = int(np.prod(h.sync_manager.shared_labels["L"].shape))
    monkeypatch.setattr(R, "WHOLE_FILL_MAX_VOXELS", n - 1)
    want = AP.whole_too_big_line(n)
    state = h._add_s1_state()
    assert AP.preview_block(state) == want
    h._add_refresh_ui()
    assert h.adv.last("preview_enabled") is False
    assert h.adv.last("status_text") == want
    # refused before any read of the volume or a channel
    vol = h.sync_manager.shared_labels["L"].view(_NoRead)
    h.sync_manager.shared_labels["L"] = vol

    def _no_channel(*_a, **_k):
        raise AssertionError("a channel was read")

    h._refine_channel_source = _no_channel
    _NoRead.armed = True
    try:
        kw, why = h._add_gather_preview()
    finally:
        _NoRead.armed = False
    assert kw is None and why == want
    # at the ceiling it is not refused
    monkeypatch.setattr(R, "WHOLE_FILL_MAX_VOXELS", n)
    assert AP.preview_block(h._add_s1_state()) is None


def test_over_the_ceiling_says_no_slow_note(monkeypatch):
    h = _harness(monkeypatch)
    h._start_add_whole()
    n = int(np.prod(h.sync_manager.shared_labels["L"].shape))
    monkeypatch.setattr(AP, "ADD_SLOW_PATCH_VOXELS", 10)
    assert h._add_cells_note_text() == AP.patch_slow_line(
        n, h._add_wall_params().get("wall_kind"))
    monkeypatch.setattr(R, "WHOLE_FILL_MAX_VOXELS", n - 1)
    assert h._add_cells_note_text() == ""
    h._add_refresh_ui()
    assert h.adv.last("derive_note") == ""
    assert h.adv.last("status_text") == AP.whole_too_big_line(n)


def test_over_the_ceiling_the_info_door_reads_nothing(monkeypatch):
    h = _harness(monkeypatch)
    n = int(np.prod(h.sync_manager.shared_labels["L"].shape))
    monkeypatch.setattr(R, "WHOLE_FILL_MAX_VOXELS", n - 1)
    vol = h.sync_manager.shared_labels["L"].view(_NoRead)
    h.sync_manager.shared_labels["L"] = vol

    def _no_channel(*_a, **_k):
        raise AssertionError("a channel was read")

    h._start_add_whole()
    _land_info(h)
    h._add_info = None
    h._refine_channel_source = _no_channel
    _NoRead.armed = True
    try:
        assert h._add_gather_info() is None
        h._request_add_box_info()
    finally:
        _NoRead.armed = False
    assert getattr(h, "_add_info_thread", None) is None
    h._add_refresh_ui()
    assert h.adv.last("status_text") == AP.whole_too_big_line(n)
    # under the ceiling the info job runs again
    monkeypatch.setattr(R, "WHOLE_FILL_MAX_VOXELS", n)
    del h._refine_channel_source
    assert h._add_gather_info() is not None


class _Th:
    def __init__(self, running):
        from .add_harness import FakeSignal

        self.running = running
        self.finished = FakeSignal()

    def isRunning(self):              # noqa: N802 - Qt name
        return self.running


class _W:
    def __init__(self, kind):
        self.kind = kind


def test_a_parked_fill_refuses_the_next_preview(monkeypatch):
    h = _harness(monkeypatch)
    h._start_add_whole()
    _land_info(h)
    parked = _Th(True)
    h._refine_parked_threads = [(parked, _W("add"))]
    n_threads = len(FakeThread.instances)
    h._preview_add()
    assert h._add_transient == AP.ADD_S1_PARKED_REFUSED
    assert h.adv.last("status_text") == AP.ADD_S1_PARKED_REFUSED
    assert len(FakeThread.instances) == n_threads
    assert h._add_worker is None
    # a second press: refused again, the thread watched once
    h._preview_add()
    assert len(parked.finished.cbs) == 1
    assert parked.finished.cbs[0] == h._on_add_parked_finished
    # the earlier fill ends: the sentence goes by itself
    parked.running = False
    parked.finished.emit()
    assert h._add_transient is None
    assert h.adv.last("status_text") != AP.ADD_S1_PARKED_REFUSED
    # another sentence on S1 by then is left alone
    h._add_transient = AP.ADD_S1_STOPPED_PATCH
    h._on_add_parked_finished()
    assert h._add_transient == AP.ADD_S1_STOPPED_PATCH
    h._add_transient = None
    # a parked Refine thread, or an Add thread that has finished: no refusal
    h._refine_parked_threads = [(_Th(True), _W("refine")),
                                (_Th(False), _W("add"))]
    h._preview_add()
    assert h._add_transient != AP.ADD_S1_PARKED_REFUSED
    assert h._add_worker is not None


def test_the_parked_sentence_clears_safely_with_no_page(monkeypatch):
    h = _harness(monkeypatch)
    h._start_add_whole()
    h._add_transient = AP.ADD_S1_PARKED_REFUSED

    def _gone(**_k):
        raise RuntimeError("wrapped C/C++ object has been deleted")

    h._push_add_ui = _gone
    h._on_add_parked_finished()          # must not raise
    assert h._add_transient is None


def test_the_parked_handler_is_a_bound_method():
    import inspect

    src = inspect.getsource(_cls()._preview_add)
    assert "_on_add_parked_finished" in src or "_add_watch_parked" in src
    src = inspect.getsource(_cls()._add_watch_parked)
    assert "lambda" not in src and "_on_add_parked_finished" in src


# --------------------------------------------------------------------------- #
# S1 on a whole patch, and the retired names
# --------------------------------------------------------------------------- #
def test_the_s1_line_of_a_whole_patch_counts_no_pieces(monkeypatch):
    h = _harness(monkeypatch)
    h._start_add_whole()
    _land_info(h)
    h._add_refresh_ui()
    s = h._add_s1_state()
    assert "n_pieces" not in s
    z, y, x = h.sync_manager.shared_labels["L"].shape
    assert h.adv.last("status_text") == AP.box_line(
        (z, y, x), preset="cells", empty_share=1.0, whole_patch=True)
    assert AP.box_line((10, 20, 30), preset="cells", empty_share=0.25,
                       whole_patch=True) == AP.ADD_S1_PATCH_PLAIN.format(
        z=10, y=20, x=30, m="0.1", p=25)
    assert AP.box_line((10, 20, 30), preset="nuclei", empty_share=0.25,
                       whole_patch=True) == AP.ADD_S1_PATCH_PLAIN.format(
        z=10, y=20, x=30, m="0.1", p=25)


RETIRED = ("ADD_S1_PATCH", "ADD_S1_PATCH_ONE", "ADD_STAGE_PIECE",
           "ADD_NOTE_PATCH_SLOW", "ADD_GROW_FLOOD_WHOLE",
           "ADD_WHOLE_NEEDS_SPREAD", "ADD_WHOLE_NEEDS_SPREAD_CACHE",
           "ADD_S1_PATCH_NUCLEI", "ADD_NOTE_PATCH_SLOW_NUCLEI",
           "piece_stage_line", "whole_needs_spread_sentence")


@pytest.mark.parametrize("name", RETIRED)
def test_a_retired_name_stays_retired(name):
    assert not hasattr(AP, name), name


def test_the_retired_orchestrator_parts():
    cls = _cls()
    assert not hasattr(cls, "_add_n_pieces")
    import inspect

    from napari_epicycle.annotator import advanced_features_widget as AFW
    from napari_epicycle.annotator import curation_3d_widget as CW

    for mod in (CW, AFW):
        src = inspect.getsource(mod)
        for word in ("_add_launch_pieces", "_add_grow_before_whole",
                     "_add_n_pieces", "piece_stage_line",
                     "growth_limit\", 14.0"):
            assert word not in src, (mod.__name__, word)


def test_the_renamed_sentences_keep_their_words():
    assert AP.ADD_S1_PATCH_PLAIN == (
        "Whole patch {z} × {y} × {x} voxels ({m} million) · {p} % of it is "
        "empty space")
    assert AP.ADD_NOTE_PATCH_SLOW_STEPS == (
        "This patch is {m} million voxels; a Preview may take {t}, "
        "longer on a slower computer. Stop acts between steps.")
    # {t} carries its own "about" ("under a minute" has none)
    assert AP.wait_phrase(12.74) == "about 13 minutes"
    assert AP.wait_phrase(1.0) == "about a minute"
    assert AP.wait_phrase(0.5) == "under a minute"
    assert AP.whole_needs_numba_sentence(AP.NUMBA_CACHE_KEY) == (
        AP.ADD_WHOLE_NEEDS_NUMBA_CACHE)
    assert AP.whole_needs_numba_sentence(AP.NUMBA_MISSING_KEY) == (
        AP.ADD_WHOLE_NEEDS_NUMBA)
