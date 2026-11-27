"""The Add page says what it can do.

Pinned here, through the real page (``update_add_ui_state`` pushes) and the
real orchestrator methods on the Qt-free harness:

* a Cells whole patch offers "Objects from ..." and Click markers under
  Markers ▸ From (nuclei masks hidden, not grey); the list is rebuilt once
  per change of the whole-patch flag, silently -- no source callback,
  nothing marked touched;
* the box's nuclei-masks pick comes back on a box -- after S / M / L, after
  Close box and a whole patch again, across a preset round trip -- while a
  run or Click markers picked stays picked both ways; a grey item is never
  selected;
* on a whole patch Place markers stays as it was and a click places a
  marker (also a click that was waiting for the next event-loop turn); the
  balls already placed stay; click points reach a Preview only with the
  clicks source;
* a blank From combo on a whole patch reads no run and counts nothing;
* the "Measuring the box for Auto…" line shows only while something is
  being measured: never with no box, never on a whole patch that measures
  nothing, never after a flat crop or a measurement that ended without an
  answer;
* one rule says what a whole patch measures, for the gather and the page;
* turning Place markers on or off refreshes the first line;
* the ball's radius sits beside Place markers (1.0 µm by default on every
  path), and a change redraws the balls once; the placed count is off the
  card body;
* the idle strip says what Clear objects is for while it is on screen;
* each channel item carries its layer's colour, its text the layer name;
* leaving a whole patch by S / M / L goes back to the box from before it.
"""

from __future__ import annotations

import numpy as np
import pytest

from packages.epicycle_annotator import add_preview as AP

from .add_harness import Harness, install_fakes, marker_zyx
from .add_widget_fixture import qapp, reset_add, widget  # noqa: F401
from .test_add_click_markers import h  # noqa: F401
from .test_add_d_fixpass import _cells_page_with_a_run
from .test_add_outline_wiring import _land_info, _world

RUN = ("labels", "cells")
NUCLEI = (AP.SOURCE_NUCLEI, None)
CLICKS = (AP.SOURCE_CLICKS, None)
BLANK = ("labels", None)
ENTRIES = [(AP.ADD_ITEM_RUN.format(name="cells"), True, RUN),
           (AP.ADD_ITEM_NUCLEI, True, NUCLEI),
           (AP.ADD_ITEM_CLICKS, True, CLICKS)]
NO_RUN_ENTRIES = [(AP.ADD_ITEM_NO_RUN, False, ("labels", None)),
                  (AP.ADD_ITEM_NUCLEI, True, NUCLEI),
                  (AP.ADD_ITEM_CLICKS, True, CLICKS)]
NAN = float("nan")


class _Count:
    def __init__(self):
        self.n = 0

    def hit(self, *_a):
        self.n += 1


def _texts(widget):
    c = widget.add_marker_source_combo
    return [c.itemText(i) for i in range(c.count())]


def _leave(widget):
    """Back to a fresh page with no box and no whole patch."""
    widget.update_add_ui_state(box_active=False, whole_patch=False,
                               review_active=False)
    widget.set_add_marker_sources([])
    widget.add_dark_outside_cb.setChecked(True)
    reset_add(widget)


def _cells_box(widget, entries=ENTRIES):
    _leave(widget)
    widget.refresh_add_image_combo(["membrane", "nuclei"])
    widget.set_add_preset("cells")
    widget.set_add_marker_sources(entries)
    widget.update_add_ui_state(box_active=True, whole_patch=False)


# --------------------------------------------------------------------------- #
# the From list of a Cells whole patch
# --------------------------------------------------------------------------- #
def test_whole_patch_offers_runs_and_click_markers(widget, monkeypatch):
    _cells_box(widget)
    combo = widget.add_marker_source_combo
    calls = []
    monkeypatch.setattr(widget, "_on_add_source_cb", calls.append,
                        raising=False)
    clears = []
    real_clear = combo.clear

    def _spy_clear():
        clears.append(1)
        real_clear()

    monkeypatch.setattr(combo, "clear", _spy_clear)
    hits = _Count()
    combo.currentIndexChanged.connect(hits.hit)
    try:
        combo.setCurrentIndex(2)                   # the user: clicks
        assert widget.get_add_marker_source() == CLICKS
        calls.clear()
        hits.n = 0
        widget.clear_add_touched()
        full = [t for t, _e, _k in ENTRIES]
        assert _texts(widget) == full
        widget.update_add_ui_state(whole_patch=True)
        assert _texts(widget) == full, "every source a box takes"
        assert widget.get_add_marker_source() == CLICKS, "kept picked"
        assert len(clears) == 1
        # the same flag again, a push without it, the box pushed again:
        # nothing is rebuilt
        widget.update_add_ui_state(whole_patch=True)
        widget.update_add_ui_state(status_text="x")
        widget.update_add_ui_state(box_active=True)
        assert len(clears) == 1, "one rebuild per change of the flag"
        assert calls == [] and hits.n == 0, "a rebuild fires nothing"
        assert "add_marker_source_combo" not in widget._add_touched
        # Click markers keep their rows on a whole patch; no nuclei rows
        assert not widget._add_click_rows.isHidden()
        assert not widget.add_place_markers_btn.isHidden()
        assert widget._add_nuclei_rows.isHidden()
        # back to a box: the full list, once, Click markers still picked
        widget.update_add_ui_state(whole_patch=False)
        assert _texts(widget) == full
        assert widget.get_add_marker_source() == CLICKS
        assert len(clears) == 2
        assert calls == [] and hits.n == 0
        assert "add_marker_source_combo" not in widget._add_touched
        assert not widget._add_click_rows.isHidden()
    finally:
        combo.currentIndexChanged.disconnect(hits.hit)
        _leave(widget)


# --------------------------------------------------------------------------- #
# the box's pick comes back
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("other_run", [True, False])
@pytest.mark.parametrize("pick", [1, 2])        # nuclei masks, clicks
def test_box_pick_comes_back(widget, monkeypatch, pick, other_run):
    """Every source picked on the box stays picked on the whole patch, and
    on the next box."""
    entries = ENTRIES if other_run else NO_RUN_ENTRIES
    _cells_box(widget, entries)
    combo = widget.add_marker_source_combo
    calls = []
    monkeypatch.setattr(widget, "_on_add_source_cb", calls.append,
                        raising=False)
    try:
        combo.setCurrentIndex(pick)
        key = entries[pick][2]
        assert widget.get_add_marker_source() == key
        calls.clear()
        widget.update_add_ui_state(whole_patch=True)
        # nuclei masks and Click markers fill a whole patch: kept picked
        assert widget.get_add_marker_source() == key
        if not other_run:
            # the grey item (never selected), the nuclei masks and Click
            # markers, as on a box
            assert combo.count() == 3
            assert not widget._add_source_enabled(0)
            assert widget._add_source_enabled(1)
            assert widget._add_source_enabled(2)
        widget.update_add_ui_state(whole_patch=False)
        assert widget.get_add_marker_source() == key
        assert calls == []
    finally:
        _leave(widget)


def test_a_box_run_pick_stays_on_the_whole_patch(widget, monkeypatch):
    _cells_box(widget)
    combo = widget.add_marker_source_combo
    calls = []
    monkeypatch.setattr(widget, "_on_add_source_cb", calls.append,
                        raising=False)
    try:
        combo.setCurrentIndex(0)
        calls.clear()
        widget.update_add_ui_state(whole_patch=True)
        assert widget.get_add_marker_source() == RUN
        widget.update_add_ui_state(whole_patch=False)
        assert widget.get_add_marker_source() == RUN
        assert calls == []
    finally:
        _leave(widget)


def test_box_pick_survives_close_box_and_preset_switch(widget, monkeypatch,
                                                       tmp_path):
    hh, _cells = _cells_page_with_a_run(widget, monkeypatch, tmp_path)
    monkeypatch.setattr(widget, "_on_add_preset_cb",
                        hh._on_add_preset_changed, raising=False)
    try:
        hh._start_add()
        widget.add_marker_source_combo.setCurrentIndex(1)     # nuclei masks
        assert widget.get_add_marker_source() == NUCLEI
        hh._start_add_whole()                     # Whole on the box row
        assert hh._add_whole_patch
        assert widget.get_add_marker_source() == NUCLEI
        hh._add_secondary()                       # Close box
        assert not hh._add_running
        assert widget.get_add_marker_source() == NUCLEI
        hh._start_add_whole()                     # Whole patch, no box
        assert hh._add_running and hh._add_whole_patch
        assert widget.get_add_marker_source() == NUCLEI
        widget.add_preset_combo.setCurrentIndex(0)            # Nuclei
        widget.add_preset_combo.setCurrentIndex(1)            # Cells
        assert hh._add_whole_patch, "a preset switch keeps the whole patch"
        assert widget.get_add_marker_source() == NUCLEI
        hh._set_add_box_size_fraction(1.0 / 8.0)  # S
        assert not hh._add_whole_patch
        assert widget.get_add_marker_source() == NUCLEI
        # Click markers ride the same round trip, picked all the way
        widget.add_marker_source_combo.setCurrentIndex(2)     # clicks
        hh._start_add_whole()
        assert widget.get_add_marker_source() == CLICKS
        hh._add_secondary()                       # Close box
        assert widget.get_add_marker_source() == CLICKS
        hh._start_add_whole()
        widget.add_preset_combo.setCurrentIndex(0)            # Nuclei
        widget.add_preset_combo.setCurrentIndex(1)            # Cells
        assert widget.get_add_marker_source() == CLICKS
        hh._set_add_box_size_fraction(1.0 / 8.0)  # S
        assert widget.get_add_marker_source() == CLICKS
    finally:
        hh._reject_add()
        _leave(widget)


def test_a_run_picked_on_the_whole_patch_stays(widget, monkeypatch,
                                               tmp_path):
    hh, _cells = _cells_page_with_a_run(widget, monkeypatch, tmp_path)
    try:
        hh._start_add()
        widget.add_marker_source_combo.setCurrentIndex(1)     # nuclei masks
        hh._start_add_whole()
        assert widget.get_add_marker_source() == NUCLEI
        widget.add_marker_source_combo.setCurrentIndex(0)     # the user: run
        assert widget.get_add_marker_source() == RUN
        assert hh._add_run_state == "loading", "the picked run is read"
        hh._set_add_box_size_fraction(1.0 / 8.0)  # S
        assert widget.get_add_marker_source() == RUN, "the latest pick wins"
    finally:
        hh._reject_add()
        _leave(widget)


def test_close_box_from_whole_fires_no_source_callback(widget, monkeypatch,
                                                       tmp_path):
    hh, _cells = _cells_page_with_a_run(widget, monkeypatch, tmp_path)
    combo = widget.add_marker_source_combo
    hits = _Count()
    try:
        hh._start_add()
        combo.setCurrentIndex(1)                              # nuclei masks
        widget.clear_add_touched()                            # untouched
        hh._start_add_whole()
        assert widget.get_add_marker_source() == NUCLEI
        seen = []
        monkeypatch.setattr(widget, "_on_add_source_cb", seen.append,
                            raising=False)
        combo.currentIndexChanged.connect(hits.hit)
        hh._add_secondary()                                   # Close box
        assert not hh._add_running
        assert seen == [] and hits.n == 0
        assert widget.get_add_marker_source() == NUCLEI
        assert "add_marker_source_combo" not in widget._add_touched
    finally:
        try:
            combo.currentIndexChanged.disconnect(hits.hit)
        except (TypeError, RuntimeError):
            pass
        hh._reject_add()
        _leave(widget)


# --------------------------------------------------------------------------- #
# clicks on a whole patch
# --------------------------------------------------------------------------- #
def test_whole_patch_takes_clicks(h):
    assert h._add_marker_click((12, 20, 20))
    balls = h._add_marker_layer
    h._start_add_whole()                          # Whole on the box row
    assert h._add_whole_patch
    # Place markers stays on: markers are placed on a whole patch too
    assert h._add_place_markers_on is True
    assert h.adv.last("place_markers_on") is not False
    # the markers already placed stay, and stay drawn
    assert marker_zyx(h) == [(12, 20, 20)]
    assert h._add_marker_layer is balls and balls.data.any()
    # a click on the whole patch -- outside the box from before it --
    # places a marker
    assert h._add_marker_click((19, 1, 1))
    assert marker_zyx(h) == [(12, 20, 20), (19, 1, 1)]
    # turning Place markers off and on again is honoured
    h._on_add_place_markers_toggled(False)
    assert h._add_place_markers_on is False
    h._on_add_place_markers_toggled(True)
    assert h._add_place_markers_on is True


def test_whole_patch_with_no_box_keeps_place_markers(monkeypatch):
    """Click markers picked and Place markers on, no box: Whole patch
    leaves Place markers on (a run source never has it on: a source change
    turns it off)."""
    install_fakes(monkeypatch)
    hh = Harness(np.zeros((12, 20, 20), np.int32),
                 images={"nuc": np.zeros((12, 20, 20), np.float32)},
                 spacing=(0.5, 0.25, 0.25))
    hh._add_restore_preset = lambda: None
    hh.adv.preset = "cells"
    hh.adv.source = CLICKS
    hh._add_place_markers_on = True
    hh._start_add_whole()
    assert hh._add_running and hh._add_whole_patch
    # the whole patch no longer turns Place markers off
    assert hh._add_place_markers_on is True
    assert hh.adv.last("place_markers_on") is not False
    hh._reject_add()


def test_a_deferred_click_after_whole_lands(h):
    pending = []
    h._add_defer = pending.append
    h._add_defer_marker_click(getattr(h, "_add_drag_token", 0),
                              (12, 20, 20), False)
    assert pending, "the click waits for the next event-loop turn"
    h._start_add_whole()
    for fn in pending:
        fn()
    # the whole patch takes markers: the click is placed, not dropped
    assert marker_zyx(h) == [(12, 20, 20)]


@pytest.mark.parametrize("source", [RUN, NUCLEI, CLICKS])
def test_click_points_only_for_the_clicks_source(h, source):
    from napari.layers import Image

    assert h._add_marker_click((12, 20, 20))
    h._add_info = AP.AddBoxInfo(empty_share=1.0, run_count=(3, 3))
    h.adv.params["dark_outside"] = False
    h.adv.wall["channel"] = "mem"
    h.adv.nuclei_channel = "nuc"
    h.viewer.layers.append(Image(np.random.rand(20, 30, 30), name="mem"))
    h.viewer.layers.append(Image(np.random.rand(20, 30, 30), name="nuc"))
    h.adv.source = source
    if source[0] == AP.SOURCE_RUN:
        h._add_run_guide = np.zeros((20, 30, 30), np.int32)
    kw, why = h._add_gather_preview()
    assert why is None, why
    if source == CLICKS:
        assert kw["click_points"] == [(12, 20, 20)]
        assert kw["reserved_ids"] == [8]
    else:
        assert not kw.get("click_points"), "hidden markers are not sent"
        assert not kw.get("reserved_ids")


# --------------------------------------------------------------------------- #
# a blank From combo on a whole patch
# --------------------------------------------------------------------------- #
def test_blank_run_reads_nothing(widget, monkeypatch, tmp_path):
    hh, _cells = _cells_page_with_a_run(widget, monkeypatch, tmp_path)
    try:
        hh._start_add()
        widget.add_marker_source_combo.setCurrentIndex(1)     # nuclei masks
        hh._start_add_whole()
        assert widget.get_add_marker_source() == NUCLEI
        # a blank From on the whole patch (a run is never picked for the
        # user there): nothing is read
        combo = widget.add_marker_source_combo
        combo.blockSignals(True)
        try:
            combo.setCurrentIndex(-1)
        finally:
            combo.blockSignals(False)
        hh._on_add_source_changed(widget.get_add_marker_source())
        assert widget.get_add_marker_source() == BLANK
        assert hh._add_run_guide is None and hh._add_run_state is None
        loads = []
        monkeypatch.setattr(hh, "_add_load_run_guide", loads.append,
                            raising=False)
        hh._add_ensure_run_guide()
        _land_info(hh)
        hh._add_refresh_ui()
        assert loads == [], "a blank combo reads no run"
        assert widget._add_run_markers is None
        assert widget.add_run_count_label.isHidden()
        assert widget.add_status_label.text() == AP.ADD_S1_WHOLE_PICK_RUN
        assert not widget.add_preview_btn.isEnabled()
    finally:
        hh._reject_add()
        _leave(widget)


# --------------------------------------------------------------------------- #
# the Auto line: only while something is measured
# --------------------------------------------------------------------------- #
def _nuclei_page(widget):
    _leave(widget)
    widget.refresh_add_image_combo(["membrane", "nuclei"])
    widget.set_add_preset("nuclei")
    widget.add_channel_combo.setCurrentIndex(
        widget.add_channel_combo.findText("nuclei"))


def _brightness(widget):
    lb = widget._add_threshold_help
    return None if lb.isHidden() else lb.text()


def _markers(widget):
    return widget._add_source_help.text()


def _page_brightness(widget, pushes):
    _nuclei_page(widget)
    for kw in pushes:
        widget.update_add_ui_state(**kw)
    return _brightness(widget)


_BOX = dict(box_active=True, whole_patch=False)


def _cells_nuclei_page(widget):
    _cells_box(widget)
    widget.add_marker_source_combo.setCurrentIndex(1)         # nuclei masks
    widget.add_nuclei_channel_combo.setCurrentIndex(
        widget.add_nuclei_channel_combo.findText("nuclei"))


def _page_markers(widget, pushes):
    _cells_nuclei_page(widget)
    for kw in pushes:
        widget.update_add_ui_state(**kw)
    return _markers(widget)


def _orch_nuclei(widget, monkeypatch):
    """The real orchestrator on the real page, Nuclei ▸ Anywhere."""
    install_fakes(monkeypatch)
    ch, cells = _world()
    hh = Harness(np.zeros(cells.shape, np.int32), images=ch)
    hh.adv = widget
    hh.ctx["spacing_source"] = "qc_settings"
    hh._add_session_memory = {"last_preset": "nuclei"}
    _nuclei_page(widget)
    return hh


def _orch_cells_nuclei(widget, monkeypatch, *, flat=False):
    """The real orchestrator on the real page, Cells ▸ Nuclei masks; the
    nuclei channel flat (one brightness everywhere) when asked."""
    install_fakes(monkeypatch)
    ch, cells = _world()
    if flat:
        ch["nuclei"] = np.full(cells.shape, 3.0, np.float32)
    hh = Harness(np.zeros(cells.shape, np.int32), images=ch)
    hh.adv = widget
    hh.ctx["spacing_source"] = "qc_settings"
    hh._add_session_memory = {"last_preset": "cells"}
    hh._add_marker_source_entries = lambda: list(ENTRIES)
    _cells_box(widget)
    widget.update_add_ui_state(box_active=False)
    widget.add_wall_channel_combo.setCurrentIndex(
        widget.add_wall_channel_combo.findText("membrane"))
    widget.add_nuclei_channel_combo.setCurrentIndex(
        widget.add_nuclei_channel_combo.findText("nuclei"))
    widget.add_dark_outside_cb.setChecked(False)
    monkeypatch.setattr(widget, "_on_add_source_cb",
                        hh._on_add_source_changed, raising=False)
    hh._start_add()
    widget.add_marker_source_combo.setCurrentIndex(1)         # nuclei masks
    assert widget.get_add_marker_source() == NUCLEI
    return hh


def _end_without_answer(hh, monkeypatch, how, read):
    """The box's first measurement lands; an input changes (pending again);
    the next measurement fails or is cancelled and nothing relaunches it."""
    def _raise(**_kw):
        raise RuntimeError("no memory")

    def _cancel(**_kw):
        return AP.AddBoxInfo(status="cancelled")

    _land_info(hh)
    hh._add_drop_info()
    assert read(hh.adv) == AP.ADD_AUTO_PENDING
    monkeypatch.setattr(AP, "add_box_info",
                        _raise if how == "error" else _cancel)
    hh._request_add_box_info()
    assert hh._add_info_thread is not None
    _land_info(hh)
    assert hh._add_info is None, "nothing landed"
    assert hh._add_info_thread is None, "and nothing was relaunched"


def _case_b_no_box(widget, _mp):
    assert _page_brightness(widget, [dict(box_active=False)]) == (
        AP.ADD_AUTO_NEEDS_BOX)


def _case_b_no_box_after_a_box(widget, _mp):
    assert _page_brightness(widget, [dict(_BOX, brightness_auto=NAN),
                                     dict(box_active=False)]) == (
        AP.ADD_AUTO_NEEDS_BOX)


def _case_b_whole(widget, _mp):
    # Anywhere is measured on a whole patch as on a box
    assert _page_brightness(widget, [dict(box_active=True, whole_patch=True,
                                          brightness_auto=NAN)]) == (
        AP.ADD_AUTO_PENDING)


def _case_b_measuring(widget, _mp):
    assert _page_brightness(widget, [dict(_BOX, brightness_auto=NAN)]) == (
        AP.ADD_AUTO_PENDING)


def _case_b_flat(widget, _mp):
    assert _page_brightness(widget, [
        dict(_BOX, brightness_auto=NAN),
        dict(brightness_auto=NAN, brightness_flat=True)]) == (
        AP.ADD_AUTO_NOTHING_TO_SPLIT)
    assert not widget.add_otsu_btn.isEnabled()


def _case_b_landed(widget, _mp):
    assert _page_brightness(widget, [
        dict(_BOX, brightness_auto=NAN),
        dict(brightness_auto=0.4, brightness_flat=False)]) == (
        AP.ADD_HELP_THRESHOLD)
    assert widget.add_otsu_btn.isEnabled()


def _case_b_candidates(widget, _mp):
    assert _page_brightness(widget, [
        dict(_BOX, brightness_auto=0.4, candidate_ids=[3],
             candidate_sizes={3: 10})]) is None


def _case_b_orch_no_box(widget, mp):
    hh = _orch_nuclei(widget, mp)
    try:
        hh._start_add()
        hh._add_drop_info()
        assert _brightness(widget) == AP.ADD_AUTO_PENDING
        hh._add_secondary()                                   # Close box
        assert not hh._add_running
        assert _brightness(widget) == AP.ADD_AUTO_NEEDS_BOX
    finally:
        hh._reject_add()
        _leave(widget)


def _case_b_orch_whole(widget, mp):
    hh = _orch_nuclei(widget, mp)
    try:
        hh._start_add_whole()
        assert hh._add_whole_patch
        # Anywhere is measured on a whole patch as on a box
        assert _brightness(widget) == AP.ADD_AUTO_PENDING
    finally:
        hh._reject_add()
        _leave(widget)


def _case_b_orch_ended(widget, mp, how):
    hh = _orch_nuclei(widget, mp)
    try:
        hh._start_add()
        _end_without_answer(hh, mp, how, _brightness)
        assert _brightness(widget) == AP.ADD_HELP_THRESHOLD
        assert not widget.add_otsu_btn.isEnabled(), "no answer: Auto grey"
    finally:
        hh._reject_add()
        _leave(widget)


def _case_m_no_channel(widget, _mp):
    _cells_nuclei_page(widget)
    widget.add_nuclei_channel_combo.setCurrentIndex(-1)
    widget.update_add_ui_state(nuclei_auto=NAN)
    assert _markers(widget) == AP.ADD_AUTO_NEEDS_CHANNEL


def _case_m_no_box(widget, _mp):
    assert _page_markers(widget, [dict(nuclei_auto=NAN),
                                  dict(box_active=False)]) == (
        AP.ADD_HELP_NUCLEI_MASKS)


def _case_m_whole_belt(widget, mp):
    _cells_nuclei_page(widget)
    widget.update_add_ui_state(nuclei_auto=NAN)
    # the nuclei masks are measured on a whole patch as on a box
    widget.update_add_ui_state(whole_patch=True)
    assert widget.get_add_marker_source() == NUCLEI
    assert _markers(widget) == AP.ADD_AUTO_PENDING


def _case_m_measuring(widget, _mp):
    assert _page_markers(widget, [dict(nuclei_auto=NAN)]) == (
        AP.ADD_AUTO_PENDING)


def _case_m_flat(widget, _mp):
    assert _page_markers(widget, [dict(nuclei_auto=NAN),
                                  dict(nuclei_flat=True)]) == (
        AP.ADD_AUTO_NOTHING_TO_SPLIT)


def _case_m_landed(widget, _mp):
    assert _page_markers(widget, [dict(nuclei_auto=NAN),
                                  dict(nuclei_auto=0.3)]) == (
        AP.ADD_HELP_NUCLEI_MASKS)


def _case_m_orch_flat_crop(widget, mp):
    hh = _orch_cells_nuclei(widget, mp, flat=True)
    try:
        assert _markers(widget) == AP.ADD_AUTO_PENDING
        hh._request_add_box_info()
        _land_info(hh)
        assert hh._add_info is not None and hh._add_info.nuclei_auto is None
        assert _markers(widget) == AP.ADD_AUTO_NOTHING_TO_SPLIT
    finally:
        hh._reject_add()
        _leave(widget)


def _case_m_orch_landed(widget, mp):
    hh = _orch_cells_nuclei(widget, mp)
    try:
        hh._request_add_box_info()
        _land_info(hh)
        assert hh._add_info.nuclei_auto is not None
        assert _markers(widget) == AP.ADD_HELP_NUCLEI_MASKS
    finally:
        hh._reject_add()
        _leave(widget)


def _case_m_orch_ended(widget, mp, how):
    hh = _orch_cells_nuclei(widget, mp)
    try:
        _end_without_answer(hh, mp, how, _markers)
        assert _markers(widget) == AP.ADD_HELP_NUCLEI_MASKS
    finally:
        hh._reject_add()
        _leave(widget)


def _case_m_orch_relaunch_queued(widget, mp):
    """An answer with no Auto lands while another measurement is queued
    (an input changed while it ran): not flat, not ended -- still
    measuring until the queued one lands."""
    hh = _orch_cells_nuclei(widget, mp)
    real = AP.add_box_info
    calls = []

    def _first_without_auto(**kw):
        calls.append(1)
        if len(calls) == 1:
            return AP.AddBoxInfo(empty_share=1.0)
        return real(**kw)

    try:
        _land_info(hh)
        hh._add_drop_info()
        mp.setattr(AP, "add_box_info", _first_without_auto)
        hh._request_add_box_info()
        th = hh._add_info_thread
        hh._add_info_dirty = True              # a change queued a relaunch
        th.run()                               # the first answer lands
        assert hh._add_info is not None and hh._add_info.nuclei_auto is None
        assert _markers(widget) == AP.ADD_AUTO_PENDING
        th.end()                               # ... and the relaunch runs
        _land_info(hh)
        assert len(calls) == 2
        assert hh._add_info.nuclei_auto is not None
        assert _markers(widget) == AP.ADD_HELP_NUCLEI_MASKS
    finally:
        hh._reject_add()
        _leave(widget)


def _case_b_blank_channel(widget, _mp):
    _nuclei_page(widget)
    widget._set_add_channel_blank(widget.add_channel_combo)
    widget.update_add_ui_state(box_active=True, whole_patch=False,
                               brightness_auto=NAN)
    # nothing can be measured without a channel: the normal help line
    assert _brightness(widget) == AP.ADD_HELP_THRESHOLD


class _Timer:
    def __init__(self, active):
        self.active = active

    def isActive(self):                  # noqa: N802 - Qt name
        return self.active

    def stop(self):
        self.active = False


def _case_b_orch_timer_due(widget, mp):
    """A measurement that ends without an answer while the debounced
    request is due: the line keeps measuring (the next one is coming)."""
    hh = _orch_nuclei(widget, mp)
    try:
        hh._start_add()
        _land_info(hh)
        hh._add_drop_info()
        mp.setattr(AP, "add_box_info",
                   lambda **_k: AP.AddBoxInfo(status="cancelled"))
        hh._request_add_box_info()
        hh._add_box_timer = _Timer(True)
        _land_info(hh)
        assert hh._add_info is None
        assert _brightness(widget) == AP.ADD_AUTO_PENDING
        hh._add_box_timer = _Timer(False)
    finally:
        hh._reject_add()
        _leave(widget)


_AUTO_CASES = {
    "brightness-blank-channel": _case_b_blank_channel,
    "brightness-orchestrator-timer-due": _case_b_orch_timer_due,
    "markers-orchestrator-relaunch-queued": _case_m_orch_relaunch_queued,
    "brightness-no-box": _case_b_no_box,
    "brightness-no-box-after-a-box": _case_b_no_box_after_a_box,
    "brightness-whole-patch": _case_b_whole,
    "brightness-measuring": _case_b_measuring,
    "brightness-flat": _case_b_flat,
    "brightness-landed": _case_b_landed,
    "brightness-candidates": _case_b_candidates,
    "brightness-orchestrator-close-box": _case_b_orch_no_box,
    "brightness-orchestrator-whole-patch": _case_b_orch_whole,
    "brightness-orchestrator-error": (
        lambda w, mp: _case_b_orch_ended(w, mp, "error")),
    "brightness-orchestrator-cancelled": (
        lambda w, mp: _case_b_orch_ended(w, mp, "cancelled")),
    "markers-no-channel": _case_m_no_channel,
    "markers-no-box": _case_m_no_box,
    "markers-whole-patch": _case_m_whole_belt,
    "markers-measuring": _case_m_measuring,
    "markers-flat": _case_m_flat,
    "markers-landed": _case_m_landed,
    "markers-orchestrator-flat-crop": _case_m_orch_flat_crop,
    "markers-orchestrator-landed": _case_m_orch_landed,
    "markers-orchestrator-error": (
        lambda w, mp: _case_m_orch_ended(w, mp, "error")),
    "markers-orchestrator-cancelled": (
        lambda w, mp: _case_m_orch_ended(w, mp, "cancelled")),
}


@pytest.mark.parametrize("case", sorted(_AUTO_CASES))
def test_auto_line(widget, monkeypatch, case):
    try:
        _AUTO_CASES[case](widget, monkeypatch)
    finally:
        _leave(widget)


def test_the_auto_line_words():
    assert AP.ADD_AUTO_NEEDS_BOX == "Auto is measured once a box is placed."


# --------------------------------------------------------------------------- #
# the box's pick falls back to the default when it is gone
# --------------------------------------------------------------------------- #
def test_a_dropped_run_falls_back_to_the_default(widget, monkeypatch):
    """A run picked on the whole patch leaves the list while the box is
    whole: back on a box the default is picked (not a blank combo, which a
    box reads as clicks), silently, and the dropped run's count is gone."""
    _cells_box(widget)
    combo = widget.add_marker_source_combo
    try:
        combo.setCurrentIndex(2)                              # clicks
        widget.update_add_ui_state(whole_patch=True)
        combo.setCurrentIndex(0)                              # the run
        widget.update_add_ui_state(run_count=(7, 5))
        assert widget._add_run_markers == 5
        widget.set_add_marker_sources(NO_RUN_ENTRIES)         # run gone
        assert widget._add_run_markers is None, "no count for no run"
        calls = []
        monkeypatch.setattr(widget, "_on_add_source_cb", calls.append,
                            raising=False)
        widget.clear_add_touched()
        widget.update_add_ui_state(whole_patch=False)
        assert combo.currentIndex() >= 0, "a box never shows a blank From"
        assert widget.get_add_marker_source() == NUCLEI
        assert calls == []
        assert "add_marker_source_combo" not in widget._add_touched
        assert widget._add_click_rows.isHidden()
    finally:
        _leave(widget)


def test_a_greyed_box_pick_falls_back_to_the_default(widget, monkeypatch):
    _cells_box(widget)
    combo = widget.add_marker_source_combo
    try:
        combo.setCurrentIndex(1)                              # nuclei masks
        widget.update_add_ui_state(whole_patch=True)
        grey = [ENTRIES[0], (AP.ADD_ITEM_NUCLEI_GREY, False, NUCLEI),
                ENTRIES[2]]
        widget.set_add_marker_sources(grey)
        calls = []
        monkeypatch.setattr(widget, "_on_add_source_cb", calls.append,
                            raising=False)
        widget.clear_add_touched()
        widget.update_add_ui_state(whole_patch=False)
        assert combo.currentIndex() == 2, "the default: clicks"
        assert widget.get_add_marker_source() == CLICKS
        assert calls == []
        assert "add_marker_source_combo" not in widget._add_touched
    finally:
        _leave(widget)


def test_a_closed_whole_patch_does_not_filter_the_next_box(
        widget, monkeypatch, tmp_path):
    hh, _cells = _cells_page_with_a_run(widget, monkeypatch, tmp_path)
    try:
        hh._start_add_whole()
        assert _texts(widget) == [t for t, _e, _k in ENTRIES]
        hh._add_secondary()                                   # Close box
        assert not hh._add_running
        assert widget._add_ui.get("whole_patch") is False
        widget.update_add_ui_state(box_active=True)
        assert _texts(widget) == [t for t, _e, _k in ENTRIES]
    finally:
        hh._reject_add()
        _leave(widget)


# --------------------------------------------------------------------------- #
# a Clear never leaves a stale zero-markers line behind
# --------------------------------------------------------------------------- #
def _zero_run_lines():
    return [v for k, v in vars(AP).items()
            if k.startswith(("ADD_S1_RUN_NO", "ADD_S1_RUN_NONE"))
            and isinstance(v, str)]


def test_a_clear_drops_the_stale_zero_markers_line(monkeypatch):
    from napari.layers import Image

    from .test_add_clear import _clear, _harness

    install_fakes(monkeypatch)
    vol = np.zeros((8, 16, 12), np.int32)
    vol[1:5, 2:8, 2:8] = 3                 # one solid object, in the box
    run = vol.copy()                       # the run's object is kept here
    h = _harness(vol)
    h.viewer.layers.append(Image(np.random.rand(*vol.shape), name="mem"))
    h.adv.preset = "cells"
    h.adv.source = ("labels", "r")
    h.adv.params["dark_outside"] = False
    h.adv.wall["channel"] = "mem"
    h._add_run_sources = {"r": object()}
    h._add_run_guide = run
    # the box was measured before the Clear: every run object over kept
    h._add_info = AP.AddBoxInfo(empty_share=0.5, run_count=(1, 0))
    h._add_refresh_ui()
    dark = getattr(AP, "ADD_DARK_OUTSIDE_LABEL", "")
    assert h.adv.last("status_text") in [
        s.format(dark=dark) for s in _zero_run_lines()]
    assert h.adv.last("preview_enabled") is False
    _clear(h)
    s1 = h._add_transient or ""
    assert s1.startswith(AP.count_phrase(1, AP.ADD_S1_CLEARED_ONE,
                                         AP.ADD_S1_CLEARED))
    assert not any(z[:30] in s1 for z in _zero_run_lines()), s1
    _land_info(h)
    assert h._add_info is not None and h._add_info.run_count[1] > 0
    h._add_refresh_ui()
    assert not any(z[:30] in (h.adv.last("status_text") or "")
                   for z in _zero_run_lines())
    assert h.adv.last("preview_enabled") is True


# --------------------------------------------------------------------------- #
# one rule for what a whole patch measures
# --------------------------------------------------------------------------- #
_RULE_STATES = [
    # (preset, From pick on the box, Look for nuclei)
    ("cells", 0, None),            # a run: kept on the whole patch
    ("cells", 1, None),            # nuclei masks: kept on the whole patch
    ("cells", 2, None),            # Click markers: kept on the whole patch
    ("nuclei", 1, "anywhere"),
    ("nuclei", 1, "inside"),
    ("nuclei", 1, "markers"),
]


@pytest.mark.parametrize("preset,pick,look", _RULE_STATES)
def test_one_whole_patch_rule(widget, monkeypatch, tmp_path, preset, pick,
                              look):
    """The box-info gather and the page's Auto lines ask the one rule with
    the same inputs on a whole patch, so they give the same answer."""
    assert AP.whole_measures_nothing("cells", AP.SOURCE_RUN,
                                     AP.LOOK_ANYWHERE) is False
    assert AP.whole_measures_nothing("cells", AP.SOURCE_NUCLEI,
                                     AP.LOOK_INSIDE) is False
    assert AP.whole_measures_nothing("cells", "other",
                                     AP.LOOK_INSIDE) is True
    assert AP.whole_measures_nothing("nuclei", AP.SOURCE_RUN,
                                     AP.LOOK_ANYWHERE) is False
    assert AP.whole_measures_nothing("nuclei", AP.SOURCE_CLICKS,
                                     AP.LOOK_INSIDE) is False
    # markers are measured on a whole patch: Click markers on Cells, the
    # markers on Nuclei
    assert AP.whole_measures_nothing("cells", AP.SOURCE_CLICKS,
                                     AP.LOOK_ANYWHERE) is False
    assert AP.whole_measures_nothing("nuclei", AP.SOURCE_RUN,
                                     AP.LOOK_MARKERS) is False
    real = AP.whole_measures_nothing
    seen = []

    def _spy(*a):
        seen.append((a, real(*a)))
        return real(*a)

    hh, _cells = _cells_page_with_a_run(widget, monkeypatch, tmp_path)
    try:
        hh._start_add()
        widget.add_marker_source_combo.setCurrentIndex(pick)
        if preset == "nuclei":
            widget.set_add_look_for_items([
                (AP.ADD_LOOK_INSIDE.format(name="cells", target=""), True,
                 ("inside", "cells"))])
            keys = list(widget._add_look_keys)
            want = {"inside": ("inside", "cells"),
                    "markers": (AP.LOOK_MARKERS, None)}.get(look)
            widget.add_look_for_combo.setCurrentIndex(
                keys.index(want) if want is not None else 0)
            widget.add_preset_combo.setCurrentIndex(0)        # Nuclei
            assert widget.get_add_preset() == "nuclei"
        hh._start_add_whole()
        assert hh._add_whole_patch
        monkeypatch.setattr(AP, "whole_measures_nothing", _spy)
        hh._add_gather_info()
        gather = list(seen)
        seen.clear()
        widget.update_add_ui_state(status_text="x")
        page = list(seen)
        assert gather, "the gather asks the rule"
        assert page, "the page's Auto lines ask the rule"
        assert {g for g in gather} == {p for p in page}, (gather, page)
    finally:
        monkeypatch.setattr(AP, "whole_measures_nothing", real)
        hh._reject_add()
        _leave(widget)


# --------------------------------------------------------------------------- #
# Place markers and the first line
# --------------------------------------------------------------------------- #
def test_place_markers_toggle_refreshes_s1(widget, monkeypatch, tmp_path):
    hh, _cells = _cells_page_with_a_run(widget, monkeypatch, tmp_path)
    btn = widget.add_place_markers_btn
    try:
        hh._start_add()
        widget.add_marker_source_combo.setCurrentIndex(2)     # clicks
        assert widget.add_status_label.text() == AP.ADD_S1_NO_MARKERS
        btn.toggled.connect(hh._on_add_place_markers_toggled)
        btn.setChecked(True)
        assert hh._add_place_markers_on
        assert widget.add_status_label.text() == AP.ADD_S1_NO_MARKERS_ON
        btn.setChecked(False)
        assert widget.add_status_label.text() == AP.ADD_S1_NO_MARKERS
    finally:
        try:
            btn.toggled.disconnect(hh._on_add_place_markers_toggled)
        except (TypeError, RuntimeError):
            pass
        hh._reject_add()
        _leave(widget)


# =========================================================================== #
# The ball beside Place markers, the Clear line, the channel colours, and
# the box that comes back after a whole patch
# =========================================================================== #
def _shown_on_page(widget, w):
    """Shown on the Add page once the page is: nothing between *w* and the
    page is hidden (the module's widget itself is never shown)."""
    page = widget._panel_slots["Add"]["page"]
    x = w
    while x is not None and x is not page:
        if x.isHidden():
            return False
        x = x.parentWidget()
    return x is page


def _row_widgets(layout):
    out = []
    for i in range(layout.count()):
        it = layout.itemAt(i)
        if it.widget() is not None:
            out.append(it.widget())
    return out


def test_ball_radius_beside_place_markers(widget):
    from qtpy.QtWidgets import QLabel

    from napari_epicycle.annotator.advanced_features_widget import (
        ADD_HELP_LINES,
    )
    from napari_epicycle.ui_shell.theme import PANEL_ROW_LABEL_STYLE

    sb = widget.add_click_radius_sb
    row = widget._add_click_radius_row
    try:
        _cells_box(widget)
        widget.add_marker_source_combo.setCurrentIndex(2)      # clicks
        # row 1 of the card: Place markers, then the "Ball" row
        assert row.parentWidget() is widget._add_click_rows
        first = widget._add_click_rows.layout().itemAt(0).layout()
        got = _row_widgets(first)
        assert got[:2] == [widget.add_place_markers_btn, row], got
        inner = _row_widgets(row.layout())
        assert len(inner) == 2 and inner[1] is sb, inner
        label = inner[0]
        assert isinstance(label, QLabel)
        assert label.text() == AP.ADD_BALL_LABEL == "Ball"
        assert label.parentWidget() is row
        assert label.styleSheet() == PANEL_ROW_LABEL_STYLE
        assert not widget._add_cells_adv.isAncestorOf(sb)
        # the spin box keeps its range, step and tooltip; no live typing
        assert (sb.minimum(), sb.maximum(), sb.singleStep()) == (0.5, 5.0,
                                                                 0.5)
        assert sb.toolTip() == ADD_HELP_LINES["add_click_radius_sb"]
        assert not sb.keyboardTracking()
        # shown exactly with Click markers on a box (or the whole patch)
        assert _shown_on_page(widget, sb)
        assert _shown_on_page(widget, label)
        # the count is off the card body; the title carries it
        widget.update_add_ui_state(click_count=3)
        assert not _shown_on_page(widget, widget.add_click_count_label)
        assert widget.add_markers_btn.property(
            "_epicycle_chevron_title") == "Markers: clicks · 3 placed"
        # no "ball" in the Advanced title any more
        assert "ball" not in widget._add_advanced_toggle.text().lower()
        for index in (0, 1):                     # a run, nuclei masks
            widget.add_marker_source_combo.setCurrentIndex(index)
            assert not _shown_on_page(widget, sb), index
        widget.add_marker_source_combo.setCurrentIndex(2)
        assert _shown_on_page(widget, sb)
        # Click markers on a whole patch keep their Ball
        widget.update_add_ui_state(whole_patch=True)          # whole patch
        assert _shown_on_page(widget, sb)
        widget.update_add_ui_state(whole_patch=False)
        assert _shown_on_page(widget, sb)
        widget.update_add_ui_state(box_active=False)          # no box
        assert not _shown_on_page(widget, sb)
        widget.update_add_ui_state(click_count=0)
    finally:
        _leave(widget)


def test_ball_radius_defaults_to_one_micron(widget):
    from napari_epicycle.annotator import advanced_features_widget as AFW

    assert AFW._ADD_PRESET_VALUES["cells"]["add_click_radius_sb"] == 1.0
    reset_add(widget)
    try:
        widget.set_add_preset("cells")
        assert widget.add_click_radius_sb.value() == 1.0
        assert widget.get_add_params()["click_radius_um"] == 1.0
        # an untouched value goes back to the preset's 1.0
        widget.add_click_radius_sb.setValue(3.0)
        widget.clear_add_touched()
        widget.set_add_preset("nuclei")
        widget.set_add_preset("cells")
        assert widget.add_click_radius_sb.value() == 1.0
    finally:
        reset_add(widget)


def test_ball_radius_fallbacks_are_one_micron(h, monkeypatch):
    """With no radius in the snapshot every Add path uses 1.0 µm: the
    ball drawn on screen, the Preview's request and the engine call."""
    from napari.layers import Image

    from packages.epicycle_annotator.reshape import region as R

    from .test_reshape_region import ISO, _box, _threshold, _tissue

    h.adv.params.pop("click_radius_um", None)
    # the ball on screen: 1.0 µm at 0.5 µm per voxel is 2 voxels
    assert h._add_ball_radius_vox() == (2.0, 2.0, 2.0)
    # the Preview's request
    assert h._add_marker_click((12, 20, 20))
    h._add_info = AP.AddBoxInfo(empty_share=1.0, run_count=(3, 3))
    h.adv.params["dark_outside"] = False
    h.adv.wall["channel"] = "mem"
    h.adv.nuclei_channel = "nuc"
    h.viewer.layers.append(Image(np.random.rand(20, 30, 30), name="mem"))
    h.viewer.layers.append(Image(np.random.rand(20, 30, 30), name="nuc"))
    h.adv.source = CLICKS
    kw, why = h._add_gather_preview()
    assert why is None, why
    assert kw["params"]["click_radius_um"] == 1.0
    # the engine call and the reserved-id mapping
    seen = {}
    real = R.reshape_empty_region
    real_map = AP._reserved_mapping

    def spy(*a, **k):
        seen["spec"] = a[2]
        return real(*a, **k)

    def spy_map(*a, **k):
        seen["radius_um"] = k.get("radius_um")
        return real_map(*a, **k)

    monkeypatch.setattr(R, "reshape_empty_region", spy)
    monkeypatch.setattr(AP, "_reserved_mapping", spy_map)
    mem, nuc, _cells = _tissue()
    chans = {"membrane": mem, "nuclei": nuc}
    out = AP.run_add_preview(
        preset="cells", roi=_box(mem.shape),
        existing_roi=np.zeros(mem.shape, np.int32), next_label=1000,
        spacing=ISO, params={"source": AP.SOURCE_CLICKS,
                             "dark_outside": False, "auto_smooth": False,
                             "clear_border": False},
        wall_roi=mem, dark_rois=chans, dark_threshold=_threshold(chans),
        click_points=[(6, 5, 5)], reserved_ids=[1000], id_max=2 ** 31 - 1)
    assert out.ok, (out.status, out.refusal)
    assert seen["spec"].params["click_radius_um"] == 1.0
    assert seen["radius_um"] == 1.0


class _FakeTimer:
    """A single-shot timer the test fires by hand."""

    made = []

    def __init__(self, parent=None):
        from .add_harness import FakeSignal

        self.parent = parent
        self.timeout = FakeSignal()
        self.starts = []
        self.active = False
        self.single = None
        _FakeTimer.made.append(self)

    def setSingleShot(self, v):            # noqa: N802 - Qt name
        self.single = bool(v)

    def start(self, ms=None):
        self.starts.append(ms)
        self.active = True

    def stop(self):
        self.active = False

    def isActive(self):                    # noqa: N802 - Qt name
        return self.active

    def fire(self):
        self.active = False
        self.timeout.emit()


def test_radius_change_redraws_once(h, monkeypatch):
    import qtpy.QtCore

    h.adv.params["click_radius_um"] = 1.0
    assert h._add_marker_click((12, 20, 20))
    layer = h._add_marker_layer
    # 1.0 µm at 0.5 µm per voxel: 2 voxels; 5 voxels away is empty
    assert layer.data[12, 20, 20] and not layer.data[12, 20, 25]
    renders = []
    real_render = h._render_add_balls

    def spy(*a, **k):
        renders.append((a, k))
        return real_render(*a, **k)

    h._render_add_balls = spy
    _FakeTimer.made = []
    monkeypatch.setattr(qtpy.QtCore, "QTimer", _FakeTimer)
    pushes = len(h.adv.pushes)
    # holding the arrow key: five changes, one timer, restarted each time
    h.adv.params["click_radius_um"] = 3.0
    for _ in range(5):
        h._on_add_advanced_changed()
    timers = [t for t in _FakeTimer.made if t.parent is h]
    assert len(timers) == 1 and timers[0].single is True
    t = timers[0]
    assert len(t.starts) == 5 and t.active
    assert renders == [], "nothing is redrawn while the key is held"
    t.fire()
    assert len(renders) == 1 and renders[0] == ((), {}), renders
    assert h._add_marker_layer.data[12, 20, 25], "the ball is 3 µm now"
    assert len(h.adv.pushes) > pushes, "the page is refreshed after"
    # the same radius again: no redraw
    h._on_add_advanced_changed()
    t.fire()
    assert len(renders) == 1
    # no markers: the timer does not start, nothing is drawn
    h._add_click_markers = []
    h.adv.params["click_radius_um"] = 1.5
    n_starts = len(t.starts)
    h._on_add_advanced_changed()
    assert len(t.starts) == n_starts
    t.fire()
    assert len(renders) == 1
    # a box end stops it
    assert h._add_marker_click((12, 10, 20))
    h.adv.params["click_radius_um"] = 2.0
    h._on_add_advanced_changed()
    assert t.active
    h._reset_add_state()
    assert not t.active, "the box end stops the ball timer"


def test_a_ball_drawn_at_another_radius_is_redrawn(h, monkeypatch):
    """A marker placed while no ball was drawn at the recorded radius (Undo
    last, then a Ball change with no marker, then a click) is drawn at the
    new radius; going back to the old radius must redraw it."""
    import qtpy.QtCore

    _FakeTimer.made = []
    monkeypatch.setattr(qtpy.QtCore, "QTimer", _FakeTimer)
    h.adv.params["click_radius_um"] = 1.0
    assert h._add_marker_click((12, 20, 20))
    h._add_click_undo()
    assert not h._add_markers()
    h.adv.params["click_radius_um"] = 3.0
    h._on_add_advanced_changed()            # no marker: nothing to redraw
    assert h._add_marker_click((12, 10, 20))
    h.adv.params["click_radius_um"] = 1.0
    h._on_add_advanced_changed()
    for t in [t for t in _FakeTimer.made if t.parent is h]:
        if t.active:
            t.fire()
    data = np.asarray(h._add_marker_layer.data)
    zz, yy, xx = np.nonzero(data)
    sp = np.asarray(h.ctx["spacing"], float)
    dist = np.sqrt(((zz - 12) * sp[0]) ** 2 + ((yy - 10) * sp[1]) ** 2
                   + ((xx - 20) * sp[2]) ** 2)
    assert zz.size, "the ball is drawn"
    assert float(dist.max()) <= 1.0 + 1e-9, (
        "a ball voxel {:.2f} um from its marker at Ball 1.0 um".format(
            float(dist.max())))


def test_clear_hint_in_the_idle_strip(widget):
    hint = AP.ADD_STRIP_CLEAR_HINT
    assert hint == "Clear makes room by deleting old objects."
    label = widget.add_stage_label
    try:
        _cells_box(widget)
        assert not widget.add_clear_btn.isHidden()
        assert label.text() == hint
        assert not label.isEnabled(), "dimmed, as the idle strip is"
        assert widget.add_clear_btn.toolTip() == AP.ADD_CLEAR_TIP
        # a run takes the line; the hint comes back when it ends
        widget.update_add_ui_state(busy=True, stop_enabled=True,
                                   stage_text="Finding markers...")
        assert hint not in label.text()
        assert "Finding markers" in label.toolTip()
        widget.update_add_ui_state(busy=False)
        assert label.text() == hint
        # over a result Clear is hidden, and so is the hint
        widget.update_add_ui_state(candidate_ids=[3])
        assert widget.add_clear_btn.isHidden()
        assert label.text() == ""
        widget.update_add_ui_state(candidate_ids=[])
        assert label.text() == hint
        # the review, and no box: no Clear, no hint
        widget.update_add_ui_state(review_active=True)
        assert label.text() == ""
        widget.update_add_ui_state(review_active=False)
        assert label.text() == hint
        widget.update_add_ui_state(box_active=False)
        assert widget.add_clear_btn.isHidden()
        assert label.text() == ""
    finally:
        _leave(widget)


def test_clear_hint_gives_way_to_measuring_and_waits(widget):
    """The hint and the strip's other words share one line: the measuring
    words and a wait's words take it, the hint comes back when they end,
    and a busy page with no words of its own shows an empty line with no
    tooltip left over."""
    from napari_epicycle.annotator.reshape_worker import stage_text

    hint = AP.ADD_STRIP_CLEAR_HINT
    label = widget.add_stage_label
    try:
        _cells_box(widget)
        assert label.text() == hint and label.toolTip() == hint
        # measuring words take the line, and come off it
        words = stage_text("add_info")
        widget.update_add_ui_state(busy=False, measuring=words)
        assert widget._add_stage_full_text == words
        assert label.toolTip() == words
        widget.update_add_ui_state(busy=False, measuring="")
        assert label.text() == hint and label.toolTip() == hint
        # a main-thread wait's words take the line; the next push ends them
        wait = stage_text("add_clear")
        widget.paint_add_wait(status_text="Clearing", stage_text=wait)
        assert widget._add_stage_full_text == wait
        widget.update_add_ui_state(busy=False)
        assert label.text() == hint
        # busy with no words of its own (Clear counting): empty, no tooltip
        widget.update_add_ui_state(busy=True, stop_enabled=True)
        assert label.text() == "", label.text()
        assert label.toolTip() == "", label.toolTip()
        widget.update_add_ui_state(busy=False)
        assert label.text() == hint and label.toolTip() == hint
    finally:
        _leave(widget)


_FONT_DIR = "C:/Windows/Fonts"

_HINT_CHILD = """
import json, os, sys
os.environ["QT_QPA_PLATFORM"] = "offscreen"
if os.path.isdir(FONT_DIR):
    os.environ["QT_QPA_FONTDIR"] = FONT_DIR
from qtpy.QtGui import QFont
from qtpy.QtWidgets import QApplication
app = QApplication([])
app.setFont(QFont("Segoe UI", 9))
from napari_epicycle.annotator.advanced_features_widget import (
    AdvancedFeaturesWidget)
from packages.epicycle_annotator import add_preview as AP
w = AdvancedFeaturesWidget(mode="3d_curation")
w.resize(340, 700)
w.show()
for _ in range(5):
    app.processEvents()
w.focus_tab("Add")
w.refresh_add_image_combo(["nuc", "mem"])
w.set_add_preset("cells")
w.update_add_ui_state(box_active=True)
for _ in range(10):
    app.processEvents()
lb = w.add_stage_label
out = {"font": app.font().family(), "shown": lb.text(),
       "need": lb.fontMetrics().horizontalAdvance(AP.ADD_STRIP_CLEAR_HINT),
       "room": lb.contentsRect().width(), "dock": w.width()}
print("HINTJSON" + json.dumps(out), flush=True)
os._exit(0)
"""


def test_clear_hint_fits_the_strip_with_real_fonts():
    """Measured with real fonts in a child process (offscreen Qt here has
    no fonts at all): the hint is drawn whole, not elided, at the dock's
    width."""
    import json
    import os
    import pathlib
    import subprocess
    import sys

    repo = pathlib.Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUTF8"] = "1"
    code = "FONT_DIR = {!r}\n".format(_FONT_DIR) + _HINT_CHILD
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(repo),
                          env=env, capture_output=True, text=True,
                          timeout=300)
    lines = [ln for ln in proc.stdout.splitlines()
             if ln.startswith("HINTJSON")]
    assert lines, (proc.returncode, proc.stdout[-1500:], proc.stderr[-3000:])
    got = json.loads(lines[-1][len("HINTJSON"):])
    print("[ADD-HINT] {}".format(json.dumps(got)))
    assert got["shown"] == AP.ADD_STRIP_CLEAR_HINT, got
    assert got["need"] <= got["room"], got


_BALL_CHILD = """
import json, os, sys
os.environ["QT_QPA_PLATFORM"] = "offscreen"
if os.path.isdir(FONT_DIR):
    os.environ["QT_QPA_FONTDIR"] = FONT_DIR
from qtpy.QtGui import QFont
from qtpy.QtWidgets import QApplication
app = QApplication([])
app.setFont(QFont("Segoe UI", 9))
from napari_epicycle.annotator.advanced_features_widget import (
    AdvancedFeaturesWidget)
from packages.epicycle_annotator import add_preview as AP
w = AdvancedFeaturesWidget(mode="3d_curation")
w.resize(340, 900)
w.show()
for _ in range(5):
    app.processEvents()
w.focus_tab("Add")
w.refresh_add_image_combo(["nuc", "mem"])
w.set_add_preset("cells")
items = [(AP.ADD_ITEM_RUN.format(name="r") + AP.ADD_ITEM_RAW_SUFFIX, True,
          ("labels", "r")),
         (AP.ADD_ITEM_NUCLEI, True, ("nuclei_masks", None)),
         (AP.ADD_ITEM_CLICKS, True, ("paint", None))]
w.update_add_ui_state(marker_sources=items, box_active=True,
                      voxel_size_known=True)
w.add_markers_btn.setChecked(True)
w.add_marker_source_combo.setCurrentIndex(2)
for _ in range(20):
    app.processEvents()
row = w._add_click_radius_row
sb = w.add_click_radius_sb
le = sb.lineEdit()
out = {"font": app.font().family(), "visible": sb.isVisible(),
       "row_w": row.width(), "sb_x": sb.x(), "sb_w": sb.width(),
       "sb_hint_w": sb.sizeHint().width(), "text": sb.text(),
       "text_need": le.fontMetrics().horizontalAdvance(sb.text()),
       "text_room": le.contentsRect().width(),
       "content_min_w":
           w._panel_slots["Add"]["content"].minimumSizeHint().width(),
       "dock_min_w": w.minimumSizeHint().width()}
print("BALLJSON" + json.dumps(out), flush=True)
os._exit(0)
"""

#: the Add page's minimum widths with real fonts before the Ball row was
#: given room (the row's fix must not widen the page)
_BALL_CONTENT_MIN_W = 267
_BALL_DOCK_MIN_W = 241


def test_ball_spin_fits_its_row_with_real_fonts():
    """Measured with real fonts in a child process: the Ball spin box lies
    inside its own row (its arrows are not cut) and shows its whole value
    with " µm", and the page is no wider for it."""
    import json
    import os
    import pathlib
    import subprocess
    import sys

    repo = pathlib.Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUTF8"] = "1"
    code = "FONT_DIR = {!r}\n".format(_FONT_DIR) + _BALL_CHILD
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(repo),
                          env=env, capture_output=True, text=True,
                          timeout=300)
    lines = [ln for ln in proc.stdout.splitlines()
             if ln.startswith("BALLJSON")]
    assert lines, (proc.returncode, proc.stdout[-1500:], proc.stderr[-3000:])
    got = json.loads(lines[-1][len("BALLJSON"):])
    print("[ADD-BALL] {}".format(json.dumps(got)))
    assert got["visible"], got
    assert got["text"].endswith(" µm"), got
    assert got["sb_x"] + got["sb_w"] <= got["row_w"], got
    assert got["sb_w"] >= got["sb_hint_w"], got
    assert got["text_need"] <= got["text_room"], got
    assert got["content_min_w"] <= _BALL_CONTENT_MIN_W, got
    assert got["dock_min_w"] <= _BALL_DOCK_MIN_W, got


def _swatch_rgb(icon):
    img = icon.pixmap(48, 48).toImage()
    c = img.pixelColor(img.width() // 2, img.height() // 2)
    return (c.red(), c.green(), c.blue())


def test_channel_items_show_layer_colours(monkeypatch, qapp):
    from napari_epicycle.annotator.advanced_features_widget import (
        AdvancedFeaturesWidget,
    )

    install_fakes(monkeypatch)
    vol = np.zeros((12, 20, 20), np.int32)
    hh = Harness(vol, images={"nuc": np.random.rand(12, 20, 20),
                              "mem": np.random.rand(12, 20, 20)},
                 spacing=(0.5, 0.5, 0.5))
    hh.viewer.layers[0].colormap = "magenta"
    hh.viewer.layers[1].colormap = "green"
    want = {}
    for layer in hh.viewer.layers:
        rgb = np.asarray(layer.colormap.colors)[-1][:3]
        want[layer.name] = tuple(int(round(float(v) * 255)) for v in rgb)
    assert want["nuc"] != want["mem"]
    panel = AdvancedFeaturesWidget(mode="3d_curation")
    try:
        # without colours (the old call) the items carry no swatch
        panel.refresh_add_image_combo(["nuc", "mem"])
        assert panel.add_channel_combo.itemIcon(0).isNull()
        hh.adv = panel
        hh._refresh_add_image_combo()
        for attr in ("add_channel_combo", "add_nuclei_channel_combo",
                     "add_wall_channel_combo"):
            combo = getattr(panel, attr)
            names = [combo.itemText(i) for i in range(combo.count())]
            assert names == ["nuc", "mem"], (attr, names)
            for i, name in enumerate(names):
                icon = combo.itemIcon(i)
                assert not icon.isNull(), (attr, name)
                got = _swatch_rgb(icon)
                assert all(abs(a - b) <= 2
                           for a, b in zip(got, want[name])), (
                    attr, name, got, want[name])
        # the text stays the layer name: every currentText() reader is
        # untouched
        panel.add_wall_channel_combo.setCurrentIndex(1)
        assert panel.add_wall_channel_combo.currentText() == "mem"
        assert panel.add_channel_combo.currentText() in ("nuc", "mem")
    finally:
        panel.setParent(None)
        panel.deleteLater()


# --------------------------------------------------------------------------- #
# leaving a whole patch: the box comes back
# --------------------------------------------------------------------------- #
#: an off-centre box on the (20, 30, 30) patch of the ``h`` fixture, of
#: exactly the L size (half of each axis: 10 x 15 x 15)
_OLD = (slice(8, 18), slice(14, 29), slice(1, 16))


def _box_tuple(b):
    return (b.cz, b.cy, b.cx, b.sz, b.sy, b.sx)


def test_leaving_whole_returns_to_the_old_box(h):
    h.place_box(_OLD)
    old = h._add_box
    assert (old.sz, old.sy, old.sx) == (10, 15, 15)
    assert h._add_marker_click((13, 21, 8))       # the box's centre
    assert h._add_marker_click((9, 15, 2))        # near its corner
    h._start_add_whole()
    assert h._add_whole_patch
    # leaving a whole patch that holds markers asks first; Go back keeps
    # the whole patch and both markers
    asked = []
    h._add_leave_whole_confirm = lambda text: asked.append(text) or False
    h._set_add_box_size_fraction(0.5)             # L: the same size
    assert asked == [AP.ADD_LEAVE_WHOLE_CONFIRM.format(n=2)]
    assert h._add_whole_patch
    assert marker_zyx(h) == [(13, 21, 8), (9, 15, 2)], "both kept"
    # Remove them: exactly that box, every marker removed, and S1 says so
    h._add_leave_whole_confirm = lambda text: True
    h._set_add_box_size_fraction(0.5)
    assert not h._add_whole_patch
    assert _box_tuple(h._add_box) == _box_tuple(old), "exactly that box"
    assert marker_zyx(h) == []
    assert h._add_transient == AP.ADD_S1_WHOLE_MARKERS_CLEARED
    assert h._add_roi == old.bounds_slices((20, 30, 30))
    # another size: centred on the old box's centre (no marker: nothing
    # is asked)
    h._start_add_whole()
    asked.clear()
    h._add_leave_whole_confirm = lambda text: asked.append(text) or False
    h._set_add_box_size_fraction(0.25)            # M: 5 x 8 x 8
    assert asked == []
    b = h._add_box
    assert (b.cz, b.cy, b.cx) == (old.cz, old.cy, old.cx)
    assert (b.sz, b.sy, b.sx) == (5, 8, 8)


def test_leaving_whole_by_numbers_keeps_todays_box(h):
    h.place_box(_OLD)
    h._start_add_whole()
    whole = h._add_box
    # a Numbers... slider: the numbers the user typed, as today
    h._on_add_box_changed(whole.cz, whole.cy, whole.cx, 4, 4, 4)
    assert not h._add_whole_patch
    assert _box_tuple(h._add_box) == (whole.cz, whole.cy, whole.cx, 4, 4, 4)


def test_the_old_box_is_forgotten_at_every_box_end(h):
    h.place_box(_OLD)
    h._start_add_whole()
    assert h._add_box_before_whole is not None
    h._reset_add_state()                  # Close box / patch / run / close
    assert h._add_box_before_whole is None
    h.place_box(_OLD)
    h._start_add_whole()
    assert h._add_box_before_whole is not None
    # a box opened by "+ Add another" (the session holds one Accept)
    h._add_session = [{"ids": [9]}]
    h._add_resume_review()
    assert h._add_box_before_whole is None


def test_whole_with_no_earlier_box_leaves_as_today(monkeypatch):
    install_fakes(monkeypatch)
    hh = Harness(np.zeros((12, 20, 20), np.int32),
                 images={"nuc": np.zeros((12, 20, 20), np.float32)},
                 spacing=(0.5, 0.25, 0.25))
    hh._add_restore_preset = lambda: None
    hh.adv.preset = "cells"
    hh.adv.source = RUN
    hh._start_add_whole()                 # Whole patch, no box before
    assert hh._add_box_before_whole is None
    hh._set_add_box_size_fraction(0.25)
    b = hh._add_box
    assert (b.cz, b.cy, b.cx) == (6, 10, 10), "the patch's centre"
    hh._reject_add()
