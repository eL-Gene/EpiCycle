"""Cells on the Add page after the live run on a real patch: what it showed.

Pinned here:

* Whole patch pressed with no box (S2) keeps the run picked under Markers,
  as Whole patch at the top of Numbers… does: the page it was offered for
  is the page it starts from;
* that run is READ when the whole patch (or a box) starts with it picked
  and not yet in memory -- a run picked with no box up is not read then,
  and a combo that keeps its run fires nothing;
* the notes on a whole patch never name a box edge, and the log says
  "patch" when a whole patch is stopped;
* the pieces the growth leaves cut off from their marker are said as that
  (never "cut off by kept objects" when there may be none);
* the tissue outline is a hard edge after the smoothing too: what the
  smoothing rounds past it is cleared; with no outline nothing changes.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from napari_epicycle.annotator import reshape_worker as RW
from packages.epicycle_annotator import add_preview as AP
from packages.epicycle_annotator import add_segment as AS

from .add_harness import Harness, install_fakes
from .add_widget_fixture import qapp, reset_add, widget  # noqa: F401
from .test_add_outline_wiring import (  # noqa: F401
    _land_info,
    _show,
    _world,
    h,
)
from .test_reshape_region import _threshold


# --------------------------------------------------------------------------- #
# Whole patch from S2 keeps the picked run
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pick", [0, 2])       # a run, Click markers
def test_whole_patch_with_no_box_keeps_the_run_picked_under_markers(
        widget, monkeypatch, pick):
    """A run and Click markers both fill a whole patch: either one picked
    stays picked, and S1 never asks for a run."""
    install_fakes(monkeypatch)
    ch, cells = _world()
    hh = Harness(cells.copy(), images=ch)
    hh.adv = widget
    hh.ctx["spacing_source"] = "qc_settings"
    hh._add_run_guide = cells
    hh._add_dark_cache = {hh._add_dark_key(): _threshold(ch)}
    hh._add_session_memory = {"last_preset": "cells"}
    reset_add(widget)
    widget.refresh_add_image_combo(["membrane", "nuclei"])
    widget.set_add_preset("cells")
    run = ("labels", "cells")
    entries = [(AP.ADD_ITEM_RUN.format(name="cells"), True, run),
               (AP.ADD_ITEM_NUCLEI, True, (AP.SOURCE_NUCLEI, None)),
               (AP.ADD_ITEM_CLICKS, True, ("paint", None))]
    hh._add_marker_source_entries = lambda: list(entries)
    widget.set_add_marker_sources(entries)
    widget.add_marker_source_combo.setCurrentIndex(pick)
    want = entries[pick][2]
    # the last session ended: nothing is "touched" any more, the pick is
    # still the one on screen
    widget.clear_add_touched()
    assert widget.get_add_marker_source() == want
    try:
        hh._start_add_whole()
        assert hh._add_running and hh._add_whole_patch
        assert widget.get_add_marker_source() == want, "the pick is kept"
        assert widget.add_status_label.text() not in (
            AP.ADD_S1_WHOLE_PICK_RUN, AP.ADD_S1_WHOLE_NO_RUN)
    finally:
        hh._reject_add()
        widget.set_add_marker_sources([])
        reset_add(widget)


def test_an_ordinary_start_still_takes_the_presets_values(monkeypatch):
    """The usual Add a box still starts from the preset used last (the
    Whole patch rule above does not reach it)."""
    install_fakes(monkeypatch)
    ch, cells = _world()
    hh = Harness(cells.copy(), images=ch)
    seen = []
    hh._add_restore_preset = lambda: seen.append("restored")
    hh._start_add()
    assert seen == ["restored"]
    hh._reject_add()
    seen.clear()
    hh._start_add(whole=True)
    assert seen == [], "Whole patch keeps the page as it stands"
    hh._reject_add()


# --------------------------------------------------------------------------- #
# the picked run is read when the whole patch (or a box) starts
# --------------------------------------------------------------------------- #
class _GuideWorkerHere(RW.GuideLoadWorker):
    def moveToThread(self, _th):      # noqa: N802 - Qt name
        return None


def _cells_page_with_a_run(widget, monkeypatch, tmp_path):
    """The real page on Cells with Markers ▸ Objects from run ▸ cells
    picked while NO box is up (the page's own source signal reaches the
    orchestrator, so the "not read with no box" rule runs), and the run on
    disk, read by the real worker on a fake thread."""
    tifffile = pytest.importorskip("tifffile")
    install_fakes(monkeypatch)
    monkeypatch.setattr(RW, "GuideLoadWorker", _GuideWorkerHere)
    ch, cells = _world()
    path = tmp_path / "labels_curated.tif"
    tifffile.imwrite(str(path), cells.astype(np.uint16))
    hh = Harness(np.zeros(cells.shape, np.int32), images=ch)
    hh.adv = widget
    hh.ctx["spacing_source"] = "qc_settings"
    hh._add_run_sources = {"cells": types.SimpleNamespace(
        path=str(path), run_id="cells", display_name="cells",
        target="cells", which_file="labels_curated.tif")}
    hh._add_dark_cache = {hh._add_dark_key(): _threshold(ch)}
    hh._add_session_memory = {"last_preset": "cells"}
    reset_add(widget)
    widget.refresh_add_image_combo(["membrane", "nuclei"])
    widget.set_add_preset("cells")
    run = ("labels", "cells")
    entries = [(AP.ADD_ITEM_RUN.format(name="cells"), True, run),
               (AP.ADD_ITEM_NUCLEI, True, (AP.SOURCE_NUCLEI, None)),
               (AP.ADD_ITEM_CLICKS, True, ("paint", None))]
    hh._add_marker_source_entries = lambda: list(entries)
    widget.set_add_marker_sources(entries)
    widget.add_wall_channel_combo.setCurrentIndex(
        widget.add_wall_channel_combo.findText("membrane"))
    monkeypatch.setattr(widget, "_on_add_source_cb",
                        hh._on_add_source_changed, raising=False)
    # picked by hand, with no box up: the source signal fires, and the
    # run is not read then (it has no page to land on)
    widget.add_marker_source_combo.setCurrentIndex(1)
    widget.add_marker_source_combo.setCurrentIndex(0)
    assert widget.get_add_marker_source() == run
    assert hh._add_run_guide is None and hh._add_run_state is None
    assert getattr(hh, "_add_guide_worker", None) is None
    return hh, cells


def _land_the_run(hh):
    w = hh._add_guide_worker
    th = hh._add_guide_thread
    assert w is not None and th is not None, "the run read never started"
    hh.sender = w
    th.run()
    hh.sender = None


def _assert_the_run_is_read_and_counted(hh, widget, cells):
    assert hh._add_run_state == "loading"
    assert not widget.add_preview_btn.isEnabled(), "nothing to preview yet"
    _land_the_run(hh)
    assert hh._add_run_guide is not None
    assert np.array_equal(hh._add_run_guide, cells)
    assert hh._add_run_state is None
    _land_info(hh)
    hh._add_refresh_ui()
    count = hh._add_info.run_count
    assert count is not None, "the run's objects were never counted"
    n_obj = int(count[0])
    assert n_obj > 0
    assert widget.add_preview_btn.isEnabled(), widget.add_status_label.text()


def test_whole_patch_with_no_box_reads_the_run_picked_under_markers(
        widget, monkeypatch, tmp_path):
    hh, cells = _cells_page_with_a_run(widget, monkeypatch, tmp_path)
    # the last session ended: nothing touched, the run still on screen
    widget.clear_add_touched()
    try:
        hh._start_add_whole()
        assert hh._add_running and hh._add_whole_patch
        assert widget.get_add_marker_source() == ("labels", "cells")
        _assert_the_run_is_read_and_counted(hh, widget, cells)
        status = widget.add_status_label.text()
        assert status.startswith("Whole patch "), status
    finally:
        hh._reject_add()
        widget.set_add_marker_sources([])
        reset_add(widget)


def test_a_box_reads_the_run_picked_before_it(widget, monkeypatch,
                                              tmp_path):
    hh, cells = _cells_page_with_a_run(widget, monkeypatch, tmp_path)
    try:
        hh._start_add()               # Place box: the preset is restored
        assert hh._add_running and not hh._add_whole_patch
        # the source was picked by hand: the preset's defaults keep it
        assert widget.get_add_marker_source() == ("labels", "cells")
        _assert_the_run_is_read_and_counted(hh, widget, cells)
    finally:
        hh._reject_add()
        widget.set_add_marker_sources([])
        reset_add(widget)


def test_place_box_after_a_full_forget_still_restores_the_preset(
        widget, monkeypatch, tmp_path):
    """A box after every touched key was forgotten starts from the
    preset's defaults (Markers back to nuclei masks), so no run is read.
    The end of a box no longer forgets the Markers source (see
    ``test_add_run_pick_sticks``); a full forget still does."""
    hh, _cells = _cells_page_with_a_run(widget, monkeypatch, tmp_path)
    widget.clear_add_touched()
    try:
        hh._start_add()
        assert widget.get_add_marker_source() == (AP.SOURCE_NUCLEI, None)
        assert getattr(hh, "_add_guide_worker", None) is None
        assert hh._add_run_state is None
    finally:
        hh._reject_add()
        widget.set_add_marker_sources([])
        reset_add(widget)


# --------------------------------------------------------------------------- #
# the whole patch's words
# --------------------------------------------------------------------------- #
def test_the_thin_part_names_no_box_edge_on_a_whole_patch():
    for n, want in ((1, "1 run object was too small to shrink into a marker "
                        "and was skipped"),
                    (8, "8 run objects were too small to shrink into a "
                        "marker and were skipped")):
        text = AP.counts_note({"n_dropped_thin": n}, source=AP.SOURCE_RUN,
                              whole_patch=True)
        assert text == want
        assert "box" not in text
    # a box keeps its words
    assert "box edge" in AP.counts_note({"n_dropped_thin": 2},
                                        source=AP.SOURCE_RUN)


def test_a_stopped_whole_patch_says_patch_in_the_log(h, capsys):
    h.place_box((slice(0, 12), slice(0, 20), slice(0, 20)))
    h._add_whole_patch = True
    capsys.readouterr()
    h._on_add_stopped()
    out = capsys.readouterr().out
    assert "nothing from this patch was added" in out
    assert "this box" not in out
    assert h._add_transient == AP.ADD_S1_STOPPED_PATCH
    h._add_whole_patch = False
    h._on_add_stopped()
    assert "nothing from this box was added" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# the pieces cut off from their marker
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n, want", [
    (1, "1 piece not joined to its marker was left out"),
    (1009, "1009 pieces not joined to their marker were left out"),
])
def test_the_pieces_cut_off_from_their_marker_are_said_as_that(n, want):
    text = AP.counts_note({"n_fragments_dropped": n}, source=AP.SOURCE_RUN,
                          whole_patch=True)
    assert text == want
    # the count is the growth's, whatever cut the piece off: never said
    # as "cut off by kept objects" (after a Clear of every object there
    # are none)
    assert "kept objects" not in text
    assert "kept objects" not in AP.counts_note({"n_fragments_dropped": n})


# --------------------------------------------------------------------------- #
# the outline is a hard edge after the smoothing
# --------------------------------------------------------------------------- #
def _cube_world():
    result = np.zeros((10, 20, 20), np.int32)
    result[3:7, 4:10, 4:10] = 5
    outline = np.zeros(result.shape, bool)
    outline[:, :, :10] = True          # the cell touches the outline's edge
    return result, outline


def _grow_one(result, sigma=1.5, cancel_event=None):
    """A stand-in smoothing that rounds every piece one voxel outward (as
    an SDF smoothing can, at a corner)."""
    from scipy import ndimage as ndi

    out = result.copy()
    for i in [int(v) for v in np.unique(result) if v]:
        grown = ndi.binary_dilation(result == i)
        out[grown & (out == 0)] = i
    return out


def test_a_cell_the_smoothing_rounds_past_the_outline_is_clipped(
        monkeypatch, capsys):
    monkeypatch.setattr(AS, "smooth_pieces", _grow_one)
    result, outline = _cube_world()
    assert (_grow_one(result)[~outline] > 0).any(), "it would cross"
    out = AP.AddPreviewResult(status="ok", preset="cells")
    AP._finish(out, result, {"auto_smooth": True, "clear_border": False},
               None, lambda *_a: None, outline=outline)
    assert out.status == "ok"
    assert not (out.result[~outline] > 0).any(), "nothing past the outline"
    assert (out.result[outline] == 5).sum() > (result == 5).sum(), (
        "the smoothing inside the outline is kept")
    assert out.ids == [5]
    assert "past the tissue outline cleared" in capsys.readouterr().out


def test_with_no_outline_the_smoothing_is_not_clipped(monkeypatch):
    monkeypatch.setattr(AS, "smooth_pieces", _grow_one)
    result, _outline = _cube_world()
    out = AP.AddPreviewResult(status="ok", preset="cells")
    AP._finish(out, result, {"auto_smooth": True, "clear_border": False},
               None, lambda *_a: None)
    assert np.array_equal(out.result, _grow_one(result))


def test_a_cell_the_smoothing_leaves_wholly_outside_counts_as_vanished(
        monkeypatch):
    def _shift_out(result, sigma=1.5, cancel_event=None):
        return np.roll(result, 10, axis=2)

    monkeypatch.setattr(AS, "smooth_pieces", _shift_out)
    result, outline = _cube_world()
    out = AP.AddPreviewResult(status="ok", preset="cells")
    AP._finish(out, result, {"auto_smooth": True, "clear_border": False},
               None, lambda *_a: None, outline=outline)
    assert out.n_emptied_by_smooth == 1
    assert out.ids == []


def test_the_preview_hands_the_boxs_crop_of_the_outline_to_the_finish(
        h, monkeypatch):
    seen = {}
    real = AP._finish

    def _spy(out, result, params, cancel_event, step, outline=None):
        seen["outline"] = outline
        seen["shape"] = np.shape(result)
        return real(out, result, params, cancel_event, step, outline=outline)

    monkeypatch.setattr(AP, "_finish", _spy)
    _show(h)
    h._preview_add()
    th = h._add_thread
    th.run()
    th.end()
    res = h._add_preview_res
    assert res is not None and res.ok, getattr(res, "refusal", None)
    roi = h._add_roi
    whole = h._add_cached_outline()
    assert seen["outline"] is not None
    assert seen["outline"].shape == seen["shape"]
    assert np.array_equal(seen["outline"], whole[roi])
    # and nothing of the answer lies past it
    assert not (res.result[~whole[roi]] > 0).any()
