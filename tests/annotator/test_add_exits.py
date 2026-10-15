"""Every exit of Add removes exactly its artefacts (B2.5).

Columns: cand, walls, balls, box overlay, side callbacks, preview worker,
info worker, Numbers..., run guide + dark cache.  The real orchestrator on
the Qt-free harness; the box overlay and the Numbers dialog are recorded by
the harness stubs.
"""

from __future__ import annotations

import numpy as np
import pytest

from .add_harness import FakeThread, Harness, install_fakes
from .test_add_click_markers import h  # noqa: F401 - the fixture


class _Overlay:
    def destroy(self):
        pass


def _prime(h):
    """Candidates up, a ball, a live info worker, Numbers open, a guide."""
    h._add_marker_click((12, 20, 20))
    h._add_box_overlay = _Overlay()
    h._add_candidate_layer = object()
    h._add_preview_result = np.zeros((2, 2, 2), np.int32)
    h._add_run_guide = np.zeros((1,))
    h._add_dark_cache = {("P", 1, ()): {"t": 1}}
    h.numbers_open = True
    h._request_add_box_info()
    h._install_add_side_callbacks()


@pytest.fixture
def hx(h, monkeypatch):  # noqa: F811
    h.adv.close_add_numbers = lambda: setattr(h, "numbers_open", False)
    return h


def test_discard_keeps_the_box_balls_and_the_lock(hx):
    _prime(hx)
    hx._add_secondary()                       # Discard (candidates up)
    assert hx._add_preview_result is None
    assert hx._add_candidate_layer is None
    assert hx._add_marker_layer is not None and hx._add_click_markers
    assert hx._add_box_overlay is not None
    assert hx._add_running
    assert hx._add_side_callbacks_installed
    assert hx.locks == []                     # no unlock


def test_cancel_removes_everything(hx):
    _prime(hx)
    w = hx._add_info_worker
    hx._add_preview_result = None             # Cancel: no candidates
    hx._add_secondary()
    assert hx._add_candidate_layer is None
    assert hx._add_marker_layer is None and hx._add_click_markers == []
    assert hx._add_box_overlay is None
    assert not hx._add_side_callbacks_installed
    assert hx._add_info_thread is None and w.cancel_event.is_set()
    assert hx.numbers_open is False
    assert hx._add_run_guide is None and hx._add_dark_cache == {}
    assert not hx._add_running


def test_accept_to_review_keeps_the_box_drawn_and_drops_the_rest(hx):
    _prime(hx)
    res = np.zeros((16, 26, 26), np.int32)
    res[8:12, 15:19, 15:19] = 40
    hx._add_preview_result = res
    hx._add_preview_roi = tuple(hx._add_roi)
    hx.adv.checked = [40]
    hx._on_add_committed_cb = None
    hx._accept_add()
    assert hx._add_review
    assert hx._add_box_overlay is not None, "the box stays drawn in REVIEW"
    assert not hx._add_side_callbacks_installed, "no live drag in REVIEW"
    assert hx._add_marker_layer is None and hx._add_click_markers == []
    assert hx._add_info_thread is None
    assert hx.numbers_open is False
    assert hx._add_run_guide is not None, "kept for + Add another"
    assert hx._add_box is not None


def test_add_another_reinstalls_callbacks_balls_help_and_info(hx):
    _prime(hx)
    res = np.zeros((16, 26, 26), np.int32)
    res[8:12, 15:19, 15:19] = 40
    hx._add_preview_result = res
    hx._add_preview_roi = tuple(hx._add_roi)
    hx.adv.checked = [40]
    hx._on_add_committed_cb = None
    box = hx._add_box
    hx._accept_add()
    n = len(FakeThread.instances)
    hx._add_restore_preset = lambda: None
    hx._add_another()
    assert hx._add_box == box, "re-placed, not re-centred"
    assert hx._add_side_callbacks_installed
    assert hx._add_click_markers == [] and hx._add_marker_layer is None
    assert len(FakeThread.instances) == n + 1, "box info requested"
    assert hx.adv.last("box_active") is True


def test_review_done_removes_the_box(hx):
    _prime(hx)
    hx._add_running = False
    hx._add_review = True
    hx._add_secondary()                       # Done
    assert hx._add_box_overlay is None
    assert hx._add_run_guide is None


def test_patch_switch_tears_everything_down(hx):
    _prime(hx)
    hx._teardown_add_for_switch()
    assert hx._add_candidate_layer is None and hx._add_marker_layer is None
    assert hx._add_box_overlay is None and hx._add_info_thread is None
    assert not hx._add_side_callbacks_installed and not hx._add_running


def test_patch_switch_in_review_tears_down_too(hx):
    _prime(hx)
    hx._add_running = False
    hx._add_review = True
    hx._teardown_add_for_switch()
    assert hx._add_box_overlay is None and not hx._add_review


def test_a_tab_switch_keeps_every_artefact(hx):
    _prime(hx)
    hx._refresh_all_image_combos = lambda: None
    from napari_epicycle.annotator.curation_3d_widget import (
        Curation3DAnnotator,
    )

    Curation3DAnnotator._on_tool_tab_changed(hx)
    assert hx._add_marker_layer is not None
    assert hx._add_box_overlay is not None
    assert hx._add_preview_result is not None
    assert hx._add_side_callbacks_installed
    assert hx._add_place_markers_on is False


def test_reject_twice_is_a_no_op_the_second_time(hx):
    _prime(hx)
    hx._reject_add()
    locks = list(hx.locks)
    hx._reject_add()
    assert hx.locks == locks


def test_the_undo_all_confirm_can_keep_them(hx):
    hx._add_running = False
    hx._add_review = True
    hx._add_session = [{"ids": [40, 41], "sizes": {}}]
    hx._add_piece_ids = [40, 41]
    calls = []
    hx._on_add_undo_session_cb = lambda n: calls.append(n) or n
    hx._add_confirm = lambda text: (calls.append(text), False)[1]
    hx._undo_add_all()
    assert calls == ["Take back all 2 objects added this session?"]
    assert hx._add_review


def test_undo_all_partial_says_it_in_objects(hx):
    from packages.epicycle_annotator import add_preview as AP

    hx._add_running = False
    hx._add_review = True
    hx._add_session = [{"ids": [40, 41]}, {"ids": [50]}]
    hx._add_piece_ids = [40, 41, 50]
    hx._on_add_undo_session_cb = lambda n: 1
    hx._add_confirm = lambda text: True
    hx._undo_add_all()
    assert AP.ADD_UNDO_PARTIAL.format(k=1, n=3) in [
        p.get("status_text") for p in hx.adv.pushes]


# --------------------------------------------------------------------------- #
# the walls picture
# --------------------------------------------------------------------------- #
def test_walls_grey_on_nuclei_and_placed_on_cells(hx):
    from packages.epicycle_annotator import add_preview as AP

    hx.adv.preset = "nuclei"
    assert hx._add_walls_toggle_state() == (False, AP.WALLS_TOGGLE_NO_WALLS)
    hx.adv.preset = "cells"
    res = AP.AddPreviewResult(roi=tuple(hx._add_roi),
                              elevation=np.ones((16, 26, 26), np.float32))
    hx._add_preview_res = res
    assert hx._add_walls_toggle_state() == (True, "")
    hx._on_add_show_walls(True)
    arr, kw = hx.temp_layers["add_walls"]
    assert kw["translate"] == (2.0, 2.0, 2.0)       # scale 1 x roi start
    assert kw["kind"] == "image"
    assert hx._add_walls_wanted
    dropped = []
    hx._remove_temp_layer = lambda key: dropped.append(key)
    hx._invalidate_add_preview()
    assert dropped == ["add_walls"] and hx._add_walls_wanted, (
        "the picture goes, the intent stays")


def test_a_walls_picture_over_the_budget_says_so_in_the_legend(hx, monkeypatch):
    """B2 fix pass L-L7: the refusal used to be a print only."""
    from napari_epicycle.annotator import curation_3d_widget as CW
    from napari_epicycle.annotator.advanced_features_widget import (
        WALLS_TOGGLE_TOO_LARGE,
    )
    from packages.epicycle_annotator import add_preview as AP

    monkeypatch.setattr(CW, "WALLS_DISPLAY_BUDGET_BYTES", 16)
    hx.adv.preset = "cells"
    hx._add_preview_res = AP.AddPreviewResult(
        roi=tuple(hx._add_roi), elevation=np.ones((16, 26, 26), np.float32))
    assert hx._show_add_walls() is False
    assert "add_walls" not in getattr(hx, "temp_layers", {})
    assert hx.adv.last("walls_toggle_enabled") is False
    reason = hx.adv.last("walls_toggle_reason")
    assert reason.startswith(WALLS_TOGGLE_TOO_LARGE)
    assert AP.ADD_WALLS_TOO_LARGE in reason


def test_a_patch_switch_in_review_goes_through_the_s1_table(hx):
    """B2 fix pass L-L2: the REVIEW branch of the switch pushed a hard-coded
    sentence from the old page and kept the touched set."""
    import inspect

    from napari_epicycle.annotator import curation_3d_widget as CW

    cleared = []
    hx.adv.clear_add_touched = lambda: cleared.append(True)
    _prime(hx)
    hx._add_running = False
    hx._add_review = True
    hx._teardown_add_for_switch()
    assert cleared, "a full exit forgets the touched keys"
    from packages.epicycle_annotator import add_preview as AP

    assert hx.adv.last("status_text") == AP.ADD_S1_IDLE
    src = inspect.getsource(CW.Curation3DAnnotator._load_selected_layers)
    assert "Pick a channel, then Place Box." not in src


# --------------------------------------------------------------------------- #
# the outline tick: the "outline" column of 4.7
# --------------------------------------------------------------------------- #
def _outlined(h):
    _prime(h)
    h._add_contour_on = True


def test_discard_keeps_the_outline_preference(hx):
    _outlined(hx)
    hx._add_secondary()                       # Discard
    assert hx._add_contour_on is True
    assert hx.adv.last("contour_on") is not False


def test_a_tab_switch_keeps_the_outline_preference(hx):
    _outlined(hx)
    hx._refresh_all_image_combos = lambda: None
    from napari_epicycle.annotator.curation_3d_widget import (
        Curation3DAnnotator,
    )

    Curation3DAnnotator._on_tool_tab_changed(hx)
    assert hx._add_contour_on is True


def test_an_invalidation_keeps_the_outline_preference(hx):
    _outlined(hx)
    hx._invalidate_add_preview()
    assert hx._add_contour_on is True


@pytest.mark.parametrize("exit_", ["cancel", "accept", "switch", "review",
                                   "another"])
def test_every_full_exit_puts_the_outline_down(hx, exit_):
    _outlined(hx)
    if exit_ == "cancel":
        hx._add_preview_result = None
        hx._add_secondary()
    elif exit_ in ("accept", "another"):
        res = np.zeros((16, 26, 26), np.int32)
        res[8:12, 15:19, 15:19] = 40
        hx._add_preview_result = res
        hx._add_preview_roi = tuple(hx._add_roi)
        hx.adv.checked = [40]
        hx._on_add_committed_cb = None
        hx._accept_add()
        if exit_ == "another":
            hx._add_restore_preset = lambda: None
            hx._add_contour_on = True       # even if something set it again
            hx._add_another()
    elif exit_ == "switch":
        hx._teardown_add_for_switch()
    else:                                     # REVIEW -> Done
        hx._add_running = False
        hx._add_review = True
        hx._add_secondary()
    assert hx._add_contour_on is False, exit_
    assert hx.adv.last("contour_on") is False, exit_


# --------------------------------------------------------------------------- #
# the marker list and its ticks: the "marker list / ticks"
# column of 4.7 -- kept on a tab switch, Discard and an invalidation;
# hidden (not cleared) over candidates; cleared at every full exit
# --------------------------------------------------------------------------- #
def _ticked_markers(h):
    _prime(h)                                 # marker 8 at (12, 20, 20)
    assert h._add_marker_click((12, 10, 20))  # marker 9
    h._add_preview_result = None
    h._on_add_click_tick(9, False)
    h._add_preview_result = np.zeros((2, 2, 2), np.int32)   # candidates up
    return [(8, (12, 20, 20), True), (9, (12, 10, 20), False)]


def _view(h):
    return [(m["id"], tuple(m["zyx"]), m["ticked"])
            for m in h._add_click_markers]


def test_discard_keeps_the_markers_and_their_ticks(hx):
    want = _ticked_markers(hx)
    hx._add_secondary()                       # Discard
    assert _view(hx) == want
    assert hx.adv.last("click_markers") == want


def test_a_tab_switch_keeps_the_markers_and_their_ticks(hx):
    want = _ticked_markers(hx)
    hx._refresh_all_image_combos = lambda: None
    from napari_epicycle.annotator.curation_3d_widget import (
        Curation3DAnnotator,
    )

    Curation3DAnnotator._on_tool_tab_changed(hx)
    assert _view(hx) == want


def test_an_invalidation_keeps_the_markers_and_their_ticks(hx):
    want = _ticked_markers(hx)
    hx._invalidate_add_preview()
    assert _view(hx) == want


@pytest.mark.parametrize("exit_", ["cancel", "accept", "switch", "another",
                                   "close"])
def test_every_full_exit_clears_the_markers(hx, exit_):
    _ticked_markers(hx)
    e0 = hx._add_markers_epoch
    if exit_ in ("cancel", "close"):
        hx._add_preview_result = None
        if exit_ == "cancel":
            hx._add_secondary()
        else:
            hx._reject_add()                  # the close path, twice
            hx._reject_add()
    elif exit_ in ("accept", "another"):
        res = np.zeros((16, 26, 26), np.int32)
        res[8:12, 15:19, 15:19] = 40
        hx._add_preview_result = res
        hx._add_preview_roi = tuple(hx._add_roi)
        hx.adv.checked = [40]
        hx._on_add_committed_cb = None
        hx._accept_add()
        if exit_ == "another":
            hx._add_restore_preset = lambda: None
            hx._add_another()
    else:
        hx._teardown_add_for_switch()
    assert hx._add_click_markers == [], exit_
    assert hx._add_markers_base is None, exit_
    assert hx._add_markers_epoch > e0, exit_
    assert hx.adv.last("click_markers") == [], exit_
    assert hx._add_marker_layer is None, exit_
