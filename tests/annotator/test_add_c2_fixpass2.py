"""Add: the review follows the main Undo by the ids each
Accept wrote, a Clear stays in force when a later Clear lands, every
new-label path stays above the Clear floor, the run is read only with a box
up, the wall picture is dropped off Cells, and the outline packing is timed.

Driven through the real page code (the borrowed orchestrator) and the real
controller's undo stack.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from packages.epicycle_annotator import add_preview as AP

from .add_harness import FakeThread, Harness, install_fakes
from .test_add_c2_fixpass import _measure
from .test_add_outline_wiring import h  # noqa: F401


class _Layer:
    def __init__(self, name, data):
        self.name = name
        self.data = data

    def refresh(self):
        pass


@pytest.fixture
def fakes(monkeypatch):
    install_fakes(monkeypatch)


def _page_and_controller(shape=(8, 40, 40)):
    from tests.fusion_reconstruction.test_undo_realigns_annotator_shared import (
        _make_controller,
    )

    vol = np.zeros(shape, np.int32)
    hh = Harness(vol)
    hh.viewer.layers.append(_Layer("L", vol))
    ctrl = _make_controller(hh.viewer.layers[-1], hh)
    hh._on_add_committed_cb = ctrl._on_annotator_add_committed
    hh._on_discard_garbage_cb = ctrl._on_annotator_discard_garbage
    hh._get_kept_tier_count_cb = ctrl._add_kept_tier_count
    hh._get_live_add_clears_cb = ctrl._live_add_clear_seqs
    hh._on_add_undo_session_cb = ctrl._on_annotator_add_undo_session
    hh._add_clear_post = lambda fn: fn()
    hh._add_clear_confirm = lambda text: True
    hh._add_confirm = lambda text: True
    return hh, ctrl, vol


def _accept(hh, roi, pieces):
    """The real Accept of ``pieces`` = ``{id: slices inside roi}``."""
    hh.place_box(roi)
    shape = tuple(s.stop - s.start for s in roi)
    res = np.zeros(shape, np.int32)
    for i, sl in pieces.items():
        res[sl] = i
    hh._add_preview_result = res
    hh._add_preview_res = None
    hh._add_preview_roi = tuple(roi)
    hh.adv.checked = list(pieces)
    hh._accept_add()
    assert hh._add_review, hh._add_transient


def _another(hh):
    hh._add_another()
    assert hh._add_running and not hh._add_review


def _clear_box(hh, roi):
    hh.place_box(roi)
    hh._on_add_clear_requested("box")
    th = FakeThread.instances[-1]
    th.run()
    th.end()


def _rows(hh):
    return sorted(int(i) for i in hh._add_piece_ids)


OBJ = (slice(1, 3), slice(1, 3), slice(1, 3))
ROI0 = (slice(0, 8), slice(0, 10), slice(0, 10))
ROI1 = (slice(0, 8), slice(10, 20), slice(0, 10))
ROI2 = (slice(0, 8), slice(20, 40), slice(0, 40))
# around 111 only (global y 21-22, x 1-2); 112 starts at x 4
ROI3 = (slice(0, 8), slice(20, 24), slice(0, 3))


def _session_one(hh):
    """Session 1: Accept 50, then Done."""
    _accept(hh, ROI0, {50: OBJ})
    hh._reject_add()                       # Done
    assert not hh._add_review and hh._add_session == []


def _a2_pieces():
    pieces = {111: OBJ}
    for k in range(1, 10):                 # 112..120 along x
        pieces[111 + k] = (slice(1, 3), slice(1, 3),
                           slice(4 * k, 4 * k + 2))
    return pieces


# --------------------------------------------------------------------------- #
# the review follows the main Undo by the ids each Accept wrote
# --------------------------------------------------------------------------- #
def test_a_clear_of_an_accepts_first_id_then_three_undos_and_undo_all(fakes):
    hh, ctrl, vol = _page_and_controller()
    _session_one(hh)
    # session 2: A1 (101), A2 (111-120)
    _accept(hh, ROI1, {101: OBJ})
    _another(hh)
    _accept(hh, ROI2, _a2_pieces())
    assert _rows(hh) == [101] + list(range(111, 121))
    # "+ Add another": a box Clear takes 111, the first id of A2
    _another(hh)
    _clear_box(hh, ROI3)
    assert not (vol == 111).any()
    a2 = hh._add_session[1]
    assert a2["ids"] == list(range(112, 121))
    # A3 in the same box
    _accept(hh, ROI3, {121: (slice(5, 7), slice(0, 2), slice(0, 2))})
    assert _rows(hh) == [101] + list(range(112, 122))
    # Undo 1 takes back A3
    ctrl._undo_last_curation()
    assert hh._add_review
    assert _rows(hh) == [101] + list(range(112, 121))
    assert hh.adv.last("piece_ids") is not None
    assert sorted(hh.adv.last("piece_ids")) == _rows(hh)
    # Undo 2 takes back the Clear: 111 is back, and listed again
    ctrl._undo_last_curation()
    assert (vol == 111).any()
    assert hh._add_review
    assert _rows(hh) == [101] + list(range(111, 121))
    assert sorted(hh.adv.last("piece_ids")) == _rows(hh)
    # Undo 3 takes back A2: the hook hears 111 and finds its Accept
    ctrl._undo_last_curation()
    assert not (vol >= 111).any()
    assert hh._add_review
    assert _rows(hh) == [101]
    assert len(hh._add_session) == 1
    # Undo all takes back A1 only -- never the Accept of session 1
    hh._undo_add_all()
    assert not (vol == 101).any()
    assert (vol == 50).any(), "Undo all reached an earlier session"
    assert len(ctrl._curation_undo_stack) == 1
    assert ctrl._curation_undo_stack.peek().label_id == 50
    assert not hh._add_review


def test_an_accept_cleared_to_nothing_stays_in_the_session(fakes):
    hh, ctrl, vol = _page_and_controller()
    _session_one(hh)
    _accept(hh, ROI1, {101: OBJ})
    _another(hh)
    _accept(hh, ROI2, {111: OBJ})
    _another(hh)
    _clear_box(hh, ROI3)                   # takes 111: A2 has no rows left
    assert len(hh._add_session) == 2
    assert hh._add_session[1]["ids"] == []
    assert hh._add_session[1]["accept_ids"] == [111]
    _accept(hh, ROI3, {121: (slice(5, 7), slice(0, 2), slice(0, 2))})
    assert _rows(hh) == [101, 121]
    ctrl._undo_last_curation()             # A3
    assert _rows(hh) == [101]
    ctrl._undo_last_curation()             # the Clear: 111 listed again
    assert _rows(hh) == [101, 111]
    ctrl._undo_last_curation()             # A2: its entry leaves
    assert _rows(hh) == [101] and len(hh._add_session) == 1
    hh._undo_add_all()
    assert (vol == 50).any() and not (vol == 101).any()
    assert len(ctrl._curation_undo_stack) == 1


def test_undo_all_counts_an_accept_cleared_to_nothing(fakes):
    """With the Clear still on the stack Undo all stops at it, as before:
    the cleared Accept is counted, so nothing below the Clear moves."""
    hh, ctrl, vol = _page_and_controller()
    _accept(hh, ROI1, {101: OBJ})
    _another(hh)
    _accept(hh, ROI2, {111: OBJ})
    _another(hh)
    _clear_box(hh, ROI3)
    _accept(hh, ROI3, {121: (slice(5, 7), slice(0, 2), slice(0, 2))})
    hh._undo_add_all()
    assert not (vol == 121).any()
    assert (vol == 101).any(), "stopped at the Clear"
    assert ctrl._curation_undo_stack.peek().action_type == "discard_garbage"


def test_a_clear_undone_outside_the_review_lists_nothing(fakes):
    hh, _ctrl, _vol = _page_and_controller()
    hh._add_session = [{"ids": [5], "sizes": {5: 1}, "accept_ids": [5, 6],
                        "accept_sizes": {5: 1, 6: 2}}]
    hh._add_review = False
    hh._add_after_clear_undone([6])
    assert hh._add_session[0]["ids"] == [5]
    hh._add_review = True
    hh._add_after_clear_undone([6])
    assert hh._add_session[0]["ids"] == [5, 6]
    assert hh._add_session[0]["sizes"] == {5: 1, 6: 2}
    # ids no Accept of the review wrote: nothing moves
    hh._add_after_clear_undone([999])
    assert hh._add_session[0]["ids"] == [5, 6]


# --------------------------------------------------------------------------- #
# a later Clear keeps an evicted Clear in force
# --------------------------------------------------------------------------- #
def test_an_evicted_clear_stays_in_the_outline_after_another_clear(fakes):
    hh, ctrl, vol = _page_and_controller()
    vol[1:3, 21:23, 1:3] = 7               # cleared by Clear #1
    vol[5:7, 30:32, 30:32] = 8             # cleared by Clear #2
    _clear_box(hh, ROI3)
    s1 = ctrl._curation_undo_stack.peek().extra["add_clear_seq"]
    hh._reject_add()
    # 20 Accepts push Clear #1 off the 20-deep stack
    for k in range(20):
        i = 200 + k
        roi = (slice(7, 8), slice(k, k + 1), slice(39, 40))
        snap = vol[roi].copy()
        vol[roi] = i
        ctrl._on_annotator_add_committed(
            [i], roi_slices=roi, old_roi_data=snap,
            piece_stats={i: (1, (7.0, float(k), 39.0))})
    assert s1 not in ctrl._live_add_clear_seqs()
    assert [s for s, _ids in hh._add_cleared_in_force()] == [s1]
    # Clear #2
    _clear_box(hh, (slice(4, 8), slice(28, 34), slice(28, 34)))
    s2 = ctrl._curation_undo_stack.peek().extra["add_clear_seq"]
    assert s1 in hh._add_clear_masks and s1 in hh._add_clear_records
    assert [s for s, _ids in hh._add_cleared_in_force()] == [s1, s2]
    # and the cleared outline is built from both masks
    hh.adv.preset = "cells"
    hh.adv.params["outline"] = AP.OUTLINE_CLEARED
    job = hh._add_outline_job()
    union = AP.cleared_union(job["shape"], job["cleared"])
    assert union[1:3, 21:23, 1:3].all(), "Clear #1 is still in the outline"
    assert union[5:7, 30:32, 30:32].all()


# --------------------------------------------------------------------------- #
# every new-label path stays above the Clear floor
# --------------------------------------------------------------------------- #
def test_the_next_label_after_an_accept_skips_cleared_ids():
    from napari_epicycle.annotator.curation_3d_widget import (
        Curation3DAnnotator,
    )

    vol = np.zeros((4, 4, 4), np.int32)
    vol[0, 0, 0] = 5
    ann = types.SimpleNamespace(
        _navigator_widget=None, _active_labels_name="L",
        sync_manager=types.SimpleNamespace(shared_labels={"L": vol}),
        _add_clear_floor=30)
    nxt = Curation3DAnnotator._next_label_id_after_accept
    assert nxt(ann) == 31, "the volume max is below the floor"
    ann._navigator_widget = types.SimpleNamespace(
        _items=[{"label_id": 3}, {"label_id": 10}])
    assert nxt(ann) == 31, "the navigator max is below the floor"
    ann._navigator_widget._items.append({"label_id": 40})
    assert nxt(ann) == 41
    ann._add_clear_floor = 0
    ann._navigator_widget = None
    assert nxt(ann) == 6, "no Clear: as before"


# --------------------------------------------------------------------------- #
# the Cells source reads a run only with a box up
# --------------------------------------------------------------------------- #
def test_a_run_picked_with_no_box_is_not_read(h):
    loads = []
    h._add_load_run_guide = loads.append
    h._add_running = False
    h._on_add_source_changed(("labels", "run-2"))
    assert loads == []
    h.place_box((slice(0, 12), slice(0, 20), slice(0, 20)))
    h._on_add_source_changed(("labels", "run-2"))
    assert loads == ["run-2"]


# --------------------------------------------------------------------------- #
# off Cells the wall picture is dropped, the packed outline kept
# --------------------------------------------------------------------------- #
def test_the_wall_picture_is_dropped_when_the_preset_leaves_cells(h):
    h.place_box((slice(0, 12), slice(0, 20), slice(0, 20)))
    _measure(h)
    h._add_elev_cache = {"key": h._add_elev_key(), "elev": np.zeros(3)}
    h._reject_add()                        # the outline is packed
    assert h._add_elev_cache is not None
    packed = h._add_outline_cache
    assert packed and packed.get("packed") is not None
    h.adv.preset = "nuclei"
    h._add_refresh_ui()
    assert h._add_elev_cache is None
    assert h._add_outline_cache is packed, "the packed outline stays"


def test_the_outline_packing_at_cancel_prints_a_timing_line(h, capsys):
    h.place_box((slice(0, 12), slice(0, 20), slice(0, 20)))
    _measure(h)
    capsys.readouterr()
    h._reject_add()
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines()
             if ln.startswith("[ADD-PERF] outline pack")]
    assert len(lines) == 1, out
    assert "voxels=" in lines[0] and lines[0].endswith("ms")
