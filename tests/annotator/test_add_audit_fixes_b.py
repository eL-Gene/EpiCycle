"""The combined fix pass after the A5 data-flow audit (group B).

* B-M1 -- a marker whose ball sits inside a kept object at Preview time is
  ABSORBED: it made nothing, and until now nothing on screen said so.  The
  counts note's first part says it, and its row is dimmed (with a tooltip)
  when the marker list returns on Discard.
* B-M2 -- a re-base recolours every ball and row: S1 says why.
* B-L1 -- a downward re-base keeps every id at or above next_label.
* B-L2 -- the dtype guard on the reserved ids runs BEFORE the engine.
* B-L3 -- two clicks read the full volume's max once.
* B-L4 -- every early return of a tick pushes the list back.
"""

from __future__ import annotations

import numpy as np
import pytest

from packages.epicycle_annotator import add_preview as AP

from .add_harness import Harness, install_fakes
from .add_widget_fixture import qapp, reset_add, widget  # noqa: F401
from .test_add_click_markers import A, B, C, _ids, _place
from .test_add_click_markers import h  # noqa: F401  (the marker fixture)
from .test_add_preview import ISO, R, _box, _cells_params, _tissue


# --------------------------------------------------------------------------- #
# B-M1: the absorbed marker is on screen
# --------------------------------------------------------------------------- #
_PA, _PB, _PC = (6, 6, 5), (6, 6, 15), (6, 6, 25)


def _three_with_the_middle_covered():
    """The audit's scratch: three markers, the middle one's space filled
    with a kept object after it was placed."""
    mem, nuc, cells = _tissue()
    labels = np.zeros_like(cells)
    z, y, x = _PB
    labels[z - 4:z + 5, y - 4:y + 5, x - 4:x + 5] = 9     # a kept object
    chans = {"membrane": mem, "nuclei": nuc}
    return AP.run_add_preview(
        preset="cells", roi=_box(cells.shape), existing_roi=labels,
        next_label=500, spacing=ISO, params=_cells_params("paint"),
        wall_roi=mem, dark_rois=chans, click_points=[_PA, _PB, _PC],
        reserved_ids=[500, 501, 502],
        dark_threshold=R.patch_dark_threshold(chans, spacing=ISO, stride=1))


def test_the_absorbed_marker_is_counted_first_in_the_note():
    out = _three_with_the_middle_covered()
    assert out.ok, out.refusal
    assert sorted(out.ids) == [500, 502]
    assert out.absorbed_ids == [501]
    assert out.n_absorbed == 1
    note = AP.counts_note(out.counts(), source=out.source)
    assert note.startswith(AP.ADD_NOTE_ABSORBED_ONE), note


def test_the_sentence_and_its_singular():
    assert AP.counts_note({"n_absorbed": 2}) == (
        "2 markers now sit inside kept objects and made nothing")
    assert AP.counts_note({"n_absorbed": 1}) == (
        "1 marker now sits inside a kept object and made nothing")
    assert "cell" not in AP.ADD_NOTE_ABSORBED.lower()
    assert "cell" not in AP.ADD_MARKER_ABSORBED_TIP.lower()


@pytest.fixture
def hh(monkeypatch):
    install_fakes(monkeypatch)
    x = Harness(np.zeros((12, 36, 36), np.int32), images={})
    x.adv.preset = "cells"
    x.place_box((slice(0, 12), slice(0, 12), slice(0, 30)))
    x._add_click_markers = [
        {"zyx": _PA, "id": 500, "ticked": True},
        {"zyx": _PB, "id": 501, "ticked": True},
        {"zyx": _PC, "id": 502, "ticked": True}]
    x._add_markers_base = 500
    return x


def test_the_orchestrator_shows_it_and_marks_the_row(hh):
    out = _three_with_the_middle_covered()
    hh._apply_add_result(out)
    notes = hh.adv.last("notes")
    assert notes and notes[0][1].startswith(AP.ADD_NOTE_ABSORBED_ONE), notes
    assert hh.adv.last("click_absorbed_ids") == [501]
    # Discard: the list comes back with the mark
    hh._invalidate_add_preview()
    push = hh._add_marker_push()
    assert push["click_absorbed_ids"] == [501]


def test_the_panel_dims_an_absorbed_row_with_its_tooltip(widget):
    from qtpy.QtCore import Qt

    from napari_epicycle.annotator.advanced_features_widget import (
        ADD_HELP_LINES,
    )

    reset_add(widget)
    widget.set_add_preset("cells", silent=True)
    try:
        widget.update_add_ui_state(
            click_absorbed_ids=[501], click_count=3, click_markers=[
                (500, _PA, True), (501, _PB, True), (502, _PC, True)])
        lst = widget.add_click_list
        rows = {lst.item(i).data(Qt.UserRole): lst.item(i)
                for i in range(lst.count())}
        assert rows[501].toolTip() == AP.ADD_MARKER_ABSORBED_TIP
        assert rows[501].foreground().color().name() == "#777777"
        assert rows[500].toolTip() == ADD_HELP_LINES["add_click_list"]
        assert rows[500].foreground().color().name() == "#eeeeee"
        assert rows[501].checkState() == Qt.Checked, "a mark, not an untick"
        # the mark alone, without a rebuild
        widget.update_add_ui_state(click_absorbed_ids=[])
        assert rows[501].toolTip() == ADD_HELP_LINES["add_click_list"]
    finally:
        widget.update_add_ui_state(click_absorbed_ids=[], click_count=0,
                                   click_markers=[])
        reset_add(widget)


# --------------------------------------------------------------------------- #
# B-M2 / B-L1: the re-base on screen, and downward
# --------------------------------------------------------------------------- #
def test_a_rebase_says_why_in_s1(h):  # noqa: F811
    _place(h, A, B)                              # ids 8, 9
    h._get_label_id_floor_cb = lambda: 20        # next_label 8 -> 21
    _place(h, C)
    assert _ids(h) == [21, 22, 23]
    assert h._add_transient == AP.ADD_S1_REBASED.format(n=13)
    assert h.adv.last("status_text") == AP.ADD_S1_REBASED.format(n=13)
    assert h.adv.last("status_kind") == "warn"


def test_the_rebase_sentence_singular():
    assert AP.rebased_line(4, 1) == AP.ADD_S1_REBASED_ONE.format(n=4)
    assert AP.rebased_line(4, 3) == (
        "The object numbers moved up by 4 — something else took the old "
        "ones.")
    assert AP.rebased_line(-2, 3) == AP.ADD_S1_REBASED_DOWN.format(n=2)


def test_a_rebase_at_preview_says_why_too(h):  # noqa: F811
    from .test_add_click_markers import _cells_ready

    _place(h, A, C)
    h._get_label_id_floor_cb = lambda: 30
    _cells_ready(h)
    kw, why = h._add_gather_preview()
    assert why is None, why
    assert h._add_transient == AP.ADD_S1_REBASED.format(n=23)


def test_a_downward_rebase_keeps_every_id_at_or_above_next_label(h):  # noqa: F811
    h._get_label_id_floor_cb = lambda: 20
    _place(h, A, B)                              # ids 21, 22
    h._get_label_id_floor_cb = lambda: 0         # next_label 21 -> 8
    _place(h, C)
    nl = h._add_next_label()
    assert nl == 8
    assert min(_ids(h)) >= nl
    assert _ids(h) == [8, 9, 10]
    assert h._add_transient == AP.ADD_S1_REBASED_DOWN.format(n=13)


# --------------------------------------------------------------------------- #
# B-L2: the dtype guard before the engine
# --------------------------------------------------------------------------- #
def test_a_reserved_id_over_the_dtype_refuses_before_the_engine(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("the engine ran")

    monkeypatch.setattr(R, "reshape_empty_region", boom)
    out = AP.run_add_preview(
        preset="cells", roi=_box((4, 4, 4)),
        existing_roi=np.zeros((4, 4, 4), np.uint16), next_label=1,
        spacing=ISO, params=_cells_params("paint"),
        wall_roi=np.zeros((4, 4, 4), np.float32),
        click_points=[(1, 1, 1)], reserved_ids=[70000], id_max=65535)
    assert not out.ok
    assert out.refusal == AP.ADD_S1_IDS_FULL


# --------------------------------------------------------------------------- #
# B-L3: one full-volume max for two clicks
# --------------------------------------------------------------------------- #
class _Counting(np.ndarray):
    calls = 0

    def max(self, *a, **k):
        if self.ndim == 3 and self.shape == _Counting.full_shape:
            _Counting.calls += 1
        return super().max(*a, **k)


def test_two_clicks_do_one_full_volume_max(h):  # noqa: F811
    vol = h.sync_manager.shared_labels["L"]
    _Counting.full_shape = vol.shape
    _Counting.calls = 0
    h.sync_manager.shared_labels["L"] = vol.view(_Counting)
    _place(h, A, B)
    assert _ids(h) == [8, 9]
    assert _Counting.calls == 1, _Counting.calls


def test_accept_reset_and_realign_forget_the_cached_max(h):  # noqa: F811
    _place(h, A)
    assert h._add_volume_max is not None
    h._add_forget_volume_max()
    assert h._add_volume_max is None
    _place(h, B)
    assert h._add_volume_max is not None
    h._reset_add_state()
    assert h._add_volume_max is None


# --------------------------------------------------------------------------- #
# B-L4: early returns push the list back
# --------------------------------------------------------------------------- #
def test_a_tick_on_an_unknown_or_unchanged_marker_pushes_the_list(h):  # noqa: F811
    _place(h, A)
    n = len(h.adv.pushes)
    h._on_add_click_tick(8, True)                # no change
    assert len(h.adv.pushes) > n
    assert h.adv.last("click_markers") == [(8, A, True)]
    n = len(h.adv.pushes)
    h._on_add_click_tick(999, False)             # nobody's id
    assert len(h.adv.pushes) > n
    assert h.adv.last("click_markers") == [(8, A, True)]


def test_a_tick_with_no_session_pushes_the_list(h):  # noqa: F811
    _place(h, A)
    h._add_running = False
    n = len(h.adv.pushes)
    h._on_add_click_tick(8, False)
    assert len(h.adv.pushes) > n
    assert h._add_click_markers[0]["ticked"] is True
    assert h.adv.last("click_markers") == [(8, A, True)]
