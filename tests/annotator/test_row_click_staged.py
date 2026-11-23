"""A row click while an edit is open changes nothing: no camera move, no
selection, no isolation, and the highlight returns to the edit's own cell."""
from __future__ import annotations

import types
from types import SimpleNamespace

import pytest
from qtpy.QtCore import Qt

from napari_epicycle.annotator._staged_op_text import STAGED_WORK_OPEN
from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator as K
from napari_epicycle.annotator.curation_label_navigator import (
    CurationLabelNavigator as N,
)
from napari_epicycle.annotator.refinement_state import RefinementOpState


class _Item:
    def data(self, role):
        return {"label_id": 5, "centroid": (1, 2, 3)}


class _Table:
    def item(self, r, c):
        return _Item()

    def setFocus(self):
        pass


class _Cb:
    def isChecked(self):
        return True


def _nav(blocked):
    n = SimpleNamespace(
        _table=_Table(), _isolate_cb=_Cb(), calls=[], header=[],
        _on_isolation_changed_cb=None,
        _on_navigate_cb=lambda *a: n.calls.append("navigate"),
        _on_select_label_cb=lambda *a: n.calls.append("select"),
        _select_blocked_cb=blocked,
    )
    n.selected_label_ids = lambda: {5}
    n._curation_debug = lambda *a, **k: None
    n._update_editing_header = lambda: n.calls.append("header")
    n._fire_isolation_if_active = lambda: n.calls.append("isolate")
    n._show_op_active_status = lambda text="", msec=2000: n.header.append(text)
    n._snap_back_selection = lambda lid: n.calls.append(("snap", lid))
    n._select_block_anchor = types.MethodType(N._select_block_anchor, n)
    return n


def test_blocked_row_click_moves_nothing_and_snaps_back_to_the_edit():
    n = _nav(blocked=lambda: 9)
    N._on_row_clicked(n, 0)
    assert n.calls == [("snap", 9)]
    assert n.header == [STAGED_WORK_OPEN]


def test_unblocked_row_click_still_navigates_selects_and_isolates():
    n = _nav(blocked=lambda: None)
    N._on_row_clicked(n, 0)
    assert n.calls == ["header", "navigate", "select", "isolate"]


def test_a_raising_block_check_does_not_block():
    def boom():
        raise RuntimeError("x")

    n = _nav(blocked=boom)
    N._on_row_clicked(n, 0)
    assert "navigate" in n.calls


# --- the annotator's answer: which label anchors the highlight ---------------

_FLAGS = dict(
    _split_running=False, _split_fragment_refine=False, _smooth_running=False,
    _interp_running=False, _interp_snapshot=None, _interp_refine_mode=False,
    _add_running=False, _batch_running=False, _batch_staged=False,
    _batch_refine_dialog=None, _merge_dialog=None, _draw_snapshot=None,
    _split_label_id=11, _smooth_label_id=12, _interp_label_id=13,
)


def _ann(**flags):
    f = SimpleNamespace(_refine_state=RefinementOpState(), **_FLAGS)
    f.__dict__.update(flags)
    for name in ("_sync_refine_state", "staged_kinds"):
        setattr(f, name, types.MethodType(getattr(K, name), f))
    return f


@pytest.mark.parametrize("flags,anchor", [
    ({"_split_running": True}, 11),
    ({"_smooth_running": True}, 12),
    ({"_interp_snapshot": object()}, 13),
    ({"_interp_refine_mode": True}, 13),
])
def test_blocking_kinds_anchor_on_their_own_label(flags, anchor):
    assert K._row_click_block_anchor(_ann(**flags)) == anchor


def test_fragment_refine_without_a_preview_still_selects():
    f = _ann(_split_running=True, _split_fragment_refine=True)
    assert K._row_click_block_anchor(f) is None


def test_fragment_refine_with_a_refine_preview_blocks():
    f = _ann(_split_running=True, _split_fragment_refine=True,
             _smooth_running=True)
    assert K._row_click_block_anchor(f) == 12


@pytest.mark.parametrize("flags", [
    {}, {"_add_running": True}, {"_batch_staged": True},
    {"_merge_dialog": object()},
])
def test_other_kinds_and_idle_do_not_block_a_row_click(flags):
    assert K._row_click_block_anchor(_ann(**flags)) is None


def test_navigator_gets_the_callback_wired():
    import inspect

    assert "is_select_blocked=self._row_click_block_anchor" in inspect.getsource(
        K._wire_navigator_callbacks)


def test_refused_delete_does_not_advance_and_a_done_delete_does():
    for result, advanced in ((False, 0), (None, 1), (True, 1)):
        n = SimpleNamespace(_on_delete_label_cb=lambda lid, r=result: r,
                            _auto_advance_cb=_Cb(), advanced=0)
        n.current_label_id = lambda: 7
        n._advance_to_next_visible = lambda: setattr(n, "advanced", 1)
        N._on_delete_label_clicked(n)
        assert n.advanced == advanced


def test_header_takes_the_sentence_as_an_argument():
    import inspect

    sig = inspect.signature(N._show_op_active_status)
    assert "text" in sig.parameters
    assert "Finish or reject current refinement op" not in inspect.getsource(N)
