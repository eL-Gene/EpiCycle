"""Isolate-while-painting never fights an open edit for the isolated layer."""
from __future__ import annotations

import types
from types import SimpleNamespace

from napari_epicycle.annotator._staged_op_text import STAGED_WORK_OPEN
from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator as K
from napari_epicycle.annotator.curation_label_navigator import (
    CurationLabelNavigator as N,
)
from napari_epicycle.annotator.refinement_state import StagedOp


def _ann(*, blocking=False, box=False, paint_active=False, enabled=False):
    f = SimpleNamespace(
        _paint_isolation_enabled=enabled, _paint_isolation_active=paint_active,
        _paint_isolation_label_id=4 if paint_active else None,
        isolation=[], nav=[],
    )
    kinds = frozenset({StagedOp.REFINE}) if blocking else frozenset()
    f.staged_kinds = lambda: kinds
    f._on_isolation_changed = lambda ids, **k: f.isolation.append(set(ids))
    nav = SimpleNamespace(
        is_isolate_checked=lambda: box,
        selected_label_ids=lambda: {5},
        _fire_isolation_if_active=lambda: f.nav.append("refire"),
        set_paint_isolate_checked=lambda v: f.nav.append(("paint_box", v)),
        _show_op_active_status=lambda text="", msec=2000: f.nav.append(
            ("header", text)),
    )
    f._navigator_widget = nav
    f._layer_by_name = lambda *a: SimpleNamespace(selected_label=7)
    f.viewer = SimpleNamespace(layers=[])
    f._active_labels_name = "x"
    return f


def test_toggle_on_over_a_blocking_edit_is_refused_and_unticked_silently():
    f = _ann(blocking=True)
    K._on_paint_isolation_toggled(f, True)
    assert f.isolation == []
    assert f._paint_isolation_enabled is False
    assert ("paint_box", False) in f.nav
    assert ("header", STAGED_WORK_OPEN) in f.nav


def test_toggle_on_with_nothing_open_isolates_the_brush_label():
    f = _ann()
    K._on_paint_isolation_toggled(f, True)
    assert f.isolation == [{7}]


def test_paint_label_change_during_an_edit_is_silent():
    f = _ann(blocking=True, enabled=True)
    K._on_paint_label_changed(f, 9)
    assert f.isolation == [] and f.nav == []
    assert f._paint_isolation_label_id is None


def test_toggle_off_during_an_edit_only_resets_the_flags():
    f = _ann(blocking=True, enabled=True, paint_active=True)
    K._on_paint_isolation_toggled(f, False)
    assert f.isolation == [] and f.nav == []
    assert f._paint_isolation_active is False
    assert f._paint_isolation_label_id is None


def test_toggle_off_with_the_isolate_box_ticked_refires_the_rows():
    f = _ann(box=True, enabled=True, paint_active=True)
    K._on_paint_isolation_toggled(f, False)
    assert f.nav == ["refire"] and f.isolation == []


def test_toggle_off_with_the_box_clear_clears_isolation():
    f = _ann(box=False, enabled=True, paint_active=True)
    K._on_paint_isolation_toggled(f, False)
    assert f.isolation == [set()] and f.nav == []


class _Cb:
    def __init__(self):
        self.log = []
        self.checked = True

    def blockSignals(self, v):
        self.log.append(("block", v))

    def setChecked(self, v):
        self.checked = v
        self.log.append(("set", v))

    def isChecked(self):
        return self.checked


def test_navigator_untick_and_read_helpers():
    n = SimpleNamespace(_paint_isolate_cb=_Cb(), _isolate_cb=_Cb())
    N.set_paint_isolate_checked(n, False)
    assert n._paint_isolate_cb.log == [("block", True), ("set", False),
                                       ("block", False)]
    assert N.is_isolate_checked(n) is True


def test_toggle_off_with_several_rows_ticked_renders_the_selection():
    f = _ann(box=True, enabled=True, paint_active=True)
    f._navigator_widget.selected_label_ids = lambda: {1, 2}
    f._navigator_widget._on_render_selected_clicked = lambda: f.nav.append(
        "render")
    K._on_paint_isolation_toggled(f, False)
    assert f.nav == ["render"] and f.isolation == []


def _refiring(f):
    """Isolation that re-announces the brush label, as napari does when a
    colormap is assigned back (the Labels layer re-emits selected_label)."""
    calls = f.isolation

    def iso(ids):
        calls.append(set(ids))
        assert len(calls) < 20, "isolation re-entered without end"
        K._on_paint_label_changed(f, next(iter(ids)))
    f._on_isolation_changed = iso


def test_label_change_whose_isolation_re_announces_the_label_isolates_once():
    f = _ann(enabled=True)
    _refiring(f)
    K._on_paint_label_changed(f, 9)
    assert f.isolation == [{9}]
    assert f._paint_isolation_active is True
    assert f._paint_isolation_label_id == 9


def test_toggle_on_whose_isolation_re_announces_the_label_isolates_once():
    f = _ann()
    _refiring(f)
    K._on_paint_isolation_toggled(f, True)
    assert f.isolation == [{7}]
    assert f._paint_isolation_label_id == 7


def test_a_failed_isolation_leaves_the_label_retryable():
    f = _ann(enabled=True)

    def boom(ids):
        raise RuntimeError("x")
    f._on_isolation_changed = boom
    K._on_paint_label_changed(f, 9)
    assert f._paint_isolation_label_id is None
    assert f._paint_isolation_active is False
