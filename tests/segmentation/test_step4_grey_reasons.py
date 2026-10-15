"""Every grey control of ① and ② has its reason ON SCREEN (I-9): Qt shows no tooltip on a disabled widget, so a tooltip is never
the reason."""
from __future__ import annotations

import numpy as np
import pytest
from qtpy.QtCore import Qt

from napari_epicycle.segmentation import _step4_text as T
from napari_epicycle.ui_shell.theme import MODEL_NONE

from ._step4_panel import close, make_workspace, panel, pump, tick


def _status(w):
    return w.run_status_label.toolTip() or w.run_status_label.text()


def _model(w, qapp):
    w.controller._on_model_use({"_kind": "pretrained", "name": "cyto3"})
    pump(qapp)


def _chan(w, qapp, c1=0):
    w.set_channel_choice(w.chan1_combo, c1)
    w.controller._on_channels_changed()
    pump(qapp)


def _views():
    return {v: (np.zeros((2, 6, 24, 24), np.float32),
                np.zeros((6, 24, 24), np.float32), None)
            for v in ("xy", "xz", "yz")}


@pytest.fixture
def ws(tmp_path):
    return make_workspace(tmp_path, patches=("P1.tif", "P2.tif"),
                          one_channel=("P2.tif",))


def test_no_model(qapp, ws):
    w = panel(qapp, ws)
    try:
        w.accordion.setCurrentIndex(1)
        tick(w, qapp, "P1.tif")
        assert not w.predict_btn.isEnabled()
        assert w.model_active_indicator_label.text() == MODEL_NONE
        assert w.model_active_indicator_label.isVisible()
    finally:
        close(w)


def test_model_files_missing(qapp, ws, tmp_path):
    w = panel(qapp, ws)
    try:
        w.accordion.setCurrentIndex(1)
        w.controller._on_model_use({
            "_kind": "external",
            "lateral_weights": str(tmp_path / "gone" / "l.pt")})
        pump(qapp)
        _chan(w, qapp)
        tick(w, qapp, "P1.tif")
        assert not w.predict_btn.isEnabled()
        assert w.model_trained_label.text() == T.MODEL_MISSING
        assert w.model_trained_label.isVisible()
    finally:
        close(w)


def test_blank_membrane_channel(qapp, ws):
    w = panel(qapp, ws)
    try:
        _model(w, qapp)
        tick(w, qapp, "P1.tif")
        assert w.chan1_combo.currentIndex() == -1          # blank stays blank
        w.controller._refresh_stacks()                    # a refresh...
        pump(qapp)
        assert w.chan1_combo.currentIndex() == -1          # ...keeps it blank
        assert not w.predict_btn.isEnabled()
        assert _status(w) == T.SEG_PICK_CHANNEL
    finally:
        close(w)


def test_no_ticked_stack_can_run(qapp, ws):
    from ._step4_panel import refined_bundle
    from napari_epicycle.fusion_reconstruction._run_management import (
        list_cross_run_refined_models,
    )
    from napari_epicycle.segmentation import _step4_models as M
    refined_bundle(ws, None, "two-channel", channels=(2, 1))
    w = panel(qapp, ws)
    try:
        rows = M.step4_model_rows(list_cross_run_refined_models(
            ws, trained_only=True))
        w.controller._on_model_use(dict(rows[0], _kind="refined"))
        pump(qapp)                         # trained on 2 channels
        tick(w, qapp, "P2.tif", on=True)   # the 1-channel stack
        assert not w.predict_btn.isEnabled()
        assert _status(w) == T.SEG_NONE_RUNNABLE
        row = next(r for r in w.controller._rows if r["name"] == "P2.tif")
        assert T.ROW_WARN_CHANNELS.format(n=2, k=1) in row["warns_narrow"]
    finally:
        close(w)


def test_nothing_to_reconstruct(qapp, ws):
    w = panel(qapp, ws)
    try:
        _model(w, qapp)
        _chan(w, qapp)
        tick(w, qapp, "P1.tif")
        assert w.predict_btn.isEnabled()
        assert not w.reconstruct_btn.isEnabled()
        # F1: the idle line shows, and says why Reconstruct is grey
        assert _status(w) == T.SEG_TICKED_JOIN.join(
            (T.seg_idle_line(1), T.SEG_NOTHING_TO_RECONSTRUCT))
    finally:
        close(w)


def test_nothing_ticked(qapp, ws):
    """F1 (row added): with a model and nothing ticked, Predict is
    grey and says so, instead of "cannot run"."""
    w = panel(qapp, ws)
    try:
        _model(w, qapp)
        _chan(w, qapp)
        tick(w, qapp, "P1.tif", on=False)
        assert not w.predict_btn.isEnabled()
        assert _status(w) == T.SEG_NOTHING_TICKED
    finally:
        close(w)


def test_a_model_reason_stops_at_the_card(qapp, ws):
    """F1: with no model the card says why; the status line adds no
    second reason ("Predict first" etc.)."""
    w = panel(qapp, ws)
    try:
        w.accordion.setCurrentIndex(1)
        tick(w, qapp, "P1.tif")
        assert not w.predict_btn.isEnabled()
        assert _status(w) == ""
    finally:
        close(w)


def test_a_stack_that_cannot_run_stays_ticked(qapp, ws):
    """F2: a ticked stack that cannot run keeps its stored tick, so the
    skipped count keeps showing; ticking another does not drop it."""
    from ._step4_panel import refined_bundle
    from napari_epicycle.fusion_reconstruction._run_management import (
        list_cross_run_refined_models,
    )
    from napari_epicycle.segmentation import _step4_models as M
    refined_bundle(ws, None, "two-channel", channels=(2, 1))
    w = panel(qapp, ws)
    try:
        c = w.controller
        c.state_manager.set_ticked(["P2.tif"])
        rows = M.step4_model_rows(list_cross_run_refined_models(
            ws, trained_only=True))
        c._on_model_use(dict(rows[0], _kind="refined"))   # 2 channels
        pump(qapp)
        row = next(r for r in c._rows if r["name"] == "P2.tif")
        assert not row["tickable"]          # its box is greyed now
        tick(w, qapp, "P1.tif")
        pump(qapp, 5)
        assert set(c.state_manager.get_ticked()) == {"P1.tif", "P2.tif"}
        assert "1 cannot run" in w.seg_ticked_label.text()
    finally:
        close(w)


def test_view_without_a_result(qapp, ws):
    w = panel(qapp, ws)
    try:
        c = w.controller
        _model(w, qapp)
        _chan(w, qapp)
        snap = c._model_snapshot()
        c._on_prediction_completed("P1.tif", {
            "views": _views(),
            "source_stack_path": str(ws / "artifacts/step1/patches/P1.tif"),
            "model_snapshot": snap, "cache_meta": {"model": snap,
                                                   "axes": "CZYX"}})
        tick(w, qapp, "P1.tif")
        w.stacks_list.setCurrentRow(0)
        pump(qapp)
        assert w.reconstruct_btn.isEnabled()
        assert not w.view_btn.isEnabled()
        # the highlighted row's status word says it has no result; the
        # composed line does not repeat it (re-audit)
        assert _status(w) == T.seg_idle_line(1)
        # a highlighted stack with a result: View is live and the line idle
        c.state_manager.set_stack_status("P1.tif", "reconstructed",
                                         labels_path=str(ws / "l.tif"))
        c._refresh_stacks()
        w.stacks_list.setCurrentRow(0)
        pump(qapp)
        assert w.view_btn.isEnabled()
        assert _status(w) == T.seg_idle_line(1) == "Ready · 1 stack ticked"
    finally:
        close(w)


def test_cancel_is_grey_with_nothing_running(qapp, ws):
    w = panel(qapp, ws)
    try:
        assert not w.cancel_btn.isEnabled()
    finally:
        close(w)


def test_named_runs_reason_is_on_screen(qapp, ws):
    w = panel(qapp, ws)
    try:
        w.accordion.setCurrentIndex(1)
        pump(qapp, 5)
        assert not w.new_run_btn.isEnabled()
        assert not w.manage_runs_btn.isEnabled()
        assert w.run_row_note.isVisible()
        assert w.run_row_note.text() == T.RUN_ROW_AFTER_RELEASE
    finally:
        close(w)


def test_set_spacing_reasons(qapp, tmp_path):
    empty = make_workspace(tmp_path / "e", patches=())
    w = panel(qapp, empty)
    try:
        assert not w.set_spacing_btn.isEnabled()
        assert w.set_spacing_note.isVisible()
        assert w.set_spacing_note.text() == T.SET_SPACING_NEEDS_STACK
    finally:
        close(w)
    ws = make_workspace(tmp_path / "f", patches=("P1.tif",))
    w = panel(qapp, ws)
    try:
        assert not w.set_spacing_btn.isEnabled()
        assert w.set_spacing_note.text() == T.SET_SPACING_NEEDS_TICK
        tick(w, qapp, "P1.tif")
        assert w.set_spacing_btn.isEnabled()
        assert not w.set_spacing_note.isVisible()
    finally:
        close(w)


def test_clear_predictions_on_a_3d_stack_says_why_in_its_text(qapp, ws):
    w = panel(qapp, ws)
    try:
        c = w.controller
        c.state_manager.set_stack_status("P1.tif", "reconstructed",
                                         labels_path="x")
        menu, clear, spacing = c._build_stack_menu("P1.tif")
        assert not clear.isEnabled()
        assert clear.text() == T.MENU_CLEAR_PREDICTIONS_HAS_3D
        assert spacing.text() == T.MENU_SET_SPACING
        menu.deleteLater()
        menu, clear, _s = c._build_stack_menu("P2.tif")
        assert clear.isEnabled() and clear.text() == T.MENU_CLEAR_PREDICTIONS
        menu.deleteLater()
    finally:
        close(w)


def test_a_row_that_cannot_run_says_why_on_its_row(qapp, tmp_path):
    from ._step4_panel import write_stack
    ws = make_workspace(tmp_path, patches=("P1.tif",))
    write_stack(tmp_path / "extra" / "P1.tif")
    w = panel(qapp, ws)
    try:
        w.controller._scan_stacks_directory(str(tmp_path / "extra"))
        pump(qapp)
        items = [w.stacks_list.item(i) for i in range(w.stacks_list.count())]
        dup = [it for it in items
               if (it.data(Qt.UserRole + 2) or {}).get("conflict")]
        assert len(dup) == 1
        assert not (dup[0].flags() & Qt.ItemIsUserCheckable)
        assert T.ROW_WARN_SAME_NAME in dup[0].data(Qt.UserRole + 2)[
            "warns_narrow"]
    finally:
        close(w)


def _greyed_and_ticked(qapp, ws):
    """P2 has one channel; a two-channel model greys it while it is
    ticked (with P1)."""
    from ._step4_panel import refined_bundle
    from napari_epicycle.fusion_reconstruction._run_management import (
        list_cross_run_refined_models,
    )
    from napari_epicycle.segmentation import _step4_models as M
    refined_bundle(ws, None, "two-channel", channels=(2, 1))
    w = panel(qapp, ws)
    c = w.controller
    c.state_manager.set_ticked(["P1.tif", "P2.tif"])
    c._refresh_stacks()
    rows = M.step4_model_rows(list_cross_run_refined_models(
        ws, trained_only=True))
    c._on_model_use(dict(rows[0], _kind="refined"))   # 2 channels
    pump(qapp)
    return w, c


def _item(lw, name):
    for i in range(lw.count()):
        if lw.item(i).data(Qt.UserRole) == name:
            return lw.item(i)
    return None


def test_a_greyed_ticked_stack_can_be_unticked(qapp, ws):
    """Decided 2026-10-05: a greyed stack that was ticked shows
    its tick and can be unticked; the counts and the gate follow."""
    w, c = _greyed_and_ticked(qapp, ws)
    try:
        assert not next(r for r in c._rows
                        if r["name"] == "P2.tif")["tickable"]
        it = _item(w.stacks_list, "P2.tif")
        assert it.checkState() == Qt.Checked           # its tick is shown
        assert it.flags() & Qt.ItemIsUserCheckable      # and can be taken
        assert "1 cannot run" in w.seg_ticked_label.text()
        it.setCheckState(Qt.Unchecked)                 # the user's untick
        pump(qapp, 5)
        assert c.state_manager.get_ticked() == ["P1.tif"]
        assert "cannot run" not in w.seg_ticked_label.text()
        # once unticked it can never be ticked while greyed
        it = _item(w.stacks_list, "P2.tif")
        assert not it.flags() & Qt.ItemIsUserCheckable
        assert _item(w.session_stacks_list, "P2.tif").checkState() \
            == Qt.Unchecked
    finally:
        close(w)


def test_a_greyed_stack_cannot_be_ticked_through_the_delegate(qapp, ws):
    """The delegate refuses a tick on a greyed row, whatever its flags."""
    from qtpy.QtCore import QEvent, QPointF
    from qtpy.QtGui import QMouseEvent
    from qtpy.QtWidgets import QStyleOptionViewItem
    w, c = _greyed_and_ticked(qapp, ws)
    try:
        lw = w.stacks_list
        _item(lw, "P2.tif").setCheckState(Qt.Unchecked)
        pump(qapp, 5)
        it = _item(lw, "P2.tif")
        it.setFlags(it.flags() | Qt.ItemIsUserCheckable)   # even so
        idx = lw.indexFromItem(it)
        opt = QStyleOptionViewItem()
        opt.rect = lw.visualItemRect(it)
        opt.widget = lw
        deleg = lw.itemDelegate()
        g, _rec = deleg._geometry_at(opt, idx)
        pos = QPointF(g["tick"].center())
        ev = QMouseEvent(QEvent.MouseButtonRelease, pos, pos, Qt.LeftButton,
                         Qt.LeftButton, Qt.NoModifier)
        deleg.editorEvent(ev, lw.model(), opt, idx)
        pump(qapp, 5)
        assert _item(lw, "P2.tif").checkState() == Qt.Unchecked
        assert "P2.tif" not in c.state_manager.get_ticked()
    finally:
        close(w)


def test_under_a_lock_a_greyed_tick_cannot_be_taken_either(qapp, ws):
    w, c = _greyed_and_ticked(qapp, ws)
    try:
        c._set_step4_flag("_measure_running", True)
        pump(qapp)
        it = _item(w.stacks_list, "P2.tif")
        assert not it.flags() & Qt.ItemIsUserCheckable
        c._set_step4_flag("_measure_running", False)
        pump(qapp)
        assert _item(w.stacks_list, "P2.tif").flags() \
            & Qt.ItemIsUserCheckable
        assert "P2.tif" in c.state_manager.get_ticked()
    finally:
        close(w)
