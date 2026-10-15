"""Step 2 look, polish pass (fix pass 5): the look constants (phase 1) and the
room and gates (phase 2).  Qt-backed: run on its own with ``-o addopts=""``."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest                                                         # noqa: E402
from qtpy.QtCore import Qt                                            # noqa: E402
from qtpy.QtWidgets import (QApplication, QComboBox, QPushButton,    # noqa: E402
                            QToolButton)

sys.path.insert(0, os.path.dirname(__file__))
from test_step2_curation import _real_widget                          # noqa: E402
from test_step2_fix_pass3 import _in_dock, _open                      # noqa: E402

from napari_epicycle.hitl_training_annotation.ui import _step2_lists as L  # noqa: E402
from napari_epicycle.hitl_training_annotation.ui import _step2_text as T   # noqa: E402
from napari_epicycle.ui_shell import theme as TH                      # noqa: E402
from napari_epicycle.ui_shell._recon_settings_widgets import (        # noqa: E402
    ENGINE_NAME_STYLE, LockLine as SharedLockLine)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


# ===========================================================================
# PHASE 1 -- the look constants
# ===========================================================================

# -- theme (additive) --------------------------------------------------------
def test_chip_checked_pair_is_the_one_in_panel_chip_style():
    assert TH.PANEL_CHIP_CHECKED_BG in TH.PANEL_CHIP_STYLE
    assert TH.PANEL_CHIP_CHECKED_BORDER in TH.PANEL_CHIP_STYLE


def test_progress_bar_style_height_param_keeps_the_default():
    assert "min-height: 14px; max-height: 14px;" in TH.PROGRESS_BAR_STYLE
    s = TH.progress_bar_style("#d4a843", 8)
    assert "min-height: 8px; max-height: 8px;" in s
    assert TH.progress_bar_style("#d4a843") == TH.PROGRESS_BAR_STYLE


def test_derived_styles_carry_their_parents_values():
    assert "background-color: rgba(255, 255, 255, 0.05)" in \
        TH.SECONDARY_TOOLBTN_STYLE
    assert "QPushButton" not in TH.SECONDARY_TOOLBTN_STYLE
    d = TH.disclosure_btn_style("x")
    assert "QPushButton#x {" in d and "padding: 6px 10px" in d
    assert "QCollapsible" not in d
    assert TH.panel_foot_style("f") == \
        "QWidget#f { border-top: 1px solid #555555; }"


def test_combo_popup_style_has_amber_item_rules():
    s = TH.PANEL_COMBO_POPUP_STYLE
    assert "::item:selected" in s and "::item:hover" in s
    assert "rgba(212, 168, 67, 0.35)" in s


# -- words ---------------------------------------------------------------------
def test_the_top_hint_says_finalize_not_stage():
    assert "finalize" in T.FLOW_HINT and "stage" not in T.FLOW_HINT
    assert T.FLOW_HINT == ("Select a patch → curate slices in the active "
                           "view → finalize → train a round")


def test_the_curation_lock_line_does_not_claim_the_run_is_read_only():
    assert "the run" not in T.CUR_LOCK_LINE
    assert "slices and the model" in T.CUR_LOCK_LINE


def test_run_picker_label_words_and_numbers():
    f = T.run_picker_label
    # counts of trained rounds in, the LAST round's disk number out
    assert f("Cell", "whole cell", 3, 2, 83) == \
        "Cell (whole cell · lat 2 · ax 1 · 83 pairs)"
    assert f("Default", "", 1, 0, 1) == "Default (lat 0 · ax — · 1 pair)"
    assert f("x", "", "?", None, "?") == "x (lat — · ax — · 0 pairs)"


def test_the_card_last_round_agrees_with_the_picker():
    """picker lat 2 (3 rounds trained) == the card's Last round 2"""
    rounds = L.run_summary_rounds({"lateral": 3, "axial": 2}, "Cell")
    assert rounds["lateral"]["last"] == "2" and rounds["axial"]["last"] == "1"
    assert "lat 2 · ax 1" in T.run_picker_label("Cell", "", 3, 2, 83)


def test_the_controller_picker_label_uses_the_words_module():
    from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
        TrainingAnnotationController as Ctl)
    s = {"display_name": "Cell", "target": "whole cell",
         "lateral_round_count": 3, "axial_round_count": 2,
         "accepted_pair_count": 83}
    assert Ctl._format_run_picker_label(s) == \
        "Cell (whole cell · lat 2 · ax 1 · 83 pairs)"


# -- the real widget -------------------------------------------------------------
def _icons_resolve() -> bool:
    return TH._qta_icon_or_none("mdi6.brain", "#ffffff") is not None


def test_primaries_are_translucent_and_every_secondary_is_one_kind(qapp):
    w = _real_widget(qapp)
    for name in ("launch_multiview_button", "update_and_train_button"):
        assert getattr(w, name).styleSheet() == TH.PRIMARY_BTN_STYLE, name
    for name in ("accept_annotation_button", "stage_current_view_button",
                 "reject_annotation_button", "unstage_selected_button",
                 "train_cancel_btn", "training_history_button",
                 "new_run_button", "set_workdir_browse_button",
                 "train_patch_browse_button", "train_patch_refresh_button"):
        assert getattr(w, name).styleSheet() == TH.SECONDARY_BTN_STYLE, name
    manage = w.curation_run_row.manage_menu_button
    assert isinstance(manage, QToolButton)
    assert manage.styleSheet() == TH.SECONDARY_TOOLBTN_STYLE
    if _icons_resolve():
        assert not manage.icon().isNull()
    # no solid-gold sheet is left on the panel
    from napari_epicycle.hitl_training_annotation.ui import main_widget as MW
    src = open(MW.__file__, encoding="utf-8").read()
    assert "panel_primary_btn_style" not in src


def test_train_and_cancel_carry_their_icons_and_share_a_height(qapp):
    w = _real_widget(qapp)
    t, c = w.update_and_train_button, w.train_cancel_btn
    if _icons_resolve():
        assert not t.icon().isNull() and not c.icon().isNull()
    assert t.height() == c.height() or t.minimumHeight() == c.minimumHeight()
    assert t.minimumHeight() == t.maximumHeight() == c.minimumHeight()


def test_the_view_switch_checked_segment_is_amber(qapp):
    w = _real_widget(qapp)
    b = w.curation_view_switch.buttons["lateral"]
    assert TH.PANEL_CHIP_CHECKED_BG in b.styleSheet()


def test_filter_chips_keep_the_pill_but_check_in_amber(qapp):
    w = _real_widget(qapp)
    s = w.filter_predicted_btn.styleSheet()
    assert "border-radius: 10px" in s                         # the v8 pill
    assert TH.PANEL_CHIP_CHECKED_BG in s
    assert TH.PANEL_CHIP_CHECKED_BORDER in s
    assert "rgba(255,255,255,0.12)" not in s                  # not white


def test_slices_header_patch_is_grey_mono_not_amber(qapp):
    w = _real_widget(qapp)
    s = w.slices_header_patch.styleSheet()
    assert "#d4a843" not in s and "#999999" in s and "Consolas" in s


def test_training_lock_line_is_the_shared_left_edge_one(qapp):
    w = _real_widget(qapp)
    ll = w.training_panel.lock_line
    assert isinstance(ll, SharedLockLine)
    assert "border-left: 3px solid #d4a843" in ll.styleSheet()
    if _icons_resolve():
        assert not ll._icon.pixmap().isNull()
    ll.set(True, "Cell_lateral_round_03", "lateral")
    assert not ll.isHidden() and "Round 3 (lateral)" in ll.text()
    ll.set(False)
    assert ll.isHidden() and ll.text() == ""


def test_loss_header_is_the_boxed_disclosure_bar(qapp):
    w = _real_widget(qapp)
    loss = w.training_foot.loss
    b = loss.toggle_btn
    assert isinstance(b, QPushButton)
    assert b.styleSheet() == TH.disclosure_btn_style("step2LossToggle")
    assert b.text().startswith("▶") and "Loss curves" in b.text()
    b.setChecked(True)
    assert b.text().startswith("▼")
    b.setChecked(False)
    assert b.text().startswith("▶")
    # the caption sits INSIDE the bar, in the engine-name grey
    assert loss.caption_label.parent() is b
    assert loss.caption_label.styleSheet() == ENGINE_NAME_STYLE


def test_engine_names_use_the_engine_name_style(qapp):
    w = _real_widget(qapp)
    for tag in w.training_setup_column.engine_labels:
        assert tag.engine_label.styleSheet() == ENGINE_NAME_STYLE


def test_disclosures_are_step3s_collapsible(qapp):
    w = _real_widget(qapp)
    for c in (w._inference_collapsible, w._training_adv_collapsible):
        assert getattr(c, "_epicycle_collapsed_symbol", "") == "▶"
        m = c.layout().contentsMargins()
        assert (m.left(), m.top(), m.right(), m.bottom()) == (0, 0, 0, 0)


def test_the_gpu_line_uses_theme_colours(qapp):
    w = _real_widget(qapp)
    src = open(sys.modules[type(w).__module__].__file__,
               encoding="utf-8").read()
    assert "#4caf50" not in src and "#ff9800" not in src


def test_the_root_sheet_carries_the_amber_popup_rule(qapp):
    w = _real_widget(qapp)
    assert "QComboBox QAbstractItemView::item:selected" in w.styleSheet()


def test_feet_carry_the_top_rule_and_the_busy_bar_is_14(qapp):
    w = _real_widget(qapp)
    for foot in (w.curation_foot, w.training_foot):
        assert "border-top: 1px solid #555555" in foot.styleSheet()
    # 14 px in a tall dock (a short one shrinks it to 8, phase 2)
    w.curation_foot.set_short(False)
    w.training_foot.set_short(False)
    assert w.patch_busy_bar.maximumHeight() == 14
    assert w.training_progress_bar.maximumHeight() == 14


def test_the_stage_line_is_mono_in_both_feet(qapp):
    w = _real_widget(qapp)
    assert w.curation_foot.stage_label.styleSheet() == TH.MONO_LABEL_STYLE
    assert w.training_foot.stage_label.styleSheet() == TH.MONO_LABEL_STYLE


def test_the_run_picker_follows_the_column_not_the_text(qapp):
    w = _real_widget(qapp)
    p = w.run_picker_combo
    assert p.sizeAdjustPolicy() == \
        QComboBox.AdjustToMinimumContentsLengthWithIcon


def test_left_edges_line_up(qapp):
    """Run: and Patches; Staged for training and Train (polish D-F)."""
    w = _real_widget(qapp)
    sa = _in_dock(qapp, w, 576, 600)
    _open(qapp, w, 1)
    run_lbl = w.curation_run_row.findChildren(type(w.slices_header_patch))[0]
    patches = w.train_patch_list
    x_run = run_lbl.mapTo(w, run_lbl.rect().topLeft()).x()
    x_pat = patches.mapTo(w, patches.rect().topLeft()).x()
    assert abs(x_run - x_pat) <= 2, (x_run, x_pat)
    _open(qapp, w, 2)
    staged = w.training_staged_column.title_label
    train = w.update_and_train_button
    xs = staged.mapTo(w, staged.rect().topLeft()).x()
    xt = train.mapTo(w, train.rect().topLeft()).x()
    assert abs(xs - xt) <= 2, (xs, xt)
    sa.close()


def test_chip_words_follow_the_column_width_not_the_dock_height(qapp):
    w = _real_widget(qapp)
    sa = _in_dock(qapp, w, 576, 470)          # SHORT dock, narrow column
    _open(qapp, w, 1)
    recs = {"s%d" % i: {"sample_id": "s%d" % i, "patch_id": "p",
                        "bucket": "lateral", "status": "predicted",
                        "slice_index": i} for i in range(3)}
    w.on_slice_rows({"records": recs, "workspace": "ws", "ids": list(recs),
                     "bucket": "lateral", "patch": "p",
                     "loaded_patch": "p"})
    qapp.processEvents()
    assert w._chip_column.width() < 330
    assert w.filter_predicted_btn.text() == "3"          # icon + count
    assert w.filter_predicted_btn.toolTip() == "Predicted 3"
    # the same short dock, 1000 wide: the column is wide, the words return
    sa.resize(1000, 470)
    qapp.processEvents()
    qapp.processEvents()
    assert w._chip_column.width() >= 330, w._chip_column.width()
    assert w.filter_predicted_btn.text() == "Predicted 3"
    assert w.filter_predicted_btn.toolTip() == ""
    # and a TALL dock but a narrow column keeps the compact form
    sa.resize(576, 900)
    qapp.processEvents()
    qapp.processEvents()
    assert w._chip_column.width() < 330
    assert w.filter_predicted_btn.text() == "3"
    sa.close()


# ===========================================================================
# PHASE 2 -- the room and the gates
# ===========================================================================
from types import SimpleNamespace                                      # noqa: E402

from qtpy.QtCore import QPoint                                         # noqa: E402
from qtpy.QtWidgets import QLabel, QListWidgetItem                     # noqa: E402

from napari_epicycle.hitl_training_annotation.ui import _step2_curation as C  # noqa: E402

#: the dock height whose Curation section has the real 576 px dock's room
#: (~266 px under the section header; the shots measure it)
REAL_DOCK_H = 487
_QMAX = 16777215


def _short_dock(qapp, height=REAL_DOCK_H, section=1):
    w = _real_widget(qapp)
    sa = _in_dock(qapp, w, 576, height)
    _open(qapp, w, section)
    w._curation_short = None
    w._apply_curation_room()
    for _ in range(4):
        qapp.processEvents()
    return w, sa


def _shown(lab):
    """What the label DRAWS (``text()`` is the full text)."""
    return QLabel.text(lab)


# -- ElidedLabel -------------------------------------------------------------------
def test_elided_label_keeps_the_full_text_and_hovers_it(qapp):
    lab = C.ElidedLabel("", None, 2)
    lab.resize(200, 30)
    long = ("Active: lat + ax: domain-specialized_Bootrap4Finetuned_axial_"
            "500epoch (file)")
    lab.setText(long)
    assert lab.text() == long
    assert lab.toolTip() == long and not lab.wordWrap()
    assert _shown(lab) != long and "…" in _shown(lab)
    lab.setText("short")
    assert _shown(lab) == "short" and lab.toolTip() == ""


def test_elided_label_wraps_what_fits_in_two_lines(qapp):
    lab = C.ElidedLabel("", None, 2)
    text = "aaaaaaaa bbbbbbbb cccccccc dddddddd"
    lab.setText(text)
    lab.ensurePolished()
    # scan the width: too narrow -> one elided line (full text on hover);
    # from some width on the text shows whole (wrapped in <= 2 lines) and
    # stays whole as the label widens -- whatever the font
    states = []
    for width in range(40, 700, 6):
        lab.resize(width, 40)
        lab.set_max_lines(2)       # (a hidden label gets no resize event)
        whole = _shown(lab) == text
        assert whole == (lab.toolTip() == ""), (width, _shown(lab))
        assert whole == lab.wordWrap() or not whole, width
        states.append(whole)
    assert states[0] is False and states[-1] is True
    assert states == sorted(states)            # False... then True...
    assert any(states)


def test_the_active_line_of_an_external_model_never_overflows(qapp):
    w, sa = _short_dock(qapp)
    lab = w.active_model_summary_value
    full = ("Active: lat + ax: domain-specialized_Bootrap4Finetuned_axial_"
            "500epoch_scratch_lenie (file)")
    lab.setText(full)
    qapp.processEvents()
    assert lab.text() == full and lab.toolTip() == full
    assert not lab.wordWrap()
    assert lab.fontMetrics().horizontalAdvance(_shown(lab)) <= lab.width() + 1
    sa.close()


# -- Curation, short mode -------------------------------------------------------------
def test_short_mode_folds_the_view_and_model_headers_into_their_rows(qapp):
    w, sa = _short_dock(qapp)
    rc = w.curation_right_column
    assert rc._descs[0][0].isHidden() and rc._descs[2][0].isHidden()
    assert not rc.view_help.isHidden() and not rc.model_help.isHidden()
    assert rc.view_help.parent() is rc.switch_row
    assert w.curation_model_block.use_row.isAncestorOf(rc.model_help)
    assert rc.scroll.minimumHeight() == C.SCROLL_MIN_H_SHORT
    # tall again: headers back, the extra rings hidden
    sa.resize(576, 1100)
    qapp.processEvents()
    qapp.processEvents()
    assert w._curation_short is False
    assert not rc._descs[0][0].isHidden() and not rc._descs[2][0].isHidden()
    assert rc.view_help.isHidden() and rc.model_help.isHidden()
    assert rc.scroll.minimumHeight() == C.SCROLL_MIN_H
    sa.close()


def test_short_mode_use_and_active_follow_the_counts_without_a_hole(qapp):
    w, sa = _short_dock(qapp)
    rc = w.curation_right_column
    mb = w.curation_model_block
    room = rc.height() - rc.foot.height()
    bottoms = sorted(
        b.mapTo(rc.content, QPoint(0, b.height())).y()
        for b in rc._blocks if not b.isHidden() and b.height() > 0)
    chrome = C.snap_cap(rc.scroll, 0) - C.SNAP_SLACK
    fits = [b for b in bottoms
            if rc.scroll.minimumHeight() <= b and b + chrome <= room]
    cut = rc.scroll.maximumHeight()
    # the viewport ends at the LAST block boundary that fits: no snap hole
    # (whatever the font, whether Use and Active fit is the room's call)
    assert cut == (C.snap_cap(rc.scroll, max(fits)) if fits else _QMAX), (
        cut, bottoms, room)
    if bottoms[-1] <= room:
        # the room holds everything: Use and Active follow the counts whole
        vp = rc.scroll.viewport().height()
        for part in (mb.use_row, mb.active_label):
            bottom = part.mapTo(rc.content, QPoint(0, part.height())).y()
            assert bottom <= vp, (part, bottom, vp)
    sa.close()


def test_short_foot_has_the_8px_bar_and_one_elided_stage_line(qapp):
    w, sa = _short_dock(qapp)
    f = w.curation_foot
    assert f.busy_bar.maximumHeight() == 8
    assert "min-height: 8px" in f.busy_bar.styleSheet()
    f.set_stage("Ready · WholeScale_patch_8 · lateral · " + "x" * 120)
    qapp.processEvents()
    assert not f.stage_label.wordWrap()
    assert f.stage_label.toolTip().startswith("Ready · WholeScale_patch_8")
    sa.resize(576, 1100)
    qapp.processEvents()
    qapp.processEvents()
    assert f.busy_bar.maximumHeight() == 14
    sa.close()


def test_short_mode_patches_rows_use_the_56px_picture(qapp):
    w, sa = _short_dock(qapp)
    lst = w.train_patch_list
    it = QListWidgetItem("WholeScale_patch_8.tiff")
    it.setData(Qt.UserRole, "WholeScale_patch_8.tiff")
    lst.addItem(it)
    payload = {"records": {}, "shapes": {}, "round_by_view": {}, "loaded": "",
               "view": "XY", "workspace": "", "stack_dir": "", "run_id": "x"}
    w._patch_lists._last = payload
    w._patch_lists._refill_now()
    rec = lst.item(0).data(L.ROLE_RECORD)
    assert rec["thumb_px"] == 56
    one_row = lst.sizeHintForRow(0)
    assert w.train_patch_list.minimumHeight() >= one_row, (
        w.train_patch_list.minimumHeight(), one_row)
    sa.resize(576, 1100)
    qapp.processEvents()
    qapp.processEvents()
    assert "thumb_px" not in lst.item(0).data(L.ROLE_RECORD)
    assert w.train_patch_list.minimumHeight() == 70
    sa.close()


def test_the_curation_lock_state_fits_the_dock(qapp):
    w, sa = _short_dock(qapp)
    w.on_training_locked(True, "Cell_lateral_round_03", "lateral")
    for _ in range(4):
        qapp.processEvents()
    vp = sa.viewport()
    assert sa.verticalScrollBar().maximum() == 0
    ll = w.curation_lock_line
    assert not ll.isHidden()
    assert ll.mapTo(vp, QPoint(0, 0)).y() >= 0
    st = w.curation_foot.stage_label
    assert st.text().startswith("Locked · round 3")
    assert st.mapTo(vp, QPoint(0, st.height())).y() <= vp.height()
    f = w.curation_foot
    assert f.mapTo(vp, QPoint(0, f.height())).y() <= vp.height()
    sa.close()


# -- Training, short mode ---------------------------------------------------------------
def test_training_short_foot_is_one_elided_line_and_an_8px_bar(qapp):
    w, sa = _short_dock(qapp, 470, 2)
    f = w.training_foot
    f.set_stage_line("Ready · Train writes Cell_lateral_round_03 from 14 "
                     "lateral pairs (run Cell). Cancel works while a round "
                     "trains.")
    qapp.processEvents()
    assert f.progress_bar.maximumHeight() == 8
    lab = f.stage_label
    assert not lab.wordWrap()
    assert lab.height() <= lab.fontMetrics().height() + 4
    assert lab.toolTip().startswith("Ready · Train writes")
    assert lab.text().endswith("trains.")            # text() is the FULL text
    sa.resize(576, 1100)
    qapp.processEvents()
    qapp.processEvents()
    assert f.progress_bar.maximumHeight() == 14
    assert lab.wordWrap()
    sa.close()


def test_training_bar_shows_epoch_over_epochs_while_a_round_runs(qapp):
    w, sa = _short_dock(qapp, 1100, 2)           # tall: the bar has room
    p = w.training_panel
    bar = w.training_progress_bar
    assert not bar.isTextVisible()
    p.on_train_started("lateral", "Cell_lateral_round_03")
    p.on_epoch(412, 1000, 0.81)
    assert bar.format() == "412 / 1000" and bar.isTextVisible()
    p.on_training_locked(False, "", "")
    assert not bar.isTextVisible()
    sa.close()


def test_training_columns_snap_to_a_block_boundary_in_short_mode(qapp):
    w, sa = _short_dock(qapp, 470, 2)
    p = w.training_panel
    p.snap_columns()
    for sc in (p.left_scroll, p.right_scroll):
        lay = sc.widget().layout()
        bottoms = {lay.itemAt(i).geometry().bottom() + 1
                   for i in range(lay.count())
                   if lay.itemAt(i).spacerItem() is None}
        m = sc.maximumHeight()
        assert m == _QMAX or any(
            C.snap_cap(sc, b) == m for b in bottoms), (m, sorted(bottoms))
    sa.resize(576, 1100)
    qapp.processEvents()
    qapp.processEvents()
    p.snap_columns()
    assert p.left_scroll.maximumHeight() == _QMAX
    assert p.right_scroll.maximumHeight() == _QMAX
    sa.close()


def test_scroll_snap_never_cuts_inside_a_block(qapp):
    from qtpy.QtWidgets import QVBoxLayout, QWidget
    from napari_epicycle.hitl_training_annotation.ui import _step2_training as TR
    content = QWidget()
    lay = QVBoxLayout(content)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(0)
    for i in range(6):
        lab = QLabel(f"row {i}")
        lab.setFixedHeight(30)
        lay.addWidget(lab)
    lay.addStretch(1)
    sc = TR._Scroll(content)
    sc.resize(200, 100)
    sc.show()
    qapp.processEvents()
    sc.snap_viewport(100)
    assert sc.maximumHeight() == C.snap_cap(sc, 90)   # three whole rows
    sc.snap_viewport(0)
    assert sc.maximumHeight() == _QMAX
    sc.snap_viewport(500)                          # it all fits
    assert sc.maximumHeight() == _QMAX
    sc.close()


# -- S2c, the gate ---------------------------------------------------------------------------
_GATED = ("accept_annotation_button", "stage_current_view_button",
          "reject_annotation_button", "unstage_selected_button")


def _states(w):
    return [getattr(w, n).isEnabled() for n in _GATED]


def test_s2c_no_slices_greys_the_four_foot_buttons(qapp):
    w = _real_widget(qapp)
    assert _states(w) == [True] * 4
    w.refresh_stage_line()                      # no slice rows at all
    assert _states(w) == [False] * 4
    w.segment_slice_data_list.addItem(QListWidgetItem("s1"))
    w.refresh_stage_line()
    assert _states(w) == [True] * 4


def test_the_gate_never_fights_the_training_lock(qapp):
    w = _real_widget(qapp)
    w.segment_slice_data_list.addItem(QListWidgetItem("s1"))
    w.on_training_locked(True, "Cell_lateral_round_03", "lateral")
    for n in _GATED:
        getattr(w, n).setEnabled(False)          # the lock list's own write
    w.refresh_stage_line()
    assert _states(w) == [False] * 4
    # unlock: the controller re-enables, the hook runs, slices exist -> live
    for n in _GATED:
        getattr(w, n).setEnabled(True)
    w.on_training_locked(False, "", "")
    assert _states(w) == [True] * 4
    # a view with NO slices stays grey after the unlock
    w.segment_slice_data_list.clear()
    w.on_training_locked(True, "Cell_lateral_round_03", "lateral")
    for n in _GATED:
        getattr(w, n).setEnabled(False)
    for n in _GATED:
        getattr(w, n).setEnabled(True)           # unlock's re-enable
    w.on_training_locked(False, "", "")
    assert _states(w) == [False] * 4


# -- the Run summary's pair count ---------------------------------------------------------------
def test_run_summary_pairs_show_for_the_active_bucket_only(qapp):
    w = _real_widget(qapp)
    card = w.run_summary_card
    card.set_active_bucket("lateral")
    assert not card._pair_cells["lateral"].isHidden()
    assert card._pair_cells["axial"].isHidden()
    d = card._pair_dash["axial"]
    assert not d.isHidden() and d.text() == "—"
    assert d.toolTip() == "Shown for the active view."
    card.update_from({"workspace": "ws", "view": "XZ", "round_by_view": {},
                      "run_id": "Cell"})
    assert card._pair_cells["lateral"].isHidden()
    assert not card._pair_cells["axial"].isHidden()
    assert not card._pair_dash["lateral"].isHidden()
    # the View switch drives it too
    w.get_view_axial.setChecked(True)
    w.get_view_lateral.setChecked(True)
    assert not card._pair_cells["lateral"].isHidden()


# -- the staged row's shape line --------------------------------------------------------------------
def test_a_staged_row_offers_shorter_spellings_of_its_shape_line():
    meta = SimpleNamespace(shape=(240, 240), n_channels=2, n_labels=127,
                           has_mask=True, reason="")
    rec = L.build_staged_row("WholeScale_patch_2_XY_26",
                             {"status": "staged", "bucket": "lateral",
                              "patch_id": "WholeScale_patch_2"}, meta=meta)
    v = rec["shape_variants"]
    assert v[0] == "240 × 240 · 2 ch · 127 labels"
    assert len(v) >= 2 and all(len(a) > len(b) for a, b in zip(v, v[1:]))


def test_the_staged_shape_line_is_one_line_at_a_narrow_width(qapp):
    base = {"status": "staged", "bucket": "lateral", "patch_id": "p"}

    def row_h(n_labels):
        m = SimpleNamespace(shape=(240, 240), n_channels=2, n_labels=n_labels,
                            has_mask=True, reason="")
        lst = L.build_staged_list(None)
        lst.resize(215, 200)
        lst.show()
        it = QListWidgetItem("s1")
        it.setData(Qt.UserRole, "s1")
        lst.addItem(it)
        L.apply_slice_records(lst, {"s1": dict(base)}, staged=True,
                              metas={"s1": m})
        qapp.processEvents()
        h = lst.sizeHintForRow(0)
        lst.close()
        return h

    assert row_h(127) == row_h(1)               # the long line did not wrap


# -- the empty message ------------------------------------------------------------------------------
def _text_bands(img):
    bg = img.pixelColor(0, 0).lightness()
    bands, inside = 0, False
    for y in range(img.height()):
        lit = any(abs(img.pixelColor(x, y).lightness() - bg) > 25
                  for x in range(0, img.width(), 2))
        if lit and not inside:
            bands += 1
        inside = lit
    return bands


def test_the_empty_message_returns_whole_when_a_tight_box_holds_it(qapp):
    lst = L.build_slice_list(None)
    lst.set_empty_lines(("No slices in this view yet.", "Second sentence."))
    lst.resize(330, 62)
    lst.show()
    qapp.processEvents()
    img = lst.viewport().grab().toImage()
    assert _text_bands(img) >= 2                 # both sentences
    lst.resize(330, 28)                          # too short for two lines
    qapp.processEvents()
    img = lst.viewport().grab().toImage()
    assert _text_bands(img) == 1                 # the first line alone
    lst.close()


# -- the snap runs again when the room settles ------------------------------------------------
def test_snap_watcher_fires_once_after_a_burst_of_resizes(qapp):
    from qtpy.QtCore import QObject
    calls = []

    class Host(QObject):
        def cb(self):
            calls.append(1)

    host = Host()
    w = QLabel("x")
    w.show()
    watcher = C.SnapWatcher(host, host.cb)
    watcher.watch(w)
    for i in range(5):
        w.resize(100 + i, 50)
    assert calls == []                         # deferred, not synchronous
    for _ in range(4):
        qapp.processEvents()
    assert calls == [1]                        # ONE call for the burst
    w.close()


def test_the_curation_snap_follows_a_foot_that_grows_later(qapp):
    w, sa = _short_dock(qapp)
    rc = w.curation_right_column
    rc.foot.setMinimumHeight(rc.foot.height() + 40)    # e.g. a wrapped line
    for _ in range(6):
        qapp.processEvents()
    room = rc.height() - rc.foot.height()
    bottoms = sorted(
        b.mapTo(rc.content, QPoint(0, b.height())).y()
        for b in rc._blocks if not b.isHidden() and b.height() > 0)
    chrome = C.snap_cap(rc.scroll, 0) - C.SNAP_SLACK
    fits = [b for b in bottoms
            if rc.scroll.minimumHeight() <= b and b + chrome <= room]
    after = rc.scroll.maximumHeight()
    assert after == (C.snap_cap(rc.scroll, max(fits)) if fits else _QMAX), (
        after, bottoms, room)
    sa.close()


def test_a_short_dock_tightens_the_curation_panels_own_chrome(qapp):
    w, sa = _short_dock(qapp)
    root = w._curation_root_layout
    assert w._curation_sep.isHidden()
    assert root.spacing() == 2
    assert w.segment_slice_data_list.minimumHeight() == 30
    sa.resize(576, 1100)
    qapp.processEvents()
    qapp.processEvents()
    assert not w._curation_sep.isHidden()
    assert root.spacing() == 4
    assert w.segment_slice_data_list.minimumHeight() == 40
    sa.close()
