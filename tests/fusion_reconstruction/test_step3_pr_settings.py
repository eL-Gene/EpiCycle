"""Step 3's settings column.

The column is rebuilt on Step 4's ``.frm4`` parts with Step 3's OWN widgets:
names, ranges, steps, decimals and defaults kept (``batch_sb``
1-1024); plain names over grey engine names; the seven groups; the two
thresholds, "Compute masks" (OFF), the indirect method and the six
flow-consistency widgets built and HIDDEN (decisions 11-13); the channel
rows relabelled with the items unchanged (17); notes dim, no blue headers
(18); batch size at the TOP level (the owner's 2026-10-06 addition).

The UI is built WITHOUT a controller (``__new__`` + ``_build_ui``), as the
parity test does.  Run one file at a time with ``-o addopts=""``.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")


@pytest.fixture(scope="module")
def app():
    from qtpy.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def ui(app):
    from qtpy.QtWidgets import QWidget

    from napari_epicycle.fusion_reconstruction.ui import (
        FusionReconstructionCuration,
    )
    w = FusionReconstructionCuration.__new__(FusionReconstructionCuration)
    QWidget.__init__(w)
    w.viewer = None
    w._build_ui()
    yield w
    w.deleteLater()


def _hidden_in(w, root) -> bool:
    """True when *w* or one of its ancestors up to *root* is hidden."""
    p = w
    while p is not None and p is not root:
        if p.isHidden():
            return True
        p = p.parentWidget()
    return False


# ---------------------------------------------------------------- the words

def test_the_channel_rows_are_relabelled_and_the_items_kept(ui):
    from qtpy.QtWidgets import QLabel

    from napari_epicycle.ui_shell import _recon_settings_text as R
    texts = {lb.text() for lb in ui._pr_settings.findChildren(QLabel)}
    assert R.SEG_MEMBRANE_LABEL in texts
    assert R.SEG_NUCLEI_LABEL in texts
    assert R.SEG_CELL_WIDTH_LABEL in texts
    items1 = [ui.chan1_combo.itemText(i) for i in range(ui.chan1_combo.count())]
    items2 = [ui.chan2_combo.itemText(i) for i in range(ui.chan2_combo.count())]
    assert items1 == ["0: gray", "1: red", "2: green", "3: blue"]
    assert items2 == ["0: none", "1: red", "2: green", "3: blue"]
    assert ui.chan1_combo.currentIndex() == 2
    assert ui.chan2_combo.currentIndex() == 1


def test_no_old_engine_label_or_blue_header_is_left(ui):
    from qtpy.QtWidgets import QLabel
    texts = [lb.text() for lb in ui._pr_settings.findChildren(QLabel)]
    for old in ("Flow thresh:", "Cellprob:", "Diameter:", "Batch size:",
                "method:", "n_cpu:", "sigma_perp:", "ksize:",
                "Inference Parameters", "Reconstruction Parameters"):
        assert old not in texts, old
    styles = [lb.styleSheet() for lb in ui._pr_settings.findChildren(QLabel)]
    assert not any("#90caf9" in s for s in styles)
    assert not hasattr(ui, "_param_section_header")


def test_every_visible_setting_has_its_plain_name_over_its_engine(ui):
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
    labels = ui._pr_engine_labels
    assert set(labels) == set(P.S3_ADVANCED_LABELS)
    assert len(labels) == 43
    for name, lbl in labels.items():
        plain, engine = P.S3_ADVANCED_LABELS[name]
        assert lbl.text().startswith(plain.replace("&", "&amp;")), name
        assert engine.split("_")[0] in lbl.text(), name


def test_batch_size_is_at_the_top_level_with_the_owners_words(ui):
    lbl = ui._pr_engine_labels["batch_sb"]
    assert lbl.text().startswith("Batch size (slices at once)<br>")
    assert "batch_" in lbl.text()
    # inside the open Inference settings, not under Advanced
    adv = ui._pr_adv_collapsible
    p = lbl
    while p is not None and p is not adv:
        p = p.parentWidget()
    assert p is None
    p = ui.batch_sb
    while p is not None and p is not ui._pr_inf_collapsible:
        p = p.parentWidget()
    assert p is ui._pr_inf_collapsible
    assert ui._pr_inf_collapsible.isExpanded()
    assert not adv.isExpanded()


def test_the_seven_groups_carry_their_engine_names(ui):
    from napari_epicycle.ui_shell import _recon_settings_text as R
    assert list(ui._pr_adv_groups) == [g[1] for g in R.STEP4_ADVANCED_GROUPS]
    for plain, engine, _names in R.STEP4_ADVANCED_GROUPS:
        coll = ui._pr_adv_groups[engine]
        assert coll.text() == plain
        assert coll._epicycle_engine_label.text() == engine
        assert not coll.isExpanded()


def test_every_help_line_is_a_tooltip(ui):
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
    for name, line in P.S3_HELP_LINES.items():
        assert getattr(ui, name).toolTip() == line, name
    # Step 3's own lines win over the shared ones
    # R2b: the width is in µm, so the shared line is Step 3's too
    assert "micrometres" in ui.diameter_sb.toolTip()
    # R2c: the prediction manifest records the resize RULE, not the factor
    # (the factor needs the model's trained size, R6), so Step 3 has its own
    # Rescale line: it must not promise the factor (R2c audit MED-1)
    tip = ui.rescale_cb.toolTip()
    assert "rule used is recorded with the result" in tip
    assert "factor" not in tip
    assert ui.batch_sb.toolTip().startswith("How many slices")


def test_the_cell_width_is_in_um_since_r2b(ui):
    """R2b changed this test deliberately: the suffix was
    " px" until R2b; it is Step 4's shared " µm" now."""
    assert ui.diameter_sb.suffix() == " \u00b5m"
    assert ui.diameter_sb.specialValueText() == "Auto"
    assert ui.diameter_sb.value() == 0


# ---------------------------------------------------------------- the widgets

def test_the_inference_widgets_keep_their_names_ranges_and_defaults(ui):
    assert (ui.batch_sb.minimum(), ui.batch_sb.maximum(),
            ui.batch_sb.value()) == (1, 1024, 128)
    assert (ui.diameter_sb.minimum(), ui.diameter_sb.maximum()) == (0, 500)
    assert (ui.flow_thresh.minimum(), ui.flow_thresh.maximum(),
            ui.flow_thresh.singleStep(), ui.flow_thresh.value()) == (
        0, 3.0, 0.1, pytest.approx(0.4))
    assert (ui.cellprob_thresh.minimum(), ui.cellprob_thresh.maximum(),
            ui.cellprob_thresh.singleStep(),
            ui.cellprob_thresh.value()) == (-10, 10, 0.5, 0.0)
    assert ui.norm_cb.isChecked() and ui.rescale_cb.isChecked()
    assert (ui.n_cpu_sb.minimum(), ui.n_cpu_sb.maximum(),
            ui.n_cpu_sb.value()) == (1, 64, 6)
    assert ui.skip_postprocess_cb.isChecked() and ui.do_mp_cb.isChecked()


def test_the_hidden_knobs_are_built_hidden_and_masks_off(ui):
    for name in ("flow_thresh", "cellprob_thresh", "compute_masks_cb"):
        assert _hidden_in(getattr(ui, name), ui._pr_settings), name
    assert not ui.compute_masks_cb.isChecked()


def test_the_method_list_is_direct_only_and_the_indirect_group_inert(ui):
    assert ui.method_combo.count() == 1
    assert ui.method_combo.itemText(0) == "Direct"
    assert ui.method_combo.itemData(0) == "direct"
    assert ui.method_combo.currentData() == "direct"
    assert ui._indirect_method_container.isHidden()
    for name in ("do_flow_remove_cb", "flow_threshold_sb", "fc_dtform_combo",
                 "fc_edt_percentile_sb", "fc_power_dist_sb",
                 "fc_smooth_skel_sb"):
        assert _hidden_in(getattr(ui, name), ui._pr_settings), name
    # nothing listens to the method combo, and the old show/hide handler
    # for the indirect group is gone (audit A LOW-6: it was dead code)
    assert ui.method_combo.receivers(
        ui.method_combo.currentTextChanged) == 0
    assert not hasattr(type(ui), "_on_method_changed")


def test_the_hidden_recon_widgets_hold_reconstruction_config(ui):
    from packages.epicycle_reconstruction.config import ReconstructionConfig
    cfg = ReconstructionConfig()
    fc, im = cfg.flow_consistency, cfg.indirect_method
    assert ui.do_flow_remove_cb.isChecked() == fc["do_flow_remove"]
    assert ui.flow_threshold_sb.value() == pytest.approx(fc["flow_threshold"])
    assert ui.fc_dtform_combo.currentText() == fc["dtform_method"]
    assert ui.fc_edt_percentile_sb.value() == pytest.approx(
        fc["edt_fixed_point_percentile"])
    assert ui.fc_power_dist_sb.value() == -1.0     # None
    assert ui.fc_smooth_skel_sb.value() == pytest.approx(
        fc["smooth_skel_sigma"])
    assert ui.im_dtform_combo.currentText() == im["dtform_method"]
    assert ui.im_smooth_binary_sb.value() == pytest.approx(im["smooth_binary"])
    assert ui.im_smooth_skel_sb.value() == pytest.approx(
        im["smooth_skel_sigma"])
    assert ui.im_edt_percentile_sb.value() == pytest.approx(
        im["edt_fixed_point_percentile"])
    assert ui.im_power_dist_sb.value() == -1.0


def test_the_old_preset_buttons_are_built_and_hidden(ui):
    assert _hidden_in(ui.epicycle_defaults_btn, ui._pr_settings)
    assert _hidden_in(ui.useg3d_defaults_btn, ui._pr_settings)


def test_start_from_offers_the_two_standard_sets(ui):
    from napari_epicycle.ui_shell import _recon_settings_text as R
    c = ui.start_from_combo
    assert [(c.itemText(i), c.itemData(i)) for i in range(c.count())] == [
        (R.SEG_PRESET_EPICYCLE, "epicycle"),
        (R.SEG_PRESET_USEG3D, "useg3d")]
    assert R.SEG_PRESET_USEG3D == "u-Segment3D defaults (without its flow filter)"


def test_notes_are_dim_and_follow_their_ticks(ui):
    from napari_epicycle.ui_shell.theme import DIM_LABEL_STYLE
    assert ui.seg_size_off_note.styleSheet() == DIM_LABEL_STYLE
    assert ui.ps_flows_off_note.styleSheet() == DIM_LABEL_STYLE
    assert ui.pr_last_run_note.styleSheet() == DIM_LABEL_STYLE
    assert ui.pr_last_run_note.isHidden()
    assert not ui.seg_size_off_note.isHidden()       # skip is ticked
    assert not ui.ps_flows_off_note.isHidden()       # smooth flows off
    assert not ui.ps_mask_thresh_sb.isEnabled()
    ui.ps_smooth_flows_cb.setChecked(True)
    assert ui.ps_flows_off_note.isHidden() and ui.ps_mask_thresh_sb.isEnabled()
    ui.ps_smooth_flows_cb.setChecked(False)
    ui.skip_postprocess_cb.setChecked(False)
    assert ui.seg_size_off_note.isHidden()
    ui.skip_postprocess_cb.setChecked(True)


def test_the_settings_desc_follows_the_last_run(ui):
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
    ui.set_pr_settings_desc(True)
    assert ui.pr_settings_desc.text() == P.S3_SETTINGS_DESC
    ui.set_pr_settings_desc(False)
    assert ui.pr_settings_desc.text() == P.S3_SETTINGS_DESC_NO_RUN
    ui.show_pr_last_run_note(True)
    assert not ui.pr_last_run_note.isHidden()
    ui.show_pr_last_run_note(False)


def test_the_ticks_are_gold(ui):
    from napari_epicycle.ui_shell.theme import PANEL_CHECKBOX_STYLE
    assert ui._pr_settings.styleSheet() == PANEL_CHECKBOX_STYLE


def test_every_settings_widget_has_a_parent(ui):
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
    names = set(P.S3_HELP_LINES) | {
        "start_from_combo", "pr_settings_desc", "pr_last_run_note",
        "flow_thresh", "cellprob_thresh", "compute_masks_cb",
        "epicycle_defaults_btn", "useg3d_defaults_btn",
        "seg_size_off_note", "seg_size_desc", "ps_flows_off_note",
        "do_flow_remove_cb", "flow_threshold_sb", "fc_dtform_combo",
        "fc_edt_percentile_sb", "fc_power_dist_sb", "fc_smooth_skel_sb",
        "im_dtform_combo", "im_smooth_binary_sb", "im_smooth_skel_sb",
        "im_edt_percentile_sb", "im_power_dist_sb"}
    for name in sorted(names):
        w = getattr(ui, name)
        p = w
        while p is not None and p is not ui._pr_settings:
            p = p.parentWidget()
        assert p is ui._pr_settings, name


def test_the_settings_header_has_its_guide(ui):
    from qtpy.QtWidgets import QPushButton

    from napari_epicycle.fusion_reconstruction import ui as U
    assert U._PR_SETTINGS_GUIDE == "step3_predict_reconstruct_settings.md"
    assert (U._GUIDES_DIR / U._PR_SETTINGS_GUIDE).exists()
    helps = [b for b in ui._pr_settings.findChildren(QPushButton)
             if b.toolTip() == "Help: Settings"]
    assert len(helps) == 1
