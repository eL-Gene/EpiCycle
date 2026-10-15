"""The rebuilt Model Refinement section (Brick B1, I6).

* (a) every widget of the map exists under its name on a built widget
  (the controller binds them by name), each with a parent;
* (b) I6: at the 578 px dock (the section's viewport is 566 px) the section
  has no horizontal scrollbar and nothing it shows reaches past the
  viewport, in every state -- empty list, three rows, Mode = Both with the
  axial rows, Advanced and Extraction Settings open, the lock line shown,
  the result card and the status filled;
* (c) the ui-side tick API round-trips;
* (d) ``refresh_refinement_patch_list`` with 0 / 3 rows paints the chips and
  the markers (and the empty list its two lines).

The measurements run in a SUBPROCESS with the real font and
``QT_QPA_PLATFORM=offscreen`` (offscreen Qt draws every glyph
~1.6x too wide without a font directory, and changing the app font leaks
between tests), the pattern of ``test_step3_pr_layout_ceilings.py``.  Run one
file at a time with ``-o addopts=""``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_FONT_DIR = "C:/Windows/Fonts"
_REPO = Path(__file__).resolve().parents[2]
VIEWPORT = 566

#: Every existing name plus the ones B1 adds
WIDGET_MAP = (
    "refine_patch_list", "refine_select_all_btn", "refine_deselect_all_btn",
    "pairs_summary_label", "extract_training_pairs_btn",
    "refine_extraction_progress", "refine_extraction_status_label",
    "clear_extracted_pairs_btn", "refine_per_orientation_summary_label",
    "refine_configure_per_orientation_btn", "refine_train_frac_sb",
    "refine_chunk_size_sb", "refine_spatial_split_cb",
    "refine_split_summary_label",
    "refinement_mode_combo", "refinement_mode_hint", "refine_init_combo",
    "refine_source_row", "refine_source_row_axial", "refine_source_label",
    "refine_source_label_axial", "refinement_source_combo",
    "refinement_source_combo_axial", "refine_pretrained_row",
    "refine_pretrained_row_axial", "refine_pretrained_label",
    "refine_pretrained_label_axial", "refine_pretrained_combo",
    "refine_pretrained_combo_axial", "refine_import_row",
    "refine_import_row_axial", "refine_import_label",
    "refine_import_label_axial", "refine_import_path",
    "refine_import_path_axial", "refine_import_browse_btn",
    "refine_import_browse_btn_axial", "refine_model_name",
    "refine_preset_combo", "refine_apply_preset_btn",
    "refine_preset_description", "refine_epochs_sb", "refine_lr_sb",
    "refine_batch_sb",
    "refine_wd_sb", "refine_momentum_sb", "refine_min_masks_sb",
    "refine_chan1_combo", "refine_chan2_combo", "refine_seed_sb",
    "refine_normalize_cb", "refine_rescale_cb", "refine_sgd_cb",
    "refine_compute_flows_cb", "refine_device_gpu", "refine_device_cpu",
    "refine_save_config_btn", "refine_load_config_btn",
    "refinement_output_label", "refine_use_for_predict_btn",
    "refine_view_models_btn", "refine_pack_status_label",
    "launch_refinement_btn", "refinement_cancel_btn", "refinement_progress",
    "refinement_status_label", "refinement_loss_plot",
    # the ones B1 adds (the card, the lines)
    "refine_card_title", "refine_card_trained_on", "refine_card_started_from",
    "refine_card_lateral_loss", "refine_card_axial_loss", "refine_card_took",
    "refine_card_new_run_hint", "refine_run_line", "refine_lock_line",
    "refine_ticked_count_label", "refine_few_pairs_label",
)
STATES = ("empty", "rows3", "both", "advanced", "locked", "card")
#: fix pass H1: the states at the REAL dock height (the section's viewport is
#: ~500 px tall in an 848 px window) in which the section must not scroll
V_STATES = ("empty", "rows3", "training", "extraction", "advanced")
VIEWPORT_H = 500
LONG_NAME = "WholeScale_membrane_patch_0042_long_name_for_the_list.tiff"
LONG_MODEL = ("wound-v2-membrane-nuclei-refined-2026-10-04-from-step2 · "
              "round 12 (lateral)")


def measure_all(app) -> dict:
    """Runs in the subprocess."""
    from qtpy.QtCore import QPoint, QRect, Qt
    from qtpy.QtWidgets import QApplication, QWidget

    from napari_epicycle.fusion_reconstruction import _step3_mr_text as M
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
    from napari_epicycle.fusion_reconstruction.ui import (
        FusionReconstructionCuration,
    )

    def pump(n=6):
        for _ in range(n):
            app.processEvents()

    w = FusionReconstructionCuration.__new__(FusionReconstructionCuration)
    QWidget.__init__(w)
    w.viewer = None
    w._build_ui()
    w._apply_theme()
    w._accordion.setCurrentIndex(3)
    w.resize(700, 2400)
    w.show()
    pump(10)
    sec = w._accordion._sections[3]
    content = sec._scroll.widget()
    out = {"font": app.font().family()}

    # (a) the widget map
    out["missing"] = [n for n in WIDGET_MAP
                      if not hasattr(w, n)]
    out["parentless"] = [n for n in WIDGET_MAP
                         if hasattr(w, n)
                         and isinstance(getattr(w, n), QWidget)
                         and getattr(w, n).parentWidget() is None]

    for _ in range(6):               # settle the dock so the viewport is 566
        delta = VIEWPORT - sec._scroll.viewport().width()
        if delta == 0:
            break
        w.resize(w.width() + delta, 2400)
        pump(5)
    out["viewport"] = sec._scroll.viewport().width()

    names = {f"{LONG_NAME[:-5]}_{i}.tiff": "finalized" for i in range(3)}
    keys = sorted(names)
    info = {Path(keys[0]).stem: {"spatial": 3, "axial": 900},
            Path(keys[1]).stem: {"spatial": 400, "axial": 380}}

    def reset():
        w.set_refine_lock_line("")
        w.set_refine_ticks(False)
        w.refine_advanced.collapse(animate=False)
        w.refine_extraction_settings.collapse(animate=False)
        w.refine_loss_collapsible.collapse(animate=False)
        w.refinement_mode_combo.setCurrentIndex(0)
        for n in ("refine_source_row_axial", "refine_pretrained_row",
                  "refine_pretrained_row_axial", "refine_import_row",
                  "refine_import_row_axial"):
            getattr(w, n).setVisible(False)
        w.refresh_refinement_patch_list({})
        w.set_refine_will_train(0, 0, 0)
        w.show_refine_status("", "neutral")
        w.set_refine_card("")
        w.refine_extraction_status_label.setText("")

    def stage(state: str) -> None:
        reset()
        if state == "empty":
            pass
        elif state == "rows3":
            w.refresh_refinement_patch_list(
                names, info, stale_stems={Path(keys[0]).stem},
                mixed_settings_stems={Path(keys[0]).stem,
                                      Path(keys[1]).stem})
            w.set_refine_ticks(True)
            w.set_refine_will_train(3, 1280, 3)
            w.refine_extraction_status_label.setText(
                M.MR_EXTRACT_DONE.format(time="[1:23:45 | 2:10:00]",
                                         n="1,234", m="1,280", t="2,514",
                                         k=3))
            w.refine_per_orientation_summary_label.setText(
                "Lateral: " + "x" * 8 + " defaults  |  Axial: custom, "
                "min size 30, diameter 25")
        elif state == "both":
            w.refinement_mode_combo.setCurrentIndex(2)
            w.refine_init_combo.setCurrentIndex(3)
            for n in ("refine_source_row_axial", "refine_import_row",
                      "refine_import_row_axial", "refine_pretrained_row",
                      "refine_pretrained_row_axial"):
                getattr(w, n).setVisible(True)
            for c in (w.refinement_source_combo,
                      w.refinement_source_combo_axial):
                c.addItem(LONG_MODEL)
                c.setCurrentIndex(c.count() - 1)
            w.refine_import_path.setText("C:/" + "very_long_folder/" * 8
                                         + "model.pth")
            w.refine_import_path_axial.setText(w.refine_import_path.text())
            w.refine_model_name.setText("a_really_long_model_name_" * 3)
            w.refinement_mode_hint.setText(M.mode_hint(
                "Fine-tune Both", LONG_MODEL, LONG_MODEL, "refined_03"))
        elif state == "advanced":
            w.refine_advanced.expand(animate=False)
            w.refine_extraction_settings.expand(animate=False)
            w.refine_preset_combo.setCurrentIndex(5)
            w.refine_preset_description.setText(
                M.PRESET_DESCRIPTIONS["stratified_epithelia"])
        elif state == "training":
            w.refresh_refinement_patch_list(
                names, info, stale_stems={Path(keys[0]).stem},
                mixed_settings_stems={Path(keys[1]).stem})
            w.set_refine_ticks(True)
            w.set_refine_will_train(3, 1280, 3)
            w.set_refine_lock_line(M.MR_LOCK_TRAINING.format(
                bucket="lateral", i=1, n=2))
            w.refinement_mode_combo.setCurrentIndex(2)
            w.show_refine_status(
                "Training lateral… (loss 0.4123, val 0.4412) "
                "[12m 03s | ~19m 40s remaining] · axial next", "info")
            w.refinement_progress.setRange(0, 100)
            w.refinement_progress.setValue(20)
            w.refine_loss_collapsible.expand(animate=False)
        elif state == "extraction":
            w.refresh_refinement_patch_list(
                names, info, stale_stems={Path(keys[0]).stem})
            w.set_refine_ticks(True)
            w.set_refine_will_train(3, 1280, 3)
            w.refine_extraction_settings.expand(animate=False)
        elif state == "locked":
            text = max((t for _f, t in P.S3_LOCK_SENTENCES), key=len)
            w.set_refine_lock_line(text)
        elif state == "card":
            w.set_refine_card(
                "refined_03", trained_on="543 pairs (80 lateral, 463 axial)"
                " from 2 patches", started_from=LONG_MODEL,
                lateral_loss="0.41 · validation 0.44",
                axial_loss="0.38 · validation 0.40",
                took="24 min · today 15:48",
                new_run_hint=M.MR_NEW_RUN_HINT.format(name=LONG_MODEL))
            w.refinement_output_label.setText(
                "Output: " + "C:/" + "very_long_folder/" * 8 + "refined_03")
            w.show_refine_status(M.MR_STATUS_FAILED.format(
                bucket="lateral", e=412, first="CUDA out of memory " * 6),
                "error")
            w.refine_loss_collapsible.expand(animate=False)
        pump(10)

    def widest_child() -> int:
        right = 0
        for ch in content.findChildren(QWidget):
            if not ch.isVisible():
                continue
            x = ch.mapTo(content, QPoint(ch.width(), 0)).x()
            right = max(right, x)
        return right

    for state in STATES:
        stage(state)
        sb = sec._scroll.horizontalScrollBar()
        out[state] = {
            "hscroll_visible": bool(sb.isVisible()),
            "hscroll_max": int(sb.maximum()),
            "section_min_w": content.minimumSizeHint().width(),
            "right_min_w": w.refine_right_content.minimumSizeHint().width(),
            "right_scroll_w": w.refine_right_scroll.viewport().width(),
            "left_min_w": w.refine_left.minimumSizeHint().width(),
            "left_w": w.refine_left.width(),
            "widest_child": widest_child(),
            "content_w": content.width(),
        }
    reset()

    # (e) fix pass H1: at the real dock height the section does not scroll
    for _ in range(8):
        delta = VIEWPORT_H - sec._scroll.viewport().height()
        if delta == 0:
            break
        w.resize(w.width(), w.height() + delta)
        pump(5)
    out["viewport_h"] = sec._scroll.viewport().height()
    for state in V_STATES:
        stage(state)
        pump(10)
        vsb = sec._scroll.verticalScrollBar()
        out["v_" + state] = {
            "vscroll_visible": bool(vsb.isVisible()),
            "vscroll_max": int(vsb.maximum()),
            "content_min_h": content.minimumSizeHint().height(),
            "plot_h": w.refinement_loss_plot.height(),
            "status_lines": w.refinement_status_label.text().count(chr(10)) + 1,
        }
    reset()
    w.resize(w.width(), 2400)
    pump(10)

    # (c) the tick API
    w.refresh_refinement_patch_list(names, info)
    tick = {"rows": w.refine_row_count(), "start": w.refine_ticked_names()}
    w.set_refine_ticks(True)
    tick["all"] = w.refine_ticked_names()
    tick["label_all"] = w.refine_ticked_count_label.text()
    w.refine_patch_list.item(1).setCheckState(Qt.Unchecked)
    tick["one_off"] = w.refine_ticked_names()
    tick["label_two"] = w.refine_ticked_count_label.text()
    w.refresh_refinement_patch_list(names, info)       # the ticks survive
    tick["after_refresh"] = w.refine_ticked_names()
    w.refresh_refinement_patch_list({keys[0]: "finalized"}, info)
    tick["after_shrink"] = w.refine_ticked_names()
    w.set_refine_ticks(False)
    tick["none"] = w.refine_ticked_names()
    w.refresh_refinement_patch_list({})
    tick["empty_rows"] = w.refine_row_count()
    tick["empty_label"] = w.refine_ticked_count_label.text()
    out["tick"] = tick

    # (d) the chips and the markers, painted
    lst = w.refine_patch_list
    w.refresh_refinement_patch_list({})
    out["empty_lines"] = list(lst._empty_lines)
    w.refresh_refinement_patch_list(
        names, info, stale_stems={Path(keys[0]).stem},
        mixed_settings_stems={Path(keys[1]).stem})
    pump(8)
    d = lst.itemDelegate()
    width = max(lst.viewport().width(), 60)
    rows = []
    for i in range(lst.count()):
        it = lst.item(i)
        rec = dict(it.data(Qt.UserRole + 2) or {})
        g = d.layout(QRect(0, 0, width, 0), rec, lst.font())
        full = g["height"]
        n_chips, n_warns = len(g["chips"]), len(g["warns"])
        rows.append({"chips": [list(c) for c in rec.get("chips", ())],
                     "warns": list(rec.get("warns_narrow", ())),
                     "tickable": rec.get("tickable"),
                     "tip": it.toolTip(), "height": full,
                     "n_chips": n_chips, "n_warns": n_warns,
                     "flags_checkable": bool(
                         it.flags() & Qt.ItemIsUserCheckable)})
    out["rows"] = rows
    out["thumb_lists"] = lst in w._thumb_lists()
    return out


_SUBPROCESS = """
import json, os, sys
os.environ["QT_QPA_PLATFORM"] = "offscreen"
if os.path.isdir(FONT_DIR):
    os.environ["QT_QPA_FONTDIR"] = FONT_DIR
from qtpy.QtGui import QFont
from qtpy.QtWidgets import QApplication
app = QApplication([])
app.setFont(QFont("Segoe UI", 9))
import napari_epicycle  # the import order the app uses
from tests.fusion_reconstruction import test_step3_mr_panel as t
print("S3MRJSON" + json.dumps(t.measure_all(app)), flush=True)
os._exit(0)     # the numbers are out; skip a Qt teardown that can crash
"""


@pytest.fixture(scope="module")
def measured():
    code = f"FONT_DIR = {_FONT_DIR!r}\n" + _SUBPROCESS
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(_REPO),
                          env=env, capture_output=True, text=True,
                          timeout=300)
    lines = [ln for ln in proc.stdout.splitlines()
             if ln.startswith("S3MRJSON")]
    assert lines, (proc.returncode, proc.stdout[-1500:],
                   proc.stderr[-3000:])
    got = json.loads(lines[-1][len("S3MRJSON"):])
    print(f"[MR] {json.dumps(got)}")
    return got


def test_measured_in_the_users_font(measured):
    if sys.platform == "win32" and os.path.isdir(_FONT_DIR):
        assert measured["font"] == "Segoe UI"


def test_the_viewport_is_the_one_asked_for(measured):
    assert measured["viewport"] == VIEWPORT


# ── (a) the widget map ─────────────────────────────────────────────────────
def test_every_widget_of_the_map_exists(measured):
    assert measured["missing"] == []


def test_every_widget_of_the_map_has_a_parent(measured):
    assert measured["parentless"] == []


# ── (b) I6 ─────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("state", STATES)
def test_no_horizontal_scrollbar(measured, state):
    got = measured[state]
    assert not got["hscroll_visible"], got
    assert got["hscroll_max"] == 0, got


@pytest.mark.parametrize("state", STATES)
def test_every_minimum_width_fits(measured, state):
    got = measured[state]
    assert got["section_min_w"] <= VIEWPORT, got
    assert got["right_min_w"] <= got["right_scroll_w"], got
    assert got["left_min_w"] <= 240, got


@pytest.mark.parametrize("state", STATES)
def test_the_widest_child_fits_the_viewport(measured, state):
    got = measured[state]
    assert got["widest_child"] <= VIEWPORT, got


# ── (e) fix pass H1: no vertical scroll of the section at the dock ────────
def test_the_viewport_height_is_the_dock(measured):
    assert measured["viewport_h"] == VIEWPORT_H


@pytest.mark.parametrize("state", V_STATES)
def test_the_section_does_not_scroll_as_a_whole(measured, state):
    got = measured["v_" + state]
    assert not got["vscroll_visible"], got
    assert got["vscroll_max"] == 0, got
    assert got["content_min_h"] <= VIEWPORT_H, got


def test_the_foot_status_is_at_most_two_lines(measured):
    assert measured["v_training"]["status_lines"] <= 2


def test_an_open_loss_plot_is_at_least_its_floor(measured):
    from napari_epicycle.fusion_reconstruction import ui as U
    assert measured["v_training"]["plot_h"] >= U.MR_LOSS_MIN_H


# ── (c) the tick API ───────────────────────────────────────────────────────
def test_the_tick_api_round_trips(measured):
    t = measured["tick"]
    assert t["rows"] == 3
    assert t["start"] == []
    assert len(t["all"]) == 3 and t["all"] == sorted(t["all"])
    assert t["label_all"] == "3 of 3 ticked"
    assert len(t["one_off"]) == 2 and t["all"][1] not in t["one_off"]
    assert t["label_two"] == "2 of 3 ticked"
    assert t["after_refresh"] == t["one_off"]      # the ticks survive
    assert t["after_shrink"] == [t["all"][0]]      # a gone row's tick goes
    assert t["none"] == []
    assert t["empty_rows"] == 0
    assert t["empty_label"] == "0 of 0 ticked"


# ── (d) the rows ───────────────────────────────────────────────────────────
def test_the_empty_list_paints_its_two_lines(measured):
    from napari_epicycle.fusion_reconstruction import _step3_mr_text as M
    assert measured["empty_lines"] == list(M.MR_EMPTY_LINES)


def test_three_rows_paint_chips_and_markers(measured):
    from napari_epicycle.fusion_reconstruction import _step3_mr_text as M
    rows = measured["rows"]
    assert len(rows) == 3
    # row 0: 3 lateral + 900 axial, stale; row 1: 400 + 380, mixed; row 2:
    # nothing extracted
    texts0 = [c[0] for c in rows[0]["chips"]]
    assert texts0 == [M.MR_ROW_PAIRS_LATERAL.format(n="3"),
                      M.MR_ROW_PAIRS_AXIAL.format(m="900")]
    assert [c[1] for c in rows[0]["chips"]] == [True, False]   # amber lateral
    assert rows[0]["warns"] == [M.MR_ROW_STALE]
    assert M.MR_ROW_STALE_TIP in rows[0]["tip"]
    assert [c[0] for c in rows[1]["chips"]] == [
        M.MR_ROW_PAIRS_LATERAL.format(n="400"),
        M.MR_ROW_PAIRS_AXIAL.format(m="380")]
    assert rows[1]["warns"] == [M.MR_ROW_MIXED]
    assert M.MR_ROW_MIXED_TIP in rows[1]["tip"]
    assert [c[0] for c in rows[2]["chips"]] == [M.MR_ROW_NOT_EXTRACTED]
    assert rows[2]["warns"] == []
    assert [r["n_chips"] for r in rows] == [2, 2, 1]
    assert [r["n_warns"] for r in rows] == [1, 1, 0]
    for r in rows:
        assert r["tickable"] is True          # finalized rows tick
        assert r["flags_checkable"] is True
        # the chips and the markers are PAINTED: the delegate lays out one
        # rectangle for each (the 80 px picture sets the row's height)
        assert r["height"] >= 80


def test_the_refine_list_gets_the_step3_pictures(measured):
    assert measured["thumb_lists"] is True


# ── the row model (Qt-free) ────────────────────────────────────────────────
def test_build_refine_row_is_a_tickable_finalized_row():
    from napari_epicycle.fusion_reconstruction import _step3_rows as R
    rec = R.build_refine_row("p.tif", "finalized", n_lateral=50,
                             n_axial=10, ticked=True)
    assert rec["tickable"] is True and rec["ticked"] is True
    assert "notes_narrow" in rec and rec["notes_narrow"] == ()   # no lock note
    assert [c[1] for c in rec["chips"]] == [False, False]
    # plain build_row marks the same row protected and untickable
    assert R.build_row("p.tif", "finalized")["tickable"] is False


def test_the_few_pairs_threshold_is_twenty():
    from napari_epicycle.fusion_reconstruction import _step3_rows as R
    def lateral_warn(n):
        return R.build_refine_row("p", "finalized", n_lateral=n,
                                  n_axial=5)["chips"][0][1]
    assert lateral_warn(20) is False
    assert lateral_warn(19) is True


def test_will_train_text_forms():
    from napari_epicycle.fusion_reconstruction import _step3_mr_text as M
    assert M.will_train_text(0, 0, 0) == M.MR_WILL_TRAIN_NOTHING
    assert M.will_train_text(10, 20, 1).endswith("from 1 patch.")
    assert M.will_train_text(10, 20, 3).endswith("from 3 patches.")
