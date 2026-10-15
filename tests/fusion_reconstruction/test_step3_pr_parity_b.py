"""Brick B (Predict + Reconstruct parity).

Run one file at a time with ``-o addopts=""``.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")

from tests.fusion_reconstruction.test_step3_pr_lock import (  # noqa: E402,F401
    app, full, _set, _P,
)


# ---------------------------------------------------- 5: the second family

def test_qc_extraction_shows_the_line_and_keeps_predict_live(full):
    w, c = full
    # enabled WITHOUT the flag first (a lock test passes
    # vacuously when another guard already greys the button)
    assert w.predict_selected_btn.isEnabled()
    assert w.reconstruct_selected_btn.isEnabled()
    assert w.pr_lock_line.text() == ""
    _set(c, "_qc_extract_running")
    try:
        assert w.pr_lock_line.text() == _P().S3_LOCK_QC_EXTRACT
        assert w.predict_selected_btn.isEnabled()
        assert w.reconstruct_selected_btn.isEnabled()
        assert w.view_reconstructed_button.isEnabled() is False
    finally:
        _set(c, "_qc_extract_running", False)
    assert w.pr_lock_line.text() == ""


# ----------------------------------------------- 7: the kinds
import shutil  # noqa: E402

from napari_epicycle.fusion_reconstruction import _model_provenance as MP  # noqa: E402
from napari_epicycle.fusion_reconstruction import _step3_pr_text as PT  # noqa: E402
from tests.fusion_reconstruction.test_model_provenance import (  # noqa: E402,F401
    refined_bundle, step2_model, ws,
)


def test_a_fork_of_a_bootstrap_run_stays_domain_specialized(ws):
    """Call 1: a Step 2 run started from ANOTHER run's model (a plain
    bootstrap run) reads domain_specialized, not bootstrap."""
    parent = step2_model(ws, run="cells")
    assert MP.model_kind(parent, workspace=str(ws)) == "bootstrap"
    fork = step2_model(ws, run="fork", init_mode="inherited",
                       inherited_from_run="cells",
                       inherited_from_run_bucket="lateral",
                       inherited_from_run_round_id="round_02")
    assert MP.model_kind(fork, workspace=str(ws)) == "domain_specialized"


def _copy_tree(src: Path, dst_ws: Path, ws: Path) -> Path:
    rel = src.relative_to(ws)
    dst = dst_ws / rel
    shutil.copytree(ws / rel.parts[0], dst_ws / rel.parts[0],
                    dirs_exist_ok=True)
    return dst


def test_a_refined_bundle_browsed_from_another_workspace_keeps_its_kind(
        ws, tmp_path):
    """Call 2: not 'external' -- by its manifest AND by its path."""
    w = refined_bundle(ws, name="mine")
    other = tmp_path / "other_ws"
    other.mkdir()
    moved = _copy_tree(w, other, ws)
    assert MP.model_kind(moved, workspace=str(ws)) == "refined"
    assert MP.model_kind(moved, workspace=str(other)) == "refined"
    (moved.parent / "manifest.json").unlink()      # the path alone
    assert MP.model_kind(moved, workspace=str(ws)) == "refined"


def test_a_step2_model_browsed_from_another_workspace_keeps_its_kind(
        ws, tmp_path):
    w = step2_model(ws)
    other = tmp_path / "other_ws"
    other.mkdir()
    moved = _copy_tree(w, other, ws)
    assert MP.model_kind(moved, workspace=str(ws)) == "bootstrap"
    (moved.parent / "manifest.json").unlink()
    assert MP.model_kind(moved, workspace=str(ws)) == "bootstrap"


def test_a_legacy_round_folder_browsed_from_another_workspace_is_bootstrap(
        ws, tmp_path):
    legacy = ws / "models" / "round_03" / "Cell_lateral_round_03"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"w")
    assert MP.model_kind(legacy, workspace=str(ws)) == "bootstrap"
    other = tmp_path / "other_ws"
    other.mkdir()
    assert MP.model_kind(legacy, workspace=str(other)) == "bootstrap"


def test_a_stray_file_in_a_models_folder_is_still_external(ws):
    stray = ws / "models" / "mine" / "weights"
    stray.parent.mkdir(parents=True)
    stray.write_bytes(b"w")
    assert MP.model_kind(stray, workspace=str(ws)) == "external"


def test_the_details_window_stays_open_after_save(app):
    from tests.fusion_reconstruction.test_model_provenance_dialog import (
        make, Saver)
    host, dlg, saver = make(app, saver=Saver(None))
    try:
        host.show()
        dlg.show()
        app.processEvents()
        dlg.save_btn.click()
        app.processEvents()
        assert saver.calls and dlg.isVisible()
        assert dlg.status_label.text() == PT.PROV_SAVED
    finally:
        host.deleteLater()


# ------------------------------------------ 7: the chip elides in the middle
def _narrow_delegate():
    from napari_epicycle.ui_shell import _stack_row_delegate as D
    return D, D.StackRowDelegate(D.MODE_NARROW, show_info=True)


def _chip_rec(name="wound-v2-membrane-nuclei-refined · lateral · round_12",
              active=True):
    from napari_epicycle.fusion_reconstruction import _step3_rows as R
    return R.build_row("p.tif", "predicted", "CZYX",
                       made_by=PT.MADE_BY.format(short=name),
                       made_by_active=active)


def _lay(d, rec, width=240):
    from qtpy.QtCore import QRect
    from qtpy.QtGui import QFont
    return d.layout(QRect(0, 0, width, 0), rec, QFont())


def test_a_narrow_chip_is_one_line_elided_at_the_tail(app):
    D, d = _narrow_delegate()
    rec = _chip_rec()
    g = _lay(d, rec)
    (r, shown, warn), = g["chips_top"]
    full = rec["chips_top"][0][0]
    assert shown != full and shown.endswith("…")
    assert shown.startswith("made by wound")     # the head stays readable
    assert r.height() <= 22                      # one line, not two
    assert d.tip_text is not None


def test_the_not_active_tail_is_kept_when_the_whole_chip_fits(app):
    D, d = _narrow_delegate()
    rec = _chip_rec("cyto3", active=False)
    g = _lay(d, rec, width=420)
    (r, shown, warn), = g["chips_top"]
    assert warn and shown == rec["chips_top"][0][0]
    assert shown.endswith(PT.S3_MADE_BY_NOT_ACTIVE_TAIL)


def test_the_tail_gives_way_to_the_name_in_a_narrow_column(app):
    """At the 240 px column the tail alone is wider than the room: the name
    is what is shown, the amber border and the tooltip carry the rest."""
    D, d = _narrow_delegate()
    rec = _chip_rec(active=False)
    g = _lay(d, rec, width=240)
    (r, shown, warn), = g["chips_top"]
    assert warn and not shown.endswith(PT.S3_MADE_BY_NOT_ACTIVE_TAIL)
    assert shown.startswith("made by wound") and shown.endswith("…")
    assert r.height() <= 22


def test_a_chip_that_fits_is_not_touched(app):
    D, d = _narrow_delegate()
    rec = _chip_rec("cyto3")
    g = _lay(d, rec, width=420)
    (r, shown, warn), = g["chips_top"]
    assert shown == rec["chips_top"][0][0]


def test_the_full_chip_text_is_the_tooltip(app):
    from qtpy.QtCore import QModelIndex
    from qtpy.QtWidgets import QListWidget, QListWidgetItem, QStyleOptionViewItem
    from napari_epicycle.ui_shell._stack_row_delegate import ROLE_RECORD
    D, d = _narrow_delegate()
    d.set_help_lines({"row_made_by": "Which model made it."}) \
        if hasattr(d, "set_help_lines") else setattr(
            d, "_help_lines", {"row_made_by": "Which model made it."})
    lst = QListWidget()
    lst.setItemDelegate(d)
    lst.resize(240, 300)
    rec = _chip_rec(active=False)
    it = QListWidgetItem("x")
    it.setData(ROLE_RECORD, rec)
    lst.addItem(it)
    lst.show()
    opt = QStyleOptionViewItem()
    opt.initFrom(lst)
    opt.widget = lst
    opt.rect = lst.visualItemRect(it)
    opt.font = lst.font()
    g = d.layout(opt.rect, rec, opt.font)
    r = g["chips_top"][0][0]
    tip = d.tip_text(opt, lst.indexFromItem(it), r.center())
    assert rec["chips_top"][0][0] in tip
    assert "Which model made it." in tip
    lst.deleteLater()


# -------------------------------------------- 6: the status line (the foot)
@pytest.fixture
def lay(app):
    from napari_epicycle.fusion_reconstruction.ui import (
        FusionReconstructionCuration,
    )
    w = FusionReconstructionCuration(viewer=None)
    c = w.controller
    statuses = {}
    c.state_manager.get_all_patch_statuses = lambda: dict(statuses)
    # a foot wide enough for the whole line (offscreen glyphs are wide)
    w.pipeline_status_label.setFixedWidth(1400)
    yield w, c, statuses
    w.deleteLater()


def _list(w, statuses, rows, ticked=()):
    statuses.clear()
    statuses.update(rows)
    w.refresh_patch_state_list(dict(rows), ticked=set(ticked))


def _tick(w, *names):
    from qtpy.QtCore import Qt
    lst = w._patch_state_list
    for i in range(lst.count()):
        item = lst.item(i)
        item.setCheckState(Qt.Checked if item.data(Qt.UserRole) in names
                           else Qt.Unchecked)


def _models(w, c):
    w.lateral_model_path.setText("cyto3")
    w.axial_model_path.setText("cyto3")
    c._refresh_model_status()


def _foot(w) -> str:
    return w.pipeline_status_label.toolTip()


def _tick_protected(w, c, statuses, rows, ticked):
    """Ticks that were set BEFORE the patch became protected (a protected
    row cannot be ticked now, only kept ticked)."""
    c.state_manager.get_ticked = lambda: set(ticked)
    _list(w, statuses, rows, ticked)
    c._refresh_pr_foot_gates()


def test_the_status_names_a_skipped_protected_patch(lay):
    w, c, statuses = lay
    _models(w, c)
    rows = {"a.tif": "unprocessed", "b.tif": "predicted",
            "c.tif": "reconstructed", "d.tif": "curating",
            "e.tif": "finalized"}
    # untouched first: nothing protected ticked -> the plain line
    _tick_protected(w, c, statuses, rows, {"a.tif", "b.tif", "c.tif",
                                           "d.tif"})
    assert _foot(w) == "Ready · 4 ticked"
    _tick_protected(w, c, statuses, rows, {"a.tif", "b.tif", "c.tif",
                                           "d.tif", "e.tif"})
    assert _foot(w) == ("Ready · 5 ticked · 1 protected (finalized) "
                        "is skipped")      # M5: every tick, as the card


def test_the_skipped_line_agrees_in_number_and_names_each_status(lay):
    w, c, statuses = lay
    _models(w, c)
    rows = {"a.tif": "predicted", "b.tif": "finalized",
            "c.tif": "accepted", "d.tif": "finalized"}
    _tick_protected(w, c, statuses, rows, set(rows))
    assert _foot(w) == ("Ready · 4 ticked · 3 protected (accepted, "
                        "finalized) are skipped")


def test_the_skipped_clause_follows_the_not_ready_clause(lay):
    w, c, statuses = lay
    P = _P()
    _models(w, c)
    rows = {"a.tif": "unprocessed", "b.tif": "finalized"}
    _tick_protected(w, c, statuses, rows, set(rows))
    assert _foot(w) == (
        "Ready · 2 ticked" + P.S3_GATE_JOIN + P.S3_GATE_NOTHING_TO_RECONSTRUCT
        + P.S3_GATE_JOIN + "1 protected (finalized) is skipped")


def test_every_ticked_protected_patch_alone_keeps_its_reason(lay):
    w, c, statuses = lay
    P = _P()
    _models(w, c)
    _tick_protected(w, c, statuses, {"b.tif": "finalized"}, {"b.tif"})
    assert _foot(w) == P.S3_GATE_NONE_RUNNABLE


# ---------------------- 2: the unprocessed row without a voxel size
def _vox_tip(rec, width=240):
    """The tooltip of the painted voxel line, through the real delegate."""
    from qtpy.QtWidgets import (QListWidget, QListWidgetItem,
                                QStyleOptionViewItem)
    from napari_epicycle.ui_shell.theme import PATCH_STATUS_COLORS  # noqa
    from napari_epicycle.ui_shell._stack_row_delegate import ROLE_RECORD
    D, d = _narrow_delegate()
    d._help_lines = {"row_vox": "help"}
    lst = QListWidget()
    lst.setItemDelegate(d)
    lst.resize(width, 400)
    it = QListWidgetItem("x")
    it.setData(ROLE_RECORD, rec)
    lst.addItem(it)
    lst.show()
    opt = QStyleOptionViewItem()
    opt.initFrom(lst)
    opt.widget = lst
    opt.rect = lst.visualItemRect(it)
    opt.font = lst.font()
    g = d.layout(opt.rect, rec, opt.font)
    r = g["lines"]["l3n"]
    tip = d.tip_text(opt, lst.indexFromItem(it), r.center())
    lst.deleteLater()
    return tip


def _row(status, spacing, **kw):
    from napari_epicycle.fusion_reconstruction import _step3_rows as R
    return R.build_row("p.tif", status, "CZYX", source_shape=(2, 10, 20, 20),
                       spacing=spacing, **kw)


@pytest.mark.parametrize("status", ["unprocessed", "predicted",
                                    "reconstructed", "curating"])
def test_a_row_without_a_voxel_size_explains_the_pixel_fallback(app, status):
    rec = _row(status, None)
    assert rec["vox_text"] == PT.S3_ROW_SPACING_NONE and rec["vox_warn"]
    assert rec["vox_tip"] == PT.S3_NO_VOXEL_WARNING
    assert _vox_tip(rec) == PT.S3_NO_VOXEL_WARNING


def test_no_second_amber_line_is_drawn_for_it(app):
    """The amber 'voxels' line (A3) IS the row's amber line: the sentence is
    its tooltip, so the row gains no line and no height."""
    D, d = _narrow_delegate()
    with_sentence = _lay(d, _row("unprocessed", None))
    rec = _row("unprocessed", None)
    rec.pop("vox_tip")
    assert _lay(d, rec)["height"] == with_sentence["height"]
    assert not with_sentence["warns"]


def test_a_row_with_a_voxel_size_keeps_its_source_tip(app):
    spacing = {"zyx": [0.52, 0.52, 0.52], "source": "step1"}
    rec = _row("unprocessed", spacing)
    assert "vox_tip" not in rec
    assert _vox_tip(rec) == rec["vox_text"] == PT.spacing_row_text(spacing)


def test_a_protected_row_does_not_carry_the_pixel_sentence(app):
    """Predict skips it, so the cell width never reaches it."""
    rec = _row("finalized", None)
    assert "vox_tip" not in rec


# -------------------------------------- 1: the Model card's trained line
from napari_epicycle.fusion_reconstruction import _step3_rows as R  # noqa: E402


def _rec(kind="bootstrap", ch=(2, 1), run="wound-v2"):
    return {"kind": kind, "trained_channels": list(ch) if ch else None,
            "run_id": run}


def test_the_trained_line_names_the_roles_and_the_step2_run():
    text, known = R.trained_line(_rec(), _rec())
    assert known
    assert text == ("Trained on 2 channels: membrane (1), nuclei (0) · "
                    "Step 2 run “wound-v2”.")


def test_one_channel_reads_singular():
    text, known = R.trained_line(_rec(ch=(1, 0)), _rec(ch=(1, 0)))
    assert known and text.startswith("Trained on 1 channel: membrane (0) · ")


def test_a_refined_bundle_names_no_step2_run():
    text, known = R.trained_line(_rec("refined", (2, 1), "mine"),
                                 _rec("refined", (2, 1), "mine"))
    assert known and text == "Trained on 2 channels: membrane (1), nuclei (0)."


def test_a_domain_specialized_model_names_its_step2_run():
    text, _ = R.trained_line(_rec("domain_specialized"),
                             _rec("domain_specialized"))
    assert text.endswith("Step 2 run “wound-v2”.")


@pytest.mark.parametrize("lat,ax", [
    (_rec(), _rec(ch=None)),
    (_rec("refined", None, "x"), _rec("refined", None, "x")),
    (_rec(ch=None), _rec()),
    (None, None),
])
def test_unknown_channels_are_never_invented(lat, ax):
    text, known = R.trained_line(lat, ax)
    assert not known and text == PT.S3_MODEL_TRAINED_UNKNOWN


@pytest.mark.parametrize("lat,ax", [
    (_rec("pretrained", None, None), _rec("pretrained", None, None)),
    (_rec("pretrained", None, None), _rec()),
    (_rec(), _rec("pretrained", None, None)),
])
def test_a_builtin_model_hides_the_line(lat, ax):
    """A Cellpose built-in records no channels by nature: nothing to say."""
    assert R.trained_line(lat, ax) == ("", False)


def test_two_models_trained_on_different_channels_name_both_sets():
    text, known = R.trained_line(_rec(ch=(2, 1), run="a"),
                                 _rec(ch=(1, 0), run="b"))
    assert known and text == (
        "Trained on different channels: lateral membrane (1), nuclei (0) · "
        "axial membrane (0). Step 2 runs “a” / “b”.")


def test_different_channels_of_one_run_and_of_refined_models():
    text, _ = R.trained_line(_rec(ch=(2, 1), run="a"), _rec(ch=(1, 0), run="a"))
    assert text.endswith("axial membrane (0). Step 2 run “a”.")
    text, _ = R.trained_line(_rec("refined", (2, 1), "m"),
                             _rec("refined", (1, 0), "m"))
    assert text == ("Trained on different channels: lateral membrane (1), "
                    "nuclei (0) · axial membrane (0).")


def test_two_different_step2_runs_name_none():
    text, _ = R.trained_line(_rec(run="a"), _rec(run="b"))
    assert text == "Trained on 2 channels: membrane (1), nuclei (0)."


def test_the_card_shows_the_line_for_a_step2_model(lay, ws):
    w, c, statuses = lay
    model = step2_model(ws, run="cells", channels=(2, 1))
    c.working_dir = str(ws)
    w.lateral_model_path.setText(str(model))
    w.axial_model_path.setText(str(model))
    c._refresh_model_status()
    lbl = w.pr_model_trained_label
    assert lbl.text() == ("Trained on 2 channels: membrane (1), nuclei (0)"
                          " · Step 2 run “cells”.")
    assert not lbl.isHidden()
    from napari_epicycle.ui_shell.theme import DIM_LABEL_STYLE
    assert lbl.styleSheet() == DIM_LABEL_STYLE          # M3, known or not


def test_the_card_greys_an_unrecorded_step2_model_and_hides_a_builtin(lay, ws):
    w, c, statuses = lay
    model = step2_model(ws, run="cells")
    (model.parent / f"{model.name}_training_config.json").unlink()
    c.working_dir = str(ws)
    w.lateral_model_path.setText(str(model))
    w.axial_model_path.setText(str(model))
    c._refresh_model_status()
    lbl = w.pr_model_trained_label
    assert lbl.text() == PT.S3_MODEL_TRAINED_UNKNOWN and not lbl.isHidden()
    _models(w, c)                                     # cyto3: a built-in
    assert lbl.isHidden()
    w.lateral_model_path.setText("")
    w.axial_model_path.setText("")
    c._refresh_model_status()
    assert lbl.isHidden()


def test_the_trained_line_wraps_and_keeps_the_whole_sentence_as_tooltip(lay):
    w, c, statuses = lay
    w.resize(700, 900)
    w.show()
    text = PT.S3_MODEL_TRAINED_ON.format(
        n=2, names="membrane (1), nuclei (0)",
        run=PT.S3_MODEL_TRAINED_RUN.format(run="w" * 44))
    w.show_pr_trained(text, True)
    lbl = w.pr_model_trained_label
    assert lbl.wordWrap() and lbl.toolTip() == text
    # L8: at the REAL 240 px column (card width ~ 240 - margins)
    fm = lbl.fontMetrics()
    line = fm.lineSpacing()
    usual = lbl.heightForWidth(228) // line
    assert 2 <= usual <= 3, usual
    differ = PT.S3_MODEL_TRAINED_DIFFER.format(
        lat="membrane (1), nuclei (0)", ax="membrane (0)",
        runs=PT.S3_MODEL_TRAINED_DIFFER_RUNS.format(
            lat="w" * 44, ax="w" * 44 + "-b"))
    w.show_pr_trained(differ, True)
    # (offscreen glyphs are ~1.6x too wide: the ceiling test, measured in
    # the real font, is the height authority for the longest form)
    assert lbl.heightForWidth(228) // line > usual


# ----------------------------------- 4: the note under the two settings ticks
def test_the_unused_thresholds_are_explained_under_the_ticks(lay):
    w, c, statuses = lay
    w.resize(600, 900)
    w.show()
    note = w.pr_unused_knobs_note
    assert note.text() == PT.S3_UNUSED_KNOBS_NOTE == (
        "Outline tolerance and cell confidence are not used by the 3D "
        "build, so they are not shown.")
    assert note.wordWrap() and not note.isHidden()
    # the thresholds themselves stay built and never shown
    assert w._hidden_knobs.isHidden()
    # directly under the ticks row of the same card
    lay_ = w.norm_cb.parentWidget().layout()
    ticks = lay_.itemAt(lay_.indexOf(note) - 1)
    assert ticks is not None and ticks.layout() is not None
    assert ticks.layout().indexOf(w.norm_cb) >= 0


# ----------------------------------------------------------- the guides
def test_the_guides_quote_what_brick_b_added():
    from tests.fusion_reconstruction.test_step3_pr_guide_parity import (
        GUIDE, MODEL_GUIDE, PATCHES_GUIDE, quotes)
    norm = lambda s: " ".join(str(s).split())      # noqa: E731
    model = {norm(q) for q in quotes(MODEL_GUIDE)}
    assert norm(PT.S3_MODEL_TRAINED_UNKNOWN) in model
    assert norm(PT.S3_MODEL_TRAINED_DIFFER.format(
        lat="membrane (1), nuclei (0)", ax="membrane (0)",
        runs=PT.S3_MODEL_TRAINED_DIFFER_RUNS.format(lat="a", ax="b"))) in model
    assert norm(PT.S3_MODEL_TRAINED_ON.format(
        n=2, names="membrane (1), nuclei (0)",
        run=PT.S3_MODEL_TRAINED_RUN.format(run="wound-v2"))) in model
    assert norm(PT.S3_UNUSED_KNOBS_NOTE) in {norm(q) for q in quotes(GUIDE)}
    patches = {norm(q) for q in quotes(PATCHES_GUIDE)}
    assert norm(PT.S3_NO_VOXEL_WARNING) in patches
    assert norm(PT.S3_GATE_SKIPPED.format(k=1, statuses="finalized")) in patches


# ------------------------------------------ 3: the protected rows (verified)
def test_a_protected_row_carries_its_note_its_tip_and_its_tier_counts(lay):
    """Already built (V1 note / tip, A2 + A3 counts); held here so board D's
    line cannot go missing again."""
    from qtpy.QtCore import Qt
    from napari_epicycle.ui_shell._stack_row_delegate import ROLE_RECORD
    from napari_epicycle.fusion_reconstruction import (
        _step3_tier_counts as TC)
    w, c, statuses = lay
    _list(w, statuses, {"a.tif": "reconstructed", "b.tif": "finalized"})
    five = TC.tier_items({t: 3 + i for i, t in enumerate(TC.TIERS)},
                         {t: (100, 100, 100) for t in TC.TIERS})
    w.set_tier_counts({"a.tif": five, "b.tif": five}, replace=True)
    recs = {w._patch_state_list.item(i).data(Qt.UserRole):
            w._patch_state_list.item(i).data(ROLE_RECORD)
            for i in range(w._patch_state_list.count())}
    fin, rec = recs["b.tif"], recs["a.tif"]
    assert fin["notes_narrow"] == (PT.S3_ROW_PROTECTED,)
    assert fin["row_tip"] == ("finalized — protected; Reset to Predicted in "
                              "Curation to redo it")
    assert len(fin["tier_counts"]) == len(TC.TIERS)
    assert "tier_counts" not in rec and rec["notes_narrow"] == ()


# ============================================ the audit fold (2026-10-10)
import json  # noqa: E402


def _cfg(folder, name, channels):
    (folder / name).write_text(json.dumps(
        {"hyperparameters": {"channels": list(channels)}}), encoding="utf-8")


def test_m1_a_bundles_axial_weights_read_their_own_config(tmp_path):
    from napari_epicycle.ui_shell import _model_files as MF
    f = tmp_path / "bundle"
    f.mkdir()
    for b in ("lateral", "axial"):
        (f / b).write_bytes(b"w")
    _cfg(f, "lateral_training_config.json", (2, 1))
    _cfg(f, "axial_training_config.json", (1, 0))
    assert MF.read_trained_channels(f / "lateral") == [2, 1]
    assert MF.read_trained_channels(f / "axial") == [1, 0]


def test_m1_the_card_names_both_sets_for_a_refined_bundle(lay, ws):
    w, c, statuses = lay
    lat = refined_bundle(ws, name="mix")
    ax = lat.parent / "axial"
    _cfg(lat.parent, "lateral_training_config.json", (2, 1))
    _cfg(lat.parent, "axial_training_config.json", (1, 0))
    c.working_dir = str(ws)
    w.lateral_model_path.setText(str(lat))
    w.axial_model_path.setText(str(ax))
    c._refresh_model_status()
    assert w.pr_model_trained_label.text().startswith(
        "Trained on different channels: lateral membrane (1), nuclei (0) · "
        "axial membrane (0).")


def test_m2_a_missing_axial_config_never_takes_the_lateral_one(lay, ws):
    from napari_epicycle.ui_shell import _model_files as MF
    folder = ws / "models" / "round_03"
    folder.mkdir(parents=True)
    for n in ("Cell_lateral_round_03", "Cell_axial_round_03"):
        (folder / n).write_bytes(b"w")
    _cfg(folder, "Cell_lateral_round_03_training_config.json", (2, 1))
    assert MF.read_trained_channels(folder / "Cell_lateral_round_03") == [2, 1]
    assert MF.read_trained_channels(folder / "Cell_axial_round_03") is None
    w, c, statuses = lay
    c.working_dir = str(ws)
    w.lateral_model_path.setText(str(folder / "Cell_lateral_round_03"))
    w.axial_model_path.setText(str(folder / "Cell_axial_round_03"))
    c._refresh_model_status()
    assert w.pr_model_trained_label.text() == PT.S3_MODEL_TRAINED_UNKNOWN


def test_m2_a_folder_with_one_weights_file_keeps_the_glob_fallback(tmp_path):
    from napari_epicycle.ui_shell import _model_files as MF
    (tmp_path / "model_x").write_bytes(b"w")
    _cfg(tmp_path, "something_training_config.json", (1, 2))
    assert MF.read_trained_channels(tmp_path / "model_x") == [1, 2]


def _fits_two_lines(w):
    from napari_epicycle.fusion_reconstruction.ui import _two_lines
    lbl = w.pipeline_status_label
    return "…" not in lbl.text()


def test_m4_the_skipped_clause_survives_a_narrow_foot(lay):
    w, c, statuses = lay
    _models(w, c)
    w.pipeline_status_label.setFixedWidth(480)   # ~300 px in the real font (offscreen glyphs are ~1.6x wide)
    rows = {"a.tif": "unprocessed", "b.tif": "unprocessed",
            "c.tif": "unprocessed", "d.tif": "accepted",
            "e.tif": "finalized"}
    _tick_protected(w, c, statuses, rows, set(rows))
    text = w.pipeline_status_label.text()
    assert "…" not in text, text
    full = _foot(w)
    assert "2 protected" in full and "skipped" in full
    # the statuses were the first thing to go
    assert "accepted" not in full and "finalized" not in full


def test_l1_a_one_bucket_model_hides_the_trained_line(lay, ws):
    w, c, statuses = lay
    model = step2_model(ws)
    c.working_dir = str(ws)
    w.lateral_model_path.setText(str(model))
    w.axial_model_path.setText("")
    c._refresh_model_status()
    assert w.pr_model_trained_label.isHidden()


def test_l2_a_missing_model_file_hides_the_line():
    assert R.trained_line(_rec("unknown", None, None),
                          _rec("unknown", None, None)) == ("", False)


def test_l3_a_manifestless_legacy_round_names_no_run(ws, tmp_path):
    legacy = tmp_path / "other" / "models" / "round_03" / "Cell_lateral_round_03"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"w")
    rec = MP.view_record(legacy, workspace=str(ws))
    assert rec["kind"] == "bootstrap" and rec["run_id"] is None
    assert "None" not in str(rec["name"])
    # a Step 2 model with a manifest still names its run
    assert MP.view_record(step2_model(ws, run="cells"),
                          workspace=str(ws))["run_id"] == "cells"


def test_l4_grayscale_is_said_as_such():
    text, known = R.trained_line(_rec(ch=(0, 0), run=None),
                                 _rec(ch=(0, 0), run=None))
    assert known and text == PT.S3_MODEL_TRAINED_GRAY.format(run="")
    assert "membrane" not in text
    text, _ = R.trained_line(_rec("refined", (0, 2), "m"),
                             _rec("refined", (0, 2), "m"))
    assert text == "Trained on 2 channels: grayscale, nuclei (1)."


def test_l6_a_pretrained_pick_refreshes_the_cards_lines(lay, ws):
    w, c, statuses = lay
    model = step2_model(ws)
    c.working_dir = str(ws)
    w.lateral_model_path.setText(str(model))
    w.axial_model_path.setText(str(model))
    c._refresh_model_status()
    assert not w.pr_model_trained_label.isHidden()
    c._model_source_mode = "pretrained"
    c._on_pretrained_changed("cyto3")
    assert w.pr_model_trained_label.isHidden()
    assert "cyto3" in w.pr_model_lat_label.text()


# ---- the S1 coupling: the voxel tooltip follows the EFFECTIVE size
def test_s1_a_row_with_a_qc_fallback_size_carries_no_pixel_sentence(lay):
    w, c, statuses = lay
    c.loaded_stack_dir = ""
    c._qc_settings = {"spacing": [1.0, 0.5, 0.5]}
    statuses.update({"a.tif": "unprocessed"})
    unsized = c._pr_unsized_patches(set(statuses))
    assert unsized == set()                # the cell width uses the fallback
    c._qc_settings = {"spacing": None}
    assert c._pr_unsized_patches(set(statuses)) == {"a.tif"}


def test_s1_the_list_passes_the_effective_answer_to_the_rows(lay):
    from qtpy.QtCore import Qt
    from napari_epicycle.ui_shell._stack_row_delegate import ROLE_RECORD
    w, c, statuses = lay
    _list(w, statuses, {"a.tif": "unprocessed"})
    w.refresh_patch_state_list({"a.tif": "unprocessed"},
                               spacing_by_patch={}, unsized=set())
    rec = w._patch_state_list.item(0).data(ROLE_RECORD)
    assert rec["vox_warn"] and "vox_tip" not in rec
    w.refresh_patch_state_list({"a.tif": "unprocessed"},
                               spacing_by_patch={}, unsized={"a.tif"})
    rec = w._patch_state_list.item(0).data(ROLE_RECORD)
    assert rec["vox_tip"] == PT.S3_NO_VOXEL_WARNING
