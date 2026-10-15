"""The B1b audit's findings, one test each (B1-*, G1-*, Q1, Q2).

Qt-free where the controller runs on the fakes; the Export-window cases use
the real panel.
"""
from __future__ import annotations

import os
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from napari_epicycle.segmentation import _step4_models as M
from napari_epicycle.segmentation import _step4_rows as R
from napari_epicycle.segmentation import _step4_text as T

from ._step4_fakes import controller

ONES = {"zyx": [1.0, 1.0, 1.0], "source": "user"}
UM = {"zyx": [1.0, 0.5, 0.5], "source": "step1"}


# ------------------------------------------------------------- B1-1 (HIGH)


@pytest.fixture
def ones(tmp_path):
    c = controller(tmp_path)
    run = tmp_path / "artifacts" / "step4" / "runs" / "r"
    run.mkdir(parents=True)
    lab = run / "labels_3d.tif"
    tifffile.imwrite(str(lab), np.zeros((2, 3, 3), np.uint32))
    c.state_manager.set_stack_status("s.tif", "reconstructed",
                                     labels_path=str(lab), spacing=ONES)
    c._current_stack_name = "s.tif"
    c._labels_volume = np.zeros((2, 3, 3), np.uint32)
    c._open_analysis_dock = lambda: None
    return c


def _measure(c, monkeypatch):
    """Run Measure's launch and a current finish, no thread."""
    launched = []
    monkeypatch.setattr(c, "_launch_measure_worker",
                        lambda w, t: launched.append(w) or object())
    c._run_feature_extraction(skip_probability=True)
    assert launched
    c._on_extract_finished(pd.DataFrame({"label": [1, 2]}), "s.tif")


def test_a_user_spacing_of_ones_is_a_spacing(ones):
    assert ones._spacing_for("s.tif") == (1.0, 1.0, 1.0)
    assert ones._get_spacing() == (1.0, 1.0, 1.0)


def test_no_record_spacing_is_none(ones):
    ones.state_manager.update_record("s.tif", spacing=None)
    assert ones._spacing_for("s.tif") is None


def test_ones_measured_is_not_stale_for_ever(ones, monkeypatch):
    _measure(ones, monkeypatch)
    rec = ones.state_manager.get_record("s.tif")
    assert rec["measured_spacing"]["zyx"] == [1.0, 1.0, 1.0]
    assert not ones._measure_stale(rec)
    ones._refresh_mt_panel()
    assert ones.ui.finalize_btn.isEnabled()


def test_ones_finalized_can_go_in_um(ones, monkeypatch, tmp_path):
    _measure(ones, monkeypatch)
    csv = tmp_path / "s.tif_features.csv"
    pd.DataFrame({"label": [1]}).to_csv(csv, index=False)
    ones.state_manager.update_record("s.tif", qc_features_csv=str(csv))
    rec = ones.state_manager.get_record("s.tif")
    ones.state_manager.set_stack_status(
        "s.tif", "finalized", finalized_fingerprint=M.finalized_fingerprint(
            rec["labels_path"], rec["qc_features_csv"]))
    row = R.export_row("s.tif", ones.state_manager.get_record("s.tif"))
    assert row["can_go"] and row["why"] == ""
    df = R.add_meta_columns(pd.DataFrame({"stack_name": ["s.tif"]}),
                            {"s.tif": ones.state_manager.get_record(
                                "s.tif")})
    assert df.loc[0, "meta_units"] == "um"


# ------------------------------------------------------------------ B1-2


@pytest.fixture
def smoothed(tmp_path):
    c = controller(tmp_path)
    c.state_manager.set_stack_status("s.tif", "qc_extracted")
    c._current_stack_name = "s.tif"
    c._labels_volume = np.zeros((2, 3, 3), np.uint32)
    new = c._labels_volume.copy()
    new[0, 0, 0] = 1
    c._on_smooth_finished(types.SimpleNamespace(
        working_volume=new, emptied_ids=[], touched_ids=[1],
        n_changed_total=1))
    return c


def test_undo_of_smooth_is_refused_after_a_paint(smoothed):
    c = smoothed
    c._labels_volume[1, 2, 2] = 7                    # a hand edit ...
    c._on_labels_painted()                           # ... napari's paint
    vol = c._labels_volume
    c._undo_last_edit()
    assert c._labels_volume is vol and vol[1, 2, 2] == 7
    assert c.ui.qc_status_line.sets[-1] == T.MT_UNDO_AFTER_PAINT


def test_undo_of_smooth_without_a_paint_still_works(smoothed):
    smoothed._undo_last_edit()
    assert smoothed._labels_volume[0, 0, 0] == 0


# ------------------------------------------------------------------ B1-6


def test_an_exact_undo_restores_the_held_edit_time(tmp_path, monkeypatch):
    import napari_epicycle.ui_shell.theme as theme
    monkeypatch.setattr(theme, "themed_confirm", lambda *a, **k: True)
    c = controller(tmp_path)
    c.state_manager.set_stack_status("s.tif", "qc_extracted")
    c._current_stack_name = "s.tif"
    vol = np.zeros((2, 3, 3), np.uint32)
    vol[0, 0, 0] = 1
    c._labels_volume = vol
    c._qc_features_df = pd.DataFrame({"label": [1], "qc_tier": ["bad"]})
    c._delete_bad_labels()
    assert c._pending_edits.get("labels_changed_at")
    c._undo_last_edit()
    assert "labels_changed_at" not in c._pending_edits


# ------------------------------------------------------------------ B1-5


def test_the_mesh_flag_clears_even_if_the_ending_fails(tmp_path,
                                                       monkeypatch):
    c = controller(tmp_path)
    c._mesh_running = True

    class _Proc:
        stdout = stderr = None

        def poll(self):
            return 0
    c._mesh_proc = _Proc()
    c._mesh_job = {"dir": tmp_path, "n": 1, "script": tmp_path / "x",
                   "stack": "s.tif"}
    monkeypatch.setattr(c, "_export_status",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError))
    with pytest.raises(RuntimeError):
        c._poll_mesh_export(tmp_path, 1, tmp_path / "x")
    assert c._mesh_running is False


# ------------------------------------------------- export rows (pure) Q1 Q2


def _files(tmp_path, name="x.tif"):
    lab = tmp_path / f"{name}.labels.tif"
    csv = tmp_path / f"{name}_features.csv"
    lab.write_bytes(b"l")
    csv.write_text("label\n1\n", encoding="utf-8")
    return str(lab), str(csv)


def _iso(ts):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


def test_q1_legacy_finalized_untouched_goes_with_a_note(tmp_path):
    lab, csv = _files(tmp_path)
    rec = {"status": "finalized", "labels_path": lab, "qc_features_csv": csv,
           "finalized_at": _iso(time.time() + 60), "spacing": UM,
           "measured_spacing": UM}
    row = R.export_row("x.tif", rec)
    assert row["can_go"] and row["why"] == T.EXPORT_ROW_LEGACY


def test_q1_legacy_finalized_edited_after_is_refused(tmp_path):
    lab, csv = _files(tmp_path)
    rec = {"status": "finalized", "labels_path": lab, "qc_features_csv": csv,
           "finalized_at": _iso(time.time() - 3600), "spacing": UM,
           "measured_spacing": UM}
    row = R.export_row("x.tif", rec)
    assert not row["can_go"] and row["why"] == T.EXPORT_ROW_WHY_EDITED
    rec.pop("finalized_at")                   # nothing to compare with
    assert not R.export_row("x.tif", rec)["can_go"]


def test_q2_units_unknown_before_the_revamp(tmp_path):
    lab, csv = _files(tmp_path)
    rec = {"status": "finalized", "labels_path": lab, "qc_features_csv": csv,
           "finalized_fingerprint": M.finalized_fingerprint(lab, csv),
           "spacing": UM}                     # no measured_spacing key
    row = R.export_row("x.tif", rec)
    assert row["can_go"] and row["why"] == T.EXPORT_ROW_UNITS_UNKNOWN
    df = R.add_meta_columns(pd.DataFrame({"stack_name": ["x.tif"]}),
                            {"x.tif": rec})
    assert df.loc[0, "meta_units"] == "unknown"
    assert pd.isna(df.loc[0, "meta_spacing_z"])


def test_b1_7_only_the_csv_changed_says_finalize_again(tmp_path):
    lab, csv = _files(tmp_path)
    rec = {"status": "finalized", "labels_path": lab, "qc_features_csv": csv,
           "finalized_fingerprint": M.finalized_fingerprint(lab, csv),
           "spacing": UM, "measured_spacing": UM}
    time.sleep(0.02)
    Path(csv).write_text("label,qc_tier\n1,good\n", encoding="utf-8")
    row = R.export_row("x.tif", rec)
    assert not row["can_go"] and row["why"] == T.EXPORT_ROW_WHY_REFINALIZE


def test_g1_2_the_loaded_stacks_unsaved_edits_are_said(tmp_path):
    lab, csv = _files(tmp_path)
    rec = {"status": "finalized", "labels_path": lab, "qc_features_csv": csv,
           "finalized_fingerprint": M.finalized_fingerprint(lab, csv),
           "spacing": UM, "measured_spacing": UM}
    rows = R.export_rows({"x.tif": rec}, unsaved="x.tif")
    assert rows[0]["can_go"] and rows[0]["why"] == T.EXPORT_ROW_UNSAVED


def test_g1_5_the_count_reads_right():
    assert T.say(T.EXPORT_FINALIZED_COUNT, k=1, n=1).startswith(
        "1 of 1 stack ")
    assert T.say(T.EXPORT_FINALIZED_COUNT, k=2, n=5).startswith(
        "2 of 5 stacks ")


# ------------------------------------------------------------------ G1-4


def test_g1_4_the_row_counts_the_measured_cells():
    rec = {"status": "qc_extracted", "n_cells": 221, "measured_cells": 165,
           "stack_shape": [2, 4, 8, 8], "axes": "CZYX"}
    row = R.build_row({"name": "x.tif", "path": ""}, rec)
    assert "165 cells" in row["shape_text"]
    assert "221" not in row["shape_text"]


# ------------------------------------------------------------------ G1-9


def test_g1_9_review_chip_says_not_measured(tmp_path):
    c = controller(tmp_path)
    c.state_manager.set_stack_status("s.tif", "reconstructed")
    c._current_stack_name = "s.tif"
    c._labels_volume = np.zeros((2, 2, 2), np.uint32)
    c._refresh_mt_panel()
    assert c.ui.chips[2] == T.MT_CHIP_NEVER


# ------------------------------------------------------------------ G1-3


def test_g1_3_measure_start_clears_the_status_and_shows_the_eta(
        ones, monkeypatch):
    ones.ui.show_mt_status("Saved s.tif.")
    launched = []
    monkeypatch.setattr(ones, "_launch_measure_worker",
                        lambda w, t: launched.append(w) or object())
    ones._run_feature_extraction(skip_probability=True)
    assert ones.ui.qc_status_line.sets[-1] == ""
    ones._measure_t0 = time.monotonic() - 30.0
    ones._on_measure_progress(50)
    assert ones.ui.qc_status_line.sets[-1] == T.BUSY_MEASURE_ETA.format(
        t="30 s")


# ------------------------------------------------------------------ B1-8


def test_b1_8_meta_keys_on_the_record_name_not_the_csv_stem(tmp_path):
    csv = tmp_path / "renamed_by_hand.csv"
    pd.DataFrame({"label": [1], "label_status": ["active"]}).to_csv(
        csv, index=False)
    out = R.merge_stack_csvs({"x.tif": str(csv)})
    assert list(out["stack_name"]) == ["x.tif"]
