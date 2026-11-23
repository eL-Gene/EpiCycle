"""The QC table in memory belongs to the patch it was loaded for.  Every
write of it goes to that patch or is refused, and leaving Step 3 asks about
unsaved edits first."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from napari_epicycle.annotator.refinement_state import StagedOp
from napari_epicycle.fusion_reconstruction import _paths
from napari_epicycle.fusion_reconstruction import _step3_pr_text as _pr_text
from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController as C,
    _active_run_path_id,
)


class _SM:
    def get_active_run_id(self):
        return "default"

    def get_patch_downsample_factor(self, name):
        return 1


def _df():
    return pd.DataFrame({"label": [1, 2], "qc_tier": ["good", "bad"],
                         "label_status": ["active", "active"]})


def _ctrl(tmp_path, *, owner="P1.tif", view="", selected="P2.tif"):
    c = C.__new__(C)
    c.working_dir = str(tmp_path)
    c.state_manager = _SM()
    c._qc_features_df = _df()
    c._qc_table_patch = owner
    c._qc_view_patch_name = view
    c._selected_curation_patch = lambda: selected
    c._selected_patch_name = lambda: selected
    c._tiers_dirty = True
    c._curation_dirty = False
    c.logged = []
    c._append_log = c.logged.append
    c.statuses = []
    c._pr_status = lambda text, kind="neutral": c.statuses.append(text)
    c.cards = []
    c._set_plot_status = lambda text, kind="info": c.cards.append(
        (text, kind))
    c.session_lines = []
    c._session_say = c.session_lines.append
    c._refresh_save_button_state = lambda: None
    c._scan_qc_df_for_duplicates = lambda *_a, **_k: None
    return c


def _csv(c, name):
    return (_paths.qc_dir(c.working_dir, run_id=_active_run_path_id(c))
            / f"{name.rsplit('.', 1)[0]}_features.csv")


def test_a_tier_save_after_the_view_is_cleared_writes_the_tables_patch(
        tmp_path):
    c = _ctrl(tmp_path, owner="P1.tif", view="", selected="P2.tif")
    c._save_tiers_csv()
    assert _csv(c, "P1.tif").exists()
    assert not _csv(c, "P2.tif").exists()
    assert c._tiers_dirty is False


def test_a_tier_save_for_another_patch_than_the_table_is_refused(tmp_path):
    c = _ctrl(tmp_path, owner="P1.tif", view="P2.tif")
    c._save_tiers_csv()
    assert not _csv(c, "P1.tif").exists() and not _csv(c, "P2.tif").exists()
    assert c._tiers_dirty is True
    assert c.cards == [(
        _pr_text.S3_SAVE_REFUSED_OTHER_PATCH.format(patch="P1.tif"), "warn")]


def test_a_failing_tier_save_keeps_the_edits_unsaved(tmp_path, monkeypatch):
    from napari_epicycle.fusion_reconstruction import qc_features
    monkeypatch.setattr(qc_features, "save_patch_csv",
                        MagicMock(side_effect=PermissionError("locked")))
    c = _ctrl(tmp_path, owner="P1.tif", view="P1.tif")
    c._save_tiers_csv()
    assert c._tiers_dirty is True
    assert _pr_text.S3_SAVE_FAILED in " ".join(c.logged)
    # whichever section is open shows it
    assert (_pr_text.S3_SAVE_FAILED, "warn") in c.cards
    assert c.session_lines == [_pr_text.S3_SAVE_FAILED]
    assert c.statuses == [_pr_text.S3_SAVE_FAILED]


def test_a_curation_save_for_another_patch_than_the_table_is_refused(tmp_path):
    c = _ctrl(tmp_path, owner="P1.tif", view="P2.tif")
    c._get_active_labels_path = lambda: str(tmp_path / "x.tif")
    c._find_active_labels_layer = lambda: MagicMock(data=np.zeros((2, 2, 2)))
    assert c._save_curation_prepare() is None


def test_commit_triage_for_another_patch_than_the_table_is_refused(tmp_path):
    c = _ctrl(tmp_path, owner="P1.tif", view="P2.tif")
    c._zero_labels_in_viewer = lambda ids: pytest.fail("zeroed")
    assert c._commit_triage_prepare() is None


def test_the_extract_finish_records_the_tables_patch(tmp_path):
    from .test_qc_tier_preservation import (
        _fresh_features_df,
        _make_bare_controller,
    )
    ctrl = _make_bare_controller(prev_df=None, working_dir=tmp_path)
    ctrl._qc_table_patch = ""
    C._on_qc_extract_finished(ctrl, _fresh_features_df([1, 2]), "P1.tif")
    assert ctrl._qc_table_patch == "P1.tif"


# ── leaving Step 3 ────────────────────────────────────────────────────────

def _exit_ctrl(tmp_path):
    c = _ctrl(tmp_path, owner="P1.tif", view="P1.tif")
    c._curation_dirty = True
    c.calls = []
    c._stop_step3_thumbs = lambda: c.calls.append("thumbs")
    c._stop_tier_counts = lambda: c.calls.append("counts")
    c._close_provenance_windows = lambda: c.calls.append("prov")
    c._close_3d_curation_annotator = lambda *a, **k: c.calls.append("close")
    c._get_curation_annotator_widget = lambda: None
    c._reset_qc_plot_area = lambda *a, **k: c.calls.append("plot")
    c._sync_loaded_patch_marker = lambda: None
    return c


def test_leaving_step_3_asks_first_and_a_refusal_keeps_the_user(tmp_path):
    c = _exit_ctrl(tmp_path)
    c._guard_unsaved_changes = lambda: c.calls.append("guard") or False
    assert c.on_step_exit() is False
    assert c.calls == ["guard"]          # nothing was stopped or closed
    assert c._qc_features_df is not None


def test_a_guard_that_raises_on_exit_keeps_the_user(tmp_path):
    c = _exit_ctrl(tmp_path)

    def boom():
        raise PermissionError("locked")

    c._guard_unsaved_changes = boom
    assert c.on_step_exit() is False
    assert c.calls == []
    assert _pr_text.S3_SAVE_FAILED in " ".join(c.logged)


def test_staged_annotator_work_is_refused_before_the_guard(tmp_path):
    c = _exit_ctrl(tmp_path)
    c._guard_unsaved_changes = lambda: pytest.fail("asked")
    w = MagicMock()
    w.staged_kinds.return_value = frozenset({StagedOp.INTERPOLATE})
    c._get_curation_annotator_widget = lambda: w
    assert c.on_step_exit() is False
    assert (_pr_text.S3_STAGED_WORK_OPEN, "warn") in c.cards


def test_a_passed_guard_drops_the_table_and_leaves(tmp_path):
    c = _exit_ctrl(tmp_path)
    c._guard_unsaved_changes = lambda: True
    assert c.on_step_exit() is not False
    assert c._qc_features_df is None and c._qc_table_patch == ""
    assert "close" in c.calls and "plot" in c.calls


# ── opening the annotator on another patch than the table ────────────────

def test_opening_the_annotator_on_another_patch_asks_about_unsaved_edits(
        tmp_path):
    c = _ctrl(tmp_path, owner="P1.tif", view="", selected="P2.tif")
    for flag in ("_curation_busy", "_patch_load_in_flight",
                 "_prediction_running", "_reconstruction_running",
                 "_refinement_running", "_extraction_running"):
        setattr(c, flag, False)
    c._ensure_llm_removal_callback = lambda: None
    asked = []
    c._guard_unsaved_changes = lambda: asked.append(1) or False
    c._resolve_patch_payload = lambda name: pytest.fail("opened")
    c._open_3d_curation_annotator()
    assert asked == [1]


def test_opening_on_another_patch_leaves_none_of_the_old_patchs_layers(
        tmp_path):
    c = _ctrl(tmp_path, owner="P1.tif", view="", selected="P2.tif")
    for flag in ("_curation_busy", "_patch_load_in_flight",
                 "_prediction_running", "_reconstruction_running",
                 "_refinement_running", "_extraction_running"):
        setattr(c, flag, False)
    c._ensure_llm_removal_callback = lambda: None
    c._guard_unsaved_changes = lambda: True
    layers = ["stack_P1", "curation_3d_P1"]
    c._close_3d_curation_annotator = (
        lambda clear_layers=True: layers.clear() if clear_layers else None)
    c._reset_qc_plot_area = lambda *a, **k: None
    seen = []

    def _payload(name):
        seen.append(list(layers))
        return None

    c._resolve_patch_payload = _payload
    c._open_3d_curation_annotator()
    assert seen == [[]]


def test_a_staged_edit_refuses_the_open_before_any_question(tmp_path):
    c = _ctrl(tmp_path, owner="P1.tif", view="", selected="P2.tif")
    for flag in ("_curation_busy", "_patch_load_in_flight",
                 "_prediction_running", "_reconstruction_running",
                 "_refinement_running", "_extraction_running"):
        setattr(c, flag, False)
    c._ensure_llm_removal_callback = lambda: None
    c._staged_annotator_work = lambda: True
    c._guard_unsaved_changes = lambda: pytest.fail("asked")
    c._close_3d_curation_annotator = lambda *a, **k: pytest.fail("closed")
    c._resolve_patch_payload = lambda name: pytest.fail("opened")
    c._open_3d_curation_annotator()
    assert (_pr_text.S3_STAGED_WORK_OPEN, "warn") in c.cards
    assert c._qc_features_df is not None


def test_a_tier_save_refreshes_the_rows_counts_for_the_saved_patch(tmp_path):
    c = _ctrl(tmp_path, owner="P1.tif", view="P1.tif")
    c._start_tier_counts = MagicMock()
    c._save_tiers_csv()
    c._start_tier_counts.assert_called_once_with("save tiers", ["P1.tif"])


def test_a_refused_tier_save_does_not_refresh_the_counts(tmp_path):
    c = _ctrl(tmp_path, owner="P1.tif", view="P2.tif")
    c._start_tier_counts = MagicMock()
    c._save_tiers_csv()
    c._start_tier_counts.assert_not_called()


def test_a_failing_tier_save_does_not_refresh_the_counts(tmp_path, monkeypatch):
    from napari_epicycle.fusion_reconstruction import qc_features
    monkeypatch.setattr(qc_features, "save_patch_csv",
                        MagicMock(side_effect=PermissionError("locked")))
    c = _ctrl(tmp_path, owner="P1.tif", view="P1.tif")
    c._start_tier_counts = MagicMock()
    c._save_tiers_csv()
    c._start_tier_counts.assert_not_called()


def test_a_count_failure_after_a_tier_save_does_not_unsave_it(tmp_path):
    c = _ctrl(tmp_path, owner="P1.tif", view="P1.tif")
    c._start_tier_counts = MagicMock(side_effect=RuntimeError("boom"))
    c._save_tiers_csv()
    assert c._tiers_dirty is False
