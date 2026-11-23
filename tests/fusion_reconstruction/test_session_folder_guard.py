"""Changing the Session folder is a guarded move: refused while a job runs
(no dialog), asks about unsaved edits and the open annotator while the old
folder is still the working one, and resets the per-patch state before the
new folder is adopted."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from napari_epicycle.fusion_reconstruction import _step3_pr_text as _pr_text
from napari_epicycle.fusion_reconstruction import controller as ctl_mod
from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController as C,
)


class _Dialog:
    """Stands in for QFileDialog: records that it was reached."""

    def __init__(self, pick):
        self.pick = pick
        self.calls = 0

    def getExistingDirectory(self, *a, **k):
        self.calls += 1
        return self.pick


def _ctrl(monkeypatch, tmp_path, *, pick="NEW"):
    c = C.__new__(C)
    c.working_dir = str(tmp_path / "old")
    c.viewer = None
    c._qc_features_df = "TABLE"
    c._qc_table_patch = "P1.tif"
    c._qc_view_patch_name = "P1.tif"
    c._qc_view_layers = ["lyr"]
    c._cached_source_path = "/x"
    c._curation_dirty = False
    c._tiers_dirty = False
    c._run_epoch = 0
    for t in ("_compute_dispatch_token", "_patch_load_token",
              "_commit_triage_token", "_save_curation_token"):
        setattr(c, t, 0)
    for flag in ("_curation_busy", "_patch_load_in_flight",
                 "_prediction_running", "_reconstruction_running",
                 "_refinement_running", "_extraction_running",
                 "_commit_triage_in_flight", "_save_in_flight",
                 "_batch_refine_running", "_extract_all_running",
                 "_qc_extract_running", "_reshape_running",
                 "_add_mode_running", "_refine_running"):
        setattr(c, flag, False)
    c.events = []
    c.statuses = []
    c._session_say = lambda text: c.statuses.append(text)
    c.logged = []
    c._append_log = c.logged.append
    c._get_open_annotator_widget = lambda: None
    c._get_curation_annotator_widget = lambda: None
    c._sync_loaded_patch_marker = lambda: None
    c._reset_qc_plot_area = lambda *a, **k: c.events.append("plot")
    c._guard_unsaved_changes = lambda: (
        c.events.append(("guard", c.working_dir)) or True)
    c._close_3d_curation_annotator = lambda *a, **k: c.events.append("close")
    c._set_workspace_path_quiet = lambda p: (
        c.events.append("path"), setattr(c, "working_dir", p))
    c._prompt_step3_migration = lambda p: c.events.append("migrate")
    c._adopt_workspace_quiet = lambda p, _path_already_set=False: \
        c.events.append("adopt")
    c._publish_workspace_to_shell = lambda p: c.events.append("publish")
    c._load_stack_dir = lambda p, **k: c.events.append("load_stack")
    dlg = _Dialog(pick)
    monkeypatch.setattr(ctl_mod, "QFileDialog", dlg)
    c.dlg = dlg
    return c


def test_browse_is_refused_while_a_job_runs_and_never_reaches_a_dialog(
        monkeypatch, tmp_path):
    c = _ctrl(monkeypatch, tmp_path)
    c._prediction_running = True
    c._browse_workdir()
    assert c.dlg.calls == 0
    assert c.working_dir == str(tmp_path / "old")
    assert c.statuses == [_pr_text.S3_BROWSE_LOCKED]
    assert any("[STEP3-BROWSE]" in t for t in c.logged)
    assert c.events == []


def test_use_another_folder_is_refused_while_a_job_runs(monkeypatch, tmp_path):
    c = _ctrl(monkeypatch, tmp_path)
    c._reconstruction_running = True
    c._browse_stack_dir()
    assert c.dlg.calls == 0
    assert c.events == []


def test_staged_annotator_work_refuses_the_change(monkeypatch, tmp_path):
    c = _ctrl(monkeypatch, tmp_path)
    c._staged_annotator_work = lambda: True
    c._browse_workdir()
    assert c.statuses == [_pr_text.S3_STAGED_WORK_OPEN]
    assert c.events == []
    assert c.working_dir == str(tmp_path / "old")


def test_a_refused_unsaved_edits_question_changes_nothing(
        monkeypatch, tmp_path):
    c = _ctrl(monkeypatch, tmp_path)
    c._guard_unsaved_changes = lambda: False
    c._browse_workdir()
    assert c.events == []
    assert c.working_dir == str(tmp_path / "old")
    assert c._qc_features_df == "TABLE"


def test_the_guard_runs_on_the_old_folder_then_the_state_is_reset(
        monkeypatch, tmp_path):
    c = _ctrl(monkeypatch, tmp_path, pick=str(tmp_path / "new"))
    c._browse_workdir()
    old = str(tmp_path / "old")
    assert c.events[0] == ("guard", old)
    assert c.events.index("close") < c.events.index("path")
    assert c.events.index("path") < c.events.index("adopt")
    assert c._qc_features_df is None and c._qc_table_patch == ""
    assert c._qc_view_patch_name == ""
    assert c._run_epoch >= 1 and c._patch_load_token >= 1


def test_an_open_annotator_needs_a_browse_worded_confirm(
        monkeypatch, tmp_path):
    c = _ctrl(monkeypatch, tmp_path, pick=str(tmp_path / "new"))
    c._get_curation_annotator_widget = lambda: object()
    asked = []
    c.ui = object()
    from napari_epicycle.ui_shell import theme
    monkeypatch.setattr(
        theme, "themed_confirm",
        lambda *a, **k: asked.append(a[2]) or False)
    c._browse_workdir()
    assert asked == [_pr_text.S3_BROWSE_CONFIRM_BODY]
    assert "path" not in c.events and "close" not in c.events


def test_adopting_a_workspace_resets_the_per_patch_state():
    """The shell's adopt door reaches the same reset (it does not go through
    Browse)."""
    c = C.__new__(C)
    c._qc_features_df = "TABLE"
    c._qc_table_patch = "P1.tif"
    c._qc_view_patch_name = "P1.tif"
    c._qc_view_layers = []
    c._run_epoch = 0
    c._curation_dirty = c._tiers_dirty = True
    c.state_manager = MagicMock()
    c.state_manager.loaded_stack_dir = ""
    c.ui = MagicMock()
    c._sync_loaded_patch_marker = lambda: None
    c._reset_qc_plot_area = lambda *a, **k: None
    c._append_log = lambda t: None
    for name in ("_reset_workspace_scoped_caches", "_post_load_legacy_notice",
                 "_load_stack_dir", "_refresh_patch_state_list",
                 "_clear_model_widgets", "_discover_workspace_models",
                 "_restore_start_from", "_refresh_patch_list_colors",
                 "_refresh_summary_dashboard",
                 "_populate_refinement_source_combo",
                 "_refresh_refinement_patch_list",
                 "_refresh_per_orientation_summary", "_refresh_mode_hint",
                 "_refresh_run_picker", "_set_workspace_path_quiet"):
        setattr(c, name, lambda *a, **k: None)
    c._state_file_is_pre_v3 = lambda p: False
    c._use_step1_patch_default = lambda: True
    c._restore_persisted_model_selection = lambda: True
    c._adopt_workspace_quiet("somewhere")
    assert c._qc_features_df is None
    assert c._qc_table_patch == "" and c._qc_view_patch_name == ""
    assert c._run_epoch >= 1
    assert c._curation_dirty is False and c._tiers_dirty is False


def test_the_two_session_buttons_lock_with_the_curation_surface():
    assert "set_workdir_browse_button" in C._CURATION_LOCK_WIDGETS
    assert "set_patch_directory_button" in C._CURATION_LOCK_WIDGETS
    from napari_epicycle.fusion_reconstruction import ui as ui_mod
    pr = getattr(ui_mod, "_PR_SURFACE_LOCK_WIDGETS", ())
    assert "set_workdir_browse_button" not in pr


def test_use_another_folder_forgets_the_old_table(monkeypatch, tmp_path):
    c = _ctrl(monkeypatch, tmp_path, pick=str(tmp_path / "stacks"))
    c._browse_stack_dir()
    assert "load_stack" in c.events
    assert c._qc_features_df is None and c._qc_table_patch == ""


def test_use_another_folder_forgets_an_entered_qc_spacing(
        monkeypatch, tmp_path):
    """A 1 / 1 / 1 the user entered in QC Settings was for the old folder's
    images: the new folder starts with it unknown again."""
    c = _ctrl(monkeypatch, tmp_path, pick=str(tmp_path / "stacks"))
    c._qc_spacing_entered = True
    c._browse_stack_dir()
    assert "load_stack" in c.events
    assert c._qc_spacing_entered is False


def test_a_cancelled_folder_pick_keeps_an_entered_qc_spacing(
        monkeypatch, tmp_path):
    c = _ctrl(monkeypatch, tmp_path, pick="")
    c._qc_spacing_entered = True
    c._browse_stack_dir()
    assert c._qc_spacing_entered is True


def test_adopting_a_workspace_forgets_an_entered_qc_spacing():
    """The shell's adopt door and Browse (which goes through it) both
    start the new workspace with QC Settings' 1 / 1 / 1 unknown again."""
    c = C.__new__(C)
    c._qc_spacing_entered = True
    c._qc_features_df = None
    c._qc_view_layers = []
    c._run_epoch = 0
    c.state_manager = MagicMock()
    c.state_manager.loaded_stack_dir = ""
    c.ui = MagicMock()
    c._sync_loaded_patch_marker = lambda: None
    c._reset_qc_plot_area = lambda *a, **k: None
    c._append_log = lambda t: None
    for name in ("_reset_workspace_scoped_caches", "_post_load_legacy_notice",
                 "_load_stack_dir", "_refresh_patch_state_list",
                 "_clear_model_widgets", "_discover_workspace_models",
                 "_restore_start_from", "_refresh_patch_list_colors",
                 "_refresh_summary_dashboard",
                 "_populate_refinement_source_combo",
                 "_refresh_refinement_patch_list",
                 "_refresh_per_orientation_summary", "_refresh_mode_hint",
                 "_refresh_run_picker", "_set_workspace_path_quiet"):
        setattr(c, name, lambda *a, **k: None)
    c._state_file_is_pre_v3 = lambda p: False
    c._use_step1_patch_default = lambda: True
    c._restore_persisted_model_selection = lambda: True
    c._adopt_workspace_quiet("somewhere")
    assert c._qc_spacing_entered is False
