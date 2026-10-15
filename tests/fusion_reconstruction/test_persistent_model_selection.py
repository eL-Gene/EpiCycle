"""Tests for persistent model selection (2026-05-18).

Covers:
  - state_manager.get_model_selection / set_model_selection round-trip
  - state_manager backwards-compat: old state files without the new fields
  - controller._restore_persisted_model_selection: True/False outcomes
  - controller._refresh_model_status: label formatting
  - controller broadened _discover_workspace_models: scans
    refinement/ + round_N/ by mtime
"""
from __future__ import annotations

import json
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from napari_epicycle.fusion_reconstruction.state_manager import (
    ReconstructionStateManager,
)


# ---------------------------------------------------------------------------
# state_manager helpers
# ---------------------------------------------------------------------------


class TestStateManagerModelSelection:

    def test_default_get_returns_none_paths(self, tmp_path):
        sm = ReconstructionStateManager()
        sm.load(str(tmp_path))
        sel = sm.get_model_selection()
        assert sel["mode"] == "auto"
        assert sel["lateral_path"] is None
        assert sel["axial_path"] is None

    def test_set_then_get_round_trip(self, tmp_path):
        sm = ReconstructionStateManager()
        sm.load(str(tmp_path))
        sm.set_model_selection(
            mode="import",
            lateral_path="/path/to/lat.npy",
            axial_path="/path/to/ax.npy",
        )
        sel = sm.get_model_selection()
        assert sel["mode"] == "import"
        assert sel["lateral_path"] == "/path/to/lat.npy"
        assert sel["axial_path"] == "/path/to/ax.npy"

    def test_partial_update_preserves_other_fields(self, tmp_path):
        sm = ReconstructionStateManager()
        sm.load(str(tmp_path))
        sm.set_model_selection(
            mode="import", lateral_path="/a.npy", axial_path="/b.npy",
        )
        # Now update only axial — lateral + mode must persist.
        sm.set_model_selection(axial_path="/c.npy")
        sel = sm.get_model_selection()
        assert sel["mode"] == "import"
        assert sel["lateral_path"] == "/a.npy"
        assert sel["axial_path"] == "/c.npy"

    def test_empty_string_clears_path(self, tmp_path):
        sm = ReconstructionStateManager()
        sm.load(str(tmp_path))
        sm.set_model_selection(lateral_path="/a.npy")
        sm.set_model_selection(lateral_path="")
        sel = sm.get_model_selection()
        assert sel["lateral_path"] is None

    def test_invalid_mode_raises(self, tmp_path):
        sm = ReconstructionStateManager()
        sm.load(str(tmp_path))
        with pytest.raises(ValueError, match="Invalid model_source_mode"):
            sm.set_model_selection(mode="not_a_real_mode")

    def test_pretrained_mode_persists_bare_name(self, tmp_path):
        sm = ReconstructionStateManager()
        sm.load(str(tmp_path))
        sm.set_model_selection(
            mode="pretrained",
            lateral_path="cyto3",
            axial_path="cyto3",
        )
        sel = sm.get_model_selection()
        assert sel["mode"] == "pretrained"
        assert sel["lateral_path"] == "cyto3"
        assert sel["axial_path"] == "cyto3"

    def test_old_workspace_without_fields_defaults_safely(self, tmp_path):
        """Backwards compat: a state file written before 2026-05-18 (no
        model_source_mode / lateral_model_path / axial_model_path
        fields) should load cleanly and return defaults via .get().

        Post-2026-05-20 cutover: state JSON filename is step3_state.json.
        Previously this test wrote to "reconstruction_state.json"
        which state_manager no longer reads — assertion passed
        against the empty default rather than exercising the v1
        backward-compat path.
        """
        # Write a synthetic v1 state file with no model fields.
        state_path = tmp_path / "step3_state.json"
        state_path.write_text(
            json.dumps({
                "schema_version": 1,
                "patches": {},
                "active_patch": None,
                "loaded_stack_dir": None,
                "last_updated": None,
            }),
            encoding="utf-8",
        )
        sm = ReconstructionStateManager()
        sm.load(str(tmp_path))
        sel = sm.get_model_selection()
        assert sel["mode"] == "auto"  # default
        assert sel["lateral_path"] is None
        assert sel["axial_path"] is None

    def test_persists_through_save_load_cycle(self, tmp_path):
        sm1 = ReconstructionStateManager()
        sm1.load(str(tmp_path))
        sm1.set_model_selection(
            mode="pretrained",
            lateral_path="cyto3",
            axial_path="nuclei",
        )
        sm1.save()  # flush to disk

        # Fresh instance, reload — should see persisted fields.
        sm2 = ReconstructionStateManager()
        sm2.load(str(tmp_path))
        sel = sm2.get_model_selection()
        assert sel["mode"] == "pretrained"
        assert sel["lateral_path"] == "cyto3"
        assert sel["axial_path"] == "nuclei"


# ---------------------------------------------------------------------------
# controller helpers — bare-instance pattern
# ---------------------------------------------------------------------------


def _make_bare_controller_with_widgets():
    """Build a Step-3 controller stub with the minimum widgets needed
    for the new helpers to run without Qt."""
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )
    ctrl = FusionReconstructionController.__new__(FusionReconstructionController)
    # Path / status widgets — use MagicMock so .text() returns
    # a real string we can control via .text.return_value.
    lateral_path = MagicMock()
    lateral_path.text = MagicMock(return_value="")
    axial_path = MagicMock()
    axial_path.text = MagicMock(return_value="")
    ctrl.ui = types.SimpleNamespace(
        lateral_model_path=lateral_path,
        axial_model_path=axial_path,
        status_lateral_model_value=MagicMock(),
        status_axial_model_value=MagicMock(),
    )
    ctrl._log = []
    ctrl._append_log = lambda msg: ctrl._log.append(msg)
    return ctrl


def _bind_real(ctrl, *names):
    """Bind real controller methods to a bare instance."""
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )
    for name in names:
        real = getattr(FusionReconstructionController, name)
        ctrl.__dict__[name] = types.MethodType(real, ctrl)


class TestRefreshModelStatus:

    def test_empty_paths_show_dash(self):
        ctrl = _make_bare_controller_with_widgets()
        _bind_real(ctrl, "_refresh_model_status")
        ctrl._refresh_model_status()
        ctrl.ui.status_lateral_model_value.setText.assert_called_with("-")
        ctrl.ui.status_axial_model_value.setText.assert_called_with("-")

    def test_npy_path_shows_basename(self):
        ctrl = _make_bare_controller_with_widgets()
        _bind_real(ctrl, "_refresh_model_status")
        ctrl.ui.lateral_model_path.text.return_value = "/some/long/path/foo.npy"
        ctrl.ui.axial_model_path.text.return_value = "/another/bar.npy"
        ctrl._refresh_model_status()
        ctrl.ui.status_lateral_model_value.setText.assert_called_with("foo.npy")
        ctrl.ui.status_axial_model_value.setText.assert_called_with("bar.npy")

    def test_pretrained_name_shows_with_prefix(self):
        ctrl = _make_bare_controller_with_widgets()
        _bind_real(ctrl, "_refresh_model_status")
        ctrl.ui.lateral_model_path.text.return_value = "cyto3"
        ctrl.ui.axial_model_path.text.return_value = "nuclei"
        ctrl._refresh_model_status()
        ctrl.ui.status_lateral_model_value.setText.assert_called_with(
            "Pretrained: cyto3"
        )
        ctrl.ui.status_axial_model_value.setText.assert_called_with(
            "Pretrained: nuclei"
        )

    def test_refinement_path_shows_user_model_name(self):
        """Step 3 refinement saves to `.../refinement/{user_model_name}/lateral_model`.
        The filename ('lateral_model') is generic boilerplate; the
        user-chosen name is the parent dir.  Status label should
        show the meaningful name, not 'lateral_model'."""
        ctrl = _make_bare_controller_with_widgets()
        _bind_real(ctrl, "_refresh_model_status")
        ctrl.ui.lateral_model_path.text.return_value = (
            "/workspace/models/refinement/mouse_intestine_round_3/lateral_model"
        )
        ctrl.ui.axial_model_path.text.return_value = (
            "/workspace/models/refinement/mouse_intestine_round_3/axial_model"
        )
        ctrl._refresh_model_status()
        ctrl.ui.status_lateral_model_value.setText.assert_called_with(
            "mouse_intestine_round_3"
        )
        ctrl.ui.status_axial_model_value.setText.assert_called_with(
            "mouse_intestine_round_3"
        )

    def test_post_restructure_bare_bucket_filename_extracts_user_name(self):
        """Post-2026-05-20 Phase 6 restructure: weight files lifted out
        of the cellpose-inserted `models/` subdir end up at
        `.../artifacts/step3/refinement/models/{user_name}/lateral`
        (bare bucket name, no `_model` suffix).

        Audit fix 2026-05-21: `_label_for` extended to recognize bare
        "lateral" / "axial" filenames as boilerplate, mirroring the
        legacy "lateral_model" / "axial_model" handling.  This test
        guards against future regressions where the post-restructure
        naming might be dropped from the recognized list.
        """
        ctrl = _make_bare_controller_with_widgets()
        _bind_real(ctrl, "_refresh_model_status")
        ctrl.ui.lateral_model_path.text.return_value = (
            "/workspace/artifacts/step3/refinement/models/2026-05-21-test2/lateral"
        )
        ctrl.ui.axial_model_path.text.return_value = (
            "/workspace/artifacts/step3/refinement/models/2026-05-21-test2/axial"
        )
        ctrl._refresh_model_status()
        # Should extract the user-chosen bundle name "2026-05-21-test2",
        # not the bare bucket filename "lateral"/"axial".
        ctrl.ui.status_lateral_model_value.setText.assert_called_with(
            "2026-05-21-test2"
        )
        ctrl.ui.status_axial_model_value.setText.assert_called_with(
            "2026-05-21-test2"
        )

    def test_step2_round_path_falls_back_to_filename(self):
        """For Step 2 HITL round outputs (parent dir = 'round_001'),
        the filename itself is the meaningful identifier (e.g.
        'cyto3_lateral_round_001') so don't substitute parent dir."""
        ctrl = _make_bare_controller_with_widgets()
        _bind_real(ctrl, "_refresh_model_status")
        ctrl.ui.lateral_model_path.text.return_value = (
            "/workspace/models/round_001/cyto3_lateral_round_001"
        )
        ctrl._refresh_model_status()
        ctrl.ui.status_lateral_model_value.setText.assert_called_with(
            "cyto3_lateral_round_001"
        )

    def test_descriptive_filename_kept_as_is(self):
        """Non-generic filenames (e.g. user's custom model that ended
        up at `.../mymodel_lateral.pth`) should be shown verbatim,
        not substituted with parent dir."""
        ctrl = _make_bare_controller_with_widgets()
        _bind_real(ctrl, "_refresh_model_status")
        ctrl.ui.lateral_model_path.text.return_value = (
            "/some/dir/my_custom_lateral.pth"
        )
        ctrl._refresh_model_status()
        ctrl.ui.status_lateral_model_value.setText.assert_called_with(
            "my_custom_lateral.pth"
        )


# ---------------------------------------------------------------------------
# _restore_persisted_model_selection
# ---------------------------------------------------------------------------


class TestDiscoverWorkspaceModelsFilter:
    """Regression coverage for the 2026-05-18 .npy-loss-file bug:
    auto-discovery was matching `*_train_loss.npy` / `*_val_loss.npy`
    (Cellpose's loss-array companions to a saved model) instead of
    the actual model file.

    Filter must:
      - exclude any file with 'loss' / 'stat' / 'diam' / '.log' /
        '.json' in the name (companion artifacts)
      - accept files with an orientation keyword (lateral / axial /
        xy / xz / yz) — that's the model-for-orientation signal
      - work for Step 2 HITL `round_N/` AND Step 3 `refinement/`
    """

    def _make_workspace_with_models(self, tmp_path: Path,
                                    refinement_files=None,
                                    round_files=None):
        """Build a fake workspace dir with the requested model + loss
        files in their expected subdirectories.  Returns the workspace
        Path.  Files are created with a tiny content so they exist on
        disk and have a valid mtime.
        """
        # Post-2026-05-20 cutover:
        # * Step 2 HITL output stays under {ws}/models/round_*/
        # * Step 3 refinement bundles moved to
        #   {ws}/artifacts/step3/refinement/models/{name}/
        if refinement_files:
            refine = (
                tmp_path / "artifacts" / "step3" / "refinement"
                / "models" / "test_run"
            )
            refine.mkdir(parents=True)
            import time
            for i, name in enumerate(refinement_files):
                p = refine / name
                p.write_bytes(b"\x00")
                # Tick mtimes so the sort order is deterministic
                time.sleep(0.005)
        if round_files:
            rd = tmp_path / "models" / "round_001"
            rd.mkdir(parents=True)
            for name in round_files:
                p = rd / name
                p.write_bytes(b"\x00")
        return tmp_path

    def _make_controller_for_discovery(self, working_dir):
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        ctrl = FusionReconstructionController.__new__(FusionReconstructionController)
        ctrl.working_dir = str(working_dir)
        ctrl._log = []
        ctrl._append_log = lambda msg: ctrl._log.append(msg)
        # Path / status widgets — capture .setText calls.
        lateral_path = MagicMock()
        lateral_path.text = MagicMock(return_value="")
        axial_path = MagicMock()
        axial_path.text = MagicMock(return_value="")
        ctrl.ui = types.SimpleNamespace(
            lateral_model_path=lateral_path,
            axial_model_path=axial_path,
            status_lateral_model_value=MagicMock(),
            status_axial_model_value=MagicMock(),
        )
        # Bind real methods.
        _bind_real(ctrl, "_discover_workspace_models",
                   "_refresh_model_status")
        return ctrl

    def test_loss_companion_files_excluded(self, tmp_path):
        """`lateral_model_train_loss.npy` etc. must NOT be selected as
        the lateral model — `lateral_model` (the real file, no ext)
        must win."""
        ws = self._make_workspace_with_models(
            tmp_path,
            refinement_files=[
                "lateral_model",  # actual model — should be picked
                "lateral_model_train_loss.npy",  # companion — skip
                "lateral_model_val_loss.npy",    # companion — skip
                "axial_model",
                "axial_model_train_loss.npy",
                "axial_model_val_loss.npy",
            ],
        )
        ctrl = self._make_controller_for_discovery(ws)
        ctrl._discover_workspace_models()
        # lateral path widget should have been set to the actual model
        # file path, NOT to one of the loss arrays.  Check basename
        # because pytest's tmp_path may contain "loss" in the temp
        # dir name (test name gets embedded in the path).
        calls = ctrl.ui.lateral_model_path.setText.call_args_list
        assert len(calls) == 1
        lat_set_value = calls[0][0][0]
        lat_basename = Path(lat_set_value).name
        assert lat_basename == "lateral_model", (
            f"Expected lateral_model selected, got {lat_basename!r}"
        )

    def test_step2_round_naming_accepted(self, tmp_path):
        """Step 2 HITL `round_N/` files don't necessarily have 'model'
        in the filename — `cyto3_lateral_round_001` should still be
        recognized via the orientation keyword."""
        ws = self._make_workspace_with_models(
            tmp_path,
            round_files=[
                "cyto3_lateral_round_001",
                "cyto3_lateral_round_001_train_loss.npy",
                "cyto3_axial_round_001",
                "cyto3_axial_round_001_train_loss.npy",
            ],
        )
        ctrl = self._make_controller_for_discovery(ws)
        ctrl._discover_workspace_models()
        lat_set = ctrl.ui.lateral_model_path.setText.call_args_list
        ax_set = ctrl.ui.axial_model_path.setText.call_args_list
        assert len(lat_set) == 1
        assert lat_set[0][0][0].endswith("cyto3_lateral_round_001")
        assert "loss" not in lat_set[0][0][0]
        assert len(ax_set) == 1
        assert ax_set[0][0][0].endswith("cyto3_axial_round_001")

    def test_no_orientation_keyword_excluded(self, tmp_path):
        """A model file in `refinement/` with no orientation keyword
        (e.g. `model.pth`) shouldn't be picked as either lateral or
        axial — we can't infer orientation.  Both widgets get set to
        empty string (the always-write-both clear-stale behavior)."""
        ws = self._make_workspace_with_models(
            tmp_path,
            refinement_files=["model.pth", "random_other_file"],
        )
        ctrl = self._make_controller_for_discovery(ws)
        ctrl._discover_workspace_models()
        # Both widgets set, but to empty string (not the model.pth path).
        lat_calls = ctrl.ui.lateral_model_path.setText.call_args_list
        ax_calls = ctrl.ui.axial_model_path.setText.call_args_list
        assert len(lat_calls) == 1
        assert lat_calls[0][0][0] == "", (
            f"Expected lateral cleared to empty, got {lat_calls[0][0][0]!r}"
        )
        assert len(ax_calls) == 1
        assert ax_calls[0][0][0] == ""

    def test_diam_stat_companions_excluded(self, tmp_path):
        """Cellpose can also write `*_diam.npy` and `*_stats.json`
        companions.  Filter excludes them too."""
        ws = self._make_workspace_with_models(
            tmp_path,
            refinement_files=[
                "lateral_model",
                "lateral_model_diam.npy",
                "lateral_model_stats.json",
                "lateral_model_train.log",
            ],
        )
        ctrl = self._make_controller_for_discovery(ws)
        ctrl._discover_workspace_models()
        lat = ctrl.ui.lateral_model_path.setText.call_args_list[0][0][0]
        # Check basename only — tmp_path may contain "diam" etc. in
        # its temp dir name.
        lat_basename = Path(lat).name
        assert lat_basename == "lateral_model"


class TestWorkspaceSwitchClearsStale:
    """When switching workspaces, the new workspace may define only
    one orientation's model (e.g. lateral trained, axial not yet).
    The slot that the new workspace doesn't define MUST be cleared,
    not retain the previous workspace's stale value."""

    def test_clear_model_widgets_clears_both(self):
        """Direct test of the helper."""
        ctrl = _make_bare_controller_with_widgets()
        _bind_real(ctrl, "_clear_model_widgets", "_refresh_model_status")
        ctrl._clear_model_widgets()
        # Both widgets should be set to empty string.
        ctrl.ui.lateral_model_path.setText.assert_called_with("")
        ctrl.ui.axial_model_path.setText.assert_called_with("")

    def test_discovery_writes_both_slots_even_when_one_empty(self, tmp_path):
        """Discovery finds only a lateral model — axial must reset
        to empty (not retain stale)."""
        models_dir = tmp_path / "artifacts" / "step3" / "refinement" / "models" / "new_workspace_run"
        models_dir.mkdir(parents=True)
        (models_dir / "lateral_model").write_bytes(b"\x00")
        # NOTE: no axial_model in new workspace.

        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        ctrl = FusionReconstructionController.__new__(FusionReconstructionController)
        ctrl.working_dir = str(tmp_path)
        ctrl._log = []
        ctrl._append_log = lambda msg: ctrl._log.append(msg)
        lateral_path = MagicMock()
        lateral_path.text = MagicMock(return_value="")
        axial_path = MagicMock()
        axial_path.text = MagicMock(return_value="")
        ctrl.ui = types.SimpleNamespace(
            lateral_model_path=lateral_path,
            axial_model_path=axial_path,
            status_lateral_model_value=MagicMock(),
            status_axial_model_value=MagicMock(),
        )
        _bind_real(ctrl, "_discover_workspace_models",
                   "_refresh_model_status")
        ctrl._discover_workspace_models()
        # Lateral should be set to the found model.
        lat_calls = ctrl.ui.lateral_model_path.setText.call_args_list
        ax_calls = ctrl.ui.axial_model_path.setText.call_args_list
        assert len(lat_calls) == 1
        assert Path(lat_calls[0][0][0]).name == "lateral_model"
        # Axial should be set to empty string (was NOT skipped) so
        # any stale value from a previous workspace would be cleared.
        assert len(ax_calls) == 1
        assert ax_calls[0][0][0] == ""

    def test_restore_writes_both_slots_when_persisted_only_has_one(self, tmp_path):
        """Restore from persisted state where axial is None — axial
        widget must be set to empty, not skipped."""
        sm = ReconstructionStateManager()
        sm.load(str(tmp_path))
        # Pre-create the lateral model file so the existence check
        # passes.
        fake_lat = tmp_path / "fake_lat.npy"
        fake_lat.write_bytes(b"\x00")
        sm.set_model_selection(
            mode="import", lateral_path=str(fake_lat),
            # No axial_path — defaults to None in persistence.
        )
        ctrl = _make_bare_controller_with_widgets()
        ctrl.state_manager = sm
        _bind_real(ctrl, "_restore_persisted_model_selection",
                   "_refresh_model_status")
        result = ctrl._restore_persisted_model_selection()
        assert result is True
        lat_calls = ctrl.ui.lateral_model_path.setText.call_args_list
        ax_calls = ctrl.ui.axial_model_path.setText.call_args_list
        # Lateral set to the persisted path.
        assert len(lat_calls) == 1
        assert lat_calls[0][0][0] == str(fake_lat)
        # Axial set to empty (was None in persistence — must clear,
        # not skip).
        assert len(ax_calls) == 1
        assert ax_calls[0][0][0] == ""


class TestRestorePersistedModelSelection:

    def test_no_state_manager_returns_false(self):
        ctrl = _make_bare_controller_with_widgets()
        _bind_real(ctrl, "_restore_persisted_model_selection")
        # No state_manager attribute at all.
        assert ctrl._restore_persisted_model_selection() is False

    def test_empty_persisted_returns_false(self, tmp_path):
        sm = ReconstructionStateManager()
        sm.load(str(tmp_path))
        ctrl = _make_bare_controller_with_widgets()
        ctrl.state_manager = sm
        _bind_real(ctrl, "_restore_persisted_model_selection",
                   "_refresh_model_status")
        # Nothing persisted yet — should return False so caller falls
        # back to discovery.
        assert ctrl._restore_persisted_model_selection() is False

    def test_pretrained_persisted_restores(self, tmp_path):
        sm = ReconstructionStateManager()
        sm.load(str(tmp_path))
        sm.set_model_selection(
            mode="pretrained", lateral_path="cyto3", axial_path="cyto3",
        )
        ctrl = _make_bare_controller_with_widgets()
        ctrl.state_manager = sm
        _bind_real(ctrl, "_restore_persisted_model_selection",
                   "_refresh_model_status")
        result = ctrl._restore_persisted_model_selection()
        assert result is True
        ctrl.ui.lateral_model_path.setText.assert_called_with("cyto3")
        ctrl.ui.axial_model_path.setText.assert_called_with("cyto3")

    def test_path_persisted_but_file_missing_returns_false(self, tmp_path):
        """If persisted path doesn't exist on disk anymore, treat
        state as stale and fall back to discovery."""
        sm = ReconstructionStateManager()
        sm.load(str(tmp_path))
        sm.set_model_selection(
            mode="import",
            lateral_path="/nonexistent/missing.npy",
        )
        ctrl = _make_bare_controller_with_widgets()
        ctrl.state_manager = sm
        _bind_real(ctrl, "_restore_persisted_model_selection",
                   "_refresh_model_status")
        result = ctrl._restore_persisted_model_selection()
        assert result is False

    def test_path_persisted_and_file_exists_restores(self, tmp_path):
        sm = ReconstructionStateManager()
        sm.load(str(tmp_path))
        # Create a real file so the existence check passes.
        fake_model = tmp_path / "lat_model.npy"
        fake_model.write_bytes(b"\x00")
        sm.set_model_selection(
            mode="import", lateral_path=str(fake_model),
        )
        ctrl = _make_bare_controller_with_widgets()
        ctrl.state_manager = sm
        _bind_real(ctrl, "_restore_persisted_model_selection",
                   "_refresh_model_status")
        result = ctrl._restore_persisted_model_selection()
        assert result is True
        ctrl.ui.lateral_model_path.setText.assert_called_with(str(fake_model))
