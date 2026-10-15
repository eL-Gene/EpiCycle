"""WP1 + WP5: Real-path tests for persistence gate logic.

These tests exercise the actual ``_save_curation_checked``, ``_save_curation``,
and ``_save_tiers_csv`` methods on a minimal controller-like object that mirrors
production attribute layout, ensuring dirty flags are only cleared when writes
actually succeed.
"""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Minimal controller harness — uses real methods, fake infrastructure
# ---------------------------------------------------------------------------

class _ControllerHarness:
    """Wraps actual controller save methods with minimal fake state.

    We import and bind the real unbound methods from the controller module
    so that we test actual production logic, not reimplemented stubs.
    """

    def __init__(self, tmp_path):
        self.working_dir = str(tmp_path)
        self._curation_dirty = False
        self._tiers_dirty = False
        self._edited_patches = set()
        self._qc_features_df = None
        self._curation_undo_stack = MagicMock()
        self._curation_undo_stack.__bool__ = MagicMock(return_value=False)
        self._log = []
        # A1: the thumbnail re-trigger after a labels write (not under test here)
        self._refresh_step3_thumbs_after_labels = lambda why: None

        # Fake UI
        self.ui = MagicMock()
        self.ui.save_curation_btn = MagicMock()

        # Fake viewer with labels layer
        self.viewer = MagicMock()
        self._labels_layer = None
        self._labels_path = None

        # Fake state manager
        self.state_manager = MagicMock()
        # D2 (perf push 2026-05-09): default to no-downsample so existing
        # save tests behave as before.  Tests that exercise the lift
        # path explicitly override these.
        self.state_manager.get_patch_downsample_factor = MagicMock(return_value=1)
        self.state_manager.get_patch_original_labels_shape = MagicMock(
            return_value=None
        )

    def _append_log(self, msg):
        self._log.append(msg)

    def _find_active_labels_layer(self):
        return self._labels_layer

    def _get_active_labels_path(self):
        return self._labels_path

    def _selected_curation_patch(self):
        return getattr(self, "_cur_patch", None)

    def _selected_patch_name(self):
        return getattr(self, "_patch_name", None)

    # _is_population_mode harness stub removed 2026-05-03
    # (drop-population-mode Commit 5/6) — production method gone.

    def _refresh_patch_list_colors(self):
        pass

    def _refresh_summary_dashboard(self):
        pass

    def _refresh_patch_state_list(self):
        pass

    def _refresh_curation_ui(self):
        pass


def _bind_real_methods(harness):
    """Import and bind real controller methods to the harness."""
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )

    # Bind real methods as bound methods on our harness
    import types

    for method_name in (
        "_save_curation",
        "_save_curation_checked",
        "_save_tiers_csv",
        # 2026-06-07: _save_curation went async.  The SYNCHRONOUS _save_curation
        # is now a thin orchestrator (prep → pure persist → finalize) that the
        # gate tests drive without a live QThread.  Bind the prep + finalize
        # helpers it delegates to.
        "_save_curation_prepare",
        "_on_save_curation_finished",
        "_maybe_lift_for_save",  # D2 helper used by save sites
        # 2026-05-17 fix: save_curation / save_tiers now call this
        # to refresh the save button based on both dirty flags.
        "_refresh_save_button_state",
        # 2026-05-17 diagnostic: save_tiers_csv calls this before
        # the disk write.  Read-only logging helper.
        "_scan_qc_df_for_duplicates",
        # 2026-06-05 (integration/step3-patch-perf): _save_curation_checked
        # now wraps the save in the generalized busy bar (sync-paint).  Bind
        # the self-contained helpers so the gate tests don't AttributeError.
        "_busy_show",
        "_busy_progress",
        "_busy_hide",
    ):
        real_method = getattr(FusionReconstructionController, method_name)
        harness.__dict__[method_name] = types.MethodType(real_method, harness)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def harness(tmp_path):
    h = _ControllerHarness(tmp_path)
    _bind_real_methods(h)
    return h


def _setup_labels(harness, tmp_path):
    """Set up a fake labels layer + path so mask writes succeed."""
    labels = np.zeros((4, 8, 8), dtype=np.uint32)
    labels[1, 2:5, 2:5] = 1
    layer = MagicMock()
    layer.data = labels
    harness._labels_layer = layer
    harness._labels_path = tmp_path / "labels_3d.tif"
    return layer


def _setup_csv(harness, tmp_path, patch_name="test_patch.tif"):
    """Set up a QC DataFrame + patch name so CSV writes succeed."""
    harness._qc_features_df = pd.DataFrame({
        "label": [1, 2],
        "area": [100, 200],
        "qc_tier": ["good", "review"],
        "feature_state": ["fresh", "fresh"],
    })
    harness._cur_patch = patch_name
    harness._patch_name = patch_name
    # Ensure qc directory exists
    (tmp_path / "artifacts" / "step3" / "qc").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Tests: WP1 — _save_curation only clears dirty when writes succeed
# ---------------------------------------------------------------------------

class TestSaveCurationConditionalClear:
    """_save_curation must NOT clear _curation_dirty if writes are skipped."""

    def test_no_layer_no_clear(self, harness, tmp_path):
        """No labels layer → mask write skipped → dirty flag stays True."""
        _setup_csv(harness, tmp_path)
        harness._labels_layer = None
        harness._labels_path = None
        harness._curation_dirty = True

        harness._save_curation()

        assert harness._curation_dirty is True

    def test_no_patch_name_mask_only_clears(self, harness, tmp_path):
        """No patch name → CSV skipped → mask-only save succeeds → dirty cleared."""
        _setup_labels(harness, tmp_path)
        harness._cur_patch = None
        harness._patch_name = None
        harness._qc_features_df = None
        harness._curation_dirty = True

        harness._save_curation()

        assert harness._curation_dirty is False

    def test_both_present_clears(self, harness, tmp_path):
        """Both mask + CSV succeed → dirty flag cleared."""
        _setup_labels(harness, tmp_path)
        _setup_csv(harness, tmp_path)
        harness._curation_dirty = True

        harness._save_curation()

        assert harness._curation_dirty is False
        assert "test_patch" in str(harness._edited_patches)

    def test_mask_file_actually_written(self, harness, tmp_path):
        """Verify the mask TIFF is actually written to disk."""
        _setup_labels(harness, tmp_path)
        _setup_csv(harness, tmp_path)
        harness._curation_dirty = True

        harness._save_curation()

        assert harness._labels_path.exists()

    def test_csv_file_actually_written(self, harness, tmp_path):
        """Verify the CSV is actually written to disk."""
        _setup_labels(harness, tmp_path)
        _setup_csv(harness, tmp_path)
        harness._curation_dirty = True

        harness._save_curation()

        csv_path = tmp_path / "artifacts" / "step3" / "qc" / "test_patch_features.csv"
        assert csv_path.exists()
        loaded = pd.read_csv(csv_path)
        assert len(loaded) == 2


# ---------------------------------------------------------------------------
# Tests: WP1 — _save_curation_checked gates on both dirty flags
# ---------------------------------------------------------------------------

class TestSaveCurationCheckedBothFlags:
    """_save_curation_checked must block on EITHER dirty flag."""

    def test_clean_state_returns_true(self, harness):
        harness._curation_dirty = False
        harness._tiers_dirty = False
        assert harness._save_curation_checked() is True

    @patch("napari_epicycle.fusion_reconstruction.controller.QMessageBox")
    def test_tiers_dirty_only_blocks_without_csv_target(self, mock_mb, harness, tmp_path):
        """Tier dirty + no patch name → save fails → returns False."""
        harness._curation_dirty = False
        harness._tiers_dirty = True
        harness._cur_patch = None
        harness._patch_name = None

        result = harness._save_curation_checked()

        # _save_tiers_csv won't clear the flag without a patch target
        assert result is False
        mock_mb.critical.assert_called_once()

    def test_tiers_dirty_clears_with_valid_target(self, harness, tmp_path):
        """Tier dirty + valid patch → CSV written → returns True."""
        _setup_csv(harness, tmp_path)
        harness._curation_dirty = False
        harness._tiers_dirty = True

        result = harness._save_curation_checked()

        assert result is True
        assert harness._tiers_dirty is False

    def test_both_dirty_needs_both_writes(self, harness, tmp_path):
        """Both dirty + both targets present → both clear → True."""
        _setup_labels(harness, tmp_path)
        _setup_csv(harness, tmp_path)
        harness._curation_dirty = True
        harness._tiers_dirty = True

        result = harness._save_curation_checked()

        assert result is True
        assert harness._curation_dirty is False
        assert harness._tiers_dirty is False

    @patch("napari_epicycle.fusion_reconstruction.controller.QMessageBox")
    def test_curation_dirty_no_layer_blocks(self, mock_mb, harness, tmp_path):
        """Curation dirty + no labels layer → save incomplete → False."""
        _setup_csv(harness, tmp_path)
        harness._labels_layer = None
        harness._labels_path = None
        harness._curation_dirty = True
        harness._tiers_dirty = False

        result = harness._save_curation_checked()

        assert result is False
        assert harness._curation_dirty is True
        mock_mb.critical.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: R2-WP1 — Population + dirty tiers must block status transition
# ---------------------------------------------------------------------------

# TestPopulationDirtyTiersBlocked class removed 2026-05-03
# (drop-population-mode Commit 3/6).  All 4 tests asserted population-
# mode behaviour that no longer exists.


# ---------------------------------------------------------------------------
# Tests: R2-WP2 — _save_tiers_csv scope safety
# ---------------------------------------------------------------------------

class TestSaveTiersCsvScopeSafe:
    """_save_tiers_csv must never write mixed-patch rows to one CSV."""

    # test_population_mode_does_not_write removed 2026-05-03
    # (drop-population-mode Commit 3/6).

    # test_mixed_patch_names_does_not_write removed 2026-05-03
    # (drop-population-mode Commit 4/6) — population merged DataFrame
    # is gone; patch_name column never appears so this guard no longer
    # exists.
    # test_mismatched_patch_stem_does_not_write removed for the same
    # reason.
    # test_matching_patch_stem_writes removed because it asserted
    # behaviour around the now-deleted patch_name column path.

    def test_single_patch_writes_normally(self, harness, tmp_path):
        """Single-patch DataFrame writes and clears dirty flag."""
        _setup_csv(harness, tmp_path)
        harness._tiers_dirty = True

        harness._save_tiers_csv()

        assert harness._tiers_dirty is False
        csv_path = tmp_path / "artifacts" / "step3" / "qc" / "test_patch_features.csv"
        assert csv_path.exists()
