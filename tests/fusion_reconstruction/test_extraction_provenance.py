"""Tests for extraction-provenance helpers added 2026-05-19.

Covers:
  - FusionReconstructionController._load_extraction_manifest
  - FusionReconstructionController._check_extraction_staleness
  - FusionReconstructionController._check_extraction_mixed_settings

These three helpers underlie the per-patch badge rendering (pair count,
⏱ staleness marker, ⚠ mixed-settings marker) and gate the staleness +
mixed-settings dialogs at training launch / extraction launch.

Tests use a minimal controller built via ``object.__new__`` so we
exercise the real methods without booting napari/Qt.  Manifest fixtures
are written to ``tmp_path`` and read back through the real on-disk path.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from napari_epicycle.fusion_reconstruction import _step3_mr_text as _MRT
from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_controller(working_dir, settings_override=None):
    """Build a minimal controller with no UI, no state manager.

    `settings_override` lets tests stub out `_read_extraction_settings`
    to control what "current UI knob values" the mixed-settings check
    compares against.
    """
    ctrl = object.__new__(FusionReconstructionController)
    ctrl.working_dir = str(working_dir)
    if settings_override is not None:
        ctrl._read_extraction_settings = lambda: dict(settings_override)
    return ctrl


def _write_manifest(workspace_dir, per_patch=None, extra=None):
    """Write a synthetic refinement manifest under workspace_dir.

    2026-05-20 audit fix (F1): writes to the canonical
    ``refinement/manifest.json`` (sibling of ``pairs/``) per
    the path helper.  Pre-fix the fixture wrote to
    ``refinement/pairs/extraction_manifest.json`` which the
    runtime code also (incorrectly) used — tests passed but
    fixtures contradicted the design + migration script.
    """
    from napari_epicycle.fusion_reconstruction import _paths
    workspace_dir = Path(workspace_dir)
    manifest_path = _paths.refinement_manifest_path(workspace_dir)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "1.1",
        "per_patch": per_patch or {},
    }
    if extra:
        manifest.update(extra)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path.parent


def _make_labels_file(path, mtime=None):
    """Create a small file at `path` (synthetic labels TIFF).

    If `mtime` is given, set the file's mtime to that timestamp.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"FAKE_TIFF_CONTENT")
    if mtime is not None:
        import os
        os.utime(p, (mtime, mtime))
    return p


# Default current-UI settings for the mixed-settings tests.  Match the
# shape that ``_read_extraction_settings`` produces.
_DEFAULT_CURRENT_SETTINGS = {
    "views": ("xy", "xz", "yz"),
    "min_labels_per_slice": 3,
    "edge_buffer": 50,
    "slices_per_patch": 0,
    "min_distance": 5,
    "remove_boundaries": True,
    "relabel": True,
    "min_size": 100,
    "smoothing_disk_size": 1,
    "expand_labels_distance": 0,
    "axes": "ZCYX",
}


# Matching per-patch snapshot in the manifest schema.
def _matching_snapshot():
    return {
        "extraction_settings": {
            "views": ["xy", "xz", "yz"],
            "min_labels_per_slice": 3,
            "edge_buffer": 50,
            "slices_per_patch": 0,
            "min_distance": 5,
            "axes": "ZCYX",
        },
        "mask_cleaning": {
            "remove_boundaries": True,
            "relabel": True,
            "min_size": 100,
            "smoothing_disk_size": 1,
            "expand_labels_distance": 0,
        },
    }


# ---------------------------------------------------------------------------
# _load_extraction_manifest
# ---------------------------------------------------------------------------


class TestLoadExtractionManifest:

    def test_returns_none_when_no_working_dir(self):
        ctrl = _make_controller(working_dir="")
        ctrl.working_dir = ""
        assert ctrl._load_extraction_manifest() is None

    def test_returns_none_when_no_manifest_file(self, tmp_path):
        ctrl = _make_controller(tmp_path)
        assert ctrl._load_extraction_manifest() is None

    def test_returns_parsed_dict_when_manifest_exists(self, tmp_path):
        _write_manifest(tmp_path, per_patch={"patchA": {"spatial": 5}})
        ctrl = _make_controller(tmp_path)
        manifest = ctrl._load_extraction_manifest()
        assert manifest is not None
        assert manifest["per_patch"]["patchA"]["spatial"] == 5

    def test_returns_none_on_corrupt_json(self, tmp_path):
        from napari_epicycle.fusion_reconstruction import _paths
        manifest_path = _paths.refinement_manifest_path(tmp_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text("{not valid json", encoding="utf-8")
        ctrl = _make_controller(tmp_path)
        assert ctrl._load_extraction_manifest() is None


# ---------------------------------------------------------------------------
# _check_extraction_staleness
# ---------------------------------------------------------------------------


class TestCheckExtractionStaleness:

    def test_empty_when_no_manifest(self, tmp_path):
        ctrl = _make_controller(tmp_path)
        assert ctrl._check_extraction_staleness() == set()

    def test_empty_when_source_unchanged(self, tmp_path):
        labels_path = _make_labels_file(
            tmp_path / "labels" / "patchA.tif", mtime=1_700_000_000.0,
        )
        _write_manifest(tmp_path, per_patch={
            "patchA": {
                "source_path": str(labels_path),
                "source_mtime": 1_700_000_000.0,
            },
        })
        ctrl = _make_controller(tmp_path)
        assert ctrl._check_extraction_staleness() == set()

    def test_returns_stem_when_source_newer(self, tmp_path):
        # File mtime newer than recorded mtime
        labels_path = _make_labels_file(
            tmp_path / "labels" / "patchA.tif",
            mtime=1_700_000_100.0,  # 100s later
        )
        _write_manifest(tmp_path, per_patch={
            "patchA": {
                "source_path": str(labels_path),
                "source_mtime": 1_700_000_000.0,
            },
        })
        ctrl = _make_controller(tmp_path)
        assert ctrl._check_extraction_staleness() == {"patchA"}

    def test_one_second_jitter_slack(self, tmp_path):
        """A 0.5-second mtime drift is not considered stale (FS jitter)."""
        labels_path = _make_labels_file(
            tmp_path / "labels" / "patchA.tif",
            mtime=1_700_000_000.5,
        )
        _write_manifest(tmp_path, per_patch={
            "patchA": {
                "source_path": str(labels_path),
                "source_mtime": 1_700_000_000.0,
            },
        })
        ctrl = _make_controller(tmp_path)
        assert ctrl._check_extraction_staleness() == set()

    def test_skips_entry_missing_source_path(self, tmp_path):
        _write_manifest(tmp_path, per_patch={
            "patchA": {"source_mtime": 1_700_000_000.0},  # no source_path
        })
        ctrl = _make_controller(tmp_path)
        assert ctrl._check_extraction_staleness() == set()

    def test_skips_entry_missing_source_mtime(self, tmp_path):
        labels_path = _make_labels_file(tmp_path / "labels" / "patchA.tif")
        _write_manifest(tmp_path, per_patch={
            "patchA": {"source_path": str(labels_path)},  # no mtime
        })
        ctrl = _make_controller(tmp_path)
        assert ctrl._check_extraction_staleness() == set()

    def test_skips_entry_with_missing_source_file(self, tmp_path):
        _write_manifest(tmp_path, per_patch={
            "patchA": {
                "source_path": str(tmp_path / "nonexistent.tif"),
                "source_mtime": 1_700_000_000.0,
            },
        })
        ctrl = _make_controller(tmp_path)
        # File doesn't exist → can't compare → not in stale set
        assert ctrl._check_extraction_staleness() == set()

    def test_mixed_stale_and_fresh(self, tmp_path):
        fresh_path = _make_labels_file(
            tmp_path / "labels" / "fresh.tif", mtime=1_700_000_000.0,
        )
        stale_path = _make_labels_file(
            tmp_path / "labels" / "stale.tif",
            mtime=1_700_000_500.0,  # newer than recorded
        )
        _write_manifest(tmp_path, per_patch={
            "fresh": {
                "source_path": str(fresh_path),
                "source_mtime": 1_700_000_000.0,
            },
            "stale": {
                "source_path": str(stale_path),
                "source_mtime": 1_700_000_000.0,
            },
        })
        ctrl = _make_controller(tmp_path)
        assert ctrl._check_extraction_staleness() == {"stale"}


# ---------------------------------------------------------------------------
# _check_extraction_mixed_settings
# ---------------------------------------------------------------------------


class TestCheckExtractionMixedSettings:

    def test_empty_when_no_manifest(self, tmp_path):
        ctrl = _make_controller(
            tmp_path, settings_override=_DEFAULT_CURRENT_SETTINGS,
        )
        assert ctrl._check_extraction_mixed_settings() == set()

    def test_empty_when_all_settings_match(self, tmp_path):
        _write_manifest(tmp_path, per_patch={
            "patchA": _matching_snapshot(),
        })
        ctrl = _make_controller(
            tmp_path, settings_override=_DEFAULT_CURRENT_SETTINGS,
        )
        assert ctrl._check_extraction_mixed_settings() == set()

    def test_returns_stem_when_mask_cleaning_differs(self, tmp_path):
        snap = _matching_snapshot()
        snap["mask_cleaning"]["smoothing_disk_size"] = 3  # was 1 in current
        _write_manifest(tmp_path, per_patch={"patchA": snap})
        ctrl = _make_controller(
            tmp_path, settings_override=_DEFAULT_CURRENT_SETTINGS,
        )
        assert ctrl._check_extraction_mixed_settings() == {"patchA"}

    def test_returns_stem_when_extraction_settings_differs(self, tmp_path):
        snap = _matching_snapshot()
        snap["extraction_settings"]["edge_buffer"] = 100  # was 50 in current
        _write_manifest(tmp_path, per_patch={"patchA": snap})
        ctrl = _make_controller(
            tmp_path, settings_override=_DEFAULT_CURRENT_SETTINGS,
        )
        assert ctrl._check_extraction_mixed_settings() == {"patchA"}

    def test_returns_stem_when_views_differs(self, tmp_path):
        snap = _matching_snapshot()
        snap["extraction_settings"]["views"] = ["xy"]  # was all 3 in current
        _write_manifest(tmp_path, per_patch={"patchA": snap})
        ctrl = _make_controller(
            tmp_path, settings_override=_DEFAULT_CURRENT_SETTINGS,
        )
        assert ctrl._check_extraction_mixed_settings() == {"patchA"}

    def test_skips_legacy_entry_without_provenance(self, tmp_path):
        """Schema 1.0 entries (no extraction_settings/mask_cleaning) skip."""
        _write_manifest(tmp_path, per_patch={
            "patchA": {"spatial": 5, "axial": 3},  # legacy, no provenance
        })
        ctrl = _make_controller(
            tmp_path, settings_override=_DEFAULT_CURRENT_SETTINGS,
        )
        assert ctrl._check_extraction_mixed_settings() == set()

    def test_mixed_matching_and_differing(self, tmp_path):
        _write_manifest(tmp_path, per_patch={
            "patchA": _matching_snapshot(),
            "patchB": {
                **_matching_snapshot(),
                "mask_cleaning": {
                    **_matching_snapshot()["mask_cleaning"],
                    "smoothing_disk_size": 5,
                },
            },
        })
        ctrl = _make_controller(
            tmp_path, settings_override=_DEFAULT_CURRENT_SETTINGS,
        )
        assert ctrl._check_extraction_mixed_settings() == {"patchB"}

    def test_legacy_manifest_without_axes_field_not_flagged(self, tmp_path):
        """2026-05-21 audit regression guard: a manifest written
        BEFORE `axes` was added to extraction_settings should NOT be
        flagged as mixed merely because the new current_extraction
        dict has `axes` and the recorded dict doesn't.  The pre-fix
        comparison would naively dict-equal-compare and flag every
        pre-fix patch.  The normalization layer treats missing `axes`
        as the default "ZCYX" so old manifests compare cleanly.
        """
        snap = _matching_snapshot()
        # Simulate a pre-2026-05-21 manifest: no `axes` field anywhere.
        snap["extraction_settings"].pop("axes", None)
        _write_manifest(tmp_path, per_patch={"patchA": snap})
        ctrl = _make_controller(
            tmp_path, settings_override=_DEFAULT_CURRENT_SETTINGS,
        )
        # _DEFAULT_CURRENT_SETTINGS now has axes="ZCYX", and the
        # legacy snapshot is missing the field.  Without the fix,
        # this would return {"patchA"}.  With the fix, it returns
        # an empty set (legacy treated as default ZCYX).
        assert ctrl._check_extraction_mixed_settings() == set()

    def test_axes_difference_no_longer_flagged(self, tmp_path):
        """2026-06-07: axes is auto-detected per patch (resolve_patch_axes),
        not a user setting, so an axes-ONLY difference between recorded and
        current must NOT flag the patch as mixed.  Pre-fix this WAS flagged;
        excluding axes from the drift comparison prevents false 'mixed'
        warnings on workspaces extracted under the old global ZCYX combo
        (and on any mixed-layout folder, which is now correct, not 'mixed')."""
        snap = _matching_snapshot()
        # Recorded snapshot has ZCYX explicitly; everything else matches.
        snap["extraction_settings"]["axes"] = "ZCYX"
        _write_manifest(tmp_path, per_patch={"patchA": snap})
        # Current UI would say CZYX — axes differs, all real settings match.
        current = dict(_DEFAULT_CURRENT_SETTINGS)
        current["axes"] = "CZYX"
        ctrl = _make_controller(tmp_path, settings_override=current)
        assert ctrl._check_extraction_mixed_settings() == set()


# ---------------------------------------------------------------------------
# Staleness gate at training launch
# ---------------------------------------------------------------------------


def _make_controller_for_launch_test(tmp_path, *,
                                     stale_stems=frozenset(),
                                     dialog_decision="cancel"):
    """Build a controller wired up for `_launch_refinement` gate tests.

    - Creates the training_data dir so the precondition passes.
    - Stubs `_check_extraction_staleness` to return `stale_stems`.
    - Stubs `_show_extraction_staleness_dialog` to return `dialog_decision`.
    - Stubs `_launch_next_refinement` + `_extract_training_pairs_for_stems`
      as recorders so we can assert which branch fired.
    """
    ctrl = object.__new__(FusionReconstructionController)
    ctrl.working_dir = str(tmp_path)
    (Path(tmp_path) / "artifacts" / "step3" / "refinement" / "pairs").mkdir(
        parents=True, exist_ok=True,
    )

    # Recorders for the two action branches.
    ctrl._launch_next_recorded = False
    ctrl._extract_for_stems_recorded = None

    ctrl._check_extraction_staleness = lambda: set(stale_stems)
    ctrl._show_extraction_staleness_dialog = (
        lambda stems: dialog_decision
    )

    def _fake_launch_next():
        ctrl._launch_next_recorded = True
    ctrl._launch_next_refinement = _fake_launch_next

    def _fake_extract_for_stems(stems):
        ctrl._extract_for_stems_recorded = list(stems)
    ctrl._extract_training_pairs_for_stems = _fake_extract_for_stems

    # Minimal UI shim — `_launch_refinement` reads
    # `self.ui.refinement_mode_combo.currentText()` and (to choose the
    # run's bundle name once) `self.ui.refine_model_name.text()` only
    # when we get past the staleness gate.
    class _FakeCombo:
        def currentText(self):
            return "Fine-tune Lateral"
    class _FakeLineEdit:
        def text(self):
            return "gate_test_bundle"
    class _FakeUI:
        refinement_mode_combo = _FakeCombo()
        refine_model_name = _FakeLineEdit()
    ctrl.ui = _FakeUI()

    ctrl._refinement_queue = []
    ctrl._log_lines = []
    ctrl._append_log = lambda text: ctrl._log_lines.append(text)
    # Brick A's launch snapshot reads the whole panel; this gate test only
    # cares about the staleness branch, so the snapshot is a plain dict.
    ctrl._take_refinement_launch_snapshot = lambda mode: {
        "mode": mode, "buckets": ["lateral"], "name": None,
        "inits": {"lateral": ("scratch", None)},
        "init_combo_text": "From scratch", "patches": [],
        "config": {}, "split": {},
    }
    ctrl._check_refinement_collision = lambda mode: "no_collision"

    return ctrl


class TestLaunchRefinementStalenessGate:

    def test_no_stale_patches_proceeds_normally(self, tmp_path):
        ctrl = _make_controller_for_launch_test(tmp_path)
        ctrl._launch_refinement()
        assert ctrl._launch_next_recorded is True
        assert ctrl._extract_for_stems_recorded is None

    def test_stale_cancel_aborts(self, tmp_path):
        ctrl = _make_controller_for_launch_test(
            tmp_path, stale_stems={"patchA"}, dialog_decision="cancel",
        )
        ctrl._launch_refinement()
        assert ctrl._launch_next_recorded is False
        assert ctrl._extract_for_stems_recorded is None
        assert any("cancelled by user" in line for line in ctrl._log_lines)

    def test_stale_train_anyway_proceeds_with_warning(self, tmp_path):
        ctrl = _make_controller_for_launch_test(
            tmp_path, stale_stems={"patchA", "patchB"},
            dialog_decision="train_anyway",
        )
        ctrl._launch_refinement()
        assert ctrl._launch_next_recorded is True
        assert ctrl._extract_for_stems_recorded is None
        assert any(
            "[WARN]" in line and "stale" in line
            for line in ctrl._log_lines
        )

    def test_stale_reextract_kicks_off_extraction(self, tmp_path):
        ctrl = _make_controller_for_launch_test(
            tmp_path, stale_stems={"patchA", "patchB"},
            dialog_decision="reextract",
        )
        ctrl._launch_refinement()
        assert ctrl._launch_next_recorded is False  # NO training yet
        assert ctrl._extract_for_stems_recorded == ["patchA", "patchB"]

    def test_missing_training_dir_short_circuits_before_gate(self, tmp_path):
        """When no training_data dir, error logs but staleness check
        is skipped (no dialog spawn)."""
        ctrl = object.__new__(FusionReconstructionController)
        ctrl.working_dir = str(tmp_path)
        ctrl._log_lines = []
        ctrl._append_log = lambda t: ctrl._log_lines.append(t)
        dialog_called = []
        ctrl._show_extraction_staleness_dialog = (
            lambda s: dialog_called.append(True) or "cancel"
        )
        ctrl._launch_refinement()
        assert any("Extract training pairs first" in line
                   for line in ctrl._log_lines)
        assert not dialog_called


# ---------------------------------------------------------------------------
# Mixed-settings gate at extraction launch
# ---------------------------------------------------------------------------


def _make_controller_for_extract_test(tmp_path, *,
                                      selected_stems=("patchA",),
                                      mixed_stems=frozenset(),
                                      dialog_decision="cancel"):
    """Controller fixture for `_extract_training_pairs` mixed-settings tests.

    Stubs out:
      - `refine_patch_list.selectedItems()` to return items whose
        `data(256)` produces `selected_stems`.
      - `_check_extraction_mixed_settings` → `mixed_stems`.
      - `_show_extraction_mixed_settings_dialog` → `dialog_decision`.
      - Spawns the worker only via a fake `_ExtractionWorker_spawned`
        flag so we don't kick off a real QThread.
    """
    ctrl = object.__new__(FusionReconstructionController)
    ctrl.working_dir = str(tmp_path)
    ctrl.loaded_stack_dir = str(tmp_path / "stacks")
    Path(ctrl.loaded_stack_dir).mkdir(parents=True, exist_ok=True)
    ctrl._log_lines = []
    ctrl._append_log = lambda t: ctrl._log_lines.append(t)

    # Fake selected items
    class _Item:
        def __init__(self, name):
            self._name = name
        def data(self, role):
            return self._name
    class _PatchList:
        def __init__(self, stems):
            self._items = [_Item(s) for s in stems]
        def selectedItems(self):
            return list(self._items)
        def count(self):
            return len(self._items)
    class _Label:
        def __init__(self):
            self.text = ""
            self.style = ""
        def setText(self, t):
            self.text = t
        def setStyleSheet(self, s):
            self.style = s
    class _Button:
        def __init__(self):
            self.enabled = True
            self.text = ""
        def setText(self, t):
            self.text = t
        def setEnabled(self, b):
            self.enabled = b
        def setIcon(self, icon):  # absorb apply_button_icon
            pass
        def setIconSize(self, size):
            pass
        @property
        def clicked(self):
            class _Sig:
                def disconnect(self_inner):
                    pass
                def connect(self_inner, f):
                    pass
            return _Sig()
    class _Progress:
        def setRange(self, lo, hi):
            pass
    class _FakeUI:
        pass
    ui = _FakeUI()
    ui.refine_patch_list = _PatchList(selected_stems)
    # the ui-side tick API (Brick B1 / B2): every listed stem is ticked
    ui.refine_ticked_names = lambda: [i._name for i in
                                      ui.refine_patch_list._items]
    ui.set_refine_ticks = lambda on: None
    ui.refine_row_count = lambda: ui.refine_patch_list.count()
    ui.refinement_status_label = _Label()
    ui.refine_extraction_status_label = _Label()
    ui.extract_training_pairs_btn = _Button()
    ui.launch_refinement_btn = _Button()
    ui.refine_extraction_progress = _Progress()
    ctrl.ui = ui

    ctrl._check_extraction_mixed_settings = lambda: set(mixed_stems)
    ctrl._show_extraction_mixed_settings_dialog = (
        lambda stems: dialog_decision
    )
    ctrl._read_extraction_settings = lambda: dict(_DEFAULT_CURRENT_SETTINGS)

    # Recorder: which stems were ultimately handed to the worker.
    ctrl._stems_passed_to_worker = None

    # Override the worker construction by monkey-patching the module.
    return ctrl


class TestExtractTrainingPairsMixedSettingsGate:

    def _patch_worker_spawn(self, ctrl, monkeypatch):
        """Capture stems handed to _ExtractionWorker without spawning.

        Fake worker + thread absorb all the wiring calls that the real
        Qt classes need (.run, .deleteLater, .quit, signal .connect).
        """
        import napari_epicycle.fusion_reconstruction.controller as mod

        class _NullSig:
            def connect(self, f):
                pass

        class _FakeWorker:
            def __init__(self, working_dir, patch_stems, stack_dir, kwargs,
                         run_id=None):
                ctrl._stems_passed_to_worker = list(patch_stems)
                ctrl._run_id_passed_to_worker = run_id
                # Cover every signal the controller wires up.
                self.progress = _NullSig()
                self.finished = _NullSig()
                self.error = _NullSig()
                self.cancelled = _NullSig()
                self.log = _NullSig()
            def moveToThread(self, t):
                pass
            def run(self):
                pass
            def deleteLater(self):
                pass

        class _FakeThread:
            def __init__(self):
                self.started = _NullSig()
                self.finished = _NullSig()
            def start(self):
                pass
            def quit(self):
                pass
            def deleteLater(self):
                pass

        monkeypatch.setattr(mod, "_ExtractionWorker", _FakeWorker)
        monkeypatch.setattr(mod, "QThread", _FakeThread)

    def test_no_mixed_proceeds_normally(self, tmp_path, monkeypatch):
        ctrl = _make_controller_for_extract_test(
            tmp_path, selected_stems=("patchA",), mixed_stems=set(),
        )
        self._patch_worker_spawn(ctrl, monkeypatch)
        ctrl._extract_training_pairs()
        assert ctrl._stems_passed_to_worker == ["patchA"]

    def test_mixed_cancel_aborts(self, tmp_path, monkeypatch):
        ctrl = _make_controller_for_extract_test(
            tmp_path, selected_stems=("patchA",),
            mixed_stems={"patchB"}, dialog_decision="cancel",
        )
        self._patch_worker_spawn(ctrl, monkeypatch)
        ctrl._extract_training_pairs()
        assert ctrl._stems_passed_to_worker is None
        assert any("[INFO]" in line and "cancelled" in line
                   for line in ctrl._log_lines)

    def test_mixed_keep_mixed_proceeds_with_selection_only(
        self, tmp_path, monkeypatch,
    ):
        ctrl = _make_controller_for_extract_test(
            tmp_path, selected_stems=("patchA",),
            mixed_stems={"patchB"}, dialog_decision="secondary",
        )
        self._patch_worker_spawn(ctrl, monkeypatch)
        ctrl._extract_training_pairs()
        assert ctrl._stems_passed_to_worker == ["patchA"]
        assert any("[WARN]" in line and "Mixed extraction settings" in line
                   for line in ctrl._log_lines)

    def test_mixed_reextract_all_folds_in_mixed_stems(
        self, tmp_path, monkeypatch,
    ):
        ctrl = _make_controller_for_extract_test(
            tmp_path, selected_stems=("patchA",),
            mixed_stems={"patchB", "patchC"}, dialog_decision="primary",
        )
        self._patch_worker_spawn(ctrl, monkeypatch)
        ctrl._extract_training_pairs()
        assert ctrl._stems_passed_to_worker == ["patchA", "patchB", "patchC"]
        assert any(
            "Expanding extraction" in line
            for line in ctrl._log_lines
        )

    def test_mixed_stems_in_selection_dont_trigger_dialog(
        self, tmp_path, monkeypatch,
    ):
        """If the only mixed stems are ALREADY in the selection, no
        dialog should fire — they're about to be re-extracted with
        current settings anyway."""
        dialog_called = []
        ctrl = _make_controller_for_extract_test(
            tmp_path, selected_stems=("patchA",),
            mixed_stems={"patchA"},  # SAME as selection
        )
        ctrl._show_extraction_mixed_settings_dialog = (
            lambda s: dialog_called.append(True) or "cancel"
        )
        self._patch_worker_spawn(ctrl, monkeypatch)
        ctrl._extract_training_pairs()
        assert ctrl._stems_passed_to_worker == ["patchA"]
        assert not dialog_called


# ---------------------------------------------------------------------------
# Clear-all-extracted-pairs handler
# ---------------------------------------------------------------------------


def _make_controller_for_clear_test(tmp_path, *,
                                    per_patch=None,
                                    confirmed=False):
    """Controller fixture for `_on_clear_extracted_pairs_clicked` tests.

    Optionally seeds an extraction_manifest + at least one per-patch
    directory so the rmtree can do something visible.
    """
    ctrl = object.__new__(FusionReconstructionController)
    ctrl.working_dir = str(tmp_path)
    ctrl._log_lines = []
    ctrl._append_log = lambda t: ctrl._log_lines.append(t)
    ctrl._refresh_refinement_patch_list = lambda: None

    class _Label:
        def __init__(self):
            self.text = ""
        def setText(self, t):
            self.text = t
    class _FakeUI:
        pass
    ui = _FakeUI()
    ui.pairs_summary_label = _Label()
    ctrl.ui = ui

    # 2026-05-20 audit fix: manifest lives at the canonical helper
    # path (refinement/manifest.json), per-patch dirs under
    # refinement/pairs/.  Pre-fix this fixture mixed the two
    # (manifest inside pairs/) — matched the old runtime bug.
    from napari_epicycle.fusion_reconstruction import _paths
    refinement_pairs_dir = _paths.refinement_pairs_root(tmp_path)
    if per_patch is not None:
        refinement_pairs_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": "1.1",
            "per_patch": per_patch,
        }
        manifest_path = _paths.refinement_manifest_path(tmp_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        # Create at least one per-patch dir so the visible filesystem
        # state matches the manifest claim.
        for stem in per_patch:
            (refinement_pairs_dir / stem / "spatial").mkdir(parents=True)
    # Keep a local alias under the old variable name for any
    # downstream code in this helper that references it.
    refinement_dir = refinement_pairs_dir

    # Monkey-patch themed_confirm so tests don't spawn Qt.
    import napari_epicycle.fusion_reconstruction.controller as mod
    ctrl._patched_modules = (mod,)
    mod.themed_confirm = (
        lambda **kwargs: confirmed
    )
    # Inject themed_confirm into module namespace at the import site
    # used inside the handler (it's imported lazily inside the method).
    import napari_epicycle.ui_shell.theme as theme_mod
    theme_mod.themed_confirm = lambda **kwargs: confirmed

    return ctrl


class TestClearExtractedPairsHandler:

    def test_no_working_dir_logs_error(self, tmp_path):
        ctrl = object.__new__(FusionReconstructionController)
        ctrl.working_dir = ""
        ctrl._log_lines = []
        ctrl._append_log = lambda t: ctrl._log_lines.append(t)
        ctrl._on_clear_extracted_pairs_clicked()
        assert any("Set a working directory" in line
                   for line in ctrl._log_lines)

    def test_no_refinement_dir_logs_info_noop(self, tmp_path):
        ctrl = _make_controller_for_clear_test(tmp_path)
        ctrl._on_clear_extracted_pairs_clicked()
        assert any("No extracted pairs to clear" in line
                   for line in ctrl._log_lines)

    def test_cancel_preserves_data(self, tmp_path):
        ctrl = _make_controller_for_clear_test(
            tmp_path,
            per_patch={"patchA": {"spatial": 5, "axial": 3}},
            confirmed=False,  # user clicks Cancel
        )
        ctrl._on_clear_extracted_pairs_clicked()
        # Dir survives, summary label not touched, info logged.
        refinement_dir = (
            Path(tmp_path) / "artifacts" / "step3" / "refinement" / "pairs"
        )
        assert refinement_dir.exists()
        assert (refinement_dir / "patchA").is_dir()
        assert any("cancelled by user" in line for line in ctrl._log_lines)

    def test_confirmed_clears_dir(self, tmp_path):
        ctrl = _make_controller_for_clear_test(
            tmp_path,
            per_patch={
                "patchA": {"spatial": 5, "axial": 3},
                "patchB": {"spatial": 10, "axial": 2},
            },
            confirmed=True,
        )
        ctrl._on_clear_extracted_pairs_clicked()
        refinement_dir = (
            Path(tmp_path) / "artifacts" / "step3" / "refinement" / "pairs"
        )
        assert not refinement_dir.exists()
        # Total pairs reported: 5+3+10+2 = 20
        assert any("Cleared 20 training pairs" in line
                   for line in ctrl._log_lines)
        # Summary label reset.
        # "Will train on" counts the disk, and the disk is empty now.
        assert ctrl.ui.pairs_summary_label.text == (
            _MRT.MR_WILL_TRAIN_NOTHING)

    def test_confirmed_triggers_patch_list_refresh(self, tmp_path):
        ctrl = _make_controller_for_clear_test(
            tmp_path,
            per_patch={"patchA": {"spatial": 5, "axial": 3}},
            confirmed=True,
        )
        refresh_called = []
        ctrl._refresh_refinement_patch_list = (
            lambda: refresh_called.append(True)
        )
        ctrl._on_clear_extracted_pairs_clicked()
        assert refresh_called == [True]


# ---------------------------------------------------------------------------
# Per-orientation extraction settings — controller helpers
# (2026-05-19, feature/refinement-extraction-redesign sub-commit 1)
# ---------------------------------------------------------------------------


class _FakeStateManagerWithExtractionConfig:
    """Minimal state_manager exposing the extraction config field."""
    def __init__(self, config=None):
        self._config = config or {}
    def get_refinement_extraction_config(self):
        return dict(self._config)


def _make_ctrl_with_scalar_settings(scalar_settings, sm_config=None):
    """Build a controller where _read_extraction_settings returns the
    given scalar dict, and state_manager.get_refinement_extraction_config
    returns sm_config (or {} by default)."""
    ctrl = object.__new__(FusionReconstructionController)
    ctrl._read_extraction_settings = lambda: dict(scalar_settings)
    ctrl.state_manager = _FakeStateManagerWithExtractionConfig(sm_config)
    return ctrl


_BASE_SCALAR_SETTINGS = {
    "min_labels_per_slice": 3,
    "edge_buffer": 50,
    "slices_per_patch": 0,
    "min_distance": 5,
    "remove_boundaries": True,
    "relabel": True,
    "min_size": 100,
    "smoothing_disk_size": 1,
    "views": ("xy", "xz", "yz"),
}


class TestPerOrientationSettingsHelpers:

    def test_no_state_manager_config_returns_orientation_defaults(self):
        """2026-05-19 sub-commit 4: when state_manager has no override,
        each orientation returns its OWN default (LATERAL_DEFAULT_CONFIG
        vs AXIAL_DEFAULT_CONFIG) — NOT the legacy scalar UI defaults."""
        from napari_epicycle.fusion_reconstruction.extraction_config_dialog import (
            LATERAL_DEFAULT_CONFIG, AXIAL_DEFAULT_CONFIG,
        )
        ctrl = _make_ctrl_with_scalar_settings(
            _BASE_SCALAR_SETTINGS, sm_config={},
        )
        lat = ctrl._read_lateral_extraction_settings()
        ax = ctrl._read_axial_extraction_settings()
        # Each orientation returns its own constants.
        for key, value in LATERAL_DEFAULT_CONFIG.items():
            assert lat[key] == value, (
                f"lateral['{key}']={lat[key]!r} != "
                f"LATERAL_DEFAULT_CONFIG['{key}']={value!r}"
            )
        for key, value in AXIAL_DEFAULT_CONFIG.items():
            assert ax[key] == value
        # `views` is excluded from per-orientation params.
        assert "views" not in lat
        assert "views" not in ax

    def test_lateral_override_takes_precedence(self):
        from napari_epicycle.fusion_reconstruction.extraction_config_dialog import (
            AXIAL_DEFAULT_CONFIG,
        )
        ctrl = _make_ctrl_with_scalar_settings(
            _BASE_SCALAR_SETTINGS,
            sm_config={
                "lateral": {"smoothing_disk_size": 5, "min_size": 50},
            },
        )
        lat = ctrl._read_lateral_extraction_settings()
        ax = ctrl._read_axial_extraction_settings()
        # Lateral has the override values.
        assert lat["smoothing_disk_size"] == 5
        assert lat["min_size"] == 50
        # Axial stays at its orientation default.
        assert ax["smoothing_disk_size"] == AXIAL_DEFAULT_CONFIG["smoothing_disk_size"]
        assert ax["min_size"] == AXIAL_DEFAULT_CONFIG["min_size"]

    def test_axial_override_takes_precedence(self):
        from napari_epicycle.fusion_reconstruction.extraction_config_dialog import (
            LATERAL_DEFAULT_CONFIG,
        )
        ctrl = _make_ctrl_with_scalar_settings(
            _BASE_SCALAR_SETTINGS,
            sm_config={
                "axial": {"min_labels_per_slice": 7},
            },
        )
        lat = ctrl._read_lateral_extraction_settings()
        ax = ctrl._read_axial_extraction_settings()
        assert lat["min_labels_per_slice"] == LATERAL_DEFAULT_CONFIG["min_labels_per_slice"]
        assert ax["min_labels_per_slice"] == 7

    def test_partial_override_inherits_orientation_defaults(self):
        """Partial override: unspecified keys fall back to the
        orientation default (not the legacy scalar)."""
        from napari_epicycle.fusion_reconstruction.extraction_config_dialog import (
            LATERAL_DEFAULT_CONFIG,
        )
        ctrl = _make_ctrl_with_scalar_settings(
            _BASE_SCALAR_SETTINGS,
            sm_config={
                "lateral": {"smoothing_disk_size": 3},  # only this key
            },
        )
        lat = ctrl._read_lateral_extraction_settings()
        assert lat["smoothing_disk_size"] == 3   # overridden
        # Unspecified keys come from LATERAL_DEFAULT_CONFIG, not scalar.
        assert lat["min_size"] == LATERAL_DEFAULT_CONFIG["min_size"]
        assert lat["edge_buffer"] == LATERAL_DEFAULT_CONFIG["edge_buffer"]

    def test_invalid_orientation_raises(self):
        ctrl = _make_ctrl_with_scalar_settings(_BASE_SCALAR_SETTINGS)
        with pytest.raises(ValueError):
            ctrl._read_per_orientation_extraction_settings("diagonal")

    def test_unknown_keys_in_state_manager_config_are_ignored(self):
        """Per-orientation config may contain stale or unknown keys
        from older schemas — they must be silently dropped, not
        leaked into the kwargs dict."""
        ctrl = _make_ctrl_with_scalar_settings(
            _BASE_SCALAR_SETTINGS,
            sm_config={
                "lateral": {
                    "smoothing_disk_size": 2,
                    "future_knob": "should_be_ignored",
                },
            },
        )
        lat = ctrl._read_lateral_extraction_settings()
        assert lat["smoothing_disk_size"] == 2
        assert "future_knob" not in lat


# ---------------------------------------------------------------------------
# Per-orientation config dialog launcher
# (sub-commit 2 of feature/refinement-extraction-redesign)
# ---------------------------------------------------------------------------


class _RecordingStateManager:
    """State manager that records all set calls + returns canned configs."""
    def __init__(self, initial_config=None):
        self._config = initial_config or {}
        self.set_calls = []
    def get_refinement_extraction_config(self):
        return dict(self._config)
    def set_refinement_extraction_config(self, *, lateral=None, axial=None):
        self.set_calls.append({"lateral": lateral, "axial": axial})
        if lateral is not None:
            self._config["lateral"] = dict(lateral)
        if axial is not None:
            self._config["axial"] = dict(axial)


def _make_ctrl_for_dialog_test(initial_config=None, dialog_result=None):
    """Build a controller wired for `_on_configure_per_orientation_clicked`
    tests.  The dialog's `show_dialog` classmethod is monkey-patched to
    return ``dialog_result`` so we never spawn real Qt.
    """
    ctrl = object.__new__(FusionReconstructionController)
    ctrl.state_manager = _RecordingStateManager(initial_config)
    ctrl._log_lines = []
    ctrl._append_log = lambda t: ctrl._log_lines.append(t)
    ctrl._read_extraction_settings = lambda: dict(_BASE_SCALAR_SETTINGS)
    ctrl._refresh_refinement_patch_list = lambda: None

    class _FakeUI:
        pass
    ctrl.ui = _FakeUI()

    # Patch the dialog launcher to return the canned result.
    import napari_epicycle.fusion_reconstruction.extraction_config_dialog as dlg_mod
    captured = {"called_with": None}

    class _FakeDialog:
        @classmethod
        def show_dialog(cls, parent=None, *, initial_config=None,
                        scalar_defaults=None):
            captured["called_with"] = {
                "initial_config": initial_config,
                "scalar_defaults": scalar_defaults,
            }
            return dialog_result

    dlg_mod.ExtractionConfigDialog = _FakeDialog
    ctrl._dialog_captured = captured
    return ctrl


class TestConfigurePerOrientationHandler:

    def test_cancel_does_not_persist(self):
        ctrl = _make_ctrl_for_dialog_test(dialog_result=None)
        ctrl._on_configure_per_orientation_clicked()
        assert ctrl.state_manager.set_calls == []

    def test_save_persists_to_state_manager(self):
        ctrl = _make_ctrl_for_dialog_test(
            dialog_result={
                "lateral": {"smoothing_disk_size": 5, "min_size": 50},
                "axial": {"smoothing_disk_size": 3, "min_size": 100},
            },
        )
        ctrl._on_configure_per_orientation_clicked()
        assert len(ctrl.state_manager.set_calls) == 1
        call = ctrl.state_manager.set_calls[0]
        assert call["lateral"]["smoothing_disk_size"] == 5
        assert call["axial"]["smoothing_disk_size"] == 3

    def test_dialog_receives_current_state_manager_config(self):
        ctrl = _make_ctrl_for_dialog_test(
            initial_config={"lateral": {"smoothing_disk_size": 7}},
            dialog_result=None,
        )
        ctrl._on_configure_per_orientation_clicked()
        captured = ctrl._dialog_captured["called_with"]
        assert captured["initial_config"]["lateral"]["smoothing_disk_size"] == 7

    def test_dialog_receives_empty_scalar_defaults(self):
        """2026-05-19 sub-commit 4: the dialog no longer uses
        scalar_defaults as a fallback (it has orientation-specific
        constants of its own).  Controller passes an empty dict for
        back-compat with the constructor signature."""
        ctrl = _make_ctrl_for_dialog_test(dialog_result=None)
        ctrl._on_configure_per_orientation_clicked()
        captured = ctrl._dialog_captured["called_with"]
        assert captured["scalar_defaults"] == {}

    def test_save_refreshes_patch_list(self):
        ctrl = _make_ctrl_for_dialog_test(
            dialog_result={"lateral": {"min_size": 200}, "axial": {}},
        )
        refresh_called = []
        ctrl._refresh_refinement_patch_list = (
            lambda: refresh_called.append(True)
        )
        ctrl._on_configure_per_orientation_clicked()
        assert refresh_called == [True]

    def test_missing_state_manager_logs_warning(self):
        ctrl = _make_ctrl_for_dialog_test(
            dialog_result={"lateral": {"min_size": 50}, "axial": {}},
        )
        ctrl.state_manager = None  # simulate bare instance
        ctrl._on_configure_per_orientation_clicked()
        assert any(
            "State manager unavailable" in line
            for line in ctrl._log_lines
        )


class TestExtractionConfigDialogKeysContract:
    """Pure-Python invariants on the dialog module (no Qt needed).

    These guard against the key vocabulary drifting out of sync between
    the dialog, the controller helpers, and the extractor.
    """

    def test_key_lists_cover_all_per_orientation_keys(self):
        from napari_epicycle.fusion_reconstruction import (
            extraction_config_dialog as dlg_mod,
        )
        keys = set(dlg_mod._SAMPLING_KEYS) | set(dlg_mod._CLEANING_KEYS)
        # 2026-05-21: added `expand_labels_distance` to compensate
        # for the `remove_boundaries=True` erosion bias that causes
        # mask shrinkage across refinement rounds.  Keep this set in
        # sync when adding/removing per-orientation params.
        assert keys == {
            "min_labels_per_slice", "edge_buffer", "slices_per_patch",
            "min_distance",
            "remove_boundaries", "relabel", "min_size",
            "smoothing_disk_size",
            "expand_labels_distance",
        }

    def test_key_lists_match_controller_per_view_param_keys(self):
        from napari_epicycle.fusion_reconstruction import (
            extraction_config_dialog as dlg_mod,
        )
        keys = set(dlg_mod._SAMPLING_KEYS) | set(dlg_mod._CLEANING_KEYS)
        assert keys == set(FusionReconstructionController._PERVIEW_PARAM_KEYS)

    def test_every_key_has_label_and_tooltip(self):
        from napari_epicycle.fusion_reconstruction import (
            extraction_config_dialog as dlg_mod,
        )
        for key in dlg_mod._SAMPLING_KEYS + dlg_mod._CLEANING_KEYS:
            assert key in dlg_mod._LABELS, f"missing label for {key}"
            assert key in dlg_mod._TOOLTIPS, f"missing tooltip for {key}"


# ---------------------------------------------------------------------------
# Per-orientation default config constants (sub-commit 4)
# ---------------------------------------------------------------------------


class TestPerOrientationDefaults:

    def test_lateral_and_axial_defaults_differ(self):
        from napari_epicycle.fusion_reconstruction.extraction_config_dialog import (
            LATERAL_DEFAULT_CONFIG, AXIAL_DEFAULT_CONFIG,
        )
        assert LATERAL_DEFAULT_CONFIG != AXIAL_DEFAULT_CONFIG, (
            "Sub-commit 4 explicitly differentiates defaults — they "
            "must not be identical."
        )

    def test_both_default_dicts_have_all_eight_keys(self):
        from napari_epicycle.fusion_reconstruction.extraction_config_dialog import (
            LATERAL_DEFAULT_CONFIG, AXIAL_DEFAULT_CONFIG,
            _SAMPLING_KEYS, _CLEANING_KEYS,
        )
        expected = set(_SAMPLING_KEYS) | set(_CLEANING_KEYS)
        assert set(LATERAL_DEFAULT_CONFIG.keys()) == expected
        assert set(AXIAL_DEFAULT_CONFIG.keys()) == expected

    def test_get_default_config_for_returns_fresh_copy(self):
        from napari_epicycle.fusion_reconstruction.extraction_config_dialog import (
            get_default_config_for, LATERAL_DEFAULT_CONFIG,
        )
        cfg = get_default_config_for("lateral")
        cfg["min_size"] = 99999  # mutate
        # Mutating the returned dict must not poison the constant.
        assert LATERAL_DEFAULT_CONFIG["min_size"] != 99999

    def test_get_default_config_for_invalid_orientation_raises(self):
        from napari_epicycle.fusion_reconstruction.extraction_config_dialog import (
            get_default_config_for,
        )
        with pytest.raises(ValueError):
            get_default_config_for("diagonal")

    def test_controller_fallback_uses_orientation_defaults(self):
        """When state_manager has no override, the controller's
        per-orientation helper returns the orientation-specific
        default — NOT the legacy scalar UI values."""
        from napari_epicycle.fusion_reconstruction.extraction_config_dialog import (
            LATERAL_DEFAULT_CONFIG, AXIAL_DEFAULT_CONFIG,
        )
        ctrl = _make_ctrl_with_scalar_settings(
            _BASE_SCALAR_SETTINGS, sm_config={},
        )
        lat = ctrl._read_lateral_extraction_settings()
        ax = ctrl._read_axial_extraction_settings()
        # Lateral defaults: edge_buffer=5, min_size=100
        assert lat["edge_buffer"] == LATERAL_DEFAULT_CONFIG["edge_buffer"]
        assert lat["min_size"] == LATERAL_DEFAULT_CONFIG["min_size"]
        # Axial defaults: edge_buffer=20, min_size=50
        assert ax["edge_buffer"] == AXIAL_DEFAULT_CONFIG["edge_buffer"]
        assert ax["min_size"] == AXIAL_DEFAULT_CONFIG["min_size"]
        # They are NOT equal — the controller no longer returns
        # identical values for both orientations.
        assert lat["edge_buffer"] != ax["edge_buffer"]
        assert lat["min_size"] != ax["min_size"]


# ---------------------------------------------------------------------------
# Per-orientation summary label refresher
# ---------------------------------------------------------------------------


class _SummaryFakeUI:
    """Fake UI exposing only the summary label."""
    def __init__(self):
        self.refine_per_orientation_summary_label = _SummaryFakeLabel()


class _SummaryFakeLabel:
    def __init__(self):
        self.text = ""
    def setText(self, t):
        self.text = t


class TestRefreshPerOrientationSummary:

    def test_renders_default_marker_when_no_override(self):
        ctrl = _make_ctrl_with_scalar_settings(
            _BASE_SCALAR_SETTINGS, sm_config={},
        )
        ctrl.ui = _SummaryFakeUI()
        ctrl._refresh_per_orientation_summary()
        text = ctrl.ui.refine_per_orientation_summary_label.text
        assert "Lateral (default)" in text
        assert "Axial (default)" in text

    def test_renders_no_default_marker_when_overridden(self):
        ctrl = _make_ctrl_with_scalar_settings(
            _BASE_SCALAR_SETTINGS,
            sm_config={"lateral": {"smoothing_disk_size": 3}},
        )
        ctrl.ui = _SummaryFakeUI()
        ctrl._refresh_per_orientation_summary()
        text = ctrl.ui.refine_per_orientation_summary_label.text
        # Lateral has an override — no "(default)" suffix.
        assert "Lateral:" in text
        assert "Lateral (default)" not in text
        # Axial has no override — still tagged "(default)".
        assert "Axial (default)" in text

    def test_renders_all_slices_when_cap_zero(self):
        ctrl = _make_ctrl_with_scalar_settings(
            _BASE_SCALAR_SETTINGS, sm_config={},
        )
        ctrl.ui = _SummaryFakeUI()
        ctrl._refresh_per_orientation_summary()
        text = ctrl.ui.refine_per_orientation_summary_label.text
        # Both orientations have slices_per_patch=0 by default.
        assert "all slices" in text

    def test_renders_n_slices_when_cap_set(self):
        ctrl = _make_ctrl_with_scalar_settings(
            _BASE_SCALAR_SETTINGS,
            sm_config={
                "lateral": {"slices_per_patch": 25},
                "axial": {"slices_per_patch": 15},
            },
        )
        ctrl.ui = _SummaryFakeUI()
        ctrl._refresh_per_orientation_summary()
        text = ctrl.ui.refine_per_orientation_summary_label.text
        assert "25 slices" in text
        assert "15 slices" in text

    def test_no_ui_no_crash(self):
        ctrl = object.__new__(FusionReconstructionController)
        ctrl.ui = None
        # Must not raise.
        ctrl._refresh_per_orientation_summary()


# ---------------------------------------------------------------------------
# Fine-Tuning Mode hint label refresher
# (sub-commit 11 of feature/refinement-extraction-redesign)
# ---------------------------------------------------------------------------


class _ModeHintFakeUI:
    """Fake UI exposing only the mode combo + hint label."""
    def __init__(self, current_mode="Fine-tune Lateral"):
        self.refinement_mode_combo = _ModeHintCombo(current_mode)
        self.refinement_mode_hint = _SummaryFakeLabel()


class _ModeHintCombo:
    def __init__(self, text):
        self._text = text
    def currentText(self):
        return self._text


class TestRefreshModeHint:

    def test_lateral_hint_mentions_lateral_only(self):
        ctrl = object.__new__(FusionReconstructionController)
        ctrl.ui = _ModeHintFakeUI(current_mode="Fine-tune Lateral")
        ctrl._refresh_mode_hint()
        text = ctrl.ui.refinement_mode_hint.text
        # ONE sentence, singular (Brick B2, M13 / spec section 5)
        assert text == _MRT.MR_HINT_LATERAL.format(
            lat=_MRT.MR_CARD_INIT_SCRATCH, name=_MRT.MR_HINT_NAME_AUTO)
        assert "axial" not in text and "\n" not in text

    def test_axial_hint_mentions_axial_only(self):
        ctrl = object.__new__(FusionReconstructionController)
        ctrl.ui = _ModeHintFakeUI(current_mode="Fine-tune Axial")
        ctrl._refresh_mode_hint()
        text = ctrl.ui.refinement_mode_hint.text
        assert text == _MRT.MR_HINT_AXIAL.format(
            ax=_MRT.MR_CARD_INIT_SCRATCH, name=_MRT.MR_HINT_NAME_AUTO)
        assert "lateral" not in text and "\n" not in text

    def test_both_hint_mentions_sequential_and_2x_runtime(self):
        ctrl = object.__new__(FusionReconstructionController)
        ctrl.ui = _ModeHintFakeUI(current_mode="Fine-tune Both")
        ctrl._refresh_mode_hint()
        text = ctrl.ui.refinement_mode_hint.text
        assert text == _MRT.MR_HINT_BOTH.format(
            lat=_MRT.MR_CARD_INIT_SCRATCH, ax=_MRT.MR_CARD_INIT_SCRATCH,
            name=_MRT.MR_HINT_NAME_AUTO)
        assert "\n" not in text

    def test_explicit_mode_arg_overrides_combo(self):
        """The signal connection passes the new text as an arg —
        should be honored over the combo's currentText."""
        ctrl = object.__new__(FusionReconstructionController)
        ctrl.ui = _ModeHintFakeUI(current_mode="Fine-tune Lateral")
        ctrl._refresh_mode_hint(mode="Fine-tune Both")
        text = ctrl.ui.refinement_mode_hint.text
        # Should show the Both hint, not the Lateral hint.
        assert "one name for both models" in text

    def test_no_ui_no_crash(self):
        ctrl = object.__new__(FusionReconstructionController)
        ctrl.ui = None
        ctrl._refresh_mode_hint()  # must not raise

    def test_missing_hint_label_no_crash(self):
        """ui present but missing the hint label widget."""
        ctrl = object.__new__(FusionReconstructionController)
        class _NoHintUI:
            refinement_mode_combo = _ModeHintCombo("Fine-tune Lateral")
        ctrl.ui = _NoHintUI()
        ctrl._refresh_mode_hint()  # must not raise
