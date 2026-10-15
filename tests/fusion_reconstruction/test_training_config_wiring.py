"""Tests for training presets, config building, and init mode wiring (Phase 4).

Covers:
- Preset application fills UI fields correctly
- TrainingConfig built from UI values
- _views_from_combo_text (already tested but included for completeness)
- Init mode visibility toggles
- Save/load config round-trip
"""

import json
import pytest
from unittest.mock import MagicMock

from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeSpinBox:
    def __init__(self, val=0, lo=0, hi=100):
        self._val = val
    def value(self):
        return self._val
    def setValue(self, v):
        self._val = v
    def setRange(self, lo, hi):
        pass


class FakeDoubleSpinBox(FakeSpinBox):
    pass


class FakeCheckBox:
    def __init__(self, checked=False):
        self._checked = checked
    def isChecked(self):
        return self._checked
    def setChecked(self, v):
        self._checked = v


class FakeWidget:
    """Generic fake widget with visibility tracking."""
    def __init__(self, visible=True):
        self._visible = visible
    def setVisible(self, v):
        self._visible = v
    def isVisible(self):
        return self._visible


class FakeComboBox:
    def __init__(self, items=None, idx=0):
        self._items = items or []
        self._idx = idx
        self._visible = True
    def currentText(self):
        return self._items[self._idx] if self._items else ""
    def currentIndex(self):
        return self._idx
    def setCurrentIndex(self, i):
        self._idx = i
    def setCurrentText(self, t):
        if t in self._items:
            self._idx = self._items.index(t)
    def addItems(self, items):
        self._items.extend(items)
    def addItem(self, item):
        self._items.append(item)
    def clear(self):
        self._items.clear()
        self._idx = 0
    def setVisible(self, v):
        self._visible = v
    def isVisible(self):
        return self._visible
    def count(self):
        return len(self._items)


class FakeRadioButton:
    def __init__(self, checked=False):
        self._checked = checked
    def isChecked(self):
        return self._checked
    def setChecked(self, v):
        self._checked = v


class FakeButton:
    def __init__(self):
        self._enabled = True
        self._connections = []
    def setEnabled(self, v):
        self._enabled = v
    def isEnabled(self):
        return self._enabled
    @property
    def clicked(self):
        return self
    def connect(self, slot):
        self._connections.append(slot)


class FakeLabel:
    def __init__(self, text=""):
        self._text = text
    def setText(self, t):
        self._text = t
    def text(self):
        return self._text


class FakeLineEdit:
    def __init__(self, text=""):
        self._text = text
    def text(self):
        return self._text
    def setText(self, t):
        self._text = t
    def setPlaceholderText(self, t):
        pass


class FakeListWidget:
    def __init__(self):
        self._items = []
    def clear(self):
        self._items.clear()
    def selectedItems(self):
        return []
    def selectAll(self):
        pass
    def clearSelection(self):
        pass


class FakeProgressBar:
    def __init__(self):
        self._value = 0
    def setValue(self, v):
        self._value = v
    def setRange(self, lo, hi):
        pass


class FakeLog:
    def __init__(self):
        self._lines = []
    def append(self, text):
        self._lines.append(text)


def _make_full_ui():
    """Build a fake UI with all Model Refinement widgets."""
    ui = MagicMock()

    # Core training params
    ui.refine_epochs_sb = FakeSpinBox(500)
    ui.refine_lr_sb = FakeDoubleSpinBox(0.01)
    ui.refine_batch_sb = FakeSpinBox(16)

    # Advanced params
    ui.refine_wd_sb = FakeDoubleSpinBox(1e-5)
    ui.refine_momentum_sb = FakeDoubleSpinBox(0.9)
    ui.refine_min_masks_sb = FakeSpinBox(0)
    ui.refine_chan1_combo = FakeComboBox(
        ["0: gray", "1: red", "2: green", "3: blue"], idx=2
    )
    ui.refine_chan2_combo = FakeComboBox(
        ["0: none", "1: red", "2: green", "3: blue"], idx=1
    )
    ui.refine_seed_sb = FakeSpinBox(0)
    ui.refine_normalize_cb = FakeCheckBox(True)
    ui.refine_rescale_cb = FakeCheckBox(True)
    ui.refine_sgd_cb = FakeCheckBox(True)
    ui.refine_compute_flows_cb = FakeCheckBox(False)
    ui.refine_device_gpu = FakeRadioButton(True)
    ui.refine_device_cpu = FakeRadioButton(False)

    # Preset
    ui.refine_preset_combo = FakeComboBox([
        "default", "quick", "thorough", "finetune",
        "dense_tissue", "stratified_epithelia",
    ])
    ui.refine_apply_preset_btn = FakeButton()

    # Init mode
    ui.refine_init_combo = FakeComboBox([
        "From HITL model", "From prior refinement",
        "From Cellpose pretrained", "From scratch",
    ])
    ui.refinement_source_combo = FakeComboBox()
    ui.refine_source_row = FakeWidget(visible=True)
    ui.refine_pretrained_combo = FakeComboBox(
        ["cyto3", "cyto2", "cyto", "nuclei"]
    )
    ui.refine_pretrained_row = FakeWidget(visible=False)
    ui.refine_import_row = FakeWidget(visible=False)
    ui.refine_import_path = FakeLineEdit()
    ui.refine_import_browse_btn = FakeButton()

    # Save/Load
    ui.refine_save_config_btn = FakeButton()
    ui.refine_load_config_btn = FakeButton()

    # Other existing widgets
    ui.refinement_mode_combo = FakeComboBox([
        "Fine-tune Lateral", "Fine-tune Axial", "Fine-tune Both",
    ])
    ui.refinement_status_label = FakeLabel()
    ui.refinement_progress = FakeProgressBar()
    ui.refinement_output_label = FakeLabel()
    ui.refine_model_name = FakeLineEdit()

    # Extraction widgets (from Phase 2/3)
    ui.refine_patch_list = FakeListWidget()
    ui.refine_select_all_btn = FakeButton()
    ui.refine_deselect_all_btn = FakeButton()
    ui.extract_training_pairs_btn = FakeButton()
    ui.launch_refinement_btn = FakeButton()
    ui.pairs_summary_label = FakeLabel()
    ui.refine_extraction_progress = FakeProgressBar()
    ui.refine_views_combo = FakeComboBox([
        "All (XY + XZ + YZ)", "Lateral (XY only)", "Axial (XZ + YZ only)",
    ])
    ui.refine_min_labels_sb = FakeSpinBox(3)
    ui.refine_slices_per_patch_sb = FakeSpinBox(0)
    ui.refine_min_distance_sb = FakeSpinBox(5)
    ui.refine_edge_buffer_sb = FakeSpinBox(50)
    ui.refine_remove_boundaries_cb = FakeCheckBox(True)
    ui.refine_relabel_cb = FakeCheckBox(True)
    ui.refine_min_size_sb = FakeSpinBox(100)
    ui.refine_smoothing_disk_sb = FakeSpinBox(1)

    # Split
    ui.refine_train_frac_sb = FakeDoubleSpinBox(0.8)
    ui.refine_spatial_split_cb = FakeCheckBox(True)
    ui.refine_chunk_size_sb = FakeSpinBox(10)
    ui.refine_split_summary_label = FakeLabel()

    # Model output (2026-05-21 Stage 1: refine_pack_model_btn removed —
    # _register_model_pack now auto-fires at end of training)
    ui.refine_use_for_predict_btn = FakeButton()
    ui.refine_pack_status_label = FakeLabel()

    # Log
    ui.log_output = FakeLog()

    # refresh method
    ui.refresh_refinement_patch_list = MagicMock()

    return ui


def _make_controller(ui=None):
    """Create a controller with fake UI for testing config wiring."""
    if ui is None:
        ui = _make_full_ui()
    ctrl = object.__new__(FusionReconstructionController)
    ctrl.ui = ui
    ctrl.state_manager = MagicMock()
    ctrl.state_manager.get_all_patch_statuses.return_value = {}
    ctrl.working_dir = "/fake/workspace"
    ctrl.loaded_stack_dir = "/fake/stacks"
    return ctrl


# ---------------------------------------------------------------------------
# Tests: Build TrainingConfig from UI
# ---------------------------------------------------------------------------

class TestBuildTrainingConfig:
    """Test _build_training_config reads UI values correctly."""

    def test_core_params(self):
        ctrl = _make_controller()
        ctrl.ui.refine_epochs_sb.setValue(200)
        ctrl.ui.refine_lr_sb.setValue(0.005)
        ctrl.ui.refine_batch_sb.setValue(32)

        config = ctrl._build_training_config()
        assert config["n_epochs"] == 200
        assert config["learning_rate"] == 0.005
        assert config["batch_size"] == 32

    def test_advanced_params(self):
        ctrl = _make_controller()
        ctrl.ui.refine_wd_sb.setValue(0.001)
        ctrl.ui.refine_momentum_sb.setValue(0.95)
        ctrl.ui.refine_min_masks_sb.setValue(5)
        ctrl.ui.refine_sgd_cb.setChecked(False)
        ctrl.ui.refine_normalize_cb.setChecked(False)
        ctrl.ui.refine_compute_flows_cb.setChecked(True)

        config = ctrl._build_training_config()
        assert config["weight_decay"] == 0.001
        assert config["momentum"] == 0.95
        assert config["min_train_masks"] == 5
        assert config["SGD"] is False
        assert config["normalize"] is False
        assert config["compute_flows"] is True

    def test_channels_from_combos(self):
        ctrl = _make_controller()
        # chan1 = "2: green" (index 2), chan2 = "1: red" (index 1)
        ctrl.ui.refine_chan1_combo.setCurrentIndex(2)
        ctrl.ui.refine_chan2_combo.setCurrentIndex(1)

        config = ctrl._build_training_config()
        assert config["channels"] == [2, 1]

    def test_seed_zero_means_none(self):
        ctrl = _make_controller()
        ctrl.ui.refine_seed_sb.setValue(0)
        config = ctrl._build_training_config()
        assert config["seed"] is None

    def test_seed_nonzero(self):
        ctrl = _make_controller()
        ctrl.ui.refine_seed_sb.setValue(42)
        config = ctrl._build_training_config()
        assert config["seed"] == 42

    def test_device_gpu(self):
        ctrl = _make_controller()
        ctrl.ui.refine_device_gpu.setChecked(True)
        ctrl.ui.refine_device_cpu.setChecked(False)
        config = ctrl._build_training_config()
        assert config["device"] == "gpu"

    def test_device_cpu(self):
        ctrl = _make_controller()
        ctrl.ui.refine_device_gpu.setChecked(False)
        ctrl.ui.refine_device_cpu.setChecked(True)
        config = ctrl._build_training_config()
        assert config["device"] == "cpu"


# ---------------------------------------------------------------------------
# Tests: Presets
# ---------------------------------------------------------------------------

class TestPresetApplication:
    """Test _apply_training_preset fills UI fields."""

    def test_default_preset(self):
        ctrl = _make_controller()
        ctrl._apply_training_preset("default")
        # 2026-05-22: default preset bumped to 100 epochs to match the
        # UI spinbox's initial value (so clicking Apply on the
        # default-selected combo doesn't jump from 100 → 500).
        assert ctrl.ui.refine_epochs_sb.value() == 100
        assert ctrl.ui.refine_lr_sb.value() == 0.01
        assert ctrl.ui.refine_batch_sb.value() == 16

    def test_quick_preset(self):
        ctrl = _make_controller()
        ctrl._apply_training_preset("quick")
        assert ctrl.ui.refine_epochs_sb.value() == 100
        assert ctrl.ui.refine_batch_sb.value() == 32

    def test_finetune_preset(self):
        ctrl = _make_controller()
        ctrl._apply_training_preset("finetune")
        assert ctrl.ui.refine_lr_sb.value() == 0.001

    def test_unknown_preset_does_not_crash(self):
        ctrl = _make_controller()
        # Should not raise — log a warning instead
        ctrl._apply_training_preset("nonexistent_preset")


# ---------------------------------------------------------------------------
# Tests: Init mode visibility
# ---------------------------------------------------------------------------

class TestInitModeVisibility:
    """Test that init mode toggles source/pretrained row visibility."""

    def test_hitl_shows_source_hides_pretrained(self):
        ctrl = _make_controller()
        ctrl._on_init_mode_changed("From HITL model")
        assert ctrl.ui.refine_source_row.isVisible()
        assert not ctrl.ui.refine_pretrained_row.isVisible()

    def test_prior_shows_source_hides_pretrained(self):
        ctrl = _make_controller()
        ctrl._on_init_mode_changed("From prior refinement")
        assert ctrl.ui.refine_source_row.isVisible()
        assert not ctrl.ui.refine_pretrained_row.isVisible()

    def test_pretrained_hides_source_shows_pretrained(self):
        ctrl = _make_controller()
        ctrl._on_init_mode_changed("From Cellpose pretrained")
        assert not ctrl.ui.refine_source_row.isVisible()
        assert ctrl.ui.refine_pretrained_row.isVisible()

    def test_import_shows_import_row(self):
        ctrl = _make_controller()
        ctrl._on_init_mode_changed("Import model")
        assert not ctrl.ui.refine_source_row.isVisible()
        assert not ctrl.ui.refine_pretrained_row.isVisible()
        assert ctrl.ui.refine_import_row.isVisible()

    def test_scratch_hides_all(self):
        ctrl = _make_controller()
        ctrl._on_init_mode_changed("From scratch")
        assert not ctrl.ui.refine_source_row.isVisible()
        assert not ctrl.ui.refine_pretrained_row.isVisible()
        assert not ctrl.ui.refine_import_row.isVisible()


# ---------------------------------------------------------------------------
# Tests: Save/Load config
# ---------------------------------------------------------------------------

class TestConfigPersistence:
    """Test save/load config round-trip."""

    def test_save_load_round_trip(self, tmp_path):
        ctrl = _make_controller()
        ctrl.working_dir = str(tmp_path)

        # Set non-default values
        ctrl.ui.refine_epochs_sb.setValue(750)
        ctrl.ui.refine_lr_sb.setValue(0.005)
        ctrl.ui.refine_batch_sb.setValue(8)
        ctrl.ui.refine_sgd_cb.setChecked(False)
        ctrl.ui.refine_seed_sb.setValue(42)

        ctrl._save_refinement_config()

        # B4: it lives under artifacts/ now.
        from napari_epicycle.fusion_reconstruction._paths import (
            step3_refinement_config_path,
        )
        config_path = step3_refinement_config_path(tmp_path)
        assert config_path.exists()
        assert not (tmp_path / "refinement_config.json").exists()

        # Reset UI to defaults
        ctrl.ui.refine_epochs_sb.setValue(500)
        ctrl.ui.refine_lr_sb.setValue(0.01)
        ctrl.ui.refine_batch_sb.setValue(16)
        ctrl.ui.refine_sgd_cb.setChecked(True)
        ctrl.ui.refine_seed_sb.setValue(0)

        ctrl._load_refinement_config()

        # Should be restored
        assert ctrl.ui.refine_epochs_sb.value() == 750
        assert ctrl.ui.refine_lr_sb.value() == 0.005
        assert ctrl.ui.refine_batch_sb.value() == 8
        assert ctrl.ui.refine_sgd_cb.isChecked() is False
        assert ctrl.ui.refine_seed_sb.value() == 42

    def test_load_missing_file_does_not_crash(self, tmp_path):
        ctrl = _make_controller()
        ctrl.working_dir = str(tmp_path)
        # No config file exists
        ctrl._load_refinement_config()  # should not raise
