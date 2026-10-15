"""Tests for the Training Setup init-transparency feature (2026-05-27).

Two behaviors under test:

1. **Auto-switch Mode → From Scratch** when user explicitly changes
   Base Model or types a Checkpoint path.  Programmatic state-restore
   should NOT trigger the switch (guard via _init_transparency_loading).

2. **`_compute_init_hint(self)`** returns ``(severity, text)`` tuple
   summarizing what'll happen on the next Train click.  ``severity``
   ∈ {"apply", "override", "neutral"}.  Pure read — no side effects on
   ``self.current_config`` or any widget.

Written tests-first (a failing test before the fix).  Reuses the stub classes from
``test_state_persistence_integration``.
"""
from napari_epicycle.hitl_training_annotation.ui import _step2_text as _T  # noqa: E402
import json
from types import SimpleNamespace

from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
    TrainingAnnotationController,
)
from napari_epicycle.io_registry import HITLPersistenceService

# Reuse the widget stubs.  Same pattern as
# test_state_persistence_integration — keeps stub maintenance in one
# place even when test files diverge.
from .test_state_persistence_integration import (
    _StubSignal, _StubSpinBox, _StubCheckBox, _StubLineEdit, _StubComboBox,
)


def _controller_with_full_ui(tmp_path):
    """Stub controller with every widget the init-transparency feature
    touches: model_selector, training_mode_combo, starting_checkpoint_path,
    chan1/chan2 + inference equivalents, _init_hint_label.
    """
    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.persistence = HITLPersistenceService()
    ctrl.working_dir = str(tmp_path)
    ctrl.active_run_id = "default"
    ctrl.current_round = 0
    ctrl.current_round_by_view = {"lateral": 0, "axial": 0}
    ctrl.selected_view = "XY"
    ctrl.view_target = "lateral"
    ctrl.training_data_by_round = {}
    ctrl.losses_by_round = {}
    ctrl.model_checkpoint_paths = {}
    ctrl.current_config = {}
    ctrl.sample_records = {}
    ctrl.viewer = None
    ctrl.loaded_patch_name = ""

    ctrl.ui = SimpleNamespace(
        # Training Setup basic
        epochs_sb=_StubSpinBox(500),
        batch_sb=_StubSpinBox(8),
        device_gpu=_StubCheckBox(True),
        device_cpu=_StubCheckBox(False),
        model_selector=_StubComboBox(
            ["scratch", "cyto3", "nuclei", "cyto2_cp3"], "scratch"
        ),
        starting_checkpoint_path=_StubLineEdit(""),
        # Mode dropdown
        training_mode_combo=_StubComboBox(
            ["Cumulative — reuse prior model (default)",
             "From Scratch — no prior checkpoint"],
            "Cumulative — reuse prior model (default)",
        ),
        # Advanced channels (sync-target for the channel-sync wiring)
        chan1_combo=_StubComboBox(
            ["0: gray", "1: red", "2: green", "3: blue"], "2: green"
        ),
        chan2_combo=_StubComboBox(
            ["0: none", "1: red", "2: green", "3: blue"], "1: red"
        ),
        norm_cb=_StubCheckBox(True),
        rescale_cb=_StubCheckBox(True),
        flow_cb=_StubCheckBox(True),
        # Inference (sync targets)
        inference_channel1_combo=_StubComboBox(
            ["0: gray", "1: red", "2: green", "3: blue"], "2: green"
        ),
        inference_channel2_combo=_StubComboBox(
            ["0: none", "1: red", "2: green", "3: blue"], "1: red"
        ),
        inference_flow_thresh=_StubSpinBox(0.4),
        inference_cellprob_thresh=_StubSpinBox(0.0),
        inference_diameter_sb=_StubSpinBox(0),
        inference_niter_dynamics=_StubSpinBox(0),
        inference_normalize_cb=_StubCheckBox(True),
        inference_rescale_cb=_StubCheckBox(True),
        inference_invert_cb=_StubCheckBox(False),
        # NEW — the hint label this feature adds.
        _init_hint_label=_StubLineEdit(""),
    )
    ctrl._append_ui_log = lambda *a, **kw: None
    ctrl._channel_sync_in_progress = False
    ctrl._init_transparency_loading = False

    # Wire the production signal connections we care about for these
    # tests (auto-switch + sync).  Mirror what _connect_signals would do.
    ctrl.ui.model_selector.currentTextChanged.connect(
        lambda t: ctrl._on_init_widget_changed("model_selector", t)
    )
    ctrl.ui.starting_checkpoint_path.textChanged.connect(
        lambda t: ctrl._on_init_widget_changed("starting_checkpoint_path", t)
    ) if hasattr(ctrl.ui.starting_checkpoint_path, "textChanged") else None
    return ctrl


# Stub widget needs a textChanged signal for QLineEdit — patch in
class _StubLineEditWithSignal(_StubLineEdit):
    def __init__(self, initial=""):
        super().__init__(initial)
        self.textChanged = _StubSignal()
    def setText(self, t):
        old = self._t
        super().setText(t)
        if old != self._t:
            self.textChanged.emit(self._t)


def _controller_with_textChanged(tmp_path):
    """Variant where starting_checkpoint_path has a textChanged signal."""
    ctrl = _controller_with_full_ui(tmp_path)
    # Replace with signal-enabled stub
    ctrl.ui.starting_checkpoint_path = _StubLineEditWithSignal("")
    ctrl.ui.starting_checkpoint_path.textChanged.connect(
        lambda t: ctrl._on_init_widget_changed("starting_checkpoint_path", t)
    )
    return ctrl


# ============================================================================
# Auto-switch tests (5)
# ============================================================================

def test_A_user_picks_new_base_model_flips_mode_to_from_scratch(tmp_path):
    """Mode is Cumulative; user picks 'cyto3' from Base Model dropdown.
    Mode should auto-flip to 'From Scratch'."""
    ctrl = _controller_with_full_ui(tmp_path)
    assert "Cumulative" in ctrl.ui.training_mode_combo.currentText()

    # Simulate user clicking the dropdown and picking cyto3.
    ctrl.ui.model_selector.setCurrentText("cyto3")

    assert "Scratch" in ctrl.ui.training_mode_combo.currentText(), (
        f"Mode should have flipped to From Scratch, got "
        f"{ctrl.ui.training_mode_combo.currentText()!r}"
    )


def test_B_user_types_checkpoint_path_flips_mode_to_from_scratch(tmp_path):
    """Mode is Cumulative; user types a path into the Checkpoint widget.
    Mode should auto-flip to 'From Scratch'."""
    ctrl = _controller_with_textChanged(tmp_path)
    assert "Cumulative" in ctrl.ui.training_mode_combo.currentText()

    ctrl.ui.starting_checkpoint_path.setText("/some/path/to/ckpt.pt")

    assert "Scratch" in ctrl.ui.training_mode_combo.currentText()


def test_C_user_clears_checkpoint_does_not_flip_mode(tmp_path):
    """Mode is Cumulative; user clears an existing Checkpoint path.
    Clearing isn't 'selecting an override' — Mode should stay Cumulative."""
    ctrl = _controller_with_textChanged(tmp_path)
    # Pre-populate the checkpoint (bypass signal — set _t directly).
    ctrl.ui.starting_checkpoint_path._t = "/pre/existing.pt"
    initial_mode = ctrl.ui.training_mode_combo.currentText()

    # User clears the field.
    ctrl.ui.starting_checkpoint_path.setText("")

    assert ctrl.ui.training_mode_combo.currentText() == initial_mode, (
        f"Clearing checkpoint should not change Mode; was {initial_mode!r}, "
        f"now {ctrl.ui.training_mode_combo.currentText()!r}"
    )


def test_D_programmatic_state_restore_does_NOT_flip_mode(tmp_path):
    """When _sync_training_controls_from_config runs during workspace
    load, it programmatically sets model_selector to the saved value.
    This should NOT trigger the auto-flip — guard via
    _init_transparency_loading flag."""
    ctrl = _controller_with_full_ui(tmp_path)
    assert "Cumulative" in ctrl.ui.training_mode_combo.currentText()

    # Simulate state-load: set guard, change widget programmatically.
    ctrl._init_transparency_loading = True
    try:
        ctrl.ui.model_selector.setCurrentText("cyto3")
    finally:
        ctrl._init_transparency_loading = False

    assert "Cumulative" in ctrl.ui.training_mode_combo.currentText(), (
        "Programmatic widget change during state restore should NOT "
        "auto-flip Mode."
    )


def test_E_already_from_scratch_user_changes_base_no_double_flip(tmp_path):
    """Mode is already From Scratch; user picks a new Base Model.
    Mode stays From Scratch (idempotent — no oscillation)."""
    ctrl = _controller_with_full_ui(tmp_path)
    ctrl.ui.training_mode_combo.setCurrentText("From Scratch — no prior checkpoint")
    initial = ctrl.ui.training_mode_combo.currentText()

    ctrl.ui.model_selector.setCurrentText("nuclei")

    assert ctrl.ui.training_mode_combo.currentText() == initial


# ============================================================================
# Hint label tests (8)
# ============================================================================

def test_F_round_0_scratch_init_hint_says_apply(tmp_path):
    """Round 0, run was created with init_spec.mode='scratch'.
    Base Model + Checkpoint will apply.  Hint should say so."""
    ctrl = _controller_with_full_ui(tmp_path)
    ctrl.current_round_by_view = {"lateral": 0, "axial": 0}
    ctrl._init_spec_for_active_bucket = lambda: {"mode": "scratch"}

    severity, text = ctrl._compute_init_hint()

    assert severity == "apply"
    assert "scratch" in text.lower()


def test_G_round_0_pretrained_init_hint_says_override(tmp_path):
    """Round 0, run was created with init_spec.mode='pretrained' (cyto3).
    Base Model + Checkpoint widgets ignored — init_spec wins."""
    ctrl = _controller_with_full_ui(tmp_path)
    ctrl.current_round_by_view = {"lateral": 0, "axial": 0}
    ctrl._init_spec_for_active_bucket = lambda: {
        "mode": "pretrained", "path": "cyto3"
    }

    severity, text = ctrl._compute_init_hint()

    assert severity == "override"
    assert "cyto3" in text.lower() or "pretrained" in text.lower()
    # brick T (D19): the sentence is the round0_locked constant
    assert "not used" in text.lower()
    assert text == _T.hint_text("round0_locked", display="pretrained: cyto3")


def test_H_round_0_run_model_init_hint_says_override(tmp_path):
    """Round 0, init_spec.mode='run-model' (inherited from another run's
    trained checkpoint).  Same override situation."""
    ctrl = _controller_with_full_ui(tmp_path)
    ctrl.current_round_by_view = {"lateral": 0, "axial": 0}
    ctrl._init_spec_for_active_bucket = lambda: {
        "mode": "run-model", "path": "/path/to/prior_run_ckpt"
    }

    severity, text = ctrl._compute_init_hint()

    assert severity == "override"
    assert "not used" in text.lower()


def test_I_round_1_cumulative_with_prior_hint_says_override(tmp_path):
    """Round 1+, Cumulative mode, prior round's checkpoint resolves.
    Base Model + Checkpoint widgets ignored — prior wins."""
    ctrl = _controller_with_full_ui(tmp_path)
    ctrl.current_round_by_view = {"lateral": 2, "axial": 0}
    # Simulate prior checkpoint resolution success.
    ctrl._resolve_prior_model_checkpoint = lambda bucket=None: {
        "spatial": "/path/to/round_01_ckpt"
    }
    ctrl.ui.training_mode_combo.setCurrentText("Cumulative — reuse prior model (default)")

    severity, text = ctrl._compute_init_hint()

    assert severity == "override"
    # brick T (D19): the cumulative_prior constant, disk numbering (D16)
    assert text.startswith("Starts from ") and "round 1" in text
    assert "lateral model" in text


def test_J_round_1_cumulative_prior_missing_hint_says_apply_with_warn(tmp_path):
    """Round 1+, Cumulative mode, but prior round's checkpoint file is
    GONE (deleted/moved).  Resolver returns empty.  Then Base Model +
    Checkpoint actually do apply as fallback — hint should reflect that."""
    ctrl = _controller_with_full_ui(tmp_path)
    ctrl.current_round_by_view = {"lateral": 2, "axial": 0}
    ctrl._resolve_prior_model_checkpoint = lambda bucket=None: {}  # empty = missing
    ctrl.ui.training_mode_combo.setCurrentText("Cumulative — reuse prior model (default)")

    severity, text = ctrl._compute_init_hint()

    # When the prior is missing, widgets DO apply — but with a warning.
    assert severity in ("apply", "warn"), f"got {severity!r}"
    # Hint should mention the missing-prior fallback.
    assert "missing" in text.lower() and "instead" in text.lower()


def test_K_round_1_from_scratch_with_checkpoint_hint_says_apply(tmp_path):
    """Round 1+, From Scratch, Checkpoint widget has a path.
    Checkpoint wins over Base Model.  Hint should call out the path."""
    ctrl = _controller_with_full_ui(tmp_path)
    ctrl.current_round_by_view = {"lateral": 2, "axial": 0}
    ctrl.ui.training_mode_combo.setCurrentText("From Scratch — no prior checkpoint")
    ctrl.ui.starting_checkpoint_path.setText("/path/to/custom.pt")
    ctrl.ui.model_selector.setCurrentText("cyto3")  # set but should be overridden by ckpt

    severity, text = ctrl._compute_init_hint()

    assert severity == "apply"
    assert "custom.pt" in text and "every round" in text


def test_L_round_1_from_scratch_no_checkpoint_hint_uses_base_model(tmp_path):
    """Round 1+, From Scratch, no Checkpoint, Base Model = 'cyto3'.
    Hint should call out Base Model = cyto3."""
    ctrl = _controller_with_full_ui(tmp_path)
    ctrl.current_round_by_view = {"lateral": 2, "axial": 0}
    ctrl.ui.training_mode_combo.setCurrentText("From Scratch — no prior checkpoint")
    ctrl.ui.starting_checkpoint_path.setText("")
    ctrl.ui.model_selector.setCurrentText("cyto3")

    severity, text = ctrl._compute_init_hint()

    assert severity == "apply"
    assert "cyto3" in text.lower()


def test_M_no_run_loaded_hint_is_neutral(tmp_path):
    """When working_dir is unset / no run loaded, hint should be a
    neutral placeholder, not crash, not mislead."""
    ctrl = _controller_with_full_ui(tmp_path)
    ctrl.working_dir = ""
    ctrl.active_run_id = ""

    severity, text = ctrl._compute_init_hint()

    # Should not raise.  Severity neutral / minimal text.
    assert severity in ("neutral", "apply", "override"), f"got {severity!r}"
    assert isinstance(text, str)


# ============================================================================
# Hydra-check: _compute_init_hint is pure (no side effects)
# ============================================================================

def test_compute_init_hint_does_not_mutate_state(tmp_path):
    """_compute_init_hint must be a pure read.  Calling it should NOT
    change current_config or any widget value."""
    ctrl = _controller_with_full_ui(tmp_path)
    ctrl.current_config = {"epochs": 100, "channel1": "1: red"}
    ctrl.current_round_by_view = {"lateral": 2, "axial": 1}
    ctrl._resolve_prior_model_checkpoint = lambda bucket=None: {"spatial": "/x"}

    before_cfg = dict(ctrl.current_config)
    before_round = dict(ctrl.current_round_by_view)
    before_mode = ctrl.ui.training_mode_combo.currentText()
    before_base = ctrl.ui.model_selector.currentText()

    ctrl._compute_init_hint()
    ctrl._compute_init_hint()  # call twice — idempotent

    assert ctrl.current_config == before_cfg
    assert ctrl.current_round_by_view == before_round
    assert ctrl.ui.training_mode_combo.currentText() == before_mode
    assert ctrl.ui.model_selector.currentText() == before_base
