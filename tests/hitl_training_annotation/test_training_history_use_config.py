"""Tests for the "Use this config" wiring — click a round in the
training-history timeline, get its training params restored to the
widgets via _sync_training_controls_from_config.

Written tests-first (a failing test before the fix).
"""
from __future__ import annotations

from types import SimpleNamespace

from napari_epicycle.hitl_training_annotation._training_history import (
    RoundHistoryEntry,
)

from tests.hitl_training_annotation.controller.test_state_persistence_integration import (
    _StubSpinBox, _StubCheckBox, _StubLineEdit, _StubComboBox,
    _StubSignal,
)


def _make_controller_stub(tmp_path):
    """Stub controller with the widgets `_sync_training_controls_from_config`
    pushes into.  Same shape as test_state_persistence_integration but
    minimal (only the writable widgets)."""
    from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
        TrainingAnnotationController,
    )
    from napari_epicycle.io_registry import HITLPersistenceService

    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.persistence = HITLPersistenceService()
    ctrl.working_dir = str(tmp_path)
    ctrl.active_run_id = "default"
    ctrl.current_config = {}
    ctrl.current_round = 0
    ctrl.current_round_by_view = {"lateral": 0, "axial": 0}
    ctrl.selected_view = "XY"
    ctrl.view_target = "lateral"
    ctrl.viewer = None
    ctrl.loaded_patch_name = ""

    ctrl.ui = SimpleNamespace(
        epochs_sb=_StubSpinBox(500),
        batch_sb=_StubSpinBox(8),
        device_gpu=_StubCheckBox(True),
        device_cpu=_StubCheckBox(False),
        model_selector=_StubComboBox(
            ["scratch", "cyto3", "nuclei"], "scratch"
        ),
        starting_checkpoint_path=_StubLineEdit(""),
        lr_db=_StubSpinBox(0.01),
        wd_db=_StubSpinBox(0.0001),
        minmask_sb=_StubSpinBox(5),
        chan1_combo=_StubComboBox(
            ["0: gray", "1: red", "2: green", "3: blue"], "2: green"
        ),
        chan2_combo=_StubComboBox(
            ["0: none", "1: red", "2: green", "3: blue"], "1: red"
        ),
        norm_cb=_StubCheckBox(True),
        rescale_cb=_StubCheckBox(True),
        flow_cb=_StubCheckBox(True),
        # Inference (sync targets — _set_combo_text_if_present needs them)
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
        _init_hint_label=_StubLineEdit(""),
    )
    ctrl._append_ui_log = lambda *a, **kw: None
    ctrl._channel_sync_in_progress = False
    ctrl._init_transparency_loading = False
    return ctrl


# ============================================================================
# Tests for the new _apply_history_entry_to_widgets method
# ============================================================================

def test_apply_history_entry_restores_basic_widgets(tmp_path):
    """Calling _apply_history_entry_to_widgets with an entry should
    set epochs / channels via _sync_training_controls_from_config path."""
    ctrl = _make_controller_stub(tmp_path)
    entry = RoundHistoryEntry(
        bucket="lateral",
        round_idx=2,
        init_mode="inherited",
        init_path="/path/to/round_01/model/x",
        channels_used=["membrane", "nuclei"],
        n_train_pairs=12,
        n_epochs=300,
        weights_filename="nuclei_lateral_round_02",
    )

    ctrl._apply_history_entry_to_widgets(entry)

    assert ctrl.ui.epochs_sb.value() == 300


def test_apply_history_entry_with_scratch_init_widgets_apply(tmp_path):
    """Scratch init → Mode flips to From Scratch (since the entry's
    init lineage was scratch).  Base Model + Checkpoint from manifest
    can apply."""
    ctrl = _make_controller_stub(tmp_path)
    entry = RoundHistoryEntry(
        bucket="lateral",
        round_idx=0,
        init_mode="scratch",
        init_path=None,
        channels_used=["membrane", "none"],
        n_epochs=200,
    )

    ctrl._apply_history_entry_to_widgets(entry)

    assert ctrl.ui.epochs_sb.value() == 200


def test_apply_history_entry_with_inherited_init_keeps_cumulative_mode(tmp_path):
    """Inherited mode → keep / set Mode to Cumulative (the entry was
    trained that way).  This makes the user's next click of Train
    reproduce the same inheritance chain."""
    ctrl = _make_controller_stub(tmp_path)
    # Pre-set Mode to From Scratch so we can verify the apply switches it.
    ctrl.ui.training_mode_combo = _StubComboBox(
        ["Cumulative — reuse prior model (default)",
         "From Scratch — no prior checkpoint"],
        "From Scratch — no prior checkpoint",
    )
    entry = RoundHistoryEntry(
        bucket="lateral",
        round_idx=2,
        init_mode="inherited",
        init_path="/path/to/round_01/model/x",
        n_epochs=200,
    )

    ctrl._apply_history_entry_to_widgets(entry)

    assert "cumulative" in ctrl.ui.training_mode_combo.currentText().lower()


def test_apply_history_entry_does_not_corrupt_current_config(tmp_path):
    """current_config gets the entry's training params merged in but
    non-widget keys (e.g. pretrained_model_path) are preserved."""
    ctrl = _make_controller_stub(tmp_path)
    ctrl.current_config["pretrained_model_path"] = "/should/survive"
    entry = RoundHistoryEntry(
        bucket="lateral", round_idx=1, init_mode="inherited",
        n_epochs=150,
    )

    ctrl._apply_history_entry_to_widgets(entry)

    # The merged key survives.
    assert ctrl.current_config.get("pretrained_model_path") == "/should/survive"
    # And the entry's value is now in cfg.
    assert ctrl.current_config.get("epochs") == 150
