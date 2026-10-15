"""Tests for model-source selection persistence (2026-05-29, fix/step2-model-state).

User-reported: set an Imported model (+ channels), exit, resume → the model is
gone and "segment does nothing", you must re-import.  Root cause being fixed
here: selecting the model source (`_on_model_source_changed`) and browsing an
imported checkpoint (`_browse_imported_model_checkpoint`) update the controller
state + apply it, but never schedule a state save — so the choice only survives
if some OTHER action (accept/stage/train/segment) happens to persist first.

These are source guards (the handlers are Qt-bound; behavioral exercise needs
widgets).  They assert the save is scheduled so the selection survives a
session restart.
"""
from __future__ import annotations

import inspect

from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
    TrainingAnnotationController,
)


def test_model_source_change_schedules_state_save():
    src = inspect.getsource(TrainingAnnotationController._on_model_source_changed)
    assert "_schedule_state_save" in src, (
        "_on_model_source_changed must persist the chosen model source so it "
        "survives exit/resume (else the user must re-pick it every session)"
    )


def test_browse_imported_model_schedules_state_save():
    src = inspect.getsource(TrainingAnnotationController._browse_imported_model_checkpoint)
    assert "_schedule_state_save" in src, (
        "_browse_imported_model_checkpoint must persist the imported path so it "
        "survives exit/resume"
    )


def test_inference_sync_guards_init_transparency():
    """Audit-found (2026-05-29): _sync_inference_controls_from_config must set
    the _init_transparency_loading guard while it programmatically sets the
    channel combos — otherwise their currentTextChanged -> _sync_channel_combos
    writes current_config + schedules a spurious save on every workspace open.
    Mirrors the guard already present in _sync_training_controls_from_config."""
    src = inspect.getsource(TrainingAnnotationController._sync_inference_controls_from_config)
    assert "_init_transparency_loading" in src, (
        "_sync_inference_controls_from_config must set the init-transparency "
        "guard so its programmatic channel-combo sync can't trigger a save"
    )


def test_restore_blocks_signals_on_model_source_combo():
    """Audit-found root cause: restore's combo.setCurrentText fires
    _on_model_source_changed, which re-reads the not-yet-populated imported-path
    widget and clobbers the restored path to "".  Restore must block signals
    around the model-source combo repopulation."""
    src = inspect.getsource(TrainingAnnotationController._load_internal_state)
    assert "blockSignals" in src, (
        "_load_internal_state must block the model-source combo's signals while "
        "repopulating it, so the change handler can't clobber the restored "
        "imported_curation_model_path mid-restore"
    )


# ---------------------------------------------------------------------------
# Channel persistence — changing the channel combo must reach current_config
# + schedule a save (else 'set channels -> exit without segmenting' loses them)
# ---------------------------------------------------------------------------


class _FakeCombo:
    def __init__(self, text=""):
        self._t = text

    def findText(self, t):
        return 0  # pretend present so _set_combo_text_if_present proceeds

    def setCurrentText(self, t):
        self._t = str(t)

    def currentText(self):
        return self._t


def _channel_fake():
    f = TrainingAnnotationController.__new__(TrainingAnnotationController)
    f._channel_sync_in_progress = False
    f._init_transparency_loading = False
    f.current_config = {}
    f._save_count = 0
    f._schedule_state_save = lambda *a, **k: setattr(f, "_save_count", f._save_count + 1)

    class _UI:
        pass

    f.ui = _UI()
    f.ui.inference_channel1_combo = _FakeCombo()
    f.ui.inference_channel2_combo = _FakeCombo()
    f.ui.chan1_combo = _FakeCombo()
    f.ui.chan2_combo = _FakeCombo()
    return f


def test_channel_change_writes_config_and_schedules_save():
    f = _channel_fake()
    TrainingAnnotationController._sync_channel_combos(f, "chan1_combo", "3: blue")
    assert f.current_config["channel1"] == "3: blue"
    assert f._save_count == 1
    TrainingAnnotationController._sync_channel_combos(f, "inference_channel2_combo", "0: none")
    assert f.current_config["channel2"] == "0: none"
    assert f._save_count == 2


def test_channel_change_skips_save_during_restore():
    """During restore (_init_transparency_loading) the handler must NOT save —
    the loaded state is already correct; saving here is redundant/racey."""
    f = _channel_fake()
    f._init_transparency_loading = True
    TrainingAnnotationController._sync_channel_combos(f, "chan1_combo", "3: blue")
    assert f._save_count == 0


# ---------------------------------------------------------------------------
# Training Checkpoint (starting_checkpoint) path persistence
# ---------------------------------------------------------------------------


def _ckpt_fake(loading=False):
    f = TrainingAnnotationController.__new__(TrainingAnnotationController)
    f._init_transparency_loading = loading
    f.current_config = {}
    f._save_count = 0
    f._schedule_state_save = lambda *a, **k: setattr(f, "_save_count", f._save_count + 1)

    class _UI:
        pass

    f.ui = _UI()
    f.ui.starting_checkpoint_status = None  # handler tolerates a missing label
    return f


def test_checkpoint_field_change_persists_and_saves():
    """Browsing/typing a training Checkpoint must write current_config +
    schedule a save so it survives exit/resume (used at train time)."""
    f = _ckpt_fake()
    TrainingAnnotationController._on_starting_checkpoint_changed(f, "/models/foo.model")
    assert f.current_config["starting_checkpoint"] == "/models/foo.model"
    assert f._save_count == 1


def test_checkpoint_field_skips_save_during_restore():
    f = _ckpt_fake(loading=True)
    TrainingAnnotationController._on_starting_checkpoint_changed(f, "/models/foo.model")
    assert f._save_count == 0


# ---------------------------------------------------------------------------
# Post-training UX (2026-05-29, feature/step2-new-run-model-prepopulate)
# ---------------------------------------------------------------------------


def test_post_training_flips_to_round_model_not_auto():
    """After training (auto-update on), the Model Source dropdown should flip
    to 'Round model' so the user VISIBLY sees the just-trained model is now
    active.  Previously flipped to 'Auto' which resolved-to-round correctly
    but gave no visible confirmation.

    Persistence then carries this 'round' choice through exit/resume so the
    next session also opens on 'Round model' (no more "looks like it's still
    on Auto after I trained")."""
    src = inspect.getsource(
        TrainingAnnotationController._promote_round_checkpoints_to_curation
    )
    assert '"Round model"' in src, (
        "_promote_round_checkpoints_to_curation must call "
        'combo.setCurrentText("Round model") so the dropdown flips '
        "to Round model after training (auto-update on)."
    )
    assert '"round"' in src, (
        "controller field must be set to 'round' (was 'auto')."
    )


def test_post_training_checkpoint_setText_is_guarded():
    """`_on_training_finished` auto-populates the Checkpoint widget with the
    just-trained round's path.  That `setText` fires
    `_on_init_widget_changed` which would otherwise auto-flip Training Mode
    from Cumulative to From Scratch — silently undoing the user's
    Cumulative choice every time they train.  Guard with
    `_init_transparency_loading` so the system-populate doesn't trigger
    the user-only auto-flip."""
    src = inspect.getsource(TrainingAnnotationController._on_training_finished)
    assert "_init_transparency_loading" in src, (
        "_on_training_finished's ckpt_edit.setText(latest_ckpt) must be "
        "wrapped in _init_transparency_loading guard — otherwise it fires "
        "_on_init_widget_changed and flips Training Mode out of Cumulative."
    )


def test_view_switch_checkpoint_setText_is_guarded():
    """`_update_training_ui_for_view` (fires on lateral<->axial view switch)
    auto-populates the Checkpoint widget with the new view's prior-round
    checkpoint.  Same auto-flip risk as the post-training path — guard with
    `_init_transparency_loading`."""
    src = inspect.getsource(TrainingAnnotationController._update_training_ui_for_view)
    assert "_init_transparency_loading" in src, (
        "_update_training_ui_for_view must wrap its setText calls in the "
        "_init_transparency_loading guard so view-switch doesn't silently "
        "flip Training Mode."
    )
