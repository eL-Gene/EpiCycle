"""Phase B — Tests for the TrainingHistoryDialog shell.

Uses minimal Qt-free verification: we don't instantiate the QDialog
(QApplication isn't running in headless test mode) but we do verify
the rendering helpers that turn ``RoundHistoryEntry`` lists into
display strings.

Phase B of the training-history plan.
"""
from __future__ import annotations

from napari_epicycle.hitl_training_annotation._training_history import (
    RoundHistoryEntry,
)


def test_format_round_summary_inherited_round():
    """A single inherited round renders cleanly."""
    from napari_epicycle.hitl_training_annotation._training_history_dialog import (
        format_round_summary,
    )
    entry = RoundHistoryEntry(
        bucket="lateral",
        round_idx=2,
        init_mode="inherited",
        init_path="/some/round_01/model/nuclei_lateral_round_01",
        channels_used=["membrane", "nuclei"],
        n_train_pairs=12,
        n_epochs=200,
        finished_at="2026-05-27T14:30:00+00:00",
    )
    s = format_round_summary(entry)
    assert "round_02" in s
    assert "inherited" in s
    assert "membrane" in s or "nuclei" in s  # channels rendered
    assert "12" in s  # pair count


def test_format_round_summary_scratch_round():
    """Scratch init renders without an init_path."""
    from napari_epicycle.hitl_training_annotation._training_history_dialog import (
        format_round_summary,
    )
    entry = RoundHistoryEntry(
        bucket="lateral",
        round_idx=0,
        init_mode="scratch",
        init_path=None,
        channels_used=["membrane", "none"],
        n_train_pairs=4,
        n_epochs=200,
    )
    s = format_round_summary(entry)
    assert "round_00" in s
    assert "scratch" in s.lower()
    assert "4" in s


def test_format_history_empty_bucket():
    """Empty history list renders a placeholder."""
    from napari_epicycle.hitl_training_annotation._training_history_dialog import (
        format_history_for_bucket,
    )
    s = format_history_for_bucket([], bucket="lateral")
    assert "no" in s.lower() or "—" in s or "empty" in s.lower()


def test_format_history_multi_round():
    """Multi-round history renders rounds in order, with lineage hint."""
    from napari_epicycle.hitl_training_annotation._training_history_dialog import (
        format_history_for_bucket,
    )
    entries = [
        RoundHistoryEntry(bucket="lateral", round_idx=0, init_mode="scratch",
                          n_train_pairs=4),
        RoundHistoryEntry(bucket="lateral", round_idx=1, init_mode="inherited",
                          init_path="...", n_train_pairs=8),
        RoundHistoryEntry(bucket="lateral", round_idx=2, init_mode="pretrained",
                          init_path="cyto3", n_train_pairs=10),
    ]
    s = format_history_for_bucket(entries, bucket="lateral")
    assert "round_00" in s
    assert "round_01" in s
    assert "round_02" in s
    # The pretrained round has a broken lineage — should be marked
    # somehow in the rendering (test for either "broken", "★", "↺", etc.)
    # We accept any of several markers since the rendering style is flexible.
    assert any(marker in s.lower() for marker in (
        "broken", "restart", "new lineage", "from cyto3", "↺", "reset",
    ))


def test_dialog_class_importable():
    """The dialog class is importable from the module (Qt construction
    deferred — test doesn't instantiate)."""
    from napari_epicycle.hitl_training_annotation._training_history_dialog import (
        TrainingHistoryDialog,
    )
    # Just verify the class exists and has the expected signature.
    assert hasattr(TrainingHistoryDialog, "__init__")
    # Verify it requires workspace + run_id at construction.
    import inspect
    params = inspect.signature(TrainingHistoryDialog.__init__).parameters
    assert "workspace" in params
    assert "run_id" in params
