"""Tests for model output actions and loop closure (Phase 6).

Covers:
- "Use for Re-Prediction" sets correct model path(s) based on training mode
- "Register as Model Pack" creates model pack JSON
- Round tracking
"""

import json
import time
import pytest
from unittest.mock import MagicMock
from pathlib import Path

from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeLabel:
    def __init__(self, text=""):
        self._text = text
    def setText(self, t):
        self._text = t
    def text(self):
        return self._text
    def setStyleSheet(self, _s):
        pass


class FakeLineEdit:
    def __init__(self, text=""):
        self._text = text
    def text(self):
        return self._text
    def setText(self, t):
        self._text = t


class FakeComboBox:
    def __init__(self, items=None, idx=0):
        self._items = items or []
        self._idx = idx
    def currentText(self):
        return self._items[self._idx] if self._items else ""
    def setCurrentIndex(self, i):
        self._idx = i
    def currentIndex(self):
        return self._idx


class FakeButton:
    def __init__(self):
        self._enabled = True
    def setEnabled(self, v):
        self._enabled = v
    @property
    def clicked(self):
        return self
    def connect(self, slot):
        pass


class FakeLog:
    def __init__(self):
        self._lines = []
    def append(self, text):
        self._lines.append(text)


def _make_controller(tmp_path):
    """Create minimal controller with working directory."""
    ctrl = object.__new__(FusionReconstructionController)
    ctrl.working_dir = str(tmp_path)
    ctrl.loaded_stack_dir = str(tmp_path)

    ui = MagicMock()
    ui.lateral_model_path = FakeLineEdit()
    ui.axial_model_path = FakeLineEdit()
    ui.refinement_mode_combo = FakeComboBox([
        "Fine-tune Lateral", "Fine-tune Axial", "Fine-tune Both",
    ])
    ui.refinement_output_label = FakeLabel()
    ui.refinement_status_label = FakeLabel()
    ui.refine_pack_status_label = FakeLabel()
    ui.refine_model_name = FakeLineEdit()
    ui.log_output = FakeLog()
    # the foot's status line (Brick B2): the sentence lands on the label
    ui.show_refine_status = (
        lambda text, kind="neutral": ui.refinement_status_label.setText(text))

    ctrl.ui = ui
    ctrl._last_refined_model_dir = None
    return ctrl



def _both_run(tmp_path, monkeypatch, *, no_pairs=(), axial_fails=False,
              axial_empty=False):
    """Run "Fine-tune Both" through the REAL launch / finish / error path
    (harness: fake worker + QThread only) and return the controller.

    ``no_pairs``: pass modes whose pair collection is empty.
    ``axial_fails``: the axial worker reports an error instead of finishing.
    ``axial_empty``: the axial worker finishes with ``model_path=None`` (the
    worker's ``trainer.model_path`` fallback can be None).
    """
    from types import SimpleNamespace

    from tests.fusion_reconstruction._refine_launch_harness import (
        finish, make_launch_controller, run_timers,
    )

    ctrl, h = make_launch_controller(
        tmp_path, monkeypatch, name="round_0", no_pairs=no_pairs,
        ui=SimpleNamespace(
            lateral_model_path=FakeLineEdit(),
            axial_model_path=FakeLineEdit(),
            refinement_status_label=FakeLabel(),
            refine_pack_status_label=FakeLabel(),
            log_output=FakeLog(),
        ),
    )
    ctrl._last_refined_model_dir = None
    ctrl._launch_refinement()
    done = 0
    while done < len(h.workers):
        worker = h.workers[done]
        done += 1
        if worker.bucket == "axial" and axial_fails:
            ctrl._on_refinement_error("boom")
        elif worker.bucket == "axial" and axial_empty:
            ctrl._on_refinement_finished({
                "model_path": None,
                "model_name": "axial",
                "bundle_name": worker.bundle_name,
                "bucket": "axial",
                "final_train_loss": None,
                "train_losses": [],
            })
        else:
            finish(ctrl, worker)
        run_timers(h)  # the deferred launch of the next queued pass
    return ctrl


# ---------------------------------------------------------------------------
# Tests: Use for Re-Prediction
# ---------------------------------------------------------------------------

class TestUseForRePrediction:
    """Test that refined model paths propagate to prediction fields."""

    def test_lateral_mode_sets_lateral_path(self, tmp_path):
        ctrl = _make_controller(tmp_path)
        model_dir = tmp_path / "artifacts" / "step3" / "refinement" / "models" / "round_0"
        model_dir.mkdir(parents=True)
        lateral_model = model_dir / "lateral_model"
        lateral_model.touch()
        ctrl._last_refined_model_dir = str(model_dir)
        ctrl._last_refinement_mode = "Fine-tune Lateral"

        ctrl._use_for_re_prediction()

        assert str(lateral_model) in ctrl.ui.lateral_model_path.text()
        # Axial should be unchanged (empty)
        assert ctrl.ui.axial_model_path.text() == ""

    def test_axial_mode_sets_axial_path(self, tmp_path):
        ctrl = _make_controller(tmp_path)
        model_dir = tmp_path / "artifacts" / "step3" / "refinement" / "models" / "round_0"
        model_dir.mkdir(parents=True)
        axial_model = model_dir / "axial_model"
        axial_model.touch()
        ctrl._last_refined_model_dir = str(model_dir)
        ctrl._last_refinement_mode = "Fine-tune Axial"

        ctrl._use_for_re_prediction()

        assert ctrl.ui.lateral_model_path.text() == ""
        assert str(axial_model) in ctrl.ui.axial_model_path.text()

    def test_both_mode_sets_both_paths(self, tmp_path, monkeypatch):
        ctrl = _both_run(tmp_path, monkeypatch)

        assert ctrl._last_refinement_mode == "Fine-tune Both"
        ctrl._use_for_re_prediction()

        assert "lateral" in ctrl.ui.lateral_model_path.text().lower()
        assert "axial" in ctrl.ui.axial_model_path.text().lower()

    def test_no_model_dir_shows_error(self, tmp_path):
        ctrl = _make_controller(tmp_path)
        ctrl._last_refined_model_dir = None
        ctrl._last_refinement_mode = "Fine-tune Lateral"

        ctrl._use_for_re_prediction()

        assert "no refined model" in ctrl.ui.refinement_status_label.text().lower()


# ---------------------------------------------------------------------------
# Tests: a "Both" run that trained only ONE bucket records only that bucket
# ---------------------------------------------------------------------------

class TestPartialBothRun:
    """"Fine-tune Both" is two passes; if only one of them produced a model,
    the recorded mode must name that one bucket.  Recording "Both" made
    "Use for Re-Prediction" (and the model pack) also take the OTHER
    bucket's file -- on a re-used, overwritten bundle that is the OLD
    model, silently paired with the new one.
    """

    def test_axial_errors_after_lateral(self, tmp_path, monkeypatch):
        ctrl = _both_run(tmp_path, monkeypatch, axial_fails=True)
        assert ctrl._last_refinement_mode == "Fine-tune Lateral"

    def test_axial_finishes_with_no_model(self, tmp_path, monkeypatch):
        """The last pass finishes but reports no model: the record must
        name the bucket that DID train, not the pass that produced nothing."""
        ctrl = _both_run(tmp_path, monkeypatch, axial_empty=True)
        assert ctrl._last_refinement_mode == "Fine-tune Lateral"

    def test_axial_has_no_pairs(self, tmp_path, monkeypatch):
        ctrl = _both_run(tmp_path, monkeypatch, no_pairs=("Fine-tune Axial",))
        assert ctrl._last_refinement_mode == "Fine-tune Lateral"

    def test_lateral_has_no_pairs_axial_trains(self, tmp_path, monkeypatch):
        ctrl = _both_run(
            tmp_path, monkeypatch, no_pairs=("Fine-tune Lateral",),
        )
        assert ctrl._last_refinement_mode == "Fine-tune Axial"

    def test_old_axial_model_is_not_picked_up(self, tmp_path, monkeypatch):
        """The owner's case: a typed name re-used (Overwrite), the bundle
        still holds an OLD axial model, and this run's axial pass fails."""
        from napari_epicycle.fusion_reconstruction import _paths

        old = _paths.refined_model_dir(tmp_path, "round_0") / "axial_model"
        old.parent.mkdir(parents=True)
        old.touch()
        ctrl = _both_run(tmp_path, monkeypatch, axial_fails=True)

        ctrl._use_for_re_prediction()

        assert "lateral_model" in ctrl.ui.lateral_model_path.text()
        assert ctrl.ui.axial_model_path.text() == "", (
            "the OLD axial model must not be paired with the new lateral"
        )
