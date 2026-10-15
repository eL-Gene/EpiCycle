"""Model Refinement look, Brick B2 -- the loss-plot feed.

Cellpose computes validation at epoch 5 and every 10th epoch (``0.0``
elsewhere and when there is no validation set), so the live series and the
worker's ``val_losses`` take SAMPLED epochs of a run WITH a validation set
only; ``set_palette`` is called once at construction; the finished pass hands
``add_round_curve`` its sparse validation pairs.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from napari_epicycle.fusion_reconstruction import _step3_mr_text as M
from napari_epicycle.fusion_reconstruction import controller as cmod
from napari_epicycle.ui_shell import theme
from tests.fusion_reconstruction._refine_launch_harness import (
    make_launch_controller,
)


@pytest.mark.parametrize("epoch,sampled", [
    (0, True), (1, False), (4, False), (5, True), (6, False), (9, False),
    (10, True), (15, False), (20, True), (95, False), (100, True)])
def test_the_sampling_rule(epoch, sampled):
    assert M.is_val_sampled_epoch(epoch) is sampled


def _live(tmp_path, monkeypatch, n_val):
    plot = MagicMock()
    ui = SimpleNamespace(refinement_loss_plot=plot)
    ctrl, h = make_launch_controller(
        tmp_path, monkeypatch, ui=ui, mode="Fine-tune Lateral")
    ctrl._append_log = lambda _m: None
    ctrl._launch_refinement()
    ctrl._refinement_n_val = n_val
    plot.reset_mock()
    return ctrl, plot


def test_live_val_only_on_a_sampled_epoch_with_a_validation_set(
        tmp_path, monkeypatch):
    ctrl, plot = _live(tmp_path, monkeypatch, n_val=6)
    for epoch in (0, 4, 5, 6, 10, 11):
        ctrl._on_refinement_epoch(epoch, 100, 0.5, 0.7)
    got = {c.args[0]: c.kwargs["val_loss"]
           for c in plot.append_live_point.call_args_list}
    assert got == {0: 0.7, 4: None, 5: 0.7, 6: None, 10: 0.7, 11: None}


def test_no_validation_set_never_draws_a_val_point(tmp_path, monkeypatch):
    ctrl, plot = _live(tmp_path, monkeypatch, n_val=0)
    for epoch in (0, 5, 10, 20):
        ctrl._on_refinement_epoch(epoch, 100, 0.5, 0.0)
    assert all(c.kwargs["val_loss"] is None
               for c in plot.append_live_point.call_args_list)
    assert ctrl._refinement_last_val is None


def test_finished_hands_the_sparse_val_pairs_to_the_plot(
        tmp_path, monkeypatch):
    ctrl, plot = _live(tmp_path, monkeypatch, n_val=4)
    from pathlib import Path
    worker = ctrl._refinement_worker
    weights = Path(worker.save_path) / "lateral_model"
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.touch()
    ctrl._on_refinement_finished({
        "model_path": str(weights), "model_name": "lateral",
        "bundle_name": worker.bundle_name, "bucket": "lateral",
        "final_train_loss": 0.1, "n_train": 4,
        "train_losses": [0.3, 0.2], "val_losses": [(5, 0.4), (10, 0.35)],
    })
    key = f"{worker.bundle_name}/lateral"
    plot.add_round_curve.assert_called_once_with(
        key, [0.3, 0.2], [(5, 0.4), (10, 0.35)])


def test_set_palette_is_called_once_at_construction():
    ctrl = cmod.FusionReconstructionController.__new__(
        cmod.FusionReconstructionController)
    ctrl.ui = MagicMock()
    ctrl.viewer = MagicMock()
    try:
        ctrl._connect_signals()
    except Exception as exc:            # a bare controller lacks other wiring
        print("connect_signals stopped late:", repr(exc))
    plot = ctrl.ui.refinement_loss_plot
    plot.set_palette.assert_called_once_with(theme.STEP3_PLOT_PALETTE)


class _FakeTrainer:
    """Stands in for CellposeTrainer: calls the epoch callback for 31 epochs
    like cellpose's log hook does (every epoch here), val 0.0 off-sample."""
    def __init__(self, config=None, use_gpu=False):
        self.train_loss = np.array([0.5, 0.4])
        self.val_loss = np.array([0.0, 0.3])
        self.model_path = "x"

    def fit(self, *, epoch_callback=None, **_kw):
        for e in range(31):
            sampled = M.is_val_sampled_epoch(e)
            epoch_callback(e, 31, 1.0 / (e + 1), 0.9 if sampled else 0.0)


@pytest.mark.parametrize("n_val,expected", [
    (3, [(0, 0.9), (5, 0.9), (10, 0.9), (20, 0.9), (30, 0.9)]),
    (0, []),
])
def test_the_worker_collects_sparse_val_pairs(monkeypatch, n_val, expected):
    import tifffile
    import packages.epicycle_training.cellpose_trainer as ct
    monkeypatch.setattr(tifffile, "imread", lambda p: np.zeros((4, 4)))
    monkeypatch.setattr(ct, "CellposeTrainer", _FakeTrainer)
    w = cmod._RefinementTrainingWorker(
        {"n_epochs": 31, "learning_rate": 0.01, "batch_size": 2},
        ["a", "b"], ["a", "b"], ["v"] * n_val, ["v"] * n_val,
        "scratch", None, "lateral", "unused")
    out = []
    w.finished.connect(out.append)
    w.error.connect(lambda m: out.append({"error": m}))
    w.run()
    assert out and "error" not in out[0], out
    assert out[0]["val_losses"] == expected
    assert out[0]["n_val"] == n_val
