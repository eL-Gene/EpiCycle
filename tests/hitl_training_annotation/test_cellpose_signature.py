"""Release-hardening #10 — cellpose API tripwire.

`CellposeTrainer.fit` is the only caller of `cellpose.train.train_seg`,
and every existing test mocks `fit` away — so a cellpose upgrade that
renames or drops a parameter is discovered minutes into the user's first
real training round, after the dataset has been prepared and the UI
locked.  These tests bind the REAL kwargs dicts (the same builders `fit`
uses) against the installed cellpose signatures.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from packages.epicycle_training.cellpose_trainer import (
    build_cellpose_model_kwargs,
    build_train_seg_kwargs,
)
from packages.epicycle_training.config import TrainingConfig


def _bind(fn, kwargs):
    """bind_partial raises TypeError on an unexpected/renamed kwarg."""
    inspect.signature(fn).bind_partial(**kwargs)


def test_train_seg_accepts_every_kwarg_fit_passes():
    cellpose_train = pytest.importorskip("cellpose.train")
    kwargs = build_train_seg_kwargs(
        net=object(),
        config=TrainingConfig(n_epochs=1),
        train_data=[np.zeros((2, 8, 8), dtype=np.float32)],
        train_labels=[np.zeros((8, 8), dtype=np.int32)],
        save_path="/tmp/x",
        model_name="w",
    )
    _bind(cellpose_train.train_seg, kwargs)


@pytest.mark.parametrize(
    "builder_kwargs",
    [
        {"use_gpu": False, "pretrained_model": "/tmp/ckpt"},
        {"use_gpu": False, "base_model": "cyto3", "n_channels": 2},
        {"use_gpu": False},
    ],
    ids=["custom-checkpoint", "built-in-model", "scratch"],
)
def test_cellpose_model_accepts_every_kwarg_fit_passes(builder_kwargs):
    cellpose_models = pytest.importorskip("cellpose.models")
    _bind(
        cellpose_models.CellposeModel.__init__,
        build_cellpose_model_kwargs(**builder_kwargs),
    )


def test_fit_passes_the_builder_dicts_straight_through(tmp_path, monkeypatch):
    """`fit` must CALL the builders, so the bind tests above cover reality.

    Behavioural, not source-text: both builders are replaced with sentinel
    dicts and we assert cellpose received exactly those.
    """
    models = pytest.importorskip("cellpose.models")
    train = pytest.importorskip("cellpose.train")
    import packages.epicycle_training.cellpose_trainer as ct

    seen = {}

    class _FakeModel:
        def __init__(self, **kwargs):
            seen["model"] = kwargs
            self.net = object()

    def _fake_train_seg(**kwargs):
        seen["train_seg"] = kwargs
        return str(tmp_path / "w"), np.array([0.5]), None

    monkeypatch.setattr(models, "CellposeModel", _FakeModel)
    monkeypatch.setattr(train, "train_seg", _fake_train_seg)
    monkeypatch.setattr(
        ct, "build_cellpose_model_kwargs", lambda **k: {"sentinel_model": 1}
    )
    monkeypatch.setattr(
        ct, "build_train_seg_kwargs", lambda **k: {"sentinel_train": 2}
    )

    trainer = ct.CellposeTrainer(config=TrainingConfig(n_epochs=1), use_gpu=False)
    trainer.fit(
        train_data=[np.zeros((2, 8, 8), dtype=np.float32)],
        train_labels=[np.zeros((8, 8), dtype=np.int32)],
        model_name=None,
        save_path=None,
    )

    assert seen["model"] == {"sentinel_model": 1}
    assert seen["train_seg"] == {"sentinel_train": 2}
