"""Release-hardening #7 — val loss must be absent, not zero.

The Step 2 adapter never passes `val_data`, so cellpose returns an
all-zeros test-loss array.  Pre-fix that array was saved as
`{model}_val_loss.npy` and surfaced in the manifest as
`final_loss.val = 0.0` — which reads as a perfect validation score.
"""

from __future__ import annotations

import numpy as np
import pytest

from packages.epicycle_training.cellpose_trainer import CellposeTrainer
from packages.epicycle_training.config import TrainingConfig


class _FakeNet:
    pass


class _FakeModel:
    def __init__(self, *a, **k):
        self.net = _FakeNet()


def _install_fakes(monkeypatch, tmp_path, *, returned_val):
    cellpose = pytest.importorskip("cellpose")
    from cellpose import models, train

    monkeypatch.setattr(models, "CellposeModel", _FakeModel)

    def _fake_train_seg(**kwargs):
        out = tmp_path / "models"
        out.mkdir(parents=True, exist_ok=True)
        return str(out / "w"), np.array([0.9, 0.4]), returned_val

    monkeypatch.setattr(train, "train_seg", _fake_train_seg)
    return cellpose


def _fit(tmp_path, **kwargs):
    trainer = CellposeTrainer(config=TrainingConfig(n_epochs=2), use_gpu=False)
    trainer.fit(
        train_data=[np.zeros((2, 8, 8), dtype=np.float32)],
        train_labels=[np.zeros((8, 8), dtype=np.int32)],
        model_name="w",
        save_path=str(tmp_path),
        **kwargs,
    )
    return trainer


def test_val_loss_is_none_when_no_val_data(tmp_path, monkeypatch):
    _install_fakes(monkeypatch, tmp_path, returned_val=np.zeros(2))
    trainer = _fit(tmp_path)

    assert trainer.val_loss is None, (
        "cellpose returns zeros when no validation set was supplied; "
        "reporting them as a val loss is a lie"
    )
    assert not (tmp_path / "models" / "w_val_loss.npy").exists()

    meta = (tmp_path / "models" / "w_training_config.json")
    if meta.exists():
        import json
        assert json.loads(meta.read_text(encoding="utf-8"))["final_val_loss"] is None


def test_val_loss_kept_when_val_data_supplied(tmp_path, monkeypatch):
    _install_fakes(monkeypatch, tmp_path, returned_val=np.array([0.7, 0.5]))
    trainer = _fit(
        tmp_path,
        val_data=([np.zeros((2, 8, 8), dtype=np.float32)],
                  [np.zeros((8, 8), dtype=np.int32)]),
    )

    assert trainer.val_loss is not None
    assert list(trainer.val_loss) == [0.7, 0.5]
    assert (tmp_path / "models" / "w_val_loss.npy").exists()
