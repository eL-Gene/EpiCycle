"""Shared harness: drive Step 3 fine-tuning launches with no Qt thread.

The REAL ``_launch_refinement`` / ``_launch_next_refinement`` /
``_on_refinement_finished`` / ``_on_refinement_error`` /
``_restore_launch_button`` run; only the training worker, its QThread,
the button icon and the zero-delay timer are faked.  The timer records its
callable so a test runs the deferred second pass of "Both" by hand
(``run_timers``), the way the event loop would.

Used by test_refinement_collision.py and test_loop_closure.py.
"""

from __future__ import annotations

import datetime as _dt_mod
from types import SimpleNamespace
from unittest.mock import MagicMock

from napari_epicycle.fusion_reconstruction import _paths


class _FakeSignal:
    def connect(self, *_a, **_k):
        pass


class _Combo:
    def __init__(self, text):
        self._t = text

    def currentText(self):
        return self._t


class _LineEdit:
    def __init__(self, text=""):
        self._t = text

    def text(self):
        return self._t

    def setText(self, t):
        self._t = t


def make_launch_controller(
    tmp_path, monkeypatch, *, mode="Fine-tune Both", name="",
    no_pairs=(), ui=None,
):
    """A bare controller whose launch path runs for real.

    ``no_pairs``: pass modes ("Fine-tune Lateral" / "Fine-tune Axial")
    whose pair collection comes back empty.  ``ui``: an object whose
    attributes override the MagicMock UI's.
    """
    import qtpy.QtCore as _qtcore

    from napari_epicycle.fusion_reconstruction import controller as cmod
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )

    h = SimpleNamespace(workers=[], timers=[])

    # A clock that moves one second per call, so two calls never share a
    # timestamp.  ``h.clock_reads[0]`` counts the reads.
    _real_datetime = _dt_mod.datetime
    h.clock_reads = [0]

    class _FakeDateTime:
        @classmethod
        def now(cls, *a, **k):
            h.clock_reads[0] += 1
            return _real_datetime(2026, 10, 5, 12, 0, h.clock_reads[0])

    monkeypatch.setattr(_dt_mod, "datetime", _FakeDateTime)

    class _FakeWorker:
        def __init__(self, *args, **kwargs):
            self.bundle_name = kwargs.get("bundle_name")
            self.bucket = kwargs.get("bucket")
            self.save_path = args[8]
            for sig in ("epoch_update", "finished", "error", "cancelled", "log"):
                setattr(self, sig, _FakeSignal())
            h.workers.append(self)

        def moveToThread(self, _t):
            pass

        def run(self):
            pass

    class _FakeThread:
        def __init__(self, *a, **k):
            self.started = _FakeSignal()
            self.finished = _FakeSignal()

        def start(self):
            pass

        def quit(self):
            pass

    class _Timer:
        @staticmethod
        def singleShot(_ms, fn):
            h.timers.append(fn)

    monkeypatch.setattr(cmod, "_RefinementTrainingWorker", _FakeWorker)
    monkeypatch.setattr(cmod, "QThread", _FakeThread)
    monkeypatch.setattr(cmod, "apply_button_icon", lambda *a, **k: True)
    monkeypatch.setattr(_qtcore, "QTimer", _Timer)

    ctrl = object.__new__(FusionReconstructionController)
    ctrl.working_dir = str(tmp_path)
    base = MagicMock()
    base.refinement_mode_combo = _Combo(mode)
    base.refine_init_combo = _Combo("From scratch")
    base.refine_model_name = _LineEdit(name)
    for attr, value in vars(ui or SimpleNamespace()).items():
        setattr(base, attr, value)
    ctrl.ui = base
    ctrl._run_epoch = 0
    ctrl._refinement_queue = []
    ctrl._refresh_step3_lock = lambda: None
    ctrl._check_extraction_staleness = lambda: set()
    pairs = [str(tmp_path / f"p{i}_img.tif") for i in range(4)]

    def _collect(pass_mode):
        if pass_mode in no_pairs:
            return [], []
        return list(pairs), list(pairs)

    ctrl._collect_training_pairs = _collect
    ctrl._read_split_config = lambda: dict(
        train_frac=0.5, spatial_split=False, chunk_size=1,
    )
    ctrl._build_training_config = lambda: {"n_epochs": 2}
    ctrl._resolve_init_mode_for_bucket = lambda bucket: ("scratch", None)
    ctrl._populate_refinement_source_combo = lambda: None

    _paths.refinement_pairs_root(tmp_path).mkdir(parents=True, exist_ok=True)
    return ctrl, h


def finish(ctrl, worker):
    """The worker's ``finished``: write its weights file, then hand the
    controller the result the real worker would."""
    from pathlib import Path

    weights = Path(worker.save_path) / f"{worker.bucket}_model"
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.touch()
    ctrl._on_refinement_finished({
        "model_path": str(weights),
        "model_name": worker.bucket,
        "bundle_name": worker.bundle_name,
        "bucket": worker.bucket,
        "final_train_loss": 0.1,
        "train_losses": [],
    })


def run_timers(h):
    """Run the deferred callables (the event loop's zero-delay timers)."""
    while h.timers:
        h.timers.pop(0)()
