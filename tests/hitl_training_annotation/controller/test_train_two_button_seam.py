"""Seam (a) end to end (audit M4): ``_train_model`` in sync mode with a UI that
has the two buttons, through finish, error and cancel -- the state of Train
and Cancel DURING the round and after it, and hook 3's order against
``begin_live_round``.  Qt-free fakes."""
import numpy as np
import pytest

from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
    TrainingAbortedError,
)
from napari_epicycle.hitl_training_annotation.ui import _step2_text as T

from .test_training_round_flow import _ListWidget, _make_train_ctrl


class _Btn:
    def __init__(self, text=""):
        self._enabled = True
        self._text = text

    def setEnabled(self, v):
        self._enabled = bool(v)

    def isEnabled(self):
        return self._enabled

    def setText(self, t):
        self._text = t

    def text(self):
        return self._text


class _Plot:
    def __init__(self, events):
        self.events = events

    def begin_live_round(self, rid):
        self.events.append(("begin_live_round", rid))

    def finish_live_round(self):
        self.events.append(("finish_live_round",))

    def add_round_curve(self, *a, **k):
        self.events.append(("add_round_curve",))

    def cancel_live_round(self):
        self.events.append(("cancel_live_round",))

    def append_live_point(self, *a, **k):
        pass


def _ctrl(tmp_path):
    ctrl = _make_train_ctrl(tmp_path)
    xy = "stack01_XY_12"
    ctrl.segmentation_manager.set(
        xy, np.random.rand(2, 16, 16).astype(np.float32),
        np.ones((16, 16), dtype=np.uint16))
    ctrl._touch_sample(xy, status="staged", view="XY")
    ctrl.ui.data_to_add_list = _ListWidget([xy])
    ctrl.ui.update_and_train_button = _Btn("Train")
    ctrl.ui.train_cancel_btn = _Btn(T.CANCEL_BTN_TEXT)
    ctrl.ui.train_cancel_btn.setEnabled(False)       # born disabled
    events = []
    ctrl.ui.loss_plot = _Plot(events)
    ctrl.ui.on_train_started = lambda bucket, name: events.append(
        ("on_train_started", bucket, name))
    ctrl._events = events
    ctrl._during = []
    real = ctrl._run_training_work

    def spy(*a, **k):
        ctrl._during.append((ctrl.ui.update_and_train_button.isEnabled(),
                             ctrl.ui.train_cancel_btn.isEnabled()))
        if getattr(ctrl, "_raise", None) is not None:
            raise ctrl._raise
        return real(*a, **k)

    ctrl._run_training_work = spy
    return ctrl


def _idle(ctrl):
    return (ctrl.ui.update_and_train_button.isEnabled(),
            ctrl.ui.train_cancel_btn.isEnabled(),
            ctrl.ui.train_cancel_btn.text(),
            ctrl.ui.update_and_train_button.text())


def test_a_finished_round_greys_train_then_restores_both(tmp_path):
    ctrl = _ctrl(tmp_path)
    ctrl._train_model()
    assert ctrl._during == [(False, True)]            # Train off, Cancel lit
    assert _idle(ctrl) == (True, False, T.CANCEL_BTN_TEXT, "Train")
    assert ctrl.current_round == 1                    # it really trained


def test_an_error_restores_both_buttons(tmp_path):
    ctrl = _ctrl(tmp_path)
    ctrl._raise = RuntimeError("boom")
    ctrl._train_model()
    assert ctrl._during == [(False, True)]
    assert _idle(ctrl) == (True, False, T.CANCEL_BTN_TEXT, "Train")
    assert ctrl.current_round == 0                    # nothing was kept


def test_a_cancel_restores_both_buttons_and_the_cancel_word(tmp_path):
    ctrl = _ctrl(tmp_path)
    ctrl._raise = TrainingAbortedError("stop")
    ctrl._train_model()
    assert ctrl._during == [(False, True)]
    assert _idle(ctrl) == (True, False, T.CANCEL_BTN_TEXT, "Train")


def test_cancel_says_cancelling_and_greys_until_the_round_ends(tmp_path):
    ctrl = _ctrl(tmp_path)

    class _Thread:
        def isRunning(self):
            return True

    class _Flag:
        def __init__(self):
            self.n = 0

        def set(self):
            self.n += 1

        def is_set(self):
            return self.n > 0

    ctrl._training_thread = _Thread()
    ctrl._training_abort_flag = _Flag()
    ctrl.ui.train_cancel_btn.setEnabled(True)
    ctrl.ui.update_and_train_button.setEnabled(False)
    ctrl.cancel_training()
    assert ctrl.ui.train_cancel_btn.text() == "Cancelling..."
    assert not ctrl.ui.train_cancel_btn.isEnabled()
    assert not ctrl.ui.update_and_train_button.isEnabled()   # still locked
    ctrl._restore_train_button()
    assert _idle(ctrl) == (True, False, T.CANCEL_BTN_TEXT, "Train")


def test_hook_3_runs_right_after_begin_live_round(tmp_path):
    ctrl = _ctrl(tmp_path)
    ctrl._train_model()
    names = [e[0] for e in ctrl._events]
    i = names.index("begin_live_round")
    assert names[i + 1] == "on_train_started"
    assert ctrl._events[i][1] == "round_00"
    assert ctrl._events[i + 1][1] == "lateral"
    assert ctrl._events[i + 1][2].endswith("_lateral_round_00")


def test_the_early_return_leaves_the_idle_buttons_alone(tmp_path):
    ctrl = _ctrl(tmp_path)
    ctrl.ui.data_to_add_list = _ListWidget([])
    ctrl.sample_records = {}
    ctrl._train_model()
    assert ctrl._during == []                          # never started
    assert _idle(ctrl) == (True, False, T.CANCEL_BTN_TEXT, "Train")
