"""Brick T: hooks 1, 3 and 6 and seam (a):
each ``ui.on_*`` is called with the right arguments at its seam, a UI WITHOUT
the hooks still runs, a failing hook never breaks the controller, Cancel is
its own button and Train is never renamed.  Qt-free fakes."""
import inspect

from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
    TrainingAnnotationController as C,
)
from tests.hitl_training_annotation.test_step2_hooks import _Sig, _ctrl


class _Btn:
    def __init__(self, text=""):
        self._t = text
        self.enabled = True
        self.clicked = _Sig()

    def text(self):
        return self._t

    def setText(self, t):
        self._t = t

    def setEnabled(self, on):
        self.enabled = bool(on)


class _TrainUI:
    """Just what ``_set_training_ui_locked`` and the hooks touch."""

    def __init__(self, hooks=True):
        self.calls = []
        self.update_and_train_button = _Btn("Train")
        self.train_cancel_btn = _Btn("Cancel")
        if hooks:
            self.on_training_locked = lambda *a: self.calls.append(("locked",) + a)
            self.on_train_started = lambda *a: self.calls.append(("started",) + a)
            self.on_train_finished = lambda *a: self.calls.append(("finished",))
            self.on_training_epoch = lambda *a: self.calls.append(("epoch",) + a)


def _lock_ctrl(ui):
    c = _ctrl(ui)
    c.view_target = "lateral"
    c._find_open_2d_annotator = lambda: None
    c._workflow_shell = None
    return c


class TestHook1TrainingLocked:
    def test_it_fires_with_the_writes_round_name_on_lock_and_unlock(self):
        ui = _TrainUI()
        c = _lock_ctrl(ui)
        c._set_training_ui_locked(True)
        c._set_training_ui_locked(False)
        assert ui.calls == [("locked", True, "Cell_lateral_round_02", "lateral"),
                            ("locked", False, "Cell_lateral_round_02", "lateral")]

    def test_it_is_the_last_thing_the_lock_does(self):
        src = inspect.getsource(C._set_training_ui_locked)
        assert src.index("set_training_lock") < src.index('"on_training_locked"')

    def test_a_ui_without_the_hook_still_locks(self):
        c = _lock_ctrl(_TrainUI(hooks=False))
        c._set_training_ui_locked(True)
        assert c._train_ui_locked is True

    def test_a_failing_hook_does_not_break_the_lock(self):
        ui = _TrainUI()

        def boom(*a):
            raise RuntimeError("boom")
        ui.on_training_locked = boom
        c = _lock_ctrl(ui)
        c._set_training_ui_locked(True)
        assert c._train_ui_locked is True

    def test_cancel_is_not_in_the_lock_list(self):
        assert "train_cancel_btn" not in C._TRAIN_LOCK_WIDGET_ATTRS
        assert "update_and_train_button" not in C._TRAIN_LOCK_WIDGET_ATTRS


class TestHook3TrainStarted:
    def test_it_sits_beside_begin_live_round(self):
        # behavioural now: tests/.../controller/test_train_two_button_seam.py
        # :test_hook_3_runs_right_after_begin_live_round drives _train_model
        # and checks the call order; here, only that the hook is wired by name
        assert '"on_train_started"' in inspect.getsource(C._train_model)

    def test_the_round_name_is_the_writes_name(self):
        c = _lock_ctrl(_TrainUI())
        assert c._training_round_name("axial", 3) == "Cell_axial_round_03"


class TestHook6TrainFinished:
    def test_finish_and_error_paths_notify_after_the_plot_call(self):
        done = inspect.getsource(C._on_training_finished)
        assert done.index("loss_plot.add_round_curve(label, losses)") < \
            done.index("self._notify_train_finished()")
        err = inspect.getsource(C._on_training_error)
        assert err.index("loss_plot.cancel_live_round()") < \
            err.index("self._notify_train_finished()")

    def test_it_calls_the_hook_and_survives_a_failing_one(self):
        ui = _TrainUI()
        c = _lock_ctrl(ui)
        c._notify_train_finished()
        assert ui.calls == [("finished",)]
        ui.on_train_finished = lambda: 1 / 0
        c._notify_train_finished()
        _lock_ctrl(_TrainUI(hooks=False))._notify_train_finished()


class TestEpochSeam:
    def test_progress_forwards_epoch_total_and_loss(self):
        ui = _TrainUI()
        ui.training_progress_bar = type(
            "P", (), {"setValue": lambda s, v: None})()
        pts = []
        ui.loss_plot = type("L", (), {
            "append_live_point": lambda s, e, v: pts.append((e, v))})()
        c = _lock_ctrl(ui)
        c._append_ui_log = lambda *a, **k: None
        c._on_training_progress(40, "Epoch 412/1000 | loss: 0.81")
        assert pts == [(412, 0.81)]
        assert ui.calls == [("epoch", 412, 1000, 0.81)]


class TestSeamACancel:
    def test_cancel_is_connected_once_beside_train(self):
        src = inspect.getsource(C._connect_signals)
        assert src.count('"train_cancel_btn", "clicked", self.cancel_training') == 1

    def test_train_is_never_renamed_or_rewired(self):
        for fn in (C._train_model, C._restore_train_button):
            src = inspect.getsource(fn)
            assert "Cancel Training" not in src
            assert "clicked.disconnect" not in src
            assert "clicked.connect" not in src
        assert "_train_button_original_text" not in inspect.getsource(
            C._restore_train_button)

    def test_cancel_training_words_the_cancel_button_not_train(self):
        ui = _TrainUI()
        c = _lock_ctrl(ui)
        c._sync_training_in_flight = True
        c._training_abort_flag = type("F", (), {"set": lambda s: None})()
        c._append_ui_log = lambda *a, **k: None
        c.cancel_training()
        assert ui.train_cancel_btn.text() == "Cancelling..."
        assert not ui.train_cancel_btn.enabled
        assert ui.update_and_train_button.text() == "Train"

    def test_restore_brings_back_cancel_and_train(self):
        ui = _TrainUI()
        ui.train_cancel_btn.setText("Cancelling...")
        ui.train_cancel_btn.setEnabled(False)
        ui.update_and_train_button.setEnabled(False)
        _lock_ctrl(ui)._restore_train_button()
        assert ui.train_cancel_btn.text() == "Cancel"
        assert not ui.train_cancel_btn.enabled
        assert ui.update_and_train_button.enabled
        assert ui.update_and_train_button.text() == "Train"
