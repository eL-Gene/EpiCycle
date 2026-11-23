"""Closing napari asks each step about its unsaved work first.

The shell installs one event filter on napari's main window.  On a Close
event it asks every step controller that has an ``on_app_close`` hook; a
step that answers False keeps the window open; a hook that raises asks
"Close anyway?", so no step can trap the window.  A Close that arrives while
the question is being asked is refused without asking again.  A re-created
shell replaces the old filter on the same window.

Qt-backed: a plain ``QWidget`` stands in for napari's main window and a
small fake stands in for the shell.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

qtpy = pytest.importorskip("qtpy")
from qtpy.QtGui import QCloseEvent
from qtpy.QtWidgets import QApplication, QMessageBox, QWidget

from napari_epicycle.ui_shell import workflow_shell as ws
from napari_epicycle.ui_shell.workflow_shell import EpiCycleWorkflowShell


@pytest.fixture(scope="module")
def app():
    """Ensure a QApplication exists for QWidget instantiation."""
    existing = QApplication.instance()
    if existing is not None:
        yield existing
        return
    new_app = QApplication([])
    yield new_app


class _Ctrl:
    """A step controller with an ``on_app_close`` hook."""

    def __init__(self, answer=True, raises=None, on_call=None):
        self.answer = answer
        self.raises = raises
        self.on_call = on_call
        self.calls = 0

    def on_app_close(self):
        self.calls += 1
        if self.on_call is not None:
            self.on_call()
        if self.raises is not None:
            raise self.raises
        return self.answer


class _FakeShell:
    """Just the parts of the shell the guard reads."""

    def __init__(self, window, controllers):
        self.viewer = SimpleNamespace(
            window=SimpleNamespace(_qt_window=window))
        self._controllers = dict(controllers)
        self._step_specs = [SimpleNamespace(step_id=sid)
                            for sid in self._controllers]

    def _get_step_controller(self, step_id):
        return self._controllers.get(step_id)


class _Window(QWidget):
    """A stand-in main window that records removed event filters."""

    def __init__(self):
        super().__init__()
        self.removed = []

    def removeEventFilter(self, obj):  # noqa: N802 (Qt name)
        self.removed.append(obj)
        super().removeEventFilter(obj)


@pytest.fixture
def window(app):
    w = _Window()
    yield w
    w.deleteLater()


@pytest.fixture
def close_anyway(monkeypatch):
    """The shell's "Close anyway?" box, answered by ``answer``."""
    rec = {"asked": [], "answer": QMessageBox.No}

    def _question(*args, **kwargs):
        rec["asked"].append(args)
        return rec["answer"]

    monkeypatch.setattr(ws.QMessageBox, "question", _question)
    return rec


def _install(shell):
    EpiCycleWorkflowShell._install_app_close_guard(shell)


def _close(window):
    event = QCloseEvent()
    event.accept()
    QApplication.sendEvent(window, event)
    return event


def test_no_hook_lets_the_close_through(window):
    shell = _FakeShell(window, {
        "a": object(),                 # no on_app_close
        "b": None,                     # step widget never built
    })
    _install(shell)
    event = _close(window)
    assert event.isAccepted()


def test_a_step_that_refuses_keeps_the_window_and_stops_asking(window):
    first = _Ctrl(answer=True)
    refusing = _Ctrl(answer=False)
    later = _Ctrl(answer=True)
    shell = _FakeShell(window, {"a": first, "b": refusing, "c": later})
    _install(shell)
    event = _close(window)
    assert not event.isAccepted()
    assert first.calls == 1
    assert refusing.calls == 1
    assert later.calls == 0


def test_every_step_agrees_so_the_close_goes_on(window):
    ctrls = {"a": _Ctrl(), "b": object(), "c": _Ctrl()}
    shell = _FakeShell(window, ctrls)
    _install(shell)
    event = _close(window)
    assert event.isAccepted()
    assert ctrls["a"].calls == 1
    assert ctrls["c"].calls == 1


def test_a_hook_that_raises_asks_and_no_keeps_the_window(
        window, close_anyway, capsys):
    close_anyway["answer"] = QMessageBox.No
    bad = _Ctrl(raises=RuntimeError("boom"))
    later = _Ctrl()
    shell = _FakeShell(window, {"a": bad, "b": later})
    _install(shell)
    event = _close(window)
    assert not event.isAccepted()
    assert len(close_anyway["asked"]) == 1
    assert close_anyway["asked"][0][0] is shell
    assert "Close anyway?" in close_anyway["asked"][0][2]
    assert later.calls == 0
    assert "[APP-CLOSE]" in capsys.readouterr().out


def test_a_hook_that_raises_asks_and_yes_goes_on(window, close_anyway):
    close_anyway["answer"] = QMessageBox.Yes
    bad = _Ctrl(raises=RuntimeError("boom"))
    later = _Ctrl()
    shell = _FakeShell(window, {"a": bad, "b": later})
    _install(shell)
    event = _close(window)
    assert event.isAccepted()
    assert len(close_anyway["asked"]) == 1
    assert later.calls == 1


def test_a_hook_that_returns_false_is_not_asked_close_anyway(
        window, close_anyway):
    shell = _FakeShell(window, {"a": _Ctrl(answer=False)})
    _install(shell)
    event = _close(window)
    assert not event.isAccepted()
    assert close_anyway["asked"] == []


def test_a_close_while_asking_is_refused_without_asking_again(window):
    inner = {}

    def _second_close():
        inner["event"] = _close(window)

    asking = _Ctrl(answer=True, on_call=_second_close)
    other = _Ctrl(answer=True)
    shell = _FakeShell(window, {"a": asking, "b": other})
    _install(shell)
    outer = _close(window)
    assert "event" in inner
    assert not inner["event"].isAccepted()
    assert asking.calls == 1
    assert other.calls == 1
    assert outer.isAccepted()


def test_installing_twice_installs_one_filter(window):
    ctrl = _Ctrl()
    shell = _FakeShell(window, {"a": ctrl})
    _install(shell)
    first = shell._app_close_guard
    _install(shell)
    assert shell._app_close_guard is first
    _close(window)
    assert ctrl.calls == 1


def test_a_new_shell_replaces_the_old_guard_on_the_same_window(window):
    old_ctrl = _Ctrl(answer=False)
    old_shell = _FakeShell(window, {"a": old_ctrl})
    _install(old_shell)
    old_guard = old_shell._app_close_guard
    new_ctrl = _Ctrl(answer=True)
    new_shell = _FakeShell(window, {"a": new_ctrl})
    _install(new_shell)
    assert window.removed == [old_guard]
    assert window._epicycle_app_close_guard is new_shell._app_close_guard
    event = _close(window)
    assert event.isAccepted()
    assert old_ctrl.calls == 0
    assert new_ctrl.calls == 1


def test_the_guard_holds_the_shell_weakly(window):
    ctrl = _Ctrl(answer=False)
    shell = _FakeShell(window, {"a": ctrl})
    _install(shell)
    guard = shell._app_close_guard
    del shell
    import gc
    gc.collect()
    event = _close(window)
    # The shell is gone: nothing is asked and the close goes on.
    assert event.isAccepted()
    assert ctrl.calls == 0
    window.removeEventFilter(guard)


def test_a_viewer_without_a_qt_window_is_a_no_op():
    shell = SimpleNamespace(viewer=SimpleNamespace(window=SimpleNamespace()),
                            _step_specs=[],
                            _get_step_controller=lambda sid: None)
    EpiCycleWorkflowShell._install_app_close_guard(shell)
    assert getattr(shell, "_app_close_guard", None) is None

    shell2 = SimpleNamespace(viewer=None, _step_specs=[],
                             _get_step_controller=lambda sid: None)
    EpiCycleWorkflowShell._install_app_close_guard(shell2)
    assert getattr(shell2, "_app_close_guard", None) is None
