"""The shell's step-nav lock has OWNERS: the nav is locked while any owner holds it, and a call adds or
removes only its own owner, so Step 4's job ending can never release Step
2's training lock (or the other way round).  Steps 2 and 3 call it without
``owner`` -- the ``"default"`` owner, behaviour unchanged.

Also the exit veto: ``_select_step`` stays when the
previous step's ``on_step_exit`` returns ``False`` -- and only then.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def app():
    from qtpy.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _shell(app):
    from qtpy.QtWidgets import QPushButton

    from napari_epicycle.ui_shell.workflow_shell import EpiCycleWorkflowShell

    class _S:
        pass

    s = _S()
    s._nav_items = {k: QPushButton(k) for k in ("a", "b", "c")}
    s.lock_step_nav = EpiCycleWorkflowShell.lock_step_nav.__get__(s)
    s._select_step = EpiCycleWorkflowShell._select_step.__get__(s)
    return s


def _locked(s):
    return (not any(b.isEnabled() for b in s._nav_items.values())
            and s._step_nav_locked is True)


def test_the_default_owner_behaves_as_before(app):
    s = _shell(app)
    s.lock_step_nav(True)
    assert _locked(s)
    s.lock_step_nav(True)                   # idempotent
    s.lock_step_nav(False)
    assert not _locked(s)
    assert all(b.isEnabled() for b in s._nav_items.values())


def test_step4_ending_never_releases_step2(app):
    s = _shell(app)
    s.lock_step_nav(True)                               # Step 2 training
    s.lock_step_nav(True, owner="step4")
    s.lock_step_nav(False, owner="step4")
    assert _locked(s)
    s.lock_step_nav(False)
    assert not _locked(s)


def test_step2_ending_never_releases_step4(app):
    s = _shell(app)
    s.lock_step_nav(True, owner="step4")
    s.lock_step_nav(True)
    s.lock_step_nav(False)
    assert _locked(s)
    s.lock_step_nav(False, owner="step4")
    assert not _locked(s)


def test_releasing_an_owner_that_holds_nothing_is_harmless(app):
    s = _shell(app)
    s.lock_step_nav(False, owner="step4")
    assert not _locked(s)
    s.lock_step_nav(True)
    s.lock_step_nav(False, owner="step4")
    assert _locked(s)


def test_a_locked_nav_refuses_select_step(app):
    s = _shell(app)
    s._step_index_by_id = {"a": 0, "b": 1}
    s._active_step_id = "a"
    s.lock_step_nav(True, owner="step4")
    s._select_step("b")
    assert s._active_step_id == "a"
