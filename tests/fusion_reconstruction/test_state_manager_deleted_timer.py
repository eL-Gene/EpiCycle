"""`ReconstructionStateManager.flush()` survives a deleted debounce QTimer
(Step 3 R2a fold, audit B LOW-4; Step 2's twin fixed in ``779c6eec``).

The debounce timer is a parentless ``QTimer``.  At napari close Qt can
delete its C++ side first, and ``isActive()`` on such a wrapper raises
``RuntimeError: wrapped C/C++ object of type QTimer has been deleted`` --
which would skip the save a caller flushes for (Reset to Predicted, the
run switch, workspace close).  ``flush()`` now saves anyway (the timer can
no longer fire, so a pending save would be lost) and drops the dead timer,
which ``schedule_save()`` then builds again.
"""
from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")

from napari_epicycle.fusion_reconstruction.state_manager import (  # noqa: E402
    ReconstructionStateManager,
)

_DELETED = "wrapped C/C++ object of type QTimer has been deleted"


class _DeletedTimer:
    """Every call raises, as a wrapper whose C++ side Qt deleted."""

    def __getattr__(self, name):
        def dead(*_a, **_k):
            raise RuntimeError(_DELETED)
        return dead


@pytest.fixture(scope="module")
def app():
    from qtpy.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _manager(ws):
    sm = ReconstructionStateManager()
    sm.load(str(ws))
    return sm


def test_flush_saves_through_a_deleted_timer(app, tmp_path):
    sm = _manager(tmp_path)
    sm.set_start_from("useg3d")          # a pending debounced save
    sm._save_timer = _DeletedTimer()     # Qt deleted it at close
    sm.flush()                           # must not raise
    data = json.loads((tmp_path / sm.STATE_FILENAME).read_text(
        encoding="utf-8"))
    assert data["runs"]["default"]["start_from"] == "useg3d"
    assert sm._save_timer is None


def test_schedule_save_after_a_deleted_timer_builds_a_new_one(app, tmp_path):
    sm = _manager(tmp_path)
    sm._save_timer = _DeletedTimer()
    sm.schedule_save()                   # must not raise
    assert sm._save_timer is not None
    assert sm._save_timer.isActive()
    sm._save_timer.stop()


def test_flush_with_a_live_timer_is_unchanged(app, tmp_path):
    sm = _manager(tmp_path)
    sm.set_start_from("epicycle")
    assert sm._save_timer.isActive()
    sm.flush()
    assert not sm._save_timer.isActive()
    data = json.loads((tmp_path / sm.STATE_FILENAME).read_text(
        encoding="utf-8"))
    assert data["runs"]["default"]["start_from"] == "epicycle"
