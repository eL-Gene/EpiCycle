"""Releasing the Step 3 lock must not light Undo curation on an empty stack."""
from __future__ import annotations

from .test_step3_mr_lock import _ctl, ui  # noqa: F401  (the real panel)


class _Stack:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __bool__(self):
        return self.n > 0


def test_unlock_leaves_undo_curation_grey_when_nothing_can_be_undone(ui):
    c = _ctl(ui)
    c._curation_undo_stack = _Stack(0)
    c._prediction_running = True
    c._refresh_step3_lock()
    c._prediction_running = False
    c._refresh_step3_lock()
    assert ui.undo_curation_btn.isEnabled() is False


def test_unlock_keeps_undo_curation_live_with_something_to_undo(ui):
    c = _ctl(ui)
    c._curation_undo_stack = _Stack(2)
    c._prediction_running = True
    c._refresh_step3_lock()
    c._prediction_running = False
    c._refresh_step3_lock()
    assert ui.undo_curation_btn.isEnabled() is True
