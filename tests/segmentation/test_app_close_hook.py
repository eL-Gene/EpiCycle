"""Closing napari asks Step 4 about unsaved tier work.

``on_app_close`` reuses the Save / Discard / Stay question of a load or a
workspace change, in its closing mode: napari may still keep the window
open after it, so Discard changes nothing (no unload, the flags stay).  A
question that raises asks "Close anyway?" so the window is never trapped.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from qtpy.QtWidgets import QMessageBox

from napari_epicycle.segmentation import controller as ctl
from napari_epicycle.segmentation.controller import SegmentStackController as C


def _fake(choice):
    fake = MagicMock()
    fake._current_stack_name = "stack_a"
    fake._has_unsaved_changes = True
    fake._ask_unsaved.return_value = choice
    fake._settle_unsaved = (
        lambda what, **kw: C._settle_unsaved(fake, what, **kw))
    return fake


@pytest.mark.parametrize("answer", [True, False])
def test_on_app_close_answers_what_settle_unsaved_answers(answer):
    fake = MagicMock()
    fake._settle_unsaved = MagicMock(return_value=answer)
    assert C.on_app_close(fake) is answer
    fake._settle_unsaved.assert_called_once_with("closing EpiCycle",
                                                 closing=True)


@pytest.mark.parametrize("reply, expected", [
    (QMessageBox.Yes, True),
    (QMessageBox.No, False),
])
def test_a_question_that_raises_asks_close_anyway(reply, expected,
                                                  monkeypatch, capsys):
    asked = []

    def _question(*args, **kwargs):
        asked.append(args)
        return reply

    monkeypatch.setattr(ctl.QMessageBox, "question", _question)
    fake = MagicMock()
    fake._settle_unsaved = MagicMock(side_effect=OSError("locked"))
    assert C.on_app_close(fake) is expected
    assert len(asked) == 1
    assert asked[0][1] == "Closing EpiCycle"
    assert "Close anyway?" in asked[0][2]
    assert "[APP-CLOSE]" in capsys.readouterr().out


def test_closing_discard_goes_on_without_unloading():
    fake = _fake("discard")
    assert C.on_app_close(fake) is True
    fake._unload_stack.assert_not_called()
    assert fake._has_unsaved_changes is True


def test_closing_save_saves_as_before():
    fake = _fake("save")
    fake._save_tier_assignments.return_value = True
    assert C.on_app_close(fake) is True
    fake._save_tier_assignments.assert_called_once_with(confirm=False)


def test_closing_stay_keeps_the_window():
    fake = _fake("stay")
    assert C.on_app_close(fake) is False
    fake._unload_stack.assert_not_called()


def test_the_default_discard_still_unloads():
    fake = _fake("discard")
    assert C._settle_unsaved(fake, "a load") is True
    fake._unload_stack.assert_called_once_with("discarded before a load")
