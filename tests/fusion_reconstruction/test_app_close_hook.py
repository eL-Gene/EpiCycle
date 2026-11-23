"""Closing napari asks Step 3 about unsaved tiers and curation edits.

``on_app_close`` is the shell's question on a window close.  It reuses the
patch-switch guard (Save / Discard / Cancel) in its closing mode: napari may
still ask its own question afterwards and keep the window open, so a Discard
there changes nothing (the next close asks again; a real close ends the
process).  An open edit in the annotator is said in a box, since the user
may be in another step.  A guard that raises asks "Close anyway?" so the
window can never be trapped.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from napari_epicycle.fusion_reconstruction import _step3_pr_text as _pr_text
from napari_epicycle.fusion_reconstruction import controller as ctl
from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController as C,
)
from qtpy.QtWidgets import QMessageBox


@pytest.fixture
def boxes(monkeypatch):
    """Every QMessageBox call the controller makes, answered by ``answer``."""
    rec = {"question": [], "information": [], "critical": [],
           "answer": QMessageBox.Discard}

    def _question(*args, **kwargs):
        rec["question"].append(args)
        return rec["answer"]

    monkeypatch.setattr(ctl.QMessageBox, "question", _question)
    monkeypatch.setattr(ctl.QMessageBox, "information",
                        lambda *a, **k: rec["information"].append(a))
    monkeypatch.setattr(ctl.QMessageBox, "critical",
                        lambda *a, **k: rec["critical"].append(a))
    return rec


def _fake(guard=None):
    fake = MagicMock()
    fake._staged_annotator_work.return_value = False
    fake._save_in_flight = False
    fake._tiers_dirty = True
    fake._curation_dirty = True
    fake._qc_view_patch_name = "patch_a"
    fake._curation_undo_stack = ["entry"]
    fake._revert_labels_from_disk.return_value = True
    if guard is not None:
        fake._guard_unsaved_changes = guard
    return fake


def _real_guard(fake):
    """Run the real guard on ``fake`` (closing passed through)."""
    fake._guard_unsaved_changes = (
        lambda **kw: C._guard_unsaved_changes(fake, **kw))
    return fake


# --- on_app_close over a faked guard ------------------------------------

@pytest.mark.parametrize("answer", [True, False])
def test_on_app_close_answers_what_the_guard_answers(answer, boxes):
    fake = _fake(MagicMock(return_value=answer))
    assert C.on_app_close(fake) is answer
    fake._guard_unsaved_changes.assert_called_once_with(closing=True)


def test_on_app_close_prints_the_dirty_flags(capsys, boxes):
    fake = _fake(MagicMock(return_value=True))
    C.on_app_close(fake)
    out = capsys.readouterr().out
    assert "[APP-CLOSE] step 3" in out
    assert "tiers_dirty=True" in out
    assert "curation_dirty=True" in out


def test_an_open_edit_is_said_in_a_box_and_keeps_the_window(boxes):
    fake = _fake(MagicMock(return_value=True))
    fake._staged_annotator_work.return_value = True
    assert C.on_app_close(fake) is False
    assert len(boxes["information"]) == 1
    args = boxes["information"][0]
    assert args[1] == "Open edit"
    assert args[2] == _pr_text.S3_STAGED_WORK_OPEN
    fake._guard_unsaved_changes.assert_not_called()


@pytest.mark.parametrize("reply, expected", [
    (QMessageBox.Yes, True),
    (QMessageBox.No, False),
])
def test_a_guard_that_raises_asks_close_anyway(reply, expected, boxes,
                                               capsys):
    boxes["answer"] = reply
    fake = _fake(MagicMock(side_effect=OSError("locked")))
    assert C.on_app_close(fake) is expected
    assert len(boxes["question"]) == 1
    args = boxes["question"][0]
    assert args[1] == "Closing EpiCycle"
    assert "Close anyway?" in args[2]
    # Not a save failure: nothing says the save failed.
    logged = " ".join(str(c.args[0])
                      for c in fake._append_log.call_args_list)
    assert _pr_text.S3_SAVE_FAILED not in logged
    assert "[APP-CLOSE]" in capsys.readouterr().out
    # The edits stay marked unsaved.
    assert fake._tiers_dirty is True
    assert fake._curation_dirty is True


# --- the guard's closing mode --------------------------------------------

def test_closing_discard_changes_nothing_and_goes_on(boxes):
    boxes["answer"] = QMessageBox.Discard
    fake = _real_guard(_fake())
    assert C.on_app_close(fake) is True
    assert len(boxes["question"]) == 2          # tiers, then curation
    assert fake._tiers_dirty is True
    assert fake._curation_dirty is True
    fake._revert_labels_from_disk.assert_not_called()
    assert fake._curation_undo_stack == ["entry"]


def test_closing_save_saves_as_before(boxes):
    boxes["answer"] = QMessageBox.Save
    fake = _real_guard(_fake())

    def _save_tiers():
        fake._tiers_dirty = False

    def _save_curation(**kw):
        fake._curation_dirty = False

    fake._save_tiers_csv.side_effect = _save_tiers
    fake._save_curation.side_effect = _save_curation
    assert C.on_app_close(fake) is True
    fake._save_tiers_csv.assert_called_once_with()
    fake._save_curation.assert_called_once_with(refresh_rows_ui=False)
    assert fake._curation_undo_stack == ["entry"]


def test_closing_save_that_fails_keeps_the_window(boxes):
    boxes["answer"] = QMessageBox.Save
    fake = _real_guard(_fake())
    fake._tiers_dirty = False
    # _save_curation leaves the flag set: the save failed.
    assert C.on_app_close(fake) is False
    fake._save_curation.assert_called_once_with(refresh_rows_ui=False)
    assert fake._curation_dirty is True


def test_closing_cancel_keeps_the_window(boxes):
    boxes["answer"] = QMessageBox.Cancel
    fake = _real_guard(_fake())
    assert C.on_app_close(fake) is False
    assert fake._tiers_dirty is True


def test_the_default_guard_still_discards_as_before(boxes):
    boxes["answer"] = QMessageBox.Discard
    fake = _fake()
    assert C._guard_unsaved_changes(fake) is True
    assert fake._tiers_dirty is False
    assert fake._curation_dirty is False
    fake._revert_labels_from_disk.assert_called_once_with()
    assert fake._curation_undo_stack == []
