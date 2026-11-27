"""The New Run dialog for a folder's first run: no "Create only".

"Create only" makes a run and stays on the current one; with no current run
there is nothing to stay on, so the dialog built for the first run hides it
(and the button cannot submit).  Qt-backed (offscreen); the dialog is built,
never shown or run.
"""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication, QWidget  # noqa: E402

from napari_epicycle.hitl_training_annotation._new_run_dialog import (  # noqa: E402
    NewRunDialog,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def parent(qapp):
    w = QWidget()
    yield w
    w.deleteLater()


def test_create_only_hidden_on_first_run(parent):
    dlg = NewRunDialog(parent=parent, allow_create_only=False)
    assert dlg._create_only_btn.isHidden() is True
    assert dlg._create_activate_btn.isHidden() is False
    dlg._slug_edit.setText("pilot")
    dlg._on_create_only()                      # a stray press submits nothing
    assert dlg.get_result() is None


def test_create_only_offered_by_default(parent):
    dlg = NewRunDialog(parent=parent)
    assert dlg._create_only_btn.isHidden() is False
