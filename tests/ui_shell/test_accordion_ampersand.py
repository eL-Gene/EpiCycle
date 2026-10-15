"""B1a audit A1-2: an Accordion title shows its ampersand once, whether the
caller wrote "&" (Step 4's "Measure & tidy") or already escaped it as "&&"
(Step 1's "Load && Adjust")."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    from qtpy.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_a_lone_ampersand_is_doubled_and_a_doubled_one_is_kept(qapp):
    from qtpy.QtWidgets import QWidget

    from napari_epicycle.ui_shell.theme import Accordion
    acc = Accordion()
    for title in ("Load && Adjust", "Measure & tidy", "Plain"):
        acc.addItem(QWidget(), title)
    texts = [s._header.text() for s in acc._sections]
    assert texts[0].endswith("Load && Adjust")
    assert texts[1].endswith("Measure && tidy")
    assert texts[2].endswith("Plain")
    acc.deleteLater()
