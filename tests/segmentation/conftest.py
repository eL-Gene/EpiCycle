"""Shared fixtures for tests/segmentation/."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session")
def qapp():
    """One offscreen QApplication for the whole session (no pytest-qt)."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("qtpy")
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app
