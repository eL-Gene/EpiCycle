"""A module-scoped ``AdvancedFeaturesWidget`` for the Add page tests."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("qtpy")


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture(scope="module")
def widget(qapp):
    from napari_epicycle.annotator.advanced_features_widget import (
        AdvancedFeaturesWidget,
    )

    w = AdvancedFeaturesWidget()
    w.refresh_add_image_combo(["nuc", "mem"])
    yield w
    w.close()
    w.deleteLater()


def reset_add(w):
    """Back to a fresh Nuclei page with nothing touched."""
    w.clear_add_touched()
    w.set_add_preset("nuclei", silent=True)
    w.clear_add_touched()
