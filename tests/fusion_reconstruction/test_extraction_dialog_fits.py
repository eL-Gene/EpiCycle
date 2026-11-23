"""The Extract Features dialog fits the screen and keeps OK / Cancel in view."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")

from napari_epicycle.fusion_reconstruction import _step3_pr_text as T  # noqa: E402


@pytest.fixture(scope="module")
def app():
    from qtpy.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _dialog():
    from napari_epicycle.fusion_reconstruction.qc_extraction_dialog import (
        QCExtractionDialog,
    )
    manifest = {
        f"Group {i}": [f"feature_{i}_{j}" for j in range(8)] for i in range(5)
    }
    channels = [{"index": i, "prefix": f"ch{i}"} for i in range(3)]
    return QCExtractionDialog(
        None,
        feature_manifest=manifest,
        available_channels=channels,
        patch_spacing_lines=[(T.S3_EXTRACT_SPACING_NONE, True)],
    )


class _SmallScreen:
    """A 1280-wide available area, shorter than the dialog's natural height."""

    def __init__(self, height, width=1280):
        self._h = height
        self._w = width

    def availableGeometry(self):
        from qtpy.QtCore import QRect
        return QRect(0, 0, self._w, self._h)

    def geometry(self):
        return self.availableGeometry()


@pytest.mark.parametrize("screen_h", [600, 480])
def test_dialog_fits_the_screen_and_shows_ok(app, monkeypatch, screen_h):
    from qtpy.QtWidgets import QDialogButtonBox

    from napari_epicycle.fusion_reconstruction.qc_extraction_dialog import (
        QCExtractionDialog,
    )

    monkeypatch.setattr(QCExtractionDialog, "screen", lambda self: _SmallScreen(screen_h))
    d = _dialog()
    d.show()
    for _ in range(4):
        app.processEvents()
    assert d.height() <= screen_h
    ok = d.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok)
    bottom = ok.mapTo(d, ok.rect().bottomLeft()).y()
    assert 0 <= bottom <= d.height()
    d.close()


@pytest.mark.parametrize("size", [(1280, 600), (1200, 1600)])
def test_every_input_is_inside_the_visible_width(app, monkeypatch, size):
    from qtpy.QtWidgets import QAbstractSpinBox, QCheckBox, QComboBox

    from napari_epicycle.fusion_reconstruction.qc_extraction_dialog import (
        QCExtractionDialog,
    )

    monkeypatch.setattr(
        QCExtractionDialog, "screen", lambda self: _SmallScreen(size[1], size[0]))
    d = _dialog()
    d.show()
    for _ in range(4):
        app.processEvents()
    viewport_w = d._scroll.viewport().width()
    widgets = [
        w for w in d._body.findChildren((QAbstractSpinBox, QComboBox, QCheckBox))
        if w.isVisibleTo(d)
    ]
    assert widgets
    for w in widgets:
        right = w.mapTo(d._body, w.rect().topRight()).x()
        assert right <= viewport_w, (type(w).__name__, w.objectName(), right, viewport_w)
    d.close()


def test_a_narrow_screen_gets_a_horizontal_bar_and_a_wide_one_does_not(app, monkeypatch):
    from qtpy.QtCore import Qt

    from napari_epicycle.fusion_reconstruction.qc_extraction_dialog import (
        QCExtractionDialog,
    )

    from qtpy.QtCore import QRect

    # The body's width depends on the platform's fonts, so the narrow screen
    # is derived from it: its 90% cap lands just under the wanted width.
    probe = _dialog()
    wanted = probe._wanted_width(QRect(0, 0, 100000, 100000))
    probe.close()
    narrow = int(wanted / 0.9) - 20
    assert int(narrow * 0.9) < wanted

    monkeypatch.setattr(
        QCExtractionDialog, "screen", lambda self: _SmallScreen(1000, narrow))
    d = _dialog()
    d.show()
    for _ in range(4):
        app.processEvents()
    assert d.width() <= 0.9 * narrow
    assert d._scroll.horizontalScrollBarPolicy() == Qt.ScrollBarAsNeeded
    assert d._scroll.horizontalScrollBar().maximum() > 0
    d.close()

    monkeypatch.setattr(
        QCExtractionDialog, "screen", lambda self: _SmallScreen(1000, 1920))
    d = _dialog()
    d.show()
    for _ in range(4):
        app.processEvents()
    assert d._scroll.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
    assert d._scroll.horizontalScrollBar().maximum() == 0
    d.close()


@pytest.mark.parametrize("title_bar", [0, 31])
def test_whole_frame_stays_inside_the_available_area(app, monkeypatch, title_bar):
    """Offscreen windows have no title bar, so a 31 px one is faked on the
    frame; with 0 this is a plain guard."""
    from qtpy.QtCore import QRect

    from napari_epicycle.fusion_reconstruction.qc_extraction_dialog import (
        QCExtractionDialog,
    )

    monkeypatch.setattr(QCExtractionDialog, "screen", lambda self: _SmallScreen(600))
    monkeypatch.setattr(
        QCExtractionDialog, "frameGeometry",
        lambda self: self.geometry().adjusted(0, -title_bar, 0, 0),
    )
    avail = QRect(0, 0, 1280, 600)
    d = _dialog()
    d.move(100, -5)
    d.show()
    for _ in range(4):
        app.processEvents()
    frame = d.frameGeometry()
    assert frame.top() >= avail.top()
    assert frame.bottom() <= avail.bottom()
    d.close()


def test_wheel_over_a_spin_box_does_not_change_it(app):
    from qtpy.QtCore import QPoint, QPointF, Qt
    from qtpy.QtGui import QWheelEvent
    from qtpy.QtWidgets import QApplication, QAbstractSpinBox, QComboBox

    d = _dialog()
    d.show()
    app.processEvents()
    sbs = d.findChildren(QAbstractSpinBox) + d.findChildren(QComboBox)
    assert sbs
    assert all(w.focusPolicy() == Qt.StrongFocus for w in sbs)
    d.erosion_iters_sb.clearFocus()
    before = d.erosion_iters_sb.value()
    ev = QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(0, 0), QPoint(0, 120),
                     Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
    QApplication.sendEvent(d.erosion_iters_sb, ev)
    assert d.erosion_iters_sb.value() == before
    d.close()
