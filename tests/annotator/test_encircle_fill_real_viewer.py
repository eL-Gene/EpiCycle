"""Real-viewer coverage for the napari-0.8 encircle-fill containment.

The Qt-free fakes in ``test_encircle_fill_disabled.py`` cannot see the hazard
that made the first version of this fix a CRASH:
``VispyLabelsBrushStrokeOverlay.close()`` does an UNGUARDED
``layer.mouse_drag_callbacks.remove(self._on_mouse_press)``.  Removing that
callback ourselves therefore turned every subviewer Labels-layer removal into
``ValueError: list.remove(x): x not in list`` raised out of
``viewer.layers.remove(...)`` — inside ``_add_or_replace_labels``,
``remove_layer_from_subviewers`` and ``clear()``, the last of which would
abort mid-loop and leave orphan layers.  Only a REAL ViewerModel + QtViewer
(which actually builds and closes the vispy overlay) exercises that path.

This file builds QtViewers, so run it ALONE —
``pytest tests/annotator/test_encircle_fill_real_viewer.py`` — not as part of
the whole ``tests/annotator/`` directory.
"""

from __future__ import annotations

import numpy as np
import pytest

napari = pytest.importorskip("napari")


@pytest.fixture(scope="module")
def qt_app():
    """A QApplication, or a clean skip when no display/Qt is available."""
    try:
        from qtpy.QtWidgets import QApplication
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Qt is unavailable: {exc}")
    app = QApplication.instance()
    if app is None:
        try:
            app = QApplication([])
        except Exception as exc:  # pragma: no cover - environment dependent
            pytest.skip(f"cannot create a QApplication: {exc}")
    return app


@pytest.fixture
def subviewers(qt_app):
    """A real SubviewerManager over three ViewerModel + QtViewer pairs.

    Mirrors ``NativeMultiviewAnnotator.__init__`` (native_multiview_widget.py
    :178-183) — the construction path both annotators actually use.
    """
    try:
        from napari.components import ViewerModel
        from napari._qt.qt_viewer import QtViewer
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"napari Qt stack unavailable: {exc}")

    from packages.epicycle_annotator.subviewers import SubviewerManager

    vms, qts = [], []
    try:
        for _ in range(3):
            vm = ViewerModel()
            vms.append(vm)
            qts.append(QtViewer(vm))
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"cannot construct a QtViewer (no GL/display?): {exc}")

    mgr = SubviewerManager(*vms, *qts)
    yield mgr, vms, qts

    for qt in qts:
        try:
            qt.close()
        except Exception:
            pass


class _Sync:
    def __init__(self):
        self.shared_labels = {}

    def register_shared_data(self, name, data):
        self.shared_labels[name] = data

    def connect_label_sync(self, layer):
        pass


class _Drawing:
    def create_mouse_drag_callback(self, orientation, viewer_model):
        def _cb(layer, event):  # pragma: no cover - never fired here
            pass

        return _cb


def _add_labels(mgr):
    from napari.layers import Labels

    src = Labels(np.zeros((4, 8, 8), np.int32), name="lbl")
    mgr.add_labels_to_subviewers(src, _Sync(), _Drawing())
    return src


def _sub_labels(vms):
    from napari.layers import Labels

    out = []
    for vm in vms:
        out.extend([l for l in vm.layers if isinstance(l, Labels)])
    return out


def _napari_press_callbacks(layer):
    """napari's brush-stroke press handlers currently on the layer."""
    found = []
    for cb in layer.mouse_drag_callbacks:
        func = getattr(cb, "__func__", cb)
        owner = getattr(cb, "__self__", None)
        module = str(getattr(func, "__module__", "") or "")
        if "labels_brush_stroke" in module or (
            owner is not None and "LabelsBrushStroke" in type(owner).__name__
        ):
            found.append(cb)
    return found


def test_qt_viewer_really_constructed(subviewers):
    """Guards against this whole file silently degrading into a no-op."""
    mgr, vms, qts = subviewers
    assert len(qts) == 3
    assert all(qt is not None for qt in qts)
    assert type(qts[0]).__name__ == "QtViewer"


def test_overlay_disabled_but_napari_callback_left_in_place(subviewers):
    mgr, vms, _ = subviewers
    _add_labels(mgr)

    subs = _sub_labels(vms)
    assert len(subs) == 3
    for lbl in subs:
        assert lbl._overlays["brush_stroke"].enabled is False, (
            "the encircle-fill gesture must be disabled through the overlay model"
        )
        assert _napari_press_callbacks(lbl), (
            "napari's press callback must STAY in mouse_drag_callbacks — its "
            "overlay's close() removes it unguarded, so stripping it makes "
            "every later layer removal raise ValueError"
        )


def test_disabled_survives_paint_pan_zoom_paint(subviewers):
    mgr, vms, _ = subviewers
    _add_labels(mgr)

    for lbl in _sub_labels(vms):
        lbl.mode = "pan_zoom"
        lbl.mode = "paint"
        assert lbl._overlays["brush_stroke"].enabled is False, (
            "napari re-enables the overlay on every switch into PAINT; the "
            "events.mode hook must re-assert"
        )


def test_layer_removal_does_not_raise(subviewers):
    """`remove_layer_from_subviewers` closes the vispy overlay -> close()."""
    mgr, vms, _ = subviewers
    _add_labels(mgr)

    mgr.remove_layer_from_subviewers("lbl")

    assert _sub_labels(vms) == []


def test_replace_and_clear_do_not_raise(subviewers):
    """`_add_or_replace_labels` (shape change) then `clear()`."""
    from napari.layers import Labels

    mgr, vms, _ = subviewers
    _add_labels(mgr)

    # Different shape -> _add_or_replace_labels removes + re-adds.
    src2 = Labels(np.zeros((6, 8, 8), np.int32), name="lbl")
    mgr.add_labels_to_subviewers(src2, _Sync(), _Drawing())
    assert len(_sub_labels(vms)) == 3

    mgr.clear()

    for vm in vms:
        assert len(vm.layers) == 0, "clear() must not abort mid-loop"
