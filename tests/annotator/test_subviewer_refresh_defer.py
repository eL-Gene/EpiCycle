"""Subviewer refresh defer in 3D mode — Phase A #2 (perf push 2026-05-09).

`_refresh_label_data_in_subviewers` is called after segmentation, refinement
op accept, and similar mask-changing operations.  Each call:

  - Sets ``sub_layer.data = transpose_data(main_data, orient)`` for XY/XZ/YZ
  - Calls ``sub_layer.refresh()``

When a subviewer is in **3D mode**, ``refresh()`` triggers a synchronous
``iso_categorical`` GPU recompute against the full-volume label texture.
At 1024³ this can take 1–2 seconds per subviewer.  With 3 subviewers all
in 3D, every refinement-op accept stalls 3–6 seconds.

Three changes:

  1. Identity check: skip the data setter when the transposed view is
     the same object the subviewer already holds (XY case where main
     and subviewer share the data array).
  2. ``QTimer.singleShot(50, _safe_deferred_refresh(sub_layer))`` instead
     of synchronous refresh() when ``vm.dims.ndisplay == 3``.
  3. The deferred refresh wrapper is sip-deletion-safe.  If sub_layer
     is destroyed (annotator close, patch switch, layer removal) within
     the 50 ms deferral window, the deferred .refresh() call would
     raise ``RuntimeError: wrapped C++ object has been deleted``.  The
     wrapper catches this and no-ops.

Subviewers in 2D mode keep synchronous refresh — 2D refresh is cheap.

These tests pin all three halves of the contract.  Lightweight fakes
only — no Qt event loop, no real ViewerModel.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class _FakeLabelsLayer:
    """Minimal Labels-layer stand-in.  ``isinstance(_, Labels)`` calls
    in the production code are patched at test time so this fake
    satisfies the type check without subclassing napari's Labels (which
    would require a Qt event loop).

    Tracks setter invocations via ``data_setter_calls`` so tests can
    verify the identity-check skip — Python rebinding to the same
    object is observationally a no-op, so a counter is the only way
    to detect whether the setter was invoked at all."""

    def __init__(self, name, data):
        self.name = name
        self._data = data
        self.refresh = MagicMock()
        self.data_setter_calls = 0

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self.data_setter_calls += 1
        self._data = value


class _FakeLayerList(list):
    """List-like collection of layers, supporting attribute-name lookup."""


class _FakeDims:
    def __init__(self, ndisplay):
        self.ndisplay = ndisplay


class _FakeViewerModel:
    def __init__(self, layers, ndisplay=2):
        self.layers = layers
        self.dims = _FakeDims(ndisplay)


class _AnnotatorFake:
    """Plain Python fake of NativeMultiviewAnnotator — enough state for
    ``_refresh_label_data_in_subviewers`` without instantiating a Qt
    widget (Qt blocks ``object.__new__`` on QWidget subclasses).

    The unbound method is invoked via
    ``NativeMultiviewAnnotator._refresh_label_data_in_subviewers(fake)``
    so the production code runs against this fake."""


def _make_annotator(main_data, *, sub_ndisplay=(2, 2, 2),
                    sub_data_for_orient=None):
    """Build an annotator fake with the minimum state needed.

    Args:
        main_data: ndarray to put on the main labels layer.
        sub_ndisplay: tuple (xy, xz, yz) of ``ndisplay`` values for the
            three subviewer ViewerModels.
        sub_data_for_orient: optional dict {orient: ndarray} to
            pre-seed subviewer layer data.
    """
    ann = _AnnotatorFake()
    ann._active_labels_name = "labels"
    ann._active_labels_names = ["labels"]

    main_layer = _FakeLabelsLayer("labels", main_data)
    main_list = _FakeLayerList([main_layer])
    ann.viewer = SimpleNamespace(layers=main_list)

    from packages.epicycle_annotator.orientation import transpose_data
    sub_data_for_orient = sub_data_for_orient or {}

    xy_layer = _FakeLabelsLayer("labels_XY",
        sub_data_for_orient.get("XY", transpose_data(main_data, "XY")))
    xz_layer = _FakeLabelsLayer("labels_XZ",
        sub_data_for_orient.get("XZ", transpose_data(main_data, "XZ")))
    yz_layer = _FakeLabelsLayer("labels_YZ",
        sub_data_for_orient.get("YZ", transpose_data(main_data, "YZ")))

    ann.viewer_model_xy = _FakeViewerModel(
        _FakeLayerList([xy_layer]), ndisplay=sub_ndisplay[0])
    ann.viewer_model_xz = _FakeViewerModel(
        _FakeLayerList([xz_layer]), ndisplay=sub_ndisplay[1])
    ann.viewer_model_yz = _FakeViewerModel(
        _FakeLayerList([yz_layer]), ndisplay=sub_ndisplay[2])

    return ann, main_layer, xy_layer, xz_layer, yz_layer


def _patch_isinstance_for_labels(monkeypatch):
    """Patch the Labels symbol in the production module so our fake
    satisfies the ``isinstance(_, Labels)`` check."""
    from napari_epicycle.annotator import native_multiview_widget as mod
    monkeypatch.setattr(mod, "Labels", _FakeLabelsLayer)


def _patch_layer_by_name(ann):
    """Replace _layer_by_name with a name-lookup."""
    def _lookup(layer_list, name):
        for layer in layer_list:
            if getattr(layer, "name", "") == name:
                return layer
        return None
    ann._layer_by_name = _lookup


def _call_refresh(ann):
    """Invoke the real `_refresh_label_data_in_subviewers` on the fake."""
    from napari_epicycle.annotator.native_multiview_widget import (
        NativeMultiviewAnnotator,
    )
    return NativeMultiviewAnnotator._refresh_label_data_in_subviewers(ann)


# ---------------------------------------------------------------------------
# 2D mode — synchronous refresh
# ---------------------------------------------------------------------------


class TestSyncRefreshIn2DMode:
    """When all subviewers are in 2D, refresh fires synchronously."""

    def test_refresh_called_immediately_for_all_subviewers(self, monkeypatch):
        _patch_isinstance_for_labels(monkeypatch)
        main_data = np.zeros((10, 20, 30), dtype=np.int32)
        ann, main_layer, xy, xz, yz = _make_annotator(
            main_data, sub_ndisplay=(2, 2, 2),
        )
        _patch_layer_by_name(ann)

        # Mock QTimer to verify it is NOT used in 2D path.
        with patch(
            "qtpy.QtCore.QTimer.singleShot"
        ) as qtimer_singleshot:
            _call_refresh(ann)
            qtimer_singleshot.assert_not_called()

        xy.refresh.assert_called_once()
        xz.refresh.assert_called_once()
        yz.refresh.assert_called_once()


# ---------------------------------------------------------------------------
# 3D mode — deferred refresh via QTimer
# ---------------------------------------------------------------------------


class TestDeferredRefreshIn3DMode:
    """When a subviewer is in 3D mode, its refresh is deferred via
    ``QTimer.singleShot(50, ...)``.  ``refresh()`` itself does not
    fire synchronously."""

    def test_xy_in_3d_uses_qtimer(self, monkeypatch):
        _patch_isinstance_for_labels(monkeypatch)
        main_data = np.zeros((10, 20, 30), dtype=np.int32)
        ann, main_layer, xy, xz, yz = _make_annotator(
            main_data, sub_ndisplay=(3, 2, 2),
        )
        _patch_layer_by_name(ann)

        with patch(
            "qtpy.QtCore.QTimer.singleShot"
        ) as qtimer_singleshot:
            _call_refresh(ann)

        # XY (3D) → deferred via QTimer; XZ + YZ (2D) → sync refresh
        qtimer_singleshot.assert_called_once()
        args = qtimer_singleshot.call_args
        assert args[0][0] == 50, "QTimer delay should be 50 ms"
        # Second arg is the deferred callable.  It should be a
        # callable (the safe-refresh wrapper, not the bare method).
        assert callable(args[0][1])
        xy.refresh.assert_not_called()
        xz.refresh.assert_called_once()
        yz.refresh.assert_called_once()

    def test_all_three_in_3d_all_deferred(self, monkeypatch):
        _patch_isinstance_for_labels(monkeypatch)
        main_data = np.zeros((10, 20, 30), dtype=np.int32)
        ann, main_layer, xy, xz, yz = _make_annotator(
            main_data, sub_ndisplay=(3, 3, 3),
        )
        _patch_layer_by_name(ann)

        with patch("qtpy.QtCore.QTimer.singleShot") as qtimer_singleshot:
            _call_refresh(ann)

        assert qtimer_singleshot.call_count == 3
        xy.refresh.assert_not_called()
        xz.refresh.assert_not_called()
        yz.refresh.assert_not_called()


# ---------------------------------------------------------------------------
# Sip-deletion safety — deferred refresh must not crash on dead layer
# ---------------------------------------------------------------------------


class TestDeferredRefreshSipSafe:
    """If the subviewer layer is destroyed (Qt sip-deletion) within
    the 50 ms deferral window, the deferred refresh callback must
    NOT crash the process.  Calling .refresh() on a deleted Qt object
    raises ``RuntimeError: wrapped C++ object has been deleted``.

    Verifies the callable passed to QTimer.singleShot is a
    sip-deletion-safe wrapper, not the bare ``sub_layer.refresh``."""

    def test_deferred_callable_swallows_runtime_error(self, monkeypatch):
        """Simulate the deferred callback firing AFTER sub_layer was
        deleted — refresh raises RuntimeError; the wrapper must catch
        it and no-op."""
        _patch_isinstance_for_labels(monkeypatch)
        main_data = np.zeros((10, 20, 30), dtype=np.int32)
        ann, main_layer, xy, xz, yz = _make_annotator(
            main_data, sub_ndisplay=(3, 2, 2),
        )
        _patch_layer_by_name(ann)

        # Make XY's refresh raise the canonical sip-deletion error.
        xy.refresh.side_effect = RuntimeError(
            "wrapped C++ object of type Labels has been deleted"
        )

        # Capture the callable passed to QTimer.singleShot.
        captured = {}

        def fake_singleshot(delay, callable_):
            captured["delay"] = delay
            captured["callable"] = callable_

        with patch("qtpy.QtCore.QTimer.singleShot", side_effect=fake_singleshot):
            _call_refresh(ann)

        # Now invoke the captured callable as if QTimer had fired —
        # the wrapper MUST swallow the RuntimeError and not propagate.
        assert "callable" in captured, "QTimer.singleShot was not invoked"
        # If the wrapper is missing or wrong, this raises and the test
        # fails.
        captured["callable"]()  # MUST NOT raise

    def test_deferred_callable_swallows_attribute_error(self, monkeypatch):
        """A defensively-also-caught case: an attribute access on a
        partially-disposed layer might raise AttributeError before
        RuntimeError.  Wrapper must tolerate that too."""
        _patch_isinstance_for_labels(monkeypatch)
        main_data = np.zeros((10, 20, 30), dtype=np.int32)
        ann, main_layer, xy, xz, yz = _make_annotator(
            main_data, sub_ndisplay=(3, 2, 2),
        )
        _patch_layer_by_name(ann)

        xy.refresh.side_effect = AttributeError(
            "'NoneType' object has no attribute 'refresh'"
        )

        captured = {}

        def fake_singleshot(delay, callable_):
            captured["callable"] = callable_

        with patch("qtpy.QtCore.QTimer.singleShot", side_effect=fake_singleshot):
            _call_refresh(ann)

        captured["callable"]()  # MUST NOT raise


# ---------------------------------------------------------------------------
# Identity check — skip the data setter when transposed view unchanged
# ---------------------------------------------------------------------------


class TestIdentityCheckSkipsRedundantSetter:
    """When ``transpose_data(main_data, orient)`` returns an object
    that is the same identity as ``sub_layer.data``, skip the
    assignment to avoid napari's setter cascade.  Refresh still fires."""

    def test_xy_setter_skipped_when_identity_same(self, monkeypatch):
        """For XY, ``transpose_data`` returns ``data`` itself.  If the
        subviewer already holds that exact object, the data setter
        must not be invoked."""
        _patch_isinstance_for_labels(monkeypatch)
        main_data = np.zeros((10, 20, 30), dtype=np.int32)

        ann, main_layer, xy, xz, yz = _make_annotator(
            main_data, sub_data_for_orient={"XY": main_data},
        )
        _patch_layer_by_name(ann)

        _call_refresh(ann)

        assert xy.data_setter_calls == 0, (
            f"XY setter fired {xy.data_setter_calls} time(s) even though "
            f"transposed view shares object identity with sub_layer.data"
        )
        xy.refresh.assert_called_once()

    def test_xz_setter_fires_for_new_view_object(self, monkeypatch):
        """For XZ, ``transpose_data`` returns a NEW view object each
        call (data.transpose(...)).  The identity check fails so the
        setter fires normally — once."""
        _patch_isinstance_for_labels(monkeypatch)
        main_data = np.zeros((10, 20, 30), dtype=np.int32)
        ann, main_layer, xy, xz, yz = _make_annotator(main_data)
        _patch_layer_by_name(ann)

        _call_refresh(ann)

        assert xz.data_setter_calls == 1
        xz.refresh.assert_called_once()


# ---------------------------------------------------------------------------
# Robustness — early-return on missing layers, exception swallowing
# ---------------------------------------------------------------------------


class TestRobustness:
    """The function must keep its existing tolerance for missing or
    broken sublayer state."""

    def test_skip_when_main_layer_absent(self, monkeypatch):
        _patch_isinstance_for_labels(monkeypatch)
        ann = _AnnotatorFake()
        ann._active_labels_name = "labels"
        ann._active_labels_names = ["labels"]
        ann.viewer = SimpleNamespace(layers=_FakeLayerList([]))
        ann.viewer_model_xy = _FakeViewerModel(_FakeLayerList([]))
        ann.viewer_model_xz = _FakeViewerModel(_FakeLayerList([]))
        ann.viewer_model_yz = _FakeViewerModel(_FakeLayerList([]))
        _patch_layer_by_name(ann)

        # Should not raise.
        _call_refresh(ann)

    def test_skip_when_subviewer_layer_absent(self, monkeypatch):
        _patch_isinstance_for_labels(monkeypatch)
        main_data = np.zeros((10, 20, 30), dtype=np.int32)
        ann, _main, _xy, _xz, _yz = _make_annotator(main_data)
        _patch_layer_by_name(ann)

        # Strip the XY subviewer layer.
        ann.viewer_model_xy.layers.clear()

        _call_refresh(ann)
        _xz.refresh.assert_called_once()
        _yz.refresh.assert_called_once()
