"""A3 -- "Show candidates as outlines".

One tick on the Add result card, one property: ``contour`` on the
candidate layer ``_epicycle_add_candidate``.  ``sync_label_properties``
already carries ``contour`` to every layer with the same base name, i.e.
the candidate's three side-view copies -- and to nothing else: the main
``curation_3d_*`` layer has another base name.  napari draws no contours in
3D, so the one warning it raises there is filtered and nothing else is.
Mirrors ``test_refine_contour_toggle.py`` (Reshape's outline tick).
"""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pytest

from .add_widget_fixture import qapp, reset_add, widget  # noqa: F401


def _K():
    from napari_epicycle.annotator.curation_3d_widget import (
        Curation3DAnnotator,
    )

    return Curation3DAnnotator


# --------------------------------------------------------------------------- #
# the control
# --------------------------------------------------------------------------- #
class TestTheCheckbox:
    def test_it_exists_says_candidates_and_starts_off(self, widget):
        from qtpy.QtWidgets import QCheckBox

        cb = widget.add_contour_cb
        assert isinstance(cb, QCheckBox)
        assert cb.text() == "Show candidates as outlines"
        assert cb.isChecked() is False

    def test_it_lives_on_the_result_card_under_the_walls_tick(self, widget):
        card = widget._add_result_container
        assert widget.add_contour_cb.parentWidget() is card
        lay = card.layout()
        assert lay.indexOf(widget.add_contour_cb) == lay.indexOf(
            widget.add_show_walls_cb) + 1

    def test_its_help_line(self, widget):
        from napari_epicycle.annotator.advanced_features_widget import (
            ADD_HELP_LINES,
        )

        line = ADD_HELP_LINES["add_contour_cb"]
        assert "side views" in line and "3D" in line
        assert widget.add_contour_cb.toolTip() == line

    def test_it_is_wired_through_the_frozen_setter(self, widget):
        got = []
        widget.set_add_callbacks(on_contour_toggled=got.append)
        widget.add_contour_cb.setChecked(True)
        widget.add_contour_cb.setChecked(False)
        assert got == [True, False]

    def test_the_orchestrator_can_put_it_down_silently(self, widget):
        got = []
        widget.set_add_callbacks(on_contour_toggled=got.append)
        widget.add_contour_cb.setChecked(True)
        got.clear()
        widget.update_add_ui_state(contour_on=False)
        assert widget.add_contour_cb.isChecked() is False
        assert got == [], "a reset is not a user tick"

    def test_the_orchestrator_hands_it_over(self):
        import pathlib

        src = (pathlib.Path(__file__).resolve().parents[2] / "napari_epicycle"
               / "annotator" / "curation_3d_widget.py").read_text(
                   encoding="utf-8")
        assert "on_contour_toggled=self._on_add_contour_toggled" in src


# --------------------------------------------------------------------------- #
# the handler
# --------------------------------------------------------------------------- #
class _Layer:
    """A layer that RECORDS every attribute written to it."""

    def __init__(self, name):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "contour", 0)
        object.__setattr__(self, "writes", [])

    def __setattr__(self, key, value):
        self.writes.append((key, value))
        object.__setattr__(self, key, value)


class _Fake:
    def __init__(self, cand=True):
        self._add_candidate_layer = (_Layer("_epicycle_add_candidate")
                                     if cand else None)
        self._add_contour_on = False
        self.main = _Layer("curation_3d_p1")


def _f(cand=True):
    f = _Fake(cand)
    K = _K()
    for n in ("_on_add_contour_toggled", "_add_set_contour"):
        setattr(f, n, getattr(K, n).__get__(f, _Fake))
    return f


class TestTheHandler:
    def test_ticking_sets_contour_to_one_and_back(self):
        f = _f()
        f._on_add_contour_toggled(True)
        assert f._add_candidate_layer.contour == 1
        assert f._add_contour_on is True
        f._on_add_contour_toggled(False)
        assert f._add_candidate_layer.contour == 0

    def test_it_writes_nothing_else_and_never_the_main_layer(self):
        f = _f()
        f._on_add_contour_toggled(True)
        assert [k for k, _v in f._add_candidate_layer.writes] == ["contour"]
        assert f.main.writes == []

    def test_remembered_with_no_candidate_yet(self):
        f = _f(cand=False)
        f._on_add_contour_toggled(True)
        assert f._add_contour_on is True

    @pytest.mark.parametrize("forbidden", [
        "sync_label_properties", "_update_isolation_proxy",
        "_sync_all_views_after_operation", "_refresh_annotator_subviewers",
        "shared_labels", "labels_render", "refresh(",
    ])
    def test_it_calls_no_sync_isolation_or_render_path(self, forbidden):
        for name in ("_on_add_contour_toggled", "_add_set_contour"):
            fn = getattr(_K(), name)
            body = inspect.getsource(fn).replace(fn.__doc__ or "", "")
            assert forbidden not in body, (name, forbidden)

    def test_reshapes_helper_is_untouched(self):
        src = inspect.getsource(_K()._set_refine_contour)
        assert "self._refine_layer.contour = 1 if on else 0" in src


class TestTheThreeDWarning:
    @staticmethod
    def _warn_layer(message):
        class _WarnLayer:
            def __init__(self):
                object.__setattr__(self, "contour", 0)

            def __setattr__(self, key, value):
                if key == "contour" and value:
                    warnings.warn(message)
                object.__setattr__(self, key, value)

        return _WarnLayer()

    def test_the_3d_warning_does_not_get_out(self):
        f = _f()
        f._add_candidate_layer = self._warn_layer(
            "Contours are not displayed during 3D rendering")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            f._on_add_contour_toggled(True)
        assert f._add_candidate_layer.contour == 1

    def test_a_different_warning_still_does(self):
        f = _f()
        f._add_candidate_layer = self._warn_layer("something else")
        with pytest.warns(UserWarning, match="something else"):
            f._on_add_contour_toggled(True)


# --------------------------------------------------------------------------- #
# a rebuilt candidate gets it AFTER its side copies exist
# --------------------------------------------------------------------------- #
def test_render_applies_it_after_the_subviewer_copies(monkeypatch):
    order = []

    class _Cand:
        def __init__(self, data, name=None, opacity=None):
            object.__setattr__(self, "data", data)
            object.__setattr__(self, "name", name)
            object.__setattr__(self, "contour", 0)

        def __setattr__(self, k, v):
            if k == "contour":
                order.append("contour={}".format(v))
            object.__setattr__(self, k, v)

        def refresh(self):
            pass

    class _Layers(list):
        class selection:
            active = None

    class _Viewer:
        layers = _Layers()

    class _Sub:
        def add_labels_to_subviewers(self, layer, **kw):
            order.append("subviewers")

        def get_subviewer_layer(self, name, orient):
            return None

    from napari_epicycle.annotator import curation_3d_widget as CW
    from packages.epicycle_annotator import sync as sync_mod

    monkeypatch.setattr(CW, "Labels", _Cand)
    monkeypatch.setattr(sync_mod, "lock_brush_size_to_1", lambda *a: None)

    class _Main:
        data = np.zeros((4, 6, 6), np.int32)
        scale = (1.0, 1.0, 1.0)
        translate = (0.0, 0.0, 0.0)

    f = _Fake(cand=False)
    f._add_contour_on = True
    f.viewer = _Viewer()
    f.subviewer_manager = _Sub()
    f.sync_manager = None
    f.drawing_handler = None
    f._add_main_label_layer = lambda: _Main()
    f._layer_by_name = lambda layers, name: None
    f._restore_side_active_layers = lambda: None
    K = _K()
    for n in ("_render_add_candidate", "_add_set_contour",
              "_remove_add_candidate"):
        setattr(f, n, getattr(K, n).__get__(f, _Fake))
    roi = (slice(0, 2), slice(0, 3), slice(0, 3))
    f._render_add_candidate(np.ones((2, 3, 3), np.int32), roi)
    assert "subviewers" in order and "contour=1" in order, order
    assert order.index("subviewers") < order.index("contour=1"), order


# --------------------------------------------------------------------------- #
# the propagation, through the real sync code (not edited)
# --------------------------------------------------------------------------- #
def test_sync_label_properties_hands_it_to_the_three_copies_only():
    from napari.layers import Labels

    from packages.epicycle_annotator.sync import LabelSyncManager

    cand = Labels(np.zeros((4, 4, 4), np.uint8),
                  name="_epicycle_add_candidate")
    copies = [Labels(np.zeros((4, 4), np.uint8),
                     name="_epicycle_add_candidate_{}".format(o))
              for o in ("XY", "XZ", "YZ")]
    other = Labels(np.zeros((4, 4), np.uint8), name="curation_3d_p6")

    class _VM:
        def __init__(self, layers):
            self.layers = layers

    fake = _VM([cand])
    fake.main_viewer = _VM([cand, other])
    fake.viewer_model_xy = _VM([copies[0], other])
    fake.viewer_model_xz = _VM([copies[1]])
    fake.viewer_model_yz = _VM([copies[2]])
    fake._syncing_properties = False
    for name in ("sync_label_properties", "_base_label_name",
                 "_layer_by_name"):
        setattr(fake, name, getattr(LabelSyncManager, name).__get__(fake, _VM))
    cand.contour = 1
    fake.sync_label_properties(cand)
    assert [c.contour for c in copies] == [1, 1, 1]
    assert other.contour == 0
