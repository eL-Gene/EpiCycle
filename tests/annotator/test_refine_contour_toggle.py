"""Brick U2 -- "Show this object as an outline".

One checkbox, one property set, and a very specific promise: the
outline reaches the **three side views**, because that is where the
markers are painted, and it reaches **nothing else** -- in particular
not the main ``curation_3d_*`` labels layer, which carries every other
object in the patch.

The promise is kept by `sync.py::sync_label_properties`, which already
propagates ``contour`` to every layer whose ``_base_label_name``
matches the one that changed.  `Refine` matches `Refine_XY`,
`Refine_XZ` and `Refine_YZ`; `curation_3d_<id>` does not match any of
them.  **No sync, isolation, mirror, proxy or rendering code is edited
to make this work**, which is the thing these tests are really here to
pin.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

pytest.importorskip("qtpy")


def _K():
    from napari_epicycle.annotator.curation_3d_widget import (
        Curation3DAnnotator,
    )

    return Curation3DAnnotator


@pytest.fixture(scope="module")
def qapp():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        try:
            app = QApplication([])
        except Exception as exc:  # pragma: no cover - CI without Qt
            pytest.skip("cannot create a QApplication: {}".format(exc))
    return app


@pytest.fixture
def widget(qapp):
    from napari_epicycle.annotator.advanced_features_widget import (
        AdvancedFeaturesWidget,
    )

    w = AdvancedFeaturesWidget(mode="3d_curation")
    yield w
    try:
        w.setParent(None)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# the control
# --------------------------------------------------------------------------- #
class TestTheCheckbox:
    def test_it_exists_and_says_object(self, widget):
        from qtpy.QtWidgets import QCheckBox

        assert isinstance(widget.split_contour_cb, QCheckBox)
        assert widget.split_contour_cb.text() == (
            "Show this object as an outline"), widget.split_contour_cb.text()

    def test_it_starts_off(self, widget):
        assert widget.split_contour_cb.isChecked() is False

    def test_it_lives_in_the_TARGET_slot(self, widget):
        parent = widget.split_contour_cb.parentWidget()
        assert parent is widget._split_s2_container, (
            "the tick sits directly under the Start/Change button"
        )

    def test_it_is_parented(self, widget):
        assert widget.split_contour_cb.parentWidget() is not None

    def test_its_help_line_is_on_screen(self, widget):
        from qtpy.QtWidgets import QLabel

        from napari_epicycle.annotator import advanced_features_widget as M

        text = M.HELP_LINES["split_contour_cb"]
        assert "side views" in text
        shown = [lb for lb in widget.findChildren(QLabel) if lb.text() == text]
        assert shown, "the tick is unexplained"

    def test_the_callback_is_wired_through_the_frozen_setter(self, widget):
        seen = []
        widget.set_split_callbacks(on_contour_toggled=seen.append)
        widget.split_contour_cb.setChecked(True)
        assert seen == [True]
        widget.split_contour_cb.setChecked(False)
        assert seen == [True, False]

    def test_the_orchestrator_hands_it_over(self):
        src = inspect.getsource(_K()._open_advanced_features)
        assert '"on_contour_toggled"' in src


# --------------------------------------------------------------------------- #
# what the handler touches -- and what it must not
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
    def __init__(self, refine=True):
        self._refine_layer = _Layer("Refine") if refine else None
        self._refine_contour_on = False
        self.main = _Layer("curation_3d_p1")


def _bind(f, *names):
    K = _K()
    for n in names:
        setattr(f, n, getattr(K, n).__get__(f, _Fake))
    return f


class TestTheHandler:
    def _f(self, refine=True):
        return _bind(_Fake(refine), "_on_refine_contour_toggled",
                     "_set_refine_contour")

    def test_ticking_sets_contour_to_one(self):
        f = self._f()
        f._on_refine_contour_toggled(True)
        assert f._refine_layer.contour == 1

    def test_unticking_sets_it_back_to_zero(self):
        f = self._f()
        f._on_refine_contour_toggled(True)
        f._on_refine_contour_toggled(False)
        assert f._refine_layer.contour == 0

    def test_it_writes_NOTHING_else_to_the_layer(self):
        f = self._f()
        f._on_refine_contour_toggled(True)
        assert [k for k, _v in f._refine_layer.writes] == ["contour"]

    def test_the_MAIN_labels_layer_is_never_written(self):
        f = self._f()
        f._on_refine_contour_toggled(True)
        assert f.main.writes == [], (
            "every other object in the patch lives on that layer"
        )

    def test_the_preference_is_remembered_with_no_layer_yet(self):
        f = self._f(refine=False)
        f._on_refine_contour_toggled(True)
        assert f._refine_contour_on is True

    def test_it_does_not_raise_with_no_layer_yet(self):
        self._f(refine=False)._on_refine_contour_toggled(True)

    @pytest.mark.parametrize("forbidden", [
        "sync_label_properties", "sync_selected_label",
        "_update_isolation_proxy", "_build_single_label_subproxy",
        "_sync_all_views_after_operation", "_refresh_annotator_subviewers",
        "_get_toggle_targets_for_name", "shared_labels",
        "labels_render", "refresh(",
    ])
    def test_it_calls_no_sync_isolation_or_render_path(self, forbidden):
        """The propagation is Qt's, through an event `sync.py` has been
        connected to since long before this round.  Calling any of these
        by hand would be a second, parallel path."""
        fn = _K()._on_refine_contour_toggled
        src = inspect.getsource(fn)
        # the docstring NAMES the propagation it relies on; what matters
        # is that the body does not go and call it
        body = src.replace(fn.__doc__ or "", "")
        assert forbidden not in body, forbidden


# --------------------------------------------------------------------------- #
# it survives the next object
# --------------------------------------------------------------------------- #
class TestItSurvivesTheNextObject:
    def test_the_layer_is_born_with_the_remembered_value(self):
        src = inspect.getsource(_K()._create_refine_layer)
        assert "_refine_contour_on" in src

    def test_it_is_applied_AFTER_the_subviewer_copies_are_made(self):
        """...and this is the order, which is the opposite of the one
        that looks right.

        MEASURED on the live panel (driver check 11i, 2026-09-20): with
        the write placed BEFORE `add_labels_to_subviewers` the next
        object came up outlined in the 3D view and SOLID in all three
        side views -- `mirrors=[0, 0, 0]`.  The reason is that nothing
        copies `contour` at creation; the mirrors get it from the
        property-change propagation `sync_label_properties` runs, and a
        change that happened before those layers existed has no
        listener.  Writing it after they exist fires the event with the
        listeners attached, and all three follow.

        The side views are the whole point of the feature (spec
        section 3), so "the tick persists across objects" is only true
        if it persists THERE.
        """
        src = inspect.getsource(_K()._create_refine_layer)
        assert src.index("_refine_contour_on") > src.index(
            "add_labels_to_subviewers"), (
            "the outline is written before the mirrors exist, so nothing "
            "propagates it to them"
        )

    def test_the_write_really_lands_after_the_subviewer_call(self,
                                                             monkeypatch):
        """The order above, proved by running it rather than reading it.

        A source-order assertion cannot tell which order PROPAGATES --
        that is exactly how the wrong one shipped.  This records the
        real sequence of (subviewer copies made, contour written)."""
        import numpy as np

        from packages.epicycle_annotator import sync as sync_mod

        order = []

        class _L:
            def __init__(self):
                self.name = "Refine"
                self._contour = 0

            def __setattr__(self, k, v):
                if k == "contour":
                    order.append("contour={}".format(v))
                object.__setattr__(self, k, v)

        class _Viewer:
            def __init__(self, layer):
                self._layer = layer

            def add_labels(self, data, name=None, opacity=None):
                return self._layer

        class _SubMgr:
            def add_labels_to_subviewers(self, layer, **kw):
                order.append("subviewers")

            def get_subviewer_layer(self, name, orient):
                return None

        layer = _L()
        f = _Fake()
        f._refine_contour_on = True
        f.viewer = _Viewer(layer)
        f.subviewer_manager = _SubMgr()
        f.sync_manager = None
        f.drawing_handler = None
        monkeypatch.setattr(sync_mod, "lock_brush_size_to_1",
                            lambda *a, **k: None)
        _bind(f, "_create_refine_layer", "_set_refine_contour")
        f._create_refine_layer(np.zeros((2, 2, 2), np.int32), 7)

        assert "subviewers" in order and "contour=1" in order, order
        assert order.index("subviewers") < order.index("contour=1"), (
            "the outline was written before the mirrors existed: "
            "{}".format(order)
        )

    def test_the_flag_lives_on_the_annotator_not_on_the_panel(self):
        """A panel rebuilt mid-session must not silently turn it off."""
        src = inspect.getsource(_K().__init__)
        assert "self._refine_contour_on = False" in src

    def test_no_session_reset_clears_it(self):
        for name in ("_reset_split_state", "_accept_split", "_reject_split",
                     "_teardown_reshape_for_switch"):
            src = inspect.getsource(getattr(_K(), name))
            assert "_refine_contour_on" not in src, (
                "{} forgets the outline preference".format(name)
            )


# --------------------------------------------------------------------------- #
# the propagation itself, read off the module we did NOT edit
# --------------------------------------------------------------------------- #
class TestTheSideViewsAreReachedByTheExistingSync:
    def test_sync_label_properties_really_hands_contour_over(self):
        """RUN it (audit M4).  `"contour" in <source>` passes on a
        function that merely mentions the word in a comment, and this is
        the load-bearing claim of the whole feature: the side views
        follow because THIS function carries the value."""
        from napari.layers import Labels
        from packages.epicycle_annotator.sync import LabelSyncManager

        refine = Labels(np.zeros((4, 4, 4), np.uint8), name="Refine")
        mirrors = [Labels(np.zeros((4, 4), np.uint8),
                          name="Refine_{}".format(o))
                   for o in ("XY", "XZ", "YZ")]
        # ...and the layer that carries every OTHER object in the patch.
        other = Labels(np.zeros((4, 4), np.uint8), name="curation_3d_p6")

        class _VM:
            def __init__(self, layers):
                self.layers = layers

        # The real manager wants seven viewers to build; the method
        # wants four attributes.  Bind it onto the four.
        fake = _VM([refine])
        fake.main_viewer = _VM([refine, other])
        fake.viewer_model_xy = _VM([mirrors[0], other])
        fake.viewer_model_xz = _VM([mirrors[1]])
        fake.viewer_model_yz = _VM([mirrors[2]])
        fake._syncing_properties = False
        for name in ("sync_label_properties", "_base_label_name",
                     "_layer_by_name"):
            setattr(fake, name,
                    getattr(LabelSyncManager, name).__get__(fake, _VM))

        refine.contour = 1
        fake.sync_label_properties(refine)

        assert [m.contour for m in mirrors] == [1, 1, 1], (
            "the side views did not receive the outline -- which is the "
            "one thing this feature is for"
        )
        assert other.contour == 0, (
            "the main labels layer was outlined too; every other object "
            "in the patch lives on it"
        )

    def test_a_refine_mirror_shares_the_base_name(self):
        from packages.epicycle_annotator import sync as sync_mod

        base = sync_mod.LabelSyncManager._base_label_name
        assert base(None, "Refine_XY") == "Refine"
        assert base(None, "Refine_XZ") == "Refine"
        assert base(None, "Refine_YZ") == "Refine"

    def test_the_main_curation_layer_does_NOT(self):
        from packages.epicycle_annotator import sync as sync_mod

        base = sync_mod.LabelSyncManager._base_label_name
        assert base(None, "curation_3d_patch_6") != "Refine"


# --------------------------------------------------------------------------- #
# H2 (audit, 2026-09-20) -- napari does not draw contours in 3D
# --------------------------------------------------------------------------- #
class TestTheThreeDViewerWarnsAndWeDoNotRelayIt:
    """`Labels._calculate_contour` returns None and warns "Contours are
    not displayed during 3D rendering" whenever the sliced labels have
    more than two dimensions.

    The main curation viewer IS 3D.  So the tick changes the three side
    views -- which is the point, they are where markers are painted --
    and leaves the 3D view exactly as it was, while napari raises a
    warning that the app surfaces as a toast every single time the box
    is ticked.  The property set is correct and stays; the toast is
    noise about a limitation the help line now states outright.
    """

    def test_napari_really_refuses_contours_in_3D(self):
        """If this ever stops being true, the wording below is a lie and
        the suppression is hiding something real."""
        import inspect

        from napari.layers.labels.labels import Labels

        src = inspect.getsource(Labels._calculate_contour)
        assert "Contours are not displayed during 3D rendering" in src
        assert "labels.ndim > 2" in src

    def _warning_layer(self):
        import warnings as _w

        class _WarnLayer:
            """A layer that warns the way napari's does."""

            def __init__(self):
                object.__setattr__(self, "contour", 0)

            def __setattr__(self, key, value):
                if key == "contour" and value:
                    _w.warn("Contours are not displayed during 3D rendering")
                object.__setattr__(self, key, value)

        return _WarnLayer()

    def test_the_toggle_does_not_let_the_warning_out(self):
        import warnings

        f = _bind(_Fake(), "_on_refine_contour_toggled",
                  "_set_refine_contour")
        f._refine_layer = self._warning_layer()
        with warnings.catch_warnings():
            warnings.simplefilter("error")      # any escape is a failure
            f._on_refine_contour_toggled(True)
        assert f._refine_layer.contour == 1

    def test_a_DIFFERENT_warning_is_still_allowed_out(self):
        """The filter names the one message.  Swallowing everything a
        layer might say would hide the next real problem."""
        import warnings

        class _OtherWarnLayer:
            def __init__(self):
                object.__setattr__(self, "contour", 0)

            def __setattr__(self, key, value):
                if key == "contour" and value:
                    warnings.warn("something else entirely")
                object.__setattr__(self, key, value)

        f = _bind(_Fake(), "_on_refine_contour_toggled",
                  "_set_refine_contour")
        f._refine_layer = _OtherWarnLayer()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            f._on_refine_contour_toggled(True)
        assert any("something else entirely" in str(w.message)
                   for w in caught), [str(w.message) for w in caught]

    def test_the_one_writer_suppresses_it(self):
        import inspect

        src = inspect.getsource(_K()._set_refine_contour)
        assert "catch_warnings" in src
        assert "filterwarnings" in src
        from napari_epicycle.annotator import curation_3d_widget as C

        assert C._CONTOUR_3D_WARNING == "Contours are not displayed"

    def test_both_assignment_sites_go_through_that_one_writer(self):
        """Two places write `contour`; neither may do it raw, or half
        the toasts come back."""
        import inspect

        for name in ("_on_refine_contour_toggled", "_create_refine_layer"):
            src = inspect.getsource(getattr(_K(), name))
            assert "_set_refine_contour" in src, name
            assert ".contour = " not in src, (
                "{} writes contour directly, bypassing the filter".format(
                    name)
            )


class TestTheWordsMatchWhatActuallyHappens:
    """The first version promised the 3D view would outline too.  It
    does not, and a help line that describes a thing the tool does not
    do is worse than no help line."""

    def test_the_help_line_says_the_3D_view_is_unchanged(self):
        from napari_epicycle.annotator import advanced_features_widget as M

        text = M.HELP_LINES["split_contour_cb"]
        assert "side views" in text
        assert "3D" in text, text

    def test_the_help_line_does_not_promise_the_3D_view(self):
        from napari_epicycle.annotator import advanced_features_widget as M

        text = M.HELP_LINES["split_contour_cb"].lower()
        for lie in ("in the 3d view too", "including the 3d",
                    "and the 3d view"):
            assert lie not in text, lie

    def test_the_console_line_says_it_too(self):
        import inspect

        src = inspect.getsource(_K()._on_refine_contour_toggled)
        assert "side views" in src

    def test_the_guide_says_it(self):
        import pathlib as _p

        guide = _p.Path("docs/guides/annotator_3d_reshape.md").read_text(
            encoding="utf-8")
        i = guide.index("Show this object as an outline")
        near = guide[i:i + 1400]
        assert "3D view" in near
        assert "side views" in near
