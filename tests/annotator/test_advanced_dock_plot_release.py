"""The Advanced panel's pyqtgraph plots are let go BEFORE the panel goes.

Traced 2026-11-14 on a live napari (one Add session: Whole patch, Close
box, then close napari): the close died with "QObject: shared QObject was
deleted directly", "Fatal Python error: Aborted", exit 3.  The Add page's
in-box histogram is a pyqtgraph ``PlotWidget`` built lazily the first time
an Add box measures itself, and it lived on, hidden, inside the panel until
the panel was destroyed with the annotator (``_destroy_satellite_docks``).
Six runs with that plot alive at the close died; five runs that never
built it, and one where the driver unparented and deleted it on its own
before the close, exited 0.  No Add thread or worker was alive at the
close in any run.

The panel now has ``release_plots()``: each lazily built plot is
unparented and scheduled for deletion on its own, and
``_destroy_satellite_docks`` calls it before it closes the panel.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from napari_epicycle.annotator import launcher as L
from napari_epicycle.annotator.advanced_features_widget import (
    AdvancedFeaturesWidget,
)

from .test_advanced_dock_x_hides import (  # noqa: F401 - fixtures
    _flush,
    _is_deleted,
    napari_viewer_class,
    qapp,
    viewer,
)


def _panel_with_plots(viewer):
    widget, ok, msg = L.launch_advanced_features_dock(viewer,
                                                      mode="3d_curation")
    assert ok, msg
    widget._update_add_histogram({"counts": [], "edges": []})
    widget._update_detect_histogram({"edt_values": [1.0, 2.0, 2.5],
                                     "n_bins": 4})
    add_pw, det_pw = widget._add_plot_widget, widget._detect_plot_widget
    assert add_pw is not None and det_pw is not None
    assert add_pw.parent() is not None and det_pw.parent() is not None
    return widget, add_pw, det_pw


def test_release_lets_both_plots_go_on_their_own(qapp, napari_viewer_class,
                                                 viewer):
    widget, add_pw, det_pw = _panel_with_plots(viewer)
    widget.release_plots()
    assert widget._add_plot_widget is None
    assert widget._detect_plot_widget is None
    assert widget._add_thresh_line is None and widget._add_bar_item is None
    assert widget._detect_thresh_line is None
    assert widget._detect_bar_item is None
    for pw in (add_pw, det_pw):
        assert _is_deleted(pw) or pw.parent() is None
    _flush()
    assert _is_deleted(add_pw) and _is_deleted(det_pw)
    assert not _is_deleted(widget), "the panel itself stays"
    widget.release_plots()               # idempotent, never raises


def test_a_panel_that_never_built_a_plot_releases_nothing(
        qapp, napari_viewer_class, viewer):
    widget, ok, _msg = L.launch_advanced_features_dock(viewer,
                                                       mode="3d_curation")
    assert ok
    assert widget._add_plot_widget is None
    widget.release_plots()
    assert widget._add_plot_widget is None


def test_the_satellite_teardown_releases_the_plots_before_the_panel(
        qapp, napari_viewer_class, viewer, monkeypatch):
    widget, add_pw, det_pw = _panel_with_plots(viewer)
    seen = {}
    orig_close = AdvancedFeaturesWidget.close

    def spy_close(self):
        seen["at_close"] = [
            None if _is_deleted(pw) else pw.parent() for pw in (add_pw,
                                                               det_pw)]
        return orig_close(self)

    monkeypatch.setattr(AdvancedFeaturesWidget, "close", spy_close)
    L._destroy_satellite_docks(viewer)
    _flush()
    assert seen.get("at_close") == [None, None], (
        "a plot was still inside the panel when it closed: {!r}".format(
            seen.get("at_close")))
    assert _is_deleted(add_pw) and _is_deleted(det_pw)


# --------------------------------------------------------------------------- #
# the Add histogram goes with the Add session (Close box / Done / Cancel)
# --------------------------------------------------------------------------- #
def test_release_add_plot_puts_a_placeholder_back_and_a_new_box_rebuilds(
        qapp, napari_viewer_class, viewer):
    widget, ok, _msg = L.launch_advanced_features_dock(viewer,
                                                       mode="3d_curation")
    assert ok
    widget._update_add_histogram({"counts": [], "edges": []})
    pw = widget._add_plot_widget
    lay = pw.parentWidget().layout()
    idx = lay.indexOf(pw)
    assert idx >= 0 and widget._add_plot_placeholder is None
    widget.release_add_plot()
    ph = widget._add_plot_placeholder
    assert widget._add_plot_widget is None and ph is not None
    assert lay.indexOf(ph) == idx, "the placeholder takes the plot's place"
    assert _is_deleted(pw) or pw.parent() is None
    _flush()
    assert _is_deleted(pw)
    widget.release_add_plot()            # idempotent
    # the next box measures itself: a fresh plot, in the same place
    widget._update_add_histogram({"counts": [], "edges": []})
    pw2 = widget._add_plot_widget
    assert pw2 is not None and not _is_deleted(pw2)
    assert lay.indexOf(pw2) == idx
    assert widget._add_plot_placeholder is None


def test_every_add_session_end_releases_the_histogram(monkeypatch):
    from .add_harness import Harness, install_fakes, nuclei_volume
    from .test_add_nuclei_golden import box_of

    install_fakes(monkeypatch)
    img, labels = nuclei_volume()
    h = Harness(labels, images={"nuc": img})
    calls = []
    h.adv.release_add_plot = lambda: calls.append("released")
    h.place_box(box_of(labels.shape))
    h._reset_add_state()
    assert calls == ["released"]
    h.adv.release_add_plot = lambda: (_ for _ in ()).throw(
        RuntimeError("boom"))
    h._reset_add_state()                 # a failing release never raises


# --------------------------------------------------------------------------- #
# review round (2026-11-14): nothing rebuilds the plot outside a box, Accept
# releases it too, the placeholder is never a window
# --------------------------------------------------------------------------- #
def _idle_panel(viewer):
    widget, ok, msg = L.launch_advanced_features_dock(viewer,
                                                      mode="3d_curation")
    assert ok, msg
    return widget


def test_a_typed_threshold_on_an_idle_panel_builds_no_plot(
        qapp, napari_viewer_class, viewer):
    widget = _idle_panel(viewer)
    assert widget._add_plot_widget is None
    widget._add_threshold_is_pending = False
    widget.add_threshold_sb.setValue(0.4)
    widget.add_threshold_sb.editingFinished.emit()
    assert widget._add_plot_widget is None, "one focus-out built the plot"
    # after a session released it, the same
    widget._update_add_histogram({"counts": [], "edges": []})
    widget.release_add_plot()
    widget.add_threshold_sb.editingFinished.emit()
    assert widget._add_plot_widget is None


def test_the_next_plot_starts_its_line_at_the_box_value(
        qapp, napari_viewer_class, viewer):
    widget = _idle_panel(viewer)
    widget._sync_add_threshold_box(0.25)          # typed while no plot
    widget._update_add_histogram({"counts": [1.0, 2.0],
                                  "edges": [0.0, 0.5, 1.0]})
    assert widget._add_thresh_line is not None
    assert abs(widget._add_thresh_line.value() - 0.25) < 1e-9


def test_the_placeholder_is_parented_from_its_creation(
        qapp, napari_viewer_class, viewer, monkeypatch):
    from napari_epicycle.annotator import advanced_features_widget as AFW

    widget = _idle_panel(viewer)
    widget._update_add_histogram({"counts": [], "edges": []})
    pw = widget._add_plot_widget
    parent = pw.parentWidget()
    made = []
    real = AFW.QWidget

    class _Rec(real):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append((self, a[0] if a else k.get("parent")))

    monkeypatch.setattr(AFW, "QWidget", _Rec)
    widget.release_add_plot()
    assert made and made[0][1] is parent, "created parentless"
    ph = widget._add_plot_placeholder
    assert ph is made[0][0] and not ph.isWindow()


def test_a_hidden_plot_leaves_a_hidden_placeholder(
        qapp, napari_viewer_class, viewer):
    """During an Inside-each-cell session the histogram row is hidden: the
    placeholder that takes its place stays hidden."""
    widget = _idle_panel(viewer)
    widget._update_add_histogram({"counts": [], "edges": []})
    pw = widget._add_plot_widget
    parent = pw.parentWidget()
    pw.setVisible(False)
    widget.release_add_plot()
    ph = widget._add_plot_placeholder
    assert ph.parentWidget() is parent
    assert not ph.isVisibleTo(parent)


def test_a_torn_down_panel_builds_no_parentless_plot(
        qapp, napari_viewer_class, viewer):
    widget = _idle_panel(viewer)
    widget._update_add_histogram({"counts": [], "edges": []})
    widget._update_detect_histogram({"edt_values": [1.0, 2.0], "n_bins": 4})
    widget.release_plots()
    widget._update_add_histogram({"counts": [1.0], "edges": [0.0, 1.0]})
    widget._update_detect_histogram({"edt_values": [1.0, 2.0], "n_bins": 4})
    assert widget._add_plot_widget is None
    assert widget._detect_plot_widget is None


def test_accept_into_review_releases_the_histogram(monkeypatch):
    import numpy as np

    from .add_harness import Harness, install_fakes, nuclei_volume
    from .test_add_nuclei_golden import box_of

    install_fakes(monkeypatch)
    img, labels = nuclei_volume()
    h = Harness(labels, images={"nuc": img})
    h.place_box(box_of(labels.shape))
    h.adv.close_add_numbers = lambda: None
    shape = h.sync_manager.shared_labels["L"].shape
    res = np.zeros(tuple(s.stop - s.start for s in h._add_roi), np.int32)
    res[(slice(1, 3),) * 3] = 4000
    h._add_preview_result = res
    h._add_preview_roi = tuple(h._add_roi)
    h.adv.checked = [4000]
    calls = []
    h.adv.release_add_plot = lambda: calls.append("released")
    h._accept_add()
    assert h._add_review, "Accept goes to REVIEW (skips _reset_add_state)"
    assert calls == ["released"]
    assert shape == h.sync_manager.shared_labels["L"].shape


# --------------------------------------------------------------------------- #
# the auto-detect histogram goes with the Reshape session (driver, 11-14:
# Detect, Cancel, then napari's close exited 3 with it alive)
# --------------------------------------------------------------------------- #
def test_release_detect_plot_puts_a_placeholder_back(
        qapp, napari_viewer_class, viewer):
    widget = _idle_panel(viewer)
    widget._update_detect_histogram({"edt_values": [1.0, 2.0, 2.5],
                                     "n_bins": 4})
    pw = widget._detect_plot_widget
    lay = pw.parentWidget().layout()
    idx = lay.indexOf(pw)
    widget.release_detect_plot()
    ph = widget._detect_plot_placeholder
    assert widget._detect_plot_widget is None and ph is not None
    assert lay.indexOf(ph) == idx and not ph.isWindow()
    assert widget._detect_last_edt_values == []
    _flush()
    assert _is_deleted(pw)
    widget.release_detect_plot()          # idempotent
    # an idle page: a Bins change or a pushed threshold builds nothing
    widget.split_detect_bins_sb.setValue(
        widget.split_detect_bins_sb.value() + 1)
    widget._set_detect_threshold(1.5)
    assert widget._detect_plot_widget is None
    # the next Detect builds a fresh one, in the same place
    widget._update_detect_histogram({"edt_values": [1.0, 2.0], "n_bins": 4})
    assert lay.indexOf(widget._detect_plot_widget) == idx


def test_every_reshape_session_end_releases_the_detect_plot():
    from napari_epicycle.annotator.curation_3d_widget import (
        Curation3DAnnotator,
    )

    from .test_refinement_state_cleanup import (
        TestP01_ResetSplitStateClearsIsolationCache as _T,
    )

    ann = _T()._make_bare_annotator()
    # the bare annotator has no navigator: stub its refresher too
    ann._refresh_label_action_buttons_state = lambda: None
    calls = []

    class _Adv:
        def release_detect_plot(self):
            calls.append("released")

    ann._get_advanced_features_widget = lambda: _Adv()
    Curation3DAnnotator._reset_split_state(ann)
    assert calls == ["released"]
    assert ann._split_running is False
