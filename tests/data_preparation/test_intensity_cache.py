"""④ depth-profile cache — a re-select / subsection-return must NOT re-read the
TIFF + re-reduce (a reopened patch's plot is a full disk read otherwise), and the
cache-hit render is DEFERRED to the event loop (like the disk worker's queued
`done` signal) so it never rebuilds the plot mid-section-switch — a synchronous
rebuild there paints the LinearRegionItem before its ViewBox exists (crash).

Qt-free: the deferred render + populate are pure enough to exercise via the
unbound method on a ``SimpleNamespace`` stand-in.
"""

import types

from napari_epicycle.data_preparation.controller import DataPreparationController


def _ctl_with_item(name):
    item = types.SimpleNamespace(text=lambda: name)
    ctl = types.SimpleNamespace(
        widget=types.SimpleNamespace(
            extracted_roi_list=types.SimpleNamespace(currentItem=lambda: item)),
        _intensity_curve_cache={},
        extracted_patches={},
    )
    ctl._log = lambda *a, **k: None
    ctl._rendered = {}
    ctl._render_intensity_profile = lambda n, c, z: ctl._rendered.update(
        name=n, curves=c, n_z=z)
    return ctl


def test_cache_hit_defers_render_and_sets_snap(monkeypatch):
    ctl = _ctl_with_item("p1")
    ctl._intensity_curve_cache = {"p1": (["curve"], 10, (2, 8))}
    ctl._render_cached_intensity = types.MethodType(
        DataPreparationController._render_cached_intensity, ctl)
    scheduled = []
    import qtpy.QtCore as qtc
    monkeypatch.setattr(qtc.QTimer, "singleShot",
                        staticmethod(lambda ms, fn: scheduled.append(fn)))

    DataPreparationController._on_plot_intensity_clicked(ctl)

    # Snap range set synchronously; the render is DEFERRED (not run yet).
    assert ctl._snap_range == ("p1", (2, 8))
    assert ctl._rendered == {}
    # Firing the scheduled callback renders (the deferred tick).
    scheduled[0]()
    assert ctl._rendered == {"name": "p1", "curves": ["curve"], "n_z": 10}


def test_render_cached_renders_when_selection_matches():
    ctl = _ctl_with_item("p1")
    DataPreparationController._render_cached_intensity(ctl, "p1", ["curve"], 10)
    assert ctl._rendered == {"name": "p1", "curves": ["curve"], "n_z": 10}


def test_render_cached_stale_guard_when_selection_moved():
    ctl = _ctl_with_item("p1")   # p1 selected, but a p2 render is queued
    DataPreparationController._render_cached_intensity(ctl, "p2", ["c"], 10)
    assert ctl._rendered == {}   # discarded — selection moved


def test_ready_populates_the_cache():
    ctl = _ctl_with_item("p1")
    DataPreparationController._on_intensity_profile_ready(
        ctl, "p1", ["c"], 10, (2, 8))
    assert ctl._intensity_curve_cache["p1"] == (["c"], 10, (2, 8))
    assert ctl._snap_range == ("p1", (2, 8))


def test_ready_discards_when_selection_changed():
    ctl = _ctl_with_item("p1")
    DataPreparationController._on_intensity_profile_ready(
        ctl, "p2", ["c"], 10, (2, 8))
    assert "p2" not in ctl._intensity_curve_cache
    assert ctl._rendered == {}


# ---------------------------------------------------------------------------
# Skip the rebuild on a spurious re-select of the already-plotted patch
# ---------------------------------------------------------------------------

def _sel_ctl(cur_name, plotted, has_widget):
    item = types.SimpleNamespace(text=lambda: cur_name)
    ctl = types.SimpleNamespace(
        widget=types.SimpleNamespace(
            extracted_roi_list=types.SimpleNamespace(currentItem=lambda: item)),
        _highlighted_patch=None,
        _plot_n_z_patch=plotted,
        selected_z_range="SENTINEL",
    )
    ctl._refresh_conditioning_status = lambda: None
    ctl._plot_area_has_widget = lambda: has_widget
    ctl._plot_calls = []
    ctl._on_plot_intensity_clicked = lambda: ctl._plot_calls.append(1)
    return ctl


def test_selection_skips_replot_when_same_patch_plotted():
    ctl = _sel_ctl("p1", plotted="p1", has_widget=True)
    DataPreparationController._on_roi_selection_changed(ctl)
    assert ctl._plot_calls == []                 # rebuild skipped (no stutter)
    assert ctl.selected_z_range == "SENTINEL"    # dragged band preserved


def test_selection_replots_on_different_patch():
    ctl = _sel_ctl("p2", plotted="p1", has_widget=True)
    DataPreparationController._on_roi_selection_changed(ctl)
    assert ctl._plot_calls == [1]                # genuine change → re-plot
    assert ctl.selected_z_range is None          # band reset


def test_selection_replots_when_no_plot_widget_yet():
    ctl = _sel_ctl("p1", plotted="p1", has_widget=False)
    DataPreparationController._on_roi_selection_changed(ctl)
    assert ctl._plot_calls == [1]                # first render, nothing to skip
