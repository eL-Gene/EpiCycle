"""The Workbench's "Colour by" and the viewer's labels show the SAME colours
(owner question 2026-10-05; live driver ``drive_wb_colour.py``).

Measured on the owner's data before the fix: continuous features, Cluster,
Border cell and Topology matched on every cell, but

* C1 -- QC tier: the viewer used Step 3's five tier colours, the plot the
  Workbench's three buckets (garbage red vs orange, review yellow vs grey,
  curated cyan vs grey, unmarked (140,150,170) vs (139,144,154));
* C2 -- "QC status", and "QC tier" with no ``qc_tier`` column: the plot drew
  its fallback grey, the viewer KEPT THE PREVIOUS colouring;
* C3 -- "Colour plot by this" set the combo's text but redrew nothing (and
  recoloured the viewer even with "Colours in viewer" unticked);
* C4 -- a NaN cell: lowest colour in the plot, RGB NaN in the viewer
  (Linear, Log); a constant feature on Linear: 0.5 vs 0;
* C5 -- HDBSCAN noise alpha 0.3 vs 0.5; a removed cell kept a tier colour;
* C6 -- every redraw recoloured the viewer, identical colours included.

The rule now: the viewer takes the colours the plot draws
(``QCPlotWidget.viewer_colours``, Workbench layout only), so no option can
drift.  A real ``AnalysisControlsWidget``, a real napari ``ViewerModel``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ._workbench_ctrl import app, pump, wired

napari = pytest.importorskip("napari")

N = 60
#: options whose colour is a category, not a scale
CATEGORICAL = ("QC status", "QC tier", "Cluster", "Border cell", "Topology")
CONTINUOUS = ("with_nan", "constant", "area")
SCALES = ("Linear", "Log", "Rank")
#: the row that was removed (a ghost in the plot, no voxels in the volume)
DELETED_ID = 7


def _df(*, with_tier=True):
    from napari_epicycle.fusion_reconstruction.qc_features import PCA_FEATURES

    rng = np.random.RandomState(3)
    cols = {"label": np.arange(1, N + 1)}
    for f in list(PCA_FEATURES)[:6]:
        cols[f] = rng.lognormal(size=N)
    cols["area"] = rng.lognormal(3.0, 1.0, size=N)
    nan_col = rng.rand(N) * 10
    nan_col[[2, 11, 30]] = np.nan
    cols["with_nan"] = nan_col
    cols["constant"] = np.full(N, 4.2)
    if with_tier:
        legacy = ["good", "bad", "garbage", "review", "curated", ""]
        cols["qc_tier"] = [legacy[i % len(legacy)] for i in range(N)]
    cols["cluster_id"] = [(-1 if i % 7 == 0 else i % 4) for i in range(N)]
    cols["is_border_cell"] = (np.arange(N) % 5 == 0)
    cols["euler_number"] = [(1, 2, 0, -1)[i % 4] for i in range(N)]
    status = ["active"] * N
    status[DELETED_ID - 1] = "deleted"
    cols["label_status"] = status
    return pd.DataFrame(cols)


def _setup(tmp_path, df):
    from napari.components import ViewerModel

    from napari_epicycle.ui_shell.layer_lifecycle_manager import (
        LayerLifecycleManager,
    )
    app()
    c, wb = wired(tmp_path, df)
    vol = np.zeros((1, 2, N + 2), np.uint32)
    for lid in range(1, N + 1):
        if lid != DELETED_ID:
            vol[0, 0, lid] = lid
    c._labels_volume = vol
    viewer = ViewerModel()
    c.viewer = viewer
    c.layer_lifecycle_manager = LayerLifecycleManager(viewer)
    c._add_stack_labels_layer()
    c._step4_active = True
    pw = wb.qc_plot_widget
    pw.viewer_color_requested.connect(c._on_viewer_color_requested)
    wb.sync_viewer_toggled.connect(c._on_sync_toggled)
    wb.color_by_feature.connect(c._on_eda_color_by_feature)
    pw.plot_type_combo.setCurrentText("Scatter")
    pump()
    return c, wb, pw


def _pick_colour(pw, text):
    """What a user's pick does: the combo, then its ``activated``."""
    idx = pw._color_combo.findText(text)
    assert idx >= 0, (text, [pw._color_combo.itemText(i)
                             for i in range(pw._color_combo.count())])
    pw._color_combo.setCurrentIndex(idx)
    pw._color_combo.activated.emit(idx)
    pump()


def _main_scatter(pw):
    import pyqtgraph as pg
    n = len(pw._scatter_label_ids)
    best = None
    for it in pw._plot_widget.getPlotItem().items:
        if (isinstance(it, pg.ScatterPlotItem) and len(it.data) == n and n
                and it.opts.get("symbol") != "x"):
            best = it
    return best


def plot_rgb(pw):
    """The colours the plot actually DREW, by label id (0-255)."""
    sc = _main_scatter(pw)
    assert sc is not None
    out = {}
    for i, b in enumerate(sc.data["brush"]):
        if b is None:
            b = sc.opts["brush"]
        col = b.color()
        out[int(pw._scatter_label_ids[i])] = (col.red(), col.green(),
                                              col.blue())
    return out


def viewer_rgba(c, ids):
    lyr = c._find_labels_layer()
    arr = np.asarray(sorted(ids), dtype=np.int64)
    cols = np.asarray(lyr.colormap.map(arr), dtype=np.float64)
    return {int(i): cols[k] for k, i in enumerate(arr)}


def _assert_same(c, pw, what):
    plot = plot_rgb(pw)
    assert DELETED_ID not in plot                  # a ghost, not a point
    view = viewer_rgba(c, plot)
    bad = []
    for lid, rgb in plot.items():
        v = view[lid]
        assert np.all(np.isfinite(v)), (what, lid, v)
        d = float(np.max(np.abs(np.asarray(rgb, float) - v[:3] * 255)))
        if d > 2:
            bad.append((lid, rgb, tuple(np.round(v[:3] * 255, 1))))
    assert not bad, (what, len(bad), bad[:4])


def _assert_plot_drew_its_combo(pw):
    """The plot DREW what its Colour by names (not only the combo text)."""
    want = pw._get_point_colors_for(pw._df.loc[pw._scatter_to_df_index])
    drawn = plot_rgb(pw)
    worst = max(float(np.max(np.abs(np.asarray(drawn[int(lid)])
                                    - want[k][:3] * 255)))
                for k, lid in enumerate(pw._scatter_label_ids))
    assert worst <= 2, worst


def _options():
    out = [(opt, None) for opt in CATEGORICAL]
    out += [(opt, scale) for opt in CONTINUOUS for scale in SCALES]
    return out


@pytest.mark.parametrize("option,scale", _options(),
                         ids=[f"{o}-{s}" for o, s in _options()])
def test_every_colour_by_option_shows_the_same_colours(tmp_path, option,
                                                       scale):
    c, wb, pw = _setup(tmp_path, _df())
    if scale is not None:
        pw._norm_combo.setCurrentText(scale)
    _pick_colour(pw, "Topology")          # a colouring that is NOT the next
    _pick_colour(pw, option)
    _assert_same(c, pw, (option, scale))


def test_qc_tier_with_no_tier_column_paints_the_plot_grey(tmp_path):
    """C2: the viewer never keeps a colouring the combo does not name."""
    c, wb, pw = _setup(tmp_path, _df(with_tier=False))
    _pick_colour(pw, "Topology")
    _pick_colour(pw, "QC tier")
    _assert_same(c, pw, "QC tier, no column")
    _pick_colour(pw, "QC status")
    _assert_same(c, pw, "QC status, no column")


def test_a_removed_cell_gets_no_colour(tmp_path):
    """C5: the removed row is not coloured by its old tier."""
    c, wb, pw = _setup(tmp_path, _df())
    _pick_colour(pw, "QC tier")
    lyr = c._find_labels_layer()
    rgba = np.asarray(lyr.colormap.map(np.array([DELETED_ID])))[0]
    assert rgba[3] == 0, rgba


def test_noise_alpha_is_the_plots(tmp_path):
    """C5: HDBSCAN noise (-1) has the plot's alpha in the viewer."""
    c, wb, pw = _setup(tmp_path, _df())
    _pick_colour(pw, "Cluster")
    lyr = c._find_labels_layer()
    noise = int(_df()["label"][_df()["cluster_id"] == -1].iloc[0])
    alpha = float(np.asarray(lyr.colormap.map(np.array([noise])))[0][3])
    assert abs(alpha - 0.5) < 0.01, alpha


# --------------------------------------------------------------------------
# C3: "Colour plot by this"
# --------------------------------------------------------------------------

def test_colour_plot_by_this_recolours_the_plot_and_the_viewer(tmp_path):
    c, wb, pw = _setup(tmp_path, _df())
    _pick_colour(pw, "Topology")
    wb.color_by_feature.emit("area")
    pump()
    assert pw._color_combo.currentText() == "area"
    _assert_plot_drew_its_combo(pw)
    _assert_same(c, pw, "Colour plot by this")


def test_colour_plot_by_this_leaves_an_unticked_viewer_alone(tmp_path):
    c, wb, pw = _setup(tmp_path, _df())
    _pick_colour(pw, "Topology")
    wb.sync_viewer_colors_cb.setChecked(False)
    pump()
    lyr = c._find_labels_layer()
    before = np.asarray(lyr.colormap.map(np.arange(1, N + 1)))
    wb.color_by_feature.emit("area")
    pump()
    after = np.asarray(lyr.colormap.map(np.arange(1, N + 1)))
    assert np.allclose(before, after)


def test_unpressing_colour_plot_by_this_goes_back_on_both(tmp_path):
    c, wb, pw = _setup(tmp_path, _df())
    _pick_colour(pw, "Topology")
    wb.color_by_feature.emit("area")
    pump()
    wb.color_by_feature.emit("")
    pump()
    assert pw._color_combo.currentText() == "Topology"
    _assert_plot_drew_its_combo(pw)
    _assert_same(c, pw, "unpressed")


def test_the_default_view_redraws_what_it_names(tmp_path):
    """C3: _set_default_plot_view routes through the same redraw."""
    df = _df()
    df["qc_score"] = np.linspace(0, 1, N)
    c, wb, pw = _setup(tmp_path, df)
    # only the colour differs from the default view, so nothing else
    # redraws on its way
    pw._cmap_combo.setCurrentText("Coolwarm")
    pw.plot_type_combo.setCurrentText("PCA")
    pump()
    _pick_colour(pw, "Topology")
    c._set_default_plot_view()
    pump()
    assert pw._color_combo.currentText() == "qc_score"
    _assert_plot_drew_its_combo(pw)
    _assert_same(c, pw, "default view")


# --------------------------------------------------------------------------
# C6: the viewer is recoloured only when its colours change
# --------------------------------------------------------------------------

def test_a_redraw_with_the_same_colours_does_not_recolour_the_viewer(
        tmp_path):
    c, wb, pw = _setup(tmp_path, _df())
    _pick_colour(pw, "area")
    n0 = c._viewer_recolours
    pw.plot_type_combo.setCurrentText("Violin")        # a tab switch
    pump()
    pw.plot_type_combo.setCurrentText("Scatter")
    pump()
    assert c._viewer_recolours == n0
    _pick_colour(pw, "Topology")                       # a colour-by change
    assert c._viewer_recolours == n0 + 1


def test_a_new_labels_layer_or_a_retick_is_recoloured(tmp_path):
    c, wb, pw = _setup(tmp_path, _df())
    _pick_colour(pw, "area")
    n0 = c._viewer_recolours
    wb.sync_viewer_colors_cb.setChecked(False)
    pump()
    wb.sync_viewer_colors_cb.setChecked(True)
    pump()
    assert c._viewer_recolours == n0 + 1
    lyr = c._find_labels_layer()
    c.viewer.layers.remove(lyr)
    c._add_stack_labels_layer()
    c._reapply_viewer_colours("test")
    assert c._viewer_recolours == n0 + 2
    _assert_same(c, pw, "re-added layer")
