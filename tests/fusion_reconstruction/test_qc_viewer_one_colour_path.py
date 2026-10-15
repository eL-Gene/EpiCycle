""""Colour" (brick P3): ONE colour path from the
QC plot to the napari viewer.

The viewer takes the colours the plot draws (``QCPlotWidget.viewer_colours``)
through ``_apply_plot_colours_to_viewer(why)`` -- Step 4's C1-C6 rule
(``tests/segmentation/test_workbench_colour_sync.py`` is the model), here for
Step 3: a bare controller (``object.__new__``) holding a REAL
``QCPlotArea`` (the amber Workbench plot with Step 3's texts, palette and
five-tier scheme) and a real napari ``ViewerModel`` with one or two labels
layers, over a five-tier table with ghosts, a NaN ``label_status``, clusters
with noise (-1), a border column, a constant and a with-NaN feature.

Held here: parity per Colour by option and Scale (viewer RGBA == plot RGBA
within 2/255); ghosts get no colour; isolation hides the rest and leaves the
isolated colours alone; the tick OFF writes nothing; the
identical-recolour skip and what invalidates it (a foreign write, a rebuilt
layer, a new colour); the authority's refusal; every layer written; the
hover properties; the one active rule (the plot's, Appendix C); decision
23's alphas.
"""
from __future__ import annotations

import inspect
import os
import types

import numpy as np
import pandas as pd
import pytest

napari = pytest.importorskip("napari")

N = 60
#: two ghosts (a deleted and a superseded row): no voxels, no colour
GHOST_IDS = (7, 19)
#: a row whose ``label_status`` is NaN: the plot draws it, so it is coloured
NAN_STATUS_ID = 23
CATEGORICAL = ("QC status", "QC tier", "Cluster", "Border cell")
CONTINUOUS = ("with_nan", "constant", "area")
SCALES = ("Linear", "Log", "Rank")

_APP = []


def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication

    a = QApplication.instance() or QApplication([])
    if not _APP:
        _APP.append(a)
    return a


def pump(n=3):
    from qtpy.QtWidgets import QApplication

    for _ in range(n):
        QApplication.processEvents()


def _df():
    from napari_epicycle.fusion_reconstruction.qc_features import PCA_FEATURES

    rng = np.random.RandomState(5)
    cols = {"label": np.arange(1, N + 1)}
    for f in list(PCA_FEATURES)[:6]:
        cols[f] = rng.lognormal(size=N)
    cols["area"] = rng.lognormal(3.0, 1.0, size=N)
    cols["volume"] = rng.lognormal(2.0, 1.0, size=N)
    nan_col = rng.rand(N) * 10
    nan_col[[2, 11, 30]] = np.nan
    cols["with_nan"] = nan_col
    cols["constant"] = np.full(N, 4.2)
    cols["qc_score"] = rng.rand(N)
    tiers = ["curated", "good", "review", "bad", "garbage", ""]
    cols["qc_tier"] = [tiers[i % len(tiers)] for i in range(N)]
    statuses = ["pass", "flagged", "excluded"]
    cols["qc_status"] = [statuses[i % 3] for i in range(N)]
    cols["cluster_id"] = [(-1 if i % 7 == 0 else i % 4) for i in range(N)]
    cols["is_border_cell"] = (np.arange(N) % 5 == 0).astype(int)
    status = ["active"] * N
    status[GHOST_IDS[0] - 1] = "deleted"
    status[GHOST_IDS[1] - 1] = "superseded"
    df = pd.DataFrame(cols)
    df["label_status"] = pd.Series(status, dtype=object)
    df.loc[df["label"] == NAN_STATUS_ID, "label_status"] = np.nan
    return df


def _volume():
    vol = np.zeros((1, 2, N + 2), np.uint32)
    for lid in range(1, N + 1):
        if lid not in GHOST_IDS:
            vol[0, 0, lid] = lid
    return vol


class _Rig:
    """A bare Step 3 controller wired to a real plot area and viewer."""

    def __init__(self, df=None, n_layers=1):
        app()
        from napari.components import ViewerModel

        from napari_epicycle.fusion_reconstruction._qc_plot_area import (
            QCPlotArea,
        )
        from napari_epicycle.fusion_reconstruction.colormap_authority import (
            ColormapAuthority,
        )
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController as C,
        )

        self.df = _df() if df is None else df
        self.area = QCPlotArea()
        self.area.resize(640, 760)
        self.pw = self.area.qc_plot_widget
        self.cb = self.area.colours_in_viewer_cb
        c = object.__new__(C)
        c.ui = types.SimpleNamespace(
            qc_plot_area=self.area, qc_plot_widget=self.pw,
            sync_viewer_colors_cb=self.cb)
        c._qc_features_df = self.df
        c._isolated_label_ids = set()
        c._colormap_authority = ColormapAuthority()
        c._qc_view_layers = []
        self.viewer = ViewerModel()
        c.viewer = self.viewer
        self.layers = [self.viewer.add_labels(_volume(),
                                              name=f"curation_3d_p{k}")
                       for k in range(n_layers)]
        rig = self
        c._find_all_patch_labels_layers = (
            lambda: [ly for ly in rig.layers if ly in rig.viewer.layers])
        c._find_active_labels_layer = (
            lambda: rig.layers[0] if rig.layers else None)
        self.c = c
        # The connection the controller's wiring makes.
        self.pw.viewer_color_requested.connect(c._on_plot_viewer_colours)
        self.cb.setChecked(True)
        self.pw.set_features(self.df)
        self.pw.plot_type_combo.setCurrentText("Scatter")
        pump()

    def pick(self, text, scale=None):
        if scale is not None:
            was = self.pw._norm_combo.blockSignals(True)
            self.pw._norm_combo.setCurrentText(scale)
            self.pw._norm_combo.blockSignals(was)
        assert self.pw.set_colour_by("Border cell")   # not the next one
        assert self.pw.set_colour_by(text), text
        pump()

    def viewer_rgba(self, ids, layer=None):
        lyr = self.layers[0] if layer is None else layer
        arr = np.asarray(sorted(int(i) for i in ids), dtype=np.int64)
        cols = np.asarray(lyr.colormap.map(arr), dtype=np.float64)
        return {int(i): cols[k] for k, i in enumerate(arr)}

    def plot_rgba(self):
        ids, rgba = self.pw.viewer_colours()
        return {int(i): np.asarray(rgba[k], np.float64)
                for k, i in enumerate(ids)}


def _drawn_rgb(pw):
    """The colours the plot actually DREW for its active points (0-255)."""
    import pyqtgraph as pg

    n = len(pw._scatter_label_ids)
    best = None
    for it in pw._plot_widget.getPlotItem().items:
        if (isinstance(it, pg.ScatterPlotItem) and len(it.data) == n and n
                and it.opts.get("symbol") != "x"):
            best = it
    assert best is not None
    out = {}
    for i, b in enumerate(best.data["brush"]):
        if b is None:
            b = best.opts["brush"]
        col = b.color()
        out[int(pw._scatter_label_ids[i])] = (col.red(), col.green(),
                                              col.blue(), col.alpha())
    return out


def _assert_parity(rig, what):
    plot = rig.plot_rgba()
    assert plot, what
    for g in GHOST_IDS:
        assert g not in plot, (what, g)
    for lyr in rig.layers:
        view = rig.viewer_rgba(plot, lyr)
        bad = []
        for lid, rgba in plot.items():
            v = view[lid]
            assert np.all(np.isfinite(v)), (what, lid, v)
            d = float(np.max(np.abs(np.nan_to_num(rgba) - v)) * 255)
            if d > 2:
                bad.append((lid, np.round(rgba * 255), np.round(v * 255)))
        assert not bad, (what, lyr.name, len(bad), bad[:3])


@pytest.fixture
def rig():
    r = _Rig()
    yield r
    r.area.deleteLater()
    pump()


def _options():
    out = [(opt, None) for opt in CATEGORICAL]
    out += [(opt, scale) for opt in CONTINUOUS for scale in SCALES]
    return out


# --------------------------------------------------------------------------
# Parity
# --------------------------------------------------------------------------

@pytest.mark.parametrize("option,scale", _options(),
                         ids=[f"{o}-{s}" for o, s in _options()])
def test_every_colour_by_shows_the_plots_colours(rig, option, scale):
    rig.pick(option, scale)
    _assert_parity(rig, (option, scale))


@pytest.mark.parametrize("option", ("QC tier", "area"))
def test_the_viewer_matches_what_the_plot_drew(rig, option):
    """Not only ``viewer_colours``: the brushes on the scatter."""
    rig.pick(option, "Rank")
    drawn = _drawn_rgb(rig.pw)
    view = rig.viewer_rgba(drawn)
    worst = max(float(np.max(np.abs(np.asarray(drawn[lid][:3], float)
                                    - view[lid][:3] * 255)))
                for lid in drawn)
    assert worst <= 2, (option, worst)


def test_ghosts_get_no_colour(rig):
    rig.pick("QC tier")
    lyr = rig.layers[0]
    rgba = np.asarray(lyr.colormap.map(np.array(GHOST_IDS)))
    assert np.all(rgba[:, 3] == 0), rgba


def test_the_active_rule_is_the_plots(rig):
    """Appendix C: a NaN-status row is drawn by the plot, so the viewer
    colours it; ``qc_features.active_rows`` (== "active") is not the rule
    the viewer follows."""
    from napari_epicycle.fusion_reconstruction.qc_features import active_rows

    assert NAN_STATUS_ID not in set(active_rows(rig.df)["label"])
    rig.pick("QC tier")
    ids, _ = rig.pw.viewer_colours()
    assert NAN_STATUS_ID in set(int(i) for i in ids)
    v = rig.viewer_rgba([NAN_STATUS_ID])[NAN_STATUS_ID]
    assert v[3] > 0, v
    _assert_parity(rig, "nan status")


def test_decision_23_status_alpha_and_noise_alpha(rig):
    rig.pick("QC status")
    v = rig.viewer_rgba([1])[1]
    assert abs(v[3] - 0.85) < 0.01, v              # was 1.0
    rig.pick("Cluster")
    noise = int(rig.df["label"][(rig.df["cluster_id"] == -1)
                                & (rig.df["label_status"] == "active")]
                .iloc[0])
    plot_alpha = rig.plot_rgba()[noise][3]
    v = rig.viewer_rgba([noise])[noise]
    assert abs(v[3] - plot_alpha) < 0.01, (v, plot_alpha)
    assert abs(v[3] - 0.3) > 0.1, v                # not the old 0.3


def test_every_layer_is_written():
    r = _Rig(n_layers=2)
    try:
        r.pick("area", "Linear")
        _assert_parity(r, "two layers")
        n0 = r.c._viewer_recolours
        # A foreign write on ONE layer: the next call writes, and only
        # that layer is stale.
        from napari.utils.colormaps import label_colormap
        keep = r.layers[0].colormap
        r.layers[1].colormap = label_colormap(49, seed=0.5)
        assert r.c._apply_plot_colours_to_viewer("one stale")
        assert r.c._viewer_recolours == n0 + 1
        assert r.layers[0].colormap is keep
        _assert_parity(r, "one stale layer")
    finally:
        r.area.deleteLater()
        pump()


# --------------------------------------------------------------------------
# Isolation
# --------------------------------------------------------------------------

def test_isolation_hides_the_rest_and_keeps_the_isolated(rig):
    rig.pick("QC tier")
    before = rig.viewer_rgba(range(1, N + 1))
    iso = {1, 2, 3, NAN_STATUS_ID}
    rig.c._isolated_label_ids = set(iso)
    assert rig.c._apply_plot_colours_to_viewer("test-isolate")
    after = rig.viewer_rgba(range(1, N + 1))
    for lid in range(1, N + 1):
        if lid in iso:
            assert np.allclose(after[lid], before[lid]), lid
        else:
            assert after[lid][3] == 0, (lid, after[lid])
    rig.c._isolated_label_ids = set()
    assert rig.c._apply_plot_colours_to_viewer("test-clear")
    _assert_parity(rig, "isolation cleared")


# --------------------------------------------------------------------------
# The tick
# --------------------------------------------------------------------------

def test_tick_off_resets_and_the_one_path_writes_nothing(rig):
    rig.pick("QC tier")
    calls = []
    real_reset = rig.c._reset_viewer_label_colors
    rig.c._reset_viewer_label_colors = (
        lambda: calls.append("reset") or real_reset())
    rig.cb.setChecked(False)
    rig.c._on_sync_viewer_colors_toggled(False)
    assert calls == ["reset"]
    lyr = rig.layers[0]
    cm = lyr.colormap
    n0 = rig.c._viewer_recolours
    assert rig.c._apply_plot_colours_to_viewer("tick-off") is False
    rig.pick("area", "Log")                    # the plot's own emit
    assert lyr.colormap is cm
    assert rig.c._viewer_recolours == n0


def test_tick_on_again_recolours(rig):
    rig.pick("area", "Rank")
    rig.cb.setChecked(False)
    rig.c._on_sync_viewer_colors_toggled(False)
    n0 = rig.c._viewer_recolours
    rig.cb.setChecked(True)
    rig.c._on_sync_viewer_colors_toggled(True)
    assert rig.c._viewer_recolours == n0 + 1
    _assert_parity(rig, "re-ticked")


# --------------------------------------------------------------------------
# The identical-recolour skip
# --------------------------------------------------------------------------

def test_an_identical_call_writes_nothing(rig):
    rig.pick("QC tier")
    lyr = rig.layers[0]
    cm = lyr.colormap
    n0 = rig.c._viewer_recolours
    assert rig.c._apply_plot_colours_to_viewer("same") is True
    assert lyr.colormap is cm
    assert rig.c._viewer_recolours == n0
    rig.pw._redraw()                           # a redraw, same colours
    pump()
    assert lyr.colormap is cm
    assert rig.c._viewer_recolours == n0


def test_a_new_colour_writes(rig):
    rig.pick("QC tier")
    n0 = rig.c._viewer_recolours
    assert rig.pw.set_colour_by("area")
    pump()
    assert rig.c._viewer_recolours == n0 + 1
    _assert_parity(rig, "new colour")


def test_a_foreign_write_invalidates_the_skip(rig):
    from napari.utils.colormaps import label_colormap

    rig.pick("QC tier")
    lyr = rig.layers[0]
    n0 = rig.c._viewer_recolours
    lyr.colormap = label_colormap(49, seed=0.5)          # someone else
    assert rig.c._apply_plot_colours_to_viewer("after foreign") is True
    assert rig.c._viewer_recolours == n0 + 1
    _assert_parity(rig, "after a foreign write")


def test_the_controllers_own_reset_invalidates_the_skip(rig):
    """``_reset_viewer_label_colors`` is a foreign writer too."""
    rig.pick("QC tier")
    rig.c._reset_viewer_label_colors()
    n0 = rig.c._viewer_recolours
    assert rig.c._apply_plot_colours_to_viewer("after reset") is True
    assert rig.c._viewer_recolours == n0 + 1
    _assert_parity(rig, "after the reset")


def test_a_rebuilt_layer_invalidates_the_skip(rig):
    """A rebuilt layer (new object, same name) is written even when it
    already holds the OLD colormap object -- so the layer key, not the
    colormap identity, is what invalidates the skip (P3 audit 5)."""
    rig.pick("area", "Linear")
    old = rig.layers[0]
    name = old.name
    old_cmap = old.colormap
    rig.viewer.layers.remove(old)
    new = rig.viewer.add_labels(_volume(), name=name)
    new.colormap = old_cmap                     # the same object as before
    assert new.colormap is old_cmap
    assert "area" not in new.properties
    rig.layers[0] = new
    n0 = rig.c._viewer_recolours
    assert rig.c._apply_plot_colours_to_viewer("rebuilt") is True
    assert rig.c._viewer_recolours == n0 + 1
    assert "area" in new.properties             # the hover landed on it
    _assert_parity(rig, "rebuilt layer")


def test_the_authority_refusing_writes_nothing(rig):
    from napari.utils.colormaps import label_colormap

    from napari_epicycle.fusion_reconstruction.colormap_authority import (
        ColormapOwner,
    )

    rig.pick("QC tier")
    lyr = rig.layers[0]
    rig.c._colormap_authority.reset()
    assert rig.c._colormap_authority.claim(ColormapOwner.ANNOTATOR_ISOLATION)
    lyr.colormap = label_colormap(49, seed=0.5)          # the annotator's
    cm = lyr.colormap
    n0 = rig.c._viewer_recolours
    assert rig.c._apply_plot_colours_to_viewer("refused") is False
    assert rig.pw.set_colour_by("area")
    pump()
    assert lyr.colormap is cm
    assert rig.c._viewer_recolours == n0


# --------------------------------------------------------------------------
# Hover
# --------------------------------------------------------------------------

def test_hover_properties_follow_colour_by(rig):
    rig.pick("area", "Linear")
    props = rig.layers[0].properties
    assert "area" in props
    arr = np.asarray(props["area"])
    row = rig.df[rig.df["label"] == 5].iloc[0]
    assert abs(float(arr[5]) - round(float(row["area"]), 2)) < 1e-6
    assert len(arr) == N + 1
    rig.pick("QC tier")
    props = rig.layers[0].properties
    assert "qc_tier" in props
    assert str(np.asarray(props["qc_tier"])[2]) == "good"


def test_hover_size_skips_ghosts():
    df = _df()
    df.loc[df["label"] == N, "label_status"] = "deleted"   # the max id
    r = _Rig(df=df)
    try:
        r.pick("area", "Linear")
        arr = np.asarray(r.layers[0].properties["area"])
        assert len(arr) == N            # ids 0..N-1: the ghost N is out
    finally:
        r.area.deleteLater()
        pump()


# --------------------------------------------------------------------------
# The old names and the wiring
# --------------------------------------------------------------------------

def test_the_feature_wrapper_is_the_one_path(rig):
    rig.pick("QC tier")
    calls = []
    orig = rig.c._apply_plot_colours_to_viewer
    rig.c._apply_plot_colours_to_viewer = (
        lambda why: calls.append(why) or orig(why))
    rig.c._update_viewer_feature_colors("area", "Viridis", "Linear")
    assert len(calls) == 1


def test_the_signal_slot_is_the_one_path(rig):
    rig.pick("QC tier")
    calls = []
    orig = rig.c._apply_plot_colours_to_viewer
    rig.c._apply_plot_colours_to_viewer = (
        lambda why: calls.append(why) or orig(why))
    rig.c._on_plot_viewer_colours("area", "Viridis", "Linear")
    assert len(calls) == 1


def _src():
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController as C,
    )
    return inspect.getsource(C)


def test_the_signal_is_wired_to_the_one_path():
    src = _src()
    assert ("viewer_color_requested.connect(self._on_plot_viewer_colours)"
            in src)
    assert ("viewer_color_requested.connect("
            "self._update_viewer_feature_colors)") not in src


def test_no_private_combo_read_chooses_a_viewer_colour():
    """Every colouring call site asks the one path; none reads the
    plot's private Colour by / Scale combos any more."""
    src = _src()
    for name in ("_color_combo", "_norm_combo"):
        assert f"pw._{name[1:]}.currentText()" not in src, name


def test_the_folded_methods_are_gone():
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController as C,
    )
    assert not hasattr(C, "_update_viewer_qc_colors")
    assert not hasattr(C, "_update_viewer_categorical_colors")


# --------------------------------------------------------------------------
# The plugin's viewer is napari's PublicOnlyProxy (live driver, 2026-10-06)
# --------------------------------------------------------------------------

def test_the_skip_holds_through_naparis_public_only_proxy(rig):
    """A plugin's ``viewer`` is a ``PublicOnlyProxy``: every
    ``viewer.layers[name]`` and every ``layer.colormap`` is a NEW wrapper.
    Keyed on the wrapper, the skip never fired live (every call looked like
    a new layer or a foreign write); it is keyed on the wrapped objects."""
    from napari.utils._proxies import PublicOnlyProxy

    rig.c._find_all_patch_labels_layers = (
        lambda: [PublicOnlyProxy(ly) for ly in rig.layers])
    rig.pick("QC tier")
    n0 = rig.c._viewer_recolours
    cm = rig.layers[0].colormap
    assert rig.c._apply_plot_colours_to_viewer("proxy same") is True
    rig.pw._redraw()
    pump()
    assert rig.c._viewer_recolours == n0
    assert rig.layers[0].colormap is cm
    assert rig.pw.set_colour_by("area")              # a real change
    pump()
    assert rig.c._viewer_recolours == n0 + 1
    _assert_parity(rig, "through the proxy")
    from napari.utils.colormaps import label_colormap
    rig.layers[0].colormap = label_colormap(49, seed=0.5)   # foreign
    assert rig.c._apply_plot_colours_to_viewer("proxy foreign") is True
    assert rig.c._viewer_recolours == n0 + 2


# --------------------------------------------------------------------------
# P3 audits (2026-10-06)
# --------------------------------------------------------------------------

def _side_annotator(rig, vms):
    """A stand-in for the 3D annotator: its sync manager's three side
    ``ViewerModel``s, and the annotator's REAL side-view propagation."""
    from napari_epicycle.annotator.curation_3d_widget import (
        Curation3DAnnotator,
    )

    name = rig.layers[0].name
    sync = types.SimpleNamespace(
        viewer_model_xy=vms[0], viewer_model_xz=vms[1],
        viewer_model_yz=vms[2], _base_label_name=lambda n: n)
    ann = types.SimpleNamespace(sync_manager=sync, _active_labels_name=name)
    ann._propagate_colormap_to_subviewers = types.MethodType(
        Curation3DAnnotator._propagate_colormap_to_subviewers, ann)
    return ann


def test_a_skipped_call_still_fixes_stale_side_views(rig):
    """HIGH 1: a cleared 3D multi-label isolation leaves isolation colours
    in the annotator's side views only (the annotator never writes the
    main layer); the controller's call finds the main layer identical and
    skips -- the skip must still bring the side views back to the main's
    colormap."""
    from napari.components import ViewerModel
    from napari.utils.colormaps import DirectLabelColormap

    vms = [ViewerModel() for _ in range(3)]
    sides = [vm.add_labels(_volume(), name=rig.layers[0].name) for vm in vms]
    ann = _side_annotator(rig, vms)
    rig.c._curation_annotator_widget = lambda: ann
    rig.pick("QC tier")
    main_cmap = rig.layers[0].colormap
    for s in sides:
        assert s.colormap is main_cmap          # the write propagated
    iso = DirectLabelColormap(color_dict={
        None: np.zeros(4, np.float32), 0: np.zeros(4, np.float32),
        1: np.array([1, 0, 0, 1], np.float32)})
    for s in sides:
        s.colormap = iso                        # the annotator's isolation
    n0 = rig.c._viewer_recolours
    assert rig.c._apply_plot_colours_to_viewer("iso-cleared") is True
    assert rig.c._viewer_recolours == n0        # main skipped
    assert rig.layers[0].colormap is main_cmap
    for s in sides:
        assert s.colormap is main_cmap, s


def test_a_skip_with_fresh_side_views_writes_nothing_there(rig):
    """The side-view check is cheap: when they already hold the main's
    colormap, a skipped call writes none of them."""
    from napari.components import ViewerModel

    vms = [ViewerModel() for _ in range(3)]
    sides = [vm.add_labels(_volume(), name=rig.layers[0].name) for vm in vms]
    ann = _side_annotator(rig, vms)
    rig.c._curation_annotator_widget = lambda: ann
    rig.pick("QC tier")
    events = []
    for s in sides:
        s.events.colormap.connect(lambda e: events.append(1))
    assert rig.c._apply_plot_colours_to_viewer("same") is True
    assert events == []


def test_clear_isolation_leaves_the_annotators_colormap_alone(rig):
    """LOW 2: with ANNOTATOR_ISOLATION owning the colormap the one path is
    refused, and ``_clear_label_isolation``'s default-colormap fallback must
    not overwrite the annotator's colours either."""
    from napari.utils.colormaps import label_colormap

    from napari_epicycle.fusion_reconstruction.colormap_authority import (
        ColormapOwner,
    )

    rig.c._set_plot_status = lambda *a, **k: None
    rig.c._diag_reshow = lambda layer: setattr(layer, "visible", True)
    rig.pick("QC tier")
    lyr = rig.layers[0]
    rig.c._colormap_authority.reset()
    assert rig.c._colormap_authority.claim(ColormapOwner.ANNOTATOR_ISOLATION)
    lyr.colormap = label_colormap(49, seed=0.7)          # the annotator's
    cm = lyr.colormap
    rig.c._clear_label_isolation()
    pump()
    assert lyr.colormap is cm
    assert lyr.visible


def test_clear_isolation_falls_back_without_an_annotator_owner(rig):
    """The fallback itself still runs when the one path cannot colour for
    any other reason (here: the tick is off)."""
    rig.c._set_plot_status = lambda *a, **k: None
    rig.c._diag_reshow = lambda layer: setattr(layer, "visible", True)
    rig.pick("QC tier")
    cm = rig.layers[0].colormap
    rig.cb.setChecked(False)
    rig.c._clear_label_isolation()
    assert rig.layers[0].colormap is not cm


def test_no_column_clears_the_hover(rig):
    """LOW 3: Colour by with no column of the table (a PC with no
    projection here) leaves no previous column's hover behind."""
    rig.pick("area", "Linear")
    assert "area" in rig.layers[0].properties
    rig.pw._color_combo.addItem("PC9")
    assert rig.pw.set_colour_by("PC9")
    pump()
    assert "area" not in rig.layers[0].properties
    assert dict(rig.layers[0].properties) == {}


def test_hover_values_are_in_the_digest(rig):
    """LOW 3: a value change that leaves the colours identical (Rank is
    blind to a monotonic change) still refreshes the hover."""
    rig.pick("area", "Rank")
    n0 = rig.c._viewer_recolours
    table = rig.pw._df
    table["area"] = table["area"] * 2.0                 # same ranks
    rig.pw._redraw()
    pump()
    assert rig.c._viewer_recolours == n0 + 1
    arr = np.asarray(rig.layers[0].properties["area"])
    row = table[table["label"] == 5].iloc[0]
    assert abs(float(arr[5]) - round(float(row["area"]), 2)) < 1e-6


def test_a_huge_sparse_id_skips_the_hover():
    """LOW 3: one id far above the rest would size the hover array by it;
    above ``10 * len(ids) + 1_000_000`` the hover is skipped (colours
    still written)."""
    df = _df()
    df.loc[df["label"] == 1, "label"] = 50_000_000
    r = _Rig(df=df)
    try:
        r.pick("area", "Linear")
        assert dict(r.layers[0].properties) == {}
        v = r.viewer_rgba([2])[2]
        assert v[3] > 0
    finally:
        r.area.deleteLater()
        pump()


def test_held_colours_write_nothing_and_the_load_writes_once(rig):
    """LOW 4: during the load chain (set_features with the PREVIOUS Colour
    by, then set_colour_by) the one path is held, and ONE
    explicit "load" call writes."""
    rig.pick("Border cell")
    n0 = rig.c._viewer_recolours
    rig.c._colours_held = True
    try:
        assert rig.c._apply_plot_colours_to_viewer("held") is False
        rig.pw.set_features(rig.df.copy())
        assert rig.pw.set_colour_by("qc_score", colours="Coolwarm",
                                    scale="Rank")
        pump()
    finally:
        rig.c._colours_held = False
    assert rig.c._viewer_recolours == n0
    assert rig.c._apply_plot_colours_to_viewer("load") is True
    assert rig.c._viewer_recolours == n0 + 1
    _assert_parity(rig, "after the load")


def test_the_load_chains_hold_and_push_once():
    """LOW 4: both load chains (a patch load's reload and an extract's
    finish) hold the one path around set_features ... the default colour,
    then push once with why="load"."""
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController as C,
    )

    for name in ("_reload_current_patch_features", "_on_qc_extract_finished"):
        src = inspect.getsource(getattr(C, name))
        a = src.index("self._colours_held = True")
        b = src.index("pw.set_features(df)")
        c = src.index('"_apply_default_plot_colour", pw)')
        d = src.index("self._colours_held = _held")
        e = src.index('"_apply_plot_colours_to_viewer", "load")')
        assert a < b < c < d < e, name


def test_unwrap_is_bounded():
    """LOW 6: a ``__wrapped__`` chain that never ends stops after 8 hops."""
    from napari_epicycle.fusion_reconstruction.controller import (
        _unwrap_proxy,
    )

    class Loop:
        @property
        def __wrapped__(self):
            return Loop()

    assert isinstance(_unwrap_proxy(Loop()), Loop)
    inner = object()
    chain = types.SimpleNamespace(__wrapped__=types.SimpleNamespace(
        __wrapped__=inner))
    assert _unwrap_proxy(chain) is inner
    assert _unwrap_proxy(inner) is inner
