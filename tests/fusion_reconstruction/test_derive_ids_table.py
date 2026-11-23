"""The table side of the reserved ids.

* The rows a re-extraction keeps for removed ids count every inactive
  status (``deleted`` and ``superseded``), so the id floor never falls below
  a superseded id.
* Those rows hold the old objects' values, PC columns included (another
  extraction's basis).  The re-extraction re-projects them into the new
  basis, fit on the live rows only, so the live PCs stay exactly as the
  extraction wrote them; their UMAP / t-SNE columns, which cannot be
  re-projected, are emptied.  The QC plot's own fallback fit is unchanged.
* A Delete or a Clear takes the removed ids' reshape verdicts out of the
  sidecar (a later object with the same id must not inherit "dividing"),
  and its Undo puts them back.
* Emptying the curation undo tells the open annotator, so an Add review
  whose Accepts can no longer be taken back closes.
"""

from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from napari_epicycle.fusion_reconstruction import _reshape_flags
from napari_epicycle.fusion_reconstruction import controller as C
from napari_epicycle.fusion_reconstruction.qc_features import (
    PCA_FEATURES,
    compute_pca,
    compute_pca_full,
)


# --------------------------------------------------------------------------- #
# the kept rows count superseded ids too
# --------------------------------------------------------------------------- #
def _prev():
    return pd.DataFrame({
        "label": [1, 2, 3, 9, 9, 12, 15],
        "label_status": ["active", "active", "deleted", "deleted", "deleted",
                         "superseded", "deleted"],
        "qc_tier": ["good", "review", "garbage", "good", "bad", "good",
                    "curated"],
        "area": [10, 20, 30, 40, 41, 50, 60],
    })


def _fresh():
    return pd.DataFrame({"label": [1, 2, 5], "area": [11, 21, 55],
                         "label_status": "active",
                         "qc_tier": ["good", "review", ""]})


def test_ids_above_the_fresh_top_count_a_superseded_id():
    assert C._deleted_ids_above(_prev(), _fresh()) == {9, 12, 15}


def test_a_superseded_id_alone_above_the_top_keeps_the_floor():
    prev = _prev()
    prev = prev[prev["label"] != 15]          # no deleted id above 12
    fresh = _fresh()
    above = C._deleted_ids_above(prev, fresh)
    assert 12 in above
    out = C._carry_cleared_rows(prev, fresh, above)
    assert int(pd.to_numeric(out["label"]).max()) == 12
    row = out[out["label"] == 12]
    assert len(row) == 1
    assert row["label_status"].iloc[0] == "superseded", \
        "a kept row keeps its own inactive status"


def test_the_kept_rows_are_inactive_and_once_per_id():
    out = C._carry_cleared_rows(_prev(), _fresh(), {3, 9, 12, 15})
    kept = out.iloc[3:]
    assert sorted(kept["label"]) == [3, 9, 12, 15]
    assert set(kept["label_status"]) <= {"deleted", "superseded"}
    assert out.iloc[:3]["label_status"].eq("active").all()


# --------------------------------------------------------------------------- #
# the kept rows never join the PCA fit
# --------------------------------------------------------------------------- #
def _active_and_carried(with_border):
    rng = np.random.default_rng(5)
    feats = list(PCA_FEATURES)[:12]
    n = 300
    act = pd.DataFrame(rng.lognormal(0, 0.6, (n, len(feats))), columns=feats)
    act["label"] = np.arange(1000, 1000 + n)
    act["label_status"] = "active"
    act["qc_tier"] = ""
    if with_border:
        act["is_border_cell"] = rng.random(n) < 0.1
    prev = pd.DataFrame({"label": np.arange(1, 845),
                         "label_status": "deleted", "qc_tier": "good"})
    for c in feats:
        prev[c] = rng.lognormal(0, 0.6, 844)
    if with_border:
        prev["is_border_cell"] = rng.random(844) < 0.1
    carried = C._carry_cleared_rows(prev, act, set(range(1, 845)))
    assert len(carried) == n + 844
    return act, carried, feats, n


@pytest.mark.parametrize("with_border", [False, True])
def test_kept_rows_do_not_move_the_active_points(with_border):
    act, carried, feats, n = _active_and_carried(with_border)
    p0 = compute_pca(act, n_components=2, features=feats)
    full = compute_pca(carried, n_components=2, features=feats,
                       exclude_inactive=True)
    p1 = full[:n].copy()
    for k in range(2):
        if np.dot(p0[:, k], p1[:, k]) < 0:
            p1[:, k] *= -1
    assert np.abs(p0 - p1).max() < 1e-9
    assert np.isfinite(full[n:]).all(), "the kept rows are still projected"


def test_the_plot_fallback_fit_is_unchanged():
    """The plot's own fit (a table with no PC columns: Step 4) fits every
    row as before: Step 4's map after "Remove bad cells" must agree with
    its Workbench PCA."""
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
        QCPlotWidget,
    )

    _act, carried, feats, _n = _active_and_carried(True)
    pw = types.SimpleNamespace(_df=carried, _pca_cache=None,
                               _pca_features=feats)
    got = QCPlotWidget._get_pca_projection(pw)
    want = compute_pca(carried, n_components=2, features=feats)
    assert np.array_equal(got, want)


# --------------------------------------------------------------------------- #
# the real Step 3 path: the extraction wrote PC columns on the live table,
# the carry brings rows with another extraction's PCs / UMAP / t-SNE
# --------------------------------------------------------------------------- #
_EMBED = ("UMAP_1", "UMAP_2", "tSNE_1", "tSNE_2")
_PCS = ["PC1", "PC2", "PC3"]


def _live_with_pcs(n, feats, nan_cell=False, seed=3):
    from napari_epicycle.fusion_reconstruction.qc_features import (
        compute_pca_columns,
    )

    rng = np.random.default_rng(seed)
    live = pd.DataFrame(rng.lognormal(0, 0.6, (n, len(feats))),
                        columns=feats)
    if nan_cell:
        live.loc[0, feats[0]] = np.nan
    live["label"] = np.arange(1, n + 1)
    live["label_status"] = "active"
    live["qc_tier"] = ""
    compute_pca_columns(live, n_components=3, features=feats)
    for c in _EMBED:
        live[c] = rng.normal(size=n)
    return live


def _prev_old_basis(feats, ids, seed=4):
    rng = np.random.default_rng(seed)
    prev = pd.DataFrame(rng.lognormal(1.0, 0.9, (len(ids), len(feats))),
                        columns=feats)
    prev["label"] = list(ids)
    prev["label_status"] = "deleted"
    prev["qc_tier"] = "good"
    for c in _PCS:
        prev[c] = 100.0
    for c in _EMBED:
        prev[c] = 50.0
    return prev


def _reextract_with(prev, fresh, tmp_path, feats):
    from .test_qc_tier_preservation import _make_bare_controller

    ctrl = _make_bare_controller(prev_df=prev, working_dir=tmp_path)
    ctrl._qc_settings = {"pca_features": feats, "n_cluster_components": 3,
                         "exclude_border": False}
    C.FusionReconstructionController._on_qc_extract_finished(
        ctrl, fresh, "p.tif")
    return ctrl._qc_features_df


@pytest.mark.parametrize("nan_cell", [False, True])
def test_kept_rows_are_re_projected_and_the_live_pcs_stay(tmp_path,
                                                          nan_cell):
    feats = list(PCA_FEATURES)[:8]
    live = _live_with_pcs(40, feats, nan_cell=nan_cell)
    before = live.sort_values("label").copy()
    prev = _prev_old_basis(feats, range(500, 520))
    out = _reextract_with(prev, live, tmp_path, feats)
    act = out[out["label_status"] == "active"].sort_values("label")
    kept = out[out["label_status"] == "deleted"]
    assert sorted(kept["label"]) == list(range(500, 520))
    assert np.abs(act[_PCS].to_numpy()
                  - before[_PCS].to_numpy()).max() < 1e-9
    assert np.isfinite(kept[_PCS].to_numpy()).all()
    assert not np.isclose(kept[_PCS].to_numpy(), 100.0).any(), \
        "the old basis is gone"
    want = compute_pca_full(out, n_components=3, features=feats,
                            exclude_inactive=True)["projected"]
    sel = (out["label_status"] == "deleted").to_numpy()
    assert np.abs(out[_PCS].to_numpy()[sel] - want[sel]).max() < 1e-9
    assert kept[list(_EMBED)].isna().all().all(), "no stale map position"
    assert np.array_equal(act[list(_EMBED)].to_numpy(),
                          before[list(_EMBED)].to_numpy())


def test_one_live_row_keeps_its_pcs_and_the_kept_rows_have_none(tmp_path):
    feats = list(PCA_FEATURES)[:8]
    live = _live_with_pcs(1, feats)
    before = live[_PCS].to_numpy().copy()
    prev = _prev_old_basis(feats, range(500, 506))
    out = _reextract_with(prev, live, tmp_path, feats)
    act = out[out["label_status"] == "active"]
    kept = out[out["label_status"] == "deleted"]
    assert len(kept) == 6
    assert np.array_equal(act[_PCS].to_numpy(), before)
    assert kept[_PCS + list(_EMBED)].isna().all().all()


def test_pca_columns_opt_in_matches_a_live_only_fit_with_nan_cells():
    from napari_epicycle.fusion_reconstruction.qc_features import (
        compute_pca_columns,
    )

    feats = list(PCA_FEATURES)[:8]
    live = _live_with_pcs(40, feats, nan_cell=True)
    prev = _prev_old_basis(feats, range(500, 520))
    both = pd.concat([live.drop(columns=_PCS), prev.drop(columns=_PCS)],
                     ignore_index=True)
    compute_pca_columns(both, n_components=3, features=feats,
                        exclude_inactive=True)
    a = both.iloc[:40][_PCS].to_numpy()
    b = live[_PCS].to_numpy()
    assert np.abs(a - b).max() < 1e-9


def test_a_ghost_with_no_position_is_not_drawn():
    from qtpy.QtWidgets import QApplication

    from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
        QCPlotWidget,
    )

    _app = QApplication.instance() or QApplication([])
    added = []
    pw = types.SimpleNamespace(
        _layout_mode="classic", _GHOST_COLOR=(150, 150, 150, 80),
        _plot_widget=types.SimpleNamespace(addItem=added.append))
    x = np.array([0.0, 1.0, np.nan, 3.0])
    y = np.array([0.0, np.nan, 2.0, 3.0])
    QCPlotWidget._add_ghost_scatter(pw, x, y, np.ones(4, bool))
    assert len(added) == 1
    assert len(added[0].data) == 2, "only the ghosts with a position"
    added.clear()
    QCPlotWidget._add_ghost_scatter(pw, x[[1, 2]], y[[1, 2]],
                                    np.ones(2, bool))
    assert added == [], "no ghost with a position: nothing drawn"


def test_the_inactive_rule_is_opt_in():
    """Callers that pass a table with removed rows and no keyword (Step 4's
    workers, the plot's own fit) fit exactly as before."""
    act, carried, feats, n = _active_and_carried(True)
    a = compute_pca_full(carried, n_components=2, features=feats)
    b = compute_pca_full(carried.assign(label_status="active"),
                         n_components=2, features=feats)
    assert np.array_equal(a["projected"], b["projected"])


def test_pca_full_with_every_row_inactive_still_fits():
    rng = np.random.default_rng(1)
    feats = list(PCA_FEATURES)[:5]
    df = pd.DataFrame(rng.lognormal(0, 0.5, (20, 5)), columns=feats)
    df["label_status"] = "deleted"
    out = compute_pca_full(df, n_components=2, features=feats,
                           exclude_inactive=True)
    assert out["projected"].shape == (20, 2)
    assert np.isfinite(out["projected"]).all()


# --------------------------------------------------------------------------- #
# a Delete or a Clear drops the removed ids' reshape verdicts; Undo puts them
# back
# --------------------------------------------------------------------------- #
class _Layer:
    def __init__(self, data):
        self.data = data

    def refresh(self):
        pass


def _ctrl(tmp_path, labels=(7, 8, 9)):
    from packages.epicycle_annotator.curation_undo import CurationUndoStack

    vol = np.zeros((4, 4, 4), np.int32)
    for k, lab in enumerate(labels):
        vol[k, 0, 0] = lab
    c = C.FusionReconstructionController.__new__(
        C.FusionReconstructionController)
    c.working_dir = str(tmp_path)
    c._qc_view_patch_name = "p01.tif"
    c._qc_table_patch = "p01.tif"
    c._qc_features_df = pd.DataFrame({
        "label": list(labels), "label_status": "active",
        "qc_tier": "review", "area": [5] * len(labels)})
    c._curation_undo_stack = CurationUndoStack(max_depth=20)
    c._curation_dirty = False
    c._tiers_dirty = False
    c._curation_dirty_epoch = 0
    c._run_epoch = 1
    c._qc_settings = {"spacing": None}
    c._current_downsample_factor = lambda *a, **k: 1
    c._append_log = lambda msg: None
    c._refresh_curation_ui = lambda: None
    layer = _Layer(vol)
    c._find_active_labels_layer = lambda: layer
    c.viewer = types.SimpleNamespace(
        window=types.SimpleNamespace(_dock_widgets={}))
    return c, layer


def _flags(tmp_path):
    return _reshape_flags.read_flags(str(tmp_path), "p01")


def test_a_delete_drops_the_reshape_verdict_and_undo_puts_it_back(tmp_path):
    c, layer = _ctrl(tmp_path)
    _reshape_flags.write_flags(str(tmp_path), "p01",
                               {7: "dividing", 8: "low-evidence"})
    roi = (slice(0, 1), slice(0, 1), slice(0, 1))
    old = layer.data[roi].copy()
    layer.data[roi] = 0
    c._on_annotator_label_deleted(7, roi_slices=roi, old_roi_data=old)
    assert _flags(tmp_path) == {8: "low-evidence"}
    c._undo_last_curation()
    assert _flags(tmp_path) == {7: "dividing", 8: "low-evidence"}


def test_a_clear_drops_every_cleared_ids_verdict_and_undo_puts_them_back(
        tmp_path):
    c, layer = _ctrl(tmp_path)
    _reshape_flags.write_flags(str(tmp_path), "p01",
                               {7: "dividing", 8: "low-evidence",
                                9: "split-override"})
    roi = (slice(0, 2), slice(0, 1), slice(0, 1))
    old = layer.data[roi].copy()
    layer.data[roi] = 0
    c._on_annotator_discard_garbage(
        [7, 8], roi_slices=roi, old_roi_data=old,
        extra={"add_clear_seq": 1})
    assert _flags(tmp_path) == {9: "split-override"}
    c._undo_last_curation()
    assert _flags(tmp_path) == {7: "dividing", 8: "low-evidence",
                                9: "split-override"}


def test_a_delete_with_no_verdicts_writes_no_sidecar(tmp_path):
    c, layer = _ctrl(tmp_path)
    roi = (slice(0, 1), slice(0, 1), slice(0, 1))
    old = layer.data[roi].copy()
    layer.data[roi] = 0
    c._on_annotator_label_deleted(7, roi_slices=roi, old_roi_data=old)
    assert not _reshape_flags.sidecar_path(str(tmp_path), "p01").exists()


def _ctrl_for_extract(tmp_path, labels=(7, 8, 9)):
    """``_ctrl`` plus what ``_on_qc_extract_finished`` touches."""
    from .test_qc_tier_preservation import _make_bare_controller

    c, layer = _ctrl(tmp_path, labels)
    bare = _make_bare_controller(prev_df=None, working_dir=tmp_path)
    for k in ("_auto_assign_qc_tiers", "_update_qc_summary",
              "_populate_threshold_feature_combo",
              "_update_flagged_labels_list", "_refresh_patch_list_colors",
              "ui"):
        setattr(c, k, getattr(bare, k))
    return c, layer


def test_the_re_extract_reads_the_verdicts_the_reshape_wrote(tmp_path):
    """The reshape writes the sidecar under the patch's stem; the
    re-extraction must read the same file (the patch name carries its
    extension)."""
    c, _layer = _ctrl_for_extract(tmp_path)
    c._on_annotator_reshape_committed(
        9, [11], flags={11: "low-evidence"},
        piece_stats=[{"id": 11, "voxels": 4, "centroid_zyx": (1, 1, 1)}])
    fresh = pd.DataFrame({"label": [7, 8, 11], "area": [3, 4, 5]})
    C.FusionReconstructionController._on_qc_extract_finished(
        c, fresh, "p01.tif")
    df = c._qc_features_df
    col = _reshape_flags.DEFAULT_FLAG_COLUMN
    assert df.loc[df["label"] == 11, col].iloc[0] == "low-evidence"


def test_a_reused_id_does_not_inherit_a_dropped_verdict_at_re_extract(
        tmp_path):
    """Through the real reader: the verdict of a deleted id is gone, so the
    re-extraction writes nothing onto a new object with that id, while a
    kept verdict is still applied."""
    c, layer = _ctrl_for_extract(tmp_path)
    _reshape_flags.write_flags(str(tmp_path), "p01",
                               {7: "dividing", 8: "low-evidence"})
    roi = (slice(0, 1), slice(0, 1), slice(0, 1))
    old = layer.data[roi].copy()
    layer.data[roi] = 0
    c._on_annotator_label_deleted(7, roi_slices=roi, old_roi_data=old)
    fresh = pd.DataFrame({"label": [7, 8], "area": [3, 4]})
    C.FusionReconstructionController._on_qc_extract_finished(
        c, fresh, "p01.tif")
    df = c._qc_features_df
    col = _reshape_flags.DEFAULT_FLAG_COLUMN
    assert df.loc[df["label"] == 8, col].iloc[0] == "low-evidence"
    assert df.loc[df["label"] == 7, col].iloc[0] == ""


def test_a_reshape_undo_drops_its_verdict_from_the_stem_file(tmp_path):
    c, _layer = _ctrl(tmp_path)
    c._on_annotator_reshape_committed(
        9, [11], flags={11: "low-evidence"},
        piece_stats=[{"id": 11, "voxels": 4, "centroid_zyx": (1, 1, 1)}])
    assert _flags(tmp_path) == {11: "low-evidence"}
    c._undo_last_curation()
    assert _flags(tmp_path) == {}


def test_the_verdict_is_dropped_only_after_the_commit(tmp_path):
    c, layer = _ctrl(tmp_path)
    _reshape_flags.write_flags(str(tmp_path), "p01", {7: "dividing"})

    def refuse(*a, **k):
        raise RuntimeError("commit refused")

    c._commit_curation_action = refuse
    roi = (slice(0, 1), slice(0, 1), slice(0, 1))
    with pytest.raises(RuntimeError):
        c._on_annotator_label_deleted(
            7, roi_slices=roi, old_roi_data=layer.data[roi].copy())
    with pytest.raises(RuntimeError):
        c._on_annotator_discard_garbage(
            [7], roi_slices=roi, old_roi_data=layer.data[roi].copy())
    assert _flags(tmp_path) == {7: "dividing"}


def test_a_discard_puts_back_the_verdicts_of_the_cells_it_brings_back(
        tmp_path):
    """Discard reloads the saved mask: a Delete not saved is undone by it,
    so its verdict goes back; a cell the saved mask lacks keeps none."""
    c, layer = _ctrl(tmp_path)
    _reshape_flags.write_flags(str(tmp_path), "p01",
                               {7: "dividing", 8: "low-evidence"})
    saved = layer.data.copy()
    saved[1, 0, 0] = 0                       # 8's delete was saved
    for lab, z in ((7, 0), (8, 1)):
        roi = (slice(z, z + 1), slice(0, 1), slice(0, 1))
        old = layer.data[roi].copy()
        layer.data[roi] = 0
        c._on_annotator_label_deleted(lab, roi_slices=roi, old_roi_data=old)
    assert _flags(tmp_path) == {}
    layer.data = saved                       # what the reload does
    c._put_back_flags_after_revert()
    assert _flags(tmp_path) == {7: "dividing"}


# --------------------------------------------------------------------------- #
# emptying the undo stacks tells the open annotator
# --------------------------------------------------------------------------- #
def test_emptying_the_undo_stacks_tells_the_annotator(tmp_path):
    c, _layer = _ctrl(tmp_path)
    told = []
    ann = types.SimpleNamespace(_add_after_undo_emptied=lambda: told.append(1))
    c.viewer.window._dock_widgets["EpiCycle 3D Curation Annotator"] = ann
    c._clear_all_undo_stacks()
    assert told == [1]


def test_emptying_the_undo_stacks_with_no_annotator_is_quiet(tmp_path):
    c, _layer = _ctrl(tmp_path)
    c._clear_all_undo_stacks()          # no dock: nothing raised
