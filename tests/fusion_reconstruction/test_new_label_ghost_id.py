"""New Label must not be swallowed by a ghost (deleted / merged) df row."""
from __future__ import annotations

from types import MethodType, SimpleNamespace

import numpy as np
import pandas as pd

from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator
from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController as C,
)
from packages.epicycle_annotator.curation_undo import CurationUndoStack


def _controller(df, volume):
    c = object.__new__(C)
    c._qc_view_patch_name = "p01.tif"
    c._qc_features_df = df
    c._curation_undo_stack = CurationUndoStack(max_depth=20)
    c._curation_dirty = False
    c._tiers_dirty = False
    c._curation_dirty_epoch = 0
    c._current_downsample_factor = lambda *a, **k: 1
    c._append_log = lambda m: None
    c.refreshed = []
    c._refresh_curation_ui = lambda: c.refreshed.append(
        sorted(
            set(
                c._qc_features_df.loc[
                    c._qc_features_df["label_status"] == "active", "label"
                ]
            )
        )
    )
    c._find_active_labels_layer = lambda: SimpleNamespace(data=volume)
    c._drop_tier_undo = lambda: None
    c._scroll_nav_to_label = lambda lid: None
    return c


def _df(statuses):
    labels = list(statuses)
    return pd.DataFrame(
        {
            "label": labels,
            "qc_tier": ["review"] * len(labels),
            "feature_state": ["fresh"] * len(labels),
            "label_status": [statuses[k] for k in labels],
            "area": [9] * len(labels),
            "centroid-0": [1.0] * len(labels),
            "centroid-1": [3.0] * len(labels),
            "centroid-2": [3.0] * len(labels),
        }
    )


def test_ghost_row_does_not_swallow_new_label():
    vol = np.zeros((4, 8, 8), np.int32)
    vol[1, 1:3, 1:3] = 7
    vol[2, 4:6, 4:6] = 8  # the fresh paint reusing the ghost id
    c = _controller(_df({7: "active", 8: "deleted"}), vol)
    c._on_annotator_new_label_painted(8)
    df = c._qc_features_df
    assert int((df["label"] == 8).sum()) == 2
    assert df.loc[df["label"] == 8, "label_status"].tolist() == [
        "deleted", "active",
    ]
    kinds = [(e.action_type, e.label_id) for e in c._curation_undo_stack._stack]
    assert ("new_label_painted", 8) in kinds
    assert c.refreshed and 8 in c.refreshed[-1]


def test_active_row_still_idempotent():
    vol = np.zeros((4, 8, 8), np.int32)
    vol[1, 1:3, 1:3] = 7
    c = _controller(_df({7: "active"}), vol)
    c._on_annotator_new_label_painted(7)
    assert int((c._qc_features_df["label"] == 7).sum()) == 1


def test_missing_status_column_counts_as_active():
    vol = np.zeros((4, 8, 8), np.int32)
    vol[1, 1:3, 1:3] = 7
    df = _df({7: "active"}).drop(columns=["label_status"])
    c = _controller(df, vol)
    c._on_annotator_new_label_painted(7)
    assert int((c._qc_features_df["label"] == 7).sum()) == 1


def _annotator(volume, floor):
    from napari.components import ViewerModel

    vm = ViewerModel()
    layer = vm.add_labels(volume, name="curation_3d_p01")
    a = SimpleNamespace(viewer=vm, picked=None)
    a._get_label_id_floor_cb = (lambda: floor) if floor is not None else None
    a._add_label_id_floor = MethodType(
        Curation3DAnnotator._add_label_id_floor, a
    )
    a._set_selected_label_all = lambda n: setattr(a, "picked", n)
    a._on_new_label_from_navigator = MethodType(
        Curation3DAnnotator._on_new_label_from_navigator, a
    )
    return a, layer


def test_new_label_skips_ghost_ids_via_floor():
    vol = np.zeros((3, 6, 6), np.int32)
    vol[1, 1:3, 1:3] = 7
    a, _ = _annotator(vol, floor=12)
    a._on_new_label_from_navigator()
    assert a.picked == 13  # same convention as Add: max(volume, floor) + 1


def test_new_label_without_floor_uses_volume():
    vol = np.zeros((3, 6, 6), np.int32)
    vol[1, 1:3, 1:3] = 7
    a, _ = _annotator(vol, floor=None)
    a._on_new_label_from_navigator()
    assert a.picked == 8
