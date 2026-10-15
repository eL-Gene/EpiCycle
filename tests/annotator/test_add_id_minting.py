"""Add's new ids skip the ids of deleted objects.

* the controller's ``_add_label_id_floor`` = the highest df id over ALL
  rows, ghosts (``deleted`` / ``superseded``) included; 0 with no df;
* the annotator's ``_add_next_label`` = ``max(max(volume), floor) + 1``;
* the dtype guard: ``_add_id_max`` is the label dtype's largest id;
* the callback is threaded launcher -> ``set_curation_callbacks`` and nulled
  on close like ``get_plot_selection``;
* end to end: delete the three highest ids (ghost rows stay), Add -> every
  new id is above the highest ghost id and gets an ``active`` row.

Qt-free: the controller and the annotator are built with ``object.__new__``.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from napari_epicycle.annotator import launcher
from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator
from packages.epicycle_annotator import add_preview as AP

from tests.fusion_reconstruction.test_step3_commit_before_qc_guard import (
    _make_controller,
)


class _SM:
    def __init__(self, vol):
        self.shared_labels = {"L": vol}


def _annotator(vol, floor=None):
    a = Curation3DAnnotator.__new__(Curation3DAnnotator)
    a.sync_manager = _SM(vol)
    a._active_labels_name = "L"
    a._get_label_id_floor_cb = (None if floor is None else (lambda: floor))
    return a


def _ghost_df():
    return pd.DataFrame({
        "label": [3, 7, 40, 41, 42],
        "qc_tier": ["review"] * 5,
        "feature_state": ["fresh"] * 5,
        "label_status": ["active", "active", "deleted", "superseded",
                         "deleted"],
        "area": [1] * 5,
        "centroid-0": [0.0] * 5, "centroid-1": [0.0] * 5,
        "centroid-2": [0.0] * 5,
    })


# --------------------------------------------------------------------------- #
# the controller's floor
# --------------------------------------------------------------------------- #
def test_the_floor_counts_ghost_rows():
    c = _make_controller(df=_ghost_df())
    assert c._add_label_id_floor() == 42


def test_the_floor_is_zero_without_a_df():
    c = _make_controller(df=None)
    assert c._add_label_id_floor() == 0
    c = _make_controller(df=pd.DataFrame({"label": []}))
    assert c._add_label_id_floor() == 0


# --------------------------------------------------------------------------- #
# the annotator's next label and the dtype guard
# --------------------------------------------------------------------------- #
def test_next_label_is_above_the_floor_and_the_volume():
    vol = np.zeros((4, 4, 4), np.uint16)
    vol[0, 0, 0] = 7
    assert _annotator(vol)._add_next_label() == 8
    assert _annotator(vol, floor=42)._add_next_label() == 43
    assert _annotator(vol, floor=3)._add_next_label() == 8


def test_a_raising_floor_callback_falls_back_to_the_volume():
    vol = np.zeros((4, 4, 4), np.uint16)
    vol[0, 0, 0] = 7
    a = _annotator(vol)

    def boom():
        raise RuntimeError("no df")

    a._get_label_id_floor_cb = boom
    assert a._add_next_label() == 8


def test_the_id_max_is_the_dtype_max():
    assert _annotator(np.zeros((2, 2, 2), np.uint16))._add_id_max() == 65535
    assert _annotator(np.zeros((2, 2, 2), np.int32))._add_id_max() == 2 ** 31 - 1


# --------------------------------------------------------------------------- #
# the wiring
# --------------------------------------------------------------------------- #
def test_the_callback_is_threaded_and_stored():
    assert "get_label_id_floor" in inspect.signature(
        Curation3DAnnotator.set_curation_callbacks).parameters
    assert "get_label_id_floor" in inspect.signature(
        launcher.launch_3d_curation_annotator).parameters
    a = Curation3DAnnotator.__new__(Curation3DAnnotator)
    fn = lambda: 5  # noqa: E731
    a.set_curation_callbacks(get_label_id_floor=fn)
    assert a._get_label_id_floor_cb is fn


def test_the_controller_passes_it_at_both_call_sites():
    from napari_epicycle.fusion_reconstruction import controller as C

    src = inspect.getsource(C)
    assert src.count("get_label_id_floor=self._add_label_id_floor") == 2


# --------------------------------------------------------------------------- #
# end to end: delete the three highest, Add, every new id above the ghosts
# --------------------------------------------------------------------------- #
def test_new_ids_never_hide_behind_a_ghost_row():
    from .test_add_nuclei_golden import box_of, nuclei_fixture

    img, labels = nuclei_fixture(0)
    ids = sorted(int(v) for v in np.unique(labels) if v)      # 5, 9, 17
    df = pd.DataFrame({
        "label": ids, "qc_tier": ["review"] * 3,
        "feature_state": ["fresh"] * 3, "label_status": ["active"] * 3,
        "area": [1] * 3, "centroid-0": [0.0] * 3, "centroid-1": [0.0] * 3,
        "centroid-2": [0.0] * 3,
    })
    # delete the three highest: the voxels go, the ghost rows stay
    for i in ids[-3:]:
        labels[labels == i] = 0
    df.loc[df["label"].isin(ids[-3:]), "label_status"] = "deleted"
    c = _make_controller(df=df)
    a = _annotator(labels, floor=c._add_label_id_floor())
    next_label = a._add_next_label()
    assert next_label == max(ids) + 1
    roi = box_of(labels.shape)
    out = AP.run_add_preview(
        preset="nuclei", roi=roi, existing_roi=labels[roi].copy(),
        next_label=next_label, spacing=(1.0, 1.0, 1.0),
        params={"threshold": "otsu"}, image_roi=img[roi].astype(np.float64),
        id_max=a._add_id_max())
    assert out.ok
    assert min(out.ids) > max(ids)
    c._find_active_labels_layer = lambda: type("L", (), {"data": labels})()
    stats = {i: (out.sizes[i], (0.0, 0.0, 0.0)) for i in out.ids}
    c._on_annotator_add_committed(out.ids, roi_slices=roi,
                                  old_roi_data=labels[roi].copy(),
                                  piece_stats=stats)
    got = c._qc_features_df
    for i in out.ids:
        rows = got[got["label"] == i]
        assert len(rows) == 1
        assert rows.iloc[0]["label_status"] == "active"
