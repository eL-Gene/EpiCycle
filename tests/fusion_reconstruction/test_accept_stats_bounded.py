"""Step-3 accept perf: the df stat refresh must stay INSIDE the accept ROI.

Both handlers used to scan the WHOLE labels volume (`data == label_id` +
`np.argwhere`) for every accepted label / de-novo piece.  These tests use a
volume proxy that REFUSES a full-volume comparison, so a regression back to the
full scan fails loudly, and pin that the numbers are unchanged.
"""

import numpy as np
import pandas as pd
import pytest

from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController,
)


class _NoFullScanVolume:
    """Acts like the labels array for cropped access; explodes on a full scan."""

    def __init__(self, arr):
        self._a = arr
        self.full_scans = 0

    @property
    def shape(self):
        return self._a.shape

    @property
    def ndim(self):
        return self._a.ndim

    @property
    def dtype(self):
        return self._a.dtype

    def __getitem__(self, key):
        return self._a[key]

    def __eq__(self, other):  # pragma: no cover - the assertion IS the test
        self.full_scans += 1
        raise AssertionError(
            "full-volume `data == label_id` scan on the accept path"
        )

    __hash__ = None


class _Layer:
    def __init__(self, data):
        self.data = data


def _bare_controller(df, layer):
    ctrl = FusionReconstructionController.__new__(FusionReconstructionController)
    ctrl._qc_features_df = df
    ctrl._find_active_labels_layer = lambda: layer
    ctrl._current_downsample_factor = lambda: 1
    ctrl._append_log = lambda *a, **k: None
    ctrl._scan_qc_df_for_duplicates = lambda *a, **k: None
    # Run the df updater inline; the undo/dirty plumbing is out of scope here.
    # The keyword list is explicit (no **kw) so it pins the real signature.
    ctrl._commit_calls = []

    def _commit(action, label_id, desc, *, roi_slices=None,
                old_roi_data=None, df_updater=None, extra=None):
        ctrl._commit_calls.append({"action": action, "extra": extra})
        if df_updater is not None:
            df_updater()
    ctrl._commit_curation_action = _commit
    return ctrl


def _df(labels):
    return pd.DataFrame(
        {
            "label": list(labels),
            "area": [0] * len(labels),
            "centroid-0": [0.0] * len(labels),
            "centroid-1": [0.0] * len(labels),
            "centroid-2": [0.0] * len(labels),
            "qc_tier": [""] * len(labels),
            "feature_state": ["fresh"] * len(labels),
            "label_status": ["active"] * len(labels),
        }
    )


def _reference(vol, label_id):
    mask = vol == label_id
    coords = np.argwhere(mask)
    return int(mask.sum()), coords.mean(axis=0)


# --------------------------------------------------------------------------
# _update_label_basic_stats  (interpolation / smooth accepts)
# --------------------------------------------------------------------------

def test_update_label_basic_stats_uses_roi_and_matches_full_volume():
    vol = np.zeros((40, 44, 46), dtype=np.uint16)
    vol[10:16, 12:20, 14:23] = 8
    roi = (slice(5, 25), slice(6, 30), slice(7, 34))  # bbox + margin

    want_area, want_centroid = _reference(vol, 8)

    df = _df([8])
    ctrl = _bare_controller(df, _Layer(_NoFullScanVolume(vol)))
    ctrl._update_label_basic_stats(8, roi_slices=roi)

    row = ctrl._qc_features_df.loc[ctrl._qc_features_df["label"] == 8].iloc[0]
    assert int(row["area"]) == want_area
    assert float(row["centroid-0"]) == want_centroid[0]
    assert float(row["centroid-1"]) == want_centroid[1]
    assert float(row["centroid-2"]) == want_centroid[2]


def test_update_label_basic_stats_falls_back_to_full_volume_without_roi():
    """Merge (and any caller with no ROI) must keep the whole-volume scan."""
    vol = np.zeros((20, 20, 20), dtype=np.uint16)
    vol[2:5, 3:6, 4:8] = 9
    vol[15:18, 15:18, 15:18] = 9  # a second lobe, far away
    want_area, want_centroid = _reference(vol, 9)

    df = _df([9])
    ctrl = _bare_controller(df, _Layer(vol))
    ctrl._update_label_basic_stats(9)

    row = ctrl._qc_features_df.loc[ctrl._qc_features_df["label"] == 9].iloc[0]
    assert int(row["area"]) == want_area
    assert float(row["centroid-0"]) == want_centroid[0]


def test_update_label_basic_stats_absent_label_is_a_noop():
    vol = np.zeros((16, 16, 16), dtype=np.uint16)
    roi = (slice(0, 8), slice(0, 8), slice(0, 8))
    df = _df([3])
    ctrl = _bare_controller(df, _Layer(_NoFullScanVolume(vol)))
    ctrl._update_label_basic_stats(3, roi_slices=roi)
    row = ctrl._qc_features_df.iloc[0]
    assert int(row["area"]) == 0
    assert float(row["centroid-0"]) == 0.0


def test_interpolation_accept_passes_roi_through(monkeypatch):
    vol = np.zeros((30, 30, 30), dtype=np.uint16)
    vol[8:12, 9:14, 10:16] = 21
    roi = (slice(4, 20), slice(4, 22), slice(4, 24))
    want_area, want_centroid = _reference(vol, 21)

    df = _df([21])
    ctrl = _bare_controller(df, _Layer(_NoFullScanVolume(vol)))
    ctrl._on_annotator_interpolation_accepted(
        21, roi_slices=roi, old_roi_data=None
    )

    active = ctrl._qc_features_df[
        (ctrl._qc_features_df["label"] == 21)
        & (ctrl._qc_features_df["label_status"] == "active")
    ].iloc[0]
    assert int(active["area"]) == want_area
    assert float(active["centroid-2"]) == want_centroid[2]
    assert active["qc_tier"] == "curated"


def test_smooth_accept_passes_roi_through():
    vol = np.zeros((30, 30, 30), dtype=np.uint16)
    vol[3:9, 4:11, 5:13] = 33
    roi = (slice(0, 20), slice(0, 22), slice(0, 24))
    want_area, want_centroid = _reference(vol, 33)

    df = _df([33])
    ctrl = _bare_controller(df, _Layer(_NoFullScanVolume(vol)))
    ctrl._on_annotator_label_smoothed(33, roi_slices=roi, old_roi_data=None)

    # Recipes provenance rides the commit's extra= (steps + recipes keys).
    extra = ctrl._commit_calls[-1]["extra"]
    assert isinstance(extra, dict)
    assert "steps" in extra and "recipes" in extra

    active = ctrl._qc_features_df[
        (ctrl._qc_features_df["label"] == 33)
        & (ctrl._qc_features_df["label_status"] == "active")
    ].iloc[0]
    assert int(active["area"]) == want_area
    assert float(active["centroid-1"]) == want_centroid[1]


def test_downsample_factor_still_scales_the_centroid():
    vol = np.zeros((20, 20, 20), dtype=np.uint16)
    vol[4:8, 5:9, 6:10] = 12
    roi = (slice(0, 16), slice(0, 16), slice(0, 16))
    _, want_centroid = _reference(vol, 12)

    df = _df([12])
    ctrl = _bare_controller(df, _Layer(_NoFullScanVolume(vol)))
    ctrl._current_downsample_factor = lambda: 2
    ctrl._update_label_basic_stats(12, roi_slices=roi)

    row = ctrl._qc_features_df.iloc[0]
    assert float(row["centroid-0"]) == want_centroid[0] * 2


# --------------------------------------------------------------------------
# _on_annotator_add_committed  (de-novo Add)
# --------------------------------------------------------------------------

def test_add_committed_uses_supplied_stats_and_matches_full_volume():
    vol = np.zeros((40, 40, 40), dtype=np.uint16)
    roi = (slice(6, 26), slice(7, 28), slice(8, 30))
    result = np.zeros(vol[roi].shape, dtype=np.uint16)
    result[0:4, 0:5, 0:6] = 500
    result[8:12, 9:13, 10:15] = 501
    write_mask = (result > 0) & (vol[roi] == 0)
    vol[roi][write_mask] = result[write_mask]

    from packages.epicycle_annotator.roi_stats import piece_stats_from_write

    stats = piece_stats_from_write(
        result, write_mask, tuple(s.start for s in roi)
    )

    df = _df([1])
    ctrl = _bare_controller(df, _Layer(_NoFullScanVolume(vol)))
    ctrl._on_annotator_add_committed(
        [500, 501], roi_slices=roi, old_roi_data=None, piece_stats=stats
    )

    out = ctrl._qc_features_df
    for pid in (500, 501):
        want_area, want_centroid = _reference(vol, pid)
        row = out.loc[out["label"] == pid].iloc[0]
        assert int(row["area"]) == want_area
        for i in range(3):
            assert float(row[f"centroid-{i}"]) == want_centroid[i]
        assert row["feature_state"] == "stale"
        assert row["qc_tier"] == ""
        assert row["label_status"] == "active"


def test_add_committed_without_stats_falls_back_to_full_volume():
    vol = np.zeros((20, 20, 20), dtype=np.uint16)
    vol[3:6, 4:8, 5:10] = 700
    want_area, want_centroid = _reference(vol, 700)

    df = _df([1])
    ctrl = _bare_controller(df, _Layer(vol))
    ctrl._on_annotator_add_committed([700], roi_slices=None, old_roi_data=None)

    row = ctrl._qc_features_df.loc[
        ctrl._qc_features_df["label"] == 700
    ].iloc[0]
    assert int(row["area"]) == want_area
    assert float(row["centroid-0"]) == want_centroid[0]


def test_add_committed_skips_ids_already_registered():
    vol = np.zeros((20, 20, 20), dtype=np.uint16)
    vol[2:4, 2:4, 2:4] = 42
    df = _df([42])
    ctrl = _bare_controller(df, _Layer(_NoFullScanVolume(vol)))
    ctrl._on_annotator_add_committed(
        [42], roi_slices=(slice(0, 8),) * 3, old_roi_data=None,
        piece_stats={42: (8, np.array([2.5, 2.5, 2.5]))},
    )
    assert (ctrl._qc_features_df["label"] == 42).sum() == 1
    assert int(ctrl._qc_features_df.iloc[0]["area"]) == 0  # untouched


def test_add_committed_skips_zero_size_pieces():
    vol = np.zeros((20, 20, 20), dtype=np.uint16)
    df = _df([1])
    ctrl = _bare_controller(df, _Layer(_NoFullScanVolume(vol)))
    ctrl._on_annotator_add_committed(
        [800], roi_slices=(slice(0, 8),) * 3, old_roi_data=None,
        piece_stats={},  # the piece was fully blocked — no written voxels
    )
    assert (ctrl._qc_features_df["label"] == 800).sum() == 0


def test_add_committed_downsample_scales_supplied_centroid():
    vol = np.zeros((20, 20, 20), dtype=np.uint16)
    df = _df([1])
    ctrl = _bare_controller(df, _Layer(_NoFullScanVolume(vol)))
    ctrl._current_downsample_factor = lambda: 4
    ctrl._on_annotator_add_committed(
        [900], roi_slices=(slice(0, 8),) * 3, old_roi_data=None,
        piece_stats={900: (10, np.array([1.0, 2.0, 3.0]))},
    )
    row = ctrl._qc_features_df.loc[
        ctrl._qc_features_df["label"] == 900
    ].iloc[0]
    assert int(row["area"]) == 10
    assert float(row["centroid-0"]) == 4.0
    assert float(row["centroid-2"]) == 12.0


class _FakeAdvanced:
    def __init__(self, checked):
        self._checked = checked

    def get_checked_add_pieces(self):
        return list(self._checked)


class _FakeAnnotator:
    """Just enough of Curation3DAnnotator to run the real `_accept_add`."""

    def __init__(self, volume, roi, result, checked=None):
        self._volume = volume
        self._add_roi = roi
        self._add_preview_result = result
        self._add_running = True
        self._add_session = []
        self._add_piece_ids = []
        self._add_review = False
        self._add_box_timer = None
        self._advanced = _FakeAdvanced(
            checked
            if checked is not None
            else [int(v) for v in np.unique(result) if v != 0]
        )
        self.callback_kwargs = None
        self.callback_ids = None
        self.ui = []

    # --- the seams `_accept_add` reaches through -------------------------
    def _add_shared_volume(self):
        return self._volume

    def _add_refused_while_batch(self, door):
        # Batch recipe brick: Add refuses while Batch Refine runs or is
        # staged.  No batch here, so it never refuses.
        return False

    def _get_advanced_features_widget(self):
        return self._advanced

    def _push_add_ui(self, **kw):
        self.ui.append(kw)

    def _remove_add_candidate(self):
        pass

    def _remove_add_box_shapes(self):
        pass

    def _sync_all_views_after_operation(self, roi=None, **kw):
        pass

    def _fire_add_mode_lock(self, active):
        pass

    def _restore_ready_to_draw_state(self):
        pass

    # The teardown seams the Add rebuild added to `_accept_add` (B2: the
    # walls picture, the two workers, the side callbacks, the ball layer,
    # Place markers, Numbers...; follow-up brick A3: the outline tick; the
    # S1 refresh).  No-ops here: this file tests the stats, not the exits
    # (`tests/annotator/test_add_exits.py` does).
    def _drop_add_walls(self):
        pass

    def _add_abandon_workers(self):
        pass

    def _remove_add_side_callbacks(self):
        pass

    def _remove_add_marker_layer(self):
        pass

    def _add_place_markers_off(self):
        pass

    def _add_bump_markers_epoch(self):
        # Accept clears the click markers and bumps the
        # markers epoch; the real one is exercised in test_add_exits.py
        pass

    def _add_forget_volume_max(self):
        # B-L3 of the A5 audit: Accept drops the session's cached volume
        # max; the real one is exercised in test_add_audit_fixes_b.py
        pass

    def _close_add_numbers(self):
        pass

    def _add_put_outline_down(self):
        pass

    def _add_refresh_ui(self, **_kw):
        pass

    def _on_add_committed_cb(self, ids, **kwargs):
        self.callback_ids = list(ids)
        self.callback_kwargs = kwargs


def test_accept_add_hands_bounded_stats_to_the_controller():
    """Behavioural: the real `_accept_add` computes the per-piece stats itself
    and the numbers it hands over match a post-commit full-volume scan."""
    from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator

    vol = np.zeros((30, 32, 34), dtype=np.uint16)
    vol[12:15, 13:16, 14:17] = 77          # an existing cell inside the box
    roi = (slice(6, 26), slice(7, 28), slice(8, 30))
    result = np.zeros(vol[roi].shape, dtype=np.uint16)
    result[0:4, 0:5, 0:6] = 500            # flush with the ROI corner
    result[5:11, 5:11, 5:11] = 501         # overlaps the existing 77 block
    result[14:17, 15:19, 16:21] = 502

    ann = _FakeAnnotator(vol, roi, result.copy())
    Curation3DAnnotator._accept_add(ann)

    assert ann.callback_ids == [500, 501, 502]
    stats = ann.callback_kwargs["piece_stats"]
    assert ann.callback_kwargs["roi_slices"] == roi
    for pid in ann.callback_ids:
        want_area, want_centroid = _reference(vol, pid)
        assert stats[pid][0] == want_area
        assert np.array_equal(stats[pid][1], want_centroid)
    # The REVIEW list sizes come from the very same pass.
    assert ann._add_session[-1]["sizes"] == {
        pid: stats[pid][0] for pid in ann.callback_ids
    }
    assert ann._add_running is False


def test_accept_add_stats_feed_the_controller_end_to_end():
    """The stats `_accept_add` produces, fed to the real controller handler,
    give exactly the rows a full-volume scan would have produced."""
    from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator

    vol = np.zeros((24, 24, 24), dtype=np.uint16)
    roi = (slice(4, 20), slice(4, 20), slice(4, 20))
    result = np.zeros(vol[roi].shape, dtype=np.uint16)
    result[1:5, 2:6, 3:7] = 610
    result[8:12, 9:13, 10:14] = 611

    ann = _FakeAnnotator(vol, roi, result.copy())
    Curation3DAnnotator._accept_add(ann)

    ctrl = _bare_controller(_df([1]), _Layer(_NoFullScanVolume(vol)))
    ctrl._on_annotator_add_committed(
        ann.callback_ids, **ann.callback_kwargs
    )
    out = ctrl._qc_features_df
    for pid in ann.callback_ids:
        want_area, want_centroid = _reference(vol, pid)
        row = out.loc[out["label"] == pid].iloc[0]
        assert int(row["area"]) == want_area
        for i in range(3):
            assert float(row[f"centroid-{i}"]) == want_centroid[i]


# --------------------------------------------------------------------------
# containment tripwire on _update_label_basic_stats
# --------------------------------------------------------------------------

def test_stats_fall_back_to_full_scan_when_the_label_escapes_the_roi():
    """A label flush against an UNCLIPPED ROI face may continue outside it, so
    the bounded path must be abandoned and the full-volume numbers used."""
    vol = np.zeros((30, 30, 30), dtype=np.uint16)
    vol[8:14, 8:14, 8:14] = 61      # crosses the ROI's high faces
    roi = (slice(4, 12), slice(4, 12), slice(4, 12))
    want_area, want_centroid = _reference(vol, 61)

    df = _df([61])
    ctrl = _bare_controller(df, _Layer(vol))   # plain array: full scan allowed
    ctrl._update_label_basic_stats(61, roi_slices=roi)

    row = ctrl._qc_features_df.iloc[0]
    assert int(row["area"]) == want_area       # 216, not the 64 inside the ROI
    for i in range(3):
        assert float(row[f"centroid-{i}"]) == want_centroid[i]


def test_stats_keep_the_bounded_path_on_a_clipped_volume_edge_face():
    """Touching an ROI face that IS the volume boundary is not an escape."""
    vol = np.zeros((20, 20, 20), dtype=np.uint16)
    vol[0:4, 0:5, 0:6] = 62         # flush with the volume corner
    roi = (slice(0, 12), slice(0, 12), slice(0, 12))
    want_area, want_centroid = _reference(vol, 62)

    df = _df([62])
    # _NoFullScanVolume raises if the bounded path is abandoned.
    ctrl = _bare_controller(df, _Layer(_NoFullScanVolume(vol)))
    ctrl._update_label_basic_stats(62, roi_slices=roi)

    row = ctrl._qc_features_df.iloc[0]
    assert int(row["area"]) == want_area
    assert float(row["centroid-0"]) == want_centroid[0]
