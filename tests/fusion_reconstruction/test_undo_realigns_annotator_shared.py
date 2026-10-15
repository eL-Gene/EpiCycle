"""An undo must leave the annotator's shared volume equal to the main layer.

Bug (pre-existing, traced 2026-09-27): on a patch whose labels TIFF is
uint32, ``NativeMultiviewAnnotator._normalize_labels_volume`` COPIES the
volume at launch (``astype(np.int32, copy=False)`` cannot alias a uint32
array), so ``sync_manager.shared_labels[name]`` and the main napari
``layer.data`` are two arrays.  ``_undo_last_curation`` restores
``layer.data[roi]`` and calls ``_refresh_annotator_subviewers`` -- main and
the side views revert, the shared copy does NOT.  The next
``_sync_all_views_after_operation`` (shared -> main over a ROI) then
RESURRECTS the undone cells with no df row and no undo entry, and Save
writes them.

Fix: ``_refresh_annotator_subviewers`` re-points shared at main via the
annotator's own ``_ensure_shared_data_aligned`` before refreshing.

Qt-free: controller via ``__new__`` + stubs (as test_batch_refine_callback).
"""
from __future__ import annotations

import types

import numpy as np
import pytest


class _FakeLayer:
    def __init__(self, name, data):
        self.name = name
        self.data = data
        self.refreshed = 0

    def refresh(self):
        self.refreshed += 1


class _FakeSync:
    def __init__(self):
        self.shared_labels = {}

    def register_shared_data(self, name, data):
        self.shared_labels[name] = data


NAME = "curation_3d_patch"


def _make_annotator(layer, shared):
    from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator
    from napari_epicycle.annotator.native_multiview_widget import (
        NativeMultiviewAnnotator,
    )

    ann = types.SimpleNamespace()
    ann._active_labels_name = NAME
    ann.sync_manager = _FakeSync()
    ann.sync_manager.shared_labels[NAME] = shared
    ann.viewer = types.SimpleNamespace(layers=[layer])
    ann._layer_by_name = types.MethodType(
        NativeMultiviewAnnotator._layer_by_name, ann
    )
    ann._ensure_shared_data_aligned = types.MethodType(
        Curation3DAnnotator._ensure_shared_data_aligned, ann
    )
    ann.refreshed_subviewers = 0

    def _refresh():
        ann.refreshed_subviewers += 1

    ann._refresh_label_data_in_subviewers = _refresh
    return ann


def _make_controller(layer, ann):
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )
    from packages.epicycle_annotator.curation_undo import CurationUndoStack

    ctrl = FusionReconstructionController.__new__(FusionReconstructionController)
    ctrl._qc_features_df = None
    ctrl._curation_undo_stack = CurationUndoStack(max_depth=20)
    ctrl._curation_dirty = False
    ctrl._curation_dirty_epoch = 0
    ctrl._append_log = lambda msg: None
    ctrl._refresh_curation_ui = lambda: None
    ctrl._find_active_labels_layer = lambda: layer
    ctrl.viewer = types.SimpleNamespace(
        window=types.SimpleNamespace(
            _dock_widgets={"EpiCycle 3D Curation Annotator": ann}
        )
    )
    return ctrl


def _launch(dtype):
    """Return (pre-accept volume, main layer, shared array) as the
    annotator launch leaves them for a labels TIFF of *dtype*."""
    base = np.zeros((6, 8, 8), dtype=dtype)
    base[1:3, 1:3, 1:3] = 4  # an existing cell
    layer = _FakeLayer(NAME, base)
    # What _normalize_labels_volume does: a copy for uint32, same object
    # for int32.
    shared = base.astype(np.int32, copy=False)
    return base.copy(), layer, shared


def _accept_add(ctrl, layer, ann, roi):
    """Simulate ``_accept_add`` exactly: snapshot shared, background-only
    write into shared, shared -> main over the ROI, commit callback."""
    shared = ann.sync_manager.shared_labels[NAME]
    target = shared[roi]
    snap = shared[roi].copy()
    write = target == 0
    target[write] = 9
    layer.data[roi] = shared[roi]  # _sync_all_views_after_operation(roi)
    n = int(write.sum())
    ctrl._on_annotator_add_committed(
        [9], roi_slices=roi, old_roi_data=snap,
        piece_stats={9: (n, (3.0, 4.0, 4.0))},
    )


@pytest.mark.parametrize("dtype", [np.uint32, np.int32])
def test_add_undo_reverts_main_and_shared(dtype):
    before, layer, shared = _launch(dtype)
    assert (shared is layer.data) == (dtype == np.int32)
    ann = _make_annotator(layer, shared)
    ctrl = _make_controller(layer, ann)
    roi = (slice(0, 5), slice(0, 6), slice(0, 6))

    _accept_add(ctrl, layer, ann, roi)
    assert (layer.data == 9).any()

    assert ctrl._on_annotator_add_undo_session(1) == 1

    assert np.array_equal(layer.data, before)
    assert np.array_equal(ann.sync_manager.shared_labels[NAME], before)
    assert ann.refreshed_subviewers >= 1


@pytest.mark.parametrize("dtype", [np.uint32, np.int32])
def test_undone_cells_do_not_resurrect_on_next_sync(dtype):
    before, layer, shared = _launch(dtype)
    ann = _make_annotator(layer, shared)
    ctrl = _make_controller(layer, ann)
    roi = (slice(0, 5), slice(0, 6), slice(0, 6))
    _accept_add(ctrl, layer, ann, roi)
    assert ctrl._on_annotator_add_undo_session(1) == 1

    # The annotator's next op copies shared -> main over an overlapping ROI
    # (_sync_all_views_after_operation), with no write of its own.
    sh = ann.sync_manager.shared_labels[NAME]
    other = (slice(2, 6), slice(3, 8), slice(3, 8))
    if layer.data is not sh:
        layer.data[other] = sh[other]
    assert not (layer.data == 9).any()
    assert np.array_equal(layer.data, before)


def test_normalize_labels_volume_copies_uint32():
    """The root: a uint32 labels volume is COPIED at launch, so shared and
    main start as two arrays.  Passes before and after the fix -- it
    documents why the undo must realign."""
    from napari_epicycle.annotator.native_multiview_widget import (
        NativeMultiviewAnnotator,
    )

    w = NativeMultiviewAnnotator.__new__(NativeMultiviewAnnotator)
    u = np.zeros((4, 5, 5), dtype=np.uint32)
    out, _ = w._normalize_labels_volume(u, u.shape, "labels")
    assert out.dtype == np.int32
    assert not np.shares_memory(out, u)
    i = np.zeros((4, 5, 5), dtype=np.int32)
    out_i, _ = w._normalize_labels_volume(i, i.shape, "labels")
    assert out_i is i


# --------------------------------------------------------------------------- #
# Final fix pass U1 / U2 -- Commit Triage and Discard reassign layer.data
# --------------------------------------------------------------------------- #
class _ReassignLayer(_FakeLayer):
    """``layer.data = x`` fires ``on_reassign`` -- the proxy's
    ``_data_reassigned_cb`` on the slow launch path, which points
    ``shared_labels`` back at the ABANDONED int32 copy."""

    def __init__(self, name, data):
        self.on_reassign = None
        super().__init__(name, data)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value
        if self.on_reassign is not None:
            self.on_reassign()


def _stale_proxy(ann, stale):
    def _cb():
        ann.sync_manager.shared_labels[NAME] = stale
    return _cb


def _accept_undo_then(dtype):
    before, _layer, shared = _launch(dtype)
    layer = _ReassignLayer(NAME, _layer.data)
    ann = _make_annotator(layer, shared)
    ctrl = _make_controller(layer, ann)
    roi = (slice(0, 5), slice(0, 6), slice(0, 6))
    _accept_add(ctrl, layer, ann, roi)
    stale = ann.sync_manager.shared_labels[NAME]    # still holds the 9s
    assert ctrl._on_annotator_add_undo_session(1) == 1
    if dtype == np.uint32:
        assert (stale == 9).any(), "the abandoned copy keeps the undone cell"
    layer.on_reassign = _stale_proxy(ann, stale)
    return before, layer, ann, ctrl


def _next_sync(ann, layer):
    sh = ann.sync_manager.shared_labels[NAME]
    other = (slice(0, 6), slice(0, 8), slice(0, 8))
    if layer.data is not sh:
        layer.data[other] = sh[other]


@pytest.mark.parametrize("dtype", [np.uint32, np.int32])
def test_commit_triage_after_undo_does_not_resurrect(dtype):
    before, layer, ann, ctrl = _accept_undo_then(dtype)
    ctrl._zero_labels_in_viewer({4})           # Commit Triage drops cell 4
    assert ann.sync_manager.shared_labels[NAME] is layer.data
    _next_sync(ann, layer)
    assert not (layer.data == 9).any()
    assert not (layer.data == 4).any()


@pytest.mark.parametrize("dtype", [np.uint32, np.int32])
def test_discard_after_undo_does_not_resurrect(dtype, tmp_path):
    import tifffile

    before, layer, ann, ctrl = _accept_undo_then(dtype)
    path = tmp_path / "labels.tif"
    tifffile.imwrite(str(path), before)
    ctrl._get_active_labels_path = lambda: str(path)
    ctrl._reload_current_patch_features = lambda: None
    assert ctrl._revert_labels_from_disk() is True
    assert ann.sync_manager.shared_labels[NAME] is layer.data
    _next_sync(ann, layer)
    assert np.array_equal(layer.data, before)


def test_refresh_realigns_every_active_labels_layer():
    """U2: every name in ``_active_labels_names``, not only the active one."""
    before, layer, shared = _launch(np.uint32)
    ann = _make_annotator(layer, shared)
    other = _FakeLayer("second", np.zeros((2, 2, 2), np.uint32))
    ann.viewer.layers.append(other)
    ann.sync_manager.shared_labels["second"] = other.data.astype(np.int32)
    ann._active_labels_names = [NAME, "second"]
    ctrl = _make_controller(layer, ann)
    ctrl._refresh_annotator_subviewers()
    assert ann.sync_manager.shared_labels[NAME] is layer.data
    assert ann.sync_manager.shared_labels["second"] is other.data
