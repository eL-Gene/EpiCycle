"""Tests for the cached enclosed-fill next-label counter (2026-05-28,
fix/step2-autoinc-perf).

Pre-fix, every enclosed-fill ran np.max over ALL work_labels volumes
(~265ms on busy axial patches).  Now the sync manager seeds a counter
once (one scan) then increments it per fill.  Correctness keystone: the
counter must never hand out a colliding ID (→ merged cells), so it
re-seeds whenever labels are introduced outside the fill path
(invalidate_next_fill_label_cache, wired into the controller's
_insert_slice_mask_with_unique_labels chokepoint).

Pure-Python — no Qt/napari widgets.
"""
from __future__ import annotations

import inspect
import numpy as np
from unittest.mock import MagicMock

from packages.epicycle_annotator.sync import LabelSyncManager


class _FakeLayer:
    def __init__(self, data, name="work_labels"):
        self.data = np.asarray(data)
        self.name = name


def _make_sync(main_layers=None):
    mv = MagicMock()
    mv.layers = list(main_layers or [])
    sm = LabelSyncManager(
        mv, MagicMock(), MagicMock(), MagicMock(),
        MagicMock(), MagicMock(), MagicMock(),
    )
    return sm, mv


# ---------------------------------------------------------------------------
# next_fill_label — seed once, then increment
# ---------------------------------------------------------------------------


def test_seeds_then_increments():
    sm, mv = _make_sync()
    active = _FakeLayer([[0, 5], [3, 5]])  # max 5
    assert sm.next_fill_label(active) == 6   # seed=5, +1
    assert sm.next_fill_label(active) == 7   # increment
    assert sm.next_fill_label(active) == 8


def test_no_rescan_after_seed():
    """Proves the cache: after seeding, an external change to the layer's
    data is NOT re-scanned (the counter just increments)."""
    sm, mv = _make_sync()
    active = _FakeLayer([5])
    assert sm.next_fill_label(active) == 6   # seed 5
    active.data = np.array([100])            # external change
    assert sm.next_fill_label(active) == 7   # IGNORED — no re-scan (cache)


def test_just_painted_advances_counter():
    """An R-picked / manually-overridden label higher than the counter
    advances it, so the next fill can't collide with it."""
    sm, mv = _make_sync()
    active = _FakeLayer([5])
    sm.next_fill_label(active)               # seed 5 -> 6
    assert sm.next_fill_label(active, just_painted=50) == 51


def test_invalidate_forces_reseed():
    sm, mv = _make_sync()
    active = _FakeLayer([5])
    assert sm.next_fill_label(active) == 6
    active.data = np.array([200])            # external labels added
    sm.invalidate_next_fill_label_cache()
    assert sm.next_fill_label(active) == 201  # re-scanned -> sees 200 -> 201


def test_monotonic_unique_sequence():
    sm, mv = _make_sync()
    active = _FakeLayer([0])
    labels = [sm.next_fill_label(active) for _ in range(20)]
    assert labels == sorted(labels)              # strictly increasing
    assert len(set(labels)) == len(labels)       # all unique


def test_seed_scans_all_main_layers_for_global_max(monkeypatch):
    """The seed scan must find the GLOBAL max across the active layer +
    every main-viewer Labels layer (cross-orientation uniqueness)."""
    from packages.epicycle_annotator import sync as sync_mod
    active = _FakeLayer([5], name="work_labels_axial_xz_3d_XZ")
    other = _FakeLayer([99], name="work_labels_axial_yz_3d")  # higher in YZ
    sm, mv = _make_sync(main_layers=[active, other])
    monkeypatch.setattr(sync_mod, "Labels", _FakeLayer)
    # global max = max(active=5, other=99) = 99 -> next = 100
    assert sm.next_fill_label(active) == 100


def test_empty_volume_seeds_to_one():
    sm, mv = _make_sync()
    active = _FakeLayer(np.zeros((4, 4), dtype=np.int32))  # all background
    assert sm.next_fill_label(active) == 1   # seed 0 -> 1


def test_cache_starts_none():
    sm, _ = _make_sync()
    assert sm._next_fill_label_cache is None


# ---------------------------------------------------------------------------
# Wiring (source inspection) — the collision-avoidance chokepoint
# ---------------------------------------------------------------------------


def test_insert_chokepoint_invalidates_cache():
    """The single label-introducing chokepoint must invalidate the cache
    so the counter can't go stale-LOW and collide."""
    from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
        TrainingAnnotationController,
    )
    src = inspect.getsource(
        TrainingAnnotationController._insert_slice_mask_with_unique_labels
    )
    # Both return paths must invalidate.
    assert src.count("_invalidate_annotator_fill_label_cache()") == 2, (
        "both return paths of _insert_slice_mask_with_unique_labels must "
        "invalidate the fill-label cache"
    )


def test_invalidate_helper_reaches_sync_manager():
    from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
        TrainingAnnotationController,
    )
    src = inspect.getsource(
        TrainingAnnotationController._invalidate_annotator_fill_label_cache
    )
    assert "invalidate_next_fill_label_cache" in src
    assert "_find_open_2d_annotator" in src


def test_drawing_uses_cached_counter_not_scan():
    """The drawing handler's auto-increment must call next_fill_label, NOT
    the old per-fill np.max scan."""
    from packages.epicycle_annotator.drawing import DrawingHandler
    src = inspect.getsource(DrawingHandler._on_mouse_drag)
    assert "next_fill_label" in src, (
        "auto-increment must use the cached next_fill_label"
    )
    # The old scan loop (np.max over main_viewer.layers) must be gone from
    # the auto-increment block.
    assert "for lyr in main_viewer.layers" not in src, (
        "the per-fill full-volume scan loop must be removed"
    )
