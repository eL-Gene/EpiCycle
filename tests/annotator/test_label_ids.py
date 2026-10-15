"""Tests for ``next_unused_label_id`` — the centralized "fresh
label ID" helper used by the Label Navigator's New Label button
and (eventually) by other label-allocating callers.

These tests use pure numpy fakes — no Qt / napari instantiation.
See ``packages/epicycle_annotator/label_ids.py`` for the function
under test.
"""
from __future__ import annotations

import numpy as np
import pytest

from packages.epicycle_annotator.label_ids import next_unused_label_id


class _FakeLayer:
    """Minimal layer-like object — just exposes ``.data``."""

    def __init__(self, data):
        self.data = np.asarray(data) if data is not None else None


# =============================================================================
# Basic behavior
# =============================================================================


class TestBasic:
    def test_empty_iterable_returns_1(self):
        assert next_unused_label_id([]) == 1

    def test_single_layer_returns_max_plus_1(self):
        layer = _FakeLayer([1, 2, 3])
        assert next_unused_label_id([layer]) == 4

    def test_multiple_layers_uses_global_max(self):
        l1 = _FakeLayer([1, 2])
        l2 = _FakeLayer([3, 4, 5])
        l3 = _FakeLayer([6])
        assert next_unused_label_id([l1, l2, l3]) == 7

    def test_uses_max_not_first_unused(self):
        """Gap-filling is INTENTIONALLY not done — fresh IDs always
        come from above the max so they're guaranteed never to have
        been used in this workspace.  Gap-filling would risk reusing
        a previously-deleted label's slot."""
        layer = _FakeLayer([1, 3, 5])
        assert next_unused_label_id([layer]) == 6

    def test_zeros_treated_as_background(self):
        """np.max already handles this — zeros don't artificially
        inflate the result."""
        layer = _FakeLayer([0, 0, 5, 0])
        assert next_unused_label_id([layer]) == 6

    def test_all_zeros_returns_1(self):
        layer = _FakeLayer([0, 0, 0])
        assert next_unused_label_id([layer]) == 1


# =============================================================================
# Multidimensional arrays
# =============================================================================


class TestMultidim:
    def test_2d_volume(self):
        layer = _FakeLayer([[1, 2], [3, 4]])
        assert next_unused_label_id([layer]) == 5

    def test_3d_volume(self):
        data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        layer = _FakeLayer(data)
        assert next_unused_label_id([layer]) == 9


# =============================================================================
# Defensive paths
# =============================================================================


class TestDefensive:
    def test_layer_without_data_attr_is_skipped(self):
        from types import SimpleNamespace
        bad = SimpleNamespace()  # no .data
        good = _FakeLayer([5])
        assert next_unused_label_id([bad, good]) == 6

    def test_layer_with_none_data_is_skipped(self):
        bad = _FakeLayer(None)
        good = _FakeLayer([5])
        assert next_unused_label_id([bad, good]) == 6

    def test_empty_array_is_skipped(self):
        empty = _FakeLayer(np.array([], dtype=np.int32))
        good = _FakeLayer([5])
        assert next_unused_label_id([empty, good]) == 6

    def test_non_numeric_data_is_skipped(self):
        """A layer whose .data raises on np.max should not crash the
        helper — returns the max from the remaining valid layers."""
        class _BadLayer:
            data = object()  # not array-like
        good = _FakeLayer([5])
        assert next_unused_label_id([_BadLayer(), good]) == 6

    def test_all_layers_invalid_returns_1(self):
        """Floor: even when nothing usable is provided, return 1."""
        from types import SimpleNamespace
        bad1 = SimpleNamespace()
        bad2 = _FakeLayer(None)
        assert next_unused_label_id([bad1, bad2]) == 1


# =============================================================================
# The bug case that motivated this helper (regression, 2026-05-16)
# =============================================================================


class TestNewLabelButtonBugCase:
    """User reported: New Label button produced an ID that collided
    with an existing label, joining the new paint stroke to a pre-
    existing cell.  Root cause: ``_next_label_id_after_accept`` used
    max(navigator items) + 1, but the navigator was showing a filtered
    view of the labels — so max(items) was LESS than the volume's
    actual max.

    These tests pin down the contract: regardless of what subset is
    visible to any other component, ``next_unused_label_id`` returns
    a value strictly greater than every label present in the volume.
    """

    def test_filtered_navigator_does_not_affect_result(self):
        """The navigator might know about only {1, 2, 3, 4} but the
        volume actually has {1, 2, 3, 4, 5, 6}.  Helper must return
        7, not 5 (which is what the buggy navigator-fast path
        returned)."""
        volume_layer = _FakeLayer(
            np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
        )
        # next_unused_label_id is given the layer, not the navigator.
        # The navigator's filtered view is irrelevant.
        assert next_unused_label_id([volume_layer]) == 7

    def test_collision_avoided_with_sparse_volume(self):
        """Volume has labels {1, 2, 100} — far apart.  Result must
        be 101 (max+1), not anything in between that already exists."""
        volume_layer = _FakeLayer(
            np.array([1, 2, 100, 0, 0, 0], dtype=np.int32)
        )
        result = next_unused_label_id([volume_layer])
        assert result == 101
        # And critically: result is NOT one of the existing IDs
        existing = {1, 2, 100}
        assert result not in existing
