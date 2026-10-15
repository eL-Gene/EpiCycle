"""Tests for Phase 1 — 3D single-label fast lane.

Verifies that when ndisplay=3 and a single label is selected with
isolate ON, the annotator dispatches mode="focus" (not "isolate")
to the controller bridge, skipping the expensive controller-side
_isolate_labels_in_viewer().  The annotator's _fire_isolation_if_active
handles isolation locally via show_selected_label=True.

Multi-label isolation in 3D and all 2D isolation paths remain unchanged.

All tests are Qt-free using source inspection + lightweight fakes.
"""

from __future__ import annotations

import inspect
import types

import numpy as np
import pytest


class TestFastLaneBridgeDispatch:
    """Navigator bridge should skip controller isolation in 3D single-label."""

    def test_on_navigator_select_label_checks_ndisplay(self):
        """_on_navigator_select_label should check ndisplay for 3D fast lane."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        source = inspect.getsource(
            Curation3DAnnotator._on_navigator_select_label
        )
        assert "ndisplay" in source, (
            "_on_navigator_select_label should check ndisplay for 3D fast lane"
        )

    def test_3d_single_label_forces_focus_mode(self):
        """In 3D with single label, bridge should dispatch mode='focus'."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        source = inspect.getsource(
            Curation3DAnnotator._on_navigator_select_label
        )
        # Should have logic that sets isolate=False when ndisplay==3
        # and single label selected
        assert "3" in source and "ndisplay" in source, (
            "Should check for ndisplay == 3"
        )
        # Should reference the fast lane concept or explicitly set isolate=False
        assert "isolate = False" in source or "isolate=False" in source, (
            "Should force isolate=False for 3D single-label fast lane"
        )

    def test_multi_label_3d_still_dispatches_isolate(self):
        """Multi-label in 3D should NOT use the fast lane."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        source = inspect.getsource(
            Curation3DAnnotator._on_navigator_select_label
        )
        # The fast lane condition should check len(all_sel) == 1
        assert "len(all_sel)" in source or "len(all_sel) == 1" in source, (
            "Fast lane should only apply to single-label selection"
        )


class TestFastLanePreservesAnnotatorIsolation:
    """Annotator local isolation path must still work in 3D."""

    def test_on_isolation_changed_single_label_uses_show_selected(self):
        """Single-label path should use show_selected_label (fast shader)."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        source = inspect.getsource(
            Curation3DAnnotator._on_isolation_changed
        )
        assert "show_selected_label" in source, (
            "Single-label isolation should use show_selected_label"
        )

    def test_fire_isolation_still_called_after_bridge(self):
        """Navigator must fire _fire_isolation_if_active after bridge dispatch."""
        from napari_epicycle.annotator.curation_label_navigator import (
            CurationLabelNavigator,
        )

        source = inspect.getsource(
            CurationLabelNavigator._on_row_clicked
        )
        # _fire_isolation_if_active must appear AFTER _on_select_label_cb
        select_pos = source.find("_on_select_label_cb")
        isolate_pos = source.find("_fire_isolation_if_active")
        assert select_pos > 0 and isolate_pos > 0, (
            "Both callbacks should exist in _on_row_clicked"
        )
        assert isolate_pos > select_pos, (
            "_fire_isolation_if_active must fire AFTER _on_select_label_cb"
        )


class TestFastLaneDoesNotAffectLasso:
    """Lasso isolation (controller path) must be unaffected by fast lane."""

    def test_controller_isolate_labels_independent_of_fast_lane(self):
        """_isolate_labels_in_viewer should not check ndisplay for skip."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )

        source = inspect.getsource(
            FusionReconstructionController._isolate_labels_in_viewer
        )
        # The controller's isolation should NOT have an ndisplay-based skip —
        # that would break lasso isolation in 3D.  The fast lane is only in
        # the annotator's bridge dispatch.
        lines = source.split('\n')
        ndisplay_skip_lines = [
            line for line in lines
            if 'ndisplay' in line and ('return' in line or 'skip' in line.lower())
        ]
        assert len(ndisplay_skip_lines) == 0, (
            "_isolate_labels_in_viewer should NOT have ndisplay-based skip "
            "(would break lasso isolation)"
        )


# -----------------------------------------------------------------------
# Behavioral tests with lightweight fakes
# -----------------------------------------------------------------------

def _make_fast_lane_annotator(*, ndisplay=3, isolate_checked=True,
                               multi_select=False):
    """Create annotator fake for fast lane behavioral tests.

    Returns (ann, layer, bridge_calls).
    """
    from napari.layers import Labels as _Labels
    from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator
    from napari_epicycle.fusion_reconstruction.colormap_authority import (
        ColormapAuthority,
    )

    layer = _Labels(np.zeros((4, 4, 4), dtype=np.int32), name="labels")
    layer.data[1, 1, 1] = 5
    if multi_select:
        layer.data[2, 2, 2] = 7
    layer.selected_label = 5

    class _FakeLayerList:
        def __contains__(self, name):
            return name == "labels"
        def __getitem__(self, name):
            if name == "labels":
                return layer
            raise KeyError(name)
        def __iter__(self):
            return iter([layer])

    dims = types.SimpleNamespace(ndisplay=ndisplay)
    viewer = types.SimpleNamespace(layers=_FakeLayerList(), dims=dims)

    bridge_calls = []

    def _bridge(label_id, *, all_selected=None, isolate=False):
        bridge_calls.append({
            "label_id": label_id,
            "all_selected": all_selected,
            "isolate": isolate,
        })

    selected_ids = {5, 7} if multi_select else {5}
    nav = types.SimpleNamespace()
    nav.selected_label_ids = lambda: selected_ids
    nav._isolate_cb = types.SimpleNamespace(
        isChecked=lambda: isolate_checked,
    )
    nav.remove_item = lambda lid: None

    auth = ColormapAuthority()

    class _FakeCombo:
        _text = "labels"
        def setCurrentText(self, t):
            self._text = t
        def currentText(self):
            return self._text

    ann = types.SimpleNamespace()
    ann._active_labels_name = "labels"
    ann.viewer = viewer
    ann.labels_combo = _FakeCombo()
    ann._split_running = False
    ann._split_fragment_refine = False
    ann._smooth_running = False
    ann._last_navigated_label_id = None
    ann._navigator_widget = nav
    ann._on_navigator_label_selected_cb = _bridge
    ann._colormap_authority = auth
    ann._isolation_bridge_dispatched = False

    ann._set_selected_label_all = lambda lid: None
    ann._check_label_exists_in_volume = types.MethodType(
        Curation3DAnnotator._check_label_exists_in_volume, ann
    )
    ann._handle_manually_deleted_label = lambda lid: None
    ann._layer_by_name = types.MethodType(
        Curation3DAnnotator._layer_by_name, ann
    )
    ann._on_navigator_select_label = types.MethodType(
        Curation3DAnnotator._on_navigator_select_label, ann
    )

    return ann, layer, bridge_calls


class TestFastLaneBehavioral:
    """Behavioral tests: verify bridge dispatch mode with fake viewer."""

    def test_3d_single_label_dispatches_focus(self):
        """ndisplay=3 + single label + isolate ON → mode='focus'."""
        ann, layer, calls = _make_fast_lane_annotator(
            ndisplay=3, isolate_checked=True, multi_select=False,
        )
        ann._on_navigator_select_label(5)
        assert len(calls) == 1
        assert calls[0]["isolate"] is False, (
            "3D single-label should force isolate=False (fast lane)"
        )

    def test_2d_single_label_dispatches_isolate(self):
        """ndisplay=2 + single label + isolate ON → mode='isolate'."""
        ann, layer, calls = _make_fast_lane_annotator(
            ndisplay=2, isolate_checked=True, multi_select=False,
        )
        ann._on_navigator_select_label(5)
        assert len(calls) == 1
        assert calls[0]["isolate"] is True, (
            "2D single-label should keep isolate=True (no fast lane)"
        )

    def test_3d_multi_label_dispatches_isolate(self):
        """ndisplay=3 + multi label + isolate ON → mode='isolate'."""
        ann, layer, calls = _make_fast_lane_annotator(
            ndisplay=3, isolate_checked=True, multi_select=True,
        )
        ann._on_navigator_select_label(5)
        assert len(calls) == 1
        assert calls[0]["isolate"] is True, (
            "3D multi-label should keep isolate=True (no fast lane)"
        )

    def test_3d_isolate_off_dispatches_focus(self):
        """ndisplay=3 + isolate OFF → mode='focus' regardless."""
        ann, layer, calls = _make_fast_lane_annotator(
            ndisplay=3, isolate_checked=False, multi_select=False,
        )
        ann._on_navigator_select_label(5)
        assert len(calls) == 1
        assert calls[0]["isolate"] is False

    def test_guard_flag_set_in_3d_fast_lane(self):
        """Guard flag should still be set even when fast lane triggers."""
        ann, layer, calls = _make_fast_lane_annotator(
            ndisplay=3, isolate_checked=True, multi_select=False,
        )
        ann._on_navigator_select_label(5)
        assert ann._isolation_bridge_dispatched is True, (
            "Guard flag must be set even in fast lane mode"
        )
