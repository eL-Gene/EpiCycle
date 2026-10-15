"""Tests for 3D isolation instrumentation (Phase 0).

Verifies that the navigation+isolation pipeline has end-to-end timing
instrumentation at each stage, so we can identify where latency comes
from in 3D mode.

All tests are Qt-free using source inspection.
"""

from __future__ import annotations

import inspect

import pytest


class TestNavigatorInstrumentation:
    """Navigator should have timing spans for row click dispatch."""

    def test_on_row_clicked_has_timing(self):
        """_on_row_clicked should measure total callback dispatch time."""
        from napari_epicycle.annotator.curation_label_navigator import (
            CurationLabelNavigator,
        )

        source = inspect.getsource(CurationLabelNavigator._on_row_clicked)
        assert "perf_counter" in source or "ISO-TRACE" in source, (
            "_on_row_clicked should have timing instrumentation"
        )

    def test_fire_isolation_has_timing(self):
        """_fire_isolation_if_active should measure isolation callback time."""
        from napari_epicycle.annotator.curation_label_navigator import (
            CurationLabelNavigator,
        )

        source = inspect.getsource(
            CurationLabelNavigator._fire_isolation_if_active
        )
        assert "perf_counter" in source or "ISO-TRACE" in source, (
            "_fire_isolation_if_active should have timing instrumentation"
        )


class TestAnnotatorInstrumentation:
    """Annotator should have timing spans for selection + isolation."""

    def test_on_navigator_select_label_has_timing(self):
        """_on_navigator_select_label should measure bridge dispatch time."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        source = inspect.getsource(
            Curation3DAnnotator._on_navigator_select_label
        )
        assert "perf_counter" in source or "ISO-TRACE" in source, (
            "_on_navigator_select_label should have timing instrumentation"
        )

    def test_on_isolation_changed_has_timing(self):
        """_on_isolation_changed should measure isolation apply/clear time."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        source = inspect.getsource(
            Curation3DAnnotator._on_isolation_changed
        )
        # Already has perf_counter from prior work
        assert "perf_counter" in source, (
            "_on_isolation_changed should have timing instrumentation"
        )


class TestCoordinatorInstrumentation:
    """Coordinator should have timing spans for dispatch routing."""

    def test_dispatch_has_timing(self):
        """dispatch() should measure total routing time."""
        from napari_epicycle.fusion_reconstruction.curation_selection_sync import (
            CurationSelectionSyncCoordinator,
        )

        source = inspect.getsource(
            CurationSelectionSyncCoordinator.dispatch
        )
        assert "perf_counter" in source or "ISO-TRACE" in source, (
            "dispatch should have timing instrumentation"
        )


class TestControllerInstrumentation:
    """Controller isolation adapter should have timing spans."""

    def test_apply_viewer_isolation_adapter_has_timing(self):
        """_apply_viewer_isolation_adapter should measure isolation time."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )

        source = inspect.getsource(
            FusionReconstructionController._apply_viewer_isolation_adapter
        )
        assert "perf_counter" in source or "ISO-TRACE" in source, (
            "_apply_viewer_isolation_adapter should have timing instrumentation"
        )

    def test_isolate_labels_in_viewer_has_timing(self):
        """_isolate_labels_in_viewer should measure colormap build + apply time."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )

        source = inspect.getsource(
            FusionReconstructionController._isolate_labels_in_viewer
        )
        assert "perf_counter" in source or "ISO-TRACE" in source, (
            "_isolate_labels_in_viewer should have timing instrumentation"
        )


class TestWriteCountInstrumentation:
    """Pipeline should count property writes per navigation step."""

    def test_set_selected_label_all_counts_writes(self):
        """_set_selected_label_all should report property write to ISO-TRACE."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        source = inspect.getsource(
            Curation3DAnnotator._set_selected_label_all
        )
        assert "ISO-TRACE" in source or "perf_counter" in source, (
            "_set_selected_label_all should have timing instrumentation"
        )
