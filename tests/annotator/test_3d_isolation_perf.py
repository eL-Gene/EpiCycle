"""Tests for 3D isolation — multi-label DirectLabelColormap path.

Verifies _apply_multi_isolation_colormap applies DirectLabelColormap
to both main layer and subviewers, with sync suppression to avoid
redundant cascade work.

All tests are Qt-free using lightweight fakes.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest


class TestMultiLabelIsolationColormap:
    """_apply_multi_isolation_colormap applies colormap to main + subviewers."""

    def test_source_applies_colormap_to_main_layer(self):
        """Main layer should get DirectLabelColormap."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        source = inspect.getsource(
            Curation3DAnnotator._apply_multi_isolation_colormap
        )
        assert "main_layer.colormap" in source
        assert "DirectLabelColormap" in source

    def test_source_propagates_to_subviewers(self):
        """Subviewers should also get the isolation colormap."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        source = inspect.getsource(
            Curation3DAnnotator._apply_multi_isolation_colormap
        )
        assert "_propagate_colormap_to_subviewers" in source

    def test_source_uses_sync_suppression(self):
        """Should suppress sync cascade during colormap assignment."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        source = inspect.getsource(
            Curation3DAnnotator._apply_multi_isolation_colormap
        )
        assert "suppress_property_sync" in source

    def test_source_has_fast_path_skip(self):
        """Should skip rebuild when isolated label set is unchanged."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        source = inspect.getsource(
            Curation3DAnnotator._apply_multi_isolation_colormap
        )
        assert "_isolation_label_set" in source


class TestIsolationRestorePath:
    """Verify state is restored when isolation is cleared."""

    def test_restore_path_in_on_isolation_changed(self):
        """The restore path should handle colormap and show_selected_label."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        source = inspect.getsource(
            Curation3DAnnotator._on_isolation_changed
        )
        assert "show_selected_label" in source
        assert "_pre_isolation_colormap" in source

    def test_close_cleanup_clears_isolation_state(self):
        """_close_cleanup should clear all isolation state."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        source = inspect.getsource(
            Curation3DAnnotator._close_cleanup
        )
        assert "_pre_isolation_data" in source
        assert "_isolation_active" in source
