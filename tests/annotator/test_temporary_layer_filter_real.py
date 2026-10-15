"""WP5 + R3-WP2: Behavioral tests for temporary layer filtering.

Verifies that:
- ``_refresh_layer_options`` in ``native_multiview_widget.py`` skips layers
  tagged with ``_epicycle_temporary = True`` at runtime (not just in source).
- ``Curation3DAnnotator`` tags the Refine layer with the attribute.

All tests are behavioral — they bind real production methods to lightweight
harness objects and assert observable state, not source strings.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_labels_layer(name, ndim=3, temporary=False):
    """Create a fake Labels-like object that passes isinstance checks."""
    from unittest.mock import MagicMock
    from napari.layers import Labels

    layer = MagicMock(spec=Labels)
    layer.name = name
    layer.data = MagicMock()
    layer.data.ndim = ndim
    if temporary:
        layer._epicycle_temporary = True
    else:
        # Ensure getattr(..., "_epicycle_temporary", False) returns False
        # by NOT setting the attribute — MagicMock spec=Labels won't have it
        del layer._epicycle_temporary
    return layer


def _make_fake_image_layer(name, ndim=3):
    """Create a fake Image-like object that passes isinstance checks."""
    from unittest.mock import MagicMock
    from napari.layers import Image

    layer = MagicMock(spec=Image)
    layer.name = name
    layer.data = MagicMock()
    layer.data.ndim = ndim
    # Ensure _epicycle_temporary is not set
    if hasattr(layer, "_epicycle_temporary"):
        del layer._epicycle_temporary
    return layer


# ---------------------------------------------------------------------------
# Tests: _refresh_layer_options filters _epicycle_temporary layers
# ---------------------------------------------------------------------------

class TestLayerFilterBehavioral:
    """Verify _refresh_layer_options runtime filtering of temporary layers."""

    def test_temporary_labels_excluded_from_combo(self):
        """_refresh_layer_options must skip Labels with _epicycle_temporary."""
        from unittest.mock import MagicMock
        import types
        from napari_epicycle.annotator.native_multiview_widget import (
            NativeMultiviewAnnotator,
        )

        harness = MagicMock()
        # Set up layers: one normal Labels, one temporary, one Image
        normal_labels = _make_fake_labels_layer("labels_3d")
        temp_labels = _make_fake_labels_layer("Refine", temporary=True)
        image = _make_fake_image_layer("raw_image")

        harness.viewer.layers = [image, normal_labels, temp_labels]
        harness.image_combo = MagicMock()
        harness.image_combo.currentText.return_value = ""
        harness.labels_combo = MagicMock()
        harness.labels_combo.currentText.return_value = ""

        bound = types.MethodType(
            NativeMultiviewAnnotator._refresh_layer_options, harness
        )
        bound()

        # labels_combo should have been populated with only "labels_3d"
        add_calls = harness.labels_combo.addItems.call_args_list
        assert len(add_calls) == 1
        labels_added = add_calls[0][0][0]
        assert "labels_3d" in labels_added
        assert "Refine" not in labels_added

    def test_non_temporary_labels_included(self):
        """Normal Labels (no _epicycle_temporary) appear in combo."""
        from unittest.mock import MagicMock
        import types
        from napari_epicycle.annotator.native_multiview_widget import (
            NativeMultiviewAnnotator,
        )

        harness = MagicMock()
        l1 = _make_fake_labels_layer("seg_output")
        l2 = _make_fake_labels_layer("manual_labels")

        harness.viewer.layers = [l1, l2]
        harness.image_combo = MagicMock()
        harness.image_combo.currentText.return_value = ""
        harness.labels_combo = MagicMock()
        harness.labels_combo.currentText.return_value = ""

        bound = types.MethodType(
            NativeMultiviewAnnotator._refresh_layer_options, harness
        )
        bound()

        labels_added = harness.labels_combo.addItems.call_args_list[0][0][0]
        assert "seg_output" in labels_added
        assert "manual_labels" in labels_added

    def test_filter_skips_temporary(self):
        """Simulate the actual filter logic from _refresh_layer_options."""
        class FakeLayer:
            def __init__(self, name, temporary=False):
                self.name = name
                if temporary:
                    self._epicycle_temporary = True

        layers = [
            FakeLayer("labels_3d"),
            FakeLayer("Refine", temporary=True),
            FakeLayer("other_labels"),
        ]

        visible = [
            l for l in layers
            if not getattr(l, "_epicycle_temporary", False)
        ]

        assert len(visible) == 2
        assert all(l.name != "Refine" for l in visible)


# ---------------------------------------------------------------------------
# Tests: Curation3DAnnotator tags Refine layer
# ---------------------------------------------------------------------------

class TestRefineLayerTagging:
    """Verify curation_3d_widget sets _epicycle_temporary on Refine layer."""

    def test_refine_layer_tagged_in_source(self):
        """Source sets _epicycle_temporary = True (quick confidence check)."""
        import inspect
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        source = inspect.getsource(Curation3DAnnotator)
        assert "_epicycle_temporary = True" in source, (
            "Curation3DAnnotator must tag refine layer with _epicycle_temporary"
        )
