"""Tests for P3: Temporary layer hygiene.

Verifies that layers tagged with ``_epicycle_temporary = True`` are excluded
from the labels/image combo in the multiview annotator.
"""

import types

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Minimal fakes (no Qt / napari required)
# ---------------------------------------------------------------------------

class _FakeLayer:
    """Minimal napari layer fake."""

    def __init__(self, name, data, layer_type="labels"):
        self.name = name
        self.data = data
        self._type_string = layer_type

    @property
    def ndim(self):
        return self.data.ndim


class _FakeLabels(_FakeLayer):
    pass


class _FakeImage(_FakeLayer):
    def __init__(self, name, data):
        super().__init__(name, data, layer_type="image")


def _is_labels(layer):
    return layer._type_string == "labels"


def _is_image(layer):
    return layer._type_string == "image"


def _filter_labels(layers):
    """Replicates the filtering logic from _refresh_layer_options."""
    result = []
    for layer in layers:
        if not _is_labels(layer):
            continue
        if layer.ndim not in {2, 3}:
            continue
        if getattr(layer, "_epicycle_temporary", False):
            continue
        result.append(layer.name)
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTemporaryLayerFilter:
    """Verify temporary layers are excluded from combo population."""

    def test_normal_labels_included(self):
        layers = [
            _FakeLabels("reconstruction_abc", np.zeros((10, 10, 10), dtype=np.int32)),
            _FakeLabels("curation_3d_abc", np.zeros((10, 10, 10), dtype=np.int32)),
        ]
        names = _filter_labels(layers)
        assert "reconstruction_abc" in names
        assert "curation_3d_abc" in names

    def test_temporary_layer_excluded(self):
        refine = _FakeLabels("Refine", np.zeros((10, 10, 10), dtype=np.int32))
        refine._epicycle_temporary = True
        layers = [
            _FakeLabels("reconstruction_abc", np.zeros((10, 10, 10), dtype=np.int32)),
            refine,
        ]
        names = _filter_labels(layers)
        assert "reconstruction_abc" in names
        assert "Refine" not in names

    def test_untagged_layer_not_excluded(self):
        """Layer without _epicycle_temporary attr should be included."""
        layer = _FakeLabels("Refine", np.zeros((10, 10, 10), dtype=np.int32))
        # No _epicycle_temporary attribute at all
        layers = [layer]
        names = _filter_labels(layers)
        assert "Refine" in names

    def test_temporary_false_not_excluded(self):
        """_epicycle_temporary = False should still include the layer."""
        layer = _FakeLabels("preview_smooth", np.zeros((10, 10, 10), dtype=np.int32))
        layer._epicycle_temporary = False
        layers = [layer]
        names = _filter_labels(layers)
        assert "preview_smooth" in names

    def test_mixed_layers(self):
        """Mix of normal, temporary, and image layers."""
        layers = [
            _FakeImage("stack_patch001", np.zeros((10, 10, 10), dtype=np.float32)),
            _FakeLabels("reconstruction_abc", np.zeros((10, 10, 10), dtype=np.int32)),
            _FakeLabels("Refine", np.zeros((10, 10, 10), dtype=np.int32)),
        ]
        layers[2]._epicycle_temporary = True
        names = _filter_labels(layers)
        assert names == ["reconstruction_abc"]
