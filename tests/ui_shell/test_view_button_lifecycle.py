"""Tests for Phase 2: View button layer lifecycle integration.

Verifies that _view_reconstruction() registers layers with the global
LayerLifecycleManager and that clear_context("step3/view") removes them.
Uses lightweight fakes — no Qt or napari required.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Fakes (reuse from test_layer_lifecycle_manager.py)
# ---------------------------------------------------------------------------


class FakeLayer:
    def __init__(self, name, visible=True):
        self.name = name
        self.visible = visible


class FakeLayerList:
    def __init__(self):
        self._layers: dict[str, FakeLayer] = {}
        self.events = MagicMock()

    def __contains__(self, name):
        return name in self._layers

    def __getitem__(self, name):
        return self._layers[name]

    def __iter__(self):
        return iter(self._layers.values())

    def add(self, layer):
        self._layers[layer.name] = layer
        return layer

    def remove(self, layer):
        if isinstance(layer, str):
            self._layers.pop(layer, None)
        else:
            self._layers.pop(layer.name, None)


class FakeViewer:
    def __init__(self):
        self.layers = FakeLayerList()

    def add_image(self, data, *, name, **kwargs):
        layer = FakeLayer(name)
        self.layers.add(layer)
        return layer

    def add_labels(self, data, *, name, **kwargs):
        layer = FakeLayer(name)
        self.layers.add(layer)
        return layer


@pytest.fixture
def viewer():
    return FakeViewer()


@pytest.fixture
def manager(viewer):
    from napari_epicycle.ui_shell.layer_lifecycle_manager import (
        LayerLifecycleManager,
    )
    return LayerLifecycleManager(viewer)


# ---------------------------------------------------------------------------
# Tests — View button registration pattern
# ---------------------------------------------------------------------------


class TestViewButtonRegistration:
    """Simulate the View button's layer creation and verify lifecycle tracking."""

    def test_view_layers_registered_under_step3_view(self, manager, viewer):
        """Layers created by View should be tracked in step3/view context."""
        # Simulate _view_reconstruction behavior: add stack + labels
        viewer.add_image(None, name="stack_patch1")
        manager.register_existing("stack_patch1", "step3/view")

        viewer.add_labels(None, name="reconstruction_abc123")
        manager.register_existing("reconstruction_abc123", "step3/view")

        names = manager.layers_in_context("step3/view")
        assert "stack_patch1" in names
        assert "reconstruction_abc123" in names

    def test_clear_view_removes_all_view_layers(self, manager, viewer):
        """clear_context('step3/view') should remove View layers."""
        viewer.add_image(None, name="stack_patch1")
        manager.register_existing("stack_patch1", "step3/view")
        viewer.add_labels(None, name="reconstruction_abc123")
        manager.register_existing("reconstruction_abc123", "step3/view")

        manager.clear_context("step3/view")

        assert "stack_patch1" not in viewer.layers
        assert "reconstruction_abc123" not in viewer.layers

    def test_view_patch_a_then_view_patch_b_clears_a(self, manager, viewer):
        """Viewing a new patch should clear the previous View layers."""
        # View patch A
        viewer.add_image(None, name="stack_patchA")
        manager.register_existing("stack_patchA", "step3/view")
        viewer.add_labels(None, name="reconstruction_A")
        manager.register_existing("reconstruction_A", "step3/view")

        # View patch B (clear first, then add)
        manager.clear_context("step3/view")
        viewer.add_image(None, name="stack_patchB")
        manager.register_existing("stack_patchB", "step3/view")
        viewer.add_labels(None, name="reconstruction_B")
        manager.register_existing("reconstruction_B", "step3/view")

        assert "stack_patchA" not in viewer.layers
        assert "reconstruction_A" not in viewer.layers
        assert "stack_patchB" in viewer.layers
        assert "reconstruction_B" in viewer.layers

    def test_view_layers_dont_affect_other_contexts(self, manager, viewer):
        """Clearing step3/view should not touch curation or QC layers."""
        viewer.add_labels(None, name="curation_3d_001")
        manager.register_existing("curation_3d_001", "step3/curation")
        viewer.add_image(None, name="stack_view")
        manager.register_existing("stack_view", "step3/view")

        manager.clear_context("step3/view")

        assert "curation_3d_001" in viewer.layers
        assert "stack_view" not in viewer.layers

    def test_channel_split_layers_tracked(self, manager, viewer):
        """Multi-channel stacks create 'stack_foo [0]', 'stack_foo [1]' etc."""
        # Simulate napari channel split naming
        for ch in range(3):
            name = f"stack_patch1 [{ch}]"
            viewer.add_image(None, name=name)
            manager.register_existing(name, "step3/view")

        names = manager.layers_in_context("step3/view")
        assert len(names) == 3
        manager.clear_context("step3/view")
        assert len(list(viewer.layers)) == 0
