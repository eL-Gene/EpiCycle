"""Tests: View layers cleared when entering Curation.

When the user Views patches via Predict+Reconstruct and then loads
a patch in Curation, the stale View layers should be auto-cleared.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock


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

    def __len__(self):
        return len(self._layers)

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


class TestViewToCurationCleanup:
    """View layers auto-cleared when entering curation."""

    def test_view_layers_cleared_when_opening_curation(self, manager, viewer):
        """Simulate: View patch A, then open curation on patch B."""
        # User views patch A via Predict+Reconstruct
        manager.add_labels(None, name="reconstruction_A", context="step3/view")
        manager.add_image(None, name="stack_patchA", context="step3/view")

        # User opens curation on patch B — curation registers its own layers
        viewer.add_labels(None, name="curation_3d_B")
        manager.register_existing("curation_3d_B", "step3/curation")
        # Then clears step3/view (what our code now does)
        manager.clear_context("step3/view")

        # View layers gone, curation layer remains
        assert "reconstruction_A" not in viewer.layers
        assert "stack_patchA" not in viewer.layers
        assert "curation_3d_B" in viewer.layers

    def test_reused_view_layer_survives_clear(self, manager, viewer):
        """Simulate: View patch A, then open curation on SAME patch A.
        The reconstruction layer is renamed to curation_3d — it should
        survive the step3/view clear because it was moved to step3/curation."""
        # View patch A
        manager.add_labels(None, name="reconstruction_A", context="step3/view")
        manager.add_image(None, name="stack_patchA", context="step3/view")

        # Curation opens on patch A — rename reconstruction → curation_3d
        manager.rename("reconstruction_A", "curation_3d_A", "step3/curation")
        # Stack also moves to curation context
        manager.rename("stack_patchA", "stack_patchA", "step3/curation")

        # Clear remaining View layers (should be empty now)
        manager.clear_context("step3/view")

        # Both layers survive — they're in step3/curation, not step3/view
        assert "curation_3d_A" in manager.layers_in_context("step3/curation")
        assert "stack_patchA" in manager.layers_in_context("step3/curation")

    def test_multiple_viewed_patches_all_cleared(self, manager, viewer):
        """User views 3 patches, then enters curation — all cleared."""
        for i in range(3):
            manager.add_labels(None, name=f"reconstruction_{i}", context="step3/view")
            manager.add_image(None, name=f"stack_patch{i}", context="step3/view")

        assert len(list(viewer.layers)) == 6
        manager.clear_context("step3/view")
        assert len(list(viewer.layers)) == 0

    def test_qc_view_load_clears_view_layers(self):
        """_load_patch_for_qc_view should clear step3/view context."""
        import inspect
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        source = inspect.getsource(
            FusionReconstructionController._load_patch_for_qc_view
        )
        assert 'clear_context("step3/view")' in source

    def test_open_annotator_clears_view_layers(self):
        """_open_3d_curation_annotator should clear step3/view context."""
        import inspect
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        source = inspect.getsource(
            FusionReconstructionController._open_3d_curation_annotator
        )
        assert 'clear_context("step3/view")' in source
