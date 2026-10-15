"""Tests: Bidirectional View ↔ Curation cleanup.

View → Curation: View layers cleared on curation entry (tested elsewhere).
Curation → View: Annotator closed + curation layers cleared when View is used.
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


class TestCurationToViewCleanup:
    """Using View button while curating should close annotator + clear curation layers."""

    def test_view_clears_curation_layers(self, manager, viewer):
        """Curation layers removed when View is invoked."""
        manager.add_labels(None, name="curation_3d_abc", context="step3/curation")
        manager.add_image(None, name="stack_patch1", context="step3/curation")

        # Simulate _view_reconstruction clearing curation context
        manager.clear_context("step3/curation")

        assert "curation_3d_abc" not in viewer.layers
        assert "stack_patch1" not in viewer.layers

    def test_view_clears_qc_single_layers(self, manager, viewer):
        """QC single-patch layers removed when View is invoked."""
        manager.add_labels(None, name="reconstruction_abc", context="step3/qc/single")
        manager.add_image(None, name="stack_qc", context="step3/qc/single")

        manager.clear_context("step3/qc/single")

        assert "reconstruction_abc" not in viewer.layers
        assert "stack_qc" not in viewer.layers

    def test_view_clears_all_step3_then_adds_fresh(self, manager, viewer):
        """Full flow: curation active → View → only View layers remain."""
        # Curation is active with layers
        manager.add_labels(None, name="curation_3d_old", context="step3/curation")
        manager.add_labels(None, name="recon_qc", context="step3/qc/single")

        # View button fires — clear everything, add fresh View layer
        manager.clear_context("step3/view")
        manager.clear_context("step3/curation")
        manager.clear_context("step3/qc/single")
        manager.add_labels(None, name="reconstruction_new", context="step3/view")

        assert "curation_3d_old" not in viewer.layers
        assert "recon_qc" not in viewer.layers
        assert "reconstruction_new" in viewer.layers

    def test_view_reconstruction_source_closes_annotator(self):
        """_view_reconstruction should close the annotator dock."""
        import inspect
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        source = inspect.getsource(
            FusionReconstructionController._view_reconstruction
        )
        assert "EpiCycle 3D Curation Annotator" in source
        assert "dock.close()" in source

    def test_view_reconstruction_source_clears_curation(self):
        """_view_reconstruction should clear step3/curation context (A2:
        through ``_view_patch``, the door Session's eye shares)."""
        import inspect
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        assert "_view_patch(" in inspect.getsource(
            FusionReconstructionController._view_reconstruction)
        source = inspect.getsource(
            FusionReconstructionController._view_patch
        )
        assert 'clear_context("step3/curation")' in source
        assert 'clear_context("step3/qc/single")' in source

    def test_round_trip_view_curate_view(self, manager, viewer):
        """View → Curate → View should leave only final View layers."""
        # Step 1: View patch A
        manager.add_labels(None, name="recon_A", context="step3/view")

        # Step 2: Enter curation (clears view, adds curation)
        manager.clear_context("step3/view")
        manager.add_labels(None, name="curation_3d_A", context="step3/curation")

        # Step 3: Back to View (clears curation, adds view)
        manager.clear_context("step3/curation")
        manager.clear_context("step3/view")
        manager.add_labels(None, name="recon_B", context="step3/view")

        assert "recon_A" not in viewer.layers
        assert "curation_3d_A" not in viewer.layers
        assert "recon_B" in viewer.layers
        assert len(list(viewer.layers)) == 1
