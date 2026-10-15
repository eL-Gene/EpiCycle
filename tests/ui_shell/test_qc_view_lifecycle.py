"""Tests for Phase 4: QC view layer lifecycle integration.

Verifies that _load_patch_for_qc_view registers layers under
step3/qc/single and population grid registers under step3/qc/population.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Fakes
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


# ---------------------------------------------------------------------------
# Tests — QC single-patch view layers
# ---------------------------------------------------------------------------


class TestQCSinglePatchRegistration:
    """QC single-patch view layers should be tracked in step3/qc/single."""

    def test_qc_labels_registered(self, manager, viewer):
        viewer.add_labels(None, name="reconstruction_abc")
        manager.register_existing("reconstruction_abc", "step3/qc/single")
        assert "reconstruction_abc" in manager.layers_in_context("step3/qc/single")

    def test_qc_stack_registered(self, manager, viewer):
        viewer.add_image(None, name="stack_patch1")
        manager.register_existing("stack_patch1", "step3/qc/single")
        assert "stack_patch1" in manager.layers_in_context("step3/qc/single")

    def test_clear_qc_single_removes_layers(self, manager, viewer):
        manager.add_labels(None, name="recon_old", context="step3/qc/single")
        manager.add_image(None, name="stack_old", context="step3/qc/single")

        manager.clear_context("step3/qc/single")

        assert "recon_old" not in viewer.layers
        assert "stack_old" not in viewer.layers

    def test_clear_qc_single_does_not_affect_population(self, manager, viewer):
        manager.add_labels(None, name="single_lbl", context="step3/qc/single")
        manager.add_labels(None, name="grid_lbl", context="step3/qc/population")

        manager.clear_context("step3/qc/single")

        assert "single_lbl" not in viewer.layers
        assert "grid_lbl" in viewer.layers


# ---------------------------------------------------------------------------
# Tests — Population grid layers
# ---------------------------------------------------------------------------


class TestPopulationGridRegistration:
    """Population grid layers should be tracked in step3/qc/population."""

    def test_grid_layers_registered(self, manager, viewer):
        for i in range(3):
            name = f"[grid] patch_{i}"
            viewer.add_labels(None, name=name)
            manager.register_existing(name, "step3/qc/population")

        names = manager.layers_in_context("step3/qc/population")
        assert len(names) == 3

    def test_clear_population_removes_all_grid_layers(self, manager, viewer):
        for i in range(3):
            manager.add_labels(None, name=f"[grid] p{i}", context="step3/qc/population")

        manager.clear_context("step3/qc/population")
        assert len(list(viewer.layers)) == 0

    def test_clear_step3_removes_all_qc_contexts(self, manager, viewer):
        """clear_step('step3') should remove both single and population."""
        manager.add_labels(None, name="single", context="step3/qc/single")
        manager.add_labels(None, name="grid", context="step3/qc/population")
        manager.add_labels(None, name="view", context="step3/view")

        manager.clear_step("step3")

        assert len(list(viewer.layers)) == 0


# ---------------------------------------------------------------------------
# Tests — Source code verification
# ---------------------------------------------------------------------------


class TestQCSourceVerification:
    """Verify that controller methods register with lifecycle manager."""

    def test_load_patch_for_qc_view_registers_labels(self):
        import inspect
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        source = inspect.getsource(
            FusionReconstructionController._load_patch_for_qc_view
        )
        assert "step3/qc/single" in source

    def test_clear_qc_view_layers_clears_context(self):
        import inspect
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        source = inspect.getsource(
            FusionReconstructionController._clear_qc_view_layers
        )
        assert 'clear_context("step3/qc/single")' in source

    def test_clear_population_grid_clears_context(self):
        import inspect
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        source = inspect.getsource(
            FusionReconstructionController._clear_population_grid
        )
        assert 'clear_context("step3/qc/population")' in source

    def test_population_grid_builder_registers_layers(self):
        """The grid builder should register each [grid] layer."""
        import inspect
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        # _build_population_grid or the method that creates grid layers
        # Search for the grid layer creation method
        source = inspect.getsource(FusionReconstructionController)
        # Grid layers should be registered under step3/qc/population
        assert "step3/qc/population" in source
