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

    def test_view_reconstruction_source_closes_annotator(
        self, monkeypatch, tmp_path
    ):
        """Viewing a reconstruction closes the 3D curation annotator dock,
        and does so before the curation layers are cleared.

        Drives the real View door (``_view_patch``, which the foot's View
        button reaches through ``_view_reconstruction``) on a bare
        controller with a fake dock lookup, so the test checks what
        happens rather than how the code is spelled.
        """
        import numpy as np
        import tifffile
        import napari_epicycle.annotator.launcher as launcher_mod
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )

        events = []
        looked_up = []

        class _FakeDock:
            closed = False

            def close(self):
                self.closed = True
                events.append("dock_close")

        dock = _FakeDock()

        def _fake_find(viewer, title):
            looked_up.append(title)
            return dock

        monkeypatch.setattr(launcher_mod, "_find_dock_by_title", _fake_find)

        class _RecordingLLM:
            def clear_context(self, context):
                events.append(f"clear_context:{context}")

            def register_existing(self, name, context):
                events.append(f"register:{name}:{context}")

            def clear_step(self, step):
                events.append(f"clear_step:{step}")

        labels_path = tmp_path / "patch_A_labels.tif"
        tifffile.imwrite(str(labels_path), np.zeros((2, 4, 4), np.uint16))

        c = FusionReconstructionController.__new__(
            FusionReconstructionController
        )
        c.viewer = FakeViewer()
        c.layer_lifecycle_manager = _RecordingLLM()
        c._guard_unsaved_changes = lambda: True
        c._resolve_patch_payload = lambda name: {
            "labels_path": str(labels_path),
            "source_stack_path": "",
        }
        c._clear_curation_layers = lambda: events.append("clear_curation")
        c._clear_qc_view_layers = lambda: events.append("clear_qc")
        c._append_log = lambda msg: None

        c._view_patch("patch_A.tif")

        assert looked_up == ["EpiCycle 3D Curation Annotator"]
        assert dock.closed, "View must close the 3D curation annotator dock"
        # The close comes first; the View path then does its own clears.
        assert events[0] == "dock_close", events
        assert "clear_context:step3/curation" in events
        # Close-only: the View path clears its own contexts, never the whole
        # of Step 3, and the new View layer is in the viewer at the end.
        assert "clear_step:step3" not in events, events
        assert "register:reconstruction_patch_A:step3/view" in events
        assert "reconstruction_patch_A" in c.viewer.layers

    def test_view_cancelled_at_unsaved_question_keeps_annotator(
        self, monkeypatch
    ):
        """When the user cancels the unsaved-edits question, View does
        nothing: the annotator dock is not even looked up, let alone closed,
        and no layers are cleared."""
        import napari_epicycle.annotator.launcher as launcher_mod
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )

        looked_up = []
        events = []
        monkeypatch.setattr(
            launcher_mod, "_find_dock_by_title",
            lambda viewer, title: looked_up.append(title),
        )

        c = FusionReconstructionController.__new__(
            FusionReconstructionController
        )
        c.viewer = FakeViewer()
        c.layer_lifecycle_manager = MagicMock()
        c._guard_unsaved_changes = lambda: False
        c._resolve_patch_payload = lambda name: events.append("payload")
        c._close_3d_curation_annotator = (
            lambda **kw: events.append("close"))
        c._clear_curation_layers = lambda: events.append("clear_curation")
        c._clear_qc_view_layers = lambda: events.append("clear_qc")
        c._append_log = lambda msg: None

        c._view_patch("patch_A.tif")

        assert looked_up == [], "a cancelled View must not touch the dock"
        assert events == [], events
        assert c.layer_lifecycle_manager.mock_calls == []
        assert len(c.viewer.layers) == 0

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
