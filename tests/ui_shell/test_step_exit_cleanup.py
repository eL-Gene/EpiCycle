"""Tests for Phase 5: Step exit cleanup — layers cleared on step switch.

Verifies that on_step_exit() hooks are called during step transitions
and that they clean up layers. Uses lightweight fakes — no Qt required.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


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
# Tests — Controller lifecycle hooks exist
# ---------------------------------------------------------------------------


class TestControllerHooksExist:
    """All controllers must have on_step_exit and on_step_enter methods."""

    def test_data_preparation_controller_has_hooks(self):
        import inspect
        from napari_epicycle.data_preparation.controller import (
            DataPreparationController,
        )
        assert hasattr(DataPreparationController, "on_step_exit")
        assert hasattr(DataPreparationController, "on_step_enter")

    def test_fusion_reconstruction_controller_has_hooks(self):
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        assert hasattr(FusionReconstructionController, "on_step_exit")
        assert hasattr(FusionReconstructionController, "on_step_enter")

    def test_hitl_controller_has_hooks(self):
        from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
            TrainingAnnotationController,
        )
        assert hasattr(TrainingAnnotationController, "on_step_exit")
        assert hasattr(TrainingAnnotationController, "on_step_enter")

    def test_segmentation_controller_has_hooks(self):
        from napari_epicycle.segmentation.controller import (
            SegmentStackController,
        )
        assert hasattr(SegmentStackController, "on_step_exit")
        assert hasattr(SegmentStackController, "on_step_enter")

    def test_tiled_segmentation_controller_has_hooks(self):
        from napari_epicycle.tiled_segmentation.controller import (
            LargeScaleSegmentationController,
        )
        assert hasattr(LargeScaleSegmentationController, "on_step_exit")
        assert hasattr(LargeScaleSegmentationController, "on_step_enter")

    def test_model_selection_controller_has_hooks(self):
        from napari_epicycle.model_selection.controller import (
            ModelSelectionController,
        )
        assert hasattr(ModelSelectionController, "on_step_exit")
        assert hasattr(ModelSelectionController, "on_step_enter")


# ---------------------------------------------------------------------------
# Tests — Step exit clears lifecycle manager layers
# ---------------------------------------------------------------------------


class TestStepExitClearsLayers:
    """on_step_exit() should clear layers from the lifecycle manager."""

    def test_step3_exit_clears_step3_managed_layers(self, manager, viewer):
        """Layers registered under step3/* should be removed on exit."""
        # Simulate layers created by View button (Phase 2)
        manager.add_labels(None, name="reconstruction_abc", context="step3/view")
        manager.add_image(None, name="stack_patch1", context="step3/view")
        assert len(list(viewer.layers)) == 2

        # Simulate on_step_exit calling clear_step
        manager.clear_step("step3")
        assert len(list(viewer.layers)) == 0

    def test_step3_exit_does_not_clear_step2(self, manager, viewer):
        """Clearing step3 should not affect step2 layers."""
        manager.add_labels(None, name="step2_labels", context="step2/patch")
        manager.add_labels(None, name="step3_labels", context="step3/view")

        manager.clear_step("step3")

        assert "step2_labels" in viewer.layers
        assert "step3_labels" not in viewer.layers


# ---------------------------------------------------------------------------
# Tests — Workflow shell _select_step fires hooks
# ---------------------------------------------------------------------------


class TestWorkflowShellStepHooks:
    """The workflow shell should call on_step_exit when switching steps."""

    def test_select_step_fires_exit_source(self):
        """_select_step calls previous controller's on_step_exit."""
        import inspect
        from napari_epicycle.ui_shell.workflow_shell import EpiCycleWorkflowShell
        source = inspect.getsource(EpiCycleWorkflowShell._select_step)
        assert "on_step_exit" in source, (
            "_select_step must call previous controller's on_step_exit()"
        )

    def test_select_step_fires_enter_source(self):
        """_select_step calls new controller's on_step_enter."""
        import inspect
        from napari_epicycle.ui_shell.workflow_shell import EpiCycleWorkflowShell
        source = inspect.getsource(EpiCycleWorkflowShell._select_step)
        assert "on_step_enter" in source, (
            "_select_step must call new controller's on_step_enter()"
        )

    def test_get_step_controller_returns_controller(self):
        """_get_step_controller accesses widget.controller."""
        from napari_epicycle.ui_shell.workflow_shell import EpiCycleWorkflowShell
        shell = EpiCycleWorkflowShell.__new__(EpiCycleWorkflowShell)
        shell._step_widgets = {}
        mock_widget = MagicMock()
        mock_widget.controller = MagicMock()
        shell._step_widgets["test_step"] = mock_widget
        ctrl = shell._get_step_controller("test_step")
        assert ctrl is mock_widget.controller

    def test_get_step_controller_returns_none_for_missing(self):
        """_get_step_controller returns None for unknown step."""
        from napari_epicycle.ui_shell.workflow_shell import EpiCycleWorkflowShell
        shell = EpiCycleWorkflowShell.__new__(EpiCycleWorkflowShell)
        shell._step_widgets = {}
        assert shell._get_step_controller("nonexistent") is None


# ---------------------------------------------------------------------------
# Tests — FusionReconstructionController on_step_exit behavior
# ---------------------------------------------------------------------------


class TestFusionControllerOnStepExit:
    """FusionReconstructionController.on_step_exit cleans up Step 3.

    These drive the real ``on_step_exit`` on a bare controller: the two
    refusal checks are stubbed to let the user leave, the dock lookup is
    faked, and the layer clears and the lifecycle manager record what they
    were asked to do.
    """

    @pytest.fixture
    def exited(self, monkeypatch):
        import napari_epicycle.annotator.launcher as launcher_mod
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )

        calls = []
        looked_up = []

        class _FakeDock:
            closed = False

            def close(self):
                self.closed = True
                calls.append("dock_close")

        dock = _FakeDock()

        def _fake_find(viewer, title):
            looked_up.append(title)
            return dock

        monkeypatch.setattr(launcher_mod, "_find_dock_by_title", _fake_find)

        class _RecordingLLM:
            def __init__(self):
                self.cleared = []

            def clear_step(self, step):
                self.cleared.append(step)
                calls.append(f"clear_step:{step}")

        c = FusionReconstructionController.__new__(
            FusionReconstructionController
        )
        c.viewer = MagicMock()
        c.layer_lifecycle_manager = _RecordingLLM()
        c._staged_annotator_work = lambda: False
        c._guard_unsaved_changes = lambda: True
        c._reset_qc_plot_area = lambda *a, **k: None
        c._stop_step3_thumbs = lambda: None
        c._stop_tier_counts = lambda: None
        c._close_provenance_windows = lambda: None
        c._close_prediction_windows = lambda: None
        c._clear_qc_view_layers = lambda: calls.append("clear_qc")
        c._clear_curation_layers = lambda: calls.append("clear_curation")

        result = c.on_step_exit()
        assert result is not False, "the user should be let out of Step 3"
        return {"ctrl": c, "calls": calls, "dock": dock,
                "looked_up": looked_up}

    def test_on_step_exit_calls_clear_methods(self, exited):
        """on_step_exit runs both layer clears, QC view first, after the
        dock is closed.  (The population-grid clear went with population
        mode in May 2026.)"""
        calls = exited["calls"]
        assert [x for x in calls if x.startswith("clear_")
                and not x.startswith("clear_step")] == [
            "clear_qc", "clear_curation",
        ]
        assert calls.index("dock_close") < calls.index("clear_qc"), calls

    def test_on_step_exit_calls_clear_step_on_manager(self, exited):
        """on_step_exit asks the layer lifecycle manager to clear Step 3,
        once, after the old clears."""
        assert exited["ctrl"].layer_lifecycle_manager.cleared == ["step3"]
        assert exited["calls"][-1] == "clear_step:step3", exited["calls"]

    def test_on_step_exit_closes_annotator_dock(self, exited):
        """on_step_exit closes the 3D curation annotator dock."""
        assert exited["looked_up"] == ["EpiCycle 3D Curation Annotator"]
        assert exited["dock"].closed


# ---------------------------------------------------------------------------
# Tests — HITL Controller on_step_exit behavior
# ---------------------------------------------------------------------------


class TestHITLControllerOnStepExit:
    """TrainingAnnotationController.on_step_exit calls cleanup methods."""

    def test_on_step_exit_closes_annotator_dock(self):
        """on_step_exit should close the HITL 2D annotator dock."""
        import inspect
        from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
            TrainingAnnotationController,
        )
        source = inspect.getsource(TrainingAnnotationController.on_step_exit)
        assert "EpiCycle 2D Training Annotator" in source

    def test_on_step_exit_cleans_prompt_layers(self):
        """on_step_exit should call cleanup_prompt_layers."""
        import inspect
        from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
            TrainingAnnotationController,
        )
        source = inspect.getsource(TrainingAnnotationController.on_step_exit)
        assert "cleanup_prompt_layers" in source
