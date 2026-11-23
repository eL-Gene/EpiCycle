"""Tests for Phase 3: Curation layer lifecycle integration.

Verifies that curation layers are registered with the global
LayerLifecycleManager, trash-can deletion is detected, and patch
switching clears old curation layers. Uses lightweight fakes.
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
# Tests — Curation layer registration
# ---------------------------------------------------------------------------


class TestCurationLayerRegistration:
    """Curation layers should be registered under step3/curation context."""

    def test_curation_labels_registered(self, manager, viewer):
        viewer.add_labels(None, name="curation_3d_abc123")
        manager.register_existing("curation_3d_abc123", "step3/curation")
        assert "curation_3d_abc123" in manager.layers_in_context("step3/curation")

    def test_curation_stack_registered(self, manager, viewer):
        viewer.add_image(None, name="stack_patch1")
        manager.register_existing("stack_patch1", "step3/curation")
        assert "stack_patch1" in manager.layers_in_context("step3/curation")

    def test_view_to_curation_rename(self, manager, viewer):
        """When a reconstruction_ layer is renamed to curation_3d_, the
        manager should update from step3/view to step3/curation."""
        viewer.add_labels(None, name="reconstruction_abc")
        manager.register_existing("reconstruction_abc", "step3/view")

        # Simulate the rename that happens in _open_3d_curation_annotator
        manager.rename("reconstruction_abc", "curation_3d_abc", "step3/curation")

        assert manager.context_of("reconstruction_abc") is None
        assert manager.context_of("curation_3d_abc") == "step3/curation"
        assert manager.layers_in_context("step3/view") == []


# ---------------------------------------------------------------------------
# Tests — Trash can deletion handling
# ---------------------------------------------------------------------------


class TestTrashCanDeletion:
    """External removal of curation layers should fire callbacks."""

    def test_external_removal_fires_callback_for_curation_layer(self, manager, viewer):
        manager.add_labels(None, name="curation_3d_abc", context="step3/curation")
        removed = []
        manager.on_external_removal(lambda n, c: removed.append((n, c)))

        event = MagicMock()
        event.value = viewer.layers["curation_3d_abc"]
        manager._on_layer_removed(event)

        assert removed == [("curation_3d_abc", "step3/curation")]

    def test_controller_handler_closes_annotator_on_curation_delete(
        self, monkeypatch
    ):
        """_on_layer_externally_removed should close the annotator dock
        when a step3/curation layer is deleted.

        Calls the real handler on a bare controller with a fake dock lookup
        and checks the dock was closed.  The layers are already going away
        with the deletion, so the handler must not clear any layers itself.
        """
        import napari_epicycle.annotator.launcher as launcher_mod
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )

        looked_up = []

        class _FakeDock:
            closed = False

            def close(self):
                self.closed = True

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

        cleared = []
        logs = []
        ctrl = FusionReconstructionController.__new__(
            FusionReconstructionController
        )
        ctrl.viewer = MagicMock()
        ctrl.layer_lifecycle_manager = _RecordingLLM()
        ctrl._clear_qc_view_layers = lambda: cleared.append("qc")
        ctrl._clear_curation_layers = lambda: cleared.append("curation")
        ctrl._append_log = logs.append

        ctrl._on_layer_externally_removed("curation_3d_x", "step3/curation")

        assert looked_up == ["EpiCycle 3D Curation Annotator"]
        assert dock.closed, "deleting a curation layer must close the dock"
        assert cleared == [], "close-only: no layer clears from this handler"
        assert ctrl.layer_lifecycle_manager.cleared == []
        assert any("curation_3d_x" in m for m in logs), logs

    def test_controller_handler_ignores_non_curation_context(self):
        """_on_layer_externally_removed should ignore non-curation layers."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        ctrl = FusionReconstructionController.__new__(
            FusionReconstructionController
        )
        ctrl.viewer = MagicMock()
        ctrl.ui = MagicMock()
        # Should not raise or attempt dock close for step3/view
        ctrl._on_layer_externally_removed("recon_abc", "step3/view")


# ---------------------------------------------------------------------------
# Tests — Patch switching clears curation context
# ---------------------------------------------------------------------------


class TestPatchSwitchClearsCuration:
    """Patch switching should clear step3/curation from the manager."""

    def test_clear_curation_removes_curation_layers(self, manager, viewer):
        manager.add_labels(None, name="curation_3d_old", context="step3/curation")
        manager.add_image(None, name="stack_old", context="step3/curation")

        manager.clear_context("step3/curation")

        assert "curation_3d_old" not in viewer.layers
        assert "stack_old" not in viewer.layers
        assert manager.layers_in_context("step3/curation") == []

    def test_clear_curation_does_not_affect_view(self, manager, viewer):
        manager.add_labels(None, name="recon_view", context="step3/view")
        manager.add_labels(None, name="curation_3d_cur", context="step3/curation")

        manager.clear_context("step3/curation")

        assert "recon_view" in viewer.layers
        assert "curation_3d_cur" not in viewer.layers

    def test_patch_switch_source_has_clear_context(self):
        """_switch_to_curation_patch should call clear_context('step3/curation')."""
        import inspect
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        source = inspect.getsource(
            FusionReconstructionController._switch_to_curation_patch
        )
        assert 'forget_context("step3/curation")' in source


# ---------------------------------------------------------------------------
# Tests — Lazy callback registration
# ---------------------------------------------------------------------------


class TestLazyCallbackRegistration:
    """External removal callback should be registered once, lazily."""

    def test_ensure_llm_removal_callback_registers_once(self, manager):
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        ctrl = FusionReconstructionController.__new__(
            FusionReconstructionController
        )
        ctrl.layer_lifecycle_manager = manager
        ctrl._llm_removal_cb_installed = False
        ctrl.viewer = MagicMock()
        ctrl.ui = MagicMock()

        ctrl._ensure_llm_removal_callback()
        assert ctrl._llm_removal_cb_installed is True
        assert len(manager._removal_callbacks) == 1

        # Second call should not add another callback
        ctrl._ensure_llm_removal_callback()
        assert len(manager._removal_callbacks) == 1

    def test_ensure_llm_noop_without_manager(self):
        """Should not crash if layer_lifecycle_manager is not set."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        ctrl = FusionReconstructionController.__new__(
            FusionReconstructionController
        )
        ctrl._llm_removal_cb_installed = False
        # No layer_lifecycle_manager attribute
        ctrl._ensure_llm_removal_callback()
        assert ctrl._llm_removal_cb_installed is False

    def test_open_annotator_source_calls_ensure(self):
        """_open_3d_curation_annotator should call _ensure_llm_removal_callback."""
        import inspect
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        source = inspect.getsource(
            FusionReconstructionController._open_3d_curation_annotator
        )
        assert "_ensure_llm_removal_callback" in source
