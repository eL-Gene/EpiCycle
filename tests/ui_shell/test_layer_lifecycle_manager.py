"""Tests for the global LayerLifecycleManager.

Uses lightweight fakes — no Qt or napari required.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeLayer:
    """Minimal stand-in for a napari layer."""

    def __init__(self, name: str, visible: bool = True):
        self.name = name
        self.visible = visible


class FakeLayerList:
    """Minimal stand-in for napari viewer.layers with events."""

    def __init__(self):
        self._layers: dict[str, FakeLayer] = {}
        self.events = MagicMock()

    def __contains__(self, name: str) -> bool:
        return name in self._layers

    def __getitem__(self, name: str) -> FakeLayer:
        return self._layers[name]

    def __iter__(self):
        return iter(self._layers.values())

    def add(self, layer: FakeLayer) -> FakeLayer:
        self._layers[layer.name] = layer
        return layer

    def remove(self, layer: FakeLayer) -> None:
        self._layers.pop(layer.name, None)


class FakeViewer:
    """Minimal napari viewer stand-in."""

    def __init__(self):
        self.layers = FakeLayerList()
        self._added_images = []
        self._added_labels = []

    def add_image(self, data, *, name, **kwargs):
        layer = FakeLayer(name)
        self.layers.add(layer)
        self._added_images.append(name)
        return layer

    def add_labels(self, data, *, name, **kwargs):
        layer = FakeLayer(name)
        self.layers.add(layer)
        self._added_labels.append(name)
        return layer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
# Tests — Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    """Layer registration and context tracking."""

    def test_add_image_registers_under_context(self, manager, viewer):
        manager.add_image(None, name="stack_patch1", context="step3/view")
        assert "stack_patch1" in manager.layers_in_context("step3/view")
        assert "stack_patch1" in viewer.layers

    def test_add_labels_registers_under_context(self, manager, viewer):
        manager.add_labels(None, name="recon_001", context="step3/view")
        assert "recon_001" in manager.layers_in_context("step3/view")
        assert "recon_001" in viewer.layers

    def test_register_existing_tracks_preexisting_layer(self, manager, viewer):
        viewer.add_labels(None, name="curation_3d_001")
        manager.register_existing("curation_3d_001", "step3/curation")
        assert "curation_3d_001" in manager.layers_in_context("step3/curation")

    def test_register_existing_ignores_missing_layer(self, manager, viewer):
        manager.register_existing("nonexistent", "step3/curation")
        assert manager.layers_in_context("step3/curation") == []

    def test_context_of_returns_correct_context(self, manager):
        manager.add_labels(None, name="labels_A", context="step3/curation")
        assert manager.context_of("labels_A") == "step3/curation"

    def test_context_of_returns_none_for_unknown(self, manager):
        assert manager.context_of("unknown_layer") is None

    def test_is_managed_true_for_registered(self, manager):
        manager.add_image(None, name="img1", context="step2/patch")
        assert manager.is_managed("img1") is True

    def test_is_managed_false_for_unregistered(self, manager):
        assert manager.is_managed("random_layer") is False

    def test_multiple_layers_same_context(self, manager):
        manager.add_image(None, name="stack", context="step3/view")
        manager.add_labels(None, name="recon", context="step3/view")
        names = manager.layers_in_context("step3/view")
        assert "stack" in names
        assert "recon" in names
        assert len(names) == 2

    def test_register_existing_is_idempotent(self, manager, viewer):
        """Calling register_existing twice must not create duplicates."""
        viewer.add_labels(None, name="work_labels")
        manager.register_existing("work_labels", "step2/patch")
        manager.register_existing("work_labels", "step2/patch")
        names = manager.layers_in_context("step2/patch")
        assert names.count("work_labels") == 1


# ---------------------------------------------------------------------------
# Tests — Rename
# ---------------------------------------------------------------------------


class TestRename:
    """Registry update on layer rename."""

    def test_rename_updates_registry(self, manager):
        manager.add_labels(None, name="reconstruction_001", context="step3/view")
        manager.rename("reconstruction_001", "curation_3d_001")
        assert manager.context_of("reconstruction_001") is None
        assert manager.context_of("curation_3d_001") == "step3/view"

    def test_rename_with_new_context(self, manager):
        manager.add_labels(None, name="recon_001", context="step3/view")
        manager.rename("recon_001", "curation_3d_001", new_context="step3/curation")
        assert manager.context_of("curation_3d_001") == "step3/curation"
        assert manager.layers_in_context("step3/view") == []

    def test_rename_unknown_layer_is_noop(self, manager):
        manager.rename("nonexistent", "also_nonexistent")
        assert manager.context_of("also_nonexistent") is None


# ---------------------------------------------------------------------------
# Tests — Cleanup
# ---------------------------------------------------------------------------


class TestCleanup:
    """Context and step cleanup."""

    def test_clear_context_removes_layers_from_viewer(self, manager, viewer):
        manager.add_labels(None, name="recon_001", context="step3/view")
        manager.add_image(None, name="stack_001", context="step3/view")
        manager.clear_context("step3/view")
        assert "recon_001" not in viewer.layers
        assert "stack_001" not in viewer.layers

    def test_clear_context_empties_registry(self, manager):
        manager.add_labels(None, name="recon_001", context="step3/view")
        manager.clear_context("step3/view")
        assert manager.layers_in_context("step3/view") == []
        assert manager.context_of("recon_001") is None

    def test_clear_context_noop_for_unknown_context(self, manager):
        """Clearing a nonexistent context should not raise."""
        manager.clear_context("step99/nonexistent")

    def test_forget_context_removes_from_registry_not_viewer(self, manager, viewer):
        """forget_context clears registry but leaves layers in viewer."""
        manager.add_labels(None, name="cur_lbl", context="step3/curation")
        manager.add_image(None, name="cur_img", context="step3/curation")

        manager.forget_context("step3/curation")

        # Registry is empty
        assert manager.layers_in_context("step3/curation") == []
        assert manager.context_of("cur_lbl") is None
        # But layers are still in the viewer
        assert "cur_lbl" in viewer.layers
        assert "cur_img" in viewer.layers

    def test_forget_context_noop_for_unknown(self, manager):
        """forget_context on nonexistent context should not raise."""
        manager.forget_context("step99/nonexistent")

    def test_clear_step_removes_all_sub_contexts(self, manager, viewer):
        manager.add_labels(None, name="view_lbl", context="step3/view")
        manager.add_labels(None, name="cur_lbl", context="step3/curation")
        manager.add_image(None, name="qc_img", context="step3/qc/single")
        manager.clear_step("step3")
        assert "view_lbl" not in viewer.layers
        assert "cur_lbl" not in viewer.layers
        assert "qc_img" not in viewer.layers
        assert manager.layers_in_context("step3/view") == []
        assert manager.layers_in_context("step3/curation") == []

    def test_clear_step_does_not_affect_other_steps(self, manager, viewer):
        manager.add_labels(None, name="step2_lbl", context="step2/patch")
        manager.add_labels(None, name="step3_lbl", context="step3/view")
        manager.clear_step("step3")
        assert "step2_lbl" in viewer.layers
        assert manager.context_of("step2_lbl") == "step2/patch"

    def test_clear_all_removes_everything(self, manager, viewer):
        manager.add_labels(None, name="a", context="step2/patch")
        manager.add_labels(None, name="b", context="step3/view")
        manager.add_image(None, name="c", context="step5/segmentation")
        manager.clear_all()
        assert len(list(viewer.layers)) == 0

    def test_hide_before_remove_pattern(self, manager, viewer):
        """Layers must be hidden before removal (vispy safety)."""
        layer = manager.add_labels(None, name="test_lbl", context="step3/view")
        assert layer.visible is True
        manager.clear_context("step3/view")
        # After removal, the layer object should have visible=False
        assert layer.visible is False

    def test_clear_context_tolerates_already_removed_layer(self, manager, viewer):
        """If a layer was removed externally before clear, don't crash."""
        manager.add_labels(None, name="ghost", context="step3/view")
        # Manually remove it from the viewer (simulates external delete)
        layer = viewer.layers["ghost"]
        viewer.layers.remove(layer)
        # This should not raise
        manager.clear_context("step3/view")


# ---------------------------------------------------------------------------
# Tests — External removal callbacks
# ---------------------------------------------------------------------------


class TestExternalRemoval:
    """Trash can deletion detection."""

    def test_external_removal_fires_callback(self, manager, viewer):
        manager.add_labels(None, name="cur_lbl", context="step3/curation")
        removed = []
        manager.on_external_removal(lambda name, ctx: removed.append((name, ctx)))

        # Simulate external removal event
        event = MagicMock()
        event.value = viewer.layers["cur_lbl"]
        manager._on_layer_removed(event)

        assert removed == [("cur_lbl", "step3/curation")]

    def test_internal_removal_does_not_fire_callback(self, manager, viewer):
        manager.add_labels(None, name="cur_lbl", context="step3/curation")
        removed = []
        manager.on_external_removal(lambda name, ctx: removed.append((name, ctx)))

        # Internal removal via clear_context
        manager.clear_context("step3/curation")
        assert removed == []

    def test_unmanaged_layer_removal_ignored(self, manager, viewer):
        """Removing a layer not tracked by the manager should not fire callbacks."""
        viewer.add_labels(None, name="unmanaged")
        removed = []
        manager.on_external_removal(lambda name, ctx: removed.append((name, ctx)))

        event = MagicMock()
        event.value = viewer.layers["unmanaged"]
        manager._on_layer_removed(event)

        assert removed == []

    def test_callback_exception_does_not_crash(self, manager, viewer):
        """A failing callback should not prevent other callbacks from running."""
        manager.add_labels(None, name="lbl", context="step3/view")
        results = []
        manager.on_external_removal(lambda n, c: (_ for _ in ()).throw(ValueError("boom")))
        manager.on_external_removal(lambda n, c: results.append("ok"))

        event = MagicMock()
        event.value = viewer.layers["lbl"]
        manager._on_layer_removed(event)
        assert results == ["ok"]

    def test_multiple_callbacks(self, manager, viewer):
        manager.add_labels(None, name="lbl", context="step3/curation")
        results = []
        manager.on_external_removal(lambda n, c: results.append("cb1"))
        manager.on_external_removal(lambda n, c: results.append("cb2"))

        event = MagicMock()
        event.value = viewer.layers["lbl"]
        manager._on_layer_removed(event)
        assert results == ["cb1", "cb2"]


# ---------------------------------------------------------------------------
# Tests — Shutdown
# ---------------------------------------------------------------------------


class TestShutdown:
    """Manager cleanup."""

    def test_shutdown_disconnects_events(self, viewer):
        from napari_epicycle.ui_shell.layer_lifecycle_manager import (
            LayerLifecycleManager,
        )
        mgr = LayerLifecycleManager(viewer)
        # Should not raise even if disconnect fails
        mgr.shutdown()


# ---------------------------------------------------------------------------
# Tests — SubviewerMirrorManager backward compat
# ---------------------------------------------------------------------------


class TestSubviewerMirrorManagerRename:
    """The old LayerLifecycleManager import path still works via shim."""

    def test_old_import_path_still_works(self):
        from packages.epicycle_annotator.layer_lifecycle import (
            LayerLifecycleManager,
        )
        from packages.epicycle_annotator.subviewer_mirror import (
            SubviewerMirrorManager,
        )
        assert LayerLifecycleManager is SubviewerMirrorManager

    def test_new_import_via_package_init(self):
        from packages.epicycle_annotator import SubviewerMirrorManager
        assert SubviewerMirrorManager is not None
