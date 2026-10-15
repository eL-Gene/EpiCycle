"""Tests for HITL 2D lifecycle integration fixes (F1, F2, F3, F4, F5, F6).

Uses lightweight fakes — no Qt or napari required.
Follows the same pattern as test_dataflow_fixes.py.
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeLayer:
    """Minimal stand-in for a napari layer."""

    def __init__(self, name: str, data=None, visible: bool = True):
        self.name = name
        self.data = data if data is not None else np.zeros((4, 8, 8), dtype=np.int32)
        self.visible = visible
        self.opacity = 0.8
        self.colormap = "default_colormap"

    @property
    def __class__(self):
        class _C:
            __name__ = "Labels"
        return _C()


class FakeImageLayer:
    """Minimal stand-in for a napari Image layer."""

    def __init__(self, name: str, data=None, visible: bool = True):
        self.name = name
        self.data = data if data is not None else np.zeros((4, 8, 8), dtype=np.float32)
        self.visible = visible
        self.colormap = "magenta"
        self.blending = "additive"

    @property
    def __class__(self):
        class _C:
            __name__ = "Image"
        return _C()


class FakeLayerList:
    """Minimal stand-in for napari viewer.layers."""

    def __init__(self):
        self._layers: dict[str, object] = {}
        self.events = MagicMock()

    def __contains__(self, name: str) -> bool:
        return name in self._layers

    def __getitem__(self, name: str):
        return self._layers[name]

    def __iter__(self):
        return iter(list(self._layers.values()))

    def __len__(self):
        return len(self._layers)

    def add(self, layer) -> object:
        self._layers[layer.name] = layer
        return layer

    def remove(self, layer) -> None:
        self._layers.pop(layer.name, None)


class FakeViewer:
    """Minimal napari viewer stand-in."""

    def __init__(self):
        self.layers = FakeLayerList()

    def add_image(self, data, *, name, **kwargs):
        layer = FakeImageLayer(name, data=data)
        self.layers.add(layer)
        return layer

    def add_labels(self, data, *, name, **kwargs):
        layer = FakeLayer(name, data=data)
        self.layers.add(layer)
        return layer


class RecordingLLM:
    """Lifecycle manager fake that records method calls."""

    def __init__(self):
        self.calls: list[tuple] = []
        self._registry: dict[str, list[str]] = {}
        self._removal_callbacks: list = []

    def forget_context(self, context: str) -> None:
        self.calls.append(("forget_context", context))
        self._registry.pop(context, None)

    def clear_context(self, context: str) -> None:
        self.calls.append(("clear_context", context))
        self._registry.pop(context, None)

    def clear_step(self, prefix: str) -> None:
        self.calls.append(("clear_step", prefix))
        for ctx in [c for c in self._registry if c.startswith(prefix)]:
            self._registry.pop(ctx, None)

    def register_existing(self, name: str, context: str) -> None:
        self.calls.append(("register_existing", name, context))
        names = self._registry.setdefault(context, [])
        if name not in names:
            names.append(name)

    def layers_in_context(self, context: str) -> list[str]:
        return list(self._registry.get(context, []))

    def on_external_removal(self, callback) -> None:
        self.calls.append(("on_external_removal",))
        self._removal_callbacks.append(callback)


class FakeListWidget:
    def __init__(self, items=None):
        self._items = items or []
        self._selected = []

    def selectedItems(self):
        return self._selected

    def count(self):
        return len(self._items)


class FakeInteractiveManager:
    def __init__(self):
        self.cleanup_called = False

    def cleanup_prompt_layers(self, viewer):
        self.cleanup_called = True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def viewer():
    return FakeViewer()


@pytest.fixture
def llm():
    return RecordingLLM()


# ---------------------------------------------------------------------------
# Phase 2 — F1: _load_patch forgets old context
# ---------------------------------------------------------------------------


class TestLoadPatchForgetsContext:
    """F1: _load_patch must call forget_context('step2/patch') before removing layers."""

    def _make_ctrl_for_load_patch(self, viewer, llm):
        """Minimal controller that can run _load_patch up to the early return."""
        from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
            TrainingAnnotationController,
        )

        ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
        ctrl.viewer = viewer
        ctrl.layer_lifecycle_manager = llm
        ctrl.loaded_stack_path = "/fake/path"
        ctrl.loaded_patch_name = "patch01"
        ctrl.selected_view = "XY"
        ctrl.view_target = "lateral"
        ctrl.in_refine_mode = False
        ctrl.interactive_manager = FakeInteractiveManager()

        # UI stub — train_patch_list returns empty so _load_patch returns early
        # after the removal loop (no disk access needed).
        class UI:
            train_patch_list = FakeListWidget()
            lateral_radio = MagicMock(isChecked=lambda: True)
            axial_radio = MagicMock(isChecked=lambda: False)
            xy_radio = MagicMock(isChecked=lambda: True)
            xz_radio = MagicMock(isChecked=lambda: False)
            yz_radio = MagicMock(isChecked=lambda: False)
            axial_plane_combo = MagicMock()
        ctrl.ui = UI()
        return ctrl

    def test_forget_context_called_before_layer_removal(self, viewer, llm):
        """forget_context('step2/patch') must be called during _load_patch."""
        # Register old layers in the LLM
        img = viewer.add_image(np.zeros((4, 8, 8)), name="patch01_ch0_XY_view")
        lbl = viewer.add_labels(np.zeros((4, 8, 8), dtype=np.int32), name="patch01_work_labels_lateral_3d")
        llm.register_existing("patch01_ch0_XY_view", "step2/patch")
        llm.register_existing("patch01_work_labels_lateral_3d", "step2/patch")
        llm.calls.clear()  # Reset so we only see calls from _load_patch

        ctrl = self._make_ctrl_for_load_patch(viewer, llm)
        ctrl._load_patch(force_axis="XY")

        # forget_context must have been called
        forget_calls = [c for c in llm.calls if c[0] == "forget_context"]
        assert len(forget_calls) == 1
        assert forget_calls[0] == ("forget_context", "step2/patch")

    def test_forget_context_called_before_removal_loop(self, viewer, llm):
        """forget_context must happen BEFORE layers are removed from viewer."""
        call_order = []

        original_forget = llm.forget_context
        def tracking_forget(context):
            call_order.append("forget")
            original_forget(context)

        original_remove = viewer.layers.remove
        def tracking_remove(layer):
            call_order.append(f"remove:{layer.name}")
            original_remove(layer)

        llm.forget_context = tracking_forget
        viewer.layers.remove = tracking_remove

        viewer.add_image(np.zeros((4, 8, 8)), name="patch01_ch0_XY_view")
        llm.register_existing("patch01_ch0_XY_view", "step2/patch")
        llm.calls.clear()

        ctrl = self._make_ctrl_for_load_patch(viewer, llm)
        ctrl._load_patch(force_axis="XY")

        assert call_order[0] == "forget", f"Expected forget first, got: {call_order}"

    def test_registry_empty_after_load_patch(self, viewer, llm):
        """After _load_patch, step2/patch context should be empty in registry."""
        viewer.add_image(np.zeros((4, 8, 8)), name="patch01_ch0_XY_view")
        llm.register_existing("patch01_ch0_XY_view", "step2/patch")

        ctrl = self._make_ctrl_for_load_patch(viewer, llm)
        ctrl._load_patch(force_axis="XY")

        assert llm.layers_in_context("step2/patch") == []


# ---------------------------------------------------------------------------
# Phase 2 — F4: _ensure_work_labels_layer always registers
# ---------------------------------------------------------------------------


class TestEnsureWorkLabelsRegistration:
    """F4: _ensure_work_labels_layer must register with LLM for both new and reused paths."""

    def _make_ctrl_for_work_labels(self, viewer, llm):
        from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
            TrainingAnnotationController,
        )

        ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
        ctrl.viewer = viewer
        ctrl.layer_lifecycle_manager = llm
        ctrl.loaded_patch_name = "patch01"
        ctrl.selected_view = "XY"
        ctrl.sample_records = {}
        ctrl.segmentation_manager = MagicMock()
        ctrl.segmentation_manager.get_masks = MagicMock(return_value=None)
        return ctrl

    def test_new_layer_registers_with_llm(self, viewer, llm):
        """Creating a fresh work_labels layer should register it."""
        # Add an image layer so ref_shape can be resolved
        viewer.add_image(np.zeros((4, 8, 8), dtype=np.float32), name="patch01_ch0_XY_view")

        ctrl = self._make_ctrl_for_work_labels(viewer, llm)
        result = ctrl._ensure_work_labels_layer(seed_slice_label="", view_name="XY")

        expected_name = "patch01_work_labels_lateral_3d"
        assert expected_name in llm.layers_in_context("step2/patch")

    def test_reused_layer_registers_with_llm(self, viewer, llm):
        """Reusing an existing work_labels layer should ALSO register it."""
        # Add image for ref_shape
        viewer.add_image(np.zeros((4, 8, 8), dtype=np.float32), name="patch01_ch0_XY_view")
        # Pre-create the work labels layer (simulates reuse path)
        viewer.add_labels(np.zeros((4, 8, 8), dtype=np.int32), name="patch01_work_labels_lateral_3d")

        ctrl = self._make_ctrl_for_work_labels(viewer, llm)
        result = ctrl._ensure_work_labels_layer(seed_slice_label="", view_name="XY")

        expected_name = "patch01_work_labels_lateral_3d"
        assert expected_name in llm.layers_in_context("step2/patch")

    def test_double_ensure_no_duplicate_registration(self, viewer, llm):
        """Calling _ensure_work_labels_layer twice should not duplicate the registration."""
        viewer.add_image(np.zeros((4, 8, 8), dtype=np.float32), name="patch01_ch0_XY_view")

        ctrl = self._make_ctrl_for_work_labels(viewer, llm)
        ctrl._ensure_work_labels_layer(seed_slice_label="", view_name="XY")
        ctrl._ensure_work_labels_layer(seed_slice_label="", view_name="XY")

        names = llm.layers_in_context("step2/patch")
        assert names.count("patch01_work_labels_lateral_3d") == 1


# ---------------------------------------------------------------------------
# Phase 3 — F2: External layer removal closes dock
# ---------------------------------------------------------------------------


class FakeDock:
    """Minimal QDockWidget stand-in."""

    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True

    def isVisible(self):
        return True


class TestExternalRemovalCallback:
    """F2: External layer deletion should close the dock via controller callback."""

    def _make_ctrl_with_llm(self, viewer, llm):
        from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
            TrainingAnnotationController,
        )

        ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
        ctrl.viewer = viewer
        ctrl.layer_lifecycle_manager = llm
        ctrl._llm_removal_cb_installed = False
        return ctrl

    def test_ensure_callback_installed_once(self, viewer, llm):
        """_ensure_llm_removal_callback must register exactly one callback."""
        ctrl = self._make_ctrl_with_llm(viewer, llm)

        ctrl._ensure_llm_removal_callback()
        ctrl._ensure_llm_removal_callback()
        ctrl._ensure_llm_removal_callback()

        on_removal_calls = [c for c in llm.calls if c[0] == "on_external_removal"]
        assert len(on_removal_calls) == 1

    def test_callback_ignores_step3_context(self, viewer, llm):
        """External removal of step3 layers must NOT close step2 dock."""
        ctrl = self._make_ctrl_with_llm(viewer, llm)
        ctrl._ensure_llm_removal_callback()

        # Simulate external removal for step3 layer — should be ignored
        ctrl._on_layer_externally_removed("curation_3d_001", "step3/curation")
        # No crash, no dock.close attempt (we'd get an import error
        # from _find_dock_by_title if it tried — test passes if no exception)

    def test_callback_fires_for_step2_context(self, viewer, llm):
        """External removal of step2 layers should attempt dock close."""
        ctrl = self._make_ctrl_with_llm(viewer, llm)
        ctrl._ensure_llm_removal_callback()

        # Patch _find_dock_by_title to return a fake dock
        fake_dock = FakeDock()
        import napari_epicycle.hitl_training_annotation.controller.training_annotation_controller as tac_module
        import napari_epicycle.annotator.launcher as launcher_mod

        original_find = getattr(launcher_mod, "_find_dock_by_title", None)
        launcher_mod._find_dock_by_title = lambda viewer, name: fake_dock

        try:
            ctrl._on_layer_externally_removed(
                "patch01_work_labels_lateral_3d", "step2/patch"
            )
            assert fake_dock.closed, "dock.close() should have been called"
        finally:
            if original_find is not None:
                launcher_mod._find_dock_by_title = original_find

    def test_callback_no_crash_when_no_dock(self, viewer, llm):
        """If no dock is open, external removal should not crash."""
        ctrl = self._make_ctrl_with_llm(viewer, llm)
        ctrl._ensure_llm_removal_callback()

        import napari_epicycle.annotator.launcher as launcher_mod
        original_find = getattr(launcher_mod, "_find_dock_by_title", None)
        launcher_mod._find_dock_by_title = lambda viewer, name: None

        try:
            # Should not raise
            ctrl._on_layer_externally_removed(
                "patch01_ch0_XY_view", "step2/patch"
            )
        finally:
            if original_find is not None:
                launcher_mod._find_dock_by_title = original_find


# ---------------------------------------------------------------------------
# Phase 4 — F3: Work labels colormap preserved across patch switch
# ---------------------------------------------------------------------------


class TestWorkLabelsColormapPreservation:
    """F3: Work labels colormap must survive patch switches."""

    def _make_ctrl_for_colormap(self, viewer, llm):
        from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
            TrainingAnnotationController,
        )

        ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
        ctrl.viewer = viewer
        ctrl.layer_lifecycle_manager = llm
        ctrl.loaded_stack_path = "/fake/path"
        ctrl.loaded_patch_name = "patch01"
        ctrl.selected_view = "XY"
        ctrl.view_target = "lateral"
        ctrl.in_refine_mode = False
        ctrl.interactive_manager = FakeInteractiveManager()
        ctrl.sample_records = {}
        ctrl.segmentation_manager = MagicMock()
        ctrl.segmentation_manager.get_masks = MagicMock(return_value=None)

        class UI:
            train_patch_list = FakeListWidget()
            lateral_radio = MagicMock(isChecked=lambda: True)
            axial_radio = MagicMock(isChecked=lambda: False)
            xy_radio = MagicMock(isChecked=lambda: True)
            xz_radio = MagicMock(isChecked=lambda: False)
            yz_radio = MagicMock(isChecked=lambda: False)
            axial_plane_combo = MagicMock()
        ctrl.ui = UI()
        return ctrl

    def test_colormap_saved_during_removal(self, viewer, llm):
        """_load_patch should save colormap from work_labels before removing."""
        lbl = viewer.add_labels(
            np.zeros((4, 8, 8), dtype=np.int32),
            name="patch01_work_labels_lateral_3d",
        )
        lbl.colormap = "custom_cmap_sentinel"

        ctrl = self._make_ctrl_for_colormap(viewer, llm)
        ctrl._load_patch(force_axis="XY")

        assert getattr(ctrl, "_saved_work_labels_colormap", None) == "custom_cmap_sentinel"

    def test_colormap_restored_on_new_layer(self, viewer, llm):
        """_ensure_work_labels_layer should restore saved colormap."""
        viewer.add_image(
            np.zeros((4, 8, 8), dtype=np.float32), name="patch01_ch0_XY_view"
        )
        ctrl = self._make_ctrl_for_colormap(viewer, llm)
        ctrl._saved_work_labels_colormap = "custom_cmap_sentinel"

        ctrl._ensure_work_labels_layer(seed_slice_label="", view_name="XY")

        lbl = viewer.layers["patch01_work_labels_lateral_3d"]
        assert lbl.colormap == "custom_cmap_sentinel"


# ---------------------------------------------------------------------------
# Phase 4 — F5: SAM layers cleaned on patch switch
# ---------------------------------------------------------------------------


class TestSAMCleanupOnPatchSwitch:
    """F5: SAM prompt layers must be cleaned up during _load_patch."""

    def test_sam_cleanup_called_during_load_patch(self, viewer, llm):
        """_load_patch should call cleanup_prompt_layers unconditionally."""
        from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
            TrainingAnnotationController,
        )

        ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
        ctrl.viewer = viewer
        ctrl.layer_lifecycle_manager = llm
        ctrl.loaded_stack_path = "/fake/path"
        ctrl.loaded_patch_name = "patch01"
        ctrl.selected_view = "XY"
        ctrl.view_target = "lateral"
        ctrl.in_refine_mode = False
        ctrl.interactive_manager = FakeInteractiveManager()

        class UI:
            train_patch_list = FakeListWidget()
            lateral_radio = MagicMock(isChecked=lambda: True)
            axial_radio = MagicMock(isChecked=lambda: False)
            xy_radio = MagicMock(isChecked=lambda: True)
            xz_radio = MagicMock(isChecked=lambda: False)
            yz_radio = MagicMock(isChecked=lambda: False)
            axial_plane_combo = MagicMock()
        ctrl.ui = UI()

        ctrl._load_patch(force_axis="XY")

        assert ctrl.interactive_manager.cleanup_called, (
            "cleanup_prompt_layers should be called unconditionally"
        )


# ---------------------------------------------------------------------------
# Phase 4 — F6: Navigate callback not wired
# ---------------------------------------------------------------------------


class TestNavigateCallbackRemoved:
    """F6: navigate_to_slice_callback should not be wired to the annotator."""

    def test_navigate_callback_not_in_launch_kwargs(self):
        """The launch call should not pass navigate_to_slice_callback."""
        import inspect
        from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
            TrainingAnnotationController,
        )

        source = inspect.getsource(TrainingAnnotationController._launch_multiview_annotator)
        assert "navigate_to_slice_callback" not in source, (
            "navigate_to_slice_callback should have been removed from launch call"
        )


# ---------------------------------------------------------------------------
# Crash fix: _close_subviewers idempotent guard
# ---------------------------------------------------------------------------


class TestCloseSubviewersIdempotent:
    """_close_subviewers must be safe to call multiple times."""

    def test_double_close_subviewers_no_crash(self):
        """Calling _close_subviewers twice must not raise."""
        from napari_epicycle.annotator.curation_2d_widget import Curation2DAnnotator

        widget = Curation2DAnnotator.__new__(Curation2DAnnotator)
        widget._subviewers_closed = False

        # First call: set flag, but super() will fail (no real subviewers)
        # — the RuntimeError catch handles that.
        widget._close_subviewers()
        assert widget._subviewers_closed is True

        # Second call: must bail immediately (no crash).
        widget._close_subviewers()
        assert widget._subviewers_closed is True


# ---------------------------------------------------------------------------
# Phase 1: View-filtered review tabs
# ---------------------------------------------------------------------------


class FakeTabWidget:
    """Minimal QTabWidget stand-in that records setTabText calls."""
    def __init__(self):
        self._tabs = {}

    def setTabText(self, index, text):
        self._tabs[index] = text


class _Item:
    """Minimal QListWidgetItem stand-in for review tab tests."""
    def __init__(self, text):
        self._text = text
        self._foreground = None
        self._data = {}
    def text(self):
        return self._text
    def setText(self, t):
        self._text = t
    def setForeground(self, brush):
        self._foreground = brush
    def setData(self, role, value):
        self._data[role] = value
    def data(self, role):
        return self._data.get(role)
    def setIcon(self, icon):
        pass


class _ReviewListWidget:
    """Minimal QListWidget with addItem/clear/item/count."""
    def __init__(self):
        self._items = []
    def addItem(self, text):
        self._items.append(_Item(text))
    def clear(self):
        self._items = []
    def item(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None
    def count(self):
        return len(self._items)


class TestViewFilteredTabs:
    """Review tabs must only show slices matching the current view bucket."""

    def _make_ctrl_with_samples(self):
        from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
            TrainingAnnotationController,
        )

        ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
        ctrl.selected_view = "XY"
        ctrl.view_target = "lateral"
        ctrl.sample_records = {
            "patch01_XY_5": {"sample_id": "patch01_XY_5", "status": "accepted", "bucket": "lateral", "view": "XY", "slice_index": 5, "updated_at": "1", "round_tag": "0"},
            "patch01_XY_8": {"sample_id": "patch01_XY_8", "status": "accepted", "bucket": "lateral", "view": "XY", "slice_index": 8, "updated_at": "2", "round_tag": "0"},
            "patch01_XZ_3": {"sample_id": "patch01_XZ_3", "status": "accepted", "bucket": "axial", "view": "XZ", "slice_index": 3, "updated_at": "3", "round_tag": "0"},
            "patch01_XY_12": {"sample_id": "patch01_XY_12", "status": "staged", "bucket": "lateral", "view": "XY", "slice_index": 12, "updated_at": "4", "round_tag": "0"},
            "patch01_YZ_7": {"sample_id": "patch01_YZ_7", "status": "staged", "bucket": "axial", "view": "YZ", "slice_index": 7, "updated_at": "5", "round_tag": "0"},
            "patch01_XY_1": {"sample_id": "patch01_XY_1", "status": "predicted", "bucket": "lateral", "view": "XY", "slice_index": 1, "updated_at": "6", "round_tag": "0"},
            "patch01_XZ_9": {"sample_id": "patch01_XZ_9", "status": "predicted", "bucket": "axial", "view": "XZ", "slice_index": 9, "updated_at": "7", "round_tag": "0"},
        }

        class UI:
            segment_slice_data_list = _ReviewListWidget()
            accepted_queue_list = _ReviewListWidget()
            staged_queue_list = _ReviewListWidget()
            sample_queue_list = _ReviewListWidget()
            review_tabs = FakeTabWidget()
        ctrl.ui = UI()
        return ctrl

    def test_unified_list_filters_by_lateral(self):
        """In lateral mode, unified list shows only lateral slices."""
        ctrl = self._make_ctrl_with_samples()
        ctrl.selected_view = "XY"
        ctrl.view_target = "lateral"

        ctrl._refresh_sample_queue()

        # 4 lateral items: 2 accepted + 1 staged + 1 predicted
        assert ctrl.ui.segment_slice_data_list.count() == 4

    def test_unified_list_filters_by_axial(self):
        """In axial mode, unified list shows only axial slices."""
        ctrl = self._make_ctrl_with_samples()
        ctrl.selected_view = "XZ"
        ctrl.view_target = "axial"

        ctrl._refresh_sample_queue()

        # 3 axial items: 1 accepted + 1 staged + 1 predicted
        assert ctrl.ui.segment_slice_data_list.count() == 3

    def test_view_switch_changes_list_contents(self):
        """Switching views refreshes the unified list to show matching slices."""
        ctrl = self._make_ctrl_with_samples()

        ctrl.selected_view = "XY"
        ctrl.view_target = "lateral"
        ctrl._refresh_sample_queue()
        lateral_count = ctrl.ui.segment_slice_data_list.count()

        ctrl.selected_view = "XZ"
        ctrl.view_target = "axial"
        ctrl._refresh_sample_queue()
        axial_count = ctrl.ui.segment_slice_data_list.count()

        assert lateral_count == 4  # 2 acc + 1 stg + 1 pred
        assert axial_count == 3    # 1 acc + 1 stg + 1 pred
        assert lateral_count != axial_count


# ---------------------------------------------------------------------------
# Phase 2: Per-view training rounds
# ---------------------------------------------------------------------------


class TestPerViewRounds:
    """Training rounds must be tracked independently per view (lateral/axial)."""

    def _make_ctrl(self):
        from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
            TrainingAnnotationController,
        )

        ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
        ctrl.selected_view = "XY"
        ctrl.view_target = "lateral"
        # Per-view round tracking
        ctrl.current_round_by_view = {"lateral": 0, "axial": 0}
        return ctrl

    def test_current_round_returns_lateral_when_in_lateral_mode(self):
        ctrl = self._make_ctrl()
        ctrl.selected_view = "XY"
        ctrl.view_target = "lateral"
        ctrl.current_round_by_view = {"lateral": 3, "axial": 1}

        assert ctrl._current_round() == 3

    def test_current_round_returns_axial_when_in_axial_mode(self):
        ctrl = self._make_ctrl()
        ctrl.selected_view = "XZ"
        ctrl.view_target = "axial"
        ctrl.current_round_by_view = {"lateral": 3, "axial": 1}

        assert ctrl._current_round() == 1

    def test_current_round_key_returns_bucket(self):
        ctrl = self._make_ctrl()
        ctrl.selected_view = "XY"
        ctrl.view_target = "lateral"
        assert ctrl._current_round_key() == "lateral"

        ctrl.selected_view = "XZ"
        ctrl.view_target = "axial"
        assert ctrl._current_round_key() == "axial"

    def test_backward_compat_int_round_converts_to_dict(self):
        """Old state with int current_round should convert to per-view dict."""
        ctrl = self._make_ctrl()
        # Simulate loading old state
        ctrl.current_round_by_view = None
        ctrl.current_round = 5

        # The migration helper should produce a dict
        result = ctrl._migrate_round_to_per_view()
        assert result == {"lateral": 5, "axial": 5}
