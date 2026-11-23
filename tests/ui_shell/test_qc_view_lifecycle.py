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
        """A patch opened for the QC view registers its layers under
        step3/qc/single.

        The patch now loads in two halves: ``_load_patch_for_qc_view``
        starts a background read, and ``_on_patch_loaded`` adds the layers
        on the main thread when it finishes.  The registration lives in the
        second half, so this drives ``_on_patch_loaded`` on a bare
        controller with a finished load result and checks what it
        registered.
        """
        import numpy as np
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )

        class _RecordingLLM:
            def __init__(self):
                self.registered = []

            def register_existing(self, name, context):
                self.registered.append((name, context))

        class _StateManager:
            def update_patch_metadata(self, patch_name, **kwargs):
                pass

        viewer = FakeViewer()
        # A layer whose name only starts with the stack name is not the
        # stack and must not be registered.
        viewer.add_image(None, name="stack_patch_AB")

        def _add_stack(data, name):
            viewer.add_image(data, name=name)
            viewer.add_image(data, name=f"{name} ch1")

        c = FusionReconstructionController.__new__(
            FusionReconstructionController
        )
        c.viewer = viewer
        c.layer_lifecycle_manager = _RecordingLLM()
        c.state_manager = _StateManager()
        c._patch_load_token = 7
        c._qc_view_layers = []
        c._add_image_channel_first = _add_stack
        c._sync_loaded_patch_marker = lambda: None
        c._busy_progress = lambda *a, **k: None
        c._busy_hide = lambda *a, **k: None
        c._deferred_apply_qc_colors = lambda stem: None
        c._reload_current_patch_features = lambda: None
        c._get_open_annotator_widget = lambda: None
        c._post_switch_annotator_sync = lambda widget, name: None
        c._set_curation_status = lambda text: None
        c._append_log = lambda msg: None

        result = {
            "labels": np.zeros((2, 4, 4), np.uint16),
            "source": np.zeros((2, 2, 4, 4), np.uint16),
            "downsample_factor": 1,
            "original_labels_shape": (2, 4, 4),
        }
        FusionReconstructionController._on_patch_loaded(
            c, result, "patch_A.tif", "patch_A", "stack_patch_A", 7,
        )

        assert c.layer_lifecycle_manager.registered == [
            ("stack_patch_A", "step3/qc/single"),
            ("stack_patch_A ch1", "step3/qc/single"),
            ("reconstruction_patch_A", "step3/qc/single"),
        ]
        assert [lyr.name for lyr in c._qc_view_layers] == [
            "stack_patch_A", "stack_patch_A ch1", "reconstruction_patch_A",
        ]

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
        """The population grid's teardown stays gone.

        Population mode (a tiled grid of every patch) was removed from
        Step 3 in May 2026, and its teardown method went with it.  This
        test used to check the teardown cleared its layers; it now checks
        the method has not come back without its mode.
        """
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        assert not hasattr(
            FusionReconstructionController, "_clear_population_grid"
        )

    def test_population_grid_builder_registers_layers(self):
        """No Step 3 code registers population-grid layers any more.

        Population mode was removed in May 2026, together with its grid
        builder and teardown.  The context name "step3/qc/population" is
        still mentioned in a comment of the controller, so a plain text
        search would find it; this reads only the controller's string
        values in code (comments and docstrings left out, the text parts of
        f-strings kept) and checks the context is not used.
        """
        import ast
        import inspect
        from napari_epicycle.fusion_reconstruction import controller as ctl
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        for gone in ("_load_population_grid", "_clear_population_grid"):
            assert not hasattr(FusionReconstructionController, gone), gone
        tree = ast.parse(inspect.getsource(ctl))
        # A docstring is the first statement of a module, class or function
        # when that statement is a bare string; those are prose, not code.
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
                body = node.body
                if (body and isinstance(body[0], ast.Expr)
                        and isinstance(body[0].value, ast.Constant)
                        and isinstance(body[0].value.value, str)):
                    docstrings.add(id(body[0].value))
        # ast.walk also reaches the text parts inside f-strings (the
        # Constant values of each JoinedStr).
        strings = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        ]
        assert strings, "the controller source should hold string values"
        assert any(isinstance(n, ast.JoinedStr) for n in ast.walk(tree)), (
            "the controller should hold f-strings, so their parts are read")
        assert not any("step3/qc/population" in s for s in strings)
