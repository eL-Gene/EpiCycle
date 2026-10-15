"""Tests for ``_switch_run_guard`` pure helpers (Phase 8a polish).

Qt dialog is not tested here (manual + Qt-aware integration).
"""

from __future__ import annotations

import pytest

from napari_epicycle.hitl_training_annotation._switch_run_guard import (
    SwitchSafety,
    check_switch_safety,
    collect_patch_layer_names,
)


# ----------------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------------

class FakeThread:
    def __init__(self, running=False):
        self._running = running

    def isRunning(self):
        return self._running


class FakeLayer:
    def __init__(self, name):
        self.name = name


class FakeViewerLayers(list):
    """Behaves like napari viewer.layers — iterable + supports `in` check
    by name + indexing by name."""

    def __contains__(self, item):
        if isinstance(item, str):
            return any(getattr(l, "name", None) == item for l in self)
        return list.__contains__(self, item)

    def __getitem__(self, key):
        if isinstance(key, str):
            for l in self:
                if getattr(l, "name", None) == key:
                    return l
            raise KeyError(key)
        return list.__getitem__(self, key)


class FakeViewer:
    def __init__(self, layer_names=None):
        self.layers = FakeViewerLayers(
            FakeLayer(n) for n in (layer_names or [])
        )


class FakeController:
    def __init__(self, *, training=None, loaded_patch_name="", viewer=None):
        self._training_thread = training
        self.loaded_patch_name = loaded_patch_name
        self.viewer = viewer


# ----------------------------------------------------------------------------
# check_switch_safety
# ----------------------------------------------------------------------------

class TestCheckSwitchSafety:

    def test_no_training_no_annotator_no_layers_is_safe(self):
        ctrl = FakeController()
        viewer = FakeViewer()
        safety, reasons = check_switch_safety(ctrl, viewer)
        assert safety == SwitchSafety.SAFE
        assert reasons == []

    def test_training_in_flight_blocks(self):
        ctrl = FakeController(training=FakeThread(running=True))
        viewer = FakeViewer()
        safety, reasons = check_switch_safety(ctrl, viewer)
        assert safety == SwitchSafety.BLOCKED
        assert any("Training is currently running" in r for r in reasons)

    def test_training_thread_not_running_doesnt_block(self):
        ctrl = FakeController(training=FakeThread(running=False))
        viewer = FakeViewer()
        safety, reasons = check_switch_safety(ctrl, viewer)
        assert safety == SwitchSafety.SAFE

    def test_training_thread_missing_attr_doesnt_block(self):
        """Defensive: controller without _training_thread attribute is safe."""
        class MinimalCtrl:
            pass
        safety, reasons = check_switch_safety(MinimalCtrl(), FakeViewer())
        assert safety == SwitchSafety.SAFE

    def test_isrunning_raising_treated_as_blocked(self):
        class ExplodingThread:
            def isRunning(self):
                raise RuntimeError("boom")
        ctrl = FakeController(training=ExplodingThread())
        safety, reasons = check_switch_safety(ctrl, FakeViewer())
        assert safety == SwitchSafety.BLOCKED
        assert any("indeterminate" in r for r in reasons)

    def test_no_viewer_doesnt_crash(self):
        ctrl = FakeController()
        safety, reasons = check_switch_safety(ctrl, None)
        assert safety == SwitchSafety.SAFE
        assert reasons == []

    def test_loaded_patch_layers_alone_doesnt_warn(self):
        """Layer warnings only fire if there's already another warning;
        otherwise the cleanup happens silently."""
        ctrl = FakeController(loaded_patch_name="my_patch")
        viewer = FakeViewer(["my_patch_image", "work_labels_XY"])
        safety, reasons = check_switch_safety(ctrl, viewer)
        # No annotator dock open (no real viewer), so no warning is added.
        # The stale-layers branch is silent (cleanup-on-switch is enough).
        assert safety == SwitchSafety.SAFE


# ----------------------------------------------------------------------------
# collect_patch_layer_names
# ----------------------------------------------------------------------------

class TestCollectPatchLayerNames:

    def test_empty_returns_empty(self):
        ctrl = FakeController()
        assert collect_patch_layer_names(FakeViewer(), ctrl) == []

    def test_no_loaded_patch_still_collects_managed_prefixes(self):
        """Expanded heuristic (audit-fix 2026-05-23): controller-managed
        prefixes (work_labels_, predictions_, isolation_proxy, etc.) are
        collected even without a loaded_patch_name — they're owned by the
        controller and safe to clean up on switch."""
        ctrl = FakeController(loaded_patch_name="")
        viewer = FakeViewer(["random_layer", "work_labels_XY"])
        result = collect_patch_layer_names(viewer, ctrl)
        assert "work_labels_XY" in result
        assert "random_layer" not in result

    def test_collects_predictions_and_isolation_proxy_layers(self):
        """Audit-fix: layer cleanup must catch predictions, EDT, SAM,
        isolation proxies, refine layers (not just work_labels_)."""
        ctrl = FakeController(loaded_patch_name="patch_3")
        viewer = FakeViewer([
            "patch_3_image",
            "predictions_round_03",
            "isolation_proxy",
            "edt_overlay",
            "sam_preview",
            "refine_layer",
            "fragment_refine_xy",
            "current_prediction_xy",
            "curation_3d_abc123",
            "some_other_overlay_mask_",
            "completely_unrelated",
        ])
        names = collect_patch_layer_names(viewer, ctrl)
        # Should catch all the controller-managed prefixes
        for expected in [
            "patch_3_image", "predictions_round_03", "isolation_proxy",
            "edt_overlay", "sam_preview", "refine_layer",
            "fragment_refine_xy", "current_prediction_xy",
            "curation_3d_abc123",
        ]:
            assert expected in names, f"missing {expected}"
        assert "completely_unrelated" not in names

    def test_collects_patch_named_layers(self):
        ctrl = FakeController(loaded_patch_name="patch_3")
        viewer = FakeViewer([
            "patch_3_image",
            "some_other_layer",
            "patch_3_labels",
            "patch_4_image",  # different patch — not collected
        ])
        names = collect_patch_layer_names(viewer, ctrl)
        assert "patch_3_image" in names
        assert "patch_3_labels" in names
        assert "patch_4_image" not in names
        assert "some_other_layer" not in names

    def test_collects_work_labels_layers(self):
        ctrl = FakeController(loaded_patch_name="patch_3")
        viewer = FakeViewer([
            "work_labels_XY",
            "work_labels_XZ",
            "work_labels_YZ",
            "random",
        ])
        names = collect_patch_layer_names(viewer, ctrl)
        assert set(names) >= {"work_labels_XY", "work_labels_XZ", "work_labels_YZ"}
        assert "random" not in names

    def test_no_viewer_returns_empty(self):
        ctrl = FakeController(loaded_patch_name="x")
        assert collect_patch_layer_names(None, ctrl) == []

    def test_layer_with_unreadable_name_skipped(self):
        ctrl = FakeController(loaded_patch_name="patch")
        class BadLayer:
            @property
            def name(self):
                raise RuntimeError("explode")
        viewer = FakeViewer(["patch_image"])
        viewer.layers.append(BadLayer())
        # Should skip the bad one without raising
        names = collect_patch_layer_names(viewer, ctrl)
        assert "patch_image" in names
