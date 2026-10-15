"""Tests for launcher wiring — patch options, context, triage callbacks.

Phase K11d — comprehensive curation test suite.
"""

from __future__ import annotations

import importlib
import types

import pytest

from napari_epicycle.annotator.launcher import launch_3d_curation_annotator


# ---------------------------------------------------------------------------
# Lightweight fakes
# ---------------------------------------------------------------------------


class _FakeDock:
    def __init__(self, visible=True):
        self._visible = visible
        self.shown = False
        self.raised = False
        self._widget = self
        self._last_context = None
        self._last_flagged = None
        self._last_triage = None
        self._last_patch_options = None

    def isVisible(self):
        return self._visible

    def show(self):
        self.shown = True

    def raise_(self):
        self.raised = True

    def widget(self):
        return self._widget

    def windowTitle(self):
        return ""

    def setParent(self, parent):
        pass

    def deleteLater(self):
        pass

    def set_context(self, **kwargs):
        self._last_context = kwargs

    def set_focused_refinement(self, ids, centroids):
        self._last_flagged = (ids, centroids)

    def set_triage_callbacks(
        self, on_mark_tier=None, on_label_deleted=None, on_merge=None
    ):
        self._last_triage = (on_mark_tier, on_label_deleted, on_merge)

    def set_patch_options(self, **kwargs):
        self._last_patch_options = kwargs

    def set_flagged_labels(self, items):
        self._last_flagged_labels = items

    def set_discard_garbage_callback(self, callback):
        self._last_discard_garbage_cb = callback


class _FakeWindow:
    def __init__(self):
        self._dock_widgets = {}
        self.add_calls = []

    def add_dock_widget(self, widget, name=None, area=None):
        self.add_calls.append((widget, name, area))
        self._dock_widgets[name] = _FakeDock()

    def remove_dock_widget(self, widget):
        pass


class _FakeViewer:
    def __init__(self):
        self.window = _FakeWindow()


# ---------------------------------------------------------------------------
# Tests — patch options wiring on existing dock
# ---------------------------------------------------------------------------


class TestPatchOptionsWiring:
    def test_patch_options_passed_to_existing_dock(self):
        viewer = _FakeViewer()
        dock = _FakeDock(visible=True)
        viewer.window._dock_widgets["EpiCycle 3D Curation Annotator"] = dock

        launch_3d_curation_annotator(
            viewer,
            patch_ids=["p1.tif", "p2.tif"],
            active_patch_id="p1.tif",
            switch_patch_callback=lambda x: x,
            patch_statuses={"p1.tif": "curating", "p2.tif": "reconstructed"},
        )

        assert dock._last_patch_options is not None
        assert dock._last_patch_options["patch_ids"] == ["p1.tif", "p2.tif"]
        assert dock._last_patch_options["active_patch_id"] == "p1.tif"
        assert dock._last_patch_options["patch_statuses"]["p2.tif"] == "reconstructed"

    def test_no_patch_ids_passes_empty_list(self):
        viewer = _FakeViewer()
        dock = _FakeDock(visible=True)
        viewer.window._dock_widgets["EpiCycle 3D Curation Annotator"] = dock

        launch_3d_curation_annotator(viewer)

        assert dock._last_patch_options is not None
        assert dock._last_patch_options["patch_ids"] == []


# ---------------------------------------------------------------------------
# Tests — context wiring on existing dock
# ---------------------------------------------------------------------------


class TestContextWiring:
    def test_context_set_on_existing_dock(self):
        viewer = _FakeViewer()
        dock = _FakeDock(visible=True)
        viewer.window._dock_widgets["EpiCycle 3D Curation Annotator"] = dock

        launch_3d_curation_annotator(
            viewer,
            image_layer_name="stack_p1",
            labels_layer_name="curation_3d_abc",
        )

        assert dock._last_context is not None
        assert dock._last_context["image_layer_name"] == "stack_p1"
        assert dock._last_context["labels_layer_name"] == "curation_3d_abc"
        assert dock._last_context["auto_load"] is True

    def test_empty_layer_names_still_set_context(self):
        viewer = _FakeViewer()
        dock = _FakeDock(visible=True)
        viewer.window._dock_widgets["EpiCycle 3D Curation Annotator"] = dock

        launch_3d_curation_annotator(viewer)

        assert dock._last_context is not None


# ---------------------------------------------------------------------------
# Tests — flagged labels wiring
# ---------------------------------------------------------------------------


class TestFlaggedLabelsWiring:
    def test_flagged_labels_passed_to_dock(self):
        viewer = _FakeViewer()
        dock = _FakeDock(visible=True)
        viewer.window._dock_widgets["EpiCycle 3D Curation Annotator"] = dock

        launch_3d_curation_annotator(
            viewer,
            flagged_label_ids=[1, 5, 10],
            flagged_centroids={1: (0, 0, 0), 5: (1, 2, 3), 10: (4, 5, 6)},
        )

        assert dock._last_flagged is not None
        assert dock._last_flagged[0] == [1, 5, 10]
        assert dock._last_flagged[1][5] == (1, 2, 3)

    def test_no_flagged_labels_skips_wiring(self):
        viewer = _FakeViewer()
        dock = _FakeDock(visible=True)
        viewer.window._dock_widgets["EpiCycle 3D Curation Annotator"] = dock

        launch_3d_curation_annotator(viewer)

        assert dock._last_flagged is None


# ---------------------------------------------------------------------------
# Tests — triage callbacks wiring
# ---------------------------------------------------------------------------


class TestTriageCallbacksWiring:
    def test_triage_callbacks_passed_to_dock(self):
        viewer = _FakeViewer()
        dock = _FakeDock(visible=True)
        viewer.window._dock_widgets["EpiCycle 3D Curation Annotator"] = dock

        def on_mark_tier(lid, tier):
            pass
        def on_label_deleted(lid, **kwargs):
            pass

        launch_3d_curation_annotator(
            viewer,
            on_mark_tier=on_mark_tier,
            on_label_deleted=on_label_deleted,
        )

        assert dock._last_triage is not None
        # Tuple: (on_mark_tier, on_label_deleted, on_merge)
        assert dock._last_triage[0] is on_mark_tier
        assert dock._last_triage[1] is on_label_deleted


    def test_plot_selection_getter_reaches_set_curation_callbacks(self):
        """The live QC-plot getter is one named
        kwarg on the launcher, forwarded to set_curation_callbacks."""
        viewer = _FakeViewer()
        dock = _FakeDock(visible=True)
        seen = {}
        dock.set_curation_callbacks = lambda **kw: seen.update(kw)
        viewer.window._dock_widgets["EpiCycle 3D Curation Annotator"] = dock

        def getter():
            return [1, 2]

        launch_3d_curation_annotator(viewer, get_plot_selection=getter)
        assert seen.get("get_plot_selection") is getter

    def test_no_plot_selection_getter_forwards_none(self):
        viewer = _FakeViewer()
        dock = _FakeDock(visible=True)
        seen = {}
        dock.set_curation_callbacks = lambda **kw: seen.update(kw)
        viewer.window._dock_widgets["EpiCycle 3D Curation Annotator"] = dock
        launch_3d_curation_annotator(viewer)
        assert "get_plot_selection" in seen
        assert seen["get_plot_selection"] is None


# ---------------------------------------------------------------------------
# Tests — hidden dock gets destroyed
# ---------------------------------------------------------------------------


class TestHiddenDockHandling:
    def test_hidden_dock_triggers_fresh_launch(self, monkeypatch):
        viewer = _FakeViewer()
        # Add a hidden dock
        hidden_dock = _FakeDock(visible=False)
        viewer.window._dock_widgets["EpiCycle 3D Curation Annotator"] = hidden_dock

        class _FakeWidget:
            def __init__(self, v):
                self.viewer = v
            def set_context(self, **kwargs):
                pass
            def set_patch_options(self, **kwargs):
                pass
            def set_triage_callbacks(self, *args, **kwargs):
                pass

        fake_module = types.SimpleNamespace(Curation3DAnnotator=_FakeWidget)
        monkeypatch.setattr(importlib, "import_module", lambda _name: fake_module)

        ok, msg = launch_3d_curation_annotator(viewer)
        assert ok is True
        assert "Opened" in msg


# ---------------------------------------------------------------------------
# Tests — error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_no_viewer(self):
        ok, msg = launch_3d_curation_annotator(None)
        assert ok is False
        assert "Viewer" in msg

    def test_no_window(self):
        viewer = types.SimpleNamespace(window=None)
        ok, msg = launch_3d_curation_annotator(viewer)
        assert ok is False

    def test_import_failure(self, monkeypatch):
        viewer = _FakeViewer()

        def _raise(*args, **kwargs):
            raise ImportError("test error")

        monkeypatch.setattr(importlib, "import_module", _raise)

        ok, msg = launch_3d_curation_annotator(viewer)
        assert ok is False
        assert "Failed" in msg

    def test_reuse_returns_already_open(self):
        viewer = _FakeViewer()
        dock = _FakeDock(visible=True)
        viewer.window._dock_widgets["EpiCycle 3D Curation Annotator"] = dock

        ok, msg = launch_3d_curation_annotator(viewer)
        assert ok is True
        assert "already open" in msg


# ---------------------------------------------------------------------------
# Tests — same-patch click guard & clear_curation_layers dedup
# ---------------------------------------------------------------------------


class TestSamePatchGuard:
    """_on_curation_patch_clicked should skip reload when clicking the same patch."""

    def test_same_patch_skips_reload(self):
        """When _qc_view_patch_name matches the clicked patch, no work is done."""
        load_calls = []

        class _FakeController:
            def __init__(self):
                self._qc_view_patch_name = "patch_a.tif"

            def _selected_curation_patch(self):
                return "patch_a.tif"

            def _clear_qc_view_layers(self):
                load_calls.append("clear_qc")

            def _clear_curation_layers(self):
                load_calls.append("clear_curation")

        ctrl = _FakeController()

        # Simulate the guard logic.  Population-mode early return removed
        # 2026-05-03 (drop-population-mode Commit 5/6).
        clicked = ctrl._selected_curation_patch()
        if clicked and clicked == ctrl._qc_view_patch_name:
            pass  # early return
        else:
            ctrl._clear_qc_view_layers()
            ctrl._clear_curation_layers()

        assert len(load_calls) == 0, "Same-patch click should not clear/reload"

    def test_different_patch_proceeds(self):
        """When _qc_view_patch_name differs, reload proceeds."""
        load_calls = []

        class _FakeController:
            def __init__(self):
                self._qc_view_patch_name = "patch_a.tif"

            def _selected_curation_patch(self):
                return "patch_b.tif"

            def _clear_qc_view_layers(self):
                load_calls.append("clear_qc")

            def _clear_curation_layers(self):
                load_calls.append("clear_curation")

        ctrl = _FakeController()

        clicked = ctrl._selected_curation_patch()
        if clicked and clicked == ctrl._qc_view_patch_name:
            pass
        else:
            ctrl._clear_qc_view_layers()
            ctrl._clear_curation_layers()

        assert "clear_qc" in load_calls
        assert "clear_curation" in load_calls

    def test_empty_patch_name_proceeds(self):
        """First click (no patch loaded yet) should always proceed."""
        load_calls = []

        class _FakeController:
            def __init__(self):
                self._qc_view_patch_name = ""

            def _selected_curation_patch(self):
                return "patch_a.tif"

            def _clear_qc_view_layers(self):
                load_calls.append("clear_qc")

        ctrl = _FakeController()

        clicked = ctrl._selected_curation_patch()
        if clicked and clicked == ctrl._qc_view_patch_name:
            pass
        else:
            ctrl._clear_qc_view_layers()

        assert "clear_qc" in load_calls


class TestClearCurationLayersDedup:
    """_clear_curation_layers must skip stack_ layers tracked by _qc_view_layers."""

    def test_skips_qc_tracked_stack_layers(self):
        """Stack layers in _qc_view_layers should NOT be removed by _clear_curation_layers."""

        class _Layer:
            def __init__(self, name):
                self.name = name

        qc_stack = _Layer("stack_wound_3-4")
        qc_stack2 = _Layer("stack_wound_3-4 [1]")
        curation_lbl = _Layer("curation_3d_abc123")
        orphan_stack = _Layer("stack_old_patch")

        all_layers = [qc_stack, qc_stack2, curation_lbl, orphan_stack]
        qc_view_layers = [qc_stack, qc_stack2]  # tracked by QC view

        qc_tracked = set(id(l) for l in qc_view_layers)
        to_remove = []
        for layer in all_layers:
            if layer.name.startswith("curation_3d_"):
                to_remove.append(layer)
            elif layer.name.startswith("stack_") and id(layer) not in qc_tracked:
                to_remove.append(layer)

        remove_names = [l.name for l in to_remove]

        # curation_3d_ always removed
        assert "curation_3d_abc123" in remove_names
        # orphan stack_ removed (not tracked by QC)
        assert "stack_old_patch" in remove_names
        # QC-tracked stack_ layers NOT removed
        assert "stack_wound_3-4" not in remove_names
        assert "stack_wound_3-4 [1]" not in remove_names


class TestAnnotatorSwitchSamePatchGuard:
    """_switch_patch_from_3d_annotator should skip reload when patch already loaded."""

    def _make_fake(self, current_patch, switch_to):
        """Build a minimal fake that tracks whether reload happened."""
        calls = []

        class _Fake:
            def __init__(self):
                self._qc_view_patch_name = current_patch
                self.state_manager = types.SimpleNamespace(
                    _state={"patches": {
                        "patch_a.tif": {"reconstruction_run_id": "run1"},
                        "patch_b.tif": {"reconstruction_run_id": "run2"},
                    }}
                )

            def _clear_qc_view_layers(self):
                calls.append("clear")

            def _load_patch_for_qc_view(self, name):
                calls.append(f"load:{name}")

        return _Fake(), calls

    def test_same_patch_returns_early(self):
        """Switching to the already-loaded patch should not clear or reload."""
        fake, calls = self._make_fake("patch_a.tif", "patch_a.tif")
        patch_name = "patch_a.tif"

        # Simulate the guard logic from _switch_patch_from_3d_annotator
        if patch_name == fake._qc_view_patch_name:
            record = fake.state_manager._state["patches"].get(patch_name, {})
            run_id = record.get("reconstruction_run_id", "")
            from pathlib import Path
            stem = Path(patch_name).stem
            result = {
                "image_layer_name": f"stack_{stem}",
                "labels_layer_name": f"reconstruction_{run_id}" if run_id else "",
                "active_patch_id": patch_name,
            }
        else:
            fake._clear_qc_view_layers()
            fake._load_patch_for_qc_view(patch_name)
            result = None

        assert len(calls) == 0, "Same-patch switch should not clear/reload"
        assert result["active_patch_id"] == "patch_a.tif"
        assert result["image_layer_name"] == "stack_patch_a"
        assert result["labels_layer_name"] == "reconstruction_run1"

    def test_different_patch_proceeds(self):
        """Switching to a different patch should clear and reload."""
        fake, calls = self._make_fake("patch_a.tif", "patch_b.tif")
        patch_name = "patch_b.tif"

        if patch_name == fake._qc_view_patch_name:
            result = {}
        else:
            fake._clear_qc_view_layers()
            fake._load_patch_for_qc_view(patch_name)
            result = None

        assert "clear" in calls
        assert "load:patch_b.tif" in calls

    def test_empty_current_patch_proceeds(self):
        """First switch (no patch loaded yet) should always proceed."""
        fake, calls = self._make_fake("", "patch_a.tif")
        patch_name = "patch_a.tif"

        if patch_name == fake._qc_view_patch_name:
            result = {}
        else:
            fake._clear_qc_view_layers()
            fake._load_patch_for_qc_view(patch_name)
            result = None

        assert "clear" in calls
        assert "load:patch_a.tif" in calls


# ---------------------------------------------------------------------------
# Tests — flagged_label_items + on_discard_garbage wiring
# ---------------------------------------------------------------------------


class TestFlaggedLabelItemsWiring:
    """launch_3d_curation_annotator passes flagged_label_items to set_flagged_labels."""

    def test_flagged_items_passed_to_existing_dock(self):
        viewer = _FakeViewer()
        dock = _FakeDock(visible=True)
        viewer.window._dock_widgets["EpiCycle 3D Curation Annotator"] = dock

        items = [
            {"label_id": 1, "tier": "good", "centroid": (0, 0, 0), "area": 100},
            {"label_id": 2, "tier": "garbage", "centroid": (1, 1, 1), "area": 50},
        ]
        launch_3d_curation_annotator(viewer, flagged_label_items=items)

        assert hasattr(dock, "_last_flagged_labels")
        assert dock._last_flagged_labels == items

    def test_no_items_skips_wiring(self):
        viewer = _FakeViewer()
        dock = _FakeDock(visible=True)
        viewer.window._dock_widgets["EpiCycle 3D Curation Annotator"] = dock

        launch_3d_curation_annotator(viewer)
        assert not hasattr(dock, "_last_flagged_labels")


class TestDiscardGarbageCallbackWiring:
    """launch_3d_curation_annotator passes on_discard_garbage to set_discard_garbage_callback."""

    def test_callback_passed_to_existing_dock(self):
        viewer = _FakeViewer()
        dock = _FakeDock(visible=True)
        viewer.window._dock_widgets["EpiCycle 3D Curation Annotator"] = dock

        def _on_discard(ids):
            pass

        launch_3d_curation_annotator(viewer, on_discard_garbage=_on_discard)

        assert hasattr(dock, "_last_discard_garbage_cb")
        assert dock._last_discard_garbage_cb is _on_discard

    def test_no_callback_skips_wiring(self):
        viewer = _FakeViewer()
        dock = _FakeDock(visible=True)
        viewer.window._dock_widgets["EpiCycle 3D Curation Annotator"] = dock

        launch_3d_curation_annotator(viewer)
        assert not hasattr(dock, "_last_discard_garbage_cb")


# ---------------------------------------------------------------------------
# Tests — Navigator dock reuse callback rebinding
# ---------------------------------------------------------------------------

from napari_epicycle.annotator.launcher import launch_curation_label_navigator


class _FakeNavigatorWidget:
    """Lightweight stand-in for CurationLabelNavigator with callback tracking.

    Mirrors the real set_callbacks() signature and attribute names exactly.
    """

    def __init__(self):
        self._on_navigate_cb = None
        self._on_select_label_cb = None
        self._on_mark_good_cb = None
        self._on_delete_cb = None
        self._on_discard_garbage_cb = None
        self._on_merge_cb = None
        self._on_focused_mode_changed_cb = None
        self._on_isolation_changed_cb = None
        self._on_scroll_subviewer_cb = None
        self._on_recalculate_cb = None
        self._set_callbacks_calls = 0
        self._pushed_items = []

    def set_callbacks(
        self,
        *,
        on_navigate=None,
        on_select_label=None,
        on_mark_good=None,
        on_delete=None,
        on_discard_garbage=None,
        on_merge=None,
        on_focused_mode_changed=None,
        on_show_selected_label_changed=None,
        on_isolation_changed=None,
        on_scroll_subviewer=None,
        on_recalculate=None,
    ):
        self._set_callbacks_calls += 1
        self._on_navigate_cb = on_navigate
        self._on_select_label_cb = on_select_label
        self._on_mark_good_cb = on_mark_good
        self._on_delete_cb = on_delete
        self._on_discard_garbage_cb = on_discard_garbage
        self._on_merge_cb = on_merge
        self._on_focused_mode_changed_cb = on_focused_mode_changed
        self._on_isolation_changed_cb = on_isolation_changed
        self._on_scroll_subviewer_cb = on_scroll_subviewer
        self._on_recalculate_cb = on_recalculate

    def set_items(self, items):
        self._pushed_items = list(items)


class _FakeAnnotatorForNav:
    """Lightweight stand-in for Curation3DAnnotator with navigator wiring."""

    def __init__(self):
        self._navigator_widget = None
        self._cached_flagged_items = []
        # Callback methods that navigator should be wired to
        self._navigator_navigate = lambda lid, c: None
        self._on_navigator_select_label = lambda lid: None
        self._on_mark_good_from_navigator = lambda lid: None
        self._on_delete_from_navigator = lambda lid: None
        self._on_discard_garbage_from_navigator = lambda ids: None
        self._on_merge_from_navigator = lambda k, a: None
        self._toggle_focused_mode = lambda e: None
        self._on_isolation_changed = lambda s: None
        self._on_scroll_subviewer = lambda d: None
        self._on_recalculate_cb = None

    def _wire_navigator_callbacks(self, nav):
        """Wire navigator callbacks to this annotator's methods."""
        nav.set_callbacks(
            on_navigate=self._navigator_navigate,
            on_select_label=self._on_navigator_select_label,
            on_mark_good=self._on_mark_good_from_navigator,
            on_delete=self._on_delete_from_navigator,
            on_discard_garbage=self._on_discard_garbage_from_navigator,
            on_merge=self._on_merge_from_navigator,
            on_focused_mode_changed=self._toggle_focused_mode,
            on_isolation_changed=self._on_isolation_changed,
            on_scroll_subviewer=self._on_scroll_subviewer,
            on_recalculate=self._on_recalculate_cb,
        )


class _FakeNavDock:
    """Dock wrapper for navigator widget."""

    def __init__(self, widget):
        self._widget = widget
        self._visible = True

    def widget(self):
        return self._widget

    def windowTitle(self):
        return "EpiCycle Label Navigator"

    def show(self):
        pass

    def raise_(self):
        pass

    def isVisible(self):
        return self._visible


class TestNavigatorDockReuse:
    """launch_curation_label_navigator must rebind callbacks when reusing
    an existing navigator dock, update the annotator reference, and push
    cached items."""

    def test_reuse_existing_navigator_rebinds_callbacks_to_current_annotator(self):
        """When a navigator dock already exists, its callbacks must be
        re-wired to the current annotator's bound methods."""
        viewer = _FakeViewer()
        nav = _FakeNavigatorWidget()

        # Wire stale callbacks to simulate old annotator
        stale_fn = lambda lid, c: "stale"
        nav._on_navigate_cb = stale_fn

        dock = _FakeNavDock(nav)
        viewer.window._dock_widgets["EpiCycle Label Navigator"] = dock

        # New annotator instance (simulates relaunch)
        annotator = _FakeAnnotatorForNav()

        widget, ok, msg = launch_curation_label_navigator(viewer, annotator)

        assert ok is True
        assert nav._set_callbacks_calls > 0, (
            "Navigator set_callbacks must be called on dock reuse "
            "to rebind callbacks to the current annotator"
        )
        assert nav._on_navigate_cb is annotator._navigator_navigate, (
            "Navigate callback must point to current annotator, not stale"
        )

    def test_reuse_existing_navigator_sets_annotator_ref_and_pushes_cached_items(self):
        """Dock reuse must update annotator._navigator_widget and push
        cached flagged items to the reused navigator."""
        viewer = _FakeViewer()
        nav = _FakeNavigatorWidget()
        dock = _FakeNavDock(nav)
        viewer.window._dock_widgets["EpiCycle Label Navigator"] = dock

        annotator = _FakeAnnotatorForNav()
        annotator._cached_flagged_items = [
            {"label_id": 1, "tier": "review", "centroid": (0, 0, 0), "area": 100},
            {"label_id": 2, "tier": "bad", "centroid": (1, 1, 1), "area": 50},
        ]

        widget, ok, msg = launch_curation_label_navigator(viewer, annotator)

        assert ok is True
        assert annotator._navigator_widget is nav, (
            "annotator._navigator_widget must be set to the reused navigator"
        )
        assert len(nav._pushed_items) == 2, (
            "Cached flagged items must be pushed to the reused navigator"
        )

    def test_closing_annotator_closes_or_invalidates_navigator_dock_link(self):
        """After annotator close cleanup, navigator callbacks must be
        invalidated so no stale callback fires into the dead annotator.

        This tests the behavioral contract: between close and relaunch,
        the navigator's callbacks must not target the old annotator.
        The real _close_cleanup should clear navigator callbacks before
        nulling _navigator_widget.
        """
        from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator
        from napari_epicycle.fusion_reconstruction.colormap_authority import (
            ColormapAuthority,
        )

        # Build a minimal fake annotator that can exercise the real _close_cleanup
        ann = types.SimpleNamespace()

        # Navigator with stale callbacks
        nav = _FakeNavigatorWidget()
        nav.set_callbacks(on_navigate=lambda lid, c: "stale_target")
        assert nav._on_navigate_cb is not None
        ann._navigator_widget = nav

        # Attributes required by _close_cleanup
        ann._smooth_running = False
        ann._split_running = False
        ann._interp_snapshot = None
        ann._interp_refine_mode = False
        ann._isolation_active = False
        ann._isolation_label_set = None
        ann._pre_isolation_colormap = None
        ann._colormap_authority = ColormapAuthority()
        ann._alive_check_timer = types.SimpleNamespace(stop=lambda: None)
        ann._disconnect_viewer_events = lambda: None
        ann._disconnect_label_handlers = lambda: None
        ann._restore_main_viewer_arrow_keys = lambda: None
        ann._on_isolation_cleared_cb = None
        ann._on_mark_good_cb = None
        ann._on_delete_label_cb = None
        ann._on_discard_garbage_cb = None
        ann._on_merge_cb = None
        ann._on_navigator_label_selected_cb = None
        ann._on_interpolation_accepted_cb = None
        ann._on_label_split_cb = None
        ann._on_label_smoothed_cb = None
        ann._on_recalculate_cb = None
        ann._isolation_bridge_dispatched = False

        # Fake viewer.layers.events.inserted for disconnect
        class _FakeEvents:
            class inserted:
                @staticmethod
                def disconnect(handler):
                    pass
        ann.viewer = types.SimpleNamespace(
            layers=types.SimpleNamespace(events=_FakeEvents())
        )
        ann._on_layer_inserted = lambda *a: None

        # Run the real _close_cleanup
        bound = types.MethodType(Curation3DAnnotator._close_cleanup, ann)
        bound()

        # Navigator callbacks must be cleared (not left pointing at dead annotator)
        assert nav._on_navigate_cb is None, (
            "Navigator on_navigate callback must be cleared on annotator close "
            "to prevent stale callback firings between close and relaunch"
        )


class TestNavigatorDockReuseRound2:
    """Round 2: Navigator reuse must always push items (even empty) and
    reset stale UI state (selection/isolate) from the previous session."""

    def test_reuse_existing_navigator_pushes_empty_items_to_clear_stale_rows(self):
        """When cached items is an empty list, the reuse path must still call
        set_items([]) so stale rows from the previous session are cleared.

        Bug: ``if cached:`` is falsy for empty list, leaving old rows visible.
        """
        viewer = _FakeViewer()
        nav = _FakeNavigatorWidget()
        # Simulate stale rows from previous session
        nav._pushed_items = [
            {"label_id": 99, "tier": "garbage", "centroid": (0, 0, 0), "area": 10},
        ]
        dock = _FakeNavDock(nav)
        viewer.window._dock_widgets["EpiCycle Label Navigator"] = dock

        annotator = _FakeAnnotatorForNav()
        annotator._cached_flagged_items = []  # Empty cache

        widget, ok, msg = launch_curation_label_navigator(viewer, annotator)

        assert ok is True
        assert nav._pushed_items == [], (
            "Reuse path must call set_items([]) even for empty cache "
            "to clear stale rows from the previous session"
        )

    def test_reuse_existing_navigator_rebinds_callbacks_and_resets_ui_state(self):
        """Reuse path must call reset_selection_ui() on the navigator to
        clear stale table selection and uncheck the Isolate checkbox.

        Without this, the navigator carries over stale selection/isolate
        state from the previous annotator session.
        """
        viewer = _FakeViewer()
        nav = _FakeNavigatorWidget()
        # Add reset tracking
        nav._reset_selection_ui_calls = 0

        def _fake_reset():
            nav._reset_selection_ui_calls += 1

        nav.reset_selection_ui = _fake_reset

        dock = _FakeNavDock(nav)
        viewer.window._dock_widgets["EpiCycle Label Navigator"] = dock

        annotator = _FakeAnnotatorForNav()
        annotator._cached_flagged_items = [
            {"label_id": 1, "tier": "review", "centroid": (0, 0, 0), "area": 100},
        ]

        widget, ok, msg = launch_curation_label_navigator(viewer, annotator)

        assert ok is True
        assert nav._reset_selection_ui_calls > 0, (
            "Reuse path must call reset_selection_ui() to clear stale "
            "selection/isolate state from the previous session"
        )

    def test_relaunch_reuse_still_rebinds_callbacks_after_close_dock_handling(self):
        """After close hides/closes the navigator dock, relaunch reuse must
        still find the dock, rebind callbacks, and push items.

        The dock may be hidden (not destroyed) — the reuse path must
        show it, rebind, and reset.
        """
        viewer = _FakeViewer()
        nav = _FakeNavigatorWidget()
        nav._reset_selection_ui_calls = 0

        def _fake_reset():
            nav._reset_selection_ui_calls += 1

        nav.reset_selection_ui = _fake_reset

        dock = _FakeNavDock(nav)
        # Simulate dock was hidden by close cleanup
        dock._visible = False
        viewer.window._dock_widgets["EpiCycle Label Navigator"] = dock

        annotator = _FakeAnnotatorForNav()
        annotator._cached_flagged_items = [
            {"label_id": 5, "tier": "review", "centroid": (0, 0, 0), "area": 50},
        ]

        widget, ok, msg = launch_curation_label_navigator(viewer, annotator)

        assert ok is True
        assert nav._set_callbacks_calls > 0, (
            "Reuse must rebind callbacks even after dock was hidden by close"
        )
        assert annotator._navigator_widget is nav
        assert len(nav._pushed_items) == 1


# ---------------------------------------------------------------------------
# Tests — widget cleanup before dock destruction on relaunch
# ---------------------------------------------------------------------------


class TestDestroyDockCallsCleanup:
    """Verify that when a closed/hidden dock is destroyed on relaunch,
    the inner widget's _close_cleanup() and _close_subviewers() are called.

    This is the critical fix: _destroy_dock() does NOT trigger closeEvent,
    so the launcher must explicitly call cleanup before destroying.

    Tests use _cleanup_old_annotator_widget directly (extracted helper)
    to avoid triggering the full Qt launch path.
    """

    def test_close_cleanup_called(self):
        """_cleanup_old_annotator_widget must call _close_cleanup."""
        from napari_epicycle.annotator.launcher import _cleanup_old_annotator_widget

        cleanup_log = []
        old_widget = types.SimpleNamespace(
            _close_cleanup=lambda: cleanup_log.append("cleanup"),
            _close_subviewers=lambda: cleanup_log.append("subviewers"),
        )

        _cleanup_old_annotator_widget(old_widget)

        assert "cleanup" in cleanup_log, (
            "_close_cleanup must be called before _destroy_dock on relaunch"
        )

    def test_close_subviewers_called(self):
        """_cleanup_old_annotator_widget must call _close_subviewers."""
        from napari_epicycle.annotator.launcher import _cleanup_old_annotator_widget

        cleanup_log = []
        old_widget = types.SimpleNamespace(
            _close_cleanup=lambda: cleanup_log.append("cleanup"),
            _close_subviewers=lambda: cleanup_log.append("subviewers"),
        )

        _cleanup_old_annotator_widget(old_widget)

        assert "subviewers" in cleanup_log, (
            "_close_subviewers must be called to prevent magicgui leaks"
        )

    def test_tolerates_missing_methods(self):
        """Must not crash if the old widget lacks cleanup methods."""
        from napari_epicycle.annotator.launcher import _cleanup_old_annotator_widget

        # No _close_cleanup or _close_subviewers
        _cleanup_old_annotator_widget(types.SimpleNamespace())

    def test_tolerates_cleanup_exception(self):
        """Must not crash if cleanup throws."""
        from napari_epicycle.annotator.launcher import _cleanup_old_annotator_widget

        def _boom():
            raise RuntimeError("cleanup failed")

        old_widget = types.SimpleNamespace(
            _close_cleanup=_boom,
            _close_subviewers=lambda: None,
        )

        # Must not raise
        _cleanup_old_annotator_widget(old_widget)

    def test_skips_if_dock_close_hook_already_fired(self):
        """If close hook already fired, relaunch cleanup should skip."""
        from napari_epicycle.annotator.launcher import _cleanup_old_annotator_widget

        cleanup_log = []
        old_widget = types.SimpleNamespace(
            _close_cleanup=lambda: cleanup_log.append("cleanup"),
            _close_subviewers=lambda: cleanup_log.append("subviewers"),
            _dock_close_hook_fired=True,
        )

        _cleanup_old_annotator_widget(old_widget)

        assert len(cleanup_log) == 0


class TestDockCloseHook:
    """Close-event hook should trigger cleanup exactly once per lifecycle."""

    @staticmethod
    def _fake_close_event():
        from qtpy.QtCore import QEvent
        return types.SimpleNamespace(type=lambda: QEvent.Close)

    @staticmethod
    def _fake_non_close_event():
        from qtpy.QtCore import QEvent
        return types.SimpleNamespace(type=lambda: QEvent.Show)

    @staticmethod
    def _fake_hide_event():
        from qtpy.QtCore import QEvent
        return types.SimpleNamespace(type=lambda: QEvent.Hide)

    class _FakeSignal:
        def __init__(self):
            self._handlers = []

        def connect(self, cb):
            self._handlers.append(cb)

        def emit(self, value):
            for cb in list(self._handlers):
                cb(value)

    class _FakeToggleAction:
        def __init__(self, checked=True):
            self._checked = checked

        def isChecked(self):
            return self._checked

        def setChecked(self, checked):
            self._checked = bool(checked)

    class _FakeHookDock:
        def __init__(self, *, toggle_checked=True):
            self.filters = []
            self.visibilityChanged = TestDockCloseHook._FakeSignal()
            self._toggle_action = TestDockCloseHook._FakeToggleAction(
                checked=toggle_checked
            )
            self._visible = True

        def installEventFilter(self, filt):
            self.filters.append(filt)

        def toggleViewAction(self):
            return self._toggle_action

        def isVisible(self):
            return self._visible

    def test_cleanup_fires_on_close_event(self):
        from napari_epicycle.annotator.launcher import _install_dock_close_hook

        cleanup_log = []
        widget = types.SimpleNamespace(
            _close_cleanup=lambda: cleanup_log.append("cleanup"),
            _close_subviewers=lambda: cleanup_log.append("subviewers"),
        )
        dock = self._FakeHookDock()

        _install_dock_close_hook(dock, widget)
        assert len(dock.filters) == 1

        dock.filters[0].eventFilter(dock, self._fake_close_event())
        assert "cleanup" in cleanup_log
        assert "subviewers" in cleanup_log

    def test_non_close_event_does_not_cleanup(self):
        from napari_epicycle.annotator.launcher import _install_dock_close_hook

        cleanup_log = []
        widget = types.SimpleNamespace(
            _close_cleanup=lambda: cleanup_log.append("cleanup"),
            _close_subviewers=lambda: cleanup_log.append("subviewers"),
        )
        dock = self._FakeHookDock()

        _install_dock_close_hook(dock, widget)
        dock.filters[0].eventFilter(dock, self._fake_non_close_event())
        assert len(cleanup_log) == 0

    def test_one_shot_guard_prevents_double_cleanup(self):
        from napari_epicycle.annotator.launcher import _install_dock_close_hook

        cleanup_count = []
        widget = types.SimpleNamespace(
            _close_cleanup=lambda: cleanup_count.append(1),
            _close_subviewers=lambda: None,
        )
        dock = self._FakeHookDock()

        _install_dock_close_hook(dock, widget)
        dock.filters[0].eventFilter(dock, self._fake_close_event())
        dock.filters[0].eventFilter(dock, self._fake_close_event())
        assert len(cleanup_count) == 1

    def test_relaunch_cleanup_skips_after_hook_fired(self):
        from napari_epicycle.annotator.launcher import (
            _install_dock_close_hook,
            _cleanup_old_annotator_widget,
        )

        cleanup_count = []
        widget = types.SimpleNamespace(
            _close_cleanup=lambda: cleanup_count.append(1),
            _close_subviewers=lambda: None,
        )
        dock = self._FakeHookDock()

        _install_dock_close_hook(dock, widget)
        dock.filters[0].eventFilter(dock, self._fake_close_event())
        _cleanup_old_annotator_widget(widget)
        assert len(cleanup_count) == 1

    def test_hide_event_with_toggle_off_triggers_cleanup(self):
        from napari_epicycle.annotator.launcher import _install_dock_close_hook

        cleanup_log = []
        widget = types.SimpleNamespace(
            _close_cleanup=lambda: cleanup_log.append("cleanup"),
            _close_subviewers=lambda: cleanup_log.append("subviewers"),
        )
        dock = self._FakeHookDock(toggle_checked=False)

        _install_dock_close_hook(dock, widget)
        dock.filters[0].eventFilter(dock, self._fake_hide_event())
        assert "cleanup" in cleanup_log
        assert "subviewers" in cleanup_log

    def test_hide_event_with_toggle_on_skips_cleanup(self):
        from napari_epicycle.annotator.launcher import _install_dock_close_hook

        cleanup_log = []
        widget = types.SimpleNamespace(
            _close_cleanup=lambda: cleanup_log.append("cleanup"),
            _close_subviewers=lambda: cleanup_log.append("subviewers"),
        )
        dock = self._FakeHookDock(toggle_checked=True)

        _install_dock_close_hook(dock, widget)
        dock.filters[0].eventFilter(dock, self._fake_hide_event())
        assert len(cleanup_log) == 0

    def test_visibility_changed_hidden_toggle_off_schedules_cleanup(self):
        """visibilityChanged(False) should schedule debounced cleanup, not
        fire immediately — transient hides from napari layout updates
        (layer add/remove) must not trigger subviewer teardown."""
        from napari_epicycle.annotator.launcher import _install_dock_close_hook

        cleanup_log = []
        widget = types.SimpleNamespace(
            _close_cleanup=lambda: cleanup_log.append("cleanup"),
            _close_subviewers=lambda: cleanup_log.append("subviewers"),
        )
        dock = self._FakeHookDock(toggle_checked=False)

        _install_dock_close_hook(dock, widget)
        dock.visibilityChanged.emit(False)
        # Cleanup is debounced — NOT immediate.
        assert len(cleanup_log) == 0, (
            "visibilityChanged cleanup should be debounced, not immediate"
        )
        # Verify a debounce timer was installed.
        assert hasattr(widget, "_dock_close_vc_debounce"), (
            "Debounce timer should be stored on widget"
        )

    def test_visibility_changed_visible_cancels_pending_cleanup(self):
        """visibilityChanged(True) should cancel any pending debounced cleanup.

        In a real Qt env the timer would be active, but in tests without an
        event loop QTimer.start() is a no-op.  Verify that the visible=True
        path calls stop() and that no cleanup fires.
        """
        from napari_epicycle.annotator.launcher import _install_dock_close_hook

        cleanup_log = []
        widget = types.SimpleNamespace(
            _close_cleanup=lambda: cleanup_log.append("cleanup"),
            _close_subviewers=lambda: cleanup_log.append("subviewers"),
        )
        dock = self._FakeHookDock(toggle_checked=False)

        _install_dock_close_hook(dock, widget)
        dock.visibilityChanged.emit(False)
        # Now emit visible=True (transient hide resolved).
        dock.visibilityChanged.emit(True)
        assert len(cleanup_log) == 0, (
            "Cleanup should not fire after transient hide/show"
        )

    def test_visibility_changed_hidden_toggle_on_skips_cleanup(self):
        from napari_epicycle.annotator.launcher import _install_dock_close_hook

        cleanup_log = []
        widget = types.SimpleNamespace(
            _close_cleanup=lambda: cleanup_log.append("cleanup"),
            _close_subviewers=lambda: cleanup_log.append("subviewers"),
        )
        dock = self._FakeHookDock(toggle_checked=True)

        _install_dock_close_hook(dock, widget)
        dock.visibilityChanged.emit(False)
        # Even with debounce, toggle=True means the dock isn't really closing.
        # Timer may still be scheduled but the check should skip cleanup.
        assert len(cleanup_log) == 0

    def test_minimize_does_not_trigger_cleanup(self):
        """Minimizing the main napari window should NOT trigger annotator cleanup.

        When the main window is minimized, the dock widget fires
        visibilityChanged(False) and toggleViewAction may report unchecked.
        The fix checks ``main_window.isMinimized()`` to distinguish minimize
        from a real dock close.
        """
        from napari_epicycle.annotator.launcher import _install_dock_close_hook

        cleanup_log = []
        # Simulate a minimized main window.
        fake_qt_window = types.SimpleNamespace(isMinimized=lambda: True)
        fake_window = types.SimpleNamespace(_qt_window=fake_qt_window)
        fake_viewer = types.SimpleNamespace(window=fake_window)
        widget = types.SimpleNamespace(
            _close_cleanup=lambda: cleanup_log.append("cleanup"),
            _close_subviewers=lambda: cleanup_log.append("subviewers"),
            viewer=fake_viewer,
        )
        # toggle_checked=False simulates the ambiguous state where Qt reports
        # the toggle as unchecked during minimize.
        dock = self._FakeHookDock(toggle_checked=False)

        _install_dock_close_hook(dock, widget)

        # Emit visibility=False (what happens on minimize).
        dock.visibilityChanged.emit(False)
        assert len(cleanup_log) == 0, (
            "Cleanup should NOT fire when main window is minimized"
        )

        # Also verify the eventFilter Hide path respects the minimize guard.
        dock.filters[0].eventFilter(dock, self._fake_hide_event())
        assert len(cleanup_log) == 0, (
            "Hide event should NOT trigger cleanup when main window is minimized"
        )
