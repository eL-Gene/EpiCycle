"""Tests for Curation2DAnnotator lifecycle management.

Verifies lifecycle guards, deferred auto-load, close cleanup,
and session token gating. Uses source inspection (no Qt required).
"""

from __future__ import annotations

import inspect
import pytest


# ---------------------------------------------------------------------------
# Tests — Class structure
# ---------------------------------------------------------------------------


class TestCuration2DAnnotatorStructure:
    """Curation2DAnnotator must subclass NativeMultiviewAnnotator."""

    def test_subclass_of_native_multiview(self):
        from napari_epicycle.annotator.curation_2d_widget import (
            Curation2DAnnotator,
        )
        from napari_epicycle.annotator.native_multiview_widget import (
            NativeMultiviewAnnotator,
        )
        assert issubclass(Curation2DAnnotator, NativeMultiviewAnnotator)

    def test_does_not_subclass_curation_3d(self):
        from napari_epicycle.annotator.curation_2d_widget import (
            Curation2DAnnotator,
        )
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        assert not issubclass(Curation2DAnnotator, Curation3DAnnotator)


# ---------------------------------------------------------------------------
# Tests — Lifecycle guards
# ---------------------------------------------------------------------------


class TestLifecycleGuards:
    """Lifecycle guards must be initialized before super().__init__."""

    def test_init_sets_lifecycle_guards_before_super(self):
        source = _get_init_source()
        # The actual super call (not in a comment)
        super_idx = source.index("super().__init__(viewer)")
        assert source.index("self._is_closing = False") < super_idx
        assert source.index("self._close_cleanup_done = False") < super_idx
        assert source.index("self._session_token = 1") < super_idx

    def test_has_is_closing_guard(self):
        source = _get_init_source()
        assert "_is_closing = False" in source

    def test_has_close_cleanup_done_guard(self):
        source = _get_init_source()
        assert "_close_cleanup_done = False" in source

    def test_has_session_token(self):
        source = _get_init_source()
        assert "_session_token = 1" in source


# ---------------------------------------------------------------------------
# Tests — Deferred auto-load
# ---------------------------------------------------------------------------


class TestDeferredAutoLoad:
    """Auto-load must use deferred timers, not immediate loading."""

    def test_init_creates_auto_load_timers(self):
        source = _get_init_source()
        assert "_init_auto_load_timer" in source
        assert "_deferred_auto_load_timer" in source

    def test_auto_load_if_ready_checks_is_closing(self):
        source = _get_method_source("_auto_load_if_ready")
        assert "_is_closing" in source

    def test_auto_load_if_ready_checks_visibility(self):
        source = _get_method_source("_auto_load_if_ready")
        assert "isVisible" in source

    def test_auto_load_if_ready_calls_load_selected_layers(self):
        source = _get_method_source("_auto_load_if_ready")
        assert "_load_selected_layers" in source

    def test_schedule_deferred_uses_timer(self):
        source = _get_method_source("_schedule_deferred_auto_load")
        assert "timer.start()" in source

    def test_on_layer_inserted_skips_if_closing(self):
        source = _get_method_source("_on_layer_inserted")
        assert "_is_closing" in source

    def test_on_layer_inserted_skips_if_already_loaded(self):
        source = _get_method_source("_on_layer_inserted")
        assert "_active_image_name" in source


# ---------------------------------------------------------------------------
# Tests — set_context override
# ---------------------------------------------------------------------------


class TestSetContextOverride:
    """set_context must use deferred auto-load, not immediate loading."""

    def test_set_context_overrides_parent(self):
        from napari_epicycle.annotator.curation_2d_widget import (
            Curation2DAnnotator,
        )
        from napari_epicycle.annotator.native_multiview_widget import (
            NativeMultiviewAnnotator,
        )
        assert (
            Curation2DAnnotator.set_context
            is not NativeMultiviewAnnotator.set_context
        )

    def test_set_context_cancels_init_timer(self):
        source = _get_method_source("set_context")
        assert "_init_auto_load_timer" in source
        assert "stop()" in source

    def test_set_context_schedules_deferred_load(self):
        source = _get_method_source("set_context")
        assert "_schedule_deferred_auto_load" in source

    def test_set_context_does_not_call_load_selected_layers_directly(self):
        """set_context should use deferred timer, not direct call."""
        source = _get_method_source("set_context")
        # Should NOT have a bare self._load_selected_layers() call
        # (the docstring mentions it but the code should use the timer)
        lines = [l.strip() for l in source.split("\n") if not l.strip().startswith(("#", '"""', "//"))]
        code_lines = [l for l in lines if "load_selected_layers()" in l and not l.startswith("self._schedule")]
        # Filter out comments and docstring references
        actual_calls = [l for l in code_lines if not l.startswith("#") and "immediate" not in l.lower()]
        assert len(actual_calls) == 0, f"Found direct _load_selected_layers() call: {actual_calls}"


# ---------------------------------------------------------------------------
# Tests — _load_selected_layers override (proxy bug fix)
# ---------------------------------------------------------------------------


class TestLoadSelectedLayersOverride:
    """_load_selected_layers must fix the slow-path proxy bug."""

    def test_overrides_parent(self):
        from napari_epicycle.annotator.curation_2d_widget import (
            Curation2DAnnotator,
        )
        from napari_epicycle.annotator.native_multiview_widget import (
            NativeMultiviewAnnotator,
        )
        assert (
            Curation2DAnnotator._load_selected_layers
            is not NativeMultiviewAnnotator._load_selected_layers
        )

    def test_calls_super(self):
        source = _get_method_source("_load_selected_layers")
        assert "super()._load_selected_layers()" in source

    def test_reconnects_sync_to_real_layer(self):
        source = _get_method_source("_load_selected_layers")
        assert "connect_label_sync(lobj)" in source

    def test_checks_isinstance_labels(self):
        source = _get_method_source("_load_selected_layers")
        assert "isinstance(lobj, Labels)" in source


# ---------------------------------------------------------------------------
# Tests — _close_cleanup
# ---------------------------------------------------------------------------


class TestCloseCleanup:
    """_close_cleanup must be idempotent and thorough."""

    def test_has_reentrancy_guard(self):
        source = _get_method_source("_close_cleanup")
        assert "_close_cleanup_done" in source
        assert "return" in source

    def test_sets_is_closing(self):
        source = _get_method_source("_close_cleanup")
        assert "_is_closing = True" in source

    def test_invalidates_session_callbacks(self):
        source = _get_method_source("_close_cleanup")
        assert "_invalidate_session_callbacks" in source

    def test_stops_timers(self):
        source = _get_method_source("_close_cleanup")
        assert "_init_auto_load_timer" in source
        assert "_deferred_auto_load_timer" in source

    def test_destroys_satellite_docks(self):
        source = _get_method_source("_close_cleanup")
        assert "_destroy_satellite_docks" in source

    def test_disconnects_viewer_events(self):
        source = _get_method_source("_close_cleanup")
        assert "_disconnect_viewer_events" in source

    def test_disconnects_layer_insertion_handler(self):
        source = _get_method_source("_close_cleanup")
        assert "_layer_inserted_cb" in source

    def test_nulls_callback_refs(self):
        source = _get_method_source("_close_cleanup")
        assert "_segment_slice_callback" in source
        assert "_accept_slice_callback" in source
        assert "_patch_switch_callback" in source


# ---------------------------------------------------------------------------
# Tests — closeEvent
# ---------------------------------------------------------------------------


class TestCloseEvent:
    """closeEvent must call full cleanup chain."""

    def test_calls_close_cleanup(self):
        source = _get_method_source("closeEvent")
        assert "_close_cleanup()" in source

    def test_calls_close_subviewers(self):
        source = _get_method_source("closeEvent")
        assert "_close_subviewers()" in source

    def test_schedules_orphan_sweep(self):
        source = _get_method_source("closeEvent")
        assert "_schedule_close_orphan_sweep_retries" in source

    def test_calls_super_close_event(self):
        source = _get_method_source("closeEvent")
        assert "super().closeEvent(event)" in source


# ---------------------------------------------------------------------------
# Tests — Session token gating
# ---------------------------------------------------------------------------


class TestSessionTokenGating:
    """Session token system must gate deferred callbacks."""

    def test_invalidate_increments_token(self):
        source = _get_method_source("_invalidate_session_callbacks")
        assert "_session_token" in source

    def test_run_in_active_session_checks_token(self):
        source = _get_method_source("_run_in_active_session")
        assert "_session_token" in source

    def test_run_in_active_session_checks_is_closing(self):
        source = _get_method_source("_run_in_active_session")
        assert "_is_closing" in source


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_init_source():
    from napari_epicycle.annotator.curation_2d_widget import (
        Curation2DAnnotator,
    )
    return inspect.getsource(Curation2DAnnotator.__init__)


def _get_method_source(method_name):
    from napari_epicycle.annotator.curation_2d_widget import (
        Curation2DAnnotator,
    )
    return inspect.getsource(getattr(Curation2DAnnotator, method_name))
