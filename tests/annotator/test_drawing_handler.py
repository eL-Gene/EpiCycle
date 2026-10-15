"""Tests for DrawingHandler — throttling, fill, state accessors.

Phase K11d — comprehensive curation test suite.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from packages.epicycle_annotator.drawing import DrawingHandler


# ---------------------------------------------------------------------------
# Lightweight fakes
# ---------------------------------------------------------------------------


class _FakeSyncManager:
    def __init__(self):
        self.sync_calls = []
        self.track_calls = []
        self.sync_selected_label_calls = []
        # 2026-05-28 fix/step2-autoinc-perf: faithful mimic of the real
        # cached next-fill-label counter (LabelSyncManager.next_fill_label).
        self._next_fill_label_cache = None
        self.main_viewer = None  # subclass may set a fake main_viewer

    def sync_to_main_view(self, layer, orientation, viewer_model, **kwargs):
        self.sync_calls.append((layer.name, orientation, kwargs))

    def sync_other_views(self, name, orientation):
        pass

    def track_slice_edit(self, layer, viewer_model, orientation):
        self.track_calls.append((layer.name, orientation))

    def sync_selected_label(self, layer):
        self.sync_selected_label_calls.append(layer.name)

    def next_fill_label(self, active_layer, just_painted=0):
        if self._next_fill_label_cache is None:
            m = 0
            try:
                m = int(np.max(active_layer.data))
            except Exception:
                m = 0
            mv = getattr(self, "main_viewer", None)
            if mv is not None:
                for lyr in getattr(mv, "layers", []):
                    if lyr is not active_layer:
                        try:
                            m = max(m, int(np.max(lyr.data)))
                        except Exception:
                            pass
            self._next_fill_label_cache = m
        self._next_fill_label_cache = max(self._next_fill_label_cache, int(just_painted))
        self._next_fill_label_cache += 1
        return max(1, self._next_fill_label_cache)

    def invalidate_next_fill_label_cache(self):
        self._next_fill_label_cache = None


class _FakeLayer:
    def __init__(self, data, name="", mode="paint"):
        self.data = data
        self.name = name
        self.mode = mode
        self.selected_label = 1
        self._epicycle_enclosed_fill_enabled = True
        self._epicycle_auto_increment_label = True
        self.mouse_drag_callbacks = []

    def refresh(self):
        pass


# ---------------------------------------------------------------------------
# Tests — adaptive throttling
# ---------------------------------------------------------------------------


class TestAdaptiveThrottle:
    def test_small_volume_50ms(self):
        handler = DrawingHandler(_FakeSyncManager())
        layer = _FakeLayer(np.zeros((10, 10, 10), dtype=np.int32))  # 1000 voxels
        assert handler._adaptive_throttle(layer) == 0.05

    def test_medium_volume_100ms(self):
        handler = DrawingHandler(_FakeSyncManager())
        layer = _FakeLayer(np.zeros((170, 170, 170), dtype=np.int32))  # ~4.9M
        assert handler._adaptive_throttle(layer) == 0.05  # still < 5M

    def test_medium_tier_100ms(self):
        handler = DrawingHandler(_FakeSyncManager())
        # Create a fake layer with data.size > 5M but < 20M
        layer = _FakeLayer(np.empty(0))
        layer.data = types.SimpleNamespace(size=10_000_000)
        assert handler._adaptive_throttle(layer) == 0.10

    def test_large_volume_200ms(self):
        handler = DrawingHandler(_FakeSyncManager())
        layer = _FakeLayer(np.empty(0))
        layer.data = types.SimpleNamespace(size=50_000_000)
        assert handler._adaptive_throttle(layer) == 0.20

    def test_very_large_volume_disabled(self):
        handler = DrawingHandler(_FakeSyncManager())
        layer = _FakeLayer(np.empty(0))
        layer.data = types.SimpleNamespace(size=200_000_000)
        assert handler._adaptive_throttle(layer) == handler._THROTTLE_DISABLED

    def test_broken_data_falls_back(self):
        handler = DrawingHandler(_FakeSyncManager())
        layer = _FakeLayer(np.empty(0))
        layer.data = None  # will raise on .size
        result = handler._adaptive_throttle(layer)
        assert result == handler.throttle_seconds


# ---------------------------------------------------------------------------
# Tests — state accessors
# ---------------------------------------------------------------------------


class TestStateAccessors:
    def test_not_editing_initially(self):
        handler = DrawingHandler(_FakeSyncManager())
        assert handler.is_editing() is False
        assert handler.get_active_edit_view() is None


# ---------------------------------------------------------------------------
# Tests — fill outline
# ---------------------------------------------------------------------------


class TestFillOutline:
    def test_fills_polygon(self):
        handler = DrawingHandler(_FakeSyncManager())
        data = np.zeros((1, 20, 20), dtype=np.int32)
        layer = _FakeLayer(data, mode="paint")
        layer.selected_label = 5

        # Set up polygon points: (i, j, z_index) — a simple rectangle
        handler._polygon_points = [
            (2, 2, 0), (2, 10, 0), (10, 10, 0), (10, 2, 0)
        ]
        handler._fill_outline(layer)

        # Check that the interior of the polygon is filled
        assert data[0, 5, 5] == 5
        assert data[0, 0, 0] == 0  # outside polygon

    def test_fill_clips_to_existing_labels(self):
        # The enclosed-fill must fill ONLY background
        # inside the polygon — a neighbouring label that falls within it is
        # preserved, and our own label is untouched.
        handler = DrawingHandler(_FakeSyncManager())
        data = np.zeros((1, 20, 20), dtype=np.int32)
        data[0, 5, 5] = 7  # a NEIGHBOUR label inside the polygon
        data[0, 6, 6] = 5  # our OWN label inside the polygon
        layer = _FakeLayer(data, mode="paint")
        layer.selected_label = 5
        handler._polygon_points = [
            (2, 2, 0), (2, 10, 0), (10, 10, 0), (10, 2, 0)
        ]
        handler._fill_outline(layer)

        assert data[0, 5, 5] == 7  # neighbour PRESERVED (not overwritten)
        assert data[0, 6, 6] == 5  # own label untouched
        assert data[0, 3, 3] == 5  # background inside polygon -> filled
        assert data[0, 0, 0] == 0  # outside polygon -> untouched

    def test_fill_exempt_layer_raw_write(self):
        # A layer that opts OUT (_epicycle_clip_to_existing=False, e.g. the split
        # Refine layer) must do a RAW fill — overwriting whatever's inside the
        # polygon — so a seed loop can paint over the target label.
        handler = DrawingHandler(_FakeSyncManager())
        data = np.zeros((1, 20, 20), dtype=np.int32)
        data[0, 5, 5] = 7  # target-label voxel inside the polygon
        layer = _FakeLayer(data, mode="paint")
        layer.selected_label = 11  # a split seed id
        layer._epicycle_clip_to_existing = False
        handler._polygon_points = [
            (2, 2, 0), (2, 10, 0), (10, 10, 0), (10, 2, 0)
        ]
        handler._fill_outline(layer)

        assert data[0, 5, 5] == 11  # overwritten (raw fill, not clipped)
        assert data[0, 3, 3] == 11  # background -> filled

    def test_fill_clears_polygon_points(self):
        handler = DrawingHandler(_FakeSyncManager())
        data = np.zeros((1, 20, 20), dtype=np.int32)
        layer = _FakeLayer(data, mode="paint")
        layer.selected_label = 1
        handler._polygon_points = [(2, 2, 0), (2, 10, 0), (10, 10, 0)]
        handler._fill_outline(layer)
        assert len(handler._polygon_points) == 0

    def test_fill_fewer_than_3_points_is_noop(self):
        handler = DrawingHandler(_FakeSyncManager())
        data = np.zeros((1, 20, 20), dtype=np.int32)
        layer = _FakeLayer(data, mode="paint")
        handler._polygon_points = [(2, 2, 0), (2, 10, 0)]
        handler._fill_outline(layer)
        assert data.sum() == 0  # nothing filled


# ---------------------------------------------------------------------------
# Tests — callback creation
# ---------------------------------------------------------------------------


class TestCallbackCreation:
    def test_returns_callable(self):
        handler = DrawingHandler(_FakeSyncManager())
        vm = types.SimpleNamespace(cursor=types.SimpleNamespace(position=(0, 0, 0)))
        cb = handler.create_mouse_drag_callback("XY", vm)
        assert callable(cb)

    def test_non_edit_mode_returns_empty_generator(self):
        """Drag callback should return an immediately-exhausted generator for non-edit modes."""
        handler = DrawingHandler(_FakeSyncManager())
        vm = types.SimpleNamespace(cursor=types.SimpleNamespace(position=(0, 0, 0)))
        cb = handler.create_mouse_drag_callback("XY", vm)
        layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), mode="pan_zoom")
        event = types.SimpleNamespace(type="mouse_press")
        result = cb(layer, event)
        # _on_mouse_drag is a generator function (has yield), so it always
        # returns a generator.  For non-edit modes it returns/exhausts immediately.
        items = list(result)
        assert items == []
        assert handler.is_editing() is False


# ---------------------------------------------------------------------------
# Tests — button filter (Mechanism B companion, 2026-05-16)
# ---------------------------------------------------------------------------


class TestButtonFilter:
    """Drag callback must short-circuit on anything other than left-
    button (event.button == 1).  Without this gate, right-click + drag
    on a paint-mode subviewer label accumulates polygon points across
    mouse_move events and _fill_outline paints the enclosed polygon
    on release — observed 2026-05-16 by user after the napari-side
    paint suppression (install_button_aware_paint) closed the first
    paint path.  This gate closes the second."""

    def _make_handler_and_cb(self):
        handler = DrawingHandler(_FakeSyncManager())
        vm = types.SimpleNamespace(
            cursor=types.SimpleNamespace(position=(0, 5, 5)),
            dims=types.SimpleNamespace(current_step=(0,)),
        )
        cb = handler.create_mouse_drag_callback("XY", vm)
        return handler, cb

    def _drive_drag(self, cb, layer, button, n_moves=3):
        """Drive the generator through press → N moves → release.
        Returns whether the generator produced any yields (proxy for
        'entered the editing loop')."""
        event = types.SimpleNamespace(
            type="mouse_press", button=button, position=(0, 5, 5),
        )
        gen = cb(layer, event)
        if gen is None:
            return False
        produced_yield = False
        try:
            next(gen)
            produced_yield = True
            for i in range(n_moves):
                event.type = "mouse_move"
                next(gen)
            event.type = "mouse_release"
            next(gen)
        except StopIteration:
            pass
        return produced_yield

    def test_right_click_drag_does_not_enter_editing(self):
        """Right-button drag must short-circuit before any drawing
        state is touched."""
        handler, cb = self._make_handler_and_cb()
        layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), mode="paint")
        before = layer.data.copy()
        produced = self._drive_drag(cb, layer, button=2, n_moves=5)
        assert produced is False, (
            "right-click must not yield — should short-circuit before "
            "entering the editing loop"
        )
        assert handler.is_editing() is False, (
            "is_editing flag must NOT be set during a right-button drag"
        )
        assert np.array_equal(layer.data, before), (
            "right-click drag must not modify the volume via "
            "_fill_outline on release"
        )

    def test_middle_click_drag_does_not_enter_editing(self):
        handler, cb = self._make_handler_and_cb()
        layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), mode="paint")
        before = layer.data.copy()
        produced = self._drive_drag(cb, layer, button=3, n_moves=5)
        assert produced is False
        assert np.array_equal(layer.data, before)

    def test_left_click_drag_DOES_enter_editing(self):
        """Regression guard: left-button drag must still enter the
        editing loop and paint normally."""
        handler, cb = self._make_handler_and_cb()
        layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), mode="paint")
        produced = self._drive_drag(cb, layer, button=1, n_moves=5)
        assert produced is True, (
            "left-button drag must enter the editing loop"
        )

    def test_missing_button_attribute_treated_as_left(self):
        """If event.button is missing (some fake events / older
        callers), default to left-button behavior — better than
        silently dropping legitimate paint events."""
        handler, cb = self._make_handler_and_cb()
        layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), mode="paint")
        event = types.SimpleNamespace(type="mouse_press", position=(0, 5, 5))
        # No `button` attribute at all.
        gen = cb(layer, event)
        try:
            next(gen)  # should reach first yield without raising
        except StopIteration:
            pytest.fail(
                "missing event.button must not short-circuit — "
                "defaults to button=1 (left)"
            )

    def test_non_edit_mode_short_circuits_before_button_check(self):
        """Mode gate runs first: pan_zoom with any button stays a no-op
        (preserves the existing behavior, doesn't accidentally regress
        from the button-filter addition)."""
        handler, cb = self._make_handler_and_cb()
        layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), mode="pan_zoom")
        produced = self._drive_drag(cb, layer, button=1, n_moves=5)
        assert produced is False


# ---------------------------------------------------------------------------
# Tests — new_label_callback
# ---------------------------------------------------------------------------


class _FakeSyncManagerWithMainViewer(_FakeSyncManager):
    """Sync manager fake that exposes a main_viewer with one Labels-like
    layer so the DrawingHandler auto-increment vol_max sweep doesn't
    short-circuit on AttributeError."""

    def __init__(self, layer):
        super().__init__()
        # napari's _Labels isinstance check fails for fakes, so the
        # sweep silently skips them — same outcome as an empty viewer.
        # The point is just to exercise the code path without crashing.
        self.main_viewer = types.SimpleNamespace(layers=[])


class TestNewLabelCallback:
    """When the user paints an enclosed polygon in paint mode AND the
    layer's auto-increment flag is enabled, the DrawingHandler bumps
    selected_label to max+1 and fires ``new_label_callback`` with the
    LABEL THAT WAS JUST PAINTED (the pre-bump selected_label) so the
    annotator/controller can UPSERT a navigator row for the new cell.

    The callback must NOT fire when:
      - auto-increment is disabled (refining an existing label)
      - fewer than 3 polygon points were captured (incomplete drag)
      - mode is not paint
      - button is not left (1)
    """

    def _make_handler(self):
        vm = types.SimpleNamespace(
            cursor=types.SimpleNamespace(position=(0, 5, 5)),
            dims=types.SimpleNamespace(current_step=(0,)),
        )
        return DrawingHandler(_FakeSyncManager()), vm

    def _drive_left_polygon(self, handler, vm, layer, painted_label=7,
                            n_moves=5):
        """Drive a left-button drag that builds a 5-point polygon
        suitable for enclosed-fill."""
        cb = handler.create_mouse_drag_callback("XY", vm)
        # Set selected_label BEFORE the stroke — this is what gets
        # painted and what new_label_callback should report.
        layer.selected_label = painted_label
        event = types.SimpleNamespace(
            type="mouse_press", button=1, position=(0, 5, 5),
        )
        gen = cb(layer, event)
        try:
            next(gen)
            for i in range(n_moves):
                # Move the cursor — _on_mouse_drag reads
                # viewer_model.cursor.position to append to
                # _polygon_points.  Modify the cursor to a different
                # position each move so the polygon has 5 distinct
                # points.
                vm.cursor.position = (0, 5 + i, 5 + i)
                event.type = "mouse_move"
                next(gen)
            event.type = "mouse_release"
            next(gen)
        except StopIteration:
            pass

    def test_fires_with_painted_label_on_auto_increment(self):
        """The callback receives the OLD selected_label (the label that
        was just painted), NOT the new max+1 bump value."""
        handler, vm = self._make_handler()
        layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), mode="paint")
        layer._epicycle_auto_increment_label = True
        layer._epicycle_enclosed_fill_enabled = True

        received = []
        handler.new_label_callback = lambda lid: received.append(lid)

        self._drive_left_polygon(handler, vm, layer, painted_label=7)

        assert received == [7], (
            f"expected callback fired once with painted_label=7, got {received}"
        )
        # And selected_label SHOULD have advanced past 7 (max+1 of
        # the filled volume).
        assert layer.selected_label > 7

    def test_fires_even_when_auto_increment_disabled(self):
        """REGRESSION GUARD 2026-05-16: previously the callback was
        gated INSIDE the auto-increment branch.  But Curation3DAnnotator
        disables auto-increment by default
        (curation_3d_widget.py:288 ``_auto_increment_labels = False``)
        — which broke in production (new labels never registered
        in the navigator).  Fix: callback fires regardless of
        auto-increment state, on any enclosed-fill with a real label.
        Controller's UPSERT is idempotent so paint-over of existing
        labels is a no-op."""
        handler, vm = self._make_handler()
        layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), mode="paint")
        layer._epicycle_auto_increment_label = False  # 3D curation case
        layer._epicycle_enclosed_fill_enabled = True

        received = []
        handler.new_label_callback = lambda lid: received.append(lid)

        self._drive_left_polygon(handler, vm, layer, painted_label=7)

        assert received == [7], (
            f"callback must fire on enclosed-fill regardless of "
            f"auto-increment state, got {received}"
        )
        # selected_label must NOT have been bumped (auto-increment off)
        assert layer.selected_label == 7

    def test_does_not_fire_on_incomplete_polygon(self):
        """A drag with fewer than 3 polygon points doesn't reach
        _fill_outline and doesn't trigger auto-increment — no
        callback."""
        handler, vm = self._make_handler()
        layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), mode="paint")
        layer._epicycle_auto_increment_label = True
        layer._epicycle_enclosed_fill_enabled = True

        received = []
        handler.new_label_callback = lambda lid: received.append(lid)

        # n_moves=1 → only 1 polygon point appended → _fill_outline
        # branch skipped.
        self._drive_left_polygon(
            handler, vm, layer, painted_label=7, n_moves=1
        )

        assert received == []

    def test_does_not_fire_on_right_click_drag(self):
        """Button gate already blocks right-click; new_label_callback
        gets no chance to fire because _on_mouse_drag returns early."""
        handler, vm = self._make_handler()
        layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), mode="paint")
        layer._epicycle_auto_increment_label = True
        layer._epicycle_enclosed_fill_enabled = True

        received = []
        handler.new_label_callback = lambda lid: received.append(lid)

        cb = handler.create_mouse_drag_callback("XY", vm)
        layer.selected_label = 7
        event = types.SimpleNamespace(
            type="mouse_press", button=2, position=(0, 5, 5),
        )
        gen = cb(layer, event)
        try:
            next(gen)
        except StopIteration:
            pass
        # No moves, no release — short-circuited by button gate

        assert received == []

    def test_callback_failure_is_swallowed(self):
        """The callback runs inside a try/except — a buggy controller
        callback must not crash the drag handler.  We assert no
        exception propagates."""
        handler, vm = self._make_handler()
        layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), mode="paint")
        layer._epicycle_auto_increment_label = True
        layer._epicycle_enclosed_fill_enabled = True

        def _bad_cb(lid):
            raise RuntimeError("simulated controller bug")

        handler.new_label_callback = _bad_cb

        # Should not raise
        self._drive_left_polygon(handler, vm, layer, painted_label=7)

    def test_callback_default_is_none(self):
        """Fresh handler has new_label_callback = None — no surprises
        if no one sets it."""
        handler = DrawingHandler(_FakeSyncManager())
        assert handler.new_label_callback is None

    def test_callback_does_not_fire_when_unset(self):
        """No subscriber means no fire, no error."""
        handler, vm = self._make_handler()
        layer = _FakeLayer(np.zeros((4, 8, 8), dtype=np.int32), mode="paint")
        layer._epicycle_auto_increment_label = True
        layer._epicycle_enclosed_fill_enabled = True
        # Leave handler.new_label_callback at default (None)
        # Should not raise
        self._drive_left_polygon(handler, vm, layer, painted_label=7)


# ---------------------------------------------------------------------------
# Tests — production controller has _on_annotator_new_label_painted
# ---------------------------------------------------------------------------


class TestProductionControllerHasNewLabelHandler:
    """Smoke check that the FusionReconstructionController exposes the
    handler that __on_annotator_new_label_painted__ wires through.
    Pure import check — no controller instantiation needed."""

    def test_controller_has_method(self):
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        assert hasattr(
            FusionReconstructionController,
            "_on_annotator_new_label_painted",
        ), (
            "Controller must define _on_annotator_new_label_painted "
            "for new-label registration to flow through the standard "
            "_commit_curation_action chain."
        )
        assert callable(
            FusionReconstructionController._on_annotator_new_label_painted
        )


# ---------------------------------------------------------------------------
# Tests — throttle tiers configuration
# ---------------------------------------------------------------------------


class TestThrottleTiers:
    def test_tiers_are_ascending(self):
        """Throttle tiers should have increasing voxel thresholds and intervals."""
        tiers = DrawingHandler._THROTTLE_TIERS
        for i in range(len(tiers) - 1):
            assert tiers[i][0] < tiers[i + 1][0]  # ascending voxel thresholds
            assert tiers[i][1] <= tiers[i + 1][1]  # non-decreasing intervals

    def test_minimum_throttle_enforced(self):
        """Constructor enforces minimum throttle of 0.01s."""
        handler = DrawingHandler(_FakeSyncManager(), throttle_seconds=0.001)
        assert handler.throttle_seconds >= 0.01
