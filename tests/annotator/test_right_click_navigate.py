"""Tests for the subviewer right-click → navigator-focus feature.

Two surfaces covered:

1. ``Curation3DAnnotator._right_click_navigate_allowed`` — the 12-guard
   gate function.  Each guard tested in isolation.
2. The right-click callback returned by
   ``_make_subviewer_right_click_handler`` — verifies button-filter,
   bounds-check, background skip, and the
   nav.select_label(label_id) dispatch.

No napari / Qt instantiation — uses lightweight fakes, so this file is safe
to run as part of the annotator batch.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


# Import the method under test as an unbound function.  We bind it to
# a minimal fake annotator instance so we don't have to construct a
# full Curation3DAnnotator (which needs napari + qtpy + a viewer).
from napari_epicycle.annotator import curation_3d_widget as _c3d
from napari_epicycle.annotator.curation_3d_widget import (
    Curation3DAnnotator,
)
from napari_epicycle.annotator.refinement_state import RefinementOpState


_gate = Curation3DAnnotator._right_click_navigate_allowed
_make_handler = Curation3DAnnotator._make_subviewer_right_click_handler
_alive = Curation3DAnnotator._rclick_select_alive


# ---------------------------------------------------------------------------
# 2026-09-07 crash fix: the right-click handler no longer calls
# nav.select_label() inside vispy's mouse-release dispatch — it defers one
# event-loop turn via the module seam ``_defer_to_event_loop``.  These tests
# have no Qt event loop, so the seam is monkeypatched to a queue that the
# drive helpers flush explicitly.
# ---------------------------------------------------------------------------
_DEFERRED: list = []


@pytest.fixture(autouse=True)
def _capture_deferred(monkeypatch):
    _DEFERRED.clear()
    monkeypatch.setattr(_c3d, "_defer_to_event_loop", _DEFERRED.append)
    yield
    _DEFERRED.clear()


def _flush_deferred():
    """Run every queued deferred callable, in order (one event-loop turn)."""
    pending, _DEFERRED[:] = list(_DEFERRED), []
    for fn in pending:
        fn()


class _FakeNav:
    def __init__(self, right_click_enabled=True):
        self._enabled = right_click_enabled
        self.selected = []

    def is_right_click_select_enabled(self):
        return self._enabled

    def select_label(self, label_id):
        self.selected.append(label_id)


class _FakeMergeDialog:
    def __init__(self, visible=True):
        self._visible = visible

    def isVisible(self):
        return self._visible


class _FakeLayer:
    def __init__(self, data, mode="pan_zoom", name="fake_label_XY"):
        self.data = np.asarray(data)
        self.mode = mode
        self.name = name


class _FakeEvent:
    """Fake napari event.  Mutable position + type — tests can flip
    these between yields to simulate the press → move → release
    lifecycle of a napari mouse_drag_callbacks generator."""

    def __init__(self, button=2, position=(0, 0, 0), type="mouse_press"):
        self.button = button
        self.position = position
        self.type = type


def _drive_click(cb, layer, button=2, position=(0, 0, 0)):
    """Simulate napari driving the generator callback for a pure CLICK
    (press → release at same position, no movement)."""
    event = _FakeEvent(button=button, position=position, type="mouse_press")
    gen = cb(layer, event)
    if gen is None:
        return  # callback short-circuited before yield (e.g., button != 2)
    try:
        next(gen)  # run press code, hit first yield
        event.type = "mouse_release"
        next(gen)  # exits while loop, runs release code
    except StopIteration:
        pass
    _flush_deferred()


def _drive_drag(cb, layer, button=2, start=(0, 0, 0), end=(20, 20, 20), n_moves=3):
    """Simulate napari driving the generator callback for a DRAG
    (press → N moves → release at different position)."""
    event = _FakeEvent(button=button, position=start, type="mouse_press")
    gen = cb(layer, event)
    if gen is None:
        return
    try:
        next(gen)  # press code
        # Generate intermediate move positions
        for i in range(1, n_moves + 1):
            t = i / (n_moves + 1)
            event.position = tuple(
                s + (e - s) * t for s, e in zip(start, end)
            )
            event.type = "mouse_move"
            next(gen)  # one yield iter
        event.position = end
        event.type = "mouse_release"
        next(gen)
    except StopIteration:
        pass
    _flush_deferred()


def _make_ann(**overrides):
    """Build a minimal fake annotator with all gate-relevant fields.

    Binds ``_right_click_navigate_allowed`` so the handler closure
    (which calls ``self._right_click_navigate_allowed(layer)``) can
    dispatch through the fake.
    """
    nav = _FakeNav(right_click_enabled=overrides.pop("nav_enabled", True))
    if overrides.pop("nav_missing", False):
        nav = None
    ann = SimpleNamespace(
        _is_closing=overrides.pop("is_closing", False),
        _navigator_widget=nav,
        _refine_state=overrides.pop("refine_state", RefinementOpState()),
        _interp_running=overrides.pop("interp_running", False),
        _split_running=overrides.pop("split_running", False),
        _smooth_running=overrides.pop("smooth_running", False),
        _merge_dialog=overrides.pop("merge_dialog", None),
    )
    if overrides:
        raise ValueError(f"unknown overrides: {sorted(overrides)}")
    # Bind the gate method so the handler closure can call it as
    # ``self._right_click_navigate_allowed(layer)``.
    ann._right_click_navigate_allowed = _gate.__get__(ann)
    ann._rclick_select_alive = _alive.__get__(ann)
    return ann


# =============================================================================
# Tests — _right_click_navigate_allowed gate
# =============================================================================


class TestGateAllAllowed:
    def test_idle_browsing_allows(self):
        ann = _make_ann()
        layer = _FakeLayer([[0, 1], [2, 0]])
        assert _gate(ann, layer) is True


class TestGateLifecycleBlocks:
    def test_closing_blocks(self):
        ann = _make_ann(is_closing=True)
        assert _gate(ann, _FakeLayer([0])) is False

    def test_missing_navigator_blocks(self):
        ann = _make_ann(nav_missing=True)
        assert _gate(ann, _FakeLayer([0])) is False

    def test_toggle_off_blocks(self):
        ann = _make_ann(nav_enabled=False)
        assert _gate(ann, _FakeLayer([0])) is False


class TestGateRefinementBlocks:
    def test_interp_active_blocks(self):
        rs = RefinementOpState()
        rs.interp_active = True
        ann = _make_ann(refine_state=rs)
        assert _gate(ann, _FakeLayer([0])) is False

    def test_interp_refine_mode_blocks(self):
        rs = RefinementOpState()
        rs.interp_active = True  # refine_mode requires interp_active
        rs.interp_refine_mode = True
        ann = _make_ann(refine_state=rs)
        assert _gate(ann, _FakeLayer([0])) is False

    def test_split_active_blocks(self):
        rs = RefinementOpState()
        rs.split_active = True
        ann = _make_ann(refine_state=rs)
        assert _gate(ann, _FakeLayer([0])) is False

    def test_split_previewed_blocks(self):
        rs = RefinementOpState()
        rs.split_active = True
        rs.split_previewed = True
        ann = _make_ann(refine_state=rs)
        assert _gate(ann, _FakeLayer([0])) is False

    def test_split_fragment_refine_blocks(self):
        rs = RefinementOpState()
        rs.split_active = True
        rs.split_fragment_refine = True
        ann = _make_ann(refine_state=rs)
        assert _gate(ann, _FakeLayer([0])) is False

    def test_smooth_active_blocks(self):
        rs = RefinementOpState()
        rs.smooth_active = True
        ann = _make_ann(refine_state=rs)
        assert _gate(ann, _FakeLayer([0])) is False


class TestGateRunningBlocks:
    def test_interp_running_blocks(self):
        ann = _make_ann(interp_running=True)
        assert _gate(ann, _FakeLayer([0])) is False

    def test_split_running_blocks(self):
        ann = _make_ann(split_running=True)
        assert _gate(ann, _FakeLayer([0])) is False

    def test_smooth_running_blocks(self):
        ann = _make_ann(smooth_running=True)
        assert _gate(ann, _FakeLayer([0])) is False


class TestGateDialogBlocks:
    def test_merge_dialog_visible_blocks(self):
        ann = _make_ann(merge_dialog=_FakeMergeDialog(visible=True))
        assert _gate(ann, _FakeLayer([0])) is False

    def test_merge_dialog_hidden_allows(self):
        ann = _make_ann(merge_dialog=_FakeMergeDialog(visible=False))
        assert _gate(ann, _FakeLayer([0])) is True

    def test_merge_dialog_broken_isvisible_treated_as_open(self):
        class _BrokenDialog:
            def isVisible(self):
                raise RuntimeError("widget deleted")
        ann = _make_ann(merge_dialog=_BrokenDialog())
        assert _gate(ann, _FakeLayer([0])) is False


class TestGateLayerMode:
    """All layer modes allow right-click — paint/erase gate removed
    2026-05-10 because subviewer label layers are ALWAYS in paint mode
    by default (subviewers.py:139).  Gating on paint would block right-
    click 100% of the time.  Right-button click is safe in paint mode:
    napari only triggers paint behavior on LEFT-drag."""

    def test_paint_mode_allows(self):
        ann = _make_ann()
        layer = _FakeLayer([[5, 6]], mode="paint")
        assert _gate(ann, layer) is True

    def test_erase_mode_allows(self):
        ann = _make_ann()
        layer = _FakeLayer([[5, 6]], mode="erase")
        assert _gate(ann, layer) is True

    def test_pan_zoom_mode_allows(self):
        ann = _make_ann()
        layer = _FakeLayer([[5, 6]], mode="pan_zoom")
        assert _gate(ann, layer) is True

    def test_pick_mode_allows(self):
        ann = _make_ann()
        layer = _FakeLayer([[5, 6]], mode="pick")
        assert _gate(ann, layer) is True


# =============================================================================
# Tests — _make_subviewer_right_click_handler callback
# =============================================================================


class TestHandlerButtonFilter:
    def test_left_click_no_op(self):
        ann = _make_ann()
        cb = _make_handler(ann)
        layer = _FakeLayer([[5, 6], [7, 8]])
        _drive_click(cb, layer, button=1, position=(0, 0))
        assert ann._navigator_widget.selected == []

    def test_middle_click_no_op(self):
        ann = _make_ann()
        cb = _make_handler(ann)
        layer = _FakeLayer([[5, 6], [7, 8]])
        _drive_click(cb, layer, button=3, position=(0, 0))
        assert ann._navigator_widget.selected == []

    def test_right_click_fires(self):
        ann = _make_ann()
        cb = _make_handler(ann)
        layer = _FakeLayer([[5, 6], [7, 8]])
        _drive_click(cb, layer, button=2, position=(0, 0))
        assert ann._navigator_widget.selected == [5]


class TestHandlerBoundsCheck:
    def test_out_of_bounds_positive_no_op(self):
        ann = _make_ann()
        cb = _make_handler(ann)
        layer = _FakeLayer([[5, 6], [7, 8]])
        _drive_click(cb, layer, button=2, position=(99, 99))
        assert ann._navigator_widget.selected == []

    def test_out_of_bounds_negative_no_op(self):
        ann = _make_ann()
        cb = _make_handler(ann)
        layer = _FakeLayer([[5, 6], [7, 8]])
        _drive_click(cb, layer, button=2, position=(-1, 0))
        assert ann._navigator_widget.selected == []

    def test_wrong_ndim_no_op(self):
        ann = _make_ann()
        cb = _make_handler(ann)
        layer = _FakeLayer([[5, 6], [7, 8]])
        _drive_click(cb, layer, button=2, position=(0, 0, 0))
        assert ann._navigator_widget.selected == []

    def test_position_none_no_op(self):
        ann = _make_ann()
        cb = _make_handler(ann)
        layer = _FakeLayer([[5, 6], [7, 8]])
        _drive_click(cb, layer, button=2, position=None)
        assert ann._navigator_widget.selected == []


class TestHandlerBackgroundSkip:
    def test_label_id_zero_no_op(self):
        ann = _make_ann()
        cb = _make_handler(ann)
        layer = _FakeLayer([[0, 7], [0, 0]])  # (0,0) is background
        _drive_click(cb, layer, button=2, position=(0, 0))
        assert ann._navigator_widget.selected == []

    def test_label_id_nonzero_fires(self):
        ann = _make_ann()
        cb = _make_handler(ann)
        layer = _FakeLayer([[0, 7], [0, 0]])
        _drive_click(cb, layer, button=2, position=(0, 1))
        assert ann._navigator_widget.selected == [7]


class TestHandlerGateRespect:
    def test_paint_mode_does_NOT_gate_off(self):
        """Regression guard 2026-05-10: subviewer label layers are
        ALWAYS in paint mode; gate must not block."""
        ann = _make_ann()
        cb = _make_handler(ann)
        layer = _FakeLayer([[5, 6]], mode="paint")
        _drive_click(cb, layer, button=2, position=(0, 0))
        assert ann._navigator_widget.selected == [5]

    def test_refinement_active_gates_off(self):
        rs = RefinementOpState()
        rs.interp_active = True
        ann = _make_ann(refine_state=rs)
        cb = _make_handler(ann)
        layer = _FakeLayer([[5, 6]])
        _drive_click(cb, layer, button=2, position=(0, 0))
        assert ann._navigator_widget.selected == []


class TestHandlerFractionalPosition:
    def test_rounds_to_nearest_voxel(self):
        ann = _make_ann()
        cb = _make_handler(ann)
        layer = _FakeLayer([[5, 6], [7, 8]])
        # 0.49 rounds to 0, 0.51 rounds to 1
        _drive_click(cb, layer, button=2, position=(0.49, 0.51))
        assert ann._navigator_widget.selected == [6]


class TestHandlerNavigatorMissing:
    def test_no_navigator_does_not_crash(self):
        ann = _make_ann(nav_missing=True)
        cb = _make_handler(ann)
        layer = _FakeLayer([[5, 6]])
        # Should silently no-op
        _drive_click(cb, layer, button=2, position=(0, 0))


class TestNavigatorAPI:
    def test_is_right_click_select_enabled_returns_bool(self):
        """Navigator's polling method returns a bool (not None) so the
        gate doesn't need to wrap it."""
        nav = _FakeNav(right_click_enabled=True)
        assert nav.is_right_click_select_enabled() is True
        nav._enabled = False
        assert nav.is_right_click_select_enabled() is False


# =============================================================================
# Click vs Drag — generator behavior (2026-05-10 zoom-coexistence)
# =============================================================================


class TestHandlerClickVsDrag:
    """The handler is a generator callback.  Right-click + drag in
    napari = built-in viewer zoom binding (lives at viewer level,
    parallel to layer callbacks).  We yield through the drag and only
    fire navigator-focus on RELEASE if the cursor didn't move more
    than the drag-threshold (3 voxels).  This prevents zoom +
    navigate firing together."""

    def test_pure_click_fires_navigator(self):
        ann = _make_ann()
        cb = _make_handler(ann)
        layer = _FakeLayer([[5, 6], [7, 8]])
        _drive_click(cb, layer, button=2, position=(0, 0))
        assert ann._navigator_widget.selected == [5]

    def test_long_drag_skips_navigator(self):
        """Drag of 20 voxels in each axis = ~28 voxel displacement,
        way over the 3-voxel threshold → zoom path, no navigation."""
        ann = _make_ann()
        cb = _make_handler(ann)
        layer = _FakeLayer([[5, 6], [7, 8]])
        _drive_drag(cb, layer, button=2, start=(0, 0), end=(20, 20))
        assert ann._navigator_widget.selected == []

    def test_tiny_wobble_still_clicks(self):
        """1-voxel wobble during the click is below threshold → still
        treated as click → navigator fires.  Label_id is read at END
        position (more accurate when user moved 1 voxel)."""
        ann = _make_ann()
        cb = _make_handler(ann)
        layer = _FakeLayer([[5, 6], [7, 8]])
        # end=(1,0) → array[1][0] = 7
        _drive_drag(cb, layer, button=2, start=(0, 0), end=(1, 0))
        assert ann._navigator_widget.selected == [7]

    def test_drag_with_left_button_not_handled(self):
        """Left-button drag isn't our concern — short-circuit at
        button check before yield."""
        ann = _make_ann()
        cb = _make_handler(ann)
        layer = _FakeLayer([[5, 6], [7, 8]])
        _drive_drag(cb, layer, button=1, start=(0, 0), end=(20, 20))
        assert ann._navigator_widget.selected == []

    def test_drag_with_middle_button_not_handled(self):
        ann = _make_ann()
        cb = _make_handler(ann)
        layer = _FakeLayer([[5, 6], [7, 8]])
        _drive_drag(cb, layer, button=3, start=(0, 0), end=(20, 20))
        assert ann._navigator_widget.selected == []

    def test_drag_just_over_threshold_skips(self):
        """At exactly 4 voxels on one axis (sqrt(16)=4 > threshold 3),
        skip navigation."""
        ann = _make_ann()
        cb = _make_handler(ann)
        layer = _FakeLayer([[5, 6], [7, 8]])
        _drive_drag(cb, layer, button=2, start=(0, 0), end=(4, 0))
        assert ann._navigator_widget.selected == []

    def test_drag_just_under_threshold_clicks(self):
        """At 2 voxels on one axis (under threshold 3), still treated
        as click — small overshoot doesn't lose the click intent.
        End=(2,0) is out-of-bounds for the 2x2 fake → silently no-op
        (bounds check)."""
        ann = _make_ann()
        cb = _make_handler(ann)
        # Use 4x4 so end=(2,0) is in-bounds
        layer = _FakeLayer([[5, 6, 1, 2], [7, 8, 3, 4],
                            [9, 10, 11, 12], [13, 14, 15, 16]])
        # end=(2,0) → array[2][0] = 9
        _drive_drag(cb, layer, button=2, start=(0, 0), end=(2, 0))
        assert ann._navigator_widget.selected == [9]


# Note: paint-suppression tests for the Mechanism B install function
# live in tests/annotator/test_paint_button_filter.py — it wraps
# napari's draw callback at layer-creation time so paint never fires
# on right-click, instead of trying to suppress paint inside the
# right-click handler (which is too late — paint runs before the
# handler can intervene).


# =============================================================================
# 2026-09-07 crash fix — deferral out of vispy's mouse-release dispatch
# =============================================================================


def _drive_click_no_flush(cb, layer, button=2, position=(0, 0, 0)):
    """Same as ``_drive_click`` but leaves the deferred call queued."""
    event = _FakeEvent(button=button, position=position, type="mouse_press")
    gen = cb(layer, event)
    if gen is None:
        return
    try:
        next(gen)
        event.type = "mouse_release"
        next(gen)
    except StopIteration:
        pass


class TestRightClickSelectIsDeferred:
    def test_select_label_not_called_synchronously(self):
        """The crash frame was ``select_label`` INSIDE vispy's release
        dispatch.  The handler must only QUEUE the select."""
        ann = _make_ann()
        cb = _make_handler(ann)
        layer = _FakeLayer([[5, 6], [7, 8]])
        _drive_click_no_flush(cb, layer, position=(0, 0))
        assert ann._navigator_widget.selected == []
        assert len(_DEFERRED) == 1

    def test_fires_once_after_event_loop_turn(self):
        ann = _make_ann()
        cb = _make_handler(ann)
        layer = _FakeLayer([[5, 6], [7, 8]])
        _drive_click_no_flush(cb, layer, position=(0, 0))
        _flush_deferred()
        assert ann._navigator_widget.selected == [5]
        # and exactly once — no re-queue
        assert _DEFERRED == []

    def test_rapid_clicks_coalesce_to_last_label(self):
        """A burst of right-clicks must not run N full isolate/render
        passes — only the newest select survives."""
        ann = _make_ann()
        cb = _make_handler(ann)
        layer = _FakeLayer([[5, 6], [7, 8]])
        _drive_click_no_flush(cb, layer, position=(0, 0))  # 5
        _drive_click_no_flush(cb, layer, position=(0, 1))  # 6
        _drive_click_no_flush(cb, layer, position=(1, 1))  # 8
        assert len(_DEFERRED) == 3
        _flush_deferred()
        assert ann._navigator_widget.selected == [8]

    def test_skips_when_annotator_closing(self):
        ann = _make_ann()
        cb = _make_handler(ann)
        layer = _FakeLayer([[5, 6], [7, 8]])
        _drive_click_no_flush(cb, layer, position=(0, 0))
        ann._is_closing = True
        _flush_deferred()
        assert ann._navigator_widget.selected == []

    def test_skips_when_navigator_gone(self):
        ann = _make_ann()
        cb = _make_handler(ann)
        layer = _FakeLayer([[5, 6], [7, 8]])
        _drive_click_no_flush(cb, layer, position=(0, 0))
        ann._navigator_widget = None
        _flush_deferred()  # must not raise

    def test_skips_when_navigator_destroyed(self):
        ann = _make_ann()
        cb = _make_handler(ann)
        layer = _FakeLayer([[5, 6], [7, 8]])
        _drive_click_no_flush(cb, layer, position=(0, 0))
        ann._navigator_widget._destroyed = True
        _flush_deferred()
        assert ann._navigator_widget.selected == []

    def test_breadcrumbs_emitted(self, monkeypatch):
        crumbs: list = []
        monkeypatch.setattr(_c3d, "_crumb", lambda msg: crumbs.append(msg))
        ann = _make_ann()
        cb = _make_handler(ann)
        layer = _FakeLayer([[5, 6], [7, 8]])
        _drive_click_no_flush(cb, layer, position=(0, 0))
        assert any("DEFERRED" in c for c in crumbs)
        _drive_click_no_flush(cb, layer, position=(0, 1))
        _flush_deferred()
        joined = " | ".join(crumbs)
        assert "skipped-stale" in joined
        assert "fired" in joined
        assert "AFTER-OK" in joined

    def test_breadcrumb_skipped_dead(self, monkeypatch):
        crumbs: list = []
        monkeypatch.setattr(_c3d, "_crumb", lambda msg: crumbs.append(msg))
        ann = _make_ann()
        cb = _make_handler(ann)
        layer = _FakeLayer([[5, 6], [7, 8]])
        _drive_click_no_flush(cb, layer, position=(0, 0))
        ann._navigator_widget._destroyed = True
        _flush_deferred()
        assert any("skipped-dead" in c for c in crumbs)


class TestDeferredFireRechecksClickTimeGate:
    """The deferred fire must re-validate state that can change during the
    one event-loop turn it waits.  A patch switch does NOT delete the
    annotator — it resets + rebuilds the navigator with the NEW patch's
    rows — so a stale select could otherwise match the same label_id in a
    different patch and isolate the wrong cell."""

    def test_rebuild_pending_drops_the_select(self):
        ann = _make_ann()
        cb = _make_handler(ann)
        layer = _FakeLayer([[5, 6], [7, 8]])
        _drive_click_no_flush(cb, layer, position=(0, 0))
        # simulate a patch switch landing between click and fire
        ann._navigator_widget._rebuild_pending = True
        _flush_deferred()
        assert ann._navigator_widget.selected == []

    def test_refine_mode_entered_between_click_and_fire_drops_it(self):
        ann = _make_ann()
        cb = _make_handler(ann)
        layer = _FakeLayer([[5, 6], [7, 8]])
        _drive_click_no_flush(cb, layer, position=(0, 0))
        ann._refine_state.split_active = True
        _flush_deferred()
        assert ann._navigator_widget.selected == []

    def test_toggle_turned_off_between_click_and_fire_drops_it(self):
        ann = _make_ann()
        cb = _make_handler(ann)
        layer = _FakeLayer([[5, 6], [7, 8]])
        _drive_click_no_flush(cb, layer, position=(0, 0))
        ann._navigator_widget._enabled = False
        _flush_deferred()
        assert ann._navigator_widget.selected == []

    def test_normal_fire_still_selects(self):
        ann = _make_ann()
        cb = _make_handler(ann)
        layer = _FakeLayer([[5, 6], [7, 8]])
        _drive_click_no_flush(cb, layer, position=(0, 0))
        _flush_deferred()
        assert ann._navigator_widget.selected == [5]

    def test_gate_breadcrumbs(self, monkeypatch):
        crumbs: list = []
        monkeypatch.setattr(_c3d, "_crumb", lambda msg: crumbs.append(msg))
        ann = _make_ann()
        cb = _make_handler(ann)
        layer = _FakeLayer([[5, 6], [7, 8]])
        _drive_click_no_flush(cb, layer, position=(0, 0))
        ann._refine_state.split_active = True
        _flush_deferred()
        assert any("skipped-gate" in c for c in crumbs)

        crumbs.clear()
        ann2 = _make_ann()
        cb2 = _make_handler(ann2)
        _drive_click_no_flush(cb2, layer, position=(0, 0))
        ann2._navigator_widget._rebuild_pending = True
        _flush_deferred()
        assert any("skipped-rebuild-pending" in c for c in crumbs)
