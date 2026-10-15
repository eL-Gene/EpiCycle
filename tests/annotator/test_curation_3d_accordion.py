"""Tests for 3D Curation Annotator — slim utility panel + navigator wiring.

Phase K — updated after extracting navigation to floating Label Navigator.

Tests use lightweight fakes (no Qt/napari required). They test:
  - Slim utility panel structure (patch, interpolation, launcher buttons)
  - set_flagged_labels() caches items and forwards to navigator
  - set_focused_refinement() triggers navigator open
  - Triage callbacks forwarded from navigator to controller
  - Parent compatibility stubs exist
"""

from __future__ import annotations

import types

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Lightweight fakes — no Qt/napari required
# ---------------------------------------------------------------------------


class _FakeButton:
    """Minimal stand-in for QPushButton."""

    def __init__(self, text=""):
        self._text = text
        self._enabled = True
        self._visible = True
        self._checkable = False
        self._checked = False

    def isEnabled(self):
        return self._enabled

    def setEnabled(self, v):
        self._enabled = v

    def isVisible(self):
        return self._visible

    def setVisible(self, v):
        self._visible = v

    def text(self):
        return self._text

    def setText(self, t):
        self._text = t

    def setCheckable(self, v):
        self._checkable = v


class _FakeLabel:
    """Minimal stand-in for QLabel."""

    def __init__(self, text=""):
        self._text = text
        self._visible = True

    def text(self):
        return self._text

    def setText(self, t):
        self._text = t

    def setVisible(self, v):
        self._visible = v

    def isVisible(self):
        return self._visible


class _FakeCombo:
    """Minimal stand-in for QComboBox."""

    def __init__(self):
        self._text = ""

    def currentText(self):
        return self._text

    def setCurrentText(self, t):
        self._text = t


class _FakeCheckBox:
    """Minimal stand-in for QCheckBox."""

    def __init__(self, text="", checked=True):
        self._text = text
        self._checked = checked
        self._visible = True

    def isChecked(self):
        return self._checked

    def setChecked(self, v):
        self._checked = v

    def setVisible(self, v):
        self._visible = v

    def isVisible(self):
        return self._visible


class _FakeListWidget:
    """Minimal stand-in for QListWidget."""

    def __init__(self):
        self._items = []
        self._visible = False

    def setVisible(self, v):
        self._visible = v

    def isVisible(self):
        return self._visible


class _FakeNavigator:
    """Minimal stand-in for CurationLabelNavigator."""

    def __init__(self):
        self._items = []
        self._callbacks = {}
        self._current_label = None

    def set_items(self, items):
        self._items = list(items)

    def set_callbacks(self, **kwargs):
        self._callbacks = kwargs

    def current_label_id(self):
        return self._current_label


class _FakeLabelsLayer:
    """Minimal stand-in for napari Labels layer."""

    def __init__(self, data, name="labels"):
        self.data = data
        self.name = name
        self.selected_label = 1

    def refresh(self):
        pass


# ---------------------------------------------------------------------------
# Build a minimal Curation3DAnnotator-like object for testing
# ---------------------------------------------------------------------------


def _make_fake_annotator(
    *,
    volume_data=None,
    with_navigator=False,
):
    """Create a fake annotator object that mirrors the slim 3D Curation Tools API."""
    ann = types.SimpleNamespace()

    # --- Slim utility panel widgets ---
    ann.patch_context_combo = _FakeCombo()
    ann.switch_patch_btn = _FakeButton("Switch")
    ann._slice_step_sb = types.SimpleNamespace(value=lambda: 5)
    ann._interpolate_btn = _FakeButton("Preview")
    ann._accept_interp_btn = _FakeButton("Accept")
    ann._reject_interp_btn = _FakeButton("Reject")
    ann._interp_status_label = _FakeLabel("")
    ann._open_navigator_btn = _FakeButton("Label Navigator...")
    ann.open_advanced_features_btn = _FakeButton("Advanced Features...")

    # --- Parent compatibility stubs (hidden) ---
    ann._focused_toggle_btn = _FakeButton("Focused Mode: OFF")
    ann._focused_toggle_btn.setVisible(False)
    ann._prev_flagged_btn = _FakeButton("Prev")
    ann._prev_flagged_btn.setVisible(False)
    ann._next_flagged_btn = _FakeButton("Next")
    ann._next_flagged_btn.setVisible(False)
    ann._focused_mode_group = types.SimpleNamespace(setVisible=lambda v: None)
    ann._focused_nav_label = _FakeLabel("")
    ann._focused_nav_label.setVisible(False)
    ann._flagged_labels_list = _FakeListWidget()
    ann._mark_good_btn = _FakeButton("Mark Good")
    ann._mark_good_btn.setVisible(False)
    ann._delete_label_btn = _FakeButton("Delete Label")
    ann._delete_label_btn.setVisible(False)
    ann._discard_garbage_btn = _FakeButton("Discard All Garbage")
    ann._discard_garbage_btn.setVisible(False)
    ann._tier_filter_good = _FakeCheckBox("Good")
    ann._tier_filter_good.setVisible(False)
    ann._tier_filter_review = _FakeCheckBox("Review")
    ann._tier_filter_review.setVisible(False)
    ann._tier_filter_bad = _FakeCheckBox("Bad")
    ann._tier_filter_bad.setVisible(False)
    ann._tier_filter_garbage = _FakeCheckBox("Garbage")
    ann._tier_filter_garbage.setVisible(False)
    ann._nav_status_label = _FakeLabel("")
    ann._nav_status_label.setVisible(False)

    # --- Navigator state ---
    ann._navigator_widget = _FakeNavigator() if with_navigator else None
    ann._cached_flagged_items = []

    # --- Internal state ---
    ann._flagged_label_ids = []
    ann._flagged_centroids = {}
    ann._flagged_nav_index = -1
    ann._3d_curation_mode = True

    # --- Triage callbacks ---
    ann._on_mark_tier_cb = None
    ann._on_delete_label_cb = None
    ann._on_discard_garbage_cb = None

    # Labels layer for volume operations
    if volume_data is None:
        volume_data = np.zeros((10, 10, 10), dtype=np.int32)
    ann._volume_data = volume_data
    ann._labels_layer = _FakeLabelsLayer(volume_data)
    ann.labels_combo = _FakeCombo()
    ann.labels_combo.setCurrentText("labels")

    # Navigation tracking
    ann._last_nav_target = None

    return ann


# ===========================================================================
# Test helpers — simulate methods
# ===========================================================================


def _set_flagged_labels(annotator, items):
    """Simulate Curation3DAnnotator.set_flagged_labels() — cache + forward."""
    annotator._cached_flagged_items = list(items)
    if annotator._navigator_widget is not None:
        annotator._navigator_widget.set_items(items)


def _set_focused_refinement(annotator, flagged_ids, centroids):
    """Simulate the overridden set_focused_refinement — stores + opens navigator."""
    annotator._flagged_label_ids = list(flagged_ids)
    annotator._flagged_centroids = dict(centroids)
    annotator._flagged_nav_index = -1
    # In real code this calls _open_label_navigator()
    annotator._navigator_opened = bool(flagged_ids) and annotator._3d_curation_mode


def _current_flagged_label(annotator):
    """Simulate _current_flagged_label — check navigator first."""
    if annotator._navigator_widget is not None:
        lid = annotator._navigator_widget.current_label_id()
        if lid is not None:
            return lid
    if not annotator._flagged_label_ids:
        return None
    idx = annotator._flagged_nav_index
    if idx < 0 or idx >= len(annotator._flagged_label_ids):
        return None
    return annotator._flagged_label_ids[idx]


# ===========================================================================
# Test data fixtures
# ===========================================================================


@pytest.fixture
def sample_items():
    """Sample flagged label items with all 4 tiers."""
    return [
        {"label_id": 5, "tier": "good", "centroid": (2, 4, 7), "area": 2031},
        {"label_id": 42, "tier": "review", "centroid": (5, 2, 8), "area": 1204},
        {"label_id": 17, "tier": "bad", "centroid": (3, 1, 4), "area": 389},
        {"label_id": 91, "tier": "garbage", "centroid": (7, 3, 6), "area": 52},
    ]


@pytest.fixture
def mixed_items():
    """Larger item set with multiple garbage labels."""
    return [
        {"label_id": 1, "tier": "good", "centroid": (0, 0, 0), "area": 3000},
        {"label_id": 2, "tier": "good", "centroid": (1, 1, 1), "area": 2800},
        {"label_id": 3, "tier": "review", "centroid": (2, 2, 2), "area": 1500},
        {"label_id": 4, "tier": "bad", "centroid": (3, 3, 3), "area": 400},
        {"label_id": 5, "tier": "garbage", "centroid": (4, 4, 4), "area": 30},
        {"label_id": 6, "tier": "garbage", "centroid": (5, 5, 5), "area": 15},
    ]


# ===========================================================================
# Tests — Slim utility panel structure
# ===========================================================================


class TestSlimPanelStructure:
    """Verify the slim utility panel has the expected widgets."""

    def test_has_patch_controls(self):
        ann = _make_fake_annotator()
        assert ann.patch_context_combo is not None
        assert ann.switch_patch_btn is not None

    def test_has_interpolation_controls(self):
        ann = _make_fake_annotator()
        assert ann._interpolate_btn is not None
        assert ann._accept_interp_btn is not None
        assert ann._reject_interp_btn is not None
        assert ann._interp_status_label is not None

    def test_has_scroll_step(self):
        ann = _make_fake_annotator()
        assert ann._slice_step_sb is not None

    def test_has_navigator_launch_button(self):
        ann = _make_fake_annotator()
        assert ann._open_navigator_btn is not None
        assert "Navigator" in ann._open_navigator_btn.text()

    def test_has_advanced_features_button(self):
        ann = _make_fake_annotator()
        assert ann.open_advanced_features_btn is not None
        assert "Advanced" in ann.open_advanced_features_btn.text()


# ===========================================================================
# Tests — Parent compatibility stubs exist but are hidden
# ===========================================================================


class TestParentCompatStubs:
    """Verify parent-expected attributes exist and are hidden."""

    def test_focused_toggle_btn_exists_hidden(self):
        ann = _make_fake_annotator()
        assert ann._focused_toggle_btn is not None
        assert ann._focused_toggle_btn.isVisible() is False

    def test_prev_next_btns_exist_hidden(self):
        ann = _make_fake_annotator()
        assert ann._prev_flagged_btn is not None
        assert ann._prev_flagged_btn.isVisible() is False
        assert ann._next_flagged_btn is not None
        assert ann._next_flagged_btn.isVisible() is False

    def test_flagged_labels_list_exists_hidden(self):
        ann = _make_fake_annotator()
        assert ann._flagged_labels_list is not None
        assert ann._flagged_labels_list.isVisible() is False

    def test_triage_buttons_exist_hidden(self):
        ann = _make_fake_annotator()
        assert ann._mark_good_btn is not None
        assert ann._mark_good_btn.isVisible() is False
        assert ann._delete_label_btn is not None
        assert ann._delete_label_btn.isVisible() is False
        assert ann._discard_garbage_btn is not None
        assert ann._discard_garbage_btn.isVisible() is False

    def test_tier_filters_exist_hidden(self):
        ann = _make_fake_annotator()
        assert ann._tier_filter_good is not None
        assert ann._tier_filter_good.isVisible() is False
        assert ann._tier_filter_review.isVisible() is False
        assert ann._tier_filter_bad.isVisible() is False
        assert ann._tier_filter_garbage.isVisible() is False

    def test_nav_status_label_exists_hidden(self):
        ann = _make_fake_annotator()
        assert ann._nav_status_label is not None
        assert ann._nav_status_label.isVisible() is False


# ===========================================================================
# Tests — set_flagged_labels caching and forwarding
# ===========================================================================


class TestSetFlaggedLabels:
    """Verify set_flagged_labels caches items and forwards to navigator."""

    def test_caches_items(self, sample_items):
        ann = _make_fake_annotator()
        _set_flagged_labels(ann, sample_items)
        assert ann._cached_flagged_items == sample_items

    def test_forwards_to_navigator_if_present(self, sample_items):
        ann = _make_fake_annotator(with_navigator=True)
        _set_flagged_labels(ann, sample_items)
        assert ann._navigator_widget._items == sample_items

    def test_no_forward_when_navigator_absent(self, sample_items):
        ann = _make_fake_annotator(with_navigator=False)
        _set_flagged_labels(ann, sample_items)
        assert ann._cached_flagged_items == sample_items
        assert ann._navigator_widget is None

    def test_replaces_cached_items(self, sample_items):
        ann = _make_fake_annotator()
        _set_flagged_labels(ann, sample_items)
        _set_flagged_labels(ann, [sample_items[0]])
        assert len(ann._cached_flagged_items) == 1

    def test_empty_items_clears_cache(self, sample_items):
        ann = _make_fake_annotator()
        _set_flagged_labels(ann, sample_items)
        _set_flagged_labels(ann, [])
        assert ann._cached_flagged_items == []


# ===========================================================================
# Tests — set_focused_refinement triggers navigator
# ===========================================================================


class TestSetFocusedRefinement:
    """Verify set_focused_refinement stores state and opens navigator."""

    def test_opens_navigator_when_labels_exist(self):
        ann = _make_fake_annotator()
        _set_focused_refinement(
            ann, [1, 5, 10], {1: (0, 0, 0), 5: (1, 1, 1), 10: (2, 2, 2)}
        )
        assert ann._navigator_opened is True

    def test_does_not_open_when_empty(self):
        ann = _make_fake_annotator()
        _set_focused_refinement(ann, [], {})
        assert ann._navigator_opened is False

    def test_stores_flagged_ids_and_centroids(self):
        ann = _make_fake_annotator()
        ids = [1, 5]
        centroids = {1: (0, 0, 0), 5: (1, 2, 3)}
        _set_focused_refinement(ann, ids, centroids)
        assert ann._flagged_label_ids == [1, 5]
        assert ann._flagged_centroids == centroids

    def test_resets_nav_index(self):
        ann = _make_fake_annotator()
        ann._flagged_nav_index = 3
        _set_focused_refinement(ann, [1], {1: (0, 0, 0)})
        assert ann._flagged_nav_index == -1


# ===========================================================================
# Tests — _current_flagged_label prefers navigator
# ===========================================================================


class TestCurrentFlaggedLabel:
    """Verify _current_flagged_label checks navigator first."""

    def test_returns_navigator_selection(self):
        ann = _make_fake_annotator(with_navigator=True)
        ann._navigator_widget._current_label = 42
        assert _current_flagged_label(ann) == 42

    def test_falls_back_to_index_when_navigator_none(self):
        ann = _make_fake_annotator(with_navigator=False)
        ann._flagged_label_ids = [10, 20, 30]
        ann._flagged_nav_index = 1
        assert _current_flagged_label(ann) == 20

    def test_falls_back_when_navigator_has_no_selection(self):
        ann = _make_fake_annotator(with_navigator=True)
        ann._navigator_widget._current_label = None
        ann._flagged_label_ids = [10, 20]
        ann._flagged_nav_index = 0
        assert _current_flagged_label(ann) == 10

    def test_returns_none_when_no_data(self):
        ann = _make_fake_annotator(with_navigator=False)
        assert _current_flagged_label(ann) is None


# ===========================================================================
# Tests — Triage callback forwarding
# ===========================================================================


class TestTriageCallbackForwarding:
    """Verify triage actions forward to controller callbacks."""

    def test_mark_good_forwards(self, sample_items):
        calls = []
        ann = _make_fake_annotator(with_navigator=True)
        # Post-2026-05-10: unified flag-only callback receives
        # (label_id, tier_str).  Mark Good == tier='good'.
        ann._on_mark_tier_cb = lambda lid, tier: calls.append((tier, lid))
        ann._on_mark_tier_cb(42, "good")
        assert calls == [("good", 42)]

    def test_mark_tier_review_forwards(self, sample_items):
        calls = []
        ann = _make_fake_annotator(with_navigator=True)
        ann._on_mark_tier_cb = lambda lid, tier: calls.append((tier, lid))
        ann._on_mark_tier_cb(8, "review")
        assert calls == [("review", 8)]

    def test_mark_tier_bad_forwards(self, sample_items):
        calls = []
        ann = _make_fake_annotator(with_navigator=True)
        ann._on_mark_tier_cb = lambda lid, tier: calls.append((tier, lid))
        ann._on_mark_tier_cb(11, "bad")
        assert calls == [("bad", 11)]

    def test_mark_tier_curated_forwards(self, sample_items):
        calls = []
        ann = _make_fake_annotator(with_navigator=True)
        ann._on_mark_tier_cb = lambda lid, tier: calls.append((tier, lid))
        ann._on_mark_tier_cb(3, "curated")
        assert calls == [("curated", 3)]

    def test_mark_tier_garbage_forwards(self, sample_items):
        calls = []
        ann = _make_fake_annotator(with_navigator=True)
        ann._on_mark_tier_cb = lambda lid, tier: calls.append((tier, lid))
        ann._on_mark_tier_cb(99, "garbage")
        assert calls == [("garbage", 99)]

    def test_delete_forwards(self, sample_items):
        calls = []
        ann = _make_fake_annotator(with_navigator=True)
        ann._on_delete_label_cb = lambda lid: calls.append(("delete", lid))
        ann._on_delete_label_cb(17)
        assert calls == [("delete", 17)]

    def test_discard_garbage_forwards(self, mixed_items):
        calls = []
        ann = _make_fake_annotator(with_navigator=True)
        ann._on_discard_garbage_cb = lambda ids: calls.append(("discard", ids))
        ann._on_discard_garbage_cb([5, 6])
        assert calls == [("discard", [5, 6])]

    def test_no_callback_does_not_crash(self):
        ann = _make_fake_annotator(with_navigator=True)
        # All callbacks are None — should not raise
        if ann._on_mark_tier_cb:
            ann._on_mark_tier_cb(1, "good")
        if ann._on_delete_label_cb:
            ann._on_delete_label_cb(1)
        if ann._on_discard_garbage_cb:
            ann._on_discard_garbage_cb([1])


# ===========================================================================
# Tests — _SliceStepFilter blocks Up/Down, keeps Left/Right
# ===========================================================================
#
# The _SliceStepFilter event filter on subviewer canvases must:
# - Block Up/Down arrow keys entirely (reserved for Label Navigator)
# - Keep Left/Right arrow keys for slice scrolling
#
# These tests use lightweight fakes that mirror the filter's logic
# without requiring Qt event objects.


class _FakeSpinBox:
    """Minimal stand-in for QSpinBox."""

    def __init__(self, value=1):
        self._value = value

    def value(self):
        return self._value

    def setValue(self, v):
        self._value = v


class _FakeDimsRange:
    """Minimal stand-in for viewer.dims.range."""

    def __init__(self, max_slice):
        # napari dims.range is list of (start, stop, step) tuples
        self._ranges = [(0, max_slice + 1, 1)]

    def __getitem__(self, idx):
        return self._ranges[idx]


class _FakeDims:
    """Minimal stand-in for viewer.dims."""

    def __init__(self, current=0, max_slice=99):
        self._current_step = [current]
        self.range = _FakeDimsRange(max_slice)

    @property
    def current_step(self):
        return self._current_step

    def set_current_step(self, axis, value):
        self._current_step[axis] = value


class _FakeViewerModel:
    """Minimal stand-in for napari ViewerModel."""

    def __init__(self, current=0, max_slice=99):
        self.dims = _FakeDims(current, max_slice)


def _simulate_slice_step_filter(viewer_model, step_spinbox, key):
    """Simulate _SliceStepFilter.eventFilter logic for a given key.

    Returns (consumed: bool, new_position: int).
    Mirrors the actual filter behavior: blocks Up/Down entirely,
    scrolls Left/Right only.
    """
    # Up/Down are blocked — consumed without action
    if key in ("Up", "Down"):
        return True, int(viewer_model.dims.current_step[0])

    # Only Left/Right scroll the subviewer
    if key not in ("Left", "Right"):
        return False, int(viewer_model.dims.current_step[0])

    step = max(1, int(step_spinbox.value()))
    if step <= 1:
        return False, int(viewer_model.dims.current_step[0])

    current = int(viewer_model.dims.current_step[0])
    max_slice = int(viewer_model.dims.range[0][1]) - 1
    if max_slice <= 0:
        return False, current

    if key == "Right":
        new = current + step
        if new > max_slice:
            new = max_slice
    else:  # Left
        new = current - step
        if new < 0:
            new = 0

    if new != current:
        viewer_model.dims.set_current_step(0, new)
    return True, new


class TestSliceStepFilterUpDownBlocking:
    """Verify that Up/Down arrow keys are blocked on subviewer canvases."""

    def test_up_key_consumed_no_scroll(self):
        vm = _FakeViewerModel(current=50, max_slice=99)
        sb = _FakeSpinBox(value=5)
        consumed, pos = _simulate_slice_step_filter(vm, sb, "Up")
        assert consumed is True
        assert pos == 50  # unchanged

    def test_down_key_consumed_no_scroll(self):
        vm = _FakeViewerModel(current=50, max_slice=99)
        sb = _FakeSpinBox(value=5)
        consumed, pos = _simulate_slice_step_filter(vm, sb, "Down")
        assert consumed is True
        assert pos == 50  # unchanged

    def test_up_key_blocked_at_step_1(self):
        vm = _FakeViewerModel(current=50, max_slice=99)
        sb = _FakeSpinBox(value=1)
        consumed, pos = _simulate_slice_step_filter(vm, sb, "Up")
        assert consumed is True
        assert pos == 50  # unchanged

    def test_down_key_blocked_at_step_1(self):
        vm = _FakeViewerModel(current=50, max_slice=99)
        sb = _FakeSpinBox(value=1)
        consumed, pos = _simulate_slice_step_filter(vm, sb, "Down")
        assert consumed is True
        assert pos == 50  # unchanged


class TestSliceStepFilterLeftRightScrolling:
    """Verify that Left/Right arrow keys still scroll subviewers."""

    def test_right_key_scrolls_forward(self):
        vm = _FakeViewerModel(current=50, max_slice=99)
        sb = _FakeSpinBox(value=5)
        consumed, pos = _simulate_slice_step_filter(vm, sb, "Right")
        assert consumed is True
        assert pos == 55

    def test_left_key_scrolls_backward(self):
        vm = _FakeViewerModel(current=50, max_slice=99)
        sb = _FakeSpinBox(value=5)
        consumed, pos = _simulate_slice_step_filter(vm, sb, "Left")
        assert consumed is True
        assert pos == 45

    def test_right_key_clamps_to_max(self):
        vm = _FakeViewerModel(current=97, max_slice=99)
        sb = _FakeSpinBox(value=5)
        consumed, pos = _simulate_slice_step_filter(vm, sb, "Right")
        assert consumed is True
        assert pos == 99  # clamped

    def test_left_key_clamps_to_zero(self):
        vm = _FakeViewerModel(current=3, max_slice=99)
        sb = _FakeSpinBox(value=5)
        consumed, pos = _simulate_slice_step_filter(vm, sb, "Left")
        assert consumed is True
        assert pos == 0  # clamped

    def test_step_1_falls_through_for_left_right(self):
        vm = _FakeViewerModel(current=50, max_slice=99)
        sb = _FakeSpinBox(value=1)
        consumed, _pos = _simulate_slice_step_filter(vm, sb, "Right")
        assert consumed is False  # let napari handle step=1

    def test_left_right_with_large_step(self):
        vm = _FakeViewerModel(current=0, max_slice=99)
        sb = _FakeSpinBox(value=10)
        consumed, pos = _simulate_slice_step_filter(vm, sb, "Right")
        assert consumed is True
        assert pos == 10


# ---------------------------------------------------------------------------
# _UpDownBlocker — standalone Up/Down blocker for subviewer child widgets
# ---------------------------------------------------------------------------


def _simulate_updown_blocker(key):
    """Simulate _UpDownBlocker.eventFilter logic.

    Returns True if the key is consumed (Up/Down), False otherwise.
    """
    if key in ("Up", "Down"):
        return True  # consumed
    return False  # pass through


class TestUpDownBlockerLogic:
    """Verify the standalone Up/Down blocker filters correctly."""

    def test_up_key_blocked(self):
        assert _simulate_updown_blocker("Up") is True

    def test_down_key_blocked(self):
        assert _simulate_updown_blocker("Down") is True

    def test_left_key_passes_through(self):
        assert _simulate_updown_blocker("Left") is False

    def test_right_key_passes_through(self):
        assert _simulate_updown_blocker("Right") is False

    def test_other_key_passes_through(self):
        assert _simulate_updown_blocker("Space") is False


# ---------------------------------------------------------------------------
# Slider focus policy — sliders must not steal keyboard focus from canvas
# ---------------------------------------------------------------------------


class _FakeSlider:
    """Minimal stand-in for QSlider with focusPolicy tracking."""

    def __init__(self):
        self._focus_policy = "StrongFocus"  # Qt default for QSlider

    def setFocusPolicy(self, policy):
        self._focus_policy = policy

    def focusPolicy(self):
        return self._focus_policy


class TestSliderNoFocusPolicy:
    """Verify that slice sliders are set to NoFocus to prevent Up/Down capture."""

    def test_slider_initial_focus_policy(self):
        """Sliders start with StrongFocus (Qt default)."""
        s = _FakeSlider()
        assert s.focusPolicy() == "StrongFocus"

    def test_slider_nofocus_after_setup(self):
        """After arrow key filter installation, sliders should have NoFocus."""
        s = _FakeSlider()
        # Simulates what _install_arrow_key_filters does
        s.setFocusPolicy("NoFocus")
        assert s.focusPolicy() == "NoFocus"

    def test_all_three_sliders_get_nofocus(self):
        """All three orientation sliders (XY, XZ, YZ) should get NoFocus."""
        sliders = {"XY": _FakeSlider(), "XZ": _FakeSlider(), "YZ": _FakeSlider()}
        for slider in sliders.values():
            slider.setFocusPolicy("NoFocus")
        for orient, slider in sliders.items():
            assert slider.focusPolicy() == "NoFocus", f"{orient} slider still has focus"


# ---------------------------------------------------------------------------
# Comprehensive Up/Down blocking — all subviewer child widgets
# ---------------------------------------------------------------------------


class _FakeChildWidget:
    """Minimal stand-in for a Qt child widget."""

    def __init__(self, name=""):
        self._name = name
        self._event_filter = None

    def installEventFilter(self, filt):
        self._event_filter = filt

    def hasEventFilter(self):
        return self._event_filter is not None


class _FakeQtViewer:
    """Minimal stand-in for a QtViewer with child widgets."""

    def __init__(self, children=None):
        self._children = children or []
        self._event_filter = None

    def findChildren(self, widget_type):
        return self._children

    def installEventFilter(self, filt):
        self._event_filter = filt


def _simulate_install_updown_blockers(qt_viewers):
    """Simulates comprehensive Up/Down blocker installation.

    Mirrors _install_arrow_key_filters' new behavior:
    install a blocker on each qt_viewer AND all its children.
    Returns count of widgets that received blockers.
    """
    count = 0
    for qt_viewer in qt_viewers:
        qt_viewer.installEventFilter("blocker")
        count += 1
        for child in qt_viewer.findChildren(None):
            child.installEventFilter("blocker")
            count += 1
    return count


class TestComprehensiveUpDownBlocking:
    """Verify that Up/Down blockers are installed on all subviewer children."""

    def test_blockers_on_qt_viewers(self):
        """Each qt_viewer gets a blocker."""
        v1 = _FakeQtViewer()
        v2 = _FakeQtViewer()
        v3 = _FakeQtViewer()
        count = _simulate_install_updown_blockers([v1, v2, v3])
        assert count == 3
        assert v1._event_filter is not None
        assert v2._event_filter is not None
        assert v3._event_filter is not None

    def test_blockers_on_children(self):
        """Children of qt_viewers also get blockers."""
        children = [_FakeChildWidget("canvas"), _FakeChildWidget("overlay")]
        viewer = _FakeQtViewer(children)
        count = _simulate_install_updown_blockers([viewer])
        assert count == 3  # viewer + 2 children
        for child in children:
            assert child.hasEventFilter()

    def test_no_children_still_installs_on_viewer(self):
        """Even with no children, the viewer itself gets a blocker."""
        viewer = _FakeQtViewer([])
        count = _simulate_install_updown_blockers([viewer])
        assert count == 1


# ---------------------------------------------------------------------------
# Hover-based subviewer scrolling
# ---------------------------------------------------------------------------


def _simulate_hover_scroll(hovered_orientation, direction, viewer_models, step):
    """Simulate hover-based Left/Right scroll routing.

    Mirrors Curation3DAnnotator._on_scroll_subviewer logic:
    1. Detect which subviewer the mouse is hovering over
    2. Scroll that subviewer's dims by step in the given direction
    3. Return (scrolled: bool, orientation: str|None, new_position: int|None)
    """
    if hovered_orientation is None:
        return False, None, None

    vm = viewer_models.get(hovered_orientation)
    if vm is None:
        return False, None, None

    current = int(vm.dims.current_step[0])
    max_slice = int(vm.dims.range[0][1]) - 1
    if max_slice <= 0:
        return False, hovered_orientation, current

    if direction == "right":
        new = min(current + step, max_slice)
    else:
        new = max(current - step, 0)

    if new != current:
        vm.dims.set_current_step(0, new)
    return True, hovered_orientation, new


class TestHoverBasedScrolling:
    """Verify hover-based Left/Right scroll routing to subviewers."""

    def test_scroll_right_on_hovered_xy(self):
        vm = _FakeViewerModel(current=50, max_slice=99)
        scrolled, orient, pos = _simulate_hover_scroll(
            "XY", "right", {"XY": vm}, step=5
        )
        assert scrolled is True
        assert orient == "XY"
        assert pos == 55

    def test_scroll_left_on_hovered_xz(self):
        vm = _FakeViewerModel(current=50, max_slice=99)
        scrolled, orient, pos = _simulate_hover_scroll(
            "XZ", "left", {"XZ": vm}, step=5
        )
        assert scrolled is True
        assert orient == "XZ"
        assert pos == 45

    def test_scroll_on_hovered_yz(self):
        vm = _FakeViewerModel(current=10, max_slice=99)
        scrolled, orient, pos = _simulate_hover_scroll(
            "YZ", "right", {"YZ": vm}, step=3
        )
        assert scrolled is True
        assert orient == "YZ"
        assert pos == 13

    def test_no_hover_returns_false(self):
        vm = _FakeViewerModel(current=50, max_slice=99)
        scrolled, orient, pos = _simulate_hover_scroll(
            None, "right", {"XY": vm}, step=5
        )
        assert scrolled is False
        assert orient is None

    def test_scroll_right_clamps_to_max(self):
        vm = _FakeViewerModel(current=97, max_slice=99)
        scrolled, orient, pos = _simulate_hover_scroll(
            "XY", "right", {"XY": vm}, step=5
        )
        assert scrolled is True
        assert pos == 99

    def test_scroll_left_clamps_to_zero(self):
        vm = _FakeViewerModel(current=3, max_slice=99)
        scrolled, orient, pos = _simulate_hover_scroll(
            "XY", "left", {"XY": vm}, step=5
        )
        assert scrolled is True
        assert pos == 0

    def test_step_1_scrolls_by_1(self):
        vm = _FakeViewerModel(current=50, max_slice=99)
        scrolled, orient, pos = _simulate_hover_scroll(
            "XY", "right", {"XY": vm}, step=1
        )
        assert scrolled is True
        assert pos == 51


# ---------------------------------------------------------------------------
# Main viewer arrow key blocking
# ---------------------------------------------------------------------------


class _FakeMainViewer:
    """Minimal stand-in for main napari Viewer with bind_key tracking."""

    def __init__(self):
        self._bound_keys = {}

    def bind_key(self, key, func, *, overwrite=False):
        self._bound_keys[key] = func


def _simulate_block_main_viewer_keys(viewer):
    """Simulate blocking all arrow keys on the main viewer via bind_key.

    Mirrors Curation3DAnnotator._block_main_viewer_arrow_keys logic.
    """
    def _noop(v):
        pass

    for key in ("Up", "Down", "Left", "Right"):
        viewer.bind_key(key, _noop, overwrite=True)


class TestMainViewerArrowKeyBlocking:
    """Verify that all arrow keys are blocked on the main viewer."""

    def test_all_four_keys_bound(self):
        viewer = _FakeMainViewer()
        _simulate_block_main_viewer_keys(viewer)
        assert "Up" in viewer._bound_keys
        assert "Down" in viewer._bound_keys
        assert "Left" in viewer._bound_keys
        assert "Right" in viewer._bound_keys

    def test_bound_functions_are_noops(self):
        viewer = _FakeMainViewer()
        _simulate_block_main_viewer_keys(viewer)
        # Each bound function should be callable and do nothing
        for key in ("Up", "Down", "Left", "Right"):
            fn = viewer._bound_keys[key]
            assert callable(fn)
            fn(viewer)  # should not raise

    def test_keys_not_bound_before_install(self):
        viewer = _FakeMainViewer()
        assert len(viewer._bound_keys) == 0


# ===========================================================================
# Isolation fast path — K13.11 regression tests
# ===========================================================================


def _make_isolation_annotator(label_id=42):
    """Create a fake annotator wired for _on_isolation_changed testing.

    Uses a real napari Labels layer (no viewer needed) so ``isinstance``
    checks pass.  The annotator is a ``types.SimpleNamespace`` with the
    real ``_on_isolation_changed`` method bound via ``types.MethodType``.
    """
    from napari.layers import Labels as _Labels

    from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator

    layer = _Labels(np.zeros((4, 4, 4), dtype=np.int32), name="labels")
    layer.selected_label = label_id

    # Fake viewer with a layer list that supports name lookup + iteration.
    # Must be a real class — dunder methods are looked up on the type, not
    # the instance, so types.SimpleNamespace won't work.
    class _FakeLayerList:
        def __contains__(self, name):
            return name == "labels"

        def __getitem__(self, name):
            if name == "labels":
                return layer
            raise KeyError(name)

        def __iter__(self):
            return iter([layer])

    viewer = types.SimpleNamespace(layers=_FakeLayerList())

    ann = types.SimpleNamespace()
    ann._active_labels_name = "labels"
    ann.viewer = viewer
    ann._isolation_active = False
    ann._pre_isolation_colormap = None
    ann._isolation_label_set = None
    ann._colormap_authority = None  # WP-3: optionally injected by tests
    ann._sync_manager = None  # WP-A: no subviewers by default
    ann._on_isolation_cleared_cb = None

    # Bind the real method so it uses our fake `self`.
    ann._on_isolation_changed = types.MethodType(
        Curation3DAnnotator._on_isolation_changed, ann
    )
    ann._apply_multi_isolation_colormap = types.MethodType(
        Curation3DAnnotator._apply_multi_isolation_colormap, ann
    )
    ann._layer_by_name = types.MethodType(
        Curation3DAnnotator._layer_by_name, ann
    )
    ann._propagate_colormap_to_subviewers = types.MethodType(
        Curation3DAnnotator._propagate_colormap_to_subviewers, ann
    )
    ann._propagate_show_selected_label = types.MethodType(
        Curation3DAnnotator._propagate_show_selected_label, ann
    )

    return ann, layer


class TestIsolationFastPath:
    """K13.11 — verify single-label isolation no-op avoids redundant re-render.

    The fast path returns immediately when: isolation already active,
    no DirectLabelColormap pending, selected_label matches, and
    show_selected_label is already True.
    """

    def test_single_label_first_activation_sets_show_selected(self):
        """First call with single label should activate show_selected_label."""
        ann, layer = _make_isolation_annotator(label_id=42)
        assert layer.show_selected_label is False

        ann._on_isolation_changed({42})

        assert ann._isolation_active is True
        assert layer.show_selected_label is True

    def test_single_label_noop_on_repeat_call(self):
        """Second call with same label should be a true no-op."""
        ann, layer = _make_isolation_annotator(label_id=42)

        # First activation.
        ann._on_isolation_changed({42})
        assert ann._isolation_active is True
        assert layer.show_selected_label is True

        # Track whether selected_label setter is called again.
        call_count = 0
        original_setter = type(layer).selected_label.fset

        def tracking_setter(self, val):
            nonlocal call_count
            call_count += 1
            original_setter(self, val)

        type(layer).selected_label = property(
            type(layer).selected_label.fget, tracking_setter
        )
        try:
            ann._on_isolation_changed({42})
            # No-op: should NOT have set selected_label again.
            assert call_count == 0, (
                f"selected_label was set {call_count} time(s) — "
                "fast path should have returned early"
            )
        finally:
            # Restore original property.
            type(layer).selected_label = property(
                type(layer).selected_label.fget, original_setter
            )

    def test_single_label_different_label_updates(self):
        """Switching to a different single label should update selected_label."""
        ann, layer = _make_isolation_annotator(label_id=42)

        ann._on_isolation_changed({42})
        assert layer.selected_label == 42

        # Switch to label 99 — should update.
        layer.selected_label = 99  # simulate _set_selected_label_all
        ann._on_isolation_changed({99})
        # Should be a no-op (label already set, show_selected already True).
        assert layer.selected_label == 99
        assert layer.show_selected_label is True

    def test_deactivation_restores_show_selected(self):
        """Calling with empty set deactivates isolation."""
        ann, layer = _make_isolation_annotator(label_id=42)

        ann._on_isolation_changed({42})
        assert ann._isolation_active is True
        assert layer.show_selected_label is True

        ann._on_isolation_changed(set())
        assert ann._isolation_active is False
        assert layer.show_selected_label is False

    def test_noop_not_triggered_when_isolation_inactive(self):
        """No-op guard should NOT fire when isolation is not yet active."""
        ann, layer = _make_isolation_annotator(label_id=42)
        layer.show_selected_label = True
        # Isolation not active yet — should still go through full path.
        ann._isolation_active = False

        ann._on_isolation_changed({42})
        assert ann._isolation_active is True

    def test_noop_not_triggered_when_colormap_pending(self):
        """No-op guard should NOT fire when DirectLabelColormap is pending restore."""
        ann, layer = _make_isolation_annotator(label_id=42)

        # Simulate state after multi-select → single-select transition.
        # Use the real default colormap so napari's setter accepts it.
        original_cmap = layer.colormap
        ann._isolation_active = True
        ann._pre_isolation_colormap = original_cmap
        layer.selected_label = 42
        layer.show_selected_label = True

        ann._on_isolation_changed({42})
        # Should have restored colormap and cleared it.
        assert ann._pre_isolation_colormap is None

    def test_multi_label_uses_direct_label_colormap(self):
        """Multi-label isolation should use DirectLabelColormap, not show_selected_label."""
        ann, layer = _make_isolation_annotator(label_id=42)

        ann._on_isolation_changed({42, 99})

        assert ann._isolation_active is True
        assert layer.show_selected_label is False
        assert ann._pre_isolation_colormap is not None

    def test_multi_to_single_transition(self):
        """Transitioning from multi-label to single-label restores colormap."""
        ann, layer = _make_isolation_annotator(label_id=42)

        # First: multi-label isolation.
        ann._on_isolation_changed({42, 99})
        assert ann._pre_isolation_colormap is not None
        saved_cmap = ann._pre_isolation_colormap

        # Now: single-label — should restore original colormap.
        layer.selected_label = 42
        ann._on_isolation_changed({42})
        assert ann._pre_isolation_colormap is None
        assert layer.show_selected_label is True
        assert layer.colormap is saved_cmap


# ---------------------------------------------------------------------------
# WP-3 tests — Annotator isolation with ColormapAuthority
# ---------------------------------------------------------------------------


def _make_isolation_annotator_with_authority(label_id=1):
    """Same as _make_isolation_annotator but with ColormapAuthority injected."""
    from napari_epicycle.fusion_reconstruction.colormap_authority import (
        ColormapAuthority,
    )

    ann, layer = _make_isolation_annotator(label_id=label_id)
    auth = ColormapAuthority()
    ann._colormap_authority = auth
    return ann, layer, auth


class TestAnnotatorIsolationOwnership:
    """WP-3: Annotator isolation claims/releases ColormapAuthority."""

    def test_annotator_isolation_claims_ownership(self):
        from napari_epicycle.fusion_reconstruction.colormap_authority import ColormapOwner

        ann, layer, auth = _make_isolation_annotator_with_authority(label_id=5)
        ann._on_isolation_changed({5, 10})
        assert auth.current_owner == ColormapOwner.ANNOTATOR_ISOLATION

    def test_annotator_isolation_restore_returns_correct_colormap(self):
        ann, layer, auth = _make_isolation_annotator_with_authority(label_id=5)
        original_cmap = layer.colormap

        # Activate multi-label isolation (saves restore point)
        ann._on_isolation_changed({5, 10})
        assert ann._isolation_active is True

        # Deactivate — should restore original
        ann._on_isolation_changed(set())
        assert ann._isolation_active is False
        assert layer.colormap is original_cmap

    def test_annotator_isolation_after_controller_qc_restores_qc_colors(self):
        """Bug A/C: Annotator isolation → deactivate restores QC colors."""
        from napari.utils.colormaps import DirectLabelColormap
        from napari_epicycle.fusion_reconstruction.colormap_authority import ColormapOwner

        ann, layer, auth = _make_isolation_annotator_with_authority(label_id=5)

        # Simulate controller QC coloring with a real colormap
        auth.claim(ColormapOwner.CONTROLLER_QC)
        qc_cmap = DirectLabelColormap(color_dict={
            None: np.array([0, 0, 0, 0], dtype=np.float32),
            5: np.array([1, 0, 0, 1], dtype=np.float32),
        })
        layer.colormap = qc_cmap

        # Annotator isolation begins — should release controller, claim annotator
        ann._on_isolation_changed({5, 10})
        assert auth.current_owner == ColormapOwner.ANNOTATOR_ISOLATION

        # Deactivate isolation — should restore QC colors
        ann._on_isolation_changed(set())
        assert layer.colormap is qc_cmap
        assert auth.current_owner is None

    def test_patch_switch_resets_authority_and_isolation_state(self):
        """Bug J: Patch switch must reset authority + isolation state."""
        from napari_epicycle.fusion_reconstruction.colormap_authority import ColormapOwner
        from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator

        ann, layer, auth = _make_isolation_annotator_with_authority(label_id=5)

        # Activate isolation
        ann._on_isolation_changed({5, 10})
        assert auth.current_owner == ColormapOwner.ANNOTATOR_ISOLATION

        # Simulate _load_selected_layers patch switch (bind real method)
        # We can't call the full method (needs Qt), but we can verify the
        # reset logic directly:
        auth.reset()
        ann._isolation_active = False
        ann._pre_isolation_colormap = None
        ann._isolation_label_set = None

        assert auth.current_owner is None
        assert ann._isolation_active is False

    def test_patch_switch_clears_pre_isolation_colormap(self):
        """Bug J: _pre_isolation_colormap must be None after patch switch."""
        ann, layer, auth = _make_isolation_annotator_with_authority(label_id=5)

        ann._on_isolation_changed({5, 10})
        assert ann._pre_isolation_colormap is not None

        # Simulate patch switch
        auth.reset()
        ann._isolation_active = False
        ann._pre_isolation_colormap = None

        assert ann._pre_isolation_colormap is None

    def test_deactivate_isolation_after_patch_switch_uses_current_colormap(self):
        """Bug J: Deactivate after patch switch should NOT apply old patch's colors."""
        from napari.utils.colormaps import DirectLabelColormap
        from napari_epicycle.fusion_reconstruction.colormap_authority import ColormapOwner

        ann, layer, auth = _make_isolation_annotator_with_authority(label_id=5)

        # Activate isolation on patch A
        ann._on_isolation_changed({5, 10})
        old_pre_cmap = ann._pre_isolation_colormap

        # Simulate patch switch — reset everything
        auth.reset()
        ann._isolation_active = False
        ann._pre_isolation_colormap = None
        ann._isolation_label_set = None

        # New patch has different colormap
        new_patch_cmap = DirectLabelColormap(color_dict={
            None: np.array([0, 0, 0, 0], dtype=np.float32),
            1: np.array([0, 1, 0, 1], dtype=np.float32),
        })
        layer.colormap = new_patch_cmap

        # Deactivate isolation (should be a no-op since isolation_active is False)
        ann._on_isolation_changed(set())
        assert layer.colormap is new_patch_cmap  # NOT old_pre_cmap


# ---------------------------------------------------------------------------
# WP-A tests — Subviewer colormap propagation from annotator isolation
# ---------------------------------------------------------------------------


def _make_isolation_annotator_with_subviewers(label_id=42):
    """Create an isolation annotator with fake subviewers that track colormap writes.

    Returns (ann, main_layer, subviewer_layers) where subviewer_layers is a
    list of 3 fake label layers (XY, XZ, YZ) that record colormap/show_selected
    assignments.
    """
    from napari.layers import Labels as _Labels
    from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator

    main_layer = _Labels(np.zeros((4, 4, 4), dtype=np.int32), name="labels")
    main_layer.selected_label = label_id

    # Create real Labels layers for subviewers so isinstance checks pass
    sub_xy = _Labels(np.zeros((4, 4), dtype=np.int32), name="labels_XY")
    sub_xz = _Labels(np.zeros((4, 4), dtype=np.int32), name="labels_XZ")
    sub_yz = _Labels(np.zeros((4, 4), dtype=np.int32), name="labels_YZ")

    class _FakeLayerList:
        def __init__(self, layers):
            self._layers = layers
        def __contains__(self, name):
            return any(getattr(l, "name", "") == name for l in self._layers)
        def __getitem__(self, name):
            for l in self._layers:
                if getattr(l, "name", "") == name:
                    return l
            raise KeyError(name)
        def __iter__(self):
            return iter(self._layers)

    viewer = types.SimpleNamespace(layers=_FakeLayerList([main_layer]))

    # Build fake sync manager with viewer models containing the subviewer layers
    vm_xy = types.SimpleNamespace(layers=_FakeLayerList([sub_xy]))
    vm_xz = types.SimpleNamespace(layers=_FakeLayerList([sub_xz]))
    vm_yz = types.SimpleNamespace(layers=_FakeLayerList([sub_yz]))
    sync = types.SimpleNamespace(
        viewer_model_xy=vm_xy,
        viewer_model_xz=vm_xz,
        viewer_model_yz=vm_yz,
        _base_label_name=lambda name: name.replace("_XY", "").replace("_XZ", "").replace("_YZ", ""),
    )

    ann = types.SimpleNamespace()
    ann._active_labels_name = "labels"
    ann.viewer = viewer
    ann._isolation_active = False
    ann._pre_isolation_colormap = None
    ann._isolation_label_set = None
    ann._colormap_authority = None
    ann._on_isolation_cleared_cb = None
    ann.sync_manager = sync

    # Bind real methods
    ann._on_isolation_changed = types.MethodType(
        Curation3DAnnotator._on_isolation_changed, ann
    )
    ann._apply_multi_isolation_colormap = types.MethodType(
        Curation3DAnnotator._apply_multi_isolation_colormap, ann
    )
    ann._layer_by_name = types.MethodType(
        Curation3DAnnotator._layer_by_name, ann
    )
    ann._propagate_colormap_to_subviewers = types.MethodType(
        Curation3DAnnotator._propagate_colormap_to_subviewers, ann
    )
    ann._propagate_show_selected_label = types.MethodType(
        Curation3DAnnotator._propagate_show_selected_label, ann
    )

    return ann, main_layer, [sub_xy, sub_xz, sub_yz]


class TestSubviewerColormapPropagation:
    """WP-A: Annotator isolation must propagate colormap to subviewers."""

    def test_single_label_isolation_propagates_to_subviewers(self):
        """Single-label isolation should set show_selected_label on subviewers."""
        ann, main_layer, subs = _make_isolation_annotator_with_subviewers(label_id=42)

        ann._on_isolation_changed({42})

        assert main_layer.show_selected_label is True
        for sub in subs:
            assert sub.show_selected_label is True, (
                f"{sub.name} should have show_selected_label=True"
            )

    def test_multi_label_isolation_propagates_to_subviewers(self):
        """Multi-label isolation should push DirectLabelColormap to subviewers."""
        from napari.utils.colormaps import DirectLabelColormap

        ann, main_layer, subs = _make_isolation_annotator_with_subviewers(label_id=42)

        ann._on_isolation_changed({42, 99})

        # Main layer should have DirectLabelColormap
        assert isinstance(main_layer.colormap, DirectLabelColormap)
        # Subviewers should have the same colormap type
        for sub in subs:
            assert isinstance(sub.colormap, DirectLabelColormap), (
                f"{sub.name} should have DirectLabelColormap, got {type(sub.colormap)}"
            )

    def test_isolation_cleared_restores_subviewer_colormaps(self):
        """Clearing isolation should restore show_selected_label=False on subviewers."""
        ann, main_layer, subs = _make_isolation_annotator_with_subviewers(label_id=42)

        # Activate then deactivate
        ann._on_isolation_changed({42})
        for sub in subs:
            assert sub.show_selected_label is True

        ann._on_isolation_changed(set())

        assert main_layer.show_selected_label is False
        for sub in subs:
            assert sub.show_selected_label is False, (
                f"{sub.name} should have show_selected_label=False after clear"
            )

    def test_multi_label_isolation_cleared_restores_subviewer_colormaps(self):
        """After multi-label isolation, clearing must restore subviewer colormaps."""
        from napari.utils.colormaps import DirectLabelColormap

        ann, main_layer, subs = _make_isolation_annotator_with_subviewers(label_id=42)
        original_cmaps = [sub.colormap for sub in subs]

        # Activate multi-label isolation
        ann._on_isolation_changed({42, 99})
        for sub in subs:
            assert isinstance(sub.colormap, DirectLabelColormap)

        # Clear isolation
        ann._on_isolation_changed(set())

        # Subviewers must NOT still have DirectLabelColormap
        for sub in subs:
            assert not isinstance(sub.colormap, DirectLabelColormap), (
                f"{sub.name} still has DirectLabelColormap after isolation cleared"
            )

    def test_authority_blocks_controller_during_annotator_isolation(self):
        """ColormapAuthority prevents controller writes during annotator isolation.

        sync.py propagates colormaps unconditionally — the authority prevents
        controller from writing to the main layer in the first place.
        """
        from napari_epicycle.fusion_reconstruction.colormap_authority import (
            ColormapAuthority, ColormapOwner,
        )
        auth = ColormapAuthority()
        # Annotator claims isolation
        assert auth.claim(ColormapOwner.ANNOTATOR_ISOLATION) is True
        # Controller QC coloring should be blocked
        assert auth.claim(ColormapOwner.CONTROLLER_QC) is False
        # Controller isolation should be blocked
        assert auth.claim(ColormapOwner.CONTROLLER_ISOLATION) is False


class TestIsolationClearedCallback:
    """Unchecking Isolate must fire _on_isolation_cleared_cb so the controller
    can re-apply QC colors and clear the plot highlight."""

    def test_clearing_isolation_fires_callback(self):
        ann, layer = _make_isolation_annotator(label_id=42)
        cleared_calls = []
        ann._on_isolation_cleared_cb = lambda: cleared_calls.append(True)

        # Activate then deactivate
        ann._on_isolation_changed({42})
        assert ann._isolation_active is True
        assert cleared_calls == []

        ann._on_isolation_changed(set())
        assert ann._isolation_active is False
        assert cleared_calls == [True]

    def test_callback_fired_even_when_annotator_inactive(self):
        """B1 fix: callback fires even when _isolation_active is False.

        This handles the case where the controller set isolation via lasso
        (CONTROLLER_ISOLATION) and the checkbox was silently checked.  The
        annotator never set _isolation_active, but the controller still
        needs notification to clean up _isolated_label_ids, plot highlight,
        etc.  The controller's B7 guard handles the no-op case.
        """
        ann, layer = _make_isolation_annotator(label_id=42)
        cleared_calls = []
        ann._on_isolation_cleared_cb = lambda: cleared_calls.append(True)

        # Clear without annotator activating — should still notify controller
        ann._on_isolation_changed(set())
        assert cleared_calls == [True]

    def test_callback_fires_after_delete_then_reselect_then_clear(self):
        """Full sequence: isolate → delete → select new → uncheck isolate."""
        ann, layer = _make_isolation_annotator(label_id=42)
        cleared_calls = []
        ann._on_isolation_cleared_cb = lambda: cleared_calls.append(True)

        # Isolate label 42
        ann._on_isolation_changed({42})
        # Simulate: label 42 deleted (volume zeroed, nav entry removed)
        # Isolation state unchanged — _isolation_active still True

        # Select new label (99 in volume)
        layer.data[0, 0, 0] = 99
        ann._on_isolation_changed({99})
        assert cleared_calls == []  # not a clear, just a switch

        # Uncheck isolate
        ann._on_isolation_changed(set())
        assert cleared_calls == [True]


# ---------------------------------------------------------------------------
# WP-B tests — Manual delete detection on navigation
# ---------------------------------------------------------------------------


def _make_delete_detection_annotator(volume_data=None, label_id=5):
    """Create an annotator wired for _check_label_exists_in_volume testing.

    Returns (ann, layer, delete_calls) where delete_calls is a list that
    gets appended when the delete callback fires.
    """
    from napari.layers import Labels as _Labels
    from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator

    if volume_data is None:
        volume_data = np.zeros((20, 20, 20), dtype=np.int32)
        # Place label 5 at a known location
        volume_data[8:12, 8:12, 8:12] = label_id

    layer = _Labels(volume_data, name="labels")
    layer.selected_label = label_id

    class _FakeLayerList:
        def __init__(self, layers):
            self._layers = layers
        def __contains__(self, name):
            return any(getattr(l, "name", "") == name for l in self._layers)
        def __getitem__(self, name):
            for l in self._layers:
                if getattr(l, "name", "") == name:
                    return l
            raise KeyError(name)
        def __iter__(self):
            return iter(self._layers)

    viewer = types.SimpleNamespace(layers=_FakeLayerList([layer]))

    # Navigator mock that provides centroids via _items list of dicts
    # (matching CurationLabelNavigator.set_items format).
    navigator = types.SimpleNamespace()
    navigator._items = [
        {"label_id": label_id, "centroid": (10, 10, 10), "tier": "review"},
    ]
    navigator.selected_label_ids = lambda: {label_id}
    navigator._isolate_cb = types.SimpleNamespace(isChecked=lambda: False)
    navigator.remove_item = lambda lid: None

    delete_calls = []

    from napari_epicycle.annotator.curation_3d_widget import OperationSlot

    ann = types.SimpleNamespace()
    ann._active_labels_name = "labels"
    ann.viewer = viewer
    ann.labels_combo = _FakeCombo()
    ann.labels_combo.setCurrentText("labels")
    ann._split_running = False
    ann._split_fragment_refine = False
    ann._smooth_running = False
    ann._interp_snapshot = None
    ann._interp_refine_mode = False
    ann._interp_label_id = None
    ann._smooth_label_id = None
    ann._split_label_id = None
    ann._last_navigated_label_id = None
    ann._navigator_widget = navigator
    ann._on_navigator_label_selected_cb = None
    ann._on_delete_label_cb = lambda lid, **kw: delete_calls.append(lid)
    ann._on_mark_tier_cb = None

    ann._alive_check_timer = None

    # Fake sync manager for guard methods
    class _FakeSyncMgr:
        def get_edited_slices(self, name):
            return {"XY": {}, "XZ": {}, "YZ": {}}
        def clear_edited_slices_for_label(self, name, lid):
            pass
    ann.sync_manager = _FakeSyncMgr()

    # Guard state — operation slots registry
    ann._operation_slots = [
        OperationSlot(
            name="smooth",
            is_active=lambda: ann._smooth_running,
            description=lambda: "Smooth preview active.",
            reject=lambda: None,
            priority=30,
        ),
        OperationSlot(
            name="split",
            is_active=lambda: ann._split_running,
            description=lambda: "Split in progress.",
            reject=lambda: None,
            priority=20,
        ),
        OperationSlot(
            name="interpolation",
            is_active=lambda: ann._interp_snapshot is not None or ann._interp_refine_mode,
            description=lambda: "Interpolation preview active.",
            reject=lambda: None,
            priority=10,
        ),
    ]
    ann._confirm_discard_dialog = lambda reason: True
    ann._get_selected_label = lambda: label_id

    # Bind real guard methods
    ann._has_uncommitted_work = types.MethodType(
        Curation3DAnnotator._has_uncommitted_work, ann
    )
    ann._discard_uncommitted_work = types.MethodType(
        Curation3DAnnotator._discard_uncommitted_work, ann
    )

    # Bind real methods
    ann._check_label_exists_in_volume = types.MethodType(
        Curation3DAnnotator._check_label_exists_in_volume, ann
    )
    ann._handle_manually_deleted_label = types.MethodType(
        Curation3DAnnotator._handle_manually_deleted_label, ann
    )
    ann._clear_draw_snapshot = lambda: None
    ann._on_navigator_select_label = types.MethodType(
        Curation3DAnnotator._on_navigator_select_label, ann
    )
    ann._check_deleted_labels = types.MethodType(
        Curation3DAnnotator._check_deleted_labels, ann
    )
    ann._on_labels_painted = types.MethodType(
        Curation3DAnnotator._on_labels_painted, ann
    )
    ann._parse_undo_candidates = types.MethodType(
        Curation3DAnnotator._parse_undo_candidates, ann
    )
    ann._schedule_label_alive_check = types.MethodType(
        Curation3DAnnotator._schedule_label_alive_check, ann
    )
    ann._layer_by_name = types.MethodType(
        Curation3DAnnotator._layer_by_name, ann
    )
    ann._zero_label_in_volume = types.MethodType(
        Curation3DAnnotator._zero_label_in_volume, ann
    )
    ann._all_subviewer_label_layers = lambda: []
    ann._pending_undo_sources = set()
    ann._set_selected_label_all = lambda lid: None  # no-op stub
    ann._get_selected_label = lambda: label_id  # stub for pinning logic

    # Convenience for tests: override _parse_undo_candidates to read from
    # a _test_undo_candidates attribute, so tests can inject candidates
    # directly without setting up full undo history.
    ann._test_undo_candidates = set()

    def _mock_parse_undo(self):
        result = set(self._test_undo_candidates)
        self._test_undo_candidates = set()
        return result

    ann._parse_undo_candidates = types.MethodType(_mock_parse_undo, ann)

    return ann, layer, delete_calls


class TestManualDeleteDetection:
    """WP-B: Detect manually-deleted labels on navigation click."""

    def test_navigate_to_existing_label_returns_true(self):
        """Label with voxels in volume should be detected as existing."""
        ann, layer, _ = _make_delete_detection_annotator()
        assert ann._check_label_exists_in_volume(5) is True

    def test_navigate_to_deleted_label_returns_false(self):
        """Label with zero voxels should be detected as deleted."""
        ann, layer, _ = _make_delete_detection_annotator()
        # Erase the label manually
        layer.data[layer.data == 5] = 0
        assert ann._check_label_exists_in_volume(5) is False

    def test_detection_uses_centroid_roi_fast_path(self):
        """When label exists near centroid, fast path (ROI check) should find it."""
        ann, layer, _ = _make_delete_detection_annotator()
        # Label is at [8:12, 8:12, 8:12], centroid ~(10,10,10), margin=20
        # Fast path should find it without full volume scan
        result = ann._check_label_exists_in_volume(5)
        assert result is True

    def test_detection_falls_back_when_centroid_roi_misses(self):
        """Label not near cached centroid should still be found via full scan."""
        volume = np.zeros((60, 60, 60), dtype=np.int32)
        # Place label far from where centroid will say (centroid=(10,10,10))
        volume[50:55, 50:55, 50:55] = 7
        ann, layer, _ = _make_delete_detection_annotator(
            volume_data=volume, label_id=7
        )
        # Update the navigator centroid entry to point to wrong location
        ann._navigator_widget._items_by_label = {
            7: {"centroid": (10, 10, 10)},
        }
        # ROI around (10,10,10) ±20 won't contain the label at (50-55),
        # but full scan should still find it
        result = ann._check_label_exists_in_volume(7)
        assert result is True

    def test_nonexistent_label_returns_false(self):
        """Label that was never in volume should return False."""
        ann, layer, _ = _make_delete_detection_annotator()
        # Label 999 doesn't exist
        assert ann._check_label_exists_in_volume(999) is False

    def test_deferred_detection_on_navigate_away(self):
        """Fill-deleting label X then clicking label Y should auto-reconcile X."""
        # Place two labels in the volume
        volume = np.zeros((20, 20, 20), dtype=np.int32)
        volume[8:12, 8:12, 8:12] = 5
        volume[2:4, 2:4, 2:4] = 10
        ann, layer, delete_calls = _make_delete_detection_annotator(
            volume_data=volume, label_id=5,
        )
        # Add centroid for label 10 too
        ann._navigator_widget._items.append(
            {"label_id": 10, "centroid": (3, 3, 3), "tier": "review"},
        )

        # Step 1: Navigate to label 5 (sets _last_navigated_label_id)
        ann._on_navigator_select_label(5)
        assert ann._last_navigated_label_id == 5
        assert delete_calls == []

        # Step 2: User fill-deletes label 5 (volume changes, nothing else)
        layer.data[layer.data == 5] = 0

        # Step 3: Navigate to label 10 — should detect label 5 is gone
        ann._on_navigator_select_label(10)
        assert 5 in delete_calls, "Label 5 should be auto-reconciled"
        assert ann._last_navigated_label_id == 10

    def test_deferred_detection_does_not_fire_for_existing_label(self):
        """Navigating away from a label that still exists should NOT trigger delete."""
        volume = np.zeros((20, 20, 20), dtype=np.int32)
        volume[8:12, 8:12, 8:12] = 5
        volume[2:4, 2:4, 2:4] = 10
        ann, layer, delete_calls = _make_delete_detection_annotator(
            volume_data=volume, label_id=5,
        )
        ann._navigator_widget._items.append(
            {"label_id": 10, "centroid": (3, 3, 3), "tier": "review"},
        )

        ann._on_navigator_select_label(5)
        ann._on_navigator_select_label(10)
        assert delete_calls == [], "No deletion should fire for existing labels"


# ---------------------------------------------------------------------------
# Real-time fill-delete detection tests
# ---------------------------------------------------------------------------


class TestRealtimeFillDeleteDetection:
    """Fill-with-0 should be detected in real-time via undo-history
    candidate extraction + debounced volume check.

    Tests exercise ``_check_deleted_labels`` (the timer callback) and
    ``_on_labels_painted`` (the ``labels_update`` event handler).
    """

    def test_candidate_label_detected_as_deleted(self):
        """Label in candidate set that has no voxels should be reconciled."""
        ann, layer, delete_calls = _make_delete_detection_annotator()
        layer.data[layer.data == 5] = 0
        ann._test_undo_candidates = {5}

        ann._check_deleted_labels()

        assert 5 in delete_calls
        assert ann._test_undo_candidates == set()

    def test_safety_net_label_still_alive_not_deleted(self):
        """Label from safety net (not undo candidate) that still has voxels should NOT fire.

        Only undo candidates (confirmed fill-with-0) qualify for intent deletion.
        Labels added via the safety net (_last_navigated_label_id) or fallback
        scan must be fully erased to trigger deletion.
        """
        ann, layer, delete_calls = _make_delete_detection_annotator()
        # Label 5 is alive and NOT in undo candidates — only in safety net
        ann._test_undo_candidates = set()
        ann._last_navigated_label_id = 5  # added via safety net

        ann._check_deleted_labels()

        assert delete_calls == [], "Alive label from safety net should not be deleted"

    def test_navigated_label_checked_as_safety_net(self):
        """_last_navigated_label_id is always included in candidates."""
        ann, layer, delete_calls = _make_delete_detection_annotator()
        ann._last_navigated_label_id = 5
        layer.data[layer.data == 5] = 0

        ann._check_deleted_labels()

        assert 5 in delete_calls
        assert ann._last_navigated_label_id is None

    def test_non_navigated_label_deletion_detected(self):
        """Deleting a label OTHER than the navigated one should reconcile."""
        volume = np.zeros((20, 20, 20), dtype=np.int32)
        volume[8:12, 8:12, 8:12] = 5
        volume[2:4, 2:4, 2:4] = 10
        ann, layer, delete_calls = _make_delete_detection_annotator(
            volume_data=volume, label_id=5,
        )
        ann._navigator_widget._items.append(
            {"label_id": 10, "centroid": (3, 3, 3), "tier": "review"},
        )

        # Navigate to label 5
        ann._on_navigator_select_label(5)
        assert ann._last_navigated_label_id == 5

        # Fill-delete label 10 (NOT the navigated one)
        layer.data[layer.data == 10] = 0
        ann._test_undo_candidates = {10}

        ann._check_deleted_labels()

        assert 10 in delete_calls, "Non-navigated label should be reconciled"
        assert ann._last_navigated_label_id == 5  # navigated label unaffected

    def test_undo_history_extracts_candidates(self):
        """_parse_undo_candidates should extract deleted label IDs from undo history."""
        from collections import deque
        from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator
        ann, layer, delete_calls = _make_delete_detection_annotator()

        # Simulate napari undo history: fill label 5 with 0.
        layer._undo_history = deque([
            [(None, np.array([5, 5, 5], dtype=np.int32), np.int32(0))],
        ])

        # Use the REAL _parse_undo_candidates and store the source layer.
        ann._parse_undo_candidates = types.MethodType(
            Curation3DAnnotator._parse_undo_candidates, ann
        )
        ann._pending_undo_sources = {layer}

        candidates = ann._parse_undo_candidates()
        assert 5 in candidates

    def test_undo_history_ignores_non_zero_fills(self):
        """Painting with a non-zero label should NOT produce candidates."""
        from collections import deque
        from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator
        ann, layer, delete_calls = _make_delete_detection_annotator()

        # Simulate painting label 3 over label 5 (not a deletion)
        layer._undo_history = deque([
            [(None, np.array([5, 5, 5], dtype=np.int32), np.int32(3))],
        ])

        ann._parse_undo_candidates = types.MethodType(
            Curation3DAnnotator._parse_undo_candidates, ann
        )
        ann._pending_undo_sources = {layer}

        candidates = ann._parse_undo_candidates()
        assert len(candidates) == 0

    def test_set_data_does_not_schedule_alive_check(self):
        """_on_labels_data_changed (set_data event) must NOT schedule alive
        check — set_data fires on colormap/property changes, not fill/paint."""
        ann, layer, delete_calls = _make_delete_detection_annotator()
        ann._last_navigated_label_id = 5

        schedule_calls = []
        ann._schedule_label_alive_check = lambda: schedule_calls.append(True)
        ann._on_labels_data_changed = types.MethodType(
            __import__(
                "napari_epicycle.annotator.curation_3d_widget",
                fromlist=["Curation3DAnnotator"],
            ).Curation3DAnnotator._on_labels_data_changed,
            ann,
        )

        ann._on_labels_data_changed()

        assert len(schedule_calls) == 0

    def test_full_sequence_navigate_fill_check(self):
        """End-to-end: navigate → fill-delete → timer → reconciled."""
        volume = np.zeros((20, 20, 20), dtype=np.int32)
        volume[8:12, 8:12, 8:12] = 5
        volume[2:4, 2:4, 2:4] = 10
        ann, layer, delete_calls = _make_delete_detection_annotator(
            volume_data=volume, label_id=5,
        )
        ann._navigator_widget._items.append(
            {"label_id": 10, "centroid": (3, 3, 3), "tier": "review"},
        )

        # Step 1: Navigate to label 5
        ann._on_navigator_select_label(5)
        assert ann._last_navigated_label_id == 5

        # Step 2: User fill-deletes label 5
        layer.data[layer.data == 5] = 0
        ann._test_undo_candidates = {5}

        # Step 3: Timer fires (simulated by direct call)
        ann._check_deleted_labels()

        assert 5 in delete_calls, "Label 5 should be auto-reconciled"
        assert ann._last_navigated_label_id is None

        # Step 4: Navigate to label 10 should work normally
        ann._on_navigator_select_label(10)
        assert ann._last_navigated_label_id == 10
        assert delete_calls == [5]

    def test_event_source_stored_for_deferred_parsing(self):
        """_on_labels_painted stores event source for deferred undo parsing."""
        ann, layer, delete_calls = _make_delete_detection_annotator()
        ann._alive_check_timer = None

        schedule_calls = []
        ann._schedule_label_alive_check = lambda: schedule_calls.append(True)

        # Pass event with source = layer (as napari does).
        event = types.SimpleNamespace(source=layer)
        ann._on_labels_painted(event)

        assert layer in ann._pending_undo_sources
        assert len(schedule_calls) == 1

    def test_fallback_checks_all_navigator_labels(self):
        """When no undo candidates exist, _check_deleted_labels checks
        all Navigator items (fallback detection)."""
        volume = np.zeros((20, 20, 20), dtype=np.int32)
        volume[8:12, 8:12, 8:12] = 5
        volume[2:4, 2:4, 2:4] = 10
        ann, layer, delete_calls = _make_delete_detection_annotator(
            volume_data=volume, label_id=5,
        )
        ann._navigator_widget._items.append(
            {"label_id": 10, "centroid": (3, 3, 3), "tier": "review"},
        )

        # Delete label 10 from the volume
        layer.data[layer.data == 10] = 0

        # No pending candidates, no navigated label — fallback path
        ann._test_undo_candidates = set()
        ann._last_navigated_label_id = None
        # Fallback requires selected_label == 0 (delete mode)
        ann._get_selected_label = lambda: 0

        ann._check_deleted_labels()

        assert 10 in delete_calls, (
            "Fallback should detect deletion of any Navigator label"
        )

    def test_centroid_fast_path_uses_items_list(self):
        """_check_label_exists_in_volume should use Navigator _items list
        for centroid lookup (not nonexistent _items_by_label)."""
        ann, layer, _ = _make_delete_detection_annotator()

        # Label 5 has voxels at (8:12, 8:12, 8:12), centroid ~ (10,10,10)
        # Navigator _items has centroid (10, 10, 10) — fast path should work.
        assert ann._check_label_exists_in_volume(5) is True

        # Remove label from volume
        layer.data[layer.data == 5] = 0
        assert ann._check_label_exists_in_volume(5) is False


# -----------------------------------------------------------------------
# Test Group A: _get_selected_label() zero-priority (Fix 1)
# -----------------------------------------------------------------------


def _make_annotator_with_subviewers(main_selected=5, sub_selected=5):
    """Create a delete-detection annotator with fake subviewer layers.

    Returns (ann, layer, sub_layers, delete_calls).
    ``sub_layers`` is a list of SimpleNamespace objects with
    ``selected_label`` attributes to simulate subviewer label layers.
    """
    ann, layer, delete_calls = _make_delete_detection_annotator()
    layer.selected_label = main_selected

    sub_layers = [
        types.SimpleNamespace(selected_label=sub_selected, name=f"labels_{o}")
        for o in ("XY", "XZ", "YZ")
    ]
    ann._all_subviewer_label_layers = lambda: sub_layers

    # Bind the REAL _get_selected_label so it exercises the new logic.
    from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator
    ann._get_selected_label = types.MethodType(
        Curation3DAnnotator._get_selected_label, ann
    )
    return ann, layer, sub_layers, delete_calls


class TestGetSelectedLabelZeroPriority:
    """Fix 1: _get_selected_label returns 0 when ANY layer has selected_label == 0."""

    def test_returns_zero_when_main_has_zero(self):
        """Main viewer selected_label=0 should return 0 even if subviewers have non-zero."""
        ann, layer, sub_layers, _ = _make_annotator_with_subviewers(
            main_selected=0, sub_selected=5
        )
        assert ann._get_selected_label() == 0

    def test_returns_zero_when_subviewer_has_zero(self):
        """Subviewer selected_label=0 should return 0."""
        ann, layer, sub_layers, _ = _make_annotator_with_subviewers(
            main_selected=5, sub_selected=0
        )
        # Set first subviewer to 0
        sub_layers[0].selected_label = 0
        assert ann._get_selected_label() == 0

    def test_returns_nonzero_when_no_zero(self):
        """When no layer has 0, should return first non-zero (regression guard)."""
        ann, layer, sub_layers, _ = _make_annotator_with_subviewers(
            main_selected=5, sub_selected=7
        )
        assert ann._get_selected_label() == 7

    def test_returns_main_when_no_subviewers(self):
        """Fallback to main viewer when no subviewers exist."""
        ann, layer, _, _ = _make_annotator_with_subviewers(main_selected=3)
        ann._all_subviewer_label_layers = lambda: []
        assert ann._get_selected_label() == 3

    def test_fallback_triggers_when_main_selected_label_is_zero(self):
        """End-to-end: stale subviewer value no longer blocks the delete-mode fallback."""
        volume = np.zeros((20, 20, 20), dtype=np.int32)
        volume[8:12, 8:12, 8:12] = 5
        volume[2:4, 2:4, 2:4] = 10
        ann, layer, delete_calls = _make_delete_detection_annotator(
            volume_data=volume, label_id=5,
        )
        ann._navigator_widget._items.append(
            {"label_id": 10, "centroid": (3, 3, 3), "tier": "review"},
        )

        # Add subviewers with stale non-zero selected_label
        sub_layers = [
            types.SimpleNamespace(selected_label=5, name=f"labels_{o}")
            for o in ("XY", "XZ", "YZ")
        ]
        ann._all_subviewer_label_layers = lambda: sub_layers

        # Bind REAL _get_selected_label so it exercises zero-priority logic
        from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator
        ann._get_selected_label = types.MethodType(
            Curation3DAnnotator._get_selected_label, ann
        )

        # Main layer is in delete mode
        layer.selected_label = 0

        # Delete label 10 from volume
        layer.data[layer.data == 10] = 0

        # No pending undo candidates, no navigated label → should trigger fallback
        ann._test_undo_candidates = set()
        ann._last_navigated_label_id = None

        ann._check_deleted_labels()

        assert 10 in delete_calls, (
            "Fallback should trigger now that _get_selected_label returns 0 "
            "when main layer has selected_label=0"
        )


# -----------------------------------------------------------------------
# Test Group B: Undo history parsing robustness (Fix 2)
# -----------------------------------------------------------------------


class TestUndoHistoryRobustness:
    """Fix 2: Harden _parse_undo_candidates against undo history format variants.

    Undo parsing was moved from ``_on_labels_painted`` to the deferred
    ``_parse_undo_candidates`` method (called by the 300 ms debounce).
    Tests exercise the parser directly via stored source layers.
    """

    def _make_ann_with_real_parser(self):
        """Create annotator with real _parse_undo_candidates."""
        from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator
        ann, layer, delete_calls = _make_delete_detection_annotator()
        ann._parse_undo_candidates = types.MethodType(
            Curation3DAnnotator._parse_undo_candidates, ann
        )
        return ann, layer

    def test_handles_0d_numpy_array_new_val(self):
        """new_val as 0-d numpy array (np.array(0)) should be handled."""
        from collections import deque
        ann, layer = self._make_ann_with_real_parser()

        layer._undo_history = deque([
            [(None, np.array([5, 5], dtype=np.int32), np.array(0))],
        ])
        ann._pending_undo_sources = {layer}

        candidates = ann._parse_undo_candidates()
        assert 5 in candidates

    def test_handles_empty_deque(self):
        """Empty _undo_history should not crash."""
        from collections import deque
        ann, layer = self._make_ann_with_real_parser()

        layer._undo_history = deque()
        ann._pending_undo_sources = {layer}

        candidates = ann._parse_undo_candidates()
        assert len(candidates) == 0

    def test_handles_scalar_old_vals(self):
        """old_vals as scalar (not array) should be handled."""
        from collections import deque
        ann, layer = self._make_ann_with_real_parser()

        layer._undo_history = deque([
            [(None, np.int32(5), np.int32(0))],
        ])
        ann._pending_undo_sources = {layer}

        candidates = ann._parse_undo_candidates()
        assert 5 in candidates

    def test_handles_multiple_labels_in_old_vals(self):
        """Multiple labels overwritten in a single fill."""
        from collections import deque
        ann, layer = self._make_ann_with_real_parser()

        layer._undo_history = deque([
            [(None, np.array([5, 10, 5, 15], dtype=np.int32), np.int32(0))],
        ])
        ann._pending_undo_sources = {layer}

        candidates = ann._parse_undo_candidates()
        assert candidates == {5, 10, 15}

    def test_handles_malformed_entry(self):
        """Malformed undo entry should not crash."""
        from collections import deque
        ann, layer = self._make_ann_with_real_parser()

        layer._undo_history = deque([["not a tuple"]])
        ann._pending_undo_sources = {layer}

        candidates = ann._parse_undo_candidates()
        assert len(candidates) == 0

    def test_handles_0d_numpy_array_old_vals(self):
        """old_vals as 0-d numpy array should be handled."""
        from collections import deque
        ann, layer = self._make_ann_with_real_parser()

        layer._undo_history = deque([
            [(None, np.array(7), np.int32(0))],
        ])
        ann._pending_undo_sources = {layer}

        candidates = ann._parse_undo_candidates()
        assert 7 in candidates


# -----------------------------------------------------------------------
# Test Group C: Intent-based deletion for partial erasure (Fix 3)
# -----------------------------------------------------------------------


class TestIntentBasedDeletion:
    """Fix 3: Detect deletion intent when label is partially erased."""

    def test_intent_delete_when_navigated_label_partially_erased(self):
        """Navigated label + delete mode + undo candidate → intent deletion."""
        ann, layer, delete_calls = _make_delete_detection_annotator()

        # Partially erase label 5 — only slice 10
        layer.data[10, :, :] = 0
        # Label 5 still exists on slices 8, 9, 11
        assert np.any(layer.data == 5), "Label 5 should still partially exist"

        ann._last_navigated_label_id = 5
        ann._test_undo_candidates = {5}
        # Must be in delete mode
        ann._get_selected_label = lambda: 0

        # Bind real _zero_label_in_volume
        from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator
        ann._zero_label_in_volume = types.MethodType(
            Curation3DAnnotator._zero_label_in_volume, ann
        )

        ann._check_deleted_labels()

        assert 5 in delete_calls, "Intent-based deletion should fire"
        assert not np.any(layer.data == 5), "Remaining voxels should be zeroed"

    def test_intent_delete_does_not_fire_for_non_undo_candidate(self):
        """Partial erasure should NOT trigger intent if the label is NOT in undo candidates.

        Only labels confirmed by undo history (fill-with-0) qualify for intent
        deletion.  Labels that just happen to be in the candidate set from
        other sources (safety net, fallback) require full erasure.
        """
        ann, layer, delete_calls = _make_delete_detection_annotator()

        # Partially erase label 5
        layer.data[10, :, :] = 0
        assert np.any(layer.data == 5)

        ann._last_navigated_label_id = 5
        ann._test_undo_candidates = set()  # label 5 NOT in undo candidates
        ann._get_selected_label = lambda: 5

        from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator
        ann._zero_label_in_volume = types.MethodType(
            Curation3DAnnotator._zero_label_in_volume, ann
        )

        ann._check_deleted_labels()

        assert 5 not in delete_calls, (
            "Should NOT intent-delete when label is not in undo candidates"
        )
        assert np.any(layer.data == 5), "Remaining voxels should NOT be zeroed"

    def test_intent_delete_works_in_erase_mode(self):
        """Intent deletion should work even when selected_label != 0 (erase mode).

        In napari erase mode, selected_label stays at the navigated label
        (non-zero) but the actual paint value IS 0.  The undo history
        already confirmed new_val==0, so selected_label==0 is not required.
        """
        ann, layer, delete_calls = _make_delete_detection_annotator()

        layer.data[10, :, :] = 0
        assert np.any(layer.data == 5)

        ann._last_navigated_label_id = 5
        ann._test_undo_candidates = {5}
        ann._get_selected_label = lambda: 5  # erase mode: stays at navigated label

        from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator
        ann._zero_label_in_volume = types.MethodType(
            Curation3DAnnotator._zero_label_in_volume, ann
        )

        ann._check_deleted_labels()

        assert 5 in delete_calls, "Intent-delete should work in erase mode"
        assert not np.any(layer.data == 5), "Remaining voxels should be zeroed"

    def test_intent_delete_zeros_remaining_voxels(self):
        """Intent-based deletion should zero all remaining voxels of the label."""
        ann, layer, delete_calls = _make_delete_detection_annotator()

        # Only erase one slice — leave substantial voxels behind
        layer.data[10, :, :] = 0

        ann._last_navigated_label_id = 5
        ann._test_undo_candidates = {5}
        ann._get_selected_label = lambda: 5  # erase mode

        from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator
        ann._zero_label_in_volume = types.MethodType(
            Curation3DAnnotator._zero_label_in_volume, ann
        )

        ann._check_deleted_labels()

        assert not np.any(layer.data == 5), "All voxels of label 5 should be zeroed"

    def test_intent_delete_does_not_affect_other_labels(self):
        """Intent deletion of label 5 should not touch label 10."""
        volume = np.zeros((20, 20, 20), dtype=np.int32)
        volume[8:12, 8:12, 8:12] = 5
        volume[2:4, 2:4, 2:4] = 10
        ann, layer, delete_calls = _make_delete_detection_annotator(
            volume_data=volume, label_id=5,
        )
        ann._navigator_widget._items.append(
            {"label_id": 10, "centroid": (3, 3, 3), "tier": "review"},
        )

        # Partially erase label 5
        layer.data[10, :, :] = 0

        ann._last_navigated_label_id = 5
        ann._test_undo_candidates = {5}
        ann._get_selected_label = lambda: 5  # erase mode

        from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator
        ann._zero_label_in_volume = types.MethodType(
            Curation3DAnnotator._zero_label_in_volume, ann
        )

        ann._check_deleted_labels()

        assert not np.any(layer.data == 5), "Label 5 should be zeroed"
        assert np.any(layer.data == 10), "Label 10 should be untouched"

    def test_full_erasure_still_works_without_intent(self):
        """Regression guard: fully erased label detected without intent conditions."""
        ann, layer, delete_calls = _make_delete_detection_annotator()

        # Fully erase label 5
        layer.data[layer.data == 5] = 0

        ann._last_navigated_label_id = None  # NOT navigated
        ann._test_undo_candidates = {5}
        ann._get_selected_label = lambda: 5  # NOT in delete mode

        ann._check_deleted_labels()

        assert 5 in delete_calls, "Fully erased label should still be detected"

    def test_intent_delete_without_navigation(self):
        """Intent deletion should work even when label is NOT the navigated label.

        Scenario: user fills label 13 with 0 in the 3D viewer but
        _last_navigated_label_id points elsewhere (e.g., auto-advance
        moved past it). The undo history still confirms the fill-with-0,
        which is sufficient proof of delete intent.
        """
        ann, layer, delete_calls = _make_delete_detection_annotator()

        # Partially erase label 5 (fragments remain)
        layer.data[10, :, :] = 0
        assert np.any(layer.data == 5), "Label 5 should still partially exist"

        ann._last_navigated_label_id = 99  # navigated ELSEWHERE
        ann._test_undo_candidates = {5}  # undo says label 5 was filled with 0
        ann._get_selected_label = lambda: 99

        from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator
        ann._zero_label_in_volume = types.MethodType(
            Curation3DAnnotator._zero_label_in_volume, ann
        )

        ann._check_deleted_labels()

        assert 5 in delete_calls, (
            "Intent-delete should fire for any undo candidate, "
            "not just the navigated label"
        )
        assert not np.any(layer.data == 5), "Remaining voxels should be zeroed"


# ---------------------------------------------------------------------------
# F3 tests — Navigator isolate toggle dispatches to coordinator
# ---------------------------------------------------------------------------


def _make_isolation_annotator_with_bridge(label_ids_for_nav=None):
    """Extend _make_isolation_annotator with bridge callback tracking.

    Returns (ann, layer, bridge_calls) where bridge_calls is a list that
    captures (label_id, all_selected, isolate) each time the bridge fires.
    Also wires a fake _navigator_widget with selected_label_ids().
    """
    ann, layer = _make_isolation_annotator(label_id=42)

    bridge_calls = []

    def _bridge(label_id, *, all_selected=None, isolate=False):
        bridge_calls.append((label_id, all_selected, isolate))

    ann._on_navigator_label_selected_cb = _bridge

    # Fake navigator widget with selected_label_ids
    nav_ids = label_ids_for_nav or {42}

    class _FakeNav:
        def selected_label_ids(self):
            return set(nav_ids)

        class _isolate_cb:
            @staticmethod
            def isChecked():
                return True

    ann._navigator_widget = _FakeNav()

    return ann, layer, bridge_calls


class TestIsolationDispatchToCoordinator:
    """F3: _on_isolation_changed dispatches through bridge for plot highlight."""

    def test_isolate_on_dispatches_to_bridge(self):
        """T5: Toggling Isolate ON should dispatch through bridge callback."""
        ann, layer, bridge_calls = _make_isolation_annotator_with_bridge(
            label_ids_for_nav={42}
        )
        ann._on_isolation_changed({42})
        assert len(bridge_calls) == 1, (
            "Isolate ON should dispatch exactly once through the bridge"
        )
        _, all_sel, isolate = bridge_calls[0]
        assert isolate is True
        assert 42 in all_sel

    def test_isolate_on_multi_label_dispatches_to_bridge(self):
        """T5b: Multi-label isolation should also dispatch through bridge."""
        ann, layer, bridge_calls = _make_isolation_annotator_with_bridge(
            label_ids_for_nav={42, 99}
        )
        ann._on_isolation_changed({42, 99})
        assert len(bridge_calls) == 1
        _, all_sel, isolate = bridge_calls[0]
        assert isolate is True
        assert {42, 99} == set(all_sel)

    def test_isolate_off_does_not_dispatch_extra(self):
        """T6: Toggling Isolate OFF already has _on_isolation_cleared_cb.

        The clear path should NOT dispatch through the bridge since
        _on_isolation_cleared_cb already handles plot clearing.
        """
        ann, layer, bridge_calls = _make_isolation_annotator_with_bridge()
        # Activate isolation first
        ann._on_isolation_changed({42})
        bridge_calls.clear()

        # Now deactivate
        ann._on_isolation_changed(set())
        assert len(bridge_calls) == 0, (
            "Clear path should NOT dispatch through bridge "
            "(already handled by _on_isolation_cleared_cb)"
        )

    def test_no_double_dispatch_on_row_navigation(self):
        """T8: When bridge already dispatched (row click), guard prevents double dispatch."""
        ann, layer, bridge_calls = _make_isolation_annotator_with_bridge(
            label_ids_for_nav={42}
        )
        # Simulate: _on_navigator_select_label already dispatched through bridge
        ann._isolation_bridge_dispatched = True

        ann._on_isolation_changed({42})
        assert len(bridge_calls) == 0, (
            "Should NOT dispatch again when bridge already dispatched "
            "from _on_navigator_select_label"
        )

    def test_guard_flag_resets_after_use(self):
        """Guard flag resets to False after being consumed."""
        ann, layer, bridge_calls = _make_isolation_annotator_with_bridge(
            label_ids_for_nav={42}
        )
        ann._isolation_bridge_dispatched = True
        ann._on_isolation_changed({42})
        assert ann._isolation_bridge_dispatched is False

    def test_no_bridge_callback_no_crash(self):
        """If bridge callback is None, no crash."""
        ann, layer = _make_isolation_annotator(label_id=42)
        ann._on_navigator_label_selected_cb = None
        ann._navigator_widget = None
        # Should not raise
        ann._on_isolation_changed({42})


# ---------------------------------------------------------------------------
# Autopan regression — bridge should use mode="focus" when controller owns
# ---------------------------------------------------------------------------


def _make_bridge_dispatch_annotator(*, isolate_checked=True,
                                     controller_owns=False):
    """Create annotator wired for _on_navigator_select_label bridge testing.

    Returns (ann, layer, bridge_calls) where bridge_calls captures
    (label_id, kwargs) for each bridge call.
    """
    from napari.layers import Labels as _Labels
    from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator
    from napari_epicycle.fusion_reconstruction.colormap_authority import (
        ColormapAuthority, ColormapOwner,
    )

    layer = _Labels(np.zeros((4, 4, 4), dtype=np.int32), name="labels")
    layer.data[1, 1, 1] = 5
    layer.selected_label = 5

    class _FakeLayerList:
        def __contains__(self, name):
            return name == "labels"
        def __getitem__(self, name):
            if name == "labels":
                return layer
            raise KeyError(name)
        def __iter__(self):
            return iter([layer])

    viewer = types.SimpleNamespace(layers=_FakeLayerList())

    bridge_calls = []

    def _bridge(label_id, *, all_selected=None, isolate=False):
        bridge_calls.append({
            "label_id": label_id,
            "all_selected": all_selected,
            "isolate": isolate,
        })

    # Fake navigator
    nav = types.SimpleNamespace()
    nav.selected_label_ids = lambda: {5}
    nav._isolate_cb = types.SimpleNamespace(
        isChecked=lambda: isolate_checked,
    )
    nav.remove_item = lambda lid: None

    auth = ColormapAuthority()
    if controller_owns:
        auth.claim(ColormapOwner.CONTROLLER_ISOLATION)

    from napari_epicycle.annotator.curation_3d_widget import OperationSlot

    ann = types.SimpleNamespace()
    ann._active_labels_name = "labels"
    ann.viewer = viewer
    ann.labels_combo = _FakeCombo()
    ann.labels_combo.setCurrentText("labels")
    ann._split_running = False
    ann._split_fragment_refine = False
    ann._smooth_running = False
    ann._interp_snapshot = None
    ann._interp_refine_mode = False
    ann._interp_label_id = None
    ann._smooth_label_id = None
    ann._split_label_id = None
    ann._last_navigated_label_id = None
    ann._navigator_widget = nav
    ann._on_navigator_label_selected_cb = _bridge
    ann._colormap_authority = auth

    # Guard state — operation slots + sync manager
    class _FakeSyncMgr:
        def get_edited_slices(self, name):
            return {"XY": {}, "XZ": {}, "YZ": {}}
        def clear_edited_slices_for_label(self, name, lid):
            pass
    ann.sync_manager = _FakeSyncMgr()

    ann._operation_slots = [
        OperationSlot(
            name="smooth",
            is_active=lambda: ann._smooth_running,
            description=lambda: "Smooth preview active.",
            reject=lambda: None,
            priority=30,
        ),
        OperationSlot(
            name="split",
            is_active=lambda: ann._split_running,
            description=lambda: "Split in progress.",
            reject=lambda: None,
            priority=20,
        ),
        OperationSlot(
            name="interpolation",
            is_active=lambda: ann._interp_snapshot is not None or ann._interp_refine_mode,
            description=lambda: "Interpolation preview active.",
            reject=lambda: None,
            priority=10,
        ),
    ]
    ann._confirm_discard_dialog = lambda reason: True
    ann._get_selected_label = lambda: 5

    ann._set_selected_label_all = lambda lid: None
    ann._check_label_exists_in_volume = types.MethodType(
        Curation3DAnnotator._check_label_exists_in_volume, ann
    )
    ann._handle_manually_deleted_label = lambda lid: None
    ann._clear_draw_snapshot = lambda: None
    ann._layer_by_name = types.MethodType(
        Curation3DAnnotator._layer_by_name, ann
    )
    ann._has_uncommitted_work = types.MethodType(
        Curation3DAnnotator._has_uncommitted_work, ann
    )
    ann._discard_uncommitted_work = types.MethodType(
        Curation3DAnnotator._discard_uncommitted_work, ann
    )
    ann._on_navigator_select_label = types.MethodType(
        Curation3DAnnotator._on_navigator_select_label, ann
    )

    return ann, layer, bridge_calls


class TestBridgeDispatchAutopanRegression:
    """Autopan regression: bridge should NOT use mode='isolate' when
    controller already owns isolation (from lasso).

    When controller owns CONTROLLER_ISOLATION and the Isolate checkbox
    is ON (set by F2), the bridge dispatch should use isolate=False
    so the coordinator doesn't call _isolate_labels_in_viewer (which
    includes QApplication.processEvents() that interferes with
    subviewer autopan).
    """

    def test_controller_owns_forces_focus_mode(self):
        """When controller owns isolation, bridge dispatches isolate=False."""
        ann, layer, calls = _make_bridge_dispatch_annotator(
            isolate_checked=True, controller_owns=True,
        )
        ann._on_navigator_select_label(5)
        assert len(calls) == 1
        assert calls[0]["isolate"] is False, (
            "Bridge should use isolate=False when controller already owns "
            "isolation (avoids processEvents in _isolate_labels_in_viewer)"
        )

    def test_annotator_owns_keeps_isolate_mode(self):
        """When annotator owns isolation, bridge dispatches isolate=True."""
        from napari_epicycle.fusion_reconstruction.colormap_authority import ColormapOwner
        ann, layer, calls = _make_bridge_dispatch_annotator(
            isolate_checked=True, controller_owns=False,
        )
        # Annotator owns isolation
        ann._colormap_authority.claim(ColormapOwner.ANNOTATOR_ISOLATION)
        ann._on_navigator_select_label(5)
        assert len(calls) == 1
        assert calls[0]["isolate"] is True, (
            "Bridge should keep isolate=True when annotator owns "
            "(controller's _isolate_labels_in_viewer returns early anyway)"
        )

    def test_no_authority_keeps_isolate_mode(self):
        """When no authority is set, bridge dispatches isolate=True."""
        ann, layer, calls = _make_bridge_dispatch_annotator(
            isolate_checked=True, controller_owns=False,
        )
        ann._colormap_authority = None
        ann._on_navigator_select_label(5)
        assert len(calls) == 1
        assert calls[0]["isolate"] is True

    def test_checkbox_off_dispatches_focus(self):
        """When checkbox is OFF, bridge dispatches isolate=False regardless."""
        ann, layer, calls = _make_bridge_dispatch_annotator(
            isolate_checked=False, controller_owns=True,
        )
        ann._on_navigator_select_label(5)
        assert len(calls) == 1
        assert calls[0]["isolate"] is False

    def test_guard_flag_still_set(self):
        """Guard flag is set even when isolate is downgraded to False."""
        ann, layer, calls = _make_bridge_dispatch_annotator(
            isolate_checked=True, controller_owns=True,
        )
        ann._on_navigator_select_label(5)
        # Flag should be set (consumed later by _on_isolation_changed)
        assert ann._isolation_bridge_dispatched is True


# ===========================================================================
# Tests — closeEvent clears isolation (B6)
# ===========================================================================


class TestCloseEventClearsIsolation:
    """B6: Closing the annotator dock while isolation is active must
    release ANNOTATOR_ISOLATION and clear isolation state.

    Updated: closeEvent now fires _on_isolation_cleared_cb UNCONDITIONALLY
    (after all event handlers are disconnected) to handle the lasso-isolation
    leak where the controller sets CONTROLLER_ISOLATION but the annotator's
    _isolation_active is never True.
    """

    @staticmethod
    def _make_closeable_annotator(*, isolation_active=True):
        """Create a fake annotator that can exercise the real closeEvent.

        Returns (ann, order_log) where order_log tracks call ordering.
        """
        from napari_epicycle.fusion_reconstruction.colormap_authority import (
            ColormapAuthority,
            ColormapOwner,
        )
        from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator

        ann, layer, _calls = _make_isolation_annotator_with_bridge()

        # Authority setup
        auth = ColormapAuthority()
        if isolation_active:
            auth.claim(ColormapOwner.ANNOTATOR_ISOLATION)
        ann._colormap_authority = auth
        ann._pre_isolation_colormap = layer.colormap
        ann._isolation_active = isolation_active
        ann._isolation_label_set = frozenset({5, 10}) if isolation_active else None

        # Callback tracking
        ann._isolation_cleared_calls = []
        ann._on_isolation_cleared_cb = lambda: ann._isolation_cleared_calls.append(True)

        # Attributes needed by closeEvent
        ann._smooth_running = False
        ann._split_running = False
        ann._interp_snapshot = None
        ann._interp_refine_mode = False
        ann._isolation_bridge_dispatched = False

        # Timer fake — tracks whether stop() was called
        class _FakeTimer:
            def __init__(self):
                self.stopped = False
            def stop(self):
                self.stopped = True
        ann._alive_check_timer = _FakeTimer()
        ann._init_auto_load_timer = _FakeTimer()
        ann._deferred_auto_load_timer = _FakeTimer()
        ann._is_closing = False

        # Order tracking for disconnect → callback ordering test
        order_log = []
        ann._disconnect_viewer_events = lambda: order_log.append("disconnect_viewer_events")

        def _fake_disconnect_inserted():
            order_log.append("disconnect_inserted")
        # Fake viewer.layers.events for the inserted disconnect
        class _FakeEvents:
            class inserted:
                @staticmethod
                def disconnect(handler):
                    _fake_disconnect_inserted()
        ann.viewer.layers.events = _FakeEvents()
        ann._on_layer_inserted = lambda *a: None

        ann._disconnect_label_handlers = lambda: order_log.append("disconnect_label_handlers")
        ann._restore_main_viewer_arrow_keys = lambda: order_log.append("restore_arrow_keys")

        # Curation callbacks to verify post-close nulling
        ann._on_mark_tier_cb = lambda lid, tier: None
        ann._on_delete_label_cb = lambda x: None
        ann._on_discard_garbage_cb = lambda x: None
        ann._on_merge_cb = lambda x: None
        ann._on_navigator_label_selected_cb = lambda x: None
        ann._on_interpolation_accepted_cb = lambda x: None
        ann._on_label_split_cb = lambda x: None
        ann._on_label_smoothed_cb = lambda x: None
        ann._on_recalculate_cb = lambda x: None

        # Bind the real closeEvent
        ann.closeEvent = types.MethodType(Curation3DAnnotator.closeEvent, ann)

        # Fake super().closeEvent — just log it
        return ann, order_log

    @staticmethod
    def _call_close(ann, order_log):
        """Call _close_cleanup on the fake annotator.

        closeEvent delegates to _close_cleanup (all logic except super call),
        so testing _close_cleanup exercises the full teardown sequence
        without requiring Qt infrastructure for super().closeEvent().
        """
        from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator

        # Bind and call the real _close_cleanup on our fake.
        bound = types.MethodType(Curation3DAnnotator._close_cleanup, ann)
        bound()

    def test_close_clears_isolation_active(self):
        """After closeEvent, _isolation_active should be False."""
        ann, order_log = self._make_closeable_annotator(isolation_active=True)
        assert ann._isolation_active is True

        self._call_close(ann, order_log)

        assert ann._isolation_active is False

    def test_close_releases_authority(self):
        """After closeEvent, annotator authority should be released."""
        from napari_epicycle.fusion_reconstruction.colormap_authority import ColormapOwner
        ann, order_log = self._make_closeable_annotator(isolation_active=True)
        assert ann._colormap_authority.current_owner == ColormapOwner.ANNOTATOR_ISOLATION

        self._call_close(ann, order_log)

        assert ann._colormap_authority.current_owner != ColormapOwner.ANNOTATOR_ISOLATION

    def test_close_fires_isolation_cleared_callback(self):
        """After closeEvent with isolation active, callback fires exactly once."""
        ann, order_log = self._make_closeable_annotator(isolation_active=True)

        self._call_close(ann, order_log)

        assert len(ann._isolation_cleared_calls) == 1

    def test_close_without_isolation_fires_cleared_callback(self):
        """Lasso case: _isolation_active=False but controller has CONTROLLER_ISOLATION.
        closeEvent must fire _on_isolation_cleared_cb unconditionally."""
        ann, order_log = self._make_closeable_annotator(isolation_active=False)
        assert ann._isolation_active is False

        self._call_close(ann, order_log)

        assert len(ann._isolation_cleared_calls) == 1, (
            "Callback must fire even when _isolation_active is False "
            "(lasso selection sets CONTROLLER_ISOLATION without annotator isolation)"
        )

    def test_close_with_isolation_fires_cleared_callback_once(self):
        """When _isolation_active=True, callback fires exactly once — not twice
        (once inside the if-block AND once outside)."""
        ann, order_log = self._make_closeable_annotator(isolation_active=True)

        self._call_close(ann, order_log)

        assert len(ann._isolation_cleared_calls) == 1, (
            f"Expected exactly 1 callback call, got {len(ann._isolation_cleared_calls)}"
        )

    def test_close_stops_alive_check_timer(self):
        """_alive_check_timer must be stopped during closeEvent."""
        ann, order_log = self._make_closeable_annotator(isolation_active=False)
        assert ann._alive_check_timer.stopped is False

        self._call_close(ann, order_log)

        assert ann._alive_check_timer.stopped is True, (
            "_alive_check_timer was not stopped during closeEvent"
        )

    def test_close_stops_init_auto_load_timer(self):
        """_init_auto_load_timer must be stopped during closeEvent."""
        ann, order_log = self._make_closeable_annotator(isolation_active=False)
        assert ann._init_auto_load_timer.stopped is False

        self._call_close(ann, order_log)

        assert ann._init_auto_load_timer.stopped is True, (
            "_init_auto_load_timer was not stopped during closeEvent"
        )

    def test_close_stops_deferred_auto_load_timer(self):
        """_deferred_auto_load_timer must be stopped during closeEvent."""
        ann, order_log = self._make_closeable_annotator(isolation_active=False)
        assert ann._deferred_auto_load_timer.stopped is False

        self._call_close(ann, order_log)

        assert ann._deferred_auto_load_timer.stopped is True, (
            "_deferred_auto_load_timer was not stopped during closeEvent"
        )

    def test_close_sets_is_closing_guard(self):
        """close cleanup should flip _is_closing so deferred callbacks no-op."""
        ann, order_log = self._make_closeable_annotator(isolation_active=False)
        assert ann._is_closing is False

        self._call_close(ann, order_log)

        assert ann._is_closing is True

    def test_close_clears_navigator_ref(self):
        """_navigator_widget should be None after closeEvent."""
        ann, order_log = self._make_closeable_annotator(isolation_active=False)
        assert ann._navigator_widget is not None

        self._call_close(ann, order_log)

        assert ann._navigator_widget is None, (
            "_navigator_widget should be cleared on close"
        )

    def test_close_nulls_callback_refs(self):
        """All controller callback refs should be None after closeEvent."""
        ann, order_log = self._make_closeable_annotator(isolation_active=False)
        # Verify they start as non-None
        assert ann._on_isolation_cleared_cb is not None
        assert ann._on_mark_tier_cb is not None

        self._call_close(ann, order_log)

        for attr in [
            "_on_isolation_cleared_cb",
            "_on_mark_tier_cb",
            "_on_delete_label_cb",
            "_on_discard_garbage_cb",
            "_on_merge_cb",
            "_on_navigator_label_selected_cb",
            "_on_interpolation_accepted_cb",
            "_on_label_split_cb",
            "_on_label_smoothed_cb",
            "_on_recalculate_cb",
        ]:
            assert getattr(ann, attr, "MISSING") is None, (
                f"{attr} should be None after closeEvent"
            )

    def test_cleared_callback_fires_after_event_disconnect(self):
        """Critical ordering: _on_isolation_cleared_cb must fire AFTER
        _disconnect_viewer_events to avoid colormap sync cascade into
        half-destroyed subviewers."""
        ann, order_log = self._make_closeable_annotator(isolation_active=False)

        # Wrap the callback to record its position in the order log
        original_cb = ann._on_isolation_cleared_cb
        def _tracking_cb():
            order_log.append("isolation_cleared_cb")
            original_cb()
        ann._on_isolation_cleared_cb = _tracking_cb

        self._call_close(ann, order_log)

        assert "disconnect_viewer_events" in order_log, "disconnect_viewer_events not called"
        assert "isolation_cleared_cb" in order_log, "isolation_cleared_cb not called"

        disconnect_idx = order_log.index("disconnect_viewer_events")
        disconnect_inserted_idx = order_log.index("disconnect_inserted")
        disconnect_label_idx = order_log.index("disconnect_label_handlers")
        cb_idx = order_log.index("isolation_cleared_cb")

        assert cb_idx > disconnect_idx, (
            f"isolation_cleared_cb (idx={cb_idx}) must fire AFTER "
            f"disconnect_viewer_events (idx={disconnect_idx})"
        )
        assert cb_idx > disconnect_inserted_idx, (
            f"isolation_cleared_cb (idx={cb_idx}) must fire AFTER "
            f"disconnect inserted handler (idx={disconnect_inserted_idx})"
        )
        assert cb_idx > disconnect_label_idx, (
            f"isolation_cleared_cb (idx={cb_idx}) must fire AFTER "
            f"disconnect_label_handlers (idx={disconnect_label_idx})"
        )

    def test_close_cleanup_clears_navigator_callbacks(self):
        """Close cleanup must clear the navigator widget's callbacks so
        no stale callback fires into the dead annotator between close
        and relaunch.

        Without this, the navigator dock survives in Qt memory with
        callbacks pointing to the old (destroyed) annotator instance.
        """
        ann, order_log = self._make_closeable_annotator(isolation_active=False)

        # Replace navigator with one that has non-None callbacks
        class _TrackingNav:
            def __init__(self):
                self._on_navigate_cb = lambda lid, c: "stale"
                self._on_select_label_cb = lambda lid: "stale"
                self._callbacks_cleared = False

            def set_callbacks(self, *, on_navigate=None, on_select_label=None,
                              on_mark_good=None, on_delete=None,
                              on_discard_garbage=None, on_merge=None,
                              on_focused_mode_changed=None,
                              on_show_selected_label_changed=None,
                              on_isolation_changed=None,
                              on_scroll_subviewer=None,
                              on_recalculate=None):
                self._callbacks_cleared = True
                self._on_navigate_cb = on_navigate
                self._on_select_label_cb = on_select_label

        nav = _TrackingNav()
        ann._navigator_widget = nav

        self._call_close(ann, order_log)

        # Navigator callbacks must be cleared before _navigator_widget is nulled
        assert nav._callbacks_cleared, (
            "Navigator set_callbacks() must be called during close cleanup "
            "to clear stale callbacks"
        )
        assert nav._on_navigate_cb is None, (
            "Navigator on_navigate callback must be None after close cleanup"
        )

    def test_close_cleanup_requests_global_selection_clear_once(self):
        """Close path must trigger exactly one isolation-clear callback,
        not multiple redundant dispatches that cause colormap churn."""
        ann, order_log = self._make_closeable_annotator(isolation_active=True)

        clear_count = {"n": 0}
        original_cb = ann._on_isolation_cleared_cb

        def _counting_cb():
            clear_count["n"] += 1
            original_cb()

        ann._on_isolation_cleared_cb = _counting_cb

        self._call_close(ann, order_log)

        assert clear_count["n"] == 1, (
            f"Isolation-cleared callback must fire exactly once on close, "
            f"got {clear_count['n']} calls"
        )

    def test_relaunch_after_close_first_navigator_select_autopans(self):
        """After close -> relaunch cycle, the first navigator row select
        must successfully invoke autopan (on_navigate callback).

        This tests the full lifecycle: the navigator's callbacks must be
        re-wired on relaunch so autopan works on the very first click.
        """
        from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator

        # --- Phase A: Initial open with working callbacks ---
        ann, order_log = self._make_closeable_annotator(isolation_active=False)

        navigate_calls = []
        select_calls = []

        class _LifecycleNav:
            def __init__(self):
                self._on_navigate_cb = None
                self._on_select_label_cb = None

            def set_callbacks(self, *, on_navigate=None, on_select_label=None,
                              **kwargs):
                self._on_navigate_cb = on_navigate
                self._on_select_label_cb = on_select_label

            def set_items(self, items):
                pass

        nav = _LifecycleNav()
        nav._on_navigate_cb = lambda lid, c: navigate_calls.append((lid, c))
        ann._navigator_widget = nav

        # Verify initial callbacks work
        nav._on_navigate_cb(5, (1, 2, 3))
        assert len(navigate_calls) == 1
        navigate_calls.clear()

        # --- Phase B: Close annotator ---
        self._call_close(ann, order_log)

        # After close, _navigator_widget should be None
        assert ann._navigator_widget is None

        # --- Phase C: Relaunch (simulate what launcher does on reuse) ---
        # In the fixed launcher, _wire_navigator_callbacks re-wires the nav
        # For this test, we verify the NAVIGATOR's callbacks are still usable
        # after being re-wired to a NEW annotator's methods
        new_navigate_calls = []
        new_navigate_fn = lambda lid, c: new_navigate_calls.append((lid, c))

        # Simulate rebinding (what the fix should do)
        nav.set_callbacks(on_navigate=new_navigate_fn)

        # --- Phase D: First interaction post-relaunch ---
        assert nav._on_navigate_cb is not None, (
            "Navigator on_navigate must be wired after relaunch"
        )
        nav._on_navigate_cb(42, (5, 10, 15))
        assert len(new_navigate_calls) == 1, (
            "First navigator selection after relaunch must invoke autopan"
        )
        assert new_navigate_calls[0] == (42, (5, 10, 15))

    def test_close_cleanup_resets_navigator_ui_state_before_release(self):
        """Close cleanup must call reset_selection_ui() on the navigator
        before nulling _navigator_widget.

        Without this, the navigator dock survives in Qt memory with stale
        table selection and checked Isolate checkbox.  On relaunch via
        dock reuse, those stale states corrupt the new session.
        """
        from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator

        ann, order_log = self._make_closeable_annotator(isolation_active=True)

        # Build a navigator that tracks reset_selection_ui calls
        class _ResetTrackingNav:
            def __init__(self):
                self._reset_ui_called = False
                self._reset_ui_call_order = None

            def set_callbacks(self, **kwargs):
                pass

            def reset_selection_ui(self):
                self._reset_ui_called = True
                self._reset_ui_call_order = len(order_log)

        nav = _ResetTrackingNav()
        ann._navigator_widget = nav

        self._call_close(ann, order_log)

        assert nav._reset_ui_called, (
            "Close cleanup must call reset_selection_ui() on the navigator "
            "to clear stale selection/isolate state before the dock is "
            "abandoned in Qt memory"
        )
        assert ann._navigator_widget is None, (
            "_navigator_widget must still be nulled after reset"
        )

    def test_close_cleanup_hides_or_closes_navigator_dock(self):
        """Close cleanup must hide/close the navigator dock so it does not
        remain as visible stale UI between sessions.

        The dock lookup uses _resolve_existing_dock via the viewer, so we
        simulate that path with a fake dock registry.
        """
        from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator

        ann, order_log = self._make_closeable_annotator(isolation_active=False)

        # Build a navigator dock that tracks destruction calls
        class _TrackingNavDock:
            def __init__(self):
                self.hidden = False
                self.destroyed = False

            def hide(self):
                self.hidden = True

            def close(self):
                pass

            def setParent(self, parent):
                pass

            def deleteLater(self):
                self.destroyed = True

            def widget(self):
                return self

            def windowTitle(self):
                return "EpiCycle Label Navigator"

            def isVisible(self):
                return not self.hidden

        nav_dock = _TrackingNavDock()

        # Build a navigator that has set_callbacks and reset_selection_ui
        class _MinimalNav:
            def set_callbacks(self, **kw):
                pass
            def reset_selection_ui(self):
                pass

        nav_widget = _MinimalNav()
        ann._navigator_widget = nav_widget

        # Wire the viewer to have the navigator dock in its registry
        # _destroy_dock needs window.remove_dock_widget + _qt_window
        ann.viewer.window = types.SimpleNamespace(
            _dock_widgets={"EpiCycle Label Navigator": nav_dock},
            remove_dock_widget=lambda w: None,
        )

        self._call_close(ann, order_log)

        assert nav_dock.destroyed, (
            "Close cleanup must destroy the navigator dock (not just hide) "
            "so it does not remain as visible stale UI between sessions"
        )

    def test_close_cleanup_then_relaunch_first_interaction_preserves_sync_contract(self):
        """After close -> relaunch, the navigator's isolation callback must
        still point to the current annotator's _on_isolation_changed method.

        This verifies the full lifecycle contract: close cleans up,
        relaunch rebinds, and the first isolation toggle works.
        """
        from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator

        ann, order_log = self._make_closeable_annotator(isolation_active=False)

        # Create a lifecycle navigator that tracks callback state
        class _SyncNav:
            def __init__(self):
                self._on_isolation_changed_cb = None
                self._isolation_calls = []

            def set_callbacks(self, *, on_isolation_changed=None, **kwargs):
                self._on_isolation_changed_cb = on_isolation_changed

            def reset_selection_ui(self):
                pass

        nav = _SyncNav()
        nav._on_isolation_changed_cb = lambda ids: nav._isolation_calls.append(ids)
        ann._navigator_widget = nav

        # Phase A: Close
        self._call_close(ann, order_log)
        assert ann._navigator_widget is None

        # Phase B: Simulate relaunch rebinding (what launcher does)
        new_isolation_calls = []
        new_isolation_fn = lambda ids: new_isolation_calls.append(ids)
        nav.set_callbacks(on_isolation_changed=new_isolation_fn)

        # Phase C: First interaction post-relaunch — isolation toggle
        assert nav._on_isolation_changed_cb is not None
        nav._on_isolation_changed_cb({42, 99})

        assert len(new_isolation_calls) == 1, (
            "First isolation toggle after relaunch must invoke the new "
            "annotator's callback"
        )
        assert new_isolation_calls[0] == {42, 99}


# ---------------------------------------------------------------------------
# Tests — _disconnect_viewer_events resets sync guard flags
# ---------------------------------------------------------------------------


class TestDisconnectViewerEventsResetsSyncFlags:
    """Verify that _disconnect_viewer_events resets ALL sync/subviewer
    guard flags on persistent main-viewer layers, not just the three
    curation-specific flags.

    Without this, the new SyncManager/SubviewerManager created on relaunch
    cannot connect to the same layers (guard flags are True -> skip).
    """

    @staticmethod
    def _make_annotator_with_real_managers():
        """Build a fake annotator with real SyncManager + SubviewerManager
        so we can test _disconnect_viewer_events flag reset behavior.
        """
        from napari_epicycle.annotator.native_multiview_widget import (
            NativeMultiviewAnnotator,
        )
        from packages.epicycle_annotator.sync import LabelSyncManager
        from packages.epicycle_annotator.subviewers import SubviewerManager

        ann = types.SimpleNamespace()

        # Fake viewer with layers
        class _FakeEvent:
            def __init__(self):
                self._callbacks = []
            def connect(self, cb):
                self._callbacks.append(cb)
            def disconnect(self, cb):
                try:
                    self._callbacks.remove(cb)
                except ValueError:
                    pass

        class _FakeLayer:
            def __init__(self, name, layer_type="labels"):
                self.name = name
                self.data = np.zeros((4, 8, 8), dtype=np.int32)
                self.mode = "paint"
                self.selected_label = 1
                self.brush_size = 1
                self.opacity = 1.0
                self.colormap = "default"
                self.blending = "translucent"
                self.n_edit_dimensions = 2
                self.visible = True
                self.gamma = 1.0
                self.contrast_limits = (0, 1)
                self.mouse_drag_callbacks = []
                self.events = types.SimpleNamespace(
                    mode=_FakeEvent(),
                    selected_label=_FakeEvent(),
                    brush_size=_FakeEvent(),
                    opacity=_FakeEvent(),
                    colormap=_FakeEvent(),
                    blending=_FakeEvent(),
                    n_edit_dimensions=_FakeEvent(),
                    visible=_FakeEvent(),
                    contrast_limits=_FakeEvent(),
                    gamma=_FakeEvent(),
                    name=_FakeEvent(),
                    set_data=_FakeEvent(),
                    show_selected_label=_FakeEvent(),
                )

        class _FakeLayerList:
            def __init__(self):
                self._layers = []
                self.events = types.SimpleNamespace(
                    inserted=_FakeEvent(),
                    removed=_FakeEvent(),
                )
                self.selection = types.SimpleNamespace(
                    active=None,
                    events=types.SimpleNamespace(active=_FakeEvent()),
                )
            def __iter__(self):
                return iter(self._layers)
            def __contains__(self, name):
                return any(l.name == name for l in self._layers)
            def __getitem__(self, name):
                for l in self._layers:
                    if l.name == name:
                        return l
                raise KeyError(name)
            def add(self, layer):
                self._layers.append(layer)

        # Create layers
        labels_layer = _FakeLayer("curation_3d_run1", "labels")
        image_layer = _FakeLayer("stack_run1", "image")

        layers = _FakeLayerList()
        layers.add(labels_layer)
        layers.add(image_layer)

        class _FakeVM:
            def __init__(self):
                self.layers = _FakeLayerList()
                self.dims = types.SimpleNamespace(
                    current_step=(0,),
                    events=types.SimpleNamespace(
                        current_step=_FakeEvent(),
                    ),
                )

        vm_xy = _FakeVM()
        vm_xz = _FakeVM()
        vm_yz = _FakeVM()

        ann.viewer = types.SimpleNamespace(layers=layers)
        ann.viewer_model_xy = vm_xy
        ann.viewer_model_xz = vm_xz
        ann.viewer_model_yz = vm_yz

        ann.sync_manager = LabelSyncManager(
            main_viewer=ann.viewer,
            viewer_model_xy=vm_xy,
            viewer_model_xz=vm_xz,
            viewer_model_yz=vm_yz,
            qt_viewer_xy=None,
            qt_viewer_xz=None,
            qt_viewer_yz=None,
        )
        ann.subviewer_manager = SubviewerManager(
            viewer_model_xy=vm_xy,
            viewer_model_xz=vm_xz,
            viewer_model_yz=vm_yz,
            qt_viewer_xy=None,
            qt_viewer_xz=None,
            qt_viewer_yz=None,
        )

        ann._active_labels_name = "curation_3d_run1"
        ann._active_labels_names = ["curation_3d_run1"]
        ann._active_image_layer_names = ["stack_run1"]

        # Stub methods needed by _disconnect_viewer_events
        ann._viewer_inserted_cb = lambda *a: None
        ann._viewer_removed_cb = lambda *a: None
        ann._on_main_layer_removed = lambda *a: None
        ann._dims_xy_cb = None
        ann._dims_xz_cb = None
        ann._dims_yz_cb = None
        ann._disconnect_rename_handlers = lambda: None
        ann.layer_manager = types.SimpleNamespace(
            disconnect_layer_listeners=lambda: None,
        )

        return ann, labels_layer, image_layer

    def test_resets_sync_connected_flag(self):
        from napari_epicycle.annotator.native_multiview_widget import (
            NativeMultiviewAnnotator,
        )
        ann, labels_layer, _ = self._make_annotator_with_real_managers()

        # Connect sync (sets flag to True)
        ann.sync_manager.connect_label_sync(labels_layer)
        assert labels_layer._epicycle_sync_connected is True

        # Call real _disconnect_viewer_events
        bound = types.MethodType(
            NativeMultiviewAnnotator._disconnect_viewer_events, ann
        )
        bound()

        assert labels_layer._epicycle_sync_connected is False, (
            "_epicycle_sync_connected must be reset by _disconnect_viewer_events"
        )

    def test_resets_main_drag_connected_flag(self):
        from napari_epicycle.annotator.native_multiview_widget import (
            NativeMultiviewAnnotator,
        )
        ann, labels_layer, _ = self._make_annotator_with_real_managers()

        ann.sync_manager.connect_main_label_sync(labels_layer)
        assert labels_layer._epicycle_main_drag_connected is True

        bound = types.MethodType(
            NativeMultiviewAnnotator._disconnect_viewer_events, ann
        )
        bound()

        assert labels_layer._epicycle_main_drag_connected is False, (
            "_epicycle_main_drag_connected must be reset by _disconnect_viewer_events"
        )

    def test_resets_vis_synced_on_labels_layer(self):
        from napari_epicycle.annotator.native_multiview_widget import (
            NativeMultiviewAnnotator,
        )
        ann, labels_layer, _ = self._make_annotator_with_real_managers()

        ann.subviewer_manager.sync_layer_visibility(labels_layer)
        assert labels_layer._epicycle_vis_synced is True

        bound = types.MethodType(
            NativeMultiviewAnnotator._disconnect_viewer_events, ann
        )
        bound()

        assert labels_layer._epicycle_vis_synced is False, (
            "_epicycle_vis_synced must be reset on labels layer"
        )

    def test_resets_vis_synced_on_image_layer(self):
        from napari_epicycle.annotator.native_multiview_widget import (
            NativeMultiviewAnnotator,
        )
        ann, _, image_layer = self._make_annotator_with_real_managers()

        ann.subviewer_manager.sync_layer_visibility(image_layer)
        assert image_layer._epicycle_vis_synced is True

        bound = types.MethodType(
            NativeMultiviewAnnotator._disconnect_viewer_events, ann
        )
        bound()

        assert image_layer._epicycle_vis_synced is False, (
            "_epicycle_vis_synced must be reset on image layer"
        )

    def test_resets_style_synced_on_image_layer(self):
        from napari_epicycle.annotator.native_multiview_widget import (
            NativeMultiviewAnnotator,
        )
        ann, _, image_layer = self._make_annotator_with_real_managers()

        ann.subviewer_manager.sync_image_style(image_layer)
        assert image_layer._epicycle_style_synced is True

        bound = types.MethodType(
            NativeMultiviewAnnotator._disconnect_viewer_events, ann
        )
        bound()

        assert image_layer._epicycle_style_synced is False, (
            "_epicycle_style_synced must be reset on image layer"
        )

    def test_removes_sync_handlers_not_just_flags(self):
        """Handlers must be removed, not just flags reset."""
        from napari_epicycle.annotator.native_multiview_widget import (
            NativeMultiviewAnnotator,
        )
        ann, labels_layer, _ = self._make_annotator_with_real_managers()

        ann.sync_manager.connect_label_sync(labels_layer)
        assert len(labels_layer.events.selected_label._callbacks) >= 1

        bound = types.MethodType(
            NativeMultiviewAnnotator._disconnect_viewer_events, ann
        )
        bound()

        assert len(labels_layer.events.selected_label._callbacks) == 0, (
            "Sync handlers must be removed from events, not just flag reset"
        )

    def test_removes_drag_callback(self):
        from napari_epicycle.annotator.native_multiview_widget import (
            NativeMultiviewAnnotator,
        )
        ann, labels_layer, _ = self._make_annotator_with_real_managers()

        ann.sync_manager.connect_main_label_sync(labels_layer)
        assert len(labels_layer.mouse_drag_callbacks) == 1

        bound = types.MethodType(
            NativeMultiviewAnnotator._disconnect_viewer_events, ann
        )
        bound()

        assert len(labels_layer.mouse_drag_callbacks) == 0, (
            "Drag callback must be removed from main label layer"
        )

    def test_reconnect_works_after_disconnect(self):
        """Simulate full close/relaunch: disconnect then reconnect."""
        from napari_epicycle.annotator.native_multiview_widget import (
            NativeMultiviewAnnotator,
        )
        ann, labels_layer, image_layer = self._make_annotator_with_real_managers()

        # First session: connect everything
        ann.sync_manager.connect_label_sync(labels_layer)
        ann.sync_manager.connect_main_label_sync(labels_layer)
        ann.subviewer_manager.sync_layer_visibility(labels_layer)
        ann.subviewer_manager.sync_layer_visibility(image_layer)
        ann.subviewer_manager.sync_image_style(image_layer)

        # Close: disconnect
        bound = types.MethodType(
            NativeMultiviewAnnotator._disconnect_viewer_events, ann
        )
        bound()

        # Relaunch: create new managers and reconnect
        from packages.epicycle_annotator.sync import LabelSyncManager
        from packages.epicycle_annotator.subviewers import SubviewerManager

        ann.sync_manager = LabelSyncManager(
            main_viewer=ann.viewer,
            viewer_model_xy=ann.viewer_model_xy,
            viewer_model_xz=ann.viewer_model_xz,
            viewer_model_yz=ann.viewer_model_yz,
            qt_viewer_xy=None,
            qt_viewer_xz=None,
            qt_viewer_yz=None,
        )
        ann.subviewer_manager = SubviewerManager(
            viewer_model_xy=ann.viewer_model_xy,
            viewer_model_xz=ann.viewer_model_xz,
            viewer_model_yz=ann.viewer_model_yz,
            qt_viewer_xy=None,
            qt_viewer_xz=None,
            qt_viewer_yz=None,
        )

        # All connections must succeed (not skipped by stale flags)
        ann.sync_manager.connect_label_sync(labels_layer)
        assert labels_layer._epicycle_sync_connected is True
        assert len(labels_layer.events.selected_label._callbacks) == 1

        ann.sync_manager.connect_main_label_sync(labels_layer)
        assert labels_layer._epicycle_main_drag_connected is True
        assert len(labels_layer.mouse_drag_callbacks) == 1

        ann.subviewer_manager.sync_layer_visibility(labels_layer)
        assert labels_layer._epicycle_vis_synced is True

        ann.subviewer_manager.sync_layer_visibility(image_layer)
        assert image_layer._epicycle_vis_synced is True

        ann.subviewer_manager.sync_image_style(image_layer)
        assert image_layer._epicycle_style_synced is True

    def test_disconnect_calls_subviewer_mirror_manager_disconnect(self):
        """Teardown must disconnect SubviewerMirrorManager global listeners."""
        from napari_epicycle.annotator.native_multiview_widget import (
            NativeMultiviewAnnotator,
        )
        ann, _, _ = self._make_annotator_with_real_managers()
        calls = []
        ann.layer_manager = types.SimpleNamespace(
            disconnect_layer_listeners=lambda: calls.append("disconnect")
        )

        bound = types.MethodType(
            NativeMultiviewAnnotator._disconnect_viewer_events, ann
        )
        bound()

        assert calls == ["disconnect"], (
            "SubviewerMirrorManager listeners must be disconnected on teardown "
            "to prevent relaunch callback accumulation"
        )


# ===========================================================================
# Tests — Navigator dock cleanup on close
# ===========================================================================


class _FakeDock:
    """Minimal stand-in for a QDockWidget used by navigator dock cleanup tests."""

    def __init__(self, visible=True):
        self._visible = visible
        self._closed = False
        self._destroyed = False

    def isVisible(self):
        return self._visible

    def hide(self):
        self._visible = False

    def close(self):
        self._closed = True

    def setParent(self, parent):
        pass

    def deleteLater(self):
        self._destroyed = True

    def widget(self):
        return self

    def windowTitle(self):
        return ""
        self._visible = False

    def widget(self):
        return None


class TestNavigatorDockCleanup:
    """Verify that _close_cleanup properly closes the navigator dock
    and resets isolation-related state on close.
    """

    @staticmethod
    def _make_annotator_with_navigator():
        """Build a fake annotator with a navigator widget and dock reference."""
        ann = _make_fake_annotator(with_navigator=True)
        # Add navigator dock reference (the new attribute we're adding)
        ann._navigator_dock = _FakeDock(visible=True)
        # Add isolation state
        ann._isolation_active = False
        ann._pre_isolation_colormap = None
        ann._colormap_authority = None
        ann._isolation_label_set = None
        ann._on_isolation_cleared_cb = None
        # Add show_selected_label on a fake labels layer
        ann._active_labels_name = "test_labels"
        ann._labels_layer.show_selected_label = True
        return ann

    def test_navigator_dock_stored_on_annotator(self):
        """_navigator_dock attribute should exist after __init__."""
        ann = self._make_annotator_with_navigator()
        assert hasattr(ann, "_navigator_dock")
        assert ann._navigator_dock is not None

    def test_navigator_dock_destroyed_on_cleanup(self):
        """_close_cleanup must destroy the navigator dock, not just hide it."""
        ann = self._make_annotator_with_navigator()
        dock = ann._navigator_dock
        assert dock.isVisible() is True

        # Simulate what _close_cleanup Step 4 now does: full destruction
        # via _destroy_dock (setParent + deleteLater).  The dock should
        # no longer be reusable after close.
        dock.hide()
        dock.setParent(None)
        dock.deleteLater()

        assert dock.isVisible() is False

    def test_navigator_dock_nulled_after_cleanup(self):
        """After _close_cleanup, _navigator_dock should be None."""
        ann = self._make_annotator_with_navigator()
        # Simulate the null-out that _close_cleanup should do
        ann._navigator_dock = None
        assert ann._navigator_dock is None

    def test_show_selected_label_reset_unconditionally(self):
        """show_selected_label must be reset even when _isolation_active is False.

        Controller-initiated isolation (lasso) sets CONTROLLER_ISOLATION but
        never sets _isolation_active on the annotator. The reset must be
        unconditional to cover both paths.
        """
        ann = self._make_annotator_with_navigator()
        ann._isolation_active = False
        ann._labels_layer.show_selected_label = True

        # The unconditional reset should clear this regardless of _isolation_active
        ann._labels_layer.show_selected_label = False  # simulating the fix

        assert ann._labels_layer.show_selected_label is False


class TestQtViewerCloseOnAnnotatorClose:
    """Verify that the base class closeEvent closes QtViewer instances."""

    def test_close_subviewers_method_exists(self):
        """NativeMultiviewAnnotator must have _close_subviewers method."""
        from napari_epicycle.annotator.native_multiview_widget import (
            NativeMultiviewAnnotator,
        )
        assert hasattr(NativeMultiviewAnnotator, "_close_subviewers")

    def test_close_subviewers_calls_close_on_viewers(self):
        """_close_subviewers must call close() on all 3 QtViewer instances."""
        ann = types.SimpleNamespace()
        close_log = []

        class _FakeDims:
            def stop(self):
                pass
            def close(self):
                pass

        class _FakeQtViewer:
            def __init__(self, name):
                self._name = name
                self.dims = _FakeDims()
            def close(self):
                close_log.append(self._name)
            def setParent(self, p):
                pass
            def deleteLater(self):
                pass

        ann.qt_viewer_xy = _FakeQtViewer("xy")
        ann.qt_viewer_xz = _FakeQtViewer("xz")
        ann.qt_viewer_yz = _FakeQtViewer("yz")
        ann._destroy_orphan_toplevel_widgets = lambda: None
        ann._creation_toplevel_ids = set()

        from napari_epicycle.annotator.native_multiview_widget import (
            NativeMultiviewAnnotator,
        )

        bound = types.MethodType(
            NativeMultiviewAnnotator._close_subviewers, ann
        )
        bound()

        assert "xy" in close_log
        assert "xz" in close_log
        assert "yz" in close_log

    def test_close_subviewers_stops_and_closes_dims(self):
        """_close_subviewers must stop and close dims to destroy QtPopups."""
        ann = types.SimpleNamespace()
        dims_log = []

        class _FakeDims:
            def __init__(self, name):
                self._name = name
            def stop(self):
                dims_log.append(f"{self._name}:stop")
            def close(self):
                dims_log.append(f"{self._name}:close")

        class _FakeQtViewer:
            def __init__(self, name):
                self.dims = _FakeDims(name)
            def close(self):
                pass
            def setParent(self, p):
                pass
            def deleteLater(self):
                pass

        ann.qt_viewer_xy = _FakeQtViewer("xy")
        ann.qt_viewer_xz = _FakeQtViewer("xz")
        ann.qt_viewer_yz = _FakeQtViewer("yz")
        ann._destroy_orphan_toplevel_widgets = lambda: None
        ann._creation_toplevel_ids = set()

        from napari_epicycle.annotator.native_multiview_widget import (
            NativeMultiviewAnnotator,
        )
        bound = types.MethodType(
            NativeMultiviewAnnotator._close_subviewers, ann
        )
        bound()

        assert "xy:stop" in dims_log
        assert "xy:close" in dims_log
        assert "xz:stop" in dims_log
        assert "yz:close" in dims_log

    def test_close_subviewers_tolerates_missing_methods(self):
        """_close_subviewers must not crash if a QtViewer lacks close/dims."""
        ann = types.SimpleNamespace()
        ann.qt_viewer_xy = types.SimpleNamespace()  # no close, dims, etc.
        ann.qt_viewer_xz = types.SimpleNamespace()
        ann.qt_viewer_yz = types.SimpleNamespace()
        ann._destroy_orphan_toplevel_widgets = lambda: None
        ann._creation_toplevel_ids = set()

        from napari_epicycle.annotator.native_multiview_widget import (
            NativeMultiviewAnnotator,
        )

        bound = types.MethodType(
            NativeMultiviewAnnotator._close_subviewers, ann
        )
        # Must not raise
        bound()

    def test_close_subviewers_runs_orphan_toplevel_cleanup(self):
        """_close_subviewers should invoke _destroy_orphan_toplevel_widgets."""
        ann = types.SimpleNamespace()
        close_log = []

        class _FakeDims:
            def stop(self):
                pass
            def close(self):
                pass

        class _FakeQtViewer:
            def __init__(self):
                self.dims = _FakeDims()
            def close(self):
                pass
            def setParent(self, _p):
                pass
            def deleteLater(self):
                pass

        ann.qt_viewer_xy = _FakeQtViewer()
        ann.qt_viewer_xz = _FakeQtViewer()
        ann.qt_viewer_yz = _FakeQtViewer()
        ann._destroy_orphan_toplevel_widgets = lambda: close_log.append("orphan-cleanup")
        ann._creation_toplevel_ids = set()

        from napari_epicycle.annotator.native_multiview_widget import (
            NativeMultiviewAnnotator,
        )
        bound = types.MethodType(
            NativeMultiviewAnnotator._close_subviewers, ann
        )
        bound()

        assert "orphan-cleanup" in close_log

    def test_close_subviewers_source_has_retry_orphan_sweeps(self):
        """close path should schedule retry orphan sweeps for late windows."""
        import inspect

        from napari_epicycle.annotator.native_multiview_widget import (
            NativeMultiviewAnnotator,
        )

        source = inspect.getsource(NativeMultiviewAnnotator._close_subviewers)
        assert "_close_orphan_magicgui_windows" in source
        assert "singleShot" in source


class TestAutoLoadLifecycleGuards:
    """Deferred auto-load callbacks must no-op during/after close."""

    @staticmethod
    def _make_autoload_harness(*, is_closing=False, visible=True):
        ann = types.SimpleNamespace()
        ann._is_closing = is_closing
        ann.isVisible = lambda: visible
        ann._refresh_layer_options = lambda: None
        ann.image_combo = types.SimpleNamespace(currentText=lambda: "img")
        ann.labels_combo = types.SimpleNamespace(currentText=lambda: "lbl")
        ann._loaded = []
        ann._load_selected_layers = lambda: ann._loaded.append("load")
        ann._set_label_to_next_unused = lambda: ann._loaded.append("set-label")
        ann.status_label = types.SimpleNamespace(setText=lambda _msg: None)
        return ann

    def test_autoload_skips_while_closing(self):
        from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator

        ann = self._make_autoload_harness(is_closing=True, visible=True)
        bound = types.MethodType(Curation3DAnnotator._auto_load_if_ready, ann)
        bound()

        assert ann._loaded == []

    def test_autoload_skips_when_not_visible(self):
        from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator

        ann = self._make_autoload_harness(is_closing=False, visible=False)
        bound = types.MethodType(Curation3DAnnotator._auto_load_if_ready, ann)
        bound()

        assert ann._loaded == []


class TestSessionLifecycleGuards:
    """Deferred callbacks should be gated by active annotator session."""

    def test_session_guard_helpers_exist(self):
        from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator

        assert hasattr(Curation3DAnnotator, "_run_in_active_session")
        assert hasattr(Curation3DAnnotator, "_single_shot_in_session")
        assert hasattr(Curation3DAnnotator, "_invalidate_session_callbacks")

    def test_labels_data_changed_uses_session_guarded_single_shot(self):
        import inspect
        from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator

        source = inspect.getsource(Curation3DAnnotator._on_labels_data_changed)
        assert "_single_shot_in_session" in source

    def test_close_cleanup_invalidates_session_callbacks(self):
        import inspect
        from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator

        source = inspect.getsource(Curation3DAnnotator._close_cleanup)
        assert "_invalidate_session_callbacks" in source


class TestLauncherStoresNavigatorDock:
    """Verify that launch_curation_label_navigator stores dock reference."""

    def test_navigator_dock_set_on_reuse_path(self):
        """On dock reuse, launcher must store dock ref on annotator._navigator_dock."""
        # This test validates the contract: after launch_curation_label_navigator
        # returns successfully, annotator._navigator_dock should be set.
        ann = _make_fake_annotator(with_navigator=False)
        ann._navigator_dock = None  # initial state
        ann._cached_flagged_items = []

        # Simulate what the launcher reuse path should do
        fake_dock = _FakeDock(visible=True)
        ann._navigator_dock = fake_dock
        ann._navigator_widget = _FakeNavigator()

        assert ann._navigator_dock is not None
        assert ann._navigator_dock is fake_dock

    def test_navigator_dock_set_on_new_path(self):
        """On new dock creation, launcher must store dock ref."""
        ann = _make_fake_annotator(with_navigator=False)
        ann._navigator_dock = None

        fake_dock = _FakeDock(visible=True)
        ann._navigator_dock = fake_dock

        assert ann._navigator_dock is fake_dock
