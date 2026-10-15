"""Tests for the navigator-side isolation guard (FM3 + FM7 fix).

Background
----------
The 2026-05-16 visibility/isolation trace identified seven failure modes
in the hide/unhide system.  FM3 and FM7 share a root cause: when a
refinement op is staged (interp / split / smooth preview), toggling
the Isolate checkbox or firing isolation from a row click clobbers the
op's isolation state (proxy / colormap) and leaves the user with a
blank subviewer or hidden Refine layer.

The fix is a navigator-side gate.  ``CurationLabelNavigator`` gets a
new ``_is_op_active_cb`` predicate slot, wired by
``Curation3DAnnotator._wire_navigator_callbacks`` to
``_is_op_active_for_navigator`` which returns ``_refinement_op_active()``.
The three user-initiated isolation entry points
(``_on_isolate_toggled``, ``_fire_isolation_if_active``,
``_on_render_selected_clicked``) consult the predicate and bail out
silently with a transient status message when True.

These tests are Qt-free.  They use bare-instance pattern (no
``__init__`` call) + lightweight fake checkbox + monkey-patched
``_show_op_active_status``.  Source-level tests confirm the gate is
present at each entry point.

Why not test against full QWidget?  The Qt instantiation pulls in the
whole napari/qtpy stack and triggers process-exit crashes on Windows.
Bare-instance is the project-standard pattern for annotator unit tests.
"""
from __future__ import annotations

import inspect


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeCheckBox:
    """Minimal QCheckBox stand-in with ``blockSignals`` semantics."""

    def __init__(self, checked=False):
        self._checked = checked
        self._signals_blocked = False

    def isChecked(self):
        return self._checked

    def setChecked(self, v):
        self._checked = bool(v)

    def blockSignals(self, v):
        self._signals_blocked = bool(v)
        return not self._signals_blocked


class _FakeLabel:
    """Minimal QLabel stand-in."""

    def __init__(self):
        self._text = ""

    def setText(self, v):
        self._text = str(v)

    def text(self):
        return self._text


def _make_bare_navigator(*, op_active_cb=None, isolate_checked=False):
    """Build a bare-instance CurationLabelNavigator with the minimum
    state needed for the three gate methods to run end-to-end without
    Qt.

    Patches ``_show_op_active_status`` to a tracker so tests can
    assert it was called.
    """
    from napari_epicycle.annotator.curation_label_navigator import (
        CurationLabelNavigator,
    )
    nav = CurationLabelNavigator.__new__(CurationLabelNavigator)
    nav._is_op_active_cb = op_active_cb
    nav._op_active_status_timer = None
    nav._editing_header = _FakeLabel()
    nav._isolate_cb = _FakeCheckBox(checked=isolate_checked)
    nav._on_isolation_changed_cb = None
    nav._on_show_selected_label_changed_cb = None
    nav._suppress_isolation_dispatch = False
    nav._last_rendered_label_ids = set()
    # Tracker for status calls.
    nav._status_calls = []
    nav._show_op_active_status = lambda msec=2000: nav._status_calls.append(msec)
    # Tracker for _update_render_button_state (called by some gate
    # paths to refresh button enabled state).
    nav._update_render_button_state = lambda: None
    # Curation debug is a no-op in tests.
    nav._curation_debug = lambda *a, **kw: None
    # Selected label IDs — empty by default.
    nav.selected_label_ids = lambda: set()
    return nav


# ===========================================================================
# _is_op_active() helper
# ===========================================================================


class TestIsOpActiveHelper:
    """Predicate returns the callback's result, defaulting to False
    when the callback is unset or raises."""

    def test_returns_false_when_no_callback(self):
        nav = _make_bare_navigator(op_active_cb=None)
        assert nav._is_op_active() is False

    def test_returns_true_when_callback_returns_true(self):
        nav = _make_bare_navigator(op_active_cb=lambda: True)
        assert nav._is_op_active() is True

    def test_returns_false_when_callback_returns_false(self):
        nav = _make_bare_navigator(op_active_cb=lambda: False)
        assert nav._is_op_active() is False

    def test_returns_false_when_callback_raises(self):
        def _raises():
            raise RuntimeError("simulated annotator teardown")
        nav = _make_bare_navigator(op_active_cb=_raises)
        # Predicate must be defensive — the navigator may outlive a
        # crashing annotator briefly during close cleanup.
        assert nav._is_op_active() is False

    def test_returns_truthy_value_coerced_to_bool(self):
        nav = _make_bare_navigator(op_active_cb=lambda: "yes")
        assert nav._is_op_active() is True


# ===========================================================================
# _on_isolate_toggled gate (FM3 / FM7 — the Isolate checkbox)
# ===========================================================================


class TestIsolateToggledGate:

    def test_no_op_active_proceeds_normally(self):
        """When no op is staged, toggling fires the isolation callback
        and the checkbox stays at the toggled value."""
        from napari_epicycle.annotator.curation_label_navigator import (
            CurationLabelNavigator,
        )
        fired = []
        nav = _make_bare_navigator(op_active_cb=lambda: False)
        nav._on_isolation_changed_cb = lambda ids: fired.append(set(ids))
        nav._isolate_cb._checked = True  # user just checked it
        nav.selected_label_ids = lambda: {7}
        CurationLabelNavigator._on_isolate_toggled(nav, True)
        assert fired == [{7}]
        assert nav._isolate_cb.isChecked() is True
        assert nav._status_calls == []  # no status message

    def test_op_active_reverts_toggle_and_blocks_fire(self):
        """When an op is staged, the gate reverts the checkbox to its
        pre-click state, fires the status message, and does NOT call
        the isolation callback."""
        from napari_epicycle.annotator.curation_label_navigator import (
            CurationLabelNavigator,
        )
        fired = []
        nav = _make_bare_navigator(op_active_cb=lambda: True)
        nav._on_isolation_changed_cb = lambda ids: fired.append(set(ids))
        nav._isolate_cb._checked = True  # user just checked it
        CurationLabelNavigator._on_isolate_toggled(nav, True)
        # Reverted to pre-click state (False, since user just turned it
        # on — the revert flips it back to off).
        assert nav._isolate_cb.isChecked() is False
        assert fired == []
        assert nav._status_calls == [2000]

    def test_op_active_uncheck_reverts_to_checked(self):
        """Symmetric case: user unchecks isolate while op is staged —
        gate reverts to checked."""
        from napari_epicycle.annotator.curation_label_navigator import (
            CurationLabelNavigator,
        )
        nav = _make_bare_navigator(op_active_cb=lambda: True)
        nav._on_isolation_changed_cb = lambda ids: None
        nav._isolate_cb._checked = False  # user just unchecked it
        CurationLabelNavigator._on_isolate_toggled(nav, False)
        # Reverted to pre-click state (True).
        assert nav._isolate_cb.isChecked() is True
        assert nav._status_calls == [2000]


# ===========================================================================
# _fire_isolation_if_active gate (FM3 / FM7 — row click path)
# ===========================================================================


class TestFireIsolationIfActiveGate:

    def test_no_op_active_fires_normally(self):
        from napari_epicycle.annotator.curation_label_navigator import (
            CurationLabelNavigator,
        )
        fired = []
        nav = _make_bare_navigator(
            op_active_cb=lambda: False, isolate_checked=True
        )
        nav._on_isolation_changed_cb = lambda ids: fired.append(set(ids))
        nav.selected_label_ids = lambda: {12}
        CurationLabelNavigator._fire_isolation_if_active(nav)
        assert fired == [{12}]
        assert nav._status_calls == []

    def test_op_active_blocks_fire_and_shows_status(self):
        from napari_epicycle.annotator.curation_label_navigator import (
            CurationLabelNavigator,
        )
        fired = []
        nav = _make_bare_navigator(
            op_active_cb=lambda: True, isolate_checked=True
        )
        nav._on_isolation_changed_cb = lambda ids: fired.append(set(ids))
        nav.selected_label_ids = lambda: {12}
        CurationLabelNavigator._fire_isolation_if_active(nav)
        assert fired == []
        assert nav._status_calls == [2000]

    def test_isolate_off_short_circuits_before_gate(self):
        """If Isolate checkbox is OFF, the method returns before
        consulting the op-active gate.  No status is shown because
        the gate path isn't reached.
        """
        from napari_epicycle.annotator.curation_label_navigator import (
            CurationLabelNavigator,
        )
        fired = []
        nav = _make_bare_navigator(
            op_active_cb=lambda: True, isolate_checked=False
        )
        nav._on_isolation_changed_cb = lambda ids: fired.append(set(ids))
        CurationLabelNavigator._fire_isolation_if_active(nav)
        assert fired == []
        assert nav._status_calls == []  # gate not reached


# ===========================================================================
# _on_render_selected_clicked gate (FM3 / FM7 — Render Selected button)
# ===========================================================================


class TestRenderSelectedGate:

    def test_no_op_active_fires_normally(self):
        from napari_epicycle.annotator.curation_label_navigator import (
            CurationLabelNavigator,
        )
        fired = []
        nav = _make_bare_navigator(
            op_active_cb=lambda: False, isolate_checked=True
        )
        nav._on_isolation_changed_cb = lambda ids: fired.append(set(ids))
        nav.selected_label_ids = lambda: {3, 5, 9}
        CurationLabelNavigator._on_render_selected_clicked(nav)
        assert fired == [{3, 5, 9}]
        assert nav._status_calls == []

    def test_op_active_blocks_render(self):
        from napari_epicycle.annotator.curation_label_navigator import (
            CurationLabelNavigator,
        )
        fired = []
        nav = _make_bare_navigator(
            op_active_cb=lambda: True, isolate_checked=True
        )
        nav._on_isolation_changed_cb = lambda ids: fired.append(set(ids))
        nav.selected_label_ids = lambda: {3, 5, 9}
        CurationLabelNavigator._on_render_selected_clicked(nav)
        assert fired == []
        assert nav._status_calls == [2000]


# ===========================================================================
# Source-level: wiring + gate presence
# ===========================================================================


class TestGateWiring:
    """Source-level smoke checks — confirm the wiring stays in place
    across refactors."""

    def test_navigator_init_creates_predicate_slot(self):
        from napari_epicycle.annotator.curation_label_navigator import (
            CurationLabelNavigator,
        )
        src = inspect.getsource(CurationLabelNavigator.__init__)
        assert "self._is_op_active_cb = None" in src

    def test_set_callbacks_accepts_is_op_active(self):
        from napari_epicycle.annotator.curation_label_navigator import (
            CurationLabelNavigator,
        )
        src = inspect.getsource(CurationLabelNavigator.set_callbacks)
        assert "is_op_active" in src
        assert "self._is_op_active_cb = is_op_active" in src

    def test_widget_wires_predicate(self):
        """`Curation3DAnnotator._wire_navigator_callbacks` must pass
        `is_op_active=self._is_op_active_for_navigator` to
        `nav.set_callbacks`."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        src = inspect.getsource(Curation3DAnnotator._wire_navigator_callbacks)
        assert "is_op_active=self._is_op_active_for_navigator" in src

    def test_widget_predicate_method_exists(self):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        assert hasattr(Curation3DAnnotator, "_is_op_active_for_navigator")
        src = inspect.getsource(Curation3DAnnotator._is_op_active_for_navigator)
        # Must read through _refinement_op_active (the most conservative
        # predicate — covers interp_snapshot held + refine_mode).
        assert "_refinement_op_active" in src

    def test_isolate_toggled_has_gate(self):
        from napari_epicycle.annotator.curation_label_navigator import (
            CurationLabelNavigator,
        )
        src = inspect.getsource(CurationLabelNavigator._on_isolate_toggled)
        assert "self._is_op_active()" in src
        assert "_show_op_active_status" in src
        # The gate must revert the checkbox state silently — verify the
        # blockSignals + setChecked pattern is present.
        assert "blockSignals(True)" in src
        assert "setChecked(not checked)" in src

    def test_fire_isolation_has_gate(self):
        from napari_epicycle.annotator.curation_label_navigator import (
            CurationLabelNavigator,
        )
        src = inspect.getsource(
            CurationLabelNavigator._fire_isolation_if_active
        )
        assert "self._is_op_active()" in src

    def test_render_selected_has_gate(self):
        from napari_epicycle.annotator.curation_label_navigator import (
            CurationLabelNavigator,
        )
        src = inspect.getsource(
            CurationLabelNavigator._on_render_selected_clicked
        )
        assert "self._is_op_active()" in src


# ===========================================================================
# Widget predicate: _is_op_active_for_navigator
# ===========================================================================


class TestWidgetPredicate:
    """The widget-side predicate that the navigator's callback returns."""

    def _make_bare_widget(self, **flags):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        ann = Curation3DAnnotator.__new__(Curation3DAnnotator)
        ann._split_running = flags.get("split", False)
        ann._smooth_running = flags.get("smooth", False)
        ann._interp_snapshot = flags.get("interp_snapshot", None)
        ann._interp_refine_mode = flags.get("interp_refine_mode", False)
        return ann

    def test_returns_false_when_idle(self):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        ann = self._make_bare_widget()
        assert Curation3DAnnotator._is_op_active_for_navigator(ann) is False

    def test_returns_true_when_split_running(self):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        ann = self._make_bare_widget(split=True)
        assert Curation3DAnnotator._is_op_active_for_navigator(ann) is True

    def test_returns_true_when_smooth_running(self):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        ann = self._make_bare_widget(smooth=True)
        assert Curation3DAnnotator._is_op_active_for_navigator(ann) is True

    def test_returns_true_when_interp_snapshot_held(self):
        """Interp held a snapshot — preview is staged.  Critical: this
        catches the case where `_interp_running` was cleared by a
        partial cleanup but the snapshot is still live (which would
        bypass the dataclass-level `interp_active` flag).
        """
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        ann = self._make_bare_widget(interp_snapshot=object())
        assert Curation3DAnnotator._is_op_active_for_navigator(ann) is True

    def test_returns_true_when_interp_refine_mode(self):
        """FM6-class protection: refine-further mode counts as
        op-active even if interp_running was cleared.
        """
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        ann = self._make_bare_widget(interp_refine_mode=True)
        assert Curation3DAnnotator._is_op_active_for_navigator(ann) is True
