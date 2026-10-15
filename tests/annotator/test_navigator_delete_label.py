"""Tests for the navigator Delete Label button + canonical curation-undo
integration (Task A, 2026-06-01,
feature/delete-label-via-curation-undo).

Verifies:
- ``CurationLabelNavigator`` constructs ``_delete_label_btn`` with
  ``DESTRUCTIVE_BTN_STYLE`` and adds it to the sidebar layout directly
  under ``+ New Label``.
- ``set_callbacks(on_delete_label=...)`` accepts the new kwarg and
  stores it as ``self._on_delete_label_cb``.
- ``_on_delete_label_clicked`` fires the callback with
  ``current_label_id()`` and respects ``_auto_advance_cb``.
- ``_wire_signals`` wires ``_delete_label_btn.clicked`` to the handler.
- ``Curation3DAnnotator._on_delete_label_from_navigator`` follows the
  split-accept canonical pattern (snapshot ROI -> mutate -> sync ->
  commit) and gates against in-flight ops + paint snapshot + merge
  dialog.
- ``nav.set_callbacks(...)`` in ``_wire_navigator_callbacks`` passes
  ``on_delete_label=self._on_delete_label_from_navigator``.
- The navigator has a ``set_label_action_buttons_enabled`` method that
  toggles BOTH ``_delete_label_btn`` AND ``_new_label_btn`` together.
- Negative assertions: the new code does NOT push to
  ``_tier_undo_stack`` (that was the abandoned 0ae3039 mistake) and
  does NOT use any 4-tuple format.

All tests are Qt-free — source-inspect + filesystem checks.
"""
from __future__ import annotations

import inspect

from napari_epicycle.annotator.curation_label_navigator import (
    CurationLabelNavigator,
)
from napari_epicycle.annotator.curation_3d_widget import (
    Curation3DAnnotator,
)
from napari_epicycle.annotator.refinement_state import RefinementOpState


# ---------------------------------------------------------------------------
# Navigator: button construction + styling
# ---------------------------------------------------------------------------


def test_delete_label_button_constructed_with_destructive_style():
    """``_delete_label_btn`` must be a QPushButton attribute on the
    navigator, styled with ``DESTRUCTIVE_BTN_STYLE`` to match the other
    destructive actions in the curation UI.
    """
    src = inspect.getsource(CurationLabelNavigator._build_ui)
    assert 'self._delete_label_btn = QPushButton("Delete Label")' in src, (
        "self._delete_label_btn must be constructed as a QPushButton "
        "labeled 'Delete Label' in _build_ui"
    )
    assert "self._delete_label_btn.setStyleSheet(DESTRUCTIVE_BTN_STYLE)" in src, (
        "Delete Label button must use DESTRUCTIVE_BTN_STYLE (already "
        "imported in _build_ui's local imports)"
    )


def test_delete_label_button_has_tooltip():
    """Tooltip must explain destructiveness + undo path."""
    src = inspect.getsource(CurationLabelNavigator._build_ui)
    assert "self._delete_label_btn.setToolTip(" in src, (
        "Delete Label button must have setToolTip(...) explaining "
        "what it does"
    )


def test_delete_label_button_added_under_new_label_in_sidebar():
    """Layout placement: the Delete Label addWidget must occur AFTER
    the New Label addWidget in the same sidebar layout — create on top,
    destroy underneath, vertically paired."""
    src = inspect.getsource(CurationLabelNavigator._build_ui)
    assert "sidebar_layout.addWidget(self._delete_label_btn)" in src, (
        "Delete Label button must be added to sidebar_layout"
    )
    # Find the order — New Label add must precede Delete Label add.
    new_idx = src.find("sidebar_layout.addWidget(self._new_label_btn)")
    del_idx = src.find("sidebar_layout.addWidget(self._delete_label_btn)")
    assert new_idx >= 0, "_new_label_btn must be added to sidebar"
    assert del_idx >= 0, "_delete_label_btn must be added to sidebar"
    assert new_idx < del_idx, (
        "+ New Label must be added BEFORE Delete Label so create sits "
        "above destroy vertically"
    )


# ---------------------------------------------------------------------------
# Navigator: callback wiring
# ---------------------------------------------------------------------------


def test_set_callbacks_accepts_on_delete_label_kwarg():
    """``set_callbacks`` must accept an ``on_delete_label`` keyword
    argument and store it on ``self._on_delete_label_cb``."""
    sig = inspect.signature(CurationLabelNavigator.set_callbacks)
    assert "on_delete_label" in sig.parameters, (
        "set_callbacks must accept on_delete_label keyword argument"
    )
    # Default must be None (no callback wired).
    assert sig.parameters["on_delete_label"].default is None
    src = inspect.getsource(CurationLabelNavigator.set_callbacks)
    assert "self._on_delete_label_cb = on_delete_label" in src, (
        "set_callbacks must assign the on_delete_label arg to "
        "self._on_delete_label_cb"
    )


def test_on_delete_label_cb_initialized_in_init():
    """``_on_delete_label_cb`` must be initialized to ``None`` in
    ``__init__`` so the attribute always exists before set_callbacks
    is called."""
    src = inspect.getsource(CurationLabelNavigator.__init__)
    assert "self._on_delete_label_cb = None" in src, (
        "self._on_delete_label_cb must be initialized to None in __init__"
    )


def test_wire_signals_connects_delete_button_click():
    """``_wire_signals`` must connect ``_delete_label_btn.clicked`` to
    ``_on_delete_label_clicked``."""
    src = inspect.getsource(CurationLabelNavigator._wire_signals)
    assert (
        "self._delete_label_btn.clicked.connect("
        "self._on_delete_label_clicked)" in src
    ), (
        "_wire_signals must connect _delete_label_btn.clicked -> "
        "_on_delete_label_clicked"
    )


def test_on_delete_label_clicked_handler_exists_and_fires_cb():
    """The click handler must exist, read ``current_label_id()``, fire
    the callback, and respect the auto-advance checkbox."""
    assert hasattr(CurationLabelNavigator, "_on_delete_label_clicked"), (
        "CurationLabelNavigator must define _on_delete_label_clicked"
    )
    src = inspect.getsource(
        CurationLabelNavigator._on_delete_label_clicked
    )
    # Reads current label id (None-guarded).
    assert "self.current_label_id()" in src, (
        "_on_delete_label_clicked must read the current label id"
    )
    # Fires the stored callback (callable form).
    assert "self._on_delete_label_cb" in src, (
        "_on_delete_label_clicked must reference self._on_delete_label_cb"
    )
    # Auto-advance after the destructive action — same UX as tier flags.
    assert "self._auto_advance_cb.isChecked()" in src, (
        "_on_delete_label_clicked must check the auto-advance checkbox"
    )
    assert "self._advance_to_next_visible()" in src, (
        "_on_delete_label_clicked must call _advance_to_next_visible "
        "when auto-advance is on"
    )


# ---------------------------------------------------------------------------
# Navigator: button enable-state gating
# ---------------------------------------------------------------------------


def test_set_label_action_buttons_enabled_method_exists():
    """The navigator must expose a method that toggles BOTH the
    Delete Label and the + New Label buttons together."""
    assert hasattr(
        CurationLabelNavigator, "set_label_action_buttons_enabled"
    ), (
        "CurationLabelNavigator must expose "
        "set_label_action_buttons_enabled for the annotator to toggle "
        "both label-action buttons together"
    )
    src = inspect.getsource(
        CurationLabelNavigator.set_label_action_buttons_enabled
    )
    # Must touch BOTH buttons in the same call.
    assert "self._delete_label_btn.setEnabled(" in src, (
        "set_label_action_buttons_enabled must call setEnabled on "
        "_delete_label_btn"
    )
    assert "self._new_label_btn.setEnabled(" in src, (
        "set_label_action_buttons_enabled must call setEnabled on "
        "_new_label_btn"
    )


# ---------------------------------------------------------------------------
# Annotator: destructive handler
# ---------------------------------------------------------------------------


def test_curation3d_has_delete_label_handler():
    """``Curation3DAnnotator`` must define
    ``_on_delete_label_from_navigator`` as the destructive handler
    bound to the navigator's on_delete_label callback."""
    assert hasattr(
        Curation3DAnnotator, "_on_delete_label_from_navigator"
    ), (
        "Curation3DAnnotator must define _on_delete_label_from_navigator"
    )


def test_delete_handler_follows_split_accept_canonical_pattern():
    """The handler must:
      1. Defensive int() cast and >0 check on label_id.
      2. Resolve shared_data from sync_manager.
      3. Snapshot via _snapshot_label_roi (existing helper).
      4. Mutate the volume in place.
      5. Call _sync_all_views_after_operation.
      6. Fire _on_delete_label_cb with roi_slices + old_roi_data.
    """
    src = inspect.getsource(
        Curation3DAnnotator._on_delete_label_from_navigator
    )
    # Defensive type cast.
    assert "int(label_id)" in src, (
        "Handler must do a defensive int(label_id) cast"
    )
    assert "lid <= 0" in src or "lid > 0" in src, (
        "Handler must reject non-positive label ids"
    )
    # Shared data resolution.
    assert "self.sync_manager.shared_labels" in src, (
        "Handler must resolve shared_data from sync_manager.shared_labels"
    )
    # Snapshot via existing helper — DO NOT modify _snapshot_label_roi.
    assert "self._snapshot_label_roi(" in src, (
        "Handler must call self._snapshot_label_roi to capture the ROI"
    )
    # Mutate in place — zero the label inside the snapshotted ROI view.
    assert "roi_view = shared_data[roi_slices]" in src, (
        "Handler must take a view of shared_data[roi_slices] and "
        "mutate it in place"
    )
    assert "roi_view[roi_view == lid] = 0" in src, (
        "Handler must zero out the label_id inside the ROI view"
    )
    # Refresh all views — DO NOT modify _sync_all_views_after_operation.
    assert "self._sync_all_views_after_operation(" in src, (
        "Handler must call _sync_all_views_after_operation to refresh "
        "main + subviewers + isolation proxy"
    )
    # Fire callback to controller.
    assert "self._on_delete_label_cb(" in src, (
        "Handler must fire self._on_delete_label_cb with the mutation "
        "metadata so the controller commits the action to its "
        "curation-undo stack"
    )
    assert "roi_slices=roi_slices" in src, (
        "Handler must pass roi_slices= as a keyword to the callback"
    )
    assert "old_roi_data=old_roi_data" in src, (
        "Handler must pass old_roi_data= as a keyword to the callback"
    )


def test_delete_handler_gates_against_in_flight_ops():
    """Silent no-op when any refinement op is staged OR an uncommitted
    paint stroke is in flight OR the merge dialog is open.  The
    handler delegates the predicate to ``_label_action_buttons_active``
    so a single source of truth covers all gates."""
    src = inspect.getsource(
        Curation3DAnnotator._on_delete_label_from_navigator
    )
    # The handler must consult the unified gate predicate.
    assert "self._label_action_buttons_active()" in src, (
        "Delete handler must consult _label_action_buttons_active() "
        "as the unified gate (covers refine ops + draw snapshot + "
        "merge dialog)"
    )


# ---------------------------------------------------------------------------
# Canonical state-machine predicate: RefinementOpState.can_start_destructive_label_op
# ---------------------------------------------------------------------------


def test_refinement_op_state_has_destructive_predicate():
    """``RefinementOpState`` must expose
    ``can_start_destructive_label_op`` as the canonical predicate for
    "is a destructive label-action button safe to enable now?".
    """
    assert hasattr(RefinementOpState, "can_start_destructive_label_op"), (
        "RefinementOpState must define can_start_destructive_label_op"
    )


def test_destructive_predicate_true_at_idle():
    """Idle dataclass (all flags False / None) → predicate returns True."""
    s = RefinementOpState()
    assert s.can_start_destructive_label_op() is True


def test_destructive_predicate_false_when_interp_active():
    s = RefinementOpState(interp_active=True)
    assert s.can_start_destructive_label_op() is False


def test_destructive_predicate_false_when_split_active():
    s = RefinementOpState(split_active=True)
    assert s.can_start_destructive_label_op() is False


def test_destructive_predicate_false_when_smooth_active():
    s = RefinementOpState(smooth_active=True)
    assert s.can_start_destructive_label_op() is False


def test_destructive_predicate_false_when_draw_snapshot_present():
    s = RefinementOpState(draw_snapshot_present=True)
    assert s.can_start_destructive_label_op() is False


def test_destructive_predicate_false_when_merge_dialog_open():
    s = RefinementOpState(merge_dialog_open=True)
    assert s.can_start_destructive_label_op() is False


def test_refinement_op_state_has_new_fields():
    """The two new fields mirror annotator-side flags into the
    dataclass so the canonical predicate can read them."""
    s = RefinementOpState()
    assert hasattr(s, "draw_snapshot_present")
    assert hasattr(s, "merge_dialog_open")
    # Defaults are clean / idle.
    assert s.draw_snapshot_present is False
    assert s.merge_dialog_open is False


# ---------------------------------------------------------------------------
# Delegation: _label_action_buttons_active is a thin wrapper
# ---------------------------------------------------------------------------


def test_label_action_buttons_active_delegates_to_canonical_predicate():
    """``_label_action_buttons_active`` must be a thin delegate that
    calls ``_sync_refine_state()`` then reads the canonical predicate
    off ``self._refine_state``.  Inline gating logic (the previous
    agent's _interp_running / _draw_snapshot / _merge_dialog inline
    checks) MUST be gone."""
    pred_src = inspect.getsource(
        Curation3DAnnotator._label_action_buttons_active
    )
    # Must sync the dataclass before reading.
    assert "self._sync_refine_state()" in pred_src, (
        "_label_action_buttons_active must call _sync_refine_state() "
        "before reading the canonical predicate"
    )
    # Must delegate to the canonical predicate.
    assert (
        "self._refine_state.can_start_destructive_label_op()" in pred_src
    ), (
        "_label_action_buttons_active must delegate to "
        "self._refine_state.can_start_destructive_label_op()"
    )


def test_label_action_buttons_active_no_inline_gating_logic():
    """Negative assertion: the previous agent's inline gating checks
    (``_refinement_op_active()`` and ``getattr(self, '_draw_snapshot'...)``
    style direct reads inside the method body) must be gone.  The
    canonical predicate owns the logic now.
    """
    pred_src = inspect.getsource(
        Curation3DAnnotator._label_action_buttons_active
    )
    # No direct call to _refinement_op_active() inside the method body.
    assert "self._refinement_op_active()" not in pred_src, (
        "_label_action_buttons_active must NOT inline "
        "_refinement_op_active() — the canonical predicate covers it"
    )
    # No direct getattr / attribute read of _draw_snapshot inline.
    assert "_draw_snapshot" not in pred_src, (
        "_label_action_buttons_active must NOT inline a _draw_snapshot "
        "check — the canonical predicate covers it via "
        "draw_snapshot_present mirrored by _sync_refine_state"
    )
    # No direct getattr / attribute read of _merge_dialog inline.
    assert "_merge_dialog" not in pred_src, (
        "_label_action_buttons_active must NOT inline a _merge_dialog "
        "check — the canonical predicate covers it via "
        "merge_dialog_open mirrored by _sync_refine_state"
    )


def test_sync_refine_state_mirrors_new_flags():
    """``_sync_refine_state`` must mirror ``_draw_snapshot`` and
    ``_merge_dialog`` into the canonical dataclass so the predicate
    sees them.
    """
    src = inspect.getsource(Curation3DAnnotator._sync_refine_state)
    assert "draw_snapshot_present" in src, (
        "_sync_refine_state must set draw_snapshot_present on the "
        "dataclass"
    )
    assert "merge_dialog_open" in src, (
        "_sync_refine_state must set merge_dialog_open on the "
        "dataclass"
    )
    assert "_draw_snapshot" in src, (
        "_sync_refine_state must read self._draw_snapshot to set "
        "draw_snapshot_present"
    )
    assert "_merge_dialog" in src, (
        "_sync_refine_state must read self._merge_dialog to set "
        "merge_dialog_open"
    )


# ---------------------------------------------------------------------------
# Annotator: wiring
# ---------------------------------------------------------------------------


def test_wire_navigator_callbacks_passes_on_delete_label():
    """``_wire_navigator_callbacks`` must pass
    ``on_delete_label=self._on_delete_label_from_navigator`` so the
    navigator click is routed to the destructive handler."""
    src = inspect.getsource(
        Curation3DAnnotator._wire_navigator_callbacks
    )
    assert (
        "on_delete_label=self._on_delete_label_from_navigator" in src
    ), (
        "_wire_navigator_callbacks must pass on_delete_label="
        "self._on_delete_label_from_navigator to nav.set_callbacks"
    )


def test_refresh_merge_button_state_is_single_responsibility():
    """``_refresh_merge_button_state`` must ONLY update the merge
    button.  The label-action refresh lives in the sibling method
    ``_refresh_label_action_buttons_state``.  Sharing a single
    refresh function for two unrelated UI areas was a side-channel
    — the post-refactor shape has two single-responsibility refreshers
    that the call-sites invoke together.
    """
    src = inspect.getsource(
        Curation3DAnnotator._refresh_merge_button_state
    )
    # Must update the merge button.
    assert "set_merge_enabled" in src, (
        "_refresh_merge_button_state must call set_merge_enabled"
    )
    # MUST NOT also touch the label-action buttons.
    assert "set_label_action_buttons_enabled" not in src, (
        "_refresh_merge_button_state must NOT call "
        "set_label_action_buttons_enabled — that's the sibling "
        "_refresh_label_action_buttons_state's job"
    )


def test_refresh_label_action_buttons_state_method_exists():
    """The sibling refresher must exist as a method on the annotator."""
    assert hasattr(
        Curation3DAnnotator, "_refresh_label_action_buttons_state"
    ), (
        "Curation3DAnnotator must define "
        "_refresh_label_action_buttons_state as the dedicated refresher "
        "for the label-action buttons"
    )
    src = inspect.getsource(
        Curation3DAnnotator._refresh_label_action_buttons_state
    )
    # Must call the navigator's enable-toggle.
    assert "set_label_action_buttons_enabled" in src, (
        "_refresh_label_action_buttons_state must call "
        "set_label_action_buttons_enabled on the navigator"
    )
    # Must consult the canonical predicate.
    assert "can_start_destructive_label_op" in src, (
        "_refresh_label_action_buttons_state must read the canonical "
        "predicate can_start_destructive_label_op off self._refine_state"
    )


def test_refresh_label_action_buttons_state_called_alongside_merge_refresh():
    """Every site that calls ``_refresh_merge_button_state`` must
    ALSO call ``_refresh_label_action_buttons_state`` — they're paired
    because op-state transitions affect both gates.  This is the
    integration glue that replaced the previous agent's side-channel.
    """
    import re
    src = inspect.getsource(Curation3DAnnotator)
    # Count the call sites of each (excluding the method definitions /
    # docstring mentions).
    merge_calls = len(re.findall(
        r"self\._refresh_merge_button_state\(\)", src
    ))
    label_calls = len(re.findall(
        r"self\._refresh_label_action_buttons_state\(\)", src
    ))
    assert merge_calls > 0, (
        "expected _refresh_merge_button_state to be called from the "
        "annotator"
    )
    assert merge_calls == label_calls, (
        f"_refresh_merge_button_state call count ({merge_calls}) "
        f"must equal _refresh_label_action_buttons_state call count "
        f"({label_calls}) — every site must call both"
    )


# ---------------------------------------------------------------------------
# Negative assertions — abandoned-branch artifacts must NOT appear
# ---------------------------------------------------------------------------


def test_no_tier_undo_stack_in_delete_handler():
    """The abandoned branch (0ae3039) invented a parallel undo path on
    ``_tier_undo_stack``.  The canonical path is the curation-undo
    system (commit via controller -> CurationUndoEntry).  Ensure the
    new handler does NOT reference _tier_undo_stack."""
    handler_src = inspect.getsource(
        Curation3DAnnotator._on_delete_label_from_navigator
    )
    assert "_tier_undo_stack" not in handler_src, (
        "Delete handler must NOT push to _tier_undo_stack — that's "
        "the abandoned 0ae3039 mistake.  Undo flows through the "
        "controller's curation-undo system via _on_delete_label_cb."
    )

    nav_click_src = inspect.getsource(
        CurationLabelNavigator._on_delete_label_clicked
    )
    assert "_tier_undo_stack" not in nav_click_src, (
        "Navigator click handler must NOT reference _tier_undo_stack"
    )


def test_no_4tuple_format_in_delete_handler():
    """The abandoned branch used a 4-tuple format on _tier_undo_stack
    entries.  The canonical path uses keyword args (roi_slices=,
    old_roi_data=) on the callback.  No 4-tuple literals should
    appear in either handler."""
    handler_src = inspect.getsource(
        Curation3DAnnotator._on_delete_label_from_navigator
    )
    # Heuristic: a 4-tuple format would look like
    # `(lid, "delete", roi_slices, old_roi_data)` or similar.  Look for
    # tell-tale shapes.
    import re
    # Match parenthesized 4-element tuples that include a string-literal
    # tier / op name — the abandoned-branch signature.
    pat = re.compile(r"\(\s*\w+\s*,\s*['\"]\w+['\"]\s*,\s*\w+\s*,\s*\w+\s*\)")
    assert not pat.search(handler_src), (
        "Delete handler must NOT use a 4-tuple (lid, op_str, roi, "
        "old_data) format — that's the abandoned 0ae3039 mistake.  "
        "Use the keyword-arg callback signature instead."
    )
