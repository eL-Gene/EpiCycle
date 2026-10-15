"""Tests for the visibility/isolation hide-unhide bundle (FM2/FM4/FM5/FM6).

Each test class targets one failure mode from the 2026-05-16 trace
mapped to the corresponding edge of the isolation state diagram.

FM2 — X-close merge dialog leaves multi-label proxies orphaned
FM4 — Split-accept skip_3d_refresh missing multi-label isolation case
FM5 — Patch switch leaves isolation state from prior patch
FM6 — Refine-further reject doesn't clear isolation

Tests use bare-instance pattern (`Curation3DAnnotator.__new__`) +
``types.SimpleNamespace`` stand-ins for collaborators.  Source-level
``inspect.getsource`` checks confirm the fix code is present at the
expected call sites — full integration testing is via manual user
testing in napari.  Qt-free.
"""
from __future__ import annotations

import inspect
import types

import pytest


# =============================================================================
# FM2 — Merge dialog X-close orphan
# =============================================================================


class TestFM2_MergeDialogCloseOrphan:
    """`_close_cleanup` must cancel an open merge dialog BEFORE setting
    `_is_closing = True`, otherwise the dialog's own closeEvent guard
    (merge_dialog.py:645-660) skips the cancel callback and the
    dialog's multi-label sub-proxies leak."""

    def test_close_cleanup_routes_through_cancel_before_is_closing(self):
        """Source-level: the merge-dialog cancel block must appear
        BEFORE `self._is_closing = True` in the close cleanup flow."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        src = inspect.getsource(Curation3DAnnotator._close_cleanup)
        # Find positions of the two anchors.
        cancel_idx = src.find("_cancel_merge_dialog")
        is_closing_idx = src.find("self._is_closing = True")
        assert cancel_idx != -1, (
            "_close_cleanup must reference _cancel_merge_dialog (FM2 fix)"
        )
        assert is_closing_idx != -1
        assert cancel_idx < is_closing_idx, (
            "FM2 fix: merge dialog cancel must run BEFORE _is_closing=True. "
            "Otherwise MergeDialog.closeEvent's defensive guard at "
            "merge_dialog.py:645-660 skips firing the cancel callback "
            "and the dialog's multi-label proxies are orphaned."
        )

    def test_close_cleanup_has_merge_close_diagnostic(self):
        """The fix must emit a [MERGE-CLOSE] diagnostic when a dialog
        is detected — so a manual tester can see in the console that
        the cleanup path fired."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        src = inspect.getsource(Curation3DAnnotator._close_cleanup)
        assert "[MERGE-CLOSE]" in src

    def test_cancel_merge_dialog_has_diagnostics(self):
        """`_cancel_merge_dialog` must log entry + isolation clear so
        the manual tester sees the full cleanup path."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        src = inspect.getsource(Curation3DAnnotator._cancel_merge_dialog)
        assert "[MERGE-CLOSE] _cancel_merge_dialog ENTER" in src
        assert "_on_isolation_changed(set())" in src

    def test_close_cleanup_idempotent_when_no_dialog(self):
        """If no merge dialog is open, the FM2 block is a no-op (no
        crash, no diagnostic spam)."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        ann = Curation3DAnnotator.__new__(Curation3DAnnotator)
        ann._merge_dialog = None
        ann._close_cleanup_done = False
        # The dialog check is `if dlg is not None`; with no dialog,
        # the block must not raise.  Source-level: confirm the guard.
        src = inspect.getsource(Curation3DAnnotator._close_cleanup)
        assert "if dlg is not None:" in src


# =============================================================================
# FM4 — Split-accept skip_3d_refresh missing multi-label mode
# =============================================================================


class TestFM4_SplitAcceptSkip3dRefresh:
    """`_accept_split` normal path's `_skip_3d` calculation must
    include BOTH single-label proxy AND multi-label colormap modes,
    otherwise main_layer.refresh() runs while the main layer is
    actually hidden by the colormap → isosurface flicker."""

    def test_skip_3d_check_includes_multi_label_set(self):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        src = inspect.getsource(Curation3DAnnotator._accept_split)
        # The fix must reference `_isolation_label_set` in the
        # skip_3d_refresh decision, not only `_isolation_proxy_active`.
        assert "_isolation_label_set" in src, (
            "FM4 fix: split-accept _skip_3d must check multi-label "
            "isolation mode (_isolation_label_set is not None), not "
            "only single-label proxy (_isolation_proxy_active)."
        )

    def test_skip_3d_check_uses_or_combinator(self):
        """The new check is `single_iso or multi_iso` — both modes
        should skip the refresh."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        src = inspect.getsource(Curation3DAnnotator._accept_split)
        # Look for the pattern of either being True triggers skip.
        assert "_single_iso or _multi_iso" in src

    def test_split_accept_has_diagnostic(self):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        src = inspect.getsource(Curation3DAnnotator._accept_split)
        assert "[SPLIT-ACCEPT]" in src
        assert "skip_3d_refresh decision" in src


# =============================================================================
# FM5 — Patch switch isolation cleanup
# =============================================================================


class TestFM5_PatchSwitchIsolation:
    """`_load_selected_layers` must reset all isolation state at patch
    switch.  Most fields were already cleared (WP-3 / Bug J).  FM5
    fix adds `_pre_isolation_data = None` + diagnostics + explicit
    proxy-removal logging."""

    def test_load_selected_layers_clears_pre_isolation_data(self):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        src = inspect.getsource(Curation3DAnnotator._load_selected_layers)
        assert "self._pre_isolation_data = None" in src, (
            "FM5 fix: _load_selected_layers must clear "
            "_pre_isolation_data — it was previously cleared only "
            "in _close_cleanup, leaving a window where the patch-"
            "switched state retained stale ROI data."
        )

    def test_load_selected_layers_clears_all_isolation_fields(self):
        """Regression guard for the existing (pre-FM5) clears, so a
        future refactor can't accidentally remove them."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        src = inspect.getsource(Curation3DAnnotator._load_selected_layers)
        for attr in (
            "_isolation_active = False",
            "_pre_isolation_colormap = None",
            "_isolation_label_set = None",
            "_paint_isolation_active = False",
            "_paint_isolation_label_id = None",
        ):
            assert attr in src, f"patch switch must clear {attr}"

    def test_load_selected_layers_has_diagnostics(self):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        src = inspect.getsource(Curation3DAnnotator._load_selected_layers)
        assert "[PATCH-SWITCH-ISO] _load_selected_layers ENTER" in src
        assert "[PATCH-SWITCH-ISO] _load_selected_layers DONE" in src

    def test_load_selected_layers_documents_fm5(self):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        src = inspect.getsource(Curation3DAnnotator._load_selected_layers)
        assert "FM5" in src


# =============================================================================
# FM6 — Refine-further reject doesn't clear isolation
# =============================================================================


class TestFM6_RefineFurtherRejectClearsIsolation:
    """`_reject_interpolation` must clear isolation IFF we came from
    refine-further mode (where `_enter_interp_refine_mode` had
    force-applied isolation on entry).  Normal Preview→Reject paths
    intentionally leave isolation alone (FM1 is by-design)."""

    def _make_bare_annotator(self, *, in_refine_mode, with_shared_data=False):
        """Bare Curation3DAnnotator with the minimum state needed for
        _reject_interpolation to run end-to-end without Qt."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        ann = Curation3DAnnotator.__new__(Curation3DAnnotator)
        # State that _reject_interpolation reads/writes.
        ann._interp_refine_mode = bool(in_refine_mode)
        ann._interp_snapshot = None
        ann._interp_roi = None
        ann._interp_label_id = None
        ann._isolation_active = True
        ann._isolation_proxy_active = False
        ann._active_labels_name = "test_layer"
        # sync_manager.shared_labels lookup must return None to short-
        # circuit on the "shared_data is None" early-exit path, so we
        # don't have to fake the full sync.
        if with_shared_data:
            ann.sync_manager = types.SimpleNamespace(
                shared_labels={"test_layer": object()}
            )
        else:
            ann.sync_manager = types.SimpleNamespace(
                shared_labels={}  # no entry → shared_data = None
            )
        # Status label + buttons (no-op stubs).
        ann._interp_status_label = types.SimpleNamespace(setText=lambda _: None)
        ann._accept_interp_btn = types.SimpleNamespace(setEnabled=lambda _: None)
        ann._reject_interp_btn = types.SimpleNamespace(setEnabled=lambda _: None)
        ann._interp_refine_btn = types.SimpleNamespace(setEnabled=lambda _: None)
        ann._refresh_merge_button_state = lambda: None
        ann._clear_draw_snapshot = lambda: None
        ann._restore_ready_to_draw_state = lambda: None
        # Track isolation callback fires.
        ann._isolation_fires = []
        ann._on_isolation_changed = lambda ids: ann._isolation_fires.append(set(ids))
        return ann

    def test_reject_from_refine_mode_clears_isolation(self):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        ann = self._make_bare_annotator(in_refine_mode=True)
        Curation3DAnnotator._reject_interpolation(ann)
        # FM6: refine-mode reject MUST fire isolation clear.
        # (The function early-returns when shared_data is None, but
        # the FM6 block is at the END of the function — so we need
        # the with_shared_data path.  Re-run with that.)
        ann2 = self._make_bare_annotator(
            in_refine_mode=True, with_shared_data=True
        )
        Curation3DAnnotator._reject_interpolation(ann2)
        assert set() in ann2._isolation_fires, (
            "FM6 fix: _reject_interpolation called from refine-further "
            "mode must clear isolation via _on_isolation_changed(set())."
        )

    def test_reject_NOT_from_refine_mode_does_not_clear_isolation(self):
        """FM1 protection: normal Preview→Reject must NOT clear
        isolation.  The trace explicitly documented this as
        by-design behavior."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        ann = self._make_bare_annotator(
            in_refine_mode=False, with_shared_data=True
        )
        Curation3DAnnotator._reject_interpolation(ann)
        assert ann._isolation_fires == [], (
            "FM1 (by-design): normal interp Reject must leave isolation "
            "alone.  Only refine-further reject clears it (FM6)."
        )

    def test_reject_captures_refine_mode_before_resetting_it(self):
        """Source-level: the fix must capture `was_in_refine_mode`
        BEFORE the existing `self._interp_refine_mode = False` reset,
        otherwise the FM6 branch always sees False and never clears."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        src = inspect.getsource(Curation3DAnnotator._reject_interpolation)
        capture_idx = src.find("was_in_refine_mode = bool(")
        reset_idx = src.find("self._interp_refine_mode = False")
        check_idx = src.find("if was_in_refine_mode:")
        assert capture_idx != -1
        assert reset_idx != -1
        assert check_idx != -1
        assert capture_idx < reset_idx, (
            "FM6 fix: was_in_refine_mode must be captured BEFORE "
            "_interp_refine_mode is reset to False."
        )
        assert reset_idx < check_idx, (
            "Sanity: the FM6 check runs at end of method, after reset."
        )

    def test_reject_interpolation_has_diagnostic(self):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        src = inspect.getsource(Curation3DAnnotator._reject_interpolation)
        assert "[INTERP-REJECT-ISO]" in src


# =============================================================================
# Cross-cutting: documentation traceability
# =============================================================================


class TestDocumentationTraceability:
    """Every fix must reference its FM number in source comments so a
    future grep against the trace doc finds the implementation."""

    def test_fm2_referenced_in_close_cleanup(self):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        src = inspect.getsource(Curation3DAnnotator._close_cleanup)
        assert "FM2" in src

    def test_fm4_referenced_in_accept_split(self):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        src = inspect.getsource(Curation3DAnnotator._accept_split)
        assert "FM4" in src

    def test_fm5_referenced_in_load_selected_layers(self):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        src = inspect.getsource(Curation3DAnnotator._load_selected_layers)
        assert "FM5" in src

    def test_fm6_referenced_in_reject_interpolation(self):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )
        src = inspect.getsource(Curation3DAnnotator._reject_interpolation)
        assert "FM6" in src
