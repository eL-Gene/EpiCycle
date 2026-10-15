"""closeEvent / orphan-prevention tests for MergeDialog.

Background: the dialog can be closed by paths OTHER than the explicit
Accept / Cancel buttons — window X, Esc (already wired to cancel),
parent destroyed, napari dock close, etc. Without a closeEvent
override, those paths do NOT fire the on_cancel callback, leaving:

  - the per-label multi-label sub-proxies orphaned in the viewer
    (created by Curation3DAnnotator._apply_multi_label_proxy when
    the user ticked candidates);
  - the volume snapshot un-restored if a Merge Selected was performed.

Subsequent operations — particularly napari `ndisplay` switches that
trigger iso_categorical re-renders — can crash on the orphaned proxy
state.

Fix: closeEvent override on MergeDialog fires on_cancel for any close
path NOT marked clean. Accept / Cancel button handlers set
``_clean_close = True`` so they are not double-handled.

These tests are source-inspection (Qt-free), following the pattern of
tests/annotator/test_edt_overlay_lock.py.
"""

from __future__ import annotations

import inspect


def _dialog_cls():
    from napari_epicycle.annotator.merge_dialog import MergeDialog
    return MergeDialog


class TestCleanCloseGuardField:
    def test_init_initializes_clean_close_flag(self):
        src = inspect.getsource(_dialog_cls().__init__)
        assert "_clean_close = False" in src, (
            "__init__ must initialize _clean_close = False so closeEvent "
            "knows whether the close was explicit (Accept/Cancel) or "
            "needs cancel-callback fallback"
        )


class TestExplicitButtonsSetCleanClose:
    """Accept and Cancel button handlers must set _clean_close = True
    so closeEvent treats them as already-handled."""

    def test_accept_handler_sets_clean_close(self):
        src = inspect.getsource(_dialog_cls()._on_accept_clicked)
        assert "_clean_close = True" in src, (
            "_on_accept_clicked must set _clean_close = True so the "
            "subsequent close() call doesn't re-fire the cancel callback"
        )

    def test_cancel_handler_sets_clean_close(self):
        src = inspect.getsource(_dialog_cls()._on_cancel_clicked)
        assert "_clean_close = True" in src, (
            "_on_cancel_clicked must set _clean_close = True so the "
            "subsequent close() call doesn't re-fire the cancel callback "
            "(double-fire would crash on stale dialog state)"
        )

    def test_clean_close_set_BEFORE_callback(self):
        """Setting _clean_close BEFORE firing the callback ensures that
        even if the callback raises, closeEvent still treats this as a
        clean close (no double-fire of the cancel path)."""
        src = inspect.getsource(_dialog_cls()._on_accept_clicked)
        # _clean_close = True must appear before the on_accept_cb call.
        clean_idx = src.find("_clean_close = True")
        cb_idx = src.find("_on_accept_cb(")
        assert clean_idx > 0 and cb_idx > 0
        assert clean_idx < cb_idx, (
            "_clean_close = True must be set BEFORE firing the callback "
            "so an exception in the callback still results in a clean close"
        )


class TestCloseEventOverride:
    def test_close_event_method_exists(self):
        cls = _dialog_cls()
        assert "closeEvent" in cls.__dict__, (
            "MergeDialog must define closeEvent to catch X-close, "
            "parent-destroyed, and other non-Accept/Cancel close paths"
        )

    def test_close_event_fires_cancel_when_not_clean(self):
        src = inspect.getsource(_dialog_cls().closeEvent)
        # Branch on _clean_close.
        assert "_clean_close" in src
        # When not clean: fire the cancel callback.
        assert "_on_cancel_cb" in src

    def test_close_event_calls_super(self):
        """Must chain to QWidget.closeEvent so Qt's normal teardown runs."""
        src = inspect.getsource(_dialog_cls().closeEvent)
        assert "super().closeEvent" in src

    def test_close_event_is_idempotent(self):
        """After firing the cancel callback, _clean_close must be set
        so a second closeEvent (rare but possible during teardown)
        doesn't double-fire."""
        src = inspect.getsource(_dialog_cls().closeEvent)
        # _clean_close = True appears in the not-clean branch.
        assert "_clean_close = True" in src

    def test_close_event_skips_callback_during_parent_teardown(self):
        """If the parent annotator's _close_cleanup is running,
        firing on_cancel here would touch already-destroyed state
        (labels_combo, subviewers). The annotator teardown is the
        source of truth for cleanup in that case."""
        src = inspect.getsource(_dialog_cls().closeEvent)
        assert "_is_closing" in src, (
            "closeEvent must check parent._is_closing and skip the "
            "cancel callback when the parent annotator is mid-teardown"
        )


class TestEscKeyStillFiresCleanCancel:
    """The Esc key already routes to _on_cancel_clicked, which now
    sets _clean_close. Make sure the keyPressEvent path didn't
    accidentally bypass that route."""

    def test_esc_routes_to_cancel_handler(self):
        src = inspect.getsource(_dialog_cls().keyPressEvent)
        # Esc routes to _on_cancel_clicked (which sets _clean_close).
        assert "Key_Escape" in src
        assert "_on_cancel_clicked" in src
