"""Switch-safety hardening tests for CurationLabelNavigator.

These cover the navigator rebuild-timer cancellation that closes the
unclosed door on the B5 navigator access-violation crash:

  - ``_do_pending_rebuild`` short-circuits on the ``_destroyed`` flag
    BEFORE touching the (possibly mid-destruction) Qt table.
  - ``_schedule_rebuild`` refuses to arm a new deferred rebuild once the
    dock is being torn down.
  - ``closeEvent`` sets ``_destroyed``; ``reset_selection_ui`` clears it
    so a reused dock can schedule rebuilds again.
  - the lone bare ``clearSelection()`` in ``_on_search_enter`` carries the
    B5 selection-model signal-block.

All tests use ``__new__`` to build a navigator without Qt ``__init__`` and
set only the attributes each method touches (no display required).
"""

from __future__ import annotations

import inspect
import types


def _bare_nav():
    from napari_epicycle.annotator.curation_label_navigator import (
        CurationLabelNavigator,
    )
    return CurationLabelNavigator.__new__(CurationLabelNavigator)


# ---------------------------------------------------------------------------
# _do_pending_rebuild short-circuit
# ---------------------------------------------------------------------------


def test_do_pending_rebuild_short_circuits_when_destroyed():
    nav = _bare_nav()
    nav._destroyed = True
    nav._rebuild_pending = True
    calls = []
    nav._rebuild_table = lambda: calls.append("rebuilt")
    # A live table is present — the _destroyed gate must fire FIRST.
    nav._table = types.SimpleNamespace(rowCount=lambda: 5)
    nav._do_pending_rebuild()
    assert calls == [], "rebuild must be skipped when _destroyed is True"


def test_do_pending_rebuild_runs_when_not_destroyed():
    nav = _bare_nav()
    nav._destroyed = False
    nav._rebuild_pending = True
    calls = []
    nav._rebuild_table = lambda: calls.append("rebuilt")
    nav._table = types.SimpleNamespace(rowCount=lambda: 5)
    nav._do_pending_rebuild()
    assert calls == ["rebuilt"]
    assert nav._rebuild_pending is False


def test_do_pending_rebuild_survives_runtimeerror_on_destroyed():
    """If the wrapper is already sip-deleted, reading ``_destroyed`` raises
    RuntimeError — must be caught (no rebuild, no crash)."""
    from napari_epicycle.annotator.curation_label_navigator import (
        CurationLabelNavigator,
    )
    calls = []

    class _Boom:
        @property
        def _destroyed(self):
            raise RuntimeError("wrapped C/C++ object has been deleted")

        def _rebuild_table(self):
            calls.append("rebuilt")

    # Call the production method unbound on a boom object.
    CurationLabelNavigator._do_pending_rebuild(_Boom())
    assert calls == []


def test_do_pending_rebuild_survives_missing_destroyed_attr():
    """AttributeError (flag never set) is also caught — defensive."""
    nav = _bare_nav()  # _destroyed intentionally absent
    calls = []
    nav._rebuild_table = lambda: calls.append("rebuilt")
    nav._table = types.SimpleNamespace(rowCount=lambda: 5)
    nav._do_pending_rebuild()
    assert calls == []


# ---------------------------------------------------------------------------
# _schedule_rebuild guard
# ---------------------------------------------------------------------------


def test_schedule_rebuild_skipped_when_destroyed():
    nav = _bare_nav()
    nav._destroyed = True
    nav._rebuild_pending = False
    nav._schedule_rebuild()
    # Must not arm a timer → _rebuild_pending stays False.
    assert nav._rebuild_pending is False


def test_schedule_rebuild_arms_timer_when_alive(monkeypatch):
    nav = _bare_nav()
    nav._destroyed = False
    nav._rebuild_pending = False
    captured = []
    import qtpy.QtCore
    monkeypatch.setattr(
        qtpy.QtCore.QTimer, "singleShot",
        lambda ms, cb: captured.append((ms, cb)),
    )
    nav._schedule_rebuild()
    assert nav._rebuild_pending is True
    assert len(captured) == 1
    assert captured[0][0] == 0


# ---------------------------------------------------------------------------
# closeEvent / reset_selection_ui flag lifecycle
# ---------------------------------------------------------------------------


def test_close_event_sets_destroyed():
    nav = _bare_nav()
    nav._destroyed = False
    # super().closeEvent on an uninitialized QWidget raises RuntimeError,
    # which the override swallows AFTER setting the flag.
    nav.closeEvent(None)
    assert nav._destroyed is True


def _wire_reset_fakes(nav):
    class _SM:
        def blockSignals(self, b):
            pass

        def clearSelection(self):
            pass

    nav._table = types.SimpleNamespace(
        selectionModel=lambda: _SM(),
        setCurrentCell=lambda r, c: None,
    )
    nav._editing_header = types.SimpleNamespace(setText=lambda s: None)
    nav._isolate_cb = types.SimpleNamespace(
        blockSignals=lambda b: None, setChecked=lambda b: None,
    )


def test_reset_selection_ui_clears_destroyed():
    nav = _bare_nav()
    nav._destroyed = True
    _wire_reset_fakes(nav)
    nav.reset_selection_ui()
    assert nav._destroyed is False


def test_reset_selection_ui_clears_sticky_rebuild_pending():
    """A stale ``_rebuild_pending=True`` (timer dropped on a prior session)
    must be cleared on (re)use so ``_schedule_rebuild`` is not wedged
    forever on the reused dock."""
    nav = _bare_nav()
    nav._destroyed = True
    nav._rebuild_pending = True  # sticky from a prior session
    _wire_reset_fakes(nav)
    nav.reset_selection_ui()
    assert nav._rebuild_pending is False
    assert nav._destroyed is False


# ---------------------------------------------------------------------------
# the canonical scenario, end to end
# ---------------------------------------------------------------------------


def test_scheduled_rebuild_after_close_is_noop(monkeypatch):
    """Schedule a rebuild, tear down (closeEvent), then fire the queued
    timer — it must be a no-op instead of touching the dead widget."""
    nav = _bare_nav()
    nav._destroyed = False
    nav._rebuild_pending = False
    calls = []
    nav._rebuild_table = lambda: calls.append("rebuilt")
    nav._table = types.SimpleNamespace(rowCount=lambda: 3)
    captured = []
    import qtpy.QtCore
    monkeypatch.setattr(
        qtpy.QtCore.QTimer, "singleShot",
        lambda ms, cb: captured.append(cb),
    )
    # 1. A data update schedules a deferred rebuild.
    nav._schedule_rebuild()
    assert len(captured) == 1
    # 2. The dock is torn down (closeEvent) BEFORE the timer fires.
    nav.closeEvent(None)
    assert nav._destroyed is True
    # 3. The queued timer fires on the now-torn-down widget.
    captured[0]()
    assert calls == [], "rebuild after teardown must be a no-op"


def test_reuse_after_close_rearms_rebuild(monkeypatch):
    """After a close, a reused dock (reset_selection_ui) must be able to
    schedule rebuilds again — the flag is not sticky across reuse."""
    nav = _bare_nav()
    nav._destroyed = False
    nav._rebuild_pending = False
    nav.closeEvent(None)
    assert nav._destroyed is True
    # Reuse path resets via reset_selection_ui.
    _wire_reset_fakes(nav)
    nav.reset_selection_ui()
    assert nav._destroyed is False
    # Now scheduling works again.
    captured = []
    import qtpy.QtCore
    monkeypatch.setattr(
        qtpy.QtCore.QTimer, "singleShot",
        lambda ms, cb: captured.append(cb),
    )
    nav._rebuild_pending = False
    nav._schedule_rebuild()
    assert len(captured) == 1


# ---------------------------------------------------------------------------
# bare clearSelection in _on_search_enter carries the B5 signal-block
# ---------------------------------------------------------------------------


def test_search_enter_clearselection_blocks_signals():
    from napari_epicycle.annotator.curation_label_navigator import (
        CurationLabelNavigator,
    )
    src = inspect.getsource(CurationLabelNavigator._on_search_enter)
    assert "blockSignals(True)" in src
    assert "clearSelection()" in src
    assert "blockSignals(False)" in src
    # Ordering: block(True) → clearSelection → block(False).
    i_on = src.index("blockSignals(True)")
    i_clear = src.index("clearSelection()")
    i_off = src.index("blockSignals(False)")
    assert i_on < i_clear < i_off
