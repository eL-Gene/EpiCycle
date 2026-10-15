"""Tests for ``fusion_reconstruction._manage_runs_panel`` pure helpers (Phase 4a).

The Qt panel itself is manual-tested (Qt teardown).  The action-enablement
decision + the dependency-warn decision are extracted into pure helpers
(``_action_enablement`` / ``_should_warn_dependents``) and tested here
without a display.
"""

from __future__ import annotations

import pytest

try:
    import qtpy  # noqa: F401
    qt_available = True
except ImportError:
    qt_available = False

pytestmark = pytest.mark.skipif(not qt_available, reason="qtpy not available")


def _enablement():
    from napari_epicycle.fusion_reconstruction._manage_runs_panel import (
        _action_enablement,
    )
    return _action_enablement


# ----------------------------------------------------------------------------
# Default-run guard
# ----------------------------------------------------------------------------

class TestDefaultRunGuard:

    def test_default_run_not_editable_archivable_deletable(self):
        # default run is synthesized (no manifest) — Edit/Archive/Delete off
        d = _enablement()("default", "default", is_default=True)
        assert d["edit"] is False
        assert d["archive"] is False
        assert d["unarchive"] is False
        assert d["delete"] is False

    def test_default_run_not_switchable_when_active(self):
        d = _enablement()("default", "default", is_default=True)
        assert d["switch"] is False

    def test_default_run_switchable_when_not_active(self):
        # active run is some named run; default is selectable to switch to
        d = _enablement()("default", "cells_v1", is_default=True)
        assert d["switch"] is True
        # still never editable / deletable
        assert d["edit"] is False
        assert d["delete"] is False


# ----------------------------------------------------------------------------
# Active-run guard
# ----------------------------------------------------------------------------

class TestActiveRunGuard:

    def test_active_named_run_undeletable(self):
        d = _enablement()("cells_v1", "cells_v1", is_default=False)
        assert d["delete"] is False
        # can't switch to the run you're already on
        assert d["switch"] is False
        # can't archive the active run (must switch off first)
        assert d["archive"] is False
        # editing the active run's metadata is fine
        assert d["edit"] is True

    def test_inactive_named_run_deletable(self):
        d = _enablement()("cells_v1", "nuclei_v2", is_default=False)
        assert d["delete"] is True
        assert d["switch"] is True
        assert d["archive"] is True
        assert d["edit"] is True


# ----------------------------------------------------------------------------
# Archive / unarchive toggling
# ----------------------------------------------------------------------------

class TestArchiveState:

    def test_archived_named_run_offers_unarchive_not_archive(self):
        d = _enablement()(
            "cells_v1", "nuclei_v2", is_default=False, is_archived=True
        )
        assert d["archive"] is False
        assert d["unarchive"] is True

    def test_unarchived_named_inactive_run_offers_archive(self):
        d = _enablement()(
            "cells_v1", "nuclei_v2", is_default=False, is_archived=False
        )
        assert d["archive"] is True
        assert d["unarchive"] is False


# ----------------------------------------------------------------------------
# Dependency-warn decision
# ----------------------------------------------------------------------------

class TestShouldWarnDependents:

    def _fn(self):
        from napari_epicycle.fusion_reconstruction._manage_runs_panel import (
            _should_warn_dependents,
        )
        return _should_warn_dependents

    def test_empty_no_warn(self):
        assert self._fn()([]) is False
        assert self._fn()(None) is False

    def test_nonempty_warns(self):
        assert self._fn()(["cells_v2"]) is True
        assert self._fn()(["a", "b"]) is True


# ----------------------------------------------------------------------------
# Clean-intermediates enablement (simpler path: any selected run is cleanable,
# including the default run, which has its own flat prediction cache).
# ----------------------------------------------------------------------------

class TestCleanIntermediatesEnablement:

    def test_clean_enabled_for_named_run(self):
        d = _enablement()("cells_v1", "nuclei_v2", is_default=False)
        assert d["clean"] is True

    def test_clean_enabled_for_default_run(self):
        d = _enablement()("default", "default", is_default=True)
        assert d["clean"] is True

    def test_clean_enabled_for_active_run(self):
        # cleaning the run you're on is fine (cache is regenerable).
        d = _enablement()("cells_v1", "cells_v1", is_default=False)
        assert d["clean"] is True


# ----------------------------------------------------------------------------
# Clean-intermediates click handler (constructed via __new__ to dodge Qt
# teardown — the handler logic is independent of the QDialog machinery).
# ----------------------------------------------------------------------------

class TestCleanIntermediatesClickHandler:

    def _panel(self, *, summary, on_clean):
        from napari_epicycle.fusion_reconstruction._manage_runs_panel import (
            ManageRunsPanel,
        )
        panel = ManageRunsPanel.__new__(ManageRunsPanel)
        panel._on_clean_intermediates = on_clean
        panel._selected_summary = lambda: summary
        panel._reloaded = []
        panel._reload_from_callback = lambda: panel._reloaded.append(True)
        panel._show_error = lambda t, b: panel.__dict__.setdefault(
            "_errors", []
        ).append((t, b))
        return panel

    def test_invokes_callback_with_selected_slug_and_reloads(self):
        calls = []
        panel = self._panel(
            summary={"run_id": "cells_v1"},
            on_clean=lambda slug: calls.append(slug) or True,
        )
        panel._on_clean_intermediates_clicked()
        assert calls == ["cells_v1"]
        assert panel._reloaded == [True]

    def test_no_reload_when_callback_returns_false(self):
        panel = self._panel(
            summary={"run_id": "cells_v1"},
            on_clean=lambda slug: False,
        )
        panel._on_clean_intermediates_clicked()
        assert panel._reloaded == []

    def test_noop_without_selection(self):
        calls = []
        panel = self._panel(summary=None, on_clean=lambda s: calls.append(s))
        panel._on_clean_intermediates_clicked()
        assert calls == []

    def test_noop_without_callback(self):
        panel = self._panel(summary={"run_id": "cells_v1"}, on_clean=None)
        # Must not raise.
        panel._on_clean_intermediates_clicked()
        assert panel._reloaded == []
