"""Routing tests for ``_migrate_step3_dialog.prompt_and_migrate_if_needed``.

The dialog itself is Qt-dependent and not unit-testable without a
display.  However, the high-level convenience function's flow
control (skip vs prompt vs execute) is testable via monkey-patching
the prompt entry point — that's what this file covers.

The dialog's visual content + click handling is verified by manual
testing post-merge.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from napari_epicycle.fusion_reconstruction import _paths as p
from napari_epicycle.fusion_reconstruction import _migrate_step3_layout as m
from napari_epicycle.fusion_reconstruction import _migrate_step3_dialog as d


def _make_minimal_legacy(tmp_path) -> Path:
    """Synthetic legacy workspace with enough state to need migration."""
    ws = tmp_path / "ws"
    ws.mkdir()
    # Legacy state JSON.
    state = {
        "schema_version": 1,
        "patches": {
            "p1": {
                "reconstruction_run_id": "uuid-1",
                "labels_path": "/old.tif",
            },
        },
    }
    (ws / "reconstruction_state.json").write_text(json.dumps(state))
    # Matching reconstruction dir.
    run_dir = ws / "artifacts" / "reconstruction" / "uuid-1"
    run_dir.mkdir(parents=True)
    (run_dir / "labels_3d.tif").write_bytes(b"x")
    return ws


class TestPromptAndMigrateRouting:
    """Verify the routing logic of prompt_and_migrate_if_needed."""

    def test_no_migration_needed_short_circuits(self, tmp_path, monkeypatch):
        """Workspace already on v2 → returns False without prompting."""
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "step3_state.json").write_text("{}")

        called = []
        monkeypatch.setattr(
            d, "prompt_migration",
            lambda parent, w: (called.append(("prompt", w)), d.UserChoice.MIGRATE)[1],
        )

        result = d.prompt_and_migrate_if_needed(None, ws)
        assert result is False
        assert called == []  # never prompted

    def test_skipped_workspace_short_circuits(self, tmp_path, monkeypatch):
        """User previously chose Don't Ask Again → no prompt."""
        ws = _make_minimal_legacy(tmp_path)
        m.mark_migration_skipped(ws)

        called = []
        monkeypatch.setattr(
            d, "prompt_migration",
            lambda parent, w: (called.append("prompt"), d.UserChoice.MIGRATE)[1],
        )

        result = d.prompt_and_migrate_if_needed(None, ws)
        assert result is False
        assert called == []

    def test_skip_session_does_not_migrate(self, tmp_path, monkeypatch):
        """User picks Skip For This Session → no migration."""
        ws = _make_minimal_legacy(tmp_path)
        monkeypatch.setattr(
            d, "prompt_migration",
            lambda parent, w: d.UserChoice.SKIP_SESSION,
        )

        result = d.prompt_and_migrate_if_needed(None, ws)
        assert result is False
        # Legacy state still present.
        assert p.legacy_state_json_path(ws).is_file()

    def test_skip_forever_writes_sentinel(self, tmp_path, monkeypatch):
        """User picks Don't Ask Again → sentinel written, no migration."""
        ws = _make_minimal_legacy(tmp_path)

        def _fake_prompt(parent, w):
            # The prompt persists the sentinel itself (see
            # _show_dialog).  Simulate that here.
            m.mark_migration_skipped(w)
            return d.UserChoice.SKIP_FOREVER

        monkeypatch.setattr(d, "prompt_migration", _fake_prompt)

        result = d.prompt_and_migrate_if_needed(None, ws)
        assert result is False
        assert m.is_migration_skipped(ws)
        assert p.legacy_state_json_path(ws).is_file()  # not migrated

    def test_migrate_executes_and_succeeds(self, tmp_path, monkeypatch):
        """User picks Migrate Now → backup + execute → True returned."""
        ws = _make_minimal_legacy(tmp_path)
        monkeypatch.setattr(
            d, "prompt_migration",
            lambda parent, w: d.UserChoice.MIGRATE,
        )
        # Suppress the success dialog so the test doesn't try to
        # open Qt.
        monkeypatch.setattr(d, "_show_success_dialog", lambda *a, **k: None)

        result = d.prompt_and_migrate_if_needed(None, ws)
        assert result is True
        # Now on v2 layout.
        assert p.is_new_layout(ws)
        assert not p.is_legacy_layout(ws)
        # Backup created.
        backups = list(ws.glob(".legacy_pre_restructure_*"))
        assert len(backups) == 1

    def test_migrate_failure_offers_rollback(self, tmp_path, monkeypatch):
        """If execute_migration fails, failure dialog fires + False returned."""
        ws = _make_minimal_legacy(tmp_path)
        monkeypatch.setattr(
            d, "prompt_migration",
            lambda parent, w: d.UserChoice.MIGRATE,
        )
        # Force execute_migration to fail.
        monkeypatch.setattr(
            m, "execute_migration",
            lambda plan, backup_dir=None: m.ExecutionResult(
                success=False,
                completed_steps=["did one thing"],
                failed_step="did one thing",
                error="boom",
                backup_dir=backup_dir,
            ),
        )
        called = []
        monkeypatch.setattr(
            d, "_show_failure_dialog",
            lambda parent, result, ws: called.append(result),
        )

        result = d.prompt_and_migrate_if_needed(None, ws)
        assert result is False
        assert len(called) == 1
        assert called[0].success is False
        assert called[0].error == "boom"


class TestPromptMigrationEmptyPlan:
    """prompt_migration with an empty plan returns SKIP_SESSION without
    opening the dialog."""

    def test_empty_workspace_returns_skip_session(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        # No legacy markers → empty plan.
        result = d.prompt_migration(None, ws)
        assert result == d.UserChoice.SKIP_SESSION


class TestUserChoiceEnum:

    def test_three_distinct_values(self):
        choices = {d.UserChoice.MIGRATE, d.UserChoice.SKIP_SESSION,
                   d.UserChoice.SKIP_FOREVER}
        assert len(choices) == 3

    def test_string_values_stable(self):
        # These string values may be persisted in logs / settings —
        # keep them stable.
        assert d.UserChoice.MIGRATE.value == "migrate"
        assert d.UserChoice.SKIP_SESSION.value == "skip_session"
        assert d.UserChoice.SKIP_FOREVER.value == "skip_forever"
