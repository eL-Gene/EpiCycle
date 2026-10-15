"""Smoke tests for ``_migrate_step2_dialog.py``.

The actual Qt dialog rendering requires a display and is not covered
here.  These tests exercise:

* Early-return paths in :func:`prompt_and_migrate_if_needed`
  (new layout / empty / sentinel) — these return BEFORE the dialog
  is constructed, so no event loop is needed.
* Pure helper functions (``_format_plan_summary``, ``_short_path``).
* Public API surface (``__all__`` correctness, enum values).

The full migration logic is covered in
``test_migrate_step2_layout.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Skip the entire module if qtpy isn't importable (headless CI etc.)
qtpy = pytest.importorskip("qtpy")

from napari_epicycle.hitl_training_annotation import (
    _migrate_step2_dialog as dialog,
    _migrate_step2_layout as layout,
)
from napari_epicycle.hitl_training_annotation._migrate_step2_layout import (
    FileMove,
    MetadataSplit,
    MigrationPlan,
    RunManifestInit,
    StateSchemaWrap,
    LegacyDelete,
)


# ============================================================================
# Early-return paths (no Qt event loop needed)
# ============================================================================

class TestEarlyReturns:
    def test_new_layout_skips_dialog(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "artifacts" / "step2" / "runs" / "default").mkdir(parents=True)
        choice, result = dialog.prompt_and_migrate_if_needed(None, ws)
        assert choice == dialog.UserChoice.SKIP_SESSION
        assert result is None

    def test_empty_workspace_skips_dialog(self, tmp_path):
        ws = tmp_path / "empty"
        ws.mkdir()
        choice, result = dialog.prompt_and_migrate_if_needed(None, ws)
        assert choice == dialog.UserChoice.SKIP_SESSION
        assert result is None

    def test_sentinel_skips_dialog_before_planning(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / "training_data" / "round_00").mkdir(parents=True)
        layout.mark_migration_skipped(ws)

        choice, result = dialog.prompt_and_migrate_if_needed(None, ws)
        assert choice == dialog.UserChoice.SKIP_FOREVER
        assert result is None

    def test_prompt_migration_empty_plan_returns_skip_session(self, tmp_path):
        ws = tmp_path / "empty"
        ws.mkdir()
        choice = dialog.prompt_migration(None, ws)
        assert choice == dialog.UserChoice.SKIP_SESSION


# ============================================================================
# Plan summary formatting
# ============================================================================

class TestFormatPlanSummary:
    def test_empty_plan(self):
        plan = MigrationPlan(workspace=Path("/tmp"))
        assert "No operations" in dialog._format_plan_summary(plan)

    def test_with_moves_only(self):
        plan = MigrationPlan(
            workspace=Path("/tmp"),
            moves=[FileMove(Path("/a"), Path("/b"), "file", "test")] * 3,
        )
        summary = dialog._format_plan_summary(plan)
        assert "3 file move(s)" in summary

    def test_with_splits(self):
        plan = MigrationPlan(
            workspace=Path("/tmp"),
            splits=[
                MetadataSplit(
                    src=Path("/a"),
                    dst_lateral=Path("/b"),
                    dst_axial=Path("/c"),
                    kind="round_metadata",
                    round_idx=0,
                    purpose="test",
                )
            ],
        )
        summary = dialog._format_plan_summary(plan)
        assert "1 metadata split(s)" in summary

    def test_with_state_wrap(self):
        plan = MigrationPlan(
            workspace=Path("/tmp"),
            state_wrap=StateSchemaWrap(src=Path("/a"), dst=Path("/b")),
        )
        summary = dialog._format_plan_summary(plan)
        assert "state-JSON wrap" in summary

    def test_with_run_manifest_init(self):
        plan = MigrationPlan(
            workspace=Path("/tmp"),
            run_manifest_init=RunManifestInit(
                dst=Path("/a"),
                run_id="default",
                manifest_data={},
            ),
        )
        summary = dialog._format_plan_summary(plan)
        assert "run-manifest init" in summary

    def test_with_legacy_deletes(self):
        plan = MigrationPlan(
            workspace=Path("/tmp"),
            legacy_deletes=[
                LegacyDelete(path=Path("/a"), purpose="x"),
                LegacyDelete(path=Path("/b"), purpose="y"),
            ],
        )
        summary = dialog._format_plan_summary(plan)
        assert "2 legacy delete(s)" in summary

    def test_combined(self):
        plan = MigrationPlan(
            workspace=Path("/tmp"),
            moves=[FileMove(Path("/a"), Path("/b"), "file", "test")],
            state_wrap=StateSchemaWrap(src=Path("/a"), dst=Path("/b")),
        )
        summary = dialog._format_plan_summary(plan)
        assert "1 file move(s)" in summary
        assert "state-JSON wrap" in summary


# ============================================================================
# Path-shortening helper
# ============================================================================

class TestShortPath:
    def test_short_path_below_threshold_unchanged(self):
        # Use a relative path so behavior is consistent across Windows/Unix
        # (absolute paths on Windows include the drive/root as a `parts` element)
        p = Path("a/b.txt")
        assert dialog._short_path(p, max_segments=6) == str(p)

    def test_short_path_at_threshold_unchanged(self):
        p = Path("a/b/c/d/e/f")  # exactly 6 parts
        assert dialog._short_path(p, max_segments=6) == str(p)

    def test_long_path_truncated(self):
        p = Path("a/b/c/d/e/f/g/h/file.txt")  # 9 parts > 3
        result = dialog._short_path(p, max_segments=3)
        # Should keep last 3 segments + leading marker
        assert "file.txt" in result
        assert "..." in result

    def test_truncated_path_keeps_tail(self):
        # Relative path with > 4 segments
        p = Path("very/long/legacy/path/to/round_00/accepted/file.tif")
        result = dialog._short_path(p, max_segments=4)
        assert "round_00" in result
        assert "file.tif" in result


# ============================================================================
# Public API
# ============================================================================

class TestPublicApi:
    def test_all_exports_present(self):
        assert hasattr(dialog, "__all__")
        for name in dialog.__all__:
            assert hasattr(dialog, name), f"__all__ lists {name!r} but module has no such attribute"

    def test_user_choice_enum_values(self):
        assert dialog.UserChoice.MIGRATE.value == "migrate"
        assert dialog.UserChoice.SKIP_SESSION.value == "skip_session"
        assert dialog.UserChoice.SKIP_FOREVER.value == "skip_forever"
        assert dialog.UserChoice.CANCEL.value == "cancel"

    def test_user_choice_is_str_enum(self):
        # UserChoice values should compare equal to str (str subclass)
        assert dialog.UserChoice.MIGRATE == "migrate"
        assert dialog.UserChoice.SKIP_SESSION == "skip_session"

    def test_prompt_helpers_in_all(self):
        assert "prompt_and_migrate_if_needed" in dialog.__all__
        assert "prompt_migration" in dialog.__all__
        assert "UserChoice" in dialog.__all__


# ============================================================================
# Theme integration sanity (does the styles helper resolve?)
# ============================================================================

class TestThemeIntegration:
    def test_dialog_styles_callable(self):
        # Should resolve without error.  The values are non-None strings.
        try:
            style, btn_cancel, btn_confirm = dialog._dialog_styles()
        except ImportError:
            pytest.skip("theme.py not importable in this environment")
        assert isinstance(style, str)
        assert isinstance(btn_cancel, str)
        assert isinstance(btn_confirm, str)
        assert len(style) > 0
