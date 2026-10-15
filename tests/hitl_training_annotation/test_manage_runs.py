"""Tests for Phase 8c — archive / unarchive / edit_run_metadata helpers.

Qt dialog is manual-tested.  Pure data layer tested here.
"""

from __future__ import annotations

import json

import pytest

from napari_epicycle.hitl_training_annotation import _paths as _step2_paths
from napari_epicycle.hitl_training_annotation._run_management import (
    InvalidRunIdError,
    RunNotFoundError,
    archive_run,
    create_run_on_disk,
    edit_run_metadata,
    list_run_summaries,
    unarchive_run,
)
from packages.epicycle_io.step2_run_manifest import (
    RUN_MANIFEST_FILENAME,
    is_run_archived,
    read_run_manifest,
)


# ----------------------------------------------------------------------------
# archive_run
# ----------------------------------------------------------------------------

class TestArchiveRun:

    def test_marks_run_archived(self, tmp_path):
        create_run_on_disk(tmp_path, "default", display_name="Default")
        archive_run(tmp_path, "default")
        manifest = read_run_manifest(
            _step2_paths.run_root(tmp_path, "default") / RUN_MANIFEST_FILENAME
        )
        assert is_run_archived(manifest)
        # archived_at should be ISO timestamp
        assert isinstance(manifest["archived_at"], str)
        assert "T" in manifest["archived_at"]
        assert manifest["archived_at"].endswith("Z")

    def test_archived_run_excluded_from_active_summaries(self, tmp_path):
        create_run_on_disk(tmp_path, "default", display_name="Default")
        create_run_on_disk(tmp_path, "other", display_name="Other")
        archive_run(tmp_path, "other")

        summaries = list_run_summaries(tmp_path)
        active = [s for s in summaries if not s["archived"]]
        archived = [s for s in summaries if s["archived"]]
        assert len(active) == 1
        assert active[0]["run_id"] == "default"
        assert len(archived) == 1
        assert archived[0]["run_id"] == "other"

    def test_missing_run_raises(self, tmp_path):
        with pytest.raises(RunNotFoundError):
            archive_run(tmp_path, "ghost")

    def test_invalid_slug_raises(self, tmp_path):
        with pytest.raises(InvalidRunIdError):
            archive_run(tmp_path, "has spaces")

    def test_double_archive_is_safe(self, tmp_path):
        """Calling archive on an already-archived run is OK (idempotent)."""
        create_run_on_disk(tmp_path, "default", display_name="Default")
        archive_run(tmp_path, "default")
        # archived_at updates to new timestamp
        import time
        time.sleep(1.1)  # ensure timestamp diff
        archive_run(tmp_path, "default")
        manifest = read_run_manifest(
            _step2_paths.run_root(tmp_path, "default") / RUN_MANIFEST_FILENAME
        )
        assert is_run_archived(manifest)


# ----------------------------------------------------------------------------
# unarchive_run
# ----------------------------------------------------------------------------

class TestUnarchiveRun:

    def test_clears_archived_flag(self, tmp_path):
        create_run_on_disk(tmp_path, "default", display_name="Default")
        archive_run(tmp_path, "default")
        unarchive_run(tmp_path, "default")
        manifest = read_run_manifest(
            _step2_paths.run_root(tmp_path, "default") / RUN_MANIFEST_FILENAME
        )
        assert not is_run_archived(manifest)
        assert manifest["archived_at"] is None

    def test_unarchive_never_archived_is_safe(self, tmp_path):
        """Unarchive on a non-archived run is idempotent."""
        create_run_on_disk(tmp_path, "default", display_name="Default")
        unarchive_run(tmp_path, "default")  # no-op
        manifest = read_run_manifest(
            _step2_paths.run_root(tmp_path, "default") / RUN_MANIFEST_FILENAME
        )
        assert not is_run_archived(manifest)

    def test_missing_run_raises(self, tmp_path):
        with pytest.raises(RunNotFoundError):
            unarchive_run(tmp_path, "ghost")


# ----------------------------------------------------------------------------
# edit_run_metadata
# ----------------------------------------------------------------------------

class TestEditRunMetadata:

    def test_updates_display_name(self, tmp_path):
        create_run_on_disk(tmp_path, "default", display_name="Default",
                           target="general")
        manifest = edit_run_metadata(
            tmp_path, "default", display_name="New Default Name",
        )
        assert manifest["display_name"] == "New Default Name"
        assert manifest["target"] == "general"  # unchanged
        # Persisted
        on_disk = read_run_manifest(
            _step2_paths.run_root(tmp_path, "default") / RUN_MANIFEST_FILENAME
        )
        assert on_disk["display_name"] == "New Default Name"

    def test_updates_target(self, tmp_path):
        create_run_on_disk(tmp_path, "default", display_name="Default")
        manifest = edit_run_metadata(tmp_path, "default", target="nuclei")
        assert manifest["target"] == "nuclei"

    def test_clears_target_with_empty_string(self, tmp_path):
        create_run_on_disk(tmp_path, "default", display_name="Default",
                           target="general")
        manifest = edit_run_metadata(tmp_path, "default", target="")
        assert manifest["target"] == ""

    def test_updates_description(self, tmp_path):
        create_run_on_disk(tmp_path, "default", display_name="Default")
        manifest = edit_run_metadata(
            tmp_path, "default", description="Some longer notes."
        )
        assert manifest["description"] == "Some longer notes."

    def test_none_leaves_field_unchanged(self, tmp_path):
        create_run_on_disk(tmp_path, "default", display_name="Default",
                           target="general", description="orig desc")
        manifest = edit_run_metadata(tmp_path, "default", display_name="New")
        assert manifest["display_name"] == "New"
        assert manifest["target"] == "general"  # untouched
        assert manifest["description"] == "orig desc"  # untouched

    def test_empty_display_name_raises(self, tmp_path):
        create_run_on_disk(tmp_path, "default", display_name="Default")
        with pytest.raises(ValueError, match="display_name cannot be empty"):
            edit_run_metadata(tmp_path, "default", display_name="")

    def test_whitespace_display_name_raises(self, tmp_path):
        create_run_on_disk(tmp_path, "default", display_name="Default")
        with pytest.raises(ValueError):
            edit_run_metadata(tmp_path, "default", display_name="   ")

    def test_missing_run_raises(self, tmp_path):
        with pytest.raises(RunNotFoundError):
            edit_run_metadata(tmp_path, "ghost", display_name="X")

    def test_invalid_slug_raises(self, tmp_path):
        with pytest.raises(InvalidRunIdError):
            edit_run_metadata(tmp_path, "has spaces", display_name="X")

    def test_does_not_mutate_immutable_fields(self, tmp_path):
        """Edit can't touch slug / backend / created_at / created_from."""
        create_run_on_disk(tmp_path, "default", display_name="Default")
        before = read_run_manifest(
            _step2_paths.run_root(tmp_path, "default") / RUN_MANIFEST_FILENAME
        )
        edit_run_metadata(tmp_path, "default", display_name="New")
        after = read_run_manifest(
            _step2_paths.run_root(tmp_path, "default") / RUN_MANIFEST_FILENAME
        )
        # These immutable fields should be byte-identical pre + post
        for k in ("run_id", "backend", "created_at", "created_from",
                  "schema_version"):
            assert before[k] == after[k], f"field {k!r} mutated"
