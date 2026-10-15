"""Tests for Phase 8e provenance badge rendering + tooltip.

Pure-Python tests for the format/tooltip helpers + sample-record build.
"""

from __future__ import annotations

import pytest


# ----------------------------------------------------------------------------
# Lightweight controller stub (avoids Qt + persistence imports)
# ----------------------------------------------------------------------------

class FakeController:
    """Minimal stub exposing the format + tooltip methods under test."""

    _STATUS_DOTS = {"predicted": "○", "accepted": "●", "staged": "●●", "committed": "■"}

    # Pull the real methods off the controller for testing
    def __init__(self):
        from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
            TrainingAnnotationController,
        )
        self._format_queue_row = TrainingAnnotationController._format_queue_row.__get__(self)
        self._build_provenance_tooltip = TrainingAnnotationController._build_provenance_tooltip.__get__(self)


@pytest.fixture
def ctrl():
    return FakeController()


# ----------------------------------------------------------------------------
# _format_queue_row glyph prefix
# ----------------------------------------------------------------------------

class TestFormatQueueRowGlyph:

    def test_native_record_has_blank_prefix_for_alignment(self, ctrl):
        record = {
            "view": "XY",
            "slice_index": 42,
            "patch_id": "patch_3",
            "status": "accepted",
        }
        row = ctrl._format_queue_row(record)
        # No provenance → 2-space padding so columns align with imported rows
        assert row.startswith("  XY ") or row.startswith("  XY")
        # No arrow
        assert "↳" not in row

    def test_inherited_from_pair_gets_arrow_prefix(self, ctrl):
        record = {
            "view": "XY",
            "slice_index": 42,
            "patch_id": "patch_3",
            "status": "accepted",
            "inherited_from_pair": {
                "source_run_id": "default",
                "source_bucket": "lateral",
                "source_round_id": "round_03",
                "source_label": "patch_3_XY_z0042",
            },
        }
        row = ctrl._format_queue_row(record)
        assert row.startswith("↳ ")
        # Still has the rest of the row
        assert "XY" in row
        assert "patch_3" in row

    def test_inherited_from_external_gets_arrow_prefix(self, ctrl):
        """Phase 8f forward-compat — external provenance also gets the glyph."""
        record = {
            "view": "XZ",
            "slice_index": 7,
            "patch_id": "patch_x",
            "status": "accepted",
            "inherited_from_external": {
                "source_path": "/some/folder",
                "source_filename": "ext_001_img.tif",
            },
        }
        row = ctrl._format_queue_row(record)
        assert row.startswith("↳ ")

    def test_arrow_does_not_break_alignment(self, ctrl):
        """↳ + space should consume the same display width as 2 spaces.

        Trivial assertion: both prefixes are 2 chars long so the rest of
        the row stays aligned in monospaced lists.
        """
        native = ctrl._format_queue_row({
            "view": "XY", "slice_index": 1, "patch_id": "p", "status": "predicted",
        })
        imported = ctrl._format_queue_row({
            "view": "XY", "slice_index": 1, "patch_id": "p", "status": "predicted",
            "inherited_from_pair": {"source_run_id": "x"},
        })
        # Both should have prefix length 2 (either "  " or "↳ ")
        assert len(native) == len(imported), (
            f"alignment broken: native={native!r} imported={imported!r}"
        )


# ----------------------------------------------------------------------------
# _build_provenance_tooltip
# ----------------------------------------------------------------------------

class TestBuildProvenanceTooltip:

    def test_no_provenance_returns_empty(self, ctrl):
        record = {"view": "XY", "status": "predicted"}
        assert ctrl._build_provenance_tooltip(record) == ""

    def test_inherited_from_pair_full_tooltip(self, ctrl):
        record = {
            "inherited_from_pair": {
                "source_run_id": "default",
                "source_bucket": "lateral",
                "source_round_id": "round_03",
                "source_label": "patch_3_XY_z0042",
            },
            "imported_at": "2026-05-24T03:15:42.123456+00:00",
        }
        tooltip = ctrl._build_provenance_tooltip(record)
        assert "↳ Imported from another run" in tooltip
        assert "default" in tooltip
        assert "lateral" in tooltip
        assert "round_03" in tooltip
        assert "patch_3_XY_z0042" in tooltip
        assert "2026-05-24T03:15:42" in tooltip  # truncated, no ms

    def test_inherited_from_pair_missing_imported_at(self, ctrl):
        record = {
            "inherited_from_pair": {
                "source_run_id": "x", "source_bucket": "axial",
                "source_round_id": "round_00", "source_label": "y",
            },
        }
        tooltip = ctrl._build_provenance_tooltip(record)
        # No "when:" line if imported_at missing
        assert "when:" not in tooltip
        assert "↳ Imported from another run" in tooltip

    def test_inherited_from_pair_partial_fields_defaults_to_question_mark(self, ctrl):
        record = {"inherited_from_pair": {"source_run_id": "x"}}
        tooltip = ctrl._build_provenance_tooltip(record)
        assert "x" in tooltip
        assert "?" in tooltip  # missing fields shown as ?

    def test_inherited_from_external_tooltip(self, ctrl):
        record = {
            "inherited_from_external": {
                "source_path": "/some/folder",
                "source_filename": "ext_001_img.tif",
            },
            "imported_at": "2026-05-24T03:15:42Z",
        }
        tooltip = ctrl._build_provenance_tooltip(record)
        assert "↳ Imported from external file" in tooltip
        assert "ext_001_img.tif" in tooltip
        assert "/some/folder" in tooltip
        assert "2026-05-24T03:15:42" in tooltip

    def test_non_dict_provenance_returns_empty(self, ctrl):
        """Defensive: corrupted record with non-dict provenance shouldn't crash."""
        record = {"inherited_from_pair": "not a dict"}
        assert ctrl._build_provenance_tooltip(record) == ""

    def test_inherited_from_pair_takes_priority_over_external(self, ctrl):
        """If somehow both fields are set, inherited_from_pair wins (it's
        the cross-run case which is more informative than external)."""
        record = {
            "inherited_from_pair": {"source_run_id": "x"},
            "inherited_from_external": {"source_filename": "y"},
        }
        tooltip = ctrl._build_provenance_tooltip(record)
        assert "another run" in tooltip
        assert "external" not in tooltip


# ----------------------------------------------------------------------------
# Combined: full render path
# ----------------------------------------------------------------------------

class TestCombined:

    def test_imported_pair_row_renders_correctly(self, ctrl):
        record = {
            "view": "XY",
            "slice_index": 100,
            "patch_id": "my_patch",
            "status": "accepted",
            "round_tag": "round_05",
            "inherited_from_pair": {
                "source_run_id": "default",
                "source_bucket": "lateral",
                "source_round_id": "round_03",
                "source_label": "my_patch_XY_z0100",
            },
        }
        row = ctrl._format_queue_row(record)
        tooltip = ctrl._build_provenance_tooltip(record)
        # Row has glyph
        assert row.startswith("↳ ")
        assert "XY" in row
        assert "100" in row
        assert "my_patch" in row
        # Tooltip has full provenance
        assert "default" in tooltip
        assert "round_03" in tooltip
