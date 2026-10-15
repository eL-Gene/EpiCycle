"""Tests for ``fusion_reconstruction._new_run_dialog`` pure helpers (Phase 4a).

The Qt dialog itself is not exercised (Qt teardown crashes the process,
as the repo test notes say).  The pure helpers — slug validation,
collision suggestion, display-name derivation, and the per-mode init-spec
builder — are importable WITHOUT constructing the QDialog and are covered
here.
"""

from __future__ import annotations

import pytest

# Importing the module needs qtpy (module-level Qt imports), but the
# helpers under test are pure.  Skip the file if qtpy is unavailable.
try:
    import qtpy  # noqa: F401
    qt_available = True
except ImportError:
    qt_available = False

pytestmark = pytest.mark.skipif(not qt_available, reason="qtpy not available")


# ----------------------------------------------------------------------------
# validate_slug_input
# ----------------------------------------------------------------------------

class TestValidateSlugInput:

    def _fn(self):
        from napari_epicycle.fusion_reconstruction._new_run_dialog import (
            validate_slug_input,
        )
        return validate_slug_input

    def test_valid_slug_available(self):
        ok, msg = self._fn()("cells_v1", set())
        assert ok is True
        assert msg == "Available"

    def test_empty_slug_required(self):
        ok, msg = self._fn()("", set())
        assert ok is False
        assert "Required" in msg

    def test_reserved_default_rejected(self):
        ok, msg = self._fn()("default", set())
        assert ok is False
        assert "reserved" in msg.lower()

    def test_reserved_default_case_folded(self):
        for variant in ("Default", "DEFAULT", "DeFaUlT"):
            ok, _ = self._fn()(variant, set())
            assert ok is False, variant

    def test_collision_case_folded_duplicate(self):
        ok, msg = self._fn()("Cells_V1", {"cells_v1"})
        assert ok is False
        assert "already exists" in msg

    def test_collision_same_case(self):
        ok, msg = self._fn()("nuclei", {"nuclei", "membrane"})
        assert ok is False
        assert "already exists" in msg

    def test_invalid_chars_rejected(self):
        ok, _ = self._fn()("has spaces", set())
        assert ok is False
        ok, _ = self._fn()("cells/v1", set())
        assert ok is False
        ok, _ = self._fn()("_leading", set())
        assert ok is False

    def test_too_long_rejected(self):
        from napari_epicycle.fusion_reconstruction import _paths
        ok, _ = self._fn()("a" * (_paths.RUN_ID_MAX_LENGTH + 1), set())
        assert ok is False


# ----------------------------------------------------------------------------
# suggest_alternate_slug + derive_display_name_from_slug
# ----------------------------------------------------------------------------

class TestSuggestAlternateSlug:

    def _fn(self):
        from napari_epicycle.fusion_reconstruction._new_run_dialog import (
            suggest_alternate_slug,
        )
        return suggest_alternate_slug

    def test_no_collision_returns_input(self):
        assert self._fn()("cells", {"default"}) == "cells"

    def test_increments_trailing_digit(self):
        assert self._fn()("cells_v1", {"cells_v1"}) == "cells_v2"

    def test_appends_underscore_2(self):
        assert self._fn()("cells", {"cells"}) == "cells_2"

    def test_skips_existing(self):
        assert self._fn()("cells", {"cells", "cells_2"}) == "cells_3"


class TestDeriveDisplayName:

    def _fn(self):
        from napari_epicycle.fusion_reconstruction._new_run_dialog import (
            derive_display_name_from_slug,
        )
        return derive_display_name_from_slug

    def test_underscore(self):
        assert self._fn()("cells_v1") == "Cells V1"

    def test_dash(self):
        assert self._fn()("membrane-test") == "Membrane Test"

    def test_empty(self):
        assert self._fn()("") == ""


# ----------------------------------------------------------------------------
# build_init_spec_for_mode — matches step3_run_manifest builders
# ----------------------------------------------------------------------------

class TestBuildInitSpecForMode:

    def _fn(self):
        from napari_epicycle.fusion_reconstruction._new_run_dialog import (
            build_init_spec_for_mode,
        )
        return build_init_spec_for_mode

    def test_pretrained_matches_builder(self):
        from packages.epicycle_io.step3_run_manifest import init_spec_pretrained
        got = self._fn()("pretrained", pretrained_name="cyto3")
        assert got == init_spec_pretrained(pretrained_name="cyto3")
        assert got["mode"] == "pretrained"

    def test_step2_run_matches_builder(self):
        from packages.epicycle_io.step3_run_manifest import init_spec_step2_run
        got = self._fn()("step2_run", step2_run_id="nuclei_run")
        assert got == init_spec_step2_run(source_step2_run_id="nuclei_run")
        assert got["mode"] == "step2_run"

    def test_step3_refined_matches_builder(self):
        from packages.epicycle_io.step3_run_manifest import init_spec_step3_refined
        got = self._fn()("step3_refined", step3_bundle="bundle_a")
        assert got == init_spec_step3_refined(source_step3_bundle="bundle_a")
        assert got["mode"] == "step3_refined"
        # Name-only (no run_id passed) stays backward-compatible.
        assert "source_step3_run_id" not in got

    def test_step3_refined_carries_run_id(self):
        from packages.epicycle_io.step3_run_manifest import init_spec_step3_refined
        got = self._fn()(
            "step3_refined", step3_bundle="bundle_a", step3_run_id="runA"
        )
        assert got == init_spec_step3_refined(
            source_step3_bundle="bundle_a", source_step3_run_id="runA"
        )
        assert got["source_step3_run_id"] == "runA"

    def test_step3_refined_blank_run_id_omits_key(self):
        for rid in (None, ""):
            got = self._fn()(
                "step3_refined", step3_bundle="bundle_a", step3_run_id=rid
            )
            assert "source_step3_run_id" not in got

    def test_browse_matches_builder(self):
        from packages.epicycle_io.step3_run_manifest import init_spec_browse
        got = self._fn()("browse", browse_path="/tmp/weights.pth")
        assert got == init_spec_browse(external_path="/tmp/weights.pth")
        assert got["mode"] == "browse"

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            self._fn()("scratch")

    def test_missing_required_arg_raises(self):
        # builders reject empty required fields
        with pytest.raises(ValueError):
            self._fn()("pretrained", pretrained_name="")
        with pytest.raises(ValueError):
            self._fn()("step2_run", step2_run_id="")


# ----------------------------------------------------------------------------
# _group_step3_sources / _format_run_label — static helpers (no QDialog)
# ----------------------------------------------------------------------------

class TestGroupStep3Sources:

    def _cls(self):
        from napari_epicycle.fusion_reconstruction._new_run_dialog import (
            NewRunDialog,
        )
        return NewRunDialog

    def test_collapses_buckets_to_one_row_per_run_bundle(self):
        cls = self._cls()
        sources = [
            {"bundle_name": "bA", "run_id": None, "bucket": "lateral",
             "mtime_iso": "2026-06-01T00:00:00Z"},
            {"bundle_name": "bA", "run_id": None, "bucket": "axial",
             "mtime_iso": "2026-06-02T00:00:00Z"},
            {"bundle_name": "bA", "run_id": "cells", "bucket": "lateral",
             "mtime_iso": "2026-06-03T00:00:00Z"},
            {"bundle_name": "bA", "run_id": "cells", "bucket": "axial",
             "mtime_iso": "2026-06-03T00:00:00Z"},
        ]
        rows = cls._group_step3_sources(sources)
        # Two groups: (None, bA) and (cells, bA) — NOT four bucket rows.
        keys = {(r["run_id"], r["bundle_name"]) for r in rows}
        assert keys == {(None, "bA"), ("cells", "bA")}
        # Latest mtime kept per group.
        by_run = {r["run_id"]: r for r in rows}
        assert by_run[None]["mtime_iso"] == "2026-06-02T00:00:00Z"
        assert by_run["cells"]["mtime_iso"] == "2026-06-03T00:00:00Z"

    def test_preserves_first_seen_order(self):
        cls = self._cls()
        sources = [
            {"bundle_name": "zeta", "run_id": None, "bucket": "lateral"},
            {"bundle_name": "alpha", "run_id": None, "bucket": "lateral"},
        ]
        rows = cls._group_step3_sources(sources)
        assert [r["bundle_name"] for r in rows] == ["zeta", "alpha"]

    def test_ignores_non_dict_entries(self):
        cls = self._cls()
        rows = cls._group_step3_sources([None, "x", {"bundle_name": "b"}])
        assert [r["bundle_name"] for r in rows] == ["b"]

    def test_empty(self):
        cls = self._cls()
        assert cls._group_step3_sources([]) == []
        assert cls._group_step3_sources(None) == []

    def test_format_run_label(self):
        cls = self._cls()
        assert cls._format_run_label(None) == "(default)"
        assert cls._format_run_label("") == "(default)"
        assert cls._format_run_label("cells") == "cells"
