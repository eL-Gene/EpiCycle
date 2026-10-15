"""Tests for ``_new_run_dialog`` pure-helper functions (Phase 8a).

The Qt dialog itself is not exercised here (Qt teardown crashes the
process, as the repo test notes say).  Pure helpers — slug
validation, collision suggestion, display-name derivation — are
covered.
"""

from __future__ import annotations

import pytest

# Import the pure helpers directly (skip the Qt module-level import that
# would pull in qtpy).  The dialog class itself isn't tested here.
import importlib.util
import sys
from pathlib import Path

# Workaround: importing _new_run_dialog requires qtpy.  Test in
# the Qt-aware annotator env; otherwise skip the file.
qt_available = True
try:
    import qtpy  # noqa: F401
except ImportError:
    qt_available = False


@pytest.mark.skipif(not qt_available, reason="qtpy not available")
class TestValidateSlugInput:

    def test_valid_slug_available(self):
        from napari_epicycle.hitl_training_annotation._new_run_dialog import (
            validate_slug_input,
        )
        ok, msg = validate_slug_input("nuclei_v1", set())
        assert ok is True
        assert msg == "Available"

    def test_empty_slug_required(self):
        from napari_epicycle.hitl_training_annotation._new_run_dialog import (
            validate_slug_input,
        )
        ok, msg = validate_slug_input("", set())
        assert ok is False
        assert "Required" in msg

    def test_slug_with_spaces_rejected(self):
        from napari_epicycle.hitl_training_annotation._new_run_dialog import (
            validate_slug_input,
        )
        ok, msg = validate_slug_input("has spaces", set())
        assert ok is False

    def test_slug_with_special_chars_rejected(self):
        from napari_epicycle.hitl_training_annotation._new_run_dialog import (
            validate_slug_input,
        )
        ok, msg = validate_slug_input("nuclei/v1", set())
        assert ok is False

    def test_slug_starting_with_underscore_rejected(self):
        from napari_epicycle.hitl_training_annotation._new_run_dialog import (
            validate_slug_input,
        )
        ok, msg = validate_slug_input("_nuclei", set())
        assert ok is False

    def test_collision_returns_message(self):
        from napari_epicycle.hitl_training_annotation._new_run_dialog import (
            validate_slug_input,
        )
        ok, msg = validate_slug_input("default", {"default", "nuclei"})
        assert ok is False
        assert "already exists" in msg

    def test_collision_case_insensitive(self):
        """Windows / macOS filesystems are case-insensitive — collision
        check must be too, or create_run_on_disk's mkdir will fail
        after the dialog promised the slug was available.
        Audit-found 2026-05-23."""
        from napari_epicycle.hitl_training_annotation._new_run_dialog import (
            validate_slug_input,
        )
        ok, msg = validate_slug_input("Default", {"default"})
        assert ok is False
        assert "already exists" in msg

        ok, msg = validate_slug_input("DEFAULT", {"default"})
        assert ok is False

        ok, msg = validate_slug_input("default", {"Default"})
        assert ok is False


@pytest.mark.skipif(not qt_available, reason="qtpy not available")
class TestSuggestAlternateSlug:

    def test_no_collision_returns_input(self):
        from napari_epicycle.hitl_training_annotation._new_run_dialog import (
            suggest_alternate_slug,
        )
        assert suggest_alternate_slug("nuclei", {"default"}) == "nuclei"

    def test_collision_increments_trailing_digit(self):
        from napari_epicycle.hitl_training_annotation._new_run_dialog import (
            suggest_alternate_slug,
        )
        assert suggest_alternate_slug("nuclei_v1", {"nuclei_v1"}) == "nuclei_v2"

    def test_collision_skips_multiple_existing(self):
        from napari_epicycle.hitl_training_annotation._new_run_dialog import (
            suggest_alternate_slug,
        )
        existing = {"nuclei_v1", "nuclei_v2", "nuclei_v3"}
        assert suggest_alternate_slug("nuclei_v1", existing) == "nuclei_v4"

    def test_no_trailing_digit_appends_underscore_2(self):
        from napari_epicycle.hitl_training_annotation._new_run_dialog import (
            suggest_alternate_slug,
        )
        assert suggest_alternate_slug("nuclei", {"nuclei"}) == "nuclei_2"

    def test_no_trailing_digit_skips_existing_underscore_n(self):
        from napari_epicycle.hitl_training_annotation._new_run_dialog import (
            suggest_alternate_slug,
        )
        assert suggest_alternate_slug("nuclei", {"nuclei", "nuclei_2"}) == "nuclei_3"

    def test_multi_digit_trailing(self):
        from napari_epicycle.hitl_training_annotation._new_run_dialog import (
            suggest_alternate_slug,
        )
        assert suggest_alternate_slug("run_99", {"run_99"}) == "run_100"


@pytest.mark.skipif(not qt_available, reason="qtpy not available")
class TestDeriveDisplayNameFromSlug:

    def test_underscore_separated(self):
        from napari_epicycle.hitl_training_annotation._new_run_dialog import (
            derive_display_name_from_slug,
        )
        assert derive_display_name_from_slug("nuclei_v1") == "Nuclei V1"

    def test_dash_separated(self):
        from napari_epicycle.hitl_training_annotation._new_run_dialog import (
            derive_display_name_from_slug,
        )
        assert derive_display_name_from_slug("membrane-test") == "Membrane Test"

    def test_mixed_separators(self):
        from napari_epicycle.hitl_training_annotation._new_run_dialog import (
            derive_display_name_from_slug,
        )
        assert derive_display_name_from_slug("foo_bar-baz") == "Foo Bar Baz"

    def test_single_word(self):
        from napari_epicycle.hitl_training_annotation._new_run_dialog import (
            derive_display_name_from_slug,
        )
        assert derive_display_name_from_slug("default") == "Default"

    def test_empty_returns_empty(self):
        from napari_epicycle.hitl_training_annotation._new_run_dialog import (
            derive_display_name_from_slug,
        )
        assert derive_display_name_from_slug("") == ""
