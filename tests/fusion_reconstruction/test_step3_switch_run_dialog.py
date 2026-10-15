"""Tests for ``fusion_reconstruction._switch_run_dialog`` pure helper (Phase 4a).

The two Qt-constructing functions (``show_switch_blocked`` /
``show_switch_confirm``) are manual-tested.  The pure label formatter
``_format_run_label`` is covered here without a display.
"""

from __future__ import annotations

import pytest

try:
    import qtpy  # noqa: F401
    qt_available = True
except ImportError:
    qt_available = False

pytestmark = pytest.mark.skipif(not qt_available, reason="qtpy not available")


def _fn():
    from napari_epicycle.fusion_reconstruction._switch_run_dialog import (
        _format_run_label,
    )
    return _format_run_label


class TestFormatRunLabel:

    def test_display_and_handle_differ(self):
        assert _fn()("cells_v1", "Cells v1") == "Cells v1 (handle: cells_v1)"

    def test_no_display_name_falls_back_to_handle(self):
        assert _fn()("cells_v1", None) == "cells_v1"

    def test_empty_display_name_falls_back(self):
        assert _fn()("cells_v1", "") == "cells_v1"
        assert _fn()("cells_v1", "   ") == "cells_v1"

    def test_display_matches_handle_returns_bare_handle(self):
        assert _fn()("cells_v1", "cells_v1") == "cells_v1"

    def test_whitespace_trimmed(self):
        assert _fn()("  cells_v1  ", "  Cells v1  ") == "Cells v1 (handle: cells_v1)"

    def test_default_run(self):
        assert _fn()("default", "Default") == "Default (handle: default)"
