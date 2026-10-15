"""Tests for the 2D annotator help-button wiring (2026-06-01,
feature/annotator-help-system).

Verifies:
- Base `NativeMultiviewAnnotator` exposes `_crosshair_row_layout` and
  `_focused_toggle_row_layout` attributes so subclasses can append
  per-section help buttons.
- `Curation2DAnnotator._build_draw_group` appends a `help_button` into
  each of those two rows.
- The guide files exist on disk (would-be-shipped paths).

All tests are Qt-free — source-inspect via `inspect.getsource` + on-disk
existence checks.  No napari/Qt setup required.
"""
from __future__ import annotations

import inspect
from pathlib import Path

from napari_epicycle.annotator.curation_2d_widget import Curation2DAnnotator
from napari_epicycle.annotator.native_multiview_widget import (
    NativeMultiviewAnnotator,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
GUIDES_DIR = REPO_ROOT / "docs" / "guides"


def test_base_exposes_crosshair_row_layout_attr():
    """Base `_build_draw_group` must wrap `crosshairs_check` in a
    QHBoxLayout exposed as `_crosshair_row_layout` so subclasses can
    append controls (e.g. help button) inline."""
    src = inspect.getsource(NativeMultiviewAnnotator._build_draw_group)
    assert "_crosshair_row_layout" in src, (
        "Base must expose self._crosshair_row_layout (QHBoxLayout) so "
        "subclasses can append per-section help buttons inline with the "
        "crosshair checkbox"
    )
    assert "addStretch" in src, (
        "Crosshair row must have a stretch so the help button right-"
        "aligns instead of sitting next to the checkbox"
    )


def test_base_exposes_focused_toggle_row_layout_attr():
    """Same pattern for the Refinement Navigation group's toggle row.

    Even though Curation2DAnnotator deliberately does NOT wire a help
    button here (the group is hidden in 2D HITL), the base must still
    expose the layout as forward-compat scaffolding for the future Step
    3 / 3D annotator help wiring.
    """
    src = inspect.getsource(NativeMultiviewAnnotator._build_draw_group)
    assert "_focused_toggle_row_layout" in src, (
        "Base must expose self._focused_toggle_row_layout (QHBoxLayout) "
        "so subclasses can append a help button inline with the "
        "focused-mode toggle"
    )


def test_2d_annotator_wires_quick_annotate_help_button():
    """`Curation2DAnnotator._build_draw_group` must call help_button()
    and append the result into the crosshair row (Quick Annotate group).

    Refinement Navigation help is intentionally NOT wired in 2D — the
    panel is gated on _3d_curation_mode=True which the 2D HITL flow
    never sets.  See in-source comment for rationale (audit finding
    2026-06-01).
    """
    src = inspect.getsource(Curation2DAnnotator._build_draw_group)
    assert "_help_button" in src, (
        "Curation2DAnnotator must import and call help_button() from "
        "ui_shell.theme to surface per-section help"
    )
    assert "_crosshair_row_layout.addWidget" in src, (
        "Help button for Quick Annotate must be appended to the base's "
        "_crosshair_row_layout"
    )
    # Negative assertion — refinement-nav wiring is deliberately absent.
    assert "_focused_toggle_row_layout.addWidget" not in src, (
        "_focused_toggle_row_layout.addWidget(...) must NOT be wired in "
        "Curation2DAnnotator — the group is hidden in 2D HITL.  Forward-"
        "compat scaffolding only."
    )


def test_2d_quick_annotate_guide_exists_on_disk():
    """The Quick Annotate guide must exist at the path the code
    references — otherwise clicking the ? button shows a 'file not
    found' dialog."""
    quick_annotate_guide = GUIDES_DIR / "annotator_2d_quick_annotate.md"
    assert quick_annotate_guide.exists(), (
        f"Missing guide: {quick_annotate_guide} — the help button in "
        "Quick Annotate references this file"
    )


def test_2d_quick_annotate_guide_is_non_trivial():
    """Guard against a future PR shipping an empty markdown file."""
    path = GUIDES_DIR / "annotator_2d_quick_annotate.md"
    content = path.read_text(encoding="utf-8")
    assert len(content) > 500, (
        f"annotator_2d_quick_annotate.md is suspiciously short "
        f"({len(content)} chars) — biologist-accessible guides should "
        f"be substantial"
    )
    assert content.startswith("#"), (
        "annotator_2d_quick_annotate.md must start with a top-level heading"
    )


def test_2d_annotator_references_annotator_guides_dir():
    """Module-level `_ANNOTATOR_GUIDES_DIR` must resolve to "
    `<repo>/docs/guides`."""
    from napari_epicycle.annotator import curation_2d_widget as mod
    assert hasattr(mod, "_ANNOTATOR_GUIDES_DIR"), (
        "Module must define _ANNOTATOR_GUIDES_DIR for guide path "
        "construction"
    )
    p = mod._ANNOTATOR_GUIDES_DIR
    assert p.name == "guides" and p.parent.name == "docs", (
        f"_ANNOTATOR_GUIDES_DIR should resolve to docs/guides, got {p}"
    )
