"""Tests for the 3D annotator help-button wiring (2026-06-01,
feature/annotator-help-system, 3D phase).

Verifies:
- `Curation3DAnnotator._build_draw_group` calls help_button() and
  attaches a `?` to the top-of-panel patch row AND the Refinement
  section header row.
- The two guide files exist + are non-trivial.
- Module-level `_ANNOTATOR_GUIDES_DIR` constant is defined and resolves
  to repo `docs/guides`.

All tests are Qt-free — source-inspect + filesystem checks.  No napari/
Qt setup needed.
"""
from __future__ import annotations

import inspect
from pathlib import Path

from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator


REPO_ROOT = Path(__file__).resolve().parents[2]
GUIDES_DIR = REPO_ROOT / "docs" / "guides"


def test_3d_annotator_wires_help_buttons():
    """`Curation3DAnnotator._build_draw_group` must call help_button()
    in two places: the patch row (top-of-panel) and the Refinement
    section header.  Refinement Navigation deliberately is NOT wired —
    `_focused_mode_group` is forced to zero size in 3D (line 801)
    because triage lives in the Label Navigator dock instead.
    """
    src = inspect.getsource(Curation3DAnnotator._build_draw_group)
    # Two help_button calls — top section + Refinement header.
    assert src.count("_help_button(") >= 2, (
        "3D annotator must wire at least two help buttons: one in the "
        "top patch row, one inline with the Refinement header.  "
        f"Found {src.count('_help_button(')} calls."
    )
    # The two guides referenced by name.
    assert "annotator_3d_main.md" in src, (
        "Top patch row must reference annotator_3d_main.md"
    )
    assert "annotator_3d_refinement.md" in src, (
        "Refinement section must reference annotator_3d_refinement.md"
    )
    # Negative: don't wire to the hidden focused_mode_group (same reason
    # as 2D — _focused_mode_group has setMaximumSize(0,0) in 3D).
    assert "_focused_toggle_row_layout.addWidget" not in src, (
        "_focused_toggle_row_layout.addWidget must NOT be wired in 3D — "
        "the group is forced to zero-size at line ~801"
    )


def test_3d_annotator_imports_help_button():
    """The help_button helper must be imported from ui_shell.theme."""
    import napari_epicycle.annotator.curation_3d_widget as mod
    src = inspect.getsource(mod)
    assert "help_button" in src, (
        "Module must import help_button from ui_shell.theme"
    )


def test_3d_annotator_references_annotator_guides_dir():
    """Module-level `_ANNOTATOR_GUIDES_DIR` must resolve to "
    `<repo>/docs/guides`."""
    from napari_epicycle.annotator import curation_3d_widget as mod
    assert hasattr(mod, "_ANNOTATOR_GUIDES_DIR"), (
        "Module must define _ANNOTATOR_GUIDES_DIR for guide path "
        "construction"
    )
    p = mod._ANNOTATOR_GUIDES_DIR
    assert p.name == "guides" and p.parent.name == "docs", (
        f"_ANNOTATOR_GUIDES_DIR should resolve to docs/guides, got {p}"
    )


def test_3d_guides_exist_on_disk():
    """Both 3D guide markdown files must exist."""
    main_guide = GUIDES_DIR / "annotator_3d_main.md"
    refinement_guide = GUIDES_DIR / "annotator_3d_refinement.md"
    assert main_guide.exists(), (
        f"Missing guide: {main_guide} — referenced by the top-of-panel "
        "? button"
    )
    assert refinement_guide.exists(), (
        f"Missing guide: {refinement_guide} — referenced by the "
        "Refinement header ? button"
    )


# --------------------------------------------------------------------------- #
# The curation tools panel: one `?` per visible tab
# --------------------------------------------------------------------------- #

def _panel_guide_names():
    """The guide filenames the Advanced Features panel points its `?`s at."""
    import re

    from napari_epicycle.annotator import advanced_features_widget as M

    src = inspect.getsource(M)
    return sorted(set(re.findall(r'_GUIDES_DIR / "([^"]+)"', src)))


def test_the_panel_wires_one_help_button_per_visible_tab():
    """It had exactly ONE `?` (the Split tab's) while
    `docs/guides/step3_curation_curate.md` already promised the dock
    "separate guides for each tool"."""
    from napari_epicycle.annotator import advanced_features_widget as M

    for builder in (
        "_build_assisted_split_section",
        "_build_add_denovo_section",
        "_build_refine_section",
    ):
        src = inspect.getsource(getattr(M.AdvancedFeaturesWidget, builder))
        assert "_help_button(" in src, (
            "{} has no ? button; its tool is undocumented in the UI".format(
                builder)
        )


def test_the_three_panel_guides_exist_and_are_non_trivial():
    names = _panel_guide_names()
    assert len(names) == 3, (
        "one guide per visible tab (Reshape / Add / Refine); found {}".format(
            names)
    )
    for name in names:
        path = GUIDES_DIR / name
        assert path.exists(), "{} is referenced by a ? button".format(name)
        content = path.read_text(encoding="utf-8")
        assert len(content) > 500, "{} is suspiciously short".format(name)
        assert content.startswith("#"), (
            "{} must start with a top-level heading".format(name)
        )


def test_the_add_and_refine_guides_are_named_for_their_tabs():
    names = _panel_guide_names()
    assert "annotator_3d_add.md" in names
    assert "annotator_3d_refine.md" in names


def test_3d_guides_are_non_trivial():
    """Guard against shipping empty markdown files."""
    for guide in ("annotator_3d_main.md", "annotator_3d_refinement.md"):
        path = GUIDES_DIR / guide
        content = path.read_text(encoding="utf-8")
        assert len(content) > 500, (
            f"{guide} is suspiciously short ({len(content)} chars)"
        )
        assert content.startswith("#"), (
            f"{guide} must start with a top-level heading"
        )
