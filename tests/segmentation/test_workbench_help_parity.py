"""The Workbench's 36 help lines are the accepted wording (fixture
``fixtures/workbench_help_lines.json``), word for word."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "workbench_help_lines.json"

SPEC_36 = (
    "Plot type", "Lasso", "Density", "Removed", "Colour by", "Scale",
    "Colours", "Colours in viewer", "Mark good", "Mark bad", "Unmark", "Undo",
    "Clear", "Shape", "Skew", "Bell", "Rescale", "Auto rescale", "Defaults",
    "Gate", "Colour plot by this", "Method (Map)", "Components", "Scaling",
    "Clip extremes", "Neighbours", "Spacing", "Start from", "Distance",
    "Perplexity", "Method (Groups)", "Resolution", "Number of groups",
    "Smallest group", "Seed", "Divider",
)


def mockup_help() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["data"]


def test_the_fixture_exists():
    assert FIXTURE.is_file() and FIXTURE.stat().st_size > 0


def test_the_mockup_has_the_spec_36():
    assert tuple(mockup_help()) == SPEC_36


def test_the_help_lines_are_verbatim():
    from napari_epicycle.segmentation._workbench_text import WORKBENCH_HELP_LINES

    # Step 4's one word "unmarked".
    reworded = {"Unmark": "Return the selected cells to unmarked."}
    assert WORKBENCH_HELP_LINES == dict(mockup_help(), **reworded)
