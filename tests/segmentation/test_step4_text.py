"""Every agreed Step 4 sentence is a named constant.

Qt-free for ``_step4_text``; the shared block is read from
``ui_shell/theme.py``.  The constant NAMES are read from a fixture kept beside
this test (``fixtures/step4_spec_constant_names.json``: the first-column names
of the agreed sentence tables), so a sentence added there cannot be
forgotten.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from napari_epicycle.segmentation import _step4_text as T

ROOT = Path(__file__).resolve().parents[2]
NAMES_FIXTURE = (Path(__file__).parent / "fixtures"
                 / "step4_spec_constant_names.json")

#: The agreed placeholder table (``{hh:mm}`` included).
SPEC_PLACEHOLDERS = {
    "n", "k", "m", "g", "b", "u", "i", "s", "v", "stack", "name", "run",
    "model", "lateral", "axial", "names", "status", "stage", "what", "action",
    "job", "map", "t", "hh:mm", "pct", "Z", "Y", "X", "c", "z", "y", "x", "um",
    "px", "source", "VIEW", "file", "folder", "time", "unit", "units",
}
#: Placeholders the agreed sentences use outside the table: ``{msg}``
#: (PIPE_ERROR_*, log only), ``{shape}`` (MT_LOADED, log only), ``{slices}``
#: (PIPE_SEGMENTING_START, Step 3's own sentence).  And the
#: B1a additions, each in a log line or a composed row part: ``{index}``
#: (SEG_MODEL_CHANNEL: the agreed "{name} ({index})"), ``{axes}`` /
#: ``{spacing}`` (ROW_WIDE_LINE2, the agreed "{axes} · {c} ch · {spacing}"),
#: ``{why}`` / ``{preset}`` (log only).  And B2's ``{col}`` / ``{stacks}``
#: (EXPORT_COLS_MISSING_FROM, one row of the merged-columns question; the
#: sentence is part of the agreed wording).
EXTRA_PLACEHOLDERS = {"msg", "shape", "slices", "index", "axes", "spacing", "why",
                      "preset",
                      # SEG_CHANNEL_ITEM_ROLE (owner 2026-09-29): the
                      # model's role for a channel, "membrane" / "nuclei"
                      "role", "col", "stacks",
                      # LOG_MESH_SKIPPED (log only): the skipped cell ids
                      "ids",
                      # EXPORT_DEFAULT_NAME: today's date
                      "date",
                      # STATE_REROOTED / DELETE_OUTSIDE_REFUSED (log only):
                      # the copied-from workspace and a refused path
                      "old", "path"}


def _table_names() -> list:
    """The agreed constant names, each a tuple of alternatives (the
    "A / _B" shorthand already expanded both ways)."""
    data = json.loads(NAMES_FIXTURE.read_text(encoding="utf-8"))["data"]
    return [tuple(alts) for alts in data]


def test_the_names_fixture_exists():
    assert NAMES_FIXTURE.is_file() and NAMES_FIXTURE.stat().st_size > 0


def _all_constants():
    from napari_epicycle.ui_shell import theme
    found = {k: v for k, v in vars(T).items() if k.isupper()}
    for k in ("PIPE_SEGMENTING_START", "PIPE_SEGMENTING_ITEM",
              "PIPE_SEGMENTING_VIEW", "PIPE_SEGMENTATION_DONE",
              "PIPE_FAILED_SUFFIX", "PIPE_TIME_SUFFIX",
              "PIPE_RECONSTRUCTING_START", "PIPE_RECONSTRUCTING_ITEM",
              "PIPE_RECONSTRUCTION_DONE", "PIPE_CANCELLING",
              "PIPE_ERROR_PREDICT", "PIPE_ERROR_RECON", "ROW_SEGMENTING",
              "ROW_RECONSTRUCTING", "RUN_ROW_DEFAULT", "MODEL_NONE",
              "MODEL_ACTIVE_PAIR"):
        found[k] = getattr(theme, k)
    return found


def test_every_spec_constant_exists():
    names = _table_names()
    assert len(names) > 150
    have = _all_constants()
    missing = [c for c in names if not any(n in have for n in c)]
    assert not missing, missing


def _sentences():
    for k, v in _all_constants().items():
        if isinstance(v, str):
            yield k, v
        elif isinstance(v, (tuple, list)):
            for i, x in enumerate(v):
                if isinstance(x, str):
                    yield f"{k}[{i}]", x
                elif isinstance(x, (tuple, list)):
                    for j, y in enumerate(x):
                        if isinstance(y, str):
                            yield f"{k}[{i}][{j}]", y
        elif isinstance(v, dict):
            for kk, x in v.items():
                if isinstance(x, str):
                    yield f"{k}[{kk}]", x
                elif isinstance(x, tuple):
                    for j, y in enumerate(x):
                        yield f"{k}[{kk}][{j}]", y


def test_every_placeholder_is_defined():
    bad = []
    allowed = SPEC_PLACEHOLDERS | EXTRA_PLACEHOLDERS
    for name, text in _sentences():
        for ph in re.findall(r"\{([^{}]*)\}", text):
            if ph not in allowed:
                bad.append((name, ph))
    assert not bad, bad


@pytest.mark.parametrize("pattern", [
    r"[A-Za-z]:[\\/]", r"\\\\", "Traceback", r"\{exc", r"\{e\}", "Errno",
    "Exception", r"\.py\b",
])
def test_no_raw_path_or_exception_text(pattern):
    hits = [(n, t) for n, t in _sentences() if re.search(pattern, t)]
    assert not hits, hits


def test_step4_text_is_qt_free():
    tree = ast.parse(Path(T.__file__).read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add(node.module or "")
    assert not any(m.startswith(("qtpy", "PyQt", "napari")) for m in mods), mods


def test_count_phrase():
    assert T.count_phrase(1, "stack") == "1 stack"
    assert T.count_phrase(3, "stack") == "3 stacks"
    assert T.fmt_count(4653) == "4,653"


def test_the_advanced_tables_agree():
    grouped = [w for _l, _e, ws in T.STEP4_ADVANCED_GROUPS for w in ws]
    knobs = ("batch_size_sb", "method_combo", "n_cpu_sb")
    assert len(grouped) == len(set(grouped)) == 40
    assert set(grouped) | set(knobs) == set(T.STEP4_ADVANCED_LABELS)
    assert len(T.STEP4_ADVANCED_LABELS) == 43
    for w in T.STEP4_ADVANCED_LABELS:
        assert w in T.STEP4_HELP_LINES, w
    for w in T.STEP4_HIDDEN_KNOBS:
        assert w not in T.STEP4_HELP_LINES
        assert w not in T.STEP4_ADVANCED_LABELS


def test_the_shared_block_renders_both_units():
    from napari_epicycle.ui_shell import theme
    assert theme.RUN_ROW_DEFAULT.format(
        n=6, units=theme.pipe_units("stack", 6)) == "● Default  (6 stacks)"
    assert theme.RUN_ROW_DEFAULT.format(
        n=1, units=theme.pipe_units("stack", 1)) == "● Default  (1 stack)"
    assert theme.PIPE_CANCELLING.format(unit="stack") == (
        "Cancelling... will stop after current stack completes")


def test_the_reference_window_is_titled_with_the_buttons_words():
    """The owner's ruling (2026-10-05): the window "What is measured?"
    opens is titled with the button's own words."""
    assert T.FEATURE_REFERENCE_TITLE == T.WHAT_MEASURED_BTN == \
        "What is measured?"
