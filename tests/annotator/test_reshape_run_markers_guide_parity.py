"""The Reshape guide says what the tab shows about markers taken from labels.

``docs/guides/annotator_3d_reshape.md`` is the ``?`` page of the Reshape
tab.  In the passages about markers from another run or a mask file, the
marker list, the wall line and the empty run picker, every straight
double-quoted string must be one of the tab's own sentences (placeholders
read as ``.+?``), so a sentence reworded in the code turns this red until
the guide follows.  The sentences that went away must be gone from the
guide and from the code.

Pure text; no Qt.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
GUIDE = REPO / "docs" / "guides" / "annotator_3d_reshape.md"
TEXT_MODULE = REPO / "napari_epicycle" / "annotator" / "_reshape_text.py"

#: (first words, words that end the passage) -- the passages this test reads.
PASSAGES = (
    ("**Markers from another run of this patch**",
     "**Markers from a mask file**"),
    ("Press **Find markers** and the tool looks now",
     "Change any marker setting"),
    ("Under the list, in grey, the tool reports what it saw.",
     "### The amber block above the list"),
    ("**The run picker says", "**Markers from the last object"),
)

#: The sentences that were retired with the touching rule.
RETIRED = (
    "Nothing is unticked for you",
    "Use Browse to pick a mask file instead",
    "keeps them all",
    "Left unticked: it touches",
    "Tick it to use it",
)

#: A quoted sentence may wrap onto the next line of the markdown.
_QUOTED = re.compile(r'"([^"]+)"')
_PLACEHOLDER = re.compile(r"\{[^{}]*\}")


def _guide() -> str:
    return GUIDE.read_text(encoding="utf-8")


def _passage(text, start, end) -> str:
    i = text.index(start)
    j = text.index(end, i + len(start))
    return text[i:j]


def _templates():
    """Every sentence constant of the text module, plus the wall line."""
    import ast

    tree = ast.parse(TEXT_MODULE.read_text(encoding="utf-8"))
    out = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, (
                ast.Constant, ast.JoinedStr, ast.BinOp)):
            try:
                value = ast.literal_eval(node.value)
            except Exception:
                continue
            if isinstance(value, str):
                out.append(value)
    from packages.epicycle_annotator.reshape.engine import WALL_CONTRAST_LINE

    out.append(WALL_CONTRAST_LINE)
    return out


def _as_regex(template: str):
    parts = _PLACEHOLDER.split(template)
    return re.compile("^" + ".+?".join(re.escape(p) for p in parts) + "$")


def _matches(sentence: str, templates) -> bool:
    return any(_as_regex(t).match(sentence) for t in templates)


def _quoted_in_passages():
    text = _guide()
    out = []
    for start, end in PASSAGES:
        out.extend(" ".join(q.split()) for q in
                   _QUOTED.findall(_passage(text, start, end)))
    return out


def test_every_passage_is_in_the_guide():
    text = _guide()
    for start, end in PASSAGES:
        assert _passage(text, start, end)


def test_every_quoted_sentence_is_one_the_tab_shows():
    templates = _templates()
    bad = [q for q in _quoted_in_passages() if not _matches(q, templates)]
    assert not bad, "quoted in the guide but shown nowhere: {}".format(bad)


def test_the_left_out_line_is_quoted():
    """Preview with no Find leaves the touching labels out and says so
    beside the cut; the guide shows that sentence."""
    from napari_epicycle.annotator import _reshape_text as T

    rx = _as_regex(T.MARKERS_LEFT_OUT)
    assert any(rx.match(q) for q in _quoted_in_passages())


def test_the_all_touching_refusal_is_quoted():
    from napari_epicycle.annotator import _reshape_text as T

    rx = _as_regex(T.MARKERS_ALL_TOUCHING)
    assert any(rx.match(q) for q in _quoted_in_passages())


def test_the_passages_quote_enough_to_mean_something():
    assert len(_quoted_in_passages()) >= 4


def test_the_test_has_teeth():
    templates = _templates()
    assert not _matches("4 markers found, 5 grazing ones dropped", templates)
    assert _matches("2 markers kept, 3 touching ones left unticked",
                    templates)
    assert _matches("Wall contrast 0.12-0.81.", templates)


@pytest.mark.parametrize("sentence", RETIRED)
def test_a_retired_sentence_is_gone_from_the_guide(sentence):
    assert sentence not in _guide()


@pytest.mark.parametrize("sentence", RETIRED)
def test_a_retired_sentence_is_gone_from_the_code(sentence):
    roots = (REPO / "napari_epicycle" / "annotator",
             REPO / "packages" / "epicycle_annotator" / "reshape")
    hits = []
    for root in roots:
        for path in root.rglob("*.py"):
            if sentence in path.read_text(encoding="utf-8"):
                hits.append(str(path.relative_to(REPO)))
    assert not hits, hits
