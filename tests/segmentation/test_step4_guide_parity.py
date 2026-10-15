"""The Step 4 guides quote only what the page says (the guides live in docs/guides/).

Every straight-double-quoted string in the three guides must match a
template from one of three sources: ``_step4_text`` (its constants, the
labels and ``STEP4_HELP_LINES``), the shared pipeline block of
``ui_shell/theme.py``, and the composed ``PIPE_*_DONE`` forms (a done head
with its failure and time suffixes).  ``{placeholders}`` become ``.+?``;
a template whose fixed text is short may only match a quote through
a count placeholder (``{n}``, ``{k}``…) matches a number only, so "{n} ticked"
cannot match "Ready · 4 stacks ticked"; a template with almost no fixed
text lets its placeholders hold one word only (the notes' rule 4).  Whitespace is normalised (rule 5).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from napari_epicycle.segmentation import _step4_text as T

ROOT = Path(__file__).resolve().parents[2]
GUIDES = [ROOT / "docs" / "guides" / n for n in
          ("step4_session.md", "step4_segment.md", "step4_measure_tidy.md")]
SHORT_FIXED = 5
META = re.compile(r"^meta_[a-z_]+$")


def _norm(s: str) -> str:
    return " ".join(str(s).split())


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, (tuple, list)):
        for v in value:
            yield from _strings(v)
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)


def sentences() -> list:
    from napari_epicycle.ui_shell import theme as th
    out = []
    for k, v in vars(T).items():
        if k.isupper():
            out.extend(_strings(v))
    shared = ("PIPE_SEGMENTING_START", "PIPE_SEGMENTING_ITEM",
              "PIPE_SEGMENTING_VIEW", "PIPE_SEGMENTATION_DONE",
              "PIPE_RECONSTRUCTING_START", "PIPE_RECONSTRUCTING_ITEM",
              "PIPE_RECONSTRUCTION_DONE", "PIPE_CANCELLING",
              "PIPE_ERROR_PREDICT", "PIPE_ERROR_RECON", "ROW_SEGMENTING",
              "ROW_RECONSTRUCTING", "RUN_ROW_DEFAULT", "MODEL_NONE",
              "MODEL_ACTIVE_PAIR")
    out.extend(getattr(th, k) for k in shared)
    for head in (th.PIPE_SEGMENTATION_DONE, th.PIPE_RECONSTRUCTION_DONE):
        out.append(head + th.PIPE_TIME_SUFFIX)
        out.append(head + th.PIPE_FAILED_SUFFIX + th.PIPE_TIME_SUFFIX)
    return [_norm(s) for s in out if s and s.strip()]


#: count placeholders: a number, never a phrase
NUMERIC = {"n", "k", "m", "g", "b", "u", "i", "s", "v", "c", "Z", "Y", "X",
           "z", "y", "x", "um", "px", "pct"}


def template(s: str):
    """A constant as a regex.  A count placeholder matches a number only; a
    text placeholder matches any text, unless the template's fixed text is
    short (rule 4), when it matches one word at most."""
    names = re.findall(r"\{([^{}]*)\}", s)
    parts = re.split(r"\{[^{}]*\}", s)
    fixed = "".join(parts)
    text_hole = ".+?" if len(fixed.strip()) >= SHORT_FIXED else r"\S+?"
    out = re.escape(parts[0])
    for name, part in zip(names, parts[1:]):
        out += (r"[0-9][0-9.,:]*" if name in NUMERIC else text_hole)
        out += re.escape(part)
    return re.compile("^" + out + "$")


@pytest.fixture(scope="module")
def templates():
    return [template(s) for s in sentences()]


def quotes(path: Path):
    return [_norm(q) for q in re.findall(r'"([^"]+)"',
                                         path.read_text(encoding="utf-8"))]


def matches(q, templates) -> bool:
    return bool(META.match(q)) or any(t.match(q) for t in templates)


@pytest.mark.parametrize("guide", GUIDES, ids=lambda p: p.name)
def test_the_guide_exists_and_quotes_something(guide):
    assert guide.exists()
    assert len(quotes(guide)) >= 25


@pytest.mark.parametrize("guide", GUIDES, ids=lambda p: p.name)
def test_every_quoted_sentence_is_a_real_constant(guide, templates):
    bad = [q for q in quotes(guide) if not matches(q, templates)]
    assert not bad, f"{guide.name} quotes what the page does not say:\n" + \
        "\n".join(bad)


def test_short_templates_do_not_match_longer_sentences(templates):
    assert not matches("Done: 4,410 cells, saved to labels_3d.tif",
                       [template(T.QC_BAR_IDLE)])
    assert not matches("Ready · 4 stacks ticked",
                       [template(T.SEG_TICKED_LINE)])
    assert matches("Ready · 4 stacks ticked", [template(T.STEP4_SEG_IDLE)])
    assert matches("4 ticked", [template(T.SEG_TICKED_LINE)])
    assert matches("Done", [template(T.QC_BAR_IDLE)])


def test_a_made_up_sentence_matches_nothing(templates):
    assert not matches("Press the big red button to segment everything.",
                       templates)
    assert not matches("Segmentation finished without problems.", templates)


def test_the_composed_done_lines_are_templates(templates):
    assert matches("Segmentation complete: 3/3 stacks [11:44]", templates)
    assert matches("Reconstruction complete: 2/3 stacks (1 failed) [7:38]",
                   templates)
