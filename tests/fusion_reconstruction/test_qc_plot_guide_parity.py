"""Guides row (brick P2): every straight-double-
quoted sentence in the three Curation guides is a sentence the screen can
show -- a ``_qc_plot_text`` constant, a ``_qc_pairing_text`` constant (the
pairing check: the Extract Features row, the plot's match words and
notes, the refusals) or a ``HIST_*`` word of the triage histogram,
placeholders read as ``.+?`` and a ``"title|body"`` sentence as its two
halves.  A reworded sentence turns this red until the guide follows.

Curly quotes are not read (so they cannot carry a claim past the check);
anything that is not an on-screen sentence is written in **bold** or
*italics* instead.
"""

from __future__ import annotations

import re
from pathlib import Path

GUIDES = Path(__file__).resolve().parents[2] / "docs" / "guides"
FILES = ("step3_curation_qc_plot.md", "step3_curation_qc_features.md",
         "step3_curation_curate.md")
QUOTE = re.compile(r'"([^"\n]+)"')


def _templates() -> list:
    from napari_epicycle.fusion_reconstruction import _qc_pairing_text as PT
    from napari_epicycle.fusion_reconstruction import _qc_plot_text as T
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as ST
    from napari_epicycle.fusion_reconstruction import triage_histogram_panel as H

    raw = []

    def take(v):
        if isinstance(v, str):
            raw.extend(v.split("|"))
        elif isinstance(v, dict):
            for x in v.values():
                take(x)

    for module in (T, PT):
        for name in dir(module):
            if name.isupper():
                take(getattr(module, name))
    for name in dir(H):
        if name.startswith("HIST_"):
            take(getattr(H, name))
    # the QC Settings dialog's own sentences (the Features guide quotes them)
    for name in dir(ST):
        if name.startswith("QC_"):
            take(getattr(ST, name))
    out = []
    for t in raw:
        t = t.strip()
        if not t:
            continue
        parts = re.split(r"\{[^}]*\}", t)
        # A template that is mostly placeholders ("{tier} {n}", "C{id}")
        # would match any quote: it counts only with a real word in it.
        if not re.search(r"[A-Za-z]{3}", "".join(parts)):
            continue
        out.append(re.compile("^" + ".+?".join(re.escape(p) for p in parts)
                              + "$"))
    return out


def quotes(text: str) -> list:
    # code spans are not prose
    text = re.sub(r"`[^`]*`", "", text)
    return [q.strip() for q in QUOTE.findall(text)]


def unmatched(text: str, templates) -> list:
    return [q for q in quotes(text)
            if not any(t.match(q) for t in templates)]


def test_every_quoted_sentence_is_on_screen():
    templates = _templates()
    bad = {}
    for f in FILES:
        miss = unmatched((GUIDES / f).read_text(encoding="utf-8"), templates)
        if miss:
            bad[f] = miss
    assert not bad, bad


def test_the_plot_guide_quotes_enough_sentences():
    n = len(quotes((GUIDES / FILES[0]).read_text(encoding="utf-8")))
    assert n >= 12, n


def test_the_check_has_teeth():
    templates = _templates()
    assert unmatched('It says "Lasso: 15 labels." here.', templates) == []
    assert unmatched('"Loading Wound_patch_1…"', templates) == []
    assert unmatched('"Press Extract under QC Features."', templates) == []
    assert unmatched('It says "Lasso selected 15 cells." here.',
                     templates) == ["Lasso selected 15 cells."]
    assert unmatched('"Sync viewer colors"', templates) == [
        "Sync viewer colors"]


# --- The pairing check's section of the features guide ---------------------
PAIRING_HEADING = "## Comparing with another run"


def _pairing_section() -> str:
    text = (GUIDES / FILES[1]).read_text(encoding="utf-8")
    start = text.index(PAIRING_HEADING)
    end = text.find("\n## ", start + len(PAIRING_HEADING))
    return text[start:] if end < 0 else text[start:end]


def test_the_pairing_section_quotes_enough_sentences():
    q = quotes(_pairing_section())
    assert len(q) >= 13, q


def test_the_pairing_section_quotes_every_match_word_and_the_limit():
    from napari_epicycle.fusion_reconstruction import _qc_pairing_text as PT

    q = quotes(_pairing_section())
    for s in (PT.PICKER_LABEL, PT.colour_by_match("Nuclei"), PT.STATUS_NONE,
              PT.STATUS_ONE, PT.STATUS_SEVERAL, PT.STATUS_SHARED,
              PT.STATUS_TOO_SMALL, PT.STATUS_NOT_MEASURED, PT.LIMIT,
              PT.PREDATE_EDITS):
        assert s in q, s


def test_the_pairing_section_states_the_link_floor_in_use():
    from napari_epicycle.fusion_reconstruction import _qc_pairing_text as PT
    from napari_epicycle.fusion_reconstruction.qc_pairing_group import (
        LINK_FLOOR,
    )

    assert PT.link_rule(LINK_FLOOR) in quotes(_pairing_section())
    assert PT.link_rule(LINK_FLOOR) == (
        "Two objects are linked when their shared voxels are at least "
        "0.20 of the smaller one.")


def test_the_pairing_section_check_has_teeth():
    templates = _templates()
    section = _pairing_section()
    assert unmatched(section, templates) == []
    for old, new in (("0.20 of the smaller one", "0.20 of the larger one"),
                     ("not that the cell is right", "not that it is right"),
                     ('"shared match"', '"match shared"'),
                     ("predate your edits", "predate the edits")):
        assert old in section, old
        assert unmatched(section.replace(old, new), templates), old


def test_no_retired_words_in_the_plot_guide():
    text = (GUIDES / FILES[0]).read_text(encoding="utf-8")
    for retired in ("Sync Viewer Colors", "Sync viewer colors",
                    "histogram on the left", "Colormap", "Normalization",
                    "population mode", "Plot type (top combo)"):
        assert retired not in text, retired
    assert " cell " not in text and " cells " not in text
