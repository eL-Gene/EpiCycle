"""The Step 2 guides quote only what the page says.

Every straight-double-quoted string in the nine ``docs/guides/step2_*.md``
guides must be a sentence the panel really shows.  Two checks, so the test
has teeth before and after the words module is complete:

1. **Against the agreed wording (self-contained, always on).**  Every quote
   must occur inside the agreed sentences kept as a fixture beside this test
   (``fixtures/step2_look_corpus.txt``: the state text and the "New" column of
   the sentence table), a ``{placeholder}`` there standing for any text.
2. **Against ``ui/_step2_text.py`` (the stricter one).**  Every quote must
   match a constant of that module (placeholders as ``.+?``).  It is
   enforced for the guides whose words brick L already owns (Session) and for ALL guides once the module sets
   ``STEP2_TEXT_COMPLETE = True`` (the integration pass); until then the
   other guides' check is skipped with a reason.

Quoted sentences must not contain a straight double quote inside; a
sentence that does is described in prose, not quoted.  Modelled on
``tests/fusion_reconstruction/test_step3_pr_guide_parity.py``.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GUIDES_DIR = ROOT / "docs" / "guides"
SPEC = Path(__file__).parent / "fixtures" / "step2_look_corpus.txt"

#: every guide of Step 2, with the fewest quoted sentences it must carry
ALL_GUIDES = {
    GUIDES_DIR / "step2_session.md": 14,
    GUIDES_DIR / "step2_patches.md": 7,
    GUIDES_DIR / "step2_slices.md": 4,
    GUIDES_DIR / "step2_curation_progress.md": 2,
    GUIDES_DIR / "step2_model.md": 5,
    GUIDES_DIR / "step2_round_run.md": 10,
    GUIDES_DIR / "step2_training_setup.md": 14,
    GUIDES_DIR / "step2_run_management.md": 7,
    GUIDES_DIR / "step2_diagnostics.md": 2,
}
#: the guides whose words ``_step2_text`` already holds in full (brick L)
CONSTANTS_NOW = {GUIDES_DIR / "step2_session.md"}
SHORT_FIXED = 5
NUMERIC = {"n", "k", "m", "i", "s", "t", "f", "p", "z", "y", "x", "c", "px",
           "um", "e", "N", "r", "a", "b"}


def _norm(s: str) -> str:
    return " ".join(str(s).split())


def quotes(path: Path):
    return [_norm(q) for q in re.findall(r'"([^"]+)"',
                                         path.read_text(encoding="utf-8"))]


# -- check 1: the spec ---------------------------------------------------------

WILD = "\x00"
#: a placeholder holds a name, a number or a short phrase, never a sentence
WILD_MAX = 30


def _spec_corpus() -> list:
    """The fixture's rows (the "New" column cells and the state lines,
    already cleaned), normalised, each ``{placeholder}`` replaced by
    ``WILD``.  The first line of the fixture is its origin note."""
    cells = SPEC.read_text(encoding="utf-8").splitlines()[1:]
    return [_norm(re.sub(r"\{[^{}]*\}", WILD, c)) for c in cells]


def test_the_corpus_fixture_exists():
    assert SPEC.is_file() and SPEC.stat().st_size > 0
    assert len(_spec_corpus()) > 100


def occurs_in(quote: str, cell: str) -> bool:
    """Does ``quote`` equal some substring of ``cell``, a WILD of the cell
    standing for one or more characters of the quote?"""
    q, c = quote, cell

    @lru_cache(maxsize=None)
    def go(qi: int, ci: int, run: int) -> bool:
        # run = how many quote chars the WILD at ci has eaten so far
        if qi == len(q):
            return True          # the quote may end anywhere in the cell
        if run:
            if run < WILD_MAX and go(qi + 1, ci, run + 1):
                return True
            return go(qi, ci + 1, 0)
        if ci == len(c):
            return False
        if c[ci] == WILD:
            return go(qi + 1, ci, 1)
        return q[qi] == c[ci] and go(qi + 1, ci + 1, 0)

    # a quote never STARTS inside a placeholder
    return any(c[start] != WILD and go(0, start, 0)
               for start in range(len(c)))


def in_spec(quote: str, corpus: list) -> bool:
    return any(occurs_in(quote, c) for c in corpus if quote[:4] in c
               or WILD in c)


@pytest.fixture(scope="module")
def corpus():
    return _spec_corpus()


# -- check 2: the constants module -------------------------------------------------


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, (tuple, list)):
        for v in value:
            yield from _strings(v)
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)


def template(s: str):
    names = re.findall(r"\{([^{}]*)\}", s)
    parts = re.split(r"\{[^{}]*\}", s)
    fixed = "".join(parts)
    text_hole = ".+?" if len(fixed.strip()) >= SHORT_FIXED else r"\S+?"
    out = re.escape(parts[0])
    for name, part in zip(names, parts[1:]):
        base = name.split(":")[0]
        out += (r"[0-9][0-9.,:]*" if base in NUMERIC else text_hole)
        out += re.escape(part)
    return re.compile("^" + out + "$")


def _text_module():
    return pytest.importorskip(
        "napari_epicycle.hitl_training_annotation.ui._step2_text")


def constant_templates() -> list:
    T = _text_module()
    out = []
    for k, v in vars(T).items():
        if k.isupper() or k == "_SPACING_BY_SOURCE":
            out.extend(_strings(v))
    return [template(_norm(s)) for s in out if s and s.strip()]


def _matches(q: str, templates) -> bool:
    return any(t.match(q) for t in templates)


# -- the tests -------------------------------------------------------------------


@pytest.mark.parametrize("guide", sorted(ALL_GUIDES, key=str),
                         ids=lambda g: g.stem)
def test_the_guide_exists_and_quotes_enough(guide):
    assert guide.exists(), guide
    assert len(quotes(guide)) >= ALL_GUIDES[guide], len(quotes(guide))


@pytest.mark.parametrize("guide", sorted(ALL_GUIDES, key=str),
                         ids=lambda g: g.stem)
def test_no_stray_straight_quote_breaks_the_pairing(guide):
    """An odd number of straight double quotes would pair the wrong ones."""
    assert guide.read_text(encoding="utf-8").count('"') % 2 == 0


@pytest.mark.parametrize("guide", sorted(ALL_GUIDES, key=str),
                         ids=lambda g: g.stem)
def test_every_quoted_sentence_is_in_the_spec(corpus, guide):
    bad = [q for q in quotes(guide) if not in_spec(q, corpus)]
    assert not bad, "the guide quotes what the spec does not say:\n" + \
        "\n".join(bad)


@pytest.mark.parametrize("guide", sorted(ALL_GUIDES, key=str),
                         ids=lambda g: g.stem)
def test_every_quoted_sentence_is_a_real_constant(guide):
    T = _text_module()
    if guide not in CONSTANTS_NOW and not getattr(
            T, "STEP2_TEXT_COMPLETE", False):
        pytest.skip("_step2_text.py is not complete yet "
                    "(set STEP2_TEXT_COMPLETE = True at integration)")
    templates = constant_templates()
    bad = [q for q in quotes(guide) if not _matches(q, templates)]
    assert not bad, "the guide quotes what the page does not say:\n" + \
        "\n".join(bad)


def test_the_session_guide_quotes_the_session_words():
    T = _text_module()
    qs = set(quotes(GUIDES_DIR / "step2_session.md"))
    for const in (T.SESSION_WORKSPACE_DESC, T.SESSION_PATCH_FOLDER_DESC,
                  T.SESSION_PATCH_FOLDER_IDLE, T.SESSION_PATCHES_DESC,
                  T.SESSION_PATCHES_EMPTY_LINES[0],
                  T.SESSION_PATCHES_EMPTY_LINES[1],
                  T.RUN_SUMMARY_DESC, T.RUN_SUMMARY_FOOT,
                  T.RUN_SUMMARY_EMPTY_LINES[0], T.ROW_NO_RECORDS,
                  T.ROW_VOXEL_NONE):
        assert _norm(const) in qs, const


def test_the_patches_guide_quotes_the_curation_patches_words():
    T = _text_module()
    qs = set(quotes(GUIDES_DIR / "step2_patches.md"))
    for const in (T.CURATION_PATCHES_DESC, T.ROW_NOT_STARTED,
                  T.CURATION_PATCHES_EMPTY_LINES[0]):
        assert _norm(const) in qs, const


def test_the_guides_name_every_status_and_its_word():
    """D8: the five status words are in the Slices guide's table."""
    T = _text_module()
    text = (GUIDES_DIR / "step2_slices.md").read_text(encoding="utf-8")
    for word in T.STATUS_WORDS.values():
        assert word in text, word


def test_the_round_numbers_are_the_disk_numbers():
    """D16: the guides say the first round is ``round_00``."""
    for name in ("step2_session.md", "step2_curation_progress.md",
                 "step2_round_run.md", "step2_training_setup.md"):
        assert "round_00" in (GUIDES_DIR / name).read_text(
            encoding="utf-8"), name


def test_the_old_words_are_gone():
    """The words this pass retired must not survive in a guide."""
    old = ("Update + Train", "Cancel Training", "Open Multiview Workspace",
           "Round Status", "Target View", "Patch Directory", "Base Model",
           "Min mask size", "Auxiliary channel", "Compute flows",
           "From Scratch", "Auto-update model")
    for guide in ALL_GUIDES:
        text = guide.read_text(encoding="utf-8")
        for w in old:
            assert w not in text, (guide.name, w)


# -- teeth -------------------------------------------------------------------------


def test_a_made_up_sentence_matches_nothing(corpus):
    assert not in_spec("Press the big red button to train everything.",
                       corpus)
    assert not in_spec("Update + Train", corpus)
    assert not in_spec("Open Multiview Workspace", corpus)
    assert not _matches("Press the big red button to train everything.",
                        constant_templates())


def test_the_spec_check_accepts_a_placeholder_example(corpus):
    assert in_spec("15 patches · 4 in a trained round · 2 finalized · "
                   "8 in progress · 1 not started", corpus)
    assert in_spec("Click a patch to load it. Up / Down load too.", corpus)
    # a sentence with a changed word is refused
    assert not in_spec("Click a patch to unload it. Up / Down load too.",
                       corpus)


def test_a_reworded_sentence_turns_the_constants_check_red():
    templates = constant_templates()
    assert _matches("Click a patch to load it. Up / Down load too.",
                    templates)
    assert not _matches("Click a patch to load it. Up / Down load it too.",
                        templates)
