"""The Model Refinement guides quote only what the page says
(Brick B1).

Every straight-double-quoted string in the section's three guides --
``step3_model_refinement_patch_selection.md`` (extraction merged in),
``_fine_tuning.md`` and ``_training_config.md`` -- must match a constant of
``fusion_reconstruction/_step3_mr_text.py`` (a ``{placeholder}`` becomes
``.+?``, a count placeholder matches a number only, a template with almost
no fixed text lets a placeholder hold one word only; whitespace is
normalised), so a reworded on-screen sentence turns this red until the
guide follows.  Modelled on ``test_step3_pr_guide_parity.py``.  Straight
double quotes are for ON-SCREEN sentences only; anything else in a guide
wears typographic quotes or backticks.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GUIDES_DIR = ROOT / "docs" / "guides"
PATCHES = GUIDES_DIR / "step3_model_refinement_patch_selection.md"
EXTRACTION = GUIDES_DIR / "step3_model_refinement_extraction.md"  # merged away
FINE_TUNING = GUIDES_DIR / "step3_model_refinement_fine_tuning.md"
TRAINING = GUIDES_DIR / "step3_model_refinement_training_config.md"
#: every guide of the section, with the fewest quotes it must carry
ALL_GUIDES = {PATCHES: 19, FINE_TUNING: 14, TRAINING: 8}
SHORT_FIXED = 5
NUMERIC = {"n", "k", "m", "i", "t", "e", "l", "a"}


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
    from napari_epicycle.fusion_reconstruction import _step3_mr_text as M
    out = []
    for k, v in vars(M).items():
        if k.isupper():
            out.extend(_strings(v))
    return [_norm(s) for s in out if s and s.strip()]


def template(s: str):
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
    return any(t.match(q) for t in templates)


@pytest.mark.parametrize("guide", sorted(ALL_GUIDES, key=str),
                         ids=lambda g: g.stem)
def test_the_guide_exists_and_quotes_something(guide):
    assert guide.exists()
    assert len(quotes(guide)) >= ALL_GUIDES[guide]


@pytest.mark.parametrize("guide", sorted(ALL_GUIDES, key=str),
                         ids=lambda g: g.stem)
def test_every_quoted_sentence_is_a_real_constant(templates, guide):
    bad = [q for q in quotes(guide) if not matches(q, templates)]
    assert not bad, "the guide quotes what the page does not say:\n" + \
        "\n".join(bad)


def test_the_patch_selection_guide_quotes_the_list_and_its_lines():
    from napari_epicycle.fusion_reconstruction import _step3_mr_text as M
    qs = set(quotes(PATCHES))
    for const in (M.MR_PATCHES_DESC, *M.MR_EMPTY_LINES,
                  M.MR_ROW_NOT_EXTRACTED, M.MR_ROW_STALE, M.MR_ROW_MIXED,
                  M.MR_WILL_TRAIN_NOTHING, M.MR_EXTRACT_NEEDS_TICK):
        assert _norm(const) in qs, const
    # the patterns are quoted as examples
    for tmpl in (M.MR_ROW_PAIRS_LATERAL, M.MR_ROW_PAIRS_AXIAL, M.MR_TICKED_COUNT,
                 M.MR_WILL_TRAIN, M.MR_FEW_PAIRS_WARN):
        t = template(_norm(tmpl))
        assert any(t.match(q) for q in qs), tmpl


def test_the_extraction_guide_quotes_the_extract_lines():
    from napari_epicycle.fusion_reconstruction import _step3_mr_text as M
    qs = set(quotes(PATCHES))
    for const in (M.MR_EXTRACT_NEEDS_TICK, M.MR_ROW_STALE, M.MR_ROW_MIXED):
        assert _norm(const) in qs, const
    for tmpl in (M.MR_EXTRACT_RUNNING, M.MR_EXTRACT_DONE,
                 M.MR_EXTRACT_NO_PAIRS):
        t = template(_norm(tmpl))
        assert any(t.match(q) for q in qs), tmpl


def test_the_fine_tuning_guide_quotes_the_foot_and_the_card():
    from napari_epicycle.fusion_reconstruction import _step3_mr_text as M
    qs = set(quotes(FINE_TUNING))
    for const in (M.MR_FT_DESC, M.MR_STATUS_NOTHING, M.MR_STATUS_BUSY,
                  M.MR_LOSS_EMPTY, M.MR_OUT_DESC, M.MR_OUT_EMPTY):
        assert _norm(const) in qs, const
    for tmpl in (M.MR_HINT_BOTH, M.MR_HINT_LATERAL, M.MR_STATUS_IDLE,
                 M.MR_STATUS_NAME_TAKEN, M.MR_STATUS_NO_SOURCE,
                 M.MR_STATUS_RUNNING_VAL, M.MR_STATUS_DONE,
                 M.MR_STATUS_FAILED, M.MR_NEW_RUN_HINT):
        t = template(_norm(tmpl))
        assert any(t.match(q) for q in qs), tmpl


def test_the_training_guide_quotes_the_presets_and_the_name_box():
    from napari_epicycle.fusion_reconstruction import _step3_mr_text as M
    qs = set(quotes(TRAINING))
    assert _norm(M.MR_TC_DESC) in qs
    assert _norm(M.MR_MODEL_NAME_PLACEHOLDER) in qs
    for name, desc in M.PRESET_DESCRIPTIONS.items():
        assert _norm(desc) in qs, name


def test_the_presets_say_what_they_do_and_never_where_training_starts():
    from napari_epicycle.fusion_reconstruction import _step3_mr_text as M
    for name, desc in M.PRESET_DESCRIPTIONS.items():
        assert "from scratch" not in desc.lower(), name
    assert set(M.PRESET_DESCRIPTIONS) == set(M.MR_PRESET_ITEMS)


def test_the_stale_claims_are_gone_from_the_guides():
    """B1 retired these: the button's second name, the 'stays disabled'
    claim, the clean-cancel claim, the model pack and the old model path."""
    text = {g: g.read_text(encoding="utf-8") for g in ALL_GUIDES}
    ft = text[FINE_TUNING]
    assert "Start Fine-Tuning" not in ft
    assert "Cancel Training" not in ft
    assert "stays disabled" not in ft
    assert "Cancellation is clean" not in ft
    assert "Register as Model Pack" not in ft
    assert "models/refinement/" not in ft
    assert "Start Fine-Tuning" not in text[TRAINING]
    for g, t in text.items():
        assert "refinement_training_data" not in t, g
        assert "Select All" not in t, g


def test_a_reworded_sentence_would_fail(templates):
    """Teeth: a guide quoting a reworded sentence is caught, and a
    constant's own text is accepted."""
    from napari_epicycle.fusion_reconstruction import _step3_mr_text as M
    assert matches(_norm(M.MR_PATCHES_DESC), templates)
    assert matches("2 of 5 ticked", templates)
    assert not matches("Ticked patches are what Extract runs on.",
                       templates)
    assert not matches("Press the big red button to train everything.",
                       templates)
    assert not matches("Start Fine-Tuning", templates)
    assert not matches("Cancel Training", templates)


def test_the_extraction_guide_is_merged_into_patch_selection():
    assert not EXTRACTION.exists()
    text = PATCHES.read_text(encoding="utf-8")
    for h in ("Extracting training pairs", "Extraction settings",
              "Train / validation split"):
        assert f"## {h}" in text, h


def test_the_loss_header_suffix_stays_empty():
    from types import SimpleNamespace
    from napari_epicycle.fusion_reconstruction.ui import (
        FusionReconstructionCuration as C)
    from qtpy.QtWidgets import QApplication, QLabel
    app = QApplication.instance() or QApplication([])
    eng = QLabel("")
    ui = SimpleNamespace(
        refine_loss_collapsible=SimpleNamespace(_epicycle_engine_label=eng))
    C.set_refine_loss_names(ui, "refined_04")
    assert eng.text() == ""
