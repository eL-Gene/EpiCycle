"""The Predict + Reconstruct guides quote only what the page says
(Guides).

Every straight-double-quoted string in the section's three guides --
``docs/guides/step3_predict_reconstruct_settings.md`` (R2a) and, since R3,
``step3_predict_reconstruct_model_source.md`` (the Model card and the run
row) and ``step3_predict_reconstruct_patch_state.md`` (the Patches card and
the pinned foot) -- must match a template
from ``fusion_reconstruction/_step3_pr_text.py`` (every constant, among
them the alias-mapped shared help lines ``S3_HELP_LINES`` and Advanced
labels ``S3_ADVANCED_LABELS``) or ONE OF THE SHARED CONSTANTS STEP 3
ACTUALLY SHOWS: the names of ``ui_shell/_recon_settings_text.py`` that
Step 3's ``ui.py`` / ``controller.py`` read as ``_RT.<NAME>`` /
``_rs_text.<NAME>`` (audit A LOW-4 -- not every uppercase name of the
shared module, which holds Step 4's words too).  ``{placeholders}`` become ``.+?``, a count
placeholder matches a number only, a template with almost no fixed text
lets a placeholder hold one word only; whitespace is normalised.  Modelled
on ``tests/segmentation/test_step4_guide_parity.py``.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GUIDES_DIR = ROOT / "docs" / "guides"
GUIDE = GUIDES_DIR / "step3_predict_reconstruct_settings.md"
MODEL_GUIDE = GUIDES_DIR / "step3_predict_reconstruct_model_source.md"
PATCHES_GUIDE = GUIDES_DIR / "step3_predict_reconstruct_patch_state.md"
#: V2: Session's patch list and its pictures
SESSION_GUIDE = GUIDES_DIR / "step3_session.md"
#: every guide of the section, with the fewest quotes it must carry
ALL_GUIDES = {GUIDE: 25, MODEL_GUIDE: 6, PATCHES_GUIDE: 12,
              SESSION_GUIDE: 6}
SHORT_FIXED = 5
NUMERIC = {"n", "k", "m", "i", "s", "px", "um"}


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


FR = ROOT / "napari_epicycle" / "fusion_reconstruction"
#: the module aliases Step 3's code reads the shared words through
SHARED_ALIASES = {"_RT", "_rs_text"}


def shared_names_step3_shows() -> set:
    """The shared constants Step 3's own code reads (``_RT.X`` /
    ``_rs_text.X`` in ``ui.py`` and ``controller.py``)."""
    names = set()
    for mod in ("ui.py", "controller.py"):
        tree = ast.parse((FR / mod).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id in SHARED_ALIASES
                    and node.attr.isupper()):
                names.add(node.attr)
    return names


def sentences() -> list:
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
    from napari_epicycle.ui_shell import _recon_settings_text as R
    out = []
    for k, v in vars(P).items():
        if k.isupper():
            out.extend(_strings(v))
    for k in shared_names_step3_shows():
        out.extend(_strings(getattr(R, k)))
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


def test_the_model_guide_quotes_the_model_card_and_the_run_row():
    """R3: the Model card's sentence, Session's pointer line, the run row's
    and Change…'s help lines, and the foot's no-model sentence."""
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
    qs = set(quotes(MODEL_GUIDE))
    help_ = P.S3_LAYOUT_HELP_LINES
    for const in (P.S3_MODEL_DESC, P.S3_SESSION_MODEL_POINTER,
                  P.S3_GATE_PICK_MODEL, help_["run_picker_combo"],
                  help_["new_run_button"], help_["manage_runs_button"],
                  help_["model_change_btn"]):
        assert _norm(const) in qs, const


def test_the_patches_guide_quotes_the_card_and_the_foot():
    """R3: the Patches card's sentence, the four buttons' and the link's
    help lines, and every gate sentence."""
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
    qs = set(quotes(PATCHES_GUIDE))
    help_ = P.S3_LAYOUT_HELP_LINES
    for const in (P.S3_PATCHES_DESC, help_["predict_selected_btn"],
                  help_["reconstruct_selected_btn"],
                  help_["pipeline_cancel_btn"],
                  help_["view_reconstructed_button"],
                  help_["pr_tick_all_link"], help_["_patch_state_list"],
                  P.S3_GATE_PICK_MODEL,
                  P.S3_GATE_NOTHING_TICKED, P.S3_GATE_NONE_RUNNABLE,
                  P.S3_GATE_NOTHING_TO_RECONSTRUCT):
        assert _norm(const) in qs, const


def test_the_patches_guide_quotes_the_ticks_the_lock_and_forget():
    """V1: the ticks, the protected row, the missing line, Forget, both
    lock families and the locked list note."""
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
    qs = set(quotes(PATCHES_GUIDE))
    for const in (P.S3_LIST_HELP, P.S3_TICK_ALL_HELP, P.S3_ROW_PROTECTED,
                  P.S3_CHIP_HAS_CURATION, P.S3_PATCH_MISSING,
                  P.S3_FORGET_MENU, P.S3_FORGET_REFUSED_LOCKED,
                  P.S3_FORGET_REFUSED_UNSAVED, P.S3_LOCK_PREDICT,
                  P.S3_LOCK_REFINE, P.S3_LOCK_LIST_NOTE):
        assert _norm(const) in qs, const


def test_the_patches_guide_quotes_the_chip_and_the_info():
    """R6: the "made by" chip in its three forms, its help line, the ⓘ's
    help line and the details window's own words."""
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
    qs = set(quotes(PATCHES_GUIDE))
    for const in (P.MADE_BY_UNKNOWN, P.S3_ROW_HELP_LINES["row_made_by"],
                  P.S3_ROW_HELP_LINES["row_info"], P.PROV_VIEW_LATERAL,
                  P.PROV_VIEW_AXIAL, P.PROV_SAVE, P.PROV_SAVED,
                  P.PROV_NO_BLOCK, P.PROV_LOCKED):
        assert _norm(const) in qs, const
    for tmpl in (P.MADE_BY, P.S3_MADE_BY_NOT_ACTIVE, P.PROV_TITLE):
        head = _norm(tmpl.split("{")[0]) if not tmpl.startswith("{")             else _norm(tmpl.split("}")[-1])
        assert any(head in q for q in qs), tmpl


def test_the_session_guide_quotes_the_list_and_its_pictures():
    """V2: Session's patch list, the picture's three tooltips and the
    missing line with its menu."""
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
    qs = set(quotes(SESSION_GUIDE))
    for const in (P.S3_SESSION_PATCHES_TITLE, P.S3_SESSION_LIST_HELP,
                  P.S3_THUMB_TIP, P.S3_THUMB_PENDING, P.S3_PATCH_MISSING,
                  P.S3_FORGET_MENU,
                  # A2 (board B): the cards, the folder row, the eye, the
                  # tier counts and the Pipeline status words
                  P.S3_SESSION_WORKSPACE_TITLE, P.S3_SESSION_WORKSPACE_DESC,
                  P.S3_SESSION_PATCHES_DESC, P.S3_SESSION_HEADER_EMPTY,
                  P.S3_SESSION_EMPTY_LINES[1], P.S3_SESSION_USE_FOLDER_BTN,
                  P.S3_SESSION_USING_DEFAULT, P.S3_SESSION_EYE_TIP,
                  P.S3_SESSION_TIERS_TIP, P.S3_SET_SPACING_BTN,
                  P.S3_SET_SPACING_LOCKED, P.S3_PIPELINE_TITLE,
                  P.S3_PIPELINE_DESC,
                  # A2 fold L5: the third empty line, the folder sentence,
                  # the count line's pattern, the click's sentence
                  P.S3_SESSION_EMPTY_LINES[2], P.S3_SESSION_USING_FOLDER,
                  P.S3_STATUS_PREDICTIONS_CACHED):
        assert _norm(const) in qs, const
    # the count line is quoted as an example of its pattern
    counts = template(_norm(P.S3_SESSION_HEADER_COUNTS))
    assert any(counts.match(q) for q in qs), P.S3_SESSION_HEADER_COUNTS
    failed = [q for q in qs if q.startswith(
        P.S3_THUMB_FAILED.split("{")[0])]
    assert failed, "the guide shows a failure tooltip"


def test_the_old_card_names_are_gone_from_the_two_guides():
    """R3 renamed the cards (Model Source -> Model, Patch State ->
    Patches) and hid Session's Model Status."""
    for guide, old in ((MODEL_GUIDE, "Model Status"),
                       (PATCHES_GUIDE, "Patch State")):
        assert old not in guide.read_text(encoding="utf-8"), (guide, old)


@pytest.mark.parametrize("guide", sorted(ALL_GUIDES, key=str),
                         ids=lambda g: g.stem)
def test_every_quoted_sentence_is_a_real_constant(templates, guide):
    bad = [q for q in quotes(guide) if not matches(q, templates)]
    assert not bad, "the guide quotes what the page does not say:\n" + \
        "\n".join(bad)


def test_the_guide_quotes_steps_3s_own_words():
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
    qs = set(quotes(GUIDE))
    for const in (P.S3_SETTINGS_DESC, P.S3_SETTINGS_DESC_NO_RUN,
                  P.S3_PRESET_LAST_RUN, P.S3_LAST_RUN_NOTE,
                  P.S3_HELP_LINES["start_from_combo"],
                  P.S3_HELP_LINES["chan1_combo"],
                  P.S3_HELP_LINES["chan2_combo"],
                  P.S3_HELP_LINES["diameter_sb"],
                  P.S3_HELP_LINES["rescale_cb"]):
        assert _norm(const) in qs, const


def test_every_visible_help_line_is_quoted():
    """Each control on the page has its help line in the guide."""
    from napari_epicycle.fusion_reconstruction import _step3_pr_text as P
    qs = set(quotes(GUIDE))
    missing = [k for k, v in P.S3_HELP_LINES.items() if _norm(v) not in qs]
    assert not missing, missing


def test_the_shared_set_is_what_step3_reads():
    """The shared words Step 3 shows are found (not an empty set that would
    leave only Step 3's own constants), and Step 4-only words are not."""
    names = shared_names_step3_shows()
    for must in ("SEG_PRESET_EPICYCLE", "SEG_INFERENCE_TITLE",
                 "SEG_SIZE_OFF_NOTE", "PS_FLOWS_OFF_NOTE",
                 "STEP4_ADVANCED_GROUPS"):
        assert must in names, must
    # (R2b: SEG_CELL_WIDTH_PX / SEG_CELL_WIDTH_SUFFIX are Step 3 words now,
    # bound in ``_step3_pr_text`` as S3_CELL_WIDTH_PX / _SUFFIX, so they
    # left this list)
    for step4_only in ("STEP4_HIDDEN_KNOBS", "RECON_HELP_LINES",
                       "STEP4_ADVANCED_LABELS"):
        assert step4_only not in names, step4_only


def test_a_step4_only_sentence_is_not_a_step3_quote(templates):
    """Teeth: Step 4's own no-voxel row warning is not a Step 3 word, so a
    guide quoting it must fail.  (Until R2b this used Step 4's µm line,
    which R2b made Step 3's: it matches now.)"""
    from napari_epicycle.segmentation import _step4_text as T
    assert not matches(T.ROW_WARN_NO_SPACING_SHORT, templates)
    assert matches("\u2248 12 pixels at 0.50 \u00b5m per pixel", templates)


def test_a_made_up_sentence_matches_nothing(templates):
    assert not matches("Press the big red button to segment everything.",
                       templates)
    assert not matches("Slices at once", templates)          # the old word
    assert not matches("u-Segment3D defaults", templates)    # the old word
    assert matches("Custom", templates)
