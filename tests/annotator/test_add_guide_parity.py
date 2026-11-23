"""The Add guide says what the page says (for this dock the guide is the only place a knob's meaning is
written down at length, so drift is silent misinformation).

* every sentence the guide QUOTES (between straight double quotes) is a
  named constant the page or the engine really shows -- ``add_preview``,
  the panel's module constants and help lines, the engine's Add refusals,
  the shipped ``ADD_BATCH_BUSY``, the worker's stage lines -- with
  ``{placeholders}`` filled by example values;
* every row label and button word the Add page shows is named in the guide;
* none of the old guide's false claims or retired words is back.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from .add_widget_fixture import qapp, widget  # noqa: F401

REPO = pathlib.Path(__file__).resolve().parents[2]
GUIDE = REPO / "docs" / "guides" / "annotator_3d_add.md"


def _guide():
    return GUIDE.read_text(encoding="utf-8")


def _norm(s):
    return " ".join(str(s).split())


def _module_strings(mod):
    """The ``ADD_*`` sentences of *mod* only (final fix pass G3: a Refine or
    Reshape sentence must not be able to satisfy an Add quote)."""
    return [v for k, v in vars(mod).items()
            if k.startswith("ADD_") and isinstance(v, str) and v.strip()]


#: non-``ADD_`` names the Add page really shows, allowed by name: the walls
#: tick's legend on the Nuclei preset (``add_preview``)
_ALLOWED_OTHER = ("WALLS_TOGGLE_NO_WALLS",)


def _sentences():
    from napari_epicycle.annotator import advanced_features_widget as AFW
    from napari_epicycle.annotator.reshape_worker import STAGE_TEXT
    from packages.epicycle_annotator import add_preview as AP
    from packages.epicycle_annotator.batch_refine import ADD_BATCH_BUSY
    from packages.epicycle_annotator.reshape import region as R
    # the refusal every door shares when an edit is open; the Add tab says it too
    from napari_epicycle.annotator._staged_op_text import STAGED_WORK_OPEN

    out = [STAGED_WORK_OPEN]
    out += _module_strings(AP)
    out += [v for _k, many, one in AP._COUNT_PARTS for v in (many, one)]
    out += _module_strings(AFW)
    out += list(AFW.ADD_HELP_LINES.values())
    out += [v for k, v in vars(R).items()
            if k.startswith("ADD_") and isinstance(v, str)]
    out += [ADD_BATCH_BUSY]
    out += list(STAGE_TEXT.values())          # the busy strip's stage line
    # Inside each cell's stage lines come from its recipe, word for word
    from packages.epicycle_annotator import add_derive as AD

    out += list(AD.DERIVE_STAGES)
    out += [AD.DERIVE_LOOP_CELLS_WITH, AD.DERIVE_LOOP_EMPTY_CELLS,
            AD.DERIVE_LOOP_NUCLEI]
    # S1's found / added lines with the two-nuclei tail
    for head in (AP.ADD_S1_FOUND, AP.ADD_S1_FOUND_ONE):
        for tail in (AP.ADD_S1_TWO_TAIL, AP.ADD_S1_TWO_TAIL_ONE):
            out.append(head + AP.ADD_NOTE_COUNTS_SEP + tail)
            # the two-nuclei tail comes LAST, after the unticked one
            for mid in (AP.ADD_S1_UNTICKED, AP.ADD_S1_UNTICKED_PLAIN):
                out.append(head + mid + AP.ADD_NOTE_COUNTS_SEP + tail)
    for head in (AP.ADD_S1_ADDED, AP.ADD_S1_ADDED_ONE):
        for tail in (AP.ADD_S1_ADDED_TWO_TAIL, AP.ADD_S1_ADDED_TWO_TAIL_ONE):
            out.append(head + AP.ADD_NOTE_COUNTS_SEP + tail)
    # S1 with nothing ticked: the count, the two-nuclei tail if any, then
    # the none-ticked tail that names All
    for head, none in ((AP.ADD_S1_FOUND, AP.ADD_S1_NONE_TICKED),
                       (AP.ADD_S1_FOUND_ONE, AP.ADD_S1_NONE_TICKED_ONE)):
        out.append(head + none)
        for tail in (AP.ADD_S1_TWO_TAIL, AP.ADD_S1_TWO_TAIL_ONE):
            out.append(head + AP.ADD_NOTE_COUNTS_SEP + tail + none)
    out += [getattr(mod, name) for mod in (AP, AFW, R)
            for name in _ALLOWED_OTHER if hasattr(mod, name)]
    # S1's two composed lines: the found count + its unticked tail, and
    # REVIEW's count + what Refine needs (two sentences, live-shot F5)
    for head in (AP.ADD_S1_FOUND, AP.ADD_S1_FOUND_ONE):
        for tail in (AP.ADD_S1_UNTICKED, AP.ADD_S1_UNTICKED_PLAIN):
            out.append(head + tail)
    # the ready clicks line + its ticked tail
    for head in (AP.ADD_S1_BOX_CLICKS, AP.ADD_S1_BOX_CLICKS_ONE):
        out.append(head + AP.ADD_S1_TICKED_TAIL)
    for head in (AP.ADD_S1_ADDED, AP.ADD_S1_ADDED_ONE):
        for tail in (AP.ADD_S1_SELECT_ONE, AP.ADD_S1_SELECT_JUST_ONE):
            out.append(head + "." + tail)
    return [_norm(s) for s in out]


def _template(s):
    """A constant as a regex: ``{...}`` placeholders match example values."""
    parts = re.split(r"\{[^{}]*\}", s)
    return re.compile("^" + ".+?".join(re.escape(p) for p in parts) + "$")


@pytest.fixture(scope="module")
def templates():
    return [_template(s) for s in _sentences()]


def _quotes():
    return [_norm(q) for q in re.findall(r'"([^"]+)"', _guide())]


def _matches(q, templates):
    if any(t.match(q) for t in templates):
        return True
    # a counts note is its parts joined by " · "
    parts = q.split(" · ")
    return len(parts) > 1 and all(
        any(t.match(p) for t in templates) for p in parts)


def test_the_guide_quotes_something():
    assert len(_quotes()) >= 25


def test_every_quoted_sentence_is_a_real_constant(templates):
    bad = [q for q in _quotes() if not _matches(q, templates)]
    assert not bad, "quoted but not on the page:\n" + "\n".join(bad)


def test_a_refine_sentence_cannot_satisfy_an_add_quote(templates):
    """G3: only ``ADD_*`` names (+ the short allow-list) are templates."""
    from napari_epicycle.annotator import advanced_features_widget as AFW

    add = set(_sentences())
    others = [_norm(v) for k, v in vars(AFW).items()
              if k.startswith(("REFINE_", "SPLIT_", "WALLS_"))
              and isinstance(v, str) and v.strip() and "{" not in v]
    assert others
    for s in others:
        if s in add:
            continue
        assert not _matches(s, templates), s


def test_the_template_matcher_is_not_vacuous(templates):
    assert _matches("Box 1 × 2 × 3 voxels (0.1 million) · 5 % of it is "
                    "empty space", templates)
    assert not _matches("Drag the red line until the cell lights up.",
                        templates)


# --------------------------------------------------------------------------- #
# every word the page shows is named in the guide
# --------------------------------------------------------------------------- #
#: shown but not a control the guide has to name: the page's own title, the
#: hidden "Size" label (live-shot V8), the never-placed Merge button, a
#: live count and the chevron glyphs
_NOT_NAMED = {"add objects", "size", "merge selected", "0 placed", "x",
              "▸", "▾"}


def _page_words(widget):
    from qtpy.QtWidgets import QAbstractButton, QLabel

    root = widget._panel_slots["Add"]["page"]
    words = set()
    for cls in (QLabel, QAbstractButton):
        for w in root.findChildren(cls):
            t = _norm(w.text())
            if not t or len(t) > 40 or "<" in t:
                continue
            if t.startswith(("▸", "▾")):      # a chevron's elided summary
                continue
            if t.lower() in _NOT_NAMED:
                continue
            words.add(t)
    return words


def test_every_label_on_the_page_is_named_in_the_guide(widget):
    guide = _norm(_guide())
    low = guide.lower()
    missing = []
    for t in sorted(_page_words(widget)):
        if len(t) == 1:                       # S / M / L
            if "**{}**".format(t) not in guide:
                missing.append(t)
        elif t.lower() not in low:
            missing.append(t)
    assert not missing, "the guide never names: {}".format(missing)


def test_every_wall_type_and_preset_is_named(widget):
    guide = _norm(_guide())
    for combo in ("add_wall_kind_combo", "add_preset_combo"):
        c = getattr(widget, combo)
        for i in range(c.count()):
            assert c.itemText(i) in guide, c.itemText(i)


def test_the_marker_items_are_named():
    from packages.epicycle_annotator import add_preview as AP

    guide = _norm(_guide())
    for item in (AP.ADD_ITEM_NUCLEI, AP.ADD_ITEM_NUCLEI_GREY,
                 AP.ADD_ITEM_CLICKS, AP.ADD_ITEM_NO_RUN,
                 AP.ADD_ITEM_RAW_SUFFIX.strip()):
        assert item in guide, item
    assert AP.ADD_ITEM_RUN.split("{")[0].strip() in guide


def test_the_look_for_items_and_the_new_rows_are_named():
    """Look for nuclei's items, the Whole patch buttons, the two chips and
    the three Inside rows of Advanced -- every word the page shows for
    them."""
    from packages.epicycle_annotator import add_preview as AP

    guide = _norm(_guide())
    for item in (AP.ADD_LOOK_FOR_LABEL, AP.ADD_LOOK_ANYWHERE,
                 AP.ADD_LOOK_NO_RUN, AP.ADD_LOOK_INSIDE.split("{")[0].strip(),
                 AP.ADD_WHOLE_BTN, AP.ADD_CHIP_TWO, AP.ADD_CHIP_CUT,
                 "Nucleus radius", "Nucleus edge",
                 "Gap that splits two nuclei"):
        assert item in guide, item


def test_the_cells_items_and_rows_are_named():
    """Tissue outline's and Grow by's items (the grey forms too), Show the
    tissue outline and the Advanced rows of Cells -- every word the page
    shows for them."""
    from packages.epicycle_annotator import add_preview as AP

    guide = _norm(_guide())
    for item in (AP.ADD_OUTLINE_LABEL, AP.ADD_OUTLINE_CLEARED,
                 AP.ADD_OUTLINE_CLEARED_GREY, AP.ADD_OUTLINE_SHRINK,
                 AP.ADD_OUTLINE_NONE, AP.ADD_SHOW_OUTLINE,
                 AP.ADD_GROW_LABEL, AP.ADD_GROW_FLOOD,
                 AP.ADD_GROW_SPREAD,
                 AP.ADD_GROW_SPREAD_GREY, AP.ADD_WALL_COST_LABEL,
                 AP.ADD_GROWTH_LIMIT_LABEL, AP.ADD_INCLUDE_MEMBRANE,
                 AP.ADD_BRIDGE_LABEL):
        assert item in guide, item
    for section in ("## How cells grow", "## The tissue outline",
                    "### Cells: the whole patch"):
        assert section in _guide(), section


def test_the_guide_names_no_banned_word():
    body = _guide().lower()
    for word in ("envelope", "geodesic", "provenance", " proxy", " roi ",
                 " lut "):
        assert word not in body, word


# --------------------------------------------------------------------------- #
# the old guide's false claims stay gone
# --------------------------------------------------------------------------- #
def _body_without_what_changed():
    return _norm(_guide().split("## What changed", 1)[0])


@pytest.mark.parametrize("phrase", [
    "centre of the view",            # (1) it drops at the crosshair
    "drag the rectangle",            # (2) the OUTLINE is dragged, on its line
    "Connectivity",                  # (3) collected, never passed on
    "Signal",                        # the "Boundary -- soon" stub
    "Boundary",
    "lights up",                     # nothing lights up
    "on the way in",                 # (7) smoothing runs at Preview
    "Accept / Cancel",               # (6) the second button changes
    "Place Box",                     # the old button word
    "Enhance contrast",
    "CLAHE",
    "Seed spacing",
    "Min size",
    "Uncheck cells",
    "Smooth added cells",
    "Recenter on crosshair",
    # the tissue outline is drawn in the side views too now
    "main view only",
    "not drawn in the three side views",
    # a Cells whole patch is filled to the walls in one go, not in pieces
    "Cells are filled in pieces",
    "filled in one piece",
    "Piece 2 of 5",
    "A whole patch always spreads",
    "Stop works between pieces",
    "Fill to the walls (boxes only)",
    "every setting of Spread is in microns",
    "about 5 µm",
    # markers are placed on a whole patch too, and From lists Click markers
    "places nothing",
    "Boxes only",
    "lists only the other runs",
])
def test_a_retired_claim_is_not_back(phrase):
    assert phrase not in _body_without_what_changed()


def test_the_corrected_claims_are_there():
    from packages.epicycle_annotator import add_preview as AP

    body = _body_without_what_changed()
    assert "where the crosshair is" in body                      # (1)
    assert "Drag the outline" in body                            # (2)
    assert "voxels (0.6 million)" in body                        # (4)
    assert "just above the two buttons at the bottom" in body    # (5)
    for word in ("**Discard**", "**Close box**", "**Done**",    # (6)
                 "**Undo all**"):
        assert word in body
    assert "in the preview" in body                              # (7)
    # the outline in the side views: the claim itself must be there, not
    # only the old claim absent
    assert "in the main view and in the three side views" in body
    assert _norm(AP.ADD_HELP_SHOW_OUTLINE) in body
    assert _norm(AP.ADD_OUTLINE_DRAWING) in body
    # a slice along the tissue's top or bottom tints the whole tissue
    assert "the whole tissue there takes a cyan tint" in body


def test_the_no_clahe_sentence_lives_here():
    from napari_epicycle.annotator.advanced_features_widget import (
        REFINE_NO_CLAHE_SENTENCE,
    )

    assert REFINE_NO_CLAHE_SENTENCE.rstrip(".") in _norm(_guide())


def test_the_voxel_size_is_sent_to_step_3_not_step_1():
    body = _body_without_what_changed()
    assert "QC Features → Settings… in Step 3" in body
    assert "Set the voxel size in Step 1" not in body


def test_no_code_identifier_in_the_prose():
    """Biologist voice: no ``snake_case`` or backticked code words."""
    text = _guide()
    assert "`" not in text
    idents = re.findall(r"\b[a-z]+_[a-z_]+\b", text)
    assert not idents, idents


def test_the_marker_list_is_explained():
    """The list, the gap rule and the refusals."""
    from napari_epicycle.annotator.advanced_features_widget import (
        ADD_HELP_LINES,
    )

    guide = _norm(_guide())
    assert _norm(ADD_HELP_LINES["add_click_list"]) in guide
    assert ("Unticking a marker leaves a gap in the numbering; the other "
            "colours do not change.") in guide
    assert "A double click places two markers" not in guide
    for q in ("Tick at least one object to grow from",
              "That spot is already inside marker"):
        assert q in guide, q


def test_the_clear_menu_items_and_confirm_buttons_are_named(widget,
                                                            monkeypatch):
    """The menu's two items (QActions) and the confirm's two buttons are not
    on the page, so the page scan above never sees them: pinned here."""
    import napari_epicycle.ui_shell.theme as TH
    from napari_epicycle.annotator.curation_3d_widget import (
        Curation3DAnnotator,
    )
    from packages.epicycle_annotator import add_preview as AP

    guide = _norm(_guide())
    items = [a.text() for a in widget.add_clear_menu.actions()]
    assert items == [AP.ADD_CLEAR_ITEM_BOX, AP.ADD_CLEAR_ITEM_PATCH]
    seen = {}

    def _fake_confirm(parent, title, text, **kw):
        seen.update(kw, title=title)
        return False

    monkeypatch.setattr(TH, "themed_confirm", _fake_confirm)
    assert Curation3DAnnotator._add_clear_confirm(object(), "text") is False
    assert seen["title"] == AP.ADD_CLEAR_CONFIRM_TITLE
    assert seen["confirm_text"] == AP.ADD_CLEAR_CONFIRM_OK
    assert seen["cancel_text"] == AP.ADD_CLEAR_CONFIRM_BACK
    assert seen["accent"] == "red"
    for word in (AP.ADD_CLEAR_ITEM_BOX, AP.ADD_CLEAR_ITEM_PATCH,
                 AP.ADD_CLEAR_CONFIRM_OK, AP.ADD_CLEAR_CONFIRM_BACK):
        assert "**{}**".format(word) in guide, word
    assert "titled *{}*".format(AP.ADD_CLEAR_CONFIRM_TITLE) in guide


def test_the_whole_patch_fill_is_described():
    """A Cells whole patch fills to the walls with a background marker: the
    new sentences are quoted, the growth limit's default and what it means
    in very dark cytoplasm are said, and the walls picture is one slice."""
    from packages.epicycle_annotator import add_preview as AP

    body = _body_without_what_changed()
    for text in (AP.ADD_HELP_WHOLE_FILL, AP.ADD_AMBER_ONE_CHANNEL_WHOLE,
                 AP.ADD_HELP_GROWTH_LIMIT, AP.ADD_HELP_GROW,
                 AP.ADD_WHOLE_TIP, AP.ADD_NOTE_DARK_OFF_LEAK_WHOLE,
                 AP.ADD_S1_WARMING, AP.ADD_WHOLE_NEEDS_NUMBA,
                 AP.ADD_WHOLE_NEEDS_NUMBA_CACHE,
                 AP.whole_too_big_line(150_000_000),
                 AP.ADD_S1_PARKED_REFUSED,
                 AP.ADD_SUM_ADVANCED_FILL,
                 AP.ADD_SUM_ADVANCED_FILL_NO_MEMBRANE):
        assert _norm(text) in body, text
    assert ("In very dark cytoplasm the same limit lets a cell go further, "
            "about 10 µm.") in body
    assert "28 by default" in body
    assert "at the depth of the XY view" in body
    # the size limit the guide names is the fill's own ceiling
    from packages.epicycle_annotator.reshape import region as R

    limit = "{:g}".format(R.WHOLE_FILL_MAX_VOXELS / 1_000_000)
    assert "A whole patch can be filled up to {} million voxels".format(
        limit) in body
    # the stage line Stop is said to act during, as the strip shows it
    assert "while the grey line says *\"Finding markers...\"*" in body
    # the parked refusal no longer claims what a reopened annotator undoes
    assert "two fills never run at once" not in body
    assert "Stop, honestly" not in body
    assert "Tissue far from every marker stays empty." in body
    assert "Tissue with no marker stays empty" not in body
