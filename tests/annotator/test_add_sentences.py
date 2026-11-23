"""Every Add count sentence at 1 and at 2.

Pure pytest -- no Qt, no napari.
"""

from __future__ import annotations

import pytest

from packages.epicycle_annotator import add_preview as AP


def test_count_phrase():
    assert AP.count_phrase(1, "1 thing", "{n} things") == "1 thing"
    assert AP.count_phrase(2, "1 thing", "{n} things") == "2 things"


@pytest.mark.parametrize("n,want", [
    (1, "Box 2 × 3 × 4 voxels · 1 object marked"),
    (2, "Box 2 × 3 × 4 voxels · 2 objects marked"),
])
def test_box_clicks(n, want):
    assert AP.box_line((2, 3, 4), preset="cells", n_clicks=n) == want


def test_box_lines_and_millions():
    assert AP.box_line((100, 100, 100), preset="cells", empty_share=0.42) == (
        "Box 100 × 100 × 100 voxels (1.0 million; the limit for cells is 4 "
        "million) · 42 % of it is empty space")
    assert AP.box_line((10, 10, 10), preset="nuclei", empty_share=1.0) == (
        "Box 10 × 10 × 10 voxels (0.1 million) · 100 % of it is empty space")
    # at or under the Cells limit {m} rounds DOWN; above it, UP -- a box
    # over the limit never reads as the limit (B2 fix pass LOW)
    assert AP.millions(3_999_999) == "3.9"
    assert AP.millions(4_999_999) == "5.0"
    assert AP.over_limit_line((200, 200, 200)).startswith(
        "This box is 8.0 million voxels; the limit for cells is 4 million.")


@pytest.mark.parametrize("n,want", [
    (1, "1 marker was outside the moved box and was removed."),
    (2, "2 markers were outside the moved box and were removed."),
])
def test_clicks_dropped(n, want):
    assert AP.count_phrase(n, AP.ADD_S1_CLICKS_DROPPED_ONE,
                           AP.ADD_S1_CLICKS_DROPPED) == want


def test_found_and_unticked():
    assert AP.found_line(1) == "1 new object found"
    assert AP.found_line(2) == "2 new objects found"
    assert AP.found_line(5, k_edge=2) == (
        "5 new objects found · 2 left unticked (at the box edge)")
    assert AP.found_line(5, k_small=1) == (
        "5 new objects found · 1 left unticked (small)")
    assert AP.found_line(5, k_edge=2, k_small=2, k_both=1) == (
        "5 new objects found · 3 left unticked (at the box edge or small)")


def test_added():
    assert AP.added_line(1, 1234) == "Added 1 object · 1,234 voxels"
    assert AP.added_line(2, 5) == "Added 2 objects · 5 voxels"
    assert AP.added_line(2, 5, has_selection=False).endswith(
        " Select one object in the list to refine it.")


def test_run_count():
    assert AP.run_count_line(43, 14) == (
        "43 in the box · 14 over empty space become markers.")
    assert AP.run_count_line(1, 1) == (
        "1 in the box · 1 over empty space becomes a marker.")


@pytest.mark.parametrize("n", [1, 2])
def test_every_counts_part(n):
    counts = {k: 0 for k, _m, _o in AP._COUNT_PARTS}
    for key, many, one in AP._COUNT_PARTS:
        c = dict(counts, **{key: n})
        text = AP.counts_note(c)
        assert text == (one if n == 1 else many.format(n=n))


def test_counts_note_joins_and_omits_over_kept_for_run_cores():
    c = {"n_dropped_dark": 2, "n_markers_over_kept": 5, "n_dropped_thin": 1}
    assert AP.counts_note(c, source="nuclei_masks") == (
        "2 markers fell in dark space and were skipped · 1 run object the box "
        "edge cut too thin to use · 5 markers sat on kept objects")
    assert "kept objects" not in AP.counts_note(c, source="labels")
    assert AP.counts_note({}) == ""


@pytest.mark.parametrize("n", [1, 2])
def test_confirm_close_and_footer_singulars(n):
    pairs = [
        (AP.ADD_CONFIRM_UNDO_ALL_ONE, AP.ADD_CONFIRM_UNDO_ALL),
        (AP.ADD_FOOT_BOX_SESSION_ONE, AP.ADD_FOOT_BOX_SESSION),
        (AP.ADD_CLOSE_CANDIDATES_ONE, AP.ADD_CLOSE_CANDIDATES),
        (AP.ADD_CLOSE_CLICKS_ONE, AP.ADD_CLOSE_CLICKS),
    ]
    for one, many in pairs:
        text = AP.count_phrase(n, one, many)
        assert "{" not in text
        if n == 1:
            assert text == one
        else:
            assert str(n) in text
    assert AP.ADD_UNDO_PARTIAL.format(k=1, n=2).startswith("Took back 1 of 2")
    assert AP.ADD_UNDO_PARTIAL_ONE.format(k=0).startswith(
        "Took back 0 of 1 added object ")


def test_help_sentences_with_numbers():
    assert "about 4 µm across" in AP.help_clicks(2.0)
    assert "shrunk by 1 µm" in AP.help_run(1.0)
    assert "shrunk by 1.5 µm" in AP.help_run(1.5)


@pytest.mark.parametrize("n,want", [
    (1, "1 cell of that run in the box."),
    (2, "2 cells of that run in the box."),
    (None, "Counting that run's cells in the box…"),
])
def test_the_inside_count(n, want):
    assert AP.inside_count_line(n) == want


@pytest.mark.parametrize("s,found,added", [
    (1, "1 cell holds two nuclei",
     "1 cell holds two nuclei — after Save, run Extract and pick that run "
     "under Compare with to find it."),
    (2, "2 cells hold two nuclei",
     "2 cells hold two nuclei — after Save, run Extract and pick that run "
     "under Compare with to find them."),
])
def test_the_two_nuclei_tails(s, found, added):
    assert AP.two_tail(s) == found
    assert AP.found_line(4, n_two=s) == "4 new objects found · " + found
    assert AP.added_line(4, 10, n_two=s) == (
        "Added 4 objects · 10 voxels · " + added)
    assert AP.two_tail(0) == ""
    assert AP.found_line(4, n_two=0) == "4 new objects found"


def test_the_two_nuclei_tail_comes_last():
    """The mockup's order: the count, what was left unticked, then the cells
    holding two nuclei."""
    assert AP.found_line(4, n_two=1) == (
        "4 new objects found · 1 cell holds two nuclei")
    assert AP.found_line(46, k_edge=1, k_small=1, n_two=3) == (
        "46 new objects found · 2 left unticked (at the box edge or small) · "
        "3 cells hold two nuclei")
    assert AP.found_line(5, k_plain=2, n_two=2) == (
        "5 new objects found · 2 left unticked · 2 cells hold two nuclei")
    assert AP.found_line(5, k_plain=0, n_two=2) == (
        "5 new objects found · 2 cells hold two nuclei")


def test_none_ticked_order():
    """No row ticked (every one started unticked): the count, the cells
    holding two nuclei, then the none-ticked tail that names All -- the
    reverse of the found line's order, so the instruction ends the line."""
    assert AP.ADD_S1_NONE_TICKED == (
        " · none ticked ({why}) — press All, or tick the ones to add")
    assert AP.ADD_S1_NONE_TICKED_ONE == (
        " · not ticked ({why}) — press All, or tick it")
    assert AP.ADD_S1_TICK_ONE == (
        "None is ticked — tick the objects to add, or press All.")
    assert AP.found_none_ticked_line(12, AP.ADD_WHY_EDGE) == (
        "12 new objects found · none ticked (at the box edge) — press All, "
        "or tick the ones to add")
    assert AP.found_none_ticked_line(1, AP.ADD_WHY_SMALL) == (
        "1 new object found · not ticked (small) — press All, or tick it")
    assert AP.found_none_ticked_line(16, AP.ADD_WHY_BOTH, 3) == (
        "16 new objects found · 3 cells hold two nuclei · none ticked (at "
        "the box edge or small) — press All, or tick the ones to add")
    assert AP.found_none_ticked_line(4, AP.ADD_WHY_EDGE, n_two=1) == (
        "4 new objects found · 1 cell holds two nuclei · none ticked (at "
        "the box edge) — press All, or tick the ones to add")


def test_the_faint_alpha_is_one_named_number():
    assert AP.ADD_UNTICKED_ALPHA == 0.4


@pytest.mark.parametrize("c,box,patch", [
    (None, "Counting that run's cells in the box…",
     "Counting that run's cells in the patch…"),
    (1, "1 cell of that run in the box.", "1 cell of that run in the patch."),
    (312, "312 cells of that run in the box.",
     "312 cells of that run in the patch."),
])
def test_the_inside_count_says_patch_on_the_whole_patch(c, box, patch):
    assert AP.inside_count_line(c) == box
    assert AP.inside_count_line(c, whole_patch=True) == patch


def test_no_cell_in_the_whole_patch_asks_for_another_run_only():
    s = dict(has_patch=True, has_box=True, preset="nuclei", channel="nuc",
             look_for=AP.LOOK_INSIDE, inside_count=0, box_shape=(4, 4, 4))
    assert AP.preview_block(s) == AP.ADD_S1_INSIDE_NO_CELLS
    assert AP.preview_block(dict(s, whole_patch=True)) == (
        AP.ADD_S1_INSIDE_NO_CELLS_PATCH)
    assert "move the box" not in AP.ADD_S1_INSIDE_NO_CELLS_PATCH


def test_the_words_after_the_page_fix_pass():
    assert AP.ADD_CHIP_CUT == "cut at edge"
    assert "cell edge" not in AP.ADD_CHIP_CUT
    assert AP.ADD_HELP_NUCLEI == AP.ADD_PRESET_TIPS[0]
    assert "alone" not in AP.ADD_HELP_NUCLEI
    assert "inside each cell of another run" in AP.ADD_HELP_NUCLEI
    assert "inside each cell of another run" in AP.ADD_ABOUT_BODY
    # Cells has Whole patch too (filled to the walls in one go)
    assert AP.ADD_WHOLE_TIP == "Use the whole patch as the box."
    assert AP.ADD_S1_INSIDE_NOTHING.endswith(
        "check the channel, and Nucleus radius under Advanced.")
    for tail in (AP.ADD_S1_ADDED_TWO_TAIL, AP.ADD_S1_ADDED_TWO_TAIL_ONE):
        assert "run Extract and pick that run under Compare with" in tail
        assert "Extract with Compare with" not in tail
    assert "New Run" in AP.ADD_HELP_INSIDE_NO_RUN


def test_the_look_inside_items():
    assert AP.look_inside_text("default", "cells") == (
        "Inside each cell of run ▸ default · cells")
    assert AP.look_inside_text("default", "") == (
        "Inside each cell of run ▸ default")
    assert AP.look_inside_text("r", "cells", curated=False) == (
        "Inside each cell of run ▸ r · cells (reconstruction, not curated)")
    assert AP.target_is_nuclei("Nuclei") and not AP.target_is_nuclei("cells")


@pytest.mark.parametrize("n", [1, 2])
def test_the_whole_patch_sentences(n):
    m = AP.millions(n * 21_000_000)
    assert AP.patch_slow_line(n * 21_000_000).startswith(
        "This patch is {} million voxels".format(m))
    assert AP.ADD_AMBER_BOX_MOSTLY_FULL.format(p=n).startswith(
        "Only {} % of the box is empty".format(n))


def test_the_inside_sentences_name_no_engine_word():
    banned = ("envelope", "geodesic", "provenance", "reach", "nuclei_masks",
              "look_for", "otsu", "Otsu")
    names = [k for k in vars(AP) if k.startswith((
        "ADD_LOOK", "ADD_HELP_INSIDE", "ADD_INSIDE", "ADD_S1_INSIDE",
        "ADD_S1_TWO", "ADD_S1_ADDED_TWO", "ADD_CHIP", "ADD_TIP",
        "ADD_WALLS_INSIDE", "ADD_NOTE_CELL", "ADD_NOTE_NO_NUCLEUS",
        "ADD_NOTE_SPLIT", "ADD_NOTE_OUTSIDE_CELLS", "ADD_HELP_NUC",
        "ADD_HELP_TWO", "ADD_WHOLE", "ADD_HELP_WHOLE", "ADD_S1_PATCH",
        "ADD_NOTE_PATCH_SLOW", "ADD_NOTE_HALFSIZE", "ADD_S1_NEEDS",
        "ADD_AMBER_BOX", "ADD_S1_GATHERING", "ADD_S1_IDLE",
        "ADD_AMBER_ONE_CHANNEL_WHOLE", "ADD_HELP_GROWTH_LIMIT",
        "ADD_S1_WHOLE_TOO_BIG", "ADD_SUM_ADVANCED_FILL",
        "ADD_S1_PARKED_REFUSED", "ADD_S1_WHOLE",
        "ADD_S1_RUN_NO_MARKERS", "ADD_NOTE_RIDGES"))]
    assert len(names) >= 30
    for k in names:
        v = getattr(AP, k)
        if isinstance(v, str):
            for word in banned:
                assert word not in v, (k, word)


def test_the_dropped_notes_do_not_exist():
    """Decision 21: no faint-wall note, no unfilled note in S1."""
    assert not hasattr(AP, "ADD_NOTE_FAINT")
    assert not hasattr(AP, "ADD_NOTE_UNFILLED")


def test_no_sentence_has_a_stray_brace():
    for name in dir(AP):
        if name.startswith("ADD_") and isinstance(getattr(AP, name), str):
            text = getattr(AP, name)
            assert text.count("{") == text.count("}"), name


# --------------------------------------------------------------------------- #
# live-shot fix F1 -- the spacing sentence names where it is really set
# (the two refusals retired, the amber note took their place)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("const", ["ADD_NOTE_DEFAULT_SPACING",
                                   "ADD_NOTE_DEFAULT_SPACING_BOXED",
                                   "ADD_S1_NEEDS_SPACING",
                                   "ADD_S1_NEEDS_SPACING_BOXED"])
def test_the_voxel_sentences_point_at_set_spacing(const):
    """The sentences send the user to Set spacing… on the Session list: it
    keeps a voxel size per patch (1 / 1 / 1 too, which an OK of QC Settings
    cannot tell from its factory default -- a dead end for a true 1.0 µm
    voxel), and it stays live while an Add box holds the Step 3 lock."""
    text = getattr(AP, const)
    assert "Step 1" not in text, text
    assert "QC Features" not in text, text
    assert "Set spacing… in the Session list" in text, text


# --------------------------------------------------------------------------- #
# A5 -- the click marker list's words
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n,k,want", [
    (7, 5, "Box 2 × 3 × 4 voxels · 7 objects marked · 5 of 7 ticked"),
    (2, 1, "Box 2 × 3 × 4 voxels · 2 objects marked · 1 of 2 ticked"),
    (3, 3, "Box 2 × 3 × 4 voxels · 3 objects marked"),
    (1, 1, "Box 2 × 3 × 4 voxels · 1 object marked"),
])
def test_box_clicks_with_the_ticked_tail(n, k, want):
    assert AP.box_line((2, 3, 4), preset="cells", n_clicks=n,
                       n_ticked=k) == want


@pytest.mark.parametrize("n,want", [
    (1, "Tick the object to grow from — it is unticked."),
    (2, "Tick at least one object to grow from — all 2 are unticked."),
])
def test_tick_one_marker(n, want):
    assert AP.tick_one_marker_line(n) == want


def test_same_spot_names_the_marker():
    assert AP.ADD_S1_CLICK_SAME_SPOT.format(id=412) == (
        "That spot is already inside marker 412 — click a little further "
        "away.")


def test_the_count_line_and_summary_say_how_many_are_ticked():
    assert AP.click_count_text(7) == "7 placed"
    assert AP.click_count_text(7, n_ticked=7) == "7 placed"
    assert AP.click_count_text(7, n_ticked=5) == "7 placed · 5 ticked"
    assert AP.markers_summary("paint", n_clicks=7, n_ticked=5) == (
        "Markers: clicks · 7 placed · 5 ticked")
    assert AP.markers_summary("paint", n_clicks=7) == "Markers: clicks · 7 placed"


def test_ball_values_follow_napari_s_49_colour_cycle():
    assert [AP.ball_value(i) for i in (1, 2, 49, 50, 97, 412, 65535)] == [
        1, 2, 49, 1, 48, 20, 22]
    assert all(1 <= AP.ball_value(i) <= 49 for i in range(1, 500))


def test_a_ball_value_draws_the_colour_of_its_id():
    """Invariant 23: ball (uint8, cyclic value), candidate (the id itself,
    napari's default colormap) and the row swatch agree."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("napari")
    from napari.layers import Labels
    from napari.utils.colormaps import label_colormap

    cand = Labels(np.zeros((2, 2, 2), np.int32))
    ball = Labels(np.zeros((2, 2, 2), np.uint8))
    swatch = label_colormap(49, seed=0.5)
    for i in (1, 2, 48, 49, 50, 97, 412, 65535):
        c = cand.colormap.map(np.array([i]))[0]
        b = ball.colormap.map(np.array([AP.ball_value(i)], np.uint8))[0]
        s = swatch.map(np.array([i], np.intp))[0]
        assert np.allclose(c, b) and np.allclose(c, s), i


def test_a_marker_row_reads_its_id_and_position():
    assert AP.marker_row_text(412, (12, 20, 305)) == (
        "Object 412  ·  12 / 20 / 305")


def test_the_marker_list_is_three_rows():
    assert AP.ADD_MARKER_LIST_ROWS == 3
    assert AP.ADD_LIST_ROWS == 4


# --------------------------------------------------------------------------- #
# Clear objects -- every count sentence at 1 and at 2
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [1, 2])
def test_clear_count_sentences(n):
    pairs = [
        (AP.ADD_CLEAR_CONFIRM_BOX_ONE, AP.ADD_CLEAR_CONFIRM_BOX, {}),
        (AP.ADD_CLEAR_CONFIRM_PATCH_ONE, AP.ADD_CLEAR_CONFIRM_PATCH, {}),
        (AP.ADD_CLEAR_CONFIRM_OUTSIDE_ONE, AP.ADD_CLEAR_CONFIRM_OUTSIDE,
         {"k": n}),
        (AP.ADD_CLEAR_CONFIRM_GOOD_ONE, AP.ADD_CLEAR_CONFIRM_GOOD, {"g": n}),
        (AP.ADD_S1_CLEARING_ONE, AP.ADD_S1_CLEARING, {}),
        (AP.ADD_S1_CLEARED_ONE, AP.ADD_S1_CLEARED, {}),
        (AP.ADD_CLEAR_UNDO_TEXT_ONE, AP.ADD_CLEAR_UNDO_TEXT, {}),
    ]
    for one, many, kw in pairs:
        text = AP.count_phrase(n, one, many, **kw)
        assert "{" not in text
        if n == 1:
            assert text == one
            assert "objects" not in text.split("of them")[0]
        else:
            assert str(n) in text
    assert AP.count_phrase(2, AP.ADD_CLEAR_CONFIRM_BOX_ONE,
                           AP.ADD_CLEAR_CONFIRM_BOX) == (
        "Clear the 2 objects that sit mostly inside this box?")
    assert AP.count_phrase(1, AP.ADD_S1_CLEARED_ONE, AP.ADD_S1_CLEARED) == (
        "Cleared 1 object. The main Undo button brings it back once this "
        "box is closed.")


def test_the_clear_sentences_name_no_engine_word():
    for name in dir(AP):
        if name.startswith(("ADD_CLEAR", "ADD_S1_CLEAR")) or name in (
                "ADD_FOOT_BOX_CLEARED", "ADD_UNDO_DONE_CLEARED"):
            text = getattr(AP, name).lower()
            for word in ("envelope", "geodesic", "provenance", "reach",
                         "label", "mask"):
                assert word not in text, (name, word)


def test_the_clear_confirm_of_one_object_says_it():
    """n == 1: "it", never "1 of them" -- the confirm's own singulars."""
    for name in ("ADD_CLEAR_CONFIRM_BOX_ONE", "ADD_CLEAR_CONFIRM_PATCH_ONE",
                 "ADD_CLEAR_CONFIRM_OUTSIDE_IT", "ADD_CLEAR_CONFIRM_GOOD_IT",
                 "ADD_CLEAR_CONFIRM_UNDO_IT", "ADD_S1_CLEARED_OUTSIDE_IT"):
        text = getattr(AP, name)
        assert "{" not in text and "1" not in text, name
        assert "of them" not in text and "them" not in text, name
    assert AP.ADD_CLEAR_CONFIRM_GOOD.format(g=2) == (
        "2 of them are marked Curated or Good.")
    assert "main Undo button" in AP.ADD_CLEAR_CONFIRM_UNDO
    assert "main Undo button" in AP.ADD_CLEAR_TIP
    assert "once the box is closed" in AP.ADD_CLEAR_TIP
    assert "Save keeps" not in AP.ADD_CLEAR_CONFIRM_UNDO


def test_the_whole_patch_fill_sentences_word_for_word():
    """A Cells whole patch fills to the walls with a background marker: its
    words, and the growth words that changed with it."""
    assert AP.ADD_HELP_WHOLE_FILL == (
        "A whole patch is filled to the walls from every marker at once. "
        "Tissue far from every marker stays empty; to fill it, click markers "
        "there too.")
    # the same slot on a whole patch filled as you click
    assert AP.ADD_HELP_WHOLE_FILL_LIVE == (
        "Filled as you click, the way a box is: tissue far from every marker "
        "fills too. To keep it empty, untick Update as I click and press "
        "Preview.")
    assert AP.ADD_SUM_ADVANCED_FILL_LIVE == "filled to the walls"
    assert AP.advanced_fill_summary(False, live=True) == "filled to the walls"
    assert AP.ADD_NOTE_PATCH_SLOW_LIVE_CELLS == (
        "This patch is {m} million voxels; the first update prepares it once "
        "and may take a while. Later clicks are quick.")
    assert AP.ADD_S1_LIVE_OVER_GATE_PATCH == (
        "Press Preview to see the cells — this patch is not updated as you "
        "click.")
    assert AP.ADD_S1_WHOLE_TOO_BIG == (
        "This patch is {m} million voxels; a whole patch can be filled up to "
        "{k} million. Press S, M or L for a box.")
    # a Preview while an earlier one still finishes in the background (its
    # thread could not be waited for): Stop is grey then, so not named
    assert AP.ADD_S1_PARKED_REFUSED == (
        "An earlier preview is still finishing in the background. Wait for "
        "it to end, then press Preview again.")
    assert "Stop" not in AP.ADD_S1_PARKED_REFUSED
    assert AP.ADD_SUM_ADVANCED_FILL == "filled to the walls · membrane included"
    assert AP.ADD_SUM_ADVANCED_FILL_NO_MEMBRANE == (
        "filled to the walls · membrane left out")
    assert AP.ADD_AMBER_ONE_CHANNEL_WHOLE == (
        "Only one channel. If it is the nuclear stain, walls read from nuclei "
        "squeeze cells down to their nuclei — press S, M or L for a box, then "
        "pick Grow by ▸ Spread from each marker and set Wall cost to 0 under "
        "Advanced.")
    assert AP.ADD_HELP_GROWTH_LIMIT == (
        "How far a cell may spread: a budget, not a distance. Plain cytoplasm "
        "uses it up about 6 µm from the marker; a bright membrane costs "
        "nearly the whole limit per µm, so a cell stops in its wall. Lower it "
        "if cells grow past their walls.")
    assert AP.ADD_HELP_GROW == (
        "Fill: the fill runs to the walls, the box edges and dark space, as "
        "before. Spread: each marker grows along its easiest path and stops "
        "where a neighbour got there first, at a wall, or at the growth "
        "limit. Pick Spread when the fill leaks through a gap in the walls "
        "or merges neighbours.")
    assert AP.ADD_NOTE_DARK_OFF_LEAK_WHOLE == (
        "Dark space is not treated as outside and there is no tissue outline, "
        "so cells grow a few µm into open background — tick it in Advanced, "
        "or pick a Tissue outline under Walls.")
    assert AP.ADD_S1_WARMING == (
        "Preparing the cell-growth engine for its first run on this computer "
        "— about 10 s, once.")
    assert AP.ADD_WHOLE_NEEDS_NUMBA == (
        "Cells fills a whole patch only with the numba package, which is "
        "missing from this install. Use boxes.")
    assert AP.ADD_WHOLE_NEEDS_NUMBA_CACHE == (
        "Cells fills a whole patch only with the numba package, which cannot "
        "write its cache folder here. Use boxes.")
    # the reused sentences, unchanged
    assert AP.ADD_S1_BUSY_REFUSED == (
        "Wait for the preview to finish, or press Stop.")
    assert AP.ADD_S1_PICK_NUCLEI_CHANNEL == (
        "Pick the nuclei channel under Markers.")
    assert AP.ADD_S1_CHANNEL_CLASH == (
        "The nuclei and wall channels must be different.")
    # no sentence of the fill says "reach" or promises how far it grows
    for v in (AP.ADD_HELP_WHOLE_FILL, AP.ADD_HELP_WHOLE_FILL_LIVE,
              AP.ADD_S1_WHOLE_TOO_BIG,
              AP.ADD_SUM_ADVANCED_FILL, AP.ADD_SUM_ADVANCED_FILL_NO_MEMBRANE,
              AP.ADD_AMBER_ONE_CHANNEL_WHOLE, AP.ADD_WHOLE_TIP,
              AP.ADD_S1_PARKED_REFUSED):
        assert "reach" not in v and "µm" not in v, v
