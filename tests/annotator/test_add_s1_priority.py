"""Which sentence S1 shows -- every row of the table, first match wins, and
transient sentences give way (B2.7)."""

from __future__ import annotations

import numpy as np
import pytest

from packages.epicycle_annotator import add_preview as AP

from .add_harness import Harness, install_fakes


def _ready(**kw):
    s = dict(has_patch=True, transient=None, busy=False, review=None,
             candidates=None, has_box=True, preset="cells",
             box_shape=(10, 20, 20), box_budget=AP.ADD_BOX_BUDGET,
             channel="nuc", wall_channel="mem", nuclei_channel="nuc",
             source="nuclei_masks", run_state=None, n_clicks=0,
             info_pending=False, empty_share=0.5,
             # a run picked under Markers, and another run on the patch
             run_id="r", has_other_run=True)
    s.update(kw)
    return s


ROWS = [
    (1, dict(has_patch=False, transient="x", busy=True), AP.ADD_S1_NO_PATCH),
    (2, dict(transient=AP.ADD_S1_STOPPED, busy=True), AP.ADD_S1_STOPPED),
    (3, dict(busy=True, review=(2, 5, False)), AP.ADD_S1_WORKING),
    (3, dict(busy=True, whole_patch=True), AP.ADD_S1_WORKING_PATCH),
    (4, dict(review=(2, 1234, False), candidates=(1, 0, 0, 0, 1)),
     AP.added_line(2, 1234, has_selection=False)),
    (5, dict(candidates=(4, 1, 1, 0, 2), has_box=False),
     AP.found_line(4, k_edge=1, k_small=1)),
    (5, dict(candidates=(4, 1, 1, 0, 0), user_emptied=True),
     AP.ADD_S1_TICK_ONE),
    # every row STARTED unticked: the count and what to press, not TICK_ONE
    # (live-shot V3)
    (5, dict(candidates=(2, 2, 0, 0, 0)),
     "2 new objects found · none ticked (at the box edge) — press All, or "
     "tick the ones to add"),
    # no box: one idle line on both presets, naming Whole patch (never
    # grey: a choice that cannot fill a whole patch greys Preview later)
    (6, dict(has_box=False, source="labels"), AP.ADD_S1_IDLE),
    # ... except on Cells where Spread cannot run: no whole patch there
    (6, dict(has_box=False, source="labels", numba_ok=False),
     AP.ADD_S1_IDLE_BOX),
    (6, dict(has_box=False), AP.ADD_S1_IDLE),
    # row 7 (ADD_S1_NO_VOXEL_SIZE) is retired: the
    # old key, if a caller still sends it, blocks nothing
    (8, dict(voxel_known=False, box_shape=(200, 200, 200)),
     AP.over_limit_line((200, 200, 200))),
    (8, dict(box_shape=(200, 200, 200), wall_channel=""),
     AP.over_limit_line((200, 200, 200))),
    (9, dict(wall_channel="", nuclei_channel=""), AP.ADD_S1_PICK_WALL_CHANNEL),
    (9, dict(nuclei_channel=""), AP.ADD_S1_PICK_NUCLEI_CHANNEL),
    (9, dict(preset="nuclei", channel=""), AP.ADD_S1_PICK_CHANNEL),
    (10, dict(nuclei_channel="mem"), AP.ADD_S1_CHANNEL_CLASH),
    (11, dict(source="labels", run_state="loading"), AP.ADD_S1_RUN_LOADING),
    (11, dict(source="labels", run_state="refused"), AP.ADD_S1_RUN_UNUSABLE),
    (12, dict(source="paint", n_clicks=0, info_pending=True),
     AP.ADD_S1_NO_MARKERS),
    # row 12b: every marker unticked
    (12.5, dict(source="paint", n_clicks=3, n_ticked=0, info_pending=True),
     AP.tick_one_marker_line(3)),
    (12.5, dict(source="paint", n_clicks=1, n_ticked=0),
     AP.ADD_S1_TICK_ONE_MARKER_ONE),
    (13, dict(info_pending=True), AP.ADD_S1_MEASURING),
    (14, dict(source="paint", n_clicks=3), AP.box_line(
        (10, 20, 20), preset="cells", n_clicks=3)),
    (14, dict(source="paint", n_clicks=7, n_ticked=5),
     "Box 10 × 20 × 20 voxels · 7 objects marked · 5 of 7 ticked"),
    (14, dict(), AP.box_line((10, 20, 20), preset="cells", empty_share=0.5)),
    (14, dict(preset="nuclei"), AP.box_line((10, 20, 20), preset="nuclei",
                                             empty_share=0.5)),
    # the same idle line on Nuclei, whatever Look for nuclei says
    (6, dict(has_box=False, preset="nuclei", look_for="inside"),
     AP.ADD_S1_IDLE),
    (6, dict(has_box=False, preset="nuclei", look_for="anywhere"),
     AP.ADD_S1_IDLE),
    (6, dict(has_box=False, preset="nuclei"), AP.ADD_S1_IDLE),
    # Inside with the spacing never set: said before a box (M7)
    (6, dict(has_box=False, preset="nuclei", look_for="inside",
             spacing_unknown=True), AP.ADD_S1_NEEDS_SPACING),
    (6, dict(has_box=False, preset="nuclei", look_for="anywhere",
             spacing_unknown=True), AP.ADD_S1_IDLE),
    # Inside each cell: row 11 on the Nuclei preset, then the new blocking
    # rows (the spacing, no cell of that run), then 13
    (11, dict(preset="nuclei", look_for="inside", run_state="loading",
              spacing_unknown=True), AP.ADD_S1_RUN_LOADING),
    (11, dict(preset="nuclei", look_for="inside", run_state="refused"),
     AP.ADD_S1_INSIDE_UNUSABLE),
    (11.2, dict(preset="nuclei", look_for="inside", spacing_unknown=True,
                inside_count=0), AP.ADD_S1_NEEDS_SPACING_BOXED),
    # the patch's levels refused the channel: the channel's line, after the
    # spacing (no levels are measured at an unknown spacing)
    (11.3, dict(preset="nuclei", look_for="inside", levels_error=True,
                inside_count=0, info_pending=True),
     AP.ADD_S1_INSIDE_CHANNEL_UNUSABLE),
    (11.2, dict(preset="nuclei", look_for="inside", spacing_unknown=True,
                levels_error=True), AP.ADD_S1_NEEDS_SPACING_BOXED),
    (11.4, dict(preset="nuclei", look_for="inside", inside_count=0,
                info_pending=True), AP.ADD_S1_INSIDE_NO_CELLS),
    (13, dict(preset="nuclei", look_for="inside", inside_count=None,
              info_pending=True), AP.ADD_S1_MEASURING),
    (9, dict(preset="nuclei", look_for="inside", channel="",
             run_state="loading"), AP.ADD_S1_PICK_CHANNEL),
    (14, dict(preset="nuclei", look_for="inside", inside_count=3),
     AP.box_line((10, 20, 20), preset="nuclei", empty_share=0.5)),
    # Anywhere never waits for the other run or the spacing
    (14, dict(preset="nuclei", look_for="anywhere", run_state="loading",
              spacing_unknown=True, info_pending=True),
     AP.box_line((10, 20, 20), preset="nuclei", empty_share=0.5)),
    (14, dict(preset="nuclei", whole_patch=True, look_for="inside"),
     AP.ADD_S1_PATCH_PLAIN.format(z=10, y=20, x=20, m="0.1", p=50)),
    # a whole patch stays whole on Anywhere: Preview is grey and says why
    (8.05, dict(preset="nuclei", whole_patch=True, look_for="anywhere"),
     AP.ADD_S1_WHOLE_ANYWHERE),
    # Cells on a whole patch: never over the box limit (it is filled to the
    # walls in one go); the solver, the whole-patch ceiling and the spacing
    # first, then the outline being measured, then the ready line (the same
    # words as on Nuclei)
    (14, dict(whole_patch=True, box_shape=(200, 200, 200), source="labels"),
     AP.ADD_S1_PATCH_PLAIN.format(z=200, y=200, x=200, m="8.0", p=50)),
    (14, dict(whole_patch=True, source="labels"),
     AP.ADD_S1_PATCH_PLAIN.format(z=10, y=20, x=20, m="0.1", p=50)),
    (8.1, dict(whole_patch=True, numba_ok=False, spacing_unknown=True,
               box_shape=(500, 500, 500), source="labels"),
     AP.ADD_WHOLE_NEEDS_NUMBA),
    # numba is there but cannot write its cache folder: its own words
    (8.1, dict(whole_patch=True, numba_ok=False, numba_why="cache",
               source="labels"),
     AP.ADD_WHOLE_NEEDS_NUMBA_CACHE),
    # a whole patch over the fill's ceiling: a box is the way
    (8.12, dict(whole_patch=True, box_shape=(500, 500, 500), source="paint",
                spacing_unknown=True),
     AP.whole_too_big_line(125_000_000)),
    # a Cells whole patch takes its markers from another run or from the
    # clicks: nuclei masks ask for one of them
    (8.15, dict(whole_patch=True, source="nuclei_masks",
                spacing_unknown=True),
     AP.ADD_S1_WHOLE_PICK_RUN),
    (8.15, dict(whole_patch=True), AP.ADD_S1_WHOLE_PICK_RUN),
    # Click markers on a whole patch pass that row and reach the spacing
    # refusal (in microns: the fill and the Ball)
    (8.2, dict(whole_patch=True, source="paint", n_clicks=2,
               spacing_unknown=True),
     AP.ADD_S1_NEEDS_SPACING_BOXED),
    # ... and the marker rows of a box
    (12, dict(whole_patch=True, source="paint", n_clicks=0),
     AP.ADD_S1_NO_MARKERS),
    (14, dict(whole_patch=True, source="paint", n_clicks=3),
     "Whole patch 10 × 20 × 20 voxels · 3 objects marked"),
    (8.2, dict(whole_patch=True, spacing_unknown=True, wall_channel="",
               source="labels"),
     AP.ADD_S1_NEEDS_SPACING_BOXED),
    # the patch's own words while it is measured
    (13, dict(whole_patch=True, source="labels", info_pending=True),
     AP.ADD_S1_MEASURING_PATCH),
    # an ordinary Cells box never refuses at the default spacing
    (14, dict(spacing_unknown=True),
     AP.box_line((10, 20, 20), preset="cells", empty_share=0.5)),
    (12.8, dict(outline_pending=True, info_pending=True),
     AP.ADD_OUTLINE_PENDING),
    (12.7, dict(outline_error="The outline could not be measured.",
                outline_pending=True), "The outline could not be measured."),
    (11, dict(source="labels", run_state="loading", outline_pending=True),
     AP.ADD_S1_RUN_LOADING),
]


@pytest.mark.parametrize("row,kw,want", ROWS)
def test_every_row(row, kw, want):
    assert AP.s1_sentence(_ready(**kw)) == want, row


def test_rows_6_to_13_are_exactly_the_grey_preview_states():
    for row, kw, _want in ROWS:
        s = _ready(**kw)
        if s.get("transient") or s.get("busy") or s.get("review") \
                or s.get("candidates") or not s.get("has_patch"):
            continue
        blocked = AP.preview_block(s) is not None
        assert blocked is (6 <= row <= 13), row
    assert 7 not in {row for row, _kw, _w in ROWS}, "row 7 is retired"


def test_the_default_spacing_blocks_nothing():
    """Decision 22: a ready Cells state stays ready whatever the spacing."""
    for flag in (True, False):
        s = _ready(default_spacing=flag, voxel_known=False)
        assert AP.preview_block(s) is None


def test_whole_patch_run_rows():
    """A Cells whole patch grows its cells from another run's objects or
    from Click markers: with neither picked (nuclei masks, or the From combo
    left blank) S1 asks for one -- or, on a patch with no other run, names
    Click markers or a box.  Both grey Preview, after the solver and the
    size ceiling."""
    whole = dict(whole_patch=True)
    assert AP.preview_block(_ready(source=AP.SOURCE_NUCLEI, **whole)) == (
        AP.ADD_S1_WHOLE_PICK_RUN)
    assert AP.preview_block(_ready(source=AP.SOURCE_NUCLEI,
                                   has_other_run=False, **whole)) == (
        AP.ADD_S1_WHOLE_NO_RUN)
    # Click markers pass the row, with or without another run: the marker
    # rows of a box follow, then the ready line
    for other in (True, False):
        clicks = dict(whole, source=AP.SOURCE_CLICKS, has_other_run=other)
        assert AP.preview_block(_ready(n_clicks=0, **clicks)) == (
            AP.ADD_S1_NO_MARKERS)
        assert AP.preview_block(_ready(n_clicks=2, n_ticked=0, **clicks)) == (
            AP.tick_one_marker_line(2))
        assert AP.preview_block(_ready(n_clicks=2, n_ticked=2,
                                       **clicks)) is None
        # the solver and the ceiling still come first
        assert AP.preview_block(_ready(numba_ok=False, n_clicks=2,
                                       **clicks)) == AP.ADD_WHOLE_NEEDS_NUMBA
        assert AP.preview_block(_ready(box_shape=(500, 500, 500), n_clicks=2,
                                       **clicks)) == (
            AP.whole_too_big_line(125_000_000))
    # the blank combo of a whole patch: a run, none picked
    blank = dict(whole, source=AP.SOURCE_RUN, run_id=None)
    assert AP.preview_block(_ready(**blank)) == AP.ADD_S1_WHOLE_PICK_RUN
    assert AP.preview_block(_ready(has_other_run=False, **blank)) == (
        AP.ADD_S1_WHOLE_NO_RUN)
    for kw in (dict(blank), dict(blank, has_other_run=False)):
        assert AP.s1_kind(_ready(**kw)) == "warn"
        assert AP.s1_sentence(_ready(**kw)) == AP.preview_block(_ready(**kw))
    # a run picked: the patch's ready line
    assert AP.preview_block(_ready(source=AP.SOURCE_RUN, run_id="r",
                                   **whole)) is None
    # the solver and the ceiling still come first
    assert AP.preview_block(_ready(numba_ok=False, **blank)) == (
        AP.ADD_WHOLE_NEEDS_NUMBA)
    assert AP.preview_block(_ready(box_shape=(500, 500, 500), **blank)) == (
        AP.whole_too_big_line(125_000_000))
    # never on a box
    assert AP.preview_block(_ready(source=AP.SOURCE_CLICKS, n_clicks=2,
                                   n_ticked=2)) is None
    # the words, and the old sentence is gone (a stale reader fails)
    assert AP.ADD_S1_WHOLE_PICK_RUN == (
        "Pick a run or Click markers under Markers ▸ From — a whole patch "
        "grows its cells from those markers.")
    assert AP.ADD_S1_WHOLE_NO_RUN == (
        "This patch has no other run — pick Click markers under Markers ▸ "
        "From, or press S, M or L for a box.")
    for line in (AP.ADD_S1_WHOLE_PICK_RUN, AP.ADD_S1_WHOLE_NO_RUN):
        assert AP.ADD_ITEM_CLICKS in line
    assert not hasattr(AP, "ADD_S1_WHOLE_NEEDS_RUN")


def test_zero_run_markers_grey_preview():
    """A run that gives no marker in this box (or patch) greys Preview
    BEFORE the press, from the box-info run count (found, markers).  What
    S1 says is what is true: the outline's sentence when the tissue outline
    dropped them; no object of the run here at all; markers dropped while
    objects are kept here (dark space, over an object, too small); or
    dropped with nothing kept here (dark space, too small) -- naming the
    dark-space tick only while it is on and no outline decides instead."""
    run = dict(source=AP.SOURCE_RUN, run_id="r")
    dark = AP.ADD_DARK_OUTSIDE_LABEL
    rows = [
        # (state, box sentence, whole-patch sentence)
        (dict(n_run_found=0, n_run_markers=0),
         AP.ADD_S1_RUN_NONE_HERE, AP.ADD_S1_RUN_NONE_HERE_PATCH),
        (dict(n_run_found=5, n_run_markers=0, empty_share=0.5),
         AP.ADD_S1_RUN_NO_MARKERS, AP.ADD_S1_RUN_NO_MARKERS_PATCH),
        (dict(n_run_found=5, n_run_markers=0, empty_share=1.0,
              dark_outside=True),
         AP.ADD_S1_RUN_NO_MARKERS_EMPTY.format(dark=dark),
         AP.ADD_S1_RUN_NO_MARKERS_EMPTY_PATCH.format(dark=dark)),
        # the tick off: nothing to turn off
        (dict(n_run_found=5, n_run_markers=0, empty_share=1.0,
              dark_outside=False),
         AP.ADD_S1_RUN_NO_MARKERS_EMPTY_PLAIN,
         AP.ADD_S1_RUN_NO_MARKERS_EMPTY_PLAIN_PATCH),
        # an outline decides what is outside (the tick is not used then)
        (dict(n_run_found=5, n_run_markers=0, empty_share=1.0,
              dark_outside=True, outline=AP.OUTLINE_SHRINK,
              n_outside_outline=0),
         AP.ADD_S1_RUN_NO_MARKERS_EMPTY_PLAIN,
         AP.ADD_S1_RUN_NO_MARKERS_EMPTY_PLAIN_PATCH),
    ]
    for kw, box, whole in rows:
        assert AP.preview_block(_ready(**kw, **run)) == box, kw
        assert AP.preview_block(_ready(whole_patch=True, **kw, **run)) == (
            whole), kw
        assert AP.s1_kind(_ready(**kw, **run)) == "warn"
    # the outline's own sentences come first
    out = dict(n_run_found=5, n_run_markers=0, n_outside_outline=4)
    assert AP.preview_block(_ready(outline=AP.OUTLINE_SHRINK, **out,
                                   **run)) == AP.ADD_S1_NO_MARKERS_OUTLINE
    assert AP.preview_block(_ready(outline=AP.OUTLINE_CLEARED, **out,
                                   **run)) == (
        AP.ADD_S1_NO_MARKERS_OUTLINE_CLEARED)
    # markers to grow from, or not counted yet: not grey for this
    assert AP.preview_block(_ready(n_run_found=5, n_run_markers=3,
                                   **run)) is None
    assert AP.preview_block(_ready(n_run_markers=None, **run)) is None
    # while the box is measured S1 says so first
    assert AP.preview_block(_ready(n_run_found=0, n_run_markers=0,
                                   info_pending=True, **run)) == (
        AP.ADD_S1_MEASURING)
    # only the run source has a count before the press
    assert AP.preview_block(_ready(source=AP.SOURCE_NUCLEI, n_run_found=0,
                                   n_run_markers=0)) is None
    assert AP.ADD_DARK_OUTSIDE_LABEL == "Treat dark space as outside"
    assert AP.ADD_S1_RUN_NONE_HERE == (
        "That run has no object in this box — move the box, or pick another "
        "run.")
    assert AP.ADD_S1_RUN_NONE_HERE_PATCH == (
        "That run has no object in this patch — pick another run.")
    assert AP.ADD_S1_RUN_NO_MARKERS == (
        "None of that run's objects here can be a marker: each is in dark "
        "space, on a kept object, or too small. Move the box, or Clear "
        "objects.")
    assert AP.ADD_S1_RUN_NO_MARKERS_PATCH == (
        "None of that run's objects can be a marker: each is in dark space, "
        "on an object already here, or too small. Clear objects to make "
        "room.")
    assert AP.ADD_S1_RUN_NO_MARKERS_EMPTY == (
        "None of that run's objects here can be a marker: each is in dark "
        "space or too small. Turn off {dark} under Advanced.")
    assert AP.ADD_S1_RUN_NO_MARKERS_EMPTY_PATCH == (
        "None of that run's objects can be a marker: each is in dark space "
        "or too small. Turn off {dark} under Advanced.")
    assert AP.ADD_S1_RUN_NO_MARKERS_EMPTY_PLAIN == (
        "None of that run's objects here can be a marker: each is in dark "
        "space or too small. Move the box.")
    assert AP.ADD_S1_RUN_NO_MARKERS_EMPTY_PLAIN_PATCH == (
        "None of that run's objects can be a marker: each is in dark space "
        "or too small.")
    # no sentence claims the objects lie over empty space any more, and
    # none is longer than the longest Cells line the page was measured
    # with (138 characters): no height moves
    for name in dir(AP):
        if name.startswith(("ADD_S1_RUN_NO", "ADD_S1_RUN_NONE")):
            text = getattr(AP, name).format(dark=dark)
            assert "over empty space" not in text, name
            assert len(text) <= 138, (name, len(text))


def test_no_markers_follows_place_markers():
    """With no click marker placed, S1 never asks to turn on what is on."""
    off = _ready(source=AP.SOURCE_CLICKS, n_clicks=0)
    on = _ready(source=AP.SOURCE_CLICKS, n_clicks=0, place_markers_on=True)
    assert AP.preview_block(off) == AP.ADD_S1_NO_MARKERS
    assert AP.preview_block(on) == AP.ADD_S1_NO_MARKERS_ON
    assert AP.s1_kind(off) == AP.s1_kind(on) == "warn"
    # markers placed: the ready line whatever the button says
    placed = _ready(source=AP.SOURCE_CLICKS, n_clicks=2, n_ticked=2,
                    place_markers_on=True)
    assert AP.preview_block(placed) is None
    assert AP.ADD_S1_NO_MARKERS_ON == (
        "Place at least one marker: click inside the box in a side view.")
    assert "turn on Place markers" not in AP.ADD_S1_NO_MARKERS_ON


def test_nuclei_measures_until_the_empty_share_lands():
    assert AP.s1_sentence(_ready(preset="nuclei", empty_share=None)) == (
        AP.ADD_S1_MEASURING)
    assert AP.preview_block(_ready(preset="nuclei", empty_share=None)) is None


# --------------------------------------------------------------------------- #
# transient sentences give way (the orchestrator)
# --------------------------------------------------------------------------- #
@pytest.fixture
def h(monkeypatch):
    install_fakes(monkeypatch)
    hh = Harness(np.zeros((20, 30, 30), np.int32), spacing=(0.5, 0.5, 0.5))
    hh.adv.preset = "nuclei"
    hh.place_box((slice(2, 18), slice(2, 28), slice(2, 28)))
    hh._add_info = AP.AddBoxInfo(empty_share=1.0)
    return hh


@pytest.mark.parametrize("change", [
    lambda h: h._on_add_box_changed(10, 15, 15, 8, 8, 8),
    lambda h: h._on_add_advanced_changed(),
    lambda h: h._on_add_preset_changed("nuclei"),
    lambda h: h._on_add_source_changed(("paint", None)),
])
def test_a_transient_gives_way_to_the_next_change(h, change):
    h._add_transient = AP.ADD_S1_STOPPED
    change(h)
    assert h._add_transient is None
    assert h.adv.last("status_text") != AP.ADD_S1_STOPPED


def test_busy_refused_gives_way_at_the_end_of_the_run(h):
    h._add_busy = True
    h._add_thread = object()
    h._recenter_add_box_on_crosshair()
    assert h.adv.last("status_text") == AP.ADD_S1_BUSY_REFUSED
    h._add_thread = None
    h._on_add_thread_finished()
    assert h._add_transient is None


# --------------------------------------------------------------------------- #
# B3.4 (a) -- "N left unticked" follows the ticks
# --------------------------------------------------------------------------- #
def test_the_unticked_count_follows_the_ticks(h):
    from types import SimpleNamespace

    h._add_preview_result = np.zeros((2, 2, 2), np.int32)
    h._add_preview_res = SimpleNamespace(ids=[1, 2, 3, 4, 5],
                                         edge_ids=[1, 2], small_ids=[3])

    def s1(checked):
        h.adv.checked = checked
        h._add_refresh_ui()
        return h.adv.last("status_text")

    # as it started: the flagged rows unticked, for their reasons
    assert s1([4, 5]) == (
        "5 new objects found · 3 left unticked (at the box edge or small)")
    # the user ticks an edge row: the live count, the reasons still true
    assert s1([1, 4, 5]) == (
        "5 new objects found · 2 left unticked (at the box edge or small)")
    assert s1([1, 3, 4, 5]) == (
        "5 new objects found · 1 left unticked (at the box edge)")
    assert s1([1, 2, 3, 4, 5]) == "5 new objects found"
    # the user unticks a row that started ticked: the count, no reason
    assert s1([1, 3, 5]) == "5 new objects found · 2 left unticked"
    assert s1([1, 2, 3, 5]) == "5 new objects found · 1 left unticked"


def test_the_inside_transients_and_tails():
    """Nothing found / every one taken are transients (they give way); the
    two-nuclei tail follows the count, after Accept too."""
    for t in (AP.ADD_S1_INSIDE_NOTHING, AP.ADD_S1_INSIDE_ALL_TAKEN):
        s = _ready(preset="nuclei", look_for="inside", transient=t)
        assert AP.s1_sentence(s) == t
        assert AP.s1_kind(s) == "warn"
    s = dict(has_box=True, preset="nuclei", candidates=(6, 0, 0, 0, 6),
             n_two=2)
    assert AP.s1_sentence(s) == "6 new objects found · 2 cells hold two nuclei"
    s = dict(has_box=True, preset="nuclei", review=(6, 100, 1),
             review_two=1)
    assert AP.s1_sentence(s) == (
        "Added 6 objects · 100 voxels · " + AP.ADD_S1_ADDED_TWO_TAIL_ONE)
    s["review"] = (6, 100, 0)
    assert AP.s1_sentence(s) == (
        "Added 6 objects · 100 voxels · " + AP.ADD_S1_ADDED_TWO_TAIL_ONE
        + AP.ADD_S1_SELECT_ONE)


def test_found_line_plain_form():
    assert AP.found_line(5, k_plain=2) == "5 new objects found" + (
        AP.ADD_S1_UNTICKED_PLAIN.format(k=2))
    assert AP.found_line(5, k_edge=3, k_plain=0) == "5 new objects found"
    # the 6-entry candidates tuple reaches it through s1_sentence
    s = dict(has_box=True, preset="cells", candidates=(5, 0, 0, 0, 3, 2))
    assert AP.s1_sentence(s) == "5 new objects found · 2 left unticked"


def test_none_ticked_names_all():
    """No row ticked: the line names All and the dot is amber, whether the
    rows started unticked or the user unticked the last one."""
    s = _ready(candidates=(2, 2, 0, 0, 0))
    assert AP.s1_sentence(s) == AP.found_none_ticked_line(2, AP.ADD_WHY_EDGE)
    assert "press All" in AP.s1_sentence(s)
    assert AP.s1_kind(s) == "warn"
    s = _ready(candidates=(1, 0, 1, 0, 0))
    assert AP.s1_sentence(s) == (
        "1 new object found · not ticked (small) — press All, or tick it")
    assert AP.s1_kind(s) == "warn"
    s = _ready(candidates=(5, 4, 3, 2, 0))
    assert AP.s1_sentence(s) == AP.found_none_ticked_line(5, AP.ADD_WHY_BOTH)
    # the two-nuclei tail comes before the none-ticked tail
    s = _ready(preset="nuclei", candidates=(3, 3, 0, 0, 0), n_two=1)
    assert AP.s1_sentence(s) == AP.found_none_ticked_line(
        3, AP.ADD_WHY_EDGE, 1)
    assert AP.s1_kind(s) == "warn"
    # the user unticked the last row
    s = _ready(candidates=(4, 1, 1, 0, 0), user_emptied=True)
    assert AP.s1_sentence(s) == AP.ADD_S1_TICK_ONE
    assert "press All" in AP.ADD_S1_TICK_ONE
    assert AP.s1_kind(s) == "warn"
    # one row ticked: the found line, the review dot (as before)
    s = _ready(candidates=(4, 1, 0, 0, 3))
    assert AP.s1_sentence(s) == AP.found_line(4, k_edge=1)
    assert AP.s1_kind(s) == "review"


def test_row_12b_is_amber_and_greys_preview():
    """Every marker unticked is a blocking reason."""
    s = _ready(source="paint", n_clicks=3, n_ticked=0)
    assert AP.preview_block(s) == AP.tick_one_marker_line(3)
    assert AP.s1_kind(s) == "warn"
    # an older caller that sends no n_ticked: every marker counts as ticked
    assert AP.preview_block(_ready(source="paint", n_clicks=3)) is None
