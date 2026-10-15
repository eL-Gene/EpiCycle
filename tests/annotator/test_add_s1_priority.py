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
             info_pending=False, empty_share=0.5)
    s.update(kw)
    return s


ROWS = [
    (1, dict(has_patch=False, transient="x", busy=True), AP.ADD_S1_NO_PATCH),
    (2, dict(transient=AP.ADD_S1_STOPPED, busy=True), AP.ADD_S1_STOPPED),
    (3, dict(busy=True, review=(2, 5, False)), AP.ADD_S1_WORKING),
    (4, dict(review=(2, 1234, False), candidates=(1, 0, 0, 0, 1)),
     AP.added_line(2, 1234, has_selection=False)),
    (5, dict(candidates=(4, 1, 1, 0, 2), has_box=False),
     AP.found_line(4, k_edge=1, k_small=1)),
    (5, dict(candidates=(4, 1, 1, 0, 0), user_emptied=True),
     AP.ADD_S1_TICK_ONE),
    # every row STARTED unticked: the count, not TICK_ONE (live-shot V3)
    (5, dict(candidates=(2, 2, 0, 0, 0)),
     "2 new objects found · 2 left unticked (at the box edge)"),
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
     "Box 10 × 20 × 20 voxels · 7 markers placed · 5 of 7 ticked"),
    (14, dict(), AP.box_line((10, 20, 20), preset="cells", empty_share=0.5)),
    (14, dict(preset="nuclei"), AP.box_line((10, 20, 20), preset="nuclei",
                                             empty_share=0.5)),
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


def test_found_line_plain_form():
    assert AP.found_line(5, k_plain=2) == "5 new objects found" + (
        AP.ADD_S1_UNTICKED_PLAIN.format(k=2))
    assert AP.found_line(5, k_edge=3, k_plain=0) == "5 new objects found"
    # the 6-entry candidates tuple reaches it through s1_sentence
    s = dict(has_box=True, preset="cells", candidates=(5, 0, 0, 0, 3, 2))
    assert AP.s1_sentence(s) == "5 new objects found · 2 left unticked"


def test_row_12b_is_amber_and_greys_preview():
    """Every marker unticked is a blocking reason."""
    s = _ready(source="paint", n_clicks=3, n_ticked=0)
    assert AP.preview_block(s) == AP.tick_one_marker_line(3)
    assert AP.s1_kind(s) == "warn"
    # an older caller that sends no n_ticked: every marker counts as ticked
    assert AP.preview_block(_ready(source="paint", n_clicks=3)) is None
