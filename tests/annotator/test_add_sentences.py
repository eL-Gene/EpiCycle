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
    (1, "Box 2 × 3 × 4 voxels · 1 marker placed"),
    (2, "Box 2 × 3 × 4 voxels · 2 markers placed"),
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
@pytest.mark.parametrize("const", ["ADD_NOTE_DEFAULT_SPACING"])
def test_the_voxel_sentences_point_at_step3_qc_settings(const):
    """Step 3's spacing comes from QC Settings, not Step 1 (the live run on
    WholeScale_patch_4 had a Step 1 voxel size and was still refused), so
    the sentence must send the user to QC Features -> Settings..., never to
    Step 1."""
    text = getattr(AP, const)
    assert "Step 1" not in text, text
    assert "QC Features → Settings…" in text, text
    assert "Spacing" in text, text


# --------------------------------------------------------------------------- #
# A5 -- the click marker list's words
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n,k,want", [
    (7, 5, "Box 2 × 3 × 4 voxels · 7 markers placed · 5 of 7 ticked"),
    (2, 1, "Box 2 × 3 × 4 voxels · 2 markers placed · 1 of 2 ticked"),
    (3, 3, "Box 2 × 3 × 4 voxels · 3 markers placed"),
    (1, 1, "Box 2 × 3 × 4 voxels · 1 marker placed"),
])
def test_box_clicks_with_the_ticked_tail(n, k, want):
    assert AP.box_line((2, 3, 4), preset="cells", n_clicks=n,
                       n_ticked=k) == want


@pytest.mark.parametrize("n,want", [
    (1, "Tick the marker to cut — it is unticked."),
    (2, "Tick at least one marker to cut — all 2 are unticked."),
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
