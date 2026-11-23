"""Nuclei ▸ the Tissue outline: an option, off by default.

The engine half (pure): with no outline chosen a Nuclei Preview is what it
always was, byte for byte; on Anywhere in the box a nucleus lying mostly
outside the outline (at least 70 % of it) is left out and the rest are cut
to the outline, renumbered without gaps; on the markers the outline only
cuts -- a clicked nucleus is never dropped for lying mostly outside, and a
marker is marked only when the cut empties its nucleus; the counts note
says it in the Nuclei words; one shrink-wrap serves both presets; a stain
of nuclei alone keeps a crowded field and leaves a lone nucleus outside.

The page half (a real panel, offscreen): the Tissue outline card on Nuclei
▸ Anywhere and Nuclei ▸ markers (a box, before a result), its own combo
starting at "No outline (as before)", the two presets' choices apart, one
card open at a time, the histogram folded while the card is open, Dark
below and Bridge gaps under Advanced with the shrink-wrap, the spacing
amber, the summaries; never on Inside each cell or a whole patch.

The orchestrator half (the Qt-free harness): the outline mode, the gather
(nothing new without an outline), the box-info job and the cache one
measurement fills for both presets, Show, the full exit, the cleared
fall-back, half size, the failed-outline sentence.
"""

from __future__ import annotations

import numpy as np
import pytest

from packages.epicycle_annotator import add_preview as AP
from packages.epicycle_annotator.reshape import outline as O

from .add_harness import FakeAdv, Harness, install_fakes
from .add_widget_fixture import qapp, reset_add, widget  # noqa: F401

ISO = (1.0, 1.0, 1.0)
MARKERS = (AP.LOOK_MARKERS, None)
ANYWHERE = (AP.LOOK_ANYWHERE, None)

#: the patch, the box (not at the origin: the patch's outline is cut to it)
PATCH = (20, 50, 80)
ROI = (slice(2, 18), slice(5, 45), slice(5, 75))
#: the outline: everything left of x = 45 in the patch
EDGE_X = 45
#: nuclei (box-local centres, radius 4): A wholly inside, B wholly outside,
#: C cut by the edge (more than half outside, under 70 %), D almost wholly
#: outside
CENTRES = {"A": (8, 10, 12), "B": (8, 30, 64), "C": (8, 20, 40),
           "D": (8, 32, 43)}
R_NUC = 4


def _ball(shape, c, r):
    zz, yy, xx = np.indices(shape)
    return ((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) <= r * r


def _world(names=("A", "B", "C", "D")):
    box = tuple(int(sl.stop - sl.start) for sl in ROI)
    img = np.zeros(box, np.float32)
    for n in names:
        img[_ball(box, CENTRES[n], R_NUC)] = 1.0
    outline = np.zeros(PATCH, bool)
    outline[:, :, :EDGE_X] = True
    return img, outline


def _outline_box(outline):
    return outline[ROI]


def _share_outside(img, outline, name):
    box = img.shape
    m = _ball(box, CENTRES[name], R_NUC)
    o = _outline_box(outline)
    return float(np.count_nonzero(m & ~o)) / float(np.count_nonzero(m))


def _params(**kw):
    p = {"clahe": False, "clahe_clip": 0.01, "threshold": 0.5,
         "min_size": 0, "separate": True, "min_distance": 3,
         "clear_border": False, "edge_margin": 3, "auto_smooth": False,
         "smooth_sigma": 1.5}
    p.update(kw)
    return p


def _anywhere(img, **kw):
    return AP.run_add_preview(
        preset="nuclei", roi=ROI, existing_roi=np.zeros(img.shape, np.int32),
        next_label=500, spacing=ISO, image_roi=img, **kw)


# --------------------------------------------------------------------------- #
# the engine (pure)
# --------------------------------------------------------------------------- #
def test_the_world_is_what_the_tests_say():
    img, outline = _world()
    assert _share_outside(img, outline, "A") == 0.0
    assert _share_outside(img, outline, "B") == 1.0
    assert 0.5 < _share_outside(img, outline, "C") < O.OUTLINE_DROP_SHARE
    assert _share_outside(img, outline, "D") >= O.OUTLINE_DROP_SHARE


def test_no_outline_is_todays_preview_byte_for_byte():
    img, outline = _world()
    base = _anywhere(img, params=_params())
    for kw in (dict(params=_params(), outline_roi=outline),
               dict(params=_params(outline="none"), outline_roi=outline),
               dict(params=_params(outline="none", bridge_um=16.0))):
        res = _anywhere(img, **kw)
        assert res.ok and res.refusal is None
        assert np.array_equal(res.result, base.result)
        assert res.result.dtype == base.result.dtype
        assert list(res.ids) == list(base.ids)
        assert res.counts() == base.counts()
        assert res.n_outside_outline == 0
    assert list(base.ids) == [500, 501, 502, 503]


@pytest.mark.parametrize("mode", [AP.OUTLINE_SHRINK, AP.OUTLINE_CLEARED])
def test_anywhere_leaves_out_what_lies_mostly_outside_and_cuts_the_rest(mode):
    img, outline = _world()
    res = _anywhere(img, params=_params(outline=mode, bridge_um=10.0),
                    outline_roi=outline)
    assert res.ok, res.refusal
    # B and D (at least 70 % outside) are left out; A and C are kept,
    # renumbered from next_label without a gap
    assert list(res.ids) == [500, 501]
    assert res.n_outside_outline == 2
    o = _outline_box(outline)
    assert not np.any(res.result[~o]), "nothing past the outline"
    a = _ball(img.shape, CENTRES["A"], R_NUC)
    c = _ball(img.shape, CENTRES["C"], R_NUC)
    assert np.array_equal(res.result > 0, a | (c & o))
    # nothing dropped is called small or smoothed away
    assert res.n_below_min_size == 0 and res.n_emptied_by_smooth == 0
    note = AP.counts_note(res.counts(), source=res.source, preset=res.preset)
    assert note == AP.ADD_NOTE_NUCLEI_OUTSIDE_OUTLINE.format(n=2)


def test_anywhere_with_every_nucleus_outside_refuses_in_its_words():
    img, outline = _world(names=("B", "D"))
    res = _anywhere(img, params=_params(outline=AP.OUTLINE_SHRINK),
                    outline_roi=outline)
    assert res.status == "refused"
    assert res.refusal == AP.ADD_S1_NUCLEI_ALL_OUTSIDE
    res = _anywhere(img, params=_params(outline=AP.OUTLINE_CLEARED),
                    outline_roi=outline)
    assert res.refusal == AP.ADD_S1_NUCLEI_ALL_OUTSIDE_CLEARED
    # an empty box is still the empty box's refusal
    empty = np.zeros_like(img)
    res = _anywhere(empty, params=_params(outline=AP.OUTLINE_SHRINK),
                    outline_roi=outline)
    assert res.refusal == AP.ADD_S1_NOTHING_NUCLEI


def _vol(name):
    """A box-local centre as volume ZYX (the markers are volume points)."""
    starts = tuple(int(sl.start) for sl in ROI)
    return tuple(int(c) + s for c, s in zip(CENTRES[name], starts))


def _markers(img, names, ids, **kw):
    return AP.run_add_preview(
        preset="nuclei", roi=ROI, existing_roi=np.zeros(img.shape, np.int32),
        next_label=500, spacing=ISO, image_roi=img,
        click_points=[_vol(n) for n in names], reserved_ids=list(ids), **kw)


def test_the_markers_only_cut_and_never_drop_a_clicked_nucleus():
    img, outline = _world()
    p = _params(look_for=AP.LOOK_MARKERS, click_radius_um=1.0,
                outline=AP.OUTLINE_SHRINK, bridge_um=10.0)
    res = _markers(img, ("A", "C", "D"), (101, 102, 103), params=p,
                   outline_roi=outline)
    assert res.ok, res.refusal
    # D lies at least 70 % outside: Anywhere drops it, a click keeps what
    # lies inside
    assert sorted(res.ids) == [101, 102, 103]
    o = _outline_box(outline)
    assert not np.any(res.result[~o])
    d = _ball(img.shape, CENTRES["D"], R_NUC)
    assert np.array_equal(res.result == 103, d & o)
    assert res.n_outside_outline == 0
    assert res.marker_why == {}


def test_a_marker_whose_nucleus_the_cut_empties_is_marked_and_counted():
    img, outline = _world()
    p = _params(look_for=AP.LOOK_MARKERS, click_radius_um=1.0,
                outline=AP.OUTLINE_SHRINK, bridge_um=10.0)
    res = _markers(img, ("A", "B"), (101, 102), params=p, outline_roi=outline)
    assert res.ok, res.refusal
    assert list(res.ids) == [101]
    assert res.marker_why == {102: "outline"}
    assert res.n_outside_outline == 1
    assert AP.marker_why_tip("outline") == AP.ADD_MARKER_OUTLINE_TIP
    assert AP.counts_note(res.counts(), preset="nuclei") == (
        AP.ADD_NOTE_NUCLEI_OUTSIDE_OUTLINE_ONE)
    # every marker's nucleus outside: the outline's refusal, rows marked
    res = _markers(img, ("B",), (102,), params=p, outline_roi=outline)
    assert res.status == "refused"
    assert res.refusal == AP.ADD_S1_NUCLEI_ALL_OUTSIDE
    assert res.marker_why == {102: "outline"}
    res = _markers(img, ("B",), (102,), outline_roi=outline,
                   params=dict(p, outline=AP.OUTLINE_CLEARED))
    assert res.refusal == AP.ADD_S1_NUCLEI_ALL_OUTSIDE_CLEARED


def test_the_markers_with_no_outline_are_unchanged():
    img, outline = _world()
    p = _params(look_for=AP.LOOK_MARKERS, click_radius_um=1.0)
    base = _markers(img, ("A", "B"), (101, 102), params=p)
    for kw in (dict(params=p, outline_roi=outline),
               dict(params=dict(p, outline="none"), outline_roi=outline)):
        res = _markers(img, ("A", "B"), (101, 102), **kw)
        assert np.array_equal(res.result, base.result)
        assert list(res.ids) == list(base.ids) == [101, 102]
        assert res.marker_why == base.marker_why == {}


# --- after the cut: one piece per nucleus, never under Smallest object ---
#: a small box at the origin (its own outline, the box's shape)
CUT_BOX = (10, 30, 30)
CUT_ROI = tuple(slice(0, s) for s in CUT_BOX)
#: a 6 x 20 x 4 bright slab (480 voxels), y 5..24
SLAB = (slice(2, 8), slice(5, 25), slice(10, 14))


def _pieces(labels, i):
    """The sizes of the 6-connected pieces of id ``i``, largest first."""
    from scipy import ndimage as ndi

    cc, n = ndi.label(np.asarray(labels) == i)
    return sorted(np.bincount(cc.ravel())[1:].tolist(), reverse=True)


def _cut_run(img, outline, **kw):
    clicks = kw.pop("clicks", None)
    extra = {}
    if clicks is not None:
        extra = dict(click_points=[c for c, _i in clicks],
                     reserved_ids=[i for _c, i in clicks])
    return AP.run_add_preview(
        preset="nuclei", roi=CUT_ROI,
        existing_roi=np.zeros(CUT_BOX, np.int32), next_label=500,
        spacing=ISO, image_roi=img, outline_roi=outline,
        params=_params(outline=AP.OUTLINE_SHRINK, separate=False, **kw),
        **extra)


def _slab_world(cut_y):
    img = np.zeros(CUT_BOX, np.float32)
    img[SLAB] = 1.0
    outline = np.ones(CUT_BOX, bool)
    outline[:, cut_y[0]:cut_y[1], :] = False
    return img, outline


def test_anywhere_a_nucleus_the_cut_trims_under_smallest_object_is_left_out():
    """A 300-voxel nucleus two thirds outside (under the 70 % drop) is cut to
    100 voxels: under Smallest object (150) it is left out and counted with
    the nuclei outside, never offered under the floor."""
    img = np.zeros(CUT_BOX, np.float32)
    img[:, 10:20, 5:8] = 1.0                # 300 voxels, x 5..7
    outline = np.zeros(CUT_BOX, bool)
    outline[:, :, :6] = True                # x 5 stays: 100 voxels
    res = _cut_run(img, outline, min_size=150)
    assert res.status == "refused"
    assert res.refusal == AP.ADD_S1_NUCLEI_ALL_OUTSIDE
    # with a nucleus wholly inside beside it: that one is kept, renumbered
    # without a gap, and the note says one nucleus lay outside
    img[:, 22:28, 0:5] = 1.0                # 300 voxels, inside
    res = _cut_run(img, outline, min_size=150)
    assert res.ok, res.refusal
    assert list(res.ids) == [500]
    assert res.sizes == {500: 300}
    assert res.n_outside_outline == 1
    assert res.n_below_min_size == 0
    assert AP.counts_note(res.counts(), preset="nuclei") == (
        AP.ADD_NOTE_NUCLEI_OUTSIDE_OUTLINE_ONE)


def test_anywhere_a_nucleus_the_cut_splits_keeps_its_largest_piece():
    img, outline = _slab_world((10, 16))    # pieces y 5..9 and y 16..24
    res = _cut_run(img, outline)
    assert res.ok, res.refusal
    assert list(res.ids) == [500]
    assert _pieces(res.result, 500) == [6 * 9 * 4]
    assert not np.any(res.result[:, :16, :])
    # the piece the floor leaves is the one kept: under it, left out
    res = _cut_run(img, outline, min_size=6 * 9 * 4 + 1)
    assert res.refusal == AP.ADD_S1_NUCLEI_ALL_OUTSIDE
    # two equal pieces: one piece, one id
    img, outline = _slab_world((12, 18))
    res = _cut_run(img, outline)
    assert res.ok and list(res.ids) == [500]
    assert _pieces(res.result, 500) == [6 * 7 * 4]


def test_the_markers_keep_the_piece_holding_the_marker_ball():
    """On the markers the piece holding the marker's ball is kept, even the
    smaller one; with the ball cut away, the largest piece; then Smallest
    object, the row marked "outline"."""
    img, outline = _slab_world((10, 16))
    res = _cut_run(img, outline, look_for=AP.LOOK_MARKERS,
                   click_radius_um=1.0, clicks=[((5, 7, 12), 77)])
    assert res.ok, res.refusal
    assert list(res.ids) == [77]
    assert _pieces(res.result, 77) == [6 * 5 * 4]
    assert res.result[5, 7, 12] == 77
    assert res.marker_why == {}
    # the click in the middle, the cut over y 8..21
    # leaves two 72-voxel pieces, neither holding the ball
    img, outline = _slab_world((8, 22))
    res = _cut_run(img, outline, look_for=AP.LOOK_MARKERS,
                   click_radius_um=1.0, clicks=[((5, 15, 12), 77)])
    assert res.ok, res.refusal
    assert _pieces(res.result, 77) == [72]
    res = _cut_run(img, outline, look_for=AP.LOOK_MARKERS,
                   click_radius_um=1.0, min_size=100,
                   clicks=[((5, 15, 12), 77)])
    assert res.status == "refused"
    assert res.refusal == AP.ADD_S1_NUCLEI_ALL_OUTSIDE
    assert res.marker_why == {77: "outline"}
    assert res.n_outside_outline == 1 and res.n_below_min_size == 0


def test_a_marker_whose_cut_sliver_the_smoothing_empties_is_outline():
    """D lies almost wholly outside: the cut leaves a sliver the smoothing
    empties -- its row says the outline, not the size."""
    img, outline = _world()
    p = _params(look_for=AP.LOOK_MARKERS, click_radius_um=1.0,
                outline=AP.OUTLINE_SHRINK, bridge_um=10.0, auto_smooth=True)
    res = _markers(img, ("A", "D"), (101, 103), params=p,
                   outline_roi=outline)
    assert res.ok, res.refusal
    assert list(res.ids) == [101]
    assert res.marker_why == {103: "outline"}
    assert res.n_outside_outline == 1
    assert res.n_emptied_by_smooth == 0
    # without the outline the same smoothing keeps D: "small" stays the
    # reason for what the smoothing alone empties
    res = _markers(img, ("A", "D"), (101, 103), params=dict(p, outline="none"))
    assert sorted(res.ids) == [101, 103]


def _mixed_world():
    """Four 5-voxel cubes outside the outline (x >= 20) and one 2-voxel cube
    inside it, with their clicks."""
    img = np.zeros(CUT_BOX, np.float32)
    clicks = []
    for k, y in enumerate((1, 8, 15, 22)):
        img[2:7, y:y + 5, 22:27] = 1.0
        clicks.append(((4, y + 2, 24), 200 + k))
    img[4:6, 4:6, 4:6] = 1.0
    clicks.append(((4, 4, 4), 300))
    outline = np.zeros(CUT_BOX, bool)
    outline[:, :, :20] = True
    return img, outline, clicks


def test_a_mixed_refusal_says_the_reason_most_markers_share():
    img, outline, clicks = _mixed_world()
    res = _cut_run(img, outline, look_for=AP.LOOK_MARKERS,
                   click_radius_um=1.0, min_size=50, clicks=clicks)
    assert res.status == "refused"
    assert res.marker_why == {200: "outline", 201: "outline",
                              202: "outline", 203: "outline", 300: "small"}
    # four markers emptied by the outline, one small: the outline's words
    assert res.refusal == AP.ADD_S1_NUCLEI_ALL_OUTSIDE
    # one of each: a tie goes in the order (the size floor first)
    res = _cut_run(img, outline, look_for=AP.LOOK_MARKERS,
                   click_radius_um=1.0, min_size=50,
                   clicks=[clicks[0], clicks[-1]])
    assert res.marker_why == {200: "outline", 300: "small"}
    assert res.refusal == AP.ADD_S1_ALL_FILTERED


def test_the_counts_note_says_the_nuclei_words_on_nuclei_only():
    counts = {k: 0 for k, _m, _o in AP._COUNT_PARTS}
    counts["n_outside_outline"] = 3
    assert AP.counts_note(counts) == AP.ADD_NOTE_OUTSIDE_OUTLINE.format(n=3)
    assert AP.counts_note(counts, preset="cells") == (
        AP.ADD_NOTE_OUTSIDE_OUTLINE.format(n=3))
    assert AP.counts_note(counts, preset="nuclei") == (
        AP.ADD_NOTE_NUCLEI_OUTSIDE_OUTLINE.format(n=3))
    # the parts keep their order: the outline's part after the dim one
    counts.update(n_markers_dim=2, n_below_min_size=1)
    parts = AP.counts_note(counts, preset="nuclei").split(
        AP.ADD_NOTE_COUNTS_SEP)
    assert parts == [AP.ADD_NOTE_MARKERS_DIM.format(n=2),
                     AP.ADD_NOTE_NUCLEI_OUTSIDE_OUTLINE.format(n=3),
                     "1 object was below Smallest object"]
    assert set(AP._COUNT_PARTS_NUCLEI) == {"n_outside_outline"}


def test_the_new_sentences_and_their_singulars():
    assert AP.count_phrase(1, AP.ADD_NOTE_NUCLEI_OUTSIDE_OUTLINE_ONE,
                           AP.ADD_NOTE_NUCLEI_OUTSIDE_OUTLINE) == (
        "1 nucleus lay outside the tissue outline and was left out")
    assert AP.ADD_NOTE_NUCLEI_OUTSIDE_OUTLINE.format(n=2) == (
        "2 nuclei lay outside the tissue outline and were left out")
    for key, text in ((AP.OUTLINE_CLEARED, AP.ADD_HELP_NUC_OUTLINE_CLEARED),
                      (AP.OUTLINE_SHRINK, AP.ADD_HELP_NUC_OUTLINE_SHRINK),
                      (AP.OUTLINE_NONE, AP.ADD_HELP_NUC_OUTLINE_NONE)):
        assert AP.nuclei_outline_help(key) == text
    assert AP.nuclei_outline_summary(AP.OUTLINE_NONE) == "Tissue outline: none"
    assert AP.nuclei_outline_summary(AP.OUTLINE_SHRINK) == (
        "Tissue outline: shrink-wrap")
    assert AP.nuclei_outline_summary(AP.OUTLINE_CLEARED) == (
        "Tissue outline: cleared objects")
    # the shrink-wrap help says a nuclear stain alone leaves lone nuclei out
    assert "lone nuclei can fall outside it" in AP.ADD_HELP_NUC_OUTLINE_SHRINK
    assert "under Walls" not in AP.ADD_S1_OUTLINE_FAILED_NUCLEI
    assert AP.ADD_S1_OUTLINE_FAILED_NUCLEI.startswith(
        "The tissue outline could not be measured")
    assert "Lower it if dim tissue" in AP.ADD_HELP_DARK_BELOW_NUCLEI


def test_one_shrink_wrap_for_both_presets_at_a_fixed_spacing():
    """The Nuclei job hands no dark threshold when Cells has not measured
    one; the outline then takes it from the same strided sample -- one
    mask either way."""
    from packages.epicycle_annotator.reshape.region import (
        patch_dark_threshold,
    )

    rng = np.random.default_rng(1)
    ch = {"nuc": (_dense_field() * 100.0
                  + rng.random((30, 100, 100)) * 5.0).astype(np.float32),
          "mem": (rng.random((30, 100, 100)) * 40.0).astype(np.float32)}
    for sp in ((1.0, 1.0, 1.0), (2.0, 0.5, 0.5)):
        job = {"mode": "shrink", "channels": ch, "factor": 0.5,
               "bridge_um": 10.0}
        thr = patch_dark_threshold(ch, spacing=sp, stride=AP.dark_stride(
            ch["nuc"].shape))
        with_cells = AP.measure_outline(job, spacing=sp, dark_threshold=thr)
        alone = AP.measure_outline(job, spacing=sp, dark_threshold=None)
        assert np.array_equal(with_cells, alone)


def _dense_field():
    shape = (30, 100, 100)
    img = np.zeros(shape, np.float32)
    for c in _CLUSTER + _LONE:
        img[_ball(shape, c, 3)] = 1.0
    return img


_CLUSTER = [(z, y, x) for z in (8, 16, 22) for y in range(14, 50, 8)
            for x in range(14, 50, 8)]
_LONE = [(15, 85, 85)]


def test_a_nuclear_stain_alone_keeps_a_crowded_field_and_leaves_a_lone_one():
    rng = np.random.default_rng(0)
    img = _dense_field() * 100.0 + rng.random((30, 100, 100)) * 5.0
    out = AP.measure_outline({"mode": "shrink", "channels": {"nuc": img},
                              "factor": 0.5,
                              "bridge_um": AP.ADD_BRIDGE_UM_ONE_CHANNEL},
                             spacing=ISO)
    assert all(out[c] for c in _CLUSTER)
    assert not out[_LONE[0]]


def test_box_info_measures_the_outline_on_nuclei_too():
    img, _outline = _world()
    ch = {"nuc": np.pad(img, ((2, 2), (5, 5), (5, 5))) * 100.0}
    job = {"mode": "shrink", "channels": ch, "factor": 0.5, "bridge_um": 10.0}
    kw = dict(existing_roi=np.zeros(img.shape, np.int32), dark_sources=None,
              dark_rois=None, spacing=ISO, stride=1, dark_cache=None,
              source="paint", run_guide_roi=None, nuclei_roi=None,
              roi=ROI, image_roi=img,
              params={"preset": "nuclei", "outline": "shrink",
                      "bridge_um": 10.0})
    info = AP.add_box_info(outline_job=job, want_shell=True, **kw)
    assert info.status == "ok"
    assert info.outline is not None
    assert np.array_equal(info.outline, AP.measure_outline(job, spacing=ISO))
    assert np.array_equal(info.outline_shell,
                          AP.outline_display_shell(info.outline, ISO))
    assert info.run_count is None and info.histogram is not None


# --------------------------------------------------------------------------- #
# the page
# --------------------------------------------------------------------------- #
def _pick(w, key):
    keys = list(w._add_look_keys)
    w.add_look_for_combo.setCurrentIndex(keys.index(key))


def _histogram(w):
    w.update_add_ui_state(histogram_data={
        "values": list(np.linspace(0.0, 1.0, 200)), "n_bins": 32})


def _plot(w):
    return (getattr(w, "_add_plot_widget", None)
            or getattr(w, "_add_plot_placeholder", None))


def _nuc_page(w, *, look=ANYWHERE, whole=False, in_force=False):
    reset_add(w)
    w.update_add_ui_state(look_for_items=[
        (AP.look_inside_text("default", "cells"), True,
         (AP.LOOK_INSIDE, "default"), False)])
    w.update_add_ui_state(box_active=True, whole_patch=whole,
                          cleared_in_force=in_force)
    _histogram(w)
    _pick(w, look)


def _back(w):
    w.add_look_for_combo.setCurrentIndex(0)
    w.add_nuc_show_outline_cb.setChecked(False)
    w.add_nuc_outline_combo.setCurrentIndex(
        AP.OUTLINE_KEYS.index(AP.OUTLINE_NONE))
    w.add_nuc_outline_btn.setChecked(False)
    w.add_markers_btn.setChecked(False)
    w.add_walls_btn.setChecked(False)
    w._add_advanced_toggle.setChecked(False)
    w.add_outline_combo.setCurrentIndex(AP.OUTLINE_KEYS.index(
        AP.OUTLINE_NONE))
    w.add_dark_outside_cb.setChecked(True)
    w.update_add_ui_state(box_active=False, whole_patch=False,
                          cleared_in_force=False, default_spacing=False,
                          candidate_ids=[], candidate_sizes={},
                          candidate_edge_ids=set(), look_for_items=[],
                          click_count=0, click_markers=[])
    reset_add(w)


def test_the_card_is_on_anywhere_with_a_box_and_starts_at_no_outline(widget):
    _nuc_page(widget)
    try:
        c = widget.add_nuc_outline_combo
        assert [c.itemText(i) for i in range(c.count())] == [
            AP.ADD_OUTLINE_CLEARED_GREY, AP.ADD_OUTLINE_SHRINK,
            AP.ADD_OUTLINE_NONE]
        assert widget.get_add_nuc_outline() == AP.OUTLINE_NONE
        assert not c.model().item(0).isEnabled()
        assert not widget._add_cells_container.isHidden()
        assert not widget.add_nuc_outline_btn.isHidden()
        assert not widget.add_nuc_outline_btn.isChecked()
        assert widget._add_nuc_outline_container.isHidden()
        # the Cells cards are Cells' only
        assert widget.add_markers_btn.isHidden()
        assert widget.add_walls_btn.isHidden()
        assert not _plot(widget).isHidden()
        assert str(widget.add_nuc_outline_btn.property(
            "_epicycle_chevron_title")) == AP.nuclei_outline_summary("none")
        # the snapshot is the one it always was
        p = widget.get_add_params()
        assert "outline" not in p and "bridge_um" not in p
        # no box: no card
        widget.update_add_ui_state(box_active=False)
        assert widget.add_nuc_outline_btn.isHidden()
    finally:
        _back(widget)


def test_choosing_the_shrink_wrap_adds_exactly_two_keys(widget):
    _nuc_page(widget)
    try:
        before = widget.get_add_params()
        widget.add_nuc_outline_btn.setChecked(True)
        widget.add_nuc_outline_combo.setCurrentIndex(
            AP.OUTLINE_KEYS.index(AP.OUTLINE_SHRINK))
        assert widget.get_add_nuc_outline() == AP.OUTLINE_SHRINK
        p = widget.get_add_params()
        assert p == dict(before, outline=AP.OUTLINE_SHRINK,
                         bridge_um=float(widget.add_bridge_sb.value()))
        assert "add_nuc_outline_combo" in widget._add_touched
        assert widget._add_nuc_outline_help.text() == (
            AP.ADD_HELP_NUC_OUTLINE_SHRINK)
        # the help line is off the page (the take-off under the ceiling):
        # each item's tooltip says what that choice does
        from qtpy.QtCore import Qt

        assert widget._add_nuc_outline_help.isHidden()
        c = widget.add_nuc_outline_combo
        for i, key in enumerate(AP.OUTLINE_KEYS):
            assert c.itemData(i, Qt.ToolTipRole) == (
                AP.nuclei_outline_help(key))
        assert str(widget.add_nuc_outline_btn.property(
            "_epicycle_chevron_title")) == AP.nuclei_outline_summary("shrink")
        # on the markers: the same two keys after look_for
        _pick(widget, MARKERS)
        p = widget.get_add_params()
        assert p["outline"] == AP.OUTLINE_SHRINK and "bridge_um" in p
        assert p["look_for"] == AP.LOOK_MARKERS
        # Inside each cell: never an outline (the cells confine the nuclei)
        _pick(widget, (AP.LOOK_INSIDE, "default"))
        p = widget.get_add_params()
        assert "outline" not in p and "bridge_um" not in p
        assert widget.add_nuc_outline_btn.isHidden()
        assert widget._add_nuc_outline_container.isHidden()
    finally:
        _back(widget)


def test_the_cells_combo_and_the_nuclei_combo_are_apart(widget):
    _nuc_page(widget)
    try:
        widget.add_nuc_outline_combo.setCurrentIndex(
            AP.OUTLINE_KEYS.index(AP.OUTLINE_SHRINK))
        widget.set_add_preset("cells")
        assert widget.get_add_outline() == AP.OUTLINE_NONE
        assert widget.get_add_params()["outline"] == AP.OUTLINE_NONE
        assert widget.add_nuc_outline_btn.isHidden()
        widget.add_walls_btn.setChecked(True)
        widget.update_add_ui_state(cleared_in_force=True)
        widget.add_outline_combo.setCurrentIndex(
            AP.OUTLINE_KEYS.index(AP.OUTLINE_CLEARED))
        widget.set_add_preset("nuclei")
        assert widget.get_add_nuc_outline() == AP.OUTLINE_SHRINK
        assert widget.get_add_params()["outline"] == AP.OUTLINE_SHRINK
        widget.set_add_preset("cells")
        assert widget.get_add_outline() == AP.OUTLINE_CLEARED
        # a Walls card left open on Cells: Nuclei shows no Walls body
        widget.set_add_preset("nuclei")
        assert widget._add_walls_container.isHidden()
    finally:
        widget.set_add_preset("nuclei")
        _back(widget)


def test_one_card_open_at_a_time_and_the_histogram_folds(widget):
    _nuc_page(widget)
    try:
        widget.add_nuc_outline_btn.setChecked(True)
        assert not widget._add_nuc_outline_container.isHidden()
        assert _plot(widget).isHidden(), "an open Nuclei card folds it"
        assert widget._add_threshold_help.isHidden()
        widget.add_nuc_outline_btn.setChecked(False)
        assert not _plot(widget).isHidden()
        # on the markers: Markers opens when picked, Tissue outline closes
        widget.add_nuc_outline_btn.setChecked(True)
        _pick(widget, MARKERS)
        assert widget.add_markers_btn.isChecked()
        assert not widget.add_nuc_outline_btn.isChecked()
        assert not widget._add_markers_container.isHidden()
        assert widget._add_nuc_outline_container.isHidden()
        widget.add_nuc_outline_btn.setChecked(True)
        assert not widget.add_markers_btn.isChecked()
        assert widget._add_markers_container.isHidden()
        assert not widget._add_nuc_outline_container.isHidden()
        widget.add_markers_btn.setChecked(True)
        assert not widget.add_nuc_outline_btn.isChecked()
    finally:
        _back(widget)


def test_a_cells_card_left_open_is_not_closed_by_the_nuclei_card(widget):
    reset_add(widget)
    widget.set_add_preset("cells")
    widget.update_add_ui_state(box_active=True)
    widget.add_walls_btn.setChecked(True)
    try:
        widget.set_add_preset("nuclei")
        _histogram(widget)
        widget.add_nuc_outline_btn.setChecked(True)
        assert widget.add_walls_btn.isChecked(), "hidden: left as it was"
        widget.set_add_preset("cells")
        assert widget.add_walls_btn.isChecked()
        assert not widget._add_walls_container.isHidden()
        assert widget._add_nuc_outline_container.isHidden()
    finally:
        widget.set_add_preset("nuclei")
        _back(widget)


def test_never_on_a_whole_patch_or_over_candidates(widget):
    _nuc_page(widget, whole=True)
    try:
        assert widget.add_nuc_outline_btn.isHidden()
        assert widget._add_cells_container.isHidden()
        widget.update_add_ui_state(whole_patch=False)
        widget.add_nuc_outline_btn.setChecked(True)
        widget.add_nuc_outline_combo.setCurrentIndex(
            AP.OUTLINE_KEYS.index(AP.OUTLINE_SHRINK))
        widget.update_add_ui_state(candidate_ids=[5], candidate_sizes={5: 9},
                                   candidate_edge_ids=set())
        assert widget.add_nuc_outline_btn.isHidden()
        assert widget._add_nuc_outline_container.isHidden()
        assert widget._add_cells_container.isHidden()
        # over candidates the spacing amber does not show
        widget.update_add_ui_state(default_spacing=True)
        assert widget.add_spacing_note.isHidden()
    finally:
        _back(widget)


def test_show_and_its_legend_need_an_outline(widget):
    _nuc_page(widget)
    try:
        widget.add_nuc_outline_btn.setChecked(True)
        assert widget.add_nuc_show_outline_cb.isHidden()
        widget.add_nuc_outline_combo.setCurrentIndex(
            AP.OUTLINE_KEYS.index(AP.OUTLINE_SHRINK))
        assert not widget.add_nuc_show_outline_cb.isHidden()
        assert widget._add_nuc_show_outline_legend.isHidden()
        widget.add_nuc_show_outline_cb.setChecked(True)
        assert widget._add_nuc_show_outline_legend.text() == (
            AP.ADD_HELP_SHOW_OUTLINE)
        widget.update_add_ui_state(outline_pending=True)
        assert widget._add_nuc_show_outline_legend.text() == (
            AP.ADD_OUTLINE_PENDING)
        # the Cells tick and legend are Cells' own
        assert not widget.add_show_outline_cb.isChecked()
        widget.add_nuc_outline_combo.setCurrentIndex(
            AP.OUTLINE_KEYS.index(AP.OUTLINE_NONE))
        assert widget.add_nuc_show_outline_cb.isHidden()
        assert widget._add_nuc_show_outline_legend.isHidden()
    finally:
        widget.update_add_ui_state(outline_pending=False)
        _back(widget)


def test_the_cleared_item_waits_for_a_clear(widget):
    _nuc_page(widget)
    try:
        c = widget.add_nuc_outline_combo
        assert not c.model().item(0).isEnabled()
        widget.update_add_ui_state(cleared_in_force=True)
        assert c.itemText(0) == AP.ADD_OUTLINE_CLEARED
        assert c.model().item(0).isEnabled()
        c.setCurrentIndex(0)
        assert widget._add_nuc_outline_help.text() == (
            AP.ADD_HELP_NUC_OUTLINE_CLEARED)
        widget.add_nuc_outline_btn.setChecked(True)
        assert widget.get_add_params()["outline"] == AP.OUTLINE_CLEARED
        # the help of No outline
        c.setCurrentIndex(2)
        assert widget._add_nuc_outline_help.text() == (
            AP.ADD_HELP_NUC_OUTLINE_NONE)
    finally:
        _back(widget)


def test_a_default_never_turns_the_outline_on(widget):
    """The touched set forgotten (a full exit) or a whole-patch and Clear
    push never picks an outline for the user on Nuclei."""
    _nuc_page(widget, in_force=True)
    try:
        widget.add_nuc_outline_combo.setCurrentIndex(
            AP.OUTLINE_KEYS.index(AP.OUTLINE_SHRINK))
        widget.clear_add_touched()
        widget.update_add_ui_state(box_active=True)
        assert widget.get_add_nuc_outline() == AP.OUTLINE_NONE
        widget.update_add_ui_state(whole_patch=True, cleared_in_force=True)
        assert widget.get_add_nuc_outline() == AP.OUTLINE_NONE
        # a Cells pick does not reach Nuclei
        widget.update_add_ui_state(whole_patch=False)
        widget.set_add_preset("cells")
        widget.add_outline_combo.setCurrentIndex(
            AP.OUTLINE_KEYS.index(AP.OUTLINE_SHRINK))
        widget.set_add_preset("nuclei")
        assert widget.get_add_nuc_outline() == AP.OUTLINE_NONE
        assert "outline" not in widget.get_add_params()
    finally:
        widget.set_add_preset("nuclei")
        _back(widget)


def test_dark_below_and_bridge_gaps_under_advanced_with_the_shrink_wrap(
        widget):
    from napari_epicycle.annotator.advanced_features_widget import (
        ADD_HELP_LINES,
    )

    _nuc_page(widget)
    widget._add_advanced_toggle.setChecked(True)
    try:
        assert widget._add_cells_adv.isHidden()
        # the persisted dark tick is off: Dark below is live all the same
        widget.add_dark_outside_cb.setChecked(False)
        widget.add_nuc_outline_combo.setCurrentIndex(
            AP.OUTLINE_KEYS.index(AP.OUTLINE_SHRINK))
        assert not widget._add_cells_adv.isHidden()
        assert not widget._add_dark_below_row.isHidden()
        assert not widget._add_bridge_row.isHidden()
        assert widget.add_dark_factor_sb.isEnabled()
        assert widget.add_dark_factor_sb.toolTip() == (
            AP.ADD_HELP_DARK_BELOW_NUCLEI)
        for hidden in (widget._add_dark_reach_row, widget.add_dark_outside_cb,
                       widget._add_dark_help, widget._add_grow_row,
                       widget._add_spread_adv, widget._add_core_erode_row,
                       widget._add_whole_fill_label,
                       widget.add_include_membrane_cb,
                       widget.add_dark_off_note):
            assert hidden.isHidden()
        assert AP.ADD_DARK_BY_OUTLINE not in widget._add_advanced_help.text()
        title = str(widget._add_advanced_toggle.property(
            "_epicycle_chevron_title"))
        assert "bridge" in title
        # the cleared outline: Dark below and Bridge gaps do not shape it
        widget.update_add_ui_state(cleared_in_force=True)
        widget.add_nuc_outline_combo.setCurrentIndex(
            AP.OUTLINE_KEYS.index(AP.OUTLINE_CLEARED))
        assert widget._add_cells_adv.isHidden()
        # back on Cells the tooltip is Cells' own
        widget.set_add_preset("cells")
        assert widget.add_dark_factor_sb.toolTip() == (
            ADD_HELP_LINES["add_dark_factor_sb"])
        assert not widget.add_dark_outside_cb.isHidden()
        assert not widget._add_dark_reach_row.isHidden()
    finally:
        widget.set_add_preset("nuclei")
        _back(widget)


def test_the_default_spacing_amber_shows_with_an_outline(widget):
    _nuc_page(widget)
    try:
        widget.update_add_ui_state(default_spacing=True)
        assert widget.add_spacing_note.isHidden(), "Anywhere, no outline"
        widget.add_nuc_outline_combo.setCurrentIndex(
            AP.OUTLINE_KEYS.index(AP.OUTLINE_SHRINK))
        assert not widget.add_spacing_note.isHidden()
    finally:
        _back(widget)


# --------------------------------------------------------------------------- #
# the orchestrator (the Qt-free harness)
# --------------------------------------------------------------------------- #
class _Tick:
    def __init__(self):
        self.on = False

    def isChecked(self):              # noqa: N802 - Qt name
        return self.on

    def setChecked(self, on):         # noqa: N802 - Qt name
        self.on = bool(on)

    def blockSignals(self, _b):       # noqa: N802 - Qt name
        return False


class _NucAdv(FakeAdv):
    """The Nuclei page with its Tissue outline, as far as the orchestrator
    reads it: the snapshot carries ``outline`` / ``bridge_um`` only with an
    outline chosen, as the real page."""

    def __init__(self):
        super().__init__()
        self.params.update(dark_otsu_factor=0.5)
        self.nuc_outline = AP.OUTLINE_NONE
        self.cells_outline = AP.OUTLINE_NONE
        self.add_nuc_show_outline_cb = _Tick()
        self.add_show_outline_cb = _Tick()
        self._add_touched = set()

    def get_add_params(self):
        p = dict(self.params)
        if self.preset == "cells":
            p.update(outline=self.cells_outline, bridge_um=10.0,
                     growth="flood", wall_cost=40.0, growth_limit=28.0,
                     include_membrane=True)
        elif (self.look[0] != AP.LOOK_INSIDE
              and self.nuc_outline != AP.OUTLINE_NONE):
            p.update(outline=self.nuc_outline, bridge_um=10.0)
        if self.look[0] == AP.LOOK_MARKERS and self.preset != "cells":
            p["look_for"] = AP.LOOK_MARKERS
        return p

    def get_add_outline(self):
        return self.cells_outline

    def set_add_outline(self, key):
        self.cells_outline = key

    def get_add_nuc_outline(self):
        return self.nuc_outline

    def set_add_nuc_outline(self, key):
        self.nuc_outline = key


class _Layer:
    def __init__(self):
        self.visible = False


def _channels():
    img, _o = _world()
    nuc = np.zeros(PATCH, np.float32)
    nuc[ROI] = img * 100.0
    mem = np.full(PATCH, 3.0, np.float32)
    mem[:, :, :EDGE_X] = 60.0
    return {"nuc": nuc, "mem": mem}


@pytest.fixture
def h(monkeypatch):
    install_fakes(monkeypatch)
    ch = _channels()
    hh = Harness(np.zeros(PATCH, np.int32), images=ch)
    hh.adv = _NucAdv()
    hh.ctx["spacing_source"] = "qc_settings"
    hh.added = []
    hh.removed = []

    def _temp(key, array, **kw):
        layer = _Layer()
        hh.added.append((key, array, kw, layer))
        return layer

    hh._add_temp_layer = _temp
    hh._remove_temp_layer = lambda key: hh.removed.append(key)
    hh.place_box(ROI)
    hh._add_threshold_value = 50.0
    return hh


def _land_info(h):
    for _ in range(3):
        th = getattr(h, "_add_info_thread", None)
        if th is None:
            return
        th.run()
        th.end()


def test_the_mode_is_none_unless_chosen_and_never_on_inside_or_whole(h):
    assert h._add_outline_mode() == AP.OUTLINE_NONE
    assert h._add_outline_key() is None
    h.adv.nuc_outline = AP.OUTLINE_SHRINK
    assert h._add_outline_mode() == AP.OUTLINE_SHRINK
    h.adv.look = MARKERS
    assert h._add_outline_mode() == AP.OUTLINE_SHRINK
    h.adv.look = (AP.LOOK_INSIDE, "r")
    assert h._add_outline_mode() == AP.OUTLINE_NONE
    h.adv.look = ANYWHERE
    h._add_whole_patch = True
    assert h._add_outline_mode() == AP.OUTLINE_NONE
    h._add_whole_patch = False
    # a page without the Nuclei card (an older panel): none
    plain = FakeAdv()
    h.adv, keep = plain, h.adv
    try:
        assert h._add_outline_mode() == AP.OUTLINE_NONE
    finally:
        h.adv = keep


def test_the_gather_sends_nothing_new_without_an_outline(h):
    kw, why = h._add_gather_preview()
    assert why is None, why
    assert "outline_roi" not in kw
    assert "outline" not in kw["params"] and "bridge_um" not in kw["params"]
    h._request_add_box_info()
    w = h._add_info_worker
    assert "outline_job" not in w._kwargs and "outline" not in w._kwargs
    assert w._epicycle_outline_key is None
    assert w._kwargs["dark_cache"] is None
    _land_info(h)


def test_the_shrink_wrap_is_measured_once_and_handed_to_the_preview(h):
    h.adv.nuc_outline = AP.OUTLINE_SHRINK
    kw, why = h._add_gather_preview()
    assert kw is None and why == AP.ADD_OUTLINE_PENDING
    h._add_refresh_ui()
    assert h.adv.last("status_text") == AP.ADD_OUTLINE_PENDING
    assert h.adv.last("outline_pending") is True
    h._request_add_box_info()
    w = h._add_info_worker
    assert w._kwargs["outline_job"]["mode"] == AP.OUTLINE_SHRINK
    assert set(w._kwargs["outline_job"]["channels"]) == {"nuc", "mem"}
    assert w._epicycle_outline_key == h._add_outline_key()
    _land_info(h)
    assert h._add_cached_outline() is not None
    kw, why = h._add_gather_preview()
    assert why is None, why
    assert kw["outline_roi"] is h._add_cached_outline()
    assert kw["params"]["outline"] == AP.OUTLINE_SHRINK
    assert kw["params"]["bridge_um"] == 10.0
    # the engine runs with it: the nucleus outside the membrane is gone
    res = AP.run_add_preview(**kw)
    assert res.ok, res.refusal
    assert res.n_outside_outline >= 1


def test_one_measurement_serves_both_presets_and_a_changed_input_measures_again(
        h):
    h.adv.nuc_outline = AP.OUTLINE_SHRINK
    h._request_add_box_info()
    _land_info(h)
    nuc_key = h._add_outline_key()
    cached = h._add_cached_outline()
    assert cached is not None
    # Cells with the same inputs: the same key, the same mask, no new job
    h.adv.preset = "cells"
    h.adv.source = ("paint", None)
    h.adv.cells_outline = AP.OUTLINE_SHRINK
    assert h._add_outline_key() == nuc_key
    assert h._add_cached_outline() is cached
    h._add_prune_caches()
    assert h._add_cached_outline() is cached
    h._request_add_box_info()
    w = h._add_info_worker
    assert "outline_job" not in w._kwargs
    assert w._kwargs["outline"] is cached
    _land_info(h)
    # back on Nuclei, Dark below changed: dropped and measured again
    h.adv.preset = "nuclei"
    h.adv.params["dark_otsu_factor"] = 0.4
    h._add_prune_caches()
    assert h._add_cached_outline() is None
    h._add_info = None
    h._request_add_box_info()
    assert "outline_job" in h._add_info_worker._kwargs
    _land_info(h)


def test_the_cached_dark_threshold_is_handed_in_when_cells_left_one(h):
    h.adv.nuc_outline = AP.OUTLINE_SHRINK
    thr = {"t": 0.4, "lo": {"nuc": 0.0, "mem": 0.0},
           "hi": {"nuc": 100.0, "mem": 60.0}}
    h._add_dark_cache = {h._add_dark_key(): thr}
    h._request_add_box_info()
    assert h._add_info_worker._kwargs["dark_cache"] is thr
    _land_info(h)


def test_show_draws_the_skin_from_the_nuclei_tick(h):
    h.adv.nuc_outline = AP.OUTLINE_SHRINK
    assert not h._add_show_outline_wanted()
    h.adv.add_show_outline_cb.on = True          # the Cells tick: not read
    assert not h._add_show_outline_wanted()
    h.adv.add_show_outline_cb.on = False
    h.adv.add_nuc_show_outline_cb.on = True
    assert h._add_show_outline_wanted()
    h._on_add_show_outline(True)
    assert h._add_info_worker._kwargs.get("want_shell") is True
    _land_info(h)
    h._add_refresh_ui()
    key, array, kw, layer = h.added[-1]
    assert key == "add_outline" and kw["kind"] == "image"
    assert array is h._add_cached_outline(shell=True)
    assert layer.visible is True
    # a full exit unticks both presets' ticks
    h.adv.add_show_outline_cb.on = True
    h._add_drop_outline_full()
    assert h.adv.add_nuc_show_outline_cb.on is False
    assert h.adv.add_show_outline_cb.on is False
    assert "add_outline" in h.removed


def test_the_cleared_choice_falls_back_on_nuclei(h):
    h.adv.nuc_outline = AP.OUTLINE_CLEARED
    h._add_clear_records = {4: (7,)}
    live = {4}
    h._get_live_add_clears_cb = lambda: set(live)
    h._add_refresh_ui()
    assert h.adv.nuc_outline == AP.OUTLINE_CLEARED
    live.clear()
    h._add_clears_back = True
    h._add_refresh_ui()
    assert h.adv.nuc_outline == AP.OUTLINE_NONE
    assert h.adv.cells_outline == AP.OUTLINE_NONE
    assert h.adv.last("status_text") == AP.ADD_S1_OUTLINE_CLEARED_GONE.format(
        choice=AP.ADD_OUTLINE_NONE)


def test_a_failed_outline_is_said_in_the_nuclei_words(h):
    h.adv.nuc_outline = AP.OUTLINE_SHRINK
    h._add_outline_errors = {h._add_outline_key(): AP.ADD_S1_OUTLINE_FAILED}
    s = h._add_s1_state()
    assert s["outline_error"] == AP.ADD_S1_OUTLINE_FAILED_NUCLEI
    assert AP.preview_block(s) == AP.ADD_S1_OUTLINE_FAILED_NUCLEI
    h._add_refresh_ui()
    assert h.adv.last("outline_refused") is True


def test_the_markers_wait_for_a_marker_before_the_outline(h):
    h.adv.nuc_outline = AP.OUTLINE_SHRINK
    h.adv.look = MARKERS
    s = h._add_s1_state()
    assert AP.preview_block(s) == AP.ADD_S1_NO_MARKERS
    h._add_click_markers = [{"zyx": _vol("A"), "id": 1, "ticked": True}]
    s = h._add_s1_state()
    assert AP.preview_block(s) == AP.ADD_OUTLINE_PENDING


def test_half_size_is_in_the_key(h):
    h.adv.nuc_outline = AP.OUTLINE_SHRINK
    k1 = h._add_outline_key()
    h.ctx["downsample_factor"] = 2
    k2 = h._add_outline_key()
    assert k1 != k2 and k2[-1] == 2


def test_a_marker_the_outline_emptied_carries_its_tip(h):
    h._add_marker_why = {7: "outline"}
    h._add_click_markers = [{"zyx": (3, 3, 3), "id": 7, "ticked": True}]
    push = h._add_marker_push()
    assert push["click_absorbed_tips"] == {7: AP.ADD_MARKER_OUTLINE_TIP}


def test_the_result_note_is_said_in_the_nuclei_words(h):
    h.adv.nuc_outline = AP.OUTLINE_SHRINK
    h._request_add_box_info()
    _land_info(h)
    h._preview_add()
    th = h._add_thread
    th.run()
    th.end()
    res = h._add_preview_res
    assert res is not None and res.ok, h._add_transient
    notes = h.adv.last("notes")
    assert notes and notes[0][1].startswith(
        AP.count_phrase(res.n_outside_outline,
                        AP.ADD_NOTE_NUCLEI_OUTSIDE_OUTLINE_ONE,
                        AP.ADD_NOTE_NUCLEI_OUTSIDE_OUTLINE))


def test_the_remembered_state_never_turns_the_outline_on(h, widget):
    """The preset and the dark tick a workspace remembers (Nuclei, the tick
    off) leave the Nuclei outline at none."""
    reset_add(widget)
    h.adv = widget
    h._add_session_memory = {"last_preset": "nuclei", "dark_outside": False}
    try:
        h._add_restore_preset()
        assert widget.get_add_preset() == "nuclei"
        assert widget.get_add_nuc_outline() == AP.OUTLINE_NONE
        assert "outline" not in widget.get_add_params()
        assert h._add_outline_mode() == AP.OUTLINE_NONE
    finally:
        widget.add_dark_outside_cb.setChecked(True)
        reset_add(widget)


def test_the_real_page_and_the_gather_agree_on_the_outline(h, widget):
    """The real page, not a fake: with the shrink-wrap picked on Nuclei ▸
    Anywhere, a box carries it into the Preview; on a whole patch the page
    and the orchestrator both read no outline (no key in the snapshot, no
    Dark below / Bridge gaps under Advanced, no bridge in its summary)."""
    reset_add(widget)
    h.adv = widget
    widget.update_add_ui_state(look_for_items=[], box_active=True)
    _histogram(widget)
    widget.add_nuclei_channel_combo.setCurrentIndex(
        widget.add_nuclei_channel_combo.findText("nuc"))
    widget.add_nuc_outline_combo.setCurrentIndex(
        AP.OUTLINE_KEYS.index(AP.OUTLINE_SHRINK))
    widget._add_advanced_toggle.setChecked(True)
    try:
        assert h._add_outline_mode() == AP.OUTLINE_SHRINK
        h._request_add_box_info()
        _land_info(h)
        kw, why = h._add_gather_preview()
        assert why is None, why
        assert kw["params"]["outline"] == AP.OUTLINE_SHRINK
        assert kw["params"]["bridge_um"] == float(widget.add_bridge_sb.value())
        assert kw["outline_roi"] is h._add_cached_outline()
        assert not widget._add_dark_below_row.isHidden()
        assert "bridge" in str(widget._add_advanced_toggle.property(
            "_epicycle_chevron_title"))
        # a whole patch: no outline, on the page and in the orchestrator
        h._add_whole_patch = True
        widget.update_add_ui_state(whole_patch=True)
        assert h._add_outline_mode() == AP.OUTLINE_NONE
        p = widget.get_add_params()
        assert "outline" not in p and "bridge_um" not in p
        assert widget._add_dark_below_row.isHidden()
        assert widget._add_bridge_row.isHidden()
        assert "bridge" not in str(widget._add_advanced_toggle.property(
            "_epicycle_chevron_title"))
        kw, _why = h._add_gather_preview()
        assert kw is None or "outline" not in kw["params"]
        assert kw is None or "outline_roi" not in kw
        # the pick itself is kept for the next box
        assert widget.get_add_nuc_outline() == AP.OUTLINE_SHRINK
    finally:
        h._add_whole_patch = False
        widget._set_add_channel_blank(widget.add_nuclei_channel_combo)
        _back(widget)


def test_an_undo_seen_on_cells_is_still_said_back_on_nuclei(h):
    """Nuclei's pick is the cleared outline; on Cells the Clear is undone;
    back on Nuclei the pick falls back to No outline and S1 says why (the
    Cells refresh must not use up the "objects are back" word)."""
    h.adv.nuc_outline = AP.OUTLINE_CLEARED
    h._add_clear_records = {4: (7,)}
    live = {4}
    h._get_live_add_clears_cb = lambda: set(live)
    h._add_refresh_ui()
    assert h.adv.nuc_outline == AP.OUTLINE_CLEARED
    h.adv.preset = "cells"
    h.adv.source = ("paint", None)
    h._add_refresh_ui()
    live.clear()
    h._add_clears_back = True
    h._add_refresh_ui()
    assert h.adv.nuc_outline == AP.OUTLINE_CLEARED, "not Cells' to change"
    h.adv.preset = "nuclei"
    h._add_refresh_ui()
    assert h.adv.nuc_outline == AP.OUTLINE_NONE
    assert h.adv.last("status_text") == AP.ADD_S1_OUTLINE_CLEARED_GONE.format(
        choice=AP.ADD_OUTLINE_NONE)


# --------------------------------------------------------------------------- #
# the snapshot, the help lines, the guide
# --------------------------------------------------------------------------- #
def test_a_nuclei_outline_adds_only_outline_and_bridge(widget):
    """Tissue outline on Nuclei ▸ Anywhere (and ▸ markers) adds exactly two
    keys to the frozen Nuclei snapshot; with No outline -- the default --
    nothing is added; back at No outline the snapshot is the frozen one."""
    from .test_add_params_golden import NUCLEI_AT_DEFAULTS

    reset_add(widget)
    widget.update_add_ui_state(look_for_items=[], box_active=True)
    c = widget.add_nuc_outline_combo
    try:
        assert widget.get_add_nuc_outline() == AP.OUTLINE_NONE
        assert widget.get_add_params() == NUCLEI_AT_DEFAULTS
        c.setCurrentIndex(AP.OUTLINE_KEYS.index(AP.OUTLINE_SHRINK))
        assert widget.get_add_params() == dict(
            NUCLEI_AT_DEFAULTS, outline=AP.OUTLINE_SHRINK,
            bridge_um=AP.ADD_BRIDGE_UM)
        m = widget.add_look_for_combo
        m.setCurrentIndex(m.count() - 1)
        # (the markers' own Spacing rides with Look for on the markers)
        assert widget.get_add_params() == dict(
            NUCLEI_AT_DEFAULTS, look_for=AP.LOOK_MARKERS,
            marker_spacing=AP.ADD_MARKER_SPACING_DEFAULT,
            outline=AP.OUTLINE_SHRINK, bridge_um=AP.ADD_BRIDGE_UM)
        m.setCurrentIndex(0)
    finally:
        c.setCurrentIndex(AP.OUTLINE_KEYS.index(AP.OUTLINE_NONE))
        widget.update_add_ui_state(box_active=False)
    assert widget.get_add_params() == NUCLEI_AT_DEFAULTS
    reset_add(widget)


def test_the_two_new_controls_carry_their_help(widget):
    from napari_epicycle.annotator.advanced_features_widget import (
        ADD_HELP_LINES,
    )

    assert ADD_HELP_LINES["add_nuc_outline_combo"] == (
        "Where new nuclei may be kept: inside the outline of the objects you "
        "cleared, inside a shrink-wrap around the stain, or (as before) "
        "anywhere in the box.")
    assert ADD_HELP_LINES["add_nuc_show_outline_cb"] == (
        ADD_HELP_LINES["add_show_outline_cb"])
    for attr in ("add_nuc_outline_combo", "add_nuc_show_outline_cb"):
        assert getattr(widget, attr).toolTip() == ADD_HELP_LINES[attr]
    # the combo is sized like the Cells one
    a, b = widget.add_nuc_outline_combo, widget.add_outline_combo
    assert a.sizeAdjustPolicy() == b.sizeAdjustPolicy()
    assert a.minimumContentsLength() == b.minimumContentsLength()


def test_the_guide_says_the_nuclei_outline():
    import pathlib

    guide = (pathlib.Path(__file__).resolve().parents[2] / "docs" / "guides"
             / "annotator_3d_add.md").read_text(encoding="utf-8")
    body = " ".join(guide.split())
    for text in (AP.ADD_HELP_NUC_OUTLINE_SHRINK, AP.ADD_HELP_NUC_OUTLINE_NONE,
                 AP.ADD_HELP_NUC_OUTLINE_CLEARED,
                 AP.ADD_NOTE_NUCLEI_OUTSIDE_OUTLINE_ONE,
                 AP.ADD_S1_NUCLEI_ALL_OUTSIDE, AP.ADD_MARKER_OUTLINE_TIP):
        assert " ".join(text.split()) in body, text
    sec = guide.split("## The tissue outline", 1)[1].split("\n## ", 1)[0]
    assert "Nuclei" in sec and "off by default" in sec
    # Dark below on Nuclei, in the guide's words (its tooltip opens with a
    # word the guide retired)
    assert "Lower it if dim tissue falls outside the outline." in body
