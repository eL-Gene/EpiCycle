"""Every Add feature behaves the same on any region.

A box and the whole patch are one thing to the page: Nuclei ▸ Anywhere in
the box and Cells ▸ Nuclei masks from a channel run on a whole patch as on
a box (Stop acts between the finder's steps), and the Tissue outline starts
on the same default everywhere.  The one difference left -- a whole patch
fills with a background marker and open faces, because a patch edge is the
end of the data while a box edge cuts through tissue -- is said on the
page.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from packages.epicycle_annotator import add_preview as AP
from packages.epicycle_annotator import add_segment as AS
from packages.epicycle_annotator.reshape.engine import ReshapeCancelled

from .add_harness import Harness, install_fakes
from .add_widget_fixture import qapp, reset_add, widget  # noqa: F401
from .test_add_nuclei_golden import nuclei_fixture


# --------------------------------------------------------------------------- #
# the finder: Stop between its steps
# --------------------------------------------------------------------------- #
def _finder_kw():
    return dict(clahe=False, threshold="otsu", min_size=0, separate=True,
                min_distance=3, next_label=20, spacing=(1.0, 1.0, 1.0))


def test_no_cancel_event_is_the_finder_as_it_always_was():
    img, labels = nuclei_fixture(0)
    plain = AS.add_segment_region(img, labels, **_finder_kw())
    none = AS.add_segment_region(img, labels, cancel_event=None,
                                 **_finder_kw())
    unset = AS.add_segment_region(img, labels,
                                  cancel_event=threading.Event(),
                                  **_finder_kw())
    assert plain.dtype == none.dtype == unset.dtype
    assert np.array_equal(plain, none) and np.array_equal(plain, unset)
    assert plain.tobytes() == unset.tobytes()


@pytest.mark.parametrize("stage", ["threshold_otsu", "distance_transform_edt",
                                   "gaussian_filter", "peak_local_max",
                                   "watershed"])
def test_a_stop_during_a_step_ends_the_finder_when_that_step_ends(
        monkeypatch, stage):
    """Stop pressed while a step runs: that step finishes (it cannot be cut
    short) and the finder stops before the next one."""
    import scipy.ndimage
    import skimage.feature
    import skimage.filters
    import skimage.segmentation

    img, labels = nuclei_fixture(0)
    ev = threading.Event()
    mod = {"threshold_otsu": skimage.filters,
           "distance_transform_edt": scipy.ndimage,
           "gaussian_filter": scipy.ndimage,
           "peak_local_max": skimage.feature,
           "watershed": skimage.segmentation}[stage]
    real = getattr(mod, stage)
    ran = []

    def _spy(*a, **k):
        out = real(*a, **k)
        ran.append(stage)
        ev.set()                       # the user presses Stop meanwhile
        return out

    monkeypatch.setattr(mod, stage, _spy)
    with pytest.raises(ReshapeCancelled):
        AS.add_segment_region(img, labels, cancel_event=ev, **_finder_kw())
    assert ran == [stage]


def test_a_stop_before_the_finder_starts_reads_nothing():
    img, labels = nuclei_fixture(0)
    ev = threading.Event()
    ev.set()
    with pytest.raises(ReshapeCancelled):
        AS.add_segment_region(img, labels, cancel_event=ev, **_finder_kw())


# --------------------------------------------------------------------------- #
# S1 (pure): nothing is refused for being a whole patch
# --------------------------------------------------------------------------- #
def test_anywhere_on_a_whole_patch_is_not_refused():
    s = dict(has_box=True, whole_patch=True, box_shape=(20, 100, 100),
             preset="nuclei", channel="nuc", look_for=AP.LOOK_ANYWHERE,
             empty_share=0.5)
    assert AP.preview_block(s) is None
    assert not hasattr(AP, "ADD_S1_WHOLE_ANYWHERE")
    assert not AP.whole_measures_nothing("nuclei", None, AP.LOOK_ANYWHERE)


def test_anywhere_on_a_whole_patch_over_the_limit_is_refused_by_size():
    big = (200, 1000, 1000)                 # 200 million voxels
    s = dict(has_box=True, whole_patch=True, box_shape=big,
             preset="nuclei", channel="nuc", look_for=AP.LOOK_ANYWHERE)
    assert AP.preview_block(s) == AP.whole_too_big_line(200_000_000)
    # the markers and Inside each cell keep running on any size
    for look in (AP.LOOK_MARKERS, AP.LOOK_INSIDE):
        assert AP.preview_block(dict(s, look_for=look)) != (
            AP.whole_too_big_line(200_000_000))


def test_nuclei_masks_on_a_whole_patch_are_not_refused():
    s = dict(has_box=True, whole_patch=True, box_shape=(20, 100, 100),
             preset="cells", source=AP.SOURCE_NUCLEI, wall_channel="mem",
             nuclei_channel="nuc", empty_share=0.5)
    assert AP.preview_block(s) is None
    assert not AP.whole_measures_nothing("cells", AP.SOURCE_NUCLEI,
                                         AP.LOOK_ANYWHERE)
    # the whole patch's sentences name the nuclei masks as a way
    for line in (AP.ADD_S1_WHOLE_PICK_RUN, AP.ADD_S1_WHOLE_NO_RUN):
        assert AP.ADD_ITEM_NUCLEI.split()[0] in line, line


def test_the_slow_note_names_a_time_for_anywhere_on_a_whole_patch():
    line = AP.nuclei_patch_slow_line(60_000_000, anywhere=True)
    assert line == AP.ADD_NOTE_PATCH_SLOW_STEPS.format(
        m="60.0", t=AP.wait_phrase(AP.anywhere_minutes(60.0)))
    # the markers' line is unchanged
    assert AP.nuclei_patch_slow_line(60_000_000) == (
        AP.ADD_NOTE_PATCH_SLOW_PLAIN.format(m="60.0"))


def test_stop_says_it_takes_effect_when_the_step_ends():
    assert AP.ADD_STOP_STEP_ENDS == "Stop takes effect when this step ends."


def test_the_one_whole_patch_difference_is_said():
    assert AP.ADD_HELP_WHOLE_FILL_WHY in AP.whole_fill_help(False)
    assert "end of the data" in AP.ADD_HELP_WHOLE_FILL_WHY


# --------------------------------------------------------------------------- #
# the orchestrator: Nuclei ▸ Anywhere on a whole patch
# --------------------------------------------------------------------------- #
def _land_info(h):
    for _ in range(3):
        th = getattr(h, "_add_info_thread", None)
        if th is None:
            return
        th.run()
        th.end()


@pytest.fixture
def hn(monkeypatch):
    install_fakes(monkeypatch)
    img, labels = nuclei_fixture(0)
    hh = Harness(labels.copy(), images={"nuc": img})
    hh.ctx["spacing_source"] = "qc_settings"
    hh.img = img
    return hh


def test_anywhere_on_a_whole_patch_previews_with_stop(hn, monkeypatch):
    seen = []
    real = AS.add_segment_region

    def _spy(*a, **k):
        seen.append(k.get("cancel_event"))
        return real(*a, **k)

    monkeypatch.setattr(AS, "add_segment_region", _spy)
    hn._start_add_whole()
    assert hn._add_whole_patch
    _land_info(hn)
    hn._add_refresh_ui()
    assert hn.adv.last("preview_enabled") is True
    kw, why = hn._add_gather_preview()
    assert why is None, why
    hn._preview_add()
    th = hn._add_thread
    th.run()
    th.end()
    res = hn._add_preview_res
    assert res is not None and res.ok, getattr(res, "refusal", None)
    assert seen and seen[-1] is not None
    # the objects are the finder's over the whole patch, as on a box
    want = real(hn.img, hn.sync_manager.shared_labels["L"],
                threshold=float(hn._add_threshold_value), min_size=0,
                separate=True, min_distance=3, next_label=1,
                spacing=(1.0, 1.0, 1.0))
    assert len(res.ids) == len(set(np.unique(want)) - {0})


def test_a_stop_between_the_finders_steps_on_a_whole_patch(hn, monkeypatch):
    import scipy.ndimage

    real = scipy.ndimage.distance_transform_edt

    def _edt(*a, **k):
        out = real(*a, **k)
        hn._stop_add_preview()          # the user presses Stop
        return out

    monkeypatch.setattr(scipy.ndimage, "distance_transform_edt", _edt)
    # a region over the slow size: the strip says Stop waits for the step
    monkeypatch.setattr(AP, "ADD_SLOW_PATCH_VOXELS", 10)
    before = hn.sync_manager.shared_labels["L"].copy()
    hn._start_add_whole()
    _land_info(hn)
    hn._preview_add()
    assert hn._add_busy
    th = hn._add_thread
    th.run()
    th.end()
    assert hn._add_preview_res is None
    assert hn._add_transient == AP.ADD_S1_STOPPED_PATCH
    said = [str(kw.get("stage_text", "")) for kw in hn.adv.pushes]
    assert any(t.endswith(AP.ADD_STOP_STEP_ENDS) for t in said), said
    assert np.array_equal(hn.sync_manager.shared_labels["L"], before)
    assert not hn._add_busy


# --------------------------------------------------------------------------- #
# the orchestrator: Cells ▸ Nuclei masks on a whole patch
# --------------------------------------------------------------------------- #
@pytest.fixture
def hm(monkeypatch):
    from .test_add_outline_wiring import _CellsAdv, _threshold, _world

    install_fakes(monkeypatch)
    ch, cells = _world()
    hh = Harness(np.zeros(cells.shape, np.int32), images=ch)
    hh.adv = _CellsAdv()
    hh.adv.source = (AP.SOURCE_NUCLEI, None)
    hh.adv.nuclei_channel = "nuclei"
    hh.ctx["spacing_source"] = "qc_settings"
    hh._add_dark_cache = {hh._add_dark_key(): _threshold(ch)}
    return hh


def _run(h):
    h._preview_add()
    th = h._add_thread
    th.run()
    th.end()
    return h._add_preview_res


def test_nuclei_masks_on_a_whole_patch_preview_and_keep_their_seeds(
        hm, monkeypatch):
    seen = []
    real = AS.add_segment_region

    def _spy(*a, **k):
        seen.append(k.get("cancel_event"))
        return real(*a, **k)

    monkeypatch.setattr(AS, "add_segment_region", _spy)
    hm._start_add_whole()
    _land_info(hm)
    hm._add_refresh_ui()
    s = hm._add_s1_state()
    assert AP.preview_block(s) is None, AP.preview_block(s)
    assert hm.adv.last("preview_enabled") is True
    res = _run(hm)
    assert res is not None and res.ok, getattr(res, "refusal", None)
    assert len(seen) == 1 and seen[0] is not None
    assert AP.cells_growth(dict(hm._add_params(), whole_patch=True)) == (
        AP.GROW_PATCH)
    cache = hm._add_nuclei_seeds_cache
    assert cache and cache["seeds"] is not None
    # the next Preview is handed the seeds, and finds no nuclei again
    kw, why = hm._add_gather_preview()
    assert why is None, why
    assert kw["nuclei_seeds"]["seeds"] is cache["seeds"]
    hm._invalidate_add_preview(live_schedule=False)
    res2 = _run(hm)
    assert res2 is not None and res2.ok
    assert len(seen) == 1
    assert np.array_equal(res2.result, res.result)


def test_the_kept_seeds_go_when_what_they_were_found_from_changes(hm):
    hm._start_add_whole()
    _land_info(hm)
    _run(hm)
    assert hm._add_nuclei_seeds_cache

    def _handed():
        kw, why = hm._add_gather_preview()
        assert why is None, why
        return kw.get("nuclei_seeds") is not None

    assert _handed()
    # a labels write
    hm._add_volume_epoch = int(getattr(hm, "_add_volume_epoch", 0)) + 1
    assert not _handed()
    hm._invalidate_add_preview(live_schedule=False)
    _run(hm)
    assert _handed()
    # a Clear in force
    hm._add_cleared_in_force = lambda: [(1, (3,))]
    assert not _handed()
    del hm._add_cleared_in_force
    assert _handed()
    # the nuclei channel
    hm.adv.nuclei_channel = "membrane"
    assert hm._add_nuclei_seeds_key() != hm._add_nuclei_seeds_cache["key"]
    hm.adv.nuclei_channel = "nuclei"
    assert _handed()
    # the threshold
    hm._add_nuclei_threshold_value = 0.123
    assert not _handed()
    # the prune drops a stale seed picture
    hm._add_prune_caches()
    assert hm._add_nuclei_seeds_cache is None


def test_a_box_finds_its_nuclei_at_every_preview(hm):
    """A box keeps its bytes: its seeds are found at every Preview and
    nothing is kept."""
    hm._start_add_whole()
    _land_info(hm)
    b = hm._add_box
    hm._on_add_box_changed(b.cz, b.cy, b.cx, 8, 20, 20)
    assert not hm._add_whole_patch
    hm._request_add_box_info()
    _land_info(hm)
    kw, why = hm._add_gather_preview()
    assert why is None, why
    assert kw.get("nuclei_seeds") is None
    res = _run(hm)
    assert res is not None
    assert getattr(hm, "_add_nuclei_seeds_cache", None) is None


def test_the_counts_note_says_how_many_markers_were_small(hm):
    hm._start_add_whole()
    _land_info(hm)
    res = _run(hm)
    assert res is not None and res.ok
    assert res.n_small_markers >= 0
    out = AP.AddPreviewResult(n_small_markers=3)
    assert "3 markers were under 10 µm³" in AP.counts_note(out.counts())
    out = AP.AddPreviewResult(n_small_markers=1)
    assert "1 marker was under 10 µm³" in AP.counts_note(out.counts())


# --------------------------------------------------------------------------- #
# one Tissue outline default for every region (the panel)
# --------------------------------------------------------------------------- #
def _cells(w, *, whole=False, in_force=False):
    reset_add(w)
    w.set_add_preset("cells")
    w.update_add_ui_state(box_active=True, whole_patch=whole,
                          cleared_in_force=in_force, numba_ok=True)


def _undo(w):
    w.update_add_ui_state(box_active=False, whole_patch=False,
                          cleared_in_force=False, numba_ok=True)
    reset_add(w)


@pytest.mark.parametrize("whole", [False, True])
def test_the_outline_starts_on_the_shrink_wrap_on_any_region(widget, whole):
    _cells(widget, whole=whole)
    try:
        assert widget.get_add_outline() == AP.OUTLINE_SHRINK
        widget.update_add_ui_state(cleared_in_force=True)
        assert widget.get_add_outline() == AP.OUTLINE_CLEARED
        widget.update_add_ui_state(cleared_in_force=False)
        assert widget.get_add_outline() == AP.OUTLINE_SHRINK
    finally:
        _undo(widget)


def test_a_picked_no_outline_stays_on_any_region(widget):
    _cells(widget)
    try:
        widget.add_outline_combo.setCurrentIndex(
            AP.OUTLINE_KEYS.index(AP.OUTLINE_NONE))
        for whole in (True, False):
            widget.update_add_ui_state(whole_patch=whole,
                                       cleared_in_force=True)
            assert widget.get_add_outline() == AP.OUTLINE_NONE
    finally:
        _undo(widget)


def test_the_no_outline_item_drops_as_before():
    assert AP.ADD_OUTLINE_NONE == "No outline"
    assert "as before" not in AP.ADD_HELP_OUTLINE_NONE


def test_the_nuclei_outline_row_opens_on_a_whole_patch(widget):
    reset_add(widget)
    widget.update_add_ui_state(box_active=True, whole_patch=True)
    try:
        assert widget._add_nuc_outline_offered()
        assert not widget.add_nuc_outline_btn.isHidden()
    finally:
        _undo(widget)
