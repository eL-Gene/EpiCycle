"""Whole patch on the Nuclei preset ▸ Look for nuclei ▸ Inside each cell.

* the box IS the patch: no edge flag at a patch face (an ordinary box keeps
  every face, see ``test_add_edge_flags_golden``);
* the box is never clamped and never over a budget on Nuclei; Recenter is
  grey; the outline drag and Shift + arrow do nothing;
* only a Numbers slider move or S / M / L leaves whole-patch mode (a
  marker source, Look for nuclei or a preset change keep it, and Preview
  says what is missing -- ``test_add_page_redesign``);
* Stop pressed inside the recipe writes nothing;
* Cells: a whole patch fills to the walls in one go (Grow by is hidden),
  is never over the box limit, says its slow note, counts its run markers
  "in the patch"; leaving it shrinks the box to the Cells limit and S1 says
  so.
"""

from __future__ import annotations

import numpy as np
import pytest

from packages.epicycle_annotator import add_preview as AP
from packages.epicycle_annotator.add_segment import BOX_FACES, edge_flag_ids


# --------------------------------------------------------------------------- #
# the edge rule (pure)
# --------------------------------------------------------------------------- #
def _two_at_faces():
    lab = np.zeros((10, 12, 12), np.int32)
    lab[0:3, 4:6, 4:6] = 1            # the low z face
    lab[4:6, 4:6, 9:12] = 2           # the high x face
    lab[4:6, 4:6, 4:6] = 3            # the middle
    return lab


def test_every_face_open_flags_nothing():
    assert edge_flag_ids(_two_at_faces(), 2) == {1, 2}
    assert edge_flag_ids(_two_at_faces(), 2, open_faces=BOX_FACES) == set()


def test_one_open_face_keeps_the_others():
    assert edge_flag_ids(_two_at_faces(), 2, open_faces=("z0",)) == {2}
    assert edge_flag_ids(_two_at_faces(), 2, open_faces=("x1",)) == {1}


def test_an_unknown_face_is_refused():
    with pytest.raises(ValueError):
        edge_flag_ids(_two_at_faces(), 2, open_faces=("top",))


def test_a_huge_id_is_flagged_without_a_huge_table():
    lab = _two_at_faces().astype(np.uint32)
    lab[lab == 2] = 2 ** 31 - 1
    assert edge_flag_ids(lab, 2) == {1, 2 ** 31 - 1}


def test_the_finish_flags_no_patch_face_on_whole_patch():
    lab = _two_at_faces()
    out = AP.AddPreviewResult()
    step = AP._Steps(None, ("add_smooth",))
    AP._finish(out, lab.copy(), {"edge_margin": 2, "auto_smooth": False,
                                 "whole_patch": True}, None, step)
    assert out.edge_ids == set()
    out = AP.AddPreviewResult()
    AP._finish(out, lab.copy(), {"edge_margin": 2, "auto_smooth": False},
               None, step)
    assert out.edge_ids == {1, 2}


def test_a_cells_whole_patch_is_never_over_the_box_limit():
    """A Cells whole patch is filled to the walls in one go whatever Grow by
    says, so the 4 million voxel limit of a box never applies to it; an
    ordinary box is held to it with either Grow by."""
    existing = np.zeros((2, 2, 4), np.int32)
    p = {"preset": "cells", "box_budget": 10, "dark_outside": False}
    kw = dict(existing_roi=existing, dark_sources=None, dark_rois=None,
              spacing=(1.0, 1.0, 1.0), stride=1, dark_cache=None,
              source="paint", run_guide_roi=None, nuclei_roi=None)
    assert AP.add_box_info(params=p, **kw).over_limit
    # a whole patch, whatever the hidden Grow by: never over
    for g in (None, "flood", "compete"):
        q = dict(p, whole_patch=True)
        if g is not None:
            q["growth"] = g
        assert not AP.add_box_info(params=q, **kw).over_limit, g
    # either Grow by on an ordinary box is still held to the limit
    assert AP.add_box_info(params=dict(p, growth="compete"), **kw).over_limit
    assert AP.add_box_info(params=dict(p, growth="flood"), **kw).over_limit


def test_the_ready_line_says_whole_patch():
    s = dict(has_patch=True, has_box=True, preset="nuclei", channel="nuc",
             box_shape=(10, 20, 30), empty_share=0.5, whole_patch=True,
             look_for=AP.LOOK_INSIDE)
    assert AP.s1_sentence(s) == AP.ADD_S1_PATCH_PLAIN.format(
        z=10, y=20, x=30, m="0.1", p=50)
    assert AP.s1_sentence(dict(s, whole_patch=False)).startswith("Box ")


# --------------------------------------------------------------------------- #
# the orchestrator (the Qt-free harness)
# --------------------------------------------------------------------------- #
from .add_harness import FakeAdv, FakeThread, Harness, install_fakes  # noqa: E402,E501


class _InsideAdv(FakeAdv):
    """Look for nuclei ▸ Inside each cell of run "cells" -- the one Look
    for nuclei that Whole patch runs in this version."""

    def __init__(self):
        super().__init__()
        self.look = (AP.LOOK_INSIDE, "cells")
        self.params.update(look_for=AP.LOOK_INSIDE, nuc_radius_um=2.5,
                           nuc_edge=0.25, two_gap=0.70)

    def get_add_look_for(self):
        return self.look


@pytest.fixture
def h(monkeypatch):
    from packages.epicycle_annotator.add_derive import patch_levels

    from .test_add_derive_nuclei import ISO, three_cells_scene

    install_fakes(monkeypatch)
    img, cells = three_cells_scene()
    hh = Harness(np.zeros(cells.shape, np.int32), images={"nuc": img},
                 spacing=ISO)
    hh.adv = _InsideAdv()
    hh.ctx["spacing_source"] = "qc_settings"
    # the other run in memory (as its read leaves it) and the patch's levels
    hh._add_run_guide = cells
    hh._add_levels_cache = {}
    hh._add_levels_cache[hh._add_levels_key()] = patch_levels(img, cells, ISO)
    hh._get_advanced_features_widget = lambda: hh.adv
    hh.cells = cells
    return hh


def _land_info(h):
    """The box-info worker runs and its thread finishes."""
    for _ in range(3):
        th = getattr(h, "_add_info_thread", None)
        if th is None:
            return
        th.run()
        th.end()


def _shape(h):
    return tuple(h.sync_manager.shared_labels["L"].shape)


def test_whole_patch_starts_a_box_that_is_the_patch(h):
    h._start_add_whole()
    assert h._add_running and h._add_whole_patch
    assert h._add_roi == tuple(slice(0, n) for n in _shape(h))
    assert h.adv.last("whole_patch") is True


def test_a_whole_patch_is_never_clamped(h, monkeypatch):
    called = []
    monkeypatch.setattr(type(h), "_add_clamp_box",
                        lambda self, box, shape: called.append(box) or box,
                        raising=False)
    h._start_add_whole()
    assert called == []


def test_the_cells_preset_has_a_whole_patch(h):
    h.adv.preset = "cells"
    h.adv.source = ("labels", "cells")
    h._start_add_whole()
    assert h._add_running and h._add_whole_patch
    assert h._add_roi == tuple(slice(0, n) for n in _shape(h))


def test_whole_patch_from_numbers_makes_the_box_the_patch(h):
    h.place_box((slice(2, 6), slice(2, 10), slice(2, 10)))
    h._start_add_whole()
    assert h._add_whole_patch
    assert h._add_roi == tuple(slice(0, n) for n in _shape(h))
    assert h.adv.last("box_size") == _shape(h)


def test_a_slider_or_s_m_l_leaves_whole_patch(h):
    h._start_add_whole()
    b = h._add_box
    h._on_add_box_changed(b.cz, b.cy, b.cx, 4, 8, 8)
    assert not h._add_whole_patch
    assert h.adv.last("whole_patch") is False
    h._start_add_whole()
    assert h._add_whole_patch
    h._set_add_box_size_fraction(0.25)
    assert not h._add_whole_patch


def test_the_whole_patch_does_not_move(h, monkeypatch):
    h._start_add_whole()
    box = h._add_box
    monkeypatch.setattr(type(h), "_add_side_ready", lambda self: True,
                        raising=False)
    from packages.epicycle_annotator import box_drag as BD

    monkeypatch.setattr(BD, "nudge_delta", lambda orient, key: (0, 0, 1))
    assert h._add_nudge_box("xy", "right") is True    # the key is taken
    assert h._add_box == box and h._add_whole_patch
    h._recenter_add_box_on_crosshair()
    assert h._add_box == box and h._add_whole_patch


def test_no_patch_face_is_an_edge_on_whole_patch(h):
    h._start_add_whole()
    _land_info(h)
    kw, why = h._add_gather_preview()
    assert why is None and kw["params"]["whole_patch"] is True


def test_the_patch_is_read_with_its_sentence_painted_first(h):
    h._start_add_whole()
    _land_info(h)
    texts = []
    real = h._push_add_ui

    def spy(**kw):
        if "status_text" in kw:
            texts.append(kw["status_text"])
        return real(**kw)

    h._push_add_ui = spy
    h._preview_add()
    assert AP.ADD_S1_GATHERING in texts
    assert h._add_busy


def test_stop_inside_the_recipe_on_a_whole_patch_writes_nothing(
        h, monkeypatch):
    """Stop pressed while the recipe is finding nuclei (not before it
    starts): the recipe's own poll raises, the answer is "cancelled", S1
    says Stopped, nothing is written and the other run is untouched."""
    from packages.epicycle_annotator import add_derive as AD

    h._start_add_whole()
    _land_info(h)
    before = h.sync_manager.shared_labels["L"].copy()
    cells_before = h.cells.copy()
    real = AD.nuclei_in_cells
    seen = []

    def recipe(*a, progress=None, **k):
        def spy(text, i, n):
            seen.append(text)
            if text == AD.DERIVE_STAGE_CENTRES:
                h._stop_add_preview()          # the user presses Stop
            if progress is not None:
                progress(text, i, n)
        return real(*a, progress=spy, **k)

    monkeypatch.setattr(AD, "nuclei_in_cells", recipe)
    h._preview_add()
    assert h._add_busy
    th = h._add_thread
    th.run()
    th.end()
    assert AD.DERIVE_STAGE_CENTRES in seen, "Stop came inside the recipe"
    assert AD.DERIVE_STAGE_TIDY not in seen, "the recipe stopped early"
    assert h._add_preview_result is None and h._add_preview_res is None
    assert h._add_transient == AP.ADD_S1_STOPPED_PATCH
    assert np.array_equal(h.sync_manager.shared_labels["L"], before)
    assert np.array_equal(h.cells, cells_before)
    assert not h._add_busy


def test_the_bar_counts_steps_on_a_whole_patch_only(h):
    h._start_add_whole()
    h._on_add_progress("add_find", 2, 3)
    assert h.adv.pushes[-1]["progress"] == (2, 3)
    h._on_add_box_changed(10, 20, 20, 4, 8, 8)
    h._on_add_progress("add_find", 2, 3)
    assert "progress" not in h.adv.pushes[-1]


def test_long_stages_keep_the_bar_moving(h):
    """A whole patch: the fill, the wall picture and a stage with one step
    or none run the moving bar while the line names them; the others
    count."""
    from napari_epicycle.annotator.reshape_worker import stage_text

    h._start_add_whole()
    for stage, i, n in (("add_cut", 4, 4), ("elevation", 2, 4),
                        ("add_smooth", 1, 1), ("add_find", 0, 0)):
        h._on_add_progress(stage, i, n)
        assert h.adv.pushes[-1]["progress"] == (), stage
        assert h.adv.pushes[-1]["stage_text"] == stage_text(stage)
    h._on_add_progress("add_find", 2, 3)
    assert h.adv.pushes[-1]["progress"] == (2, 3)
    # a box never counts: its bar always moves
    h._on_add_box_changed(10, 20, 20, 4, 8, 8)
    h._on_add_progress("elevation", 2, 4)
    assert "progress" not in h.adv.pushes[-1]


def test_a_large_whole_patch_says_a_preview_may_take_minutes(h, monkeypatch):
    monkeypatch.setattr(AP, "ADD_SLOW_PATCH_VOXELS", 1000)
    h._start_add_whole()
    h._add_refresh_ui()
    n = int(np.prod(_shape(h)))
    # Nuclei: no figure (that wait was never timed)
    assert h.adv.last("derive_note") == AP.nuclei_patch_slow_line(n)


def test_every_exit_drops_the_whole_patch(h):
    """Close box and the next box both leave the whole patch -- and the
    markers placed on it: a marker in the patch's far corner (outside any
    box the page would place by default) is gone, and Place markers is
    off."""
    h.adv.look = (AP.LOOK_MARKERS, None)
    h._start_add_whole()
    h._add_place_markers_on = True
    corner = tuple(n - 1 for n in _shape(h))
    assert h._add_marker_click(corner), "a marker on the whole patch"
    assert [tuple(m["zyx"]) for m in h._add_markers()] == [corner]
    h._reject_add()                                # Close box
    assert not h._add_whole_patch
    assert h._add_markers() == []
    assert not h._add_place_markers_on
    h._start_add()                                 # the next box
    assert not h._add_whole_patch
    assert h._add_markers() == []
    assert not h._add_place_markers_on


# --------------------------------------------------------------------------- #
# Cells: a whole patch spreads, in pieces
# --------------------------------------------------------------------------- #
@pytest.fixture
def hc(monkeypatch):
    """A Cells page with Markers ▸ a run and the shrink-wrap (the whole
    patch's default outline)."""
    from .test_add_outline_wiring import _CellsAdv, _threshold, _world

    install_fakes(monkeypatch)
    ch, cells = _world()
    hh = Harness(np.zeros(cells.shape, np.int32), images=ch)
    hh.adv = _CellsAdv()
    hh.ctx["spacing_source"] = "qc_settings"
    hh._add_run_guide = cells
    hh._add_dark_cache = {hh._add_dark_key(): _threshold(ch)}
    return hh


def test_a_cells_whole_patch_is_never_clamped_nor_over_the_limit(
        hc, monkeypatch):
    called = []
    monkeypatch.setattr(type(hc), "_add_clamp_box",
                        lambda self, box, shape: called.append(box) or box,
                        raising=False)
    hc.adv.params["box_budget"] = 10           # far under the patch
    hc._start_add_whole()
    assert called == [] and hc._add_whole_patch
    _land_info(hc)
    assert hc._add_info is not None and not hc._add_info.over_limit
    s = hc._add_s1_state()
    assert AP.preview_block(s) is None, AP.preview_block(s)


def test_a_cells_whole_patch_refuses_at_the_default_spacing(hc):
    hc.ctx["spacing_source"] = "unknown"
    hc._start_add_whole()
    _land_info(hc)
    hc._add_refresh_ui()
    assert hc.adv.last("status_text") == AP.ADD_S1_NEEDS_SPACING_BOXED
    assert hc.adv.last("preview_enabled") is False
    # an ordinary Cells box keeps running at the default spacing
    b = hc._add_box
    hc._on_add_box_changed(b.cz, b.cy, b.cx, 6, 10, 10)
    _land_info(hc)
    hc._add_refresh_ui()
    assert hc.adv.last("status_text") != AP.ADD_S1_NEEDS_SPACING_BOXED


def test_no_patch_face_is_an_edge_on_a_cells_whole_patch(hc):
    hc._start_add_whole()
    _land_info(hc)
    kw, why = hc._add_gather_preview()
    assert why is None, why
    assert kw["params"]["whole_patch"] is True
    assert AP.cells_growth(kw["params"]) == AP.GROW_PATCH


def test_stop_at_the_fill_on_a_cells_whole_patch_writes_nothing(
        hc, monkeypatch):
    """Stop pressed as the fill starts: the engine's own poll raises, the
    answer is "cancelled", S1 says Stopped, nothing is written."""
    from packages.epicycle_annotator.reshape import region as R

    hc._start_add_whole()
    _land_info(hc)
    before = hc.sync_manager.shared_labels["L"].copy()
    real = R.reshape_empty_region
    seen = []

    def wrapped(*a, progress=None, **k):
        def spy(stage, i, n):
            seen.append(stage)
            if stage == "add_cut":
                hc._stop_add_preview()          # the user presses Stop
            if progress is not None:
                progress(stage, i, n)
        return real(*a, progress=spy, **k)

    monkeypatch.setattr(R, "reshape_empty_region", wrapped)
    hc._preview_add()
    assert hc._add_busy
    th = hc._add_thread
    th.run()
    th.end()
    assert "add_cut" in seen and "add_grow" not in seen
    assert hc._add_preview_result is None and hc._add_preview_res is None
    assert hc._add_transient == AP.ADD_S1_STOPPED_PATCH
    assert np.array_equal(hc.sync_manager.shared_labels["L"], before)
    assert not hc._add_busy


def test_a_cells_whole_patch_fills_to_the_walls_and_keeps_its_walls(hc):
    """The full path on the harness: the Preview fills the whole patch to
    the walls in one go (no tiles), and the wall picture is handed to the
    next Preview."""
    hc._start_add_whole()
    _land_info(hc)
    hc._preview_add()
    th = hc._add_thread
    th.run()
    th.end()
    res = hc._add_preview_res
    assert res is not None and res.ok, getattr(res, "refusal", None)
    assert res.n_tiles == 0 and res.ids
    assert hc._add_elev_cache["elev"] is res.elevation
    kw, why = hc._add_gather_preview()
    assert kw["elevation_roi"] is res.elevation


def _clicks_whole(hc):
    """The Cells whole patch with Click markers picked and two placed."""
    hc.adv.source = (AP.SOURCE_CLICKS, None)
    hc._start_add_whole()
    _land_info(hc)
    hc._add_place_markers_on = True
    shape = _shape(hc)
    pts = [(shape[0] // 2, shape[1] // 4, shape[2] // 4),
           (shape[0] // 2, 3 * shape[1] // 4, 3 * shape[2] // 4)]
    vol = hc.sync_manager.shared_labels["L"]
    hc._add_click_markers = [
        {"zyx": p, "id": int(vol.max()) + 1 + i, "ticked": True}
        for i, p in enumerate(pts)]
    return pts


def test_a_cells_whole_patch_takes_click_markers(hc):
    """Click markers fill a whole patch as another run's objects do: the
    run rows let them pass, the gather sends the ticked points with the
    whole patch's own growth rule, and the box-info worker measures the
    patch for it (the strip names the patch's read)."""
    pts = _clicks_whole(hc)
    s = hc._add_s1_state()
    assert s["whole_patch"] and s["source"] == AP.SOURCE_CLICKS
    assert AP.preview_block(s) is None, AP.preview_block(s)
    assert AP.s1_sentence(s).startswith("Whole patch ")
    kw, why = hc._add_gather_preview()
    assert why is None, why
    assert kw["click_points"] == pts
    assert kw["params"]["whole_patch"] is True
    assert AP.cells_growth(kw["params"]) == AP.GROW_PATCH
    assert AP.whole_measures_nothing("cells", AP.SOURCE_CLICKS,
                                     None) is False
    assert AP.measuring_stage(s, reader="info") == "add_info_patch"


def test_a_cells_whole_patch_with_click_markers_refuses_at_default_spacing(
        hc):
    """The default-spacing refusal holds for Click markers on a whole
    patch (now reachable)."""
    hc.ctx["spacing_source"] = "unknown"
    _clicks_whole(hc)
    hc._add_refresh_ui()
    assert hc.adv.last("status_text") == AP.ADD_S1_NEEDS_SPACING_BOXED
    assert hc.adv.last("preview_enabled") is False


def _fill(hc):
    hc._preview_add()
    th = hc._add_thread
    th.run()
    th.end()
    return hc._add_preview_res


def test_a_cells_whole_patch_fills_from_click_markers(hc):
    """The full path on the harness with Click markers: one Preview fills
    the whole patch to the walls, and EVERY click becomes the id it
    reserved (all three clicks inside the tissue outline)."""
    _clicks_whole(hc)
    outline = hc._add_cached_outline()
    assert outline is not None
    # the second click moved inside the outline, and a third added there
    hc._add_click_markers[1]["zyx"] = (6, 25, 30)
    hc._add_click_markers.append(
        {"zyx": (6, 20, 30),
         "id": max(m["id"] for m in hc._add_click_markers) + 1,
         "ticked": True})
    assert all(outline[tuple(m["zyx"])] for m in hc._add_click_markers)
    want = [m["id"] for m in hc._add_click_markers]
    res = _fill(hc)
    assert res is not None and res.ok, getattr(res, "refusal", None)
    assert res.n_tiles == 0
    assert sorted(res.ids) == sorted(want)


@pytest.mark.xfail(strict=True, reason=(
    "a click outside the shrink-wrap outline makes nothing and says "
    "nothing -- on a box as well as on a whole patch (the outline is a hard "
    "edge for the fill and is clipped again after the smoothing); recorded, "
    "not fixed here"))
def test_a_click_outside_the_outline_is_kept(hc):
    _clicks_whole(hc)
    outline = hc._add_cached_outline()
    # the second click of _clicks_whole lies outside the outline
    assert not outline[tuple(hc._add_click_markers[1]["zyx"])]
    want = [m["id"] for m in hc._add_click_markers]
    res = _fill(hc)
    assert res is not None and res.ok
    assert (sorted(res.ids) == sorted(want)
            or set(want) - set(res.ids) <= set(hc._add_marker_why)), (
        "a clicked marker is kept, or its row says why it made nothing")


# --------------------------------------------------------------------------- #
# leaving a whole patch with markers on it: asked first, then all removed
# --------------------------------------------------------------------------- #
def _answer(hc, answer):
    """The leave-the-whole-patch confirm, answered ``answer``; the texts
    it was shown are collected."""
    asked = []
    hc._add_leave_whole_confirm = lambda text: asked.append(text) or answer
    return asked


def test_leaving_a_whole_patch_with_markers_asks_and_go_back_keeps_all(hc):
    _clicks_whole(hc)
    before = [dict(m) for m in hc._add_click_markers]
    box = hc._add_box
    hc._add_transient = None
    asked = _answer(hc, False)
    hc._set_add_box_size_fraction(0.25)           # M
    assert asked == [AP.ADD_LEAVE_WHOLE_CONFIRM.format(n=2)]
    assert asked[0] == "Leaving the whole patch removes its 2 markers."
    # Go back: the whole patch and its markers exactly as they were
    assert hc._add_whole_patch
    assert hc._add_box == box
    assert hc._add_click_markers == before
    assert hc._add_transient is None
    assert hc._add_place_markers_on is True
    # the sliders are put back on the whole patch
    assert hc.adv.last("box_size") == (box.sz, box.sy, box.sx)


def test_leaving_a_whole_patch_remove_them_clears_every_marker(hc):
    _clicks_whole(hc)
    asked = _answer(hc, True)
    hc._set_add_box_size_fraction(0.25)           # M
    assert len(asked) == 1
    assert not hc._add_whole_patch
    assert hc._add_markers() == []
    assert hc._add_transient == AP.ADD_S1_WHOLE_MARKERS_CLEARED
    assert hc._add_place_markers_on is True       # left as it was


def test_one_marker_is_named_singular(hc):
    _clicks_whole(hc)
    hc._add_click_markers = hc._add_click_markers[:1]
    asked = _answer(hc, False)
    hc._set_add_box_size_fraction(0.25)
    assert asked == ["Leaving the whole patch removes its 1 marker."]
    assert asked == [AP.ADD_LEAVE_WHOLE_CONFIRM_ONE]


def test_leaving_a_whole_patch_without_markers_asks_nothing(hc):
    hc.adv.source = (AP.SOURCE_CLICKS, None)
    hc._start_add_whole()
    asked = _answer(hc, False)
    hc._set_add_box_size_fraction(0.25)
    assert asked == []
    assert not hc._add_whole_patch
    assert hc._add_transient != AP.ADD_S1_WHOLE_MARKERS_CLEARED


def test_a_slider_that_clamps_names_the_removed_markers(hc):
    """A Numbers... slider on a Cells whole patch bigger than the Cells
    limit: the box shrinks AND the markers go -- the markers sentence is
    the one S1 shows."""
    hc.adv.params["box_budget"] = 1000
    _clicks_whole(hc)
    _answer(hc, True)
    b = hc._add_box
    hc._on_add_box_changed(b.cz, b.cy, b.cx, b.sz, b.sy, b.sx - 1)
    assert not hc._add_whole_patch
    assert hc._add_box.sz * hc._add_box.sy * hc._add_box.sx <= 1000
    assert hc._add_markers() == []
    assert hc._add_transient == AP.ADD_S1_WHOLE_MARKERS_CLEARED


def test_an_exit_while_the_leave_confirm_is_open_changes_nothing(hc):
    """Close box lands while the confirm is open (the modal spins the
    event loop): the answer finds the session gone and does nothing."""
    _clicks_whole(hc)

    def _closed_meanwhile(text):
        hc._reject_add()
        return True

    hc._add_leave_whole_confirm = _closed_meanwhile
    hc._set_add_box_size_fraction(0.25)
    assert not hc._add_running
    assert hc._add_transient != AP.ADD_S1_WHOLE_MARKERS_CLEARED


def test_the_leave_confirm_is_the_red_themed_one(monkeypatch, hc):
    from napari_epicycle.annotator.curation_3d_widget import (
        Curation3DAnnotator,
    )
    from napari_epicycle.ui_shell import theme as TH

    seen = {}

    def _fake(parent, title, text, **kw):
        seen.update(kw, parent=parent, title=title, text=text)
        return False

    monkeypatch.setattr(TH, "themed_confirm", _fake)
    assert Curation3DAnnotator._add_leave_whole_confirm(hc, "t") is False
    assert seen["parent"] is hc.adv          # the Advanced dock
    assert seen["title"] == AP.ADD_LEAVE_WHOLE_TITLE == (
        "Leaving the whole patch")
    assert seen["confirm_text"] == AP.ADD_LEAVE_WHOLE_OK == "Remove them"
    assert seen["cancel_text"] == AP.ADD_LEAVE_WHOLE_BACK == "Go back"
    assert seen["accent"] == "red"
    assert seen["default"] == "cancel"       # Enter = Go back; Escape too
    assert seen["text"] == "t"


def test_themed_confirm_can_make_cancel_the_default(qapp):
    """``default="cancel"`` makes the cancel button the Enter default; the
    old signature keeps the confirm button there."""
    from qtpy.QtWidgets import QDialog, QPushButton
    from napari_epicycle.ui_shell import theme as TH

    seen = {}

    def _exec(self):
        btns = {b.text(): b for b in self.findChildren(QPushButton)}
        seen["default"] = [t for t, b in btns.items() if b.isDefault()]
        return QDialog.Rejected

    old = QDialog.exec_
    QDialog.exec_ = _exec
    try:
        TH.themed_confirm(None, "t", "m", confirm_text="Remove them",
                          cancel_text="Go back", accent="red",
                          default="cancel")
        assert seen["default"] == ["Go back"]
        TH.themed_confirm(None, "t", "m", confirm_text="Remove them",
                          cancel_text="Go back", accent="red")
        assert seen["default"] == ["Remove them"]
    finally:
        QDialog.exec_ = old


# --------------------------------------------------------------------------- #
# the panel
# --------------------------------------------------------------------------- #
from .add_widget_fixture import qapp, reset_add, widget  # noqa: E402,F401


def test_the_button_shows_idle_on_both_presets(widget):
    reset_add(widget)
    widget.update_add_ui_state(box_active=False)
    assert not widget.add_whole_btn.isHidden()
    assert widget.add_whole_btn.text() == AP.ADD_WHOLE_BTN
    widget.set_add_preset("cells")
    run = ("labels", "cells")
    widget.set_add_marker_sources([
        (AP.ADD_ITEM_RUN.format(name="cells"), True, run),
        (AP.ADD_ITEM_NUCLEI, True, ("nuclei_masks", None)),
        (AP.ADD_ITEM_CLICKS, True, ("paint", None))])
    try:
        # a run is never pre-selected, and Whole patch is live anyway (a
        # choice that cannot fill a whole patch greys Preview instead);
        # its tooltip is plain help, never a reason
        assert widget.get_add_marker_source() != run
        assert widget.add_whole_btn.isEnabled()
        assert widget.add_whole_btn.toolTip() == AP.ADD_WHOLE_TIP
        widget.add_marker_source_combo.setCurrentIndex(0)
        assert widget.get_add_marker_source() == run
        assert not widget.add_whole_btn.isHidden()
        assert widget.add_whole_btn.isEnabled()
        assert widget.add_whole_btn.toolTip() == AP.ADD_WHOLE_TIP
        # ... where Spread cannot run in this install too; the reason is
        # in Advanced (numba missing, or its cache folder)
        widget.update_add_ui_state(numba_ok=False, numba_why="missing")
        assert widget.add_whole_btn.isEnabled()
        assert widget.add_whole_btn.toolTip() == AP.ADD_WHOLE_TIP
        widget.update_add_ui_state(numba_ok=False, numba_why="cache")
        assert AP.ADD_NO_NUMBA_CACHE in widget._add_advanced_help.text()
        assert AP.ADD_NO_NUMBA not in widget._add_advanced_help.text()
    finally:
        widget.update_add_ui_state(numba_ok=True, numba_why="")
        widget.set_add_marker_sources([])
        reset_add(widget)
    widget.update_add_ui_state(box_active=True)
    try:
        assert widget.add_whole_btn.isHidden()
    finally:
        widget.update_add_ui_state(box_active=False)


def test_on_whole_patch_recenter_is_grey_and_the_box_clear_is_gone(widget):
    reset_add(widget)
    widget.update_add_ui_state(box_active=True, whole_patch=True)
    try:
        assert not widget.add_recenter_btn.isEnabled()
        assert not widget.add_clear_box_action.isVisible()
        assert widget.add_clear_patch_action.isVisible()
        assert widget._add_box_help.text() == AP.ADD_HELP_WHOLE
    finally:
        widget.update_add_ui_state(whole_patch=False)
    assert widget.add_recenter_btn.isEnabled()
    assert widget.add_clear_box_action.isVisible()
    widget.update_add_ui_state(box_active=False)


def test_the_bar_is_determinate_while_busy_and_idle_after(widget):
    widget.update_add_ui_state(busy=True, progress=(3, 11))
    try:
        assert widget.add_busy_bar.maximum() == 11
        assert widget.add_busy_bar.value() == 3
        widget.update_add_ui_state(busy=True)     # a refresh keeps the step
        assert widget.add_busy_bar.maximum() == 11
    finally:
        widget.update_add_ui_state(busy=False)
    assert widget.add_busy_bar.maximum() == 1
    assert widget.add_busy_bar.value() == 0
    widget.update_add_ui_state(busy=True)
    try:
        assert widget.add_busy_bar.maximum() == 0, "no step kept: marquee"
    finally:
        widget.update_add_ui_state(busy=False)
