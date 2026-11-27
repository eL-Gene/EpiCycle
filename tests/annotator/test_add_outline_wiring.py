"""Cells on the Add page: the tissue outline, how cells grow, the whole patch.

Pinned here:

* the worker functions: the outline is measured once (the shrink-wrap from
  the dark channels, or the outline of the objects the Clears in force
  removed), its display skin is at least one voxel on every axis, the run
  count drops what the Preview's outline drops (the engine's own count),
  the outline's two refusals in the page's words, the growth engine's
  first set-up (its line, and a plain refusal where it cannot run);
* the page: the Tissue outline items (the cleared one grey, its reason in
  its words, until a Clear is in force), its help, Show the tissue outline
  and its legend, Grow by and Spread's three knobs, Bridge gaps (16 µm on a
  one-channel patch), the dark rows grey under an outline with the reason
  in the Advanced help slot, Dark below live, the dark-off note hidden, a
  whole patch hiding Grow by and keeping the box's pick, an install that
  cannot spread, the summaries, the Cells keys of ``get_add_params``;
* the orchestrator (the Qt-free harness): the cache keys, the box-info
  job then the cache, the Preview waiting for the outline, the dark
  channels gathered under an outline with the tick off, the whole patch's
  wall picture kept for the next Preview, the outline drawn in the main
  view (built hidden, then shown; hidden over candidates; gone on untick
  and at every exit), the cleared-outline choice falling back when its
  Clears are undone, the amber slot's order on Cells, the bar counting
  the steps, the warm-up line.
"""

from __future__ import annotations

import numpy as np
import pytest

from packages.epicycle_annotator import add_preview as AP
from packages.epicycle_annotator.reshape import ReshapeSpec
from packages.epicycle_annotator.reshape import outline as O
from packages.epicycle_annotator.reshape import region as R

from .add_harness import FakeAdv, FakeThread, Harness, install_fakes
from .add_widget_fixture import qapp, reset_add, widget  # noqa: F401
from .test_reshape_region import _box, _threshold, _tissue

ISO = (1.0, 1.0, 1.0)


def _world(shape=(12, 40, 40)):
    mem, nuc, cells = _tissue(shape=shape)
    return {"membrane": mem, "nuclei": nuc}, cells


# --------------------------------------------------------------------------- #
# the worker functions (pure)
# --------------------------------------------------------------------------- #
def test_the_shrink_job_is_the_engines_tissue_outline():
    ch, _cells = _world()
    thr = _threshold(ch)
    got = AP.measure_outline({"mode": "shrink", "channels": ch, "factor": 0.5,
                              "bridge_um": 10.0}, spacing=ISO,
                             dark_threshold=thr)
    want = O.tissue_outline(ch, ISO, dark_factor=0.5, bridge_um=10.0,
                            dark_threshold=thr)
    assert got.dtype == bool and np.array_equal(got, want)


def test_the_cleared_job_is_the_outline_of_the_union_of_the_clears():
    shape = (6, 20, 20)
    a = np.zeros((3, 5, 5), bool)
    a[1:, 1:4, 1:4] = True
    b = np.zeros((2, 4, 6), bool)
    b[:, 1:3, 2:5] = True
    roi_a = (slice(0, 3), slice(2, 7), slice(2, 7))
    roi_b = (slice(3, 5), slice(10, 14), slice(8, 14))
    recs = [(roi_a,) + AP.pack_mask(a), (roi_b,) + AP.pack_mask(b)]
    union = AP.cleared_union(shape, recs)
    want = np.zeros(shape, bool)
    want[roi_a] |= a
    want[roi_b] |= b
    assert np.array_equal(union, want)
    got = AP.measure_outline({"mode": "cleared", "shape": shape,
                              "cleared": recs}, spacing=ISO)
    assert np.array_equal(got, O.cleared_outline(want, ISO))
    # no Clear in force: an empty outline, not a refusal
    assert not AP.measure_outline({"mode": "cleared", "shape": shape,
                                   "cleared": []}, spacing=ISO).any()


def test_a_mask_packs_to_one_bit_per_voxel_and_back():
    rng = np.random.default_rng(3)
    m = rng.random((5, 7, 9)) > 0.5
    shape, packed = AP.pack_mask(m)
    assert packed.dtype == np.uint8 and packed.size == -(-m.size // 8)
    assert np.array_equal(AP.unpack_mask(shape, packed), m)


@pytest.mark.parametrize("spacing", [(1.0, 1.0, 1.0), (2.0, 0.5, 0.5),
                                     (0.25, 0.25, 0.25)])
def test_the_display_skin_is_thin_and_never_empty_on_an_axis(spacing):
    o = np.zeros((20, 40, 40), bool)
    o[5:15, 10:30, 10:30] = True
    shell = AP.outline_display_shell(o, spacing)
    assert shell.dtype == np.uint8
    assert set(np.unique(shell)) <= {0, 1}
    assert not (shell.astype(bool) & ~o).any(), "never outside the outline"
    # every face of the block has its skin: the caps (z) and the sides
    assert shell[5, 20, 20] == 1 and shell[14, 20, 20] == 1
    assert shell[10, 10, 20] == 1 and shell[10, 20, 29] == 1
    # and it is a skin, not the block (about 1 um thick, at least one voxel)
    assert shell[10, 20, 20] == 0
    want_xy = max(1, int(round(AP.ADD_OUTLINE_SHOW_UM / spacing[1])))
    assert shell[10, 20, 10:10 + want_xy].all()
    assert shell[10, 20, 10 + want_xy + 1] == 0


def _drop_world():
    """Two rows of cells; the outline leaves out the second row's two right
    cells, so their run markers lie outside it."""
    ch, cells = _world()
    outline = np.zeros(cells.shape, bool)
    outline[:, :24, :] = True
    outline[:, 13:24, 21:] = False
    return ch, cells, outline


def test_the_run_count_drops_what_the_preview_drops():
    ch, cells, outline = _drop_world()
    labels = np.zeros_like(cells)
    roi = _box(labels.shape)
    thr = _threshold(ch)
    params = {"preset": "cells", "dark_outside": False, "outline": "cleared",
              "growth": "compete"}
    info = AP.add_box_info(
        existing_roi=labels, dark_sources=None, dark_rois=ch, spacing=ISO,
        stride=1, dark_cache=thr, source=AP.SOURCE_RUN, run_guide_roi=cells,
        nuclei_roi=None, params=params, roi=roi, outline=outline)
    spec = ReshapeSpec(target="region", seeds="labels", walls="membrane",
                       params={"wall_channel": "membrane", "ridge": False,
                               "outside": "faces+dark", "growth": "compete",
                               "outline": "cleared"})
    rep = R.find_region_seeds(labels, {}, spec, roi=roi, dark_channels=ch,
                              seed_layer=cells, dark_threshold=thr,
                              spacing=ISO, outline=outline)
    assert info.n_outside_outline == 2
    assert info.run_count == (rep.n_found, rep.n_in_region)
    res = AP.run_add_preview(
        preset="cells", roi=roi, existing_roi=labels, next_label=1000,
        spacing=ISO, params=dict(params, source=AP.SOURCE_RUN,
                                 auto_smooth=False),
        wall_roi=ch["membrane"], dark_rois=ch, run_guide_roi=cells,
        dark_threshold=thr, outline_roi=outline)
    assert res.ok, res.refusal
    assert res.n_outside_outline == info.n_outside_outline
    assert len(res.ids) == info.run_count[1]
    note = AP.counts_note(res.counts(), source=res.source)
    assert AP.ADD_NOTE_OUTSIDE_OUTLINE.format(n=2) in note
    # the share and its amber
    assert AP.outline_drops_line(2, info.run_count[1]) == (
        AP.ADD_AMBER_OUTLINE_DROPS.format(
            p=int(round(100.0 * 2 / (2 + info.run_count[1])))))


def test_box_info_measures_the_outline_once_and_its_skin_when_asked():
    ch, cells = _world()
    labels = np.zeros_like(cells)
    job = {"mode": "shrink", "channels": ch, "factor": 0.5, "bridge_um": 10.0}
    kw = dict(existing_roi=labels, dark_sources=ch, dark_rois=ch, spacing=ISO,
              stride=1, dark_cache=None, source=AP.SOURCE_RUN,
              run_guide_roi=cells, nuclei_roi=None, roi=_box(labels.shape),
              params={"preset": "cells", "dark_outside": False,
                      "outline": "shrink"})
    info = AP.add_box_info(outline_job=job, **kw)
    assert info.outline is not None and info.outline_shell is None
    assert info.run_count is not None
    assert np.array_equal(info.outline, O.tissue_outline(
        ch, ISO, dark_factor=0.5, bridge_um=10.0,
        dark_threshold=info.dark_threshold))
    again = AP.add_box_info(outline=info.outline, want_shell=True, **kw)
    assert again.outline is None, "handed in: not measured again"
    assert np.array_equal(again.outline_shell,
                          AP.outline_display_shell(info.outline, ISO))
    # an outline chosen and none given: the run count waits for it
    assert AP.add_box_info(**kw).run_count is None


@pytest.mark.parametrize("mode,want", [
    ("shrink", AP.ADD_S1_NO_MARKERS_OUTLINE),
    # Bridge gaps and Dark below do not shape the outline of the objects
    # you cleared: its refusal points only to another outline
    ("cleared", AP.ADD_S1_NO_MARKERS_OUTLINE_CLEARED)])
def test_the_outline_refusals_are_said_in_the_pages_words(mode, want):
    ch, cells = _world()
    labels = np.zeros_like(cells)
    roi = _box(labels.shape)
    nothing = np.zeros(cells.shape, bool)
    res = AP.run_add_preview(
        preset="cells", roi=roi, existing_roi=labels, next_label=1000,
        spacing=ISO, params={"source": AP.SOURCE_RUN, "dark_outside": False,
                             "outline": mode, "auto_smooth": False},
        wall_roi=ch["membrane"], dark_rois=ch, run_guide_roi=cells,
        dark_threshold=_threshold(ch), outline_roi=nothing)
    assert res.status == "refused"
    assert res.refusal_reason == "no-markers-outline"
    assert res.refusal == want
    assert AP._PAGE_REFUSALS["outline-per-box"] == AP.ADD_S1_OUTLINE_PER_BOX
    if mode == "cleared":
        assert "Bridge gaps" not in want and "Dark below" not in want


def test_the_outline_drops_amber_on_the_cleared_outline_names_no_knob():
    assert AP.outline_drops_line(10, 38, "cleared") == (
        AP.ADD_AMBER_OUTLINE_DROPS_CLEARED.format(p=21))
    for w in ("Bridge gaps", "Dark below"):
        assert w not in AP.ADD_AMBER_OUTLINE_DROPS_CLEARED
    assert AP.outline_drops_line(10, 38, "shrink") == (
        AP.ADD_AMBER_OUTLINE_DROPS.format(p=21))


def test_spread_sets_the_engine_up_first_and_says_so_when_it_is_slow(
        monkeypatch):
    seen = []

    def warm(*, announce=None):
        announce()
        return 12.0

    monkeypatch.setattr(R, "compete_is_warm", lambda: False)
    monkeypatch.setattr(R, "warm_up_compete", warm)
    assert AP._warm_up(lambda *a: seen.append(a)) == (AP.NUMBA_OK, "")
    assert seen == [("add_warm", 0, 0)]


def test_an_install_that_cannot_spread_refuses_in_plain_words(monkeypatch):
    def broken(*, announce=None):
        raise ImportError("numba is missing")

    monkeypatch.setattr(R, "compete_is_warm", lambda: False)
    monkeypatch.setattr(R, "warm_up_compete", broken)
    ch, cells = _world()
    res = AP.run_add_preview(
        preset="cells", roi=_box(cells.shape),
        existing_roi=np.zeros_like(cells), next_label=1000, spacing=ISO,
        params={"source": AP.SOURCE_RUN, "growth": "compete",
                "dark_outside": True},
        wall_roi=ch["membrane"], dark_rois=ch, run_guide_roi=cells,
        dark_threshold=_threshold(ch))
    assert res.status == "refused" and res.refusal == AP.ADD_NO_NUMBA
    assert res.refusal_reason == "numba-missing"
    assert any("numba is missing" in m for m in res.messages), "logged"


def test_a_cache_folder_that_cannot_be_written_is_not_called_missing(
        monkeypatch):
    """numba is there but cannot keep its compiled code: the page says to
    point its cache at a folder you can write, not to reinstall; the
    engine's own sentence goes to the log."""
    from packages.epicycle_annotator.reshape import compete as CP

    def broken(*, announce=None):
        raise RuntimeError(CP.NUMBA_CACHE_FAILED)

    monkeypatch.setattr(R, "compete_is_warm", lambda: False)
    monkeypatch.setattr(R, "warm_up_compete", broken)
    ch, cells = _world()
    res = AP.run_add_preview(
        preset="cells", roi=_box(cells.shape),
        existing_roi=np.zeros_like(cells), next_label=1000, spacing=ISO,
        params={"source": AP.SOURCE_RUN, "growth": "compete",
                "dark_outside": True},
        wall_roi=ch["membrane"], dark_rois=ch, run_guide_roi=cells,
        dark_threshold=_threshold(ch))
    assert res.refusal == AP.ADD_NO_NUMBA_CACHE
    assert res.refusal_reason == "numba-cache"
    assert "NUMBA_CACHE_DIR" in AP.ADD_NO_NUMBA_CACHE
    assert "Reinstall" not in AP.ADD_NO_NUMBA_CACHE
    assert any(CP.NUMBA_CACHE_FAILED in m for m in res.messages)


@pytest.mark.parametrize("exc,key", [
    (ImportError("gone"), AP.NUMBA_MISSING_KEY),
    (RuntimeError("no locator"), AP.NUMBA_CACHE_KEY)])
def test_the_solver_check_tells_the_two_failures_apart(monkeypatch, exc,
                                                        key):
    from packages.epicycle_annotator.reshape import compete as CP

    def broken():
        raise exc

    monkeypatch.setattr(CP, "_kernels", broken)
    assert AP.numba_answer() == (key, str(exc))
    monkeypatch.setattr(CP, "_kernels", lambda: {})
    assert AP.numba_answer() == (AP.NUMBA_OK, "")


def test_the_result_declares_its_outline_fields():
    fields = AP.AddPreviewResult.__dataclass_fields__
    for key in ("refusal_reason", "n_outside_outline", "n_slivers",
                "n_tiles"):
        assert key in fields, key
    assert {"n_outside_outline", "n_slivers"} <= set(
        AP.AddPreviewResult().counts())


def test_the_new_count_parts_have_their_singular():
    for key, many, one in AP._COUNT_PARTS:
        if key in ("n_outside_outline", "n_slivers"):
            assert AP.count_phrase(1, one, many) == one
            assert "2 " in AP.count_phrase(2, one, many)


def test_the_walls_summary_names_the_outline():
    assert AP.walls_summary(channel="membrane", outline="shrink").endswith(
        AP.ADD_SUM_OUTLINE_SHRINK)
    assert AP.walls_summary(outline="cleared").endswith(
        AP.ADD_SUM_OUTLINE_CLEARED)
    assert "outline" not in AP.walls_summary(channel="m", outline="none")


def test_the_whole_patch_lines():
    # one line for both presets: a whole patch is filled in one go
    assert AP.box_line((10, 20, 30), preset="cells", empty_share=0.25,
                       whole_patch=True) == (
        AP.ADD_S1_PATCH_PLAIN.format(z=10, y=20, x=30, m="0.1", p=25))
    assert AP.run_count_line(7, 5, whole_patch=True) == (
        AP.ADD_RUN_COUNT_PATCH.format(n=7, m=5))
    assert AP.run_count_line(7, 1, whole_patch=True) == (
        AP.ADD_RUN_COUNT_PATCH_ONE.format(n=7))
    assert AP.outline_drops_line(5, 95) == ""          # 5 %: not over
    assert AP.outline_drops_line(6, 94) != ""


def test_no_banned_word_on_the_page():
    for name, value in vars(AP).items():
        if name.startswith("ADD_") and isinstance(value, str):
            low = value.lower()
            for word in ("envelope", "geodesic", "provenance", " roi",
                         " lut"):
                assert word not in low, (name, word)


# --------------------------------------------------------------------------- #
# the page
# --------------------------------------------------------------------------- #
def _cells_page(w, *, box=True, whole=False, in_force=False, numba=True,
                single=False):
    reset_add(w)
    w.set_add_preset("cells")
    w.update_add_ui_state(box_active=box, whole_patch=whole,
                          cleared_in_force=in_force, numba_ok=numba,
                          single_channel=single)
    w.add_walls_btn.setChecked(True)


def _undo_page(w):
    w.update_add_ui_state(box_active=False, whole_patch=False,
                          cleared_in_force=False, numba_ok=True,
                          single_channel=False, outline_pending=False)
    w.add_show_outline_cb.setChecked(False)
    w.add_grow_combo.setCurrentIndex(0)
    w.add_outline_combo.setCurrentIndex(2)
    w.add_dark_outside_cb.setChecked(True)
    w.add_walls_btn.setChecked(False)
    reset_add(w)


def _item(combo, i):
    return combo.model().item(i)


def test_the_outline_items_read_by_index_and_the_cleared_one_waits(widget):
    _cells_page(widget)
    try:
        c = widget.add_outline_combo
        assert [c.itemText(i) for i in range(c.count())] == [
            AP.ADD_OUTLINE_CLEARED_GREY, AP.ADD_OUTLINE_SHRINK,
            AP.ADD_OUTLINE_NONE]
        assert not _item(c, 0).isEnabled()
        # the default on a box as on a whole patch: the shrink-wrap
        assert widget.get_add_outline() == AP.OUTLINE_SHRINK
        widget.update_add_ui_state(cleared_in_force=True)
        assert c.itemText(0) == AP.ADD_OUTLINE_CLEARED
        assert _item(c, 0).isEnabled()
        for i, key in enumerate(AP.OUTLINE_KEYS):
            c.setCurrentIndex(i)
            assert widget.get_add_outline() == key
            assert widget._add_outline_help.text() == AP.outline_help(key)
            assert not widget._add_outline_help.isHidden()
        assert not widget._add_outline_row.isHidden()
    finally:
        _undo_page(widget)


def test_the_outline_help_hides_over_candidates(widget):
    _cells_page(widget)
    try:
        widget.update_add_ui_state(candidate_ids=[5], candidate_sizes={5: 9},
                                   candidate_edge_ids=set())
        assert widget._add_outline_help.isHidden()
    finally:
        widget.update_add_ui_state(candidate_ids=[], candidate_sizes={},
                                   candidate_edge_ids=set())
        _undo_page(widget)


def test_show_the_outline_needs_an_outline_and_says_where(widget):
    _cells_page(widget)
    try:
        widget.add_outline_combo.setCurrentIndex(2)   # the user's "none"
        assert widget.add_show_outline_cb.isHidden()
        widget.add_outline_combo.setCurrentIndex(1)
        assert not widget.add_show_outline_cb.isHidden()
        assert widget._add_show_outline_legend.isHidden()
        widget.add_show_outline_cb.setChecked(True)
        assert widget._add_show_outline_legend.text() == (
            AP.ADD_HELP_SHOW_OUTLINE)
        widget.update_add_ui_state(outline_pending=True)
        assert widget._add_show_outline_legend.text() == (
            AP.ADD_OUTLINE_PENDING)
        widget.add_outline_combo.setCurrentIndex(2)
        assert widget.add_show_outline_cb.isHidden()
        assert widget._add_show_outline_legend.isHidden()
    finally:
        _undo_page(widget)


def test_spread_and_bridge_rows_follow_their_choices(widget):
    _cells_page(widget)
    widget._add_advanced_toggle.setChecked(True)
    try:
        widget.add_outline_combo.setCurrentIndex(2)   # the user's "none"
        assert widget._add_spread_adv.isHidden()
        assert widget._add_bridge_row.isHidden()
        widget.add_grow_combo.setCurrentIndex(1)
        assert widget.get_add_grow() == AP.GROW_SPREAD
        assert not widget._add_spread_adv.isHidden()
        widget.add_outline_combo.setCurrentIndex(1)
        assert not widget._add_bridge_row.isHidden()
        assert widget.add_bridge_sb.value() == AP.ADD_BRIDGE_UM
        title = str(widget._add_advanced_toggle.property(
            "_epicycle_chevron_title"))
        assert "spreading" in title and "bridge 10" in title
    finally:
        widget._add_advanced_toggle.setChecked(False)
        _undo_page(widget)


def test_the_dark_rows_under_an_outline(widget):
    _cells_page(widget)
    widget._add_advanced_toggle.setChecked(True)
    try:
        widget.add_outline_combo.setCurrentIndex(2)   # the user's "none"
        widget.add_dark_outside_cb.setChecked(False)
        assert not widget.add_dark_factor_sb.isEnabled()
        assert not widget.add_dark_off_note.isHidden()
        assert widget._add_advanced_help.text().startswith(AP.ADD_DARK_OFF)
        widget.add_outline_combo.setCurrentIndex(1)
        # the outline decides what is outside: the tick and the reach grey,
        # Dark below live (it sets the shrink-wrap), the reason on screen
        assert not widget.add_dark_outside_cb.isEnabled()
        assert not widget.add_dark_clearance_sb.isEnabled()
        assert widget.add_dark_factor_sb.isEnabled()
        assert widget.add_dark_off_note.isHidden()
        assert widget._add_dark_help.isHidden()
        assert widget._add_advanced_help.text().startswith(
            AP.ADD_DARK_BY_OUTLINE)
        assert not widget._add_advanced_help.isHidden()
        # Spread with no outline and the tick off: the open-background note
        widget.add_outline_combo.setCurrentIndex(2)
        widget.add_grow_combo.setCurrentIndex(1)
        assert AP.ADD_NOTE_SPREAD_OPEN in widget._add_advanced_help.text()
    finally:
        widget._add_advanced_toggle.setChecked(False)
        _undo_page(widget)


def test_a_whole_patch_shows_grow_by_and_keeps_the_box_pick(widget):
    _cells_page(widget)
    try:
        g = widget.add_grow_combo
        assert widget.get_add_grow() == AP.GROW_FLOOD
        widget.update_add_ui_state(whole_patch=True)
        # Grow by is the user's pick on a whole patch too: the row shows,
        # both items pickable, and the pick is one for the next box
        assert not widget._add_grow_row.isHidden()
        assert _item(g, 1).isEnabled()
        assert g.itemText(0) == AP.ADD_GROW_FLOOD
        assert _item(g, 0).isEnabled()
        assert widget.get_add_grow() == AP.GROW_FLOOD
        # the default outline on a whole patch: the shrink-wrap, or the
        # outline of the objects cleared when a Clear is in force
        assert widget.get_add_outline() == AP.OUTLINE_SHRINK
        widget.update_add_ui_state(cleared_in_force=True)
        assert widget.get_add_outline() == AP.OUTLINE_CLEARED
        p = widget.get_add_params()
        assert p["growth"] == "flood" and p["outline"] == "cleared"
        assert AP.cells_growth(dict(p, whole_patch=True)) == AP.GROW_PATCH
        widget.update_add_ui_state(whole_patch=False)
        assert not widget._add_grow_row.isHidden()
        assert g.itemText(0) == AP.ADD_GROW_FLOOD and _item(g, 0).isEnabled()
        assert widget.get_add_grow() == AP.GROW_FLOOD, "the earlier pick"
        # one default for every region: the cleared outline stays (a
        # Clear is still in force)
        assert widget.get_add_outline() == AP.OUTLINE_CLEARED
    finally:
        _undo_page(widget)


def test_a_chosen_outline_is_never_overwritten_by_a_default(widget):
    _cells_page(widget)
    try:
        widget.add_outline_combo.setCurrentIndex(2)   # the user's "none"
        widget.add_outline_combo.setCurrentIndex(1)   # then "shrink"
        widget.add_outline_combo.setCurrentIndex(2)   # then "none" again
        widget.update_add_ui_state(whole_patch=True, cleared_in_force=True)
        assert widget.get_add_outline() == AP.OUTLINE_NONE
    finally:
        _undo_page(widget)


def test_an_install_that_cannot_spread_greys_spread_with_its_reason(widget):
    _cells_page(widget, numba=False)
    widget._add_advanced_toggle.setChecked(True)
    try:
        g = widget.add_grow_combo
        assert g.itemText(1) == AP.ADD_GROW_SPREAD_GREY
        assert not _item(g, 1).isEnabled()
        assert AP.ADD_NO_NUMBA in widget._add_advanced_help.text()
        # Whole patch stays live: on a whole patch Preview says why
        assert widget.add_whole_btn.isEnabled()
    finally:
        widget._add_advanced_toggle.setChecked(False)
        _undo_page(widget)


def test_bridge_gaps_starts_at_16_on_one_channel_unless_set(widget):
    _cells_page(widget, single=True)
    try:
        assert widget.add_bridge_sb.value() == AP.ADD_BRIDGE_UM_ONE_CHANNEL
        widget.update_add_ui_state(single_channel=False)
        assert widget.add_bridge_sb.value() == AP.ADD_BRIDGE_UM
        widget.add_bridge_sb.setValue(12.0)           # the user's value
        widget.update_add_ui_state(single_channel=True)
        assert widget.add_bridge_sb.value() == 12.0
    finally:
        widget.clear_add_touched()
        widget.update_add_ui_state(single_channel=False)
        _undo_page(widget)


def test_the_cells_keys_ride_on_cells_only(widget):
    _cells_page(widget)
    try:
        p = widget.get_add_params()
        assert p["growth"] == "flood" and p["outline"] == "shrink"
        assert p["wall_cost"] == 40.0 and p["growth_limit"] == 28.0
        assert p["include_membrane"] is True and p["bridge_um"] == 10.0
    finally:
        _undo_page(widget)
    for key in ("growth", "outline", "bridge_um", "wall_cost"):
        assert key not in widget.get_add_params()


def test_the_knob_ranges_are_the_engines(widget):
    from packages.epicycle_annotator.reshape import spec as S

    for attr, key in (("add_wall_cost_sb", "wall_cost"),
                      ("add_growth_limit_sb", "growth_limit"),
                      ("add_bridge_sb", "bridge_um")):
        sb = getattr(widget, attr)
        assert (sb.minimum(), sb.maximum()) == S._REGION_RANGES[key], attr
    assert widget.add_wall_cost_sb.value() == S.DEFAULT_PARAMS["wall_cost"]
    assert widget.add_growth_limit_sb.value() == (
        S.DEFAULT_PARAMS["growth_limit"])


def test_the_walls_summary_on_the_page(widget):
    _cells_page(widget)
    try:
        widget.add_outline_combo.setCurrentIndex(1)
        assert str(widget.add_walls_btn.property(
            "_epicycle_chevron_title")).endswith(AP.ADD_SUM_OUTLINE_SHRINK)
    finally:
        _undo_page(widget)


def test_the_new_combos_are_sized_like_the_markers_combo(widget):
    m = widget.add_marker_source_combo
    for c in (widget.add_outline_combo, widget.add_grow_combo):
        assert c.sizeAdjustPolicy() == m.sizeAdjustPolicy()
        assert c.minimumContentsLength() == m.minimumContentsLength()
        assert c.sizePolicy().horizontalPolicy() == (
            m.sizePolicy().horizontalPolicy())


def test_the_whole_patch_run_count_says_patch(widget):
    _cells_page(widget, whole=True)
    try:
        widget.update_add_ui_state(run_count=(12, 9))
        assert widget.add_run_count_label.text() == (
            AP.ADD_RUN_COUNT_PATCH.format(n=12, m=9))
    finally:
        _undo_page(widget)


# --------------------------------------------------------------------------- #
# the orchestrator (the Qt-free harness)
# --------------------------------------------------------------------------- #
class _CellsAdv(FakeAdv):
    """The Cells page as far as the orchestrator reads it."""

    def __init__(self):
        super().__init__()
        self.preset = "cells"
        self.source = ("labels", "nuc")
        self.params.update(dark_outside=False, outline="shrink",
                           growth="flood", wall_cost=40.0, growth_limit=14.0,
                           include_membrane=True, bridge_um=10.0,
                           dark_otsu_factor=0.5)
        self.wall = dict(self.wall, channel="membrane")
        self._add_touched = set()
        self.show = _Tick()
        self.add_show_outline_cb = self.show

    def get_add_outline(self):
        return self.params["outline"]

    def set_add_outline(self, key):
        self.params["outline"] = key


class _Tick:
    def __init__(self):
        self.on = False

    def isChecked(self):              # noqa: N802 - Qt name
        return self.on

    def setChecked(self, on):         # noqa: N802 - Qt name
        self.on = bool(on)

    def blockSignals(self, _b):       # noqa: N802 - Qt name
        return False


class _Layer:
    def __init__(self):
        self.visible = False


@pytest.fixture
def h(monkeypatch):
    install_fakes(monkeypatch)
    ch, cells = _world()
    hh = Harness(np.zeros(cells.shape, np.int32), images=ch)
    hh.adv = _CellsAdv()
    hh.ctx["spacing_source"] = "qc_settings"
    hh._add_run_guide = cells
    hh.ch = ch
    hh.added = []
    hh.removed = []

    def _temp(key, array, **kw):
        layer = _Layer()
        hh.added.append((key, array, kw, layer))
        return layer

    hh._add_temp_layer = _temp
    hh._remove_temp_layer = lambda key: hh.removed.append(key)
    hh._add_dark_cache = {hh._add_dark_key(): _threshold(ch)}
    return hh


def _land_info(h):
    for _ in range(3):
        th = getattr(h, "_add_info_thread", None)
        if th is None:
            return
        th.run()
        th.end()


def test_the_outline_keys(h):
    h.place_box((slice(0, 12), slice(0, 20), slice(0, 20)))
    k = h._add_outline_key()
    assert k[0] == "shrink" and k[1] == "P" and k[3] == ("membrane",
                                                         "nuclei")
    h.adv.params["bridge_um"] = 12.0
    assert h._add_outline_key() != k
    h.adv.params["bridge_um"] = 10.0
    h.adv.params["dark_otsu_factor"] = 0.4
    assert h._add_outline_key() != k
    h.adv.params["dark_otsu_factor"] = 0.5
    assert h._add_outline_key() == k
    # the box is not in the key: the outline is the patch's
    h.place_box((slice(0, 6), slice(0, 10), slice(0, 10)))
    assert h._add_outline_key() == k
    h.ctx["patch_name"] = "Q"
    assert h._add_outline_key() != k
    # the cleared outline: keyed by the Clears in force
    h.ctx["patch_name"] = "P"
    h.adv.params["outline"] = "cleared"
    h._add_clear_records = {4: (7,), 9: (8,)}
    h._get_live_add_clears_cb = lambda: {4, 9}
    assert h._add_outline_key()[3] == (4, 9)
    h._get_live_add_clears_cb = lambda: {9}
    assert h._add_outline_key()[3] == (9,)
    h._get_live_add_clears_cb = lambda: set()
    assert h._add_outline_mode() == "none" and h._add_outline_key() is None


def test_the_info_job_measures_once_then_the_cache_is_handed_in(h):
    h.place_box((slice(0, 12), slice(0, 20), slice(0, 20)))
    h._request_add_box_info()
    w = h._add_info_worker
    assert "outline_job" in w._kwargs and "outline" not in w._kwargs
    assert w._kwargs["outline_job"]["mode"] == "shrink"
    assert set(w._kwargs["outline_job"]["channels"]) == {"membrane",
                                                         "nuclei"}
    _land_info(h)
    assert h._add_cached_outline() is not None
    h._request_add_box_info()
    w = h._add_info_worker
    assert "outline_job" not in w._kwargs
    assert w._kwargs["outline"] is h._add_cached_outline()
    _land_info(h)


def test_an_outline_for_other_inputs_is_dropped(h):
    h.place_box((slice(0, 12), slice(0, 20), slice(0, 20)))
    h._request_add_box_info()
    h.adv.params["bridge_um"] = 14.0          # changed while it measured
    _land_info(h)
    cache = getattr(h, "_add_outline_cache", None)
    assert cache is None or cache["key"] == h._add_outline_key()
    assert h._add_cached_outline() is None


def test_the_preview_waits_for_the_outline(h):
    h.place_box((slice(0, 12), slice(0, 20), slice(0, 20)))
    kw, why = h._add_gather_preview()
    assert kw is None and why == AP.ADD_OUTLINE_PENDING
    h._add_refresh_ui()
    assert h.adv.last("status_text") == AP.ADD_OUTLINE_PENDING
    assert h.adv.last("outline_pending") is True


def test_an_outline_gathers_the_dark_channels_with_the_tick_off(h):
    h.place_box((slice(0, 12), slice(0, 20), slice(0, 20)))
    h._request_add_box_info()
    _land_info(h)
    kw, why = h._add_gather_preview()
    assert why is None, why
    assert kw["params"]["dark_outside"] is False
    assert kw["params"]["outline"] == "shrink"
    assert set(kw["dark_rois"]) == {"membrane", "nuclei"}
    assert kw["dark_threshold"] is not None
    assert kw["outline_roi"] is h._add_cached_outline()
    # no outline, tick off: no dark channels, as before
    h.adv.params["outline"] = "none"
    h._add_info = None
    h._request_add_box_info()
    _land_info(h)
    kw, why = h._add_gather_preview()
    assert why is None and kw["dark_rois"] is None
    assert kw["outline_roi"] is None


def test_a_whole_patch_fills_to_the_walls_and_keeps_its_wall_picture(h):
    h._start_add_whole()
    assert h._add_whole_patch
    _land_info(h)
    kw, why = h._add_gather_preview()
    assert why is None, why
    p = kw["params"]
    assert p["whole_patch"] is True
    assert AP.cells_growth(p) == AP.GROW_PATCH
    assert kw["elevation_roi"] is None, "the first Preview builds it"
    # the whole patch hands the layers' own arrays to the worker
    assert kw["wall_roi"] is h.viewer.layers[0].data \
        or kw["wall_roi"] is h.viewer.layers[1].data
    res = AP.AddPreviewResult(preset="cells")
    res.elevation = np.zeros(h.sync_manager.shared_labels["L"].shape,
                             np.float32)
    h._add_keep_elevation(res)
    kw, why = h._add_gather_preview()
    assert kw["elevation_roi"] is res.elevation
    # another wall channel preparation: built again
    h.adv.wall = dict(h.adv.wall, chain="zcorrect")
    kw, why = h._add_gather_preview()
    assert kw["elevation_roi"] is None


def test_the_s1_line_of_the_whole_patch(h):
    h._start_add_whole()
    _land_info(h)
    h._add_refresh_ui()
    s = h._add_s1_state()
    assert "n_pieces" not in s
    assert h.adv.last("status_text").startswith("Whole patch ")
    assert "piece" not in h.adv.last("status_text")


def test_leaving_a_cells_whole_patch_shrinks_the_box_and_says_so(h):
    h.adv.params["box_budget"] = 1000
    h._start_add_whole()
    b = h._add_box
    h._on_add_box_changed(b.cz, b.cy, b.cx, b.sz, b.sy, b.sx - 1)
    assert not h._add_whole_patch
    n = h._add_box.sz * h._add_box.sy * h._add_box.sx
    assert n <= 1000
    assert h._add_transient == AP.ADD_S1_LEFT_WHOLE_CLAMPED
    assert h.adv.last("box_size") == (h._add_box.sz, h._add_box.sy,
                                      h._add_box.sx)
    # S / M / L from a whole patch: the same words when it had to shrink
    h._start_add_whole()
    h._set_add_box_size_fraction(0.5)
    assert not h._add_whole_patch
    assert h._add_transient == AP.ADD_S1_LEFT_WHOLE_CLAMPED
    # under the limit nothing is said
    h.adv.params["box_budget"] = 10 ** 9
    h._start_add_whole()
    h._set_add_box_size_fraction(0.25)
    assert h._add_transient != AP.ADD_S1_LEFT_WHOLE_CLAMPED


def test_the_bar_counts_the_steps_and_the_fill_moves(h):
    from napari_epicycle.annotator.reshape_worker import stage_text

    h._start_add_whole()
    h._on_add_progress("add_markers", 2, 4)
    assert h.adv.pushes[-1]["progress"] == (2, 4)
    # the wall picture is one long step (longer on a bigger patch, longest
    # with the ridge wall types): the bar moves while the line names it
    h._on_add_progress("elevation", 2, 4)
    assert h.adv.pushes[-1]["progress"] == ()
    assert h.adv.pushes[-1]["stage_text"] == stage_text("elevation")
    # the fill itself cannot be stopped and has no steps of its own: the
    # bar moves to and fro while the line names it
    h._on_add_progress("add_cut", 4, 4)
    assert h.adv.pushes[-1]["progress"] == ()
    assert h.adv.pushes[-1]["stage_text"] == stage_text("add_cut")
    assert "Piece" not in h.adv.pushes[-1]["stage_text"]


def test_the_warm_up_line_paints_s1_while_it_runs(h):
    h.place_box((slice(0, 12), slice(0, 20), slice(0, 20)))
    h._add_busy = True
    h._on_add_progress("add_warm", 0, 0)
    assert h.adv.last("status_text") == AP.ADD_S1_WARMING
    h._on_add_progress("add_box", 1, 4)
    assert h._add_warm_text is None


def _show(h):
    h.place_box((slice(0, 12), slice(0, 20), slice(0, 20)))
    h.adv.show.on = True
    h._on_add_show_outline(True)
    _land_info(h)
    h._add_refresh_ui()


def test_show_draws_the_skin_in_the_main_view_built_hidden(h):
    _show(h)
    assert h.added, "drawn"
    key, array, kw, layer = h.added[-1]
    assert key == "add_outline" and kw["kind"] == "image"
    assert kw["image_kw"]["visible"] is False, "built hidden, then shown"
    assert kw["image_kw"]["blending"] == "additive"
    assert array is h._add_cached_outline(shell=True)
    assert layer.visible is True
    # drawn once: a refresh does not add it again
    n = len(h.added)
    h._add_refresh_ui()
    assert len(h.added) == n
    # stays on screen while candidates are up (what the cells grew to)
    h._add_preview_result = np.zeros((1, 1, 1), np.int32)
    h._add_refresh_ui()
    assert layer.visible is True
    h._add_preview_result = None
    h._add_refresh_ui()
    assert layer.visible is True


def test_untick_removes_the_skin(h):
    _show(h)
    h.adv.show.on = False
    h._on_add_show_outline(False)
    assert "add_outline" in h.removed
    assert h._add_outline_layer is None


@pytest.mark.parametrize("exit_", ["cancel", "switch", "close", "accept",
                                   "outline_none"])
def test_every_exit_removes_the_skin(h, exit_):
    _show(h)
    h.removed.clear()
    if exit_ == "cancel":
        h._reject_add()
    elif exit_ == "switch":
        h._teardown_add_for_switch()
    elif exit_ == "close":
        h._reset_add_state()
    elif exit_ == "outline_none":
        h.adv.params["outline"] = "none"
        h._on_add_outline_changed("none")
    else:
        # the real Accept: a Preview, then Accept
        h._preview_add()
        th = h._add_thread
        th.run()
        th.end()
        res = h._add_preview_res
        assert res is not None and res.ok, getattr(res, "refusal", None)
        h.adv.checked = list(res.ids)
        h._on_add_committed_cb = None
        h._accept_add()
        assert h._add_review
        assert h.adv.show.on is True, "Accept keeps the tick"
    assert "add_outline" in h.removed, exit_
    if exit_ in ("cancel", "switch", "close"):
        # a full exit: the tick goes down
        assert h.adv.show.on is False
    if exit_ == "cancel":
        # the outline is the patch's: kept for the next session, packed,
        # its skin dropped
        cache = h._add_outline_cache
        assert cache is not None and cache["outline"] is None
        assert cache["shell"] is None and cache["packed"] is not None
    if exit_ == "switch":
        # another patch: gone
        assert h._add_outline_cache is None


# Accept keeping the tick and the skin never being a labels-key target are
# pinned by behaviour in test_add_c2_fixpass.py (the real Accept path, the
# labels key's own target list).


def test_a_real_image_layer_is_built_hidden_then_shown():
    from types import SimpleNamespace

    from napari.components import ViewerModel

    from napari_epicycle.annotator import curation_3d_widget as CW

    vm = ViewerModel()
    host = SimpleNamespace(viewer=vm, _temp_guide_layers={},
                           sync_manager=None, _active_labels_name="")
    host._remove_temp_layer = lambda key: CW.Curation3DAnnotator.\
        _remove_temp_layer(host, key)
    shell = np.zeros((4, 6, 6), np.uint8)
    shell[1, 1:5, 1:5] = 1
    layer = CW.Curation3DAnnotator._add_temp_layer(
        host, "add_outline", shell, kind="image", name="guide_add_outline",
        image_kw=dict(opacity=0.3, blending="additive", colormap="cyan",
                      contrast_limits=(0, 1), visible=False))
    assert layer is not None and layer in vm.layers
    assert getattr(layer, "_epicycle_temporary", False)
    assert layer.visible is True
    assert abs(layer.opacity - 0.3) < 1e-9 and layer.blending == "additive"
    host._remove_temp_layer("add_outline")
    assert layer not in vm.layers and layer.visible is False


def test_the_cleared_choice_falls_back_when_its_clears_are_undone(h):
    h.place_box((slice(0, 12), slice(0, 20), slice(0, 20)))
    h.adv.params["outline"] = "cleared"
    h._add_clear_records = {4: (7,)}
    live = {4}
    h._get_live_add_clears_cb = lambda: set(live)
    h._add_refresh_ui()
    assert h.adv.params["outline"] == "cleared"
    live.clear()                         # the Clear was undone
    h._add_refresh_ui()
    # chosen by default on a box: back to the default, the shrink-wrap
    assert h.adv.params["outline"] == "shrink"
    assert h.adv.last("status_text") == AP.ADD_S1_OUTLINE_CLEARED_GONE.format(
        choice=AP.ADD_OUTLINE_SHRINK)
    assert h.adv.last("cleared_in_force") is False
    # on a whole patch the user had left at its default: the shrink-wrap
    h._start_add_whole()
    h.adv.params["outline"] = "cleared"
    h._add_clear_records = {5: (8,)}
    live.add(5)
    h._add_refresh_ui()
    live.clear()
    h._add_refresh_ui()
    assert h.adv.params["outline"] == "shrink"


def test_a_clear_keeps_what_it_zeroed_until_its_undo_leaves(monkeypatch):
    install_fakes(monkeypatch)
    vol = np.zeros((6, 12, 12), np.int32)
    vol[1:3, 2:5, 2:5] = 3
    vol[4, 6:9, 6:9] = 4
    before = vol.copy()
    h = Harness(vol)
    h.place_box((slice(0, 6), slice(0, 12), slice(0, 12)))
    live = set()

    def _cb(ids, **kw):
        live.add(kw["extra"]["add_clear_seq"])

    h._on_discard_garbage_cb = _cb
    h._get_live_add_clears_cb = lambda: set(live)
    plan = AP.clear_plan(labels=vol, mode="patch")
    h._apply_add_clear(plan)
    (seq,) = list(h._add_clear_masks)
    roi, shape, packed = h._add_clear_masks[seq]
    m = np.zeros(vol.shape, bool)
    m[roi] = AP.unpack_mask(shape, packed)
    assert np.array_equal(m, np.isin(before, [3, 4]))
    # the undo stack let the entry go (20 Accepts later, an Extract): the
    # objects are still gone, so the Clear is still in force
    live.clear()
    assert h._add_cleared_in_force() == [(seq, (3, 4))]
    assert seq in h._add_clear_masks
    # judged once: asked again, nothing is measured again
    assert h._add_clear_kept.get(seq) == id(vol)
    # new objects drawn where they were do not bring them back (judged
    # again from the voxels)
    vol[1, 2, 2] = 999
    h._add_clear_kept = {}
    assert h._add_cleared_in_force() == [(seq, (3, 4))]


def test_an_undone_clear_is_dropped_when_its_objects_are_back(monkeypatch):
    install_fakes(monkeypatch)
    vol = np.zeros((6, 12, 12), np.int32)
    vol[1:3, 2:5, 2:5] = 3
    before = vol.copy()
    h = Harness(vol)
    h.place_box((slice(0, 6), slice(0, 12), slice(0, 12)))
    live = set()
    h._on_discard_garbage_cb = lambda ids, **kw: live.add(
        kw["extra"]["add_clear_seq"])
    h._get_live_add_clears_cb = lambda: set(live)
    h._apply_add_clear(AP.clear_plan(labels=vol, mode="patch"))
    (seq,) = list(h._add_clear_masks)
    vol[...] = before                     # the main Undo pasted them back
    live.clear()
    assert h._add_cleared_in_force() == []
    assert h._add_clear_masks == {}
    assert h._add_clears_back is True


def test_the_cleared_outline_on_an_evicted_clear_stays_and_says_nothing(h):
    """An evicted Clear keeps its outline: no fall-back, no "are back"."""
    shape = h.sync_manager.shared_labels["L"].shape
    h.place_box(tuple(slice(0, n) for n in shape))
    h.adv.params["outline"] = "cleared"
    m = np.zeros(shape, bool)
    m[2:4, 2:6, 2:6] = True
    h._add_clear_records = {4: (7,)}
    h._add_clear_masks = {4: (tuple(slice(0, n) for n in shape),)
                          + AP.pack_mask(m)}
    live = {4}
    h._get_live_add_clears_cb = lambda: set(live)
    h._add_refresh_ui()
    live.clear()                          # evicted; the voxels stay 0
    h._add_transient = None
    h._add_refresh_ui()
    assert h.adv.params["outline"] == "cleared"
    assert h._add_transient is None
    # the objects come back (an Undo pasted id 7): the fall-back, said once
    h.sync_manager.shared_labels["L"][2, 3, 3] = 7
    h._add_clear_kept = {}
    h._add_refresh_ui()
    assert h.adv.params["outline"] == "shrink"
    assert h.adv.last("status_text") == AP.ADD_S1_OUTLINE_CLEARED_GONE.format(
        choice=AP.ADD_OUTLINE_SHRINK)


def test_the_amber_slot_on_cells_in_its_order(h, monkeypatch):
    h.place_box((slice(0, 12), slice(0, 20), slice(0, 20)))
    info = AP.AddBoxInfo(run_count=(40, 38), n_outside_outline=10)
    h._add_info = info
    h.adv.source = ("labels", "nuc")
    h.ctx["downsample_factor"] = 2
    assert h._add_derive_note_text() == AP.ADD_NOTE_HALFSIZE_BOXED
    h.ctx["downsample_factor"] = 1
    monkeypatch.setattr(type(h), "_add_image_channel_names",
                        lambda self: ["only"], raising=False)
    # Fill picked (a box), the shrink-wrap: Spread, Wall cost, Bridge gaps
    assert h._add_derive_note_text() == AP.one_channel_note(spread=False,
                                                            shrink=True)
    assert "Bridge gaps" in h._add_derive_note_text()
    assert "pick Grow by ▸ Spread" in h._add_derive_note_text()
    # no outline: Bridge gaps is not on the page, so not named
    h.adv.params["outline"] = "none"
    assert "Bridge gaps" not in h._add_derive_note_text()
    # Spread picked already: only Wall cost
    h.adv.params["growth"] = "compete"
    assert h._add_derive_note_text() == AP.one_channel_note(spread=True,
                                                            shrink=False)
    assert "pick Grow by" not in h._add_derive_note_text()
    h.adv.params.update(outline="shrink", growth="flood")
    monkeypatch.setattr(type(h), "_add_image_channel_names",
                        lambda self: ["membrane", "nuclei"], raising=False)
    assert h._add_derive_note_text() == AP.outline_drops_line(10, 38)
    info.n_outside_outline = 1
    assert h._add_derive_note_text() == ""
    monkeypatch.setattr(AP, "ADD_SLOW_PATCH_VOXELS", 10)
    h._start_add_whole()
    h._add_info = AP.AddBoxInfo(run_count=(40, 39), n_outside_outline=1)
    n = int(np.prod(h.sync_manager.shared_labels["L"].shape))
    assert h._add_derive_note_text() == AP.patch_slow_line(
        n, h._add_wall_params().get("wall_kind"))
    # never over candidates
    h._add_preview_result = np.zeros((1, 1, 1), np.int32)
    assert h._add_derive_note_text() == ""


def test_the_whole_patch_says_its_run_markers_are_in_the_patch(h):
    """The panel words the count per box or per patch (the page reads its
    own whole-patch flag); the harness pushes the flag."""
    h._start_add_whole()
    assert h.adv.last("whole_patch") is True
