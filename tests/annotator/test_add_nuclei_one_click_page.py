"""Nuclei ▸ "At the markers I place": one click, one nucleus -- the page.

"Separate touching objects" (one tick, shared with Anywhere in the box)
is the markers' on / off: unticked, the markers behave as before (the
engine gets ``split="off"``); ticked, the engine gets the one mode named
by ``add_preview.ADD_MARKER_SPLIT_DEFAULT``.  The markers have their own
Spacing box (``add_marker_spacing_sb``), shown in place of Anywhere's; the
gather turns its number N into microns, N x the xy voxel size / the
half-size factor, so N means the same tissue distance at half size.

Tested through the real page widget (offscreen), the real orchestrator
gather (the Qt-free harness borrows it) and the real preview; tests that
depend on the default read the constant, and each mode is run through the
preview by name.
"""

from __future__ import annotations

import numpy as np
import pytest

from packages.epicycle_annotator import add_preview as AP
from packages.epicycle_annotator import add_segment as AS

from .add_harness import Harness, install_fakes
from .add_widget_fixture import qapp, reset_add, widget  # noqa: F401

MARKERS = (AP.LOOK_MARKERS, None)

TICK_TIP = (
    "Splits objects that touch, so a marker takes the nucleus it sits on, "
    "not its neighbours. Off: one object per bright blob; at the markers, "
    "two clicks in one blob still split it.")
SPACING_TIP = (
    "How far apart the middles of two nuclei must be for them to count as "
    "two, in voxels of the full-size image across (the same distance in "
    "microns between slices). The default suits most nuclei; much higher "
    "joins close neighbours to the nucleus you click; much lower can let a "
    "click spread through rough, uneven stain.")


# --------------------------------------------------------------------------- #
# the page
# --------------------------------------------------------------------------- #
def _pick_markers(w):
    keys = list(w._add_look_keys)
    w.add_look_for_combo.setCurrentIndex(keys.index(MARKERS))


def _back(w):
    w.add_look_for_combo.setCurrentIndex(0)
    w.update_add_ui_state(box_active=False, whole_patch=False,
                          click_count=0, click_markers=[],
                          click_absorbed_ids=[], candidate_ids=[],
                          candidate_sizes={}, candidate_edge_ids=set())
    w.add_markers_btn.setChecked(False)
    reset_add(w)


def _adv_title(w):
    return str(w._add_advanced_toggle.property("_epicycle_chevron_title"))


def _set_untouched(w, attr, value):
    """Set a control as a preset application would (not the user's pick)."""
    from qtpy.QtWidgets import QCheckBox

    x = getattr(w, attr)
    w._add_applying_preset = True
    try:
        if isinstance(x, QCheckBox):
            x.setChecked(bool(value))
        else:
            x.setValue(value)
    finally:
        w._add_applying_preset = False


def test_the_constants():
    assert AP.ADD_MARKER_SPLIT_DEFAULT in AS.MARKER_SPLIT_MODES
    assert AP.ADD_MARKER_SPLIT_DEFAULT != "off"
    assert AP.ADD_MARKER_SPACING_DEFAULT == AS.ADD_MARKER_SPACING_DEFAULT
    # the markers' summary reads as Anywhere's now: no "split at your
    # markers" any more
    assert not hasattr(AP, "ADD_SUM_ADVANCED_MARKERS")


def test_the_twin_box_is_built_with_the_engine_s_default(widget):
    from qtpy.QtWidgets import QSpinBox

    reset_add(widget)
    sb = widget.add_marker_spacing_sb
    assert isinstance(sb, QSpinBox)
    assert (sb.minimum(), sb.maximum()) == (1, 30)
    assert sb.value() == AP.ADD_MARKER_SPACING_DEFAULT
    # its own row, under Advanced beside Anywhere's, parented
    row = widget._add_marker_spacing_row
    assert row.parent() is widget._add_anywhere_adv
    assert sb.parent() is row
    lay = widget._add_anywhere_adv.layout()
    order = [lay.itemAt(i).widget() for i in range(lay.count())]
    assert order.index(row) == order.index(widget._add_spacing_row) + 1
    # Anywhere's box is untouched
    assert widget.add_min_distance_sb.value() == 3


def test_on_the_markers_the_tick_shows_ticked_and_the_twin_replaces_spacing(
        widget):
    reset_add(widget)
    widget.update_add_ui_state(look_for_items=[], box_active=True)
    _pick_markers(widget)
    try:
        assert not widget._add_anywhere_adv.isHidden()
        assert not widget.add_separate_cb.isHidden()
        assert widget.add_separate_cb.isChecked()
        assert not widget._add_marker_spacing_row.isHidden()
        assert widget._add_spacing_row.isHidden()
        assert widget.add_separate_cb.isEnabled()
        assert widget.add_marker_spacing_sb.isEnabled()
    finally:
        _back(widget)


def test_on_anywhere_and_inside_the_twin_is_hidden(widget):
    reset_add(widget)
    widget.update_add_ui_state(look_for_items=[
        (AP.look_inside_text("cells", "cells"), True,
         (AP.LOOK_INSIDE, "cells"), False)], box_active=True)
    try:
        assert widget.get_add_look_for() == (AP.LOOK_ANYWHERE, None)
        assert not widget.add_separate_cb.isHidden()
        assert not widget._add_spacing_row.isHidden()
        assert widget._add_marker_spacing_row.isHidden()
        widget.add_look_for_combo.setCurrentIndex(1)      # Inside each cell
        assert widget.add_look_inside()
        assert widget._add_anywhere_adv.isHidden()
        assert widget._add_marker_spacing_row.isHidden()
        _pick_markers(widget)
        assert not widget._add_marker_spacing_row.isHidden()
        widget.add_look_for_combo.setCurrentIndex(0)
        assert widget._add_marker_spacing_row.isHidden()
        assert not widget._add_spacing_row.isHidden()
    finally:
        _back(widget)
        widget.update_add_ui_state(look_for_items=[])


def test_the_tooltips(widget):
    from napari_epicycle.annotator.advanced_features_widget import (
        ADD_HELP_LINES,
    )

    reset_add(widget)
    assert ADD_HELP_LINES["add_separate_cb"] == TICK_TIP
    assert ADD_HELP_LINES["add_marker_spacing_sb"] == SPACING_TIP
    assert widget.add_separate_cb.toolTip() == TICK_TIP
    assert widget.add_marker_spacing_sb.toolTip() == SPACING_TIP
    # Anywhere's Spacing keeps its own line
    assert ADD_HELP_LINES["add_min_distance_sb"] == (
        "The closest two object centres may be, in voxels. Lower gives "
        "more, smaller objects.")
    for word in ("watershed", "saddle", "distance map", "seed"):
        assert word not in TICK_TIP.lower() and word not in SPACING_TIP


def test_the_advanced_summary_on_the_markers(widget):
    n = AP.ADD_MARKER_SPACING_DEFAULT
    reset_add(widget)
    widget.update_add_ui_state(look_for_items=[], box_active=True)
    _pick_markers(widget)
    try:
        assert _adv_title(widget).startswith(
            "Advanced: separate touching · spacing {}".format(n))
        # the twin's number, not Anywhere's
        widget.add_marker_spacing_sb.setValue(9)
        assert _adv_title(widget).startswith(
            "Advanced: separate touching · spacing 9")
        # unticked, the markers' Spacing does nothing: not in the summary
        widget.add_separate_cb.setChecked(False)
        assert _adv_title(widget).startswith("Advanced: one per blob")
        assert "spacing" not in _adv_title(widget)
        assert "split at your markers" not in _adv_title(widget)
        # Anywhere's summary keeps its Spacing, ticked or not
        widget.add_look_for_combo.setCurrentIndex(0)
        assert _adv_title(widget).startswith(
            "Advanced: one per blob · spacing 3")
    finally:
        _back(widget)


def test_the_tick_is_shared_with_anywhere(widget):
    """Unticked on Anywhere is unticked on the markers (and back); the
    summary says so in the unticked words; the untick survives a preset
    round trip until a full Add exit, then the next preset ticks it."""
    reset_add(widget)
    widget.update_add_ui_state(look_for_items=[], box_active=True)
    try:
        widget.add_separate_cb.setChecked(False)      # on Anywhere
        _pick_markers(widget)
        assert not widget.add_separate_cb.isHidden()
        assert not widget.add_separate_cb.isChecked()
        assert _adv_title(widget).startswith("Advanced: one per blob · ")
        assert widget.get_add_params()["separate"] is False
        widget.add_separate_cb.setChecked(True)       # on the markers
        widget.add_look_for_combo.setCurrentIndex(0)
        assert widget.add_separate_cb.isChecked()
        assert _adv_title(widget).startswith("Advanced: separate touching")
        widget.add_separate_cb.setChecked(False)
        widget.set_add_preset("cells")
        widget.set_add_preset("nuclei")
        assert not widget.add_separate_cb.isChecked()
        # a full Add exit forgets the touched set; the next preset ticks it
        widget.clear_add_touched(keep_source=True)
        widget.set_add_preset("nuclei", silent=True)
        assert widget.add_separate_cb.isChecked()
    finally:
        _back(widget)


def test_a_preset_application_resets_an_untouched_twin(widget):
    reset_add(widget)
    try:
        _set_untouched(widget, "add_marker_spacing_sb", 12)
        widget.apply_add_preset_values("nuclei")
        assert widget.add_marker_spacing_sb.value() == (
            AP.ADD_MARKER_SPACING_DEFAULT)
        # the user's own number is kept across a preset round trip ...
        widget.add_marker_spacing_sb.setValue(12)
        widget.set_add_preset("cells")
        widget.set_add_preset("nuclei")
        assert widget.add_marker_spacing_sb.value() == 12
        # ... and Anywhere's untouched Spacing still goes back to 3
        _set_untouched(widget, "add_min_distance_sb", 17)
        widget.apply_add_preset_values("nuclei")
        assert widget.add_min_distance_sb.value() == 3
        assert widget.add_marker_spacing_sb.value() == 12
        # a full Add exit, then the next preset: back to the default
        widget.clear_add_touched(keep_source=True)
        widget.set_add_preset("nuclei", silent=True)
        assert widget.add_marker_spacing_sb.value() == (
            AP.ADD_MARKER_SPACING_DEFAULT)
    finally:
        reset_add(widget)


def test_the_snapshot_carries_the_twin_on_the_markers_only(widget):
    reset_add(widget)
    widget.update_add_ui_state(look_for_items=[])
    try:
        assert "marker_spacing" not in widget.get_add_params()
        _pick_markers(widget)
        p = widget.get_add_params()
        assert p["marker_spacing"] == AP.ADD_MARKER_SPACING_DEFAULT
        assert isinstance(p["marker_spacing"], int)
        assert p["separate"] is True
        widget.add_marker_spacing_sb.setValue(9)
        assert widget.get_add_params()["marker_spacing"] == 9
        # Anywhere's number is what it was
        assert widget.get_add_params()["min_distance"] == 3
    finally:
        _back(widget)


def test_a_twin_change_drops_the_preview(qapp):
    from napari_epicycle.annotator.advanced_features_widget import (
        AdvancedFeaturesWidget,
    )

    w = AdvancedFeaturesWidget()
    try:
        hits = []
        w.set_add_callbacks(on_advanced_changed=lambda: hits.append(1))
        w.add_marker_spacing_sb.setValue(8)
        assert hits
    finally:
        w.setParent(None)
        w.deleteLater()


# --------------------------------------------------------------------------- #
# heights: Advanced open on the markers is no taller than on Anywhere
# --------------------------------------------------------------------------- #
@pytest.fixture
def panel(qapp):
    from napari_epicycle.annotator.advanced_features_widget import (
        AdvancedFeaturesWidget,
    )

    w = AdvancedFeaturesWidget(mode="3d_curation")
    yield w
    try:
        w.setParent(None)
    except Exception:
        pass


def _content_h(w, width=312):
    content = w._panel_slots["Add"]["content"]
    lay = content.layout()
    if lay.hasHeightForWidth():
        return lay.totalHeightForWidth(width)
    return content.sizeHint().height()


def _settle(w):
    from qtpy.QtWidgets import QApplication

    w.resize(340, 350)
    for _ in range(3):
        QApplication.processEvents()


def _shown_rows(w):
    lay = w._add_anywhere_adv.layout()
    out = []
    for i in range(lay.count()):
        x = lay.itemAt(i).widget()
        if x is not None and not x.isHidden():
            out.append(x)
    return out


def test_markers_with_advanced_open_are_no_taller_than_anywhere(panel):
    """The markers' tallest Advanced-open state -- the tick on (its Spacing
    row shown) and the marker list up with 4, 8 or 12 markers -- is no
    taller than Anywhere with Advanced open."""
    w = panel
    w.resize(340, 350)
    w.show()
    w.focus_tab("Add")
    w.refresh_add_image_combo(["nuc", "mem"])
    w.set_add_preset("nuclei")
    w.update_add_ui_state(look_for_items=[], box_active=True,
                          voxel_size_known=True, histogram_data={
                              "values": list(np.linspace(0.0, 1.0, 500)),
                              "n_bins": 32})
    w._add_advanced_toggle.setChecked(True)
    _settle(w)
    assert not w._add_advanced_container.isHidden()
    any_rows = _shown_rows(w)
    any_adv = w._add_anywhere_adv.sizeHint().height()
    any_h = _content_h(w)
    _pick_markers(w)
    _settle(w)
    assert w.add_separate_cb.isChecked()
    mk_rows = _shown_rows(w)
    mk_adv = w._add_anywhere_adv.sizeHint().height()
    # the same rows: the tick and ONE Spacing row (the twin for Anywhere's)
    assert len(mk_rows) == len(any_rows)
    assert w.add_separate_cb in mk_rows
    assert w._add_marker_spacing_row in mk_rows
    assert w._add_spacing_row not in mk_rows
    assert mk_adv <= any_adv
    heights = {}
    for n in (0, 4, 8, 12):
        w.update_add_ui_state(click_count=n, click_markers=[
            (65520 + i, (1234, 1234, 1234), i % 3 != 1) for i in range(n)])
        _settle(w)
        assert w.add_click_list.isHidden() is (n == 0)
        heights[n] = _content_h(w)
    print("[ADD-HEIGHT] Advanced open: Anywhere {} px (rows {}), markers "
          "{} px (rows {}) by markers placed".format(any_h, any_adv, heights,
                                                     mk_adv))
    # "Update as I click" is the one Advanced row the markers add (Anywhere
    # has no clicks); without it the markers are no taller than Anywhere,
    # and it costs one checkbox row, pinned exactly (2026-11-22: 814 px with
    # it, 799 without, Advanced open, outside the height ceiling)
    assert not w.add_live_cb.isHidden()
    w.add_live_cb.setVisible(False)
    try:
        _settle(w)
        without = _content_h(w)
    finally:
        w.add_live_cb.setVisible(True)
        _settle(w)
    # Measured pin, not a rule (2026-11-22, the Extend page).  Offscreen there
    # is no real font: the two-line gesture hint under the marker list wraps
    # to four lines (47 px) where the one-line help it replaced took three,
    # so the markers page reads 5 px over Anywhere here (810 vs 805).  With
    # the real Segoe UI font both take two lines and the markers page is
    # 785 px against Anywhere's 775 -- 1 px SHORTER than before the hint
    # (786).  Nothing on the real page grew; the fontless gap is held at 5.
    assert without <= any_h + 5, (without, any_h)
    assert max(heights.values()) - without == LIVE_TICK_ROW_PX


#: the "Update as I click" row with Advanced open, measured offscreen at
#: 340 px (the checkbox and the layout's spacing above it)
LIVE_TICK_ROW_PX = 15


# --------------------------------------------------------------------------- #
# the gather (the real orchestrator method, borrowed by the harness)
# --------------------------------------------------------------------------- #
def _harness(monkeypatch, *, look=MARKERS, spacing=(1.0, 1.0, 1.0),
             factor=None):
    install_fakes(monkeypatch)
    vol = np.zeros((20, 30, 30), np.int32)
    zz, yy, xx = np.indices(vol.shape)
    img = np.zeros(vol.shape, np.float64)
    img[((zz - 10) ** 2 + (yy - 10) ** 2 + (xx - 10) ** 2) <= 16] = 1.0
    hh = Harness(vol, images={"nuc": img}, spacing=spacing)
    if factor is not None:
        hh.ctx["downsample_factor"] = factor
    hh.adv.look = look
    hh.place_box((slice(0, 20), slice(0, 30), slice(0, 30)))
    hh._add_threshold_value = 0.5
    hh._add_click_markers = [{"zyx": (10, 10, 10), "id": 1, "ticked": True}]
    return hh


@pytest.mark.parametrize("spacing,factor,n,want", [
    ((1.0, 1.0, 1.0), None, None, 1.0 * AS.ADD_MARKER_SPACING_DEFAULT),
    ((2.0, 0.5, 0.5), None, 5, 2.5),            # anisotropic
    ((2.0, 0.5, 0.5), 1, 7, 3.5),
    ((4.0, 1.0, 1.0), 2, 5, 2.5),               # half size: 0.5 µm full-size
    ((4.0, 1.0, 1.0), 2, 9, 4.5),
])
def test_the_gather_turns_the_twin_into_microns(monkeypatch, spacing, factor,
                                                n, want):
    hh = _harness(monkeypatch, spacing=spacing, factor=factor)
    if n is not None:
        hh.adv.params["marker_spacing"] = n
    kw, why = hh._add_gather_preview()
    assert why is None
    p = kw["params"]
    assert p["peak_spacing_um"] == pytest.approx(want)
    assert p["separate"] is True
    assert kw["spacing"] == spacing


def test_the_gather_sends_the_untick(monkeypatch):
    hh = _harness(monkeypatch)
    hh.adv.params["separate"] = False
    kw, _ = hh._add_gather_preview()
    assert kw["params"]["separate"] is False


def test_anywhere_s_gather_is_unchanged(monkeypatch):
    hh = _harness(monkeypatch, look=(AP.LOOK_ANYWHERE, None), factor=2)
    hh.adv.params["marker_spacing"] = 9
    kw, why = hh._add_gather_preview()
    assert why is None
    assert set(kw["params"]) == {"min_size", "clear_border", "edge_margin",
                                 "auto_smooth", "smooth_sigma", "clahe",
                                 "clahe_clip", "threshold", "separate",
                                 "min_distance"}
    assert "click_points" not in kw


# --------------------------------------------------------------------------- #
# the preview: the tick's mapping (the engine recorded)
# --------------------------------------------------------------------------- #
def _params(**kw):
    p = {"look_for": AP.LOOK_MARKERS, "threshold": 0.5, "clahe": False,
         "clahe_clip": 0.01, "min_size": 0, "min_distance": 3,
         "click_radius_um": 1.0, "auto_smooth": False,
         "clear_border": False, "edge_margin": 3}
    p.update(kw)
    return p


def _preview(img, points, *, spacing=(1.0, 1.0, 1.0), **kw):
    return AP.run_add_preview(
        preset="nuclei", roi=tuple(slice(0, n) for n in img.shape),
        existing_roi=np.zeros(img.shape, np.int32), next_label=50,
        spacing=spacing, params=_params(**kw), image_roi=img,
        click_points=list(points),
        reserved_ids=[61 + i for i in range(len(points))])


def _recording_engine(monkeypatch, calls):
    """The engine, recorded: every call's keywords are appended to
    ``calls``; the real engine still runs."""
    real = AS.segment_at_markers

    def _rec(*a, **kw):
        calls.append(dict(kw))
        return real(*a, **kw)

    monkeypatch.setattr(AS, "segment_at_markers", _rec)


def _small_img():
    img = np.zeros((10, 10, 10))
    img[3:7, 3:7, 3:7] = 1.0
    return img


@pytest.mark.parametrize("params,want", [
    ({"separate": True, "peak_spacing_um": 2.5}, "default"),
    ({"separate": False, "peak_spacing_um": 2.5}, "off"),
    ({"peak_spacing_um": 2.5}, "off"),     # hand-built, no tick key
    # ticked but no spacing in microns (a hand-built params dict): the
    # engine's own default counts voxels of the call's spacing, which is
    # not the tissue distance at half size -- so the markers run as before
    ({"separate": True}, "off"),
    ({"separate": True, "peak_spacing_um": None}, "off"),
    ({"separate": True, "peak_spacing_um": float("nan")}, "off"),
    ({"separate": True, "peak_spacing_um": 0.0}, "off"),
])
def test_the_tick_reaches_the_engine_as_a_mode(monkeypatch, params, want):
    calls = []
    _recording_engine(monkeypatch, calls)
    res = _preview(_small_img(), [(5, 5, 5)], **params)
    assert res.status == "ok", res.refusal
    assert len(calls) == 1
    if want == "default":
        assert calls[0]["split"] == AP.ADD_MARKER_SPLIT_DEFAULT
        assert calls[0]["peak_spacing_um"] == 2.5
    else:
        assert calls[0]["split"] == "off"


def _assert_split_calls_carry_microns(calls):
    assert calls
    for kw in calls:
        if kw.get("split", "off") != "off":
            um = kw.get("peak_spacing_um")
            assert um is not None and np.isfinite(um) and um > 0, kw


def test_the_microns_reach_the_engine(monkeypatch):
    calls = []
    _recording_engine(monkeypatch, calls)
    _preview(_small_img(), [(5, 5, 5)], separate=True, peak_spacing_um=2.5)
    assert calls[0]["peak_spacing_um"] == 2.5
    _assert_split_calls_carry_microns(calls)


@pytest.mark.parametrize("spacing,factor", [
    ((2.0, 0.5, 0.5), None),                 # full size
    ((4.0, 1.0, 1.0), 2),                    # half size
])
def test_a_split_never_reaches_the_engine_without_microns(
        monkeypatch, widget, spacing, factor):
    """The page's own params, the real gather, the real preview: whenever
    the engine is asked to split, it is handed a finite Spacing in microns
    (its own default would count voxels of the session's spacing, twice
    the tissue distance at half size)."""
    reset_add(widget)
    widget.update_add_ui_state(look_for_items=[])
    _pick_markers(widget)
    try:
        page_params = widget.get_add_params()
    finally:
        _back(widget)
    assert page_params["separate"] is True
    calls = []
    _recording_engine(monkeypatch, calls)
    hh = _harness(monkeypatch, spacing=spacing, factor=factor)
    hh.adv.params = dict(page_params, channel="nuc")
    hh._preview_add()
    assert hh._add_busy, hh._add_transient
    from .add_harness import FakeThread

    th = FakeThread.instances[-1]
    th.run()
    th.end()
    assert len(calls) == 1
    assert calls[0]["split"] == AP.ADD_MARKER_SPLIT_DEFAULT
    _assert_split_calls_carry_microns(calls)
    full_xy = spacing[-1] / (factor or 1)
    assert calls[0]["peak_spacing_um"] == pytest.approx(
        AP.ADD_MARKER_SPACING_DEFAULT * full_xy)


# --------------------------------------------------------------------------- #
# the preview, each mode by name, on a small synthetic clump
# --------------------------------------------------------------------------- #
R = 10.0                    # both nuclei, in voxels at (1, 1, 1)
NECK = 0.4 * R              # a clear waist
SHAPE = (28, 28, 52)
C1 = (14, 14, 15)
C2 = (14, 14, int(round(15 + 2 * np.sqrt(R ** 2 - NECK ** 2))))
#: the markers' Spacing in microns, as the page sends it at (1, 1, 1)
UM = float(AS.ADD_MARKER_SPACING_DEFAULT)


def _clump():
    zz, yy, xx = np.indices(SHAPE)
    a = ((zz - C1[0]) ** 2 + (yy - C1[1]) ** 2 + (xx - C1[2]) ** 2) <= R * R
    b = ((zz - C2[0]) ** 2 + (yy - C2[1]) ** 2 + (xx - C2[2]) ** 2) <= R * R
    from scipy import ndimage as ndi

    assert ndi.label(a | b)[1] == 1                # one touching piece
    return (a | b).astype(np.float64), a, b


def _just(obj, nucleus):
    inside = np.count_nonzero(obj & nucleus) / np.count_nonzero(nucleus)
    outside = np.count_nonzero(obj & ~nucleus) / max(1, np.count_nonzero(obj))
    return inside >= 0.8 and outside <= 0.2


def _run_mode(monkeypatch, mode, points):
    img, a, b = _clump()
    if mode == "off":
        res = _preview(img, points, separate=False)
    else:
        monkeypatch.setattr(AP, "ADD_MARKER_SPLIT_DEFAULT", mode)
        res = _preview(img, points, separate=True, peak_spacing_um=UM)
    assert res.status == "ok", res.refusal
    return res, a, b


@pytest.mark.parametrize("mode", AS.MARKER_SPLIT_MODES)
def test_one_click_in_a_clump_per_mode(monkeypatch, mode):
    res, a, b = _run_mode(monkeypatch, mode, [C1])
    assert res.ids == [61]
    obj = res.result == 61
    if mode == "off":
        # as before: the whole clump
        assert obj[a].all() and obj[b].all()
    else:
        assert _just(obj, a)
        assert not _just(obj, a | b)


def test_the_default_mode_gives_one_nucleus_through_the_preview():
    img, a, _b = _clump()
    res = _preview(img, [C1], separate=True, peak_spacing_um=UM)
    assert res.status == "ok", res.refusal
    if AP.ADD_MARKER_SPLIT_DEFAULT == "off":       # pragma: no cover
        pytest.fail("the tick's default mode must split")
    assert _just(res.result == 61, a)


def test_two_clicks_in_one_piece_single_equals_off(monkeypatch):
    pts = [C1, C2]
    off, _a, _b = _run_mode(monkeypatch, "off", pts)
    single, _a, _b = _run_mode(monkeypatch, "single", pts)
    assert np.array_equal(off.result, single.result)
    assert off.ids == single.ids
    assert off.marker_why == single.marker_why


def test_the_harness_markers_preview_runs_with_the_tick(monkeypatch):
    """The orchestrator's whole path: gather, worker, engine -- ticked, the
    engine is asked for the default mode at the twin's microns."""
    seen = {}
    real = AS.segment_at_markers

    def _rec(*a, **kw):
        seen.update(kw)
        return real(*a, **kw)

    monkeypatch.setattr(AS, "segment_at_markers", _rec)
    hh = _harness(monkeypatch, spacing=(2.0, 0.5, 0.5))
    hh.adv.params["marker_spacing"] = 6
    hh.adv.params["click_radius_um"] = 1.5
    hh._preview_add()
    assert hh._add_busy, hh._add_transient
    from .add_harness import FakeThread

    th = FakeThread.instances[-1]
    th.run()
    th.end()
    assert seen["split"] == AP.ADD_MARKER_SPLIT_DEFAULT
    assert seen["peak_spacing_um"] == pytest.approx(3.0)


# --------------------------------------------------------------------------- #
# the tick without a Spacing in microns says so in the console
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("extra", [{}, {"peak_spacing_um": None},
                                   {"peak_spacing_um": float("nan")},
                                   {"peak_spacing_um": 0.0}])
def test_a_ticked_call_without_microns_prints_one_line(capsys, extra):
    res = _preview(_small_img(), [(5, 5, 5)], separate=True, **extra)
    assert res.status == "ok", res.refusal
    lines = [ln for ln in capsys.readouterr().out.splitlines()
             if ln.startswith("[ADD-SPLIT]")]
    assert len(lines) == 1, lines
    assert "off" in lines[0]


@pytest.mark.parametrize("params", [
    {"separate": True, "peak_spacing_um": 2.5},
    {"separate": False},
    {}])
def test_no_line_when_the_tick_has_its_microns_or_is_off(capsys, params):
    _preview(_small_img(), [(5, 5, 5)], **params)
    assert "[ADD-SPLIT]" not in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# the new rule through the preview with an outline, the box edge, kept
# objects and Smallest object (tick on, with the Spacing in microns)
# --------------------------------------------------------------------------- #
def _preview_with(img, points, *, existing=None, outline_roi=None, **kw):
    if existing is None:
        existing = np.zeros(img.shape, np.int32)
    extra = {} if outline_roi is None else {"outline_roi": outline_roi}
    return AP.run_add_preview(
        preset="nuclei", roi=tuple(slice(0, n) for n in img.shape),
        existing_roi=existing, next_label=50, spacing=(1.0, 1.0, 1.0),
        params=_params(**kw), image_roi=img, click_points=list(points),
        reserved_ids=[61 + i for i in range(len(points))], **extra)


def test_the_rule_with_a_tissue_outline():
    img, a, b = _clump()
    outline = np.ones(img.shape, bool)
    outline[:, :, :8] = False                  # cuts the clicked nucleus
    kw = dict(outline=AP.OUTLINE_SHRINK, bridge_um=10.0, outline_roi=outline)
    on = _preview_with(img, [C1], separate=True, peak_spacing_um=UM, **kw)
    off = _preview_with(img, [C1], separate=False, **kw)
    assert on.status == "ok" and off.status == "ok"
    obj = on.result == 61
    assert not obj[~outline].any()
    assert _just(obj, a & outline)
    # unticked: the whole clump, cut to the outline
    assert (off.result == 61)[b].all()


def test_the_rule_at_the_box_edge():
    """A clicked nucleus cut by the box face starts unticked as before (a
    lone object stays ticked, so both nuclei are clicked)."""
    img, a, b = _clump()
    cut = (slice(None), slice(None), slice(8, None))   # A meets the box face
    pts = [(C1[0], C1[1], C1[2] - 8), (C2[0], C2[1], C2[2] - 8)]
    res = _preview_with(img[cut], pts, separate=True, peak_spacing_um=UM,
                        clear_border=True)
    assert res.status == "ok", res.refusal
    assert sorted(res.ids) == [61, 62]
    assert res.edge_ids == {61}
    assert _just(res.result == 61, a[cut])
    assert _just(res.result == 62, b[cut])
    # one click at the edge, ticked: just that nucleus (a lone object
    # stays ticked); unticked: the whole clump
    one = _preview_with(img[cut], pts[:1], separate=True, peak_spacing_um=UM,
                        clear_border=True)
    assert _just(one.result == 61, a[cut])
    off = _preview_with(img[cut], pts[:1], separate=False, clear_border=True)
    assert (off.result == 61)[b[cut]].all()


def test_the_rule_beside_kept_objects():
    img, a, b = _clump()
    existing = np.zeros(img.shape, np.int32)
    existing[:, 20:, :] = 9                    # kept, along the clump's side
    free = existing == 0
    res = _preview_with(img, [C1], existing=existing, separate=True,
                        peak_spacing_um=UM)
    assert res.status == "ok", res.refusal
    obj = res.result == 61
    assert not obj[~free].any()
    assert _just(obj, a & free)
    assert res.absorbed_ids == []
    off = _preview_with(img, [C1], existing=existing, separate=False)
    assert (off.result == 61)[b & free].all()


def test_a_neighbour_mostly_kept_joins_the_click_a_recorded_limit():
    """Recorded, not required: when most of the neighbour is already kept,
    what is left of it is too shallow to stand apart and joins the clicked
    nucleus -- the same with the tick on or off."""
    img, a, b = _clump()
    existing = np.zeros(img.shape, np.int32)
    far = np.zeros(img.shape, bool)
    far[:, :, C2[2]:] = True
    existing[b & far] = 9
    on = _preview_with(img, [C1], existing=existing, separate=True,
                       peak_spacing_um=UM)
    off = _preview_with(img, [C1], existing=existing, separate=False)
    assert np.array_equal(on.result, off.result)
    assert not (on.result == 61)[existing > 0].any()
    assert (on.result == 61)[a].all()


def test_the_rule_with_smallest_object():
    img, a, b = _clump()
    floor = int(1.3 * np.count_nonzero(a))
    # unticked: the clump is well over the floor and comes back whole
    off = _preview_with(img, [C1], separate=False, min_size=floor)
    assert off.status == "ok", off.refusal
    assert (off.result == 61)[b].all()
    # ticked: the clicked nucleus alone is under it -- left out, said why
    on = _preview_with(img, [C1], separate=True, peak_spacing_um=UM,
                       min_size=floor)
    assert on.status == "refused"
    assert on.refusal == AP.ADD_S1_ALL_FILTERED
    assert on.marker_why == {61: "small"}
    # and just under the nucleus it is kept
    ok = _preview_with(img, [C1], separate=True, peak_spacing_um=UM,
                       min_size=int(0.7 * np.count_nonzero(a)))
    assert ok.status == "ok", ok.refusal
    assert _just(ok.result == 61, a)


# --------------------------------------------------------------------------- #
# the markers' Spacing only with the tick on (hide, never grey)
# --------------------------------------------------------------------------- #
def test_the_markers_spacing_hides_while_the_tick_is_off(widget):
    reset_add(widget)
    widget.update_add_ui_state(look_for_items=[], box_active=True)
    _pick_markers(widget)
    try:
        cb = widget.add_separate_cb
        assert not cb.isHidden() and cb.isChecked()
        assert not widget._add_marker_spacing_row.isHidden()
        keys_on = set(widget.get_add_params())
        cb.click()
        assert not cb.isChecked()
        assert widget._add_marker_spacing_row.isHidden()
        assert widget._add_spacing_row.isHidden()          # Anywhere's too
        assert widget.add_marker_spacing_sb.isEnabled()    # hidden, not grey
        assert set(widget.get_add_params()) == keys_on     # same shape
        assert "spacing" not in _adv_title(widget)
        cb.click()
        assert cb.isChecked()
        assert not widget._add_marker_spacing_row.isHidden()
        assert _adv_title(widget).startswith(
            "Advanced: separate touching · spacing ")
    finally:
        _back(widget)


def test_anywhere_s_spacing_row_ignores_the_tick(widget):
    reset_add(widget)
    widget.update_add_ui_state(look_for_items=[], box_active=True)
    try:
        cb = widget.add_separate_cb
        cb.click()
        assert not cb.isChecked()
        assert not widget._add_spacing_row.isHidden()
        assert widget._add_marker_spacing_row.isHidden()
        assert _adv_title(widget).startswith(
            "Advanced: one per blob · spacing 3")
        cb.click()
        assert not widget._add_spacing_row.isHidden()
    finally:
        _back(widget)


# --------------------------------------------------------------------------- #
# the gather at an unknown spacing, and through the controller's context
# --------------------------------------------------------------------------- #
def test_the_gather_at_an_unknown_spacing_at_half_size(monkeypatch):
    """A never-set voxel size is not doubled at half size: (1, 1, 1) with
    factor 2 -- the Spacing still counts full-size voxels."""
    hh = _harness(monkeypatch, spacing=(1.0, 1.0, 1.0), factor=2)
    kw, why = hh._add_gather_preview()
    assert why is None
    assert kw["params"]["peak_spacing_um"] == pytest.approx(
        AP.ADD_MARKER_SPACING_DEFAULT / 2.0)


def _controller_context(own_zyx, factor):
    """The Step 3 controller's own context provider, on a controller built
    without a viewer (only the attributes the provider reads)."""
    import types

    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController as K,
    )

    c = object.__new__(K)
    c.ui = types.SimpleNamespace()
    c.viewer = None
    c.working_dir = ""
    c._qc_view_patch_name = "p01.tif"
    c._qc_settings = {"spacing": None, "spacing_values": (1.0, 1.0, 1.0)}
    c._qc_spacing_entered = False
    c._run_epoch = 3
    c._current_downsample_factor = lambda *_a, **_k: factor
    c._patch_own_spacing = (lambda *_a, **_k: None if own_zyx is None
                            else {"zyx": own_zyx})
    return c._on_annotator_reshape_context


@pytest.mark.parametrize("own,factor,want", [
    ((2.0, 0.5, 0.5), 1, 5 * 0.5),
    ((2.0, 0.5, 0.5), 2, 5 * 0.5),     # half size: the same tissue distance
    (None, 1, 5 * 1.0),                # never set: voxels
    (None, 2, 5 * 1.0 / 2),            # never set, half size: not doubled
])
def test_the_gather_through_the_controller_s_context(monkeypatch, own,
                                                     factor, want):
    import types

    from napari_epicycle.annotator.curation_3d_widget import (
        Curation3DAnnotator as C,
    )

    hh = _harness(monkeypatch)
    # the real context step and the real spacing reader, fed by the
    # controller's provider, instead of the harness's stubs
    hh._reshape_context_cb = _controller_context(own, factor)
    hh._reshape_context = types.MethodType(C._reshape_context, hh)
    hh._refine_spacing = types.MethodType(C._refine_spacing, hh)
    ctx = hh._reshape_context()
    assert ctx["downsample_factor"] == factor
    kw, why = hh._add_gather_preview()
    assert why is None
    assert kw["params"]["peak_spacing_um"] == pytest.approx(want)
    assert kw["spacing"] == tuple(ctx["spacing"])
