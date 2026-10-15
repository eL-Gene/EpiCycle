"""The B2 live-shot fix pass, functional items (F2-F9).

A themed live run of the rebuilt Add page (driver
``EpiCycle-drivers/shots_step3_add_b2.py``, 2026-09-27) found these; each
test failed on ``f4fc6c24``.

* F2 the Markers "Threshold" box read 0.000 after Auto landed while Preview
  used 84.605 -- the landing stored the value and never showed it;
* F3 with the dark tick off, ``ADD_HELP_DARK`` still showed under the grey
  rows beside ``ADD_DARK_OFF`` (not shown);
* F4 the amber ``ADD_AMBER_MEMBRANE_ONLY`` on a single-channel patch with no
  other run, through the real channel refresh (a layer removed at run time,
  and at launch);
* F5 REVIEW's S1 had no period between its two sentences;
* F6 closing through napari's X detached the box overlay from a canvas
  already deleted (vispy logged the RuntimeError as ERROR);
* F7 a DISABLED dark tick drew as unticked although it was checked;
* F9 a blank nuclei channel left "Measuring the box for Auto..." up for
  ever; and a change that is not an input of the box info (a click, a typed
  threshold, Auto, Discard, "Preview again") threw the info away without
  asking again -- on Cells, "Preview again" was refused with "Measuring the
  box..." and nothing ever measured it.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("qtpy")

from packages.epicycle_annotator import add_preview as AP  # noqa: E402

from .add_harness import FakeThread, Harness, install_fakes  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _items(run_ok=True, nuclei_ok=True):
    return [
        (AP.ADD_ITEM_RUN.format(name="r1") if run_ok else AP.ADD_ITEM_NO_RUN,
         run_ok, ("labels", "r1" if run_ok else None)),
        (AP.ADD_ITEM_NUCLEI if nuclei_ok else AP.ADD_ITEM_NUCLEI_GREY,
         nuclei_ok, ("nuclei_masks", None)),
        (AP.ADD_ITEM_CLICKS, True, ("paint", None)),
    ]


@pytest.fixture
def panel(qapp):
    from napari_epicycle.annotator.advanced_features_widget import (
        AdvancedFeaturesWidget,
    )

    w = AdvancedFeaturesWidget(mode="3d_curation")
    w.resize(340, 350)
    w.show()
    qapp.processEvents()
    w.focus_tab("Add")
    qapp.processEvents()
    w.refresh_add_image_combo(["DAPI", "membrane"])
    w.set_add_preset("cells")
    w.update_add_ui_state(marker_sources=_items(), voxel_size_known=True,
                          box_active=True)
    yield w
    w.hide()
    w.setParent(None)


@pytest.fixture
def cells(monkeypatch):
    """A Cells session on the nuclei-masks source, info not yet landed."""
    install_fakes(monkeypatch)
    from .test_reshape_region import _tissue

    mem, nuc, lab = _tissue()
    hh = Harness(np.zeros_like(lab), images={"mem": mem, "nuc": nuc},
                 spacing=(0.5, 0.5, 0.5))
    hh.adv.preset = "cells"
    hh.adv.source = ("nuclei_masks", None)
    hh.adv.params["dark_outside"] = False
    shown = []
    hh.adv.set_add_nuclei_threshold = shown.append
    hh.shown = shown
    hh.place_box(tuple(slice(0, n) for n in lab.shape))
    return hh


def _land_info(h):
    h._request_add_box_info()
    th = FakeThread.instances[-1]
    th.run()
    th.end()
    return h._add_info


# --------------------------------------------------------------------------- #
# F2 -- the box shows the value Auto landed
# --------------------------------------------------------------------------- #
def test_an_auto_landing_shows_the_value_in_the_box(cells):
    info = _land_info(cells)
    assert info is not None and info.nuclei_auto is not None
    assert cells._add_nuclei_threshold_value == pytest.approx(
        info.nuclei_auto)
    assert cells.shown, "the landing never reached the Threshold box"
    assert cells.shown[-1] == pytest.approx(info.nuclei_auto)


def test_a_typed_value_is_not_overwritten_by_a_later_landing(cells):
    _land_info(cells)
    cells._on_add_nuclei_threshold_typed(123.0)
    n_shown = len(cells.shown)
    # the box moves: the info is asked again and lands again
    b = cells._add_box
    cells._on_add_box_changed(b.cz, b.cy, b.cx, b.sz - 2, b.sy - 2, b.sx - 2)
    _land_info(cells)
    assert cells._add_nuclei_threshold_value == 123.0
    assert len(cells.shown) == n_shown, "a landing overwrote a typed value"


def test_an_untyped_value_follows_the_next_landing(cells):
    _land_info(cells)
    first = cells._add_nuclei_threshold_value
    cells._add_info = None
    cells._add_roi = tuple(slice(0, max(2, s.stop - 4))
                           for s in cells._add_roi)
    info = _land_info(cells)
    assert cells._add_nuclei_threshold_value == pytest.approx(
        info.nuclei_auto)
    assert cells.shown[-1] == pytest.approx(info.nuclei_auto)
    assert first is not None


def test_the_real_box_shows_the_landed_value_to_its_decimals(panel, qapp):
    panel.add_marker_source_combo.setCurrentIndex(1)
    panel.set_add_nuclei_threshold(84.60546875)
    assert panel.add_nuclei_threshold_sb.text().replace(",", ".") \
        .endswith("84.605")


# --------------------------------------------------------------------------- #
# F3 -- the dark help goes with the dark tick
# --------------------------------------------------------------------------- #
def test_the_dark_help_is_hidden_when_the_dark_tick_is_off(panel):
    assert not panel._add_dark_help.isHidden()
    panel.add_dark_outside_cb.setChecked(False)
    assert panel._add_dark_help.isHidden()
    assert panel._add_advanced_help.text() == AP.ADD_DARK_OFF
    panel.add_dark_outside_cb.setChecked(True)
    assert not panel._add_dark_help.isHidden()


def test_the_dark_help_is_hidden_on_nuclei(panel):
    panel.set_add_preset("nuclei")
    assert panel._add_dark_help.isHidden() or \
        panel._add_cells_adv.isHidden()


# --------------------------------------------------------------------------- #
# F4 -- the amber membrane-only line through the REAL refresh
# --------------------------------------------------------------------------- #
@pytest.fixture
def two_channels(monkeypatch, panel):
    install_fakes(monkeypatch)
    from .test_reshape_region import _tissue

    mem, nuc, lab = _tissue()
    hh = Harness(np.zeros_like(lab), images={"mem": mem, "nuc": nuc},
                 spacing=(0.5, 0.5, 0.5))
    hh.adv = panel          # the real page, not the fake
    hh.ctx["workspace"] = ""
    hh.place_box(tuple(slice(0, n) for n in lab.shape))
    panel.update_add_ui_state(box_active=True)
    return hh


def test_the_amber_line_after_a_layer_is_removed(two_channels, panel):
    h = two_channels
    h._refresh_add_image_combo()
    assert panel._add_membrane_only_label.isHidden(), "two channels"
    h.viewer.layers.remove(h.viewer.layers[1])
    h._refresh_add_image_combo()
    assert panel.get_add_marker_source()[0] == "paint"
    assert panel._add_auto_sources_grey()
    assert not panel._add_membrane_only_label.isHidden(), (
        "one channel and no other run: both automatic sources are grey")


def test_the_amber_line_at_launch_on_a_single_channel_patch(
        monkeypatch, panel):
    install_fakes(monkeypatch)
    from .test_reshape_region import _tissue

    mem, _nuc, lab = _tissue()
    h = Harness(np.zeros_like(lab), images={"mem": mem},
                spacing=(0.5, 0.5, 0.5))
    h.adv = panel
    h.place_box(tuple(slice(0, n) for n in lab.shape))
    panel.update_add_ui_state(box_active=True)
    h._refresh_add_image_combo()
    assert panel.get_add_marker_source()[0] == "paint"
    assert not panel._add_membrane_only_label.isHidden()


def test_the_amber_line_when_the_user_had_picked_nuclei(two_channels, panel):
    """The user's own pick of nuclei masks becomes impossible when the
    second channel goes: the source falls back to clicks, and the amber
    line comes with it."""
    h = two_channels
    h._refresh_add_image_combo()
    panel.add_marker_source_combo.setCurrentIndex(1)
    assert panel.get_add_marker_source()[0] == "nuclei_masks"
    h.viewer.layers.remove(h.viewer.layers[1])
    h._refresh_add_image_combo()
    assert panel.get_add_marker_source()[0] == "paint"
    assert not panel._add_membrane_only_label.isHidden()


# --------------------------------------------------------------------------- #
# F5 -- REVIEW's two sentences
# --------------------------------------------------------------------------- #
def test_review_s1_has_a_period_between_its_sentences():
    text = AP.added_line(7, 27027, n_selected=0)
    assert text == ("Added 7 objects · 27,027 voxels. Select one object in "
                    "the list to refine it.")
    assert AP.added_line(2, 5, n_selected=2) == (
        "Added 2 objects · 5 voxels. Select just one object to refine it.")
    assert AP.added_line(1, 9, n_selected=1) == "Added 1 object · 9 voxels"


# --------------------------------------------------------------------------- #
# F6 -- the box overlay on a deleted canvas
# --------------------------------------------------------------------------- #
class _RaisingLine:
    def __init__(self):
        self.sets = 0

    @property
    def parent(self):
        return None

    @parent.setter
    def parent(self, _v):
        self.sets += 1
        raise RuntimeError("wrapped C/C++ object of type "
                           "CanvasBackendDesktop has been deleted")


def test_destroy_on_a_deleted_canvas_does_not_touch_vispy(monkeypatch):
    from packages.epicycle_annotator import box_overlay as BO

    ov = BO.AddBoxOverlay(None, None, None, None)
    line = _RaisingLine()
    ov._lines["MAIN"] = line
    ov._natives["MAIN"] = object()
    monkeypatch.setattr(BO, "_native_deleted", lambda _n: True)
    ov.destroy()
    assert line.sets == 0, "detached a visual from a deleted canvas"
    assert not ov._lines and not ov._natives


def test_destroy_swallows_the_runtime_error_of_a_dying_canvas(monkeypatch):
    from packages.epicycle_annotator import box_overlay as BO

    ov = BO.AddBoxOverlay(None, None, None, None)
    line = _RaisingLine()
    ov._lines["XY"] = line
    ov._natives["XY"] = object()
    monkeypatch.setattr(BO, "_native_deleted", lambda _n: False)
    ov.destroy()                     # must not raise
    assert line.sets == 1
    assert not ov._lines


def test_native_deleted_reads_sip(qapp):
    from qtpy.QtWidgets import QWidget

    from packages.epicycle_annotator import box_overlay as BO

    w = QWidget()
    assert BO._native_deleted(w) is False
    assert BO._native_deleted(None) is False
    w.deleteLater()
    from qtpy.QtCore import QCoreApplication, QEvent

    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert BO._native_deleted(w) is True


# --------------------------------------------------------------------------- #
# F7 -- a disabled tick keeps (and SHOWS) its checked state
# --------------------------------------------------------------------------- #
def _gold_pixels(cb):
    """Gold-family pixels in the indicator's corner of the check box (a
    dimmed gold counts): red well above blue, red above green."""
    img = cb.grab().toImage()
    n = 0
    for y in range(min(img.height(), 24)):
        for x in range(min(img.width(), 20)):
            c = img.pixelColor(x, y)
            if c.red() > 70 and c.red() - c.blue() > 40                     and c.red() >= c.green():
                n += 1
    return n


def test_a_disabled_dark_tick_stays_ticked(panel, qapp):
    panel._add_advanced_toggle.setChecked(True)
    cb = panel.add_dark_outside_cb
    cb.setChecked(True)
    # the only thing left that disables it: a running preview
    panel.update_add_ui_state(busy=True)
    qapp.processEvents()
    assert not cb.isEnabled()
    assert cb.isChecked(), "disabling the tick unticked it"


def test_a_disabled_checked_tick_does_not_draw_as_unchecked(panel, qapp):
    """The theme's ``::indicator:disabled`` rule used to override the
    checked one's gold, so a disabled, checked box drew exactly as an
    unchecked one (live shot B01b)."""
    panel._add_advanced_toggle.setChecked(True)
    cb = panel.add_dark_outside_cb
    # the only thing left that disables it: a running preview
    panel.update_add_ui_state(busy=True)
    qapp.processEvents()
    cb.setChecked(True)
    qapp.processEvents()
    on = _gold_pixels(cb)
    cb.setChecked(False)
    qapp.processEvents()
    off = _gold_pixels(cb)
    cb.setChecked(True)
    assert off == 0, "the measurement sees gold in an unticked box"
    assert on > 20, (
        "a disabled checked tick draws as an empty box ({} gold px)".format(
            on))


# --------------------------------------------------------------------------- #
# F9 -- Auto with a blank nuclei channel; the info is not thrown away
# --------------------------------------------------------------------------- #
def test_auto_help_asks_for_the_channel_when_it_is_blank(panel):
    panel.add_marker_source_combo.setCurrentIndex(1)
    panel._set_add_channel_blank(panel.add_nuclei_channel_combo)
    panel.update_add_ui_state(nuclei_auto=float("nan"))
    assert panel._add_source_help.text() == AP.ADD_AUTO_NEEDS_CHANNEL
    panel.add_nuclei_channel_combo.setCurrentIndex(0)
    panel.update_add_ui_state(nuclei_auto=float("nan"))
    assert panel._add_source_help.text() == AP.ADD_AUTO_PENDING
    panel.update_add_ui_state(nuclei_auto=0.3)
    assert panel._add_source_help.text() == AP.ADD_HELP_NUCLEI_MASKS


def test_the_blank_channel_sentence_is_a_named_constant():
    assert AP.ADD_AUTO_NEEDS_CHANNEL == "Pick the nuclei channel first."


@pytest.mark.parametrize("change", ["typed", "auto", "click", "discard"])
def test_a_change_that_is_not_an_info_input_keeps_the_info(cells, change):
    """The box info depends on the box, the channels, the source and the
    Advanced knobs -- not on a threshold, a click or a Discard.  Throwing
    it away there left nothing to ask again (S1 "Measuring the box..." and
    Preview grey until some unrelated layer event came along)."""
    info = _land_info(cells)
    n_threads = len(FakeThread.instances)
    if change == "typed":
        cells._on_add_nuclei_threshold_typed(50.0)
    elif change == "auto":
        cells._on_add_nuclei_auto()
    elif change == "click":
        cells.adv.source = ("paint", None)
        cells._add_after_click_change()
    else:
        cells._discard_add_candidates()
    assert cells._add_info is info, "{} threw the box info away".format(
        change)
    assert len(FakeThread.instances) == n_threads


def test_preview_again_on_cells_is_not_refused_as_measuring(cells):
    _land_info(cells)
    cells._preview_add()
    assert cells._add_busy, cells._add_transient
    th = FakeThread.instances[-1]
    th.run()
    th.end()
    assert not cells._add_busy
    assert cells._add_preview_result is not None
    cells._preview_add()          # "Preview again"
    assert cells._add_transient != AP.ADD_S1_MEASURING
    assert cells._add_busy


def test_an_info_input_change_still_asks_again(cells):
    _land_info(cells)
    n = len(FakeThread.instances)
    b = cells._add_box
    cells._on_add_box_changed(b.cz, b.cy, b.cx, b.sz - 2, b.sy - 2, b.sx - 2)
    assert cells._add_info is None
    cells._do_add_histogram()          # the debounced request
    assert len(FakeThread.instances) == n + 1


def test_the_info_landing_logs_its_elapsed_time(cells, capsys):
    _land_info(cells)
    out = capsys.readouterr().out
    assert "[ADD-INFO] landed in" in out and " ms" in out


def test_the_default_run_is_not_offered_as_another_run(
        monkeypatch, panel):
    """On the DEFAULT run the controller's context says ``run_id=None``, and
    ``list_guide_sources`` excludes nothing for None: the patch's own run
    came back as "another run" -- an ENABLED automatic source -- so on a
    single-channel patch the amber membrane-only line never showed (the
    live shot B02, on Wound_patch_1, which has only its default run)."""
    from napari_epicycle.annotator import reshape_guide as RG

    class _Src:
        run_id = "default"
        kind = "run"
        which_file = "labels_3d.tif"
        display_name = "default"

    def fake(ws, patch, *, exclude_run_id=None):
        return [] if exclude_run_id == "default" else [_Src()]

    monkeypatch.setattr(RG, "list_guide_sources", fake)
    install_fakes(monkeypatch)
    from .test_reshape_region import _tissue

    mem, _nuc, lab = _tissue()
    h = Harness(np.zeros_like(lab), images={"mem": mem},
                spacing=(0.5, 0.5, 0.5))
    h.ctx.update(workspace="ws", patch_name="P", run_id=None)
    entries = h._add_marker_source_entries()
    assert entries[0] == (AP.ADD_ITEM_NO_RUN, False, ("labels", None))
    h.adv = panel
    h.place_box(tuple(slice(0, n) for n in lab.shape))
    panel.update_add_ui_state(box_active=True)
    h._refresh_add_image_combo()
    assert not panel._add_membrane_only_label.isHidden()
    # a real OTHER run is still offered
    h.ctx["run_id"] = "r2"
    assert h._add_marker_source_entries()[0][1] is True
