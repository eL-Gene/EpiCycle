"""The B2 live-shot fix pass, visual items (V1-V8).

The themed live run (``EpiCycle-drivers/shots_step3_add_b2.py``,
2026-09-27) against the mockup, which wins for what is on screen:

* V1 the shared header's description button clipped its chevron (22 px
  wide, needs 26) -- on Add, Refine and Reshape alike;
* V2 the pinned strip drew only Preview when idle; the mockup always draws
  Stop, the busy bar and the stage line, dimmed;
* V3 when every candidate STARTED unticked (they all touch the box edge),
  S1 read TICK_ONE and the found count never showed -- TICK_ONE is for the
  USER unticking the last row;
* V4 the footer read "Accept" with candidates up; the mockup "Accept (n)";
* V5 REVIEW's "Done" was red (danger); the mockup's is neutral;
* V6 a Cells preview's stage read "Finding bright objects..." while it
  found nuclei-mask markers (spec: "Finding markers...");
* V7 a BLOCKING S1 reason ("Pick the membrane channel under Walls.") sat
  beside a green dot;
* V8 S / M / L were clipped in fixed 26 px buttons ("M" drew as a "V").
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


def _items():
    return [
        (AP.ADD_ITEM_RUN.format(name="r1"), True, ("labels", "r1")),
        (AP.ADD_ITEM_NUCLEI, True, ("nuclei_masks", None)),
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
    w.update_add_ui_state(marker_sources=_items(), voxel_size_known=True)
    yield w
    w.hide()
    w.setParent(None)


# --------------------------------------------------------------------------- #
# V1 -- the header's description button fits its chevron
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("attr", ["add_about_btn", "refine_about_btn",
                                  "split_about_btn"])
def test_the_description_button_is_wide_enough_for_its_chevron(
        panel, qapp, attr):
    btn = getattr(panel, attr)
    qapp.processEvents()
    assert btn.width() >= btn.sizeHint().width(), (
        "{} is {} px, its chevron needs {}".format(
            attr, btn.width(), btn.sizeHint().width()))
    assert btn.width() >= 26


# --------------------------------------------------------------------------- #
# V2 -- the strip draws Stop, the bar and the stage line when idle
# --------------------------------------------------------------------------- #
def test_the_idle_strip_draws_everything_dimmed(panel, qapp):
    panel.update_add_ui_state(busy=False)
    qapp.processEvents()
    assert not panel.add_stop_btn.isHidden()
    assert not panel.add_stop_btn.isEnabled()
    assert not panel._add_busy_container.isHidden()
    assert not panel.add_busy_bar.isEnabled()
    assert panel.add_busy_bar.maximum() > 0, "no marquee while idle"
    assert not panel.add_stage_label.isHidden()
    assert panel.add_stage_label.text() == ""


def test_the_busy_strip_runs_the_marquee(panel, qapp):
    h0 = panel._add_s4_container.height()
    panel.update_add_ui_state(busy=True, stop_enabled=True,
                              stage_text="Finding markers...")
    qapp.processEvents()
    assert panel.add_stop_btn.isEnabled()
    assert panel.add_busy_bar.isEnabled()
    assert panel.add_busy_bar.maximum() == 0, "a marquee while busy"
    assert "Finding markers" in panel.add_stage_label.toolTip()
    panel.update_add_ui_state(busy=False)
    qapp.processEvents()
    assert not panel.add_stop_btn.isEnabled()
    assert not panel.add_stop_btn.isHidden()
    assert panel.add_stage_label.text() == ""
    assert panel._add_s4_container.height() == h0


# --------------------------------------------------------------------------- #
# V3 -- TICK_ONE only when the USER unticks the last row
# --------------------------------------------------------------------------- #
def test_all_started_unticked_shows_the_found_count():
    s = dict(has_box=True, preset="cells",
             candidates=(7, 7, 0, 0, 0), user_emptied=False)
    assert AP.s1_sentence(s) == AP.found_line(7, k_edge=7)
    assert AP.s1_sentence(s) == (
        "7 new objects found · 7 left unticked (at the box edge)")


def test_the_user_unticking_the_last_row_is_tick_one():
    s = dict(has_box=True, preset="cells",
             candidates=(7, 2, 0, 0, 0), user_emptied=True)
    assert AP.s1_sentence(s) == AP.ADD_S1_TICK_ONE


@pytest.fixture
def h(monkeypatch):
    install_fakes(monkeypatch)
    from .add_harness import nuclei_volume
    from .test_add_nuclei_golden import box_of

    img, labels = nuclei_volume()
    hh = Harness(labels, images={"nuc": img})
    hh.place_box(box_of(labels.shape))
    return hh


def _preview(h):
    h._preview_add()
    th = FakeThread.instances[-1]
    th.run()
    th.end()
    assert h._add_preview_res is not None


def test_the_orchestrator_tells_started_unticked_from_user_unticked(h):
    h.adv.checked = []                 # every row starts unticked
    _preview(h)
    res = h._add_preview_res
    assert h.adv.last("status_text") != AP.ADD_S1_TICK_ONE
    assert h.adv.last("status_text").startswith(
        AP.count_phrase(len(res.ids), AP.ADD_S1_FOUND_ONE, AP.ADD_S1_FOUND))
    # the user ticks one, then unticks it
    h.adv.checked = [res.ids[0]]
    h._add_candidate_layer = object()
    h._sync_add_candidate_overlay()
    h.adv.checked = []
    h._sync_add_candidate_overlay()
    assert h.adv.last("status_text") == AP.ADD_S1_TICK_ONE
    # and a new preview starts over
    h._invalidate_add_preview()
    _preview(h)
    assert h.adv.last("status_text") != AP.ADD_S1_TICK_ONE


# --------------------------------------------------------------------------- #
# V4 -- "Accept (n)" with the ticked count
# --------------------------------------------------------------------------- #
def test_accept_carries_the_ticked_count(panel):
    panel.update_add_ui_state(box_active=True)
    assert panel.add_accept_btn.text() == "Accept"
    panel.update_add_ui_state(candidate_edge_ids={2},
                              candidate_ids=[1, 2, 3],
                              candidate_sizes={1: 5, 2: 5, 3: 5})
    assert panel.add_accept_btn.text() == "Accept (2)"
    panel.add_none_btn.click()
    assert panel.add_accept_btn.text() == "Accept (0)"
    assert not panel.add_accept_btn.isEnabled()
    panel.update_add_ui_state(accept_count=0, candidate_ids=[])
    assert panel.add_accept_btn.text() == "Accept"


# --------------------------------------------------------------------------- #
# V5 -- Done is neutral
# --------------------------------------------------------------------------- #
def test_done_is_neutral_and_cancel_goes_back(panel):
    from napari_epicycle.annotator import advanced_features_widget as M

    red = panel.add_reject_btn.styleSheet()
    panel.update_add_ui_state(review_active=True)
    assert panel.add_reject_btn.text() == "Done"
    assert panel.add_reject_btn.styleSheet() == M.PANEL_BTN_STYLE
    assert panel.add_reject_btn.styleSheet() != red
    panel.update_add_ui_state(review_active=False)
    assert panel.add_reject_btn.text() == "Cancel"
    assert panel.add_reject_btn.styleSheet() == red


# --------------------------------------------------------------------------- #
# V6 -- the Cells nuclei-mask stage is "Finding markers..."
# --------------------------------------------------------------------------- #
def test_a_cells_nuclei_preview_reports_finding_markers():
    from napari_epicycle.annotator.reshape_worker import stage_text

    from .test_add_preview import _cells

    names = []
    out, _ = _cells("nuclei_masks", progress=lambda s, i, n: names.append(s))
    assert out.ok
    assert names[0] == "add_markers"
    assert "add_find" not in names
    assert stage_text(names[0]) == "Finding markers..."


# --------------------------------------------------------------------------- #
# V7 -- the dot follows what S1 means
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("state,kind", [
    (dict(has_box=False), "idle"),
    (dict(has_box=True, preset="cells", wall_channel=""), "warn"),
    (dict(has_box=True, preset="cells", wall_channel="m",
          source="nuclei_masks", nuclei_channel="m"), "warn"),
    (dict(has_box=True, preset="nuclei", channel=""), "warn"),
    (dict(has_box=True, preset="cells", wall_channel="m",
          source="nuclei_masks", nuclei_channel="n", info_pending=True),
     "busy"),
    (dict(has_box=True, preset="nuclei", channel="c", empty_share=0.5),
     "ready"),
    (dict(has_box=True, busy=True), "busy"),
    (dict(has_box=True, transient=AP.ADD_S1_STOPPED), "warn"),
    (dict(has_box=True, candidates=(3, 0, 0, 0, 3)), "review"),
    (dict(has_box=True, review=(3, 30, 0)), "review"),
])
def test_s1_kind(state, kind):
    assert AP.s1_kind(state) == kind


def test_a_blocking_reason_has_the_blocking_dot(panel):
    from napari_epicycle.annotator import advanced_features_widget as M

    panel.update_add_ui_state(box_active=True,
                              status_text=AP.ADD_S1_PICK_WALL_CHANNEL,
                              status_kind="warn")
    assert panel._add_status_dot.styleSheet() == M._STATUS_DOT_STYLES["warn"]
    panel.update_add_ui_state(status_text="Box ...", status_kind="ready")
    assert panel._add_status_dot.styleSheet() == \
        M._STATUS_DOT_STYLES["ready"]


def test_the_orchestrator_pushes_the_kind(h):
    h.adv.params["channel"] = ""
    h._add_refresh_ui()
    assert h.adv.last("status_text") == AP.ADD_S1_PICK_CHANNEL
    assert h.adv.last("status_kind") == "warn"


# --------------------------------------------------------------------------- #
# V8 -- S / M / L are readable
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", ["s", "m", "l"])
def test_the_size_letters_are_not_clipped(panel, qapp, key):
    panel.update_add_ui_state(box_active=True)
    qapp.processEvents()
    b = getattr(panel, "add_size_{}_btn".format(key))
    assert b.width() >= b.sizeHint().width(), (
        "{} is {} px, needs {}".format(key.upper(), b.width(),
                                      b.sizeHint().width()))


def test_the_box_row_still_fits(panel, qapp):
    panel.update_add_ui_state(box_active=True)
    qapp.processEvents()
    assert panel._add_box_container.minimumSizeHint().width() <= 286
    # the decision that made it fit: "Size" is off the row
    assert panel._add_size_label.isHidden()


# --------------------------------------------------------------------------- #
# V9 -- what came off the page for the 312 px ceiling (steps 2-3)
# --------------------------------------------------------------------------- #
def test_the_take_offs_are_in_place(panel):
    import pathlib

    from napari_epicycle.annotator.advanced_features_widget import (
        REFINE_NO_CLAHE_SENTENCE,
    )

    assert panel._add_prep_no_clahe.isHidden()
    assert not panel.add_wall_prep_btn.isChecked(), "opens closed"
    assert AP.ADD_NOTES_LINES == 1
    guide = (pathlib.Path(__file__).resolve().parents[2] / "docs" / "guides"
             / "annotator_3d_add.md").read_text(encoding="utf-8")
    assert REFINE_NO_CLAHE_SENTENCE.rstrip(".") in " ".join(guide.split())


# --------------------------------------------------------------------------- #
# B3.4 (b) -- the Walls summary says the ROLE, the combo keeps the layer name
# --------------------------------------------------------------------------- #
def test_the_walls_summary_uses_the_role_word_not_the_layer_name(qapp):
    from napari_epicycle.annotator.advanced_features_widget import (
        AdvancedFeaturesWidget,
    )

    w = AdvancedFeaturesWidget(mode="3d_curation")
    try:
        w.focus_tab("Add")
        w.refresh_add_image_combo(["C1-DAPI", "C2-memb-long-name"])
        w.update_add_ui_state(marker_sources=_items(), voxel_size_known=True,
                              box_active=True)
        w.add_wall_channel_combo.setCurrentIndex(1)
        wl = w.add_walls_btn.property("_epicycle_chevron_title")
        assert wl.startswith("Walls: membrane · "), wl
        assert "C2-memb" not in wl
        assert w.add_wall_channel_combo.currentText() == "C2-memb-long-name"
        # no wall channel picked: no role word either
        w.add_wall_channel_combo.setCurrentIndex(-1)
        w._refresh_add_summaries()
        wl = w.add_walls_btn.property("_epicycle_chevron_title")
        assert "membrane ·" not in wl.split(":", 1)[-1][:12], wl
    finally:
        w.setParent(None)


def test_walls_summary_role_constant():
    assert AP.ADD_SUM_WALLS_ROLE == "membrane"
