"""The two presets.

What a switch sets (only the keys the user has not touched), that it never
resizes the box, the Cells-only 4 M limit (S/M/L and the start box clamp,
Nuclei L does not), the default-spacing note that replaced the voxel-size
refusal, the dark tick that is
NOT a preset value, and the per-workspace memory of the last preset.
"""

from __future__ import annotations

import numpy as np
import pytest

from packages.epicycle_annotator import add_preview as AP

from .add_harness import Harness, install_fakes
from .add_widget_fixture import qapp, reset_add, widget  # noqa: F401

NUCLEI = {"add_clahe_cb": False, "add_clahe_clip_sb": 0.01,
          "add_separate_cb": True, "add_min_distance_sb": 3,
          "add_clear_border_cb": True, "add_edge_margin_sb": 3,
          "add_min_size_sb": 0, "add_auto_smooth_cb": True,
          "add_smooth_sigma_sb": 1.5}
CELLS = {"add_wall_kind_combo": 0, "add_prep_zcorrect_cb": True,
         "add_prep_bgsub_cb": False, "add_prep_lift_cb": False,
         "add_prep_denoise_cb": False, "add_click_radius_sb": 2.0,
         "add_core_erode_sb": 1.0, "add_dark_clearance_sb": 2.0,
         "add_dark_factor_sb": 0.5, "add_clear_border_cb": True,
         "add_edge_margin_sb": 3, "add_min_size_sb": 0,
         "add_auto_smooth_cb": True, "add_smooth_sigma_sb": 1.5}


def _value(w, attr):
    from qtpy.QtWidgets import QCheckBox, QComboBox

    x = getattr(w, attr)
    if isinstance(x, QCheckBox):
        return x.isChecked()
    if isinstance(x, QComboBox):
        return x.currentIndex()
    return x.value()


# --------------------------------------------------------------------------- #
# the widget
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("preset,table", [("nuclei", NUCLEI), ("cells", CELLS)])
def test_a_switch_sets_the_table(widget, preset, table):
    reset_add(widget)
    for attr in table:                      # dirty every key, untouched
        x = getattr(widget, attr)
        widget._add_applying_preset = True
        try:
            if hasattr(x, "setChecked"):
                x.setChecked(not table[attr])
            elif hasattr(x, "setCurrentIndex"):
                x.setCurrentIndex(1)
            else:
                x.setValue(x.maximum())
        finally:
            widget._add_applying_preset = False
    widget.set_add_preset(preset)
    for attr, val in table.items():
        got = _value(widget, attr)
        assert got == pytest.approx(val), attr


def test_only_untouched_keys_are_set(widget):
    reset_add(widget)
    widget.add_min_size_sb.setValue(50)      # the user touches it
    widget.set_add_preset("cells")
    widget.set_add_preset("nuclei")
    assert widget.add_min_size_sb.value() == 50
    widget.clear_add_touched()
    widget.set_add_preset("cells")
    assert widget.add_min_size_sb.value() == 0


def test_the_preset_combo_keeps_its_name_and_reads_by_index(widget):
    reset_add(widget)
    assert [widget.add_preset_combo.itemText(i)
            for i in range(widget.add_preset_combo.count())] == list(
                AP.ADD_PRESET_LABELS)
    widget.add_preset_combo.setCurrentIndex(1)
    assert widget.get_add_preset() == "cells"
    assert widget.get_add_params()["preset"] == "cells"


def test_the_dark_tick_is_not_a_preset_value(widget):
    reset_add(widget)
    widget.add_dark_outside_cb.setChecked(False)
    widget.set_add_preset("cells")
    widget.set_add_preset("nuclei")
    widget.set_add_preset("cells")
    assert widget.add_dark_outside_cb.isChecked() is False
    assert widget.get_add_params()["dark_outside"] is False
    widget.add_dark_outside_cb.setChecked(True)


def test_dark_off_greys_the_two_dark_rows_with_its_sentence(widget):
    reset_add(widget)
    widget.set_add_preset("cells")
    widget.update_add_ui_state(voxel_size_known=True)
    widget.add_dark_outside_cb.setChecked(False)
    assert not widget.add_dark_clearance_sb.isEnabled()
    assert not widget.add_dark_factor_sb.isEnabled()
    assert widget._add_advanced_help.text() == AP.ADD_DARK_OFF
    widget.add_dark_outside_cb.setChecked(True)
    assert widget.add_dark_clearance_sb.isEnabled()


def test_the_micron_rows_stay_live_at_the_default_spacing(widget):
    """The spacing is used as it is -- no micron
    row is greyed for it, whatever the old ``voxel_size_known`` says."""
    reset_add(widget)
    widget.set_add_preset("cells")
    widget.add_dark_outside_cb.setChecked(True)
    widget.update_add_ui_state(voxel_size_known=False, default_spacing=True)
    for attr in ("add_click_radius_sb", "add_core_erode_sb",
                 "add_dark_clearance_sb", "add_dark_factor_sb",
                 "add_dark_outside_cb"):
        assert getattr(widget, attr).isEnabled(), attr
    assert widget._add_advanced_help.text() != "Voxel size unknown"
    assert not hasattr(AP, "ADD_UM_OFF"), "retired"
    widget.update_add_ui_state(default_spacing=False)


def test_the_default_spacing_note_is_amber_under_s1_on_cells(widget):
    from napari_epicycle.ui_shell.theme import PANEL_WARN_STYLE

    reset_add(widget)
    widget.set_add_preset("cells")
    widget.update_add_ui_state(default_spacing=True, box_active=True)
    note = widget.add_spacing_note
    # with a box the Settings button is locked: the boxed variant (A-M2)
    assert note.text() == AP.ADD_NOTE_DEFAULT_SPACING_BOXED
    assert note.styleSheet() == PANEL_WARN_STYLE
    assert not note.isHidden()
    # under S1: the next widget after the S1 pill in the page layout
    lay = widget._add_s1_container.parentWidget().layout()
    assert lay.indexOf(note) == lay.indexOf(widget._add_s1_container) + 1
    # a real spacing: gone
    widget.update_add_ui_state(default_spacing=False)
    assert note.isHidden()
    # Nuclei works in voxels: gone
    widget.update_add_ui_state(default_spacing=True)
    widget.set_add_preset("nuclei")
    assert note.isHidden()
    # Cells with candidates on screen: gone (the ceiling states, 12.1)
    widget.set_add_preset("cells")
    assert not note.isHidden()
    widget.update_add_ui_state(candidate_ids=[5], candidate_sizes={5: 10},
                               candidate_edge_ids=set())
    assert note.isHidden()
    widget.update_add_ui_state(candidate_ids=[], candidate_sizes={},
                               candidate_edge_ids=set())
    assert not note.isHidden()
    widget.update_add_ui_state(review_active=True)
    assert note.isHidden()
    widget.update_add_ui_state(review_active=False, default_spacing=False,
                               box_active=False)


def test_a_switch_unchecks_place_markers(widget):
    reset_add(widget)
    widget.set_add_preset("cells")
    widget.add_place_markers_btn.setChecked(True)
    widget.set_add_preset("nuclei")
    assert not widget.add_place_markers_btn.isChecked()


def test_the_wall_kinds_are_the_engines_add_kinds():
    from napari_epicycle.annotator.advanced_features_widget import (
        ADD_WALL_KINDS,
    )
    from packages.epicycle_annotator.reshape.spec import (
        ADD_WALL_KINDS as ENGINE,
    )

    assert set(ADD_WALL_KINDS) == set(ENGINE)
    assert ADD_WALL_KINDS[0] == "membrane"


def test_the_default_source_is_nuclei_on_two_channels_clicks_on_one(widget):
    reset_add(widget)
    run = (AP.ADD_ITEM_RUN.format(name="r1"), True, ("labels", "r1"))
    widget.set_add_marker_sources(
        [run, (AP.ADD_ITEM_NUCLEI, True, ("nuclei_masks", None)),
         (AP.ADD_ITEM_CLICKS, True, ("paint", None))])
    widget.add_marker_source_combo.setCurrentIndex(-1)
    widget._select_default_add_source()
    assert widget.get_add_marker_source() == ("nuclei_masks", None)
    widget.set_add_marker_sources(
        [run, (AP.ADD_ITEM_NUCLEI_GREY, False, ("nuclei_masks", None)),
         (AP.ADD_ITEM_CLICKS, True, ("paint", None))])
    widget.add_marker_source_combo.setCurrentIndex(-1)
    widget._select_default_add_source()
    assert widget.get_add_marker_source() == ("paint", None), (
        "a run is listed first but never pre-selected")


def test_the_new_channel_combos_start_blank(qapp):  # noqa: F811
    from napari_epicycle.annotator.advanced_features_widget import (
        AdvancedFeaturesWidget,
    )

    w = AdvancedFeaturesWidget()
    try:
        w.refresh_add_image_combo(["nuc", "mem"])
        assert w.add_nuclei_channel_combo.currentIndex() == -1
        assert w.add_wall_channel_combo.currentIndex() == -1
        assert w.add_nuclei_channel_name() == ""
        w.add_wall_channel_combo.setCurrentIndex(1)
        w.refresh_add_image_combo(["nuc", "mem", "extra"])
        assert w.add_wall_channel_name() == "mem"
        w.refresh_add_image_combo(["nuc"])        # renamed away: blank
        assert w.add_wall_channel_name() == ""
        w.refresh_add_image_combo(["nuc", "x"])   # and it STAYS blank
        assert w.add_wall_channel_name() == ""
    finally:
        w.deleteLater()


# --------------------------------------------------------------------------- #
# the orchestrator: the box limit, the voxel size, the memory
# --------------------------------------------------------------------------- #
@pytest.fixture
def big(monkeypatch):
    install_fakes(monkeypatch)
    vol = np.zeros((300, 400, 400), np.uint8)
    hh = Harness(vol, images={}, spacing=(0.5, 0.5, 0.5))
    hh.place_box((slice(100, 200), slice(100, 300), slice(100, 300)))
    return hh


def _n(box):
    return int(box.sz) * int(box.sy) * int(box.sx)


def test_nuclei_l_is_not_clamped(big):
    big.adv.preset = "nuclei"
    big._set_add_box_size_fraction(0.5)
    assert _n(big._add_box) == 150 * 200 * 200


def test_cells_l_is_clamped_to_the_budget(big):
    big.adv.preset = "cells"
    big._set_add_box_size_fraction(0.5)
    assert 0 < _n(big._add_box) <= AP.ADD_BOX_BUDGET
    assert _n(big._add_box) > 0.9 * AP.ADD_BOX_BUDGET


def test_a_preset_switch_never_resizes_the_box(big):
    big.adv.preset = "nuclei"
    big._set_add_box_size_fraction(0.5)
    box = big._add_box
    big.adv.preset = "cells"
    big._on_add_preset_changed("cells")
    assert big._add_box == box
    state = big._add_s1_state()
    assert AP.preview_block(state).startswith("This box is 6.0 million")
    big.adv.preset = "nuclei"
    big._on_add_preset_changed("nuclei")
    big.adv.params["channel"] = "nuc"
    assert AP.preview_block(big._add_s1_state()) is None


def test_cells_runs_at_the_default_spacing(big):
    """The default
    (1, 1, 1) blocks nothing -- S1 answers exactly as at a real spacing."""
    big._set_add_box_size_fraction(0.125)
    big.adv.preset = "cells"
    big._add_click_markers = [(150, 200, 200)]
    big._add_info = type("I", (), {"run_count": 3, "empty_share": 1.0})()
    big._add_cached_dark = lambda: {"t": 1}
    big.ctx["spacing"] = (0.5, 0.517, 0.517)
    real = AP.preview_block(big._add_s1_state())
    big.ctx["spacing"] = (1.0, 1.0, 1.0)            # the fallback
    state = big._add_s1_state()
    assert AP.preview_block(state) == real
    assert AP.preview_block(state) is None, "Preview is live"
    assert "voxel_known" not in state
    assert not hasattr(AP, "ADD_S1_NO_VOXEL_SIZE"), "retired"
    big.adv.preset = "nuclei"
    big.adv.params["channel"] = "nuc"
    assert AP.preview_block(big._add_s1_state()) is None


@pytest.mark.parametrize("spacing,default", [
    ((1.0, 1.0, 1.0), True), ((1.0, 1.0, 1.0000000001), True),
    ((0.5, 0.517, 0.517), False), ((1.0, 1.0, 0.9), False),
    (None, True), ("junk", True), ((1.0, 1.0), True)])
def test_spacing_is_default(spacing, default):
    assert AP.spacing_is_default(spacing) is default


def test_the_orchestrator_pushes_the_default_spacing_flag(big):
    big.ctx["spacing"] = (1.0, 1.0, 1.0)
    big._add_push_context(sources=False)
    assert big.adv.last("default_spacing") is True
    big.ctx["spacing"] = (0.5, 0.517, 0.517)
    big._add_push_context(sources=False)
    assert big.adv.last("default_spacing") is False


def test_the_start_box_is_clamped_on_cells(big, monkeypatch):
    big.adv.preset = "cells"
    big._add_running = False
    big._add_box = None
    big._add_last_size = (200, 300, 300)
    monkeypatch.setattr(type(big), "_add_restore_preset", lambda self: None)
    big._start_add()
    assert _n(big._add_box) <= AP.ADD_BOX_BUDGET


class _FakeSettings:
    store: dict = {}

    def __init__(self, *a):
        pass

    def value(self, key, default=None):
        return _FakeSettings.store.get(key, default)

    def setValue(self, key, value):             # noqa: N802 - Qt name
        _FakeSettings.store[key] = value


def test_the_last_preset_is_remembered_per_workspace(big, monkeypatch):
    import qtpy.QtCore

    _FakeSettings.store = {}
    monkeypatch.setattr(qtpy.QtCore, "QSettings", _FakeSettings)
    big.ctx["workspace"] = "C:/ws/one"
    big._on_add_preset_changed("cells")
    assert big._add_load_setting("last_preset", "nuclei") == "cells"
    big.ctx["workspace"] = "C:/ws/two"
    assert big._add_load_setting("last_preset", "nuclei") == "nuclei"
    assert any(k.startswith("last_preset/") for k in _FakeSettings.store)


def test_without_a_workspace_the_memory_is_the_session(big, monkeypatch):
    import qtpy.QtCore

    _FakeSettings.store = {}
    monkeypatch.setattr(qtpy.QtCore, "QSettings", _FakeSettings)
    big.ctx["workspace"] = ""
    big._on_add_preset_changed("cells")
    assert _FakeSettings.store == {}
    assert big._add_load_setting("last_preset", "nuclei") == "cells"
