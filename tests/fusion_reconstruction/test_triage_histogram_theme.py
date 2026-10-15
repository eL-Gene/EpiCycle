"""Step 3 plot pass, brick P0: the triage histogram on the
panel theme -- no navy surface, no dummy data, no chips, plain words."""

from __future__ import annotations

import ast
import pathlib
import re

import pytest


@pytest.fixture(scope="module")
def qapp():
    from qtpy.QtWidgets import QApplication
    import sys
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


_MODULE = (
    pathlib.Path(__file__).resolve().parents[2]
    / "napari_epicycle" / "fusion_reconstruction" / "triage_histogram_panel.py"
)


def _mod():
    from napari_epicycle.fusion_reconstruction import triage_histogram_panel
    return triage_histogram_panel


def _panel():
    return _mod().TriageHistogramPanel()


# ---------------------------------------------------------------------------
# Source-level guards
# ---------------------------------------------------------------------------


def test_no_dummy_anywhere_in_the_module():
    src = _MODULE.read_text(encoding="utf-8")
    assert re.search(r"dummy", src, re.IGNORECASE) is None


def test_no_phase_prose_anywhere_in_the_module():
    src = _MODULE.read_text(encoding="utf-8")
    assert re.search(r"Phase\s*(1\.5|2)\b", src) is None


def test_no_navy_background_in_the_module():
    src = _MODULE.read_text(encoding="utf-8")
    assert "1a1a2e" not in src.lower()


def test_no_chips_left():
    tree = ast.parse(_MODULE.read_text(encoding="utf-8"))
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "_tier_chips" not in names
    assert "tier_pill_style" not in names


def test_no_cyan_gate_band():
    """The gate band takes the selection colour, not cyan (= curated)."""
    src = _MODULE.read_text(encoding="utf-8")
    assert "mkBrush(0, 188, 212" not in src
    assert "(0, 188, 212, 220)" not in src


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_named_constants(qapp):
    m = _mod()
    assert m.HIST_BG == "#23242b"
    assert m.HIST_PLOT_MIN_H == 130
    assert m.GATE_COLOUR == "#59b3ff"
    assert m.GATE_RGB == (0x59, 0xB3, 0xFF)
    assert m.GHOST_GREY == (140, 140, 160)
    assert m._TIER_COLORS_RGB["ghost"] == m.GHOST_GREY
    assert set(m.HIST_HELP) >= {"Triage by", "Colour bars by",
                                "Reset lines", "Gate"}
    for key, line in m.HIST_HELP.items():
        assert line and "Phase" not in line, key


# ---------------------------------------------------------------------------
# The built widget
# ---------------------------------------------------------------------------


def test_combos_on_the_panel_input_style(qapp):
    from napari_epicycle.ui_shell import theme
    panel = _panel()
    assert panel._feature_combo.styleSheet() == theme.PANEL_INPUT_STYLE
    assert panel._color_combo.styleSheet() == theme.PANEL_INPUT_STYLE


def test_reset_button_and_gate_on_panel_styles(qapp):
    from napari_epicycle.ui_shell import theme
    panel = _panel()
    assert panel._reset_btn.text() == "Reset lines"
    assert panel._reset_btn.styleSheet() == theme.PANEL_BTN_STYLE
    assert panel._gate_cb.text() == "Gate"
    assert panel._gate_cb.styleSheet() == theme.PANEL_CHECKBOX_STYLE
    assert ":checked:disabled" in panel._gate_cb.styleSheet()


def test_field_labels_have_no_colons(qapp):
    from qtpy.QtWidgets import QLabel
    from napari_epicycle.ui_shell import theme
    panel = _panel()
    texts = {lab.text(): lab for lab in panel.findChildren(QLabel)}
    assert "Triage by" in texts
    assert "Colour bars by" in texts
    assert "Triage by:" not in texts and "Color by:" not in texts
    for word in ("Triage by", "Colour bars by"):
        assert texts[word].styleSheet() == theme.PANEL_ROW_LABEL_STYLE


def test_tooltips_come_from_the_help_table(qapp):
    m = _mod()
    panel = _panel()
    assert panel._feature_combo.toolTip() == m.HIST_HELP["Triage by"]
    assert panel._color_combo.toolTip() == m.HIST_HELP["Colour bars by"]
    assert panel._reset_btn.toolTip() == m.HIST_HELP["Reset lines"]
    assert panel._gate_cb.toolTip() == m.HIST_HELP["Gate"]


def test_note_line_is_a_neutral_help_line(qapp):
    """Demoted (orchestrator decision): one neutral help line, no amber
    block inside the histogram."""
    from napari_epicycle.ui_shell import theme
    panel = _panel()
    assert panel._note_label.styleSheet() == theme.PANEL_HELP_STYLE
    assert not panel._note_label.wordWrap()


def test_panel_margins_are_zero(qapp):
    panel = _panel()
    m = panel.layout().contentsMargins()
    assert (m.left(), m.top(), m.right(), m.bottom()) == (0, 0, 0, 0)


def test_plot_background_and_floor(qapp):
    m = _mod()
    panel = _panel()
    assert panel._plot_widget.minimumHeight() == m.HIST_PLOT_MIN_H
    bg = panel._plot_widget.backgroundBrush().color().name()
    assert bg == m.HIST_BG


def test_gate_band_in_the_selection_colour(qapp):
    m = _mod()
    panel = _panel()
    region = panel._gate_region
    brush = region.brush.color()
    assert (brush.red(), brush.green(), brush.blue()) == m.GATE_RGB
    assert brush.alpha() < 128, "the band is translucent"
    pen = region.lines[0].pen.color()
    assert (pen.red(), pen.green(), pen.blue()) == m.GATE_RGB


def test_colour_combo_seeded_with_the_default_only(qapp):
    panel = _panel()
    items = [panel._color_combo.itemText(i)
             for i in range(panel._color_combo.count())]
    assert items == ["qc_tier"]
