"""The Workbench plot's palette.

The three Workbench style builders are functions of a ``PlotPalette``; the
default palette reproduces Step 4's shipped strings byte for byte (the
Workbench golden holds them), no raw green hides in a builder, and an
amber palette reaches the built styles and nothing else.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import pytest

from tests.fusion_reconstruction import _qc_plot_snap as S

GOLDEN = Path(__file__).with_name("golden") / "qc_plot_workbench.json"
_RAW_GREEN = re.compile(r"6abf69|106\s*,\s*191\s*,\s*105", re.IGNORECASE)


def _golden_styles() -> dict:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))["styles"]


def test_default_builders_are_byte_identical_to_the_golden():
    from napari_epicycle.fusion_reconstruction import qc_plot_widget as M

    g = _golden_styles()
    d = M.DEFAULT_PLOT_PALETTE
    assert M._wb_combo_style(d) == g["_WB_COMBO_STYLE"]
    assert M._wb_seg_style(d) == g["_WB_SEG_STYLE"]
    assert M._wb_toggle_style(d) == g["_WB_TOGGLE_STYLE"]
    # and the module constants of the shipped names still hold them
    for name in ("_WB_COMBO_STYLE", "_WB_SEG_STYLE", "_WB_TOGGLE_STYLE",
                 "_WB_CLEAR_STYLE", "_WB_LABEL_STYLE", "_WB_BG"):
        assert getattr(M, name) == g[name], name


def test_the_default_palette_is_steps_4_green():
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
        DEFAULT_PLOT_PALETTE,
        PlotPalette,
    )

    assert DEFAULT_PLOT_PALETTE == PlotPalette("#6abf69", (106, 191, 105),
                                               "#6abf69")
    with pytest.raises(Exception):
        DEFAULT_PLOT_PALETTE.accent = "#000000"   # frozen


def test_no_raw_green_in_the_workbench_builders():
    from napari_epicycle.fusion_reconstruction import qc_plot_widget as M

    for fn in (M._wb_combo_style, M._wb_seg_style, M._wb_toggle_style,
               M._accent, M.as_plot_palette):
        src = inspect.getsource(fn)
        assert not _RAW_GREEN.search(src), (fn.__name__, _RAW_GREEN.findall(src))
    # the only green literal in the module's palette section is the default
    src = inspect.getsource(M)
    hits = [ln for ln in src.splitlines()
            if "PlotPalette(" in ln and _RAW_GREEN.search(ln)]
    assert hits == [
        'DEFAULT_PLOT_PALETTE = PlotPalette("#6abf69", (106, 191, 105), "#6abf69")']


def test_theme_builds_the_amber_mapping():
    from napari_epicycle.ui_shell import theme

    p = theme.plot_palette("#d4a843")
    assert p == {"accent": "#d4a843", "accent_rgb": (212, 168, 67),
                 "ok_text": "#d4a843"}
    assert theme.STEP3_PLOT_PALETTE["accent"] == theme.PANEL_GOLD
    assert theme.STEP3_PLOT_PALETTE["accent_rgb"] == (212, 168, 67)
    # an ok line is neutral, never amber text
    assert theme.STEP3_PLOT_PALETTE["ok_text"] != theme.PANEL_GOLD
    with pytest.raises(ValueError):
        theme.plot_palette("d4a843")
    with pytest.raises(ValueError):
        theme.plot_palette("#zzzzzz")


def test_mapping_and_dataclass_are_the_same_palette():
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
        PlotPalette,
        as_plot_palette,
    )
    from napari_epicycle.ui_shell.theme import STEP3_PLOT_PALETTE

    p = as_plot_palette(STEP3_PLOT_PALETTE)
    assert p == PlotPalette("#d4a843", (212, 168, 67), "#bbbbbb")
    assert as_plot_palette(p) is p
    with pytest.raises(TypeError):
        as_plot_palette("#d4a843")
    with pytest.raises(ValueError):
        as_plot_palette({"accent": "#000000", "accent_rgb": (1, 2)})


def test_amber_reaches_the_built_styles_and_nothing_else():
    S.app()
    from napari_epicycle.fusion_reconstruction import qc_plot_widget as M
    from napari_epicycle.fusion_reconstruction._qc_plot_text import (
        STEP3_PLOT_TEXTS,
    )
    from napari_epicycle.ui_shell.theme import STEP3_PLOT_PALETTE

    pw = M.QCPlotWidget(layout="workbench", texts=STEP3_PLOT_TEXTS,
                        palette=STEP3_PLOT_PALETTE)
    try:
        amber = "rgba(212,168,67,"
        assert amber in pw._wb_type_buttons["Scatter"].styleSheet()
        assert amber in pw._select_btn.styleSheet()
        assert amber in pw._history_cb.styleSheet()
        assert amber in pw._color_combo.styleSheet()
        assert amber in pw.plot_type_combo.styleSheet()
        # the ticked-but-greyed rule exists for Step 3 only
        assert ":checked:disabled" in pw._history_cb.styleSheet()
        assert ":checked:disabled" not in M._WB_TOGGLE_STYLE
        # no green left anywhere in the built tree
        tree = json.dumps(S.tree(pw))
        assert not _RAW_GREEN.search(tree)
        # the shipped constants did not move
        assert M._WB_TOGGLE_STYLE == _golden_styles()["_WB_TOGGLE_STYLE"]
    finally:
        pw.deleteLater()


def test_classic_ignores_the_palette():
    S.app()
    from napari_epicycle.fusion_reconstruction import qc_plot_widget as M
    from napari_epicycle.ui_shell.theme import STEP3_PLOT_PALETTE

    a = M.QCPlotWidget()
    b = M.QCPlotWidget(palette=STEP3_PLOT_PALETTE)
    try:
        assert S.tree(a) == S.tree(b)
        assert b._palette == M.DEFAULT_PLOT_PALETTE
    finally:
        a.deleteLater()
        b.deleteLater()


def test_the_seg_row_greys_a_disabled_segment_for_step3_only():
    from napari_epicycle.fusion_reconstruction import qc_plot_widget as M
    from napari_epicycle.ui_shell.theme import STEP3_PLOT_PALETTE

    amber = M._wb_seg_style(M.as_plot_palette(STEP3_PLOT_PALETTE))
    assert "QPushButton:disabled { color:#666666; }" in amber
    assert ("QPushButton:checked:disabled { background: "
            "rgba(212,168,67,0.10); color:#888888; font-weight:bold; }") in amber
    assert ":disabled" not in M._wb_seg_style(M.DEFAULT_PLOT_PALETTE)
    assert M._wb_seg_style(M.DEFAULT_PLOT_PALETTE) == _golden_styles()[
        "_WB_SEG_STYLE"]


def _pixel(widget, x=3):
    img = widget.grab().toImage()
    c = img.pixelColor(x, img.height() // 2)
    return c.red(), c.green(), c.blue()


def test_a_disabled_checked_segment_renders_dimmed_not_lit():
    S.app()
    from napari_epicycle.fusion_reconstruction import qc_plot_widget as M
    from napari_epicycle.fusion_reconstruction._qc_plot_text import (
        STEP3_PLOT_TEXTS,
    )
    from napari_epicycle.ui_shell.theme import STEP3_PLOT_PALETTE

    pw = M.QCPlotWidget(layout="workbench", texts=STEP3_PLOT_TEXTS,
                        palette=STEP3_PLOT_PALETTE)
    try:
        pw.resize(562, 600)
        lit = pw._wb_type_buttons["Scatter"]
        assert lit.isChecked()
        lit_px = _pixel(lit)
        idle = pw._wb_type_buttons["PCA"]
        idle.setEnabled(False)
        idle_px = _pixel(idle)
        lit.setEnabled(False)
        dim_px = _pixel(lit)
        assert dim_px != lit_px and dim_px != idle_px
        # still a hint of amber (red over blue), but less than when lit
        amberness = [p[0] - p[2] for p in (idle_px, dim_px, lit_px)]
        assert amberness[0] < amberness[1] < amberness[2], amberness
    finally:
        pw.deleteLater()
