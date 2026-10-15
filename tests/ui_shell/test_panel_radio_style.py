"""PANEL_RADIO_STYLE (Step 2 look, brick L): additive, 11 px gold radio.

Seen live: napari's own ``::indicator::checked`` rule (``image`` + 5 px
padding + 6 px size) beats a state-less rule, so the size, the padding and
``image: none`` must ride the checked state too."""
import re

from napari_epicycle.ui_shell import theme


def _rule(css, selector):
    m = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", css)
    assert m, selector
    return m.group(1)


def test_it_exists_beside_the_checkbox_style():
    assert isinstance(theme.PANEL_RADIO_STYLE, str)
    assert isinstance(theme.PANEL_CHECKBOX_STYLE, str)


def test_the_gold_dot_is_the_only_mark_on_the_checked_state():
    css = theme.PANEL_RADIO_STYLE
    checked = _rule(css, "QRadioButton::indicator:checked")
    assert "image:none" in checked.replace(" ", "")
    assert "#d4a843" in checked
    sized = _rule(css, "QRadioButton::indicator, QRadioButton::indicator:unchecked,"
                       " QRadioButton::indicator:checked, QRadioButton::indicator:disabled,"
                       " QRadioButton::indicator:hover")
    flat = sized.replace(" ", "")
    assert "width:11px" in flat and "height:11px" in flat
    assert "padding:0px" in flat


def test_step_3s_own_radio_constant_is_untouched():
    from napari_epicycle.fusion_reconstruction.ui import MR_RADIO_STYLE

    assert "width:11px;height:11px" in MR_RADIO_STYLE
    assert "image" not in MR_RADIO_STYLE


def test_no_accent_text_colour():
    # the accent is never the colour of a sentence (look rule 1)
    assert re.search(r"color:\s*#d4a843", theme.PANEL_RADIO_STYLE) is None
