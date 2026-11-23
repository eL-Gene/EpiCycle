"""``get_add_params()`` on the Nuclei preset at defaults is FROZEN.

The Nuclei golden (``test_add_nuclei_golden``) holds because these values
reach ``add_segment_region`` unchanged; ``dark_outside`` rides along
and is not a Nuclei knob.
"""

from __future__ import annotations

from .add_widget_fixture import qapp, reset_add, widget  # noqa: F401
from .test_add_nuclei_golden import DEFAULTS

NUCLEI_AT_DEFAULTS = {
    "channel": "nuc",
    "clahe": False, "clahe_clip": 0.01, "min_size": 0, "separate": True,
    "min_distance": 3, "clear_border": True, "edge_margin": 3,
    "auto_smooth": True, "smooth_sigma": 1.5,
    "preset": "nuclei", "click_radius_um": 1.0, "core_erode_um": 1.0,
    "dark_clearance_um": 2.0, "dark_otsu_factor": 0.5,
    "nuclei_threshold": 0.0, "dark_outside": True,
}


def test_nuclei_params_at_defaults_are_frozen(widget):
    reset_add(widget)
    assert widget.get_add_params() == NUCLEI_AT_DEFAULTS


def test_the_golden_defaults_are_the_nuclei_goldens_defaults(widget):
    reset_add(widget)
    p = widget.get_add_params()
    for key, val in DEFAULTS.items():
        assert p[key] == val, key


def test_inside_each_cell_adds_its_keys_and_anywhere_comes_back(widget):
    """Look for nuclei ▸ Inside each cell adds four keys (and starts with
    smoothing off); back on Anywhere the snapshot is the frozen one."""
    from packages.epicycle_annotator import add_preview as AP

    reset_add(widget)
    widget.update_add_ui_state(look_for_items=[
        (AP.look_inside_text("cells", "cells"), True,
         (AP.LOOK_INSIDE, "cells"), False)])
    widget.add_look_for_combo.setCurrentIndex(1)
    try:
        p = widget.get_add_params()
        want = dict(NUCLEI_AT_DEFAULTS, auto_smooth=False,
                    look_for=AP.LOOK_INSIDE, nuc_radius_um=2.5,
                    nuc_edge=0.25, two_gap=0.70)
        assert p == want
    finally:
        widget.add_look_for_combo.setCurrentIndex(0)
    assert widget.get_add_params() == NUCLEI_AT_DEFAULTS
    widget.update_add_ui_state(look_for_items=[])


def test_connectivity_and_marker_are_gone(widget):
    p = widget.get_add_params()
    assert "connectivity" not in p and "marker" not in p
    assert not hasattr(widget, "add_connectivity_combo")
    assert not hasattr(widget, "add_marker_combo")


def test_the_markers_add_only_look_for_and_their_spacing(widget):
    """Look for nuclei ▸ At the markers I place adds exactly two keys,
    ``look_for`` and the markers' own Spacing (``marker_spacing``, which
    round-trips); back on Anywhere the snapshot is the frozen one."""
    from packages.epicycle_annotator import add_preview as AP

    reset_add(widget)
    widget.update_add_ui_state(look_for_items=[])
    c = widget.add_look_for_combo
    c.setCurrentIndex(c.count() - 1)
    try:
        assert widget.get_add_look_for() == (AP.LOOK_MARKERS, None)
        assert widget.get_add_params() == dict(
            NUCLEI_AT_DEFAULTS, look_for=AP.LOOK_MARKERS,
            marker_spacing=AP.ADD_MARKER_SPACING_DEFAULT)
        widget.add_marker_spacing_sb.setValue(9)
        assert widget.get_add_params()["marker_spacing"] == 9
    finally:
        c.setCurrentIndex(0)
        reset_add(widget)
    assert widget.get_add_params() == NUCLEI_AT_DEFAULTS
