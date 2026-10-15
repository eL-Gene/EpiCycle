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
    "preset": "nuclei", "click_radius_um": 2.0, "core_erode_um": 1.0,
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


def test_connectivity_and_marker_are_gone(widget):
    p = widget.get_add_params()
    assert "connectivity" not in p and "marker" not in p
    assert not hasattr(widget, "add_connectivity_combo")
    assert not hasattr(widget, "add_marker_combo")
