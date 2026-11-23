"""A channel combo's first fill is BLANK, never the first image layer.

The Refine, Reshape wall and Reshape marker combos used to keep Qt's
auto-selected index 0 on a first fill, so a tool run with nothing chosen
silently used stack channel 0.  Like the Add tab, they now start blank so the
"pick a channel" refusals fire.  A remembered name still restores by name,
survives a dock relaunch inside the same annotator, and a remembered layer
that is gone leaves the combo blank.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

pytest.importorskip("qtpy")

from tests.annotator.test_channel_combo_refresh import qapp, widget  # noqa: F401,E402

COMBOS = (
    ("set_refine_channels", "refine_channel_combo"),
    ("set_wall_channels", "split_wall_channel_combo"),
    ("set_seed_channels", "split_seed_channel_combo"),
)


@pytest.mark.parametrize("setter,attr", COMBOS)
def test_first_fill_is_blank(widget, setter, attr):
    getattr(widget, setter)(["membrane", "dapi"])
    assert getattr(widget, attr).currentIndex() == -1


@pytest.mark.parametrize("setter,attr", COMBOS)
def test_remembered_name_is_selected(widget, setter, attr):
    getattr(widget, setter)(["membrane", "dapi"], "dapi")
    assert getattr(widget, attr).currentText() == "dapi"


@pytest.mark.parametrize("setter,attr", COMBOS)
def test_remembered_name_gone_is_blank(widget, setter, attr):
    getattr(widget, setter)(["membrane", "dapi"], "gone")
    assert getattr(widget, attr).currentIndex() == -1


@pytest.mark.parametrize("setter,attr", COMBOS)
def test_annotator_memory_survives_a_new_dock(widget, setter, attr):
    from napari_epicycle.annotator.curation_3d_widget import (
        Curation3DAnnotator,
    )

    fake = types.SimpleNamespace()
    mem = Curation3DAnnotator._channel_memory
    getattr(widget, setter)(["membrane", "dapi"])
    getattr(widget, attr).setCurrentIndex(1)
    name = mem(fake, widget, attr, "_mem")
    assert name == "dapi"
    # a fresh dock (blank combo) is given the annotator's remembered name
    from napari_epicycle.annotator.advanced_features_widget import (
        AdvancedFeaturesWidget,
    )

    fresh = AdvancedFeaturesWidget(mode="3d_curation")
    try:
        got = mem(fake, fresh, attr, "_mem")
        assert got == "dapi"
        getattr(fresh, setter)(["membrane", "dapi"], got)
        assert getattr(fresh, attr).currentText() == "dapi"
        getattr(fresh, setter)(["membrane"], got)
        assert getattr(fresh, attr).currentIndex() == -1
    finally:
        fresh.setParent(None)


def test_contour_line_names_the_channel(capsys):
    from packages.epicycle_annotator.refine import (
        ContourResult,
        _print_contour_line,
    )

    vol = np.zeros((3, 3, 3), dtype=np.uint16)
    vol[1, 1, 1] = 5
    res = ContourResult(vol.copy(), 0, "x", reason="ok")
    _print_contour_line(res, vol, 5, 0, (1.0, 1.0, 1.0), "test", None,
                        None, {"channel_name": "membrane-ch"}, 0.1)
    assert "channel_name='membrane-ch'" in capsys.readouterr().out
