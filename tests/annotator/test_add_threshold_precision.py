"""The Nuclei threshold keeps its full precision (grill U-H4,
invariant 18; B2.4).

The spin box shows 3 decimals and writes back only on a TYPED edit; a line
drag or Auto sets the stored float and updates the box under blockSignals,
never round-tripping through it.  The Markers-card threshold is a separate
widget with a separate value.
"""

from __future__ import annotations

import numpy as np
import pytest
from skimage.filters import threshold_otsu

from .add_harness import FakeThread, Harness, install_fakes, nuclei_volume
from .add_widget_fixture import qapp, reset_add, widget  # noqa: F401
from .test_add_nuclei_golden import box_of


def test_a_line_value_reaches_the_callback_at_full_precision(widget):
    reset_add(widget)
    got = []
    widget._on_add_threshold_cb = got.append

    class _Line:
        def value(self):
            return 0.123456789

    widget._on_add_line_moved(_Line())
    assert got == [0.123456789]
    assert widget.add_threshold_sb.value() == pytest.approx(0.123, abs=5e-4)


def test_auto_moves_the_box_without_firing(widget):
    reset_add(widget)
    got = []
    widget._on_add_threshold_cb = got.append
    widget.update_add_ui_state(threshold_value=0.7654321)
    assert got == [], "the box must not round-trip Auto's value"
    assert widget.add_threshold_sb.value() == pytest.approx(0.765, abs=5e-4)


def test_a_typed_value_round_trips(widget):
    reset_add(widget)
    got = []
    widget._on_add_threshold_cb = got.append
    widget.add_threshold_sb.setRange(0.0, 1.0)
    widget.add_threshold_sb.setValue(0.25)
    widget._on_add_threshold_typed()
    assert got == [pytest.approx(0.25)]


def test_the_two_threshold_widgets_are_independent(widget):
    reset_add(widget)
    widget.update_add_ui_state(threshold_value=0.4)
    widget.set_add_nuclei_threshold(0.9)
    assert widget.add_threshold_sb.value() == pytest.approx(0.4)
    assert widget.add_nuclei_threshold_sb.value() == pytest.approx(0.9)
    assert widget.add_threshold_sb is not widget.add_nuclei_threshold_sb
    p = widget.get_add_params()
    assert p["nuclei_threshold"] == pytest.approx(0.9)


def test_after_auto_the_engine_gets_otsu_exactly(monkeypatch):
    install_fakes(monkeypatch)
    img, labels = nuclei_volume()
    h = Harness(labels, images={"nuc": img})
    h.place_box(box_of(labels.shape))
    h.adv.params["channel"] = "nuc"
    # B4.2: Auto is measured on the box-info worker; land it, then press
    h._request_add_box_info()
    FakeThread.instances[-1].run()
    FakeThread.instances[-1].end()
    h._add_threshold_value = None
    h._on_add_otsu_reset()
    want = float(threshold_otsu(img[box_of(labels.shape)].astype(np.float64)))
    assert h._add_threshold_value == want
    kw, why = h._add_gather_preview()
    assert why is None
    assert kw["params"]["threshold"] == want      # the float, not 3 dp


def test_the_brightness_auto_is_grey_until_it_lands(widget):
    """B4.2: pending -> Auto grey + ADD_AUTO_PENDING in the card's help
    slot; landed -> live + the threshold help back."""
    from packages.epicycle_annotator import add_preview as AP

    reset_add(widget)
    widget.update_add_ui_state(brightness_auto=float("nan"))
    assert not widget.add_otsu_btn.isEnabled()
    assert widget._add_threshold_help.text() == AP.ADD_AUTO_PENDING
    widget.update_add_ui_state(brightness_auto=0.4)
    assert widget.add_otsu_btn.isEnabled()
    assert widget._add_threshold_help.text() == AP.ADD_HELP_THRESHOLD


def test_worker_bins_draw_and_pending_clears(widget):
    reset_add(widget)
    widget.update_add_ui_state(histogram_data={
        "counts": [1, 5, 2], "edges": [0.0, 0.5, 1.0, 1.5]})
    if widget._add_plot_widget is None:
        pytest.skip("pyqtgraph unavailable")
    assert widget._add_bar_item is not None
    assert widget.add_threshold_sb.maximum() >= 1.5
    widget.update_add_ui_state(histogram_data={"counts": [], "edges": []})
    assert widget._add_bar_item is None
