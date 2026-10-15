"""The Brightness card, final fix pass before the owner's live test.

B1 an untouched threshold follows the new Auto after EVERY landing; a value
   the user set is kept; Auto / Local contrast / a Brightness channel change
   make it untouched again.
B2 a flat box (bars, no Auto) says so instead of "Measuring..." for ever.
B3 a box / channel / contrast change blanks the bars and greys Auto NOW,
   before the 150 ms debounce.
B4 while the stored threshold is None the box shows pending, never the old
   value-space number.
B5 the histogram and its Auto are measured on a strided sample.
B6 a Brightness channel change resets the threshold.

The orchestrator half runs the real methods on ``add_harness.Harness``; the
panel half on a real ``AdvancedFeaturesWidget`` (offscreen).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from packages.epicycle_annotator import add_preview as AP

from .add_harness import FakeThread, Harness, install_fakes, nuclei_volume
from .add_widget_fixture import qapp, reset_add, widget  # noqa: F401
from .test_add_nuclei_golden import box_of


@pytest.fixture
def h(monkeypatch):
    install_fakes(monkeypatch)
    img, labels = nuclei_volume()
    hh = Harness(labels, images={"nuc": img, "nuc2": img * 3.0 + 50.0})
    hh.place_box(box_of(labels.shape))
    return hh


def _land(h):
    h._request_add_box_info()
    th = FakeThread.instances[-1]
    th.run()
    th.end()
    return h._add_info


def _move_box(h):
    """A real box move through the orchestrator (a smaller, shifted box)."""
    b = h._add_box
    h._on_add_box_changed(b.cz, b.cy + 3, b.cx + 3,
                          b.sz, max(4, b.sy // 2), max(4, b.sx // 2))


def _pushed_after(h, n, key):
    for kw in h.adv.pushes[n:]:
        if key in kw:
            return kw[key]
    return None


# --------------------------------------------------------------------------- #
# B1 -- the untouched rule on every landing
# --------------------------------------------------------------------------- #
def test_an_untouched_threshold_follows_every_landing(h):
    first = _land(h).brightness_auto
    assert h._add_threshold_value == first
    _move_box(h)
    second = _land(h).brightness_auto
    assert second != first, "the fixture must move Auto"
    assert h._add_threshold_value == second
    assert h.adv.last("threshold_value") == second


def test_a_typed_threshold_is_kept_across_a_move(h):
    _land(h)
    h._on_add_threshold_changed(0.4242)
    _move_box(h)
    _land(h)
    assert h._add_threshold_value == 0.4242


def test_auto_makes_it_untouched_again(h):
    _land(h)
    h._on_add_threshold_changed(0.4242)
    h._on_add_otsu_reset()
    assert h._add_threshold_value == h._add_info.brightness_auto
    _move_box(h)
    second = _land(h).brightness_auto
    assert h._add_threshold_value == second


def test_local_contrast_makes_it_untouched_again(h):
    _land(h)
    h._on_add_threshold_changed(0.4242)
    h.adv.params["clahe"] = True
    h._on_add_clahe_toggled(True)
    assert h._add_threshold_value is None
    auto = _land(h).brightness_auto
    assert 0.0 <= auto <= 1.0
    assert h._add_threshold_value == auto


# --------------------------------------------------------------------------- #
# B2 -- a flat box is said, not pending for ever
# --------------------------------------------------------------------------- #
def test_a_flat_box_says_nothing_to_split(monkeypatch):
    install_fakes(monkeypatch)
    _img, labels = nuclei_volume()
    flat = np.full(labels.shape, 7.0)
    hh = Harness(labels, images={"nuc": flat})
    hh.place_box(box_of(labels.shape))
    info = _land(hh)
    assert info.brightness_auto is None
    assert info.histogram is not None
    assert sum(hh.adv.last("histogram_data")["counts"]) > 0   # bars shown
    assert math.isnan(hh.adv.last("brightness_auto"))         # Auto grey
    assert hh.adv.last("brightness_flat") is True


def test_a_real_auto_is_not_flat(h):
    _land(h)
    assert h.adv.last("brightness_flat") is False


# --------------------------------------------------------------------------- #
# B3 -- pending right after the change, before the debounce
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("change", ["box", "channel", "contrast"])
def test_bars_blank_and_auto_grey_at_once(h, change):
    _land(h)
    n = len(h.adv.pushes)
    if change == "box":
        _move_box(h)
    elif change == "channel":
        h.adv.params["channel"] = "nuc2"
        h._on_add_channels_changed()
    else:
        h.adv.params["clahe"] = True
        h._on_add_clahe_toggled(True)
    assert _pushed_after(h, n, "histogram_data") == {"counts": [],
                                                     "edges": []}
    assert math.isnan(_pushed_after(h, n, "brightness_auto"))
    # the landed answer is gone, so Auto has nothing to read
    assert h._add_info is None


# --------------------------------------------------------------------------- #
# B4 / B6 -- the stored None is shown as pending; a channel change resets
# --------------------------------------------------------------------------- #
def test_local_contrast_pushes_the_threshold_pending(h):
    _land(h)
    n = len(h.adv.pushes)
    h._on_add_clahe_toggled(True)
    assert _pushed_after(h, n, "threshold_pending") is True


def test_a_brightness_channel_change_resets_the_threshold(h):
    _land(h)
    h._add_brightness_channel_seen = "nuc"
    h._on_add_threshold_changed(0.4242)
    n = len(h.adv.pushes)
    h.adv.params["channel"] = "nuc2"
    h._on_add_channels_changed()
    assert h._add_threshold_value is None
    assert h._add_threshold_touched is False
    assert _pushed_after(h, n, "threshold_pending") is True
    auto = _land(h).brightness_auto
    assert h._add_threshold_value == auto


def test_another_channel_combo_does_not_reset_the_threshold(h):
    _land(h)
    h._add_brightness_channel_seen = "nuc"
    h._on_add_threshold_changed(0.4242)
    h.adv.nuclei_channel = "nuc2"          # a Cells combo, not Brightness
    h._on_add_channels_changed()
    assert h._add_threshold_value == 0.4242
    assert h._add_threshold_touched is True


# --------------------------------------------------------------------------- #
# the panel half
# --------------------------------------------------------------------------- #
def test_panel_flat_box_sentence(widget):  # noqa: F811
    reset_add(widget)
    widget.update_add_ui_state(
        histogram_data={"counts": [10], "edges": [7.0, 7.5]},
        brightness_auto=float("nan"), brightness_flat=True)
    assert not widget.add_otsu_btn.isEnabled()
    assert widget._add_threshold_help.text() == AP.ADD_AUTO_NOTHING_TO_SPLIT
    widget.update_add_ui_state(brightness_auto=0.4, brightness_flat=False)
    assert widget.add_otsu_btn.isEnabled()
    assert widget._add_threshold_help.text() == AP.ADD_HELP_THRESHOLD


def test_panel_pending_threshold_never_shows_the_old_number(widget):  # noqa: F811
    from napari_epicycle.annotator.advanced_features_widget import (
        ADD_THRESHOLD_PENDING_TEXT,
    )

    reset_add(widget)
    widget.update_add_ui_state(
        histogram_data={"counts": [1, 5, 2],
                        "edges": [0.0, 50.0, 100.0, 150.0]},
        threshold_value=105.469)
    sb = widget.add_threshold_sb
    assert "105" in sb.text()
    widget.update_add_ui_state(threshold_pending=True)
    assert sb.text() == ADD_THRESHOLD_PENDING_TEXT
    # the 0-1 bars land while still pending: still pending, 0-1 range
    widget.update_add_ui_state(
        histogram_data={"counts": [3, 3], "edges": [0.0, 0.5, 1.0]})
    assert sb.text() == ADD_THRESHOLD_PENDING_TEXT
    assert sb.maximum() <= 1.0 + 1e-9
    # leaving the box without typing sets nothing
    fired = []
    widget._on_add_threshold_cb = fired.append
    try:
        widget._on_add_threshold_typed()
        assert fired == []
        widget.update_add_ui_state(threshold_value=0.529)
        assert sb.text() != ADD_THRESHOLD_PENDING_TEXT
        assert abs(sb.value() - 0.529) < 1e-6
    finally:
        widget._on_add_threshold_cb = None


# --------------------------------------------------------------------------- #
# B5 -- the sample
# --------------------------------------------------------------------------- #
def _blobs(shape=(64, 256, 256), n=40, seed=0):
    rng = np.random.default_rng(seed)
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    img = rng.normal(100.0, 10.0, shape)
    for _ in range(n):
        c = [int(rng.integers(0, s)) for s in shape]
        img[((z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2) < 36] += 120
    return img


def test_the_histogram_is_measured_on_a_capped_sample():
    img = _blobs()
    assert img.size > AP.BRIGHTNESS_SAMPLE_VOXELS
    hist, _auto = AP.brightness_histogram(img)
    assert 0 < sum(hist["counts"]) <= AP.BRIGHTNESS_SAMPLE_VOXELS
    full, _ = AP.brightness_histogram(img, sample_voxels=None)
    assert sum(full["counts"]) == img.size


def test_the_sampled_auto_is_within_two_percent_of_the_full_crop():
    img = _blobs()
    _h, sampled = AP.brightness_histogram(img)
    _h, full = AP.brightness_histogram(img, sample_voxels=None)
    assert abs(sampled - full) / abs(full) <= 0.02


def test_the_sampled_local_contrast_auto_is_within_two_percent_of_range():
    img = _blobs()
    _h, sampled = AP.brightness_histogram(img, clahe=True)
    _h, full = AP.brightness_histogram(img, clahe=True, sample_voxels=None)
    # Local contrast lives in 0-1: 2 % of that range
    assert abs(sampled - full) <= 0.02 + 1e-9


def test_a_small_crop_is_not_sampled():
    from skimage.filters import threshold_otsu

    img = _blobs(shape=(20, 40, 40), n=5)
    hist, auto = AP.brightness_histogram(img)
    assert sum(hist["counts"]) == img.size
    assert auto == float(threshold_otsu(img))


def test_the_preview_never_uses_the_sample(monkeypatch):
    """The Nuclei Preview thresholds the FULL crop (the golden): it does not
    go through brightness_histogram at all."""
    def boom(*a, **k):
        raise AssertionError("the Preview used the Brightness sample")

    monkeypatch.setattr(AP, "brightness_histogram", boom)
    from .test_add_nuclei_golden import nuclei_fixture

    img, labels = nuclei_fixture(0)
    roi = box_of(labels.shape)
    res = AP.run_add_preview(
        preset="nuclei", roi=roi, image_roi=img[roi].astype(np.float64),
        existing_roi=labels[roi].copy(),
        params={"threshold": "otsu", "clahe": False, "clahe_clip": 0.01,
                "min_size": 0, "separate": True, "min_distance": 3,
                "clear_border": True, "edge_margin": 3},
        next_label=int(labels.max()) + 1, spacing=(1.0, 1.0, 1.0))
    assert res.status == "ok"
