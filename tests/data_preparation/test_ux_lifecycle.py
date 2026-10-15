"""④ UX-polish lifecycle holes the two audits flagged: the slim scan bar must
never get stuck, the double-click highlight must not linger, and the skip-rebuild
must never leave a shown band contradicting `selected_z_range` (Trim guard).

Qt-free: exercised via the unbound methods on ``SimpleNamespace`` stand-ins.
"""

import types

from napari_epicycle.data_preparation.controller import DataPreparationController


class _FakeBar:
    def __init__(self):
        self.range = None
        self.value = None

    def setRange(self, lo, hi):
        self.range = (lo, hi)

    def setValue(self, v):
        self.value = v


# ---------------------------------------------------------------------------
# The slim ④ scan bar (_set_review_busy) must clear on every terminal
# ---------------------------------------------------------------------------

def _busy_ctl():
    bar = _FakeBar()
    ctl = types.SimpleNamespace(
        widget=types.SimpleNamespace(export_progress_bar=bar))
    return ctl, bar


def test_set_review_busy_on_off():
    ctl, bar = _busy_ctl()
    DataPreparationController._set_review_busy(ctl, True)
    assert bar.range == (0, 0)          # native marquee — same cue as the ② bar
    DataPreparationController._set_review_busy(ctl, False)
    assert bar.range == (0, 1) and bar.value == 0


def test_finish_trim_unlock_clears_the_scan_bar():
    # Trim error routes through _finish_trim_unlock — it MUST clear the ④ bar
    # (the success path used to be the only one that did).
    bar = _FakeBar()
    bar.setRange(0, 0)                  # was scanning
    ctl = types.SimpleNamespace(
        widget=types.SimpleNamespace(
            export_progress_bar=bar,
            _busy_hide=lambda: None),
        _trim_running=True)
    ctl._set_step1_busy = lambda on: None
    ctl._set_review_busy = types.MethodType(
        DataPreparationController._set_review_busy, ctl)
    DataPreparationController._finish_trim_unlock(ctl)
    assert bar.range == (0, 1)         # cleared


# ---------------------------------------------------------------------------
# _clear_highlight
# ---------------------------------------------------------------------------

def test_clear_highlight_drops_and_rebuilds():
    calls = {"rebuild": 0, "refresh": 0}
    ctl = types.SimpleNamespace(_highlighted_patch="p1")
    ctl._rebuild_loc_boxes = lambda: calls.__setitem__("rebuild", 1)
    ctl._refresh_library_list = lambda: calls.__setitem__("refresh", 1)
    DataPreparationController._clear_highlight(ctl)
    assert ctl._highlighted_patch is None
    assert calls == {"rebuild": 1, "refresh": 1}


def test_clear_highlight_noop_when_none():
    ctl = types.SimpleNamespace(_highlighted_patch=None)
    ctl._rebuild_loc_boxes = lambda: (_ for _ in ()).throw(AssertionError("rebuilt"))
    ctl._refresh_library_list = lambda: None
    DataPreparationController._clear_highlight(ctl)   # must not rebuild
    assert ctl._highlighted_patch is None


# ---------------------------------------------------------------------------
# Skip-rebuild re-adopts the shown band so selected_z_range can't diverge
# ---------------------------------------------------------------------------

def _sel_ctl(cur_name, plotted, band_region):
    item = types.SimpleNamespace(text=lambda: cur_name)
    region = types.SimpleNamespace(getRegion=lambda: band_region)
    ctl = types.SimpleNamespace(
        widget=types.SimpleNamespace(
            extracted_roi_list=types.SimpleNamespace(currentItem=lambda: item)),
        _highlighted_patch=None,
        _plot_n_z_patch=plotted,
        selected_z_range=None,
        _zrange_region=region,
    )
    ctl._refresh_conditioning_status = lambda: None
    ctl._plot_area_has_widget = lambda: True
    ctl._plot_calls = []
    ctl._on_plot_intensity_clicked = lambda: ctl._plot_calls.append(1)
    return ctl


def test_skip_readopts_band_when_zrange_went_none():
    # A prior deferred select left selected_z_range=None while A's band shows
    # (2,5).  Re-selecting A skips the rebuild but MUST re-adopt (2,5), so Trim
    # never falsely reports "no range" while a band is drawn.
    ctl = _sel_ctl("A", plotted="A", band_region=(2.0, 5.0))
    DataPreparationController._on_roi_selection_changed(ctl)
    assert ctl._plot_calls == []                 # rebuild skipped
    assert ctl.selected_z_range == (2, 5)        # re-adopted the shown band
