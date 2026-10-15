"""Quirk B (owner live test 2026-09-23): S4's button read "Preview again"
for ever.

It was set to "Preview again" only by `_finish_refine_preview`, and reset
only by `_reset_smooth_state` and the Stop path -- Accept and Discard never
put it back, nor the S5 texts or the "Accept (n)" wording.  The fix derives
the word from the spec's own "is a result staged" test (1.3: the footer
secondary is live), in the panel, and has Accept / Discard clear S5.

The annotator half DRIVES the real methods on the bare fake of
`test_refine_cross_method_chain.py` and forwards every push to a REAL
`AdvancedFeaturesWidget`, so what is asserted is the button a user sees.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qtpy")

from tests.annotator.test_refine_cross_method_chain import (  # noqa: E402
    Fake, _Layer, _vol,
)


@pytest.fixture(scope="module")
def qapp():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def panel(qapp):
    from napari_epicycle.annotator.advanced_features_widget import (
        AdvancedFeaturesWidget,
    )

    w = AdvancedFeaturesWidget(mode="3d_curation")
    yield w
    try:
        w.setParent(None)
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _labels(monkeypatch, qapp):
    import napari_epicycle.annotator.curation_3d_widget as mod

    monkeypatch.setattr(mod, "Labels", _Layer)


class _PanelFake(Fake):
    def __init__(self, vol, panel):
        super().__init__(vol)
        self.panel = panel

    def _update_smooth_ui(self, **kw):
        self.ui_calls.append(kw)
        self.panel.update_smooth_ui_state(**kw)


def _staged(panel):
    vol = _vol()
    before = vol.copy()
    f = _PanelFake(vol, panel)
    f.adv.method = "sdf"
    f._preview_smooth()
    assert f._smooth_running
    return f, vol, before


def _s5_empty(panel):
    return (panel.refine_result_note.text() == ""
            and panel.refine_result_warning.text() == ""
            and panel.refine_breadcrumb.text() == "")


# --------------------------------------------------------------------------- #
# the panel derivation
# --------------------------------------------------------------------------- #
class TestThePanelDerivesTheWord:
    def test_a_staged_result_reads_preview_again(self, panel):
        panel.update_smooth_ui_state(reject_enabled=True)
        assert panel.smooth_preview_btn.text() == "Preview again"

    def test_nothing_staged_reads_preview(self, panel):
        panel.update_smooth_ui_state(reject_enabled=True)
        panel.update_smooth_ui_state(reject_enabled=False)
        assert panel.smooth_preview_btn.text() == "Preview"

    def test_an_explicit_preview_text_still_wins(self, panel):
        panel.update_smooth_ui_state(reject_enabled=False, preview_text="X")
        assert panel.smooth_preview_btn.text() == "X"

    def test_a_push_without_reject_leaves_the_word_alone(self, panel):
        panel.update_smooth_ui_state(reject_enabled=True)
        panel.update_smooth_ui_state(status_text="something")
        assert panel.smooth_preview_btn.text() == "Preview again"


# --------------------------------------------------------------------------- #
# the annotator, through the real panel
# --------------------------------------------------------------------------- #
class TestAcceptAndDiscardPutThePageBack:
    def test_a_staged_result_reads_preview_again(self, panel):
        f, _v, _b = _staged(panel)
        assert panel.smooth_preview_btn.text() == "Preview again"

    def test_accept(self, panel):
        f, _v, _b = _staged(panel)
        f._on_label_smoothed_cb = None
        f._accept_smooth()
        assert panel.smooth_preview_btn.text() == "Preview"
        assert panel.smooth_accept_btn.text() == "Accept"
        assert _s5_empty(panel)
        assert panel._refine_result_container.isHidden() is True

    def test_accept_after_a_chain(self, panel):
        f, _v, _b = _staged(panel)
        f.adv.method = "dilate_erode"
        f._on_refine_method_combo_changed(2)
        f._preview_smooth()
        assert panel.smooth_accept_btn.text() == "Accept (2)"
        assert panel.refine_breadcrumb.text() != ""
        f._on_label_smoothed_cb = None
        f._accept_smooth()
        assert panel.smooth_preview_btn.text() == "Preview"
        assert panel.smooth_accept_btn.text() == "Accept"
        assert _s5_empty(panel)

    def test_discard(self, panel):
        f, vol, before = _staged(panel)
        f._reject_smooth()
        assert np.array_equal(vol, before)
        assert panel.smooth_preview_btn.text() == "Preview"
        assert panel.smooth_accept_btn.text() == "Accept"
        assert _s5_empty(panel)
        assert panel._refine_result_container.isHidden() is True

    def test_a_chained_nothing_moved_still_reads_preview_again(self, panel):
        f, _v, _b = _staged(panel)
        f.adv.method = "fill_holes"         # the object has no cavity
        f._preview_smooth()
        assert f._smooth_chain_count == 1
        assert panel.smooth_preview_btn.text() == "Preview again"
        assert panel.smooth_reject_btn.isEnabled()


class TestStopPutsTheWordBack:
    def test_a_mid_run_push_then_stop_reads_preview(self, panel):
        """The worker launch's own push (`rows_enabled=False,
        reject_enabled=False`) and then Stop: S4 reads "Preview", with the
        default tooltip, whatever was staged before the run."""
        from tests.annotator.test_refine_cross_method_chain import _K

        f, vol, before = _staged(panel)
        f._on_refine_stopped = _K()._on_refine_stopped.__get__(f, type(f))
        f._refine_running = True
        f._refine_stopping = True
        f._update_smooth_ui(
            preview_enabled=False, accept_enabled=False,
            reject_enabled=False, busy=True, stop_visible=True,
            rows_enabled=False)
        assert panel.smooth_preview_btn.text() == "Preview"
        f._on_refine_stopped()
        assert panel.smooth_preview_btn.text() == "Preview"
        assert panel.smooth_preview_btn.isEnabled() is True
        assert panel.smooth_accept_btn.text() == "Accept"
        assert panel.smooth_reject_btn.isEnabled() is False
        assert np.array_equal(vol, before)
