"""B1b audit, UX lens (2026-09-23): the words the Refine page shows after
Accept, Discard and every refused step.

Every test DRIVES the real methods on the bare fake of
`test_refine_cross_method_chain.py`; the ones about screen text use a REAL
`AdvancedFeaturesWidget` as the fake's panel and read the widget back.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("qtpy")

from tests.annotator.test_refine_cross_method_chain import (  # noqa: E402
    Fake, _K, _Layer, _vol,
)

# B3 audit: "nothing is FINAL" -- a staged shape is in the volume.
DEFAULT_TIP = ("Run the chosen method and show the result. Nothing is "
               "final until you Accept.")


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


class _RealFake(Fake):
    """The fake with the REAL panel as its advanced-features widget."""

    def __init__(self, vol, panel):
        super().__init__(vol)
        self.adv = panel
        self.panel = panel
        self._on_label_smoothed_cb = None

    def _update_smooth_ui(self, **kw):
        self.ui_calls.append(kw)
        self.panel.update_smooth_ui_state(**kw)


def _bind(f, *names):
    K = _K()
    for name in names:
        setattr(f, name, getattr(K, name).__get__(f, type(f)))


def _pick(panel, key):
    from napari_epicycle.annotator.advanced_features_widget import (
        REFINE_METHOD_KEYS,
    )

    for i, _t, k in REFINE_METHOD_KEYS:
        if k == key:
            panel.refine_method_combo.setCurrentIndex(i)
            return i
    raise KeyError(key)


def _real_staged(panel, channel=True):
    f = _RealFake(_vol(), panel)
    panel.set_refine_channels(["membrane"])
    panel.refine_channel_combo.setCurrentIndex(0 if channel else -1)
    _pick(panel, "sdf")
    f._preview_smooth()
    assert f._smooth_running
    return f


def _staged(method="sdf"):
    f = Fake(_vol())
    f._on_label_smoothed_cb = None
    f.adv.method = method
    f._preview_smooth()
    assert f._smooth_running
    return f


# --------------------------------------------------------------------------- #
# UX-M1 -- Accept / Discard hand the button back to the gate
# --------------------------------------------------------------------------- #
class TestAcceptHandsPreviewBackToTheGate:
    def test_settle_under_a_staged_smooth_is_live_through_accept(self, panel):
        """B1c: no Settle crossing -- Settle is offered with the default
        tooltip while Smooth is staged, and after Accept."""
        f = _real_staged(panel)
        f._on_refine_method_combo_changed(_pick(panel, "settle"))
        assert panel.smooth_preview_btn.isEnabled() is True
        assert panel.smooth_preview_btn.toolTip() == DEFAULT_TIP
        f._accept_smooth()
        assert panel.smooth_preview_btn.toolTip() == DEFAULT_TIP
        assert panel.smooth_preview_btn.isEnabled() is True

    def test_and_through_discard(self, panel):
        f = _real_staged(panel)
        f._on_refine_method_combo_changed(_pick(panel, "settle"))
        f._reject_smooth()
        assert panel.smooth_preview_btn.toolTip() == DEFAULT_TIP
        assert panel.smooth_preview_btn.isEnabled() is True

    def test_settle_with_no_channel_comes_back_refused(self, panel):
        f = _real_staged(panel, channel=False)
        f._on_refine_method_combo_changed(_pick(panel, "settle"))
        f._accept_smooth()
        assert panel.smooth_preview_btn.isEnabled() is False
        assert panel.smooth_preview_btn.toolTip() == "Pick a channel first."
        assert panel.smooth_status_label.text() == "Pick a channel first."


# --------------------------------------------------------------------------- #
# UX-M3 -- Accept / Discard say what happened to the OBJECT
# --------------------------------------------------------------------------- #
class _Btn:
    def setEnabled(self, _v):
        pass


class _Lbl:
    def __init__(self):
        self.text = ""

    def setText(self, t):
        self.text = t


def _interp(f):
    f._interp_refine_mode = True
    f._interp_snapshot = None
    f._interp_roi = None
    f._interp_label_id = 7
    f._interp_status_label = _Lbl()
    for name in ("_interpolate_btn", "_accept_interp_btn",
                 "_reject_interp_btn", "_interp_refine_btn"):
        setattr(f, name, _Btn())
    f.sync_manager.get_edited_slices = lambda _n: {}
    f._on_interpolation_accepted_cb = None
    return f


class TestTheWordsAfterAcceptAndDiscard:
    @pytest.mark.parametrize("method", ["sdf", "dilate_erode"])
    def test_accept(self, method):
        f = _staged(method)
        f._accept_smooth()
        assert f.last("status_text") == "Object 7 - accepted."

    @pytest.mark.parametrize("method", ["sdf", "dilate_erode"])
    def test_discard(self, method):
        f = _staged(method)
        f._reject_smooth()
        assert f.last("status_text") == "Object 7 - put back as it was."

    def test_accept_from_an_interpolation(self):
        f = _interp(_staged())
        f._accept_smooth()
        assert f.last("status_text") == "Object 7 - accepted."

    def test_discard_from_an_interpolation(self):
        f = _interp(_staged())
        f._reject_smooth()
        assert f.last("status_text") == "Object 7 - put back as it was."


# --------------------------------------------------------------------------- #
# UX-M4 / UX-L1 -- the Guided Grow refusal and the "still staged" tail
# --------------------------------------------------------------------------- #
class TestTheRefusalSentences:
    def test_guided_grow_first_step_offers_both(self):
        f = Fake(_vol())
        f.adv.method = "guided_grow"
        f._preview_smooth()
        assert "Dilate/Erode or Settle work without one" in (
            f.last("status_text"))

    def test_guided_grow_mid_chain_offers_settle_too(self):
        """B1c: Settle chains like any other method, so it is offered
        mid-chain as well (it was withheld while the crossing stood)."""
        f = _staged()
        f.adv.method = "guided_grow"
        f._preview_smooth()
        text = f.last("status_text")
        assert "Dilate/Erode or Settle work without one." in text
        assert text.endswith("The staged result is still there.")

    def test_two_steps_left_is_plural(self):
        f = _staged()
        f.adv.method = "dilate_erode"
        f._preview_smooth()
        assert f._smooth_chain_count == 2
        f.adv.method = "guided_grow"
        f._preview_smooth()
        assert f.last("status_text").endswith(
            "The earlier steps are still staged.")

    def test_the_click_time_channel_refusal_is_singular(self):
        f = _staged()
        f.adv.method = "contour"             # needs a channel, none chosen
        f._preview_smooth()
        assert f.last("status_text") == (
            "Pick a channel first - this method reads one. The staged "
            "result is still there.")

    def test_a_worker_error_mid_chain_is_singular(self):
        f = _staged()
        _bind(f, "_on_refine_error", "_refine_is_stale")
        f._result_token = lambda: None
        f._refine_launch_labels_name = "labels"
        f._refine_launch_patch = "p"
        f._refine_launch_run_epoch = 0
        f._smooth_chain_count = 2
        f._on_refine_error("boom")
        assert f.last("status_text") == (
            "That step could not be run. The staged result is still there.")


# --------------------------------------------------------------------------- #
# UX-L2 -- a method switch and back restores the INFORMATIVE status
# --------------------------------------------------------------------------- #
class TestTheStagedStatusComesBack:
    def test_on_the_real_panel(self, panel):
        f = _real_staged(panel, channel=False)
        written = panel.smooth_status_label.text()
        assert "voxels changed" in written
        f._on_refine_method_combo_changed(_pick(panel, "contour"))
        assert panel.smooth_status_label.text() == "Pick a channel first."
        f._on_refine_method_combo_changed(_pick(panel, "sdf"))
        assert panel.smooth_status_label.text() == written


# --------------------------------------------------------------------------- #
# UX-L3 -- a refusal is amber, not the staged blue
# --------------------------------------------------------------------------- #
class TestRefusalsAreAmber:
    def test_settle_at_the_click_is_not_a_refusal(self):
        """B1c: the old crossing refusal (amber) is gone -- it chains."""
        from tests.annotator.test_refine_cross_method_chain import (
            _sync_settle,
        )

        f = _sync_settle(_staged())
        n = len(f.ui_calls)
        f.adv.method = "settle"
        f.adv.channel = "membrane"
        f._preview_smooth()
        assert f._smooth_chain_count == 2
        assert all(kw.get("status_kind") != "warn" for kw in f.ui_calls[n:])

    def test_the_channel_at_the_click(self):
        f = _staged()
        f.adv.method = "contour"
        f._preview_smooth()
        assert f.last("status_kind") == "warn"

    def test_the_panel_draws_warn_amber(self, panel):
        from napari_epicycle.annotator.advanced_features_widget import (
            _STATUS_DOT_STYLES,
        )

        assert "warn" in _STATUS_DOT_STYLES
        panel.update_smooth_ui_state(status_kind="warn")
        assert panel._refine_status_dot.styleSheet() == (
            _STATUS_DOT_STYLES["warn"])
        assert _STATUS_DOT_STYLES["warn"] != _STATUS_DOT_STYLES["idle"]


# --------------------------------------------------------------------------- #
# UX-L4 -- a mid-chain Settle that moved nothing says so, not "unchanged"
# --------------------------------------------------------------------------- #
class TestAMidChainSettleThatMovedNothing:
    def _staged_settle_chain(self):
        f = _staged()
        f._refine_staged_method = "settle"
        f._smooth_chain_count = 2
        return f

    def test_a_refused_settle_result(self):
        f = self._staged_settle_chain()
        _bind(f, "_finish_settle_result")
        f._finish_settle_result(
            SimpleNamespace(status="refused", messages=["all neighbours"]),
            7, f._smooth_roi)
        assert f._smooth_chain_count == 1
        assert f.last("status_text") == (
            "Nothing moved in that step. The staged result is still there.")
        assert f.last("accept_enabled") is True

    def test_a_settle_that_would_empty_the_object(self):
        f = self._staged_settle_chain()
        roi = f._smooth_roi
        shape = tuple(s.stop - s.start for s in roi)
        f._finish_refine_preview(
            "settle", np.zeros(shape, np.int32), 5, "m", 7, roi)
        assert f._smooth_chain_count == 1
        assert f.last("status_text") == (
            "Nothing moved in that step. The staged result is still there.")


# --------------------------------------------------------------------------- #
# UX-L5 -- chaining onto another object
# --------------------------------------------------------------------------- #
class TestChainingNeedsTheSameObject:
    def test_the_sentence(self):
        f = _staged()

        class _Other(_Layer):
            selected_label = 9

        f._layer_by_name = lambda _layers, _name: _Other()
        f.adv.method = "dilate_erode"
        f._preview_smooth()
        assert f.last("status_text") == (
            "Chaining needs the same object. Accept or Discard first.")
        assert f._smooth_chain_count == 1
