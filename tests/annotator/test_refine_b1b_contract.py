"""B1b audit, contract lens (2026-09-23): what a refused or failed step
leaves on screen and in the session once cross-method chaining is allowed.

Every test DRIVES the real methods on the bare fake of
`test_refine_cross_method_chain.py`; the ones about screen text forward
each push to a REAL `AdvancedFeaturesWidget` and read the widget back.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("qtpy")

from tests.annotator.test_refine_cross_method_chain import (  # noqa: E402
    Fake, _K, _Layer, _vol,
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
    """The fake, with every push landing on a real panel."""

    def __init__(self, vol, panel):
        super().__init__(vol)
        self.panel = panel

    def _update_smooth_ui(self, **kw):
        self.ui_calls.append(kw)
        self.panel.update_smooth_ui_state(**kw)


def _bind(f, *names):
    K = _K()
    for name in names:
        setattr(f, name, getattr(K, name).__get__(f, type(f)))


def _staged(f):
    f.adv.method = "sdf"
    f._preview_smooth()
    assert f._smooth_running and f._smooth_chain_count == 1
    return f


def _staged_empty(monkeypatch, f):
    """A staged Smooth that emptied the object."""
    import packages.epicycle_annotator.smooth as smooth_mod

    def _gone(roi_data, label_id, **_kw):
        n = int(np.count_nonzero(roi_data == label_id))
        return np.where(roi_data == label_id, 0, roi_data), n, "gone"

    monkeypatch.setattr(smooth_mod, "smooth_label_sdf", _gone)
    f.adv.method = "sdf"
    f._preview_smooth()
    assert f.panel.smooth_accept_btn.text() == "Delete object 7"
    assert "removes the whole object" in f.panel.refine_result_warning.text()
    return f


# --------------------------------------------------------------------------- #
# C-M1 -- a refused step keeps the staged S5 card
# --------------------------------------------------------------------------- #
class TestARefusedStepKeepsTheStagedCard:
    def test_guided_grow_refused_after_an_empty_result(self, panel,
                                                        monkeypatch):
        f = _staged_empty(monkeypatch, _PanelFake(_vol(), panel))
        f.adv.method = "guided_grow"            # no probability map
        f._preview_smooth()
        assert f._smooth_chain_count == 1
        assert panel.smooth_accept_btn.text() == "Delete object 7"
        assert "removes the whole object" in (
            panel.refine_result_warning.text())
        assert panel._refine_result_container.isHidden() is False

    def test_a_chained_nothing_moved_after_an_empty_result(self, panel,
                                                           monkeypatch):
        f = _staged_empty(monkeypatch, _PanelFake(_vol(), panel))
        f.adv.method = "fill_holes"             # nothing left to fill
        f._preview_smooth()
        assert f._smooth_chain_count == 1
        assert panel.smooth_accept_btn.text() == "Delete object 7"
        assert "removes the whole object" in (
            panel.refine_result_warning.text())

    def test_a_refused_step_leaves_the_note_alone(self, panel):
        f = _staged(_PanelFake(_vol(), panel))
        note = panel.refine_result_note.text()
        assert note.startswith("From Smooth")
        f.adv.method = "guided_grow"
        f._preview_smooth()
        assert panel.refine_result_note.text() == note


# --------------------------------------------------------------------------- #
# C-M2 -- a worker that cannot start mid-chain keeps the chain
# --------------------------------------------------------------------------- #
@pytest.fixture
def broken_thread(monkeypatch, qapp):
    import qtpy.QtCore as qc

    real = qc.QThread

    class _NoStart(real):
        def start(self, *a, **k):
            raise RuntimeError("no thread for you")

    monkeypatch.setattr(qc, "QThread", _NoStart)
    return _NoStart


def _contour_ready(f):
    _bind(f, "_refine_contour_call", "_launch_refine_worker",
          "_on_refine_progress", "_on_refine_message",
          "_on_refine_finished", "_on_refine_error",
          "_on_refine_thread_finished")
    f.adv.method = "contour"
    f.adv.channel = "membrane"
    f.adv.wall_kind = "distance"               # no image read at all
    return f


class TestALaunchFailureMidChain:
    def test_the_chain_stays_staged_and_locked(self, broken_thread):
        vol = _vol()
        before = vol.copy()
        f = _staged(Fake(vol))
        staged_vol = vol.copy()
        _contour_ready(f)
        f._preview_smooth()
        assert f._refine_running is False
        assert f._smooth_running is True
        assert f._smooth_chain_count == 1
        assert f._refine_lock_fired is True
        assert False not in f.lock_calls
        assert f.last("accept_enabled") is True
        assert f.last("reject_enabled") is True
        assert "Could not start" in f.last("status_text")
        assert np.array_equal(vol, staged_vol)
        f._reject_smooth()
        assert np.array_equal(vol, before)

    def test_a_first_step_that_cannot_start_ends_the_session(
            self, broken_thread):
        vol = _vol()
        f = _contour_ready(Fake(vol))
        f._preview_smooth()
        assert f._refine_running is False
        assert f._smooth_running is False
        assert f._smooth_snapshot is None
        assert f._refine_lock_fired is False
        assert "Could not start" in f.last("status_text")


# --------------------------------------------------------------------------- #
# C-L1 -- a worker answer with no result, mid-chain
# --------------------------------------------------------------------------- #
class TestANoAnswerMidChain:
    @pytest.mark.parametrize("result", [None, ("only", "two")])
    def test_keeps_the_earlier_steps(self, result):
        vol = _vol()
        before = vol.copy()
        f = _staged(Fake(vol))
        _bind(f, "_on_refine_finished", "_refine_is_stale")
        f._result_token = lambda: None
        f._refine_launch_labels_name = "labels"
        f._refine_launch_patch = "p"
        f._refine_launch_run_epoch = 0
        f._smooth_chain_count = 2           # as the chained launch left it
        f._refine_running = True
        f._on_refine_finished(SimpleNamespace(kind="contour", result=result))
        assert f._smooth_running is True
        assert f._smooth_snapshot is not None
        assert f._smooth_chain_count == 1
        assert f.last("accept_enabled") is True
        f._reject_smooth()
        assert np.array_equal(vol, before)


# --------------------------------------------------------------------------- #
# C-L2 -- the gate puts S1 back once its own refusal no longer holds
# --------------------------------------------------------------------------- #
def _real(panel):
    """The fake with the REAL panel as its advanced-features widget."""
    f = _PanelFake(_vol(), panel)
    f.adv = panel
    return f


def _pick(panel, key):
    from napari_epicycle.annotator.advanced_features_widget import (
        REFINE_METHOD_KEYS,
    )

    for i, _t, k in REFINE_METHOD_KEYS:
        if k == key:
            panel.refine_method_combo.setCurrentIndex(i)
            return i
    raise KeyError(key)


def _blank_channel(panel):
    panel.set_refine_channels(["membrane"])
    panel.refine_channel_combo.setCurrentIndex(-1)
    assert panel.refine_channel_name() == ""


class TestTheGatePutsTheStatusBack:
    def test_staged_then_a_channel_is_picked(self, panel):
        f = _real(panel)
        _blank_channel(panel)
        f._preview_smooth()
        assert f._smooth_running
        staged_status = panel.smooth_status_label.text()
        i = _pick(panel, "contour")
        f._on_refine_method_combo_changed(i)
        assert panel.smooth_status_label.text() == "Pick a channel first."
        assert panel.smooth_preview_btn.isEnabled() is False
        panel.refine_channel_combo.setCurrentIndex(0)
        f._refresh_refine_preview_gate()      # what the wired combo calls
        assert panel.smooth_status_label.text() == staged_status
        assert panel.smooth_preview_btn.isEnabled() is True

    def test_idle_settle_then_smooth(self, panel):
        f = _real(panel)
        _blank_channel(panel)
        i = _pick(panel, "settle")
        f._on_refine_method_combo_changed(i)
        assert panel.smooth_status_label.text() == "Pick a channel first."
        i = _pick(panel, "sdf")
        f._on_refine_method_combo_changed(i)
        assert "channel" not in panel.smooth_status_label.text()
        assert panel.smooth_preview_btn.isEnabled() is True


# --------------------------------------------------------------------------- #
# C-L3 -- the synchronous path's processEvents is not a door back in
# --------------------------------------------------------------------------- #
class TestTheSyncStepCannotBeReentered:
    def test_gate_and_preview_are_refused_inside_the_pump(self, monkeypatch):
        from qtpy.QtWidgets import QApplication

        f = _staged(Fake(_vol()))
        seen = {}

        def _pump(*_a, **_k):
            if seen:
                return
            seen["gate"] = f._refresh_refine_preview_gate()
            f._preview_smooth()               # a click delivered mid-step
            seen["count"] = f._smooth_chain_count

        monkeypatch.setattr(QApplication, "processEvents",
                            staticmethod(_pump))
        f.adv.method = "dilate_erode"
        f._preview_smooth()
        assert seen["gate"] is None
        assert seen["count"] == 2
        assert f._smooth_chain_count == 2
        assert f._refine_steps == ["Smooth", "Dilate/Erode"]
        assert getattr(f, "_refine_sync_busy", False) is False


# --------------------------------------------------------------------------- #
# C-L4 -- a chained Settle that needs more room GROWS the region (B1c,
# decided 2026-09-23; it was a refusal in B1b)
# --------------------------------------------------------------------------- #
class TestSettleOnSettleKeepsItsRegion:
    def _staged_settle(self, band_um):
        vol = _vol()
        f = Fake(vol)
        roi, snap = f._snapshot_label_roi(vol, 7, margin=15)
        f._smooth_running = True
        f._smooth_snapshot = snap
        f._smooth_roi = roi
        f._smooth_label_id = 7
        f._smooth_chain_count = 1
        f._smooth_roi_margin = 15
        f._refine_staged_method = "settle"
        f._refine_steps = ["Settle"]
        f.adv.method = "settle"
        f.adv.channel = "membrane"
        f.adv.band_um = band_um
        return f

    def test_a_bigger_edge_may_move_grows_the_region(self):
        f = self._staged_settle(40.0)       # 40 um at 1 um xy = 40 voxels
        roi1 = f._smooth_roi
        # B3 audit: the rule is the object's REAL clearance.  This fixture's
        # 15-voxel ROI already reaches the volume's edge on every side, so
        # there is nothing to grow into: the clearance is infinite and the
        # region stays -- a 40-voxel need is met by the whole volume.
        assert f._refine_room("settle", {"band_um": 40.0}) == (
            40, float("inf"))
        f._preview_smooth()
        assert f._smooth_running is True
        assert all(n.start <= o.start and o.stop <= n.stop
                   for o, n in zip(roi1, f._smooth_roi))
        assert "bigger region" not in (f.last("status_text") or "")

    def test_the_same_edge_may_move_keeps_the_region(self):
        f = self._staged_settle(2.0)
        roi1 = f._smooth_roi
        f._preview_smooth()
        assert f._smooth_roi == roi1
        assert f._smooth_roi_margin == 15
