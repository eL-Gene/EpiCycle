"""A recipe over a REAL QThread.

The Settle step of a recipe runs on the real ``RefineWorker``: the run must
advance only after ``_on_refine_finished`` wrote, its one judge must be
connected BEFORE ``start()`` (asserted deterministically by wrapping the
thread's start), Stop mid-Settle restores the pre-chain snapshot, an error
ends as refused, and no run outlives its worker (invariant 13).
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from packages.epicycle_annotator import refine_recipe as rr
from tests.annotator import _recipe_fake as rf


@pytest.fixture(scope="module")
def qapp():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _labels(monkeypatch, qapp):
    import napari_epicycle.annotator.curation_3d_widget as mod

    monkeypatch.setattr(mod, "Labels", rf._Layer)


@pytest.fixture
def panel(qapp):
    w = rf.make_panel()
    w.set_refine_channels(["membrane"], current="membrane")
    yield w
    w.setParent(None)


def _walls(shape):
    img = np.zeros(shape, np.float64)
    img[:, 8, 8:32] = img[:, 31, 8:32] = 1.0
    img[:, 8:32, 8] = img[:, 8:32, 31] = 1.0
    return img


def _fake(panel):
    vol = rf.vol_with_notch()
    return rf.Fake(vol, panel, channel_roi=_walls(vol.shape))


def pump(qapp, predicate, seconds=15.0):
    t0 = time.time()
    while time.time() - t0 < seconds:
        qapp.processEvents()
        if predicate():
            qapp.processEvents()
            return True
        time.sleep(0.01)
    return False


SDF = {"sigma_z": 1.5, "sigma_xy": 1.0, "preserve_topology": True}


def _settle_params(panel):
    from napari_epicycle.annotator.advanced_features_widget import (
        REFINE_METHOD_INDEX,
    )

    # read while Settle is ON the combo: its chain drops "Local contrast"
    panel.refine_method_combo.setCurrentIndex(REFINE_METHOD_INDEX["settle"])
    ok, why = panel.set_refine_step_params("settle", dict(
        panel.get_refine_step_params("settle"), wall_kind="membrane",
        band_um=2.0))
    assert ok, why
    return dict(panel.get_refine_step_params("settle"))


def recipe(*steps):
    return rr.recipe_to_json([rr.RecipeStep(m, dict(p)) for m, p in steps],
                             created="t", spacing_known=True,
                             spacing=(3.0, 1.0, 1.0))


def _slow_settle(monkeypatch, gate, entered, *, raise_=False):
    import packages.epicycle_annotator.reshape.settle as settle

    real = settle.settle_with_walls

    def slow(**kw):
        entered.set()
        gate.wait(10.0)
        if raise_:
            raise RuntimeError("the solver fell over")
        return real(**kw)

    monkeypatch.setattr(settle, "settle_with_walls", slow)


class TestASettleStepInsideARecipe:
    def test_it_advances_only_after_the_finish_handler_wrote(self, qapp,
                                                             panel,
                                                             monkeypatch):
        f = _fake(panel)
        settle = _settle_params(panel)
        order = []
        K = rf.K()
        real_fin = K._on_refine_finished
        real_done = K._recipe_on_worker_done

        def fin(self_, outcome):
            order.append(("finished", len(self_._refine_steps)))
            return real_fin(self_, outcome)

        def done(self_, token, i, *a):
            order.append(("recipe", len(self_._refine_steps)))
            assert qapp.thread() is __import__(
                "qtpy.QtCore", fromlist=["QThread"]).QThread.currentThread()
            return real_done(self_, token, i, *a)

        f._on_refine_finished = fin.__get__(f, rf.Fake)
        f._recipe_on_worker_done = done.__get__(f, rf.Fake)
        f.store["s"] = recipe(("sdf", SDF), ("settle", settle),
                              ("dilate_erode", {"radius": 1, "axis": "3D"}))
        before = f.vol().copy()
        f._on_recipe_run("s")
        assert pump(qapp, lambda: f._refine_recipe_run is None)
        assert order == [("finished", 1), ("recipe", 2)]
        assert f._refine_steps == ["Smooth", "Settle", "Dilate/Erode"]
        assert f._smooth_chain_count == 3
        assert f._refine_thread is None
        f._reject_smooth()
        assert np.array_equal(f.vol(), before)

    def test_the_slot_is_connected_BEFORE_the_thread_starts(
            self, qapp, panel, monkeypatch):
        """Deterministic: a wrapper around QThread.start asserts there are
        two receivers on `finished` (the main handler + the recipe's judge)
        before the real start, with a zero-work worker."""
        import packages.epicycle_annotator.reshape.settle as settle
        import qtpy.QtCore as QtCore

        real_settle = settle.settle_with_walls
        precomputed = {}

        def zero_work(**kw):
            return precomputed["res"]

        f = _fake(panel)
        seen = []

        class Checked(QtCore.QThread):
            def start(self, *a):
                w = f._refine_worker
                seen.append((w.receivers(w.finished), w.receivers(w.error)))
                return super().start(*a)

        monkeypatch.setattr(QtCore, "QThread", Checked)
        settle_p = _settle_params(panel)
        # the answer, computed in advance so the worker does no work
        panel.set_refine_step_params("sdf", SDF)
        g = _fake(panel)
        g._preview_smooth()
        roi = g._smooth_roi
        precomputed["res"] = real_settle(
            labels_roi=np.array(g.vol()[roi]), label_id=7,
            wall_kind="membrane", channel_roi=_walls(g.vol().shape)[roi],
            prep_chain=settle_p["prep_chain"], spacing=(3.0, 1.0, 1.0),
            band_um=2.0)
        monkeypatch.setattr(settle, "settle_with_walls", zero_work)
        f.store["s"] = recipe(("sdf", SDF), ("settle", settle_p))
        f._on_recipe_run("s")
        assert pump(qapp, lambda: f._refine_recipe_run is None)
        assert seen == [(2, 2)]
        assert f._refine_steps == ["Smooth", "Settle"]

    def test_stop_mid_settle_restores_the_pre_chain_snapshot(
            self, qapp, panel, monkeypatch):
        gate, entered = threading.Event(), threading.Event()
        _slow_settle(monkeypatch, gate, entered)
        f = _fake(panel)
        before = f.vol().copy()
        panel.set_refine_step_params("sdf", SDF)
        f._preview_smooth()                          # staged before
        settle = _settle_params(panel)
        f.store["s"] = recipe(("fill_holes", {"axis": "3D"}),
                              ("settle", settle))
        f._on_recipe_run("s")
        assert pump(qapp, entered.is_set)
        assert f._refine_recipe_run["phase"] == "waiting"
        assert panel.refine_stage_label.text().startswith("2/2")
        assert panel.refine_cancel_preview_btn.isEnabled()
        f._cancel_refine_preview()
        assert not panel.refine_cancel_preview_btn.isEnabled()  # stopping
        gate.set()
        assert pump(qapp, lambda: f._refine_recipe_run is None)
        assert pump(qapp, lambda: f._refine_thread is None)
        assert np.array_equal(f.vol(), before)
        assert not f._smooth_running
        assert panel.smooth_status_label.text() == (
            "Recipe stopped at step 2 of 2. The object is back as it was "
            "before the first staged step.")
        assert panel.refine_cancel_preview_btn.isHidden()

    def test_a_worker_error_ends_as_refused_and_keeps_step_1(
            self, qapp, panel, monkeypatch):
        gate, entered = threading.Event(), threading.Event()
        gate.set()
        _slow_settle(monkeypatch, gate, entered, raise_=True)
        f = _fake(panel)
        settle = _settle_params(panel)
        f.store["s"] = recipe(("sdf", SDF), ("settle", settle),
                              ("fill_holes", {"axis": "3D"}))
        f._on_recipe_run("s")
        assert pump(qapp, lambda: f._refine_recipe_run is None)
        assert f._smooth_running and f._smooth_chain_count == 1
        assert f._refine_steps == ["Smooth"]
        assert panel.refine_stage_label.toolTip() == (
            "Stopped at 2/3 · Settle — 's'")
        assert "the solver fell over" in panel.refine_result_warning.text()

    def test_abandon_while_waiting_ends_the_run(self, qapp, panel,
                                                monkeypatch):
        """H1: `_abandon_refine_worker` parks the worker's signals; a run
        waiting on them would wait for ever."""
        gate, entered = threading.Event(), threading.Event()
        _slow_settle(monkeypatch, gate, entered)
        f = _fake(panel)
        settle = _settle_params(panel)
        f.store["s"] = recipe(("sdf", SDF), ("settle", settle))
        f._on_recipe_run("s")
        assert pump(qapp, entered.is_set)
        f._abandon_refine_worker()
        assert f._refine_recipe_run is None
        gate.set()
        time.sleep(0.2)
        pump(qapp, lambda: False, seconds=0.5)
        assert f._refine_recipe_run is None
        assert panel.refine_cancel_preview_btn.isHidden()

    def test_a_patch_switch_while_waiting_ends_the_run(self, qapp, panel,
                                                       monkeypatch):
        gate, entered = threading.Event(), threading.Event()
        _slow_settle(monkeypatch, gate, entered)
        f = _fake(panel)
        settle = _settle_params(panel)
        f.store["s"] = recipe(("sdf", SDF), ("settle", settle),
                              ("fill_holes", {"axis": "3D"}))
        f._on_recipe_run("s")
        assert pump(qapp, entered.is_set)
        f._teardown_refine_for_switch()
        assert f._refine_recipe_run is None
        gate.set()
        pump(qapp, lambda: False, seconds=0.5)
        assert f._refine_recipe_run is None
        assert f._refine_steps == []

    def test_cancel_before_run_ends_the_run_through_the_backstop(
            self, qapp, panel, monkeypatch):
        """Stop lands before `run()`: the worker returns without emitting,
        so only the thread's own `finished` (`_on_refine_thread_finished`)
        can end the stop -- and the recipe with it."""
        import qtpy.QtCore as QtCore

        f = _fake(panel)
        stops = []

        class CancelFirst(QtCore.QThread):
            def start(self, *a):
                f._cancel_refine_preview()       # before run() is entered
                stops.append(f._refine_stopping)
                super().start(*a)
                # NO quit() here (audit H2): the worker itself must answer

        monkeypatch.setattr(QtCore, "QThread", CancelFirst)
        before = f.vol().copy()
        settle = _settle_params(panel)
        f.store["s"] = recipe(("sdf", SDF), ("settle", settle))
        f._on_recipe_run("s")
        assert pump(qapp, lambda: f._refine_recipe_run is None
                    and f._refine_thread is None, seconds=10)
        assert stops == [True]
        assert not f._refine_running and not f._refine_stopping
        assert f._refine_lock_fired is False
        assert np.array_equal(f.vol(), before)
        assert panel.smooth_status_label.text().startswith(
            "Recipe stopped at step 2 of 2.")


class TestStopInsideThePumpBeforeAWorkerStep:
    """Audit H1 (b): the Stop lands in the pump of a step that then launches
    a worker.  Nothing may be written, the worker must not strand the run,
    and the lock must come off."""

    def test_nothing_written_and_nothing_stranded(self, qapp, panel,
                                                  monkeypatch):
        from qtpy.QtWidgets import QApplication

        f = _fake(panel)
        settle = _settle_params(panel)
        before = f.vol().copy()
        real = QApplication.processEvents
        fired = []

        def pump_(*a):
            run = f._refine_recipe_run
            if (not fired and run is not None and run["i"] == 1
                    and f._refine_sync_busy):
                fired.append(1)
                f._cancel_refine_preview()
            return real(*a)

        monkeypatch.setattr(QApplication, "processEvents",
                            staticmethod(pump_))
        f.store["s"] = recipe(("sdf", SDF), ("settle", settle))
        f._on_recipe_run("s")
        assert pump(qapp, lambda: f._refine_recipe_run is None
                    and not f._refine_running and f._refine_thread is None)
        assert fired == [1]
        assert np.array_equal(f.vol(), before)
        assert not f._smooth_running
        assert f._refine_lock_fired is False
        assert "stopped at step 2 of 2." in panel.smooth_status_label.text()


class TestAStartThatFails:
    def test_the_run_ends_and_the_lock_comes_off(self, qapp, panel,
                                                 monkeypatch):
        """Audit L3: `start()` raising must not leave the run waiting."""
        import qtpy.QtCore as QtCore

        f = _fake(panel)

        class Broken(QtCore.QThread):
            def start(self, *a):
                raise RuntimeError("no threads left")

        monkeypatch.setattr(QtCore, "QThread", Broken)
        settle = _settle_params(panel)
        before = f.vol().copy()
        f.store["s"] = recipe(("settle", settle), ("sdf", SDF))
        f._on_recipe_run("s")
        assert pump(qapp, lambda: f._refine_recipe_run is None, seconds=5)
        assert not f._refine_running
        assert f._refine_lock_fired is False
        assert np.array_equal(f.vol(), before)
        assert "1/2" in panel.refine_stage_label.toolTip()


class TestAStaleAnswerOnADeadSession:
    def test_releases_the_flag_and_the_lock(self, qapp, panel):
        from napari_epicycle.annotator.reshape_worker import RefineOutcome

        f = _fake(panel)
        f._refine_running = True
        f._refine_lock_fired = False
        f._fire_refine_lock(True)
        f._smooth_running = False            # the session is gone
        f._on_refine_finished(RefineOutcome(result=None, kind="settle"))
        assert f._refine_running is False
        assert f._refine_lock_fired is False
