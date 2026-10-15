"""A saved Refine recipe, headless, over N objects -- the pure dispatcher.

Invariants 6, 12, 17: one
case per verdict (refused -> the object STOPS and keeps its earlier steps;
skipped -> the chain goes on; emptied -> nothing of the object is written;
cancelled), the replica ROI sequence and its growth, the room refusal, the
clip cases, cancel between steps, the cancel poll and the orphans, the
worker cap, bounded outcomes.  Napari-free.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from packages.epicycle_annotator import batch_chain as bc
from packages.epicycle_annotator import batch_refine as br
from packages.epicycle_annotator import refine_recipe as rr
from packages.epicycle_annotator import refine as refine_mod
from packages.epicycle_annotator import smooth as smooth_mod
from packages.epicycle_annotator.batch_refine import run_batch_recipe
from packages.epicycle_annotator.reshape import settle as settle_mod
from packages.epicycle_annotator.refine import ContourResult

TICKS = {k: False for k in rr.PREP_TICKS}


def sdf(sz=1.5, sxy=1.0):
    return rr.RecipeStep("sdf", {"sigma_z": sz, "sigma_xy": sxy,
                                 "preserve_topology": True})


def dil(r, axis="3D"):
    return rr.RecipeStep("dilate_erode", {"radius": r, "axis": axis})


def fill(axis="3D"):
    return rr.RecipeStep("fill_holes", {"axis": axis})


def guided(r=5, thr=0.5):
    return rr.RecipeStep("guided_grow", {"max_radius": r,
                                         "prob_threshold": thr, "axis": "3D"})


def settle(kind="membrane", band=2.0, solver="watershed"):
    return rr.RecipeStep("settle", {
        "wall_kind": kind, "solver": solver, "band_um": band, "n_iter": 10,
        "clamp": 0.5, "compactness": 0.0, "prep_ticks": dict(TICKS),
        "prep_chain": "none"})


def contour(axis="3D", limit=3.0, gate=0.3, balloon=0.0, kind="membrane"):
    return rr.RecipeStep("contour", {
        "mode": "snap", "wall_kind": kind, "axis": axis, "sampling": "share",
        "share_pct": 25, "n_slices": 12, "limit_um": limit,
        "iterations": 20, "balloon": balloon, "gate": gate, "blur_um": 2.0,
        "sharpness": 100, "rebuild_um": 1.5, "lambda1": 1.0, "lambda2": 1.0,
        "prep_ticks": dict(TICKS), "prep_chain": "none"})


def recipe(*steps):
    return rr.Recipe(steps=tuple(steps))


def cube_vol():
    """Object 7 (a notched cube) with a neighbour 3 beside it."""
    v = np.zeros((40, 60, 60), np.int32)
    v[10:26, 15:35, 15:35] = 7
    v[10:26, 15:18, 15:18] = 0
    v[10:26, 15:35, 38:48] = 3
    return v


def box(v, lid):
    idx = np.argwhere(v == lid)
    return tuple(slice(int(a), int(b) + 1)
                 for a, b in zip(idx.min(0), idx.max(0)))


def run_one(v, steps, lid=7, **kw):
    kw.setdefault("spacing", (1.0, 1.0, 1.0))
    return bc.run_chain_on_label(v, lid, tuple(steps), bbox=box(v, lid), **kw)


def paste(v, out, lid=7):
    """The object's final mask as run_chain's outcome describes it."""
    m = np.zeros(v.shape, bool)
    if out.mask is not None:
        sl = tuple(slice(o, o + n) for o, n in zip(out.offset, out.mask.shape))
        m[sl] = out.mask
    return m


class _FakeSettle:
    """Replaces ``settle_with_walls`` with a scripted answer."""

    def __init__(self, status, messages=("m",), grow=0):
        self.status = status
        self.messages = list(messages)
        self.grow = grow
        self.calls = []

    def __call__(self, labels_roi, label_id, **kw):
        self.calls.append((labels_roi.shape, kw))
        mask = labels_roi == label_id
        if self.status == "ok" and self.grow:
            from scipy.ndimage import binary_dilation

            mask = binary_dilation(mask, iterations=self.grow) & (
                (labels_roi == 0) | (labels_roi == label_id))
        return settle_mod.SettleResult(
            result_roi=np.where(mask, label_id, 0).astype(labels_roi.dtype),
            status=self.status, messages=list(self.messages),
            elevation=np.ones(labels_roi.shape))


def _fake_contour(reason, move=0):
    def fn(volume_roi, label_id, **kw):
        mask = volume_roi == label_id
        if move:
            from scipy.ndimage import binary_dilation

            mask = binary_dilation(mask, iterations=move) & (
                (volume_roi == 0) | (volume_roi == label_id))
        res = ContourResult(np.where(mask, label_id, 0).astype(
            volume_roi.dtype), int(move), "the message " + str(reason),
            reason=reason)
        res.details["elevation"] = np.ones(volume_roi.shape)
        return res
    return fn


# --------------------------------------------------------------- verdicts


class TestSettleVerdicts:
    def test_ok_is_written(self, monkeypatch):
        monkeypatch.setattr(settle_mod, "settle_with_walls",
                            _FakeSettle("ok", grow=1))
        v = cube_vol()
        out = run_one(v, [settle()], channel=np.zeros(v.shape))
        assert out.status == "ok" and out.n_changed > 0

    @pytest.mark.parametrize("msg", [
        "Nothing moved: this object's outline already touches its "
        "neighbours everywhere, and Settle only moves edges that face "
        "background.",
        "Nothing moved: the wall signal already runs along this object's "
        "outline.",
    ])
    def test_unchanged_is_skipped_and_the_chain_goes_on(self, monkeypatch,
                                                         msg):
        monkeypatch.setattr(settle_mod, "settle_with_walls",
                            _FakeSettle("unchanged", [msg]))
        v = cube_vol()
        out = run_one(v, [settle(), fill()], channel=np.zeros(v.shape))
        # the chain went on to Fill Holes (nothing to fill either) and the
        # object is simply unchanged -- NOT stopped
        assert out.status == "unchanged"
        assert out.step == 2

    def test_refused_stops_and_keeps_the_earlier_steps(self, monkeypatch):
        monkeypatch.setattr(settle_mod, "settle_with_walls",
                            _FakeSettle("refused", ["would empty it"]))
        v = cube_vol()
        smooth_only = run_one(v, [sdf()])
        out = run_one(v, [sdf(), settle()], channel=np.zeros(v.shape))
        assert out.status == "stopped" and out.step == 2
        assert out.reason == "would empty it"
        assert np.array_equal(paste(v, out), paste(v, smooth_only))

    def test_smooth_then_settle_on_an_enclosed_object_keeps_the_smooth(
            self, monkeypatch):
        monkeypatch.setattr(settle_mod, "settle_with_walls", _FakeSettle(
            "unchanged", ["Nothing moved: this object's outline already "
                          "touches its neighbours everywhere, and Settle "
                          "only moves edges that face background."]))
        v = cube_vol()
        smooth_only = run_one(v, [sdf()])
        out = run_one(v, [sdf(), settle()], channel=np.zeros(v.shape))
        assert out.status == "ok"
        assert np.array_equal(paste(v, out), paste(v, smooth_only))

    def test_real_settle_on_a_fully_enclosed_object_stops_and_keeps(self):
        """REAL Settle on an object with NO background within its band
        returns ``status == "refused"`` ("There is no background left around
        this object at that distance"), not "unchanged" -- so by the verdict
        table the object STOPS at that step, and keeps the Smooth before it
        (D12).  The live driver check expected
        "nothing moved"."""
        v = np.full((40, 40, 40), 2, np.int32)
        v[12:28, 12:28, 12:28] = 7
        v[12:28, 12:15, 12:15] = 2
        smooth_only = run_one(v, [sdf()])
        out = run_one(v, [sdf(), settle(kind="distance")])
        assert out.status == "stopped" and out.step == 2
        assert "no background" in out.reason
        # the prefix the summary keys its "enclosed" wording on
        assert out.reason.startswith(br.SETTLE_ENCLOSED_REASON)
        assert np.array_equal(paste(v, out), paste(v, smooth_only))


class TestContourVerdicts:
    @pytest.mark.parametrize("reason", [None, "limit", "fence"])
    def test_ran_but_unmoved_is_skipped(self, monkeypatch, reason):
        monkeypatch.setattr(refine_mod, "contour_with_walls",
                            _fake_contour(reason))
        v = cube_vol()
        out = run_one(v, [contour(), fill()], channel=np.zeros(v.shape))
        assert out.status == "unchanged" and out.step == 2

    def test_ran_and_moved_is_written(self, monkeypatch):
        monkeypatch.setattr(refine_mod, "contour_with_walls",
                            _fake_contour(None, move=1))
        v = cube_vol()
        out = run_one(v, [contour()], channel=np.zeros(v.shape))
        assert out.status == "ok" and out.n_changed > 0

    @pytest.mark.parametrize("reason",
                             ["missing", "too_large", "push_below_gate"])
    def test_refusals_stop(self, monkeypatch, reason):
        monkeypatch.setattr(refine_mod, "contour_with_walls",
                            _fake_contour(reason))
        v = cube_vol()
        out = run_one(v, [sdf(), contour()], channel=np.zeros(v.shape))
        assert out.status == "stopped" and out.step == 2
        assert out.reason == "the message " + reason

    def test_stopped_is_a_cancel_not_a_refusal(self, monkeypatch):
        monkeypatch.setattr(refine_mod, "contour_with_walls",
                            _fake_contour("stopped"))
        v = cube_vol()
        out = run_one(v, [sdf(), contour()], channel=np.zeros(v.shape))
        assert out.status == "cancelled" and out.mask is None

    def test_smooth_then_a_refused_contour_keeps_the_smooth(self):
        # REAL contour: a push below the gate is its own refusal
        v = cube_vol()
        smooth_only = run_one(v, [sdf()])
        out = run_one(v, [sdf(), contour(balloon=0.1, gate=0.9)],
                      channel=np.zeros(v.shape))
        assert out.status == "stopped" and out.step == 2
        assert np.array_equal(paste(v, out), paste(v, smooth_only))


class TestSyncMethodsUnmoved:
    @pytest.mark.parametrize("step", [dil(0), fill()])
    def test_unmoved_is_skipped(self, step):
        v = cube_vol()
        out = run_one(v, [step, sdf()])
        # the chain went on and smoothed
        assert out.status == "ok" and out.step == 2

    def test_fill_then_smooth_on_an_object_without_holes_smooths_it(self):
        v = cube_vol()
        a = run_one(v, [fill(), sdf()])
        b = run_one(v, [sdf()])
        assert a.status == "ok"
        assert np.array_equal(paste(v, a), paste(v, b))

    def test_guided_grow_unmoved_is_skipped(self):
        v = cube_vol()
        prob = np.zeros(v.shape, np.float32)          # nothing above 0.5
        out = run_one(v, [guided(), sdf()], prob=prob)
        assert out.status == "ok" and out.step == 2


class TestExceptions:
    def test_an_exception_stops_with_the_fixed_sentence(self, monkeypatch,
                                                        capsys):
        def boom(*a, **k):
            raise KeyError("secret-raw-reason")

        monkeypatch.setattr(refine_mod, "fill_holes_label", boom)
        v = cube_vol()
        smooth_only = run_one(v, [sdf()])
        out = run_one(v, [sdf(), fill()])
        assert out.status == "stopped" and out.step == 2
        assert out.reason == bc.BATCH_REFUSED_FAILED == "the step failed"
        assert np.array_equal(paste(v, out), paste(v, smooth_only))
        assert "secret-raw-reason" in capsys.readouterr().out


class TestStoppedVsEmptied:
    def test_refusal_at_step_one_writes_nothing_but_is_listed(
            self, monkeypatch):
        monkeypatch.setattr(settle_mod, "settle_with_walls",
                            _FakeSettle("refused", ["no"]))
        v = cube_vol()
        res = run_batch_recipe(v, [7], recipe(settle(), sdf()),
                               channel=np.zeros(v.shape), n_workers=1)
        assert np.array_equal(res.working_volume, v)
        assert res.stopped == {7: (1, "no")}
        assert res.touched_ids == [] and res.emptied_ids == []

    def test_an_emptying_step_writes_nothing_the_smooth_included(self):
        v = np.zeros((30, 30, 30), np.int32)
        v[10:14, 5:25, 5:25] = 7                      # a thin plate
        res = run_batch_recipe(v, [7], recipe(sdf(), dil(-10)),
                               n_workers=1)
        assert np.array_equal(res.working_volume, v)
        assert res.emptied_ids == [7]
        assert 7 not in res.stopped

    def test_smooth_then_a_refused_settle_keeps_the_smooth(self,
                                                          monkeypatch):
        monkeypatch.setattr(settle_mod, "settle_with_walls",
                            _FakeSettle("refused", ["would empty"]))
        v = cube_vol()
        a = run_batch_recipe(v, [7], recipe(sdf()), n_workers=1)
        b = run_batch_recipe(v, [7], recipe(sdf(), settle()),
                             channel=np.zeros(v.shape), n_workers=1)
        assert np.array_equal(a.working_volume, b.working_volume)
        assert b.stopped == {7: (2, "would empty")}
        assert b.touched_ids == [7]

    def test_a_stopped_object_emptied_at_write_moves_to_emptied(self):
        # object 5 (lower id) grows over all of object 7's kept mask
        # object 7 (at x 30:32) moved its kept mask onto background at
        # x 18:20; object 5 (lower id, written first) grew over x 2:22.
        v = np.zeros((20, 20, 40), np.int32)
        v[8:12, 8:12, 30:32] = 7
        v[2:18, 2:18, 2:17] = 5
        outs = {
            5: bc.ChainOutcome("ok", 1, mask=np.ones((16, 16, 20), bool),
                               offset=(2, 2, 2), n_changed=1,
                               box=((slice(2, 18), slice(2, 18),
                                     slice(2, 17)))),
            7: bc.ChainOutcome("stopped", 2, "r",
                               mask=np.ones((4, 4, 2), bool),
                               offset=(8, 8, 18), n_changed=1,
                               box=(slice(8, 12), slice(8, 12),
                                    slice(30, 32))),
        }
        import packages.epicycle_annotator.batch_chain as mod

        orig = mod.run_chain_batch
        try:
            mod.run_chain_batch = lambda *a, **k: (
                sorted(outs.items()), False)
            res = run_batch_recipe(v, [5, 7], recipe(sdf(), fill()))
        finally:
            mod.run_chain_batch = orig
        assert 7 in res.emptied_ids and 7 not in res.stopped
        assert np.array_equal(res.working_volume == 7, v == 7)


class TestReplicaRoi:
    def test_first_roi_is_bbox_plus_the_step_margin(self, monkeypatch):
        seen = []
        real = refine_mod.fill_holes_label

        def spy(volume_roi, label_id, **kw):
            seen.append(volume_roi.shape)
            return real(volume_roi, label_id, **kw)

        monkeypatch.setattr(refine_mod, "fill_holes_label", spy)
        v = np.zeros((80, 80, 80), np.int32)
        v[30:40, 30:45, 30:50] = 7
        run_one(v, [fill()])
        assert seen == [(10 + 30, 15 + 30, 20 + 30)]

    def test_the_roi_grows_as_the_refine_tab_grows_it(self, monkeypatch):
        seen = []
        real = refine_mod.dilate_erode_label

        def spy(volume_roi, label_id, **kw):
            seen.append(volume_roi.shape)
            return real(volume_roi, label_id, **kw)

        monkeypatch.setattr(refine_mod, "dilate_erode_label", spy)
        v = np.zeros((120, 120, 120), np.int32)
        v[50:60, 50:60, 50:60] = 7
        run_one(v, [dil(10), dil(10)])
        # step 1: bbox + 15 = 40; the object then clears the wall by 5 < 15,
        # so step 2's ROI = old U (bbox(cur) + 15) = 30 + 30 = 60
        assert seen == [(40, 40, 40), (60, 60, 60)]

    def test_dilate_ten_twice_is_not_clipped(self):
        from scipy.ndimage import binary_dilation

        v = np.zeros((120, 120, 120), np.int32)
        v[50:60, 50:60, 50:60] = 7
        out = run_one(v, [dil(10), dil(10)])
        want = binary_dilation(v == 7, iterations=20)
        assert np.array_equal(paste(v, out), want)

    def test_growth_past_its_bound_stops_with_the_room_reason(
            self, monkeypatch):
        from scipy.ndimage import binary_dilation

        def greedy(volume_roi, label_id, **kw):     # a "fill" that bulges 13
            m = binary_dilation(volume_roi == label_id, iterations=13)
            return np.where(m, label_id, 0).astype(volume_roi.dtype), 1, ""

        monkeypatch.setattr(refine_mod, "fill_holes_label", greedy)
        v = np.zeros((120, 120, 120), np.int32)
        v[50:60, 50:60, 50:60] = 7
        out = run_one(v, [fill(), fill()])
        assert out.status == "stopped" and out.step == 2
        assert out.reason == bc.BATCH_REFUSED_ROOM
        assert out.n_changed > 0                     # step 1 kept

    @pytest.mark.parametrize("step", [sdf(5.0, 5.0), guided(15, 0.2)])
    def test_the_clip_cases_clip_as_the_tab_does(self, step):
        v = np.zeros((90, 90, 90), np.int32)
        v[35:55, 35:55, 35:55] = 7
        prob = np.full(v.shape, 0.9, np.float32)
        roi = tuple(slice(20, 70) for _ in range(3))    # bbox + 15
        fn, kw = bc.chain_step_call(step, v[roi], 7, None,
                                    prob[roi].astype(np.float64),
                                    (1.0, 1.0, 1.0))
        want = np.zeros(v.shape, bool)
        want[roi] = fn(**kw)[0] == 7
        out = run_one(v, [step], prob=prob)
        assert np.array_equal(paste(v, out), want)


class TestCancel:
    def test_cancel_between_steps(self, monkeypatch):
        ev = threading.Event()
        real = refine_mod.fill_holes_label
        calls = []

        def first(volume_roi, label_id, **kw):
            calls.append(1)
            ev.set()
            return real(volume_roi, label_id, **kw)

        monkeypatch.setattr(refine_mod, "fill_holes_label", first)
        v = cube_vol()
        out = run_one(v, [fill(), fill()], cancel_event=ev)
        assert out.status == "cancelled" and calls == [1]

    def test_the_cancel_poll_returns_promptly_and_orphans_the_slow_one(
            self, monkeypatch):
        ev = threading.Event()
        release = threading.Event()

        def slow(volume_roi, label_id, **kw):
            release.wait(5.0)
            return (volume_roi == label_id).astype(volume_roi.dtype) * \
                label_id, 0, ""

        monkeypatch.setattr(refine_mod, "fill_holes_label", slow)
        v = cube_vol()
        orphans = []
        threading.Timer(0.3, ev.set).start()
        t0 = time.monotonic()
        try:
            res = run_batch_recipe(v, [7], recipe(fill()), cancel_event=ev,
                                   orphans=orphans, n_workers=1)
            elapsed = time.monotonic() - t0
        finally:
            release.set()
        assert res.cancelled
        assert elapsed < 0.3 + 0.5
        assert len(orphans) == 1

    def test_worker_emits_finished_cancelled(self, monkeypatch):
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from qtpy.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        from napari_epicycle.annotator.batch_refine_worker import (
            BatchRefineWorker,
        )

        release = threading.Event()

        def slow(volume_roi, label_id, **kw):
            release.wait(5.0)
            return volume_roi, 0, ""

        monkeypatch.setattr(refine_mod, "fill_holes_label", slow)
        v = cube_vol()
        orphans = []
        w = BatchRefineWorker(v, [7], recipe(fill()), orphans=orphans)
        got = []
        w.finished.connect(lambda r: got.append(("finished", r)))
        w.error.connect(lambda m: got.append(("error", m)))
        threading.Timer(0.2, w.cancel).start()
        t0 = time.monotonic()
        try:
            w.run()
        finally:
            release.set()
        assert time.monotonic() - t0 < 0.2 + 0.5
        assert len(got) == 1 and got[0][0] == "finished"
        assert got[0][1].cancelled


class TestPoolAndOutcomes:
    def test_worker_cap(self, monkeypatch):
        import os

        monkeypatch.setattr(os, "cpu_count", lambda: 32)
        assert bc.chain_worker_count() == bc.CHAIN_MAX_WORKERS == 2
        monkeypatch.setattr(os, "cpu_count", lambda: 2)
        assert bc.chain_worker_count() == 1
        assert bc.chain_worker_count(3) == 3

    def test_outcomes_are_bounded(self, monkeypatch):
        monkeypatch.setattr(settle_mod, "settle_with_walls",
                            _FakeSettle("ok", grow=1))
        v = cube_vol()
        out = run_one(v, [sdf(), settle()], channel=np.zeros(v.shape))
        m = paste(v, out)
        assert out.mask.shape == tuple(s.stop - s.start for s in box(
            m.astype(np.int32), 1))
        assert out.mask.dtype == bool
        assert not any(isinstance(getattr(out, f), np.ndarray)
                       for f in ("offset", "box", "reason"))
        assert not hasattr(out, "elevation")

    def test_many_objects_are_deterministic_across_worker_counts(self):
        v = np.zeros((30, 40, 120), np.int32)
        for i in range(5):
            v[8:20, 10:30, 5 + 23 * i:25 + 23 * i] = i + 1
        a = run_batch_recipe(v, [1, 2, 3, 4, 5], recipe(dil(2), sdf()),
                             n_workers=1)
        b = run_batch_recipe(v, [1, 2, 3, 4, 5], recipe(dil(2), sdf()),
                             n_workers=3)
        assert np.array_equal(a.working_volume, b.working_volume)
        assert a.touched_ids == b.touched_ids

    def test_no_write_outside_the_targets_and_absent_reported(self):
        v = cube_vol()
        res = run_batch_recipe(v, [7, 99], recipe(dil(2)), n_workers=1)
        assert res.absent_ids == [99]
        changed = res.working_volume != v
        assert set(np.unique(res.working_volume[changed])) <= {0, 7}
        assert (res.working_volume == 3).sum() == (v == 3).sum()
