"""Settle's wall picture is built on the WORKER (REFINE B3).

The contour moved first (B2 audit C-M3, ``refine.contour_with_walls``).
Settle still built its picture on the main thread inside
``_launch_refine_method`` -- 1.47 s of frozen GUI on the owner's patch
before the busy strip even appeared.  ``settle.settle_with_walls`` is the
twin: the orchestrator hands over the channel / probability CROPS and the
decisions, and the preparation, the picture and the solve all run on the
worker.  The picture rides the answer (``SettleResult.elevation``) and is
parked as the pending picture exactly where the contour's is.

The same file pins the other half of B3's strip change: the stage line no
longer carries "Drawing is off until this finishes." (owner decision on B2
audit UX-M7).
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

pytest.importorskip("qtpy")


@pytest.fixture(scope="module")
def qapp():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        try:
            app = QApplication([])
        except Exception as exc:  # pragma: no cover - CI without Qt
            pytest.skip("cannot create a QApplication: {}".format(exc))
    return app


def _ring():
    """An object with a bright shell two voxels outside it."""
    vol = np.zeros((12, 30, 30), np.int32)
    vol[3:9, 10:20, 10:20] = 7
    img = np.zeros(vol.shape, np.float64)
    img[2:10, 8:22, 8] = 1.0
    img[2:10, 8:22, 21] = 1.0
    img[2:10, 8, 8:22] = 1.0
    img[2:10, 21, 8:22] = 1.0
    return vol, img


# --------------------------------------------------------------------------- #
# the pure function
# --------------------------------------------------------------------------- #
class TestSettleWithWalls:
    def test_it_matches_the_old_main_thread_recipe_byte_for_byte(self):
        """The picture the main thread used to build, and the solve on it."""
        from packages.epicycle_annotator.reshape.preproc import apply_preproc
        from packages.epicycle_annotator.reshape.settle import (
            settle_label,
            settle_with_walls,
        )
        from packages.epicycle_annotator.reshape.walls import build_elevation

        vol, img = _ring()
        spacing = (3.0, 1.0, 1.0)
        res = settle_with_walls(
            vol, 7, wall_kind="membrane", channel_roi=img,
            prep_chain="bgsub", spacing=spacing, band_um=1.5)
        want, _c, _r = build_elevation(
            "membrane", apply_preproc(img, "bgsub"), vol == 7,
            spacing=spacing)
        assert np.array_equal(res.elevation, want)
        ref = settle_label(vol, want, 7, band_um=1.5, spacing=spacing)
        assert res.status == ref.status == "ok", res.messages
        assert np.array_equal(res.result_roi, ref.result_roi)
        assert res.n_modified == ref.n_modified
        assert res.border_background_fraction == ref.border_background_fraction

    def test_it_names_its_stages_in_order(self):
        from packages.epicycle_annotator.reshape.settle import settle_with_walls

        vol, img = _ring()
        calls = []
        settle_with_walls(vol, 7, channel_roi=img,
                          progress=lambda *a: calls.append(a))
        assert calls == [("elevation", 0, 0), ("settle", 0, 0)]

    def test_the_shape_and_probability_kinds_need_no_channel(self):
        from packages.epicycle_annotator.reshape.settle import settle_with_walls

        vol, _img = _ring()
        a = settle_with_walls(vol, 7, wall_kind="distance")
        assert a.elevation is not None and a.elevation.shape == vol.shape
        b = settle_with_walls(vol, 7, wall_kind="probability",
                              prob_roi=(vol == 7).astype(np.float32))
        assert b.elevation is not None

    def test_a_missing_input_raises_rather_than_guessing(self):
        from packages.epicycle_annotator.reshape.settle import settle_with_walls

        vol, _img = _ring()
        with pytest.raises(ValueError):
            settle_with_walls(vol, 7, wall_kind="membrane")
        with pytest.raises(ValueError):
            settle_with_walls(vol, 7, wall_kind="probability")

    def test_a_bare_settle_label_carries_no_picture(self):
        from packages.epicycle_annotator.reshape.settle import settle_label

        vol, _img = _ring()
        res = settle_label(vol, np.zeros(vol.shape, np.float32), 7)
        assert res.elevation is None


# --------------------------------------------------------------------------- #
# the orchestrator's launch: crops and decisions only
# --------------------------------------------------------------------------- #
def _K_bind(f, name):
    from tests.annotator.test_refine_lock_lifecycle import _K

    return getattr(_K(), name).__get__(f, type(f))


def _fake():
    from tests.annotator.test_refine_lock_lifecycle import Fake

    vol, _img = _ring()
    f = Fake(vol)
    for name in ("_launch_refine_method", "_launch_refine_settle",
                 "_launch_refine_contour", "_refuse_refine_step",
                 "_on_refine_progress", "_on_refine_finished"):
        setattr(f, name, _K_bind(f, name))
    f._refine_running = True
    f._result_token = lambda: 1
    return f, vol


class TestTheLaunchHandsOverCrops:
    def _launch(self, kind="membrane", prob=True, channel=True):
        f, vol = _fake()
        launched = []
        f._launch_refine_worker = lambda *a, **k: launched.append((a, k))
        f._get_refine_channel_roi = (
            (lambda roi: np.ones(vol.shape)) if channel else (lambda roi: None))
        f._get_prob_roi = (
            (lambda roi: np.ones(vol.shape)) if prob else (lambda roi: None))
        refused = []
        f._refuse_refine_step = refused.append
        params = {"method": "settle", "wall_kind": kind, "prep_chain": "none",
                  "band_um": 2.0}
        f._launch_refine_method("settle", params, f._smooth_roi, vol, 7)
        return f, launched, refused

    def test_the_worker_gets_settle_with_walls_and_the_channel_crop(self):
        from packages.epicycle_annotator.reshape.settle import settle_with_walls

        f, launched, refused = self._launch()
        assert not refused
        (args, opts), = launched
        fn, kw, method = args[0], args[1], args[2]
        assert fn is settle_with_walls and method == "settle"
        assert kw["channel_roi"] is not None and kw["prob_roi"] is None
        assert "elevation" not in kw, "a picture was built on the main thread"
        assert opts["wants_progress"] is True
        assert f._refine_pending_elevation is None

    def test_the_labels_are_a_copy_never_a_view(self):
        f, launched, _r = self._launch()
        kw = launched[0][0][1]
        shared = f.sync_manager.shared_labels["labels"]
        assert not np.shares_memory(kw["labels_roi"], shared)

    def test_a_shape_wall_needs_no_channel(self):
        f, launched, refused = self._launch(kind="skeleton", channel=False)
        assert not refused
        assert launched[0][0][1]["channel_roi"] is None

    def test_the_probability_wall_hands_over_the_probability_crop(self):
        f, launched, _r = self._launch(kind="probability", channel=False)
        kw = launched[0][0][1]
        assert kw["prob_roi"] is not None and kw["channel_roi"] is None

    def test_no_probability_map_refuses_before_the_worker(self):
        f, launched, refused = self._launch(kind="probability", prob=False)
        assert not launched
        assert "probability map" in refused[0]

    def test_a_channel_gone_from_the_viewer_refuses_before_the_worker(self):
        f, launched, refused = self._launch(channel=False)
        assert not launched
        assert "not in the viewer any more" in refused[0]


class _Outcome:
    def __init__(self, result, kind="settle"):
        self.result = result
        self.kind = kind


class TestThePictureArrivesWithTheAnswer:
    def test_it_is_parked_where_the_contour_parks_its_own(self):
        from packages.epicycle_annotator.reshape.settle import settle_with_walls

        f, vol = _fake()
        f._refine_launch_roi = f._smooth_roi
        f._refine_launch_label = 7
        seen = {}
        f._finish_settle_result = lambda res, lid, roi: seen.setdefault(
            "pending", f._refine_pending_elevation)
        _v, img = _ring()
        res = settle_with_walls(np.array(vol), 7, channel_roi=img)
        f._on_refine_finished(_Outcome(res))
        assert seen["pending"][0] is res.elevation
        assert seen["pending"][1] == tuple(f._smooth_roi)


class TestTheStageLine:
    def test_a_settle_stage_is_the_stage_alone(self):
        f, _v = _fake()
        f._on_refine_progress("elevation", 0, 0)
        assert f.last_ui("stage_text") == "Building the wall picture..."
        f._on_refine_progress("settle", 0, 0)
        assert f.last_ui("stage_text") == (
            "Settling the outline onto the wall...")

    def test_a_plane_count_is_the_count_alone(self):
        f, _v = _fake()
        f._on_refine_progress("contour", 1, 11)
        assert f.last_ui("stage_text") == "Plane 1 of 11"


# --------------------------------------------------------------------------- #
# a REAL QThread: the picture is built OFF the main thread
# --------------------------------------------------------------------------- #
class TestOnARealThread:
    def _annotator(self, qapp, vol):
        from tests.annotator.test_refine_worker import (
            TestOnARealThread as _W,
        )

        a = _W()._annotator(qapp, vol)
        for name in ("_launch_refine_method", "_launch_refine_settle",
                     "_launch_refine_contour", "_refuse_refine_step",
                     "_push_still_staged", "_refine_still_staged_text",
                     "_finish_settle_result", "_refine_band_um"):
            setattr(a, name, _K_bind(a, name))
        _v, img = _ring()
        a._get_refine_channel_roi = lambda roi: img[roi]
        a._get_prob_roi = lambda roi: None
        a._refine_pending_elevation = None
        a._refine_staged_s5 = None
        return a

    def _pump(self, qapp, pred, seconds=30.0):
        import time

        t0 = time.time()
        while time.time() - t0 < seconds:
            qapp.processEvents()
            if pred():
                return True
            time.sleep(0.005)
        return False

    def test_the_picture_is_built_off_the_main_thread_and_stages(
            self, qapp, monkeypatch):
        from packages.epicycle_annotator.reshape import walls

        vol, _img = _ring()
        a = self._annotator(qapp, vol)
        threads = []
        real = walls.build_elevation

        def _spy(*args, **kw):
            threads.append(threading.get_ident())
            return real(*args, **kw)

        monkeypatch.setattr(walls, "build_elevation", _spy)
        params = {"method": "settle", "wall_kind": "membrane",
                  "prep_chain": "none", "band_um": 1.5}
        main = threading.get_ident()
        a._launch_refine_method("settle", params, a._smooth_roi,
                                vol.copy(), 7)
        # the strip went busy at once, with a stage and no drawing-off tail
        first = next(kw for kw in a.ui if kw.get("busy") is True)
        assert first["stage_text"] == "Building the wall picture..."
        assert self._pump(qapp, lambda: not a._refine_running)
        assert threads and all(t != main for t in threads), (
            "the wall picture was built on the MAIN thread")
        assert a._refine_thread is None
        assert a._refine_staged_method == "settle"
        assert a._refine_elevation is not None, "the walls tick has nothing"
        assert a._refine_elevation.shape == vol.shape
        stages = [kw.get("stage_text") for kw in a.ui if kw.get("stage_text")]
        assert not any("Drawing is off" in t for t in stages), stages
