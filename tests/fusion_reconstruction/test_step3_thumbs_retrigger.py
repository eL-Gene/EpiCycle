"""A1 follow-up: the thumbnail batch is re-triggered where this run's
labels_3d is written or deleted, so the outlines appear / go without a
folder load -- and the audit folds around it (teardown gate, no flicker, a
raw fallback never cached as outlined, a written file always re-triggers).
"""
from __future__ import annotations

import os
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")
tifffile = pytest.importorskip("tifffile")
np = pytest.importorskip("numpy")

from napari_epicycle.fusion_reconstruction import _paths  # noqa: E402
from napari_epicycle.fusion_reconstruction import controller as C  # noqa: E402

CTL = C.FusionReconstructionController


@pytest.fixture(autouse=True)
def _qt(monkeypatch):
    """A QApplication (a modal built without one aborts the process) and no
    real message boxes from the failure branches."""
    from qtpy.QtWidgets import QApplication, QMessageBox
    app = QApplication.instance() or QApplication([])
    for name in ("warning", "critical", "information", "question"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(lambda *a, **k: 0))
    return app


class Bare:
    """A controller stand-in: any attribute not set is a MagicMock, so a real
    handler runs its whole body; the re-trigger is a recorder."""

    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.triggers = []
        self._refresh_step3_thumbs_after_labels = self.triggers.append

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        m = MagicMock(name=name)
        self.__dict__[name] = m
        return m


# -- item 7: behavioural, one per site --------------------------------------

def test_a_reconstructed_patch_retriggers():
    b = Bare(_serial_recon_launch_epoch=None, _run_epoch=0)
    result = {"run_id": "r", "result": types.SimpleNamespace(
        stats={"num_cells_final": 3})}
    CTL._on_serial_recon_patch_done(b, "p.tif", result)
    assert b.triggers == ["reconstructed"]


def test_a_finished_reconstruction_retriggers():
    b = Bare(_serial_recon_launch_epoch=None, _run_epoch=0,
             _pipeline_patch_count=2, _pipeline_errors=0,
             _pipeline_start_time=0.0)
    CTL._on_serial_recon_complete(b)
    assert b.triggers == ["reconstruction complete"]


def test_reset_to_predicted_retriggers():
    sm = MagicMock()
    sm.get_patch_status.return_value = "reconstructed"
    b = Bare(state_manager=sm, _edited_patches=set())
    assert CTL._reset_patches_to_predicted(b, ["p.tif"]) == 1
    assert b.triggers == ["reset to predicted"]


def test_nothing_to_reset_does_not_retrigger():
    sm = MagicMock()
    sm.get_patch_status.return_value = "predicted"
    b = Bare(state_manager=sm)
    assert CTL._reset_patches_to_predicted(b, ["p.tif"]) == 0
    assert b.triggers == []


def test_forget_retriggers():
    b = Bare(working_dir="", loaded_stack_dir="")
    b.ui = types.SimpleNamespace()
    CTL._forget_patch_rows(b, "p.tif")
    assert b.triggers == ["forget"]


def _save_ctl():
    return Bare(_run_epoch=0, _save_curation_token=1,
                _qc_view_patch_name="p.tif")


def test_a_save_that_wrote_retriggers_even_when_superseded():
    ok = {"ok": True, "warnings": []}
    for token, patch in ((1, "p.tif"), (0, "p.tif"), (1, "other.tif")):
        b = _save_ctl()
        CTL._on_save_curation_finished(b, ok, token, patch)
        assert b.triggers == ["curation saved"], (token, patch)


def test_a_save_that_failed_does_not_retrigger():
    b = _save_ctl()
    CTL._on_save_curation_finished(b, {"ok": False, "stage": "csv",
                                       "error": "x"}, 1, "p.tif")
    assert b.triggers == []


def test_a_commit_triage_that_wrote_retriggers_even_when_superseded():
    ok = {"ok": True, "triage_map": {}, "n_labels": 1, "labels_path": "l.tif"}

    meta = {"garbage_ids": set(), "labels_path": "l.tif"}

    def make():
        return Bare(_run_epoch=0, _commit_triage_token=1,
                    _qc_view_patch_name="p.tif", _qc_features_df=[1])

    for token, patch in ((1, "p.tif"), (0, "p.tif"), (1, "other.tif")):
        b = make()
        CTL._on_commit_triage_finished(b, ok, token, patch, meta)
        assert b.triggers == ["triage committed"], (token, patch)
    b = make()
    CTL._on_commit_triage_finished(b, {"ok": False, "stage": "csv"}, 1,
                                   "p.tif", meta)
    assert b.triggers == []


# -- the helper and the teardown gate (item 2) ------------------------------

def test_the_helper_calls_start_and_never_raises():
    calls = []

    def boom(why, **kw):
        calls.append((why, kw))
        raise RuntimeError("x")

    ns = types.SimpleNamespace(_start_step3_thumbs=boom)
    CTL._refresh_step3_thumbs_after_labels(ns, "why")
    assert calls == [("why", {"keep_previous": True})]


def _ns(tmp_path, statuses, run_id=None):
    class _SM:
        suppress_save = False

        def get_patch_status(self, n):
            return statuses[n]

        def get_all_patch_statuses(self):
            return dict(statuses)

        def get_active_run_id(self):
            return run_id

    captured = []
    ui = types.SimpleNamespace(
        set_patch_thumbs=lambda m: captured.append(m), _patch_thumbs={})
    ns = types.SimpleNamespace(
        working_dir=str(tmp_path), loaded_stack_dir=str(tmp_path),
        state_manager=_SM(), ui=ui,
        _listed_patch_names=lambda: list(statuses),
        _s3_thumb_axes_for=lambda n: "ZYX",
        _S3_THUMB_LABEL_STATUSES=CTL._S3_THUMB_LABEL_STATUSES)
    ns._s3_thumb_names = lambda: CTL._s3_thumb_names(ns)
    ns._s3_thumb_labels_for = lambda n: CTL._s3_thumb_labels_for(ns, n)
    ns._s3_thumb_cache_dir = lambda: tmp_path / "cache"
    ns._s3_thumb_stamp = CTL._s3_thumb_stamp
    ns._start_step3_thumbs = (lambda why, keep_previous=False:
                              CTL._start_step3_thumbs(
                                  ns, why, keep_previous=keep_previous))
    ns.captured = captured
    return ns


def _patch_file(tmp_path, name="p.tif"):
    tifffile.imwrite(str(tmp_path / name), np.zeros((4, 8, 8), np.uint16))


def _labels_file(tmp_path, name="p.tif", run_id=None, ext=".tif"):
    stem = _paths.patch_labels_stem(str(tmp_path), name, run_id=run_id)
    stem.parent.mkdir(parents=True, exist_ok=True)
    f = stem.with_name(stem.name + ext)
    if ext == ".zarr":
        f.mkdir()
    else:
        tifffile.imwrite(str(f), np.ones((4, 8, 8), np.uint16))
    return f


@pytest.fixture
def spy(monkeypatch):
    from napari_epicycle.fusion_reconstruction import _step3_thumbs as S3
    seen, started = [], []
    real = S3.patch_thumb_requests

    def req_spy(*a, **k):
        out = real(*a, **k)
        seen.append({r.name: r.labels_path for r in out})
        return out

    monkeypatch.setattr(S3, "patch_thumb_requests", req_spy)
    monkeypatch.setattr(S3, "start_step3_thumbs",
                        lambda *a, **k: started.append(list(a[1])))
    return seen, started


def test_after_reconstruct_the_request_carries_labels_and_after_forget_not(
        tmp_path, spy):
    seen, _started = spy
    _patch_file(tmp_path)
    statuses = {"p.tif": "predicted"}
    ns = _ns(tmp_path, statuses)
    CTL._start_step3_thumbs(ns, "folder load")
    assert seen[-1] == {"p.tif": ""}
    _labels_file(tmp_path)
    statuses["p.tif"] = "reconstructed"
    CTL._refresh_step3_thumbs_after_labels(ns, "reconstructed")
    assert seen[-1]["p.tif"].endswith("labels_3d.tif")
    os.remove(tmp_path / "p.tif")
    CTL._refresh_step3_thumbs_after_labels(ns, "forget")
    assert "p.tif" not in seen[-1]


def test_no_batch_starts_after_shutdown(tmp_path, spy):
    _seen, started = spy
    _patch_file(tmp_path)
    ns = _ns(tmp_path, {"p.tif": "predicted"})
    ns._shutdown_done = True
    CTL._refresh_step3_thumbs_after_labels(ns, "late")
    CTL._start_step3_thumbs(ns, "late")
    assert started == [] and ns.captured == []


# -- a named run is really seen ----------------------------------------------

def test_labels_for_names_the_run_labels_of_a_reconstructed_patch(tmp_path):
    ns = _ns(tmp_path, {"a.tif": "reconstructed", "b.tif": "predicted",
                        "c.tif": "finalized", "d.tif": "curating"})
    fa = _labels_file(tmp_path, "a.tif")
    _labels_file(tmp_path, "b.tif")
    fc = _labels_file(tmp_path, "c.tif", ext=".zarr")
    f = CTL._s3_thumb_labels_for
    assert Path(f(ns, "a.tif")) == fa
    assert f(ns, "b.tif") is None
    assert Path(f(ns, "c.tif")) == fc
    assert f(ns, "d.tif") is None


def test_labels_for_is_per_run(tmp_path):
    ns = _ns(tmp_path, {"a.tif": "reconstructed"}, run_id="other")
    rid = C._active_run_path_id(ns)
    assert rid not in (None, "")                  # the named run is seen
    mine = _labels_file(tmp_path, "a.tif", run_id=rid)
    default = _labels_file(tmp_path, "a.tif", run_id=None)
    got = CTL._s3_thumb_labels_for(ns, "a.tif")
    assert Path(got) == mine and Path(got) != default
    mine.unlink()                  # only the default run's file is left
    assert CTL._s3_thumb_labels_for(ns, "a.tif") is None


def test_the_stamp_includes_the_labels_mtime():
    from napari_epicycle.ui_shell._stack_thumbs import ThumbRequest
    a = ThumbRequest("p", "x/p.tif", "ZYX", 1, 2)
    b = a._replace(labels_path="l", labels_mtime_ns=77)
    st = CTL._s3_thumb_stamp
    assert st(a) != st(b) and st(b)[3] == 77 and st(a)[3] == 0


def test_the_pixmap_cache_keys_on_the_labels_mtime():
    from napari_epicycle.ui_shell._stack_thumbs import ThumbCache
    k = ThumbCache.key
    assert k("p", 1, 2) == k("p", 1, 2, 0) and len(k("p", 1, 2)) == 3
    assert k("p", 1, 2, 5) != k("p", 1, 2)


# -- item 4: no flicker ------------------------------------------------------

def test_a_retrigger_keeps_the_old_pixmap_until_the_new_one_lands(
        tmp_path, spy):
    _patch_file(tmp_path)
    ns = _ns(tmp_path, {"p.tif": "predicted"})
    old = object()
    ns.ui._patch_thumbs = {"p.tif": (old, "tip")}
    _labels_file(tmp_path)
    ns.state_manager.get_patch_status = lambda n: "reconstructed"
    CTL._refresh_step3_thumbs_after_labels(ns, "reconstructed")
    pm, _tip = ns.captured[-1]["p.tif"]
    assert pm is old                      # no flash of the placeholder


def test_a_folder_load_does_not_keep_an_older_folders_pixmap(tmp_path, spy):
    _patch_file(tmp_path)
    ns = _ns(tmp_path, {"p.tif": "predicted"})
    ns.ui._patch_thumbs = {"p.tif": (object(), "tip")}
    CTL._start_step3_thumbs(ns, "folder load")
    assert ns.captured[-1]["p.tif"][0] is None


# -- item 3: a raw fallback is never cached as the outlined picture ----------

def test_a_fallback_picture_is_cached_under_the_raw_key(tmp_path):
    from qtpy.QtCore import QBuffer, QByteArray, QIODevice
    from qtpy.QtGui import QColor, QImage
    from qtpy.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from napari_epicycle.ui_shell._stack_thumbs import ThumbCache
    img = QImage(8, 8, QImage.Format_RGB888)
    img.fill(QColor("red"))
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.WriteOnly)
    img.save(buf, "PNG")
    buf.close()
    data = bytes(ba)

    shown = []
    p = str(tmp_path / "p.tif")
    ns = types.SimpleNamespace(
        _s3_thumb_token=1,
        _s3_thumb_requested={"p.tif": (os.path.normcase(os.path.abspath(p)),
                                       5, 6, 99)},
        _s3_thumb_cache=ThumbCache(), _s3_thumb_failed={},
        ui=types.SimpleNamespace(
            set_patch_thumb=lambda n, pm, tip: shown.append((n, pm))))
    ns._on_thumb_failed = lambda *a: shown.append(("failed",))
    # the worker's PNG name has no ".l": it fell back to the raw picture
    CTL._on_thumb_ready(ns, 1, "p.tif", p,
                        str(tmp_path / "p.tif.ab.5.6.c2i.png"), data)
    assert shown and shown[0][0] == "p.tif"
    assert ns._s3_thumb_cache.get(p, 5, 6, 99) is None   # not the outlined key
    assert ns._s3_thumb_cache.get(p, 5, 6) is not None   # the raw key
    # an outlined picture is cached under the outlined key -- the name
    # tail carries the labels stamp AND the overlay token (A3)
    CTL._on_thumb_ready(ns, 1, "p.tif", p,
                        str(tmp_path / "p.tif.ab.5.6.c2i.l99.ofo45.png"),
                        data)
    assert ns._s3_thumb_cache.get(p, 5, 6, 99) is not None
