"""Forget a patch whose file is gone (brick V1).

The ACTIVE run's Step 3 artifacts of the patch are MOVED (never deleted) to
``artifacts/step3/.trash/{patch}-{stamp}/``, each at its path relative to
``artifacts/step3/``, with ``forgotten.json`` written FIRST and updated per
item; a failure part of the way stops the move and leaves a complete list;
the durable trees (``snapshots/``, ``training_archive/``) and other patches'
files stay; the record and its tick leave the run; refused under any lock,
with unsaved curation, or while the file is there; the patch in view is
reset in the run-switch order.  Run one file at a time with
``-o addopts=""``.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")

from napari_epicycle.fusion_reconstruction import _paths  # noqa: E402
from napari_epicycle.fusion_reconstruction import _qc_extraction_context as QX  # noqa: E402
from napari_epicycle.fusion_reconstruction import _step3_forget as F  # noqa: E402
from napari_epicycle.fusion_reconstruction import _step3_pr_text as P  # noqa: E402

PATCH = "a.tif"


def _touch(p: Path, text="x"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def _plant(ws: Path, run_id=None, patch=PATCH):
    """Every kind of artifact of *patch* in run *run_id*, plus a neighbour
    whose name starts with the same stem and the durable trees."""
    stem = Path(patch).stem
    run_dir = _paths.patch_run_dir(ws, patch, run_id)
    _touch(run_dir / "labels_3d.tif")
    _touch(run_dir / "manifest.json", "{}")
    _touch(_paths.qc_features_csv(ws, stem, run_id))
    _touch(_paths.qc_adjacency_raw_csv(ws, stem, run_id))
    _touch(QX._sidecar_path(_paths.qc_features_csv(ws, stem, run_id)), "{}")
    _touch(_paths.patch_curation_dir(ws, stem, run_id) / "tiers.csv")
    cache = _paths.prediction_cache_dir(ws, run_id)
    for v in ("xy", "xz", "yz"):
        _touch(cache / f"{stem}_{v}_flows.npy")
    _touch(cache / f"{stem}_prediction_manifest.json", "{}")
    _touch(cache / f"{stem}_extra_xy_flows.npy")          # ANOTHER patch
    _touch(_paths.patch_pairs_dir(ws, stem, run_id) / "p.tif")
    _touch(_paths.snapshots_dir(ws) / f"{stem}_snap.tif")
    _touch(_paths.patch_training_archive_dir(ws, patch) / "meta.json")


# ------------------------------------------------------------ the pure parts

def test_the_items_are_what_is_on_disk_by_exact_name(tmp_path):
    _plant(tmp_path)
    items = F.forget_items(tmp_path, PATCH)
    paths = {str(p) for _k, p in items}
    assert not any("a_extra_xy" in p for p in paths)       # the neighbour
    assert not any("snap" in p or "training_archive" in p for p in paths)
    assert F.kinds_present(items) == list(F.KINDS)
    cache = [p for k, p in items if k == "predictions"]
    assert len(cache) == 4                                 # 3 arrays + meta
    # nothing on disk: nothing listed
    assert F.forget_items(tmp_path, "zz.tif") == []


def test_the_per_patch_extraction_context_goes_with_the_features(tmp_path):
    """B MED-2 / : the per-patch QC sidecar
    ``<stem>_extraction_context.json`` is one of the patch's artifacts."""
    _plant(tmp_path)
    sidecar = QX._sidecar_path(_paths.qc_features_csv(tmp_path, "a", None))
    assert sidecar.name == "a_extraction_context.json"
    assert ("features", sidecar) in F.forget_items(tmp_path, PATCH)
    _plant(tmp_path, run_id="cells")
    named = QX._sidecar_path(_paths.qc_features_csv(tmp_path, "a", "cells"))
    assert ("features", named) in F.forget_items(tmp_path, PATCH, "cells")


def test_the_move_keeps_the_layout_and_writes_the_log_first(tmp_path):
    _plant(tmp_path)
    items = F.forget_items(tmp_path, PATCH)
    seen_log_before_move = []

    def mover(src, dst):
        logs = list(F.trash_root(tmp_path).glob("*/" + F.LOG_NAME))
        seen_log_before_move.append(bool(logs))
        os.replace(src, dst)

    res = F.move_to_trash(tmp_path, PATCH, None, items, {"status": "x"},
                          now=datetime(2026, 10, 7, 9, 30, 0), mover=mover)
    assert res.ok and len(res.moved) == len(items)
    assert all(seen_log_before_move)
    root = _paths.step3_root(tmp_path)
    assert res.trash == root / ".trash" / "a.tif-20261007-093000"
    # each item at its path relative to artifacts/step3/
    assert (res.trash / "runs" / PATCH / "labels_3d.tif").exists()
    assert not (_paths.patch_run_dir(tmp_path, PATCH)).exists()
    log = json.loads((res.trash / F.LOG_NAME).read_text(encoding="utf-8"))
    assert log["patch"] == PATCH and log["record"] == {"status": "x"}
    assert all(e["moved"] for e in log["items"])
    # the durable trees and the neighbour stay
    assert (_paths.snapshots_dir(tmp_path) / "a_snap.tif").exists()
    assert (_paths.patch_training_archive_dir(tmp_path, PATCH)
            / "meta.json").exists()
    assert (_paths.prediction_cache_dir(tmp_path)
            / "a_extra_xy_flows.npy").exists()


def test_a_second_forget_the_same_second_gets_a_suffix(tmp_path):
    now = datetime(2026, 10, 7, 9, 30, 0)
    first = F.new_trash_dir(tmp_path, PATCH, now)
    first.mkdir(parents=True)
    assert F.new_trash_dir(tmp_path, PATCH, now).name == \
        "a.tif-20261007-093000-2"


def test_a_failure_stops_the_move_and_the_log_says_how_far(tmp_path):
    _plant(tmp_path)
    items = F.forget_items(tmp_path, PATCH)
    n = {"k": 0}

    def mover(src, dst):
        n["k"] += 1
        if n["k"] == 2:
            raise OSError("locked by another program")
        os.replace(src, dst)

    res = F.move_to_trash(tmp_path, PATCH, None, items, mover=mover)
    assert not res.ok and len(res.moved) == 1
    assert "locked by another program" in res.error
    log = json.loads((res.trash / F.LOG_NAME).read_text(encoding="utf-8"))
    assert [e["moved"] for e in log["items"]][:2] == [True, False]
    assert "stopped" in log
    assert Path(items[1][1]).exists()                     # still in place


def test_a_named_runs_items_and_the_trash_beside_runs(tmp_path):
    _plant(tmp_path, run_id="cells")
    items = F.forget_items(tmp_path, PATCH, "cells")
    assert items and all("cells" in str(p) for _k, p in items
                         if _k != "predictions" or "cells" in str(p))
    res = F.move_to_trash(tmp_path, PATCH, "cells", items)
    assert res.ok
    assert (res.trash / "runs" / "cells" / "reconstructions" / PATCH
            / "manifest.json").exists()
    # .trash is NOT under runs/, which run discovery and the layout
    # conversion read
    assert _paths.runs_dir(tmp_path) not in res.trash.parents
    assert F.trash_root(tmp_path).parent == _paths.step3_root(tmp_path)


# ------------------------------------------------------------ the controller

@pytest.fixture(scope="module")
def app():
    from qtpy.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def full(app, tmp_path, monkeypatch):
    import numpy as np
    import tifffile

    import napari_epicycle.fusion_reconstruction.controller as C
    from napari_epicycle.fusion_reconstruction.ui import (
        FusionReconstructionCuration,
    )
    w = FusionReconstructionCuration(viewer=None)
    c = w.controller
    c.working_dir = str(tmp_path)
    c.state_manager.active_run_id = "default"
    c.state_manager.load(str(tmp_path))
    folder = tmp_path / "patches"
    folder.mkdir()
    tifffile.imwrite(str(folder / "b.tif"), np.zeros((4, 8, 8), np.uint8))
    c.loaded_stack_dir = str(folder)
    c.state_manager._state["patches"] = {
        PATCH: {"status": "reconstructed"}, "b.tif": {"status": "predicted"}}
    c.state_manager._state["ticked"] = [PATCH, "b.tif"]
    _plant(tmp_path)
    c._recompute_missing_patches()
    c._refresh_patch_state_list()
    asked = []

    def confirm(parent, title, msg, **k):
        asked.append((title, msg, k))
        return True

    monkeypatch.setattr(C, "themed_confirm", confirm)
    yield w, c, asked
    timer = getattr(c.state_manager, "_save_timer", None)
    if timer is not None:
        timer.stop()
    w.deleteLater()
    app.processEvents()


def test_forget_moves_drops_the_record_and_the_tick(full, tmp_path):
    w, c, asked = full
    assert PATCH in c._missing_patches
    assert c._forget_patch(PATCH) is True
    (title, msg, kw) = asked[0]
    assert title == P.S3_FORGET_TITLE and kw["accent"] == "red"
    assert kw["confirm_text"] == P.S3_FORGET_CONFIRM
    for kind in F.KINDS:
        assert P.S3_FORGET_WHAT[kind] in msg
    assert "reconstructed" in msg and PATCH in msg
    assert PATCH not in c.state_manager.get_all_patch_statuses()
    assert c.state_manager.get_ticked() == ["b.tif"]
    assert PATCH not in c._missing_patches
    names = [w._patch_state_list.item(i).data(0x0100)
             for i in range(w._patch_state_list.count())]
    assert names == ["b.tif"]
    trashed = list(F.trash_root(tmp_path).glob("a.tif-*"))
    assert len(trashed) == 1 and (trashed[0] / F.LOG_NAME).exists()
    assert w.pipeline_status_label.toolTip().startswith("Forgot a.tif")
    # the next folder load does not bring it back
    c._load_stack_dir(c.loaded_stack_dir)
    assert PATCH not in c.state_manager.get_all_patch_statuses()


def test_forget_is_refused_while_the_file_is_there(full):
    w, c, asked = full
    assert c._forget_patch("b.tif") is False
    assert asked == []
    assert w.pipeline_status_label.toolTip() == P.S3_FORGET_REFUSED_PRESENT


@pytest.mark.parametrize("flag", ["_prediction_running", "_curation_busy",
                                  "_refine_running", "_qc_extract_running"])
def test_forget_is_refused_under_any_lock(full, tmp_path, flag):
    w, c, asked = full
    setattr(c, flag, True)
    try:
        assert c._forget_patch(PATCH) is False
    finally:
        setattr(c, flag, False)
    assert asked == []
    assert w.pipeline_status_label.toolTip() == P.S3_FORGET_REFUSED_LOCKED
    assert _paths.patch_run_dir(tmp_path, PATCH).exists()


def test_unsaved_curation_of_the_patch_in_view_refuses(full, tmp_path,
                                                      monkeypatch):
    w, c, asked = full
    c._qc_view_patch_name = PATCH
    c._curation_dirty = True
    monkeypatch.setattr(c, "_guard_unsaved_changes", lambda: False)
    assert c._forget_patch(PATCH) is False
    assert w.pipeline_status_label.toolTip() == P.S3_FORGET_REFUSED_UNSAVED
    assert _paths.patch_run_dir(tmp_path, PATCH).exists()
    c._curation_dirty = False


def test_the_patch_in_view_is_reset_in_the_run_switch_order(full,
                                                           monkeypatch):
    w, c, asked = full
    order = []
    monkeypatch.setattr(c, "_close_3d_curation_annotator",
                        lambda clear_layers=False: order.append("close"))
    monkeypatch.setattr(c, "_clear_curation_layers",
                        lambda: order.append("layers"))
    monkeypatch.setattr(c, "_reset_qc_plot_area",
                        lambda *a, **k: order.append("plot"))
    c._qc_view_patch_name = PATCH
    c._qc_features_df = object()
    assert c._forget_patch(PATCH) is True
    assert order == ["close", "layers", "plot"]
    assert c._qc_features_df is None and c._qc_view_patch_name == ""


def test_a_cancelled_confirm_moves_nothing(full, tmp_path, monkeypatch):
    import napari_epicycle.fusion_reconstruction.controller as C
    w, c, _asked = full
    monkeypatch.setattr(C, "themed_confirm", lambda *a, **k: False)
    assert c._forget_patch(PATCH) is False
    assert _paths.patch_run_dir(tmp_path, PATCH).exists()
    assert PATCH in c.state_manager.get_all_patch_statuses()


def test_the_menu_offers_forget_only_on_a_missing_row(full, monkeypatch):
    from qtpy.QtCore import QPoint
    w, c, _asked = full
    shown = []

    class _Menu:
        def __init__(self, *a):
            self.acts = []

        def addAction(self, text):
            from types import SimpleNamespace
            act = SimpleNamespace(text=text, enabled=True)
            act.setEnabled = lambda v: setattr(act, "enabled", v)
            self.acts.append(act)
            return act

        def exec_(self, pos):
            shown.append([a.text for a in self.acts])
            return None

        def deleteLater(self):
            pass

    import napari_epicycle.fusion_reconstruction.controller as C
    monkeypatch.setattr(C, "QMenu", _Menu)
    c._pr_row_menu("b.tif", QPoint(0, 0))
    assert shown == []
    c._pr_row_menu(PATCH, QPoint(0, 0))
    assert shown == [[P.S3_FORGET_MENU]]


def test_forget_flushes_the_state_save_at_once(full, monkeypatch):
    """B LOW-1: the dropped record is written now (``flush`` right after
    ``forget_patch``), as Finalize / Reset do -- not 500 ms later, where a
    close in between would bring the patch back."""
    w, c, _asked = full
    order = []
    real_forget = c.state_manager.forget_patch
    real_flush = c.state_manager.flush

    def forget(name):
        order.append(("forget", name))
        return real_forget(name)

    def flush():
        order.append(("flush",))
        return real_flush()

    monkeypatch.setattr(c.state_manager, "forget_patch", forget)
    monkeypatch.setattr(c.state_manager, "flush", flush)
    assert c._forget_patch(PATCH) is True
    assert order[:2] == [("forget", PATCH), ("flush",)]


def test_remove_patch_pngs_takes_only_that_patchs_pictures(tmp_path):
    from napari_epicycle.fusion_reconstruction import _step3_thumbs as T
    d = T.thumb_cache_dir(tmp_path)
    d.mkdir(parents=True)
    mine = d / "a.tif.0123abcd.1.2.png"
    other = d / "b.tif.0123abcd.1.2.png"
    near = d / "a.tiff.0123abcd.1.2.png"
    stray = d / "a.tif.notes.txt"
    for p in (mine, other, near, stray):
        p.write_bytes(b"x")
    assert T.remove_patch_pngs(d, PATCH, workspace=tmp_path) == 1
    assert not mine.exists()
    assert other.exists() and near.exists() and stray.exists()
    # a cache folder outside the workspace is refused
    assert T.remove_patch_pngs(d, "b.tif", workspace=tmp_path / "elsewhere") \
        == 0 and other.exists()


def test_forget_drops_the_session_row_and_the_pictures(full, tmp_path,
                                                     monkeypatch):
    """A LOW-1 / B LOW-3: Session's list loses the row (the file is gone,
    the record too), its record rebuilt; the patch's PNG(s) and its
    in-memory picture and failure entries go; another patch's stay."""
    from qtpy.QtWidgets import QListWidgetItem
    from napari_epicycle.fusion_reconstruction import _step3_thumbs as T
    from napari_epicycle.ui_shell._stack_thumbs import ThumbCache
    w, c, _asked = full
    lst = w.loaded_patch_list
    lst.clear()
    for n in (PATCH, "b.tif"):
        it = QListWidgetItem(n)
        it.setData(0x0100, n)
        lst.addItem(it)
    d = T.thumb_cache_dir(tmp_path)
    d.mkdir(parents=True, exist_ok=True)
    (d / "a.tif.0123abcd.1.2.png").write_bytes(b"x")
    (d / "b.tif.0123abcd.1.2.png").write_bytes(b"x")
    cache = c._s3_thumb_cache = ThumbCache()
    folder = Path(c.loaded_stack_dir)
    ka = cache.key(folder / PATCH, 1, 2)
    kb = cache.key(folder / "b.tif", 1, 2)
    cache._d[ka] = "pm-a"
    cache._d[kb] = "pm-b"
    c._s3_thumb_failed = {PATCH: ("x", "tip"), "b.tif": ("y", "tip")}
    calls = []
    real = c._refresh_patch_list_colors
    monkeypatch.setattr(c, "_refresh_patch_list_colors",
                        lambda: (calls.append(1), real()))
    assert c._forget_patch(PATCH) is True
    names = [lst.item(i).data(0x0100) for i in range(lst.count())]
    assert names == ["b.tif"]
    assert calls
    assert not (d / "a.tif.0123abcd.1.2.png").exists()
    assert (d / "b.tif.0123abcd.1.2.png").exists()
    assert ka not in cache._d and kb in cache._d
    assert list(c._s3_thumb_failed) == ["b.tif"]


def test_remove_patch_pngs_takes_inferno_and_outlined_pngs(tmp_path):
    """A1 audit HIGH: Forget must find the ``c2i`` / ``.l<ns>`` shapes."""
    from napari_epicycle.fusion_reconstruction import _step3_thumbs as T
    from napari_epicycle.ui_shell import _stack_thumbs as ST
    d = T.thumb_cache_dir(tmp_path)
    d.mkdir(parents=True)
    a = ST.thumb_png_path(d, PATCH, "x/" + PATCH, 5, 6, single="inferno")
    b = ST.thumb_png_path(d, PATCH, "x/" + PATCH, 5, 6, single="inferno",
                          labels_mtime_ns=1)
    for p in (a, b):
        p.write_bytes(b"x")
    assert T.remove_patch_pngs(d, PATCH, workspace=tmp_path) == 2
    assert not a.exists() and not b.exists()
