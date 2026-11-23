"""The Patches rows' eye, the row menu and the prediction window's wiring.

* The eye sits in the slot the ⓘ had, on every row that is not missing:
  predicted -> the prediction window; reconstructed / curating / accepted
  / finalized -> Session's eye (the same method, the same gate as View);
  unprocessed -> dimmed, a click says "Nothing to show yet — predict it
  first."; missing -> no eye, its ⋯ menu.  The tooltip names the action.
* Its gate is View's: live (at the row's own dim) while Predict or
  Reconstruct runs, dimmed and refused with the lock sentence under every
  other lock flag -- through its own list property.  The dim MULTIPLIES
  the row's opacity and gives it back.
* The menu opens on every present row: "Show prediction…" (a prediction on
  disk) and "Model details…" (a made-by record; never greyed); a missing
  row keeps Forget; a row with neither shows no menu.
* The window: one per patch (a second open raises it); refused for the
  patch being predicted (or waiting in the running Predict); its reads on
  real loader threads kept in ``_pred_threads`` until finished; a late
  answer is dropped; it closes on a folder change, a run switch / workspace
  load, a step exit, shutdown, a Forget of that patch and right after the
  re-Predict confirm of that patch (before its cache is cleared).
* ``_pred_launch_set`` holds only the patches of the running Predict not
  yet done.

Run one file at a time with ``-o addopts=""``.
"""
from __future__ import annotations

import ast
import inspect
import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")

from napari_epicycle.fusion_reconstruction import _model_provenance as MP  # noqa: E402
from napari_epicycle.fusion_reconstruction import _paths  # noqa: E402
from napari_epicycle.fusion_reconstruction import _prediction_preview as PP  # noqa: E402
from napari_epicycle.fusion_reconstruction import _step3_manifests as S  # noqa: E402
from napari_epicycle.fusion_reconstruction import _step3_pr_text as T  # noqa: E402
from napari_epicycle.fusion_reconstruction import _step3_rows as R  # noqa: E402
from tests.fusion_reconstruction._one_channel import write_zyx_patch  # noqa: E402


@pytest.fixture(scope="module")
def app():
    from qtpy.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _pump(app, n=3):
    from qtpy.QtCore import QCoreApplication, QEvent
    for _ in range(n):
        app.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def _wait(app, cond, timeout=15.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.005)
    return cond()


STATUSES = {"a.tif": "unprocessed", "b.tif": "predicted",
            "c.tif": "predicted", "e.tif": "reconstructed",
            "one_ch.tif": "predicted"}


def _write_cache(c, name, shape=(4, 8, 8), codes=(1, 0), manifest=True):
    cache = c._prediction_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    stem = Path(name).stem
    rng = np.random.default_rng(len(name))
    for v in PP.PASSES:
        np.save(cache / f"{stem}_{v}_probs.npy",
                rng.normal(0, 4, size=shape).astype(np.float16))
        np.save(cache / f"{stem}_{v}_flows.npy",
                rng.normal(0, 1, size=(2,) + shape).astype(np.float16))
    if manifest:
        S.write_json_atomic(S.prediction_manifest_path(cache, name), {
            "manifest_version": 1, "patch_name": name, "axes": "ZCYX",
            "predicted_at": "2026-10-26T21:34:00+00:00",
            "seg_config": {"channels": list(codes)},
            MP.PROVENANCE_KEY: MP.provenance_block(
                "cyto3", "cyto3", now="2026-10-26T21:00:00+00:00")})


@pytest.fixture
def full(app, tmp_path):
    import tifffile

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
    for name in STATUSES:
        if name == "one_ch.tif":
            write_zyx_patch(folder)
        else:
            tifffile.imwrite(str(folder / name),
                             np.zeros((4, 2, 8, 8), np.uint16),
                             metadata={"axes": "ZCYX"})
    c.loaded_stack_dir = str(folder)
    c.state_manager._state["patches"] = {
        n: {"status": s} for n, s in STATUSES.items()}
    w.lateral_model_path.setText("cyto3")
    w.axial_model_path.setText("cyto3")
    _write_cache(c, "b.tif")
    _write_cache(c, "c.tif", codes=(2, 1))
    _write_cache(c, "e.tif")
    _write_cache(c, "one_ch.tif", codes=(2, 1))
    c._recompute_missing_patches()
    c._refresh_patch_state_list()
    yield w, c, folder
    for f in ("_prediction_running", "_reconstruction_running",
              "_curation_busy"):
        setattr(c, f, False)
    try:
        c._close_prediction_windows()
        c._close_provenance_windows()
    except Exception:
        pass
    _wait(app, lambda: not c._pred_threads, 10)
    timer = getattr(c.state_manager, "_save_timer", None)
    if timer is not None:
        timer.stop()
    w.deleteLater()
    _pump(app)


def _rec(w, name):
    from qtpy.QtCore import Qt

    from napari_epicycle.ui_shell._stack_row_delegate import ROLE_RECORD
    lst = w._patch_state_list
    for i in range(lst.count()):
        if lst.item(i).data(Qt.UserRole) == name:
            return lst.item(i).data(ROLE_RECORD)
    raise KeyError(name)


def _open_window(app, c, name):
    c._on_show_prediction(name)
    dlg = c._pred_windows[name][0]
    assert _wait(app, lambda: dlg.last_right is not None), \
        dlg.right_box.text()
    return dlg


# ------------------------------------------------------------ the records

@pytest.mark.parametrize("status,tip,dim", [
    ("unprocessed", T.S3_PR_EYE_NOTHING, True),
    ("predicted", T.S3_PR_EYE_PREDICTION_TIP, False),
    ("reconstructed", T.S3_PR_EYE_VIEW_TIP, False),
    ("curating", T.S3_PR_EYE_VIEW_TIP, False),
    ("accepted", T.S3_PR_EYE_VIEW_TIP, False),
    ("finalized", T.S3_PR_EYE_VIEW_TIP, False)])
def test_every_present_row_carries_the_eye_with_its_words(status, tip, dim):
    rec = R.build_row("p.tif", status, "CZYX")
    assert rec["eye"] is True and rec["eye_tip"] == tip
    assert bool(rec["eye_dim"]) is dim


def test_a_missing_row_has_no_eye_but_its_menu():
    rec = R.build_row("p.tif", "predicted", "CZYX", missing=True,
                      made_by="made by x")
    assert rec["eye"] is False and rec["menu"] is True
    assert "eye_tip" not in rec


def test_the_list_paints_the_eye_in_the_info_slot(full):
    from qtpy.QtCore import QRect

    from napari_epicycle.fusion_reconstruction.ui import (
        PR_EYE_LOCK_PROPERTY, PR_LOCK_PROPERTY,
    )
    w, c, folder = full
    d = w._patch_state_list.itemDelegate()
    assert d._show_eye is True and d._show_info is False
    assert d._eye_lock_property == PR_EYE_LOCK_PROPERTY
    assert d._eye_lock_property != PR_LOCK_PROPERTY
    for name in STATUSES:
        g = d.layout(QRect(0, 0, 224, 0), _rec(w, name), w._patch_state_list.font())
        assert "eye" in g and "info" not in g and "menu" not in g, name
        assert g["axis"].right() < g["eye"].left()
    os.remove(folder / "b.tif")
    c._recompute_missing_patches()
    c._refresh_patch_state_list()
    g = d.layout(QRect(0, 0, 224, 0), _rec(w, "b.tif"),
                 w._patch_state_list.font())
    assert "menu" in g and "eye" not in g


def test_the_eye_tooltip_is_the_rows_own(full):
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QStyleOptionViewItem
    w, c, _f = full
    lst = w._patch_state_list
    d = lst.itemDelegate()
    for name, want in (("a.tif", T.S3_PR_EYE_NOTHING),
                       ("b.tif", T.S3_PR_EYE_PREDICTION_TIP),
                       ("e.tif", T.S3_PR_EYE_VIEW_TIP)):
        idx = next(lst.model().index(i, 0) for i in range(lst.count())
                   if lst.item(i).data(Qt.UserRole) == name)
        opt = QStyleOptionViewItem()
        rec = idx.data(Qt.UserRole + 2)
        from qtpy.QtCore import QRect
        g = d.layout(QRect(0, 0, 260, 0), rec, lst.font())
        opt.rect = QRect(0, 0, 260, g["height"])
        opt.font = lst.font()
        assert d.tip_text(opt, idx, g["eye"].center()) == want


def test_other_lists_lay_out_exactly_as_before(app):
    """A record without the eye keys and one with ``"eye": True`` lay out
    the same in every painter mode; Step 4's narrow row keeps its eye."""
    from qtpy.QtCore import QRect
    from qtpy.QtGui import QFont

    from napari_epicycle.ui_shell import _stack_row_delegate as D
    base = {"name": "s.tif", "status": "predicted", "axes": "ZYX",
            "word": "predicted", "shape_text": "4×8×8"}
    for mode in (D.MODE_NARROW, D.MODE_WIDE, D.MODE_MEASURE):
        for kw in ({}, {"show_eye": False}, {"show_info": True}):
            d = D.StackRowDelegate(mode, **kw)
            a = d.layout(QRect(0, 0, 260, 0), dict(base), QFont())
            b = d.layout(QRect(0, 0, 260, 0), dict(base, eye=True,
                                                   eye_tip="x"), QFont())
            assert {k: v for k, v in a.items()} == \
                {k: v for k, v in b.items()}, (mode, kw)
    g = D.StackRowDelegate(D.MODE_NARROW).layout(
        QRect(0, 0, 260, 0), dict(base), QFont())
    assert "eye" in g


class _Painter:
    """Records the opacity the eye is drawn at and what it leaves."""

    def __init__(self, opacity):
        self._o = opacity
        self.drawn_at = []

    def opacity(self):
        return self._o

    def setOpacity(self, v):
        self._o = v

    def drawPixmap(self, *a):
        self.drawn_at.append(self._o)


@pytest.mark.parametrize("row,dim,eye_locked,want", [
    (0.55, False, False, 0.55),            # the Predict lock: the row's dim
    (0.55, True, False, 0.55 * 0.35),      # unprocessed during Predict
    (1.0, False, True, 0.35),              # another lock
    (1.0, True, False, 0.35)])             # unprocessed
def test_the_eyes_dim_multiplies_and_gives_the_opacity_back(
        app, row, dim, eye_locked, want):
    from qtpy.QtCore import QRect
    from qtpy.QtWidgets import QListWidget

    from napari_epicycle.ui_shell import _stack_row_delegate as D
    view = QListWidget()
    view.setProperty("eyeLock", eye_locked)
    d = D.StackRowDelegate(D.MODE_NARROW, eye_lock_property="eyeLock")
    p = _Painter(row)
    d._paint_eye(p, QRect(0, 0, 16, 16), {"eye_dim": dim}, view)
    assert p.drawn_at and abs(p.drawn_at[0] - want) < 1e-9
    assert p.opacity() == row                 # the rest keeps the row's
    view.deleteLater()


def test_no_icon_is_asked_of_qtawesome_without_an_application(monkeypatch):
    """qtawesome builds its icon fonts once, on the first icon asked for;
    built with no application they hold nothing and are never rebuilt, so
    one icon asked for by a Qt-free caller broke every later icon of the
    process (the eye above drew nothing after such a test).  The helpers
    answer None instead and leave qtawesome alone."""
    import qtawesome as qta

    from napari_epicycle.ui_shell import _stack_row_delegate as D
    from napari_epicycle.ui_shell import theme as TH

    class _NoApp:
        @staticmethod
        def instance():
            return None

    asked = []
    monkeypatch.setattr(TH, "QApplication", _NoApp)
    monkeypatch.setattr(qta, "icon", lambda *a, **k: asked.append(a))
    assert TH._qta_icon_or_none("mdi6.eye-outline", "#888") is None
    assert D._icon_pixmap("mdi6.eye-outline", "#888") is None
    assert asked == []


# ------------------------------------------------------------ the gate

def test_the_eye_lock_follows_views_rule(full):
    from napari_epicycle.fusion_reconstruction.ui import PR_EYE_LOCK_PROPERTY
    w, c, _f = full
    lst = w._patch_state_list
    c._refresh_pr_lock()
    assert not lst.property(PR_EYE_LOCK_PROPERTY)
    c._prediction_running = True
    c._refresh_pr_lock()
    assert not lst.property(PR_EYE_LOCK_PROPERTY)     # live, row-dimmed
    c._prediction_running = False
    c._curation_busy = True
    c._refresh_pr_lock()
    assert lst.property(PR_EYE_LOCK_PROPERTY)
    c._curation_busy = False
    c._refresh_pr_lock()
    assert not lst.property(PR_EYE_LOCK_PROPERTY)


# ------------------------------------------------------------ the actions

def test_each_status_does_its_own_thing(full, app, monkeypatch):
    w, c, _f = full
    viewed = []
    monkeypatch.setattr(c, "_view_patch", viewed.append)
    c._on_pr_eye("e.tif")
    assert viewed == ["e.tif"]                  # through Session's eye
    c._on_pr_eye("a.tif")
    assert c._pr_job_line == T.S3_PR_EYE_NOTHING
    assert "a.tif" not in c._pred_windows
    c._on_pr_eye("b.tif")
    assert "b.tif" in c._pred_windows and viewed == ["e.tif"]


def test_a_reconstructed_eye_goes_through_session_eyes_gate(full, monkeypatch):
    w, c, _f = full
    seen = []
    monkeypatch.setattr(c, "_on_session_eye", seen.append)
    c._on_pr_eye("e.tif")
    assert seen == ["e.tif"]


def test_the_window_opens_with_the_model_input_and_xy_probability(full, app):
    w, c, _f = full
    dlg = _open_window(app, c, "c.tif")
    assert dlg.windowTitle() == "Prediction — c.tif"
    assert dlg.parent() is w
    # the run as the run combo names it, not its folder name
    assert dlg.line1_label.text().startswith("Default · cyto3 (built-in)")
    assert dlg.line2_label.text() == \
        "channel to segment 2: green · helper 1: red"
    assert dlg.left_caption.text() == "2: green (segment) + 1: red (helper)"
    assert dlg.current_choice() == ("xy", "probs")
    assert dlg.details_btn.isVisibleTo(dlg)


def test_the_window_names_the_runs_display_name(full, app):
    """The provenance's first line names the run as the run combo does
    ("Default", "Nuclei Fixed"), not its folder name."""
    w, c, _f = full
    c._refresh_run_picker()
    dlg = _open_window(app, c, "b.tif")
    assert _wait(app, lambda: dlg.line1_label.text().startswith(
        "Default · ")), dlg.line1_label.text()


def test_a_one_channel_patch_reads_grey(full, app):
    w, c, _f = full
    dlg = _open_window(app, c, "one_ch.tif")
    assert dlg.left_caption.text() == T.PRED_LEFT_ONE_CHANNEL
    left = dlg.last_left
    assert np.array_equal(left[..., 0], left[..., 1])


def test_one_window_per_patch_a_second_open_raises_it(full, app):
    w, c, _f = full
    dlg = _open_window(app, c, "b.tif")
    n_threads_before = len(c._pred_threads)
    c._on_show_prediction("b.tif")
    assert c._pred_windows["b.tif"][0] is dlg
    assert len(c._pred_threads) == n_threads_before    # nothing re-read
    dlg.close()
    _pump(app)
    again = _open_window(app, c, "b.tif")
    assert again is not dlg


def test_a_choice_reads_only_that_file(full, app, monkeypatch):
    w, c, _f = full
    dlg = _open_window(app, c, "b.tif")
    seen = []
    real = PP.load_map

    def spy(cache_dir, stem, pass_, kind):
        seen.append((stem, pass_, kind))
        return real(cache_dir, stem, pass_, kind)

    monkeypatch.setattr(PP, "load_map", spy)
    monkeypatch.setattr(PP, "load_image",
                        lambda *a, **k: pytest.fail("the image read again"))
    dlg.pass_combo.setCurrentIndex(1)
    assert _wait(app, lambda: dlg._map is not None)
    dlg.map_combo.setCurrentIndex(1)
    assert _wait(app, lambda: dlg._map is not None
                 and dlg._map["kind"] == "flows")
    assert seen == [("b", "xz", "probs"), ("b", "xz", "flows")]


def test_a_newer_choice_wins(full, app):
    w, c, _f = full
    dlg = _open_window(app, c, "b.tif")
    dlg.pass_combo.setCurrentIndex(1)
    dlg.pass_combo.setCurrentIndex(2)
    assert _wait(app, lambda: not c._pred_threads)
    _pump(app)
    assert dlg._map is not None and dlg._map["pass"] == "yz"


@pytest.mark.parametrize("flag", ["_curation_busy", "_patch_load_in_flight"])
def test_refused_with_the_lock_sentence_under_another_lock(full, flag):
    w, c, _f = full
    setattr(c, flag, True)
    try:
        c._on_pr_eye("b.tif")
        assert "b.tif" not in c._pred_windows
        assert c._pr_job_line == dict(T.S3_LOCK_SENTENCES)[flag]
    finally:
        setattr(c, flag, False)


def test_a_finished_patch_opens_while_predict_runs(full, app):
    w, c, _f = full
    c._prediction_running = True
    c._predicting_patches.add("c.tif")
    c._pred_launch_set = {"c.tif"}
    try:
        _open_window(app, c, "b.tif")              # finished: opens
        c._on_show_prediction("c.tif")             # being predicted
        assert "c.tif" not in c._pred_windows
        assert c._pr_job_line == T.S3_PRED_REFUSED_RUNNING.format(
            patch="c.tif")
    finally:
        c._prediction_running = False
        c._predicting_patches.discard("c.tif")
        c._pred_launch_set = set()


def test_a_prediction_gone_from_disk_is_said(full):
    w, c, _f = full
    for f in c._prediction_cache_dir().glob("b_*"):
        f.unlink()
    c._on_show_prediction("b.tif")
    assert "b.tif" not in c._pred_windows
    assert c._pr_job_line == T.S3_PRED_GONE.format(patch="b.tif")


def test_a_too_large_patch_is_refused_in_the_window(full, app, monkeypatch):
    w, c, _f = full
    monkeypatch.setattr(PP, "PRED_MAX_VOXELS", 10)
    c._on_show_prediction("b.tif")
    dlg = c._pred_windows["b.tif"][0]
    want = PP.too_large_sentence((4, 8, 8))
    assert "4 × 8 × 8" in want
    assert _wait(app, lambda: dlg.right_box.text() == want)
    assert dlg.left_box.text() == want


def test_the_eye_of_a_patch_being_predicted_says_so(full):
    w, c, _f = full
    for busy in ("_predicting_patches", "_pred_launch_set"):
        getattr(c, busy).add("a.tif")
        try:
            c._on_pr_eye("a.tif")
            assert c._pr_job_line == T.S3_PRED_REFUSED_RUNNING.format(
                patch="a.tif"), busy
        finally:
            getattr(c, busy).discard("a.tif")
    c._on_pr_eye("a.tif")
    assert c._pr_job_line == T.S3_PR_EYE_NOTHING


@pytest.mark.parametrize("status,want_tip", [
    ("unprocessed", "busy"), ("predicted", "busy"),
    ("curating", T.S3_PR_EYE_VIEW_TIP),
    ("reconstructed", T.S3_PR_EYE_VIEW_TIP)])
def test_a_busy_rows_eye_tip_follows_the_click(status, want_tip):
    """A patch in the running Predict (being predicted or still queued):
    its eye's tooltip is the click's refusal, not "Nothing to show yet";
    a patch with labels still views them (the click's own order)."""
    rec = R.build_row("p.tif", status, "CZYX", busy=True)
    want = (T.S3_PRED_REFUSED_RUNNING.format(patch="p.tif")
            if want_tip == "busy" else want_tip)
    assert rec["eye_tip"] == want
    assert bool(rec["eye_dim"]) is (want_tip == "busy")
    assert R.build_row("p.tif", status, "CZYX")["eye_tip"] !=         T.S3_PRED_REFUSED_RUNNING.format(patch="p.tif")


def test_the_running_and_the_queued_rows_say_being_predicted(full):
    """The list's records read BOTH sets: the patch being predicted
    (``_predicting_patches``) and the ones still queued in the running
    Predict (``_pred_launch_set``)."""
    w, c, _f = full
    c._predicting_patches.add("a.tif")
    c._pred_launch_set = {"a.tif", "c.tif"}
    try:
        c._refresh_patch_state_list()
        for name in ("a.tif", "c.tif"):
            assert _rec(w, name)["eye_tip"] ==                 T.S3_PRED_REFUSED_RUNNING.format(patch=name), name
        assert _rec(w, "b.tif")["eye_tip"] == T.S3_PR_EYE_PREDICTION_TIP
    finally:
        c._predicting_patches.discard("a.tif")
        c._pred_launch_set = set()
    c._refresh_patch_state_list()
    assert _rec(w, "a.tif")["eye_tip"] == T.S3_PR_EYE_NOTHING


@pytest.mark.parametrize("how", ["signal", "flag"])
def test_after_cancel_the_cut_and_unreached_rows_are_live(full, how):
    """Cancel: ``all_completed`` (and the rows' rebuild) comes BEFORE the
    thread's cleanup.  The patch the run cut and the one it never reached
    are not "being predicted" any more: their eyes say what they do."""
    w, c, _f = full
    c._prediction_running = True
    c._predicting_patches.add("a.tif")
    c._pred_launch_set = {"a.tif", "c.tif"}
    try:
        c._refresh_patch_state_list()
        assert _rec(w, "c.tif")["eye_dim"]
        c._pipeline_start_time = time.monotonic()
        c._pipeline_errors = 0
        if how == "signal":
            c._on_serial_pred_cancelled(0, 2)
        else:                      # the flag alone: the top of complete
            c._pipeline_cancelled = (0, 2)
        c._on_serial_pred_complete()
        assert c._pred_launch_set == set()
        a, cc = _rec(w, "a.tif"), _rec(w, "c.tif")
        assert a["eye_tip"] == T.S3_PR_EYE_NOTHING
        assert cc["eye_tip"] == T.S3_PR_EYE_PREDICTION_TIP
        assert not cc["eye_dim"]
    finally:
        c._prediction_running = False
        c._predicting_patches.clear()
        c._pred_launch_set = set()


def test_the_refusal_shows_under_the_list_while_the_foot_runs(full, app):
    """While Predict runs, its progress line owns the foot: the eye's
    refusal goes on the line under the list (the one that says "You can
    still click rows…"), for that click; the job's line is left alone."""
    w, c, _f = full
    job = "Segmenting patch 1/1: a.tif — XZ (11/256)"
    c._prediction_running = True
    c._predicting_patches.add("a.tif")
    c._pred_launch_set = {"a.tif"}
    w.set_pr_surface_locked(True)
    try:
        c._pr_status(job)
        assert w.pr_locked_list_note.text() == T.S3_LOCK_LIST_NOTE
        c._on_pr_eye("a.tif")
        assert w.pr_locked_list_note.text() ==             T.S3_PRED_REFUSED_RUNNING.format(patch="a.tif")
        assert c._pr_job_line == job
        assert not w.pr_locked_list_note.isHidden()
        # the next click that is not refused gives the line back
        _open_window(app, c, "b.tif")
        assert w.pr_locked_list_note.text() == T.S3_LOCK_LIST_NOTE
        c._on_show_prediction("a.tif")         # the menu's door too
        assert w.pr_locked_list_note.text() ==             T.S3_PRED_REFUSED_RUNNING.format(patch="a.tif")
        # the next patch's start gives the line back
        c._pr_status(job)
        c._pipeline_start_time = time.monotonic()
        c._pipeline_slices_done, c._pipeline_total_slices = 0, 10
        c._on_serial_pred_started("c.tif", 1, 2)
        assert w.pr_locked_list_note.text() == T.S3_LOCK_LIST_NOTE
        c._on_pr_eye("a.tif")
    finally:
        c._prediction_running = False
        c._predicting_patches.clear()
        c._pred_launch_set = set()
        w.set_pr_surface_locked(False)
    assert w.pr_locked_list_note.text() == T.S3_LOCK_LIST_NOTE


def test_one_loader_in_flight_per_window(full, app, monkeypatch):
    """Quick Pass / Map changes read one file at a time; only the newest
    waiting choice is read next."""
    w, c, _f = full
    dlg = _open_window(app, c, "b.tif")
    assert _wait(app, lambda: not c._pred_threads)
    gate = threading.Event()
    real = PP.load_map
    seen, most = [], [0]

    def slow(cache_dir, stem, pass_, kind):
        seen.append((pass_, kind))
        gate.wait(10)
        return real(cache_dir, stem, pass_, kind)

    monkeypatch.setattr(PP, "load_map", slow)
    real_start = c._start_prediction_load

    def counting(link, what, fn, args):
        real_start(link, what, fn, args)
        most[0] = max(most[0], len(c._pred_threads))

    monkeypatch.setattr(c, "_start_prediction_load", counting)
    dlg.pass_combo.setCurrentIndex(1)              # XZ: read starts
    dlg.pass_combo.setCurrentIndex(2)              # YZ: waits
    dlg.map_combo.setCurrentIndex(1)               # YZ flows: replaces it
    assert len(c._pred_threads) == 1
    gate.set()
    assert _wait(app, lambda: dlg._map is not None
                 and dlg._map["kind"] == "flows" and not c._pred_threads)
    assert seen == [("xz", "probs"), ("yz", "flows")]
    assert most[0] == 1
    assert dlg._map["pass"] == "yz"


def test_a_thread_still_running_after_finished_is_parked(full, monkeypatch):
    import napari_epicycle.fusion_reconstruction.controller as C
    w, c, _f = full

    class Thread:
        def isRunning(self):
            return True

        def wait(self, ms):
            return False

    link = C._PredictionLink(c, "b.tif")
    load = C._PredictionLoad(c, link, "map", 0, 0)
    load.thread, load.worker = Thread(), object()
    c._pred_threads.add(load)
    c._on_prediction_thread_finished(load)
    assert (load.thread, load.worker) in C._LIVE_STEP3_THREADS
    assert load not in c._pred_threads
    C._LIVE_STEP3_THREADS.discard((load.thread, load.worker))


def test_shutdown_with_a_live_loader_runs_no_handler_after(full, app,
                                                          monkeypatch):
    """``shutdown()`` with a loader mid-read: the loader's answer, sent
    while the sweep joins it, reaches no handler."""
    w, c, _f = full
    gate = threading.Event()
    real = PP.load_open

    def slow(*a, **k):
        gate.wait(10)
        return real(*a, **k)

    monkeypatch.setattr(PP, "load_open", slow)
    c._on_show_prediction("b.tif")
    load = next(iter(c._pred_threads))
    thread = load.thread
    called = []
    monkeypatch.setattr(c, "_on_prediction_preview_loaded",
                        lambda *a: called.append("loaded"))
    monkeypatch.setattr(c, "_on_prediction_preview_failed",
                        lambda *a: called.append("failed"))
    monkeypatch.setattr(c, "_on_prediction_thread_finished",
                        lambda *a: called.append("finished"))
    threading.Timer(0.3, gate.set).start()       # it answers mid-join
    c.shutdown()
    assert thread.wait(10000)
    _pump(app, 10)
    assert called == []
    assert c._pred_threads == set()


# ------------------------------------------------------------ the menu

class _Menu:
    shown = []
    pick = None

    def __init__(self, *a):
        self.acts = []

    def addAction(self, text):
        from types import SimpleNamespace
        act = SimpleNamespace(text=text, enabled=True)
        act.setEnabled = lambda v: setattr(act, "enabled", v)
        self.acts.append(act)
        return act

    def exec_(self, pos):
        _Menu.shown.append([(a.text, a.enabled) for a in self.acts])
        for a in self.acts:
            if a.text == _Menu.pick:
                return a
        return None

    def deleteLater(self):
        pass


@pytest.fixture
def menu(monkeypatch):
    import napari_epicycle.fusion_reconstruction.controller as C
    _Menu.shown, _Menu.pick = [], None
    monkeypatch.setattr(C, "QMenu", _Menu)
    return _Menu


def test_the_menu_opens_on_every_present_row(full, menu):
    from qtpy.QtCore import QPoint
    w, c, folder = full
    show, det = T.S3_SHOW_PREDICTION_MENU, T.S3_MODEL_DETAILS
    for name, want in (("b.tif", [(show, True), (det, True)]),
                       ("e.tif", [(show, True), (det, True)])):
        menu.shown.clear()
        c._pr_row_menu(name, QPoint(0, 0))
        assert menu.shown == [want], name
    menu.shown.clear()
    c._pr_row_menu("a.tif", QPoint(0, 0))          # nothing to offer
    assert menu.shown == []
    os.remove(folder / "c.tif")
    c._recompute_missing_patches()
    c._pr_row_menu("c.tif", QPoint(0, 0))
    assert menu.shown == [[(T.S3_FORGET_MENU, True)]]


def test_model_details_is_never_greyed_and_opens_the_details(full, menu, app):
    from qtpy.QtCore import QPoint
    w, c, _f = full
    c._curation_busy = True
    try:
        menu.pick = T.S3_MODEL_DETAILS
        c._pr_row_menu("b.tif", QPoint(0, 0))
        assert menu.shown[-1][1] == (T.S3_MODEL_DETAILS, True)
        assert "b.tif" in c._prov_windows
    finally:
        c._curation_busy = False


def test_show_prediction_from_the_menu(full, menu, app):
    from qtpy.QtCore import QPoint
    w, c, _f = full
    menu.pick = T.S3_SHOW_PREDICTION_MENU
    c._pr_row_menu("e.tif", QPoint(0, 0))          # a reconstructed row too
    assert "e.tif" in c._pred_windows
    menu.pick = T.S3_SHOW_PREDICTION_MENU
    c._on_curation_list_menu  # the same function serves Curation's list
    assert "_pr_row_menu(" in inspect.getsource(
        type(c)._on_curation_list_menu)


def test_the_window_foot_opens_the_same_details_window(full, app):
    w, c, _f = full
    dlg = _open_window(app, c, "b.tif")
    dlg.details_btn.click()
    first = c._prov_windows["b.tif"][0]
    dlg.details_btn.click()
    assert c._prov_windows["b.tif"][0] is first and len(c._prov_windows) == 1


def test_the_chip_says_where_the_details_are():
    assert T.S3_ROW_HELP_LINES["row_made_by"].endswith(
        "Right-click for the model details.")
    assert "right-click" in T.S3_LIST_HELP


# ------------------------------------------------------------ lifecycle

@pytest.mark.parametrize("how", ["folder", "workspace_or_run", "exit",
                                 "shutdown", "forget", "repredict"])
def test_the_windows_close_on_change(full, app, how, monkeypatch):
    import napari_epicycle.fusion_reconstruction.controller as C
    from napari_epicycle.fusion_reconstruction.controller import (
        _prov_window_alive,
    )
    w, c, folder = full
    dlg = _open_window(app, c, "b.tif")
    order = []
    if how == "folder":
        c._load_stack_dir(str(folder), persist=False)
    elif how == "workspace_or_run":
        c._reset_workspace_scoped_caches()
    elif how == "exit":
        c.on_step_exit()
    elif how == "shutdown":
        c.shutdown()
    elif how == "forget":
        os.remove(folder / "b.tif")
        c._recompute_missing_patches()
        monkeypatch.setattr(C, "themed_confirm", lambda *a, **k: True)
        real = C._s3forget.move_to_trash

        def mover(*a, **k):
            order.append(("move", "b.tif" in c._pred_windows))
            return real(*a, **k)

        monkeypatch.setattr(C._s3forget, "move_to_trash", mover)
        assert c._forget_patch("b.tif")
        assert order == [("move", False)]
    else:
        monkeypatch.setattr(C, "themed_confirm", lambda *a, **k: True)
        monkeypatch.setattr(C, "_SerialPredictionWorker", _FakeWorker)
        monkeypatch.setattr(C, "QThread", _FakeThread)
        real_clear = c._clear_prediction_cache

        def clear(name):
            order.append(("clear", name, name in c._pred_windows))
            real_clear(name)

        monkeypatch.setattr(c, "_clear_prediction_cache", clear)
        c._pr_set_item_ticked(_item(w, "b.tif"), True)
        c._predict_all()
        c._prediction_running = False
        assert ("clear", "b.tif", False) in order
    _pump(app, 5)
    assert "b.tif" not in c._pred_windows
    assert not _prov_window_alive(dlg, visible=False)


def test_a_cache_file_that_cannot_be_removed_is_said(full, monkeypatch):
    w, c, _f = full
    logged = []
    monkeypatch.setattr(c, "_append_log", logged.append)
    real_unlink = Path.unlink

    def unlink(self, *a, **k):
        if self.name == "b_xy_probs.npy":
            raise PermissionError(32, "in use")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(Path, "unlink", unlink)
    c._clear_prediction_cache("b.tif")
    assert any(line.startswith("[PREDICT] Could not remove b_xy_probs.npy")
               for line in logged)


@pytest.mark.parametrize("which", ["session", "step4"])
@pytest.mark.parametrize("eye_locked", [False, True])
def test_the_shared_eye_paints_as_before_on_an_enabled_list(
        app, which, eye_locked):
    """Session's (wide) and Step 4's (narrow) eye under an enabled list are
    pixel-identical to the old absolute 0.35 / 1.0 painting."""
    from qtpy.QtCore import QRect, Qt
    from qtpy.QtGui import QImage, QPainter
    from qtpy.QtWidgets import QListWidget, QListWidgetItem, \
        QStyleOptionViewItem

    from napari_epicycle.ui_shell import _stack_row_delegate as D

    class Old(D.StackRowDelegate):
        def _paint_eye(self, painter, r, rec, view):
            off = bool(view is not None
                       and view.property(self._eye_lock_property))
            if off:
                painter.setOpacity(0.35)
            self._paint_icon(painter, r, "mdi6.eye-outline")
            if off:
                painter.setOpacity(1.0)

        def _paint_menu(self, painter, r, locked):
            if locked:
                painter.setOpacity(0.35)
            self._paint_icon(painter, r, "mdi6.dots-horizontal")
            if locked:
                painter.setOpacity(1.0)

    mode = D.MODE_WIDE if which == "session" else D.MODE_NARROW
    lst = QListWidget()
    lst.setProperty("eyeLock", eye_locked)
    rec = {"name": "s.tif", "status": "reconstructed", "axes": "ZCYX",
           "word": "reconstructed", "shape_text": "4×8×8", "line2": "x"}
    item = QListWidgetItem("s.tif")
    item.setData(D.ROLE_RECORD, rec)
    lst.addItem(item)
    idx = lst.model().index(0, 0)
    pics = []
    for cls in (D.StackRowDelegate, Old):
        d = cls(mode, lst, eye_lock_property="eyeLock")
        opt = QStyleOptionViewItem()
        opt.rect = QRect(0, 0, 320, 120)
        opt.font = lst.font()
        opt.state = opt.state | D.QStyle.State_Enabled
        opt.widget = lst
        img = QImage(320, 120, QImage.Format.Format_ARGB32)
        img.fill(Qt.GlobalColor.black)
        painter = QPainter(img)
        d.paint(painter, opt, idx)
        painter.end()
        pics.append(img)
    assert pics[0] == pics[1]
    lst.deleteLater()


def test_the_closers_are_in_the_source():
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController as C,
    )
    for fn in (C._reset_workspace_scoped_caches, C._load_stack_dir,
               C.on_step_exit, C.shutdown):
        assert "_close_prediction_windows()" in inspect.getsource(fn), fn
    src = inspect.getsource(C._predict_all)
    assert src.index("_close_prediction_windows(non_unprocessed)") < \
        src.index("_clear_prediction_cache(name)")
    assert "_sweep_prediction_threads()" in inspect.getsource(C.shutdown)
    assert "_close_prediction_windows([name])" in inspect.getsource(
        C._forget_patch)


def test_a_late_answer_is_dropped(full, app, monkeypatch):
    w, c, _f = full
    gate = threading.Event()
    real = PP.load_open

    def slow(*a, **k):
        gate.wait(10)
        return real(*a, **k)

    monkeypatch.setattr(PP, "load_open", slow)
    c._on_show_prediction("b.tif")
    dlg = c._pred_windows["b.tif"][0]
    assert len(c._pred_threads) == 1
    c._reset_workspace_scoped_caches()              # a run switch's reset
    gate.set()
    assert _wait(app, lambda: not c._pred_threads)
    assert dlg.last_right is None


def test_the_files_can_be_replaced_and_moved_with_a_loader_between_reads(
        full, app, monkeypatch):
    """No file stays open between the loader's steps, and a closer never
    waits: with a loader stopped between its header read and its full
    read, the re-Predict's unlink and np.save and Forget's move succeed."""
    w, c, _f = full
    gate = threading.Event()
    reached = threading.Event()
    real = PP._load_full

    def paused(path):
        reached.set()
        gate.wait(10)
        return real(path)

    monkeypatch.setattr(PP, "_load_full", paused)
    c._on_show_prediction("b.tif")
    assert reached.wait(10)
    t0 = time.monotonic()
    c._close_prediction_windows(["b.tif"])
    assert time.monotonic() - t0 < 1.0              # never waits
    cache = c._prediction_cache_dir()
    probs = cache / "b_xy_probs.npy"
    probs.unlink()
    np.save(probs, np.zeros((4, 8, 8), np.float16))
    trash = cache / "trash"
    trash.mkdir()
    os.replace(cache / "b_xy_flows.npy", trash / "b_xy_flows.npy")
    gate.set()
    assert _wait(app, lambda: not c._pred_threads)


def test_shutdown_joins_and_parks_a_loader(full, app, monkeypatch):
    import napari_epicycle.fusion_reconstruction.controller as C
    w, c, _f = full
    gate = threading.Event()
    real = PP.load_open

    def slow(*a, **k):
        gate.wait(10)
        return real(*a, **k)

    monkeypatch.setattr(PP, "load_open", slow)
    c._on_show_prediction("b.tif")
    load = next(iter(c._pred_threads))
    thread = load.thread
    waited = []

    def wait(ms=None):
        waited.append(ms)
        return False          # still running after the join

    thread.wait = wait
    before = set(C._LIVE_STEP3_THREADS)
    try:
        c._sweep_prediction_threads()
    finally:
        del thread.wait
    assert waited == [5000]
    parked = C._LIVE_STEP3_THREADS - before
    assert (thread, load.worker) in parked
    assert c._pred_threads == set()
    gate.set()
    assert thread.wait(10000)
    C._LIVE_STEP3_THREADS.discard((thread, load.worker))


def test_no_lambda_in_the_new_wiring():
    src = (Path(__file__).resolve().parents[2] / "napari_epicycle" /
           "fusion_reconstruction" / "controller.py").read_text("utf-8")
    tree = ast.parse(src)
    wanted = {"_on_pr_eye", "_on_show_prediction", "_on_prediction_choice",
              "_start_prediction_load", "_on_prediction_preview_loaded",
              "_on_prediction_preview_failed",
              "_on_prediction_thread_finished", "_close_prediction_windows",
              "_live_prediction_windows", "_sweep_prediction_threads",
              "_prediction_refusal", "_pr_row_menu", "_pr_has_made_by"}
    seen = set()
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name in wanted:
            seen.add(fn.name)
            assert not any(isinstance(n, ast.Lambda) for n in ast.walk(fn)), \
                fn.name
            for n in ast.walk(fn):
                if isinstance(n, ast.Call) and getattr(n.func, "id", "") \
                        == "partial":
                    pytest.fail(fn.name)
    assert seen == wanted


# ------------------------------------------------------------ _pred_launch_set

class _Sig:
    def connect(self, *a, **k):
        pass


class _FakeThread:
    def __init__(self, *a, **k):
        self.started = self.finished = _Sig()

    def start(self):
        pass

    def quit(self):
        pass

    def isRunning(self):
        return False


class _FakeWorker:
    made = []

    def __init__(self, fn, configs):
        self.patch_started = self.patch_completed = _Sig()
        self.view_progress = self.all_completed = self.error = _Sig()
        self.cancelled = _Sig()
        self.configs = configs
        _FakeWorker.made.append(self)

    def moveToThread(self, t):
        pass

    def run(self):
        pass


def _item(w, name):
    from qtpy.QtCore import Qt
    lst = w._patch_state_list
    return next(lst.item(i) for i in range(lst.count())
                if lst.item(i).data(Qt.UserRole) == name)


def test_the_launch_set_holds_only_what_is_not_done(full, monkeypatch):
    import napari_epicycle.fusion_reconstruction.controller as C
    w, c, _f = full
    monkeypatch.setattr(C, "_SerialPredictionWorker", _FakeWorker)
    monkeypatch.setattr(C, "QThread", _FakeThread)
    monkeypatch.setattr(C, "themed_confirm", lambda *a, **k: True)
    for n in ("a.tif", "b.tif", "c.tif"):
        c._pr_set_item_ticked(_item(w, n), True)
    c._predict_all()
    assert c._serial_prediction_worker is _FakeWorker.made[-1]
    assert c._pred_launch_set == {"a.tif", "b.tif", "c.tif"}
    c._on_serial_pred_patch_done("a.tif", {})
    assert c._pred_launch_set == {"b.tif", "c.tif"}
    c._pipeline_errors = 0
    c._on_serial_pred_error("b.tif", "boom")
    assert c._pred_launch_set == {"c.tif"}
    c._cleanup_serial_prediction_thread()
    assert c._pred_launch_set == set()


def test_a_failed_launch_clears_the_launch_set(full, monkeypatch):
    import napari_epicycle.fusion_reconstruction.controller as C
    w, c, _f = full

    class Boom(_FakeThread):
        def start(self):
            raise RuntimeError("no thread")

    monkeypatch.setattr(C, "_SerialPredictionWorker", _FakeWorker)
    monkeypatch.setattr(C, "QThread", Boom)
    c._pr_set_item_ticked(_item(w, "a.tif"), True)
    c._predict_all()
    assert c._pred_launch_set == set()
    assert c._prediction_running is False
