"""The data / lifecycle / UX audit fixes of Batch Refine commits 6-8.

U-H1 (gate 0: a busy open dialog is raised, its targets untouched), D-M1
(the probability map's shape read from its header BEFORE the blocker, the
blocker re-run after the load, the load under the marquee), D-M2 (one
object's crop or pool failure stops that object, not the batch; orphans
on every exit; a 2-D channel refused), U-M1 / U-M2 (a failure before
``start()`` puts the dialog back to idle), D-L1 (``OrphanList``), D-L2 (a
late Cancel still cancels), D-L3 (the voxel count after lower-id-wins),
D-L4 (``themed_notice`` deleted; an idle open dialog raised), D-L5 (no
"Preview on object 1" when nothing is selected), D-L6 (one pass at Run),
D-L7 (an Undo all that undid nothing keeps REVIEW), U-L7 (the poll leaves
a busy dialog's count line alone).

Plain ``QApplication`` (pytest-qt is not installed).
"""
import os
import threading

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from packages.epicycle_annotator import batch_chain as bc  # noqa: E402
from packages.epicycle_annotator import batch_refine as br  # noqa: E402
from packages.epicycle_annotator import refine_recipe as rr  # noqa: E402
from tests.annotator.test_batch_chain_dialog import (  # noqa: E402
    C,
    _Img,
    make_ann,
    make_dialog,
    pump,
    stored,
    vol5,
)
from tests.annotator.test_batch_chain_dispatch import (  # noqa: E402
    dil,
    guided,
    settle,
)


@pytest.fixture(scope="module")
def qapp():
    from qtpy.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def widgets(qapp):
    made = []
    yield made
    for w in made:
        try:
            if hasattr(w, "mark_clean_close"):
                w.mark_clean_close()
            w.close()
            w.deleteLater()
        except RuntimeError:
            pass
    qapp.processEvents()


# ======================================================= U-H1 gate 0


class TestGateZero:
    def _busy(self, ann, how):
        if how == "running":
            ann._batch_running = True
            ann._batch_worker = object()
        else:
            ann._batch_staged = True

    @pytest.mark.parametrize("how", ["running", "staged"])
    def test_a_second_door_raises_and_changes_nothing(self, monkeypatch,
                                                      widgets, how):
        ann = make_ann(monkeypatch, widgets)
        assert ann._open_batch_refine_dialog(
            explicit_targets=[1, 2], default_target="selected") is None
        dlg = ann._batch_refine_dialog
        self._busy(ann, how)
        # the navigator door, while the Add-targeted batch is busy
        why = ann._open_batch_refine_dialog()
        assert why == br.BATCH_ALREADY_OPEN
        assert why != br.BATCH_STILL_STOPPING
        assert ann._batch_refine_dialog is dlg
        assert ann._batch_explicit_targets == [1, 2]
        ann._batch_worker = None

    def test_the_sentence(self):
        assert br.BATCH_ALREADY_OPEN == (
            "Batch Refine is already open — Accept, Reject or Cancel it "
            "there first.")

    def test_the_navigator_door_shows_no_notice(self, monkeypatch, widgets):
        import napari_epicycle.ui_shell.theme as theme

        ann = make_ann(monkeypatch, widgets)
        ann._open_batch_refine_dialog()
        self._busy(ann, "running")
        monkeypatch.setattr(theme, "themed_notice",
                            lambda p, t, m, **k: ann.notices.append(m))
        ann._open_batch_refine_from_navigator()
        assert ann.notices == []
        ann._batch_worker = None

    def test_refine_all_says_it_in_the_add_status(self, monkeypatch,
                                                  widgets):
        ann = make_ann(monkeypatch, widgets)
        ann._open_batch_refine_dialog()
        self._busy(ann, "staged")
        said = []
        ann._push_add_ui = lambda **k: said.append(k.get("status_text"))
        ann._add_piece_ids = [4, 5]
        assert ann._batch_refine_add_pieces() == br.BATCH_ALREADY_OPEN
        assert said == [br.BATCH_ALREADY_OPEN]
        assert ann._batch_explicit_targets is None

    def test_the_stopping_gate_still_names_itself_without_a_dialog(
            self, monkeypatch, widgets):
        ann = make_ann(monkeypatch, widgets)
        ann._batch_worker = object()
        ann._load_locked = True
        assert ann._open_batch_refine_dialog() == br.BATCH_STILL_STOPPING
        ann._batch_worker = None


class TestAnIdleOpenDialog:
    def test_the_navigator_door_raises_it(self, monkeypatch, widgets):
        import napari_epicycle.ui_shell.theme as theme

        ann = make_ann(monkeypatch, widgets)
        ann._open_batch_refine_dialog()
        dlg = ann._batch_refine_dialog
        monkeypatch.setattr(theme, "themed_notice",
                            lambda p, t, m, **k: ann.notices.append(m))
        ann._open_batch_refine_from_navigator()
        assert ann._batch_refine_dialog is dlg
        assert ann.notices == []             # not BATCH_NOT_CLEAN

    def test_refine_all_replaces_it_with_its_targets(self, monkeypatch,
                                                     widgets):
        ann = make_ann(monkeypatch, widgets)
        ann._open_batch_refine_dialog()
        first = ann._batch_refine_dialog
        ann._push_add_ui = lambda **k: None
        ann._add_piece_ids = [1, 2]
        assert ann._batch_refine_add_pieces() is None
        dlg = ann._batch_refine_dialog
        assert dlg is not first
        assert ann._batch_explicit_targets == [1, 2]
        assert dlg._radio_selected.isChecked()
        assert dlg._radio_selected.text().startswith("Added this session")


# ============================================ D-M1 the probability map


def _npy(tmp_path, shape):
    p = tmp_path / "prob.npy"
    np.save(str(p), np.zeros(shape, np.float16))
    return str(p)


class TestProbShapeBeforeTheLoad:
    def test_an_unloaded_mis_sized_map_is_refused_at_once(
            self, monkeypatch, widgets, tmp_path):
        ann = make_ann(monkeypatch, widgets, store={"g": stored(guided())})
        ann._prob_volume_path = _npy(tmp_path, (60, 80, 280))  # 2x labels
        ann._open_batch_refine_dialog()
        dlg = ann._batch_refine_dialog
        assert dlg.select_chain("g")
        assert dlg.reason_text() == rr.BATCH_PROB_SHAPE
        assert not dlg._run_btn.isEnabled()
        ann._run_batch_refine([1, 2])
        assert dlg._status_lbl.text() == rr.BATCH_PROB_SHAPE
        assert ann.locks == [] and ann._batch_worker is None
        assert ann._prob_volume is None          # nothing was loaded

    def test_the_header_read_loads_no_voxel(self, monkeypatch, widgets,
                                            tmp_path):
        ann = make_ann(monkeypatch, widgets)
        ann._prob_volume_path = _npy(tmp_path, (3, 4, 5))
        assert C()._prob_volume_shape(ann) == (3, 4, 5)
        assert ann._prob_volume is None

    def test_the_blocker_runs_again_after_the_load(self, monkeypatch,
                                                   widgets, tmp_path):
        ann = make_ann(monkeypatch, widgets, store={"g": stored(guided())})
        ann._prob_volume_path = _npy(tmp_path, (60, 80, 280))
        # the header cannot be read: only the real load reveals the size
        monkeypatch.setattr(
            C(), "_prob_volume_shape",
            lambda self: (tuple(self._prob_volume.shape)
                          if self._prob_volume is not None else None))
        states = []
        real = C()._ensure_prob_volume

        def load(self):
            states.append(self._batch_refine_dialog._state)
            return real(self)

        monkeypatch.setattr(C(), "_ensure_prob_volume", load)
        ann._open_batch_refine_dialog()
        dlg = ann._batch_refine_dialog
        assert dlg.select_chain("g")
        assert dlg._run_btn.isEnabled()          # not known yet
        ann._run_batch_refine([1, 2])
        # the full load ran under the running state and its marquee
        assert states == ["running"]
        assert dlg._state == "idle"
        assert dlg._status_lbl.text() == rr.BATCH_PROB_SHAPE
        assert ann.locks == [] and ann._batch_worker is None
        assert dlg.reason_text() == rr.BATCH_PROB_SHAPE


# ======================================================= D-M2 failures


class _Raising:
    shape = (30, 40, 140)
    ndim = 3

    def __getitem__(self, key):
        raise IndexError("too many indices for array")


class TestOneObjectFailsAlone:
    def test_a_crop_that_raises_stops_the_object(self):
        v = vol5()
        out = bc.run_chain_on_label(
            v, 1, (settle(),), bbox=br.label_bboxes(v)[1],
            spacing=(1.0, 1.0, 1.0), channel=_Raising())
        assert out.status == "stopped" and out.step == 1
        assert out.reason == bc.BATCH_REFUSED_FAILED and out.mask is None

    def test_a_pool_exception_stops_that_object_only(self, monkeypatch):
        v = vol5()
        real = bc.run_chain_on_label

        def flaky(snapshot, lid, steps, **kw):
            if lid == 2:
                raise RuntimeError("boom")
            return real(snapshot, lid, steps, **kw)

        monkeypatch.setattr(bc, "run_chain_on_label", flaky)
        boxes = br.label_bboxes(v)
        outs, cancelled = bc.run_chain_batch(
            v, [1, 2, 3], boxes, (dil(1),), spacing=(1.0, 1.0, 1.0),
            n_workers=2)
        got = dict(outs)
        assert not cancelled and set(got) == {1, 2, 3}
        assert got[2].status == "stopped"
        assert got[2].reason == bc.BATCH_REFUSED_FAILED
        assert got[1].status == "ok"

    def test_the_whole_batch_is_staged_with_the_failure_listed(
            self, monkeypatch):
        v = vol5()
        real = bc.run_chain_on_label

        def flaky(snapshot, lid, steps, **kw):
            if lid == 2:
                raise RuntimeError("boom")
            return real(snapshot, lid, steps, **kw)

        monkeypatch.setattr(bc, "run_chain_on_label", flaky)
        res = br.run_batch_recipe(v, [1, 2, 3], rr.Recipe(steps=(dil(1),)),
                                  n_workers=2)
        assert 2 in res.stopped and res.stopped[2][0] == 1
        assert sorted(res.touched_ids) == [1, 3]
        assert np.array_equal(res.working_volume == 2, v == 2)

    def test_running_futures_become_orphans_on_an_exception_exit(
            self, monkeypatch):
        v = vol5()
        gate = threading.Event()
        started = threading.Event()

        def slow(snapshot, lid, steps, **kw):
            if lid == 2:
                started.set()                   # really RUNNING, not queued
                gate.wait(5)
            else:
                started.wait(5)
            return bc.ChainOutcome("unchanged")

        monkeypatch.setattr(bc, "run_chain_on_label", slow)
        orphans = bc.OrphanList()

        def tick():
            raise RuntimeError("tick failed")

        try:
            with pytest.raises(RuntimeError):
                bc.run_chain_batch(v, [1, 2], br.label_bboxes(v), (dil(1),),
                                   spacing=(1.0, 1.0, 1.0), n_workers=2,
                                   tick=tick, orphans=orphans)
            assert len(orphans) == 1
        finally:
            gate.set()

    def test_a_2d_channel_is_refused_by_the_blocker(self, monkeypatch,
                                                    widgets):
        layers = [_Img("flat", np.zeros((40, 140)))]
        ann = make_ann(monkeypatch, widgets, layers=layers, refine_ch="flat",
                       store={"s": stored(settle())})
        ann._open_batch_refine_dialog()
        dlg = ann._batch_refine_dialog
        assert dlg.select_chain("s")
        assert dlg.chain_channel() == "flat"
        assert dlg.reason_text() == rr.BATCH_CHANNEL_SHAPE
        assert not dlg._run_btn.isEnabled()


# ================================================= U-M1 / U-M2 launch


class TestLaunchFailures:
    def test_an_exception_before_start_goes_back_to_idle(
            self, monkeypatch, widgets):
        ann = make_ann(monkeypatch, widgets, store={"g": stored(dil(1))})
        ann._open_batch_refine_dialog()
        dlg = ann._batch_refine_dialog
        assert dlg.select_chain("g")

        def boom():
            raise RuntimeError("spacing lookup")

        ann._refine_spacing = boom
        ann._run_batch_refine([1, 2])
        assert dlg._state == "idle"
        assert dlg._status_lbl.text() == br.BATCH_LAUNCH_FAILED
        assert ann.locks == [] and ann._batch_worker is None
        assert dlg._run_btn.isEnabled()

    def test_a_memory_error_in_the_copy_goes_back_to_idle(
            self, monkeypatch, widgets, capsys):
        ann = make_ann(monkeypatch, widgets)
        ann._open_batch_refine_dialog()
        dlg = ann._batch_refine_dialog

        class _Huge:
            shape = (30, 40, 140)

            def copy(self):
                raise MemoryError("Unable to allocate 4.2 GiB")

        ann._launch_batch_worker(dlg, _Huge(), "lbl", [1],
                                 [br.RefineStep("sdf", {})])
        assert dlg._state == "idle"
        assert dlg._status_lbl.text() == br.BATCH_LAUNCH_FAILED
        assert not ann._batch_running and ann._batch_snapshot is None
        assert True not in ann.locks
        assert "MemoryError" in capsys.readouterr().out


# ============================================================ D-L1..D-L3


class _Fut:
    def __init__(self, done):
        self._done = done

    def done(self):
        return self._done


class TestOrphanList:
    def test_prune_and_append(self):
        ol = bc.OrphanList([_Fut(True), _Fut(False)])
        assert len(ol) == 2
        assert ol.prune() == 1
        ol.append(_Fut(False))
        assert len(ol) == 2 and len(list(ol)) == 2

    def test_the_poll_prunes_through_it(self, monkeypatch, widgets):
        ann = make_ann(monkeypatch, widgets)
        ann._batch_orphans = bc.OrphanList([_Fut(False), _Fut(True)])
        assert ann._live_batch_orphans() == 1

    def test_a_chain_run_uses_one(self, monkeypatch, widgets, qapp):
        ann = make_ann(monkeypatch, widgets, store={"g": stored(dil(1))})
        ann._batch_orphans = []
        ann._open_batch_refine_dialog()
        assert ann._batch_refine_dialog.select_chain("g")
        ann._on_batch_run()
        assert isinstance(ann._batch_orphans, bc.OrphanList)
        assert pump(qapp, lambda: not ann._batch_running)


class TestLateCancelAndCount:
    def test_a_cancel_after_the_last_object_still_cancels(self):
        v = vol5()
        ev = threading.Event()

        def progress(done, total):
            if done == total:
                ev.set()

        res = br.run_batch_recipe(v, [1, 2], rr.Recipe(steps=(dil(1),)),
                                  n_workers=1, progress_cb=progress,
                                  cancel_event=ev)
        assert res.cancelled and res.touched_ids == []
        assert np.array_equal(res.working_volume, v)

    def test_the_voxel_count_is_what_changed(self):
        v = np.zeros((20, 30, 40), np.int32)
        v[5:15, 5:25, 4:14] = 1
        v[5:15, 5:25, 17:27] = 2                 # a 3-voxel gap
        res = br.run_batch_recipe(v, [1, 2], rr.Recipe(steps=(dil(3),)),
                                  n_workers=1)
        actual = int(np.count_nonzero(res.working_volume != v))
        assert res.n_changed_total == actual
        # both grew into the same gap: the per-object counts over-count
        assert actual > 0

    @pytest.mark.parametrize("empty", ["skip", "erase"])
    def test_the_geometry_path_counts_what_changed_too(self, empty):
        # Two objects grow into the same 3-voxel gap; lower id wins it, so
        # the per-object counts (taken before contention) sum to more
        # than the voxels that actually changed.
        v = np.zeros((20, 30, 40), np.int32)
        v[5:15, 5:25, 4:14] = 1
        v[5:15, 5:25, 17:27] = 2
        recipe = [br.RefineStep("dilate_erode", {"radius": 3})]
        res = br.run_batch_recipe(v, [1, 2], recipe, n_workers=1,
                                  empty=empty)
        actual = int(np.count_nonzero(res.working_volume != v))
        per_object = sum(
            br.apply_recipe_to_label(
                v, lid, br.compute_label_roi(
                    v, lid, br._recipe_margin(recipe, v.shape)),
                recipe)[1]
            for lid in (1, 2))
        assert per_object > actual, "fixture: the objects must contend"
        assert res.n_changed_total == actual
        assert res.touched_ids == [1, 2]


# =================================================================== D-L4


def test_themed_notice_is_deleted(qapp, monkeypatch):
    from qtpy.QtCore import QCoreApplication, QEvent
    from qtpy.QtWidgets import QDialog, QWidget

    from napari_epicycle.ui_shell import theme

    monkeypatch.setattr(QDialog, "exec_", lambda self: 0)
    parent = QWidget()
    try:
        theme.themed_notice(parent, "T", "M")
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        assert parent.findChildren(QDialog) == []
    finally:
        parent.deleteLater()


# =================================================================== D-L5


class TestPreviewObject:
    def test_an_absent_id_is_nothing_selected(self, monkeypatch, widgets):
        ann = make_ann(monkeypatch, widgets)
        ann._selected = 1_000                   # the fallback, not an object
        ann._open_batch_refine_dialog()
        dlg = ann._batch_refine_dialog
        assert dlg._preview_btn.text() == br.BATCH_PREVIEW_BTN_NONE
        assert not dlg._preview_btn.isEnabled()
        ann._on_batch_preview_one()
        assert dlg._status_lbl.text() == br.BATCH_PREVIEW_NO_OBJECT
        assert ann.locks == []

    def test_an_object_painted_after_the_open_can_be_previewed(
            self, monkeypatch, widgets, qapp):
        ann = make_ann(monkeypatch, widgets)
        ann._open_batch_refine_dialog()
        ann.sync_manager.shared_labels["lbl"][1:4, 1:4, 1:4] = 42
        ann._selected = 42
        ann._on_batch_preview_one()
        assert ann.locks == [True]
        assert pump(qapp, lambda: not ann._batch_running)


# =================================================================== D-L6


def test_run_takes_one_bbox_pass_on_the_main_thread(monkeypatch, widgets,
                                                    qapp):
    ann = make_ann(monkeypatch, widgets, store={"g": stored(dil(1))})
    ann._open_batch_refine_dialog()
    dlg = ann._batch_refine_dialog
    assert dlg.select_chain("g")
    ann.sync_manager.shared_labels["lbl"][1:4, 1:4, 1:4] = 42   # after open
    main = threading.current_thread()
    calls = []
    real = br.label_bboxes

    def counting(vol, ids=None):
        if threading.current_thread() is main:
            calls.append(ids)
        return real(vol, ids)

    monkeypatch.setattr(br, "label_bboxes", counting)
    ann._on_batch_run()
    assert calls == [None]
    assert pump(qapp, lambda: not ann._batch_running)


# =================================================================== D-L7


def test_an_undo_all_that_undid_nothing_keeps_review(monkeypatch, widgets):
    ann = make_ann(monkeypatch, widgets)
    said = []
    ann._push_add_ui = lambda **k: said.append(k)
    ann._batch_running = False
    ann._batch_staged = False
    # the session entries are {"ids": ...} dicts since the Undo-all confirm
    # (f1f730ed; this fake had object() and failed at every commit since),
    # and the confirm is a seam -- answered "Take them back"
    from packages.epicycle_annotator import add_preview as AP

    ann._add_session = [{"ids": [1]}, {"ids": [2]}]
    asked = []
    ann._add_confirm = lambda text: asked.append(text) or True
    ann._add_review = True
    ann._add_piece_ids = [1, 2]
    ann._on_add_undo_session_cb = lambda n: 0
    ann._undo_add_all()
    assert asked, "Undo all asks first"
    assert ann._add_review is True and ann._add_piece_ids == [1, 2]
    assert said[-1]["status_text"] == AP.ADD_UNDO_NONE
    assert said[-1]["status_kind"] == "warn"


# =================================================================== U-L7


def test_the_poll_leaves_a_busy_dialog_alone(widgets):
    calls = []

    def provider(spec):
        calls.append(spec)
        return 7

    dlg = make_dialog(widgets, count_provider=provider)
    dlg.set_preview_object(obj_id=3)
    dlg.set_state("running")
    dlg._count_lbl.setText("kept")
    run_text = dlg._run_btn.text()
    calls.clear()
    dlg.set_preview_object(obj_id=0)
    dlg.set_preview_object(obj_id=9)
    assert calls == []
    assert dlg._count_lbl.text() == "kept"
    assert dlg._run_btn.text() == run_text
    dlg.set_state("idle")                      # redrawn from the last count
    assert dlg._count_lbl.text() == br.BATCH_COUNT.format(k="7 objects")
    assert dlg._preview_btn.text() == "Preview on object 9"



# ================================================== wording (commit 2)


class TestStoppedWording:
    def test_step_one_keeps_nothing_so_it_says_so(self):
        lines = br.summary_lines(stopped={4: (1, "Settle", "no room"),
                                          9: (1, "Settle", "no room")})
        assert lines == [br.BATCH_SUMMARY_STOPPED_FIRST.format(
            n="2 objects", method="Settle", ids="4, 9")]
        assert "keep the steps before it" not in lines[0]

    def test_an_enclosed_settle_does_not_read_as_an_error(self):
        why = br.SETTLE_ENCLOSED_REASON + " at that distance, so nothing " \
            "could move."
        lines = br.summary_lines(stopped={5: (2, "Settle", why),
                                          6: (2, "Settle", "other")})
        assert br.BATCH_SUMMARY_STOPPED_ENCLOSED.format(
            n="1 object", ids="5") in lines
        assert br.BATCH_SUMMARY_STOPPED.format(
            n="1 object", k=2, method="Settle", ids="6") in lines

    def test_two_element_values_still_read(self):
        lines = br.summary_lines(stopped={3: (2, "Dilate/Erode")})
        assert lines == [br.BATCH_SUMMARY_STOPPED.format(
            n="1 object", k=2, method="Dilate/Erode", ids="3")]

    def test_the_console_not_the_log(self):
        for name in ("BATCH_SUMMARY_STOPPED", "BATCH_SUMMARY_STOPPED_FIRST"):
            text = getattr(br, name)
            assert "(details in the console)" in text and "log" not in text

    def test_the_annotator_hands_the_reason_through(self, monkeypatch,
                                                   widgets, qapp):
        import packages.epicycle_annotator.reshape.settle as settle_mod
        from tests.annotator.test_batch_chain_dispatch import _FakeSettle

        why = br.SETTLE_ENCLOSED_REASON + " at that distance, so nothing " \
            "could move."
        monkeypatch.setattr(settle_mod, "settle_with_walls",
                            _FakeSettle("refused", [why]))
        layers = [_Img("mem", np.zeros(vol5().shape))]
        ann = make_ann(monkeypatch, widgets, layers=layers, refine_ch="mem",
                       store={"t": stored(dil(1), settle())})
        ann._open_batch_refine_dialog()
        dlg = ann._batch_refine_dialog
        dlg.select_chain("t")
        ann._on_batch_run()
        assert pump(qapp, lambda: not ann._batch_running)
        assert br.BATCH_SUMMARY_STOPPED_ENCLOSED.format(
            n="5 objects", ids="1, 2, 3, 4, 5") in dlg._summary_lbl.text()


class TestDialogWording:
    def test_an_unreadable_recipe_item_is_a_fixed_sentence(self, widgets,
                                                          capsys):
        dlg = make_dialog(widgets)
        dlg.set_chain_recipes({"bad": {"kind": "nope"}})
        assert dlg._chain_combo.itemText(1) == \
            br.BATCH_CHAIN_UNREADABLE.format(name="bad")
        assert "cannot be read" in capsys.readouterr().out

    def test_the_literals_are_constants(self, widgets):
        dlg = make_dialog(widgets)
        assert dlg._recent_combo.toolTip() == br.BATCH_RECENT_TIP
        assert dlg._saved_combo.toolTip() == br.BATCH_SAVED_TIP
        assert dlg._keep_tier_cb.text() == br.BATCH_KEEP_TIER_LABEL

    def test_the_save_prompt(self, widgets, monkeypatch):
        from qtpy.QtWidgets import QInputDialog

        seen = []
        monkeypatch.setattr(QInputDialog, "getText",
                            staticmethod(lambda *a: (seen.append(a[1:]),
                                                     ("", False))[1]))
        dlg = make_dialog(widgets, can_save_recipe=True,
                          on_save_recipe=lambda n: None)
        dlg._on_save_recipe_clicked()
        assert seen == [(br.BATCH_SAVE_PROMPT_TITLE,
                         br.BATCH_SAVE_PROMPT_LABEL)]

    def test_added_this_session_groups_thousands(self):
        assert br.BATCH_TARGET_ADDED.format(n=1204) == \
            "Added this session (1,204)"

    def test_the_dialog_opens_at_its_minimum_width(self, monkeypatch,
                                                   widgets):
        layers = [_Img("mem", np.zeros(vol5().shape))]
        ann = make_ann(monkeypatch, widgets, layers=layers,
                       store={"wall": stored(settle())})
        ann._batch_last_chain_name = "wall"     # the reason line is up
        ann._open_batch_refine_dialog()
        dlg = ann._batch_refine_dialog
        assert dlg.reason_text()
        assert dlg.width() == dlg.minimumSizeHint().width()
        assert dlg.width() < dlg.sizeHint().width() or \
            dlg.width() == dlg.sizeHint().width()


# ======================================= the count path (commit 3, perf)


def test_the_count_path_takes_no_full_volume_pass(monkeypatch, widgets):
    """The driver measured 1.13 s (lasso) / 1.84 s (navigator click) to a
    new "Plot selection (N)"; the annotator's own refresh measured 0.3-3.4
    ms on a 317 x 480 x 480 patch with 450 objects.  It must stay that way:
    no pass over the volume on any refresh, in any mode, chain or not."""
    import numpy

    layers = [_Img("mem", np.zeros(vol5().shape))]
    ann = make_ann(monkeypatch, widgets, layers=layers, refine_ch="mem",
                   store={"tidy": stored(dil(1), settle())})
    ann._get_plot_selection_cb = lambda: [1, 2, 3]
    ann._navigator_widget = type(ann._navigator_widget)(
        _items=[{"label_id": i, "tier": "good"} for i in range(1, 6)],
        selected_label_ids=lambda: {2, 4})
    ann._open_batch_refine_dialog()
    dlg = ann._batch_refine_dialog
    passes = []

    def no_pass(*a, **k):
        passes.append(a)
        raise AssertionError("a full-volume pass in the count path")

    monkeypatch.setattr(br, "label_bboxes", no_pass)
    monkeypatch.setattr(br, "resolve_target_labels", no_pass)
    real_unique = numpy.unique
    monkeypatch.setattr(numpy, "unique", lambda a, *x, **k: (
        passes.append("unique") if getattr(a, "size", 0) >= 30 * 40 * 140
        else None) or real_unique(a, *x, **k))
    for chain in (False, True):
        if chain:
            assert dlg.select_chain("tidy")
        for radio in ("_radio_all", "_radio_selected", "_radio_plot",
                      "_radio_tier"):
            getattr(dlg, radio).setChecked(True)
            ann._do_refresh_batch_count()
    assert passes == []
    assert dlg._radio_plot.text() == br.BATCH_TARGET_PLOT.format(n=3)
