"""Batch Refine runs a saved Refine recipe -- the dialog and the annotator.

Invariants 10, 11, 13, 14, 16: the
recipe row (its items, greyed items' text, the per-step summary in the page's
own words), the hidden builder and disabled "Recent" / "Saved steps" /
"Save recipe…", the channel combo (blank stays blank, refreshed on layer
insert / remove), the reason line (Run-only prefix, orphans), reopen
precedence (plain vs explicit-target vs preset), the histories kept apart
and the Refine file never written, the open gate order at both doors,
refusals before any work, the worker's slots connected before ``start()``,
the stale-token belt and the navigator tooltip.

Plain ``QApplication`` (pytest-qt is not installed).
"""
import hashlib
import json
import os
import threading
import time
import types

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from packages.epicycle_annotator import batch_refine as br  # noqa: E402
from packages.epicycle_annotator import refine_recipe as rr  # noqa: E402
from tests.annotator.test_batch_chain_dispatch import (  # noqa: E402
    TICKS,
    contour,
    dil,
    fill,
    sdf,
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


def stored(*steps, spacing_known=False):
    return rr.recipe_to_json(list(steps), created="t",
                             spacing_known=spacing_known, spacing=())


def settle_ticked():
    t = dict(TICKS)
    t["bgsub"] = t["lift"] = True
    return rr.RecipeStep("settle", {
        "wall_kind": "membrane", "solver": "watershed", "band_um": 2.0,
        "n_iter": 10, "clamp": 0.5, "compactness": 0.0, "prep_ticks": t,
        "prep_chain": "bgsub+gamma"})


def make_dialog(widgets, count=5, **kw):
    from napari_epicycle.annotator.batch_refine_dialog import BatchRefineDialog

    base = dict(count_provider=lambda spec: count, all_count=count,
                selected_count=0, available_tiers=["good"],
                on_preview_one=lambda: None, on_run=lambda: None,
                on_cancel=lambda: None, on_accept=lambda: None,
                on_reject=lambda: None, on_close=lambda: None)
    base.update(kw)
    dlg = BatchRefineDialog(**base)
    widgets.append(dlg)
    return dlg


# ================================================================ dialog


class TestRecipeRow:
    def test_items_and_greyed_reasons(self, widgets):
        dlg = make_dialog(widgets)
        dlg.set_chain_recipes({"tidy": stored(sdf(), settle()),
                               "bad": {"kind": "nope"}})
        c = dlg._chain_combo
        texts = [c.itemText(i) for i in range(c.count())]
        assert texts[0] == br.BATCH_CHAIN_NONE
        assert texts[1] == "tidy — Smooth -> Settle"
        assert texts[2] == br.BATCH_CHAIN_UNREADABLE.format(name="bad")
        assert not c.model().item(2).isEnabled()
        assert c.model().item(1).isEnabled()

    def test_a_file_problem_is_one_greyed_item(self, widgets):
        dlg = make_dialog(widgets)
        dlg.set_chain_recipes({"tidy": stored(sdf())}, problem="unreadable")
        c = dlg._chain_combo
        assert c.count() == 2 and c.itemText(1) == rr.FILE_UNREADABLE
        assert not c.model().item(1).isEnabled()

    def test_no_workspace(self, widgets):
        dlg = make_dialog(widgets)
        dlg.set_chain_recipes({}, no_workspace=True)
        assert not dlg._chain_combo.isEnabled()
        assert dlg.reason_text() == rr.NO_WORKSPACE

    def test_choosing_hides_the_builder_and_greys_the_history(self, widgets):
        dlg = make_dialog(widgets, can_save_recipe=True,
                          on_save_recipe=lambda n: None)
        assert dlg._save_recipe_btn.isEnabled()
        dlg.set_chain_recipes({"tidy": stored(sdf(), settle_ticked())})
        assert dlg.select_chain("tidy")
        assert dlg.chain_name() == "tidy"
        assert dlg._scroll.isHidden() and dlg._add_btn.isHidden()
        assert dlg._preset_btn.isHidden()
        assert not dlg._rows_off_lbl.isHidden()
        assert dlg._rows_off_lbl.text() == br.BATCH_ROWS_OFF
        assert not dlg._recent_combo.isEnabled()
        assert not dlg._saved_combo.isEnabled()
        assert not dlg._save_recipe_btn.isEnabled()
        # back to "none": the builder returns
        dlg._chain_combo.setCurrentIndex(0)
        assert dlg.chain_name() is None and not dlg._scroll.isHidden()
        assert dlg._save_recipe_btn.isEnabled()

    def test_the_summary_in_the_pages_own_words(self, widgets):
        dlg = make_dialog(widgets)
        dlg.set_chain_recipes({"tidy": stored(sdf(), settle_ticked(),
                                              contour(axis="XY"))})
        dlg.select_chain("tidy")
        lines = dlg._chain_summary_lbl.text().split("\n")
        assert lines == [
            "1. Smooth",
            "2. Settle — walls: Membrane sheets; preparation: Remove "
            "haze, Lift dim signal",
            "3. Active contour (Snap to edges) — walls: Membrane sheets",
        ]
        for token in ("settle", "membrane", "bgsub", "gamma", "snap"):
            assert token not in dlg._chain_summary_lbl.text()

    def test_the_labels(self, widgets):
        from qtpy.QtWidgets import QLabel

        dlg = make_dialog(widgets)
        texts = {lbl.text() for lbl in dlg.findChildren(QLabel)}
        assert br.BATCH_SAVED_STEPS_LABEL in texts and "Saved" not in texts
        assert br.BATCH_RECIPE_TITLE in texts
        assert br.BATCH_CHAIN_LABEL in texts

    def test_channel_row_only_when_a_step_reads_one(self, widgets):
        dlg = make_dialog(widgets)
        dlg.set_chain_recipes({"geo": stored(sdf(), dil(2)),
                               "wall": stored(settle())})
        dlg.select_chain("geo")
        assert dlg._chain_channel_row.isHidden()
        dlg.select_chain("wall")
        assert not dlg._chain_channel_row.isHidden()


class TestChannelCombo:
    def test_preferred_then_blank_stays_blank(self, widgets):
        dlg = make_dialog(widgets)
        dlg.set_chain_channels(["a", "b"], preferred="b")
        assert dlg.chain_channel() == "b"
        dlg.set_chain_channels(["a"])            # b left the viewer
        assert dlg.chain_channel() == ""
        dlg.set_chain_channels(["a", "c"])       # still blank, never "a"
        assert dlg.chain_channel() == ""
        dlg.set_chain_channels(["a", "b"])       # renamed back: restored
        assert dlg.chain_channel() == "b"

    def test_preferred_not_listed_is_blank(self, widgets):
        dlg = make_dialog(widgets)
        dlg.set_chain_channels(["a", "b"], preferred="zzz")
        assert dlg.chain_channel() == ""

    def test_no_preferred_is_blank(self, widgets):
        dlg = make_dialog(widgets)
        dlg.set_chain_channels(["a", "b"])
        assert dlg.chain_channel() == ""


class TestReasonLine:
    def test_run_only(self, widgets):
        dlg = make_dialog(widgets)
        dlg.set_preview_object(obj_id=7)
        dlg.set_chain_block(run="R", preview=None)
        assert dlg.reason_text() == br.BATCH_REASON_RUN_ONLY + "R"
        assert not dlg._run_btn.isEnabled() and dlg._preview_btn.isEnabled()

    def test_both_names_no_button(self, widgets):
        dlg = make_dialog(widgets)
        dlg.set_preview_object(obj_id=7)
        dlg.set_chain_block(run="R", preview="R")
        assert dlg.reason_text() == "R"
        assert not dlg._run_btn.isEnabled()
        assert not dlg._preview_btn.isEnabled()

    def test_orphans(self, widgets):
        dlg = make_dialog(widgets)
        dlg.set_preview_object(obj_id=7)
        dlg.set_orphans_running(n=3)
        assert dlg.reason_text() == br.BATCH_ORPHANS_RUNNING.format(
            n="3 objects")
        assert not dlg._run_btn.isEnabled()
        assert not dlg._preview_btn.isEnabled()
        dlg.set_orphans_running(n=0)
        assert dlg.reason_text() == "" and dlg._run_btn.isEnabled()

    def test_the_buttons_words(self, widgets):
        dlg = make_dialog(widgets, count=12)
        assert dlg._run_btn.text() == br.BATCH_RUN_BTN.format(
            n="12 objects") == "Run on 12 objects"
        dlg.set_preview_object(obj_id=0)
        assert dlg._preview_btn.text() == br.BATCH_PREVIEW_BTN_NONE
        assert not dlg._preview_btn.isEnabled()
        assert br.BATCH_PREVIEW_NO_OBJECT in dlg._count_lbl.text()
        dlg.set_preview_object(obj_id=31)
        assert dlg._preview_btn.text() == "Preview on object 31"
        assert "labels" not in dlg._preview_btn.toolTip()


# ============================================================= annotator


def C():
    from napari_epicycle.annotator.curation_3d_widget import (
        Curation3DAnnotator,
    )

    return Curation3DAnnotator


class _Img:
    def __init__(self, name, data):
        self.name = name
        self.data = data


class _Timer:
    def __init__(self):
        self.active = False

    def isActive(self):
        return self.active

    def start(self, *a):
        self.active = True

    def stop(self):
        self.active = False


class Ann(types.SimpleNamespace):
    def __getattr__(self, name):
        fn = C().__dict__.get(name)
        if fn is not None and (callable(fn) or isinstance(fn, staticmethod)):
            return fn.__get__(self, type(self))
        raise AttributeError(name)


class _Adv:
    def __init__(self, ch=""):
        self.ch = ch

    def refine_channel_name(self):
        return self.ch


def vol5():
    v = np.zeros((30, 40, 140), np.int32)
    for i in range(5):
        v[8:22, 10:30, 5 + 27 * i:25 + 27 * i] = i + 1
    return v


def make_ann(monkeypatch, widgets, *, store=None, problem=None,
             layers=(), refine_ch=""):
    import napari_epicycle.annotator.batch_refine_dialog as brd
    import napari_epicycle.annotator.curation_3d_widget as mod

    real = brd.BatchRefineDialog

    def ctor(**kw):
        kw.pop("parent", None)
        d = real(**kw)
        widgets.append(d)
        return d

    monkeypatch.setattr(brd, "BatchRefineDialog", ctor)
    monkeypatch.setattr(mod, "Image", _Img)
    v = vol5()
    ann = Ann(
        _batch_refine_dialog=None, _batch_worker=None, _batch_thread=None,
        _load_locked=False, _active_labels_name="lbl",
        sync_manager=types.SimpleNamespace(shared_labels={"lbl": v}),
        _refine_state=types.SimpleNamespace(
            can_start_batch_refine=lambda: True),
        _sync_refine_state=lambda: None,
        _navigator_widget=types.SimpleNamespace(
            _items=[], selected_label_ids=lambda: set()),
        _save_batch_recipe_cb=None, _get_saved_batch_recipes_cb=None,
        _batch_last_recipe=None, _batch_recipe_history=[],
        _get_plot_selection_cb=None,
        _refresh_merge_button_state=lambda: None,
        _refresh_label_action_buttons_state=lambda: None,
        _batch_poll_timer=_Timer(),
        viewer=types.SimpleNamespace(layers=list(layers)),
        _selected=3, _prob_volume=None, _prob_volume_path=None,
        _fire_batch_refine_lock=lambda active: ann.locks.append(active),
        _lock_subviewers_for_preview=lambda: None,
        _restore_ready_to_draw_state=lambda: None,
        _sync_all_views_after_operation=lambda roi=None: None,
        _on_batch_refined_cb=lambda *a, **k: ann.accepted.append((a, k)),
        locks=[], accepted=[], notices=[],
    )
    ann._get_saved_refine_recipes_cb = (
        lambda: (dict(store or {}), problem))
    ann._get_advanced_features_widget = lambda: _Adv(refine_ch)
    ann._get_selected_label = lambda: ann._selected
    ann._reshape_context = lambda: {"spacing": (1.0, 1.0, 1.0)}
    ann._layer_by_name = lambda layers, name: next(
        (lay for lay in layers if lay.name == name), None)
    return ann


def pump(qapp, predicate, seconds=30.0):
    t0 = time.time()
    while time.time() - t0 < seconds:
        qapp.processEvents()
        if predicate():
            qapp.processEvents()
            return True
        time.sleep(0.01)
    return False


class TestOpenGates:
    """Invariant 14: the gate order, and the sentence at both doors."""

    def test_stopping_worker_first_even_when_locked(self, monkeypatch,
                                                     widgets):
        ann = make_ann(monkeypatch, widgets)
        ann._batch_worker = object()
        ann._load_locked = True
        assert ann._open_batch_refine_dialog() == br.BATCH_STILL_STOPPING
        assert ann._batch_refine_dialog is None

    def test_the_lock(self, monkeypatch, widgets):
        ann = make_ann(monkeypatch, widgets)
        ann._load_locked = True
        assert ann._open_batch_refine_dialog() == br.BATCH_BUSY

    def test_not_clean(self, monkeypatch, widgets):
        ann = make_ann(monkeypatch, widgets)
        ann._refine_state = types.SimpleNamespace(
            can_start_batch_refine=lambda: False)
        assert ann._open_batch_refine_dialog() == br.BATCH_NOT_CLEAN

    def test_no_volume(self, monkeypatch, widgets):
        ann = make_ann(monkeypatch, widgets)
        ann.sync_manager = types.SimpleNamespace(shared_labels={})
        assert ann._open_batch_refine_dialog() == br.BATCH_NO_VOLUME

    def test_an_open_returns_none(self, monkeypatch, widgets):
        ann = make_ann(monkeypatch, widgets)
        assert ann._open_batch_refine_dialog() is None
        assert ann._batch_refine_dialog is not None

    def test_the_navigator_door_shows_a_notice(self, monkeypatch, widgets):
        import napari_epicycle.ui_shell.theme as theme

        ann = make_ann(monkeypatch, widgets)
        ann._load_locked = True
        monkeypatch.setattr(theme, "themed_notice",
                            lambda p, t, m, **k: ann.notices.append((t, m)))
        ann._open_batch_refine_from_navigator()
        assert ann.notices == [(br.BATCH_NOTICE_TITLE, br.BATCH_BUSY)]

    def test_the_navigator_is_wired_to_that_door(self):
        import inspect

        src = inspect.getsource(C()._wire_navigator_callbacks)
        assert "on_batch_refine=self._open_batch_refine_from_navigator" in src


class TestReopenPrecedence:
    STORE = {"tidy": stored(sdf(), dil(1))}

    def _run_chain(self, qapp, ann):
        dlg = ann._batch_refine_dialog
        assert dlg.select_chain("tidy")
        ann._on_batch_run()
        assert pump(qapp, lambda: not ann._batch_running)

    def test_a_plain_open_reselects_the_last_chain(self, monkeypatch,
                                                   widgets, qapp):
        ann = make_ann(monkeypatch, widgets, store=self.STORE)
        ann._open_batch_refine_dialog()
        self._run_chain(qapp, ann)
        assert ann._batch_last_chain_name == "tidy"
        ann._on_batch_reject()
        ann._teardown_batch_dialog()
        ann._open_batch_refine_dialog()
        assert ann._batch_refine_dialog.chain_name() == "tidy"

    def test_an_explicit_target_open_never_reselects(self, monkeypatch,
                                                     widgets, qapp):
        ann = make_ann(monkeypatch, widgets, store=self.STORE)
        ann._batch_last_chain_name = "tidy"
        ann._open_batch_refine_dialog(explicit_targets=[1, 2],
                                      default_target="selected")
        assert ann._batch_refine_dialog.chain_name() is None

    def test_a_geometry_run_clears_it(self, monkeypatch, widgets, qapp):
        ann = make_ann(monkeypatch, widgets, store=self.STORE)
        ann._batch_last_chain_name = "tidy"
        ann._open_batch_refine_dialog()
        dlg = ann._batch_refine_dialog
        dlg._chain_combo.setCurrentIndex(0)
        ann._on_batch_run()
        assert pump(qapp, lambda: not ann._batch_running)
        assert ann._batch_last_chain_name is None

    def test_a_preset_open_never_reselects(self, monkeypatch, widgets):
        ann = make_ann(monkeypatch, widgets, store=self.STORE)
        ann._batch_last_chain_name = "tidy"
        ann._add_piece_ids = [1, 2]
        ann._push_add_ui = lambda **k: None
        ann._smooth_add_pieces()
        dlg = ann._batch_refine_dialog
        assert dlg.chain_name() is None
        assert [s.method for s in dlg.recipe()] == ["sdf"]


class TestHistoriesApart:
    """Invariant 10: a chain run + Accept leaves `_batch_last_recipe`,
    "Recent" and Batch's own file alone, and never writes the Refine
    file."""

    def test_after_a_chain_run_and_accept(self, monkeypatch, widgets, qapp,
                                          tmp_path):
        from napari_epicycle.fusion_reconstruction import (
            _batch_recipe_store as store_mod,
        )

        refine_file = tmp_path / "refine_chain_recipes.json"
        batch_file = tmp_path / "batch_refine_recipes.json"
        store_mod.save_named_recipe(refine_file, "tidy",
                                    stored(sdf(), dil(1)), value_type=dict)
        store_mod.save_named_recipe(batch_file, "old",
                                    [{"method": "sdf", "params": {}}])
        before_r = hashlib.sha256(refine_file.read_bytes()).hexdigest()
        before_b = hashlib.sha256(batch_file.read_bytes()).hexdigest()

        ann = make_ann(monkeypatch, widgets)
        ann._get_saved_refine_recipes_cb = lambda: (
            store_mod.load_named_recipes_checked(refine_file,
                                                 value_type=dict))
        ann._save_batch_recipe_cb = lambda name, rj: (
            store_mod.save_named_recipe(batch_file, name, rj))
        ann._batch_last_recipe = [{"method": "fill_holes",
                                   "params": {"axis": "3D"}}]
        sticky = json.dumps(ann._batch_last_recipe)
        ann._batch_recipe_history = [[{"method": "sdf", "params": {}}]]
        recent = json.dumps(ann._batch_recipe_history)

        ann._open_batch_refine_dialog()
        assert ann._batch_refine_dialog.select_chain("tidy")
        ann._on_batch_run()
        assert pump(qapp, lambda: not ann._batch_running)
        assert ann._batch_staged
        ann._on_batch_accept()
        assert ann.accepted, "the accept callback fired"

        assert json.dumps(ann._batch_last_recipe) == sticky
        assert json.dumps(ann._batch_recipe_history) == recent
        assert hashlib.sha256(refine_file.read_bytes()).hexdigest() == before_r
        assert hashlib.sha256(batch_file.read_bytes()).hexdigest() == before_b


class TestRunRefusals:
    def test_a_blocker_refusal_does_no_work(self, monkeypatch, widgets):
        """Invariant 11: nothing copied, no worker, no lock."""
        ann = make_ann(monkeypatch, widgets,
                       store={"wall": stored(settle())})
        ann._open_batch_refine_dialog()
        dlg = ann._batch_refine_dialog
        dlg.select_chain("wall")                   # needs a channel: none
        ann._run_batch_refine([1, 2, 3])
        assert ann.locks == [] and ann._batch_worker is None
        assert _safe(ann, "_batch_snapshot") is None
        assert dlg._status_lbl.text() == rr.BATCH_NEEDS_CHANNEL.format(
            k=1, word="Settle")

    def test_the_reason_line_shows_the_channel_refusal(self, monkeypatch,
                                                      widgets):
        ann = make_ann(monkeypatch, widgets,
                       store={"wall": stored(settle())})
        ann._open_batch_refine_dialog()
        dlg = ann._batch_refine_dialog
        dlg.select_chain("wall")
        assert dlg.reason_text() == rr.BATCH_NEEDS_CHANNEL.format(
            k=1, word="Settle")
        assert not dlg._run_btn.isEnabled()

    def test_the_3d_cap_greys_run_only(self, monkeypatch, widgets):
        import packages.epicycle_annotator.refine_recipe as mod

        monkeypatch.setattr(mod, "BATCH_CONTOUR_3D_MAX", 2)
        layers = [_Img("mem", np.zeros(vol5().shape))]
        ann = make_ann(monkeypatch, widgets, layers=layers, refine_ch="mem",
                       store={"c3": stored(contour(axis="3D"))})
        ann._open_batch_refine_dialog()
        dlg = ann._batch_refine_dialog
        dlg.select_chain("c3")
        assert dlg.chain_channel() == "mem"        # the Refine page's pick
        assert dlg.reason_text().startswith(br.BATCH_REASON_RUN_ONLY)
        assert not dlg._run_btn.isEnabled()
        assert dlg._preview_btn.isEnabled()

    def test_orphans_refuse_run(self, monkeypatch, widgets):
        ann = make_ann(monkeypatch, widgets)
        ann._open_batch_refine_dialog()

        class _Busy:
            def done(self):
                return False

        ann._batch_orphans = [_Busy(), _Busy()]
        ann._run_batch_refine([1, 2])
        dlg = ann._batch_refine_dialog
        assert dlg._status_lbl.text() == br.BATCH_ORPHANS_RUNNING.format(
            n="2 objects")
        assert ann.locks == []
        ann._on_batch_poll()
        assert dlg.reason_text() == br.BATCH_ORPHANS_RUNNING.format(
            n="2 objects")
        assert not dlg._run_btn.isEnabled()
        ann._batch_orphans = []
        ann._on_batch_poll()
        assert dlg.reason_text() == ""


class TestLifecycle:
    def test_worker_slots_connected_before_start(self, monkeypatch, widgets):
        from qtpy.QtCore import QThread

        ann = make_ann(monkeypatch, widgets, store={"g": stored(dil(1))})
        ann._open_batch_refine_dialog()
        ann._batch_refine_dialog.select_chain("g")
        seen = {}
        real_start = QThread.start

        def start(th, *a):
            w = ann._batch_worker
            try:
                seen["finished"] = w.receivers(w.finished)
            except Exception:
                seen["finished"] = None
            seen["connected"] = True
            return real_start(th, *a)

        monkeypatch.setattr(QThread, "start", start)
        ann._run_batch_refine([1, 2])
        assert seen.get("connected")
        # the annotator's finished handler was connected BEFORE start()
        assert seen["finished"] >= 1
        assert pump(QApplication_(), lambda: not ann._batch_running)

    def test_a_stale_signal_is_ignored(self, monkeypatch, widgets):
        ann = make_ann(monkeypatch, widgets)
        ann._batch_token = 5
        old = types.SimpleNamespace(_epicycle_token=4)
        ann._signal_sender = lambda: old
        ann._batch_running = True
        ann._batch_thread = "keep"
        ann._on_batch_worker_finished(types.SimpleNamespace(cancelled=True))
        assert ann._batch_thread == "keep" and ann._batch_running

    def test_cancel_sentence_names_settle(self, monkeypatch, widgets):
        ann = make_ann(monkeypatch, widgets)
        ann._open_batch_refine_dialog()
        ann._batch_worker = types.SimpleNamespace(
            cancel=lambda: None, cancel_event=threading.Event())
        ann._batch_chain_has_settle = True
        ann._on_batch_cancel()
        assert ann._batch_refine_dialog._status_lbl.text() == \
            br.BATCH_CANCELLING_SETTLE
        ann._batch_chain_has_settle = False
        ann._on_batch_cancel()
        assert ann._batch_refine_dialog._status_lbl.text() == \
            br.BATCH_CANCELLING
        ann._batch_worker = None

    def test_a_queued_tick_after_cancel_keeps_cancelling(self, monkeypatch,
                                                        widgets):
        ann = make_ann(monkeypatch, widgets)
        ann._open_batch_refine_dialog()
        ev = threading.Event()
        ev.set()
        ann._batch_worker = types.SimpleNamespace(cancel_event=ev)
        dlg = ann._batch_refine_dialog
        dlg.set_status(br.BATCH_CANCELLING)
        ann._on_batch_worker_progress(3, 9)
        ann._on_batch_worker_stage("refining", 0, 9)
        assert dlg._status_lbl.text() == br.BATCH_CANCELLING
        ann._batch_worker = None

    def test_the_summary_names_the_stopped(self, monkeypatch, widgets,
                                           qapp):
        """A chain whose step 2 refuses: the object keeps step 1 and the
        staged summary says so, one line per step."""
        import packages.epicycle_annotator.reshape.settle as settle_mod
        from tests.annotator.test_batch_chain_dispatch import _FakeSettle

        monkeypatch.setattr(settle_mod, "settle_with_walls",
                            _FakeSettle("refused", ["no room"]))
        layers = [_Img("mem", np.zeros(vol5().shape))]
        ann = make_ann(monkeypatch, widgets, layers=layers, refine_ch="mem",
                       store={"t": stored(dil(1), settle())})
        ann._open_batch_refine_dialog()
        dlg = ann._batch_refine_dialog
        dlg.select_chain("t")
        ann._on_batch_run()
        assert pump(qapp, lambda: not ann._batch_running)
        text = dlg._summary_lbl.text()
        assert br.BATCH_SUMMARY_STOPPED.format(
            n="5 objects", k=2, method="Settle",
            ids="1, 2, 3, 4, 5") in text
        assert ann._batch_staged


def QApplication_():
    from qtpy.QtWidgets import QApplication

    return QApplication.instance()


def _safe(obj, name):
    try:
        return getattr(obj, name)
    except AttributeError:
        return None


def test_the_navigator_tooltip(qapp):
    from napari_epicycle.annotator.curation_label_navigator import (
        CurationLabelNavigator,
    )

    nav = CurationLabelNavigator()
    try:
        assert nav._batch_refine_btn.toolTip() == br.NAV_BATCH_REFINE_TIP
        assert "geometry-only" not in nav._batch_refine_btn.toolTip()
    finally:
        nav.setParent(None)
        nav.deleteLater()
