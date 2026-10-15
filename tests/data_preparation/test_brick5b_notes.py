"""Brick 5b notes 3-6 - the leftovers the no-reload change exposed.

3. `_on_layers_changed` stays CONNECTED to the shared viewer after step-exit, so
   Step 2/3 layer churn re-keyed (3)'s channel picker and (2)'s recipe cards off
   layers that are not ours.  The step-enter re-show used to heal it by
   accident; Brick 5 removed the re-show.
4. `_clear_patch_locations` emptied the STATE but left the WIDGETS advertising
   it - the (4) master checkbox checked/partial and the row eyes lit, for boxes
   that no longer exist (and it is called from step-exit, so they came back lit
   on re-entry with nothing drawn).
5. Both export `finally` blocks wrote the bar BEFORE releasing the lock, so a
   bar whose C++ object had gone (napari closing mid-export) stranded the
   surface locked forever.
6. `_largedata_busy_label` outlived its Prepare run.

Qt-free: `__new__` controllers + recorders.
"""
from __future__ import annotations

import types

import pytest

from napari_epicycle.data_preparation import controller as ctlmod

C = ctlmod.DataPreparationController


# --------------------------------------------------------------------------- #
# 3. the layers-changed handler is inert while Step 1 is not the active step
# --------------------------------------------------------------------------- #

def _layers_ctl(left):
    c = C.__new__(C)
    c._left_step1 = left
    c._adding_preview_layers = False
    c._adding_processed_layers = False
    c._swapping_source_display = False
    c.touched = []
    c.viewer = types.SimpleNamespace(layers=[])
    c.widget = types.SimpleNamespace(
        update_image_layer_list=lambda names: c.touched.append(names),
        update_shape_layer_list=lambda names: c.touched.append(names))
    c._log = lambda *a, **k: None
    return c


def test_layer_churn_in_another_step_does_not_re_key_step_1():
    c = _layers_ctl(left=True)
    c._on_layers_changed()
    assert c.touched == [], "Step 1 owns none of the canvas while it is away"


def test_the_handler_still_runs_while_step_1_is_active():
    c = _layers_ctl(left=False)
    c._on_layers_changed()
    assert c.touched, "the guard must not disable the normal path"


def test_the_guard_is_the_first_thing_in_the_handler():
    """A guard placed after the rebuild would be decoration."""
    import inspect

    src = inspect.getsource(C._on_layers_changed)
    body = src[src.index('"""', src.index('"""') + 3):]
    assert body.index("_left_step1") < body.index("_adding_preview_layers")


# --------------------------------------------------------------------------- #
# 4. clearing the locations also un-lights the toggles
# --------------------------------------------------------------------------- #

class _BoxList:
    def __init__(self, boom=False):
        self.refreshed = 0
        self._boom = boom

    def refresh(self):
        if self._boom:
            raise RuntimeError("row gone")
        self.refreshed += 1


def _loc_ctl(**kw):
    c = C.__new__(C)
    c._visible_patch_locations = {"p1", "p2"}
    c._loc_overlay = None
    c.synced = []
    c._log = lambda *a, **k: None
    c.widget = types.SimpleNamespace(
        sync_show_locations_master=lambda: c.synced.append(1),
        library_box_list=_BoxList(**kw))
    return c


def test_clearing_locations_resyncs_the_master_toggle_and_the_row_eyes():
    c = _loc_ctl()
    c._clear_patch_locations()
    assert c._visible_patch_locations == set()
    assert c.synced == [1], "the master checkbox stayed checked for no boxes"
    assert c.widget.library_box_list.refreshed == 1, "the row eyes stayed lit"


def test_a_raising_row_refresh_never_breaks_the_clear():
    """This runs from step-exit and unload; it is cleanup, not a feature."""
    c = _loc_ctl(boom=True)
    c._clear_patch_locations()              # must not raise
    assert c.synced == [1]


def test_the_clear_survives_a_widget_without_the_helpers():
    c = _loc_ctl()
    c.widget = types.SimpleNamespace()
    c._clear_patch_locations()
    assert c._visible_patch_locations == set()


# --------------------------------------------------------------------------- #
# 5. a dead bar can never strand the surface lock
# --------------------------------------------------------------------------- #

class _DeadBar:
    """A QProgressBar whose C++ object is gone - every call raises."""

    def _boom(self, *_a):
        raise RuntimeError("wrapped C/C++ object has been deleted")

    setRange = setValue = setFormat = _boom


def _export_ctl(bar):
    c = C.__new__(C)
    c.widget = types.SimpleNamespace(export_progress_bar=bar)
    c._left_step1 = False
    c._migration_offer_timer = None
    c.busy = []
    c.status = []
    c._set_step1_busy = lambda on: c.busy.append(bool(on))
    c._set_export_status = lambda m, success=True: c.status.append((m, success))
    c._refresh_conditioning_status = lambda: None
    c._export_patch_by_name = lambda n, require_conditioned=False: True
    return c


def test_the_multi_patch_loop_unlocks_even_with_a_dead_bar(monkeypatch):
    import sys

    mod = types.ModuleType("qtpy.QtWidgets")
    mod.QApplication = types.SimpleNamespace(processEvents=lambda: None)
    monkeypatch.setitem(sys.modules, "qtpy.QtWidgets", mod)

    c = _export_ctl(_DeadBar())
    c._run_patch_export_loop(["a"], require_conditioned=False, noun="patch(es)")
    assert c.busy == [True, False], "the lock is the thing that must release"
    assert c.status == [("Exported 1 patch(es).", True)], (
        "a dead bar must not abort the files the user asked for")


def test_the_single_patch_export_unlocks_even_with_a_dead_bar():
    c = _export_ctl(_DeadBar())
    c.widget.extracted_roi_list = types.SimpleNamespace(
        currentItem=lambda: types.SimpleNamespace(text=lambda: "p1"))
    c._export_current_patch()
    assert c.busy == [True, False]


# --------------------------------------------------------------------------- #
# 6. the Prepare bar label does not outlive its run
# --------------------------------------------------------------------------- #

def test_the_prepare_label_is_dropped_on_the_terminal():
    c = C.__new__(C)
    thread = object()
    c._largedata_thread = thread
    c._largedata_worker = object()
    c._draining_largedata_threads = []
    c._largedata_busy_label = "Preparing wound.tif…"
    c._cleanup_largedata_thread(thread)
    assert c._largedata_busy_label is None
    assert c._largedata_thread is None


def test_a_draining_thread_does_not_clear_the_live_run_label():
    c = C.__new__(C)
    live, old = object(), object()
    c._largedata_thread = live
    c._largedata_worker = object()
    c._draining_largedata_threads = [(old, object())]
    c._largedata_busy_label = "Preparing new.tif…"
    c._cleanup_largedata_thread(old)
    assert c._largedata_busy_label == "Preparing new.tif…"
    assert c._largedata_thread is live
