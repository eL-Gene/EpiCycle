"""Brick L's controller seams (hooks 2 and 4): each ``ui.on_*`` is called with the right arguments, a UI WITHOUT the
hooks still runs every seam (the getattr rule), a failing hook never breaks
the controller.  Qt-free fakes."""
import inspect
import re

from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
    TrainingAnnotationController as C,
)


class _Sig:
    def __init__(self):
        self.slots = []

    def connect(self, fn):
        self.slots.append(fn)


class _PatchList:
    """The controller's own accesses of ``train_patch_list`` (count / item)."""

    def __init__(self, n=2):
        self._n = n
        self.itemClicked = _Sig()
        self.loadRequested = _Sig()

    def count(self):
        return self._n

    def item(self, i):
        return object()


class _UIWithHooks:
    def __init__(self):
        self.train_patch_list = _PatchList()
        self.calls = []

    def on_session_patches(self, payload):
        self.calls.append(("session_patches", payload))

    def on_shutdown(self):
        self.calls.append(("shutdown",))


class _UIWithoutHooks:
    def __init__(self):
        self.train_patch_list = _PatchList()


def _ctrl(ui):
    c = C.__new__(C)
    c.ui = ui
    c.sample_records = {
        "a": {"status": "staged", "bucket": "lateral"},
        "b": {"status": "staged", "bucket": "axial"},
        "c": {"status": "predicted", "bucket": "lateral"},
    }
    c._patch_shapes = {"p": (3, 4, 5)}
    c.current_round_by_view = {"lateral": 2, "axial": 1}
    c.loaded_patch_name = "p"
    c.selected_view = "XY"
    c.working_dir = "/ws"
    c.active_run_id = "Cell"
    c._effective_stack_path = lambda: "/ws/patches"
    return c


class TestHook4SessionPatches:
    def test_the_payload(self):
        ui = _UIWithHooks()
        _ctrl(ui)._refresh_patch_list_badges()
        (name, payload), = ui.calls
        assert name == "session_patches"
        assert set(payload) >= {"records", "shapes", "round_by_view", "loaded",
                                "view", "staged_by_bucket", "workspace",
                                "stack_dir", "run_id"}
        assert payload["round_by_view"] == {"lateral": 2, "axial": 1}
        assert payload["staged_by_bucket"] == {"lateral": 1, "axial": 1}
        assert payload["loaded"] == "p" and payload["view"] == "XY"
        assert payload["workspace"] == "/ws" and payload["run_id"] == "Cell"
        assert payload["stack_dir"] == "/ws/patches"

    def test_the_payload_hands_over_the_live_records_read_only(self):
        ui = _UIWithHooks()
        c = _ctrl(ui)
        c._refresh_patch_list_badges()
        assert ui.calls[0][1]["records"] is c.sample_records

    def test_a_ui_without_the_hook_still_runs_the_seam(self):
        _ctrl(_UIWithoutHooks())._refresh_patch_list_badges()

    def test_no_patch_list_is_a_no_op(self):
        ui = _UIWithHooks()
        del ui.train_patch_list
        _ctrl(ui)._refresh_patch_list_badges()
        assert ui.calls == []

    def test_a_failing_hook_does_not_break_the_refresh(self):
        class Bad(_UIWithHooks):
            def on_session_patches(self, payload):
                raise RuntimeError("boom")

        _ctrl(Bad())._refresh_patch_list_badges()

    def test_the_rows_are_no_longer_widgets_and_text_is_not_blanked(self):
        src = inspect.getsource(C._refresh_patch_list_badges)
        assert ".setItemWidget(" not in src
        assert 'setText("")' not in src


class TestHook2Shutdown:
    def test_shutdown_calls_the_hook_first(self):
        ui = _UIWithHooks()
        c = _ctrl(ui)
        c._shutdown_done = False
        try:
            c.shutdown()
        except Exception:
            pass        # later steps of a bare controller may fail; not ours
        assert ("shutdown",) in ui.calls

    def test_a_ui_without_the_hook_still_shuts_down(self):
        c = _ctrl(_UIWithoutHooks())
        c._shutdown_done = False
        try:
            c.shutdown()
        except AttributeError as exc:
            assert "on_shutdown" not in str(exc)
        assert c._shutdown_done is True

    def test_a_failing_hook_does_not_stop_the_shutdown(self):
        class Bad(_UIWithHooks):
            def on_shutdown(self):
                raise RuntimeError("boom")

        c = _ctrl(Bad())
        c._shutdown_done = False
        try:
            c.shutdown()
        except RuntimeError as exc:
            assert "boom" not in str(exc)

    def test_it_is_idempotent(self):
        ui = _UIWithHooks()
        c = _ctrl(ui)
        c._shutdown_done = True
        c.shutdown()
        assert ui.calls == []

    def test_the_hook_sits_after_the_done_flag_and_the_abort_flag(self):
        src = inspect.getsource(C.shutdown)
        a = src.index("self._shutdown_done = True")
        b = src.index('"on_shutdown"')
        c_ = src.index("_training_abort_flag")
        assert a < c_ < b   # audit: the abort flag first (bounded wait after)


class TestUpDownWiring:
    def test_load_requested_goes_to_the_click_slot_and_nothing_else(self):
        src = inspect.getsource(C._connect_signals)
        assert re.search(
            r'_connect\("train_patch_list", "loadRequested",\s*'
            r'self\._on_patch_list_clicked\)', src)
        assert '"train_patch_list", "currentItemChanged"' not in src  # D18
        assert re.search(r'"train_patch_list", "itemClicked"', src)

    def test_a_list_without_the_signal_is_skipped_silently(self):
        class Old:
            itemClicked = _Sig()

        ui = _UIWithoutHooks()
        ui.train_patch_list = Old()
        c = _ctrl(ui)
        try:
            c._connect_signals()
        except AttributeError as exc:
            assert "loadRequested" not in str(exc)
        assert Old.itemClicked.slots


# ---------------------------------------------------------------------------
# Brick A: the Slices / staged / mask-write / workspace-open seams
# ---------------------------------------------------------------------------
import pytest  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from qtpy.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])

import os  # noqa: E402

import numpy as np  # noqa: E402


class _FakeItem:
    def __init__(self, text):
        self._t = text
        self._d = {}

    def text(self):
        return self._t

    def data(self, role):
        return self._d.get(role)

    def setData(self, role, value):
        self._d[role] = value


class _FakeList:
    """clear / addItem / count / item -- all the staged seam touches."""

    def __init__(self):
        self.items = []

    def clear(self):
        self.items = []

    def addItem(self, text):
        self.items.append(_FakeItem(text))

    def count(self):
        return len(self.items)

    def item(self, i):
        return self.items[i]


class _NamedPatchList(_PatchList):
    def __init__(self, names):
        super().__init__(len(names))
        self._names = names

    def item(self, i):
        from qtpy.QtCore import Qt
        it = _FakeItem(self._names[i])
        it.setData(Qt.UserRole, self._names[i])
        return it


class _SliceUI:
    def __init__(self):
        self.data_to_add_list = _FakeList()
        self.dataset_staging = None
        self.train_patch_list = _PatchList()
        self.calls = []

    def on_slice_rows(self, payload):
        self.calls.append(("slice_rows", payload))

    def on_slice_written(self, sid):
        self.calls.append(("written", sid))

    def on_workspace_opened(self, ws, ids):
        self.calls.append(("opened", ws, list(ids)))


def _sctrl(ui):
    c = _ctrl(ui)
    c.view_target = "lateral"
    c.sample_records["a"]["updated_at"] = "1"
    return c


class TestSliceRowsHookStaged:
    def test_the_staged_fill_calls_the_hook_with_staged_true(self):
        ui = _SliceUI()
        c = _sctrl(ui)
        c._sync_data_to_add_list()
        (name, payload), = ui.calls
        assert name == "slice_rows" and payload["staged"] is True
        assert payload["ids"] == ["a"] and payload["bucket"] == "lateral"
        assert payload["records"] is c.sample_records
        assert payload["workspace"] == "/ws" and payload["loaded_patch"] == "p"

    def test_item_text_is_still_the_sample_id(self):
        ui = _SliceUI()
        _sctrl(ui)._sync_data_to_add_list()
        assert [ui.data_to_add_list.item(i).text()
                for i in range(ui.data_to_add_list.count())] == ["a"]

    def test_the_hook_runs_after_the_rows_are_added(self):
        seen = []

        class UI(_SliceUI):
            def on_slice_rows(self, payload):
                seen.append(self.data_to_add_list.count())

        _sctrl(UI())._sync_data_to_add_list()
        assert seen == [1]

    def test_a_ui_without_the_hook_still_fills_the_list(self):
        ui = _SliceUI()
        ui.on_slice_rows = None
        c = _sctrl(ui)
        c._sync_data_to_add_list()
        assert ui.data_to_add_list.count() == 1

    def test_a_failing_hook_does_not_break_the_fill(self):
        class Bad(_SliceUI):
            def on_slice_rows(self, payload):
                raise RuntimeError("boom")

        ui = Bad()
        _sctrl(ui)._sync_data_to_add_list()
        assert ui.data_to_add_list.count() == 1

    def test_stack_paths_come_from_the_patch_list_names_without_disk(self):
        ui = _SliceUI()
        ui.train_patch_list = _NamedPatchList(["p.tif", "q.tiff"])
        c = _sctrl(ui)
        c._effective_stack_path = lambda: "/ws/patches"
        sp = c._slice_stack_paths()
        assert sp == {"p": os.path.join("/ws/patches", "p.tif"),
                      "q": os.path.join("/ws/patches", "q.tiff")}


class TestSliceRowsHookSlices:
    def _run(self, ui):
        from qtpy.QtWidgets import QListWidget

        ui.segment_slice_data_list = QListWidget()
        c = _sctrl(ui)
        c.sample_records = {
            "p_XY_3": {"status": "predicted", "bucket": "lateral",
                       "patch_id": "p", "view": "XY", "slice_index": 3,
                       "updated_at": "1"},
            "p_XY_9": {"status": "curating", "bucket": "lateral",
                       "patch_id": "p", "view": "XY", "slice_index": 9,
                       "updated_at": "2"},
            "p_XY_5": {"status": "staged", "bucket": "lateral",
                       "patch_id": "p", "view": "XY", "slice_index": 5,
                       "updated_at": "3"},
            "p_XY_7": {"status": "accepted", "bucket": "lateral",
                       "patch_id": "p", "view": "XY", "slice_index": 7,
                       "updated_at": "4"},
            "p_XZ_1": {"status": "predicted", "bucket": "axial",
                       "patch_id": "p", "view": "XZ", "slice_index": 1,
                       "updated_at": "5"},
        }
        c._get_active_status_filters = lambda: {
            "predicted", "curating", "accepted", "staged", "committed"}
        c._refresh_patch_list_badges = lambda: None
        c._refresh_curation_stage_line = lambda: None
        c._build_provenance_tooltip = lambda rec: ""
        c._refresh_sample_queue()
        return c

    def test_the_hook_gets_the_listed_ids_in_the_slices_order(self, qapp):
        ui = _SliceUI()
        self._run(ui)
        calls = [p for n, p in ui.calls if n == "slice_rows"]
        (payload,) = calls
        assert payload["staged"] is False and payload["bucket"] == "lateral"
        assert payload["patch"] == "p"
        # finalized, accepted, curating (D24: ranked), predicted
        assert payload["ids"] == ["p_XY_5", "p_XY_7", "p_XY_9", "p_XY_3"]
        assert "stack_paths" in payload and "workspace" in payload

    def test_the_rows_keep_their_text_and_userrole(self, qapp):
        from qtpy.QtCore import Qt

        ui = _SliceUI()
        self._run(ui)
        lst = ui.segment_slice_data_list
        ids = [lst.item(i).data(Qt.UserRole) for i in range(lst.count())]
        assert ids == ["p_XY_5", "p_XY_7", "p_XY_9", "p_XY_3"]
        assert all(lst.item(i).text() for i in range(lst.count()))

    def test_a_ui_without_the_hook_still_fills_the_list(self, qapp):
        ui = _SliceUI()
        ui.on_slice_rows = None
        self._run(ui)
        assert ui.segment_slice_data_list.count() == 4

    def test_the_old_status_rank_dict_is_gone(self):
        src = inspect.getsource(C._refresh_sample_queue)
        assert "_STATUS_RANK" not in src and "slice_sort_key" in src


class _Layer:
    data = np.zeros((4, 5, 5), dtype="uint8")


def _flush_ctrl(ui, result):
    c = _sctrl(ui)
    c._work_labels_layer_name_for_view = lambda view_name=None: "L"
    c._layer_by_name = lambda name: _Layer()
    c.segmentation_manager = type("S", (), {"data_masks": {}})()
    c._persist_curated_training_pair = lambda sl, async_write=False: result
    c._schedule_state_save = lambda: None
    c._refresh_run_picker = lambda: None
    c._curating_persist_pending = {"p_XY_2": ("p_XY_2", "XY", 2)}
    return c


class TestMaskWriteHook:
    def test_the_flush_tells_the_ui_after_the_write(self):
        src = inspect.getsource(C._flush_curating_persist_pending)
        a = src.index("_persist_curated_training_pair(sl, async_write=True)")
        b = src.index("on_slice_written")
        assert a < b

    def test_the_hook_gets_the_slice_label(self):
        ui = _SliceUI()
        _flush_ctrl(ui, True)._flush_curating_persist_pending()
        assert ("written", "p_XY_2") in ui.calls

    def test_no_write_no_hook(self):
        ui = _SliceUI()
        _flush_ctrl(ui, False)._flush_curating_persist_pending()
        assert not [x for x in ui.calls if x[0] == "written"]

    def test_a_failing_hook_does_not_break_the_persist(self):
        class Bad(_SliceUI):
            def on_slice_written(self, sid):
                raise RuntimeError("boom")

        c = _flush_ctrl(Bad(), True)
        c._flush_curating_persist_pending()
        assert c._curating_persist_pending == {}

    def test_a_ui_without_the_hook_still_persists(self):
        ui = _SliceUI()
        ui.on_slice_written = None
        c = _flush_ctrl(ui, True)
        c._flush_curating_persist_pending()
        assert c._curating_persist_pending == {}


class TestWorkspaceOpenedAndRunSeams:
    def test_the_quiet_load_calls_the_hook_after_the_refreshes(self):
        src = inspect.getsource(C._load_workspace_quiet)
        a = src.index("self._refresh_sample_queue()")
        b = src.index("on_workspace_opened")
        assert a < b

    def test_the_run_summary_keys_ride_the_session_payload(self):
        ui = _UIWithHooks()
        c = _ctrl(ui)
        c._run_summaries_cache = [
            {"run_id": "Old", "is_active": False, "accepted_pair_count": 1},
            {"run_id": "Cell", "is_active": True, "accepted_pair_count": 83,
             "description": "whole cell"}]
        c._refresh_patch_list_badges()
        payload = ui.calls[0][1]
        assert payload["run_pairs"] == 83
        assert payload["run_desc"] == "whole cell"

    def test_no_summaries_means_none(self):
        ui = _UIWithHooks()
        _ctrl(ui)._refresh_patch_list_badges()
        payload = ui.calls[0][1]
        assert payload["run_pairs"] is None and payload["run_desc"] is None
