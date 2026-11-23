"""Isolation survives an edit: refreshing and forgetting isolated cells.

With several cells isolated (3D), a Refine preview, an interpolation, a Delete
or a Batch Refine step used to collapse the isolation to one cell or leave a
deleted cell on screen.  These tests pin what reaches the proxies, the side
views and the isolation callback.  Qt-free: real napari Labels layers for the
volume and the sub-proxies, recorders for the calls that would redraw.
"""
from __future__ import annotations

import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator

_M = Curation3DAnnotator


class _Layers(list):
    def __getitem__(self, key):
        if isinstance(key, str):
            for lyr in self:
                if getattr(lyr, "name", None) == key:
                    return lyr
            raise KeyError(key)
        return super().__getitem__(key)


def _bind(ann, *names):
    for n in names:
        setattr(ann, n, types.MethodType(getattr(_M, n), ann))


def _env(mode, ids=(1, 2, 3)):
    """A volume with cells 1, 2, 3, 4 and an isolation of *ids* in *mode*."""
    from napari.layers import Labels

    data = np.zeros((30, 30, 30), dtype=np.int32)
    data[2:6, 2:6, 2:6] = 1
    data[10:14, 10:14, 10:14] = 2
    data[20:24, 20:24, 20:24] = 3
    data[2:5, 25:28, 25:28] = 4
    main = Labels(data, name="labels")
    main.visible = False
    viewer = SimpleNamespace(layers=_Layers([main]))
    ann = SimpleNamespace(
        viewer=viewer,
        sync_manager=SimpleNamespace(shared_labels={"labels": data}),
        _active_labels_name="labels",
        _isolation_proxy_layer=None,
        _isolation_proxy_active=False,
        _isolation_proxy_label_id=None,
        _multi_label_proxy_layers={},
        _multi_label_proxy_set=None,
        _isolation_label_set=None,
        _isolation_active=True,
        _fragment_refine_roi=None,
        calls=[],
    )
    _bind(ann, "_layer_by_name", "_iso_refresh", "_iso_forget", "_iso_live_ids",
          "_label_crop", "_build_single_label_subproxy")
    ann._update_isolation_proxy = lambda *a, **k: ann.calls.append(("UP", a[0], k))
    ann._on_isolation_changed = (
        lambda ids, **k: ann.calls.append(("OI", frozenset(ids), k)))
    ann._propagate_multi_label_colormap_to_subviewers = (
        lambda ids: ann.calls.append(("CMAP", frozenset(ids))))
    ann._propagate_colormap_to_subviewers = (
        lambda cm: ann.calls.append(("MAINCMAP", cm)))
    ann._refresh_label_data_in_subviewers = lambda: None
    if mode == "multi":
        for lid in ids:
            ann._build_single_label_subproxy(main, lid)
        ann._multi_label_proxy_set = frozenset(ids)
        ann._isolation_label_set = frozenset(ids)
        ann._isolation_proxy_active = True
    elif mode == "single":
        ann._isolation_proxy_active = True
        ann._isolation_proxy_label_id = ids[0]
    ann.calls.clear()
    return ann, main, data


def _kinds(ann):
    return [c[0] for c in ann.calls]


class _Nav:
    """Navigator stand-in: the Isolate box.  Nothing unticks it, because a
    clear from the annotator is not echoed back by the plot."""

    def __init__(self, ticked):
        self.box = ticked
        self.set_calls = []

    def is_isolate_checked(self):
        return self.box

    def set_isolate_checked(self, value):
        self.set_calls.append(value)
        self.box = value

    def remove_item(self, lid):
        pass


# --- refresh ---------------------------------------------------------------

def test_multi_refresh_resizes_isolated_cells_in_place_and_leaves_the_others():
    ann, main, data = _env("multi", ids=(1, 2))
    p1 = ann._multi_label_proxy_layers[1]
    p2_data = ann._multi_label_proxy_layers[2].data
    old_len = p1.data.shape[0]
    data[2:12, 2:6, 2:6] = 1                  # cell 1 grew along the first axis
    main.data[:] = data
    ann._iso_refresh([1])
    assert ann._multi_label_proxy_layers[1] is p1      # resized in place
    assert p1.data.shape[0] > old_len
    assert ann._multi_label_proxy_layers[2].data is p2_data
    assert "UP" not in _kinds(ann)            # no single-label takeover


def test_multi_refresh_of_a_cell_outside_the_set_grows_the_set():
    ann, main, data = _env("multi", ids=(1, 2))
    ann._iso_refresh([3])
    assert ann._multi_label_proxy_layers.keys() == {1, 2, 3}
    assert ann._multi_label_proxy_set == frozenset({1, 2, 3})
    assert ann._isolation_label_set == frozenset({1, 2, 3})
    assert ("CMAP", frozenset({1, 2, 3})) in ann.calls


def test_multi_refresh_of_a_cell_with_no_voxels_does_not_grow_the_set():
    ann, main, data = _env("multi", ids=(1, 2))
    ann._iso_refresh([77])
    assert ann._multi_label_proxy_set == frozenset({1, 2})
    assert 77 not in ann._multi_label_proxy_layers


def test_multi_refresh_keeps_the_multi_colormap_and_the_set():
    ann, main, data = _env("multi", ids=(1, 2, 3))
    ann._iso_refresh([2])
    assert ("CMAP", frozenset({1, 2, 3})) in ann.calls
    assert ann._multi_label_proxy_set == frozenset({1, 2, 3})
    assert ann._multi_label_proxy_layers.keys() == {1, 2, 3}


def test_single_refresh_is_the_old_forced_proxy_update():
    ann, main, data = _env("single", ids=(2,))
    ann._fragment_refine_roi = (slice(0, 5),) * 3
    ann._iso_refresh([2])
    assert ann.calls == [("UP", 2, {"force": True, "roi_hint": ann._fragment_refine_roi})]


def test_refresh_with_nothing_isolated_does_nothing():
    ann, main, data = _env("off")
    ann._isolation_proxy_active = False
    ann._iso_refresh([1])
    assert ann.calls == []


def test_refresh_of_a_cell_gone_from_the_volume_forgets_it():
    ann, main, data = _env("multi", ids=(1, 2, 3))
    gone = ann._multi_label_proxy_layers[2]
    data[data == 2] = 0
    main.data[:] = data
    ann._iso_refresh([2])
    assert gone not in ann.viewer.layers
    assert gone.visible is False              # hidden before it was removed
    assert ann._multi_label_proxy_set == frozenset({1, 3})
    assert ("CMAP", frozenset({1, 3})) in ann.calls


def test_merge_set_of_one_rebuilds_only_the_kept_cell():
    ann, main, data = _env("multi", ids=(1, 2, 3))
    # Merge leaves the isolation set stale and the proxy set at {keep}.
    for lid in (2, 3):
        ann.viewer.layers.remove(ann._multi_label_proxy_layers.pop(lid))
    ann._multi_label_proxy_set = frozenset({1})
    ann._iso_refresh([1])
    assert ann._multi_label_proxy_layers.keys() == {1}
    assert ("CMAP", frozenset({1})) in ann.calls


# --- forget ----------------------------------------------------------------

def test_forget_in_multi_removes_the_sub_proxy_and_the_id():
    ann, main, data = _env("multi", ids=(1, 2, 3))
    gone = ann._multi_label_proxy_layers[2]
    ann._iso_forget([2])
    assert gone not in ann.viewer.layers and gone.visible is False
    assert ann._multi_label_proxy_set == frozenset({1, 3})
    assert ann._isolation_label_set == frozenset({1, 3})
    assert ("CMAP", frozenset({1, 3})) in ann.calls
    assert "OI" not in _kinds(ann)


def test_forget_of_the_last_multi_id_unticks_the_isolation():
    ann, main, data = _env("multi", ids=(2,))
    ann._iso_forget([2])
    assert ("OI", frozenset(), {"reason": "forget"}) in ann.calls
    assert ann._multi_label_proxy_layers == {}


def test_forget_in_single_clears_only_when_the_isolated_cell_goes():
    ann, main, data = _env("single", ids=(2,))
    ann._iso_forget([3])
    assert ann.calls == []
    ann._iso_forget([2])
    assert ann.calls == [("OI", frozenset(), {"reason": "forget"})]


# --- the shared sync and the doors that use it -------------------------------

def _sync_env(mode, ids=(1, 2, 3)):
    ann, main, data = _env(mode, ids)
    ann._pre_isolation_colormap = object()
    ann._isolation_active = True
    ann._get_selected_label = lambda: 0
    _bind(ann, "_sync_all_views_after_operation")
    return ann, main, data


def test_sync_in_multi_mode_keeps_the_set_and_the_side_view_colormap():
    ann, main, data = _sync_env("multi")
    ann._sync_all_views_after_operation(label_id=2)
    assert "UP" not in _kinds(ann)
    assert ann._multi_label_proxy_set == frozenset({1, 2, 3})
    assert ("CMAP", frozenset({1, 2, 3})) in ann.calls
    assert "MAINCMAP" not in _kinds(ann)      # the main layer colormap is not pushed


def test_sync_in_single_mode_is_the_old_forced_update():
    ann, main, data = _sync_env("single", ids=(2,))
    ann._sync_all_views_after_operation(label_id=2)
    assert ("UP", 2, {"force": True, "roi_hint": None}) in ann.calls
    assert "CMAP" not in _kinds(ann)


def test_sync_that_skips_the_3d_refresh_leaves_the_proxies_alone():
    ann, main, data = _sync_env("multi")
    ann._iso_refresh = MagicMock()
    ann._sync_all_views_after_operation(label_id=2, skip_3d_refresh=True)
    ann._iso_refresh.assert_not_called()


def _delete_env(mode, ids):
    ann, main, data = _sync_env(mode, ids)
    # Delete reads the brush from the main layer; on background (0) it acts
    # on the row it is given, which is what these tests drive.
    main.selected_label = 0
    ann._label_action_buttons_active = lambda: True
    ann._snapshot_label_roi = lambda vol, lid: ((slice(0, 30),) * 3, vol.copy())
    ann.staged_op_kind = lambda: None
    ann._navigator_widget = _Nav(False)
    ann._on_delete_label_cb = lambda lid, **k: ann.calls.append(("CB", lid))
    real_sync = ann._sync_all_views_after_operation

    def _sync(**k):
        ann.calls.append(("SYNC", k["label_id"]))
        real_sync(**k)

    ann._sync_all_views_after_operation = _sync
    _bind(ann, "_on_delete_label_from_navigator")
    return ann, main, data


def test_delete_forgets_the_cell_after_the_sync_and_before_the_controller():
    ann, main, data = _delete_env("multi", (1, 2, 3))
    real_forget = ann._iso_forget

    def _forget(ids):
        ann.calls.append(("FORGET", list(ids)))
        real_forget(ids)

    ann._iso_forget = _forget
    ann._on_delete_label_from_navigator(2)
    order = [c[0] for c in ann.calls if c[0] in ("SYNC", "FORGET", "CB")]
    # The sync may already have forgotten the missing cell; the explicit call
    # that follows is then a no-op.  Either way both come before the callback.
    assert order[0] == "SYNC" and order[-1] == "CB"
    assert "FORGET" in order[1:-1]


def test_deleted_cell_is_in_no_proxy_or_id_set_in_multi_mode():
    ann, main, data = _delete_env("multi", (1, 2, 3))
    gone = ann._multi_label_proxy_layers[2]
    ann._on_delete_label_from_navigator(2)
    assert gone not in ann.viewer.layers
    assert 2 not in ann._multi_label_proxy_layers
    assert 2 not in ann._multi_label_proxy_set
    assert 2 not in ann._isolation_label_set


def test_deleting_the_single_isolated_cell_clears_synchronously_and_leaves_the_box():
    ann, main, data = _delete_env("single", (2,))
    nav = ann._navigator_widget = _Nav(True)
    ann._on_delete_label_from_navigator(2)
    assert ("OI", frozenset(), {"reason": "forget"}) in ann.calls
    assert nav.box is True and nav.set_calls == []


def test_deleting_the_last_multi_isolated_cell_clears_synchronously():
    ann, main, data = _delete_env("multi", (2,))
    nav = ann._navigator_widget = _Nav(True)
    ann._on_delete_label_from_navigator(2)
    assert ("OI", frozenset(), {"reason": "forget"}) in ann.calls
    assert nav.box is True and nav.set_calls == []
    order = [c[0] for c in ann.calls if c[0] in ("SYNC", "OI", "CB")]
    assert order.index("OI") < order.index("CB")


def test_forget_clear_comes_before_the_navigator_advances():
    # The navigator advances after the delete callback returns; whatever it
    # isolates then is not touched, because the clear already ran.
    ann, main, data = _delete_env("single", (2,))
    ann._navigator_widget = _Nav(True)
    ann._on_delete_label_from_navigator(2)
    n_before = [c for c in ann.calls if c[0] == "OI"]
    ann._isolation_proxy_label_id = 3          # the advance isolates the next row
    assert [c for c in ann.calls if c[0] == "OI"] == n_before


# --- Batch Refine ----------------------------------------------------------

def _batch_env(mode, ids, touched):
    ann, main, data = _env(mode, ids)
    ann._iso_refresh = MagicMock()
    ann._sync_all_views_after_operation = lambda **k: None
    ann._batch_labels_name = "labels"
    ann._batch_is_stale = lambda what: False
    ann._batch_thread_cleanup = lambda: None
    ann._live_batch_orphans = lambda: False
    ann._fire_batch_refine_lock = lambda v: None
    ann._restore_ready_to_draw_state = lambda: None
    ann._sync_refine_state = lambda: None
    ann._refresh_merge_button_state = lambda: None
    ann._refresh_label_action_buttons_state = lambda: None
    ann._teardown_batch_dialog = lambda: None
    ann._batch_refine_dialog = MagicMock()
    ann._batch_touched_ids = list(touched)
    ann._batch_union_roi = (slice(0, 30),) * 3
    ann._batch_snapshot = data.copy()
    ann._batch_staged = True
    ann._batch_running = True
    _bind(ann, "_on_batch_worker_finished", "_on_batch_reject")
    return ann, main, data


def test_batch_run_refreshes_the_touched_cells_that_are_isolated():
    ann, main, data = _batch_env("multi", (1, 2), touched=[])
    result = SimpleNamespace(
        cancelled=False, touched_ids=[2, 3, 4], emptied_ids=[], absent_ids=[],
        stopped={}, union_roi=(slice(0, 30),) * 3, working_volume=data.copy(),
        n_changed_total=5)
    ann._on_batch_worker_finished(result)
    ann._iso_refresh.assert_called_once()
    assert sorted(ann._iso_refresh.call_args[0][0]) == [2]


def test_batch_reject_refreshes_the_touched_cells_that_are_isolated():
    ann, main, data = _batch_env("multi", (1, 2), touched=[1, 3])
    ann._on_batch_reject()
    ann._iso_refresh.assert_called_once()
    assert sorted(ann._iso_refresh.call_args[0][0]) == [1]


# --- the deferred re-show after an isolation clear -----------------------------

def _clear_env():
    """Single isolation of cell 2, cleared through the real isolation handler."""
    ann, main, data = _env("single", ids=(2,))
    shots = []
    ann._single_shot_in_session = lambda ms, cb: shots.append(cb)
    ann._colormap_authority = None
    ann._on_isolation_cleared_cb = None
    ann._pre_isolation_colormap = None
    ann._propagate_show_selected_label = lambda *a, **k: None
    ann.viewer.dims = SimpleNamespace(ndisplay=3)
    ann.sync_manager = SimpleNamespace(shared_labels={}, _suppress_show_selected_sync=True)
    proxy_removed = []

    def _remove(restore_visibility=True):
        proxy_removed.append(restore_visibility)
        ann._isolation_proxy_active = False
        ann._isolation_proxy_label_id = None

    ann._remove_isolation_proxy = _remove
    _bind(ann, "_on_isolation_changed")
    return ann, main, shots


def test_reshow_after_a_clear_shows_main_when_no_new_proxy_exists():
    ann, main, shots = _clear_env()
    ann._on_isolation_changed(set())
    assert len(shots) == 1 and main.visible is False
    shots[0]()
    assert main.visible is True


def test_reshow_after_a_clear_keeps_main_hidden_under_a_new_proxy():
    ann, main, shots = _clear_env()
    ann._on_isolation_changed(set())
    # Auto-advance isolates the next row before the queued re-show runs.
    ann._isolation_proxy_active = True
    ann._isolation_proxy_label_id = 3
    shots[0]()
    assert main.visible is False


# --- Close while a batch is staged --------------------------------------------

def test_batch_close_while_staged_refreshes_the_isolated_touched_cells():
    ann, main, data = _batch_env("multi", (1, 2), touched=[1, 3])
    _bind(ann, "_on_batch_close")
    ann._on_batch_close()
    ann._iso_refresh.assert_called_once()
    assert sorted(ann._iso_refresh.call_args[0][0]) == [1]


# --- scanning only inside a region ----------------------------------------------

def test_crop_with_a_region_equals_the_full_scan_for_a_cell_inside_it():
    ann, main, data = _env("off")
    hint = (slice(8, 18), slice(8, 18), slice(8, 18))   # holds cell 2 with room
    full = ann._label_crop(main, 2)
    near = ann._label_crop(main, 2, roi_hint=hint)
    assert near[0] == full[0]
    assert np.array_equal(near[1], full[1])


def test_crop_with_a_region_the_cell_leaves_falls_back_to_the_full_scan():
    ann, main, data = _env("off")
    hint = (slice(10, 12), slice(10, 12), slice(10, 12))  # inside cell 2 only
    full = ann._label_crop(main, 2)
    near = ann._label_crop(main, 2, roi_hint=hint)
    assert near[0] == full[0]
    # A cell elsewhere that the region does not contain is still found.
    assert ann._label_crop(main, 3, roi_hint=hint)[0] == ann._label_crop(main, 3)[0]
    assert ann._label_crop(main, 77, roi_hint=hint) is None


def test_sync_passes_its_region_to_the_refresh():
    ann, main, data = _sync_env("multi")
    ann._iso_refresh = MagicMock()
    roi = (slice(0, 20),) * 3
    ann._sync_all_views_after_operation(roi=roi, label_id=2)
    assert ann._iso_refresh.call_args[1]["roi_hint"] == roi


def test_batch_run_passes_the_union_region_to_the_refresh():
    ann, main, data = _batch_env("multi", (1, 2), touched=[])
    union = (slice(0, 25),) * 3
    result = SimpleNamespace(
        cancelled=False, touched_ids=[2], emptied_ids=[], absent_ids=[],
        stopped={}, union_roi=union, working_volume=data.copy(), n_changed_total=1)
    ann._on_batch_worker_finished(result)
    assert ann._iso_refresh.call_args[1]["roi_hint"] == union


def test_refresh_prints_its_cost(capsys):
    ann, main, data = _env("multi", ids=(1, 2))
    ann._iso_refresh([1, 2])
    out = capsys.readouterr().out
    assert "[ISO-PERF]" in out and "ids=2" in out


def test_sync_in_multi_mode_sends_the_colormap_once():
    ann, main, data = _sync_env("multi")
    ann._sync_all_views_after_operation(label_id=2)
    assert _kinds(ann).count("CMAP") == 1


# --- the region scan is exact -----------------------------------------------------

def _counting_crop(ann):
    """Count calls of the crop helper, recursion included."""
    real = ann._label_crop
    ann.crop_calls = 0

    def _wrapped(*a, **k):
        ann.crop_calls += 1
        return real(*a, **k)

    ann._label_crop = _wrapped
    return real


def _tall_cell_env():
    ann, main, data = _env("off")
    data[...] = 0
    data[2:15, 2:15, 2:15] = 9          # one cell, 13 voxels on each axis
    main.data[:] = data
    return ann, main, data


def test_a_region_ending_inside_the_cell_rescans_on_every_axis():
    for axis in range(3):
        ann, main, data = _tall_cell_env()
        hint = [slice(0, 30)] * 3
        hint[axis] = slice(0, 8)            # widened to 0:13, the cell runs to 14
        full = ann._label_crop(main, 9)
        _counting_crop(ann)
        near = ann._label_crop(main, 9, roi_hint=tuple(hint))
        assert near[0] == full[0], axis
        assert ann.crop_calls == 2, axis    # the region scan, then the rescan


def test_a_region_at_the_volume_edge_does_not_rescan():
    ann, main, data = _env("off")
    data[...] = 0
    data[0:5, 0:5, 0:5] = 9
    main.data[:] = data
    _counting_crop(ann)
    near = ann._label_crop(main, 9, roi_hint=(slice(0, 8),) * 3)
    assert ann.crop_calls == 1
    assert near[0] == (slice(0, 10),) * 3


def test_refresh_scans_everything_when_the_old_box_is_outside_the_region():
    ann, main, data = _env("multi", ids=(2,))
    # A second piece of the same cell, far from the old box and from the edit.
    data[25:28, 25:28, 25:28] = 2
    main.data[:] = data
    hint = (slice(22, 30),) * 3
    full = ann._label_crop(main, 2)
    _counting_crop(ann)
    ann._iso_refresh([2], roi_hint=hint)
    proxy = ann._multi_label_proxy_layers[2]
    assert tuple(slice(int(t), int(t) + n) for t, n in
                 zip(proxy.translate, proxy.data.shape)) == full[0]
    assert ann.crop_calls == 2


# --- label 0 is never a cell ----------------------------------------------------------

def test_label_zero_is_skipped_by_refresh_and_forget():
    ann, main, data = _env("multi", ids=(1, 2))
    ann._iso_refresh([0])
    ann._iso_forget([0])
    assert ann._multi_label_proxy_set == frozenset({1, 2})
    assert ann._multi_label_proxy_layers.keys() == {1, 2}
    assert ann.calls == []
