"""Above a size limit, 3D multi-isolation colours the main layer.

One small layer per isolated cell costs about half a second each to add and
again to remove, so a large set (a lasso of a few hundred cells) would freeze
the window.  Up to ``ISO_PROXY_MAX_CELLS`` cells the 3D view keeps one layer
per cell; above it the main layer gets one colour map instead (the same state
the 2D view already uses).  These tests pin the choice, the crossing in both
directions, and the readers that must see a colour-map isolation.  Qt-free
fakes with a real, tiny napari Labels layer as the main layer.

Open risk: the colour-map look for many isolated cells in 3D must be checked
on screen on napari 0.8; a 2026-05 bug made the plugin switch to per-cell
layers.

Known limits, written down and not fixed here:
- A cell that joins a colour-map isolation is recoloured (3D only).
- Forgetting cells keeps the colour map even when the set drops below the
  limit again.
- The forget branch also runs in 2D: deleting the last of several isolated
  cells now clears the isolation there.
"""
from __future__ import annotations

import types
from types import SimpleNamespace

import numpy as np
import pytest

from napari.layers import Labels

from napari_epicycle.annotator import curation_3d_widget as W
from napari_epicycle.annotator._isolation_bracket import DEFAULT_PIECES_CAP
from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator
from napari_epicycle.annotator.refinement_state import StagedOp

from .test_isolation_bracket_ops import _Layers, _Nav, _bind_helpers

_K = Curation3DAnnotator
K = W.ISO_PROXY_MAX_CELLS if hasattr(W, "ISO_PROXY_MAX_CELLS") else 2


def _fake(ndisplay=3, real_colormap=False):
    main = Labels(np.zeros((4, 4, 4), dtype=np.uint32), name="labels")
    ann = SimpleNamespace(
        viewer=SimpleNamespace(layers=_Layers([main]),
                               dims=SimpleNamespace(ndisplay=ndisplay)),
        _active_labels_name="labels",
        _isolation_proxy_layer=None, _isolation_proxy_active=False,
        _isolation_proxy_label_id=None, _multi_label_proxy_layers={},
        _multi_label_proxy_set=None, _isolation_label_set=None,
        _isolation_active=False, _pre_isolation_colormap=None,
        _colormap_authority=None, _iso_bracket=None, sync_manager=None,
        _split_label_id=1, _interp_label_id=1,
        _navigator_widget=_Nav(ticked=True),
        _on_isolation_cleared_cb=None,
        calls=[], main=main, builds=[], colormap_calls=[], oi=[],
    )
    _bind_helpers(ann)
    for name in ("_on_isolation_changed", "_apply_multi_label_proxy",
                 "_layer_by_name", "_iso_forget", "_iso_refresh"):
        setattr(ann, name, types.MethodType(getattr(_K, name), ann))
    ann.staged_kinds = lambda: frozenset({StagedOp.RESHAPE})
    ann._get_advanced_features_widget = lambda: None

    def _build(main_layer, lid, crop=None):
        ann.builds.append(int(lid))
        ann._multi_label_proxy_layers[int(lid)] = SimpleNamespace(
            name=f"p{lid}", visible=True)

    def _colormap(main_layer, ids):
        ann.colormap_calls.append(frozenset(ids))
        ann._isolation_label_set = frozenset(int(i) for i in ids)

    def _remove(restore_visibility=True):
        ann.calls.append(("REMOVE", restore_visibility))
        ann._multi_label_proxy_layers = {}
        ann._multi_label_proxy_set = None
        ann._isolation_proxy_active = False

    ann._build_single_label_subproxy = _build
    ann._apply_multi_isolation_colormap = _colormap
    ann._remove_isolation_proxy = _remove
    if real_colormap:
        ann._apply_multi_isolation_colormap = types.MethodType(
            _K._apply_multi_isolation_colormap, ann)
    ann._propagate_show_selected_label = lambda v, **k: 0
    ann._propagate_multi_label_colormap_to_subviewers = lambda s: 0
    ann._propagate_colormap_to_subviewers = lambda cm: 0
    return ann


def _ids(n):
    return set(range(1, n + 1))


def test_k_is_a_sane_int():
    assert isinstance(W.ISO_PROXY_MAX_CELLS, int)
    assert W.ISO_PROXY_MAX_CELLS == 2


def test_small_set_in_3d_builds_proxies():
    ann = _fake()
    ann._on_isolation_changed(_ids(K))
    assert sorted(ann.builds) == sorted(_ids(K))
    assert ann.colormap_calls == []
    assert ann._isolation_proxy_active


def test_large_set_in_3d_uses_the_colormap_not_proxies():
    ann = _fake()
    ids = _ids(K + 4)
    ann._on_isolation_changed(ids)
    assert ann.builds == []
    assert ann.colormap_calls == [frozenset(ids)]
    assert ann._isolation_label_set == frozenset(ids)
    assert not ann._isolation_proxy_active
    assert ann._multi_label_proxy_set is None
    assert ann._multi_label_proxy_layers == {}
    assert ann._isolation_active


def test_crossing_up_tears_down_existing_proxies():
    ann = _fake()
    ann._on_isolation_changed(_ids(K))
    assert ann._multi_label_proxy_layers
    ann._on_isolation_changed(_ids(K + 5))
    assert ("REMOVE", False) in ann.calls
    assert ann._multi_label_proxy_layers == {}
    assert ann.colormap_calls == [frozenset(_ids(K + 5))]


def _capture_timer(monkeypatch):
    queued = []
    from qtpy import QtCore
    monkeypatch.setattr(QtCore.QTimer, "singleShot",
                        staticmethod(lambda ms, cb: queued.append(cb)))
    return queued


def test_crossing_up_with_an_equal_cached_set_still_recolours(monkeypatch):
    queued = _capture_timer(monkeypatch)
    ann = _fake(real_colormap=True)
    ids = _ids(K + 4)
    ann._on_isolation_changed(_ids(K))
    # The cache grew to the incoming set (a refresh does this) while the
    # per-cell layers are still up.
    ann._isolation_label_set = frozenset(ids)
    ann._on_isolation_changed(ids)
    assert ("REMOVE", False) in ann.calls
    assert len(queued) == 1
    assert hasattr(ann.main.colormap, "color_dict")
    queued[0]()
    assert ann.main.visible is True


def test_the_queued_reshow_is_skipped_after_a_token_bump(monkeypatch):
    queued = _capture_timer(monkeypatch)
    ann = _fake(real_colormap=True)
    ann._on_isolation_changed(_ids(K + 4))
    assert len(queued) == 1
    assert ann.main.visible is False
    ann._perf_reshow_token = getattr(ann, "_perf_reshow_token", 0) + 1
    queued[0]()
    assert ann.main.visible is False


def test_the_queued_reshow_shows_main_without_a_bump(monkeypatch):
    queued = _capture_timer(monkeypatch)
    ann = _fake(real_colormap=True)
    ann._on_isolation_changed(_ids(K + 4))
    queued[0]()
    assert ann.main.visible is True


def test_crossing_down_cancels_the_queued_reshow(monkeypatch):
    queued = _capture_timer(monkeypatch)
    ann = _fake(real_colormap=True)
    ann._on_isolation_changed(_ids(K + 5))
    assert len(queued) == 1
    ann._on_isolation_changed(_ids(K))
    assert sorted(ann.builds) == sorted(_ids(K))
    assert ann.main.visible is False
    queued[0]()
    assert ann.main.visible is False


def test_live_ids_see_a_colormap_isolation():
    ann = _fake()
    ann._isolation_active = True
    ann._isolation_label_set = frozenset(_ids(K + 4))
    assert ann._iso_live_ids() == frozenset(_ids(K + 4))


def test_live_ids_empty_when_nothing_isolated():
    assert _fake()._iso_live_ids() == frozenset()


def test_forget_shrinks_a_colormap_isolation():
    ann = _fake()
    ann._on_isolation_changed(_ids(K + 4))
    ann.colormap_calls.clear()
    ann._iso_forget({1})
    assert ann._isolation_label_set == frozenset(_ids(K + 4) - {1})
    assert ann.colormap_calls == [frozenset(_ids(K + 4) - {1})]


def test_forget_the_last_cell_clears_a_colormap_isolation():
    ann = _fake()
    ann._isolation_active = True
    ann._isolation_label_set = frozenset({7})
    ann._on_isolation_changed = lambda ids, **k: ann.oi.append(
        (frozenset(ids), k.get("reason")))
    ann._iso_forget({7})
    assert ann.oi == [(frozenset(), "forget")]


def test_forget_of_an_unrelated_id_changes_nothing_in_colormap_mode():
    ann = _fake()
    ann._on_isolation_changed(_ids(K + 4))
    ann.colormap_calls.clear()
    ann._iso_forget({9999})
    assert ann.colormap_calls == []


def test_bracket_round_trip_restores_a_large_set_by_colormap():
    ann = _fake()
    ids = _ids(12)
    ann._on_isolation_changed(ids)
    real = ann._on_isolation_changed
    ann._on_isolation_changed = lambda i, **k: (
        ann.oi.append(frozenset(i)), real(i, **k))[1]
    snap = ann._iso_begin("reshape")
    assert snap.mode == "multi" and snap.ids == frozenset(ids)
    ann._iso_suspend()
    assert not ann._isolation_active
    ann._iso_end("reshape")
    assert ann.oi == [frozenset(ids)]
    assert ann._isolation_label_set == frozenset(ids)
    assert ann.builds == []


def test_render_selected_of_three_ids_uses_the_colormap_and_two_use_proxies():
    assert K == 2
    big = _fake()
    big._on_isolation_changed({1, 2, 3})
    assert big.builds == [] and big.colormap_calls == [frozenset({1, 2, 3})]
    small = _fake()
    small._on_isolation_changed({1, 2})
    assert sorted(small.builds) == [1, 2] and small.colormap_calls == []


def _colormap_mode(ids=(1, 2, 3)):
    ann = _fake()
    ann._on_isolation_changed(set(ids))
    ann.colormap_calls.clear()
    return ann


def test_iso_refresh_adds_a_joining_cell_in_colormap_mode():
    ann = _colormap_mode()
    ann.main.data[0, 0, 0] = 4          # a joining cell has voxels
    ann._iso_refresh([4])
    assert ann._isolation_label_set == frozenset({1, 2, 3, 4})
    assert ann.colormap_calls == [frozenset({1, 2, 3, 4})]


def test_iso_refresh_of_an_already_isolated_cell_does_nothing_in_colormap_mode():
    ann = _colormap_mode()
    ann._iso_refresh([2])
    assert ann.colormap_calls == []
    assert ann._isolation_label_set == frozenset({1, 2, 3})


def _sync_ready(ann):
    shared = ann.main.data
    ann.sync_manager = SimpleNamespace(shared_labels={"labels": shared})
    ann._refresh_label_data_in_subviewers = lambda: None
    ann._get_selected_label = lambda: 0
    ann._sync_all_views_after_operation = types.MethodType(
        _K._sync_all_views_after_operation, ann)
    return ann


def test_sync_after_an_op_recolours_a_cell_that_joined_a_colormap_isolation():
    ann = _sync_ready(_colormap_mode())
    ann.main.data[0, 0, 0] = 4
    ann._sync_all_views_after_operation(label_id=4)
    assert ann._isolation_label_set == frozenset({1, 2, 3, 4})
    assert ann.colormap_calls == [frozenset({1, 2, 3, 4})]


def test_sync_after_an_op_on_an_isolated_cell_leaves_the_colormap_alone():
    ann = _sync_ready(_colormap_mode())
    ann._sync_all_views_after_operation(label_id=2)
    assert ann.colormap_calls == []
    assert ann._isolation_label_set == frozenset({1, 2, 3})


def _painted_env(monkeypatch):
    _capture_timer(monkeypatch)
    ann = _fake(real_colormap=True)
    ann._iso_refresh_painted_label = types.MethodType(
        _K._iso_refresh_painted_label, ann)
    ann._push_paint_stroke = lambda: None
    ann._split_running = False
    ann._schedule_label_alive_check = lambda: None
    ann._on_isolation_changed({1, 2, 3})
    assert ann._isolation_label_set == frozenset({1, 2, 3})
    return ann


def test_a_cell_painted_in_colour_map_mode_is_visible_at_once(monkeypatch):
    ann = _painted_env(monkeypatch)
    ann.main.data[1, 1, 1] = 99
    ann.main.selected_label = 99
    _K._on_subviewer_draw_complete(ann)
    assert 99 in ann._isolation_label_set
    assert ann.main.colormap.map(99)[3] > 0


def test_a_painted_id_with_no_voxels_does_not_join(monkeypatch):
    ann = _painted_env(monkeypatch)
    ann.main.selected_label = 98          # the next unused id
    _K._on_subviewer_draw_complete(ann)
    assert 98 not in ann._isolation_label_set


def test_the_stroke_region_is_passed_to_the_refresh(monkeypatch):
    ann = _painted_env(monkeypatch)
    roi = (slice(0, 2), slice(0, 2), slice(0, 2))
    ann.main.data[1, 1, 1] = 97
    ann.main.selected_label = 97
    ann._stroke_plane = (roi, ann.main.data[roi].copy())
    seen = []
    real = ann._iso_refresh
    ann._iso_refresh = lambda ids, **k: (seen.append(k), real(ids, **k))[1]
    _K._on_subviewer_draw_complete(ann)
    assert seen and seen[0].get("roi_hint") == roi


def test_a_stroke_on_an_isolated_cell_does_no_refresh(monkeypatch):
    ann = _painted_env(monkeypatch)
    ann.main.selected_label = 2
    called = []
    ann._iso_refresh = lambda ids, **k: called.append(ids)
    _K._on_subviewer_draw_complete(ann)
    assert called == []


def test_a_paint_event_on_main_only_schedules_the_join(monkeypatch):
    ann = _painted_env(monkeypatch)
    ann._schedule_painted_join = types.MethodType(
        _K._schedule_painted_join, ann)
    ann._join_painted_label = types.MethodType(_K._join_painted_label, ann)
    started = []

    class _Timer:
        def __init__(self, *a):
            self.timeout = SimpleNamespace(connect=lambda f: None)

        def setSingleShot(self, v):
            pass

        def setInterval(self, v):
            pass

        def start(self):
            started.append(1)

    from qtpy import QtCore
    monkeypatch.setattr(QtCore, "QTimer", _Timer)
    ann.main.data[1, 1, 1] = 7
    ann.main.selected_label = 7
    _K._on_labels_painted(ann, None)
    _K._on_labels_painted(ann, None)
    assert 7 not in ann._isolation_label_set          # nothing mid-stroke
    assert len(started) == 2                          # one pinned timer, restarted
    ann._join_painted_label()                         # the timer fires
    assert 7 in ann._isolation_label_set


def test_a_paint_event_leaves_proxy_isolation_alone():
    ann = _fake()
    ann._iso_refresh_painted_label = types.MethodType(
        _K._iso_refresh_painted_label, ann)
    ann._schedule_painted_join = types.MethodType(
        _K._schedule_painted_join, ann)
    ann._schedule_label_alive_check = lambda: None
    ann._on_isolation_changed({1, 2})
    ann.main.selected_label = 7
    before = list(ann.builds)
    _K._on_labels_painted(ann, None)
    assert ann.builds == before
    assert getattr(ann, "_paint_join_timer", None) is None
