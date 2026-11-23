"""Delete Label acts on the brush cell inside an isolation.

With several cells isolated, the user moves the brush to one of them and
presses the navigator's Delete Label: that cell goes, not the navigator's
current row.  The current row follows the brush without firing the
navigate / select / isolation callbacks.  Outside an isolation, Delete keeps
acting on the current row.  The Painting badge shows the brush label from the
moment the annotator loads its layers.

The annotator half is Qt-free (real napari Labels layers, recorders for the
calls that would redraw); the navigator half uses a real navigator offscreen.
"""
from __future__ import annotations

import os
import types
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

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


class _Nav:
    """Navigator stand-in: the Isolate box, the selection, the current row."""

    def __init__(self, ticked=False, selected=(), current=None):
        self.box = ticked
        self.selected = set(selected)
        self.current = current
        self.calls = []

    def is_isolate_checked(self):
        return self.box

    def selected_label_ids(self):
        return set(self.selected)

    def set_current_row_silently(self, lid):
        self.calls.append(("CURRENT", lid))
        self.current = lid
        return True

    def remove_item(self, lid):
        self.calls.append(("REMOVE", lid))


def _env(mode, ids=(1, 3), brush=0, nav=None):
    """Cells 1, 2, 3, 4; an isolation of *ids* in *mode*; the brush on
    *brush*.  *mode*: "multi" (3D sub-proxies), "single" (3D proxy),
    "single2d" (the 2D show-selected isolation), "none"."""
    from napari.layers import Labels

    data = np.zeros((30, 30, 30), dtype=np.int32)
    data[2:6, 2:6, 2:6] = 1
    data[10:14, 10:14, 10:14] = 2
    data[20:24, 20:24, 20:24] = 3
    data[2:5, 25:28, 25:28] = 4
    main = Labels(data, name="labels")
    main.visible = False
    ann = SimpleNamespace(
        viewer=SimpleNamespace(layers=_Layers([main])),
        sync_manager=SimpleNamespace(shared_labels={"labels": data}),
        _active_labels_name="labels",
        _isolation_proxy_layer=None,
        _isolation_proxy_active=False,
        _isolation_proxy_label_id=None,
        _multi_label_proxy_layers={},
        _multi_label_proxy_set=None,
        _isolation_label_set=None,
        _isolation_active=mode != "none",
        _pre_isolation_colormap=object(),
        _fragment_refine_roi=None,
        calls=[],
    )
    _bind(ann, "_layer_by_name", "_iso_refresh", "_iso_forget", "_iso_live_ids",
          "_label_crop", "_build_single_label_subproxy",
          "_sync_all_views_after_operation", "_on_delete_label_from_navigator")
    ann._update_isolation_proxy = lambda *a, **k: ann.calls.append(("UP", a[0]))
    ann._on_isolation_changed = (
        lambda ids, **k: ann.calls.append(("OI", frozenset(ids))))
    ann._propagate_multi_label_colormap_to_subviewers = (
        lambda ids: ann.calls.append(("CMAP", frozenset(ids))))
    ann._propagate_colormap_to_subviewers = (
        lambda cm: ann.calls.append(("MAINCMAP", cm)))
    ann._refresh_label_data_in_subviewers = lambda: None
    ann._label_action_buttons_active = lambda: True
    ann._snapshot_label_roi = lambda vol, lid: ((slice(0, 30),) * 3, vol.copy())
    ann.staged_op_kind = lambda: None
    # The brush is the MAIN layer's selected label (what the Painting badge
    # shows); the side views' reader must not be consulted.
    main.selected_label = brush
    ann._get_selected_label = lambda: (_ for _ in ()).throw(
        AssertionError("the side views' brush reader was used"))
    ann._on_delete_label_cb = lambda lid, **k: ann.calls.append(("CB", lid))
    ann._navigator_widget = nav if nav is not None else _Nav()
    if mode == "multi":
        for lid in ids:
            ann._build_single_label_subproxy(main, lid)
        ann._multi_label_proxy_set = frozenset(ids)
        ann._isolation_label_set = frozenset(ids)
        ann._isolation_proxy_active = True
    elif mode == "single":
        ann._isolation_proxy_active = True
        ann._isolation_proxy_label_id = ids[0]
    elif mode == "single2d":
        main.show_selected_label = True
    ann.calls.clear()
    return ann, data


def _deleted(ann):
    return [c[1] for c in ann.calls if c[0] == "CB"]


def test_multi_isolation_deletes_the_brush_cell_and_the_row_follows():
    nav = _Nav(ticked=True, selected={1, 3}, current=1)
    ann, data = _env("multi", ids=(1, 3), brush=3, nav=nav)
    ann._on_delete_label_from_navigator(1)
    assert not (data == 3).any()
    assert (data == 1).any()
    assert _deleted(ann) == [3]
    assert ("CURRENT", 3) in nav.calls
    assert ("REMOVE", 3) in nav.calls and ("REMOVE", 1) not in nav.calls
    assert 3 not in ann._multi_label_proxy_set
    assert 1 in ann._multi_label_proxy_set


def test_controller_held_isolation_deletes_the_brush_cell():
    # The controller isolates by colour map: the annotator holds no
    # isolation of its own, the navigator's ticked selection is the set.
    nav = _Nav(ticked=True, selected={1, 3}, current=1)
    ann, data = _env("none", brush=3, nav=nav)
    assert ann._iso_live_ids() == frozenset()
    ann._on_delete_label_from_navigator(1)
    assert _deleted(ann) == [3]
    assert not (data == 3).any() and (data == 1).any()
    assert ("CURRENT", 3) in nav.calls


def test_no_isolation_deletes_the_current_row():
    nav = _Nav(ticked=False, selected={1, 3}, current=1)
    ann, data = _env("none", brush=3, nav=nav)
    ann._on_delete_label_from_navigator(1)
    assert _deleted(ann) == [1]
    assert (data == 3).any() and not (data == 1).any()
    assert not [c for c in nav.calls if c[0] == "CURRENT"]


def test_single_3d_proxy_on_another_cell_deletes_the_current_row():
    nav = _Nav(ticked=True, selected={1}, current=1)
    ann, data = _env("single", ids=(1,), brush=3, nav=nav)
    ann._on_delete_label_from_navigator(1)
    assert _deleted(ann) == [1]
    assert (data == 3).any()
    assert not [c for c in nav.calls if c[0] == "CURRENT"]


def test_brush_on_background_deletes_the_current_row():
    nav = _Nav(ticked=True, selected={1, 3}, current=1)
    ann, data = _env("multi", ids=(1, 3), brush=0, nav=nav)
    ann._on_delete_label_from_navigator(1)
    assert _deleted(ann) == [1]
    assert (data == 3).any()
    assert not [c for c in nav.calls if c[0] == "CURRENT"]


def test_single_2d_isolation_deletes_the_brush_cell():
    # The 2D isolation shows the brush label itself: no id set, no proxy.
    nav = _Nav(ticked=False, selected={1}, current=1)
    ann, data = _env("single2d", brush=3, nav=nav)
    assert ann.viewer.layers["labels"].show_selected_label is True
    ann._on_delete_label_from_navigator(1)
    assert _deleted(ann) == [3]
    assert (data == 1).any() and not (data == 3).any()
    assert ("CURRENT", 3) in nav.calls


def test_a_dead_brush_in_2d_does_not_redirect_a_second_delete():
    # After the redirected Delete the brush stays on the dead id; the next
    # Delete on row 1 must delete 1, and a no-op must say False.
    nav = _Nav(ticked=False, selected={1}, current=1)
    ann, data = _env("single2d", brush=3, nav=nav)
    assert ann._on_delete_label_from_navigator(1) is not False
    assert _deleted(ann) == [3]
    nav.calls.clear()
    assert ann._on_delete_label_from_navigator(1) is not False
    assert _deleted(ann) == [3, 1]
    assert not (data == 1).any()
    assert not [c for c in nav.calls if c[0] == "CURRENT"]
    # Nothing left to delete on row 1 and the brush is dead: a no-op.
    assert ann._on_delete_label_from_navigator(1) is False
    assert _deleted(ann) == [3, 1]


def test_every_no_op_returns_false():
    ann, data = _env("none", brush=0)
    assert ann._on_delete_label_from_navigator(77) is False   # not in volume
    ann.sync_manager.shared_labels = {}
    assert ann._on_delete_label_from_navigator(1) is False    # no volume
    ann._active_labels_name = ""
    assert ann._on_delete_label_from_navigator(1) is False    # no layer name
    assert _deleted(ann) == []


def test_a_colour_map_isolation_whose_id_set_was_cleared_does_not_redirect():
    # A split reject clears the id set while a multi colour-map isolation
    # stays on screen: the old negative test read that as the 2D isolation
    # and redirected to a hidden cell.
    nav = _Nav(ticked=True, selected={1}, current=1)
    ann, data = _env("none", brush=3, nav=nav)
    ann._isolation_active = True
    ann._isolation_label_set = None
    ann._pre_isolation_colormap = object()
    assert ann.viewer.layers["labels"].show_selected_label is False
    assert 3 not in ann._iso_live_ids()
    ann._on_delete_label_from_navigator(1)
    assert _deleted(ann) == [1]
    assert (data == 3).any()
    assert not [c for c in nav.calls if c[0] == "CURRENT"]


def test_a_ctrl_selected_row_not_yet_rendered_is_not_isolated():
    # The annotator isolates {1}; 3 is Ctrl-selected but not rendered.
    nav = _Nav(ticked=True, selected={1, 3}, current=1)
    ann, data = _env("multi", ids=(1,), brush=3, nav=nav)
    ann._on_delete_label_from_navigator(1)
    assert _deleted(ann) == [1]
    assert (data == 3).any()
    assert not [c for c in nav.calls if c[0] == "CURRENT"]


def test_a_navigator_without_the_new_methods_still_deletes_the_row():
    class _OldNav:
        def is_isolate_checked(self):
            return True

        def remove_item(self, lid):
            pass

    ann, data = _env("none", brush=3, nav=_OldNav())
    ann._on_delete_label_from_navigator(1)
    assert _deleted(ann) == [1]


# --- the navigator (real widget, offscreen) ---------------------------------

@pytest.fixture(scope="module")
def qapp():
    from qtpy.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _nav(qapp, n=8):
    from napari_epicycle.annotator.curation_label_navigator import (
        CurationLabelNavigator)

    nav = CurationLabelNavigator()
    nav.set_items([
        {"label_id": i, "tier": "review", "centroid": (i, i, i), "area": 10 * i}
        for i in range(1, n + 1)
    ])
    for _ in range(5):
        qapp.processEvents()
    return nav


def _row_of(nav, lid):
    from qtpy.QtCore import Qt
    from napari_epicycle.annotator.curation_label_navigator import _COL_ID

    for r in range(nav._table.rowCount()):
        data = nav._table.item(r, _COL_ID).data(Qt.UserRole)
        if data and data.get("label_id") == lid:
            return r
    raise AssertionError(lid)


def _set_current(nav, lid):
    from qtpy.QtCore import QItemSelectionModel

    nav._table.selectionModel().setCurrentIndex(
        nav._table.model().index(_row_of(nav, lid), 0),
        QItemSelectionModel.NoUpdate)


def _record_callbacks(nav):
    fired = []
    nav._on_navigate_cb = lambda *a, **k: fired.append("navigate")
    nav._on_select_label_cb = lambda *a, **k: fired.append("select")
    nav._on_isolation_changed_cb = lambda *a, **k: fired.append("isolation")
    nav._on_selection_changed_cb = lambda *a, **k: fired.append("selection")
    return fired


def test_set_current_row_silently_keeps_the_selection_and_fires_nothing(qapp):
    nav = _nav(qapp)
    nav.select_rows_silently({2, 5, 7})
    _set_current(nav, 2)
    assert nav.current_label_id() == 2
    fired = _record_callbacks(nav)
    assert nav.set_current_row_silently(5) is True
    assert nav.current_label_id() == 5
    assert nav.selected_label_ids() == {2, 5, 7}
    assert fired == []
    assert "Label 5" in nav._editing_header.text()


def test_set_current_row_silently_on_a_missing_label_changes_nothing(qapp):
    nav = _nav(qapp)
    nav.select_rows_silently({2, 5})
    _set_current(nav, 2)
    assert nav.set_current_row_silently(99) is False
    assert nav.current_label_id() == 2
    assert nav.selected_label_ids() == {2, 5}


def test_a_redirected_row_removed_keeps_the_rest_of_the_selection(qapp):
    # What the deferred rebuild does after a redirected Delete: the row the
    # cursor moved to goes, the other selected rows stay selected.
    from qtpy.QtCore import QCoreApplication, QEvent

    nav = _nav(qapp)
    nav.select_rows_silently({2, 5, 7})
    _set_current(nav, 2)
    fired = _record_callbacks(nav)
    assert nav.set_current_row_silently(5) is True
    nav.remove_item(5)
    for _ in range(5):
        qapp.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not nav._rebuild_pending
    assert nav.selected_label_ids() == {2, 7}
    assert nav.current_label_id() in {2, 7}
    assert [f for f in fired if f != "selection"] == []


def test_painting_badge_follows_a_layer_set_after_a_none(qapp):
    from napari.layers import Labels

    nav = _nav(qapp)
    layer = Labels(np.zeros((4, 4, 4), dtype=np.int32), name="labels")
    nav.set_label_layer(None)
    nav.set_label_layer(layer)
    layer.selected_label = 48
    assert nav._editing_header.text().startswith("Painting: 48")


def test_loading_the_layers_hands_the_labels_layer_to_the_navigator(
        qapp, monkeypatch):
    # At launch the labels layer name is only known once the base class has
    # loaded the layers; the badge must subscribe then, not at the first QC
    # refresh.
    from napari.layers import Labels
    from napari_epicycle.annotator import native_multiview_widget as NMW

    layer = Labels(np.zeros((4, 4, 4), dtype=np.int32), name="labels")
    layer.selected_label = 48

    def _base_load(self):
        self._active_labels_name = "labels"

    monkeypatch.setattr(
        NMW.NativeMultiviewAnnotator, "_load_selected_layers", _base_load)
    for name in ("_iso_drop", "_teardown_refine_for_switch",
                 "_remove_add_candidate", "_remove_add_box_shapes",
                 "_clear_draw_snapshot", "_refresh_merge_button_state",
                 "_refresh_label_action_buttons_state",
                 "_apply_smart_step_size", "_install_arrow_key_filters",
                 "_sync_show_selected_label", "_connect_label_pinning"):
        monkeypatch.setattr(_M, name, lambda self, *a, **k: None)

    ann = _M.__new__(_M)
    ann.viewer = SimpleNamespace(layers=_Layers([layer]))
    ann.sync_manager = SimpleNamespace(_suppress_show_selected_sync=False)
    ann._active_labels_name = ""
    ann._smooth_running = False
    ann._split_running = False
    ann._add_running = False
    ann._add_review = False
    ann._colormap_authority = None
    # An uninitialised Qt wrapper raises on a missing attribute (even inside
    # getattr with a default), so every attribute the method reads is set.
    ann._isolation_proxy_active = False
    ann._isolation_proxy_label_id = None
    ann._isolation_label_set = None
    ann._isolation_active = False
    ann._paint_isolation_active = False
    ann._paint_isolation_label_id = None
    ann._pre_isolation_colormap = None
    ann._pre_isolation_data = None
    ann._interp_refine_mode = False
    nav = _nav(qapp)
    nav.set_label_layer(None)
    ann._navigator_widget = nav

    _M._load_selected_layers(ann)

    assert nav._main_label_layer is layer
    assert nav._editing_header.text().startswith("Painting: 48")
    layer.selected_label = 7
    assert nav._editing_header.text().startswith("Painting: 7")


def test_a_redirect_whose_row_is_filtered_out_returns_false():
    """The brush cell is deleted, but the navigator could not move the
    highlighted row onto it (the row is filtered out of the table): the
    handler returns False so the navigator does not auto-advance past the
    untouched highlighted row."""
    nav = _Nav(ticked=True, selected={1, 3}, current=1)
    nav.set_current_row_silently = lambda lid: (
        nav.calls.append(("CURRENT", lid)) or False)
    ann, data = _env("multi", (1, 3), brush=3, nav=nav)
    assert ann._on_delete_label_from_navigator(1) is False
    assert _deleted(ann) == [3]
    assert (data == 3).sum() == 0 and (data == 1).sum() > 0


def test_a_bad_id_returns_false():
    ann, _ = _env("none")
    assert ann._on_delete_label_from_navigator("x") is False
    assert ann._on_delete_label_from_navigator(0) is False
    assert _deleted(ann) == []
