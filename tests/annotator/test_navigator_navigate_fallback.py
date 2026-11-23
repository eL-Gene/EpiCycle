"""The navigator pans the side views to the cell the user picked.

The saved table position can describe a volume that no longer exists
(edits discarded or never saved).  ``_navigator_navigate`` therefore checks
the position against the labels in memory and, when it is not inside the
cell, pans to a voxel of the cell itself.  A right-click on a label with
no selectable table row pans the same way.

The real methods are bound onto a plain namespace (a ``Curation3DAnnotator``
cannot be built without Qt); no widget is built.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from napari_epicycle.annotator import curation_3d_widget as _c3d
from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator


def _make(data, name="curation_3d_p"):
    ann = SimpleNamespace()
    calls = []
    ann.navigate_to_slice = lambda orient, idx: calls.append((orient, int(idx)))
    ann.viewer = SimpleNamespace(layers={name: SimpleNamespace(data=data)})
    ann._active_labels_name = name
    ann._navigator_navigate = Curation3DAnnotator._navigator_navigate.__get__(ann)
    return ann, calls


def _zyx(calls):
    d = dict(calls)
    return (d["XY"], d["XZ"], d["YZ"])


def _vol():
    v = np.zeros((20, 40, 40), dtype=np.uint32)
    v[5:8, 10:14, 10:14] = 7
    v[12:15, 25:30, 25:30] = 9
    return v


def test_point_inside_cell_pans_to_table_position(capsys):
    ann, calls = _make(_vol())
    ann._navigator_navigate(7, (6, 11, 12))
    assert _zyx(calls) == (6, 11, 12)
    assert len(calls) == 3
    assert "[NAV-FIX]" not in capsys.readouterr().out


def test_point_on_another_label_pans_inside_the_cell(capsys):
    v = _vol()
    ann, calls = _make(v)
    ann._navigator_navigate(7, (13, 27, 27))  # sits on label 9
    assert len(calls) == 3
    z, y, x = _zyx(calls)
    assert v[z, y, x] == 7
    out = capsys.readouterr().out
    assert out.count("[NAV-FIX] label 7: the table position (13,27,27) "
                     "is not inside the cell; panned to (") == 1


def test_out_of_bounds_centroid_falls_back(capsys):
    v = _vol()
    ann, calls = _make(v)
    ann._navigator_navigate(7, (99, 99, 99))
    z, y, x = _zyx(calls)
    assert v[z, y, x] == 7
    assert capsys.readouterr().out.count("[NAV-FIX]") == 1


def test_none_centroid_falls_back(capsys):
    v = _vol()
    ann, calls = _make(v)
    ann._navigator_navigate(9, None)
    z, y, x = _zyx(calls)
    assert v[z, y, x] == 9
    assert capsys.readouterr().out.count("[NAV-FIX] label 9 has no table position; panned to") == 1


def test_label_absent_with_centroid_pans_to_centroid(capsys):
    ann, calls = _make(_vol())
    ann._navigator_navigate(55, (3, 4, 5))
    assert _zyx(calls) == (3, 4, 5)
    assert ("[NAV-FIX] label 55 is not in the volume; panned to the table "
            "position (3,4,5)") in capsys.readouterr().out


def test_label_absent_without_centroid_does_not_pan(capsys):
    ann, calls = _make(_vol())
    ann._navigator_navigate(55, None)
    assert calls == []
    assert "[NAV-FIX] label 55 is not in the volume; no pan" in (
        capsys.readouterr().out
    )


def test_c_shaped_cell_gets_a_point_inside_itself():
    v = np.zeros((9, 40, 40), dtype=np.uint32)
    v[4, 5:35, 5:9] = 3      # left bar
    v[4, 5:9, 5:35] = 3      # top bar
    v[4, 31:35, 5:35] = 3    # bottom bar (a C open to the right)
    ann, calls = _make(v)
    # the mean of the voxels sits in the hollow, outside the cell
    ann._navigator_navigate(3, (0, 0, 0))
    z, y, x = _zyx(calls)
    assert v[z, y, x] == 3


def test_empty_active_name_pans_to_centroid_as_before(capsys):
    ann, calls = _make(_vol())
    ann._active_labels_name = ""
    ann._navigator_navigate(7, (13, 27, 27))
    assert _zyx(calls) == (13, 27, 27)
    assert "[NAV-FIX]" not in capsys.readouterr().out


def test_empty_active_name_without_centroid_returns():
    ann, calls = _make(_vol())
    ann._active_labels_name = ""
    ann._navigator_navigate(7, None)
    assert calls == []


# ---- the deferred right-click select -------------------------------------

class _Nav:
    def __init__(self, rows, items=None):
        self._items = [{"label_id": r} for r in (rows if items is None else items)]
        self._table = _Table(rows)
        self.selected = []

    def is_right_click_select_enabled(self):
        return True

    def select_label(self, label_id):
        self.selected.append(label_id)


class _Item:
    def __init__(self, label_id):
        self._d = {"label_id": label_id}

    def data(self, role):
        return self._d


class _Table:
    def __init__(self, rows):
        self._rows = list(rows)

    def rowCount(self):
        return len(self._rows)

    def item(self, r, c):
        return _Item(self._rows[r])


def _click(rows, clicked_label, monkeypatch, items=None, data=None,
           name="x_XY", position=(0, 0), rebuild_pending=False):
    queue = []
    monkeypatch.setattr(_c3d, "_defer_to_event_loop", queue.append)
    nav = _Nav(rows, items)
    nav._rebuild_pending = rebuild_pending
    ann = SimpleNamespace(
        _is_closing=False,
        _navigator_widget=nav,
        _refine_state=None,
        _interp_running=False, _split_running=False, _smooth_running=False,
        _merge_dialog=None,
    )
    ann._right_click_navigate_allowed = (
        Curation3DAnnotator._right_click_navigate_allowed.__get__(ann)
    )
    ann._rclick_select_alive = Curation3DAnnotator._rclick_select_alive.__get__(ann)
    navigated = []
    ann._navigator_navigate = lambda lid, c: navigated.append((lid, c))
    cb = Curation3DAnnotator._make_subviewer_right_click_handler(ann)
    layer = SimpleNamespace(
        data=np.array([[clicked_label]]) if data is None else data,
        mode="pan_zoom", name=name,
    )
    ev = SimpleNamespace(button=2, position=position, type="mouse_press")
    gen = cb(layer, ev)
    next(gen)
    ev.type = "mouse_release"
    with pytest.raises(StopIteration):
        next(gen)
    # nothing ran inside the mouse handler
    assert navigated == [] and ann._navigator_widget.selected == []
    for fn in queue:
        fn()
    return ann, navigated


def test_right_click_label_without_row_navigates_without_centroid(monkeypatch):
    ann, navigated = _click([1, 2], 7, monkeypatch)
    assert navigated == [(7, None)]
    assert ann._navigator_widget.selected == []


def test_right_click_label_with_row_selects_it(monkeypatch):
    ann, navigated = _click([1, 7], 7, monkeypatch)
    assert ann._navigator_widget.selected == [7]
    assert navigated == []


def test_right_click_label_in_items_but_hidden_by_filter(monkeypatch):
    ann, navigated = _click([1], 7, monkeypatch, items=[1, 7])
    assert navigated == [(7, None)]
    assert ann._navigator_widget.selected == []


def test_nan_centroid_falls_back(capsys):
    v = _vol()
    ann, calls = _make(v)
    ann._navigator_navigate(7, (float("nan"), 1, 1))
    z, y, x = _zyx(calls)
    assert v[z, y, x] == 7
    assert capsys.readouterr().out.count("[NAV-FIX]") == 1


def test_centroid_on_last_index_clips_cube_and_finds_label(capsys):
    v = _vol()
    ann, calls = _make(v)
    ann._navigator_navigate(7, (19, 39, 39))  # last voxel, label elsewhere
    z, y, x = _zyx(calls)
    assert v[z, y, x] == 7
    assert capsys.readouterr().out.count("[NAV-FIX]") == 1


def test_rebuild_pending_drops_the_no_row_navigation(monkeypatch):
    ann, navigated = _click([1, 2], 7, monkeypatch, rebuild_pending=True)
    assert navigated == []
    assert ann._navigator_widget.selected == []


def _click3(layer_name, idx, monkeypatch):
    data = np.zeros((4, 5, 6), dtype=np.uint32)
    data[idx] = 7
    _, navigated = _click(
        [1, 2], 7, monkeypatch, data=data, name=layer_name, position=idx
    )
    return navigated


def test_click_voxel_in_xy_layer_is_passed_as_is(monkeypatch):
    assert _click3("p_XY", (1, 2, 3), monkeypatch) == [(7, (1, 2, 3))]


def test_click_voxel_in_xz_layer_swaps_first_two_axes(monkeypatch):
    # (a, b, c) in the XZ view is main (b, a, c)
    assert _click3("p_XZ", (1, 2, 3), monkeypatch) == [(7, (2, 1, 3))]


def test_click_voxel_in_yz_layer_reverses_the_axes(monkeypatch):
    # (a, b, c) in the YZ view is main (c, b, a)
    assert _click3("p_YZ", (1, 2, 3), monkeypatch) == [(7, (3, 2, 1))]


def test_layer_name_without_a_suffix_passes_none(monkeypatch):
    assert _click3("plain", (1, 2, 3), monkeypatch) == [(7, None)]
