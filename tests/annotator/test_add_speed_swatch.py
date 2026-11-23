"""The navigator's swatch colours come from a plain lookup, not napari's map.

napari's ``DirectLabelColormap.map`` on an int64 array builds a numba typed
dictionary, a compile of about a second the first time after a new colour map
(inside Accept on a whole patch and inside the main Undo).  The colour it
returns is ``color_dict[id]`` for a key it holds and the ``None`` entry (else
transparent) for any other id, so ``_batch_swatch_colors`` now reads the
dictionary directly for that colour map.  These tests hold the colours equal
to the old map path for every id of a realistic table: ids in the dictionary,
ids missing from it, the background, a ``None`` entry that is hidden and one
that is visible, and a selection-mode map (which keeps the old path).
"""

from __future__ import annotations

import numpy as np
import pytest

from napari_epicycle.annotator import curation_label_navigator as NAV


def _old_batch_swatch_colors(label_layer, label_ids):
    """The map-based function as it was, kept here as the reference."""
    from qtpy.QtGui import QColor

    from napari_epicycle.ui_shell.theme import (
        default_label_colormap,
        swatch_color_for_label,
    )

    fallback = QColor(204, 204, 204)
    result = {}
    if not label_ids:
        return result
    if label_layer is None:
        return {int(i): fallback for i in label_ids}
    cmap = getattr(label_layer, "colormap", None)
    if cmap is None or not hasattr(cmap, "map"):
        return {int(i): fallback for i in label_ids}
    ids = [int(i) for i in label_ids]
    try:
        arr = np.asarray(cmap.map(np.array(ids, dtype=np.intp))).reshape(-1, 4)
    except Exception:
        return {i: swatch_color_for_label(label_layer, i) for i in ids}
    if arr.shape[0] != len(ids):
        return {i: swatch_color_for_label(label_layer, i) for i in ids}
    hidden = []
    for row, lid in enumerate(ids):
        r, g, b, a = arr[row]
        if float(a) <= 0.001:
            hidden.append(lid)
            continue
        result[lid] = QColor(int(np.clip(float(r) * 255, 0, 255)),
                             int(np.clip(float(g) * 255, 0, 255)),
                             int(np.clip(float(b) * 255, 0, 255)))
    if hidden:
        darr = np.asarray(default_label_colormap().map(
            np.array(hidden, dtype=np.intp))).reshape(-1, 4)
        for j, lid in enumerate(hidden):
            dr, dg, db, _da = darr[j]
            result[lid] = QColor(int(np.clip(float(dr) * 255, 0, 255)),
                                 int(np.clip(float(dg) * 255, 0, 255)),
                                 int(np.clip(float(db) * 255, 0, 255)))
    return result


def _table_ids(n=812, seed=0):
    """A realistic id list: 812 ids up to a few thousand, unsorted."""
    rng = np.random.default_rng(seed)
    ids = rng.choice(np.arange(1, 4000), size=n, replace=False)
    return [int(i) for i in ids]


def _colour_dict(ids, *, none="hidden", missing=40, seed=1):
    """The controller's shape: ``{None: hidden, 0: hidden, id: rgba}`` with
    ``missing`` table ids left out of the dictionary."""
    rng = np.random.default_rng(seed)
    hidden = np.array([0, 0, 0, 0], np.float32)
    d = {0: hidden}
    if none == "hidden":
        d[None] = hidden
    elif none == "visible":
        d[None] = np.array([0.9, 0.2, 0.4, 1.0], np.float32)
    for lid in ids[missing:]:
        rgba = rng.random(4).astype(np.float32)
        rgba[3] = rng.choice([1.0, 0.7, 0.0005])     # a few nearly hidden
        d[lid] = rgba
    return d


def _layer(color_dict, *, use_selection=False, selection=0):
    import warnings

    from napari.layers import Labels
    from napari.utils.colormaps import DirectLabelColormap

    layer = Labels(np.zeros((2, 4, 4), np.int32))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        layer.colormap = DirectLabelColormap(color_dict=color_dict,
                                             use_selection=use_selection,
                                             selection=selection)
    return layer


def _rgb(colours):
    return {k: (c.red(), c.green(), c.blue()) for k, c in colours.items()}


@pytest.mark.parametrize("none", ["hidden", "visible", "absent"])
def test_lookup_equals_the_map_for_every_id(none):
    ids = _table_ids()
    layer = _layer(_colour_dict(ids, none=none))
    query = ids + [0, 5000, 2 ** 40]          # background, unknown, huge
    assert _rgb(NAV._batch_swatch_colors(layer, query)) == \
        _rgb(_old_batch_swatch_colors(layer, query))


def test_lookup_takes_numpy_ints_and_repeats():
    ids = _table_ids(60)
    layer = _layer(_colour_dict(ids, missing=5))
    query = [np.int64(i) for i in ids] + [np.uint32(ids[0]), ids[1]]
    assert _rgb(NAV._batch_swatch_colors(layer, query)) == \
        _rgb(_old_batch_swatch_colors(layer, query))


def test_selection_mode_keeps_the_old_path():
    ids = _table_ids(60)
    layer = _layer(_colour_dict(ids, missing=5), use_selection=True,
                   selection=ids[10])
    assert _rgb(NAV._batch_swatch_colors(layer, ids)) == \
        _rgb(_old_batch_swatch_colors(layer, ids))


def test_direct_map_is_not_called(monkeypatch):
    """The numba-backed map is skipped for a plain direct colour map."""
    from napari.utils.colormaps import DirectLabelColormap

    ids = _table_ids(100)
    layer = _layer(_colour_dict(ids))
    want = _rgb(_old_batch_swatch_colors(layer, ids))

    def no_map(self, values):
        raise AssertionError("DirectLabelColormap.map was called")

    monkeypatch.setattr(DirectLabelColormap, "map", no_map)
    assert _rgb(NAV._batch_swatch_colors(layer, ids)) == want
