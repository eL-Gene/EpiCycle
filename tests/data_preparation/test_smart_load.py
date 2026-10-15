"""Tests for Step 1 "Smart Load" — reuse a fresh prepared multiscale store
instead of re-reading the whole TIFF.

Three surfaces are covered:

* :meth:`DataPreparationController._fresh_prepared_store_for` — the STRICT
  freshness gate.  Returns the store path only when the store dir exists AND its
  OWN ``.zattrs`` ``epicycle_source`` provenance resolves to this source AND its
  ``source_mtime`` matches the source's CURRENT mtime.  The provenance is
  PER-STORE (each store is self-describing), so it works for EVERY prepared
  source in a shared ``largedata/`` dir — not just the most-recently-prepared
  one (the shared ``result_manifest.json`` was last-writer-wins).  Any drift (no
  store, edited source, wrong source, missing / corrupt store attrs) returns
  ``None`` so the caller falls back to the TIFF read.  These are Qt-free (only
  ``.text()`` off a tiny fake widget is needed).

* :meth:`load_source` — when a fresh store exists it loads the store (lazy,
  multiscale) and does NOT read the TIFF; when none exists the normal TIFF path
  is taken UNCHANGED.  Proven by spying that the TIFF pre-flight (the only path
  to the read worker) is / isn't reached.

* :meth:`prepare_large_data` — a fresh store SKIPS ``convert_to_ome_zarr`` and
  just loads the store; a missing / stale store converts as before.

Plus an end-to-end byte-identity check that an extract off a smart-loaded store
equals a direct crop of the source (the L0 guarantee).

Store construction goes through the REAL ``convert_to_ome_zarr`` so the per-store
``.zattrs`` provenance fields under test are the ones production writes.
"""
from __future__ import annotations

import json
import os
from unittest.mock import MagicMock

import numpy as np
import pytest
import tifffile

from napari_epicycle.data_preparation import controller as ctl_mod
from napari_epicycle.data_preparation.controller import DataPreparationController
from napari_epicycle.data_preparation.logic import extract_patches_from_shapes
from napari_epicycle.data_preparation._export import persist_patch
from napari_epicycle.data_preparation._state import (
    empty_state, load_state, save_state, upsert_source,
)
from packages.epicycle_largedata.multiscale import (
    convert_to_ome_zarr, read_store_source,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _distinct(shape, dtype=np.uint16):
    """An array whose every voxel is unique, so any mis-order / off-by-one in a
    crop comparison is caught."""
    return np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)


def _write_tif(tmp_path, name, arr):
    p = tmp_path / name
    tifffile.imwrite(str(p), arr)
    return str(p)


def _box(y0, x0, y1, x1):
    return np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]], dtype=float)


def _store_path(workspace, display_name):
    return os.path.join(
        workspace, "artifacts", "step1", "largedata", f"{display_name}.zarr")


def _prepare_store(workspace, display_name, src, image_axes=None):
    """Build a real multiscale store at the canonical Step-1 location."""
    store = _store_path(workspace, display_name)
    os.makedirs(os.path.dirname(store), exist_ok=True)
    convert_to_ome_zarr(
        src, store, chunk=16, min_size=8, overwrite=True, image_axes=image_axes)
    return store


def _norm(p):
    return os.path.normcase(os.path.abspath(str(p)))


def _store_zattrs_path(store):
    """The DirectoryStore's group ``.zattrs`` — where the per-store
    ``epicycle_source`` provenance lives."""
    return os.path.join(store, ".zattrs")


def _read_store_zattrs(store):
    with open(_store_zattrs_path(store)) as f:
        return json.load(f)


def _write_store_zattrs(store, attrs):
    with open(_store_zattrs_path(store), "w") as f:
        json.dump(attrs, f)


# ===========================================================================
# _fresh_prepared_store_for — the STRICT freshness gate (Qt-free)
# ===========================================================================

class _FakeLineEdit:
    def __init__(self, text=""):
        self._t = text

    def text(self):
        return self._t


class _FakeFreshWidget:
    """Only what `_fresh_prepared_store_for` reads: the workspace + display-name
    line edits' `.text()`."""

    def __init__(self, workspace, display_name):
        self.output_file_path = _FakeLineEdit(workspace)
        self.display_name_edit = _FakeLineEdit(display_name)


def _fresh_controller(workspace, display_name):
    ctl = DataPreparationController.__new__(DataPreparationController)
    ctl.widget = _FakeFreshWidget(workspace, display_name)
    return ctl


class TestFreshPreparedStoreFor:
    def test_returns_store_when_fresh_provenance_matches(self, tmp_path):
        ws = str(tmp_path)
        arr = _distinct((10, 40, 40))
        src = _write_tif(tmp_path, "vol.tif", arr)
        store = _prepare_store(ws, "vol", src)

        ctl = _fresh_controller(ws, "vol")
        got = ctl._fresh_prepared_store_for(src)
        assert got is not None
        assert _norm(got) == _norm(store)

    def test_none_when_no_store_exists(self, tmp_path):
        ws = str(tmp_path)
        src = _write_tif(tmp_path, "vol.tif", _distinct((8, 32, 32)))
        # never prepared
        ctl = _fresh_controller(ws, "vol")
        assert ctl._fresh_prepared_store_for(src) is None

    def test_none_when_source_content_changed_stale(self, tmp_path):
        """An EDITED source (its bytes changed) must NOT be served the stale
        store."""
        ws = str(tmp_path)
        src = _write_tif(tmp_path, "vol.tif", _distinct((10, 40, 40)))
        _prepare_store(ws, "vol", src)

        ctl = _fresh_controller(ws, "vol")
        assert ctl._fresh_prepared_store_for(src) is not None  # fresh now

        with open(src, "ab") as f:                 # a real edit: size moves
            f.write(b"\x00" * 32)
        assert ctl._fresh_prepared_store_for(src) is None

    def test_mtime_only_change_still_serves_the_store(self, tmp_path):
        """Brick 4a a3: a moved timestamp on byte-identical content is a COPY
        (cp without -p / cloud sync / restore), not an edit — re-Preparing a
        whole stack for it was pure waste.  The size guard still catches a real
        replacement; the caller shows an amber line."""
        ws = str(tmp_path)
        src = _write_tif(tmp_path, "vol.tif", _distinct((10, 40, 40)))
        _prepare_store(ws, "vol", src)

        ctl = _fresh_controller(ws, "vol")
        now = os.path.getmtime(src)
        os.utime(src, (now + 1000, now + 1000))
        reasons = []
        assert ctl._fresh_prepared_store_for(src, reasons=reasons) is not None
        assert reasons == ["mtime_drift"]

    def test_none_when_store_source_size_mismatch(self, tmp_path):
        """Audit finding #1 (isolated): the freshness gate must reject a store
        whose recorded ``source_size`` no longer matches the source — even when
        the ``path`` and ``source_mtime`` still match.  This is the signal that
        catches an mtime-preserving content replacement (``cp -p`` / ``rsync
        -a`` / restore-previous-version / cloud sync) that mtime alone misses.
        Rewrite ONLY ``source_size`` so the size guard is the sole reason for
        rejection."""
        ws = str(tmp_path)
        src = _write_tif(tmp_path, "vol.tif", _distinct((10, 40, 40)))
        store = _prepare_store(ws, "vol", src)

        ctl = _fresh_controller(ws, "vol")
        assert ctl._fresh_prepared_store_for(src) is not None  # fresh now

        attrs = _read_store_zattrs(store)
        # Keep path + mtime correct; corrupt only the size (a replacement whose
        # bytes differ but whose mtime was preserved).
        attrs["epicycle_source"]["source_size"] = \
            int(attrs["epicycle_source"]["source_size"]) + 12345
        _write_store_zattrs(store, attrs)

        assert ctl._fresh_prepared_store_for(src) is None

    def test_none_when_mtime_preserved_but_content_replaced(self, tmp_path):
        """Audit finding #1 (end-to-end): overwrite the source with genuinely
        different content of a different byte-size, then RESTORE the original
        mtime (exactly what ``cp -p`` / ``rsync -a`` / cloud sync do).  mtime
        alone would call this fresh and serve the OLD store's bytes for the NEW
        source (silent wrong data for display AND extract); the size guard
        rejects it."""
        ws = str(tmp_path)
        src = _write_tif(tmp_path, "vol.tif", _distinct((10, 40, 40)))
        _prepare_store(ws, "vol", src)

        ctl = _fresh_controller(ws, "vol")
        assert ctl._fresh_prepared_store_for(src) is not None  # fresh now
        mtime_before = os.path.getmtime(src)

        # Replace with DIFFERENT content + a different byte-size, then restore
        # the mtime so the mtime-only check would still pass.
        tifffile.imwrite(src, _distinct((10, 40, 80)))   # 2x the pixels
        os.utime(src, (mtime_before, mtime_before))
        assert abs(os.path.getmtime(src) - mtime_before) <= 1e-6  # mtime restored
        assert os.path.getsize(src) != 0

        # mtime matches, size does not -> NOT served.
        assert ctl._fresh_prepared_store_for(src) is None

    def test_none_when_store_provenance_missing_size_legacy(self, tmp_path):
        """A legacy store written before ``source_size`` existed (has ``path`` +
        ``source_mtime`` but no ``source_size``) must be treated as NOT fresh —
        fail-safe: without the size signal we cannot prove freshness, so we fall
        back to the TIFF read / re-prepare rather than risk serving stale bytes."""
        ws = str(tmp_path)
        src = _write_tif(tmp_path, "vol.tif", _distinct((10, 40, 40)))
        store = _prepare_store(ws, "vol", src)

        attrs = _read_store_zattrs(store)
        attrs["epicycle_source"].pop("source_size", None)  # simulate legacy store
        _write_store_zattrs(store, attrs)

        ctl = _fresh_controller(ws, "vol")
        assert ctl._fresh_prepared_store_for(src) is None

    def test_multi_source_each_finds_its_own_store(self, tmp_path):
        """THE per-store fix: two sources prepared into the SAME ``largedata/``
        dir are each recognized INDEPENDENTLY.  The old shared-sidecar version
        only saw the most-recently-prepared source (last-writer-wins), so the
        earlier one silently fell back to a full TIFF read."""
        ws = str(tmp_path)
        src_a = _write_tif(tmp_path, "a.tif", _distinct((10, 40, 40)))
        src_b = _write_tif(tmp_path, "b.tif", _distinct((12, 36, 36)))
        store_a = _prepare_store(ws, "a", src_a)
        store_b = _prepare_store(ws, "b", src_b)   # same parent dir as A
        # Same parent dir (the shared-sidecar collision point).
        assert os.path.dirname(store_a) == os.path.dirname(store_b)

        got_a = _fresh_controller(ws, "a")._fresh_prepared_store_for(src_a)
        got_b = _fresh_controller(ws, "b")._fresh_prepared_store_for(src_b)

        # BOTH resolve — independently — to their own store.
        assert got_a is not None and got_b is not None
        assert _norm(got_a) == _norm(store_a)
        assert _norm(got_b) == _norm(store_b)

    def test_none_when_store_source_path_mismatch(self, tmp_path):
        ws = str(tmp_path)
        src = _write_tif(tmp_path, "vol.tif", _distinct((10, 40, 40)))
        store = _prepare_store(ws, "vol", src)

        # Rewrite the store's OWN epicycle_source.path to a different file (keep
        # source_mtime correct, isolating the source-path guard).
        attrs = _read_store_zattrs(store)
        attrs["epicycle_source"]["path"] = os.path.abspath(
            str(tmp_path / "somethingelse.tif"))
        _write_store_zattrs(store, attrs)

        ctl = _fresh_controller(ws, "vol")
        assert ctl._fresh_prepared_store_for(src) is None

    def test_none_when_store_provenance_missing(self, tmp_path):
        ws = str(tmp_path)
        src = _write_tif(tmp_path, "vol.tif", _distinct((10, 40, 40)))
        store = _prepare_store(ws, "vol", src)

        # Drop the per-store provenance block (a legacy store with no attrs).
        attrs = _read_store_zattrs(store)
        attrs.pop("epicycle_source", None)
        _write_store_zattrs(store, attrs)

        ctl = _fresh_controller(ws, "vol")
        assert ctl._fresh_prepared_store_for(src) is None

    def test_none_when_store_attrs_corrupt(self, tmp_path):
        ws = str(tmp_path)
        src = _write_tif(tmp_path, "vol.tif", _distinct((10, 40, 40)))
        store = _prepare_store(ws, "vol", src)
        with open(_store_zattrs_path(store), "w") as f:
            f.write("{ this is not valid json ")

        ctl = _fresh_controller(ws, "vol")
        assert ctl._fresh_prepared_store_for(src) is None

    def test_none_when_no_workspace_set(self, tmp_path):
        src = _write_tif(tmp_path, "vol.tif", _distinct((8, 32, 32)))
        ctl = _fresh_controller("", "vol")  # workspace blank
        assert ctl._fresh_prepared_store_for(src) is None

    def test_never_raises_on_garbage_input(self, tmp_path):
        ctl = _fresh_controller(str(tmp_path), "vol")
        assert ctl._fresh_prepared_store_for(None) is None
        assert ctl._fresh_prepared_store_for("") is None


# ===========================================================================
# load_source / prepare_large_data — the branch behavior (real widget)
# ===========================================================================

class _FakeEvent:
    def __init__(self):
        self._cbs = []

    def connect(self, cb=None, *a, **k):
        if cb is not None:
            self._cbs.append(cb)

    def disconnect(self, cb=None):
        if cb is None:
            self._cbs.clear()
        elif cb in self._cbs:
            self._cbs.remove(cb)


class _FakeLayerEvents:
    def __init__(self):
        self.inserted = _FakeEvent()
        self.removed = _FakeEvent()


class _FakeLayer:
    def __init__(self, name, data):
        self.name = name
        self.data = data
        self.visible = True


class _FakeLayerList(list):
    """`viewer.layers` slice the source-layer + crop helpers use: `name in`,
    `layers[name]`, `.remove(layer)`, iteration, and `.events`."""

    def __init__(self):
        super().__init__()
        self.events = _FakeLayerEvents()

    def __contains__(self, item):
        if isinstance(item, str):
            return any(getattr(l, "name", None) == item for l in self)
        return list.__contains__(self, item)

    def __getitem__(self, key):
        if isinstance(key, str):
            for l in self:
                if getattr(l, "name", None) == key:
                    return l
            raise KeyError(key)
        return list.__getitem__(self, key)

    def remove(self, layer):
        list.remove(self, layer)


class _FakeViewer:
    def __init__(self):
        self.layers = _FakeLayerList()
        self.mouse_drag_callbacks = []

    def add_image(self, data, name=None, colormap=None, blending=None,
                  multiscale=False):
        layer = _FakeLayer(name, data)
        self.layers.append(layer)
        return layer


@pytest.fixture(scope="module")
def app():
    pytest.importorskip("qtpy")
    from qtpy.QtWidgets import QApplication

    existing = QApplication.instance()
    if existing is not None:
        yield existing
        return
    yield QApplication([])


@pytest.fixture
def widget(app):
    from napari_epicycle.data_preparation.ui import DataPreparationWidget

    w = DataPreparationWidget(viewer=_FakeViewer())
    yield w
    w.setParent(None)


def _select_source(widget, path):
    from qtpy.QtWidgets import QListWidgetItem
    from qtpy.QtCore import Qt

    item = QListWidgetItem(os.path.basename(path))
    item.setData(Qt.UserRole, path)
    widget.source_file_list.addItem(item)
    widget.source_file_list.setCurrentItem(item)


def _setup(widget, workspace, display_name, path):
    widget.output_file_path.setText(workspace)
    widget.display_name_edit.setText(display_name)
    _select_source(widget, path)


class TestLoadSourceSmartLoad:
    def test_fresh_store_loads_multiscale_and_skips_tiff_read(self, widget, tmp_path):
        ws = str(tmp_path)
        arr = _distinct((2, 8, 40, 40))
        src = _write_tif(tmp_path, "czyx.tif", arr)
        _prepare_store(ws, "czyx", src, image_axes="CZYX")
        _setup(widget, ws, "czyx", src)

        ctl = widget.controller
        # The TIFF read is reached ONLY through the pre-flight; spy it must NOT
        # be called on the smart-load path.
        ctl._preflight_source_route = MagicMock(
            side_effect=AssertionError("TIFF pre-flight must not run"))

        ctl.load_source()

        # Multiscale state wired from the store (level 0 lazy handle present).
        assert ctl._source_lazy is not None
        assert tuple(ctl._source_lazy.shape) == arr.shape
        assert ctl._source_axes_resolved == "CZYX"
        assert len(ctl._loaded_source_layers) == 2       # one layer per channel
        assert ctl._loaded_source_file == src
        # No background read machinery was launched.
        assert ctl._source_load_thread is None
        assert "prepared large-data store" in \
            widget.source_status_label.text()

    def test_no_store_takes_normal_tiff_path(self, widget, tmp_path):
        ws = str(tmp_path)
        src = _write_tif(tmp_path, "plain.tif", _distinct((2, 6, 32, 32)))
        # NOT prepared.
        _setup(widget, ws, "plain", src)

        ctl = widget.controller
        # Decline at the pre-flight so the normal path bails without launching a
        # real read thread — but the spy proves we REACHED the TIFF branch.
        spy = MagicMock(return_value=(False, False))
        ctl._preflight_source_route = spy

        ctl.load_source()

        spy.assert_called_once()                 # took the TIFF branch
        assert ctl._source_lazy is None          # no multiscale store loaded
        assert ctl._loaded_source_layers == []

    def test_edited_source_is_not_served_stale_store(self, widget, tmp_path):
        """STRICT: after the source is edited (mtime moved) the fresh store is
        rejected and the load falls back to the TIFF path."""
        ws = str(tmp_path)
        src = _write_tif(tmp_path, "edit.tif", _distinct((2, 8, 40, 40)))
        _prepare_store(ws, "edit", src, image_axes="CZYX")
        _setup(widget, ws, "edit", src)

        with open(src, "ab") as f:               # "edit" the source for real
            f.write(b"\x00" * 32)                 # (a3: size, not timestamp)

        ctl = widget.controller
        spy = MagicMock(return_value=(False, False))
        ctl._preflight_source_route = spy

        ctl.load_source()

        spy.assert_called_once()                 # fell back to the TIFF branch
        assert ctl._source_lazy is None          # stale store NOT served

    def test_smart_loaded_extract_is_byte_identical_to_source_crop(
            self, widget, tmp_path):
        """The L0 guarantee end-to-end: a crop off the smart-loaded store's
        lazy level-0 handle equals a direct crop of the source array."""
        ws = str(tmp_path)
        arr = _distinct((2, 8, 40, 40))
        src = _write_tif(tmp_path, "x.tif", arr)
        _prepare_store(ws, "x", src, image_axes="CZYX")
        _setup(widget, ws, "x", src)

        ctl = widget.controller
        ctl._preflight_source_route = MagicMock(
            side_effect=AssertionError("TIFF pre-flight must not run"))
        ctl.load_source()

        boxes = [_box(2, 3, 30, 33), _box(10, 10, 25, 40)]
        crop_src, idx = ctl._crop_source_for(["x_ch0", "x_ch1"])
        assert idx == [0, 1]
        got = extract_patches_from_shapes(crop_src, boxes, channel_indices=idx)
        want = extract_patches_from_shapes(arr, boxes)

        assert list(got) == list(want)
        for k in got:
            assert np.array_equal(np.asarray(got[k]), np.asarray(want[k])), k


class TestPrepareLargeDataSkip:
    def test_fresh_store_skips_convert_and_loads(self, widget, tmp_path, monkeypatch):
        ws = str(tmp_path)
        arr = _distinct((10, 40, 40))
        src = _write_tif(tmp_path, "mono.tif", arr)
        _prepare_store(ws, "mono", src)        # 3D ZYX store, already prepared
        _setup(widget, ws, "mono", src)

        convert_spy = MagicMock()
        monkeypatch.setattr(ctl_mod, "convert_to_ome_zarr", convert_spy)

        ctl = widget.controller
        ctl.prepare_large_data()

        convert_spy.assert_not_called()            # convert was skipped
        assert ctl._source_lazy is not None        # store was loaded
        assert ctl._source_axes_resolved == "ZYX"
        assert ctl._largedata_preparing is False   # returned before any launch
        assert "Already prepared" in widget.source_status_label.text()

    def test_no_store_converts_as_before(self, widget, tmp_path, monkeypatch):
        ws = str(tmp_path)
        src = _write_tif(tmp_path, "fresh.tif", _distinct((10, 40, 40)))
        # NOT prepared.
        _setup(widget, ws, "fresh", src)

        # Stub the worker + thread so no real convert / thread runs; we only
        # assert the CONVERT LAUNCH path was taken (worker constructed) and the
        # store-load SKIP path was NOT.
        worker_cls = MagicMock()
        monkeypatch.setattr(ctl_mod, "_LargeDataPrepWorker", worker_cls)
        monkeypatch.setattr(ctl_mod, "QThread", MagicMock())

        ctl = widget.controller
        apply_spy = MagicMock()
        ctl._apply_multiscale_store = apply_spy

        ctl.prepare_large_data()

        apply_spy.assert_not_called()              # did NOT skip-load a store
        worker_cls.assert_called_once()            # took the convert launch path
        assert ctl._largedata_preparing is True

    def test_stale_store_reconverts(self, widget, tmp_path, monkeypatch):
        ws = str(tmp_path)
        src = _write_tif(tmp_path, "stale.tif", _distinct((10, 40, 40)))
        _prepare_store(ws, "stale", src)
        _setup(widget, ws, "stale", src)

        with open(src, "ab") as f:                 # edit -> store now stale
            f.write(b"\x00" * 32)                  # (a3: size, not timestamp)

        worker_cls = MagicMock()
        monkeypatch.setattr(ctl_mod, "_LargeDataPrepWorker", worker_cls)
        monkeypatch.setattr(ctl_mod, "QThread", MagicMock())

        ctl = widget.controller
        apply_spy = MagicMock()
        ctl._apply_multiscale_store = apply_spy

        ctl.prepare_large_data()

        apply_spy.assert_not_called()              # stale store NOT loaded
        worker_cls.assert_called_once()            # re-converted
        assert ctl._largedata_preparing is True


# ===========================================================================
# Brick 1 — schema-v2 store resolution (pointer / reconstruction / move)
# ===========================================================================

def _largedata_dir(ws):
    return os.path.join(ws, "artifacts", "step1", "largedata")


def _convert_at(ws, store_name, src, image_axes=None, voxel_size=None):
    """Build a store at an ARBITRARY name under ``largedata/`` (so a
    collision-suffixed name reconstruction can't guess is exercised)."""
    d = _largedata_dir(ws)
    os.makedirs(d, exist_ok=True)
    store = os.path.join(d, store_name)
    convert_to_ome_zarr(src, store, chunk=16, min_size=8, overwrite=True,
                        image_axes=image_axes, voxel_size=voxel_size)
    return store


class TestFreshStorePointerResolution:
    def test_reconstruction_fallback_when_no_pointer_v1(self, tmp_path):
        """A v1 workspace has NO registry pointer; the store is still resolved
        by RECONSTRUCTING ``{display_name}.zarr`` — Brick 1 must not break the
        Smart-Load of an existing v1 workspace."""
        ws = str(tmp_path)
        src = _write_tif(tmp_path, "vol.tif", _distinct((10, 40, 40)))
        store = _prepare_store(ws, "vol", src)   # base name; NO state file
        ctl = _fresh_controller(ws, "vol")
        got = ctl._fresh_prepared_store_for(src)
        assert got is not None
        assert _norm(got) == _norm(store)

    def test_pointer_resolves_suffixed_store_name(self, tmp_path):
        """The registry ``prepared_store`` pointer resolves a COLLISION-SUFFIXED
        store name that reconstruction (which only tries ``{display_name}.zarr``)
        would miss."""
        ws = str(tmp_path)
        src = _write_tif(tmp_path, "vol.tif", _distinct((10, 40, 40)))
        store = _convert_at(ws, "vol__src_0.zarr", src)
        # The base name does NOT exist -> reconstruction alone would fail.
        assert not os.path.isdir(os.path.join(_largedata_dir(ws), "vol.zarr"))

        st = empty_state()
        upsert_source(st, path=src, source_id="src_0",
                      prepared_store="artifacts/step1/largedata/vol__src_0.zarr")
        save_state(ws, st)

        got = _fresh_controller(ws, "vol")._fresh_prepared_store_for(src)
        assert got is not None
        assert _norm(got) == _norm(store)

    def test_pointer_bound_store_tolerates_path_mismatch_after_move(self, tmp_path):
        """Workspace-move resolution: the store's recorded
        abspath is stale after a move, but the ws-relative pointer still binds
        it and the abspath is demoted to a HINT — the store is served (its
        ``{size,mtime}`` fingerprint is fresh) DESPITE the path mismatch that the
        reconstruction fallback rejects."""
        ws = str(tmp_path)
        src = _write_tif(tmp_path, "vol.tif", _distinct((10, 40, 40)))
        store = _prepare_store(ws, "vol", src)   # base name

        # "prepared elsewhere, workspace since moved": rewrite the store's own
        # recorded source path to a stale abspath; keep mtime+size correct.
        attrs = _read_store_zattrs(store)
        attrs["epicycle_source"]["path"] = os.path.abspath(
            str(tmp_path / "OLD_LOCATION" / "vol.tif"))
        _write_store_zattrs(store, attrs)

        # Reconstruction fallback (no pointer) REJECTS on the path mismatch.
        assert _fresh_controller(ws, "vol")._fresh_prepared_store_for(src) is None

        # With the pointer, the mismatch is tolerated and the store is served.
        st = empty_state()
        upsert_source(st, path=src, source_id="src_0",
                      prepared_store="artifacts/step1/largedata/vol.zarr")
        save_state(ws, st)
        got = _fresh_controller(ws, "vol")._fresh_prepared_store_for(src)
        assert got is not None
        assert _norm(got) == _norm(store)


class TestCollisionSafeStoreNaming:
    def test_same_stem_different_source_gets_suffixed_name(self, tmp_path):
        """A1 data-loss guard: two DIFFERENT sources that share a display-name
        must NOT resolve to the same ``{alias}.zarr`` — the second gets a
        ``{alias}__{source_id}.zarr`` name so ``overwrite=True`` can't wipe the
        first's store."""
        ws = str(tmp_path)
        (tmp_path / "A").mkdir()
        (tmp_path / "B").mkdir()
        src_a = _write_tif(tmp_path / "A", "kidney.tif", _distinct((10, 40, 40)))
        src_b = _write_tif(tmp_path / "B", "kidney.tif", _distinct((12, 36, 36)))

        ctl = _fresh_controller(ws, "kidney")
        ctl._log = lambda *a, **k: None

        # First source: base name (nothing there yet) — build A's store there.
        p_a = ctl._resolve_prepare_store_path(ws, "kidney", src_a)
        assert os.path.basename(p_a) == "kidney.zarr"
        os.makedirs(os.path.dirname(p_a), exist_ok=True)
        convert_to_ome_zarr(src_a, p_a, chunk=16, min_size=8, overwrite=True)

        # Second, DIFFERENT source, same stem -> suffixed (A's store untouched).
        p_b = ctl._resolve_prepare_store_path(ws, "kidney", src_b)
        assert os.path.basename(p_b) != "kidney.zarr"
        assert "kidney__src" in os.path.basename(p_b)
        assert _norm(p_a) != _norm(p_b)
        assert os.path.isdir(p_a)   # first source's store not clobbered

    def test_same_source_reuses_base_name(self, tmp_path):
        """Re-preparing the SAME source reuses (overwrites in place) the base
        name — the freshness of a re-prepare is a legitimate overwrite."""
        ws = str(tmp_path)
        src = _write_tif(tmp_path, "kidney.tif", _distinct((10, 40, 40)))
        store = _prepare_store(ws, "kidney", src)
        ctl = _fresh_controller(ws, "kidney")
        ctl._log = lambda *a, **k: None
        p = ctl._resolve_prepare_store_path(ws, "kidney", src)
        assert _norm(p) == _norm(store)


class TestAxesRoundTrip:
    def test_zcyx_source_registry_records_zcyx_prepare_then_export(self, tmp_path):
        """THE central correctness test: a ZCYX-origin source
        prepared → smart-loaded → exported records registry
        ``source_axes == "ZCYX"``, NOT the canonical ``"CZYX"``.

        Prepare writes the honest on-disk order from the store's own attrs;
        export is NON-CLOBBERING (A4) so handing it the canonical
        ``_source_axes_resolved`` (what a smart-loaded store displays) can't
        overwrite the recorded ``ZCYX``."""
        ws = str(tmp_path)
        arr = _distinct((6, 2, 24, 24), dtype=np.uint16)   # Z C Y X on disk
        src = _write_tif(tmp_path, "zcyx.tif", arr)
        store = _prepare_store(ws, "zcyx", src, image_axes="ZCYX")
        assert read_store_source(store)["source_axes"] == "ZCYX"

        # --- Prepare-finish registry upsert (A2b) ---
        ctl = _fresh_controller(ws, "zcyx")
        ctl._log = lambda *a, **k: None
        ctl._largedata_src_path = src
        ctl._largedata_source_id = "src_0"
        ctl._largedata_prepare_params = {"image_axes": "ZCYX"}
        ctl._persist_prepared_source(store, "zcyx")

        row = load_state(ws)["sources"][0]
        assert row["source_axes"] == "ZCYX"        # honest at prepare
        assert "prepared" in row["origin"]
        assert row["prepared_store"] == "artifacts/step1/largedata/zcyx.zarr"
        assert row["voxel_size_source"] == "unknown"

        # --- export hands the CANONICAL order (a smart-loaded store displays
        # CZYX), which must NOT clobber the recorded ZCYX ---
        czyx_patch = np.zeros((2, 4, 8, 8), dtype=np.uint16)
        persist_patch(
            ws, "zcyx_patch_1", czyx_patch,
            channel_names=["c0", "c1"], original_shape=arr.shape,
            cropped_shape=czyx_patch.shape, selected_z_range=[0, 4],
            bounding_box="n/a", source_path=src,
            source_display_name="zcyx", axes="CZYX", source_axes="CZYX")

        row2 = load_state(ws)["sources"][0]
        assert row2["source_axes"] == "ZCYX"       # NOT overwritten by export
        assert len(load_state(ws)["sources"]) == 1  # same source row (dedup)


class TestPersistPreparedSourceVoxelCoherence:
    """`_persist_prepared_source` keeps the voxel value + type COHERENT AND HONEST
    (Brick-2 grill FIX 5): a baked store voxel with no prior user value is typed
    "detected" (not silently promoted to "user"); a prior user value stays "user";
    a 1/1/1 store voxel touches nothing."""

    def _prep_controller(self, ws, src, display_name):
        ctl = _fresh_controller(ws, display_name)
        ctl._log = lambda *a, **k: None
        ctl._largedata_src_path = src
        ctl._largedata_source_id = "src_0"
        ctl._largedata_prepare_params = None
        return ctl

    def test_real_store_voxel_no_prior_user_is_detected(self, tmp_path):
        """A non-1/1/1 store voxel with NO prior user override is a value the
        Prepare baked from a header/detected spacing -> typed "detected", NEVER
        "user" (only a genuine user edit is "user")."""
        ws = str(tmp_path)
        src = _write_tif(tmp_path, "v.tif", _distinct((10, 40, 40)))
        store = _convert_at(ws, "v.zarr", src, voxel_size=(2.0, 0.5, 0.5))

        self._prep_controller(ws, src, "v")._persist_prepared_source(store, "v")

        row = load_state(ws)["sources"][0]
        assert row["voxel_size"] == [2.0, 0.5, 0.5]
        assert row["voxel_size_source"] == "detected"   # honest, not "user"

    def test_real_store_voxel_keeps_prior_user_type(self, tmp_path):
        """If the source row is ALREADY "user" (the user typed the spacing before
        Prepare), a matching store voxel keeps the "user" type."""
        ws = str(tmp_path)
        src = _write_tif(tmp_path, "v.tif", _distinct((10, 40, 40)))
        store = _convert_at(ws, "v.zarr", src, voxel_size=(2.0, 0.5, 0.5))
        st = empty_state()
        upsert_source(st, path=src, source_id="src_0",
                      voxel_size=[2.0, 0.5, 0.5], voxel_size_source="user")
        save_state(ws, st)

        self._prep_controller(ws, src, "v")._persist_prepared_source(store, "v")

        row = load_state(ws)["sources"][0]
        assert row["voxel_size"] == [2.0, 0.5, 0.5]
        assert row["voxel_size_source"] == "user"     # prior user type preserved

    def test_default_store_voxel_leaves_prior_user_value_intact(self, tmp_path):
        """A 1/1/1 (None) store voxel must NOT flip a PRIOR user value's type to
        "unknown" nor drop the value — both are skipped."""
        ws = str(tmp_path)
        src = _write_tif(tmp_path, "v.tif", _distinct((10, 40, 40)))
        store = _convert_at(ws, "v.zarr", src)   # no voxel -> store voxel None

        st = empty_state()
        upsert_source(st, path=src, source_id="src_0",
                      voxel_size=[3.0, 1.0, 1.0], voxel_size_source="user")
        save_state(ws, st)

        self._prep_controller(ws, src, "v")._persist_prepared_source(store, "v")

        row = load_state(ws)["sources"][0]
        assert row["voxel_size"] == [3.0, 1.0, 1.0]    # prior value intact
        assert row["voxel_size_source"] == "user"      # type NOT flipped
        assert "prepared" in row["origin"]             # but the prepare DID run


# ===========================================================================
# Brick 4c item 8 — journey (a) DRIVEN through the worker's finished handler
# ===========================================================================
from qtpy.QtCore import QThread as _RealQThread


class _SyncThread(_RealQThread):
    """A QThread that runs `load_source`'s worker on the CALLING thread.

    A REAL QThread subclass, because `worker.moveToThread()` will not accept a
    duck type.  `start()` moves the worker back to the current thread (so its
    `finished` signal is delivered DIRECTLY rather than queued to a thread that
    is not running) and calls `run()` itself.

    The point is to exercise the wiring `load_source` actually builds — worker
    construction, the `finished` connection, `_on_source_loaded` — instead of
    calling `_on_source_loaded` directly with hand-made arguments.  That
    fake-bypass is the class that hid a HIGH bug during Brick 4a.

    The controller is held PER INSTANCE, handed in by the factory the test
    monkeypatches in (`lambda: _SyncThread(ctl)`).  It used to be a class
    attribute, which is shared mutable state across every test in the file and
    leaks into whatever runs next.
    """

    def __init__(self, controller):
        super().__init__()
        self._controller = controller

    def start(self, *_a, **_k):
        worker = self._controller._source_load_worker
        worker.moveToThread(_RealQThread.currentThread())
        worker.run()


class TestJourneyACopiedWorkspaceDrivenEndToEnd:
    """The workspace AND its data were copied; the user browses the new folder.

    Driven all the way through `load_source` -> the worker -> its `finished`
    handler, then ④ is asked whether its rows are interactive.
    """

    def test_the_library_stays_live_after_the_move(
            self, widget, tmp_path, monkeypatch):
        from napari_epicycle.data_preparation._state import (
            register_patch, register_source,
        )
        from napari_epicycle.data_preparation.controller import (
            LIBRARY_ROLE_INTERACTIVE,
        )

        ws = str(tmp_path / "ws")
        os.makedirs(ws, exist_ok=True)
        old_dir = tmp_path / "pc1"
        old_dir.mkdir()
        src = _write_tif(old_dir, "wound.tif", _distinct((6, 24, 24)))

        # A source with two patches cut from it, never Prepared.
        state = load_state(ws)
        sid = register_source(state, path=src, display_name="wound",
                              axes="ZYX")
        state["sources"][0]["fingerprint"] = {
            "size": os.path.getsize(src), "mtime": os.path.getmtime(src)}
        state["sources"][0]["source_shape"] = [6, 24, 24]
        for name in ("wound_1", "wound_2"):
            register_patch(
                state, patch_name=name,
                relative_path=f"artifacts/step1/patches/{name}.tiff",
                sidecar_path=f"artifacts/step1/patches/{name}_metadata.json",
                source_id=sid)
        save_state(ws, state)

        # THE MOVE: the data now lives somewhere else, same filename.
        new_dir = tmp_path / "pc2"
        new_dir.mkdir()
        moved = str(new_dir / "wound.tif")
        os.replace(src, moved)

        _setup(widget, ws, "wound", moved)
        ctl = widget.controller
        monkeypatch.setattr(ctl_mod, "QThread", lambda: _SyncThread(ctl))
        # The >RAM route dialog is not what this test is about.
        ctl._preflight_source_route = MagicMock(return_value=(True, False))

        ctl.load_source()

        # The read really ran through the worker's finished handler.
        assert ctl._loaded_source_file == moved
        assert ctl._loaded_source_layers

        # The row followed the file: same id, new path, still exactly one row.
        after = load_state(ws)
        assert len(after["sources"]) == 1, after["sources"]
        assert after["sources"][0]["source_id"] == sid
        assert after["sources"][0]["path"] == moved
        assert ctl._loaded_source_id == sid

        # ...and ④ is live, which is the whole point of the brick.
        lst = widget.extracted_roi_list
        names = []
        for i in range(lst.count()):
            item = lst.item(i)
            names.append(item.text())
            assert item.data(LIBRARY_ROLE_INTERACTIVE) is True, item.text()
        assert sorted(names) == ["wound_1", "wound_2"]
        assert sorted(ctl._current_source_patch_names()) == [
            "wound_1", "wound_2"]
