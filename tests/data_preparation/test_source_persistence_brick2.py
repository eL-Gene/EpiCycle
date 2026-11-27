"""Step 1 source-metadata persistence — **Brick 2** (the "no re-specify"
round-trip: save-on-edit + restore-on-load).

Covers the Brick-2 scope:

* ``read_voxel_spacing`` — the header-detect "detected" tier (TIFF ImageJ / OME +
  ``.zarr`` NGFF scale), never raises.
* voxel precedence + honest typing (``_resolve_voxel_for_source`` / A7).
* save-on-edit (alias / axes / voxel) routed through ``upsert_source``, debounced,
  and FLUSHED into the RIGHT workspace before a context change (A8 / Step-2
  gotcha).
* restore-on-load: the once-per-session ``largedata/`` store reconciliation
  (orphan synth, freshest-of-two, abs->rel, dedup — A6 / B2-N2) + widget restore.
* alias-in-combo + tooltip + the pick-clobber fix.
* B2-N1: a ``source_id`` allocated at SELECT sets ``path``, so a later export
  (path-keyed ``register_source``) does NOT append a duplicate row.

Store construction goes through the REAL ``convert_to_ome_zarr`` so the per-store
``.zattrs`` the reconcile reads are production's.
"""
from __future__ import annotations

import json
import os
import types

import numpy as np
import pytest
import tifffile

from napari_epicycle.data_preparation.controller import DataPreparationController
from napari_epicycle.data_preparation._state import (
    empty_state, load_state, save_state, register_source, upsert_source,
    _norm_path,
)
from napari_epicycle.data_preparation._paths import step1_state_path
from packages.epicycle_largedata.axes import read_voxel_spacing
from packages.epicycle_largedata.multiscale import convert_to_ome_zarr


# ---------------------------------------------------------------------------
# fixtures / fakes
# ---------------------------------------------------------------------------

def _tif(tmp_path, name, shape=(6, 24, 24), imagej=False, spacing=None, res=None):
    p = str(tmp_path / name)
    arr = np.zeros(shape, dtype=np.uint16)
    if imagej:
        tifffile.imwrite(p, arr, imagej=True,
                         resolution=(res, res) if res else None,
                         metadata=({"spacing": spacing, "unit": "um"}
                                   if spacing is not None else None))
    else:
        tifffile.imwrite(p, arr)
    return p


def _prep_store(ws, name, src, voxel_size=None, image_axes=None):
    store = os.path.join(ws, "artifacts", "step1", "largedata", f"{name}.zarr")
    os.makedirs(os.path.dirname(store), exist_ok=True)
    convert_to_ome_zarr(src, store, chunk=8, min_size=4, overwrite=True,
                        image_axes=image_axes, voxel_size=voxel_size)
    return store


def _zattrs(store):
    with open(os.path.join(store, ".zattrs")) as f:
        return json.load(f)


def _write_zattrs(store, attrs):
    with open(os.path.join(store, ".zattrs"), "w") as f:
        json.dump(attrs, f)


class _FakeTimer:
    def __init__(self):
        self.started = 0
        self.start_args = []
        self.stopped = 0

    def setSingleShot(self, *_):
        pass

    def start(self, *a):
        self.started += 1
        self.start_args.append(a)

    def stop(self):
        self.stopped += 1


class _FakeSpin:
    def __init__(self, v=1.0):
        self._v = float(v)

    def value(self):
        return self._v

    def setValue(self, v):
        self._v = float(v)

    def blockSignals(self, _b):
        pass


class _FakeLine:
    def __init__(self, t=""):
        self._t = t

    def text(self):
        return self._t

    def setText(self, t):
        self._t = str(t)


class _FakeCombo:
    def __init__(self, t="CZYX"):
        self._t = t

    def currentText(self):
        return self._t

    def setCurrentText(self, t):
        self._t = t


class _FakeSourceCombo:
    """Item-list stand-in for the ② source dropdown — enough for the Bug-1
    live-relabel path (``currentIndex`` + ``setItemText``)."""

    def __init__(self):
        self._texts, self._data, self._cur = [], [], -1

    def clear(self):
        self._texts, self._data, self._cur = [], [], -1

    def count(self):
        return len(self._texts)

    def addItem(self, text):
        self._texts.append(text); self._data.append({})

    def setItemText(self, i, text):
        self._texts[i] = text

    def itemText(self, i):
        return self._texts[i]

    def setItemData(self, i, val, role=None):
        self._data[i][role] = val

    def itemData(self, i, role=None):
        return self._data[i].get(role)

    def setCurrentIndex(self, i):
        self._cur = i

    def currentIndex(self):
        return self._cur


class _RecordWidget:
    """Minimal widget stand-in exposing only what the Brick-2 controller paths
    read/write, recording the voxel/axis pushes so restore can be asserted."""

    def __init__(self, ws=""):
        self.output_file_path = _FakeLine(ws)
        self.display_name_edit = _FakeLine("")
        self.voxel_z_spin = _FakeSpin(1.0)
        self.voxel_y_spin = _FakeSpin(1.0)
        self.voxel_x_spin = _FakeSpin(1.0)
        self.axis_order_combo = _FakeCombo("CZYX")
        self.source_combo = _FakeSourceCombo()
        self._detected_voxel_size = (1.0, 1.0, 1.0)
        self.applied_voxel = None
        self.detected_pushed = None
        self.applied_axis = None
        self._auto_alias = ""

    def set_detected_voxel_size(self, z, y, x):
        self._detected_voxel_size = (z, y, x)
        self.detected_pushed = (z, y, x)

    def apply_voxel_size(self, z, y, x, source="detected"):
        self.applied_voxel = (z, y, x, source)

    def apply_axis_order(self, order, hint):
        self.applied_axis = (order, hint)


class _FakeItem:
    """Stand-in for a QListWidgetItem carrying a source path in UserRole."""

    def __init__(self, path):
        self._p = path

    def data(self, _role):
        return self._p


def _ctl(ws="", widget=None):
    ctl = DataPreparationController.__new__(DataPreparationController)
    ctl.widget = widget if widget is not None else _RecordWidget(ws)
    ctl._pending_source_edits = []
    ctl._pending_source_ws = None
    ctl._state_save_timer = _FakeTimer()
    ctl._selected_source_path = None
    ctl._suspend_source_restore = False
    ctl._voxel_detect_cache = {}
    ctl._axis_order_overrides = {}
    ctl._axis_combo_source = None
    ctl._loaded_source_file = None
    ctl._scanned_files = []
    ctl._step1_sources_restored = set()
    return ctl


# ===========================================================================
# read_voxel_spacing
# ===========================================================================

class TestReadVoxelSpacing:
    def test_imagej_tiff_spacing_and_resolution(self, tmp_path):
        p = _tif(tmp_path, "ij.tif", shape=(5, 32, 32),
                 imagej=True, spacing=0.5, res=1.0 / 0.1)
        assert read_voxel_spacing(p) == pytest.approx((0.5, 0.1, 0.1))

    def test_plain_tiff_returns_unit_or_none(self, tmp_path):
        p = _tif(tmp_path, "plain.tif")
        v = read_voxel_spacing(p)
        # No physical spacing recorded -> unit (treated as "unknown") or None.
        assert v is None or all(abs(c - 1.0) < 1e-9 for c in v)

    def test_zarr_ngff_scale(self, tmp_path):
        ws = str(tmp_path)
        src = _tif(tmp_path, "z.tif")
        store = _prep_store(ws, "z", src, voxel_size=(0.4, 0.2, 0.2))
        assert read_voxel_spacing(store) == pytest.approx((0.4, 0.2, 0.2))

    def test_zarr_without_voxel_is_unit(self, tmp_path):
        ws = str(tmp_path)
        src = _tif(tmp_path, "z2.tif")
        store = _prep_store(ws, "z2", src)
        assert read_voxel_spacing(store) == pytest.approx((1.0, 1.0, 1.0))

    def test_missing_path_is_none_never_raises(self, tmp_path):
        assert read_voxel_spacing(str(tmp_path / "nope.tif")) is None
        assert read_voxel_spacing(str(tmp_path / "nope.zarr")) is None
        assert read_voxel_spacing(None) is None

    # --- FIX 2: non-physical resolution units must NOT read as spacing --------
    def test_dpi_inch_resolution_is_not_physical(self, tmp_path):
        """A print-DPI TIFF (ResolutionUnit=INCH, tifffile's default) must NOT be
        read as a physical voxel (would give ~84.67 µm for 300 DPI)."""
        p = str(tmp_path / "dpi.tif")
        tifffile.imwrite(p, np.zeros((32, 32), np.uint16), resolution=(300, 300))
        assert read_voxel_spacing(p) is None

    def test_explicit_inch_resolution_is_not_physical(self, tmp_path):
        p = str(tmp_path / "inch.tif")
        tifffile.imwrite(p, np.zeros((32, 32), np.uint16),
                         resolution=(300, 300), resolutionunit="INCH")
        assert read_voxel_spacing(p) is None

    def test_none_resolution_unit_is_not_physical(self, tmp_path):
        p = str(tmp_path / "none.tif")
        tifffile.imwrite(p, np.zeros((32, 32), np.uint16),
                         resolution=(2, 1), resolutionunit="NONE")
        assert read_voxel_spacing(p) is None

    def test_centimeter_resolution_unit_is_physical(self, tmp_path):
        """A CENTIMETER-unit TIFF IS physical: 1000 px/cm -> 10 µm/px."""
        p = str(tmp_path / "cm.tif")
        tifffile.imwrite(p, np.zeros((32, 32), np.uint16),
                         resolution=(1000, 1000), resolutionunit="CENTIMETER")
        assert read_voxel_spacing(p) == pytest.approx((10.0, 10.0, 10.0))

    def test_imagej_inch_unit_z_is_not_physical(self, tmp_path):
        """FIX B: an ImageJ stack with `unit="inch"` must NOT yield a Z voxel of
        `spacing * 25400` — inch is not physical microscopy spacing (consistent
        with the XY inch rejection)."""
        p = str(tmp_path / "inchz.tif")
        tifffile.imwrite(p, np.zeros((5, 32, 32), np.uint16), imagej=True,
                         resolution=(10, 10), metadata={"spacing": 0.5, "unit": "inch"})
        assert read_voxel_spacing(p) is None


# ===========================================================================
# voxel precedence + honest typing (A7)
# ===========================================================================

class TestVoxelPrecedence:
    def test_user_override_beats_header_and_store(self, tmp_path):
        ctl = _ctl()
        src = _tif(tmp_path, "hd.tif", shape=(5, 32, 32),
                   imagej=True, spacing=0.5, res=1.0 / 0.1)
        row = {"voxel_size": [0.9, 0.3, 0.3], "voxel_size_source": "user"}
        eff, eff_src, baseline, base_src = ctl._resolve_voxel_for_source(src, row)
        assert eff == (0.9, 0.3, 0.3) and eff_src == "user"
        # FIX A: resolve is HEADER-FREE for a user row (the detect would be
        # discarded by precedence anyway), so with no store the reset baseline
        # is the (1,1,1) unknown placeholder — the header is NOT opened on
        # select; ↺ re-detects it lazily (see the lazy-reset test below).
        assert baseline == (1.0, 1.0, 1.0) and base_src == "unknown"

    def test_store_scale_beats_header(self, tmp_path):
        ctl = _ctl()
        src = _tif(tmp_path, "hd2.tif", shape=(5, 32, 32),
                   imagej=True, spacing=0.5, res=1.0 / 0.1)
        store = _prep_store(str(tmp_path), "hd2", src, voxel_size=(0.8, 0.4, 0.4))
        _levels_meta = {"scales": [[0.8, 0.4, 0.4]]}
        eff, eff_src, _, _ = ctl._resolve_voxel_for_source(
            src, None, store_meta=_levels_meta)
        assert eff == pytest.approx((0.8, 0.4, 0.4)) and eff_src == "detected"

    def test_header_detect_when_no_user_no_store(self, tmp_path):
        ctl = _ctl()
        src = _tif(tmp_path, "hd3.tif", shape=(5, 32, 32),
                   imagej=True, spacing=0.25, res=1.0 / 0.05)
        eff, eff_src, _, _ = ctl._resolve_voxel_for_source(src, None)
        assert eff == pytest.approx((0.25, 0.05, 0.05)) and eff_src == "detected"

    def test_unknown_unit_default(self, tmp_path):
        ctl = _ctl()
        src = _tif(tmp_path, "hd4.tif")  # no spacing
        eff, eff_src, _, _ = ctl._resolve_voxel_for_source(src, None)
        assert eff == (1.0, 1.0, 1.0) and eff_src == "unknown"

    def test_unit_store_scale_ignored(self, tmp_path):
        ctl = _ctl()
        src = _tif(tmp_path, "hd5.tif")
        eff, eff_src, _, _ = ctl._resolve_voxel_for_source(
            src, None, store_meta={"scales": [[1.0, 1.0, 1.0, 1.0]]})
        assert eff_src == "unknown"

    # --- FIX A: no wasteful header opens on select --------------------------
    def test_user_row_opens_zero_headers(self, tmp_path, monkeypatch):
        """A `"user"`-override source must NOT open the raw header on select
        (the detect would be discarded by precedence anyway)."""
        import napari_epicycle.data_preparation.controller as ctl_mod
        calls = []
        monkeypatch.setattr(ctl_mod, "read_voxel_spacing_detail",
                            lambda p: calls.append(p))
        ctl = _ctl()
        src = _tif(tmp_path, "u.tif", shape=(5, 32, 32),
                   imagej=True, spacing=0.5, res=1.0 / 0.1)
        eff, eff_src, _, _ = ctl._resolve_voxel_for_source(
            src, {"voxel_size": [0.9, 0.3, 0.3], "voxel_size_source": "user"})
        assert eff == (0.9, 0.3, 0.3) and eff_src == "user"
        assert calls == []   # ZERO header opens

    def test_detect_is_cached_across_reselect(self, tmp_path, monkeypatch):
        """Arrowing A→B→A opens each header at most once (memoized per path)."""
        import napari_epicycle.data_preparation.controller as ctl_mod
        real = ctl_mod.read_voxel_spacing_detail
        calls = []
        monkeypatch.setattr(ctl_mod, "read_voxel_spacing_detail",
                            lambda p: (calls.append(p) or real(p)))
        ctl = _ctl()
        a = _tif(tmp_path, "a.tif")
        b = _tif(tmp_path, "b.tif")
        ctl._resolve_voxel_for_source(a, None)
        ctl._resolve_voxel_for_source(b, None)
        ctl._resolve_voxel_for_source(a, None)   # cached — no re-open
        assert len(calls) == 2


# ===========================================================================
# restore-on-load reconciliation
# ===========================================================================

class TestReconcileSourcesWithStores:
    def test_orphan_store_synthesizes_row(self, tmp_path):
        ws = str(tmp_path)
        src = _tif(tmp_path, "orphan.tif")
        _prep_store(ws, "orphan", src, voxel_size=(0.4, 0.1, 0.1))
        st = empty_state()
        assert _ctl()._reconcile_sources_with_stores(ws, st) is True
        assert len(st["sources"]) == 1
        row = st["sources"][0]
        assert _norm_path(row["path"]) == _norm_path(src)
        assert row["prepared_store"] == "artifacts/step1/largedata/orphan.zarr"
        assert "prepared" in row["origin"]
        # An orphan store's baked voxel is honest "detected", never silent "user".
        assert row["voxel_size_source"] == "detected"   # real store voxel

    def test_backfills_pointer_for_existing_row(self, tmp_path):
        ws = str(tmp_path)
        src = _tif(tmp_path, "kidney.tif")
        _prep_store(ws, "kidney", src)
        st = empty_state()
        register_source(st, path=src, display_name="MyKidney", axes="ZYX")
        assert st["sources"][0].get("prepared_store") is None
        assert _ctl()._reconcile_sources_with_stores(ws, st) is True
        row = st["sources"][0]
        assert row["display_name"] == "MyKidney"       # user field untouched
        assert row["prepared_store"] == "artifacts/step1/largedata/kidney.zarr"
        assert row["fingerprint"] is not None
        # no duplicate synth
        assert len(st["sources"]) == 1

    def test_freshest_of_two_stores_wins(self, tmp_path):
        ws = str(tmp_path)
        src = _tif(tmp_path, "twin.tif")
        store_a = _prep_store(ws, "twin", src)         # base name
        store_b = _prep_store(ws, "twin__src_9", src)  # second store, same source
        # Make B the freshest by source_mtime.
        a = _zattrs(store_a); a["epicycle_source"]["source_mtime"] = 100.0
        _write_zattrs(store_a, a)
        b = _zattrs(store_b); b["epicycle_source"]["source_mtime"] = 200.0
        _write_zattrs(store_b, b)
        st = empty_state()
        _ctl()._reconcile_sources_with_stores(ws, st)
        # exactly one row (dedup by path), pointer -> the freshest store.
        assert len(st["sources"]) == 1
        assert st["sources"][0]["prepared_store"].endswith("twin__src_9.zarr")

    def test_never_points_row_at_another_sources_store(self, tmp_path):
        ws = str(tmp_path)
        src_a = _tif(tmp_path, "a.tif")
        src_b = _tif(tmp_path, "b.tif")
        _prep_store(ws, "a", src_a)
        _prep_store(ws, "b", src_b)
        st = empty_state()
        register_source(st, path=src_a, display_name="A", axes="ZYX")
        register_source(st, path=src_b, display_name="B", axes="ZYX")
        _ctl()._reconcile_sources_with_stores(ws, st)
        by_path = {_norm_path(s["path"]): s for s in st["sources"]}
        assert by_path[_norm_path(src_a)]["prepared_store"].endswith("a.zarr")
        assert by_path[_norm_path(src_b)]["prepared_store"].endswith("b.zarr")

    def test_abs_pointer_rewritten_to_relative(self, tmp_path):
        ws = str(tmp_path)
        src = _tif(tmp_path, "abs.tif")
        store = _prep_store(ws, "abs", src)
        st = empty_state()
        sid = register_source(st, path=src, display_name="Abs", axes="ZYX")
        # Simulate a pre-Brick-2 absolute pointer (a moved workspace's stale abs).
        for s in st["sources"]:
            if s["source_id"] == sid:
                s["prepared_store"] = os.path.abspath(store)
        _ctl()._reconcile_sources_with_stores(ws, st)
        ptr = st["sources"][0]["prepared_store"]
        assert not os.path.isabs(ptr)
        assert ptr == "artifacts/step1/largedata/abs.zarr"

    def test_abs_pointer_basename_collision_not_cross_bound(self, tmp_path):
        """FIX 3: an ABS pointer whose basename matches a store recorded to a
        DIFFERENT source must NOT be rebound to it (loading B's pixels as A).
        The pointer is nulled; step 3 re-derives A's own store by recorded path."""
        ws = str(tmp_path)
        src_a = _tif(tmp_path, "a.tif")
        src_b = _tif(tmp_path, "b.tif")
        # Only B's store exists on disk, under the name "shared.zarr".
        store_b = _prep_store(ws, "shared", src_b)
        st = empty_state()
        sid = register_source(st, path=src_a, display_name="A", axes="ZYX")
        # A's row carries a stale ABS pointer whose basename == B's store name.
        for s in st["sources"]:
            if s["source_id"] == sid:
                s["prepared_store"] = os.path.abspath(store_b)
        _ctl()._reconcile_sources_with_stores(ws, st)
        row_a = next(s for s in st["sources"]
                     if _norm_path(s["path"]) == _norm_path(src_a))
        # A must NOT be bound to B's store; it has no store of its own -> null.
        assert row_a["prepared_store"] is None

    def test_dead_store_pointer_nulled(self, tmp_path):
        ws = str(tmp_path)
        os.makedirs(os.path.join(ws, "artifacts", "step1", "largedata"))
        st = empty_state()
        sid = register_source(st, path=str(tmp_path / "gone.tif"),
                              display_name="Gone", axes="ZYX")
        for s in st["sources"]:
            if s["source_id"] == sid:
                s["prepared_store"] = "artifacts/step1/largedata/missing.zarr"
        assert _ctl()._reconcile_sources_with_stores(ws, st) is True
        assert st["sources"][0]["prepared_store"] is None

    def test_dead_processed_store_pointer_nulled(self, tmp_path):
        """A deleted ``*_processed.zarr`` left a PHANTOM `processed_store`
        pointer forever: the raw fallback is correct, but the pointer was never
        cleared, so the ⟨processed⟩ toggle kept advertising a store that is not
        there."""
        ws = str(tmp_path)
        os.makedirs(os.path.join(ws, "artifacts", "step1", "largedata"))
        st = empty_state()
        src = str(tmp_path / "proc.tif")
        sid = register_source(st, path=src, display_name="Proc", axes="ZYX")
        for s in st["sources"]:
            if s["source_id"] == sid:
                s["processed_store"] = {
                    "path": "artifacts/step1/largedata/proc__src_0_processed.zarr",
                    "recipe_hash": "deadbeef",
                }
        assert _ctl()._reconcile_sources_with_stores(ws, st) is True
        assert st["sources"][0]["processed_store"] is None

    def test_live_processed_store_pointer_kept(self, tmp_path):
        ws = str(tmp_path)
        store_rel = "artifacts/step1/largedata/live__src_0_processed.zarr"
        os.makedirs(os.path.join(ws, *store_rel.split("/")))
        st = empty_state()
        src = str(tmp_path / "live.tif")
        sid = register_source(st, path=src, display_name="Live", axes="ZYX")
        ptr = {"path": store_rel, "recipe_hash": "abc"}
        for s in st["sources"]:
            if s["source_id"] == sid:
                s["processed_store"] = ptr
        _ctl()._reconcile_sources_with_stores(ws, st)
        assert st["sources"][0]["processed_store"] == ptr

    def test_orphan_dedup_against_existing_path(self, tmp_path):
        """Two stores for the SAME source path + an existing row -> no dup row."""
        ws = str(tmp_path)
        src = _tif(tmp_path, "dup.tif")
        _prep_store(ws, "dup", src)
        _prep_store(ws, "dup__x", src)
        st = empty_state()
        register_source(st, path=src, display_name="Dup", axes="ZYX")
        _ctl()._reconcile_sources_with_stores(ws, st)
        assert len(st["sources"]) == 1

    def test_reconcile_idempotent(self, tmp_path):
        ws = str(tmp_path)
        src = _tif(tmp_path, "idem.tif")
        _prep_store(ws, "idem", src)
        st = empty_state()
        ctl = _ctl()
        ctl._reconcile_sources_with_stores(ws, st)
        n = len(st["sources"])
        changed2 = ctl._reconcile_sources_with_stores(ws, st)
        assert len(st["sources"]) == n
        assert changed2 is False


# ===========================================================================
# save-on-edit (debounced) + flush
# ===========================================================================

class TestSaveOnEdit:
    def _ws(self, tmp_path):
        ws = str(tmp_path / "ws")
        os.makedirs(ws)
        return ws

    def test_display_name_edit_persists(self, tmp_path):
        ws = self._ws(tmp_path)
        src = str(tmp_path / "s.tif")
        ctl = _ctl(ws)
        ctl._selected_source_path = src
        ctl.widget.display_name_edit.setText("Renamed")
        ctl.on_display_name_edited()
        ctl._flush_source_state_save()
        row = load_state(ws)["sources"][0]
        assert row["display_name"] == "Renamed"
        assert _norm_path(row["path"]) == _norm_path(src)

    def test_alias_edit_relabels_combo_live(self, tmp_path):
        """REGRESSION (Bug 1): typing an alias relabels the ② dropdown item in
        place (setItemText) — no reload, and no stale pre-flush registry read.
        Old behaviour: the combo kept the filename stem until a reload."""
        ws = self._ws(tmp_path)
        src = str(tmp_path / "kidney.tif")
        ctl = _ctl(ws)
        ctl._selected_source_path = src
        ctl.widget.source_combo.addItem("kidney")      # the stem label
        ctl.widget.source_combo.setCurrentIndex(0)
        ctl.widget.display_name_edit.setText("MyKidney")
        ctl.on_display_name_edited()
        assert ctl.widget.source_combo.itemText(0) == "MyKidney"   # live relabel

    def test_voxel_edit_persists_as_user(self, tmp_path):
        ws = self._ws(tmp_path)
        src = str(tmp_path / "v.tif")
        ctl = _ctl(ws)
        ctl._selected_source_path = src
        ctl.widget.voxel_z_spin.setValue(0.5)
        ctl.widget.voxel_y_spin.setValue(0.2)
        ctl.widget.voxel_x_spin.setValue(0.2)
        ctl.on_voxel_edited()
        ctl._flush_source_state_save()
        row = load_state(ws)["sources"][0]
        assert row["voxel_size"] == [0.5, 0.2, 0.2]
        assert row["voxel_size_source"] == "user"

    def test_voxel_default_typed_unknown_not_user(self, tmp_path):
        ws = self._ws(tmp_path)
        src = str(tmp_path / "v1.tif")
        ctl = _ctl(ws)
        ctl._selected_source_path = src
        # spins left at 1/1/1
        ctl.on_voxel_edited()
        ctl._flush_source_state_save()
        row = load_state(ws)["sources"][0]
        assert row["voxel_size_source"] == "unknown"

    def test_voxel_reset_clears_user_type(self, tmp_path):
        ws = self._ws(tmp_path)
        src = str(tmp_path / "vr.tif")
        # A prior user override on disk.
        st = empty_state()
        upsert_source(st, path=src, voxel_size=[0.5, 0.2, 0.2],
                      voxel_size_source="user")
        save_state(ws, st)
        ctl = _ctl(ws)
        ctl._selected_source_path = src
        ctl.widget._detected_voxel_size = (0.4, 0.1, 0.1)  # detected baseline
        ctl.on_voxel_reset()
        ctl._flush_source_state_save()
        row = load_state(ws)["sources"][0]
        assert row["voxel_size_source"] == "detected"    # no longer "user"
        assert row["voxel_size"] == [0.4, 0.1, 0.1]

    def test_voxel_reset_lazily_redetects_header_for_user_row(self, tmp_path):
        """FIX A companion: a user-override RAW TIFF whose file HAS header
        spacing is selected header-free (baseline left at the (1,1,1)
        placeholder), so ↺ must open the header LAZILY (once) and reset to the
        real detected spacing — persisted typed "detected" AND reflected in the
        spins — not the stale (1,1,1) "unknown"."""
        ws = self._ws(tmp_path)
        src = _tif(tmp_path, "lz.tif", shape=(5, 32, 32),
                   imagej=True, spacing=0.5, res=1.0 / 0.1)
        # A prior user override on disk.
        st = empty_state()
        upsert_source(st, path=src, voxel_size=[0.9, 0.3, 0.3],
                      voxel_size_source="user")
        save_state(ws, st)
        ctl = _ctl(ws)
        ctl._selected_source_path = src
        # SELECT left the baseline at the unit placeholder (header-free for a
        # user row — FIX A); ↺ must lazily recover the header spacing.
        ctl.widget._detected_voxel_size = (1.0, 1.0, 1.0)
        ctl.on_voxel_reset()
        ctl._flush_source_state_save()
        row = load_state(ws)["sources"][0]
        assert row["voxel_size"] == pytest.approx([0.5, 0.1, 0.1])
        assert row["voxel_size_source"] == "detected"    # header re-detected
        # The re-detected value is reflected in the widget (baseline + spins).
        assert ctl.widget._detected_voxel_size == pytest.approx((0.5, 0.1, 0.1))
        assert ctl.widget.applied_voxel[:3] == pytest.approx((0.5, 0.1, 0.1))
        assert ctl.widget.applied_voxel[3] == "detected"

    def test_voxel_reset_unit_header_stays_unknown(self, tmp_path):
        """A user-override RAW TIFF with NO physical header spacing: ↺ opens the
        header once, finds nothing physical, and keeps the (1,1,1) "unknown"
        behaviour (no false "detected")."""
        ws = self._ws(tmp_path)
        src = _tif(tmp_path, "lz_plain.tif")   # no spacing recorded
        st = empty_state()
        upsert_source(st, path=src, voxel_size=[0.9, 0.3, 0.3],
                      voxel_size_source="user")
        save_state(ws, st)
        ctl = _ctl(ws)
        ctl._selected_source_path = src
        ctl.widget._detected_voxel_size = (1.0, 1.0, 1.0)
        ctl.on_voxel_reset()
        ctl._flush_source_state_save()
        row = load_state(ws)["sources"][0]
        assert row["voxel_size_source"] == "unknown"
        assert row["voxel_size"] == [1.0, 1.0, 1.0]

    def test_axis_order_override_persists_source_axes(self, tmp_path):
        ws = self._ws(tmp_path)
        src = str(tmp_path / "ax.tif")
        ctl = _ctl(ws)
        ctl._axis_combo_source = src
        ctl.widget.axis_order_combo.setCurrentText("ZCYX")
        ctl.note_axis_order_override()
        ctl._flush_source_state_save()
        row = load_state(ws)["sources"][0]
        assert row["source_axes"] == "ZCYX"
        assert ctl._axis_order_overrides[_norm_path(src)] == "ZCYX"

    def test_empty_workspace_is_a_noop(self, tmp_path):
        ctl = _ctl("")   # no workspace
        ctl._selected_source_path = str(tmp_path / "x.tif")
        ctl.widget.display_name_edit.setText("Nope")
        ctl.on_display_name_edited()      # must not raise
        assert ctl._pending_source_edits == []


# ===========================================================================
# debounce flush-before-context-change (Step-2 gotcha)
# ===========================================================================

class TestFlushBeforeContextChange:
    def test_pending_save_lands_in_captured_workspace(self, tmp_path):
        ws_a = str(tmp_path / "A"); os.makedirs(ws_a)
        ws_b = str(tmp_path / "B"); os.makedirs(ws_b)
        src = str(tmp_path / "s.tif")
        w = _RecordWidget(ws_a)
        ctl = _ctl(widget=w)
        ctl._selected_source_path = src
        # Edit made against workspace A.
        w.display_name_edit.setText("FromA")
        ctl.on_display_name_edited()
        assert ctl._pending_source_ws == ws_a
        # The workspace text now changes to B; the flush uses the CAPTURED ws.
        w.output_file_path.setText(ws_b)
        ctl._flush_source_state_save()
        # A got the edit; B did NOT (the wrong-slot bug).
        assert load_state(ws_a)["sources"][0]["display_name"] == "FromA"
        assert load_state(ws_b)["sources"] == []

    def test_queue_across_workspaces_flushes_old_first(self, tmp_path):
        ws_a = str(tmp_path / "A"); os.makedirs(ws_a)
        ws_b = str(tmp_path / "B"); os.makedirs(ws_b)
        src_a = str(tmp_path / "a.tif")
        src_b = str(tmp_path / "b.tif")
        w = _RecordWidget(ws_a)
        ctl = _ctl(widget=w)
        ctl._selected_source_path = src_a
        w.display_name_edit.setText("A1")
        ctl.on_display_name_edited()
        # Switch workspace + queue a new edit -> the pending A edit auto-flushes.
        w.output_file_path.setText(ws_b)
        ctl._selected_source_path = src_b
        w.display_name_edit.setText("B1")
        ctl.on_display_name_edited()
        ctl._flush_source_state_save()
        assert load_state(ws_a)["sources"][0]["display_name"] == "A1"
        assert load_state(ws_b)["sources"][0]["display_name"] == "B1"

    def test_flush_reads_fresh_state_preserving_concurrent_write(self, tmp_path):
        """A Prepare/Export writing the registry between queue and flush must not
        be clobbered — flush re-reads disk and layers the edits on top."""
        ws = str(tmp_path / "ws"); os.makedirs(ws)
        src = str(tmp_path / "s.tif")
        w = _RecordWidget(ws)
        ctl = _ctl(widget=w)
        ctl._selected_source_path = src
        w.display_name_edit.setText("Alias")
        ctl.on_display_name_edited()   # queued, not yet flushed
        # Concurrent write (like _persist_prepared_source): a store pointer.
        st = load_state(ws)
        upsert_source(st, path=src, prepared_store="artifacts/step1/largedata/s.zarr")
        save_state(ws, st)
        ctl._flush_source_state_save()
        row = load_state(ws)["sources"][0]
        assert row["display_name"] == "Alias"                     # our edit
        assert row["prepared_store"] == "artifacts/step1/largedata/s.zarr"  # preserved


# ===========================================================================
# restore into widgets
# ===========================================================================

class TestRestoreWidgets:
    def test_restores_saved_alias(self, tmp_path):
        src = str(tmp_path / "s.tif")
        ctl = _ctl(str(tmp_path))
        row = {"display_name": "SavedAlias"}
        ctl._restore_source_widgets_for(src, row)
        assert ctl.widget.display_name_edit.text() == "SavedAlias"

    def test_falls_back_to_stem_without_row(self, tmp_path):
        src = str(tmp_path / "kidney.tif")
        ctl = _ctl(str(tmp_path))
        ctl._restore_source_widgets_for(src, None)
        assert ctl.widget.display_name_edit.text() == "kidney"

    def test_seeds_axis_override_from_saved_axes(self, tmp_path):
        src = str(tmp_path / "s.tif")
        ctl = _ctl(str(tmp_path))
        # A scanned 4D shape so `_apply_detected_axes_for_source` applies it.
        ctl._scanned_files = [{"path": src, "shape": (2, 10, 20, 20)}]
        ctl._restore_source_widgets_for(src, {"source_axes": "ZCYX"})
        assert ctl._axis_order_overrides[_norm_path(src)] == "ZCYX"
        assert ctl.widget.applied_axis[0] == "ZCYX"

    def test_seeds_axis_override_without_a_scan(self, tmp_path):
        """REGRESSION (Bug 2): after a restart `_scanned_files` is EMPTY (restore
        repopulates from the registry with NO folder re-scan).  The saved order
        must STILL reach the combo — the override check runs BEFORE the shape
        guard.  The old order early-returned on the empty scan, leaving the combo
        at its "CZYX" default (which swaps C/Z on a raw ZCYX extract)."""
        src = str(tmp_path / "s.tif")
        ctl = _ctl(str(tmp_path))
        ctl._scanned_files = []                        # <-- restart: nothing scanned
        ctl._restore_source_widgets_for(src, {"source_axes": "ZCYX"})
        assert ctl._axis_order_overrides[_norm_path(src)] == "ZCYX"
        assert ctl.widget.applied_axis[0] == "ZCYX"    # NOT the CZYX default
        assert ctl._axis_combo_source == src           # combo now describes this source

    def test_restores_user_voxel_to_spins(self, tmp_path):
        src = str(tmp_path / "s.tif")
        ctl = _ctl(str(tmp_path))
        row = {"voxel_size": [0.7, 0.3, 0.3], "voxel_size_source": "user"}
        ctl._restore_source_widgets_for(src, row)
        assert ctl.widget.applied_voxel == (0.7, 0.3, 0.3, "user")


# ===========================================================================
# preprocessing recipe round-trip
# ===========================================================================

class TestRecipePersistence:
    """The ② preprocessing recipe round-trips like alias/voxel: an edit
    persists it onto the source's registry row (path-keyed), and restore pushes
    it back into the columns.  Stored as a DICT, never None (a cleared recipe is
    an empty-but-present dict), so `upsert_source`'s skip-None can't leave a
    stale recipe on disk."""

    class _RecipeWidget(_RecordWidget):
        def __init__(self, ws=""):
            super().__init__(ws)
            self._recipe_json = None
            self.restored_recipe = "UNSET"

        def get_recipe_json(self):
            return self._recipe_json

        def set_recipe_from_json(self, raw):
            self.restored_recipe = raw

    def test_edit_persists_recipe_on_the_source_row(self, tmp_path):
        ws = str(tmp_path)
        w = self._RecipeWidget(ws)
        ctl = _ctl(ws, widget=w)
        ctl._selected_source_path = str(tmp_path / "src.tif")
        w._recipe_json = {
            "channels": [[{"key": "gaussian", "params": {"sigma": 1.5}}]],
            "geometry": None,
        }
        ctl.on_recipe_edited()
        ctl._flush_source_state_save()
        row = ctl._find_source_row(load_state(ws), ctl._selected_source_path)
        assert row is not None            # a recipe makes the source worked-with
        assert row["recipe"] == w._recipe_json

    def test_no_selected_source_is_a_no_op(self, tmp_path):
        ws = str(tmp_path)
        w = self._RecipeWidget(ws)
        ctl = _ctl(ws, widget=w)
        ctl._selected_source_path = None
        w._recipe_json = {"channels": [], "geometry": None}
        ctl.on_recipe_edited()            # nothing selected -> nothing queued
        assert ctl._pending_source_edits == []

    def test_clear_persists_present_recipe_not_none(self, tmp_path):
        ws = str(tmp_path)
        w = self._RecipeWidget(ws)
        ctl = _ctl(ws, widget=w)
        ctl._selected_source_path = str(tmp_path / "src.tif")
        w._recipe_json = {"channels": [[{"key": "gaussian", "params": {}}]],
                          "geometry": None}
        ctl.on_recipe_edited(); ctl._flush_source_state_save()
        # Clearing writes an EMPTY-but-present recipe, not None.
        w._recipe_json = {"channels": [[]], "geometry": None}
        ctl.on_recipe_edited(); ctl._flush_source_state_save()
        row = ctl._find_source_row(load_state(ws), ctl._selected_source_path)
        assert row["recipe"] == {"channels": [[]], "geometry": None}

    def test_restore_pushes_saved_recipe_into_columns(self, tmp_path):
        src = str(tmp_path / "s.tif")
        ctl = _ctl(str(tmp_path), widget=self._RecipeWidget(str(tmp_path)))
        recipe = {"channels": [[{"key": "gaussian", "params": {}}]],
                  "geometry": None}
        ctl._restore_source_widgets_for(src, {"recipe": recipe})
        assert ctl.widget.restored_recipe == recipe

    def test_restore_without_a_row_resets_recipe(self, tmp_path):
        src = str(tmp_path / "s.tif")
        ctl = _ctl(str(tmp_path), widget=self._RecipeWidget(str(tmp_path)))
        ctl._restore_source_widgets_for(src, None)
        assert ctl.widget.restored_recipe is None    # reset to empty on select


# ===========================================================================
# FIX 1 (HIGH): the restore funnel runs for EVERY selection route (mouse,
# keyboard, combo-pick), not just mouse — the silent-data-corruption bug.
# ===========================================================================

class TestSelectionRestoreAllRoutes:
    def _two_sources(self, tmp_path):
        ws = str(tmp_path / "ws"); os.makedirs(ws)
        a = str(tmp_path / "a.tif")
        b = str(tmp_path / "b.tif")
        st = empty_state()
        upsert_source(st, path=a, display_name="A",
                      voxel_size=[0.9, 0.3, 0.3], voxel_size_source="user")
        upsert_source(st, path=b, display_name="B",
                      voxel_size=[0.5, 0.2, 0.2], voxel_size_source="user")
        save_state(ws, st)
        return ws, a, b

    def test_funnel_restores_each_source_voxel(self, tmp_path):
        """The single funnel `_on_source_current_item_changed` (fired by mouse,
        keyboard AND combo-pick's setCurrentRow) restores the NEW source's voxel.
        A revert of FIX 1 (funnel does axes only) leaves A's voxel in the spins
        when B is selected -> this asserts B's, so it FAILS on the revert."""
        ws, a, b = self._two_sources(tmp_path)
        w = _RecordWidget(ws)
        ctl = _ctl(widget=w)
        ctl._on_source_current_item_changed(_FakeItem(a))
        assert w.applied_voxel == (0.9, 0.3, 0.3, "user")
        assert ctl._selected_source_path == a
        # A second selection via the SAME funnel (combo-pick / keyboard route).
        ctl._on_source_current_item_changed(_FakeItem(b))
        assert w.applied_voxel == (0.5, 0.2, 0.2, "user")   # B's, NOT A's
        assert ctl._selected_source_path == b

    def test_reset_persists_the_currently_selected_source(self, tmp_path):
        """select A, then select B, then reset -> the cleared baseline persists on
        B (the selected source), NOT A (the FIX-1 failure would target A)."""
        ws, a, b = self._two_sources(tmp_path)
        w = _RecordWidget(ws)
        ctl = _ctl(widget=w)
        ctl._on_source_current_item_changed(_FakeItem(a))
        ctl._on_source_current_item_changed(_FakeItem(b))    # B now selected
        w._detected_voxel_size = (0.4, 0.1, 0.1)
        ctl.on_voxel_reset()
        ctl._flush_source_state_save()
        by_path = {_norm_path(s["path"]): s for s in load_state(ws)["sources"]}
        assert by_path[_norm_path(b)]["voxel_size"] == [0.4, 0.1, 0.1]
        assert by_path[_norm_path(b)]["voxel_size_source"] == "detected"
        assert by_path[_norm_path(a)]["voxel_size"] == [0.9, 0.3, 0.3]  # A untouched

    def test_reselecting_same_source_is_idempotent(self, tmp_path):
        ws, a, b = self._two_sources(tmp_path)
        w = _RecordWidget(ws)
        ctl = _ctl(widget=w)
        ctl._on_source_current_item_changed(_FakeItem(a))
        n_after_first = len(ctl._pending_source_edits)
        ctl._on_source_current_item_changed(_FakeItem(a))   # same -> no-op
        assert len(ctl._pending_source_edits) == n_after_first

    def test_suspend_flag_blocks_restore_during_populate(self, tmp_path):
        ws, a, b = self._two_sources(tmp_path)
        w = _RecordWidget(ws)
        ctl = _ctl(widget=w)
        ctl._suspend_source_restore = True
        ctl._on_source_current_item_changed(_FakeItem(a))
        assert w.applied_voxel is None            # no restore
        assert ctl._selected_source_path is None  # no selection recorded


class TestRestoreDoesNotSave:
    def test_pure_restore_queues_no_save(self, tmp_path):
        """`_restore_source_widgets_for` must NOT persist (it only READS the
        registry).  A spy fails if restore ever re-fires a save (A9)."""
        src = str(tmp_path / "s.tif")
        ctl = _ctl(str(tmp_path))
        calls = []
        ctl._queue_source_save = lambda **kw: calls.append(kw)
        ctl._restore_source_widgets_for(
            src, {"display_name": "X", "voxel_size": [0.5, 0.2, 0.2],
                  "voxel_size_source": "user", "source_axes": "ZCYX"})
        assert calls == []


# ===========================================================================
# Worked-with persistence: a passive select does NOT persist; a real action
# (edit / Prepare / Extract) creates the path-keyed row, which a later export's
# register_source then dedups (no duplicate) — so B2-N1 is moot (no browsed row).
# ===========================================================================

class TestWorkedWithPersist:
    def test_passive_select_does_not_persist(self, tmp_path):
        """A genuine (passive) select writes NO registry row — only a real
        action (edit / Prepare / Extract) persists a source."""
        ws = str(tmp_path / "ws"); os.makedirs(ws)
        src = _tif(tmp_path, "unworked.tif", shape=(5, 32, 32))
        ctl = _ctl(widget=_RecordWidget(ws))
        ctl._on_source_current_item_changed(_FakeItem(src))   # genuine select
        ctl._flush_source_state_save()
        assert load_state(ws).get("sources", []) == []        # nothing persisted
        assert ctl._selected_source_path == src               # tracked in-memory only

    def test_edited_row_export_dedups_single_row(self, tmp_path):
        ws = str(tmp_path / "ws"); os.makedirs(ws)
        src = str(tmp_path / "s.tif")
        # A path-keyed row created by an EDIT / Prepare (NOT a passive select).
        w = _RecordWidget(ws)
        ctl = _ctl(widget=w)
        ctl._queue_source_save(path=src, source_id=None, origin=["browsed"],
                               fingerprint={"size": 1, "mtime": 2.0})
        ctl._flush_source_state_save()
        st = load_state(ws)
        assert len(st["sources"]) == 1
        # Later export via the path-keyed register_source must dedup, not append.
        sid = register_source(st, path=src, display_name="X", axes="ZYX",
                              prefix="X_patch")
        assert len(st["sources"]) == 1
        assert sid == st["sources"][0]["source_id"]


# ===========================================================================
# alias in combo + tooltip + pick-clobber fix — real widget
# ===========================================================================

@pytest.fixture(scope="module")
def qt_app():
    pytest.importorskip("qtpy")
    from qtpy.QtWidgets import QApplication
    existing = QApplication.instance()
    if existing is not None:
        yield existing
        return
    yield QApplication([])


class _RealImageViewer:
    def __init__(self):
        from qtpy.QtCore import QObject

        class _Ev:
            def connect(self, *_):
                pass

            def disconnect(self, *_):
                pass

        class _Events:
            def __init__(self):
                self.inserted = _Ev()
                self.removed = _Ev()

        class _Layers(list):
            def __init__(self):
                super().__init__()
                self.events = _Events()

            def __contains__(self, item):
                if isinstance(item, str):
                    return any(getattr(l, "name", None) == item for l in self)
                return list.__contains__(self, item)

        self.layers = _Layers()
        self.mouse_drag_callbacks = []


class TestAliasInCombo:
    def _widget(self, ws):
        from napari_epicycle.data_preparation.ui import DataPreparationWidget
        w = DataPreparationWidget(viewer=_RealImageViewer())
        w.output_file_path.setText(ws)
        return w

    def _add_item(self, w, path, stem):
        from qtpy.QtWidgets import QListWidgetItem
        from qtpy.QtCore import Qt
        item = QListWidgetItem()
        item.setData(Qt.UserRole, path)
        item.setData(Qt.UserRole + 1, stem)
        w.source_file_list.addItem(item)
        return item

    def test_combo_labels_registered_source_by_alias(self, qt_app, tmp_path):
        from qtpy.QtCore import Qt
        ws = str(tmp_path / "ws"); os.makedirs(ws)
        src = str(tmp_path / "kidney.tif")
        st = empty_state()
        upsert_source(st, path=src, display_name="MyKidney")
        save_state(ws, st)
        w = self._widget(ws)
        try:
            self._add_item(w, src, "kidney")
            w._rebuild_source_combo()
            assert w.source_combo.itemText(0) == "MyKidney"
            tip = w.source_combo.itemData(0, Qt.ToolTipRole)
            assert "kidney" in tip and src in tip
        finally:
            w.setParent(None)

    def test_unregistered_source_labels_by_stem(self, qt_app, tmp_path):
        ws = str(tmp_path / "ws2"); os.makedirs(ws)
        src = str(tmp_path / "raw.tif")
        w = self._widget(ws)
        try:
            self._add_item(w, src, "raw")
            w._rebuild_source_combo()
            assert w.source_combo.itemText(0) == "raw"
        finally:
            w.setParent(None)

    def test_combo_pick_restores_saved_alias_not_stem(self, qt_app, tmp_path):
        ws = str(tmp_path / "ws3"); os.makedirs(ws)
        src = str(tmp_path / "kidney.tif")
        st = empty_state()
        upsert_source(st, path=src, display_name="MyKidney")
        save_state(ws, st)
        w = self._widget(ws)
        try:
            self._add_item(w, src, "kidney")
            w._rebuild_source_combo()
            w.display_name_edit.setText("")   # nothing hand-typed
            w._auto_alias = ""
            w._on_source_combo_picked(0)
            assert w.display_name_edit.text() == "MyKidney"
        finally:
            w.setParent(None)

    def test_repick_same_source_is_idempotent_keeps_hand_typed(self, qt_app, tmp_path):
        """Re-picking the ALREADY-selected source is a no-op (idempotency guard),
        so an in-progress hand-typed alias on the current source is not clobbered
        by a redundant pick."""
        ws = str(tmp_path / "ws4"); os.makedirs(ws)
        src = str(tmp_path / "kidney.tif")
        w = self._widget(ws)
        try:
            self._add_item(w, src, "kidney")
            w._rebuild_source_combo()
            # Make kidney the current selection first.
            w._on_source_combo_picked(0)
            # Hand-type an alias (unsaved), then re-pick the SAME source.
            w.display_name_edit.setText("HandTyped")
            w._on_source_combo_picked(0)   # same source -> idempotent no-op
            assert w.display_name_edit.text() == "HandTyped"
        finally:
            w.setParent(None)


# ===========================================================================
# state-file problems are surfaced INLINE, never only in Diagnostics
#
# ===========================================================================

class _StatusLabel:
    def __init__(self):
        self.text_ = ""
        self.style = ""

    def setText(self, t):
        self.text_ = t

    def setStyleSheet(self, s):
        self.style = s


class TestStateProblemsAreVisible:
    def _ctl_with_status(self, ws):
        ctl = _ctl(ws)
        ctl.widget.workspace_status_label = _StatusLabel()
        return ctl

    def test_corrupt_state_on_restore_shows_inline_warning(self, tmp_path):
        """A truncated state file is quarantined AND announced in ① — the user
        must never silently continue on an empty state."""
        ws = str(tmp_path / "ws"); os.makedirs(ws)
        step1_state_path(ws).write_text('{"sources": [', encoding="utf-8")
        ctl = self._ctl_with_status(ws)

        ctl._restore_step1_sources(ws)

        lbl = ctl.widget.workspace_status_label
        assert "step1_state.json" in lbl.text_
        assert "corrupt-" in lbl.text_
        assert "#e07070" in lbl.style          # the red inline-error colour
        # quarantined, not overwritten
        assert any(".corrupt-" in f for f in os.listdir(ws))

    def test_healthy_state_leaves_the_status_label_alone(self, tmp_path):
        ws = str(tmp_path / "ws_ok"); os.makedirs(ws)
        save_state(ws, empty_state())
        ctl = self._ctl_with_status(ws)

        ctl._restore_step1_sources(ws)

        assert ctl.widget.workspace_status_label.text_ == ""

    def test_failed_save_flush_is_shown_inline(self, tmp_path, monkeypatch):
        """A failed persist (disk full / OneDrive lock) was Diagnostics-only."""
        ws = str(tmp_path / "ws_fail"); os.makedirs(ws)
        ctl = self._ctl_with_status(ws)
        ctl._selected_source_path = str(tmp_path / "s.tif")
        ctl.widget.display_name_edit.setText("Renamed")
        ctl.on_display_name_edited()

        import napari_epicycle.data_preparation.controller as C

        def boom(*_a, **_k):
            raise OSError("disk full")

        monkeypatch.setattr(C, "save_state", boom)
        ctl._flush_source_state_save()

        lbl = ctl.widget.workspace_status_label
        assert "disk full" in lbl.text_
        assert "#e07070" in lbl.style
        # the edit is KEPT for a retry (existing invariant, unchanged)
        assert ctl._pending_source_edits


class TestStateStatusHygiene:
    """Brick 1b #8.  `_report_state_problem` paints a RED workspace status that
    nothing ever cleared: a transient failure (OneDrive lock, momentarily full
    disk) left "⚠ source save flush FAILED" on screen for the rest of the
    session even after the very next save succeeded.  And a failed flush kept
    the edits but stopped RETRYING — only a further user edit rescheduled the
    timer, so the last edit before a lull could sit unsaved indefinitely."""

    def _ctl_with_status(self, ws):
        from napari_epicycle.data_preparation.ui import DataPreparationWidget

        ctl = _ctl(ws)
        ctl.widget.workspace_status_label = _StatusLabel()
        # Reuse the real ①-status renderer (the one the path field drives), so
        # the restored text can never drift from what the user normally sees.
        ctl.widget._update_workspace_status = types.MethodType(
            DataPreparationWidget._update_workspace_status, ctl.widget)
        return ctl

    def _queue_one_edit(self, ctl, tmp_path):
        ctl._selected_source_path = str(tmp_path / "s.tif")
        ctl.widget.display_name_edit.setText("Renamed")
        ctl.on_display_name_edited()

    def test_successful_flush_clears_a_stale_red_warning(self, tmp_path):
        ws = str(tmp_path / "ws_ok2"); os.makedirs(ws)
        ctl = self._ctl_with_status(ws)
        ctl._report_state_problem("STEP1-SAVE", "an earlier transient failure")
        assert "#e07070" in ctl.widget.workspace_status_label.style

        self._queue_one_edit(ctl, tmp_path)
        ctl._flush_source_state_save()

        lbl = ctl.widget.workspace_status_label
        assert "FAILED" not in lbl.text_
        assert "an earlier transient failure" not in lbl.text_
        assert "#e07070" not in lbl.style          # no longer the error colour
        # ... and it says what the normal ① status says for this workspace
        assert "workspace" in lbl.text_.lower() or "directory" in lbl.text_.lower()

    def test_successful_flush_still_persists_the_edit(self, tmp_path):
        ws = str(tmp_path / "ws_ok3"); os.makedirs(ws)
        ctl = self._ctl_with_status(ws)
        self._queue_one_edit(ctl, tmp_path)
        ctl._flush_source_state_save()
        assert load_state(ws)["sources"][0]["display_name"] == "Renamed"
        assert ctl._pending_source_edits == []

    def test_a_no_op_flush_does_not_repaint_the_status(self, tmp_path):
        """Nothing pending -> the early return must not clobber whatever the ①
        status currently says (incl. a genuine, still-relevant warning)."""
        ws = str(tmp_path / "ws_ok4"); os.makedirs(ws)
        ctl = self._ctl_with_status(ws)
        ctl._report_state_problem("STEP1-STATE", "state file is corrupt")
        ctl._flush_source_state_save()
        assert "corrupt" in ctl.widget.workspace_status_label.text_

    def test_failed_flush_reschedules_a_retry(self, tmp_path, monkeypatch):
        ws = str(tmp_path / "ws_retry"); os.makedirs(ws)
        ctl = self._ctl_with_status(ws)
        self._queue_one_edit(ctl, tmp_path)
        before = ctl._state_save_timer.started

        import napari_epicycle.data_preparation.controller as C

        def boom(*_a, **_k):
            raise OSError("disk full")

        monkeypatch.setattr(C, "save_state", boom)
        ctl._flush_source_state_save()

        assert ctl._pending_source_edits          # kept (existing invariant)
        assert ctl._state_save_timer.started > before, (
            "a failed save must reschedule itself, not wait for a user edit")
        assert ctl._state_save_timer.start_args[-1] == (4000,)

    def test_the_retry_eventually_succeeds_and_clears_the_warning(
            self, tmp_path, monkeypatch):
        ws = str(tmp_path / "ws_retry2"); os.makedirs(ws)
        ctl = self._ctl_with_status(ws)
        self._queue_one_edit(ctl, tmp_path)

        import napari_epicycle.data_preparation.controller as C

        real_save = C.save_state
        monkeypatch.setattr(
            C, "save_state",
            lambda *_a, **_k: (_ for _ in ()).throw(OSError("locked")))
        ctl._flush_source_state_save()
        assert "#e07070" in ctl.widget.workspace_status_label.style

        monkeypatch.setattr(C, "save_state", real_save)
        ctl._flush_source_state_save()             # the rescheduled retry

        assert load_state(ws)["sources"][0]["display_name"] == "Renamed"
        assert "#e07070" not in ctl.widget.workspace_status_label.style


class TestFailedSaveBackoff:
    """Brick 1c #4.  The flat 4 s retry hammered a genuinely-unavailable
    workspace (a disconnected network share) forever, and re-`_log`ged the SAME
    line every 4 s — a Diagnostics pane that scrolls one message off the top."""

    def _ctl_with_status(self, ws):
        from napari_epicycle.data_preparation.ui import DataPreparationWidget

        ctl = _ctl(ws)
        ctl.widget.workspace_status_label = _StatusLabel()
        ctl.widget._update_workspace_status = types.MethodType(
            DataPreparationWidget._update_workspace_status, ctl.widget)
        ctl._logged = []
        ctl._log = lambda p, m: ctl._logged.append((p, m))
        return ctl

    def _queue(self, ctl, tmp_path):
        ctl._selected_source_path = str(tmp_path / "s.tif")
        ctl.widget.display_name_edit.setText("Renamed")
        ctl.on_display_name_edited()

    def _fail(self, monkeypatch):
        import napari_epicycle.data_preparation.controller as C

        monkeypatch.setattr(
            C, "save_state",
            lambda *_a, **_k: (_ for _ in ()).throw(OSError("locked")))

    def test_the_retry_delay_doubles_and_caps(self, tmp_path, monkeypatch):
        ws = str(tmp_path / "ws_bo"); os.makedirs(ws)
        ctl = self._ctl_with_status(ws)
        self._queue(ctl, tmp_path)
        self._fail(monkeypatch)
        n0 = len(ctl._state_save_timer.start_args)   # skip the ~400ms debounce

        for _ in range(8):
            ctl._flush_source_state_save()

        delays = [a[0] for a in ctl._state_save_timer.start_args[n0:]]
        assert delays[:4] == [4000, 8000, 16000, 30000]
        assert all(d <= 30000 for d in delays)     # capped, never unbounded

    def test_the_repeated_message_is_reported_once_per_streak(
            self, tmp_path, monkeypatch):
        ws = str(tmp_path / "ws_dedup"); os.makedirs(ws)
        ctl = self._ctl_with_status(ws)
        self._queue(ctl, tmp_path)
        self._fail(monkeypatch)

        for _ in range(5):
            ctl._flush_source_state_save()

        saves = [m for (p, m) in ctl._logged if "FAILED" in m]
        assert len(saves) == 1, f"the same failure was logged {len(saves)}x"
        # the red inline status is still up for the whole streak
        assert "#e07070" in ctl.widget.workspace_status_label.style

    def test_recovery_says_how_many_retries_it_took(self, tmp_path, monkeypatch):
        ws = str(tmp_path / "ws_rec"); os.makedirs(ws)
        ctl = self._ctl_with_status(ws)
        self._queue(ctl, tmp_path)

        import napari_epicycle.data_preparation.controller as C

        real = C.save_state
        self._fail(monkeypatch)
        ctl._flush_source_state_save()
        ctl._flush_source_state_save()
        monkeypatch.setattr(C, "save_state", real)
        ctl._flush_source_state_save()

        assert load_state(ws)["sources"][0]["display_name"] == "Renamed"
        recovered = [m for (p, m) in ctl._logged if "retries" in m]
        assert len(recovered) == 1
        assert "2" in recovered[0]
        assert "#e07070" not in ctl.widget.workspace_status_label.style

    def test_a_new_failure_streak_reports_again(self, tmp_path, monkeypatch):
        """Dedup is per STREAK — a later failure must not be swallowed."""
        ws = str(tmp_path / "ws_streak"); os.makedirs(ws)
        ctl = self._ctl_with_status(ws)
        self._queue(ctl, tmp_path)

        import napari_epicycle.data_preparation.controller as C

        real = C.save_state
        self._fail(monkeypatch)
        ctl._flush_source_state_save()
        monkeypatch.setattr(C, "save_state", real)
        ctl._flush_source_state_save()             # recovered
        self._queue(ctl, tmp_path)
        self._fail(monkeypatch)
        ctl._flush_source_state_save()             # a NEW failure

        assert len([m for (p, m) in ctl._logged if "FAILED" in m]) == 2
        assert ctl._state_save_timer.start_args[-1] == (4000,)   # backoff reset


    def test_a_clean_run_never_logs_a_recovery_line(self, tmp_path):
        ws = str(tmp_path / "ws_clean"); os.makedirs(ws)
        ctl = self._ctl_with_status(ws)
        self._queue(ctl, tmp_path)
        ctl._flush_source_state_save()
        assert [m for (p, m) in ctl._logged if "retries" in m] == []


# ===========================================================================
# recipe edited with NO source selected (Brick 3b — item 1)
# ===========================================================================

class TestPendingRecipeAdoption:
    """A recipe edit made before any source row is selected used to be dropped
    on the floor.  It is now HELD and adopted by the next selected source — but
    ONLY when that source has no saved recipe of its own (a saved recipe always
    wins; the held one is discarded with a one-line status)."""

    class _W(_RecordWidget):
        def __init__(self, ws=""):
            super().__init__(ws)
            self._recipe_json = None
            self.restored_recipe = "UNSET"
            self.status = []

        def get_recipe_json(self):
            return self._recipe_json

        def set_recipe_from_json(self, raw):
            self.restored_recipe = raw

        def set_preview_status(self, message, ok=True):
            self.status.append((message, ok))

    @staticmethod
    def _ctl_ready(ws, w):
        ctl = _ctl(ws, widget=w)
        ctl._pending_recipe_for_next_selection = None
        ctl._bump_preview_token = lambda *a: None
        ctl._clear_preview_layers = lambda *a: None
        return ctl

    _RECIPE = {"channels": [[{"key": "gaussian", "params": {"sigma": 1.5}}]],
               "geometry": None}

    def test_edit_with_no_selection_is_held_not_dropped(self, tmp_path):
        ws = str(tmp_path)
        w = self._W(ws)
        ctl = self._ctl_ready(ws, w)
        w._recipe_json = self._RECIPE
        ctl.on_recipe_edited()
        assert ctl._pending_source_edits == []          # nothing written yet
        assert ctl._pending_recipe_for_next_selection == self._RECIPE
        assert "Select a source" in w.status[-1][0]

    def test_next_selection_without_a_recipe_adopts_it(self, tmp_path):
        ws = str(tmp_path)
        w = self._W(ws)
        ctl = self._ctl_ready(ws, w)
        w._recipe_json = self._RECIPE
        ctl.on_recipe_edited()
        src = str(tmp_path / "src.tif")
        ctl._select_source(src)
        ctl._flush_source_state_save()
        row = ctl._find_source_row(load_state(ws), src)
        assert row is not None and row["recipe"] == self._RECIPE
        assert w.restored_recipe == self._RECIPE        # pushed into the columns
        assert ctl._pending_recipe_for_next_selection is None
        assert "kept with" in w.status[-1][0]

    def test_a_saved_recipe_wins_and_is_never_overwritten(self, tmp_path):
        ws = str(tmp_path)
        src = str(tmp_path / "src.tif")
        saved = {"channels": [[{"key": "clahe", "params": {}}]], "geometry": None}
        state = empty_state()
        upsert_source(state, path=src, recipe=saved, display_name="My source")
        save_state(ws, state)

        w = self._W(ws)
        ctl = self._ctl_ready(ws, w)
        w._recipe_json = self._RECIPE
        ctl.on_recipe_edited()
        ctl._select_source(src)
        ctl._flush_source_state_save()
        row = ctl._find_source_row(load_state(ws), src)
        assert row["recipe"] == saved                   # NOT overwritten
        assert w.restored_recipe == saved               # columns keep the saved one
        assert ctl._pending_recipe_for_next_selection is None
        assert "My source" in w.status[-1][0]

    def test_an_empty_recipe_is_not_held(self, tmp_path):
        ws = str(tmp_path)
        w = self._W(ws)
        ctl = self._ctl_ready(ws, w)
        w._recipe_json = {"channels": [[]], "geometry": None}
        ctl.on_recipe_edited()
        assert ctl._pending_recipe_for_next_selection is None
        assert w.status == []


class TestPendingRecipeHygiene:
    """Brick 3d note 2 — a HELD recipe must not outlive the context it was
    built in, and must not be adopted where it would silently lose steps."""

    _W = TestPendingRecipeAdoption._W
    _RECIPE = TestPendingRecipeAdoption._RECIPE

    def _ctl(self, tmp_path):
        w = self._W(str(tmp_path))
        ctl = TestPendingRecipeAdoption._ctl_ready(str(tmp_path), w)
        return ctl, w

    def test_a_context_change_drops_the_held_recipe_and_clears_the_panel(
            self, tmp_path):
        ctl, w = self._ctl(tmp_path)
        w._recipe_json = self._RECIPE
        ctl.on_recipe_edited()
        assert ctl._pending_recipe_for_next_selection is not None

        ctl._drop_pending_recipe("changed folder")
        assert ctl._pending_recipe_for_next_selection is None
        assert w.restored_recipe is None          # panel reset, not a phantom
        assert "changed folder" in w.status[-1][0]

    def test_dropping_nothing_is_silent(self, tmp_path):
        ctl, w = self._ctl(tmp_path)
        ctl._drop_pending_recipe("changed folder")
        assert w.status == []
        assert w.restored_recipe == "UNSET"        # panel untouched

    def test_browse_and_workspace_change_route_through_the_drop(self):
        import inspect
        from napari_epicycle.data_preparation.controller import (
            DataPreparationController as C)

        assert "_drop_pending_recipe" in inspect.getsource(C.browse_source_folder)
        assert "_drop_pending_recipe" in inspect.getsource(C.__init__)

    def test_adoption_is_refused_when_the_recipe_has_more_channels(
            self, tmp_path):
        ctl, w = self._ctl(tmp_path)
        src = str(tmp_path / "src.tif")
        # A 3-channel recipe held; the incoming source is single-channel.
        w._recipe_json = {
            "channels": [[{"key": "gaussian", "params": {}}],
                         [], [{"key": "clahe", "params": {}}]],
            "geometry": None,
        }
        ctl.on_recipe_edited()
        ctl._known_channel_count_for = lambda _p: 1
        ctl._select_source(src)
        ctl._flush_source_state_save()
        assert ctl._find_source_row(load_state(str(tmp_path)), src) is None
        assert ctl._pending_recipe_for_next_selection is None
        assert "channel" in w.status[-1][0].lower()

    def test_a_fewer_channel_recipe_is_still_adopted(self, tmp_path):
        ctl, w = self._ctl(tmp_path)
        src = str(tmp_path / "src.tif")
        w._recipe_json = self._RECIPE          # one channel list
        ctl.on_recipe_edited()
        ctl._known_channel_count_for = lambda _p: 3
        ctl._select_source(src)
        ctl._flush_source_state_save()
        row = ctl._find_source_row(load_state(str(tmp_path)), src)
        assert row is not None and row["recipe"] == self._RECIPE

    def test_an_unknown_channel_count_never_blocks_adoption(self, tmp_path):
        ctl, w = self._ctl(tmp_path)
        src = str(tmp_path / "src.tif")
        w._recipe_json = self._RECIPE
        ctl.on_recipe_edited()
        ctl._known_channel_count_for = lambda _p: None
        ctl._select_source(src)
        ctl._flush_source_state_save()
        assert ctl._find_source_row(load_state(str(tmp_path)), src) is not None


# ===========================================================================
# The spins' type travels to the export; a "detected" row; Z assumed = XY
# ===========================================================================

class _TypedWidget(_RecordWidget):
    """`_RecordWidget` plus the type the panel shows and the Z-assumed note."""

    def __init__(self, ws="", vtype="detected"):
        super().__init__(ws)
        self._vtype = vtype
        self.z_assumed = None

    def voxel_source_type(self):
        return self._vtype

    def set_voxel_z_assumed(self, flag):
        self.z_assumed = bool(flag)

    def set_voxel_detected_note(self, note):
        self.note = note


class _SpinWidget(_TypedWidget):
    """Like the real widget: a pushed voxel lands in the spins and sets the
    type the panel shows."""

    def __init__(self, ws="", vtype="unknown"):
        super().__init__(ws, vtype)
        self.note = None

    def apply_voxel_size(self, z, y, x, source="detected"):
        super().apply_voxel_size(z, y, x, source)
        for s, v in zip((self.voxel_z_spin, self.voxel_y_spin,
                         self.voxel_x_spin), (z, y, x)):
            s.setValue(v)
        self._vtype = source

    def spins(self):
        return (self.voxel_z_spin.value(), self.voxel_y_spin.value(),
                self.voxel_x_spin.value())


def _persist_from(prm, name):
    from napari_epicycle.data_preparation._export import persist_patch
    return persist_patch(
        prm["workspace"], name, np.zeros((1, 2, 8, 8), np.uint8),
        channel_names=["c0"], original_shape=(1, 2, 8, 8),
        cropped_shape=(1, 2, 8, 8), selected_z_range=[0, 2],
        bounding_box=[[0, 0], [0, 8], [8, 8], [8, 0]],
        source_path=prm["source_path"],
        source_display_name=prm["source_display_name"],
        source_axes=prm["source_axes"], voxel_size=prm["voxel_size"],
        voxel_size_source=prm["voxel_size_source"],
        voxel_z_from_xy=prm.get("voxel_z_from_xy"))


def _bump_mtime(path, by=10.0):
    t = os.path.getmtime(path) + by
    os.utime(path, (t, t))


def _xy_only_tif(tmp_path, name):
    """A TIFF whose header records only an XY pixel size (1000 px/cm = 10 um)."""
    p = str(tmp_path / name)
    tifffile.imwrite(p, np.zeros((4, 16, 16), np.uint16),
                     resolution=(1000, 1000), resolutionunit="CENTIMETER")
    return p


def _ome_iso_tif(tmp_path, name):
    """An OME-TIFF recording all three sizes, equal (a genuine isotropic Z)."""
    p = str(tmp_path / name)
    tifffile.imwrite(p, np.zeros((4, 16, 16), np.uint16), ome=True,
                     metadata={"axes": "ZYX", "PhysicalSizeX": 0.5,
                               "PhysicalSizeY": 0.5, "PhysicalSizeZ": 0.5})
    return p


class TestGatherPersistVoxelType:
    def _gather(self, tmp_path, vtype, spins=(2.0, 0.5, 0.5)):
        w = _TypedWidget(str(tmp_path), vtype)
        for s, v in zip((w.voxel_z_spin, w.voxel_y_spin, w.voxel_x_spin), spins):
            s.setValue(v)
        ctl = _ctl(widget=w)
        ctl._loaded_source_file = str(tmp_path / "s.tif")
        return ctl._gather_persist_params()

    def test_detected_spins_travel_as_detected(self, tmp_path):
        p = self._gather(tmp_path, "detected")
        assert p["voxel_size"] == [2.0, 0.5, 0.5]
        assert p["voxel_size_source"] == "detected"

    def test_edited_spins_travel_as_user(self, tmp_path):
        assert self._gather(tmp_path, "user")["voxel_size_source"] == "user"

    def test_unit_spins_travel_no_type(self, tmp_path):
        p = self._gather(tmp_path, "detected", spins=(1.0, 1.0, 1.0))
        assert p["voxel_size"] is None and p["voxel_size_source"] is None

    def test_widget_without_a_type_gives_none(self, tmp_path):
        """An older / fake widget: the value-based typing stays the fallback."""
        w = _RecordWidget(str(tmp_path))
        w.voxel_z_spin.setValue(2.0)
        ctl = _ctl(widget=w)
        ctl._loaded_source_file = str(tmp_path / "s.tif")
        assert ctl._gather_persist_params()["voxel_size_source"] is None

    def test_alias_row_then_extract_keeps_the_detected_spacing(self, tmp_path):
        """End to end through the controller's gather: a row an alias edit
        created (no voxel) takes the header spacing shown at extract."""
        from napari_epicycle.data_preparation._export import persist_patch
        ws = str(tmp_path / "ws"); os.makedirs(ws)
        src = _tif(tmp_path, "hd.tif", shape=(5, 32, 32),
                   imagej=True, spacing=0.5, res=1.0 / 0.1)
        w = _TypedWidget(ws, "detected")
        ctl = _ctl(widget=w)
        ctl._selected_source_path = src
        w.display_name_edit.setText("kid")
        ctl._queue_source_save(path=src, display_name="kid")   # the alias edit
        ctl._flush_source_state_save()
        assert ctl._find_source_row(load_state(ws), src).get("voxel_size") is None
        for s, v in zip((w.voxel_z_spin, w.voxel_y_spin, w.voxel_x_spin),
                        (0.5, 0.1, 0.1)):
            s.setValue(v)
        ctl._loaded_source_file = src
        prm = ctl._gather_persist_params()
        persist_patch(
            prm["workspace"], "kid_patch_1", np.zeros((1, 2, 8, 8), np.uint8),
            channel_names=["c0"], original_shape=(1, 2, 8, 8),
            cropped_shape=(1, 2, 8, 8), selected_z_range=[0, 2],
            bounding_box=[[0, 0], [0, 8], [8, 8], [8, 0]],
            source_path=prm["source_path"],
            source_display_name=prm["source_display_name"],
            source_axes=prm["source_axes"], voxel_size=prm["voxel_size"],
            voxel_size_source=prm["voxel_size_source"])
        row = ctl._find_source_row(load_state(ws), src)
        assert row["voxel_size"] == [0.5, 0.1, 0.1]
        assert row["voxel_size_source"] == "detected"


class TestDetectedRowResolve:
    def test_detected_row_shows_its_recorded_value_header_free(
            self, tmp_path, monkeypatch):
        """What is RECORDED is what the panel shows: a "detected" row is
        resolved to its recorded value (what Steps 3 and 4 read), with no
        header opened by the resolver -- even when the file now says
        otherwise."""
        import napari_epicycle.data_preparation.controller as ctl_mod
        calls = []
        monkeypatch.setattr(ctl_mod, "read_voxel_spacing_detail",
                            lambda p: calls.append(p))
        ctl = _ctl()
        src = _tif(tmp_path, "hd.tif", shape=(5, 32, 32),
                   imagej=True, spacing=0.25, res=1.0 / 0.05)
        row = {"voxel_size": [9.0, 9.0, 9.0], "voxel_size_source": "detected"}
        eff, eff_src, base, base_src = ctl._resolve_voxel_for_source(src, row)
        assert eff == (9.0, 9.0, 9.0) and eff_src == "detected"
        assert base == (9.0, 9.0, 9.0) and base_src == "detected"
        assert calls == []

    def test_detected_row_kept_when_the_file_is_gone(self, tmp_path):
        """The header cannot be read (file moved): the value recorded from it
        is shown, still as "detected" -- not the 1/1/1 unknown."""
        ctl = _ctl()
        src = str(tmp_path / "gone.tif")
        row = {"voxel_size": [0.5, 0.1, 0.1], "voxel_size_source": "detected"}
        eff, eff_src, base, base_src = ctl._resolve_voxel_for_source(src, row)
        assert eff == (0.5, 0.1, 0.1) and eff_src == "detected"
        assert base == (0.5, 0.1, 0.1) and base_src == "detected"

    def test_unknown_row_stays_unknown_when_the_file_is_gone(self, tmp_path):
        ctl = _ctl()
        src = str(tmp_path / "gone.tif")
        row = {"voxel_size": None, "voxel_size_source": "unknown"}
        eff, eff_src, _, _ = ctl._resolve_voxel_for_source(src, row)
        assert eff == (1.0, 1.0, 1.0) and eff_src == "unknown"


class TestZAssumedFromXY:
    def _restore(self, tmp_path, src, row=None):
        w = _TypedWidget(str(tmp_path))
        ctl = _ctl(widget=w)
        ctl._restore_source_widgets_for(src, row)
        return ctl, w

    def test_xy_only_header_flags_z_assumed(self, tmp_path):
        src = _xy_only_tif(tmp_path, "xy.tif")
        _, w = self._restore(tmp_path, src)
        assert w.applied_voxel[:3] == pytest.approx((10.0, 10.0, 10.0))
        assert w.applied_voxel[3] == "detected"
        assert w.z_assumed is True

    def test_header_with_a_z_spacing_is_not_flagged(self, tmp_path):
        src = _tif(tmp_path, "hd.tif", shape=(5, 32, 32),
                   imagej=True, spacing=0.5, res=1.0 / 0.1)
        _, w = self._restore(tmp_path, src)
        assert w.z_assumed is False

    def test_genuine_isotropic_header_is_not_flagged(self, tmp_path):
        src = _ome_iso_tif(tmp_path, "iso.ome.tif")
        _, w = self._restore(tmp_path, src)
        assert w.applied_voxel[:3] == pytest.approx((0.5, 0.5, 0.5))
        assert w.z_assumed is False

    def test_user_row_is_not_flagged(self, tmp_path):
        src = _xy_only_tif(tmp_path, "xy2.tif")
        _, w = self._restore(tmp_path, src, {"voxel_size": [3.0, 1.0, 1.0],
                                             "voxel_size_source": "user"})
        assert w.applied_voxel == (3.0, 1.0, 1.0, "user")
        assert w.z_assumed is False

    def test_detail_reader_keeps_read_voxel_spacing_unchanged(self, tmp_path):
        from packages.epicycle_largedata.axes import read_voxel_spacing_detail
        src = _xy_only_tif(tmp_path, "xy3.tif")
        d = read_voxel_spacing_detail(src)
        assert d["zyx"] == read_voxel_spacing(src)
        assert d["z_from_xy"] is True
        iso = _ome_iso_tif(tmp_path, "iso2.ome.tif")
        assert read_voxel_spacing_detail(iso)["z_from_xy"] is False
        assert read_voxel_spacing_detail(str(tmp_path / "nope.tif")) is None


class TestSpinsDescribeTheLoadedSource:
    """The spins follow the SELECTED source; an extract cuts the LOADED one.
    A passive select of another file must never hand its spacing to the
    loaded source's row."""

    def _two(self, tmp_path):
        ws = str(tmp_path / "ws"); os.makedirs(ws)
        a = _tif(tmp_path, "a.tif", shape=(5, 32, 32),
                 imagej=True, spacing=2.0, res=1.0 / 0.5)
        b = _tif(tmp_path, "b.tif", shape=(5, 32, 32),
                 imagej=True, spacing=0.3, res=1.0 / 0.1)
        w = _SpinWidget(ws)
        return ws, a, b, w, _ctl(widget=w)

    def test_alias_row_of_the_loaded_source_gets_its_own_spacing(self, tmp_path):
        ws, a, b, w, ctl = self._two(tmp_path)
        ctl._select_source(a)
        ctl._queue_source_save(path=a, display_name="A")   # alias edit: no voxel
        ctl._flush_source_state_save()
        ctl._loaded_source_file = a                          # A loaded
        ctl._select_source(b)                                # B only selected
        assert w.spins() == pytest.approx((0.3, 0.1, 0.1))   # the panel shows B
        prm = ctl._gather_persist_params()
        assert prm["source_path"] == a
        assert prm["voxel_size"] == pytest.approx([2.0, 0.5, 0.5])
        assert prm["voxel_size_source"] == "detected"
        r = _persist_from(prm, "A_patch_1")
        st = load_state(ws)
        row = ctl._find_source_row(st, a)
        assert row["voxel_size"] == pytest.approx([2.0, 0.5, 0.5])
        assert row["voxel_size_source"] == "detected"
        from napari_epicycle.data_preparation._voxel_spacing import (
            resolve_stack_spacing)
        got = resolve_stack_spacing(None, st, os.path.join(ws, r["relative_path"]),
                                    ws, read_file=False)
        assert got["zyx"] == pytest.approx([2.0, 0.5, 0.5])
        ctl._select_source(a)                                # back on A
        assert w.applied_voxel[:3] == pytest.approx((2.0, 0.5, 0.5))

    def test_create_branch_takes_the_loaded_sources_spacing(self, tmp_path):
        ws, a, b, w, ctl = self._two(tmp_path)
        ctl._select_source(a)
        ctl._loaded_source_file = a
        ctl._select_source(b)
        _persist_from(ctl._gather_persist_params(), "x_patch_1")
        row = ctl._find_source_row(load_state(ws), a)
        assert row["voxel_size"] == pytest.approx([2.0, 0.5, 0.5])
        assert row["voxel_size_source"] == "detected"
        assert ctl._find_source_row(load_state(ws), b) is None

    def test_loaded_user_row_is_read_from_the_registry_header_free(
            self, tmp_path, monkeypatch):
        import napari_epicycle.data_preparation.controller as ctl_mod
        ws, a, b, w, ctl = self._two(tmp_path)
        st = empty_state()
        upsert_source(st, path=a, voxel_size=[3.0, 0.7, 0.7],
                      voxel_size_source="user")
        save_state(ws, st)
        ctl._select_source(a)
        ctl._loaded_source_file = a
        ctl._select_source(b)
        calls = []
        monkeypatch.setattr(ctl_mod, "read_voxel_spacing_detail",
                            lambda p: calls.append(p))
        prm = ctl._gather_persist_params()
        assert prm["voxel_size"] == [3.0, 0.7, 0.7]
        assert prm["voxel_size_source"] == "user"
        assert calls == []

    def test_spins_of_the_loaded_source_are_used_as_shown(self, tmp_path):
        ws, a, b, w, ctl = self._two(tmp_path)
        ctl._select_source(a)
        ctl._loaded_source_file = a
        prm = ctl._gather_persist_params()
        assert prm["voxel_size"] == pytest.approx([2.0, 0.5, 0.5])
        assert prm["voxel_size_source"] == "detected"

    def test_store_load_records_which_file_the_spins_describe(self, tmp_path):
        ws, a, b, w, ctl = self._two(tmp_path)
        ctl._select_source(b)
        ctl._apply_store_voxel_to_widget({"scales": [[2.0, 0.5, 0.5]]}, a)
        assert ctl._voxel_spins_source == a


class TestRecordedValueIsWhatThePanelShows:
    def _ctl_on(self, tmp_path, a):
        ws = str(tmp_path / "ws"); os.makedirs(ws, exist_ok=True)
        w = _SpinWidget(ws)
        return ws, w, _ctl(widget=w)

    def test_header_changed_after_recording(self, tmp_path):
        a = _tif(tmp_path, "a.tif", shape=(5, 32, 32),
                 imagej=True, spacing=2.0, res=1.0 / 0.5)
        ws, w, ctl = self._ctl_on(tmp_path, a)
        ctl._select_source(a)
        ctl._loaded_source_file = a
        r = _persist_from(ctl._gather_persist_params(), "a_patch_1")
        # the file's header is fixed later
        tifffile.imwrite(a, np.zeros((5, 32, 32), np.uint16), imagej=True,
                         resolution=(1 / 0.2, 1 / 0.2),
                         metadata={"spacing": 1.0, "unit": "um"})
        _bump_mtime(a)
        ctl._selected_source_path = None
        ctl._select_source(a)
        st = load_state(ws)
        # the panel shows the recorded value (what Steps 3 and 4 read) ...
        assert w.applied_voxel == (2.0, 0.5, 0.5, "detected")
        assert ctl._find_source_row(st, a)["voxel_size"] == pytest.approx(
            [2.0, 0.5, 0.5])
        # ... and says the file now says something else
        assert w.note == pytest.approx((1.0, 0.2, 0.2))
        # the reset follows the file, recorded as detected
        ctl.on_voxel_reset()
        ctl._flush_source_state_save()
        row = ctl._find_source_row(load_state(ws), a)
        assert row["voxel_size"] == pytest.approx([1.0, 0.2, 0.2])
        assert row["voxel_size_source"] == "detected"
        assert w.applied_voxel[:3] == pytest.approx((1.0, 0.2, 0.2))
        assert w._detected_voxel_size == pytest.approx((1.0, 0.2, 0.2))
        assert w.note is None

    def test_header_unchanged_has_no_note(self, tmp_path):
        a = _tif(tmp_path, "a.tif", shape=(5, 32, 32),
                 imagej=True, spacing=2.0, res=1.0 / 0.5)
        ws, w, ctl = self._ctl_on(tmp_path, a)
        st = empty_state()
        upsert_source(st, path=a, voxel_size=[2.0, 0.5, 0.5],
                      voxel_size_source="detected")
        save_state(ws, st)
        ctl._select_source(a)
        assert w.applied_voxel == (2.0, 0.5, 0.5, "detected")
        assert w.note is None

    def test_file_gone_is_labelled_recorded(self, tmp_path):
        a = str(tmp_path / "gone.tif")
        ws, w, ctl = self._ctl_on(tmp_path, a)
        st = empty_state()
        upsert_source(st, path=a, voxel_size=[2.0, 0.5, 0.5],
                      voxel_size_source="detected")
        save_state(ws, st)
        ctl._select_source(a)
        assert w.applied_voxel == (2.0, 0.5, 0.5, "detected")
        assert w.note == "recorded"
        # the reset cannot read the file: the recorded value stays
        ctl.on_voxel_reset()
        ctl._flush_source_state_save()
        row = ctl._find_source_row(load_state(ws), a)
        assert row["voxel_size"] == [2.0, 0.5, 0.5]
        assert row["voxel_size_source"] == "detected"

    def test_user_row_opens_zero_headers_on_select_once_checked(
            self, tmp_path, monkeypatch):
        """A typed value's header is read ONCE (the one-time check of a row
        that may hold a header value typed as the user's); once the row is
        marked checked, every later select opens zero headers."""
        import napari_epicycle.data_preparation.controller as ctl_mod
        from packages.epicycle_largedata import axes as axes_mod
        a = _tif(tmp_path, "a.tif", shape=(5, 32, 32),
                 imagej=True, spacing=2.0, res=1.0 / 0.5)
        ws, w, ctl = self._ctl_on(tmp_path, a)
        st = empty_state()
        upsert_source(st, path=a, voxel_size=[3.0, 0.7, 0.7],
                      voxel_size_source="user")
        save_state(ws, st)
        real = axes_mod.read_voxel_spacing_detail
        calls = []
        monkeypatch.setattr(ctl_mod, "read_voxel_spacing_detail",
                            lambda p: (calls.append(p) or real(p)))
        ctl._select_source(a)
        assert w.applied_voxel == (3.0, 0.7, 0.7, "user")
        assert len(calls) == 1                         # the one-time check
        ctl._flush_source_state_save()
        row = ctl._find_source_row(load_state(ws), a)
        assert row["voxel_size_source"] == "user"      # header differs
        assert row["voxel_header_checked"] is True
        calls.clear()
        ctl._voxel_detect_cache.clear()                # no memo to hide behind
        ctl._selected_source_path = None
        ctl._select_source(a)
        assert w.applied_voxel == (3.0, 0.7, 0.7, "user")
        assert calls == []


class TestZAssumedCarriedAndOneOpen:
    def test_store_load_keeps_z_assumed(self, tmp_path):
        ws = str(tmp_path / "ws"); os.makedirs(ws)
        src = _xy_only_tif(tmp_path, "xy.tif")
        w = _SpinWidget(ws)
        ctl = _ctl(widget=w)
        ctl._select_source(src)
        assert w.z_assumed is True
        ctl._apply_store_voxel_to_widget({"scales": [[10.0, 10.0, 10.0]]}, src)
        assert w.applied_voxel[3] == "detected"
        assert w.z_assumed is True

    def test_xy_only_select_opens_the_file_once(self, tmp_path, monkeypatch):
        import napari_epicycle.data_preparation.controller as ctl_mod
        src = _xy_only_tif(tmp_path, "xy.tif")
        from packages.epicycle_largedata import axes as axes_mod
        real = axes_mod.read_voxel_spacing_detail
        real_plain = axes_mod.read_voxel_spacing
        calls = []
        # every header read the controller can make, through either reader
        monkeypatch.setattr(ctl_mod, "read_voxel_spacing_detail",
                            lambda p: (calls.append(p) or real(p)))
        monkeypatch.setattr(ctl_mod, "read_voxel_spacing",
                            lambda p: (calls.append(p) or real_plain(p)),
                            raising=False)
        w = _SpinWidget(str(tmp_path))
        ctl = _ctl(widget=w)
        ctl._restore_source_widgets_for(src, None)
        assert w.z_assumed is True
        assert len(calls) == 1


class _RoundSpin(_FakeSpin):
    """A spin that keeps 3 decimals, like the panel's QDoubleSpinBox."""

    def setValue(self, v):
        self._v = round(float(v), 3)

    def decimals(self):
        return 3


class _RoundSpinWidget(_SpinWidget):
    def __init__(self, ws=""):
        super().__init__(ws)
        self.voxel_z_spin = _RoundSpin(1.0)
        self.voxel_y_spin = _RoundSpin(1.0)
        self.voxel_x_spin = _RoundSpin(1.0)


class TestRecordedAtSpinPrecision:
    """An extract records the spins' 3-decimal value; the file's header keeps
    its full precision.  The two agree at the precision the panel shows, so an
    untouched file never reads as changed."""

    def _round_trip(self, tmp_path, src):
        ws = str(tmp_path / "ws"); os.makedirs(ws)
        w = _RoundSpinWidget(ws)
        ctl = _ctl(widget=w)
        ctl._select_source(src)
        before = (w.z_assumed, w.note, w.spins())
        ctl._loaded_source_file = src
        _persist_from(ctl._gather_persist_params(), "p_patch_1")
        ctl._selected_source_path = None
        ctl._select_source(src)                       # file untouched
        return ws, w, ctl, before

    def test_a_non_3_decimal_header_gives_no_false_note(self, tmp_path):
        px = 0.20756646
        src = _tif(tmp_path, "a.tif", shape=(5, 32, 32),
                   imagej=True, spacing=1.0, res=1.0 / px)
        ws, w, ctl, _ = self._round_trip(tmp_path, src)
        row = ctl._find_source_row(load_state(ws), src)
        assert row["voxel_size"] == pytest.approx([1.0, 0.208, 0.208])
        assert w.applied_voxel == (1.0, 0.208, 0.208, "detected")
        assert w.note is None

    def test_xy_only_header_keeps_z_assumed_after_extract(self, tmp_path):
        p = str(tmp_path / "xy.tif")
        # 48177 px/cm -> a 0.20757 um pixel, no Z recorded
        tifffile.imwrite(p, np.zeros((4, 16, 16), np.uint16),
                         resolution=(48177, 48177), resolutionunit="CENTIMETER")
        ws, w, ctl, before = self._round_trip(tmp_path, p)
        assert before[0] is True and before[1] is None
        assert w.note is None
        assert w.z_assumed is True

    def test_a_real_change_still_gives_the_note(self, tmp_path):
        src = _tif(tmp_path, "c.tif", shape=(5, 32, 32),
                   imagej=True, spacing=1.0, res=1.0 / 0.20756646)
        ws, w, ctl, _ = self._round_trip(tmp_path, src)
        tifffile.imwrite(src, np.zeros((5, 32, 32), np.uint16), imagej=True,
                         resolution=(1 / 0.2101, 1 / 0.2101),
                         metadata={"spacing": 1.0, "unit": "um"})
        _bump_mtime(src)
        ctl._selected_source_path = None
        ctl._select_source(src)
        assert w.note == pytest.approx((1.0, 0.2101, 0.2101))


class TestAliasDescribesTheLoadedSource:
    def test_alias_of_another_selected_file_is_not_sent(self, tmp_path):
        ws = str(tmp_path / "ws"); os.makedirs(ws)
        a = _tif(tmp_path, "a.tif", shape=(5, 32, 32),
                 imagej=True, spacing=2.0, res=1.0 / 0.5)
        b = _tif(tmp_path, "b.tif", shape=(5, 32, 32),
                 imagej=True, spacing=0.3, res=1.0 / 0.1)
        w = _SpinWidget(ws)
        ctl = _ctl(widget=w)
        ctl._select_source(a)
        ctl._loaded_source_file = a
        ctl._select_source(b)
        w.display_name_edit.setText("B alias")
        prm = ctl._gather_persist_params()
        assert prm["source_display_name"] is None
        _persist_from(prm, "a_patch_1")
        row = ctl._find_source_row(load_state(ws), a)
        assert row.get("display_name") != "B alias"
        # on the loaded file, the alias is sent as typed
        ctl._select_source(a)
        w.display_name_edit.setText("A alias")
        assert ctl._gather_persist_params()["source_display_name"] == "A alias"


# ===========================================================================
# The header size is RECORDED at select on a row the user worked with
# ===========================================================================

class _SelWidget(_SpinWidget):
    """`_SpinWidget` plus the over-ceiling note."""

    def __init__(self, ws=""):
        super().__init__(ws)
        self.over_ceiling = None

    def set_detected_voxel_size(self, z, y, x):
        super().set_detected_voxel_size(z, y, x)
        self.over_ceiling = False

    def set_voxel_over_ceiling(self, over):
        self.over_ceiling = bool(over)


def _ws(tmp_path):
    ws = str(tmp_path / "ws")
    os.makedirs(ws, exist_ok=True)
    return ws


def _unset_rows(ws, *paths):
    st = empty_state()
    for p in paths:
        upsert_source(st, path=p, display_name=os.path.basename(p),
                      voxel_size_source="unknown")
    save_state(ws, st)


def _sel_ctl(ws):
    w = _SelWidget(ws)
    return _ctl(widget=w), w


def _state_text(ws):
    with open(step1_state_path(ws), encoding="utf-8") as f:
        return f.read()


def _mtime_ns(ws):
    return os.stat(step1_state_path(ws)).st_mtime_ns


def _zyx_tif(tmp_path, name, z=2.0, xy=0.5):
    return _tif(tmp_path, name, shape=(5, 32, 32), imagej=True,
                spacing=z, res=1.0 / xy)


class TestRecordedAtSelect:
    def test_an_unset_row_records_the_header_size_at_select(self, tmp_path):
        ws = _ws(tmp_path)
        a = _zyx_tif(tmp_path, "a.tif")
        _unset_rows(ws, a)
        ctl, w = _sel_ctl(ws)
        ctl._select_source(a)
        ctl._flush_source_state_save()
        row = ctl._find_source_row(load_state(ws), a)
        assert row["voxel_size"] == pytest.approx([2.0, 0.5, 0.5])
        assert row["voxel_size_source"] == "detected"
        assert row["voxel_z_from_xy"] is False

    def test_a_typed_value_is_never_overwritten(self, tmp_path):
        ws = _ws(tmp_path)
        a = _zyx_tif(tmp_path, "a.tif")
        st = empty_state()
        upsert_source(st, path=a, voxel_size=[3.0, 0.7, 0.7],
                      voxel_size_source="user", voxel_header_checked=True)
        save_state(ws, st)
        ctl, w = _sel_ctl(ws)
        ctl._select_source(a)                       # select
        ctl._flush_source_state_save()
        ctl._loaded_source_file = a                 # Load follows the select
        _persist_from(ctl._gather_persist_params(), "a_patch_1")   # extract
        ctl._selected_source_path = None
        ctl._select_source(a)                       # a later select
        ctl._flush_source_state_save()
        row = ctl._find_source_row(load_state(ws), a)
        assert row["voxel_size"] == [3.0, 0.7, 0.7]
        assert row["voxel_size_source"] == "user"

    def test_a_typed_edit_right_after_the_fill_wins_in_the_same_flush(
            self, tmp_path):
        ws = _ws(tmp_path)
        a = _zyx_tif(tmp_path, "a.tif")
        _unset_rows(ws, a)
        ctl, w = _sel_ctl(ws)
        ctl._select_source(a)                       # the fill is queued
        for s, v in zip((w.voxel_z_spin, w.voxel_y_spin, w.voxel_x_spin),
                        (4.0, 0.5, 0.5)):
            s.setValue(v)
        ctl.on_voxel_edited()                       # queued after it
        assert len(ctl._pending_source_edits) == 2
        ctl._flush_source_state_save()
        row = ctl._find_source_row(load_state(ws), a)
        assert row["voxel_size"] == [4.0, 0.5, 0.5]
        assert row["voxel_size_source"] == "user"
        assert row["voxel_z_from_xy"] is False

    def test_an_xy_only_header_is_recorded_with_z_assumed(self, tmp_path):
        from napari_epicycle.data_preparation._state import (
            z_assumed_patch_paths,
        )
        from napari_epicycle.data_preparation._voxel_spacing import (
            resolve_stack_spacing,
        )
        ws = _ws(tmp_path)
        a = _xy_only_tif(tmp_path, "xy.tif")
        _unset_rows(ws, a)
        ctl, w = _sel_ctl(ws)
        ctl._select_source(a)
        ctl._flush_source_state_save()
        row = ctl._find_source_row(load_state(ws), a)
        assert row["voxel_size"] == pytest.approx([10.0, 10.0, 10.0])
        assert row["voxel_z_from_xy"] is True
        ctl._loaded_source_file = a
        r = _persist_from(ctl._gather_persist_params(), "xy_patch_1")
        st = load_state(ws)
        rel = r["relative_path"]
        assert _norm_path(rel) in z_assumed_patch_paths(st)
        got = resolve_stack_spacing(None, st, os.path.join(ws, rel), ws,
                                    read_file=False)
        assert got == {"zyx": pytest.approx([10.0, 10.0, 10.0]),
                       "source": "step1"}
        assert set(got) == {"zyx", "source"}

    def test_select_is_not_loaded_each_row_gets_its_own_header(self, tmp_path):
        ws = _ws(tmp_path)
        a = _zyx_tif(tmp_path, "a.tif", z=2.0, xy=0.5)
        b = _zyx_tif(tmp_path, "b.tif", z=0.3, xy=0.1)
        _unset_rows(ws, a)
        ctl, w = _sel_ctl(ws)
        ctl._loaded_source_file = a                 # A is loaded ...
        _unset_rows(ws, a, b)
        ctl._select_source(b)                       # ... B is selected
        ctl._flush_source_state_save()
        st = load_state(ws)
        assert ctl._find_source_row(st, b)["voxel_size"] == pytest.approx(
            [0.3, 0.1, 0.1])
        assert ctl._find_source_row(st, a).get("voxel_size") is None

    @pytest.mark.parametrize("kind", ["plain", "unit"])
    def test_a_header_with_no_size_records_nothing(self, tmp_path, kind):
        ws = _ws(tmp_path)
        if kind == "plain":
            a = _tif(tmp_path, "a.tif", shape=(5, 32, 32))
        else:
            a = str(tmp_path / "a.tif")
            tifffile.imwrite(a, np.zeros((5, 32, 32), np.uint16), imagej=True,
                             resolution=(1, 1),
                             metadata={"spacing": 1.0, "unit": "um"})
        _unset_rows(ws, a)
        before = _mtime_ns(ws)
        ctl, w = _sel_ctl(ws)
        ctl._select_source(a)
        assert ctl._pending_source_edits == []
        ctl._flush_source_state_save()
        assert _mtime_ns(ws) == before
        row = ctl._find_source_row(load_state(ws), a)
        assert row["voxel_size_source"] == "unknown"

    def test_a_passive_select_of_a_rowless_source_writes_nothing(
            self, tmp_path):
        ws = _ws(tmp_path)
        a = _zyx_tif(tmp_path, "a.tif")
        save_state(ws, empty_state())
        before = _mtime_ns(ws)
        ctl, w = _sel_ctl(ws)
        ctl._select_source(a)
        ctl._flush_source_state_save()
        assert load_state(ws)["sources"] == []
        assert _mtime_ns(ws) == before

    def test_a_detected_row_without_the_flag_is_back_filled(self, tmp_path):
        ws = _ws(tmp_path)
        a = _xy_only_tif(tmp_path, "xy.tif")
        st = empty_state()
        upsert_source(st, path=a, voxel_size=[10.0, 10.0, 10.0],
                      voxel_size_source="detected")
        save_state(ws, st)
        ctl, w = _sel_ctl(ws)
        ctl._select_source(a)
        ctl._flush_source_state_save()
        row = ctl._find_source_row(load_state(ws), a)
        assert row["voxel_z_from_xy"] is True
        assert row["voxel_size"] == [10.0, 10.0, 10.0]
        assert row["voxel_size_source"] == "detected"

    def test_no_back_fill_when_the_value_changed_meanwhile(self, tmp_path):
        ws = _ws(tmp_path)
        a = _xy_only_tif(tmp_path, "xy.tif")
        st = empty_state()
        upsert_source(st, path=a, voxel_size=[10.0, 10.0, 10.0],
                      voxel_size_source="detected")
        save_state(ws, st)
        ctl, w = _sel_ctl(ws)
        ctl._select_source(a)                        # back-fill queued
        # another writer (a Prepare) records a different value before the flush
        st = load_state(ws)
        ctl._find_source_row(st, a)["voxel_size"] = [3.0, 10.0, 10.0]
        save_state(ws, st)
        ctl._flush_source_state_save()
        row = ctl._find_source_row(load_state(ws), a)
        assert "voxel_z_from_xy" not in row
        assert row["voxel_size"] == [3.0, 10.0, 10.0]

    def test_no_back_fill_from_a_header_that_differs(self, tmp_path):
        ws = _ws(tmp_path)
        a = _xy_only_tif(tmp_path, "xy.tif")
        st = empty_state()
        upsert_source(st, path=a, voxel_size=[2.0, 10.0, 10.0],
                      voxel_size_source="detected")
        save_state(ws, st)
        ctl, w = _sel_ctl(ws)
        ctl._select_source(a)
        assert ctl._pending_source_edits == []

    def test_the_guard_reads_the_flush_working_state(self, tmp_path):
        """A value set by an EARLIER queued edit (not yet on disk) blocks the
        marked fill: the guard sees the row as the flush builds it."""
        ws = _ws(tmp_path)
        a = _zyx_tif(tmp_path, "a.tif")
        _unset_rows(ws, a)
        ctl, w = _sel_ctl(ws)
        ctl._pending_source_ws = ws
        ctl._pending_source_edits.append(
            dict(path=a, voxel_size=[5.0, 0.5, 0.5], voxel_size_source="user"))
        ctl._queue_source_save(path=a, voxel_size=[2.0, 0.5, 0.5],
                               voxel_size_source="detected",
                               voxel_z_from_xy=False, _fill_unset=True)
        ctl._flush_source_state_save()
        row = ctl._find_source_row(load_state(ws), a)
        assert row["voxel_size"] == [5.0, 0.5, 0.5]
        assert row["voxel_size_source"] == "user"

    def test_a_failed_save_keeps_the_marked_edit_guarded(
            self, tmp_path, monkeypatch):
        import napari_epicycle.data_preparation.controller as ctl_mod
        ws = _ws(tmp_path)
        a = _zyx_tif(tmp_path, "a.tif")
        _unset_rows(ws, a)
        ctl, w = _sel_ctl(ws)
        ctl._report_state_problem = lambda *a, **k: None
        ctl._clear_state_problem = lambda *a, **k: None
        ctl._log = lambda *a, **k: None
        ctl._select_source(a)
        real_save = ctl_mod.save_state

        def boom(*_a, **_k):
            raise OSError("disk full")
        monkeypatch.setattr(ctl_mod, "save_state", boom)
        ctl._flush_source_state_save()
        assert len(ctl._pending_source_edits) == 1
        assert ctl._pending_source_edits[0].get("_fill_unset") is True
        # meanwhile the user types a value straight into the file's row
        st = load_state(ws)
        ctl._find_source_row(st, a).update(voxel_size=[7.0, 0.5, 0.5],
                                           voxel_size_source="user")
        real_save(ws, st)
        monkeypatch.setattr(ctl_mod, "save_state", real_save)
        ctl._flush_source_state_save()                # the retry
        assert ctl._pending_source_edits == []
        row = ctl._find_source_row(load_state(ws), a)
        assert row["voxel_size"] == [7.0, 0.5, 0.5]   # guarded again
        assert "_fill_unset" not in _state_text(ws)

    def test_a_quick_a_then_b_select_keeps_each_header_on_its_row(
            self, tmp_path):
        ws = _ws(tmp_path)
        a = _zyx_tif(tmp_path, "a.tif", z=2.0, xy=0.5)
        b = _zyx_tif(tmp_path, "b.tif", z=0.3, xy=0.1)
        _unset_rows(ws, a, b)
        ctl, w = _sel_ctl(ws)
        ctl._select_source(a)
        ctl._select_source(b)                         # within the 400 ms
        ctl._flush_source_state_save()
        st = load_state(ws)
        assert ctl._find_source_row(st, a)["voxel_size"] == pytest.approx(
            [2.0, 0.5, 0.5])
        assert ctl._find_source_row(st, b)["voxel_size"] == pytest.approx(
            [0.3, 0.1, 0.1])

    def test_the_marker_never_reaches_the_file(self, tmp_path):
        ws = _ws(tmp_path)
        a = _zyx_tif(tmp_path, "a.tif")
        b = _xy_only_tif(tmp_path, "b.tif")
        c = _zyx_tif(tmp_path, "c.tif", z=1.0, xy=0.2)
        st = empty_state()
        upsert_source(st, path=a, voxel_size_source="unknown")
        upsert_source(st, path=b, voxel_size=[10.0, 10.0, 10.0],
                      voxel_size_source="detected")
        upsert_source(st, path=c, voxel_size=[1.0, 0.2, 0.2],
                      voxel_size_source="user")
        save_state(ws, st)
        ctl, w = _sel_ctl(ws)
        for p in (a, b, c):
            ctl._select_source(p)
        ctl._flush_source_state_save()
        text = _state_text(ws)
        assert "_fill_unset" not in text
        for key in ("recorded_zyx", "user_zyx", "header_zyx", "decimals"):
            assert key not in text

    def test_an_xy_over_the_ceiling_is_shown_not_recorded(self, tmp_path):
        """72 dpi read as centimetres: a 352.8 um pixel."""
        ws = _ws(tmp_path)
        a = str(tmp_path / "dpi.tif")
        tifffile.imwrite(a, np.zeros((4, 16, 16), np.uint16),
                         resolution=(72 / 2.54, 72 / 2.54),
                         resolutionunit="CENTIMETER")
        _unset_rows(ws, a)
        ctl, w = _sel_ctl(ws)
        ctl._log = lambda *a, **k: None
        ctl._select_source(a)
        # the boxes do not take a print DPI: they keep the unknown default
        assert w.applied_voxel == (1.0, 1.0, 1.0, "unknown")
        assert w.over_ceiling is True
        assert ctl._pending_source_edits == []
        ctl._flush_source_state_save()
        assert ctl._find_source_row(load_state(ws), a)[
            "voxel_size_source"] == "unknown"

    def test_a_header_out_of_the_spin_range_is_not_recorded(self, tmp_path):
        ws = _ws(tmp_path)
        a = str(tmp_path / "far.tif")
        tifffile.imwrite(a, np.zeros((4, 16, 16), np.uint16),
                         resolution=(0.5, 0.5), resolutionunit="CENTIMETER")
        _unset_rows(ws, a)                            # 20000 um pixels
        ctl, w = _sel_ctl(ws)
        ctl._log = lambda *a, **k: None
        ctl._select_source(a)
        assert ctl._pending_source_edits == []


class TestUserRowHeaderCheck:
    def test_a_user_row_equal_to_the_header_becomes_detected(self, tmp_path):
        ws = _ws(tmp_path)
        a = _xy_only_tif(tmp_path, "xy.tif")
        st = empty_state()
        upsert_source(st, path=a, voxel_size=[10.0, 10.0, 10.0],
                      voxel_size_source="user")
        save_state(ws, st)
        ctl, w = _sel_ctl(ws)
        ctl._select_source(a)
        ctl._flush_source_state_save()
        row = ctl._find_source_row(load_state(ws), a)
        assert row["voxel_size_source"] == "detected"
        assert row["voxel_z_from_xy"] is True
        assert row["voxel_header_checked"] is True
        assert row["voxel_size"] == [10.0, 10.0, 10.0]
        assert w.applied_voxel == (10.0, 10.0, 10.0, "detected")
        assert w.z_assumed is True

    def test_reset_on_a_v014_user_row_with_an_xy_only_header(self, tmp_path):
        ws = _ws(tmp_path)
        a = _xy_only_tif(tmp_path, "xy.tif")
        st = empty_state()
        upsert_source(st, path=a, voxel_size=[3.0, 10.0, 10.0],
                      voxel_size_source="user")
        save_state(ws, st)
        ctl, w = _sel_ctl(ws)
        ctl._select_source(a)
        ctl._flush_source_state_save()
        assert ctl._find_source_row(load_state(ws), a)[
            "voxel_size_source"] == "user"            # unequal: stays typed
        w._detected_voxel_size = (1.0, 1.0, 1.0)      # a user row's baseline
        ctl.on_voxel_reset()
        ctl._flush_source_state_save()
        row = ctl._find_source_row(load_state(ws), a)
        assert row["voxel_size"] == pytest.approx([10.0, 10.0, 10.0])
        assert row["voxel_size_source"] == "detected"
        assert row["voxel_z_from_xy"] is True
        assert w.z_assumed is True


class TestFlagWriters:
    def test_reset_with_an_unreadable_header_keeps_the_flag(self, tmp_path):
        ws = _ws(tmp_path)
        a = str(tmp_path / "gone.tif")
        st = empty_state()
        upsert_source(st, path=a, voxel_size=[0.2, 0.2, 0.2],
                      voxel_size_source="detected", voxel_z_from_xy=True)
        save_state(ws, st)
        ctl, w = _sel_ctl(ws)
        ctl._select_source(a)
        ctl.on_voxel_reset()
        ctl._flush_source_state_save()
        row = ctl._find_source_row(load_state(ws), a)
        assert row["voxel_size"] == [0.2, 0.2, 0.2]
        assert row["voxel_z_from_xy"] is True

    def test_reset_with_a_readable_header_writes_the_bool(self, tmp_path):
        ws = _ws(tmp_path)
        a = _zyx_tif(tmp_path, "a.tif", z=2.0, xy=0.5)
        st = empty_state()
        upsert_source(st, path=a, voxel_size=[0.5, 0.5, 0.5],
                      voxel_size_source="detected", voxel_z_from_xy=True)
        save_state(ws, st)
        ctl, w = _sel_ctl(ws)
        ctl._select_source(a)
        ctl.on_voxel_reset()
        ctl._flush_source_state_save()
        row = ctl._find_source_row(load_state(ws), a)
        assert row["voxel_size"] == pytest.approx([2.0, 0.5, 0.5])
        assert row["voxel_z_from_xy"] is False        # a stale flag rewritten

    def _prepare(self, tmp_path, src, row_kw):
        ws = _ws(tmp_path)
        store = _prep_store(ws, "p", src, voxel_size=(0.2, 0.2, 0.2))
        st = empty_state()
        upsert_source(st, path=src, source_id="src_0", **row_kw)
        save_state(ws, st)
        ctl, w = _sel_ctl(ws)
        ctl._log = lambda *a, **k: None
        ctl._largedata_src_path = src
        ctl._largedata_source_id = "src_0"
        ctl._largedata_prepare_params = None
        ctl._persist_prepared_source(store, "p")
        return ctl._find_source_row(load_state(ws), src)

    def test_prepare_with_an_unreadable_header_keeps_the_flag(self, tmp_path):
        src = _tif(tmp_path, "p.tif", shape=(5, 32, 32))   # no size tags
        row = self._prepare(tmp_path, src, dict(
            voxel_size=[0.2, 0.2, 0.2], voxel_size_source="detected",
            voxel_z_from_xy=True))
        assert row["voxel_size_source"] == "detected"
        assert row["voxel_z_from_xy"] is True

    def test_prepare_with_a_readable_header_writes_the_bool(self, tmp_path):
        src = str(tmp_path / "p.tif")
        tifffile.imwrite(src, np.zeros((4, 16, 16), np.uint16),
                         resolution=(50000, 50000), resolutionunit="CENTIMETER")
        row = self._prepare(tmp_path, src, dict(voxel_size_source="unknown"))
        assert row["voxel_size"] == pytest.approx([0.2, 0.2, 0.2])
        assert row["voxel_size_source"] == "detected"
        assert row["voxel_z_from_xy"] is True

    def test_the_orphan_store_synthesis_writes_false(self, tmp_path):
        ws = str(tmp_path)
        src = _tif(tmp_path, "orphan.tif")
        _prep_store(ws, "orphan", src, voxel_size=(0.4, 0.1, 0.1))
        st = empty_state()
        assert _ctl()._reconcile_sources_with_stores(ws, st) is True
        row = st["sources"][0]
        assert row["voxel_size_source"] == "detected"
        assert row["voxel_z_from_xy"] is False

    def test_an_extract_of_a_new_row_writes_the_flag(self, tmp_path):
        ws = _ws(tmp_path)
        a = _xy_only_tif(tmp_path, "xy.tif")
        ctl, w = _sel_ctl(ws)
        ctl._select_source(a)                          # rowless: no fill
        ctl._loaded_source_file = a
        _persist_from(ctl._gather_persist_params(), "xy_patch_1")
        row = ctl._find_source_row(load_state(ws), a)
        assert row["voxel_size_source"] == "detected"
        assert row["voxel_z_from_xy"] is True

    def test_an_extract_of_another_loaded_file_writes_its_flag(self, tmp_path):
        ws = _ws(tmp_path)
        a = _xy_only_tif(tmp_path, "xy.tif")
        b = _zyx_tif(tmp_path, "b.tif")
        ctl, w = _sel_ctl(ws)
        ctl._select_source(a)
        ctl._loaded_source_file = a
        ctl._select_source(b)                          # spins describe B
        prm = ctl._gather_persist_params()
        assert prm["voxel_z_from_xy"] is True          # A's, not B's
        _persist_from(prm, "xy_patch_1")
        assert ctl._find_source_row(load_state(ws), a)["voxel_z_from_xy"] \
            is True

    def test_confirming_z_makes_the_value_the_users(self, tmp_path):
        ws = _ws(tmp_path)
        a = _xy_only_tif(tmp_path, "xy.tif")
        st = empty_state()
        upsert_source(st, path=a, voxel_size=[10.0, 10.0, 10.0],
                      voxel_size_source="detected", voxel_z_from_xy=True)
        save_state(ws, st)
        ctl, w = _sel_ctl(ws)
        ctl._select_source(a)
        ctl.on_voxel_z_confirmed()                     # same number
        ctl._flush_source_state_save()
        row = ctl._find_source_row(load_state(ws), a)
        assert row["voxel_size"] == [10.0, 10.0, 10.0]
        assert row["voxel_size_source"] == "user"
        assert row["voxel_z_from_xy"] is False


class TestRecordedRowFlagShown:
    def test_a_recorded_row_whose_file_moved_shows_its_flag(self, tmp_path):
        ws = _ws(tmp_path)
        a = str(tmp_path / "gone.tif")
        st = empty_state()
        upsert_source(st, path=a, voxel_size=[0.2, 0.2, 0.2],
                      voxel_size_source="detected", voxel_z_from_xy=True)
        save_state(ws, st)
        ctl, w = _sel_ctl(ws)
        ctl._select_source(a)
        assert w.note == "recorded"
        assert w.z_assumed is True

    def test_a_recorded_flag_false_wins_over_the_header(self, tmp_path):
        """The file changed to an XY-only header since: the row's own False
        is what Steps 3 and 4 read, so the panel says the same."""
        ws = _ws(tmp_path)
        a = _xy_only_tif(tmp_path, "xy.tif")
        st = empty_state()
        upsert_source(st, path=a, voxel_size=[10.0, 10.0, 10.0],
                      voxel_size_source="detected", voxel_z_from_xy=False)
        save_state(ws, st)
        ctl, w = _sel_ctl(ws)
        ctl._select_source(a)
        assert w.z_assumed is False


# ===========================================================================
# Review fixes: a typed value sticks; Prepare keeps a recorded flag; the
# ceiling at extract; the log line; the marker handed on; never the spins
# ===========================================================================

class TestReviewFixes:
    def test_a_confirmed_value_equal_to_the_header_sticks(
            self, tmp_path, monkeypatch):
        """Confirm an assumed Z (same number), select again: still the
        user's, flag False, and no header opened for it."""
        import napari_epicycle.data_preparation.controller as ctl_mod
        from packages.epicycle_largedata import axes as axes_mod
        ws = _ws(tmp_path)
        a = _xy_only_tif(tmp_path, "xy.tif")
        _unset_rows(ws, a)
        ctl, w = _sel_ctl(ws)
        ctl._select_source(a)                        # the fill: flagged
        ctl.on_voxel_z_confirmed()
        ctl._flush_source_state_save()
        row = ctl._find_source_row(load_state(ws), a)
        assert row["voxel_size_source"] == "user"
        assert row["voxel_header_checked"] is True
        real = axes_mod.read_voxel_spacing_detail
        calls = []
        monkeypatch.setattr(ctl_mod, "read_voxel_spacing_detail",
                            lambda p: (calls.append(p) or real(p)))
        ctl._voxel_detect_cache.clear()
        ctl._selected_source_path = None
        ctl._select_source(a)
        ctl._flush_source_state_save()
        row = ctl._find_source_row(load_state(ws), a)
        assert row["voxel_size_source"] == "user"
        assert row["voxel_z_from_xy"] is False
        assert calls == []
        assert w.applied_voxel[3] == "user"

    def test_a_typed_value_is_marked_checked(self, tmp_path):
        ws = _ws(tmp_path)
        a = _zyx_tif(tmp_path, "a.tif")
        _unset_rows(ws, a)
        ctl, w = _sel_ctl(ws)
        ctl._select_source(a)
        for s, v in zip((w.voxel_z_spin, w.voxel_y_spin, w.voxel_x_spin),
                        (2.0, 0.5, 0.5)):                  # = the header
            s.setValue(v)
        ctl.on_voxel_edited()
        ctl._flush_source_state_save()
        ctl._selected_source_path = None
        ctl._select_source(a)
        ctl._flush_source_state_save()
        row = ctl._find_source_row(load_state(ws), a)
        assert row["voxel_size_source"] == "user"
        assert row["voxel_header_checked"] is True

    def test_prepare_of_the_recorded_value_keeps_its_flag(self, tmp_path):
        """The row recorded 10/10/10 with Z assumed; the file's header now
        says 2/0.5/0.5; Prepare bakes the recorded 10s: the flag stays."""
        ws = _ws(tmp_path)
        src = _zyx_tif(tmp_path, "p.tif", z=2.0, xy=0.5)
        store = _prep_store(ws, "p", src, voxel_size=(10.0, 10.0, 10.0))
        st = empty_state()
        upsert_source(st, path=src, source_id="src_0",
                      voxel_size=[10.0, 10.0, 10.0],
                      voxel_size_source="detected", voxel_z_from_xy=True)
        save_state(ws, st)
        ctl, w = _sel_ctl(ws)
        ctl._log = lambda *a, **k: None
        ctl._largedata_src_path = src
        ctl._largedata_source_id = "src_0"
        ctl._largedata_prepare_params = None
        ctl._persist_prepared_source(store, "p")
        row = ctl._find_source_row(load_state(ws), src)
        assert row["voxel_size"] == [10.0, 10.0, 10.0]
        assert row["voxel_z_from_xy"] is True

    def test_prepare_of_another_value_the_header_does_not_give(self, tmp_path):
        ws = _ws(tmp_path)
        src = _zyx_tif(tmp_path, "p.tif", z=2.0, xy=0.5)
        store = _prep_store(ws, "p", src, voxel_size=(3.0, 0.4, 0.4))
        st = empty_state()
        upsert_source(st, path=src, source_id="src_0",
                      voxel_size_source="unknown", voxel_z_from_xy=True)
        save_state(ws, st)
        ctl, w = _sel_ctl(ws)
        ctl._log = lambda *a, **k: None
        ctl._largedata_src_path = src
        ctl._largedata_source_id = "src_0"
        ctl._largedata_prepare_params = None
        ctl._persist_prepared_source(store, "p")
        row = ctl._find_source_row(load_state(ws), src)
        assert row["voxel_size_source"] == "detected"
        assert row["voxel_z_from_xy"] is False

    def test_an_over_ceiling_header_is_never_recorded_at_extract(
            self, tmp_path):
        """72 dpi read as cm (352.8 um): the extract of a new source records
        no voxel size, and the note stays on the row's next select."""
        ws = _ws(tmp_path)
        a = str(tmp_path / "dpi.tif")
        tifffile.imwrite(a, np.zeros((4, 16, 16), np.uint16),
                         resolution=(72 / 2.54, 72 / 2.54),
                         resolutionunit="CENTIMETER")
        ctl, w = _sel_ctl(ws)
        ctl._log = lambda *a, **k: None
        ctl._select_source(a)                          # no row yet
        assert w.over_ceiling is True                  # shown, not recorded
        ctl._loaded_source_file = a
        _persist_from(ctl._gather_persist_params(), "dpi_patch_1")
        row = ctl._find_source_row(load_state(ws), a)
        assert row.get("voxel_size") is None
        assert row["voxel_size_source"] == "unknown"
        ctl._selected_source_path = None
        ctl._select_source(a)
        assert w.over_ceiling is True
        assert ctl._pending_source_edits == []

    def test_the_retype_of_a_user_row_is_said_in_the_log(self, tmp_path):
        ws = _ws(tmp_path)
        a = _xy_only_tif(tmp_path, "xy.tif")
        st = empty_state()
        upsert_source(st, path=a, display_name="kidney",
                      voxel_size=[10.0, 10.0, 10.0], voxel_size_source="user")
        save_state(ws, st)
        ctl, w = _sel_ctl(ws)
        lines = []
        ctl._log = lambda prefix, msg: lines.append(f"[{prefix}] {msg}")
        ctl._select_source(a)
        ctl._flush_source_state_save()
        assert ("[STEP1-RESTORE] kidney: the voxel size saved as typed "
                "equals the file's header; now marked as read from the file"
                ) in lines

    def test_no_log_line_when_the_user_row_stays_typed(self, tmp_path):
        ws = _ws(tmp_path)
        a = _xy_only_tif(tmp_path, "xy.tif")
        st = empty_state()
        upsert_source(st, path=a, display_name="kidney",
                      voxel_size=[3.0, 10.0, 10.0], voxel_size_source="user")
        save_state(ws, st)
        ctl, w = _sel_ctl(ws)
        lines = []
        ctl._log = lambda prefix, msg: lines.append(msg)
        ctl._select_source(a)
        ctl._flush_source_state_save()
        assert not any("now marked as read from the file" in m
                       for m in lines)

    def test_reset_to_unknown_writes_the_flag_false(self, tmp_path):
        ws = _ws(tmp_path)
        a = _tif(tmp_path, "plain.tif", shape=(5, 32, 32))   # no size tags
        st = empty_state()
        upsert_source(st, path=a, voxel_size=[3.0, 1.0, 1.0],
                      voxel_size_source="user", voxel_z_from_xy=True)
        save_state(ws, st)
        ctl, w = _sel_ctl(ws)
        ctl._select_source(a)
        ctl.on_voxel_reset()
        ctl._flush_source_state_save()
        row = ctl._find_source_row(load_state(ws), a)
        assert row["voxel_size_source"] == "unknown"
        assert row["voxel_z_from_xy"] is False

    def test_the_select_fill_never_takes_the_spins(self, tmp_path):
        """The spins hold another value (typed, not yet saved, or left by
        another source): the fill records the FILE's header only."""
        ws = _ws(tmp_path)
        a = _zyx_tif(tmp_path, "a.tif", z=2.0, xy=0.5)
        _unset_rows(ws, a)
        ctl, w = _sel_ctl(ws)
        real_apply = w.apply_voxel_size

        def apply_then_scramble(z, y, x, source="detected"):
            real_apply(z, y, x, source)
            for s, v in zip((w.voxel_z_spin, w.voxel_y_spin, w.voxel_x_spin),
                            (9.0, 7.0, 7.0)):
                s.setValue(v)
        w.apply_voxel_size = apply_then_scramble
        ctl._select_source(a)
        ctl._flush_source_state_save()
        row = ctl._find_source_row(load_state(ws), a)
        assert row["voxel_size"] == pytest.approx([2.0, 0.5, 0.5])

    def test_the_marker_is_not_handed_on(self, tmp_path, monkeypatch):
        """The flush hands the guarded writer (and upsert) a dict WITHOUT the
        marker; the buffered edit keeps it until the save succeeds."""
        import napari_epicycle.data_preparation.controller as ctl_mod
        ws = _ws(tmp_path)
        a = _zyx_tif(tmp_path, "a.tif")
        _unset_rows(ws, a)
        ctl, w = _sel_ctl(ws)
        seen = []
        real_guarded = ctl._apply_guarded_voxel_edit
        monkeypatch.setattr(
            ctl, "_apply_guarded_voxel_edit",
            lambda state, guard, kw: (seen.append(dict(kw))
                                      or real_guarded(state, guard, kw)))
        real_upsert = ctl_mod.upsert_source
        monkeypatch.setattr(
            ctl_mod, "upsert_source",
            lambda state, **kw: (seen.append(dict(kw))
                                 or real_upsert(state, **kw)))
        ctl._select_source(a)
        ctl._queue_source_save(path=a, display_name="x")
        buffered = [dict(e) for e in ctl._pending_source_edits]
        ctl._flush_source_state_save()
        assert len(seen) == 2
        assert all("_fill_unset" not in kw for kw in seen)
        assert buffered[0].get("_fill_unset") is True


class _DepthWidget(_SelWidget):
    def __init__(self, ws=""):
        super().__init__(ws)
        self.default_depth = None

    def set_voxel_z_default_depth(self, flag):
        self.default_depth = bool(flag)


def test_imagejs_default_depth_reaches_the_label(tmp_path):
    ws = _ws(tmp_path)
    a = str(tmp_path / "ij.tif")
    tifffile.imwrite(a, np.zeros((3, 8, 8), np.uint8), imagej=True,
                     resolution=(5, 5),
                     metadata={"axes": "ZYX", "unit": "um", "ImageJ": "1.54f"})
    w = _DepthWidget(ws)
    ctl = _ctl(widget=w)
    ctl._select_source(a)
    assert w.applied_voxel[:3] == pytest.approx((1.0, 0.2, 0.2))
    assert w.default_depth is True
    b = _zyx_tif(tmp_path, "b.tif")                # an explicit spacing
    ctl._select_source(b)
    assert w.default_depth is False


# ===========================================================================
# One ceiling for every writer that types a value "detected"
# ===========================================================================

def _dpi_tif(tmp_path, name="dpi.tif"):
    """72 dpi read as centimetres: a 352.78 um pixel, no Z."""
    p = str(tmp_path / name)
    tifffile.imwrite(p, np.zeros((4, 16, 16), np.uint16),
                     resolution=(72 / 2.54, 72 / 2.54),
                     resolutionunit="CENTIMETER")
    return p


class TestOneCeiling:
    def test_reset_over_the_ceiling_saves_unknown_and_keeps_the_note(
            self, tmp_path):
        ws = _ws(tmp_path)
        a = _dpi_tif(tmp_path)
        st = empty_state()
        upsert_source(st, path=a, voxel_size=[3.0, 1.0, 1.0],
                      voxel_size_source="user", voxel_header_checked=True)
        save_state(ws, st)
        ctl, w = _sel_ctl(ws)
        ctl._select_source(a)
        w._detected_voxel_size = (1.0, 1.0, 1.0)      # a user row's baseline
        ctl.on_voxel_reset()
        ctl._flush_source_state_save()
        row = ctl._find_source_row(load_state(ws), a)
        assert row["voxel_size_source"] == "unknown"
        assert row["voxel_size"] == [1.0, 1.0, 1.0]
        assert w.over_ceiling is True

    def test_prepare_never_types_a_baked_print_dpi_detected(self, tmp_path):
        ws = _ws(tmp_path)
        src = _dpi_tif(tmp_path, "p.tif")
        store = _prep_store(ws, "p", src, voxel_size=(352.78, 352.78, 352.78))
        st = empty_state()
        upsert_source(st, path=src, source_id="src_0",
                      voxel_size_source="unknown")
        save_state(ws, st)
        ctl, w = _sel_ctl(ws)
        ctl._log = lambda *a, **k: None
        ctl._largedata_src_path = src
        ctl._largedata_source_id = "src_0"
        ctl._largedata_prepare_params = None
        ctl._persist_prepared_source(store, "p")
        row = ctl._find_source_row(load_state(ws), src)
        assert row["voxel_size_source"] == "unknown"
        assert row.get("voxel_size") is None

    def test_the_orphan_store_row_never_types_a_print_dpi_detected(
            self, tmp_path):
        ws = str(tmp_path)
        src = _tif(tmp_path, "orphan.tif")
        _prep_store(ws, "orphan", src, voxel_size=(352.78, 352.78, 352.78))
        st = empty_state()
        assert _ctl()._reconcile_sources_with_stores(ws, st) is True
        row = st["sources"][0]
        assert row["voxel_size_source"] == "unknown"
        assert row.get("voxel_size") is None

    def test_the_one_time_check_never_relabels_a_print_dpi(self, tmp_path):
        """A legacy typed row holding the 352.78 a print header gives stays
        typed (checked), never relabelled as read from the file."""
        ws = _ws(tmp_path)
        a = _dpi_tif(tmp_path)
        st = empty_state()
        v = [352.778, 352.778, 352.778]
        upsert_source(st, path=a, voxel_size=v, voxel_size_source="user")
        save_state(ws, st)
        ctl, w = _sel_ctl(ws)
        ctl._select_source(a)
        ctl._flush_source_state_save()
        row = ctl._find_source_row(load_state(ws), a)
        assert row["voxel_size_source"] == "user"
        assert row["voxel_header_checked"] is True

    def test_the_carried_stub_keeps_its_checked_mark(self):
        stub = {"voxel_size": [2.0, 0.5, 0.5], "voxel_size_source": "user",
                "voxel_header_checked": True}
        winner = {"voxel_size": None, "voxel_size_source": "unknown",
                  "voxel_z_from_xy": True}
        DataPreparationController._carry_stub_fields(stub, winner)
        assert winner["voxel_size_source"] == "user"
        assert winner["voxel_header_checked"] is True
        assert winner["voxel_z_from_xy"] is False

    def test_prepare_of_a_new_value_with_no_header_drops_the_old_flag(
            self, tmp_path):
        """Case D: the header cannot be read and the store holds another
        value than the recorded one: the old flag spoke of the old value, so
        it is removed (absent reads as not assumed)."""
        ws = _ws(tmp_path)
        src = _tif(tmp_path, "p.tif", shape=(5, 32, 32))    # no size tags
        store = _prep_store(ws, "p", src, voxel_size=(3.0, 0.4, 0.4))
        st = empty_state()
        upsert_source(st, path=src, source_id="src_0",
                      voxel_size=[0.2, 0.2, 0.2],
                      voxel_size_source="detected", voxel_z_from_xy=True)
        save_state(ws, st)
        ctl, w = _sel_ctl(ws)
        lines = []
        ctl._log = lambda prefix, msg: lines.append(msg)
        ctl._largedata_src_path = src
        ctl._largedata_source_id = "src_0"
        ctl._largedata_prepare_params = None
        ctl._persist_prepared_source(store, "p")
        row = ctl._find_source_row(load_state(ws), src)
        assert row["voxel_size"] == [3.0, 0.4, 0.4]
        assert "voxel_z_from_xy" not in row
        assert any("Z-assumed mark was removed" in m for m in lines)



# ===========================================================================
# ImageJ's default depth counts as assumed; a print DPI is not in the boxes
# ===========================================================================

def _fiji_fixture():
    return os.path.join(os.path.dirname(__file__), "fixtures", "imagej_depth",
                        "zyx_depth_absent.tif")


class _DefaultWidget(_SelWidget):
    def __init__(self, ws=""):
        super().__init__(ws)
        self.default_depth = None
        self.default_unit = None

    def set_voxel_z_default_depth(self, flag, unit=None):
        self.default_depth = bool(flag)
        self.default_unit = unit


class TestImageJDefaultDepthIsAssumed:
    def test_the_select_fill_flags_the_default_depth(self, tmp_path):
        ws = _ws(tmp_path)
        a = _fiji_fixture()
        _unset_rows(ws, a)
        w = _DefaultWidget(ws)
        ctl = _ctl(widget=w)
        ctl._select_source(a)
        assert w.applied_voxel[:3] == pytest.approx((1.0, 0.2, 0.2))
        assert w.z_assumed is True
        assert w.default_depth is True and w.default_unit == "um"
        ctl._flush_source_state_save()
        row = ctl._find_source_row(load_state(ws), a)
        assert row["voxel_size"] == pytest.approx([1.0, 0.2, 0.2])
        assert row["voxel_z_from_xy"] is True

    def test_the_extract_flags_the_default_depth_and_steps_3_4_see_it(
            self, tmp_path):
        from napari_epicycle.data_preparation._state import (
            z_assumed_patch_paths,
        )
        ws = _ws(tmp_path)
        a = _fiji_fixture()
        w = _DefaultWidget(ws)
        ctl = _ctl(widget=w)
        ctl._select_source(a)                       # row-less: no fill
        ctl._loaded_source_file = a
        r = _persist_from(ctl._gather_persist_params(), "fiji_patch_1")
        st = load_state(ws)
        row = ctl._find_source_row(st, a)
        assert row["voxel_size_source"] == "detected"
        assert row["voxel_z_from_xy"] is True
        assert _norm_path(r["relative_path"]) in z_assumed_patch_paths(st)

    def test_an_explicit_spacing_is_not_assumed(self, tmp_path):
        ws = _ws(tmp_path)
        a = os.path.join(os.path.dirname(_fiji_fixture()), "zyx_depth_2.tif")
        _unset_rows(ws, a)
        w = _DefaultWidget(ws)
        ctl = _ctl(widget=w)
        ctl._select_source(a)
        ctl._flush_source_state_save()
        assert w.z_assumed is False and w.default_depth is False
        assert ctl._find_source_row(load_state(ws), a)["voxel_z_from_xy"] \
            is False


class TestPrintDpiNotInTheBoxes:
    def test_typing_only_z_saves_no_print_dpi(self, tmp_path):
        ws = _ws(tmp_path)
        a = _dpi_tif(tmp_path)
        _unset_rows(ws, a)
        ctl, w = _sel_ctl(ws)
        ctl._log = lambda *a, **k: None
        ctl._select_source(a)
        assert w.spins() == (1.0, 1.0, 1.0)         # not the header's 352.78
        assert w.over_ceiling is True
        w.voxel_z_spin.setValue(2.0)                # the user types Z only
        ctl.on_voxel_edited()
        ctl._flush_source_state_save()
        row = ctl._find_source_row(load_state(ws), a)
        assert row["voxel_size"] == [2.0, 1.0, 1.0]
        assert row["voxel_size_source"] == "user"

    def test_reset_never_puts_a_print_dpi_in_the_boxes(self, tmp_path):
        ws = _ws(tmp_path)
        a = _dpi_tif(tmp_path)
        st = empty_state()
        upsert_source(st, path=a, voxel_size=[3.0, 1.0, 1.0],
                      voxel_size_source="user", voxel_header_checked=True)
        save_state(ws, st)
        ctl, w = _sel_ctl(ws)
        ctl._select_source(a)
        w._detected_voxel_size = (1.0, 1.0, 1.0)
        w.applied_voxel = None
        ctl.on_voxel_reset()
        assert w.applied_voxel is None or w.applied_voxel[0] < 50
        assert w.over_ceiling is True


class TestPrintDpiLegacyRows:
    def test_a_processed_patch_of_a_print_dpi_file_gets_no_352(self, tmp_path):
        """Prepare and Process bake the SHOWN boxes into the store scale; the
        boxes no longer take a print DPI, so a processed patch of such a file
        never resolves to 352.78 um (it gets what the boxes hold: unknown)."""
        ws = _ws(tmp_path)
        a = _dpi_tif(tmp_path)
        ctl, w = _sel_ctl(ws)
        ctl._log = lambda *a, **k: None
        ctl._select_source(a)
        assert ctl._prepare_voxel_size() is None        # nothing to bake
        assert ctl._preview_spacing() == (1.0, 1.0, 1.0)
        from napari_epicycle.data_preparation._voxel_spacing import (
            resolve_stack_spacing,
        )
        store = _prep_store(ws, "dpi", a, voxel_size=ctl._prepare_voxel_size())
        st = empty_state()
        upsert_source(st, path=a, source_id="src_0", voxel_size_source="unknown",
                      processed_store={"path": os.path.relpath(store, ws)
                                       .replace(os.sep, "/"),
                                       "format": "ome-zarr"})
        st["patches"].append({"patch_name": "p", "source_id": "src_0",
                              "source_kind": "processed",
                              "relative_path": "artifacts/step1/patches/p.tiff"})
        got = resolve_stack_spacing(
            None, st, os.path.join(ws, "artifacts", "step1", "patches",
                                   "p.tiff"), ws, read_file=True)
        assert got is None or all(v < 50 for v in got["zyx"])

    def test_a_legacy_detected_print_dpi_row_is_cleared(self, tmp_path):
        ws = _ws(tmp_path)
        a = _dpi_tif(tmp_path)
        st = empty_state()
        upsert_source(st, path=a, voxel_size=[352.778, 352.778, 352.778],
                      voxel_size_source="detected", voxel_z_from_xy=True)
        save_state(ws, st)
        ctl, w = _sel_ctl(ws)
        lines = []
        ctl._log = lambda prefix, msg: lines.append(msg)
        ctl._select_source(a)
        assert w.applied_voxel == (1.0, 1.0, 1.0, "unknown")
        assert w.over_ceiling is True
        ctl._flush_source_state_save()
        row = ctl._find_source_row(load_state(ws), a)
        assert row["voxel_size"] is None
        assert row["voxel_size_source"] == "unknown"
        assert "voxel_z_from_xy" not in row
        assert any("looks like a print resolution and was dropped" in m
                   for m in lines)

    def test_a_legacy_detected_print_dpi_row_is_cleared_when_the_file_is_gone(
            self, tmp_path):
        ws = _ws(tmp_path)
        a = str(tmp_path / "gone.tif")
        st = empty_state()
        upsert_source(st, path=a, voxel_size=[352.778, 352.778, 352.778],
                      voxel_size_source="detected")
        save_state(ws, st)
        ctl, w = _sel_ctl(ws)
        ctl._log = lambda *a, **k: None
        ctl._select_source(a)
        assert w.applied_voxel[3] == "unknown"
        assert w.over_ceiling is True
        ctl._flush_source_state_save()
        assert ctl._find_source_row(load_state(ws), a)["voxel_size"] is None

    def test_a_typed_print_dpi_is_kept_and_said(self, tmp_path):
        """A legacy typed 352.78: it stays the user's, the panel never shows
        it as read from the file, and the typed form of the line shows."""
        ws = _ws(tmp_path)
        a = _dpi_tif(tmp_path)
        st = empty_state()
        v = [2.0, 352.778, 352.778]
        upsert_source(st, path=a, voxel_size=v, voxel_size_source="user")
        save_state(ws, st)
        ctl, w = _sel_ctl(ws)
        ctl._select_source(a)
        assert w.applied_voxel == (2.0, 352.778, 352.778, "user")
        assert w.over_ceiling is True
        ctl._flush_source_state_save()
        row = ctl._find_source_row(load_state(ws), a)
        assert row["voxel_size"] == v and row["voxel_size_source"] == "user"

    def test_the_panel_never_shows_a_legacy_typed_print_dpi_as_detected(
            self, tmp_path):
        """The controller half of the one-time check: a typed row equal to a
        print-resolution header is not re-shown as read from the file."""
        ws = _ws(tmp_path)
        a = _dpi_tif(tmp_path)
        det = read_voxel_spacing(a)
        st = empty_state()
        upsert_source(st, path=a, voxel_size=[round(x, 3) for x in det],
                      voxel_size_source="user")
        save_state(ws, st)
        ctl, w = _sel_ctl(ws)
        ctl._select_source(a)
        assert w.applied_voxel[3] == "user"
        ctl._flush_source_state_save()
        row = ctl._find_source_row(load_state(ws), a)
        assert row["voxel_size_source"] == "user"
        assert row["voxel_header_checked"] is True

    def test_an_orphan_store_with_a_header_borne_nan_or_5000_is_not_detected(
            self, tmp_path):
        ws = str(tmp_path)
        src = _tif(tmp_path, "orphan.tif")
        _prep_store(ws, "orphan", src, voxel_size=(5000.0, 0.5, 0.5))
        st = empty_state()
        _ctl()._reconcile_sources_with_stores(ws, st)
        row = st["sources"][0]
        assert row["voxel_size_source"] == "unknown"
        assert row.get("voxel_size") is None



# ===========================================================================
# The ceiling in every READER
# ===========================================================================

class TestCeilingInEveryReader:
    def test_a_store_baked_at_a_print_dpi_does_not_fill_the_boxes(
            self, tmp_path):
        ws = _ws(tmp_path)
        src = _tif(tmp_path, "s.tif", shape=(5, 32, 32))
        ctl, w = _sel_ctl(ws)
        ctl._log = lambda *a, **k: None
        ctl._apply_store_voxel_to_widget(
            {"scales": [[352.78, 352.78, 352.78]]}, src)
        assert w.applied_voxel == (1.0, 1.0, 1.0, "unknown")
        assert w.over_ceiling is True
        assert ctl._preview_spacing() == (1.0, 1.0, 1.0)   # nothing to bake

    def test_a_real_store_scale_still_fills_the_boxes(self, tmp_path):
        ws = _ws(tmp_path)
        src = _tif(tmp_path, "s.tif", shape=(5, 32, 32))
        ctl, w = _sel_ctl(ws)
        ctl._apply_store_voxel_to_widget({"scales": [[2.0, 0.5, 0.5]]}, src)
        assert w.applied_voxel == (2.0, 0.5, 0.5, "detected")

    def test_a_processed_store_at_a_print_dpi_is_not_step3s_value(
            self, tmp_path):
        from napari_epicycle.data_preparation._voxel_spacing import (
            resolve_stack_spacing,
        )
        ws = _ws(tmp_path)
        src = _tif(tmp_path, "s.tif", shape=(5, 32, 32))
        store = _prep_store(ws, "s", src, voxel_size=(352.78, 352.78, 352.78))
        st = empty_state()
        upsert_source(st, path=src, source_id="src_0",
                      voxel_size_source="unknown",
                      processed_store={"path": os.path.relpath(store, ws)
                                       .replace(os.sep, "/"),
                                       "format": "ome-zarr"})
        st["patches"].append({"patch_name": "p", "source_id": "src_0",
                              "source_kind": "processed",
                              "relative_path": "artifacts/step1/patches/p.tiff"})
        got = resolve_stack_spacing(
            None, st, os.path.join(ws, "artifacts", "step1", "patches",
                                   "p.tiff"), ws, read_file=True)
        assert got is None

    def test_the_file_now_says_note_never_offers_a_print_dpi(self, tmp_path):
        ws = _ws(tmp_path)
        a = _dpi_tif(tmp_path)
        st = empty_state()
        upsert_source(st, path=a, voxel_size=[2.0, 0.5, 0.5],
                      voxel_size_source="detected", voxel_z_from_xy=False)
        save_state(ws, st)
        ctl, w = _sel_ctl(ws)
        ctl._select_source(a)
        assert w.applied_voxel == (2.0, 0.5, 0.5, "detected")
        assert w.note == "recorded"              # not a "file now says" tuple
        assert w.over_ceiling is True

    def test_reset_on_a_fiji_row_says_imagejs_default_depth(self, tmp_path):
        ws = _ws(tmp_path)
        a = os.path.join(os.path.dirname(__file__), "fixtures",
                         "imagej_depth", "zyx_depth_absent.tif")
        st = empty_state()
        upsert_source(st, path=a, voxel_size=[3.0, 0.2, 0.2],
                      voxel_size_source="user", voxel_header_checked=True)
        save_state(ws, st)
        w = _DefaultWidget(ws)
        ctl = _ctl(widget=w)
        ctl._select_source(a)
        w._detected_voxel_size = (1.0, 1.0, 1.0)
        ctl.on_voxel_reset()
        assert w.applied_voxel[:3] == pytest.approx((1.0, 0.2, 0.2))
        assert w.default_depth is True and w.default_unit == "um"
        assert w.z_assumed is True

    def test_a_typed_over_ceiling_value_is_said_only_on_a_legacy_row(
            self, tmp_path):
        ws = _ws(tmp_path)
        a = _tif(tmp_path, "s.tif", shape=(5, 32, 32))
        st = empty_state()
        upsert_source(st, path=a, voxel_size=[2.0, 80.0, 80.0],
                      voxel_size_source="user", voxel_header_checked=True)
        save_state(ws, st)
        ctl, w = _sel_ctl(ws)
        ctl._select_source(a)
        assert w.applied_voxel == (2.0, 80.0, 80.0, "user")
        assert w.over_ceiling is False           # typed now: no ceiling

    def test_a_header_z_out_of_range_is_said(self, tmp_path):
        class _ZWidget(_SelWidget):
            z_out = "unset"

            def set_voxel_z_out_of_range(self, z):
                self.z_out = z
        ws = _ws(tmp_path)
        a = str(tmp_path / "z.tif")
        tifffile.imwrite(a, np.zeros((3, 8, 8), np.uint8), imagej=True,
                         resolution=(1 / 0.2, 1 / 0.2),
                         metadata={"axes": "ZYX", "unit": "um",
                                   "spacing": 5000.0})
        w = _ZWidget(ws)
        ctl = _ctl(widget=w)
        ctl._select_source(a)
        assert w.applied_voxel[3] == "unknown"
        assert w.z_out == pytest.approx(5000.0)

    def test_the_drop_line_names_the_value_and_what_follows(self, tmp_path):
        ws = _ws(tmp_path)
        a = _dpi_tif(tmp_path)
        st = empty_state()
        upsert_source(st, path=a, display_name="kidney",
                      voxel_size=[352.778, 352.778, 352.778],
                      voxel_size_source="detected")
        save_state(ws, st)
        ctl, w = _sel_ctl(ws)
        lines = []
        ctl._log = lambda prefix, msg: lines.append(f"[{prefix}] {msg}")
        ctl._select_source(a)
        assert ("[STEP1-RESTORE] the voxel size recorded earlier for kidney "
                "(352.8 µm) looks like a print resolution and was dropped; its "
                "patches now measure in voxels until you type the real values"
                ) in lines

    def test_a_false_flag_on_a_fiji_row_is_upgraded(self, tmp_path):
        ws = _ws(tmp_path)
        a = os.path.join(os.path.dirname(__file__), "fixtures",
                         "imagej_depth", "zyx_depth_absent.tif")
        st = empty_state()
        upsert_source(st, path=a, voxel_size=[1.0, 0.2, 0.2],
                      voxel_size_source="detected", voxel_z_from_xy=False)
        save_state(ws, st)
        ctl, w = _sel_ctl(ws)
        ctl._select_source(a)
        ctl._flush_source_state_save()
        assert ctl._find_source_row(load_state(ws), a)["voxel_z_from_xy"] \
            is True
