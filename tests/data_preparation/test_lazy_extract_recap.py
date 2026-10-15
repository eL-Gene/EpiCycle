"""Step 1 EXTRACT off the LAZY full-res handle — shape + recap sourcing (Brick 1).

The extract path no longer materializes the whole volume via
``get_combined_image_stack``'s ``np.stack``: when a usable ``_source_lazy``
handle exists, the box-filter SHAPE comes from that handle's ``.shape`` (no
read) and the recap MIP comes from a small strided OVERVIEW
(``downsampled_max_intensity_project``) — never the whole volume.  The extracted
PATCH PIXELS stay byte-identical (only the crop source's *shape* + the recap
*source* change).  A multiscale display layer (whose ``.data`` is a LIST of
pyramid levels) is routed entirely through the lazy handle, so no code path calls
``.ndim`` / ``np.stack`` on the list.  When there is no lazy handle the old
materialize path is the byte-identical fallback.

Coverage:

* ``downsampled_max_intensity_project`` — byte-identical to the full-stack MIP
  when no striding is needed; channel selection/order; strided overview shrinks
  the read; 4D-only guard.
* ``_resolve_extract_source`` — lazy path returns the handle + shape with NO
  whole-volume array; fallback materializes exactly as before.
* multiscale (list ``.data``) — ``_live_channel_names`` keeps them,
  ``_crop_source_for`` crops them off the lazy handle, extraction matches the
  eager reference and never throws on the list.
* ``tracemalloc`` — a box extract + recap off a big memmap source reads ≪ the
  full volume.
* full ``extract_patches()`` end-to-end (Qt) — lazy path is byte-identical AND
  never calls ``get_combined_image_stack``; multiscale display layers don't
  throw; the no-lazy fallback works.
"""
from __future__ import annotations

import gc
import time
import tracemalloc
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from napari_epicycle.data_preparation.controller import DataPreparationController
from napari_epicycle.data_preparation.logic import (
    box_signature,
    downsampled_max_intensity_project,
    extract_patches_from_shapes,
    generate_max_intensity_project,
)
from napari_epicycle.data_preparation._largedata import open_lazy


# ---------------------------------------------------------------------------
# helpers / fakes
# ---------------------------------------------------------------------------

def _box(y0, x0, y1, x1):
    return np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]], dtype=float)


def _write(tmp_path, name, arr, compression=None):
    p = tmp_path / name
    tifffile.imwrite(str(p), arr, compression=compression)
    return str(p)


class _FakeLayer:
    def __init__(self, data):
        self.data = data


class _MultiscaleLayer:
    """A napari multiscale layer: ``.data`` is a LIST of pyramid-level arrays."""

    def __init__(self, levels):
        self.data = list(levels)


class _NamedLayerList:
    """`name in layers` + `layers[name].data`, the two accessors the crop-source
    + eager-stack helpers use."""

    def __init__(self, mapping):
        self._m = dict(mapping)

    def __contains__(self, name):
        return name in self._m

    def __getitem__(self, name):
        return self._m[name]

    def __iter__(self):
        return iter(self._m.values())

    def __len__(self):
        return len(self._m)


class _ViewerWithLayers:
    def __init__(self, mapping):
        self.layers = _NamedLayerList(mapping)


def _seed_session_caches(ctl):
    """Seed the ④ session caches the real ``__init__`` creates.

    Every controller in this file is built with ``__new__`` (no ``__init__``),
    so it carries only the attributes each fixture sets by hand.  When the ④
    depth-profile cache landed (`_intensity_curve_cache`, 2026-09) the
    extract / trim / delete paths all started touching it, and this file went
    red: three tests raised ``AttributeError`` outright, and twelve more timed
    out because the exception fired INSIDE the extract worker's finished slot,
    so ``_extract_running`` never cleared and `_drive_extract` span until its
    deadline.  Rot, not a product bug — but it took out the only end-to-end
    coverage of the Trim / Delete downstream guard.

    Seeded in one place so the next attribute added to the ④ session state is a
    one-line fix here rather than fifteen red tests.
    """
    ctl._intensity_curve_cache = {}
    ctl._open_capture_patches = set()
    ctl._highlighted_patch = None
    ctl._visible_patch_locations = set()
    ctl._loc_overlay = None
    return ctl


def _bare_controller(**l0_state):
    ctl = DataPreparationController.__new__(DataPreparationController)
    _seed_session_caches(ctl)
    ctl._source_lazy = l0_state.get("lazy")
    ctl._source_axes_resolved = l0_state.get("order")
    ctl._channel_index_by_name = l0_state.get("index_map", {})
    return ctl


# ---------------------------------------------------------------------------
# downsampled_max_intensity_project — the pure recap projection
# ---------------------------------------------------------------------------

class TestDownsampledMip:
    def test_no_striding_is_byte_identical_to_full_mip(self):
        """Small source → factors (1, 1) → identical to a whole-stack MIP, so
        the small-file recap is unchanged."""
        arr = np.random.randint(0, 60000, (2, 8, 40, 40)).astype(np.uint16)
        proj, (fy, fx) = downsampled_max_intensity_project(arr, [0, 1])
        assert (fy, fx) == (1, 1)
        assert np.array_equal(proj, generate_max_intensity_project(arr))

    def test_channel_selection_and_order_preserved(self):
        arr = np.random.randint(0, 60000, (3, 6, 32, 32)).astype(np.uint16)
        proj, facs = downsampled_max_intensity_project(arr, [2, 0])  # subset+reorder
        assert facs == (1, 1)
        assert proj.shape[0] == 2
        assert np.array_equal(proj, generate_max_intensity_project(arr[[2, 0]]))

    def test_channel_none_keeps_all(self):
        arr = np.random.randint(0, 255, (2, 4, 16, 16)).astype(np.uint16)
        proj, _ = downsampled_max_intensity_project(arr, None)
        assert proj.shape[0] == 2
        assert np.array_equal(proj, generate_max_intensity_project(arr))

    def test_striding_shrinks_the_projection(self):
        arr = np.random.randint(0, 255, (1, 8, 100, 100)).astype(np.uint16)
        proj, (fy, fx) = downsampled_max_intensity_project(
            arr, [0], max_xy=25, max_z=4)
        assert fy > 1 and fx > 1
        assert proj.shape[1] < 100 and proj.shape[2] < 100
        # equals the MIP of the same strided overview (label-safe striding)
        want = np.max(arr[[0], ::2, ::4, ::4], axis=1)
        assert np.array_equal(proj, want)

    def test_rejects_3d_source(self):
        with pytest.raises(ValueError):
            downsampled_max_intensity_project(np.zeros((8, 16, 16)), [0])


# ---------------------------------------------------------------------------
# _resolve_extract_source — lazy vs materialize
# ---------------------------------------------------------------------------

class _FakeItem:
    def __init__(self, text):
        self._t = text

    def text(self):
        return self._t


class _FakeCombo:
    def __init__(self, names):
        self._names = list(names)

    def selectedItems(self):
        return [_FakeItem(n) for n in self._names]


class _FakeComboWidget:
    def __init__(self, names):
        self.image_layer_combo = _FakeCombo(names)

    def checked_channel_names(self):
        # A1: the extract readers now call this (inclusion = checkbox set, not
        # selection).  All fake channels are included, in row order.
        return list(self.image_layer_combo._names)


class TestResolveExtractSource:
    def test_lazy_path_returns_handle_shape_and_no_stack(self, tmp_path):
        arr = np.random.randint(0, 60000, (2, 8, 40, 40)).astype(np.uint16)
        path = _write(tmp_path, "l.tif", arr, "deflate")
        ctl = _bare_controller(
            lazy=open_lazy(path), order="CZYX",
            index_map={"s_ch0": 0, "s_ch1": 1})
        ctl.viewer = _ViewerWithLayers(
            {"s_ch0": _FakeLayer(arr[0]), "s_ch1": _FakeLayer(arr[1])})

        crop, idx, shape, image_data = ctl._resolve_extract_source(
            ["s_ch0", "s_ch1"])
        assert image_data is None            # NOTHING whole-volume materialized
        assert idx == [0, 1]
        assert tuple(shape[-2:]) == (40, 40)  # Y, X from the lazy handle's shape
        # cropping the handle equals the eager crop (byte-identity substrate)
        box = _box(2, 3, 30, 33)
        got = extract_patches_from_shapes(crop, [box], channel_indices=idx)
        want = extract_patches_from_shapes(arr, [box])
        for k, wk in zip(got, want):
            assert np.array_equal(got[k], want[wk])

    def test_fallback_materializes_when_no_lazy_handle(self):
        arr0 = np.random.randint(0, 255, (5, 16, 16)).astype(np.uint16)
        arr1 = np.random.randint(0, 255, (5, 16, 16)).astype(np.uint16)
        ctl = _bare_controller(lazy=None)
        ctl.widget = _FakeComboWidget(["s_ch0", "s_ch1"])
        ctl.viewer = _ViewerWithLayers(
            {"s_ch0": _FakeLayer(arr0), "s_ch1": _FakeLayer(arr1)})

        crop, idx, shape, image_data = ctl._resolve_extract_source(
            ["s_ch0", "s_ch1"])
        assert image_data is not None        # the old materialize path
        assert idx is None
        assert crop is image_data
        assert tuple(shape) == (2, 5, 16, 16)
        assert np.array_equal(image_data, np.stack([arr0, arr1], axis=0))


# ---------------------------------------------------------------------------
# _crop_source_for_processed — extract-from-processed channel map (P1)
# ---------------------------------------------------------------------------

_CZYX_AXES = [{"name": "c", "type": "channel"},
              {"name": "z", "type": "space"},
              {"name": "y", "type": "space"},
              {"name": "x", "type": "space"}]
_ZYX_AXES = [{"name": "z", "type": "space"},
             {"name": "y", "type": "space"},
             {"name": "x", "type": "space"}]


def _patch_open_multiscale(monkeypatch, level0, axes):
    """Make `controller.open_multiscale` return an `(levels, meta)` for level0."""
    import napari_epicycle.data_preparation.controller as C
    monkeypatch.setattr(
        C, "open_multiscale", lambda p: ([level0], {"axes": axes}))


class TestCropSourceForProcessed:
    def _ctl(self, **attrs):
        from napari_epicycle.data_preparation.controller import (
            DataPreparationController,
        )
        ctl = DataPreparationController.__new__(DataPreparationController)
        _seed_session_caches(ctl)
        ctl._current_processed_multiscale_store = "proc.zarr"
        ctl._source_lazy = None
        for k, v in attrs.items():
            setattr(ctl, k, v)
        return ctl

    def test_channel_map_preserves_ticked_row_order(self, monkeypatch):
        # store channels are ordered a, b, c; a REORDERED subset is ticked ->
        # indices resolve into STORE order, in the ticked ROW order (NOT range(n))
        store = np.arange(3 * 4 * 8 * 8, dtype=np.uint16).reshape(3, 4, 8, 8)
        _patch_open_multiscale(monkeypatch, store, _CZYX_AXES)
        ctl = self._ctl(_processed_redisplay={
            "kind": "store", "store_path": "proc.zarr", "names": ["a", "b", "c"]})
        handle, idx = ctl._crop_source_for_processed(["c", "a"])
        assert idx == [2, 0]
        assert tuple(handle.shape) == (3, 4, 8, 8)
        # cropping the processed handle at box coords reads the store's pixels
        box = _box(1, 1, 5, 5)
        got = extract_patches_from_shapes(handle, [box], channel_indices=idx)
        want = extract_patches_from_shapes(store[[2, 0]], [box])
        for k, wk in zip(got, want):
            assert np.array_equal(got[k], want[wk])

    def test_m3_refuses_channel_not_in_store(self, monkeypatch):
        store = np.zeros((2, 3, 8, 8), dtype=np.uint16)
        _patch_open_multiscale(monkeypatch, store, _CZYX_AXES)
        ctl = self._ctl(_processed_redisplay={"names": ["a", "b"]})
        with pytest.raises(ValueError, match="isn't in the processed set"):
            ctl._crop_source_for_processed(["a", "ghost"])

    def test_no_store_and_no_mem_raises(self):
        ctl = self._ctl(_current_processed_multiscale_store=None,
                        _processed_redisplay={})
        with pytest.raises(ValueError, match="No processed data to extract"):
            ctl._crop_source_for_processed(["a"])

    def test_in_memory_regime_crops_ram_arrays(self):
        # The eager (in-memory) Process regime has no on-disk store — the
        # processed volume is per-channel level-0 arrays in RAM.  Extract must
        # crop those (wrapped lazily), mapping ticked names to their index in the
        # mem `names` (ROW order), and read the same pixels a raw-coord crop would.
        a = np.random.randint(0, 60000, (4, 8, 8)).astype(np.uint16)
        b = np.random.randint(0, 60000, (4, 8, 8)).astype(np.uint16)
        ctl = self._ctl(
            _current_processed_multiscale_store=None,
            _processed_redisplay={
                "kind": "mem",
                "channel_levels": [[a], [b]],   # [level0, coarser…] per channel
                "names": ["a", "b"]})
        handle, idx = ctl._crop_source_for_processed(["b", "a"])   # reordered ticks
        assert idx == [1, 0]
        assert tuple(handle.shape) == (2, 4, 8, 8)                 # (C, Z, Y, X)
        box = _box(1, 1, 5, 5)
        got = extract_patches_from_shapes(handle, [box], channel_indices=idx)
        want = extract_patches_from_shapes(np.stack([b, a], 0), [box])
        for k, wk in zip(got, want):
            assert np.array_equal(got[k], want[wk])

    def test_no_channels_ticked_raises(self, monkeypatch):
        store = np.zeros((2, 3, 8, 8), dtype=np.uint16)
        _patch_open_multiscale(monkeypatch, store, _CZYX_AXES)
        ctl = self._ctl(_processed_redisplay={"names": ["a", "b"]})
        with pytest.raises(ValueError, match="No channels ticked"):
            ctl._crop_source_for_processed([])

    def test_single_channel_zyx_store_presents_as_czyx(self, monkeypatch):
        store = np.zeros((4, 8, 8), dtype=np.uint16)  # ZYX, no channel axis
        _patch_open_multiscale(monkeypatch, store, _ZYX_AXES)
        ctl = self._ctl(_processed_redisplay={"names": ["only"]})
        handle, idx = ctl._crop_source_for_processed(["only"])
        assert tuple(handle.shape) == (1, 4, 8, 8)   # (1, Z, Y, X)
        assert idx == [0]

    def test_names_fallback_to_marker_stripped_display_channels(self, monkeypatch):
        from napari_epicycle.data_preparation.controller import (
            PROCESSED_LAYER_MARKER,
        )
        store = np.zeros((2, 3, 8, 8), dtype=np.uint16)
        _patch_open_multiscale(monkeypatch, store, _CZYX_AXES)
        # No `_processed_redisplay["names"]` -> fall back to the display channels,
        # stripping the ⟨processed⟩ marker to recover the plain store order.
        disp = [(f"{PROCESSED_LAYER_MARKER}a", []), (f"{PROCESSED_LAYER_MARKER}b", [])]
        ctl = self._ctl(_processed_redisplay=None,
                        _processed_display_channels=disp)
        _handle, idx = ctl._crop_source_for_processed(["b"])
        assert idx == [1]

    def test_grid_mismatch_refused(self, monkeypatch):
        store = np.zeros((2, 3, 8, 8), dtype=np.uint16)   # processed Y/X = 8/8
        _patch_open_multiscale(monkeypatch, store, _CZYX_AXES)

        class _Raw:
            shape = (2, 3, 16, 16)   # raw Y/X = 16/16 -> differs

        ctl = self._ctl(_processed_redisplay={"names": ["a", "b"]},
                        _source_lazy=_Raw())
        with pytest.raises(ValueError, match="differs from the raw grid"):
            ctl._crop_source_for_processed(["a"])

    def test_resolve_extract_source_routes_to_processed(self, monkeypatch):
        # end-to-end through `_resolve_extract_source` with the processed view on:
        # returns the store handle + store-order indices, NO whole-volume array.
        store = np.arange(2 * 3 * 8 * 8, dtype=np.uint16).reshape(2, 3, 8, 8)
        _patch_open_multiscale(monkeypatch, store, _CZYX_AXES)
        ctl = self._ctl(_processed_redisplay={"names": ["a", "b"]})
        ctl._is_processed_view_showing = lambda: True
        crop, idx, shape, image_data = ctl._resolve_extract_source(["b", "a"])
        assert image_data is None
        assert idx == [1, 0]
        assert tuple(shape) == (2, 3, 8, 8)


# ---------------------------------------------------------------------------
# multiscale (list .data) — kept live + cropped off the lazy handle
# ---------------------------------------------------------------------------

class TestMultiscaleSafety:
    def test_live_channel_names_keeps_multiscale_layers(self, tmp_path):
        arr = np.random.randint(0, 60000, (2, 6, 32, 32)).astype(np.uint16)
        path = _write(tmp_path, "ms.tif", arr, "deflate")
        ctl = _bare_controller(
            lazy=open_lazy(path), order="CZYX",
            index_map={"s_ch0": 0, "s_ch1": 1})
        ctl.viewer = _ViewerWithLayers({
            "s_ch0": _MultiscaleLayer([arr[0], arr[0][::2, ::2, ::2]]),
            "s_ch1": _MultiscaleLayer([arr[1], arr[1][::2, ::2, ::2]]),
        })
        # a list .data must NOT be dropped (its effective ndim is a level's ndim)
        assert ctl._live_channel_names(["s_ch0", "s_ch1"]) == ["s_ch0", "s_ch1"]

    def test_crop_source_for_multiscale_matches_eager(self, tmp_path):
        arr = np.random.randint(0, 60000, (2, 6, 32, 32)).astype(np.uint16)
        path = _write(tmp_path, "ms2.tif", arr, "deflate")
        ctl = _bare_controller(
            lazy=open_lazy(path), order="CZYX",
            index_map={"s_ch0": 0, "s_ch1": 1})
        ctl.viewer = _ViewerWithLayers({
            "s_ch0": _MultiscaleLayer([arr[0], arr[0][::2, ::2, ::2]]),
            "s_ch1": _MultiscaleLayer([arr[1], arr[1][::2, ::2, ::2]]),
        })
        src, idx = ctl._crop_source_for(["s_ch0", "s_ch1"])
        assert idx == [0, 1]                  # multiscale layers NOT dropped
        box = _box(2, 3, 20, 21)
        got = extract_patches_from_shapes(src, [box], channel_indices=idx)
        want = extract_patches_from_shapes(arr, [box])
        for k, wk in zip(got, want):
            assert np.array_equal(got[k], want[wk])

    def test_dask_level0_handle_with_multiscale_display(self):
        """As if ``_source_lazy`` = ``open_multiscale`` level 0 (a lazy CZYX dask
        array) with multiscale display layers whose ``.data`` is a list."""
        da = pytest.importorskip("dask.array")
        arr = np.random.randint(0, 60000, (2, 6, 32, 32)).astype(np.uint16)
        lazy = da.from_array(arr, chunks=(1, 3, 16, 16))
        ctl = _bare_controller(
            lazy=lazy, order="CZYX", index_map={"s_ch0": 0, "s_ch1": 1})
        ctl.viewer = _ViewerWithLayers({
            "s_ch0": _MultiscaleLayer([arr[0]]),
            "s_ch1": _MultiscaleLayer([arr[1]]),
        })
        src, idx = ctl._crop_source_for(["s_ch0", "s_ch1"])
        box = _box(2, 3, 20, 21)
        got = extract_patches_from_shapes(src, [box], channel_indices=idx)
        want = extract_patches_from_shapes(arr, [box])
        for k, wk in zip(got, want):
            assert np.array_equal(got[k], want[wk])
        # the recap MIP also runs off the lazy handle (never the list .data)
        proj, _facs = downsampled_max_intensity_project(src, idx)
        assert proj.shape[0] == 2


# ---------------------------------------------------------------------------
# tracemalloc — a box extract + recap reads ≪ the full volume
# ---------------------------------------------------------------------------

class TestNoWholeVolumeMaterialization:
    def test_extract_and_recap_peak_ram_well_below_full_volume(self, tmp_path):
        shape = (2, 64, 320, 320)                       # uint16 ≈ 26 MB
        arr = np.random.randint(0, 60000, shape).astype(np.uint16)
        path = _write(tmp_path, "big.tif", arr)          # uncompressed → memmap
        full_bytes = int(arr.nbytes)
        del arr
        gc.collect()

        lazy = open_lazy(path)
        if hasattr(lazy, "compute"):
            pytest.skip("source did not open as a memmap; RAM ceiling is only "
                        "deterministic on the memmap path")

        box = _box(10, 10, 74, 74)                        # 64×64 ROI

        tracemalloc.start()
        try:
            patches = extract_patches_from_shapes(
                lazy, [box], channel_indices=[0, 1])
            proj, _facs = downsampled_max_intensity_project(
                lazy, [0, 1], max_xy=80, max_z=16)
            _cur, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        # Only the ROI crop + a strided overview were read — nothing near the
        # whole volume (which the old np.stack + full MIP would have needed).
        ceiling = full_bytes // 4
        assert peak < ceiling, (
            f"peak={peak} exceeded ceiling={ceiling} (full={full_bytes})")
        assert list(patches)                              # a real patch came back
        assert proj.shape[0] == 2


# ---------------------------------------------------------------------------
# full extract_patches() end-to-end (Qt) — the wiring
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def app():
    pytest.importorskip("qtpy")
    from qtpy.QtWidgets import QApplication

    existing = QApplication.instance()
    if existing is not None:
        yield existing
        return
    yield QApplication([])


class _FakeExtractWidget:
    """Minimal widget exposing exactly what ``extract_patches`` touches — real
    Qt sub-widgets for the ones it reads/writes, nothing else (the box-list
    getters are absent → the positional tick filter is skipped)."""

    def __init__(self, app, names, out_dir, display_name="kidney"):
        from qtpy.QtWidgets import (
            QAbstractItemView, QLabel, QLineEdit, QListWidget, QListWidgetItem,
            QProgressBar,
        )

        self.image_layer_combo = QListWidget()
        self.image_layer_combo.setSelectionMode(
            QAbstractItemView.MultiSelection)
        for n in names:
            it = QListWidgetItem(n)
            self.image_layer_combo.addItem(it)
            it.setSelected(True)
        self.progress_bar = QProgressBar()
        self.extracted_roi_list = QListWidget()
        self.roi_recap = QLabel()
        self.roi_recap.resize(200, 200)
        self.display_name_edit = QLineEdit(display_name)
        self.output_file_path = QLineEdit(out_dir)
        self.status = []

    def checked_channel_names(self):
        # A1: inclusion moved from row-selection to the checkbox set; the extract
        # readers (`_selected_channel_names` + `get_combined_image_stack`) now
        # call this.  The fake marks every channel included, so return all item
        # texts IN ROW ORDER (== the old all-selected-in-row-order behavior).
        return [self.image_layer_combo.item(r).text()
                for r in range(self.image_layer_combo.count())]

    def set_export_status(self, message, success=True):
        self.status.append((message, success))

    # F1: ③ Extract now names the op on the shared ② busy bar via this contract.
    # The controller calls these in try/except, so no-ops are enough for the
    # Qt-free harness.
    def _busy_show(self, label):
        pass

    def _busy_hide(self, idle="— idle"):
        pass


def _extract_controller(app, lazy, order, index_map, layer_map, names, out_dir):
    ctl = DataPreparationController.__new__(DataPreparationController)
    _seed_session_caches(ctl)
    ctl.widget = _FakeExtractWidget(app, names, out_dir)
    ctl.viewer = _ViewerWithLayers(layer_map)
    ctl._source_lazy = lazy
    ctl._source_axes_resolved = order
    ctl._channel_index_by_name = index_map
    ctl.extracted_patches = {}
    ctl._committed_box_sigs = set()
    ctl.box_tool = None
    # F1 async-extract state (the controller's __init__ normally seeds these;
    # this bare `__new__` controller must carry them for `extract_patches` PREP).
    ctl._extract_token = 0
    ctl._extract_running = False
    ctl._extract_launching = False
    ctl._extract_thread = None
    ctl._extract_worker = None
    ctl._draining_extract_threads = []
    ctl._extract_button_text = None
    ctl._extract_button_icon = None
    ctl._extract_status_ctx = {}
    # HIGH-1 single-writer: PREP flushes the debounced source-save + stops the
    # timer.  Both are no-ops here (no pending edits / no real timer).
    ctl._flush_source_state_save = lambda *a: None
    ctl._state_save_timer = SimpleNamespace(stop=lambda: None)
    # stub the parts unrelated to the L0 change
    ctl._refresh_conditioning_status = lambda: None
    ctl._refresh_source_patch_counts = lambda: None
    ctl.get_selected_shapes_layer = lambda: None
    return ctl


def _drive_extract(ctl, app, timeout_s=15.0):
    """Pump the main-thread event loop until the async ③ Extract worker has
    finished (F1) — the finished/written/progress slots are queued cross-thread
    connections delivered by `processEvents`.  Returns once `_extract_running`
    clears (or immediately if PREP early-returned without launching a worker)."""
    from qtpy.QtWidgets import QApplication
    deadline = time.time() + timeout_s
    while getattr(ctl, "_extract_running", False):
        QApplication.processEvents()
        if time.time() > deadline:
            raise AssertionError("extract worker did not finish within timeout")
        time.sleep(0.003)
    # Fully JOIN the worker thread before returning: block on the QThread's own
    # event loop exiting (isRunning -> False), then pump so `thread.finished` is
    # delivered and `_cleanup_extract_thread` nulls + deleteLaters the pair.  A
    # QThread GC'd (with the test-local `ctl`) while still running aborts with
    # "QThread destroyed while running".
    th = getattr(ctl, "_extract_thread", None)
    if th is not None:
        try:
            th.wait(3000)
        except Exception:
            pass
    for _ in range(10):
        QApplication.processEvents()


class TestExtractPatchesEndToEnd:
    def test_lazy_path_byte_identical_and_no_whole_volume_stack(
            self, app, tmp_path):
        arr = np.random.randint(0, 60000, (2, 8, 40, 40)).astype(np.uint16)
        path = _write(tmp_path, "src.tif", arr, "deflate")
        out = tmp_path / "ws"
        out.mkdir()
        ctl = _extract_controller(
            app, open_lazy(path), "CZYX",
            {"s_ch0": 0, "s_ch1": 1},
            {"s_ch0": _FakeLayer(arr[0]), "s_ch1": _FakeLayer(arr[1])},
            ["s_ch0", "s_ch1"], str(out))
        boxes = [_box(2, 3, 30, 33), _box(10, 10, 25, 40)]
        ctl.get_drawn_shapes = lambda: boxes

        calls = []
        real_stack = ctl.get_combined_image_stack
        ctl.get_combined_image_stack = lambda: (calls.append(1), real_stack())[1]

        ctl.extract_patches()
        _drive_extract(ctl, app)              # F1: extract runs on a worker

        assert calls == []                    # whole-volume np.stack NEVER built
        got = [info["data"] for info in ctl.extracted_patches.values()]
        want = list(extract_patches_from_shapes(arr, boxes).values())
        assert len(got) == 2
        for g, w in zip(got, want):
            assert np.array_equal(g, w)
        # channels stored verbatim (sidecar contract, unchanged)
        for info in ctl.extracted_patches.values():
            assert info["channels"] == ["s_ch0", "s_ch1"]

    def test_narrowed_box_writes_absolute_source_z_range(
            self, app, tmp_path, monkeypatch):
        """B3: a Z-narrowed box records SOURCE-ABSOLUTE `selected_z_range`
        `[z0, z0+depth]` (was `[0, depth]`, discarding the origin) — while a
        full-Z box stays `[0, Z]` byte-identical.  Also `width == cropped depth`."""
        import napari_epicycle.data_preparation.controller as ctrl_mod
        arr = np.random.randint(0, 60000, (2, 8, 40, 40)).astype(np.uint16)
        path = _write(tmp_path, "src_b3.tif", arr, "deflate")
        orig_persist = ctrl_mod.persist_patch

        def _run(z_range, out_name):
            out = tmp_path / out_name
            out.mkdir()
            ctl = _extract_controller(
                app, open_lazy(path), "CZYX",
                {"s_ch0": 0, "s_ch1": 1},
                {"s_ch0": _FakeLayer(arr[0]), "s_ch1": _FakeLayer(arr[1])},
                ["s_ch0", "s_ch1"], str(out))
            box = _box(2, 3, 30, 33)
            ctl.get_drawn_shapes = lambda: [box]
            ctl.box_tool = SimpleNamespace(
                has_boxes=lambda: True, z_ranges=lambda: [list(z_range)])
            captured = {}

            def _spy(workspace, name, data, **kw):
                captured["zr"] = kw.get("selected_z_range")
                captured["cs"] = kw.get("cropped_shape")
                return orig_persist(workspace, name, data, **kw)

            monkeypatch.setattr(ctrl_mod, "persist_patch", _spy)
            ctl.extract_patches()
            _drive_extract(ctl, app)
            return captured

        # Narrowed slab [2,6] → source-absolute [2,6], depth 4, width == depth.
        c = _run((2, 6), "ws_b3_narrow")
        assert c["zr"] == [2, 6]
        assert c["cs"][1] == 4
        assert c["zr"][1] - c["zr"][0] == c["cs"][1]
        # Full-Z [0,8] → [0,8] byte-identical, depth 8 (no regression).
        c = _run((0, 8), "ws_b3_full")
        assert c["zr"] == [0, 8]
        assert c["cs"][1] == 8

    def test_multiscale_display_layers_do_not_throw(self, app, tmp_path):
        arr = np.random.randint(0, 60000, (2, 8, 40, 40)).astype(np.uint16)
        path = _write(tmp_path, "ms_src.tif", arr, "deflate")
        out = tmp_path / "ws_ms"
        out.mkdir()
        layer_map = {
            "s_ch0": _MultiscaleLayer([arr[0], arr[0][::2, ::2, ::2]]),
            "s_ch1": _MultiscaleLayer([arr[1], arr[1][::2, ::2, ::2]]),
        }
        ctl = _extract_controller(
            app, open_lazy(path), "CZYX",
            {"s_ch0": 0, "s_ch1": 1}, layer_map, ["s_ch0", "s_ch1"], str(out))
        boxes = [_box(2, 3, 30, 33)]
        ctl.get_drawn_shapes = lambda: boxes

        calls = []
        real_stack = ctl.get_combined_image_stack
        ctl.get_combined_image_stack = lambda: (calls.append(1), real_stack())[1]

        ctl.extract_patches()                 # must not raise on the list .data
        _drive_extract(ctl, app)

        assert calls == []                    # routed through the lazy handle
        got = list(ctl.extracted_patches.values())
        assert len(got) == 1
        want = list(extract_patches_from_shapes(arr, boxes).values())[0]
        assert np.array_equal(got[0]["data"], want)

    def test_fallback_no_lazy_handle_still_extracts(self, app, tmp_path):
        arr = np.random.randint(0, 60000, (2, 8, 40, 40)).astype(np.uint16)
        out = tmp_path / "ws_fb"
        out.mkdir()
        ctl = _extract_controller(
            app, None, None, {},
            {"s_ch0": _FakeLayer(arr[0]), "s_ch1": _FakeLayer(arr[1])},
            ["s_ch0", "s_ch1"], str(out))
        boxes = [_box(2, 3, 30, 33)]
        ctl.get_drawn_shapes = lambda: boxes

        ctl.extract_patches()
        _drive_extract(ctl, app)

        got = list(ctl.extracted_patches.values())
        assert len(got) == 1
        want = list(extract_patches_from_shapes(
            np.stack([arr[0], arr[1]], axis=0), boxes).values())[0]
        assert np.array_equal(got[0]["data"], want)


# ---------------------------------------------------------------------------
# F1 — ③ Extract WRITES to disk immediately (persist merged into extract)
# ---------------------------------------------------------------------------

class TestExtractWritesToDiskImmediately:
    def test_extract_writes_tif_sidecar_registry_and_commits_sig(
            self, app, tmp_path):
        """③ Extract now persists each staged patch RIGHT AWAY: the .tif +
        FROZEN sidecar + registry row land on disk, the box signature is
        committed (so a re-extract of the same box is deduped), and the
        in-memory stage SURVIVES (④'s session list is untouched here)."""
        from napari_epicycle.data_preparation._paths import step1_patches_dir
        from napari_epicycle.data_preparation._state import load_state
        from napari_epicycle.data_preparation.logic import (
            box_signature, select_uncommitted_boxes,
        )

        arr = np.random.randint(0, 60000, (2, 8, 40, 40)).astype(np.uint16)
        path = _write(tmp_path, "src_w.tif", arr, "deflate")
        out = tmp_path / "ws_w"
        out.mkdir()
        ctl = _extract_controller(
            app, open_lazy(path), "CZYX",
            {"s_ch0": 0, "s_ch1": 1},
            {"s_ch0": _FakeLayer(arr[0]), "s_ch1": _FakeLayer(arr[1])},
            ["s_ch0", "s_ch1"], str(out))
        box = _box(2, 3, 30, 33)
        ctl.get_drawn_shapes = lambda: [box]

        ctl.extract_patches()
        _drive_extract(ctl, app)

        # The stage is KEPT (not popped) so ④ still shows this run's patch.
        assert set(ctl.extracted_patches) == {"kidney_patch_1"}

        # The .tif + FROZEN sidecar landed under artifacts/step1/patches/.
        pdir = step1_patches_dir(str(out))
        assert (pdir / "kidney_patch_1.tiff").exists()
        assert (pdir / "kidney_patch_1_metadata.json").exists()

        # A registry row was written for the patch.
        state = load_state(str(out))
        assert "kidney_patch_1" in [p["patch_name"] for p in state["patches"]]

        # The box signature was committed on the ③ write, so a second extract
        # of the same box is skipped by select_uncommitted_boxes (dedup).
        assert box_signature(box.tolist()) in ctl._committed_box_sigs
        assert select_uncommitted_boxes([box], ctl._committed_box_sigs) == []

    def test_persist_keeps_stage_while_export_pops(self, app, tmp_path):
        """`_persist_patch_to_disk` writes but must NOT pop the stage;
        `_export_patch_by_name` still pops (stage + UI list shrink)."""
        from qtpy.QtWidgets import QListWidgetItem

        out = tmp_path / "ws_pop"
        out.mkdir()
        ctl = _extract_controller(app, None, None, {}, {}, [], str(out))
        for nm in ("kidney_patch_1", "kidney_patch_2"):
            ctl.extracted_patches[nm] = {
                "data": np.random.randint(0, 255, (1, 3, 6, 6)).astype(np.uint16),
                "channels": ["s_ch0"],
                "bounding_box": _box(0, 0, 6, 6).tolist(),
                "source_kind": "raw",
            }
            ctl.widget.extracted_roi_list.addItem(QListWidgetItem(nm))

        # persist writes to disk but leaves the stage + list intact.
        result = ctl._persist_patch_to_disk("kidney_patch_1")
        assert result is not None
        assert set(ctl.extracted_patches) == {"kidney_patch_1", "kidney_patch_2"}
        assert ctl.widget.extracted_roi_list.count() == 2

        # export DOES pop the original key from both the stage and the list.
        assert ctl._export_patch_by_name(
            "kidney_patch_1", require_conditioned=False)
        assert set(ctl.extracted_patches) == {"kidney_patch_2"}
        assert ctl.widget.extracted_roi_list.count() == 1


# ---------------------------------------------------------------------------
# B1 + B10 — extract must not leak a Shapes layer or render the hidden recap
# ---------------------------------------------------------------------------

class TestExtractNoShapesLeakOrRecap:
    """B1: with the ③ box tool owning the boxes (a vispy overlay, NOT a napari
    layer), ``extract_patches`` used to call ``get_selected_shapes_layer()``
    unconditionally to label shapes — which in real napari CREATES an empty
    "ROI Shapes" layer that then strands cross-step (the Shapes-in-3D freed-VBO
    class the box-tool rework exists to eliminate).  B10: it also rendered a
    matplotlib recap into a hidden QLabel on every extract.  Both are removed —
    extract must touch neither.

    NOTE the pre-existing end-to-end tests stub ``get_selected_shapes_layer`` to
    ``None`` (``_extract_controller``), so they never exercised — nor caught —
    the leak.  This test SPIES on the call instead of neutering it.
    """

    def test_extract_touches_no_shapes_layer_and_no_recap(
            self, app, tmp_path, monkeypatch):
        from qtpy.QtGui import QPixmap
        import napari_epicycle.data_preparation.controller as C

        arr = np.random.randint(0, 60000, (2, 8, 40, 40)).astype(np.uint16)
        path = _write(tmp_path, "b1b10.tif", arr, "deflate")
        out = tmp_path / "ws_b1b10"
        out.mkdir()
        ctl = _extract_controller(
            app, open_lazy(path), "CZYX",
            {"s_ch0": 0, "s_ch1": 1},
            {"s_ch0": _FakeLayer(arr[0]), "s_ch1": _FakeLayer(arr[1])},
            ["s_ch0", "s_ch1"], str(out))
        boxes = [_box(2, 3, 30, 33)]
        ctl.get_drawn_shapes = lambda: boxes

        # B1 spy: extract must NOT reach for a napari Shapes layer.
        shapes_calls = []
        ctl.get_selected_shapes_layer = lambda: (
            shapes_calls.append(1), None)[1]

        # B10 spy: the hidden matplotlib recap must NOT be rendered.  Return a
        # real pixmap so the PRE-fix path completes and FAILS on the assert
        # rather than crashing on ``.scaled(None)``.
        recap_calls = []
        monkeypatch.setattr(
            C, "draw_boxes_on_projection_matplotlib",
            lambda *a, **k: (recap_calls.append(1), QPixmap(4, 4))[1])

        ctl.extract_patches()
        _drive_extract(ctl, app)

        assert shapes_calls == [], "extract created/reused a Shapes layer (B1)"
        assert recap_calls == [], "extract rendered the dead recap (B10)"
        # the extract itself still worked
        assert set(ctl.extracted_patches) == {"kidney_patch_1"}

    def test_get_drawn_shapes_fallback_never_creates_a_layer(self):
        """B1 hardening: with no box-tool boxes AND no existing Shapes layer,
        ``get_drawn_shapes`` returns ``[]`` via the non-creating
        ``_current_shapes_layer`` read — it must NEVER call ``add_shapes`` (the
        old fallback did, via ``get_selected_shapes_layer`` -> ``_ensure_...``)."""
        ctl = DataPreparationController.__new__(DataPreparationController)
        _seed_session_caches(ctl)
        ctl.box_tool = None

        class _Combo:
            def currentText(self):
                return "ROI Shapes"

        class _Widget:
            bbox_layer_combo = _Combo()

        class _Layers:
            def __contains__(self, name):
                return False

            def __iter__(self):
                return iter(())

        class _Viewer:
            layers = _Layers()

            def add_shapes(self, *a, **k):
                raise AssertionError("fallback created a Shapes layer (B1 leak)")

        ctl.widget = _Widget()
        ctl.viewer = _Viewer()

        assert ctl.get_drawn_shapes() == []


# ---------------------------------------------------------------------------
# F1 — stale-finish guard: a slot stamped with a superseded job token is inert
# ---------------------------------------------------------------------------

class TestExtractStaleToken:
    """The extract worker's slots (`_on_extract_written`/`_on_extract_finished`)
    each early-return on a token mismatch BEFORE touching the stage, the
    committed sigs, or the depth-counted busy lock — so a late emit from a
    superseded run can never corrupt a newer run's state or free the surface
    mid-run (HIGH-2 stale-token invariant)."""

    def _result(self):
        return {
            "name": "ghost_patch_1",
            "data": np.zeros((1, 2, 3, 3), dtype=np.uint16),
            "bounding_box": _box(0, 0, 3, 3).tolist(),
            "cropped_z_range": [0, 2],
            "source_kind": "raw",
            "channels": ["s_ch0"],
            "persist_ok": True,
            "persist_result": {"patch_name": "ghost_patch_1"},
            "sig_xy": box_signature(_box(0, 0, 3, 3).tolist()),
            "sig_z": box_signature(_box(0, 0, 3, 3).tolist(), [0, 2]),
        }

    def test_stale_written_is_a_noop(self, app, tmp_path):
        out = tmp_path / "ws_stale_w"
        out.mkdir()
        ctl = _extract_controller(app, None, None, {}, {}, [], str(out))
        ctl._extract_token = 7
        ctl._extract_running = True
        res = self._result()

        ctl._on_extract_written(res, token=6)          # stale — inert
        assert ctl.extracted_patches == {}
        assert ctl._committed_box_sigs == set()
        assert ctl.widget.extracted_roi_list.count() == 0

        ctl._on_extract_written(res, token=7)          # current — processes
        assert set(ctl.extracted_patches) == {"ghost_patch_1"}
        assert res["sig_xy"] in ctl._committed_box_sigs
        assert res["sig_z"] in ctl._committed_box_sigs

    def test_stale_finished_does_not_unlock(self, app, tmp_path):
        out = tmp_path / "ws_stale_f"
        out.mkdir()
        ctl = _extract_controller(app, None, None, {}, {}, [], str(out))
        ctl._extract_token = 7
        ctl._extract_running = True
        # A refresh stub that would blow up if the stale guard let it through —
        # proves the early-return happens BEFORE any FINISH work.
        ctl._refresh_library_from_registry = lambda: (_ for _ in ()).throw(
            AssertionError("stale finished slot ran FINISH work"))

        ctl._on_extract_finished(3, False, token=6)    # stale — inert
        assert ctl._extract_running is True            # NOT unlocked


# ---------------------------------------------------------------------------
# F1 — per-run stage reset (FIX 1): a SECOND extract must not accumulate the
# prior run's patches (else FINISH fabricates phantom write failures)
# ---------------------------------------------------------------------------

class TestExtractSequentialRuns:
    def test_second_extract_resets_stage_not_cumulative(self, app, tmp_path):
        arr = np.random.randint(0, 60000, (2, 8, 40, 40)).astype(np.uint16)
        path = _write(tmp_path, "seq.tif", arr, "deflate")
        out = tmp_path / "ws_seq"
        out.mkdir()
        ctl = _extract_controller(
            app, open_lazy(path), "CZYX",
            {"s_ch0": 0, "s_ch1": 1},
            {"s_ch0": _FakeLayer(arr[0]), "s_ch1": _FakeLayer(arr[1])},
            ["s_ch0", "s_ch1"], str(out))

        # Run 1 — boxes {A, B}.
        boxes1 = [_box(2, 3, 30, 33), _box(5, 5, 20, 20)]
        ctl.get_drawn_shapes = lambda: boxes1
        ctl.extract_patches()
        _drive_extract(ctl, app)
        assert set(ctl.extracted_patches) == {"kidney_patch_1", "kidney_patch_2"}

        # Run 2 — NEW boxes {C, D} (no discard-confirm modal any more — B5 removed;
        # extracted boxes clear from ③ + the F1 stage-reset discards run 1).
        boxes2 = [_box(1, 1, 15, 15), _box(8, 8, 35, 35)]
        ctl.get_drawn_shapes = lambda: boxes2
        ctl.extract_patches()
        _drive_extract(ctl, app)

        # THE FIX-1 CHECK: the in-memory STAGE was RESET (not cumulative) — it
        # holds ONLY run-2 patches (start index advanced past run 1's registry
        # rows → _3 / _4).  Without the reset the stage would be {_1,_2,_3,_4}.
        assert set(ctl.extracted_patches) == {"kidney_patch_3", "kidney_patch_4"}

        # …and therefore NO fabricated failure: FINISH reports 2 new / 0 failed
        # off the reset stage (the bug computes `n_staged(4) - n_written(2)` and
        # reports "2 new … 2 failed to write" with a red status).
        msg, ok = ctl.widget.status[-1]
        assert ok is True
        assert "2 new patch" in msg
        assert "failed to write" not in msg

        # ④'s Library list is REGISTRY-backed (F2 — `_refresh_library_from_registry`
        # repopulates it from all persisted patches), so it correctly shows all 4
        # saved patches across both runs — that is NOT the stage, and NOT affected
        # by the FIX-1 bug.
        rows = {ctl.widget.extracted_roi_list.item(i).text()
                for i in range(ctl.widget.extracted_roi_list.count())}
        assert rows == {"kidney_patch_1", "kidney_patch_2",
                        "kidney_patch_3", "kidney_patch_4"}


# ---------------------------------------------------------------------------
# ③ box-clear on extract — saved boxes leave the box tool; kept boxes stay
# ---------------------------------------------------------------------------

class TestClearExtractedBoxesFromTool:
    """After a successful extract, the SAVED boxes (XY signature now committed)
    are removed from ③'s box tool; un-ticked / persist-failed boxes (sig NOT
    committed) are kept — their bbox lives in ④ Review & Library instead."""

    def _fake_tool(self, rects, removed_out):
        class _FakeBoxTool:
            def has_boxes(self):
                return bool(rects)

            def rectangles(self):
                return list(rects)

            def remove_boxes(self, indices):
                removed_out.append(list(indices))

        return _FakeBoxTool()

    def test_removes_only_committed_boxes(self):
        rect_a = _box(0, 0, 10, 10)     # extracted -> committed -> removed
        rect_b = _box(20, 20, 30, 30)   # un-ticked -> uncommitted -> kept
        ctl = DataPreparationController.__new__(DataPreparationController)
        _seed_session_caches(ctl)
        ctl._committed_box_sigs = {box_signature(rect_a)}
        removed = []
        ctl.box_tool = self._fake_tool([rect_a, rect_b], removed)

        ctl._clear_extracted_boxes_from_tool()
        assert removed == [[0]]         # only box A; box B survives in ③

    def test_shared_xy_footprint_clears_nothing(self):
        # editable-Z multi-slab: A + B share an XY footprint (identical XY sig);
        # only A was extracted.  XY-only matching can't tell the slabs apart, so
        # the uniqueness guard leaves BOTH — the extracted A is not auto-cleared
        # (user removes it), but the kept B is never wrongly removed (audit MED).
        rect_a = _box(0, 0, 10, 10)
        rect_b = _box(0, 0, 10, 10)     # same XY, a different Z-slab in real use
        ctl = DataPreparationController.__new__(DataPreparationController)
        _seed_session_caches(ctl)
        ctl._committed_box_sigs = {box_signature(rect_a)}
        removed = []
        ctl.box_tool = self._fake_tool([rect_a, rect_b], removed)

        ctl._clear_extracted_boxes_from_tool()
        assert removed == []            # shared XY -> nothing auto-cleared

    def test_noop_when_nothing_committed(self):
        ctl = DataPreparationController.__new__(DataPreparationController)
        _seed_session_caches(ctl)
        ctl._committed_box_sigs = set()
        removed = []
        ctl.box_tool = self._fake_tool([_box(0, 0, 10, 10)], removed)

        ctl._clear_extracted_boxes_from_tool()
        assert removed == []            # remove_boxes never called

    def test_noop_when_no_box_tool(self):
        ctl = DataPreparationController.__new__(DataPreparationController)
        _seed_session_caches(ctl)
        ctl.box_tool = None
        ctl._committed_box_sigs = set()
        ctl._clear_extracted_boxes_from_tool()   # must not raise


# ---------------------------------------------------------------------------
# ④ patch-delete — bin removes the TIFF + sidecar + registry row + stage + sig
# ---------------------------------------------------------------------------

class TestDeletePatch:
    def _extract_one(self, app, tmp_path, tag):
        arr = np.random.randint(0, 60000, (2, 8, 40, 40)).astype(np.uint16)
        path = _write(tmp_path, f"src_{tag}.tif", arr, "deflate")
        out = tmp_path / f"ws_{tag}"
        out.mkdir()
        ctl = _extract_controller(
            app, open_lazy(path), "CZYX",
            {"s_ch0": 0, "s_ch1": 1},
            {"s_ch0": _FakeLayer(arr[0]), "s_ch1": _FakeLayer(arr[1])},
            ["s_ch0", "s_ch1"], str(out))
        box = _box(2, 3, 30, 33)
        ctl.get_drawn_shapes = lambda: [box]
        ctl.extract_patches()
        _drive_extract(ctl, app)
        return ctl, str(out), box

    def test_delete_removes_file_registry_stage_and_sig(
            self, app, tmp_path, monkeypatch):
        from qtpy.QtWidgets import QMessageBox
        from napari_epicycle.data_preparation._paths import step1_patches_dir
        from napari_epicycle.data_preparation._state import load_state

        ctl, ws, box = self._extract_one(app, tmp_path, "del")
        name = "kidney_patch_1"
        pdir = step1_patches_dir(ws)
        tiff, side = pdir / f"{name}.tiff", pdir / f"{name}_metadata.json"
        assert tiff.exists() and side.exists()
        assert name in ctl.extracted_patches
        assert box_signature(box.tolist()) in ctl._committed_box_sigs

        monkeypatch.setattr(QMessageBox, "question",
                            lambda *a, **k: QMessageBox.Yes)   # confirm
        ctl._on_delete_patch(name)

        assert not tiff.exists() and not side.exists()   # files gone
        state = load_state(ws)
        assert name not in [p["patch_name"] for p in state["patches"]]  # row gone
        assert name not in ctl.extracted_patches                        # stage gone
        assert box_signature(box.tolist()) not in ctl._committed_box_sigs  # sig freed

    def test_delete_declined_keeps_everything(self, app, tmp_path, monkeypatch):
        from qtpy.QtWidgets import QMessageBox
        from napari_epicycle.data_preparation._paths import step1_patches_dir

        ctl, ws, _box_used = self._extract_one(app, tmp_path, "del2")
        name = "kidney_patch_1"
        monkeypatch.setattr(QMessageBox, "question",
                            lambda *a, **k: QMessageBox.No)    # decline
        ctl._on_delete_patch(name)

        assert (step1_patches_dir(ws) / f"{name}.tiff").exists()   # nothing removed
        assert name in ctl.extracted_patches


# ---------------------------------------------------------------------------
# F1 — partial write failure (MED-4 + the n_failed path): a write-failed box
# stays staged for retry, its sig is NOT committed, FINISH counts it failed
# ---------------------------------------------------------------------------

class TestExtractPersistFailure:
    def test_write_failed_box_stays_staged_and_counts_failed(
            self, app, tmp_path, monkeypatch):
        import napari_epicycle.data_preparation.controller as C

        arr = np.random.randint(0, 60000, (2, 8, 40, 40)).astype(np.uint16)
        path = _write(tmp_path, "pf.tif", arr, "deflate")
        out = tmp_path / "ws_pf"
        out.mkdir()
        ctl = _extract_controller(
            app, open_lazy(path), "CZYX",
            {"s_ch0": 0, "s_ch1": 1},
            {"s_ch0": _FakeLayer(arr[0]), "s_ch1": _FakeLayer(arr[1])},
            ["s_ch0", "s_ch1"], str(out))
        box = _box(2, 3, 30, 33)
        ctl.get_drawn_shapes = lambda: [box]

        # The pure writer signals failure by RAISING (it never returns None);
        # the worker's per-patch guard must catch it, keep the patch staged, and
        # NOT commit its sig.
        def _boom(*a, **k):
            raise RuntimeError("disk full")
        monkeypatch.setattr(C, "persist_patch", _boom)

        ctl.extract_patches()
        _drive_extract(ctl, app)

        # The write-failed box is STILL staged (for retry) + still in the list…
        assert set(ctl.extracted_patches) == {"kidney_patch_1"}
        rows = {ctl.widget.extracted_roi_list.item(i).text()
                for i in range(ctl.widget.extracted_roi_list.count())}
        assert rows == {"kidney_patch_1"}
        # …its box signature was NOT committed (so a retry re-extracts it)…
        assert box_signature(box.tolist()) not in ctl._committed_box_sigs
        # …and FINISH reports the failure honestly (not a green success).
        msg, ok = ctl.widget.status[-1]
        assert ok is False
        assert "1 failed to write" in msg


# ---------------------------------------------------------------------------
# F1 — HIGH-1 single-writer seam (FIX 2): PREP flushes + stops the timer before
# launch; on_step_exit DEFERS its flush while a run is in flight; FINISH re-runs
# ---------------------------------------------------------------------------

class TestExtractFlushGuard:
    def test_prep_flushes_and_stops_timer_before_launch(self, app, tmp_path):
        arr = np.random.randint(0, 60000, (2, 8, 40, 40)).astype(np.uint16)
        path = _write(tmp_path, "fg.tif", arr, "deflate")
        out = tmp_path / "ws_fg"
        out.mkdir()
        ctl = _extract_controller(
            app, open_lazy(path), "CZYX",
            {"s_ch0": 0, "s_ch1": 1},
            {"s_ch0": _FakeLayer(arr[0]), "s_ch1": _FakeLayer(arr[1])},
            ["s_ch0", "s_ch1"], str(out))
        # Replace the no-op stubs with SPIES on the single-writer seam.
        flush_calls = []
        ctl._flush_source_state_save = lambda *a: flush_calls.append(1)
        stop_calls = []
        ctl._state_save_timer = SimpleNamespace(
            stop=lambda: stop_calls.append(1))

        ctl.get_drawn_shapes = lambda: [_box(2, 3, 30, 33)]
        ctl.extract_patches()
        # PREP flushed + stopped the timer BEFORE launching (worker becomes the
        # sole writer), exactly once each — FINISH's re-flush hasn't run yet.
        assert flush_calls == [1]
        assert stop_calls == [1]

        _drive_extract(ctl, app)
        # FINISH re-ran the deferred flush (FIX 2b) → one more call.
        assert flush_calls == [1, 1]

    def test_step_exit_defers_flush_while_extract_running(self, app, tmp_path):
        out = tmp_path / "ws_fg2"
        out.mkdir()
        ctl = _extract_controller(app, None, None, {}, {}, [], str(out))
        flush_calls = []
        ctl._flush_source_state_save = lambda *a: flush_calls.append(1)

        # A run is in flight (no real thread needed — on_step_exit's teardowns
        # are all getattr/try-guarded, so it runs through on the bare double).
        ctl._extract_running = True
        ctl.on_step_exit()
        assert flush_calls == []            # deferred while in flight (FIX 2a)

        # The guard is the discriminator: with no run, step-exit flushes as before.
        ctl._extract_running = False
        ctl.on_step_exit()
        assert flush_calls == [1]


# ---------------------------------------------------------------------------
# ④ Z-crop TRIM (crop-in-place) — end-to-end flow on the async _TrimWorker
# ---------------------------------------------------------------------------

def _seed_trim_state(ctl):
    """Seed the trim-worker state the bare `__new__` controller needs (its
    __init__ normally does this)."""
    ctl._trim_token = 0
    ctl._trim_running = False
    ctl._trim_launching = False
    ctl._trim_thread = None
    ctl._trim_worker = None
    ctl._draining_trim_threads = []
    ctl._left_step1 = False
    ctl._plot_n_z = None
    ctl._plot_n_z_patch = None
    ctl.selected_z_range = None
    ctl._visible_patch_locations = set()
    ctl._intensity_worker = None


def _drive_trim(ctl, app, timeout_s=15.0):
    """Pump the main-thread loop until the async ④ Trim worker finishes — the
    finished/error slots are queued cross-thread connections delivered by
    `processEvents` (mirrors `_drive_extract`)."""
    from qtpy.QtWidgets import QApplication
    deadline = time.time() + timeout_s
    while getattr(ctl, "_trim_running", False):
        QApplication.processEvents()
        if time.time() > deadline:
            raise AssertionError("trim worker did not finish within timeout")
        time.sleep(0.003)
    th = getattr(ctl, "_trim_thread", None)
    if th is not None:
        try:
            th.wait(3000)
        except Exception:
            pass
    for _ in range(10):
        QApplication.processEvents()


class TestTrimFlow:
    def _extract_one(self, app, tmp_path, tag):
        arr = np.random.randint(0, 60000, (2, 8, 40, 40)).astype(np.uint16)
        path = _write(tmp_path, f"src_{tag}.tif", arr, "deflate")
        out = tmp_path / f"ws_{tag}"
        out.mkdir()
        ctl = _extract_controller(
            app, open_lazy(path), "CZYX",
            {"s_ch0": 0, "s_ch1": 1},
            {"s_ch0": _FakeLayer(arr[0]), "s_ch1": _FakeLayer(arr[1])},
            ["s_ch0", "s_ch1"], str(out))
        _seed_trim_state(ctl)
        box = _box(2, 3, 30, 33)
        ctl.get_drawn_shapes = lambda: [box]
        ctl.extract_patches()
        _drive_extract(ctl, app)
        # Select the saved patch in ④ (the trim reads `currentItem()`); the
        # registry-backed library refresh at extract-finish clears the row.
        lst = ctl.widget.extracted_roi_list
        for i in range(lst.count()):
            if lst.item(i).text() == "kidney_patch_1":
                lst.setCurrentRow(i)
                break
        return ctl, str(out), arr, box

    def test_trim_shrinks_disk_and_stage_absolute_range(
            self, app, tmp_path, monkeypatch):
        import json
        from qtpy.QtWidgets import QMessageBox
        from napari_epicycle.data_preparation._paths import step1_patches_dir
        from napari_epicycle.data_preparation.logic import (
            extract_patches_from_shapes,
        )

        ctl, ws, arr, box = self._extract_one(app, tmp_path, "trim")
        name = "kidney_patch_1"
        pdir = step1_patches_dir(ws)
        tiff = pdir / f"{name}.tiff"
        side = pdir / f"{name}_metadata.json"

        # The extracted patch (byte-identical reference for the crop).
        full_patch = list(
            extract_patches_from_shapes(arr, [box]).values())[0]
        _, fz, fy, fx = full_patch.shape
        assert fz == 8

        # Pre-trim: full-Z on disk AND in the in-memory stage.
        assert tifffile.imread(str(tiff), squeeze=False).shape == full_patch.shape
        assert ctl.extracted_patches[name]["data"].shape == full_patch.shape

        # Choose Z [2, 6] (current-patch coords), confirm Yes, trim.
        ctl.selected_z_range = (2, 6)
        monkeypatch.setattr(QMessageBox, "question",
                            lambda *a, **k: QMessageBox.Yes)
        ctl._on_set_zrange_clicked()
        _drive_trim(ctl, app)

        # Disk TIFF shrank in Z (8 -> 4), byte-identical to a manual crop.
        disk = tifffile.imread(str(tiff), squeeze=False)
        assert disk.shape == (2, 4, fy, fx)
        assert np.array_equal(disk, full_patch[:, 2:6])
        # Sidecar updated: new cropped_shape + ABSOLUTE selected_z_range
        # (prev_z0 was 0 -> [2, 6]); provenance unchanged.
        meta = json.loads(side.read_text())
        assert meta["cropped_shape"] == [2, 4, fy, fx]
        assert meta["selected_z_range"] == [2, 6]
        assert meta["original_shape"] == [2, fz, fy, fx]
        assert meta["patch_name"] == name
        # In-memory stage cropped BY NAME + selected_z_range absolute.
        assert ctl.extracted_patches[name]["data"].shape == (2, 4, fy, fx)
        assert np.array_equal(
            ctl.extracted_patches[name]["data"], full_patch[:, 2:6])
        assert ctl.extracted_patches[name]["selected_z_range"] == [2, 6]
        # Surface unlocked; no .bak left behind.
        assert ctl._trim_running is False
        assert ctl._step1_busy is False
        assert not (pdir / f"{name}.tiff.bak").exists()

        # A green status was written.
        msg, ok = ctl.widget.status[-1]
        assert ok is True
        assert "Trimmed" in msg and "2" in msg

    def test_trim_declined_changes_nothing(self, app, tmp_path, monkeypatch):
        import json
        from qtpy.QtWidgets import QMessageBox
        from napari_epicycle.data_preparation._paths import step1_patches_dir

        ctl, ws, arr, box = self._extract_one(app, tmp_path, "trimno")
        name = "kidney_patch_1"
        pdir = step1_patches_dir(ws)
        tiff = pdir / f"{name}.tiff"
        side = pdir / f"{name}_metadata.json"
        before_meta = json.loads(side.read_text())

        before_disk = tifffile.imread(str(tiff), squeeze=False)
        before_stage = ctl.extracted_patches[name]["data"].shape

        ctl.selected_z_range = (2, 6)
        monkeypatch.setattr(QMessageBox, "question",
                            lambda *a, **k: QMessageBox.No)     # decline
        ctl._on_set_zrange_clicked()

        # Nothing launched, nothing changed on disk or in the stage.
        assert ctl._trim_running is False
        assert ctl._trim_thread is None
        assert tifffile.imread(str(tiff), squeeze=False).shape == before_disk.shape
        assert ctl.extracted_patches[name]["data"].shape == before_stage
        assert json.loads(side.read_text()) == before_meta

    def test_trim_without_saved_tiff_is_a_clear_noop(self, app, tmp_path):
        out = tmp_path / "ws_notiff"
        out.mkdir()
        ctl = _extract_controller(app, None, None, {}, {}, [], str(out))
        _seed_trim_state(ctl)
        ctl.widget.extracted_roi_list.addItem("ghost")
        ctl.widget.extracted_roi_list.setCurrentRow(0)
        ctl.selected_z_range = (1, 3)
        ctl._on_set_zrange_clicked()            # no on-disk TIFF -> clean message
        assert ctl._trim_running is False
        msg, ok = ctl.widget.status[-1]
        assert ok is False
        assert "no saved patch file" in msg

    def test_hard_guard_blocks_reentry_while_running(
            self, app, tmp_path, monkeypatch):
        from qtpy.QtWidgets import QMessageBox
        out = tmp_path / "ws_guard"
        out.mkdir()
        ctl = _extract_controller(app, None, None, {}, {}, [], str(out))
        _seed_trim_state(ctl)
        ctl._trim_running = True                 # a trim is already in flight
        called = []
        monkeypatch.setattr(
            QMessageBox, "question",
            lambda *a, **k: (called.append(1), QMessageBox.Yes)[1])
        ctl.widget.extracted_roi_list.addItem("kidney_patch_1")
        ctl.widget.extracted_roi_list.setCurrentRow(0)
        ctl.selected_z_range = (2, 6)
        ctl._on_set_zrange_clicked()             # blocked BEFORE the confirm
        assert called == []

    def test_stale_trim_finished_is_a_noop(self, app, tmp_path):
        out = tmp_path / "ws_stale_t"
        out.mkdir()
        ctl = _extract_controller(app, None, None, {}, {}, [], str(out))
        _seed_trim_state(ctl)
        ctl._trim_token = 7
        ctl._trim_running = True
        ctl.extracted_patches["p"] = {
            "data": np.zeros((2, 8, 4, 4), dtype=np.uint16),
            "channels": ["a"], "bounding_box": [], "source_kind": "raw"}
        info = {"name": "p", "z0": 2, "z1": 6,
                "selected_z_range": [2, 6], "cropped_shape": [2, 4, 4, 4]}

        ctl._on_trim_finished(info, token=6)     # stale — inert
        assert ctl.extracted_patches["p"]["data"].shape == (2, 8, 4, 4)  # NOT cropped
        assert ctl._trim_running is True                                 # NOT unlocked

    def test_finish_after_step_exit_crops_memory_but_ui_refresh_inert(
            self, app, tmp_path):
        """MED-1 (data audit): a trim that finishes AFTER step-exit still crops
        the in-memory stage (pure data, no Qt) so the staged copy can't diverge
        from the trimmed disk + mislead a later trim into cutting the wrong slab
        — but the ④/canvas REFRESH is inert (disk already trimmed)."""
        out = tmp_path / "ws_exit_t"
        out.mkdir()
        ctl = _extract_controller(app, None, None, {}, {}, [], str(out))
        _seed_trim_state(ctl)
        ctl._trim_token = 3
        ctl._trim_running = True
        ctl._left_step1 = True                   # step already exited
        ctl.extracted_patches["p"] = {
            "data": np.zeros((2, 8, 4, 4), dtype=np.uint16),
            "channels": ["a"], "bounding_box": [], "source_kind": "raw"}
        refreshed = {"n": 0}
        ctl.widget.library_box_list = SimpleNamespace(
            refresh=lambda: refreshed.__setitem__("n", refreshed["n"] + 1))
        info = {"name": "p", "z0": 2, "z1": 6,
                "selected_z_range": [2, 6], "cropped_shape": [2, 4, 4, 4]}

        ctl._on_trim_finished(info, token=3)     # current token, but step LEFT
        assert ctl._trim_running is False         # the one unlock STILL ran
        # MED-1: the in-memory crop RAN even after step-exit (stage == disk).
        assert ctl.extracted_patches["p"]["data"].shape == (2, 4, 4, 4)
        assert ctl.extracted_patches["p"]["selected_z_range"] == [2, 6]
        # …but the ④ REFRESH (library rebuild) was inert.
        assert refreshed["n"] == 0

    def test_shutdown_sweeps_trim_family(self, app, tmp_path):
        """Threading-audit HIGH-1: the hard-close sweep MUST join + null the trim
        thread family + bump `_trim_token`, so closing napari mid-trim never
        destroys a running QThread and a late finish is inert."""
        out = tmp_path / "ws_shut_t"
        out.mkdir()
        ctl = _extract_controller(app, None, None, {}, {}, [], str(out))
        _seed_trim_state(ctl)
        calls = {"quit": 0}
        fake_thread = SimpleNamespace(
            quit=lambda: calls.__setitem__("quit", calls["quit"] + 1),
            wait=lambda ms=0: None,
            isRunning=lambda: False)
        ctl._trim_thread = fake_thread
        ctl._trim_worker = None                  # _TrimWorker has no request_cancel
        ctl._trim_running = True
        ctl._trim_token = 3

        ctl.shutdown()

        assert ctl._trim_token == 4              # bumped → any late finish inert
        assert ctl._trim_thread is None
        assert ctl._trim_running is False
        assert ctl._draining_trim_threads == []
        assert calls["quit"] == 1                # the in-flight thread was joined
        # A stale finish delivered AFTER shutdown is a no-op (pre-shutdown token).
        ctl.extracted_patches["p"] = {
            "data": np.zeros((2, 8, 4, 4), dtype=np.uint16)}
        ctl._on_trim_finished(
            {"name": "p", "z0": 2, "z1": 6, "selected_z_range": [2, 6]}, token=3)
        assert ctl.extracted_patches["p"]["data"].shape == (2, 8, 4, 4)

    def test_capture_blocked_while_trim_running(self, app, tmp_path):
        """Threading-audit MED-1: opening the capture inspector while a trim is
        overwriting the same TIFF must be blocked BEFORE the path resolves (else
        `_PatchLoadWorker` holds the file open across `os.replace`)."""
        out = tmp_path / "ws_cap_block"
        out.mkdir()
        ctl = _extract_controller(app, None, None, {}, {}, [], str(out))
        _seed_trim_state(ctl)
        ctl._trim_running = True
        ctl._set_export_status = lambda *a, **k: None
        resolved = {"n": 0}
        ctl._patch_disk_path = lambda name: resolved.__setitem__(
            "n", resolved["n"] + 1) or None
        ctl._on_capture_patch_clicked("p")
        assert resolved["n"] == 0                # guarded out before path resolve

    def test_delete_blocked_while_trim_running(
            self, app, tmp_path, monkeypatch):
        """Threading-audit MED-1: deleting a patch (os.remove) while a trim is
        replacing it must be blocked BEFORE the confirm modal."""
        out = tmp_path / "ws_del_block"
        out.mkdir()
        ctl = _extract_controller(app, None, None, {}, {}, [], str(out))
        _seed_trim_state(ctl)
        ctl._trim_running = True
        ctl._set_export_status = lambda *a, **k: None
        from qtpy.QtWidgets import QMessageBox
        shown = {"n": 0}
        monkeypatch.setattr(
            QMessageBox, "question",
            lambda *a, **k: shown.__setitem__("n", shown["n"] + 1) or
            QMessageBox.No)
        ctl._on_delete_patch("p")
        assert shown["n"] == 0                    # guarded out before the confirm


class TestReviewLibraryUX:
    """④ honest-UI wins: the ③→④ auto-open bridge + the depth-plot empty state."""

    def test_extract_finish_opens_review_library_when_written(
            self, app, tmp_path):
        out = tmp_path / "ws_bridge"
        out.mkdir()
        ctl = _extract_controller(app, None, None, {}, {}, [], str(out))
        # Isolate the finish to the bridge decision.
        ctl._refresh_source_patch_counts = lambda: None
        ctl._refresh_library_from_registry = lambda: None
        ctl._clear_extracted_boxes_from_tool = lambda: None
        ctl._refresh_conditioning_status = lambda: None
        ctl._finish_extract_unlock = lambda: None
        ctl._extract_token = 5
        opened = {"n": 0}
        ctl.widget.open_review_library_section = lambda: opened.__setitem__(
            "n", opened["n"] + 1)

        ctl._on_extract_finished(2, False, token=5)   # wrote 2 → bridge opens ④
        assert opened["n"] == 1
        ctl._on_extract_finished(0, False, token=5)   # wrote 0 → no jump
        assert opened["n"] == 1

    def test_deselect_clears_depth_profile(self, app, tmp_path):
        out = tmp_path / "ws_deselect"
        out.mkdir()
        ctl = _extract_controller(app, None, None, {}, {}, [], str(out))
        ctl._refresh_conditioning_status = lambda: None
        cleared = {"n": 0}
        plotted = {"n": 0}
        ctl._clear_intensity_profile = lambda *a, **k: cleared.__setitem__(
            "n", cleared["n"] + 1)
        ctl._on_plot_intensity_clicked = lambda: plotted.__setitem__(
            "n", plotted["n"] + 1)

        # Empty list → currentItem() is None → clear the plot, do NOT plot.
        ctl._on_roi_selection_changed()
        assert cleared["n"] == 1
        assert plotted["n"] == 0

    def test_pending_intensity_load_coalesces_to_current(self, app, tmp_path):
        """audit MED: a load queued while another ran re-triggers for the CURRENT
        selection when that load finishes (so it never sticks on 'Loading…') —
        but only if the queued patch is still selected."""
        out = tmp_path / "ws_coalesce"
        out.mkdir()
        ctl = _extract_controller(app, None, None, {}, {}, [], str(out))
        triggered = {"n": 0}
        ctl._on_plot_intensity_clicked = lambda: triggered.__setitem__(
            "n", triggered["n"] + 1)
        ctl.widget.extracted_roi_list.addItem("pB")
        ctl.widget.extracted_roi_list.setCurrentRow(0)

        # pending == current → re-trigger + clear the latch.
        ctl._intensity_pending_patch = "pB"
        ctl._on_intensity_worker_finished()
        assert triggered["n"] == 1
        assert ctl._intensity_pending_patch is None
        # latch cleared → a second finish is a no-op.
        ctl._on_intensity_worker_finished()
        assert triggered["n"] == 1
        # pending for a DIFFERENT patch than current → cleared, NOT triggered.
        ctl._intensity_pending_patch = "pOther"
        ctl._on_intensity_worker_finished()
        assert triggered["n"] == 1
        assert ctl._intensity_pending_patch is None

    def test_patch_select_resets_z_band(self, app, tmp_path):
        """Band-reset: selecting a patch clears the carried Z-truncate range so a
        Trim reflects the NEW patch, not the previously-selected one (the range
        used to carry over → Trim would cut the wrong slab)."""
        out = tmp_path / "ws_zreset"
        out.mkdir()
        ctl = _extract_controller(app, None, None, {}, {}, [], str(out))
        ctl._refresh_conditioning_status = lambda: None
        ctl._on_plot_intensity_clicked = lambda: None   # isolate the reset
        ctl.widget.extracted_roi_list.addItem("p_new")
        ctl.widget.extracted_roi_list.setCurrentRow(0)
        ctl.selected_z_range = (33, 115)                # carried from a prior patch
        ctl._on_roi_selection_changed()
        assert ctl.selected_z_range is None             # reset for the new patch

    def test_intensity_ready_caches_snap_range(self, app, tmp_path):
        """The depth-profile load caches the off-thread-detected sample range so
        ④'s 'snap' works for a REOPENED patch (no in-memory pixels)."""
        out = tmp_path / "ws_snap_cache"
        out.mkdir()
        ctl = _extract_controller(app, None, None, {}, {}, [], str(out))
        ctl._render_intensity_profile = lambda *a, **k: None
        ctl.widget.extracted_roi_list.addItem("p")
        ctl.widget.extracted_roi_list.setCurrentRow(0)
        ctl._on_intensity_profile_ready("p", [], 10, (12, 62))
        assert ctl._snap_range == ("p", (12, 62))

    def test_snap_applies_cached_range_for_reopened_patch(self, app, tmp_path):
        """Snap FIX: a reopened patch snaps to the cached sample range (was a
        silent no-op before — it only looked at the in-memory stage)."""
        out = tmp_path / "ws_snap"
        out.mkdir()
        ctl = _extract_controller(app, None, None, {}, {}, [], str(out))
        ctl._refresh_conditioning_status = lambda: None
        ctl._zrange_region = None
        ctl.widget.extracted_roi_list.addItem("p_disk")   # NOT in extracted_patches
        ctl.widget.extracted_roi_list.setCurrentRow(0)
        ctl._snap_range = ("p_disk", (12, 62))            # cached by the plot load
        applied = {}
        ctl._set_selected_z_range = lambda zr: applied.__setitem__(
            "zr", tuple(zr))
        ctl._on_autoselect_zrange_clicked()
        assert applied.get("zr") == (12, 62)

    def test_snap_reopened_not_ready_asks_retry(self, app, tmp_path):
        """Snap before the plot load finished → graceful retry, nothing applied."""
        out = tmp_path / "ws_snap2"
        out.mkdir()
        ctl = _extract_controller(app, None, None, {}, {}, [], str(out))
        ctl._refresh_conditioning_status = lambda: None
        status = []
        ctl._set_export_status = lambda m, success=True: status.append(
            (m, success))
        ctl.widget.extracted_roi_list.addItem("p_disk")
        ctl.widget.extracted_roi_list.setCurrentRow(0)
        ctl._snap_range = None                            # load not finished
        applied = {}
        ctl._set_selected_z_range = lambda zr: applied.__setitem__(
            "zr", tuple(zr))
        ctl._on_autoselect_zrange_clicked()
        assert "zr" not in applied                        # nothing snapped
        assert status and status[-1][1] is False          # a retry message shown
