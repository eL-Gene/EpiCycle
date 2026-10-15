"""Controller-level tests for the L0 lazy-crop extraction + axis-detection wiring.

The invariant that matters most (and "Analysis Data
Integrity"): extracting through the LAZY full-res handle — where only the ROI is
read off disk — must be **byte-identical** to the old path that stacked the
materialized display layers.  These tests build the controller's crop source
(`_crop_source_for`) from a lazy dask/memmap handle and prove the extracted
patch equals the eager reference for CZYX, ZCYX (reordered), 3D single-channel,
a channel SUBSET, and a channel REORDER.

The `_crop_source_for` tests are Qt-free (they only touch the controller's L0
state).  The `apply_axis_order` / override tests need the real widget, so they
carry a local QApplication fixture mirroring `test_ui.py`.
"""
from __future__ import annotations

import numpy as np
import pytest
import tifffile

from napari_epicycle.data_preparation.controller import DataPreparationController
from napari_epicycle.data_preparation.logic import (
    extract_patches_from_shapes,
    transpose_to_czyx,
)
from napari_epicycle.data_preparation._largedata import open_lazy, resolve_axes


def _box(y0, x0, y1, x1):
    return np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]], dtype=float)


def _bare_controller(**l0_state):
    """A controller with ONLY the L0 crop state set (no widget / viewer).

    `_crop_source_for` reads `_source_lazy`, `_source_axes_resolved`,
    `_channel_index_by_name` — nothing else — so this is enough to exercise it.
    """
    ctl = DataPreparationController.__new__(DataPreparationController)
    ctl._source_lazy = l0_state.get("lazy")
    ctl._source_axes_resolved = l0_state.get("order")
    ctl._channel_index_by_name = l0_state.get("index_map", {})
    return ctl


def _write(tmp_path, name, arr, compression=None):
    p = tmp_path / name
    tifffile.imwrite(str(p), arr, compression=compression)
    return str(p)


def _write_meta(tmp_path, name, arr, axes, compression=None):
    """Write a TIFF carrying an explicit clean ``axes`` string in its shaped
    metadata (tifffile reads it back as ``series.axes``)."""
    p = tmp_path / name
    tifffile.imwrite(str(p), arr, compression=compression,
                     metadata={"axes": axes})
    return str(p)


# ---------------------------------------------------------------------------
# _crop_source_for + extract equivalence — the byte-identical guarantee
# ---------------------------------------------------------------------------

class TestLazyCropEqualsEager:
    BOXES = [_box(2, 3, 30, 33), _box(10, 10, 25, 40)]

    @pytest.mark.parametrize("compression", [None, "deflate"])
    def test_czyx_crop_matches_materialized_stack(self, tmp_path, compression):
        arr = np.random.randint(0, 60000, (2, 8, 40, 40)).astype(np.uint16)
        path = _write(tmp_path, "czyx.tif", arr, compression)
        ctl = _bare_controller(
            lazy=open_lazy(path), order="CZYX",
            index_map={"s_ch0": 0, "s_ch1": 1})

        src, idx = ctl._crop_source_for(["s_ch0", "s_ch1"])
        assert idx == [0, 1]
        lazy_patches = extract_patches_from_shapes(
            src, self.BOXES, channel_indices=idx)
        eager_patches = extract_patches_from_shapes(arr, self.BOXES)

        assert list(lazy_patches) == list(eager_patches)
        for k in lazy_patches:
            assert np.array_equal(lazy_patches[k], eager_patches[k]), k

    def test_zcyx_source_is_reordered_to_czyx(self, tmp_path):
        """A ZCYX on-disk stack must extract identically to its CZYX transpose —
        the lazy handle is moveaxis'd the same way the display was."""
        zcyx = np.random.randint(0, 60000, (8, 2, 40, 40)).astype(np.uint16)
        path = _write(tmp_path, "zcyx.tif", zcyx, "deflate")
        ctl = _bare_controller(
            lazy=open_lazy(path), order="ZCYX",
            index_map={"s_ch0": 0, "s_ch1": 1})

        src, idx = ctl._crop_source_for(["s_ch0", "s_ch1"])
        lazy_patches = extract_patches_from_shapes(
            src, self.BOXES, channel_indices=idx)
        # Eager reference: the same transpose the loader applies for ZCYX.
        eager_patches = extract_patches_from_shapes(
            transpose_to_czyx(zcyx), self.BOXES)

        for k in lazy_patches:
            assert np.array_equal(lazy_patches[k], eager_patches[k]), k

    def test_3d_single_channel_matches_stacked_shape(self, tmp_path):
        """A 3D source stacks to (1, Z, Y, X) in the old path; the lazy handle
        must produce the same (1, Z, h, w) patch, not a bare (Z, h, w)."""
        arr = np.random.randint(0, 60000, (8, 40, 40)).astype(np.uint16)
        path = _write(tmp_path, "zyx.tif", arr, "deflate")
        ctl = _bare_controller(
            lazy=open_lazy(path), order="ZYX", index_map={"mono": 0})

        src, idx = ctl._crop_source_for(["mono"])
        assert idx == [0]
        lazy_patches = extract_patches_from_shapes(
            src, self.BOXES, channel_indices=idx)
        eager_patches = extract_patches_from_shapes(arr[None], self.BOXES)

        for k in lazy_patches:
            assert lazy_patches[k].shape[0] == 1
            assert np.array_equal(lazy_patches[k], eager_patches[k]), k

    def test_channel_subset_selection(self, tmp_path):
        arr = np.random.randint(0, 60000, (3, 6, 32, 32)).astype(np.uint16)
        path = _write(tmp_path, "sub.tif", arr, "deflate")
        ctl = _bare_controller(
            lazy=open_lazy(path), order="CZYX",
            index_map={"s_ch0": 0, "s_ch1": 1, "s_ch2": 2})

        src, idx = ctl._crop_source_for(["s_ch1"])
        assert idx == [1]
        got = extract_patches_from_shapes(src, [self.BOXES[0]], channel_indices=idx)
        want = extract_patches_from_shapes(arr[[1]], [self.BOXES[0]])
        for k in got:
            assert got[k].shape[0] == 1
            assert np.array_equal(got[k], want[k])

    def test_channel_reorder_is_preserved(self, tmp_path):
        arr = np.random.randint(0, 60000, (2, 6, 32, 32)).astype(np.uint16)
        path = _write(tmp_path, "reorder.tif", arr, "deflate")
        ctl = _bare_controller(
            lazy=open_lazy(path), order="CZYX",
            index_map={"s_ch0": 0, "s_ch1": 1})

        src, idx = ctl._crop_source_for(["s_ch1", "s_ch0"])  # reversed
        assert idx == [1, 0]
        got = extract_patches_from_shapes(src, [self.BOXES[0]], channel_indices=idx)
        want = extract_patches_from_shapes(arr[[1, 0]], [self.BOXES[0]])
        for k in got:
            assert np.array_equal(got[k], want[k])

    def test_clean_czyx_c_gt_z_extracts_correct_pixels_not_transposed(self, tmp_path):
        """Clean ``axes='CZYX'`` with C>Z (10 channels, 5 Z): the lazy crop must
        extract the CORRECTLY-interpreted (CZYX) pixels.  The old
        ``shape[0] > shape[1]`` rule would have transposed C/Z and cropped
        garbage — this locks in the bugfix (audit Fix 3b)."""
        arr = np.random.randint(0, 60000, (10, 5, 64, 64)).astype(np.uint16)
        path = _write_meta(tmp_path, "cgtz.tif", arr, axes="CZYX",
                           compression="deflate")
        with tifffile.TiffFile(path) as tf:
            meta_axes = tf.series[0].axes
        # (a) clean metadata trusted despite C>Z (shape alone would guess ZCYX)
        r = resolve_axes(arr.shape, meta_axes)
        assert r.axes == "CZYX" and r.confident is True and r.channel_axis == 0

        # (b) lazy crop == crop of the CORRECTLY-interpreted (CZYX) volume, and
        #     is NOT the transposed-to-garbage crop the old rule would produce.
        ctl = _bare_controller(
            lazy=open_lazy(path), order="CZYX",
            index_map={f"s_ch{i}": i for i in range(10)})
        names = [f"s_ch{i}" for i in range(10)]
        src, idx = ctl._crop_source_for(names)
        assert idx == list(range(10))

        lazy_patches = extract_patches_from_shapes(
            src, [self.BOXES[0]], channel_indices=idx)
        correct = extract_patches_from_shapes(arr, [self.BOXES[0]])          # CZYX
        garbage = extract_patches_from_shapes(
            transpose_to_czyx(arr), [self.BOXES[0]])                         # old rule
        for k in lazy_patches:
            assert lazy_patches[k].shape[0] == 10          # 10 channels, not 5
            assert np.array_equal(lazy_patches[k], correct[k]), k
            assert not np.array_equal(lazy_patches[k], garbage[k]), k


class TestCropSourceFallback:
    def test_no_lazy_handle_returns_none(self):
        ctl = _bare_controller(lazy=None)
        assert ctl._crop_source_for(["s_ch0"]) == (None, None)

    def test_unknown_channel_name_falls_back(self, tmp_path):
        arr = np.random.randint(0, 100, (2, 4, 16, 16)).astype(np.uint16)
        path = _write(tmp_path, "u.tif", arr)
        ctl = _bare_controller(
            lazy=open_lazy(path), order="CZYX", index_map={"s_ch0": 0})
        # 's_ch1' is not mapped -> bail to the materialized fallback.
        assert ctl._crop_source_for(["s_ch0", "s_ch1"]) == (None, None)

    def test_empty_selection_returns_none(self, tmp_path):
        arr = np.random.randint(0, 100, (2, 4, 16, 16)).astype(np.uint16)
        path = _write(tmp_path, "e.tif", arr)
        ctl = _bare_controller(
            lazy=open_lazy(path), order="CZYX", index_map={"s_ch0": 0})
        assert ctl._crop_source_for([]) == (None, None)


# ---------------------------------------------------------------------------
# Live channel filter — lazy crop set must equal the eager stack set
# ---------------------------------------------------------------------------

class _FakeLayer:
    def __init__(self, data):
        self.data = data


class _NamedLayerList:
    """Minimal stand-in for napari's LayerList: `name in layers` +
    `layers[name].data`, the two accessors get_combined_image_stack uses."""

    def __init__(self, mapping):
        self._m = dict(mapping)

    def __contains__(self, name):
        return name in self._m

    def __getitem__(self, name):
        return self._m[name]


class _ViewerWithLayers:
    def __init__(self, mapping):
        self.layers = _NamedLayerList(mapping)


class TestCropSourceLiveChannelFilter:
    """A channel ticked but since REMOVED from the viewer must be dropped by the
    lazy crop exactly as the eager `get_combined_image_stack` drops it (it keeps
    a layer only if `name in viewer.layers` and `data.ndim == 3`).  Otherwise the
    lazy patch carries an extra channel the eager path does not — a patch
    channel-count mismatch (audit Fix 2)."""

    BOX = _box(2, 3, 30, 33)

    def test_removed_ticked_channel_matches_eager_set(self, tmp_path):
        arr = np.random.randint(0, 60000, (3, 6, 32, 32)).astype(np.uint16)
        path = _write(tmp_path, "removed.tif", arr, "deflate")
        ctl = _bare_controller(
            lazy=open_lazy(path), order="CZYX",
            index_map={"s_ch0": 0, "s_ch1": 1, "s_ch2": 2})
        # ch1 was removed from the viewer; ch0 + ch2 remain as 3D display layers
        # holding their materialized channel data.
        ctl.viewer = _ViewerWithLayers({
            "s_ch0": _FakeLayer(arr[0]), "s_ch2": _FakeLayer(arr[2])})

        # All three are still ticked (get_combined_image_stack returns every
        # ticked name), but ch1 is gone from the viewer.
        src, idx = ctl._crop_source_for(["s_ch0", "s_ch1", "s_ch2"])
        assert idx == [0, 2]          # ch1 dropped, not [0, 1, 2]

        lazy_patches = extract_patches_from_shapes(
            src, [self.BOX], channel_indices=idx)
        # Eager reference: only the channels still present, stacked in tick order.
        eager_patches = extract_patches_from_shapes(
            np.stack([arr[0], arr[2]], axis=0), [self.BOX])
        for k in lazy_patches:
            assert lazy_patches[k].shape[0] == 2   # matches the eager 2-ch stack
            assert np.array_equal(lazy_patches[k], eager_patches[k]), k

    def test_all_present_channels_pass_through_unchanged(self, tmp_path):
        """Normal case: every ticked channel is a live 3D layer -> the filter is
        a no-op and the full ticked set is extracted (no behaviour change)."""
        arr = np.random.randint(0, 60000, (2, 6, 32, 32)).astype(np.uint16)
        path = _write(tmp_path, "present.tif", arr, "deflate")
        ctl = _bare_controller(
            lazy=open_lazy(path), order="CZYX",
            index_map={"s_ch0": 0, "s_ch1": 1})
        ctl.viewer = _ViewerWithLayers({
            "s_ch0": _FakeLayer(arr[0]), "s_ch1": _FakeLayer(arr[1])})

        _src, idx = ctl._crop_source_for(["s_ch0", "s_ch1"])
        assert idx == [0, 1]


# ---------------------------------------------------------------------------
# Widget axis-order wiring (Qt) — the confirmable editor
# ---------------------------------------------------------------------------

class _FakeEvent:
    def connect(self, *_a, **_k):
        pass


class _FakeLayerEvents:
    def __init__(self):
        self.inserted = _FakeEvent()
        self.removed = _FakeEvent()


class _FakeLayerList(list):
    def __init__(self):
        super().__init__()
        self.events = _FakeLayerEvents()


class _FakeViewer:
    def __init__(self):
        self.layers = _FakeLayerList()
        self.mouse_drag_callbacks = []


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


class TestApplyAxisOrder:
    def test_zcyx_sets_combo_and_transpose_flag(self, widget):
        widget.apply_axis_order("ZCYX", "guessed: ZCYX · 2 ch — confirm")
        assert widget.axis_order_combo.currentText() == "ZCYX"
        assert widget.transpose_button.isChecked()
        assert widget.axis_order_hint.text() == "guessed: ZCYX · 2 ch — confirm"

    def test_czyx_clears_transpose_flag(self, widget):
        widget.apply_axis_order("ZCYX", "x")   # first flip it on
        widget.apply_axis_order("CZYX", "detected: CZYX · 2 ch — confirm")
        assert widget.axis_order_combo.currentText() == "CZYX"
        assert not widget.transpose_button.isChecked()

    def test_zyx_maps_to_czyx_no_transpose(self, widget):
        """The combo has no ZYX entry (frozen); a 3D source is CZYX/no-swap."""
        widget.apply_axis_order("ZYX", "single channel (3D)")
        assert widget.axis_order_combo.currentText() == "CZYX"
        assert not widget.transpose_button.isChecked()
        assert widget.axis_order_hint.text() == "single channel (3D)"

    def test_apply_does_not_record_a_user_override(self, widget):
        """Programmatic detection must NOT look like a user override — only the
        combo's `activated` (user) signal records one."""
        widget.controller._axis_order_overrides = {}
        widget.apply_axis_order("ZCYX", "x")
        assert widget.controller._axis_order_overrides == {}


class TestAxisOrderOverride:
    def test_user_pick_records_override_for_current_source(self, widget):
        from qtpy.QtWidgets import QListWidgetItem
        from qtpy.QtCore import Qt

        from napari_epicycle.data_preparation._state import _norm_path

        item = QListWidgetItem("src")
        item.setData(Qt.UserRole, r"C:/data/src.tif")
        widget.source_file_list.addItem(item)
        widget.source_file_list.setCurrentItem(item)

        widget.controller._axis_order_overrides = {}
        widget.axis_order_combo.setCurrentText("ZCYX")
        widget._on_axis_order_user_picked()   # what `activated` triggers

        assert widget.controller._axis_order_overrides.get(
            _norm_path(r"C:/data/src.tif")) == "ZCYX"

    def test_override_keys_to_combo_described_source_not_loaded(self, widget):
        """Load A, then select the (scanned) row B — detect-on-selection points
        the Order combo at B.  Picking an Order now binds it to B (the file the
        combo DESCRIBES), NOT the loaded file A.  Keying on A would misattribute
        B's chosen Order onto A and silently mis-orient A's store (3-audit HIGH
        2026-08-02, inverting the pre-detect-on-selection behavior)."""
        from qtpy.QtWidgets import QListWidgetItem
        from qtpy.QtCore import Qt

        from napari_epicycle.data_preparation._state import _norm_path

        b = r"C:/data/B.tif"
        # Scanned BEFORE selection so detect-on-selection can point the combo at B.
        widget.controller._scanned_files = [
            {"path": b, "stem": "B", "shape": (2, 4, 8, 8), "axes": None}]
        item_b = QListWidgetItem("B")
        item_b.setData(Qt.UserRole, b)
        widget.source_file_list.addItem(item_b)
        widget.source_file_list.setCurrentItem(item_b)  # -> combo describes B

        widget.controller._loaded_source_file = r"C:/data/A.tif"
        widget.controller._axis_order_overrides = {}
        widget.axis_order_combo.setCurrentText("ZCYX")  # user flips (for B)
        widget._on_axis_order_user_picked()

        overrides = widget.controller._axis_order_overrides
        assert overrides.get(_norm_path(b)) == "ZCYX"           # bound to B
        assert _norm_path(r"C:/data/A.tif") not in overrides    # NOT loaded A


# ---------------------------------------------------------------------------
# Extract on a converted MULTISCALE OME-Zarr store — the byte-identity gate for
# the "Prepare for large-data viewing" path.  Level 0 of the store is byte-exact
# (never coarsened), so cropping the lazy level-0 handle through the SAME extract
# path must equal a direct crop of the source fixture.
# ---------------------------------------------------------------------------

from packages.epicycle_largedata.multiscale import (  # noqa: E402
    convert_to_ome_zarr,
    open_multiscale,
)


class TestExtractOnMultiscaleStore:
    BOXES = [_box(2, 3, 30, 33), _box(10, 10, 25, 40)]

    def test_czyx_store_level0_crop_is_byte_identical(self, tmp_path):
        arr = np.random.randint(0, 60000, (2, 8, 40, 40)).astype(np.uint16)
        src = _write(tmp_path, "src.tif", arr)
        store = str(tmp_path / "src.zarr")
        convert_to_ome_zarr(src, store, image_axes="CZYX", chunk=16,
                            min_size=8, overwrite=True)
        levels, _meta = open_multiscale(store)
        level0 = levels[0]
        assert tuple(level0.shape) == arr.shape  # level 0 == source (byte-exact)

        ctl = _bare_controller(
            lazy=level0, order="CZYX",
            index_map={"s_ch0": 0, "s_ch1": 1})
        src_handle, idx = ctl._crop_source_for(["s_ch0", "s_ch1"])
        assert idx == [0, 1]

        got = extract_patches_from_shapes(
            src_handle, self.BOXES, channel_indices=idx)
        want = extract_patches_from_shapes(arr, self.BOXES)
        assert list(got) == list(want)
        for k in got:
            assert np.array_equal(np.asarray(got[k]), np.asarray(want[k])), k

    def test_zyx_store_level0_crop_is_byte_identical(self, tmp_path):
        arr = np.random.randint(0, 60000, (8, 40, 40)).astype(np.uint16)
        src = _write(tmp_path, "mono.tif", arr)
        store = str(tmp_path / "mono.zarr")
        convert_to_ome_zarr(src, store, chunk=16, min_size=8, overwrite=True)
        levels, _meta = open_multiscale(store)
        level0 = levels[0]
        assert tuple(level0.shape) == arr.shape

        ctl = _bare_controller(
            lazy=level0, order="ZYX", index_map={"mono": 0})
        src_handle, idx = ctl._crop_source_for(["mono"])
        assert idx == [0]

        got = extract_patches_from_shapes(
            src_handle, [self.BOXES[0]], channel_indices=idx)
        want = extract_patches_from_shapes(arr[None], [self.BOXES[0]])
        for k in got:
            assert got[k].shape[0] == 1
            assert np.array_equal(np.asarray(got[k]), np.asarray(want[k])), k
