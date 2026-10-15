"""F2 — ④ Review & Library reads the PERSISTED patch registry.

Covers the three moving parts of the "④ reads the saved patches" brick:

* the sidecar-derived detail string (no pixel read),
* per-source interactivity (`_refresh_library_from_registry` stamps + gates),
* the saved-patch thumbnail helper (cached).

Qt is required (the panel builds real row widgets + QPixmaps), so the whole
module skips cleanly when no Qt binding is present — mirroring `test_ui.py`.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

pytest.importorskip("qtpy")

from qtpy.QtCore import Qt  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from napari_epicycle.data_preparation._paths import step1_patches_dir  # noqa: E402
from napari_epicycle.data_preparation._state import (  # noqa: E402
    empty_state,
    register_patch,
    register_source,
    save_state,
)
from napari_epicycle.data_preparation.controller import (  # noqa: E402
    LIBRARY_ROLE_INTERACTIVE,
)
from napari_epicycle.data_preparation.ui import (  # noqa: E402
    _patch_thumbnail_pixmap,
    _sidecar_detail_for_path,
)


# ---------------------------------------------------------------------------
# Fixtures (mirroring tests/data_preparation/test_ui.py)
# ---------------------------------------------------------------------------
class _FakeLayerEvents:
    def __init__(self):
        class _E:
            def connect(self, *_a, **_k):
                pass
        self.inserted = _E()
        self.removed = _E()


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


# ---------------------------------------------------------------------------
# Sidecar detail — Y×X · Cch · Z, from the frozen sidecar (no pixel read)
# ---------------------------------------------------------------------------
class TestSidecarDetail:
    def test_derives_from_a_written_sidecar(self, tmp_path):
        sc = tmp_path / "p_1_metadata.json"
        sc.write_text(json.dumps({
            "patch_name": "p_1",
            "channel_names": ["nuclei", "membrane"],
            "cropped_shape": [2, 50, 512, 384],   # C, Z, Y, X
            "selected_z_range": [12, 62],
        }), encoding="utf-8")
        _sidecar_detail_for_path.cache_clear()
        got = _sidecar_detail_for_path(str(sc), os.stat(sc).st_mtime_ns)
        assert got == "512×384 · 2ch · Z12–62"

    def test_single_channel_3d_sidecar(self, tmp_path):
        sc = tmp_path / "g_1_metadata.json"
        sc.write_text(json.dumps({
            "channel_names": ["gray"],
            "cropped_shape": [30, 64, 64],        # Z, Y, X (3D)
        }), encoding="utf-8")
        _sidecar_detail_for_path.cache_clear()
        got = _sidecar_detail_for_path(str(sc), os.stat(sc).st_mtime_ns)
        assert got == "64×64 · 1ch · Z30"

    def test_missing_or_corrupt_sidecar_is_empty(self, tmp_path):
        assert _sidecar_detail_for_path(str(tmp_path / "nope.json"), 0) == ""
        bad = tmp_path / "bad_metadata.json"
        bad.write_text("{ not json", encoding="utf-8")
        assert _sidecar_detail_for_path(str(bad), os.stat(bad).st_mtime_ns) == ""


# ---------------------------------------------------------------------------
# Per-source interactivity — current source clickable, others greyed/locked
# ---------------------------------------------------------------------------
def _seed_workspace(ws):
    """Two sources + patches: src_0 (loaded), src_1 (other), plus one legacy
    row with source_id == ''.  Returns the src_0 file path."""
    st = empty_state()
    a_path = os.path.join(ws, "sourceA.tif")
    b_path = os.path.join(ws, "sourceB.tif")
    register_source(st, path=a_path, display_name="A", axes="CZYX")
    register_source(st, path=b_path, display_name="B", axes="CZYX")
    register_patch(st, patch_name="A_1", relative_path="artifacts/step1/patches/A_1.tiff",
                   sidecar_path="artifacts/step1/patches/A_1_metadata.json", source_id="src_0")
    register_patch(st, patch_name="A_2", relative_path="artifacts/step1/patches/A_2.tiff",
                   sidecar_path="artifacts/step1/patches/A_2_metadata.json", source_id="src_0")
    register_patch(st, patch_name="B_1", relative_path="artifacts/step1/patches/B_1.tiff",
                   sidecar_path="artifacts/step1/patches/B_1_metadata.json", source_id="src_1")
    register_patch(st, patch_name="LEG_1", relative_path="artifacts/step1/patches/LEG_1.tiff",
                   sidecar_path="artifacts/step1/patches/LEG_1_metadata.json", source_id="")
    save_state(ws, st)
    return a_path


def _item_by_name(lst, name):
    for i in range(lst.count()):
        if lst.item(i).text() == name:
            return lst.item(i)
    return None


class TestPerSourceInteractivity:
    def test_current_source_rows_are_interactive_others_are_not(self, widget, tmp_path):
        ws = str(tmp_path)
        a_path = _seed_workspace(ws)
        widget.output_file_path.setText(ws)
        ctl = widget.controller
        ctl._loaded_source_file = a_path        # source A is loaded

        ctl._refresh_library_from_registry()

        lst = widget.extracted_roi_list
        names = {lst.item(i).text() for i in range(lst.count())}
        assert names == {"A_1", "A_2", "B_1", "LEG_1"}     # ALL saved patches

        for cur in ("A_1", "A_2"):
            it = _item_by_name(lst, cur)
            assert it.data(LIBRARY_ROLE_INTERACTIVE) is True
            assert it.flags() & Qt.ItemIsSelectable, f"{cur} must be selectable"

        for other in ("B_1", "LEG_1"):
            it = _item_by_name(lst, other)
            assert it.data(LIBRARY_ROLE_INTERACTIVE) is False
            assert not (it.flags() & Qt.ItemIsSelectable), \
                f"{other} must NOT be selectable"

    def test_no_source_loaded_makes_everything_non_interactive(self, widget, tmp_path):
        ws = str(tmp_path)
        _seed_workspace(ws)
        widget.output_file_path.setText(ws)
        ctl = widget.controller
        ctl._loaded_source_file = None          # nothing loaded

        ctl._refresh_library_from_registry()

        lst = widget.extracted_roi_list
        assert lst.count() == 4
        for i in range(lst.count()):
            assert lst.item(i).data(LIBRARY_ROLE_INTERACTIVE) is False

    def test_other_source_row_stays_untickable_even_if_stale_staged(self, widget, tmp_path):
        # Edge: source B's patch is still staged in memory (a prior extract)
        # when source A is loaded.  Its registry row is source B → non-interactive
        # → it must NOT be tickable, and Export must never pick it, even though
        # `_status_of` would read "staged".
        ws = str(tmp_path)
        a_path = _seed_workspace(ws)
        widget.output_file_path.setText(ws)
        ctl = widget.controller
        ctl._loaded_source_file = a_path
        ctl.extracted_patches = {"B_1": {"data": None}}   # stale cross-source stage

        ctl._refresh_library_from_registry()

        lst = widget.extracted_roi_list
        b = _item_by_name(lst, "B_1")
        assert b.data(LIBRARY_ROLE_INTERACTIVE) is False
        assert not (b.flags() & Qt.ItemIsUserCheckable), "cross-source row must be untickable"
        assert b.checkState() == Qt.Unchecked
        assert "B_1" not in widget.checked_patch_names()

    def test_registry_only_row_selection_does_not_crash(self, widget, tmp_path):
        # A registry-only (on-disk, no in-memory data) row is selectable for the
        # current source; selecting it must not raise (the plot stays blank —
        # `_on_plot_intensity_clicked` None-guards on `extracted_patches`).
        ws = str(tmp_path)
        a_path = _seed_workspace(ws)
        widget.output_file_path.setText(ws)
        ctl = widget.controller
        ctl._loaded_source_file = a_path
        ctl._refresh_library_from_registry()

        lst = widget.extracted_roi_list
        lst.setCurrentItem(_item_by_name(lst, "A_1"))
        ctl._on_roi_selection_changed()         # must not raise


# ---------------------------------------------------------------------------
# Thumbnail helper — QPixmap from a small saved TIFF, cached by (path, mtime)
# ---------------------------------------------------------------------------
class TestThumbnail:
    def _write_patch_tiff(self, ws, name, shape):
        import tifffile
        d = step1_patches_dir(ws)
        os.makedirs(d, exist_ok=True)
        path = str(d / f"{name}.tiff")
        arr = (np.random.default_rng(0).random(shape) * 1000).astype(np.uint16)
        tifffile.imwrite(path, arr, metadata={"axes": "CZYX" if len(shape) == 4 else "ZYX"})
        return path

    def test_returns_pixmap_and_caches(self, app, tmp_path):
        ws = str(tmp_path)
        path = self._write_patch_tiff(ws, "T_1", (2, 6, 32, 24))
        mtime = os.stat(path).st_mtime_ns
        _patch_thumbnail_pixmap.cache_clear()

        pm = _patch_thumbnail_pixmap(path, mtime, 28)
        assert pm is not None and not pm.isNull()
        misses_after_first = _patch_thumbnail_pixmap.cache_info().misses

        pm2 = _patch_thumbnail_pixmap(path, mtime, 28)       # same key
        assert pm2 is pm                                     # served from cache
        assert _patch_thumbnail_pixmap.cache_info().misses == misses_after_first

    def test_single_channel_volume_thumbnail(self, app, tmp_path):
        ws = str(tmp_path)
        path = self._write_patch_tiff(ws, "G_1", (8, 16, 16))   # 3D ZYX
        _patch_thumbnail_pixmap.cache_clear()
        pm = _patch_thumbnail_pixmap(path, os.stat(path).st_mtime_ns, 28)
        assert pm is not None and not pm.isNull()

    def test_missing_tiff_returns_none(self, app, tmp_path):
        _patch_thumbnail_pixmap.cache_clear()
        assert _patch_thumbnail_pixmap(str(tmp_path / "gone.tiff"), 0, 28) is None
