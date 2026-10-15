"""Tests for per-patch axis-order handling in Step 3 source-stack loading
(fix: CZXY support / mixed-layout freeze).

Bug: the channel-first transpose was driven by a single global ``axes_combo``
set once from the first patch and never re-detected per patch.  A CZXY patch
loaded while the combo said ZCYX (or vice-versa) had its Z axis moved to the
front, and ``add_image(channel_axis=0)`` then split hundreds of Z-slices into
hundreds of layers → freeze.  Fix: detect the channel axis per-patch from the
array's real shape (`_to_channel_first`) + guard the split (`_add_image_channel_first`).
"""

from __future__ import annotations

import inspect

import numpy as np

from napari_epicycle.fusion_reconstruction.qc_features import (
    detect_axes_order,
    resolve_patch_axes,
)
from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController,
)


def _ctrl():
    # The two helpers under test don't touch __init__ state (only the array +
    # self.viewer), so a bare instance is sufficient.
    return FusionReconstructionController.__new__(FusionReconstructionController)


class _FakeViewer:
    def __init__(self):
        self.calls = []

    def add_image(self, arr, channel_axis=None, name=None):
        self.calls.append({"shape": tuple(arr.shape), "channel_axis": channel_axis, "name": name})


# ---------------------------------------------------------------------------
# detect_axes_order (per-patch, from real shape)
# ---------------------------------------------------------------------------

def test_detect_czyx_channel_first():
    assert detect_axes_order((2, 300, 512, 512)) == "CZYX"


def test_detect_zcyx_z_first():
    assert detect_axes_order((300, 2, 512, 512)) == "ZCYX"


def test_detect_3d_no_channel():
    assert detect_axes_order((300, 512, 512)) == "ZYX"


# ---------------------------------------------------------------------------
# _to_channel_first — both layouts end channel-first, regardless of any combo
# ---------------------------------------------------------------------------

def test_to_channel_first_czyx_unchanged():
    out = _ctrl()._to_channel_first(np.zeros((2, 40, 8, 8)))
    assert out.shape == (2, 40, 8, 8)  # already channel-first


def test_to_channel_first_zcyx_transposed():
    out = _ctrl()._to_channel_first(np.zeros((40, 2, 8, 8)))
    assert out.shape == (2, 40, 8, 8)  # Z-first -> channel-first


def test_to_channel_first_3d_unchanged():
    out = _ctrl()._to_channel_first(np.zeros((40, 8, 8)))
    assert out.shape == (40, 8, 8)


def test_to_channel_first_both_layouts_converge():
    """The whole point: a CZXY and a ZCXY patch of the same logical data both
    end up channel-first with 2 channels — no dependence on a global combo."""
    czyx = _ctrl()._to_channel_first(np.zeros((2, 50, 8, 8)))
    zcyx = _ctrl()._to_channel_first(np.zeros((50, 2, 8, 8)))
    assert czyx.shape == zcyx.shape == (2, 50, 8, 8)


# ---------------------------------------------------------------------------
# _add_image_channel_first — guard against the layer explosion
# ---------------------------------------------------------------------------

def test_add_image_splits_small_channel_count():
    c = _ctrl(); c.viewer = _FakeViewer()
    c._add_image_channel_first(np.zeros((2, 40, 8, 8)), "src")
    assert c.viewer.calls[0]["channel_axis"] == 0  # split into 2 channels


def test_add_image_guard_refuses_huge_channel_dim():
    """A misread layout leaving Z (hundreds) at axis 0 must NOT be split into
    hundreds of layers — the guard loads it as a single stack instead."""
    c = _ctrl(); c.viewer = _FakeViewer()
    c._add_image_channel_first(np.zeros((300, 2, 8, 8)), "src")
    assert c.viewer.calls[0]["channel_axis"] is None  # plain add, no explosion


def test_add_image_3d_plain():
    c = _ctrl(); c.viewer = _FakeViewer()
    c._add_image_channel_first(np.zeros((40, 8, 8)), "src")
    assert c.viewer.calls[0]["channel_axis"] is None


def test_add_image_guard_boundary_8_splits():
    """Exactly 8 channel-first dim is still split (threshold is > 8)."""
    c = _ctrl(); c.viewer = _FakeViewer()
    c._add_image_channel_first(np.zeros((8, 40, 8, 8)), "src")
    assert c.viewer.calls[0]["channel_axis"] == 0


def test_add_image_guard_boundary_9_refuses():
    """9 trips the guard (no split) — pins the exact >8 boundary."""
    c = _ctrl(); c.viewer = _FakeViewer()
    c._add_image_channel_first(np.zeros((9, 40, 8, 8)), "src")
    assert c.viewer.calls[0]["channel_axis"] is None


def test_single_channel_4d_handled():
    """C=1 4D (1,Z,Y,X): stays channel-first, splits into one channel."""
    c = _ctrl(); c.viewer = _FakeViewer()
    arr = c._to_channel_first(np.zeros((1, 50, 8, 8)))
    assert arr.shape == (1, 50, 8, 8)
    c._add_image_channel_first(arr, "src")
    assert c.viewer.calls[0]["channel_axis"] == 0


def test_ambiguous_tie_is_treated_channel_first():
    """Tie (2,2,Y,X): detect resolves to CZYX (channel-first) and leaves it —
    either interpretation yields the same channel count, so it's safe."""
    out = _ctrl()._to_channel_first(np.zeros((2, 2, 8, 8)))
    assert out.shape == (2, 2, 8, 8)


# ---------------------------------------------------------------------------
# Wiring: the load sites use the per-patch helpers, not the global combo
# ---------------------------------------------------------------------------

def test_load_sites_use_per_patch_helpers_not_global_combo():
    import inspect
    from napari_epicycle.fusion_reconstruction.controller import _PatchLoadWorker
    # The QC-view load is async (2026-06-05): _load_patch_for_qc_view passes
    # self._to_channel_first to _PatchLoadWorker.run (which calls it off-thread),
    # and _on_patch_loaded calls _add_image_channel_first on the main thread.
    # Check the combined async path rather than the single old method.
    qc = (
        inspect.getsource(FusionReconstructionController._load_patch_for_qc_view)
        + inspect.getsource(FusionReconstructionController._on_patch_loaded)
        + inspect.getsource(_PatchLoadWorker.run)
    )
    src = inspect.getsource(FusionReconstructionController._load_source_stack)
    for name, body in (("async QC load path", qc), ("_load_source_stack", src)):
        assert "_to_channel_first" in body, f"{name} must use _to_channel_first"
        assert "_add_image_channel_first" in body, f"{name} must use _add_image_channel_first"
        # The stale global-combo-driven transpose must be gone from these paths.
        assert "axes_combo.currentText" not in body, (
            f"{name} must not drive the transpose from the global axes_combo"
        )


# ---------------------------------------------------------------------------
# Per-patch axes cache on the state manager (cosmetic Axis column, 2026-06-07)
# ---------------------------------------------------------------------------

def _make_state_manager(tmp_path):
    from napari_epicycle.fusion_reconstruction.state_manager import (
        ReconstructionStateManager,
    )
    mgr = ReconstructionStateManager()
    mgr.load(str(tmp_path))
    return mgr


def test_get_patch_axes_default_when_unset(tmp_path):
    mgr = _make_state_manager(tmp_path)
    # Unknown patch → default.
    assert mgr.get_patch_axes("nope.tif") == "CZYX"
    # Custom default honored.
    assert mgr.get_patch_axes("nope.tif", default="ZYX") == "ZYX"


def test_update_and_get_patch_axes_round_trip(tmp_path):
    mgr = _make_state_manager(tmp_path)
    mgr.update_patch_metadata("p0.tif", axes="ZCYX")
    assert mgr.get_patch_axes("p0.tif") == "ZCYX"
    # Overwrite.
    mgr.update_patch_metadata("p0.tif", axes="CZYX")
    assert mgr.get_patch_axes("p0.tif") == "CZYX"


def test_patch_axes_persists_through_disk(tmp_path):
    from napari_epicycle.fusion_reconstruction.state_manager import (
        ReconstructionStateManager,
    )
    mgr = _make_state_manager(tmp_path)
    mgr.update_patch_metadata("p1.tif", axes="ZCYX")
    mgr.save()  # bypass QTimer debounce — no event loop in unit tests

    mgr2 = ReconstructionStateManager()
    mgr2.load(str(tmp_path))
    assert mgr2.get_patch_axes("p1.tif") == "ZCYX"


def test_patch_axes_does_not_disturb_status(tmp_path):
    """Writing axes must not clobber an existing status (metadata-only)."""
    mgr = _make_state_manager(tmp_path)
    mgr.set_patch_status("p2.tif", "reconstructed")
    mgr.update_patch_metadata("p2.tif", axes="CZYX")
    assert mgr.get_patch_status("p2.tif") == "reconstructed"
    assert mgr.get_patch_axes("p2.tif") == "CZYX"


def test_get_patch_axes_sentinel_default_survives_when_unset(tmp_path):
    """HIGH-bug regression (2026-06-07): when a patch record EXISTS but has no
    `axes` key, get_patch_axes must return the BARE `default` — not
    str(default).  _build_axes_by_patch passes a sentinel object as default to
    detect 'uncached'; str()-wrapping it leaked '<object object at 0x...>' into
    the Patch State column AND made the lazy-detect/adaptivity path dead code
    (the value was never the bare sentinel, so it always looked 'cached')."""
    mgr = _make_state_manager(tmp_path)
    mgr.set_patch_status("p3.tif", "predicted")  # record exists, no axes key
    sentinel = object()
    # Must return the exact sentinel object (identity), not its str().
    assert mgr.get_patch_axes("p3.tif", default=sentinel) is sentinel
    # A normal string default still round-trips.
    assert mgr.get_patch_axes("p3.tif", default="CZYX") == "CZYX"


# ---------------------------------------------------------------------------
# Header-only detection helper — caches the resolved axes from a real TIFF
# ---------------------------------------------------------------------------

def test_detect_and_cache_patch_axes_caches_resolved_value(tmp_path):
    """A header-only read of a CZYX-shaped TIFF resolves + caches 'CZYX'."""
    import tifffile

    mgr = _make_state_manager(tmp_path)
    # (C, Z, Y, X): channel axis 0 is the small dim → CZYX by size heuristic.
    arr = np.zeros((2, 30, 16, 16), dtype=np.uint8)
    fname = "patchA.tif"
    tifffile.imwrite(str(tmp_path / fname), arr)

    ctrl = FusionReconstructionController.__new__(FusionReconstructionController)
    ctrl.state_manager = mgr
    ctrl.loaded_stack_dir = str(tmp_path)

    axes = ctrl._detect_and_cache_patch_axes(fname)
    assert axes == "CZYX"
    # Cached on the record, retrievable via the accessor.
    assert mgr.get_patch_axes(fname) == "CZYX"


def test_detect_and_cache_patch_axes_zcyx(tmp_path):
    """A (Z, C, Y, X) thin-channel stack resolves + caches 'ZCYX'."""
    import tifffile

    mgr = _make_state_manager(tmp_path)
    arr = np.zeros((30, 2, 16, 16), dtype=np.uint8)  # Z-first
    fname = "patchB.tif"
    tifffile.imwrite(str(tmp_path / fname), arr)

    ctrl = FusionReconstructionController.__new__(FusionReconstructionController)
    ctrl.state_manager = mgr
    ctrl.loaded_stack_dir = str(tmp_path)

    axes = ctrl._detect_and_cache_patch_axes(fname)
    assert axes == "ZCYX"
    assert mgr.get_patch_axes(fname) == "ZCYX"


def test_detect_and_cache_patch_axes_3d(tmp_path):
    """A 3D (Z, Y, X) stack resolves to 'ZYX' (no C/Z ambiguity)."""
    import tifffile

    mgr = _make_state_manager(tmp_path)
    arr = np.zeros((30, 16, 16), dtype=np.uint8)
    fname = "patchC.tif"
    tifffile.imwrite(str(tmp_path / fname), arr)

    ctrl = FusionReconstructionController.__new__(FusionReconstructionController)
    ctrl.state_manager = mgr
    ctrl.loaded_stack_dir = str(tmp_path)

    assert ctrl._detect_and_cache_patch_axes(fname) == "ZYX"
    assert mgr.get_patch_axes(fname) == "ZYX"


def test_detect_and_cache_patch_axes_single_channel_4d_zcyx(tmp_path):
    """Header-only detection of a 4D SINGLE-channel (Z, 1, Y, X) TIFF caches
    'ZCYX' — the singleton channel (axis 1) is the smallest dim.  The header
    read preserves the C=1 shape, so the size heuristic resolves correctly."""
    import tifffile

    mgr = _make_state_manager(tmp_path)
    arr = np.zeros((30, 1, 16, 16), dtype=np.uint8)  # (Z, C=1, Y, X)
    fname = "patchSCz.tif"
    tifffile.imwrite(str(tmp_path / fname), arr)

    ctrl = FusionReconstructionController.__new__(FusionReconstructionController)
    ctrl.state_manager = mgr
    ctrl.loaded_stack_dir = str(tmp_path)

    assert ctrl._detect_and_cache_patch_axes(fname) == "ZCYX"
    assert mgr.get_patch_axes(fname) == "ZCYX"


def test_detect_and_cache_patch_axes_single_channel_4d_czyx(tmp_path):
    """Header-only detection of a 4D SINGLE-channel (1, Z, Y, X) TIFF caches
    'CZYX' — the singleton channel (axis 0) is the smallest dim."""
    import tifffile

    mgr = _make_state_manager(tmp_path)
    arr = np.zeros((1, 30, 16, 16), dtype=np.uint8)  # (C=1, Z, Y, X)
    fname = "patchSCc.tif"
    tifffile.imwrite(str(tmp_path / fname), arr)

    ctrl = FusionReconstructionController.__new__(FusionReconstructionController)
    ctrl.state_manager = mgr
    ctrl.loaded_stack_dir = str(tmp_path)

    assert ctrl._detect_and_cache_patch_axes(fname) == "CZYX"
    assert mgr.get_patch_axes(fname) == "CZYX"


def test_detect_and_cache_patch_axes_bad_file_falls_back(tmp_path):
    """A missing / unreadable file must NOT raise — falls back to 'CZYX'
    and caches it so the list never re-reads a broken patch repeatedly."""
    mgr = _make_state_manager(tmp_path)
    ctrl = FusionReconstructionController.__new__(FusionReconstructionController)
    ctrl.state_manager = mgr
    ctrl.loaded_stack_dir = str(tmp_path)

    axes = ctrl._detect_and_cache_patch_axes("does_not_exist.tif")
    assert axes == "CZYX"
    assert mgr.get_patch_axes("does_not_exist.tif") == "CZYX"


def test_resolve_patch_axes_matches_detection_for_shapes():
    """Sanity: the helper the detection wraps yields the expected axes for
    pure shape carriers (mirrors what _detect_and_cache_patch_axes feeds it)."""
    czyx = np.broadcast_to(np.uint8(0), (2, 30, 16, 16))
    zcyx = np.broadcast_to(np.uint8(0), (30, 2, 16, 16))
    assert resolve_patch_axes(czyx, fallback="CZYX") == "CZYX"
    assert resolve_patch_axes(zcyx, fallback="CZYX") == "ZCYX"


# ---------------------------------------------------------------------------
# The global Axes combo is gone — controller no longer reads it
# ---------------------------------------------------------------------------

def test_controller_source_has_no_axes_combo_reads():
    """The vestigial global Axes combo was removed (2026-06-07).  No
    controller code path may read it (currentText / findText / setCurrentIndex);
    only history comments may mention the name."""
    import napari_epicycle.fusion_reconstruction.controller as ctrl_mod

    src = inspect.getsource(ctrl_mod)
    assert "axes_combo.currentText" not in src
    assert "axes_combo.findText" not in src
    assert "axes_combo.setCurrentIndex" not in src
    assert "axes_combo.currentIndex" not in src


# ---------------------------------------------------------------------------
# Patch-state row formatting (pure helper — Qt-free)
# ---------------------------------------------------------------------------

def test_format_patch_state_row_includes_axis():
    from napari_epicycle.fusion_reconstruction.ui import _format_patch_state_row

    row = _format_patch_state_row(
        "patch_0", "predicted", "CZYX", "●●○○○○", max_name=10, max_status=12
    )
    assert "patch_0" in row
    assert "predicted" in row
    assert "CZYX" in row
    assert "●●○○○○" in row


def test_format_patch_state_row_blank_axis_when_missing():
    """A None/empty axis renders a blank placeholder (no crash, alignment kept)."""
    from napari_epicycle.fusion_reconstruction.ui import (
        _format_patch_state_row,
        _AXIS_COL_WIDTH,
    )

    row_known = _format_patch_state_row(
        "p", "unprocessed", "ZYX", "○○○○○○", max_name=4, max_status=12
    )
    row_blank = _format_patch_state_row(
        "p", "unprocessed", None, "○○○○○○", max_name=4, max_status=12
    )
    # The blank-axis row reserves the same column width so dots stay aligned.
    assert "ZYX" in row_known
    assert "ZYX" not in row_blank
    # Same total length up to the axis column → dots aligned across rows.
    assert len(row_known) == len(row_blank)


def test_format_patch_state_row_axis_padded_to_width():
    from napari_epicycle.fusion_reconstruction.ui import (
        _format_patch_state_row,
        _AXIS_COL_WIDTH,
    )

    # ZYX (3 chars) is padded to _AXIS_COL_WIDTH so 4-char CZYX rows align.
    row = _format_patch_state_row(
        "p", "predicted", "ZYX", "", max_name=4, max_status=12
    )
    assert ("ZYX".ljust(_AXIS_COL_WIDTH)) in row
