"""C#2 — the ② region PREVIEW must be WYSIWYG against "Apply → whole volume".

The Step-1 preview suite is ~31 lifecycle/wiring tests and ZERO numeric
equivalence tests, which is exactly why the preview could normalize against the
CROP's own p0.1/p99.5 while the whole-volume apply normalizes against the
SOURCE's — an invisible, systematic brightness/contrast difference between what
the user previews and what the button writes.  This file is the missing lens:
run BOTH pure compute paths over the same synthetic volume and compare the
pixels.

Scope note — halo (C#5) is deliberately OUT of this brick.  A local op
(gaussian) reflect-pads at the array border, so the crop's own faces genuinely
differ from the same voxels inside the whole volume.  The comparison is
therefore made on the crop's INTERIOR, eroded by more than the filter's reach;
the faces are C#5's problem, not C#2's.
"""

from __future__ import annotations

import numpy as np
import pytest

from napari_epicycle.data_preparation import controller
from packages.epicycle_preprocessing import ops


# Volume small enough that BOTH paths take exact (unsubsampled) percentiles, so
# any difference is the normalization WINDOW, not sampling noise.
_SHAPE = (32, 96, 96)
_SPACING = (1.0, 1.0, 1.0)
_SIGMA = 1.0
# scipy's gaussian_filter default truncate=4.0 → reach = ceil(4*sigma) voxels.
# Erode by more than that so no compared voxel can see either array's border.
_HALO = int(np.ceil(4.0 * _SIGMA)) + 2


def _volume():
    """A volume whose CENTRE is much brighter than the bulk.

    That gradient is the point: the centre crop's own p0.1/p99.5 window is far
    narrower and far higher than the whole volume's, so a preview that measures
    percentiles on the crop lands on visibly different pixels.  A flat or
    uniformly-random volume would let the bug pass unnoticed.
    """
    z, y, x = _SHAPE
    rng = np.random.RandomState(7)
    vol = (rng.rand(*_SHAPE) * 400.0 + 100.0)
    zz, yy, xx = np.ogrid[:z, :y, :x]
    r2 = (((zz - z / 2) / (z / 2)) ** 2
          + ((yy - y / 2) / (y / 2)) ** 2
          + ((xx - x / 2) / (x / 2)) ** 2)
    vol += 8000.0 * np.exp(-3.0 * r2)          # bright core
    return np.clip(vol, 0, 65535).astype(np.uint16)


def _recipe():
    return {"channels": [[{"key": "gaussian", "params": {"sigma": _SIGMA}}]],
            "geometry": None}


def _centre_bounds(shape, dz, dy, dx):
    z, y, x = shape
    z0, y0, x0 = (z - dz) // 2, (y - dy) // 2, (x - dx) // 2
    return {"z0": z0, "z1": z0 + dz, "y0": y0, "y1": y0 + dy,
            "x0": x0, "x1": x0 + dx}


class TestPreviewMatchesWholeVolume:

    def test_region_preview_equals_the_whole_volume_result(self):
        vol = _volume()
        recipe = _recipe()
        bounds = _centre_bounds(_SHAPE, 24, 48, 48)

        # --- the WHOLE-VOLUME path's pure compute (what "Apply" writes) -------
        whole_list, _sp, _bounds, _stamp = controller._process_whole_volume_core(
            recipe, [vol], _SPACING)
        whole_u16 = whole_list[0]

        # --- the PREVIEW path's pure compute on the centre region -------------
        crop = controller._crop_center(vol, bounds)
        _befores, afters, _sp2 = controller._compute_preview(
            recipe, [crop], _SPACING,
            bounds_per_channel=controller._whole_volume_norm_bounds([vol]))
        preview_u16 = ops.to_uint16(afters[0])

        assert preview_u16.shape == crop.shape

        # Compare the crop's INTERIOR only — the faces are halo (C#5), not C#2.
        h = _HALO
        pv = preview_u16[h:-h, h:-h, h:-h]
        wv = whole_u16[bounds["z0"] + h:bounds["z1"] - h,
                       bounds["y0"] + h:bounds["y1"] - h,
                       bounds["x0"] + h:bounds["x1"] - h]
        assert pv.shape == wv.shape and pv.size > 0

        diff = np.abs(pv.astype(np.int32) - wv.astype(np.int32))
        # 1 DN of 65535 = float32 rounding at the bake, nothing more.
        assert int(diff.max()) <= 1, (
            f"preview differs from the whole-volume result by up to "
            f"{int(diff.max())} DN (mean {float(diff.mean()):.1f}) — the two "
            f"paths are not normalizing against the same window")

    def test_the_window_is_the_whole_volume_window_not_the_crop_window(self):
        """The guard behind the guard: state the actual defect numerically, so a
        future refactor that silently reverts to crop-local percentiles fails
        here with a readable reason rather than only as a pixel diff."""
        vol = _volume()
        bounds = _centre_bounds(_SHAPE, 24, 48, 48)
        crop = controller._crop_center(vol, bounds)

        whole = controller._whole_volume_norm_bounds([vol])[0]
        crop_local = ops.percentile_bounds(np.asarray(crop, dtype=np.float32))

        assert whole is not None and crop_local is not None
        # The fixture is built so these are far apart — otherwise the
        # equivalence test above would pass for the wrong reason.
        assert abs(whole[0] - crop_local[0]) + abs(whole[1] - crop_local[1]) > 100.0
        # And the window the whole-volume core actually used is the same one.
        _p, _s, used, _st = controller._process_whole_volume_core(
            {"channels": [[]], "geometry": None}, [vol], _SPACING)
        assert used[0] == pytest.approx(whole, rel=1e-6)

    def test_bounds_helper_tolerates_a_degenerate_channel(self):
        flat = np.zeros((4, 8, 8), dtype=np.uint16)
        assert controller._whole_volume_norm_bounds([flat]) == [None]

    def test_bounds_helper_matches_the_apply_helper_for_in_memory_arrays(self):
        """For a plain ndarray the preview must measure the window with the SAME
        helper the apply path uses — not a parallel approximation."""
        vol = _volume()
        assert (controller._whole_volume_norm_bounds([vol])[0]
                == controller._strided_percentile_bounds(vol))


class TestPreviewWiring:
    """The equivalence above is a property of the PURE compute; these guard the
    wiring that has to hold for a real Preview click to inherit it."""

    def test_worker_measures_the_window_on_its_own_thread(self, monkeypatch):
        """No cached window → the worker measures it in `run()` (the worker
        thread), and `execute` receives it.  It must never be measured in
        `on_preview_recipe` — that is the UI thread."""
        seen = {}

        def fake_execute(recipe, crops, spacing, bounds_per_channel=None):
            seen["bounds"] = bounds_per_channel
            return ([np.asarray(c, dtype=np.float32) for c in crops],
                    tuple(spacing), {})

        monkeypatch.setattr(controller, "_pp_execute", fake_execute)
        vol = _volume()
        b = _centre_bounds(_SHAPE, 8, 16, 16)
        worker = controller._PreprocPreviewWorker(
            {"channels": [[]], "geometry": None}, [vol], [b], _SPACING,
            ["cad"], ("#ffffff",))
        worker.run()
        assert seen["bounds"] == controller._whole_volume_norm_bounds([vol])

    def test_a_supplied_window_is_used_verbatim(self, monkeypatch):
        seen = {}

        def fake_execute(recipe, crops, spacing, bounds_per_channel=None):
            seen["bounds"] = bounds_per_channel
            return ([np.asarray(c, dtype=np.float32) for c in crops],
                    tuple(spacing), {})

        monkeypatch.setattr(controller, "_pp_execute", fake_execute)
        vol = _volume()
        b = _centre_bounds(_SHAPE, 8, 16, 16)
        worker = controller._PreprocPreviewWorker(
            {"channels": [[]], "geometry": None}, [vol], [b], _SPACING,
            ["cad"], ("#ffffff",), norm_bounds=[(3.0, 9.0)])
        worker.run()
        assert seen["bounds"] == [(3.0, 9.0)]

    def test_histogram_frame_uses_the_same_window(self):
        """The before/after curves must be drawn on the axis `execute` really
        used, or a matched pair reads as shifted (the second half of C#2)."""
        raw = (np.arange(1000, dtype=np.float32) * 10.0).reshape(10, 10, 10)
        after = np.zeros_like(raw)
        shared = [(0.0, 20000.0)]
        entries = controller.build_preview_histogram_entries(
            ["cad"], ("#ffffff",), [raw], [after],
            bounds_per_channel=shared)
        assert entries[0].stats["p_lo"] == 0.0
        assert entries[0].stats["p_hi"] == 20000.0
        # No shared window → falls back to this crop's own percentiles, which is
        # what `execute` does for that channel too.
        fallback = controller.build_preview_histogram_entries(
            ["cad"], ("#ffffff",), [raw], [after], bounds_per_channel=None)
        assert fallback[0].stats["p_hi"] != 20000.0


class TestCachedWindowReuse:

    class _Ctl:
        _processed_cache = None

        def _current_source_id(self):
            return "src-1"

    def _ctl(self):
        c = self._Ctl()
        c._cached_norm_bounds = controller.DataPreparationController._cached_norm_bounds.__get__(c)
        return c

    def test_reuses_the_apply_runs_window(self):
        c = self._ctl()
        c._processed_cache = {"source_id": "src-1", "names": ["cad"],
                              "bounds": [(1.0, 2.0)]}
        assert c._cached_norm_bounds(["cad"]) == [(1.0, 2.0)]

    def test_ignores_another_sources_window(self):
        c = self._ctl()
        c._processed_cache = {"source_id": "other", "names": ["cad"],
                              "bounds": [(1.0, 2.0)]}
        assert c._cached_norm_bounds(["cad"]) is None

    def test_ignores_a_mismatched_channel_count(self):
        c = self._ctl()
        c._processed_cache = {"source_id": "src-1", "names": ["cad"],
                              "bounds": [(1.0, 2.0)]}
        assert c._cached_norm_bounds(["cad", "nuc"]) is None

    def test_ignores_the_same_count_under_different_names(self):
        """A window is per-CHANNEL.  Two sources with two channels each would
        pass a count check while the cached window belongs to the wrong
        channels — the C#2 fix would then normalise against another image's
        percentiles, which is the very defect it exists to close."""
        c = self._ctl()
        c._processed_cache = {"source_id": "src-1", "names": ["cad", "nuc"],
                              "bounds": [(1.0, 2.0), (3.0, 4.0)]}
        assert c._cached_norm_bounds(["nuc", "cad"]) is None
        assert c._cached_norm_bounds(["dapi", "gfp"]) is None
        assert c._cached_norm_bounds(["cad", "nuc"]) == [(1.0, 2.0), (3.0, 4.0)]

    def test_a_cache_without_names_is_not_trusted(self):
        """Legacy cache shape (pre-names): fall back to measuring rather than
        guess which channels the window belongs to."""
        c = self._ctl()
        c._processed_cache = {"source_id": "src-1", "bounds": [(1.0, 2.0)]}
        assert c._cached_norm_bounds(["cad"]) is None

    def test_no_cache_is_not_an_error(self):
        assert self._ctl()._cached_norm_bounds(["cad"]) is None


class TestHistogramDensity:
    """C#7 — a geometry (resample) step changes the voxel COUNT, so raw counts
    put the two curves on different scales and the processed one reads as
    uniformly taller/lower.  Each curve is normalised by its OWN voxel count
    (expressed per million voxels, so the log-Y toggle stays useful), which is
    what makes them comparable at all."""

    def test_resampled_after_is_comparable_to_before(self):
        before = np.full((10, 10, 10), 0.5, dtype=np.float32)     # 1000 voxels
        after = np.full((5, 5, 5), 0.5, dtype=np.float32)         #  125 voxels
        e = controller.build_preview_histogram_entries(
            ["cad"], ("#ffffff",), [before], [after],
            bounds_per_channel=[(0.0, 1.0)])[0]
        b = np.asarray(e.before_counts, dtype=np.float64)
        a = np.asarray(e.after_counts, dtype=np.float64)
        # Same distribution, 8x fewer voxels -> the curves must coincide.
        assert b.max() == pytest.approx(a.max(), rel=1e-6)
        assert b.sum() == pytest.approx(a.sum(), rel=1e-6)
        assert b.sum() == pytest.approx(1e6, rel=1e-6)   # per-million density

    def test_empty_channel_does_not_divide_by_zero(self):
        empty = np.zeros((0,), dtype=np.float32)
        e = controller.build_preview_histogram_entries(
            ["cad"], ("#ffffff",), [empty], [empty],
            bounds_per_channel=[(0.0, 1.0)])[0]
        assert np.all(np.isfinite(np.asarray(e.before_counts, dtype=np.float64)))
        assert np.all(np.isfinite(np.asarray(e.after_counts, dtype=np.float64)))


#: The one sentence every fidelity note ends with — plain language, no jargon.
_TAIL = ("this region is too small to show this step honestly — the whole "
         "volume will look different")


class TestPreviewFidelityNotes:
    """C#3 / C#4 — HONESTY, not numerics.

    Two ops genuinely cannot be shown faithfully on a test region: bleach fits
    its decay over the crop's Z (at a shallow region it is a silent no-op in
    preview and active on apply), and CLAHE clamps its tile size to the crop's
    dimensions (a 64 px kernel over an 8-plane crop is ONE tile in Z, nothing
    like what the whole volume gets).  We do not clamp quietly and we do not
    change the whole-volume numerics — we say so, and we name the axis.
    """

    def _notes(self, recipe, crop, full, n=1):
        return controller._preview_fidelity_notes(recipe, n, crop, full)

    def test_bleach_over_a_partial_z_is_flagged(self):
        recipe = {"channels": [[{"key": "bleach", "params": {}}]],
                  "geometry": None}
        notes = self._notes(recipe, (16, 64, 64), (200, 512, 512))
        assert notes
        note = notes[0]
        # Plain language, with both numbers so the user can judge the gap.
        assert "Depth-brightness is measured across the slices it can see" in note
        assert "16 here" in note and "200 in the whole stack" in note
        assert _TAIL in note

    def test_bleach_over_the_full_z_is_not_flagged(self):
        recipe = {"channels": [[{"key": "bleach", "params": {}}]],
                  "geometry": None}
        assert self._notes(recipe, (200, 64, 64), (200, 512, 512)) == []

    def test_bleach_on_a_single_plane_region_says_it_does_nothing_here(self):
        recipe = {"channels": [[{"key": "bleach", "params": {}}]],
                  "geometry": None}
        notes = self._notes(recipe, (1, 64, 64), (200, 512, 512))
        assert notes and "does nothing" in notes[0]
        assert _TAIL in notes[0]

    def test_clahe_kernel_larger_than_the_region_names_the_axis(self):
        recipe = {"channels": [[{"key": "clahe",
                                 "params": {"kernel": 64, "clip": 0.01}}]],
                  "geometry": None}
        notes = self._notes(recipe, (8, 256, 256), (200, 512, 512))
        assert notes
        note = notes[0]
        # Names the tile size AND the axis, in words rather than jargon.
        assert "works in 64-pixel tiles" in note
        assert "thinner than that in Z" in note
        assert "treated as a single tile here" in note
        assert _TAIL in note

    def test_clahe_that_fits_the_region_is_not_flagged(self):
        recipe = {"channels": [[{"key": "clahe",
                                 "params": {"kernel": 8, "clip": 0.01}}]],
                  "geometry": None}
        assert self._notes(recipe, (32, 256, 256), (200, 512, 512)) == []

    def test_a_plain_recipe_has_nothing_to_say(self):
        recipe = {"channels": [[{"key": "gaussian", "params": {"sigma": 1.0}}]],
                  "geometry": None}
        assert self._notes(recipe, (8, 64, 64), (200, 512, 512)) == []

    def test_a_corrupt_recipe_is_silent_not_fatal(self):
        assert controller._preview_fidelity_notes(
            {"channels": "nonsense"}, 1, (8, 8, 8), (8, 8, 8)) == []
        assert controller._preview_fidelity_notes(
            None, 1, None, None) == []


class TestBoundsOnAwkwardArrays:
    """`_strided_percentile_bounds` flattens with `reshape(-1)`, which for a
    NON-CONTIGUOUS array (a transposed memmap — exactly what an axis-reordered
    TIFF gives you) is a full copy of the whole volume.  The preview must not
    pay that: `_whole_volume_norm_bounds` routes such an array through the same
    per-axis strided slice it uses for a lazy handle, so the read stays bounded.
    The apply-path helper is deliberately left alone."""

    def test_a_transposed_view_is_not_fully_copied(self, monkeypatch):
        vol = _volume().transpose(2, 1, 0)          # non-contiguous view
        assert not vol.flags.c_contiguous
        seen = {}
        real = controller._strided_percentile_bounds

        def _spy(arr, sample_cap=None):
            seen["size"] = int(np.asarray(arr).size)
            return real(arr) if sample_cap is None else real(arr, sample_cap)

        monkeypatch.setattr(controller, "_strided_percentile_bounds", _spy)
        # A cap well below the volume forces the slice path.
        out = controller._whole_volume_norm_bounds([vol], sample_cap=1000)
        assert out and out[0] is not None
        assert seen["size"] < vol.size,             "the whole non-contiguous array was materialised for a percentile"

    def test_a_contiguous_array_still_matches_the_apply_helper(self):
        vol = _volume()
        assert vol.flags.c_contiguous
        assert (controller._whole_volume_norm_bounds([vol])[0]
                == controller._strided_percentile_bounds(vol))

    def test_a_transposed_view_still_gives_a_usable_window(self):
        vol = _volume().transpose(2, 1, 0)
        lo, hi = controller._whole_volume_norm_bounds([vol])[0]
        full_lo, full_hi = controller._strided_percentile_bounds(
            np.ascontiguousarray(vol))
        # Sampled differently, so not identical — but a robust percentile over a
        # subsample must land in the same ballpark as the full pass.
        assert abs(lo - full_lo) < 0.05 * (full_hi - full_lo)
        assert abs(hi - full_hi) < 0.05 * (full_hi - full_lo)


class TestPreviewWindowMatchesTheStreamingEngine:
    """C#2 across the LARGE-DATA regime.

    A prepared source applies through `streaming.py`, which measures its
    normalize window from an exact streamed uint16 histogram
    (`_measure_bounds` -> `ops.percentile_bounds_from_hist`), not from
    `_strided_percentile_bounds`.  That is precisely the regime the C#8 warning
    is about, so the preview's window has to agree with THAT engine too — the
    in-memory equivalence test above does not cover it.
    """

    def _stream_bounds(self, vol, edge=16):
        from packages.epicycle_preprocessing import streaming as st

        def _reader(sl):
            return vol[sl]

        return st._measure_bounds(
            _reader, vol.shape, edge, 1, None, st._Progress(None))

    def test_the_preview_window_matches_the_streamed_window(self):
        vol = _volume()
        preview = controller._whole_volume_norm_bounds([vol])[0]
        streamed = self._stream_bounds(vol)
        assert preview is not None and streamed is not None
        # The streamed window is EXACT over uint16 DN; the preview measures the
        # same voxels but through float32 (`percentile_bounds` casts).  DN are
        # integers up to 65535, so the only legitimate gap is that cast's
        # rounding — well under one DN.  Anything larger means the two paths are
        # measuring different voxels, which is the C#2 defect in the large-data
        # regime.
        assert preview[0] == pytest.approx(streamed[0], abs=1.0)
        assert preview[1] == pytest.approx(streamed[1], abs=1.0)

    def test_block_size_does_not_move_the_streamed_window(self):
        """Sanity on the comparison itself: the streamed histogram is
        order/block independent, so the number above is a property of the
        volume, not of the tiling I happened to pick."""
        vol = _volume()
        assert self._stream_bounds(vol, edge=8) == self._stream_bounds(
            vol, edge=32)

    def test_a_flat_volume_is_degenerate_on_both_paths(self):
        flat = np.zeros((8, 16, 16), dtype=np.uint16)
        assert controller._whole_volume_norm_bounds([flat]) == [None]
        assert self._stream_bounds(flat) is None
