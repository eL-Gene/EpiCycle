"""Pure-logic tests for the post-stitch cleanup + smoothing core.

No GPU / Cellpose / u-Segment3D — numpy + scipy (+ a lazy skimage path only when
smoothing splits a label, not exercised here)."""

import numpy as np
import pytest

from packages.epicycle_scaledseg.core.cleanup import (
    select_dropped_by_size,
    select_dropped_by_max_size,
    select_dropped_islands,
    build_relabel_lut,
    lut_to_array,
)
from packages.epicycle_scaledseg.core.smooth import resolve_sigmas, smooth_label_sdf, smooth_instance_mask


def _row(dense_id, voxel_count, bbox):
    z0, z1, y0, y1, x0, x1 = bbox
    return {
        "dense_id": dense_id, "voxel_count": voxel_count,
        "z_min": z0, "z_max": z1, "y_min": y0, "y_max": y1, "x_min": x0, "x_max": x1,
    }


# --------------------------------------------------------------------------- #
# Size filter
# --------------------------------------------------------------------------- #

class TestSelectDroppedBySize:
    def test_drops_below_threshold(self):
        rows = [_row(1, 50, (0, 1, 0, 1, 0, 1)),
                _row(2, 5000, (0, 9, 0, 9, 0, 9)),
                _row(3, 199, (0, 2, 0, 2, 0, 2))]
        assert select_dropped_by_size(rows, 200) == {1, 3}

    def test_boundary_is_kept(self):
        # voxel_count == min_voxels is KEPT (strict <).
        assert select_dropped_by_size([_row(1, 200, (0, 1, 0, 1, 0, 1))], 200) == set()

    def test_disabled_when_zero(self):
        assert select_dropped_by_size([_row(1, 1, (0, 1, 0, 1, 0, 1))], 0) == set()

    def test_csv_string_values(self):
        # Values arrive as strings from a CSV read — must still compare numerically.
        rows = [{"dense_id": "1", "voxel_count": "50"},
                {"dense_id": "2", "voxel_count": "300"}]
        assert select_dropped_by_size(rows, 200) == {1}


class TestSelectDroppedByMaxSize:
    def test_drops_above_threshold(self):
        rows = [_row(1, 50, (0, 1, 0, 1, 0, 1)),
                _row(2, 5000, (0, 9, 0, 9, 0, 9)),
                _row(3, 100000, (0, 40, 0, 40, 0, 40))]
        assert select_dropped_by_max_size(rows, 10000) == {3}

    def test_boundary_is_kept(self):
        # voxel_count == max_voxels is KEPT (strict >).
        assert select_dropped_by_max_size([_row(1, 10000, (0, 1, 0, 1, 0, 1))], 10000) == set()

    def test_disabled_when_zero(self):
        assert select_dropped_by_max_size([_row(1, 999999, (0, 1, 0, 1, 0, 1))], 0) == set()


# --------------------------------------------------------------------------- #
# Detached-island filter
# --------------------------------------------------------------------------- #

class TestSelectDroppedIslands:
    def test_far_island_dropped(self):
        # One big tissue blob (idx 5..10) + one tiny island in the far corner (idx 0).
        rows = [_row(1, 100000, (40, 80, 40, 80, 40, 80)),
                _row(2, 50, (0, 7, 0, 7, 0, 7))]
        dropped = select_dropped_islands(
            rows, (96, 96, 96), downsample=8, min_component_frac=0.1, dilate=0
        )
        assert dropped == {2}

    def test_two_lobes_both_kept(self):
        # Two separated big blobs — multi-lobe tissue — must BOTH survive.
        rows = [_row(1, 80000, (8, 40, 8, 40, 8, 40)),
                _row(2, 80000, (56, 88, 56, 88, 56, 88))]
        dropped = select_dropped_islands(
            rows, (96, 96, 96), downsample=8, min_component_frac=0.1, dilate=0
        )
        assert dropped == set()

    def test_single_blob_none_dropped(self):
        rows = [_row(1, 100000, (40, 80, 40, 80, 40, 80))]
        assert select_dropped_islands(rows, (96, 96, 96), downsample=8, dilate=0) == set()

    def test_excluded_ids_ignored(self):
        # An id already dropped by size is not reconsidered (and not "kept" by the blob).
        rows = [_row(1, 100000, (40, 80, 40, 80, 40, 80)),
                _row(2, 50, (0, 7, 0, 7, 0, 7))]
        dropped = select_dropped_islands(
            rows, (96, 96, 96), downsample=8, min_component_frac=0.1, dilate=0,
            exclude={2},
        )
        assert dropped == set()

    def test_dilate_bridges_close_blobs(self):
        # Two blobs one grid-cell apart: dilate=1 bridges them into one component.
        rows = [_row(1, 4000, (0, 24, 0, 24, 0, 24)),     # idx 0..3
                _row(2, 4000, (40, 64, 40, 64, 40, 64))]  # idx 5..8
        # gap at idx 4 -> dilate=1 fills it -> one component -> nothing dropped.
        dropped = select_dropped_islands(
            rows, (72, 72, 72), downsample=8, min_component_frac=0.1, dilate=1
        )
        assert dropped == set()


# --------------------------------------------------------------------------- #
# Relabel LUT
# --------------------------------------------------------------------------- #

class TestRelabelLut:
    def test_dense_renumber(self):
        lut = build_relabel_lut([1, 2, 3, 4, 5], {2, 4})
        assert lut == {1: 1, 3: 2, 5: 3}

    def test_empty_when_all_dropped(self):
        assert build_relabel_lut([1, 2], {1, 2}) == {}

    def test_lut_array_maps_and_zeros(self):
        lut = build_relabel_lut([1, 2, 3], {2})
        arr = lut_to_array(lut, max_id=3)
        assert arr.dtype == np.int32
        assert arr[0] == 0          # background
        assert arr[1] == 1
        assert arr[2] == 0          # dropped
        assert arr[3] == 2
        # Round-trip: applying the LUT to a label slice renumbers densely.
        sl = np.array([[0, 1, 2], [3, 3, 0]], dtype=np.int32)
        assert arr[sl].tolist() == [[0, 1, 0], [2, 2, 0]]


# --------------------------------------------------------------------------- #
# Smoothing (mirrored Step-3 SDF smooth)
# --------------------------------------------------------------------------- #

class TestSmooth:
    def test_resolve_sigmas_isotropic(self):
        assert resolve_sigmas(1.5) == (1.5, 1.5)

    def test_resolve_sigmas_overrides(self):
        assert resolve_sigmas(1.5, sigma_z=3.0) == (3.0, 1.5)
        assert resolve_sigmas(1.5, sigma_xy=0.8) == (1.5, 0.8)

    def test_smooth_keeps_label_and_connectivity(self):
        from scipy.ndimage import label as nd_label
        vol = np.zeros((20, 20, 20), dtype=np.int32)
        vol[5:15, 5:15, 5:15] = 7  # a solid cube
        result, n_mod, _msg = smooth_label_sdf(vol, 7, sigma_z=1.5, sigma_xy=1.5)
        mask = result == 7
        assert mask.any()                         # didn't vanish
        assert set(np.unique(result)).issubset({0, 7})  # only the target label
        _, n_comp = nd_label(mask)
        assert n_comp == 1                        # stays a single connected component

    def test_smooth_missing_label_is_noop(self):
        vol = np.zeros((8, 8, 8), dtype=np.int32)
        result, n_mod, _msg = smooth_label_sdf(vol, 99)
        assert n_mod == 0 and not (result == 99).any()

    def test_smooth_instance_mask_returns_bool(self):
        vol = np.zeros((16, 16, 16), dtype=np.int32)
        vol[4:12, 4:12, 4:12] = 3
        m = smooth_instance_mask(vol, 3, sigma=1.5)
        assert m.dtype == bool and m.any()
