"""
Self-tests for the shared fixtures in ``tests/conftest.py``.

These guard the test *foundation*: every downstream stack (characterization,
Mode-B IoU stitch, tiling, axis) builds on ``make_tile_mask`` /
``two_overlapping_tiles``, so a drift here would silently invalidate many
tests. They assert the fixtures honor the engine's real on-disk contract and
that the straddle-cell co-occurrence (the property the IoU seam-stitch relies
on) actually holds.
"""

import json

import numpy as np
import tifffile

from packages.epicycle_scaledseg.core.assembly import parse_tile_positions


# ---------------------------------------------------------------------------
# make_tile_mask
# ---------------------------------------------------------------------------


def test_make_tile_mask_default_contract(make_tile_mask):
    """Default tile honors the Stage-2 mask contract: uint32 ZYX BigTIFF."""
    spec = make_tile_mask()

    assert spec.mask_path.exists()
    arr = tifffile.imread(str(spec.mask_path))
    assert arr.shape == (16, 16, 16)  # (Z, Y, X)
    assert arr.dtype == np.uint32

    # BigTIFF flag is set (Stage-2 always writes bigtiff=True).
    with tifffile.TiffFile(str(spec.mask_path)) as tf:
        assert tf.is_bigtiff


def test_make_tile_mask_zero_padded_to_full_z(make_tile_mask):
    """Labels live only inside truncated_range; the rest is zero-padded."""
    spec = make_tile_mask(z_range=(4, 12), full_z=16)
    arr = spec.array

    # Outside [4, 12) everything is background.
    assert arr[:4].sum() == 0
    assert arr[12:].sum() == 0
    # Inside the range the default label 1 is present.
    assert (arr[4:12] == 1).any()


def test_make_tile_mask_config_schema(make_tile_mask):
    """Per-tile config carries the exact Stage-0 tiling schema keys."""
    spec = make_tile_mask(row=2, col=3, patch_size=16, overlap=0.25)
    cfg = json.loads(spec.config_path.read_text())

    assert set(cfg) == {
        "tile_name",
        "row",
        "col",
        "truncated_range",
        "original_z",
        "patch_size",
        "step_size",
        "overlap",
    }
    assert cfg["tile_name"] == "tile_2_3"
    assert cfg["row"] == 2 and cfg["col"] == 3
    assert cfg["truncated_range"] == [4, 12]
    assert cfg["original_z"] == 16
    assert cfg["step_size"] == int(16 * (1 - 0.25))  # 12


def test_make_tile_mask_custom_labels(make_tile_mask):
    """Arbitrary label placement is honored across the full truncated Z."""
    spec = make_tile_mask(
        labels={5: (slice(0, 4), slice(0, 4)), 9: (slice(8, 12), slice(8, 12))},
        z_range=(2, 6),
        full_z=8,
    )
    arr = spec.array
    assert arr.shape == (8, 16, 16)
    assert set(np.unique(arr)) == {0, 5, 9}
    # Each label fills its (Y,X) block across exactly the truncated Z slices.
    assert (arr[2:6, 0:4, 0:4] == 5).all()
    assert (arr[2:6, 8:12, 8:12] == 9).all()


def test_make_tile_mask_origin_matches_parser(make_tile_mask):
    """TileSpec pixel origin equals what assembly's parse_tile_positions derives.

    Pins both an INDEPENDENT ground-truth origin (hand-derived: step =
    int(16*0.75) = 12, so row=3,col=2 -> (36, 24)) AND agreement with the real
    parser. The hardcoded value guards against a shared-formula error that a
    pure fixture-vs-parser consistency check would miss.
    """
    spec = make_tile_mask(row=3, col=2, patch_size=16, overlap=0.25)
    assert (spec.y0, spec.x0) == (36, 24)  # ground truth, not derived from fixture
    (y0, x0, _td) = parse_tile_positions(
        [spec.dir], tile_size=16, overlap=0.25
    )[0]
    assert (spec.y0, spec.x0) == (y0, x0)


def test_make_tile_mask_config_overrides(make_tile_mask):
    """config_overrides emits the reconstruction-input flavor (original_dimensions)."""
    spec = make_tile_mask(
        full_z=16, config_overrides={"original_dimensions": {"z": 16}}
    )
    cfg = json.loads(spec.config_path.read_text())
    # Stage-0 geometry keys are preserved...
    assert cfg["original_z"] == 16 and "step_size" in cfg
    # ...and the recon-readable key is now present (reconstruction.py:711 contract).
    assert cfg["original_dimensions"] == {"z": 16}


# ---------------------------------------------------------------------------
# two_overlapping_tiles
# ---------------------------------------------------------------------------


def test_two_tiles_distinct_names_and_grid(two_overlapping_tiles):
    t0, t1 = two_overlapping_tiles
    assert (t0.name, t1.name) == ("tile_0_0", "tile_0_1")
    assert (t0.row, t0.col) == (0, 0)
    assert (t1.row, t1.col) == (0, 1)


def test_two_tiles_geometry_self_consistent(two_overlapping_tiles):
    """Origins + step imply a 4px X overlap; parser agrees."""
    t0, t1 = two_overlapping_tiles
    assert t0.step_size == 12 and t1.step_size == 12
    assert t0.x0 == 0 and t1.x0 == 12

    positions = parse_tile_positions([t0.dir, t1.dir], tile_size=16, overlap=0.25)
    derived = {(y, x) for (y, x, _d) in positions}
    assert derived == {(0, 0), (0, 12)}

    # Overlap in global X = [x1_0_end - x0_1] = [16 - 12] = 4 px.
    overlap_px = (t0.x0 + t0.patch_size) - t1.x0
    assert overlap_px == 4


def test_two_tiles_straddle_cell_cooccurs_in_overlap(two_overlapping_tiles):
    """Label 3 occupies the SAME global voxels in both tiles' overlap zone.

    This is the load-bearing property for the Mode-B IoU seam-stitch: the
    straddle cell must have non-zero intersection in global coordinates so an
    IoU matcher can fuse the two local instances into one.
    """
    t0, t1 = two_overlapping_tiles

    # Global overlap zone in X is [12, 16).
    # tile_0_0 (origin 0): global X[12,16) == local x[12,16).
    t0_overlap = t0.array[:, :, 12:16]
    # tile_0_1 (origin 12): global X[12,16) == local x[0,4).
    t1_overlap = t1.array[:, :, 0:4]

    # In both views the straddle label 3 is present on the SAME (Z,Y) voxels.
    t0_has3 = t0_overlap == 3
    t1_has3 = t1_overlap == 3
    assert t0_has3.any() and t1_has3.any()
    # Co-occurrence: identical voxel mask across the seam (perfect IoU == 1.0).
    assert np.array_equal(t0_has3, t1_has3)
    inter = np.logical_and(t0_has3, t1_has3).sum()
    union = np.logical_or(t0_has3, t1_has3).sum()
    assert inter > 0 and inter == union  # IoU == 1.0


def test_two_tiles_singletons_outside_overlap(two_overlapping_tiles):
    """Labels 1 and 2 are distinct cells that never enter the seam."""
    t0, t1 = two_overlapping_tiles

    # Label 1 only in tile0, and only at local x[2,6) -> global X[2,6), < 12.
    assert (t0.array == 1).any()
    assert not (t1.array == 1).any()
    assert not (t0.array[:, :, 12:16] == 1).any()

    # Label 2 only in tile1, at local x[10,14) -> global X[22,26), >= 16.
    assert (t1.array == 2).any()
    assert not (t0.array == 2).any()
    assert not (t1.array[:, :, 0:4] == 2).any()


def test_two_tiles_share_z_window(two_overlapping_tiles):
    """Both tiles share full_z and truncated range, so the seam co-occurs in Z."""
    t0, t1 = two_overlapping_tiles
    assert t0.full_z == t1.full_z == 16
    assert t0.z_range == t1.z_range == (4, 12)
