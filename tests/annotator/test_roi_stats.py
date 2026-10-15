"""Golden tests: bounded ROI stats must equal the full-volume computation.

The Step-3 accept path used to compute `area` + `centroid` for every accepted
label / de-novo piece with a FULL-VOLUME `data == label_id` + `np.argwhere`.
These tests pin that the bounded replacements return byte-identical numbers.
"""

import numpy as np
import pytest

from packages.epicycle_annotator.roi_stats import (
    label_stats_in_roi,
    piece_stats_from_write,
)


def _full_volume_stats(volume, label_id):
    """The exact pre-optimisation computation (controller._update_label_basic_stats)."""
    mask = volume == label_id
    if not np.any(mask):
        return None
    area = int(np.sum(mask))
    coords = np.argwhere(mask)
    return area, coords.mean(axis=0)


def _roi_origin(roi):
    return tuple(int(s.start) for s in roi)


# --------------------------------------------------------------------------
# label_stats_in_roi
# --------------------------------------------------------------------------

def test_label_stats_match_full_volume_simple():
    vol = np.zeros((20, 24, 26), dtype=np.uint16)
    vol[5:9, 6:11, 7:13] = 7
    roi = (slice(2, 14), slice(3, 18), slice(4, 20))

    got = label_stats_in_roi(vol[roi], 7, _roi_origin(roi))
    want = _full_volume_stats(vol, 7)

    assert got[0] == want[0]
    assert np.array_equal(got[1], want[1])  # byte-identical, not approx


def test_label_stats_match_full_volume_touching_roi_edge():
    vol = np.zeros((20, 24, 26), dtype=np.uint16)
    # Label sits flush against every face of the ROI.
    vol[4:8, 5:9, 6:10] = 3
    roi = (slice(4, 8), slice(5, 9), slice(6, 10))

    got = label_stats_in_roi(vol[roi], 3, _roi_origin(roi))
    want = _full_volume_stats(vol, 3)
    assert got[0] == want[0]
    assert np.array_equal(got[1], want[1])


def test_label_stats_roi_at_origin():
    vol = np.zeros((12, 12, 12), dtype=np.uint16)
    vol[0:3, 0:4, 0:5] = 9
    roi = (slice(0, 6), slice(0, 6), slice(0, 6))
    got = label_stats_in_roi(vol[roi], 9, _roi_origin(roi))
    want = _full_volume_stats(vol, 9)
    assert got[0] == want[0]
    assert np.array_equal(got[1], want[1])


def test_label_stats_absent_label_returns_none():
    vol = np.zeros((8, 8, 8), dtype=np.uint16)
    vol[1:3, 1:3, 1:3] = 5
    roi = (slice(4, 8), slice(4, 8), slice(4, 8))
    assert label_stats_in_roi(vol[roi], 5, _roi_origin(roi)) is None
    assert label_stats_in_roi(vol, 12345, (0, 0, 0)) is None


def test_label_stats_are_roi_scoped_when_label_leaks_outside():
    """PINNED SEMANTICS: voxels outside the ROI are NOT counted.

    This is why merge (whose ROI is the absorbed labels' bbox and may miss part
    of the keep label) must keep the full-volume path, while interp/smooth --
    whose ROI is the label's own bounding box + margin -- may use the crop.
    """
    vol = np.zeros((20, 20, 20), dtype=np.uint16)
    vol[2:4, 2:4, 2:4] = 4          # inside the ROI
    vol[15:17, 15:17, 15:17] = 4    # a second blob, OUTSIDE the ROI
    roi = (slice(0, 8), slice(0, 8), slice(0, 8))

    bounded = label_stats_in_roi(vol[roi], 4, _roi_origin(roi))
    full = _full_volume_stats(vol, 4)

    assert bounded[0] == 8          # only the in-ROI blob
    assert full[0] == 16
    assert not np.array_equal(bounded[1], full[1])

    # ...and when the ROI DOES contain the whole label, they agree exactly.
    roi_all = (slice(0, 20), slice(0, 20), slice(0, 20))
    both = label_stats_in_roi(vol[roi_all], 4, _roi_origin(roi_all))
    assert both[0] == full[0]
    assert np.array_equal(both[1], full[1])


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_label_stats_random_blobs_match_full_volume(seed):
    rng = np.random.default_rng(seed)
    vol = np.zeros((31, 29, 33), dtype=np.uint16)
    mask = rng.random(vol.shape) < 0.02
    vol[mask] = 11
    coords = np.argwhere(vol == 11)
    lo = coords.min(axis=0)
    hi = coords.max(axis=0) + 1
    roi = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))

    got = label_stats_in_roi(vol[roi], 11, _roi_origin(roi))
    want = _full_volume_stats(vol, 11)
    assert got[0] == want[0]
    assert np.array_equal(got[1], want[1])


# --------------------------------------------------------------------------
# piece_stats_from_write
# --------------------------------------------------------------------------

def _commit_and_reference(vol, result_roi, roi):
    """Perform the real Add safe-write, return (committed, stats, reference)."""
    target = vol[roi]
    write_mask = (result_roi > 0) & (target == 0)
    committed = [int(v) for v in np.unique(result_roi[write_mask]) if v != 0]
    target[write_mask] = result_roi[write_mask]
    reference = {pid: _full_volume_stats(vol, pid) for pid in committed}
    stats = piece_stats_from_write(result_roi, write_mask, _roi_origin(roi))
    return committed, stats, reference


def test_piece_stats_match_full_volume_scan():
    vol = np.zeros((24, 26, 28), dtype=np.uint16)
    roi = (slice(4, 16), slice(5, 18), slice(6, 20))
    result = np.zeros(vol[roi].shape, dtype=np.uint16)
    result[0:3, 0:4, 0:5] = 100      # flush against the ROI corner
    result[5:8, 6:9, 7:11] = 101
    result[9:12, 10:13, 11:14] = 102

    committed, stats, reference = _commit_and_reference(vol, result, roi)
    assert set(committed) == {100, 101, 102}
    for pid in committed:
        assert stats[pid][0] == reference[pid][0]
        assert np.array_equal(stats[pid][1], reference[pid][1])


def test_piece_stats_only_count_written_voxels():
    """Pieces overlapping an existing label are partially written; the stats
    must reflect the WRITTEN voxels only, exactly like a post-commit scan."""
    vol = np.zeros((20, 20, 20), dtype=np.uint16)
    vol[6:10, 6:10, 6:10] = 55       # an existing cell occupying part of the box
    roi = (slice(4, 16), slice(4, 16), slice(4, 16))
    result = np.zeros(vol[roi].shape, dtype=np.uint16)
    result[2:8, 2:8, 2:8] = 200      # overlaps the existing 55 block

    committed, stats, reference = _commit_and_reference(vol, result, roi)
    assert committed == [200]
    assert stats[200][0] == reference[200][0]
    assert np.array_equal(stats[200][1], reference[200][1])
    assert stats[200][0] < 6 ** 3    # the overlap really was clipped


def test_piece_stats_skip_fully_blocked_piece():
    """A piece whose voxels are all taken contributes nothing (zero size)."""
    vol = np.zeros((16, 16, 16), dtype=np.uint16)
    vol[5:9, 5:9, 5:9] = 44
    roi = (slice(4, 12), slice(4, 12), slice(4, 12))
    result = np.zeros(vol[roi].shape, dtype=np.uint16)
    result[1:5, 1:5, 1:5] = 300      # exactly the occupied block
    result[6:8, 6:8, 6:8] = 301      # free

    committed, stats, reference = _commit_and_reference(vol, result, roi)
    assert committed == [301]
    assert 300 not in stats
    assert stats[301][0] == reference[301][0]
    assert np.array_equal(stats[301][1], reference[301][1])


def test_piece_stats_empty_write_mask():
    result = np.zeros((4, 4, 4), dtype=np.uint16)
    mask = np.zeros((4, 4, 4), dtype=bool)
    assert piece_stats_from_write(result, mask, (1, 2, 3)) == {}


def test_piece_stats_many_pieces_match_full_volume():
    rng = np.random.default_rng(7)
    vol = np.zeros((30, 32, 34), dtype=np.uint16)
    roi = (slice(3, 27), slice(4, 28), slice(5, 30))
    sub_shape = vol[roi].shape
    result = np.zeros(sub_shape, dtype=np.uint16)
    next_id = 900
    for _ in range(12):
        z = int(rng.integers(0, sub_shape[0] - 4))
        y = int(rng.integers(0, sub_shape[1] - 4))
        x = int(rng.integers(0, sub_shape[2] - 4))
        result[z:z + 3, y:y + 3, x:x + 3] = next_id
        next_id += 1

    committed, stats, reference = _commit_and_reference(vol, result, roi)
    assert committed
    for pid in committed:
        assert stats[pid][0] == reference[pid][0]
        assert np.array_equal(stats[pid][1], reference[pid][1])


def test_piece_stats_handles_very_large_label_ids():
    """Binning must be by dense slot, not by raw id (ids can be huge)."""
    vol = np.zeros((12, 12, 12), dtype=np.uint32)
    roi = (slice(2, 10), slice(2, 10), slice(2, 10))
    result = np.zeros(vol[roi].shape, dtype=np.uint32)
    big = 4_000_000_000
    result[1:4, 1:4, 1:4] = big

    committed, stats, reference = _commit_and_reference(vol, result, roi)
    assert committed == [big]
    assert stats[big][0] == reference[big][0]
    assert np.array_equal(stats[big][1], reference[big][1])


# --------------------------------------------------------------------------
# label_escapes_roi  (containment tripwire)
# --------------------------------------------------------------------------

def _escapes(vol, label_id, roi):
    from packages.epicycle_annotator.roi_stats import label_escapes_roi
    return label_escapes_roi(vol[roi], label_id, roi, vol.shape)


def test_escapes_false_for_a_label_well_inside_the_roi():
    vol = np.zeros((20, 20, 20), dtype=np.uint16)
    vol[6:9, 6:9, 6:9] = 5
    assert _escapes(vol, 5, (slice(2, 15), slice(2, 15), slice(2, 15))) is False


def test_escapes_true_on_each_unclipped_face():
    for axis in range(3):
        for side in ("lo", "hi"):
            vol = np.zeros((20, 20, 20), dtype=np.uint16)
            roi = [slice(5, 15), slice(5, 15), slice(5, 15)]
            blob = [slice(7, 9), slice(7, 9), slice(7, 9)]
            blob[axis] = slice(5, 7) if side == "lo" else slice(13, 15)
            vol[tuple(blob)] = 6
            assert _escapes(vol, 6, tuple(roi)) is True, (axis, side)


def test_escapes_false_when_the_face_is_the_volume_edge():
    """A CLIPPED face cannot hide voxels — the bounded path stays valid."""
    vol = np.zeros((20, 20, 20), dtype=np.uint16)
    vol[0:3, 0:3, 0:3] = 7          # flush with the volume corner
    vol[17:20, 17:20, 17:20] = 7    # and the opposite corner
    roi = (slice(0, 20), slice(0, 20), slice(0, 20))
    assert _escapes(vol, 7, roi) is False


def test_escapes_ignores_other_labels():
    vol = np.zeros((20, 20, 20), dtype=np.uint16)
    vol[5:7, 5:7, 5:7] = 99         # a neighbour flush with the ROI face
    vol[8:10, 8:10, 8:10] = 4
    assert _escapes(vol, 4, (slice(5, 15), slice(5, 15), slice(5, 15))) is False
