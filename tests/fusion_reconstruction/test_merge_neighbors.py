"""Tests for `napari_epicycle.fusion_reconstruction.neighbor_query`.

Pure-Python (numpy + scipy), Qt-free. Covers the algorithmic correctness
required by the merge dialog spec:

- Adjacent labels are discovered (touching → first_radius=1).
- Gap bridging works (labels separated by N voxels found at radius N).
- Truly isolated label → empty neighbor list (handled by dialog as
  empty-table-with-placeholder, not an error).
- Self-exclusion (target label not in its own neighbor list).
- Background-exclusion (label 0 not in neighbor list).
- Sort ordering: (first_radius asc, contact_voxels desc).
- max_radius parameter respected (default 3).
- Missing label → empty list, not error.
"""
import numpy as np
import pytest

from napari_epicycle.fusion_reconstruction.neighbor_query import (
    get_neighbor_ids,
)


# -------- helpers ---------------------------------------------------


def _two_touching_cubes():
    """Two 4x4x4 cubes labeled 1 and 2, touching face-to-face."""
    vol = np.zeros((10, 10, 10), dtype=np.int32)
    vol[2:6, 2:6, 2:6] = 1
    vol[2:6, 2:6, 6:10] = 2  # touches label 1 along x=6 plane
    return vol


def _three_in_a_row(gap: int = 0):
    """Three cubes labeled 1, 2, 3, in a row along x.

    With gap=0 they're all touching. With gap>0 there are *gap*
    background voxels between consecutive cubes.
    """
    vol = np.zeros((10, 10, 30), dtype=np.int32)
    x0 = 2
    for label_id in (1, 2, 3):
        vol[2:6, 2:6, x0:x0 + 4] = label_id
        x0 += 4 + gap
    return vol


def _isolated_label():
    """One small cube in the middle of an otherwise empty volume."""
    vol = np.zeros((10, 10, 10), dtype=np.int32)
    vol[4:6, 4:6, 4:6] = 1
    return vol


# -------- core behaviors --------------------------------------------


def test_two_touching_cubes_each_finds_the_other():
    vol = _two_touching_cubes()
    n1 = get_neighbor_ids(vol, 1)
    n2 = get_neighbor_ids(vol, 2)
    assert len(n1) == 1
    assert n1[0]["label_id"] == 2
    assert n1[0]["first_radius"] == 1
    assert len(n2) == 1
    assert n2[0]["label_id"] == 1
    assert n2[0]["first_radius"] == 1


def test_isolated_label_returns_empty_list():
    """Per merge spec: empty neighbor list opens dialog with placeholder
    text, no error. Function must return [] not raise."""
    vol = _isolated_label()
    result = get_neighbor_ids(vol, 1)
    assert result == []


def test_missing_label_returns_empty_list():
    """A label_id not present in the volume must return [] gracefully."""
    vol = _two_touching_cubes()
    result = get_neighbor_ids(vol, 999)
    assert result == []


def test_label_zero_returns_empty_list():
    """label_id=0 is background; returns empty list (not an error)."""
    vol = _two_touching_cubes()
    result = get_neighbor_ids(vol, 0)
    assert result == []


def test_negative_label_id_returns_empty_list():
    vol = _two_touching_cubes()
    result = get_neighbor_ids(vol, -1)
    assert result == []


def test_self_exclusion():
    """Target label must NOT appear in its own neighbor list."""
    vol = _two_touching_cubes()
    n1 = get_neighbor_ids(vol, 1)
    assert all(d["label_id"] != 1 for d in n1)


def test_background_exclusion():
    """Background (label 0) must NOT appear as a 'neighbor' even though
    the dilation hits background voxels."""
    vol = _isolated_label()
    # Add a touching neighbor to make it non-empty.
    vol[4:6, 4:6, 6:8] = 2
    n1 = get_neighbor_ids(vol, 1)
    assert all(d["label_id"] != 0 for d in n1)


# -------- gap bridging ---------------------------------------------


def test_gap_bridging_at_radius_1():
    """Touching labels are found at first_radius=1."""
    vol = _three_in_a_row(gap=0)
    n2 = get_neighbor_ids(vol, 2)
    assert len(n2) == 2
    radii = {d["label_id"]: d["first_radius"] for d in n2}
    assert radii[1] == 1
    assert radii[3] == 1


def test_gap_bridging_at_radius_2():
    """Labels separated by 1 voxel of background are found at first_radius=2."""
    vol = _three_in_a_row(gap=1)
    n2 = get_neighbor_ids(vol, 2)
    assert len(n2) == 2
    radii = {d["label_id"]: d["first_radius"] for d in n2}
    assert radii[1] == 2
    assert radii[3] == 2


def test_gap_beyond_max_radius_excluded():
    """Labels separated by more than max_radius voxels are NOT found."""
    vol = _three_in_a_row(gap=5)  # 5-voxel gap > default max_radius=3
    n2 = get_neighbor_ids(vol, 2, max_radius=3)
    # With gap=5 and a 4-voxel cube, dilation of label 2 by 3 voxels
    # reaches into the gap but cannot bridge to label 1 or 3 (dilation
    # only travels 3 voxels in one direction; gap is 5).
    assert n2 == []


def test_gap_beyond_default_but_within_explicit_radius():
    """A wider max_radius reaches further-away neighbors."""
    vol = _three_in_a_row(gap=3)
    # Default max_radius=3 should NOT bridge a 3-voxel gap (need r=4).
    assert get_neighbor_ids(vol, 2, max_radius=3) == []
    # max_radius=5 should bridge the 3-voxel gap (found at r=4).
    n2_wide = get_neighbor_ids(vol, 2, max_radius=5)
    assert len(n2_wide) == 2
    radii = {d["label_id"]: d["first_radius"] for d in n2_wide}
    assert radii[1] == 4
    assert radii[3] == 4


# -------- sort ordering -------------------------------------------


def test_sort_by_first_radius_then_contact_desc():
    """Sort key: (first_radius asc, contact_voxels desc).

    Construction: label 2 has three neighbors:
    - label 1 directly touching with a small contact patch
    - label 3 directly touching with a larger contact patch
    - label 4 separated by 2-voxel gap (large contact)
    Expected order: 3 (r=1, large), 1 (r=1, small), 4 (r=3, any).
    """
    vol = np.zeros((20, 20, 20), dtype=np.int32)
    # Central label 2 occupies a 4x4x4 cube.
    vol[8:12, 8:12, 8:12] = 2

    # Label 3: large touching contact face on +X.
    vol[8:12, 8:12, 12:16] = 3
    # Label 1: small touching contact face on -X (only 2x2 voxels of contact).
    vol[8:10, 8:10, 4:8] = 1
    # Label 4: larger contact area but separated by a 2-voxel gap.
    vol[8:12, 4:8, 8:12] = 0  # ensure gap region clean
    vol[14:18, 8:12, 8:12] = 4   # along +Y, with 2-voxel gap at rows 12-13

    result = get_neighbor_ids(vol, 2, max_radius=3)
    # All three should be found.
    found_ids = [d["label_id"] for d in result]
    assert set(found_ids) == {1, 3, 4}

    # Sort assertions: r=1 entries before r>1.
    radii = [d["first_radius"] for d in result]
    assert radii == sorted(radii)

    # Within r=1, larger contact first (label 3 should precede label 1).
    r1 = [d for d in result if d["first_radius"] == 1]
    assert len(r1) == 2
    assert r1[0]["label_id"] == 3
    assert r1[0]["contact_voxels"] >= r1[1]["contact_voxels"]


# -------- input validation -----------------------------------------


def test_invalid_max_radius_raises():
    vol = _two_touching_cubes()
    with pytest.raises(ValueError, match="max_radius must be >= 1"):
        get_neighbor_ids(vol, 1, max_radius=0)


def test_negative_max_radius_raises():
    vol = _two_touching_cubes()
    with pytest.raises(ValueError, match="max_radius must be >= 1"):
        get_neighbor_ids(vol, 1, max_radius=-1)


# -------- return shape -------------------------------------------


def test_return_dict_keys():
    """Each result dict has exactly the keys promised by the docstring."""
    vol = _two_touching_cubes()
    result = get_neighbor_ids(vol, 1)
    for d in result:
        assert set(d.keys()) == {"label_id", "contact_voxels", "first_radius"}
        assert isinstance(d["label_id"], int)
        assert isinstance(d["contact_voxels"], int)
        assert isinstance(d["first_radius"], int)
        assert d["contact_voxels"] > 0
        assert d["first_radius"] >= 1
