"""Tests for packages.epicycle_annotator.refine — label refinement algorithms.

Pure algorithm tests (no Qt, no napari).

REFINE Brick B2: ``morphgac_refine`` and ``morphacwe_refine`` are no longer
methods of their own -- they are the two private bodies of
``contour_refine`` (the merged Active contour), called with an explicit gate
and ``distance_clamp=0``.  They keep their signatures, so the cases below
still pin them directly; ``morphacwe_refine`` gained one ADDITIVE keyword,
``band_mask``, pinned in ``TestMorphacweBand``.  The merged method itself is
``test_refine_contour.py`` and its serial-2D driver ``test_refine_serial_2d.py``.
"""

import numpy as np
import pytest

from packages.epicycle_annotator.refine import (
    _make_axis_structure,
    build_edge_stopping,
    two_sided_limit,
    dilate_erode_label,
    fill_holes_label,
    flood_fill_label,
    guided_grow,
    morphacwe_refine,
    morphgac_refine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _axis_extent(mask: np.ndarray, axis: int) -> int:
    """Return the extent (max - min + 1) of True voxels along *axis*."""
    proj = np.any(mask, axis=tuple(i for i in range(mask.ndim) if i != axis))
    indices = np.where(proj)[0]
    if indices.size == 0:
        return 0
    return int(indices[-1] - indices[0] + 1)


def _make_sphere(shape=(30, 30, 30), label_id=3, center=(15, 15, 15), radius=8):
    """Create a volume with a spherical label."""
    vol = np.zeros(shape, dtype=np.int32)
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist = np.sqrt(
        (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2
    )
    vol[dist <= radius] = label_id
    return vol


def _make_dented_sphere(
    shape=(30, 30, 30), label_id=3, center=(15, 15, 15), radius=8
):
    """Create a sphere with a dent removed (simulates deformed label)."""
    vol = _make_sphere(shape, label_id, center, radius)
    # Remove a wedge to create a dent on one side.
    vol[12:18, 12:18, center[2] + radius - 4 : center[2] + radius + 1] = 0
    return vol


def _make_edge_image(vol, label_id=3):
    """Create a synthetic edge image: strong edge at label boundary."""
    from scipy.ndimage import distance_transform_edt

    mask = vol == label_id
    # Create a smooth sphere boundary as the "membrane" signal.
    dist_in = distance_transform_edt(mask)
    dist_out = distance_transform_edt(~mask)
    # Membrane = thin shell at boundary (high intensity = bright membrane).
    membrane = np.exp(-0.5 * (dist_in + dist_out - 1) ** 2)
    return membrane.astype(np.float64)


def _make_intensity_image(vol, label_id=3, inside_val=0.8, outside_val=0.2):
    """Create a synthetic intensity image with inside/outside contrast."""
    mask = vol == label_id
    img = np.full(vol.shape, outside_val, dtype=np.float64)
    img[mask] = inside_val
    return img


# ===========================================================================
# TestBuildEdgeStopping
# ===========================================================================

class TestBuildEdgeStopping:
    """`build_edge_stopping` after the edge-stopping rework.

    It no longer takes a raw channel and no longer blends a probability
    gradient: it takes the SHARED wall ELEVATION (high = wall, already in
    [0, 1]) and is the transfer function 1 / (1 + k E^2) onto it.  The old
    "Image weight" alpha is retired -- it was a measured no-op without a
    probability file -- so its blend case is REWRITTEN as the sharpness
    case, not deleted.
    """

    def test_output_shape_matches_input(self):
        elev = np.random.rand(20, 20, 20)
        gimage = build_edge_stopping(elev)
        assert gimage.shape == elev.shape

    def test_values_in_zero_one_range(self):
        elev = np.random.rand(20, 20, 20)
        gimage = build_edge_stopping(elev)
        assert gimage.min() >= 0.0 - 1e-10
        assert gimage.max() <= 1.0 + 1e-10

    def test_a_flat_elevation_gives_high_values(self):
        """No wall anywhere -> gimage == 1.0, so nothing stops the contour."""
        gimage = build_edge_stopping(np.zeros((20, 20, 20)))
        assert np.allclose(gimage, 1.0)

    def test_a_wall_drops_it_to_about_a_hundredth(self):
        """The whole point of the form: the two ends are pushed apart, so
        an explicit gate anywhere in (0, 1) separates them cleanly."""
        gimage = build_edge_stopping(np.ones((8, 8, 8)), sharpness=100.0)
        assert gimage.max() < 0.02

    def test_it_never_reaches_exactly_zero(self):
        """A morphological GAC that reaches 0 is frozen there for the rest
        of the run."""
        gimage = build_edge_stopping(np.ones((4, 4, 4)) * 5.0)
        assert float(gimage.min()) > 0.0

    def test_sharpness_is_the_knee_and_it_does_something(self):
        elev = np.full((6, 6, 6), 0.3, np.float32)
        soft = build_edge_stopping(elev, sharpness=1.0)
        hard = build_edge_stopping(elev, sharpness=1000.0)
        assert soft.mean() > hard.mean()

    def test_it_is_monotone_decreasing_in_the_elevation(self):
        elev = np.linspace(0.0, 1.0, 64).reshape(4, 4, 4)
        g = build_edge_stopping(elev).ravel()
        assert np.all(np.diff(g) <= 1e-7)


# ===========================================================================
# TestTwoSidedLimit
# ===========================================================================

class TestTwoSidedLimit:
    """The second line is the one that did not exist."""

    def _shell(self):
        original = np.zeros((12, 20, 20), bool)
        original[3:9, 6:14, 6:14] = True
        return original

    def test_growth_is_capped(self):
        original = self._shell()
        grown = np.zeros_like(original)
        grown[1:11, 4:16, 4:16] = True      # 2 voxels out on every side
        out, hit, _mean, mx = two_sided_limit(
            original, grown, limit_um=1.0, spacing=(1.0, 1.0, 1.0))
        assert hit is True
        assert mx <= 1.0 + 1e-6
        assert out.sum() < grown.sum()

    def test_collapse_is_restrained_which_the_old_clamp_could_not_do(self):
        original = self._shell()
        collapsed = np.zeros_like(original)
        collapsed[5:7, 9:11, 9:11] = True   # a 4-voxel-deep collapse
        out, hit, _mean, mx = two_sided_limit(
            original, collapsed, limit_um=1.0, spacing=(1.0, 1.0, 1.0))
        assert hit is True
        assert out.sum() > collapsed.sum(), (
            "the one-sided clamp let this through untouched")
        assert mx <= 1.0 + 1e-6

    def test_the_old_one_sided_expression_is_vacuous_inside_the_footprint(self):
        """The measurement that motivated 4.2: `edt(~original)` is zero
        everywhere inside the original, so the old guard was `True`."""
        from scipy.ndimage import distance_transform_edt

        original = self._shell()
        d = distance_transform_edt(~original)
        assert float(d[original].max()) == 0.0

    def test_the_limit_is_physical(self):
        original = self._shell()
        grown = np.zeros_like(original)
        grown[0:12, 3:17, 3:17] = True
        coarse, _, _, _ = two_sided_limit(
            original, grown, limit_um=3.0, spacing=(3.0, 1.0, 1.0))
        fine, _, _, _ = two_sided_limit(
            original, grown, limit_um=3.0, spacing=(1.0, 1.0, 1.0))
        assert fine.sum() > coarse.sum()

    def test_zero_means_no_limit(self):
        original = self._shell()
        grown = np.zeros_like(original)
        grown[0:12, 2:18, 2:18] = True
        out, hit, _mean, _mx = two_sided_limit(
            original, grown, limit_um=0.0, spacing=(1.0, 1.0, 1.0))
        assert hit is False
        assert np.array_equal(out, grown)

    def test_nothing_moved_measures_zero(self):
        original = self._shell()
        out, hit, mean, mx = two_sided_limit(
            original, original.copy(), limit_um=3.0)
        assert hit is False and mean == 0.0 and mx == 0.0
        assert np.array_equal(out, original)


# ===========================================================================
# TestMorphgacRefine
# ===========================================================================

class TestMorphgacRefine:
    """Tests for morphgac_refine()."""

    def test_refine_modifies_deformed_label(self):
        """Dented sphere + edge image → should modify voxels."""
        vol = _make_dented_sphere(label_id=3)
        # Build edge image from the FULL sphere (target shape).
        target = _make_sphere(label_id=3)
        edge_img = _make_edge_image(target, label_id=3)
        gimage = build_edge_stopping(edge_img)

        result, n_mod, msg = morphgac_refine(
            vol, 3, gimage, iterations=50, smoothing=1, balloon=0.3,
        )
        assert n_mod > 0, "Expected MorphGAC to modify deformed label"
        assert "refined" in msg.lower() or "changed" in msg.lower()

    def test_result_contains_only_label_id_and_zero(self):
        """Result should contain only label_id and 0."""
        vol = _make_sphere(label_id=5)
        gimage = build_edge_stopping(np.random.rand(*vol.shape))

        result, _, _ = morphgac_refine(vol, 5, gimage, iterations=20)
        unique = set(np.unique(result))
        assert unique <= {0, 5}, f"Unexpected labels in result: {unique}"

    def test_neighbor_constraint_prevents_invasion(self):
        """Adjacent label should not be overwritten."""
        vol = _make_sphere(label_id=3, center=(15, 15, 12), radius=7)
        # Place a neighbor label nearby.
        vol[10:20, 10:20, 20:28] = 99
        gimage = build_edge_stopping(np.random.rand(*vol.shape))

        result, _, _ = morphgac_refine(
            vol, 3, gimage, iterations=50, balloon=0.5, neighbor_constraint=True,
        )
        # Result should not contain label 99.
        assert 99 not in result, "Neighbor label was invaded"

    def test_distance_clamp_limits_expansion(self):
        """Expansion should be bounded by the clamp radius."""
        vol = _make_sphere(label_id=3, radius=5, center=(15, 15, 15))
        gimage = build_edge_stopping(np.zeros_like(vol, dtype=np.float64))

        result, _, _ = morphgac_refine(
            vol, 3, gimage, iterations=100, balloon=0.5, distance_clamp=3,
        )
        # Check that the result doesn't extend more than 3 voxels from original.
        from scipy.ndimage import distance_transform_edt

        orig_mask = vol == 3
        result_mask = result == 3
        # Any new voxel should be within 3 of the original boundary.
        new_voxels = result_mask & ~orig_mask
        if np.any(new_voxels):
            dist = distance_transform_edt(~orig_mask)
            max_dist = dist[new_voxels].max()
            assert max_dist <= 3.0 + 0.5, (
                f"Expansion exceeded clamp: max_dist={max_dist}"
            )

    def test_label_not_found_returns_zeros(self):
        """Absent label → empty result, 0 modified."""
        vol = np.zeros((10, 10, 10), dtype=np.int32)
        gimage = np.ones((10, 10, 10), dtype=np.float64)
        result, n_mod, msg = morphgac_refine(vol, 42, gimage)
        assert n_mod == 0
        assert np.all(result == 0)
        assert "not found" in msg

    def test_balloon_positive_expands(self):
        """Positive balloon toward a larger membrane edge → label should expand."""
        vol = _make_sphere(label_id=3, radius=5, center=(15, 15, 15))
        # Build edge image from a LARGER sphere (radius=9) so the contour
        # has edges to attract it outward.
        target = _make_sphere(label_id=3, radius=9, center=(15, 15, 15))
        edge_img = _make_edge_image(target, label_id=3)
        gimage = build_edge_stopping(edge_img)
        orig_count = np.count_nonzero(vol == 3)

        result, _, _ = morphgac_refine(
            vol, 3, gimage, iterations=50, balloon=0.5, distance_clamp=6,
        )
        new_count = np.count_nonzero(result == 3)
        assert new_count >= orig_count, (
            f"Positive balloon toward larger edge should expand: {new_count} < {orig_count}"
        )

    def test_balloon_negative_contracts(self):
        """Negative balloon should contract the label (or at least not expand)."""
        vol = _make_sphere(label_id=3, radius=8, center=(15, 15, 15))
        gimage = build_edge_stopping(np.zeros_like(vol, dtype=np.float64))
        orig_count = np.count_nonzero(vol == 3)

        result, _, _ = morphgac_refine(
            vol, 3, gimage, iterations=30, balloon=-0.8, distance_clamp=0,
        )
        new_count = np.count_nonzero(result == 3)
        assert new_count <= orig_count, (
            f"Negative balloon should contract: {new_count} > {orig_count}"
        )

    def test_no_crash_single_voxel(self):
        """Single-voxel label should not crash."""
        vol = np.zeros((10, 10, 10), dtype=np.int32)
        vol[5, 5, 5] = 1
        gimage = np.ones((10, 10, 10), dtype=np.float64)
        result, n_mod, _ = morphgac_refine(vol, 1, gimage, iterations=10)
        assert result.shape == vol.shape


# ===========================================================================
# TestMorphacweRefine
# ===========================================================================

class TestMorphacweRefine:
    """Tests for morphacwe_refine()."""

    def test_acwe_refines_deformed_label(self):
        """Dented sphere + intensity contrast → should modify voxels."""
        vol = _make_dented_sphere(label_id=3)
        # Build intensity image from full sphere (strong inside/outside contrast).
        target = _make_sphere(label_id=3)
        img = _make_intensity_image(target, label_id=3)

        result, n_mod, msg = morphacwe_refine(
            vol, 3, img, iterations=50, smoothing=1,
        )
        assert n_mod > 0, "Expected MorphACWE to modify deformed label"

    def test_acwe_result_contains_only_label_id_and_zero(self):
        """Result should contain only label_id and 0."""
        vol = _make_sphere(label_id=5)
        img = np.random.rand(*vol.shape)

        result, _, _ = morphacwe_refine(vol, 5, img, iterations=20)
        unique = set(np.unique(result))
        assert unique <= {0, 5}, f"Unexpected labels in result: {unique}"

    def test_acwe_neighbor_constraint(self):
        """Adjacent label should not be overwritten."""
        vol = _make_sphere(label_id=3, center=(15, 15, 12), radius=7)
        vol[10:20, 10:20, 20:28] = 99
        img = _make_intensity_image(vol, label_id=3)

        result, _, _ = morphacwe_refine(
            vol, 3, img, iterations=50, neighbor_constraint=True,
        )
        assert 99 not in result, "Neighbor label was invaded"

    def test_acwe_distance_clamp(self):
        """Expansion should be bounded by the clamp radius."""
        vol = _make_sphere(label_id=3, radius=5, center=(15, 15, 15))
        img = _make_intensity_image(vol, label_id=3)

        result, _, _ = morphacwe_refine(
            vol, 3, img, iterations=100, distance_clamp=3,
        )
        from scipy.ndimage import distance_transform_edt

        orig_mask = vol == 3
        result_mask = result == 3
        new_voxels = result_mask & ~orig_mask
        if np.any(new_voxels):
            dist = distance_transform_edt(~orig_mask)
            max_dist = dist[new_voxels].max()
            assert max_dist <= 3.0 + 0.5, (
                f"Expansion exceeded clamp: max_dist={max_dist}"
            )

    def test_acwe_label_not_found(self):
        """Absent label → empty result, 0 modified."""
        vol = np.zeros((10, 10, 10), dtype=np.int32)
        img = np.ones((10, 10, 10), dtype=np.float64)
        result, n_mod, msg = morphacwe_refine(vol, 42, img)
        assert n_mod == 0
        assert np.all(result == 0)
        assert "not found" in msg


# ===========================================================================
# TestMorphacweBand
# ===========================================================================

class TestMorphacweBand:
    """The additive ``band_mask`` keyword."""

    def test_no_band_is_skimage_unchanged(self):
        vol = _make_sphere(radius=7)
        img = _make_intensity_image(vol)
        a = morphacwe_refine(vol, 3, img, iterations=10)
        b = morphacwe_refine(vol, 3, img, iterations=10, band_mask=None)
        assert np.array_equal(a[0], b[0])

    def test_nothing_outside_the_band_changes(self):
        vol = _make_sphere(radius=7)
        img = np.full(vol.shape, 0.9)          # everything looks "inside"
        band = np.zeros(vol.shape, bool)
        band[:, :, 15:] = True
        res, _n, _m = morphacwe_refine(vol, 3, img, iterations=20,
                                       band_mask=band, distance_clamp=0)
        assert np.array_equal((res == 3)[~band], (vol == 3)[~band])


# ===========================================================================
# TestDilateErodeLabel
# ===========================================================================

class TestDilateErodeLabel:
    """Tests for dilate_erode_label()."""

    def test_dilate_expands_label(self):
        """Positive radius should expand the label."""
        vol = _make_sphere(label_id=3, radius=5, center=(15, 15, 15))
        orig_count = np.count_nonzero(vol == 3)

        result, n_mod, msg = dilate_erode_label(vol, 3, radius=2)
        new_count = np.count_nonzero(result == 3)
        assert new_count > orig_count, "Dilation should expand the label"
        assert n_mod > 0
        assert "Dilated" in msg

    def test_erode_shrinks_label(self):
        """Negative radius should shrink the label."""
        vol = _make_sphere(label_id=3, radius=8, center=(15, 15, 15))
        orig_count = np.count_nonzero(vol == 3)

        result, n_mod, msg = dilate_erode_label(vol, 3, radius=-2)
        new_count = np.count_nonzero(result == 3)
        assert new_count < orig_count, "Erosion should shrink the label"
        assert n_mod > 0
        assert "Eroded" in msg

    def test_radius_zero_no_change(self):
        """radius=0 should produce no change."""
        vol = _make_sphere(label_id=3, radius=5)
        result, n_mod, msg = dilate_erode_label(vol, 3, radius=0)
        assert n_mod == 0

    def test_result_contains_only_label_id_and_zero(self):
        """Result should contain only label_id and 0."""
        vol = _make_sphere(label_id=5)
        result, _, _ = dilate_erode_label(vol, 5, radius=3)
        unique = set(np.unique(result))
        assert unique <= {0, 5}, f"Unexpected labels: {unique}"

    def test_neighbor_constraint_prevents_invasion(self):
        """Dilation should not overwrite adjacent labels."""
        vol = _make_sphere(label_id=3, center=(15, 15, 12), radius=6)
        vol[10:20, 10:20, 20:28] = 99

        result, _, _ = dilate_erode_label(vol, 3, radius=5, neighbor_constraint=True)
        assert 99 not in result, "Neighbor label was invaded by dilation"

    def test_label_not_found(self):
        """Absent label → empty result, 0 modified."""
        vol = np.zeros((10, 10, 10), dtype=np.int32)
        result, n_mod, msg = dilate_erode_label(vol, 42, radius=3)
        assert n_mod == 0
        assert np.all(result == 0)
        assert "not found" in msg

    def test_erode_to_nothing(self):
        """Eroding a small label with large radius should not crash."""
        vol = np.zeros((20, 20, 20), dtype=np.int32)
        vol[9:12, 9:12, 9:12] = 1  # 3x3x3 cube
        result, n_mod, _ = dilate_erode_label(vol, 1, radius=-5)
        assert result.shape == vol.shape
        # Label may be completely eroded away — that's fine.


# ===========================================================================
# TestGuidedGrow
# ===========================================================================

class TestGuidedGrow:
    """Tests for guided_grow()."""

    def test_prob_guided_growth(self):
        """Growth into high-probability voxels should expand the label."""
        vol = _make_sphere(label_id=3, radius=5, center=(15, 15, 15))
        orig_count = np.count_nonzero(vol == 3)

        # Prob volume: high probability in a larger sphere.
        prob = np.zeros(vol.shape, dtype=np.float64)
        zz, yy, xx = np.ogrid[:30, :30, :30]
        dist = np.sqrt((zz - 15) ** 2 + (yy - 15) ** 2 + (xx - 15) ** 2)
        prob[dist <= 9] = 0.8  # high prob in larger region

        result, n_added, msg = guided_grow(
            vol, 3, prob_roi=prob, max_radius=5, prob_threshold=0.5,
        )
        new_count = np.count_nonzero(result == 3)
        assert new_count > orig_count, "Prob-guided grow should expand label"
        assert n_added > 0
        assert "prob" in msg

    def test_intensity_guided_growth(self):
        """Growth into matching-intensity voxels should expand the label."""
        vol = _make_sphere(label_id=3, radius=5, center=(15, 15, 15))
        orig_count = np.count_nonzero(vol == 3)

        # Image: uniform 0.8 inside a larger sphere, 0.2 outside.
        target = _make_sphere(label_id=3, radius=9, center=(15, 15, 15))
        img = _make_intensity_image(target, label_id=3, inside_val=0.8, outside_val=0.2)

        result, n_added, msg = guided_grow(
            vol, 3, image_roi=img, max_radius=5,
        )
        new_count = np.count_nonzero(result == 3)
        assert new_count > orig_count, "Intensity-guided grow should expand label"
        assert n_added > 0
        assert "intensity" in msg

    def test_combined_prob_and_intensity(self):
        """Using both prob and intensity should grow (and mention both in msg)."""
        vol = _make_sphere(label_id=3, radius=5, center=(15, 15, 15))

        # Prob: high in larger sphere.
        prob = np.zeros(vol.shape, dtype=np.float64)
        zz, yy, xx = np.ogrid[:30, :30, :30]
        dist = np.sqrt((zz - 15) ** 2 + (yy - 15) ** 2 + (xx - 15) ** 2)
        prob[dist <= 9] = 0.8

        # Image: matching intensity in same larger sphere.
        target = _make_sphere(label_id=3, radius=9, center=(15, 15, 15))
        img = _make_intensity_image(target, label_id=3)

        result, n_added, msg = guided_grow(
            vol, 3, image_roi=img, prob_roi=prob,
            max_radius=5, prob_threshold=0.5,
        )
        assert n_added > 0
        assert "prob" in msg and "intensity" in msg

    def test_neighbor_constraint(self):
        """Growth should not invade neighbor labels."""
        vol = _make_sphere(label_id=3, center=(15, 15, 12), radius=5)
        vol[10:20, 10:20, 20:28] = 99

        prob = np.ones(vol.shape, dtype=np.float64) * 0.9
        result, _, _ = guided_grow(
            vol, 3, prob_roi=prob, max_radius=5, neighbor_constraint=True,
        )
        assert 99 not in result, "Neighbor label was invaded"

    def test_max_radius_limits_growth(self):
        """Growth should not exceed max_radius from original boundary."""
        vol = _make_sphere(label_id=3, radius=5, center=(15, 15, 15))
        prob = np.ones(vol.shape, dtype=np.float64) * 0.9  # all eligible

        result, _, _ = guided_grow(vol, 3, prob_roi=prob, max_radius=3)

        from scipy.ndimage import distance_transform_edt
        orig_mask = vol == 3
        result_mask = result == 3
        new_voxels = result_mask & ~orig_mask
        if np.any(new_voxels):
            dist = distance_transform_edt(~orig_mask)
            max_dist = dist[new_voxels].max()
            assert max_dist <= 3.0 + 0.5, (
                f"Growth exceeded max_radius: {max_dist}"
            )

    def test_label_not_found(self):
        """Absent label → empty result, 0 added."""
        vol = np.zeros((10, 10, 10), dtype=np.int32)
        prob = np.ones((10, 10, 10), dtype=np.float64)
        result, n_added, msg = guided_grow(vol, 42, prob_roi=prob)
        assert n_added == 0
        assert "not found" in msg

    def test_no_source_returns_error(self):
        """No image or prob → should return 0 with error message."""
        vol = _make_sphere(label_id=3)
        result, n_added, msg = guided_grow(vol, 3)
        assert n_added == 0
        assert "requires" in msg.lower()

    def test_low_prob_threshold_grows_more(self):
        """Lower prob threshold should allow more growth."""
        vol = _make_sphere(label_id=3, radius=5, center=(15, 15, 15))

        # Prob: moderate values (0.4) in surrounding region.
        prob = np.full(vol.shape, 0.4, dtype=np.float64)
        prob[vol == 3] = 0.9

        _, n_high, _ = guided_grow(vol, 3, prob_roi=prob, prob_threshold=0.6)
        _, n_low, _ = guided_grow(vol, 3, prob_roi=prob, prob_threshold=0.3)
        assert n_low >= n_high, (
            f"Lower threshold should grow >= higher: {n_low} < {n_high}"
        )

    def test_grow_axis_xy_constrains_growth(self):
        """Growth with axis='XY' should not expand in the Z direction."""
        vol = _make_sphere(label_id=3, radius=5, center=(15, 15, 15))
        orig_z_extent = _axis_extent(vol == 3, axis=0)

        prob = np.ones(vol.shape, dtype=np.float64) * 0.9
        result, n_added, _ = guided_grow(
            vol, 3, prob_roi=prob, max_radius=3, axis="XY",
        )
        assert n_added > 0, "XY-constrained grow should still add voxels"
        new_z_extent = _axis_extent(result == 3, axis=0)
        assert new_z_extent == orig_z_extent, (
            f"Z-extent changed: {orig_z_extent} → {new_z_extent}"
        )

    def test_grow_axis_yz(self):
        """Growth with axis='YZ' should not expand in the X direction."""
        vol = _make_sphere(label_id=3, radius=5, center=(15, 15, 15))
        orig_x_extent = _axis_extent(vol == 3, axis=2)

        prob = np.ones(vol.shape, dtype=np.float64) * 0.9
        result, n_added, _ = guided_grow(
            vol, 3, prob_roi=prob, max_radius=3, axis="YZ",
        )
        assert n_added > 0, "YZ-constrained grow should still add voxels"
        new_x_extent = _axis_extent(result == 3, axis=2)
        assert new_x_extent == orig_x_extent, (
            f"X-extent changed: {orig_x_extent} → {new_x_extent}"
        )


# ===========================================================================
# TestMakeAxisStructure
# ===========================================================================

class TestMakeAxisStructure:
    """Tests for _make_axis_structure()."""

    def test_3d_returns_none(self):
        """axis='3D' should return None (scipy default)."""
        assert _make_axis_structure("3D") is None

    def test_xy_structure_shape_and_plane(self):
        """axis='XY' should produce a (3,3,3) array active only at z=1."""
        s = _make_axis_structure("XY")
        assert s is not None
        assert s.shape == (3, 3, 3)
        assert s.dtype == bool
        # Only z=1 plane should be active.
        assert s[0].sum() == 0
        assert s[2].sum() == 0
        assert s[1].sum() > 0


# ===========================================================================
# TestDilateErodeLabel — per-axis
# ===========================================================================

class TestDilateErodeLabelAxis:
    """Per-axis tests for dilate_erode_label()."""

    def test_dilate_axis_xy_only_expands_in_plane(self):
        """Dilation with axis='XY' should not change Z extent."""
        vol = _make_sphere(label_id=3, radius=5, center=(15, 15, 15))
        orig_z_extent = _axis_extent(vol == 3, axis=0)

        result, n_mod, _ = dilate_erode_label(vol, 3, radius=2, axis="XY")
        assert n_mod > 0, "XY dilation should modify voxels"
        new_z_extent = _axis_extent(result == 3, axis=0)
        assert new_z_extent == orig_z_extent, (
            f"Z-extent changed: {orig_z_extent} → {new_z_extent}"
        )

    def test_dilate_axis_xz_preserves_y(self):
        """Dilation with axis='XZ' should not change Y extent."""
        vol = _make_sphere(label_id=3, radius=5, center=(15, 15, 15))
        orig_y_extent = _axis_extent(vol == 3, axis=1)

        result, n_mod, _ = dilate_erode_label(vol, 3, radius=2, axis="XZ")
        assert n_mod > 0, "XZ dilation should modify voxels"
        new_y_extent = _axis_extent(result == 3, axis=1)
        assert new_y_extent == orig_y_extent, (
            f"Y-extent changed: {orig_y_extent} → {new_y_extent}"
        )


# ===========================================================================
# Helpers for fill tests
# ===========================================================================

def _make_hollow_sphere(
    shape=(30, 30, 30), label_id=3, center=(15, 15, 15),
    outer_radius=10, inner_radius=4,
):
    """Create a sphere with a hollow interior (internal void)."""
    vol = np.zeros(shape, dtype=np.int32)
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist = np.sqrt(
        (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2
    )
    vol[(dist <= outer_radius) & (dist > inner_radius)] = label_id
    return vol


# ===========================================================================
# TestFillHolesLabel
# ===========================================================================

class TestFillHolesLabel:
    """Tests for fill_holes_label()."""

    def test_fill_3d_restores_hollow_sphere(self):
        """3D fill on hollow sphere should fill the internal void."""
        vol = _make_hollow_sphere(label_id=3)
        result, n_mod, msg = fill_holes_label(vol, 3, axis="3D")
        assert n_mod > 0, "Internal void should be filled"
        # Result should be a solid sphere (no background inside).
        unique = set(np.unique(result))
        assert unique <= {0, 3}, f"Unexpected labels: {unique}"
        assert "filled" in msg.lower() or "fill" in msg.lower()

    def test_fill_per_slice_xy(self):
        """Per-slice XY: fills in-plane holes slice-by-slice."""
        vol = _make_hollow_sphere(label_id=3)
        result, n_mod, msg = fill_holes_label(vol, 3, axis="XY")
        assert n_mod > 0, "Per-slice XY should fill in-plane holes"
        unique = set(np.unique(result))
        assert unique <= {0, 3}

    def test_fill_per_slice_xz(self):
        """Per-slice XZ: fills in-plane holes for XZ orientation."""
        vol = _make_hollow_sphere(label_id=3)
        result, n_mod, msg = fill_holes_label(vol, 3, axis="XZ")
        assert n_mod > 0, "Per-slice XZ should fill in-plane holes"
        unique = set(np.unique(result))
        assert unique <= {0, 3}

    def test_fill_per_slice_yz(self):
        """Per-slice YZ: fills in-plane holes for YZ orientation."""
        vol = _make_hollow_sphere(label_id=3)
        result, n_mod, msg = fill_holes_label(vol, 3, axis="YZ")
        assert n_mod > 0, "Per-slice YZ should fill in-plane holes"
        unique = set(np.unique(result))
        assert unique <= {0, 3}

    def test_empty_label_no_change(self):
        """Label not present → n_modified=0."""
        vol = np.zeros((10, 10, 10), dtype=np.int32)
        result, n_mod, msg = fill_holes_label(vol, 42, axis="3D")
        assert n_mod == 0
        assert "not found" in msg.lower()

    def test_neighbor_constraint_preserves_neighbors(self):
        """Void between label and neighbor → fill doesn't overwrite neighbor."""
        vol = _make_hollow_sphere(
            label_id=3, center=(15, 15, 15), outer_radius=10, inner_radius=4,
        )
        # Place a neighbor label inside the void.
        vol[14:17, 14:17, 14:17] = 99
        result, n_mod, msg = fill_holes_label(vol, 3, axis="3D")
        # Neighbor label must NOT be overwritten.
        assert 99 not in np.unique(result), "Neighbor label was overwritten"
        # The fill should still add some voxels (the parts of the void not
        # occupied by 99).
        assert n_mod > 0

    def test_return_contract(self):
        """Return (result, n_modified, message) with result containing only {0, label_id}."""
        vol = _make_hollow_sphere(label_id=5)
        result, n_mod, msg = fill_holes_label(vol, 5)
        assert isinstance(result, np.ndarray)
        assert isinstance(n_mod, int)
        assert isinstance(msg, str)
        unique = set(np.unique(result))
        assert unique <= {0, 5}

    def test_solid_label_no_change(self):
        """Solid sphere (no holes) → n_modified=0."""
        vol = _make_sphere(label_id=3, radius=8)
        result, n_mod, msg = fill_holes_label(vol, 3, axis="3D")
        assert n_mod == 0


# ===========================================================================
# TestFloodFillLabel
# ===========================================================================

class TestFloodFillLabel:
    """Tests for flood_fill_label()."""

    def test_fill_adjacent_void(self):
        """Fills connected background region at seed point."""
        vol = _make_sphere(label_id=3, radius=5, center=(15, 15, 15))
        # Create a small void pocket next to the sphere.
        vol[15, 15, 22:27] = 0  # background corridor
        seed = (15, 15, 25)  # inside the corridor
        result, n_mod, msg = flood_fill_label(vol, 3, seed, axis="3D")
        assert n_mod > 0, "Should fill background at seed"
        assert result[seed] == 3

    def test_seed_on_label_voxel_noop(self):
        """Seed on a label voxel (not background) → no-op."""
        vol = _make_sphere(label_id=3, radius=5, center=(15, 15, 15))
        seed = (15, 15, 15)  # inside the sphere
        result, n_mod, msg = flood_fill_label(vol, 3, seed)
        assert n_mod == 0
        assert "not background" in msg.lower() or "not empty" in msg.lower()

    def test_seed_out_of_bounds_noop(self):
        """Seed outside ROI bounds → no-op with message."""
        vol = _make_sphere(label_id=3, radius=5, center=(15, 15, 15))
        seed = (50, 50, 50)  # out of bounds
        result, n_mod, msg = flood_fill_label(vol, 3, seed)
        assert n_mod == 0
        assert "out of bounds" in msg.lower() or "outside" in msg.lower()

    def test_max_fill_size_exceeded_noop(self):
        """Region larger than max_fill_size → no-op."""
        vol = np.zeros((30, 30, 30), dtype=np.int32)
        # Just a tiny label, huge background.
        vol[15, 15, 15] = 3
        seed = (0, 0, 0)  # background, connected to massive region
        result, n_mod, msg = flood_fill_label(
            vol, 3, seed, max_fill_size=100,
        )
        assert n_mod == 0
        assert "too large" in msg.lower() or "exceeds" in msg.lower()

    def test_per_axis_xy_fills_only_2d(self):
        """Per-axis XY: only fills 2D connected component on the seed's Z-slice."""
        vol = np.zeros((30, 30, 30), dtype=np.int32)
        # Create a ring label on one slice (z=15).
        zz, yy, xx = np.ogrid[:30, :30, :30]
        dist = np.sqrt((yy - 15) ** 2 + (xx - 15) ** 2)
        vol[15, (dist[0] <= 8) & (dist[0] > 4)] = 3  # ring on z=15
        seed = (15, 15, 15)  # inside the ring on z=15 (background)
        result, n_mod, msg = flood_fill_label(vol, 3, seed, axis="XY")
        assert n_mod > 0
        # Only z=15 should be modified.
        modified = (result != vol)
        modified_z_slices = np.any(modified, axis=(1, 2))
        assert np.sum(modified_z_slices) == 1, "Should only modify one Z-slice"

    def test_neighbor_constraint_no_overwrite(self):
        """Flood fill stays within background only — no label overwrite."""
        vol = np.zeros((20, 20, 20), dtype=np.int32)
        vol[5:15, 5:15, 5:8] = 3   # target label
        vol[5:15, 5:15, 12:15] = 99  # neighbor
        # Background gap between labels: z=5:15, y=5:15, x=8:12
        seed = (10, 10, 10)  # in the gap
        result, n_mod, msg = flood_fill_label(vol, 3, seed, axis="3D")
        assert n_mod > 0
        # Neighbor must not be overwritten.
        assert np.all(result[vol == 99] != 3), "Neighbor label overwritten"

    def test_return_contract(self):
        """Return (result, n_modified, message)."""
        vol = np.zeros((10, 10, 10), dtype=np.int32)
        vol[0:5, :, :] = 3
        seed = (7, 5, 5)
        result, n_mod, msg = flood_fill_label(vol, 3, seed, max_fill_size=100000)
        assert isinstance(result, np.ndarray)
        assert isinstance(n_mod, int)
        assert isinstance(msg, str)
        unique = set(np.unique(result))
        assert unique <= {0, 3}
