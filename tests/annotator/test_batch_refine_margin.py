"""R1b — the Batch Refine ROI crop must be wide enough for an SDF smooth.

Pure pytest — no Qt / napari.  Small synthetic int label volumes only.

The bug this file pins down
---------------------------
``_recipe_margin`` used to count **only positive dilate radii** (+2 slack),
so an SDF-only recipe ran inside a ``bbox + 2`` crop.  ``gaussian_filter``
reflects at the crop wall and ``smooth_label_sdf`` hard-clips its result to
the crop (``smooth.py`` :64, :97-99), so the SAME sigma produced a visibly
weaker / different shape than the interactive Refine tab, which crops with
``margin=15`` (``_snapshot_label_roi``).  A batch result could therefore look
like it had "reverted" when in fact the compute had under-delivered.

The contract asserted here: for a label that is far from every volume wall,
the batch executor and the interactive smooth path must produce the SAME
voxels at the same sigma.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from packages.epicycle_annotator.batch_refine import (
    RefineStep,
    _recipe_margin,
    apply_recipe_to_label,
    compute_label_roi,
    run_batch_recipe,
)
from packages.epicycle_annotator.smooth import apply_smooth_to_volume, smooth_label_sdf

# The interactive Refine tab's hard-coded ROI padding
# (``Curation3DAnnotator._snapshot_label_roi(..., margin=15)``).
INTERACTIVE_MARGIN = 15


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
def _interior_blob(shape=(60, 60, 60)):
    """A sharp-cornered label 1 sitting far (>= 24 voxels) from every wall.

    A cube with one corner notched out — sharp enough that an SDF smooth
    genuinely changes it, and asymmetric enough that a reflect-at-the-wall
    artefact would not cancel out.
    """
    vol = np.zeros(shape, dtype=np.int32)
    vol[24:36, 24:36, 24:36] = 1
    vol[24:28, 24:28, 24:28] = 0  # notch one corner
    return vol


def _smooth_via_roi(vol, label_id, roi, *, sigma_z, sigma_xy):
    """Run the interactive smooth path over an explicit ROI → new volume.

    Mirrors ``curation_3d_widget._preview_smooth``'s SDF branch verbatim:
    ``smooth_label_sdf(shared[roi], ...)`` then
    ``apply_smooth_to_volume(shared, roi, result, label_id)``.
    """
    out = vol.copy()
    result_roi, _n, _msg = smooth_label_sdf(
        out[roi], label_id, sigma_z=sigma_z, sigma_xy=sigma_xy,
        preserve_topology=True,
    )
    apply_smooth_to_volume(out, roi, result_roi, label_id)
    return out


# --------------------------------------------------------------------------- #
# (a) _recipe_margin accounts for every SDF step's sigma
# --------------------------------------------------------------------------- #
def _as_axes(margin, ndim=3):
    """Normalize whatever ``_recipe_margin`` returns into a per-axis list."""
    try:
        axes = [int(m) for m in margin]
    except TypeError:
        axes = [int(margin)] * ndim
    assert len(axes) == ndim
    return axes


def test_recipe_margin_sdf_only_covers_three_sigma():
    for sigma in (0.5, 1.5, 3.0, 7.0):
        recipe = [RefineStep("sdf", {"sigma_z": sigma, "sigma_xy": sigma})]
        axes = _as_axes(_recipe_margin(recipe))
        floor = math.ceil(3.0 * sigma)
        assert all(m >= floor for m in axes), (sigma, axes, floor)


def test_recipe_margin_sdf_defaults_are_anisotropic():
    # smooth_label_sdf defaults: sigma_z=3.0, sigma_xy=1.5.
    axes = _as_axes(_recipe_margin([RefineStep("sdf", {})]))
    assert axes[0] >= math.ceil(3.0 * 3.0)
    assert axes[1] >= math.ceil(3.0 * 1.5)
    assert axes[2] >= math.ceil(3.0 * 1.5)
    assert axes[0] > axes[1], "sigma_z > sigma_xy must widen Z more than XY"


def test_recipe_margin_adds_dilate_on_top_of_sigma():
    sigma = 2.0
    sdf_only = _as_axes(_recipe_margin([RefineStep("sdf", {"sigma_z": sigma, "sigma_xy": sigma})]))
    with_dilate = _as_axes(_recipe_margin([
        RefineStep("sdf", {"sigma_z": sigma, "sigma_xy": sigma}),
        RefineStep("dilate_erode", {"radius": 4}),
    ]))
    assert all(b == a + 4 for a, b in zip(sdf_only, with_dilate)), (sdf_only, with_dilate)
    assert all(m >= math.ceil(3.0 * sigma) + 4 for m in with_dilate)


def test_recipe_margin_chained_sdf_steps_accumulate():
    step = RefineStep("sdf", {"sigma_z": 2.0, "sigma_xy": 2.0})
    one = _as_axes(_recipe_margin([step]))
    two = _as_axes(_recipe_margin([step, step]))
    assert all(b > a for a, b in zip(one, two)), (one, two)


def test_recipe_margin_negative_dilate_does_not_shrink():
    base = _as_axes(_recipe_margin([RefineStep("sdf", {"sigma_z": 2.0, "sigma_xy": 2.0})]))
    eroded = _as_axes(_recipe_margin([
        RefineStep("sdf", {"sigma_z": 2.0, "sigma_xy": 2.0}),
        RefineStep("dilate_erode", {"radius": -5}),
    ]))
    assert eroded == base


def test_recipe_margin_non_sdf_recipe_matches_legacy_two_plus_radii():
    # Unchanged behaviour for the geometry-only recipes: dilate radii + 2.
    assert _as_axes(_recipe_margin([RefineStep("fill_holes", {})])) == [2, 2, 2]
    assert _as_axes(_recipe_margin([RefineStep("dilate_erode", {"radius": 3})])) == [5, 5, 5]


def test_recipe_margin_capped_by_volume_shape():
    recipe = [RefineStep("sdf", {"sigma_z": 9.0, "sigma_xy": 9.0})]
    axes = _as_axes(_recipe_margin(recipe, (8, 10, 12)))
    assert axes[0] <= 8 and axes[1] <= 10 and axes[2] <= 12


# --------------------------------------------------------------------------- #
# (b) equivalence: batch executor == interactive smooth, same sigma
# --------------------------------------------------------------------------- #
def test_batch_sdf_matches_interactive_smooth_for_interior_label():
    vol = _interior_blob()
    sigma_z, sigma_xy = 3.0, 1.5

    roi = compute_label_roi(vol, 1, INTERACTIVE_MARGIN)
    expected = _smooth_via_roi(vol, 1, roi, sigma_z=sigma_z, sigma_xy=sigma_xy)

    recipe = [RefineStep("sdf", {"sigma_z": sigma_z, "sigma_xy": sigma_xy})]
    got = run_batch_recipe(vol, [1], recipe, n_workers=1).working_volume

    np.testing.assert_array_equal(got, expected)


def test_batch_sdf_matches_interactive_smooth_isotropic():
    """The de-novo Add auto-smooth shape: isotropic sigma through the same path."""
    vol = _interior_blob()
    sigma = 1.5

    roi = compute_label_roi(vol, 1, INTERACTIVE_MARGIN)
    expected = _smooth_via_roi(vol, 1, roi, sigma_z=sigma, sigma_xy=sigma)

    recipe = [RefineStep("sdf", {"sigma_z": sigma, "sigma_xy": sigma})]
    got = run_batch_recipe(vol, [1], recipe, n_workers=1).working_volume

    np.testing.assert_array_equal(got, expected)


# --------------------------------------------------------------------------- #
# (c) the old margin (2) provably differs — this documents the bug
# --------------------------------------------------------------------------- #
def test_legacy_margin_two_differs_from_interactive_smooth():
    vol = _interior_blob()
    sigma_z, sigma_xy = 3.0, 1.5
    recipe = [RefineStep("sdf", {"sigma_z": sigma_z, "sigma_xy": sigma_xy})]

    expected = _smooth_via_roi(
        vol, 1, compute_label_roi(vol, 1, INTERACTIVE_MARGIN),
        sigma_z=sigma_z, sigma_xy=sigma_xy,
    )

    # The pre-fix crop: bbox + 2 (positive dilate radii only, and there are none).
    legacy_roi = compute_label_roi(vol, 1, 2)
    legacy_result, _n = apply_recipe_to_label(vol, 1, legacy_roi, recipe)
    legacy_out = vol.copy()
    apply_smooth_to_volume(legacy_out, legacy_roi, legacy_result, 1)

    assert not np.array_equal(legacy_out, expected), (
        "bbox+2 must NOT reproduce the interactive smooth — if it does, this "
        "fixture no longer exercises the reflect/clip artefact."
    )
    # And the artefact is a real difference in the label's footprint, not noise.
    assert int(np.count_nonzero((legacy_out == 1) != (expected == 1))) > 0


# --------------------------------------------------------------------------- #
# (d) a label hard against the volume wall still works
# --------------------------------------------------------------------------- #
def test_label_at_volume_edge_is_clipped_only_by_the_volume():
    vol = np.zeros((30, 30, 30), dtype=np.int32)
    vol[0:6, 0:6, 0:6] = 1  # corner-anchored, touches three walls
    vol[24:30, 24:30, 24:30] = 2  # opposite corner

    recipe = [RefineStep("sdf", {"sigma_z": 3.0, "sigma_xy": 1.5})]
    res = run_batch_recipe(vol, [1, 2], recipe, n_workers=1)
    out = res.working_volume

    assert out.shape == vol.shape
    assert set(np.unique(out)).issubset({0, 1, 2})
    assert np.any(out == 1) and np.any(out == 2)

    # The crop is clamped to the volume, so it may not extend past the walls.
    roi = compute_label_roi(vol, 1, _recipe_margin(recipe, vol.shape))
    for d, s in enumerate(roi):
        assert s.start >= 0 and s.stop <= vol.shape[d]


# --------------------------------------------------------------------------- #
# R1c — fixed in the batch path:
# an object whose result is empty is left as it was and listed in
# ``BatchResult.emptied_ids``.  Was a strict xfail until the fix landed.
# --------------------------------------------------------------------------- #
def test_r1c_empty_step_result_must_not_erase_the_label():
    vol = np.zeros((20, 20, 20), dtype=np.int32)
    vol[10, 8:12, 8:12] = 1  # a one-voxel-thick plate
    before = int(np.count_nonzero(vol == 1))

    recipe = [RefineStep("dilate_erode", {"radius": -3, "axis": "3D"})]
    res = run_batch_recipe(vol, [1], recipe, n_workers=1)

    after = int(np.count_nonzero(res.working_volume == 1))
    assert after > 0, (
        f"label 1 erased by an empty step result ({before} -> {after} voxels)"
    )
    assert after == before
    assert res.emptied_ids == [1]
    assert res.touched_ids == []
    assert res.n_changed_total == 0
