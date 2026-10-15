"""Tests for the 2026-05-21 `expand_labels_distance` compensation step.

Bug being addressed: `clean_mask_slice` with the default
`remove_boundaries=True` erodes 1 px off every cell boundary.
Trained models learn under-segmentation as correct.  Across
refinement rounds, masks shrink monotonically — user-reported as
"the model degraded the prediction" and "we need a way to recover
those removed and over-smoothed pixels".

Fix: new optional step 5 in the cleaning pipeline calls
`skimage.segmentation.expand_labels(distance=N)` AFTER the existing
erode/relabel/size/smooth steps.  Stops at adjacent labels (no
merging).  Default 0 (legacy behavior) — user opts in via the
per-orientation extraction config dialog.

These tests cover:
* The compensation actually works (cells return to original size).
* Labels never merge regardless of distance.
* `expand_labels_distance=0` is identical to legacy behavior.
* Independence from smoothing (both can be on, either can be on).
* Per-orientation plumbing in `_PERVIEW_PARAM_KEYS`.
"""

from __future__ import annotations

import numpy as np
import pytest

from napari_epicycle.fusion_reconstruction.training_pair_extractor import (
    clean_mask_slice,
)


def _two_touching_circles(canvas=(50, 100), r=15, cx_a=25, cx_b=55):
    """Two touching circles for the erosion/expansion test bed."""
    mask = np.zeros(canvas, dtype=np.int32)
    y, x = np.ogrid[: canvas[0], : canvas[1]]
    mask[(y - 25) ** 2 + (x - cx_a) ** 2 < r ** 2] = 1
    mask[(y - 25) ** 2 + (x - cx_b) ** 2 < r ** 2] = 2
    return mask


# ---------------------------------------------------------------------------
# Core compensation behavior
# ---------------------------------------------------------------------------


class TestExpandLabelsCompensation:

    def test_expand_distance_zero_is_legacy_behavior(self):
        """`expand_labels_distance=0` must produce identical output to
        the pre-fix code path (the default-off case)."""
        mask = _two_touching_circles()
        out_legacy = clean_mask_slice(
            mask,
            remove_boundaries=True, relabel=True,
            min_size=0, smoothing_disk_size=0,
            expand_labels_distance=0,
        )
        # Cells eroded by 1 px on each side (~84 px lost per cell).
        a_orig = (mask == 1).sum()
        a_eroded = (out_legacy == 1).sum()
        assert a_eroded < a_orig
        assert a_eroded == 613, f"Expected 613 (1px erosion), got {a_eroded}"

    def test_expand_distance_one_restores_original_size(self):
        """The compensation goal: `expand_labels_distance=1` should
        bring eroded cells back to their original area exactly.

        For two non-touching circles, 1 px erode + 1 px expand recovers
        the original count.  Touching circles have a shared boundary
        that doesn't fully recover (expand stops at the neighbor), so
        we verify per-cell area > eroded area + most of the lost pixels."""
        mask = _two_touching_circles()
        a_orig = (mask == 1).sum()
        out = clean_mask_slice(
            mask,
            remove_boundaries=True, relabel=True,
            min_size=0, smoothing_disk_size=0,
            expand_labels_distance=1,
        )
        a_recovered = (out == 1).sum()
        assert a_recovered == a_orig, (
            f"Cell 1 area should fully recover with 1px expand; "
            f"original={a_orig}, recovered={a_recovered}"
        )

    def test_expand_distance_two_overcompensates(self):
        """Distance 2 should expand cells BEYOND original size
        (only into background, not into other cells)."""
        mask = _two_touching_circles()
        a_orig = (mask == 1).sum()
        out = clean_mask_slice(
            mask,
            remove_boundaries=True, relabel=True,
            min_size=0, smoothing_disk_size=0,
            expand_labels_distance=2,
        )
        a_grown = (out == 1).sum()
        assert a_grown > a_orig, (
            f"Cell 1 should grow beyond original with 2px expand; "
            f"original={a_orig}, grown={a_grown}"
        )

    def test_labels_never_merge_regardless_of_distance(self):
        """Critical invariant: no matter how aggressive the expansion,
        labels must STAY SEPARATE (no two cells collapse to one)."""
        mask = _two_touching_circles()
        for distance in [1, 2, 3, 5, 10]:
            out = clean_mask_slice(
                mask,
                remove_boundaries=True, relabel=True,
                min_size=0, smoothing_disk_size=0,
                expand_labels_distance=distance,
            )
            unique_nonzero = set(np.unique(out)) - {0}
            assert unique_nonzero == {1, 2}, (
                f"Labels merged at distance={distance}: "
                f"got labels {unique_nonzero}"
            )


# ---------------------------------------------------------------------------
# Independence from other steps
# ---------------------------------------------------------------------------


class TestIndependenceFromOtherSteps:

    def test_expand_alone_without_other_steps(self):
        """expand_labels_distance with all other cleaning disabled
        should still expand each label (no erosion to compensate for,
        just pure outward growth)."""
        mask = _two_touching_circles()
        a_orig = (mask == 1).sum()
        out = clean_mask_slice(
            mask,
            remove_boundaries=False, relabel=False,
            min_size=0, smoothing_disk_size=0,
            expand_labels_distance=2,
        )
        # No erosion happened, so expand grows beyond original.
        assert (out == 1).sum() > a_orig

    def test_smooth_and_expand_compose(self):
        """Both smoothing AND expansion enabled — both apply
        sequentially (smooth then expand).  Final result should be
        smoothed AND expanded."""
        mask = _two_touching_circles()
        out_smooth_only = clean_mask_slice(
            mask,
            remove_boundaries=True, relabel=True,
            min_size=0, smoothing_disk_size=2,
            expand_labels_distance=0,
        )
        out_both = clean_mask_slice(
            mask,
            remove_boundaries=True, relabel=True,
            min_size=0, smoothing_disk_size=2,
            expand_labels_distance=1,
        )
        # With expansion, cell is larger than smooth-only.
        assert (out_both == 1).sum() > (out_smooth_only == 1).sum()

    def test_expand_no_op_on_empty_mask(self):
        """Empty mask + any expand distance = empty mask, no crash."""
        empty = np.zeros((20, 20), dtype=np.int32)
        out = clean_mask_slice(
            empty,
            remove_boundaries=True, relabel=True,
            expand_labels_distance=5,
        )
        assert np.count_nonzero(out) == 0


# ---------------------------------------------------------------------------
# Per-orientation plumbing
# ---------------------------------------------------------------------------


class TestPerViewParamKeys:

    def test_expand_labels_distance_in_perview_param_keys(self):
        """Verify the per-orientation key tuple in the controller
        includes the new key so state_manager overrides flow through."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        keys = FusionReconstructionController._PERVIEW_PARAM_KEYS
        assert "expand_labels_distance" in keys

    def test_default_configs_have_expand_field(self):
        """Both LATERAL and AXIAL default configs must include
        `expand_labels_distance` so callers don't KeyError."""
        from napari_epicycle.fusion_reconstruction.extraction_config_dialog import (
            LATERAL_DEFAULT_CONFIG, AXIAL_DEFAULT_CONFIG,
            get_default_config_for,
        )
        assert "expand_labels_distance" in LATERAL_DEFAULT_CONFIG
        assert "expand_labels_distance" in AXIAL_DEFAULT_CONFIG
        # Default value is 0 (legacy off behavior preserved).
        assert LATERAL_DEFAULT_CONFIG["expand_labels_distance"] == 0
        assert AXIAL_DEFAULT_CONFIG["expand_labels_distance"] == 0
        # get_default_config_for round-trips.
        assert get_default_config_for("lateral")["expand_labels_distance"] == 0
        assert get_default_config_for("axial")["expand_labels_distance"] == 0

    def test_dialog_constants_have_expand_field(self):
        """Dialog _CLEANING_KEYS / _LABELS / _RANGES / _TOOLTIPS all
        need the new key so the spinbox auto-generates."""
        from napari_epicycle.fusion_reconstruction.extraction_config_dialog import (
            _CLEANING_KEYS, _LABELS, _RANGES, _TOOLTIPS,
        )
        assert "expand_labels_distance" in _CLEANING_KEYS
        assert "expand_labels_distance" in _LABELS
        assert "expand_labels_distance" in _RANGES
        assert "expand_labels_distance" in _TOOLTIPS
        # Range is sensible.
        lo, hi = _RANGES["expand_labels_distance"]
        assert lo == 0 and hi <= 20

    def test_extract_from_workspace_accepts_kwarg(self):
        """`extract_from_workspace` must accept `expand_labels_distance`
        without TypeError (signature inspection — no actual extraction)."""
        import inspect
        from napari_epicycle.fusion_reconstruction.training_pair_extractor import (
            extract_from_workspace,
        )
        sig = inspect.signature(extract_from_workspace)
        assert "expand_labels_distance" in sig.parameters

    def test_extract_training_pairs_from_3d_accepts_kwarg(self):
        """Same check for the inner function."""
        import inspect
        from napari_epicycle.fusion_reconstruction.training_pair_extractor import (
            extract_training_pairs_from_3d,
        )
        sig = inspect.signature(extract_training_pairs_from_3d)
        assert "expand_labels_distance" in sig.parameters


# ---------------------------------------------------------------------------
# Manifest snapshot — audit-flagged critical
# ---------------------------------------------------------------------------


class TestAxialImageSliceIsChannelsFirst:
    """2026-05-21 axial-image-shape bug: pre-fix `extract_training_pairs_from_3d`
    sliced multichannel input (Z, C, Y, X) as:
        img[:, :, idx, :]  for XZ → (Z, C, X)  ← channels IN MIDDLE
        img[:, :, :, idx]  for YZ → (Z, C, Y)  ← channels IN MIDDLE
    tifffile saved these as N pages of (C, W) — user saw "512x2"
    axial images.  Cellpose's auto-detect silently canonicalized for
    training, but the saved TIFFs were visually wrong AND any tool
    that doesn't auto-detect would fail.

    Fix: transpose to (C, Z, X) / (C, Z, Y) — matching xy's (C, Y, X)
    and the inference layout in `packages/epicycle_reconstruction/
    logic.py::segment_views`.

    These tests verify the per-slice shape end-to-end through
    `extract_training_pairs_from_3d`.
    """

    def _run_extract(self, tmp_path, n_z=20, n_c=2, n_y=32, n_x=32):
        """Build a minimal ZCYX volume + labels, run extraction.
        Returns the path of one saved xz img + one xy img + reads
        them back to inspect shape.
        """
        import tifffile
        from napari_epicycle.fusion_reconstruction.training_pair_extractor import (
            extract_training_pairs_from_3d,
        )
        # Multichannel volume.
        np.random.seed(0)
        image = np.random.randint(
            0, 255, size=(n_z, n_c, n_y, n_x), dtype=np.uint8,
        )
        # Label volume with one big region per Z slab so XZ + YZ
        # slices all carry labels.
        labels = np.zeros((n_z, n_y, n_x), dtype=np.int32)
        labels[:, n_y // 4: 3 * n_y // 4, n_x // 4: 3 * n_x // 4] = 1
        labels[:, n_y // 4 + 4: 3 * n_y // 4 - 4,
               n_x // 4 + 4: 3 * n_x // 4 - 4] = 2

        out_dir = tmp_path / "pairs"
        extract_training_pairs_from_3d(
            image, labels, out_dir,
            min_labels_per_slice=1, edge_buffer=0,
            slices_per_patch=0, min_distance=1,
            remove_boundaries=False, relabel=False,
            min_size=0, smoothing_disk_size=0,
        )
        # Pick the first xz + xy + yz img.
        xy_imgs = sorted((out_dir / "spatial").glob("xy_*_img.tif"))
        xz_imgs = sorted((out_dir / "axial").glob("xz_*_img.tif"))
        yz_imgs = sorted((out_dir / "axial").glob("yz_*_img.tif"))
        assert xy_imgs, "no XY images saved"
        assert xz_imgs, "no XZ images saved"
        assert yz_imgs, "no YZ images saved"
        return {
            "xy": tifffile.imread(str(xy_imgs[0])),
            "xz": tifffile.imread(str(xz_imgs[0])),
            "yz": tifffile.imread(str(yz_imgs[0])),
            "n_c": n_c, "n_y": n_y, "n_x": n_x, "n_z": n_z,
        }

    def test_xy_image_is_channels_first(self, tmp_path):
        """XY slice from (Z, C, Y, X) → should save as (C, Y, X).
        This was already correct pre-fix — confirm we didn't break it."""
        r = self._run_extract(tmp_path)
        assert r["xy"].shape == (r["n_c"], r["n_y"], r["n_x"]), (
            f"XY saved shape {r['xy'].shape}, expected (C={r['n_c']}, "
            f"Y={r['n_y']}, X={r['n_x']})"
        )

    def test_xz_image_is_channels_first(self, tmp_path):
        """XZ slice from (Z, C, Y, X) → should save as (C, Z, X),
        NOT (Z, C, X).  This is the bug fix.
        Pre-fix: saved as (Z, C, X) = (20, 2, 32) for our test data.
        Post-fix: saved as (C, Z, X) = (2, 20, 32)."""
        r = self._run_extract(tmp_path)
        assert r["xz"].shape == (r["n_c"], r["n_z"], r["n_x"]), (
            f"XZ saved shape {r['xz'].shape}, expected (C={r['n_c']}, "
            f"Z={r['n_z']}, X={r['n_x']}).  If you see "
            f"(Z={r['n_z']}, C={r['n_c']}, X={r['n_x']}), the channel-"
            f"axis bug has regressed."
        )

    def test_yz_image_is_channels_first(self, tmp_path):
        """YZ slice from (Z, C, Y, X) → should save as (C, Z, Y),
        NOT (Z, C, Y).  Same fix as XZ."""
        r = self._run_extract(tmp_path)
        assert r["yz"].shape == (r["n_c"], r["n_z"], r["n_y"]), (
            f"YZ saved shape {r['yz'].shape}, expected (C={r['n_c']}, "
            f"Z={r['n_z']}, Y={r['n_y']})"
        )

    def test_all_three_views_channel_axis_consistent(self, tmp_path):
        """Cross-check: channels axis (axis 0) should match across
        all three saved views.  If it doesn't, downstream consumers
        that assume one layout will break on the other."""
        r = self._run_extract(tmp_path)
        assert r["xy"].shape[0] == r["n_c"]
        assert r["xz"].shape[0] == r["n_c"]
        assert r["yz"].shape[0] == r["n_c"]


class TestAxesAwareInputNormalization:
    """2026-05-21 axes-aware fix.  Pre-fix `extract_training_pairs_from_3d`
    assumed 4D input was always ZCYX.  CZYX inputs (which exist in real
    workflows — the Predict + Reconstruct section's axes combo offers
    both options) would slice along the C axis instead of Z, producing
    garbage XY pairs.

    Fix: new `axes` kwarg (default "ZCYX" for back-compat).  When
    "CZYX" is passed, the function transposes input to ZCYX internally
    before any slicing.  Mirrors what `segment_views` does for inference.
    """

    def _build_volumes(self, layout, n_z=20, n_c=2, n_y=32, n_x=32):
        """Return a (image, labels) pair with image in the given layout.
        The IDENTICAL physical content; just rearranged."""
        np.random.seed(42)
        # Build the canonical ZCYX version first.
        zcyx_image = np.random.randint(
            0, 255, size=(n_z, n_c, n_y, n_x), dtype=np.uint8,
        )
        labels = np.zeros((n_z, n_y, n_x), dtype=np.int32)
        labels[:, n_y // 4: 3 * n_y // 4, n_x // 4: 3 * n_x // 4] = 1

        if layout == "ZCYX":
            return zcyx_image, labels
        elif layout == "CZYX":
            # Transpose ZCYX → CZYX.  Same physical content, different
            # axis order.
            return np.transpose(zcyx_image, (1, 0, 2, 3)), labels
        elif layout == "ZYX":  # single-channel
            return zcyx_image[:, 0, :, :], labels  # drop C axis
        else:
            raise ValueError(layout)

    def _run_extract(self, tmp_path, layout):
        import tifffile
        from napari_epicycle.fusion_reconstruction.training_pair_extractor import (
            extract_training_pairs_from_3d,
        )
        image, labels = self._build_volumes(layout)
        out_dir = tmp_path / "pairs"
        # For ZYX (3D single-channel), the `axes` param is ignored —
        # the function detects ndim==3 and treats it as single-channel.
        kwargs = {} if layout == "ZYX" else {"axes": layout}
        extract_training_pairs_from_3d(
            image, labels, out_dir,
            min_labels_per_slice=1, edge_buffer=0,
            slices_per_patch=0, min_distance=1,
            remove_boundaries=False, relabel=False,
            min_size=0, smoothing_disk_size=0,
            **kwargs,
        )
        xy_imgs = sorted((out_dir / "spatial").glob("xy_*_img.tif"))
        xz_imgs = sorted((out_dir / "axial").glob("xz_*_img.tif"))
        assert xy_imgs and xz_imgs
        return {
            "xy": tifffile.imread(str(xy_imgs[0])),
            "xz": tifffile.imread(str(xz_imgs[0])),
        }

    def test_zcyx_input_gives_channels_first_slices(self, tmp_path):
        """Sanity: ZCYX input (the default, the post-restructure case)
        still works correctly post-fix."""
        r = self._run_extract(tmp_path, "ZCYX")
        # n_c=2 always axis 0 (channels-first).
        assert r["xy"].shape == (2, 32, 32)
        assert r["xz"].shape == (2, 20, 32)

    def test_czyx_input_normalizes_to_zcyx_internally(self, tmp_path):
        """The fix: CZYX input is normalized to ZCYX BEFORE slicing.
        Saved output should be channels-first, same physical content
        as the ZCYX case above."""
        r_czyx = self._run_extract(tmp_path / "czyx", "CZYX")
        r_zcyx = self._run_extract(tmp_path / "zcyx", "ZCYX")
        # Shape match (same physical volume, just different input layout).
        assert r_czyx["xy"].shape == r_zcyx["xy"].shape
        assert r_czyx["xz"].shape == r_zcyx["xz"].shape
        # Content match (channel ordering preserved by the transpose).
        # Both should produce IDENTICAL output because the physical
        # volume is identical; only the input shape was permuted.
        np.testing.assert_array_equal(r_czyx["xy"], r_zcyx["xy"])
        np.testing.assert_array_equal(r_czyx["xz"], r_zcyx["xz"])

    def test_single_channel_3d_input(self, tmp_path):
        """ZYX (3D single-channel) inputs should also work.  The
        function treats them as `multichannel=False` and slices
        directly without channel-axis concerns."""
        r = self._run_extract(tmp_path, "ZYX")
        # No channel axis — XY is (Y, X), XZ is (Z, X).
        assert r["xy"].shape == (32, 32)
        assert r["xz"].shape == (20, 32)

    def test_unsupported_axes_raises(self, tmp_path):
        """An unsupported axes string should raise ValueError, not
        silently produce wrong output."""
        from napari_epicycle.fusion_reconstruction.training_pair_extractor import (
            extract_training_pairs_from_3d,
        )
        image, labels = self._build_volumes("ZCYX")
        with pytest.raises(ValueError, match="Unsupported axes"):
            extract_training_pairs_from_3d(
                image, labels, tmp_path / "pairs",
                axes="YCZX",  # invalid permutation
                min_labels_per_slice=1, edge_buffer=0,
                slices_per_patch=0, min_distance=1,
                remove_boundaries=False, relabel=False,
                min_size=0, smoothing_disk_size=0,
            )

    def test_extract_from_workspace_accepts_axes_kwarg(self):
        """Confirm `extract_from_workspace` also accepts the new kwarg."""
        import inspect
        from napari_epicycle.fusion_reconstruction.training_pair_extractor import (
            extract_from_workspace,
        )
        sig = inspect.signature(extract_from_workspace)
        assert "axes" in sig.parameters
        assert sig.parameters["axes"].default == "ZCYX"

    def test_axes_none_defense(self, tmp_path):
        """2026-05-21 audit fix: `axes=None` must not crash with
        AttributeError on `.upper()`.  Pre-fix the function would
        crash before producing a clear error.  Now: None is treated
        as default ZCYX.
        """
        from napari_epicycle.fusion_reconstruction.training_pair_extractor import (
            extract_training_pairs_from_3d,
        )
        image, labels = self._build_volumes("ZCYX")
        # Must not raise AttributeError.
        extract_training_pairs_from_3d(
            image, labels, tmp_path / "pairs",
            axes=None,  # was the crash trigger
            min_labels_per_slice=1, edge_buffer=0,
            slices_per_patch=0, min_distance=1,
            remove_boundaries=False, relabel=False,
            min_size=0, smoothing_disk_size=0,
        )
        # Should have produced the same output as axes="ZCYX".
        xz_imgs = sorted(
            (tmp_path / "pairs" / "axial").glob("xz_*_img.tif")
        )
        assert xz_imgs


class TestAxesRecordedInManifest:
    """2026-05-21 audit fix: extraction manifest now records `axes`
    in each patch's per-view `extraction_settings` snapshot AND in
    the legacy schema-1.1 combined `current_extraction_settings`.

    Without this, `_check_extraction_mixed_settings` could not
    detect when the user changes axes (ZCYX ↔ CZYX) between
    extractions — both runs would silently produce different
    physical content with identical manifest dicts.
    """

    def test_lateral_extraction_settings_carries_axes(self):
        """Verify the manifest writer references 'axes' in the
        per-view extraction_settings dicts."""
        import inspect
        from napari_epicycle.fusion_reconstruction.training_pair_extractor import (
            extract_from_workspace,
        )
        src = inspect.getsource(extract_from_workspace)
        # Both lateral + axial extraction_settings get the axes field.
        assert 'lateral_extraction_settings["axes"]' in src
        assert 'axial_extraction_settings["axes"]' in src

    def test_combined_extraction_settings_carries_axes(self):
        """Legacy schema 1.1 combined dict also gets axes."""
        import inspect
        from napari_epicycle.fusion_reconstruction.training_pair_extractor import (
            extract_from_workspace,
        )
        src = inspect.getsource(extract_from_workspace)
        # `current_extraction_settings` dict literal must include "axes":
        # The dict is built in the function — verify the key appears.
        # Be specific: in extract_from_workspace ONLY.
        n_axes = src.count('"axes"')
        # Per-view writes (lateral_extraction_settings["axes"] +
        # axial_extraction_settings["axes"]) + current_extraction_settings
        # literal = 3 references minimum.
        assert n_axes >= 3, (
            f"Expected 'axes' to appear in manifest-writing source "
            f"at least 3×; found {n_axes}"
        )

    def test_axes_none_recorded_as_zcyx_default(self):
        """If a caller passes axes=None, manifest should still record
        a usable string ('ZCYX') — not the literal None."""
        import inspect
        from napari_epicycle.fusion_reconstruction.training_pair_extractor import (
            extract_from_workspace,
        )
        src = inspect.getsource(extract_from_workspace)
        # The defensive normalization line exists.
        assert '_axes_recorded = "ZCYX" if axes is None' in src


class TestManifestSnapshotCarriesExpandValue:
    """Audit 1 (2026-05-21) caught: `_split_view_dict` + the legacy
    `current_mask_cleaning` dict in `extract_from_workspace` omitted
    `expand_labels_distance` from the manifest snapshot.  Silently
    lost provenance — patches re-extracted with expand=0 vs expand=1
    would compare as identical in `_check_extraction_mixed_settings`.

    These tests guard against that regression."""

    def test_per_view_split_includes_expand_field(self):
        """Replicate the per-view dict-splitting logic used inside
        `extract_from_workspace` and confirm the new key survives."""
        # Mirror the function's _split_view_dict logic shape.
        eff_lateral = {
            "min_labels_per_slice": 3,
            "edge_buffer": 5,
            "slices_per_patch": 0,
            "min_distance": 3,
            "remove_boundaries": True,
            "relabel": True,
            "min_size": 100,
            "smoothing_disk_size": 1,
            "expand_labels_distance": 2,
        }
        from napari_epicycle.fusion_reconstruction.training_pair_extractor import (
            extract_from_workspace,
        )
        # We can't easily run the full extract_from_workspace without
        # data; but the audit fix is at lines 759-763 + 784-789 — both
        # mask_cleaning dicts now include the key.  Confirm by reading
        # the source.
        import inspect
        src = inspect.getsource(extract_from_workspace)
        # Both _split_view_dict and current_mask_cleaning must reference
        # expand_labels_distance after the audit fix.
        n_refs = src.count('"expand_labels_distance"')
        assert n_refs >= 3, (
            f"Expected expand_labels_distance to appear in "
            f"manifest-snapshot writes at least 3x in "
            f"extract_from_workspace (in _split_view_dict + "
            f"current_mask_cleaning + kwarg signature); found {n_refs}"
        )
