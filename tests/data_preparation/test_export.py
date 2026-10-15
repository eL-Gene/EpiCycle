"""Tests for Step 1 patch persistence (persist_patch).

Real filesystem, no Qt. Verifies the export cutover writes into
artifacts/step1/patches/, keeps the sidecar schema frozen (Viz contract),
and registers patches + provenance in step1_state.json.
"""
import json
import os
import tempfile

import numpy as np
import pytest
import tifffile

from napari_epicycle.data_preparation._export import (
    crop_patch_in_place,
    persist_patch,
)
from napari_epicycle.data_preparation._paths import (
    step1_patches_dir,
    step1_state_path,
)
from napari_epicycle.data_preparation._state import load_state

_SIDECAR_FIELDS = {
    "patch_name", "channel_names", "original_shape",
    "cropped_shape", "selected_z_range", "bounding_box",
}


def _data():
    return np.random.randint(0, 255, (2, 5, 16, 16), dtype=np.uint8)


def _persist(ws, name="kidney_patch_1", **overrides):
    kwargs = dict(
        channel_names=["ch0", "ch1"],
        original_shape=(2, 10, 16, 16),
        cropped_shape=(2, 5, 16, 16),
        selected_z_range=[2, 7],
        bounding_box=[[0, 0], [0, 16], [16, 16], [16, 0]],
        source_path="D:/data/kidney",
        source_display_name="kidney",
        timestamp="2026-06-29T00:00:00Z",
    )
    kwargs.update(overrides)
    return persist_patch(ws, name, _data(), **kwargs)


class TestPersistLocation:
    def test_writes_into_patches_dir(self):
        with tempfile.TemporaryDirectory() as ws:
            res = _persist(ws)
            pdir = step1_patches_dir(ws)
            assert (pdir / "kidney_patch_1.tiff").exists()
            assert (pdir / "kidney_patch_1_metadata.json").exists()
            # NOT at workspace root anymore
            assert not os.path.exists(os.path.join(ws, "kidney_patch_1.tiff"))
            assert res["patch_name"] == "kidney_patch_1"

    def test_tiff_is_readable_4d(self):
        with tempfile.TemporaryDirectory() as ws:
            res = _persist(ws)
            loaded = tifffile.imread(res["tiff_path"])
            assert loaded.ndim == 4
            assert loaded.shape == (2, 5, 16, 16)


class TestSidecarContract:
    def test_sidecar_schema_frozen(self):
        with tempfile.TemporaryDirectory() as ws:
            res = _persist(ws)
            meta = json.loads(open(res["sidecar_path"]).read())
            # Exactly the historical fields — no more, no less (Viz contract).
            assert set(meta.keys()) == _SIDECAR_FIELDS
            assert meta["patch_name"] == "kidney_patch_1"
            assert meta["channel_names"] == ["ch0", "ch1"]
            assert meta["original_shape"] == [2, 10, 16, 16]
            assert meta["cropped_shape"] == [2, 5, 16, 16]
            assert meta["selected_z_range"] == [2, 7]
            assert meta["bounding_box"] == [[0, 0], [0, 16], [16, 16], [16, 0]]

    def test_no_geometry_in_state(self):
        """State registry must NOT duplicate sidecar geometry."""
        with tempfile.TemporaryDirectory() as ws:
            _persist(ws)
            st = load_state(ws)
            entry = st["patches"][0]
            assert "bounding_box" not in entry
            assert "selected_z_range" not in entry
            assert "channel_names" not in entry


class TestRegistry:
    def test_registers_source_and_patch(self):
        with tempfile.TemporaryDirectory() as ws:
            res = _persist(ws)
            st = load_state(ws)
            assert len(st["sources"]) == 1
            assert st["sources"][0]["path"] == "D:/data/kidney"
            assert len(st["patches"]) == 1
            p = st["patches"][0]
            assert p["patch_name"] == "kidney_patch_1"
            assert p["source_id"] == res["source_id"]
            assert p["source_id"] == st["sources"][0]["source_id"]

    def test_relative_paths_are_posix_and_relative(self):
        with tempfile.TemporaryDirectory() as ws:
            _persist(ws)
            p = load_state(ws)["patches"][0]
            assert p["relative_path"] == "artifacts/step1/patches/kidney_patch_1.tiff"
            assert p["sidecar_path"] == "artifacts/step1/patches/kidney_patch_1_metadata.json"
            assert not os.path.isabs(p["relative_path"])

    def test_state_file_created_at_workspace_root(self):
        with tempfile.TemporaryDirectory() as ws:
            _persist(ws)
            assert step1_state_path(ws).exists()

    def test_multiple_patches_accumulate(self):
        with tempfile.TemporaryDirectory() as ws:
            _persist(ws, name="kidney_patch_1")
            _persist(ws, name="kidney_patch_2")
            st = load_state(ws)
            assert {p["patch_name"] for p in st["patches"]} == {
                "kidney_patch_1", "kidney_patch_2"}

    def test_source_deduped_across_patches(self):
        with tempfile.TemporaryDirectory() as ws:
            _persist(ws, name="kidney_patch_1", source_path="D:/data/kidney")
            _persist(ws, name="kidney_patch_2", source_path="D:/data/kidney")
            st = load_state(ws)
            assert len(st["sources"]) == 1
            sids = {p["source_id"] for p in st["patches"]}
            assert len(sids) == 1


class TestPersistStoresPrefix:
    """Commit 3 (D-3): persist_patch records the numbering prefix (derived from
    the final name) on the source entry, keyed per-FILE."""

    def test_source_entry_gets_numbering_prefix(self):
        with tempfile.TemporaryDirectory() as ws:
            _persist(ws, name="kidney_patch_1", source_path="D:/data/kidney.tif")
            st = load_state(ws)
            assert st["sources"][0]["prefix"] == "kidney_patch"

    def test_prefix_survives_collision_bump(self):
        with tempfile.TemporaryDirectory() as ws:
            _persist(ws, name="kidney_patch_1", source_path="D:/data/kidney.tif")
            # Force a collision -> bumps to kidney_patch_2, but the prefix is stable.
            _persist(ws, name="kidney_patch_1", source_path="D:/data/kidney.tif")
            st = load_state(ws)
            assert st["sources"][0]["prefix"] == "kidney_patch"


class TestProvenanceOptional:
    def test_no_source_path_still_registers_patch(self):
        with tempfile.TemporaryDirectory() as ws:
            res = _persist(ws, source_path="")
            st = load_state(ws)
            assert st["sources"] == []
            assert len(st["patches"]) == 1
            assert res["source_id"] == ""
            assert st["patches"][0]["source_id"] == ""


class TestCollisionProtection:
    def test_collision_autoincrements(self):
        with tempfile.TemporaryDirectory() as ws:
            r1 = _persist(ws, name="p")
            r2 = _persist(ws, name="p")
            assert r1["patch_name"] == "p"
            assert r2["patch_name"] == "p_2"
            pdir = step1_patches_dir(ws)
            assert (pdir / "p.tiff").exists()
            assert (pdir / "p_2.tiff").exists()
            # Both registered under their final names.
            names = {p["patch_name"] for p in load_state(ws)["patches"]}
            assert names == {"p", "p_2"}


class TestIndexedCollisionExport:
    """Commit 1: an INDEXED-name collision must resolve to the next
    indexer-visible index ({prefix}_{max+1}), NOT a nested {name}_{k}."""

    def test_indexed_collision_lands_indexer_visible(self):
        from napari_epicycle.data_preparation.logic import find_next_patch_index
        with tempfile.TemporaryDirectory() as ws:
            for n in (1, 2, 3):
                _persist(ws, name=f"kidney_patch_{n}")
            r = _persist(ws, name="kidney_patch_3")  # collision on an existing index
            assert r["patch_name"] == "kidney_patch_4"      # not kidney_patch_3_2
            # The new patch is visible to the indexer + badge (the B2/B3 fix).
            assert find_next_patch_index(ws, "kidney_patch") == 5
            names = {p["patch_name"] for p in load_state(ws)["patches"]}
            assert "kidney_patch_4" in names
            assert "kidney_patch_3_2" not in names

    def test_tif_sibling_not_overwritten(self):
        """B5: a legacy single-f .tif at the same slot is a collision;
        persist must NOT reuse its name (which would clobber the sidecar)."""
        with tempfile.TemporaryDirectory() as ws:
            pdir = step1_patches_dir(ws)
            os.makedirs(pdir)
            open(pdir / "kidney_patch_1.tif", "w").close()  # legacy single-f
            r = _persist(ws, name="kidney_patch_1")
            assert r["patch_name"] == "kidney_patch_2"      # bumped, not reused
            assert (pdir / "kidney_patch_1.tif").exists()   # legacy untouched
            assert (pdir / "kidney_patch_2.tiff").exists()


class TestPersistRecordsAcquisitionMetadata:
    """Phase 0: the user-entered voxel spacing and the file's on-disk axis
    order reach the SOURCE registry (provenance).

    The patch DATA is transposed to CZYX at load, so the patch TIFF axes stays
    ``CZYX``; ``source_axes`` records the file's distinct on-disk order, and
    ``voxel_size`` is stored on the source (never in the FROZEN sidecar)."""

    def test_records_voxel_size_on_source(self):
        with tempfile.TemporaryDirectory() as ws:
            _persist(ws, voxel_size=[2.0, 0.5, 0.5])
            src = load_state(ws)["sources"][0]
            assert src["voxel_size"] == [2.0, 0.5, 0.5]

    def test_voxel_size_defaults_none_preserves_behaviour(self):
        with tempfile.TemporaryDirectory() as ws:
            _persist(ws)  # no voxel_size passed
            assert load_state(ws)["sources"][0]["voxel_size"] is None

    def test_voxel_size_never_leaks_into_frozen_sidecar(self):
        with tempfile.TemporaryDirectory() as ws:
            res = _persist(ws, voxel_size=[2.0, 0.5, 0.5])
            meta = json.loads(open(res["sidecar_path"]).read())
            assert set(meta.keys()) == _SIDECAR_FIELDS  # unchanged (Viz contract)
            assert "voxel_size" not in meta

    def test_source_axes_records_on_disk_order_distinct_from_patch(self):
        """A ZCYX file → CZYX patch: patch TIFF stays CZYX, source records ZCYX
        under the schema-v2 `source_axes` field (the anti-lie home)."""
        with tempfile.TemporaryDirectory() as ws:
            _persist(ws, axes="CZYX", source_axes="ZCYX")
            assert load_state(ws)["sources"][0]["source_axes"] == "ZCYX"

    def test_source_axes_defaults_to_patch_axes(self):
        with tempfile.TemporaryDirectory() as ws:
            _persist(ws, axes="ZCYX")  # no source_axes -> falls back to axes
            assert load_state(ws)["sources"][0]["source_axes"] == "ZCYX"


# --------------------------------------------------------------------------- #
# ④ Z-crop TRIM — the pure, recoverable crop_patch_in_place writer
# --------------------------------------------------------------------------- #

def _write_patch(ws, name, data, *, selected_z_range, channel_names, axes):
    """Persist a patch and return (tiff_path, sidecar_path)."""
    original_shape = tuple(data.shape)
    res = persist_patch(
        ws, name, data,
        channel_names=channel_names,
        original_shape=original_shape,
        cropped_shape=data.shape,
        selected_z_range=selected_z_range,
        bounding_box=[[0, 0], [0, 16], [16, 16], [16, 0]],
        source_path="D:/data/x", source_display_name="x",
        axes=axes,
    )
    return res["tiff_path"], res["sidecar_path"]


class TestCropPatchInPlace:
    def test_4d_czyx_crop_pixels_and_sidecar(self):
        with tempfile.TemporaryDirectory() as ws:
            data = np.random.randint(0, 60000, (2, 10, 16, 16), dtype=np.uint16)
            tp, sp = _write_patch(
                ws, "p", data, selected_z_range=[0, 10],
                channel_names=["a", "b"], axes="CZYX")
            out = crop_patch_in_place(tp, sp, 2, 7)

            arr = tifffile.imread(tp, squeeze=False)
            assert arr.shape == (2, 5, 16, 16)           # Z 10 -> 5
            assert np.array_equal(arr, data[:, 2:7])     # pixels = data[:, z0:z1]
            meta = json.loads(open(sp).read())
            assert meta["cropped_shape"] == [2, 5, 16, 16]
            # prev_z0 == 0 -> absolute range == current-coords range
            assert meta["selected_z_range"] == [2, 7]
            # unchanged provenance fields
            assert meta["original_shape"] == [2, 10, 16, 16]
            assert meta["channel_names"] == ["a", "b"]
            assert meta["patch_name"] == "p"
            assert set(meta.keys()) == _SIDECAR_FIELDS     # schema still frozen
            assert out["cropped_shape"] == [2, 5, 16, 16]
            assert out["selected_z_range"] == [2, 7]

    def test_3d_zyx_crop(self):
        with tempfile.TemporaryDirectory() as ws:
            data = np.random.randint(0, 60000, (10, 16, 16), dtype=np.uint16)
            tp, sp = _write_patch(
                ws, "p3", data, selected_z_range=[0, 10],
                channel_names=["a"], axes="ZYX")
            crop_patch_in_place(tp, sp, 3, 8)
            arr = tifffile.imread(tp, squeeze=False)
            assert arr.shape == (5, 16, 16)              # 3D crops axis 0
            assert np.array_equal(arr, data[3:8])
            meta = json.loads(open(sp).read())
            assert meta["cropped_shape"] == [5, 16, 16]
            assert meta["selected_z_range"] == [3, 8]

    def test_single_channel_4d_c1_stays_4d(self):
        """squeeze=False preserves a singleton C=1 (CZYX stays 4D, HIGH-4)."""
        with tempfile.TemporaryDirectory() as ws:
            data = np.random.randint(0, 60000, (1, 10, 16, 16), dtype=np.uint16)
            tp, sp = _write_patch(
                ws, "p1c", data, selected_z_range=[0, 10],
                channel_names=["only"], axes="CZYX")
            crop_patch_in_place(tp, sp, 2, 6)
            arr = tifffile.imread(tp, squeeze=False)
            assert arr.ndim == 4                          # NOT squeezed to 3D
            assert arr.shape == (1, 4, 16, 16)
            assert np.array_equal(arr, data[:, 2:6])

    def test_absolute_offset_composition_on_recrop(self):
        """A re-crop composes: prev_z0 offsets the new absolute range (HIGH-2)."""
        with tempfile.TemporaryDirectory() as ws:
            data = np.random.randint(0, 60000, (2, 10, 8, 8), dtype=np.uint16)
            tp, sp = _write_patch(
                ws, "pr", data, selected_z_range=[0, 10],
                channel_names=["a", "b"], axes="CZYX")
            crop_patch_in_place(tp, sp, 2, 7)            # abs [2, 7]
            crop_patch_in_place(tp, sp, 1, 4)            # current coords -> abs [3, 6]
            meta = json.loads(open(sp).read())
            assert meta["selected_z_range"] == [3, 6]
            arr = tifffile.imread(tp, squeeze=False)
            assert np.array_equal(arr, data[:, 2:7][:, 1:4])

    def test_absolute_range_honours_prior_nonzero_z0(self):
        """A patch whose sidecar already records a non-zero source-Z start keeps
        composing from THAT start (not from 0)."""
        with tempfile.TemporaryDirectory() as ws:
            data = np.random.randint(0, 60000, (2, 10, 8, 8), dtype=np.uint16)
            tp, sp = _write_patch(
                ws, "pz", data, selected_z_range=[20, 30],
                channel_names=["a", "b"], axes="CZYX")
            crop_patch_in_place(tp, sp, 2, 7)            # abs [22, 27]
            meta = json.loads(open(sp).read())
            assert meta["selected_z_range"] == [22, 27]

    def test_bak_removed_on_success(self):
        with tempfile.TemporaryDirectory() as ws:
            data = np.random.randint(0, 60000, (2, 8, 8, 8), dtype=np.uint16)
            tp, sp = _write_patch(
                ws, "pb", data, selected_z_range=[0, 8],
                channel_names=["a", "b"], axes="CZYX")
            crop_patch_in_place(tp, sp, 1, 5)
            assert not os.path.exists(tp + ".bak")       # cleaned up
            # no stray temps left in the patches dir either
            pdir = step1_patches_dir(ws)
            leftovers = [p for p in os.listdir(pdir)
                         if p.endswith(".tmp")]
            assert leftovers == []

    def test_rejects_ndim_2(self):
        with tempfile.TemporaryDirectory() as ws:
            pdir = step1_patches_dir(ws)
            os.makedirs(pdir)
            tp = str(pdir / "flat.tiff")
            sp = str(pdir / "flat_metadata.json")
            tifffile.imwrite(tp, np.zeros((16, 16), dtype=np.uint16),
                             metadata={"axes": "YX"})
            json.dump({"patch_name": "flat", "channel_names": [],
                       "original_shape": [16, 16], "cropped_shape": [16, 16],
                       "selected_z_range": [0, 0],
                       "bounding_box": []}, open(sp, "w"))
            with pytest.raises(ValueError):
                crop_patch_in_place(tp, sp, 0, 8)

    def test_rejects_inverted_or_oob_range(self):
        with tempfile.TemporaryDirectory() as ws:
            data = np.random.randint(0, 60000, (2, 6, 8, 8), dtype=np.uint16)
            tp, sp = _write_patch(
                ws, "po", data, selected_z_range=[0, 6],
                channel_names=["a", "b"], axes="CZYX")
            with pytest.raises(ValueError):
                crop_patch_in_place(tp, sp, 5, 2)        # inverted
            with pytest.raises(ValueError):
                crop_patch_in_place(tp, sp, 0, 99)       # beyond depth
            # the failed attempts left the TIFF + sidecar untouched
            assert tifffile.imread(tp, squeeze=False).shape == (2, 6, 8, 8)
            assert json.loads(open(sp).read())["cropped_shape"] == [2, 6, 8, 8]

    def test_sidecar_write_failure_rolls_back_the_tiff(self, monkeypatch):
        """HIGH-1: if the sidecar os.replace fails, the ORIGINAL full-Z TIFF is
        restored from .bak and the error propagates — never a trimmed TIFF under
        stale metadata."""
        import napari_epicycle.data_preparation._export as X

        with tempfile.TemporaryDirectory() as ws:
            data = np.random.randint(0, 60000, (2, 10, 8, 8), dtype=np.uint16)
            tp, sp = _write_patch(
                ws, "pf", data, selected_z_range=[0, 10],
                channel_names=["a", "b"], axes="CZYX")
            before_tiff = tifffile.imread(tp, squeeze=False)
            before_meta = json.loads(open(sp).read())

            real_replace = os.replace
            calls = {"n": 0}

            def _flaky_replace(src, dst):
                # 1st replace: original -> .bak (allow).  2nd: temp_tiff -> tiff
                # (allow).  3rd: temp_sidecar -> sidecar (FAIL) -> triggers the
                # TIFF rollback (4th replace, .bak -> tiff), which must succeed.
                calls["n"] += 1
                if calls["n"] == 3:
                    raise OSError("simulated sidecar replace failure")
                return real_replace(src, dst)

            monkeypatch.setattr(X.os, "replace", _flaky_replace)
            with pytest.raises(OSError):
                crop_patch_in_place(tp, sp, 2, 7)

            # The TIFF was rolled back to the FULL-Z original (not the trimmed
            # one) and the sidecar is unchanged; no .bak / temp leaked.
            after_tiff = tifffile.imread(tp, squeeze=False)
            assert after_tiff.shape == before_tiff.shape == (2, 10, 8, 8)
            assert np.array_equal(after_tiff, before_tiff)
            assert json.loads(open(sp).read()) == before_meta
            assert not os.path.exists(tp + ".bak")
            pdir = step1_patches_dir(ws)
            assert [p for p in os.listdir(pdir) if p.endswith(".tmp")] == []
