"""Tests for Phase 8 Tier 3 thumbnail helper.

Qt-aware (uses QIcon / QImage / QPixmap) but no event loop or dialog.
Tests focus on the data-pipeline correctness (read → normalize → resize
→ render) rather than the Qt rendering itself.
"""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

# QImage requires a QApplication for some platforms; skip cleanly if absent
try:
    import qtpy  # noqa: F401
    from qtpy.QtWidgets import QApplication
    qt_available = True
except ImportError:
    qt_available = False


@pytest.fixture(scope="module", autouse=True)
def qapp():
    """Ensure a QApplication exists for QImage operations."""
    if not qt_available:
        return None
    from qtpy.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.mark.skipif(not qt_available, reason="qtpy not available")
class TestMakeImageThumbnail:

    def test_grayscale_2d(self, tmp_path):
        from napari_epicycle.hitl_training_annotation._thumbnails import (
            make_image_thumbnail,
        )
        img = np.random.randint(0, 65535, size=(128, 128), dtype=np.uint16)
        p = tmp_path / "img.tif"
        tifffile.imwrite(str(p), img)
        icon = make_image_thumbnail(str(p), size=48, channel_axis="none")
        assert icon is not None
        assert not icon.isNull()

    def test_channel_first_3d(self, tmp_path):
        from napari_epicycle.hitl_training_annotation._thumbnails import (
            make_image_thumbnail,
        )
        # (C, H, W) = (2, 64, 64)
        img = np.random.randint(0, 255, size=(2, 64, 64), dtype=np.uint8)
        p = tmp_path / "img.tif"
        tifffile.imwrite(str(p), img)
        icon = make_image_thumbnail(str(p), size=32, channel_axis="first")
        assert icon is not None
        assert not icon.isNull()

    def test_channel_last_3d(self, tmp_path):
        from napari_epicycle.hitl_training_annotation._thumbnails import (
            make_image_thumbnail,
        )
        # (H, W, C) = (64, 64, 3)
        img = np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
        p = tmp_path / "img.tif"
        tifffile.imwrite(str(p), img)
        icon = make_image_thumbnail(str(p), size=32, channel_axis="last")
        assert icon is not None

    def test_missing_file_returns_placeholder(self, tmp_path):
        from napari_epicycle.hitl_training_annotation._thumbnails import (
            make_image_thumbnail,
        )
        # Doesn't raise — returns a placeholder icon
        icon = make_image_thumbnail(str(tmp_path / "ghost.tif"), size=48)
        assert icon is not None  # placeholder, not None
        assert not icon.isNull()

    def test_corrupt_file_returns_placeholder(self, tmp_path):
        from napari_epicycle.hitl_training_annotation._thumbnails import (
            make_image_thumbnail,
        )
        p = tmp_path / "corrupt.tif"
        p.write_bytes(b"NOT A TIFF")
        icon = make_image_thumbnail(str(p), size=48)
        assert icon is not None  # placeholder

    def test_large_image_downsampled(self, tmp_path):
        from napari_epicycle.hitl_training_annotation._thumbnails import (
            make_image_thumbnail,
        )
        # 1024×1024 should downsample fine without OOM
        img = np.random.randint(0, 255, size=(1024, 1024), dtype=np.uint8)
        p = tmp_path / "big.tif"
        tifffile.imwrite(str(p), img)
        icon = make_image_thumbnail(str(p), size=48, channel_axis="none")
        assert icon is not None


@pytest.mark.skipif(not qt_available, reason="qtpy not available")
class TestMakeMaskThumbnail:

    def test_label_mask(self, tmp_path):
        from napari_epicycle.hitl_training_annotation._thumbnails import (
            make_mask_thumbnail,
        )
        mask = np.zeros((64, 64), dtype=np.uint16)
        mask[10:30, 10:30] = 1
        mask[40:60, 40:60] = 2
        p = tmp_path / "mask.tif"
        tifffile.imwrite(str(p), mask)
        icon = make_mask_thumbnail(str(p), size=48)
        assert icon is not None
        assert not icon.isNull()

    def test_empty_mask_renders_black(self, tmp_path):
        from napari_epicycle.hitl_training_annotation._thumbnails import (
            make_mask_thumbnail,
        )
        mask = np.zeros((64, 64), dtype=np.uint16)
        p = tmp_path / "empty.tif"
        tifffile.imwrite(str(p), mask)
        icon = make_mask_thumbnail(str(p), size=48)
        assert icon is not None  # not None even when all-zero

    def test_missing_file_returns_placeholder(self, tmp_path):
        from napari_epicycle.hitl_training_annotation._thumbnails import (
            make_mask_thumbnail,
        )
        icon = make_mask_thumbnail(str(tmp_path / "ghost.tif"), size=48)
        assert icon is not None

    def test_many_labels(self, tmp_path):
        from napari_epicycle.hitl_training_annotation._thumbnails import (
            make_mask_thumbnail,
        )
        mask = np.zeros((128, 128), dtype=np.uint16)
        for i in range(1, 100):
            y = (i * 7) % 120
            x = (i * 11) % 120
            mask[y:y+5, x:x+5] = i
        p = tmp_path / "many.tif"
        tifffile.imwrite(str(p), mask)
        icon = make_mask_thumbnail(str(p), size=48)
        assert icon is not None


# ============================================================================
# Cache layer (2026-05-25 Group G)
# ============================================================================
#
# These tests verify the LRU thumbnail cache:
# - hit on identical (path, mtime, size, channel_axis) returns instantly
# - miss on mtime change re-decodes (file was replaced on disk)
# - miss on channel_axis change re-decodes (different axis = different
#   thumbnail, so cache must NOT key only on path)
# - LRU eviction kicks in past _THUMBNAIL_CACHE_SIZE
#
# Each test clears the cache via clear_thumbnail_cache() in setUp to
# avoid leaking entries across cases.

@pytest.mark.skipif(not qt_available, reason="qtpy not available")
class TestThumbnailCache:

    def setup_method(self):
        """Clear cache before each test so counts are deterministic."""
        from napari_epicycle.hitl_training_annotation._thumbnails import (
            clear_thumbnail_cache,
        )
        clear_thumbnail_cache()

    def test_image_cache_hit_on_repeated_call(self, tmp_path):
        from napari_epicycle.hitl_training_annotation._thumbnails import (
            make_image_thumbnail, thumbnail_cache_info,
        )
        img = np.random.randint(0, 65535, size=(128, 128), dtype=np.uint16)
        p = tmp_path / "img.tif"
        tifffile.imwrite(str(p), img)
        # First call: miss
        make_image_thumbnail(str(p), size=48, channel_axis="none")
        hits_1, misses_1, _ = thumbnail_cache_info()
        assert misses_1 >= 1
        # Second identical call: hit (no new miss)
        make_image_thumbnail(str(p), size=48, channel_axis="none")
        hits_2, misses_2, _ = thumbnail_cache_info()
        assert hits_2 == hits_1 + 1
        assert misses_2 == misses_1  # no new miss

    def test_image_cache_invalidates_on_channel_axis_change(self, tmp_path):
        """Two channels of the same file produce DIFFERENT thumbnails.
        Cache key must include channel_axis or we'd return wrong icon."""
        from napari_epicycle.hitl_training_annotation._thumbnails import (
            make_image_thumbnail, thumbnail_cache_info,
        )
        # Create a 3D (C, H, W) image with two visibly-distinct channels
        img = np.zeros((2, 64, 64), dtype=np.uint16)
        img[0] = 1000  # constant low intensity
        img[1] = 50000  # constant high intensity
        p = tmp_path / "img.tif"
        tifffile.imwrite(str(p), img)
        # First call: miss, decodes channel-first
        make_image_thumbnail(str(p), size=48, channel_axis="first")
        _, misses_1, _ = thumbnail_cache_info()
        # Different channel_axis: must be a NEW miss, not a cached hit
        make_image_thumbnail(str(p), size=48, channel_axis="last")
        _, misses_2, _ = thumbnail_cache_info()
        assert misses_2 > misses_1, (
            "channel_axis change must invalidate — otherwise the cached "
            "first-axis thumbnail would be returned for last-axis lookup"
        )

    def test_image_cache_invalidates_on_size_change(self, tmp_path):
        """Different size = different thumbnail; cache key must include
        size so the user can request multiple sizes without collision."""
        from napari_epicycle.hitl_training_annotation._thumbnails import (
            make_image_thumbnail, thumbnail_cache_info,
        )
        img = np.random.randint(0, 65535, size=(128, 128), dtype=np.uint16)
        p = tmp_path / "img.tif"
        tifffile.imwrite(str(p), img)
        make_image_thumbnail(str(p), size=48, channel_axis="none")
        _, misses_1, _ = thumbnail_cache_info()
        make_image_thumbnail(str(p), size=96, channel_axis="none")
        _, misses_2, _ = thumbnail_cache_info()
        assert misses_2 > misses_1

    def test_cache_invalidates_on_mtime_change(self, tmp_path):
        """Writing new content to the same path bumps mtime, which
        must produce a cache miss (the cached icon is now stale)."""
        from napari_epicycle.hitl_training_annotation._thumbnails import (
            make_mask_thumbnail, thumbnail_cache_info,
        )
        import os
        import time
        p = tmp_path / "mask.tif"
        # Write v1
        m1 = np.zeros((64, 64), dtype=np.uint16)
        m1[10:30, 10:30] = 1
        tifffile.imwrite(str(p), m1)
        make_mask_thumbnail(str(p), size=48)
        _, misses_1, _ = thumbnail_cache_info()
        # Sleep enough for filesystem mtime granularity (Windows NTFS is
        # ~2ms with st_mtime_ns; FAT32 is 2s).  10ms is more than enough.
        time.sleep(0.05)
        # Write v2 (different content -> different ideal thumbnail)
        m2 = np.zeros((64, 64), dtype=np.uint16)
        m2[40:60, 40:60] = 5
        tifffile.imwrite(str(p), m2)
        # Force mtime update (some filesystems may keep stat cache)
        st = os.stat(str(p))
        assert st.st_mtime_ns > 0
        make_mask_thumbnail(str(p), size=48)
        _, misses_2, _ = thumbnail_cache_info()
        assert misses_2 > misses_1, (
            "mtime change must produce cache miss — otherwise edits "
            "to the file would not refresh the thumbnail"
        )

    def test_mask_cache_hit_on_repeated_call(self, tmp_path):
        from napari_epicycle.hitl_training_annotation._thumbnails import (
            make_mask_thumbnail, thumbnail_cache_info,
        )
        mask = np.zeros((64, 64), dtype=np.uint16)
        mask[10:30, 10:30] = 1
        p = tmp_path / "mask.tif"
        tifffile.imwrite(str(p), mask)
        make_mask_thumbnail(str(p), size=48)
        hits_1, _, _ = thumbnail_cache_info()
        make_mask_thumbnail(str(p), size=48)
        hits_2, _, _ = thumbnail_cache_info()
        assert hits_2 == hits_1 + 1

    def test_clear_cache_drops_entries(self, tmp_path):
        from napari_epicycle.hitl_training_annotation._thumbnails import (
            make_image_thumbnail, clear_thumbnail_cache, thumbnail_cache_info,
        )
        img = np.random.randint(0, 65535, size=(64, 64), dtype=np.uint16)
        p = tmp_path / "img.tif"
        tifffile.imwrite(str(p), img)
        make_image_thumbnail(str(p), size=48, channel_axis="none")
        _, _, size_before = thumbnail_cache_info()
        assert size_before >= 1
        clear_thumbnail_cache()
        _, _, size_after = thumbnail_cache_info()
        assert size_after == 0

    def test_missing_file_returns_placeholder_without_caching(self, tmp_path):
        """A missing file early-returns _placeholder_icon BEFORE hitting
        the cache layer (no path to stat).  Verify the cache doesn't
        accidentally store placeholders for missing files (which would
        prevent re-trying if the file appears later)."""
        from napari_epicycle.hitl_training_annotation._thumbnails import (
            make_image_thumbnail, thumbnail_cache_info,
        )
        ghost_path = str(tmp_path / "does_not_exist.tif")
        icon = make_image_thumbnail(ghost_path, size=48, channel_axis="first")
        assert icon is not None  # placeholder returned
        # Cache should NOT have an entry for the missing file
        # (we early-return before _cached_image_thumbnail)
        _, _, size_after = thumbnail_cache_info()
        assert size_after == 0, (
            "missing-file path should bypass the cache so the user can "
            "later create the file and have it picked up"
        )
