"""Tests for ``_PatchLoadWorker`` — the off-main-thread patch loader for the
Step 3 QC-view patch switch (fix/step3-patch-switch-perf, 2026-06-05).

These call ``worker.run()`` SYNCHRONOUSLY (no QThread) and capture the
``loaded`` signal payload via a connected list, so they need no display and
run in CI alongside the rest of ``tests/fusion_reconstruction/``.

Coverage:
- factor 1 (no downsample): labels + source pass through unchanged in shape;
  source is converted to channel-first.
- factor 2 (downsample on): labels are majority-downsampled (Z,Y,X // 2);
  source is channel-first then spatial-downsampled (channel axis preserved).
- ``need_source=False``: source is None in the payload.
- ``original_labels_shape`` reflects the on-disk (pre-downsample) shape.
"""

import numpy as np
import pytest
import tifffile

from napari_epicycle.fusion_reconstruction.controller import _PatchLoadWorker
from napari_epicycle.fusion_reconstruction.qc_features import detect_axes_order


def _to_channel_first(arr):
    """Mirror of ``Controller._to_channel_first`` (pure helper).

    Reproduced here so the test exercises the worker without constructing the
    full Qt controller.  Kept verbatim with the production logic: detect the
    per-array axes order, move the channel axis to the front for Z-first 4D.
    """
    if getattr(arr, "ndim", 0) != 4:
        return arr
    order = detect_axes_order(arr.shape)
    if order == "ZCYX":
        return np.moveaxis(arr, 1, 0)
    return arr


@pytest.fixture
def patch_tiffs(tmp_path):
    """Write a small synthetic labels TIFF + Z-first 4D source TIFF.

    Labels: 8x8x8 int32 (divisible by 2 so factor-2 shapes are exact).
    Source: Z-first ``(Z=8, C=2, Y=8, X=8)`` — chosen so ``_to_channel_first``
    must transpose it to ``(C=2, Z=8, Y=8, X=8)``.
    """
    labels = np.zeros((8, 8, 8), dtype=np.int32)
    # A few blobs so majority-pool has something to chew on.
    labels[0:4, 0:4, 0:4] = 1
    labels[4:8, 4:8, 4:8] = 2
    labels_path = tmp_path / "labels_3d.tif"
    tifffile.imwrite(str(labels_path), labels)

    # Z-first source: (Z, C, Y, X). C=2 is the smallest axis → detect ZCYX.
    source = np.arange(8 * 2 * 8 * 8, dtype=np.float32).reshape(8, 2, 8, 8)
    source_path = tmp_path / "source_stack.tif"
    tifffile.imwrite(str(source_path), source)

    return labels_path, source_path, labels.shape


def _run_worker_capture(worker):
    """Run the worker synchronously, returning the emitted payload (or raising)."""
    captured = []
    errors = []
    worker.loaded.connect(captured.append)
    worker.error.connect(errors.append)
    worker.run()
    assert not errors, f"worker emitted error: {errors}"
    assert len(captured) == 1, f"expected exactly one loaded emit, got {len(captured)}"
    return captured[0]


def test_factor1_passthrough_and_channel_first(patch_tiffs):
    labels_path, source_path, orig_shape = patch_tiffs
    worker = _PatchLoadWorker(
        str(labels_path), str(source_path),
        need_source=True, downsample_factor=1,
        to_channel_first=_to_channel_first,
    )
    res = _run_worker_capture(worker)

    # Labels unchanged at factor 1.
    assert res["labels"].shape == (8, 8, 8)
    assert res["downsample_factor"] == 1
    assert res["original_labels_shape"] == orig_shape

    # Source converted Z-first (8,2,8,8) -> channel-first (2,8,8,8), no spatial
    # downsample at factor 1.
    assert res["source"] is not None
    assert res["source"].shape == (2, 8, 8, 8)


def test_factor2_downsamples_labels_and_source(patch_tiffs):
    labels_path, source_path, orig_shape = patch_tiffs
    worker = _PatchLoadWorker(
        str(labels_path), str(source_path),
        need_source=True, downsample_factor=2,
        to_channel_first=_to_channel_first,
    )
    res = _run_worker_capture(worker)

    # Labels majority-downsampled by 2 in every spatial axis.
    assert res["labels"].shape == (4, 4, 4)
    assert res["downsample_factor"] == 2
    # original_labels_shape is the ON-DISK shape, NOT the downsampled one.
    assert res["original_labels_shape"] == orig_shape
    # Majority pool never invents new label IDs.
    assert set(np.unique(res["labels"])).issubset({0, 1, 2})

    # Source: channel-first (2,8,8,8) -> spatial-downsampled (2,4,4,4),
    # channel axis preserved.
    assert res["source"] is not None
    assert res["source"].shape == (2, 4, 4, 4)


def test_need_source_false_yields_none_source(patch_tiffs):
    labels_path, source_path, orig_shape = patch_tiffs
    worker = _PatchLoadWorker(
        str(labels_path), str(source_path),
        need_source=False, downsample_factor=2,
        to_channel_first=_to_channel_first,
    )
    res = _run_worker_capture(worker)

    assert res["source"] is None
    assert res["labels"].shape == (4, 4, 4)
    assert res["original_labels_shape"] == orig_shape


def test_error_signal_on_missing_file(tmp_path):
    """A bad path makes the worker emit ``error`` (not ``loaded``)."""
    worker = _PatchLoadWorker(
        str(tmp_path / "does_not_exist.tif"), "",
        need_source=False, downsample_factor=1,
        to_channel_first=_to_channel_first,
    )
    captured = []
    errors = []
    worker.loaded.connect(captured.append)
    worker.error.connect(errors.append)
    worker.run()
    assert not captured
    assert len(errors) == 1
    assert "Traceback" in errors[0] or "Error" in errors[0] or "No such" in errors[0]
