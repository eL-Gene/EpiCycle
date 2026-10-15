import numpy as np
import tifffile
import json
import pytest
from napari_epicycle.hitl_training_annotation.controller.image_manager import ImageManager

def test_load_patch_with_metadata(tmp_path):
    # Simulated 2-channel (C=2), Z=4, Y=16, X=16
    img = np.random.rand(2, 4, 16, 16).astype(np.float32)
    tiff_path = tmp_path / "test_patch.tiff"
    tifffile.imwrite(tiff_path, img)

    # Create corresponding metadata file
    metadata = {
        "patch_name": "test_patch",
        "channel_names": ["nuclei", "membrane"],
        "original_shape": list(img.shape)
    }
    metadata_path = tmp_path / "test_patch_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    # Load via ImageManager
    manager = ImageManager()
    channels, loaded_metadata, patch_name = manager.load_patch_with_metadata(str(tiff_path))

    # --- Assertions ---
    assert isinstance(channels, dict)
    assert list(channels.keys()) == metadata["channel_names"]
    assert all(ch.shape == (4, 16, 16) for ch in channels.values())  # Each should be 3D

    # Check data consistency per channel
    for i, name in enumerate(metadata["channel_names"]):
        np.testing.assert_allclose(channels[name], img[i])

    assert patch_name == "test_patch"
    assert loaded_metadata["original_shape"] == metadata["original_shape"]


def test_load_grayscale_patch_promotes_to_single_channel(tmp_path):
    """3D grayscale TIFF (Z, Y, X) should auto-promote to (1, Z, Y, X).

    Regression test: previously raised "Expected image to have shape (C, Z, Y, X)".
    """
    # Grayscale: shape (Z=4, Y=16, X=16) — no channel dim
    img = np.random.rand(4, 16, 16).astype(np.float32)
    tiff_path = tmp_path / "gray_patch.tiff"
    tifffile.imwrite(tiff_path, img)

    metadata = {
        "patch_name": "gray_patch",
        "channel_names": ["intensity"],
        "original_shape": list(img.shape),
    }
    with open(tmp_path / "gray_patch_metadata.json", "w") as f:
        json.dump(metadata, f)

    manager = ImageManager()
    channels, loaded_metadata, patch_name = manager.load_patch_with_metadata(str(tiff_path))

    # Single channel returned, named per metadata, with original 3D shape preserved.
    assert list(channels.keys()) == ["intensity"]
    assert channels["intensity"].shape == (4, 16, 16)
    np.testing.assert_allclose(channels["intensity"], img)
    assert patch_name == "gray_patch"


def test_load_grayscale_patch_uses_default_channel_name(tmp_path):
    """Grayscale TIFF with metadata missing channel_names should fall back to 'channel_0'."""
    img = np.random.rand(4, 16, 16).astype(np.float32)
    tiff_path = tmp_path / "gray_default.tiff"
    tifffile.imwrite(tiff_path, img)

    # Metadata with no channel_names key
    metadata = {"patch_name": "gray_default", "original_shape": list(img.shape)}
    with open(tmp_path / "gray_default_metadata.json", "w") as f:
        json.dump(metadata, f)

    manager = ImageManager()
    channels, _, _ = manager.load_patch_with_metadata(str(tiff_path))
    assert list(channels.keys()) == ["channel_0"]
    assert channels["channel_0"].shape == (4, 16, 16)


def test_load_5d_patch_still_rejected(tmp_path):
    """Non-3D-non-4D shapes (e.g., 5D time-series) should still raise."""
    img = np.random.rand(2, 2, 4, 16, 16).astype(np.float32)
    tiff_path = tmp_path / "weird.tiff"
    tifffile.imwrite(tiff_path, img)
    metadata = {"patch_name": "weird", "channel_names": ["a", "b"]}
    with open(tmp_path / "weird_metadata.json", "w") as f:
        json.dump(metadata, f)

    manager = ImageManager()
    with pytest.raises(ValueError, match="Expected image to have shape"):
        manager.load_patch_with_metadata(str(tiff_path))


def test_load_grayscale_patch_with_mismatched_metadata_raises(tmp_path):
    """If metadata says 2 channels but file is 1-channel grayscale, must error."""
    img = np.random.rand(4, 16, 16).astype(np.float32)  # grayscale
    tiff_path = tmp_path / "mismatch.tiff"
    tifffile.imwrite(tiff_path, img)

    metadata = {"patch_name": "mismatch", "channel_names": ["nuclei", "membrane"]}
    with open(tmp_path / "mismatch_metadata.json", "w") as f:
        json.dump(metadata, f)

    manager = ImageManager()
    with pytest.raises(ValueError, match="Mismatch between number of channels"):
        manager.load_patch_with_metadata(str(tiff_path))


def test_convert_to_rgb_grayscale_with_single_index():
    """Regression: convert_to_rgb on grayscale (C=1) must NOT raise IndexError
    when channel_indices=[0]. Previously the segment pipeline passed
    hardcoded [0, 1] which crashed on single-channel input.
    """
    manager = ImageManager()
    img_gray = np.random.rand(1, 16, 16).astype(np.float32)  # (C=1, Y, X)
    rgb = manager.convert_to_rgb(image=img_gray, channel_indices=[0], normalize=True, is_stack=False)
    assert rgb.shape == (16, 16, 3)
    assert rgb.dtype == np.uint8


def test_convert_to_rgb_grayscale_with_two_indices_raises():
    """Documents the prior bug: passing [0, 1] to a single-channel image still raises.
    Callers must detect channel count and use [0] for grayscale.
    """
    manager = ImageManager()
    img_gray = np.random.rand(1, 16, 16).astype(np.float32)
    with pytest.raises(IndexError, match="Channel index 1 out of bounds"):
        manager.convert_to_rgb(image=img_gray, channel_indices=[0, 1], normalize=True, is_stack=False)


def test_convert_to_rgb_two_channel_works():
    """Two-channel input with [0, 1] must work (the human-data path)."""
    manager = ImageManager()
    img_2ch = np.random.rand(2, 16, 16).astype(np.float32)
    rgb = manager.convert_to_rgb(image=img_2ch, channel_indices=[0, 1], normalize=True, is_stack=False)
    assert rgb.shape == (16, 16, 3)


def test_run_segmentation_uses_dynamic_channel_indices():
    """Regression: _run_segmentation must compute channel_indices from input
    shape rather than hardcoding [0, 1]. Otherwise grayscale input crashes
    convert_to_rgb -> caught as 'Segmentation failed' -> silent early return,
    skipping _touch_sample, _persist, and _refresh_sample_queue.
    """
    import inspect
    from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
        TrainingAnnotationController,
    )
    src = inspect.getsource(TrainingAnnotationController._run_segmentation)
    # The fix introduces a dynamic computation; assert the magic value isn't
    # baked in as a literal hardcoded list anymore.
    assert "channel_indices=[0,1]" not in src.replace(" ", ""), (
        "_run_segmentation must not hardcode channel_indices=[0,1]; "
        "use dynamic detection so grayscale input doesn't crash convert_to_rgb."
    )
