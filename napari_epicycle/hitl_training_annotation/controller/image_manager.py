import tifffile
import numpy as np
import os
import json

class ImageManager:
    def __init__(self):
        self.loaded_images = {}  # Dict of {filename: numpy array}

    def load_patch_with_metadata(self, image_path):
        """
        Loads a patch and its associated metadata JSON file from previous data
        preparation results. Accepts both multi-channel (C, Z, Y, X) and
        single-channel grayscale (Z, Y, X) TIFFs; grayscale is auto-promoted
        to (1, Z, Y, X) so downstream channel-iteration logic works uniformly.

        Returns:
            dict: {channel_name: image3D} where image3D shape is (Z, Y, X)
            metadata: dict parsed from associated JSON
        """
        ext = os.path.splitext(image_path)[1].lower()
        if ext not in [".tif", ".tiff"]:
            raise ValueError(f"Unsupported file format for metadata loading: {ext}")

        # Load image using tifffile
        image = tifffile.imread(image_path)

        # Auto-promote grayscale (Z, Y, X) to single-channel (1, Z, Y, X)
        # so the rest of the pipeline (which iterates over channels) works.
        if image.ndim == 3:
            image = image[np.newaxis, ...]

        if image.ndim != 4:
            raise ValueError(
                f"Expected image to have shape (C, Z, Y, X) or (Z, Y, X), got {image.shape}"
            )

        # Load metadata from <base>_metadata.json
        meta_path = os.path.splitext(image_path)[0] + "_metadata.json"
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata JSON not found: {meta_path}")

        with open(meta_path, "r") as f:
            metadata = json.load(f)
        
        channel_names = metadata.get("channel_names", [f"channel_{i}" for i in range(image.shape[0])])
        patch_name = metadata.get("patch_name")

        if len(channel_names) != image.shape[0]:
            raise ValueError("Mismatch between number of channels in image and metadata channel_names")

        # Return per-channel 3D volumes
        channels = {
            name: image[i] for i, name in enumerate(channel_names)
        }

        return channels, metadata, patch_name

    def convert_to_rgb(
        self,
        image: np.ndarray,
        channel_indices: list = [0, 1],
        normalize: bool = True,
        is_stack: bool = False
    ) -> np.ndarray:
        """
        Converts a multi-channel image or stack to an RGB image or stack. Performs global percentile clipping.
        """
        if is_stack:
            Z, C, Y, X = image.shape
            rgb_stack = []
            for z in range(Z):
                rgb = self._convert_channels_to_rgb(image[z], channel_indices, normalize)
                rgb_stack.append(rgb)
            return np.stack(rgb_stack, axis=0)  # (Z, Y, X, 3)
        else:
            return self._convert_channels_to_rgb(image, channel_indices, normalize)  # (Y, X, 3)

    # should do global percentile clipping -- should scale better?
    def _convert_channels_to_rgb(self, img_cyx: np.ndarray, channel_indices: list, normalize: bool) -> np.ndarray:
        C, Y, X = img_cyx.shape
        rgb_channels = []

        for i in range(3):
            if i < len(channel_indices):
                ch_idx = channel_indices[i]
                if ch_idx >= C:
                    raise IndexError(f"Channel index {ch_idx} out of bounds (got {C} channels)")
                channel = img_cyx[ch_idx]

                if normalize:
                    # Clip extremes for better contrast (2nd–98th percentile)
                    vmin = np.percentile(channel, 2)
                    vmax = np.percentile(channel, 98)
                    if vmax > vmin:
                        channel = np.clip((channel - vmin) / (vmax - vmin), 0, 1)
                    else:
                        channel = np.zeros_like(channel)
                rgb_channels.append(channel)
            else:
                rgb_channels.append(np.zeros((Y, X)))

        rgb = np.stack(rgb_channels, axis=-1)  # (Y, X, 3)
        return (rgb * 255).round().astype(np.uint8)



    def clear(self):
        self.loaded_images.clear()
