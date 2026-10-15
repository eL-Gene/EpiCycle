"""
Z-stack tiling utilities.

Adapted from ``zstack_patchification.py`` — splits a large Z-stack TIFF
(Z, C, Y, X) into square XY patches with configurable overlap, computes
per-patch Z-ranges via intensity thresholding, and saves each tile.
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ZStackTiler:
    """
    Split a large Z-stack TIFF (Z, C, Y, X) into overlapping XY tiles.

    Features
    --------
    - Computes valid patch + step sizes given overlap (no padding)
    - Auto-corrects patch size if exact tiling is infeasible
    - Detects per-tile Z-range using intensity thresholds
    - Saves tiles into individual subdirectories with metadata

    Parameters
    ----------
    tif_path : path to the input Z-stack TIFF
    output_dir : root directory for tile output
    target_patch_size : desired XY patch size in pixels
    overlap : fractional overlap between adjacent tiles (0-1)
    z_threshold_method : 'hard', 'percentile', or 'default'
    threshold_value : threshold for Z-range detection
    channel_index : which channel to use for Z-range intensity analysis
    search_range : +/- pixels to search around target_patch_size
    auto_correct : if True, find a fallback patch size when exact tiling fails
    """

    def __init__(
        self,
        tif_path,
        output_dir,
        target_patch_size=1024,
        overlap=0.25,
        z_threshold_method='hard',
        threshold_value=0.07,
        channel_index=1,
        search_range=200,
        auto_correct=False,
        input_axes="auto",
    ):
        self.tif_path = str(tif_path)
        self.output_dir = str(output_dir)
        self.target_patch_size = target_patch_size
        self.overlap = overlap
        self.z_threshold_method = z_threshold_method
        self.threshold_value = threshold_value
        self.channel_index = channel_index
        self.search_range = search_range
        self.auto_correct = auto_correct

        os.makedirs(output_dir, exist_ok=True)
        import tifffile
        from packages.epicycle_scaledseg.utils.axes import canonicalize_to_zcyx, resolve_input_axes
        logger.info(f"Loading z-stack: {self.tif_path}")
        self.zstack = tifffile.imread(self.tif_path)
        self.original_dtype = self.zstack.dtype

        # Resolve the input axis order ONCE and canonicalize to the engine's
        # ZCYX form, so the positional unpack below (and every downstream stage)
        # is correct by construction. 3D ZYX -> C=1; CZYX -> Z/C swapped back.
        resolved = (
            input_axes
            if str(input_axes).lower() != "auto"
            else resolve_input_axes(self.zstack, image_path=self.tif_path, fallback="ZCYX")
        )
        self.input_axes = resolved
        self.zstack = canonicalize_to_zcyx(self.zstack, resolved)
        self.Z, self.C, self.H, self.W = self.zstack.shape

        # A grayscale (C=1) stack — e.g. canonicalized 3D ZYX input — has no
        # channel index 1; clamp so per-tile Z-range intensity slicing stays in
        # bounds. (Downstream Cellpose channels=[2,1] vs [0,0] for true
        # grayscale is a documented follow-up, not handled here.)
        if self.channel_index > self.C - 1:
            logger.warning(
                f"channel_index={self.channel_index} out of range for C={self.C}; "
                f"clamping to {self.C - 1}"
            )
            self.channel_index = self.C - 1

        logger.info(f"Z-stack shape (canonical ZCYX): Z={self.Z}, C={self.C}, "
                    f"H={self.H}, W={self.W}, axes={resolved}, dtype={self.original_dtype}")

        # Flush-anchored tiling (Stack 5) handles ANY (H, W) with no padding and
        # no param mutation, so the old perfect-tiling constraint + auto_correct
        # fallback are gone. The patch size is used as-given; step derives from
        # overlap. (`auto_correct`/`search_range` are kept as accepted no-op
        # kwargs for CLI/back-compat — flush makes them unnecessary.)
        from packages.epicycle_scaledseg.utils.tile_grid import resolve_step
        self.patch_size = int(target_patch_size)
        self.step_size, _ = resolve_step(self.patch_size, self.overlap)
        if auto_correct:
            logger.info(
                "auto_correct is a no-op since Stack 5 (flush-anchored tiling "
                "covers any dimension without mutating patch/overlap)."
            )

    def _find_sample_intensity(self, stack):
        """Detect the Z-range containing actual signal."""
        intensities = np.mean(stack, axis=(1, 2))  # (Z,)
        imin, imax = np.min(intensities), np.max(intensities)
        if imax - imin < 1e-12:
            return 0, len(intensities) - 1
        intensities = (intensities - imin) / (imax - imin)

        if self.z_threshold_method == 'hard':
            sample_range = np.where(intensities > self.threshold_value)[0]
            if len(sample_range) == 0:
                return 0, len(intensities) - 1
            return int(sample_range[0]), int(sample_range[-1])

        elif self.z_threshold_method == 'percentile':
            lower, upper = self.threshold_value
            cumul = np.cumsum(intensities) / np.sum(intensities)
            start = int(np.searchsorted(cumul, lower / 100))
            end = int(np.searchsorted(cumul, upper / 100))
            return start, end

        elif self.z_threshold_method == 'default':
            threshold = np.mean(intensities) + np.std(intensities)
            sample_range = np.where(intensities > threshold)[0]
            if len(sample_range) == 0:
                return 0, len(intensities) - 1
            return int(sample_range[0]), int(sample_range[-1])

        return 0, len(intensities) - 1

    def split_tiles(self):
        """
        Tile the Z-stack and save each tile into its own subdirectory.

        Creates ``output_dir/tile_{row}_{col}/`` with the tile TIFF and
        a ``tile_{row}_{col}_config.json`` containing Z-range metadata.

        Returns
        -------
        list of Path
            Paths to the created tile subdirectories.
        """
        import tifffile
        from packages.epicycle_scaledseg.utils.tile_grid import compute_tile_grid

        grid = compute_tile_grid(self.H, self.W, self.patch_size, self.overlap)
        logger.info(
            f"Tiling: patch_size={self.patch_size}, step={grid.step}, "
            f"overlap={self.overlap} -> {grid.n_rows} rows x {grid.n_cols} cols "
            f"= {len(grid.tiles)} tiles (flush-anchored, no padding)"
        )

        tile_dirs = []
        tile_geometry = []
        for geom in tqdm(grid.tiles, desc="Saving tiles", unit="tile"):
            y0, x0 = geom.origin
            h, w = geom.extent
            # Slice REAL image data only (no patchify, no padding). Sub-patch
            # extents (dim < patch) slice the true image size.
            patch_stack = self.zstack[:, :, y0:y0 + h, x0:x0 + w].copy()  # (Z,C,Y,X)
            intensity_stack = patch_stack[:, self.channel_index, :, :]
            z_start, z_end = self._find_sample_intensity(intensity_stack)

            tile_name = geom.name
            tile_dir = Path(self.output_dir) / tile_name
            tile_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{tile_name}_segmentrange_{z_start}_{z_end}.tif"
            tifffile.imwrite(
                str(tile_dir / filename),
                patch_stack,
                dtype=self.original_dtype,
                imagej=True,
                # Truthful by construction: self.zstack is canonicalized to
                # ZCYX in __init__, so every tile really is ZCYX. (Pre-axis-
                # resolver this tag was a latent lie for CZYX/ZYX input.)
                metadata={"axes": "ZCYX"},
            )

            # Per-tile config: legacy keys + EXPLICIT geometry (origin is STORED,
            # not regex-derived — a flush tile's true origin is dim-patch, not
            # grid_index*step, so assembly must read this, not guess).
            i, j = geom.grid_index
            config = {
                "tile_name": tile_name,
                "row": i,
                "col": j,
                "truncated_range": [z_start, z_end],
                "original_z": int(self.Z),
                "patch_size": self.patch_size,
                "step_size": grid.step,
                "overlap": self.overlap,
                "origin": [int(y0), int(x0)],
                "extent": [int(h), int(w)],
                "grid_index": [int(i), int(j)],
                "is_edge": bool(geom.is_edge),
                "is_flush": bool(geom.is_flush),
            }
            with open(tile_dir / f"{tile_name}_config.json", "w") as f:
                json.dump(config, f, indent=2)

            tile_dirs.append(tile_dir)
            tile_geometry.append({
                "tile_name": tile_name,
                "origin": [int(y0), int(x0)],
                "extent": [int(h), int(w)],
                "grid_index": [int(i), int(j)],
                "is_edge": bool(geom.is_edge),
                "is_flush": bool(geom.is_flush),
            })

        # Global tiling config. `tiles` stays a list of NAMES (back-compat with
        # the TilingStage skip path); `tile_geometry` carries the explicit
        # origins assembly + the Mode-B IoU stitcher consume.
        global_config = {
            "input_tif": self.tif_path,
            "full_size": [int(self.H), int(self.W)],
            "depth": int(self.Z),
            "channels": int(self.C),
            "patch_size": self.patch_size,
            "step_size": grid.step,
            "overlap": self.overlap,
            "overlap_px": int(grid.overlap_px),
            "n_rows": grid.n_rows,
            "n_cols": grid.n_cols,
            "n_tiles": len(grid.tiles),
            "dtype": str(self.original_dtype),
            "input_axes": self.input_axes,
            "tiles": [td.name for td in tile_dirs],
            "tile_geometry": tile_geometry,
        }
        with open(Path(self.output_dir) / "tiling_config.json", "w") as f:
            json.dump(global_config, f, indent=2)

        return tile_dirs
