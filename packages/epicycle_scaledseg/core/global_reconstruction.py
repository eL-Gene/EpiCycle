"""
Global indirect reconstruction stage (Stage 4) for EpiCycle ScaledSegmentation.

Takes the stitched outputs from Stage 3 (assembly) — mask, probabilities,
and binary — downsamples them, and runs USegment3D's indirect method on
the full volume to produce a globally consistent 3D segmentation.

The indirect method derives distance-transform gradients from label images
in three orthogonal views (XY, XZ, YZ) and runs gradient-descent watershed
to produce the final segmentation. Optionally uses the stitched binary mask
as a precomputed foreground constraint.
"""

import gc
import json
import time
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional

from packages.epicycle_scaledseg.core.pipeline import Stage, StageResult

logger = logging.getLogger(__name__)


class GlobalReconstructionStage(Stage):
    """
    Stage 4: global indirect reconstruction on the full stitched volume.

    Parameters
    ----------
    assembly_dir : directory containing Stage 3 outputs
        (full_mask.tif, full_probabilities.npy, full_binary.tif/.npy)
    output_dir : directory for Stage 4 outputs
    downsample_factor : downsampling factor (default 2 = half resolution)
    min_size : remove objects smaller than this (voxels) before reconstruction
    use_precomputed_binary : use stitched binary mask as foreground constraint
    indirect_params : USegment3D indirect aggregation params (uses defaults if None)
    n_cpu : CPU threads for multiprocess watershed
    tile_shape : tile shape for multiprocess watershed (None = auto)
    """

    def __init__(
        self,
        assembly_dir: str = None,
        output_dir: str = None,
        downsample_factor: int = 2,
        min_size: int = 1000,
        use_precomputed_binary: bool = True,
        indirect_params: dict = None,
        n_cpu: int = 64,
        tile_shape: tuple = None,
    ):
        super().__init__('global_reconstruction')
        self.assembly_dir = assembly_dir
        self.output_dir = output_dir
        self.downsample_factor = downsample_factor
        self.min_size = min_size
        self.use_precomputed_binary = use_precomputed_binary
        self.indirect_params = indirect_params
        self.n_cpu = n_cpu
        self.tile_shape = tile_shape

    def get_params(self) -> dict:
        return {
            "downsample_factor": self.downsample_factor,
            "min_size": self.min_size,
            "use_precomputed_binary": self.use_precomputed_binary,
            "indirect_params": self.indirect_params,
            "n_cpu": self.n_cpu,
            "tile_shape": self.tile_shape,
        }

    def validate_inputs(self, tile_dirs: List[Path]) -> bool:
        """Check that Stage 3 assembly outputs exist."""
        if self.assembly_dir is None:
            return False
        adir = Path(self.assembly_dir)
        if not (adir / "full_mask.tif").exists():
            return False
        return True

    def run(
        self,
        tile_dirs: List[Path],
        n_gpus: int,
        log_dir: Optional[Path] = None,
    ) -> StageResult:
        """Run global indirect reconstruction on stitched volume."""
        import tifffile
        from scipy.ndimage import label, binary_fill_holes
        from skimage.morphology import remove_small_objects

        from packages.epicycle_scaledseg.utils.downsampling import (
            maxpool_downsample_3d,
            nearest_downsample_3d,
        )

        adir = Path(self.assembly_dir)
        out_dir = Path(self.output_dir) if self.output_dir else adir.parent / "stage4_global"
        out_dir.mkdir(parents=True, exist_ok=True)

        result = StageResult()

        # Skip if already done
        if (out_dir / "global_segmentation.tif").exists():
            logger.info("Global reconstruction already complete, skipping")
            result.add_success("global")
            return result

        t0 = time.time()
        factor = self.downsample_factor

        # --- Load stitched mask ---
        logger.info(f"Loading stitched mask from {adir / 'full_mask.tif'}")
        mask = tifffile.imread(str(adir / "full_mask.tif"))
        logger.info(f"  mask shape: {mask.shape}, dtype: {mask.dtype}")

        # --- Downsample mask (nearest-neighbor preserves label IDs) ---
        logger.info(f"Downsampling mask by {factor}x (nearest-neighbor)")
        mask_ds = nearest_downsample_3d(mask, factor=factor)
        del mask
        gc.collect()
        logger.info(f"  downsampled mask shape: {mask_ds.shape}")

        # --- Load and downsample binary if using precomputed foreground ---
        binary_ds = None
        if self.use_precomputed_binary:
            bin_path = None
            for ext in (".tif", ".npy"):
                candidate = adir / f"full_binary{ext}"
                if candidate.exists():
                    bin_path = candidate
                    break

            if bin_path is not None:
                logger.info(f"Loading stitched binary from {bin_path}")
                if bin_path.suffix == ".npy":
                    binary_full = np.load(str(bin_path))
                else:
                    binary_full = tifffile.imread(str(bin_path))

                logger.info(f"Downsampling binary by {factor}x (max-pool)")
                binary_ds = maxpool_downsample_3d(
                    binary_full.astype(np.uint8), factor=factor
                ).astype(bool)
                del binary_full
                gc.collect()
                logger.info(f"  downsampled binary shape: {binary_ds.shape}")
            else:
                logger.warning("No stitched binary found, will derive from mask")

        # --- Clean up mask: remove small objects, fill holes, relabel ---
        logger.info(f"Cleaning mask (min_size={self.min_size})")
        filtered = remove_small_objects(
            mask_ds.astype(bool), min_size=self.min_size, connectivity=3
        )
        filtered = binary_fill_holes(filtered, structure=np.ones((3, 3, 3)))
        labeled, n_features = label(filtered.astype(np.uint8), structure=np.ones((3, 3, 3)))
        del mask_ds, filtered
        gc.collect()
        logger.info(f"  {n_features} connected components after cleanup")

        # --- Prepare 3-view labels ---
        labels_xy = labeled
        labels_xz = labeled.transpose(1, 0, 2).copy()
        labels_yz = labeled.transpose(2, 0, 1).copy()

        # --- Get indirect reconstruction params ---
        params = self.indirect_params
        if params is None:
            from packages.epicycle_scaledseg.config.defaults import get_default_indirect_reconstruct_params
            params = get_default_indirect_reconstruct_params(n_cpu=self.n_cpu)
        else:
            # Ensure n_cpu is set even if user provides custom params
            params['indirect_method']['n_cpu'] = self.n_cpu

        # Override tile_shape if provided
        if self.tile_shape is not None:
            params['gradient_descent']['tile_shape'] = self.tile_shape

        # --- Run indirect reconstruction ---
        logger.info("Running USegment3D indirect reconstruction...")
        import segment3D.usegment3d as uSegment3D

        segmentation3D, (probability3D, gradients3D) = (
            uSegment3D.aggregate_2D_to_3D_segmentation_indirect_method(
                segmentations=[labels_xy, labels_xz, labels_yz],
                img_xy_shape=labels_xy.shape,
                precomputed_binary=binary_ds,
                params=params,
                basename=None,
            )
        )

        del labels_xy, labels_xz, labels_yz, binary_ds
        gc.collect()

        elapsed = time.time() - t0
        logger.info(
            f"Global reconstruction complete: "
            f"shape={segmentation3D.shape}, "
            f"n_labels={len(np.unique(segmentation3D)) - 1}, "
            f"elapsed={elapsed:.1f}s"
        )

        # --- Save outputs ---
        logger.info(f"Saving outputs to {out_dir}")
        tifffile.imwrite(
            str(out_dir / "global_segmentation.tif"),
            segmentation3D,
            contiguous=True,
        )

        if probability3D is not None:
            np.save(str(out_dir / "global_probability.npy"), probability3D)

        if gradients3D is not None:
            np.save(str(out_dir / "global_gradients.npy"), gradients3D)

        # Save config
        config = {
            "downsample_factor": factor,
            "min_size": self.min_size,
            "use_precomputed_binary": self.use_precomputed_binary,
            "output_shape": list(segmentation3D.shape),
            "n_labels": int(len(np.unique(segmentation3D)) - 1),
            "elapsed_seconds": elapsed,
            "indirect_params": {
                "n_iter": params['gradient_descent']['n_iter'],
                "momenta": params['gradient_descent']['momenta'],
                "gradient_decay": params['gradient_descent']['gradient_decay'],
                "renorm_gradient": params['gradient_descent']['renorm_gradient'],
                "do_mp": params['gradient_descent']['do_mp'],
                "dtform_method": params['indirect_method']['dtform_method'],
                "smooth_skel_sigma": params['indirect_method']['smooth_skel_sigma'],
                "n_cpu": params['indirect_method']['n_cpu'],
            },
        }
        with open(out_dir / "global_reconstruction_config.json", "w") as f:
            json.dump(config, f, indent=2)

        result.add_success("global")
        return result
