"""
Segmentation stage for EpiCycle ScaledSegmentation.

Runs CellPose 2D segmentation on XY, XZ, YZ orthogonal views,
saving flows, probabilities, masks, and styles per tile.

Supports two GPU strategies:
- 'fixed' (default via 'auto'): 3 GPUs per tile, all views in parallel
- 'round_robin': 1 GPU per tile, views run sequentially
"""

import os
import gc
import json
import time
import traceback
import numpy as np
from pathlib import Path
from typing import List, Optional, Union
from multiprocessing import get_context
from itertools import cycle

from packages.epicycle_scaledseg.core.pipeline import Stage, StageResult
from packages.epicycle_scaledseg.utils.io import base_name, ensure_dir, now_iso, update_tile_config
from packages.epicycle_scaledseg.utils.gpu import GPUAllocator, FixedStrategy, RoundRobinStrategy
from packages.epicycle_scaledseg.utils.resources import get_allocated_cpus
from packages.epicycle_scaledseg.config.defaults import get_default_segment_params, get_default_save_policy


def _resolve_strategy(name: str, n_gpus: int) -> str:
    """Resolve 'auto' strategy to a concrete strategy name."""
    if name == 'auto':
        return 'fixed' if n_gpus >= 3 else 'round_robin'
    return name


def _seg_worker_threads(total_cpus: int, n_workers: int) -> int:
    """CPU threads each seg worker should request, to avoid oversubscription.

    N concurrent worker processes each spinning OpenMP/MKL/OpenBLAS thread pools
    sized to the WHOLE machine would request ``n_workers * total_cpus`` threads and
    thrash the scheduler (the canonical joblib/scikit-learn heuristic). Splitting
    the cores ``total_cpus // n_workers`` (floored at 1) keeps the total thread
    count ~= the core count. Replaces the old hardcoded ``OMP_NUM_THREADS=8`` that
    ignored both core count and worker count.
    """
    return max(1, int(total_cpus) // max(1, int(n_workers)))


def resolve_channels(channels, n_channels: int):
    """Resolve Cellpose ``channels`` for a tile's actual channel count.

    A single-channel (grayscale) tile MUST use ``[0, 0]`` — Cellpose's
    ``[signal, nucleus]`` channel indices (e.g. the default ``[2, 1]``) reference
    a second channel that doesn't exist when ``n_channels == 1`` and would error
    or mis-segment. For ``n_channels >= 2`` the configured ``channels`` are
    returned unchanged. This auto-derivation makes single-channel / 3D-ZYX input
    (which the axis resolver canonicalizes to C=1) work without the user having
    to remember to set ``channels=[0, 0]``.
    """
    if n_channels <= 1:
        return [0, 0]
    return list(channels)


class SegmentationStage(Stage):
    """
    Multi-view 2D segmentation using CellPose.

    Each tile is segmented in XY (lateral model) and XZ/YZ (axial model).
    GPU allocation is handled via a pluggable strategy.

    Parameters
    ----------
    lateral_model : path to CellPose model for XY view
    axial_model : path to CellPose model for XZ/YZ views
    segment_params : CellPose parameters (uses defaults if None)
    save_policy : what to save and compression settings
    gpu_strategy : 'auto' (default), 'fixed', or 'round_robin'
        - 'auto': uses 'fixed' when n_gpus >= 3, falls back to 'round_robin'
        - 'fixed': 3 GPUs per tile, XY/XZ/YZ run in parallel (fastest per-tile)
        - 'round_robin': 1 GPU per tile, views run sequentially (uses all GPUs)
    out_root : output root directory for stage1 results
    """

    def __init__(
        self,
        lateral_model: str,
        axial_model: str,
        segment_params: dict = None,
        save_policy: dict = None,
        gpu_strategy: str = 'auto',
        out_root: str = None,
    ):
        super().__init__('segmentation')
        self.lateral_model = str(Path(lateral_model).resolve())
        self.axial_model = str(Path(axial_model).resolve())
        self.segment_params = segment_params or get_default_segment_params()
        self.save_policy = save_policy or get_default_save_policy()
        self.gpu_strategy_name = gpu_strategy
        self.out_root = str(Path(out_root).resolve()) if out_root else None

    def get_params(self) -> dict:
        return {
            "lateral_model": self.lateral_model,
            "axial_model": self.axial_model,
            "segment_params": self.segment_params,
            "save_policy": self.save_policy,
            "gpu_strategy": self.gpu_strategy_name,
        }

    def validate_inputs(self, tile_dirs: List[Path]) -> bool:
        """Check that model files and tile directories exist."""
        if not Path(self.lateral_model).exists():
            return False
        if not Path(self.axial_model).exists():
            return False
        return all(td.is_dir() for td in tile_dirs)

    def run(
        self,
        tile_dirs: List[Path],
        n_gpus: int,
        log_dir: Optional[Path] = None,
    ) -> StageResult:
        """
        Dispatch segmentation across GPUs.

        Strategy selection:
        - 'fixed' (or 'auto' with >= 3 GPUs): groups GPUs into triplets,
          spawns 3 processes per tile (XY, XZ, YZ in parallel on separate GPUs).
          Processes tiles in batches of n_triplets.
        - 'round_robin' (or 'auto' with < 3 GPUs): spawns one process per GPU,
          each processing its assigned tiles sequentially (XY → XZ → YZ).
        """
        # Filter out already-completed tiles
        todo = []
        for td in tile_dirs:
            stage1_dir = self._stage1_dir(td)
            if (stage1_dir / "SEGMENTATION_DONE").exists():
                continue
            todo.append(td)

        if not todo:
            result = StageResult()
            for td in tile_dirs:
                result.add_success(td.name)
            return result

        # Segmentation (Cellpose) requires a GPU. With none allocated, fail loudly
        # with an actionable message instead of silently scheduling zero workers:
        # round_robin with n_gpus=0 builds an empty assignment dict and spawns
        # nothing, so every tile would "fail" with an opaque marker-not-found. The
        # CPU/in-process path is Phase 2.
        if n_gpus < 1:
            import logging
            msg = ("no GPU allocated (n_gpus=0); segmentation requires a GPU. Set "
                   "CUDA_VISIBLE_DEVICES / pass --n-gpus, or use the Phase-2 CPU path.")
            logging.getLogger("epicycle.segmentation").error(msg)
            result = StageResult()
            for td in tile_dirs:
                result.add_failure(td.name, msg)
            return result

        strategy = _resolve_strategy(self.gpu_strategy_name, n_gpus)

        # 'fixed' groups GPUs into triplets; forced on <3 GPUs it builds an empty
        # triplet list and the batch loop crashes (range() step 0). Fall back to
        # round_robin, which uses whatever GPUs exist. 'auto' never selects 'fixed'
        # below 3, so this only triggers on an explicit gpu_strategy='fixed'.
        if strategy == 'fixed' and n_gpus < 3:
            import logging
            logging.getLogger("epicycle.segmentation").warning(
                f"gpu_strategy='fixed' needs >=3 GPUs but n_gpus={n_gpus}; "
                f"falling back to 'round_robin'."
            )
            strategy = 'round_robin'

        if strategy == 'fixed':
            self._run_fixed(todo, n_gpus, log_dir)
        else:
            self._run_round_robin(todo, n_gpus, log_dir)

        # Collect results by checking for SEGMENTATION_DONE markers
        result = StageResult()
        for td in tile_dirs:
            stage1_dir = self._stage1_dir(td)
            if (stage1_dir / "SEGMENTATION_DONE").exists():
                result.add_success(td.name)
            else:
                result.add_failure(td.name, "SEGMENTATION_DONE marker not found")
        return result

    def _run_fixed(self, todo: List[Path], n_gpus: int, log_dir: Optional[Path]):
        """
        Fixed strategy: 3 GPUs per tile, all 3 views in parallel.

        Groups available GPUs into triplets and processes tiles in batches.
        With 8 GPUs → 2 triplets → 2 tiles at a time (GPUs 6,7 idle).
        With 6 GPUs → 2 triplets → 2 tiles at a time (all GPUs used).
        """
        gpu_ids = list(range(n_gpus))
        triplets = [
            tuple(gpu_ids[i:i + 3])
            for i in range(0, len(gpu_ids) - 2, 3)
        ]

        # Up to len(triplets)*3 view-processes run concurrently per batch (3 views
        # x n_triplets tiles). Split the cores among them so the per-process
        # OMP/MKL pools don't collectively oversubscribe the node.
        n_workers = max(1, len(triplets) * 3)
        n_threads = _seg_worker_threads(get_allocated_cpus(), n_workers)
        use_gpu = n_gpus > 0

        ctx = get_context("spawn")
        log_str = str(log_dir) if log_dir else "logs"

        for batch_start in range(0, len(todo), len(triplets)):
            batch = todo[batch_start:batch_start + len(triplets)]
            procs = []

            for tile, triplet in zip(batch, triplets):
                # Spawn 3 processes for this tile: XY, XZ, YZ in parallel
                for view, gpu_id in zip(
                    ('xy', 'xz', 'yz'), triplet
                ):
                    model = (
                        self.lateral_model if view == 'xy'
                        else self.axial_model
                    )
                    p = ctx.Process(
                        target=_seg_view_worker,
                        args=(
                            gpu_id,
                            str(tile),
                            view,
                            model,
                            self.segment_params,
                            self.save_policy,
                            self.out_root,
                            log_str,
                            n_threads,
                            use_gpu,
                        ),
                    )
                    p.start()
                    procs.append((p, tile, view))

            # Wait for entire batch to finish
            for p, tile, view in procs:
                p.join()

            # Mark tiles as complete if all 3 views succeeded
            for tile in batch:
                stage1_dir = self._stage1_dir(tile)
                views_done = [
                    v for v in ('xy', 'xz', 'yz')
                    if (stage1_dir / f"{v}.done").exists()
                ]
                if len(views_done) == 3:
                    (stage1_dir / "SEGMENTATION_DONE").touch()
                    # Record model paths in config
                    tif_files = list(tile.glob("*.tif"))
                    if tif_files:
                        bname = base_name(str(tif_files[0]))
                        update_tile_config(
                            str(stage1_dir / f"{bname}_config.json"),
                            {"models": {
                                "lateral_or_xy": self.lateral_model,
                                "axial_or_xz_yz": self.axial_model,
                            }},
                        )

    def _run_round_robin(self, todo: List[Path], n_gpus: int, log_dir: Optional[Path]):
        """
        Round-robin strategy: 1 GPU per tile, views run sequentially.

        Distributes tiles evenly across all GPUs. Each GPU worker processes
        its assigned tiles one at a time (XY → XZ → YZ on same GPU).
        """
        gpu_ids = list(range(n_gpus))
        assignments = {gid: [] for gid in gpu_ids}
        for tile, gid in zip(todo, cycle(gpu_ids)):
            assignments[gid].append(str(tile))

        # One worker process per busy GPU runs concurrently; split cores among them.
        n_workers = max(1, sum(1 for tiles in assignments.values() if tiles))
        n_threads = _seg_worker_threads(get_allocated_cpus(), n_workers)
        use_gpu = n_gpus > 0

        ctx = get_context("spawn")
        procs = []
        for gpu_id, tiles in assignments.items():
            if not tiles:
                continue
            p = ctx.Process(
                target=_seg_gpu_worker,
                args=(
                    gpu_id,
                    tiles,
                    self.lateral_model,
                    self.axial_model,
                    self.segment_params,
                    self.save_policy,
                    self.out_root,
                    str(log_dir) if log_dir else "logs",
                    n_threads,
                    use_gpu,
                ),
            )
            p.start()
            procs.append(p)

        for p in procs:
            p.join()

    def _stage1_dir(self, tile_dir: Path) -> Path:
        """Resolve the stage1 output directory for a tile."""
        if self.out_root:
            return Path(self.out_root) / tile_dir.name
        return tile_dir.parent / "stage1_segmentation" / tile_dir.name


# ---------------------------------------------------------------------------
# Workers (run in spawned subprocesses)
# ---------------------------------------------------------------------------

def _seg_view_worker(
    gpu_id: int,
    tile_path: str,
    view: str,
    model_path: str,
    segment_params: dict,
    save_policy: dict,
    out_root: Optional[str],
    log_dir: str,
    n_threads: int = 8,
    use_gpu: bool = True,
):
    """
    Per-view segmentation worker for the fixed (3-GPU) strategy.

    Processes a single view (xy, xz, or yz) of a single tile on one GPU.
    Three of these run in parallel per tile.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
               "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_v] = str(n_threads)

    import torch
    from cellpose import io as cpio, models
    from packages.epicycle_scaledseg.utils.logging_utils import setup_logger

    logger = setup_logger(gpu_id, log_dir)
    tile_dir = Path(tile_path)
    tile_name = tile_dir.name

    if out_root:
        out_dir = Path(out_root) / tile_name
    else:
        out_dir = tile_dir.parent / "stage1_segmentation" / tile_name
    ensure_dir(out_dir)

    try:
        # Find the .tif file
        tif_files = list(tile_dir.glob("*.tif"))
        if not tif_files:
            raise FileNotFoundError(f"No .tif files in {tile_dir}")
        patch_path = str(tif_files[0])
        bname = base_name(patch_path)

        # Read tile config for z_range
        cfg_files = list(tile_dir.glob("*_config.json"))
        z_range = None
        if cfg_files:
            with open(cfg_files[0]) as f:
                meta = json.load(f)
            if "truncated_range" in meta:
                z_range = tuple(meta["truncated_range"])

        # Load image
        img = cpio.imread(patch_path)
        origZ = img.shape[0]
        if z_range is None:
            z_range = (0, origZ)

        z0, z1 = z_range
        img = img[z0:z1]
        logger.info(f"[{bname}][{view}] GPU {gpu_id}, shape={img.shape}, z_range={z_range}")

        # Load model and segment
        model = models.CellposeModel(gpu=use_gpu, pretrained_model=str(model_path))

        _run_view_segmentation(
            img=img,
            view=view,
            model=model,
            bname=bname,
            out_dir=out_dir,
            segment_params=segment_params,
            save_policy=save_policy,
            origZ=origZ,
            z_range=z_range,
            logger=logger,
        )

        del model
        torch.cuda.empty_cache()
        gc.collect()

        logger.info(f"[{bname}][{view}] SUCCESS on GPU {gpu_id}")
    except Exception as e:
        logger.error(f"[{tile_name}][{view}] FAILED on GPU {gpu_id}: {e}\n{traceback.format_exc()}")


def _seg_gpu_worker(
    gpu_id: int,
    tile_paths: List[str],
    lateral_model: str,
    axial_model: str,
    segment_params: dict,
    save_policy: dict,
    out_root: Optional[str],
    log_dir: str,
    n_threads: int = 8,
    use_gpu: bool = True,
):
    """
    Per-GPU segmentation worker for the round-robin strategy.

    Processes assigned tiles sequentially, running XY → XZ → YZ on the same GPU.
    """
    # CRITICAL: set GPU before importing torch
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
               "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_v] = str(n_threads)

    from packages.epicycle_scaledseg.utils.logging_utils import setup_logger

    logger = setup_logger(gpu_id, log_dir)
    logger.info(f"=== SEG GPU {gpu_id} WORKER STARTED ({len(tile_paths)} tiles) ===")

    for i, tile_path in enumerate(tile_paths, 1):
        tile_name = Path(tile_path).name
        logger.info(f"Tile {i}/{len(tile_paths)}: {tile_name}")
        t0 = time.time()

        try:
            _segment_single_tile(
                tile_path=tile_path,
                lateral_model=lateral_model,
                axial_model=axial_model,
                segment_params=segment_params,
                save_policy=save_policy,
                out_root=out_root,
                logger=logger,
                use_gpu=use_gpu,
            )
            logger.info(f"[{tile_name}] SUCCESS ({time.time() - t0:.1f}s)")
        except Exception as e:
            logger.error(f"[{tile_name}] FAILED: {e}\n{traceback.format_exc()}")

    logger.info(f"=== SEG GPU {gpu_id} WORKER FINISHED ===")


def _segment_single_tile(
    tile_path: str,
    lateral_model: str,
    axial_model: str,
    segment_params: dict,
    save_policy: dict,
    out_root: Optional[str],
    logger,
    use_gpu: bool = True,
):
    """
    Segment one tile: XY (lateral) then XZ, YZ (axial), sequentially on one GPU.

    Used by the round-robin strategy.
    """
    import torch
    from cellpose import io as cpio, models

    tile_dir = Path(tile_path)
    bname = base_name(str(tile_dir / (tile_dir.name + ".tif")))

    if out_root:
        out_dir = Path(out_root) / tile_dir.name
    else:
        out_dir = tile_dir.parent / "stage1_segmentation" / tile_dir.name
    ensure_dir(out_dir)

    # Find the .tif file in the tile directory
    tif_files = list(tile_dir.glob("*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"No .tif files in {tile_dir}")
    patch_path = str(tif_files[0])
    bname = base_name(patch_path)

    # Read tile config for z_range if available
    cfg_files = list(tile_dir.glob("*_config.json"))
    z_range = None
    if cfg_files:
        with open(cfg_files[0]) as f:
            meta = json.load(f)
        if "truncated_range" in meta:
            z_range = tuple(meta["truncated_range"])

    # Load image
    img = cpio.imread(patch_path)  # (Z, C, Y, X)
    origZ = img.shape[0]

    if z_range is None:
        # Default: use full Z range
        z_range = (0, origZ)

    z0, z1 = z_range
    img = img[z0:z1]
    logger.info(f"[{bname}] loaded {patch_path}, shape={img.shape}, z_range={z_range}")

    # Check if all 3 views are complete. If any is missing, redo the
    # entire tile to avoid mixing outputs from different runs/parameters.
    views_done = [v for v in ('xy', 'xz', 'yz') if (out_dir / f"{v}.done").exists()]
    if 0 < len(views_done) < 3:
        logger.info(
            f"[{bname}] partial results ({views_done}), clearing and re-segmenting all views"
        )
        for v in views_done:
            (out_dir / f"{v}.done").unlink(missing_ok=True)
            # Remove associated output files
            for suffix in ("_masks.npy", "_masks.tif", "_flows.npy", "_flows.npz",
                           "_probs.npy", "_probs.npz", "_styles.npy"):
                p = out_dir / f"{bname}_{v}{suffix}"
                if p.exists():
                    p.unlink()
        views_done = []

    # Process each view
    for view, model_path in [
        ('xy', lateral_model),
        ('xz', axial_model),
        ('yz', axial_model),
    ]:

        logger.info(f"[{bname}][{view}] segmenting with {Path(model_path).name}")
        model = models.CellposeModel(gpu=use_gpu, pretrained_model=str(model_path))

        _run_view_segmentation(
            img=img,
            view=view,
            model=model,
            bname=bname,
            out_dir=out_dir,
            segment_params=segment_params,
            save_policy=save_policy,
            origZ=origZ,
            z_range=z_range,
            logger=logger,
        )

        del model
        torch.cuda.empty_cache()
        gc.collect()

    # Mark tile as complete
    (out_dir / "SEGMENTATION_DONE").touch()
    update_tile_config(
        str(out_dir / f"{bname}_config.json"),
        {"models": {"lateral_or_xy": lateral_model, "axial_or_xz_yz": axial_model}},
    )


def _run_view_segmentation(
    img: np.ndarray,
    view: str,
    model,
    bname: str,
    out_dir: Path,
    segment_params: dict,
    save_policy: dict,
    origZ: int,
    z_range: tuple,
    logger,
):
    """
    Run CellPose on one orthogonal view and save outputs.

    Transposes the image to the correct orientation, runs slice-wise
    inference, stacks results, and saves according to save_policy.
    """
    from tqdm import tqdm

    # Transpose to get iteration axis
    # img shape: (Zt, C, Y, X)
    im_xy = img.transpose(1, 0, 2, 3)  # (C, Zt, Y, X)

    if view == 'xy':
        stack = im_xy
    elif view == 'xz':
        stack = im_xy.transpose(0, 2, 1, 3)  # (C, Y, Zt, X)
    else:  # yz
        stack = im_xy.transpose(0, 3, 1, 2)  # (C, X, Zt, Y)

    it_len = stack.shape[1]
    masksL, flowsL, probsL, stylesL = [], [], [], []

    # 2D per-slice masks are OPTIONAL + OFF by default: the 3D reconstruction works
    # from the flows + probs, not Cellpose's 2D masks, so assembling them is wasted
    # GPU time unless explicitly wanted. Compute only if `compute_masks` is requested
    # OR `save_raw_masks` is on (can't save what we didn't compute).
    need_masks = bool(segment_params.get("compute_masks", False)) or bool(
        save_policy.get("save_raw_masks", False)
    )

    # Resolve channels for the tile's actual channel count: a single-channel
    # (grayscale) tile MUST use [0, 0]; the configured [signal, nucleus] indices
    # are invalid when there's only one channel. (Auto-derived so single-channel
    # / 3D-ZYX input Just Works even if the user left channels at the 2-ch default.)
    eff_channels = resolve_channels(segment_params['channels'], stack.shape[0])
    if eff_channels != segment_params['channels']:
        logger.info(
            f"[{bname}][{view.upper()}] single-channel tile (C={stack.shape[0]}): "
            f"using channels={eff_channels} instead of {segment_params['channels']}"
        )

    for i in tqdm(range(it_len), desc=f"[{bname}][{view.upper()}]"):
        slice_img = stack[:, i, :, :]
        masks, flow_out, style = model.eval(
            slice_img,
            channels=eff_channels,
            do_3D=False,
            normalize=segment_params['normalize'],
            rescale=segment_params['rescale'],
            compute_masks=need_masks,
            diameter=None,
            cellprob_threshold=segment_params['cellprob_threshold'],
            flow_threshold=segment_params['flow_threshold'],
            batch_size=segment_params['batch_size'],
        )
        _, flow_comp, cellprob = flow_out
        if need_masks:
            masksL.append(masks)
        flowsL.append(flow_comp)
        probsL.append(cellprob)
        stylesL.append(style)

    # Stack and transpose back to canonical (2, Zt, Y, X) / (Zt, Y, X).
    # masksA is identical across views (no transpose); only built when needed.
    masksA = np.stack(masksL, axis=0) if need_masks else None
    if view == 'xy':
        flowsA = np.stack(flowsL, axis=0).transpose(1, 0, 2, 3)
        probsA = np.stack(probsL, axis=0)
    elif view == 'xz':
        flowsA = np.stack(flowsL, axis=0)
        flowsA = flowsA.transpose(1, 0, 2, 3).transpose(0, 2, 1, 3)
        probsA = np.stack(probsL, axis=0).transpose(1, 0, 2)
    else:  # yz
        flowsA = np.stack(flowsL, axis=0).transpose(1, 2, 3, 0)
        probsA = np.stack(probsL, axis=0).transpose(1, 2, 0)

    stylesA = np.stack(stylesL, axis=0)
    prefix = str(out_dir / bname)

    # Save masks
    if save_policy.get("save_raw_masks", False):
        np.save(f"{prefix}_{view}_masks.npy", masksA)
        import tifffile
        if view == 'xy':
            masks_tif = masksA
        elif view == 'xz':
            masks_tif = masksA.transpose(1, 0, 2)
        else:
            masks_tif = masksA.transpose(1, 2, 0)
        tifffile.imwrite(
            f"{prefix}_{view}_masks.tif",
            masks_tif.astype(np.uint32),
            imagej=False,
            bigtiff=True,
        )

    # Save flows
    if save_policy.get("save_raw_flows", True):
        flows_to_save = flowsA.astype(np.float32, copy=False)
        if save_policy.get("flows_compressed", False):
            np.savez_compressed(f"{prefix}_{view}_flows.npz", arr=flows_to_save)
        else:
            np.save(f"{prefix}_{view}_flows.npy", flows_to_save)

    # Save probs
    if save_policy.get("save_raw_probs", True):
        probs_to_save = probsA.astype(np.float32, copy=False)
        if save_policy.get("probs_compressed", False):
            np.savez_compressed(f"{prefix}_{view}_probs.npz", arr=probs_to_save)
        else:
            np.save(f"{prefix}_{view}_probs.npy", probs_to_save)

    # Save styles
    if save_policy.get("save_styles", True):
        np.save(f"{prefix}_{view}_styles.npy", stylesA)

    # Mark view done
    (out_dir / f"{view}.done").touch()
    update_tile_config(f"{prefix}_config.json", {
        "tile_name": bname,
        "original_dimensions": {"z": int(origZ)},
        "truncated_range": list(z_range),
        "segmentation_params": segment_params,
        "save_policy": {"stage1": save_policy},
        "segmentation_runtime": {view: {"timestamp": now_iso()}},
    })

    logger.info(f"[{bname}][{view}] saved flows/probs/styles to {out_dir}")
