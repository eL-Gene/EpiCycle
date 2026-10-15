"""
Full-volume assembly for EpiCycle ScaledSegmentation.

Stitches per-tile reconstruction outputs (masks, probabilities, binary)
back into the full-resolution volume, handling overlapping regions.

Ported from reconst_full_v2.py.
"""

import os
import re
import logging
import time
import numpy as np
import tifffile
from pathlib import Path
from typing import List, Optional, Tuple
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from packages.epicycle_scaledseg.core.pipeline import Stage, StageResult


# ---------------------------------------------------------------------------
# Tile metadata parsing
# ---------------------------------------------------------------------------

def parse_tile_positions(
    tile_dirs: List[Path],
    tile_size: int = 512,
    overlap: float = 0.25,
) -> List[Tuple[int, int, Path]]:
    """
    Extract (y_pixel, x_pixel, tile_dir) from tile directory names.

    Assumes tile names follow the pattern ``tile_<i>_<j>_*`` where
    i and j are grid indices. Pixel offsets are computed as::

        y = i * step_size
        x = j * step_size
        step_size = tile_size * (1 - overlap)

    Parameters
    ----------
    tile_dirs : list of tile directory Paths
    tile_size : tile dimension in pixels (assumes square)
    overlap : fractional overlap between adjacent tiles

    Returns
    -------
    List of (y_pixel, x_pixel, tile_dir) sorted by (y, x).
    """
    step_size = int(tile_size * (1 - overlap))
    positions = []
    for td in tile_dirs:
        match = re.search(r'tile_(\d+)_(\d+)', td.name)
        if match:
            i, j = int(match.group(1)), int(match.group(2))
            positions.append((i * step_size, j * step_size, td))
    return sorted(positions, key=lambda x: (x[0], x[1]))


# ---------------------------------------------------------------------------
# Slice-level loader (for multiprocessing)
# ---------------------------------------------------------------------------

def _load_mask_slice(args):
    """Load a single Z-slice from a tile mask TIFF."""
    y0, x0, filepath, z = args
    try:
        with tifffile.TiffFile(filepath) as tif:
            tile_slice = tif.asarray(key=z)
        return (y0, x0, tile_slice)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Volume stitching functions
# ---------------------------------------------------------------------------

def stitch_mask_volume(
    tile_positions: List[Tuple[int, int, Path]],
    recon_root: Path,
    output_path: str,
    full_size: Tuple[int, int] = (7680, 7680),
    tile_size: int = 512,
    depth: int = None,
    dtype=np.uint32,
    n_workers: int = 16,
    logger=None,
):
    """
    Stitch per-tile mask TIFFs into a single full-resolution BigTIFF.

    Works slice-by-slice to keep memory usage bounded. Overlapping
    regions are averaged and rounded to integer labels.

    Parameters
    ----------
    tile_positions : list of (y_pixel, x_pixel, tile_dir)
    recon_root : root of stage2_reconstruction outputs
    output_path : path for the output BigTIFF
    full_size : (height, width) of the full volume in pixels
    tile_size : tile dimension in pixels
    depth : number of Z slices
    dtype : output data type
    n_workers : number of parallel workers for tile loading
    logger : optional logger
    """
    log = logger or logging.getLogger("epicycle.assembly")
    if depth is None:
        raise ValueError("depth must be specified for mask stitching")
    height, width = full_size

    # Collect mask file paths
    mask_files = []
    for y0, x0, td in tile_positions:
        mask_path = recon_root / td.name / f"{td.name}_mask.tif"
        if mask_path.exists():
            mask_files.append((y0, x0, str(mask_path)))
        else:
            log.warning(f"Missing mask: {mask_path}")

    if not mask_files:
        log.error("No mask files found for stitching")
        return

    log.info(f"Stitching {len(mask_files)} mask tiles -> {output_path}")
    n_procs = min(n_workers, cpu_count() or 1)

    with tifffile.TiffWriter(output_path, bigtiff=True) as tif:
        with Pool(processes=n_procs) as pool:
            for z in tqdm(range(depth), desc="Stitching masks", unit="slice"):
                args = [(y0, x0, path, z) for y0, x0, path in mask_files]
                results = pool.map(_load_mask_slice, args)

                slice_acc = np.zeros((height, width), dtype=np.float32)
                norm_acc = np.zeros_like(slice_acc)

                for result in results:
                    if result is None:
                        continue
                    y0, x0, tile_slice = result
                    y1 = min(y0 + tile_size, height)
                    x1 = min(x0 + tile_size, width)
                    th = y1 - y0
                    tw = x1 - x0
                    slice_acc[y0:y1, x0:x1] += tile_slice[:th, :tw]
                    norm_acc[y0:y1, x0:x1] += 1

                normalized = np.round(
                    slice_acc / np.maximum(norm_acc, 1)
                ).astype(dtype)
                tif.write(normalized, contiguous=True)

    log.info(f"Mask stitching complete -> {output_path}")


def stitch_prob_volume(
    tile_positions: List[Tuple[int, int, Path]],
    recon_root: Path,
    output_path: str,
    full_size: Tuple[int, int] = (7680, 7680),
    tile_size: int = 512,
    depth: int = None,
    stitch_mode: str = 'average',
    logger=None,
):
    """
    Stitch per-tile probability volumes into a single full volume.

    Parameters
    ----------
    stitch_mode : 'average' or 'max'
        How to blend overlapping regions.
    """
    log = logger or logging.getLogger("epicycle.assembly")
    if depth is None:
        raise ValueError("depth must be specified for probability stitching")
    height, width = full_size

    prob_files = []
    for y0, x0, td in tile_positions:
        prob_path = recon_root / td.name / f"{td.name}_probabilities.npy"
        if prob_path.exists():
            prob_files.append((y0, x0, prob_path))
        else:
            log.warning(f"Missing probabilities: {prob_path}")

    if not prob_files:
        log.error("No probability files found for stitching")
        return

    log.info(f"Stitching {len(prob_files)} prob tiles ({stitch_mode}) -> {output_path}")

    prob_vol = np.zeros((depth, height, width), dtype=np.float32)
    norm_acc = np.zeros_like(prob_vol) if stitch_mode == 'average' else None

    for y0, x0, path in tqdm(prob_files, desc="Stitching probabilities", unit="tile"):
        try:
            tile = np.load(str(path))  # (Z, Y, X)
            tz, ty, tx = tile.shape
            y1 = min(y0 + ty, height)
            x1 = min(x0 + tx, width)
            th = y1 - y0
            tw = x1 - x0

            if stitch_mode == 'average':
                prob_vol[:tz, y0:y1, x0:x1] += tile[:tz, :th, :tw]
                norm_acc[:tz, y0:y1, x0:x1] += 1
            elif stitch_mode == 'max':
                prob_vol[:tz, y0:y1, x0:x1] = np.maximum(
                    prob_vol[:tz, y0:y1, x0:x1], tile[:tz, :th, :tw]
                )
        except Exception as e:
            log.error(f"Failed to load {path}: {e}")

    if stitch_mode == 'average' and norm_acc is not None:
        prob_vol = np.divide(prob_vol, np.maximum(norm_acc, 1e-6))

    np.save(output_path, prob_vol.astype(np.float32))
    log.info(f"Prob stitching complete -> {output_path}")


def stitch_prob_volume_zarr(
    tile_positions: List[Tuple[int, int, Path]],
    recon_root: Path,
    output_path: str,
    full_size: Tuple[int, int] = (7680, 7680),
    tile_size: int = 512,
    depth: int = None,
    stitch_mode: str = 'average',
    z_chunk: int = 64,
    logger=None,
):
    """RAM-bounded probability stitch — the chunked-zarr counterpart of
    :func:`stitch_prob_volume`.

    ``stitch_prob_volume`` allocates the WHOLE ``(depth, H, W)`` float32 volume
    (and, in average mode, a second one for the normalizer) in RAM — ~226 GB
    EACH at 7680²×1000, a guaranteed OOM at scale. This variant uses a chunked,
    on-disk zarr working store (``epicycle.utils.io.open_working_volume``): it
    accumulates per tile-region via zarr read-modify-write and normalizes one
    Z-slab at a time, so RAM is bounded to a few chunks — never the full volume.
    Output is a zarr store at ``output_path`` (read back with
    ``epicycle.utils.io.load_volume``). The numpy ``stitch_prob_volume`` stays
    the byte-identical default; this is the opt-in scale path.
    """
    from packages.epicycle_scaledseg.utils.io import open_working_volume

    log = logger or logging.getLogger("epicycle.assembly")
    if depth is None:
        raise ValueError("depth must be specified for probability stitching")
    height, width = full_size

    prob_files = []
    for y0, x0, td in tile_positions:
        prob_path = recon_root / td.name / f"{td.name}_probabilities.npy"
        if prob_path.exists():
            prob_files.append((y0, x0, prob_path))
        else:
            log.warning(f"Missing probabilities: {prob_path}")
    if not prob_files:
        log.error("No probability files found for stitching")
        return

    # Tile-aligned chunks (Z-slab × tile plane), so a region write touches a few
    # chunks, never the whole array.
    chunks = (min(z_chunk, depth), min(tile_size, height), min(tile_size, width))

    # The accumulation does read-modify-write per chunk (rewriting a chunk once
    # per overlapping tile). A ZipStore's entries are WRITE-ONCE, so the working
    # store MUST be a DirectoryStore. For a .zip output we accumulate in a
    # DirectoryStore work dir, then stream-copy to the ZipStore at the end
    # (write-once, RAM-bounded).
    out_is_zip = str(output_path).lower().endswith(".zip")
    work_path = (str(output_path) + ".work.zarr") if out_is_zip else str(output_path)
    prob_vol = open_working_volume(
        (depth, height, width), np.float32, backend="zarr",
        store_path=work_path, chunks=chunks,
    )
    norm_acc = None
    norm_path = work_path + ".norm.zarr"
    if stitch_mode == 'average':
        norm_acc = open_working_volume(
            (depth, height, width), np.float32, backend="zarr",
            store_path=norm_path, chunks=chunks,
        )

    log.info(f"Stitching {len(prob_files)} prob tiles ({stitch_mode}, zarr) -> {output_path}")
    for y0, x0, path in tqdm(prob_files, desc="Stitching probabilities (zarr)", unit="tile"):
        try:
            tile = np.load(str(path))  # (Z, Y, X)
            tz, ty, tx = tile.shape
            y1 = min(y0 + ty, height)
            x1 = min(x0 + tx, width)
            th = y1 - y0
            tw = x1 - x0
            sub = tile[:tz, :th, :tw]
            if stitch_mode == 'average':
                prob_vol[:tz, y0:y1, x0:x1] = prob_vol[:tz, y0:y1, x0:x1] + sub
                norm_acc[:tz, y0:y1, x0:x1] = norm_acc[:tz, y0:y1, x0:x1] + 1
            elif stitch_mode == 'max':
                prob_vol[:tz, y0:y1, x0:x1] = np.maximum(
                    prob_vol[:tz, y0:y1, x0:x1], sub
                )
        except Exception as e:
            log.error(f"Failed to load {path}: {e}")

    import shutil

    if stitch_mode == 'average':
        # Chunk-wise normalize — one Z-slab at a time, never the whole volume.
        for z0 in range(0, depth, z_chunk):
            z1 = min(z0 + z_chunk, depth)
            slab = prob_vol[z0:z1]
            nslab = norm_acc[z0:z1]
            prob_vol[z0:z1] = slab / np.maximum(nslab, 1e-6)
        # Drop the normalizer sidecar (always a DirectoryStore work dir).
        shutil.rmtree(norm_path, ignore_errors=True)

    if out_is_zip:
        # Stream-copy the finished DirectoryStore -> a ZipStore (single GPFS-safe
        # file). Write-once + RAM-bounded: one Z-slab at a time, in order.
        import zarr
        from packages.epicycle_scaledseg.utils.io import _zarr_store
        zstore = _zarr_store(output_path, mode="w")
        try:
            zout = zarr.creation.open_array(
                store=zstore, mode="w", shape=(depth, height, width),
                dtype=np.float32, chunks=chunks,
            )
            for z0 in range(0, depth, z_chunk):
                z1 = min(z0 + z_chunk, depth)
                zout[z0:z1] = prob_vol[z0:z1]
        finally:
            if hasattr(zstore, "close"):
                zstore.close()
        shutil.rmtree(work_path, ignore_errors=True)

    log.info(f"Prob stitching (zarr) complete -> {output_path}")


def stitch_binary_volume(
    tile_positions: List[Tuple[int, int, Path]],
    recon_root: Path,
    output_path: str,
    full_size: Tuple[int, int] = (7680, 7680),
    tile_size: int = 512,
    depth: int = None,
    save_format: str = 'npy',
    logger=None,
):
    """
    Stitch per-tile binary masks using logical OR for overlaps.

    Parameters
    ----------
    save_format : 'npy' (memmap-backed) or 'tif' (BigTIFF)
    """
    log = logger or logging.getLogger("epicycle.assembly")
    if depth is None:
        raise ValueError("depth must be specified for binary stitching")
    height, width = full_size

    bin_files = []
    for y0, x0, td in tile_positions:
        bin_path = recon_root / td.name / f"{td.name}_binary.npy"
        if bin_path.exists():
            bin_files.append((y0, x0, bin_path))
        else:
            log.warning(f"Missing binary: {bin_path}")

    if not bin_files:
        log.error("No binary files found for stitching")
        return

    log.info(f"Stitching {len(bin_files)} binary tiles ({save_format}) -> {output_path}")

    # Memory-map tile arrays for efficient slice access
    tiles = []
    for y0, x0, path in bin_files:
        arr = np.load(str(path), mmap_mode='r')
        tiles.append((y0, x0, arr))

    if save_format == 'npy':
        out = np.lib.format.open_memmap(
            output_path, mode='w+', dtype=np.uint8,
            shape=(depth, height, width),
        )
        for z in tqdm(range(depth), desc="Stitching binary (npy)", unit="slice"):
            slice_acc = np.zeros((height, width), dtype=bool)
            for y0, x0, arr in tiles:
                ty, tx = arr.shape[1], arr.shape[2]
                y1 = min(y0 + ty, height)
                x1 = min(x0 + tx, width)
                th = y1 - y0
                tw = x1 - x0
                slice_acc[y0:y1, x0:x1] |= (arr[z, :th, :tw] > 0)
            out[z] = slice_acc.astype(np.uint8)

    elif save_format == 'tif':
        with tifffile.TiffWriter(output_path, bigtiff=True) as tif:
            for z in tqdm(range(depth), desc="Stitching binary (tif)", unit="slice"):
                slice_acc = np.zeros((height, width), dtype=bool)
                for y0, x0, arr in tiles:
                    ty, tx = arr.shape[1], arr.shape[2]
                    y1 = min(y0 + ty, height)
                    x1 = min(x0 + tx, width)
                    th = y1 - y0
                    tw = x1 - x0
                    slice_acc[y0:y1, x0:x1] |= (arr[z, :th, :tw] > 0)
                tif.write(slice_acc.astype(np.uint8), contiguous=True)

    log.info(f"Binary stitching complete -> {output_path}")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def downsample_binary_maxpool(binary: np.ndarray, factor: int = 2) -> np.ndarray:
    """Downsample a binary volume by blockwise max (logical OR)."""
    if factor <= 1:
        return binary
    z, y, x = binary.shape
    z2, y2, x2 = z // factor, y // factor, x // factor
    if z2 == 0 or y2 == 0 or x2 == 0:
        raise ValueError("Downsample factor too large for volume")
    cropped = binary[:z2 * factor, :y2 * factor, :x2 * factor]
    reshaped = cropped.reshape(z2, factor, y2, factor, x2, factor)
    return reshaped.max(axis=(1, 3, 5))


# ---------------------------------------------------------------------------
# Pipeline Stage
# ---------------------------------------------------------------------------

class AssemblyStage(Stage):
    """
    Full-volume assembly stage: stitches per-tile outputs into global volumes.

    Produces up to 3 outputs:
    - Full mask volume (BigTIFF, uint32)
    - Full probability volume — float32; ``full_probabilities.npy``
      (prob_backend='numpy', default) or a chunked ``full_probabilities.zarr``
      (prob_backend='zarr', RAM-bounded for whole-tissue scale). This is a
      TERMINAL sidecar — Stage 4 / nothing downstream reads it, so the format
      is a free choice.
    - Full binary volume (.npy or BigTIFF, uint8)

    Parameters
    ----------
    recon_root : stage2_reconstruction root directory
    output_dir : directory for assembled volumes
    full_size : (height, width) of the original image in pixels
    tile_size : tile dimension in pixels (assumes square)
    overlap : fractional overlap between tiles
    depth : number of Z slices in the full stack
    stitch_mask : whether to stitch mask volume
    stitch_prob : whether to stitch probability volume
    stitch_binary : whether to stitch binary volume
    prob_stitch_mode : 'average' or 'max' for probability overlap blending
    binary_format : 'npy' or 'tif' for binary output
    n_workers : parallel workers for mask slice loading
    """

    def __init__(
        self,
        recon_root: str = None,
        output_dir: str = None,
        full_size: Tuple[int, int] = None,
        tile_size: int = None,
        overlap: float = None,
        depth: int = None,
        stitch_mask: bool = True,
        stitch_prob: bool = True,
        stitch_binary: bool = True,
        prob_stitch_mode: str = 'max',
        prob_backend: str = 'numpy',
        binary_format: str = 'npy',
        n_workers: int = 16,
        tiling_config_path: str = None,
    ):
        super().__init__('assembly')
        self.recon_root = str(Path(recon_root).resolve()) if recon_root else None
        self.output_dir = str(Path(output_dir).resolve()) if output_dir else None
        self.full_size = full_size
        self.tile_size = tile_size
        self.overlap = overlap
        self.depth = depth
        self.tiling_config_path = tiling_config_path
        # Explicit per-tile origins, loaded from the manifest in _load_tiling_config.
        # Initialized here so the attribute always exists (defensive).
        self._tile_origins = None
        self.stitch_mask = stitch_mask
        self.stitch_prob = stitch_prob
        self.stitch_binary = stitch_binary
        self.prob_stitch_mode = prob_stitch_mode
        self.prob_backend = prob_backend  # 'numpy' (default) or 'zarr' (RAM-bounded)
        self.binary_format = binary_format
        self.n_workers = n_workers

    def get_params(self) -> dict:
        return {
            "full_size": self.full_size,
            "tile_size": self.tile_size,
            "overlap": self.overlap,
            "depth": self.depth,
            "stitch_mask": self.stitch_mask,
            "stitch_prob": self.stitch_prob,
            "stitch_binary": self.stitch_binary,
            "prob_stitch_mode": self.prob_stitch_mode,
            "prob_backend": self.prob_backend,
            "binary_format": self.binary_format,
            "n_workers": self.n_workers,
        }

    def validate_inputs(self, tile_dirs: List[Path]) -> bool:
        """Check that reconstruction outputs exist for at least one tile."""
        # Try to load tiling config early so params are available
        self._load_tiling_config(tile_dirs)

        recon_root = self._recon_root(tile_dirs[0] if tile_dirs else Path("."))
        if not recon_root.is_dir():
            return False
        # At least one tile should have a mask
        for td in tile_dirs:
            mask = recon_root / td.name / f"{td.name}_mask.tif"
            if mask.exists():
                return True
        return False

    def _load_tiling_config(self, tile_dirs: List[Path]) -> None:
        """
        Auto-infer assembly parameters from tiling_config.json.

        Searches for tiling_config.json in:
        1. Explicit ``tiling_config_path``
        2. Parent directory of the first tile dir (tiles are usually in a common folder)

        Only fills in parameters that are still None.
        """
        import json as _json

        # Explicit per-tile origins (populated below if the manifest carries
        # tile_geometry). A flush-anchored tile's true origin is dim-patch, NOT
        # grid_index*step, so the regex-derived origin is WRONG for it — prefer
        # these stored origins when present.
        self._tile_origins = None

        config_path = None
        if self.tiling_config_path and Path(self.tiling_config_path).exists():
            config_path = Path(self.tiling_config_path)
        elif tile_dirs:
            candidate = tile_dirs[0].parent / "tiling_config.json"
            if candidate.exists():
                config_path = candidate

        if config_path is None:
            return

        log = logging.getLogger("epicycle.assembly")
        log.info(f"Loading tiling config from {config_path}")
        with open(config_path) as f:
            cfg = _json.load(f)

        if self.full_size is None and "full_size" in cfg:
            self.full_size = tuple(cfg["full_size"])
        if self.tile_size is None and "patch_size" in cfg:
            self.tile_size = cfg["patch_size"]
        if self.overlap is None and "overlap" in cfg:
            self.overlap = cfg["overlap"]
        if self.depth is None and "depth" in cfg:
            self.depth = cfg["depth"]

        geom = cfg.get("tile_geometry")
        if geom:
            self._tile_origins = {
                g["tile_name"]: (int(g["origin"][0]), int(g["origin"][1]))
                for g in geom
                if "tile_name" in g and "origin" in g
            }

    def run(
        self,
        tile_dirs: List[Path],
        n_gpus: int,
        log_dir: Optional[Path] = None,
    ) -> StageResult:
        """Stitch all configured volume types."""
        logger = logging.getLogger("epicycle.assembly")
        result = StageResult()

        # Auto-infer missing params from tiling config
        self._load_tiling_config(tile_dirs)

        # Validate that required params are set
        missing = []
        if self.full_size is None:
            missing.append("full_size")
        if self.tile_size is None:
            missing.append("tile_size")
        if self.overlap is None:
            missing.append("overlap")
        if self.depth is None:
            missing.append("depth")
        if missing:
            msg = (
                f"Assembly parameters not set and no tiling_config.json found: "
                f"{', '.join(missing)}. Provide them explicitly or ensure "
                f"tiling_config.json exists in the tiles directory."
            )
            logger.error(msg)
            for td in tile_dirs:
                result.add_failure(td.name, msg)
            return result

        recon_root = self._recon_root(tile_dirs[0] if tile_dirs else Path("."))
        out_dir = Path(self.output_dir) if self.output_dir else recon_root.parent / "stage3_assembly"
        out_dir.mkdir(parents=True, exist_ok=True)

        if self._tile_origins:
            # Use the explicit stored origins (correct for flush tiles).
            positions = [
                (self._tile_origins[td.name][0], self._tile_origins[td.name][1], td)
                for td in tile_dirs if td.name in self._tile_origins
            ]
            positions.sort(key=lambda p: (p[0], p[1]))
            dropped = [td.name for td in tile_dirs if td.name not in self._tile_origins]
            if dropped:
                logger.warning(
                    f"{len(dropped)} tile(s) absent from tiling_config geometry, "
                    f"SKIPPED (would leave holes in the stitch): {dropped[:5]}"
                    + (" ..." if len(dropped) > 5 else "")
                )
            logger.info(f"Using {len(positions)} stored tile origins from tiling_config")
        else:
            # Legacy fallback: regex-derive origins from tile_<i>_<j> names.
            positions = parse_tile_positions(
                tile_dirs, tile_size=self.tile_size, overlap=self.overlap,
            )

        if not positions:
            logger.error("No tiles matched tile_<i>_<j> naming pattern")
            for td in tile_dirs:
                result.add_failure(td.name, "tile name does not match pattern")
            return result

        try:
            if self.stitch_mask:
                logger.info("Stitching mask volume...")
                stitch_mask_volume(
                    positions, recon_root,
                    str(out_dir / "full_mask.tif"),
                    full_size=self.full_size,
                    tile_size=self.tile_size,
                    depth=self.depth,
                    n_workers=self.n_workers,
                    logger=logger,
                )

            if self.stitch_prob:
                logger.info(f"Stitching probability volume (backend={self.prob_backend})...")
                if self.prob_backend == "zarr":
                    # RAM-bounded chunked zarr stitch — avoids the full (depth,H,W)
                    # float32 allocation (OOM at scale). Output is a zarr store.
                    # NOTE: nothing downstream reads the stitched prob (Stage 4
                    # uses only the mask + binary), so the .npy->.zarr format
                    # change is safe — it's a terminal sidecar artifact.
                    stitch_prob_volume_zarr(
                        positions, recon_root,
                        str(out_dir / "full_probabilities.zarr"),
                        full_size=self.full_size,
                        tile_size=self.tile_size,
                        depth=self.depth,
                        stitch_mode=self.prob_stitch_mode,
                        logger=logger,
                    )
                else:
                    stitch_prob_volume(
                        positions, recon_root,
                        str(out_dir / "full_probabilities.npy"),
                        full_size=self.full_size,
                        tile_size=self.tile_size,
                        depth=self.depth,
                        stitch_mode=self.prob_stitch_mode,
                        logger=logger,
                    )

            if self.stitch_binary:
                ext = "npy" if self.binary_format == "npy" else "tif"
                logger.info("Stitching binary volume...")
                stitch_binary_volume(
                    positions, recon_root,
                    str(out_dir / f"full_binary.{ext}"),
                    full_size=self.full_size,
                    tile_size=self.tile_size,
                    depth=self.depth,
                    save_format=self.binary_format,
                    logger=logger,
                )

            for td in tile_dirs:
                result.add_success(td.name)

        except Exception as e:
            logger.error(f"Assembly failed: {e}")
            for td in tile_dirs:
                result.add_failure(td.name, str(e))

        return result

    def _recon_root(self, tile_dir: Path) -> Path:
        """Resolve reconstruction root directory."""
        if self.recon_root:
            return Path(self.recon_root)
        return tile_dir.parent / "stage2_reconstruction"

    def __repr__(self) -> str:
        return (
            f"AssemblyStage(full_size={self.full_size}, "
            f"tile_size={self.tile_size}, overlap={self.overlap}, "
            f"depth={self.depth})"
        )
