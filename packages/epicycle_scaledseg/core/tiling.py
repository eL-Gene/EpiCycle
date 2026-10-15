"""
Tiling stage for EpiCycle ScaledSegmentation.

Splits a large Z-stack TIFF (Z, C, Y, X) into overlapping XY tiles,
each saved in its own subdirectory with Z-range metadata. This is
Stage 0 — it produces the tile directories consumed by all downstream
stages (segmentation, reconstruction, assembly).
"""

import json
import math
import time
import logging
from pathlib import Path
from typing import List, Optional

from packages.epicycle_scaledseg.core.pipeline import Stage, StageResult

logger = logging.getLogger(__name__)


def cap_patch_size_to_ram(
    requested_patch: int,
    full_z: int,
    avail_ram_gb: float,
    ram_fraction: float = 0.8,
    ram_reserve_gb: float = 16.0,
    min_patch: int = 128,
) -> "tuple":
    """Largest XY tile <= ``requested_patch`` whose RECON peak fits the RAM budget.

    Reconstruction peak RAM ~ ``Z * patch^2`` (tiling is XY-only, so Z is fixed);
    on a small machine the requested tile may not fit, so shrink the XY size until
    ``estimate_tile_peak_ram_gb(Z, patch, patch) <= (avail-reserve)*fraction`` (the
    same budget recon's M-sizing uses, so a tile that passes here runs at >=1
    concurrency). Since peak scales with patch^2, the fitting size is
    ``requested * sqrt(budget/peak)``, rounded down to a multiple of 64.

    Returns ``(capped_patch:int, status:str)``: ``"fits"`` (unchanged),
    ``"shrunk"``, ``"min-floor"`` (even ``min_patch`` exceeds the budget -> the Z
    depth alone is too big; caller should warn), or ``"unknown-ram"`` (RAM couldn't
    be measured -> leave the request untouched and let recon-time gating handle it).
    """
    from packages.epicycle_scaledseg.utils.resources import estimate_tile_peak_ram_gb
    req = int(requested_patch)
    if not avail_ram_gb or avail_ram_gb <= 0:
        return req, "unknown-ram"
    budget = max(0.0, float(avail_ram_gb) - float(ram_reserve_gb)) * float(ram_fraction)
    if budget <= 0:
        return min(req, int(min_patch)), "min-floor"
    peak_req = estimate_tile_peak_ram_gb(int(full_z), req, req)
    if peak_req <= budget:
        return req, "fits"
    capped = int(req * math.sqrt(budget / peak_req))
    capped = (capped // 64) * 64                       # round down to a tidy multiple
    capped = max(int(min_patch), min(req, capped))
    if estimate_tile_peak_ram_gb(int(full_z), capped, capped) > budget:
        return int(min_patch), "min-floor"             # Z alone too big to fit
    return capped, "shrunk"


def _input_z_dim(path: str, input_axes: str) -> Optional[int]:
    """The Z extent of an input stack from its TIFF HEADER (no array load), or
    None if undetermined. Resolves axis order (explicit -> metadata -> heuristic)."""
    try:
        import tifffile
        from packages.epicycle_scaledseg.utils.axes import detect_axes_order
        with tifffile.TiffFile(path) as tf:
            s = tf.series[0]
            shape = tuple(int(x) for x in s.shape)
            meta_axes = (getattr(s, "axes", "") or "").upper()
        ax = (input_axes or "").upper() if (input_axes and input_axes != "auto") else ""
        if ax not in ("ZCYX", "CZYX", "ZYX"):
            ax = meta_axes if meta_axes in ("ZCYX", "CZYX", "ZYX") else detect_axes_order(shape)
        if ax == "ZYX":
            return shape[0]
        if ax == "CZYX":
            return shape[1]
        return shape[0]                                # ZCYX (engine canonical)
    except Exception:
        return None


class TilingStage(Stage):
    """
    Stage 0: tile a full Z-stack TIFF into overlapping XY patches.

    Parameters
    ----------
    input_tif : path to the input Z-stack TIFF (Z, C, Y, X)
    output_dir : directory to save tile subdirectories
    patch_size : target XY patch size in pixels (default 1024)
    overlap : fractional overlap between tiles (default 0.25)
    z_threshold_method : method for detecting per-tile Z-range
        'hard' (default), 'percentile', or 'default'
    threshold_value : threshold for Z-range detection (default 0.07)
    channel_index : which channel for Z-range intensity analysis (default 1)
    auto_correct : auto-find a valid patch size if exact tiling fails
    """

    def __init__(
        self,
        input_tif: str,
        output_dir: str = None,
        patch_size: int = 1024,
        overlap: float = 0.25,
        z_threshold_method: str = 'hard',
        threshold_value: float = 0.07,
        channel_index: int = 1,
        auto_correct: bool = False,
        input_axes: str = "auto",
        auto_ram_cap: bool = True,
    ):
        super().__init__('tiling')
        self.input_tif = str(Path(input_tif).resolve())
        self.output_dir = str(Path(output_dir).resolve()) if output_dir else None
        self.patch_size = patch_size
        self.overlap = overlap
        self.z_threshold_method = z_threshold_method
        self.threshold_value = threshold_value
        self.channel_index = channel_index
        self.auto_correct = auto_correct
        self.input_axes = input_axes
        # Shrink the XY tile so one tile's reconstruction fits this machine's RAM
        # (a no-op on a fat node; prevents OOM on a small box). Logs loudly when it
        # shrinks. Set False to keep patch_size exactly.
        self.auto_ram_cap = auto_ram_cap
        self._tile_dirs: List[Path] = []

    def get_params(self) -> dict:
        return {
            "input_tif": self.input_tif,
            "patch_size": self.patch_size,
            "overlap": self.overlap,
            "z_threshold_method": self.z_threshold_method,
            "threshold_value": self.threshold_value,
            "channel_index": self.channel_index,
            "auto_correct": self.auto_correct,
            "input_axes": self.input_axes,
            "auto_ram_cap": self.auto_ram_cap,
        }

    def validate_inputs(self, tile_dirs: List[Path]) -> bool:
        """Check that the input TIFF exists."""
        return Path(self.input_tif).is_file()

    def _resolve_output_dir(self, log_dir: Optional[Path] = None) -> str:
        out = self.output_dir
        if out is None:
            if log_dir is not None:
                try:
                    out = str(Path(log_dir).resolve().parent / "tiles")
                except Exception:
                    out = str(Path(self.input_tif).parent / "tiles")
            else:
                out = str(Path(self.input_tif).parent / "tiles")
        return out

    def restore_outputs(self, log_dir: Optional[Path] = None) -> List[Path]:
        """Read the tile dirs back from ``tiling_config.json`` (resume skip path)."""
        out = self._resolve_output_dir(log_dir)
        cfg_path = Path(out) / "tiling_config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(str(cfg_path))
        with open(cfg_path) as f:
            cfg = json.load(f)
        self._tile_dirs = [Path(out) / name for name in cfg["tiles"]]
        return list(self._tile_dirs)

    def get_tile_dirs(self) -> List[Path]:
        """Return tile directories created by the last run."""
        return list(self._tile_dirs)

    def run(
        self,
        tile_dirs: List[Path],
        n_gpus: int,
        log_dir: Optional[Path] = None,
    ) -> StageResult:
        """
        Tile the input TIFF and return results with new tile directories.

        The ``tile_dirs`` argument is ignored for this stage (tiling creates
        the tile directories from scratch). Downstream stages receive the
        newly created tile paths via ``StageResult.tile_dirs``.
        """
        # Default output_dir: prefer pipeline work_dir/log_dir if available
        out = self._resolve_output_dir(log_dir)

        # Skip if tiling already done
        tiling_config_path = Path(out) / "tiling_config.json"
        if tiling_config_path.exists():
            logger.info(f"Tiling already complete ({tiling_config_path}), skipping")
            with open(tiling_config_path) as f:
                cfg = json.load(f)
            self._tile_dirs = [Path(out) / name for name in cfg["tiles"]]
            result = StageResult()
            result.tile_dirs = self._tile_dirs
            for td in self._tile_dirs:
                result.add_success(td.name)
            return result

        # Auto RAM cap: shrink the XY tile so ONE tile's reconstruction fits this
        # machine's RAM (no-op on a fat node; prevents an OOM on a small box).
        patch = self.patch_size
        if self.auto_ram_cap:
            from packages.epicycle_scaledseg.utils.resources import available_ram_gb_and_source
            full_z = _input_z_dim(self.input_tif, self.input_axes)
            ram, ram_src = available_ram_gb_and_source()
            if full_z:
                capped, status = cap_patch_size_to_ram(self.patch_size, full_z, ram)
                if status == "shrunk":
                    logger.warning(
                        f"Auto RAM cap: patch {self.patch_size} -> {capped} so one "
                        f"tile's recon fits {ram:.0f}GB ({ram_src}, Z={full_z}). "
                        f"Smaller tiles = more seams; pass auto_ram_cap=False to keep "
                        f"{self.patch_size}."
                    )
                    patch = capped
                elif status == "min-floor":
                    logger.warning(
                        f"Auto RAM cap: even the floor tile (Z={full_z}) exceeds the "
                        f"{ram:.0f}GB recon budget; tiling at {capped} but recon may "
                        f"still need more RAM (Z is the driver — tiling is XY-only)."
                    )
                    patch = capped

        logger.info(
            f"Tiling {self.input_tif} -> {out} "
            f"(patch={patch}, overlap={self.overlap})"
        )

        from packages.epicycle_scaledseg.utils.tiling import ZStackTiler

        tiler = ZStackTiler(
            tif_path=self.input_tif,
            output_dir=out,
            target_patch_size=patch,
            overlap=self.overlap,
            z_threshold_method=self.z_threshold_method,
            threshold_value=self.threshold_value,
            channel_index=self.channel_index,
            auto_correct=self.auto_correct,
            input_axes=self.input_axes,
        )

        self._tile_dirs = tiler.split_tiles()

        result = StageResult()
        result.tile_dirs = self._tile_dirs
        for td in self._tile_dirs:
            result.add_success(td.name)

        logger.info(
            f"Tiling complete: {len(self._tile_dirs)} tiles, "
            f"actual patch={tiler.patch_size}, step={tiler.step_size}"
        )
        return result
