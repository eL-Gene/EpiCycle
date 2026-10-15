"""Post-stitch QC cleanup + isotropic smoothing — STAGE ORCHESTRATOR + WRITER.

Runs AFTER the Mode-B IoU stitch on the single global-label volume
(``stage4_full_stitch/global_labels.npy`` + ``label_provenance.csv``). It decides
which global cells to drop — a **size floor** (tiny fragment "confetti") and
**detached-island removal** (debris physically separated from the tissue mass) —
ENTIRELY from the provenance CSV (no volume read; see :mod:`epicycle.core.cleanup`),
then writes a cleaned global volume, optionally applying the mirrored Step-3 SDF
**smooth** per surviving instance.

Each sub-op is independently toggleable (``remove_small`` / ``remove_islands`` /
``smooth``); ``dry_run`` decides + logs the drop counts and the voxel-count
distribution WITHOUT writing the (multi-GB) volume, so ``min_voxels`` can be
calibrated on the real data first. Outputs are written non-destructively with a
``_clean`` suffix next to the stitch; the stage is idempotent (skips if the clean
TIFF already exists).

Two write paths:

* **smooth OFF** — slice-streaming relabel: read each ``(H,W)`` slice from the
  source memmap, remap through the dense LUT, stream to ``.npy`` + BigTIFF. Cheap,
  RAM-bounded, mirrors the stitch writer.
* **smooth ON** — per-instance parallel smooth: a worker ``Pool`` computes each
  surviving cell's smoothed mask from its (padded) bbox crop (CPU-bound, the part
  worth the cores); the main process **serializes the writes** (cross-process memmap
  writes would race on overlapping bbox pages) via a safe-write scatter into the
  output memmap, then streams the memmap -> BigTIFF. ``tqdm`` shows progress.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from packages.epicycle_scaledseg.core.pipeline import Stage, StageResult
from packages.epicycle_scaledseg.core.cleanup import (
    select_dropped_by_size,
    select_dropped_by_max_size,
    select_dropped_islands,
    build_relabel_lut,
    lut_to_array,
    _ri,
)
from packages.epicycle_scaledseg.core.smooth import resolve_sigmas, smooth_label_sdf
from packages.epicycle_scaledseg.core.iou_stitch_stage import _write_label_provenance
from packages.epicycle_scaledseg.utils.resources import get_allocated_cpus

_MANIFEST_VERSION = 1

# Per-process source memmap handle for the smooth workers (set by the initializer).
_WORKER_SRC: Optional[np.ndarray] = None


# --------------------------------------------------------------------------- #
# Parallel-smooth worker (module-level so it is picklable under spawn)
# --------------------------------------------------------------------------- #

def _init_smooth_worker(npy_path: str) -> None:
    """Pool initializer: open the source label memmap ONCE per worker process."""
    global _WORKER_SRC
    _WORKER_SRC = np.load(npy_path, mmap_mode="r")


def _smooth_one(task):
    """Smooth ONE instance from its padded bbox crop. Returns the footprint to scatter.

    ``task = (old_id, new_id, (z0,z1,y0,y1,x0,x1), sigma_z, sigma_xy, preserve)``.
    Reads the crop from the per-process source memmap, smooths ``old_id`` in
    isolation, and returns ``(new_id, (z0,y0,x0), mask)``. Robustness contract (a
    survivor that passed the filters must NEVER be silently lost by smoothing):

    * cell genuinely ABSENT from the crop -> ``None`` (nothing to write);
    * smoothing ERASES a thin cell (mask vanishes) OR raises -> fall back to the
      cell's ORIGINAL unsmoothed footprint (written unsmoothed, not dropped);
    * the crop READ itself fails -> ``("ERROR", new_id, repr(e))`` so one bad cell
      cannot abort the whole (multi-hour) run; the caller counts + logs these.

    The crop is bounded to the cell's bbox (never the full volume).
    """
    old_id, new_id, bbox, sz, sxy, preserve = task
    z0, z1, y0, y1, x0, x1 = bbox
    try:
        crop = np.asarray(_WORKER_SRC[z0:z1, y0:y1, x0:x1])
    except Exception as e:  # pragma: no cover - rare I/O failure
        return ("ERROR", int(new_id), f"crop read: {e!r}")
    orig = crop == old_id
    if not orig.any():
        return None
    try:
        result, _n, _m = smooth_label_sdf(
            crop, old_id, sigma_z=sz, sigma_xy=sxy, preserve_topology=preserve
        )
        mask = result == old_id
        if not mask.any():
            mask = orig  # smoothing erased a thin cell -> keep it UNSMOOTHED
    except Exception:  # pragma: no cover - keep the cell rather than drop it
        mask = orig
    return (int(new_id), (int(z0), int(y0), int(x0)), mask)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _read_provenance(csv_path: Path) -> List[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _voxel_stats(vcs: np.ndarray) -> dict:
    if vcs.size == 0:
        return {}
    return {
        "n": int(vcs.size),
        "min": int(vcs.min()),
        "p1": int(np.percentile(vcs, 1)),
        "p5": int(np.percentile(vcs, 5)),
        "p25": int(np.percentile(vcs, 25)),
        "median": int(np.median(vcs)),
        "p95": int(np.percentile(vcs, 95)),
        "p99": int(np.percentile(vcs, 99)),
        "p99_9": int(np.percentile(vcs, 99.9)),
        "max": int(vcs.max()),
    }


def _accumulate_slice(stats: Dict[int, dict], slice_2d: np.ndarray, z: int, depth: int) -> None:
    """Accumulate per-label voxel_count + ZYX bbox + border flag for one slice.

    Mirrors the stitch writer's ``_accumulate`` (one stable sort + reduceat per
    slice, RAM-bounded to this slice's foreground)."""
    ys, xs = np.nonzero(slice_2d)
    if ys.size:
        vals = slice_2d[ys, xs]
        order = np.argsort(vals, kind="stable")
        vals_s = vals[order]
        ys_s = ys[order]
        xs_s = xs[order]
        uniq, start, counts = np.unique(vals_s, return_index=True, return_counts=True)
        ymin = np.minimum.reduceat(ys_s, start)
        ymax = np.maximum.reduceat(ys_s, start)
        xmin = np.minimum.reduceat(xs_s, start)
        xmax = np.maximum.reduceat(xs_s, start)
        for i in range(uniq.size):
            v = int(uniq[i])
            st = stats.get(v)
            if st is None:
                stats[v] = {
                    "voxel_count": int(counts[i]),
                    "z_min": z, "z_max": z,
                    "y_min": int(ymin[i]), "y_max": int(ymax[i]),
                    "x_min": int(xmin[i]), "x_max": int(xmax[i]),
                    "touches_border": False,
                }
            else:
                st["voxel_count"] += int(counts[i])
                st["z_min"] = min(st["z_min"], z)
                st["z_max"] = max(st["z_max"], z)
                st["y_min"] = min(st["y_min"], int(ymin[i]))
                st["y_max"] = max(st["y_max"], int(ymax[i]))
                st["x_min"] = min(st["x_min"], int(xmin[i]))
                st["x_max"] = max(st["x_max"], int(xmax[i]))
        present_vals = uniq
    else:
        present_vals = np.empty(0, dtype=slice_2d.dtype)
    edge = np.concatenate([slice_2d[0, :], slice_2d[-1, :], slice_2d[:, 0], slice_2d[:, -1]])
    border = set(np.unique(edge).tolist())
    if z == 0 or z == depth - 1:
        border |= set(present_vals.tolist())
    for v in border:
        if v != 0 and v in stats:
            stats[v]["touches_border"] = True


def _clean_provenance_inputs(rows: List[dict], lut: Dict[int, int]):
    """Build the (provenance, ) seam-field dict keyed by NEW dense id for the clean CSV.

    Carries the seam classification + ``min_match_iou`` from the surviving OLD rows
    (re-keyed to the new dense id); the size/bbox/border come from the post-write
    ``stats`` (which reflect any smoothing)."""
    rows_by_id = {_ri(r, "dense_id"): r for r in rows}
    prov: Dict[int, dict] = {}
    for old_id, new_id in lut.items():
        r = rows_by_id.get(old_id, {})
        miou = r.get("min_match_iou", "")
        prov[new_id] = {
            "seam_status": r.get("seam_status", "clean") or "clean",
            "n_tiles": _ri(r, "n_tiles") or 1,
            "min_match_iou": (float(miou) if miou not in ("", None) else None),
            "tiles": [t for t in (r.get("tiles", "") or "").split(";") if t],
            "constituents": [
                tuple(c.split(":", 1)) for c in (r.get("constituents", "") or "").split(";") if c
            ],
        }
    return prov


# --------------------------------------------------------------------------- #
# Stage
# --------------------------------------------------------------------------- #

class PostprocessStage(Stage):
    """Post-stitch cleanup (size + detached-island filter) + optional SDF smooth.

    Parameters
    ----------
    input_dir : dir holding ``global_labels.npy`` + ``label_provenance.csv`` (defaults
        to ``<tile_dirs[0].parent>/stage4_full_stitch`` — the stitch output).
    output_dir : where to write ``*_clean`` outputs (defaults to ``input_dir``).
    remove_small / min_voxels : drop cells with ``voxel_count < min_voxels``.
    remove_islands / island_* : drop cells in detached debris islands.
    smooth / smooth_sigma / smooth_sigma_z / smooth_sigma_xy / preserve_topology :
        per-instance SDF smooth (isotropic when only ``smooth_sigma`` is set).
    n_workers : parallel smooth workers (defaults to allocated CPUs - 1).
    dry_run : decide + log only; write nothing.
    """

    def __init__(
        self,
        input_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        recon_root: Optional[str] = None,
        remove_small: bool = True,
        min_voxels: int = 200,
        min_percentile: float = 0.0,
        remove_large: bool = False,
        max_voxels: int = 0,
        max_percentile: float = 0.0,
        remove_islands: bool = True,
        island_downsample: int = 8,
        island_min_component_frac: float = 0.01,
        island_dilate: int = 1,
        smooth: bool = True,
        smooth_sigma: float = 1.5,
        smooth_sigma_z: Optional[float] = None,
        smooth_sigma_xy: Optional[float] = None,
        preserve_topology: bool = True,
        n_workers: Optional[int] = None,
        voxel_size_um: Optional[Tuple[float, float, float]] = None,
        dry_run: bool = False,
    ):
        super().__init__("postprocess")
        self.input_dir = str(Path(input_dir).resolve()) if input_dir else None
        self.output_dir = str(Path(output_dir).resolve()) if output_dir else None
        self.recon_root = str(Path(recon_root).resolve()) if recon_root else None
        self.remove_small = remove_small
        self.min_voxels = int(min_voxels)
        self.min_percentile = float(min_percentile)
        self.remove_large = remove_large
        self.max_voxels = int(max_voxels)
        self.max_percentile = float(max_percentile)
        self.remove_islands = remove_islands
        self.island_downsample = int(island_downsample)
        self.island_min_component_frac = float(island_min_component_frac)
        self.island_dilate = int(island_dilate)
        self.smooth = smooth
        self.smooth_sigma = float(smooth_sigma)
        self.smooth_sigma_z = smooth_sigma_z
        self.smooth_sigma_xy = smooth_sigma_xy
        self.preserve_topology = preserve_topology
        self.n_workers = n_workers
        self.voxel_size_um = tuple(float(v) for v in voxel_size_um) if voxel_size_um else None
        self.dry_run = dry_run

    def get_params(self) -> dict:
        return {
            "remove_small": self.remove_small, "min_voxels": self.min_voxels,
            "min_percentile": self.min_percentile,
            "remove_large": self.remove_large, "max_voxels": self.max_voxels,
            "max_percentile": self.max_percentile,
            "remove_islands": self.remove_islands,
            "island_downsample": self.island_downsample,
            "island_min_component_frac": self.island_min_component_frac,
            "island_dilate": self.island_dilate,
            "smooth": self.smooth, "smooth_sigma": self.smooth_sigma,
            "smooth_sigma_z": self.smooth_sigma_z, "smooth_sigma_xy": self.smooth_sigma_xy,
            "preserve_topology": self.preserve_topology,
            "n_workers": self.n_workers, "voxel_size_um": self.voxel_size_um,
            "dry_run": self.dry_run,
        }

    def _recon_root(self, tile_dirs: List[Path]) -> Path:
        if self.recon_root:
            return Path(self.recon_root)
        return tile_dirs[0].parent / "stage2_reconstruction"

    def _in_dir(self, tile_dirs: List[Path]) -> Path:
        if self.input_dir:
            return Path(self.input_dir)
        # Mirror IoUStitchStage's default output: <recon_root>.parent/stage4_full_stitch
        # (so a default-path stitch + a default-path postprocess agree exactly).
        return self._recon_root(tile_dirs).parent / "stage4_full_stitch"

    def validate_inputs(self, tile_dirs: List[Path]) -> bool:
        if not tile_dirs:
            return False
        return (self._in_dir(tile_dirs) / "global_labels.npy").exists()

    def run(
        self,
        tile_dirs: List[Path],
        n_gpus: int,
        log_dir: Optional[Path] = None,
    ) -> StageResult:
        logger = logging.getLogger("epicycle.postprocess")
        result = StageResult()
        if not tile_dirs:
            result.add_failure("(none)", "no tile directories provided")
            return result

        in_dir = self._in_dir(tile_dirs)
        out_dir = Path(self.output_dir) if self.output_dir else in_dir
        src_npy = in_dir / "global_labels.npy"
        prov_csv = in_dir / "label_provenance.csv"
        clean_npy = out_dir / "global_labels_clean.npy"
        clean_tif = out_dir / "global_labels_clean.tif"
        clean_prov = out_dir / "label_provenance_clean.csv"
        manifest_path = out_dir / "postprocess_manifest.json"

        if not src_npy.exists():
            msg = f"postprocess: no global_labels.npy in {in_dir} (run the iou_stitch stage first)"
            logger.error(msg)
            for td in tile_dirs:
                result.add_failure(td.name, msg)
            return result
        if not prov_csv.exists():
            msg = (f"postprocess: no label_provenance.csv in {in_dir}; cleanup needs the "
                   f"per-cell provenance (run iou_stitch with emit_provenance=True)")
            logger.error(msg)
            for td in tile_dirs:
                result.add_failure(td.name, msg)
            return result

        # Idempotency: skip ONLY if a prior run COMPLETED. A clean TIFF can exist
        # while truncated (crash/OOM/timeout mid-write), so gate on the manifest
        # status — never skip on mere file existence.
        if clean_tif.exists() and not self.dry_run:
            prior_status = None
            if manifest_path.exists():
                try:
                    prior_status = json.loads(manifest_path.read_text()).get("status")
                except Exception:
                    prior_status = None
            if prior_status == "complete":
                logger.info(f"postprocess: {clean_tif.name} exists + manifest complete "
                            f"— skipping (idempotent).")
                for td in tile_dirs:
                    result.add_success(td.name)
                return result
            logger.info(f"postprocess: {clean_tif.name} exists but prior status is "
                        f"{prior_status!r} (not complete) — redoing.")

        src = np.load(str(src_npy), mmap_mode="r")
        depth, H, W = int(src.shape[0]), int(src.shape[1]), int(src.shape[2])
        full_shape = (depth, H, W)
        rows = _read_provenance(prov_csv)
        all_ids = [_ri(r, "dense_id") for r in rows]
        max_id = max(all_ids, default=0)
        n_before = len(all_ids)

        # --- decisions (CSV-only) ---------------------------------------------
        vcs = np.array([_ri(r, "voxel_count") for r in rows], dtype=np.int64)
        # A percentile (when > 0) RESOLVES to an absolute voxel threshold from THIS
        # run's distribution and OVERRIDES the absolute knob — handy when you don't
        # know your cell sizes yet (min_percentile auto-captures the confetti).
        eff_min = self.min_voxels
        if self.remove_small and self.min_percentile and self.min_percentile > 0 and vcs.size:
            eff_min = int(np.percentile(vcs, float(self.min_percentile)))
        eff_max = self.max_voxels
        if self.remove_large and self.max_percentile and self.max_percentile > 0 and vcs.size:
            eff_max = int(np.percentile(vcs, float(self.max_percentile)))

        # Cells that clipped/merged entirely out of the stitched volume carry
        # voxel_count == 0 (documented by the stitch provenance writer). They are
        # never real instances, so drop them UNCONDITIONALLY — otherwise, with the
        # size filter off (or min_voxels == 0), they would be renumbered into the
        # clean volume/CSV as zero-voxel ghost rows.
        dropped_zero = {_ri(r, "dense_id") for r in rows if _ri(r, "voxel_count") <= 0}
        dropped_size = (
            select_dropped_by_size(rows, eff_min) if self.remove_small else set()
        )
        dropped_large = (
            select_dropped_by_max_size(rows, eff_max) if self.remove_large else set()
        )
        dropped_island = (
            select_dropped_islands(
                rows, full_shape,
                downsample=self.island_downsample,
                min_component_frac=self.island_min_component_frac,
                dilate=self.island_dilate,
                exclude=dropped_size | dropped_large | dropped_zero,
            )
            if self.remove_islands else set()
        )
        dropped = dropped_size | dropped_large | dropped_island | dropped_zero
        lut = build_relabel_lut(all_ids, dropped)
        n_after = len(lut)

        vstats = _voxel_stats(vcs)
        vox_um3 = None
        if self.voxel_size_um:
            sz_um, sy_um, sx_um = self.voxel_size_um
            vox_um3 = sz_um * sy_um * sx_um

        def _um3(n: int) -> str:
            return f" (~{n * vox_um3:.3g} um^3)" if vox_um3 else ""

        logger.info(
            f"postprocess: {n_before} cells -> {n_after} kept "
            f"(dropped {len(dropped_size)} small + {len(dropped_large)} large + "
            f"{len(dropped_island)} island); voxel_count {vstats}"
            + (f"; 1 voxel = {vox_um3:.4g} um^3" if vox_um3 else "")
        )
        if self.remove_small:
            via = f"min_percentile={self.min_percentile}" if self.min_percentile else "absolute"
            logger.info(f"  small filter: drop voxel_count < {eff_min}{_um3(eff_min)} [{via}]")
        if self.remove_large:
            via = f"max_percentile={self.max_percentile}" if self.max_percentile else "absolute"
            logger.info(f"  large filter: drop voxel_count > {eff_max}{_um3(eff_max)} [{via}]")
        for thr in (100, 200, 500, 1000, 2000):
            nd = int((vcs < thr).sum())
            pct = (100.0 * nd / n_before) if n_before else 0.0
            logger.info(f"  size<{thr}: would drop {nd} ({pct:.1f}%)")
        # upper-end preview (calibrate max_voxels): cells far above the median.
        med = float(np.median(vcs)) if vcs.size else 0.0
        for mult in (5, 10, 20, 50):
            thr = int(mult * med)
            nd = int((vcs > thr).sum())
            pct = (100.0 * nd / n_before) if n_before else 0.0
            logger.info(f"  size>{mult}x median (~{thr}): would drop {nd} ({pct:.1f}%)")

        def _manifest(status: str) -> dict:
            return {
                "version": _MANIFEST_VERSION, "stage": "postprocess", "status": status,
                "input_dir": str(in_dir), "source_npy": str(src_npy),
                "output_npy": str(clean_npy), "output_tif": str(clean_tif),
                "output_provenance": str(clean_prov),
                "full_size": [H, W], "depth": depth, "dtype": "int32",
                "remove_small": self.remove_small, "min_voxels": self.min_voxels,
                "min_percentile": self.min_percentile, "eff_min_voxels": eff_min,
                "remove_large": self.remove_large, "max_voxels": self.max_voxels,
                "max_percentile": self.max_percentile, "eff_max_voxels": eff_max,
                "voxel_size_um": list(self.voxel_size_um) if self.voxel_size_um else None,
                "voxel_volume_um3": vox_um3,
                "remove_islands": self.remove_islands,
                "island_downsample": self.island_downsample,
                "island_min_component_frac": self.island_min_component_frac,
                "island_dilate": self.island_dilate,
                "smooth": self.smooth, "smooth_sigma": self.smooth_sigma,
                "smooth_sigma_z": self.smooth_sigma_z, "smooth_sigma_xy": self.smooth_sigma_xy,
                "preserve_topology": self.preserve_topology,
                "n_cells_before": n_before, "n_cells_after": n_after,
                "n_dropped_size": len(dropped_size), "n_dropped_large": len(dropped_large),
                "n_dropped_island": len(dropped_island),
                "n_dropped_zero_voxel": len(dropped_zero),
                "dry_run": self.dry_run,
                "voxel_count_stats": vstats,
            }

        out_dir.mkdir(parents=True, exist_ok=True)

        if self.dry_run:
            with open(manifest_path, "w") as f:
                json.dump(_manifest("dry_run"), f, indent=2, default=str)
            logger.info(
                f"postprocess DRY-RUN complete: would keep {n_after}/{n_before} cells "
                f"(no volume written). Tune min_voxels and re-run with dry_run=False."
            )
            for td in tile_dirs:
                result.add_success(td.name)
            return result

        lut_arr = lut_to_array(lut, max_id)
        status = "incomplete"
        try:
            with open(manifest_path, "w") as f:
                json.dump(_manifest("incomplete"), f, indent=2, default=str)

            if self.smooth:
                stats = self._write_smoothed(
                    src, src_npy, lut, rows, clean_npy, clean_tif, full_shape, logger
                )
            else:
                stats = self._write_relabel(
                    src, lut_arr, clean_npy, clean_tif, full_shape, logger
                )

            prov = _clean_provenance_inputs(rows, lut)
            _write_label_provenance(clean_prov, prov, stats)
            status = "complete"
            for td in tile_dirs:
                result.add_success(td.name)
        except Exception as e:  # pragma: no cover - exercised via failure paths
            logger.error(f"postprocess failed: {e}")
            for td in tile_dirs:
                result.add_failure(td.name, str(e))
        finally:
            with open(manifest_path, "w") as f:
                json.dump(_manifest(status), f, indent=2, default=str)

        logger.info(
            f"postprocess {status}: {n_after} cells -> {clean_npy.name} + {clean_tif.name}"
        )
        return result

    # -- writers ---------------------------------------------------------------

    def _write_relabel(
        self, src, lut_arr, clean_npy, clean_tif, full_shape, logger
    ) -> Dict[int, dict]:
        """smooth OFF: slice-streaming relabel (read slice -> LUT remap -> write)."""
        import tifffile
        from tqdm import tqdm

        depth, H, W = full_shape
        max_idx = lut_arr.shape[0] - 1  # largest id the LUT can index
        out = np.lib.format.open_memmap(
            str(clean_npy), mode="w+", dtype=np.int32, shape=full_shape
        )
        stats: Dict[int, dict] = {}
        try:
            with tifffile.TiffWriter(str(clean_tif), bigtiff=True) as tif:
                for z in tqdm(range(depth), desc="postprocess relabel"):
                    sl_src = np.asarray(src[z])
                    # Guard against a volume id beyond the provenance LUT (CSV/volume
                    # drift): map any out-of-range id to background 0 (no IndexError).
                    if int(sl_src.max()) > max_idx:
                        sl_src = np.where(sl_src <= max_idx, sl_src, 0)
                    sl = lut_arr[sl_src].astype(np.int32)
                    out[z] = sl
                    tif.write(sl, contiguous=True)
                    _accumulate_slice(stats, sl, z, depth)
            out.flush()
        finally:
            del out
        return stats

    def _write_smoothed(
        self, src, src_npy, lut, rows, clean_npy, clean_tif, full_shape, logger
    ) -> Dict[int, dict]:
        """smooth ON: parallel per-instance smooth + serialized safe-write scatter."""
        import tifffile
        from tqdm import tqdm

        depth, H, W = full_shape
        sz, sxy = resolve_sigmas(self.smooth_sigma, self.smooth_sigma_z, self.smooth_sigma_xy)
        # Pad the bbox so the smoothed (slightly grown) mask isn't clipped at the crop edge.
        pad = int(np.ceil(3.0 * max(sz, sxy))) + 1

        rows_by_id = {_ri(r, "dense_id"): r for r in rows}
        tasks = []
        for old_id, new_id in lut.items():
            r = rows_by_id[old_id]
            z0 = max(0, _ri(r, "z_min") - pad); z1 = min(depth, _ri(r, "z_max") + 1 + pad)
            y0 = max(0, _ri(r, "y_min") - pad); y1 = min(H, _ri(r, "y_max") + 1 + pad)
            x0 = max(0, _ri(r, "x_min") - pad); x1 = min(W, _ri(r, "x_max") + 1 + pad)
            tasks.append((old_id, new_id, (z0, z1, y0, y1, x0, x1), sz, sxy, self.preserve_topology))

        n_workers = self.n_workers or max(1, get_allocated_cpus() - 1)
        n_workers = max(1, min(int(n_workers), len(tasks))) if tasks else 1

        global _WORKER_SRC
        # A freshly open_memmap'd .npy is already zero-backed (unwritten file regions
        # read as 0 on Linux + Windows); the safe-write scatter only touches foreground
        # voxels, so an explicit out[:] = 0 (a redundant full-volume write pass) is
        # avoided at scale.
        out = np.lib.format.open_memmap(
            str(clean_npy), mode="w+", dtype=np.int32, shape=full_shape
        )
        n_failed = [0]

        def _scatter(res) -> None:
            if res is None:
                return
            if res[0] == "ERROR":  # ("ERROR", new_id, msg) — crop read failed
                n_failed[0] += 1
                logger.warning(f"postprocess: smooth failed for cell {res[1]}: {res[2]}")
                return
            new_id, (z0, y0, x0), mask = res
            dz, dy, dx = mask.shape
            region = out[z0:z0 + dz, y0:y0 + dy, x0:x0 + dx]
            # Safe-write: only fill voxels still background — never overwrite a
            # neighbour that already claimed them (boundary arbitration by order).
            region[mask & (region == 0)] = new_id

        logger.info(
            f"postprocess: smoothing {len(tasks)} instance(s) on {n_workers} worker(s) "
            f"(sigma_z={sz}, sigma_xy={sxy}, pad={pad})"
        )
        try:
            if n_workers <= 1:
                _init_smooth_worker(str(src_npy))
                try:
                    for t in tqdm(tasks, desc="postprocess smooth"):
                        _scatter(_smooth_one(t))
                finally:
                    _WORKER_SRC = None  # release the main-process memmap handle
            else:
                from multiprocessing import Pool
                with Pool(
                    n_workers, initializer=_init_smooth_worker, initargs=(str(src_npy),)
                ) as pool:
                    # imap (ORDERED) is REQUIRED: the safe-write scatter arbitrates
                    # overlapping smoothed boundaries by submission order, so switching
                    # to imap_unordered would make the output non-deterministic.
                    for res in tqdm(
                        pool.imap(_smooth_one, tasks, chunksize=8),
                        total=len(tasks), desc="postprocess smooth",
                    ):
                        _scatter(res)
            if n_failed[0]:
                logger.warning(
                    f"postprocess: {n_failed[0]} cell(s) could not be read for smoothing "
                    f"(dropped from the clean volume — see warnings above)"
                )
            out.flush()

            # Stream the scattered memmap -> BigTIFF + accumulate post-smooth stats.
            stats: Dict[int, dict] = {}
            with tifffile.TiffWriter(str(clean_tif), bigtiff=True) as tif:
                for z in tqdm(range(depth), desc="postprocess write tif"):
                    sl = np.asarray(out[z])
                    tif.write(sl, contiguous=True)
                    _accumulate_slice(stats, sl, z, depth)
        finally:
            del out
        return stats


__all__ = ["PostprocessStage"]
