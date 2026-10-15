"""
Reconstruction stage for EpiCycle ScaledSegmentation.

Runs USegment3D direct-method 3D reconstruction from saved 2D
segmentation outputs (flows + probabilities), with optional
post-processing (flow consistency), binary extraction with GPU
erosion, and full-Z recovery.

Supports sequential (one tile at a time) and parallel (multiple tiles
across GPUs) execution modes. Parallel mode uses a configurable
``max_workers`` to limit concurrent reconstructions and avoid OOM.
"""

import os
import sys
import gc
import json
import time
import traceback
import numpy as np
from pathlib import Path
from typing import List, Optional
from multiprocessing import get_context
from itertools import cycle

from packages.epicycle_scaledseg.core.pipeline import Stage, StageResult
from packages.epicycle_scaledseg.utils.io import (
    base_name, ensure_dir, now_iso, update_tile_config, load_array, pick_array_path,
)
from packages.epicycle_scaledseg.utils.array_ops import recover_original_dimensions

# Effective CPU cores one u-Segment3D direct-method recon can actually use. Its
# aggregation parallelizes over spatial tiles via a multiprocess.ThreadPool (the
# process Pool is commented out in the library), so it is GIL-bound and ignores
# our n_cpu; a source-level audit put the plateau at ~2-6 effective cores. We use
# the low end so concurrency (cores // this) doesn't over-subscribe.
RECON_EFFECTIVE_CORES_PER_TILE = 4


def compute_recon_concurrency(
    total_cpus: int,
    avail_ram_gb: float,
    peak_per_tile_gb: float,
    n_tiles: int,
    cores_per_tile: int = RECON_EFFECTIVE_CORES_PER_TILE,
    ram_fraction: float = 0.8,
    ram_reserve_gb: float = 16.0,
) -> int:
    """How many recon tiles to run concurrently — bounded by CPU and RAM, NOT GPUs.

    Recon's heavy step (u-Segment3D direct aggregation) is CPU+RAM bound and can
    only use ~``cores_per_tile`` effective cores per tile, so one tile leaves a fat
    node idle. Run ``M`` tiles at once where ``M`` is the min of:

      * ``total_cpus // cores_per_tile`` — each tile effectively uses ~cores_per_tile,
      * ``floor((avail_ram_gb - ram_reserve_gb) * ram_fraction / peak_per_tile_gb)``
        — the RAM bound, with a fixed reserve carved out for the parent process /
        OS / page-cache and an ``ram_fraction`` headroom on top, because the
        a-priori ``peak_per_tile_gb`` is a *model* that may undercount u-Segment3D's
        transients (the first tile's measured RSS calibrates it later),
      * ``n_tiles`` — never more workers than work.

    Floored at 1. **Unknown RAM (``avail_ram_gb <= 0``) or unknown peak
    (``peak_per_tile_gb <= 0``) is treated as the MOST conservative case (RAM bound
    = 1), never CPU-capped** — "can't measure memory" must mean run one at a time,
    not ignore memory and risk an OOM.
    """
    by_cpu = max(1, int(total_cpus) // max(1, int(cores_per_tile)))
    if peak_per_tile_gb and peak_per_tile_gb > 0 and avail_ram_gb and avail_ram_gb > 0:
        usable = max(0.0, float(avail_ram_gb) - float(ram_reserve_gb))
        by_ram = int((usable * ram_fraction) // peak_per_tile_gb)
    else:
        by_ram = 1  # unknown RAM/peak -> most conservative, NOT CPU-capped
    return max(1, min(by_cpu, by_ram, max(1, int(n_tiles))))


# ---------------------------------------------------------------------------
# Measured-RSS calibration — turn the a-priori RAM model into a measurement
# ---------------------------------------------------------------------------

_RECON_CALIBRATION_FILE = "recon_calibration.json"


def _peak_rss_gb() -> Optional[float]:
    """This process's PEAK resident-set size in GB, cross-platform. None if N/A.

    Uses the OS high-water mark (no polling): ``peak_wset`` on Windows,
    ``ru_maxrss`` on Linux/mac. In a recon worker that processed one tile in
    isolation, this is that tile's true peak RAM — the measurement that replaces
    the static estimate.
    """
    try:
        if sys.platform.startswith("win"):
            import psutil
            return float(psutil.Process().memory_info().peak_wset) / (1024 ** 3)
        import resource
        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        unit = 1024 if sys.platform.startswith("linux") else 1  # KB on Linux, bytes on mac
        return float(maxrss) * unit / (1024 ** 3)
    except Exception:
        return None


def _calibration_path(out_root) -> Optional[Path]:
    return Path(out_root) / _RECON_CALIBRATION_FILE if out_root else None


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Atomic JSON write with a PID-UNIQUE temp name, so M concurrent worker
    processes never clobber each other's stage file (a shared temp name caused
    ~20% swallowed FileNotFoundError/PermissionError on the parallel batch)."""
    ensure_dir(path.parent)
    tmp = path.with_suffix(f".{os.getpid()}.json.tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(str(tmp), str(path))


def _read_calibration(out_root) -> dict:
    """The raw calibration sidecar dict for this workspace ({} if absent/corrupt)."""
    p = _calibration_path(out_root)
    if not p or not p.exists():
        return {}
    try:
        d = json.loads(p.read_text())
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _read_calibration_peak(out_root, current_apriori_gb: Optional[float] = None) -> Optional[float]:
    """The recorded MEASURED per-tile peak RAM (GB), or None if absent OR STALE.

    Staleness: the sidecar stamps the a-priori estimate (``apriori_gb``) the
    measurement was taken for — a tile-size signature. If the CURRENT a-priori
    estimate differs from it by >20% (e.g. the workspace was re-tiled to bigger
    tiles), the old measurement would size M too high -> OOM, so we return None and
    force a re-calibration. A legacy sidecar with no ``apriori_gb`` is trusted.
    """
    d = _read_calibration(out_root)
    val = d.get("measured_peak_gb")
    try:
        if val is None or float(val) <= 0:
            return None
        val = float(val)
    except Exception:
        return None
    stored_ap = d.get("apriori_gb")
    if (current_apriori_gb and stored_ap and float(stored_ap) > 0
            and abs(float(current_apriori_gb) - float(stored_ap)) > 0.2 * float(stored_ap)):
        return None  # tile size changed since calibration -> stale, recalibrate
    return val


def _write_calibration_peak(out_root, peak_gb) -> None:
    """Worker: record the MAX measured per-tile peak (GB), preserving ``apriori_gb``.
    Best-effort + atomic. Single-tile peak (each worker measures its own process);
    max so a concurrent batch can't undercount."""
    p = _calibration_path(out_root)
    if not p or peak_gb is None:
        return
    try:
        d = _read_calibration(out_root)
        prev = float(d.get("measured_peak_gb", 0.0) or 0.0)
        if float(peak_gb) > prev:
            d["measured_peak_gb"] = float(peak_gb)
            _atomic_write_json(p, d)
    except Exception:
        pass


def _stamp_calibration_apriori(out_root, apriori_gb) -> None:
    """Stage: record the a-priori estimate this measurement corresponds to (the
    tile-size signature for staleness detection), preserving ``measured_peak_gb``."""
    p = _calibration_path(out_root)
    if not p or apriori_gb is None:
        return
    try:
        d = _read_calibration(out_root)
        d["apriori_gb"] = float(apriori_gb)
        _atomic_write_json(p, d)
    except Exception:
        pass


def _pin_inner_cpu_count(n_cpu) -> None:
    """Cap ``cpu_count()`` to ``n_cpu`` in THIS worker process, so u-Segment3D's
    hardcoded ``ThreadPool(multiprocess.cpu_count() - 1)`` — which ignores the
    OMP/MKL env pins — uses only the allocated/fair-share threads, not the machine's
    full physical core count.

    Two reasons this matters: (1) under parallel recon, M workers each spawning
    ~machine-cores threads would oversubscribe; capping to ``cores // M`` keeps the
    total near the real core count. (2) even in sequential mode it keeps u-Segment3D
    inside a SLURM CPU allocation (``cpu_count()`` reports the NODE's cores, not the
    cgroup's). Output-neutral — a ThreadPool's size never changes results, only the
    thread count. Best-effort; ``multiprocess`` is u-Segment3D's actual dependency,
    ``multiprocessing`` is patched too as belt-and-suspenders.

    Output-neutrality is verified against the installed u-Segment3D: ``cpu_count``
    feeds ONLY ``ThreadPool(cpu_count()-1)``, never a numerical result. It would
    feed tile GEOMETRY (``mp_gradient_watershed3D_binary``'s ``tile_shape is None``
    auto-infer branch) — but EpiCycle always passes an explicit ``tile_shape``
    (config default (128,256,256)), so that branch stays dead. If a caller ever
    sets ``tile_shape=None``, this cap would change the segmentation — keep
    ``tile_shape`` non-None.
    """
    try:
        n = int(n_cpu)
    except (TypeError, ValueError):
        return
    if n < 1:
        return
    target = n + 1  # u-Segment3D does cpu_count()-1, so +1 yields exactly n threads
    import importlib
    for modname in ("multiprocess", "multiprocessing"):
        try:
            mod = importlib.import_module(modname)
            mod.cpu_count = lambda _t=target: _t
        except Exception:
            pass


def apply_gd_overrides(params, n_iter=None, gradient_decay=None, momenta=None):
    """Return a COPY of ``params`` with per-tile gradient-descent overrides applied.

    Only non-None values override; the ``gradient_descent`` sub-dict is copied so the
    caller's dict (and the shared module defaults) are never mutated. This lets the
    notebook/CLI tune the per-tile reconstruction's ``n_iter`` / ``gradient_decay`` /
    ``momenta`` without rebuilding the full u-Segment3D parameter dict.
    """
    out = dict(params)
    gd = dict(out.get("gradient_descent", {}))
    if n_iter is not None:
        gd["n_iter"] = int(n_iter)
    if gradient_decay is not None:
        gd["gradient_decay"] = float(gradient_decay)
    if momenta is not None:
        gd["momenta"] = float(momenta)
    out["gradient_descent"] = gd
    return out


class ReconstructionStage(Stage):
    """
    3D reconstruction stage using USegment3D direct method.

    Loads per-view flows and probabilities from stage1 segmentation,
    aggregates them into a 3D label volume, optionally post-processes
    with flow consistency, and extracts a binary mask with GPU erosion.

    Parameters
    ----------
    seg_root : root directory containing stage1 segmentation outputs
        (each tile is a subdirectory with flows/probs .npy files)
    out_root : output root for stage2 reconstruction results
    mode : 'sequential' or 'parallel'
    reconstruct_params : USegment3D aggregation params (uses defaults if None)
    postprocess_params : USegment3D postprocess params (uses defaults if None)
    n_iter / gradient_decay / momenta : per-tile gradient-descent overrides applied
        on top of the resolved params (None = use the tuned defaults n_iter=25,
        gradient_decay=0.001, momenta=0.98). More iterations / lower decay = finer
        field integration at the cost of time. (Distinct from the Mode-A *global*
        recon's GLOBAL_N_ITER, which does not apply in Mode B / full.)
    save_prob3d : save 3D probability volume
    save_flow3d : save 3D gradient volume
    save_binary : extract and save binary mask
    binary_erode : erode instances before binary extraction
    binary_kernel_size : erosion kernel size
    binary_border_margin : restore original labels near tile borders
    binary_fill_holes : fill holes in binary mask
    cleanup_intermediates : delete stage1 flows/probs after reconstruction
    n_cpu_per_worker : CPU threads per worker process.
        If None, auto-detected from SLURM allocation / cgroup affinity.
        In sequential mode this defaults to ALL allocated cores;
        in parallel mode it defaults to total_cores / n_gpus.
    max_workers : maximum number of tiles to reconstruct concurrently
        in parallel mode. Defaults to n_gpus (one tile per GPU).
        Reduce this if you hit system RAM OOM — each tile uses ~25 GB
        peak during aggregation.
    """

    def __init__(
        self,
        seg_root: str = None,
        out_root: str = None,
        mode: str = 'sequential',
        reconstruct_params: dict = None,
        postprocess_params: dict = None,
        n_iter: int = None,
        gradient_decay: float = None,
        momenta: float = None,
        save_prob3d: bool = True,
        save_flow3d: bool = False,
        save_binary: bool = True,
        binary_erode: bool = True,
        binary_kernel_size: int = 3,
        binary_border_margin: int = 2,
        binary_fill_holes: bool = True,
        cleanup_intermediates: bool = False,
        n_cpu_per_worker: int = None,
        max_workers: int = None,
        pre_smooth_flows: bool = False,
        pre_smooth_probs: bool = False,
        pre_smooth_sigma_perp: float = 3.0,
        pre_smooth_sigma_inplane: float = 1.0,
        flow_mask_threshold: float = 0.2,
        flow_soft_weighting: bool = True,
    ):
        super().__init__('reconstruction')
        self.seg_root = str(Path(seg_root).resolve()) if seg_root else None
        self.out_root = str(Path(out_root).resolve()) if out_root else None
        self.mode = mode
        self.reconstruct_params = reconstruct_params
        self.postprocess_params = postprocess_params
        # Per-tile gradient-descent overrides (None = use the tuned defaults
        # n_iter=25, gradient_decay=0.001, momenta=0.98).
        self.gd_n_iter = n_iter
        self.gd_gradient_decay = gradient_decay
        self.gd_momenta = momenta
        self.save_prob3d = save_prob3d
        self.save_flow3d = save_flow3d
        self.save_binary = save_binary
        self.binary_erode = binary_erode
        self.binary_kernel_size = binary_kernel_size
        self.binary_border_margin = binary_border_margin
        self.binary_fill_holes = binary_fill_holes
        self.cleanup_intermediates = cleanup_intermediates
        self.n_cpu_per_worker = n_cpu_per_worker
        self.max_workers = max_workers
        self.pre_smooth_flows = pre_smooth_flows
        self.pre_smooth_probs = pre_smooth_probs
        self.pre_smooth_sigma_perp = pre_smooth_sigma_perp
        self.pre_smooth_sigma_inplane = pre_smooth_sigma_inplane
        self.flow_mask_threshold = flow_mask_threshold
        self.flow_soft_weighting = flow_soft_weighting

    def get_params(self) -> dict:
        return {
            "mode": self.mode,
            "reconstruct_params": self.reconstruct_params,
            "postprocess_params": self.postprocess_params,
            "gd_n_iter": self.gd_n_iter,
            "gd_gradient_decay": self.gd_gradient_decay,
            "gd_momenta": self.gd_momenta,
            "save_prob3d": self.save_prob3d,
            "save_flow3d": self.save_flow3d,
            "save_binary": self.save_binary,
            "binary_erode": self.binary_erode,
            "binary_kernel_size": self.binary_kernel_size,
            "binary_border_margin": self.binary_border_margin,
            "binary_fill_holes": self.binary_fill_holes,
            "cleanup_intermediates": self.cleanup_intermediates,
            "n_cpu_per_worker": self.n_cpu_per_worker,
            "max_workers": self.max_workers,
            "pre_smooth_flows": self.pre_smooth_flows,
            "pre_smooth_probs": self.pre_smooth_probs,
            "pre_smooth_sigma_perp": self.pre_smooth_sigma_perp,
            "pre_smooth_sigma_inplane": self.pre_smooth_sigma_inplane,
            "flow_mask_threshold": self.flow_mask_threshold,
            "flow_soft_weighting": self.flow_soft_weighting,
        }

    def validate_inputs(self, tile_dirs: List[Path]) -> bool:
        """Check that segmentation outputs exist for each tile."""
        import logging
        logger = logging.getLogger("epicycle.reconstruction")
        for td in tile_dirs:
            seg_dir = self._seg_dir(td)
            if not seg_dir.is_dir():
                logger.error(f"validate_inputs: seg_dir missing: {seg_dir}")
                return False
            if not (seg_dir / "SEGMENTATION_DONE").exists():
                logger.error(f"validate_inputs: SEGMENTATION_DONE missing in {seg_dir}")
                return False
        logger.info(f"validate_inputs: all {len(tile_dirs)} tiles passed")
        return True

    def _resolve_n_cpu(self, n_gpus: int) -> int:
        """Determine CPU threads per worker based on mode and allocation."""
        if self.n_cpu_per_worker is not None:
            return self.n_cpu_per_worker

        from packages.epicycle_scaledseg.utils.resources import get_allocated_cpus
        total = get_allocated_cpus()

        if self.mode == 'sequential':
            return total  # one tile at a time, use everything
        else:
            # max(1, n_gpus) guards the divisor: get_allocated_gpus() can now
            # report 0 (CPU-only), and recon aggregation is CPU-bound (erosion
            # CPU-falls-back), so 0 GPUs must not ZeroDivisionError here.
            return max(1, total // max(1, n_gpus))  # split across GPU workers

    def run(
        self,
        tile_dirs: List[Path],
        n_gpus: int,
        log_dir: Optional[Path] = None,
    ) -> StageResult:
        """Dispatch reconstruction in sequential or parallel mode."""
        # Filter out already-completed tiles
        todo = []
        for td in tile_dirs:
            out_dir = self._out_dir(td)
            bname = td.name
            if (out_dir / f"{bname}_mask.tif").exists():
                continue
            todo.append(td)

        if not todo:
            result = StageResult()
            for td in tile_dirs:
                result.add_success(td.name)
            return result

        # Resolve CPU count once for all workers
        self._resolved_n_cpu = self._resolve_n_cpu(n_gpus)
        # Resolve default params once (so we can log/save the actual values used)
        if self.reconstruct_params is None:
            from packages.epicycle_scaledseg.config.defaults import get_default_reconstruct_params
            self._resolved_reconstruct_params = get_default_reconstruct_params(
                n_cpu=self._resolved_n_cpu
            )
        else:
            self._resolved_reconstruct_params = self.reconstruct_params
        # Apply per-tile gradient-descent overrides (n_iter / gradient_decay / momenta)
        # on top of whichever params were resolved — non-mutating.
        self._resolved_reconstruct_params = apply_gd_overrides(
            self._resolved_reconstruct_params,
            n_iter=self.gd_n_iter,
            gradient_decay=self.gd_gradient_decay,
            momenta=self.gd_momenta,
        )
        if self.postprocess_params is None:
            from packages.epicycle_scaledseg.config.defaults import get_default_postprocess_segmentation_params
            self._resolved_postprocess_params = get_default_postprocess_segmentation_params(
                n_cpu=self._resolved_n_cpu
            )
        else:
            self._resolved_postprocess_params = self.postprocess_params

        # Write a single params JSON at the reconstruction root for this run
        try:
            recon_root = Path(self.out_root) if self.out_root else tile_dirs[0].parent / "stage2_reconstruction"
            ensure_dir(recon_root)
            params_path = recon_root / "reconstruction_params.json"
            params_payload = {
                "mode": self.mode,
                "max_workers": self.max_workers,
                "n_cpu_per_worker": self._resolved_n_cpu,
                "save_prob3d": self.save_prob3d,
                "save_flow3d": self.save_flow3d,
                "save_binary": self.save_binary,
                "binary_erode": self.binary_erode,
                "binary_kernel_size": self.binary_kernel_size,
                "binary_border_margin": self.binary_border_margin,
                "binary_fill_holes": self.binary_fill_holes,
                "cleanup_intermediates": self.cleanup_intermediates,
                "skip_postprocess": True,
                "pre_smooth_flows": self.pre_smooth_flows,
                "pre_smooth_probs": self.pre_smooth_probs,
                "pre_smooth_sigma_perp": self.pre_smooth_sigma_perp,
                "pre_smooth_sigma_inplane": self.pre_smooth_sigma_inplane,
                "flow_mask_threshold": self.flow_mask_threshold,
                "flow_soft_weighting": self.flow_soft_weighting,
                "reconstruct_params": self._resolved_reconstruct_params,
                "postprocess_params": self._resolved_postprocess_params,
            }
            with open(params_path, "w") as f:
                json.dump(params_payload, f, indent=2)
        except Exception as e:
            import logging
            logging.getLogger("epicycle.reconstruction").warning(
                f"Failed to write reconstruction_params.json: {e}"
            )

        if self.mode == 'sequential':
            return self._run_sequential(todo, tile_dirs, n_gpus, log_dir)
        else:
            return self._run_parallel(todo, tile_dirs, n_gpus, log_dir)

    def _run_sequential(self, todo, all_tiles, n_gpus, log_dir):
        """Process tiles one at a time on GPU 0."""
        import logging
        logger = logging.getLogger("epicycle.reconstruction")

        ctx = get_context("spawn")
        for td in todo:
            seg_dir = str(self._seg_dir(td))
            out_root = self.out_root
            logger.info(
                f"Launching reconstruction for {td.name} "
                f"(seg_dir={seg_dir}, out_root={out_root})"
            )
            p = ctx.Process(
                target=_recon_gpu_worker,
                args=(
                    0,
                    [seg_dir],
                    self._build_worker_kwargs(),
                    str(log_dir) if log_dir else "logs",
                ),
            )
            p.start()
            p.join()
            if p.exitcode != 0:
                logger.error(
                    f"Reconstruction worker for {td.name} exited with code {p.exitcode}. "
                    f"Retrying in-process to capture traceback..."
                )
                try:
                    _recon_gpu_worker(
                        0,
                        [seg_dir],
                        self._build_worker_kwargs(),
                        str(log_dir) if log_dir else "logs",
                    )
                except Exception as e:
                    logger.error(f"In-process retry for {td.name} failed: {e}\n{traceback.format_exc()}")

        return self._collect_results(all_tiles)

    def _run_parallel(self, todo, all_tiles, n_gpus, log_dir):
        """Reconstruct multiple tiles concurrently, sized by RAM + CPU (NOT GPUs).

        The heavy step (u-Segment3D direct aggregation) is CPU+RAM bound and can
        only use ~RECON_EFFECTIVE_CORES_PER_TILE effective cores per tile (GIL-bound
        ThreadPool that ignores n_cpu — source-audited), so a fat node should run
        many tiles at once. ``M = compute_recon_concurrency(...)`` is decoupled from
        GPU count; each worker's CPU threads are pinned to ``cores // M`` to avoid
        oversubscription; GPUs are round-robined only for the brief optional erosion
        (M may exceed the GPU count). An explicit ``max_workers`` acts as a CAP.
        """
        import logging
        from packages.epicycle_scaledseg.utils.resources import get_allocated_cpus, available_ram_gb_and_source
        logger = logging.getLogger("epicycle.reconstruction")

        total_cpus = get_allocated_cpus()
        avail_ram, ram_src = available_ram_gb_and_source()
        peak = self._estimate_peak_per_tile(todo)   # a-priori model

        # Prefer a MEASURED per-tile peak over the model. A recorded peak is reused
        # only if its tile-size signature still matches (else it's stale -> recal).
        # If none is usable and there is >1 tile, reconstruct ONE tile in isolation
        # first (its worker records the true peak RSS), then size M from reality.
        measured = _read_calibration_peak(self.out_root, current_apriori_gb=peak)
        if measured is None and peak is not None and len(todo) > 1 and self.out_root:
            logger.info("Calibrating recon peak RAM on one tile before parallelizing...")
            self._run_one_tile_for_calibration(todo[0], n_gpus, log_dir)
            _stamp_calibration_apriori(self.out_root, peak)  # tile-size signature
            measured = _read_calibration_peak(self.out_root, current_apriori_gb=peak)
            todo = todo[1:]                          # the calibration tile is reconstructed
            if not todo:
                return self._collect_results(all_tiles)
        peak_src = "a-priori"
        if measured is not None:
            peak, peak_src = float(measured), "measured"

        if peak is None:
            # Couldn't size the per-tile peak (no readable .npy header) — run
            # conservatively (one at a time) unless the user pinned max_workers.
            logger.warning(
                "Could not estimate per-tile recon RAM (no readable xy_probs .npy "
                "header); running 1 tile at a time. Pass max_workers to override."
            )
            M = max(1, min(self.max_workers or 1, len(todo)))
        else:
            M = compute_recon_concurrency(total_cpus, avail_ram, peak, len(todo))
            if self.max_workers is not None:
                M = max(1, min(self.max_workers, M))

        n_cpu = max(1, total_cpus // M)        # threads/worker, pinned
        n_gpus_eff = max(1, n_gpus)            # 0-GPU: erosion CPU-falls-back; clamp the round-robin

        logger.info(
            f"Recon concurrency M={M} tile(s) "
            f"[cpu {total_cpus}//{RECON_EFFECTIVE_CORES_PER_TILE}="
            f"{total_cpus // RECON_EFFECTIVE_CORES_PER_TILE}, "
            f"ram {avail_ram:.0f}GB ({ram_src}), {peak_src} peak {(peak or 0):.1f}GB/tile], "
            f"{n_cpu} CPU thread(s)/worker, {n_gpus_eff} GPU(s) round-robin for erosion"
        )
        # Only warn about node-total over-reporting when actually under SLURM (a
        # shared allocation). On a personal/local box psutil.total IS the real RAM,
        # so the warning would just be noise there.
        if ram_src.startswith(("psutil", "proc", "unknown")) and os.environ.get("SLURM_JOB_ID"):
            logger.warning(
                f"Recon RAM source is '{ram_src}' — NOT a per-job SLURM/cgroup limit "
                f"inside a SLURM job; it over-reports a shared allocation. If M looks "
                f"too high, pin RECON_MAX_WORKERS or check your --mem / --mem-per-gpu."
            )

        kwargs = self._build_worker_kwargs()
        kwargs["n_cpu"] = n_cpu                 # override the (stale) n_gpus-derived value
        log_str = str(log_dir) if log_dir else "logs"

        tasks = []
        for i, td in enumerate(todo):
            gpu_id = i % n_gpus_eff
            seg_dir = str(self._seg_dir(td))
            logger.info(f"Queuing reconstruction for {td.name} on GPU {gpu_id} (seg_dir={seg_dir})")
            tasks.append((gpu_id, [seg_dir], kwargs, log_str))

        ctx = get_context("spawn")
        with ctx.Pool(processes=M) as pool:
            pool.starmap(_recon_gpu_worker, tasks)

        return self._collect_results(all_tiles)

    def _run_one_tile_for_calibration(self, td, n_gpus, log_dir):
        """Reconstruct ONE tile in isolation (all cores, single process) so its
        worker records the true single-tile peak RSS to the calibration sidecar.
        Used once before parallelizing, to size M from a MEASUREMENT not the model.
        """
        from packages.epicycle_scaledseg.utils.resources import get_allocated_cpus
        kwargs = self._build_worker_kwargs()
        kwargs["n_cpu"] = get_allocated_cpus()   # alone -> all cores
        seg_dir = str(self._seg_dir(td))
        log_str = str(log_dir) if log_dir else "logs"
        ctx = get_context("spawn")
        p = ctx.Process(
            target=_recon_gpu_worker,
            args=(0, [seg_dir], kwargs, log_str),   # gpu 0 (or CPU-fallback erosion)
        )
        p.start()
        p.join()

    def _estimate_peak_per_tile(self, todo, sample: int = 64) -> Optional[float]:
        """A-priori per-tile recon peak RAM (GB) — the MAX over a sample of up to
        ``sample`` tiles' xy-probs array HEADERs (Z,Y,X), no array load. Max (not
        first) so heterogeneous tiles (e.g. differing truncated-Z) size M for the
        BIGGEST tile, not an unrepresentative first one. Returns None if no header
        is readable (e.g. only compressed .npz present) -> caller runs
        conservatively. The first completed tile's measured RSS calibrates this
        (follow-up).
        """
        import numpy as np
        from packages.epicycle_scaledseg.utils.resources import estimate_tile_peak_ram_gb
        peaks = []
        for td in list(todo)[:sample]:
            seg_dir = self._seg_dir(td)
            for p in sorted(seg_dir.glob("*_xy_probs.npy")):
                try:
                    with open(p, "rb") as fh:
                        version = np.lib.format.read_magic(fh)
                        if version[0] >= 2:
                            shape = np.lib.format.read_array_header_2_0(fh)[0]
                        else:
                            shape = np.lib.format.read_array_header_1_0(fh)[0]
                    if len(shape) == 3:
                        peaks.append(estimate_tile_peak_ram_gb(shape[0], shape[1], shape[2]))
                        break  # one probs file per tile
                except Exception:
                    continue
        return max(peaks) if peaks else None

    def _collect_results(self, tile_dirs):
        """Check for output mask.tif to determine success/failure."""
        result = StageResult()
        for td in tile_dirs:
            out_dir = self._out_dir(td)
            bname = td.name
            if (out_dir / f"{bname}_mask.tif").exists():
                result.add_success(td.name)
            else:
                result.add_failure(td.name, "mask.tif not found after reconstruction")
        return result

    def _seg_dir(self, tile_dir: Path) -> Path:
        """Resolve the stage1 segmentation directory for a tile."""
        if self.seg_root:
            return Path(self.seg_root) / tile_dir.name
        return tile_dir.parent / "stage1_segmentation" / tile_dir.name

    def _out_dir(self, tile_dir: Path) -> Path:
        """Resolve the stage2 output directory for a tile."""
        if self.out_root:
            return Path(self.out_root) / tile_dir.name
        return tile_dir.parent / "stage2_reconstruction" / tile_dir.name

    def _build_worker_kwargs(self) -> dict:
        """Build kwargs dict to pass to the spawned worker."""
        return {
            "out_root": self.out_root,
            "reconstruct_params": getattr(self, "_resolved_reconstruct_params", self.reconstruct_params),
            "postprocess_params": getattr(self, "_resolved_postprocess_params", self.postprocess_params),
            "save_prob3d": self.save_prob3d,
            "save_flow3d": self.save_flow3d,
            "save_binary": self.save_binary,
            "binary_erode": self.binary_erode,
            "binary_kernel_size": self.binary_kernel_size,
            "binary_border_margin": self.binary_border_margin,
            "binary_fill_holes": self.binary_fill_holes,
            "cleanup_intermediates": self.cleanup_intermediates,
            "n_cpu": self._resolved_n_cpu,
            "skip_postprocess": True,
            "pre_smooth_flows": self.pre_smooth_flows,
            "pre_smooth_probs": self.pre_smooth_probs,
            "pre_smooth_sigma_perp": self.pre_smooth_sigma_perp,
            "pre_smooth_sigma_inplane": self.pre_smooth_sigma_inplane,
            "flow_mask_threshold": self.flow_mask_threshold,
            "flow_soft_weighting": self.flow_soft_weighting,
        }


# ---------------------------------------------------------------------------
# Worker (runs in a spawned subprocess)
# ---------------------------------------------------------------------------

def _recon_gpu_worker(
    gpu_id: int,
    tile_paths: List[str],
    kwargs: dict,
    log_dir: str,
):
    """
    Per-GPU reconstruction worker.

    Processes assigned tiles sequentially on the assigned GPU.
    Runs in a spawned subprocess.

    IMPORTANT: CUDA_VISIBLE_DEVICES is NOT set here. The aggregation
    step (USegment3D) is CPU-only and can benefit from full node RAM
    without being pinned to a per-GPU SLURM cgroup. GPU pinning is
    deferred to the erosion step inside _reconstruct_single_tile.
    """
    n_cpu = kwargs.get("n_cpu", 8)
    os.environ["OMP_NUM_THREADS"] = str(n_cpu)
    os.environ["MKL_NUM_THREADS"] = str(n_cpu)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_cpu)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_cpu)
    # u-Segment3D's inner ThreadPool reads cpu_count() (ignores the env pins above),
    # so cap it to the allocated thread budget — prevents M-way oversubscription and
    # keeps it inside a SLURM CPU allocation. Must run before u-Segment3D imports.
    _pin_inner_cpu_count(n_cpu)

    from packages.epicycle_scaledseg.utils.logging_utils import setup_logger

    logger = setup_logger(gpu_id, log_dir)
    logger.info(f"=== RECON GPU {gpu_id} WORKER STARTED ({len(tile_paths)} tiles) ===")

    for i, tile_path in enumerate(tile_paths, 1):
        tile_name = Path(tile_path).name
        logger.info(f"Tile {i}/{len(tile_paths)}: {tile_name}")
        t0 = time.time()

        try:
            _reconstruct_single_tile(
                seg_dir=tile_path,
                gpu_id=gpu_id,
                logger=logger,
                **kwargs,
            )
            logger.info(f"[{tile_name}] SUCCESS ({time.time() - t0:.1f}s)")
        except Exception as e:
            logger.error(f"[{tile_name}] FAILED: {e}\n{traceback.format_exc()}")

        gc.collect()

    logger.info(f"=== RECON GPU {gpu_id} WORKER FINISHED ===")

    # Record this process's PEAK RSS -> the workspace recon-RAM calibration. With
    # one tile per worker (parallel mode + the calibration run) this is the true
    # single-tile peak, so the next batch/run sizes M from a MEASUREMENT instead of
    # the a-priori model. Best-effort; never fail the worker over it.
    try:
        _write_calibration_peak(kwargs.get("out_root"), _peak_rss_gb())
    except Exception:
        pass


# def _reconstruct_single_tile(
#     seg_dir: str,
#     out_root: Optional[str],
#     reconstruct_params: Optional[dict],
#     postprocess_params: Optional[dict],
#     save_prob3d: bool,
#     save_flow3d: bool,
#     save_binary: bool,
#     binary_erode: bool,
#     binary_kernel_size: int,
#     binary_border_margin: int,
#     binary_fill_holes: bool,
#     cleanup_intermediates: bool,
#     n_cpu: int,
#     logger,
#     gpu_id: int = 0,
# ):
#     """
#     Reconstruct one tile from saved 2D segmentation outputs.

#     Mirrors ``reconstruct_from_saved_raw_direct`` from v3_2 but
#     structured for the OOP pipeline.
#     """
#     import tifffile
#     import segment3D.usegment3d as uSegment3D
#     from scipy.ndimage import binary_fill_holes as nd_binary_fill_holes

#     from packages.epicycle_scaledseg.config.defaults import (
#         get_default_reconstruct_params,
#         get_default_postprocess_segmentation_params,
#     )

#     seg_dir = Path(seg_dir)
#     bname = seg_dir.name

#     # Load tile config
#     cfg_files = list(seg_dir.glob("*_config.json"))
#     if not cfg_files:
#         raise FileNotFoundError(f"No config.json in {seg_dir}")
#     cfg_path = cfg_files[0]
#     with open(cfg_path) as f:
#         meta = json.load(f)

#     tile_name = meta.get("tile_name", bname)
#     z0, z1 = meta["truncated_range"]
#     origZ = int(meta["original_dimensions"]["z"])

#     # Pre-flight memory check: read .npy header to get tile dims without loading
#     from packages.epicycle_scaledseg.utils.resources import check_reconstruction_memory
#     probe_path = pick_array_path(seg_dir, f"{tile_name}_xy_probs")
#     if probe_path:
#         try:
#             with open(probe_path, "rb") as fh:
#                 version = np.lib.format.read_magic(fh)
#                 if version[0] >= 2:
#                     header = np.lib.format.read_array_header_2_0(fh)
#                 else:
#                     header = np.lib.format.read_array_header_1_0(fh)
#             tile_shape = header[0]  # (Z, Y, X)
#             if len(tile_shape) == 3:
#                 check_reconstruction_memory(
#                     z_slices=tile_shape[0],
#                     height=tile_shape[1],
#                     width=tile_shape[2],
#                     n_concurrent=1,
#                     logger=logger,
#                 )
#         except Exception as e:
#             logger.warning(f"[{bname}] skipping pre-flight memory check: {e}")

#     logger.info(f"[{bname}] loading flows and probabilities...")

#     # Load all 6 arrays
#     t_load = time.time()
#     flows_xy = load_array(pick_array_path(seg_dir, f"{tile_name}_xy_flows"))
#     flows_xz = load_array(pick_array_path(seg_dir, f"{tile_name}_xz_flows"))
#     flows_yz = load_array(pick_array_path(seg_dir, f"{tile_name}_yz_flows"))
#     probs_xy = load_array(pick_array_path(seg_dir, f"{tile_name}_xy_probs"))
#     probs_xz = load_array(pick_array_path(seg_dir, f"{tile_name}_xz_probs"))
#     probs_yz = load_array(pick_array_path(seg_dir, f"{tile_name}_yz_probs"))
#     load_time = time.time() - t_load

#     logger.info(
#         f"[{bname}] loaded ({load_time:.1f}s): "
#         f"Fxy{flows_xy.shape} Fxz{flows_xz.shape} Fyz{flows_yz.shape} | "
#         f"Pxy{probs_xy.shape} Pxz{probs_xz.shape} Pyz{probs_yz.shape}"
#     )

#     # Sanity checks
#     assert flows_xy.ndim == 4 and flows_xy.shape[0] == 2
#     assert flows_xz.ndim == 4 and flows_xz.shape[0] == 2
#     assert flows_yz.ndim == 4 and flows_yz.shape[0] == 2
#     assert probs_xy.ndim == 3 and probs_xz.ndim == 3 and probs_yz.ndim == 3

#     # Reconstruct: aggregate 2D → 3D
#     if reconstruct_params is None:
#         reconstruct_params = get_default_reconstruct_params(n_cpu=n_cpu)

#     logger.info(f"[{bname}] aggregating 2D → 3D (direct method)...")
#     t_agg = time.time()
#     seg3D, (prob3D, grad3D) = uSegment3D.aggregate_2D_to_3D_segmentation_direct_method(
#         probs=[probs_xy, probs_xz, probs_yz],
#         gradients=[flows_xy, flows_xz, flows_yz],
#         params=reconstruct_params,
#     )
#     agg_time = time.time() - t_agg
#     num_cells = len(np.unique(seg3D)) - 1
#     logger.info(f"[{bname}] aggregation complete ({agg_time:.1f}s) — {num_cells} cells")

#     # Post-process (flow consistency)
#     if postprocess_params is None:
#         postprocess_params = get_default_postprocess_segmentation_params(n_cpu=n_cpu)

#     logger.info(f"[{bname}] postprocessing (flow consistency)...")
#     t_post = time.time()
#     seg3D_final, _ = uSegment3D.postprocess_3D_cell_segmentation(
#         seg3D,
#         aggregation_params=reconstruct_params,
#         postprocess_params=postprocess_params,
#         cell_gradients=grad3D,
#     )
#     post_time = time.time() - t_post
#     num_cells_final = len(np.unique(seg3D_final)) - 1
#     logger.info(f"[{bname}] postprocessing complete ({post_time:.1f}s) — {num_cells_final} cells retained")

#     # Free intermediate arrays
#     del flows_xy, flows_xz, flows_yz
#     del probs_xy, probs_xz, probs_yz
#     del seg3D
#     if not save_flow3d:
#         del grad3D
#     gc.collect()

#     # Binary extraction
#     binary_full = None
#     if save_binary:
#         labels_for_binary = seg3D_final
#         if binary_erode:
#             # Pin to GPU only for erosion — deferred to avoid SLURM cgroup
#             # constraints during the CPU-only aggregation phase above.
#             os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
#             import torch
#             from packages.epicycle_scaledseg.utils.erosion import erode_instances, protect_border_margin

#             logger.info(f"[{bname}] eroding {num_cells_final} instances (GPU {gpu_id}, kernel={binary_kernel_size})...")
#             t_erode = time.time()
#             labels_eroded = erode_instances(
#                 labels_for_binary,
#                 kernel_size=binary_kernel_size,
#             )
#             logger.info(f"[{bname}] erosion complete ({time.time() - t_erode:.1f}s)")

#             if binary_border_margin and binary_border_margin > 0:
#                 labels_eroded = protect_border_margin(
#                     labels_eroded, labels_for_binary, binary_border_margin
#                 )
#             labels_for_binary = labels_eroded

#             torch.cuda.empty_cache()

#         binary = labels_for_binary > 0
#         if binary_fill_holes:
#             logger.info(f"[{bname}] filling holes in binary mask...")
#             binary = nd_binary_fill_holes(binary, structure=np.ones((3, 3, 3), dtype=bool))
#         binary_full = recover_original_dimensions(binary, (z0, z1), origZ)

#     gc.collect()

#     # Output directory
#     if out_root:
#         out_dir = Path(out_root) / bname
#     else:
#         out_dir = seg_dir.parent.parent / "stage2_reconstruction" / bname
#     ensure_dir(out_dir)

#     logger.info(f"[{bname}] recovering full Z dimensions and writing outputs...")

#     # Save mask
#     mask_full = recover_original_dimensions(seg3D_final, (z0, z1), origZ)
#     tifffile.imwrite(
#         str(out_dir / f"{bname}_mask.tif"),
#         mask_full.astype(np.uint32),
#         bigtiff=True,
#     )

#     # Save prob3D
#     if save_prob3d:
#         prob_full = recover_original_dimensions(prob3D, (z0, z1), origZ)
#         np.save(out_dir / f"{bname}_probabilities.npy", prob_full.astype(np.float16))
#         del prob_full

#     # Save binary
#     if save_binary and binary_full is not None:
#         np.save(out_dir / f"{bname}_binary.npy", binary_full.astype(np.uint8))
#         update_tile_config(str(cfg_path), {
#             "binary": {
#                 "saved": True,
#                 "erode": bool(binary_erode),
#                 "kernel_size": int(binary_kernel_size),
#                 "border_margin": int(binary_border_margin),
#                 "fill_holes": bool(binary_fill_holes),
#                 "dtype": "uint8",
#             }
#         })

#     # Save flow3D
#     if save_flow3d:
#         grad_full = recover_original_dimensions(grad3D, (z0, z1), origZ)
#         np.save(out_dir / f"{bname}_flows3D.npy", grad_full.astype(np.float16))

#     # Update config
#     update_tile_config(str(cfg_path), {
#         "reconstruction": {
#             "method": "direct",
#             "timestamp": now_iso(),
#             "num_cells_raw": num_cells,
#             "num_cells_final": num_cells_final,
#         }
#     })

#     # Cleanup intermediates
#     if cleanup_intermediates:
#         for v in ("xy", "xz", "yz"):
#             for stem in (f"{tile_name}_{v}_flows", f"{tile_name}_{v}_probs"):
#                 for ext in (".npz", ".npy"):
#                     p = seg_dir / f"{stem}{ext}"
#                     if p.exists():
#                         p.unlink()

#     # Final cleanup
#     del seg3D_final, mask_full
#     if save_binary:
#         del binary_full
#     gc.collect()

#     # Clear GPU cache if torch was imported (erosion was used)
#     try:
#         import torch
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#     except ImportError:
#         pass

#     logger.info(f"[{bname}] reconstruction complete → {out_dir}")

def _reconstruct_single_tile(
    seg_dir: str,
    out_root: Optional[str],
    reconstruct_params: Optional[dict],
    postprocess_params: Optional[dict],
    save_prob3d: bool,
    save_flow3d: bool,
    save_binary: bool,
    binary_erode: bool,
    binary_kernel_size: int,
    binary_border_margin: int,
    binary_fill_holes: bool,
    cleanup_intermediates: bool,
    n_cpu: int,
    logger,
    gpu_id: int = 0,
    skip_postprocess: bool = True,  # NEW FLAG
    pre_smooth_flows: bool = False,
    pre_smooth_probs: bool = False,
    pre_smooth_sigma_perp: float = 3.0,
    pre_smooth_sigma_inplane: float = 1.0,
    flow_mask_threshold: float = 0.2,
    flow_soft_weighting: bool = True,
):
    """
    Reconstruct one tile from saved 2D segmentation outputs.

    Mirrors ``reconstruct_from_saved_raw_direct`` from v3_2 but
    structured for the OOP pipeline.
    """
    import tifffile
    import segment3D.usegment3d as uSegment3D
    from scipy.ndimage import binary_fill_holes as nd_binary_fill_holes

    from packages.epicycle_scaledseg.config.defaults import (
        get_default_reconstruct_params,
        get_default_postprocess_segmentation_params,
    )

    seg_dir = Path(seg_dir)
    bname = seg_dir.name

    # Load tile config
    cfg_files = list(seg_dir.glob("*_config.json"))
    if not cfg_files:
        raise FileNotFoundError(f"No config.json in {seg_dir}")
    cfg_path = cfg_files[0]
    with open(cfg_path) as f:
        meta = json.load(f)

    tile_name = meta.get("tile_name", bname)
    z0, z1 = meta["truncated_range"]
    origZ = int(meta["original_dimensions"]["z"])

    # Pre-flight memory check
    from packages.epicycle_scaledseg.utils.resources import check_reconstruction_memory
    probe_path = pick_array_path(seg_dir, f"{tile_name}_xy_probs")
    if probe_path:
        try:
            with open(probe_path, "rb") as fh:
                version = np.lib.format.read_magic(fh)
                if version[0] >= 2:
                    header = np.lib.format.read_array_header_2_0(fh)
                else:
                    header = np.lib.format.read_array_header_1_0(fh)
            tile_shape = header[0]
            if len(tile_shape) == 3:
                check_reconstruction_memory(
                    z_slices=tile_shape[0],
                    height=tile_shape[1],
                    width=tile_shape[2],
                    n_concurrent=1,
                    logger=logger,
                )
        except Exception as e:
            logger.warning(f"[{bname}] skipping pre-flight memory check: {e}")

    logger.info(f"[{bname}] loading flows and probabilities...")

    # Load all 6 arrays
    t_load = time.time()
    flows_xy = load_array(pick_array_path(seg_dir, f"{tile_name}_xy_flows"))
    flows_xz = load_array(pick_array_path(seg_dir, f"{tile_name}_xz_flows"))
    flows_yz = load_array(pick_array_path(seg_dir, f"{tile_name}_yz_flows"))
    probs_xy = load_array(pick_array_path(seg_dir, f"{tile_name}_xy_probs"))
    probs_xz = load_array(pick_array_path(seg_dir, f"{tile_name}_xz_probs"))
    probs_yz = load_array(pick_array_path(seg_dir, f"{tile_name}_yz_probs"))
    load_time = time.time() - t_load

    logger.info(
        f"[{bname}] loaded ({load_time:.1f}s): "
        f"Fxy{flows_xy.shape} Fxz{flows_xz.shape} Fyz{flows_yz.shape} | "
        f"Pxy{probs_xy.shape} Pxz{probs_xz.shape} Pyz{probs_yz.shape}"
    )

    # Sanity checks
    assert flows_xy.ndim == 4 and flows_xy.shape[0] == 2
    assert flows_xz.ndim == 4 and flows_xz.shape[0] == 2
    assert flows_yz.ndim == 4 and flows_yz.shape[0] == 2
    assert probs_xy.ndim == 3 and probs_xz.ndim == 3 and probs_yz.ndim == 3

    # --- Optional anisotropic pre-smoothing ---
    # Smooth per-view arrays before USegment3D aggregation to suppress
    # slice-to-slice discontinuities from independent 2D CellPose predictions.
    # High sigma along each view's perpendicular axis, low sigma in-plane.
    if pre_smooth_probs or pre_smooth_flows:
        from packages.epicycle_scaledseg.utils.array_ops import (
            per_view_sigmas,
            smooth_flows_3d,
            smooth_probs_3d,
        )

        s_perp = pre_smooth_sigma_perp
        s_in = pre_smooth_sigma_inplane
        sigma_xy, sigma_xz, sigma_yz = per_view_sigmas(s_perp, s_in)

        if pre_smooth_probs:
            logger.info(
                f"[{bname}] anisotropic prob pre-smoothing "
                f"(perp={s_perp}, in-plane={s_in})"
            )
            probs_xy = smooth_probs_3d(probs_xy, sigma=sigma_xy)
            probs_xz = smooth_probs_3d(probs_xz, sigma=sigma_xz)
            probs_yz = smooth_probs_3d(probs_yz, sigma=sigma_yz)

        if pre_smooth_flows:
            logger.info(
                f"[{bname}] anisotropic flow pre-smoothing "
                f"(perp={s_perp}, in-plane={s_in}, "
                f"mask_thresh={flow_mask_threshold}, "
                f"soft_weight={flow_soft_weighting})"
            )
            flows_xy = smooth_flows_3d(
                flows_xy, probs=probs_xy, sigma=sigma_xy,
                mask_threshold=flow_mask_threshold,
                soft_weighting=flow_soft_weighting,
            )
            flows_xz = smooth_flows_3d(
                flows_xz, probs=probs_xz, sigma=sigma_xz,
                mask_threshold=flow_mask_threshold,
                soft_weighting=flow_soft_weighting,
            )
            flows_yz = smooth_flows_3d(
                flows_yz, probs=probs_yz, sigma=sigma_yz,
                mask_threshold=flow_mask_threshold,
                soft_weighting=flow_soft_weighting,
            )

    # Reconstruct: aggregate 2D → 3D
    if reconstruct_params is None:
        reconstruct_params = get_default_reconstruct_params(n_cpu=n_cpu)

    logger.info(f"[{bname}] aggregating 2D -> 3D (direct method)...")
    t_agg = time.time()
    seg3D, (prob3D, grad3D) = uSegment3D.aggregate_2D_to_3D_segmentation_direct_method(
        probs=[probs_xy, probs_xz, probs_yz],
        gradients=[flows_xy, flows_xz, flows_yz],
        params=reconstruct_params,
    )
    agg_time = time.time() - t_agg
    num_cells = len(np.unique(seg3D)) - 1
    logger.info(f"[{bname}] aggregation complete ({agg_time:.1f}s) — {num_cells} cells")

    # --------------------------------------------------
    # OPTIONAL POST-PROCESSING (ONLY CHANGE)
    # --------------------------------------------------
    if skip_postprocess:
        seg3D_final = seg3D
        num_cells_final = num_cells
        logger.info(
            f"[{bname}] skipping postprocessing — using raw segmentation "
            f"({num_cells_final} cells)"
        )
    else:
        if postprocess_params is None:
            postprocess_params = get_default_postprocess_segmentation_params(n_cpu=n_cpu)

        logger.info(f"[{bname}] postprocessing (flow consistency)...")
        t_post = time.time()
        seg3D_final, _ = uSegment3D.postprocess_3D_cell_segmentation(
            seg3D,
            aggregation_params=reconstruct_params,
            postprocess_params=postprocess_params,
            cell_gradients=grad3D,
        )
        post_time = time.time() - t_post
        num_cells_final = len(np.unique(seg3D_final)) - 1
        logger.info(
            f"[{bname}] postprocessing complete ({post_time:.1f}s) — "
            f"{num_cells_final} cells retained"
        )

    # Free intermediate arrays
    del flows_xy, flows_xz, flows_yz
    del probs_xy, probs_xz, probs_yz
    del seg3D
    if not save_flow3d:
        del grad3D
    gc.collect()

    # Binary extraction
    binary_full = None
    if save_binary:
        labels_for_binary = seg3D_final
        if binary_erode:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            import torch
            from packages.epicycle_scaledseg.utils.erosion import erode_instances, protect_border_margin

            logger.info(
                f"[{bname}] eroding {num_cells_final} instances "
                f"(GPU {gpu_id}, kernel={binary_kernel_size})..."
            )
            labels_eroded = erode_instances(
                labels_for_binary,
                kernel_size=binary_kernel_size,
            )

            if binary_border_margin and binary_border_margin > 0:
                labels_eroded = protect_border_margin(
                    labels_eroded, labels_for_binary, binary_border_margin
                )
            labels_for_binary = labels_eroded
            torch.cuda.empty_cache()

        binary = labels_for_binary > 0
        if binary_fill_holes:
            logger.info(f"[{bname}] filling holes in binary mask...")
            binary = nd_binary_fill_holes(binary, structure=np.ones((3, 3, 3), dtype=bool))
        binary_full = recover_original_dimensions(binary, (z0, z1), origZ)

    gc.collect()

    # Output directory
    if out_root:
        out_dir = Path(out_root) / bname
    else:
        out_dir = seg_dir.parent.parent / "stage2_reconstruction" / bname
    ensure_dir(out_dir)

    logger.info(f"[{bname}] recovering full Z dimensions and writing outputs...")

    # Save mask
    mask_full = recover_original_dimensions(seg3D_final, (z0, z1), origZ)
    tifffile.imwrite(
        str(out_dir / f"{bname}_mask.tif"),
        mask_full.astype(np.uint32),
        bigtiff=True,
    )

    # Save prob3D
    if save_prob3d:
        prob_full = recover_original_dimensions(prob3D, (z0, z1), origZ)
        np.save(out_dir / f"{bname}_probabilities.npy", prob_full.astype(np.float16))
        del prob_full

    # Save binary
    if save_binary and binary_full is not None:
        np.save(out_dir / f"{bname}_binary.npy", binary_full.astype(np.uint8))
        update_tile_config(str(cfg_path), {
            "binary": {
                "saved": True,
                "erode": bool(binary_erode),
                "kernel_size": int(binary_kernel_size),
                "border_margin": int(binary_border_margin),
                "fill_holes": bool(binary_fill_holes),
                "dtype": "uint8",
            }
        })

    # Save flow3D
    if save_flow3d:
        grad_full = recover_original_dimensions(grad3D, (z0, z1), origZ)
        np.save(out_dir / f"{bname}_flows3D.npy", grad_full.astype(np.float16))

    # Update config
    update_tile_config(str(cfg_path), {
        "reconstruction": {
            "method": "direct",
            "timestamp": now_iso(),
            "num_cells_raw": num_cells,
            "num_cells_final": num_cells_final,
            "postprocess_skipped": bool(skip_postprocess),
        }
    })

    # Cleanup intermediates
    if cleanup_intermediates:
        for v in ("xy", "xz", "yz"):
            for stem in (f"{tile_name}_{v}_flows", f"{tile_name}_{v}_probs"):
                for ext in (".npz", ".npy"):
                    p = seg_dir / f"{stem}{ext}"
                    if p.exists():
                        p.unlink()

    # Final cleanup
    del seg3D_final, mask_full
    if save_binary:
        del binary_full
    gc.collect()

    # Clear GPU cache if torch was imported
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    logger.info(f"[{bname}] reconstruction complete -> {out_dir}")
