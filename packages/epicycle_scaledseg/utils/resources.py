"""
Resource detection utilities for SLURM HPC environments.

Detects allocated CPUs, GPUs, and memory that respect SLURM cgroup
limits rather than reporting the full node specification.
"""

import os
import logging


def get_allocated_cpus() -> int:
    """
    Return the number of CPU cores allocated to the current process.

    Detection order (most specific → most general):

    1. ``SLURM_CPUS_PER_TASK`` — set when ``--cpus-per-task`` is used
    2. ``SLURM_JOB_CPUS_PER_NODE`` — total CPUs allocated on this node
    3. ``os.sched_getaffinity(0)`` — respects cgroup CPU limits (Linux only)
    4. ``os.cpu_count()`` — fallback, returns physical cores (may over-count)
    """
    # SLURM environment variables (most reliable on HPC)
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_JOB_CPUS_PER_NODE"):
        val = os.environ.get(var)
        if val is not None:
            try:
                return int(val)
            except ValueError:
                continue

    # cgroup-aware affinity (Linux)
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        # Windows or restricted environments
        pass

    # Last resort
    return os.cpu_count() or 1


def get_allocated_gpus() -> int:
    """
    Return the number of GPUs allocated to the current process.

    Detection order:

    1. ``SLURM_GPUS_ON_NODE`` — set by ``--gpus-per-node``
    2. ``SLURM_GPUS_PER_TASK`` — set by ``--gpus-per-task`` (gres)
    3. ``CUDA_VISIBLE_DEVICES`` — comma-separated GPU IDs (honored even when
       empty / ``-1``, which mean *no* visible GPUs)
    4. ``torch.cuda.device_count()`` if torch is importable
    5. ``0`` — genuinely no GPU.

    Returns ``0`` (not 1) when no GPU is detected, so a CPU-only host is reported
    honestly. Callers route ``0`` to a CPU/in-process path rather than driving a
    ``gpu=True`` code path that would crash with no device. (A real 1-GPU box
    still reports 1 via ``torch.cuda.device_count()``.)
    """
    for var in ("SLURM_GPUS_ON_NODE", "SLURM_GPUS_PER_TASK"):
        val = os.environ.get(var)
        if val is not None:
            try:
                return int(val)
            except ValueError:
                continue

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        # Explicitly set — honor it verbatim. ``""`` and ``-1`` are the canonical
        # "hide all GPUs" values and must count as 0, not 1 (a bare ``len(split)``
        # returns 1 for both — the bug this guards).
        ids = [x for x in cvd.split(",") if x.strip() not in ("", "-1")]
        return len(ids)

    try:
        import torch
        return int(torch.cuda.device_count())  # 0 if no CUDA — do NOT coerce to 1
    except ImportError:
        return 0


# ---------------------------------------------------------------------------
# Memory estimation
# ---------------------------------------------------------------------------

# Multiplier for USegment3D internal overhead beyond the 6 input arrays
# and 3 output arrays. The direct-method peak is dominated by transients the
# raw 9-array count misses: the 0.25-overlap *padded* gradient field
# (~1.95x a 3-vector float32 volume) carried through gradient descent, plus the
# float64 vote-accumulator grid AND its smoothed copy in the final
# connected-components pass. A source-level audit of u-Segment3D showed the old
# 1.3x was optimistic; 1.5x keeps the RAM-fit math (tile sizing + recon
# concurrency divide by this) on the safe side. Recon concurrency is additionally
# calibrated from the FIRST tile's measured peak RSS, so this is the a-priori
# estimate, not the final word.
_USEGMENT3D_OVERHEAD = 1.5


def estimate_tile_peak_ram_gb(z_slices: int, height: int, width: int) -> float:
    """Estimate peak system RAM (GB) for reconstructing one tile.

    The dominant cost is having all 6 input arrays (3 flows + 3 probs)
    plus 3 output arrays (seg3D, prob3D, grad3D) resident simultaneously
    during ``aggregate_2D_to_3D_segmentation_direct_method``.

    Parameters
    ----------
    z_slices : number of Z slices in the *truncated* range (z1 - z0)
    height, width : spatial dimensions of the tile in pixels

    Returns
    -------
    Estimated peak RAM in gigabytes (float).
    """
    voxels = z_slices * height * width
    bytes_per_float32 = 4

    # 3 flow arrays: shape (2, Z, Y, X) float32
    flows = 3 * 2 * voxels * bytes_per_float32
    # 3 prob arrays: shape (Z, Y, X) float32
    probs = 3 * voxels * bytes_per_float32
    # Outputs: seg3D (int32) + prob3D (float32) + grad3D (3, Z, Y, X) float32
    outputs = voxels * 4 + voxels * 4 + 3 * voxels * 4

    total_bytes = (flows + probs + outputs) * _USEGMENT3D_OVERHEAD
    return total_bytes / (1024 ** 3)


def _env_int(name: str):
    """``int(os.environ[name])`` or None (tolerates unset / non-int like '32(x2)')."""
    v = os.environ.get(name)
    if not v:
        return None
    try:
        return int(v)
    except ValueError:
        return None


def available_ram_gb_and_source() -> "tuple":
    """``(GB, source)`` — the RAM this job may use, and WHERE the number came from.

    Detection order (most job-specific -> least). Crucially this now covers the
    SLURM ``--mem-per-cpu`` / ``--mem-per-gpu`` request styles (common on GPU
    clusters, e.g. UW Hyak), which leave ``SLURM_MEM_PER_NODE`` *unset* — without
    these a per-GPU-memory job would fall through to node-total and OVER-report its
    share, sizing recon concurrency too high -> OOM:

      1. ``SLURM_MEM_PER_NODE``                              (--mem, MB)
      2. ``SLURM_MEM_PER_CPU`` x cpus                        (--mem-per-cpu, MB)
      3. ``SLURM_MEM_PER_GPU`` x gpus                        (--mem-per-gpu, MB)
      4. cgroup v2 / v1 limit                               (the enforced per-job cap)
      5. ``psutil`` / ``/proc/meminfo`` total               (NODE total — over-reports
         a shared allocation; last resort)
      6. ``0.0`` unknown                                    (callers treat as "be conservative")

    The ``source`` string is logged at recon time so M-sizing is auditable on any
    cluster.
    """
    # 1) --mem (node memory), MB
    mpn = _env_int("SLURM_MEM_PER_NODE")
    if mpn is not None:
        return mpn / 1024.0, "SLURM_MEM_PER_NODE"

    # 2) --mem-per-cpu * allocated cpus, MB. (`if mpc:` not `is not None` so a
    # degenerate "0" falls through to cgroup/node-total + the warning, instead of
    # silently returning 0 GB under a SLURM source label.)
    mpc = _env_int("SLURM_MEM_PER_CPU")
    if mpc:
        for cvar in ("SLURM_CPUS_PER_TASK", "SLURM_JOB_CPUS_PER_NODE"):
            nc = _env_int(cvar)
            if nc:
                return mpc * nc / 1024.0, f"SLURM_MEM_PER_CPU*{cvar}"

    # 3) --mem-per-gpu * allocated gpus, MB
    mpg = _env_int("SLURM_MEM_PER_GPU")
    if mpg:
        for gvar in ("SLURM_GPUS_ON_NODE", "SLURM_GPUS_PER_TASK"):
            ng = _env_int(gvar)
            if ng:
                return mpg * ng / 1024.0, f"SLURM_MEM_PER_GPU*{gvar}"

    # 4a) cgroup v2 enforced limit
    try:
        with open("/sys/fs/cgroup/memory.max") as f:
            val = f.read().strip()
            if val != "max":
                return int(val) / (1024 ** 3), "cgroup.v2"
    except (FileNotFoundError, OSError, ValueError):
        pass

    # 4b) cgroup v1 enforced limit
    try:
        with open("/sys/fs/cgroup/memory/memory.limit_in_bytes") as f:
            val = int(f.read().strip())
            if val < 2 ** 62:  # a very large value means "no limit"
                return val / (1024 ** 3), "cgroup.v1"
    except (FileNotFoundError, OSError, ValueError):
        pass

    # 5) NODE total (over-reports a shared/partial allocation — last resort)
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3), "psutil.total(NODE-may-over-report)"
    except ImportError:
        pass

    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / (1024 ** 2), "proc.meminfo(NODE-may-over-report)"
    except (FileNotFoundError, OSError):
        pass

    return 0.0, "unknown"


def get_available_ram_gb() -> float:
    """RAM (GB) this job may use — SLURM/cgroup-aware. See
    :func:`available_ram_gb_and_source` for the detection order + the source string."""
    return available_ram_gb_and_source()[0]


def check_reconstruction_memory(
    z_slices: int,
    height: int,
    width: int,
    n_concurrent: int = 1,
    logger=None,
) -> bool:
    """Pre-flight check: will reconstruction fit in available RAM?

    Logs a warning if estimated peak exceeds available RAM, and
    suggests how many GPUs to allocate (assuming 240 GB per GPU).

    Returns True if memory looks sufficient, False otherwise.
    """
    log = logger or logging.getLogger("epicycle.resources")

    peak_per_tile = estimate_tile_peak_ram_gb(z_slices, height, width)
    total_peak = peak_per_tile * n_concurrent
    available = get_available_ram_gb()

    log.info(
        f"Memory estimate: {peak_per_tile:.1f} GB/tile × {n_concurrent} concurrent "
        f"= {total_peak:.1f} GB peak | available: {available:.1f} GB"
    )

    if available > 0 and total_peak > available * 0.9:
        gpus_needed = max(1, int(total_peak / 240) + 1)
        log.warning(
            f"MEMORY WARNING: estimated peak {total_peak:.1f} GB exceeds "
            f"~90% of available RAM ({available:.1f} GB). "
            f"Consider allocating at least {gpus_needed} GPUs "
            f"(~{gpus_needed * 240} GB) or reducing --recon-max-workers."
        )
        return False

    return True
