"""How many threads a CPU-bound pool may use in this process.

One rule for every annotator pool that splits work over threads (Add's
smoothing, the ridge wall pictures): the cores this PROCESS may run on,
minus two (the window's main thread and the rest of the work keep a core
each), capped, and at least one.  On a cluster node or in a container the
process is often allowed fewer cores than the machine has; the CPU affinity
says how many, where the platform reports one (Linux).  Elsewhere the
machine's count is used, and an unknown count means one worker.
"""

from __future__ import annotations

import os

#: The default cap.  Eight measured best for Add's smoothing on a 20-core
#: machine (18 threads were slower).
POOL_MAX_WORKERS = 8


def usable_cores():
    """Cores this process may run on, or ``None`` when unknown."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count()


def pool_worker_count(cap: int = POOL_MAX_WORKERS) -> int:
    """``min(cap, usable_cores() - 2)``, at least 1 (1 when unknown)."""
    n = usable_cores()
    if not n:
        return 1
    return int(max(1, min(int(cap), int(n) - 2)))
