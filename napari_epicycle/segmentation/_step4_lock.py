"""The Step 4 lock flags.  Qt-free.

Brick B0 adds the flags and their thread-finished backstops; brick B2 adds
``_refresh_step4_lock``'s greying, the owner-keyed step-nav lock, the lock
lines and ``lock_reason``'s sentences.
"""
from __future__ import annotations

from typing import Any, Optional

#: Each locks the Step 4 surface AND the step nav while set.
STEP4_BLOCK_FLAGS = (
    "_predict_running", "_reconstruct_running", "_load_in_flight",
    "_preview_in_flight", "_measure_running", "_smooth_running",
    "_save_in_flight", "_export_running", "_mesh_running",
)

#: A Workbench map: greys Compute / Cluster only, never the nav.
STEP4_MAP_FLAG = "_eda_running"

ALL_FLAGS = STEP4_BLOCK_FLAGS + (STEP4_MAP_FLAG,)


def running_flags(obj: Any) -> tuple:
    """The block flags set on *obj*, in tuple order."""
    return tuple(f for f in STEP4_BLOCK_FLAGS if getattr(obj, f, False))


def is_locked(obj: Any) -> bool:
    return bool(running_flags(obj))


def lock_reason(obj: Any) -> Optional[str]:
    """The first set flag's name (B2 maps it to its ``STEP4_LOCK_*`` sentence)."""
    flags = running_flags(obj)
    return flags[0] if flags else None
