"""Step-1 large-data logging entry points (forwarders to the session log).

The large-data path used to keep its own console copy and its own file.  There
is now one session log for the whole plugin (see ``utils.crash_logger``): it
holds the console, the crash dump and the breadcrumbs, so a crash stack never
lands in a different folder than the lines that led to it.  These functions
keep the old names and forward to it.  Every entry point is best-effort and
never raises into the caller.
"""

from __future__ import annotations

from pathlib import Path

from ..utils import crash_logger


def install(log_dir: Path | None = None) -> Path | None:
    """Arm the session log (idempotent).  Returns its path, or ``None``."""
    try:
        return crash_logger.install_crash_logging(log_dir)
    except Exception:  # noqa: BLE001
        return None


def breadcrumb(msg: str) -> None:
    """Write a timestamped line to the session file (best-effort)."""
    try:
        crash_logger.breadcrumb(msg)
    except Exception:  # noqa: BLE001
        pass


def log_path() -> Path | None:
    """The current session log path, or ``None`` if logging is not armed."""
    return crash_logger.log_path()
