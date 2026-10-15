"""Session error/crash logging for the Step-1 large-data (progressive-3D) path.

WHY THIS EXISTS
---------------
The vendored progressive engine hits GL errors during the 3D<->2D lifecycle —
e.g. ``GLError('invalid value')`` from an empty (0-size) texture upload, or an
``ibuf.activate()`` on a torn-down node.  These are raised INSIDE vispy's draw
callback, which **catches** them (`_handle_exception`), logs
``WARNING: Error drawing visual`` and a large traceback, and then **continues**.
So they are *not* fatal crashes — ``faulthandler`` and ``sys.excepthook`` never
see them — and the traceback (thousands of lines of GLIR command dump) scrolls
off the console before it can be copied.

This funnels the FULL console — ``stdout`` + ``stderr`` (the vispy warnings and
their tracebacks, plus the ``[PROG]`` breadcrumbs) — to a per-session file under
``~/.epicycle/step1_largedata`` so the complete error context survives on disk.
``faulthandler`` + a ``sys.excepthook`` chain are added too, so a genuine native
crash or an uncaught Python exception lands in the SAME file (one timeline).

Pure diagnostics: it OBSERVES; it never changes plugin behaviour, and every
entry point is best-effort (never raises into the caller).

USAGE
-----
Call :func:`install` once when the large-data path activates (idempotent).
:func:`log_path` returns the current session file so the UI can tell the user
where to find it.  Strip this once the progressive-3D lifecycle is stable.
"""

from __future__ import annotations

import datetime
import faulthandler
import sys
import traceback
from pathlib import Path

# Module-global state: the open file handle + the wrapped streams must outlive
# the install call for the whole process.
_INSTALLED = False
_LOG_FH = None
_LOG_PATH = None
_PREV_EXCEPTHOOK = None
_PREV_STDOUT = None
_PREV_STDERR = None

#: Keep the last N session logs so the folder does not grow without bound.
_KEEP_SESSIONS = 10


def default_log_dir() -> Path:
    """Fixed, workspace-independent location: ``~/.epicycle/step1_largedata``."""
    return Path.home() / ".epicycle" / "step1_largedata"


class _Tee:
    """Write-through wrapper: everything written to the original stream is also
    written (flushed) to the log file.  Delegates all other attributes to the
    original stream so it is a drop-in for ``sys.stdout`` / ``sys.stderr``.
    Best-effort — a failure writing to either sink never propagates."""

    def __init__(self, original, fh):
        self._original = original
        self._fh = fh

    def write(self, s):
        try:
            if self._original is not None:
                self._original.write(s)
        except Exception:  # noqa: BLE001
            pass
        try:
            self._fh.write(s)
            self._fh.flush()
        except Exception:  # noqa: BLE001
            pass
        # Return a plausible count so callers relying on write()'s return
        # value (rare) do not choke.
        try:
            return len(s)
        except Exception:  # noqa: BLE001
            return 0

    def flush(self):
        for sink in (self._original, self._fh):
            try:
                if sink is not None:
                    sink.flush()
            except Exception:  # noqa: BLE001
                pass

    def __getattr__(self, name):
        # Delegate isatty/fileno/encoding/etc. to the real stream.
        return getattr(self._original, name)


def _rotate_old_sessions(log_dir: Path) -> None:
    try:
        sessions = sorted(
            log_dir.glob("session_*.log"),
            key=lambda p: p.stat().st_mtime,
        )
        for stale in sessions[:-_KEEP_SESSIONS]:
            try:
                stale.unlink()
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass


def install(log_dir: Path | None = None) -> Path | None:
    """Arm session logging (idempotent).  Returns the session log path, or
    ``None`` if it could not be set up (in which case nothing was changed)."""
    global _INSTALLED, _LOG_FH, _LOG_PATH, _PREV_EXCEPTHOOK
    global _PREV_STDOUT, _PREV_STDERR
    if _INSTALLED:
        return _LOG_PATH
    try:
        log_dir = Path(log_dir) if log_dir is not None else default_log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        _rotate_old_sessions(log_dir)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = log_dir / f"session_{stamp}_pid{_safe_pid()}.log"
        fh = open(path, "a", encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return None

    _LOG_FH = fh
    _LOG_PATH = path

    # Header.
    try:
        fh.write(
            f"\n===== EpiCycle Step-1 large-data session log =====\n"
            f"started: {datetime.datetime.now().isoformat()}  pid: {_safe_pid()}\n"
            f"(captures stdout+stderr incl. caught vispy draw tracebacks + "
            f"faulthandler native dumps + uncaught Python exceptions)\n\n"
        )
        fh.flush()
    except Exception:  # noqa: BLE001
        pass

    # 1) faulthandler -> the file (a genuine native crash / signal).
    try:
        faulthandler.enable(file=fh, all_threads=True)
    except Exception:  # noqa: BLE001
        pass

    # 2) Tee stdout + stderr -> the file.  This is the piece that captures the
    #    CAUGHT vispy draw tracebacks (they are printed, not crashed on).
    try:
        _PREV_STDOUT = sys.stdout
        sys.stdout = _Tee(sys.stdout, fh)
    except Exception:  # noqa: BLE001
        _PREV_STDOUT = None
    try:
        _PREV_STDERR = sys.stderr
        sys.stderr = _Tee(sys.stderr, fh)
    except Exception:  # noqa: BLE001
        _PREV_STDERR = None

    # 3) Uncaught Python exceptions -> the file (chained to the previous hook).
    try:
        _PREV_EXCEPTHOOK = sys.excepthook

        def _hook(exc_type, exc, tb):
            try:
                fh.write("\n----- uncaught Python exception -----\n")
                traceback.print_exception(exc_type, exc, tb, file=fh)
                fh.flush()
            except Exception:  # noqa: BLE001
                pass
            if _PREV_EXCEPTHOOK is not None:
                _PREV_EXCEPTHOOK(exc_type, exc, tb)

        sys.excepthook = _hook
    except Exception:  # noqa: BLE001
        _PREV_EXCEPTHOOK = None

    _INSTALLED = True
    return _LOG_PATH


def breadcrumb(msg: str) -> None:
    """Write a flushed, timestamped one-liner to the session file (best-effort).
    Also visible on the teed stdout, so it interleaves with the console."""
    if _LOG_FH is None:
        return
    try:
        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        _LOG_FH.write(f"[{ts}] [DIAG] {msg}\n")
        _LOG_FH.flush()
    except Exception:  # noqa: BLE001
        pass


def log_path() -> Path | None:
    """The current session log path, or ``None`` if logging is not armed."""
    return _LOG_PATH


def _safe_pid() -> int:
    try:
        import os

        return os.getpid()
    except Exception:  # noqa: BLE001
        return 0
