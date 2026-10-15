"""Persistent crash + breadcrumb logging for the EpiCycle annotator.

WHY THIS EXISTS
---------------
The 3D curation annotator can hard-crash with a Qt/vispy **access violation**
(e.g. a right-click handler dereferencing a GPU visual / layer that was torn
down on an earlier patch switch).  An access violation kills the process at the
C level — Python ``try/except`` does NOT catch it, and the console buffer is
lost — so the existing ``print()`` diagnostics (``[RC-WHY]``, ``[XCRASH-1]``,
``[SW-PERF]`` …) tell us nothing after the fact.

This module captures the evidence on DISK so it survives the crash:

  1. ``faulthandler`` dumps the native + Python stack of ALL threads to a file
     on any fatal signal (SIGSEGV / access violation).  This is the ONLY thing
     that survives a hard C-level crash, and it names the exact call that
     touched the freed object.
  2. ``breadcrumb()`` writes a flushed, timestamped one-liner to the SAME file,
     so the last action before the crash (the code path mid-flight) is on disk.
  3. ``sys.excepthook`` captures uncaught *Python* exceptions (the non-C-crash
     flavour) with a full traceback.

All three write to one file so the timeline reads top-to-bottom:
``… breadcrumbs … [last breadcrumb] [faulthandler native dump]``.

USAGE
-----
Call ``install_crash_logging()`` once, early in the annotator lifecycle
(idempotent — safe to call on every launch).  Then sprinkle ``breadcrumb(msg)``
at crash-relevant boundaries (patch switch, subviewer teardown, right-click
enter, just-before a render-triggering call).  Use ``liveness(name=obj, …)`` to
record whether Qt objects are still alive right before they're dereferenced.

This is pure diagnostics: it OBSERVES, it never changes annotator behaviour.
"""

from __future__ import annotations

import datetime
import faulthandler
import os
import sys
import traceback
from pathlib import Path

# Module-global state (faulthandler + the open file handle must outlive the
# call that installs them, for the whole process).
_INSTALLED = False
_LOG_FH = None
_LOG_PATH = None
_PREV_EXCEPTHOOK = None

# Keep the last N session logs so the folder doesn't grow without bound.
_KEEP_SESSIONS = 10


def default_log_dir() -> Path:
    """Fixed, workspace-independent location: ``~/.epicycle/crash_logs``.

    Workspace-independent on purpose — the log must exist even if the crash
    happens before/without a workspace, and the user always knows where to
    look.  Overridable via the ``EPICYCLE_CRASH_LOG_DIR`` env var.
    """
    override = os.environ.get("EPICYCLE_CRASH_LOG_DIR")
    if override:
        return Path(override)
    return Path.home() / ".epicycle" / "crash_logs"


def _prune_old_sessions(log_dir: Path) -> None:
    """Keep only the most recent ``_KEEP_SESSIONS`` ``epicycle_session_*.log``."""
    try:
        logs = sorted(log_dir.glob("epicycle_session_*.log"))
        for stale in logs[:-_KEEP_SESSIONS]:
            try:
                stale.unlink()
            except OSError:
                pass
    except Exception:
        pass


def install_crash_logging(log_dir=None) -> "Path | None":
    """Enable faulthandler + breadcrumb logging to a per-session file.

    Idempotent: the first call wins; later calls just return the active path.
    Best-effort: any failure here must never break annotator launch, so the
    whole body is guarded.
    """
    global _INSTALLED, _LOG_FH, _LOG_PATH, _PREV_EXCEPTHOOK
    if _INSTALLED:
        return _LOG_PATH
    try:
        log_dir = Path(log_dir) if log_dir is not None else default_log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        _prune_old_sessions(log_dir)
        # datetime.now() is fine in runtime code (this is not a workflow script).
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        _LOG_PATH = log_dir / f"epicycle_session_{ts}_pid{os.getpid()}.log"
        # Line-buffered text handle: our writes flush per newline; faulthandler
        # writes raw bytes straight to the fd on a fatal signal (bypassing the
        # buffer), so as long as we flush each breadcrumb the on-disk order is
        # correct.
        _LOG_FH = open(_LOG_PATH, "a", buffering=1, encoding="utf-8")
        faulthandler.enable(file=_LOG_FH, all_threads=True)
        _PREV_EXCEPTHOOK = sys.excepthook
        sys.excepthook = _excepthook
        _INSTALLED = True
        breadcrumb(
            f"=== EpiCycle crash logging ENABLED (pid={os.getpid()}, "
            f"faulthandler all_threads) ==="
        )
        return _LOG_PATH
    except Exception as exc:  # pragma: no cover - never break launch
        try:
            print(f"[CRASH-LOG] install failed (non-fatal): {exc!r}", flush=True)
        except Exception:
            pass
        return None


def is_installed() -> bool:
    return _INSTALLED


def breadcrumb(msg: str) -> None:
    """Write a flushed, timestamped breadcrumb to the session log.

    No-op (and never raises) if logging isn't installed.  The flush is the
    whole point — the LAST line written before a crash must be on disk.
    """
    fh = _LOG_FH
    if fh is None:
        return
    try:
        # millisecond precision so press/release/render ordering is visible
        stamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        fh.write(f"{stamp} {msg}\n")
        fh.flush()
        try:
            os.fsync(fh.fileno())  # belt + suspenders for a hard crash
        except OSError:
            pass
    except Exception:
        pass


def liveness(**objs) -> str:
    """Render a compact liveness string for Qt objects about to be touched.

    e.g. ``liveness(layer=layer, nav=nav, qt_viewer=qv)`` ->
    ``"layer=alive nav=alive qt_viewer=DELETED"``.  A ``DELETED`` (or
    ``none``) entry immediately before the faulthandler dump is the smoking
    gun for a use-after-free.  Never raises.
    """
    try:
        from qtpy import sip  # type: ignore
    except Exception:
        sip = None

    parts = []
    for name, obj in objs.items():
        if obj is None:
            parts.append(f"{name}=none")
            continue
        state = "alive"
        if sip is not None:
            try:
                if sip.isdeleted(obj):
                    state = "DELETED"
            except (TypeError, RuntimeError):
                # Not a sip-wrapped Qt object (e.g. a napari layer) — fall back
                # to a plain liveness hint below.
                state = "alive?"
            except Exception:
                state = "probe-err"
        parts.append(f"{name}={state}")
    return " ".join(parts)


def _excepthook(exc_type, exc_value, exc_tb) -> None:
    """Capture uncaught Python exceptions to the session log, then chain."""
    try:
        tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        breadcrumb("!!! UNCAUGHT PYTHON EXCEPTION !!!\n" + tb)
    except Exception:
        pass
    if _PREV_EXCEPTHOOK is not None:
        try:
            _PREV_EXCEPTHOOK(exc_type, exc_value, exc_tb)
        except Exception:
            pass
