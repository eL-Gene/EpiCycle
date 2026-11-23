"""Persistent session log, crash dump and breadcrumbs for EpiCycle.

WHY THIS EXISTS
---------------
The 3D curation annotator can hard-crash with a Qt/vispy **access violation**
(e.g. a right-click handler dereferencing a GPU visual / layer that was torn
down on an earlier patch switch).  An access violation kills the process at the
C level: Python ``try/except`` does not catch it and the console is lost with
the process.

This module keeps the evidence on disk, in ONE file per session:

  1. ``faulthandler`` dumps the native + Python stack of ALL threads to the
     file on any fatal signal.  This is the only thing that survives a hard
     C-level crash, and it names the exact call that touched the freed object.
  2. ``breadcrumb()`` writes a flushed, timestamped line to the same file.
  3. ``mirror_console()`` copies everything Python prints (stdout and stderr,
     and log handlers that wrote to them) into the file, a whole line at a
     time, after the real console has received it unchanged.
  4. ``sys.excepthook`` records uncaught Python exceptions with a traceback.

The timeline reads top to bottom: ``... printed lines ... [last line] [dump]``.
A file that ends without the ``session end (clean)`` line means the session
crashed or was killed.

USAGE
-----
Call ``install_crash_logging()`` early (idempotent; safe on every launch).
Then call ``breadcrumb(msg)`` at crash-relevant boundaries, and
``liveness(name=obj, ...)`` to record whether Qt objects are still alive right
before they are used.

This is pure diagnostics: it observes, it never changes behaviour.
"""

from __future__ import annotations

import atexit
import datetime
import faulthandler
import io
import logging
import os
import platform
import re
import sys
import threading
import traceback
from pathlib import Path

# Module-global state (faulthandler + the open file handle must outlive the
# call that installs them, for the whole process).
_INSTALLED = False
_LOG_FH = None
_LOG_PATH = None
_PREV_EXCEPTHOOK = None

# One lock for every write to the session file, so lines from different
# threads never interleave inside a line.
_LOCK = threading.RLock()

# Console mirroring state.
_MIRRORED = False
_MIRROR_STOPPED = False
_ORIG_STDOUT = None
_ORIG_STDERR = None
_TEE_OUT = None
_TEE_ERR = None
_REPOINTED = []  # (handler, original stream) pairs, undone by restore_console
_FOOTER_DONE = False
_ATEXIT_ARMED = False

_FILE_NOTICE = (
    "EpiCycle session log. It lists the folders and files opened in this "
    "session; their paths include your user name. Check it before posting it "
    "publicly."
)

# Keep the last N session logs so the folder doesn't grow without bound.
_KEEP_SESSIONS = 10

_LINE_SPLIT = re.compile(r"(\r\n|\n|\r)")


def default_log_dir() -> Path:
    """Fixed, workspace-independent location: ``~/.epicycle/crash_logs``.

    Workspace-independent on purpose: the log must exist even if the crash
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


def _stamp() -> str:
    # millisecond precision so press/release/render ordering is visible
    return datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]


class _Tee:
    """A stand-in for ``sys.stdout`` / ``sys.stderr`` that also logs to the file.

    Every write goes to the real stream first, unchanged, then whole lines are
    copied into the session file with a time prefix.  Each thread keeps its
    own unfinished line, so two threads printing at once never mix their text.
    If the real stream is ``None`` (no terminal) only the file is written.
    Nothing here ever raises into the caller.
    """

    def __init__(self, original):
        self.__dict__["_original"] = original
        self.__dict__["_local"] = threading.local()

    def write(self, s):
        orig = self._original
        ret = None
        failed = False
        if orig is not None:
            try:
                ret = orig.write(s)
            except Exception:
                failed = True
        if orig is None or failed:
            try:
                ret = len(s)
            except Exception:
                ret = 0
        if not _MIRROR_STOPPED:
            self._mirror(s)
        return ret

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def _mirror(self, s) -> None:
        try:
            if not isinstance(s, str):
                return
            local = self._local
            buf = getattr(local, "buf", "") + s
            parts = _LINE_SPLIT.split(buf)
            local.buf = parts[-1]
            lines = []
            for i in range(0, len(parts) - 1, 2):
                text, sep = parts[i], parts[i + 1]
                if sep == "\r" and text == "":
                    continue
                lines.append(text)
            if not lines:
                return
            stamp = _stamp()
            with _LOCK:
                fh = _LOG_FH
                if fh is None or _MIRROR_STOPPED:
                    return
                fh.write("".join(f"{stamp} {t}\n" for t in lines))
                fh.flush()
        except OSError as exc:
            self._stop_mirror(repr(exc))
        except Exception:
            pass  # this write only; the mirror carries on

    def _stop_mirror(self, reason: str) -> None:
        """Stop copying to the file; the console keeps working as before."""
        global _MIRROR_STOPPED
        with _LOCK:
            if _MIRROR_STOPPED:
                return
            _MIRROR_STOPPED = True
        orig = self._original
        if orig is not None:
            try:
                orig.write(f"[SESSION-LOG] mirror stopped: {reason}\n")
            except Exception:
                pass

    def flush(self):
        try:
            if self._original is not None:
                self._original.flush()
        except Exception:
            pass

    def fileno(self):
        if self._original is None:
            raise io.UnsupportedOperation("fileno")
        return self._original.fileno()

    def isatty(self):
        if self._original is None:
            return False
        return self._original.isatty()

    def __getattr__(self, name):
        orig = self.__dict__.get("_original")
        if orig is None:
            if name == "encoding":
                return "utf-8"
            if name == "errors":
                return "backslashreplace"
            if name in ("closed", "line_buffering"):
                return False
            raise AttributeError(name)
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(orig, name)


def _repoint_handlers(old, new) -> None:
    """Point log handlers that hold ``old`` at ``new`` (remembers each one)."""
    if old is None:
        return
    seen = []
    loggers = [logging.root] + [
        lg for lg in list(logging.Logger.manager.loggerDict.values())
        if isinstance(lg, logging.Logger)
    ]
    for lg in loggers:
        for h in list(getattr(lg, "handlers", [])):
            if any(h is x for x in seen):
                continue
            seen.append(h)
            if isinstance(h, logging.StreamHandler) and h.stream is old:
                try:
                    h.setStream(new)
                    _REPOINTED.append((h, old))
                except Exception:
                    pass


def mirror_console(*, stdout=None, stderr=None, force=False) -> bool:
    """Copy everything printed to stdout / stderr into the session file.

    Returns True when the mirror is active.  Skipped under pytest, and when
    ``EPICYCLE_SESSION_LOG=0``, unless the caller passes streams (or
    ``force=True``).  Calling it again while active changes nothing.
    """
    global _MIRRORED, _ORIG_STDOUT, _ORIG_STDERR, _TEE_OUT, _TEE_ERR
    global _MIRROR_STOPPED
    try:
        if _MIRRORED:
            return True
        if _LOG_FH is None:
            return False
        asked = force or stdout is not None or stderr is not None
        if not asked:
            if os.environ.get("EPICYCLE_SESSION_LOG") == "0":
                print("[SESSION-LOG] skipped: EPICYCLE_SESSION_LOG=0")
                return False
            if "pytest" in sys.modules:
                print("[SESSION-LOG] skipped: running under pytest")
                return False
        out = stdout if stdout is not None else sys.stdout
        err = stderr if stderr is not None else sys.stderr
        _MIRROR_STOPPED = False
        _ORIG_STDOUT, _ORIG_STDERR = out, err
        _TEE_OUT, _TEE_ERR = _Tee(out), _Tee(err)
        _repoint_handlers(out, _TEE_OUT)
        if err is not out:
            _repoint_handlers(err, _TEE_ERR)
        sys.stdout = _TEE_OUT
        sys.stderr = _TEE_ERR
        _MIRRORED = True
        return True
    except Exception:
        return False


def restore_console() -> None:
    """Undo ``mirror_console`` (for tests).  Leaves foreign replacements alone."""
    global _MIRRORED
    try:
        for h, old in reversed(_REPOINTED):
            try:
                h.setStream(old)
            except Exception:
                pass
        _REPOINTED.clear()
        if _TEE_OUT is not None and sys.stdout is _TEE_OUT:
            sys.stdout = _ORIG_STDOUT
        if _TEE_ERR is not None and sys.stderr is _TEE_ERR:
            sys.stderr = _ORIG_STDERR
    finally:
        _MIRRORED = False


def _versions_line() -> str:
    parts = []
    try:
        from importlib.metadata import version

        for label, dist in (("plugin", "napari-epicycle"), ("napari", "napari")):
            try:
                parts.append(f"{label}={version(dist)}")
            except Exception:
                parts.append(f"{label}=?")
    except Exception:
        pass
    parts.append(f"python={platform.python_version()}")
    parts.append(f"platform={platform.platform()}")
    return "versions: " + " ".join(parts)


def log_path() -> "Path | None":
    """The current session file, or ``None`` when logging is not installed."""
    return _LOG_PATH


def write_footer() -> None:
    """Write the clean-exit line, once.  A file without it ended in a crash."""
    global _FOOTER_DONE
    if _FOOTER_DONE or _LOG_FH is None:
        return
    _FOOTER_DONE = True
    breadcrumb("=== session end (clean) ===")


def install_crash_logging(log_dir=None) -> "Path | None":
    """Enable faulthandler, breadcrumbs and the console mirror, one file.

    The first call opens the file; later calls return the same path but point
    the crash dump at it again (other code may have taken it over).
    Best-effort: any failure here must never break launch, so the whole body
    is guarded.
    """
    global _INSTALLED, _LOG_FH, _LOG_PATH, _PREV_EXCEPTHOOK
    global _FOOTER_DONE, _ATEXIT_ARMED
    if _INSTALLED:
        try:
            faulthandler.enable(file=_LOG_FH, all_threads=True)
        except Exception:
            pass
        return _LOG_PATH
    if log_dir is None and "pytest" in sys.modules:
        print("[SESSION-LOG] skipped: running under pytest")
        return None
    try:
        log_dir = Path(log_dir) if log_dir is not None else default_log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        _prune_old_sessions(log_dir)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        _LOG_PATH = log_dir / f"epicycle_session_{ts}_pid{os.getpid()}.log"
        # Line-buffered text handle: our writes flush per newline; faulthandler
        # writes raw bytes straight to the fd on a fatal signal (bypassing the
        # buffer), so as long as every line is flushed the on-disk order is
        # correct.  backslashreplace: one odd file name must not end logging.
        _LOG_FH = open(_LOG_PATH, "a", buffering=1, encoding="utf-8",
                       errors="backslashreplace")
        _LOG_FH.write(_FILE_NOTICE + "\n")
        _LOG_FH.flush()
        faulthandler.enable(file=_LOG_FH, all_threads=True)
        _PREV_EXCEPTHOOK = sys.excepthook
        sys.excepthook = _excepthook
        _INSTALLED = True
        _FOOTER_DONE = False
        breadcrumb(
            f"=== EpiCycle crash logging ENABLED (pid={os.getpid()}, "
            f"faulthandler all_threads) ==="
        )
        breadcrumb(_versions_line())
        if not _ATEXIT_ARMED:
            atexit.register(write_footer)
            _ATEXIT_ARMED = True
        if mirror_console():
            print(f"[SESSION-LOG] this session is written to {_LOG_PATH}")
        else:
            print("[SESSION-LOG] breadcrumbs and crash dumps only -> "
                  f"{_LOG_PATH}")
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
    """Write a flushed, timestamped line to the session log.

    No-op (and never raises) if logging isn't installed.  The flush is the
    whole point: the LAST line written before a crash must be on disk.  There
    is no fsync: a flush survives a process crash, and an fsync here would
    also force out everything the console mirror wrote since the last
    breadcrumb.
    """
    fh = _LOG_FH
    if fh is None:
        return
    try:
        with _LOCK:
            fh.write(f"{_stamp()} {msg}\n")
            fh.flush()
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
