"""Subprocess entry point for Step 3 QC feature extraction.

Runs the GIL-bound QC feature-extraction compute (``run_full_qc_extraction``)
in a SEPARATE process so the napari UI stays live during the 100-270s
extraction.  The parent controller spawns this via::

    python -m napari_epicycle.fusion_reconstruction._qc_extract_subprocess <config.json>

Protocol (parent ⇄ child over stdout / exit code):

  * ``argv[1]`` = path to a config JSON (the ``run_full_qc_extraction``
    kwargs dict — all disk paths + scalar config).
  * Progress lines ``[QC-PERF] ...`` / ``[QC-EXTRACT] ...`` are printed to
    stdout (flushed per line) so the parent's stdout reader can forward
    them to the diagnostic console live.
  * A cancel file ``<config_path>.cancel`` is checked between phases (in
    the ``progress`` callback).  If present → print ``QC-EXTRACT-CANCELLED``
    and ``sys.exit(2)``.
  * The full worker adjacency edge list is written to
    ``<config_path>.adjacency.json`` on success so the parent can
    re-attach it to the re-read df (the features CSV does NOT carry
    ``df.attrs``).
  * On success: print ``QC-EXTRACT-DONE csv=<csv_path> rows=<n>`` then
    ``sys.exit(0)``.
  * On exception: print ``QC-EXTRACT-ERROR <message>`` + the full traceback
    to STDOUT (the parent merges the child's stderr into stdout via
    ``stderr=subprocess.STDOUT``, so writing the traceback to stdout keeps
    it on the single drained pipe), then ``sys.exit(1)``.

The parent spawns this with ``stderr=subprocess.STDOUT`` (one pipe), so any
incidental stderr chatter (sklearn / UMAP / numba / scipy warnings — 100-270s
of it) cannot fill a separate, undrained stderr pipe buffer and deadlock the
child.  Everything the child emits flows to the parent's single stdout reader.

A parent-liveness watchdog (daemon thread) ``os._exit(3)`` if the parent
napari process disappears (force-close), so the child never orphans.

This module imports the compute chain ONLY when ``main`` runs (heavy
scientific stack), so importing the module itself stays cheap for tests.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path


# Sentinel marker strings the parent's stdout parser matches on.
MARKER_DONE = "QC-EXTRACT-DONE"
MARKER_ERROR = "QC-EXTRACT-ERROR"
MARKER_CANCELLED = "QC-EXTRACT-CANCELLED"


class _Cancelled(Exception):
    """Raised from the progress callback when the cancel file appears."""


def _cancel_path(config_path: str) -> Path:
    return Path(str(config_path) + ".cancel")


def _adjacency_path(config_path: str) -> Path:
    return Path(str(config_path) + ".adjacency.json")


def _spacing_path(config_path: str) -> Path:
    """R7: the voxel size the run read from the patch file, for the
    parent to store (``null`` when none was read)."""
    return Path(str(config_path) + ".spacing.json")


def _parent_alive(ppid: int) -> bool:
    """Best-effort cross-platform check that PID ``ppid`` is still running.

    Prefers ``psutil`` (a napari dependency).  Falls back to ``os.kill(pid, 0)``
    on POSIX and a Windows ``OpenProcess`` + exit-code probe via ctypes.  On any
    uncertainty returns ``True`` (don't kill the child on a flaky probe).
    """
    if ppid <= 0:
        return True
    try:
        import psutil  # napari dep — present in the app env
        if not psutil.pid_exists(ppid):
            return False
        try:
            proc = psutil.Process(ppid)
            # A reaped-but-not-cleared PID shows as a zombie; treat as dead.
            return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return False
    except Exception:
        pass
    # psutil unavailable — portable fallbacks.
    if os.name == "posix":
        try:
            os.kill(ppid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True  # exists but not ours to signal
        except Exception:
            return True
    # Windows fallback via ctypes OpenProcess + GetExitCodeProcess.
    try:
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, ppid
        )
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            if not ok:
                return True
            return code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    except Exception:
        return True


def _start_parent_watchdog(ppid: int, poll_s: float = 3.0) -> None:
    """Daemon thread: ``os._exit(3)`` if the parent process dies.

    Guarantees the GIL-bound child never orphans when napari is force-killed
    mid-extract (no event loop / signal handler to rely on).  Daemon + minimal
    so it never blocks normal exit; ``os._exit`` skips cleanup deliberately
    (the parent is gone — there is nothing to flush to).
    """
    if not ppid or ppid <= 0:
        return

    def _watch() -> None:
        while True:
            time.sleep(poll_s)
            if not _parent_alive(ppid):
                # Parent gone — bail hard (skip atexit / finalizers).
                os._exit(3)

    t = threading.Thread(target=_watch, name="qc-parent-watchdog", daemon=True)
    t.start()


def main(argv=None) -> int:
    """Run extraction from a config JSON path in ``argv[1]``.

    Returns the process exit code (also the value passed to ``sys.exit``
    when invoked as ``__main__``).
    """
    argv = list(sys.argv if argv is None else argv)
    if len(argv) < 2:
        print(f"{MARKER_ERROR} missing config path argument", flush=True)
        return 1

    config_path = argv[1]
    try:
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 — report any read/parse failure
        print(f"{MARKER_ERROR} failed to read config {config_path!r}: {exc}",
              flush=True)
        return 1

    # Parent-liveness watchdog — start as early as possible so a force-close
    # during the heavy import below still gets reaped.  The parent PID is
    # threaded through the config JSON under ``_parent_pid`` (popped before the
    # config is forwarded to ``run_full_qc_extraction``, which doesn't accept
    # it).
    parent_pid = 0
    try:
        parent_pid = int(config.pop("_parent_pid", 0) or 0)
    except Exception:
        parent_pid = 0
    _start_parent_watchdog(parent_pid)

    cancel_file = _cancel_path(config_path)

    def _progress(phase_name: str, ms: float) -> None:
        # Cancel between phases — cheapest cancellation granularity without
        # threading the cancel check through every compute_* function.
        if cancel_file.exists():
            raise _Cancelled()

    # Heavy import deferred to call time so module import stays cheap.
    from napari_epicycle.fusion_reconstruction._qc_extract_core import (
        run_full_qc_extraction,
    )

    try:
        # ``log`` mirrors free-text progress lines to stdout too, so the
        # parent's console shows the same messages it did when the worker
        # ran in-process.
        def _log(msg: str) -> None:
            print(msg, flush=True)

        summary = run_full_qc_extraction(
            **config,
            progress=_progress,
            log=_log,
        )
    except _Cancelled:
        print(MARKER_CANCELLED, flush=True)
        return 2
    except BaseException as exc:  # noqa: BLE001 — surface everything to parent
        import traceback
        # Parent merges child stderr into stdout (stderr=subprocess.STDOUT), so
        # print BOTH the marker and the traceback to stdout — the single drained
        # pipe — so the error path stays detectable even with stderr merged.
        print(f"{MARKER_ERROR} {exc}", flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        return 1

    # Persist the adjacency edge list so the parent can re-attach it to the
    # df it re-reads from the CSV (the CSV does not carry df.attrs).
    try:
        _adjacency_path(config_path).write_text(
            json.dumps(summary.get("adjacency_list", [])),
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001 — non-fatal; curated-adj is best-effort
        print(f"[QC-EXTRACT] WARN failed to write adjacency sidecar: {exc}",
              flush=True)
    # R7: the voxel size read from the patch file (or null)
    try:
        _spacing_path(config_path).write_text(
            json.dumps(summary.get("spacing_found")), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 — non-fatal; the size is re-read
        print(f"[QC-EXTRACT] WARN failed to write spacing sidecar: {exc}",
              flush=True)

    print(
        f"{MARKER_DONE} csv={summary['csv_path']} rows={summary['rows']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
