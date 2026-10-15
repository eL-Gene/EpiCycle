from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

_LOCK_FILENAME = "workspace.lock"


def atomic_write_json(
    path: "str | Path",
    payload: Any,
    *,
    indent: int = 2,
    tmp_prefix: str = ".epicycle_json_",
) -> Path:
    """Write ``payload`` as JSON to ``path`` atomically.

    Serialises to a temp file in the SAME directory, then ``os.replace``s it
    over the destination.  A crash (or a raising ``os.replace``) therefore
    leaves the previous file byte-intact instead of truncated — the failure
    mode that made a mid-save crash on ``step2_state.json`` fall into the
    corrupt-recovery branch and lose every other run's entry.

    Release-hardening #3 (2026-09-07): extracted from
    ``JSONStateStore.save_state`` so Step 2's four hand-rolled
    ``open(..., "w") + json.dump`` sites can share it.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        suffix=".tmp", prefix=tmp_prefix, dir=str(dest.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=indent)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, str(dest))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return dest


def quarantine_corrupt_file(path: "str | Path") -> Path | None:
    """Rename an unreadable/corrupt file aside as ``<name>.corrupt-<ts>``.

    Returns the quarantine path, or None if there was nothing to move (or the
    move failed — quarantine is best-effort and must never block a save).
    Release-hardening #3: the pre-fix corrupt branch silently overwrote the
    bad file with fresh defaults, destroying the only forensic copy of a
    state file the user might still have wanted recovered by hand.
    """
    src = Path(path)
    try:
        if not src.is_file():
            return None
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        dst = src.with_name(f"{src.name}.corrupt-{ts}")
        os.replace(str(src), str(dst))
        return dst
    except Exception:
        return None


class WorkspaceLock:
    """Advisory file lock to detect concurrent instances on the same workspace.

    Creates a ``workspace.lock`` file containing the current PID. On acquire,
    checks whether a stale lock from a dead process exists and cleans it up.
    """

    def __init__(self, directory: str) -> None:
        self._lock_path = Path(directory) / _LOCK_FILENAME
        self._held = False

    def acquire(self) -> bool:
        """Try to acquire the workspace lock.

        Returns True if the lock was acquired. Returns False if another live
        process already holds it (caller should warn the user).
        """
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        if self._lock_path.exists():
            try:
                existing_pid = int(self._lock_path.read_text(encoding="utf-8").strip())
            except (ValueError, OSError):
                existing_pid = None
            if existing_pid is not None and _pid_alive(existing_pid) and existing_pid != os.getpid():
                logger.warning(
                    "Workspace %s appears locked by PID %d. "
                    "Another instance may be using this workspace.",
                    self._lock_path.parent,
                    existing_pid,
                )
                return False
            # Stale lock (dead PID) or our own PID — safe to overwrite.
        try:
            self._lock_path.write_text(str(os.getpid()), encoding="utf-8")
            self._held = True
        except OSError as exc:
            logger.warning("Could not write workspace lock: %s", exc)
            return False
        return True

    def release(self) -> None:
        """Release the lock if we hold it."""
        if not self._held:
            return
        try:
            # Only remove if it's still our PID (don't clobber another instance).
            if self._lock_path.exists():
                existing_pid = int(self._lock_path.read_text(encoding="utf-8").strip())
                if existing_pid == os.getpid():
                    self._lock_path.unlink(missing_ok=True)
        except (ValueError, OSError):
            pass
        self._held = False

    @property
    def held(self) -> bool:
        return self._held


def _pid_alive(pid: int) -> bool:
    """Check whether a process with the given PID is still running."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class JSONStateStore:
    def __init__(self, filename: str = "hitl_state.json") -> None:
        self.filename = filename

    def save_state(self, directory: str, state_dict: Dict[str, Any]) -> Path:
        """Write state atomically via temp file + os.replace."""
        base = Path(directory)
        base.mkdir(parents=True, exist_ok=True)
        path = base / self.filename

        # Write to a temp file in the same directory, then atomically replace.
        return atomic_write_json(path, state_dict, tmp_prefix=".hitl_state_")

    def load_state(self, directory: str) -> Dict[str, Any] | None:
        """Load state with graceful handling of corrupt/truncated JSON."""
        path = Path(directory) / self.filename
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                logger.warning("State file %s contained non-dict data, ignoring", path)
                return None
            return data
        except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as exc:
            logger.warning(
                "Corrupt state file %s (%s), starting fresh", path, exc
            )
            return None
