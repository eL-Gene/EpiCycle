"""Per-stack segmentation workflow state persistence (Step 4).

Tracks each stack through the Step 4 pipeline:
    unprocessed → predicted → reconstructed → qc_extracted ("measured")
    → finalized, with ``failed`` for a Predict / Reconstruct that failed
    with nothing later to fall back on.

State lives in ``{workspace}/step4_state.json`` (schema 2), at the
workspace root beside ``step1_state.json`` / ``step2_state.json`` /
``step3_state.json`` (owner decision, 2026-09-29).  A workspace whose state
still sits at the earlier ``artifacts/step4/step4_state.json`` is read from
there and moved once: written to the root, the old copy renamed aside as
``step4_state.json.moved-<stamp>`` (an adopt reads it and moves nothing).
Loading never silently discards a file it cannot read: the file is
renamed aside where it was read (``step4_state.json.corrupt-<stamp>``) and
Step 4 falls back to the backup, then the legacy
``{workspace}/stack_segmentation_state.json`` (schema 1, migrated in
memory), then an empty state.  If the rename itself
fails, ``save_blocked`` refuses every save so the unreadable file is never
overwritten (invariant I-4).  The adopt path (``load(write=False)``) renames
nothing and writes nothing.  Saves are atomic (mkstemp + fsync +
``os.replace``) and rotate the previous good file to
``artifacts/step4/state_backup/step4_state.json.bak``.  The legacy file is
read, never written (I-15).
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from . import _step4_text as T


VALID_STATUSES = (
    "unprocessed",
    "predicted",
    "segmented",        # legacy only: migrated to "failed"
    "reconstructed",
    "qc_extracted",
    "smoothed",         # legacy only: migrated to "qc_extracted"
    "finalized",
    "failed",
)

_SCHEMA_VERSION = 2
_LEGACY_SCHEMA_VERSION = 1

LEGACY_STATE_FILENAME = "stack_segmentation_state.json"

# Metadata keys cleared on a backward transition (Reset to Original, and a
# Reconstruct over an existing 3D result).
_QC_METADATA_KEYS = (
    "qc_features_csv", "qc_tier_map", "channel_name_map",
    "border_cells_path", "smoothed_cache_path",
    "measured_at", "measured_cells",
    "measured_spacing", "measured_z_orientation",
    "smoothed_at", "n_thin_kept",
    "finalized_at", "finalized_fingerprint",
)


def step4_state_path(workspace: str) -> Path:
    """The state file: at the workspace root, like Steps 1-3's."""
    return Path(workspace) / StackStateManager.STATE_FILENAME


def step4_artifacts_state_path(workspace: str) -> Path:
    """Where the state lived before 2026-09-29; read and moved once."""
    return (Path(workspace) / "artifacts" / "step4"
            / StackStateManager.STATE_FILENAME)


def step4_backup_path(workspace: str) -> Path:
    return (Path(workspace) / "artifacts" / "step4" / "state_backup"
            / (StackStateManager.STATE_FILENAME + ".bak"))


def legacy_state_path(workspace: str) -> Path:
    return Path(workspace) / LEGACY_STATE_FILENAME


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_state() -> Dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "stacks": {},
        "session": {},
        "active_stack": None,
        "last_updated": None,
        "migrated_from_legacy_at": None,
    }


def _read_json(path: Path):
    """``(data, ok)``: ``ok`` is False for an unreadable / non-JSON file."""
    try:
        return json.loads(path.read_text(encoding="utf-8")), True
    except (OSError, ValueError, UnicodeDecodeError):
        return None, False


def _good_v2(data) -> bool:
    """A readable schema-2 state.  A wrong-typed ``stacks`` or ``session``
    is NOT good: it is quarantined, never silently emptied (audit D12)."""
    if not isinstance(data, dict) or data.get("schema_version") != _SCHEMA_VERSION:
        return False
    for key in ("stacks", "session"):
        if key in data and not isinstance(data[key], dict):
            return False
    return True


def _merge_over_empty(data: Dict[str, Any]) -> Dict[str, Any]:
    """A good schema-2 dict merged over the empty state (grill G1-28)."""
    state = _empty_state()
    state.update(data)
    if not isinstance(state.get("stacks"), dict):
        state["stacks"] = {}
    if not isinstance(state.get("session"), dict):
        state["session"] = {}
    state["stacks"] = {
        str(k): v for k, v in state["stacks"].items() if isinstance(v, dict)
    }
    return state


def _norm(p) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(str(p))))


def _strings_in(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings_in(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings_in(v)


def _infer_root(state) -> Optional[str]:
    """The workspace a state was written in, for a file older than the
    ``workspace_root`` key: the part of a RECORD's stored absolute path
    before its ``artifacts/step4`` folder -- Step 4's own outputs only.  A
    model path is never used: an imported model may live in another
    workspace's artifacts (live driver B0-1b)."""
    stacks = state.get("stacks") if isinstance(state, dict) else None
    if not isinstance(stacks, dict):
        return None
    for rec in stacks.values():
        if not isinstance(rec, dict):
            continue
        for key, val in rec.items():
            if key in ("made_with", "predicted_with") or not isinstance(
                    val, str) or not os.path.isabs(val):
                continue
            parts = Path(val).parts
            for i in range(1, len(parts) - 1):
                if parts[i] == "artifacts" and parts[i + 1] == "step4":
                    return str(Path(*parts[:i]))
    return None


def reroot_paths(obj, old_root: str, new_root: str):
    """``(new_obj, n)``: every absolute path string under *old_root*
    rewritten under *new_root* (the rest kept as it is)."""
    old = _norm(old_root)
    count = [0]

    def walk(x):
        if isinstance(x, str):
            if os.path.isabs(x):
                n = _norm(x)
                if n == old or n.startswith(old + os.sep):
                    rel = os.path.relpath(os.path.abspath(x),
                                          os.path.abspath(old_root))
                    count[0] += 1
                    return str(Path(new_root) / rel) if rel != "." \
                        else str(new_root)
            return x
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [walk(v) for v in x]
        return x

    return walk(obj), count[0]


def _model_label(path: str) -> str:
    try:
        return Path(str(path)).name or str(path)
    except Exception:      # noqa: BLE001
        return str(path)


def _external_model(lateral: str, axial: str) -> Optional[Dict[str, Any]]:
    if not lateral:
        return None
    return {
        "lateral": lateral, "axial": axial or lateral,
        "lateral_label": _model_label(lateral),
        "axial_label": _model_label(axial or lateral),
        "kind": "external",
        "run_id": None, "bundle": None,
        "lateral_rel": None, "axial_rel": None,
    }


def migrate_v1_to_v2(data: Dict[str, Any]) -> Dict[str, Any]:
    """Schema 1 → schema 2, in memory and lossless.

    The input is not mutated.  Every key the table does not name is kept
    as it is -- including schema 1's session keys, which today's widgets
    still read until brick B1a rebuilds the panel.
    """
    out = _empty_state()
    src = copy.deepcopy(data) if isinstance(data, dict) else {}
    for key, val in src.items():
        if key not in ("schema_version", "stacks", "session"):
            out[key] = val
    out["schema_version"] = _SCHEMA_VERSION

    session = dict(src.get("session") or {})
    folder = session.get("input_stacks_dir") or ""
    session["stack_folders"] = [folder] if folder else []
    mode = session.get("model_source")
    active = None
    try:
        mode = int(mode) if mode is not None else None
    except (TypeError, ValueError):
        mode = None
    if mode == 1:
        active = _external_model(session.get("lateral_model_path", "") or "",
                                 session.get("axial_model_path", "") or "")
    elif mode == 0:
        pack = session.get("model_pack_path") or ""
        if pack and Path(pack).is_file():
            payload, ok = _read_json(Path(pack))
            if ok and isinstance(payload, dict):
                lat = str(payload.get("lateral_model") or "")
                ax = str(payload.get("axial_model") or "")
                if lat and ax and Path(lat).exists() and Path(ax).exists():
                    active = _external_model(lat, ax)
    # mode 2 (Pretrained): the name was never saved -> no model.
    session["active_model"] = active
    out["session"] = session

    stacks = {}
    for name, rec in (src.get("stacks") or {}).items():
        if not isinstance(rec, dict):
            continue
        rec = dict(rec)
        status = rec.get("status")
        if status == "segmented":
            # A Reconstruct started and never finished.  Infer the status
            # from what the record holds (audit D7): an earlier result is
            # kept; only a record with nothing later becomes "failed".
            rec["failed_stage"] = "reconstruct"
            if rec.get("qc_features_csv"):
                rec["status"] = "qc_extracted"
            elif rec.get("labels_path"):
                rec["status"] = "reconstructed"
            else:
                rec["status"] = "failed"
        elif status == "smoothed":
            rec["status"] = "qc_extracted"
        stacks[str(name)] = rec
    out["stacks"] = stacks
    out["active_stack"] = src.get("active_stack")
    out["last_updated"] = src.get("last_updated")
    return out


def _quarantine(path: Path) -> Optional[str]:
    """Rename an unreadable state file aside; its new name, or None."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = f"{path}.corrupt-{stamp}"
    n = 1
    while os.path.exists(target):
        target = f"{path}.corrupt-{stamp}-{n}"
        n += 1
    try:
        os.replace(str(path), target)
    except OSError:
        return None
    return os.path.basename(target)


class StackStateManager:
    """Manages per-stack workflow state with disk persistence."""

    STATE_FILENAME = "step4_state.json"
    LEGACY_STATE_FILENAME = LEGACY_STATE_FILENAME

    def __init__(self) -> None:
        self._state: Dict[str, Any] = _empty_state()
        self._workspace: str = ""
        #: Every save refuses while True (an unreadable file could not be
        #: set aside, or an adopt saw one and has not re-loaded yet).
        self.save_blocked: bool = False
        #: An adopt saw an unreadable file: the next user action that saves
        #: must call ``load(ws)`` (write mode) first, which sets it aside.
        self.needs_write_load: bool = False
        #: Where the in-memory state came from: new / backup / legacy / empty.
        self.loaded_from: str = "empty"
        self._notices: List[str] = []
        self._logs: List[str] = []

    # ------------------------------------------------------------------
    # Messages for the controller (drained)
    # ------------------------------------------------------------------

    def take_notices(self) -> List[str]:
        """User-facing notices from the last load (drained)."""
        out, self._notices = self._notices, []
        return out

    def take_logs(self) -> List[str]:
        """Log-only lines from the last load (drained)."""
        out, self._logs = self._logs, []
        return out

    # ------------------------------------------------------------------
    # Load / save
    # ------------------------------------------------------------------

    def load(self, workspace: str, *, write: bool = True) -> Dict[str, Any]:
        """Load the state of *workspace*, then re-root it (a COPIED
        workspace, live driver 2026-09-29): every stored path under the
        workspace the state was written in moves under this one.  A write
        load saves the re-rooted state at once; an adopt keeps it in memory
        only."""
        self._pending_move = None
        state = self._load_file(workspace, write=write)
        # Re-root BEFORE anything is saved: a save stamps the current
        # workspace as workspace_root (live driver: the move to the root
        # used to save first, so the copy's paths were never re-rooted).
        n = 0
        old = self._state.get("workspace_root") or _infer_root(self._state)
        if old and _norm(old) != _norm(workspace):
            self._state, n = reroot_paths(self._state, old, workspace)
            self._state["workspace_root"] = str(workspace)
            if n:
                self._logs.append(T.STATE_REROOTED.format(n=n, old=old))
                print(f"[STEP4-STATE] re-rooted {n} paths from {old} to "
                      f"{workspace}")
            state = self._state
        pending = self._pending_move
        self._pending_move = None
        if pending is not None and write:
            self._move_to_root(pending)       # saves the re-rooted state
        elif n and write and not self.save_blocked:
            try:
                self.save()
            except Exception as exc:      # noqa: BLE001
                print(f"[STEP4-STATE] re-rooted state not saved: {exc}")
        return state

    def _load_file(self, workspace: str, *, write: bool = True
                   ) -> Dict[str, Any]:
        """Load the state of *workspace*.

        ``write=False`` is the shared-workspace ADOPT path: nothing is
        renamed or written; an unreadable file leaves an empty state with
        ``save_blocked`` and ``needs_write_load`` set.
        """
        self._workspace = workspace
        self.save_blocked = False
        self.needs_write_load = False
        self._notices = []
        self._logs = []
        path = step4_state_path(workspace)
        old = step4_artifacts_state_path(workspace)
        if not path.exists() and old.exists():
            path = old                      # the earlier place: moved below
        if path.exists():
            data, ok = _read_json(path)
            if ok and _good_v2(data):
                self._state = _merge_over_empty(data)
                self.loaded_from = "new"
                self._check_legacy_newer()
                if path == old and write:
                    self._pending_move = old  # moved by load(), re-rooted
                return self._state
            if not write:
                self._state = _empty_state()
                self.loaded_from = "empty"
                self.save_blocked = True
                self.needs_write_load = True
                self._logs.append(T.STATE_UNREADABLE_ON_ADOPT)
                print(f"[STEP4-STATE] adopt: {path} unreadable; nothing renamed")
                return self._state
            saved_as = _quarantine(path)
            print(f"[STEP4-STATE] {path.name} unreadable -> "
                  f"{saved_as or 'RENAME FAILED'}")
            if saved_as is None:
                self.save_blocked = True
                self._notices.append(T.STATE_SAVE_BLOCKED)
            self._fallback_after_bad_file(saved_as)
            return self._state
        # No new file: the legacy file, read-only, migrated in memory.
        state = self._read_legacy()
        if state is not None:
            self._state = state
            self.loaded_from = "legacy"
        else:
            self._state = _empty_state()
            self.loaded_from = "empty"
        return self._state

    def _move_to_root(self, old: Path) -> None:
        """Write the state at the root, then rename the old copy aside
        (``.moved-<stamp>``).  A failed write leaves the old file where it
        was, so nothing is ever lost."""
        try:
            if not self.save():
                return
        except Exception as exc:      # noqa: BLE001
            print(f"[STEP4-STATE] move to the root failed: {exc}; "
                  f"{old} kept")
            return
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        target = f"{old}.moved-{stamp}"
        n = 1
        while os.path.exists(target):
            target = f"{old}.moved-{stamp}-{n}"
            n += 1
        try:
            os.replace(str(old), target)
        except OSError as exc:
            print(f"[STEP4-STATE] old state not renamed aside: {exc}")
            return
        self._logs.append(T.STATE_MOVED_TO_ROOT.format(
            file=os.path.basename(target)))
        print(f"[STEP4-STATE] state moved to the workspace root; the old "
              f"copy is {os.path.basename(target)}")

    def _fallback_after_bad_file(self, saved_as: Optional[str]) -> None:
        name = saved_as or self.STATE_FILENAME
        bak = step4_backup_path(self._workspace)
        if bak.exists():
            data, ok = _read_json(bak)
            if ok and _good_v2(data):
                self._state = _merge_over_empty(data)
                self.loaded_from = "backup"
                self._notices.append(
                    T.STATE_RESTORED_FROM_BACKUP.format(file=name))
                return
        state = self._read_legacy()
        if state is not None:
            self._state = state
            self.loaded_from = "legacy"
            self._notices.append(T.STATE_RESTORED_FROM_LEGACY.format(file=name))
            return
        self._state = _empty_state()
        self.loaded_from = "empty"
        self._notices.append(T.STATE_STARTED_EMPTY.format(file=name))

    def _read_legacy(self) -> Optional[Dict[str, Any]]:
        legacy = legacy_state_path(self._workspace)
        if not legacy.exists():
            return None
        data, ok = _read_json(legacy)
        if not ok or not isinstance(data, dict) or (
                data.get("schema_version") != _LEGACY_SCHEMA_VERSION):
            self._logs.append(T.STATE_LEGACY_UNREADABLE)
            print(f"[STEP4-STATE] legacy {legacy.name} unreadable; left as is")
            return None
        state = migrate_v1_to_v2(data)
        state["migrated_from_legacy_at"] = _now()
        print(f"[STEP4-STATE] migrated legacy {legacy.name} in memory "
              f"({len(state['stacks'])} stacks)")
        return state

    def _check_legacy_newer(self) -> None:
        """Name, once, a legacy file an older build wrote after migration."""
        legacy = legacy_state_path(self._workspace)
        if not legacy.exists():
            return
        data, ok = _read_json(legacy)
        if not ok or not isinstance(data, dict):
            return
        stamp = str(data.get("last_updated") or "")
        since = str(self._state.get("migrated_from_legacy_at") or "")
        if not stamp or stamp <= since:
            return
        if self._state.get("legacy_newer_noticed") == stamp:
            return
        self._state["legacy_newer_noticed"] = stamp
        self._notices.append(T.STATE_LEGACY_NEWER)
        print(f"[STEP4-STATE] legacy file newer ({stamp} > {since})")

    def save(self, workspace: Optional[str] = None) -> bool:
        """Atomic write (mkstemp + fsync + ``os.replace``) with ``.bak``.

        Returns False, writing nothing, while ``save_blocked`` or without a
        workspace.  A failed write raises and leaves the previous file.
        """
        ws = workspace or self._workspace
        if not ws:
            return False
        if self.save_blocked:
            print("[STEP4-STATE] save refused: save_blocked")
            return False
        path = step4_state_path(ws)
        base = path.parent
        base.mkdir(parents=True, exist_ok=True)
        if path.exists():
            # The backup only ever holds a file that parses (audit D12).
            current, ok = _read_json(path)
            if ok and _good_v2(current):
                try:
                    bak = step4_backup_path(ws)
                    bak.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(path), str(bak))
                except OSError:
                    pass                      # rotation is best-effort

        self._state["schema_version"] = _SCHEMA_VERSION
        self._state["last_updated"] = _now()
        # the workspace these paths belong to, for the re-root on a copy
        self._state["workspace_root"] = str(ws)

        fd, tmp_path = tempfile.mkstemp(
            suffix=".tmp", prefix=".step4_state_", dir=str(base),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._state, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, str(path))
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        return True

    def flush(self) -> bool:
        """Immediate save."""
        return self.save()

    # ------------------------------------------------------------------
    # Stack status accessors
    # ------------------------------------------------------------------

    def get_stack_status(self, stack_name: str) -> str:
        """Return status for a stack, default 'unprocessed'."""
        record = self._state["stacks"].get(stack_name)
        if record is None:
            return "unprocessed"
        return record.get("status", "unprocessed")

    def set_stack_status(
        self, stack_name: str, status: str, **metadata: Any
    ) -> None:
        """Update status and optional metadata."""
        if status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{status}', expected one of {VALID_STATUSES}"
            )

        stacks = self._state["stacks"]
        if stack_name not in stacks:
            stacks[stack_name] = {"status": "unprocessed"}

        record = stacks[stack_name]
        record["status"] = status
        record["updated_at"] = _now()
        for key, val in metadata.items():
            record[key] = val

    def update_record(self, stack_name: str, **fields: Any) -> None:
        """Set fields on a record, keeping its status."""
        self.set_stack_status(stack_name, self.get_stack_status(stack_name),
                              **fields)

    def get_record(self, stack_name: str) -> Dict[str, Any]:
        """A deep copy of the stack's record ({} when absent)."""
        rec = self._state["stacks"].get(stack_name)
        return copy.deepcopy(rec) if isinstance(rec, dict) else {}

    def restore_record(self, stack_name: str,
                       record: Optional[Dict[str, Any]]) -> None:
        """Put back a record taken with ``get_record`` (a failed save must
        not leave the in-memory record changed, audit D5); an empty or
        ``None`` record removes the stack."""
        if record:
            self._state["stacks"][stack_name] = copy.deepcopy(record)
        else:
            self._state["stacks"].pop(stack_name, None)

    def reconcile_with_disk(self, *, cache_dir, manifests) -> Dict[str, str]:
        """Bring every record in line with the files on disk (in memory;
        nothing is deleted).  The backup is always one save behind a
        delete, so a restored state can point at files that are gone."""
        from ._step4_models import reconcile_records
        return reconcile_records(self._state["stacks"],
                                 measure_keys=_QC_METADATA_KEYS,
                                 cache_dir=cache_dir, manifests=manifests)

    def clear_stack_keys(self, stack_name: str, keys: Iterable[str]) -> None:
        rec = self._state["stacks"].get(stack_name)
        if not isinstance(rec, dict):
            return
        for key in keys:
            rec.pop(key, None)

    def get_status_counts(self) -> Dict[str, int]:
        """Return {status: count} for dashboard display."""
        counts = {s: 0 for s in VALID_STATUSES}
        for record in self._state["stacks"].values():
            st = record.get("status", "unprocessed")
            if st in counts:
                counts[st] += 1
        return counts

    def get_all_stack_statuses(self) -> Dict[str, str]:
        """Return {stack_name: status} for list coloring."""
        return {
            name: rec.get("status", "unprocessed")
            for name, rec in self._state["stacks"].items()
        }

    def register_stacks(self, stack_names: list) -> None:
        """Ensure all stack names have records without overwriting existing."""
        stacks = self._state["stacks"]
        for name in stack_names:
            if name not in stacks:
                stacks[name] = {"status": "unprocessed"}

    # ------------------------------------------------------------------
    # Active stack, ticks and model
    # ------------------------------------------------------------------

    @property
    def active_stack(self) -> Optional[str]:
        return self._state.get("active_stack")

    @active_stack.setter
    def active_stack(self, stack_name: Optional[str]) -> None:
        self._state["active_stack"] = stack_name

    def set_ticked(self, names: Iterable[str]) -> None:
        self._state.setdefault("session", {})["ticked"] = [str(n) for n in names]

    def get_ticked(self) -> List[str]:
        return list(self._state.get("session", {}).get("ticked") or [])

    def set_active_model(self, snapshot: Optional[Dict[str, Any]]) -> None:
        self._state.setdefault("session", {})["active_model"] = (
            copy.deepcopy(snapshot) if snapshot else None)

    def get_active_model(self) -> Optional[Dict[str, Any]]:
        snap = self._state.get("session", {}).get("active_model")
        return copy.deepcopy(snap) if isinstance(snap, dict) else None

    # ------------------------------------------------------------------
    # Backward transitions
    # ------------------------------------------------------------------

    def reset_to_unprocessed(self, stack_name: str) -> None:
        """Reset stack to unprocessed, clearing all metadata."""
        stacks = self._state["stacks"]
        stacks[stack_name] = {"status": "unprocessed"}

    def reset_to_reconstructed(self, stack_name: str) -> None:
        """Reset to reconstructed, clearing every measure key (F5)."""
        stacks = self._state["stacks"]
        if stack_name not in stacks:
            stacks[stack_name] = {"status": "reconstructed"}
            return
        record = stacks[stack_name]
        record["status"] = "reconstructed"
        record["updated_at"] = _now()
        for key in _QC_METADATA_KEYS:
            record.pop(key, None)

    # ------------------------------------------------------------------
    # Session metadata (workspace-level, not per-stack)
    # ------------------------------------------------------------------

    def get_session_metadata(self) -> Dict[str, Any]:
        """Return session metadata (model paths, input dir, etc.)."""
        return dict(self._state.get("session", {}))

    def set_session_metadata(self, **kwargs: Any) -> None:
        """Update session metadata (merged, not replaced)."""
        session = self._state.setdefault("session", {})
        session.update(kwargs)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_state() -> Dict[str, Any]:
        return _empty_state()
