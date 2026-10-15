"""Per-patch reconstruction workflow state persistence.

Tracks each patch through the reconstruction pipeline:
    (unprocessed) → predicted → reconstructed → curating → accepted → finalized

State is persisted to ``{workspace}/reconstruction_state.json`` using atomic
writes (tempfile + os.replace) and debounced saves via QTimer, matching the
pattern established in ``packages.epicycle_io.state_store``.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from qtpy.QtCore import QTimer

from ..data_preparation import _voxel_spacing as VS
from ..data_preparation._voxel_spacing import SPACING_SOURCES, _positive
from . import _migrate_step3_runs
from . import _paths

# Valid status values in pipeline order
VALID_STATUSES = (
    "unprocessed",
    "predicted",
    "reconstructed",
    "curating",
    "accepted",
    "finalized",
)

# Schema 3 == the runs-envelope on disk (mirrors Step 2's v2→v3 bump).
# IMPORTANT: the INTERNAL ``self._state`` stays the FLAT shape — every
# existing accessor (set_patch_status / get_all_patch_statuses / model
# getters / downsample getters …) reads the flat dict untouched.  The v3
# envelope only exists on disk; ``load`` unwraps it into the flat shape
# and ``save`` wraps the flat shape back, so the accessors never see it.
_SCHEMA_VERSION = 3


class ReconstructionStateManager:
    """Manages per-patch workflow state with disk persistence."""

    STATE_FILENAME = _paths.STEP3_STATE_FILENAME

    def __init__(self) -> None:
        self._state: Dict[str, Any] = self._empty_state()
        self._workspace: str = ""
        self._save_timer: Optional[QTimer] = None
        # Which run-context the flat ``self._state`` currently mirrors.
        # Defaults to the single-context ``"default"`` run so a fresh /
        # un-migrated workspace behaves exactly as before.  NEVER hardcode
        # "default" in a write path — always thread this attribute.
        self.active_run_id: str = _migrate_step3_runs.DEFAULT_RUN_ID
        # while True, this manager writes
        # NOTHING.  Set for the duration of Step 3's `adopt_workspace` on a
        # workspace whose state file predates the v3 envelope: `load` upgrades
        # v1/v2 -> v3 in memory and the next `save` would rewrite the file with
        # no backup, so a user who merely walked into the step would have had
        # their old state silently converted.  Only an explicit Browse does it.
        self.suppress_save: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, workspace: str) -> Dict[str, Any]:
        """Load state from workspace directory.

        Accepts the flat legacy shapes (``schema_version`` absent / 1 / 2)
        AND the v3 runs-envelope.  Flat input is wrapped → unwrapped for
        the active run (upgrade-on-load); a v3 envelope is unwrapped
        directly.  The result is merged over :meth:`_empty_state` so the
        flat ``patches`` map always exists.  Tolerates corrupt JSON by
        falling back to empty state.

        ``self._state`` always ends up in the FLAT shape (the internal
        contract every accessor relies on) for the run identified by
        ``self.active_run_id``.
        """
        self._workspace = workspace
        self._spacing_stack_dir = ""
        self._spacing_migrated = False
        self._spacing_cache = None
        path = Path(workspace) / self.STATE_FILENAME
        data = None
        if path.exists():
            try:
                raw = path.read_text(encoding="utf-8")
                data = json.loads(raw)
            except (json.JSONDecodeError, OSError):
                data = None
        # Invariant I4 (B2): a BLANK `active_run_id` means "whatever this
        # workspace says".  Callers switching workspaces blank it first, so
        # the previous workspace's run can no longer decide which run of the
        # NEW file is unwrapped (and, worse, which run a later save writes).
        if not self.active_run_id:
            peeked = data.get("active_run_id") if isinstance(data, dict) else None
            self.active_run_id = (
                str(peeked) if peeked else _migrate_step3_runs.DEFAULT_RUN_ID)
        if path.exists():
            self._state = self._flat_from_disk_data(data)
        else:
            self._state = self._empty_state()
        return self._state

    def _flat_from_disk_data(self, data: Any) -> Dict[str, Any]:
        """Convert parsed on-disk JSON into a flat state dict for the
        active run, merged over :meth:`_empty_state`.

        Returns a fresh ``_empty_state`` for anything that isn't a usable
        dict (corrupt / unexpected shape).
        """
        if not isinstance(data, dict):
            return self._empty_state()

        schema = data.get("schema_version")
        try:
            if schema == 3:
                flat = _migrate_step3_runs.unwrap_step3_state_v3(
                    data, self.active_run_id
                )
            elif schema in (None, 1, 2):
                envelope = _migrate_step3_runs.wrap_step3_state_to_v3(
                    data, active_run_id=self.active_run_id
                )
                flat = _migrate_step3_runs.unwrap_step3_state_v3(
                    envelope, self.active_run_id
                )
            else:
                # Unknown future schema — don't guess; start empty.
                return self._empty_state()
        except (ValueError, TypeError):
            return self._empty_state()

        merged = self._empty_state()
        merged.update(flat)
        return merged

    def save(self, workspace: Optional[str] = None) -> None:
        """Wrap the flat state into the v3 runs-envelope + atomic write.

        The flat ``self._state`` is wrapped into ``runs[active_run_id]``,
        then :func:`merge_forward_runs` carries every OTHER run (and the
        active run's manifest fields) forward from the existing on-disk
        file so a per-run save never clobbers a sibling run.  Atomic write
        via tempfile + os.replace.
        """
        if self.suppress_save:
            return
        ws = workspace or self._workspace
        if not ws:
            return
        base = Path(ws)
        base.mkdir(parents=True, exist_ok=True)
        path = base / self.STATE_FILENAME

        self._state["last_updated"] = datetime.now(timezone.utc).isoformat()

        # Normalize the flat shape for the wrap (it rejects schema >= 3,
        # but the internal flat state carries schema 3 — pop it so the wrap
        # treats it as flat and re-stamps the envelope to 3).
        flat_for_wrap = dict(self._state)
        flat_for_wrap.pop("schema_version", None)

        envelope = _migrate_step3_runs.wrap_step3_state_to_v3(
            flat_for_wrap, active_run_id=self.active_run_id
        )
        existing = self._read_existing_envelope(base)
        envelope = _migrate_step3_runs.merge_forward_runs(
            envelope, existing, self.active_run_id
        )

        self._atomic_write_json(base, path, envelope)

    @staticmethod
    def _atomic_write_json(base: Path, path: Path, payload: Dict[str, Any]) -> None:
        """tempfile → os.replace atomic JSON write (shared by save/create_run)."""
        fd, tmp_path = tempfile.mkstemp(
            suffix=".tmp", prefix=".recon_state_", dir=str(base),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_path, str(path))
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _read_existing_envelope(self, base: Path) -> Optional[Dict[str, Any]]:
        """Read + parse the current on-disk state file, or None.

        Tolerates a missing file and corrupt/non-dict JSON (→ None) so
        callers can pass the result straight to :func:`merge_forward_runs`.
        """
        path = base / self.STATE_FILENAME
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return data if isinstance(data, dict) else None

    # ------------------------------------------------------------------
    # Run-aware API (Phase 1a — state primitives; run lifecycle/dialogs
    # are Phase 2).  The flat ``self._state`` shape is unchanged; these
    # methods only manage which run-context it mirrors + the on-disk
    # multi-run envelope.
    # ------------------------------------------------------------------

    def get_active_run_id(self) -> str:
        """Return the run-context the flat ``self._state`` currently mirrors."""
        return self.active_run_id

    def switch_active_run(
        self, run_id: str, workspace: Optional[str] = None
    ) -> None:
        """Persist the current run, then load ``run_id`` into ``self._state``.

        Steps (order matters):

        1. ``flush()`` any pending debounced save FIRST — BEFORE mutating
           ``active_run_id`` — so the timer can't write the current run's
           edits into the NEW run's slot (the Step 2 timer-clobber gotcha).
        2. ``save()`` the current run (merge-forward-safe).
        3. Set ``self.active_run_id = run_id``.
        4. Reload ``self._state`` from the on-disk envelope's
           ``runs[run_id]``; a run with no on-disk entry starts from
           :meth:`_empty_state` (fresh run = empty patches).

        Never raises on a missing run.
        """
        ws = workspace or self._workspace
        # 1. Flush BEFORE mutating active_run_id (cross-run clobber guard).
        self.flush()
        # 2. Persist the current run.
        self.save(ws)
        # 3. Switch identity.
        self.active_run_id = run_id
        # 4. Reload the target run's flat view from disk (empty if absent).
        base = Path(ws) if ws else None
        existing = self._read_existing_envelope(base) if base is not None else None
        runs = existing.get("runs") if isinstance(existing, dict) else None
        if isinstance(runs, dict) and run_id in runs:
            self._state = self._flat_from_disk_data(existing)
        else:
            # A run with no entry starts empty, but EVERY workspace-level key
            # comes with it (``patch_spacing``, ``default_run_nested``,
            # ``downsample_mode``, ``loaded_stack_dir``...; B MED-3): the
            # next save would otherwise write them away.  Read from the
            # on-disk envelope just saved; from memory when nothing was
            # written (no file, or a read-only adopt).
            source = existing if (isinstance(existing, dict) and not
                                  getattr(self, "suppress_save", False)) \
                else self._state
            carried = {k: source[k] for k in self._WORKSPACE_KEYS_CARRIED
                       if k in source}
            self._state = self._empty_state()
            self._state.update(carried)

    def list_run_ids(self, workspace: Optional[str] = None) -> list:
        """Return the sorted run ids in the on-disk envelope.

        A flat / absent / corrupt file yields ``[DEFAULT_RUN_ID]`` (the
        implicit single context).
        """
        ws = workspace or self._workspace
        base = Path(ws) if ws else None
        existing = self._read_existing_envelope(base) if base is not None else None
        if isinstance(existing, dict):
            runs = existing.get("runs")
            if isinstance(runs, dict) and runs:
                return sorted(runs.keys())
        return [_migrate_step3_runs.DEFAULT_RUN_ID]

    def has_run(self, run_id: str, workspace: Optional[str] = None) -> bool:
        """True if ``run_id`` exists in the on-disk envelope."""
        return run_id in self.list_run_ids(workspace)

    def create_run(
        self, run_id: str, workspace: Optional[str] = None, **manifest: Any
    ) -> None:
        """Add an empty run entry to the on-disk envelope.

        Adds ``runs[run_id] = {patches:{}, display_name, target,
        created_from, **extra}`` WITHOUT switching to it and WITHOUT
        touching ``self.active_run_id``.  Merge-forward-safe (other runs +
        the active run's current state are preserved) + atomic write.  A
        no-op for the entry if ``run_id`` already exists (existing entry
        left intact).  Run lifecycle / dialogs are Phase 2 — this is just
        the state primitive.
        """
        ws = workspace or self._workspace
        if not ws:
            return
        base = Path(ws)
        base.mkdir(parents=True, exist_ok=True)
        path = base / self.STATE_FILENAME

        existing = self._read_existing_envelope(base)

        # Build the active run's envelope from the current in-memory flat
        # state, then carry every other run forward (no-clobber).
        flat_for_wrap = dict(self._state)
        flat_for_wrap.pop("schema_version", None)
        envelope = _migrate_step3_runs.wrap_step3_state_to_v3(
            flat_for_wrap, active_run_id=self.active_run_id
        )
        envelope = _migrate_step3_runs.merge_forward_runs(
            envelope, existing, self.active_run_id
        )

        runs = envelope.setdefault("runs", {})
        if run_id not in runs:
            entry: Dict[str, Any] = {
                "patches": {},
                "display_name": manifest.pop("display_name", run_id),
                "target": manifest.pop("target", None),
                "created_from": manifest.pop("created_from", None),
            }
            entry.update(manifest)
            runs[run_id] = entry

        self._atomic_write_json(base, path, envelope)

    def delete_run(self, run_id: str, workspace: Optional[str] = None) -> None:
        """Drop ``runs[run_id]`` from the on-disk v3 envelope.  DESTRUCTIVE.

        The STATE-side counterpart to
        ``_run_management.delete_run_from_disk`` (which removes the run's
        context DIRECTORY).  The controller calls both: dir + state.  Mirrors
        Step 2's ``remove_run_from_state_json`` semantics, merge-forward-safe
        + atomic.

        The active run's in-memory flat state is wrapped + every OTHER run is
        carried forward from disk, THEN the target entry is dropped, THEN the
        envelope is atomically written — so the delete never clobbers a
        sibling run's edits.

        Refuses (raises ``ValueError``):

        * ``run_id == DEFAULT_RUN_ID`` — the implicit FLAT default run has no
          standalone removable entry; dropping it would orphan every default
          patch.
        * ``run_id == self.active_run_id`` — can't delete the run currently
          mirrored into ``self._state``; the caller must switch off it first.

        A ``run_id`` with no on-disk entry is a silent no-op (idempotent).
        Returns without writing when there is no workspace / no state file.
        """
        if run_id == _migrate_step3_runs.DEFAULT_RUN_ID:
            raise ValueError(
                "refusing to delete the default run from state — it is the "
                "implicit flat layout, not a removable named-run entry"
            )
        if run_id == self.active_run_id:
            raise ValueError(
                f"refusing to delete the active run {run_id!r} from state — "
                f"switch to a different run first"
            )
        ws = workspace or self._workspace
        if not ws:
            return
        base = Path(ws)
        path = base / self.STATE_FILENAME
        if not path.exists():
            return

        existing = self._read_existing_envelope(base)

        # Build the active run's envelope from the current in-memory flat
        # state, carry every OTHER run forward (no-clobber), THEN drop target.
        flat_for_wrap = dict(self._state)
        flat_for_wrap.pop("schema_version", None)
        envelope = _migrate_step3_runs.wrap_step3_state_to_v3(
            flat_for_wrap, active_run_id=self.active_run_id
        )
        envelope = _migrate_step3_runs.merge_forward_runs(
            envelope, existing, self.active_run_id
        )

        runs = envelope.get("runs")
        if isinstance(runs, dict) and run_id in runs:
            del runs[run_id]

        # Defensive: never leave the active pointer aimed at a deleted run.
        # (We refuse to delete the active run above, so this can only fire if
        # the on-disk active_run_id disagrees with self.active_run_id.)
        if envelope.get("active_run_id") == run_id:
            envelope["active_run_id"] = self.active_run_id

        self._atomic_write_json(base, path, envelope)

    def schedule_save(self, delay_ms: int = 500) -> None:
        """Debounced save — coalesces rapid calls.

        A no-op while :attr:`suppress_save` is set, so a read-only adopt
        cannot arm a timer that fires into the workspace a moment later.
        """
        if self.suppress_save:
            return
        if self._save_timer is not None:
            try:
                self._save_timer.start(delay_ms)
                return
            except RuntimeError:
                # Qt deleted the parentless timer's C++ side: build again
                self._save_timer = None
        self._save_timer = QTimer()
        self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(self._on_save_timer)
        self._save_timer.start(delay_ms)

    def _on_save_timer(self) -> None:
        self.save()

    def flush(self) -> None:
        """Immediate save if a debounced save is pending.

        The timer is a parentless ``QTimer``: at napari close Qt can delete
        its C++ side first, and ``isActive()`` then raises RuntimeError
        (Step 2's twin, ``779c6eec``).  A dead timer can no longer fire, so
        a save it held would be lost: save anyway and drop the timer."""
        timer = self._save_timer
        if timer is None:
            return
        try:
            pending = timer.isActive()
            if pending:
                timer.stop()
        except RuntimeError:
            self._save_timer = None
            pending = True
        if pending:
            self.save()

    # ------------------------------------------------------------------
    # Patch status accessors
    # ------------------------------------------------------------------

    def get_patch_status(self, patch_name: str) -> str:
        """Return status for a patch, default 'unprocessed'."""
        record = self._state["patches"].get(patch_name)
        if record is None:
            return "unprocessed"
        return record.get("status", "unprocessed")

    def set_patch_status(self, patch_name: str, status: str, **metadata: Any) -> None:
        """Update status and optional metadata, schedule debounced save."""
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status '{status}', expected one of {VALID_STATUSES}")

        patches = self._state["patches"]
        if patch_name not in patches:
            patches[patch_name] = {"status": "unprocessed"}

        record = patches[patch_name]
        record["status"] = status
        record["updated_at"] = datetime.now(timezone.utc).isoformat()
        for key, val in metadata.items():
            record[key] = val

        self.schedule_save()

    def get_status_counts(self) -> Dict[str, int]:
        """Return {status: count} for dashboard display."""
        counts = {s: 0 for s in VALID_STATUSES}
        for record in self._state["patches"].values():
            st = record.get("status", "unprocessed")
            if st in counts:
                counts[st] += 1
        return counts

    def get_most_recent_activity(self):
        """Return ``(updated_at_iso, patch_name)`` for the most-recently
        touched patch in the workspace, or ``None`` if the workspace
        has no patches with timestamps.

        Used by the Session section's "Last activity" display so
        returning users can see how fresh the workspace is.  Reads the
        ``updated_at`` field that ``set_patch_status`` /
        ``update_patch_metadata`` write on every change.
        """
        latest_ts = None
        latest_name = None
        for name, rec in self._state["patches"].items():
            ts = rec.get("updated_at")
            if not ts:
                continue
            if latest_ts is None or ts > latest_ts:
                latest_ts = ts
                latest_name = name
        if latest_ts is None:
            return None
        return latest_ts, latest_name

    def get_resume_patch(self):
        """Return ``(patch_name, status)`` for the patch the user most
        likely wants to resume on, or ``None`` if there's nothing
        actionable.

        Priority order (first hit wins):
          1. Most recently touched ``curating`` patch — user is
             actively mid-curation.
          2. Most recently touched ``accepted`` patch — user has
             accepted but not finalized; finalize is the next step.
          3. Most recently touched ``reconstructed`` patch — user
             likely intends to curate it next.

        Patches in ``unprocessed`` / ``predicted`` / ``finalized``
        are NOT resume targets (no in-progress work to resume).
        """
        priority = ("curating", "accepted", "reconstructed")
        for target_status in priority:
            latest_ts = None
            latest_name = None
            for name, rec in self._state["patches"].items():
                if rec.get("status") != target_status:
                    continue
                ts = rec.get("updated_at")
                if not ts:
                    continue
                if latest_ts is None or ts > latest_ts:
                    latest_ts = ts
                    latest_name = name
            if latest_name is not None:
                return latest_name, target_status
        return None

    def get_all_patch_statuses(self) -> Dict[str, str]:
        """Return {patch_name: status} for list coloring."""
        return {
            name: rec.get("status", "unprocessed")
            for name, rec in self._state["patches"].items()
        }

    def register_patches(self, patch_names: list) -> None:
        """Ensure all patch names have records (without overwriting existing)."""
        patches = self._state["patches"]
        for name in patch_names:
            if name not in patches:
                patches[name] = {"status": "unprocessed"}
        self.schedule_save()

    def update_patch_metadata(self, patch_name: str, **metadata: Any) -> None:
        """Update patch metadata fields WITHOUT changing status.

        D2 (perf push 2026-05-09): the patch-load path needs to record
        the applied downsample factor on every load without re-asserting
        the status field.  ``set_patch_status`` requires a status arg
        (and validates it), which is awkward when the caller only
        wants to update one metadata field.

        Creates the patch record (with status='unprocessed') if missing,
        matching ``set_patch_status`` semantics.  Schedules debounced save.
        """
        patches = self._state["patches"]
        if patch_name not in patches:
            patches[patch_name] = {"status": "unprocessed"}

        record = patches[patch_name]
        record["updated_at"] = datetime.now(timezone.utc).isoformat()
        for key, val in metadata.items():
            record[key] = val

        self.schedule_save()

    # ------------------------------------------------------------------
    # Persistent model selection (2026-05-18)
    # ------------------------------------------------------------------
    #
    # Persists the user's last-active model selection so that on the
    # next session's workspace load the path widgets + status labels
    # restore to exactly what they had — no more re-picking every
    # session.  Three fields:
    #
    #   model_source_mode    "auto" / "import" / "pretrained" / None
    #   lateral_model_path   absolute path string OR pretrained name
    #   axial_model_path     absolute path string OR pretrained name
    #
    # Old workspaces (pre-2026-05-18 state files) lack these fields;
    # getters use `.get()` with None default so backwards compat is
    # automatic — no schema bump required.

    def get_model_selection(self) -> Dict[str, Any]:
        """Return the persisted model selection state.

        Always returns a dict (never None) so callers can dereference
        keys directly.  Missing keys default to None / "auto".
        """
        return {
            "mode": self._state.get("model_source_mode", "auto"),
            "lateral_path": self._state.get("lateral_model_path"),
            "axial_path": self._state.get("axial_model_path"),
        }

    def set_model_selection(self, *, mode: Optional[str] = None,
                            lateral_path: Optional[str] = None,
                            axial_path: Optional[str] = None) -> None:
        """Update persisted model selection.

        Any argument left as ``None`` (the default) leaves that field
        unchanged.  Pass ``""`` (empty string) explicitly to clear a
        path.  ``mode`` must be one of "auto" / "import" / "pretrained"
        if provided.
        """
        if mode is not None:
            if mode not in ("auto", "import", "pretrained"):
                raise ValueError(
                    f"Invalid model_source_mode {mode!r}; expected "
                    f"'auto' / 'import' / 'pretrained'."
                )
            self._state["model_source_mode"] = mode
        if lateral_path is not None:
            # Normalize empty string to None for cleaner JSON storage.
            self._state["lateral_model_path"] = lateral_path or None
        if axial_path is not None:
            self._state["axial_model_path"] = axial_path or None
        self.schedule_save()

    # ------------------------------------------------------------------
    # Start from
    # ------------------------------------------------------------------
    #
    # The settings column's preset, one key per RUN: an additive
    # ``start_from`` in the flat state, which the v3 wrap puts under
    # ``runs[active]`` because it is NOT in ``WORKSPACE_LEVEL_KEYS_V1``.
    # Values: "epicycle" / "useg3d" / "last_run" / "custom"; missing ->
    # "epicycle".  An old workspace has no key: no schema bump.

    START_FROM_KEYS = ("epicycle", "useg3d", "last_run", "custom")

    def get_start_from(self) -> str:
        """The run's Start from key; "epicycle" when missing or unknown."""
        key = self._state.get("start_from")
        return key if key in self.START_FROM_KEYS else "epicycle"

    def stored_start_from(self) -> Optional[str]:
        """The run's STORED Start from key, or None when the run never
        stored one (a run switch keeps the widgets then).
        """
        key = self._state.get("start_from")
        return key if key in self.START_FROM_KEYS else None

    def set_start_from(self, key: str) -> None:
        """Persist the run's Start from key (debounced save)."""
        if key not in self.START_FROM_KEYS:
            raise ValueError(f"unknown Start from key {key!r}")
        self._state["start_from"] = key
        self.schedule_save()

    # ------------------------------------------------------------------
    # Ticks
    # ------------------------------------------------------------------
    #
    # The patches Predict / Reconstruct run on, one list per RUN: an
    # additive ``ticked`` key in the flat state (per run by the v3 wrap, as
    # ``start_from``).  Missing -> nothing ticked.  No schema bump.

    def get_ticked(self) -> list:
        """The run's ticked patch names (sorted; non-strings dropped)."""
        raw = self._state.get("ticked") or []
        if not isinstance(raw, (list, tuple)):
            return []
        return sorted({n for n in raw if isinstance(n, str) and n})

    def set_ticked(self, names) -> None:
        """Persist the run's ticked patch names (debounced save); a list
        equal to the stored one schedules nothing."""
        new = sorted({str(n) for n in (names or ()) if n})
        if "ticked" in self._state and new == self.get_ticked():
            return
        self._state["ticked"] = new
        self.schedule_save()

    def forget_patch(self, patch_name: str) -> Optional[dict]:
        """Forget this patch: drop its record from the ACTIVE run
        and its tick; returns the record that was dropped (``None`` when it
        had none).  Other runs are untouched."""
        record = self._state["patches"].pop(patch_name, None)
        ticked = self.get_ticked()
        if patch_name in ticked:
            self._state["ticked"] = [n for n in ticked if n != patch_name]
        if self._state.get("active_patch") == patch_name:
            self._state["active_patch"] = None
        self.schedule_save()
        return record

    # ------------------------------------------------------------------
    # Per-patch voxel size
    # ------------------------------------------------------------------
    #
    # ``patch_spacing`` = {patch: {"zyx": [z, y, x], "source": "user" |
    # "step1" | "file"}} is WORKSPACE-LEVEL (in ``WORKSPACE_LEVEL_KEYS_V1``):
    # one value per patch for every run, at the envelope's top level.  A
    # ``user`` value is never overwritten by a ``step1`` / ``file``
    # re-resolve (the resolver's rule 1); only the user changes or clears
    # it.  ``measured_spacing`` -- the spacing an extraction used -- rides
    # the RUN's patch record with the features (Step 4's key).  Old files
    # have neither key: no schema bump.  Writes go through the debounced
    # save, which every switch / browse flushes first.

    #: workspace keys a switch to a run with no on-disk entry keeps: every
    #: workspace-level key but the envelope's own two
    _WORKSPACE_KEYS_CARRIED = tuple(sorted(
        _migrate_step3_runs.WORKSPACE_LEVEL_KEYS_V1
        - {"schema_version", "active_run_id"}))

    @staticmethod
    def _spacing_entry(raw: Any) -> Optional[Dict[str, Any]]:
        """A valid ``{"zyx", "source"}`` entry as a fresh dict, else None."""
        if not isinstance(raw, dict):
            return None
        source = raw.get("source")
        if source not in SPACING_SOURCES:
            return None
        zyx = _positive(raw.get("zyx") or ())
        if zyx is None:
            return None
        return {"zyx": zyx, "source": source}

    def _spacing_map(self) -> Dict[str, Any]:
        raw = self._state.get("patch_spacing")
        if not isinstance(raw, dict):
            raw = {}
            self._state["patch_spacing"] = raw
        return raw

    # --- S1: the workspace's patch_spacing.json backs these four -------
    #
    # With a workspace loaded the VALUE lives in the root file
    # ``patch_spacing.json`` (shared with Step 4; ``_voxel_spacing``).  The
    # envelope's ``patch_spacing`` map is still READ (the migration, a
    # manager with no workspace, a read-only adopt) and is still mirrored on
    # a write, so an older build opening the workspace sees the same values
    # (Step 3's own; never Step 4's --).

    #: set by the controller: a write with no stack folder known is refused
    #: (a bare patch name would be written as a key; audit L4).  A bare
    #: manager (the R7 tests) keeps name keys.
    require_stack_dir: bool = False
    #: the last file write that failed, for the caller to say on screen
    last_spacing_error: str = ""

    def set_spacing_stack_dir(self, stack_dir: Optional[str]) -> None:
        """The folder the listed patch NAMES live in (the controller's
        ``loaded_stack_dir``, which may be Step 1's unpersisted default);
        keys the file by workspace-relative TIFF path.  Runs the migration
        of the envelope's map."""
        self._spacing_stack_dir = str(stack_dir or "")
        self._spacing_dir_cache = None
        self._migrate_spacing_once()

    def _spacing_dir(self) -> str:
        """The folder patch names are keyed under, re-rooted onto this
        workspace when it is a COPY whose saved folder still names the
        original's (audit M4: Step 4 re-roots its keys the same way)."""
        d = (getattr(self, "_spacing_stack_dir", "")
             or str(self._state.get("loaded_stack_dir") or ""))
        if not d or not self._workspace:
            return d
        cached = getattr(self, "_spacing_dir_cache", None)
        if cached is not None and cached[0] == (d, self._workspace):
            return cached[1]
        out = d
        if VS._rel_to(d, self._workspace) is None:
            out = self._reroot_dir(d)
        self._spacing_dir_cache = ((d, self._workspace), out)
        return out

    def _reroot_dir(self, d: str) -> str:
        """*d* moved under this workspace by its longest tail that exists
        here AND holds a file of the same name as one in *d*; else *d*."""
        try:
            parts = [p for p in Path(d).parts if p not in ("\\", "/")
                     and not p.endswith(":\\") and not p.endswith(":/")]
            there = set(os.listdir(d)) if os.path.isdir(d) else None
        except OSError:
            return d
        for i in range(len(parts)):
            cand = Path(self._workspace).joinpath(*parts[i:])
            if not cand.is_dir():
                continue
            try:
                here = set(os.listdir(cand))
            except OSError:
                continue
            if there is None or (there & here):
                return str(cand)
        return d

    def _spacing_key(self, patch_name: str) -> str:
        d = self._spacing_dir()
        if not d or not self._workspace:
            return str(patch_name)
        return VS.patch_spacing_key(os.path.join(d, str(patch_name)),
                                    self._workspace)

    def _file_entries(self) -> Dict[str, Dict[str, Any]]:
        """The file's entries, re-read only when its stat changed."""
        path = VS.patch_spacing_path(self._workspace)
        try:
            st = path.stat()
            sig = (str(path), st.st_mtime_ns, st.st_size,
                   getattr(st, "st_ino", 0))
        except OSError:
            sig = (str(path), None, None, None)
        cached = getattr(self, "_spacing_cache", None)
        if cached is None or cached[0] != sig:
            # a read-only adopt (suppress_save) never renames a corrupt file
            entries = VS.read_patch_spacing_file(
                self._workspace, quarantine=not self.suppress_save)
            cached = (sig, entries)
            self._spacing_cache = cached
        return cached[1]

    def _spacing_file_live(self) -> bool:
        """True when the file is the source of truth: a workspace is
        loaded and the envelope map has been migrated or the file exists."""
        if not self._workspace:
            return False
        self._migrate_spacing_once()
        return (getattr(self, "_spacing_migrated", False)
                or VS.patch_spacing_path(self._workspace).exists()
                or not isinstance(self._state.get("patch_spacing"), dict)
                or not self._state.get("patch_spacing"))

    def _migrate_spacing_once(self) -> None:
        """Write every envelope entry the file LACKS into the file, once a
        stack folder is known (audit M1: not only when the file does not
        exist -- Step 4 may have created it first).  A ``user`` entry is
        never replaced and a same value writes nothing, so it is safe to
        repeat; ``_spacing_migrated`` is set only when every entry landed.
        A read-only adopt (``suppress_save``) writes nothing (audit M3)."""
        if (not self._workspace or getattr(self, "_spacing_migrated", False)
                or not self._spacing_dir() or self.suppress_save):
            return
        raw = self._state.get("patch_spacing")
        if not isinstance(raw, dict):
            self._spacing_migrated = True
            return
        try:
            have = VS.read_patch_spacing_file(self._workspace)
        except Exception:      # noqa: BLE001 -- never raises, belt
            have = {}
        n = 0
        failed = False
        for name, entry in raw.items():
            valid = self._spacing_entry(entry)
            if valid is None:
                continue
            key = self._spacing_key(name)
            if VS.find_patch_spacing(have, key) is not None:
                continue
            try:
                n += bool(VS.write_patch_spacing_entry(
                    self._workspace, key, valid["zyx"], valid["source"]))
            except (OSError, ValueError) as exc:
                failed = True
                print(f"[STEP3-SPACING] migration of {name!r}: {exc!r}")
        self._spacing_migrated = not failed
        if n:
            print(f"[STEP3-SPACING] migrated {n} voxel size(s) into "
                  f"{VS.PATCH_SPACING_FILENAME}")

    def _file_writable(self) -> bool:
        """The file branch of a write: not during a read-only adopt, and
        not with the folder unknown when the controller asks for it."""
        if self.suppress_save:
            return False
        if self.require_stack_dir and not self._spacing_dir():
            return False
        return True

    def _spacing_entry_for(self, patch_name: str) -> Optional[Dict[str, Any]]:
        entry = VS.find_patch_spacing(self._file_entries(),
                                      self._spacing_key(patch_name))
        if entry is None and self.suppress_save:
            # a read-only adopt keeps its values in memory only
            return self._spacing_entry(
                (self._state.get("patch_spacing") or {}).get(patch_name)
                if isinstance(self._state.get("patch_spacing"), dict)
                else None)
        return entry

    def get_patch_spacing(self, patch_name: str) -> Optional[Dict[str, Any]]:
        """This patch's ``{"zyx": [z, y, x], "source": ...}`` (a copy), or
        ``None`` (no value, or an unreadable one)."""
        if self._spacing_file_live():
            entry = self._spacing_entry_for(patch_name)
            if entry is None:
                return None
            return {"zyx": list(entry["zyx"]), "source": entry["source"]}
        raw = self._state.get("patch_spacing")
        if not isinstance(raw, dict):
            return None
        return self._spacing_entry(raw.get(patch_name))

    def get_all_patch_spacing(self) -> Dict[str, Dict[str, Any]]:
        """Every readable entry of the listed folder, by patch name."""
        if self._spacing_file_live():
            d = self._spacing_dir()
            prefix = ""
            if d:
                prefix = VS.patch_spacing_key(os.path.join(d, "x"),
                                              self._workspace)[:-1]
            fold = os.path.normcase(prefix)
            out = {}
            for key, entry in self._file_entries().items():
                if not os.path.normcase(key).startswith(fold):
                    continue
                name = key[len(prefix):]
                if name and "/" not in name:
                    out[name] = {"zyx": list(entry["zyx"]),
                                 "source": entry["source"]}
            if self.suppress_save:
                mirror = self._state.get("patch_spacing")
                for name, entry in (mirror.items()
                                    if isinstance(mirror, dict) else ()):
                    valid = self._spacing_entry(entry)
                    if valid is not None and name not in out:
                        out[str(name)] = valid
            return out
        raw = self._state.get("patch_spacing")
        if not isinstance(raw, dict):
            return {}
        out = {}
        for name, entry in raw.items():
            valid = self._spacing_entry(entry)
            if valid is not None:
                out[str(name)] = valid
        return out

    def set_patch_spacing(self, patch_name: str, zyx, source: str) -> bool:
        """Store this patch's voxel size; ``True`` when it changed.

        Raises ``ValueError`` for a *source* outside ``SPACING_SOURCES`` or
        a *zyx* that is not three positive numbers.  A ``step1`` / ``file``
        value never replaces a ``user`` one (returns ``False``); the same
        value again changes nothing and schedules no save.  A file that
        cannot be written returns ``False`` with ``last_spacing_error`` set,
        for the caller to say on screen."""
        if source not in SPACING_SOURCES:
            raise ValueError(f"unknown spacing source {source!r}; expected "
                             f"one of {SPACING_SOURCES}")
        vals = _positive(zyx if zyx is not None else ())
        if vals is None:
            raise ValueError(f"spacing must be three positive numbers, got "
                             f"{zyx!r}")
        if self._spacing_file_live() and self._file_writable():
            try:
                changed = VS.write_patch_spacing_entry(
                    self._workspace, self._spacing_key(patch_name), vals,
                    source)
            except OSError as exc:
                self.last_spacing_error = repr(exc)
                print(f"[STEP3-SPACING] {patch_name!r}: file not written: "
                      f"{exc!r}")
                return False
            self._spacing_cache = None
            if changed:
                self._spacing_map()[patch_name] = {"zyx": vals,
                                                   "source": source}
                self.schedule_save()
            return changed
        if self._spacing_file_live() and not self.suppress_save:
            self.last_spacing_error = "no patch folder known yet"
            return False            # nothing to key by
        current = (self._spacing_entry_for(patch_name)
                   if self._workspace else
                   self._spacing_entry((self._state.get("patch_spacing")
                                        or {}).get(patch_name)))
        if (current is not None and current["source"] == "user"
                and source != "user"):
            return False
        new = {"zyx": vals, "source": source}
        if current is not None and {"zyx": current["zyx"],
                                    "source": current["source"]} == new:
            return False
        self._spacing_map()[patch_name] = new
        self.schedule_save()
        return True

    def clear_patch_spacing(self, patch_name: str, *,
                            user_too: bool = False) -> bool:
        """Drop this patch's value; a ``user`` value only with
        ``user_too=True`` (the user clearing their own).  ``True`` when an
        entry was removed."""
        if self._spacing_file_live() and self._file_writable():
            try:
                changed = VS.clear_patch_spacing_entry(
                    self._workspace, self._spacing_key(patch_name),
                    user_too=user_too)
            except OSError as exc:
                self.last_spacing_error = repr(exc)
                print(f"[STEP3-SPACING] {patch_name!r}: file not written: "
                      f"{exc!r}")
                return False
            self._spacing_cache = None
            if changed:
                mirror = self._state.get("patch_spacing")
                if isinstance(mirror, dict):
                    mirror.pop(patch_name, None)
                self.schedule_save()
            return changed
        if self._spacing_file_live() and not self.suppress_save:
            return False
        raw = self._state.get("patch_spacing")
        if not isinstance(raw, dict) or patch_name not in raw:
            return False
        current = self._spacing_entry(raw.get(patch_name))
        if (current is not None and current["source"] == "user"
                and not user_too):
            return False
        del raw[patch_name]
        self.schedule_save()
        return True

    def apply_resolved_spacing(self, patch_name: str,
                               resolved: Optional[Dict[str, Any]], *,
                               read_file: bool) -> bool:
        """Store what ``resolve_stack_spacing`` answered for this patch
        (asked with the record ``{"spacing": get_patch_spacing(name)}``);
        ``True`` when the stored value changed.

        A dict is stored through :meth:`set_patch_spacing`, so a ``user``
        value is never replaced.  ``None`` drops a stored ``step1`` value
        (Step 1 no longer has one) and, only when the resolve read the file
        (``read_file=True``), a ``file`` value: a main-thread resolve cannot
        see the file's tags, so it keeps what a worker read."""
        if isinstance(resolved, dict):
            try:
                return self.set_patch_spacing(
                    patch_name, resolved.get("zyx"), resolved.get("source"))
            except ValueError:
                return False
        current = self.get_patch_spacing(patch_name)
        if current is None or current["source"] == "user":
            return False
        if current["source"] == "file" and not read_file:
            return False
        return self.clear_patch_spacing(patch_name)

    def get_measured_spacing(self, patch_name: str) -> Optional[Dict[str, Any]]:
        """The spacing this run's features of the patch were measured with
        (``{"zyx"[, "source"]}``), or ``None`` (in voxels, or never)."""
        record = self._state["patches"].get(patch_name)
        if not isinstance(record, dict):
            return None
        raw = record.get("measured_spacing")
        if not isinstance(raw, dict):
            return None
        zyx = _positive(raw.get("zyx") or ())
        if zyx is None:
            return None
        out: Dict[str, Any] = {"zyx": zyx}
        if raw.get("source") is not None:
            out["source"] = raw.get("source")
        return out

    def set_measured_spacing(self, patch_name: str,
                             spacing: Optional[Dict[str, Any]]) -> None:
        """Record on the RUN's patch record the spacing an extraction used
        (``None`` = measured in voxels).  Raises ``ValueError`` for a value
        that is neither ``None`` nor a dict holding three positive
        numbers."""
        if spacing is None:
            value = None
        else:
            if not isinstance(spacing, dict):
                raise ValueError(f"measured spacing must be a dict or None, "
                                 f"got {spacing!r}")
            zyx = _positive(spacing.get("zyx") or ())
            if zyx is None:
                raise ValueError(f"measured spacing must hold three positive "
                                 f"numbers, got {spacing!r}")
            value = {"zyx": zyx}
            if spacing.get("source") is not None:
                value["source"] = spacing.get("source")
        patches = self._state["patches"]
        if patch_name not in patches:
            patches[patch_name] = {"status": "unprocessed"}
        patches[patch_name]["measured_spacing"] = value
        self.schedule_save()

    # ------------------------------------------------------------------
    # Refinement extraction config (per-orientation, 2026-05-19)
    # ------------------------------------------------------------------
    #
    # Stores per-orientation slice-sampling + mask-cleaning params for
    # the extraction pipeline.  Persisted alongside other workspace
    # state.  When unset (or partially set), callers fall back to the
    # legacy single-knob UI values.  Schema:
    #
    #   refinement_extraction_config = {
    #     "lateral": {                  # applied to xy slices
    #       "min_labels_per_slice": int,
    #       "edge_buffer": int,
    #       "slices_per_patch": int,
    #       "min_distance": int,
    #       "remove_boundaries": bool,
    #       "relabel": bool,
    #       "min_size": int,
    #       "smoothing_disk_size": int,
    #       "expand_labels_distance": int,  # 2026-05-21 — compensates
    #                                       # for remove_boundaries=True
    #                                       # erosion bias (0 = legacy off).
    #     },
    #     "axial": { ... same keys ... },   # applied to xz + yz slices
    #   }
    #
    # The field is None until the user opens the per-orientation
    # config dialog and saves at least once.  Backward compat: old
    # workspaces have no key, get_refinement_extraction_config()
    # returns an empty dict, callers fall back to UI scalar values.

    def get_refinement_extraction_config(self) -> Dict[str, Any]:
        """Return the persisted per-orientation extraction config.

        Always returns a dict (never None) so callers can dereference
        keys safely.  When the dialog has never been used, returns
        ``{}`` — callers should fall back to UI scalar widgets.

        The returned dict may have partial coverage (e.g. only
        ``lateral`` set, no ``axial``) if the dialog was extended
        across schema versions.  Callers should handle missing keys.
        """
        cfg = self._state.get("refinement_extraction_config") or {}
        if not isinstance(cfg, dict):
            return {}
        return dict(cfg)

    def set_refinement_extraction_config(
        self,
        *,
        lateral: Optional[Dict[str, Any]] = None,
        axial: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update the persisted per-orientation extraction config.

        Any argument left as ``None`` leaves that orientation's sub-dict
        unchanged.  Pass ``{}`` explicitly to clear an orientation back
        to "use UI defaults."  Writes are debounced via
        ``schedule_save()``.
        """
        current = self._state.get("refinement_extraction_config") or {}
        if not isinstance(current, dict):
            current = {}
        if lateral is not None:
            current["lateral"] = dict(lateral)
            print(
                f"[PERVIEW-STATE] set lateral config: "
                f"{len(current['lateral'])} keys"
            )
        if axial is not None:
            current["axial"] = dict(axial)
            print(
                f"[PERVIEW-STATE] set axial config: "
                f"{len(current['axial'])} keys"
            )
        self._state["refinement_extraction_config"] = current
        self.schedule_save()

    # ------------------------------------------------------------------
    # Downsample mode (D2)
    # ------------------------------------------------------------------

    def get_downsample_mode(self) -> bool:
        """Workspace-level toggle for downsample-on-load.  Default False."""
        return bool(self._state.get("downsample_mode", False))

    def set_downsample_mode(self, enabled: bool) -> None:
        """Set the workspace-level downsample-mode toggle.

        Persists via debounced save.  Affects FUTURE patch loads only —
        currently-loaded patches keep their already-applied factor
        until reloaded.
        """
        self._state["downsample_mode"] = bool(enabled)
        self.schedule_save()

    # ------------------------------------------------------------------
    # Unified-layout flag (Phase 7 — optional default-run nesting)
    # ------------------------------------------------------------------

    def get_default_run_nested(self) -> bool:
        """Workspace-level flag — has the default run been migrated into the
        symmetric nested layout (``runs/default/...``)?  Default False.

        When False (the un-migrated case, every existing workspace) the
        default run is FLAT (``artifacts/step3/{qc,curation,...}``) and
        resolves to path-id ``None`` — byte-identical to today.  When True
        (the user ran the optional "Convert to unified layout" migration)
        the default run's data lives under ``runs/default/...`` and resolves
        to the ``"default"`` slug.  Authoritative gate for the resolver +
        the ``scan_existing_artifacts`` path-id mapping.
        """
        return bool(self._state.get("default_run_nested", False))

    def set_default_run_nested(self, enabled: bool) -> None:
        """Set the workspace-level unified-layout flag.  Persists (debounced).

        ONLY flipped True by the Phase 7 ``_do_convert_layout`` migration
        AFTER the physical move + backup succeed.  Mirrors
        ``set_downsample_mode``.
        """
        self._state["default_run_nested"] = bool(enabled)
        self.schedule_save()

    def _resolve_path_id(self, run_id: str) -> Optional[str]:
        """Map a run-id to the path-id the ``_paths`` helpers expect.

        Option A (default run = flat): the default run normally resolves to
        ``None`` (flat ``artifacts/step3/...``).  EXCEPTION (Phase 7): when
        the workspace has opted into the unified layout
        (:meth:`get_default_run_nested` is True) the default run resolves to
        the ``"default"`` slug so reads/writes route to ``runs/default/...``.
        Named runs always resolve to their own slug.
        """
        if (
            run_id == _migrate_step3_runs.DEFAULT_RUN_ID
            and not self.get_default_run_nested()
        ):
            return None
        return run_id

    def get_patch_downsample_factor(self, patch_name: str) -> int:
        """Return the downsample factor applied at this patch's last
        load.  Default 1 (no downsample) for patches with no recorded
        factor — used by ``_save_curation`` to decide whether to lift
        the mask back to full resolution before write.
        """
        record = self._state["patches"].get(patch_name)
        if record is None:
            return 1
        return int(record.get("downsample_factor", 1))

    def get_patch_downsample_warning_shown(self, patch_name: str) -> bool:
        """Return whether the lossy-downsample-load warning has been
        dismissed for this patch.

        D2.5 (perf push 2026-05-10): set to True the first time a user
        opts into Continue on the lossy-load warning modal for this
        patch.  Prevents repeat nagging on every reload.  Per-patch,
        not workspace-level — different patches each get one warning.

        Default False for patches with no recorded value (i.e., warning
        will fire on the first qualifying load).
        """
        record = self._state["patches"].get(patch_name)
        if record is None:
            return False
        return bool(record.get("downsample_warning_shown", False))

    def get_patch_original_labels_shape(
        self, patch_name: str
    ) -> Optional[tuple]:
        """Return the original (full-res) labels shape recorded at this
        patch's last load — used by ``_save_curation`` (D2.1) as the
        target shape for NN-upsample lift.

        Returns ``None`` if not recorded (load was at full-res, no
        lift needed; or this patch hasn't been loaded yet).  Stored
        as a list in JSON; this getter returns a tuple for ndarray
        compatibility.
        """
        record = self._state["patches"].get(patch_name)
        if record is None:
            return None
        shape = record.get("original_labels_shape")
        if shape is None:
            return None
        return tuple(int(x) for x in shape)

    def get_patch_source_shape(self, patch_name: str) -> Optional[tuple]:
        """The raw patch's shape as recorded by the header read that
        resolved its axes (``source_shape``, brick V2), or ``None`` when it
        was not recorded (a patch detected before V2, or a failed read).
        Session's row reads its channel count from it without opening the
        file again."""
        record = self._state["patches"].get(patch_name)
        if record is None:
            return None
        shape = record.get("source_shape")
        if not shape:
            return None
        try:
            return tuple(int(x) for x in shape)
        except (TypeError, ValueError):
            return None

    def get_patch_axes(self, patch_name: str, default: str = "CZYX") -> str:
        """Return the detected axis order for a patch's source stack.

        Cosmetic per-patch display (2026-06-07): the Patch State list shows
        each patch's auto-detected axis layout (CZYX / ZCYX / ZYX) so the
        user can verify mixed-layout folders at a glance.  The value is
        cached by the controller via ``update_patch_metadata(name,
        axes=...)`` after a cheap header-only read.

        Returns ``default`` for patches with no recorded axes (e.g. legacy
        workspaces, or a patch not yet detected) — display falls back to a
        placeholder, never crashes.
        """
        record = self._state["patches"].get(patch_name)
        if record is None:
            return default
        # IMPORTANT: return the bare `default` when the `axes` key is absent
        # (a record exists but hasn't been detected yet).  Do NOT
        # ``str(record.get("axes", default))`` — that would stringify a
        # caller's sentinel default (e.g. ``_AXES_UNSET``) into
        # "<object object at 0x...>", defeating "uncached" detection and
        # leaking garbage into the Patch State column.
        axes = record.get("axes")
        return str(axes) if axes is not None else default

    # ------------------------------------------------------------------
    # Active patch tracking
    # ------------------------------------------------------------------

    @property
    def active_patch(self) -> Optional[str]:
        return self._state.get("active_patch")

    @active_patch.setter
    def active_patch(self, patch_name: Optional[str]) -> None:
        self._state["active_patch"] = patch_name
        self.schedule_save()

    # ------------------------------------------------------------------
    # Artifact scanning (session restore)
    # ------------------------------------------------------------------

    def scan_existing_artifacts(self, workspace: str) -> None:
        """Scan workspace for prediction caches and reconstruction manifests.

        Populates status for patches that have on-disk artifacts but no
        state entry (or whose state was lost).
        """
        ws = Path(workspace)
        patches = self._state["patches"]

        # Resolve the ACTIVE run's path-id (Option A: default run = flat
        # layout → None; named runs → slug).  ``reconstructions_dir(ws,
        # None) == runs_dir(ws)`` and ``prediction_cache_dir(ws, None)``
        # == the flat cache, so the default case is byte-identical.
        # Phase 7: when the workspace opted into the unified layout the
        # default run resolves to "default" (nested) — handled by
        # ``_resolve_path_id`` (the flag-aware mapping).
        path_id = self._resolve_path_id(self.active_run_id)

        # Scan prediction cache
        pred_cache = _paths.prediction_cache_dir(ws, path_id)
        if pred_cache.is_dir():
            seen_stems = set()
            for f in pred_cache.glob("*_xy_flows.npy"):
                stem = f.name.replace("_xy_flows.npy", "")
                seen_stems.add(stem)
            for stem in seen_stems:
                # Find matching patch by stem
                for pname in list(patches.keys()):
                    if Path(pname).stem == stem:
                        if patches[pname].get("status") == "unprocessed":
                            patches[pname]["status"] = "predicted"
                        break

        # Scan reconstruction outputs.  Post-restructure: each patch
        # owns its own runs/{patch_name}/ directory rather than the
        # legacy run_id-based scattering.
        recon_dir = _paths.reconstructions_dir(ws, path_id)
        if recon_dir.is_dir():
            manifest_paths = sorted(
                (
                    run_dir / "manifest.json"
                    for run_dir in recon_dir.iterdir()
                    if run_dir.is_dir()
                ),
                key=lambda path: path.stat().st_mtime if path.exists() else 0,
                reverse=True,
            )
            for manifest_path in manifest_paths:
                if not manifest_path.exists():
                    continue
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    src = manifest.get("source_stack_path", "")
                    src_name = Path(src).name if src else ""
                    if src_name in patches:
                        record = patches[src_name]
                        if record.get("status") in ("unprocessed", "predicted"):
                            record["status"] = "reconstructed"
                        if not record.get("reconstruction_run_id"):
                            record["reconstruction_run_id"] = manifest.get("run_id", "")
                except (json.JSONDecodeError, OSError):
                    pass

        self.schedule_save()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Stack directory persistence
    # ------------------------------------------------------------------

    @property
    def loaded_stack_dir(self) -> Optional[str]:
        """Persisted stack directory path."""
        return self._state.get("loaded_stack_dir")

    @loaded_stack_dir.setter
    def loaded_stack_dir(self, path: Optional[str]) -> None:
        self._state["loaded_stack_dir"] = path
        self.schedule_save()

    # ------------------------------------------------------------------
    # Backward transitions
    # ------------------------------------------------------------------

    def reset_to_unprocessed(self, patch_name: str) -> None:
        """Reset patch to unprocessed, clearing prediction + reconstruction metadata."""
        patches = self._state["patches"]
        if patch_name not in patches:
            return
        record = patches[patch_name]
        record["status"] = "unprocessed"
        record["updated_at"] = datetime.now(timezone.utc).isoformat()
        for key in ("predicted_at", "reconstructed_at", "accepted_at",
                     "finalized_at", "reconstruction_run_id"):
            record.pop(key, None)
        self.schedule_save()

    def reset_to_predicted(self, patch_name: str) -> None:
        """Reset patch to predicted, clearing reconstruction metadata."""
        patches = self._state["patches"]
        if patch_name not in patches:
            return
        record = patches[patch_name]
        record["status"] = "predicted"
        record["updated_at"] = datetime.now(timezone.utc).isoformat()
        for key in ("reconstructed_at", "accepted_at",
                     "finalized_at", "reconstruction_run_id"):
            record.pop(key, None)
        self.schedule_save()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_state() -> Dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "patches": {},
            "active_patch": None,
            "loaded_stack_dir": None,
            "last_updated": None,
            # D2 (perf push 2026-05-09): workspace-level toggle for
            # downsample-on-load.  Old state files lack this field;
            # getter returns False by default for backwards compat.
            "downsample_mode": False,
            # Phase 7 (2026-06-28): workspace-level unified-layout flag.
            # False (default) → default run is FLAT (byte-identical to
            # today); True → default run nested under runs/default/.
            "default_run_nested": False,
        }
