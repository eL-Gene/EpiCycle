"""Step 1 (Data Preparation) state manager — ``step1_state.json``.

Pure-Python, no Qt/napari, so it is unit-testable headless.

The state file is the **workspace-level index** for Step 1:

* ``sources``  — source datasets patches were extracted from, deduplicated
  by normalized path (per-FILE since Commit 3).  Provenance: answers "which
  file did this patch come from / is this new data?".  Also carries the
  per-file numbering ``prefix`` (the display-name the file's patches were
  exported under; used by the badge, last-wins on re-extract).
* ``patches``  — registry of extracted patches, each linked to a
  ``source_id`` and pointing at its on-disk TIFF + sidecar.
* ``settings`` — last-used display name / prefix.

**Source-of-truth rule:** the per-patch ``{patch}_metadata.json`` sidecar
remains authoritative for geometry (bounding_box, z_range, channel_names).
This state file does NOT duplicate those numbers — it only indexes patches
and records provenance + settings.

Persistence mirrors the Step 3 state manager: atomic write
(tempfile → ``os.replace``) and corrupt-tolerant load (missing / bad /
non-dict JSON → a fresh empty state).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ._paths import step1_state_backup_path, step1_state_path

STEP1_SCHEMA_VERSION = 2

_SRC_ID_RE = re.compile(r"^src_(\d+)$")

#: Top-level state key holding the per-workspace large-data convert config
#: (``{preset, num_workers, chunk, downsample, min_size}``).  Preserved by
#: :func:`_normalize` so it round-trips at the current schema version.
LARGEDATA_CONFIG_KEY = "largedata_convert_config"


# ---------------------------------------------------------------------------
# Construction / persistence
# ---------------------------------------------------------------------------
def empty_state() -> Dict[str, Any]:
    """Return a fresh, independent empty Step 1 state."""
    return {
        "schema_version": STEP1_SCHEMA_VERSION,
        "sources": [],
        "patches": [],
        "settings": {},
    }


def _migrate_source_row_v2(src: Dict[str, Any]) -> Dict[str, Any]:
    """FIELD-ONLY, in-memory v1 -> v2 upgrade of ONE source row.

    NO filesystem I/O — this runs inside :func:`_normalize` (called by
    :func:`load_state` on every keystroke, e.g. the patch-count badge), so it
    must be a pure dict rewrite (the ``largedata/`` store-scan + abs->rel
    pointer rewrite + fingerprint-from-store belong in the session-guarded
    Brick-2 restore hook, never here).

    * ``source_axes`` <- ``axes`` (the legacy field IS the real on-disk order;
      the legacy ``axes`` key is KEPT so a v1 reader still finds it).
    * ``voxel_size_source`` <- ``"user"`` if a ``voxel_size`` is present, else
      ``"unknown"`` (honest voxel typing — a silent default is not "detected").
    * ``origin`` <- ``["exported"]`` for a pre-existing row (v1 only ever wrote
      a source row on patch export).

    Additive + idempotent: an already-v2 row (source_axes / voxel_size_source /
    origin present) is left untouched, and any unknown per-entry keys survive
    (the row dict is mutated in place, never rebuilt), so the upgrade is
    forward-compatible + lossless.
    """
    if "source_axes" not in src and "axes" in src:
        src["source_axes"] = src.get("axes")
    if "voxel_size_source" not in src:
        # Honest typing: a legacy row that baked the isotropic 1/1/1 default
        # (the pre-v2 export bug) upgrades to "unknown", never "user".
        src["voxel_size_source"] = _classify_voxel_source(src.get("voxel_size"))
    if "origin" not in src:
        src["origin"] = ["exported"]
    return src


def _normalize(data: Any) -> Dict[str, Any]:
    """Coerce arbitrary loaded JSON into a valid state shape.

    Missing keys are filled from :func:`empty_state`; a non-dict input
    yields a fresh empty state.  Source rows are field-upgraded to schema v2
    in memory (:func:`_migrate_source_row_v2`) — additive + lossless, NO I/O.
    """
    if not isinstance(data, dict):
        return empty_state()
    st = empty_state()
    schema_ver = data.get("schema_version")
    if isinstance(schema_ver, int):
        # Auto-upgrade a v1 file to the current version IN MEMORY (a save
        # trigger persists it); never downgrade a future file.
        st["schema_version"] = max(schema_ver, STEP1_SCHEMA_VERSION)
    # Drop non-dict garbage from the lists so downstream `.get(...)` calls can
    # never crash (preserves the "load never raises" contract); also makes a
    # fresh list rather than aliasing the loaded structure.  Each surviving
    # source row is field-upgraded to v2 (additive, no I/O).
    if isinstance(data.get("sources"), list):
        st["sources"] = [
            _migrate_source_row_v2(s)
            for s in data["sources"] if isinstance(s, dict)
        ]
        # INVARIANT (Brick 4a a7): every row in `sources[]` carries an id.
        # `_migrate_source_row_v2` is per-ROW and cannot allocate one (it can't
        # see its siblings), yet `_resolve_source_id_for` mints a FRESH `src_N`
        # for a path-matched row that lacks an id -> a DUPLICATE row for the same
        # file, and every patch cut from it orphaned in ④.  Back-filling here
        # keeps `_normalize`'s pure-dict / no-I/O contract.
        for _row in st["sources"]:
            if not _row.get("source_id"):
                _row["source_id"] = _next_source_id(st["sources"])
    if isinstance(data.get("patches"), list):
        st["patches"] = [p for p in data["patches"] if isinstance(p, dict)]
    if isinstance(data.get("settings"), dict):
        st["settings"] = data["settings"]
    # Optional per-workspace large-data convert config (added post-restructure,
    # still schema v1).  Preserve it here explicitly so it survives a
    # load -> save round-trip at the CURRENT schema version (the forward-compat
    # block below only fires for FUTURE schemas).  Absent by default, so a fresh
    # state has no key and readers fall back to the adaptive default.
    if isinstance(data.get(LARGEDATA_CONFIG_KEY), dict):
        st[LARGEDATA_CONFIG_KEY] = data[LARGEDATA_CONFIG_KEY]
    # Forward-compat: if the file was written by a FUTURE schema, preserve its
    # unknown top-level keys so an older reader doesn't silently drop them on
    # the next save (avoids v2 -> v1-read -> save data loss).
    if isinstance(schema_ver, int) and schema_ver > STEP1_SCHEMA_VERSION:
        for key, val in data.items():
            if key not in ("schema_version", "sources", "patches", "settings"):
                st[key] = val
    return st


def _quarantine_corrupt_state(path) -> Optional[str]:
    """Rename an unreadable ``step1_state.json`` aside, best-effort.

    Returns the new file's basename, or ``None`` if the rename failed.  The file
    is NEVER deleted: it is the user's only copy of the sources, voxels,
    recipes, store pointers and patch registry that could not be parsed, and it
    is often repairable by hand.
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = f"{path}.corrupt-{stamp}"
    n = 1
    while os.path.exists(target):              # two loads in the same second
        target = f"{path}.corrupt-{stamp}-{n}"
        n += 1
    try:
        os.replace(str(path), target)
    except OSError:
        return None
    return os.path.basename(target)


_CORRUPT_STATE_HOOK = None


def set_corrupt_state_hook(callback) -> None:
    """Install (or clear, with ``None``) the process-wide corrupt-state hook.

    ``load_state(ws, on_corrupt=...)`` only ever reached ONE of the ~25 call
    sites in the Step-1 controller, so a corrupt file discovered by any of the
    other 24 was still silent — the exact data-loss class the callback exists
    to expose.  The controller registers this hook once; every bare
    ``load_state`` then reports.  An explicit ``on_corrupt=`` still wins (and
    suppresses the hook) so a caller can route to its own prefix.
    """
    global _CORRUPT_STATE_HOOK
    _CORRUPT_STATE_HOOK = callback if callable(callback) else None


def load_state(workspace: str, *, on_corrupt=None) -> Dict[str, Any]:
    """Load ``step1_state.json``; tolerate missing / corrupt / non-dict.

    Always returns a valid, normalized state dict (never raises).

    A file that cannot be parsed (or is not a JSON object) is RENAMED to
    ``step1_state.json.corrupt-<YYYYmmdd-HHMMSS>`` before the empty default is
    returned, so the next save writes a fresh file instead of overwriting the
    user's only (possibly repairable) copy.  ``on_corrupt``, when given, is
    called once with a human-readable message so the caller can surface it in
    the UI — a corrupt load is a TOTAL loss of Step-1 state and must never be
    silent.  Rename and callback are both best-effort; this never raises.
    """
    path = step1_state_path(workspace)
    if not path.exists():
        return empty_state()
    corrupt_reason = None
    data = None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        corrupt_reason = f"unreadable ({exc.__class__.__name__})"
    else:
        if not isinstance(data, dict):
            corrupt_reason = "not a JSON object"
    if corrupt_reason is not None:
        saved_as = _quarantine_corrupt_state(path)
        report = on_corrupt if callable(on_corrupt) else _CORRUPT_STATE_HOOK
        if callable(report):
            where = (f"saved as {saved_as}" if saved_as
                     else "could not be renamed aside")
            try:
                report(
                    f"step1_state.json is {corrupt_reason} — {where}; "
                    f"Step 1 started from an empty state; a copy of your "
                    f"previous save is in artifacts/step1/state_backup/"
                )
            except Exception:                  # noqa: BLE001 — never raise
                pass
        return empty_state()
    return _normalize(data)


def save_state(workspace: str, state: Dict[str, Any]) -> None:
    """Atomically write *state* to ``step1_state.json`` (tempfile → replace).

    Creates the workspace directory if it does not yet exist.  The previous
    good file is first rotated to
    ``artifacts/step1/state_backup/step1_state.json.bak`` (best-effort — a
    rotation failure never blocks the save), so there is always one recoverable
    predecessor if the current file is later damaged.
    """
    path = step1_state_path(workspace)
    base = path.parent
    os.makedirs(base, exist_ok=True)
    if path.exists():
        try:
            bak = step1_state_backup_path(workspace)
            os.makedirs(str(bak.parent), exist_ok=True)
            shutil.copy2(str(path), str(bak))
        except OSError:
            pass                               # rotation is best-effort
    fd, tmp_path = tempfile.mkstemp(suffix=".tmp", prefix=".step1_state_", dir=str(base))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_path, str(path))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm_path(path: str) -> str:
    """Canonicalize a path for case/separator-insensitive dedup comparison."""
    return os.path.normcase(os.path.normpath(str(path)))


def _next_source_id(sources: List[Dict[str, Any]]) -> str:
    used = []
    for s in sources:
        m = _SRC_ID_RE.match(str(s.get("source_id", "")))
        if m:
            used.append(int(m.group(1)))
    return f"src_{(max(used) + 1) if used else 0}"


def _classify_voxel_source(voxel_size: Optional[Any]) -> str:
    """Honest voxel typing (schema-v2 invariant #4): ``"user"`` ONLY for a REAL
    spacing.

    Returns ``"unknown"`` when *voxel_size* is ``None``, empty, malformed, OR
    every component is ~1.0 — the isotropic ``(1, 1, 1)`` default that means
    "spacing unknown" (mirrors
    :meth:`DataPreparationController._prepare_voxel_size`).  A genuine 1/1/1 the
    user deliberately typed is an accepted edge (treated as unknown), so a
    silent default can never masquerade as a detected/entered value and poison a
    downstream physical-unit consumer.  Otherwise ``"user"``.  Never raises.
    """
    if voxel_size is None:
        return "unknown"
    try:
        vals = [float(v) for v in voxel_size]
    except (TypeError, ValueError):
        return "unknown"
    if not vals:
        return "unknown"
    if all(abs(v - 1.0) < 1e-9 for v in vals):
        return "unknown"
    return "user"


# ---------------------------------------------------------------------------
# Mutators (pure on the passed-in state dict)
# ---------------------------------------------------------------------------
def register_source(
    state: Dict[str, Any],
    *,
    path: str,
    display_name: str,
    axes: str,
    voxel_size: Optional[Any] = None,
    timestamp: Optional[str] = None,
    prefix: Optional[str] = None,
) -> str:
    """Record the source dataset *path* (dedup by normalized path).

    Returns the ``source_id`` — the existing one if this path was already
    registered, otherwise a freshly allocated ``src_N``.

    *prefix* (optional, Commit 3 / D-3) is the numbering prefix the source's
    patches are exported under (``{display_name}_patch``).  On a re-register
    (revisit) it is UPDATED (last wins) so the badge tracks the current naming;
    every other metadata field keeps its original (first) value.

    **The EXPORT path is deliberately NON-CLOBBERING (schema v2):** only ``prefix`` updates on a revisit; ``source_axes`` /
    ``voxel_size`` / ``display_name`` are FIRST-WINS so a later export never
    overwrites provenance a Prepare (or an explicit user edit via
    :func:`upsert_source`) already recorded.  TRUE last-wins is reserved for the
    explicit edit signals, which route through :func:`upsert_source`.

    ``axes`` is the source's REAL on-disk order; it is stored under the schema-
    v2 ``source_axes`` field (the anti-"axes lie" home).

    Raises ``ValueError`` if *path* is empty/None (an empty path would
    silently coalesce distinct datasets into one source).
    """
    if not path or not str(path).strip():
        raise ValueError("register_source: path must be a non-empty string")
    target = _norm_path(path)
    for src in state["sources"]:
        if _norm_path(src.get("path", "")) == target:
            if prefix:
                src["prefix"] = prefix  # last-wins; badge tracks current naming
            return src.get("source_id", "")  # tolerate a hand-edited source missing the id
    source_id = _next_source_id(state["sources"])
    entry = {
        "source_id": source_id,
        "path": path,
        "display_name": display_name,
        "source_axes": axes,
        "voxel_size": voxel_size,
        # Honest typing: a 1/1/1 export is "unknown", never "user" (a raw
        # isotropic default must not masquerade as an entered spacing).
        "voxel_size_source": _classify_voxel_source(voxel_size),
        "origin": ["exported"],
        "added_at": timestamp or _now(),
    }
    if prefix:
        entry["prefix"] = prefix
    state["sources"].append(entry)
    return source_id


def upsert_source(
    state: Dict[str, Any],
    *,
    path: Optional[str] = None,
    source_id: Optional[str] = None,
    **fields: Any,
) -> str:
    """Create-or-update a source registry row (**LAST-WINS** for the given
    user-editable fields), keyed by ``source_id`` when supplied, else by the
    normalized ``path``.

    This is the schema-v2 mutator for the explicit EDIT / Prepare signals
 where the caller's action IS the intent — unlike
    :func:`register_source` (the export path), which is first-wins to preserve
    provenance.  Returns the row's ``source_id`` (freshly allocated via
    :func:`_next_source_id` for a new row, or the supplied ``source_id``).

    Semantics
    ---------
    * ``origin`` (if passed) is treated specially: it is **unioned** into the
      existing ``origin`` list (cumulative provenance ``⊆ {browsed, loaded,
      prepared, exported}``), never replaced.
    * Every other ``**fields`` value OVERWRITES the row (last-wins).  A ``None``
      value is SKIPPED (an absent signal must not clobber a real stored value —
      e.g. a Prepare with no voxel leaves any prior voxel intact).
    * Unknown / future keys on the existing row are preserved (the row dict is
      mutated in place).

    Raises ``ValueError`` when neither ``path`` nor ``source_id`` is given (no
    way to locate/allocate the row).
    """
    if not (path and str(path).strip()) and not source_id:
        raise ValueError("upsert_source: need a non-empty path or a source_id")
    sources = state.setdefault("sources", [])
    row = None
    if source_id:
        for s in sources:
            if s.get("source_id") == source_id:
                row = s
                break
    if row is None and path:
        target = _norm_path(path)
        for s in sources:
            if _norm_path(s.get("path", "")) == target:
                row = s
                break

    origin_add = fields.pop("origin", None)

    if row is None:
        sid = source_id or _next_source_id(sources)
        row = {"source_id": sid}
        if path:
            row["path"] = path
        sources.append(row)
    elif path and not row.get("path"):
        row["path"] = path

    for key, val in fields.items():
        if val is not None:   # never clobber a stored value with an absent one
            row[key] = val

    if origin_add is not None:
        existing = row.get("origin")
        merged = list(existing) if isinstance(existing, list) else (
            [existing] if existing else [])
        add = (origin_add if isinstance(origin_add, (list, tuple, set))
               else [origin_add])
        for o in add:
            if o not in merged:
                merged.append(o)
        row["origin"] = merged

    return row["source_id"]


def relink_source(
    state: Dict[str, Any], source_id: str, new_path: str,
    workspace: Optional[str] = None,
) -> bool:
    """Point an EXISTING source row at a new file on disk (Brick 4a a6).

    :func:`upsert_source` deliberately never rewrites a row's ``path`` (it only
    fills a blank one), because a path is that row's dedup key — so a moved /
    renamed / copied raw file needs this explicit, id-addressed rewrite instead.

    Rewrites ``path``, re-reads ``fingerprint`` (``{size, mtime}``) from
    *new_path* when it is readable, and DROPS ``source_shape`` (it described the
    old file; the next load re-records it); every other field — ``source_id`` above all,
    plus ``display_name`` / ``prepared_store`` / ``processed_store`` /
    ``voxel_size`` / ``prefix`` / ``origin`` — is preserved verbatim, so the
    patches cut from this source stay attached to it.

    The store's own ``.zattrs epicycle_source.path`` still names the OLD file;
    that is deliberate — it is a HINT, not identity (see
    ``_fresh_prepared_store_for``'s A3a rule), and the reconcile pass skips a
    store already claimed by a row pointer rather than synthesizing a duplicate.

    When *workspace* is given, an ABSOLUTE ``prepared_store`` pointer that
    resolves under it is rewritten workspace-RELATIVE (Brick 4c item 7).  It has
    to be: the reconcile pass's step 1 re-binds an absolute pointer only when
    the store's own recorded source path matches the ROW's path — which a relink
    has just made false by design — so it would NULL the pointer, and step 3
    would then see an unclaimed store and synthesize a duplicate row.  A
    relative pointer never enters that branch at all.

    Returns ``True`` when a row was found and rewritten, ``False`` otherwise.
    Never raises.
    """
    if not source_id or not new_path:
        return False
    for row in state.get("sources", []):
        if row.get("source_id") != source_id:
            continue
        row["path"] = new_path
        try:
            if os.path.isfile(new_path):
                row["fingerprint"] = {
                    "size": os.path.getsize(new_path),
                    "mtime": os.path.getmtime(new_path),
                }
        except OSError:
            pass
        # The recorded dimensions describe the OLD file.  A "confirm" relink is
        # explicitly a shape mismatch, so leaving them would make the next
        # identity check compare against a stack that is no longer there
        # (Brick 4e item 6).  Dropped, not guessed: the next load re-records it
        # from the real header.
        row.pop("source_shape", None)
        ptr = row.get("prepared_store")
        if workspace and ptr and os.path.isabs(str(ptr)):
            try:
                rel = os.path.relpath(str(ptr), workspace)
                if not rel.startswith(".."):
                    row["prepared_store"] = rel.replace(os.sep, "/")
            except (OSError, ValueError):
                pass
        return True
    return False


def register_patch(
    state: Dict[str, Any],
    *,
    patch_name: str,
    relative_path: str,
    sidecar_path: str,
    source_id: str,
    source_kind: str = "raw",
    timestamp: Optional[str] = None,
) -> None:
    """Add (or replace, by ``patch_name``) a patch registry entry.

    ``relative_path`` / ``sidecar_path`` must be workspace-RELATIVE — an
    absolute path would break workspace portability, so it is rejected.
    Any pre-existing entries with the same ``patch_name`` are removed before
    the new one is appended (robust against an already-duplicated registry).

    ``source_kind`` ("raw" | "processed") records whether the patch pixels came
    from the raw source or a prepared processed store (provenance).
    Additive — it defaults to "raw", and readers should ``.get`` it so legacy
    rows written before this field stay valid.
    """
    if os.path.isabs(relative_path):
        raise ValueError(
            f"relative_path must be workspace-relative, not absolute: {relative_path!r}"
        )
    if os.path.isabs(sidecar_path):
        raise ValueError(
            f"sidecar_path must be workspace-relative, not absolute: {sidecar_path!r}"
        )
    entry = {
        "patch_name": patch_name,
        "relative_path": relative_path,
        "sidecar_path": sidecar_path,
        "source_id": source_id,
        "source_kind": source_kind,
        "extracted_at": timestamp or _now(),
    }
    # Case-insensitive dedup: NTFS holds a single file for case-variant names,
    # so the registry must not accrue two rows for `Kidney_1` / `kidney_1`
    # (parity with the case-folded on-disk / IGNORECASE index scans).
    target = str(patch_name).casefold()
    state["patches"] = [
        p for p in state["patches"]
        if str(p.get("patch_name", "")).casefold() != target
    ]
    state["patches"].append(entry)


def remove_patch(state: Dict[str, Any], patch_name: str) -> bool:
    """Remove the patch registry entry named ``patch_name`` (case-insensitive,
    matching :func:`register_patch`'s dedup).  Returns ``True`` if a row was
    removed, ``False`` if none matched.

    Pure: does NOT touch the on-disk TIFF / sidecar — the caller deletes those,
    keeping this state helper file-system-free + trivially testable.
    """
    patches = state.get("patches")
    if not isinstance(patches, list):
        return False
    target = str(patch_name).casefold()
    kept = [p for p in patches
            if str(p.get("patch_name", "")).casefold() != target]
    if len(kept) == len(patches):
        return False
    state["patches"] = kept
    return True


# ---------------------------------------------------------------------------
# Per-patch box styling (④ location overlay) — schema-v2 ADDITIVE
# ---------------------------------------------------------------------------

_VALID_BOX_SHAPES = ("corners", "wireframe")


def normalize_box_style(style: Any) -> Optional[Dict[str, Any]]:
    """Validate a ``box_style`` → a clean ``{color, opacity, shape}`` dict, or None.

    A style carries ONLY per-box ``color`` + ``opacity`` + ``shape`` — width is a
    GLOBAL location-overlay setting (P2 grill: a per-box width would force N vispy
    Line visuals and reintroduce the freed-VBO crash class).  ``shape``:
    ``"corners"`` (bracket segs, default) | ``"wireframe"`` (12 edges).  ``color``:
    an ``[r, g, b, a]`` (floats 0..1) or the sentinel ``"status"`` (use the
    built-in status colour).  Corrupt / empty / non-dict → ``None`` (the caller
    falls through to the source default, then the built-in default)."""
    if not isinstance(style, dict):
        return None
    out: Dict[str, Any] = {}
    color = style.get("color")
    if color == "status":
        out["color"] = "status"
    elif isinstance(color, (list, tuple)) and len(color) in (3, 4):
        try:
            rgba = [float(c) for c in color]
            if len(rgba) == 3:
                rgba.append(1.0)
            out["color"] = [min(1.0, max(0.0, c)) for c in rgba]
        except (TypeError, ValueError):
            pass
    opacity = style.get("opacity")
    if opacity is not None:
        try:
            out["opacity"] = min(1.0, max(0.0, float(opacity)))
        except (TypeError, ValueError):
            pass
    shape = style.get("shape")
    if shape in _VALID_BOX_SHAPES:
        out["shape"] = shape
    return out or None


def _find_patch_row(state: Dict[str, Any], patch_name: str) -> Optional[Dict[str, Any]]:
    target = str(patch_name).casefold()
    for p in state.get("patches", []) or []:
        if isinstance(p, dict) and str(p.get("patch_name", "")).casefold() == target:
            return p
    return None


def _find_source_row(state: Dict[str, Any], source_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if not source_id:
        return None
    for s in state.get("sources", []) or []:
        if isinstance(s, dict) and s.get("source_id") == source_id:
            return s
    return None


def set_patch_box_style(state: Dict[str, Any], patch_name: str, style: Any) -> bool:
    """Set (or clear, when *style* normalizes to None) the per-patch ``box_style``
    on the patch row.  Returns True iff a row matched.  Additive schema-v2."""
    row = _find_patch_row(state, patch_name)
    if row is None:
        return False
    clean = normalize_box_style(style)
    if clean is None:
        row.pop("box_style", None)
    else:
        row["box_style"] = clean
    return True


def set_source_box_style(state: Dict[str, Any], source_id: str, style: Any) -> bool:
    """Set (or clear) the per-source DEFAULT ``box_style`` (new patches inherit)."""
    row = _find_source_row(state, source_id)
    if row is None:
        return False
    clean = normalize_box_style(style)
    if clean is None:
        row.pop("box_style", None)
    else:
        row["box_style"] = clean
    return True


def resolve_box_style(state: Dict[str, Any], patch_name: str) -> Optional[Dict[str, Any]]:
    """The effective ``box_style`` for *patch_name*: the patch's own style, else
    its source's default, else ``None`` (→ the overlay's built-in default)."""
    row = _find_patch_row(state, patch_name)
    if row is None:
        return None
    ps = normalize_box_style(row.get("box_style"))
    if ps is not None:
        return ps
    src = _find_source_row(state, row.get("source_id"))
    if src is not None:
        return normalize_box_style(src.get("box_style"))
    return None


def set_processed_store(
    state: Dict[str, Any],
    *,
    source_id: Optional[str] = None,
    path: Optional[str] = None,
    rel_path: str,
    materialized_at: Optional[str] = None,
) -> str:
    """Write the additive ``processed_store`` pointer onto a source row (last-wins).

    The pointer records WHERE the streamed ⟨processed⟩ OME-Zarr store for this
    source lives so a reopen can re-discover it (Raw/Processed toggle Brick B):

    ``{"path": <workspace-relative posix>, "format": "ome-zarr",
       "materialized_at": <ISO>}``

    **The pointer is a re-discovery HINT only — NEVER the freshness gate.**
    Validity is decided by the store's OWN ``epicycle_processed`` attrs
    (``recipe_hash`` + source ``{mtime,size}`` fingerprint) via the controller's
    ``_fresh_processed_store_for``; the ``recipe_hash`` deliberately does NOT
    live in the pointer (a source-edit-but-recipe-unchanged store would read
    stale if the pointer hash were trusted).

    Routes through :func:`upsert_source` (last-wins for the pointer, ``"prepared"``
    unioned into ``origin`` — a streamed processed store only exists for a
    prepared/large-data source).  Returns the row's ``source_id``.
    """
    pointer = {
        "path": rel_path,
        "format": "ome-zarr",
        "materialized_at": materialized_at or _now(),
    }
    return upsert_source(
        state,
        path=path,
        source_id=source_id,
        processed_store=pointer,
        origin=["prepared"],
    )


def get_processed_store(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The ``processed_store`` pointer dict for a source *row*, or ``None`` when
    the row is absent / carries no (well-formed) pointer.  Never raises."""
    if not isinstance(row, dict):
        return None
    ptr = row.get("processed_store")
    return ptr if isinstance(ptr, dict) else None


# ---------------------------------------------------------------------------
# Registry queries (read-only; return fresh lists)
# ---------------------------------------------------------------------------
def all_patches(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every registered patch row, as a FRESH list.

    The caller may reorder / filter the result without mutating the stored
    registry.  Non-dict garbage is already dropped by :func:`load_state`; this
    is still tolerant of a hand-built state missing the ``patches`` key or
    holding stray non-dicts (never raises)."""
    rows = state.get("patches", []) if isinstance(state, dict) else []
    return [p for p in rows if isinstance(p, dict)]


def voxel_size_for_patch_path(
    state: Dict[str, Any], rel_path: str,
) -> Optional[Dict[str, Any]]:
    """What Step 1 knows about the voxel size of the patch file *rel_path*.

    Pure and additive (Step 4 reads it, nothing in
    Step 1 changes).  *rel_path* is workspace-relative; it is matched
    against each patch row's ``relative_path`` case- and slash-insensitively.

    Returns ``None`` when *rel_path* is not a registered patch.  Otherwise a
    dict:

    * ``"voxel_size"`` -- ``[z, y, x]`` from the patch's SOURCE row when that
      value is real (``_classify_voxel_source == "user"``), else ``None``.
      **Always ``None`` for a processed patch** (``source_kind ==
      "processed"``): the processed store may have been resampled by the
      shared geometry step, so the raw source's voxel size would be wrong.
      (``DataPreparationController._voxel_size_for_patch`` has that blind
      spot; it is not copied.)
    * ``"source_kind"`` -- ``"raw"`` or ``"processed"``.
    * ``"processed_store"`` -- the workspace-relative path of the source's
      processed store (a processed patch only; else ``None``), so the caller
      can read the store's own scale.

    Never raises.
    """
    try:
        target = _norm_path(str(rel_path).replace("\\", "/"))
        row = None
        for p in all_patches(state):
            rp = p.get("relative_path")
            if rp and _norm_path(str(rp).replace("\\", "/")) == target:
                row = p
                break
        if row is None:
            return None
        kind = str(row.get("source_kind") or "raw")
        src = _find_source_row(state, row.get("source_id"))
        voxel = None
        store = None
        if kind == "processed":
            ptr = get_processed_store(src)
            if ptr and ptr.get("path"):
                store = str(ptr["path"])
        elif src is not None and _classify_voxel_source(
                src.get("voxel_size")) == "user":
            vals = [float(v) for v in src.get("voxel_size")]
            if len(vals) == 3 and all(v > 0 for v in vals):
                voxel = vals
        return {"voxel_size": voxel, "source_kind": kind,
                "processed_store": store}
    except Exception:                        # noqa: BLE001 -- never raise
        return None


def patches_for_source(state: Dict[str, Any], source_id: str) -> List[Dict[str, Any]]:
    """Registered patch rows belonging to *source_id* (the per-FILE filter —
    parity with :func:`logic.count_patches_for_source`, but returns the rows).

    Returns an EMPTY list for a falsy *source_id*: legacy patches carry
    ``source_id == ""`` and must not all collapse into one 'source'."""
    if not source_id:
        return []
    return [p for p in all_patches(state) if p.get("source_id") == source_id]


# ---------------------------------------------------------------------------
# Large-data convert config (per-workspace)
# ---------------------------------------------------------------------------
def load_largedata_config(workspace: str) -> Optional[Dict[str, Any]]:
    """Return the persisted large-data convert config for *workspace*, or
    ``None``.

    ``None`` when there is no state file, no config saved, or the saved value is
    not a well-formed config dict (missing / garbage field) — so the caller falls
    back to the adaptive default.  Never raises (``load_state`` is
    corrupt-tolerant; the value is validated via
    :func:`packages.epicycle_largedata.config.normalize_convert_config`).
    """
    from packages.epicycle_largedata.config import normalize_convert_config

    state = load_state(workspace)
    return normalize_convert_config(state.get(LARGEDATA_CONFIG_KEY))


def save_largedata_config(workspace: str, config: Dict[str, Any]) -> None:
    """Persist *config* into the workspace's ``step1_state.json`` (atomic write,
    merged into the existing state).

    *config* must be a valid ``{preset, num_workers, chunk, chunk_zyx,
    compressor, downsample, min_size}`` dict (``chunk_zyx`` + ``compressor`` are
    the Brick 4b additions; both are filled in by ``normalize_convert_config``
    when an older saved config omits them, so the schema change is additive and
    a pre-4b state file still loads) — it is normalized/validated first, and a
    bad config raises
    ``ValueError`` (a caller passing garbage is a bug, unlike a corrupt file on
    load).  Only the recognized fields are stored (extra keys dropped).
    """
    from packages.epicycle_largedata.config import normalize_convert_config

    coerced = normalize_convert_config(config)
    if coerced is None:
        raise ValueError(
            f"save_largedata_config: invalid convert config {config!r}"
        )
    state = load_state(workspace)
    state[LARGEDATA_CONFIG_KEY] = coerced
    save_state(workspace, state)
