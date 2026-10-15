"""Where a patch's voxel size comes from -- the resolver Steps 3 and 4 share.

Lifted VERBATIM from Step 4's ``segmentation/_step4_models.py`` in brick R2b
("The resolver").  ``_step4_models`` re-exports every name here, so Step 4's callers and
tests are unchanged.  Two deliberate changes since the lift (its audits): a
``record`` that is not a dict is read as no record, so "never raises" holds;
and ``_rel_to`` refuses only a path that climbs out (``..`` / ``../``), not
an in-workspace folder whose name starts with two dots.

It lives in Step 1's package because Step 1 owns the registry it reads
(``_state.voxel_size_for_patch_path``), and ``ui_shell`` may not import a
step package.  Qt-free and napari-free: it imports only ``_state`` and
``_paths`` (lazily, as before) and ``packages.epicycle_largedata.axes`` --
never ``segmentation`` or ``fusion_reconstruction``.

``read_step1_state`` joined it in R2b part 2, lifted VERBATIM from
``_step4_models`` (which re-exports it): Step 3 resolves each patch's
voxel size before Predict and may not import a Step 4 module.

A test that stubs the file read patches ``_read_voxel_spacing`` on THIS
module: the resolver reads this module's global, so a patch on
``_step4_models`` (a re-exported name) would reach nothing.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

#: where a voxel size came from, in the resolver's order of trust reversed
#: (a ``user`` value always wins).  Lifted VERBATIM from ``_step4_models``
#: in R7a: Step 3's ``patch_spacing``
#: map validates its ``source`` against it; ``_step4_models`` re-exports it.
SPACING_SOURCES = ("step1", "file", "user")


# ---------------------------------------------------------------------------
# Step 1's state, read-only
# ---------------------------------------------------------------------------


def read_step1_state(workspace: str) -> Optional[Dict[str, Any]]:
    """Step 1's state, read with a guarded ``json.loads``.

    NEVER ``_state.load_state``: that renames a corrupt ``step1_state.json``
    and fires Step 1's corrupt hook (grill G1-7).  ``None`` when the
    file is missing or unreadable.
    """
    if not workspace:
        return None
    try:
        from napari_epicycle.data_preparation._paths import step1_state_path
        path = step1_state_path(workspace)
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:      # noqa: BLE001 -- read-only, never raise
        return None


def _real(zyx) -> Optional[list]:
    """``[z, y, x]`` floats when *zyx* is a real spacing, else None.

    A 1 / 1 / 1 result counts as none (it is the "unknown" default)."""
    if zyx is None:
        return None
    try:
        vals = [float(v) for v in zyx]
    except (TypeError, ValueError):
        return None
    if len(vals) != 3 or not all(v > 0 for v in vals):
        return None
    if all(abs(v - 1.0) < 1e-9 for v in vals):
        return None
    return vals


def _positive(zyx) -> Optional[list]:
    """``[z, y, x]`` floats when all three are positive (1/1/1 allowed)."""
    try:
        vals = [float(v) for v in zyx]
    except (TypeError, ValueError):
        return None
    if len(vals) != 3 or not all(v > 0 for v in vals):
        return None
    return vals


def same_spacing(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]],
                 tol: float = 0.5e-3) -> bool:
    """Two spacings are the same NUMBERS (their sources may differ;
    audit D6), to half a spin-box step (3 decimals; re-audit M3), so a
    file's 0.10833 equals the 0.108 a spin box holds.  ``None`` equals only
    ``None``."""
    if not a or not b:
        return not a and not b
    za, zb = _positive(a.get("zyx") or ()), _positive(b.get("zyx") or ())
    if za is None or zb is None:
        return za is None and zb is None
    return all(abs(x - y) <= tol + 1e-12 for x, y in zip(za, zb))


def _read_voxel_spacing(path) -> Optional[Tuple[float, float, float]]:
    try:
        from packages.epicycle_largedata.axes import read_voxel_spacing
        return read_voxel_spacing(path)
    except Exception:      # noqa: BLE001
        return None


def _rel_to(path: str, workspace: str) -> Optional[str]:
    try:
        rel = os.path.relpath(str(path), str(workspace))
    except (OSError, ValueError):
        return None
    rel = rel.replace(os.sep, "/")
    if rel == ".." or rel.startswith("../"):
        return None
    return rel


def resolve_stack_spacing(record: Optional[Dict[str, Any]],
                          step1_state: Optional[Dict[str, Any]],
                          path: str,
                          workspace: str = "",
                          *,
                          read_file: bool = True,
                          ) -> Optional[Dict[str, Any]]:
    """The first rule that gives a real spacing wins:

    1. the record's own ``spacing`` with ``source == "user"`` -- never
       overwritten;
    2. Step 1: *path* is a registered Step 1 patch whose source has a real
       voxel size (``source="step1"``); a PROCESSED patch takes its store's
       own scale instead (``read_file`` only), never the raw source's, and
       falls to rule 3 when that is unknown;
    3. the file's own tags (``read_voxel_spacing``; ``read_file`` only);
    4. ``None``: the stack runs in voxels.

    Returns ``{"zyx": [z, y, x], "source": ...}`` or ``None``.  Never raises.
    ``read_file=False`` is the adopt path: no TIFF or store is opened.
    """
    rec = record if isinstance(record, dict) else {}
    own = rec.get("spacing")
    if isinstance(own, dict) and own.get("source") == "user":
        # A user's choice is kept even at 1 / 1 / 1 (audit D10): the
        # explicit source is the marker that it was chosen.
        zyx = _positive(own.get("zyx"))
        if zyx is not None:
            return {"zyx": zyx, "source": "user"}

    if step1_state and workspace and path:
        rel = _rel_to(path, workspace)
        if rel is not None:
            try:
                from napari_epicycle.data_preparation._state import (
                    voxel_size_for_patch_path,
                )
                info = voxel_size_for_patch_path(step1_state, rel)
            except Exception:      # noqa: BLE001
                info = None
            if info is not None:
                if info.get("source_kind") == "processed":
                    store = info.get("processed_store")
                    if read_file and store:
                        zyx = _real(_read_voxel_spacing(
                            str(Path(workspace) / store)))
                        if zyx is not None:
                            return {"zyx": zyx, "source": "step1"}
                else:
                    zyx = _real(info.get("voxel_size"))
                    if zyx is not None:
                        return {"zyx": zyx, "source": "step1"}

    if read_file and path:
        zyx = _real(_read_voxel_spacing(path))
        if zyx is not None:
            return {"zyx": zyx, "source": "file"}
    return None


# ---------------------------------------------------------------------------
# The workspace's one voxel size per patch: ``patch_spacing.json`` (S1)
# ---------------------------------------------------------------------------
#
# Shape: ``{"schema_version": 1, "patches":
# {"<workspace-relative TIFF path>": {"zyx", "source", "set_at"}}}`` at the
# workspace root, shared by Steps 3 and 4.  Every write reloads the file,
# merges ONE entry and replaces it atomically, so two controllers cannot
# overwrite each other's entries.  Qt-free; it never raises on a read.

PATCH_SPACING_FILENAME = "patch_spacing.json"
PATCH_SPACING_SCHEMA = 1


def patch_spacing_path(workspace: str) -> Path:
    return Path(str(workspace)) / PATCH_SPACING_FILENAME


def patch_spacing_key(path: str, workspace: str) -> str:
    """The file's key for a patch TIFF: workspace-relative with forward
    slashes (``_rel_to``); a path outside the workspace is keyed absolute."""
    if not path:
        return ""
    rel = _rel_to(path, workspace) if workspace else None
    if rel is not None:
        return rel
    return os.path.abspath(str(path)).replace(os.sep, "/")


def _valid_entry(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict) or raw.get("source") not in SPACING_SOURCES:
        return None
    zyx = _positive(raw.get("zyx") or ())
    if zyx is None:
        return None
    out = {"zyx": zyx, "source": raw["source"]}
    if isinstance(raw.get("set_at"), str):
        out["set_at"] = raw["set_at"]
    return out


def _aside_name(path: Path) -> str:
    """``<file>.corrupt-<stamp>``, never an existing name (audit L2)."""
    import time as _time
    stamp = _time.strftime("%Y%m%d-%H%M%S")
    cand = str(path) + ".corrupt-" + stamp
    n = 1
    while os.path.exists(cand):
        n += 1
        cand = f"{path}.corrupt-{stamp}-{n}"
    return cand


def _read_patch_spacing_raw(workspace: str, *, strict: bool = False,
                            quarantine: bool = True
                            ) -> Dict[str, Dict[str, Any]]:
    """Every readable entry.

    Only an UNPARSEABLE file (bad JSON, bad bytes, the wrong shape) is
    corruption: it is renamed ``.corrupt-<stamp>`` (unless *quarantine* is
    False -- the read-only adopt paths) and read as empty.  A file that
    cannot be OPENED (a lock held by OneDrive, an antivirus, a second
    napari) is not corrupt (audit H1): a plain read answers empty and
    renames nothing; with *strict* (every writer) the ``OSError`` is raised,
    so a writer never rebuilds the file from an empty read."""
    if not workspace:
        return {}
    path = patch_spacing_path(workspace)
    try:
        if not path.is_file():
            return {}
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = None
    except OSError:
        if strict:
            raise
        return {}
    try:
        if text is None:
            raise ValueError("not UTF-8")
        data = json.loads(text)
        if not isinstance(data, dict) or not isinstance(
                data.get("patches", {}), dict):
            raise ValueError("not a patch_spacing file")
    except ValueError:
        if quarantine:
            try:
                if path.is_file():
                    os.replace(str(path), _aside_name(path))
                    print(f"[SPACING-FILE] {path.name} unreadable -- renamed "
                          f"aside, read as empty")
            except OSError:
                pass
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for key, entry in (data.get("patches") or {}).items():
        valid = _valid_entry(entry)
        if valid is not None:
            out[str(key)] = valid
    return out


def read_patch_spacing_file(workspace: str, *, quarantine: bool = True
                            ) -> Dict[str, Dict[str, Any]]:
    """``{key: {"zyx", "source"[, "set_at"]}}`` (fresh dicts).  Never
    raises.  ``quarantine=False`` (the read-only adopt) leaves a corrupt
    file where it is."""
    return _read_patch_spacing_raw(workspace, quarantine=quarantine)


def find_patch_spacing(entries: Dict[str, Dict[str, Any]], key: str
                       ) -> Optional[Dict[str, Any]]:
    """The entry for *key*: exact, else equal after ``os.path.normcase``
    (case-insensitive on Windows; the stored case is unchanged, L3)."""
    stored = _stored_key(entries, key)
    return entries.get(stored) if stored is not None else None


def _stored_key(entries, key: str) -> Optional[str]:
    if key in entries:
        return key
    folded = os.path.normcase(key)
    for k in entries:
        if os.path.normcase(k) == folded:
            return k
    return None


def _write_patch_spacing_all(workspace: str, entries) -> None:
    import tempfile
    path = patch_spacing_path(workspace)
    payload = {"schema_version": PATCH_SPACING_SCHEMA, "patches": entries}
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".patch_spacing_",
                               suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _now_iso() -> str:
    import datetime as _dt
    return _dt.datetime.now().replace(microsecond=0).isoformat()


def write_patch_spacing_entry(workspace: str, key: str, zyx, source: str
                              ) -> bool:
    """Set one patch's entry (reload, merge, atomic replace).  ``True`` when
    the file changed.  A ``step1`` / ``file`` value never replaces a ``user``
    one; the same value again writes nothing.  Raises ``ValueError`` for a
    bad source or spacing."""
    if source not in SPACING_SOURCES:
        raise ValueError(f"unknown spacing source {source!r}; expected one "
                         f"of {SPACING_SOURCES}")
    vals = _positive(zyx if zyx is not None else ())
    if vals is None:
        raise ValueError(f"spacing must be three positive numbers, got "
                         f"{zyx!r}")
    if not workspace or not key:
        return False
    entries = _read_patch_spacing_raw(workspace, strict=True)
    key = _stored_key(entries, key) or key
    current = entries.get(key)
    if current and current["source"] == "user" and source != "user":
        return False
    if current and current["source"] == source and \
            all(abs(a - b) <= 1e-12 for a, b in zip(current["zyx"], vals)):
        return False
    entries[key] = {"zyx": vals, "source": source, "set_at": _now_iso()}
    _write_patch_spacing_all(workspace, entries)
    return True


def clear_patch_spacing_entry(workspace: str, key: str, *,
                              user_too: bool = False) -> bool:
    """Drop one entry; a ``user`` one only with ``user_too``.  ``True`` when
    an entry was removed."""
    if not workspace or not key:
        return False
    entries = _read_patch_spacing_raw(workspace, strict=True)
    key = _stored_key(entries, key) or key
    current = entries.get(key)
    if current is None or (current["source"] == "user" and not user_too):
        return False
    del entries[key]
    _write_patch_spacing_all(workspace, entries)
    return True
