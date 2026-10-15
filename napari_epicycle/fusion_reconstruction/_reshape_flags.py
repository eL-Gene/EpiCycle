"""Reshape-flag sidecar - persist the curation-authored ``reshape_flag``
column beside the per-patch QC features CSV so it survives re-extraction.

Background (Brick R1): the reshape family records an arbiter verdict per label as a
string column named exactly ``reshape_flag``.  Values are ``"dividing"``
(kept whole - no membrane between the seeds), ``"low-evidence"`` (split, but
the wall was marginal) and ``"split-override"`` (the arbiter kept it whole and
the user pressed *Split anyway*).  The point of the column is that months
later an algorithmic split and a human-insisted one are still distinguishable.

The problem: when QC features are re-extracted, ONLY ``qc_tier`` is preserved
forward - a left-merge by label on active rows in
``controller._on_qc_extract_finished`` (``controller.py:~10535``).  Every other
curation-authored column, ``reshape_flag`` included, is dropped the first time
the user re-extracts.  A "dividing" verdict the tool worked to produce would
silently vanish.

The chosen fix is a small JSON sidecar beside the features CSV, written
at commit time and re-applied after every extraction - the same pattern and the
same rationale as ``extraction_context.json`` (``_qc_extraction_context.py``).
It survives extraction by construction, it is inspectable, and it does not
touch the shared QC merge path.

This module is deliberately **read/write + a df helper only - no UI, no
controller logic** - so it stays trivially testable without Qt / napari.

On-disk shape (``<patch>_reshape_flags.json``, beside ``<patch>_features.csv``)::

    {
      "schema_version": 1,
      "written_at": "2026-09-08T12:34:56.789012+00:00",
      "flags": {"12": "dividing", "77": "split-override"}
    }

JSON has string-only object keys, so label ids round-trip as strings on disk
and are normalised back to ``int`` on read.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Union

from ._paths import qc_features_csv

#: Filename suffix of the sidecar.  Replaces the ``_features.csv`` suffix of
#: the QC features CSV, so patch ``p01`` gets ``p01_reshape_flags.json``.
SIDECAR_NAME_SUFFIX = "_reshape_flags.json"

#: Sidecar schema version.  A file carrying any other value is ignored
#: (read as "no flags") rather than guessed at.
SCHEMA_VERSION = 1

#: The only values a flag may take.
VALID_FLAGS = ("dividing", "low-evidence", "split-override")

#: Suffix of the QC features CSV that :func:`sidecar_path` swaps out.
_FEATURES_CSV_SUFFIX = "_features.csv"

#: Default df column the flag is written into.
DEFAULT_FLAG_COLUMN = "reshape_flag"

_WS = Union[str, Path]


# ----------------------------------------------------------------------------
# Path
# ----------------------------------------------------------------------------

def sidecar_path(
    workspace: _WS, patch_name: str, run_id: Optional[str] = None,
) -> Path:
    """Path of the reshape-flag sidecar for one patch.

    Lives BESIDE the QC features CSV (same directory, same patch stem), so a
    named run's sidecar lands in that run's ``qc/`` dir and the default (flat)
    run's lands in ``artifacts/step3/qc/``.

    Pure - constructs a path, touches nothing on disk.
    """
    csv_path = qc_features_csv(workspace, patch_name, run_id)
    name = csv_path.name
    if name.endswith(_FEATURES_CSV_SUFFIX):
        stem = name[: -len(_FEATURES_CSV_SUFFIX)]
    else:  # defensive - _paths changed shape under us
        stem = csv_path.stem
    return csv_path.parent / f"{stem}{SIDECAR_NAME_SUFFIX}"


# ----------------------------------------------------------------------------
# Read
# ----------------------------------------------------------------------------

def read_flags(
    workspace: _WS, patch_name: str, run_id: Optional[str] = None,
) -> Dict[int, str]:
    """Read the sidecar, returning ``{label_id: flag}`` with INT keys.

    Corrupt-tolerant by design: a missing file, unparseable JSON, an
    unexpected ``schema_version``, a non-dict payload or a non-dict ``flags``
    all degrade to ``{}``.  Individual bad entries (a key that is not an
    integer, a value that is not in :data:`VALID_FLAGS`) are skipped while the
    good ones survive.

    This function NEVER raises and NEVER deletes or rewrites the bad file - a
    human (or a later successful write) can still inspect it.
    """
    try:
        path = sidecar_path(workspace, patch_name, run_id)
    except (TypeError, ValueError):
        return {}
    try:
        if not path.is_file():
            return {}
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    if data.get("schema_version") != SCHEMA_VERSION:
        return {}
    flags = data.get("flags")
    if not isinstance(flags, dict):
        return {}

    out: Dict[int, str] = {}
    for key, value in flags.items():
        # JSON turns int dict keys into strings - normalise back to int.
        try:
            label_id = int(key)
        except (TypeError, ValueError):
            continue
        if not isinstance(value, str) or value not in VALID_FLAGS:
            continue
        out[label_id] = value
    return out


# ----------------------------------------------------------------------------
# Write
# ----------------------------------------------------------------------------

def _normalize_for_write(flags: Mapping) -> Dict[int, str]:
    """Validate + normalise a caller mapping into ``{int: valid flag}``.

    Raises ``ValueError`` naming the offending value / key, so a typo'd flag
    can never reach disk.
    """
    if not isinstance(flags, Mapping):
        raise ValueError(
            f"flags must be a mapping of label id -> flag, got "
            f"{type(flags).__name__}"
        )
    out: Dict[int, str] = {}
    for key, value in flags.items():
        try:
            label_id = int(key)
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid label id {key!r} in reshape flags; label ids must "
                f"be integers."
            )
        if not isinstance(value, str) or value not in VALID_FLAGS:
            raise ValueError(
                f"Invalid reshape flag {value!r} for label {label_id}; "
                f"allowed values are {VALID_FLAGS}."
            )
        out[label_id] = value
    return out


def write_flags(
    workspace: _WS,
    patch_name: str,
    flags: Mapping,
    run_id: Optional[str] = None,
) -> Path:
    """Write ``flags`` to the patch's sidecar, replacing any existing one.

    Every value is validated against :data:`VALID_FLAGS` BEFORE anything is
    written, so a typo'd flag raises ``ValueError`` and leaves the existing
    file untouched.

    The write is atomic (``tempfile.mkstemp`` in the same directory then
    ``os.replace``), so an interrupted write cannot leave a truncated /
    corrupt sidecar behind.

    Writing an EMPTY mapping REMOVES the sidecar rather than leaving an empty
    one on disk - "no flags" and "no file" are the same state.

    Returns:
        The sidecar path (whether it was written or removed).
    """
    normalized = _normalize_for_write(flags)
    path = sidecar_path(workspace, patch_name, run_id)

    if not normalized:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "written_at": datetime.now(timezone.utc).isoformat(),
        "flags": {str(k): v for k, v in sorted(normalized.items())},
    }
    text = json.dumps(payload, indent=2)

    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_name, str(path))
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return path


# ----------------------------------------------------------------------------
# Mutating helpers
# ----------------------------------------------------------------------------

def merge_flags(
    workspace: _WS,
    patch_name: str,
    new_flags: Mapping,
    run_id: Optional[str] = None,
) -> Dict[int, str]:
    """Merge ``new_flags`` into the stored sidecar (last wins) and rewrite it.

    This is the commit-time entry point: a reshape accept hands over just the
    labels it touched, and the flags recorded by earlier accepts survive.

    Raises ``ValueError`` (before touching disk) if any new value is not in
    :data:`VALID_FLAGS`.

    Returns:
        The merged ``{label_id: flag}`` mapping that is now on disk.
    """
    incoming = _normalize_for_write(new_flags)
    merged = read_flags(workspace, patch_name, run_id)
    merged.update(incoming)
    write_flags(workspace, patch_name, merged, run_id)
    return merged


def drop_flags(
    workspace: _WS,
    patch_name: str,
    label_ids: Iterable,
    run_id: Optional[str] = None,
) -> Dict[int, str]:
    """Remove the given label ids from the sidecar and rewrite it.

    Used when a label is superseded by a later reshape or deleted outright -
    its flag should not linger and get re-applied to a recycled id.

    Ids that carry no flag (or that are not integers) are ignored.  If nothing
    remains, the sidecar is removed.

    Returns:
        The remaining ``{label_id: flag}`` mapping that is now on disk.
    """
    remaining = read_flags(workspace, patch_name, run_id)
    for raw_id in label_ids or ():
        try:
            label_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        remaining.pop(label_id, None)
    write_flags(workspace, patch_name, remaining, run_id)
    return remaining


# ----------------------------------------------------------------------------
# DataFrame application
# ----------------------------------------------------------------------------

def apply_flags_to_df(df, flags: Mapping, *, column: str = DEFAULT_FLAG_COLUMN):
    """Return a COPY of ``df`` with the reshape flags written into ``column``.

    The caller's DataFrame is never mutated in place - a copy is returned.
    This is the post-extraction re-apply step: a freshly extracted QC features
    DataFrame carries no ``reshape_flag`` column, so the sidecar's flags are
    grafted back on by matching the ``label`` column (the Step 3 QC label-id
    column - see ``controller._on_qc_extract_finished`` and
    ``qc_features.py``).

    Rows with no stored flag get the EMPTY STRING, never ``NaN`` - a string
    column with NaN does not round-trip cleanly through CSV.

    Defensive: if ``df`` has no ``label`` column there is nothing to match on,
    so the DataFrame is returned unchanged (not a copy-with-a-blank-column).
    Every other column is left exactly as it was.
    """
    if df is None:
        return df
    if "label" not in getattr(df, "columns", ()):
        return df

    out = df.copy()
    normalized: Dict[int, str] = {}
    if isinstance(flags, Mapping):
        for key, value in flags.items():
            try:
                label_id = int(key)
            except (TypeError, ValueError):
                continue
            if isinstance(value, str) and value in VALID_FLAGS:
                normalized[label_id] = value

    def _lookup(label_value) -> str:
        try:
            return normalized.get(int(label_value), "")
        except (TypeError, ValueError):
            return ""

    out[column] = [_lookup(v) for v in out["label"]]
    return out


__all__ = [
    "SIDECAR_NAME_SUFFIX",
    "SCHEMA_VERSION",
    "VALID_FLAGS",
    "DEFAULT_FLAG_COLUMN",
    "sidecar_path",
    "read_flags",
    "write_flags",
    "merge_flags",
    "drop_flags",
    "apply_flags_to_df",
]
