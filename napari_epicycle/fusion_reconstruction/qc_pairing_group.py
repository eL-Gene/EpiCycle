"""The pairing check as a QC feature group, and what feeds it.

:mod:`.qc_pairing` holds the rule (numpy only).  This module puts it into
the QC feature extraction: the feature-group body that reads the other
run's labels file, pairs the two runs and writes five match columns into
the feature table; the bundle that names the other run; the refusals; the
provenance recorded beside the features; the blanking of the match
columns when the active run is edited; and the listing of the runs a patch
can be compared with.

The other run's file is its RECONSTRUCTION labels (``labels_3d.tif``), the
file that run's own QC reads.  It is read inside the group, on the
extraction's worker or subprocess, and freed when the group returns.

The columns, per active object (``match_count``, ``match_id``,
``match_shared`` are integers; ``match_overlap`` and
``match_minor_ratio`` are floats in [0, 1]):

* measured, with links: the numbers of the rule;
* measured, no link: count 0, id 0, shared 0, overlap 0.0, minor NaN;
* too small to judge: count 0, id 0, shared 0, overlap NaN, minor NaN;
* blanked after an edit: count, shared, overlap and minor NaN, id 0.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import _qc_pairing_text as T
from .qc_pairing import (
    COLUMNS as MATCH_COLUMNS,
    FLOOR_FACTOR,
    STATUS_NOT_MEASURED,
    PairingCancelled,
    object_volumes_and_border,
    pair_runs,
    run_floor,
    status_from_columns,
)

#: two objects are linked when they share at least this share of the
#: smaller one
LINK_FLOOR = 0.20
#: each run's size floor is this share of its median interior object
SIZE_FACTOR = FLOOR_FACTOR

#: the match columns written as integers
INT_COLUMNS = ("match_count", "match_id", "match_shared")
#: the match columns written as floats
FLOAT_COLUMNS = ("match_overlap", "match_minor_ratio")
#: the columns set to NaN when a row is blanked (``match_id`` goes to 0)
_BLANK_NAN = ("match_count", "match_shared", "match_overlap",
              "match_minor_ratio")

#: the key under which the group leaves its result in the bundle
OUTCOME_KEY = "outcome"
#: the key under which the result rides a re-read feature table's attrs
ATTRS_KEY = "_pairing"

_DEFAULT_RUN_ID = "default"


# --------------------------------------------------------------------------- #
# the bundle
# --------------------------------------------------------------------------- #
def make_match_bundle(*, path, run, run_display_name="",
                        link_floor=LINK_FLOOR, factor=SIZE_FACTOR,
                        active_stack_name="", other_stack_name="",
                        other_labels_shape=None) -> Dict[str, Any]:
    """The one input the pairing group needs, as plain JSON-safe values.

    ``path`` is the other run's labels file; ``run`` its slug and
    ``run_display_name`` its name; ``link_floor`` and ``factor`` the rule's
    constants.  The two stack names (file names only) and the other run's
    labels shape, as its manifest records them, let the group refuse before
    it reads the file; empty means unknown and is not checked.
    """
    shape = None
    if other_labels_shape is not None:
        try:
            shape = [int(s) for s in other_labels_shape]
        except (TypeError, ValueError):
            shape = None
    return {
        "path": str(path),
        "run": str(run),
        "run_display_name": str(run_display_name or run),
        "link_floor": float(link_floor),
        "factor": float(factor),
        "active_stack_name": str(active_stack_name or ""),
        "other_stack_name": str(other_stack_name or ""),
        "other_labels_shape": shape,
    }


def pairing_outcome(bundle) -> Optional[Dict[str, Any]]:
    """What the group left in ``bundle`` (``None`` when it did not run)."""
    if not isinstance(bundle, dict):
        return None
    out = bundle.get(OUTCOME_KEY)
    return dict(out) if isinstance(out, dict) else None


# --------------------------------------------------------------------------- #
# the group body
# --------------------------------------------------------------------------- #
def cross_run_pairing(*, labels, df, match_labels, **kw):
    """Add the five match columns to ``df`` and return it.

    ``labels`` is the active run's labels volume, ``df`` the feature table
    built from it (its ``label`` column is the authority: rows are matched
    to the rule's table by label value, and no row is added or dropped).
    ``match_labels`` is the bundle of :func:`make_match_bundle`; the
    result (or the refusal sentence) is left in it under ``"outcome"``.

    Refused, with ``df`` returned unchanged: the two runs were made from
    different stack files, their labels differ in shape, the other run's
    file cannot be read as integer labels, or the comparison itself fails
    (out of memory, a bad file: the error's words go into the sentence).
    A cancel (:class:`.qc_pairing.PairingCancelled`) and any
    ``BaseException`` are not caught: they stop the extraction.
    """
    t0 = time.perf_counter()
    b = match_labels
    slug = str(b.get("run") or "")
    name = str(b.get("run_display_name") or slug)

    def _refuse(sentence):
        b[OUTCOME_KEY] = {"run": slug, "display_name": name,
                          "refused": sentence}
        # the sentence itself goes to the extraction log once, by the caller
        print(f"[QC-PAIR] other={slug} skipped (see log)", flush=True)
        return df

    a_stack = str(b.get("active_stack_name") or "")
    o_stack = str(b.get("other_stack_name") or "")
    # file names only; Windows names do not tell case apart
    if a_stack and o_stack and a_stack.casefold() != o_stack.casefold():
        return _refuse(T.REFUSE_STACK.format(
            other=name, other_file=o_stack, this_file=a_stack))
    shape = tuple(int(s) for s in np.shape(labels))
    known = b.get("other_labels_shape")
    if known and tuple(int(s) for s in known[-3:]) != shape[-3:]:
        return _refuse(T.REFUSE_SHAPE.format(
            other=name, a=T.shape_words(known[-3:]),
            b=T.shape_words(shape[-3:])))

    try:
        import tifffile
        other = tifffile.imread(b["path"])
    except Exception as exc:      # noqa: BLE001 -- a refusal, never a crash
        print(f"[QC-PAIR] other={slug} read failed: {exc!r}", flush=True)
        return _refuse(T.REFUSE_UNREADABLE.format(other=name))
    if tuple(other.shape) != shape:
        sentence = T.REFUSE_SHAPE.format(
            other=name, a=T.shape_words(other.shape), b=T.shape_words(shape))
        del other
        return _refuse(sentence)
    if other.dtype.kind not in "ui":
        del other
        return _refuse(T.REFUSE_UNREADABLE.format(other=name))

    lf = float(b.get("link_floor", LINK_FLOOR))
    factor = float(b.get("factor", SIZE_FACTOR))
    try:
        _, vols, border = object_volumes_and_border(labels)
        floor_active = run_floor(vols, border, factor=factor)
        _, vols, border = object_volumes_and_border(other)
        floor_other = run_floor(vols, border, factor=factor)
        del vols, border
        table = pair_runs(labels, other, link_floor=lf,
                          floor_active=floor_active, floor_other=floor_other)
        del other
        # into a copy first: a failure half-way leaves the table untouched
        cols = pd.DataFrame(index=df.index)
        if "label" in df.columns:
            cols["label"] = df["label"]
        write_match_columns(cols, table)
    except PairingCancelled:
        raise
    except Exception as exc:      # noqa: BLE001 -- a refusal, never a crash
        print(f"[QC-PAIR] other={slug} comparison failed: {exc!r}",
              flush=True)
        failed = str(exc).strip() or type(exc).__name__
    else:
        failed = None
    if failed is not None:
        # the other run's volume is freed before the refusal (the error and
        # its frames are gone once the except block ends)
        other = None
        return _refuse(T.REFUSE_FAILED.format(other=name, reason=failed))
    for c in INT_COLUMNS + FLOAT_COLUMNS:
        df[c] = cols[c]
    links = int(np.asarray(table.match_count).sum())
    wall_ms = (time.perf_counter() - t0) * 1000.0
    b[OUTCOME_KEY] = {
        "run": slug,
        "display_name": name,
        "file": os.path.basename(str(b.get("path") or "")),
        "link_floor": lf,
        "floor_active_vx": int(floor_active),
        "floor_other_vx": int(floor_other),
        "links": links,
        "wall_ms": float(round(wall_ms, 1)),
        "refused": None,
    }
    print(
        f"[QC-PAIR] other={slug} "
        f"file={os.path.basename(str(b.get('path') or ''))} "
        f"floor_active={int(floor_active)}vx "
        f"floor_other={int(floor_other)}vx link={lf:g} links={links} "
        f"wall={wall_ms:.0f}ms",
        flush=True,
    )
    return df


def write_match_columns(df, table) -> None:
    """Write the rule's five columns into ``df``, matched on ``label``.

    The feature table is the authority: an id the rule found but the table
    does not hold is dropped; a table row the rule never saw reads count 0,
    id 0, shared 0 and NaN ratios, so the integer columns stay integers.
    """
    n = len(df)
    if n and "label" in df.columns:
        lab = pd.to_numeric(df["label"], errors="coerce").to_numpy()
    else:
        lab = np.zeros(n, np.float64)
    key = np.asarray(table.label).astype(np.int64)
    pos = pd.Series(np.arange(key.size, dtype=np.int64), index=key)
    finite = np.isfinite(lab.astype(np.float64))
    looked = np.full(n, np.nan)
    if finite.any():
        looked[finite] = pos.reindex(
            lab[finite].astype(np.int64)).to_numpy(dtype=np.float64)
    found = np.isfinite(looked)
    p = looked[found].astype(np.int64)
    cols = table.columns()
    for c in INT_COLUMNS:
        out = np.zeros(n, np.int64)
        out[found] = np.asarray(cols[c]).astype(np.int64)[p]
        df[c] = out
    for c in FLOAT_COLUMNS:
        out = np.full(n, np.nan, np.float64)
        out[found] = np.asarray(cols[c], dtype=np.float64)[p]
        df[c] = out


def blank_match_row(row: dict, columns) -> None:
    """Blank the match values of one row dict about to join a table with
    ``columns`` (an edited label: its match values are unknown until the
    next extraction).  NaN for count, shared and both ratios, 0 for
    ``match_id``; nothing when the table has no match column."""
    cols = set(columns)
    for c in _BLANK_NAN:
        if c in cols:
            row[c] = np.nan
    if "match_id" in cols:
        row["match_id"] = 0


def blank_match_columns(df, mask) -> bool:
    """Blank the match columns on the rows of ``mask`` (an edit made
    their values out of date): count, shared and both ratios to NaN (the
    columns become floats), ``match_id`` to 0.  ``True`` when ``df`` has
    any match column."""
    present = [c for c in MATCH_COLUMNS if c in df.columns]
    if not present:
        return False
    for c in _BLANK_NAN:
        if c in df.columns:
            if not pd.api.types.is_float_dtype(df[c]):
                df[c] = df[c].astype(np.float64)
            df.loc[mask, c] = np.nan
    if "match_id" in df.columns:
        df.loc[mask, "match_id"] = 0
    return True


def active_label_rows(df, label_ids):
    """The rows of ``df`` that hold one of ``label_ids`` and are active
    (``label_status`` "active", or every row when the table has no such
    column): the rows an edit makes out of date.  Superseded and deleted
    rows keep what they had."""
    if isinstance(label_ids, (list, tuple, set, frozenset)):
        ids = [int(i) for i in label_ids]
    else:
        ids = [int(label_ids)]
    mask = df["label"].isin(ids)
    if "label_status" in df.columns:
        mask = mask & df["label_status"].eq("active")
    return mask


def has_match_columns(df) -> bool:
    """Whether ``df`` carries any match column."""
    return df is not None and any(c in df.columns for c in MATCH_COLUMNS)


def has_all_match_columns(df) -> bool:
    """Whether ``df`` carries all five match columns (what the QC plot and
    the triage histogram need to offer the Colour by item)."""
    return df is not None and all(c in df.columns for c in MATCH_COLUMNS)


def _float_column(df, name) -> np.ndarray:
    """``df[name]`` as float64, every missing value (NaN, ``pd.NA``, text)
    as NaN."""
    return pd.to_numeric(df[name], errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan)


def match_words(df) -> np.ndarray:
    """The match word of every row of ``df`` (an object array; ``df`` has
    the five match columns): the check's own precedence on the saved
    columns, and "not measured" for a row that is not active (a deleted or
    superseded label)."""
    words = status_from_columns(
        _float_column(df, "match_count"),
        _float_column(df, "match_shared"),
        _float_column(df, "match_overlap"))
    if "label_status" in df.columns:
        active = (df["label_status"].fillna("active").astype(str)
                  == "active").to_numpy()
        words[~active] = STATUS_NOT_MEASURED
    return words


def values_predate_edits_on_disk(df) -> bool:
    """What a table read from disk says about its match values: some
    were blanked by an edit (``match_count`` empty on a row)."""
    if df is None or "match_count" not in df.columns:
        return False
    try:
        return bool(pd.to_numeric(df["match_count"],
                                  errors="coerce").isna().any())
    except Exception:      # noqa: BLE001 -- a hint, never a crash
        return False


# --------------------------------------------------------------------------- #
# provenance and staleness
# --------------------------------------------------------------------------- #
def file_stat(path) -> Optional[Dict[str, Any]]:
    """``{"mtime", "size"}`` of a file, or ``None``.  Never raises."""
    try:
        st = os.stat(str(path))
    except (OSError, TypeError, ValueError):
        return None
    return {"mtime": float(st.st_mtime), "size": int(st.st_size)}


def patch_file_fields(labels_path) -> Dict[str, Any]:
    """The active labels file's size and mtime, as the sidecar records
    them (recorded, not compared)."""
    st = file_stat(labels_path) if labels_path else None
    return {
        "patch_file_size": st["size"] if st else None,
        "patch_file_mtime": st["mtime"] if st else None,
    }


def workspace_relative(path, workspace) -> str:
    """``path`` relative to ``workspace`` with forward slashes, or the path
    itself when it lies outside it."""
    try:
        rel = os.path.relpath(str(path), str(workspace))
    except (ValueError, TypeError):
        return str(path).replace("\\", "/")
    if rel.startswith(".."):
        return str(path).replace("\\", "/")
    return rel.replace("\\", "/")


def match_file_path(ctx, workspace) -> str:
    """The other run's file a sidecar names, as an absolute path ("")."""
    if not isinstance(ctx, dict):
        return ""
    rel = ctx.get("match_file")
    if not rel:
        return ""
    p = Path(str(rel))
    return str(p if p.is_absolute() else Path(str(workspace)) / p)


def pairing_context_fields(provenance, outcome) -> Dict[str, Any]:
    """The extraction-context fields of one extraction.

    ``provenance`` is what the main thread recorded at dispatch (the patch
    file's size and mtime, and when a run was picked: ``match_run``,
    ``match_run_display_name``, ``match_file``, ``match_file_mtime``,
    ``match_file_size``, ``link_floor``; ``match_run`` ``None`` =
    "(none)" chosen, recorded as such); ``outcome`` what the group
    returned.  A refusal keeps the run (it stays the next default) and its
    sentence, not the file facts.
    """
    out: Dict[str, Any] = {}
    prov = dict(provenance or {})
    out["patch_file_size"] = prov.get("patch_file_size")
    out["patch_file_mtime"] = prov.get("patch_file_mtime")
    if "match_run" in prov and prov["match_run"] is None:
        # "(none)" was chosen for this patch: recorded, so it survives
        out["match_run"] = None
        return out
    if not prov.get("match_run"):
        return out
    out["match_run"] = prov.get("match_run")
    out["match_run_display_name"] = prov.get("match_run_display_name")
    res = dict(outcome or {})
    if res.get("refused"):
        out["match_skipped"] = res["refused"]
        return out
    if not res:
        return out
    for k in ("match_file", "match_file_mtime", "match_file_size"):
        out[k] = prov.get(k)
    out["link_floor"] = res.get("link_floor", prov.get("link_floor"))
    out["floor_active_vx"] = res.get("floor_active_vx")
    out["floor_other_vx"] = res.get("floor_other_vx")
    return out


def _stat_pair(stat) -> Tuple[Optional[float], Optional[int]]:
    if stat is None:
        return None, None
    if isinstance(stat, dict):
        m, s = stat.get("mtime"), stat.get("size")
    else:
        m, s = getattr(stat, "st_mtime", None), getattr(stat, "st_size", None)
    try:
        return (None if m is None else float(m),
                None if s is None else int(s))
    except (TypeError, ValueError):
        return None, None


def match_stale_reason(ctx, current_file_stat) -> Optional[str]:
    """The stale sentence when the other run's file changed since the
    match values were measured, else ``None``.

    ``ctx`` is the extraction-context sidecar; ``current_file_stat`` the
    file's stat now (an ``os.stat_result`` or ``{"mtime", "size"}``;
    ``None`` = the file is gone).  A sidecar with no other run's file
    recorded has nothing to compare.  Shown with the match values only, never as
    a reason the whole table is stale.
    """
    if not isinstance(ctx, dict):
        return None
    m0, s0 = ctx.get("match_file_mtime"), ctx.get("match_file_size")
    if m0 is None or s0 is None:
        return None
    other = str(ctx.get("match_run_display_name")
                or ctx.get("match_run") or "")
    stale = T.STALE_MATCH.format(other=other)
    m1, s1 = _stat_pair(current_file_stat)
    if m1 is None or s1 is None:
        return stale
    try:
        if int(s1) != int(s0):
            return stale
        if not math.isclose(float(m1), float(m0), rel_tol=0.0,
                            abs_tol=1e-6):
            return stale
    except (TypeError, ValueError):
        return stale
    return None


# --------------------------------------------------------------------------- #
# which runs a patch can be compared with
# --------------------------------------------------------------------------- #
def _read_json(path) -> Dict[str, Any]:
    try:
        data = json.loads(Path(str(path)).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def _default_nested(workspace) -> bool:
    from . import _paths
    env = _read_json(_paths.state_json_path(workspace))
    return bool(env.get("default_run_nested", False))


def read_run_listing(workspace) -> Dict[str, Any]:
    """Every run of the workspace, read once for a whole dispatch:
    ``{"nested": the default run is nested, "runs": [{"run",
    "display_name", "archived"}]}``, the default run first.

    The state file once and each named run's manifest once; never a scan
    of a run's files (no last-activity time).  Never raises: an unreadable
    workspace lists the default run only.
    """
    from . import _paths
    env: Dict[str, Any] = {}
    runs: List[Dict[str, Any]] = []
    try:
        env = _read_json(_paths.state_json_path(workspace))
        state_runs = env.get("runs")
        default = (state_runs.get(_DEFAULT_RUN_ID)
                   if isinstance(state_runs, dict) else None)
        default = default if isinstance(default, dict) else {}
        runs.append({"run": _DEFAULT_RUN_ID,
                     "display_name": str(default.get("display_name")
                                         or "Default"),
                     "archived": False})
        from packages.epicycle_io.step3_run_manifest import (
            is_run_archived, list_runs_with_manifests,
        )
        for entry in list_runs_with_manifests(
                _paths.step3_runs_root(workspace)):
            slug = str(entry.get("run_id") or "")
            if not slug or slug == _DEFAULT_RUN_ID:
                continue
            m = entry.get("manifest")
            m = m if isinstance(m, dict) else {}
            runs.append({"run": slug,
                         "display_name": str(m.get("display_name") or slug),
                         "archived": bool(is_run_archived(m))})
    except Exception as exc:      # noqa: BLE001 -- a listing, never a crash
        print(f"[QC-PAIR] reading the runs failed: {exc!r}", flush=True)
    if not runs:
        runs = [{"run": _DEFAULT_RUN_ID, "display_name": "Default",
                 "archived": False}]
    return {"nested": bool(env.get("default_run_nested", False)),
            "runs": runs}


def run_display_name(listing, slug) -> str:
    """The name of run ``slug`` in a :func:`read_run_listing` result
    (archived runs too), or ``""``."""
    for r in (listing or {}).get("runs") or []:
        if r.get("run") == slug:
            return str(r.get("display_name") or "")
    return ""


def list_match_runs(workspace, patch_name, active_run_id, *,
                      listing=None) -> List[Dict]:
    """The other runs of this patch that have reconstruction labels.

    What the Reshape guide picker lists (archived runs skipped, the active
    run never offered, a nested default run resolved), kept only when the
    run's ``labels_3d.tif`` for this patch exists.  Each entry:
    ``{"run", "display_name", "path"}``.  ``listing`` is
    :func:`read_run_listing`'s result when the caller read it once for
    several patches.  Never raises.
    """
    out: List[Dict] = []
    try:
        if not workspace or not patch_name:
            return out
        from . import _paths
        from ._run_management import run_path_id_for

        if listing is None:
            listing = read_run_listing(workspace)
        nested = bool(listing.get("nested", False))
        excluded = str(active_run_id or _DEFAULT_RUN_ID)
        for r in listing.get("runs") or []:
            slug = str(r.get("run") or "")
            if not slug or slug == excluded or r.get("archived"):
                continue
            pid = run_path_id_for(slug, default_nested=nested)
            path = str(_paths.patch_labels_stem(
                workspace, patch_name, pid)) + ".tif"
            if not os.path.isfile(path):
                continue
            out.append({"run": slug,
                        "display_name": str(r.get("display_name") or slug),
                        "path": path})
    except Exception as exc:      # noqa: BLE001 -- a listing, never a crash
        print(f"[QC-PAIR] listing runs failed: {exc!r}", flush=True)
    return out


def resolve_match(workspace, patch_name, slug, *, active_run_id,
                    active_path_id, runs=None, listing=None,
                    link_floor=LINK_FLOOR, factor=SIZE_FACTOR):
    """``(bundle, note, provenance)`` for comparing ``patch_name`` with run
    ``slug``.

    No slug: ``(None, None, None)``.  A slug that does not list for this
    patch (no reconstruction labels, archived, the active run): no bundle,
    the log-only sentence as ``note``.  Otherwise the bundle (with both
    manifests' stack file names and the other run's labels shape) and the
    dispatch provenance (run, workspace-relative file, its mtime and size,
    the link floor).  ``runs`` is :func:`list_match_runs`' result and
    ``listing`` :func:`read_run_listing`'s when the caller has them.  Reads
    two small JSON files; never the volume.
    """
    if not slug:
        return None, None, None
    if listing is None:
        listing = read_run_listing(workspace)
    if runs is None:
        runs = list_match_runs(workspace, patch_name, active_run_id,
                                 listing=listing)
    entry = next((r for r in runs if r.get("run") == slug), None)
    if entry is None:
        name = run_display_name(listing, slug) or slug
        return None, T.NO_LABELS.format(other=name), None
    from . import _paths
    from ._run_management import run_path_id_for

    other_pid = run_path_id_for(
        slug, default_nested=bool(listing.get("nested", False)))
    other_m = _read_json(_paths.patch_run_manifest_path(
        workspace, patch_name, other_pid))
    active_m = _read_json(_paths.patch_run_manifest_path(
        workspace, patch_name, active_path_id))

    def _name(m):
        p = m.get("source_stack_path") or ""
        return Path(str(p)).name if p else ""

    bundle = make_match_bundle(
        path=entry["path"], run=slug,
        run_display_name=entry.get("display_name") or slug,
        link_floor=link_floor, factor=factor,
        active_stack_name=_name(active_m), other_stack_name=_name(other_m),
        other_labels_shape=other_m.get("labels_shape"))
    st = file_stat(entry["path"])
    prov = {
        "match_run": slug,
        "match_run_display_name": entry.get("display_name") or slug,
        "match_file": workspace_relative(entry["path"], workspace),
        "match_file_mtime": st["mtime"] if st else None,
        "match_file_size": st["size"] if st else None,
        "link_floor": float(link_floor),
    }
    return bundle, None, prov


__all__ = [
    "LINK_FLOOR", "SIZE_FACTOR", "MATCH_COLUMNS", "INT_COLUMNS",
    "FLOAT_COLUMNS", "OUTCOME_KEY", "ATTRS_KEY",
    "make_match_bundle", "pairing_outcome", "cross_run_pairing",
    "write_match_columns", "blank_match_row", "blank_match_columns",
    "active_label_rows", "has_match_columns", "values_predate_edits_on_disk",
    "file_stat", "read_run_listing", "run_display_name",
    "patch_file_fields", "workspace_relative", "match_file_path",
    "pairing_context_fields", "match_stale_reason", "list_match_runs",
    "resolve_match",
]
