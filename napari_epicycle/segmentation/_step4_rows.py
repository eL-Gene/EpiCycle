"""Step 4's stack rows, pure.

Qt-free (the running words are read from the shared table only for a
running row).  Two jobs:

* :func:`list_stack_sources` -- where stacks come from:
  Step 1's patch REGISTRY (never a folder scan), then every added folder's
  ``.tif`` / ``.tiff`` files.  De-duplicated by normalised absolute path; a
  second file with the same name or stem is listed with ``conflict=True``
  (D-g: the record key is the file name).
* :func:`build_row` -- one IMMUTABLE row record (a plain dict) that
  ``StackRowDelegate`` paints: status word / colour / icon / dots, the
  shape and spacing lines, chips, amber warnings, whether it can run.  The
  controller rebuilds every record on a refresh; nothing edits one in place.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from . import _step4_models as M
from . import _step4_text as T

STACK_SUFFIXES = (".tif", ".tiff")
HAS_3D = ("reconstructed", "qc_extracted", "finalized")
#: ③ lists these (today's ``qc_eligible``; legacy ``smoothed`` included).
MEASURE_LISTED = ("reconstructed", "qc_extracted", "smoothed", "finalized")

# The theme's status table, copied at import so this module stays Qt-free
# (theme.py imports qtpy).  test_step4_row_records pins the copy.
_STATUS_COLORS = {
    "unprocessed": "#777777", "predicted": "#5ba4cf",
    "reconstructed": "#d4a843", "qc_extracted": "#a78bca",
    "finalized": "#80deea", "failed": "#e07070",
}
_STATUS_ICONS = {
    "unprocessed": "mdi6.circle-outline", "predicted": "mdi6.check",
    "reconstructed": "mdi6.cube-outline",
    "qc_extracted": "mdi6.chart-scatter-plot",
    "finalized": "mdi6.diamond-stone", "failed": "mdi6.close-circle-outline",
}
_STATUS_DOTS = {
    "unprocessed": "○", "predicted": "●", "reconstructed": "●●",
    "qc_extracted": "●●", "finalized": "●●●", "failed": "",
}
_STATUS_WORDS = {
    "unprocessed": "unprocessed", "predicted": "predicted",
    "reconstructed": "reconstructed", "qc_extracted": "measured",
    "finalized": "finalized", "failed": "failed",
}
RUNNING_COLOR = "#777777"
RUNNING_ICON = "mdi6.timer-sand"


def norm_path(path) -> str:
    try:
        return os.path.normcase(os.path.abspath(str(path)))
    except (TypeError, ValueError):
        return str(path)


def display_status(stored: str) -> str:
    """The status a row shows: a stored ``segmented`` is a Reconstruct that
    never finished (failed); ``smoothed`` was never written (measured)."""
    if stored == "segmented":
        return "failed"
    if stored == "smoothed":
        return "qc_extracted"
    return stored if stored in _STATUS_WORDS else "unprocessed"


# ---------------------------------------------------------------------------
# Where stacks come from
# ---------------------------------------------------------------------------


def _step1_patch_files(workspace: str, step1_state) -> List[str]:
    if not workspace or not isinstance(step1_state, dict):
        return []
    out = []
    for row in step1_state.get("patches", []) or []:
        if not isinstance(row, dict):
            continue
        rel = row.get("relative_path")
        if not rel:
            continue
        full = Path(workspace) / str(rel)
        if full.suffix.lower() in STACK_SUFFIXES and full.is_file():
            out.append(str(full))
    return out


def _folder_files(folder: str) -> List[str]:
    try:
        entries = sorted(Path(folder).iterdir(), key=lambda p: p.name.lower())
    except OSError:
        return []
    return [str(p) for p in entries
            if p.suffix.lower() in STACK_SUFFIXES and p.is_file()]


def list_stack_sources(workspace: str, step1_state,
                       folders: Iterable[str]) -> List[Dict[str, Any]]:
    """Every stack Step 4 lists, in order: Step 1's patches first, then the
    added folders.  Each entry: ``{"name", "path", "origin", "conflict"}``.
    Never raises."""
    seen_paths = set()
    names = set()
    stems = set()
    out: List[Dict[str, Any]] = []

    def add(path: str, origin: str):
        key = norm_path(path)
        if key in seen_paths:
            return
        seen_paths.add(key)
        name = Path(path).name
        stem = Path(path).stem.casefold()
        conflict = name.casefold() in names or stem in stems
        if not conflict:
            names.add(name.casefold())
            stems.add(stem)
        out.append({"name": name, "path": str(path), "origin": origin,
                    "conflict": conflict})

    for path in _step1_patch_files(workspace, step1_state):
        add(path, "step1")
    for folder in folders or []:
        if folder:
            for path in _folder_files(folder):
                add(path, "folder")
    return out


# ---------------------------------------------------------------------------
# One row
# ---------------------------------------------------------------------------


def spacing_text(spacing: Optional[Dict[str, Any]]) -> str:
    """``ROW_SPACING_*`` for a record's spacing, or ``ROW_SPACING_NONE``."""
    if not isinstance(spacing, dict) or not spacing.get("zyx"):
        return T.ROW_SPACING_NONE
    try:
        z, y, x = (float(v) for v in spacing["zyx"])
    except (TypeError, ValueError):
        return T.ROW_SPACING_NONE
    tpl = {"step1": T.ROW_SPACING_STEP1, "file": T.ROW_SPACING_FILE,
           "user": T.ROW_SPACING_USER}.get(spacing.get("source"),
                                           T.ROW_SPACING_FILE)
    return tpl.format(z=f"{z:.2f}", y=f"{y:.2f}", x=f"{x:.2f}")


def _changed_since_predict(record, path) -> bool:
    fp = record.get("source_fingerprint")
    if not isinstance(fp, dict) or not path:
        return False
    now = M.file_fingerprint(path)
    if now is None:
        return False
    return (int(fp.get("size", -1)) != now["size"]
            or int(fp.get("mtime_ns", -1)) != now["mtime_ns"])


def build_row(entry: Dict[str, Any], record: Dict[str, Any], *,
              active_model: Optional[Dict[str, Any]] = None,
              needs: Optional[int] = None,
              ticked: bool = False,
              running: Optional[tuple] = None,
              cache_complete: bool = False) -> Dict[str, Any]:
    """The row record of one listed stack.

    ``needs`` is how many channels the active model needs (None: unknown,
    so no channel warning); ``running`` is ``(stage, view_text)`` while a
    job works on this stack; ``cache_complete`` says whether a complete 2D
    cache is on disk (found by the controller at refresh).
    """
    record = dict(record or {})
    name = entry.get("name") or ""
    path = entry.get("path") or record.get("source_stack_path") or ""
    stored = record.get("status") or "unprocessed"
    status = display_status(stored)
    axes = M.stack_axes(record)
    n_ch = M.stack_channels(record)
    zyx = M.zyx_shape(record.get("stack_shape") or [], axes)
    spacing = record.get("spacing") if isinstance(record.get("spacing"),
                                                   dict) else None
    has_3d = stored in HAS_3D or stored == "smoothed"
    # G1-4: after a Measure the row counts what Measure counted (border
    # cells removed), the same number as the stack card
    n_cells = record.get("measured_cells")
    if not isinstance(n_cells, int):
        n_cells = record.get("n_cells")

    # line 3 / wide line 2
    shape = ""
    if zyx is not None:
        shape = T.ROW_SHAPE.format(Z=zyx[0], Y=zyx[1], X=zyx[2],
                                   c=n_ch if n_ch is not None else "?")
        if has_3d and isinstance(n_cells, int):
            shape += T.ROW_SHAPE_CELLS.format(n=T.fmt_count(n_cells))
    sp_text = spacing_text(spacing)

    warns_wide: List[str] = []
    warns_narrow: List[str] = []
    can_run = True
    conflict = bool(entry.get("conflict"))
    if needs is not None and n_ch is not None and n_ch < needs:
        can_run = False
        w = T.ROW_WARN_CHANNELS.format(n=needs, k=n_ch)
        warns_wide.append(w)
        warns_narrow.append(w)
    if not axes:
        # N3: never run on a guessed axis order; the tick, the skipped
        # count, the gate and the done count all see this one flag
        can_run = False
        warns_wide.append(T.ROW_WARN_NO_AXES)
        warns_narrow.append(T.ROW_WARN_NO_AXES)
    if conflict:
        can_run = False
        warns_wide.append(T.ROW_WARN_SAME_NAME)
        warns_narrow.append(T.ROW_WARN_SAME_NAME)
    # (A3 fold; audit L7) no voxel size is said ONCE: by the amber voxel line
    # of the wide and the narrow row; Set spacing… is on the row's menu
    if _changed_since_predict(record, path):
        warns_wide.append(T.ROW_WARN_CHANGED_SINCE_PREDICT)
        warns_narrow.append(T.ROW_WARN_CHANGED_SINCE_PREDICT)
    failed_stage = record.get("failed_stage")
    if failed_stage and status != "failed":
        w = T.ROW_LAST_RUN_FAILED.format(
            stage=T.ROW_STAGE_WORDS.get(failed_stage, failed_stage))
        warns_wide.append(w)
        warns_narrow.append(w)

    # chips: what made the result, amber when not the active model
    chips: List[tuple] = []
    made = record.get("made_with")
    pred = record.get("predicted_with")
    if has_3d:
        if made:
            label = M.model_short_label(made)
            if active_model and not M.same_model(made, active_model):
                chips.append((T.ROW_NOT_ACTIVE_MODEL.format(model=label), True))
            else:
                chips.append((T.ROW_MADE_BY.format(model=label), False))
        else:
            chips.append((T.ROW_MADE_BY_UNKNOWN, False))
        if pred and made and not M.same_model(pred, made):
            chips.append(_chip_2d(pred, active_model))
    elif stored == "predicted" or (stored in ("failed", "segmented")
                                   and cache_complete):
        if pred:
            chips.append(_chip_2d(pred, active_model))
        else:
            chips.append((T.ROW_2D_BY_UNKNOWN, False))
    stale = bool(record.get("predicted_axes")) and bool(axes) and (
        str(record.get("predicted_axes")) != axes)
    if stale and (cache_complete or stored == "predicted"):
        chips.append((T.ROW_2D_STALE, True))

    run_stage, run_view = (running or (None, ""))
    if run_stage:
        # The running words are Step 3's, in the shared table;
        # imported here so the module stays importable without Qt.
        from napari_epicycle.ui_shell.theme import (
            ROW_RECONSTRUCTING, ROW_SEGMENTING,
        )
        word = (ROW_RECONSTRUCTING if run_stage == "reconstruct"
                else ROW_SEGMENTING)
        colour, icon, dots = RUNNING_COLOR, RUNNING_ICON, ""
        if run_view:
            shape += run_view
    else:
        word = _STATUS_WORDS[status]
        colour = _STATUS_COLORS[status]
        icon = _STATUS_ICONS[status]
        dots = _STATUS_DOTS[status]

    return {
        "name": name,
        "path": path,
        "origin": entry.get("origin") or record.get("origin") or "folder",
        "conflict": conflict,
        "stored_status": stored,
        "status": status,
        "word": word,
        "dots": dots,
        "colour": colour,
        "icon": icon,
        "running": bool(run_stage),
        "axes": axes,
        "n_channels": n_ch,
        "zyx": zyx,
        "shape_text": shape,
        "spacing": spacing,
        "spacing_text": sp_text,
        "in_voxels": spacing is None,
        "warns_wide": tuple(warns_wide),
        "warns_narrow": tuple(warns_narrow),
        "chips": tuple(chips),
        "can_run": can_run,
        "tickable": can_run,
        "ticked": bool(ticked) and can_run,
        "has_3d": has_3d,
        "has_result": has_3d and bool(record.get("labels_path")),
        "cache_complete": bool(cache_complete),
        "cache_stale": stale,
        "changed_since_predict": _changed_since_predict(record, path),
        "measure_listed": stored in MEASURE_LISTED,
        "n_cells": n_cells if isinstance(n_cells, int) else None,
        # A3: the narrow rows' voxel-size line (mono; amber "voxels" when
        # there is none), the labels the picture's overlay draws and the
        # saved features CSV the bucket counts read -- the last two only
        # for a stack that has a 3D result
        "vox_text": sp_text,
        "vox_warn": spacing is None,
        # ①'s wide row, Session's pattern: line 2 = axes + shape (+ cells),
        # line 3 = the voxel size (amber "voxels" when none); the spacing is
        # no longer part of line 2 and there is no inline button
        "line2": " · ".join(p for p in (axes, shape) if p),
        "line3": sp_text,
        "line3_warn": spacing is None,
        "labels_path": (record.get("labels_path") or "") if has_3d else "",
        "features_csv": (record.get("qc_features_csv") or "") if has_3d
        else "",
    }


def _chip_2d(pred, active):
    label = M.model_short_label(pred)
    if active and not M.same_model(pred, active):
        return (T.ROW_2D_NOT_ACTIVE.format(model=label), True)
    return (T.ROW_2D_BY.format(model=label), False)


def recon_ready(row: Dict[str, Any]) -> bool:
    """A ticked row Reconstruct can take: a complete, current cache."""
    return (bool(row.get("ticked")) and bool(row.get("cache_complete"))
            and not row.get("cache_stale")
            and not row.get("changed_since_predict"))


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def status_counts(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts = {k: 0 for k in _STATUS_WORDS}
    for r in rows:
        counts[r.get("status", "unprocessed")] = counts.get(
            r.get("status", "unprocessed"), 0) + 1
    return counts


def session_summary(rows: List[Dict[str, Any]]):
    """``(main_text, amber_text)`` of ①'s status line."""
    if not rows:
        return T.STACKS_NONE, ""
    n = len(rows)
    m = sum(1 for r in rows if r.get("in_voxels"))
    k = n - m
    main = T.STACKS_COUNT.format(n=T.fmt_count(n), k=T.fmt_count(k))
    amber = T.STACKS_IN_VOXELS.format(m=T.fmt_count(m)) if m else ""
    if not amber:
        main = main.rstrip(" ·")
    return main, amber


def ticked_line(rows: List[Dict[str, Any]], ticked_names: Iterable[str]):
    """``(dim_text, amber_text)`` under ②'s list: "{n} ticked" or
    "{n} ticked · {k} cannot run", then the voxels span."""
    names = set(ticked_names or ())
    ticked = [r for r in rows if r.get("name") in names]
    n = len(ticked)
    k = sum(1 for r in ticked if not r.get("can_run"))
    dim = (T.SEG_TICKED_SKIPPED.format(n=n, k=k) if k
           else T.SEG_TICKED_LINE.format(n=n))
    voxel = [r for r in ticked if r.get("can_run") and r.get("in_voxels")]
    amber = ""
    if len(voxel) == 1:
        amber = T.SEG_TICKED_VOXELS.format(stack=voxel[0]["name"])
    elif len(voxel) > 1:
        amber = T.SEG_TICKED_VOXELS_MANY.format(m=len(voxel))
    return dim, amber


# ---------------------------------------------------------------------------
# The Export window
# ---------------------------------------------------------------------------


LEGACY_SLACK_S = 5.0          # Q1: a file written by Finalize itself


def _finalize_matches(record: Dict[str, Any]):
    """``(labels_same, csv_same)`` against what Finalize saw, or ``None``
    when Finalize recorded nothing (a stack finalized before the revamp)."""
    fp = record.get("finalized_fingerprint")
    if not isinstance(fp, dict):
        return None
    now = M.finalized_fingerprint(record.get("labels_path") or "",
                                  record.get("qc_features_csv") or "")

    def _same(a, b):
        return list(a) == list(b) if a is not None and b is not None             else a is None and b is None
    return (_same(fp.get("labels"), now.get("labels")),
            _same(fp.get("csv"), now.get("csv")))


def _legacy_untouched(record: Dict[str, Any]) -> bool:
    """Q1: a pre-revamp finalized stack goes only when neither file is
    later than its ``finalized_at`` (plus a few seconds); with no
    ``finalized_at`` there is nothing to compare, so it does not go."""
    from datetime import datetime
    try:
        at = datetime.fromisoformat(str(record.get("finalized_at")))
        at_ts = at.timestamp()
    except (TypeError, ValueError):
        return False
    for key in ("labels_path", "qc_features_csv"):
        p = record.get(key) or ""
        try:
            if os.path.getmtime(p) > at_ts + LEGACY_SLACK_S:
                return False
        except OSError:
            return False
    return True


def export_row(name: str, record: Dict[str, Any], *,
               unsaved: bool = False) -> Dict[str, Any]:
    """One Export-window row: who can go, and why not.

    A finalized stack goes when its labels and features CSV still match
    the ``finalized_fingerprint`` of its Finalize and ``spacing`` equals
    ``measured_spacing`` (F20, I-19); otherwise it is off with
    ``EXPORT_ROW_WHY_EDITED`` (``EXPORT_ROW_WHY_REFINALIZE`` when only the
    CSV changed: marks saved after Finalize).  One amber note on a row
    that goes: its unsaved edits (G1-2), finalized before the files were
    recorded (Q1), units unknown (Q2), or in voxels."""
    stored = record.get("status") or "unprocessed"
    status = display_status(stored)
    csv = record.get("qc_features_csv") or ""
    has_csv = bool(csv) and Path(csv).is_file()
    why, can_go, legacy = "", False, False
    if stored == "finalized":
        match = _finalize_matches(record)
        spacing_ok = ("measured_spacing" not in record) or M.same_spacing(
            record.get("spacing"), record.get("measured_spacing"))
        if not has_csv:
            why = T.EXPORT_ROW_WHY_MEASURE
        elif not spacing_ok:
            why = T.EXPORT_ROW_WHY_EDITED
        elif match is None:
            legacy = True
            if _legacy_untouched(record):
                can_go = True
            else:
                why = T.EXPORT_ROW_WHY_EDITED
        elif not match[0]:
            why = T.EXPORT_ROW_WHY_EDITED
        elif not match[1]:
            why = T.EXPORT_ROW_WHY_REFINALIZE
        else:
            can_go = True
        if can_go:
            if unsaved:
                why = T.EXPORT_ROW_UNSAVED
            elif legacy:
                why = T.EXPORT_ROW_LEGACY
            elif "measured_spacing" not in record:
                why = T.EXPORT_ROW_UNITS_UNKNOWN
            elif not record.get("measured_spacing"):
                why = T.EXPORT_ROW_UNITS_VOXELS
    elif stored in ("qc_extracted", "smoothed") and has_csv:
        why = T.EXPORT_ROW_WHY_FINALIZE
    else:
        why = T.EXPORT_ROW_WHY_MEASURE
    return {
        "name": name, "status": status, "word": _STATUS_WORDS[status],
        "dots": _STATUS_DOTS[status], "colour": _STATUS_COLORS[status],
        "icon": _STATUS_ICONS[status], "can_go": can_go, "why": why,
        "legacy_finalized": legacy and can_go,
    }


def export_rows(records: Dict[str, Dict[str, Any]], *,
                unsaved: Optional[str] = None) -> List[Dict[str, Any]]:
    """The Export window's list: every reconstructed stack (③'s rule,
    ``MEASURE_LISTED`` with a labels file on disk), those that can go
    first, each group by name.  *unsaved* names the loaded stack while it
    has unsaved edits."""
    rows = []
    for name, rec in records.items():
        if (rec.get("status") or "") not in MEASURE_LISTED:
            continue
        labels = rec.get("labels_path") or ""
        if not labels or not Path(labels).is_file():
            continue
        rows.append(export_row(name, rec, unsaved=(name == unsaved)))
    rows.sort(key=lambda r: (not r["can_go"], r["name"].casefold()))
    return rows


def merge_stack_csvs(csvs: Dict[str, str]):
    """B1-8: the features CSVs of the stacks named, each frame tagged with
    its RECORD name in ``stack_name`` (never a CSV stem)."""
    import pandas as pd
    from napari_epicycle.fusion_reconstruction.qc_features import (
        merge_patch_csvs,
    )
    frames = []
    for name, path in csvs.items():
        df = merge_patch_csvs([path])
        df = df.drop(columns=["patch_name"], errors="ignore")
        df["stack_name"] = name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
META_COLUMNS = ("meta_model", "meta_units", "meta_spacing_z",
                "meta_spacing_y", "meta_spacing_x")


def add_meta_columns(df, records: Dict[str, Dict[str, Any]]):
    """F14: per row, the stack's model (``made_with``'s label) and the
    units the numbers were MEASURED in, from ``measured_spacing`` (never
    ``spacing``, grill 1 G1-2); empty spacing columns in voxels."""
    if df is None or "stack_name" not in getattr(df, "columns", ()):
        return df
    model, units, sz, sy, sx = {}, {}, {}, {}, {}
    for name, rec in records.items():
        model[name] = M.model_short_label(rec.get("made_with")) or ""
        ms = rec.get("measured_spacing")
        zyx = ms.get("zyx") if isinstance(ms, dict) else None
        try:
            vals = [float(v) for v in zyx] if zyx else None
        except (TypeError, ValueError):
            vals = None
        # Q2: measured before the revamp recorded units -> "unknown"
        units[name] = ("unknown" if "measured_spacing" not in rec
                       else "um" if vals else "voxels")
        sz[name], sy[name], sx[name] = (vals if vals else (None, None, None))
    keys = df["stack_name"].astype(str)
    df = df.copy()
    df["meta_model"] = keys.map(model).fillna("")
    df["meta_units"] = keys.map(units).fillna("unknown")
    df["meta_spacing_z"] = keys.map(sz)
    df["meta_spacing_y"] = keys.map(sy)
    df["meta_spacing_x"] = keys.map(sx)
    return df
