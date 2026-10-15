"""Step 4's pure record helpers.

Qt-free and napari-free: the spacing resolver (re-exported from
``data_preparation/_voxel_spacing`` since R2b), the model snapshot, file
fingerprints and the prediction-cache file names.  Everything here is a
plain function the controller calls and a test can drive without a viewer.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

VIEWS = ("xy", "xz", "yz")
CACHE_KINDS = ("flows", "probs", "masks")

# ``SPACING_SOURCES`` lives beside the resolver in Step 1's package since R7a: Step 3 validates against it.
from ..data_preparation._voxel_spacing import (  # noqa: E402,F401
    SPACING_SOURCES,
)


# ---------------------------------------------------------------------------
# Step 1's state, read-only
# ---------------------------------------------------------------------------


# The reader lives beside the resolver in Step 1's package since R2b part 2: Step 3 needs it and may not import
# a Step 4 module.  Re-exported here under the same name.
from ..data_preparation._voxel_spacing import (  # noqa: E402,F401
    read_step1_state,
)


# ---------------------------------------------------------------------------
# Spacing
# ---------------------------------------------------------------------------


# The resolver and its helpers live in Step 1's package since R2b: Steps 3 and 4 share it.  Re-exported
# here under the same names, so every Step 4 caller and test is unchanged
# (``model_snapshot`` below still calls ``_rel_to``).  A test that stubs the
# file read patches ``data_preparation._voxel_spacing._read_voxel_spacing``.
from ..data_preparation._voxel_spacing import (  # noqa: E402,F401
    _positive,
    _read_voxel_spacing,
    _real,
    _rel_to,
    resolve_stack_spacing,
    same_spacing,
)


# ---------------------------------------------------------------------------
# The model snapshot
# ---------------------------------------------------------------------------


def _label(path_or_name: str) -> str:
    try:
        return Path(str(path_or_name)).name or str(path_or_name)
    except Exception:      # noqa: BLE001
        return str(path_or_name)


def model_snapshot(lateral: str, axial: str, *, kind: str,
                   workspace: str = "",
                   lateral_label: Optional[str] = None,
                   axial_label: Optional[str] = None,
                   run_id: Optional[str] = None,
                   bundle: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """A model snapshot, taken at LAUNCH and carried to the finish.

    ``lateral_rel`` / ``axial_rel`` are workspace-relative when the files
    sit under the workspace, so a copied workspace compares equal.  Chips
    compare paths, never the label.  ``None`` without a lateral model.
    """
    if not lateral:
        return None
    axial = axial or lateral
    rel_l = _rel_to(lateral, workspace) if (workspace and kind != "pretrained") else None
    rel_a = _rel_to(axial, workspace) if (workspace and kind != "pretrained") else None
    return {
        "lateral": str(lateral), "axial": str(axial),
        "lateral_label": lateral_label or _label(lateral),
        "axial_label": axial_label or _label(axial),
        "kind": kind, "run_id": run_id, "bundle": bundle,
        "lateral_rel": rel_l, "axial_rel": rel_a,
    }


def same_model(a: Optional[Dict[str, Any]],
               b: Optional[Dict[str, Any]]) -> bool:
    """Two snapshots name the same files: relative paths when both have
    them, normalised absolute paths otherwise, never the labels."""
    if not a or not b:
        return False

    def key(s, side):
        rel = s.get(f"{side}_rel")
        if rel:
            return ("rel", os.path.normcase(os.path.normpath(rel)))
        return ("abs", os.path.normcase(os.path.normpath(str(s.get(side, "")))))

    return all(key(a, side) == key(b, side) for side in ("lateral", "axial"))


# ---------------------------------------------------------------------------
# Fingerprints (G1-20, F20)
# ---------------------------------------------------------------------------


# ``file_fingerprint`` lives in ``ui_shell/_model_files`` since R6a: Step 3's model provenance reads it
# too and may not import a Step 4 module.  Re-exported here, the same object.
from ..ui_shell._model_files import file_fingerprint  # noqa: E402,F401


def finalized_fingerprint(labels_path: str, csv_path: str) -> Dict[str, Any]:
    """``{"labels": [size, mtime_ns], "csv": [size, mtime_ns]}`` (U1-3)."""
    out: Dict[str, Any] = {}
    for key, p in (("labels", labels_path), ("csv", csv_path)):
        fp = file_fingerprint(p) if p else None
        out[key] = [fp["size"], fp["mtime_ns"]] if fp else None
    return out


# ---------------------------------------------------------------------------
# Prediction-cache file names (F7, F24)
# ---------------------------------------------------------------------------


def cache_array_path(cache_dir, name: str, view: str, kind: str) -> Path:
    """New caches are keyed by the FILE NAME (``a.tif`` and ``a.tiff`` differ)."""
    return Path(cache_dir) / f"{name}_{view}_{kind}.npy"


def legacy_cache_array_path(cache_dir, name: str, view: str, kind: str) -> Path:
    """Before the revamp caches were keyed by the stem."""
    return Path(cache_dir) / f"{Path(name).stem}_{view}_{kind}.npy"


def cache_meta_path(cache_dir, name: str) -> Path:
    return Path(cache_dir) / f"{name}_cache_meta.json"


def read_cache_meta(cache_dir, name: str) -> Optional[Dict[str, Any]]:
    p = cache_meta_path(cache_dir, name)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def find_prediction_cache(cache_dir, name: str) -> Dict[str, Any]:
    """Which cache a stack has, for Reconstruct.

    Returns ``{"kind", "files", "meta", "complete"}``:

    * ``kind="name"`` -- name-keyed arrays exist.  ``complete`` only when
      all three views have flows AND probs AND the meta is present (the meta
      is written LAST, so a half-written cache has none).
    * ``kind="legacy"`` -- no name-keyed array, but a stem-keyed cache with
      all three views: accepted, with ``meta=None`` (made_with unknown).
    * ``kind=None`` -- nothing usable.

    ``files`` maps each view to ``(flows, probs, masks_or_None)`` paths.
    """
    cache_dir = Path(cache_dir)

    def views_for(path_fn):
        files = {}
        for v in VIEWS:
            fp = path_fn(cache_dir, name, v, "flows")
            pp = path_fn(cache_dir, name, v, "probs")
            mp = path_fn(cache_dir, name, v, "masks")
            if fp.exists() and pp.exists():
                files[v] = (str(fp), str(pp), str(mp) if mp.exists() else None)
        return files

    named_any = any(
        cache_array_path(cache_dir, name, v, k).exists()
        for v in VIEWS for k in CACHE_KINDS
    ) or cache_meta_path(cache_dir, name).exists()
    if named_any:
        files = views_for(cache_array_path)
        meta = read_cache_meta(cache_dir, name)
        return {"kind": "name", "files": files, "meta": meta,
                "complete": len(files) == 3 and meta is not None}
    files = views_for(legacy_cache_array_path)
    if len(files) == 3:
        return {"kind": "legacy", "files": files, "meta": None,
                "complete": True}
    return {"kind": None, "files": files, "meta": None, "complete": False}


# ---------------------------------------------------------------------------
# Reconcile the state with the disk (live driver, 2026-09-29)
# ---------------------------------------------------------------------------

#: What a 3D result puts on a record (dropped when its labels are gone).
RESULT_KEYS = ("labels_path", "prob_path", "original_labels_path",
               "made_with", "n_cells")


def _missing(path) -> bool:
    return bool(path) and not Path(str(path)).exists()


def prediction_slice_count(shape, axes: str) -> Optional[int]:
    """The slices a Predict segments: Z (XY view) + Y (XZ) + X (YZ).

    *shape* is the stack's series shape and *axes* its axis letters; with
    no usable axes the last three dimensions are taken as Z, Y, X.
    ``None`` when the shape has fewer than three dimensions."""
    try:
        dims = [int(d) for d in shape]
    except (TypeError, ValueError):
        return None
    axes = (axes or "").upper()
    if len(axes) == len(dims) and all(a in axes for a in "ZYX"):
        return sum(dims[axes.index(a)] for a in "ZYX")
    if len(dims) < 3:
        return None
    return sum(dims[-3:])


def reconcile_records(stacks: Dict[str, Dict[str, Any]], *,
                      measure_keys, cache_dir, manifests) -> Dict[str, str]:
    """Bring each record in line with the files on disk, IN PLACE.

    Never deletes a file.  Returns ``{stack: what changed}``.

    (a) ``labels_path`` gone -> the result and measure keys are dropped and
        the status re-derived (``predicted`` with a complete cache, else
        ``unprocessed``; a ``failed`` record stays ``failed``);
    (b) labels there but ``qc_features_csv`` / ``border_cells_path`` /
        ``smoothed_cache_path`` gone -> the measure keys are dropped and a
        measured / finalized stack goes back to ``reconstructed``;
    (c) a run manifest under ``artifacts/step4/runs/*`` whose stack has no
        live ``labels_path`` is adopted (the first in *manifests* wins, so
        pass them newest first).

    *manifests* is a list of manifest dicts (each with ``labels_path`` and
    ``source_stack_path``).
    """
    changes: Dict[str, str] = {}
    for name, rec in stacks.items():
        if not isinstance(rec, dict):
            continue
        if _missing(rec.get("labels_path")):
            gone = rec.get("labels_path")
            for key in tuple(RESULT_KEYS) + tuple(measure_keys):
                rec.pop(key, None)
            if rec.get("status") != "failed":
                complete = find_prediction_cache(cache_dir, name)["complete"] \
                    if cache_dir else False
                rec["status"] = "predicted" if complete else "unprocessed"
            changes[name] = f"labels gone ({gone}); now {rec['status']}"
            continue
        measure_gone = [k for k in ("qc_features_csv", "border_cells_path",
                                    "smoothed_cache_path")
                        if _missing(rec.get(k))]
        if rec.get("labels_path") and measure_gone:
            for key in measure_keys:
                rec.pop(key, None)
            if rec.get("status") in ("qc_extracted", "finalized", "smoothed"):
                rec["status"] = "reconstructed"
            changes[name] = (f"{', '.join(measure_gone)} gone; "
                             f"now {rec.get('status')}")
    for payload in manifests or []:
        if not isinstance(payload, dict):
            continue
        source = payload.get("source_stack_path") or ""
        labels = payload.get("labels_path") or ""
        name = Path(source).name if source else ""
        if not name or not labels or not Path(labels).is_file():
            continue
        rec = stacks.get(name)
        live = isinstance(rec, dict) and rec.get("labels_path") \
            and not _missing(rec.get("labels_path"))
        if live:
            continue
        rec = stacks.setdefault(name, {"status": "unprocessed"})
        rec["status"] = "reconstructed"
        rec["labels_path"] = labels
        rec["source_stack_path"] = rec.get("source_stack_path") or source
        prob = Path(labels).parent / f"{Path(source).stem}_probabilities.npy"
        if prob.exists():
            rec["prob_path"] = str(prob)
        if payload.get("num_cells_final") is not None:
            rec["n_cells"] = payload.get("num_cells_final")
        rec.pop("failed_stage", None)
        rec.pop("failed_at", None)
        changes[name] = (changes.get(name, "") + "; " if name in changes
                         else "") + f"re-adopted {Path(labels).parent.name}"
    return changes


# ---------------------------------------------------------------------------
# B1a: the model card
# ---------------------------------------------------------------------------

#: Step 3's indicator colours (S3c ``_refresh_model_status``).
MODEL_COLOR_NONE = "#888"
MODEL_COLOR_FILE = "#4caf50"
MODEL_COLOR_PRETRAINED = "#42a5f5"
MODEL_COLOR_PAIR = "#d4a843"


def label_for(path_text: str) -> str:
    """Step 3's ``_label_for`` (``_refresh_model_status``), copied verbatim
    in behaviour: a bare name is a Cellpose built-in; a path shows its file
    name, or the bundle folder when the file name is generic.  A parity test
    pins it against Step 3's own nested function."""
    if not path_text:
        return "-"
    if "/" in path_text or "\\" in path_text:
        try:
            p = Path(path_text)
            name = p.name
            if name.lower() in ("lateral", "axial", "lateral_model",
                                "axial_model", "model"):
                parent_name = p.parent.name
                if (parent_name
                        and parent_name.lower() not in ("refinement", "models")
                        and not parent_name.lower().startswith("round_")):
                    return parent_name
            return name
        except Exception:      # noqa: BLE001
            return path_text
    return f"Pretrained: {path_text}"


def _side_label(snapshot: Dict[str, Any], side: str) -> str:
    """One side's label: Step 3's rule, with the run named for a refined
    model ("{bundle} · {run}", grill 1 G1-12)."""
    path = str(snapshot.get(side) or "")
    base = label_for(path)
    if snapshot.get("kind") == "refined":
        run = snapshot.get("run_id") or "default"
        return f"{base} · {run}"
    stored = snapshot.get(f"{side}_label")
    if snapshot.get("kind") == "step2_run" and stored:
        return str(stored)
    return base


def model_indicator(snapshot: Optional[Dict[str, Any]]):
    """``(text, colour)`` for the model card's ``Active:`` line.

    No model: ``MODEL_NONE`` grey.  The same model both ways: one label,
    green for a file, blue for a built-in.  Two different models:
    ``MODEL_ACTIVE_PAIR`` in amber.
    """
    from napari_epicycle.ui_shell.theme import MODEL_ACTIVE_PAIR, MODEL_NONE
    if not snapshot or not (snapshot.get("lateral") or snapshot.get("axial")):
        return MODEL_NONE, MODEL_COLOR_NONE
    lat = _side_label(snapshot, "lateral")
    ax = _side_label(snapshot, "axial")
    if lat == ax:
        path = str(snapshot.get("lateral") or "")
        colour = (MODEL_COLOR_FILE if ("/" in path or "\\" in path)
                  else MODEL_COLOR_PRETRAINED)
        return lat, colour
    return MODEL_ACTIVE_PAIR.format(lateral=lat, axial=ax), MODEL_COLOR_PAIR


def model_short_label(snapshot: Optional[Dict[str, Any]]) -> str:
    """The ``{model}`` of a chip or a log line: the lateral label, or both
    joined when the two sides differ."""
    if not snapshot:
        return ""
    lat = _side_label(snapshot, "lateral")
    ax = _side_label(snapshot, "axial")
    return lat if lat == ax else f"{lat} / {ax}"


def model_files_missing(snapshot: Optional[Dict[str, Any]]) -> bool:
    """A model given by file whose files are not there (G1-12)."""
    if not snapshot or snapshot.get("kind") == "pretrained":
        return False
    for side in ("lateral", "axial"):
        p = snapshot.get(side)
        if not p:
            continue
        if "/" not in str(p) and "\\" not in str(p):
            continue          # a bare built-in name on one side
        if not Path(str(p)).exists():
            return True
    return False


# ``_training_config_files`` and ``read_trained_channels`` live in
# ``ui_shell/_model_files`` since R6a:
# Step 3's model provenance records a model's trained channels too.
# Re-exported here, the same objects.
from ..ui_shell._model_files import (  # noqa: E402,F401
    _training_config_files,
    read_trained_channels,
)


def trained_channel_indices(codes: Optional[list]):
    """Cellpose codes -> ``(membrane_index, nuclei_index_or_None)``.

    Code 0 is "gray" / "none"; code k is array channel k - 1."""
    if not codes:
        return None
    c1, c2 = int(codes[0]), int(codes[1])
    mem = c1 - 1 if c1 > 0 else 0
    nuc = c2 - 1 if c2 > 0 else None
    return mem, nuc


def trained_channels_text(codes: Optional[list]) -> str:
    """``SEG_MODEL_TRAINED_ON`` filled, or ``SEG_MODEL_TRAINED_UNKNOWN``."""
    from . import _step4_text as T
    idx = trained_channel_indices(codes)
    if idx is None:
        return T.SEG_MODEL_TRAINED_UNKNOWN
    mem, nuc = idx
    parts = [T.SEG_MODEL_CHANNEL.format(name=T.SEG_MODEL_ROLE_MEMBRANE,
                                        index=mem)]
    if nuc is not None:
        parts.append(T.SEG_MODEL_CHANNEL.format(
            name=T.SEG_MODEL_ROLE_NUCLEI, index=nuc))
    return T.SEG_MODEL_TRAINED_ON.format(n=len(parts), names=", ".join(parts))


def channels_needed(codes: Optional[list]) -> Optional[int]:
    """How many channels a stack needs for this model (the highest trained
    index + 1), or ``None`` when the model's channels are unknown."""
    idx = trained_channel_indices(codes)
    if idx is None:
        return None
    mem, nuc = idx
    return max(mem, nuc if nuc is not None else 0) + 1


def step4_model_rows(entries) -> list:
    """``list_cross_run_refined_models`` entries (one per bundle AND bucket)
    merged per (run_id, bundle) into the rows ``RefinementRegistryDialog``
    draws and Step 4's Use reads."""
    merged: Dict[tuple, Dict[str, Any]] = {}
    for e in entries or []:
        if not isinstance(e, dict):
            continue
        key = (e.get("run_id"), e.get("bundle_name"))
        manifest = e.get("manifest") if isinstance(e.get("manifest"),
                                                   dict) else {}
        row = merged.get(key)
        if row is None:
            wp = str(e.get("weights_path") or "")
            row = merged[key] = {
                "name": e.get("bundle_name") or "?",
                "run_id": e.get("run_id"),
                "mode": "?",
                "lateral_pairs": None,
                "axial_pairs": None,
                "created": (manifest.get("created_at")
                            or e.get("mtime_iso") or ""),
                "init_summary": "—",
                "lateral_weights": None,
                "axial_weights": None,
                "bundle_dir": str(Path(wp).parent) if wp else "",
            }
        bucket = e.get("bucket")
        if bucket in ("lateral", "axial"):
            row[f"{bucket}_weights"] = e.get("weights_path")
            section = manifest.get(bucket) if isinstance(
                manifest.get(bucket), dict) else {}
            row[f"{bucket}_pairs"] = section.get("n_train_pairs")
        has_l, has_a = bool(row["lateral_weights"]), bool(row["axial_weights"])
        row["mode"] = ("Both" if has_l and has_a else
                       "Lateral" if has_l else "Axial" if has_a else "?")
    rows = list(merged.values())
    rows.sort(key=lambda r: str(r.get("created") or ""), reverse=True)
    return rows


def snapshot_from_row(row: Dict[str, Any],
                      current: Optional[Dict[str, Any]],
                      workspace: str = "") -> Optional[Dict[str, Any]]:
    """The model snapshot a picker row makes active, dispatching on
    ``row["_kind"]`` as Step 3's ``_registry_action_use`` does.  A Step 2
    row sets its ONE bucket; the other keeps the current model's side."""
    kind = row.get("_kind", "refined")
    cur = current or {}
    if kind == "pretrained":
        name = row.get("name") or ""
        return model_snapshot(name, name, kind="pretrained",
                              workspace=workspace) if name else None
    if kind == "step2_run":
        bucket = (row.get("bucket") or "").lower()
        weights = str(row.get("weights_path") or "")
        if bucket not in ("lateral", "axial") or not weights:
            return None
        run_name = (row.get("run_display_name") or row.get("run_id") or "?")
        label = (row.get("human_label")
                 or f"{run_name} · {row.get('round_id') or '?'}")
        lat = weights if bucket == "lateral" else str(cur.get("lateral") or "")
        ax = weights if bucket == "axial" else str(cur.get("axial") or "")
        lat_label = label if bucket == "lateral" else cur.get("lateral_label")
        ax_label = label if bucket == "axial" else cur.get("axial_label")
        if not lat:
            lat, lat_label = ax, ax_label
        if not ax:
            ax, ax_label = lat, lat_label
        return model_snapshot(lat, ax, kind="step2_run", workspace=workspace,
                              lateral_label=lat_label, axial_label=ax_label)
    lat = str(row.get("lateral_weights") or "")
    ax = str(row.get("axial_weights") or "")
    if kind == "external":
        lat = lat or str(cur.get("lateral") or "")
        ax = ax or str(cur.get("axial") or "")
        if not lat:
            lat = ax
        return model_snapshot(lat, ax or lat, kind="external",
                              workspace=workspace) if lat else None
    # refined
    if not lat and not ax:
        return None
    return model_snapshot(lat or ax, ax or lat, kind="refined",
                          workspace=workspace, run_id=row.get("run_id"),
                          bundle=row.get("name"))


# ---------------------------------------------------------------------------
# B1a: axes per stack
# ---------------------------------------------------------------------------


def _detect_axes(shape) -> str:
    try:
        from napari_epicycle.fusion_reconstruction.qc_features import (
            detect_axes_order,
        )
        return detect_axes_order(tuple(shape))
    except Exception:      # noqa: BLE001
        if len(shape) == 3:
            return "ZYX"
        return "CZYX" if shape[0] <= shape[1] else "ZCYX"


def resolve_stack_axes(path) -> Optional[Dict[str, Any]]:
    """``{"axes", "shape", "n_channels"}`` of a TIFF stack from its header
    only (``resolve_patch_axes``' first and third signals: the file's own
    ZCYX / CZYX axes, else the size heuristic).  ``None`` when the file
    cannot be opened.  Never raises."""
    try:
        import tifffile
        with tifffile.TiffFile(str(path)) as tf:
            series = tf.series[0]
            shape = tuple(int(s) for s in series.shape)
            file_axes = (getattr(series, "axes", "") or "").upper()
            dtype = str(getattr(series, "dtype", "") or "")
    except Exception:      # noqa: BLE001
        return None
    base = {"shape": list(shape), "dtype": dtype}
    if len(shape) == 3:
        return dict(base, axes="ZYX", n_channels=1, axes_how="metadata"
                    if file_axes == "ZYX" else "heuristic")
    if len(shape) != 4:
        return dict(base, axes="", n_channels=1, axes_how="fallback")
    if file_axes in ("ZCYX", "CZYX"):
        axes, how = file_axes, "metadata"
    else:
        axes, how = _detect_axes(shape), "heuristic"
    c = shape[axes.index("C")] if "C" in axes else 1
    return dict(base, axes=axes, n_channels=int(c), axes_how=how)


def zyx_shape(shape, axes: str):
    """``(Z, Y, X)`` of a stored stack shape under its axes, or None."""
    try:
        shape = [int(s) for s in shape]
    except (TypeError, ValueError):
        return None
    if len(shape) == 3:
        return tuple(shape)
    if len(shape) == 4 and axes in ("ZCYX", "CZYX"):
        return tuple(s for s, a in zip(shape, axes) if a != "C")
    return None


def stack_axes(record: Dict[str, Any]) -> str:
    """The axes a stack runs with: the user's override, else the resolved."""
    return str(record.get("axes_override") or record.get("axes") or "")


def stack_channels(record: Dict[str, Any]) -> Optional[int]:
    """The stack's channel count under the axes it runs with."""
    axes = stack_axes(record)
    shape = record.get("stack_shape")
    if shape and axes:
        try:
            if len(shape) == 3:
                return 1
            if "C" in axes and len(shape) == 4:
                return int(shape[axes.index("C")])
        except (TypeError, ValueError, IndexError):
            pass
    n = record.get("n_channels")
    return int(n) if isinstance(n, int) else None


# ---------------------------------------------------------------------------
# B1a: the cell width
# ---------------------------------------------------------------------------


def xy_um(spacing: Optional[Dict[str, Any]]) -> Optional[float]:
    """The mean Y/X voxel size of a spacing, or None."""
    zyx = spacing.get("zyx") if isinstance(spacing, dict) else None
    try:
        v = (float(zyx[1]) + float(zyx[2])) / 2.0
    except (TypeError, ValueError, IndexError):
        return None
    return v if v > 0 else None


def cell_width_px(width_um: float, spacing: Optional[Dict[str, Any]]):
    """The cell width in pixels for a stack's Y/X spacing, or None when the
    width is Auto (0) or the stack has no spacing."""
    if not width_um or width_um <= 0:
        return None
    um = xy_um(spacing)
    if um is None:
        return None
    return max(1, int(round(float(width_um) / um)))


# ---------------------------------------------------------------------------
# B1a: Start from
# ---------------------------------------------------------------------------

# Moved to ui_shell: the
# Start-from presets both steps share, re-exported here under the same
# names.  ``newest_step3_manifest`` below stays Step 4's.
from ..ui_shell._recon_presets import (  # noqa: E402,F401
    STEP4_PRESET_EPICYCLE,
    USEG3D_WIDGET_KEYS,
    _NONE_AS_MINUS_ONE,
    _useg3d_params,
    _key_present,
    _lookup,
    _preset_from_groups,
    useg3d_preset,
    _STEP3_TOP_KEYS,
    step3_recon_preset,
)


def newest_step3_manifest(workspace: str) -> Optional[Dict[str, Any]]:
    """``{"run", "recon_config", "channels"}`` from the newest Step 3
    reconstruction ``manifest.json`` of the ACTIVE Step 3 run, or None.
    ``channels`` are Cellpose codes (the manifest's ``seg_config``);
    Step 3 records no model, so the model card is never touched (v2.6)."""
    if not workspace:
        return None
    try:
        from napari_epicycle.fusion_reconstruction import _paths as S3P
        from napari_epicycle.fusion_reconstruction._run_management import (
            list_run_summaries,
        )
        summaries = list_run_summaries(workspace)
    except Exception:      # noqa: BLE001
        return None
    active = next((s for s in summaries if s.get("is_active")), None)
    if active is None:
        return None
    run_id = str(active.get("run_id") or "default")
    display = str(active.get("display_name") or run_id)
    try:
        root = S3P.reconstructions_dir(
            workspace, None if run_id == "default" else run_id)
        candidates = list(root.glob("*/manifest.json"))
    except Exception:      # noqa: BLE001
        return None
    best = None
    for path in candidates:
        try:
            mtime = path.stat().st_mtime
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        rc = data.get("recon_config") if isinstance(data, dict) else None
        if not isinstance(rc, dict):
            continue
        if best is None or mtime > best[0]:
            best = (mtime, rc, data.get("seg_config"))
    if best is None:
        return None
    seg = best[2] if isinstance(best[2], dict) else {}
    return {"run": display, "recon_config": best[1],
            "channels": seg.get("channels")}


# ---------------------------------------------------------------------------
# The run snapshot (v2.6)
# ---------------------------------------------------------------------------

#: A model file below this size also gets a sha1.
SHA1_MAX_BYTES = 1 << 30


def model_fingerprints(snapshot: Optional[Dict[str, Any]],
                       *, with_sha1: bool = False) -> Dict[str, Any]:
    """``{"lateral": {...}, "axial": {...}}``: size and ``mtime_ns`` of each
    model file (a built-in name has none), plus a sha1 below 1 GB when
    *with_sha1*.  Never raises."""
    out: Dict[str, Any] = {}
    for side in ("lateral", "axial"):
        path = str((snapshot or {}).get(side) or "")
        fp = file_fingerprint(path) if ("/" in path or "\\" in path) else None
        if fp is not None and with_sha1 and fp["size"] < SHA1_MAX_BYTES:
            fp = dict(fp, sha1=_sha1(path))
        out[side] = fp
    return out


def _sha1(path: str) -> Optional[str]:
    import hashlib
    h = hashlib.sha1()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def effective_rescale(rescale) -> Optional[float]:
    """The resize factor Cellpose really gets (F25): ``None`` = the model's
    own trained cell size; ``1.0`` = no resize (today's tick ON)."""
    if rescale is None or rescale is False:
        return None
    if rescale is True:
        return 1.0
    try:
        return float(rescale)
    except (TypeError, ValueError):
        return None


#: How Cellpose 3.1.1.2 decides its resize (models.py:487-490, N2): a
#: cell width (diameter > 0) always wins and the tick is ignored.
RESCALE_FROM_WIDTH = "from cell width"
RESCALE_NONE = "none"
RESCALE_TRAINED = "model's trained size"

#: diam_mean of Cellpose's built-in models (cellpose.models); their
#: diam_labels equals diam_mean.
_BUILTIN_DIAM_MEAN = {"nuclei": 17.0}
_BUILTIN_DEFAULT_DIAM = 30.0


def rescale_rule(rescale, diameter) -> str:
    """Which of the three rules Cellpose applies (N2)."""
    try:
        if diameter is not None and float(diameter) > 0:
            return RESCALE_FROM_WIDTH
    except (TypeError, ValueError):
        pass
    return RESCALE_TRAINED if effective_rescale(rescale) is None \
        else RESCALE_NONE


def model_diams(path: str, kind: Optional[str] = None):
    """``(diam_mean, diam_labels)`` of a model, or None: a built-in by its
    name, a file from its saved state (small files only; read on the
    worker).  Never raises."""
    path = str(path or "")
    if kind == "pretrained" or not ("/" in path or "\\" in path):
        name = os.path.basename(path).lower()
        d = _BUILTIN_DIAM_MEAN.get(name, _BUILTIN_DEFAULT_DIAM)
        return d, d
    fp = file_fingerprint(path)
    if fp is None or fp["size"] >= SHA1_MAX_BYTES:
        return None
    try:
        import torch
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or "diam_mean" not in state:
            return None
        mean = float(state["diam_mean"])
        labels = float(state.get("diam_labels", state["diam_mean"]))
        return mean, labels
    except Exception:      # noqa: BLE001
        return None


def real_rescale_factor(rule: str, diameter_px, diams) -> Optional[float]:
    """The factor Cellpose really uses under *rule*, when the model's
    diameters are known (N2)."""
    if rule == RESCALE_NONE:
        return 1.0
    if not diams:
        return None
    mean, labels = diams
    try:
        if rule == RESCALE_FROM_WIDTH:
            return round(mean / float(diameter_px), 6)
        return round(mean / labels, 6) if labels else None
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _plain(obj) -> Dict[str, Any]:
    import dataclasses
    if obj is None:
        return {}
    if dataclasses.is_dataclass(obj):
        return json.loads(json.dumps(dataclasses.asdict(obj), default=str))
    if isinstance(obj, dict):
        return json.loads(json.dumps(obj, default=str))
    return {}


def run_snapshot(model: Optional[Dict[str, Any]], *, record=None,
                 source_path: str = "", seg_config=None, recon_config=None,
                 cell_width_um: Optional[float] = None,
                 diameter_px: Optional[float] = None) -> Dict[str, Any]:
    """The full ``predicted_with`` snapshot (v2.6), taken at LAUNCH:
    the model part at the top level (what the chips compare) plus the model
    fingerprints, the input (axes and how, shape, dtype, the source's
    fingerprint), every ``SegmentationConfig`` field with the effective
    resize factor, every ``ReconstructionConfig`` field.  The worker adds
    the sha1s and the cellpose / torch versions."""
    snap = dict(model or {})
    rec = record or {}
    snap["model_fingerprints"] = model_fingerprints(model)
    snap["input"] = {
        "axes": stack_axes(rec),
        "axes_how": "user" if rec.get("axes_override") else
        rec.get("axes_how"),
        "shape": rec.get("stack_shape"),
        "dtype": rec.get("dtype"),
        "source_fingerprint": file_fingerprint(source_path)
        if source_path else None,
    }
    seg = _plain(seg_config)
    rule = rescale_rule(getattr(seg_config, "rescale", None),
                        getattr(seg_config, "diameter", None))
    seg["rescale_rule"] = rule
    # known at launch only for "none"; the worker fills it per model (N2)
    seg["rescale_factor"] = 1.0 if rule == RESCALE_NONE else None
    seg["rescale_factor_by_model"] = {}
    seg["diameter_um"] = cell_width_um or None
    seg["diameter_px"] = diameter_px
    snap["seg"] = seg
    snap["recon"] = _plain(recon_config)
    snap["versions"] = {}
    return snap


def finish_run_snapshot(snap: Optional[Dict[str, Any]],
                        sha1_cache: Optional[Dict[str, str]] = None):
    """On the worker: add the model sha1s and the library versions."""
    if not snap:
        return snap
    cache = sha1_cache if sha1_cache is not None else {}
    fps = snap.get("model_fingerprints") or {}
    for side in ("lateral", "axial"):
        fp = fps.get(side)
        path = str(snap.get(side) or "")
        if fp and "sha1" not in fp and fp.get("size", 0) < SHA1_MAX_BYTES:
            if path not in cache:
                cache[path] = _sha1(path)
            fp["sha1"] = cache[path]
    seg = snap.get("seg")
    if isinstance(seg, dict) and seg.get("rescale_rule"):
        by = {}
        for side in ("lateral", "axial"):
            diams = model_diams(str(snap.get(side) or ""), snap.get("kind"))
            by[side] = real_rescale_factor(seg["rescale_rule"],
                                           seg.get("diameter_px"), diams)
        seg["rescale_factor_by_model"] = by
        if seg.get("rescale_factor") is None and by.get("lateral") is not None:
            seg["rescale_factor"] = by["lateral"]
    versions = {}
    for mod in ("cellpose", "torch"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:      # noqa: BLE001
            versions[mod] = None
    snap["versions"] = versions
    return snap
