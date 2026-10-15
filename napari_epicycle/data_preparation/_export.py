"""Step 1 patch persistence — write a patch to the canonical layout.

Pure-Python (no Qt/napari): writes the patch TIFF + its ``_metadata.json``
sidecar into ``artifacts/step1/patches/`` and records the patch (and its
source dataset) in ``step1_state.json``.

**Sidecar schema is FROZEN** — it is the contract consumed by EpiCycle-Viz
(`load_patch_boxes`) and must keep exactly the historical fields:
``patch_name, channel_names, original_shape, cropped_shape,
selected_z_range, bounding_box``.  The state registry indexes patches +
records provenance; it never duplicates the sidecar geometry.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, Optional, Sequence

import tifffile

from ._paths import step1_patches_dir
from ._state import load_state, register_patch, register_source, save_state
from .logic import allocate_unique_patch_name, split_indexed_name


def persist_patch(
    workspace: str,
    patch_name: str,
    data: Any,
    *,
    channel_names: Sequence[str],
    original_shape: Sequence[int],
    cropped_shape: Sequence[int],
    selected_z_range: Sequence[int],
    bounding_box: Any,
    source_path: Optional[str] = None,
    source_display_name: str = "",
    axes: str = "CZYX",
    source_axes: Optional[str] = None,
    voxel_size: Optional[Any] = None,
    source_kind: str = "raw",
    source_from: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Write *data* + sidecar into ``artifacts/step1/patches/`` and register it.

    Returns a dict with the FINAL ``patch_name`` (may differ from the input
    on collision), ``tiff_path``, ``sidecar_path``, ``relative_path`` (the
    workspace-relative posix path stored in state), and ``source_id``.

    ``axes`` is the layout of *data* itself (the patch is CZYX after the
    load-time transpose) and is written into the TIFF metadata.  ``source_axes``
    (optional) is the SOURCE file's distinct on-disk order (e.g. ``ZCYX``)
    recorded on the source-registry entry for provenance; it defaults to
    ``axes`` when not given.  ``voxel_size`` (optional ``[Z, Y, X]`` µm) is the
    acquisition spacing, also recorded on the source entry.  Neither reaches
    the FROZEN sidecar (source-level provenance, not per-patch geometry).

    ``source_kind`` ("raw" | "processed") records WHICH pixels were cropped —
    the raw source or the prepared processed store — and is stamped on the
    patch REGISTRY row (never the FROZEN sidecar).  Additive; defaults to
    "raw".

    ``source_from`` (optional) records WHERE those pixels were read from when it
    was NOT the original file — today only ``"prepared_store"``, written while
    Step 1 is serving a source whose original image file it can't find (Brick 4a
    a4).  It is the ONE additive key on the otherwise-frozen sidecar, and it is
    omitted entirely in the normal case, so every existing reader is unaffected.
    """
    patches_dir = step1_patches_dir(workspace)
    os.makedirs(patches_dir, exist_ok=True)

    # Load state ONCE, then resolve the name against the LIVE workspace
    # (registry + disk) through the single naming authority: the export-time
    # workspace decides the final name (not the extract-time snapshot -> D8),
    # and a collision resolves to an indexer-visible {prefix}_{max+1} (B2/B3).
    state = load_state(workspace)
    name = allocate_unique_patch_name(workspace, patch_name, state=state)
    tiff_path = patches_dir / f"{name}.tiff"
    sidecar_path = patches_dir / f"{name}_metadata.json"

    # --- TIFF (axes metadata preserved) ---
    tifffile.imwrite(str(tiff_path), data, metadata={"axes": axes})

    # --- Sidecar (schema FROZEN — Viz contract) ---
    sidecar = {
        "patch_name": name,
        "channel_names": list(channel_names),
        "original_shape": list(original_shape),
        "cropped_shape": list(cropped_shape),
        "selected_z_range": list(selected_z_range),
        "bounding_box": bounding_box,
    }
    if source_from:
        sidecar["source_from"] = source_from
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=4)

    # --- Registry (workspace-relative posix paths) ---
    rel_tiff = f"artifacts/step1/patches/{name}.tiff"
    rel_sidecar = f"artifacts/step1/patches/{name}_metadata.json"
    source_id = ""
    if source_path:
        # Persist the numbering prefix (derived from the FINAL name) on the
        # source so the badge can match custom-named patches per-file (D-3).
        prefix, _ = split_indexed_name(name)
        source_id = register_source(
            state, path=source_path, display_name=source_display_name,
            axes=source_axes or axes, voxel_size=voxel_size,
            timestamp=timestamp, prefix=prefix,
        )
    register_patch(
        state, patch_name=name, relative_path=rel_tiff,
        sidecar_path=rel_sidecar, source_id=source_id,
        source_kind=source_kind, timestamp=timestamp,
    )
    save_state(workspace, state)

    return {
        "patch_name": name,
        "tiff_path": str(tiff_path),
        "sidecar_path": str(sidecar_path),
        "relative_path": rel_tiff,
        "source_id": source_id,
    }


def _fsync_path(path: str) -> None:
    """Best-effort flush a just-written file's bytes to disk (durability).

    Opens read-write so ``os.fsync`` → Windows ``FlushFileBuffers`` (which needs
    write access on the handle) actually flushes."""
    try:
        fd = os.open(path, os.O_RDWR)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except Exception:
        # fsync is a durability nicety, not a correctness requirement; a
        # platform that refuses it (or a filesystem without it) must not fail
        # the whole trim.
        pass


def _safe_remove(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def crop_patch_in_place(
    tiff_path: str,
    sidecar_path: str,
    z0: int,
    z1: int,
) -> Dict[str, Any]:
    """Crop a saved patch's TIFF in place on the Z axis, RECOVERABLY.

    This OVERWRITES the patch pixels + rewrites its FROZEN sidecar's
    ``cropped_shape`` + ``selected_z_range`` — the ④ "Trim" action.  Pure /
    file-system-only (no Qt, no napari), so it runs on a worker thread and is
    directly testable.

    *z0*, *z1* are in the CURRENT patch's Z coordinates (``0 <= z0 < z1 <=
    current_depth``).  The cropped patch pixels are ``data[:, z0:z1]`` (4D CZYX)
    / ``data[z0:z1]`` (3D ZYX) of the pre-crop TIFF — byte-identical to a manual
    crop, no re-scaling.

    **Two-file atomicity is impossible** (TIFF + sidecar are two ``os.replace``
    calls); this makes the pair RECOVERABLE instead:

    1. Read the TIFF with ``squeeze=False`` and NO memmap
       so a singleton ``C=1`` axis is preserved (CZYX stays 4D).  The read
       handle is closed before any replace (Windows ``os.replace``-over-open
       hazard).
    2. Crop the Z axis by NDIM: 4D → axis 1, 3D → axis 0.  ndim 2 / other is
       REJECTED with a clear error.
    3. Write BOTH temps (fsync) INSIDE the patches dir (same volume — else
       ``os.replace`` raises ``WinError 17``), rename the original TIFF →
       ``{name}.tiff.bak``, ``os.replace(temp_tiff)`` (TIFF-first), then
       ``os.replace(temp_sidecar)``.  If a replace fails, ROLL BACK the TIFF
       from ``.bak`` and re-raise; the ``.bak`` is removed on success.

    The sidecar is updated COHERENTLY: ``cropped_shape`` → the new (smaller)
    shape; ``selected_z_range`` → **ABSOLUTE source-Z with offset composition**
    ``[prev_z0 + z0, prev_z0 + z1]`` where ``prev_z0`` is the current sidecar's
    ``selected_z_range[0]`` (0 if absent) — so the ④ location box stays correct
    and a re-crop composes.  ``original_shape`` /
    ``bounding_box`` / ``channel_names`` / ``patch_name`` are UNCHANGED.

    Returns a dict: ``cropped_shape``, ``selected_z_range`` (absolute),
    ``z0``, ``z1``, ``ndim``.
    """
    tiff_path = str(tiff_path)
    sidecar_path = str(sidecar_path)
    z0 = int(z0)
    z1 = int(z1)
    if z1 <= z0 or z0 < 0:
        raise ValueError(
            f"invalid Z range [{z0}, {z1}] — need 0 <= z0 < z1")

    # (1) Read pixels + axes.  ``TiffFile.asarray(squeeze=False)`` is the
    # no-memmap, ndim-preserving read that ``imread(path, squeeze=False)``
    # mandates; the handle is closed by the ``with`` before any replace.
    with tifffile.TiffFile(tiff_path) as tf:
        data = tf.asarray(squeeze=False)
        try:
            axes = tf.series[0].axes
        except Exception:
            axes = None
    ndim = int(getattr(data, "ndim", 0))

    # (2) Resolve the Z axis by NDIM.
    if ndim == 4:
        zaxis = 1
    elif ndim == 3:
        zaxis = 0
    else:
        raise ValueError(
            f"cannot Z-crop a {ndim}D patch (expected 3D ZYX or 4D CZYX)")
    depth = int(data.shape[zaxis])
    if z1 > depth:
        raise ValueError(
            f"Z range [{z0}, {z1}] exceeds patch depth {depth}")

    cropped = data[:, z0:z1] if ndim == 4 else data[z0:z1]
    cropped = cropped.copy()   # detach from the (about-to-be-freed) full array

    # Preserve the SAME axes metadata; fall back to ndim-derived if the file's
    # series axes is missing or its length disagrees with the array's ndim.
    if not (isinstance(axes, str) and len(axes) == ndim):
        axes = "CZYX" if ndim == 4 else "ZYX"

    # Rewrite the FROZEN sidecar COHERENTLY (absolute Z).
    with open(sidecar_path, "r", encoding="utf-8") as f:
        sidecar = json.load(f)
    prev = sidecar.get("selected_z_range")
    if isinstance(prev, (list, tuple)) and len(prev) == 2:
        prev_z0 = int(prev[0])
    else:
        prev_z0 = 0
    sidecar["cropped_shape"] = list(cropped.shape)
    sidecar["selected_z_range"] = [prev_z0 + z0, prev_z0 + z1]

    # (3) Recoverable write — both temps INSIDE the patches dir (same volume).
    patches_dir = os.path.dirname(os.path.abspath(tiff_path))
    tmp_tiff_fd, tmp_tiff = tempfile.mkstemp(dir=patches_dir, suffix=".tiff.tmp")
    os.close(tmp_tiff_fd)
    tmp_side_fd, tmp_side = tempfile.mkstemp(dir=patches_dir, suffix=".json.tmp")
    os.close(tmp_side_fd)
    bak = tiff_path + ".bak"
    try:
        tifffile.imwrite(tmp_tiff, cropped, metadata={"axes": axes})
        _fsync_path(tmp_tiff)
        with open(tmp_side, "w", encoding="utf-8") as f:
            json.dump(sidecar, f, indent=4)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        _safe_remove(tmp_tiff)
        _safe_remove(tmp_side)
        raise

    # TIFF-first: rename original → .bak, promote the trimmed TIFF, then the
    # sidecar.  Any replace failure restores the original from .bak and re-raises
    # (a concurrent Windows reader → PermissionError → clean error, no crash).
    try:
        os.replace(tiff_path, bak)                 # original → .bak
    except BaseException:
        _safe_remove(tmp_tiff)
        _safe_remove(tmp_side)
        raise
    try:
        os.replace(tmp_tiff, tiff_path)            # trimmed TIFF into place
    except BaseException:
        try:
            os.replace(bak, tiff_path)             # restore full-Z original
        except Exception:
            pass
        _safe_remove(tmp_tiff)
        _safe_remove(tmp_side)
        raise
    try:
        os.replace(tmp_side, sidecar_path)         # trimmed sidecar into place
    except BaseException:
        try:
            os.replace(bak, tiff_path)             # ROLL BACK the TIFF (HIGH-1)
        except Exception:
            pass
        _safe_remove(tmp_side)
        raise
    _safe_remove(bak)                              # success

    return {
        "cropped_shape": list(cropped.shape),
        "selected_z_range": [prev_z0 + z0, prev_z0 + z1],
        "z0": z0,
        "z1": z1,
        "ndim": ndim,
    }
