import math
import time
import weakref
from collections import Counter, namedtuple

from napari.layers import Image, Shapes
import numpy as np
from .logic import (
    extract_patches_from_shapes, detect_sample_range_all_channels,
    find_next_patch_index, count_patches_for_prefix, count_patches_for_source,
    step1_sidecar_scan_dir,
    generate_max_intensity_project, downsampled_max_intensity_project,
    filter_extractable_boxes,
    box_signature, select_uncommitted_boxes,
    draw_boxes_on_projection_matplotlib, prepare_rgb_projection,
    scan_tiff_folder, transpose_to_czyx, split_channels,
    EAGER_RAM_FRACTION, format_bytes,
)
from ._largedata import open_lazy, materialize, resolve_axes
from ._box_tool import (
    Step1BoxTool, full_shape_from_source, place_box,
    _units_for_voxel, _apply_layer_units, _apply_layer_scale,
)
from ._export import crop_patch_in_place, persist_patch
from ._state import (
    load_state, save_state, upsert_source, relink_source,
    _norm_path, _next_source_id, _now,
    _classify_voxel_source, set_processed_store, get_processed_store,
    empty_state, all_patches,
    resolve_box_style, set_patch_box_style, set_source_box_style,
    set_corrupt_state_hook,
)
from packages.epicycle_largedata.multiscale import (
    convert_to_ome_zarr, open_multiscale, read_store_source, sampled_file_sha,
)
from packages.epicycle_largedata.axes import read_voxel_spacing
from packages.epicycle_largedata.exceptions import ConvertCancelled
# Step-1 preprocessing engine (Brick 2a — pure, napari-free).  Referenced
# through the MODULE GLOBALS below (not called by dotted path) so a test can
# monkeypatch `controller._pp_execute` — the same seam as `convert_to_ome_zarr`.
from packages.epicycle_preprocessing.ops import (
    execute as _pp_execute,
    max_local_tile_scratch_bytes as _pp_max_tile_scratch_bytes,
    percentile_bounds as _pp_percentile_bounds,
    to_uint16 as _pp_to_uint16,
)
# Streaming (out-of-core) whole-volume engine (Phase 2a — pure, napari-free).
# Referenced through the MODULE GLOBALS below (not a dotted path) so a test can
# monkeypatch `controller._pp_stream_to_store` — the same seam as `_pp_execute`.
from packages.epicycle_preprocessing.streaming import (
    stream_process_to_store as _pp_stream_to_store,
    _fused_plan as _pp_fused_plan,
    _recipe_hash as _pp_recipe_hash,
)
from packages.epicycle_preprocessing import recipe as _pp_recipe
from packages.epicycle_preprocessing.exceptions import PreprocessingCancelled

from qtpy.QtCore import Qt, QTimer, QThread, QObject, Signal


# Intensity-profile plotting (④ Z-truncate)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from qtpy.QtWidgets import QVBoxLayout
from matplotlib.widgets import SpanSelector

import json
import os
import sys

from dataclasses import dataclass, field


# Bounded parallelism for the large-data Prepare convert (multiscale OPT 2).
# Superseded as the DEFAULT by the per-workspace adaptive config
# (`_current_largedata_config` → the saved config or the machine-scaled
# "balanced" default); retained as a last-resort override sentinel.  ``None``
# lets the config / converter decide (min(4, os.cpu_count())).
_LARGEDATA_PREP_WORKERS = None


# ④ Review & Library row provenance (F2).  `_refresh_library_from_registry`
# stamps each library QListWidgetItem with the registry row's provenance so the
# `BoxListPanel` can render size/channel detail from the sidecar, thumbnail the
# saved TIFF, and gate interactivity by source WITHOUT re-reading the registry
# per row.  +7/+8/+9 are already taken (checkable init marker / box-dim delegate
# / row cache key — see ui.py), so these start at +10.  Imported by ui.py, which
# already imports from this module (no circular import — controller never
# imports ui).
LIBRARY_ROLE_SOURCE_ID = Qt.UserRole + 10
LIBRARY_ROLE_REL_PATH = Qt.UserRole + 11    # workspace-relative patch TIFF
LIBRARY_ROLE_SIDECAR = Qt.UserRole + 12     # workspace-relative _metadata.json
LIBRARY_ROLE_INTERACTIVE = Qt.UserRole + 13  # bool: belongs to the loaded source


def _effective_ndim(data):
    """The effective ndim of a napari layer's ``.data``.

    A MULTISCALE layer exposes ``.data`` as a napari ``MultiScaleData`` (a
    ``Sequence`` of pyramid-level arrays, NOT a plain ``list``); it carries its
    own ``.ndim`` (a level's ndim — all levels share it), so it takes the
    ``getattr(data, "ndim", ...)`` fallback below.  A hand-built ``list``/tuple
    of level arrays takes the explicit branch.  A plain layer exposes an array
    with ``.ndim``.  Returns ``None`` for anything without a usable ndim (a bare
    list of non-arrays, a hand-added non-array), so callers can filter on
    ``== 3`` without an ``AttributeError`` on the sequence ``.data``.
    """
    if isinstance(data, (list, tuple)):
        return getattr(data[0], "ndim", None) if len(data) else None
    return getattr(data, "ndim", None)


def _abspath_norm(p) -> str:
    """A path canonicalized for identity comparison: absolute + OS-normcased.

    Used by :meth:`DataPreparationController._fresh_prepared_store_for` to match
    a stored ``source_path`` / ``path`` against the current selection regardless
    of relative-vs-absolute form or Windows case/separator differences.
    """
    return os.path.normcase(os.path.abspath(str(p)))


def _z_intensity_curves_pure(data, channel_names):
    """Per-channel min-max-normalized mean-intensity-through-Z curves for the
    ④ intensity plot.

    Returns ``(curves, n_z)`` with ``curves = [(label, np.ndarray), ...]``.
    Accepts a 4D ``(C, Z, Y, X)`` patch (the persisted layout) or a 3D
    ``(Z, Y, X)`` single-channel patch (promoted to C=1).  Pure (no Qt) so the
    same reduction runs on the worker thread (disk patch) and inline (in-memory
    patch)."""
    data = np.asarray(data)
    if data.ndim == 3:
        data = data[None]                       # (1, Z, Y, X)
    n_c, n_z = int(data.shape[0]), int(data.shape[1])
    curves = []
    for i in range(n_c):
        label = channel_names[i] if i < len(channel_names) else f"ch{i}"
        zi = np.mean(data[i], axis=(1, 2))
        lo, hi = float(np.min(zi)), float(np.max(zi))
        zi = (zi - lo) / (hi - lo) if hi > lo else np.zeros_like(zi)
        curves.append((label, zi))
    return curves, n_z


#: Retain intensity-plot workers whose dialog/controller is being torn down while
#: their (un-interruptible `tifffile.imread`) is still running, so the QThread is
#: never destroyed mid-run (`QThread destroyed while running` abort — audit HIGH).
#: A detached worker self-removes on `finished`.
_LIVE_INTENSITY_WORKERS = set()


class _IntensityProfileWorker(QThread):
    """Reads a saved patch TIFF off the UI thread and reduces it to the tiny
    per-channel per-Z mean-intensity curves for ④'s intensity plot.

    A reopened/captured patch has no in-memory pixels (the ``extracted_patches``
    cache is session-scoped), which is exactly the "No data found" case that
    left the plot blank.  The volume (tens of MB) is read + reduced HERE; only
    the small curves cross back to the main thread (`done`)."""

    done = Signal(str, object, int, object)  # patch_name, curves, n_z, z_range|None
    failed = Signal(str, str)                 # patch_name, message

    def __init__(self, patch_name, path, channel_names, parent=None):
        super().__init__(parent)
        self._patch_name = patch_name
        self._path = path
        self._channel_names = list(channel_names or [])

    def run(self):
        try:
            import tifffile
            data = tifffile.imread(self._path)
            curves, n_z = _z_intensity_curves_pure(data, self._channel_names)
            # Also detect the sample Z-range (UNION across channels) off the UI
            # thread while the pixels are loaded, so ④'s "snap" works for a
            # REOPENED patch (which has no in-memory pixels).  Failure-safe: a
            # detection error still lets the plot render (z_range=None).
            z_range = None
            try:
                vol = data if getattr(data, "ndim", 0) >= 4 else data[None]
                z0, z1 = detect_sample_range_all_channels(vol)
                z_range = (int(z0), int(z1))
            except Exception:
                z_range = None
            self.done.emit(self._patch_name, curves, n_z, z_range)
        except Exception as exc:        # noqa: BLE001
            self.failed.emit(self._patch_name, str(exc))


class _SourceLoadWorker(QObject):
    """Reads a Step-1 source off the main thread so the GUI stays live.

    ONLY the disk read runs here (`open_lazy` + `materialize`).  ALL viewer /
    layer / UI work happens back on the main thread in the controller's
    `_on_source_loaded` handler — `viewer.add_image` is main-thread-only.

    The produced ``data`` is byte-identical to the old synchronous read; the
    extract path depends on it (analysis reads the data in memory, never a
    re-read).  `open_lazy` / `materialize` are imported at module top.
    """

    finished = Signal(object, object)   # (lazy_handle, data ndarray)
    error = Signal(str)
    cancelled = Signal()

    def __init__(self, path, use_memmap):
        super().__init__()
        self._path = path
        self._use_memmap = use_memmap
        self._cancel_requested = False

    def request_cancel(self):
        self._cancel_requested = True

    def run(self):
        try:
            if self._cancel_requested:
                self.cancelled.emit()
                return
            lazy = open_lazy(self._path)
            data = materialize(lazy)
            if self._cancel_requested:
                self.cancelled.emit()
                return
            self.finished.emit(lazy, data)
        except BaseException as exc:   # noqa: BLE001
            # A worker thread must ALWAYS emit a terminal signal — even on a
            # non-Exception (SystemExit etc.) — so `thread.quit` fires and the
            # (thread, worker) pair is never leaked mid-drain (audit LOW).
            self.error.emit(f"{exc}")


def _last_resort_report(prefix: str, msg: str) -> None:
    """Emit a state-problem report when the Diagnostics pane is unreachable.

    Only for the cases where there is no GUI to route to at all — no
    QApplication (headless / import time), no relay, or the marshalling itself
    failed.  Goes to STDERR, not `print`: this is a fallback for an
    unreportable problem, never a substitute for the `_log` sink (which is
    main-thread-only and is exactly what we could not reach here).
    """
    try:
        sys.stderr.write("[%s] %s\n" % (prefix, msg))
    except Exception:                          # noqa: BLE001
        pass


#: Failed-save retry backoff for `_flush_source_state_save`: first retry after
#: 4 s, doubling, capped at 30 s.  A transient lock recovers on the first tick;
#: a genuinely-unavailable workspace settles into a slow poll instead of
#: hammering an unreachable path for the rest of the session.
_STATE_SAVE_RETRY_MS = 4000
_STATE_SAVE_RETRY_MAX_MS = 30000


class _StateProblemRelay(QObject):
    """Marshals a state-problem report onto the GUI thread.

    Lives in the main thread (parented to the Step-1 widget), so an emit from a
    worker thread is delivered QUEUED — which is what makes it safe for
    `load_state`'s corrupt-state hook to fire from the extract QThread (via
    `_export.persist_patch`) while `_report_state_problem` writes to a QLabel
    and a QTextEdit.
    """

    reported = Signal(str, str)


class _LargeDataPrepWorker(QObject):
    """Converts a Step-1 source to a multiscale OME-Zarr store off the main
    thread so the GUI stays live during the (minutes-long, I/O-bound) write.

    ONLY the conversion runs here (`convert_to_ome_zarr`, which streams the
    pyramid chunk-by-chunk and never materializes the whole volume).  ALL
    viewer / layer work — opening the store as a lazy multiscale array and
    adding the display layers — happens back on the main thread in the
    controller's `_on_large_data_ready` handler (`viewer.add_image` is
    main-thread-only), mirroring `_SourceLoadWorker`.

    The conversion is a single, non-interruptible dask stream; Cancel therefore
    DISCARDS the result (the controller bumps its token so the finished handler
    goes stale) rather than interrupting mid-write — the same semantics as the
    source read's Cancel.  `convert_to_ome_zarr` is referenced through the
    module global so a test can monkeypatch it.
    """

    finished = Signal(str)      # out_store path
    error = Signal(str)
    cancelled = Signal()
    #: (levels written, total levels) - the convert's ONE real denominator.
    #: Emitted from the worker thread, so it must be a SIGNAL (Qt delivers it
    #: queued to the GUI thread); a direct widget call from here would touch Qt
    #: off the main thread.
    progress = Signal(int, int)

    def __init__(self, src, out_store, voxel_size, image_axes,
                 num_workers=None, chunk=None, compressor=None, downsample=None,
                 min_size=None):
        super().__init__()
        self._src = src
        self._out_store = out_store
        self._voxel_size = voxel_size
        self._image_axes = image_axes
        # Bounded parallelism + pyramid knobs for the convert (multiscale OPT 2 +
        # the adaptive Prepare-settings config).  All optional + defaulted to
        # None so existing 4-arg / num_workers-only constructions stay valid;
        # None means "use convert_to_ome_zarr's own default" (min(4, cpu) for
        # workers, 64/2/256 for chunk/downsample/min_size).
        self._num_workers = num_workers
        self._chunk = chunk
        self._compressor = compressor
        self._downsample = downsample
        self._min_size = min_size
        self._cancel_requested = False

    def request_cancel(self):
        self._cancel_requested = True

    def run(self):
        try:
            if self._cancel_requested:
                self.cancelled.emit()
                return
            # Pass only the knobs that were set; a None knob defers to the
            # converter's own default (keeps the pre-config behaviour intact).
            convert_kwargs = dict(
                voxel_size=self._voxel_size,
                image_axes=self._image_axes,
                overwrite=True,
                num_workers=self._num_workers,
            )
            if self._chunk is not None:
                convert_kwargs["chunk"] = self._chunk
            if self._compressor is not None:
                convert_kwargs["compressor"] = self._compressor
            if self._downsample is not None:
                convert_kwargs["downsample"] = self._downsample
            if self._min_size is not None:
                convert_kwargs["min_size"] = self._min_size
            # REAL cancel: the converter polls this between pyramid levels and
            # raises `ConvertCancelled`, so a cancelled Prepare actually STOPS
            # writing (and drops its partial store) instead of running the whole
            # convert to completion just to have its result discarded.
            convert_kwargs["should_cancel"] = lambda: self._cancel_requested
            # Determinate progress: the converter reports each pyramid level as
            # it lands (one level is the granularity available - a level write
            # is a single, non-interruptible dask stream).  Level 0 is the long
            # one (the only source read), so the bar sits at 0/N for most of the
            # run and then climbs; that is honest, unlike a marquee.
            convert_kwargs["on_level"] = (
                lambda done, total: self.progress.emit(int(done), int(total)))
            convert_to_ome_zarr(
                self._src, self._out_store, **convert_kwargs)
            if self._cancel_requested:
                self.cancelled.emit()
                return
            self.finished.emit(str(self._out_store))
        except ConvertCancelled:
            # A user cancel is NOT an error: routing it to `error` painted a red
            # "Failed to prepare" for a failure the user deliberately caused.
            self.cancelled.emit()
        except BaseException as exc:   # noqa: BLE001
            # Same terminal-signal invariant as `_SourceLoadWorker.run`: always
            # emit so `thread.quit` fires and the pair is never leaked.
            self.error.emit(f"{exc}")


# --------------------------------------------------------------------------- #
# Step-1 EXTRACT on a worker (F1).  `extract_patches` used to run its whole
# slice→materialize→write pipeline synchronously on the UI thread (≈600 MB per
# patch off disk + TIFF/sidecar/registry write per box) — the GUI froze.  The
# refactor keeps EVERY widget/viewer read + all UI mutation on the main thread
# (PREP + the finish/written/progress slots) and moves only numpy slicing +
# `persist_patch` file I/O onto a QThread, mirroring `_SourceLoadWorker` /
# `_PreprocProcessToStoreWorker`.  Data-integrity contract: the
# extracted pixels + the on-disk TIFF/sidecar/registry rows + the committed box
# signatures (BOTH the XY-only and the Z-inclusive forms) are byte-identical to
# the old synchronous path.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class _ExtractJob:
    """An immutable snapshot of EVERYTHING the worker needs, taken on the main
    thread in PREP so the worker never touches a Qt widget, a napari layer, or a
    live box-tool array.

    ``boxes`` + ``z_ranges`` are DEEP-COPIED off the box tool's live rectangles
    (grill HIGH-3): the surface lock disables the box *buttons* but NOT canvas
    mouse interaction, so a drag mid-run would otherwise mutate the very ndarray
    the worker slices → a region mismatch between the validated bbox and the
    persisted pixels.  ``crop_source`` is the read-only lazy full-res handle (or
    the materialized fallback stack); it is the SOLE reader during the run —
    nothing on the main thread re-reads ``_source_lazy`` while the worker is in
    flight (grill MED-6: `tifffile.memmap` / `da.from_zarr` are concurrent-read
    safe; the compressed `ZarrTiffStore` path never has a live lazy handle here
    because it displays from a fully-materialized copy).  ``persist`` is the
    complete per-run persist-param set gathered from the widgets by
    :meth:`DataPreparationController._gather_persist_params` (grill MED-2 — the
    full provenance set, incl. ``source_axes`` + ``voxel_size``).
    """

    crop_source: object
    channel_indices: object            # list[int] | None (ticked channels, order)
    boxes: tuple                       # deep-copied (4,2) np arrays, FINAL order
    z_ranges: tuple                    # per-box [z0, z1] | None, aligned to boxes
    channel_names: tuple               # ticked channel names (stage `channels`)
    name_prefix: str
    start_index: int
    source_kind: str                   # "raw" | "processed"
    persist: dict = field(default_factory=dict)


class _ExtractWorker(QObject):
    """Slices + materializes each ROI crop and writes it to disk off the UI
    thread.  For each box, IN ORDER: slice+materialize via
    ``extract_patches_from_shapes`` (``progress_bar=None`` — grill MED-7: never
    touch Qt from the worker), then ``persist_patch`` (the pure writer — TIFF +
    FROZEN sidecar + registry row).  Emits one ``written`` per materialized box
    (write success OR failure — grill MED-4) carrying the array + the persist
    result + BOTH box-signatures + the cropped Z range, so the main-thread
    ``_on_extract_written`` slot rebuilds the in-memory stage and commits the
    sigs.  Cooperative cancel: ``request_cancel`` sets a flag checked between
    boxes; the run then emits ``finished(n_written, was_cancelled=True)``.

    ``run`` wraps EVERYTHING in ``try/except BaseException`` and ALWAYS emits a
    terminal signal (``finished`` or ``error``) so ``thread.quit`` fires and the
    (thread, worker) pair is never leaked mid-flight.
    """

    #: One persisted patch: a plain dict (``object`` payload so it crosses the
    #: thread boundary intact).  Keys: name, data, bounding_box, cropped_z_range,
    #: source_kind, channels, persist_ok, sig_xy, sig_z.
    written = Signal(object)
    #: (i, n, phase) — phase is "reading" (before materialize) or "writing"
    #: (after a successful write), driving the ③ bar on the main thread.
    progress = Signal(int, int, str)
    #: (n_written, was_cancelled) — the normal terminal signal.
    finished = Signal(int, bool)
    #: fatal error message (a per-box write failure is NON-fatal — see run()).
    error = Signal(str)

    def __init__(self, job: "_ExtractJob"):
        super().__init__()
        self._job = job
        self._cancel_requested = False

    def request_cancel(self):
        # Set from the MAIN thread; read between boxes.  A plain bool read/write
        # across threads is safe under CPython's GIL.
        self._cancel_requested = True

    def run(self):
        job = self._job
        n = len(job.boxes)
        n_written = 0
        was_cancelled = False
        try:
            for i, box in enumerate(job.boxes):
                if self._cancel_requested:
                    was_cancelled = True
                    break
                self.progress.emit(i, n, "reading")
                zr = job.z_ranges[i] if job.z_ranges else None
                # Slice + materialize ONLY this crop (byte-identical to the
                # synchronous all-boxes call: naming is `{prefix}_{start+idx}`
                # with idx=0 here, so `start_index + i` reproduces the same name
                # + array the batch call produced for box i).
                patches = extract_patches_from_shapes(
                    job.crop_source,
                    [box],
                    progress_bar=None,
                    name_prefix=job.name_prefix,
                    start_index=job.start_index + i,
                    channel_indices=job.channel_indices,
                    z_ranges=[zr],
                )
                (name, array), = patches.items()

                # Mirror `_persist_patch_to_disk`'s per-patch write EXACTLY (the
                # extract-created stage carries no `selected_z_range`, so the
                # array is NOT further cropped and the frozen sidecar's Z range
                # is the full extent of the already-cropped array).
                original_shape = array.shape
                if hasattr(array, "compute"):
                    data = array.compute()
                else:
                    data = np.asarray(array)
                # B3: record the SOURCE-ABSOLUTE Z range so the ④ location box +
                # the re-extract dedup sig place the patch at its TRUE source depth.
                # Was [0, depth] (patch-relative), which discarded the box's origin
                # `zr[0]` for a Z-narrowed box → the overlay drew at depth/2.  Use
                # `zr[0] + depth` (NOT `zr[1]`) so width == the actual cropped depth
                # even when a processed-path geometry op resampled Z (grill MED-1);
                # matches the absolute convention `crop_patch_in_place` composes on.
                z0_src = int(zr[0]) if zr else 0
                cropped_z_range = [z0_src, z0_src + int(original_shape[1])]
                bb = box.tolist()

                persist_ok = False
                persist_result = None
                try:
                    persist_result = persist_patch(
                        job.persist["workspace"],
                        name,
                        data,
                        channel_names=list(job.channel_names),
                        original_shape=original_shape,
                        cropped_shape=data.shape,
                        selected_z_range=cropped_z_range,
                        bounding_box=bb,
                        source_path=job.persist["source_path"],
                        source_display_name=job.persist["source_display_name"],
                        source_axes=job.persist["source_axes"],
                        voxel_size=job.persist["voxel_size"],
                        source_kind=job.source_kind,
                        source_from=job.persist.get("source_from"),
                    )
                    persist_ok = True
                    n_written += 1
                except Exception as exc:   # noqa: BLE001 — one bad write must not
                    # crash the run (mirrors the synchronous per-patch guard).
                    persist_result = f"{exc}"

                self.written.emit({
                    "name": name,
                    "data": data,
                    "bounding_box": bb,
                    "cropped_z_range": cropped_z_range,
                    "source_kind": job.source_kind,
                    "channels": list(job.channel_names),
                    "persist_ok": persist_ok,
                    "persist_result": persist_result,
                    "sig_xy": box_signature(bb),
                    "sig_z": box_signature(bb, cropped_z_range),
                })
                if persist_ok:
                    self.progress.emit(i + 1, n, "writing")
            self.finished.emit(n_written, was_cancelled)
        except BaseException as exc:   # noqa: BLE001
            # A worker thread must ALWAYS emit a terminal signal (even on a
            # non-Exception) so `thread.quit` fires and the pair is never leaked.
            self.error.emit(f"{exc}")


# --------------------------------------------------------------------------- #
# Step-1 ④ Z-crop TRIM on a worker (mirrors the F1 EXTRACT machinery).  "Trim"
# OVERWRITES a saved patch's TIFF (crop the Z axis) + rewrites its FROZEN
# sidecar via the RECOVERABLE `crop_patch_in_place` writer.  The file I/O runs
# on a QThread (large patches for this user's data); the confirm/validate/lock
# stay on the main thread in `_on_set_zrange_clicked` (PREP), and the
# finish/error slots (④ refresh + the single unlock) run back on the main
# thread.  A trim is NOT cancellable mid-replace (unlike EXTRACT): step-exit
# RETAINS it against GC + lets it finish, and a token/`_left_step1` guard makes
# a late FINISH inert.
# --------------------------------------------------------------------------- #

class _TrimWorker(QObject):
    """Crops one saved patch's TIFF + sidecar in place off the UI thread.

    ``run`` calls the pure :func:`crop_patch_in_place` (read → crop Z → temps →
    recoverable replace) and ALWAYS emits a terminal signal so ``thread.quit``
    fires and the (thread, worker) pair is never leaked.  A residual Windows
    sharing violation on the ``os.replace`` (a stray live reader — the capture
    window) surfaces as a clean ``error`` ("Patch file busy …"), never an
    uncaught crash."""

    #: One trim finished — the info dict (name + current-coords z0/z1 + the new
    #: absolute selected_z_range + the new cropped_shape) crosses back so
    #: `_on_trim_finished` crops the in-memory stage + refreshes ④.
    finished = Signal(object)
    #: fatal error message.
    error = Signal(str)

    def __init__(self, patch_name, tiff_path, sidecar_path, z0, z1):
        super().__init__()
        self._name = patch_name
        self._tiff = tiff_path
        self._sidecar = sidecar_path
        self._z0 = int(z0)
        self._z1 = int(z1)

    def run(self):
        try:
            result = crop_patch_in_place(
                self._tiff, self._sidecar, self._z0, self._z1)
            self.finished.emit({
                "name": self._name,
                "z0": self._z0,
                "z1": self._z1,
                "selected_z_range": result["selected_z_range"],
                "cropped_shape": result["cropped_shape"],
            })
        except PermissionError:
            # HIGH-3: a live reader still held the TIFF when `os.replace` fired
            # (the original is restored from `.bak` inside crop_patch_in_place).
            self.error.emit(
                "Patch file busy — close its capture window and retry.")
        except BaseException as exc:   # noqa: BLE001
            self.error.emit(f"{exc}")


# --------------------------------------------------------------------------- #
# Step-1 preprocessing PREVIEW (Brick 2b) — pure compute helpers + QThread
# worker.  The PREVIEW computes the recipe on a small CENTER test crop of the
# loaded source (in-memory), then draws a per-channel before→after histogram in
# ②'s plot.  Apply (bake to disk) stays a stub (Brick 3).
# --------------------------------------------------------------------------- #

#: The per-channel napari colormaps the SOURCE display layers are shown with —
#: picked POSITIONALLY by channel index in `_add_source_display_layers` (a
#: single-channel source is "gray").  Promoted to a module constant so the
#: preprocessing PREVIEW render (`_preview_colormap_for_channel`) can reuse the
#: EXACT same mapping — so a ⟨preview⟩ layer's colour matches its source
#: channel instead of inventing a parallel palette (Brick 2c).
_SOURCE_DISPLAY_CMAPS = ("magenta", "green", "cyan", "yellow", "red", "blue")

#: Representative hex for each `_SOURCE_DISPLAY_CMAPS` colormap NAME — the SINGLE
#: source of truth for tinting the Step-1 preprocessing chrome (② recipe cards,
#: ③ channel-picker swatches, before/after histogram curves) so it MATCHES what
#: napari renders on the canvas for that channel.  Before this map the chrome used
#: a parallel palette in a different order, so channel 0's card read green while
#: its canvas layer was magenta (the "channels look switched" bug).  Shades are
#: softened from the pure colormap primaries so they read on the dark card header
#: band while keeping each channel's HUE identity.
_SOURCE_CMAP_HEX = {
    "magenta": "#e060e0",
    "green": "#5fd35f",
    "cyan": "#4fd0d0",
    "yellow": "#e6d24a",
    "red": "#e05a5a",
    "blue": "#5a8ce0",
}


def source_cmap_hex(cmap_name):
    """Representative hex for a `_SOURCE_DISPLAY_CMAPS` colormap NAME, for Qt /
    pyqtgraph styling that must agree with the canvas colour.  Unknown names fall
    back to a neutral grey (never raises)."""
    return _SOURCE_CMAP_HEX.get(str(cmap_name).lower(), "#9a9a9a")


def _channel_display_hex(ci):
    """Hex for source-channel index *ci*, keyed off `_SOURCE_DISPLAY_CMAPS` — the
    EXACT positional mapping `_add_source_display_layers` assigns to the canvas.
    Use this for every per-channel swatch / curve so the chrome agrees with the
    canvas layer instead of a parallel palette."""
    return source_cmap_hex(_SOURCE_DISPLAY_CMAPS[ci % len(_SOURCE_DISPLAY_CMAPS)])


#: Fallback per-channel LUTs for `build_preview_histogram_entries` — used only if
#: a caller passes no `luts` (the real Preview path always passes per-channel
#: canvas colours).  Derived from the ONE hex map above so it can never drift into
#: a second live palette.
_DEFAULT_PREVIEW_LUTS = tuple(source_cmap_hex(c) for c in _SOURCE_DISPLAY_CMAPS)

#: Preview crop caps — a fixed CENTER region keeps the compute fast + honest
#: about being a test region: ≤256×256 in XY, ≤64 in Z.
_PREVIEW_CAP_XY = 256
_PREVIEW_CAP_Z = 64

#: The before/after intensity histogram is plotted in NORMALIZED intensity ``u``
#: (the LINEAR part of percentile-normalize, computed from the RAW crop — see
#: `build_preview_histogram_entries`).  Both curves share ONE fixed bin frame so
#: identical curves ⇔ identical values; the frame runs slightly OUTSIDE ``[0, 1]``
#: so the pixels that percentile-normalize will CLIP (raw < p0.1 → u < 0, raw >
#: p99.5 → u > 1) are visible as before-tails past the 0 / 1 clip rails.  Mass
#: beyond the frame is folded into the two edge (overflow) bins, identically for
#: both curves.
_PREVIEW_HIST_BINS = 256
_PREVIEW_HIST_LO = -0.1
_PREVIEW_HIST_HI = 1.1

#: The percentiles percentile-normalize (execute's non-removable first stage)
#: uses to define the linear map — the SAME values `build_preview_histogram_entries`
#: reuses so the "before" u-axis and the "after" (already clip(u,0,1)) share the
#: exact frame.  Keep in lock-step with `ops.percentile_normalize`'s defaults.
_PREVIEW_NORM_PLO = 0.1
_PREVIEW_NORM_PHI = 99.5

#: One per-channel histogram record handed to the ② plot.  A namedtuple so the
#: widget can read fields by name (`entry.stats`) while old positional unpacking
#: (`ci, name, lut, ...`) still works.  ``stats`` is a dict of factual read-out
#: numbers (p_lo/p_hi in RAW DN, clip fractions, medians in u) — never a derived
#: "quality score".
PreviewHistEntry = namedtuple(
    "PreviewHistEntry",
    ["index", "name", "lut",
     "before_counts", "after_counts", "bin_edges", "stats"],
)

#: Name prefix stamped on every ⟨preview⟩ processed-crop layer (Brick 2c).  ONE
#: source of truth: `_add_preview_layers` builds the layer name from this, and it
#: also lets a reader spot a preview layer by name.  NB: the authoritative
#: exclusion in `_on_layers_changed` is IDENTITY-based (`layer in
#: self._preview_layers`), not this prefix — identity is marker-drift-proof and
#: collision-proof (a real source channel literally named "⟨preview⟩ foo" is
#: still counted).  The marker is display + a human-readable tell only.
PREVIEW_LAYER_MARKER = "⟨preview⟩ "

#: Name prefix stamped on every ⟨processed⟩ WHOLE-VOLUME layer (Brick 1b-1).  The
#: distinct-from-⟨preview⟩ marker keeps the two WYSIWYG renders visually apart:
#: ⟨preview⟩ is a region test-crop (float, additive over the source); ⟨processed⟩
#: is the whole volume at the recipe's fixed-scale uint16 bake.  As with
#: ⟨preview⟩ the authoritative `_on_layers_changed` exclusion is IDENTITY-based
#: (`layer in self._processed_layers`), not this prefix — the marker is display
#: + a human-readable tell only.
PROCESSED_LAYER_MARKER = "⟨processed⟩ "

#: Cap on the STRIDED subsample the whole-volume normalize window is measured
#: over (Brick 1b-1).  `percentile_normalize`'s `(lo, hi)` window is a robust
#: statistic, so a strided ~5M-voxel subsample matches a full pass to well within
#: the percentile's own noise while avoiding a multi-second full percentile sort
#: over a multi-GB array.  See `_strided_percentile_bounds`.
_PROCESS_BOUNDS_SAMPLE_CAP = 5_000_000

#: The preview REGION box overlay colour (RGBA-less RGB; alpha comes from the
#: overlay).  Deliberately DISTINCT from the ③ extract-box palette (idle teal /
#: active amber / green / magenta / white — see `_box_tool._BOX_COLOR_CHOICES`)
#: so the transient preview region is never mistaken for an extract box.  A
#: vivid orange-red sits outside that palette.  (Preview v2 Piece 2.)
_PREVIEW_BOX_COLOR = (1.0, 0.45, 0.15)

#: The ④ patch-LOCATION box overlay's default colour (magenta) — distinct from
#: ③'s transient teal boxes so "where my patches are" reads apart from "boxes
#: I'm drawing".  The Render/style combo's "status colour" keeps this default.
_LOCATION_BOX_RGB = (1.0, 0.25, 0.85)

#: ④ Snapshot (Brick 1).  The popover corner labels → napari `CanvasPosition`
#: values, and the GL max-texture ceiling the export scale factor is clamped to
#: so a high-resolution figure never exceeds the GPU limit.
_SNAPSHOT_POSITIONS = {
    "bottom right": "bottom_right",
    "bottom left": "bottom_left",
    "top right": "top_right",
    "top left": "top_left",
}
_GL_MAX_TEXTURE_SIZE = 8192

#: ④ per-patch box-style popup (P2 Brick 2): the location overlay's built-in
#: default colour + the swatch presets + the popup's dark stylesheet.
_DEFAULT_LOC_RGB = (0.83, 0.35, 0.72)   # magenta
_BOX_STYLE_PRESETS = [
    ("magenta", (0.83, 0.35, 0.72)),
    ("cyan",    (0.30, 0.80, 0.85)),
    ("yellow",  (0.92, 0.86, 0.30)),
    ("green",   (0.40, 0.80, 0.45)),
    ("orange",  (0.92, 0.58, 0.22)),
    ("white",   (0.95, 0.95, 0.95)),
]
_BOX_STYLE_POPUP_STYLE = (
    "QWidget { background-color: #2b2b2b; border: 1px solid rgba(255,255,255,0.15);"
    "  border-radius: 6px; }"
    "QLabel { color: #cccccc; font-size: 10px; background: transparent; border: 0; }"
    "QRadioButton { color: #dddddd; font-size: 10px; background: transparent; }"
    "QToolButton { color: #dddddd; font-size: 10px; }"
)

#: Peak-RAM budget for the PREVIEW region COMPUTE (float32, across all channels).
#: A full-Z box on a deep stack is the OOM class (full-Z ≈ whole-volume-in-Z), so
#: `_preview_compute_stride` strides the crop down to fit rather than ever
#: materializing a multi-GB cube (Preview v2 Piece 2, data-grill MED-6).
_PREVIEW_COMPUTE_BUDGET_BYTES = 512 * 1024 * 1024   # 512 MB


def _preview_center_bounds(shape_zyx, cap_z=_PREVIEW_CAP_Z, cap_xy=_PREVIEW_CAP_XY):
    """A centered (z0,z1,y0,y1,x0,x1) crop of *shape_zyx*, capped per axis.

    Never larger than the array; a small stack is used whole.  Returns a dict of
    slice bounds (the `crop_zyx`/`crop_czyx` bounds shape) so the worker can
    slice + materialize only the ROI.
    """
    z, y, x = (int(s) for s in shape_zyx[:3])
    dz = min(int(cap_z), z)
    dy = min(int(cap_xy), y)
    dx = min(int(cap_xy), x)
    z0 = max(0, (z - dz) // 2)
    y0 = max(0, (y - dy) // 2)
    x0 = max(0, (x - dx) // 2)
    return {"z0": z0, "z1": z0 + dz,
            "y0": y0, "y1": y0 + dy,
            "x0": x0, "x1": x0 + dx}


def _crop_center(arr, bounds):
    """Materialize ONLY the *bounds* crop of a 3D (Z,Y,X) array-like.

    Accepts a numpy ndarray or a lazy (dask) array; `.compute()` realizes only
    the small ROI, never the whole volume.  Honors optional per-axis strides
    (``stride_z``/``stride_y``/``stride_x``, default 1) so an over-budget region
    box is downsampled at READ time (the compute-budget guard); a bounds dict
    without stride keys is byte-identical to the old plain slice.
    """
    b = bounds
    sz = max(1, int(b.get("stride_z", 1) or 1))
    sy = max(1, int(b.get("stride_y", 1) or 1))
    sx = max(1, int(b.get("stride_x", 1) or 1))
    sub = arr[b["z0"]:b["z1"]:sz, b["y0"]:b["y1"]:sy, b["x0"]:b["x1"]:sx]
    if hasattr(sub, "compute"):
        sub = sub.compute()
    return np.ascontiguousarray(np.asarray(sub, dtype=np.float32))


def _preview_compute_stride(bounds, n_channels,
                            budget_bytes=_PREVIEW_COMPUTE_BUDGET_BYTES):
    """The uniform per-axis integer stride that keeps the region crop's peak
    float32 RAM (across *n_channels*) within *budget_bytes*.

    Striding by ``s`` cuts the voxel count by ~``s**3`` for a cubic box, so the
    first estimate is ``ceil((total_bytes / budget)**(1/3))``.  For an extreme
    aspect ratio (one axis ≫ the others) the small axes barely shrink under that
    ``s``, so we then verify the ACTUAL strided voxel count and bump ``s`` until
    it fits — genuinely never over budget for ANY box shape (correctness audit
    LOW-1).  Returns ``1`` when the region already fits (the common small-box
    case → no downsample).  Pure / guarded — never raises.
    """
    try:
        dz = max(1, int(bounds["z1"]) - int(bounds["z0"]))
        dy = max(1, int(bounds["y1"]) - int(bounds["y0"]))
        dx = max(1, int(bounds["x1"]) - int(bounds["x0"]))
    except Exception:
        return 1
    n = max(1, int(n_channels))
    budget = max(1, int(budget_bytes))
    if dz * dy * dx * n * 4 <= budget:
        return 1
    s = max(2, int(math.ceil((dz * dy * dx * n * 4 / budget) ** (1.0 / 3.0))))
    # The cube-root assumes all three axes shrink by ~s; verify the real strided
    # voxel count and bump until it fits (bounded: at s = max(dz,dy,dx) every
    # axis is a single voxel → n*4 bytes, always ≤ budget).
    hi = max(dz, dy, dx)
    while s < hi and (math.ceil(dz / s) * math.ceil(dy / s)
                      * math.ceil(dx / s) * n * 4) > budget:
        s += 1
    return s


def _compute_preview(recipe, crops, spacing, bounds_per_channel=None):
    """Run the recipe over the test *crops* and return (befores, afters, spacing).

    *bounds_per_channel* — the per-channel ``(lo, hi)`` normalize window measured
    over the WHOLE channel (`_whole_volume_norm_bounds`), threaded straight into
    ``execute``.  Without it the preview normalizes against the CROP's own
    p0.1/p99.5 while "Apply → whole volume" uses the source's, so the preview is
    systematically brighter or flatter than the pixels the button writes — the
    preview is not WYSIWYG (C#2).  ``None`` keeps the old crop-local behaviour
    and is only for a caller that genuinely has no source (tests).

    ``befores[ci]`` = the RAW crop (float32, un-normalized) — the honest "before"
    reference.  The histogram plots it through the LINEAR half of
    percentile-normalize (``u``, UNCLIPPED) so the clip that execute's normalize
    applies is VISIBLE, not baked into the baseline (the pre-2026-08 bug drew
    ``percentile_normalize(crop)`` as "before", sharing execute's normalization
    and hiding the single biggest transform).  ``afters`` = the `execute` output
    (float32 in ``[0, 1]`` — normalize's output IS ``clip(u, 0, 1)``), which
    already lives in the ``u`` frame and is plotted directly.  Pure — no Qt, no
    I/O.  Referenced through the module globals so a test can monkeypatch execute.
    """
    befores = [np.asarray(c, dtype=np.float32) for c in crops]
    afters, new_spacing, _stamp = _pp_execute(
        recipe, crops, spacing, bounds_per_channel=bounds_per_channel)
    return befores, afters, new_spacing


def _materialize_channel(ch):
    """Fully realize a possibly-lazy channel handle as an in-memory ndarray.

    Whole-volume process (Brick 1b-1) input channels may be dask / lazy-zarr
    handles: `.compute()` when available, else `np.asarray`.  Heavy READ — this
    MUST run on a worker thread (its callers, `_process_whole_volume_core` and
    `_whole_volume_norm_bounds`, run only inside `_PreprocProcessWorker.run` /
    `_PreprocPreviewWorker.run`), NEVER the main thread.  Pure — no Qt, no I/O beyond the array realization.
    """
    if hasattr(ch, "compute"):
        try:
            ch = ch.compute()
        except Exception:
            pass
    return np.asarray(ch)


def _strided_percentile_bounds(arr, sample_cap=_PROCESS_BOUNDS_SAMPLE_CAP):
    """The `(lo, hi)` percentile-normalize window from a STRIDED subsample.

    Delegates to `ops.percentile_bounds` (the SINGLE source of truth shared with
    `percentile_normalize`) so the whole-volume window can never drift off what
    normalize actually clips — but measures it over a flattened subsample capped
    at ~`sample_cap` voxels rather than a full percentile pass over a multi-GB
    array.  Returns `None` (like `percentile_bounds`) for a flat / empty /
    degenerate channel so `execute` falls back to that channel's own percentiles.
    Pure — no Qt, no I/O.
    """
    a = np.asarray(arr)
    flat = a.reshape(-1)
    n = int(flat.size)
    cap = int(sample_cap) if sample_cap else 0
    if cap > 0 and n > cap:
        stride = int(math.ceil(n / float(cap)))
        flat = flat[::stride]
    return _pp_percentile_bounds(flat)


def _whole_volume_norm_bounds(channels, sample_cap=_PROCESS_BOUNDS_SAMPLE_CAP):
    """Per-channel ``(lo, hi)`` normalize window over the WHOLE channels — the
    SAME window "Apply → whole volume" uses, so a region preview agrees with it
    (C#2).

    For an already-in-memory C-CONTIGUOUS ndarray this is literally the apply
    path's `_strided_percentile_bounds`, so the two windows are identical rather
    than merely similar.  Two kinds of channel take a per-axis strided SLICE
    first, so the read stays bounded to ~`sample_cap` voxels instead of pulling
    the whole volume through RAM (the preview must not become a whole-volume
    read):

      * a lazy handle (dask / zarr);
      * a NON-CONTIGUOUS ndarray — a transposed memmap, which is exactly what an
        axis-reordered TIFF gives you.  `_strided_percentile_bounds` flattens
        with ``reshape(-1)``, and on such an array that is a full copy of the
        volume.  The apply-path helper is deliberately left alone (it runs on
        materialized, contiguous arrays); only the preview's window measurement
        reroutes.  The subsample is per-axis rather than flat, so the window can
        differ from a full pass by percentile-sampling noise — acceptable for a
        robust p0.1/p99.5, and far better than a multi-GB copy on the click.

    HEAVY (a percentile pass, and for a lazy source an actual read): worker
    thread only.  Never raises: a channel that cannot be measured
    contributes ``None``, which `execute` reads as "fall back to this array's own
    percentiles" — i.e. exactly today's behaviour for that channel.
    Pure — no Qt, no controller state.
    """
    out = []
    for ch in list(channels or []):
        try:
            arr = ch
            contiguous = (isinstance(arr, np.ndarray)
                          and bool(getattr(arr.flags, "c_contiguous", False)))
            if not contiguous:
                shape = tuple(int(v) for v in (getattr(arr, "shape", ()) or ()))
                n = int(np.prod(shape)) if shape else 0
                cap = int(sample_cap) if sample_cap else 0
                if cap > 0 and n > cap and len(shape) >= 1:
                    step = int(math.ceil((float(n) / cap) ** (1.0 / len(shape))))
                    step = max(1, step)
                    arr = arr[tuple(slice(None, None, step) for _ in shape)]
                arr = _materialize_channel(arr)
            out.append(_strided_percentile_bounds(arr, sample_cap))
        except Exception:
            out.append(None)
    return out


#: The two ops whose result on a test REGION is not the result on the whole
#: volume for a structural reason, not a numeric one (C#3 / C#4):
#:   * `bleach` fits an exponential decay across the array's Z — over a cropped Z
#:     it fits a different curve, and below 2 planes it silently does nothing
#:     while the whole-volume apply is fully active (`ops.bleach`);
#:   * `clahe` clamps its tile size per axis to the array's dimensions, so a
#:     kernel bigger than the crop becomes ONE tile on that axis — nothing like
#:     the tiling the whole volume gets (`ops.clahe`).
#: We do NOT quietly clamp and we do NOT change the whole-volume numerics: the
#: preview says which op cannot be shown faithfully here, and on which axis.
#: The two classes the per-op notes do NOT cover, so one sentence always says
#: them: the region's EDGES (a local op reflect-pads at the crop face — C#5,
#: halo) and a resize step (a re-grid over a crop is not the whole volume's
#: re-grid).  Always shown on the success line.
_PREVIEW_EDGE_CAVEAT = (
    "Edges of the region and any resize step still differ on the whole volume.")

#: Plain language throughout — the reader is a biologist choosing a recipe, not
#: someone debugging a filter (help text is written for biologists: plain words, no jargon).
_PREVIEW_UNFAITHFUL_NOT_PREVIEWABLE = (
    "this region is too small to show this step honestly — the whole volume "
    "will look different")


def _preview_fidelity_notes(recipe_json, n_channels, crop_shape, full_shape):
    """Human sentences naming every op in *recipe_json* that CANNOT be previewed
    faithfully at this region size (C#3 / C#4).  Empty list = the region is a
    faithful test of every op in the recipe.

    *crop_shape* / *full_shape* are ``(Z, Y, X)``.  Pure — no Qt, no controller
    state.  Never raises: a corrupt recipe or a missing shape yields no notes
    (the preview is still shown; the apply gate remains the authority)."""
    notes = []
    try:
        crop = tuple(int(v) for v in (crop_shape or ()))
        full = tuple(int(v) for v in (full_shape or ()))
        if len(crop) != 3 or len(full) != 3:
            return []
        norm = _pp_recipe.normalize_recipe(recipe_json, int(n_channels))
        keys_seen = set()
        for ci in range(int(n_channels)):
            for step in _pp_recipe.flatten_for_channel(norm, ci):
                key = str(getattr(step, "key", "") or "")
                params = dict(getattr(step, "params", {}) or {})
                if key == "bleach" and "bleach" not in keys_seen:
                    if crop[0] < 2:
                        keys_seen.add("bleach")
                        notes.append(
                            "Depth-brightness needs more than one slice, so it "
                            "does nothing on this region — but it IS applied to "
                            "the whole volume: "
                            f"{_PREVIEW_UNFAITHFUL_NOT_PREVIEWABLE}.")
                    elif crop[0] < full[0]:
                        keys_seen.add("bleach")
                        notes.append(
                            "Depth-brightness is measured across the slices it "
                            f"can see — {crop[0]} here, {full[0]} in the whole "
                            "stack — so the correction differs: "
                            f"{_PREVIEW_UNFAITHFUL_NOT_PREVIEWABLE}.")
                elif key == "clahe" and "clahe" not in keys_seen:
                    try:
                        kernel = int(params.get("kernel", 64))
                    except (TypeError, ValueError):
                        kernel = 64
                    axes = [nm for nm, dim in zip("ZYX", crop) if dim < kernel]
                    if axes:
                        keys_seen.add("clahe")
                        notes.append(
                            f"The contrast step works in {kernel}-pixel tiles, "
                            "but the region is thinner than that in "
                            f"{' and '.join(axes)}, so it is treated as a "
                            "single tile here: "
                            f"{_PREVIEW_UNFAITHFUL_NOT_PREVIEWABLE}.")
    except Exception:  # noqa: BLE001 — honesty is best-effort, never fatal
        return notes
    return notes


def _recipe_geometry_volume_factor(recipe_json, spacing):
    """The output/input VOLUME ratio a recipe's shared geometry step imposes
    (``1.0`` if none / intensity-only), clamped to ``>= 1``.

    Upsampling geometry (factor > 1 per axis — e.g. an isotropic rescale of an
    anisotropic confocal stack: Z 5 µm → 1 µm target = 5× Z) grows ``execute``'s
    intermediate + output arrays F× the input, so the too-large guard scales the
    input-sized peak estimate by this so an upsample can't slip past and OOM.
    Downsampling (factor < 1) only shrinks the output — the input is still
    materialized, already counted — so we clamp to ``1.0`` (never shrink the
    guard).  Mirrors ``ops._resolve_factors``' iso/aniso math on the recipe JSON;
    tolerant + never raises (unknown shape → ``1.0``).
    """
    try:
        geom = recipe_json.get("geometry") if isinstance(recipe_json, dict) else None
        if not geom:
            return 1.0
        config = geom.get("config") or {}
        sp = tuple(float(s) for s in (spacing or (1.0, 1.0, 1.0)))
        if str(config.get("mode", "iso")).lower() == "aniso":
            factors = config.get("factors") or (1.0, 1.0, 1.0)
            fs = [float(f) for f in factors]
        else:
            target = float(config.get("target_um", 1.0)) or 1.0
            fs = [(s / target) if s > 0 else 1.0 for s in sp[-3:]]
        vol = 1.0
        for f in fs:
            vol *= max(float(f), 1e-6)
        return max(1.0, vol)
    except (TypeError, ValueError, AttributeError):
        return 1.0


#: Per-channel byte cap under which the processed result is shown SINGLE-SCALE
#: (full 3D detail).  Above it, a display pyramid is built so napari's 3D-volume
#: view samples a COARSE level instead of pushing the whole array to the GPU (the
#: 8GB-VRAM crash on a full-res processed volume).  A CONSERVATIVE fixed floor
#: (~256 MB/channel) — safe on a 2GB card even with two channels visible; a later
#: refinement raises it from a best-effort VRAM read.  Single-scale keeps FULL 3D
#: detail on small / downsampled data (the common case); the pyramid only kicks in
#: when a full-res 3D render would actually be unsafe (UX-expert conditional-(B)).
_PROCESSED_SINGLE_SCALE_CAP_BYTES = 256 * 1024 * 1024


def _coarsen_uint16(arr, min_axis=32):
    """Block-mean coarsen a 3D uint16 array by 2 per axis (trim-excess), skipping
    any axis that would fall below *min_axis* (mirrors ``multiscale.py``'s per-axis
    rule so an axis never vanishes).  Returns ``None`` when no axis qualifies (the
    array is already at the floor).  Pure numpy — runs on the WORKER, never the
    main thread.
    """
    shape = tuple(int(s) for s in arr.shape)
    factors = [2 if (s // 2) >= int(min_axis) else 1 for s in shape]
    if all(f == 1 for f in factors):
        return None
    trimmed = tuple(slice(0, s - (s % f)) for s, f in zip(shape, factors))
    a = np.asarray(arr[trimmed], dtype=np.float32)
    new_shape = []
    for s, f in zip(a.shape, factors):
        new_shape += [s // f, f]
    a = a.reshape(new_shape).mean(axis=(1, 3, 5))
    return np.clip(np.round(a), 0.0, 65535.0).astype(np.uint16)


def _build_display_levels(arr, cap_bytes=_PROCESSED_SINGLE_SCALE_CAP_BYTES,
                          max_levels=16):
    """A display pyramid for one processed uint16 channel (conditional).

    Returns ``[arr]`` (SINGLE-SCALE, full 3D detail) when the full-res channel
    already fits *cap_bytes*; otherwise ``[arr, coarser, …, coarsest]`` coarsened
    (block-mean ×2/axis) until the coarsest level is under the cap (or the per-axis
    floor stops it).  Level 0 is ALWAYS the full-res array (the cache + 2D full
    detail read it; napari's 3D-volume view uploads only the COARSEST level, which
    is ≤ cap by construction → safe).  ``max_levels`` is a pure runaway backstop
    (16 → an all-axes-large volume needing >16 halvings would be astronomically
    bigger than anything the RAM guard admits, so the cap is reached or the axis
    floor stops it long before) — NOT the normal terminator, so the
    "coarsest ≤ cap" guarantee is unconditional in every realistic case.  Pure —
    built on the WORKER.
    """
    levels = [arr]
    cur = arr
    while (int(cur.size) * 2) > int(cap_bytes) and len(levels) < int(max_levels):
        nxt = _coarsen_uint16(cur)
        if nxt is None:
            break
        levels.append(nxt)
        cur = nxt
    return levels


def _process_whole_volume_core(recipe_json, channels, spacing,
                               bounds_sample_cap=_PROCESS_BOUNDS_SAMPLE_CAP,
                               *, parallel=False, should_cancel=None,
                               block_workers=None, on_progress=None):
    """Run *recipe_json* over the WHOLE per-channel volumes → uint16 (Brick 1b-1).

    The pure, Qt-free heart of the "Process whole volume" button (mirrors
    `_compute_preview` for the region path).  For each channel: materialize the
    (possibly lazy) handle, measure the fixed normalize window `(lo, hi)` from a
    STRIDED subsample (`_strided_percentile_bounds`), then run `ops.execute` with
    `max_workers=1` (REQUIRED: runs the CHANNELS' heavy ops SERIALLY so only ONE
    channel's op-scratch is live at a time — never N channels' scratch
    concurrently — bounding peak RAM on a large volume; the N result arrays still
    accumulate, and the too-large guard counts them) + `parallel=True`
    (the LOCAL ops — gaussian, background — run BLOCK-parallel across all cores via
    `chunked.block_parallel_apply`, BYTE-IDENTICAL to whole-array; the multi-core
    speed win.  Block-parallel per op + serial channels = full core use with no
    oversubscription; CLAHE/bleach/normalize/geometry stay whole-array), and finally
    bake each processed float32 `[0, 1]` channel to fixed-scale uint16
    (`ops.to_uint16` — `0.0→0`, `1.0→65535`, so a uniformly dim channel STAYS dim;
    it is NOT per-channel min-max, which would re-stretch and undo dimming ops).

    Returns `(processed_uint16_list, new_spacing, bounds, stamp)`:
      * `processed_uint16_list` — one uint16 ndarray per input channel,
        spatially co-registered (shared geometry re-grids every channel alike);
      * `new_spacing` — the spacing after any geometry op (== input spacing if
        no geometry step changed the grid);
      * `bounds` — the per-channel `(lo, hi)` windows actually threaded into
        normalize (or `None` for a degenerate channel);
      * `stamp` — `execute`'s recipe/dtype stamp.

    Referenced through the module globals (`_pp_execute` / `_pp_percentile_bounds`
    / `_pp_to_uint16`) so a test can monkeypatch `execute`.  Pure — no Qt, no I/O.
    """
    mats = [_materialize_channel(ch) for ch in channels]
    bounds = [_strided_percentile_bounds(m, bounds_sample_cap) for m in mats]
    # `parallel` defaults to False (the proven serial path).  When True, the LOCAL
    # ops (gaussian, background) run BLOCK-parallel across all cores, BYTE-IDENTICAL
    # to serial (`chunked.block_parallel_apply`); CHANNELS still run serially
    # (`max_workers=1`, memory-bounded).  `should_cancel` is polled between tiles /
    # ops so Cancel stops the run promptly (it raises `PreprocessingCancelled`, a
    # BaseException, which the worker turns into a clean "cancelled").
    # `block_workers` caps the block pool (the RAM-adaptive core count).
    processed_float, new_spacing, stamp = _pp_execute(
        recipe_json, mats, spacing, max_workers=1, bounds_per_channel=bounds,
        parallel=parallel, should_cancel=should_cancel,
        block_workers=block_workers, on_progress=on_progress)
    processed_uint16 = [_pp_to_uint16(p) for p in processed_float]
    return processed_uint16, new_spacing, bounds, stamp


def _raw_percentile_bounds(raw, low=_PREVIEW_NORM_PLO, high=_PREVIEW_NORM_PHI):
    """``(p_lo, p_hi)`` in RAW DN — the linear map percentile-normalize applies.

    Delegates to ``ops.percentile_bounds`` (the SINGLE source of truth shared
    with ``percentile_normalize``), so the histogram's ``u`` frame can never
    drift off the clip ``execute`` actually performed.  Returns ``None`` for a
    flat / empty / degenerate crop (``p_hi <= p_lo`` or non-finite) so the
    caller can fall back without dividing by zero.
    """
    return _pp_percentile_bounds(np.asarray(raw, dtype=np.float32), low, high)


#: Density scale for the ② histogram curves — counts are divided by the curve's
#: OWN voxel count and multiplied by this, i.e. "voxels per million".  A bare
#: fraction would collapse the plot's log-Y toggle (log1p of a number < 1 is
#: ~the number itself); per-million keeps the magnitudes count-like.
_PREVIEW_HIST_DENSITY_SCALE = 1.0e6


def _as_density(counts, n_voxels):
    """*counts* rescaled to voxels-per-million so two curves over DIFFERENT voxel
    counts are comparable (C#7).  An empty curve stays all-zero — never a
    div-by-zero.  Pure."""
    arr = np.asarray(counts, dtype=np.float64)
    n = int(n_voxels)
    if n <= 0:
        return np.zeros_like(arr)
    return arr * (_PREVIEW_HIST_DENSITY_SCALE / float(n))


def build_preview_histogram_entries(names, luts, befores, afters,
                                    bins=_PREVIEW_HIST_BINS,
                                    bounds_per_channel=None):
    """Per-channel :class:`PreviewHistEntry` (before/after counts + shared edges +
    a factual stat dict), in NORMALIZED intensity ``u``.

    The one explicit coordinate (the correctness core): percentile-normalize is a
    linear map then a clip.  Define the plot's x-coordinate as the LINEAR part
    only, computed once from the RAW crop, per channel::

        p_lo, p_hi = bounds_per_channel[ci]               # what normalize used
        u(v)       = (v - p_lo) / (p_hi - p_lo)          # linear, NOT clipped

    *bounds_per_channel* is the WHOLE-channel window (`_whole_volume_norm_bounds`)
    that `execute` was actually given.  It must be the same window, or the plot
    lies twice over: `after` is drawn on the axis normalize really produced while
    `before` (and the "% clipped" readout) is drawn on the crop's own — so a
    perfectly matched pair looks shifted (C#2).  ``None``, or a ``None`` entry,
    falls back to that crop's own percentiles, which is exactly what `execute`
    does for the same channel — the two stay in step either way.

    - ``before`` = histogram of ``u(raw_crop)``, UNCLIPPED (raw pixels below
      ``p_lo`` land at ``u < 0``, above ``p_hi`` at ``u > 1`` — the pixels
      normalize will clip).
    - ``after``  = histogram of the ``execute`` output, plotted directly on the
      ``u`` axis (normalize's output IS ``clip(u, 0, 1)``; downstream ops stay in
      ``[0, 1]``) — NOT re-mapped.

    Identical curves ⇔ values truly unchanged (the "does nothing" bug is now
    structurally impossible); the clip shows as before-tails past 0 / 1 truncating
    into after's pile-up at the rails.  Both curves use ONE fixed bin frame
    (``bins`` over ``[_PREVIEW_HIST_LO, _PREVIEW_HIST_HI]``), with mass beyond the
    frame folded into the two edge bins IDENTICALLY.  Pure / Qt-free.

    Degenerate crop (``p_hi <= p_lo``): fall back to plotting the raw crop min-max
    scaled (never a div-by-zero); ``after`` is plotted verbatim.
    """
    luts = tuple(luts) or _DEFAULT_PREVIEW_LUTS
    lo_edge, hi_edge = float(_PREVIEW_HIST_LO), float(_PREVIEW_HIST_HI)
    bin_edges = np.linspace(lo_edge, hi_edge, int(bins) + 1)
    entries = []
    n = min(len(names), len(befores), len(afters))
    for ci in range(n):
        raw = np.asarray(befores[ci], dtype=np.float32).ravel()
        raw = raw[np.isfinite(raw)]
        after = np.asarray(afters[ci], dtype=np.float32).ravel()
        after = after[np.isfinite(after)]

        bounds = None
        if bounds_per_channel is not None and ci < len(bounds_per_channel):
            b = bounds_per_channel[ci]
            if b is not None:
                try:
                    lo, hi = float(b[0]), float(b[1])
                except (TypeError, ValueError, IndexError):
                    lo = hi = 0.0
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    bounds = (lo, hi)
        if bounds is None:
            # No shared window for this channel — `execute` fell back to this
            # array's own percentiles, so the plot must too (stay in step).
            bounds = _raw_percentile_bounds(raw)
        if bounds is None:
            # Flat / degenerate crop — no percentile spread to stretch.  This is
            # EXACTLY the case where `ops.percentile_normalize` hits its own
            # degenerate branch and collapses the channel to all-zeros, so the
            # readout must be flagged honest (the p0.1/p99.5 + "0% clipped" line
            # would be a lie here — see `_format_recipe_stat`).  Fall back to a
            # min-max frame so we never divide by zero.
            degenerate = True
            if raw.size:
                p_lo = float(raw.min())
                p_hi = float(raw.max())
            else:
                p_lo = p_hi = 0.0
            span = (p_hi - p_lo) if (p_hi > p_lo) else 1.0
            u = (raw - p_lo) / span
            pct_low = 0.0
            pct_high = 0.0
        else:
            degenerate = False
            p_lo, p_hi = bounds
            u = (raw - p_lo) / (p_hi - p_lo)     # linear, UNCLIPPED
            pct_low = float(np.mean(u < 0.0) * 100.0) if u.size else 0.0
            pct_high = float(np.mean(u > 1.0) * 100.0) if u.size else 0.0

        # Shared fixed bins; fold overflow into the edge bins by clamping into the
        # frame BEFORE binning — identically for both curves.
        before_counts, _ = np.histogram(
            np.clip(u, lo_edge, hi_edge), bins=bin_edges)
        after_counts, _ = np.histogram(
            np.clip(after, lo_edge, hi_edge), bins=bin_edges)
        # C#7: a geometry (resample) step changes the voxel COUNT, so raw counts
        # put the two curves on DIFFERENT scales — the processed one reads as
        # uniformly lower (or taller on an upsample), which looks like removed
        # voxels and is not.  Normalize each curve by its OWN voxel count.
        # Expressed per MILLION voxels rather than as a bare fraction so the
        # magnitudes stay count-like and the plot's log-Y toggle stays useful.
        before_counts = _as_density(before_counts, u.size)
        after_counts = _as_density(after_counts, after.size)

        stats = {
            "degenerate": degenerate,
            "p_lo": p_lo,
            "p_hi": p_hi,
            "pct_clipped_low": pct_low,
            "pct_clipped_high": pct_high,
            "median_before_u": float(np.median(u)) if u.size else 0.0,
            "median_after_u": float(np.median(after)) if after.size else 0.0,
        }
        lut = luts[ci % len(luts)]
        entries.append(PreviewHistEntry(
            ci, names[ci], lut, before_counts, after_counts, bin_edges, stats))
    return entries


class _PreprocPreviewWorker(QObject):
    """Runs the Step-1 preprocessing PREVIEW off the main thread (Brick 2b).

    Crops each channel to the shared center test region, runs the recipe
    (`_compute_preview` → `execute`), AND builds the per-channel before/after
    histogram entries (`build_preview_histogram_entries` — pure numpy percentile
    + histogram, no Qt).  All three are off the main thread: measurement showed
    the histogram-entries build alone is ~0.5 s/channel (percentile + medians +
    two histograms over the full crop), which used to run in `_on_preview_computed`
    on the UI thread and was the dominant Preview freeze (`[PREVIEW-PERF]`).

    Emits ``(entries, afters, hist_ms)`` — the small per-channel entries (256-bin
    count arrays + a stats dict) plus the processed ``afters`` crops the main
    thread still needs to render the ⟨preview⟩ layers.  The big raw ``befores``
    arrays are consumed HERE and NOT sent across the thread boundary (net data
    reduction).  Only the unavoidable render (histogram DRAW + napari layer-add)
    stays on the main thread in `_on_preview_computed`.
    """

    #: (entries list, afters list, hist_build_ms) — see class docstring.
    finished = Signal(object, object, float)
    error = Signal(str)

    def __init__(self, recipe_json, channels, bounds_list, spacing,
                 names, luts, norm_bounds=None):
        super().__init__()
        self._recipe_json = recipe_json
        self._channels = channels
        self._bounds_list = bounds_list
        self._spacing = spacing
        self._names = names
        self._luts = luts
        #: per-channel `(lo, hi)` normalize window over the WHOLE channels — the
        #: window "Apply → whole volume" uses (C#2).  `None` = not cached, so
        #: `run` measures it HERE, on this thread (never the UI thread).
        self._norm_bounds = norm_bounds
        self._cancel_requested = False

    def request_cancel(self):
        self._cancel_requested = True

    def run(self):
        try:
            crops = [
                _crop_center(arr, bounds)
                for arr, bounds in zip(self._channels, self._bounds_list)
            ]
            # WYSIWYG (C#2): normalize against the SAME whole-channel window the
            # apply path uses, not the crop's own percentiles.  Measured here
            # when the controller had none cached — a percentile pass (and, for
            # a lazy source, a bounded read) that belongs on this thread.
            norm_bounds = self._norm_bounds
            if norm_bounds is None:
                norm_bounds = _whole_volume_norm_bounds(self._channels)
            if self._cancel_requested:
                self.error.emit("cancelled")
                return
            befores, afters, _spacing = _compute_preview(
                self._recipe_json, crops, self._spacing,
                bounds_per_channel=norm_bounds)
            # Build the histograms HERE (off the main thread).  Byte-identical to
            # the old main-thread call — same function, same (names, luts,
            # befores, afters).  `befores` is consumed here + dropped; only the
            # small `entries` + the processed `afters` cross the thread boundary.
            _t0 = time.perf_counter()
            entries = build_preview_histogram_entries(
                self._names, self._luts, befores, afters,
                bounds_per_channel=norm_bounds)
            hist_ms = (time.perf_counter() - _t0) * 1e3
            if self._cancel_requested:
                # A superseded run: emit nothing extra, but a terminal signal is
                # still required so `thread.quit` fires + the pair is not leaked.
                self.error.emit("cancelled")
                return
            self.finished.emit(entries, afters, hist_ms)
        except BaseException as exc:   # noqa: BLE001
            # A worker thread must ALWAYS emit a terminal signal (even on a
            # non-Exception) so `thread.quit` fires and the pair is never leaked.
            self.error.emit(f"{exc}")


class _PreprocProcessWorker(QObject):
    """Runs the Step-1 WHOLE-VOLUME preprocessing process off the main thread
    (Brick 1b-1).

    Mirrors :class:`_PreprocPreviewWorker`, but on the WHOLE per-channel volumes
    (no region crop) and producing the fixed-scale uint16 bake: it materializes
    each (possibly lazy) channel, measures the normalize window from a strided
    subsample, runs `execute` serially (`max_workers=1`, memory-bounded) and bakes
    to uint16 — all in `_process_whole_volume_core`.  The whole-volume READ +
    `execute` + `to_uint16` are exactly the heavy work that must stay off the
    main thread.

    Emits ``(channel_levels, new_spacing, bounds, stamp)`` — ``channel_levels`` is
    a per-channel DISPLAY PYRAMID (level 0 = the full-res uint16 result; a
    single-element list when the channel is small enough to render single-scale).
    The main thread renders each pyramid as a ⟨processed⟩ layer (multiscale when
    coarsened) + caches level 0, plus the spacing/bounds/stamp metadata.
    """

    #: (channel_levels: list-per-channel of [uint16 level0, …coarser], new_spacing,
    #: bounds list, stamp dict).  Built on the worker so the pyramid coarsen is off
    #: the main thread.
    finished = Signal(object, object, object, object)
    error = Signal(str)
    #: (fraction 0..1, plain step label, indeterminate) — cost-weighted progress,
    #: emitted from the worker thread + delivered queued to the main-thread slot.
    progress = Signal(float, str, bool)

    def __init__(self, recipe_json, channels, spacing, *, parallel=False,
                 block_workers=None, names=None):
        super().__init__()
        self._recipe_json = recipe_json
        self._channels = channels
        self._spacing = spacing
        self._parallel = parallel
        self._block_workers = block_workers
        self._names = names
        self._cancel_requested = False

    def request_cancel(self):
        # Set from the MAIN thread; read by the worker's `should_cancel` closure.
        # A plain bool read/write across threads is safe under CPython's GIL (py312).
        self._cancel_requested = True

    def run(self):
        try:
            import time as _time
            voxels = []
            for c in self._channels:
                shp = getattr(c, "shape", None)
                voxels.append(int(math.prod(int(s) for s in shp)) if shp else 0)
            # In serial mode the tiled ops run whole-array (single-core), so cost
            # them at workers=1 — dividing by cores would understate the slowest
            # single-core op and drop its "slow step" label (grill LOW / L1).
            workers = ((self._block_workers or min(os.cpu_count() or 1, 32))
                       if self._parallel else 1)
            agg = _ProcessProgressAggregator(
                voxels, self._recipe_json, workers,
                self.progress.emit, _time.monotonic,
                channel_names=self._names)
            processed, new_spacing, bounds, stamp = _process_whole_volume_core(
                self._recipe_json, self._channels, self._spacing,
                parallel=self._parallel,
                should_cancel=lambda: self._cancel_requested,
                block_workers=self._block_workers,
                on_progress=agg.feed)
            # Build the per-channel DISPLAY pyramid on the WORKER (conditional-(B)):
            # single-scale [arr] when the channel is small (full 3D detail), else a
            # coarse pyramid [arr, …, coarsest≤cap] so napari's 3D view samples a
            # SAFE coarse level instead of pushing a full-res volume to the GPU.
            # Level 0 is always the full-res result (the cache + 2D read it).
            channel_levels = [_build_display_levels(p) for p in processed]
            if self._cancel_requested:
                # Cancel arrived during an UNINTERRUPTIBLE tail (the display-pyramid
                # build, or a whole-array op that returned normally): emit a terminal
                # signal so `thread.quit` fires + the pair isn't leaked; the token
                # guard (or this "cancelled" sentinel) discards the stale result.
                self.error.emit("cancelled")
                return
            self.finished.emit(channel_levels, new_spacing, bounds, stamp)
        except PreprocessingCancelled:
            # Cooperative cancel reached the compute (raised between tiles/ops).
            # `str(PreprocessingCancelled())` is EMPTY, so emit the literal
            # "cancelled" — otherwise `_on_process_error` would render
            # "Process failed: " (grill B4).  MUST precede the generic handler.
            self.error.emit("cancelled")
        except BaseException as exc:   # noqa: BLE001
            # A worker thread must ALWAYS emit a terminal signal (even on a
            # non-Exception) so `thread.quit` fires and the pair is never leaked.
            self.error.emit(f"{exc}")


class _ZarrChannelView:
    """A 3D ZYX direct-read VIEW of channel *ci* of a CZYX zarr array.

    Lets the streaming engine read ``view[zyx_slice]`` DIRECTLY off the underlying
    zarr — one channel's block, materialized on demand — instead of computing a
    fresh dask graph per block (~1.6× slower).  The engine's block reader only
    needs ``.shape`` (the ZYX shape) + ``__getitem__(zyx_slice)``; a ZYX
    single-channel store passes its raw zarr array straight through (no view
    needed).  Pure numpy/zarr — no Qt.

    ``chunks`` (the underlying store's ZYX chunk shape, channel axis dropped) is
    part of the contract even though the block reader does not use it: the
    streaming engine's ``_source_chunk_zyx`` reads it so a PROCESSED store
    inherits the layout of the PREPARED store it was streamed from.  Without it
    the view looked chunk-less and the engine fell back to the default shape —
    so the processed store's layout silently depended on the CHANNEL COUNT (a
    1-channel source inherited correctly; a 2-channel one did not).
    """

    __slots__ = ("_z", "_ci", "shape", "chunks")

    def __init__(self, z, ci):
        self._z = z
        self._ci = int(ci)
        self.shape = tuple(int(s) for s in z.shape[1:])
        # ZYX only — the channel axis is chunked at 1 and is not part of the
        # shape the engine reasons about.  Defensive: a duck-array source
        # without `.chunks` leaves this None, which `_source_chunk_zyx` treats
        # as "unknown" and falls back exactly as before.
        _ck = getattr(z, "chunks", None)
        self.chunks = (tuple(int(c) for c in _ck[1:])
                       if _ck is not None and len(_ck) == len(z.shape)
                       else None)

    def __getitem__(self, sl):
        if isinstance(sl, tuple):
            return self._z[(self._ci,) + sl]
        return self._z[(self._ci, sl)]


class _PreprocProcessToStoreWorker(QObject):
    """Runs the STREAMING (out-of-core) whole-volume process off the main thread
    (Phase 2a).

    Mirrors :class:`_LargeDataPrepWorker`: the heavy work (reading each channel
    block-by-block, running the recipe's fused float32 run per block, and writing
    the uint16 OME-Zarr store) runs entirely here on a QThread; ALL viewer / layer
    work (opening the finished store as a lazy multiscale array + adding the
    ⟨processed⟩ display layers) happens back on the main thread in the controller's
    :meth:`DataPreparationController._on_process_store_ready` handler.

    The engine (``stream_process_to_store``, referenced through the module global
    ``_pp_stream_to_store`` so a test can monkeypatch it) is COOPERATIVELY
    cancellable — it polls ``should_cancel`` per block wave, raises
    :class:`PreprocessingCancelled`, and removes its own partial ``.tmp`` store —
    so Cancel stops the run promptly and leaves NO half-store on disk (unlike the
    non-interruptible in-memory convert).  ``_channels`` arrive already resolved to
    direct-read ZYX handles (a raw zarr / a :class:`_ZarrChannelView` / a dask
    fallback) by the controller's ``_resolve_direct_stream_channels`` (which has
    the store path); the engine reads each block off them.
    """

    #: (out_store_path, manifest dict) — see class docstring.  ``object`` for both
    #: so the arbitrary manifest dict crosses the thread boundary intact.
    finished = Signal(object, object)
    error = Signal(str)
    #: (done_blocks, total_blocks) — determinate streaming progress, emitted from
    #: the worker thread + delivered queued to the main-thread slot.
    progress = Signal(int, int)

    def __init__(self, recipe_json, channels, spacing, out_store_path, *,
                 block_workers=None, source_fingerprint=None,
                 chunk=None, downsample=None, min_size=None):
        super().__init__()
        self._recipe_json = recipe_json
        self._channels = channels
        self._spacing = spacing
        self._out_store_path = out_store_path
        self._block_workers = block_workers
        self._source_fingerprint = source_fingerprint
        # STRUCTURE parity with the raw prepared store: the source's Prepare
        # settings (chunk / downsample / min_size) so the processed store's chunk
        # shape + pyramid levels match the raw convert's (was hardcoded 64/2/256).
        # None → the engine default (which equals the raw convert default).
        self._chunk = chunk
        self._downsample = downsample
        self._min_size = min_size
        self._cancel_requested = False

    def request_cancel(self):
        # Set from the MAIN thread; read by the engine's `should_cancel` closure.
        # A plain bool read/write across threads is safe under CPython's GIL (py312).
        self._cancel_requested = True

    def run(self):
        try:
            if self._cancel_requested:
                self.error.emit("cancelled")
                return
            # Forward the source's Prepare structure settings when set; a None
            # is OMITTED so the engine keeps its default (== the raw convert
            # default) rather than being overridden with None.
            _struct = {}
            if self._chunk is not None:
                # May be a cubic int OR a (cz, cy, cx) SHAPE since Brick 4b --
                # `int()` on the tuple raised TypeError inside run(), which would
                # have surfaced as an opaque failed Process rather than anything
                # legible.  The engine accepts either form, so pass the shape
                # through and only coerce the scalar case.
                _struct["chunk"] = (
                    int(self._chunk)
                    if isinstance(self._chunk, int)
                    and not isinstance(self._chunk, bool)
                    else tuple(int(c) for c in self._chunk))
            if self._downsample is not None:
                _struct["downsample"] = int(self._downsample)
            if self._min_size is not None:
                _struct["min_size"] = int(self._min_size)
            manifest = _pp_stream_to_store(
                self._channels,
                self._recipe_json,
                self._spacing,
                self._out_store_path,
                block_workers=self._block_workers,
                should_cancel=lambda: self._cancel_requested,
                on_progress=lambda done, total: self.progress.emit(
                    int(done), int(total)),
                source_fingerprint=self._source_fingerprint,
                **_struct,
            )
            if self._cancel_requested:
                # A late cancel after the engine returned normally: the store was
                # promoted, but the main-thread token guard discards the stale
                # result — emit a terminal "cancelled" so `thread.quit` fires.
                self.error.emit("cancelled")
                return
            self.finished.emit(str(self._out_store_path), manifest)
        except PreprocessingCancelled:
            # Cooperative cancel reached the engine (raised between block waves).
            # `str(PreprocessingCancelled())` can be empty, so emit the literal
            # "cancelled" (grill B4) — MUST precede the generic handler.  The
            # engine already removed its partial `.tmp` store.
            self.error.emit("cancelled")
        except BaseException as exc:   # noqa: BLE001
            # A worker thread must ALWAYS emit a terminal signal (even on a
            # non-Exception) so `thread.quit` fires and the pair is never leaked.
            self.error.emit(f"{exc}")


# "Zoom depth" combo — display labels + the layer `_zoom_depth` mode each maps
# to, in combo order (index <-> mode).  None = Off (byte-identical to the coarse
# camera-centered band); a float = fraction of full Z depth; "cube" = isotropic
# box.  Single source of truth shared by the controller (index -> mode) and the
# widget (mode -> index); ui.py imports these (ui depends on controller, not the
# reverse, so no import cycle).
ZOOM_DEPTH_LABELS = ("Off", "25%", "50%", "75%", "Full", "Cube (match width)")
ZOOM_DEPTH_MODES = (None, 0.25, 0.5, 0.75, 1.0, "cube")


#: A still-running background (thread, worker) — a whole-volume process OR a
#: large-data convert / source read — is parked here across controller / widget
#: teardown so the QThread C++ object is NOT garbage-collected while its OS thread
#: runs.  Module-level so it outlives the controller.
_ORPHANED_THREADS: list = []


def _orphan_running_thread(thread, worker) -> None:
    """Keep a still-running process (thread, worker) referenced past teardown so
    the QThread object is not destroyed while its OS thread is still executing —
    the "QThread: Destroyed while thread is still running" native abort.

    Cooperative cancel stops the TILED ops promptly, but a whole-array op
    (CLAHE / bleach / geometry) runs to completion and can outlast ``thread.wait``.
    Rather than block the app close for minutes, we hold the pair here and let it
    self-remove when its ``finished`` fires.  Best-effort; never raises."""
    if thread is None:
        return
    _ORPHANED_THREADS.append((thread, worker))

    def _drop(*_a):
        try:
            _ORPHANED_THREADS[:] = [
                (t, w) for (t, w) in _ORPHANED_THREADS if t is not thread
            ]
        except Exception:
            pass

    try:
        thread.finished.connect(_drop)
    except Exception:
        pass


def _preproc_parallel_enabled() -> bool:
    """Whole-volume block-parallel processing is ON by default — verified locally
    (parallel across all cores + clean cancel) and byte-identical to serial, so it
    only changes SPEED, never the result.  Escape hatch: set
    ``EPICYCLE_PREPROC_PARALLEL`` to ``0`` / ``false`` / ``no`` / ``off`` to force
    the serial path if the parallel path ever misbehaves on a machine/dataset."""
    return os.environ.get("EPICYCLE_PREPROC_PARALLEL", "").strip().lower() not in (
        "0", "false", "no", "off")


# Plain-language step labels (biologist-facing — the known op name kept in parens
# where useful).  Unknown keys fall back to the key itself.
_PROGRESS_OP_LABELS = {
    "normalize": "Normalizing",
    "gaussian": "Smoothing (Gaussian)",
    "background": "Removing background",
    "clahe": "Boosting contrast (CLAHE)",
    "bleach": "Evening depth brightness",
    "geometry": "Resizing",
}

# Rough per-voxel cost (ns), measured single-core; only the RELATIVE
# weights matter — they set how the bar apportions time across steps so the
# contrast step (which dominates wall time but has no fine progress) doesn't make
# the bar race then stall.  The tiled ops are divided by the worker count.
_PROGRESS_OP_NS_PER_VOXEL = {
    "normalize": 17.0, "gaussian": 38.0, "background": 266.0,
    "clahe": 205.0, "bleach": 3.0, "geometry": 50.0,
}
_PROGRESS_TILED_OPS = frozenset({"gaussian", "background"})
_PROGRESS_DEFAULT_NS = 50.0


def _format_eta(seconds: float) -> str:
    s = int(max(0, round(seconds)))
    if s < 60:
        return f"~{s}s left"
    m = s // 60
    if m < 60:
        return f"~{m} min left"
    return f"~{m // 60}h {m % 60}m left"


class _ProcessProgressAggregator:
    """Maps ``execute``'s op / tile events to a cost-weighted ``(fraction, label,
    indeterminate)`` for the whole-volume progress bar.

    Pure (no Qt): the worker passes ``emit`` (its Qt progress signal's ``.emit``)
    and ``now`` (a monotonic clock).  Cost-weighted because the contrast step
    (CLAHE) dominates wall time but has NO tile granularity, so an op-count or
    voxel bar would race to ~75% then stall.  A whole-array op shows
    an INDETERMINATE (marquee) state — honest "working, no fine progress" — while
    a tiled op shows a moving percent + a rough time-left.  Best-effort: a bad
    event never breaks the run."""

    def __init__(self, channel_voxels, recipe_json, workers, emit, now,
                 min_interval=0.1, channel_names=None):
        self._voxels = [int(v) for v in channel_voxels]
        self._names = list(channel_names) if channel_names else None
        self._workers = max(1, int(workers))
        self._emit = emit
        self._now = now
        self._min_interval = float(min_interval)
        self._total = self._compute_total_cost(recipe_json) or 1.0
        self._completed = 0.0
        self._cur_cost = 0.0
        self._cur_tiled = False
        self._cur_label = ""
        self._t0 = None
        self._last_emit = None

    def _op_cost(self, key, channel_index):
        ns = _PROGRESS_OP_NS_PER_VOXEL.get(key, _PROGRESS_DEFAULT_NS)
        if channel_index is None:
            vox = sum(self._voxels)                  # geometry touches all channels
        elif 0 <= channel_index < len(self._voxels):
            vox = self._voxels[channel_index]
        else:
            vox = self._voxels[0] if self._voxels else 0
        cost = ns * float(vox)
        if key in _PROGRESS_TILED_OPS:
            cost /= self._workers
        return cost

    def _compute_total_cost(self, recipe_json):
        # Count ONLY runnable ops — mirror execute's filter — so the denominator
        # matches the op events (execute emits an event only for a runnable op).  A
        # dropped / unimplemented op left in the raw recipe would otherwise inflate
        # `total` and leave the bar short of 100% (grill correctness LOW-1).
        from packages.epicycle_preprocessing import registry as _reg

        def _runnable(key):
            op = _reg.find(key) if key else None
            return bool(op and getattr(op, "implemented", False)
                        and getattr(op, "fn", None) is not None
                        and getattr(op, "kind", None) == "intensity")

        total = 0.0
        channels = recipe_json.get("channels", []) if isinstance(recipe_json, dict) else []
        geom = recipe_json.get("geometry") if isinstance(recipe_json, dict) else None
        for ci, ch in enumerate(channels or []):
            total += self._op_cost("normalize", ci)
            for step in ch or []:
                key = step.get("key") if isinstance(step, dict) else None
                if key and _runnable(key):
                    total += self._op_cost(key, ci)
        if geom:
            total += self._op_cost("geometry", None)
        return total

    def feed(self, event):
        try:
            self._feed(event)
        except Exception:
            pass                                     # progress is never load-bearing

    def _feed(self, event):
        now = self._now()
        if self._t0 is None:
            self._t0 = now
        kind = event.get("kind")
        if kind == "op":
            self._completed += self._cur_cost        # the previous op is now done
            key = event.get("op")
            ci = event.get("channel")
            self._cur_cost = self._op_cost(key, ci)
            self._cur_tiled = bool(event.get("tiled"))
            self._cur_label = self._label(key, ci, event)
            self._push(now, tile_frac=0.0, force=True)
        elif kind == "tile" and self._cur_tiled:
            total = event.get("total") or 0
            done = event.get("done") or 0
            self._push(now, tile_frac=(done / total if total else 0.0), force=False)

    def _label(self, key, ci, event):
        base = _PROGRESS_OP_LABELS.get(key, str(key or "Working"))
        if ci is None:
            return base
        # Prefer the channel NAME so a single-channel op reads clearly
        # ("Boosting contrast — test_ch0") instead of a "channel 1 of 2" that
        # looks like it processes both channels when it only touches one.
        if self._names is not None and 0 <= int(ci) < len(self._names):
            return f"{base} — {self._names[int(ci)]}"
        n = int(event.get("n_channels") or 1)
        return f"{base} — channel {int(ci) + 1} of {n}" if n > 1 else base

    def _push(self, now, tile_frac, force):
        if (not force and self._last_emit is not None
                and (now - self._last_emit) < self._min_interval):
            return
        self._last_emit = now
        frac = (self._completed + tile_frac * self._cur_cost) / self._total
        frac = min(max(frac, 0.0), 0.999)
        if self._cur_tiled:
            label = self._cur_label
            eta = self._eta(now, frac)
            if eta:
                label = f"{label} · {eta}"
            self._emit(frac, label, False)           # determinate: moving percent
        else:
            label = self._cur_label
            if self._cur_cost > 0.4 * self._total:
                label = f"{label} — the slow step, this can take a few minutes"
            self._emit(frac, label, True)            # indeterminate: marquee

    def _eta(self, now, frac):
        if self._t0 is None or frac < 0.02:
            return None
        elapsed = now - self._t0
        return _format_eta(elapsed / frac * (1.0 - frac))


class DataPreparationController:
    def __init__(self, widget, viewer):
        self.widget = widget
        self.viewer = viewer

        # Arm the native crash logger (faulthandler) for Step 1 too: a native
        # GL/vispy crash — e.g. a Raw⇄Processed swap — dies with NO Python
        # traceback, so this writes an all-threads native dump to
        # ~/.epicycle/crash_logs to capture the exact frame.  Best-effort +
        # idempotent (guarded by the logger's own module state).  Diagnostic
        # safety net (mirrors the Step-3 crash logger, retained on baseline).
        try:
            from napari_epicycle.utils.crash_logger import install_crash_logging
            install_crash_logging()
        except Exception:
            pass

        # Report a corrupt `step1_state.json` from ANY load site, not just the
        # one restore path that passes `on_corrupt=`.  A corrupt load is a
        # total loss of Step-1 state; the ~25 bare `load_state(...)` calls used
        # to swallow it silently.  Registered here (once per controller) and
        # weakly bound so a dead controller can never be resurrected by it.
        try:
            import weakref as _weakref

            self._state_problem_relay = _StateProblemRelay(widget)
            self._state_problem_relay.reported.connect(
                self._report_state_problem)
            _ref = _weakref.ref(self)

            def _corrupt_hook(msg, _ref=_ref):
                obj = _ref()
                if obj is None:
                    return
                # THREAD SAFETY: `_export.persist_patch` calls `load_state` and
                # runs on the EXTRACT QThread, so a corrupt file discovered
                # mid-extract fires this hook OFF the GUI thread.
                # `_report_state_problem` sets QLabel text/stylesheet and `_log`
                # appends to a QTextEdit — main-thread-only.  Marshal instead of
                # touching them from the worker.
                obj._report_state_problem_threadsafe("STEP1-STATE", msg)

            set_corrupt_state_hook(_corrupt_hook)
        except Exception:      # noqa: BLE001 — reporting must never break init
            pass

        # Layer list change (added/removed)-- hook onto the viewer
        # and check for the following
        self.viewer.layers.events.inserted.connect(self._on_layers_changed)
        self.viewer.layers.events.removed.connect(self._on_layers_changed)

        # Connect to each layer's name change
        for layer in self.viewer.layers:
            layer.events.name.connect(self._on_layer_renamed)

        # Again hook onto the viewer and check the following
        # If a new layer is added, also connect its name event
        self.viewer.layers.events.inserted.connect(self._on_layer_inserted)

        # Initial sync
        self._on_layers_changed()

        # Storage for zrange intensity.
        self.selected_z_range = None

        # In-memory staged patches (name -> {data, channels, ...}); reset at the
        # start of each extraction. Established here as a class invariant so
        # readers (plot-on-select, export-scope, autoselect) are safe before the
        # first extraction has run.
        self.extracted_patches = {}

        # Data browser state.
        self._loaded_source_file = None
        # Session IDENTITY of the loaded source (Brick 4a).  The registry row's
        # stable `source_id`, resolved once at load, so ④ interactivity survives
        # a relinked / moved raw file and a workspace copied to another PC.
        # Set + cleared in EXACTLY the same places as `_loaded_source_file`
        # (a stale id would silently show ANOTHER source's patches in ④).
        self._loaded_source_id = None
        # True while the loaded source is being served from its PREPARED COPY
        # because the original image file can't be found (Brick 4a a4).  Prepare
        # and "Process whole volume" read the raw block-wise, so they are
        # switched off in this state; Extract and Trim crop the store and stay
        # live.
        self._raw_missing = False
        #: Amber line to re-apply after a raw load whose prepared copy was
        #: refused because the file changed (Brick 4a a3).  None normally.
        self._source_freshness_note = None
        self._loaded_source_layers = []
        # The `(name, array)` channels of the currently-loaded source, kept so
        # `on_step_exit` can REMOVE the source Image layers (they otherwise
        # strand on the shared canvas and pollute Step 3's annotator, which
        # grabs ALL Image layers) and `on_step_enter` can re-show them.  Holds
        # only references to the display arrays (memmap views / already-in-RAM
        # channels), so this costs no extra memory over keeping the layers.
        self._source_display_channels = None
        # object -> last-known-name map for the TRACKED source layers.  The
        # progressive<->native 2D/3D swap + its two sibling readers
        # (`_apply_3d_detail_level`, the step-exit removal) all identify the
        # source layers BY NAME from the two caches above; a rename in napari
        # would leave those name caches STALE (the swap then orphans the renamed
        # layer, re-adds a DUPLICATE under the old name, and loses contrast).
        # `_on_layer_renamed` repairs the caches by matching the renamed layer on
        # OBJECT IDENTITY here (its old name is gone from the layer by the time
        # the event fires).  A WeakKeyDictionary so removed layers drop
        # automatically; rebuilt fresh at every site that (re)builds the caches
        # (load/reshow + both swap re-add sites) so it never carries a stale
        # object.  See `_track_source_layer_objects`.
        self._source_layer_name_map = weakref.WeakKeyDictionary()
        # The channel NAMES ticked in `image_layer_combo`, snapshotted on
        # step-exit (`_remove_source_layers_from_viewer`) BEFORE the source
        # layers are removed — removing them fires `_on_layers_changed` ->
        # `update_image_layer_list`, which drops the now-absent names and clears
        # the selection.  `_reshow_source_layers` re-applies this on step
        # re-enter so Extract still finds the channels without a manual re-tick.
        self._saved_channel_selection = None
        self._scanned_files = []

        # L0 large-data data layer: the full-res source stays LAZY
        # on disk and only the crop the user extracts is ever materialized.
        # `_source_lazy` is the on-disk handle (memmap / dask), in its on-disk
        # axis order; `_source_axes_resolved` is the CZYX/ZCYX/ZYX order
        # resolved at load; `_channel_index_by_name` maps a display layer name
        # to its channel index so extraction can slice the lazy handle by the
        # channels the user ticked.  `_axis_order_overrides` maps a normalized
        # source path to the Order the user deliberately picked, so a re-load of
        # the same file does not clobber that choice with re-detection.
        self._source_lazy = None
        self._source_axes_resolved = None
        # RESERVED for Brick 2 (restore/export) — WRITE-ONLY in Brick 1 (read in
        # zero places today).  Provenance companion to `_source_axes_resolved`: the store's honest ON-DISK order (schema-v2
        # `epicycle_source.source_axes`) for a prepared-store load, else None (a
        # raw load's `_source_axes_resolved` already IS the on-disk order).
        # Never read by the box tool / extract — those use `_source_axes_resolved`.
        self._source_disk_axes = None
        self._channel_index_by_name = {}
        self._axis_order_overrides = {}
        # The source path the Order combo currently DESCRIBES — set whenever the
        # combo is programmatically pointed at a file (detect-on-selection or the
        # load path).  `_resolve_prepare_image_axes` trusts the combo only for
        # this file, so a Prepare is oriented by the file being prepared, never a
        # stale combo left over from a different source.
        self._axis_combo_source = None

        # 3D-detail (manual sharp-3D) state.  A displayed multiscale store renders
        # only its COARSEST level in 3D by default (napari 0.8 does not auto-refine
        # in 3D); the 3D-detail slider pins a finer pyramid level via each layer's
        # `locked_data_level`, applied ONLY while the viewer is in 3D so 2D
        # pan/zoom keeps napari's automatic per-zoom level picking.
        # `_locked_3d_level` is the chosen level index (0 = finest); None = none
        # chosen yet.  `_pyramid_level_shapes` holds each level's (Z, Y, X) for the
        # readout.  `_ndisplay_connected` guards the one-time 2D/3D-switch hook.
        self._locked_3d_level = None
        self._pyramid_level_shapes = []
        self._ndisplay_connected = False
        # Set while a 2D<->3D source-display swap is mid-flight so the transient
        # per-channel remove/add churn doesn't fire _on_layers_changed (which
        # normalize_recipe would use to truncate the recipe's trailing-channel ops
        # + clear the picker); the swap reconciles ONCE at the settled count.
        self._swapping_source_display = False
        # Same guard for the INITIAL source LOAD / re-show: the channel-add loop
        # fires _on_layers_changed at the transient N-1 count → normalize_recipe
        # would truncate the trailing channel's recipe ops.  The load wrapper
        # reconciles ONCE at the settled full count (Bug A, same class as swap).
        self._loading_source_layers = False
        # True only while Step 1's source layers are actually ON the shared canvas
        # (they are removed on step-exit so Step 3's annotator does not grab them,
        # and re-added on re-enter).  The 3D-detail pin only applies while they
        # are shown, so the never-disconnected ndisplay hook no-ops on 2D/3D
        # toggles that happen in OTHER steps.  The chosen level (`_locked_3d_level`)
        # persists across the round-trip so returning to Step 1 restores the pin.
        self._source_layers_on_canvas = False

        # Progressive-3D display state.  `progressive_wanted()` (checkbox on +
        # checked + engine available) engages the vendored napari-#9067 seam in
        # `_add_source_display_layers`; `_progressive_active` records whether ANY
        # channel was added progressively in the current display (so the manual
        # 3D-detail slider is hidden — progressive owns level selection).  The
        # `_current_multiscale_*` fields remember the store currently shown so the
        # live toggle can re-display it in the new mode; cleared on unload.
        self._progressive_active = False
        # "Zoom depth" mode: the user's chosen depth-sizing for the progressive-3D
        # tile (None=Off, a float fraction, or "cube").  Persisted here (NOT on
        # the transient progressive layer, which is rebuilt on every 2D<->3D swap)
        # and stamped onto each fresh progressive layer's `_zoom_depth` in
        # `_swap_native_to_progressive`; read LIVE per-draw by the seam.
        self._zoom_depth_wanted = None
        self._current_multiscale_store = None
        self._current_multiscale_display_name = None
        self._current_multiscale_src = None

        # Background source-read (QThread) state.  `_source_load_token` stamps
        # each launch; a Cancel / re-launch bumps it so a superseded read's
        # finished/error handler early-returns and never touches the viewer.
        # `_source_load_thread`/`_source_load_worker` are the CURRENT in-flight
        # pair (what Cancel targets); `_draining_source_threads` holds any
        # older pair still finishing its atomic read after a Cancel unlocked
        # the UI, so the QThread is never GC'd mid-flight.
        self._source_load_token = 0
        self._source_load_thread = None
        self._source_load_worker = None
        self._draining_source_threads = []
        # The Load button doubles as Cancel WHILE a read is in flight:
        # `_source_loading` gates the dispatcher (`on_load_or_cancel_clicked`)
        # and is True only between launch and finish/error/cancel — NOT while an
        # old thread is merely draining, so a post-Cancel click starts a fresh
        # load rather than re-cancelling.  `_load_button_text` remembers the
        # idle label to restore.
        self._source_loading = False
        self._load_button_text = None
        self._load_button_icon = None

        # Background EXTRACT (QThread) state (F1) — `extract_patches` slices +
        # materializes + writes each ROI patch on a worker so the GUI stays live.
        # `_extract_token` stamps each launch (a stale finish/written/progress
        # early-returns); `_extract_running` is True from launch until the
        # finished/error slot (gates the re-entrancy guard + the button-as-Cancel
        # dispatcher); `_extract_launching` guards the pre-lock window (the
        # discard-confirm modal spins the event loop — grill MED-5); the
        # thread/worker pair is the in-flight one, `_draining_extract_threads`
        # keeps a step-exit-cancelled pair referenced so its QThread is never
        # GC'd mid-flight.  `_extract_status_ctx` carries the PREP-computed
        # skip/drop counts through to the finish status line.
        self._extract_token = 0
        self._extract_running = False
        self._extract_launching = False
        self._extract_thread = None
        self._extract_worker = None
        self._draining_extract_threads = []
        self._extract_button_text = None
        self._extract_button_icon = None
        self._extract_status_ctx = {}

        # Background ④ Z-crop TRIM (QThread) state — the "Trim" button crops the
        # selected saved patch's TIFF + sidecar in place on a worker.  Mirrors
        # the EXTRACT machinery: `_trim_token` stamps each launch (a stale
        # finish/error early-returns), `_trim_running` is True from launch until
        # the finished/error slot (gates the surface + the disk-read relaunch),
        # `_trim_launching` guards the pre-lock window (the confirm modal spins
        # the event loop — F1 MED-5); the thread/worker pair is the in-flight
        # one, `_draining_trim_threads` keeps a step-exit pair referenced so its
        # QThread is never GC'd mid-flight (a trim is NOT cancellable — it must
        # finish its atomic replace).  `_left_step1` makes a late FINISH inert.
        self._trim_token = 0
        self._trim_running = False
        self._trim_launching = False
        self._trim_thread = None
        self._trim_worker = None
        self._draining_trim_threads = []
        self._left_step1 = False
        # Brick 5: set on step-exit when source layers were on the canvas, read
        # (and cleared) once on the next step-enter so ② can say why the canvas
        # is empty.  Step 1 deliberately does NOT reload data on re-entry.
        self._source_needs_reload = False
        # Freshest current-patch depth for the trim precondition clamp: the
        # intensity plot's own slice count + which patch it was drawn for.
        self._plot_n_z = None
        self._plot_n_z_patch = None

        # Background large-data PREPARE (QThread) state — the
        # "Prepare for large-data viewing" button converts the selected source
        # to a multiscale OME-Zarr store off the main thread, then loads that
        # store as a MULTISCALE image.  Mirrors the source-read machinery above
        # exactly: `_largedata_token` stamps each launch (Cancel / re-launch
        # bumps it so a superseded convert's finished/error handler
        # early-returns), `_largedata_thread`/`_largedata_worker` are the
        # in-flight pair, `_draining_largedata_threads` holds an old pair still
        # finishing its atomic write after a Cancel unlocked the UI (so the
        # QThread is never GC'd mid-flight).  `_largedata_preparing` gates the
        # Prepare-as-Cancel dispatcher; `_largedata_src_path` remembers the
        # original source so the main-thread handler can set export provenance.
        self._largedata_token = 0
        self._largedata_thread = None
        self._largedata_worker = None
        self._draining_largedata_threads = []
        self._largedata_preparing = False
        # The thread of a CANCELLED convert that is still draining; the ② lock
        # is held until it finishes.
        self._largedata_cancel_draining_thread = None
        self._prepare_button_text = None
        self._prepare_button_icon = None
        self._largedata_src_path = None

        # Background preprocessing PREVIEW (QThread) state (Brick 2b).  Mirrors
        # the source-read machinery: `_preview_token` stamps each launch so a
        # superseded run's finished/error handler early-returns (the user edited
        # the recipe / switched source mid-run — the Step-3 QC stale-finish
        # guard); `_preview_thread`/`_preview_worker` are the in-flight pair;
        # `_draining_preview_threads` keeps an old pair referenced until it
        # finishes so the QThread is never GC'd mid-flight.  `_preview_names`
        # carries the channel names the in-flight run is computing.
        self._preview_token = 0
        self._preview_thread = None
        self._preview_worker = None
        self._draining_preview_threads = []
        self._preview_names = []
        # WYSIWYG PREVIEW render (Brick 2c): the ⟨preview⟩ napari Image layers
        # showing the PROCESSED test crop, and the per-channel center-crop bounds
        # the in-flight run was cropped to (so the finish handler can position
        # each layer where its crop was taken).  `_preview_layers` holds the live
        # layer REFS so a re-preview / source-switch / step-exit can hide-then-
        # remove them (freed-texture safety).  Both reset here.
        self._preview_layers = []
        self._preview_bounds = []
        # Layers WE hid so the ⟨preview⟩ render is readable (C#6) — restored by
        # `_clear_preview_layers`, so only ever what this controller changed.
        self._preview_hidden_layers = []
        # A "this recipe can't be applied to this source" warning resolved at
        # launch (C#8) and appended to the finish status; None = nothing to say.
        self._preview_stream_warning = None
        # Sentences naming ops this REGION cannot show faithfully (C#3 / C#4),
        # resolved at launch from the crop vs the source shape.
        self._preview_fidelity = []

        # Background WHOLE-VOLUME process (QThread) state (Brick 1b-1).  Mirrors
        # the PREVIEW machinery but is its OWN independent op: `_process_token`
        # stamps each launch so a superseded run's finished/error handler
        # early-returns (do NOT reuse `_preview_token` — the two ops must not
        # cross-cancel); `_process_thread`/`_process_worker` are the in-flight
        # pair; `_draining_process_threads` keeps an old pair referenced until it
        # finishes so its QThread is never GC'd mid-flight.  `_processing` gates
        # the button's double-life as Cancel + the launch guard.
        self._process_token = 0
        self._process_thread = None
        self._process_worker = None
        self._draining_process_threads = []
        self._processing = False
        self._process_names = []
        self._process_button_text = None
        self._process_button_icon = None
        # The ⟨processed⟩ WHOLE-VOLUME render layers (uint16, WYSIWYG) + the
        # in-memory cache holding the last computed uint16 result.  `_processed_*`
        # reset here; the cache is NOT persisted to disk in this brick (on-disk
        # persistence is a later brick).  `_adding_processed_layers` guards
        # `_on_layers_changed` mid-insert (mirrors `_adding_preview_layers`).
        self._processed_layers = []
        self._processed_cache = None
        self._adding_processed_layers = False

        # Background STREAMING (out-of-core) process (QThread) state (Phase 2a).
        # A LAZY / large-data source (a prepared multiscale store) is processed
        # block-by-block into a `__proc__` OME-Zarr store instead of the in-memory
        # path — memory-bounded + freeze-free.  Its OWN in-flight
        # pair (SEPARATE from `_process_thread` — the store worker has a different
        # `finished` signature + cleanup), but it SHARES the logical-op state
        # (`_processing` / `_process_token` / the Cancel-button flip) so only ONE
        # "Process whole volume" runs at a time regardless of regime.  Added to the
        # `shutdown()` close-sweep so a mid-run streaming write is never nulled into
        # GC while its QThread runs.  `_processed_store_layers` holds the
        # live lazy-multiscale ⟨processed⟩ display layers (hide-then-remove on
        # replace / switch / teardown).
        self._process_store_thread = None
        self._process_store_worker = None
        self._draining_process_store_threads = []
        self._processed_store_layers = []
        # Progressive-3D for the ⟨processed⟩ STORE view (item 4): route the
        # processed display through the SAME napari-#9067 progressive seam the RAW
        # source uses so a processed store refines in 3D like raw (was a plain
        # `add_image(multiscale=True)` — native/coarse-in-3D).  DISPLAY-ONLY
        # analogues of the raw `_current_multiscale_*` / `_progressive_active`
        # trio, kept SEPARATE so a processed swap NEVER clobbers the raw
        # `_source_*` state (B4 discipline).  Safe because the raw⇄processed
        # toggle is a CLEAN SWAP — only ONE set is ever on the canvas, so there is
        # never more than one progressive layer set (no "two engines" hazard).
        #   `_current_processed_multiscale_store` — the store path currently shown
        #     (gates the processed swap, mirroring `_current_multiscale_store`);
        #   `_processed_display_channels` — [(marker_name, levels), …] rebuilt
        #     into progressive layers on the swap (mirrors `_source_display_channels`);
        #   `_processed_progressive_active` — whether ANY processed channel is
        #     currently progressive (mirrors `_progressive_active`).
        self._current_processed_multiscale_store = None
        self._processed_display_channels = None
        self._processed_progressive_active = False
        # Brick A (clean-swap rework): the Raw ⇄ Processed VIEW toggle keeps ONLY
        # the viewed set on the canvas (raw REMOVED while processed shows, and vice
        # versa) — never a hide/show that leaves both sets in the layer list.
        # Because a swap-to-raw CLEARS the processed layers, the tracked layer
        # lists can no longer answer "does a processed view exist this session?" —
        # `_processed_available` records that (drives greying + the `\` bind + the
        # toggle guard) and survives the swap.  `_processed_redisplay` holds the
        # spec to RE-CREATE the processed set on a flip back to processed:
        #   {"kind":"store", "store_path":.., "names":..}                (streaming)
        #   {"kind":"mem", "channel_levels":.., "names":.., "spacing":..} (in-memory)
        # Both are cleared in `_teardown_preview` so a new source starts greyed.
        self._processed_available = False
        self._processed_redisplay = None

        # --- Preview REGION box (Preview v2 Piece 2) -------------------------
        # A single movable BoxRec choosing WHERE the preview computes, rendered
        # by its OWN `Step1BoxOverlay` instance — SEPARATE from the ③ box tool's
        # overlay (the box tool lazily creates + destroys `box_tool._overlay`
        # itself; we never reuse or touch that instance).  A distinct colour
        # (`_PREVIEW_BOX_COLOR`) keeps it from reading as an extract box.
        #
        # ndisplay OWNERSHIP: while a preview box is active NOBODY forces
        # ndisplay — the overlay is scene-anchored (a vispy Line) and renders in
        # whatever view the user is in.  The ③ box tool owns ndisplay only while
        # ITS mode is active; placing a preview region STOPS the box tool first
        # (mode arbitration), and the box tool restores its own ndisplay on stop.
        #
        # Transient: destroyed on source load/switch, source-unload, step-exit,
        # and an `aboutToQuit` hook — never persisted (no schema surface).
        self._preview_box = None
        self._preview_box_overlay = None
        self._preview_box_close_hook_connected = False
        self._preview_size_frac = 0.25   # M preset (mirrors the ③ box-tool default)

        # Signatures of boxes already EXPORTED this session or RESTORED from disk
        # -> excluded from re-extraction so old boxes aren't re-numbered high (the
        # escalating-index bug).  Populated ONLY on export-success + restore, NEVER
        # at extract/stage time (else a staged-but-unexported box could be excluded
        # AND wiped by the reset -> silent data loss).
        self._committed_box_sigs = set()

        # ④ "Show patch locations" (Brick 2): the set of saved-patch names whose
        # bounding boxes are currently drawn in the SOURCE volume, and a SEPARATE
        # vispy overlay that renders them.  Deliberately independent of the ③ box
        # tool (its `boxes` list drives drawing + extraction): a location box must
        # never be re-extracted nor pollute ③'s transient list.  The overlay
        # self-tracks the camera + 2D/3D flips (scene-attached + its own dims
        # subscription), so it needs no per-frame redraw wiring.
        self._visible_patch_locations = set()
        self._loc_overlay = None
        # ④ depth-profile cache: {patch_name: (curves, n_z, z_range)} so a
        # re-select / subsection-return doesn't re-read the TIFF + re-reduce
        # (a REOPENED patch's plot is a full disk read otherwise).  Invalidated
        # per-patch on trim/delete, cleared on extract.
        self._intensity_curve_cache = {}
        # ④ row interactions: patches whose capture inspector is OPEN (drives the
        # camera button's lit-cyan state) + the double-clicked patch whose box is
        # HIGHLIGHTED (white) in the source volume.
        self._open_capture_patches = set()
        self._highlighted_patch = None

        # Workspaces already offered the layout migration this session
        # (guards against re-prompting on every textChanged / step enter).
        self._step1_migration_offered = set()

        # --- Brick 2: source-metadata round-trip state -----------------------
        # Workspaces whose registry <-> largedata/ store reconciliation +
        # source-list repopulation already ran this session (mirrors the
        # migration guard so the once-per-workspace-per-session store scan +
        # abs->rel pointer rewrite never re-fires on every textChanged / enter).
        self._step1_sources_restored = set()
        # The source path the edit signals (alias / voxel / axes save) target —
        # set on every select so a mid-window edit lands on the RIGHT source's
        # registry row (never guesses off the loaded file).  Also the
        # restore-idempotency key (a re-select of the same path is a no-op).
        self._selected_source_path = None
        # A recipe edited while NOTHING is selected is HELD here (never silently
        # dropped) and adopted by the NEXT selected source -- but only when that
        # source has no saved recipe of its own (a saved recipe always wins).
        self._pending_recipe_for_next_selection = None
        # True only while the source list is being PROGRAMMATICALLY rebuilt /
        # populated / browsed, so a transient current-item change during that does
        # NOT trigger a restore (belt-and-suspenders — `addItem` never
        # auto-selects and `clear` yields current=None, both already safe).
        self._suspend_source_restore = False
        # `read_voxel_spacing` memo keyed by normalized path -> (mtime, voxel),
        # so arrowing a folder (or arrow-back) opens each source header at MOST
        # once (grill FIX A); a mtime change re-detects.  A "user"-override source
        # never reaches it (short-circuited in `_resolve_voxel_for_source`).
        self._voxel_detect_cache = {}
        # Debounced save-on-edit.  Edits stage an `upsert_source`
        # kwargs dict into `_pending_source_edits` (captured with the workspace
        # they were made against); the timer flushes them by RE-READING the
        # current on-disk state and layering the edits on top (so a concurrent
        # Prepare/Export write is never clobbered), then one atomic `save_state`.
        # FLUSHED before every context change (load / prepare / browse / step
        # exit / workspace change) so a pending write can't land in the wrong
        # workspace slot — the Step-2 gotcha.
        self._pending_source_edits = []
        self._pending_source_ws = None
        self._state_save_timer = QTimer(self.widget)
        self._state_save_timer.setSingleShot(True)
        self._state_save_timer.timeout.connect(self._flush_source_state_save)

        # Whole-Step-1 mutual-exclusion flag: True while an extract runs so a
        # queued/double click can't re-enter or launch a conflicting op (N1/R4).
        self._step1_busy = False

        # The ④ depth profile auto-draws on patch-select (`_on_roi_selection_
        # changed`) — the old manual "Plot Intensity" button was removed as
        # redundant (UI/UX consult), so there is no button click to wire here.

        # ④ patch-location viz (Brick 2) + capture (Brick 4): wire the per-row
        # 👁/capture buttons + the master toggle + the box render-style controls.
        try:
            self.widget.library_box_list.set_location_eye(
                self.toggle_patch_location, self.is_patch_location_visible)
            self.widget.library_box_list.set_capture_callback(
                self._on_capture_patch_clicked, self._capture_open_for)
            self.widget.library_box_list.set_double_click_callback(
                self._on_patch_double_clicked)
            self.widget.library_box_list.set_delete_callback(
                self._on_delete_patch)
            # ④ per-patch box-style swatch → popup (P2 Brick 2).
            self.widget.library_box_list.set_style_callback(
                self._on_patch_style_clicked, self._swatch_color_for)
            self.widget.show_locations_check.clicked.connect(
                self._on_show_locations_clicked)
            # ④ Snapshot split button (Brick 1): the MAIN half exports the
            # source view to a TIFF; the ▾ popover configures scale bar +
            # resolution (read at click time in `_on_snapshot_clicked`).
            self.widget.snapshot_button.clicked.connect(
                self._on_snapshot_clicked)
            # ④ live scale-bar control (Brick B): the toggle + every styling knob
            # re-applies to the viewer's napari scale bar via _apply_live_scale_bar.
            w = self.widget
            w.scalebar_toggle_button.toggled.connect(self._on_scalebar_changed)
            w.scalebar_auto_length_check.toggled.connect(self._on_scalebar_changed)
            # Pure-UI: Auto greys out the fixed-length spin (kept here so ui.py
            # carries no stray .connect — all scale-bar wiring is in one place).
            w.scalebar_auto_length_check.toggled.connect(
                lambda on: w.scalebar_length_spin.setEnabled(not on))
            w.scalebar_length_spin.valueChanged.connect(self._on_scalebar_changed)
            w.scalebar_font_spin.valueChanged.connect(self._on_scalebar_changed)
            w.scalebar_unit_combo.currentIndexChanged.connect(
                self._on_scalebar_changed)
            w.scalebar_color_combo.currentIndexChanged.connect(
                self._on_scalebar_changed)
            w.scalebar_box_check.toggled.connect(self._on_scalebar_changed)
            w.scalebar_ticks_check.toggled.connect(self._on_scalebar_changed)
            w.scalebar_position_combo.currentIndexChanged.connect(
                self._on_scalebar_changed)
        except Exception as exc:      # pragma: no cover — Qt-free fakes
            self._log("PATCHLOC", f"location viz wiring skipped: {exc}")
        # The global "Box appearance" style section was AXED in the ④ P2 rework
        # (Brick 1): box styling becomes per-patch in Brick 2, so there is no
        # global style handler to wire here any more.

        # keeps track of the set ranged button being clicked
        self.widget.set_select_zrange_button.clicked.connect(self._on_set_zrange_clicked)

        # keeps track of the autoselect zrange button being clicked
        self.widget.auto_select_zrange_button.clicked.connect(self._on_autoselect_zrange_clicked)

        # keeps track of the export button being clicked
        self.widget.save_reviewed_roi.clicked.connect(self._on_export_clicked)
        if hasattr(self.widget, "export_all_ready_roi"):
            self.widget.export_all_ready_roi.clicked.connect(self._on_export_all_ready_clicked)
        if hasattr(self.widget, "extracted_roi_list"):
            self.widget.extracted_roi_list.itemSelectionChanged.connect(self._on_roi_selection_changed)
        # A workspace change is a CONTEXT change: flush any pending source-edit
        # save FIRST (it was captured against the previous workspace, so it must
        # land there — the Step-2 gotcha), then restore/reconcile the newly-set
        # workspace's sources (once per workspace).  Connected before the badge
        # refresh so the flush + restore run ahead of the read-only refresh.
        self.widget.output_file_path.textChanged.connect(
            lambda _: self._flush_source_state_save()
        )
        # A workspace change also invalidates a recipe held for "the next
        # selection" -- that recipe belongs to the old workspace's sources.
        self.widget.output_file_path.textChanged.connect(
            lambda _: self._drop_pending_recipe("you changed the workspace")
        )
        # A workspace change repoints every store path - drop the memoized
        # contrast limits with it (Brick 3a #8).
        self.widget.output_file_path.textChanged.connect(
            lambda _: self._clear_contrast_cache("workspace change")
        )
        self.widget.output_file_path.textChanged.connect(
            lambda _: self._maybe_restore_step1_sources()
        )
        # Refresh source patch counts when workspace path changes.
        self.widget.output_file_path.textChanged.connect(
            lambda _: self._refresh_source_patch_counts()
        )
        # Offer the legacy→artifacts/step1 layout migration when a workspace
        # with loose root patches is selected.  DEBOUNCED so it fires after the
        # path settles (600 ms) rather than on every keystroke — covers Browse
        # (single setText) and manual typing without popping mid-edit, and
        # avoids a filesystem scan per keystroke.
        self._migration_offer_timer = QTimer(self.widget)
        self._migration_offer_timer.setSingleShot(True)
        self._migration_offer_timer.timeout.connect(self._maybe_offer_step1_migration)
        self.widget.output_file_path.textChanged.connect(
            lambda _: self._migration_offer_timer.start(600)
        )
        self._refresh_conditioning_status()

        # ③ box tool (Phase 1): the EpiCycle-Viz-style live 3D box.  Owns its
        # Shapes layer, ndisplay flip, click-to-place, Add, the box list, and
        # the live geometry-slider editing; its BoxRecs are the extract source
        # of truth once any box exists (see `get_drawn_shapes`).  Inert until
        # the user enters box mode; tears down on step-exit + source-unload.
        self.box_tool = Step1BoxTool(self)
        # Stop any in-flight / draining source-read QThread on viewer close,
        # widget destroy, or app quit — so a thread is never destroyed while
        # running (native crash) and no finished handler fires against a
        # deleted widget/viewer (both audits flagged this HIGH).
        self._connect_source_shutdown_hooks()
        # Entering ④ Review & Library re-reads the persisted registry (F2), so a
        # patch saved earlier this session (or in a prior one) is shown without
        # a source reload.  Fires only on EXPAND of that section, never collapse.
        self._wire_review_section_refresh()

    def _wire_review_section_refresh(self):
        """Refresh ④'s Library from the registry when its accordion section is
        expanded (F2 section-enter hook).

        The ④ header's ``clicked`` runs the accordion's own toggle FIRST (wired
        at ``addItem`` time), so by the time this slot runs the section's
        ``_expanded`` reflects the NEW state — guarding on it means we refresh on
        entering ④ but skip a wasteful rebuild when the user collapses it.
        Best-effort: silently no-ops if the accordion layout ever changes."""
        acc = getattr(self.widget, "accordion", None)
        secs = getattr(acc, "_sections", None) if acc is not None else None
        # Section order is fixed in `_init_ui`: Session, Load & Adjust,
        # Draw & Extract, Review & Library, Diagnostics.  ④ is index 3.
        if not secs or len(secs) < 4:
            return
        review = secs[3]
        header = getattr(review, "_header", None)
        if header is None or not hasattr(header, "clicked"):
            return
        header.clicked.connect(
            lambda _=False, s=review: (
                self._refresh_library_from_registry()
                if getattr(s, "_expanded", False) else None
            )
        )

    def _report_state_problem_threadsafe(self, prefix: str, msg: str) -> None:
        """`_report_state_problem`, safe to call from ANY thread.

        On the GUI thread it reports inline (unchanged).  From a worker it
        marshals a zero-delay `QTimer.singleShot` onto the widget so the QLabel
        / QTextEdit writes happen on the main thread.  With no QApplication at
        all (headless / import-time) it degrades to a plain print, never a
        widget touch.  Fully guarded: reporting must never raise into the
        caller, which is mid-`load_state`.
        """
        try:
            from qtpy.QtCore import QCoreApplication, QThread

            app = QCoreApplication.instance()
            if app is None:
                _last_resort_report(prefix, msg)
                return
            if QThread.currentThread() is app.thread():
                self._report_state_problem(prefix, msg)
                return
            # A queued SIGNAL, not `QTimer.singleShot`: singleShot starts its
            # timer in the CALLING thread, and a worker (a plain
            # `threading.Thread`, or a QThread whose loop has exited) has no
            # event loop to fire it — the report would silently vanish.  The
            # relay QObject lives in the GUI thread (parented to the widget),
            # so an auto-connection from any other thread is QUEUED and lands
            # on the main thread's next event-loop spin.
            relay = getattr(self, "_state_problem_relay", None)
            if relay is None:
                _last_resort_report(prefix, msg)
                return
            relay.reported.emit(prefix, msg)
        except Exception:      # noqa: BLE001 — never break the caller
            _last_resort_report(prefix, msg)

    def _report_state_problem(self, prefix: str, msg: str) -> None:
        """Log a ``step1_state.json`` problem AND show it inline in ①.

        Persistence failures used to be Diagnostics-pane-only — a collapsed,
        off-flow section — so a corrupt load or a failed save (disk full,
        OneDrive lock) looked exactly like success.  Both are silent data-loss
        classes, so they get the red inline status treatment the source-load /
        prepare errors already use.  Fully guarded: reporting a problem must
        never itself raise.
        """
        self._log(prefix, msg)
        try:
            label = getattr(self.widget, "workspace_status_label", None)
            if label is not None:
                label.setText(f"⚠  {msg}")
                label.setStyleSheet("font-size: 10px; color: #e07070;")
        except Exception:      # noqa: BLE001 — never break the caller
            pass

    def _clear_state_problem(self) -> None:
        """Undo a `_report_state_problem` banner once the problem is gone.

        Delegates to the widget's own `_update_workspace_status`, which repaints
        the normal ① status (and its normal colour) for the current workspace
        path -- so the cleared state can never drift from what the user sees on
        a fresh path edit.  Fully guarded: clearing must never raise.
        """
        try:
            fn = getattr(self.widget, "_update_workspace_status", None)
            if callable(fn):
                fn()
        except Exception:      # noqa: BLE001 — never break the caller
            pass

    def _log(self, prefix: str, msg: str) -> None:
        """Route a diagnostic to the widget's Diagnostics pane.

        Delegates to `DataPreparationWidget._log`, which appends to the
        read-only log pane AND mirrors to stdout.  Guarded so controller code
        can log during construction or teardown, when the widget may not have
        its pane yet.
        """
        sink = getattr(getattr(self, "widget", None), "_log", None)
        if sink is None:
            # Encode-safe fallback (no widget pane yet).  A non-ASCII char such
            # as the ⟨processed⟩ marker on a cp1252 stdout would raise
            # UnicodeEncodeError from `print` — a diagnostic must never crash
            # its caller, so degrade to ASCII rather than propagate.
            line = f"[{prefix}] {msg}"
            try:
                print(line)
            except Exception:
                try:
                    print(line.encode("ascii", "replace").decode("ascii"))
                except Exception:
                    pass
            return
        sink(prefix, msg)

    # when the layers list is changed
    def _on_layers_changed(self, event=None):
        """If the loaded data on the napari canvas changes do the following"""
        # Brick 5b item 3: this handler stays CONNECTED to the shared viewer
        # after step-exit, so Step 2/3 layer churn used to re-key ③'s channel
        # picker + ②'s recipe cards off layers that are not ours.  The step-enter
        # re-show healed it by accident; Brick 5 removed the re-show, so guard it
        # explicitly (same class as assessment #19).  Step 1 owns none of the
        # canvas while it is not the active step.
        if getattr(self, "_left_step1", False):
            return
        # While `_add_preview_layers` is mid-insert the ⟨preview⟩ layers are not
        # yet tracked in `self._preview_layers`, so a rebuild here would count
        # them as channels (quirk 1: they flashed as extra recipe columns on the
        # FIRST preview until the next re-enumeration dropped them).  Guard only
        # THIS handler (not the event) so napari's own inserted handlers still
        # add the layer to its layer list + keep it toggle-able.
        if getattr(self, "_adding_preview_layers", False):
            return
        # Same guard for the ⟨processed⟩ whole-volume render (Brick 1b-1): while
        # `_add_processed_layers` is mid-insert those layers are not yet tracked
        # in `self._processed_layers`, so a rebuild here would count them.
        if getattr(self, "_adding_processed_layers", False):
            return
        # Suppress the transient per-channel churn of a 2D<->3D source-display swap
        # (each mid-swap remove would else fire this at N-1 channels and
        # normalize_recipe would truncate the recipe's trailing-channel ops then
        # re-pad it empty); the swap reconciles ONCE at the settled count.
        if getattr(self, "_swapping_source_display", False):
            return
        # Suppress the transient per-channel churn of the INITIAL source LOAD /
        # re-show (Bug A): `_add_source_display_layers` adds channels one add_image
        # at a time; each fires this at the transient N-1 count and normalize_recipe
        # would truncate the trailing channel's recipe ops then re-pad it empty
        # (the SAME class as the swap; the swap path was guarded but the load path
        # was not).  The load wrapper reconciles ONCE at the settled full count.
        if getattr(self, "_loading_source_layers", False):
            return
        # Exclude the transient ⟨preview⟩ processed-crop render layers (Brick
        # 2c).  They ARE real napari Image layers, so without this a Preview of
        # an N-channel source transiently inflates BOTH ③'s channel picker AND
        # ②'s per-channel recipe columns to N+N (this one method feeds both via
        # `update_image_layer_list`).  IDENTITY-based, not a name-prefix match:
        # marker-drift-proof + collision-proof (`self._preview_layers` holds the
        # live layer refs).  `getattr` guard: `_on_layers_changed` can fire from
        # a layer insert before `__init__` sets `_preview_layers`.
        preview_layers = getattr(self, "_preview_layers", None) or []
        # Exclude the persistent ⟨processed⟩ whole-volume render layers too (Brick
        # 1b-1) — they ARE real napari Image layers, so without this a processed
        # N-channel source would DOUBLE both ③'s channel picker and ②'s recipe
        # columns to N+N.  IDENTITY-based (marker-drift-proof + collision-proof).
        processed_layers = getattr(self, "_processed_layers", None) or []
        # …and the lazy-multiscale ⟨processed⟩ STORE render layers (Phase 2a
        # streaming) — same reason: real Image layers that would else double ③'s
        # channel picker + ②'s recipe columns.  IDENTITY-based.
        processed_store_layers = getattr(self, "_processed_store_layers", None) or []
        excluded_layers = (list(preview_layers) + list(processed_layers)
                           + list(processed_store_layers))
        image_layer_names = [
            layer.name
            for layer in self.viewer.layers
            if isinstance(layer, Image) and layer not in excluded_layers
        ]
        # Keep ②'s per-channel recipe cards (+ ③'s picker) alive while the
        # ⟨processed⟩ view is the ONLY thing on the canvas.  The clean raw⇄processed
        # swap REMOVES the raw source layers, so the list above is empty
        # (⟨processed⟩ is excluded), and any stray layers-change would blank the
        # recipe panel.  A loaded source's channels are the truth regardless of
        # which VIEW is shown → fall back to them.  Gated on "a processed view is
        # active + raw is off-canvas" so a normal step-exit (raw removed, leaving
        # Step 1, `_processed_available` cleared by teardown) still clears.
        if not image_layer_names:
            src_channels = getattr(self, "_source_display_channels", None) or []
            if (src_channels and getattr(self, "_processed_available", False)
                    and not getattr(self, "_source_layers_on_canvas", True)):
                image_layer_names = [nm for (nm, _levels) in src_channels]
        self.widget.update_image_layer_list(image_layer_names)

        shape_layer_names = [
            layer.name
            for layer in self.viewer.layers
            if isinstance(layer, Shapes)
        ]
        self.widget.update_shape_layer_list(shape_layer_names)

    # get the new name of the images if image is inserted
    def _on_layer_inserted(self, event):
        """If the loaded data on the napari canvas changes"""
        layer = event.value
        if hasattr(layer, "events") and hasattr(layer.events, "name"):
            layer.events.name.connect(self._on_layer_renamed)
        self._on_layers_changed()

    # get the new name of the image layers if things are renamed
    def _on_layer_renamed(self, event):
        """If the loaded data name on the napari canvas changes do the following.

        Beyond re-enumerating the channel picker / recipe columns (via
        ``_on_layers_changed``), a rename of a TRACKED source layer must keep the
        two name-keyed caches in sync: ``_loaded_source_layers`` (name list) and
        ``_source_display_channels`` (``(name, levels)`` tuples).  The
        progressive<->native 2D/3D swap + its two sibling readers identify the
        source layers BY NAME from those caches; leaving them stale on a rename
        orphans the renamed layer, re-adds a DUPLICATE under the old name, and
        loses the layer's contrast/colormap on the next toggle.

        The renamed layer is identified by OBJECT IDENTITY against
        ``_source_layer_name_map`` (its old name is already gone from the layer by
        the time this fires).  A non-tracked layer (a ``⟨preview⟩``, box, or
        foreign layer) is left untouched here — only ``_on_layers_changed`` runs.
        Fully guarded — never raises from the Qt slot."""
        try:
            layer = getattr(event, "source", None)
            name_map = getattr(self, "_source_layer_name_map", None)
            if layer is not None and name_map is not None and layer in name_map:
                old = name_map[layer]
                new = getattr(layer, "name", old)
                if old != new:
                    names = self._loaded_source_layers
                    channels = self._source_display_channels
                    idx = None
                    if names:
                        for i, n in enumerate(names):
                            if n == old:
                                names[i] = new
                                idx = i
                                break
                    # Resync `_source_display_channels` at the SAME index (the two
                    # lists are index-aligned by construction).  Positional, not a
                    # second name-match: if napari COERCED the display name at add
                    # time (a load-time collision with a foreign layer), the tuple
                    # holds the un-coerced input name and a name-match would miss
                    # (audit F1).  Name-match fallback if the index is unavailable.
                    if channels:
                        if idx is not None and idx < len(channels):
                            try:
                                _cn, levels = channels[idx]
                                channels[idx] = (new, levels)
                            except Exception:
                                pass
                        else:
                            for i, entry in enumerate(channels):
                                try:
                                    cn, levels = entry
                                except Exception:
                                    continue
                                if cn == old:
                                    channels[i] = (new, levels)
                                    break
                    name_map[layer] = new
        except Exception:
            pass
        self._on_layers_changed()
    
    # ------------------------------------------------------------------
    # Data browser
    # ------------------------------------------------------------------

    def browse_source_folder(self):
        """Open a folder dialog and populate the file list with TIFFs."""
        from qtpy.QtWidgets import QFileDialog, QListWidgetItem
        folder = QFileDialog.getExistingDirectory(
            self.widget, "Select Source Data Folder",
            self.widget.source_folder_path.text() or "",
        )
        if not folder:
            return
        # Context change: land any pending source-edit save before the list is
        # rebuilt (flush before a context change).
        self._flush_source_state_save()
        self.widget.source_folder_path.setText(folder)
        self._scanned_files = scan_tiff_folder(folder)
        # The Order combo no longer describes any file from the new folder; a
        # fresh selection repoints it (prevents an override flip from keying onto
        # a file that is no longer in the list).
        self._axis_combo_source = None
        # The route cache is keyed by path, so a re-browse must drop it: a file
        # replaced in place, or a different folder holding the same filename,
        # would otherwise keep the previous header numbers.
        invalidate = getattr(self.widget, "invalidate_route_cache", None)
        if invalidate is not None:
            invalidate()

        # PROGRAMMATIC list rebuild — suspend the selection funnel so the clear +
        # bulk addItem can't restore a transient row (and force a fresh selection).
        self._suspend_source_restore = True
        try:
            self.widget.source_file_list.clear()
            self._selected_source_path = None   # a new folder = no active selection
            # ...and with it any recipe held for "the next selection": that
            # next selection is now in a different folder.
            self._drop_pending_recipe("you changed the source folder")

            try:
                import qtawesome as qta
                file_icon = qta.icon("mdi6.file-image-outline", color="#5ba4cf")
            except Exception:
                file_icon = None

            for info in self._scanned_files:
                item = QListWidgetItem()
                item.setData(Qt.UserRole, info["path"])
                item.setData(Qt.UserRole + 1, info["stem"])
                if file_icon is not None:
                    item.setIcon(file_icon)
                self.widget.source_file_list.addItem(item)
        finally:
            self._suspend_source_restore = False

        self._refresh_source_file_list()

        n = len(self._scanned_files)
        self.widget.source_status_label.setText(f"Found {n} TIFF file(s)")
        self.widget.source_status_label.setStyleSheet(
            "font-size: 10px; color: #5ba4cf;"
        )
        self._refresh_source_patch_counts()

    def on_source_file_selected(self, item):
        """Mouse ``itemClicked`` on a source row.  Delegates to :meth:`_select_source`
        which is IDEMPOTENT — the real single funnel is
        :meth:`_on_source_current_item_changed` (fires for mouse + keyboard +
        combo-pick), and it runs first, so this call is a no-op on the just-selected
        row (kept only so a direct call / any code path still resolves)."""
        path = item.data(Qt.UserRole)
        if path:
            self._select_source(path)

    def _select_source(self, path):
        """The ONE restore + register-at-select routine for EVERY selection route
        (allocate-at-select plus restore), reached through the single
        funnel :meth:`_on_source_current_item_changed`.

        IDEMPOTENT: a re-select of the ALREADY-selected source is a no-op (so a
        mouse click — which fires both ``currentItemChanged`` AND ``itemClicked``
        — restores exactly ONCE, and re-picking the same source never re-restores
        or re-saves).  ``_selected_source_path`` is set FIRST so a re-entrant call
        during the restore's widget writes also short-circuits.

        RESTORE the widgets from the source's registry row (saved alias / axis
        order / voxel), else the auto defaults.  A PASSIVE select does NOT
        persist ("worked-with" model): the source's row is created
        on the FIRST alias/voxel/axes EDIT, Prepare, or Extract, so browsing
        never accumulates rows and only sources the user worked with reappear.

        Widget-setting uses ``setText`` / ``setCurrentText`` / blocked ``setValue``,
        none of which emit the ``editingFinished`` / ``activated`` SAVE signals
        (those are user-only), so the restore can't re-fire a save and persist
        source B under alias A."""
        if not path or path == getattr(self, "_selected_source_path", None):
            return
        # Flush the OUTGOING source's pending DEBOUNCED save BEFORE restoring the
        # incoming one (Bug B, in-session variant): a switch within the 400ms save
        # window would otherwise restore the incoming source over the outgoing's
        # un-flushed recipe edits, and a later edit would then persist the stale
        # in-memory copy — permanently losing the outgoing source's last steps.
        try:
            self._flush_source_state_save()
        except Exception:
            pass
        self._selected_source_path = path
        ws = self.widget.output_file_path.text().strip()
        state = load_state(ws) if ws and os.path.isdir(ws) else None
        row = self._find_source_row(state, path) if state else None
        self._restore_source_widgets_for(path, row)
        # A recipe edited before any source was picked is adopted here (unless
        # this source already has one of its own) - see the helper.
        self._adopt_pending_recipe_for(path, row)
        # "Worked-with" persistence: a PASSIVE select does NOT write
        # to disk.  The source's registry row is created (path-keyed) on the
        # FIRST genuine action — an alias/voxel/axes EDIT, Prepare, or Extract —
        # so browsing / arrow-scrolling past a source never accumulates a row,
        # and only sources the user actually worked with reappear next session.
        # (This also makes B2-N1 moot: no path-less "browsed" row is ever
        # written, so a later export's path-keyed register_source create-or-
        # dedups without a duplicate.)

    def _on_source_current_item_changed(self, current, previous=None):
        """THE single selection funnel — fires for EVERY genuine selection route:
        mouse click, keyboard arrow, AND the ② combo-pick's ``setCurrentRow``
        (``currentItemChanged`` fires before ``currentRowChanged``, verified), so
        the full restore (alias + axis order + voxel + identity save) runs on all
        three (the FIX for the mouse-only-restore data-corruption bug).

        NOT gated on ``_source_sync_active``: the list↔combo mirror
        (:meth:`_on_source_row_changed`) writes the COMBO, never the list's current
        item, so it can't reach here; and a genuine combo-pick sets
        ``_source_sync_active`` but MUST restore.  Programmatic list REBUILD /
        POPULATE / BROWSE is guarded explicitly by ``_suspend_source_restore``
        (``addItem`` never auto-selects, and ``clear`` yields ``current is None``,
        so this is belt-and-suspenders)."""
        if current is None:
            return
        if getattr(self, "_suspend_source_restore", False):
            return
        path = current.data(Qt.UserRole)
        if path:
            self._select_source(path)

    def _apply_detected_axes_for_source(self, path):
        """Pre-fill the Order combo from the SELECTED source's header scan so the
        dropdown reflects THIS file — the signal :meth:`_resolve_prepare_image_axes`
        trusts to orient a Prepare WITHOUT a full memory Load.

        Zero extra I/O: reads the shape + tifffile axes captured by the folder
        scan (``_scanned_shape_for_path`` / ``_meta_axes_for_path``).  A prior
        user override for this file wins (mirrors the load path); otherwise the
        detected order is shown with a ``detected``/``guessed … — confirm`` hint,
        so a non-confident (heuristic) guess is VISIBLE and flippable before
        Prepare — never silently applied.  Sets it via ``apply_axis_order``
        (programmatic ``setCurrentText`` → does NOT emit ``activated`` → not
        mistaken for a user override).  Records the file the combo now describes
        in ``_axis_combo_source``.  No-op for a shape that is neither 3D nor 4D.
        """
        # A saved / user-confirmed order must win REGARDLESS of whether this
        # session has scanned the header.  After a restart `_scanned_files` is
        # empty (restore repopulates the ① list from the registry WITHOUT a
        # folder re-scan), so the shape guard below would early-return and leave
        # the restored order unapplied — the Order combo would then keep its
        # construction default (index 0 = "CZYX").  For a raw ZCYX source that
        # default silently swaps C/Z on extract (`_crop_source_for` reorders by
        # `_source_axes_resolved` = the combo), so the override check MUST run
        # before the shape guard, not after it.
        override = self._axis_order_overrides.get(_norm_path(path))
        if override is not None:
            self.widget.apply_axis_order(override, f"your choice: {override}")
            self._axis_combo_source = path
            return
        shape = self._scanned_shape_for_path(path)
        if len(shape) not in (3, 4):
            return
        try:
            detected = resolve_axes(shape, self._meta_axes_for_path(path))
        except Exception as exc:
            self._log("STEP1-AXES", f"selection axis detection failed: {exc}")
            return
        if detected is None:
            return
        if detected.axes == "ZYX":
            hint = "single channel (3D)"
        else:
            hint = (f"{'detected' if detected.confident else 'guessed'}: "
                    f"{detected.axes} · {detected.n_channels} ch — confirm")
        self.widget.apply_axis_order(detected.axes, hint)
        self._axis_combo_source = path

    # ------------------------------------------------------------------
    # Brick 2 — source-metadata round-trip (save-on-edit + restore-on-load)
    # ------------------------------------------------------------------
    @staticmethod
    def _find_source_row(state, path):
        """The registry row whose ``path`` matches *path* (normalized), or None."""
        if not state or not path:
            return None
        target = _norm_path(path)
        for s in state.get("sources", []):
            if _norm_path(s.get("path", "")) == target:
                return s
        return None

    # ---- session identity (Brick 4a) ---------------------------------
    def _resolve_loaded_source_id(self, path):
        """The registry ``source_id`` for *path*, or ``None`` when no row exists
        yet (a never-prepared, never-extracted source).

        Called at EVERY place that sets ``_loaded_source_file`` so the two stay
        in lock-step.  ``None`` is a legitimate answer: the readers below fall
        back to the path, which is exactly today's behaviour, and the id starts
        winning as soon as a row exists.  Never raises."""
        if not path:
            return None
        try:
            ws = self.widget.output_file_path.text().strip()
            if not ws or not os.path.isdir(ws):
                return None
            row = self._find_source_row(load_state(ws), path)
            return (row or {}).get("source_id") or None
        except Exception:      # noqa: BLE001 — identity is best-effort
            return None

    @staticmethod
    def _freshness_note(reasons):
        """The plain-language amber line for a freshness drift, or ``None``.

        Deliberately jargon-free (no "fingerprint" / "mtime" / "provenance"):
        these are notices a biologist reads mid-experiment, and neither of them
        needs an action."""
        reasons = list(reasons or [])
        if "unverifiable" in reasons:
            return ("⚠  This image file may have changed since it was "
                    "prepared — Step 1 is reading the original file to be "
                    "safe.  Re-Prepare to get the fast copy back.")
        if "size_drift" in reasons:
            return ("\u26a0  This image file has changed since it was prepared "
                    "\u2014 Step 1 is reading the original file, not the "
                    "prepared copy.")
        if "mtime_drift" in reasons:
            return ("\u26a0  This image file has a new date but the same "
                    "contents \u2014 Step 1 is still using the prepared copy.")
        return None

    def _set_source_status_amber(self, text):
        """Show *text* in the ② source status line in the "heads-up, not an
        error" amber.  Never a modal: these are notices the user can read and
        carry on past (Brick 4a a3/a4)."""
        try:
            self.widget.source_status_label.setText(text)
            self.widget.source_status_label.setStyleSheet(
                "font-size: 10px; color: #d4a843;")
        except Exception:      # noqa: BLE001 — a status line is never critical
            pass

    def _set_workspace_status_amber(self, text):
        """Show *text* under the ① workspace path box, in the amber that means
        "heads-up, not an error".

        Used when a Browse is REFUSED: the user picked a folder and the box did
        not move, so the reason has to be where they just clicked and not only
        in the log.  Naturally transient — the next `_update_workspace_status`
        (any real path change) overwrites it.
        """
        try:
            self.widget.workspace_status_label.setText(text)
            self.widget.workspace_status_label.setStyleSheet(
                "font-size: 10px; color: #d4a843;")
        except Exception:      # noqa: BLE001 — a status line is never critical
            pass

    def _set_raw_missing(self, missing):
        """Record whether the loaded source is being served from its prepared
        copy with the original file absent, and re-run the two button gates that
        depend on it (Prepare, "Process whole volume")."""
        self._raw_missing = bool(missing)
        fn = getattr(getattr(self, "widget", None),
                     "refresh_raw_missing_gates", None)
        if callable(fn):
            try:
                fn()
            except Exception:      # noqa: BLE001 — a gate refresh is cosmetic
                pass

    # ---- relink: a source file that moved / was renamed / was copied ----
    @staticmethod
    def _tiff_level0_shape(path):
        """The level-0 shape of a TIFF, read from its HEADER only (no pixels),
        or ``None``.  Used as the second half of the relink identity check."""
        shape, _axes = DataPreparationController._tiff_level0_shape_axes(path)
        return shape

    @staticmethod
    def _tiff_level0_shape_axes(path):
        """``(shape, axes)`` for a TIFF's level 0, from its HEADER only.

        The axes string is what lets the caller resolve a real ``(Z, Y, X)``
        instead of comparing an order-blind multiset."""
        try:
            import tifffile
            with tifffile.TiffFile(str(path)) as tf:
                series = tf.series[0]
                return (tuple(int(d) for d in series.shape),
                        getattr(series, "axes", None))
        except Exception:      # noqa: BLE001 -- identity probing never raises
            return None, None

    @staticmethod
    def _zyx_from_shape(shape, axes=None):
        """``(Z, Y, X)`` for a level-0 *shape*, dropping the channel axis.

        This is what identity should compare (Brick 4c item 1).  The earlier
        sorted-multiset compare could not tell a stack from its own transpose,
        and needed a singleton-drop hack to bridge a 3D raw against its 4D
        canonical store; resolving the axes removes both problems.  A clean
        tifffile axes string is trusted; otherwise
        :func:`packages.epicycle_largedata.axes.resolve_axes` infers the channel
        axis exactly as the loader does.  ``None`` when it cannot be resolved
        (the caller then falls back to :meth:`_shapes_agree`)."""
        if not shape:
            return None
        try:
            s = tuple(int(d) for d in shape)
        except Exception:      # noqa: BLE001
            return None
        ax = str(axes or "").upper()
        if "T" in ax:
            # A TIME series is not a stack we can identify by shape (Brick 4e
            # item 8): dropping T would silently compare one timepoint's volume
            # against a whole movie.  Refuse and let the caller fall back.
            return None
        if len(s) == 3:
            return s
        if len(s) != 4:
            return None
        ci = ax.index("C") if (len(ax) == 4 and "C" in ax) else None
        if ci is None:
            try:
                from packages.epicycle_largedata.axes import resolve_axes
                ci = resolve_axes(s, axes).channel_axis
            except Exception:      # noqa: BLE001
                ci = 0 if s[0] <= s[1] else 1
        if ci is None:
            return None
        return tuple(d for i, d in enumerate(s) if i != ci)

    @staticmethod
    def _zyx_for_file(shape, axes, row):
        """The picked file's ``(Z, Y, X)``, preferring the ROW's confirmed axis
        order when the file's own header does not state one.

        tifffile reports ``Q`` / ``S`` (or nothing) for plenty of real stacks,
        and the shape-based fallback then GUESSES the channel axis.  A user who
        confirmed ``ZCYX`` in the Order combo has already answered that question
        for this source, so their answer is used instead of a fresh guess --
        otherwise a legitimate move produced a false shape mismatch (Brick 4e
        item 5)."""
        ax = str(axes or "").upper()
        if len(ax) != len(shape or ()) or not set(ax) <= set("CZYXTS Q"):
            ax = ""
        hint = ax or (row or {}).get("source_axes")
        return DataPreparationController._zyx_from_shape(shape, hint or axes)

    def _row_zyx(self, workspace, row):
        """The ``(Z, Y, X)`` a registry *row* describes, or ``None``.

        Its prepared store's level-0 ``0/.zarray`` first (always the CANONICAL
        ``CZYX`` layout), then the row's own ``source_shape`` -- recorded at
        load, so a source that was EXTRACTED FROM but never PREPARED still has
        dimensions to be identified by (Brick 4c item 4).  Without that
        fallback the auto-relink was dead for every never-prepared source and
        the Locate modal could only ever offer "unknown before"."""
        zyx = self._zyx_from_shape(
            self._store_level0_shape(workspace, row), "CZYX")
        if zyx:
            return zyx
        recorded = (row or {}).get("source_shape")
        if isinstance(recorded, (list, tuple)) and len(recorded) == 3:
            try:
                return tuple(int(d) for d in recorded)
            except (TypeError, ValueError):
                return None
        return None

    def _persist_source_shape(self, path):
        """Record the loaded source's axis-resolved ``(Z, Y, X)`` on its
        registry row (additive ``source_shape``), if a row already exists.

        Deliberately never CREATES a row: loading a source has never registered
        it, and this must not start.  A header read only -- no pixels.  Called
        from both load finishes, beside the identity cache.  Never raises."""
        try:
            ws = self.widget.output_file_path.text().strip()
            if not path or not ws or not os.path.isdir(ws):
                return
            state = load_state(ws)
            row = self._loaded_source_row(state)
            if row is None:
                return
            zyx = self._zyx_from_shape(*self._tiff_level0_shape_axes(path))
            if not zyx or list(row.get("source_shape") or ()) == list(zyx):
                return
            row["source_shape"] = [int(d) for d in zyx]
            save_state(ws, state)
        except Exception as exc:   # noqa: BLE001 -- provenance is best-effort
            self._log("STEP1-LOAD", f"source_shape not recorded: {exc}")

    def _store_level0_shape(self, workspace, row):
        """The level-0 shape recorded in a row's prepared store (its ``0/.zarray``),
        or ``None``.  A cheap JSON read — the store is never opened."""
        ptr = (row or {}).get("prepared_store")
        if not ptr or not workspace:
            return None
        try:
            import json
            store = os.path.join(workspace, *str(ptr).split("/"))
            with open(os.path.join(store, "0", ".zarray"), encoding="utf-8") as f:
                return tuple(int(d) for d in json.load(f)["shape"])
        except Exception:      # noqa: BLE001
            return None

    @staticmethod
    def _shapes_agree(a, b):
        """Do two level-0 shapes describe the same stack?

        Compared as SORTED multisets with singleton axes dropped, deliberately:
        the raw TIFF's series[0] is in its ON-DISK order (e.g. ZCYX — and plain
        3D ZYX for a single-channel stack) while the prepared store's level 0 is
        the CANONICAL 4D layout (CZYX, with a length-1 channel axis).  An exact
        tuple compare would reject both genuine cases.  Every non-trivial axis
        agreeing is identity enough here: the byte-size check runs alongside it,
        and only rows whose file is MISSING are ever compared."""
        if not a or not b:
            return False
        aa = sorted(d for d in a if d != 1)
        bb = sorted(d for d in b if d != 1)
        return bool(aa) and aa == bb

    # ---- "Find the original file..." (the manual relink) ---------------
    @staticmethod
    def _locate_decision(new_size, new_shape, row_size, row_shape):
        """What to do with a file the user picked for an existing source row.

        Both shapes are the axis-resolved ``(Z, Y, X)`` (:meth:`_row_zyx` /
        :meth:`_zyx_from_shape`), so the compare is EXACT -- a stack and its own
        transpose are not the same image.

        * shape agrees              -> ``"relink"`` (silent; a re-export of the
          same stack legitimately has a different byte size)
        * shape differs, size agrees-> ``"confirm"`` (ask; using it anyway may
          show the wrong pixels)
        * neither agrees            -> ``"refuse"`` (not the same image)

        When either shape is unknown the size alone decides between confirm and
        refuse -- we never relink silently on a byte count."""
        shape_ok = (tuple(new_shape) == tuple(row_shape)
                    if (new_shape and row_shape) else False)
        size_ok = (row_size is not None and new_size is not None
                   and int(row_size) == int(new_size))
        if shape_ok:
            return "relink"
        return "confirm" if size_ok else "refuse"

    def _pick_source_file(self):
        """Ask the user for the original image file (``None`` if they cancel)."""
        from qtpy.QtWidgets import QFileDialog

        path, _filter = QFileDialog.getOpenFileName(
            self.widget, "Find the original image file", "",
            "Image files (*.tif *.tiff *.TIF *.TIFF);;All files (*)")
        return path or None

    @staticmethod
    def _shape_label(shape):
        """A shape for a user to read -- ``"4 x 8 x 8"``, or ``"unknown"``.

        Never the word ``None`` (Brick 4c item 4): a source that was never
        prepared has no recorded dimensions, and "(4 x 8 x 8 now, None before)"
        reads like a bug rather than an answerable question."""
        if not shape:
            return "unknown"
        try:
            return " × ".join(str(int(d)) for d in shape)
        except (TypeError, ValueError):
            return "unknown"

    def _confirm_shape_mismatch(self, basename, now_shape, before_shape):
        """The one modal in this flow, default **Cancel**."""
        from qtpy.QtWidgets import QMessageBox

        reply = QMessageBox.question(
            self.widget, "Is this the same image?",
            f"{basename} isn't the same shape as the image this source was "
            f"prepared from ({self._shape_label(now_shape)} now, "
            f"{self._shape_label(before_shape)} before).  "
            "Using it anyway may show you the wrong pixels.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return reply == QMessageBox.Yes

    def locate_source_for_row(self, source_id=None, source_path=None):
        """"Find the original file..." -- point a source row back at its image
        (Brick 4a a5).

        Reachable from the button in the (2) source card AND from the (2)
        source-list row context menu, so it works both when the source is loaded
        from its prepared copy and when the user simply knows a file moved.

        A shape match relinks silently; a shape mismatch asks (default Cancel);
        neither matching is refused without a dialog.  **Cancel leaves the row
        completely unchanged**, amber line and all.  Returns True on a relink."""
        try:
            ws = self.widget.output_file_path.text().strip()
            if not ws or not os.path.isdir(ws):
                self._set_source_status_amber(
                    "Set a workspace first, then find the original file.")
                return False
            state = load_state(ws)
            row = None
            if source_id:
                for r in state.get("sources", []):
                    if r.get("source_id") == source_id:
                        row = r
                        break
            if row is None and source_path:
                row = self._find_source_row(state, source_path)
            if row is None:
                row = self._loaded_source_row(state)
            if row is None:
                self._set_source_status_amber(
                    "Load a source first, then find its original file.")
                return False

            picked = self._pick_source_file()
            if not picked:
                return False                       # cancelled the picker

            new_size = os.path.getsize(picked)
            _pshape, _paxes = self._tiff_level0_shape_axes(picked)
            new_shape = self._zyx_for_file(_pshape, _paxes, row)
            row_size = (row.get("fingerprint") or {}).get("size")
            # Axis-resolved on BOTH sides, and the row's own recorded dimensions
            # stand in when it has no prepared store (Brick 4c item 4).
            row_shape = self._row_zyx(ws, row)
            verdict = self._locate_decision(
                new_size, new_shape, row_size, row_shape)

            if verdict == "refuse":
                self._set_source_status_amber(
                    "That doesn't look like the same image \u2014 it's a "
                    "different size.")
                return False
            if verdict == "confirm":
                if not self._confirm_shape_mismatch(
                        os.path.basename(picked), new_shape, row_shape):
                    return False                   # row untouched, amber stays

            # Brick 4e item 2: the picked path may ALREADY have a row of its
            # own -- most often the stub the silent relink left behind when it
            # found two equally-good candidates and (correctly) refused to
            # choose.  Relinking on top of that would leave TWO rows claiming
            # one file, which is the very duplicate this whole brick exists to
            # prevent.
            other = self._find_source_row(state, picked)
            if other is not None and other is not row:
                if not self._row_is_bare(state, other):
                    alias = (other.get("display_name")
                             or os.path.basename(str(other.get("path") or "")))
                    self._set_source_status_amber(
                        f"That file is already registered as {alias} — "
                        "choose it in the list instead.")
                    return False
                self._carry_stub_fields(other, row)
                try:
                    state["sources"].remove(other)
                except ValueError:
                    pass
                self._log("STEP1-LOAD",
                          "folded an empty stub row into "
                          f"{row.get('source_id')} on manual re-link")

            sid = row.get("source_id")
            if not sid or not relink_source(state, sid, picked, ws):
                return False
            save_state(ws, state)
            self._log("STEP1-LOAD", f"manual re-link: {sid} -> {picked}")
            self._refresh_after_relink(sid, picked)
            return True
        except Exception as exc:   # noqa: BLE001 -- never break the (2) surface
            self._log("STEP1-LOAD", f"locate source failed: {exc}")
            return False

    def _refresh_after_relink(self, source_id, path):
        """Post-relink UI settle: the raw is present again, so the two gated
        buttons come back and the (2) status says so."""
        # NOTE: `_loaded_source_id` is deliberately NOT touched here.  The
        # relink PRESERVES the row's id, so a loaded source already holds the
        # right one -- and the identity attribute is set/cleared only alongside
        # `_loaded_source_file` (the set-equality invariant, Brick 4a item 1).
        self._set_raw_missing(False)
        try:
            self.widget.source_status_label.setText(
                "\u2714  Now using " + os.path.basename(path))
            self.widget.source_status_label.setStyleSheet(
                "font-size: 10px; color: #6abf69;")
        except Exception:      # noqa: BLE001
            pass
        try:
            self._refresh_library_from_registry()
        except Exception:      # noqa: BLE001
            pass

    @staticmethod
    def _row_is_bare(state, row):
        """Is *row* an empty STUB — a row that exists only because a widget edit
        was saved against a path, with no work of its own behind it?

        Bare means: no prepared store, no processed store, never prepared, no
        patch cut from it, AND no preprocessing recipe.  Such a row carries only
        small, transferable settings (alias, numbering prefix, voxel size, axis
        order), which the relink CARRIES ACROSS before dropping the stub -- so
        folding it loses nothing (Brick 4c item 6).

        A RECIPE is different in kind (Brick 4e item 1): it is real authored
        work, sometimes many steps across several channels, and it is saved to a
        stub the moment the user edits it -- before the first Load.  Folding a
        row that holds one would delete it silently, so a row with recipe
        content is NOT bare and the relink bails as it did before."""
        if row.get("prepared_store") or row.get("prepared_at"):
            return False
        if row.get("processed_store"):
            return False
        if DataPreparationController._recipe_json_has_content(
                row.get("recipe")):
            return False
        sid = row.get("source_id")
        if sid:
            for pr in state.get("patches", []):
                if pr.get("source_id") == sid:
                    return False
        return True

    @staticmethod
    def _carry_stub_fields(stub, winner):
        """Move a folded stub row's settings onto the row that survives.

        Only the small, transferable ones -- the alias / numbering prefix / axis
        order / voxel size the user typed against the new path before the first
        Load.  A recipe is never carried because a row holding one is not bare
        in the first place (see :meth:`_row_is_bare`).

        ``source_axes`` matters as much as the alias (Brick 4e item 1): it is
        the order the user CONFIRMED in the Order combo, and dropping it would
        silently hand the next load a re-detected guess."""
        for key in ("display_name", "prefix", "source_axes"):
            if stub.get(key):
                winner[key] = stub[key]
        if stub.get("voxel_size") and stub.get("voxel_size_source") == "user":
            winner["voxel_size"] = stub["voxel_size"]
            winner["voxel_size_source"] = "user"

    def _relink_row_for(self, state, path, workspace):
        """Re-point an EXISTING registry row at *path* when *path* is plainly the
        same image under a new name / location (Brick 4a a1).

        This is the headline portability fix.  Session identity used to be the
        absolute raw path, so copying a workspace AND its data to another
        machine and browsing to the new folder created a SECOND row with a new
        ``source_id`` — and every patch already cut from that source stayed
        attached to the old id, greying ④ out for good.

        Candidates are ONLY rows whose own recorded ``path`` is ABSENT from disk,
        so two live files can never be merged into one row.  A candidate must
        then agree with the picked file on ALL THREE of:

        * **filename** — size and dimensions are NOT identity (Brick 4c item 1).
          Two acquisitions off the same microscope routinely share both, and
          binding row A's store to file B shows the wrong pixels with no warning
          at all.  A move, a copy and a whole-drive migration all keep the
          basename; a genuine RENAME is exactly the case that should go through
          "Find the original file…", which asks first.
        * **byte size** — the row's ``fingerprint``.
        * **(Z, Y, X)** — axis-resolved (:meth:`_zyx_from_shape`), so a stack
          and its own transpose no longer look identical.  When either side's
          axes cannot be resolved this degrades to the older
          :meth:`_shapes_agree` compare rather than guessing.

        And it must be the ONLY candidate: two rows that all match are an
        ambiguity we must not resolve by picking the first one, so the load
        falls through to today's new-row behaviour and logs why.

        Returns the reused ``source_id`` on a relink, else ``None``.  Mutates
        *state* in place; the caller persists.  Never raises."""
        try:
            if not path or not os.path.isfile(path):
                return None
            # Journey (a) ordering (Brick 4c item 6): on the new machine the
            # user often types an alias or a voxel size BEFORE the first Load.
            # That edit is saved with `upsert_source(path=...)`, which CREATES a
            # row for the new path -- and a bare "already ours" bail here then
            # left the real row (with the store and every patch) orphaned, ④
            # greyed out, for good.  A bare stub is folded into the relink
            # instead; a row with real work behind it still bails.
            bare = None
            existing = self._find_source_row(state, path)
            if existing is not None:
                if not self._row_is_bare(state, existing):
                    return None                  # already ours; nothing to do
                bare = existing
            size = os.path.getsize(path)
            shape, axes = self._tiff_level0_shape_axes(path)
            zyx = self._zyx_from_shape(shape, axes)
            base = os.path.basename(path)

            candidates = []
            for row in state.get("sources", []):
                rp = row.get("path")
                if not rp or os.path.isfile(rp):
                    continue                     # only MISSING files are candidates
                if os.path.basename(str(rp)) != base:
                    continue                     # the silent path needs the name
                fp = row.get("fingerprint") or {}
                if fp.get("size") is None or int(fp["size"]) != int(size):
                    continue
                row_zyx = self._row_zyx(workspace, row)
                row_store_shape = self._store_level0_shape(workspace, row)
                # Re-resolve per candidate: the row's own confirmed axis order
                # is the best hint we have for THIS row (item 5).
                zyx = self._zyx_for_file(shape, axes, row)
                if zyx and row_zyx:
                    if tuple(zyx) != tuple(row_zyx):
                        continue
                elif row_store_shape and shape:
                    if not self._shapes_agree(shape, row_store_shape):
                        continue
                elif row_zyx is None and row_store_shape is None:
                    # The ROW records no dimensions at all -- a legacy row for a
                    # source that was never prepared and predates
                    # `source_shape`.  Filename and byte size (both already
                    # matched) are everything we have; accept, rather than
                    # leaving the auto-relink permanently dead for those rows.
                    pass
                else:
                    # The row DOES know its dimensions, but we could not read
                    # the picked file's header (Brick 4e item 3).  That is not
                    # evidence of a match -- accepting here bound a row with a
                    # known shape to a file whose shape we never established.
                    continue
                candidates.append(row)

            if len(candidates) != 1:
                if candidates:
                    self._log(
                        "STEP1-LOAD",
                        f"no auto re-link for {base}: {len(candidates)} rows "
                        "match it equally well — use "
                        "“Find the original file…” to choose")
                return None
            winner = candidates[0]
            sid = winner.get("source_id")
            if not sid or not relink_source(state, sid, path, workspace):
                return None
            if bare is not None and bare is not winner:
                # Carry the edits the stub was created FOR onto the real row,
                # then drop it -- otherwise two rows would claim one path.
                self._carry_stub_fields(bare, winner)
                try:
                    state["sources"].remove(bare)
                except ValueError:
                    pass
                self._log("STEP1-LOAD",
                          f"folded an empty stub row into {sid} on re-link")
            return sid
        except Exception as exc:   # noqa: BLE001 — a relink is never load-critical
            self._log("STEP1-LOAD", f"relink probe skipped: {exc}")
        return None

    def _relink_row_for_path(self, path):
        """Run :meth:`_relink_row_for` against the live workspace and PERSIST a
        successful relink.  Returns the reused ``source_id`` or ``None``."""
        try:
            ws = self.widget.output_file_path.text().strip()
            if not ws or not os.path.isdir(ws):
                return None
            state = load_state(ws)
            sid = self._relink_row_for(state, path, ws)
            if sid:
                save_state(ws, state)
                self._log("STEP1-LOAD",
                          f"re-linked {sid} to {path} (same size + shape as its "
                          "missing original)")
            return sid
        except Exception as exc:   # noqa: BLE001
            self._log("STEP1-LOAD", f"relink skipped: {exc}")
            return None

    def _loaded_source_row(self, state):
        """The loaded source's registry row — ID-FIRST, path as the fallback.

        The id survives a relink (the row's ``path`` changed under us) and a
        missing raw file; the path fallback covers the window before any row
        exists (the id is then ``None``), keeping pre-Brick-4a behaviour."""
        sid = getattr(self, "_loaded_source_id", None)
        if sid and state:
            for s in state.get("sources", []):
                if s.get("source_id") == sid:
                    return s
        path = getattr(self, "_loaded_source_file", None)
        return self._find_source_row(state, path) if (state and path) else None

    def _loaded_source_path(self, state=None):
        """The loaded source's path as the REGISTRY records it (the row is
        authoritative after a relink), falling back to the session attribute.

        Used by the ``processed_store`` resolvers, which key a row lookup off a
        path: passing the row's own path keeps them correct even when the
        session attribute names a file that no longer exists."""
        try:
            if state is None:
                ws = self.widget.output_file_path.text().strip()
                state = load_state(ws) if ws and os.path.isdir(ws) else None
            row = self._loaded_source_row(state) if state else None
            p = (row or {}).get("path")
            if p:
                return p
        except Exception:      # noqa: BLE001
            pass
        return getattr(self, "_loaded_source_file", None)

    def _loaded_source_id_for(self, state):
        """The loaded source's id, resolved off *state* when the cached one is
        blank (no row existed at load time — e.g. the first extract creates it).
        Read-only: never writes the session attribute."""
        sid = getattr(self, "_loaded_source_id", None)
        if sid:
            return sid
        row = self._loaded_source_row(state)
        return (row or {}).get("source_id") or None

    @staticmethod
    def _file_fingerprint(path):
        """``{"size", "mtime"}`` for a live file (the store's freshness key), or
        None when the path is absent / unreadable (never raises)."""
        try:
            if path and os.path.isfile(path):
                return {"size": os.path.getsize(path),
                        "mtime": os.path.getmtime(path)}
        except OSError:
            pass
        return None

    def _restore_source_widgets_for(self, path, row):
        """Push a source's saved setup back into the ② widgets.

        Alias, axis order (seeding ``_axis_order_overrides`` so it counts as the
        user's confirmed order) and voxel (value + honest source label) come from
        the registry *row* when present, else the auto defaults.  All writes are
        programmatic and do NOT emit the ``editingFinished`` / ``activated`` save
        signals, so restore never re-persists (A9)."""
        w = self.widget
        stem = os.path.splitext(os.path.basename(path))[0]

        # --- alias ---------------------------------------------------------
        alias = (row.get("display_name") if row else None) or stem
        if hasattr(w, "display_name_edit"):
            w.display_name_edit.setText(alias)   # textChanged -> hint; NOT editingFinished
            # Keep the combo's auto/hand-typed discriminator coherent so a later
            # source switch updates (rather than protects) this restored alias.
            w._auto_alias = alias

        # --- axis order ----------------------------------------------------
        saved_axes = (row.get("source_axes") if row else None)
        if saved_axes:
            # Seed the override so the existing precedence treats the saved order
            # as the user's confirmed choice (kept across a re-detect).
            self._axis_order_overrides[_norm_path(path)] = str(saved_axes)
        self._apply_detected_axes_for_source(path)

        # --- voxel ---------------------------------------------------------
        try:
            effective, eff_src, baseline, _base_src = self._resolve_voxel_for_source(
                path, row)
            if hasattr(w, "set_detected_voxel_size"):
                w.set_detected_voxel_size(*baseline)   # the reset target
            if hasattr(w, "apply_voxel_size"):
                w.apply_voxel_size(*effective, source=eff_src)
        except Exception as exc:   # noqa: BLE001 — voxel restore is best-effort
            self._log("STEP1-RESTORE", f"voxel restore skipped: {exc}")

        # --- preprocessing recipe -------------------------------------
        # Load the saved recipe verbatim into the ② columns (an empty/absent
        # recipe resets the panel to empty).  It is validated against the real
        # channel count when a source loads (`update_image_layer_list`), since
        # channels are unknown at select-time, before Load.  Programmatic — no
        # `_notify_recipe_edited` re-fire (restore is read-only w.r.t. save, A9).
        if hasattr(w, "set_recipe_from_json"):
            try:
                w.set_recipe_from_json(row.get("recipe") if row else None)
            except Exception as exc:   # noqa: BLE001 — recipe restore best-effort
                self._log("STEP1-RESTORE", f"recipe restore skipped: {exc}")

    def _detected_voxel_cached(self, path):
        """``read_voxel_spacing(path)`` MEMOIZED per ``(normalized path, mtime)``
        (grill FIX A): arrowing a folder — or arrow-back — opens each source
        header at most once, and a file whose mtime changed is re-detected.  A
        cheap ``getmtime`` stat replaces a full header parse on a cache hit."""
        try:
            key = _norm_path(path)
            mtime = (os.path.getmtime(path)
                     if path and os.path.isfile(path) else None)
        except OSError:
            key, mtime = _norm_path(path), None
        cache = self._voxel_detect_cache
        hit = cache.get(key)
        if hit is not None and hit[0] == mtime:
            return hit[1]
        val = read_voxel_spacing(path)
        cache[key] = (mtime, val)
        return val

    def _resolve_voxel_for_source(self, path, row, store_meta=None):
        """The effective voxel + its honest source (A7 precedence).

        Returns ``(effective, effective_source, baseline, baseline_source)`` where
        *effective* is what the spins show and *baseline* is what the ``↺`` reset
        restores.  Precedence:

          1. registry ``voxel_size`` when ``voxel_size_source == "user"`` — the
             SOLE authority for the spins + export;
          2. the store NGFF level-0 ``scale`` (only when non-unit) — display-only;
          3. ``read_voxel_spacing`` header detect (only when non-unit, MEMOIZED);
          4. unknown ``(1, 1, 1)``.

        PERF (grill FIX A): a ``"user"`` override SHORT-CIRCUITS the header
        detect entirely — ``read_voxel_spacing`` is never called for a
        user-overridden source (the detect would be discarded by precedence
        anyway), so selecting such a source opens ZERO file headers.  A
        header-detected baseline for a NON-user source is memoized per path.  The
        store NGFF ``scale`` is a cheap in-memory read (no file open)."""
        # (2) store scale is a cheap in-memory read — compute it up front so a
        # user-override source can still expose a non-unit store baseline for
        # ↺ WITHOUT opening the raw header.
        store_baseline = None
        if store_meta is not None:
            v = self._voxel_from_store_meta(store_meta)
            if v is not None and not self._is_unit_voxel(v):
                store_baseline = v

        # (1) USER override wins the spins AND short-circuits the header detect:
        # NO `read_voxel_spacing` here (zero opens on select for a user source).
        # The ↺ baseline is the store scale when present, else unknown — the raw
        # header is never opened just to select a user-overridden source.
        if row and row.get("voxel_size_source") == "user" and row.get("voxel_size"):
            try:
                uv = tuple(float(x) for x in row["voxel_size"])
                if len(uv) == 3 and all(x > 0 for x in uv):
                    base = store_baseline or (1.0, 1.0, 1.0)
                    base_src = "detected" if store_baseline is not None else "unknown"
                    return uv, "user", base, base_src
            except (TypeError, ValueError):
                pass

        # No user override -> detected baseline: (2) store scale, else (3) the
        # MEMOIZED header detect, else (4) unknown.
        baseline = store_baseline
        if baseline is None:
            v = self._detected_voxel_cached(path)
            if v is not None and not self._is_unit_voxel(v):
                baseline = v
        baseline_source = "detected" if baseline is not None else "unknown"
        if baseline is None:
            baseline = (1.0, 1.0, 1.0)
        return baseline, baseline_source, baseline, baseline_source

    @staticmethod
    def _is_unit_voxel(v):
        """True for a ``(1, 1, 1)`` isotropic spacing (== "unknown", not a real
        physical value)."""
        try:
            return all(abs(float(x) - 1.0) < 1e-9 for x in v)
        except (TypeError, ValueError):
            return True

    @staticmethod
    def _voxel_from_store_meta(meta):
        """The ``(z, y, x)`` voxel from a store's NGFF level-0 ``scale``
        (``[1, z, y, x]`` for CZYX, ``[z, y, x]`` for ZYX), or None."""
        try:
            scales = meta.get("scales")
            if not scales:
                return None
            scale = scales[0]
            if len(scale) >= 3:
                return (float(scale[-3]), float(scale[-2]), float(scale[-1]))
        except (TypeError, ValueError, IndexError, AttributeError):
            pass
        return None

    def _saved_display_name_for(self, path):
        """The registry alias for *path*, or None (used by the ② combo so a
        pick restores the SAVED alias, not the bare stem)."""
        if not path:
            return None
        ws = self.widget.output_file_path.text().strip()
        if not ws or not os.path.isdir(ws):
            return None
        row = self._find_source_row(load_state(ws), path)
        return row.get("display_name") if row else None

    def source_alias_map(self):
        """``{normalized_path: alias}`` for every registered source with an alias,
        so the ② combo can label a registered file by its saved name."""
        ws = self.widget.output_file_path.text().strip()
        if not ws or not os.path.isdir(ws):
            return {}
        out = {}
        for s in load_state(ws).get("sources", []):
            p, a = s.get("path"), s.get("display_name")
            if p and a:
                out[_norm_path(p)] = a
        return out

    # ---- save-on-edit (debounced + flushed) --------------------------------
    def _queue_source_save(self, **upsert_kwargs):
        """Stage a source-registry edit: buffer the
        ``upsert_source`` kwargs (captured against the CURRENT workspace) and arm
        the debounce timer.  A workspace change between edits flushes the old
        buffer first so a write never lands in the wrong slot."""
        ws = self.widget.output_file_path.text().strip()
        if not ws:
            return
        if self._pending_source_edits and self._pending_source_ws != ws:
            self._flush_source_state_save()
        self._pending_source_ws = ws
        self._pending_source_edits.append(dict(upsert_kwargs))
        self._state_save_timer.start(400)

    def _flush_source_state_save(self, *_):
        """Apply + persist the buffered source edits NOW (timer fire or a forced
        flush before a context change).  RE-READS the on-disk state and layers
        the edits on top, so a concurrent Prepare / Export write is preserved,
        then one atomic ``save_state``.

        The buffer is cleared ONLY on a successful write (grill FIX 5): if
        ``save_state`` raises (disk full / permissions), the pending edits are
        KEPT so the next flush / edit retries them instead of silently dropping a
        user's rename or voxel."""
        edits = self._pending_source_edits
        ws = self._pending_source_ws
        if not edits or not ws:
            self._pending_source_edits = []
            self._pending_source_ws = None
            try:
                self._state_save_timer.stop()
            except Exception:
                pass
            return
        try:
            state = load_state(ws)   # fresh — never clobber a concurrent write
            for kw in edits:
                try:
                    upsert_source(state, **kw)
                except ValueError:
                    continue
            save_state(ws, state)
        except Exception as exc:   # noqa: BLE001 — persistence is non-critical here
            # KEEP the buffer (do not clear) so the edit is not lost; a later
            # flush / edit retries it.  Shown INLINE too: a silent failed save
            # is indistinguishable from a successful one.
            # DEDUPED per failure STREAK: a genuinely-unavailable workspace (a
            # disconnected share) would otherwise re-log the identical line on
            # every retry and scroll the Diagnostics pane away.  The red inline
            # status stays up for the whole streak either way.
            fails = getattr(self, "_state_save_fail_streak", 0)
            if fails == 0:
                self._report_state_problem(
                    "STEP1-SAVE",
                    f"source save flush FAILED (kept pending, retrying): {exc}")
            self._state_save_fail_streak = fails + 1
            # ... and RESCHEDULE with BACKOFF.  "A later flush / edit retries
            # it" was only half true: nothing re-armed the timer, so a transient
            # failure (OneDrive lock, a momentarily full disk) left the last
            # edit before a lull unsaved until the user happened to edit again.
            # 4s doubling to a 30s cap, so a persistent failure settles into a
            # slow poll instead of hammering an unreachable path forever.
            delay = min(_STATE_SAVE_RETRY_MAX_MS,
                        _STATE_SAVE_RETRY_MS * (2 ** fails))
            try:
                self._state_save_timer.start(int(delay))
            except Exception:
                pass
            return
        self._pending_source_edits = []
        self._pending_source_ws = None
        try:
            self._state_save_timer.stop()
        except Exception:
            pass
        # A save just SUCCEEDED, so any red `_report_state_problem` banner left
        # over from an earlier transient failure is now a lie — it used to
        # persist for the rest of the session.  Restore the widget's own ①
        # status renderer (the one the workspace-path field drives) rather than
        # inventing a second wording that could drift from it.
        fails = getattr(self, "_state_save_fail_streak", 0)
        if fails:
            # Close the streak explicitly: the user saw ONE "FAILED (retrying)"
            # line and then silence, which reads like it never recovered.
            self._log("STEP1-SAVE", f"source save succeeded after {fails} retries")
            self._state_save_fail_streak = 0
        self._clear_state_problem()

    def on_display_name_edited(self):
        """``display_name_edit.editingFinished`` — persist the alias (last-wins)
        against the currently-selected source."""
        path = getattr(self, "_selected_source_path", None)
        if not path:
            return
        name = self.widget.display_name_edit.text().strip()
        if not name:
            return
        self._queue_source_save(path=path, display_name=name)
        # Relabel the ② dropdown item LIVE from what was typed.  The save is
        # debounced (~400ms), so a registry-driven `_rebuild_source_combo` here
        # would repaint the STALE pre-flush alias/stem; setItemText updates the
        # label in place and emits NO `activated` (no spurious re-selection).
        # currentIndex tracks the selected source row (`_rebuild_source_combo`
        # keeps them in lockstep), and this method already targets that source.
        cb = self.widget.source_combo
        idx = cb.currentIndex()
        if idx >= 0:
            cb.setItemText(idx, name)

    def on_recipe_edited(self):
        """Persist the ② preprocessing recipe against the selected source
        (last-wins, debounced).  Called by the widget after any
        recipe mutation (add / remove / reorder / param / geometry config).

        Stored as the recipe JSON dict, which is NEVER None — clearing a recipe
        yields an empty-but-present dict, so `upsert_source`'s skip-None does not
        leave a stale recipe on disk.  Creating the row
        here (path-keyed) is what makes a source with only a recipe count as
        "worked-with" (it reappears in the ② list next session).

        A recipe edit ALSO invalidates any shown preview — mirroring the box
        slider handlers, it bumps the stale-token (an in-flight compute for the
        OLD recipe is discarded), removes any ⟨preview⟩ processed-crop layer, and
        greys the histogram with a "recipe changed — re-Preview" hint.  It NEVER
        recomputes live; the user re-clicks Preview (existing background thread).
        """
        self._invalidate_preview_on_recipe_edit()
        get_recipe = getattr(self.widget, "get_recipe_json", None)
        if not callable(get_recipe):
            return
        try:
            recipe_json = get_recipe()
        except Exception as exc:   # noqa: BLE001 — persistence is non-critical
            self._log("STEP1-RECIPE", f"recipe save skipped: {exc}")
            return
        path = getattr(self, "_selected_source_path", None)
        if not path:
            # Nothing selected: HOLD the recipe for the next selection instead of
            # dropping it, and say so plainly.  An EMPTY recipe is not worth
            # holding (it would only produce noise on the next select).
            if self._recipe_json_has_content(recipe_json):
                self._pending_recipe_for_next_selection = recipe_json
                self._set_preview_status(
                    "Select a source in the list to keep this recipe with it.",
                    ok=False)
            else:
                self._pending_recipe_for_next_selection = None
            return
        self._pending_recipe_for_next_selection = None
        self._queue_source_save(path=path, recipe=recipe_json)

    @staticmethod
    def _recipe_json_has_content(raw) -> bool:
        """True when a recipe JSON dict carries any real work (at least one
        per-channel step, or a geometry step).  The serialized form always has
        the ``channels`` / ``geometry`` keys, so plain truthiness says nothing."""
        if not isinstance(raw, dict):
            return False
        if raw.get("geometry"):
            return True
        return any(bool(ch) for ch in (raw.get("channels") or []))

    @staticmethod
    def _recipe_json_channel_count(raw) -> int:
        """How many CHANNELS a recipe JSON actually carries steps for -- the
        index of its last non-empty channel list, plus one.  A recipe the user
        only touched on channel 0 counts as 1 even on a 4-channel source, so it
        can still be adopted anywhere."""
        if not isinstance(raw, dict):
            return 0
        channels = raw.get("channels") or []
        for i in range(len(channels) - 1, -1, -1):
            if channels[i]:
                return i + 1
        return 0

    def _known_channel_count_for(self, path):
        """Channel count for *path* from the header SCAN (no extra I/O), or
        None when it is not known yet -- unknown must never block anything."""
        try:
            shape = self._scanned_shape_for_path(path)
            if len(shape) not in (3, 4):
                return None
            detected = resolve_axes(shape, self._meta_axes_for_path(path))
            if detected is None:
                return None
            return int(detected.n_channels)
        except Exception:   # noqa: BLE001 -- advisory only
            return None

    def _drop_pending_recipe(self, reason: str) -> None:
        """Forget a HELD recipe on a context change (a new source folder, a new
        workspace) and RESET the recipe panel with it, so the user is not left
        editing a phantom recipe belonging to the previous context.  Silent when
        nothing was held."""
        if getattr(self, "_pending_recipe_for_next_selection", None) is None:
            return
        self._pending_recipe_for_next_selection = None
        setter = getattr(self.widget, "set_recipe_from_json", None)
        if callable(setter):
            try:
                setter(None)
            except Exception as exc:   # noqa: BLE001 -- best-effort
                self._log("STEP1-RECIPE", f"recipe panel not reset: {exc}")
        self._set_preview_status(
            f"The unsaved recipe was discarded ({reason}).", ok=False)

    def _adopt_pending_recipe_for(self, path, row):
        """Give the recipe that was edited with nothing selected to the source
        the user just picked (design: never lose a user's work, never overwrite
        theirs).  A recipe already saved on that source's row ALWAYS wins — the
        held one is then discarded with a one-line status.  Called at the end of
        :meth:`_select_source`, after the row's own recipe has been restored."""
        pending = getattr(self, "_pending_recipe_for_next_selection", None)
        if pending is None:
            return
        self._pending_recipe_for_next_selection = None
        alias = ((row or {}).get("display_name")
                 or os.path.splitext(os.path.basename(path))[0])
        if self._recipe_json_has_content((row or {}).get("recipe")):
            self._set_preview_status(
                f"{alias} already has a saved recipe — kept it; "
                "the unsaved one was discarded.", ok=False)
            return
        # A recipe built for MORE channels than this source has would silently
        # lose the extra channels' steps (`normalize_recipe` clamps).  Say so
        # instead.  An unknown count never blocks.
        want = self._recipe_json_channel_count(pending)
        have = self._known_channel_count_for(path)
        if have is not None and want > have:
            self._set_preview_status(
                f"The unsaved recipe was built for {want} channels and "
                f"{alias} has {have} — it was discarded rather than trimmed.",
                ok=False)
            return
        setter = getattr(self.widget, "set_recipe_from_json", None)
        if callable(setter):
            try:
                setter(pending)
            except Exception as exc:   # noqa: BLE001 — best-effort
                self._log("STEP1-RECIPE", f"pending recipe not applied: {exc}")
                return
        self._queue_source_save(path=path, recipe=pending)
        self._set_preview_status(f"Recipe kept with {alias}.", ok=True)

    def on_voxel_edited(self):
        """Voxel spin ``editingFinished`` (a REAL edit only — the widget gates on
        its dirty flag) — persist the spacing + its honest type.
        A deliberate ``(1, 1, 1)`` is typed as ``"unknown"``, never ``"user"``."""
        path = getattr(self, "_selected_source_path", None)
        if not path:
            return
        v = self._prepare_voxel_size()   # (z, y, x) or None for the 1/1/1 default
        if v is None:
            self._queue_source_save(
                path=path, voxel_size=[1.0, 1.0, 1.0], voxel_size_source="unknown")
        else:
            self._queue_source_save(
                path=path, voxel_size=list(v), voxel_size_source="user")
        self._reapply_display_physical_scale()

    def on_voxel_reset(self):
        """``↺`` reset — CLEAR the user override: persist the DETECTED baseline
        typed honestly (``"detected"`` for a real spacing, ``"unknown"`` for the
        ``(1, 1, 1)`` default), so ``voxel_size_source`` is no longer ``"user"``
        and the round-trip re-detects instead of re-serving the cleared value.

        LAZY re-detect (grill FIX A companion): selecting a ``"user"``-override
        source is deliberately header-free, so the widget's ``_detected_voxel_size``
        baseline is the store NGFF scale when present, else the ``(1, 1, 1)``
        unknown placeholder — NOT the raw header spacing.  ↺ is an EXPLICIT user
        action, so when that baseline is the unit placeholder AND a source file
        is selected we open the header ONCE here (memoized via
        ``_detected_voxel_cached``) to recover the real detected spacing.  A
        non-unit result becomes the reset value: persisted typed ``"detected"``
        AND reflected in the widget (baseline + spins) so the ↺ shows the
        re-detected value instead of a stale ``(1, 1, 1)``.  A unit/None header
        keeps the ``"unknown"`` behaviour.  SELECT stays header-free (FIX A
        intact) — the raw header is opened only on this explicit ↺."""
        path = getattr(self, "_selected_source_path", None)
        if not path:
            return
        baseline = getattr(self.widget, "_detected_voxel_size", (1.0, 1.0, 1.0))
        # Only ↺ (explicit) may open the header — and only when the select-time
        # baseline is the unit placeholder (FIX A left it unit for a user row).
        if self._is_unit_voxel(baseline):
            detected = self._detected_voxel_cached(path)
            if detected is not None and not self._is_unit_voxel(detected):
                baseline = tuple(float(x) for x in detected)
                # Reflect the re-detected baseline back into the widget so the
                # ↺ shows it: the reset target the widget restores + the spins.
                try:
                    self.widget.set_detected_voxel_size(*baseline)
                    self.widget.apply_voxel_size(*baseline, source="detected")
                except Exception:   # noqa: BLE001 — UI reflect is best-effort
                    pass
        if self._is_unit_voxel(baseline):
            self._queue_source_save(
                path=path, voxel_size=[1.0, 1.0, 1.0], voxel_size_source="unknown")
        else:
            self._queue_source_save(
                path=path, voxel_size=[float(x) for x in baseline],
                voxel_size_source="detected")
        self._reapply_display_physical_scale()

    def _reapply_display_physical_scale(self):
        """Re-apply ``layer.scale`` + ``units`` to the LIVE raw-derived display
        layers (and re-render the box overlays) when the voxel spins change.

        THE gating fix for the physical display: placement + the box overlays
        read the voxel LIVE (``_preview_spacing`` / ``_apply_overlay_voxel``), but
        an image layer's ``scale`` is baked once at add time.  Without this, a
        post-load voxel edit (the routine correction for a no-metadata source)
        desyncs the two — a box then renders UNDER the cursor but stores a wrong
        voxel index, so extraction silently slices the wrong region.  Keeping the
        display scale == the live voxel closes that (a box's world = its voxel ×
        the SAME value placement divided by).

        Scope = the RAW source layers (``scale == voxel``, factor 1) + the
        large-source overview (``scale == factor × voxel``) — the placement
        anchors.  The processed store keeps its OWN baked spacing (untouched — a
        raw-voxel edit does not change the store's physical extent).  The ③ box,
        ④ location, and ② preview overlays are re-rendered only when already
        active (they live-read the voxel).  Best-effort + guarded — never raises
        out of the voxel-edit slot."""
        vox = self._preview_spacing()
        units = _units_for_voxel(vox)
        scale = tuple(float(v) for v in vox)
        viewer = getattr(self, "viewer", None)
        # (1) raw source native layers — level-0 is full-res, so scale == voxel.
        if viewer is not None:
            names = set(self._loaded_source_layers or [])
            if names:
                for layer in list(getattr(viewer, "layers", []) or []):
                    try:
                        if getattr(layer, "name", None) in names:
                            _apply_layer_scale(layer, scale)
                            _apply_layer_units(layer, units)
                    except Exception:
                        pass
        # (2) large-source overview — scale = factor × voxel (per axis).
        bt = getattr(self, "_box_tool", None)
        ov_layers = list(getattr(bt, "_overview_layers", None) or [])
        full_shape = getattr(bt, "_full_shape", None)
        if ov_layers and full_shape is not None:
            try:
                from ._box_tool import overview_factors
                fac = overview_factors(full_shape)
                _apply_layer_scale(
                    ov_layers,
                    tuple(float(f) * float(v) for f, v in zip(fac, vox)))
                _apply_layer_units(ov_layers, units)
            except Exception:
                pass
        # (3) re-render the overlays that live-read the voxel (only when active).
        try:
            if bt is not None:
                bt.redraw()
        except Exception:
            pass
        try:
            if getattr(self, "_visible_patch_locations", None):
                self._rebuild_loc_boxes()
        except Exception:
            pass
        try:
            if getattr(self, "_preview_box", None) is not None:
                self._render_preview_box()
        except Exception:
            pass
        # A voxel edit changed layer.units (pixel→µm) — refresh the live scale bar
        # so its label follows (it doesn't auto-redraw on layer.units alone).
        try:
            self._apply_live_scale_bar()
        except Exception:
            pass
        try:
            self._diag_physical_layers("after voxel-edit re-apply")
        except Exception:
            pass

    def _diag_physical_layers(self, context: str) -> None:
        """[PHYS] diagnostic (TEMP): log EVERY viewer layer's name / scale /
        units + the resolved voxel, so the physical-display scale and the napari
        'Inconsistent units across layers' warning can be traced from the
        Diagnostics pane.  Best-effort — never raises.  STRIP before merge."""
        try:
            vox = self._preview_spacing()
        except Exception:
            vox = None
        try:
            src = list(self._loaded_source_layers or [])
        except Exception:
            src = []
        self._log("PHYS", f"=== {context} === voxel={vox} source_layers={src}")
        viewer = getattr(self, "viewer", None)
        if viewer is None:
            self._log("PHYS", "  (no viewer)")
            return
        try:
            layers = list(getattr(viewer, "layers", []) or [])
        except Exception:
            layers = []
        if not layers:
            self._log("PHYS", "  (no layers)")
        for layer in layers:
            try:
                nm = getattr(layer, "name", "?")
                sc = tuple(round(float(s), 4) for s in getattr(layer, "scale", ()))
                un = tuple(str(u) for u in (getattr(layer, "units", ()) or ()))
                is_src = nm in src
                self._log("PHYS",
                          f"  layer={nm!r} src={is_src} scale={sc} units={un}")
            except Exception as exc:
                self._log("PHYS", f"  layer <introspect failed: {exc}>")

    # ---- restore-on-load + deferred filesystem migration -------------------
    def _maybe_restore_step1_sources(self, *_):
        """Run the once-per-workspace-per-session source restore + store
        reconciliation for the current workspace.  Guarded exactly
        like the layout-migration offer: fires from the workspace ``textChanged``
        and from ``on_step_enter``, at most once per valid workspace path."""
        ws = self.widget.output_file_path.text().strip()
        if not ws or not os.path.isdir(ws):
            return
        key = os.path.normcase(os.path.normpath(ws))
        if key in self._step1_sources_restored:
            return
        self._step1_sources_restored.add(key)
        # Brick 5 fix 4: this is synchronous main-thread I/O on step-enter /
        # workspace-set — `load_state`, a `largedata/` listdir and a per-store
        # `.zattrs` read each.  Whether it crosses the 100 ms budget depends on
        # how many prepared stores a real workspace holds, which cannot be
        # judged headlessly, so MEASURE it rather than guess at an indicator:
        # run with `EPICYCLE_SYNC_DEBUG=1` and read the `[STEP1-PERF]` line.  If
        # it is over budget on a real workspace, the fix is a worker + ②'s bar,
        # not a `processEvents` pump (this runs during a step transition).
        _perf = os.environ.get("EPICYCLE_SYNC_DEBUG") == "1"
        _t0 = time.perf_counter() if _perf else 0.0
        try:
            self._restore_step1_sources(ws)
        except Exception as exc:   # noqa: BLE001 — restore is best-effort
            self._log("STEP1-RESTORE", f"restore failed: {exc}")
        finally:
            if _perf:
                self._log(
                    "STEP1-PERF",
                    f"source restore + store reconcile "
                    f"{(time.perf_counter() - _t0) * 1000.0:.1f} ms")

    def _restore_step1_sources(self, ws):
        """Reconcile the registry against the ``largedata/`` stores (the filesystem
        half of the v1->v2 migration — deferred out of ``load_state`` so keystroke
        loads stay I/O-free) and repopulate ② from the registry.

        Reconcile (:meth:`_reconcile_sources_with_stores`): abs->rel pointer
        rewrite, per-store ``.zattrs`` scan (``read_store_source``, never the
        shared manifest), pointer + fingerprint backfill (freshest-of-two by
        ``source_mtime``, never cross a path onto another source's store),
        orphan-store synth (dedup against existing ``sources[].path``), dead-store
        null.  A changed state is saved back."""
        # A corrupt state file is a TOTAL loss of Step-1 state (sources, voxels,
        # recipes, store pointers, patch registry).  ``load_state`` quarantines
        # the bad file; the callback makes sure the user SEES that happened
        # instead of silently continuing on an empty workspace.
        state = load_state(              # auto-migrates the field renames
            ws,
            on_corrupt=lambda msg: self._report_state_problem("STEP1-RESTORE", msg),
        )
        changed = self._reconcile_sources_with_stores(ws, state)
        if changed:
            save_state(ws, state)
        self._populate_source_list_from_registry(state)
        self._log("STEP1-RESTORE",
                  f"restored {len(state.get('sources', []))} source(s) "
                  f"(reconciled={changed})")

    def _reconcile_sources_with_stores(self, ws, state):
        """Pure-ish reconciliation of the registry against ``largedata/*.zarr``.  Mutates *state* in place; returns True if it
        changed.  NEVER points a row at another source's store."""
        largedata = self._largedata_dir(ws)
        changed = False

        # 1. abs -> workspace-relative pointer rewrite (portability).  Resolve a
        #    stale absolute pointer to the same-named store under THIS workspace
        #    (a moved/copied workspace) — but ONLY when that store's OWN recorded
        #    ``epicycle_source.path`` matches the row's ``path`` (grill FIX 3: a
        #    basename collision must NOT cross-bind A's row to B's pixels).  Else
        #    NULL it and let the per-store scan (step 3) re-derive the pointer from
        #    each store's own recorded source path.
        for s in state.get("sources", []):
            ptr = s.get("prepared_store")
            if ptr and os.path.isabs(str(ptr)):
                base = os.path.basename(str(ptr).rstrip("/\\"))
                candidate = os.path.join(largedata, base)
                rebound = False
                if base and os.path.isdir(candidate):
                    prov = read_store_source(candidate) or {}
                    cand_src = prov.get("path")
                    row_src = s.get("path")
                    if (cand_src and row_src
                            and _norm_path(cand_src) == _norm_path(row_src)):
                        s["prepared_store"] = os.path.relpath(
                            candidate, ws).replace(os.sep, "/")
                        rebound = True
                if not rebound:
                    s["prepared_store"] = None
                changed = True

        # 2. scan every store's OWN provenance (per-store .zattrs, not the shared
        #    last-writer-wins manifest).
        stores = []
        if os.path.isdir(largedata):
            for name in sorted(os.listdir(largedata)):
                if not name.endswith(".zarr"):
                    continue
                store_abs = os.path.join(largedata, name)
                prov = read_store_source(store_abs) or {}
                stores.append((store_abs, prov))

        # 3. group stores by their recorded SOURCE path; back-fill / synth per
        #    path, picking the FRESHEST store (by source_mtime) when two exist.
        by_path = {}
        for store_abs, prov in stores:
            sp = prov.get("path")
            if not sp:
                continue
            by_path.setdefault(_norm_path(sp), []).append((store_abs, prov))

        for norm_sp, lst in by_path.items():
            lst.sort(key=lambda t: (t[1].get("source_mtime") or 0), reverse=True)
            store_abs, prov = lst[0]
            rel = os.path.relpath(store_abs, ws).replace(os.sep, "/")
            matched = [s for s in state.get("sources", [])
                       if _norm_path(s.get("path", "")) == norm_sp]
            if matched:
                for s in matched:
                    if s.get("prepared_store") != rel:
                        s["prepared_store"] = rel
                        changed = True
                    if not s.get("fingerprint") and prov.get("source_mtime") is not None:
                        s["fingerprint"] = {"size": prov.get("source_size"),
                                            "mtime": prov.get("source_mtime")}
                        changed = True
                    if not s.get("source_axes") and prov.get("source_axes"):
                        s["source_axes"] = prov.get("source_axes")
                        changed = True
            elif any(self._row_claims_store(r, ws, store_abs)
                     for r in state.get("sources", [])):
                # Brick 4a a6: this store is already CLAIMED by a row pointer,
                # it just records an older source path (the row was relinked and
                # we deliberately never rewrite a store's `.zattrs` — the
                # recorded path is a hint, not identity).  Synthesizing here
                # would produce a duplicate row for the same physical source.
                pass
            else:
                # orphan store -> synthesize a row (dedup against existing paths
                # is implicit: `matched` empty means no row has this path).
                v = prov.get("voxel_size")
                # HONEST typing (grill FIX 5 principle): an orphan store's baked
                # voxel is NOT a genuine user edit -> "detected" for a real value,
                # "unknown" for 1/1/1 / absent (never silently "user").
                v_source = ("detected"
                            if _classify_voxel_source(v) == "user" else "unknown")
                state.setdefault("sources", []).append({
                    "source_id": _next_source_id(state.get("sources", [])),
                    "path": prov.get("path"),
                    "display_name": os.path.splitext(
                        os.path.basename(str(prov.get("path", ""))))[0],
                    "source_axes": prov.get("source_axes"),
                    "voxel_size": v,
                    "voxel_size_source": v_source,
                    "prepared_store": rel,
                    "origin": ["prepared"],
                    "fingerprint": ({"size": prov.get("source_size"),
                                     "mtime": prov.get("source_mtime")}
                                    if prov.get("source_mtime") is not None else None),
                    "added_at": _now(),
                })
                changed = True

        # 4. dead-store null: a pointer that resolves nowhere is cleared.  BOTH
        #    pointers: a deleted `*_processed.zarr` used to leave a PHANTOM
        #    `processed_store` forever — the raw fallback is correct, but the
        #    pointer kept advertising a store that is gone.
        for s in state.get("sources", []):
            ptr = s.get("prepared_store")
            if ptr:
                resolved = os.path.join(ws, *str(ptr).split("/"))
                if not os.path.isdir(resolved):
                    s["prepared_store"] = None
                    changed = True
            proc = get_processed_store(s)
            proc_rel = proc.get("path") if proc else None
            if proc_rel:
                resolved = os.path.join(ws, *str(proc_rel).split("/"))
                if not os.path.isdir(resolved):
                    s["processed_store"] = None
                    changed = True
                    self._log(
                        "STEP1-RESTORE",
                        f"processed store missing for "
                        f"{s.get('display_name') or s.get('path')} "
                        f"({proc_rel}) — pointer cleared, showing raw")

        return changed

    def _populate_source_list_from_registry(self, state):
        """Add a ② list item for every registry source that has a ``path`` and is
        not already listed.  Browse still MERGES scanned files; a
        registered file shows its alias + badge via the combo alias-map + the
        per-file badge (both keyed on the path), not a bare stem."""
        from qtpy.QtWidgets import QListWidgetItem
        lst = getattr(self.widget, "source_file_list", None)
        if lst is None:
            return
        present = set()
        for i in range(lst.count()):
            p = lst.item(i).data(Qt.UserRole)
            if p:
                present.add(_norm_path(p))
        try:
            import qtawesome as qta
            icon = qta.icon("mdi6.file-image-outline", color="#5ba4cf")
        except Exception:
            icon = None
        # PROGRAMMATIC populate — suspend the selection funnel (defense in depth).
        self._suspend_source_restore = True
        try:
            for s in state.get("sources", []):
                p = s.get("path")
                if not p or _norm_path(p) in present:
                    continue
                item = QListWidgetItem()
                item.setData(Qt.UserRole, p)
                item.setData(Qt.UserRole + 1,
                             os.path.splitext(os.path.basename(p))[0])
                if icon is not None:
                    item.setIcon(icon)
                lst.addItem(item)
                present.add(_norm_path(p))
        finally:
            self._suspend_source_restore = False
        self._refresh_source_file_list()

    def load_source(self):
        """Load the selected TIFF into the viewer, splitting channels.

        The disk READ runs on a background QThread (`_SourceLoadWorker`) so the
        GUI stays live and the Cancel button can discard the result + unlock the
        surface instantly.  Everything up to the launch — validation, the >RAM
        route/confirm dialog, the busy lock — is SYNCHRONOUS on the main thread
        (the dialog needs user input and can decline the load; the lock must be
        set before the read starts).  The viewer/layer continuation runs back on
        the main thread in `_on_source_loaded` (`viewer.add_image` is
        main-thread-only), which is also where the busy lock clears.
        """
        # Context change: land any pending source-edit save before load
        # (flush before a context change) so it can't race the load onto disk.
        self._flush_source_state_save()
        item = self.widget.source_file_list.currentItem()
        if item is None:
            self.widget.source_status_label.setText(
                "\u26a0  No file selected"
            )
            self.widget.source_status_label.setStyleSheet(
                "font-size: 10px; color: #d4a843;"
            )
            return

        path = item.data(Qt.UserRole)

        # Brick 4a (a2, journey b): probe the prepared store BEFORE the
        # file-not-found gate.  A workspace copied WITHOUT its raw data can
        # still be worked on -- the prepared copy holds the pixels -- but only
        # when the registry POINTER names that store (a same-named store for
        # different data is refused inside `_fresh_prepared_store_for`).
        # The probe runs before the unload below, so a failed load still leaves
        # the current source on screen exactly as it did before.
        raw_present = bool(path) and os.path.isfile(path)
        if raw_present:
            # a1: this file may BE a row we already have, under a new name or on
            # a new drive.  Re-point that row (keeping its `source_id`) BEFORE
            # anything resolves an id off the path, so ④ stays attached.
            self._relink_row_for_path(path)
        store_reasons = []
        store = None
        if path:
            store = self._fresh_prepared_store_for(
                path, allow_missing_source=not raw_present,
                reasons=store_reasons)
        if not raw_present and store is None:
            self.widget.source_status_label.setText(
                "\u26a0  File not found"
            )
            self.widget.source_status_label.setStyleSheet(
                "font-size: 10px; color: #e07070;"
            )
            return

        # Clean up previous load + clear old shapes.
        if self._loaded_source_layers:
            self.unload_source()
        self._clear_shapes_layer()

        display_name = self.widget.display_name_edit.text().strip()
        if not display_name:
            display_name = os.path.splitext(os.path.basename(path))[0]
            self.widget.display_name_edit.setText(display_name)

        # SMART LOAD: if this source has ALREADY been prepared into a FRESH
        # multiscale store, load that store (lazy, multiscale) instead of
        # re-reading the whole TIFF into memory — Prepare becomes a one-time
        # cost.  `_fresh_prepared_store_for` is strict: a stale / edited source
        # (mtime moved), a missing store, or a mismatched manifest returns None,
        # so we fall through to the normal TIFF read below UNCHANGED.  Opening
        # the store is lazy (no heavy read), so this stays synchronous — no
        # thread/busy-bar needed.
        if store is not None:
            n = self._apply_multiscale_store(store, display_name, src_path=path)
            if n is not None:
                if raw_present:
                    self._set_raw_missing(False)
                    note = self._freshness_note(store_reasons)
                    if note:
                        self._set_source_status_amber(note)
                    else:
                        self.widget.source_status_label.setText(
                            f"✔  Loaded {display_name} from prepared "
                            f"large-data store "
                            f"({n} level{'s' if n != 1 else ''})")
                        self.widget.source_status_label.setStyleSheet(
                            "font-size: 10px; color: #6abf69;")
                else:
                    # a4: the raw file is gone.  Cutting patches + trimming
                    # still work (they crop the store's level 0); Prepare and
                    # Process read the RAW block-wise, so they are switched off
                    # until the user points Step 1 back at the original file.
                    self._set_raw_missing(True)
                    self._set_source_status_amber(
                        "⚠  Can't find the original image file — "
                        f"showing the prepared copy of {display_name}.  You can "
                        "still cut out patches, but you can't re-run "
                        "preprocessing until you point Step 1 back at the "
                        "original file.")
                self._log("STEP1-LOAD",
                          f"smart-load from prepared store {store} "
                          f"({n} levels) — TIFF read skipped"
                          + (" [raw missing]" if not raw_present else ""))
                return
            # The store was FRESH per its manifest but could not be opened
            # (corrupt on disk); `_apply_multiscale_store` set a red status.
            # Fall through to the normal TIFF read as a safety net.
            self._log("STEP1-LOAD",
                      "fresh store failed to open — falling back to TIFF read")
            if not raw_present:
                # No raw to fall back TO -- `_apply_multiscale_store` already
                # showed the red reason; never walk on into the TIFF reader.
                return
        self._set_raw_missing(False)
        # a3: the store was refused because the file's size changed.  The load
        # is correct (we read the original), but say so -- an amber LINE the
        # raw-load finish re-applies over its own success text.  No modal.
        self._source_freshness_note = self._freshness_note(store_reasons)

        # SYNCHRONOUS pre-flight (main thread): the >RAM route + confirm dialog.
        # It needs user input and can DECLINE the load — bail here, before any
        # lock or launch, with the bar untouched and the surface still live.
        proceed, use_memmap = self._preflight_source_route(path)
        if not proceed:
            return

        # The shared bar MUST name what is running, and the
        # rest of the ② surface locks while it does (Cancel is deliberately NOT
        # in the lock list — it is the escape hatch).  Both go through the
        # widget's `_busy_show`/`_busy_hide` contract rather than poking the bar
        # directly, so the label can never be forgotten.
        self._set_step1_busy(True)
        self.widget._busy_show(f"Reading {os.path.basename(path)}…")
        # No `processEvents()` here: the read is async now, so the event loop
        # is free to paint + animate the marquee on its own.  Draining the
        # queue here was the one re-entrancy surface (audit LOW) — dropped.

        # A fresh token stamps THIS load.  A later Cancel / re-launch bumps it;
        # the finished/error handlers early-return on a stale token so a
        # superseded read can never touch the viewer or the busy state.
        self._source_load_token += 1
        token = self._source_load_token
        # The Load button itself becomes Cancel while the read runs: flip its
        # label + keep it clickable (the rest of the ② surface stays locked).
        # `_source_loading` gates the dispatcher so a click now cancels.
        self._source_loading = True
        self._set_load_button_cancel(True)

        # The terminal funnel that ALWAYS clears the lock
        # is now the finished/error/cancel handlers (async) + this launch guard.
        # If worker/thread construction or start() raises, NEITHER finished nor
        # error fires, so clear the lock + restore the button here so the UI
        # can never stay stuck.
        try:
            # A previous read may still be draining — Cancel unlocks the UI
            # instantly while the atomic read finishes on its own, and a fast
            # finish→relaunch can overlap too.  Keep the old (thread, worker)
            # referenced so the QThread is never GC'd mid-flight; its result is
            # already discarded by the stale-token guard.
            if self._source_load_thread is not None:
                self._draining_source_threads.append(
                    (self._source_load_thread, self._source_load_worker))
            worker = _SourceLoadWorker(path, use_memmap)
            thread = QThread()
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.finished.connect(
                lambda lazy, data, _t=token, _p=path, _dn=display_name,
                _mm=use_memmap:
                self._on_source_loaded(lazy, data, _t, _p, _dn, _mm)
            )
            worker.error.connect(
                lambda msg, _t=token: self._on_source_load_error(msg, _t)
            )
            worker.finished.connect(thread.quit)
            worker.error.connect(thread.quit)
            worker.cancelled.connect(thread.quit)
            thread.finished.connect(
                lambda _t=thread: self._cleanup_source_load_thread(_t))
            self._source_load_thread = thread
            self._source_load_worker = worker
            self._log("STEP1-LOAD",
                      f"launch token={token} {os.path.basename(path)}")
            thread.start()
        except Exception as exc:
            self._source_load_thread = None
            self._source_load_worker = None
            self.widget._busy_hide()
            self._set_step1_busy(False)
            self._source_loading = False
            self._set_load_button_cancel(False)
            self._log("STEP1-LOAD", f"launch FAILED token={token}: {exc}")

    def _preflight_source_route(self, path):
        """Synchronous >RAM route + confirm dialog for `load_source`.

        Returns ``(proceed, use_memmap)``.  A non-eager (>RAM) file pops the
        blocking `confirm_non_eager_load` dialog; declining it returns
        ``(False, False)`` so the caller bails WITHOUT launching the read or
        touching the busy bar.  The memmap branch (only offered for an
        UNCOMPRESSED >RAM file) returns ``(True, True)``; the normal eager path
        returns ``(True, False)``.

        Moved out of the old `_load_source_body` so it runs BEFORE the worker
        launch: the dialog needs the main thread + user input, and it can abort
        the whole load.
        """
        # Never call `imread` unconditionally.  A >RAM
        # file froze the app with the event loop dead — no bar animation, no
        # cancel, just a hung window while Windows thrashed the pagefile.
        route = None
        use_memmap = False
        try:
            route = self.widget._route_for_source(path)
        except Exception as exc:
            self._log("STEP1-GUARD", f"route check failed, loading anyway: {exc}")
        if route is not None:
            # The routing triple goes to ⑤ Diagnostics, where the person who
            # needs it is reading a log.
            self._log("STEP1-GUARD",
                      f"view={route['view']} process={route['process']} "
                      f"nbytes={route['nbytes']}")
            if not route["eager"]:
                self._log("STEP1-GUARD", "imread SUPPRESSED — stack exceeds "
                                         "the memory budget")
                self.widget.source_status_label.setText(
                    "⚠  Too large to load whole — see the dialog")
                self.widget.source_status_label.setStyleSheet(
                    "font-size: 10px; color: #d4a843;")
                if not self.widget.confirm_non_eager_load(path, route):
                    return False, False
                # The user chose the memory-mapped route, which the dialog only
                # offers for an UNCOMPRESSED file — the MEMMAP branch, which
                # needs no conversion and no new backend.
                use_memmap = True

        return True, use_memmap

    def _on_source_loaded(self, lazy, data, token, path, display_name,
                          use_memmap):
        """Main-thread continuation after the background read finishes.

        STALE GUARD: a newer load or a Cancel bumped `_source_load_token`,
        so this result is superseded — do NOT touch the viewer or the busy
        state (whoever bumped the token owns the UI now).

        Otherwise adopts the lazy handle + materialized display array
        (byte-identical to the old synchronous read) and runs the
        axis-detection → transpose → split_channels → add-layers →
        box-restore → status continuation that used to follow
        `materialize(...)` synchronously.  `viewer.add_image` runs HERE, on
        the main thread.  A raise anywhere in the continuation still clears
        the busy lock (the relocated terminal funnel), so a display-time
        failure can never leave the surface stuck.
        """
        if token != self._source_load_token:
            self._log("STEP1-LOAD",
                      f"stale finish token={token} "
                      f"(current={self._source_load_token}) — discarded")
            return

        # A new source is replacing the display: route through the SINGLE preview
        # teardown — drop any ⟨preview⟩ render from the PREVIOUS source, destroy
        # the geometry-bound REGION box + its overlay, and collapse the ② controls
        # so a stale crop / box never sits over (or clamps against) new data.
        self._teardown_preview()

        self._source_lazy = lazy
        if use_memmap:
            self._log("STEP1-GUARD", "loaded memory-mapped (zero-copy)")

        try:
            # --- L0 real axis detection --------------------------------
            # `data` is still in ON-DISK order here (not yet transposed), so its
            # shape is what detection reads.  Detection drives the Order combo as
            # the DEFAULT, replacing the blind `shape[0] > shape[1]` guess; the
            # combo stays the user-confirmable editor and its value wins for a
            # source the user has already picked an order for (`_axis_order_overrides`).
            norm = _norm_path(path)
            override = self._axis_order_overrides.get(norm)
            if override is not None:
                self.widget.apply_axis_order(override, f"your choice: {override}")
            else:
                detected = None
                try:
                    detected = resolve_axes(
                        data.shape, self._meta_axes_for_path(path))
                except Exception as exc:
                    self._log("STEP1-AXES", f"axis detection failed: {exc}")
                if detected is not None:
                    if detected.axes == "ZYX":
                        hint = "single channel (3D)"
                    else:
                        hint = (f"{'detected' if detected.confident else 'guessed'}: "
                                f"{detected.axes} \u00b7 {detected.n_channels} ch \u2014 confirm")
                    self.widget.apply_axis_order(detected.axes, hint)
                    self._log("STEP1-AXES",
                              f"detected {detected.axes} ({detected.n_channels} ch, "
                              f"{'confident' if detected.confident else 'inferred'})")
            # The combo now describes the file being loaded.
            self._axis_combo_source = path

            # Transpose is driven ONLY by the resolved Order combo now (the blind
            # size heuristic is gone).  Only 4D data is ever transposed \u2014 a stale
            # ZCYX combo must not swap Z/Y of a 3D stack.
            did_transpose = False
            if data.ndim == 4 and self.widget.transpose_button.isChecked():
                data = transpose_to_czyx(data)
                did_transpose = True

            # Record the on-disk order actually IN EFFECT (= the combo) so extraction
            # reorders the lazy handle to CZYX exactly as the display was.
            if data.ndim == 4:
                self._source_axes_resolved = (
                    self.widget.axis_order_combo.currentText().strip() or "CZYX")
            else:
                self._source_axes_resolved = "ZYX"
            # A raw load's `_source_axes_resolved` already IS the on-disk order,
            # so no separate provenance value is needed.
            self._source_disk_axes = None

            channels = split_channels(data, display_name)
            # Map each display layer name -> its channel index in the CZYX/ZYX
            # `data` (== the reordered lazy handle), so extraction can pick the
            # ticked channels off the lazy source.
            self._channel_index_by_name = {
                ch_name: i for i, (ch_name, _) in enumerate(channels)
            }

            layer_names = self._add_source_display_layers(channels)

            self._loaded_source_file = path
            self._loaded_source_id = self._resolve_loaded_source_id(path)
            self._persist_source_shape(path)
            self._loaded_source_layers = layer_names
            # Keep the channel refs so step-exit can remove + step-enter can re-show
            # the source layers (see `_remove_source_layers_from_viewer`).
            self._source_display_channels = channels
            self._on_layers_changed()
            # Initialise ③'s Box List for this session (starts EMPTY — ③ is a
            # transient surface; boxes appear only as the user draws them this
            # session).  Otherwise the list is only refreshed on draw-mode start.
            refresh = getattr(self.widget, "refresh_draw_box_list", None)
            if refresh is not None:
                refresh()

            n_ch = len(channels)
            transpose_note = " (transposed ZCYX\u2192CZYX)" if did_transpose else ""

            # ③ Draw & Extract is TRANSIENT: a reopened
            # source starts with NO boxes.  Already-extracted patches live ONLY in
            # ④ Review & Library (on disk), never re-seeded into ③, so we do NOT
            # restore boxes here.  Patch NAMING stays collision-safe via
            # find_next_patch_index (reads disk), independent of any restored sigs.
            # `_restore_bounding_boxes` is retained for the deferred "click a ④ row
            # -> show its box" feature.
            #
            # ④'s Library shows the persisted registry, with THIS source's patches
            # interactive (F2).  Cross-session: patches saved in a prior session
            # appear even before anything is extracted this session.
            self._refresh_library_from_registry()

            self.widget.source_status_label.setText(
                f"\u2714  Loaded {display_name} ({n_ch} channel{'s' if n_ch != 1 else ''}){transpose_note}"
            )
            self.widget.source_status_label.setStyleSheet(
                "font-size: 10px; color: #6abf69;"
            )
            note = getattr(self, "_source_freshness_note", None)
            if note:
                self._set_source_status_amber(note)
            self._source_freshness_note = None
        finally:
            self.widget._busy_hide()
            self._set_step1_busy(False)
            self._source_loading = False
            self._set_load_button_cancel(False)

    def _on_source_load_error(self, msg, token):
        """Main-thread handler for a background read that raised.

        STALE GUARD first (a Cancel / re-launch superseded this read).
        Otherwise reproduces the old synchronous read-failure branch: drop
        the lazy handle + axis/channel state and write the red status, then
        clear the busy lock + restore the Load button.
        """
        if token != self._source_load_token:
            self._log("STEP1-LOAD",
                      f"stale error token={token} "
                      f"(current={self._source_load_token}) — discarded")
            return
        self._source_lazy = None
        self._source_axes_resolved = None
        self._source_disk_axes = None
        self._channel_index_by_name = {}
        self.widget.source_status_label.setText(
            f"⚠  Failed to read: {msg}"
        )
        self.widget.source_status_label.setStyleSheet(
            "font-size: 10px; color: #e07070;"
        )
        self.widget._busy_hide()
        self._set_step1_busy(False)
        self._source_loading = False
        self._set_load_button_cancel(False)

    def cancel_source_load(self):
        """Discard the in-flight source read and unlock the surface NOW.

        A single atomic read (open_lazy + materialize) cannot be interrupted
        mid-flight, so Cancel DISCARDS the result and unlocks the UI
        immediately; the background read still completes on its own — its
        finished handler early-returns on the bumped token, so it never
        touches the viewer.
        """
        self._source_load_token += 1
        worker = getattr(self, "_source_load_worker", None)
        if worker is not None:
            worker.request_cancel()
        self.widget._busy_hide()
        self._set_step1_busy(False)
        self._source_loading = False
        self._set_load_button_cancel(False)
        self._log("STEP1-LOAD", "cancelled")

    def on_load_or_cancel_clicked(self):
        """The Load button doubles as Cancel while a source read is in flight.

        Dispatches on `_source_loading` (True only between launch and
        finish/error/cancel) so a click DURING a load cancels it, and a click
        after a Cancel — while an old thread may still be draining — starts a
        fresh load rather than re-cancelling.
        """
        if getattr(self, "_source_loading", False):
            self.cancel_source_load()
        else:
            self.load_source()

    def _set_load_button_cancel(self, active: bool) -> None:
        """Flip the Load button to 'Cancel' (kept clickable) while a read runs,
        or back to its idle label when it finishes.  Best-effort; touches only
        the button text/enabled state."""
        btn = getattr(self.widget, "load_source_button", None)
        if btn is None:
            return
        try:
            if active:
                if self._load_button_text is None:
                    self._load_button_text = btn.text()
                    self._load_button_icon = btn.icon()   # remember to restore
                btn.setEnabled(True)          # stays clickable as Cancel
                btn.setText("Cancel")
                try:
                    from napari_epicycle.ui_shell.theme import apply_button_icon
                    apply_button_icon(btn, "mdi6.close-circle-outline")
                except Exception:
                    pass
            elif self._load_button_text is not None:
                btn.setText(self._load_button_text)
                if self._load_button_icon is not None:
                    btn.setIcon(self._load_button_icon)
                self._load_button_text = None
                self._load_button_icon = None
        except Exception:
            pass

    def _cleanup_source_load_thread(self, thread=None):
        """Drop the finished source-load (thread, worker) refs.

        Identity-guarded: only nulls the CURRENT pair if it still points at
        `thread` (a newer load may already own the refs), and removes any
        matching draining pair kept alive after a Cancel unlocked the UI.
        `deleteLater()` the finished thread (the house idiom) rather than
        relying on GC timing while Qt is still emitting `thread.finished`.
        """
        draining = getattr(self, "_draining_source_threads", None)
        if draining is not None and thread is not None:
            self._draining_source_threads = [
                (t, w) for (t, w) in draining if t is not thread]
        if thread is None or thread is self._source_load_thread:
            self._source_load_thread = None
            self._source_load_worker = None
        if thread is not None:
            try:
                thread.deleteLater()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Large-data viewing: convert to multiscale OME-Zarr + display lazily
    # ------------------------------------------------------------------

    def prepare_large_data(self):
        """Convert the selected source to a multiscale OME-Zarr store on a
        background QThread, then load that store as a MULTISCALE image so the
        user can pan/zoom (2D) + see a coarse 3D overview without loading the
        whole volume, and so the box tool + extract work on the converted store.

        MIRRORS :meth:`load_source`: validation + the busy lock are SYNCHRONOUS
        on the main thread, the (minutes-long) conversion runs on a
        `_LargeDataPrepWorker` QThread, and the viewer/layer continuation runs
        back on the main thread in :meth:`_on_large_data_ready`.  The Prepare
        button doubles as Cancel while the convert is in flight.
        """
        # Context change: land any pending source-edit save before Prepare runs
        # (its finish-time `_persist_prepared_source` also writes the registry —
        # flushing first keeps the two writes serialized).
        self._flush_source_state_save()
        item = self.widget.source_file_list.currentItem()
        if item is None:
            self.widget.source_status_label.setText("⚠  No file selected")
            self.widget.source_status_label.setStyleSheet(
                "font-size: 10px; color: #d4a843;")
            return

        path = item.data(Qt.UserRole)
        if not path or not os.path.isfile(path):
            self.widget.source_status_label.setText("⚠  File not found")
            self.widget.source_status_label.setStyleSheet(
                "font-size: 10px; color: #e07070;")
            return

        workspace = self.widget.output_file_path.text().strip()
        if not workspace:
            self.widget.source_status_label.setText(
                "⚠  Set a workspace first — large-data stores live under it")
            self.widget.source_status_label.setStyleSheet(
                "font-size: 10px; color: #d4a843;")
            return

        display_name = self.widget.display_name_edit.text().strip()
        if not display_name:
            display_name = os.path.splitext(os.path.basename(path))[0]
            self.widget.display_name_edit.setText(display_name)

        # ALREADY PREPARED?  If a FRESH store already exists for this source, the
        # (minutes-long) convert is redundant — just load it.  An explicit
        # re-prepare / force path arrives with the config UI; for now fresh =
        # skip, and `overwrite=True` still applies only on the convert paths
        # below.  `_fresh_prepared_store_for` is strict (edited source / missing
        # store / mismatched manifest → None), so a stale store re-converts.
        fresh = self._fresh_prepared_store_for(path)
        if fresh is not None:
            n = self._apply_multiscale_store(fresh, display_name, src_path=path)
            if n is not None:
                self.widget.source_status_label.setText(
                    f"✔  Already prepared — loaded {display_name} "
                    f"({n} level{'s' if n != 1 else ''})")
                self.widget.source_status_label.setStyleSheet(
                    "font-size: 10px; color: #6abf69;")
                self._log("STEP1-PREP",
                          f"already prepared — skipped convert, loaded {fresh} "
                          f"({n} levels)")
                return
            # Fresh per its manifest but unreadable — re-convert as a safety net.
            self._log("STEP1-PREP",
                      "fresh store failed to open — re-converting")

        # The on-disk axis order for the converter.  `_resolve_prepare_image_axes`
        # trusts the Order combo (which detect-on-selection points at the
        # SELECTED source) for the file being prepared, else the header
        # heuristic — so a 4D source orients WITHOUT a mandatory Load.  The
        # user-entered voxel spacing is None when left at the isotropic default.
        image_axes = self._resolve_prepare_image_axes(path)
        voxel_size = self._prepare_voxel_size()

        # Non-silent notice when the order for a 4D source was a HEURISTIC guess
        # (tifffile could not read clean axes metadata): show what we assumed so
        # a mis-orientation is visible (the store is a non-destructive copy — if
        # it looks wrong, flip the Order dropdown and Prepare again).  Any
        # residual converter guard still surfaces via `_on_large_data_prep_error`.
        shape = self._scanned_shape_for_path(path)
        has_override = _norm_path(path) in self._axis_order_overrides
        if len(shape) == 4 and image_axes and not has_override:
            res = resolve_axes(shape, self._meta_axes_for_path(path))
            if not res.confident:
                self._log(
                    "STEP1-PREP",
                    f"assuming {image_axes} for {os.path.basename(path)} "
                    "(channel/Z order guessed from shape — flip the Order "
                    "dropdown + Prepare again if the result looks wrong)")

        out_store = self._resolve_prepare_store_path(workspace, display_name, path)
        try:
            os.makedirs(os.path.dirname(out_store), exist_ok=True)
        except Exception as exc:
            self.widget.source_status_label.setText(
                f"⚠  Could not create store folder: {exc}")
            self.widget.source_status_label.setStyleSheet(
                "font-size: 10px; color: #e07070;")
            return

        # Effective large-data convert config: the per-workspace saved config
        # (Prepare-settings dialog) if present, else the machine-adaptive
        # "balanced" default.  Feeds num_workers/chunk/downsample/min_size to the
        # converter (replaces the old `_LARGEDATA_PREP_WORKERS`-only hardcode).
        ld_cfg = self._current_largedata_config(workspace)
        self._log("STEP1-PREP",
                  f"convert config {ld_cfg} (workspace-saved or adaptive)")

        # Lock the ② surface (one Step-1 op at a time), name the op on the
        # shared bar, then flip Prepare to Cancel (kept clickable) — the exact
        # sequence `load_source` uses, incl. re-enabling the button AFTER the
        # lock so it stays the escape hatch.
        self._set_step1_busy(True)
        self.widget._busy_show(
            f"Preparing {display_name} for large-data viewing…")
        self._largedata_token += 1
        token = self._largedata_token
        # The label the determinate ticks re-use, so the bar never renames the
        # op mid-run.
        self._largedata_busy_label = (
            f"Preparing {display_name} for large-data viewing…")
        self._largedata_preparing = True
        self._largedata_src_path = path
        # Prepare params snapshot for the finish-time registry upsert:
        # the effective convert knobs + the resolved on-disk order + the
        # user's voxel (None = isotropic default).  Read only in
        # `_persist_prepared_source`.
        self._largedata_prepare_params = {
            "image_axes": image_axes,
            "voxel_size": (list(voxel_size) if voxel_size is not None else None),
            "chunk": ld_cfg.get("chunk"),
            "chunk_zyx": ld_cfg.get("chunk_zyx"),
            "compressor": ld_cfg.get("compressor"),
            "downsample": ld_cfg.get("downsample"),
            "min_size": ld_cfg.get("min_size"),
            "num_workers": ld_cfg.get("num_workers"),
        }
        self._set_prepare_button_cancel(True)

        # Terminal-funnel launch guard: if worker/thread construction raises,
        # neither finished nor error fires — clear the lock + restore the button
        # here so the UI can never stay stuck.
        try:
            if self._largedata_thread is not None:
                self._draining_largedata_threads.append(
                    (self._largedata_thread, self._largedata_worker))
            worker = _LargeDataPrepWorker(
                path, out_store, voxel_size, image_axes,
                num_workers=(ld_cfg.get("num_workers")
                             if _LARGEDATA_PREP_WORKERS is None
                             else _LARGEDATA_PREP_WORKERS),
                chunk=(ld_cfg.get("chunk_zyx") or ld_cfg.get("chunk")),
                compressor=ld_cfg.get("compressor"),
                downsample=ld_cfg.get("downsample"),
                min_size=ld_cfg.get("min_size"))
            thread = QThread()
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.finished.connect(
                lambda store, _t=token, _dn=display_name:
                self._on_large_data_ready(store, _dn, _t)
            )
            worker.error.connect(
                lambda msg, _t=token: self._on_large_data_prep_error(msg, _t)
            )
            # Token BOUND IN THE LAMBDA: a draining worker from a superseded
            # Prepare must not drive the new run's bar.
            worker.progress.connect(
                lambda done, total, _t=token:
                self._on_large_data_progress(done, total, _t)
            )
            worker.finished.connect(thread.quit)
            worker.error.connect(thread.quit)
            worker.cancelled.connect(thread.quit)
            thread.finished.connect(
                lambda _t=thread: self._cleanup_largedata_thread(_t))
            self._largedata_thread = thread
            self._largedata_worker = worker
            self._log("STEP1-PREP",
                      f"launch token={token} {display_name} -> {out_store}")
            thread.start()
        except Exception as exc:
            self._largedata_thread = None
            self._largedata_worker = None
            self.widget._busy_hide()
            self._set_step1_busy(False)
            self._largedata_preparing = False
            self._set_prepare_button_cancel(False)
            self._log("STEP1-PREP", f"launch FAILED token={token}: {exc}")

    def _current_largedata_config(self, workspace=None):
        """The effective large-data convert config for *workspace*.

        Returns a full ``{preset, num_workers, chunk, chunk_zyx, compressor,
        downsample, min_size}`` dict so callers can read every knob
        unconditionally.  *workspace* defaults to the current output path; a
        missing / empty workspace skips the saved lookup and returns the adaptive
        default (nothing to persist against).

        A saved config is NOT replayed verbatim.  What is persisted is a
        SNAPSHOT of what a preset meant on the machine that saved it, so:

        * a saved **named** preset (``safe`` / ``balanced`` / ``fast``) is
          RE-DERIVED through :func:`recommend_convert_settings` for THIS machine
          and this app version.  Replaying it verbatim pinned every workspace
          prepared before Brick 4b to the old cubic chunk shape -- the user kept
          the choppy scrolling while the dialog showed them the new default --
          and also trusted a stale ``num_workers`` from whatever box wrote it.
          Even an explicit ``chunk_zyx`` is re-derived here: on a named preset
          the recommender wrote it, not the user.
        * a saved **custom** config IS honoured verbatim -- the user typed those
          exact numbers, so they are a literal, not a snapshot.

        The source shape / read regime are passed through when known so the
        worker-count clamp models this source's real peak (see
        :func:`config.estimate_peak_ram_bytes`).
        """
        from packages.epicycle_largedata.config import (
            DEFAULT_PRESET, PRESET_NAMES, detect_machine_resources,
            recommend_convert_settings,
        )
        from ._state import load_largedata_config

        if workspace is None:
            workspace = self.widget.output_file_path.text().strip()

        preset = DEFAULT_PRESET
        if workspace:
            try:
                saved = load_largedata_config(workspace)
            except Exception:
                saved = None
            if saved is not None:
                saved_preset = str(saved.get("preset", "")).strip().lower()
                if saved_preset not in PRESET_NAMES:
                    return saved          # custom (or unknown) -> verbatim
                preset = saved_preset     # named -> fall through and re-derive

        res = detect_machine_resources()
        return recommend_convert_settings(
            preset,
            total_ram_bytes=res["total_ram_bytes"],
            avail_ram_bytes=res["avail_ram_bytes"],
            cpu_count=res["cpu_count"],
            shape=self._largedata_source_shape_for_estimate(),
            regime=self._largedata_source_regime_for_estimate(),
        )

    def _largedata_source_regime_for_estimate(self):
        """Which read regime the convert will take for the selected source
        (``"memmap"`` / ``"compressed"`` / ``None``), for the RAM model.

        An UNCOMPRESSED TIFF is memmapped and read one chunk at a time, so its
        real peak is ~one chunk per worker; a COMPRESSED one must buffer a
        Z-slab per worker.  Getting this wrong in the pessimistic direction
        clamps a big machine to a single core for no reason, so it is worth
        resolving rather than assuming.

        Best-effort + never raises (it only sharpens an estimate): no source, an
        unreadable one, or a non-TIFF all return ``None``, which the model
        treats as the conservative case.
        """
        from packages.epicycle_largedata.config import resolve_source_regime

        try:
            path = getattr(self, "_largedata_src_path", None)
            if path is None:
                getter = getattr(self, "_loaded_source_path", None)
                if callable(getter):
                    path = getter()
            return resolve_source_regime(path)
        except Exception:
            return None

    def _largedata_source_shape_for_estimate(self):
        """The currently-selected source's ``(Z, Y, X)`` / ``(C, Z, Y, X)``, or
        ``None``.

        Best-effort + never raises: the shape only sharpens the dialog's RAM
        estimate, so a source that is not loaded, not readable, or has no shape
        simply falls back to the chunk-only model.
        """
        for attr in ("_source_lazy",):
            arr = getattr(self, attr, None)
            shp = getattr(arr, "shape", None)
            if shp is not None:
                try:
                    vals = tuple(int(v) for v in shp)
                except (TypeError, ValueError):
                    continue
                if len(vals) in (3, 4):
                    return vals
        return None

    def open_largedata_config_dialog(self):
        """Open the Prepare-settings dialog seeded with the current effective
        config, and on OK persist the chosen config for this workspace.

        Returns the saved config dict, or ``None`` on Cancel.  A missing
        workspace still opens the dialog (seeded with the adaptive default) but
        cannot persist — the status line says so, and the next Prepare recomputes
        the adaptive default anyway.
        """
        from packages.epicycle_largedata.config import detect_machine_resources
        from ._largedata_config_dialog import LargeDataConfigDialog
        from ._state import save_largedata_config

        workspace = self.widget.output_file_path.text().strip()
        res = detect_machine_resources()
        seed = self._current_largedata_config(workspace)
        dlg = LargeDataConfigDialog(
            self.widget,
            cpu_count=res["cpu_count"],
            total_ram_bytes=res["total_ram_bytes"],
            avail_ram_bytes=res["avail_ram_bytes"],
            initial_config=seed,
            # Brick 4b: hand the dialog the SOURCE shape when we know it, so the
            # peak-RAM readout uses the honest compressed-path slab model
            # (chunk_z x Y x X per worker) instead of the chunk-only bound.
            # None (no source picked yet) keeps the old readout.
            source_shape=self._largedata_source_shape_for_estimate(),
            # ...and WHICH read regime it will take, so an uncompressed source
            # is not charged the compressed path's Z-slab (which would clamp a
            # big machine to one worker for no reason).
            source_regime=self._largedata_source_regime_for_estimate(),
        )
        if not dlg.exec_():
            return None
        cfg = dlg.get_config()
        if not workspace:
            self._log("STEP1-PREP",
                      "large-data config chosen but no workspace set — "
                      "not persisted")
            try:
                self.widget.source_status_label.setText(
                    "⚠  Set a workspace to save large-data settings")
                self.widget.source_status_label.setStyleSheet(
                    "font-size: 10px; color: #d4a843;")
            except Exception:
                pass
            return cfg
        try:
            save_largedata_config(workspace, cfg)
            self._log("STEP1-PREP", f"saved large-data config {cfg}")
        except Exception as exc:
            self._log("STEP1-PREP", f"failed to save large-data config: {exc}")
        return cfg

    def _scanned_shape_for_path(self, path):
        """The header-scanned shape tuple for a source *path*, or ``()`` — read
        from the header scan so it needs no extra I/O."""
        for info in (getattr(self, "_scanned_files", None) or []):
            if info.get("path") == path:
                return tuple(info.get("shape") or ())
        return ()

    def _resolve_prepare_image_axes(self, path):
        """The on-disk axis order to hand `convert_to_ome_zarr` for *path*, or
        ``None`` (3D / unknown shape — the converter resolves ZYX itself).

        The Order combo is pointed at whatever source is SELECTED (see
        :meth:`_apply_detected_axes_for_source`, bound to the list selection; the
        load path sets it too), tracked in ``_axis_combo_source``.  Because
        "Prepare" always runs on the selected source, the combo describes the file
        being prepared, so it can orient it WITHOUT a full memory Load — the user
        has SEEN and can flip the ``guessed … — confirm`` value before Prepare.
        Resolution, most-trusted first:

        * A 3D shape -> ``None`` (ZYX is unambiguous; the converter resolves it).
        * The combo, when it DESCRIBES *path* (``path == _axis_combo_source``) —
          the user-confirmable order for exactly this file.
        * The header heuristic for a 4D shape (:func:`resolve_axes`) as a
          defensive fallback for a *path* the combo does not describe (e.g. a
          programmatic call) — CONFIDENT metadata OR the size guess (2 ch ≤ 8 →
          channel), so Prepare is NEVER blocked behind a mandatory Load.

        The invariant the earlier audit protected still holds: the combo is
        trusted ONLY for the file it describes (``_axis_combo_source``), so an
        order chosen for file A can never orient file B — selecting B repoints the
        combo (and ``_axis_combo_source``) at B's own detection first.
        """
        shape = self._scanned_shape_for_path(path)
        if len(shape) == 3:
            return None
        # The combo describes exactly the file `_axis_combo_source` names (both
        # detect-on-selection AND the load path set it), so trust it only for
        # THAT file — a Prepare is oriented by the file being prepared, never a
        # combo left pointing at a different source.  (`_loaded_source_file` is
        # deliberately NOT a second trust key: it can diverge from the combo when
        # a different file is selected than loaded, which would trust a combo that
        # describes another file — 3-audit LOW 2026-08-02.)
        if path == getattr(self, "_axis_combo_source", None):
            combo = getattr(self.widget, "axis_order_combo", None)
            order = (combo.currentText().strip()
                     if combo is not None else "") or None
            if order:
                return order
        if len(shape) == 4:
            # Fallback for an un-tracked 4D path: the deterministic header
            # heuristic (confident metadata, else the size guess).  Never None
            # for 4D -> Prepare proceeds without forcing a Load.
            return resolve_axes(shape, self._meta_axes_for_path(path)).axes
        return None

    def _prepare_voxel_size(self):
        """The user-entered voxel spacing ``(z, y, x)`` for the NGFF scales, or
        ``None`` when it is left at the ``(1, 1, 1)`` isotropic default (which
        means "spacing unknown" — leave the store's scales in voxel units)."""
        w = self.widget
        if not hasattr(w, "voxel_z_spin"):
            return None
        try:
            vz = float(w.voxel_z_spin.value())
            vy = float(w.voxel_y_spin.value())
            vx = float(w.voxel_x_spin.value())
        except Exception:
            return None
        if (vz, vy, vx) == (1.0, 1.0, 1.0):
            return None
        if vz <= 0 or vy <= 0 or vx <= 0:
            return None
        return (vz, vy, vx)

    def _apply_multiscale_store(self, store_path, display_name, src_path=None):
        """Open a prepared multiscale OME-Zarr *store* and show it as the Step-1
        source: per-channel MULTISCALE display layers + the L0 lazy state the
        box tool + the lazy extract path read.

        Shared by three callers so they behave identically:
        :meth:`_on_large_data_ready` (the background-convert continuation), the
        Smart-Load branch of :meth:`load_source`, and the "already prepared"
        skip in :meth:`prepare_large_data`.  Opening the store is LAZY
        (``open_multiscale`` → ``da.from_zarr`` per level), so this is cheap
        enough to run on the main thread — nothing heavy is read until a crop is
        materialized.

        *src_path* is recorded as ``_loaded_source_file`` (extract provenance);
        when ``None`` it falls back to ``_largedata_src_path`` so the existing
        convert path stays byte-identical.

        Returns the number of pyramid levels loaded, or ``None`` if the store
        could not be opened — in which case a red status is set and nothing else
        is touched (the same graceful degrade the convert path had).  The caller
        may override the (green) success status; the failure status is left as-is.
        """
        # Silence a benign napari-0.8 RuntimeWarning ("overflow encountered in
        # scalar multiply") from its multiscale level-control widget when it
        # sizes a large pyramid level — napari's own arithmetic, harmless, and it
        # clutters the console (user 2026-08-07).  One-time + targeted to that
        # widget so it can't mask our own overflows.
        if not getattr(self, "_napari_ms_warn_filtered", False):
            import warnings
            warnings.filterwarnings(
                "ignore",
                message="overflow encountered in scalar multiply",
                category=RuntimeWarning,
                module=r"napari\._qt\.layer_controls\.widgets\.qt_multiscale_level_control",
            )
            self._napari_ms_warn_filtered = True

        # Arm large-data session logging: the progressive-3D engine's GL errors
        # are CAUGHT by vispy's draw loop (not fatal, and their tracebacks scroll
        # off the console), so tee stdout+stderr + faulthandler to a per-session
        # file under ~/.epicycle/step1_largedata for post-hoc debugging.
        # Idempotent; observe-only.  (Strip once progressive-3D is stable.)  NOT
        # under pytest — teeing the global streams would break test capture.
        if "pytest" not in sys.modules:
            try:
                from ._largedata_diag_logger import install as _install_diag
                _diag_path = _install_diag()
                if _diag_path is not None:
                    self._log("STEP1-PREP",
                              f"large-data session log -> {_diag_path}")
            except Exception as exc:  # noqa: BLE001
                self._log("STEP1-PREP", f"diag logger arm skipped: {exc}")

        try:
            levels, meta = open_multiscale(store_path)
        except Exception as exc:
            self.widget.source_status_label.setText(
                f"⚠  Prepared but could not open store: {exc}")
            self.widget.source_status_label.setStyleSheet(
                "font-size: 10px; color: #e07070;")
            self._log("STEP1-PREP", f"open_multiscale failed: {exc}")
            return None

        # Fully drop any current source (layers + box tool + stale L0 state)
        # so we never stack two sources on the shared canvas.
        self.unload_source()

        level0 = levels[0]
        axes_meta = meta.get("axes") or []
        has_channel = any(
            (a or {}).get("type") == "channel" for a in axes_meta)

        if has_channel:
            # CZYX: one napari layer PER channel, each a multiscale list
            # (slice every pyramid level to channel c).  Names match
            # `split_channels` so exit/re-enter + extract stay consistent.
            n_ch = int(level0.shape[0])
            channels = [
                (f"{display_name}_ch{i}", [lvl[i] for lvl in levels])
                for i in range(n_ch)
            ]
            idx_map = {f"{display_name}_ch{i}": i for i in range(n_ch)}
            self._source_axes_resolved = "CZYX"
        else:
            # 3D ZYX single channel: one multiscale layer over the levels.
            channels = [(display_name, list(levels))]
            idx_map = {display_name: 0}
            self._source_axes_resolved = "ZYX"

        # PROVENANCE-ONLY, RESERVED for Brick 2 (restore/export) — WRITE-ONLY in
        # Brick 1 (read in zero places today).  Capture the store's honest
        # ON-DISK axis order from its own ``epicycle_source.source_axes`` (schema
        # v2), canonical only as a legacy fallback.  This is SEPARATE from
        # `_source_axes_resolved` above, which MUST stay the CANONICAL store
        # layout (CZYX/ZYX) — the box tool (`full_shape_from_source`) and the
        # lazy extract (`_crop_source_for`, which `moveaxis`es on ``== "ZCYX"``)
        # both read `_source_lazy` (= the canonical level 0) THROUGH
        # `_source_axes_resolved`, so repurposing it to the raw order would
        # double-transpose a canonical store.  Display / progressive-swap logic
        # is untouched.
        store_src = meta.get("source") if isinstance(meta.get("source"), dict) else {}
        self._source_disk_axes = (
            store_src.get("source_axes") or self._source_axes_resolved)

        # Physical-µm display scale: resolve the store's voxel EXPLICITLY and pass
        # it in — the store voxel reaches the spins only later
        # (`_apply_store_voxel_to_widget`, run below), so a live spin read at add
        # time would use the PREVIOUS source's voxel (grill 2 M1 voxel-timing).
        store_vox = None
        try:
            _p = src_path or getattr(self, "_largedata_src_path", None)
            _ws = (self.widget.output_file_path.text().strip()
                   if hasattr(self.widget, "output_file_path") else "")
            _row = (self._find_source_row(load_state(_ws), _p)
                    if _ws and os.path.isdir(_ws) else None)
            store_vox, _es, _b, _bs = self._resolve_voxel_for_source(
                _p, _row, store_meta=meta)
        except Exception as exc:   # noqa: BLE001 — scale is best-effort
            self._log("STEP1-PREP", f"store voxel resolve for scale: {exc}")
        layer_names = self._add_source_display_layers(
            channels, voxel_size=store_vox, contrast_store_key=store_path)

        # L0 lazy state: the finest level is the crop source the box tool +
        # extract read — only the ROI is ever materialized.
        self._source_lazy = level0
        self._channel_index_by_name = idx_map
        if src_path is None:
            src_path = getattr(self, "_largedata_src_path", None)
        self._loaded_source_file = src_path
        self._loaded_source_id = self._resolve_loaded_source_id(src_path)
        self._persist_source_shape(src_path)
        self._loaded_source_layers = layer_names
        self._source_display_channels = channels
        # Remember the store currently displayed so the Progressive-3D live
        # toggle can re-display it in the new mode (cleared on unload_source).
        self._current_multiscale_store = store_path
        self._current_multiscale_display_name = display_name
        self._current_multiscale_src = src_path
        self._on_layers_changed()
        # A new (multiscale) source is on display: tear down any preview REGION
        # bound to the previous source's geometry (single entry point — clears the
        # render, destroys the box + overlay, collapses the ② controls).
        self._teardown_preview()
        # Reopen detection (Brick B): teardown just greyed the Raw / Processed
        # toggle + cleared `_processed_available`.  If a FRESH on-disk processed
        # store exists for THIS source + the CURRENT recipe, re-light the toggle
        # (Processed enabled) — SYNCHRONOUSLY, so no untagged-raw frame flashes.
        # The default view stays RAW (one click to Processed); a stale store
        # (recipe edited / raw file changed) leaves it greyed.  Fully guarded.
        self._maybe_light_processed_toggle_on_load(display_name)

        # Seed the box tool's full (Z, Y, X) from level 0 so a box drawn now
        # is placed + clamped in full-res voxel world (level-0 NGFF scale=1),
        # aligned with the multiscale display.
        box_tool = getattr(self, "box_tool", None)
        if box_tool is not None:
            try:
                box_tool._full_shape = full_shape_from_source(
                    level0.shape, self._source_axes_resolved)
            except Exception as exc:
                self._log("STEP1-PREP", f"box full_shape seed: {exc}")

        refresh = getattr(self.widget, "refresh_draw_box_list", None)
        if refresh is not None:
            refresh()

        # ③ Draw & Extract is TRANSIENT: a reopened
        # source starts with NO boxes, so we do NOT restore this source's exported
        # boxes into ③ here.  Already-extracted patches live ONLY in ④ Review &
        # Library (refreshed from the registry below).  `_restore_bounding_boxes`
        # is retained for the deferred "click a ④ row -> show its box" feature.
        # ④ Review & Library: list this source's saved patches (F2).
        try:
            self._refresh_library_from_registry()
        except Exception as exc:
            self._log("STEP1-PREP", f"library refresh on prepared load: {exc}")

        n_levels = len(levels)

        # 3D-detail slider: capture each level's (Z, Y, X) for the readout (the
        # last 3 dims cover both CZYX and ZYX), then show + range the control
        # parked at the coarsest level (napari's default 3D render — no surprise
        # memory hit).  Wire the 2D/3D-switch hook so the pin only bites in 3D.
        self._pyramid_level_shapes = [
            tuple(int(s) for s in lvl.shape[-3:]) for lvl in levels]
        self._connect_ndisplay_hook()
        # The native layers are already on the canvas.  Pick the display mode +
        # slider coexistence from the CURRENT ndisplay:
        #   * 2D                   -> native, slider hidden.
        #   * 3D + progressive OFF -> native, manual 3D-detail slider shown.
        #   * 3D + progressive ON  -> schedule the deferred native->progressive
        #                             swap; slider hidden (progressive owns the
        #                             level).
        # Loading a store while ALREADY in 3D therefore gets progressive (via the
        # deferred swap) without the caller doing anything special.  These helpers
        # are the SAME ones the 2D<->3D switch (`_on_ndisplay_changed`) uses.
        self._maybe_swap_progressive_on_ndisplay()
        self._refresh_3d_detail_for_ndisplay()
        self._refresh_zoom_depth_for_ndisplay()
        self._apply_3d_detail_level()

        # VOXEL / NAME PROVENANCE ONLY (smart-load): read the store's
        # NGFF level-0 `scale` (voxel) via `meta` and push the resolved voxel to
        # the spins + honest label.  A registry user override still wins (passed
        # via `row`); a non-unit store scale is the "detected" baseline otherwise.
        # Best-effort — does NOT touch the display-layer build / progressive swap
        # above (HARD INVARIANT).
        try:
            self._apply_store_voxel_to_widget(meta, src_path)
            # Re-sync the display scale to the FINAL spins: if the store-voxel
            # resolve for the add-time `scale=` above failed (best-effort), the
            # layers were added at the previous/live spins; the spins now hold the
            # store voxel, so this re-applies the correct physical scale (closes
            # the resolver-failure timing gap).  A no-op when they already agree.
            self._reapply_display_physical_scale()
        except Exception as exc:   # noqa: BLE001 — voxel push is non-critical
            self._log("STEP1-PREP", f"store voxel push skipped: {exc}")

        self.widget.source_status_label.setText(
            f"✔  Loaded {display_name} for large-data viewing "
            f"— {n_levels} level{'s' if n_levels != 1 else ''}")
        self.widget.source_status_label.setStyleSheet(
            "font-size: 10px; color: #6abf69;")
        self._log("STEP1-PREP",
                  f"loaded multiscale store {store_path} "
                  f"({n_levels} levels, {len(layer_names)} channel(s))")
        return n_levels

    def _apply_store_voxel_to_widget(self, meta, src_path):
        """Push a smart-loaded store's voxel provenance into the ② spins/label.
        The store NGFF scale is the "detected" tier and outranks a
        header detect; a registry ``"user"`` override
        still wins (via the source row).  Display-only: it does NOT change the
        rendered scale-bar (only a re-prepare reconciles the store geometry)."""
        w = self.widget
        if not hasattr(w, "apply_voxel_size"):
            return
        path = src_path or getattr(self, "_largedata_src_path", None)
        ws = w.output_file_path.text().strip() if hasattr(w, "output_file_path") else ""
        row = self._find_source_row(load_state(ws), path) if (ws and os.path.isdir(ws)) else None
        effective, eff_src, baseline, _ = self._resolve_voxel_for_source(
            path, row, store_meta=meta)
        if hasattr(w, "set_detected_voxel_size"):
            w.set_detected_voxel_size(*baseline)
        w.apply_voxel_size(*effective, source=eff_src)

    def _connect_ndisplay_hook(self):
        """Connect (once) to the viewer's 2D<->3D switch so the 3D-detail pin is
        applied entering 3D and released (auto) entering 2D, and the progressive
        swap is driven.

        ORDERING IS NOT A LEVER HERE (audit finding — do not re-attempt an
        ordering-based 'fix').  napari's ``EventEmitter`` runs its own "core"
        callbacks (owners whose module starts with ``napari.``) BEFORE any
        non-core/plugin callback.  Our controller module is ``napari_epicycle.*``
        (non-core), so our handler ALWAYS runs AFTER napari's
        ``ViewerModel._update_layers`` re-slice (viewer_model.py:302) — a
        ``position="first"`` connect only orders us among OTHER non-core
        callbacks, and ``before=`` raises (napari stores callbacks as weakref
        tuples).  Proven inert against napari 0.8.0.  What actually keeps the
        3D->2D swap safe is that ``_swap_progressive_to_native`` removes the
        progressive VOLUME node SYNCHRONOUSLY within this dispatch, so the node
        is gone before napari's DEFERRED paint/GLIR flush runs (the flush is
        where the ``ibuf.activate()`` access violation would fire)."""
        if self._ndisplay_connected:
            return
        try:
            self.viewer.dims.events.ndisplay.connect(self._on_ndisplay_changed)
            self._ndisplay_connected = True
        except Exception as exc:
            self._log("STEP1-PREP", f"ndisplay hook connect failed: {exc}")

    def _configure_3d_detail(self, n_levels):
        """Show + range the 3D-detail slider for a store of *n_levels* levels.

        Parks at the user's EXISTING valid pin if one is set (re-entering 3D, or
        returning to Step 1 in 3D), else the coarsest level (initial display —
        index n-1 = napari's default 3D view, no surprise memory hit).  The
        pin-preservation matters because `widget.configure_3d_detail` ALWAYS
        applies its `default_level`; passing a constant coarsest here would reset
        a user-chosen level back to coarsest on every 3D re-entry / step reshow
        (regression surfaced by the reshow-in-3D re-engagement)."""
        if n_levels <= 1:
            # A single-level store has no coarser/finer choice — hide the control.
            hide_td = getattr(self.widget, "hide_3d_detail", None)
            if hide_td is not None:
                hide_td()
            self._locked_3d_level = None
            return
        w = getattr(self, "widget", None)
        cfg = getattr(w, "configure_3d_detail", None)
        if cfg is not None:
            existing = self._locked_3d_level
            if existing is not None and 0 <= existing < n_levels:
                default_level = existing
            else:
                default_level = n_levels - 1
            cfg(n_levels, default_level=default_level)

    def on_3d_detail_changed(self, value):
        """Slider moved (or configured): record the chosen 3D level, apply it to
        the multiscale layers (only while in 3D), and write the plain-language
        readout (voxel dims + a memory note on the finest level)."""
        self._locked_3d_level = int(value)
        self._apply_3d_detail_level()
        self._update_3d_detail_label()

    def _update_3d_detail_label(self):
        shapes = self._pyramid_level_shapes or []
        v = self._locked_3d_level
        if v is None or v < 0 or v >= len(shapes):
            return
        z, y, x = shapes[v]
        n = len(shapes)
        if v == 0:
            tag = " · sharpest ⚠ loads the full volume into GPU"
        elif v == n - 1:
            tag = " · coarsest (lightest)"
        else:
            tag = ""
        self.widget.set_3d_detail_label(f"{z}×{y}×{x} voxels{tag}")

    def _on_ndisplay_changed(self, *_a):
        """2D<->3D switch: swap the source display between NATIVE (2D) and
        PROGRESSIVE (3D) when progressive is wanted, and follow the manual
        3D-detail slider's show/hide + level pin.

        The native<->progressive swap is the fix for the three 2D regressions the
        forced-progressive display caused (over-saturation, 2D lag, the 2D->3D
        GLIR access-violation crash): 2D is always native; progressive is built
        fresh WHILE ALREADY in 3D (the entering-3D swap is deferred one event-loop
        tick — see `_maybe_swap_progressive_on_ndisplay` — which is the path the
        de-risk harness proved crash-free)."""
        # [PROG] trace the ndisplay switch — the 2D->3D transition is where the
        # vispy '.activate on None' crash surfaced with a progressive layer live.
        try:
            nd = int(self.viewer.dims.ndisplay)
            self._log("PROG", f"ndisplay -> {nd}D "
                      f"(progressive_active={getattr(self, '_progressive_active', False)})")
        except Exception:
            pass
        self._maybe_swap_progressive_on_ndisplay()
        self._refresh_3d_detail_for_ndisplay()
        self._refresh_zoom_depth_for_ndisplay()
        self._apply_3d_detail_level()
        # A 2D↔3D swap re-creates the display layers (native↔progressive) with
        # voxel-derived units, dropping any FORCED scale-bar unit — re-apply the
        # live scale bar AFTER the (deferred) swap so a forced µm survives the
        # switch (scheduled after the swap's own singleShot → runs FIFO after it).
        try:
            from qtpy.QtCore import QTimer
            QTimer.singleShot(0, self._apply_live_scale_bar)
        except Exception:
            pass

    def _apply_3d_detail_level(self):
        """Set ``locked_data_level`` on the displayed MULTISCALE source layers:
        the chosen level while the viewer is in 3D, ``None`` (napari's automatic
        per-zoom picking) in 2D — so 2D pan/zoom is never pinned coarse.  A
        no-op for a normal single-scale source (guarded on ``layer.multiscale``)."""
        level = self._locked_3d_level
        if level is None:
            return
        # No-op while Step 1's layers are off the shared canvas (step-exit): the
        # never-disconnected ndisplay hook must not touch other steps' layers.
        if not self._source_layers_on_canvas:
            return
        try:
            in_3d = int(self.viewer.dims.ndisplay) == 3
        except Exception:
            in_3d = False
        for name in list(self._loaded_source_layers or []):
            try:
                layer = self.viewer.layers[name]
            except Exception:
                continue
            if not getattr(layer, "multiscale", False):
                continue
            try:
                n = len(getattr(layer, "level_shapes", []) or [])
                lvl = min(max(0, level), n - 1) if n else level
                layer.locked_data_level = lvl if in_3d else None
            except Exception as exc:
                self._log("STEP1-PREP", f"locked_data_level set failed: {exc}")

    # ------------------------------------------------------------------
    # Progressive-3D <-> native display swap (ndisplay-keyed)
    # ------------------------------------------------------------------
    # Progressive loading is engaged ONLY in 3D (its payoff), and NEVER in 2D
    # (where forcing it over-saturates, lags, and — during the 2D->3D transition
    # with a progressive layer already present — trips a vispy GLIR access
    # violation).  So the display is native by default and swaps to progressive
    # on ENTERING 3D, built fresh WHILE ALREADY IN 3D (deferred one event-loop
    # tick past the ndisplay transition — the path the de-risk harness proved
    # crash-free), then swaps back to native on entering 2D.  Each swap carries
    # the layer's CURRENT (slice-computed) contrast so 3D inherits the correct
    # value (fixes the coarsest-level over-saturation), and fails CLOSED (stays
    # native, per channel) on any error.

    def _is_multiscale_store_displayed(self) -> bool:
        """True iff a prepared MULTISCALE store is the current source display
        (all display channels are pyramid-level LISTS) — the only case the
        progressive swap + the 3D-detail slider apply to.  A normal small-file
        load (ndarray channels) returns False so those paths stay inert."""
        if not getattr(self, "_current_multiscale_store", None):
            return False
        channels = getattr(self, "_source_display_channels", None)
        if not channels:
            return False
        try:
            return all(
                isinstance(ch_data, (list, tuple)) for (_n, ch_data) in channels)
        except Exception:
            return False

    def _maybe_swap_progressive_on_ndisplay(self):
        """Swap the source display between native and progressive to match the
        current ndisplay + the Progressive-3D checkbox.  A no-op unless a
        multiscale store is displayed ON the shared canvas.

        Entering 3D with progressive wanted schedules the swap via
        ``QTimer.singleShot(0, ...)`` so it runs AFTER napari's ndisplay
        transition (detach-2D-node / attach-volume-node) has settled — running
        the layer swap synchronously inside the ndisplay handler is what trips
        the vispy GLIR ``ibuf.activate()`` access violation.  Every other
        direction (turn-off-in-3D, entering 2D) swaps back to native
        synchronously (safe: 2D never carries the crashing transition)."""
        # Item 4: when the ⟨processed⟩ STORE is the active canvas view (the clean
        # swap removed raw), drive the SAME progressive swap on the PROCESSED
        # layers instead of raw's.  Only one set is ever on the canvas, so this
        # dispatch is unambiguous — and it keeps the raw `_source_*` state pristine.
        if self._processed_store_is_active_view():
            self._maybe_swap_processed_progressive_on_ndisplay()
            return
        if not self._is_multiscale_store_displayed():
            return
        if not getattr(self, "_source_layers_on_canvas", False):
            return
        try:
            in_3d = int(self.viewer.dims.ndisplay) == 3
        except Exception:
            return
        wanted = self.progressive_wanted()
        if in_3d:
            if wanted and not self._progressive_active:
                self._log("PROG", "at/entering 3D + wanted -> deferring "
                          "native -> progressive swap (post-transition)")
                QTimer.singleShot(0, self._swap_native_to_progressive)
            elif not wanted and self._progressive_active:
                self._log("PROG", "3D + progressive turned OFF -> "
                          "progressive -> native swap")
                self._swap_progressive_to_native()
        else:
            if self._progressive_active:
                self._swap_progressive_to_native()

    def _refresh_3d_detail_for_ndisplay(self):
        """Show/hide the manual 3D-detail slider (feature #2) to match ndisplay +
        progressive state: shown ONLY in 3D with progressive OFF (native
        multiscale — the user pins the level manually); hidden in 2D and in
        3D-with-progressive-ON (the progressive engine owns level selection).

        A no-op unless a multiscale store is displayed ON the shared canvas."""
        if not self._is_multiscale_store_displayed():
            return
        # Off-canvas (step-exit): the store fields stay set while the layers are
        # gone, so the never-disconnected global ndisplay hook must not drive the
        # slider from a hidden Step-1 widget (lifecycle-audit LOW-1).  Its two
        # siblings (`_maybe_swap…`, `_apply_3d_detail_level`) already guard this.
        if not getattr(self, "_source_layers_on_canvas", False):
            return
        try:
            in_3d = int(self.viewer.dims.ndisplay) == 3
        except Exception:
            in_3d = False
        if in_3d and not self.progressive_wanted():
            self._configure_3d_detail(len(self._pyramid_level_shapes or []))
        else:
            hide_td = getattr(self.widget, "hide_3d_detail", None)
            if hide_td is not None:
                hide_td()
            if self.progressive_wanted():
                # Progressive owns the level -> keep the manual pin inert so
                # `_apply_3d_detail_level` no-ops (it early-returns on None).
                self._locked_3d_level = None

    def _swap_native_to_progressive(self):
        """Guard the native->progressive swap: suppress the transient per-channel
        `_on_layers_changed` churn while the remove/add loop runs, so the ② recipe
        + ③ picker reconcile ONCE against the SETTLED channel set.  Without this,
        each mid-swap remove fires `_on_layers_changed` at N-1 channels and
        `normalize_recipe` truncates the trailing channel's intensity ops then
        re-pads it EMPTY (the 2026-08-14 tool-wipe; two independent traces).
        Mirrors the `_adding_preview_layers` guard; exception-safe (flag cleared in
        `finally`), then one settled reconcile + selection reapply."""
        saved_sel = self._capture_channel_selection()
        self._swapping_source_display = True
        try:
            self._do_swap_native_to_progressive()
        finally:
            self._swapping_source_display = False
        self._on_layers_changed()
        self._reapply_channel_selection(saved_sel)

    def _do_swap_native_to_progressive(self):
        """Deferred (post-ndisplay-transition) swap: replace the native
        multiscale source layers with progressive-loading layers, carrying each
        layer's current contrast.  Fails CLOSED per channel (re-adds the native
        layer) so a seam failure never crashes the display or strands a channel.

        Re-checks its guards because it runs one event-loop tick late — the user
        may have flipped back to 2D, toggled progressive off, unloaded the
        source, or left the step in the meantime."""
        if not self._is_multiscale_store_displayed():
            return
        if not getattr(self, "_source_layers_on_canvas", False):
            return
        if self._progressive_active:
            return
        try:
            if int(self.viewer.dims.ndisplay) != 3:
                self._log("PROG", "deferred swap: no longer in 3D -> skip")
                return
        except Exception:
            return
        if not self.progressive_wanted():
            self._log("PROG",
                      "deferred swap: progressive no longer wanted -> skip")
            return
        try:
            from packages.epicycle_largedata import progressive as _prog
        except Exception as exc:
            self._log("PROG",
                      f"deferred swap import failed (stay native): {exc}")
            return

        saved_sel = self._capture_channel_selection()
        channels = list(self._source_display_channels or [])
        current_names = list(self._loaded_source_layers or [])
        cur_names = [current_names[i] if i < len(current_names) else ch_name
                     for i, (ch_name, _lv) in enumerate(channels)]
        # Capture appearances BEFORE hiding (hide clobbers the `visible`
        # snapshot), then hide EVERY native layer up front so no live coarsest-
        # level VOLUME visual survives into the remove/add loop (a second
        # channel's native volume node must not be live when the first channel's
        # progressive add triggers a metered GLIR flush — same multi-channel
        # crash window as the reverse swap).
        appearances = [self._capture_layer_appearance(cn) for cn in cur_names]
        for cn in cur_names:
            self._hide_layer_by_name(cn)
        new_names = []
        new_layers = []
        any_progressive = False
        for i, (ch_name, ch_levels) in enumerate(channels):
            appearance = appearances[i]
            self._remove_layer_by_name(cur_names[i])
            contrast = appearance.get("contrast_limits")
            layer = None
            # Physical-µm scale on the progressive layer (level-0 full-res →
            # factor 1 → scale == raw voxel).  The vendored engine ACCEPTS a
            # `scale=` kwarg and does not clobber it (progressive.py); passing it
            # keeps the progressive 3D glued to the physical box overlay.
            # SPIKE: a physical scale disables the engine's float32
            # auto-downscale,
            # so a huge+coarse volume COULD blank — dev-PC verification item.
            raw_vox = self._preview_spacing()
            try:
                kwargs = dict(name=ch_name,
                              colormap=appearance.get("colormap"),
                              blending="additive")
                # Pass a physical `scale=` ONLY for a KNOWN voxel.  For unknown
                # (1,1,1) OMIT it so the engine's float32 auto-downscale still
                # fires (the `scale is None` path) — preserving the pre-physical
                # byte-identical render for legacy no-voxel workspaces (a literal
                # `scale=(1,1,1)` would DISABLE the auto-downscale, so a
                # huge+coarse volume could blank).  Known-voxel scale was
                # confirmed in a render spike on the dev PC.
                if not self._is_unit_voxel(raw_vox):
                    kwargs["scale"] = tuple(float(v) for v in raw_vox)
                if contrast is not None:
                    kwargs["contrast_limits"] = contrast
                layer = _prog.add_progressive_image(
                    self.viewer, ch_levels, **kwargs)
            except Exception as exc:
                self._log("PROG", f"swap 2D-native -> 3D-progressive "
                          f"{ch_name!r}: seam raised {exc}")
                layer = None
            if layer is None:
                # Fail closed: re-add the native multiscale layer for THIS
                # channel so the display never loses it — at the SAME physical
                # scale as the progressive siblings (raw voxel).
                layer = self._readd_native_layer(
                    ch_name, ch_levels, appearance, voxel_size=raw_vox)
                self._log("PROG", f"swap 2D-native -> 3D-progressive "
                          f"{ch_name!r}: FELL BACK to native")
            else:
                any_progressive = True
                # Carry the "Zoom depth" choice onto the FRESH progressive layer
                # (swaps build new layers defaulting None), so the setting
                # persists across every 2D<->3D swap.  Best-effort.
                zoom_depth = getattr(self, "_zoom_depth_wanted", None)
                try:
                    layer._zoom_depth = zoom_depth
                except Exception:
                    pass
                self._log("PROG", f"swap 2D-native -> 3D-progressive "
                          f"{ch_name!r}: contrast={contrast} "
                          f"zoom_depth={zoom_depth}")
            # Keep every NEW layer HIDDEN for the BUILD INSTANT only: a visible
            # progressive VOLUME node drawn mid-build by the vendored engine's
            # metered GLIR flush is the `ibuf.activate() on None` crash (worst
            # multichannel — a second channel's build flush hits the first
            # channel's half-built node).  Revealed on the NEXT tick below, AFTER
            # the build flush drains — NOT synchronously here.  A same-tick
            # reveal both re-exposes the crash window AND stalls streaming: the
            # engine's `_check` no-ops while a layer is hidden (`if not
            # layer.visible: return`) and, if revealed in the same tick, runs
            # before the layer is actually visible + the camera bbox is ready,
            # leaving the level pinned coarse (the keep-hidden regression).
            try:
                layer.visible = False
            except Exception:
                pass
            new_layers.append((layer, appearance))
            new_names.append(getattr(layer, "name", ch_name))
        self._loaded_source_layers = new_names
        # The swap built NEW layer objects — refresh the object->name map so a
        # later rename resyncs the name caches against the live objects.
        self._track_source_layer_objects([lyr for (lyr, _app) in new_layers])
        self._progressive_active = any_progressive
        # Progressive owns the level: hide the manual slider + clear the pin.
        self._locked_3d_level = None
        hide_td = getattr(self.widget, "hide_3d_detail", None)
        if hide_td is not None:
            hide_td()
        # Display-only swap: the box tool's L0 handle + full-shape are untouched.
        self._source_layers_on_canvas = True
        self._on_layers_changed()
        self._reapply_channel_selection(saved_sel)
        # DEFERRED reveal (next event-loop tick, after the build's GL flush has
        # drained) + a streaming kick so 3D refinement re-engages immediately.
        self._log("PROG", f"built {len(new_layers)} progressive layer(s) hidden "
                  f"-> deferring reveal + streaming kick (next tick)")
        QTimer.singleShot(
            0, lambda built=list(new_layers):
            self._reveal_progressive_after_build(built))

    def _reveal_progressive_after_build(self, built):
        """Deferred reveal of the source layers built HIDDEN by the entering-3D
        swap (crash-safe: the hide spans only the synchronous build; this runs a
        tick later, past napari's deferred build/paint flush).  Restores each
        layer's appearance (visible/gamma) and nudges the vendored loader's
        ``_check`` toward re-engaging 3D refinement (belt-and-suspenders with the
        layer's own ``visible``-event debounce; full refinement lands once the
        post-transition camera settles).

        MUST reveal even when NO channel yielded a progressive layer — a total
        seam failure re-adds NATIVE fallbacks that were ALSO hidden for the build
        and would otherwise strand the 3D display blank; so the undo-guard keys
        on ``_source_layers_on_canvas`` + ``ndisplay == 3`` (the swap was undone)
        and NOT on ``_progressive_active``.  Acts on the LIVE layer looked up by
        name (robust to a rapid re-swap replacing the captured object)."""
        if not getattr(self, "_source_layers_on_canvas", False):
            return
        try:
            if int(self.viewer.dims.ndisplay) != 3:
                self._log("PROG", "deferred reveal: no longer in 3D -> skip")
                return
        except Exception:
            return
        revealed = 0
        for captured, appearance in built:
            name = getattr(captured, "name", None)
            if name is None:
                continue
            try:
                live = (self.viewer.layers[name]
                        if name in self.viewer.layers else None)
            except Exception:
                live = None
            if live is None:
                continue
            self._restore_layer_extras(live, appearance)   # visible + gamma
            self._kick_progressive_loader(live)
            revealed += 1
        # Progressive is now live -> surface the "Full depth when zoomed" toggle.
        self._refresh_zoom_depth_for_ndisplay()
        self._log("PROG", f"deferred reveal: {revealed} layer(s) shown + kicked")

    def _kick_progressive_loader(self, layer):
        """Nudge the vendored ``ProgressiveLoader``'s camera-driven refinement
        check so 3D detail streaming engages immediately after the deferred
        reveal (the loader lives in ``layer.metadata['progressive_loader']`` and
        normally only refines on a camera event).  Best-effort / never raises."""
        try:
            md = getattr(layer, "metadata", None) or {}
            loader = md.get("progressive_loader")
        except Exception:
            loader = None
        if loader is None:
            return
        try:
            check = getattr(loader, "_check", None)
            if callable(check):
                check()
        except Exception as exc:
            self._log("PROG", f"loader streaming kick skipped: {exc}")

    def _swap_progressive_to_native(self):
        """Guard the progressive->native swap (see `_swap_native_to_progressive`):
        suppress the transient per-channel `_on_layers_changed` churn so the ②
        recipe + ③ picker reconcile ONCE against the settled channel set instead of
        being truncated by a mid-swap N-1 dip.  Exception-safe; one settled
        reconcile + selection reapply after the flag clears."""
        saved_sel = self._capture_channel_selection()
        self._swapping_source_display = True
        try:
            self._do_swap_progressive_to_native()
        finally:
            self._swapping_source_display = False
        self._on_layers_changed()
        self._reapply_channel_selection(saved_sel)

    def _do_swap_progressive_to_native(self):
        """Swap the progressive source layers back to native multiscale layers
        (entering 2D, or turning progressive off in 3D).  Removing each
        progressive layer fires the seam's ``removed`` teardown, which
        uninstalls the process-wide GLIR meter once the last one goes away."""
        if not self._is_multiscale_store_displayed():
            return
        if not getattr(self, "_source_layers_on_canvas", False):
            return
        if not self._progressive_active:
            return
        saved_sel = self._capture_channel_selection()
        channels = list(self._source_display_channels or [])
        current_names = list(self._loaded_source_layers or [])
        cur_names = [current_names[i] if i < len(current_names) else ch_name
                     for i, (ch_name, _lv) in enumerate(channels)]
        # Capture appearances BEFORE hiding (hide clobbers the `visible`
        # snapshot), then hide EVERY progressive layer up front so no live
        # progressive VOLUME visual survives into the remove/re-add loop below
        # (multi-channel crash window — see `_hide_layer_by_name`).
        appearances = [self._capture_layer_appearance(cn) for cn in cur_names]
        for cn in cur_names:
            self._hide_layer_by_name(cn)
        new_names = []
        new_layers = []
        for i, (ch_name, ch_levels) in enumerate(channels):
            appearance = appearances[i]
            self._remove_layer_by_name(cur_names[i])
            layer = self._readd_native_layer(ch_name, ch_levels, appearance)
            new_names.append(getattr(layer, "name", ch_name))
            new_layers.append(layer)
        self._loaded_source_layers = new_names
        # The swap built NEW layer objects — refresh the object->name map so a
        # later rename resyncs the name caches against the live objects.
        self._track_source_layer_objects(new_layers)
        self._progressive_active = False
        self._source_layers_on_canvas = True
        # No progressive layer left -> hide the "Full depth when zoomed" toggle.
        self._refresh_zoom_depth_for_ndisplay()
        self._on_layers_changed()
        self._reapply_channel_selection(saved_sel)
        self._log("PROG", "swap 3D-progressive -> 2D-native")

    def _capture_layer_appearance(self, name) -> dict:
        """Snapshot the display attributes to carry across a swap: colormap,
        blending, visible, gamma, and the CURRENT contrast_limits (the
        slice-computed value that fixes the coarsest-level over-saturation).
        Best-effort — a missing attribute is simply omitted (contrast omitted
        => the progressive engine auto-estimates)."""
        out = {}
        try:
            layer = self.viewer.layers[name]
        except Exception:
            return out
        for attr in ("colormap", "blending", "visible", "gamma",
                     "contrast_limits"):
            try:
                val = getattr(layer, attr)
            except Exception:
                continue
            if attr == "contrast_limits" and val is not None:
                try:
                    val = list(val)
                except Exception:
                    continue
            out[attr] = val
        return out

    def _hide_layer_by_name(self, name):
        """Set a source layer invisible by name (disconnects its vispy visual
        from the scene graph WITHOUT removing it).  Used to hide ALL layers up
        front before a multi-channel swap so no live VOLUME visual remains while
        the remove/re-add loop runs — a second channel's volume node must not be
        live when the first channel's re-add triggers a metered GLIR flush.
        Best-effort / never raises."""
        viewer = getattr(self, "viewer", None)
        if viewer is None:
            return
        try:
            if name in viewer.layers:
                viewer.layers[name].visible = False
        except Exception:
            pass

    def _remove_layer_by_name(self, name):
        """Hide-then-remove a source layer by name (hide first so the vispy
        visual disconnects from the scene graph before the GPU texture is freed).
        Removing a progressive layer fires the seam's meter teardown."""
        viewer = getattr(self, "viewer", None)
        if viewer is None:
            return
        try:
            if name in viewer.layers:
                layer = viewer.layers[name]
                try:
                    layer.visible = False
                except Exception:
                    pass
                viewer.layers.remove(layer)
        except Exception as exc:
            self._log("PROG", f"swap remove {name!r} failed: {exc}")

    def _readd_native_layer(self, name, levels, appearance, voxel_size=None):
        """Re-add a native multiscale Image layer for *name*, restoring the
        captured appearance.  Used both to swap progressive->native and to fail
        closed on a per-channel progressive add failure.

        *voxel_size* is the physical-µm ``scale`` (level-0 is full-res, so
        ``factor == 1`` → ``scale == voxel``).  ``None`` reads ``_preview_spacing()``
        — correct for the RAW swaps (return-to-2D + progressive-fallback), which
        fire post-load when the spins hold this source's voxel.  The PROCESSED
        progressive fallback passes the store's OWN spacing EXPLICITLY so the
        fallback channel matches its progressive siblings' scale (grill 2 H3 — a
        voxel-index fallback would offset them by voxel× on the same 3D canvas)."""
        vox = (tuple(voxel_size) if voxel_size is not None
               else self._preview_spacing())
        scale = tuple(float(v) for v in vox)
        units = _units_for_voxel(vox)
        kwargs = dict(name=name, blending="additive", multiscale=True)
        if appearance.get("colormap") is not None:
            kwargs["colormap"] = appearance["colormap"]
        if appearance.get("contrast_limits") is not None:
            kwargs["contrast_limits"] = appearance["contrast_limits"]
        if "gamma" in appearance:
            kwargs["gamma"] = appearance["gamma"]
        if "visible" in appearance:
            kwargs["visible"] = appearance["visible"]
        try:
            layer = self.viewer.add_image(levels, **kwargs)
        except TypeError:
            # A viewer/fake that doesn't accept the extra display kwargs — fall
            # back to the minimal signature (name/colormap/blending/multiscale).
            layer = self.viewer.add_image(
                levels, name=name,
                colormap=appearance.get("colormap"),
                blending="additive", multiscale=True)
        # Physical-µm scale + units AFTER add (guarded) — see `_apply_layer_scale`.
        _apply_layer_scale(layer, scale)
        _apply_layer_units(layer, units)
        return layer

    def _restore_layer_extras(self, layer, appearance):
        """Restore the display attributes the progressive seam doesn't take as
        constructor kwargs (visible, gamma)."""
        for attr in ("visible", "gamma"):
            if attr in appearance:
                try:
                    setattr(layer, attr, appearance[attr])
                except Exception:
                    pass

    def _capture_channel_selection(self):
        """Snapshot the ticked channel names in `image_layer_combo` before a swap
        removes+re-adds the source layers (the intermediate removal fires
        `_on_layers_changed` -> `update_image_layer_list`, which drops the absent
        names from the selection).  Restored by `_reapply_channel_selection`
        after the swap so Extract keeps its channels without a manual re-tick.
        Best-effort / never raises."""
        combo = getattr(getattr(self, "widget", None), "image_layer_combo", None)
        if combo is None:
            return None
        try:
            return [i.text() for i in combo.selectedItems()]
        except Exception:
            return None

    def _reapply_channel_selection(self, saved):
        """Re-select the channel names captured by `_capture_channel_selection`
        after a swap has re-added the source layers.  Best-effort / never raises;
        a no-op when nothing was captured or the combo is gone."""
        if not saved:
            return
        combo = getattr(getattr(self, "widget", None), "image_layer_combo", None)
        if combo is None:
            return
        try:
            combo.blockSignals(True)
            for i in range(combo.count()):
                it = combo.item(i)
                it.setSelected(it.text() in saved)
            combo.blockSignals(False)
        except Exception:
            pass

    def on_progressive_3d_toggled(self, checked):
        """Progressive-3D checkbox flipped.

        2D is ALWAYS native (progressive's payoff is 3D-only), so a toggle in 2D
        makes NO visible change — it only affects the next 3D entry.  In 3D the
        swap is applied LIVE: ON -> native->progressive (deferred one tick), OFF
        -> progressive->native + the manual 3D-detail slider re-appears.  A no-op
        when no multiscale store is displayed.  Best-effort throughout."""
        if not self._is_multiscale_store_displayed():
            return
        try:
            in_3d = int(self.viewer.dims.ndisplay) == 3
        except Exception:
            in_3d = False
        if not in_3d:
            self._log("PROG", f"toggle in 2D (checked={checked}) -> no visible "
                      "change (2D is always native)")
            return
        try:
            self._maybe_swap_progressive_on_ndisplay()
            self._refresh_3d_detail_for_ndisplay()
            self._apply_3d_detail_level()
        except Exception as exc:
            self._log("PROG", f"progressive toggle swap failed: {exc}")

    def on_zoom_depth_changed(self, selection):
        """"Zoom depth" combo changed (progressive-3D only).

        *selection* is the combo's current index; it maps to a ``_zoom_depth``
        mode via ``ZOOM_DEPTH_MODES`` (Off->None, 25%->0.25, 50%->0.5, 75%->0.75,
        Full->1.0, Cube->"cube").  LIVE: the mode is read per-draw by the seam's
        ``_corners_for_locked_level`` — no re-swap / re-Prepare.  The chosen mode
        is ALWAYS remembered in ``_zoom_depth_wanted`` (so it persists across the
        layer-rebuilding 2D<->3D swaps, which reset the layer field to None); it
        is stamped onto the LIVE progressive layers only when progressive is
        active on the shared canvas.  Each stamped layer's
        ``apply_zoom_depth_now()`` then RECOMPUTES ``corner_pixels`` for the
        current level and reslices, so the depth change takes effect immediately
        (a bare attribute flip would sit inert until the next camera move — the
        seam only re-runs ``_corners_for_locked_level`` on a real draw).  The
        loader kick re-engages the streaming fetch afterwards.  Best-effort /
        never raises."""
        try:
            idx = int(selection)
        except Exception:
            idx = 0
        mode = ZOOM_DEPTH_MODES[idx] if 0 <= idx < len(ZOOM_DEPTH_MODES) else None
        # Remember the choice FIRST so it survives a later native<->progressive
        # swap even if there is nothing live to update right now.
        self._zoom_depth_wanted = mode
        # Item 4: when the ⟨processed⟩ store is the active view, stamp its live
        # progressive layers (raw is off canvas, so the raw loop below no-ops).
        # Mirrors the raw stamp+recompute+kick below; the mode still persists via
        # `_zoom_depth_wanted` for the next processed 3D entry.
        if self._processed_store_is_active_view():
            if not getattr(self, "_processed_progressive_active", False):
                return
            applied = 0
            for layer in list(getattr(self, "_processed_store_layers", None) or []):
                try:
                    layer._zoom_depth = mode
                except Exception:
                    continue
                try:
                    layer.apply_zoom_depth_now()
                except Exception:
                    pass
                self._kick_progressive_loader(layer)
                applied += 1
            return
        if not self._is_multiscale_store_displayed():
            return
        if not getattr(self, "_source_layers_on_canvas", False):
            return
        if not self._progressive_active:
            self._log("PROG", f"zoom-depth change (mode={mode!r}) stored "
                      "-> applies on the next progressive 3D entry")
            return
        applied = 0
        for name in list(self._loaded_source_layers or []):
            try:
                layer = self.viewer.layers[name]
            except Exception:
                continue
            try:
                layer._zoom_depth = mode
            except Exception:
                continue
            # Recompute this layer's tile corners for the current level + reslice
            # so the depth change shows WITHOUT waiting for a camera move, then
            # re-engage the streaming fetch.  Both best-effort.
            try:
                layer.apply_zoom_depth_now()
            except Exception:
                pass
            self._kick_progressive_loader(layer)
            applied += 1
        self._log("PROG", f"zoom-depth mode={mode!r} -> applied "
                  f"live to {applied} progressive layer(s)")

    def _apply_zoom_depth_combo_visibility(self, show):
        """Show the "Zoom depth" combo synced to ``_zoom_depth_wanted``, else hide
        it — shared by the raw + ⟨processed⟩ branches of
        :meth:`_refresh_zoom_depth_for_ndisplay`.  Sync-on-show: the combo's
        selection ALWAYS reflects ``_zoom_depth_wanted`` whenever it appears, so a
        store change (or any reset path) can't leave a stale mode selected while
        the layer renders a different one.  Best-effort; never raises."""
        try:
            if show:
                fn = getattr(self.widget, "configure_zoom_depth", None)
                if fn is not None:
                    fn(getattr(self, "_zoom_depth_wanted", None))
            else:
                fn = getattr(self.widget, "hide_zoom_depth", None)
                if fn is not None:
                    fn()
        except Exception:
            pass

    def _refresh_zoom_depth_for_ndisplay(self):
        """Show/hide the "Zoom depth" combo to match ndisplay + progressive
        state: shown ONLY in 3D with progressive wanted/active (the streaming
        engine owns the tile, and the mode sizes its depth axis); hidden in 2D
        and in 3D-with-progressive-OFF (native multiscale has no per-draw tile).
        OPPOSITE gate to ``_refresh_3d_detail_for_ndisplay`` (that slider shows
        when progressive is OFF; this combo shows when it is ON).

        Handles BOTH the raw source view and the ⟨processed⟩ store view (each
        keyed on its own progressive state); a no-op unless one of them is the
        active multiscale canvas view."""
        try:
            in_3d = int(self.viewer.dims.ndisplay) == 3
        except Exception:
            in_3d = False
        # ⟨processed⟩ store active view: mirror the raw show/hide but keyed on the
        # PROCESSED progressive state.  The raw gates below are False while the
        # processed store is on canvas, so without this branch the combo NEVER
        # appears for processed (the "zoom-depth only works for raw" gap).  The
        # two sibling ndisplay-refresh methods (`_maybe_swap_progressive_on_
        # ndisplay`, `on_zoom_depth_changed`) already dispatch to processed; this
        # one was the missed loose end (progressive was added in e3f5e0a).
        if self._processed_store_is_active_view():
            self._apply_zoom_depth_combo_visibility(bool(
                in_3d
                and (self.progressive_wanted()
                     or getattr(self, "_processed_progressive_active", False))))
            return
        if not self._is_multiscale_store_displayed():
            return
        if not getattr(self, "_source_layers_on_canvas", False):
            return
        self._apply_zoom_depth_combo_visibility(bool(
            in_3d
            and (self.progressive_wanted()
                 or getattr(self, "_progressive_active", False))))

    def _largedata_dir(self, workspace):
        return os.path.join(workspace, "artifacts", "step1", "largedata")

    def _resolve_source_id_for(self, state, src_path):
        """The STABLE ``source_id`` for *src_path*: the existing registry row's
        id (matched by normalized path), else a freshly allocated ``src_N`` off
        the current *state* (NOT yet appended — the finish-time upsert creates
        the row).  Guarantees a collision-suffixed store filename and the
        finish-time registry row agree on the id."""
        target = _norm_path(src_path)
        for s in state.get("sources", []):
            if _norm_path(s.get("path", "")) == target:
                sid = s.get("source_id")
                if sid:
                    return sid
        return _next_source_id(state.get("sources", []))

    def _row_claims_store(self, row, workspace, store_abs):
        """True if a registry *row* points its ``prepared_store`` (workspace-
        relative posix) at *store_abs*."""
        ptr = row.get("prepared_store")
        if not ptr:
            return False
        try:
            resolved = os.path.join(workspace, *str(ptr).split("/"))
            return _abspath_norm(resolved) == _abspath_norm(store_abs)
        except Exception:
            return False

    def _resolve_prepare_store_path(self, workspace, display_name, src_path):
        """Collision-safe output store path for a Prepare.

        Never ``overwrite=True``s a store that belongs to a DIFFERENT source
        (the same-stem data-loss bug).  The guard is STORE-ATTRS-CENTRIC:
        before choosing the base ``{alias}.zarr``, ``read_store_source`` the
        store already there and reuse/overwrite it ONLY when its recorded
        ``epicycle_source.path`` matches THIS source — OR the block is absent
        AND no registry row claims that store.  Otherwise fall back to
        ``{alias}__{source_id}.zarr``.  Also stashes the resolved
        ``_largedata_source_id`` so the finish-time upsert row and the (possibly
        suffixed) filename share the id.

        NEVER keys the reuse decision on registry rows alone — an orphan /
        unmigrated colliding store has no registry row yet would still be
        wiped by a registry-only guard.
        """
        largedata_dir = self._largedata_dir(workspace)
        state = load_state(workspace)
        source_id = self._resolve_source_id_for(state, src_path)
        self._largedata_source_id = source_id

        base_store = os.path.join(largedata_dir, f"{display_name}.zarr")
        suffixed = os.path.join(
            largedata_dir, f"{display_name}__{source_id}.zarr")

        try:
            exists = os.path.isdir(base_store) or os.path.exists(base_store)
        except Exception:
            exists = False
        if not exists:
            return base_store   # nothing there — the base name is free

        prov = read_store_source(base_store)   # per-store attrs; never raises
        if isinstance(prov, dict) and prov.get("path"):
            # Block present: reuse the base ONLY for the same source.
            if _abspath_norm(prov["path"]) == _abspath_norm(src_path):
                return base_store
            self._log("STEP1-PREP",
                      f"store name collision on {display_name}.zarr "
                      f"(belongs to {prov.get('path')!r}) -> {suffixed}")
            return suffixed
        # Block absent (foreign / legacy store): overwrite the base ONLY when no
        # OTHER registry row claims it.
        claimed_by_other = any(
            self._row_claims_store(s, workspace, base_store)
            and s.get("source_id") != source_id
            for s in state.get("sources", []))
        if claimed_by_other:
            self._log("STEP1-PREP",
                      f"store {display_name}.zarr claimed by another run "
                      f"-> {suffixed}")
            return suffixed
        return base_store

    def _persist_prepared_source(self, out_store, display_name):
        """Record the just-prepared source in the schema-v2 registry when
        "Prepare finishes" — LAST-WINS via :func:`upsert_source`.

        Sources ``source_axes`` from the store's OWN ``epicycle_source`` attrs
        (the honest on-disk order written at convert — NOT
        ``_source_axes_resolved``, which `_apply_multiscale_store` sets to the
        CANONICAL store layout so the box tool / lazy extract read the canonical
        level-0 correctly).  Voxel value+type stay COHERENT and HONEST (grill
        FIX 5): the registry is authoritative for the user's spacing, so —

        * if the source's row is ALREADY ``voxel_size_source == "user"`` (the
          user typed it; edit-saves run before Prepare, flushed at Prepare-start)
          leave it untouched — the store voxel is that same value;
        * else a REAL (non-1/1/1) store voxel is a value the Prepare baked from a
          DETECTED / header spacing, so it is recorded typed ``"detected"`` — never
          silently promoted to ``"user"``;
        * a 1/1/1 / absent store voxel upserts NEITHER (both skipped) so any prior
          value+type is left intact (B2-N4).

        Writes a WORKSPACE-RELATIVE ``prepared_store`` pointer (survives a
        workspace move), ``prepared_at`` + ``prepare_params``, and unions
        ``prepared`` into ``origin``.  Best-effort; never raises into the caller.
        """
        w = getattr(self, "widget", None)
        workspace = (w.output_file_path.text() or "").strip() if w else ""
        src_path = getattr(self, "_largedata_src_path", None)
        if not workspace or not src_path:
            return
        prov = read_store_source(out_store) or {}
        store_voxel = prov.get("voxel_size")
        try:
            rel = os.path.relpath(out_store, workspace).replace(os.sep, "/")
        except Exception:
            rel = None
        state = load_state(workspace)
        # Honest voxel typing: only a genuine user edit is "user".  A non-unit
        # store voxel is "user" ONLY if the row already recorded it as such;
        # otherwise it is a baked DETECTED value -> "detected".  1/1/1 / absent
        # store voxel -> touch nothing.
        existing_row = self._find_source_row(state, src_path)
        existing_vsource = (existing_row or {}).get("voxel_size_source")
        voxel_kwargs = {}
        if _classify_voxel_source(store_voxel) == "user":   # non-unit store voxel
            voxel_kwargs = {
                "voxel_size": list(store_voxel),
                "voxel_size_source": (
                    "user" if existing_vsource == "user" else "detected"),
            }
        upsert_source(
            state,
            path=src_path,
            source_id=getattr(self, "_largedata_source_id", None),
            display_name=display_name,
            source_axes=prov.get("source_axes"),
            prepared_store=rel,
            prepared_at=_now(),
            prepare_params=getattr(self, "_largedata_prepare_params", None),
            origin=["prepared"],
            **voxel_kwargs,
        )
        save_state(workspace, state)
        self._log("STEP1-PREP",
                  f"registry upsert: {display_name} source_axes="
                  f"{prov.get('source_axes')!r} prepared_store={rel!r}")

    def _fresh_prepared_store_for(self, src_path, *,
                                  allow_missing_source=False, reasons=None):
        """The path to an EXISTING, FRESH prepared multiscale store for
        *src_path*, or ``None``.

        *allow_missing_source* (Brick 4a a2) serves the store even when the raw
        file is GONE — journey (b), a workspace copied without its data.  It is
        HARD-GATED to a store resolved from the registry ``prepared_store``
        POINTER: the ``{display_name}.zarr`` name-reconstruction fallback is
        exactly the "same name, different data" case, and serving it blind would
        show one source's pixels under another's name.  When the raw IS present
        the flag changes nothing — freshness is decided exactly as before.

        *reasons* is an optional list the caller passes in to collect drift
        notes for the ② amber line: ``"raw_missing"``, ``"mtime_drift"``,
        ``"size_drift"``.  Purely informational; it never changes the verdict.

        "Fresh" is STRICT: the store dir must exist AND its OWN ``.zattrs``
        ``epicycle_source`` provenance (read via
        :func:`packages.epicycle_largedata.multiscale.read_store_source`) must
        record a ``path`` that resolves to *src_path*, a ``source_mtime`` that
        equals the source's CURRENT ``os.path.getmtime`` (within a tiny
        float-repr tolerance), AND a ``source_size`` that equals its CURRENT
        ``os.path.getsize``.  The size guard catches an mtime-preserving
        replacement (``cp -p`` / ``rsync -a`` / restore-previous-version / cloud
        sync) that mtime alone would miss — otherwise the OLD store's bytes get
        served for a NEW source (silent wrong data for display AND extract).
        Any drift — no store, an edited source (mtime or size moved), a store
        for a different source, a store missing the provenance block (incl. a
        legacy store with no ``source_size``), or an unreadable / corrupt store
        — returns ``None`` so the caller falls back to the normal TIFF read (or
        a real convert).  Never raises.

        The expected store path mirrors :meth:`prepare_large_data`:
        ``<workspace>/artifacts/step1/largedata/<display_name>.zarr`` where
        *display_name* is the widget's display-name (falling back to *src_path*'s
        basename stem, exactly as the load/prepare paths do) and *workspace* is
        the widget's output path.

        The provenance is PER-STORE (each store is self-describing), so this
        recognises EVERY prepared source in a shared ``largedata/`` dir — not
        just the most-recently-prepared one.  (The shared ``result_manifest.json``
        sidecar ``convert_to_ome_zarr`` also writes is last-writer-wins, so it
        could only ever match one store; the per-store attrs remove that
        limitation and its former ``path``-re-check workaround.)

        Store RESOLUTION is schema-v2 aware: the registry
        ``sources[...].prepared_store`` **pointer** (workspace-relative posix)
        is tried first — it survives a collision-suffixed name AND a workspace
        move — and only when the pointer is null (every v1 workspace until a
        Brick-2 restore migrates it) is ``{display_name}.zarr`` RECONSTRUCTED.
        When the store is pointer-bound, the recorded abspath is demoted to a
        HINT (A3a): a path mismatch (workspace/source moved) must not reject a
        registry-bound store — the ``{size, mtime}`` fingerprint alone gates
        freshness.  The reconstruction fallback keeps the strict path match (the
        name is the only binding), so a same-named store for a different source
        is still refused.
        """
        try:
            if not src_path:
                return None
            w = getattr(self, "widget", None)
            if w is None:
                return None
            workspace = (w.output_file_path.text() or "").strip()
            if not workspace:
                return None
            display_name = (w.display_name_edit.text() or "").strip()
            if not display_name:
                display_name = os.path.splitext(os.path.basename(src_path))[0]

            # A5a: prefer the registry pointer; fall back to name reconstruction.
            store = None
            resolved_via_pointer = False
            try:
                state = load_state(workspace)
                target = _norm_path(src_path)
                for s in state.get("sources", []):
                    if _norm_path(s.get("path", "")) == target:
                        ptr = s.get("prepared_store")
                        if ptr:
                            store = os.path.join(
                                workspace, *str(ptr).split("/"))
                            resolved_via_pointer = True
                        break
            except Exception:
                store = None
            if store is None:
                store = os.path.join(
                    workspace, "artifacts", "step1", "largedata",
                    f"{display_name}.zarr")
            if not os.path.isdir(store):
                return None
            prov = read_store_source(store)   # per-store attrs; never raises
            if not isinstance(prov, dict):
                return None
            psrc = prov.get("path")
            pmtime = prov.get("source_mtime")
            psize = prov.get("source_size")
            # size + mtime are REQUIRED for freshness: an mtime-preserving
            # replacement (cp -p / rsync -a / restore-previous-version / cloud
            # sync) leaves the mtime untouched but changes the byte-size, so
            # mtime alone would serve the OLD store's bytes for a NEW source
            # (silent wrong data — display AND extract).  A store written before
            # source_size existed (legacy) has psize=None → NOT fresh → safe
            # fallback to the normal TIFF read (or a re-prepare); never served.
            if pmtime is None or psize is None:
                return None
            # A3a: the recorded path is IDENTITY only for the reconstruction
            # fallback (the name is the sole binding there); for a pointer-bound
            # store it is a HINT and a mismatch (workspace/source moved) must NOT
            # reject the store — the fingerprint below decides freshness.
            if not resolved_via_pointer:
                if psrc is None or _abspath_norm(psrc) != _abspath_norm(src_path):
                    return None
            phash = prov.get("source_head_sha")
            if not os.path.isfile(src_path):
                # a2: the raw is gone.  Only a POINTER-bound store is this
                # source's store beyond doubt; a name-reconstructed one is
                # refused (its name is the only binding, and the file that
                # would prove it is missing).
                if not allow_missing_source or not resolved_via_pointer:
                    return None
                if isinstance(reasons, list) and "raw_missing" not in reasons:
                    reasons.append("raw_missing")
                return store
            try:
                current = os.path.getmtime(src_path)
                current_size = os.path.getsize(src_path)
            except OSError:
                return None
            mtime_drift = abs(float(pmtime) - float(current)) > 1e-6
            size_drift = int(psize) != int(current_size)
            if isinstance(reasons, list):
                if mtime_drift and "mtime_drift" not in reasons:
                    reasons.append("mtime_drift")
                if size_drift and "size_drift" not in reasons:
                    reasons.append("size_drift")
            # a3 (as corrected by Brick 4c item 2): a size change always falls
            # back to the raw.  A size MATCH with a moved mtime is usually a
            # copy (`cp` without -p, a cloud sync, a restore) and re-Preparing a
            # whole unchanged stack for that was pure waste -- but it can ALSO
            # be an in-place pixel edit of an uncompressed TIFF, which keeps the
            # byte count exactly.  So the tie is broken on the SAMPLED content
            # digest recorded at Prepare (`source_head_sha`):
            #
            #   digest equal    -> genuinely the same bytes; serve the store,
            #                      and the "new date, same contents" amber line
            #                      is now a TRUE statement rather than a guess.
            #   digest differs  -> the file really changed; treat it exactly
            #                      like a size drift (raw + "this file has
            #                      changed"), never the stale store.
            #   no digest       -> a store written before this field existed.
            #                      Be CONSERVATIVE and fall back to the raw,
            #                      i.e. the pre-4a behaviour; a re-Prepare
            #                      records the digest and restores the shortcut.
            if size_drift:
                return None
            if mtime_drift:
                if not phash:
                    # A store from before the digest existed.  We cannot tell a
                    # copy from an edit, so we fall back to the raw -- but we
                    # must not TELL the user the file changed, because we do not
                    # know that (Brick 4e item 7).  Its own amber line says so.
                    if isinstance(reasons, list):
                        if "mtime_drift" in reasons:
                            reasons.remove("mtime_drift")
                        if "unverifiable" not in reasons:
                            reasons.append("unverifiable")
                    return None
                if sampled_file_sha(src_path) != phash:
                    if isinstance(reasons, list):
                        if "mtime_drift" in reasons:
                            reasons.remove("mtime_drift")
                        if "size_drift" not in reasons:
                            reasons.append("size_drift")
                    return None
            return store
        except Exception:
            return None

    def _on_large_data_progress(self, done, total, token):
        """A pyramid level landed - drive ②'s shared bar as a REAL percentage.

        Token-guarded: a Prepare that was cancelled (or superseded) keeps
        converting until its current level finishes, and its late ticks must not
        repaint a bar that now belongs to another operation.  Also declines once
        Step 1 has been left, and after the lock has been released (the terminal
        handler owns the bar from then on).  Never raises - progress is
        decoration, and this runs on the GUI thread.
        """
        if token != getattr(self, "_largedata_token", 0):
            return
        if getattr(self, "_left_step1", False):
            return
        if not getattr(self, "_largedata_preparing", False):
            return
        try:
            self.widget._busy_progress(
                getattr(self, "_largedata_busy_label", None)
                or "Preparing for large-data viewing…",
                int(done), int(total))
        except Exception as exc:      # noqa: BLE001
            self._log("STEP1-PREP", f"progress paint skipped: {exc}")

    def _on_large_data_ready(self, out_store, display_name, token):
        """Main-thread continuation after the background convert finishes.

        STALE GUARD first: a newer prepare or a Cancel bumped
        `_largedata_token`, so this store is superseded — do NOT touch the
        viewer or the busy state.

        Otherwise delegates the display to :meth:`_apply_multiscale_store`
        (open the store lazily → per-channel MULTISCALE layers → L0 state → box
        full_shape → status), then ALWAYS clears the busy lock in the terminal
        funnel, so a display-time failure can never leave the surface stuck.

        After a successful convert it also records the prepared source in the
        schema-v2 registry (:meth:`_persist_prepared_source`) — the honest
        on-disk ``source_axes`` + the workspace-relative store pointer — so the
        source no longer has to be re-specified.  Best-effort: the
        store is already valid on disk, so a persistence hiccup must not break
        the UI, and it runs AFTER the busy funnel so a display failure does not
        skip it.
        """
        if token != self._largedata_token:
            self._log("STEP1-PREP",
                      f"stale finish token={token} "
                      f"(current={self._largedata_token}) — discarded")
            return
        try:
            # src_path=None → falls back to `_largedata_src_path` (byte-identical
            # to the pre-refactor inline handler).
            self._apply_multiscale_store(out_store, display_name)
        finally:
            self.widget._busy_hide()
            self._set_step1_busy(False)
            self._largedata_preparing = False
            self._set_prepare_button_cancel(False)
        try:
            self._persist_prepared_source(out_store, display_name)
        except Exception as exc:   # noqa: BLE001 — registry is non-critical here
            self._log("STEP1-PREP", f"registry upsert skipped: {exc}")

    def _on_large_data_prep_error(self, msg, token):
        """Main-thread handler for a background convert that raised.

        STALE GUARD first (a Cancel / re-launch superseded this convert).
        Otherwise writes the red status + clears the busy lock and restores the
        Prepare button (mirrors :meth:`_on_source_load_error`)."""
        if token != self._largedata_token:
            self._log("STEP1-PREP",
                      f"stale error token={token} "
                      f"(current={self._largedata_token}) — discarded")
            return
        self.widget.source_status_label.setText(
            f"⚠  Failed to prepare: {msg}")
        self.widget.source_status_label.setStyleSheet(
            "font-size: 10px; color: #e07070;")
        self.widget._busy_hide()
        self._set_step1_busy(False)
        self._largedata_preparing = False
        self._set_prepare_button_cancel(False)

    def cancel_large_data_prep(self):
        """Discard the in-flight convert; unlock only once the worker DRAINS.

        The conversion is a single, non-interruptible dask stream, so Cancel
        DISCARDS the result (bumps the token so the finished handler goes stale)
        — but the worker keeps WRITING the store until it reaches its next
        cancel check.  Unlocking immediately let the user start a second Prepare
        onto the SAME ``out_store`` while the first was still writing it: two
        writers on one path.  So the surface stays
        locked (Prepare disabled, busy bar renamed) until the cancelled thread's
        ``finished`` reaches :meth:`_cleanup_largedata_thread`, which performs
        the one unlock.  With no thread to drain (a launch that never started
        one) the unlock happens here, as before.
        """
        self._largedata_token += 1
        worker = getattr(self, "_largedata_worker", None)
        if worker is not None:
            worker.request_cancel()
        thread = getattr(self, "_largedata_thread", None)
        if thread is not None:
            # Retain the pair against GC (mirrors the source-load / extract
            # drains) and hold the lock until it finishes.
            try:
                if (thread, worker) not in self._draining_largedata_threads:
                    self._draining_largedata_threads.append((thread, worker))
            except Exception:
                pass
            self._largedata_cancel_draining_thread = thread
            # Only while Step 1 is still on screen: `on_step_exit` sets
            # `_left_step1` and THEN cancels, so an unguarded `_busy_show` would
            # paint the Step-1 busy bar over whatever step the user moved to.
            # The drain's own unlock (`_cleanup_largedata_thread`) only hides
            # and re-enables, so it is safe either way.
            if not getattr(self, "_left_step1", False):
                try:
                    self.widget._busy_show(
                        "Cancelling — finishing the current write…")
                except Exception:
                    pass
            btn = getattr(self.widget, "prepare_largedata_button", None)
            if btn is not None:
                try:
                    btn.setEnabled(False)      # no second Prepare, no re-Cancel
                except Exception:
                    pass
            self._log("STEP1-PREP", "cancel requested — draining the worker")
            return
        self.widget._busy_hide()
        self._set_step1_busy(False)
        self._largedata_preparing = False
        self._set_prepare_button_cancel(False)
        self._log("STEP1-PREP", "cancelled")

    def on_prepare_or_cancel_clicked(self):
        """The Prepare button doubles as Cancel while a convert is in flight.

        Dispatches on `_largedata_preparing` (True only between launch and
        finish/error/cancel) so a click DURING a convert cancels it, and a click
        otherwise starts a fresh prepare."""
        if getattr(self, "_largedata_cancel_draining_thread", None) is not None:
            return                     # already cancelled, still draining
        if getattr(self, "_largedata_preparing", False):
            self.cancel_large_data_prep()
        else:
            self.prepare_large_data()

    def _set_prepare_button_cancel(self, active: bool) -> None:
        """Flip the Prepare button to 'Cancel' (kept clickable) while a convert
        runs, or back to its idle label when it finishes.  Best-effort; touches
        only the button text/icon/enabled state (mirrors
        :meth:`_set_load_button_cancel`)."""
        btn = getattr(self.widget, "prepare_largedata_button", None)
        if btn is None:
            return
        try:
            if active:
                if self._prepare_button_text is None:
                    self._prepare_button_text = btn.text()
                    self._prepare_button_icon = btn.icon()
                btn.setEnabled(True)          # stays clickable as Cancel
                btn.setText("Cancel")
                try:
                    from napari_epicycle.ui_shell.theme import apply_button_icon
                    apply_button_icon(btn, "mdi6.close-circle-outline")
                except Exception:
                    pass
            elif self._prepare_button_text is not None:
                btn.setText(self._prepare_button_text)
                if self._prepare_button_icon is not None:
                    btn.setIcon(self._prepare_button_icon)
                self._prepare_button_text = None
                self._prepare_button_icon = None
        except Exception:
            pass

    def _cleanup_largedata_thread(self, thread=None):
        """Drop the finished prepare (thread, worker) refs — identity-guarded,
        removes any matching draining pair, and `deleteLater`s the thread
        (mirrors :meth:`_cleanup_source_load_thread`)."""
        draining = getattr(self, "_draining_largedata_threads", None)
        if draining is not None and thread is not None:
            self._draining_largedata_threads = [
                (t, w) for (t, w) in draining if t is not thread]
        if thread is None or thread is self._largedata_thread:
            self._largedata_thread = None
            self._largedata_worker = None
            # Brick 5b item 6: the label belongs to the run that just ended.
            # `thread.finished` fires for EVERY terminal (finished / error /
            # cancelled all `thread.quit`), so this is the one place that covers
            # them all.
            self._largedata_busy_label = None
        # A CANCELLED convert holds the ② lock until its worker actually stops
        # (see :meth:`cancel_large_data_prep`) — this is that one unlock.
        if thread is not None and thread is getattr(
                self, "_largedata_cancel_draining_thread", None):
            self._largedata_cancel_draining_thread = None
            # `_set_step1_busy(False)` restores every locked widget (incl. the
            # Prepare button) to its PRE-busy enabled state, which is also what
            # undoes the extra disable Cancel applied.
            try:
                self.widget._busy_hide()
            except Exception:
                pass
            self._set_step1_busy(False)
            self._largedata_preparing = False
            self._set_prepare_button_cancel(False)
            self._log("STEP1-PREP", "cancelled convert drained — surface unlocked")
        if thread is not None:
            try:
                thread.deleteLater()
            except Exception:
                pass

    def _connect_source_shutdown_hooks(self) -> None:
        """Wire viewer-close + widget-destroy to `shutdown()` so an in-flight
        source read is always joined before teardown (mirrors
        `fusion_reconstruction/controller.py::_connect_shutdown_hooks`).

        Deliberately NOT wired to `QApplication.aboutToQuit`: that would keep
        every controller alive on the app singleton for the whole session (a
        ref leak across the test suite), and on app quit the widget tree is
        destroyed anyway — `widget.destroyed` fires and joins the thread.
        """
        try:
            if hasattr(self.viewer, "events") and hasattr(
                    self.viewer.events, "close"):
                self.viewer.events.close.connect(
                    lambda *_: self.shutdown())
        except Exception:
            pass
        try:
            if self.widget is not None and hasattr(self.widget, "destroyed"):
                self.widget.destroyed.connect(lambda *_: self.shutdown())
        except Exception:
            pass

    def shutdown(self, *_):
        """Stop every in-flight / draining Step-1 worker QThread before teardown
        — source-read, large-data-prepare, preview, process, streaming-store,
        extract, AND the ④ Z-crop trim.

        Bumps every family's token (so any queued finished/error handler early-
        returns and never touches a torn-down widget/viewer), then
        `request_cancel()` + `quit()` + `wait()` on the current pair AND every
        draining pair of ALL thread families, so a QThread is never destroyed
        while still running
        (the `QThread: Destroyed while thread is still running` native abort).
        Idempotent + best-effort; touches only controller thread state, never
        the (possibly deleted) widget/viewer.
        """
        if getattr(self, "_source_shutdown_done", False):
            return
        self._source_shutdown_done = True
        # Flush any pending DEBOUNCED source-state save BEFORE teardown (Bug B):
        # the 400ms recipe save timer is a child of the dying widget, so on a plain
        # napari close (viewer.events.close / widget.destroyed → here) the last 1-2
        # recipe edits made within that window never reach disk and vanish next
        # session.  `_flush_source_state_save` is pure disk (no widget/viewer touch)
        # and writes the edit-time-captured buffer, so it is safe during teardown.
        try:
            self._flush_source_state_save()
        except Exception:
            pass
        self._source_load_token = getattr(self, "_source_load_token", 0) + 1
        self._largedata_token = getattr(self, "_largedata_token", 0) + 1
        self._preview_token = getattr(self, "_preview_token", 0) + 1
        self._process_token = getattr(self, "_process_token", 0) + 1
        self._extract_token = getattr(self, "_extract_token", 0) + 1
        self._trim_token = getattr(self, "_trim_token", 0) + 1
        pairs = []
        cur = getattr(self, "_source_load_thread", None)
        if cur is not None:
            pairs.append((cur, getattr(self, "_source_load_worker", None)))
        pairs.extend(getattr(self, "_draining_source_threads", None) or [])
        # The EXTRACT worker (F1) rides the same never-destroy-a-running-QThread
        # invariant — a mid-run slice/write must be cancelled + joined (or
        # orphaned) in the same sweep, never nulled into GC while its QThread runs.
        cur_ex = getattr(self, "_extract_thread", None)
        if cur_ex is not None:
            pairs.append((cur_ex, getattr(self, "_extract_worker", None)))
        pairs.extend(getattr(self, "_draining_extract_threads", None) or [])
        # The large-data prepare threads share the same never-destroy-a-running-
        # QThread invariant, so they are cancelled + joined in the same sweep.
        cur_ld = getattr(self, "_largedata_thread", None)
        if cur_ld is not None:
            pairs.append((cur_ld, getattr(self, "_largedata_worker", None)))
        pairs.extend(getattr(self, "_draining_largedata_threads", None) or [])
        # The preprocessing PREVIEW thread rides the same invariant.
        cur_pv = getattr(self, "_preview_thread", None)
        if cur_pv is not None:
            pairs.append((cur_pv, getattr(self, "_preview_worker", None)))
        pairs.extend(getattr(self, "_draining_preview_threads", None) or [])
        # …as does the WHOLE-VOLUME process thread (Brick 1b-1).
        cur_proc = getattr(self, "_process_thread", None)
        if cur_proc is not None:
            pairs.append((cur_proc, getattr(self, "_process_worker", None)))
        pairs.extend(getattr(self, "_draining_process_threads", None) or [])
        # …and the STREAMING (out-of-core) process thread (Phase 2a): a
        # mid-run streaming write MUST be cancelled + joined (or orphaned) in the
        # same sweep, never nulled into GC while its QThread runs.
        cur_proc_store = getattr(self, "_process_store_thread", None)
        if cur_proc_store is not None:
            pairs.append((cur_proc_store, getattr(self, "_process_store_worker", None)))
        pairs.extend(getattr(self, "_draining_process_store_threads", None) or [])
        # …and the ④ Z-crop TRIM thread (in-place patch overwrite): a mid-run
        # `os.replace` is UNINTERRUPTIBLE, so the running QThread MUST be joined
        # (or orphaned) in this same sweep — never nulled into GC below while it
        # runs (the native "destroyed while running" abort).  `on_step_exit` is a
        # step-nav hook, NOT guaranteed on a hard napari close, so this is the only
        # place a close-mid-trim is swept.  The sweep's try/except tolerates
        # `_TrimWorker` lacking `request_cancel` (a trim can't be cancelled anyway).
        cur_trim = getattr(self, "_trim_thread", None)
        if cur_trim is not None:
            pairs.append((cur_trim, getattr(self, "_trim_worker", None)))
        pairs.extend(getattr(self, "_draining_trim_threads", None) or [])
        for thread, worker in pairs:
            try:
                if worker is not None:
                    worker.request_cancel()
            except Exception:
                pass
            try:
                if thread is not None:
                    thread.quit()
                    thread.wait(5000)
            except Exception:
                pass
        # ANY thread that did NOT stop within the wait is running an uninterruptible
        # op — a process CLAHE / bleach / geometry, the large-data convert stream
        # (`convert_to_ome_zarr`, minutes on 35GB), or an eager multi-GB source read.
        # Do NOT null it into GC below: that destroys a running QThread → the native
        # "destroyed while running" abort.  Orphan every still-running one so it
        # self-cleans when its op returns (extended from the process
        # family to ALL families after the diff-grill found close-mid-Prepare hit the
        # same crash the process guard fixed).
        for thread, worker in pairs:
            try:
                if thread is not None and thread.isRunning():
                    _orphan_running_thread(thread, worker)
            except Exception:
                pass
        self._source_load_thread = None
        self._source_load_worker = None
        self._draining_source_threads = []
        self._source_loading = False
        self._largedata_thread = None
        self._largedata_worker = None
        self._draining_largedata_threads = []
        self._largedata_preparing = False
        self._largedata_cancel_draining_thread = None
        self._preview_thread = None
        self._preview_worker = None
        self._draining_preview_threads = []
        self._process_thread = None
        self._process_worker = None
        self._draining_process_threads = []
        self._process_store_thread = None
        self._process_store_worker = None
        self._draining_process_store_threads = []
        self._processing = False
        self._extract_thread = None
        self._extract_worker = None
        self._draining_extract_threads = []
        self._extract_running = False
        self._extract_launching = False
        self._trim_thread = None
        self._trim_worker = None
        self._draining_trim_threads = []
        self._trim_running = False
        self._trim_launching = False

    def _track_source_layer_objects(self, layers):
        """(Re)seed the object->last-known-name map for the tracked source layers.

        The progressive<->native 2D/3D swap + its two sibling readers identify
        the source layers BY NAME from ``_loaded_source_layers`` /
        ``_source_display_channels``.  A rename would leave those name caches
        stale; :meth:`_on_layer_renamed` repairs them by matching the renamed
        layer on OBJECT IDENTITY against this map (the old name is already gone
        from the layer by the time the event fires).  Rebuilt fresh at EVERY site
        that (re)builds the name caches — the load/reshow site
        (:meth:`_add_source_display_layers`) and BOTH swap re-add sites (the swap
        creates NEW layer objects) — so it never carries a stale object.
        Best-effort / never raises."""
        name_map = getattr(self, "_source_layer_name_map", None)
        if name_map is None:
            name_map = self._source_layer_name_map = weakref.WeakKeyDictionary()
        try:
            name_map.clear()
            for layer in layers or []:
                if layer is None:
                    continue
                try:
                    name_map[layer] = layer.name
                except Exception:
                    pass
        except Exception:
            pass

    def _add_source_display_layers(self, channels, voxel_size=None,
                                   fit_view=True, contrast_store_key=None):
        """Add one napari Image layer per source channel; return their names.

        *voxel_size* is the RESOLVED raw ``(vz, vy, vx)`` for the physical-µm
        display ``scale`` + ``units``.  ``None`` → read the spins live
        (``_preview_spacing``) — correct on the plain-load + step-reshow paths
        where the spins already hold this source's voxel; the prepared-store path
        passes it EXPLICITLY because the store voxel is pushed to the spins AFTER
        these layers are added (`_apply_store_voxel_to_widget`).

        GUARD WRAPPER (Bug A): the channel-add loop adds layers one at a time, and
        each `add_image` fires `_on_layers_changed` → `_sync_recipe_channels` →
        `normalize_recipe(recipe, k)` at the TRANSIENT k (1, then 2, …).  A k=N-1
        pass truncates the trailing channel's recipe ops (`channels[:N-1]`), then
        the k=N pass re-pads it EMPTY — silently wiping ch1's steps on EVERY load /
        re-show (the SAME class the swap path already guards, but the load path
        never did → the persisted-recipe step-loss the user hit).  So suppress the
        transient churn with `_loading_source_layers` and reconcile ONCE at the
        settled full count.  Mirrors the swap wrapper; exception-safe.
        """
        self._loading_source_layers = True
        try:
            layer_names = self._do_add_source_display_layers(
                channels, voxel_size=voxel_size,
                contrast_store_key=contrast_store_key)
        finally:
            self._loading_source_layers = False
        # One settled reconcile against the FULL channel set (recipe + ③ picker
        # rebuild once, never the transient N-1 count).
        try:
            self._on_layers_changed()
        except Exception as exc:
            self._log("STEP1-LOAD", f"settled reconcile after load: {exc}")
        try:
            self._diag_physical_layers("after source-display add")
        except Exception:
            pass
        # Re-fit the camera to the freshly-loaded layers' PHYSICAL extent: a
        # non-unit voxel (e.g. 0.13) shrinks the world box, so without this the
        # image loads zoomed FAR OUT (the camera was framed for the old scale=1
        # extent).  Only on a genuine load — NOT a step-re-enter reshow, where the
        # camera already holds the user's Step-1 view.  Guarded.
        if fit_view and layer_names:
            try:
                self.viewer.reset_view()
            except Exception:
                pass
        return layer_names

    def _do_add_source_display_layers(self, channels, voxel_size=None,
                                      contrast_store_key=None):
        """Add one napari Image layer per source channel; return their names.

        Factored out of :meth:`_on_source_loaded` so :meth:`_reshow_source_layers`
        can re-add the exact same layers on step re-enter without re-reading the
        source (they are re-added from the retained channel refs).  Runs UNDER the
        `_add_source_display_layers` guard wrapper (Bug A) — the per-add
        `_on_layers_changed` churn is suppressed; the wrapper does one settled
        reconcile after.

        *voxel_size* (see :meth:`_add_source_display_layers`) drives the
        physical-µm ``scale`` + ``units``; ``None`` reads ``_preview_spacing()``.
        Level-0 IS full-res here (plain add or a full pyramid), so ``factor == 1``
        → ``scale == voxel_size`` (napari derives coarser levels' scale from the
        shape ratios for a ``multiscale=True`` add).
        """
        _CHANNEL_CMAPS = _SOURCE_DISPLAY_CMAPS  # single source of truth (Brick 2c)
        layer_names = []
        added_layers = []
        # Physical-µm scale + units.  Guarded read (never raises out of the add
        # loop): a controller stand-in without `_preview_spacing` reads as unit.
        if voxel_size is not None:
            vox = tuple(voxel_size)
        else:
            fn = getattr(self, "_preview_spacing", None)
            try:
                vox = tuple(fn()) if callable(fn) else (1.0, 1.0, 1.0)
            except Exception:
                vox = (1.0, 1.0, 1.0)
        units = _units_for_voxel(vox)
        scale = tuple(float(v) for v in vox)
        channels = list(channels or [])
        # This method ALWAYS adds NATIVE napari layers.  Progressive-3D is NOT
        # engaged here: forcing the whole display through the progressive path
        # taxes / breaks 2D — over-saturation (contrast pinned to the coarsest
        # mean-downsampled level), 2D lag (cubic-chunk Z-amplification), and a
        # 2D->3D vispy GLIR crash.  Progressive's payoff is 3D-only, so it is
        # engaged by a SEPARATE ndisplay-keyed swap (see
        # `_maybe_swap_progressive_on_ndisplay`) that builds the progressive
        # layer WHILE ALREADY IN 3D and carries the native slice-computed
        # contrast.  The initial display (2D or 3D) is therefore always native.
        # Re-added natively on every call (initial load + step re-enter), so
        # `_progressive_active` is reset here — a native display is never active.
        self._progressive_active = False
        for i, (ch_name, ch_data) in enumerate(channels):
            cmap = (_CHANNEL_CMAPS[i % len(_CHANNEL_CMAPS)]
                    if len(channels) > 1 else "gray")
            # A MULTISCALE channel arrives as a LIST of pyramid levels (the
            # "Prepare for large-data viewing" path) — add it with
            # multiscale=True so napari builds a lazy pyramid layer.  A plain
            # ndarray channel (the normal small-file load) is added
            # byte-identically as before (NO `multiscale` kwarg), so the eager
            # path + its viewer fakes are untouched.
            if isinstance(ch_data, (list, tuple)):
                # Physical-µm display: level-0 is full-res, so `scale = voxel`
                # (`factor == 1`) puts world at `full_res_voxel × voxel` — the
                # SAME physical world the box overlay now multiplies into (via
                # `set_voxel_size`), so they stay glued.  On the prepared-store
                # path the RESOLVED voxel is passed EXPLICITLY (the store voxel is
                # pushed to the spins only AFTER this add), never a live spin read.
                #
                # Pass an EXPLICIT contrast_limits computed from a FINE pyramid
                # level (percentiles).  napari's default takes the min/max of the
                # COARSEST level (`Image._calc_data_range` -> `self.data[-1]`),
                # which under-estimates the max for sparse-bright channels (e.g.
                # nuclei) whose peaks are diluted by mean-downsampling -> the fine
                # levels then over-saturate to white (worst in 3D, where the MIP
                # compounds it).  None -> fall back to napari's default.
                # Passing contrast_limits ALSO suppresses napari's deferred
                # `_should_calc_clims` recompute (only armed when contrast_limits
                # is None) — so it won't clobber our value with the coarsest
                # min/max after the first slice loads.  (napari pins the slider
                # RANGE to these limits; widened to the dtype extent just below
                # so the user can still adjust.)
                cl = self._estimate_multiscale_contrast(
                    ch_data, store_key=contrast_store_key, channel_index=i)
                add_kwargs = dict(name=ch_name, colormap=cmap,
                                  blending="additive", multiscale=True)
                if cl is not None:
                    add_kwargs["contrast_limits"] = cl
                try:
                    layer = self.viewer.add_image(ch_data, **add_kwargs)
                except TypeError:
                    # A viewer/fake that doesn't accept contrast_limits — add
                    # without it (real napari always does; this only spares
                    # minimal test fakes, mirroring `_readd_native_layer`).
                    add_kwargs.pop("contrast_limits", None)
                    layer = self.viewer.add_image(ch_data, **add_kwargs)
                    cl = None
                if cl is not None:
                    self._widen_contrast_range_to_dtype(layer, cl)
            else:
                layer = self.viewer.add_image(
                    ch_data, name=ch_name, colormap=cmap, blending="additive")
            # Physical-µm scale + units AFTER add (guarded) — keeps contrast and
            # the minimal-fake add signatures untouched (see `_apply_layer_scale`).
            _apply_layer_scale(layer, scale)
            _apply_layer_units(layer, units)
            layer_names.append(layer.name)
            added_layers.append(layer)
            # [PROG] per-channel display trace: layer kind + the contrast limits
            # napari computed from the shown slice — so an over-saturation can be
            # diagnosed by comparing these across paths/modes.
            try:
                self._log(
                    "PROG",
                    f"channel {ch_name!r}: {type(layer).__name__} "
                    f"multiscale={isinstance(ch_data, (list, tuple))} "
                    f"contrast={tuple(getattr(layer, 'contrast_limits', ()))} "
                    f"cmap={cmap} blending={getattr(layer, 'blending', '?')}")
            except Exception:
                pass
        # Source layers are now on the shared canvas (all add paths route here:
        # normal load, multiscale display, step-re-enter re-show).
        self._source_layers_on_canvas = True
        # Seed the object->name map so a later rename can resync the name caches
        # (covers BOTH the initial load and the step-re-enter re-show, since
        # `_reshow_source_layers` routes through here).
        self._track_source_layer_objects(added_layers)
        return layer_names

    def _clear_contrast_cache(self, reason=""):
        """Drop the memoized multiscale contrast limits (Brick 3a #8).

        Called wherever the (store, channel) -> limits mapping could go stale: a
        source unload, a preview / processed-view teardown, and a workspace
        change.  Safe to call at any time - the cache is a pure accelerator and a
        miss just recomputes.  Never raises."""
        cache = getattr(self, "_contrast_cache", None)
        if cache:
            cache.clear()
            if reason:
                self._log("PROG", f"contrast cache cleared ({reason})")

    def _estimate_multiscale_contrast(self, levels, store_key=None,
                                      channel_index=None):
        """Robust contrast limits ``[lo, hi]`` for a multiscale channel, computed
        from a FINE pyramid level via percentiles — NOT napari's default (the
        min/max of the COARSEST level, `Image._calc_data_range` -> `data[-1]`),
        which under-estimates the max for sparse-bright channels (nuclei whose
        peaks are diluted by mean-downsampling) and over-saturates.

        Reads the FINEST level whose full array is within a voxel budget (a
        sub-second zarr read); percentiles [1, 99.9] resist hot-pixel outliers.
        Returns ``None`` (napari's default) for uint8 (napari returns 0-255) or
        on any failure.  Best-effort / never raises.

        MEMOIZED per ``(store_key, channel_index)`` (Brick 3a #8 / assessment
        #14).  The limits are a pure function of the store's pixels, but the read
        is a real zarr read of up to 16 Mvox PER CHANNEL and it runs again on
        every step re-enter (`_reshow_source_layers` re-adds the layers from the
        retained channel refs) - exactly the re-enter cost the assessment
        flagged.  The key MUST be passed explicitly by the caller: deriving it
        from `self._current_multiscale_store` would be WRONG on the initial-load
        path, where that attribute still names the PREVIOUS store, and a hit
        would apply the wrong window.  ``store_key is None`` -> no caching,
        compute fresh.  Invalidated by :meth:`_clear_contrast_cache`."""
        cache_key = None
        if store_key is not None and channel_index is not None:
            cache_key = (str(store_key), int(channel_index))
            cache = getattr(self, "_contrast_cache", None)
            if cache is None:
                cache = self._contrast_cache = {}
            if cache_key in cache:
                hit = cache[cache_key]
                self._log("PROG", f"contrast cache hit {cache_key} -> {hit}")
                return list(hit) if hit is not None else None
        try:
            import numpy as np
            if not levels:
                return None
            BUDGET = 16_000_000            # voxels (~32 MB uint16)
            chosen = levels[-1]            # coarsest fallback
            for lv in levels:              # finest -> coarsest
                shape = getattr(lv, "shape", None)
                if not shape:
                    continue
                n = 1
                for s in shape:
                    n *= int(s)
                if n <= BUDGET:
                    chosen = lv
                    break
            arr = np.asarray(chosen)       # realize (bounded by BUDGET)
            if arr.dtype == np.uint8:
                return None                # napari already returns 0-255
            flat = arr.reshape(-1)
            if flat.dtype.kind == "f":
                flat = flat[np.isfinite(flat)]
            if flat.size == 0:
                return None
            lo = float(np.percentile(flat, 1.0))
            hi = float(np.percentile(flat, 99.9))
            if not (hi > lo):
                return None
            self._log("PROG", f"contrast from level shape "
                      f"{tuple(getattr(chosen, 'shape', ()))} -> [{lo:.1f}, "
                      f"{hi:.1f}] (napari default would use the coarsest level)")
            if cache_key is not None:
                self._contrast_cache[cache_key] = [lo, hi]
            return [lo, hi]
        except Exception as exc:
            self._log("PROG", f"contrast estimate failed -> napari default: {exc}")
            return None

    def _widen_contrast_range_to_dtype(self, layer, cl):
        """Widen a multiscale layer's contrast-limits slider RANGE to the full
        dtype extent while keeping the display limits at *cl* — `add_image` pins
        the range to the passed contrast_limits, which would otherwise cap the
        slider at our auto window.  Best-effort / never raises."""
        try:
            import numpy as np
            dt = np.dtype(getattr(layer, "dtype", None))
            if np.issubdtype(dt, np.integer):
                info = np.iinfo(dt)
                rng = [float(info.min), float(info.max)]
            else:
                lo, hi = float(cl[0]), float(cl[1])
                span = (hi - lo) or 1.0
                rng = [lo - span, hi + span]
            layer.contrast_limits_range = rng
            layer.contrast_limits = list(cl)   # range setter can clamp — re-assert
        except Exception as exc:
            self._log("PROG", f"widen contrast range skipped: {exc}")

    def progressive_wanted(self) -> bool:
        """True iff the Progressive-3D checkbox exists AND is checked AND the
        vendored progressive engine is available on this install.

        The availability check is what makes a checked-but-unsupported checkbox
        fail CLOSED to the plain multiscale path.  Best-effort / never raises."""
        w = getattr(self, "widget", None)
        cb = getattr(w, "progressive_3d_checkbox", None)
        if cb is None:
            return False
        try:
            if not cb.isChecked():
                return False
        except Exception:
            return False
        try:
            from packages.epicycle_largedata import progressive
            return bool(progressive.progressive_available())
        except Exception:
            return False

    def _remove_source_layers_from_viewer(self):
        """Remove Step 1's source Image layers from the shared canvas (hide-then-
        remove), WITHOUT dropping the channel refs.

        Called on step-exit so the source layers do not strand on the shared
        viewer — Step 3's annotator grabs ALL resident Image layers, so a
        left-behind Step 1 source aborts its subviewer load (the pinned
        all-Image-layers bug).  The refs are kept so :meth:`_reshow_source_layers`
        restores the exact layers when the user returns to Step 1 (a full drop is
        :meth:`unload_source`).
        """
        # Snapshot the ticked channel names BEFORE the removal loop (visual-focus
        # restore only; inclusion itself lives in the widget's `_channel_included`
        # dict, preserved by the guard below).  No-op with no widget/combo.
        combo = getattr(getattr(self, "widget", None), "image_layer_combo", None)
        if combo is not None:
            try:
                self._saved_channel_selection = [
                    i.text() for i in combo.selectedItems()]
            except Exception:
                self._saved_channel_selection = None
        viewer = getattr(self, "viewer", None)
        if viewer is None:
            return
        # Remove UNDER the `_swapping_source_display` guard so the per-removal
        # `_on_layers_changed` does NOT run `update_image_layer_list` with a
        # shrinking/empty name set — which would reconcile the ③ Channels'
        # `_channel_included` dict down to {} and, on step re-enter, silently
        # revert a deliberate channel UNTICK back to included (audit H1 — the
        # Extract-a-channel-you-excluded class).  The raw⇄processed swaps already
        # remove under this same guard, which is why inclusion survives them.
        # Nested-safe: save + restore the prior value so a caller that set the
        # guard (the processed swap) keeps it.
        _prev_swap_guard = getattr(self, "_swapping_source_display", False)
        self._swapping_source_display = True
        _perf = os.environ.get("EPICYCLE_SYNC_DEBUG") == "1"
        _t0 = time.perf_counter() if _perf else 0.0
        _n = len(list(self._loaded_source_layers))
        try:
            # DO NOT BATCH THIS LOOP.  `viewer.layers.batched_update()` looks like
            # the obvious fix (one canvas rebuild instead of N) and was tried in
            # a49e6ba7; MEASURED LIVE on the real viewer it is a REGRESSION -
            # napari's `end_batch` runs the full gc.collect + glFinish +
            # scenegraph rebuild at the end of EVERY batch, INCLUDING an empty
            # one, on top of the per-layer path.  Unload of 2 layers 511 -> 734 ms,
            # unload of ZERO layers 1 -> 576 ms, Step 1->3 switch 650 -> ~1100 ms.
            # Reverted in Brick 3e.  Plain loop, hide-before-remove.
            for name in list(self._loaded_source_layers):
                if name in viewer.layers:
                    try:
                        layer = viewer.layers[name]
                        layer.visible = False   # hide BEFORE remove
                        viewer.layers.remove(layer)
                    except Exception:
                        pass
        finally:
            self._swapping_source_display = _prev_swap_guard
            if _perf:
                self._log(
                    "STEP1-PERF",
                    f"step-exit source removal "
                    f"{(time.perf_counter() - _t0) * 1000.0:.1f} ms "
                    f"({_n} source layer(s))")
        # Layers no longer on the canvas -> the 3D-detail pin must not apply
        # (the ndisplay hook now no-ops for 2D/3D toggles in other steps).  The
        # chosen level is preserved so step re-enter restores it.
        self._source_layers_on_canvas = False

    def _reshow_source_layers(self):
        """Re-add the source Image layers from the retained channel refs.

        Brick 5: this is NO LONGER the step-re-enter path — `on_step_enter`
        deliberately leaves the canvas empty (`_note_source_unloaded_on_reentry`)
        and the user presses Load.  The remaining production caller is the
        raw⇄processed view toggle, which re-adds raw fresh after the clean swap.

        Idempotent: a no-op when the source is unloaded (no channel refs) or the
        layers are already present.  Extraction stacks the LIVE display channels,
        so they must exist for it to work — which is exactly why `extract_patches`
        checks `_source_is_off_canvas` instead of assuming a re-show happened.
        """
        viewer = getattr(self, "viewer", None)
        channels = getattr(self, "_source_display_channels", None)
        if viewer is None or not channels:
            return
        # Already shown (never left, or re-shown) -> nothing to do.
        if any(n in viewer.layers for n in self._loaded_source_layers):
            return
        try:
            self._loaded_source_layers = self._add_source_display_layers(
                channels, fit_view=False,   # re-enter: keep the user's view
                # Step re-enter: the store IS the current one here (unlike the
                # initial-load path), so this key is correct and turns the
                # per-channel 16-Mvox contrast read into a dict lookup.
                contrast_store_key=getattr(
                    self, "_current_multiscale_store", None))
            self._on_layers_changed()
            # Re-engage progressive + the 3D-detail slider to match the CURRENT
            # ndisplay, mirroring the initial-load path (see the trio at
            # `_apply_multiscale_store` end).  Without this, returning to Step 1
            # while ALREADY in 3D (no ndisplay-change event to fire the hook)
            # comes back as native-coarse even with progressive wanted, and the
            # slider show/hide state is stale (lifecycle-audit MED-1).  Both
            # no-op for a normal (ndarray) source; the swap defers its build.
            self._maybe_swap_progressive_on_ndisplay()
            self._refresh_3d_detail_for_ndisplay()
            # Re-apply the 3D-detail pin the user set before leaving Step 1: the
            # freshly re-added multiscale layers default to locked_data_level=None
            # (coarsest in 3D), so without this a pinned level silently reverts to
            # coarse on return until the next 2D/3D toggle.  No-op for a normal
            # source (`_locked_3d_level is None`) or in 2D.
            self._apply_3d_detail_level()
            # Re-apply the channel tick-selection captured on step-exit (see
            # `_remove_source_layers_from_viewer`): the removal cleared it from
            # the combo, so without this the user would have to re-tick channels
            # before Extract works again.  Self-clearing; a no-op (leaving the
            # normal in-Step-1 flow untouched) when nothing was captured.
            saved = self._saved_channel_selection
            self._saved_channel_selection = None
            combo = getattr(getattr(self, "widget", None),
                            "image_layer_combo", None)
            if saved and combo is not None:
                for i in range(combo.count()):
                    item = combo.item(i)
                    if item.text() in saved:
                        item.setSelected(True)
        except Exception as exc:
            self._log("STEP1-BOX", f"re-show source layers failed: {exc}")

    # ------------------------------------------------------------------
    # L0 axis detection helpers
    # ------------------------------------------------------------------

    def _meta_axes_for_path(self, path):
        """The tifffile ``series.axes`` string for a scanned source path, or
        ``None`` — read from the header scan so detection needs no extra I/O."""
        for info in getattr(self, "_scanned_files", None) or []:
            if info.get("path") == path:
                return info.get("axes")
        return None

    def note_axis_order_override(self):
        """Record that the user deliberately picked an Order for the source the
        combo DESCRIBES (bound to the Order combo's ``activated`` signal, which
        fires only on user interaction — not on the programmatic detection
        setter).  A re-selection / re-load of the same file then keeps their
        choice instead of clobbering it with re-detection; the combo stays the
        user-confirmable editor.

        The override is keyed on ``_axis_combo_source`` — the file the combo is
        currently pointed at (detect-on-selection and the load path both set it).
        It is NOT keyed on ``_loaded_source_file``: since detect-on-selection the
        combo tracks the SELECTED source, which may differ from the loaded one,
        so keying on the loaded file would record the SELECTED file's chosen Order
        under the LOADED file and silently mis-orient the loaded file's store
        (3-audit finding 2026-08-02).  Falls back to the selected row, then the
        loaded file, only when the combo describes nothing yet."""
        path = getattr(self, "_axis_combo_source", None)
        if not path:
            item = self.widget.source_file_list.currentItem()
            path = item.data(Qt.UserRole) if item is not None else None
        if not path:
            path = getattr(self, "_loaded_source_file", None)
        if path:
            order = self.widget.axis_order_combo.currentText().strip() or "CZYX"
            self._axis_order_overrides[_norm_path(path)] = order
            # Persist the user's on-disk order as the durable `source_axes`
            #  — last-wins against this source's registry row.
            self._queue_source_save(path=path, source_axes=order)

    def _restore_bounding_boxes(self, display_name):
        """Load bounding boxes from exported metadata JSONs in the workspace.

        Scans for ``{display_name}_patch_*_metadata.json`` and recreates
        the shapes + text labels on the viewer.

        Returns the number of boxes restored.
        """
        workspace = self.widget.output_file_path.text().strip()
        if not workspace or not os.path.isdir(workspace):
            return 0

        # Sidecars live in artifacts/step1/patches/ (new layout); fall back to
        # the workspace root for un-migrated (legacy) workspaces.
        scan_dir = step1_sidecar_scan_dir(workspace)

        # Match the prefix the loaded file's patches were exported under
        # (persisted, D-3) so restore still finds them after a display-name edit;
        # fall back to '{display_name}_patch'.
        prefix = self._prefix_for_file(
            getattr(self, "_loaded_source_file", None), display_name,
            load_state(workspace), default=f"{display_name}_patch",
        )
        import re as _re
        pattern = _re.compile(
            rf"^{_re.escape(prefix)}_(\d+)_metadata\.json$",
            _re.IGNORECASE,
        )

        boxes = []
        z_ranges = []
        for fname in sorted(os.listdir(scan_dir)):
            m = pattern.match(fname)
            if not m:
                continue
            try:
                with open(os.path.join(scan_dir, fname)) as f:
                    meta = json.load(f)
                bbox = meta.get("bounding_box")
                if bbox and isinstance(bbox, list) and len(bbox) == 4:
                    boxes.append(np.array(bbox, dtype=float))
                    # The on-disk Z identity of the patch (frozen sidecar key);
                    # None for legacy sidecars predating the field.
                    zr = meta.get("selected_z_range")
                    z_ranges.append(
                        list(zr) if isinstance(zr, (list, tuple))
                        and len(zr) == 2 else None)
            except Exception:
                continue

        if not boxes:
            return 0

        # Restore into the CRASH-SAFE ③ box tool (F3), NOT the legacy napari
        # Shapes layer.  A resident Shapes-in-3D layer's mesh VBO can be freed
        # while a draw is still queued during a Raw⇄Processed view swap, which
        # faults the 3D canvas with a native `glDrawArrays` access violation
        # (the box-tool rework replaced the Shapes layer with a vispy overlay
        # for exactly this reason).  So the restore path NO LONGER creates or
        # populates a Shapes layer — the boxes become COMMITTED box-tool rows
        # (locked = the exported visual + frozen from edits), rendered by the
        # vispy overlay once box mode is opened.  ③'s Box List is populated by
        # the box tool's own `refresh_list` (via `load_committed_boxes`), so it
        # still shows the restored boxes.
        box_tool = getattr(self, "box_tool", None)
        if box_tool is not None:
            full_shape = getattr(box_tool, "_full_shape", None) \
                or box_tool._resolve_full_shape()
            if full_shape is not None:
                boxrecs = [
                    self._boxrec_from_sidecar(b, zr, full_shape)
                    for b, zr in zip(boxes, z_ranges)
                ]
                try:
                    box_tool.load_committed_boxes(boxrecs)
                except Exception as exc:
                    self._log("STEP1-BOX", f"box-tool restore failed: {exc}")
            else:
                # No source shape resolvable yet (no lazy handle / display
                # layers) — skip the box-tool seed gracefully; the committed
                # signatures below still guard re-extraction.  (Not reached in
                # the normal load flow: `_source_lazy` is set before restore.)
                self._log("STEP1-BOX",
                          "box-tool restore deferred — source shape "
                          "unresolved")

        # Restored boxes describe already-exported patches -> commit their
        # signatures so they aren't re-extracted (the escalating-index bug).
        # Commit BOTH the XY-only sig (the legacy Shapes-fallback re-extract
        # check has no per-box Z, so it compares XY-only) AND the Z-inclusive
        # sig from the sidecar's `selected_z_range` (a box-tool re-draw of the
        # same patch compares Z-inclusively).  A 4-tuple never collides with a
        # 6-tuple, so the XY-only sig can't drop a genuinely different Z slab.
        for b, zr in zip(boxes, z_ranges):
            self._committed_box_sigs.add(box_signature(b))
            if zr is not None:
                self._committed_box_sigs.add(box_signature(b, zr))

        return len(boxes)

    @staticmethod
    def _boxrec_from_sidecar(bbox, z_range, full_shape):
        """A box-tool ``BoxRec`` from a sidecar's exported geometry (F3).

        *bbox* is the ``(4, 2)`` ``Y/X`` rectangle ``get_drawn_shapes`` wrote
        (``bounding_box``); *z_range* is the frozen ``[z0, z1]``
        (``selected_z_range``) or ``None`` for a legacy sidecar.

        Inverse of ``rect_for_box`` / ``box_signature``: the XY center+size come
        from the rect's floored min/max corners — ``cy = (y0 + y1) // 2``,
        ``sy = y1 - y0`` (same on X) — the exact integer inverse of
        ``BoxRec.bounds_dict`` (``c - s // 2 == a0`` and ``a0 + s == a1`` with
        ``c = a0 + s // 2``), so the restored box reconstructs the SAME clamped
        bounds it was exported with and its ``box_signature`` matches the
        committed sig.  Z is the TRIMMED export Z from *z_range*
        (``cz = (z0 + z1) // 2``, ``sz = z1 - z0``) — accepted as the restored
        box's Z; a legacy sidecar with no ``selected_z_range`` falls
        back to ``place_box``'s full-Z default (byte-identical to an untouched
        box).
        """
        a = np.asarray(bbox, dtype=float)
        y0 = int(np.floor(a[:, 0].min()))
        y1 = int(np.floor(a[:, 0].max()))
        x0 = int(np.floor(a[:, 1].min()))
        x1 = int(np.floor(a[:, 1].max()))
        cy = (y0 + y1) // 2
        cx = (x0 + x1) // 2
        sy = max(1, y1 - y0)
        sx = max(1, x1 - x0)
        cz = sz = None
        if z_range is not None and len(z_range) == 2:
            z0, z1 = int(z_range[0]), int(z_range[1])
            cz = (z0 + z1) // 2
            sz = max(1, z1 - z0)
        return place_box(full_shape, cy, cx, sy, sx, cz=cz, sz=sz)

    def unload_source(self):
        """Remove all source layers created by the last load.

        Wraps :meth:`_unload_source_inner` in ②`s shared busy bar and the
        `[STEP1-PERF] unload` probe (`EPICYCLE_SYNC_DEBUG=1`).  Brick 3a #7 added
        the probe rather than guessing; the live measurement came back at 511 ms
        for two lazy multiscale layers - five times the ~100 ms main-thread budget
 - because napari 0.8 does a `gc.collect()` + `glFinish()` +
        scenegraph rebuild per `LayerList` event.

        Brick 3c #7 tried to fix that by BATCHING the removal
        (`viewer.layers.batched_update()`); Brick 3e reverted it after measuring
        on the real viewer - `end_batch` runs the same full rebuild at the end of
        every batch, INCLUDING an empty one, so it only ADDED cost (see the note
        at the removal loop in :meth:`_unload_source_inner`).  What remains is the
        honest half: NAME the operation on ②`s bar while it runs, and only when
        there is something to remove - a zero-layer unload is ~1 ms and must not
        pay for a bar it does not need.
        """
        _perf = os.environ.get("EPICYCLE_SYNC_DEBUG") == "1"
        _t0 = time.perf_counter() if _perf else 0.0
        _n_layers = len(getattr(self, "_loaded_source_layers", ()) or ())
        # MEASURED live at 511 ms for two lazy multiscale layers (the
        # `[STEP1-PERF] unload` probe added in Brick 3a #7), which is well over
        # the ~100 ms main-thread budget - so name the operation on
        # ②`s shared bar for the duration.  Only when there is actually something
        # to remove: an unload with no layers is sub-millisecond and a bar that
        # blinks for nothing is noise.  Hidden in a `finally` so a raising teardown
        # can never leave it spinning.
        _busy = bool(_n_layers)
        if _busy:
            try:
                self.widget._busy_show(
                    f"Unloading {self._loaded_source_file_label()}…")
            except Exception:
                _busy = False
        try:
            self._unload_source_inner()
        finally:
            if _busy:
                try:
                    self.widget._busy_hide()
                except Exception:
                    pass
        if _perf:
            self._log(
                "STEP1-PERF",
                f"unload {(time.perf_counter() - _t0) * 1000.0:.1f} ms "
                f"({_n_layers} source layer(s))")

    def _loaded_source_file_label(self):
        """A short name for the source being unloaded (basename, or a generic
        fallback) - only used for the ② bar label.  Never raises."""
        import os as _os

        try:
            name = getattr(self, "_current_multiscale_display_name", None)
            if name:
                return str(name)
            path = getattr(self, "_loaded_source_file", None)
            if path:
                return _os.path.basename(str(path))
        except Exception:
            pass
        return "source"

    def _unload_source_inner(self):
        """The actual teardown (see :meth:`unload_source`, which wraps this in ②`s
        busy bar + the `[STEP1-PERF]` probe)."""
        # The ③ box tool renders on the (now-departing) source: fully tear it
        # down (visuals + boxes) so no Shapes-in-3D layer or stale BoxRec
        # survives the source it was drawn on.
        box_tool = getattr(self, "box_tool", None)
        if box_tool is not None:
            try:
                box_tool.reset()
            except Exception as exc:
                self._log("STEP1-BOX", f"box-tool reset on unload: {exc}")
        # ④ patch-location boxes describe THIS source's patches — drop them (and
        # the overlay) so they never draw over the next source (Brick 2).
        try:
            self._clear_patch_locations()
        except Exception as exc:
            self._log("PATCHLOC", f"clear on unload: {exc}")
        # Drop any ⟨preview⟩ processed-crop render + the transient REGION box + its
        # overlay along with the source they were taken from, and collapse the ②
        # controls (single teardown entry point; hide-then-remove).
        self._teardown_preview()
        # DO NOT BATCH THIS LOOP.  `viewer.layers.batched_update()` looks like
        # the obvious fix (one canvas rebuild instead of N) and was tried in
        # a49e6ba7; MEASURED LIVE on the real viewer it is a REGRESSION -
        # napari's `end_batch` runs the full gc.collect + glFinish +
        # scenegraph rebuild at the end of EVERY batch, INCLUDING an empty
        # one, on top of the per-layer path.  Unload of 2 layers 511 -> 734 ms,
        # unload of ZERO layers 1 -> 576 ms, Step 1->3 switch 650 -> ~1100 ms.
        # Reverted in Brick 3e.  Plain loop, hide-before-remove.
        for name in list(self._loaded_source_layers):
            if name in self.viewer.layers:
                try:
                    layer = self.viewer.layers[name]
                    layer.visible = False       # hide BEFORE remove
                    self.viewer.layers.remove(layer)
                except Exception:
                    pass
        self._loaded_source_layers.clear()
        self._loaded_source_file = None
        self._loaded_source_id = None
        # Brick 4c item 5: the raw-missing state belongs to the source that was
        # loaded, not to the session.  Leaving it set kept Prepare and "Process
        # whole volume" greyed -- with a tooltip about finding an original file
        # -- for whatever the user loaded NEXT.
        self._set_raw_missing(False)
        self._source_freshness_note = None
        # Drop the retained display-channel refs (a new load rebuilds them) so a
        # step re-enter does not re-show an unloaded source.
        self._source_display_channels = None
        # Drop the lazy full-res handle + its axis/channel maps (a new load
        # rebuilds them); keeps a stale handle from feeding the next extraction.
        self._source_lazy = None
        self._source_axes_resolved = None
        self._source_disk_axes = None
        # The combo no longer describes any file; a fresh selection repoints it.
        self._axis_combo_source = None
        # No multiscale source displayed -> hide the 3D-detail control + drop its
        # per-store state (the ndisplay hook stays connected but no-ops: the pin
        # is cleared and the layers are off-canvas).
        self._locked_3d_level = None
        self._pyramid_level_shapes = []
        self._source_layers_on_canvas = False
        # No store displayed -> the live Progressive-3D toggle has nothing to
        # re-display (it no-ops until the next load).
        self._progressive_active = False
        # Reset the "Zoom depth" choice so it does not carry from one store into
        # the next (hygiene; mirrors `_locked_3d_level` above).
        self._zoom_depth_wanted = None
        self._current_multiscale_store = None
        self._current_multiscale_display_name = None
        self._current_multiscale_src = None
        self._clear_contrast_cache("unload")
        hide_td = getattr(self.widget, "hide_3d_detail", None)
        if hide_td is not None:
            hide_td()
        # `_refresh_zoom_depth_for_ndisplay` can no longer hide the combo once the
        # store is cleared (it early-returns), so hide it here directly.
        hide_zd = getattr(self.widget, "hide_zoom_depth", None)
        if hide_zd is not None:
            hide_zd()
        self._channel_index_by_name = {}
        self._on_layers_changed()
        self.widget.source_status_label.setText("Source data unloaded")
        self.widget.source_status_label.setStyleSheet(
            "font-size: 10px; color: #5ba4cf;"
        )

    def _prefix_for_file(self, file_path, stem, state, default=None):
        """The numbering prefix a file's patches were exported under (persisted
        on its source entry, D-3), or *default* / '{stem}_patch' if the file has
        no exported patches yet.  Used by bounding-box restore to find the file's
        sidecars regardless of a display-name edit.  (Shares ``_state._norm_path``
        so path matching can't silently diverge — audit.)"""
        if state and file_path:
            target = _norm_path(file_path)
            for src in state.get("sources", []):
                p = src.get("path", "")
                if p and _norm_path(p) == target and src.get("prefix"):
                    return src["prefix"]
        return default if default is not None else f"{stem}_patch"

    def _patch_count_for_file(self, file_path, stem, workspace, state):
        """Truly per-FILE exported-patch count for the badge: counts by the
        file's ``source_id`` (so two same-display-name files don't cross-count,
        and a rename between sessions doesn't undercount — audit F1).  Falls back
        to the workspace ``{stem}_patch`` prefix scan for legacy / un-registered
        files (no source_id)."""
        if state and file_path:
            target = _norm_path(file_path)
            for src in state.get("sources", []):
                p = src.get("path", "")
                if p and _norm_path(p) == target:
                    sid = src.get("source_id", "")
                    if sid:
                        return count_patches_for_source(state, sid)
                    break
        if workspace and os.path.isdir(workspace) and stem:
            return count_patches_for_prefix(workspace, f"{stem}_patch", state=state)
        return 0

    def _refresh_source_file_list(self):
        """Rebuild text + color for every item in the source file list.

        Shows filename (truncated), shape, and exported patch count badge.
        """
        from qtpy.QtGui import QBrush, QColor

        workspace = self.widget.output_file_path.text().strip()
        file_list = self.widget.source_file_list
        max_name = 35  # truncate long filenames

        # Load the registry ONCE for the badge counts (avoids re-parsing
        # step1_state.json per source file on every textChanged tick).
        state_cache = (
            load_state(workspace)
            if workspace and os.path.isdir(workspace) else None
        )

        for i in range(file_list.count()):
            item = file_list.item(i)
            stem = item.data(Qt.UserRole + 1) or ""
            info = next((f for f in self._scanned_files if f["stem"] == stem), None)

            # Format shape compactly.
            if info and info["shape"]:
                sh = info["shape"]
                shape_str = "\u00d7".join(str(d) for d in sh)  # e.g. 145×3×1024×1024
            else:
                shape_str = "?"

            # Truncate long names with ellipsis.
            display = stem if len(stem) <= max_name else stem[:max_name - 1] + "\u2026"

            # Truly per-file exported-patch count (by source_id) so two
            # same-display-name files don't cross-count and a rename doesn't
            # undercount (audit F1); falls back to '{stem}_patch' for legacy files.
            count = 0
            if workspace and os.path.isdir(workspace) and stem:
                fpath = item.data(Qt.UserRole) or ""
                count = self._patch_count_for_file(fpath, stem, workspace, state_cache)

            # Build formatted text.
            if count > 0:
                text = f" {display}\n   {shape_str}   \u2022 {count} patch(es)"
                item.setForeground(QBrush(QColor("#5ba4cf")))
            else:
                text = f" {display}\n   {shape_str}"
                item.setForeground(QBrush(QColor("#cccccc")))
            item.setText(text)

        # Resize rows to fit two-line items — via the ITEMS, not by appending to
        # the stylesheet.  This method is bound to `output_file_path`'s
        # `textChanged`, so the old version appended a fresh
        # `QListWidget::item { min-height }` rule on every KEYSTROKE (measured:
        # 400 → 640 chars over six calls, six stacked duplicate rules) and made
        # Qt re-parse a monotonically growing sheet each time.  Setting the size
        # hint is idempotent by construction.
        from qtpy.QtCore import QSize
        for _i in range(file_list.count()):
            _it = file_list.item(_i)
            _hint = _it.sizeHint()
            if _hint.height() < 36:
                _it.setSizeHint(QSize(max(0, _hint.width()), 36))

    # Backward-compatible alias.
    _refresh_source_patch_counts = _refresh_source_file_list

    # ------------------------------------------------------------------
    # Shapes layer helpers
    # ------------------------------------------------------------------

    def get_selected_shapes_layer(self):
        """Returns the currently selected Shapes layer, creating one if needed.

        Always applies teal styling so shapes drawn after this call
        use the Step 1 accent color.
        """
        name = self.widget.bbox_layer_combo.currentText()
        if name in self.viewer.layers:
            layer = self.viewer.layers[name]
            if isinstance(layer, Shapes):
                self._apply_shapes_style(layer)
                return layer
        return self._ensure_shapes_layer()

    def _current_shapes_layer(self):
        """Return the currently-selected Shapes layer if one already exists, else
        ``None`` — WITHOUT creating one.  Extract's ``get_drawn_shapes`` fallback
        uses this so an extract can never leak an empty "ROI Shapes" layer (B1);
        layer CREATION stays confined to the explicit draw path
        (``get_selected_shapes_layer`` -> ``_ensure_shapes_layer``)."""
        name = self.widget.bbox_layer_combo.currentText()
        if name in self.viewer.layers:
            layer = self.viewer.layers[name]
            if isinstance(layer, Shapes):
                return layer
        for layer in self.viewer.layers:
            if isinstance(layer, Shapes):
                return layer
        return None

    @staticmethod
    def _apply_shapes_style(layer):
        """Apply teal Step 1 accent to a Shapes layer's drawing properties."""
        layer.current_edge_color = "#5ba4cf"
        layer.current_face_color = [0.357, 0.643, 0.812, 0.15]
        layer.current_edge_width = 2

    def _ensure_shapes_layer(self):
        """Auto-create a styled Shapes layer for ROI drawing."""
        # Reuse existing if one exists.
        for layer in self.viewer.layers:
            if isinstance(layer, Shapes):
                self._apply_shapes_style(layer)
                return layer
        layer = self.viewer.add_shapes(
            name="ROI Shapes",
            edge_color="#5ba4cf",
            face_color=[0.357, 0.643, 0.812, 0.15],
            edge_width=2,
        )
        self._apply_shapes_style(layer)
        self._on_layers_changed()
        idx = self.widget.bbox_layer_combo.findText("ROI Shapes")
        if idx >= 0:
            self.widget.bbox_layer_combo.setCurrentIndex(idx)
        return layer

    def clear_shapes(self):
        """Remove all shapes from the active shapes layer (user action).

        When the ③ box tool owns boxes, "Clear" must drop THOSE (they are the
        extract source of truth) — otherwise Clear looks like a no-op yet still
        wipes `_committed_box_sigs`, re-exposing already-captured boxes to
        re-extraction.  `clear_all` stays in box mode + clears via `redraw`.
        """
        box_tool = getattr(self, "box_tool", None)
        if box_tool is not None and box_tool.has_boxes():
            box_tool.clear_all()
        self._clear_shapes_layer()
        self.widget._set_draw_mode("Idle")

    def _clear_shapes_layer(self):
        """Clear all shapes + text from any existing Shapes layer.  Also resets
        the committed-box signatures — the on-layer boxes they described are gone;
        on a source load, `_restore_bounding_boxes` repopulates them right after.
        (Single chokepoint for both user-Clear and the load/reload path.)"""
        self._committed_box_sigs.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, Shapes):
                if len(layer.data) > 0:
                    layer.data = []
                layer.properties = {}
                layer.text = None
                break
    
    # gets the drawn shapes from the shapes layer
    def get_drawn_shapes(self):
        """
        Returns all drawn bounding boxes as a list of rectangles.
        Each rectangle is a 4x2 numpy array (Y, X) in clockwise order.

        Phase-1 box tool: once any ③ box exists, its BoxRecs are the source of
        truth — return their XY bounds as ``(4, 2)`` YX rectangles (full-res
        voxel coords, full-Z supplied by extraction).  This is the single
        chokepoint the unchanged extract path already reads, so nothing
        downstream (filter / signatures / crop) changes.  Falls back to the
        legacy Shapes layer when no box-tool box exists.
        """
        box_tool = getattr(self, "box_tool", None)
        if box_tool is not None and box_tool.has_boxes():
            return box_tool.rectangles()

        layer = self._current_shapes_layer()
        if layer is None:
            return []

        return [np.array(shape) for shape in layer.data]


    # gets the data -- image data from the selected image layers
    def get_combined_image_stack(self):
        """
            Grabs the user-selected image layer and then returns it as a stacked array (Channel, Z-slice, Y and X)
            should return combine image stack from selected channel;
        """
        # ③'s INCLUDED channels (ticked include checkboxes) IN ROW ORDER — the
        # ONE inclusion+order source of truth (`widget.checked_channel_names()`),
        # replacing the old row-SELECTION read so this eager fallback stacks the
        # exact channels + order the lazy crop, `_selected_channel_names`, and
        # the per-patch sidecar use.
        selected_names = self.widget.checked_channel_names()

        # get the stacks data from the layers and assembled it in order
        stacks = []
        for name in selected_names:
            if name in self.viewer.layers:
                layer_data = self.viewer.layers[name].data
                # Only a real `np.ndarray` channel is eagerly stackable here.  A
                # MULTISCALE layer's `.data` is a napari `MultiScaleData` — a
                # `Sequence`, NOT a list/tuple and NOT an ndarray — so an
                # `isinstance(..., (list, tuple))` guard would MISS it and let it
                # fall through to `np.stack`, which raises (or silently grabs the
                # coarsest level).  Those large sources are handled by the LAZY
                # extract path (`_crop_source_for` off `_source_lazy`), so this
                # eager FALLBACK should never see one; skip anything that is not a
                # plain ndarray (multiscale / sequence / list) rather than crash
                # (L0 multiscale safety).
                if not isinstance(layer_data, np.ndarray):
                    self._log("STEP1-EXTRACT",
                              f"Skipping non-ndarray layer {name}: extract via "
                              "the lazy path (multiscale / sequence data)")
                    continue
                if getattr(layer_data, "ndim", None) == 3:
                    stacks.append(layer_data)
                else:
                    self._log("STEP1-EXTRACT", f"Skipping layer {name}: unsupported shape {getattr(layer_data, 'shape', None)}")

        if not stacks:
            raise ValueError("No valid 3D image layers selected.")
        
        # combine the individual image channel stack to multi channel array
        combined = np.stack(stacks, axis=0)  # (C, Z, Y, X)

        return combined, selected_names

    _OFF_CANVAS_HINT = "Click Load to view the source before extracting."

    def _source_is_off_canvas(self) -> bool:
        """True iff a source is LOADED (its display-channel refs are retained)
        but NOTHING of it is on the shared canvas.

        This is the Brick-5 state: step-exit removes the source layers and
        step-enter no longer puts them back, so the controller can hold a fully
        described source with an empty canvas.  Several Step-1 operations read
        the LIVE layers rather than those refs — `_live_channel_names` (which
        feeds the lazy crop AND `get_combined_image_stack`) and the box tool's
        `_display_layers_present` — and silently degrade instead of saying so.

        The processed view is NOT off-canvas: the clean raw/processed swap
        removes the raw layers on purpose, and extracting from the processed
        store at the same full-res coords is a supported flow (P1).
        """
        if not getattr(self, "_source_display_channels", None):
            return False           # nothing loaded -> the existing messages apply
        if getattr(self, "_source_layers_on_canvas", False):
            return False
        try:
            if self._is_processed_view_showing():
                return False
        except Exception:          # noqa: BLE001 - a probe is never fatal
            pass
        return True

    def _block_if_source_off_canvas(self, what):
        """Decline *what* with the plain ③ line when the source is loaded but
        off the canvas (Brick 5b item 2).  Returns True when it blocked.

        Box mode is the second reader of the LIVE layers: with
        `_display_layers_present()` False, `_BoxTool.start()` calls
        `_build_overview` — a strided read of the WHOLE lazy store, on the main
        thread, with no indicator.  Entering box mode is a one-click action; it
        must not silently become a multi-second freeze because the user came
        back to Step 1 without pressing Load.
        """
        if not self._source_is_off_canvas():
            return False
        self._set_export_status(self._OFF_CANVAS_HINT, success=False)
        self._log("STEP1-BOX", f"{what} declined — source is off-canvas")
        return True

    def _live_channel_names(self, channel_names):
        """The subset of ``channel_names`` that are 3D Image layers CURRENTLY in
        the viewer — the SAME live filter :meth:`get_combined_image_stack`
        applies when it stacks the display channels (``name in viewer.layers``
        and ``data.ndim == 3``), in the given order.

        Keeps the lazy crop's channel set in lock-step with the eager stack: a
        channel ticked but since removed from the viewer is dropped by both, so
        they can never disagree on the extracted channel count.  When there is
        no viewer (bare controller in tests) the names pass through unchanged.
        """
        viewer = getattr(self, "viewer", None)
        if viewer is None:
            return list(channel_names)
        live = []
        for name in channel_names:
            if name in viewer.layers:
                data = viewer.layers[name].data
                # A MULTISCALE display layer's `.data` is a `MultiScaleData`
                # (a Sequence of pyramid levels, not a plain list); its effective
                # ndim is a level's ndim.  Treating it as ndim=None would wrongly
                # DROP the channel here and then crash the eager fallback, so use
                # the per-level ndim — a multiscale ZYX channel counts as live
                # exactly like a plain one (L0 multiscale safety).
                if _effective_ndim(data) == 3:
                    live.append(name)
        return live

    def _processed_store_channel_names(self):
        """The channel names of the currently-displayed ⟨processed⟩ store, IN
        STORE ORDER — the plain (marker-free) names the store's channels were
        written in.  Primary source is ``_processed_redisplay["names"]`` (the
        spec the store was displayed from); falls back to the marker-stripped
        ``_processed_display_channels`` order.  Empty list when neither is set.

        This is the KEY SPACE of the processed channel map — a ticked channel
        name resolves to its position HERE, NOT ``range(n)`` — so the extracted
        channels preserve the ticked row order + include-set."""
        redisplay = getattr(self, "_processed_redisplay", None)
        if isinstance(redisplay, dict):
            names = redisplay.get("names")
            if names:
                return [str(n) for n in names]
        out = []
        for entry in (getattr(self, "_processed_display_channels", None) or []):
            try:
                nm = str(entry[0])
            except (TypeError, IndexError):
                continue
            if nm.startswith(PROCESSED_LAYER_MARKER):
                nm = nm[len(PROCESSED_LAYER_MARKER):]
            out.append(nm)
        return out

    def _crop_source_for_processed(self, channel_names):
        """The LAZY crop source for the ticked channels read from the PROCESSED
        data — the streaming multiscale STORE or the in-memory (eager) processed
        arrays, whichever regime is live (extract-from-processed).

        Returns ``(handle, channel_indices)`` where *handle* is the store's
        level-0 reordered/shaped to ``(C, Z, Y, X)`` (a ``(1, Z, Y, X)`` for a
        single-channel ZYX store) — a lazy dask view, no read — and
        *channel_indices* selects the ticked channels IN ROW ORDER during each
        per-box slice, exactly the same contract the raw branch returns.  The
        map's key space is the store's OWN channel order
        (:meth:`_processed_store_channel_names`), NOT ``range(n)``, so the
        include-set + row order are preserved.

        Raises ``ValueError`` (surfaced by :meth:`extract_patches` as an
        export-status message) when: the store is unavailable, no channels are
        ticked, a ticked channel is not in the processed set (M3), or the
        store's Y/X grid differs from the raw grid.  Never returns
        ``(None, None)`` — there is NO eager fallback for processed pixels (the
        raw layers are off-canvas under the clean swap)."""
        if not channel_names:
            raise ValueError("No channels ticked — tick at least one channel.")
        redisplay = getattr(self, "_processed_redisplay", None) or {}
        kind = redisplay.get("kind")
        store_path = getattr(self, "_current_processed_multiscale_store", None)

        # Build a (C, Z, Y, X) LAZY handle from whichever processed regime is live:
        #   • streaming (large-data)  -> the on-disk multiscale store's level 0;
        #   • in-memory (eager)       -> the per-channel full-res level-0 arrays
        #     already in RAM, wrapped as a lazy dask stack so the per-box crop
        #     still reads only the box (no whole-volume copy).
        if store_path:
            try:
                levels, meta = open_multiscale(store_path)
            except Exception as exc:  # noqa: BLE001
                raise ValueError(
                    f"Could not open the processed store ({exc}) — switch to Raw "
                    "view to extract.") from exc
            level0 = levels[0]
            axes_meta = meta.get("axes") or []
            has_channel = any(
                (a or {}).get("type") == "channel" for a in axes_meta)
            # Single-channel ZYX store -> (1, Z, Y, X) so the crop shape matches.
            czyx = level0 if has_channel else level0[None]
            n_ch = int(level0.shape[0]) if has_channel else 1
        elif kind == "mem" and redisplay.get("channel_levels"):
            # In-memory (eager) Process regime (the common path on a big-RAM box):
            # `channel_levels[i][0]` is channel i's full-res processed volume
            # (Z, Y, X) already in RAM.  Wrap them as a lazy dask (C, Z, Y, X) —
            # `da.from_array` is a view (no copy); the crop + `.compute()` in
            # `extract_patches_from_shapes` reads only the box.
            try:
                import dask.array as da
                arrs = [da.from_array(lv[0])
                        for lv in redisplay["channel_levels"] if lv]
                if not arrs:
                    raise ValueError("the in-memory processed set is empty")
                czyx = da.stack(arrs, axis=0)
                n_ch = len(arrs)
            except ValueError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise ValueError(
                    f"Could not read the in-memory processed data ({exc}) — "
                    "switch to Raw view to extract.") from exc
        else:
            raise ValueError(
                "No processed data to extract from — run “Apply recipe → whole "
                "volume”, or switch to Raw view to extract.")

        # M3 grid guard (future-proof): the processed handle must share the raw
        # Y/X grid so boxes drawn in raw display coords address the same pixels.
        # Skipped when there is no raw lazy handle (eager regime — the processed
        # arrays derive from the same eager source, so the grid matches by
        # construction).  Streaming refuses any recipe that changes the grid
        # today, so this should never trigger — guard it anyway.
        raw_lazy = getattr(self, "_source_lazy", None)
        if raw_lazy is not None:
            raw_yx = tuple(int(s) for s in getattr(raw_lazy, "shape", ()))[-2:]
            proc_yx = tuple(int(s) for s in czyx.shape[-2:])
            if len(raw_yx) == 2 and raw_yx != proc_yx:
                raise ValueError(
                    f"Processed grid {proc_yx} differs from the raw grid "
                    f"{raw_yx} — extract from Raw view instead.")

        # Map each ticked name -> its index in the STORE's channel list.
        store_names = self._processed_store_channel_names()
        name_to_idx = {nm: i for i, nm in enumerate(store_names)}
        indices = []
        for nm in channel_names:
            idx = name_to_idx.get(nm)
            if idx is None or idx < 0 or idx >= n_ch:
                raise ValueError(
                    f"Channel “{nm}” isn't in the processed set — "
                    "re-process it or switch to Raw view to extract.")
            indices.append(idx)
        return czyx, indices

    def _crop_source_for(self, channel_names):
        """The LAZY crop source for the ticked channels, or ``(None, None)``.

        Returns ``(handle, channel_indices)`` where *handle* is the full-res
        source reordered to ``(C, Z, Y, X)`` (a lazy view — no read) and
        *channel_indices* selects the ticked channels IN ROW ORDER during each
        per-box slice, so extraction reads only the ROI (of only those channels)
        off disk (lazy crop).  The pixels are identical to cropping the
        materialized display stack — the handle is the same lazy source the
        display was materialized from, reordered the same way.

        ``(None, None)`` when there is no lazy handle or a ticked name does not
        map to a channel (e.g. a hand-added layer); the caller then crops the
        materialized display stack instead.
        """
        lazy = getattr(self, "_source_lazy", None)
        if lazy is None or not channel_names:
            return None, None
        # Intersect the ticked names with the channels ACTUALLY present in the
        # viewer, using the same live filter get_combined_image_stack applies —
        # otherwise a channel ticked but removed from the viewer would be cropped
        # here yet dropped from the eager stack, giving the lazy patch one extra
        # channel (patch channel-count mismatch).
        channel_names = self._live_channel_names(channel_names)
        if not channel_names:
            return None, None
        idx_map = getattr(self, "_channel_index_by_name", None) or {}
        ndim = getattr(lazy, "ndim", None)

        if ndim == 3:
            # Single-channel ZYX source: present as (1, Z, Y, X) so the crop
            # shape matches the materialized (1, Z, Y, X) display stack exactly.
            if len(channel_names) == 1 and channel_names[0] in idx_map:
                return lazy[None], [0]
            return None, None
        if ndim != 4:
            return None, None

        # Reorder the on-disk handle to CZYX (lazy view, no read) the SAME way
        # the display was transposed.
        if getattr(self, "_source_axes_resolved", None) == "ZCYX":
            if hasattr(lazy, "compute"):
                import dask.array as da
                czyx = da.moveaxis(lazy, 1, 0)
            else:
                czyx = np.moveaxis(lazy, 1, 0)
        else:  # "CZYX" (or unknown -> assume already channel-first)
            czyx = lazy

        try:
            indices = [idx_map[n] for n in channel_names]
        except KeyError:
            return None, None
        if any(i < 0 or i >= czyx.shape[0] for i in indices):
            return None, None
        return czyx, indices

    def _selected_channel_names(self):
        """The INCLUDED channel names from ③'s channel picker, in ROW ORDER —
        the ticked include checkboxes (`widget.checked_channel_names()`), the
        SAME names+order :meth:`get_combined_image_stack` returns for its
        ``channel_names``, but WITHOUT materializing any pixels (no ``np.stack``
        of the whole volume).  Empty only when every channel is unticked; the
        caller then falls back to :meth:`get_combined_image_stack`, whose "No
        valid 3D image layers selected." error is preserved."""
        return self.widget.checked_channel_names()

    def _resolve_extract_source(self, channel_names):
        """Resolve the crop source + box-filter shape for an extract WITHOUT
        reading the whole volume when a lazy handle exists.

        Returns ``(crop_source, channel_indices, filter_shape, image_data)``:

        * **Lazy path** (a usable ``_source_lazy``): *crop_source* is the lazy
          full-res ``(C, Z, Y, X)`` handle, *channel_indices* selects the ticked
          channels during each per-box slice, *filter_shape* is that handle's
          ``.shape`` (read for its Y / X only — no pixels materialized), and
          *image_data* is ``None`` (no whole-volume array exists).
        * **Fallback** (no lazy handle — a hand-added layer / legacy session):
          materializes the display stack via :meth:`get_combined_image_stack`
          EXACTLY as before, so the small-file path is byte-identical; here
          *crop_source* and *image_data* are the same materialized array and
          *channel_indices* is ``None``.

        The extracted pixels are identical either way — ``crop_source`` is the
        same source (or its lazy twin) the eager stack was built from — this
        only changes WHERE the shape + the recap read their data from, and how
        much is read.
        """
        # Extract-from-processed (P1): when the Processed view is on the canvas
        # the raw layers are off-canvas (clean swap), so read the PROCESSED
        # store's pixels at the same full-res box coords instead of the raw
        # handle.  `_crop_source_for_processed` raises a clear ValueError (M3 /
        # no-store / grid-mismatch) that the caller surfaces in the panel; it
        # never returns `(None, None)` (there is no eager fallback here).  The
        # RAW branch below is byte-identical to today — this only activates
        # while the processed view is showing.
        if self._is_processed_view_showing():
            crop_source, channel_indices = \
                self._crop_source_for_processed(channel_names)
            return (crop_source, channel_indices,
                    tuple(int(s) for s in crop_source.shape), None)

        crop_source, channel_indices = self._crop_source_for(channel_names)
        if crop_source is not None:
            return (crop_source, channel_indices,
                    tuple(int(s) for s in crop_source.shape), None)
        # No lazy handle → materialize the display stack (unchanged old path).
        # `get_combined_image_stack` raises "No valid 3D image layers selected."
        # when nothing usable is ticked; the caller catches + surfaces it.
        image_data, _names = self.get_combined_image_stack()
        return (image_data, None,
                tuple(int(s) for s in image_data.shape), image_data)

    # extracts the patches using the bounding boxes  by the user
    def extract_patches(self):
        """PREP + launch for the ③ Extract (F1 — the heavy slice→materialize→write
        now runs on an :class:`_ExtractWorker` QThread).

        Everything that reads a widget / layer / viewer, validates, or early-
        returns stays HERE on the main thread exactly as before (so error
        messages still surface synchronously); the deep-copied result is frozen
        into an :class:`_ExtractJob` and handed to the worker.  The lock + bar +
        button-flip go up ONLY on the success path (grill HIGH-2 validate-then-
        lock); the finished/error/launch-raise slots are the single unlock.
        """
        # Re-entrancy guard (N1/R4): one extract at a time, and never while
        # another Step-1 op holds the surface.  `_extract_launching` covers the
        # pre-lock window (the discard-confirm modal spins the event loop —
        # grill MED-5), `_extract_running` covers the in-flight run.
        if (getattr(self, "_extract_running", False)
                or getattr(self, "_extract_launching", False)
                or getattr(self, "_trim_launching", False)
                or getattr(self, "_step1_busy", False)):
            return
        # Brick 5b item 1: with the source loaded but OFF the canvas (Step 1 no
        # longer re-shows on re-entry), `_live_channel_names` filters every
        # ticked name against `viewer.layers` and returns [] -> the lazy crop is
        # skipped and the EAGER fallback runs, only to die on "No valid 3D image
        # layers selected."  Say what to do instead, plainly, in the panel: a
        # dialog for a one-click fix would be worse.
        #
        # Chosen over making `_live_channel_names` fall back to the retained
        # channel refs: the eager stack applies the SAME live filter, so a
        # fallback there would let the lazy and eager paths disagree on the
        # channel set — and for a plain ndarray source there is no pixel data to
        # fall back TO.  The honest answer is "load it first", not a half-path.
        if self._source_is_off_canvas():
            self._set_export_status(self._OFF_CANVAS_HINT, success=False)
            self._log("STEP1-EXTRACT",
                      "source loaded but off-canvas — asked the user to Load")
            return
        # Extract-from-processed (P1): while the Processed view is showing,
        # `_resolve_extract_source` reads the PROCESSED store's pixels at the
        # same full-res box coords instead of the raw handle (raw layers are
        # off-canvas under the clean swap).  The raw view is byte-identical to
        # today.  No hard-block here any more — the processed branch resolves +
        # surfaces its own honest message (M3) if a ticked channel isn't in the
        # processed set / the store is unavailable.
        # Stop the debounced layout-migration prompt so it can't fire mid-extract
        # and move files under us (R2).
        if getattr(self, "_migration_offer_timer", None) is not None:
            self._migration_offer_timer.stop()

        # `_extract_launching` goes up FIRST — BEFORE the discard-confirm modal —
        # so a second click during the modal's event-loop spin can't re-enter and
        # launch a second worker (grill MED-5).  The finally clears it on EVERY
        # path (abort / validation early-return / launch), so a bailed PREP never
        # leaves the flag stuck and the surface is never left locked (HIGH-2).
        self._extract_launching = True
        try:
            # (B5 removed) No "discard unexported?" confirm: extracted boxes now
            # clear out of ③ on a successful extract (their bbox lives in ④), and
            # the F1 stage-reset below discards the prior run's in-memory stage —
            # so there is nothing lingering to warn about.

            # Resolve the crop source WITHOUT reading the whole volume when a
            # lazy handle exists (see `_resolve_extract_source`): the fallback's
            # "No valid 3D image layers selected." is caught here so it surfaces
            # in the panel, not as a Qt slot traceback.
            try:
                channel_names = self._selected_channel_names()
                crop_source, channel_indices, filter_shape, image_data = \
                    self._resolve_extract_source(channel_names)
            except Exception as exc:
                self._set_export_status(str(exc), success=False)
                self._log("STEP1-EXTRACT", f"cannot build channel stack: {exc}")
                return

            # get the bounding boxes drawn in the shapes layer
            bounding_boxes = self.get_drawn_shapes()

            # Per-box Z crop (box tool's editable-Z), keyed by the
            # rect object's id() so it survives the tick / uncommitted / valid
            # filters below (those keep the SAME array objects, never copy).  A
            # full-Z box gives [0, Z] → the crop is byte-identical, so this is a
            # no-op for the default box AND for the legacy Shapes-layer fallback.
            z_by_id = {}
            _bt = getattr(self, "box_tool", None)
            if _bt is not None and _bt.has_boxes():
                _zrs = _bt.z_ranges()
                if len(_zrs) == len(bounding_boxes):
                    z_by_id = {id(b): [int(z[0]), int(z[1])]
                               for b, z in zip(bounding_boxes, _zrs)}

            if not bounding_boxes:
                self._set_export_status(
                    "No bounding boxes drawn — pick Draw Free or Draw Fixed "
                    "and drag on the image.", success=False)
                self._log("STEP1-EXTRACT", "No bounding boxes found.")
                return

            # "Extract checked": honour ③'s include ticks.  Row i of the
            # Box List corresponds to shape i, so an unticked row drops its box.
            # Without this the tick column is a decorative lie — it looks like a
            # filter and filters nothing.
            rows = getattr(self.widget, "box_row_count", None)
            keep = getattr(self.widget, "checked_box_indices", None)
            if keep is not None and rows is not None:
                chosen = keep()
                n_rows = rows()
                if chosen is None or n_rows != len(bounding_boxes):
                    # The filter is POSITIONAL — row i means shape i — so it is
                    # only safe when the list and the layer agree.  A stale list
                    # (2 rows, 5 shapes) would otherwise drop 3 drawn boxes
                    # silently.  Extract everything and say why.
                    if chosen is not None:
                        self._log("STEP1-EXTRACT",
                                  f"box list is stale ({n_rows} rows vs "
                                  f"{len(bounding_boxes)} boxes) — extracting "
                                  "all of them rather than guessing")
                elif len(chosen) != len(bounding_boxes):
                    n_before = len(bounding_boxes)
                    bounding_boxes = [b for i, b in enumerate(bounding_boxes)
                                      if i in chosen]
                    self._log("STEP1-EXTRACT",
                              f"{n_before - len(bounding_boxes)} box(es) "
                              "unticked and skipped")
                    if not bounding_boxes:
                        self._set_export_status(
                            "No boxes ticked — tick at least one in the box "
                            "list.", success=False)
                        return

            # Exclude boxes already exported this session or restored from disk so
            # they aren't re-extracted + re-numbered high (the escalating bug).
            # Thread each box's editable-Z range (same id()-keyed map the crop
            # uses below) so a same-XY box at a NARROWED Z is compared
            # Z-inclusively and not falsely dropped as already-captured.  A box
            # not in the map (legacy Shapes fallback) -> None -> XY-only sig.
            sig_z_ranges = ([z_by_id.get(id(b)) for b in bounding_boxes]
                            if z_by_id else None)
            uncommitted = select_uncommitted_boxes(
                bounding_boxes, self._committed_box_sigs, sig_z_ranges
            )
            n_already = len(bounding_boxes) - len(uncommitted)
            if not uncommitted:
                self._set_export_status(
                    f"No new ROIs to extract — {n_already} box(es) already "
                    "captured. Draw a new ROI first.", success=False)
                return
            bounding_boxes = [b for _, b in uncommitted]

            # Drop degenerate / out-of-bounds boxes before they burn a monotonic
            # index on an all-zero patch (R10 / D-9).  `filter_shape` is the
            # source's (…, Y, X) shape — from the lazy handle on the L0 path (no
            # read), the materialized stack on the fallback; only Y / X are used
            # and they are identical between the two, so the filter result is
            # unchanged.
            bounding_boxes, n_dropped = filter_extractable_boxes(
                bounding_boxes, filter_shape
            )
            if n_dropped:
                self._set_export_status(
                    f"Skipped {n_dropped} empty / out-of-bounds box(es).",
                    success=False,
                )
            if not bounding_boxes:
                self._set_export_status(
                    "No valid bounding boxes to extract.", success=False)
                self._log("STEP1-EXTRACT", "No valid bounding boxes to extract.")
                return

            # Determine naming prefix from display name.
            display_name = ""
            if hasattr(self.widget, "display_name_edit"):
                display_name = self.widget.display_name_edit.text().strip()
            if display_name:
                name_prefix = f"{display_name}_patch"
            else:
                name_prefix = "patch"

            # Find cumulative start index from workspace.
            workspace = self.widget.output_file_path.text().strip()
            if workspace and os.path.isdir(workspace):
                start_index = find_next_patch_index(workspace, name_prefix)
            else:
                start_index = 1

            # Align the per-box Z ranges to the FINAL (filtered) box list — the
            # id() map above survives the filters, so this is correct after any
            # tick / uncommitted / degenerate drop.  None entries → full-Z.
            z_ranges = ([z_by_id.get(id(b)) for b in bounding_boxes]
                        if z_by_id else None)

            # `source_kind` records WHICH pixels were cropped — the raw source or
            # the prepared processed store — read off the SAME view predicate
            # `_resolve_extract_source` branched on (provenance).
            source_kind = "processed" if self._is_processed_view_showing() else "raw"

            # Deep-copy the FINAL box geometry + z_ranges into the frozen job
            # (grill HIGH-3): the surface lock disables the box *buttons* but NOT
            # canvas mouse interaction, so a drag mid-run would otherwise mutate
            # the very ndarray the worker slices.  `bounding_boxes[i]` may already
            # be a numpy array or a plain list — `np.array(..., copy=True)`
            # snapshots either into an independent (4,2) array, and the worker's
            # `box_signature` / `.tolist()` read it identically to the sync path.
            boxes_snapshot = tuple(
                np.array(b, dtype=float, copy=True) for b in bounding_boxes)
            n = len(boxes_snapshot)
            if z_ranges is None:
                z_snapshot = tuple(None for _ in range(n))
            else:
                z_snapshot = tuple(
                    (list(z) if z is not None else None) for z in z_ranges)

            job = _ExtractJob(
                crop_source=crop_source,
                channel_indices=(list(channel_indices)
                                 if channel_indices is not None else None),
                boxes=boxes_snapshot,
                z_ranges=z_snapshot,
                channel_names=tuple(channel_names),
                name_prefix=name_prefix,
                start_index=int(start_index),
                source_kind=source_kind,
                # The full per-run persist-param set gathered on the MAIN thread
                # (grill MED-2 — reads every widget/self value `persist_patch`
                # needs, incl. source_axes + voxel_size); the worker never
                # touches a widget.
                persist=self._gather_persist_params(),
            )

            # HIGH-1 single-writer discipline: the worker becomes the SOLE
            # `step1_state.json` writer for the run.  Land any pending debounced
            # source-edit save NOW and stop the timer so a re-armed fire can't
            # interleave `load_state`→`save_state` with the worker's per-patch
            # `persist_patch` (lost update).  `_flush_source_state_save` is a
            # no-op when there are no buffered edits; the locked widgets emit no
            # new edits during the run, so the worker's writes stand alone.
            self._flush_source_state_save()
            try:
                self._state_save_timer.stop()
            except Exception:
                pass

            # SUCCESS PATH — reset THIS run's stage + list FIRST (FIX 1;
            # byte-faithful to HEAD, which cleared `extracted_patches` +
            # `extracted_roi_list` at the start of every extract's build and then
            # repopulated).  The async worker repopulates ONLY this run's patches
            # via `written`, so WITHOUT this a SECOND extract in a session
            # accumulates the prior run's patches — then `_on_extract_finished`'s
            # `n_staged (cumulative) - n_written (this run)` fabricates phantom
            # "N failed to write" + a red status, and the discard-confirm's
            # "extracting again will discard them" promise is silently broken.
            self.extracted_patches = {}
            try:
                self.widget.extracted_roi_list.clear()
            except Exception:
                pass

            # Stamp the run token BEFORE the lock/launch try so the except's
            # diagnostic always has a defined `token`, and record the PREP skip/
            # drop counts for the FINISH status line.
            self._extract_token += 1
            token = self._extract_token
            self._extract_status_ctx = {
                "n_already": n_already, "n_dropped": n_dropped}

            # Lock the surface, name the op on the shared bar, put the ③ bar into
            # scan mode, flip Extract to Cancel, then launch — ALL inside ONE try
            # whose except fully unlocks.  FIX 3: `progress_bar.setRange(0, 0)`
            # used to sit UNWRAPPED between the lock and the launch try, so a
            # raise there would brick the surface (locked, no worker to unlock
            # it).  Everything above early-returned BEFORE here, so the surface
            # is only ever locked when a run actually starts (HIGH-2); if
            # worker/thread construction or any of this raises, NEITHER finished
            # nor error fires, so the except clears the lock + bar + button here.
            try:
                self._set_step1_busy(True)
                try:
                    self.widget._busy_show(f"Extracting {n} patch(es)…")
                except Exception:
                    pass
                self.widget.progress_bar.setRange(0, 0)   # scanning (read phase)
                self._extract_running = True
                self._set_extract_button_cancel(True)

                if self._extract_thread is not None:
                    self._draining_extract_threads.append(
                        (self._extract_thread, self._extract_worker))
                worker = _ExtractWorker(job)
                thread = QThread()
                worker.moveToThread(thread)
                thread.started.connect(worker.run)
                worker.written.connect(
                    lambda result, _t=token: self._on_extract_written(result, _t))
                worker.progress.connect(
                    lambda i, tot, phase, _t=token:
                    self._on_extract_progress(i, tot, phase, _t))
                worker.finished.connect(
                    lambda nw, cancelled, _t=token:
                    self._on_extract_finished(nw, cancelled, _t))
                worker.error.connect(
                    lambda msg, _t=token: self._on_extract_error(msg, _t))
                worker.finished.connect(thread.quit)
                worker.error.connect(thread.quit)
                thread.finished.connect(
                    lambda _t=thread: self._cleanup_extract_thread(_t))
                self._extract_thread = thread
                self._extract_worker = worker
                self._log("STEP1-EXTRACT",
                          f"launch token={token} boxes={n} prefix={name_prefix} "
                          f"start={start_index} kind={source_kind}")
                thread.start()
            except Exception as exc:
                self._extract_thread = None
                self._extract_worker = None
                try:
                    self.widget._busy_hide()
                except Exception:
                    pass
                self.widget.progress_bar.setRange(0, 1)
                self.widget.progress_bar.setValue(0)
                self.widget.progress_bar.setFormat("")
                self._set_step1_busy(False)
                self._extract_running = False
                self._set_extract_button_cancel(False)
                self._set_export_status(
                    f"Extract could not start: {exc}", success=False)
                self._log("STEP1-EXTRACT", f"launch FAILED token={token}: {exc}")
        finally:
            self._extract_launching = False

    # ------------------------------------------------------------------
    # ③ Extract worker slots + cancel/dispatch/button (F1)
    # ------------------------------------------------------------------

    def _on_extract_written(self, result, token):
        """One patch finished on the worker — rebuild its in-memory stage entry
        + list row and commit its box signatures (ALL on the main thread).

        The stage is rebuilt for EVERY materialized box (write success OR
        failure — grill MED-4) so a write-failed patch stays in ④'s session list
        for retry, mirroring the synchronous path (which built the stage BEFORE
        persisting).  Sigs are committed only on a successful write (grill LOW-7
        — today's per-patch commit granularity); BOTH the XY-only and the
        Z-inclusive sig land (grill MED-3), so a same-box re-extract dedups."""
        if token != self._extract_token:
            return
        name = result["name"]
        self.extracted_patches[name] = {
            "data": result["data"],
            "channels": result["channels"],
            "bounding_box": result["bounding_box"],
            "source_kind": result["source_kind"],
        }
        try:
            self.widget.extracted_roi_list.addItem(name)
        except Exception:
            pass
        if result.get("persist_ok"):
            self._committed_box_sigs.add(result["sig_xy"])
            self._committed_box_sigs.add(result["sig_z"])

    def _on_extract_progress(self, i, n, phase, token):
        """Drive the ③ bar from the worker's per-box progress (main thread).

        The bar starts in scan mode (set at launch); on the first "writing" tick
        it switches to a determinate "Writing i of n".  A stale / post-finish
        emit is ignored so it can't touch the bar after the run ended."""
        if token != self._extract_token or not getattr(
                self, "_extract_running", False):
            return
        if phase != "writing":
            return
        try:
            bar = self.widget.progress_bar
            bar.setRange(0, n)
            bar.setValue(i)
            bar.setFormat("Writing %v of %m")
        except Exception:
            pass

    def _on_extract_finished(self, n_written, was_cancelled, token):
        """Terminal (normal / cancelled) — refresh the counts + library, write
        the honest status, then the SINGLE unlock.

        STALE guard first (a bumped token owns the UI) so a superseded emit never
        touches the depth-counted busy lock.  Cancel does NOT bump the token, so
        the run's own finished emit runs here and performs the one unlock —
        extract FINISH only touches Step-1 PANEL widgets + the registry (never
        the shared napari canvas), so finalizing is safe even after a step
        change."""
        if token != self._extract_token:
            return
        if int(n_written) > 0:
            # New / re-extracted patches → any cached depth profile is stale.
            self._intensity_curve_cache.clear()
        try:
            # The list may have gained rows via `_on_extract_written`; select the
            # first so ④'s intensity plot has a current patch (mirrors the old
            # end-of-run `setCurrentRow(0)`).
            try:
                lst = self.widget.extracted_roi_list
                if lst.count() > 0 and lst.currentRow() < 0:
                    lst.setCurrentRow(0)
            except Exception:
                pass
            # Registry rows just appeared → refresh the per-source counts + ④'s
            # Library from the registry (provenance + join other sources' saved
            # patches; the in-memory stage stays "staged" so those rows remain
            # tickable — F2).
            self._refresh_source_patch_counts()
            self._refresh_library_from_registry()
            # Extracted boxes graduate to ④ — clear them out of ③ so the box list
            # doesn't linger with already-saved boxes (their bbox is in the
            # Library now).  Un-ticked + persist-failed boxes are kept.
            self._clear_extracted_boxes_from_tool()

            # ③→④ bridge: a successful Extract graduates patches into the
            # Library (and their boxes just left ③), so open ④ Review & Library
            # to show them.  Only when something actually landed on disk.
            if int(n_written) > 0:
                try:
                    self.widget.open_review_library_section()
                except Exception:
                    pass

            n_staged = len(getattr(self, "extracted_patches", {}) or {})
            n_failed = n_staged - int(n_written)
            ctx = getattr(self, "_extract_status_ctx", {}) or {}
            n_already = int(ctx.get("n_already", 0))
            n_dropped = int(ctx.get("n_dropped", 0))
            if was_cancelled:
                summary = (f"Cancelled — wrote {int(n_written)} of {n_staged} "
                           "patch(es) to the library.")
            else:
                summary = (f"Extracted {n_staged} new patch(es); wrote "
                           f"{int(n_written)} to the library.")
            if n_failed > 0:
                summary += f" {n_failed} failed to write."
            if n_already:
                summary += f" {n_already} already-captured box(es) skipped."
            if n_dropped:
                summary += f" {n_dropped} empty/out-of-bounds skipped."
            self._set_export_status(
                summary, success=(n_failed == 0 and not was_cancelled))
            self._refresh_conditioning_status()
        finally:
            self._finish_extract_unlock()

    def _on_extract_error(self, msg, token):
        """Fatal worker error — red status + the SINGLE unlock (already-written
        patches were committed as they arrived via `_on_extract_written`)."""
        if token != self._extract_token:
            return
        try:
            self._set_export_status(f"Extract failed: {msg}", success=False)
            self._log("STEP1-EXTRACT", f"worker error token={token}: {msg}")
        finally:
            self._finish_extract_unlock()

    def _finish_extract_unlock(self):
        """The one place the extract run's lock/bar/button are released — reached
        from finished + error (never twice: only the token-owner reaches it, and
        the two are mutually exclusive terminal signals)."""
        try:
            self.widget.progress_bar.setRange(0, 1)
            self.widget.progress_bar.setValue(0)
            self.widget.progress_bar.setFormat("")
        except Exception:
            pass
        try:
            self.widget._busy_hide()
        except Exception:
            pass
        self._set_step1_busy(False)
        self._extract_running = False
        self._set_extract_button_cancel(False)
        # HIGH-1 (FIX 2b): the worker was the SOLE `step1_state.json` writer for
        # the run, so any main-thread source-edit flush was deferred (PREP
        # drained + stopped the timer; `on_step_exit` skips its flush while
        # `_extract_running`).  Now that `_extract_running` is clear, re-run the
        # flush so a step-exit-mid-run edit is persisted post-run — a no-op when
        # the buffer is empty (the normal case), so it never double-writes.
        try:
            self._flush_source_state_save()
        except Exception:
            pass

    def cancel_extract(self):
        """Request a cooperative stop of the in-flight extract.

        Does NOT unlock and does NOT bump the token: the worker checks the flag
        between boxes, breaks, and emits `finished(n, was_cancelled=True)`, and
        `_on_extract_finished` performs the SINGLE unlock (extract FINISH is
        canvas-safe — it touches only Step-1 panel widgets + the registry — and
        `_extract_running` serializes runs so there is no cancel→relaunch
        overlap).  Already-written patches stay on disk (each write is atomic)."""
        worker = getattr(self, "_extract_worker", None)
        if worker is not None:
            try:
                worker.request_cancel()
            except Exception:
                pass
        # Flip the button to a disabled "Cancelling…" so a second click can't
        # re-enter; the finished slot restores it.
        btn = getattr(self.widget, "extract_roi", None)
        if btn is not None:
            try:
                btn.setEnabled(False)
                btn.setText("Cancelling…")
            except Exception:
                pass
        self._log("STEP1-EXTRACT", "cancel requested")

    def on_extract_or_cancel_clicked(self):
        """The Extract button doubles as Cancel while an extract is in flight.

        Dispatches on `_extract_running` (True only between launch and
        finish/error) so a click DURING a run cancels it and a click otherwise
        starts a fresh extract.  Mirrors `on_load_or_cancel_clicked`."""
        if getattr(self, "_extract_running", False):
            self.cancel_extract()
        else:
            self.extract_patches()

    def _set_extract_button_cancel(self, active: bool) -> None:
        """Flip the Extract button to 'Cancel' (kept clickable) while a run is in
        flight, or back to its idle label when it finishes.  Best-effort; touches
        only the button text/icon/enabled state (mirrors
        :meth:`_set_load_button_cancel`)."""
        btn = getattr(self.widget, "extract_roi", None)
        if btn is None:
            return
        try:
            if active:
                if self._extract_button_text is None:
                    self._extract_button_text = btn.text()
                    self._extract_button_icon = btn.icon()
                btn.setEnabled(True)          # stays clickable as Cancel
                btn.setText("Cancel")
                try:
                    from napari_epicycle.ui_shell.theme import apply_button_icon
                    apply_button_icon(btn, "mdi6.close-circle-outline")
                except Exception:
                    pass
            elif self._extract_button_text is not None:
                btn.setEnabled(True)
                btn.setText(self._extract_button_text)
                if self._extract_button_icon is not None:
                    btn.setIcon(self._extract_button_icon)
                self._extract_button_text = None
                self._extract_button_icon = None
        except Exception:
            pass

    def _cleanup_extract_thread(self, thread=None):
        """Drop the finished extract (thread, worker) refs — identity-guarded,
        removes any matching draining pair, and `deleteLater`s the thread
        (mirrors :meth:`_cleanup_source_load_thread`)."""
        draining = getattr(self, "_draining_extract_threads", None)
        if draining is not None and thread is not None:
            self._draining_extract_threads = [
                (t, w) for (t, w) in draining if t is not thread]
        if thread is None or thread is self._extract_thread:
            self._extract_thread = None
            self._extract_worker = None
        if thread is not None:
            try:
                thread.deleteLater()
            except Exception:
                pass

    def _gather_persist_params(self) -> dict:
        """Read EVERY widget / self value the per-patch write needs, on the MAIN
        thread, into a plain dict the worker (or :meth:`_persist_patch_to_disk`)
        can pass straight to the pure :func:`persist_patch` (grill MED-2 — the
        full provenance set).  A worker thread must never touch a Qt widget, so
        this is the single main-thread gather point.

        Byte-identical to the inline reads `_persist_patch_to_disk` used to do:
        the workspace via ``output_file_path.text()`` (no strip — matches the
        historical read), the source provenance, the isotropic-1/1/1-normalized
        voxel size, and the honest on-disk source axes."""
        workspace = self.widget.output_file_path.text()

        # Register provenance against the loaded FILE (per-file source model, D-3);
        # fall back to the source folder if no file is tracked.
        source_path = getattr(self, "_loaded_source_file", None) or ""
        if not source_path and hasattr(self.widget, "source_folder_path"):
            source_path = self.widget.source_folder_path.text().strip()
        source_display_name = ""
        if hasattr(self.widget, "display_name_edit"):
            source_display_name = self.widget.display_name_edit.text().strip()

        # Read the voxel spins, NORMALIZING the isotropic 1/1/1 default to None
        # (mirror `_prepare_voxel_size`): a raw [1,1,1] must NOT be persisted as
        # an authoritative spacing.
        voxel_size = None
        if hasattr(self.widget, "voxel_z_spin"):
            try:
                vz = float(self.widget.voxel_z_spin.value())
                vy = float(self.widget.voxel_y_spin.value())
                vx = float(self.widget.voxel_x_spin.value())
            except Exception:
                vz = vy = vx = 1.0
            if (vz, vy, vx) != (1.0, 1.0, 1.0) and vz > 0 and vy > 0 and vx > 0:
                voxel_size = [vz, vy, vx]

        # The source's on-disk order.  Prefer the store's HONEST on-disk order
        # (`_source_disk_axes`, B2-N3); for a raw TIFF load that is None and
        # `_source_axes_resolved` IS the on-disk order captured at load.
        source_axes = (getattr(self, "_source_disk_axes", None)
                       or getattr(self, "_source_axes_resolved", None))
        if not source_axes and hasattr(self.widget, "axis_order_combo"):
            source_axes = self.widget.axis_order_combo.currentText().strip() or None

        return {
            "workspace": workspace,
            "source_path": source_path,
            "source_display_name": source_display_name,
            "voxel_size": voxel_size,
            "source_axes": source_axes,
            # Brick 4a a4: honest provenance for patches cut while the original
            # image file could not be found -- the pixels came out of the
            # prepared copy.  None (and so absent from the sidecar) normally.
            "source_from": ("prepared_store"
                            if getattr(self, "_raw_missing", False) else None),
        }

    def _set_step1_busy(self, busy):
        """Lock/unlock the whole Step-1 action surface during an extract so a
        queued or double click cannot re-enter or launch a conflicting op
        (N1/R4).  Best-effort: silently skips any button the widget lacks.

        **Depth-counted.**  The save/restore used to be depth-1: a nested
        lock overwrote the snapshot with the already-disabled values, and the
        first unlock then popped it and left every listed widget dead with
        `_step1_busy == False` — i.e. the surface unusable while the flag said
        idle, and the only code that re-enables it unreachable.  Only
        `extract_patches` guards on the flag before locking; `load_source` and
        both export paths do not, so the nesting has to be survivable here
        rather than assumed away at the call sites.
        """
        if not hasattr(self, "_pre_busy_enabled"):
            self._pre_busy_enabled = {}
        depth = getattr(self, "_busy_depth", 0)
        if busy:
            depth += 1
        else:
            depth = max(0, depth - 1)
        self._busy_depth = depth
        self._step1_busy = depth > 0
        # Snapshot only on 0→1 and restore only on 1→0.  Intermediate
        # transitions must leave both the snapshot and the widgets alone.
        if (busy and depth > 1) or (not busy and depth > 0):
            return
        # These are exactly what must lock while a ② op runs: the source
        # combo, the dimension-order combo, the voxel-spacing fields, the single
        # Apply, plus ③'s draw + Extract and ④'s Export.  Resolved by getattr
        # with a None guard, so a name that does not exist fails SILENTLY
        #  — tests assert every one of these resolves.
        for name in ("extract_roi", "save_reviewed_roi", "export_all_ready_roi",
                     "load_source_button", "prepare_largedata_button",
                     "clear_shapes_button",
                     "source_combo", "axis_order_combo",
                     "voxel_z_spin", "voxel_y_spin", "voxel_x_spin",
                     "voxel_reset_button", "apply_preprocess_button",
                     # ④ Z-crop Trim button: the Z buttons were
                     # NOT locked before — a queued click during a Step-1 op
                     # (extract / another trim) could otherwise launch a second
                     # destructive crop-in-place.  `auto_select_zrange_button`
                     # (snap) stays live — it only SETS the band, never writes.
                     "set_select_zrange_button",
                     # ② preview REGION controls (Preview v2 Piece 2): Place +
                     # Recenter + Exit + the four X/Y/Z/Z-depth position sliders
                     # (S/M/L presets are a LIST → locked in the loop below).  A
                     # queued click / drag on these during a Step-1 op could
                     # move/replace the region mid-compute.
                     "place_preview_button", "recenter_preview_button",
                     "exit_preview_button", "preview_x_slider",
                     "preview_y_slider", "preview_z_slider",
                     "preview_zdepth_slider",
                     "free_bbox_button", "fixed_bbox_button",
                     # ③ box-tool controls (Phase 1) lock during a Step-1 op,
                     # mirroring the retired draw buttons above.  `box_place_check`
                     # is a checkbox but still exposes setEnabled — a queued
                     # click on it during the ④ export funnel (which drains
                     # events) could otherwise flip the viewer mid-export.
                     "box3d_mode_button",
                     "box_add_button", "box_delete_button", "box_place_check",
                     # v2 absolute-size custom controls (the 128³/256³/512³
                     # buttons are a LIST -> locked in the loop below).
                     "box_custom_size_spin", "box_custom_apply_button",
                     "box_lock_size_check",
                     "unload_source_button", "add_source_button",
                     "source_browse_button",
                     # ② preprocessing: the per-channel columns live inside ONE
                     # container; disabling it locks every per-column control
                     # (+ Add step, the geometry ⚙, reorder / remove, param
                     # knobs) in one shot + survives the wholesale column
                     # rebuild.  This replaces the single `add_op_button` the
                     # flat op-chain used before Brick 1b.
                     "recipe_columns_container", "test_region_button",
                     "display_name_edit", "source_file_list",
                     # `load_source` drains the event queue (processEvents)
                     # BEFORE it launches the read worker, so a click queued on
                     # Browse lands mid-load, changes `output_file_path`, and
                     # `_restore_bounding_boxes(display_name)` (run in
                     # `_on_source_loaded`) then restores boxes from a DIFFERENT
                     # workspace than the one the load was started against.
                     "browse_button"):
            btn = getattr(self.widget, name, None)
            if btn is None or not hasattr(btn, "setEnabled"):
                continue
            if busy:
                # Remember what was enabled BEFORE the lock, so the unlock
                # restores rather than forces.  Without this, one busy cycle
                # switched on every deliberately-disabled control — e.g. Apply,
                # which has no handler at all.
                self._pre_busy_enabled[name] = btn.isEnabled()
                btn.setEnabled(False)
            else:
                btn.setEnabled(self._pre_busy_enabled.pop(name, True))

        # ③'s S/M/L size presets are a LIST (not a single attr), so they cannot
        # ride the string-literal name loop above (the busy-lock AST test would
        # then try `.setEnabled` on the list itself).  Same driver, per button:
        # a queued size click during the ④ export funnel could flip the viewer.
        for list_name in ("box_size_preset_buttons", "box_preset_buttons",
                          "preview_size_preset_buttons"):
            for i, btn in enumerate(
                    getattr(self.widget, list_name, None) or []):
                if not hasattr(btn, "setEnabled"):
                    continue
                key = f"{list_name}[{i}]"
                if busy:
                    self._pre_busy_enabled[key] = btn.isEnabled()
                    btn.setEnabled(False)
                else:
                    btn.setEnabled(self._pre_busy_enabled.pop(key, True))

    def _set_selected_z_range(self, z_range):
        """The ONE place `selected_z_range` is written, so the editor follows it.

        There were two live writers (the span's `onselect` and the autoselect
        button) and neither told the Z-truncate spinboxes, which are the
        controls that LOOK like they hold this value.  One chokepoint instead of
        two parallel writes, mirroring `_rebuild_shape_features` for the box
        features.  `Set Range` still commits from the attribute, unchanged.
        """
        self.selected_z_range = z_range
        seed = getattr(self.widget, "seed_ztrunc_spins", None)
        if seed is not None:
            try:
                seed(z_range)
            except Exception as exc:      # never let a readout break a select
                self._log("STEP1-ZRANGE", f"could not seed spins: {exc}")

    def _clear_extracted_boxes_from_tool(self):
        """After a successful extract, remove from ③'s box tool the boxes that
        were actually SAVED — their bbox lives in ④ Review & Library now.
        Un-ticked / persist-failed boxes (sig never committed) are KEPT.

        Matched by XY signature against ``_committed_box_sigs`` (reorder-safe).
        A box is cleared only when its XY sig is committed AND **unique** among
        the drawn boxes: a shared XY footprint is the editable-Z multi-slab case
        (two boxes, same XY, different Z), where XY-only matching cannot tell the
        extracted slab from a kept one — so both are left for the user rather
        than risk removing explicitly-kept geometry (audit MED)."""
        bt = getattr(self, "box_tool", None)
        if bt is None or not bt.has_boxes():
            return
        try:
            rects = bt.rectangles()
        except Exception as exc:
            self._log("STEP1-BOX", f"box-clear rectangles() failed: {exc}")
            return
        committed = getattr(self, "_committed_box_sigs", set())
        sigs = [box_signature(r) for r in rects]
        counts = Counter(sigs)
        to_remove = [i for i, s in enumerate(sigs)
                     if s in committed and counts[s] == 1]
        if to_remove:
            try:
                bt.remove_boxes(to_remove)
            except Exception as exc:
                self._log("STEP1-BOX", f"box-clear remove failed: {exc}")

    def _on_plot_intensity_clicked(self):
        current_item = self.widget.extracted_roi_list.currentItem()
        if current_item is None:
            self._log("PLOT-DISK", "No patch selected.")
            return
        patch_name = current_item.text()

        # Cache hit: re-render the already-computed curves instantly (a re-select
        # or a subsection-return must NOT re-read the TIFF + re-reduce).
        cached = self._intensity_curve_cache.get(patch_name)
        if cached is not None:
            curves, n_z, z_range = cached
            self._snap_range = (patch_name, z_range)
            # Render on the NEXT event-loop tick — as the disk worker's QUEUED
            # `done` signal did.  A SYNCHRONOUS rebuild here fires mid-section-
            # switch and paints the LinearRegionItem before its ViewBox has
            # geometry (`viewRect()` is None → `QRectF(None)` TypeError crash).
            # Deferring lets the ④ section lay out first; stale-guarded on the tick.
            from qtpy.QtCore import QTimer
            QTimer.singleShot(
                0, lambda p=patch_name, c=curves, n=n_z:
                self._render_cached_intensity(p, c, n))
            return

        # In-memory (staged this session): the pixels are already in RAM, so the
        # per-Z reduction is cheap — reduce + plot inline.
        patch_info = self.extracted_patches.get(patch_name)
        if patch_info is not None:
            self._log("PLOT-DISK",
                      f"{patch_name}: plotting from in-memory pixels")
            try:
                curves, n_z = _z_intensity_curves_pure(
                    patch_info["data"], patch_info.get("channels") or [])
            except Exception as exc:                    # noqa: BLE001
                self._log("PLOT-DISK", f"{patch_name}: reduce failed: {exc}")
                return
            self._intensity_curve_cache[patch_name] = (curves, n_z, None)
            self._render_intensity_profile(patch_name, curves, n_z)
            return

        # HIGH-3 gate: a DISK read must NOT relaunch while a Step-1 op / trim
        # holds the surface — a re-select mid-trim-replace would open the very
        # TIFF `os.replace` is about to overwrite (Windows sharing violation).
        # The in-memory branch above is always safe (no disk), so only the disk
        # branch is gated (keeps the post-extract in-memory auto-plot working).
        if getattr(self, "_step1_busy", False) or getattr(
                self, "_trim_running", False):
            self._log("PLOT-DISK",
                      f"{patch_name}: a Step-1 op is running — deferring the "
                      "disk read")
            return

        # Captured (on disk — the reopened-workspace case that used to print
        # "No data found" and leave the plot blank): load the patch TIFF OFF the
        # UI thread (it can be large for this user's data), reduce to the tiny
        # per-Z curves, then plot on the main thread.
        path = self._patch_disk_path(patch_name)
        if path is None:
            self._log("PLOT-DISK",
                      f"{patch_name}: no on-disk TIFF under the workspace — "
                      "cannot plot")
            return
        prev = getattr(self, "_intensity_worker", None)
        if prev is not None and prev.isRunning():
            # A load for a DIFFERENT patch is still running.  Latch this patch as
            # pending + show its placeholder; when that load finishes we re-trigger
            # for whatever is selected then (coalesce to latest), so a superseded
            # selection never stays stuck on a lying "Loading…" (audit MED).
            self._intensity_pending_patch = patch_name
            self._clear_intensity_profile("Loading depth profile…")
            self._log("PLOT-DISK", f"{patch_name}: a load is running — queued")
            return
        channel_names = self._patch_channel_names_from_sidecar(patch_name)
        self._log("PLOT-DISK",
                  f"{patch_name}: loading pixels from disk -> {path}")
        # Busy cue lives IN the plot area (the "Plot Intensity" button is gone):
        # a placeholder until the worker's result (or a failure) replaces it.
        self._clear_intensity_profile("Loading depth profile…")
        worker = _IntensityProfileWorker(patch_name, path, channel_names)
        self._intensity_worker = worker
        worker.done.connect(self._on_intensity_profile_ready)
        worker.failed.connect(self._on_intensity_profile_failed)
        worker.finished.connect(self._on_intensity_worker_finished)
        self._set_review_busy(True)   # slim ④ bar scans during the disk read
        worker.start()

    def _set_review_busy(self, on):
        """Drive the slim ④ (right-column) ``export_progress_bar``: an
        indeterminate scan (0,0) while a depth-load / trim / capture runs, idle
        (0,1) otherwise.  These review ops are mutually exclusive via `_step1_busy`
        + selection gating; the ③ Extract drives a DIFFERENT bar (`progress_bar`),
        so there is no determinate-vs-marquee clash on this widget.  Guarded — a
        no-op if the bar isn't present."""
        # Mirror the ② source bar's busy pattern exactly (`_busy_show`/`_busy_hide`
        # in ui.py): `setRange(0, 0)` is Qt's native indeterminate marquee, the
        # SAME cue the rest of Step 1 uses.  On a fast (small-data) load it only
        # flashes; on a real (dev-PC) load it marquees for the whole read.
        try:
            bar = self.widget.export_progress_bar
            if on:
                bar.setRange(0, 0)
            else:
                bar.setRange(0, 1)
                bar.setValue(0)
        except Exception:
            pass

    def _render_cached_intensity(self, patch_name, curves, n_z):
        """Deferred (next-tick) render of a cache-hit depth profile — stale-guarded
        (the selection may have moved before the tick) so a superseded patch's
        curves never paint over the current one."""
        try:
            cur = self.widget.extracted_roi_list.currentItem()
        except Exception:
            return
        if cur is None or cur.text() != patch_name:
            return
        self._render_intensity_profile(patch_name, curves, n_z)

    def _on_intensity_worker_finished(self):
        """A disk profile-load thread finished — drain any patch queued while it
        ran (coalesce to the CURRENT selection) so a superseded selection never
        stays stuck on the 'Loading…' placeholder (audit MED).  The finished
        worker is no longer running, so the re-triggered load passes the guard."""
        self._set_review_busy(False)   # scan done
        pending = getattr(self, "_intensity_pending_patch", None)
        self._intensity_pending_patch = None
        if not pending:
            return
        cur = self.widget.extracted_roi_list.currentItem()
        if cur is not None and cur.text() == pending:
            self._on_plot_intensity_clicked()

    def _clear_intensity_profile(
            self, message="Select a patch to see its depth profile."):
        """Empty the ④ depth-plot area and show a dim centered placeholder — used
        for no-selection (empty state), a disk load in flight ("Loading…"), or a
        failed load.  Main-thread only; best-effort.  Resets `_plot_n_z` so the
        Trim precondition can't act on a stale depth once the plot is gone."""
        try:
            from qtpy.QtWidgets import QLabel
            from qtpy.QtCore import Qt
            area = self.widget.plot_area
            layout = area.layout()
            if layout is None:
                layout = QVBoxLayout()
                area.setLayout(layout)
            else:
                while layout.count():
                    item = layout.takeAt(0)
                    if item.widget():
                        item.widget().deleteLater()
            placeholder = QLabel(message)
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setWordWrap(True)
            placeholder.setStyleSheet(
                "color:#8a8a9a; font-style:italic; padding:24px;")
            layout.addWidget(placeholder)
            self._plot_n_z = None
            self._plot_n_z_patch = None
        except Exception as exc:      # noqa: BLE001
            self._log("PLOT-DISK", f"clear-plot failed: {exc}")

    def _on_intensity_profile_ready(self, patch_name, curves, n_z, z_range=None):
        # Stale-guard: only plot if this patch is still the selected row (the
        # user may have clicked a different patch while the load ran).
        cur = self.widget.extracted_roi_list.currentItem()
        if cur is None or cur.text() != patch_name:
            self._log("PLOT-DISK",
                      f"{patch_name}: selection changed before load finished — "
                      "discarding")
            return
        # Cache the detected sample Z-range for ④'s "snap" — a REOPENED patch has
        # no in-memory pixels, so the range is computed off-thread WITH the plot
        # load here (keyed by patch so snap can't apply the wrong patch's range).
        self._snap_range = (patch_name, z_range)
        self._intensity_curve_cache[patch_name] = (curves, n_z, z_range)
        self._log("PLOT-DISK",
                  f"{patch_name}: loaded, plotting ({n_z} slices)")
        self._render_intensity_profile(patch_name, curves, n_z)

    def _on_intensity_profile_failed(self, patch_name, msg):
        self._log("PLOT-DISK", f"{patch_name}: disk load failed: {msg}")
        # Replace the "Loading…" placeholder with an honest failure note — but
        # only if this patch is still the selected row (else a newer selection
        # owns the plot).
        cur = self.widget.extracted_roi_list.currentItem()
        if cur is not None and cur.text() == patch_name:
            self._clear_intensity_profile(
                "Couldn't load this patch's depth profile.")

    def _detach_intensity_worker(self):
        """Retire the in-flight intensity-plot worker crash-safely (audit HIGH).

        `tifffile.imread` can't be interrupted, so we can't force-stop it; instead
        drop the controller ref but DISCONNECT its signals (a late emit is inert)
        and, if still running, retain it in the MODULE-level `_LIVE_INTENSITY_
        WORKERS` until it finishes — so the QThread is never destroyed while
        running (`QThread destroyed while running` abort).  Mirrors the capture
        widget's `_detach_worker`.  Best-effort; never raises."""
        w = getattr(self, "_intensity_worker", None)
        self._intensity_worker = None
        # Drop any queued-selection latch too, so a detached load can't later
        # re-trigger a stale patch (audit MED coalescing).
        self._intensity_pending_patch = None
        # Detaching disconnects the worker's `finished`, so its scan-clear never
        # fires — clear the ④ bar here so a detached depth-load leaves it idle.
        self._set_review_busy(False)
        if w is None:
            return
        for sig in ("done", "failed", "finished"):
            try:
                getattr(w, sig).disconnect()
            except Exception:
                pass
        try:
            if w.isRunning():
                _LIVE_INTENSITY_WORKERS.add(w)
                w.finished.connect(
                    lambda _w=w: _LIVE_INTENSITY_WORKERS.discard(_w))
                w.finished.connect(w.deleteLater)
            else:
                w.deleteLater()
        except Exception:
            pass

    def _patch_disk_path(self, patch_name):
        """Absolute path to a saved patch's TIFF under the workspace, or None."""
        try:
            ws = self.widget.output_file_path.text().strip()
        except Exception:
            return None
        if not ws:
            return None
        p = os.path.join(ws, "artifacts", "step1", "patches",
                         f"{patch_name}.tiff")
        return p if os.path.isfile(p) else None

    def _patch_channel_names_from_sidecar(self, patch_name):
        """`channel_names` from a saved patch's frozen sidecar, or []."""
        try:
            ws = self.widget.output_file_path.text().strip()
        except Exception:
            return []
        if not ws:
            return []
        side = os.path.join(ws, "artifacts", "step1", "patches",
                            f"{patch_name}_metadata.json")
        try:
            with open(side, "r", encoding="utf-8") as f:
                meta = json.load(f)
            cn = meta.get("channel_names")
            return list(cn) if isinstance(cn, (list, tuple)) else []
        except Exception:
            return []

    def _render_intensity_profile(self, patch_name, curves, n_z):
        """Build the ④ Z-intensity plot — pyqtgraph, themed like the ② Load &
        Adjust histogram (dark ``#1a1a2e`` ground, faint grid) — from pre-reduced
        `curves` (per-channel min-max-normalized mean-through-Z arrays) + `n_z`
        (slice count).  Main-thread only.  A draggable region writes the
        Z-truncation range onto the selected patch (replaces the old matplotlib
        SpanSelector)."""
        import pyqtgraph as pg

        # Remember the current patch's slice count (the freshest depth) so the
        # ④ Trim precondition can clamp `[0, current_depth]` without a main-
        # thread disk read (spins/band re-clamp).
        self._plot_n_z = int(n_z)
        self._plot_n_z_patch = patch_name

        # Clear the plot area (deleteLater any prior canvas/plot).
        layout = self.widget.plot_area.layout()
        if layout is None:
            layout = QVBoxLayout()
            self.widget.plot_area.setLayout(layout)
        else:
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

        plot = pg.PlotWidget()
        plot.setBackground("#1a1a2e")                       # ②'s ground
        plot.showGrid(x=True, y=True, alpha=20 / 255)
        for _axis in ("left", "bottom"):
            axis = plot.getAxis(_axis)
            axis.setPen(pg.mkPen("#5a5a6a"))
            axis.setTextPen(pg.mkPen("#c8c8d4"))
        plot.setLabel("bottom", "Z-slice")
        plot.setLabel("left", "intensity (normalized)")
        plot.setTitle(str(patch_name), color="#dddddd", size="9pt")
        plot.addLegend(offset=(-10, 8))

        _CURVE_COLORS = ("#4fc3f7", "#ff8a65", "#81c784", "#ba68c8",
                         "#fff176", "#f06292")
        z = list(range(n_z))
        for i, (label, intens) in enumerate(curves):
            pen = pg.mkPen(_CURVE_COLORS[i % len(_CURVE_COLORS)], width=2)
            plot.plot(z, [float(v) for v in intens], pen=pen, name=str(label))

        # Draggable Z-truncation band (mirrors ②'s LinearRegionItem), seeded to
        # the patch's current range or the full stack.  Its edges write the
        # truncation range on release — the same contract the SpanSelector had.
        cur = getattr(self, "selected_z_range", None)
        if cur and len(cur) == 2:
            z0, z1 = cur
        else:
            # Fresh patch (band reset on select): adopt the FULL range as the
            # current band value so the band (shown full) and Trim AGREE — a Trim
            # on an un-dragged full band reads as "nothing to trim", not a stale
            # range carried from the previously-selected patch.
            z0, z1 = 0, n_z
            self.selected_z_range = (0, n_z)
        region = pg.LinearRegionItem(
            values=(float(z0), float(z1)),
            brush=pg.mkBrush(79, 195, 247, 40),
            pen=pg.mkPen("#FAF9F6", width=1))
        region.setBounds([0, n_z])
        plot.addItem(region)

        def _on_region_changed():
            lo, hi = region.getRegion()
            lo = max(0, int(round(lo)))
            hi = min(n_z, int(round(hi)))
            self._set_selected_z_range((lo, hi))
            self._log("STEP1-ZRANGE",
                      f"Selected Z range: {self.selected_z_range}")

        region.sigRegionChangeFinished.connect(_on_region_changed)
        self._zrange_region = region      # ref for the 'snap' button
        layout.addWidget(plot)

    def _on_autoselect_zrange_clicked(self):
        """④ "snap" — set the Z-truncate band to the auto-detected SAMPLE range
        (the UNION Z-slab that contains signal across channels).  A patch
        extracted THIS SESSION has its pixels in RAM → detect synchronously; a
        REOPENED patch (no in-memory pixels) applies the range the depth-profile
        load already computed OFF the UI thread (`_snap_range`, cached in
        `_on_intensity_profile_ready`) — this is what used to make snap a silent
        no-op for reopened patches."""
        current_item = self.widget.extracted_roi_list.currentItem()
        if current_item is None:
            self._log("STEP1-ZRANGE", "No patch selected to snap.")
            return
        patch_name = current_item.text()

        # In-memory (this session): detect synchronously — the pixels are in RAM
        # and a patch is small.
        patch_info = self.extracted_patches.get(patch_name)
        if patch_info is not None:
            try:
                z0, z1 = detect_sample_range_all_channels(patch_info["data"])
                self._apply_snapped_range(patch_name, int(z0), int(z1))
            except Exception as exc:      # noqa: BLE001
                self._set_export_status(
                    "Couldn't detect a sample range for this patch.", success=False)
                self._log("STEP1-ZRANGE", f"snap (in-memory) failed: {exc}")
            return

        # Reopened (on disk): use the range the depth-profile load computed
        # off-thread; if it isn't ready yet, ask the user to retry (the plot
        # auto-loads on select, so it's normally cached by the time they click).
        cached = getattr(self, "_snap_range", None)   # (patch_name, (z0,z1)|None)
        if cached and cached[0] == patch_name and cached[1] is not None:
            z0, z1 = cached[1]
            self._apply_snapped_range(patch_name, int(z0), int(z1))
        elif cached and cached[0] == patch_name:
            self._set_export_status(
                "No sample signal detected in this patch — nothing to snap to.",
                success=False)
        else:
            self._set_export_status(
                "Still reading this patch — click Snap again in a moment.",
                success=False)

    def _apply_snapped_range(self, patch_name, z0, z1):
        """Set the ④ Z-truncate band + spins (and the in-memory stage, if the
        patch is staged) to *z0..z1* and move the draggable plot region to
        match."""
        self._set_selected_z_range((z0, z1))
        entry = self.extracted_patches.get(patch_name)
        if entry is not None:
            entry["selected_z_range"] = (z0, z1)
        region = getattr(self, "_zrange_region", None)
        if region is not None:
            try:
                region.setRegion((float(z0), float(z1)))
            except Exception:
                pass
        self._log("STEP1-ZRANGE", f"Snapped '{patch_name}' to sample Z {z0}–{z1}")
        self._refresh_conditioning_status()

    # ------------------------------------------------------------------
    # ④ Z-crop TRIM (crop-in-place) — PREP + launch + worker slots
    # ------------------------------------------------------------------

    def _patch_sidecar_path(self, patch_name):
        """Absolute path to a saved patch's FROZEN sidecar, or None."""
        try:
            ws = self.widget.output_file_path.text().strip()
        except Exception:
            return None
        if not ws:
            return None
        p = os.path.join(ws, "artifacts", "step1", "patches",
                         f"{patch_name}_metadata.json")
        return p if os.path.isfile(p) else None

    def _current_patch_depth(self, patch_name):
        """The CURRENT patch's Z depth for the trim clamp, without a main-thread
        disk read.  Prefers the intensity plot's own slice count (freshest, and
        exactly what the user sees); falls back to the sidecar's (advisory)
        ``cropped_shape`` Z; None if neither is available (the worker still
        re-validates against the true TIFF)."""
        if getattr(self, "_plot_n_z_patch", None) == patch_name:
            nz = getattr(self, "_plot_n_z", None)
            if nz:
                return int(nz)
        side = self._patch_sidecar_path(patch_name)
        if side is not None:
            try:
                with open(side, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                shp = meta.get("cropped_shape")
                if isinstance(shp, (list, tuple)) and len(shp) in (3, 4):
                    return int(shp[1] if len(shp) == 4 else shp[0])
            except Exception:
                pass
        return None

    def _on_set_zrange_clicked(self):
        """④ "Trim" — crop the selected saved patch IN PLACE to the chosen Z
        range (rewired 2026-08-19; the old inert "store the range" no-op).

        Destructive + irreversible: confirm, then launch a :class:`_TrimWorker`
        that overwrites the patch TIFF + rewrites its FROZEN sidecar via the
        recoverable :func:`crop_patch_in_place`.  Everything that reads a widget
        / validates / early-returns stays HERE on the main thread; the worker
        does only the file I/O.  Mirrors the F1 EXTRACT PREP (validate-then-lock,
        `_trim_launching` up BEFORE the confirm modal, launch-in-try whose except
        fully unlocks, one `_finish_trim_unlock` from both terminals)."""
        # HARD re-entrancy guard: the Z buttons weren't locked
        # before — no trim while another Step-1 op / a trim holds the surface.
        if (getattr(self, "_step1_busy", False)
                or getattr(self, "_extract_running", False)
                or getattr(self, "_trim_running", False)
                or getattr(self, "_trim_launching", False)):
            return

        current_item = self.widget.extracted_roi_list.currentItem()
        if current_item is None:
            self._log("STEP1-TRIM", "No patch selected to trim.")
            return
        patch_name = current_item.text()

        # Precondition: the patch must have an on-disk TIFF to overwrite (a
        # staged-only patch that failed to persist has nothing to trim).  ④'s
        # `extracted_roi_list` drives the trim; the disk path is resolved exactly
        # like the capture button (`_patch_disk_path`).
        path = self._patch_disk_path(patch_name)
        sidecar = self._patch_sidecar_path(patch_name)
        if path is None or sidecar is None:
            self._set_export_status(
                f"'{patch_name}' has no saved patch file yet — extract it before "
                "trimming.", success=False)
            return

        zr = getattr(self, "selected_z_range", None)
        if zr is None or len(zr) != 2:
            self._set_export_status(
                "No Z-range selected — drag the band or type a range first.",
                success=False)
            return
        z0, z1 = int(zr[0]), int(zr[1])
        depth = self._current_patch_depth(patch_name)
        # Confirm hard-enforces 0 <= z0 < z1 <= current_depth (the worker
        # re-validates against the true TIFF as belt-and-suspenders).
        if z0 < 0 or z1 <= z0 or (depth is not None and z1 > depth):
            self._set_export_status(
                f"Invalid Z range {z0}–{z1} for this patch"
                + (f" (depth {depth})." if depth is not None else "."),
                success=False)
            return
        if depth is not None and z0 == 0 and z1 == depth:
            self._set_export_status(
                "That range keeps the whole stack — nothing to trim.",
                success=False)
            return

        # `_trim_launching` up FIRST — BEFORE the confirm modal — so a second
        # click during the modal's event-loop spin can't re-enter + launch a
        # second worker (F1 MED-5).  The finally clears it on EVERY path.
        self._trim_launching = True
        try:
            # Brick 4a Part 2: trimming shifts every Z index, so a patch Step 2
            # has already curated slices from -- or Step 3 has reconstructed --
            # gets a different confirm.  INSIDE this try (the flag is up, and
            # its `finally` clears it on every path); the scan is wrapped so a
            # failure can never block a legitimate trim.
            use = self._downstream_for(patch_name)
            if use.any:
                if not self._confirm_downstream(patch_name, use, "trim"):
                    return
            else:
                from qtpy.QtWidgets import QMessageBox
                reply = QMessageBox.question(
                    self.widget, "Trim patch?",
                    f"Trim '{patch_name}' to Z {z0}–{z1}?\n\nThis overwrites the "
                    "patch and can't be undone.",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply != QMessageBox.Yes:
                    return

            # HIGH-3: detach any live intensity-plot worker reading THIS patch
            # (it was auto-launched on select) BEFORE the replace fires, so it
            # can't hold the TIFF open when `os.replace` overwrites it.
            self._detach_intensity_worker()

            self._trim_token += 1
            token = self._trim_token
            self._log("STEP1-TRIM",
                      f"launch token={token} patch={patch_name} z=[{z0},{z1}]")
            try:
                self._set_step1_busy(True)
                self._trim_running = True
                try:
                    self.widget._busy_show(f"Trimming {patch_name}…")
                    # Also scan the ④ (right-column) bar where the user is working,
                    # and paint it NOW so it doesn't lag behind the click while the
                    # worker spins up.
                    self._set_review_busy(True)
                    from qtpy.QtWidgets import QApplication
                    QApplication.processEvents()
                except Exception:
                    pass
                if self._trim_thread is not None:
                    self._draining_trim_threads.append(
                        (self._trim_thread, self._trim_worker))
                worker = _TrimWorker(patch_name, path, sidecar, z0, z1)
                thread = QThread()
                worker.moveToThread(thread)
                thread.started.connect(worker.run)
                worker.finished.connect(
                    lambda info, _t=token: self._on_trim_finished(info, _t))
                worker.error.connect(
                    lambda msg, _t=token: self._on_trim_error(msg, _t))
                worker.finished.connect(thread.quit)
                worker.error.connect(thread.quit)
                thread.finished.connect(
                    lambda _t=thread: self._cleanup_trim_thread(_t))
                self._trim_thread = thread
                self._trim_worker = worker
                thread.start()
            except Exception as exc:
                self._trim_thread = None
                self._trim_worker = None
                try:
                    self.widget._busy_hide()
                except Exception:
                    pass
                self._set_step1_busy(False)
                self._trim_running = False
                self._set_review_busy(False)   # launch failed → clear the ④ scan
                self._set_export_status(
                    f"Trim could not start: {exc}", success=False)
                self._log("STEP1-TRIM", f"launch FAILED token={token}: {exc}")
        finally:
            self._trim_launching = False

    def _on_trim_finished(self, info, token):
        """Terminal (success) — crop the in-memory stage BY NAME + refresh ④,
        then the SINGLE unlock.

        STALE guard first (a superseded token owns the surface); the unlock runs
        BEFORE the ④ plot-redraw so it isn't blocked by the busy gate on
        `_on_plot_intensity_clicked` (HIGH-3).  The in-memory crop runs even
        after step-exit (pure data update, no Qt) so a staged patch never lingers
        stale full-Z — otherwise a later trim off its stale profile would cut the
        WRONG slab (data-audit MED-1).  Only the ④/canvas REFRESH is inert after
        step-exit (disk already trimmed)."""
        if token != self._trim_token:
            return
        # Unlock FIRST so the FINISH redraw is ungated, then refresh.  (The ④
        # scan bar is cleared inside `_finish_trim_unlock`, covering error too.)
        self._finish_trim_unlock()
        # Crop the in-memory stage entry BY NAME (current-patch coords) + update
        # its selected_z_range coherently with the sidecar, so the ④ plot/thumb
        # never show stale full-Z pixels.  This mutates only the extracted_patches
        # dict (no widget/viewer touch), so it is SAFE after step-exit — and must
        # run there, or the staged copy diverges from the trimmed disk (MED-1).
        name = None
        z0 = z1 = 0
        abs_range = [0, 0]
        try:
            name = info["name"]
            z0, z1 = int(info["z0"]), int(info["z1"])
            abs_range = list(info["selected_z_range"])
            entry = self.extracted_patches.get(name)
            if entry is not None:
                data = entry.get("data")
                nd = getattr(data, "ndim", None)
                if nd == 4:
                    entry["data"] = data[:, z0:z1]
                elif nd == 3:
                    entry["data"] = data[z0:z1]
                entry["selected_z_range"] = abs_range
        except Exception as exc:      # noqa: BLE001
            self._log("STEP1-TRIM", f"in-memory crop failed: {exc}")
            name = None
        # The patch's pixels changed → drop its cached depth profile so the redraw
        # below (and any later re-select) recomputes the fresh curves, never stale.
        if name:
            self._intensity_curve_cache.pop(name, None)
        if getattr(self, "_left_step1", False):
            self._log("STEP1-TRIM", "finish after step-exit — ④ refresh inert")
            return
        if name is None:
            return
        try:
            # Redraw the intensity plot ONLY if the trimmed patch is still
            # selected; re-clamp the band/spins to the new depth `[0, z1 - z0]`.
            # Best-effort so a redraw hiccup never skips the status/refresh below.
            try:
                cur = self.widget.extracted_roi_list.currentItem()
                if cur is not None and cur.text() == name:
                    self._set_selected_z_range((0, z1 - z0))
                    self._on_plot_intensity_clicked()
            except Exception as exc:      # noqa: BLE001
                self._log("STEP1-TRIM", f"post-trim plot redraw failed: {exc}")
            # Rebuild the ④ location overlay if this patch's box is shown (its
            # Z-extent shrank); no-op otherwise.
            if name in getattr(self, "_visible_patch_locations", set()):
                try:
                    self._rebuild_loc_boxes()
                except Exception as exc:  # noqa: BLE001
                    self._log("STEP1-TRIM", f"loc rebuild failed: {exc}")
            # ④ Library row: new dims + thumbnail (the mtime-keyed sidecar-detail
            # + thumbnail caches self-invalidate on the rewrite).
            try:
                self.widget.library_box_list.refresh()
            except Exception:
                pass
            self._set_export_status(
                f"Trimmed '{name}' to Z {abs_range[0]}–{abs_range[1]}.",
                success=True)
            self._refresh_conditioning_status()
        except Exception as exc:      # noqa: BLE001 — never let a refresh raise
            self._log("STEP1-TRIM", f"finish refresh failed: {exc}")

    def _on_trim_error(self, msg, token):
        """Fatal worker error — red status + the SINGLE unlock.  The patch on
        disk is untouched (`crop_patch_in_place` rolled the TIFF back from
        `.bak` before re-raising)."""
        if token != self._trim_token:
            return
        try:
            self._set_export_status(f"Trim failed: {msg}", success=False)
            self._log("STEP1-TRIM", f"worker error token={token}: {msg}")
        finally:
            self._finish_trim_unlock()

    def _finish_trim_unlock(self):
        """The one place the trim run's lock is released — reached from
        finished + error (never twice: only the token-owner reaches it, and the
        two are mutually exclusive terminal signals)."""
        try:
            self.widget._busy_hide()
        except Exception:
            pass
        self._set_step1_busy(False)
        self._trim_running = False
        self._set_review_busy(False)   # clear the ④ scan on BOTH finish + error

    def _cleanup_trim_thread(self, thread=None):
        """Drop the finished trim (thread, worker) refs — identity-guarded,
        removes any matching draining pair, and `deleteLater`s the thread
        (mirrors :meth:`_cleanup_extract_thread`)."""
        draining = getattr(self, "_draining_trim_threads", None)
        if draining is not None and thread is not None:
            self._draining_trim_threads = [
                (t, w) for (t, w) in draining if t is not thread]
        if thread is None or thread is self._trim_thread:
            self._trim_thread = None
            self._trim_worker = None
        if thread is not None:
            try:
                thread.deleteLater()
            except Exception:
                pass

    def _set_export_status(self, message: str, success: bool = True) -> None:
        if hasattr(self.widget, "set_export_status"):
            self.widget.set_export_status(message, success)

    # ------------------------------------------------------------------
    # ② preprocessing PREVIEW (Brick 2b) — test-crop compute + before/after plot
    # ------------------------------------------------------------------

    def _set_preview_status(self, message: str, ok: bool = True,
                            advisories=None) -> None:
        """Route a preprocessing status to ②'s status label (themed, not a
        traceback).  *advisories* are per-run caveats rendered as their own
        amber "⚠ " lines UNDER a still-green success line — a preview that
        rendered succeeded, even when there is something to warn about.
        Best-effort — silently no-ops on a bare widget double, and falls back to
        the two-argument call on a widget that predates advisories."""
        setter = getattr(self.widget, "set_preview_status", None)
        if not callable(setter):
            return
        if advisories:
            try:
                setter(message, ok, advisories)
                return
            except TypeError:
                pass
        setter(message, ok)

    def _preview_spacing(self):
        """The current effective voxel spacing (z, y, x) from the voxel spins —
        the SAME value an extract would bake with.  Falls back to (1, 1, 1).

        The whole read (INCLUDING ``self.widget``) is inside the guard so a bare
        controller double without a widget also gets the documented ``(1, 1, 1)``
        — this is now read by the physical-scale display paths, not only the
        preview compute, so it must never raise."""
        try:
            w = self.widget
            return (float(w.voxel_z_spin.value()),
                    float(w.voxel_y_spin.value()),
                    float(w.voxel_x_spin.value()))
        except Exception:
            return (1.0, 1.0, 1.0)

    def _gather_preview_inputs(self):
        """Resolve the loaded source's per-channel test crops for a PREVIEW.

        Returns ``(names, channels, bounds_list, spacing, recipe_json)`` or
        ``None`` when there is nothing to preview (no source loaded, or no usable
        3D channel).  The channels are array-likes (numpy or lazy) + a per-channel
        center-crop bounds dict; the worker materializes only the ROI.  The recipe
        is SNAPSHOTTED here (json) on the main thread so a later edit can't mutate
        the in-flight run.  Channels are matched to the recipe columns BY NAME
        when possible (so channel-0's recipe lands on channel-0's data), else the
        source's own channel order.
        """
        display = getattr(self, "_source_display_channels", None)
        if not display:
            return None
        by_name = {}
        for entry in display:
            try:
                nm, data = entry
            except (TypeError, ValueError):
                continue
            by_name[nm] = data

        recipe_names = list(getattr(self.widget, "_recipe_channel_names", []) or [])
        if recipe_names and all(nm in by_name for nm in recipe_names):
            names = recipe_names
            datas = [by_name[nm] for nm in recipe_names]
        else:
            names, datas = [], []
            for entry in display:
                try:
                    nm, data = entry
                except (TypeError, ValueError):
                    continue
                names.append(nm)
                datas.append(data)

        # WHERE the preview computes: the placed preview REGION box if one exists
        # (Preview v2 Piece 2), else the legacy fixed centre crop.  The box bounds
        # are read from the SAME axis-resolved per-channel 3D ZYX handle we crop
        # from — never raw `_source_lazy` (un-transposed on-disk order → slices C
        # with Z bounds on a ZCYX source = silent wrong voxels; data-grill HIGH-1).
        box = getattr(self, "_preview_box", None)
        channels, bounds_list = [], []
        for data in datas:
            # A multiscale channel is a list/tuple of pyramid levels — the finest
            # (level 0) is the full-res grid the recipe would bake against.
            arr = data[0] if isinstance(data, (list, tuple)) else data
            if arr is None or getattr(arr, "ndim", None) != 3:
                return None
            channels.append(arr)
            # `arr.shape` is the channel's 3D ZYX SPATIAL shape (the channel axis
            # was already resolved into `_source_display_channels`), so it is the
            # correct 3-tuple for `bounds_dict`.
            shape_zyx = tuple(int(s) for s in arr.shape)
            if box is not None:
                bounds_list.append(dict(box.bounds_dict(shape_zyx)))
            else:
                bounds_list.append(_preview_center_bounds(shape_zyx))
        if not channels:
            return None

        # Compute-budget guard: a full-Z box on a deep stack would materialize a
        # multi-GB float32 cube — stride the READ crop down to fit instead (never
        # OOM).  The bounds' extents feed `bounds_dict`-driven layer geometry
        # unchanged, so a strided crop still spans the same world box (its layer
        # scale = extent / strided-shape).
        n_ch = len(channels)
        applied_stride = 1
        for bounds in bounds_list:
            stride = _preview_compute_stride(bounds, n_ch)
            if stride > 1:
                bounds["stride_z"] = stride
                bounds["stride_y"] = stride
                bounds["stride_x"] = stride
                applied_stride = max(applied_stride, stride)

        recipe_json = self.widget.get_recipe_json()
        spacing = self._preview_spacing()
        if applied_stride > 1:
            self._log("STEP1-PREVIEW",
                      "region large — downsampled for preview (compute budget)")
            # The strided READ has effective voxel spacing = spacing × stride, so
            # a geometry (isotropic) op must see the SCALED spacing or it previews
            # the wrong resample ratio (correctness audit MED-1).  Stride is
            # uniform across Z/Y/X, so spacing scales uniformly.
            try:
                spacing = tuple(float(v) * applied_stride for v in spacing)
            except (TypeError, ValueError):
                pass
        return names, channels, bounds_list, spacing, recipe_json

    def on_preview_recipe(self):
        """② Preview button → compute the recipe on a test crop, draw the
        before/after histogram (Brick 2b).  Delegated from the widget's
        `_on_preview_clicked`.  Runs the crop-read + `execute` on a QThread so
        the GUI stays live; ② is busy-locked meanwhile."""
        if getattr(self, "_step1_busy", False):
            return  # one Step-1 op at a time
        # Each Preview REPLACES the last one: drop the previous ⟨preview⟩ render
        # up front so a stale processed crop never lingers while the new run
        # computes (or if there is now nothing to preview).
        self._clear_preview_layers()
        inputs = self._gather_preview_inputs()
        if inputs is None:
            self._set_preview_status(
                "Load a source with channels first, then Preview.", ok=False)
            return
        names, channels, bounds_list, spacing, recipe_json = inputs
        self._preview_names = names
        # Retain the crop bounds so the main-thread finish handler can position
        # each rendered ⟨preview⟩ layer where its center crop was taken.
        self._preview_bounds = list(bounds_list)
        # Per-channel LUTs for the histogram entries — resolved HERE so the
        # worker can build the entries off-thread.  Keyed off `_SOURCE_DISPLAY_CMAPS`
        # (the SAME index→canvas-colour mapping the source layers use), so each
        # before/after curve matches its channel's CANVAS colour instead of a
        # parallel palette.
        luts = tuple(_channel_display_hex(ci) for ci in range(len(names)))

        # A fresh token stamps THIS run; a later edit / source-switch / re-launch
        # bumps it so a superseded run's finished/error handler early-returns.
        # Route through `_bump_preview_token` — it ALSO asks the run being
        # superseded to stop (C#12).  This is the most common supersede of all
        # (clicking Preview again) and the inline `+= 1` it replaces skipped the
        # cancel entirely, so the previous compute ran to completion on a result
        # the token then discarded.  Must stay ABOVE the `_preview_worker`
        # reassignment below, or it would cancel the run it just launched.
        self._bump_preview_token()
        token = self._preview_token
        self._resolve_preview_stream_warning(recipe_json, channels, spacing)
        # C#3 / C#4: name any op this region cannot show faithfully, BEFORE the
        # user reads the render as the truth.  Resolved from the SAME crop
        # bounds the worker is about to use.
        crop_shape = None
        try:
            b = bounds_list[0] if bounds_list else None
            if b:
                crop_shape = (int(b["z1"]) - int(b["z0"]),
                              int(b["y1"]) - int(b["y0"]),
                              int(b["x1"]) - int(b["x0"]))
        except Exception:
            crop_shape = None
        try:
            full_shape = self._resolve_preview_full_shape()
        except Exception:
            full_shape = None
        self._preview_fidelity = _preview_fidelity_notes(
            recipe_json, len(channels), crop_shape, full_shape)

        self._set_step1_busy(True)
        try:
            self.widget._busy_show("Preview — test region…")
        except Exception:
            pass

        try:
            # An older preview may still be finishing — keep it referenced so its
            # QThread is not GC'd mid-flight; its result is discarded by the token.
            if self._preview_thread is not None:
                self._draining_preview_threads.append(
                    (self._preview_thread, self._preview_worker))
            worker = _PreprocPreviewWorker(
                recipe_json, channels, bounds_list, spacing, names, luts,
                norm_bounds=self._cached_norm_bounds(names))
            thread = QThread()
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.finished.connect(
                lambda entries, afters, hist_ms, _t=token, _n=names:
                self._on_preview_computed(entries, afters, _n, _t, hist_ms))
            worker.error.connect(
                lambda msg, _t=token: self._on_preview_error(msg, _t))
            worker.finished.connect(thread.quit)
            worker.error.connect(thread.quit)
            thread.finished.connect(
                lambda _t=thread: self._cleanup_preview_thread(_t))
            self._preview_thread = thread
            self._preview_worker = worker
            self._log("STEP1-PREVIEW",
                      f"launch token={token} channels={len(channels)}")
            thread.start()
        except Exception as exc:
            # Neither finished nor error will fire — clear the lock + bar HERE so
            # the UI can never stay stuck (the source-read launch-guard pattern).
            self._preview_thread = None
            self._preview_worker = None
            try:
                self.widget._busy_hide()
            except Exception:
                pass
            self._set_step1_busy(False)
            self._set_preview_status(f"Preview could not start: {exc}", ok=False)
            self._log("STEP1-PREVIEW", f"launch FAILED token={token}: {exc}")

    def _resolve_preview_stream_warning(self, recipe_json, channels, spacing):
        """Decide, at Preview launch, whether "Apply → whole volume" will REFUSE
        this recipe on this source — and stash the sentence that says so (C#8).

        A prepared (lazy) source applies through the streaming engine, which
        refuses a CLAHE / bleach / resize remainder outright; Preview rendered
        exactly such a recipe happily and said nothing, so the user only found
        out at Apply.  The check calls the SAME `_recipe_has_stream_remainder`
        predicate as the apply gate, with the same spacing, so the two can never
        drift apart.  A resident in-memory source has no such restriction — a
        warning there would be a lie, so it is gated on `_should_stream_process`.
        Best-effort: any failure resolving it means no warning (the apply gate is
        still the authority that refuses)."""
        self._preview_stream_warning = None
        try:
            if not self._should_stream_process(channels):
                return
            if not self._recipe_has_stream_remainder(
                    recipe_json, len(channels), spacing=spacing):
                return
            self._preview_stream_warning = (
                "Heads-up: this image is stored in the large-data format, and "
                "the whole-volume run can't do contrast (CLAHE), "
                "depth-brightness or resize on that format.  Preview still "
                "shows the region correctly; to apply it to everything, load "
                "the image the normal way.")
        except Exception as exc:  # noqa: BLE001
            self._preview_stream_warning = None
            self._log("STEP1-PREVIEW", f"stream-warning check skipped: {exc}")

    def _cached_norm_bounds(self, names):
        """The per-channel whole-volume normalize window already MEASURED by a
        "Process whole volume" run on THIS source, or ``None``.

        Reuse over recompute (C#2): the window depends only on the raw channel
        data, so an apply run on the same source has already paid for it — and
        measuring it costs a percentile pass the preview would otherwise repeat.

        Keyed on the ordered channel NAME list, not just the count: a window is
        per-CHANNEL, so two sources with two channels each would pass a count
        check while the cached window belongs to the wrong channels — the
        preview would then normalise against another image's percentiles, which
        is the very defect C#2 exists to close.  A cache written before names
        were recorded is not trusted.  ``None`` lets the worker measure the
        window on ITS thread; never on the UI thread."""
        cache = getattr(self, "_processed_cache", None) or {}
        bounds = cache.get("bounds")
        cached_names = cache.get("names")
        want = [str(n) for n in (names or [])]
        if not bounds or not cached_names or not want:
            return None
        if [str(n) for n in cached_names] != want:
            return None
        if len(bounds) != len(want):
            return None
        try:
            if cache.get("source_id") != self._current_source_id():
                return None
        except Exception:
            return None
        return list(bounds)

    def _on_preview_computed(self, entries, afters, names, token, hist_ms=0.0):
        """Main-thread PREVIEW finish: draw the pre-computed histograms + add the
        ⟨preview⟩ layers.  The percentile / histogram COMPUTE already ran in the
        worker (`entries` arrive ready) — the only main-thread cost left is the
        pyqtgraph DRAW (~ms) and the napari layer-add / GPU upload (the
        unavoidable render tail, which must stay on the UI thread).

        Discards a STALE run (the recipe was edited / the source switched since
        launch — the Step-3 QC stale-finish guard).  Keeps the busy bar spinning
        + ② busy-locked THROUGH the render tail so the residual freeze reads as
        "rendering", not a hang (released in the finally)."""
        if token != getattr(self, "_preview_token", 0):
            self._log("STEP1-PREVIEW", f"discard stale result token={token}")
            # C#1: this run took a `_set_step1_busy(True)` in `on_preview_recipe`
            # and the ONLY release is the `finally` below — which this early
            # `return` used to skip.  A superseded finish (step exit / recipe
            # edit mid-Preview) then pinned `_busy_depth` at 1 for the rest of
            # the session: ②/③/④ dead with `_step1_busy` reading "idle".
            # Every terminal path must release exactly once.
            self._release_preview_busy()
            return
        # Keep the surface locked + name the render phase on the shared bar —
        # the layer-add below is the last main-thread work of this Preview.
        try:
            self.widget._busy_show("Rendering preview…")
        except Exception:
            pass
        try:
            try:
                _t0 = time.perf_counter()
                self.widget.set_recipe_histograms(entries)
                draw_ms = (time.perf_counter() - _t0) * 1e3
            except Exception as exc:
                self._set_preview_status(f"Preview draw failed: {exc}", ok=False)
                self._log("STEP1-PREVIEW", f"draw FAILED token={token}: {exc}")
                return

            # WYSIWYG: add the PROCESSED crop as toggleable ⟨preview⟩ layer(s) so
            # the user SEES the pixels (spatial ops barely move the histogram).
            # Best-effort — a render failure must not sink the (drawn) histogram.
            n_layers = 0
            layer_ms = 0.0
            try:
                _t1 = time.perf_counter()
                n_layers = self._add_preview_layers(afters, names)
                layer_ms = (time.perf_counter() - _t1) * 1e3
            except Exception as exc:
                # The failure is swallowed on purpose (a drawn histogram must
                # survive a render failure), but the C#6 auto-hide has already
                # run for the channels that DID render — leaving the user with
                # an invisible source channel, no ⟨preview⟩ layer and nothing
                # saying why.  Put back whatever we hid.
                self._restore_layers_hidden_under_preview()
                self._log("STEP1-PREVIEW",
                          f"preview layer render FAILED token={token}: {exc}")

            # Perf marker (like the codebase's [SW-PERF]/[QC-PERF]): the worker
            # already reports its histogram-build cost; log the main-thread tail.
            self._log(
                "PREVIEW-PERF",
                f"hist_entries(worker)={float(hist_ms):.1f}ms  "
                f"draw={draw_ms:.1f}ms  layer_add={layer_ms:.1f}ms  "
                f"(n_layers={n_layers})")

            # the crop is a TEST region — CLAHE / bleach / geometry differ
            # crop-vs-whole, so say so rather than imply the whole volume.
            # ONE terminal status call.  The success line stays GREEN — a
            # preview that rendered and drew its histogram succeeded — and the
            # caveats ride their own amber lines beneath it:
            #   * C#3 / C#4, an op this region cannot show faithfully, named
            #     with its axis rather than quietly clamped;
            #   * C#8, the whole-volume run will refuse this recipe on this
            #     source, said here rather than three minutes into an Apply.
            # `_PREVIEW_EDGE_CAVEAT` covers what those notes do NOT: the
            # region's edges (halo) and a resize step.
            advisories = list(getattr(self, "_preview_fidelity", None) or [])
            warn = getattr(self, "_preview_stream_warning", None)
            if warn:
                advisories.append(warn)
            if n_layers:
                self._set_preview_status(
                    f"Preview ready — {n_layers} ⟨preview⟩ layer(s) added"
                    f"{self._hidden_layer_phrase()}.  Histogram below.  "
                    f"{_PREVIEW_EDGE_CAVEAT}",
                    ok=True, advisories=advisories)
            else:
                self._set_preview_status(
                    f"Preview ready — histogram below.  {_PREVIEW_EDGE_CAVEAT}",
                    ok=True, advisories=advisories)
        finally:
            # Release the lock + rest the bar only AFTER the render tail — one
            # terminal funnel so a draw/render failure cannot leave ② stuck.
            self._release_preview_busy()

    def _release_preview_busy(self):
        """Drop the refcount + rest the busy bar for ONE finished Preview run.

        The single release funnel for every terminal path of a Preview —
        live finish, live error, and (C#1) a STALE finish/error.  Each of
        those paths must call it exactly once.

        The bar belongs to the OUTERMOST op, so it is rested only when this
        release actually brought the refcount to 0.  Hiding it unconditionally
        let a superseded run rest the bar under a second Preview that was still
        computing — the surface stayed (correctly) locked with nothing on screen
        saying why."""
        self._set_step1_busy(False)
        if getattr(self, "_busy_depth", 0) > 0:
            return                      # an outer op still owns the bar
        try:
            self.widget._busy_hide()
        except Exception:
            pass

    def _on_preview_error(self, msg, token):
        """Main-thread PREVIEW error: surface a themed status, clear the lock.
        Discards a stale run's error too (a superseded worker's terminal emit)
        — releasing the busy refcount on that path too (C#1)."""
        if token != getattr(self, "_preview_token", 0):
            self._release_preview_busy()
            return
        self._release_preview_busy()
        if str(msg) != "cancelled":
            self._set_preview_status(f"Preview failed: {msg}", ok=False)
        self._log("STEP1-PREVIEW", f"error token={token}: {msg}")

    def _cleanup_preview_thread(self, thread):
        """Drop the finished preview (thread, worker) pair once its QThread has
        actually stopped (`thread.finished`).  Idempotent; touches only
        controller thread state."""
        if thread is getattr(self, "_preview_thread", None):
            self._preview_thread = None
            self._preview_worker = None
        self._draining_preview_threads = [
            (t, w) for (t, w) in getattr(self, "_draining_preview_threads", [])
            if t is not thread
        ]

    # ------------------------------------------------------------------
    # ② preprocessing PROCESS WHOLE VOLUME (Brick 1b-1) — the recipe over the
    #    full per-channel volumes on a QThread → uint16 in-memory cache + a
    #    WYSIWYG ⟨processed⟩ render.  NO on-disk persistence (a later brick).
    # ------------------------------------------------------------------

    def _gather_process_inputs(self):
        """Resolve the loaded source's per-channel WHOLE-VOLUME arrays for a
        "Process whole volume" run (Brick 1b-1).

        Mirrors :meth:`_gather_preview_inputs` but takes the FULL level-0 arrays
        — NO center-crop, NO region box.  Returns ``(names, channels, spacing,
        recipe_json)`` or ``None`` when there is nothing to process (no source
        loaded, or no usable 3D channel).  The channels are array-likes (numpy or
        lazy) the worker materializes OFF the main thread; matched to the recipe
        columns BY NAME when possible.  The recipe + spacing are SNAPSHOTTED here
        on the main thread so a later edit can't mutate the in-flight run.
        """
        display = getattr(self, "_source_display_channels", None)
        if not display:
            return None
        by_name = {}
        for entry in display:
            try:
                nm, data = entry
            except (TypeError, ValueError):
                continue
            by_name[nm] = data

        recipe_names = list(getattr(self.widget, "_recipe_channel_names", []) or [])
        if recipe_names and all(nm in by_name for nm in recipe_names):
            names = recipe_names
            datas = [by_name[nm] for nm in recipe_names]
        else:
            names, datas = [], []
            for entry in display:
                try:
                    nm, data = entry
                except (TypeError, ValueError):
                    continue
                names.append(nm)
                datas.append(data)

        channels = []
        for data in datas:
            # A multiscale channel is a list/tuple of pyramid levels — level 0 is
            # the full-res grid the recipe bakes against (same handle the extract
            # / preview crop from; never raw `_source_lazy`).
            arr = data[0] if isinstance(data, (list, tuple)) else data
            if arr is None or getattr(arr, "ndim", None) != 3:
                return None
            channels.append(arr)
        if not channels:
            return None

        recipe_json = self.widget.get_recipe_json()
        spacing = self._preview_spacing()
        return names, channels, spacing, recipe_json

    def _estimate_process_peak_bytes(self, channels):
        """Conservative peak-RAM estimate (bytes) for processing *channels* whole.

        Sum over channels of ``prod(shape) × 4`` (the float32 working copy) × ~4
        for the materialized inputs + processed floats + per-op scratch + the
        uint16 bake held together across the render hand-off.  Deliberately an
        OVER-estimate so the too-large guard fails SAFE.  Returns ``0`` when a
        channel cannot be sized (→ the guard skips rather than false-refuses).
        """
        total = 0
        for arr in channels:
            shp = getattr(arr, "shape", None)
            if not shp:
                return 0
            try:
                total += int(math.prod(int(s) for s in shp)) * 4
            except (TypeError, ValueError, OverflowError):
                return 0
        return total * 4

    def _derive_block_workers(self, recipe_json):
        """Block-pool width for the parallel process, backed off under RAM pressure
        (the resource-adaptive core count).

        Returns ``None`` (the engine default ``min(cpu, 32)``) when psutil is
        unavailable or the budget comfortably fits; otherwise a smaller clamp so a
        RAM-tight machine runs with FEWER cores instead of refusing the whole run.
        Never raises."""
        cpu = os.cpu_count() or 1
        default = min(cpu, 32)
        try:
            import psutil
            avail = float(psutil.virtual_memory().available)
            tile = int(_pp_max_tile_scratch_bytes(recipe_json))
        except Exception:
            return None
        if tile <= 0 or avail <= 0:
            return None
        # Reserve most of RAM for the materialized channels + output; let the block
        # pool use a slice of what's left.  Fewer cores beats refusing on a small box.
        fit = int((0.25 * avail) // tile)
        return max(1, min(default, fit))

    def on_process_or_cancel_clicked(self):
        """The "Process whole volume" button doubles as Cancel while a process is
        in flight (mirrors :meth:`on_prepare_or_cancel_clicked`).  Dispatches on
        ``_processing`` so a click DURING a run cancels it, otherwise starts one."""
        if getattr(self, "_processing", False):
            self.cancel_process_whole_volume()
        else:
            self.on_process_whole_volume()

    def on_process_whole_volume(self):
        """"Process whole volume" → run the recipe over the FULL per-channel
        volumes on a QThread → fixed-scale uint16 (Brick 1b-1).

        REGIME ROUTER: a LAZY / large-data source routes to the STREAMING
        (out-of-core) path (:meth:`_process_whole_volume_streaming`, Phase 2a); a
        resident (whole-loaded numpy) source runs this in-memory path.  In memory:
        holds the result in an in-memory cache (`_processed_cache`) and DISPLAYS it
        as ⟨processed⟩ layers (WYSIWYG); NO on-disk persistence.  Refuses up-front
        if the estimated peak RAM exceeds the eager budget (for a large source,
        Prepare-for-large-data first so it streams).  The heavy READ + `execute` +
        `to_uint16` all run on the worker; the surface is busy-locked
        and the button flips to Cancel meanwhile."""
        if getattr(self, "_step1_busy", False):
            return  # one Step-1 op at a time
        # Refuse a new run while a PREVIOUS (cancelled) process is still draining
        # (grill lifecycle MED-1): a cancelled worker keeps materializing + holding
        # its float32 working set until it finishes, so launching now would run two
        # whole-volume pipelines concurrently → ~2× peak, past the RAM budget the
        # per-run guard enforces.  `_process_thread` stays set until its
        # `thread.finished` fires `_cleanup_process_thread`.
        if (getattr(self, "_process_thread", None) is not None
                or getattr(self, "_draining_process_threads", None)
                or getattr(self, "_process_store_thread", None) is not None
                or getattr(self, "_draining_process_store_threads", None)):
            self._set_preview_status(
                "Still finishing the previous processing — try again in a moment.",
                ok=False)
            return
        inputs = self._gather_process_inputs()
        if inputs is None:
            self._set_preview_status(
                "Load a source with channels first, then Process whole volume.",
                ok=False)
            return
        names, channels, spacing, recipe_json = inputs

        # REGIME ROUTER ): a LAZY / large-data source (a
        # prepared multiscale store — dask/zarr-backed handles) is processed
        # OUT-OF-CORE (streaming, block-by-block into a `__proc__` store) so it is
        # memory-bounded + freeze-free, regardless of whether it *would* fit RAM.
        # A resident (whole-loaded numpy) source stays on the fast in-memory path
        # below, with its peak-vs-budget refuse-guard kept as the safety net.
        if self._should_stream_process(channels):
            self._process_whole_volume_streaming(
                names, channels, spacing, recipe_json)
            return

        # Parallel is OFF by default (the proven serial path); a real-scale test
        # turns it on via EPICYCLE_PREPROC_PARALLEL (the rollout gate).  When on,
        # the LOCAL ops (gaussian, background) run block-parallel across all cores,
        # byte-identical to serial; `block_workers` backs the core count off under
        # RAM pressure (run FEWER cores, don't refuse — the resource-adaptive knob).
        parallel = _preproc_parallel_enabled()
        block_workers = self._derive_block_workers(recipe_json) if parallel else None

        # Too-large guard: estimate the in-memory peak vs the eager RAM budget and
        # REFUSE rather than OOM.  psutil is the RAM oracle; if it is missing we
        # skip the check (do NOT block) — same fail-open stance as the load route.
        # Scale the (input-sized) estimate by the recipe's geometry UPSAMPLE factor
        # (grill correctness MED-1): an isotropic upsample (anisotropic→isotropic,
        # e.g. 5× Z) grows execute's intermediate + output arrays F×, so a volume
        # that fits at 1× would OOM — the guard must see the resampled peak.
        peak = self._estimate_process_peak_bytes(channels)
        if peak > 0:
            peak = int(peak * _recipe_geometry_volume_factor(recipe_json, spacing))
            if parallel:
                # The block pool holds ≤ workers tile-scratch buffers concurrently
                # (measured ≈3× per tile) ON TOP of the whole-volume
                # arrays — count it so the refuse guard is honest for the fast path.
                workers = int(block_workers or min(os.cpu_count() or 1, 32))
                try:
                    peak += workers * int(_pp_max_tile_scratch_bytes(recipe_json))
                except Exception:
                    pass
        try:
            import psutil
            avail = float(psutil.virtual_memory().available)
            budget = EAGER_RAM_FRACTION * avail
            if peak > 0 and budget > 0 and peak > budget:
                self._set_preview_status(
                    "This volume is too large to process in memory here — click "
                    "\"Prepare for large data\" first, then Process whole volume "
                    "(it will stream, memory-bounded) "
                    f"(needs ~{format_bytes(peak)}, budget ~{format_bytes(budget)}).",
                    ok=False)
                self._log("STEP1-PROCESS",
                          f"refused: peak≈{peak} > budget≈{int(budget)}")
                return
        except ImportError:
            pass          # psutil unavailable → skip the RAM check, don't block
        except Exception as exc:
            self._log("STEP1-PROCESS", f"RAM check skipped: {exc}")

        # A new run REPLACES the last: drop the previous ⟨processed⟩ render (BOTH
        # the in-memory + the streamed-store render) up front so a stale volume
        # never lingers while the new run computes.
        self._clear_processed_layers()
        self._clear_processed_store_layers()
        self._process_names = names

        # A fresh DEDICATED token stamps THIS run (never reuses `_preview_token`
        # — the two ops must not cross-cancel); a Cancel / source-switch bumps it
        # so a superseded run's finished/error handler early-returns.
        self._process_token += 1
        token = self._process_token
        self._processing = True

        self._set_step1_busy(True)
        try:
            self.widget._busy_show("Processing whole volume…")
        except Exception:
            pass
        # Flip the button to Cancel (kept clickable) AFTER the lock so it stays
        # the escape hatch — the exact sequence the Prepare button uses.
        self._set_process_button_cancel(True)

        try:
            # An older process may still be finishing — keep it referenced so its
            # QThread isn't GC'd mid-flight; its result is discarded by the token.
            if self._process_thread is not None:
                self._draining_process_threads.append(
                    (self._process_thread, self._process_worker))
            worker = _PreprocProcessWorker(
                recipe_json, channels, spacing,
                parallel=parallel, block_workers=block_workers, names=names)
            thread = QThread()
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.finished.connect(
                lambda processed, new_spacing, bounds, stamp, _t=token,
                _n=names, _r=recipe_json:
                self._on_process_computed(
                    processed, new_spacing, bounds, stamp, _n, _r, _t))
            worker.error.connect(
                lambda msg, _t=token: self._on_process_error(msg, _t))
            # Token BOUND IN THE LAMBDA (Brick 5 fix 2): `_processing` alone
            # is not enough — a superseded run flips it back ON, so a draining
            # worker's late tick would drive the NEW run's bar.
            worker.progress.connect(
                lambda frac, label, indet, _t=token:
                self._on_process_progress(frac, label, indet, _t))
            worker.finished.connect(thread.quit)
            worker.error.connect(thread.quit)
            thread.finished.connect(
                lambda _t=thread: self._cleanup_process_thread(_t))
            self._process_thread = thread
            self._process_worker = worker
            self._log(
                "STEP1-PROCESS",
                f"launch token={token} channels={len(channels)} "
                f"parallel={parallel} block_workers={block_workers}")
            thread.start()
        except Exception as exc:
            # Neither finished nor error will fire — clear the lock + bar + button
            # HERE so the UI can never stay stuck (the launch-guard pattern).
            self._process_thread = None
            self._process_worker = None
            try:
                self.widget._busy_hide()
            except Exception:
                pass
            self._set_step1_busy(False)
            self._processing = False
            self._set_process_button_cancel(False)
            self._set_preview_status(
                f"Process could not start: {exc}", ok=False)
            self._log("STEP1-PROCESS", f"launch FAILED token={token}: {exc}")

    # ------------------------------------------------------------------
    # STREAMING (out-of-core) whole-volume process (Phase 2a)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_lazy_handle(arr):
        """True iff *arr* is a LAZY handle (dask / zarr-backed), NOT a resident
        ``np.ndarray``.  A ``np.memmap`` is an ``ndarray`` subclass → NOT
        lazy (a raw >RAM memmap routes through Prepare first)."""
        return not isinstance(arr, np.ndarray)

    def _should_stream_process(self, channels):
        """Regime oracle : stream iff the per-channel process
        handles are LAZY (a prepared multiscale / dask-zarr source — the
        large-data regime), regardless of whether the source *would* fit RAM.

        A resident whole-loaded numpy source is already in RAM → the fast
        in-memory path (its peak-vs-budget refuse-guard stays the safety net).  A
        lazy source → streaming (faster + freeze-free + memory-bounded).  ``any``
        (not ``all``): a single lazy channel is enough to prefer streaming (the
        engine reads numpy channels block-by-block too, so a mixed set is safe)."""
        return bool(channels) and any(
            self._is_lazy_handle(ch) for ch in channels)

    def _recipe_has_stream_remainder(self, recipe_json, n_channels, spacing=None):
        """True iff *recipe_json* has an op that does NOT stream in Phase 2a — a
        CLAHE / bleach (global-stat / adaptive) or geometry (resize) op that stops
        the engine's leading fused run (a non-empty ``_fused_plan`` remainder).

        The pre-launch gate: the engine would otherwise process only the leading
        fused run and SILENTLY drop the remainder (a wrong result), so a recipe
        with any remainder is refused up front instead.  Fails SAFE — on any
        error resolving the plan it returns ``True`` (refuse) rather than risk a
        silent drop.

        *spacing* is the source ``(z, y, x)`` voxel size; it lets the engine
        recognise an IDENTITY isotropic step (the seeded default) as the no-op it
        is, so a default recipe is not refused as a "resize".  It MUST be
        the same spacing passed to the engine, or the gate and the run disagree."""
        try:
            norm = _pp_recipe.normalize_recipe(recipe_json, int(n_channels))
            for ci in range(int(n_channels)):
                flat = _pp_recipe.flatten_for_channel(norm, ci)
                _fused, _halo, remainder = _pp_fused_plan(flat, spacing=spacing)
                if remainder:
                    return True
            return False
        except Exception as exc:  # noqa: BLE001 — refuse-on-uncertainty (safe)
            self._log("STEP1-PROCESS", f"remainder check failed (refusing): {exc}")
            return True

    def _processed_recipe_hash(self, recipe_json, n_channels):
        """A stable hex digest of the normalized recipe — the ``__proc__`` store
        key (matches the ``epicycle_processed.recipe_hash`` the engine writes)."""
        try:
            norm = _pp_recipe.normalize_recipe(recipe_json, int(n_channels))
            return _pp_recipe_hash(norm)
        except Exception:
            return _pp_recipe_hash(recipe_json)

    def _process_source_fingerprint(self, src_path):
        """The ``{path, source_mtime, source_size}`` provenance embedded in the
        ``__proc__`` store's ``epicycle_processed.source`` (freshness key).  Best
        effort — a missing / unreadable source omits the mtime/size."""
        fp = {"path": src_path}
        try:
            if src_path:
                fp["source_mtime"] = os.path.getmtime(src_path)
                fp["source_size"] = os.path.getsize(src_path)
        except OSError:
            pass
        return fp

    def _resolve_processed_store_path(self, workspace, display_name, src_path):
        """The CLEAN, deterministic ⟨processed⟩ store path for this source:

        ``<ws>/artifacts/step1/largedata/<display_name>__<source_id>_processed.zarr``

        ONE store per SOURCE, keyed by ``source_id`` (B2 collision guard, 2026-08-20).
        Two sources that share a filename STEM (both with a blank display name →
        the same ``display_name``) would otherwise map to the SAME
        ``<stem>_processed.zarr`` path, and the second Process's streaming commit
        (``.tmp`` → ``rmtree`` old → ``rename``, streaming.py) would DESTROY the
        first source's store — silent DATA LOSS.  ``source_id`` is stable per
        physical file, so re-processing the SAME source (any recipe) resolves the
        SAME name and the writer overwrites it in place — one store per source,
        no leak.  Identity + freshness live INSIDE the store's
        ``epicycle_processed`` attrs (``recipe_hash`` + source ``{mtime,size}``
        fingerprint), read by :meth:`_fresh_processed_store_for` — NOT the
        filename.  The persisted ``processed_store`` pointer is
        authoritative on reopen (it survives a rename); this deterministic name is
        the fallback.

        Returns ``(store_path, source_id)``."""
        largedata_dir = self._largedata_dir(workspace)
        try:
            state = load_state(workspace)
            source_id = self._resolve_source_id_for(state, src_path)
        except Exception as exc:      # noqa: BLE001
            # State unreadable → we can't resolve the stable id.  Derive a STABLE
            # per-source-PATH fallback (a short path hash, NOT a shared "src") so
            # two distinct same-stem sources still get DISTINCT store names even in
            # this degraded case — the last collision window (audit LOW).
            import hashlib
            h = hashlib.sha1(
                _norm_path(src_path or "").encode("utf-8", "replace")
            ).hexdigest()[:8]
            source_id = f"src_{h}"
            self._log("STEP1-PROCESS",
                      f"store-id fallback (state unreadable: {exc}) -> {source_id}")
        fname = f"{display_name}__{source_id}_processed.zarr"
        return os.path.join(largedata_dir, fname), source_id

    def _read_processed_attrs(self, store_path):
        """The ``epicycle_processed`` attrs block for a ``__proc__`` store, or
        ``None`` (a light attrs-only read; never raises)."""
        try:
            import zarr
            from packages.epicycle_largedata.multiscale import _zarr_store
            store = _zarr_store(str(store_path), "r")
            try:
                group = zarr.open_group(store=store, mode="r")
                attrs = group.attrs.get("epicycle_processed")
            finally:
                if hasattr(store, "close"):
                    try:
                        store.close()
                    except Exception:
                        pass
            return attrs if isinstance(attrs, dict) else None
        except Exception:
            return None

    def _fresh_processed_store_for(self, store_path, src_path, recipe_hash):
        """*store_path* if it is an EXISTING FRESH ``__proc__`` store for this
        recipe + source, else ``None`` (fresh-skip).

        Fresh = the store dir exists AND carries an ``epicycle_processed`` block
        whose ``recipe_hash`` matches AND whose ``source`` fingerprint (mtime +
        size, when present) equals the source's CURRENT values.  Any drift (edited
        source, different recipe, missing/corrupt block) → ``None`` so the caller
        re-streams.  Never raises."""
        try:
            if not os.path.isdir(store_path):
                return None
            attrs = self._read_processed_attrs(store_path)
            if not isinstance(attrs, dict):
                return None
            if str(attrs.get("recipe_hash")) != str(recipe_hash):
                return None
            fp = attrs.get("source") or {}
            if src_path and isinstance(fp, dict):
                pmtime = fp.get("source_mtime")
                psize = fp.get("source_size")
                if pmtime is not None:
                    if abs(float(pmtime) - os.path.getmtime(src_path)) > 1e-6:
                        return None
                if psize is not None:
                    if int(psize) != int(os.path.getsize(src_path)):
                        return None
            return store_path
        except Exception:
            return None

    def _delete_superseded_processed_stores(self, keep_store_path):
        """Reclaim disk from any SUPERSEDED processed store for this source,
        keeping only *keep_store_path* (the current clean
        ``<display_name>_processed.zarr``).

        Under the Brick-B clean-name scheme there is ONE store per source, so a
        recipe change now OVERWRITES the clean path (nothing to GC there).  The
        only supersedable stores are the LEGACY per-recipe caches from the old
        naming (``<display_name>__<source_id>__proc__<hash>.zarr``); this sweeps
        those for THIS source (matched by the exact ``<display_name>__<sid>__``
        prefix so a DIFFERENT source's cache is never touched), never the kept
        clean store.  A ``.zarr`` DirectoryStore has no lingering read handle, so
        it is safe even while the just-added lazy display layers read the kept
        store.  Best-effort; never raises."""
        import glob
        import shutil
        try:
            keep = str(keep_store_path)
            base = os.path.basename(keep)
            suffix = "_processed.zarr"
            if not base.endswith(suffix):
                return  # not a clean-name store → nothing to key the sweep on
            folder = os.path.dirname(keep)
            # Resolve THIS source's id so we only sweep its own legacy caches.
            sid = None
            try:
                w = getattr(self, "widget", None)
                ws = (w.output_file_path.text() or "").strip() if w else ""
                if ws:
                    # Identity, not the session path (Brick 4a): a relinked or
                    # missing raw file must still sweep ITS OWN legacy caches.
                    sid = self._loaded_source_id_for(load_state(ws))
            except Exception:
                sid = None
            # Recover the TRUE display_name.  Under the B2 clean name
            # ``<display>__<sid>_processed.zarr`` a bare ``_processed.zarr`` strip
            # leaves ``<display>__<sid>`` (sid-polluted), which would DOUBLE the sid
            # in the ``__proc__`` glob below and orphan legacy caches — so strip the
            # full ``__<sid>_processed.zarr`` suffix when the id is known (else the
            # bare suffix, for a legacy clean name or an unknown id).
            stem = base[:-len(suffix)]                 # <display>__<sid> | <display>
            if sid and stem.endswith(f"__{sid}"):
                display_name = stem[:-len(f"__{sid}")]
            else:
                display_name = stem
            # Precise prefix when the id is known; a conservative fallback (any
            # ``__*__proc__``) only when it isn't — still same-display-name scoped.
            if sid:
                pattern = os.path.join(
                    folder, glob.escape(f"{display_name}__{sid}") + "__proc__*.zarr")
            else:
                pattern = os.path.join(
                    folder, glob.escape(display_name) + "__*__proc__*.zarr")
            for p in glob.glob(pattern):
                if _abspath_norm(p) == _abspath_norm(keep):
                    continue
                try:
                    if os.path.isdir(p):
                        shutil.rmtree(p, ignore_errors=True)
                    elif os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass
        except Exception as exc:  # noqa: BLE001
            self._log("STEP1-PROCESS", f"legacy __proc__ cleanup: {exc}")

    def _persist_processed_pointer(self, store_path):
        """Record a workspace-RELATIVE ``processed_store`` pointer on the loaded
        source's schema-v2 registry row (Brick B persistence) so a
        fresh streamed ⟨processed⟩ store is re-discoverable on reopen.

        ADDITIVE + best-effort — mirrors :meth:`_persist_prepared_source`
        (relpath, :func:`upsert_source` via :func:`set_processed_store`,
        ``origin`` union); never raises into the caller.  The pointer is a
        DISCOVERY HINT only: freshness stays authoritative in the store's own
        ``epicycle_processed`` attrs via :meth:`_fresh_processed_store_for`, so
        the ``recipe_hash`` deliberately does NOT ride in the pointer.  Called at
        each streaming-Process finish AND the fresh-skip re-find so a rename or a
        collision-suffixed store is still resolved first on reopen."""
        try:
            w = getattr(self, "widget", None)
            workspace = (w.output_file_path.text() or "").strip() if w else ""
            if not workspace or not store_path:
                return
            try:
                rel = os.path.relpath(store_path, workspace).replace(os.sep, "/")
            except Exception:
                return
            state = load_state(workspace)
            # Brick 4a: the ROW is located by id; the path handed to the writer
            # is the ROW's own recorded path (never the session attribute), so a
            # relinked source keeps ONE row instead of gaining a second.
            source_id = self._loaded_source_id_for(state)
            src_path = self._loaded_source_path(state)
            if not source_id and not src_path:
                return
            if not source_id:
                source_id = self._resolve_source_id_for(state, src_path)
            set_processed_store(
                state, source_id=source_id, path=src_path, rel_path=rel)
            save_state(workspace, state)
            self._log("STEP1-PROCESS", f"processed_store pointer -> {rel}")
        except Exception as exc:  # noqa: BLE001
            self._log("STEP1-PROCESS", f"processed pointer persist skipped: {exc}")

    def _current_processed_pointer_path(self):
        """The loaded source's persisted ``processed_store`` pointer as an abspath,
        or ``None`` — the POINTER only (never the deterministic fallback), so a
        caller reclaims only a store the pointer actually named.  Never raises."""
        try:
            w = getattr(self, "widget", None)
            workspace = (w.output_file_path.text() or "").strip() if w else ""
            if not workspace:
                return None
            state = load_state(workspace)
            row = self._loaded_source_row(state)      # id-first (Brick 4a)
            ptr = get_processed_store(row)
            if ptr and ptr.get("path"):
                return os.path.join(workspace, *str(ptr["path"]).split("/"))
        except Exception:
            pass
        return None

    def _reclaim_superseded_store(self, old_store, new_store):
        """``rmtree`` *old_store* iff it EXISTS and differs from *new_store* — the
        B2-B pointer-driven reclaim of a source's superseded processed store (e.g.
        a pre-B2 ``<stem>_processed.zarr`` orphaned by the rename, or a display-name
        rename).  EXACT path, NEVER a glob (a glob would risk another same-stem
        source's live store — the B2 bug in a new dress).  Best-effort; never
        raises."""
        try:
            if (old_store and new_store
                    and _abspath_norm(old_store) != _abspath_norm(new_store)
                    and os.path.isdir(old_store)):
                import shutil
                shutil.rmtree(old_store, ignore_errors=True)
                self._log("STEP1-PROCESS",
                          f"reclaimed superseded store {old_store}")
        except Exception:
            pass

    def _resolve_processed_store_for_load(self, workspace, src_path, display_name):
        """The on-disk ⟨processed⟩ store path for the loaded source — POINTER
        FIRST (the persisted ``processed_store`` on the source row, which survives
        a display-name rename + a same-name/different-source collision), else the
        deterministic clean ``<display_name>_processed.zarr`` fallback.

        Returns an abspath (existence NOT guaranteed — the caller gates freshness
        via :meth:`_fresh_processed_store_for`) or ``None``.  Never raises."""
        try:
            state = load_state(workspace)
            row = self._find_source_row(state, src_path)
            ptr = get_processed_store(row)
            if ptr and ptr.get("path"):
                return os.path.join(workspace, *str(ptr["path"]).split("/"))
        except Exception:
            pass
        # No pointer → deterministic name.  Prefer the current per-source B2 name;
        # fall back to the LEGACY pre-B2 ``<display>_processed.zarr`` when it exists
        # on disk but the new-name one doesn't — a pointerless legacy store still
        # resolves (read-only back-compat; this is not a WRITE path, so the old
        # same-stem collision cannot recur here).
        try:
            store_path, _sid = self._resolve_processed_store_path(
                workspace, display_name, src_path)
            if not os.path.isdir(store_path):
                legacy = os.path.join(
                    self._largedata_dir(workspace),
                    f"{display_name}_processed.zarr")
                if os.path.isdir(legacy):
                    return legacy
            return store_path
        except Exception:
            return None

    def _maybe_light_processed_toggle_on_load(self, display_name):
        """Reopen detection (Brick B): after a
        prepared/streaming source loads, LIGHT the ② Raw / Processed toggle iff a
        FRESH on-disk ⟨processed⟩ store exists for the loaded data + the CURRENT
        recipe — but DO NOT show it (the default view stays RAW on reopen; one
        click / ``\\`` flips to Processed).

        Runs SYNCHRONOUSLY from :meth:`_apply_multiscale_store` AFTER
        ``_teardown_preview`` (which cleared ``_processed_available``), so no
        untagged-raw frame flashes.  The recipe hash is computed POST-LOAD with
        the LOADED channel count (``normalize_recipe`` clamps to it, so the hash
        is channel-count dependent — computing at passive select would be wrong).
        Freshness is attrs-authoritative (:meth:`_fresh_processed_store_for`): a
        recipe edit (hash mismatch) or a changed raw file (fingerprint mismatch)
        → not fresh → the toggle stays greyed (never a stale processed shown as
        current).  Fully guarded — never raises on load."""
        try:
            w = getattr(self, "widget", None)
            if w is None:
                return
            workspace = (w.output_file_path.text() or "").strip()
            # The ROW's recorded path (Brick 4a) — the resolvers below key a row
            # lookup off it, so it must survive a relink.
            src_path = self._loaded_source_path() if workspace else None
            if not workspace or not src_path:
                return
            channels = getattr(self, "_source_display_channels", None) or []
            n = len(channels)
            if n <= 0:
                return
            get_recipe = getattr(w, "get_recipe_json", None)
            if not callable(get_recipe):
                return
            recipe_json = get_recipe()
            recipe_hash = self._processed_recipe_hash(recipe_json, n)
            store_path = self._resolve_processed_store_for_load(
                workspace, src_path, display_name)
            if not store_path:
                return
            fresh = self._fresh_processed_store_for(
                store_path, src_path, recipe_hash)
            if fresh is None:
                return
            names = [nm for (nm, _levels) in channels]
            # Light the toggle (Processed enabled) but stay on RAW — remember how
            # to re-create the processed set on the first flip.
            self._processed_available = True
            self._processed_redisplay = {
                "kind": "store", "store_path": fresh, "names": names}
            self._refresh_view_toggle_state()
            self._log("STEP1-PROCESS",
                      f"reopen: fresh processed store -> {fresh} "
                      "(toggle lit, view stays raw)")
        except Exception as exc:  # noqa: BLE001 — never raise on load
            self._log("STEP1-PROCESS", f"reopen processed detect skipped: {exc}")

    def _resolve_direct_stream_channels(self, names, channels):
        """Best-effort DIRECT-ZARR ZYX handles for the streaming writer.

        Opens the displayed store's level-0 zarr array and returns one 3D ZYX
        direct-read handle per channel (a raw zarr for a ZYX store, a
        :class:`_ZarrChannelView` for a CZYX store), so the engine reads each block
        DIRECTLY off zarr rather than computing a dask graph per block (~1.6×
        faster).  Returns the ORIGINAL lazy *channels* unchanged on ANY failure
        (no store, shape mismatch, unexpected ndim) — the engine then falls back to
        ``.compute()`` per block, still correct.  Never raises.  (The controller
        resolves this — not the worker — because it holds the store path.)"""
        store = getattr(self, "_current_multiscale_store", None)
        if not store:
            return channels
        try:
            import zarr
            from packages.epicycle_largedata.multiscale import _zarr_store
            group = zarr.open_group(store=_zarr_store(str(store), "r"), mode="r")
            z0 = group["0"]
        except Exception as exc:  # noqa: BLE001
            self._log("STEP1-PROCESS", f"direct-zarr resolve skipped: {exc}")
            return channels
        try:
            ndim = int(getattr(z0, "ndim", len(z0.shape)))
            idx_by_name = getattr(self, "_channel_index_by_name", {}) or {}
            resolved = []
            for ci, name in enumerate(names):
                lazy_shape = tuple(int(s) for s in getattr(channels[ci], "shape", ()))
                if ndim == 3:
                    if tuple(int(s) for s in z0.shape) != lazy_shape:
                        return channels
                    resolved.append(z0)
                elif ndim == 4:
                    cidx = idx_by_name.get(name, ci)
                    if not (0 <= int(cidx) < int(z0.shape[0])):
                        return channels
                    view = _ZarrChannelView(z0, int(cidx))
                    if tuple(view.shape) != lazy_shape:
                        return channels
                    resolved.append(view)
                else:
                    return channels
            return resolved
        except Exception as exc:  # noqa: BLE001
            self._log("STEP1-PROCESS", f"direct-zarr resolve fallback: {exc}")
            return channels

    def _process_whole_volume_streaming(self, names, channels, spacing,
                                        recipe_json):
        """Stream the recipe over the FULL per-channel volumes into a ``__proc__``
        OME-Zarr store on a QThread → lazy multiscale ⟨processed⟩ display.

        Phase 2a scope: local + pointwise + normalize ops.  A CLAHE / bleach /
        geometry op is REFUSED up front (a clear plain-language message — NOT a
        silent drop or a crash).  A fresh store for the SAME recipe + source is
        re-displayed WITHOUT recomputing.  The raw-source extract state
        (`_source_lazy` / `_source_display_channels` / …) is NEVER touched — the
        processed store is DISPLAY-ONLY, so the next Process re-processes the RAW
        source, never the processed output (the double-processing fix by
        construction)."""
        n = len(channels)

        # (1) Phase 2a scope gate — refuse a whole-array remainder CLEARLY.
        if self._recipe_has_stream_remainder(recipe_json, n, spacing=spacing):
            self._set_preview_status(
                "This recipe includes a contrast (CLAHE), depth-brightness, or "
                "resize step that can't stream yet — that's coming in the next "
                "update.  For now, drop that step, or use a region Preview to see "
                "it on a smaller crop.",
                ok=False)
            self._log("STEP1-PROCESS", "streaming refused: whole-array remainder")
            return

        w = getattr(self, "widget", None)
        workspace = (w.output_file_path.text() or "").strip() if w else ""
        if not workspace:
            self._set_preview_status(
                "Set a workspace first — the streamed result is written under it.",
                ok=False)
            return
        src_path = self._loaded_source_path()       # ROW's path (Brick 4a)
        display_name = (w.display_name_edit.text() or "").strip() if w else ""
        if not display_name:
            base = os.path.basename(src_path) if src_path else "source"
            display_name = os.path.splitext(base)[0] or "source"

        recipe_hash = self._processed_recipe_hash(recipe_json, n)
        store_path, _source_id = self._resolve_processed_store_path(
            workspace, display_name, src_path)

        # A new run REPLACES the previous processed render (either regime) so a
        # stale volume never lingers while the new one computes / loads.
        self._clear_processed_layers()
        self._clear_processed_store_layers()
        self._process_names = names

        # (2) Fresh-skip: an up-to-date store for this recipe + source already
        # exists → just display it (no minutes-long recompute).
        fresh = self._fresh_processed_store_for(store_path, src_path, recipe_hash)
        if fresh is None:
            # B2-C: an EXISTING workspace processed under the pre-B2 name resolves
            # via the persisted ``processed_store`` pointer, not the new
            # deterministic name — probe it too so we don't needlessly recompute a
            # still-valid store (minutes on this user's large data) and orphan it.
            ptr_path = self._resolve_processed_store_for_load(
                workspace, src_path, display_name)
            if ptr_path and _abspath_norm(ptr_path) != _abspath_norm(store_path):
                fresh = self._fresh_processed_store_for(
                    ptr_path, src_path, recipe_hash)
        if fresh is not None:
            n_layers = self._display_processed_store(fresh, names)
            if n_layers:
                # Mark processed available + remember how to re-create it, then
                # clean-swap to the processed view (removes raw from the canvas).
                self._processed_available = True
                self._processed_redisplay = {
                    "kind": "store", "store_path": fresh, "names": names}
                # Persist the pointer so a fresh re-find is re-discoverable on
                # reopen (Brick B) — idempotent last-wins, additive.
                self._persist_processed_pointer(fresh)
                self._show_processed_view()
                self._set_preview_status(
                    f"Already processed — showing {n_layers} ⟨processed⟩ "
                    "layer(s) (streamed to disk).", ok=True)
            else:
                self._set_preview_status(
                    "Already processed (streamed to disk).", ok=True)
            self._log("STEP1-PROCESS",
                      f"fresh-skip: displayed existing {fresh}")
            return

        # (3) Lock the surface + stamp the run (SHARES `_processing` /
        # `_process_token` / the Cancel-button flip with the in-memory regime — one
        # Process runs at a time).
        self._process_token += 1
        token = self._process_token
        self._processing = True
        self._set_step1_busy(True)
        try:
            self.widget._busy_show("Processing (streaming) …")
        except Exception:
            pass
        self._set_process_button_cancel(True)

        # (4) Resolve direct-zarr channels (fast read path) + the RAM-adaptive
        # block-pool width + the source fingerprint for the store's provenance.
        direct_channels = self._resolve_direct_stream_channels(names, channels)
        block_workers = self._derive_block_workers(recipe_json)
        source_fingerprint = self._process_source_fingerprint(src_path)

        try:
            os.makedirs(os.path.dirname(store_path), exist_ok=True)
        except Exception as exc:
            self._set_step1_busy(False)
            self._processing = False
            self._set_process_button_cancel(False)
            try:
                self.widget._busy_hide()
            except Exception:
                pass
            self._set_preview_status(
                f"Could not create the store folder: {exc}", ok=False)
            return

        # (5) Launch the streaming worker — mirror the in-memory launch (own thread
        # + moveToThread + started→run + finished/error→handlers + →thread.quit +
        # thread.finished→cleanup); SEPARATE `_process_store_*` fields (added to the
        # `shutdown()` sweep), the shared launch-guard terminal funnel.
        try:
            if self._process_store_thread is not None:
                self._draining_process_store_threads.append(
                    (self._process_store_thread, self._process_store_worker))
            # Structure parity (chunk / pyramid) with the raw prepared store.
            # Brick 4b: the chunk SHAPE is no longer threaded from the config —
            # the engine reads it off the SOURCE store's own ``.zarray`` chunks
            # (``chunk=None`` → inherit), so a processed store always matches the
            # prepared store it was streamed from even when the workspace config
            # has since changed or the store predates the config entirely.
            # downsample / min_size still come from the config (they are pyramid
            # POLICY, not on-disk layout).
            _ld = self._current_largedata_config(workspace) or {}
            worker = _PreprocProcessToStoreWorker(
                recipe_json, direct_channels, spacing, store_path,
                block_workers=block_workers,
                source_fingerprint=source_fingerprint,
                chunk=None,
                downsample=_ld.get("downsample"),
                min_size=_ld.get("min_size"))
            thread = QThread()
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.finished.connect(
                lambda path, manifest, _t=token, _n=names:
                self._on_process_store_ready(path, manifest, _n, _t))
            worker.error.connect(
                lambda msg, _t=token: self._on_process_store_error(msg, _t))
            # Token BOUND IN THE LAMBDA (Brick 5 fix 2) — see the sibling
            # in-memory launch.
            worker.progress.connect(
                lambda done, total, _t=token:
                self._on_process_store_progress(done, total, _t))
            worker.finished.connect(thread.quit)
            worker.error.connect(thread.quit)
            thread.finished.connect(
                lambda _t=thread: self._cleanup_process_store_thread(_t))
            self._process_store_thread = thread
            self._process_store_worker = worker
            self._log(
                "STEP1-PROCESS",
                f"launch (streaming) token={token} channels={n} "
                f"block_workers={block_workers} -> {store_path}")
            thread.start()
        except Exception as exc:
            # Neither finished nor error will fire — clear the lock + bar + button
            # HERE so the UI can never stay stuck (the launch-guard pattern).
            self._process_store_thread = None
            self._process_store_worker = None
            try:
                self.widget._busy_hide()
            except Exception:
                pass
            self._set_step1_busy(False)
            self._processing = False
            self._set_process_button_cancel(False)
            self._set_preview_status(
                f"Process could not start: {exc}", ok=False)
            self._log("STEP1-PROCESS",
                      f"launch (streaming) FAILED token={token}: {exc}")

    def _on_process_store_ready(self, store_path, manifest, names, token):
        """Main-thread STREAMING finish: open the ``__proc__`` store lazily + add
        the ADDITIVE ⟨processed⟩ multiscale display layers.

        Discards a STALE run (Cancel / source-switch since launch — the token
        guard).  CRITICAL: does NOT call ``unload_source`` and NEVER touches the
        RAW-source extract state (`_source_lazy` / `_source_display_channels` /
        `_channel_index_by_name` / `_source_axes_resolved` / `_loaded_source_file`
        / `_current_multiscale_store`) — the processed store is DISPLAY-ONLY, so
        the next Process re-processes the RAW source, not the processed output."""
        if token != getattr(self, "_process_token", 0):
            self._log("STEP1-PROCESS",
                      f"discard stale streaming result token={token}")
            return
        try:
            # LATE FINISH after the user left Step 1.  The token guard above does
            # NOT cover this: `on_step_exit` only bumps the token via
            # `cancel_process_whole_volume`, and that is gated on a run it can
            # SEE in flight.  Displaying here would re-add ⟨processed⟩ Image
            # layers to the SHARED canvas (the Step-3 all-Image-layers abort) and
            # restart a progressive loader AFTER `ensure_sync_slicing` already
            # ran — the async_ poison back on, from another step's screen.  Go
            # inert for UI; the `finally` below still releases the lock.  (Same
            # shape as the ④-Trim late-finish-inert guard.)  The store stays on
            # disk and is re-discovered by the reopen reconcile.
            if getattr(self, "_left_step1", False):
                self._log("STEP1-PROCESS",
                          f"streaming finish after step exit token={token} "
                          f"— unlocking only, no display")
                return
            # Defensive (audit LOW-2): a remainder here means the pre-launch gate
            # missed a non-streaming op → the engine dropped it → this store is a
            # SILENTLY-WRONG (dropped-op) result.  The gate refuses those, so this is
            # unreachable; if it EVER fires, REFUSE + delete the store rather than
            # render dropped-op pixels as "processed".
            if isinstance(manifest, dict) and manifest.get("whole_array_remainder"):
                self._log("STEP1-PROCESS",
                          "REFUSE streamed store has a whole_array_remainder — removing")
                try:
                    import shutil
                    if os.path.isdir(str(store_path)):
                        shutil.rmtree(str(store_path), ignore_errors=True)
                except Exception:
                    pass
                self._set_preview_status(
                    "That recipe includes a step that can't stream yet — nothing was "
                    "changed.", ok=False)
                return
            n_layers = 0
            try:
                n_layers = self._display_processed_store(store_path, names)
            except Exception as exc:  # noqa: BLE001
                self._log("STEP1-PROCESS",
                          f"processed store render FAILED token={token}: {exc}")
            if n_layers:
                # Mark processed available + remember how to re-create it, then
                # clean-swap to the processed view (removes raw from the canvas).
                self._processed_available = True
                self._processed_redisplay = {
                    "kind": "store", "store_path": store_path, "names": names}
                # B2-B: capture the source's PRIOR processed-store pointer BEFORE we
                # repoint it, so we can reclaim a superseded store just below.
                old_store = self._current_processed_pointer_path()
                # Persist the pointer so the store is re-discoverable on reopen
                # (Brick B) — additive, best-effort, freshness stays store-attr
                # authoritative.
                self._persist_processed_pointer(store_path)
                # B2-B: reclaim the PREVIOUS store (exact path, pointer-driven).
                self._reclaim_superseded_store(old_store, store_path)
                self._show_processed_view()
            # Reclaim disk: drop any superseded LEGACY __proc__ cache for this
            # source (the clean-name store overwrites in place, so this only
            # sweeps pre-rename per-recipe stores).  Safe — DirectoryStore has no
            # lingering read handle even while the new lazy layers read the store.
            try:
                self._delete_superseded_processed_stores(store_path)
            except Exception:
                pass
            if n_layers:
                self._set_preview_status(
                    f"Processed (streaming): whole volume added as {n_layers} "
                    "⟨processed⟩ layer(s).  Showing processed — use the Raw / "
                    "Processed toggle (or the \\ key) to switch back.  Streamed to "
                    "disk (memory-bounded).",
                    ok=True)
            else:
                self._set_preview_status(
                    "Processed whole volume (streamed to disk).", ok=True)
        finally:
            # One terminal funnel — release the lock, rest the bar, restore the
            # button — so a render failure cannot leave the surface stuck.
            self._set_step1_busy(False)
            self._processing = False
            self._set_process_button_cancel(False)
            try:
                self.widget._busy_hide()
            except Exception:
                pass

    def _on_process_store_error(self, msg, token):
        """Main-thread STREAMING error / cancel: themed status + clear the lock +
        restore the button.  Discards a stale run's terminal emit too."""
        if token != getattr(self, "_process_token", 0):
            return
        self._set_step1_busy(False)
        self._processing = False
        self._set_process_button_cancel(False)
        try:
            self.widget._busy_hide()
        except Exception:
            pass
        if str(msg) != "cancelled":
            self._set_preview_status(f"Process failed: {msg}", ok=False)
        self._log("STEP1-PROCESS", f"streaming error token={token}: {msg}")

    def _on_process_store_progress(self, done, total, token=None):
        """Drive the shared busy bar from the streaming worker's per-wave block
        counts (main thread).  A simple determinate bar over the block budget
        (measure + apply + coarsen) — the phase-weighting is inherent (blocks are
        the unit).

        Ignored once the run is no longer active, and — Brick 5 fix 2 — once its
        launch token has been superseded.  `_processing` alone was not enough: a
        cancel bumps the token but a re-launch turns `_processing` back ON, so a
        DRAINING worker (a streaming Process is not interruptible mid-block) went
        on driving the new run's bar with the old run's block counts.  ``token``
        defaults to None so a bare call still paints (the pre-fix contract)."""
        if not getattr(self, "_processing", False):
            return
        if token is not None and token != getattr(self, "_process_token", None):
            return
        try:
            if total and int(total) > 0:
                self.widget._busy_progress(
                    "Processing (streaming) …", int(done), int(total))
            else:
                self.widget._busy_show("Processing (streaming) …")
        except Exception:
            pass

    def _cleanup_process_store_thread(self, thread):
        """Drop the finished streaming (thread, worker) pair once its QThread has
        actually stopped (`thread.finished`).  Idempotent; touches only controller
        thread state."""
        if thread is getattr(self, "_process_store_thread", None):
            self._process_store_thread = None
            self._process_store_worker = None
        self._draining_process_store_threads = [
            (t, w) for (t, w) in
            getattr(self, "_draining_process_store_threads", [])
            if t is not thread
        ]

    def _clear_processed_store_layers(self):
        """Remove the lazy-multiscale ⟨processed⟩ STORE render layers from the
        shared canvas (hide-then-remove, freed-texture safety).

        Idempotent, tolerant of a layer the user already deleted, always empties
        the tracking list.  Called before each new Process run + on source
        load / unload / switch, and DIRECTLY from `on_step_exit` (next to
        `_clear_patch_locations`, before the `ensure_sync_slicing` guard) so a
        stale processed store never outlives its source and no progressive
        loader is still live when that guard runs.  Never raises."""
        viewer = getattr(self, "viewer", None)
        # Not batched - see the measurement note in `_unload_source_inner`.
        for layer in list(getattr(self, "_processed_store_layers", None) or []):
            if viewer is None:
                continue
            try:
                if layer in viewer.layers:
                    layer.visible = False       # hide BEFORE remove
                    viewer.layers.remove(layer)
            except Exception:
                pass
        self._processed_store_layers = []
        # Item 4: the removal (hide-then-remove) fires the progressive seam's
        # `removed` teardown for any progressive processed layer, so the engine +
        # GLIR meter detach here.  Reset the active flag — no processed progressive
        # layer remains (the store/channel fields are re-set on re-display, cleared
        # on teardown).
        self._processed_progressive_active = False

    def _display_processed_store(self, store_path, names):
        """Add the streamed ``__proc__`` store as ADDITIVE lazy-multiscale
        ⟨processed⟩ Image layers — one per channel.

        MAIN-THREAD ONLY.  Opens the store LAZILY (``open_multiscale`` →
        ``da.from_zarr`` per level — nothing materialized; napari uploads only the
        coarsest level in 3D, so this is render-safe + memory-light).  Replaces any
        previous store render (clears first).  Each layer: source-channel colour,
        ``contrast_limits=(0, 65535)``, additive over the raw source (which stays
        displayed + is still the extract source).  Returns the layer count.  The
        ``_adding_processed_layers`` re-entrancy guard keeps `_on_layers_changed`
        from counting the not-yet-tracked layers as channels."""
        viewer = getattr(self, "viewer", None)
        self._clear_processed_store_layers()
        if viewer is None:
            return 0
        try:
            levels, meta = open_multiscale(store_path)
        except Exception as exc:  # noqa: BLE001
            self._log("STEP1-PROCESS", f"open processed store failed: {exc}")
            return 0
        level0 = levels[0]
        axes_meta = meta.get("axes") or []
        has_channel = any((a or {}).get("type") == "channel" for a in axes_meta)
        if has_channel:
            n_ch = int(level0.shape[0])
            chans = [[lvl[i] for lvl in levels] for i in range(n_ch)]
        else:
            chans = [list(levels)]

        # Physical-µm scale for the processed store: use the store's OWN NGFF
        # level-0 spacing (a resampled store has a DIFFERENT grid than raw, so
        # `_preview_spacing()` — the raw voxel — would be wrong).  Fall back to the
        # raw voxel when the store carries no scale (== raw grid, no resample).
        # Cached so the ndisplay progressive swap uses the same value.  `units`
        # keys on the RAW voxel known-ness (both views show the same physical
        # volume), so an unknown raw source stays labelled "px".
        proc_vox = self._voxel_from_store_meta(meta) or self._preview_spacing()
        self._processed_store_voxel = proc_vox
        pscale = tuple(float(v) for v in proc_vox)
        punits = _units_for_voxel(self._preview_spacing())

        added = []
        # Item 4: the (marker_name, levels) pairs of the layers ACTUALLY added,
        # so the progressive swap can rebuild each processed channel as a
        # progressive layer (mirrors the raw `_source_display_channels`).
        display_channels = []
        n = len(chans)
        self._adding_processed_layers = True
        try:
            for ci in range(n):
                ch_levels = chans[ci]
                name = names[ci] if names and ci < len(names) else f"ch{ci}"
                marker_name = f"{PROCESSED_LAYER_MARKER}{name}"
                add_kwargs = dict(
                    name=marker_name,
                    colormap=self._preview_colormap_for_channel(name),
                    blending="additive",
                    contrast_limits=(0, 65535),
                    translate=(0, 0, 0),
                    multiscale=True,
                    visible=True,
                )
                try:
                    layer = viewer.add_image(ch_levels, **add_kwargs)
                except TypeError:
                    # A viewer/fake that doesn't accept an optional kwarg — retry
                    # with the essentials (mirrors `_add_processed_layers`).
                    for opt in ("contrast_limits", "translate", "visible",
                                "multiscale"):
                        add_kwargs.pop(opt, None)
                    try:
                        layer = viewer.add_image(ch_levels, **add_kwargs)
                    except Exception:
                        continue
                except Exception:
                    continue
                # Physical-µm scale (store's own spacing) + units AFTER add.
                _apply_layer_scale(layer, pscale)
                _apply_layer_units(layer, punits)
                added.append(layer)
                display_channels.append((marker_name, ch_levels))
        finally:
            self._adding_processed_layers = False
        self._processed_store_layers = added
        # Item 4: record the store + per-channel levels of the freshly-added
        # NATIVE processed layers so the ndisplay progressive swap (dispatched via
        # `_maybe_swap_progressive_on_ndisplay` when processed is the active view)
        # can re-engage the progressive engine on them.  `_progressive_active` for
        # processed resets to False — these are native layers.  Only set when a
        # multiscale store rendered at least one channel; a single-scale / failed
        # add leaves the fields as-is (the swap gate no-ops on a non-list channel).
        if added and len(chans) and all(
                isinstance(cl, (list, tuple)) for (_n, cl) in display_channels):
            self._current_processed_multiscale_store = store_path
            self._processed_display_channels = display_channels
        self._processed_progressive_active = False
        # PURE (clean-swap rework): this method only ADDS the layers — the caller
        # (each Process finish site, or `_readd_processed_layers`) drives the
        # Raw→Processed clean swap via `_show_processed_view()`.  Calling it here
        # would recurse through `_readd_processed_layers`.
        self._log("STEP1-PROCESS",
                  f"render {len(added)} ⟨processed⟩ store layer(s) from {store_path}")
        return len(added)

    # ------------------------------------------------------------------
    # Progressive-3D for the ⟨processed⟩ STORE view (item 4).  Mirrors the RAW
    # progressive machinery (`_maybe_swap_progressive_on_ndisplay` /
    # `_do_swap_native_to_progressive` / `_reveal_progressive_after_build`) but
    # operates on the DISPLAY-ONLY processed store layers, and NEVER touches the
    # raw `_source_*` state or the Extract channel picker (the processed layers
    # carry the ⟨processed⟩ marker and are display-only).  Safe under the clean
    # raw⇄processed swap: only ONE set is ever on the canvas, so there is never
    # more than one progressive layer set live.
    # ------------------------------------------------------------------
    def _is_processed_multiscale_store_displayed(self) -> bool:
        """True iff a prepared MULTISCALE ⟨processed⟩ store is the current
        processed display (all channels are pyramid-level LISTS) — the only case
        the processed progressive swap applies to.  The in-memory processed
        regime (`_processed_layers`, single-scale) returns False so it stays
        native.  Mirrors :meth:`_is_multiscale_store_displayed`."""
        if not getattr(self, "_current_processed_multiscale_store", None):
            return False
        channels = getattr(self, "_processed_display_channels", None)
        if not channels:
            return False
        try:
            return all(
                isinstance(cd, (list, tuple)) for (_n, cd) in channels)
        except Exception:
            return False

    def _processed_store_is_active_view(self) -> bool:
        """True iff the ⟨processed⟩ STORE view is the ACTIVE canvas display (the
        clean swap removed raw), so the progressive machinery should operate on
        the processed layers.  Requires raw OFF the canvas + a multiscale
        processed store + at least one processed store layer resident.  False for
        the in-memory processed regime and whenever raw is on the canvas."""
        if getattr(self, "_source_layers_on_canvas", False):
            return False   # raw is the active view — raw path owns the swap
        if not self._is_processed_multiscale_store_displayed():
            return False
        return bool(getattr(self, "_processed_store_layers", None))

    def _maybe_swap_processed_progressive_on_ndisplay(self):
        """Processed twin of :meth:`_maybe_swap_progressive_on_ndisplay`.  Enter
        3D with progressive wanted -> defer the native->progressive swap one tick
        (past napari's ndisplay transition, the crash-safe path raw uses); turn
        progressive off in 3D, or enter 2D -> swap back to native synchronously.
        A no-op unless the processed store is the active canvas view."""
        if not self._processed_store_is_active_view():
            return
        try:
            in_3d = int(self.viewer.dims.ndisplay) == 3
        except Exception:
            return
        wanted = self.progressive_wanted()
        active = getattr(self, "_processed_progressive_active", False)
        if in_3d:
            if wanted and not active:
                QTimer.singleShot(
                    0, self._swap_processed_native_to_progressive)
            elif not wanted and active:
                self._swap_processed_progressive_to_native()
        else:
            if active:
                self._swap_processed_progressive_to_native()

    def _swap_processed_native_to_progressive(self):
        """Deferred (post-ndisplay-transition) processed native->progressive
        swap: replace the native ⟨processed⟩ store layers with progressive-loading
        layers, carrying each layer's current contrast.  Fails CLOSED per channel
        (re-adds the native layer) so a seam failure never blanks the processed
        display.  Re-checks its guards because it runs one event-loop tick late
        (the user may have flipped to raw / 2D / off in the meantime)."""
        if not self._processed_store_is_active_view():
            return
        if getattr(self, "_processed_progressive_active", False):
            return
        try:
            if int(self.viewer.dims.ndisplay) != 3:
                return
        except Exception:
            return
        if not self.progressive_wanted():
            return
        try:
            from packages.epicycle_largedata import progressive as _prog
        except Exception:
            return

        self._swapping_source_display = True
        try:
            channels = list(getattr(self, "_processed_display_channels", None) or [])
            live = list(getattr(self, "_processed_store_layers", None) or [])
            cur_names = [getattr(l, "name", None) for l in live]
            # Capture appearances BEFORE hiding (hide clobbers the `visible`
            # snapshot), then hide EVERY native layer up front so no live coarse
            # VOLUME visual survives into the remove/add loop (the multi-channel
            # metered-flush crash window — same as raw's swap).
            appearances = []
            for i, (ch_name, _lv) in enumerate(channels):
                cn = cur_names[i] if i < len(cur_names) else ch_name
                appearances.append(self._capture_layer_appearance(cn))
            for cn in cur_names:
                self._hide_layer_by_name(cn)
            new_layers = []
            any_progressive = False
            for i, (ch_name, ch_levels) in enumerate(channels):
                appearance = appearances[i] if i < len(appearances) else {}
                self._remove_layer_by_name(
                    cur_names[i] if i < len(cur_names) else ch_name)
                contrast = appearance.get("contrast_limits")
                layer = None
                # Physical-µm scale from the PROCESSED store's OWN spacing (a
                # resampled store has a different grid than raw); the fallback
                # native layer uses the SAME value so it matches its progressive
                # siblings (grill 2 H3).
                proc_vox = (getattr(self, "_processed_store_voxel", None)
                            or self._preview_spacing())
                try:
                    kwargs = dict(name=ch_name,
                                  colormap=appearance.get("colormap"),
                                  blending="additive")
                    # Physical `scale=` only for a KNOWN voxel; unknown (1,1,1)
                    # omits it so the engine keeps its float32 auto-downscale
                    # (byte-identical legacy render) — see the raw swap above.
                    if not self._is_unit_voxel(proc_vox):
                        kwargs["scale"] = tuple(float(v) for v in proc_vox)
                    if contrast is not None:
                        kwargs["contrast_limits"] = contrast
                    layer = _prog.add_progressive_image(
                        self.viewer, ch_levels, **kwargs)
                except Exception:
                    layer = None
                if layer is None:
                    # Fail closed: re-add the native multiscale ⟨processed⟩ layer
                    # at the store's physical scale (matches the siblings).
                    layer = self._readd_native_layer(
                        ch_name, ch_levels, appearance, voxel_size=proc_vox)
                else:
                    any_progressive = True
                    zoom_depth = getattr(self, "_zoom_depth_wanted", None)
                    try:
                        layer._zoom_depth = zoom_depth
                    except Exception:
                        pass
                # Keep every NEW layer HIDDEN for the BUILD INSTANT only (the
                # metered GLIR flush crash window); revealed on the NEXT tick.
                try:
                    layer.visible = False
                except Exception:
                    pass
                new_layers.append((layer, appearance))
            self._processed_store_layers = [
                l for (l, _a) in new_layers if l is not None]
            self._processed_progressive_active = any_progressive
            self._refresh_zoom_depth_for_ndisplay()
        finally:
            self._swapping_source_display = False
        # One settled reconcile of ②/③ against the new (marker-excluded) layers.
        self._on_layers_changed()
        QTimer.singleShot(
            0, lambda built=list(new_layers):
            self._reveal_processed_progressive_after_build(built))

    def _reveal_processed_progressive_after_build(self, built):
        """Deferred reveal of the ⟨processed⟩ layers built HIDDEN by the
        entering-3D swap (crash-safe: the hide spans only the synchronous build;
        this runs a tick later, past napari's deferred build/paint flush).
        Restores each layer's appearance (visible/gamma) and kicks the loader's
        camera-driven refinement.  Undo-guard keys on the processed store still
        being the active view + ndisplay==3.  Mirrors
        :meth:`_reveal_progressive_after_build`."""
        if not self._processed_store_is_active_view():
            return
        try:
            if int(self.viewer.dims.ndisplay) != 3:
                return
        except Exception:
            return
        revealed = 0
        for captured, appearance in built:
            name = getattr(captured, "name", None)
            if name is None:
                continue
            try:
                live = (self.viewer.layers[name]
                        if name in self.viewer.layers else None)
            except Exception:
                live = None
            if live is None:
                continue
            self._restore_layer_extras(live, appearance)   # visible + gamma
            self._kick_progressive_loader(live)
            revealed += 1
        self._refresh_zoom_depth_for_ndisplay()

    def _swap_processed_progressive_to_native(self):
        """Swap the progressive ⟨processed⟩ layers back to native multiscale
        (entering 2D, or turning progressive off in 3D).  Removing each
        progressive layer fires the seam's ``removed`` teardown, which uninstalls
        the process-wide GLIR meter once the last one goes away.  Mirrors
        :meth:`_do_swap_progressive_to_native`."""
        if not self._processed_store_is_active_view():
            return
        if not getattr(self, "_processed_progressive_active", False):
            return
        self._swapping_source_display = True
        try:
            channels = list(getattr(self, "_processed_display_channels", None) or [])
            live = list(getattr(self, "_processed_store_layers", None) or [])
            cur_names = [getattr(l, "name", None) for l in live]
            appearances = []
            for i, (ch_name, _lv) in enumerate(channels):
                cn = cur_names[i] if i < len(cur_names) else ch_name
                appearances.append(self._capture_layer_appearance(cn))
            for cn in cur_names:
                self._hide_layer_by_name(cn)
            new_layers = []
            for i, (ch_name, ch_levels) in enumerate(channels):
                appearance = appearances[i] if i < len(appearances) else {}
                self._remove_layer_by_name(
                    cur_names[i] if i < len(cur_names) else ch_name)
                # Symmetry with the forward swap (native→progressive above): the
                # processed store's OWN spacing, not the raw spins.  They differ
                # only when the raw voxel was edited AFTER processing (the store
                # is display-only and survives the edit); otherwise equal.
                proc_vox = (getattr(self, "_processed_store_voxel", None)
                            or self._preview_spacing())
                layer = self._readd_native_layer(
                    ch_name, ch_levels, appearance, voxel_size=proc_vox)
                new_layers.append(layer)
            self._processed_store_layers = [l for l in new_layers if l is not None]
            self._processed_progressive_active = False
            self._refresh_zoom_depth_for_ndisplay()
        finally:
            self._swapping_source_display = False
        self._on_layers_changed()

    # ------------------------------------------------------------------
    # Raw ⇄ Processed mutually-exclusive VIEW toggle (Brick A) — DISPLAY ONLY
    # ------------------------------------------------------------------

    def _has_processed_view(self) -> bool:
        """True iff a ⟨processed⟩ view EXISTS this session — regardless of which
        set is currently on the canvas.  Under the clean-swap contract a flip to
        raw REMOVES the processed layers, so the tracked layer lists can't answer
        this; `_processed_available` (set at each Process finish, cleared on
        teardown) does.  Governs the ② toggle's Processed enabled-state + the
        `\\` bind + the swap guard."""
        return bool(getattr(self, "_processed_available", False))

    def _is_processed_view_showing(self) -> bool:
        """The AUTHORITATIVE view state, read off the layers' ``.visible`` —
        never a private boolean that could drift across the ndisplay swap.
        True iff any ⟨processed⟩ layer is currently on the canvas + visible (under
        the clean swap the processed set is only present while it is the view)."""
        for layer in list(getattr(self, "_processed_store_layers", None) or []):
            if getattr(layer, "visible", False):
                return True
        for layer in list(getattr(self, "_processed_layers", None) or []):
            if getattr(layer, "visible", False):
                return True
        return False

    def _show_processed_view(self):
        """Show the ⟨processed⟩ set as the ONLY view — a CLEAN SWAP: re-create the
        processed layers if they were cleared (a prior flip to raw removes them),
        then REMOVE the raw source display from the canvas so only the viewed set
        remains in the layer list (2 layers for a 2-channel image, never 4 —
        mutually exclusive).

        The whole layer churn runs under the `_swapping_source_display` guard so
        the transient add/remove does NOT fire `_on_layers_changed` →
        `_sync_recipe_channels` → `_refresh_recipe_view` (which would blank the ②
        recipe panel mid-swap).  Best-effort; never raises."""
        if not self._has_processed_view():
            return
        viewer = getattr(self, "viewer", None)
        if viewer is None:
            return
        self._view_swap_busy_show("Loading processed view…")
        self._swapping_source_display = True
        try:
            # Re-create the processed set from its stored spec if a prior flip to
            # raw cleared it (idempotent no-op when it is still present).
            if not (getattr(self, "_processed_store_layers", None)
                    or getattr(self, "_processed_layers", None)):
                self._readd_processed_layers()
            # Only pull raw once processed is actually on the canvas — never blank
            # the view if the re-create failed (belt-and-suspenders, audit).
            have_processed = bool(getattr(self, "_processed_store_layers", None)
                                  or getattr(self, "_processed_layers", None))
            if have_processed and getattr(self, "_source_layers_on_canvas", False):
                self._remove_source_layers_from_viewer()  # removes raw, keeps refs
        finally:
            self._swapping_source_display = False
            self._view_swap_busy_hide()
        self._refresh_view_toggle_state()
        # Item 4: the ⟨processed⟩ store is now the active canvas view — engage the
        # progressive engine on it exactly as the raw load path does (a no-op in
        # 2D / when progressive is off / for the in-memory regime; defers its
        # native->progressive build one tick when in 3D + wanted).  Single wiring
        # point covering every `_show_processed_view` caller (finish + toggle).
        try:
            self._maybe_swap_progressive_on_ndisplay()
        except Exception:  # noqa: BLE001 — display nicety, never fatal
            pass

    def _show_raw_view(self):
        """Show the raw source display as the ONLY view — a CLEAN SWAP: REMOVE the
        ⟨processed⟩ set from the canvas (both regimes) and re-add the raw source
        layers so only the viewed set remains in the layer list (mutually
        exclusive).

        `_reshow_source_layers` re-adds the raw layers FRESH (it routes through
        `_add_source_display_layers`, which re-engages progressive + the 3D-detail
        pin for the current ndisplay), so there is no separately-paused loader to
        kick — the raw set comes back at the right detail by construction (the old
        hide/show path's `_kick_raw_loaders_if_3d` is no longer needed).  The
        churn runs under `_swapping_source_display` so it does NOT rebuild the ②
        recipe panel mid-swap.  Best-effort; never raises."""
        viewer = getattr(self, "viewer", None)
        if viewer is None:
            return
        self._view_swap_busy_show("Loading raw view…")
        self._swapping_source_display = True
        try:
            self._clear_processed_store_layers()   # remove streaming ⟨processed⟩
            self._clear_processed_layers()          # remove in-memory ⟨processed⟩
            self._reshow_source_layers()            # re-add raw (idempotent)
        finally:
            self._swapping_source_display = False
            self._view_swap_busy_hide()
        self._refresh_view_toggle_state()

    def _view_swap_busy_show(self, label):
        """Best-effort marquee "Loading…" cue during a synchronous Raw ⇄ Processed
        view swap's remove/re-add churn (Brick B).  Main-thread; a no-op when a
        whole-volume Process already owns ②'s bar (it shows its own
        Rendering/Processing message + hides in its finally — the cue is for the
        USER toggle / reopen flip), and on a Qt-free fake widget (no `_busy_show`).
        Never raises."""
        if getattr(self, "_processing", False):
            return
        try:
            self.widget._busy_show(label)
            # The swap that follows runs SYNCHRONOUSLY on the main thread and
            # blocks the event loop, so without an immediate repaint the
            # show→hide never actually paints (the user sees nothing).  Repaint
            # ONLY the bar widget (not `processEvents`, which would reenter the
            # swap — the progressive `_paint_loading` crash class) so "Loading…"
            # is visible before the blocking re-add.
            bar = getattr(self.widget, "source_progress_bar", None)
            if bar is not None:
                bar.repaint()
        except Exception:
            pass

    def _view_swap_busy_hide(self):
        """Rest ②'s bar after a view swap (pairs with :meth:`_view_swap_busy_show`;
        skipped while a Process owns the bar).  Best-effort; never raises."""
        if getattr(self, "_processing", False):
            return
        try:
            self.widget._busy_hide()
        except Exception:
            pass

    def _readd_processed_layers(self):
        """Re-create the ⟨processed⟩ display layers from `_processed_redisplay`
        (set at each Process finish) so a flip back to Processed after a clean
        swap to raw restores the exact set without recomputing.

        Routes to the SAME display method the original finish used — the streaming
        store re-opens lazily off disk; the in-memory set re-adds the cached
        display pyramid.  These display methods are PURE (they no longer call
        `_show_processed_view`), so this cannot recurse.  Guarded; never raises."""
        spec = getattr(self, "_processed_redisplay", None)
        if not spec:
            return
        # The other door onto the same display methods: a queued Raw⇄Processed
        # toggle (button or the `\` keybind) landing during / after teardown
        # would re-add the layers to another step's canvas.
        if getattr(self, "_left_step1", False):
            return
        try:
            kind = spec.get("kind")
            if kind == "store":
                self._display_processed_store(
                    spec.get("store_path"), spec.get("names"))
            elif kind == "mem":
                self._add_processed_layers(
                    spec.get("channel_levels"), spec.get("names"),
                    spec.get("spacing"))
        except Exception as exc:  # noqa: BLE001
            self._log("STEP1-PROCESS", f"re-add processed layers failed: {exc}")

    def _toggle_raw_processed_view(self, *_args):
        """Flip the Raw ⇄ Processed VIEW (the ``\\`` keybind target + the
        segmented-toggle action).  No-op when no processed view exists.  Accepts
        the napari-passed viewer arg via ``*_args``."""
        if not self._has_processed_view():
            return
        if self._is_processed_view_showing():
            self._show_raw_view()
        else:
            self._show_processed_view()

    def _bind_view_toggle_key(self, enable: bool):
        """Bind ``\\`` to the view toggle while a processed view exists; unbind
        it otherwise.  Guarded for a viewer without ``bind_key``
        (Qt-free fake).  Best-effort; never raises."""
        viewer = getattr(self, "viewer", None)
        bind = getattr(viewer, "bind_key", None) if viewer is not None else None
        if not callable(bind):
            return
        already = getattr(self, "_view_toggle_key_bound", False)
        try:
            if enable and not already:
                bind("\\", self._toggle_raw_processed_view, overwrite=False)
                self._view_toggle_key_bound = True
            elif not enable and already:
                bind("\\", None, overwrite=True)
                self._view_toggle_key_bound = False
        except Exception:
            pass

    def _refresh_view_toggle_state(self):
        """Push the ② Raw / Processed toggle's enabled + checked state to the
        widget from CURRENT truth: whether a processed view exists + which set is
        visible (read off ``.visible``).  Also (un)binds the ``\\``
        key.  Best-effort; safe when the widget/viewer lacks the toggle (Qt-free
        fakes) — called after Process + on source load / unload / teardown."""
        has_processed = self._has_processed_view()
        showing_processed = has_processed and self._is_processed_view_showing()
        self._bind_view_toggle_key(has_processed)
        widget = getattr(self, "widget", None)
        setter = getattr(widget, "_set_view_toggle_state", None)
        if callable(setter):
            try:
                setter(has_processed, showing_processed)
            except Exception:
                pass

    def cancel_process_whole_volume(self):
        """Discard the in-flight whole-volume process and unlock the surface NOW.

        Bumps the token (so the finished/error handler goes stale + does nothing),
        asks BOTH regimes' workers to stop (in-memory + streaming — whichever is
        active), and releases the lock + restores the button immediately.  The
        streaming engine is cooperatively cancellable (it removes its own partial
        ``.tmp`` store), so a streaming cancel leaves NO half-store; the in-memory
        compute still finishes on its own but its result is dropped by the
        stale-token guard.  Mirrors :meth:`cancel_large_data_prep`."""
        self._process_token += 1
        for attr in ("_process_worker", "_process_store_worker"):
            worker = getattr(self, attr, None)
            if worker is not None:
                try:
                    worker.request_cancel()
                except Exception:
                    pass
        try:
            self.widget._busy_hide()
        except Exception:
            pass
        self._set_step1_busy(False)
        self._processing = False
        self._set_process_button_cancel(False)
        self._set_preview_status("Processing cancelled.", ok=False)
        self._log("STEP1-PROCESS", "cancelled")

    def _set_process_button_cancel(self, active: bool) -> None:
        """Flip the "Process whole volume" button to 'Cancel' (kept clickable)
        while a run is in flight, or back to its idle label when it finishes.
        Best-effort; touches only the button text/icon/enabled state (mirrors
        :meth:`_set_prepare_button_cancel`)."""
        btn = getattr(self.widget, "apply_preprocess_button", None)
        if btn is None:
            return
        try:
            if active:
                if self._process_button_text is None:
                    self._process_button_text = btn.text()
                    try:
                        self._process_button_icon = btn.icon()
                    except Exception:
                        self._process_button_icon = None
                btn.setEnabled(True)          # stays clickable as Cancel
                btn.setText("Cancel")
                try:
                    from napari_epicycle.ui_shell.theme import apply_button_icon
                    apply_button_icon(btn, "mdi6.close-circle-outline")
                except Exception:
                    pass
            elif self._process_button_text is not None:
                btn.setText(self._process_button_text)
                if self._process_button_icon is not None:
                    try:
                        btn.setIcon(self._process_button_icon)
                    except Exception:
                        pass
                self._process_button_text = None
                self._process_button_icon = None
        except Exception:
            pass

    def _on_process_computed(self, channel_levels, new_spacing, bounds, stamp,
                             names, recipe_json, token):
        """Main-thread PROCESS finish: cache the full-res result + render the
        ⟨processed⟩ whole-volume layers (WYSIWYG).

        *channel_levels* is a per-channel display pyramid (level 0 = the full-res
        result; a single-element list when the channel is small enough to render
        single-scale).  The CACHE keeps level 0 (full-res, for a later extract);
        the DISPLAY adds the pyramid so 3D stays safe on a big volume.

        Discards a STALE run (Cancel / source-switch since launch — the Step-3 QC
        stale-finish guard).  Keeps the surface busy-locked THROUGH the render
        tail so the residual freeze reads as "rendering", released in the finally."""
        if token != getattr(self, "_process_token", 0):
            self._log("STEP1-PROCESS", f"discard stale result token={token}")
            return
        try:
            self.widget._busy_show("Rendering processed volume…")
        except Exception:
            pass
        try:
            # In-memory cache (NOT persisted in this brick).  `source_id` is
            # best-effort bookkeeping so a later brick can invalidate on switch.
            # Cache the FULL-RES level-0 per channel (the display pyramid's coarser
            # levels are display-only).
            processed_full = [
                (lv[0] if isinstance(lv, (list, tuple)) and lv else lv)
                for lv in channel_levels
            ]
            self._processed_cache = {
                "source_id": self._current_source_id(),
                "recipe_json": recipe_json,
                "processed": processed_full,
                "spacing": new_spacing,
                "bounds": bounds,
                # The channel names the `bounds` windows were measured FOR — a
                # window is per-channel, so a reader must be able to check it
                # belongs to the channels it is about to normalise.
                "names": [str(n) for n in (names or [])],
                "stamp": stamp,
            }
            n_layers, any_coarse = 0, False
            try:
                n_layers, any_coarse = self._add_processed_layers(
                    channel_levels, names, new_spacing)
            except Exception as exc:
                self._log("STEP1-PROCESS",
                          f"processed layer render FAILED token={token}: {exc}")
            if n_layers:
                # Mark processed available + remember how to re-create it (the
                # in-memory display pyramid), then clean-swap to the processed
                # view (removes raw from the canvas).
                self._processed_available = True
                self._processed_redisplay = {
                    "kind": "mem", "channel_levels": channel_levels,
                    "names": names, "spacing": new_spacing}
                self._show_processed_view()
                msg = (
                    f"Processed: whole volume added as {n_layers} ⟨processed⟩ "
                    "layer(s).  Showing processed — use the Raw / Processed toggle "
                    "(or the \\ key) to switch back.  Held in memory; not yet saved "
                    "to disk.")
                if any_coarse:
                    # Coarse-3D honesty guardrail: a simplified 3D
                    # view can hide the very artefacts preprocessing reveals, so
                    # steer fine judgement to 2D / a full-res region preview.
                    msg += (
                        "  This is a big image — the 3D view is simplified so it "
                        "stays smooth; 2D shows the full detail, and a region "
                        "Preview is always full-resolution for fine judgement.")
                self._set_preview_status(msg, ok=True)
            else:
                self._set_preview_status(
                    "Processed whole volume — held in memory "
                    "(not yet saved to disk).",
                    ok=True)
        finally:
            # One terminal funnel — release the lock, rest the bar, restore the
            # button — so a render failure cannot leave the surface stuck.
            self._set_step1_busy(False)
            self._processing = False
            self._set_process_button_cancel(False)
            try:
                self.widget._busy_hide()
            except Exception:
                pass

    def _on_process_error(self, msg, token):
        """Main-thread PROCESS error: surface a themed status, clear the lock +
        restore the button.  Discards a stale run's error too (a superseded /
        cancelled worker's terminal emit)."""
        if token != getattr(self, "_process_token", 0):
            return
        self._set_step1_busy(False)
        self._processing = False
        self._set_process_button_cancel(False)
        try:
            self.widget._busy_hide()
        except Exception:
            pass
        if str(msg) != "cancelled":
            self._set_preview_status(f"Process failed: {msg}", ok=False)
        self._log("STEP1-PROCESS", f"error token={token}: {msg}")

    def _on_process_progress(self, fraction, label, indeterminate, token=None):
        """Drive the shared busy bar from the worker's cost-weighted progress
        (main thread).  Marquee for a whole-array op (no fine progress); a moving
        percent + rough time-left for a tiled op.

        Ignored once the run is no longer active, and — Brick 5 fix 2 — once its
        launch token has been superseded: `_processing` goes back ON for the next
        run, so the flag alone let a draining worker paint the new run's bar."""
        if not getattr(self, "_processing", False):
            return
        if token is not None and token != getattr(self, "_process_token", None):
            return
        try:
            if indeterminate:
                self.widget._busy_show(label)
            else:
                self.widget._busy_progress(
                    label, int(round(float(fraction) * 1000)), 1000)
        except Exception:
            pass

    def _cleanup_process_thread(self, thread):
        """Drop the finished process (thread, worker) pair once its QThread has
        actually stopped (`thread.finished`).  Idempotent; touches only
        controller thread state."""
        if thread is getattr(self, "_process_thread", None):
            self._process_thread = None
            self._process_worker = None
        self._draining_process_threads = [
            (t, w) for (t, w) in getattr(self, "_draining_process_threads", [])
            if t is not thread
        ]

    def _current_source_id(self):
        """Best-effort STABLE ``source_id`` for the currently-loaded source (cache
        bookkeeping only — this brick neither persists nor reads it back).
        Returns ``None`` when nothing is loaded or the registry can't be read."""
        sid = getattr(self, "_loaded_source_id", None)
        if sid:
            return sid
        if not getattr(self, "_loaded_source_file", None):
            return None
        try:
            ws = self.widget.output_file_path.text().strip()
            state = load_state(ws) if ws and os.path.isdir(ws) else None
        except Exception:
            return None
        return self._loaded_source_id_for(state) if state else None

    def _refresh_library_from_registry(self):
        """Populate ④'s Library list from the PERSISTED patch registry (F2).

        Reads ``step1_state.json``'s ``patches[]`` so ④ shows EVERY saved patch
        across sessions — not just this session's in-memory stage.  A row whose
        ``source_id`` matches the loaded source is INTERACTIVE (selectable, and
        tickable while it is still staged in memory); rows from OTHER sources —
        and legacy rows with ``source_id == ""`` — render greyed + non-selectable
        (the ``BoxListPanel`` reads the ``LIBRARY_ROLE_*`` item data this method
        stamps).  Any in-memory staged patch NOT yet in the registry (a
        persist-failed edge) is merged in; rows are deduped by ``patch_name``
        (case-folded, matching the registry's own NTFS-safe dedup).

        Best-effort + guarded — a read failure leaves the list untouched rather
        than raising into a load / section-enter / extract caller."""
        from qtpy.QtWidgets import QListWidgetItem
        lst = getattr(self.widget, "extracted_roi_list", None)
        if lst is None:
            return
        try:
            ws = self.widget.output_file_path.text().strip()
        except Exception:
            ws = ""
        try:
            state = load_state(ws) if ws and os.path.isdir(ws) else empty_state()
        except Exception:
            state = empty_state()
        rows = all_patches(state)

        # Resolve the loaded source's id off the SAME state (one disk read).
        # ID-FIRST (Brick 4a): this is the ④ grey-out.  Keying it on the session
        # PATH greyed the whole library out whenever the raw file moved, was
        # renamed, or the workspace was copied to another machine.
        current_sid = self._loaded_source_id_for(state)

        prev = lst.currentItem().text() if lst.currentItem() is not None else None

        first_interactive = None
        target_item = None
        # Block the LIST'S MODEL, not just the QListWidget: `BoxListPanel.refresh`
        # is wired to `list.model().rowsInserted/rowsRemoved/modelReset` (a
        # SEPARATE QObject), so `lst.blockSignals` alone does NOT stop the
        # per-addItem refresh storm (O(N²) rebuild) — the same class of problem as
        # the navigator's selection refresh.  Block the model so we fill in one shot + the
        # explicit `refresh()` below reconciles once.
        model = lst.model()
        was = lst.blockSignals(True)
        was_model = model.blockSignals(True)
        try:
            lst.clear()
            seen = set()

            def _add(name, sid, rel, sidecar):
                nonlocal first_interactive, target_item
                if not name:
                    return
                key = str(name).casefold()
                if key in seen:
                    return
                seen.add(key)
                interactive = bool(current_sid) and sid == current_sid
                item = QListWidgetItem(str(name))
                item.setData(LIBRARY_ROLE_SOURCE_ID, sid or "")
                item.setData(LIBRARY_ROLE_REL_PATH, rel or "")
                item.setData(LIBRARY_ROLE_SIDECAR, sidecar or "")
                item.setData(LIBRARY_ROLE_INTERACTIVE, interactive)
                if not interactive:
                    # Enabled (so the row card still renders) but NOT selectable
                    # / checkable — the panel greys it and the shell can't pick
                    # it.  `make_checkable` only ever touches ItemIsUserCheckable,
                    # so this survives every refresh.
                    item.setFlags(Qt.ItemIsEnabled)
                lst.addItem(item)
                if interactive and first_interactive is None:
                    first_interactive = item
                if prev is not None and str(name) == prev and interactive:
                    target_item = item

            for row in rows:
                _add(row.get("patch_name"), row.get("source_id", ""),
                     row.get("relative_path", ""), row.get("sidecar_path", ""))
            # Merge any in-memory staged patch not yet persisted (persist-failed
            # edge): current source, canonical on-disk paths.
            for name in (getattr(self, "extracted_patches", {}) or {}):
                _add(name, current_sid or "",
                     f"artifacts/step1/patches/{name}.tiff",
                     f"artifacts/step1/patches/{name}_metadata.json")
        finally:
            model.blockSignals(was_model)
            lst.blockSignals(was)

        # Signals were blocked during the fill, so drive the panel's rebuild +
        # count once here.
        try:
            self.widget.library_box_list.refresh()
        except Exception:
            pass

        # Restore selection to the previously-selected interactive row, else the
        # first interactive row (mirrors the extract path's setCurrentRow(0)).  A
        # non-selectable (other-source) row is never auto-selected.
        pick = target_item or first_interactive
        if pick is not None:
            lst.setCurrentItem(pick)

    # ------------------------------------------------------------------
    # ④ "Show patch locations" (Brick 2) — draw saved-patch bounding boxes in
    # the SOURCE volume, in a SEPARATE overlay from the ③ box tool (whose
    # `boxes` list drives drawing + extraction).  Keeping them separate means a
    # location box is never re-extracted and never pollutes ③'s transient list.
    # The overlay is scene-attached + subscribes to dims itself, so it tracks
    # the camera + 2D/3D flips without per-frame redraw wiring.
    # ------------------------------------------------------------------
    def _current_source_patch_names(self):
        """Saved-patch names belonging to the CURRENTLY-LOADED source (registry
        order).  [] when nothing is loaded / no workspace / read fails."""
        try:
            ws = self.widget.output_file_path.text().strip()
        except Exception:
            return []
        if not ws or not os.path.isdir(ws):
            return []
        try:
            state = load_state(ws)
        except Exception:
            return []
        current_sid = self._loaded_source_id_for(state)   # id-first (Brick 4a)
        if not current_sid:
            return []
        names = []
        for row in all_patches(state):
            nm = row.get("patch_name")
            if nm and row.get("source_id", "") == current_sid:
                names.append(nm)
        return names

    def _loc_full_shape(self):
        """The source's full (Z, Y, X) for mapping sidecar boxes — reuses the box
        tool's already-seeded shape (set on load)."""
        bt = getattr(self, "box_tool", None)
        if bt is None:
            return None
        shape = getattr(bt, "_full_shape", None)
        if shape is None:
            try:
                shape = bt._resolve_full_shape()
            except Exception:
                shape = None
        return shape

    def _ensure_loc_overlay(self):
        """The ④ location-box overlay (created lazily on the main canvas)."""
        if getattr(self, "_loc_overlay", None) is not None:
            return self._loc_overlay
        bt = getattr(self, "box_tool", None)
        if bt is None:
            return None
        try:
            qtv = bt._main_qt_viewer()
        except Exception:
            qtv = None
        if qtv is None:
            self._log("PATCHLOC", "no main QtViewer yet — cannot draw locations")
            return None
        try:
            from ._box_overlay import Step1BoxOverlay
            ov = Step1BoxOverlay(qtv, style="corners")
            self._loc_overlay = ov
            # Distinct from ③'s transient teal boxes (magenta default), and
            # re-apply any remembered Render/style so the look PERSISTS across a
            # source-switch overlay recreation (audit MED-3).
            self._apply_loc_style_to_overlay(ov)
        except Exception as exc:
            self._log("PATCHLOC", f"overlay create failed: {exc}")
            self._loc_overlay = None
        # App-quit teardown, wired ONCE (audit HIGH-1): the location overlay lives
        # on the SHARED napari canvas, so without an aboutToQuit clear its vispy
        # Lines are still parented at Qt/vispy teardown -> the freed-VBO
        # glDrawArrays access-violation this whole overlay rework exists to avoid.
        if self._loc_overlay is not None and not getattr(
                self, "_loc_quit_hook_connected", False):
            try:
                from qtpy.QtWidgets import QApplication
                app = QApplication.instance()
                if app is not None:
                    app.aboutToQuit.connect(self._clear_patch_locations)
                    self._loc_quit_hook_connected = True
            except Exception:
                pass
        return self._loc_overlay

    # ---- downstream-used guard (Brick 4a Part 2) -----------------------
    def _downstream_for(self, patch_name):
        """What *patch_name* is used for in Steps 2 + 3.  Guarded, UNCACHED.

        Brick 4a keyed a cache on the mtime of ``artifacts/step2`` and
        ``artifacts/step3``.  That key does not work: a directory's mtime only
        changes when its OWN entries change, so a Step 3 run created under
        ``runs/<slug>/`` or a slice accepted into ``round_NN/accepted/`` -- two
        and three levels down -- left both roots untouched.  The cache then
        served a stale ``empty()`` and the user got the PLAIN confirm on a patch
        that was very much in use, which is exactly the accident this guard
        exists to prevent.  (The step-enter / extract / trim / delete
        invalidation the design promised was never wired either.)

        So: no cache.  Both call sites are user-initiated modals, the scan
        short-circuits on its first hit, and a wrong "nothing downstream" is far
        more expensive than a directory walk.

        Any failure answers "nothing downstream": a scan must never be able to
        block a legitimate trim or delete."""
        try:
            from ._downstream import downstream_uses

            ws = self.widget.output_file_path.text().strip()
            return downstream_uses(ws, patch_name)
        except Exception as exc:   # noqa: BLE001
            self._log("STEP1", f"downstream scan skipped: {exc}")
            from ._downstream import DownstreamUse
            return DownstreamUse.empty()

    def _confirm_downstream(self, patch_name, use, action):
        """The shared "already used later" confirm, default **Cancel**.

        *action* is ``"trim"`` or ``"delete"``.  Returns True to go ahead.
        Never cascades: Step 1 changes nothing in Steps 2 or 3, and says so."""
        from qtpy.QtWidgets import QMessageBox

        used_by = use.summary() or "work in Steps 2 or 3"
        # Step 4 is named when it holds work too.
        s4 = bool(getattr(use, "step4_kinds", ()))
        steps = "Steps 2, 3 or 4" if s4 else "Steps 2 or 3"
        if action == "trim":
            body = (
                f"{patch_name} is used by {used_by}.  Trimming changes the "
                "slice numbering, so that work will no longer line up with "
                "this patch.\n\nStep 1 will not change or delete anything in "
                f"{steps} — you would have to redo the affected work "
                "yourself.")
            go = "Trim anyway"
        else:
            body = (
                f"{patch_name} is used by {used_by}.  Deleting removes the "
                "image from Step 1 only.  The work in "
                f"{'Steps 2, 3 and 4' if s4 else 'Steps 2 and 3'} stays on "
                "disk but will point at a patch that no longer exists.\n\nTo "
                "remove it there too, delete that patch's run in Step 3 and "
                "its slices in Step 2 yourself."
                + (" Its Step 4 results stay on disk under artifacts/step4 "
                   "(the state file and the stack's run folder); Step 4 has "
                   "no way to remove them." if s4 else "")
                )
            go = "Delete anyway"

        box = QMessageBox(self.widget)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("This patch is already used later")
        box.setText("This patch is already being used later in the pipeline.")
        box.setInformativeText(body)
        go_btn = box.addButton(go, QMessageBox.AcceptRole)
        cancel_btn = box.addButton("Cancel", QMessageBox.RejectRole)
        box.setDefaultButton(cancel_btn)          # default is always Cancel
        box.exec_()
        return box.clickedButton() is go_btn

    def _on_delete_patch(self, patch_name):
        """④ bin: delete a patch after a single confirm — its registry row +
        on-disk TIFF/sidecar + session stage + box signature + location overlay.

        Registry-FIRST (the row vanishes from ④ immediately), THEN best-effort
        file delete — a leftover file is a harmless orphan once its row is gone,
        and if the registry save fails nothing is removed (safe)."""
        if not patch_name:
            return
        # One Step-1 op at a time: an `os.remove` of the patch's TIFF/sidecar
        # concurrent with a trim's `os.replace` (or an extract's list rebuild)
        # races the on-disk state → orphan files / registry-vs-disk drift.  Block
        # while the surface is busy / a trim is in flight or launching (audit MED-1).
        if (getattr(self, "_step1_busy", False)
                or getattr(self, "_trim_running", False)
                or getattr(self, "_trim_launching", False)):
            self._set_export_status(
                "Please wait for the current operation to finish, then delete.",
                success=False)
            return
        # Brick 4a Part 2: after the busy guard, before the modal.  A patch
        # Steps 2/3 already use gets a different confirm (default Cancel); the
        # scan is wrapped so a failure can never block a legitimate delete.
        use = self._downstream_for(patch_name)
        if use.any:
            if not self._confirm_downstream(patch_name, use, "delete"):
                return
        else:
            from qtpy.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                self.widget, "Delete patch?",
                f"Delete '{patch_name}'?\n\nThis removes it from disk and the "
                "library — it cannot be undone.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
        self._intensity_curve_cache.pop(patch_name, None)   # gone → drop its profile
        self._open_capture_patches.discard(patch_name)      # no stale lit-camera
        if getattr(self, "_highlighted_patch", None) == patch_name:
            self._highlighted_patch = None                  # no dangling highlight
        try:
            ws = self.widget.output_file_path.text().strip()
        except Exception:
            ws = ""
        if not ws or not os.path.isdir(ws):
            self._set_export_status("No workspace — cannot delete.", success=False)
            return
        from ._state import load_state, remove_patch, save_state
        try:
            state = load_state(ws)
        except Exception as exc:
            self._set_export_status(
                f"Delete failed to read state: {exc}", success=False)
            return
        target = str(patch_name).casefold()
        row = next((p for p in state.get("patches", [])
                    if str(p.get("patch_name", "")).casefold() == target), None)
        # Resolve the file paths from the row BEFORE it is dropped.
        tiff_abs = sidecar_abs = None
        if row is not None:
            rel, sc = row.get("relative_path"), row.get("sidecar_path")
            tiff_abs = os.path.join(ws, rel) if rel else None
            sidecar_abs = os.path.join(ws, sc) if sc else None
        # 1) registry-first: drop the row + save so ④ reflects it immediately.
        if remove_patch(state, patch_name):
            try:
                save_state(ws, state)
            except Exception as exc:
                self._set_export_status(
                    f"Delete failed to save: {exc}", success=False)
                return
        # Free the committed box signature (reads the sidecar bbox, still on disk)
        # so a later re-draw + re-extract of that region is not skipped as
        # already-captured.  Done AFTER the save so a failed save above leaves
        # everything — sig included — intact.
        self._free_box_sig_for_patch(sidecar_abs)
        # 2) session stage + location-visibility set.
        try:
            self.extracted_patches.pop(patch_name, None)
        except Exception:
            pass
        try:
            self._visible_patch_locations.discard(patch_name)
        except Exception:
            pass
        # 3) best-effort file delete (orphans are harmless now the row is gone).
        # Containment guard: only remove a path that resolves INSIDE the
        # workspace (register_patch rejects absolute paths on WRITE; the read
        # side stays defensive against a hand-edited state.json — audit LOW).
        ws_abs = os.path.abspath(ws)
        for path in (tiff_abs, sidecar_abs):
            if not path or not os.path.exists(path):
                continue
            if not os.path.abspath(path).startswith(ws_abs + os.sep):
                self._log("STEP1-DELETE",
                          f"refusing to remove out-of-workspace path {path}")
                continue
            try:
                os.remove(path)
            except Exception as exc:
                self._log("STEP1-DELETE", f"could not remove {path}: {exc}")
        # 4) refresh ④ + counts + the overlay — DEFERRED to the next event-loop
        # pass (QTimer.singleShot(0), audit MED).  The rebuild does `lst.clear()`,
        # which destroys the very trash button whose `clicked` slot we are still
        # inside; tearing a widget down mid-signal-emit is this project's known
        # crash class.  The deletion above already completed synchronously —
        # only the widget rebuild waits.
        def _post_delete_refresh():
            for fn in (self._refresh_source_patch_counts,
                       self._refresh_library_from_registry,
                       self._rebuild_loc_boxes):
                try:
                    fn()
                except Exception:
                    pass
        try:
            from qtpy.QtCore import QTimer
            QTimer.singleShot(0, _post_delete_refresh)
        except Exception:
            _post_delete_refresh()   # no Qt event loop (headless) → run inline
        self._set_export_status(f"Deleted '{patch_name}'.", success=True)
        self._log("STEP1-DELETE", f"deleted {patch_name}")

    def _free_box_sig_for_patch(self, sidecar_abs):
        """Best-effort: read a deleted patch's frozen sidecar bbox + Z-range and
        drop its committed box signatures (XY + Z-inclusive) so its region can be
        re-extracted rather than silently skipped as already-captured."""
        if not sidecar_abs or not os.path.exists(sidecar_abs):
            return
        try:
            with open(sidecar_abs, "r", encoding="utf-8") as f:
                meta = json.load(f)
            bbox = meta.get("bounding_box")
            if not (isinstance(bbox, list) and len(bbox) == 4):
                return
            zr = meta.get("selected_z_range")
            zr = (list(zr) if isinstance(zr, (list, tuple)) and len(zr) == 2
                  else None)
            sigs = getattr(self, "_committed_box_sigs", None)
            if sigs is None:
                return
            sigs.discard(box_signature(bbox))
            if zr is not None:
                sigs.discard(box_signature(bbox, zr))
        except Exception as exc:
            self._log("STEP1-DELETE", f"box-sig free failed: {exc}")

    def _boxrec_for_patch(self, patch_name, full_shape):
        """A box-tool BoxRec for one saved patch, from its frozen sidecar."""
        try:
            ws = self.widget.output_file_path.text().strip()
        except Exception:
            return None
        side = os.path.join(ws, "artifacts", "step1", "patches",
                            f"{patch_name}_metadata.json")
        try:
            with open(side, "r", encoding="utf-8") as f:
                meta = json.load(f)
            bbox = meta.get("bounding_box")
            if not (isinstance(bbox, list) and len(bbox) == 4):
                return None
            zr = meta.get("selected_z_range")
            zr = (list(zr) if isinstance(zr, (list, tuple)) and len(zr) == 2
                  else None)
            return self._boxrec_from_sidecar(
                np.array(bbox, dtype=float), zr, full_shape)
        except Exception as exc:
            self._log("PATCHLOC", f"{patch_name}: sidecar read failed: {exc}")
            return None

    def _rebuild_loc_boxes(self):
        """Redraw the ④ location overlay from `_visible_patch_locations`.

        Main-thread and O(visible boxes): each box costs a `_boxrec_for_patch`
        sidecar/state lookup plus one workspace-state read for the styles, and
        it re-runs on every eye toggle.  Brick 5 fix 4 measures it under
        `EPICYCLE_SYNC_DEBUG=1` (`[STEP1-PERF]`) instead of guessing — a
        workspace with hundreds of patches is the only case that could cross the
        budget, and that needs a real workspace to judge.
        """
        _perf = os.environ.get("EPICYCLE_SYNC_DEBUG") == "1"
        _t0 = time.perf_counter() if _perf else 0.0
        try:
            self._rebuild_loc_boxes_inner()
        finally:
            if _perf:
                self._log(
                    "STEP1-PERF",
                    f"patch-location overlay rebuild "
                    f"{(time.perf_counter() - _t0) * 1000.0:.1f} ms "
                    f"({len(getattr(self, '_visible_patch_locations', ()) or ())}"
                    f" visible)")

    def _rebuild_loc_boxes_inner(self):
        """The rebuild itself (see :meth:`_rebuild_loc_boxes`, which times it)."""
        names = [n for n in self._current_source_patch_names()
                 if n in self._visible_patch_locations]
        ov = self._ensure_loc_overlay()
        full_shape = self._loc_full_shape()
        if ov is None or full_shape is None:
            self._log("PATCHLOC",
                      f"cannot render (overlay={ov is not None} "
                      f"shape={full_shape is not None})")
            return
        pairs = [(n, self._boxrec_for_patch(n, full_shape)) for n in names]
        pairs = [(n, b) for n, b in pairs if b is not None]
        boxrecs = [b for _, b in pairs]
        # P2 Brick 2: each box renders in its resolved per-patch style (colour +
        # shape via a per-vertex array on the ONE Line; width global) — parallel
        # to boxrecs, None entries → the overlay's built-in default.
        styles = self._resolve_loc_box_styles([n for n, _ in pairs])
        # Double-click HIGHLIGHT: draw the highlighted patch's box WHITE (width is
        # global, so colour is the emphasis) so it pops among the coloured boxes.
        hl = getattr(self, "_highlighted_patch", None)
        if hl is not None:
            for i, (n, _b) in enumerate(pairs):
                if n == hl:
                    prev = styles[i] if isinstance(styles[i], dict) else {}
                    styles[i] = {"color": [1.0, 1.0, 1.0, 1.0], "opacity": 1.0,
                                 "shape": prev.get("shape", "corners")}
                    break
        # Physical-µm: the location boxes store RAW-full-res voxel coords, so the
        # overlay multiplies by the RAW source voxel (the current source's live
        # spacing).  Set it HERE (not only in `_ensure_loc_overlay`, which
        # early-returns the cached instance and so never refreshes after a voxel
        # edit — grill 1 L1).  Both raw + processed views render the same physical
        # volume, so the box stays glued in either.
        try:
            ov.set_voxel_size(self._preview_spacing())
        except Exception:
            pass
        try:
            ov.update(boxrecs, None, full_shape, styles)
        except Exception as exc:
            self._log("PATCHLOC", f"overlay update failed: {exc}")
        self._log("PATCHLOC",
                  f"drawing {len(boxrecs)} location box(es): {names}")

    def _resolve_loc_box_styles(self, names):
        """Resolve each patch's effective ``box_style`` (patch → source default →
        None) for the ④ location overlay.  Loads the workspace state ONCE.
        Returns a list parallel to *names* (None → the overlay's built-in
        default).  Guarded — never raises."""
        try:
            ws = self.widget.output_file_path.text().strip()
            state = self._cached_step1_state(ws) if ws and os.path.isdir(ws) else None
        except Exception:
            state = None
        if state is None:
            return [None] * len(names)
        out = []
        for n in names:
            try:
                out.append(resolve_box_style(state, n))
            except Exception:
                out.append(None)
        return out

    def _refresh_library_list(self):
        """Rebuild the ④ Library rows (best-effort) so per-row button states
        (capture-open cyan, swatch tint, eye) re-render."""
        try:
            self.widget.library_box_list.refresh()
        except Exception:
            pass

    def _capture_open_for(self, patch_name):
        """Whether *patch_name*'s capture inspector is currently open (drives the
        row camera button's lit-cyan state)."""
        return patch_name in getattr(self, "_open_capture_patches", set())

    def _clear_highlight(self):
        """Drop the double-click white highlight + re-render (guarded).  A no-op
        when nothing is highlighted."""
        if getattr(self, "_highlighted_patch", None) is None:
            return
        self._highlighted_patch = None
        try:
            self._rebuild_loc_boxes()
            self._refresh_library_list()
        except Exception:
            pass

    def _on_patch_double_clicked(self, patch_name):
        """④ row double-click → HIGHLIGHT that patch's box (white) in the source
        volume so you can tell which is which without toggling eyes.  Auto-shows
        the box if it was hidden; a second double-click on a DIFFERENT patch moves
        the highlight."""
        if not patch_name:
            return
        # Toggle: double-clicking the ALREADY-highlighted patch clears it; a
        # different patch moves the highlight.  (Single-clicking a different
        # patch also clears it — see `_on_roi_selection_changed`.)
        if self._highlighted_patch == patch_name:
            self._clear_highlight()
            return
        self._highlighted_patch = patch_name
        # Auto-show: you double-clicked it, you want to see it.
        try:
            self._visible_patch_locations = (
                getattr(self, "_visible_patch_locations", set()) | {patch_name})
        except Exception:
            pass
        try:
            self._rebuild_loc_boxes()
        except Exception:
            pass
        # Sync the per-row eye (the box is now shown) + the master toggle.
        try:
            self._refresh_library_list()
            self.widget.sync_show_locations_master()
        except Exception:
            pass

    # -- ④ per-patch box styling (P2 Brick 2) --------------------------------

    def _cached_step1_state(self, ws):
        """Load the step1 state, cached by ``(ws, mtime)`` so a ④ refresh's N
        swatch reads + the loc-overlay resolve hit disk ONCE.  The style-save
        handlers null the cache explicitly (mtime resolution can be coarse), so a
        just-saved style is never read stale."""
        try:
            from ._paths import step1_state_path
            sp = step1_state_path(ws)
            mt = os.path.getmtime(sp) if os.path.exists(sp) else 0.0
        except Exception:
            mt = 0.0
        cache = getattr(self, "_step1_state_cache", None)
        if cache and cache[0] == ws and cache[1] == mt:
            return cache[2]
        try:
            state = load_state(ws)
        except Exception:
            state = None
        self._step1_state_cache = (ws, mt, state)
        return state

    def _swatch_color_for(self, patch_name):
        """The (r, g, b) 0..1 for a patch's row swatch: its resolved box colour,
        else the built-in default (magenta) when unstyled / 'status'."""
        try:
            ws = self.widget.output_file_path.text().strip()
            state = self._cached_step1_state(ws) if ws and os.path.isdir(ws) else None
            style = resolve_box_style(state, patch_name) if state else None
            color = style.get("color") if isinstance(style, dict) else None
            if isinstance(color, (list, tuple)) and len(color) >= 3:
                return (float(color[0]), float(color[1]), float(color[2]))
        except Exception:
            pass
        return _DEFAULT_LOC_RGB

    def _current_style_for(self, patch_name):
        """The patch's resolved ``box_style`` dict (or ``{}``) for popup pre-fill."""
        try:
            ws = self.widget.output_file_path.text().strip()
            state = self._cached_step1_state(ws) if ws and os.path.isdir(ws) else None
            s = resolve_box_style(state, patch_name) if state else None
            return dict(s) if isinstance(s, dict) else {}
        except Exception:
            return {}

    def _on_patch_style_changed(self, patch_name, style, refresh_list=True):
        """Persist a patch's ``box_style`` + re-render the overlay (+ optionally
        re-tint the row swatches — skipped on opacity/shape drags for speed)."""
        try:
            ws = self.widget.output_file_path.text().strip()
            if ws and os.path.isdir(ws):
                state = load_state(ws)
                set_patch_box_style(state, patch_name, style)
                save_state(ws, state)
                self._step1_state_cache = None   # just-saved → force reload
        except Exception as exc:
            self._log("BOXSTYLE", f"persist patch style: {exc}")
        self._refresh_after_style(refresh_list)

    def _on_apply_style_to_source(self, patch_name, style):
        """Write the source DEFAULT + every current-source patch row to *style*."""
        try:
            ws = self.widget.output_file_path.text().strip()
            if ws and os.path.isdir(ws):
                state = load_state(ws)
                sid = self._current_source_id()
                if sid:
                    set_source_box_style(state, sid, style)
                for n in self._current_source_patch_names():
                    set_patch_box_style(state, n, style)
                save_state(ws, state)
                self._step1_state_cache = None   # just-saved → force reload
        except Exception as exc:
            self._log("BOXSTYLE", f"apply style to source: {exc}")
        self._refresh_after_style(True)

    def _refresh_after_style(self, refresh_list=True):
        try:
            if getattr(self, "_visible_patch_locations", None):
                self._rebuild_loc_boxes()
        except Exception:
            pass
        if refresh_list:
            try:
                self.widget.library_box_list.refresh()   # re-tint swatches
            except Exception:
                pass

    def _on_patch_style_clicked(self, patch_name, global_pos):
        """Open the per-patch box-style popup anchored at the row swatch."""
        try:
            popup = self._build_box_style_popup(patch_name)
            if popup is None:
                return
            popup.move(global_pos)
            popup.show()
        except Exception as exc:
            self._log("BOXSTYLE", f"style popup failed: {exc}")

    def _build_box_style_popup(self, patch_name):
        """A frameless ``Qt.Popup`` for one patch's box colour / shape / opacity +
        an "Apply to all from this source" button.  Live-applies on every change;
        no OK/Cancel (close = keep, already persisted)."""
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import (
            QWidget, QVBoxLayout, QHBoxLayout, QLabel, QToolButton, QSlider,
            QColorDialog, QButtonGroup, QRadioButton)

        cur = self._current_style_for(patch_name)
        work = {
            "color": cur.get("color"),
            "opacity": cur.get("opacity"),
            "shape": cur.get("shape") or "corners",
        }

        def _clean():
            st = {"shape": work["shape"]}
            if work["color"] is not None:
                st["color"] = work["color"]
            if work["opacity"] is not None:
                st["opacity"] = work["opacity"]
            return st

        def _apply(refresh_list=True):
            self._on_patch_style_changed(patch_name, _clean(), refresh_list)

        pop = QWidget(self.widget, Qt.Popup)
        pop.setAttribute(Qt.WA_DeleteOnClose, True)
        pop.setStyleSheet(_BOX_STYLE_POPUP_STYLE)
        lay = QVBoxLayout(pop)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        # --- colour ---
        lay.addWidget(QLabel("Colour"))
        col_row = QHBoxLayout()
        col_row.setSpacing(4)

        def _set_color(rgba):
            work["color"] = rgba
            _apply(True)

        def _pick_custom():
            c = QColorDialog.getColor(parent=pop)
            if c.isValid():
                _set_color([c.redF(), c.greenF(), c.blueF(), 1.0])

        for _nm, rgb in _BOX_STYLE_PRESETS:
            b = QToolButton()
            b.setFixedSize(20, 20)
            rr, gg, bb = (int(round(c * 255)) for c in rgb)
            b.setStyleSheet(
                f"QToolButton {{ background: rgb({rr},{gg},{bb});"
                "  border: 1px solid rgba(255,255,255,0.3); border-radius: 3px; }"
                "QToolButton:hover { border: 1px solid #ffffff; }")
            b.clicked.connect(
                lambda _=False, c=rgb: _set_color([c[0], c[1], c[2], 1.0]))
            col_row.addWidget(b)
        custom = QToolButton()
        custom.setText("⊕")
        custom.setFixedSize(20, 20)
        custom.setToolTip("Custom colour…")
        custom.clicked.connect(lambda: _pick_custom())
        col_row.addWidget(custom)
        col_row.addStretch(1)
        lay.addLayout(col_row)

        # --- shape ---
        shp_row = QHBoxLayout()
        shp_row.setSpacing(6)
        shp_row.addWidget(QLabel("Shape"))
        grp = QButtonGroup(pop)
        rb_c = QRadioButton("Corners")
        rb_f = QRadioButton("Full box")
        (rb_c if work["shape"] == "corners" else rb_f).setChecked(True)
        grp.addButton(rb_c)
        grp.addButton(rb_f)

        def _set_shape(s):
            work["shape"] = s
            _apply(False)

        rb_c.toggled.connect(lambda on: on and _set_shape("corners"))
        rb_f.toggled.connect(lambda on: on and _set_shape("wireframe"))
        shp_row.addWidget(rb_c)
        shp_row.addWidget(rb_f)
        shp_row.addStretch(1)
        lay.addLayout(shp_row)

        # --- opacity ---
        op_row = QHBoxLayout()
        op_row.setSpacing(6)
        op_row.addWidget(QLabel("Opacity"))
        sld = QSlider(Qt.Horizontal)
        sld.setRange(10, 100)
        sld.setValue(int(round(
            (work["opacity"] if work["opacity"] is not None else 0.9) * 100)))
        op_val = QLabel(f"{sld.value()}%")

        def _set_opacity(v):
            op_val.setText(f"{v}%")
            work["opacity"] = v / 100.0
            _apply(False)   # opacity doesn't change the swatch colour

        sld.valueChanged.connect(_set_opacity)
        op_row.addWidget(sld, 1)
        op_row.addWidget(op_val)
        lay.addLayout(op_row)

        # --- apply to all from this source ---
        n = len(self._current_source_patch_names())
        src = "this source"
        try:
            f = getattr(self, "_loaded_source_file", None)
            if f:
                src = os.path.splitext(os.path.basename(f))[0]
        except Exception:
            pass
        apply_btn = QToolButton()
        apply_btn.setText(f"Apply to all {n} from '{src}'")
        apply_btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        apply_btn.setStyleSheet(
            "QToolButton { background: rgba(255,255,255,0.06);"
            "  border: 1px solid rgba(255,255,255,0.15); border-radius: 4px;"
            "  padding: 4px 8px; }"
            "QToolButton:hover { background: rgba(255,255,255,0.12); }")
        apply_btn.clicked.connect(
            lambda: (self._on_apply_style_to_source(patch_name, _clean()),
                     pop.close()))
        lay.addWidget(apply_btn)

        pop.adjustSize()
        return pop

    def is_patch_location_visible(self, patch_name):
        """Whether *patch_name*'s location box is currently drawn (per-row eye)."""
        return patch_name in getattr(self, "_visible_patch_locations", set())

    def toggle_all_patch_locations(self, show):
        """④ master toggle: draw/clear EVERY current-source patch location."""
        if show:
            self._visible_patch_locations = set(
                self._current_source_patch_names())
        else:
            self._visible_patch_locations = set()
        self._log("PATCHLOC",
                  f"master toggle -> {'ON' if show else 'OFF'} "
                  f"({len(self._visible_patch_locations)} patch(es))")
        self._rebuild_loc_boxes()
        try:
            self.widget.library_box_list.refresh()   # sync the per-row eyes
            self.widget.sync_show_locations_master()
        except Exception:
            pass

    def _on_show_locations_clicked(self, *_):
        """④ master checkbox clicked: a click means "show ALL" unless everything
        is already shown, in which case "hide all" (binary intent — the tristate
        is display-only, set by :meth:`sync_show_locations_master`)."""
        names = self._current_source_patch_names()
        all_shown = bool(names) and all(
            n in getattr(self, "_visible_patch_locations", set())
            for n in names)
        self.toggle_all_patch_locations(not all_shown)

    def toggle_patch_location(self, patch_name, show=None):
        """④ per-row eye: draw/clear ONE patch location (toggles when show=None)."""
        if show is None:
            show = patch_name not in getattr(
                self, "_visible_patch_locations", set())
        if show:
            self._visible_patch_locations.add(patch_name)
        else:
            self._visible_patch_locations.discard(patch_name)
        self._log("PATCHLOC",
                  f"{patch_name} -> {'shown' if show else 'hidden'}")
        self._rebuild_loc_boxes()
        try:
            self.widget.library_box_list.refresh()
            self.widget.sync_show_locations_master()
        except Exception:
            pass

    def set_location_box_style(self, *, thickness=None, opacity=None,
                               color_name="__keep__"):
        """Brick 3: push ④'s Render/style choices (width / opacity / colour) onto
        the LOCATION overlay AND — so "box styling" is app-wide — the ③ box tool.

        `color_name` is a `_BOX_COLOR_CHOICES` key ("status colour", "teal",
        "magenta", …) or the sentinel "__keep__".  For the LOCATION overlay,
        "status colour" keeps its distinct magenta default (so location boxes
        read apart from ③'s teal, and a thickness/opacity change never silently
        recolours them); the ③ box tool still gets `color_name` verbatim."""
        # REMEMBER the choices on the controller (audit MED-3): after a source
        # switch destroys `_loc_overlay`, the next `_ensure_loc_overlay` re-applies
        # these so location boxes keep the user's width/opacity/colour instead of
        # reverting to the thin magenta default while ③'s boxes keep the style.
        if thickness is not None:
            self._loc_style_thickness = thickness
        if opacity is not None:
            self._loc_style_opacity = opacity
        if color_name != "__keep__":
            self._loc_style_color_name = color_name
        # Do NOT create the overlay just to style it (audit MED-2): a Render/style
        # slider drag must never spawn a persistent shared-canvas overlay when no
        # location is shown (that overlay would then leak with no teardown).  Style
        # only an EXISTING overlay; the stored params re-apply when it IS created.
        ov = getattr(self, "_loc_overlay", None)
        if ov is not None:
            self._apply_loc_style_to_overlay(ov)
            self._rebuild_loc_boxes()
        # Mirror onto ③'s transient boxes so all boxes share the look.
        bt = getattr(self, "box_tool", None)
        if bt is not None:
            try:
                bt.set_box_style(thickness=thickness, opacity=opacity,
                                 color_name=color_name)
            except Exception:
                pass
        self._log("BOXSTYLE",
                  f"thickness={thickness} opacity={opacity} color={color_name}")

    def _apply_loc_style_to_overlay(self, ov):
        """Apply the REMEMBERED ④ Render/style (thickness/opacity/colour) to the
        location overlay `ov`.  "status colour" (the default) keeps the distinct
        magenta so location boxes read apart from ③'s teal."""
        color_name = getattr(self, "_loc_style_color_name", "status colour")
        loc_rgb = _LOCATION_BOX_RGB
        if color_name and color_name != "status colour":
            try:
                from ._box_tool import _BOX_COLOR_CHOICES
                loc_rgb = _BOX_COLOR_CHOICES.get(color_name) or _LOCATION_BOX_RGB
            except Exception:
                loc_rgb = _LOCATION_BOX_RGB
        try:
            kw = {"color": loc_rgb}
            t = getattr(self, "_loc_style_thickness", None)
            o = getattr(self, "_loc_style_opacity", None)
            if t is not None:
                kw["thickness"] = t
            if o is not None:
                kw["opacity"] = o
            ov.set_style(**kw)
        except Exception as exc:
            self._log("BOXSTYLE", f"location style apply failed: {exc}")

    def _clear_patch_locations(self):
        """Drop all location boxes + the overlay (source switch / unload / exit).
        Best-effort; leaves `_visible_patch_locations` empty."""
        self._visible_patch_locations = set()
        ov = getattr(self, "_loc_overlay", None)
        if ov is not None:
            try:
                ov.destroy()
            except Exception:
                pass
        self._loc_overlay = None
        # Brick 5b item 4: the STATE is now empty but the WIDGETS still show the
        # old one — the ④ master checkbox stays checked / partial and the row
        # eyes stay lit, advertising boxes that no longer exist (and, since
        # step-exit calls this, they come back lit on re-entry with nothing
        # drawn).  Re-sync both; each is cosmetic, so neither may raise.
        widget = getattr(self, "widget", None)
        sync = getattr(widget, "sync_show_locations_master", None)
        if callable(sync):
            try:
                sync()
            except Exception as exc:      # noqa: BLE001
                self._log("PATCHLOC", f"master-toggle resync skipped: {exc}")
        refresh = getattr(getattr(widget, "library_box_list", None),
                          "refresh", None)
        if callable(refresh):
            try:
                refresh()
            except Exception as exc:      # noqa: BLE001
                self._log("PATCHLOC", f"row-eye resync skipped: {exc}")

    # ------------------------------------------------------------------
    # ④ Snapshot (Brick 1) — export the source view (+ patch locations) to TIFF
    # ------------------------------------------------------------------

    def _on_snapshot_clicked(self, *_):
        """④ Snapshot: save a WYSIWYG image of the CURRENT main-canvas view —
        EXACTLY what is on screen (whatever ndisplay / channels / boxes are shown)
        — to a TIFF via ``viewer.screenshot(canvas_only=True, scale=N)``.

        The user wants the current view captured as-is, so this does NOT force 2D,
        does NOT auto-show the location boxes, and does NOT touch the scale bar —
        the LIVE "Scale bar" control owns it, and a WYSIWYG shot captures whatever
        it is showing (font, µm/px, colour) verbatim.  The ONLY thing this mutates
        is `camera.zoom` for the N× super-sample, restored in a try/finally on
        EVERY path (success / cancel / error).  The capture is on the GL/main
        thread (not threaded); a brief status cue covers the one-shot render."""
        # Gate on the shared Step-1 busy flags (mirror `_on_delete_patch`): the
        # capture touches the shared canvas, so never overlap an extract / trim.
        if (getattr(self, "_step1_busy", False)
                or getattr(self, "_trim_running", False)
                or getattr(self, "_trim_launching", False)):
            self._set_export_status(
                "Please wait for the current operation to finish, then "
                "snapshot.", success=False)
            return
        viewer = getattr(self, "viewer", None)
        shot_fn = getattr(viewer, "screenshot", None) if viewer else None
        if not callable(shot_fn):
            self._set_export_status(
                "This napari build has no screenshot — cannot snapshot.",
                success=False)
            return

        # ── Read the ▾ popover — just the resolution now.  The scale bar is the
        #    LIVE "Scale bar" control's; the snapshot captures it WYSIWYG (font,
        #    µm, colour and all) and reconfigures NOTHING. ──
        w = self.widget
        scale_factor = 2
        try:
            scale_factor = int(w.snapshot_scale_spin.value())
        except Exception:
            pass
        cam = getattr(viewer, "camera", None)
        prior_zoom = None
        prior_font = None

        try:
            # WYSIWYG: capture the canvas EXACTLY as shown — the live scale bar
            # (its own control) included, verbatim; the ONLY compensation is the
            # scale-bar font at scale>1 (below), so the exported bar keeps its
            # on-screen SIZE despite the fixed-pixel overlay.
            path = self._ask_snapshot_path()
            if not path:
                self._set_export_status("Snapshot cancelled.", success=False)
                return
            # Size cap: `screenshot(canvas_only, scale)` renders canvas·scale, so
            # clamp on the canvas size to stay under the GL texture limit.
            scale_factor, clamped = self._clamp_snapshot_scale(scale_factor)
            # Super-sample the CURRENT framing at N×: `screenshot(scale=N)` grows
            # the canvas while holding `camera.zoom`, which WIDENS the world FOV
            # (zoomed-out).  Pre-bumping the zoom by N cancels that, so the N×
            # canvas renders the SAME framing at N× resolution.  Restored below.
            if cam is not None and scale_factor > 1:
                try:
                    prior_zoom = float(cam.zoom)
                    cam.zoom = prior_zoom * scale_factor
                except Exception:
                    prior_zoom = None
            # The scale bar is a FIXED-pixel overlay, so a scale=N screenshot
            # renders it at 1/N the relative size.  Bump its font by N for the
            # shot so the exported bar keeps its on-screen size (restored below;
            # label-size only — napari exposes no line thickness).
            sbar = getattr(viewer, "scale_bar", None)
            if sbar is not None and scale_factor > 1:
                try:
                    if bool(getattr(sbar, "visible", False)):
                        prior_font = float(sbar.font_size)
                        sbar.font_size = prior_font * scale_factor
                except Exception:
                    prior_font = None
            self._set_export_status("Saving snapshot…", success=True)
            try:
                from qtpy.QtWidgets import QApplication
                QApplication.processEvents()
            except Exception:
                pass
            shot_fn(path=path, canvas_only=True, scale=scale_factor, flash=False)
            msg = f"Saved snapshot → {path}"
            if clamped:
                msg += (f" (scale factor clamped to {scale_factor}× for the "
                        "GPU texture limit)")
            self._set_export_status(msg, success=True)
            self._log("SNAPSHOT", msg)
        except Exception as exc:
            self._set_export_status(f"Snapshot failed: {exc}", success=False)
            self._log("SNAPSHOT", f"snapshot failed: {exc}")
        finally:
            # Restore the camera zoom + the scale-bar font (the only things we
            # touched) on EVERY path incl. cancel + exception.
            if cam is not None and prior_zoom is not None:
                try:
                    cam.zoom = prior_zoom
                except Exception:
                    pass
            if prior_font is not None:
                try:
                    viewer.scale_bar.font_size = prior_font
                except Exception:
                    pass

    def _on_scalebar_changed(self, *_):
        """Any ④ live scale-bar control changed → re-apply to the viewer."""
        self._apply_live_scale_bar()

    def _set_source_layer_units(self, units):
        """Set ``layer.units`` on the live raw source display layers + the
        large-source overview (the visible physical display), so the scale bar's
        label reads *units*.  This is the RELIABLE label path — napari's
        ``scale_bar.unit`` override is ignored on a non-physical / incompatible
        layer, so a user forcing 'µm' on a genuinely-1µm source (which loads at
        the ambiguous ``(1,1,1)``) only works by setting the layer units here.
        Guarded — best-effort, never raises."""
        viewer = getattr(self, "viewer", None)
        if viewer is not None:
            names = set(self._loaded_source_layers or [])
            for layer in list(getattr(viewer, "layers", []) or []):
                try:
                    if getattr(layer, "name", None) in names:
                        _apply_layer_units(layer, units)
                except Exception:
                    pass
        bt = getattr(self, "_box_tool", None)
        for layer in list(getattr(bt, "_overview_layers", None) or []):
            try:
                _apply_layer_units(layer, units)
            except Exception:
                pass

    def _apply_live_scale_bar(self):
        """Apply the ④ live scale-bar controls to the main viewer's napari scale
        bar (Brick B).  The µm/px LABEL rides ``layer.units`` (set by the physical
        display), so the bar reads **µm** once a voxel size is set and pixels
        otherwise — this method only drives visibility + styling.  Every write is
        guarded (a Qt-free fake / an older napari without an attr just no-ops).
        Re-applied on every control change + on step re-entry; turned off on
        step-exit so it never leaks into another workflow step."""
        viewer = getattr(self, "viewer", None)
        if viewer is None:
            return
        sb = getattr(viewer, "scale_bar", None)
        if sb is None:
            return
        w = getattr(self, "widget", None)

        def _set(name, value):
            try:
                setattr(sb, name, value)
            except Exception:
                pass

        try:
            want = bool(w.scalebar_toggle_button.isChecked())
        except Exception:
            want = False
        _set("visible", want)
        if not want:
            return
        # colour: light = white bar/text (dark viewer), dark = black bar/text.
        try:
            dark = str(w.scalebar_color_combo.currentText()).strip().lower() == "dark"
        except Exception:
            dark = False
        _set("colored", True)
        _set("color", (0.0, 0.0, 0.0, 1.0) if dark else (1.0, 1.0, 1.0, 1.0))
        # box background (contrasts with the text colour).
        try:
            _set("box", bool(w.scalebar_box_check.isChecked()))
        except Exception:
            pass
        _set("box_color", (1.0, 1.0, 1.0, 0.6) if dark else (0.0, 0.0, 0.0, 0.6))
        try:
            _set("ticks", bool(w.scalebar_ticks_check.isChecked()))
        except Exception:
            pass
        try:
            _set("font_size", float(w.scalebar_font_spin.value()))
        except Exception:
            pass
        # units: drive the bar's label through layer.units (the RELIABLE path —
        # scale_bar.unit is ignored on a non-physical layer, which is why forcing
        # µm did nothing before).  The picker FORCES a unit onto the display
        # layers so a genuinely-1µm source — which loads at the ambiguous
        # (1,1,1)=="unknown"=="pixel" — can be labelled µm by the user; 'auto'
        # follows the voxel-derived unit.  Leave scale_bar.unit at auto (None) so
        # the bar reads layer.units.
        try:
            picked = str(w.scalebar_unit_combo.currentText()).strip().lower()
            forced = {"µm": "um", "um": "um", "nm": "nm", "mm": "mm",
                      "pixel": "pixel"}.get(picked)
            units = ((forced,) * 3 if forced is not None
                     else _units_for_voxel(self._preview_spacing()))
            self._set_source_layer_units(units)
            _set("unit", None)
        except Exception:
            pass
        # length: None = napari auto-picks a round value; else the fixed value
        # (world units — µm when a voxel size is set).
        try:
            if bool(w.scalebar_auto_length_check.isChecked()):
                _set("length", None)
            else:
                _set("length", float(w.scalebar_length_spin.value()))
        except Exception:
            _set("length", None)
        try:
            pos = _SNAPSHOT_POSITIONS.get(
                str(w.scalebar_position_combo.currentText()).strip().lower(),
                "bottom_right")
        except Exception:
            pos = "bottom_right"
        _set("position", pos)
        # napari 0.8 does NOT reliably re-render the scale bar when only `length`,
        # `unit`, or the underlying `layer.units` change (only `font_size` forces
        # a redraw) — so the label can stay stale ("pixel" after a voxel edit,
        # a fixed length that won't take).  Nudge `font_size` off-and-back to force
        # the overlay to recompute + repaint with everything set above.
        try:
            fs = float(getattr(sb, "font_size"))
            sb.font_size = fs + 1.0
            sb.font_size = fs
        except Exception:
            pass

    def _ask_snapshot_path(self):
        """QFileDialog for the snapshot TIFF; defaults to <workspace>/exports/.
        Returns the chosen path (str, forced to a .tif/.tiff suffix) or "" on
        cancel / no workspace."""
        from qtpy.QtWidgets import QFileDialog
        try:
            ws = self.widget.output_file_path.text().strip()
        except Exception:
            ws = ""
        # B4: the dialog only PROPOSES
        # `artifacts/step1/exports/` (matching Step 4's
        # `artifacts/step4/exports`).  Nothing is created here -- opening and
        # cancelling this dialog must leave no folder behind; the directory
        # is made below, after the user has actually chosen a file.
        default_dir = ""
        if ws and os.path.isdir(ws):
            from ._paths import step1_exports_dir
            default_dir = str(step1_exports_dir(ws))
        default_name = os.path.join(default_dir or "", "snapshot.tiff")
        try:
            path, _ = QFileDialog.getSaveFileName(
                self.widget, "Save snapshot", default_name,
                "TIFF image (*.tiff *.tif)")
        except Exception:
            return ""
        if not path:
            return ""
        lower = path.lower()
        if not (lower.endswith(".tiff") or lower.endswith(".tif")):
            path += ".tiff"
        # The user confirmed a destination: now the folder may exist.
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        except Exception:      # noqa: BLE001
            pass
        return path

    def _clamp_snapshot_scale(self, scale_factor):
        """Clamp `scale_factor` so canvas_dim · factor ≤ the GL texture limit
        (~8192), returning ``(factor, clamped_bool)``.  `screenshot(canvas_only,
        scale)` renders the CANVAS at `scale`×, so the output's longest axis is
        `max(canvas) · scale` — the canvas is the correct clamp basis.
        Best-effort: no clamp when the canvas size can't be resolved (headless)."""
        factor = max(1, int(scale_factor))
        size = self._snapshot_canvas_size()
        if not size:
            return factor, False
        longest = max(size)
        if longest <= 0:
            return factor, False
        max_factor = max(1, int(_GL_MAX_TEXTURE_SIZE // longest))
        if factor > max_factor:
            return max_factor, True
        return factor, False

    def _snapshot_canvas_size(self):
        """(w, h) of the main canvas in pixels, best-effort; None when unknown."""
        bt = getattr(self, "box_tool", None)
        if bt is None:
            return None
        try:
            qtv = bt._main_qt_viewer()
        except Exception:
            qtv = None
        if qtv is None:
            return None
        canvas = getattr(qtv, "canvas", None)
        size = getattr(canvas, "size", None)
        try:
            if size is not None:
                cw, ch = int(size[0]), int(size[1])
                if cw > 0 and ch > 0:
                    return (cw, ch)
        except Exception:
            pass
        return None

    def _voxel_size_for_patch(self, patch_name):
        """(sz, sy, sx) physical spacing for a saved patch, resolved via its
        registry ``source_id`` → that source's ``voxel_size``; None
        when unknown.  Capture-time safe: keys off the PATCH's own source, so no
        source need be loaded (a reopened ④ Library has none) — else the viewer
        would silently fall to isotropic."""
        # One try over the WHOLE resolution (audit LOW): a malformed state can
        # make `all_patches`/the sources loop raise — the contract is "None when
        # unknown", so a bad state must degrade to isotropic, never escape into
        # the capture button's click slot (which would silently fail to open).
        try:
            ws = self.widget.output_file_path.text().strip()
            if not ws or not os.path.isdir(ws):
                return None
            state = load_state(ws)
            sid = None
            for row in all_patches(state):
                if row.get("patch_name") == patch_name:
                    sid = row.get("source_id")
                    break
            if not sid:
                return None
            for src in state.get("sources", []):
                if src.get("source_id") == sid:
                    vs = src.get("voxel_size")
                    if vs is None:
                        return None
                    vals = [float(v) for v in vs]
                    if len(vals) == 3 and all(v > 0 for v in vals):
                        return (vals[0], vals[1], vals[2])
                    return None
            return None
        except Exception:
            return None

    def _on_capture_patch_clicked(self, patch_name):
        """④ per-row capture button: open *patch_name* in the vispy capture
        inspector.  Reads the patch's on-disk TIFF + channel names + physical
        voxel size (from its source row); registers an ``on_close`` that prunes
        the retained dialog so a closed capture window's GL context is released.  A logged no-op when the patch isn't on disk."""
        # One Step-1 op at a time: opening a capture window (its `_PatchLoadWorker`
        # `imread`s the same TIFF) while a TRIM is overwriting it holds the file
        # open across the worker's `os.replace` → a Windows sharing violation (a
        # spurious "Patch file busy" trim failure).  Block while the surface is
        # busy / a trim is in flight or launching (threading-audit MED-1).
        if (getattr(self, "_step1_busy", False)
                or getattr(self, "_trim_running", False)
                or getattr(self, "_trim_launching", False)):
            self._set_export_status(
                "Please wait for the current operation to finish, then open "
                "the capture view.", success=False)
            return
        path = self._patch_disk_path(patch_name)
        if path is None:
            self._log("CAPTURE",
                      f"{patch_name}: no on-disk TIFF — cannot open capture view")
            return
        channel_names = self._patch_channel_names_from_sidecar(patch_name)
        voxel_size = self._voxel_size_for_patch(patch_name)
        try:
            from ._capture_widget import launch_patch_capture
        except Exception as exc:
            self._log("CAPTURE", f"capture widget import failed: {exc}")
            return
        if not hasattr(self, "_capture_dialogs"):
            self._capture_dialogs = []
        # Prune-on-close: a retained closed dialog would pin a GL context.
        # The holder breaks the chicken-and-egg (the callback needs the dialog
        # ref, created after the callback is passed).
        holder = {}

        def _on_capture_closed():
            d = holder.get("d")
            if d is not None:
                try:
                    self._capture_dialogs.remove(d)
                except ValueError:
                    pass
            self._open_capture_patches.discard(patch_name)
            # Closed before its load finished (on_loaded won't fire — the
            # dialog disconnects its worker on close) → clear the ④ scan here.
            self._set_review_busy(False)
            self._refresh_library_list()   # un-light the camera button
            self._log("CAPTURE", f"{patch_name}: capture view closed")

        self._log("CAPTURE",
                  f"{patch_name}: opening capture view -> {path} "
                  f"(voxel_size={voxel_size})")
        # Create + show the inspector FIRST so the window pops up IMMEDIATELY on
        # click (its load runs on a worker thread inside the dialog, so the UI
        # stays live) — do NOT processEvents/scan before it, which delayed the
        # window.  THEN scan the ④ bar; the dialog's `on_loaded` clears it.
        dlg = launch_patch_capture(
            self.widget, patch_name, path, channel_names,
            voxel_size=voxel_size, on_close=_on_capture_closed,
            on_loaded=lambda: self._set_review_busy(False))
        # Hold a ref so the non-modal dialog isn't garbage-collected.
        if dlg is not None:
            holder["d"] = dlg
            self._capture_dialogs.append(dlg)
            self._open_capture_patches.add(patch_name)
            # A depth-load may still be running on the SHARED ④ scan bar; its
            # `finished` would stop OUR scan mid-capture-load (grill MED).  Detach
            # it — as Trim does — so the capture owns the bar for its whole load.
            try:
                self._detach_intensity_worker()
            except Exception:
                pass
            self._set_review_busy(True)    # scan while it reads + builds
            self._refresh_library_list()   # light the camera button cyan

    def _clear_processed_layers(self):
        """Remove the ⟨processed⟩ whole-volume render layers from the shared
        canvas (hide-then-remove, freed-texture safety).

        Idempotent, tolerant of a layer the user already deleted, and always
        empties the tracking list.  Called before each new Process run + on
        source load / unload / switch + step-exit (via `_teardown_preview`) so a
        stale processed volume never outlives its source.  Never raises."""
        viewer = getattr(self, "viewer", None)
        for layer in list(getattr(self, "_processed_layers", None) or []):
            if viewer is None:
                continue
            try:
                if layer in viewer.layers:
                    layer.visible = False
                    viewer.layers.remove(layer)
            except Exception:
                pass
        self._processed_layers = []

    def _add_processed_layers(self, channel_levels, names, new_spacing):
        """Add the PROCESSED whole volume as one ⟨processed⟩ napari Image layer
        per channel (Brick 1b-1), at the source origin.

        *channel_levels* is a per-channel DISPLAY PYRAMID: a single-element list
        ``[arr]`` renders SINGLE-SCALE (full 3D detail, small channel), a
        multi-element list ``[arr, …, coarsest]`` renders MULTISCALE so napari's
        3D-volume view samples the coarsest level (≤ the VRAM cap) instead of
        pushing a full-res volume to the GPU — the conditional-(B) fix for the
        process-without-resize 3D crash.  2D still reads the finest level per
        slice, so full detail is preserved in 2D either way.

        MAIN-THREAD ONLY (called from `_on_process_computed`).  Replaces any
        previous processed set (clears first).  Returns ``(n_layers, any_coarse)``
        — ``any_coarse`` True iff at least one channel was rendered as a pyramid
        (so the caller can add the "3D is simplified" honesty note).  Each layer's
        contrast is pinned to ``(0, 65535)``; colour matches the source channel;
        ``translate`` is the origin ``(0, 0, 0)``; ``scale`` is the processed grid's
        PHYSICAL spacing ``new_spacing`` per axis (``== new_spacing/orig × voxel``
        algebraically) so the processed volume spans the SAME physical-µm box as
        the source — a geometry op that RE-GRIDDED the volume, or a no-geometry
        recipe (``new_spacing == the raw voxel``), both land correctly."""
        viewer = getattr(self, "viewer", None)
        self._clear_processed_layers()
        if viewer is None:
            return 0, False  # honor the (n_layers, any_coarse) contract
        orig_spacing = self._preview_spacing()
        # Physical-µm scale = `new_spacing` (the processed grid's spacing).  With
        # the display now in physical world, this spans the same box as the raw
        # source in BOTH the geometry (new_spacing ≠ orig) and no-geometry
        # (new_spacing == orig == raw voxel) cases — so, unlike the old
        # voxel-index display, it is set even when the grid is unchanged (a None
        # here would render voxel-index scale=1, misaligned by voxel×).  Falls
        # back to the raw voxel, then to identity, if new_spacing is unusable.
        # `units` key on the RAW voxel known-ness (the physical-ness of the
        # underlying volume; a resample of an unknown source is still "px").
        scale = None
        try:
            ns = tuple(float(s) for s in (new_spacing or ()))
            if ns and len(ns) == 3 and all(s > 0 for s in ns):
                scale = ns
        except (TypeError, ValueError):
            scale = None
        if scale is None:
            try:
                os_ = tuple(float(s) for s in (orig_spacing or ()))
                if os_ and len(os_) == 3 and all(s > 0 for s in os_):
                    scale = os_
            except (TypeError, ValueError):
                scale = None
        punits = _units_for_voxel(orig_spacing)

        added = []
        any_coarse = False
        n = min(len(channel_levels or []), len(names or []))
        # Guard THIS controller's `_on_layers_changed` across the inserts so the
        # not-yet-tracked ⟨processed⟩ layers aren't counted as channels (mirrors
        # `_adding_preview_layers`); a re-entrancy flag, NOT an event block, so
        # napari's own layer list still shows + lets you toggle each layer.
        self._adding_processed_layers = True
        try:
            for ci in range(n):
                levels = channel_levels[ci]
                if not isinstance(levels, (list, tuple)):
                    levels = [levels]
                levels = [np.asarray(lv) for lv in levels if lv is not None]
                if not levels or levels[0].ndim != 3:
                    continue
                multiscale = len(levels) > 1
                data = levels if multiscale else levels[0]
                add_kwargs = dict(
                    name=f"{PROCESSED_LAYER_MARKER}{names[ci]}",
                    colormap=self._preview_colormap_for_channel(names[ci]),
                    blending="additive",
                    contrast_limits=(0, 65535),
                    translate=(0, 0, 0),
                    visible=True,
                )
                if multiscale:
                    add_kwargs["multiscale"] = True
                if scale is not None:
                    add_kwargs["scale"] = scale
                try:
                    layer = viewer.add_image(data, **add_kwargs)
                except TypeError:
                    # A viewer/fake that doesn't accept an optional kwarg (incl.
                    # `multiscale`) — retry with the essentials + single-scale
                    # level 0 (mirrors `_add_preview_layers`).
                    for opt in ("contrast_limits", "translate", "scale", "visible",
                                "multiscale"):
                        add_kwargs.pop(opt, None)
                    try:
                        layer = viewer.add_image(levels[0], **add_kwargs)
                    except Exception:
                        continue
                    multiscale = False  # the fallback rendered single-scale
                except Exception:
                    continue
                _apply_layer_units(layer, punits)
                added.append(layer)
                if multiscale:
                    any_coarse = True
        finally:
            self._adding_processed_layers = False
        self._processed_layers = added
        # PURE (clean-swap rework): this method only ADDS the layers — the caller
        # (`_on_process_computed`, or `_readd_processed_layers`) drives the
        # Raw→Processed clean swap via `_show_processed_view()`.  Calling it here
        # would recurse through `_readd_processed_layers`.
        self._log("STEP1-PROCESS",
                  f"render {len(added)} ⟨processed⟩ layer(s) (coarse3d={any_coarse})")
        return len(added), any_coarse

    # ------------------------------------------------------------------
    # ② preprocessing PREVIEW render (Brick 2c) — the WYSIWYG ⟨preview⟩ layers
    # ------------------------------------------------------------------

    def _hidden_layer_phrase(self):
        """", and your original <name> layer is hidden while the preview is up"
        — or "" when nothing was hidden.  Naming the layer means the user can
        find it again; "the layer underneath" left them hunting."""
        hidden = list(getattr(self, "_preview_hidden_layers", None) or [])
        names = [str(getattr(l, "name", "")) for l in hidden]
        names = [n for n in names if n]
        if not names:
            return ""
        if len(names) == 1:
            what = f"your original {names[0]} layer is"
        else:
            what = f"your original {', '.join(names)} layers are"
        return f", and {what} hidden while the preview is up"

    def _hide_layer_under_preview(self, channel_name):
        """Hide the layer a ⟨preview⟩ layer for *channel_name* now covers, and
        remember that WE hid it so `_clear_preview_layers` can put it back.

        The candidates, in the order the canvas shows them: the ⟨processed⟩
        whole-volume layer for this channel if one is up, else the raw source
        layer.  A layer the user had ALREADY hidden is not recorded — restore
        means "undo what we changed", never "turn everything on".
        Best-effort; never raises (a render failure must not sink a Preview)."""
        viewer = getattr(self, "viewer", None)
        if viewer is None:
            return
        if not hasattr(self, "_preview_hidden_layers"):
            self._preview_hidden_layers = []
        wanted = (f"{PROCESSED_LAYER_MARKER}{channel_name}", str(channel_name))
        try:
            for want in wanted:
                for layer in list(getattr(viewer, "layers", []) or []):
                    if str(getattr(layer, "name", "")) != want:
                        continue
                    if not getattr(layer, "visible", False):
                        continue
                    layer.visible = False
                    self._preview_hidden_layers.append(layer)
                    return
        except Exception as exc:  # noqa: BLE001
            self._log("STEP1-PREVIEW", f"auto-hide under preview: {exc}")

    def _restore_layers_hidden_under_preview(self):
        """Re-show whatever `_hide_layer_under_preview` hid (C#6).  Idempotent,
        tolerant of a layer the user has since deleted, always empties the list."""
        viewer = getattr(self, "viewer", None)
        for layer in list(getattr(self, "_preview_hidden_layers", None) or []):
            try:
                if viewer is not None and layer not in getattr(viewer, "layers", []):
                    continue
                layer.visible = True
            except Exception:
                pass
        self._preview_hidden_layers = []

    def _clear_preview_layers(self):
        """Remove the ⟨preview⟩ processed-crop render layers from the shared
        canvas (hide-then-remove, freed-texture safety).

        Idempotent, tolerant of a layer the user already deleted, and always
        empties the tracking list.  Called before each new Preview + on source
        load / unload / switch + on step-exit so a stale processed crop never
        outlives the source it was taken from.  Best-effort; never raises.
        """
        viewer = getattr(self, "viewer", None)
        # Not batched - see the measurement note in `_unload_source_inner`.
        for layer in list(getattr(self, "_preview_layers", None) or []):
            if viewer is None:
                continue
            try:
                if layer in viewer.layers:
                    # Hide BEFORE remove so the vispy visual disconnects from the
                    # scene graph (avoids freed-GPU-texture access violations).
                    layer.visible = False
                    viewer.layers.remove(layer)
            except Exception:
                pass
        self._preview_layers = []
        # C#6: whatever the preview covered goes back to how the user had it.
        self._restore_layers_hidden_under_preview()

    def _preview_layer_geometry(self, bounds, proc_shape, voxel_size=(1.0, 1.0, 1.0)):
        """``(translate, scale)`` placing a processed crop where it was taken, in
        PHYSICAL-µm world.

        Both parts are multiplied ELEMENT-WISE by ``voxel_size`` ``(vz, vy, vx)``
        so the crop lands at the same physical box as the raw source (which now
        sits in physical world):

        * ``translate`` = the crop origin ``(z0, y0, x0)`` voxels × ``voxel_size``.
        * ``scale`` = ``(crop_shape / processed_shape) × voxel_size`` per axis —
          the crop/proc ratio (== ``new_spacing/old_spacing``, exact under
          rounding from the actual output shape) scaled into µm.  A no-geometry
          recipe → ratio ``(1,1,1)`` → ``scale == voxel_size``.

        Returns ``(None, voxel_size)`` when bounds are missing — the caller then
        adds at the origin (translate default) but STILL at the physical voxel
        scale, not voxel-index 1.  Never raises.
        """
        try:
            vs = tuple(float(v) for v in voxel_size)
            if len(vs) != 3:
                vs = (1.0, 1.0, 1.0)
        except (TypeError, ValueError):
            vs = (1.0, 1.0, 1.0)
        if not bounds:
            return None, vs
        try:
            z0, y0, x0 = int(bounds["z0"]), int(bounds["y0"]), int(bounds["x0"])
            crop = (int(bounds["z1"]) - int(bounds["z0"]),
                    int(bounds["y1"]) - int(bounds["y0"]),
                    int(bounds["x1"]) - int(bounds["x0"]))
        except Exception:
            return None, vs
        translate = tuple(o * v for o, v in zip((z0, y0, x0), vs))
        scale = []
        for c, p, v in zip(crop, proc_shape, vs):
            p = int(p)
            scale.append(((float(c) / float(p)) if p else 1.0) * v)
        return translate, tuple(scale)

    def _preview_colormap_for_channel(self, name):
        """The napari colormap the SOURCE layer for *name* is shown with, so the
        ⟨preview⟩ colour matches its source channel.

        Replays the load path's `_add_source_display_layers` mapping EXACTLY:
        `_SOURCE_DISPLAY_CMAPS` positionally by source-channel order, or ``"gray"``
        for a single-channel source — never a parallel palette.  Falls back to the
        first colormap for an unrecognised name.  Never raises.
        """
        order = []
        for entry in getattr(self, "_source_display_channels", None) or []:
            try:
                nm, _data = entry
            except (TypeError, ValueError):
                continue
            order.append(nm)
        if len(order) <= 1:
            return "gray"
        try:
            idx = order.index(name)
        except ValueError:
            idx = 0
        return _SOURCE_DISPLAY_CMAPS[idx % len(_SOURCE_DISPLAY_CMAPS)]

    def _add_preview_layers(self, afters, names):
        """Add the PROCESSED test crop as one ⟨preview⟩ napari Image layer per
        channel (Brick 2c), positioned + scaled to sit where the crop was taken.

        MAIN-THREAD ONLY (called from `_on_preview_computed`).  Replaces any
        previous preview set (clears first).  Returns the number of layers added.
        Each layer is float32 ``[0, 1]`` (execute's working space) so the contrast
        is pinned to ``(0, 1)``; colour matches the source channel; additive
        blending overlays cleanly on the source for before/after comparison.
        """
        viewer = getattr(self, "viewer", None)
        # Always replace: drop the previous render before adding the new one.
        self._clear_preview_layers()
        if viewer is None:
            return 0
        bounds_list = list(getattr(self, "_preview_bounds", None) or [])
        # Physical-µm: the preview crop is in the RAW source's voxel grid, so its
        # world geometry rides the raw voxel (live read — a post-load user action).
        vox = self._preview_spacing()
        punits = _units_for_voxel(vox)
        added = []
        # C#10: publish the tracking list BEFORE the loop and append into it as
        # each layer lands, so an exception mid-loop cannot strand an
        # already-added ⟨preview⟩ layer on the canvas untracked (it would then
        # outlive Step 1 — `_clear_preview_layers` only reaches what is tracked).
        self._preview_layers = added
        n = min(len(afters or []), len(names or []))
        # Guard THIS controller's `_on_layers_changed` across the inserts so the
        # not-yet-tracked ⟨preview⟩ layers aren't counted as channels (quirk 1);
        # a re-entrancy flag, NOT an event block, so napari's own layer list
        # still shows + lets you toggle each preview layer.
        self._adding_preview_layers = True
        try:
            for ci in range(n):
                arr = np.asarray(afters[ci], dtype=np.float32)
                if arr.ndim != 3:
                    continue
                bounds = bounds_list[ci] if ci < len(bounds_list) else None
                translate, scale = self._preview_layer_geometry(
                    bounds, arr.shape, vox)
                add_kwargs = dict(
                    name=f"{PREVIEW_LAYER_MARKER}{names[ci]}",
                    colormap=self._preview_colormap_for_channel(names[ci]),
                    blending="additive",
                    contrast_limits=(0.0, 1.0),
                    visible=True,
                )
                if translate is not None:
                    add_kwargs["translate"] = translate
                if scale is not None:
                    add_kwargs["scale"] = scale
                try:
                    layer = viewer.add_image(arr, **add_kwargs)
                except TypeError:
                    # A viewer/fake that doesn't accept an optional kwarg — retry
                    # with only the essentials (mirrors `_add_source_display_layers`).
                    for opt in ("contrast_limits", "translate", "scale", "visible"):
                        add_kwargs.pop(opt, None)
                    try:
                        layer = viewer.add_image(arr, **add_kwargs)
                    except Exception:
                        continue
                except Exception:
                    continue
                # Track it the instant it lands on the canvas — BEFORE any
                # further per-layer setup, so nothing between here and the end
                # of the loop can strand it untracked (C#10).
                added.append(layer)
                _apply_layer_units(layer, punits)
                # C#6: the ⟨preview⟩ layer is additive with contrast pinned to
                # (0, 1), so it sits OVER the still-visible source and a no-op
                # recipe "looks processed".  Hide this channel's source (or
                # ⟨processed⟩) layer for the user instead of asking them to; the
                # restore is `_clear_preview_layers`.
                self._hide_layer_under_preview(names[ci])
        finally:
            self._adding_preview_layers = False
        self._log("STEP1-PREVIEW", f"render {len(added)} ⟨preview⟩ layer(s)")
        return len(added)

    # ------------------------------------------------------------------
    # ② preview REGION box (Preview v2 Piece 2) — its OWN overlay + lifecycle,
    #    SEPARATE from the ③ extract-box tool.
    # ------------------------------------------------------------------

    def _resolve_preview_full_shape(self):
        """The full-res ``(Z, Y, X)`` spatial shape from the axis-resolved
        per-channel display handle — the SAME axes-safe source the preview
        compute crops from (never raw ``_source_lazy``, which is un-transposed
        on-disk order).  ``None`` when nothing is loaded."""
        channels = getattr(self, "_source_display_channels", None)
        if not channels:
            return None
        try:
            _nm, data = channels[0]
        except (TypeError, ValueError, IndexError):
            return None
        arr = data[0] if isinstance(data, (list, tuple)) else data
        if arr is None or getattr(arr, "ndim", None) != 3:
            return None
        return tuple(int(s) for s in arr.shape)

    def _ensure_preview_overlay(self):
        """Lazily build the preview box's OWN ``Step1BoxOverlay`` (distinct
        colour).  SEPARATE from ``box_tool._overlay`` — never reuse or touch that
        instance.  Degenerates to a safe no-op when there is no QtViewer
        (headless / fakes: the overlay renders nothing but the instance exists)."""
        overlay = getattr(self, "_preview_box_overlay", None)
        if overlay is not None:
            return overlay
        try:
            from ._box_overlay import Step1BoxOverlay
            qtv = None
            bt = getattr(self, "box_tool", None)
            if bt is not None:
                try:
                    qtv = bt._main_qt_viewer()
                except Exception:
                    qtv = None
            overlay = Step1BoxOverlay(qtv, style="corners")
            try:
                overlay.set_style(color=_PREVIEW_BOX_COLOR)
            except Exception:
                pass
            # Physical-µm: seed the raw voxel so the preview REGION box aligns with
            # the region actually previewed (re-set in `_render_preview_box` too,
            # since a voxel edit can't refresh the cached overlay instance).
            try:
                overlay.set_voxel_size(self._preview_spacing())
            except Exception:
                pass
            self._preview_box_overlay = overlay
        except Exception as exc:
            self._log("STEP1-PREVIEW", f"preview overlay create failed: {exc}")
            self._preview_box_overlay = None
        self._connect_preview_close_hook()
        return self._preview_box_overlay

    def _connect_preview_close_hook(self):
        """Connect ``QApplication.aboutToQuit`` → destroy the preview box ONCE, so
        the overlay is never left dangling on the shared canvas during Qt teardown
        (mirrors ``Step1BoxTool._connect_close_hook``)."""
        if getattr(self, "_preview_box_close_hook_connected", False):
            return
        try:
            from qtpy.QtWidgets import QApplication
            app = QApplication.instance()
            if app is not None:
                app.aboutToQuit.connect(self._on_preview_app_quit)
                self._preview_box_close_hook_connected = True
        except Exception:
            pass

    def _on_preview_app_quit(self):
        try:
            self._destroy_preview_box()
        except Exception:
            pass

    def _bump_preview_token(self):
        """Advance the preview stale-token so any in-flight compute for the OLD
        box position/recipe is discarded by the `_on_preview_computed` guard.

        C#12: also ASK that compute to stop.  The token alone only discards the
        RESULT — a superseded run (source switch, step exit, recipe edit, box
        move) otherwise burns cores to completion on an answer nobody will read.
        `request_cancel` is cooperative: the worker checks the flag before it
        emits, so the terminal signal (hence `thread.quit` + the pair cleanup)
        still fires.  Best-effort — a missing/odd worker must never break the
        lifecycle paths that call this."""
        self._preview_token = getattr(self, "_preview_token", 0) + 1
        worker = getattr(self, "_preview_worker", None)
        cancel = getattr(worker, "request_cancel", None)
        if callable(cancel):
            try:
                cancel()
            except Exception:
                pass

    def _invalidate_preview_on_recipe_edit(self):
        """A recipe-card edit makes the shown preview stale — mirror the box
        slider handlers (bump the stale-token + clear the ⟨preview⟩ layers) and
        additionally grey the ② histogram with a "recipe changed — re-Preview"
        hint.  NEVER recomputes live; best-effort + idempotent (a no-op when no
        preview / histogram is shown).  Called from `on_recipe_edited`."""
        self._bump_preview_token()          # recipe changed → stale-guard compute
        self._clear_preview_layers()        # any shown processed crop is now stale
        mark = getattr(self.widget, "mark_recipe_preview_stale", None)
        if callable(mark):
            try:
                mark()
            except Exception:
                pass

    def _refresh_preview_region_ready(self):
        """Tell the widget whether a preview REGION exists, so it enables/disables
        the Preview button (disabled-with-hint until a region is placed)."""
        # Whole body guarded (incl. the `self.widget` access): this is the last
        # step of `_destroy_preview_box`, which the app-quit hook could in
        # principle reach with a deleted widget (lifecycle audit LOW-2).
        try:
            setter = getattr(self.widget, "set_preview_region_ready", None)
            if callable(setter):
                setter(getattr(self, "_preview_box", None) is not None)
        except Exception:
            pass

    def _refresh_preview_region_active(self, active, full_shape=None):
        """Tell the widget to REVEAL (`active=True`) or COLLAPSE the grouped
        preview-region editor (Recenter · S/M/L · X/Y/Z + Z-depth sliders · Exit),
        seeding the sliders from the current box + *full_shape* on reveal.  Fully
        guarded (incl. the `self.widget` access) so a teardown reached during Qt
        shutdown with a dying widget can't raise (mirrors
        `_refresh_preview_region_ready`)."""
        try:
            setter = getattr(self.widget, "set_preview_region_active", None)
            if callable(setter):
                setter(bool(active), full_shape)
        except Exception:
            pass

    def _render_preview_box(self, full_shape=None):
        """(Re)draw the preview box on its own overlay.  Best-effort + guarded."""
        box = getattr(self, "_preview_box", None)
        if box is None:
            return
        if full_shape is None:
            full_shape = self._resolve_preview_full_shape()
        if full_shape is None:
            return
        overlay = self._ensure_preview_overlay()
        if overlay is None:
            return
        # Physical-µm: refresh the raw voxel each redraw (a spin edit can't refresh
        # the cached overlay instance) so the region box tracks the previewed region.
        try:
            overlay.set_voxel_size(self._preview_spacing())
        except Exception:
            pass
        try:
            # A single box → the overlay's "active" line (drawn in the override
            # colour); the "idle" line stays empty.
            overlay.update([box], 0, full_shape)
        except Exception as exc:
            self._log("STEP1-PREVIEW", f"preview box redraw failed: {exc}")

    def _destroy_preview_box(self):
        """Tear down the preview box + its overlay (source load/switch, unload,
        step-exit, app-close).  Idempotent + guarded.  Does NOT touch the ③ box
        tool or its overlay.  Bumps the stale-token so a queued compute for the
        gone box is discarded, and refreshes the Preview-enabled state."""
        overlay = getattr(self, "_preview_box_overlay", None)
        if overlay is not None:
            try:
                overlay.destroy()
            except Exception as exc:
                self._log("STEP1-PREVIEW", f"preview overlay destroy: {exc}")
        self._preview_box_overlay = None
        self._preview_box = None
        self._bump_preview_token()
        self._refresh_preview_region_ready()

    def _teardown_preview(self, drop_processed: bool = True):
        """The SINGLE graceful exit for the ② preview REGION (Preview v2 Piece 2).

        ``drop_processed`` (default True) ALSO tears down the ⟨processed⟩ whole-
        volume view — the source-lifecycle callers (load/switch, unload, step-
        exit) want that, so a processed volume never outlives its source.  The ③
        box-mode arbitration passes ``drop_processed=False``: entering box mode
        must clear the mutually-exclusive preview REGION but MUST NOT drop a
        whole-volume Process result (dropping it — plus the clean-swap already
        having pulled raw off the canvas — was the "box mode hides everything"
        empty-canvas bug; see :meth:`_arbitrate_for_box_mode`).

        Drops any shown ⟨preview⟩ render layers, destroys the box + its overlay
        (which bumps the stale-token + re-disables Preview via
        `_destroy_preview_box`), and collapses the grouped ② controls.  Idempotent
        + guarded + safe when NO region exists — so every trigger (the "Exit
        preview" button AND the lifecycle sites: source load/switch, unload,
        step-exit) can route through it in any order, twice-safe.

        NOT called from the `aboutToQuit` hook — at app quit the widget may be
        dying, so that path calls `_destroy_preview_box` only (no UI touch)."""
        try:
            self._clear_preview_layers()
        except Exception as exc:
            self._log("STEP1-PREVIEW", f"teardown clear: {exc}")
        # Drop the ⟨processed⟩ whole-volume view too — but ONLY on the source-
        # lifecycle callers (load/switch, unload, step-exit).  Box-mode arbitration
        # passes drop_processed=False so drawing a box never loses a Process result
        # (nor empties the canvas — the clean-swap already pulled raw off it).
        if drop_processed:
            self._teardown_processed_view()
        try:
            self._destroy_preview_box()
        except Exception as exc:
            self._log("STEP1-PREVIEW", f"teardown destroy: {exc}")
        # Collapse the grouped controls (guarded internally).
        self._refresh_preview_region_active(False)
        # Brick A: the processed sets are now cleared — grey out the ② Raw /
        # Processed toggle (Processed disabled, back to Raw) + unbind `\`.  Runs
        # on every source load / switch / unload / step-exit (all funnel here),
        # so a stale "Processed" selection never outlives the source.
        try:
            self._refresh_view_toggle_state()
        except Exception as exc:
            self._log("STEP1-PROCESS", f"teardown view-toggle refresh: {exc}")
        self._clear_contrast_cache("preview/processed teardown")

    def _teardown_processed_view(self):
        """Tear down the ⟨processed⟩ whole-volume view (both regimes) + its RAM
        cache + re-create spec + progressive store state, and drop
        `_processed_available` (greys the ② Processed button + unbinds `\\`).

        The processed portion of :meth:`_teardown_preview`, split out so box-mode
        arbitration can SKIP it (``drop_processed=False``) — a Process result must
        survive drawing a box.  Called by the source-lifecycle callers (load/
        switch, unload, step-exit), where a processed volume must never outlive
        the source it was baked from.  Best-effort; never raises."""
        try:
            self._clear_processed_layers()
        except Exception as exc:
            self._log("STEP1-PROCESS", f"teardown clear: {exc}")
        # …and the lazy-multiscale ⟨processed⟩ STORE render (Phase 2a streaming) —
        # same lifecycle: it must never outlive the source it was streamed from.
        try:
            self._clear_processed_store_layers()
        except Exception as exc:
            self._log("STEP1-PROCESS", f"teardown clear store: {exc}")
        # If a whole-volume process is IN FLIGHT, cancel it (grill HIGH-1): its
        # async finish would otherwise pass the token guard and render ⟨processed⟩
        # Image layers onto ANOTHER step's shared canvas after we've navigated away
        # (the Step-3 all-Image-layers abort) AND leave `_step1_busy` locked.
        # `cancel_process_whole_volume` bumps `_process_token` (stales the finish) +
        # releases the lock.  GUARDED on an actual in-flight run so teardown on an
        # idle surface never spuriously unlocks it (`_set_step1_busy(False)` on a
        # zero-depth lock would re-enable the widgets).
        if (getattr(self, "_process_thread", None) is not None
                or getattr(self, "_process_store_thread", None) is not None
                or getattr(self, "_processing", False)):
            try:
                self.cancel_process_whole_volume()
            except Exception as exc:
                self._log("STEP1-PROCESS", f"teardown cancel: {exc}")
        # Drop the in-memory processed cache (grill MED-2) — a multi-GB uint16 set
        # must not linger in RAM across a source switch / unload / step-exit.  NOT
        # cleared in `_clear_processed_layers` (that runs mid-render in
        # `_add_processed_layers`, right after the cache is set → would wipe it).
        self._processed_cache = None
        # Clean-swap rework: a new source starts with NO processed view — clear the
        # availability flag (greys the ② Processed button + unbinds `\`) and the
        # re-create spec (also releases the retained in-memory display pyramid).
        self._processed_available = False
        self._processed_redisplay = None
        # Item 4: drop the processed progressive store/channel state (the layers
        # were removed above via `_clear_processed_store_layers`, which fired the
        # seam teardown + reset `_processed_progressive_active`).  This is the
        # lifecycle boundary (source load/switch, unload, step-exit all funnel
        # here), so a stale processed store never drives a swap onto the next
        # source — mirrors the raw `_current_multiscale_*` reset in `unload_source`.
        self._current_processed_multiscale_store = None
        self._processed_display_channels = None
        self._processed_progressive_active = False

    def _arbitrate_for_box_mode(self):
        """Mode arbitration when ENTERING the ③ box tool (`on_box_mode_toggled`,
        `add_box`, `on_place_toggled`).

        Clears ONLY the mutually-exclusive ② preview REGION and STAYS on whatever
        is loaded (raw or processed) — placing / moving boxes works on the
        processed image too (the box coords map 1:1 to the same-grid processed
        store).  `drop_processed=False` keeps the processed view intact, so
        entering box mode never empties the canvas (the "box mode hides
        everything" bug: the old `_teardown_preview()` removed processed AND never
        restored raw → empty canvas).

        NOTE: Extract still reads the RAW source today; wiring Extract to pull the
        LOADED (processed) data is the next piece of work.  Until then, extracting
        while viewing Processed needs a flip to Raw first (the boxes persist
        across the flip — the overlay is a scene visual, not a layer).
        `[BOX-DIAG]` lines are TEMP diagnostics — strip before the baseline
        merge."""
        try:
            showing = self._is_processed_view_showing()
        except Exception:
            showing = False
        layers = getattr(getattr(self, "viewer", None), "layers", None)
        n_before = len(layers) if layers is not None else "?"
        self._log(
            "BOX-DIAG",
            f"arbitrate enter: showing_processed={showing} "
            f"on_canvas={getattr(self, '_source_layers_on_canvas', None)} "
            f"n_layers={n_before}")
        # Stay on the loaded view (raw OR processed) — NO snap to raw.  Clear the
        # mutually-exclusive ② preview REGION only; drop_processed=False keeps the
        # processed layers on the canvas (never an empty canvas).
        self._teardown_preview(drop_processed=False)
        layers = getattr(getattr(self, "viewer", None), "layers", None)
        n_after = len(layers) if layers is not None else "?"
        self._log(
            "BOX-DIAG",
            f"arbitrate done: on_canvas={getattr(self, '_source_layers_on_canvas', None)} "
            f"n_layers={n_after}")

    def _preview_center_yx(self, full_shape):
        """The ``(cy, cx)`` a placed/recentred preview box lands at — placement is
        a 2D op, so reuse the ③ box tool's 2D-centre helper (the raw 2D camera
        centre is garbage in 3D).  Falls back to the volume centre."""
        _z, Y, X = (int(s) for s in full_shape)
        fallback = (Y // 2, X // 2)
        bt = getattr(self, "box_tool", None)
        if bt is None:
            return fallback
        try:
            if getattr(bt, "_full_shape", None) is None:
                bt._full_shape = full_shape
            cy, cx = bt._camera_center_yx()
            return int(cy), int(cx)
        except Exception:
            return fallback

    def _preview_size_yx(self, full_shape, frac=None):
        """The ``(sy, sx)`` XY size for the preview box at fraction *frac* of the
        field (defaults to the sticky ``_preview_size_frac``)."""
        _z, Y, X = (int(s) for s in full_shape)
        if frac is None:
            frac = getattr(self, "_preview_size_frac", 0.25)
        try:
            frac = float(frac)
        except (TypeError, ValueError):
            frac = 0.25
        if not (0.0 < frac <= 1.0):
            frac = 0.25
        return max(1, int(Y * frac)), max(1, int(X * frac))

    def place_preview_region(self):
        """② "Place preview region" — drop a default-sized (full-Z) preview box
        at the current 2D view centre.

        Mode arbitration (Piece 2): the preview box and the ③ extract-box
        tool are mutually exclusive owners of ndisplay, so if the box tool is
        active we STOP it first (it restores its own ndisplay on stop).  The
        preview box does NOT itself require the box tool, and does NOT force
        ndisplay — the overlay renders in whatever view is current."""
        if getattr(self, "_step1_busy", False):
            return
        full_shape = self._resolve_preview_full_shape()
        if full_shape is None:
            self._set_preview_status(
                "Load a source with channels first, then place a preview region.",
                ok=False)
            return
        bt = getattr(self, "box_tool", None)
        if bt is not None and bt.is_active():
            try:
                bt.stop()
            except Exception as exc:
                self._log("STEP1-BOX", f"box-tool stop for preview place: {exc}")
        cy, cx = self._preview_center_yx(full_shape)
        sy, sx = self._preview_size_yx(full_shape)
        self._preview_box = place_box(full_shape, cy, cx, sy, sx)   # full Z
        self._bump_preview_token()          # box changed → stale-guard compute
        self._render_preview_box(full_shape)
        self._refresh_preview_region_ready()
        # Reveal the grouped editor (Recenter · S/M/L · X/Y/Z + Z-depth sliders ·
        # Exit) + push the initial slider values off the just-placed box.
        self._refresh_preview_region_active(True, full_shape)
        self._set_preview_status(
            "Preview region placed — drag X/Y/Z or Z depth to reposition, then "
            "click Preview to compute this region.",
            ok=True)

    def recenter_preview_region(self):
        """② "Recenter" — move the preview box to the current 2D view centre,
        keeping its size + Z.  No-op with no box (or while busy)."""
        box = getattr(self, "_preview_box", None)
        if box is None or getattr(self, "_step1_busy", False):
            return
        full_shape = self._resolve_preview_full_shape()
        if full_shape is None:
            return
        cy, cx = self._preview_center_yx(full_shape)
        self._preview_box = box.with_center(box.cz, cy, cx).clamped(full_shape)
        self._bump_preview_token()          # box moved → stale-guard compute
        self._clear_preview_layers()        # shown preview is now stale
        self._render_preview_box(full_shape)
        # Push the moved box back to the sliders (guarded/suppressed → no
        # recursion) so the handles + the box never disagree (audit finding 1).
        self._refresh_preview_region_active(True, full_shape)

    def set_preview_size_fraction(self, frac):
        """② S/M/L preset — resize the preview box (keeps its centre + Z) and
        remember the sticky fraction for the next Place.  A no-op on the box while
        box-less / busy (mirrors the ③ size-preset no-op-when-off), but the sticky
        fraction is still remembered."""
        try:
            frac = float(frac)
        except (TypeError, ValueError):
            return
        if not (0.0 < frac <= 1.0):
            return
        self._preview_size_frac = frac
        box = getattr(self, "_preview_box", None)
        if box is None or getattr(self, "_step1_busy", False):
            return
        full_shape = self._resolve_preview_full_shape()
        if full_shape is None:
            return
        sy, sx = self._preview_size_yx(full_shape, frac)
        self._preview_box = box.with_size(box.sz, sy, sx).clamped(full_shape)
        self._bump_preview_token()          # box resized → stale-guard compute
        self._clear_preview_layers()        # shown preview is now stale
        self._render_preview_box(full_shape)
        self._refresh_preview_region_active(True, full_shape)  # re-seed sliders

    def set_preview_center(self, cz=None, cy=None, cx=None):
        """② X/Y/Z position sliders — MOVE the preview box along one or more axes,
        LIVE.  Updates `_preview_box` (`with_center` → `clamped`), re-renders the
        overlay, bumps the stale-token, and CLEARS any shown ⟨preview⟩ layers so a
        moved box never disagrees with the last computed preview (the user
        re-clicks Preview to recompute — we never recompute per tick).  A no-op
        with no box / while busy (mirrors `recenter_preview_region`).  Each of
        `cz`/`cy`/`cx` is optional (a per-axis slider passes only its own)."""
        box = getattr(self, "_preview_box", None)
        if box is None or getattr(self, "_step1_busy", False):
            return
        full_shape = self._resolve_preview_full_shape()
        if full_shape is None:
            return
        try:
            ncz = box.cz if cz is None else int(cz)
            ncy = box.cy if cy is None else int(cy)
            ncx = box.cx if cx is None else int(cx)
        except (TypeError, ValueError):
            return
        self._preview_box = box.with_center(ncz, ncy, ncx).clamped(full_shape)
        self._bump_preview_token()          # box moved → stale-guard compute
        self._clear_preview_layers()        # shown preview is now stale
        self._render_preview_box(full_shape)
        # Re-seed the sliders to the CLAMPED centre so an edge-clamped drag
        # doesn't leave the handle lagging the box (audit finding 2; guarded +
        # suppressed → a no-op at interior values, snaps to the clamp at edges).
        self._refresh_preview_region_active(True, full_shape)

    def set_preview_z_depth(self, sz):
        """② Z-depth slider — set how many Z slices the box spans (keeps its
        centre + XY), LIVE.  Same contract as `set_preview_center`: `with_size` →
        `clamped`, re-render, bump the token, clear the (now stale) shown preview.
        A no-op with no box / while busy."""
        box = getattr(self, "_preview_box", None)
        if box is None or getattr(self, "_step1_busy", False):
            return
        full_shape = self._resolve_preview_full_shape()
        if full_shape is None:
            return
        try:
            sz = int(sz)
        except (TypeError, ValueError):
            return
        self._preview_box = box.with_size(sz, box.sy, box.sx).clamped(full_shape)
        self._bump_preview_token()          # box resized → stale-guard compute
        self._clear_preview_layers()        # shown preview is now stale
        self._render_preview_box(full_shape)
        # Re-seed sliders to the clamped box (audit finding 2) — also refreshes
        # the Z-position enable state if the depth crossed the full-depth line.
        self._refresh_preview_region_active(True, full_shape)

    def _on_export_clicked(self):
        """④'s ONE export control (c #14) — commits the CHECKED patches.

        Was per-patch (`currentItem()`); the amendment makes ④ a single
        tick-driven commit, so this reads the Box List's check column.

        **There is no fall-back to the selected row.**  There used to be, "so
        the button is never a dead end" — but it meant unticking every row and
        pressing Export still wrote the highlighted patch, which reinstates the
        selection scope #14 rules out by name and does it on a PERMANENT action.
        An empty tick set now says so and writes nothing.
        """
        if getattr(self, "_step1_busy", False):
            return  # one Step-1 op at a time; extract already guards
        if getattr(self, "_migration_offer_timer", None) is not None:
            self._migration_offer_timer.stop()  # no file-move mid-export (R2)
        getter = getattr(self.widget, "checked_patch_names", None)
        if getter is None:
            # No Box List at all (a bare widget double) — keep the old path
            # rather than silently doing nothing.
            return self._export_current_patch()
        checked = getter()
        if not checked:
            self._set_export_status(
                "Nothing ticked — tick the patches you want to export.",
                success=False)
            return
        if not self._confirm_export(checked):
            self._set_export_status("Export cancelled — nothing written.",
                                    success=False)
            return
        return self._export_named_patches(checked)

    def _confirm_export(self, names):
        """List exactly what will be written, before writing it.

        #14's accepted cost ("exporting one patch takes one extra click") is
        explicitly justified by "an action that is permanent and already
        requires a confirm listing what will be written" — so the confirm is
        load-bearing for the decision, not decoration.  It did not exist.

        `themed_confirm` is used with the default accent: a teal
        branch would be new work in `theme.py`, and this phase limits itself to
        Step 1's own two files.
        """
        from ..ui_shell.theme import themed_confirm
        shown = list(names)[:12]
        listing = "\n".join(f"  • {n}" for n in shown)
        if len(names) > len(shown):
            listing += f"\n  … and {len(names) - len(shown)} more"
        dest = ""
        try:
            dest = (self.widget.output_file_path.text() or "").strip()
        except Exception:
            pass
        where = f"\n\nInto: {dest}" if dest else ""
        return themed_confirm(
            self.widget,
            "Export to Library?",
            f"{len(names)} patch(es) will be written to disk. This is "
            f"permanent — names are allocated now.\n\n{listing}{where}",
            confirm_text=f"Export {len(names)}",
        )

    def _run_patch_export_loop(self, names, *, require_conditioned, noun,
                               success_tail=""):
        """Write *names* to disk one patch at a time, driving ④'s bar honestly.

        The ONE export loop (Brick 5 fix 3).  Both callers used to own a copy;
        only this one pumped Qt, so "Export all ready" rendered its determinate
        bar at rest for the whole run and then jumped to idle — a decorative
        bar, which is worse than none.

        Still MAIN-THREAD.  ``tifffile.imwrite`` per patch is the cost, and
        moving it to a worker cannot reuse ``_ExtractWorker`` (that one slices
        the SOURCE volume from boxes and is the sole ``step1_state.json`` writer
        while it runs); a second state-writing worker is a data-integrity
        change, not an indicator change, so it is deliberately not attempted
        here.  What makes the bar honest instead:

        * ``QApplication.processEvents()`` per patch, so the bar really
          repaints.  Safe: ``_set_step1_busy(True)`` has already locked the
          surface, so the drained events cannot re-enter another Step-1 op.
        * a ``_left_step1`` break — ``processEvents`` can deliver a step switch,
          and going on writing patches into a step the user has left (behind a
          bar they can no longer see) is exactly what the pump makes possible.

        Terminal funnel: the bar returns to rest and the lock releases in the
        ``finally``, on success, failure and the early break alike.
        """
        from qtpy.QtWidgets import QApplication
        exported = failed = 0
        total = len(names)
        stopped = False
        bar = getattr(self.widget, "export_progress_bar", None)
        self._set_step1_busy(True)
        try:
            for i, patch_name in enumerate(list(names), start=1):
                if bar is not None:
                    # Brick 5b item 5: a bar whose C++ object has gone (napari
                    # closing mid-export) must not abort the loop — drop it and
                    # keep writing the files the user asked for.
                    try:
                        bar.setRange(0, total)
                        bar.setValue(i)
                        bar.setFormat(f"Exporting {i} of {total}  %p%")
                    except Exception:     # noqa: BLE001 - deleted C++ object
                        bar = None
                    else:
                        QApplication.processEvents()
                if getattr(self, "_left_step1", False):
                    stopped = True
                    break
                if self._export_patch_by_name(
                        patch_name, require_conditioned=require_conditioned):
                    exported += 1
                else:
                    failed += 1
        except BaseException:
            # Never leave the PREVIOUS run's "Exported N" standing after a
            # failure — a stale success line on a permanent action is worse
            # than a blunt one.
            self._set_export_status(
                f"Export failed after {exported} of {total}.", success=False)
            raise
        finally:
            # Brick 5b item 5: the bar write comes FIRST in this funnel, so a
            # bar whose C++ object has already gone (napari closing mid-export)
            # raised here and the surface stayed locked forever.  The lock is
            # the thing that must always be released; the bar is cosmetic.
            try:
                if bar is not None:
                    bar.setRange(0, 1)
                    bar.setValue(0)
                    bar.setFormat("")
            except Exception:             # noqa: BLE001 - deleted C++ object
                pass
            self._set_step1_busy(False)
        if stopped:
            self._set_export_status(
                f"Export stopped after {exported} of {total} — you left Step 1.",
                success=False)
        elif failed:
            self._set_export_status(
                f"Exported {exported} {noun}; {failed} failed.", success=False)
        else:
            self._set_export_status(
                f"Exported {exported} {noun}{success_tail}.")
        self._refresh_conditioning_status()

    def _export_named_patches(self, names):
        """Write an explicit set of patches, with a real per-patch bar."""
        if getattr(self, "_migration_offer_timer", None) is not None:
            self._migration_offer_timer.stop()  # no file-move mid-export (R2)
        return self._run_patch_export_loop(
            names, require_conditioned=False, noun="patch(es)")

    def _export_current_patch(self):
        if getattr(self, "_migration_offer_timer", None) is not None:
            self._migration_offer_timer.stop()  # no file-move mid-export (R2)
        current_item = self.widget.extracted_roi_list.currentItem()
        if current_item is None:
            self._set_export_status("No patch selected.", success=False)
            return

        patch_name = current_item.text()
        # Brick 5 fix 3: this wrote a whole TIFF on the main thread with NO
        # indicator and NO lock — on slow storage the UI simply froze, and a
        # second click could start a second write of the same patch.  One patch
        # has no denominator, so ④'s bar goes to marquee; the lock is the same
        # `_set_step1_busy` the multi-patch loop takes.  Terminal `finally`.
        bar = getattr(self.widget, "export_progress_bar", None)
        self._set_step1_busy(True)
        try:
            if bar is not None:
                try:
                    bar.setRange(0, 0)      # marquee — one file, no percentage
                    bar.setFormat(f"Exporting {patch_name}…")
                except Exception:         # noqa: BLE001 - deleted C++ object
                    bar = None
            self._export_patch_by_name(patch_name, require_conditioned=False)
        finally:
            # See the sibling funnel (Brick 5b item 5): never let a dead bar
            # strand the surface lock.
            try:
                if bar is not None:
                    bar.setRange(0, 1)
                    bar.setValue(0)
                    bar.setFormat("")
            except Exception:             # noqa: BLE001 - deleted C++ object
                pass
            self._set_step1_busy(False)

    def _on_export_all_ready_clicked(self):
        if getattr(self, "_migration_offer_timer", None) is not None:
            self._migration_offer_timer.stop()  # no file-move mid-export (R2)
        if not hasattr(self, "extracted_patches") or not self.extracted_patches:
            self._set_export_status("No extracted patches available.", success=False)
            return

        ready_names = [
            name
            for name, info in self.extracted_patches.items()
            if info.get("selected_z_range")
        ]
        if not ready_names:
            self._set_export_status("No ROIs with Z-range set. Select Z-range first.", success=False)
            return

        # ④ gets its own bar under Export, with a real denominator — one
        # tick per patch.  Brick 5 fix 3: this loop set the bar but never pumped
        # Qt, so it rendered AT REST for the whole export and then jumped to
        # idle.  Shared with `_export_named_patches` now — one loop, one pump,
        # one terminal funnel + `_left_step1` break (`_run_patch_export_loop`).
        return self._run_patch_export_loop(
            ready_names, require_conditioned=True, noun="ROI(s)",
            success_tail=" successfully")

    def _persist_patch_to_disk(self, patch_name: str,
                               require_conditioned: bool = False):
        """Write ONE staged patch to disk and commit its box signature.

        Writes the patch TIFF + FROZEN sidecar + registry row into
        ``artifacts/step1/patches/`` (via :func:`persist_patch`) and commits
        the box signature so a re-extract of the same box is deduped — but does
        NOT pop the in-memory stage or the UI list.  Split out of
        :meth:`_export_patch_by_name` so ③ Extract can write each patch
        immediately (F1) while ④'s session list still shows the kept stage.

        Returns the ``persist_patch`` result dict on success, or ``None`` on any
        skip/failure (so callers can count honestly — N3).  Idempotent-safe if
        called twice for the same name: ``persist_patch`` allocates a fresh
        unique name and the sig commit is a set-add.

        This is now a thin MAIN-THREAD wrapper: it gathers the per-run persist
        params via :meth:`_gather_persist_params` (the same seam the ③ Extract
        worker uses — F1) and then calls the pure :func:`persist_patch`.  The ④
        per-patch export callers (`_export_patch_by_name`, the Z-range path) go
        through here unchanged.
        """
        params = self._gather_persist_params()
        path = params["workspace"]
        if not path:
            self._set_export_status("No workspace path set.", success=False)
            return None

        patch_info = self.extracted_patches.get(patch_name)
        if patch_info is None:
            self._log("STEP1-ZRANGE", f"No data found for patch: {patch_name}")
            return None

        data = patch_info["data"]  # shape: (C, Z, Y, X)
        original_shape = data.shape

        # Apply patch-specific Z-cropping if available
        z_range = patch_info.get("selected_z_range")
        if require_conditioned and not z_range:
            self._log("STEP1", f"Skipping '{patch_name}' (no selected Z-range).")
            return None

        if z_range:
            z0, z1 = z_range
            data = data[:, z0:z1, :, :]
            cropped_z_range = [int(z0), int(z1)]
        else:
            cropped_z_range = [0, int(original_shape[1])]

        # Materialize the crop BEFORE persist so `tifffile.imwrite` never
        # receives a lazy dask/zarr view — the processed-path crop can be lazy;
        # the raw path is already a real ndarray, so this is a no-op there.
        if hasattr(data, "compute"):
            data = data.compute()
        else:
            data = np.asarray(data)

        # Persist into artifacts/step1/patches/ + register in step1_state.json.
        # Sidecar schema is unchanged (Viz contract); collision auto-increment
        # + source provenance are handled inside persist_patch.  The per-run
        # provenance (source path/display-name/axes + voxel size) came from
        # `_gather_persist_params` above (`params`, the F1 gather seam).
        try:
            result = persist_patch(
                path,
                patch_name,
                data,
                channel_names=patch_info["channels"],
                original_shape=original_shape,
                cropped_shape=data.shape,
                selected_z_range=cropped_z_range,
                bounding_box=patch_info.get("bounding_box", "not stored yet"),
                source_path=params["source_path"],
                source_display_name=params["source_display_name"],
                source_axes=params["source_axes"],
                voxel_size=params["voxel_size"],
                # Provenance: whether these pixels came from the raw
                # source or the prepared processed store.  Stamped on the patch
                # REGISTRY row, never the frozen sidecar.  Additive; defaults to
                # "raw" for legacy staged patches missing the key.
                source_kind=patch_info.get("source_kind", "raw"),
                source_from=params.get("source_from"),
            )
        except Exception as e:
            self._set_export_status(f"Failed to export {patch_name}: {e}", success=False)
            return None
        final_name = result["patch_name"]
        self._log("STEP1", f"Saved patch to: {result['tiff_path']}")

        self._set_export_status(f"Exported '{final_name}' successfully.")

        # Commit this box's signature so it isn't re-extracted next time (it's
        # now on disk).  Uses the geometry captured at extract time -> correct
        # even if the layer was reordered/deleted since.  Commit BOTH the
        # XY-only sig (preserves the legacy Shapes-fallback dedup, which has no
        # per-box Z) AND the Z-inclusive sig keyed on the SAME range written to
        # the sidecar (`cropped_z_range`) — the box-tool re-extract check now
        # produces Z-inclusive sigs, so without this a same-box re-extract would
        # no longer dedup.  A 4-tuple never collides with a 6-tuple, so the
        # XY-only sig can't falsely drop a genuinely different Z-slab box.
        bb = patch_info.get("bounding_box")
        if isinstance(bb, list):
            self._committed_box_sigs.add(box_signature(bb))
            self._committed_box_sigs.add(box_signature(bb, cropped_z_range))

        return result

    def _export_patch_by_name(self, patch_name: str, require_conditioned: bool) -> bool:
        """Export one staged patch, then drop it from the stage + UI list.

        The disk write + box-sig commit live in
        :meth:`_persist_patch_to_disk`; this wrapper adds the pop / list-remove
        + count refresh that the ④ per-patch export paths still want (③ Extract
        writes via :meth:`_persist_patch_to_disk` directly, keeping the stage).
        Returns True on success, False on any skip/failure (so export-all can
        count honestly — N3)."""
        result = self._persist_patch_to_disk(
            patch_name, require_conditioned=require_conditioned)
        if result is None:
            return False
        final_name = result["patch_name"]

        # Clean up after successful export.  NB: use the ORIGINAL extracted
        # name (the dict key + the UI item text) for cleanup — persist_patch
        # may have assigned a different on-disk name on collision.
        self.extracted_patches.pop(patch_name, None)
        for idx in range(self.widget.extracted_roi_list.count()):
            item = self.widget.extracted_roi_list.item(idx)
            if item and item.text() == patch_name:
                self.widget.extracted_roi_list.takeItem(idx)
                break
        self._log("STEP1", f"Patch '{final_name}' removed from memory and UI.")
        self._refresh_conditioning_status()
        self._refresh_source_patch_counts()
        return True

    def _on_roi_selection_changed(self):
        self._refresh_conditioning_status()
        # Selecting a patch auto-draws its depth profile (the redundant
        # "Plot Intensity" button was removed).  Deselecting (empty selection)
        # clears the plot to a placeholder so a stale profile can't linger.
        if self.widget.extracted_roi_list.currentItem() is None:
            self._clear_intensity_profile()
            self._clear_highlight()   # empty selection → drop any white highlight
            return
        # Clicking a DIFFERENT patch clears the double-click highlight (the white
        # box follows your attention); re-selecting the highlighted patch keeps it.
        cur_name = self.widget.extracted_roi_list.currentItem().text()
        if (getattr(self, "_highlighted_patch", None) is not None
                and self._highlighted_patch != cur_name):
            self._clear_highlight()
        # Spurious re-selection of the ALREADY-plotted patch (a section switch /
        # focus change re-fires itemSelectionChanged) → skip the band-reset AND
        # the pyqtgraph rebuild: the main-thread rebuild each section-switch is
        # the stutter, and re-plotting would also wipe a dragged band.  A GENUINE
        # patch change falls through; the trim-redraw calls the plot directly
        # (bypassing this funnel), so it still re-renders.
        if (cur_name == getattr(self, "_plot_n_z_patch", None)
                and self._plot_area_has_widget()):
            # A prior genuine select set selected_z_range=None but deferred its
            # render; skipping back onto this still-plotted patch must re-adopt the
            # SHOWN band so selected_z_range can't contradict it (else Trim would
            # falsely say "no range" while a band is drawn — audit #1).
            if getattr(self, "selected_z_range", None) is None:
                try:
                    region = getattr(self, "_zrange_region", None)
                    if region is not None:
                        lo, hi = region.getRegion()
                        self.selected_z_range = (int(round(lo)), int(round(hi)))
                except Exception:
                    pass
            return
        # Reset the Z-truncate band to THIS patch's full range on select — the
        # previously-selected patch's dragged range must NOT carry over (else Trim
        # would cut the wrong slab).  `_render_intensity_profile` re-seeds the band
        # to the full (0, n_z) and re-adopts it as the current selection.
        self.selected_z_range = None
        self._on_plot_intensity_clicked()

    def _plot_area_has_widget(self):
        """True when the ④ depth-plot area currently holds a rendered plot widget
        (so a same-patch re-select can skip the rebuild).  Guarded."""
        try:
            layout = self.widget.plot_area.layout()
            return layout is not None and layout.count() > 0
        except Exception:
            return False

    def _refresh_conditioning_status(self):
        patches = getattr(self, "extracted_patches", {})
        total = len(patches)
        conditioned = sum(1 for info in patches.values() if info.get("selected_z_range"))

        if hasattr(self.widget, "set_roi_queue_summary"):
            self.widget.set_roi_queue_summary(total, conditioned)

        current_item = self.widget.extracted_roi_list.currentItem()
        if current_item is None:
            if hasattr(self.widget, "set_selected_roi_status"):
                self.widget.set_selected_roi_status("", None)
            return

        patch_name = current_item.text()
        patch_info = patches.get(patch_name, {})
        z_range = patch_info.get("selected_z_range")

        if hasattr(self.widget, "set_selected_roi_status"):
            self.widget.set_selected_roi_status(patch_name, z_range)

    # ------------------------------------------------------------------
    # Step lifecycle hooks (called by workflow shell on step switch)
    # ------------------------------------------------------------------

    def on_step_exit(self):
        """Called when user navigates away from Step 1.

        Tears down the ③ box tool's visuals (hide-before-remove on the Shapes
        layer) and restores the prior ``ndisplay`` so a Shapes-in-3D layer is
        never left dangling on the shared canvas.  The BoxRecs themselves are
        kept (still the extract source of truth) so re-entering ③ restores them.
        """
        # A ④ Trim's FINISH touches the shared canvas (`_rebuild_loc_boxes`) +
        # ④ panels — mark the step as LEFT so a late finish (after this exit)
        # skips that refresh + only unlocks (late-finish-inert).
        self._left_step1 = True
        # Belt-and-suspenders: clear the ④ scan bar so an in-flight depth-load /
        # capture whose terminal gets disconnected on exit can't leave it stuck.
        self._set_review_busy(False)
        # Brick B: turn the live scale bar OFF so it never leaks into another
        # workflow step (re-applied on re-entry if the toggle is still checked).
        try:
            self.viewer.scale_bar.visible = False
        except Exception:
            pass
        # Context change: land any pending source-edit save before we leave the
        # step (flush before a context change).  HIGH-1 (FIX 2a): while an
        # extract worker is in flight it is the SOLE `step1_state.json` writer, so
        # this flush must be DEFERRED (a main-thread `load_state`→`save_state`
        # here would race the worker's per-patch `persist_patch`).  It is re-run
        # in `_finish_extract_unlock` once the run ends (FIX 2b).
        if not getattr(self, "_extract_running", False):
            self._flush_source_state_save()
        # A source read in flight must NOT finish onto another step's shared
        # canvas: cancel it (bumps the token so `_on_source_loaded` goes stale
        # and never calls `add_image`) + unlock, BEFORE we strip the layers.
        # Without this the async finish re-adds Step-1 Image layers after exit
        # — the pinned Step-3 all-Image-layers abort (both audits flagged this).
        if getattr(self, "_source_load_thread", None) is not None:
            try:
                self.cancel_source_load()
            except Exception as exc:
                self._log("STEP1-LOAD", f"cancel on step-exit: {exc}")
        # A large-data convert in flight must likewise not finish onto another
        # step's shared canvas: cancel it (bumps the token so
        # `_on_large_data_ready` goes stale and never calls `add_image`) +
        # unlock, BEFORE we strip the layers.
        if getattr(self, "_largedata_thread", None) is not None:
            try:
                self.cancel_large_data_prep()
            except Exception as exc:
                self._log("STEP1-PREP", f"cancel on step-exit: {exc}")
        # An extract in flight is REQUEST-cancelled (no token bump, no force-
        # unlock): extract FINISH touches only Step-1 panel widgets + the
        # registry — NOT the shared napari canvas — so the worker finalizes
        # naturally + `_on_extract_finished` performs the one unlock even after
        # we leave the step.  Retain the (thread, worker) against GC so the
        # QThread is never destroyed mid-flight (mirror the source-load drain).
        if getattr(self, "_extract_thread", None) is not None:
            try:
                self._draining_extract_threads.append(
                    (self._extract_thread, self._extract_worker))
                self.cancel_extract()
            except Exception as exc:
                self._log("STEP1-EXTRACT", f"cancel on step-exit: {exc}")
        # A ④ Trim in flight is NOT cancellable (it must finish its atomic
        # replace — a half-done crop would corrupt the patch).  RETAIN the
        # (thread, worker) against GC so its QThread is never destroyed
        # mid-replace; it finalizes naturally and `_on_trim_finished`'s
        # `_left_step1` guard makes the ④ refresh inert while the one unlock
        # still fires (so the surface isn't left locked on step re-enter).
        if getattr(self, "_trim_thread", None) is not None and getattr(
                self, "_trim_running", False):
            try:
                self._draining_trim_threads.append(
                    (self._trim_thread, self._trim_worker))
            except Exception as exc:
                self._log("STEP1-TRIM", f"drain on step-exit: {exc}")
        box_tool = getattr(self, "box_tool", None)
        if box_tool is not None:
            try:
                box_tool.stop()
            except Exception as exc:
                self._log("STEP1-BOX", f"box-tool teardown on step-exit: {exc}")
        # Tear down the transient ⟨preview⟩ render + the REGION box + its overlay
        # and collapse the ② controls so nothing strands on another step's shared
        # canvas (single entry point; NOT re-shown on step re-enter — a fresh
        # Place / Preview rebuilds them).
        try:
            self._teardown_preview()
        except Exception as exc:
            self._log("STEP1-PREVIEW", f"preview teardown on step-exit: {exc}")
        # Remove Step 1's source Image layers so they don't strand on the shared
        # canvas (the pinned Step-3 all-Image-layers abort).  The channel refs
        # are kept so a Load can rebuild the exact same display cheaply, but
        # `on_step_enter` does NOT re-show them (Brick 5) — so until the user
        # presses Load, Extract and ③ box mode decline with `_OFF_CANVAS_HINT`
        # rather than silently reading the wrong thing.
        # Brick 5: remember that a source WAS on the canvas.  `on_step_enter`
        # deliberately does NOT re-add it (see `_note_source_unloaded_on_reentry`);
        # this flag is only what lets ② explain the empty canvas on return.
        self._source_needs_reload = bool(
            getattr(self, "_source_display_channels", None))
        try:
            self._remove_source_layers_from_viewer()
        except Exception as exc:
            self._log("STEP1-BOX", f"source-layer removal on step-exit: {exc}")
        # ④ patch-location overlay: the one box artifact on the shared canvas with
        # no prior step-exit teardown (audit HIGH-1) — clear it so its vispy Lines
        # + dims subscription don't strand on the next step's canvas (freed-VBO
        # crash surface + a repaint on foreign steps).
        try:
            self._clear_patch_locations()
        except Exception as exc:
            self._log("PATCHLOC", f"clear on step-exit: {exc}")
        # The streaming ⟨processed⟩ progressive layers were the one Step-1 layer
        # class with NO step-exit teardown: they stranded on the shared canvas
        # (the pinned Step-3 all-Image-layers abort) and, worse, kept a
        # progressive LOADER live — which makes the `ensure_sync_slicing` guard
        # below correctly decline, so the persisted `experimental.async_` poison
        # survived into Steps 2-5 anyway.  Cleared here, before that guard.
        try:
            self._clear_processed_store_layers()
        except Exception as exc:
            self._log("STEP1-PROC", f"processed-layer clear on step-exit: {exc}")
        # Close any open capture windows (each owns a standalone GL context) so
        # they don't leak a context or show stale patches after a workspace switch
        # (audit MED).  Shared with `_teardown_for_workspace_change` (B5).
        self._close_capture_dialogs()
        # Retire the intensity-plot worker crash-safely (audit HIGH): a long
        # `imread` in flight must not be destroyed-while-running at teardown.
        try:
            self._detach_intensity_worker()
        except Exception as exc:
            self._log("PLOT-DISK", f"intensity detach on step-exit: {exc}")
        # LAST: hand the shared viewer back with SYNCHRONOUS slicing.  The
        # progressive engine turns napari's persisted `experimental.async_` ON
        # and snapshots the settings AFTER doing so, so the flag survives into
        # Steps 2-5 (and into the next session) and makes their hidden-layer
        # recolour paths misrender.  Runs after the Step-1 layers are gone, so
        # the helper's "a loader is still live" decline is meaningful.
        try:
            from packages.epicycle_largedata.progressive import ensure_sync_slicing
            ensure_sync_slicing(self.viewer, reason="Step 1 exit")
        except Exception as exc:
            self._log("PROG", f"sync-slicing restore on step-exit: {exc}")

    # ------------------------------------------------------------------
    # ③ box-tool delegation (Phase 1)
    # ------------------------------------------------------------------

    def on_box_mode_toggled(self, checked):
        """The ③ "📦 Box mode" toggle: enter (arbitration → box overlay in the
        CURRENT view) or leave (teardown).  No longer forces 3D (Change 1).

        `[BOX-DIAG]` lines are TEMP diagnostics — strip before the baseline
        merge."""
        box_tool = getattr(self, "box_tool", None)
        if box_tool is None:
            return
        layers = getattr(getattr(self, "viewer", None), "layers", None)
        self._log(
            "BOX-DIAG",
            f"box-mode toggle checked={checked} "
            f"n_layers={len(layers) if layers is not None else '?'} "
            f"n_boxes={len(getattr(box_tool, 'boxes', []) or [])}")
        if checked:
            # Brick 5b item 2: entering box mode with the source off-canvas runs
            # a whole-store overview read on the main thread.  Decline BEFORE
            # arbitration (which would clear the preview region for a mode we
            # are not going to enter) and snap the toggle back off.  Toggle-OFF
            # is never gated — leaving a mode must never be blocked.
            if self._block_if_source_off_canvas("box mode"):
                box_tool._sync_mode_button(False)
                return
            # Mode arbitration (Piece 2): the preview region + the ③ box tool
            # are mutually exclusive.  `_arbitrate_for_box_mode` clears the placed
            # preview region AND (if viewing Processed) snaps the display back to
            # raw WITHOUT dropping the re-flippable processed view — so entering
            # box mode never empties the canvas (the "hides everything" bug).
            self._arbitrate_for_box_mode()
            if not box_tool.start():
                # Nothing to show (no source) — snap the toggle back off.
                box_tool._sync_mode_button(False)
        else:
            # Toggle OFF: keep the placed box wireframes on the canvas as ROI
            # markers (only the ghost + editing go away).  Full teardown happens
            # on step-exit / source-unload, not here.
            box_tool.stop(keep_visible=True)
        layers = getattr(getattr(self, "viewer", None), "layers", None)
        self._log(
            "BOX-DIAG",
            f"box-mode toggle DONE checked={checked} "
            f"n_layers={len(layers) if layers is not None else '?'} "
            f"on_canvas={getattr(self, '_source_layers_on_canvas', None)} "
            f"n_boxes={len(getattr(box_tool, 'boxes', []) or [])}")

    def toggle_box_ndisplay(self):
        """The ③ 2D⇄3D affordance (flip for click-to-place / view)."""
        box_tool = getattr(self, "box_tool", None)
        if box_tool is not None:
            box_tool.toggle_ndisplay()

    def add_box(self):
        """③ "Add box" — drop a full-Z box at the volume centre."""
        box_tool = getattr(self, "box_tool", None)
        if box_tool is None:
            return
        if not box_tool.is_active():
            if self._block_if_source_off_canvas("Add box"):
                return
            self._arbitrate_for_box_mode()   # raw-safe + processed-preserving
            box_tool.start()
        box_tool.add_box()

    def delete_box(self):
        """③ "Delete box" — remove the selected box."""
        box_tool = getattr(self, "box_tool", None)
        if box_tool is not None:
            box_tool.delete_active()

    def set_box_size_fraction(self, frac):
        """③ S/M/L preset — resize the active box + remember the sticky size for
        the next Add.  A NO-OP when box mode is OFF: there is no box to resize,
        and starting box mode (flipping the canvas to 3D) as a side effect of a
        size click is surprising."""
        box_tool = getattr(self, "box_tool", None)
        if box_tool is None or not box_tool.is_active():
            return
        box_tool.set_size_fraction(frac)

    def set_box_size_voxels(self, size):
        """③ absolute-size preset (128³/256³/512³ + custom) — set the sticky
        voxel XY size for the next Add + resize the active box (keeps its Z).
        A NO-OP when box mode is OFF, mirroring `set_box_size_fraction`."""
        box_tool = getattr(self, "box_tool", None)
        if box_tool is None or not box_tool.is_active():
            return
        box_tool.set_size_voxels(size)

    def set_box_locked(self, locked):
        """③ "lock" checkbox — freeze / unfreeze the SELECTED box's geometry so
        it cannot be moved or resized until unlocked.  A NO-OP when box mode is
        off (there is no selected box to lock)."""
        box_tool = getattr(self, "box_tool", None)
        if box_tool is None or not box_tool.is_active():
            return
        box_tool.set_active_locked(bool(locked))

    def set_box_edge_mode(self, mode):
        """③ "At edge" toggle — how a live geometry edit resolves a box that
        would overflow the volume edge: ``"shrink"`` (default, byte-identical
        extraction) or ``"keep"`` (preserve size, slide inward).  Persisted
        per-session on the box tool; safe when box mode is off (the setting
        takes effect on the next edit)."""
        box_tool = getattr(self, "box_tool", None)
        if box_tool is not None:
            box_tool.set_edge_mode(mode)

    def on_place_toggled(self, _checked=None):
        """③ click-to-place toggle — the callback reads the checkbox live, so
        this ensures box mode is running to place into.

        Click-to-place is 2D-gated (`_make_place_callback`: it no-ops unless
        `ndisplay == 2`).  Box mode no longer forces 3D (Change 1), so a user
        can be in a 3D inspect look when they tick "⌖ Click-to-place" — force
        the canvas back to the flat 2D view here so the click is never silently
        inert.  Only on toggle-ON; toggle-OFF leaves the view alone."""
        box_tool = getattr(self, "box_tool", None)
        if box_tool is None:
            return
        if not box_tool.is_active():
            if self._block_if_source_off_canvas("click-to-place"):
                return
            self._arbitrate_for_box_mode()   # raw-safe + processed-preserving
            box_tool.start()
        if _checked:
            box_tool._set_ndisplay(2)     # 2D so click-to-place isn't inert

    def _note_source_unloaded_on_reentry(self):
        """Step re-enter after a step-exit that removed the source layers.

        Brick 5 (user decision): Step 1 simply TEARS DOWN on exit and does not
        put the data back by itself — re-adding a prepared store cost 1.1-1.4 s
        on every re-entry, paid whether or not the user wanted the image back.

        Everything that is not about the CANVAS survives untouched: the source
        list + its selection, the recipe, the voxel size, ``_loaded_source_id``
        (so ④ Review & Library stays interactive for the source they were working
        on) and the ④ patch registry.  Only the layer-dependent bits are cleared,
        so no code path behaves as if the layers had come back.  Never raises.
        """
        if not getattr(self, "_source_needs_reload", False):
            return
        self._source_needs_reload = False
        # Layer-dependent state.  `_source_layers_on_canvas` is already False
        # (the removal set it) and gates the progressive / 3D-detail hooks; the
        # two below are the ones the re-show used to reset itself.
        self._source_layers_on_canvas = False
        self._progressive_active = False
        # Captured by `_remove_source_layers_from_viewer` purely to feed the
        # re-show's channel re-tick; with no re-show it would only go stale.
        self._saved_channel_selection = None
        self._set_source_status_amber(
            "Source unloaded when you left Step 1 — click Load to view it again.")

    # ------------------------------------------------------------------
    # Shared workspace
    # ------------------------------------------------------------------

    def adopt_workspace(self, path: str, previous: str = "") -> None:
        """Quietly take the shell's shared workspace.

        Called by the shell when the user enters Step 1, or when the shared
        workspace moves because the user browsed in ANOTHER step (symmetric
        rule).  The whole body is "put the path in the box": the
        `textChanged` chain and `on_step_enter` do the rest (the source
        restore that repopulates ②, which reads the registry and the
        `largedata/` stores but never loads pixels).

        `previous` is the workspace this step INHERITED before.  Step 1
        re-adopts only when its box is empty or still holds exactly that
        path; anything else in the box was put there by the user and is
        never overwritten (invariant I2).

        The layout-migration one-shot is PRE-SEEDED first, so adopt never
        throws the migration modal at a user who merely walked into the step.  `on_workspace_browsed` clears that seed, so an
        explicit Browse to the same folder still offers the migration.
        """
        if not path:
            return
        current = self.widget.output_file_path.text().strip()
        replacing = False
        if current:
            if not previous or not self._same_workspace(current, previous):
                return
            if self._same_workspace(current, path):
                return
            replacing = True
        if replacing:
            # A re-adopt IS a workspace switch, driven by the shell without
            # asking anyone.  If a Step-1 job owns the folder we are on, stay
            # put (the shell then shows this step as diverged) — B5.
            job = self._step1_job_in_flight()
            if job:
                self._log(
                    "WORKSPACE",
                    f"{job} is running, so Project Setup stayed on its current "
                    f"workspace.  Switch it with Browse once that has finished.")
                return
            self._teardown_for_workspace_change("the shared workspace moved")
        try:
            key = os.path.normcase(os.path.normpath(path))
            self._step1_migration_offered.add(key)
        except Exception:      # noqa: BLE001
            pass
        self._suppress_publish = True
        try:
            self.widget.output_file_path.setText(path)
        finally:
            self._suppress_publish = False

    @staticmethod
    def _same_workspace(a: str, b: str) -> bool:
        """Path comparison for the shared-workspace handshake (case + slash)."""
        if not a or not b:
            return False
        try:
            return (os.path.normcase(os.path.normpath(a))
                    == os.path.normcase(os.path.normpath(b)))
        except Exception:      # noqa: BLE001
            return a == b

    def on_workspace_browsed(self, path: str) -> bool:
        """The user picked a workspace with Step 1's Browse button.

        Browse is the only thing that publishes a workspace to the shell
 — and, unlike adopt, it releases the migration one-shot so
        picking the same folder deliberately still offers the layout
        migration.

        Returns **False** when the change is REFUSED because a Step-1 job
        owns the current workspace (B5); the UI asks before it does anything,
        so a refusal leaves the box — and the step — exactly where it was,
        and the running job keeps its folder and finishes normally.

        On an ALLOWED change this writes the path box ITSELF, before the
        publish.  The publish marks this step `browsed` and the shell then
        recomputes divergence by READING the box: telling the shell while the
        box still held the OLD folder entered Step 1 into `_diverged_steps`
        against itself, logged a spurious "diverged" line, and left the
        sidebar tooltip claiming "Browsed elsewhere: Project Setup → <old>"
        until the next step switch (lifecycle audit).  The order here is the
        fix; `ui.py` no longer writes the box at all.
        """
        if not path:
            return False
        current = self.widget.output_file_path.text().strip()
        changing = bool(current) and not self._same_workspace(current, path)
        if changing:
            # Refuse, never cancel: Extract / Prepare / Process / Trim are
            # minute-long writes the user asked for, and a Trim's `os.replace`
            # cannot be interrupted at all.  Cancellation stays on the Cancel
            # buttons, where the user asked for it.
            job = self._step1_job_in_flight()
            if job:
                self._log(
                    "WORKSPACE",
                    f"{job} is running, so the workspace was left unchanged.  "
                    f"Pick the new folder once that has finished.")
                # The user picked a folder and the box did not move — that has
                # to be visible where they are looking, not only in the log.
                self._set_workspace_status_amber(
                    "Finish or cancel the running job before changing the "
                    "workspace.")
                return False
            self._teardown_for_workspace_change("you changed the workspace")
        try:
            key = os.path.normcase(os.path.normpath(path))
            self._step1_migration_offered.discard(key)
        except Exception:      # noqa: BLE001
            pass
        # BEFORE the publish — see the docstring.
        self.widget.output_file_path.setText(path)
        self._publish_workspace_to_shell(path)
        return True

    #: The Step-1 jobs that own the current workspace while they run.  Each
    #: entry is (flag attribute, the plain words the user sees).
    #: Each label is the SUBJECT of "<label> is running, so …", so it has to
    #: read as a noun phrase — "A source is still loading is running" was the
    #: first draft (caught by the tests/UX audit).
    _WORKSPACE_BLOCKING_JOBS = (
        ("_source_loading", "A source read"),
        ("_largedata_preparing", "Prepare for large data"),
        ("_processing", "Processing the whole volume"),
        ("_extract_running", "An extraction"),
        ("_trim_running", "A trim"),
    )

    def _step1_job_in_flight(self) -> str:
        """Name the Step-1 job that owns this workspace, or `""` if none.

        Mirrors Step 2's `_training_in_flight` guard (and, behind it,
        `_switch_run_guard.check_switch_safety`): a workspace change while a
        job runs would pull the folder out from under a running write.  A
        missing flag means "not running" — these are set at launch and
        cleared in the finish handlers, so absence is genuinely idle, not
        unknown.
        """
        for attr, label in self._WORKSPACE_BLOCKING_JOBS:
            if getattr(self, attr, False):
                return label
        return ""

    def _teardown_for_workspace_change(self, reason: str) -> None:
        """Drop everything the workspace we are LEAVING still owns (B5).

        A workspace change is a PROJECT change: the pixels on the shared
        canvas, the overlays drawn over them and the identity of the loaded
        source all belong to a folder the user is walking away from.  Before
        B5 the `textChanged` chain saved state and re-listed the new
        workspace's sources but left all of that behind, so the previous
        project's source layers, ⟨processed⟩ store layers, ③ box, ④ location
        boxes and ⟨preview⟩ render stayed on screen under the new path — and
        `_loaded_source_id` still pointed into the old registry.

        Every step here DELEGATES to the helper that already owns it (the
        same ones `on_step_exit` uses), in this fixed order: flush
        the pending save FIRST (it was captured against the OLD workspace, so
        it must land there — the Step-2 gotcha), retire the off-thread
        readers, then strip the canvas, and unload the source LAST so its
        identity is cleared once nothing else needs it.

        Called from the two doors that can change the workspace — Browse
        (`on_workspace_browsed`) and a shell re-adopt (`adopt_workspace` with
        `previous` set) — and from neither the start-up pre-fill nor an
        unchanged path.

        The ② Sources list goes too, together with the once-per-session
        restore guard `_step1_sources_restored` that describes it — see
        `_clear_source_list_for_workspace_change` for why the two cannot be
        separated.  The NEW workspace's list is rebuilt by the restore that
        runs on the `textChanged` which follows.

        Best-effort throughout: a teardown that raises half-way would strand
        exactly the layers it exists to remove, so every step is guarded
        individually and the rest still run.

        In-flight workers are NOT cancelled here — `on_workspace_browsed` and
        `adopt_workspace` REFUSE the change while one is running, so by the
        time we get here there is nothing to cancel.
        """
        self._log("WORKSPACE",
                  f"Closing the previous workspace ({reason}): unloading its "
                  f"source and clearing its overlays.")
        # 1. The pending source-edit save belongs to the OLD workspace.
        self._guarded(self._flush_source_state_save,
                      "WORKSPACE", "source-state flush")
        # 2. Off-thread readers that would finish onto the new project.
        self._guarded(self._detach_intensity_worker,
                      "PLOT-DISK", "intensity detach")
        self._guarded(self._close_capture_dialogs,
                      "CAPTURE", "capture-dialog close")
        # 3. The overlays drawn OVER the source, innermost first.  `reset`
        #    (not `stop`) — the BoxRecs describe the old project's volume.
        box_tool = getattr(self, "box_tool", None)
        if box_tool is not None:
            self._guarded(box_tool.reset, "STEP1-BOX", "box-tool reset")
        self._guarded(self._teardown_preview, "STEP1-PREVIEW",
                      "preview teardown")
        self._guarded(self._clear_patch_locations, "PATCHLOC",
                      "location-overlay clear")
        self._guarded(self._clear_processed_store_layers, "STEP1-PROC",
                      "processed-layer clear")
        # 4. The source itself, which also clears `_loaded_source_id` /
        #    `_loaded_source_file` and the display-channel refs — so nothing
        #    re-shows the old project on the next step entry.
        self._guarded(self._unload_source_inner, "STEP1-LOAD",
                      "source unload")
        #    `on_step_exit` sets `_source_needs_reload` so ② can explain the
        #    empty canvas on the way back ("Source unloaded when you left
        #    Step 1 — click Load to view it again").  That sentence is about
        #    the PREVIOUS project: a re-adopt teardown runs while Step 1 is
        #    off screen, so without this the hint greets the user under a
        #    workspace whose source they never loaded.
        self._source_needs_reload = False
        # 5. LAST: the ② Sources list.  `_populate_source_list_from_registry`
        #    only ever ADDS (Browse deliberately MERGES scanned files within a
        #    workspace), so without this the previous project's rows sit under
        #    the new workspace's path — and clicking one would load a file
        #    that belongs to a folder the user has left.  Cleared after the
        #    unload, so the restore for the NEW workspace (which runs on the
        #    `textChanged` that follows) repopulates an empty list.
        #
        #    Emptying the list means the once-per-session restore guard has to
        #    go with it: `_step1_sources_restored` exists to keep the reconcile
        #    off every keystroke and step entry, and each key in it claims "this
        #    workspace's rows are already listed" — which we have just made
        #    false for all of them.  Left standing, a user who browsed A -> B
        #    and back would land on A with an EMPTY ② list (verified live).  One
        #    reconcile per workspace CHANGE is exactly what the guard was always
        #    meant to allow.
        self._guarded(self._clear_source_list_for_workspace_change,
                      "STEP1-RESTORE", "source-list clear")

    def _clear_source_list_for_workspace_change(self) -> None:
        """Empty ② Sources and drop the selection it described.

        Mirrors the folder-rebrowse rebuild at `browse_source_folder` — the
        same suspend-the-funnel guard, so the clear cannot restore a
        transient row on its way out.
        """
        restored = getattr(self, "_step1_sources_restored", None)
        if restored is not None:
            restored.clear()
        lst = getattr(getattr(self, "widget", None), "source_file_list", None)
        if lst is None:
            return
        self._suspend_source_restore = True
        try:
            lst.clear()
            self._selected_source_path = None
        finally:
            self._suspend_source_restore = False
        # Repaint the (now empty) mirror.  The restore that follows on the
        # `textChanged` refreshes it when the new path is a real directory —
        # but when it is not (a typed path, a folder that has gone), nothing
        # else runs and the old workspace's rows would linger in the combo.
        refresh = getattr(self, "_refresh_source_file_list", None)
        if callable(refresh):
            refresh()

    def _guarded(self, fn, prefix: str, what: str) -> None:
        """Run one teardown step; log and continue if it raises."""
        try:
            fn()
        except Exception as exc:      # noqa: BLE001
            self._log(prefix, f"{what} on workspace change: {exc}")

    def _close_capture_dialogs(self) -> None:
        """Close every open capture window (each owns a standalone GL context).

        Each `closeEvent` frees its canvas and prunes `_capture_dialogs`, so
        this is safe to call twice.  Shared by `on_step_exit` and
        `_teardown_for_workspace_change`.
        """
        for dlg in list(getattr(self, "_capture_dialogs", None) or []):
            try:
                dlg.close()
            except Exception:      # noqa: BLE001
                pass

    def _publish_workspace_to_shell(self, path: str) -> None:
        """Tell the shell about a workspace the USER chose here.

        Suppressed for the duration of an adopt, so a path the shell handed
        DOWN can never be published back UP as if the user had picked it.
        """
        if not path or getattr(self, "_suppress_publish", False):
            return
        shell = getattr(self, "_workflow_shell", None)
        setter = getattr(shell, "set_workspace", None)
        if not callable(setter):
            return
        try:
            setter(path, source_step="project_setup")
        except Exception as exc:      # noqa: BLE001
            self._log("WORKSPACE", f"publish from Step 1 failed: {exc}")

    def on_step_enter(self):
        """Called when user navigates to Step 1.  Offers the layout migration if
        the already-set workspace has loose legacy patches, and explains the empty
        canvas left by the last step-exit (Step 1 does not reload by itself)."""
        # FIRST: hand Step 1 an ASYNCHRONOUS-slicing viewer.  Step 1 browses 2D
        # slices of a chunked store, and async slicing takes that fetch off the
        # UI thread -- measured live on the wound store, cost per 2D Z step
        # 36 -> 18 ms on the (32, 128, 128) shape.  It is the exact inverse of
        # `on_step_exit`'s `ensure_sync_slicing`, which exists because the
        # progressive engine PERSISTS the flag and would otherwise poison Steps
        # 2-5; this pair keeps the setting scoped to the step that wants it.
        #
        # Before the re-show below, deliberately: the layers re-added there
        # should be sliced under the policy we are switching to, not the one we
        # are leaving.  Guarded -- a settings backend that cannot be written is
        # a perf regression, never a broken step entry.
        try:
            from packages.epicycle_largedata.progressive import (
                ensure_async_slicing)
            ensure_async_slicing(self.viewer, reason="Step 1 enter")
        except Exception as exc:      # noqa: BLE001
            self._log("PROG", f"async-slicing on step-enter: {exc}")
        # Back on Step 1: a ④ Trim FINISH may again refresh the panels + canvas.
        self._left_step1 = False
        # Brick 5 (user decision): Step 1 tears down on exit and does NOT reload
        # the source automatically on re-entry.  Re-adding a prepared store cost
        # 1.1-1.4 s on EVERY step re-enter, for data the user may not want back.
        self._note_source_unloaded_on_reentry()
        # Brick B: re-apply the live scale bar (step-exit turned it off) so a
        # still-checked toggle comes back with its styling on return.
        try:
            self._apply_live_scale_bar()
        except Exception:
            pass
        # Restore the workspace's saved sources into ② (once per workspace):
        # reconcile the registry against the largedata/ stores + repopulate the
        # source list so a returning session doesn't re-specify.
        self._maybe_restore_step1_sources()
        self._maybe_offer_step1_migration()
        # progressive default-open is evaluated on step-enter.
        # It was built and never called — a UI behaviour with test-only
        # callers is not shipped.
        opener = getattr(self.widget, "apply_progressive_default_open", None)
        if opener is not None:
            opener()

    # ------------------------------------------------------------------
    # Layout migration auto-prompt
    # ------------------------------------------------------------------
    def _maybe_offer_step1_migration(self, _=None):
        """Offer the legacy→``artifacts/step1/patches`` migration at most once
        per valid workspace path per session.

        Fires from the workspace ``textChanged`` signal (covers Browse) and
        from ``on_step_enter`` (covers an already-set workspace).  The pure
        decision (needed AND not skipped) lives in ``_migrate_step1_layout``.
        """
        workspace = self.widget.output_file_path.text().strip()
        if not workspace or not os.path.isdir(workspace):
            return
        # Normalize the guard key so path variants (trailing slash, case,
        # forward/back slashes) count as the same workspace.
        key = os.path.normcase(os.path.normpath(workspace))
        if key in self._step1_migration_offered:
            return
        try:
            from ._migrate_step1_layout import should_offer_migration
            if not should_offer_migration(workspace):
                self._step1_migration_offered.add(key)
                return
        except Exception as e:
            self._log("STEP1-MIGRATE", f"decision check failed: {e}")
            return
        # Mark BEFORE prompting so a re-entrant signal can't double-open it.
        self._step1_migration_offered.add(key)
        try:
            from ._migrate_step1_dialog import prompt_and_migrate_if_needed
            _choice, result = prompt_and_migrate_if_needed(self.widget, workspace)
            if result is not None and result.moved:
                self._refresh_source_patch_counts()
        except Exception as e:
            self._log("STEP1-MIGRATE", f"prompt failed: {e}")





