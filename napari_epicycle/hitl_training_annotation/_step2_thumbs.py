"""Step 2's thumbnails -- the engine, no UI.

BRICK L built the PATCH half; BRICK A adds the SLICE half at the end of this
file (``SliceThumbRequest``, ``render_slice_thumb``, the slice worker, the
in-flight set, the trailing debounce, the slices sweep).

A picture of each patch's raw TIFF -- the middle lateral (XY) plane of up
to three channels, no labels, never following the View switch -- for the
Session list (128 px) and Curation's Patches list (the same PNG, scaled to
80 px by the delegate).  The picture is made by the SHARED machinery of
``ui_shell/_stack_thumbs.py`` (``render_thumb`` -> ``compose_rgb`` ->
``encode_png``; one channel through ``SINGLE_INFERNO``, two or three the
magenta / green / blue composite; ``ThumbCache`` on the main thread); this
module builds the requests, names the cache folder, starts / stops one
batch and sweeps PNGs of patches that are no longer listed.

Folders (D25): ``artifacts/step2/thumbs/patches/`` holds ONE 128 px PNG per
patch.  ``artifacts/step2/thumbs/slices/`` (brick A) is a SEPARATE folder
with its own sweep: ``sweep_unlisted_patches`` is only ever pointed at the
patches folder, never at the parent, because slice PNGs are named by sample
id.  Why one size per patch: the shared ``stale_pngs`` matches the same safe
name AND path hash at ANY size token, so a 128 px and an 80 px PNG of one
patch would delete each other on every refresh.

The axes check: the shared worker draws whatever axes it is
handed, and ``resolve_patch_axes`` never fails (it falls back to ``CZYX``),
so Step 2 reads the TIFF HEADER itself -- on the worker thread, in
:func:`probe_patch_header` -- and accepts only a series axes of exactly
``ZYX``, ``CZYX`` or ``ZCYX``; anything else is a blank tile with a hover
reason, never a guess.  The probe's result (axes, shape, channel count)
rides the worker's extra ``meta`` signal, because ``ready`` carries only
name / path / png path / bytes; the rows' "2 ch" comes from it, never from a
main-thread header read.

Threads: the request builder runs on the main thread
and only ``os.stat``\\ s each file.  Every file open is on the worker.  The
worker's thread is never parented and is parked in the shared live set, so
a deleted parent cannot destroy it mid-run; the owner of the batch stops it
from ``on_shutdown`` (bounded wait).

Qt-free at import: the shared worker (``qtpy``) is imported inside the
functions that need it.
"""
from __future__ import annotations

import json
import os
import re
import stat as _stat
from pathlib import Path
from typing import Iterable, List, NamedTuple, Optional

from . import _paths

#: Patch suffixes that get a picture (``.npy`` patches are skipped).
THUMB_SUFFIXES = (".tif", ".tiff")
#: ``artifacts/step2/thumbs`` and its two sub-folders (D25).
THUMBS_SUBDIR = "thumbs"
PATCHES_SUBDIR = "patches"
SLICES_SUBDIR = "slices"
#: How long a shutdown waits for a running batch (Step 4's rule).
STOP_WAIT_MS = 3000
#: The ONE size of a patch PNG: the wide row's picture; the narrow row
#: scales it (see the module docstring).
PATCH_THUMB_PX = 128
#: The only series axes a patch picture is drawn from.
ACCEPTED_AXES = ("ZYX", "CZYX", "ZCYX")

#: Whose the slice half is.  Built by brick A (below); the name stays because
#: brick L's ``test_step2_thumbs`` pins it.
SLICE_HALF = "brick A"


def _diag(msg: str) -> None:
    try:
        print(f"[STEP2-THUMB] {msg}", flush=True)
    except Exception:       # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------

def step2_thumbs_dir(workspace) -> Path:
    """``{ws}/artifacts/step2/thumbs`` -- the parent of the two folders; no
    sweep is ever pointed here.  A disposable cache; naming it creates
    nothing (the worker creates a folder on its first write)."""
    return _paths.step2_root(workspace) / THUMBS_SUBDIR


def patch_thumbs_dir(workspace) -> Path:
    """``.../thumbs/patches`` -- one 128 px PNG per patch."""
    return step2_thumbs_dir(workspace) / PATCHES_SUBDIR


def slice_thumbs_dir(workspace) -> Path:
    """``.../thumbs/slices`` -- brick A's folder (named here so the two
    sweeps can never be confused)."""
    return step2_thumbs_dir(workspace) / SLICES_SUBDIR


# ---------------------------------------------------------------------------
# The header probe (worker thread)
# ---------------------------------------------------------------------------

class PatchMeta(NamedTuple):
    """What the header says.  ``reason`` is ``""`` when the patch is
    drawable, else the hover sentence (the axes do not fit)."""
    name: str
    axes: str
    shape: tuple
    n_channels: int
    reason: str

    def to_json(self) -> str:
        return json.dumps({"name": self.name, "axes": self.axes,
                           "shape": list(self.shape),
                           "n_channels": self.n_channels,
                           "reason": self.reason})

    @staticmethod
    def from_json(text: str) -> "PatchMeta":
        d = json.loads(text)
        return PatchMeta(str(d.get("name", "")), str(d.get("axes", "")),
                         tuple(int(v) for v in d.get("shape", ())),
                         int(d.get("n_channels", 1)),
                         str(d.get("reason", "")))


def probe_patch_header(path, name: Optional[str] = None) -> PatchMeta:
    """Open the TIFF's HEADER (no pixel data) -> :class:`PatchMeta`.  Worker
    thread only.  Never raises: an unreadable file is a reason, not a
    crash."""
    name = name if name is not None else os.path.basename(str(path))
    try:
        import tifffile

        with tifffile.TiffFile(str(path)) as tf:
            series = tf.series[0]
            axes = str(series.axes).upper()
            shape = tuple(int(v) for v in series.shape)
    except Exception as exc:        # noqa: BLE001 -- a bad file, not a bug
        _diag(f"{name}: header unreadable ({type(exc).__name__}: {exc})")
        return PatchMeta(name, "", (), 1, _reason_unreadable())
    reason = axes_reason(axes, shape)
    n_ch = shape[axes.index("C")] if (not reason and "C" in axes) else 1
    return PatchMeta(name, axes, shape, int(n_ch), reason or "")


def _reason_unreadable() -> str:
    from .ui import _step2_text as T

    return T.THUMB_REASONS["THUMB_FAIL_UNREADABLE"]


def axes_reason(axes: str, shape) -> Optional[str]:
    """``None`` when *axes* is exactly ``ZYX`` / ``CZYX`` / ``ZCYX`` and has
    one entry per dimension of *shape*; else the hover reason."""
    axes = str(axes or "").upper()
    try:
        n = len(tuple(shape))
    except TypeError:
        n = -1
    if axes in ACCEPTED_AXES and n == len(axes):
        return None
    from .ui import _step2_text as T

    return T.THUMB_FAIL_AXES


# ---------------------------------------------------------------------------
# Requests (main thread; no file opened)
# ---------------------------------------------------------------------------

def patch_thumb_requests(workspace, stack_dir, patch_names: Iterable[str]
                         ) -> list:
    """One ``ThumbRequest(name, path, axes="", mtime_ns, st_size)`` per TIFF
    patch that exists, in the order given (a repeated name once).

    * ``path`` = ``stack_dir / name``; a relative ``stack_dir`` is read
      under ``workspace``.
    * ``.npy`` (and any suffix but ``.tif`` / ``.tiff``, any case) and a
      missing file or a folder are skipped.
    * ``axes`` is left EMPTY: the worker fills it from the header probe.
    * ``mtime_ns`` / ``st_size`` from ONE ``os.stat`` each; no file opened.
    """
    from ..ui_shell._stack_thumbs import ThumbRequest

    if not stack_dir:
        return []
    root = Path(stack_dir)
    if not root.is_absolute() and workspace:
        root = Path(workspace) / root
    out: List = []
    seen = set()
    for name in patch_names or ():
        name = str(name)
        if name in seen:
            continue
        seen.add(name)
        if not name.lower().endswith(THUMB_SUFFIXES):
            continue
        path = root / name
        try:
            st = os.stat(path)
        except OSError:
            continue
        if not _stat.S_ISREG(st.st_mode):
            continue
        out.append(ThumbRequest(name, str(path), "", int(st.st_mtime_ns),
                                int(st.st_size)))
    return out


def png_for(cache_dir, req, *, single) -> Path:
    """Where the PNG of *req* is / will be (the shared name, ONE size)."""
    from ..ui_shell._stack_thumbs import thumb_png_path

    return thumb_png_path(cache_dir, req.name, req.path, req.mtime_ns,
                          req.st_size, single=single, size=PATCH_THUMB_PX)


# ---------------------------------------------------------------------------
# Start / stop one batch
# ---------------------------------------------------------------------------

class Step2ThumbState:
    """The running batch, if any (``thread`` / ``worker``).  The owner keeps
    ONE per list (patches here, slices in brick A): a new batch stops the
    older batch of the SAME state only, so one shared state would cancel
    the other list's batch."""

    __slots__ = ("thread", "worker")

    def __init__(self):
        self.thread = None
        self.worker = None

    def running(self) -> bool:
        th = self.thread
        if th is None:
            return False
        try:
            return bool(th.isRunning())
        except RuntimeError:            # the C++ object is gone
            return False


def _on_thumb_error(message: str) -> None:
    _diag(f"worker error: {message}")


def _make_worker_class():
    """The patch worker: the shared ``_StackThumbWorker`` plus a header
    probe per request and the extra ``meta`` signal.  Built lazily so this
    module imports without Qt."""
    from qtpy.QtCore import Signal, Slot

    from ..ui_shell import _stack_thumbs as T

    class _Step2PatchWorker(T._StackThumbWorker):
        #: name, patch path, ``PatchMeta.to_json()``
        meta = Signal(str, str, str)

        @Slot()
        def run(self) -> None:
            _diag(f"patch run start: {self.pending()} request(s)")
            done = 0
            try:
                T.sweep_stale_tmp(self._cache_dir)
                while not self._cancel.is_set():
                    req = self._pop()
                    if req is None:
                        break
                    info = probe_patch_header(req.path, req.name)
                    if self._cancel.is_set():
                        break
                    self.meta.emit(req.name, req.path, info.to_json())
                    if info.reason:
                        self.failed.emit(req.name, req.path, info.reason)
                        done += 1
                        continue
                    req = T.ThumbRequest(*req)._replace(axes=info.axes)
                    ok, payload, data = T.render_thumb(
                        req, self._cache_dir, self._size, single=self._single)
                    if self._cancel.is_set():
                        break
                    if ok:
                        self.ready.emit(req.name, req.path, payload, data)
                    else:
                        self.failed.emit(req.name, req.path, payload)
                    done += 1
            except Exception as exc:        # noqa: BLE001 -- a bug
                _diag(f"patch run error: {type(exc).__name__}: {exc}")
                self.error.emit(f"{type(exc).__name__}: {exc}")
            finally:
                with self._lock:
                    self._closed = True
                _diag(f"patch run end: {done} done, "
                      f"cancelled={self._cancel.is_set()}")
                self.finished.emit()

    return _Step2PatchWorker


_WORKER_CLASS = None


def patch_worker_class():
    global _WORKER_CLASS
    if _WORKER_CLASS is None:
        _WORKER_CLASS = _make_worker_class()
    return _WORKER_CLASS


def start_patch_thumbs(requests, cache_dir, on_ready, on_failed, on_meta, *,
                       state: Optional[Step2ThumbState] = None,
                       on_finished=None, on_error=None) -> Step2ThumbState:
    """Start ONE batch of patch pictures -> the state.

    ``state``: its older batch is stopped FIRST (cancelled, queue drained,
    slots disconnected, no wait on the main thread); then the new batch
    starts and the same state object is returned.  No requests: the older
    batch is still stopped, and nothing starts.

    ``on_ready(name, patch_path, png_path, png_bytes)`` /
    ``on_failed(name, patch_path, reason)`` / ``on_meta(name, patch_path,
    meta_json)`` arrive on the main thread; pass bound methods, never
    lambdas (a slot held only by PyQt can lose its closure when napari
    garbage-collects during layer removal)."""
    from ..ui_shell import _stack_thumbs as T

    if state is None:
        state = Step2ThumbState()
    elif state.thread is not None or state.worker is not None:
        stop_thumbs(state, wait_ms=0)
    requests = list(requests or ())
    if not requests:
        return state
    cls = patch_worker_class()
    thread = T._ThumbThread()
    worker = cls(requests, str(cache_dir), PATCH_THUMB_PX,
                 single=T.SINGLE_INFERNO)
    worker.moveToThread(thread)
    thread._thumb_worker = worker
    worker._on_meta = on_meta
    for attr, sig, fn in (("_on_ready", worker.ready, on_ready),
                          ("_on_failed", worker.failed, on_failed),
                          ("_on_meta", worker.meta, on_meta),
                          ("_on_finished", worker.finished, on_finished),
                          ("_on_error", worker.error,
                           on_error if on_error is not None
                           else _on_thumb_error)):
        setattr(worker, attr, fn)
        if fn is not None:
            sig.connect(fn)
    from qtpy.QtCore import Qt

    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit, Qt.DirectConnection)
    worker.error.connect(thread.quit, Qt.DirectConnection)
    thread.finished.connect(thread._forget)
    T._LIVE_THUMB_WORKERS.add(thread)
    state.thread, state.worker = thread, worker
    _diag(f"{len(requests)} patch picture(s) requested -> {cache_dir}")
    thread.start()
    return state


def stop_thumbs(state: Optional[Step2ThumbState],
                wait_ms: int = STOP_WAIT_MS) -> bool:
    """Stop the state's batch (``on_shutdown`` / a workspace close) -> True
    when no thread is left running.  Cancels, disconnects the caller's
    slots (the shared four and the ``meta`` one), quits and waits
    ``wait_ms`` (3 s by default); a thread still running then stays parked
    in the shared live set, never destroyed.  The state is cleared either
    way.  Idempotent."""
    if state is None:
        return True
    thread, worker = state.thread, state.worker
    state.thread = state.worker = None
    if thread is None and worker is None:
        return True
    from ..ui_shell._stack_thumbs import stop_thumb_worker

    if worker is not None:
        fn = getattr(worker, "_on_meta", None)
        if fn is not None:
            try:
                worker.meta.disconnect(fn)
            except (TypeError, RuntimeError, AttributeError):
                pass
    try:
        return bool(stop_thumb_worker(thread, worker, wait_ms=int(wait_ms)))
    except RuntimeError:
        return True


# ---------------------------------------------------------------------------
# Sweep of PNGs whose patch is no longer listed (PATCHES folder only)
# ---------------------------------------------------------------------------

#: a copy of ``fusion_reconstruction._step3_thumbs._PNG_NAME`` (this module
#: stays Qt-free and imports no Step 3 package; a test holds the two equal)
_PNG_TAIL = r"(?:\.s\d+)?(?:\.l\d+)?(?:\.o[a-z]{2}\d+)?"
_PNG_NAME = re.compile(r"^(?P<name>.+)\.[0-9a-f]{8}\.\d+\.\d+(?:\.c\d+i?)?"
                       + _PNG_TAIL + r"\.png$")


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


def sweep_unlisted_patches(workspace, listed_names: Iterable[str]) -> int:
    """Delete the PNGs in the PATCHES folder whose patch name is not in
    ``listed_names`` -> how many.  Only plain files directly in
    ``{ws}/artifacts/step2/thumbs/patches`` whose name has the worker's PNG
    shape are touched; the slices folder and the parent are never looked
    at.  Never raises."""
    if not workspace:
        return 0
    d = patch_thumbs_dir(workspace)
    try:
        if not d.is_dir():
            return 0
    except OSError:
        return 0
    if not _inside(d, Path(workspace)):
        _diag(f"sweep refused: {d} is outside the workspace {workspace}")
        return 0
    from ..ui_shell._stack_thumbs import safe_stack_name

    keep = {safe_stack_name(str(n)) for n in (listed_names or ())}
    n = 0
    try:
        entries = list(d.iterdir())
    except OSError as exc:
        _diag(f"sweep: cannot list {d}: {exc}")
        return 0
    for p in entries:
        m = _PNG_NAME.match(p.name)
        if m is None or m.group("name") in keep:
            continue
        try:
            if p.is_symlink() or not p.is_file():
                continue
            p.unlink()
            n += 1
        except OSError as exc:
            _diag(f"sweep: could not delete {p.name}: {exc}")
    if n:
        _diag(f"swept {n} picture(s) of patches no longer listed")
    return n


def remove_patch_pngs(workspace, patch_name: str) -> int:
    """Forget: delete the PATCHES-folder PNGs of ONE patch -> how many.
    *patch_name* is the patch's file name (``patch_4.tif``) or its stem
    (``patch_4``; both ``.tif`` and ``.tiff`` are tried).  Only plain files
    directly in ``{ws}/artifacts/step2/thumbs/patches`` with the worker's PNG
    shape and exactly that name are touched; the slices folder and the parent
    are never looked at.  Never raises."""
    if not workspace or not patch_name:
        return 0
    d = patch_thumbs_dir(workspace)
    try:
        if not d.is_dir():
            return 0
    except OSError:
        return 0
    if not _inside(d, Path(workspace)):
        _diag(f"remove refused: {d} is outside the workspace {workspace}")
        return 0
    from ..ui_shell._stack_thumbs import safe_stack_name

    name = str(patch_name)
    wants = {safe_stack_name(name)}
    if not name.lower().endswith((".tif", ".tiff")):
        wants |= {safe_stack_name(name + ".tif"),
                  safe_stack_name(name + ".tiff")}
    n = 0
    try:
        entries = list(d.iterdir())
    except OSError as exc:
        _diag(f"remove: cannot list {d}: {exc}")
        return 0
    for p in entries:
        m = _PNG_NAME.match(p.name)
        if m is None or m.group("name") not in wants:
            continue
        try:
            if p.is_symlink() or not p.is_file():
                continue
            p.unlink()
            n += 1
        except OSError as exc:
            _diag(f"remove: could not delete {p.name}: {exc}")
    return n


# ---------------------------------------------------------------------------
# Tooltip
# ---------------------------------------------------------------------------

def thumb_tooltip_for_failure(reason) -> str:
    """The row's tooltip when no picture could be drawn: the reason in
    Step 2's words (the shared worker's reasons speak of a "stack")."""
    from ..ui_shell import _stack_thumbs as T
    from .ui import _step2_text as W

    text = str(reason or "").strip()
    if text:
        for const, words in W.THUMB_REASONS.items():
            if text == getattr(T, const, None):
                text = words
                break
    return W.THUMB_FAILED.format(reason=text or W.THUMB_FAIL_UNKNOWN)


# ===========================================================================
# BRICK A -- the SLICE half
# ===========================================================================
#
# One 56 px PNG per slice (Slices list and Training's staged rows), in
# ``thumbs/slices/``, named by SAMPLE ID.  The picture is the pair's own
# image (``img_uri``) with the pair's own labels (``mask_uri``) drawn over it
# by Step 3's fill + outline overlay; the stack cut is only the fallback for
# a record with no readable image.
#
# ORIENTATION (the rule, matched to the annotator).  The pair writer stores
# an axial slice exactly as ``volume[:, idx, :]`` (XZ -> a ``(Z, X)`` plane)
# or ``volume[:, :, idx]`` (YZ -> ``(Z, Y)``): controller
# ``training_annotation_controller.py`` 7561-7565 (mask), 7604-7606 (image),
# 9657-9659.  The 2D annotator displays them the same way:
# ``apply_active_view_to_viewer`` (controller 3912-3934) sets
# ``viewer.dims.order = (1, 0, 2)`` for XZ and ``(2, 0, 1)`` for YZ, so the
# displayed plane is ``np.transpose(volume, order)[idx]`` -> ``(Z, X)`` /
# ``(Z, Y)``: Z on the rows, row 0 at the top (napari's 2D default).  So the
# thumbnail draws the stored plane AS IT IS: no transpose, no flip
# (:func:`orient_plane` is the one place a different rule would go).  The
# stack-cut fallback takes the same two slices of the volume.  NOTE: the 3D
# curation annotator's YZ SUBVIEWER is transposed (``orientation.transpose_data``
# ``(2, 1, 0)`` -> ``(Y, Z)``); that is Step 3's dock, not what Step 2's pair
# was drawn on (a wording problem).
#
# Letterbox (F12): an axial tile is a wide strip; ``encode_png`` scales with
# ``KeepAspectRatio`` and the delegate (``keep_aspect=True``) centres it in
# its 56 px square -- never cropped.

import io
import time
from collections import deque

#: the slice picture's edge (the narrow row's 56 px, Step 3's number)
SLICE_THUMB_PX = 56
#: the trailing wait before a re-curated slice of the loaded patch is
#: re-requested (D24): 1.5 s of silence after the last mask write
SLICE_DEBOUNCE_MS = 1500
#: a ``PermissionError`` on a pair file is retried ONCE after this long
RETRY_DELAY_S = 0.15
SLICE_VIEWS = ("XY", "XZ", "YZ")


_sleep = time.sleep         # looked up at call time (tests replace it)

_SID = re.compile(r"^(?P<patch>.+)_(?P<view>XY|XZ|YZ)_(?P<idx>\d+)$")


def parse_sample_id(sample_id: str):
    """``"{patch}_{VIEW}_{index}"`` -> ``(patch, VIEW, index)`` or ``None``."""
    m = _SID.match(str(sample_id or ""))
    if m is None:
        return None
    return m.group("patch"), m.group("view"), int(m.group("idx"))


class SliceThumbRequest(NamedTuple):
    """One slice picture to draw.  The first seven fields have
    ``ThumbRequest``'s meaning so the shared name / sweep helpers work on it:
    ``name`` = the sample id, ``path`` = the KEY path (see
    :func:`slice_thumb_request`), ``mtime_ns`` / ``st_size`` = the image's
    (or the stack's), ``labels_path`` / ``labels_mtime_ns`` = the pair's mask
    and its mtime (``""`` / 0: no mask, the raw plane)."""
    name: str
    path: str
    axes: str
    mtime_ns: int
    st_size: int
    labels_path: str = ""
    labels_mtime_ns: int = 0
    view: str = "XY"
    index: int = 0
    status: str = ""
    img_path: str = ""          # the pair's image, "" -> the stack fallback
    stack_path: str = ""        # the patch TIFF (fallback only)


class SliceMeta(NamedTuple):
    """The worker's metadata payload (rides the ``meta`` signal).  ``busy``:
    a pair file was locked twice; nothing was drawn, the old PNG stays."""
    name: str
    n_labels: int       # distinct ids > 0; -1 unknown; 0 no labels / no mask
    n_channels: int
    shape: tuple        # the plane's (rows, cols)
    has_mask: bool
    source: str         # "pair" | "stack"
    busy: bool = False

    def to_json(self) -> str:
        return json.dumps({"name": self.name, "n_labels": self.n_labels,
                           "n_channels": self.n_channels,
                           "shape": list(self.shape),
                           "has_mask": self.has_mask, "source": self.source,
                           "busy": self.busy})

    @staticmethod
    def from_json(text: str) -> "SliceMeta":
        d = json.loads(text)
        return SliceMeta(str(d.get("name", "")), int(d.get("n_labels", -1)),
                         int(d.get("n_channels", 1)),
                         tuple(int(v) for v in d.get("shape", ())),
                         bool(d.get("has_mask", False)),
                         str(d.get("source", "pair")),
                         bool(d.get("busy", False)))


class PairBusy(Exception):
    """A pair file stayed locked after the one retry (skip, keep the old
    PNG)."""


# -- requests (main thread: ``os.stat`` only) ---------------------------------

def _abs(workspace, p) -> str:
    p = str(p or "")
    if not p:
        return ""
    if not os.path.isabs(p) and workspace:
        return str(Path(workspace) / p)
    return p


def _stat_file(path):
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st if _stat.S_ISREG(st.st_mode) else None


def slice_thumb_request(sample_id: str, record, *, stack_paths=None,
                        workspace=None) -> Optional[SliceThumbRequest]:
    """The request for one record, or ``None`` when nothing can be drawn
    (no image file and no stack to cut).  Only ``os.stat`` -- no file is
    opened here.

    * image = the record's ``img_uri`` when that file exists; else the
      patch stack in ``stack_paths[patch]`` cut at (view, index);
    * labels = the record's ``mask_uri`` when that file exists, else none;
    * the KEY path is the image path; when the mask lives in another folder
      than the image, or the picture is a stack cut, the mask / the cut is
      part of it, so a different mask path or a different cut is a different
      PNG."""
    rec = record if isinstance(record, dict) else dict(record or {})
    parsed = parse_sample_id(sample_id)
    view = str(rec.get("view") or (parsed[1] if parsed else "")).upper()
    if view not in SLICE_VIEWS:
        return None
    try:
        index = int(rec.get("slice_index", parsed[2] if parsed else -1))
    except (TypeError, ValueError):
        return None
    if index < 0:
        return None
    status = str(rec.get("status") or "")
    img = _abs(workspace, rec.get("img_uri"))
    mask = _abs(workspace, rec.get("mask_uri"))
    mst = _stat_file(mask) if mask else None
    if mst is None:
        mask = ""
    ist = _stat_file(img) if img else None
    if ist is not None:
        key = img
        if mask and os.path.dirname(os.path.abspath(mask)) != \
                os.path.dirname(os.path.abspath(img)):
            key = f"{img}|{mask}"
        return SliceThumbRequest(
            str(sample_id), key, "", int(ist.st_mtime_ns), int(ist.st_size),
            mask, int(mst.st_mtime_ns) if mst else 0, view, index, status,
            img, "")
    patch = str(rec.get("patch_id") or rec.get("roi_id") or "").strip()
    if not patch or patch == "-":
        patch = parsed[0] if parsed else ""
    stack = _abs(workspace, (stack_paths or {}).get(patch, ""))
    sst = _stat_file(stack) if stack else None
    if sst is None:
        return None
    key = f"{stack}::{view}:{index}" + (f"|{mask}" if mask else "")
    return SliceThumbRequest(
        str(sample_id), key, "", int(sst.st_mtime_ns), int(sst.st_size),
        mask, int(mst.st_mtime_ns) if mst else 0, view, index, status,
        "", stack)


def slice_png_path(cache_dir, req, *, with_labels: Optional[bool] = None
                   ) -> Path:
    """Where the PNG of *req* is / will be: the shared name (sample id, key
    path hash, image mtime + size, ``.s56``, ``.l<mask mtime>.<overlay>``
    when a mask is drawn).  ``with_labels=False`` names the raw picture."""
    from ..ui_shell._stack_thumbs import SINGLE_INFERNO, thumb_png_path

    use = bool(req.labels_path) if with_labels is None else bool(with_labels)
    lab = int(req.labels_mtime_ns or 0) if use else 0
    return thumb_png_path(cache_dir, req.name, req.path, req.mtime_ns,
                          req.st_size, single=SINGLE_INFERNO,
                          labels_mtime_ns=lab, size=SLICE_THUMB_PX)


def cache_key_for(req) -> tuple:
    """The main-thread ``ThumbCache`` key arguments of *req*:
    ``(key path, mtime, size, mask mtime)`` as ``ThumbCache.get`` takes."""
    return (req.path, int(req.mtime_ns), int(req.st_size),
            int(req.labels_mtime_ns or 0) if req.labels_path else 0)


# -- reading (worker thread) ---------------------------------------------------

def fit_stride(arr, size: int):
    """Stride-downsample the LAST TWO axes by ONE factor so the longest edge
    is about *size* (``thumb_stride`` strides each axis by its own factor,
    which squashes a wide axial strip: 124 x 240 would become ~62 x 60, a
    square).  The shared ``compose_rgb`` / ``overlay_labels`` then find
    nothing left to stride."""
    h, w = arr.shape[-2:]
    s = max(1, max(int(h), int(w)) // max(int(size), 1))
    return arr[..., ::s, ::s] if s > 1 else arr


def orient_plane(plane, view: str):
    """The plane as the annotator shows it.  The stored pair planes already
    are (see the section comment), so this returns *plane* unchanged; a
    different rule for a view would live here and nowhere else."""
    return plane


def read_file_bytes(path) -> bytes:
    """ONE short ``read_bytes()`` -> the whole file in memory; no handle is
    held afterwards (on Windows an open read handle would block the
    controller's ``os.replace`` / Discard's ``unlink``).  A
    ``PermissionError`` waits :data:`RETRY_DELAY_S` and tries ONCE more, then
    raises :class:`PairBusy`.  Any other ``OSError`` propagates."""
    for attempt in (0, 1):
        try:
            return Path(path).read_bytes()
        except PermissionError:
            if attempt == 0:
                _sleep(RETRY_DELAY_S)
                continue
            raise PairBusy(str(path))
    raise PairBusy(str(path))       # pragma: no cover


def read_pair_array(path):
    """A pair TIFF -> ndarray, decoded from bytes (never from the path)."""
    import tifffile

    return tifffile.imread(io.BytesIO(read_file_bytes(path)))


def _mask_plane(arr):
    import numpy as np

    a = np.squeeze(np.asarray(arr))
    if a.ndim != 2:
        raise ValueError(f"mask is not a 2D plane: {a.shape}")
    return a


def count_labels(mask) -> int:
    """Distinct label ids above 0 in a mask plane."""
    import numpy as np

    m = np.asarray(mask)
    return int(np.unique(m[m != 0]).size)


def read_stack_cut(path, view: str, index: int):
    """The patch TIFF cut at (view, index) -> ``(n, H, W)`` (<= 3 channels).
    The same slices the pair writer takes: XY ``vol[idx]``, XZ
    ``vol[:, idx, :]``, YZ ``vol[:, :, idx]``.  Raises ``ValueError`` for
    axes other than ``ZYX`` / ``CZYX`` / ``ZCYX`` and ``IndexError`` for an
    index outside the stack."""
    import numpy as np
    import tifffile
    import zarr

    with tifffile.TiffFile(str(path)) as tf:
        series = tf.series[0]
        axes = str(series.axes).upper()
        shape = tuple(int(v) for v in series.shape)
        reason = axes_reason(axes, shape)
        if reason:
            raise ValueError(reason)
        dims = dict(zip(axes, shape))
        cut_axis = {"XY": "Z", "XZ": "Y", "YZ": "X"}[view]
        if not 0 <= int(index) < dims[cut_axis]:
            raise IndexError(
                f"{view} index {index} outside 0..{dims[cut_axis] - 1}")
        n = min(dims.get("C", 1), 3)
        store = series.aszarr()
        try:
            z = zarr.open(store, mode="r")
            if not hasattr(z, "shape"):
                z = z[sorted(z.array_keys())[0]]
            planes = []
            for c in range(n):
                sel = []
                for a in axes:
                    if a == "C":
                        sel.append(c)
                    elif a == cut_axis:
                        sel.append(int(index))
                    else:
                        sel.append(slice(None))
                planes.append(np.asarray(z[tuple(sel)]))
        finally:
            try:
                store.close()
            except Exception:       # noqa: BLE001
                pass
    return np.stack(planes)


def _sweep_same_sample(cache_dir, name: str, dest) -> None:
    """After a PNG is written: delete this SAMPLE's other PNGs in the slices
    folder (any key path, mtime or size).  A sample id names one slice, so a
    pair that moved folder (accepted -> staged -> a round) leaves nothing
    behind."""
    from ..ui_shell._stack_thumbs import safe_stack_name

    keep = Path(dest).name
    want = safe_stack_name(name)
    try:
        entries = list(Path(cache_dir).iterdir())
    except OSError:
        return
    for p in entries:
        m = _PNG_NAME.match(p.name)
        if m is None or m.group("name") != want or p.name == keep:
            continue
        try:
            p.unlink()
        except OSError as exc:
            _diag(f"could not delete stale {p.name}: {exc}")


def render_slice_thumb(req, cache_dir, size: int = SLICE_THUMB_PX):
    """One request -> ``(state, payload, data, meta)``; never raises.

    * ``("ok", png_path, png_bytes, SliceMeta)`` -- ``png_path`` is ``""``
      when it could not be written (drawn from memory);
    * ``("failed", reason, b"", None)`` -- a blank tile with a hover reason;
    * ``("busy", "", b"", SliceMeta(busy=True))`` -- a pair file stayed
      locked: nothing drawn, nothing written, the old PNG is kept.

    Each pair file is read with ONE ``read_bytes()`` (:func:`read_file_bytes`)
    and decoded from memory.  A PNG already on disk for the key is returned
    without reading the image; the mask is read for the label count only
    (a busy mask then gives ``n_labels = -1``, the picture still stands).
    A mask that cannot be read (deleted by Discard, not a plane) gives the
    raw plane, named WITHOUT the labels part so the next run tries again --
    never a wrong picture."""
    import numpy as np

    from ..ui_shell import _stack_thumbs as S

    req = SliceThumbRequest(*req)
    source = "pair" if req.img_path else "stack"
    dest = slice_png_path(cache_dir, req)
    mask = None
    mask_busy = False
    mask_state = "none"                 # none | ok | gone
    if req.labels_path:
        try:
            mask = _mask_plane(read_pair_array(req.labels_path))
            mask_state = "ok"
        except PairBusy:
            mask_busy = True
        except Exception as exc:        # noqa: BLE001 -- Discard raced us
            _diag(f"{req.name!r}: mask unreadable "
                  f"({type(exc).__name__}: {exc}); raw picture")
            mask_state = "gone"
    n_labels = (count_labels(mask) if mask is not None
                else (-1 if mask_busy else 0))
    has_mask = mask_state == "ok" or mask_busy

    if dest.is_file() and mask_state != "gone":
        try:
            data = dest.read_bytes()
        except OSError:
            data = b""
        if data:
            # the row's "240 x 240 . 2 ch" needs the plane's shape even when
            # the picture is cached: one short read of the (small) image file,
            # here on the worker; a busy / unreadable file leaves it blank
            n_ch, shape = 1, ()
            if req.img_path:
                try:
                    a = np.asarray(read_pair_array(req.img_path))
                    if a.ndim == 2:
                        a = a[None]
                    if a.ndim == 3:
                        n_ch = int(a.shape[0])
                        shape = tuple(int(v) for v in a.shape[-2:])
                except Exception:       # noqa: BLE001 -- the picture stands
                    pass
            return "ok", str(dest), data, SliceMeta(
                req.name, n_labels, n_ch, shape, has_mask, source)
    if mask_busy:
        return "busy", "", b"", SliceMeta(req.name, -1, 1, (), True, source,
                                          True)
    try:
        if req.img_path:
            planes = np.asarray(read_pair_array(req.img_path))
            if planes.ndim == 2:
                planes = planes[None]
            if planes.ndim != 3:
                raise ValueError(f"image is not (C, Y, X): {planes.shape}")
        else:
            planes = read_stack_cut(req.stack_path, req.view, req.index)
    except PairBusy:
        return "busy", "", b"", SliceMeta(req.name, n_labels, 1, (),
                                          has_mask, source, True)
    except FileNotFoundError:
        return "failed", S.THUMB_FAIL_MISSING, b"", None
    except ValueError as exc:
        _diag(f"failed {req.name!r}: {exc}")
        reason = (S.THUMB_FAIL_AXES if "axes" in str(exc).lower()
                  else S.THUMB_FAIL_UNREADABLE)
        return "failed", reason, b"", None
    except Exception as exc:            # noqa: BLE001 -- a bad file
        _diag(f"failed {req.name!r}: {type(exc).__name__}: {exc}")
        return "failed", S.THUMB_FAIL_UNREADABLE, b"", None
    try:
        planes = np.stack([orient_plane(p, req.view) for p in planes])
        full_shape = tuple(int(v) for v in planes.shape[-2:])
        rgb = S.compose_rgb(fit_stride(planes, size), size,
                            single=S.SINGLE_INFERNO)
        drawn_mask = mask_state == "ok"
        if drawn_mask:
            rgb = S.overlay_labels(
                rgb, fit_stride(orient_plane(mask, req.view), size),
                size=size)
        data = S.encode_png(rgb, size)
    except Exception as exc:            # noqa: BLE001
        _diag(f"failed {req.name!r}: encode: {type(exc).__name__}: {exc}")
        return "failed", S.THUMB_FAIL_ENCODE, b"", None
    dest = slice_png_path(cache_dir, req, with_labels=drawn_mask)
    meta = SliceMeta(req.name, n_labels, int(planes.shape[0]), full_shape,
                     has_mask, source)
    try:
        S._write_atomic(dest, data)
    except Exception as exc:            # noqa: BLE001
        _diag(f"could not write {dest.name} ({type(exc).__name__}: {exc}); "
              "drawing from memory")
        return "ok", "", data, meta
    _sweep_same_sample(cache_dir, req.name, dest)
    return "ok", str(dest), data, meta


# -- the worker -----------------------------------------------------------------

def _make_slice_worker_class():
    """The slice worker: the shared ``_StackThumbWorker``'s queue / cancel /
    close rules, ``SliceThumbRequest`` items, ``render_slice_thumb``, and the
    same extra ``meta`` signal as the patch worker -- one signal shape for
    both halves (``ready`` name / key path / png path / bytes; ``failed``
    name / key path / reason; ``meta`` name / key path / json).  Built
    lazily so the module imports without Qt."""
    from qtpy.QtCore import Signal, Slot

    from ..ui_shell import _stack_thumbs as T

    class _Step2SliceWorker(T._StackThumbWorker):
        meta = Signal(str, str, str)

        def __init__(self, requests, cache_dir, size: int = SLICE_THUMB_PX,
                     parent=None, single=None):
            super().__init__([], cache_dir, size, parent, single=single)
            self._queue = deque(SliceThumbRequest(*r) for r in requests)

        def enqueue(self, req) -> bool:
            with self._lock:
                if self._closed:
                    return False
                self._queue.append(SliceThumbRequest(*req))
                return True

        @Slot()
        def run(self) -> None:
            _diag(f"slice run start: {self.pending()} request(s)")
            done = 0
            try:
                T.sweep_stale_tmp(self._cache_dir)
                while not self._cancel.is_set():
                    req = self._pop()
                    if req is None:
                        break
                    state, payload, data, meta = render_slice_thumb(
                        req, self._cache_dir, self._size)
                    if self._cancel.is_set():
                        break
                    if meta is not None:
                        self.meta.emit(req.name, req.path, meta.to_json())
                    if state == "ok":
                        self.ready.emit(req.name, req.path, payload, data)
                    elif state == "failed":
                        self.failed.emit(req.name, req.path, payload)
                    done += 1
            except Exception as exc:        # noqa: BLE001 -- a bug
                _diag(f"slice run error: {type(exc).__name__}: {exc}")
                self.error.emit(f"{type(exc).__name__}: {exc}")
            finally:
                with self._lock:
                    self._closed = True
                _diag(f"slice run end: {done} done, "
                      f"cancelled={self._cancel.is_set()}")
                self.finished.emit()

    return _Step2SliceWorker


_SLICE_WORKER_CLASS = None


def slice_worker_class():
    global _SLICE_WORKER_CLASS
    if _SLICE_WORKER_CLASS is None:
        _SLICE_WORKER_CLASS = _make_slice_worker_class()
    return _SLICE_WORKER_CLASS


def start_slice_thumbs(requests, cache_dir, on_ready, on_failed, on_meta, *,
                       state: Optional[Step2ThumbState] = None,
                       on_finished=None, on_error=None) -> Step2ThumbState:
    """Start ONE batch of slice pictures -> the state.  Same contract as
    :func:`start_patch_thumbs` (the older batch of the SAME state is stopped
    first; bound methods only).  Use :func:`submit_slice_thumbs` to add to a
    batch that is still running."""
    from qtpy.QtCore import Qt

    from ..ui_shell import _stack_thumbs as T

    if state is None:
        state = Step2ThumbState()
    elif state.thread is not None or state.worker is not None:
        stop_thumbs(state, wait_ms=0)
    requests = list(requests or ())
    if not requests:
        return state
    thread = T._ThumbThread()
    worker = slice_worker_class()(requests, str(cache_dir), SLICE_THUMB_PX,
                                  single=T.SINGLE_INFERNO)
    worker.moveToThread(thread)
    thread._thumb_worker = worker
    for attr, sig, fn in (("_on_ready", worker.ready, on_ready),
                          ("_on_failed", worker.failed, on_failed),
                          ("_on_meta", worker.meta, on_meta),
                          ("_on_finished", worker.finished, on_finished),
                          ("_on_error", worker.error,
                           on_error if on_error is not None
                           else _on_thumb_error)):
        setattr(worker, attr, fn)
        if fn is not None:
            sig.connect(fn)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit, Qt.DirectConnection)
    worker.error.connect(thread.quit, Qt.DirectConnection)
    thread.finished.connect(thread._forget)
    T._LIVE_THUMB_WORKERS.add(thread)
    state.thread, state.worker = thread, worker
    _diag(f"{len(requests)} slice picture(s) requested -> {cache_dir}")
    thread.start()
    return state


def submit_slice_thumbs(requests, cache_dir, on_ready, on_failed, on_meta, *,
                        state: Step2ThumbState, on_finished=None,
                        on_error=None) -> int:
    """Give *requests* to the state's running batch (``enqueue``; a refusal
    is never a silent drop: the refused ones start a NEW batch) or start one
    -> how many were submitted.  A burst of refreshes therefore never
    cancels a batch in flight."""
    requests = list(requests or ())
    if not requests:
        return 0
    worker = state.worker if state.running() else None
    rest = requests
    if worker is not None:
        rest = []
        for r in requests:
            try:
                ok = bool(worker.enqueue(r))
            except RuntimeError:
                ok = False
            if not ok:
                rest.append(r)
    if rest:
        if worker is not None:
            # the running one closed under us: it finishes by itself (its
            # queue is empty); the new batch must not cancel it
            state.thread = state.worker = None
        start_slice_thumbs(rest, cache_dir, on_ready, on_failed, on_meta,
                           state=state, on_finished=on_finished,
                           on_error=on_error)
    return len(requests)


# -- the in-flight set and the debounce -------------------------------------------

class InFlight:
    """PNG names whose request is queued or running; a request whose PNG name
    is in here is dropped (D24).  Main thread only."""

    def __init__(self):
        self._names: set = set()

    def add(self, png_name: str) -> bool:
        """True when newly added; False = already in flight (drop it)."""
        if png_name in self._names:
            return False
        self._names.add(png_name)
        return True

    def discard(self, png_name: str) -> None:
        self._names.discard(png_name)

    def clear(self) -> None:
        self._names.clear()

    def __contains__(self, png_name) -> bool:
        return png_name in self._names

    def __len__(self) -> int:
        return len(self._names)


class TrailingDebounce:
    """Collects keys and calls ``callback(set_of_keys)`` ONCE, *delay_ms*
    after the LAST ``poke`` -- a burst of mask writes is one request.
    ``callback`` is a bound method.  ``timer_factory`` builds the one-shot
    timer (``setInterval`` / ``start`` / ``stop`` / ``timeout.connect``);
    the default is a main-thread ``QTimer``."""

    def __init__(self, callback, delay_ms: int = SLICE_DEBOUNCE_MS, *,
                 timer_factory=None):
        self._callback = callback
        self._delay = int(delay_ms)
        self._factory = timer_factory
        self._timer = None
        self._keys: set = set()

    def _ensure_timer(self):
        if self._timer is None:
            if self._factory is not None:
                t = self._factory()
            else:
                from qtpy.QtCore import QTimer

                t = QTimer()
                t.setSingleShot(True)
            t.setInterval(self._delay)
            t.timeout.connect(self._fire)
            self._timer = t
        return self._timer

    def poke(self, key=None) -> None:
        if key is not None:
            self._keys.add(key)
        t = self._ensure_timer()
        t.stop()
        t.start()

    @property
    def pending(self) -> set:
        return set(self._keys)

    def cancel(self) -> None:
        self._keys.clear()
        if self._timer is not None:
            self._timer.stop()

    def _fire(self) -> None:
        keys, self._keys = self._keys, set()
        if keys:
            self._callback(keys)


# -- the slices sweep set (D25) ----------------------------------------------------

def _slice_png_entries(workspace):
    d = slice_thumbs_dir(workspace)
    try:
        if not d.is_dir() or not _inside(d, Path(workspace)):
            return []
        return [p for p in d.iterdir()
                if _PNG_NAME.match(p.name) and p.is_file()
                and not p.is_symlink()]
    except OSError:
        return []


def sweep_unlisted_slices(workspace, sample_ids: Iterable[str], *,
                          newer_than: Optional[float] = None,
                          in_flight=None) -> int:
    """At workspace open: delete the SLICES folder's PNGs whose sample id is
    not in *sample_ids* (the union of ALL runs' records) -> how many.  Only
    the slices folder is looked at: the patches folder and the parent are
    never touched.  Never raises.

    The keep set is a snapshot, so a picture rendered AFTER it was taken
    (a Predict right after open) must survive (audit L3): a PNG modified at
    or after *newer_than* (epoch seconds) is kept, and so is one whose file
    name is in *in_flight* (a container, tested with ``in``)."""
    if not workspace:
        return 0
    from ..ui_shell._stack_thumbs import safe_stack_name

    keep = {safe_stack_name(str(s)) for s in (sample_ids or ())}
    n = 0
    for p in _slice_png_entries(workspace):
        if _PNG_NAME.match(p.name).group("name") in keep:
            continue
        if in_flight is not None and p.name in in_flight:
            continue
        if newer_than is not None:
            try:
                if p.stat().st_mtime >= newer_than:
                    continue
            except OSError:
                continue
        try:
            p.unlink()
            n += 1
        except OSError as exc:
            _diag(f"slice sweep: could not delete {p.name}: {exc}")
    if n:
        _diag(f"swept {n} slice picture(s) of records that no longer exist")
    return n


def remove_patch_slice_pngs(workspace, patch_stem: str) -> int:
    """Forget: delete the slices-folder PNGs of the samples of *patch_stem*
    (``{patch}_{VIEW}_{n}``, parsed -- ``a`` never matches ``a_b``'s
    samples) -> how many."""
    if not workspace or not patch_stem:
        return 0
    from ..ui_shell._stack_thumbs import safe_stack_name

    want = safe_stack_name(str(patch_stem))
    n = 0
    for p in _slice_png_entries(workspace):
        name = _PNG_NAME.match(p.name).group("name")
        m = _SID.match(name)
        if m is None or m.group("patch") != want:
            continue
        try:
            p.unlink()
            n += 1
        except OSError:
            pass
    return n


def all_runs_sample_ids(workspace, extra: Iterable[str] = ()) -> Optional[set]:
    """The union of the sample ids of EVERY run's records (the v3 state file's
    ``runs.<slug>.sample_records``) plus *extra* (the open run's, which may be
    newer than the file) -> a set, or ``None`` when the file cannot be read or
    is not the v3 shape: the caller then sweeps NOTHING (an unreadable state
    must never delete a picture).  One file read -- call it off the main
    thread."""
    if not workspace:
        return None
    try:
        path = _paths.state_json_path(workspace)
        if not path.is_file():
            return set(str(x) for x in extra) or None
        data = json.loads(path.read_text(encoding="utf-8"))
        runs = data.get("runs") if isinstance(data, dict) else None
        if not isinstance(runs, dict):
            return None
        ids = set(str(x) for x in extra)
        for run in runs.values():
            recs = run.get("sample_records") if isinstance(run, dict) else None
            if isinstance(recs, dict):
                ids.update(str(k) for k in recs)
        return ids
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        _diag(f"all_runs_sample_ids: {exc}")
        return None


def sweep_slices_async(workspace, extra: Iterable[str] = (), *,
                       in_flight=None):
    """Workspace open: delete the slices folder's PNGs of samples no run has
    any more, on a short daemon thread (one JSON read + a folder listing) ->
    the started ``Thread`` (joinable in tests), or ``None`` when there is
    nothing to do.  Never raises."""
    import threading

    if not workspace:
        return None
    extra = list(extra)
    # anything written from now on is newer than the snapshot (two seconds of
    # slack for coarse file-time granularity)
    started = time.time() - 2.0

    def _run():
        try:
            if not slice_thumbs_dir(workspace).is_dir():
                return
            ids = all_runs_sample_ids(workspace, extra)
            if ids is None:
                return
            sweep_unlisted_slices(workspace, ids, newer_than=started,
                                  in_flight=in_flight)
        except Exception as exc:      # noqa: BLE001 -- a sweep must never raise
            _diag(f"slice sweep failed: {exc}")

    t = threading.Thread(target=_run, name="epicycle-step2-slice-sweep",
                         daemon=True)
    t.start()
    return t
