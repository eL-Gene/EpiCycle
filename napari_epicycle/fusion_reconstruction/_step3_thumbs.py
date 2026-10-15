"""Step 3's patch thumbnails -- the engine, no UI (brick V2a).



A picture of each patch's RAW TIFF -- the mid-Z slice of up to three
channels -- for Session's rows.  The pictures are made by the SHARED worker
``ui_shell/_stack_thumbs.py`` (``_StackThumbWorker`` on its own
``_ThumbThread``: ``read_midz_planes`` -> ``compose_rgb`` -> ``encode_png``),
the same one Step 4 uses; this module only builds Step 3's requests, names
Step 3's cache folder, starts / stops one batch at a time and sweeps PNGs of
patches that are no longer listed.

The owner's rule (2026-10-07, changed 2026-10-10): a patch with ONE channel
-- a 3-D ``ZYX`` file, or a 4-D ``ZCYX`` / ``CZYX`` file with C = 1 -- draws
in INFERNO (it drew grey until A1); a reconstructed patch also carries the
run's label outlines (``labels_for``); two channels draw magenta (channel 0) + green (channel 1), a third
adds blue (``compose_rgb``'s rule, ``THUMB_COLOURS``).  The axes of every
request come from Step 3's PER-PATCH resolver (the controller passes
``state_manager.get_patch_axes``, backed by
``qc_features.resolve_patch_axes``), never one global value, so a folder
that mixes grayscale and two-channel patches draws row by row.

Threads: the request builder runs on the main thread and only ``os.stat``\\ s
each file -- it opens none.  Every file read is on the worker.  A failure
(missing, unreadable, axes that do not fit) arrives on ``on_failed`` as a
sentence; :func:`thumb_tooltip_for_failure` turns it into the row's tooltip.

Qt-free at import: the shared worker (``qtpy``) is imported inside the
functions that need it.
"""

from __future__ import annotations

import os
import re
import stat as _stat
from pathlib import Path
from typing import Callable, Iterable, List, Optional

from . import _paths

#: Patch suffixes that get a picture (``.npy`` patches are skipped).
THUMB_SUFFIXES = (".tif", ".tiff")
#: Step 3's folder under ``artifacts/step3/``.
THUMBS_SUBDIR = "thumbs"
#: How long a step exit / shutdown waits for a running batch (Step 4's rule).
STOP_WAIT_MS = 3000
#: A one-channel patch draws in inferno, not grey (owner 2026-10-10, A1);
#: Step 4 keeps grey.
THUMB_SINGLE = "inferno"

# The words live in ``_step3_pr_text`` (brick V2); re-exported here so the
# engine's callers and tests read them where they always did.
from ._step3_pr_text import (  # noqa: E402,F401
    S3_THUMB_FAIL_UNKNOWN,
    S3_THUMB_FAILED,
    S3_THUMB_PENDING,
    S3_THUMB_REASONS as _STEP3_REASONS,
    S3_THUMB_TIP,
)


def _diag(msg: str) -> None:
    try:
        print(f"[STEP3-THUMB] {msg}", flush=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Cache folder and requests (main thread; no file opened)
# ---------------------------------------------------------------------------

def thumb_cache_dir(workspace, run_path_id=None) -> Path:
    """``{ws}/artifacts/step3/thumbs`` -- WORKSPACE-level.

    A thumbnail is a picture of the RAW patch, which is the same for every
    run, so ``run_path_id`` is accepted (the caller's call reads like every
    other per-run path) and ignored: one PNG per patch serves all runs.
    Naming the folder does not create it; the worker does on its first
    write."""
    del run_path_id
    return _paths.step3_root(workspace) / THUMBS_SUBDIR


def patch_thumb_requests(workspace, stack_dir, patch_names: Iterable[str],
                         axes_for: Callable[[str], str],
                         labels_for: Optional[Callable[[str], Optional[str]]]
                         = None) -> list:
    """One ``ThumbRequest(name, path, axes, mtime_ns, st_size[, labels_path,
    labels_mtime_ns])`` per TIFF
    patch that exists, in the order given (a repeated name once).

    * ``path`` = ``stack_dir / name``; a relative ``stack_dir`` is read under
      ``workspace``.
    * ``.npy`` patches (and any suffix but ``.tif`` / ``.tiff``, any case)
      are skipped before their axes are asked; a missing file or a folder is
      skipped.
    * ``axes`` = ``axes_for(name)`` upper-cased -- Step 3's per-patch
      resolver.  An empty answer or an exception skips the patch (a console
      line), never a guess.
    * ``mtime_ns`` / ``st_size`` from ONE ``os.stat`` each; no file is
      opened (the shared worker reads it).
    * ``labels_for(name)`` (A1): the run's labels file of a patch that is
      reconstructed in THIS run, else ``None``.  A path that exists gets one
      ``os.stat`` for its mtime and rides the request (the worker draws its
      mid-Z outlines); a missing file, ``None`` or an exception means no
      outline, never a skipped patch.
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
        try:
            axes = str(axes_for(name) or "").strip().upper()
        except Exception as exc:
            _diag(f"{name}: no axes ({type(exc).__name__}: {exc}); skipped")
            continue
        if not axes:
            _diag(f"{name}: no axes; skipped")
            continue
        lab_path, lab_ns = "", 0
        if labels_for is not None:
            try:
                cand = labels_for(name)
                if cand:
                    lst = os.stat(cand)
                    if _stat.S_ISREG(lst.st_mode) or _stat.S_ISDIR(lst.st_mode):
                        lab_path, lab_ns = str(cand), int(lst.st_mtime_ns)
            except Exception as exc:     # noqa: BLE001 -- decoration only
                _diag(f"{name}: no labels outline ({type(exc).__name__}); raw")
        out.append(ThumbRequest(name, str(path), axes, int(st.st_mtime_ns),
                                int(st.st_size), lab_path, lab_ns))
    return out


# ---------------------------------------------------------------------------
# Start / stop one batch
# ---------------------------------------------------------------------------

class Step3ThumbState:
    """The running batch, if any: ``thread`` / ``worker`` of the shared
    worker.  The controller keeps ONE of these and hands it back to
    :func:`start_step3_thumbs` (which stops the older batch first) and to
    :func:`stop_step3_thumbs` (step exit, ``shutdown``; idempotent)."""

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
    """A bug in the worker (not a bad patch): a console line only."""
    _diag(f"worker error: {message}")


def start_step3_thumbs(parent_qobject, requests, cache_dir,
                       on_ready, on_failed, *,
                       state: Optional[Step3ThumbState] = None,
                       on_finished=None, on_error=None) -> Step3ThumbState:
    """Start ONE batch of pictures on the shared worker -> the state.

    ``state`` (the controller's one :class:`Step3ThumbState`): its older
    batch is stopped FIRST -- cancelled, its queue drained and its slots
    disconnected, so none of its pictures reaches the caller -- without
    waiting on the main thread (a thread still finishing the picture in
    flight stays parked in the shared live set until it ends).  Then the new
    batch starts and the same state object is returned; with ``state=None``
    a new one is made.  No requests: the older batch is still stopped, and
    nothing starts.

    ``on_ready(name, patch_path, png_path, png_bytes)`` /
    ``on_failed(name, patch_path, reason)`` arrive on the main thread; pass
    bound methods, never lambdas (a slot held only by PyQt can lose its closure
    when napari garbage-collects during a layer removal).
    ``parent_qobject`` is accepted for the caller's symmetry and ignored --
    the shared worker's thread is never parented (a deleted parent would
    destroy a running thread)."""
    from ..ui_shell._stack_thumbs import start_thumb_worker

    del parent_qobject
    if state is None:
        state = Step3ThumbState()
    elif state.thread is not None or state.worker is not None:
        stopped = stop_step3_thumbs(state, wait_ms=0)
        if not stopped:
            _diag("older batch stopped; its thread finishes the picture in "
                  "flight, parked")
    requests = list(requests or ())
    if not requests:
        return state
    thread, worker = start_thumb_worker(
        None, requests, str(cache_dir), on_ready=on_ready,
        on_failed=on_failed, on_finished=on_finished,
        on_error=on_error if on_error is not None else _on_thumb_error,
        single=THUMB_SINGLE)
    state.thread, state.worker = thread, worker
    _diag(f"{len(requests)} picture(s) requested -> {cache_dir}")
    return state


def stop_step3_thumbs(state: Optional[Step3ThumbState],
                      wait_ms: int = STOP_WAIT_MS) -> bool:
    """Stop the state's batch (``on_step_exit`` / ``shutdown``) -> True when
    no thread is left running.

    Cancels, disconnects the caller's slots, quits and waits ``wait_ms``
    (3 s by default, Step 4's rule); a thread still running then stays
    parked in the shared live set, never destroyed.  The state is cleared
    either way.  Idempotent: ``None``, an empty state, or a second call is
    True at once."""
    if state is None:
        return True
    thread, worker = state.thread, state.worker
    state.thread = state.worker = None
    if thread is None and worker is None:
        return True
    from ..ui_shell._stack_thumbs import stop_thumb_worker

    try:
        return bool(stop_thumb_worker(thread, worker, wait_ms=int(wait_ms)))
    except RuntimeError:
        return True


# ---------------------------------------------------------------------------
# Sweep of PNGs whose patch is no longer listed
# ---------------------------------------------------------------------------

#: ``<safe name>.<8-hex path hash>.<mtime_ns>.<st_size>[.c<n>[i]][.s<px>][.l<ns>.o..].png`` -- the
#: shared worker's PNG name (``thumb_png_path``), with or without the colour
#: token.  Anchored; the name part is everything before the hash.
#: a copy of ``_stack_thumbs.PNG_TAIL`` (this module stays Qt-free; a test
#: holds the two equal)
_PNG_TAIL = r"(?:\.s\d+)?(?:\.l\d+)?(?:\.o[a-z]{2}\d+)?"

_PNG_NAME = re.compile(r"^(?P<name>.+)\.[0-9a-f]{8}\.\d+\.\d+(?:\.c\d+i?)?"
                       + _PNG_TAIL + r"\.png$")


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


def sweep_unlisted(cache_dir, listed_names: Iterable[str], *,
                   workspace=None) -> int:
    """Delete the PNGs in ``cache_dir`` whose patch name is not in
    ``listed_names`` -> how many were deleted (PNGs of names no
    longer listed are swept too).

    Only plain files directly in ``cache_dir`` whose name has the shared
    worker's PNG shape (anchored) are considered -- never a sub-folder, a
    ``.tmp``, a link or any other file.  A listed name is compared in its
    file-safe form (``safe_stack_name``), as the worker wrote it.  With
    ``workspace`` given, a ``cache_dir`` outside it is refused (0, a console
    line).  Never raises.  A PNG of a listed name in ANOTHER folder (same
    name, other path hash) is kept; the worker's own ``stale_pngs`` handles
    a listed patch's older PNGs."""
    if not cache_dir:
        return 0
    d = Path(cache_dir)
    try:
        if not d.is_dir():
            return 0
    except OSError:
        return 0
    if workspace is not None and not _inside(d, Path(workspace)):
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


def remove_patch_pngs(cache_dir, patch_name: str, *, workspace) -> int:
    """Delete the PNGs in ``cache_dir`` of ONE patch -> how many (a Forget).
    The same rules as :func:`sweep_unlisted`: only plain files
    directly in ``cache_dir`` with the worker's PNG shape, whose name part is
    *patch_name*'s file-safe form; a ``cache_dir`` outside *workspace* is
    refused (0).  Never raises."""
    if not cache_dir or not patch_name:
        return 0
    d = Path(cache_dir)
    try:
        if not d.is_dir():
            return 0
    except OSError:
        return 0
    if not _inside(d, Path(workspace)):
        _diag(f"remove refused: {d} is outside the workspace {workspace}")
        return 0
    from ..ui_shell._stack_thumbs import safe_stack_name

    want = safe_stack_name(str(patch_name))
    n = 0
    try:
        entries = list(d.iterdir())
    except OSError as exc:
        _diag(f"remove: cannot list {d}: {exc}")
        return 0
    for p in entries:
        m = _PNG_NAME.match(p.name)
        if m is None or m.group("name") != want:
            continue
        try:
            if p.is_symlink() or not p.is_file():
                continue
            p.unlink()
            n += 1
        except OSError as exc:
            _diag(f"remove: could not delete {p.name}: {exc}")
    if n:
        _diag(f"removed {n} picture(s) of {patch_name}")
    return n


# ---------------------------------------------------------------------------
# Tooltip
# ---------------------------------------------------------------------------

def thumb_tooltip_for_failure(reason) -> str:
    """The row's tooltip when no picture could be drawn: ``S3_THUMB_FAILED``
    with the reason in Step 3's words (the shared worker's four reasons speak
    of a "stack"; they are reworded for a patch).  Any other reason is shown
    as it came; none at all reads ``S3_THUMB_FAIL_UNKNOWN``."""
    text = str(reason or "").strip()
    if text:
        from ..ui_shell import _stack_thumbs as T

        for const, words in _STEP3_REASONS.items():
            if text == getattr(T, const, None):
                text = words
                break
    return S3_THUMB_FAILED.format(reason=text or S3_THUMB_FAIL_UNKNOWN)
