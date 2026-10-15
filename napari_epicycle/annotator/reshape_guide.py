"""Guide-mask loading for the Reshape family (Brick R1) -- the PURE half.

A "guide" is a TRUSTED label mask that seeds a guided Split: nuclei labels
produced by another Step 3 run of the same patch, or an external TIFF the
user browses to.  This module answers the three questions that do not need
napari:

  * which runs of this workspace have a labels file for this patch
    (:func:`list_guide_sources`),
  * can this file be read as a label mask at all (:func:`load_guide_array`),
  * does it line up with the volume the user is actually looking at, and
    does it fit in memory (:func:`validate_and_match_guide`).

Everything it returns is a plain numpy array, a small frozen dataclass, or a
plain-language string a biologist can act on.  Creating the read-only napari
Labels layer, docking it, and excluding it from the annotator's managed sets
is the ORCHESTRATOR's job (section 5.4, "Critical structural constraint"), not
this module's.

Hard constraints held here:

  * **napari-free and Qt-free.**  The module imports numpy and stdlib only at
    import time.  ``_paths`` / ``_run_management`` / ``downsample`` are
    imported INSIDE the functions that need them, because
    ``napari_epicycle/__init__.py`` imports ``_widget`` (and therefore napari),
    so any module-level import of a sibling would drag napari into a module
    that is supposed to be pure.  ``tifffile`` is lazy for the same reason of
    keeping import cheap.
  * **ASCII only** in every string and comment -- the Windows console is
    cp1252 and a raising ``print()`` is a control-flow hazard
    (``napari_epicycle/__init__.py::_apply_console_encoding_guard``).
  * **Plain language** in every user-facing message: no "shape mismatch",
    no "dtype", no "ndim".  Say what went wrong and what to do about it.
  * **The mask file on disk is NEVER written to.**  Matching a guide to a
    downsampled session happens in memory only; see
    :func:`validate_and_match_guide`.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "GUIDE_BYTE_BUDGET",
    "GuideSource",
    "GuideRefused",
    "list_guide_sources",
    "guide_source_for_file",
    "load_guide_array",
    "validate_and_match_guide",
    "seed_count",
    "guide_layer_name",
]


#: Default ceiling for the guide array this session will HOLD in memory.
#: A full-resolution int32 guide on the devdata patch (317x480x480) is about
#: 292 MB, held alongside the volume it guides -- hence the budget.
GUIDE_BYTE_BUDGET = 512 * 1024 * 1024

#: The DEFAULT run is the FLAT layout (design section 0.1 / Option A): it has
#: no ``runs/default/`` context dir, so the path helpers must be called with
#: ``run_id=None`` for it.
_DEFAULT_RUN_ID = "default"

#: Extensions we try for a labels stem, in preference order.
_LABEL_EXTENSIONS = (".tif", ".tiff")

_LAYER_NAME_SAFE = re.compile(r"[^A-Za-z0-9._-]+")

_MB = 1024.0 * 1024.0


class GuideRefused(Exception):
    """A guide mask cannot be used, with a biologist-readable reason.

    ``str(exc)`` is meant to be shown to the user verbatim: it names what
    went wrong and what to do next, in plain words.
    """


@dataclass(frozen=True)
class GuideSource:
    """One offerable guide mask -- a run of this workspace, or a file.

    Attributes:
        kind: ``"run"`` or ``"file"``.
        run_id: the run slug; ``""`` for a file.
        display_name: the run's display name, or the file's basename.
        target: the run manifest's free-text target (``"cells"`` /
            ``"nuclei"`` / ...).  Frequently ``""``, in which case the row
            says "target not set".
        status: this patch's status in that run
            (``"reconstructed"`` / ``"curating"`` / ...); ``""`` when the
            state file does not record one, and always ``""`` for a file.
        path: the labels file this source actually READS.
        which_file: the basename of that file -- ``"labels_curated.tif"``,
            ``"labels_3d.tif"``, or ``"external file"``.  Named out loud
            because silently reading a raw reconstruction as "trusted" is
            the whole failure mode (section 5.4).
        label: one line for the picker, ending in the plain sentence
            "you are trusting these labels."
    """

    kind: str
    run_id: str
    display_name: str
    target: str
    status: str
    path: str
    which_file: str
    label: str


# --------------------------------------------------------------------------- #
# Listing the runs that can act as a guide
# --------------------------------------------------------------------------- #
def _paths_run_id(run_id: str) -> Optional[str]:
    """Map a summary's ``run_id`` onto what the path helpers expect.

    The default run is the flat layout, so it must be passed as ``None``.
    """
    if not run_id or str(run_id) == _DEFAULT_RUN_ID:
        return None
    return str(run_id)


def _first_existing(stem: Any) -> str:
    """Return the first existing ``stem + ext`` (``.tif`` preferred), or ``""``."""
    for ext in _LABEL_EXTENSIONS:
        candidate = "{}{}".format(str(stem), ext)
        try:
            if os.path.isfile(candidate):
                return candidate
        except (OSError, ValueError):
            continue
    return ""


def _patch_status_in_run(envelope: Dict[str, Any], run_id: str, patch_name: str) -> str:
    """``runs[run_id].patches[patch_name].status``, defensively -> ``""``."""
    try:
        runs = envelope.get("runs")
        if not isinstance(runs, dict):
            return ""
        entry = runs.get(str(run_id))
        if not isinstance(entry, dict):
            return ""
        patches = entry.get("patches")
        if not isinstance(patches, dict):
            return ""
        record = patches.get(str(patch_name))
        if not isinstance(record, dict):
            return ""
        status = record.get("status")
        return str(status) if status else ""
    except Exception:
        return ""


def _build_run_label(
    display_name: str, target: str, status: str, which_file: str
) -> str:
    """The one-line picker string (section 5.4, "What the run picker shows")."""
    target_part = "target: {}".format(target) if target else "target not set"
    if status:
        status_part = "this patch is {} here".format(status)
    else:
        status_part = "this patch has no status here"
    return "{} - {} - {} - reads {} - you are trusting these labels.".format(
        display_name or "unnamed run", target_part, status_part, which_file
    )


def list_guide_sources(
    workspace: Any,
    patch_name: str,
    *,
    exclude_run_id: Optional[str] = None,
) -> List[GuideSource]:
    """List the runs of ``workspace`` that hold a labels file for ``patch_name``.

    For each run we prefer the CURATED file (``labels_curated.tif``) when it
    exists and fall back to the reconstruction (``labels_3d.tif``), and we say
    which one in the row -- a run that is only reconstructed has no curated
    file, and treating the raw reconstruction as trusted silently is the
    failure mode this picker exists to prevent.

    Archived runs are skipped.  ``exclude_run_id`` is skipped (you never guide
    a run with itself).  A run whose files or state cannot be read is skipped,
    not reported as an error.

    This function NEVER raises.  On any failure it returns what it managed to
    build, possibly an empty list.
    """
    out: List[GuideSource] = []
    try:
        if not workspace or not patch_name:
            return out

        # Lazy: these siblings live under ``napari_epicycle``, whose package
        # __init__ imports napari.  Importing them here keeps THIS module
        # importable without napari.
        from napari_epicycle.fusion_reconstruction import _paths as step3_paths
        from napari_epicycle.fusion_reconstruction import _run_management as run_mgmt

        try:
            summaries = run_mgmt.list_run_summaries(workspace)
        except Exception:
            summaries = []
        if not isinstance(summaries, (list, tuple)):
            summaries = []

        envelope = _read_state_envelope(step3_paths, workspace)
        excluded = str(exclude_run_id) if exclude_run_id else ""

        for summary in summaries:
            try:
                if not isinstance(summary, dict):
                    continue
                run_id = str(summary.get("run_id") or "")
                if not run_id:
                    continue
                if summary.get("archived"):
                    continue
                if excluded and run_id == excluded:
                    continue

                path_run_id = _paths_run_id(run_id)
                curated_stem = step3_paths.patch_curation_labels_stem(
                    workspace, patch_name, path_run_id
                )
                path = _first_existing(curated_stem)
                if not path:
                    recon_stem = step3_paths.patch_labels_stem(
                        workspace, patch_name, path_run_id
                    )
                    path = _first_existing(recon_stem)
                if not path:
                    continue

                which_file = os.path.basename(path)
                display_name = str(summary.get("display_name") or run_id)
                target = str(summary.get("target") or "")
                status = _patch_status_in_run(envelope, run_id, patch_name)

                out.append(
                    GuideSource(
                        kind="run",
                        run_id=run_id,
                        display_name=display_name,
                        target=target,
                        status=status,
                        path=path,
                        which_file=which_file,
                        label=_build_run_label(
                            display_name, target, status, which_file
                        ),
                    )
                )
            except Exception:
                # A broken run is skipped, never fatal.
                continue
    except Exception:
        return out
    return out


def _read_state_envelope(step3_paths: Any, workspace: Any) -> Dict[str, Any]:
    """Read ``step3_state.json`` read-only; ``{}`` on missing / corrupt."""
    try:
        import json

        state_path = str(step3_paths.state_json_path(workspace))
        if not os.path.isfile(state_path):
            return {}
        with open(state_path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


def guide_source_for_file(path: Any) -> GuideSource:
    """Build a ``GuideSource`` for a browsed-to external TIFF.

    Not part of the listing flow (a file is picked, not enumerated), but the
    orchestrator needs a ``GuideSource`` to hand to :func:`guide_layer_name`.
    """
    text = str(path or "")
    base = os.path.basename(text) or "external file"
    return GuideSource(
        kind="file",
        run_id="",
        display_name=base,
        target="",
        status="",
        path=text,
        which_file="external file",
        label=(
            "{} - a file you picked yourself - "
            "you are trusting these labels.".format(base)
        ),
    )


# --------------------------------------------------------------------------- #
# Reading the file
# --------------------------------------------------------------------------- #
def _describe_values(dtype: Any) -> str:
    """Plain words for what a non-label file is holding."""
    kind = getattr(dtype, "kind", "")
    if kind == "f":
        return "decimal numbers"
    if kind == "b":
        return "yes/no values"
    if kind == "c":
        return "pairs of numbers"
    if kind in ("U", "S"):
        return "text"
    return "values of a kind this tool does not understand"


def load_guide_array(path: Any) -> np.ndarray:
    """Read a label mask from a TIFF, or refuse with a plain reason.

    Size-1 axes are squeezed away first (a 1x40x100x100 stack is really a
    40x100x100 volume).  The result is always a 3D whole-number array; a
    picture holding decimal numbers or yes/no values is refused, because a
    guide mask must carry cell ids.
    """
    text = str(path or "")
    if not text:
        raise GuideRefused(
            "No mask file was picked. Choose the labels file you want to use "
            "as a guide."
        )
    if not os.path.isfile(text):
        raise GuideRefused(
            "There is no file at {}. It may have been moved or renamed - "
            "pick the mask file again.".format(text)
        )

    try:
        import tifffile
    except Exception as exc:  # pragma: no cover - tifffile is a hard dependency
        raise GuideRefused(
            "The TIFF reader is not available, so the mask could not be opened."
        ) from exc

    try:
        raw = tifffile.imread(text)
    except Exception as exc:
        raise GuideRefused(
            "That file could not be opened as a picture. It may be damaged, or "
            "it may not be a TIFF. Pick the labels file again: {}".format(
                os.path.basename(text)
            )
        ) from exc

    try:
        array = np.squeeze(np.asarray(raw))
    except Exception as exc:
        raise GuideRefused(
            "That file could not be read as a stack of pictures. "
            "Pick the labels file again: {}".format(os.path.basename(text))
        ) from exc

    if array.ndim != 3:
        raise GuideRefused(
            "A guide mask has to be a 3D stack (depth, height, width), but {} "
            "holds {} dimension(s) once empty ones are ignored. Pick a 3D "
            "labels file.".format(os.path.basename(text), int(array.ndim))
        )

    if array.dtype.kind not in ("i", "u"):
        raise GuideRefused(
            "That file isn't a label mask - it holds {}, not whole-number cell "
            "ids. Pick the labels file from a run, not the raw image.".format(
                _describe_values(array.dtype)
            )
        )

    return array


# --------------------------------------------------------------------------- #
# Matching it to the volume on screen
# --------------------------------------------------------------------------- #
def _shape_words(shape: Sequence[int]) -> str:
    return " x ".join(str(int(s)) for s in shape)


def _size_phrase(factor: int) -> str:
    if factor == 2:
        return "half size"
    if factor == 4:
        return "quarter size"
    return "1/{} size".format(int(factor))


def _mb(nbytes: Any) -> str:
    return "{:.1f}".format(float(nbytes) / _MB)


def validate_and_match_guide(
    guide: np.ndarray,
    volume_shape: Sequence[int],
    *,
    downsample_factor: int = 1,
    byte_budget: int = GUIDE_BYTE_BUDGET,
) -> Tuple[np.ndarray, List[str]]:
    """Line a guide mask up with the volume on screen, or refuse.

    Returns ``(matched_guide, notes)``.  ``notes`` are plain sentences the UI
    shows next to the picker; a refusal is a :class:`GuideRefused` whose text
    is shown to the user as-is.

    When this session is downsampled (``downsample_factor > 1``) and the mask
    is exactly that many times larger on every axis, the mask is majority-
    downsampled to match, by design.
    **The file on disk is never touched** -- the whole operation happens on a
    copy in memory, so re-opening that mask in a full-size session still gives
    full resolution.  A caveat note is returned because majority-downsampling
    a trusted mask can merge or drop small nuclei.
    """
    if not isinstance(guide, np.ndarray):
        raise GuideRefused(
            "The guide mask could not be read. Pick the labels file again."
        )
    if guide.dtype.kind not in ("i", "u"):
        raise GuideRefused(
            "That mask isn't made of whole-number cell ids, so it cannot be "
            "used as a guide. Pick the labels file from a run."
        )
    try:
        target_shape = tuple(int(s) for s in volume_shape)
    except Exception as exc:
        raise GuideRefused(
            "This patch's size could not be worked out, so the mask cannot be "
            "checked against it. Reload the patch and try again."
        ) from exc
    if len(target_shape) != 3 or guide.ndim != 3:
        raise GuideRefused(
            "A guide mask has to be a 3D stack matching this patch. Pick a 3D "
            "labels file."
        )

    try:
        factor = int(downsample_factor)
    except (TypeError, ValueError):
        factor = 1
    if factor < 1:
        factor = 1

    notes: List[str] = []
    matched = guide
    guide_shape = tuple(int(s) for s in guide.shape)

    if factor > 1 and guide_shape != target_shape:
        # ``downsample_labels_majority`` drops the trailing remainder, so the
        # shape it produces is exactly the floor division.  Match against that
        # rather than guessing, so we only downsample when it will line up.
        floored = tuple(s // factor for s in guide_shape)
        if floored == target_shape:
            from napari_epicycle.fusion_reconstruction.downsample import (
                downsample_labels_majority,
            )

            # IN MEMORY ONLY.  ``downsample_labels_majority`` reads ``guide``
            # and returns a NEW array; nothing is ever written back to the file
            # the guide was read from.  Section 5.4: "The file on disk is
            # never touched."
            matched = downsample_labels_majority(guide, factor)
            notes.append(
                "This session is showing the patch at {}, so the mask is being "
                "matched to it. Small nuclei may merge at this resolution - "
                "turn downsampling off in QC Settings if a seed goes "
                "missing.".format(_size_phrase(factor))
            )

    if tuple(int(s) for s in matched.shape) != target_shape:
        if factor > 1:
            raise GuideRefused(
                "This session is showing the patch at {} (downsampling is on), "
                "so a full-size mask doesn't line up. Turn downsampling off in "
                "QC Settings and reload the patch, or pick a mask that matches. "
                "The mask covers {} boxes and this patch is {} boxes.".format(
                    _size_phrase(factor),
                    _shape_words(guide_shape),
                    _shape_words(target_shape),
                )
            )
        raise GuideRefused(
            "That mask is a different size from this patch, so it cannot be "
            "lined up. The mask covers {} boxes and this patch is {} boxes. "
            "Pick a mask made from this same patch.".format(
                _shape_words(guide_shape), _shape_words(target_shape)
            )
        )

    try:
        budget = int(byte_budget)
    except (TypeError, ValueError):
        budget = GUIDE_BYTE_BUDGET
    if budget < 0:
        budget = 0

    # Downcast whenever the cell ids allow it, not only when the mask is over
    # budget (AUDIT 2026-09-09).  The guide is held for the whole session AND
    # handed to napari as a layer, and that layer add is main-thread work
    # proportional to its size -- halving it halves the only part of the load
    # that can stall the window.  Nothing about the ids changes.
    max_id = int(matched.max()) if matched.size else 0
    min_id = int(matched.min()) if matched.size else 0
    if matched.dtype.itemsize > 2 and -32768 <= min_id and max_id <= 32767:
        matched = matched.astype(np.int16, copy=True)
        notes.append(
            "The mask was stored using smaller whole numbers to save memory; "
            "the cell ids are unchanged."
        )

    if matched.nbytes > budget:
            raise GuideRefused(
                "That mask needs {} MB of memory, and this tool only keeps up "
                "to {} MB for a guide. Use a smaller mask, or turn downsampling "
                "on in QC Settings so the mask can be matched to a smaller "
                "patch.".format(_mb(matched.nbytes), _mb(budget))
            )

    matched = np.ascontiguousarray(matched)
    if matched.dtype.kind not in ("i", "u"):  # pragma: no cover - defensive
        matched = matched.astype(np.int32, copy=False)

    notes.append(
        "Guide mask held as {} whole numbers, using {} MB.".format(
            matched.dtype.name, _mb(matched.nbytes)
        )
    )
    return matched, notes


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def seed_count(guide_roi: Any) -> int:
    """How many different cell ids (ignoring the background 0) are in here.

    This is the number the panel reports after Preview -- the user's check
    that downsampling did not lose a nucleus (section 5.4).
    """
    try:
        array = np.asarray(guide_roi)
    except Exception:
        return 0
    if array.size == 0:
        return 0
    try:
        unique = np.unique(array)
    except Exception:
        return 0
    return int(np.count_nonzero(unique))


def _sanitize(text: Any, fallback: str) -> str:
    cleaned = _LAYER_NAME_SAFE.sub("_", str(text or "")).strip("_")
    return cleaned or fallback


def guide_layer_name(patch_name: Any, source: Any) -> str:
    """``guide_<run_or_stem>_<patch>`` -- the layer name the guide gets.

    The ``guide_`` prefix is load-bearing: the controller's
    ``_clear_curation_layers`` sweep keys on it, and without the prefix the
    layer survives every patch switch and becomes a foreign layer in the next
    session (section 5.4, "Cleanup").
    """
    key = ""
    try:
        kind = str(getattr(source, "kind", "") or "")
        if kind == "run":
            key = str(getattr(source, "run_id", "") or "")
        else:
            path = str(getattr(source, "path", "") or "")
            key = os.path.splitext(os.path.basename(path))[0]
        if not key:
            key = str(getattr(source, "display_name", "") or "")
    except Exception:
        key = ""
    return "guide_{}_{}".format(
        _sanitize(key, "source"), _sanitize(patch_name, "patch")
    )
