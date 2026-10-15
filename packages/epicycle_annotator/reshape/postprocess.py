"""Refine stage -- ported from the membrane sandbox (``guided_filter_refine``,
``_refine_result_guided``, ``merge_small_pieces``, ``merge_to_count``).

Verbatim apart from the two ``print`` sites, which become
:class:`~.notes.MessageSink` entries -- they carry the only signal that the
"guided filter unavailable" and "guided refine would erase a cell" guards fired.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .notes import MessageSink
from .optional_deps import import_guidedfilter
from .preproc import norm


def guided_filter_refine(mask, guide, radius=4, eps=0.01, *, sink=None):
    """u-Segment3D's guided filter: edge-aware smoothing that pulls a binary MASK's
    boundary onto the GUIDE image's edges (recovers membrane-following protrusions).
    Returns a refined binary mask. Falls back to the input mask if u-Segment3D isn't
    importable -- with a message, never silently."""
    sink = sink if sink is not None else MessageSink()
    m = np.asarray(mask, bool)
    try:
        guidedfilter = import_guidedfilter()
    except ImportError as exc:
        sink.emit(str(exc))
        return m
    try:
        q = guidedfilter(m.astype(np.float32), norm(np.asarray(guide, np.float32)),
                         int(radius), float(eps))
        return np.asarray(q) >= 0.5
    except Exception as exc:
        sink.emit(
            "Boundary refine failed ({!r}) -- the piece boundaries were left as the "
            "solver placed them.".format(exc)
        )
        return m


def refine_result_guided(result, region, guide, radius=4, eps=0.01, *, sink=None):
    """Snap each split piece's boundary onto the guide (membrane) edges via the guided
    filter, then reconcile into a valid partition: an uncontested refined claim wins;
    contested / unclaimed voxels keep their original watershed label. Stays in region."""
    sink = sink if sink is not None else MessageSink()
    ids = [int(i) for i in np.unique(result) if i]
    if len(ids) < 1:
        return result
    masks = {
        pid: (guided_filter_refine(result == pid, guide, radius, eps, sink=sink) & region)
        for pid in ids
    }
    claims = np.zeros(result.shape, np.int32)
    for m in masks.values():
        claims += m.astype(np.int32)
    refined = np.zeros_like(result)
    for pid, m in masks.items():
        refined[m & (claims == 1)] = pid            # uncontested refined claims
    leftover = region & (refined == 0)
    refined[leftover] = result[leftover]            # contested / unclaimed -> original label
    # SAFETY: guided refine must not ERASE a cell (a thin piece whose refined mask
    # eroded to empty while a neighbour expanded over it). If any original piece
    # vanished, keep the unrefined split -- never lose a seed-backed cell.
    if set(ids) - (set(int(v) for v in np.unique(refined)) - {0}):
        sink.emit(
            "Snapping to the wall signal would have erased an object -- kept "
            "the unsnapped cut."
        )
        return result
    return refined


def merge_small_pieces(result, min_size, protect=None):
    """Absorb pieces smaller than ``min_size`` voxels into their most-contacted
    neighbour (so a watershed sliver becomes part of a real cell, NOT a separate
    label and NOT a background hole). ``protect`` = label ids that are NEVER absorbed
    (the real seed-backed cells) -- so a genuine cell can't be silently merged away,
    even if its basin came out small (it's flagged by the quality gate instead).
    Returns the relabeled result."""
    if not min_size or int(min_size) <= 0:
        return result
    keep = set(int(p) for p in protect) if protect is not None else set()
    res = np.asarray(result).copy()
    while True:
        ids = [int(i) for i in np.unique(res) if i != 0]
        if len(ids) <= 1:
            break
        sizes = {i: int((res == i).sum()) for i in ids}
        smalls = [i for i in ids if sizes[i] < int(min_size) and i not in keep]
        if not smalls:
            break
        small = min(smalls, key=lambda i: sizes[i])
        mask = res == small
        dil = ndi.binary_dilation(mask) & ~mask & (res != 0)
        neigh = res[dil]
        neigh = neigh[neigh != small]
        if neigh.size:
            vals, counts = np.unique(neigh, return_counts=True)
            tgt = int(vals[int(np.argmax(counts))])
        else:
            others = [i for i in ids if i != small]
            tgt = max(others, key=lambda i: sizes[i])
        res[mask] = tgt
    return res


def merge_to_count(result, target):
    """Enforce an expected object count. If the split produced MORE pieces than
    ``target``, repeatedly merge the SMALLEST piece into its most-contacted
    neighbour until ``target`` remain (fixes over-segmentation from too many
    seeds). If it produced FEWER, leave as-is -- you can't invent objects without
    more seeds. Returns the relabeled result."""
    res = np.asarray(result).copy()
    while True:
        ids = [int(i) for i in np.unique(res) if i != 0]
        if len(ids) <= int(target):
            break
        sizes = {i: int((res == i).sum()) for i in ids}
        small = min(sizes, key=sizes.get)
        mask = res == small
        dil = ndi.binary_dilation(mask) & ~mask & (res != 0)
        neigh = res[dil]
        neigh = neigh[neigh != small]
        if neigh.size:
            vals, counts = np.unique(neigh, return_counts=True)
            tgt = int(vals[int(np.argmax(counts))])
        else:  # isolated island -> fold into the largest other piece
            others = [i for i in ids if i != small]
            tgt = max(others, key=lambda i: sizes[i])
        res[mask] = tgt
    return res
