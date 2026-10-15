"""Run a saved Refine recipe headless over many objects.

Pure and napari-free: no Qt, no napari, no import from ``napari_epicycle``.
``refine``, ``smooth`` and ``reshape.settle`` are imported lazily, inside the
functions that call them.

One object is a REPLICA of the Refine tab: every step runs on
exactly the ROI the tab would cut for it -- the scalar step margin, clamped
to the volume, no slack, grown before a later step exactly as
``_refine_room`` / ``_grow_refine_region`` grow it -- cut out of ONE outer
crop per object.  The wall picture and the channel's normalisation are
computed on that ROI's crop, so a crop even two voxels different gives a
different answer; the replica is what makes "batch == Refine tab" true.

Verdicts per step:

* **refused** -- the chain STOPS; the steps before it are kept (D12);
* **skipped** -- the step moved nothing; the chain goes on (D11);
* **emptied** -- the fenced result has no voxel; NOTHING of the object is
  written, its earlier steps included;
* **cancelled** -- the cancel event, or a contour that reports "stopped".
"""

from __future__ import annotations

import math
import os
import threading
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The Refine tab's shared margin (``REFINE_SHARED_ROI_MARGIN`` in the
#: annotator; a parity test pins the copy).
REFINE_SHARED_ROI_MARGIN = 15

#: At most this many objects run at once on a chain.  Every Settle and
#: snap-contour object also builds its wall picture, and the watershed /
#: contour loops hold the GIL part of the time, so more threads stop paying
#: well before the core count.  The live driver measured the knee on 10
#: objects, Smooth -> Settle: 1 worker 27.2 s, 2 workers 17.0 s, 4 workers
#: 15.4 s -- a fourth thread buys ~10 % for twice the memory.
CHAIN_MAX_WORKERS = 2

#: How much memory a chain batch may use at once, and the bytes one
#: voxel of an object's outer crop costs while its chain runs: the float64
#: channel crop, the wall picture, the band / distance arrays, the labels.
BATCH_CHAIN_RAM_BUDGET = 8 * 1024 ** 3
CHAIN_BYTES_PER_VOXEL = 64

#: The growth bound of the two methods with no hard one: a smoothed corner
#: can bulge by a voxel.  Only the outer crop uses it -- a memory
#: window -- so it may be generous.
SMOOTH_GROWTH_SLACK = 2

#: How often the collector wakes to look at the cancel event.
CANCEL_POLL_SECONDS = 0.2

#: Per-object reasons: the ``[BATCH]`` print only, never the summary.
BATCH_REFUSED_FAILED = "the step failed"
BATCH_REFUSED_ROOM = "the object grew further than this step's room allows"

#: The contour reasons that are REFUSALS (``refine.py``'s ``details``).
#: ``"stopped"`` is a cancel; ``None`` / ``"limit"`` / ``"fence"`` ran.
CONTOUR_REFUSED = frozenset({"missing", "too_large", "push_below_gate"})

#: Wall kinds the Refine tab builds WITHOUT a channel.
_NO_CHANNEL_WALLS = ("probability", "distance", "skeleton")

#: Vocabulary copies (the recipe model stores engine TOKENS; the panel's
#: ``get_refine_params()`` carries two of them as combo INDICES).
_CONTOUR_MODES = ("snap", "brightness")
_SLICE_SAMPLINGS = ("share", "count", "every")


def chain_worker_count(n_workers=None) -> int:
    """``n_workers`` when given, else ``min(max(1, cpu - 2), CHAIN_MAX_WORKERS)``."""
    if n_workers:
        return max(1, int(n_workers))
    return int(min(max(1, (os.cpu_count() or 2) - 2), CHAIN_MAX_WORKERS))


# ---------------------------------------------------------------------------
# A recipe step, read the way the Refine tab reads its page
# ---------------------------------------------------------------------------


def panel_params(step) -> dict:
    """The recipe step as the ``get_refine_params()``-shaped dict the Refine
    tab's own helpers read (``contour_mode`` as an INDEX, ``contour_limit_um``
    and friends), so the batch hands the engine exactly what the tab would.

    ``record_from_params`` is the inverse for ``contour`` / ``settle`` /
    ``sdf``; the four others keep their own keys.
    """
    m = step.method
    p = dict(step.params or {})
    if m == "contour":
        return {
            "contour_mode": _CONTOUR_MODES.index(p["mode"]),
            "wall_kind": str(p["wall_kind"]),
            "contour_axis": str(p["axis"]),
            "contour_sampling": _SLICE_SAMPLINGS.index(p["sampling"]),
            "contour_share_pct": p["share_pct"],
            "contour_n_slices": p["n_slices"],
            "contour_limit_um": p["limit_um"],
            "contour_iterations": p["iterations"],
            "contour_balloon": p["balloon"],
            "contour_gate": p["gate"],
            "contour_blur_um": p["blur_um"],
            "contour_sharpness": p["sharpness"],
            "contour_rebuild_um": p["rebuild_um"],
            "contour_lambda1": p["lambda1"],
            "contour_lambda2": p["lambda2"],
            "prep_chain": str(p["prep_chain"]),
        }
    if m == "settle":
        return {
            "wall_kind": str(p["wall_kind"]),
            "solver": str(p["solver"]),
            "band_um": p["band_um"],
            "n_iter": p["n_iter"],
            "clamp": p["clamp"],
            "compactness": p["compactness"],
            "prep_chain": str(p["prep_chain"]),
        }
    return p


def _finest(spacing) -> float:
    return max(float(min(tuple(spacing))), 1e-6)


def refine_step_margin(method, params, spacing) -> int:
    """A pure copy of the annotator's ``_refine_step_margin`` (+
    ``_contour_roi_margin`` / ``_settle_roi_margin``).  ``params`` is the
    panel-shaped dict (:func:`panel_params`)."""
    p = params or {}
    if method == "settle":
        from .reshape.settle import roi_margin_voxels

        return int(max(15, roi_margin_voxels(
            float(p.get("band_um", 2.0)), tuple(spacing))))
    if method == "contour":
        limit = float(p.get("contour_limit_um", 3.0))
        return int(max(int(REFINE_SHARED_ROI_MARGIN),
                       math.ceil(limit / _finest(spacing)) + 2))
    return int(REFINE_SHARED_ROI_MARGIN)


def growth_bound(method, params, spacing) -> int:
    """How far one step can move the object's box, in voxels."""
    p = params or {}
    if method == "dilate_erode":
        return max(int(p.get("radius", 2)), 0)
    if method == "guided_grow":
        return max(int(p.get("max_radius", 5)), 0)
    if method == "contour":
        return int(math.ceil(float(p.get("contour_limit_um", 3.0))
                             / _finest(spacing)))
    if method == "settle":
        from .reshape.settle import roi_margin_voxels

        return int(roi_margin_voxels(float(p.get("band_um", 2.0)),
                                     tuple(spacing)))
    return int(SMOOTH_GROWTH_SLACK)            # sdf, fill_holes


def outer_margin(steps, spacing) -> int:
    """``max over k of (sum_{j<k} g_j + m_k)`` -- an upper bound on every
    replica ROI, so one outer crop holds them all."""
    best = 0
    grown = 0
    for step in steps:
        p = panel_params(step)
        best = max(best, grown + refine_step_margin(step.method, p, spacing))
        grown += growth_bound(step.method, p, spacing)
    return int(best)


def estimate_chain_bytes(voxels) -> int:
    """The memory one object's chain may hold at once."""
    return int(voxels) * int(CHAIN_BYTES_PER_VOXEL)


def step_reads_channel(method, params) -> bool:
    """The Refine tab's own launch rule (``_launch_refine_contour`` /
    ``_launch_refine_settle``), on a panel-shaped dict."""
    p = params or {}
    kind = str(p.get("wall_kind", "membrane"))
    if method == "contour":
        return int(p.get("contour_mode", 0)) == 1 or kind not in _NO_CHANNEL_WALLS
    if method == "settle":
        return kind not in _NO_CHANNEL_WALLS
    return False


def step_reads_prob(method, params) -> bool:
    p = params or {}
    kind = str(p.get("wall_kind", "membrane"))
    if method == "guided_grow":
        return True
    if method == "contour":
        return int(p.get("contour_mode", 0)) == 0 and kind == "probability"
    if method == "settle":
        return kind == "probability"
    return False


def chain_step_call(step, inp, lid, ch, pr, spacing, cancel_event=None):
    """``(fn, kwargs)`` for one step -- exactly the dict the Refine tab
    builds (``_refine_contour_call`` / ``_launch_refine_settle`` and the
    synchronous branches of ``_preview_smooth``), arrays aside."""
    m = step.method
    p = panel_params(step)
    lid = int(lid)
    spacing = tuple(float(s) for s in spacing)
    if m == "sdf":
        from .smooth import smooth_label_sdf

        return smooth_label_sdf, dict(
            volume_roi=inp, label_id=lid,
            sigma_z=float(p.get("sigma_z", 3.0)),
            sigma_xy=float(p.get("sigma_xy", 1.5)),
            preserve_topology=bool(p.get("preserve_topology", True)),
        )
    if m == "dilate_erode":
        from .refine import dilate_erode_label

        return dilate_erode_label, dict(
            volume_roi=inp, label_id=lid,
            radius=int(p.get("radius", 2)), axis=str(p.get("axis", "3D")),
            neighbor_constraint=True,
        )
    if m == "fill_holes":
        from .refine import fill_holes_label

        return fill_holes_label, dict(
            volume_roi=inp, label_id=lid, axis=str(p.get("axis", "3D")))
    if m == "guided_grow":
        from .refine import guided_grow

        return guided_grow, dict(
            volume_roi=inp, label_id=lid, image_roi=None, prob_roi=pr,
            max_radius=int(p.get("max_radius", 5)),
            prob_threshold=float(p.get("prob_threshold", 0.5)),
            axis=str(p.get("axis", "3D")),
        )
    if m == "contour":
        from .refine import contour_with_walls
        from .reshape.spec import DEFAULT_SIZE_BUDGET

        mode = int(p.get("contour_mode", 0))
        kind = str(p.get("wall_kind", "membrane"))
        return contour_with_walls, dict(
            volume_roi=inp, label_id=lid,
            mode=mode,
            wall_kind=kind,
            channel_roi=ch if step_reads_channel(m, p) else None,
            prob_roi=pr if step_reads_prob(m, p) else None,
            prep_chain=str(p.get("prep_chain", "none")),
            blur_um=float(p.get("contour_blur_um", 2.0)),
            spacing=spacing,
            limit_um=float(p.get("contour_limit_um", 3.0)),
            iterations=int(p.get("contour_iterations", 100)),
            balloon=float(p.get("contour_balloon", 0.0)),
            gate=float(p.get("contour_gate", 0.30)),
            lambda1=float(p.get("contour_lambda1", 1.0)),
            lambda2=float(p.get("contour_lambda2", 1.0)),
            sharpness=float(p.get("contour_sharpness", 100.0)),
            orientation=str(p.get("contour_axis", "3D")),
            sampling=int(p.get("contour_sampling", 0)),
            share_pct=float(p.get("contour_share_pct", 25.0)),
            n_slices=int(p.get("contour_n_slices", 12)),
            rebuild_um=float(p.get("contour_rebuild_um", 1.5)),
            size_budget=int(DEFAULT_SIZE_BUDGET),
            cancel_event=cancel_event,
        )
    if m == "settle":
        from .reshape.settle import settle_with_walls

        kind = str(p.get("wall_kind", "membrane"))
        return settle_with_walls, dict(
            labels_roi=inp, label_id=lid,
            wall_kind=kind,
            channel_roi=ch if step_reads_channel(m, p) else None,
            prob_roi=pr if step_reads_prob(m, p) else None,
            prep_chain=str(p.get("prep_chain", "none")),
            band_um=float(p.get("band_um", 2.0)),
            spacing=spacing,
            solver=str(p.get("solver", "watershed")),
            compactness=float(p.get("compactness", 0.0)),
            n_iter=int(p.get("n_iter", 10)),
            clamp=float(p.get("clamp", 0.5)),
        )
    raise ValueError("a batch cannot run the method {!r}".format(m))


# ---------------------------------------------------------------------------
# One object
# ---------------------------------------------------------------------------


@dataclass
class ChainOutcome:
    """What one object's chain came to.

    ``status``: ``"ok"`` / ``"unchanged"`` / ``"stopped"`` / ``"emptied"`` /
    ``"cancelled"``.  ``mask`` is a bool mask cropped to the RESULT's own
    bounding box, placed at ``offset`` (volume coordinates) -- never the
    outer crop, never a wall picture (invariant 17).  ``box`` is the
    object's box in the snapshot, so a write can clear its old footprint.
    ``emptied`` / ``cancelled`` carry no mask.
    """

    status: str
    step: int = 0
    reason: str = ""
    mask: Optional[np.ndarray] = None
    offset: tuple = ()
    n_changed: int = 0
    box: tuple = ()


def _pad(box, margin, shape):
    """``box`` padded by a scalar margin and clamped (``_snapshot_label_roi``
    / ``_label_bbox_roi`` arithmetic)."""
    m = max(0, int(margin))
    return tuple(slice(max(0, int(s.start) - m), min(int(n), int(s.stop) + m))
                 for s, n in zip(box, shape))


def _box_of(mask):
    """The tight box of a bool mask, or None."""
    out = []
    for ax in range(mask.ndim):
        other = tuple(a for a in range(mask.ndim) if a != ax)
        idx = np.where(np.any(mask, axis=other))[0]
        if idx.size == 0:
            return None
        out.append(slice(int(idx[0]), int(idx[-1]) + 1))
    return tuple(out)


def _shift(box, by):
    return tuple(slice(int(s.start) + int(o), int(s.stop) + int(o))
                 for s, o in zip(box, by))


def _inside(inner, outer):
    return all(int(i.start) >= int(o.start) and int(i.stop) <= int(o.stop)
               for i, o in zip(inner, outer))


def refine_room(cur_in_roi, roi, shape):
    """The annotator's ``_refine_room`` ``have``: the object's smallest
    clearance to an INTERIOR ROI wall (infinite where the wall is the
    volume's own edge).  ``cur_in_roi`` is the object's mask over ``roi``."""
    if not cur_in_roi.any():
        return float("inf")
    have = float("inf")
    for ax_ in range(3):
        other = tuple(a for a in range(3) if a != ax_)
        idx = np.where(np.any(cur_in_roi, axis=other))[0]
        lo, hi = int(idx[0]), int(idx[-1])
        start = int(roi[ax_].start or 0)
        stop = int(roi[ax_].stop if roi[ax_].stop is not None else shape[ax_])
        if start > 0:
            have = min(have, lo)
        if stop < shape[ax_]:
            have = min(have, (stop - start) - 1 - hi)
    return have


def _outcome_mask(status, k, reason, cur, orig_mask, O, box):
    n_changed = int(np.count_nonzero(cur != orig_mask))
    bb = _box_of(cur)
    if bb is None:                             # pragma: no cover - guarded
        return ChainOutcome("emptied", k, reason, box=box)
    offset = tuple(int(s.start) + int(o.start) for s, o in zip(bb, O))
    return ChainOutcome(status, int(k), str(reason or ""),
                        mask=np.array(cur[bb], dtype=bool, copy=True),
                        offset=offset, n_changed=n_changed, box=tuple(box))


def _log(msg):
    print("[BATCH] " + msg)


def run_chain_on_label(snapshot, label_id, steps, *, bbox, spacing,
                       channel=None, prob=None, cancel_event=None,
                       log=_log) -> ChainOutcome:
    """Run ``steps`` (``RecipeStep``s) on one object as the Refine tab would.

    ``snapshot`` is read, never written.  ``channel`` / ``prob`` are
    volume-shaped array-likes (uncopied); the crops are copied ONCE here,
    over the outer crop, with the Refine tab's expressions.
    """
    lid = int(label_id)
    steps = tuple(steps)
    shape = tuple(int(n) for n in snapshot.shape)
    spacing = tuple(float(s) for s in spacing)
    box = tuple(slice(int(s.start), int(s.stop)) for s in bbox)

    def _cancelled():
        return cancel_event is not None and cancel_event.is_set()

    if _cancelled():
        return ChainOutcome("cancelled", box=box)

    try:
        O = _pad(box, outer_margin(steps, spacing), shape)
        o0 = tuple(int(s.start) for s in O)
        orig = np.asarray(snapshot[O])
        orig_mask = orig == lid
        params = [panel_params(s) for s in steps]
        need_ch = any(step_reads_channel(s.method, p)
                      for s, p in zip(steps, params))
        need_pr = any(step_reads_prob(s.method, p)
                      for s, p in zip(steps, params))
        # The Refine tab's own expressions: `_get_refine_channel_roi` and
        # `_get_prob_roi`.  Cropping O first and viewing each ROI inside it
        # gives the same values the tab crops at that ROI.
        ch = (np.array(channel[O], dtype=np.float64, copy=True)
              if need_ch and channel is not None else None)
        pr = (np.asarray(prob[O]).astype(np.float64)
              if need_pr and prob is not None else None)
    except Exception as exc:  # noqa: BLE001 - one object, never the batch
        # A crop that cannot be cut (a channel of another shape, a map
        # that will not read) stops THIS object before step 1 -- nothing
        # of it is written -- instead of failing the whole batch (D-M2).
        log("object {} could not be cropped: {}: {}".format(
            lid, type(exc).__name__, exc))
        return ChainOutcome("stopped", 1, BATCH_REFUSED_FAILED, box=box)

    roi = _pad(box, refine_step_margin(steps[0].method, params[0], spacing),
               shape) if steps else box
    cur = orig_mask.copy()
    for k, (step, p) in enumerate(zip(steps, params), start=1):
        if _cancelled():
            return ChainOutcome("cancelled", k, box=box)
        m = step.method
        if k > 1:
            need = refine_step_margin(m, p, spacing)
            rl = tuple(slice(int(r.start) - a, int(r.stop) - a)
                       for r, a in zip(roi, o0))
            have = refine_room(cur[rl], roi, shape)
            if need > have:
                cb = _box_of(cur)
                grown = _pad(_shift(cb, o0), need, shape)
                roi = tuple(slice(min(int(a.start), int(b.start)),
                                  max(int(a.stop), int(b.stop)))
                            for a, b in zip(roi, grown))
        if not _inside(roi, O):
            log("object {} stopped at step {} ({}): {}".format(
                lid, k, m, BATCH_REFUSED_ROOM))
            return _outcome_mask("stopped", k, BATCH_REFUSED_ROOM, cur,
                                 orig_mask, O, box)
        rl = tuple(slice(int(r.start) - a, int(r.stop) - a)
                   for r, a in zip(roi, o0))
        o = orig[rl]
        c = cur[rl]
        inp = np.where(c, lid, np.where(o == lid, 0, o)).astype(orig.dtype)
        try:
            fn, kw = chain_step_call(
                step, inp, lid, ch[rl] if ch is not None else None,
                pr[rl] if pr is not None else None, spacing, cancel_event)
            out = fn(**kw)
        except Exception as exc:  # noqa: BLE001 - one object, never the batch
            log("object {} step {} ({}) failed: {}: {}".format(
                lid, k, m, type(exc).__name__, exc))
            log(traceback.format_exc().rstrip())
            return _outcome_mask("stopped", k, BATCH_REFUSED_FAILED, cur,
                                 orig_mask, O, box)

        if m == "contour":
            reason = (getattr(out, "details", None) or {}).get("reason")
            if reason == "stopped":
                return ChainOutcome("cancelled", k, box=box)
            if reason in CONTOUR_REFUSED:
                msg = str(out[2]) if len(out) > 2 else str(reason)
                log("object {} stopped at step {} ({}): {}".format(
                    lid, k, m, msg))
                return _outcome_mask("stopped", k, msg, cur, orig_mask, O,
                                     box)
            result = out[0]
        elif m == "settle":
            status = getattr(out, "status", "ok")
            msgs = list(getattr(out, "messages", []) or [])
            if status == "refused":
                msg = msgs[0] if msgs else "refused"
                log("object {} stopped at step {} ({}): {}".format(
                    lid, k, m, msg))
                return _outcome_mask("stopped", k, msg, cur, orig_mask, O,
                                     box)
            if status == "unchanged":
                log("object {} step {} ({}) skipped: {}".format(
                    lid, k, m, msgs[0] if msgs else "nothing moved"))
                del out
                continue
            result = out.result_roi
        else:
            result = out[0]
        new_r = (np.asarray(result) == lid) & ((o == 0) | (o == lid))
        del out, result                         # drops any wall picture
        new = cur.copy()
        new[rl] = new_r
        if not new.any():
            log("object {} emptied at step {} ({}) - nothing written".format(
                lid, k, m))
            return ChainOutcome("emptied", k, box=box)
        if (new != cur).any():
            cur = new
        else:
            log("object {} step {} ({}) skipped: nothing moved".format(
                lid, k, m))
    changed = bool((cur != orig_mask).any())
    return _outcome_mask("ok" if changed else "unchanged", len(steps), "",
                         cur, orig_mask, O, box)


# ---------------------------------------------------------------------------
# Many objects
# ---------------------------------------------------------------------------


def run_chain_batch(snapshot, ids, boxes, steps, *, spacing, channel=None,
                    prob=None, n_workers=None, cancel_event=None,
                    tick=None, orphans=None):
    """Run every present id's chain on a pool; ``(outcomes, cancelled)``.

    ``outcomes`` is ``[(id, ChainOutcome)]`` for the ids in ``boxes``.
    Collection wakes every :data:`CANCEL_POLL_SECONDS` and looks at the
    cancel event BEFORE consuming anything that completed, so a cancelled
    run never hands back a result.  On a cancel the pool is shut down
    without waiting; futures still RUNNING are appended to ``orphans``
    (their results are dropped; they keep the snapshot alive until they
    end).  ``tick(done)`` is called as each object is collected.
    """
    outcomes = []
    present = [lid for lid in ids if lid in boxes]
    if not present:
        return outcomes, bool(cancel_event is not None and cancel_event.is_set())
    pool = ThreadPoolExecutor(max_workers=chain_worker_count(n_workers))
    cancelled = False
    pending = set()
    futs = {}
    try:
        futs = {
            pool.submit(run_chain_on_label, snapshot, lid, steps,
                        bbox=boxes[lid], spacing=spacing, channel=channel,
                        prob=prob, cancel_event=cancel_event): lid
            for lid in present
        }
        pending = set(futs)
        while pending:
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
            done, pending = wait(pending, timeout=CANCEL_POLL_SECONDS,
                                 return_when=FIRST_COMPLETED)
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                pending |= done
                break
            for fut in done:
                lid = futs[fut]
                try:
                    out = fut.result()
                except Exception as exc:  # noqa: BLE001 - one object only
                    # `run_chain_on_label` catches its own steps; anything
                    # that still escapes stops this object, not the batch
                    # (D-M2).
                    _log("object {} failed: {}: {}".format(
                        lid, type(exc).__name__, exc))
                    out = ChainOutcome("stopped", 1, BATCH_REFUSED_FAILED,
                                       box=tuple(boxes[lid]))
                outcomes.append((lid, out))
                if tick is not None:
                    tick()
    finally:
        # On EVERY exit -- a cancel, or an exception out of this loop --
        # a future that is still running becomes an orphan: its result is
        # dropped, and Run waits for it (D-M2).
        for fut in pending:
            if not fut.cancel() and not fut.done() and orphans is not None:
                orphans.append(fut)
        pool.shutdown(wait=False, cancel_futures=True)
    return outcomes, cancelled


class OrphanList:
    """The abandoned pool futures of a stopped chain run.

    The worker thread appends while the main thread prunes, so both go
    through one lock: a prune that re-reads and re-assigns a plain list
    could drop a future appended in between (D-L1).  ``append`` /
    ``prune`` / ``len`` / iteration are all the annotator and the pool
    use.
    """

    def __init__(self, items=()):
        self._lock = threading.Lock()
        self._items = list(items)

    def append(self, fut) -> None:
        with self._lock:
            self._items.append(fut)

    def prune(self) -> int:
        """Drop the finished futures; how many are still running."""
        with self._lock:
            self._items = [f for f in self._items if not f.done()]
            return len(self._items)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def __iter__(self):
        with self._lock:
            return iter(list(self._items))


__all__ = [
    "REFINE_SHARED_ROI_MARGIN", "CHAIN_MAX_WORKERS", "BATCH_CHAIN_RAM_BUDGET",
    "CHAIN_BYTES_PER_VOXEL", "SMOOTH_GROWTH_SLACK", "CANCEL_POLL_SECONDS",
    "BATCH_REFUSED_FAILED", "BATCH_REFUSED_ROOM", "CONTOUR_REFUSED",
    "ChainOutcome", "chain_worker_count", "panel_params",
    "refine_step_margin", "growth_bound", "outer_margin",
    "estimate_chain_bytes", "step_reads_channel", "step_reads_prob",
    "chain_step_call", "refine_room", "run_chain_on_label", "run_chain_batch",
    "OrphanList",
]
