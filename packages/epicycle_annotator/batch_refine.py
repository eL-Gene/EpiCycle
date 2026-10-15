"""Batch refine executor — apply an ordered chain of geometry-only refine
operations across many labels in a 3D label volume.

Pure algorithm module — **no Qt, no napari, no matplotlib**.  Heavy
dependencies (scipy/skimage) are pulled in transitively through the
existing pure functions in ``smooth.py`` / ``refine.py``; this module
itself only needs numpy + stdlib (``concurrent.futures``, ``os``).

A *recipe* is an ordered list of :class:`RefineStep` (geometry-only):
  - ``"sdf"``          → SDF Gaussian surface smooth (``smooth_label_sdf``)
  - ``"dilate_erode"`` → morphological dilate/erode (``dilate_erode_label``)
  - ``"fill_holes"``   → fill interior cavities (``fill_holes_label``)

Each step consumes the previous step's ROI output, so chaining (and
duplicate steps, e.g. two ``"sdf"`` passes) works.  Results are applied
back into a working copy of the volume via ``apply_smooth_to_volume`` —
the safe-write dispatcher that never clobbers neighbouring labels — in
ascending label-id order for determinism + lower-id-wins contention.
"""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np

from .refine import dilate_erode_label, fill_holes_label
from .smooth import apply_smooth_to_volume, smooth_label_sdf

_VALID_METHODS = ("sdf", "dilate_erode", "fill_holes")
_VALID_AXES = ("3D", "XY", "XZ", "YZ")

# ---------------------------------------------------------------------------
# User-facing sentences of the Batch Refine dialog.
# Every sentence the dialog or its annotator puts on screen is a named
# constant here, so a test can read it and a reword happens in one place.
# ---------------------------------------------------------------------------

#: The target radios' texts, rewritten on every debounced count refresh.
BATCH_TARGET_ALL = "All active ({n:,})"
BATCH_TARGET_SELECTED = "Selected ({n:,})"
BATCH_TARGET_PLOT = "Plot selection ({n:,})"

#: On the count line (Run's reason) when the plot radio is checked at 0.
BATCH_PLOT_EMPTY = "→ Lasso or click points in the QC plot first"
BATCH_PLOT_NONE = "→ No QC plot for this patch yet — extract QC features first."
#: The plot is showing ANOTHER patch (UX audit M6): "lasso first" would be
#: the wrong advice there.
BATCH_PLOT_OTHER_PATCH = (
    "→ The QC plot shows another patch — open this patch's plot first."
)
#: What the controller's getter returns for that case (not a list): the
#: annotator turns it into ``[]`` for the count and this sentence.
PLOT_OTHER_PATCH = "other_patch"

#: The live count line under the targets.  ``{k}`` is :func:`objects_word`.
BATCH_COUNT = "→ Will process {k}"
#: The always-visible banner at the top of the dialog.
BATCH_WARNING_BANNER = (
    "⚠ Batch refine modifies many objects at once. "
    "Preview first. After Save Curation it's permanent."
)

#: Refusals before a run starts.
BATCH_NO_VOLUME = "No label volume loaded."
BATCH_NO_TARGETS = "No objects in the target set."
BATCH_ADD_STEP = "Add at least one step."

#: The status line while a run is in flight.  Preparing and writing
#: show a marquee; refining is determinate and carries the ETA.
#: ``{n}`` is :func:`objects_word` ("1 object", "1,204 objects").
BATCH_STAGE_PREPARING = (
    "Preparing — copying the labels and finding {n}…"
)
BATCH_STAGE_REFINING = "Refining object {done:,} of {total:,}"
BATCH_STAGE_WRITING = "Writing {n} into the volume…"
BATCH_ETA_SEP = " · "
BATCH_ETA_SECONDS = "about {s} s left"
BATCH_ETA_MINUTES = "about {m} min left"
BATCH_CANCELLING = "Cancelling…"
BATCH_CANCELLED = "Batch cancelled — no changes applied."
BATCH_PATCH_CHANGED = "Patch changed — batch discarded."
BATCH_LAUNCH_FAILED = (
    "The batch could not start — nothing was changed. The console has the "
    "details."
)
BATCH_FAILED = (
    "The batch stopped with an error — nothing was changed. "
    "The console has the details."
)

#: The staged summary (one line each; the id lists at most 8, then "…").
#: ``{n}`` is :func:`objects_word` in every summary line.
BATCH_SUMMARY = "✓ Previewed {n} · {v:,} voxels changed"
BATCH_SUMMARY_EMPTIED = (
    "{n} unchanged — a step would have removed all of their voxels, so none "
    "of their steps were kept: {ids}"
)
BATCH_SUMMARY_ABSENT = "{n} no longer in the volume: {ids}"
#: The status line once a result is staged (UX audit M1): the "Writing…"
#: line used to stay on screen next to Accept / Reject.
BATCH_STAGED = "Review the result, then Accept or Reject."
#: Fixed sentences for the builder and Save (UX audit M5): the raw reason
#: goes to the ``[BATCH]`` print, never on screen.
BATCH_STEP_INVALID = "Step {k} has a setting out of range — check its numbers."
BATCH_LOAD_FAILED = "Could not load that recipe — it is empty or unreadable."
#: A standalone annotator has no workspace to save into (UX audit L8);
#: an empty builder says ``BATCH_ADD_STEP`` at Save as at Run.
BATCH_SAVE_NO_WORKSPACE = "Recipes can only be saved inside a workspace."
BATCH_SAVE_INVALID = "Fix the steps before saving."
BATCH_SAVE_NO_NAME = "A recipe name cannot be empty."
BATCH_SAVE_OVERWRITING = "Overwriting the saved recipe “{name}”."
BATCH_SAVE_FAILED = (
    "Could not save the recipe — the console has the details."
)
BATCH_SAVED = "Saved the recipe “{name}”."
BATCH_PRESET_READY = "Smooth is ready — click Run to preview it."
BATCH_KEEP_TIER_LABEL = "Keep original QC tier (don't mark curated)"
BATCH_KEEP_TIER_TIP = (
    "When checked, refined objects stay in their current QC tier "
    "(good / review / …) instead of being reclassified to 'curated'.  "
    "Their measurements are still flagged stale — run Recalculate QC "
    "to refresh."
)
#: The "nothing changed" ending; LISTED when some objects are named.
BATCH_NONE_CHANGED = "No objects changed."
BATCH_NONE_CHANGED_LISTED = "No objects changed.\n{lines}"

#: At most this many ids on one summary line, then "…".
SUMMARY_MAX_IDS = 8
#: At most this many "stopped" lines (one per distinct step), then "…".
SUMMARY_MAX_STOPPED_LINES = 3

# ---- Commit 6 (B): a saved Refine recipe over N objects ----

BATCH_RECIPE_TITLE = "RECIPE  (top → bottom, per object)"
BATCH_SAVED_STEPS_LABEL = "Saved steps"
#: The Refine-tab recipe row.
BATCH_CHAIN_LABEL = "Refine-tab recipe"
BATCH_CHAIN_NONE = "— none: build steps below —"
BATCH_CHAIN_TIP = (
    "Run a recipe saved on the Refine tab (its Recipes menu) on every "
    "target object. The steps below are then the recipe's."
)
BATCH_CHANNEL_LABEL = "Channel"
BATCH_CHANNEL_TIP = (
    "The image channel the recipe's wall or brightness steps read. A "
    "recipe stores no channel."
)
BATCH_ROWS_OFF = (
    "The steps are the Refine recipe's. Pick “none” above to build steps "
    "here."
)
#: The Selected radio when the dialog was opened on explicit targets.
BATCH_TARGET_ADDED = "Added this session ({n:,})"
#: Add's four doors while a batch runs or is staged on its cells:
#: the batch's finish writes its union and its Reject pastes the snapshot
#: back, so a cell Add commits inside it would be erased either way.
ADD_BATCH_BUSY = (
    "Batch Refine is working on these cells — Accept or Reject it first."
)
#: The two buttons.  ``{n}`` is :func:`objects_word`.
BATCH_RUN_BTN = "Run on {n}"
BATCH_PREVIEW_BTN = "Preview on object {id}"
BATCH_PREVIEW_BTN_NONE = "Preview on one object"
BATCH_PREVIEW_TIP = (
    "Run the recipe on this one object first — the object selected in the "
    "viewer or the navigator."
)
BATCH_PREVIEW_NO_OBJECT = (
    "Select one object in the viewer or the navigator to preview on."
)
#: The reason line's prefixes when only one of the two buttons is refused.
BATCH_REASON_RUN_ONLY = "Run: "
BATCH_REASON_PREVIEW_ONLY = "Preview: "
BATCH_ORPHANS_RUNNING = (
    "{n} from the stopped batch are still finishing — Run is available "
    "again in a few seconds."
)
BATCH_CANCELLING_SETTLE = (
    "Cancelling… a Settle step cannot stop halfway; the objects already "
    "started finish and are thrown away."
)
BATCH_SUMMARY_STOPPED = (
    "{n} stopped part-way: step {k} ({method}) could not run on them, so "
    "they keep the steps before it (details in the console): {ids}"
)
#: Stopped at step 1 there are no steps before it to keep (UX audit M4).
BATCH_SUMMARY_STOPPED_FIRST = (
    "{n} left unchanged: step 1 ({method}) could not run on them (details "
    "in the console): {ids}"
)
#: Settle refused because the object has no background around it -- a
#: fully enclosed object (the orchestrator's decision:
#: still "stopped", worded so it does not read as an error).
BATCH_SUMMARY_STOPPED_ENCLOSED = (
    "{n} already touching their neighbours everywhere, so Settle had "
    "nothing to move; their earlier steps were kept: {ids}"
)
#: How Settle words that refusal (``reshape/settle.py``); a test pins the
#: prefix against the real engine.
SETTLE_ENCLOSED_REASON = "There is no background left around this object"
#: The Recent / Saved-steps rows, the Save prompt and an unreadable Refine
#: recipe's greyed item (UX audit L5; the reader's reason goes to the
#: console, never into the item).
BATCH_RECENT_TIP = (
    "Reload a recipe used earlier this session (steps + params only)."
)
BATCH_SAVED_TIP = (
    "Reload a recipe saved to this workspace (steps + params only)."
)
BATCH_SAVE_PROMPT_TITLE = "Save recipe"
BATCH_SAVE_PROMPT_LABEL = "Recipe name:"
BATCH_CHAIN_UNREADABLE = "{name} — cannot be read (details in the console)"
#: The open's refusals, in gate order; shown at both doors.
BATCH_STILL_STOPPING = "The last batch is still stopping — try again in a moment."
BATCH_BUSY = "Step 3 is busy with another job — try again when it finishes."
BATCH_NOT_CLEAN = "Accept or Discard the change on screen first."
#: Gate 0 (UX audit H1): another door clicked while the dialog runs or
#: holds a staged result -- the open dialog is raised instead.
BATCH_ALREADY_OPEN = (
    "Batch Refine is already open — Accept, Reject or Cancel it there first."
)
BATCH_NOTICE_TITLE = "Batch Refine"
#: The navigator's Batch Refine tooltip (the one navigator touch).
NAV_BATCH_REFINE_TIP = (
    "Run steps, or a saved Refine recipe, on all, selected, plot-selected "
    "or tier-filtered objects at once."
)


def objects_word(n) -> str:
    """``"1 object"`` / ``"1,204 objects"``."""
    n = int(n)
    return "{:,} object{}".format(n, "" if n == 1 else "s")

#: The ETA is shown only once this many objects are done AND this many
#: seconds have passed since the refining stage began; under
#: ``ETA_MINUTES_FROM`` seconds left it is given in seconds (rounded UP to
#: 10 s), above it in whole minutes (rounded up).
ETA_MIN_DONE = 2
ETA_MIN_SECONDS = 2
ETA_MINUTES_FROM = 90


def eta_text(elapsed_s, done, total) -> str:
    """The time left, as a status suffix, or ``""`` while it is a guess.

    ``left = elapsed / done * (total - done)``.  ``""`` until ``done >=
    ETA_MIN_DONE`` and ``elapsed_s >= ETA_MIN_SECONDS``, and once the last
    object is done.  Under ``ETA_MINUTES_FROM`` seconds left:
    ``BATCH_ETA_SECONDS`` rounded UP to 10 s (never 0); else
    ``BATCH_ETA_MINUTES`` rounded up to whole minutes.
    """
    try:
        elapsed_s = float(elapsed_s)
        done = int(done)
        total = int(total)
    except (TypeError, ValueError):
        return ""
    if done < ETA_MIN_DONE or elapsed_s < ETA_MIN_SECONDS or done >= total:
        return ""
    left = elapsed_s / done * (total - done)
    if left < ETA_MINUTES_FROM:
        return BATCH_ETA_SECONDS.format(s=max(10, int(math.ceil(left / 10.0)) * 10))
    return BATCH_ETA_MINUTES.format(m=int(math.ceil(left / 60.0)))


def format_id_list(ids, cap=SUMMARY_MAX_IDS) -> str:
    """``"14, 57"``; at most ``cap`` ids (ascending), then ``"…"``."""
    ids = sorted(int(i) for i in ids)
    shown = ", ".join(str(i) for i in ids[:cap])
    return shown + (", …" if len(ids) > cap else "")


def summary_lines(*, emptied=(), absent=(), stopped=None) -> list:
    """The staged summary's extra lines, one per non-empty list.

    ``stopped`` is ``{id: (step, method_word)}`` or ``{id: (step,
    method_word, reason)}``: one line per distinct step, ascending, at most
    :data:`SUMMARY_MAX_STOPPED_LINES` then "…".  Step 1 reads
    ``BATCH_SUMMARY_STOPPED_FIRST`` (nothing before it to keep); a Settle
    that found no background around the object reads
    ``BATCH_SUMMARY_STOPPED_ENCLOSED``.
    """
    lines = []
    emptied = list(emptied or ())
    absent = list(absent or ())
    by_step = {}
    for lid, val in dict(stopped or {}).items():
        k, word = int(val[0]), str(val[1])
        reason = str(val[2]) if len(val) > 2 else ""
        kind = ("enclosed" if reason.startswith(SETTLE_ENCLOSED_REASON)
                else "first" if k == 1 else "")
        by_step.setdefault((k, word, kind), []).append(int(lid))
    for i, ((k, word, kind), ids) in enumerate(sorted(by_step.items())):
        if i >= SUMMARY_MAX_STOPPED_LINES:
            lines.append("…")
            break
        template = {"enclosed": BATCH_SUMMARY_STOPPED_ENCLOSED,
                    "first": BATCH_SUMMARY_STOPPED_FIRST}.get(
                        kind, BATCH_SUMMARY_STOPPED)
        lines.append(template.format(
            n=objects_word(len(ids)), k=k, method=word,
            ids=format_id_list(ids)))
    if emptied:
        lines.append(BATCH_SUMMARY_EMPTIED.format(
            n=objects_word(len(emptied)), ids=format_id_list(emptied)))
    if absent:
        lines.append(BATCH_SUMMARY_ABSENT.format(
            n=objects_word(len(absent)), ids=format_id_list(absent)))
    return lines


@dataclass
class RefineStep:
    """One geometry-only refine operation in a recipe.

    Parameters
    ----------
    method : str
        One of ``"sdf"``, ``"dilate_erode"``, ``"fill_holes"``.
    params : dict
        Method-specific keyword arguments (e.g. ``sigma_z`` / ``sigma_xy``
        / ``preserve_topology`` for sdf, ``radius`` / ``axis`` for
        dilate_erode, ``axis`` for fill_holes).
    """

    method: str
    params: dict = field(default_factory=dict)

    def validate(self) -> None:
        """Raise ``ValueError`` if method or params are out of range."""
        if self.method not in _VALID_METHODS:
            raise ValueError(
                f"Unknown refine method {self.method!r}; "
                f"expected one of {_VALID_METHODS}."
            )
        p = self.params
        if self.method == "sdf":
            for key in ("sigma_z", "sigma_xy"):
                if key in p and not (0.1 <= float(p[key]) <= 10.0):
                    raise ValueError(f"{key}={p[key]} out of range [0.1, 10].")
        elif self.method == "dilate_erode":
            if "radius" in p and not (-10 <= int(p["radius"]) <= 10):
                raise ValueError(f"radius={p['radius']} out of range [-10, 10].")
            if "axis" in p and p["axis"] not in _VALID_AXES:
                raise ValueError(f"axis={p['axis']!r} not in {_VALID_AXES}.")
        elif self.method == "fill_holes":
            if "axis" in p and p["axis"] not in _VALID_AXES:
                raise ValueError(f"axis={p['axis']!r} not in {_VALID_AXES}.")

    # ---- serialization (recipe history / persistence) ----

    def to_dict(self) -> dict:
        """Serialize to a plain JSON-safe dict ``{"method", "params"}``.

        ``params`` is shallow-copied so the returned dict can be mutated /
        round-tripped without aliasing the live step.
        """
        return {"method": self.method, "params": dict(self.params)}

    @classmethod
    def from_dict(cls, d: dict) -> "RefineStep":
        """Rebuild a :class:`RefineStep` from a :meth:`to_dict` dict.

        Validates on load (raises ``ValueError`` for an unknown method or
        out-of-range params).  ``d`` must be a mapping with a ``"method"``
        key; ``"params"`` is optional (defaults to ``{}``).
        """
        if not isinstance(d, dict) or "method" not in d:
            raise ValueError(
                f"Cannot build RefineStep from {d!r}; expected a dict with "
                f"a 'method' key."
            )
        params = d.get("params") or {}
        if not isinstance(params, dict):
            raise ValueError(f"RefineStep params must be a dict, got {params!r}.")
        step = cls(method=str(d["method"]), params=dict(params))
        step.validate()
        return step


def recipe_to_json(recipe) -> list:
    """Serialize an ordered recipe (``list[RefineStep]``) to a list of dicts.

    The result is JSON-safe and round-trippable via :func:`recipe_from_json`.
    """
    return [step.to_dict() for step in recipe]


def recipe_from_json(data) -> list:
    """Rebuild an ordered recipe from a list of dicts (:func:`recipe_to_json`).

    Validates each step on load (via :meth:`RefineStep.from_dict`); raises
    ``ValueError`` clearly on a bad / unknown-method entry so a corrupt
    recipe can't silently degrade into a partial run.
    """
    if not isinstance(data, (list, tuple)):
        raise ValueError(
            f"Recipe JSON must be a list of step dicts, got {type(data).__name__}."
        )
    return [RefineStep.from_dict(d) for d in data]


@dataclass
class BatchResult:
    """Outcome of :func:`run_batch_recipe`."""

    working_volume: np.ndarray
    touched_ids: list
    union_roi: tuple | None
    n_processed: int
    n_changed_total: int
    cancelled: bool
    # R1c: targets whose result had no voxel left to write (the step emptied
    # the crop, or lower ids already took every voxel of it).  Such an
    # object is NOT written — it keeps its footprint — and is not in
    # ``touched_ids``.  Always empty with ``empty="erase"``.
    emptied_ids: list = field(default_factory=list)
    # Targets the worker's own bbox pass did not find in the volume (deleted
    # or merged since the target list was resolved).  Never written.
    absent_ids: list = field(default_factory=list)
    # Chain path only: ``{id: (step, reason)}`` for every object a
    # refused step stopped part-way -- written with the steps before it, or
    # not at all when nothing before it moved; always listed.  Appended LAST
    # (keyword construction only).
    stopped: dict = field(default_factory=dict)


def apply_recipe_to_label(snapshot_volume, label_id, roi, recipe):
    """Run an ordered recipe on a single label within ``roi``.

    Each step feeds its ``result_roi`` into the next step as the new
    ``volume_roi``, so duplicate / chained steps compose.

    Note on ``neighbor_constraint``: every op returns a clean ROI crop
    containing only ``label_id`` + 0 (neighbours stripped), so a step's
    constraint only "sees" neighbour labels present in *that step's*
    input.  In a chain (e.g. ``[dilate, dilate]``) the second dilate sees
    no neighbours in the crop and may over-expand geometrically — but
    cross-label corruption is still impossible: the final write-back via
    ``apply_smooth_to_volume`` is a safe-write against the live volume
    that only fills background (== 0) voxels, never neighbour labels.

    Returns
    -------
    (result_roi, n_changed) : tuple[np.ndarray, int]
        ``result_roi`` is the final ROI crop (only ``label_id`` + 0).
        ``n_changed`` counts voxels whose membership in ``label_id``
        flipped relative to the original snapshot ROI.
    """
    original = snapshot_volume[roi]
    cur = original.copy()
    for step in recipe:
        p = step.params
        if step.method == "sdf":
            cur, _n, _msg = smooth_label_sdf(
                cur,
                label_id,
                sigma_z=p.get("sigma_z", 3.0),
                sigma_xy=p.get("sigma_xy", 1.5),
                preserve_topology=p.get("preserve_topology", True),
            )
        elif step.method == "dilate_erode":
            cur, _n, _msg = dilate_erode_label(
                cur,
                label_id,
                radius=p.get("radius", 2),
                axis=p.get("axis", "3D"),
                neighbor_constraint=True,
            )
        elif step.method == "fill_holes":
            cur, _n, _msg = fill_holes_label(
                cur,
                label_id,
                axis=p.get("axis", "3D"),
            )
        else:  # pragma: no cover - guarded by validate()
            raise ValueError(f"Unknown refine method {step.method!r}.")

    n_changed = int(np.count_nonzero((cur == label_id) != (original == label_id)))
    return cur, n_changed


def resolve_target_labels(volume, df, mode, selected_ids=None, tiers=None):
    """Resolve the set of label ids to process, sorted, excluding 0.

    Modes: ``"all"`` (every positive id present), ``"selected"``
    (``selected_ids`` ∩ present), ``"tier"`` (ids whose ``qc_tier`` is in
    ``tiers``, ∩ present).  Any id not actually present in ``volume`` is
    dropped.  ``df`` may be None (tier mode → ``[]``).
    """
    present = set(int(v) for v in np.unique(volume) if v > 0)

    if mode == "all":
        candidate = present
    elif mode == "selected":
        candidate = {int(i) for i in (selected_ids or [])}
    elif mode == "tier":
        if df is None or tiers is None or "qc_tier" not in df.columns:
            return []
        wanted = set(tiers)
        candidate = {
            int(lbl)
            for lbl, tier in zip(df["label"], df["qc_tier"])
            if tier in wanted
        }
    else:
        raise ValueError(f"Unknown mode {mode!r}; expected all/selected/tier.")

    return sorted(candidate & present)


def resolve_target_ids(present, mode, *, selected_ids=None):
    """Resolve targets against an already-known set of present ids.

    The cached twin of :func:`resolve_target_labels` for the Batch dialog:
    ``present`` is the id set taken ONCE at dialog open (the keys of
    :func:`label_bboxes`), so a count refresh never re-scans the volume.
    Modes: ``"all"`` (every present id) and ``"selected"``
    (``selected_ids`` ∩ present).  Returns a sorted list of ints.
    """
    present = {int(i) for i in present}
    if mode == "all":
        candidate = present
    elif mode == "selected":
        candidate = {int(i) for i in (selected_ids or [])}
    else:
        raise ValueError(f"Unknown mode {mode!r}; expected all/selected.")
    return sorted(candidate & present)


# ``find_objects`` allocates one slot per id up to the largest one, so it is
# used only while the largest id stays within a small multiple of the number
# of objects asked about (or under a flat million, which costs a few MB).
_FIND_OBJECTS_FLAT_BOUND = 1_000_000
_FIND_OBJECTS_PER_ID = 64


def _use_find_objects(dtype, max_id, n):
    """True when one ``ndi.find_objects`` pass is the right tool."""
    if not np.issubdtype(np.dtype(dtype), np.integer):
        return False
    return int(max_id) <= max(_FIND_OBJECTS_FLAT_BOUND, _FIND_OBJECTS_PER_ID * int(n))


def label_bboxes(volume, ids=None):
    """Tight bounding boxes, ``{id: tuple(slice, ...)}``, of positive ids.

    ``ids=None`` → every positive id present; otherwise only the requested
    ids that are present (an absent id has no key).  Each box is
    ``slice(min, max + 1)`` per axis — exactly the numbers a per-id
    ``np.argwhere`` gives, so a ROI padded from it is byte-identical to
    :func:`compute_label_roi`'s own.

    One ``scipy.ndimage.find_objects`` pass when the volume is integer and
    its largest id is within the bound of :func:`_use_find_objects`;
    otherwise the ids are compacted to ``1..K`` with ``np.unique`` and ONE
    ``find_objects`` pass runs on the compacted volume (same boxes, no
    per-id ``argwhere``).
    """
    t0 = time.perf_counter()
    volume = np.asarray(volume)
    want = None if ids is None else sorted({int(i) for i in ids if int(i) > 0})
    if volume.size == 0 or (want is not None and not want):
        return {}
    max_id = volume.max()
    if max_id <= 0:
        return {}
    max_id = int(max_id)

    from scipy import ndimage as ndi

    out = {}
    if _use_find_objects(volume.dtype, max_id, len(want) if want is not None else 1):
        method = "find_objects"
        boxes = ndi.find_objects(volume, max_label=max_id)
        wanted = want if want is not None else range(1, max_id + 1)
        for lid in wanted:
            if lid > max_id:
                continue
            bb = boxes[lid - 1]
            if bb is not None:
                out[int(lid)] = tuple(slice(int(s.start), int(s.stop)) for s in bb)
    else:
        # Sparse ids (a stray huge id) or a non-integer volume: compact the
        # ids to 1..K first, then ONE find_objects pass on the compacted
        # volume -- never an argwhere per object (audit A1-M1).  Only
        # integral positive values are ids, as ``volume == int(id)`` had it.
        method = "compact"
        uniq, inv = np.unique(volume, return_inverse=True)
        is_id = (uniq > 0) & (uniq == np.floor(uniq))
        keys = [int(v) for v in uniq[is_id]]
        k = len(keys)
        if k:
            lut = np.zeros(len(uniq), dtype=np.int32 if k < 2**31 - 1 else np.int64)
            lut[is_id] = np.arange(1, k + 1, dtype=lut.dtype)
            compact = lut[np.asarray(inv).reshape(-1)].reshape(volume.shape)
            del inv
            boxes = ndi.find_objects(compact, max_label=k)
            keep = None if want is None else set(want)
            for key, bb in zip(keys, boxes):
                if bb is None or (keep is not None and key not in keep):
                    continue
                out[key] = tuple(slice(int(s.start), int(s.stop)) for s in bb)
    # Env-gated like the other dev probes (``EPICYCLE_SYNC_DEBUG=1``): this
    # runs on every Add Preview too, so it stays silent by default.
    if os.environ.get("EPICYCLE_SYNC_DEBUG") == "1":
        ms = (time.perf_counter() - t0) * 1000.0
        print(f"[BATCH-PERF] roi: {method} n={len(out)} ms={ms:.1f}")
    return out


def compute_label_roi(volume, label_id, margin, *, bbox=None):
    """Tight bbox of ``label_id`` expanded by ``margin`` and clamped.

    ``margin`` is either a scalar (the same padding on every axis) or a
    per-axis sequence of length ``volume.ndim`` — the anisotropic form
    :func:`_recipe_margin` returns, because an SDF step's sigma is itself
    anisotropic (``sigma_z`` vs ``sigma_xy``).

    ``bbox`` (keyword-only) is the label's tight box as
    :func:`label_bboxes` returns it; given, the per-label ``argwhere`` is
    skipped and the padding / clamping below is unchanged, so the ROI is
    byte-identical.

    Returns a tuple of slices indexing ``volume`` for the label's
    margin-padded bounding box, or ``None`` if the label has zero voxels.
    """
    try:
        margins = [int(m) for m in margin]
    except TypeError:
        margins = [int(margin)] * volume.ndim
    if len(margins) != volume.ndim:
        raise ValueError(
            f"margin has {len(margins)} entries but volume is "
            f"{volume.ndim}-D; pass a scalar or one margin per axis."
        )

    if bbox is not None:
        if len(bbox) != volume.ndim:
            raise ValueError(
                f"bbox has {len(bbox)} slices but volume is {volume.ndim}-D."
            )
        mins = [int(s.start) for s in bbox]
        maxs = [int(s.stop) - 1 for s in bbox]
    else:
        coords = np.argwhere(volume == label_id)
        if coords.size == 0:
            return None
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
    slices = []
    for d in range(volume.ndim):
        lo = max(0, int(mins[d]) - margins[d])
        hi = min(volume.shape[d], int(maxs[d]) + 1 + margins[d])
        slices.append(slice(lo, hi))
    return tuple(slices)


# scipy's ``gaussian_filter`` truncates its kernel at ``truncate=4.0``
# sigmas by default, so a crop that clears the label by less than that
# lets the reflected wall into the convolution deciding the surface.
_SIGMA_KERNEL_RADII = 4.0

# Pre-existing slack, kept: the ceil(4 * sigma) term clears the wall for
# a voxel ON the bbox, but the thresholded surface can land slightly
# OUTSIDE it, and that voxel's kernel reaches correspondingly further.
# 2 is MEASURED, not proven — 72 probes (8 shapes: cube, corner-notched
# cube, plate, rod, sphere, ellipsoid, L, cross x 9 sigma pairs from
# (0.5, 0.5) to (7.0, 3.0)) all matched a margin 20 voxels more generous
# on every axis, 0 differing voxels.  If a future shape/sigma disagrees,
# the honest bound is sigma-tied — e.g. 2 + ceil(0.5 * max(sigma)).
_MARGIN_SLACK = 2


def _recipe_margin(recipe, shape=None):
    """Per-axis ROI padding (Z, Y, X) wide enough that no step feels the crop.

    Two contributions, summed because a recipe's steps compose:

    * ``dilate_erode`` with a **positive** radius grows the footprint by
      that many voxels (a negative radius erodes — it never needs room).
    * ``sdf`` runs ``gaussian_filter`` over a signed distance field and
      thresholds at zero.  scipy truncates the Gaussian at 4 sigma, and
      ``smooth_label_sdf`` hard-clips its result to the crop, so a crop
      that does not clear the label by ~4 sigma reflects the wall into
      the surface decision: the same sigma then yields a weaker, crop-
      dependent shape than the interactive Refine tab (``margin=15``).
      ``sigma_z`` pads Z, ``sigma_xy`` pads Y and X.

    ``shape`` (optional) caps each axis at the volume's own extent —
    past that the crop is the whole volume anyway, so a large sigma on a
    small stack cannot inflate the bbox arithmetic.

    Returns
    -------
    tuple[int, ...]
        One margin per axis — three, or ``len(shape)`` when ``shape`` is
        given.  Never below the legacy ``dilate radii + 2``.
    """
    n_axes = len(shape) if shape is not None else 3
    axes = [_MARGIN_SLACK] * n_axes
    for step in recipe:
        if step.method == "dilate_erode":
            r = int(step.params.get("radius", 2))
            if r > 0:
                axes = [m + r for m in axes]
        elif step.method == "sdf":
            sigma_z = float(step.params.get("sigma_z", 3.0))
            sigma_xy = float(step.params.get("sigma_xy", 1.5))
            # Axis 0 is Z; every remaining axis is lateral.
            axes[0] += math.ceil(_SIGMA_KERNEL_RADII * sigma_z)
            for d in range(1, n_axes):
                axes[d] += math.ceil(_SIGMA_KERNEL_RADII * sigma_xy)

    if shape is not None:
        axes = [min(m, int(s)) for m, s in zip(axes, shape)]
    return tuple(int(m) for m in axes)


# ``_compute_one``'s marker for "the run's bbox pass did not find it".
_ABSENT = object()


def _compute_one(snapshot, label_id, margin, recipe, *, bbox=None):
    """Worker payload: ROI + recipe for a single label (read-only snapshot).

    ``bbox`` is the label's tight box from the run's one
    :func:`label_bboxes` pass; ``_ABSENT`` means that pass did not find the
    label; ``None`` falls back to a per-label ``argwhere``.
    """
    if bbox is _ABSENT:
        return label_id, None, None, 0
    roi = compute_label_roi(snapshot, label_id, margin, bbox=bbox)
    if roi is None:
        return label_id, None, None, 0
    result_roi, n_changed = apply_recipe_to_label(snapshot, label_id, roi, recipe)
    return label_id, roi, result_roi, n_changed


def run_batch_recipe(
    volume,
    label_ids,
    recipe,
    *,
    n_workers=None,
    progress_cb=None,
    cancel_event=None,
    empty="skip",
    stage_cb=None,
    spacing=None,
    channel=None,
    prob=None,
    orphans=None,
):
    """Apply ``recipe`` to every id in ``label_ids`` over a copy of ``volume``.

    ``recipe`` is a ``list[RefineStep]`` (the geometry path, unchanged) or a
    ``refine_recipe.Recipe`` -- a saved Refine chain, which takes the chain
    path: each object is a replica of the Refine
    tab (:mod:`.batch_chain`), a refused step stops that object and keeps
    the steps before it (``BatchResult.stopped``), an emptied object writes
    nothing.  ``spacing`` (default ``(1, 1, 1)``), ``channel`` / ``prob``
    (volume-shaped, uncopied) and ``orphans`` (a list the abandoned pool
    futures of a cancelled chain run are appended to) are read by the chain
    path only.

    Compute is parallel (one ROI task per label, reading the read-only
    snapshot); apply is sequential in ascending label-id order via
    ``apply_smooth_to_volume`` (safe-write, lower-id-wins on contention).

    Progress
    --------
    ``progress_cb(done, total)`` fires as each label's COMPUTE finishes
    (the slow part), not during the fast apply pass — so the bar advances
    gradually while the work actually happens.  Updates are throttled to
    ~100 ticks for large batches.

    ``stage_cb(stage, done, total)`` names the stage: ``("preparing", 0,
    0)`` before the working copy, ``("refining", 0, n)`` BEFORE the first
    compute (the caller's ETA clock starts there) and ``("writing", 0, n)``
    before the apply loop.  A run cancelled before it starts reports only
    "preparing"; a cancelled run never reports "writing".

    Cancel semantics
    ----------------
    If ``cancel_event`` is already set, no compute runs and the returned
    ``working_volume`` is an **unmodified copy** of ``volume`` with
    ``cancelled=True``.  If it is set mid-run, collection stops, pending
    futures are cancelled, the pool is shut down without waiting, and the
    caller (which discards on cancel) gets ``cancelled=True``.

    Empty results (R1c)
    -------------------
    ``apply_smooth_to_volume`` clears an object's old footprint before it
    writes the new one, so a result with no voxel left to write would
    DELETE the object.  With ``empty="skip"`` (the default) such an object
    is not written at all — it keeps its footprint — and is listed in
    ``BatchResult.emptied_ids``; it adds nothing to ``touched_ids``, the
    counts or ``union_roi``.  "No voxel left" is decided at write time:
    ``(result == id) & ((working == 0) | (working == id))``, so a result
    whose every voxel a lower id already took counts too.
    ``empty="erase"`` is the pre-R1c loop verbatim — used by Add's own
    smoothing, where a candidate piece its smoothing empties is dropped.

    Returns
    -------
    BatchResult
    """
    if empty not in ("skip", "erase"):
        raise ValueError(f"empty={empty!r}; expected 'skip' or 'erase'.")

    def _stage(name, done, n):
        if stage_cb is not None:
            stage_cb(name, done, n)

    _stage("preparing", 0, 0)
    snapshot = volume  # treated as the read-only original
    working = volume.copy()
    ids = sorted(int(i) for i in label_ids)
    chain = _is_chain_recipe(recipe)
    # A `Recipe` is a non-iterable dataclass: the geometry margin is the
    # geometry path's alone.
    margin = None if chain else _recipe_margin(recipe, volume.shape)
    total = len(ids)
    step = max(1, total // 100)  # throttle: ~100 progress ticks max

    cancelled = bool(cancel_event is not None and cancel_event.is_set())

    def _tick(done):
        if progress_cb is not None and (done == total or done % step == 0):
            progress_cb(done, total)

    results = []  # (label_id, roi, result_roi, n_changed)
    absent_ids = []
    boxes = {}
    if not cancelled and ids:
        # One pass for every target's box, on this (worker) thread, after
        # the working copy — authoritative for this run.  It replaces one
        # ``argwhere`` over the whole volume per object; the ROIs it pads
        # are byte-identical.
        boxes = label_bboxes(snapshot, ids)
        absent_ids = sorted(lid for lid in set(ids) if lid not in boxes)

    def _box(lid):
        return boxes.get(lid, _ABSENT)

    if chain:
        # After the shared bbox pass: one pass and one `absent_ids` for
        # both paths (audit requirement).
        return _run_chain_path(
            snapshot, working, ids, boxes, absent_ids, recipe,
            cancelled=cancelled, n_workers=n_workers, tick=_tick,
            stage=_stage, cancel_event=cancel_event, spacing=spacing,
            channel=channel, prob=prob, orphans=orphans)

    if not cancelled and ids:
        _stage("refining", 0, total)
        if n_workers == 1:
            done = 0
            for lid in ids:
                if cancel_event is not None and cancel_event.is_set():
                    cancelled = True
                    break
                results.append(
                    _compute_one(snapshot, lid, margin, recipe, bbox=_box(lid))
                )
                done += 1
                _tick(done)
        else:
            max_workers = n_workers or max(1, (os.cpu_count() or 2) - 2)
            pool = ThreadPoolExecutor(max_workers=max_workers)
            try:
                fut_map = {
                    pool.submit(
                        _compute_one, snapshot, lid, margin, recipe, bbox=_box(lid)
                    ): lid
                    for lid in ids
                }
                done = 0
                # Collect as each compute finishes → gradual progress.
                for fut in as_completed(fut_map):
                    if cancel_event is not None and cancel_event.is_set():
                        cancelled = True
                        for f in fut_map:
                            f.cancel()
                        break
                    results.append(fut.result())
                    done += 1
                    _tick(done)
            finally:
                # wait=False + cancel_futures so a cancel returns promptly;
                # abandoned in-flight tasks only read the snapshot (results
                # dropped), so they're harmless.
                pool.shutdown(wait=False, cancel_futures=True)

    # Apply deterministically in ascending label-id order (fast pass).
    if not cancelled:
        _stage("writing", 0, total)
    results.sort(key=lambda r: r[0])
    touched_ids = []
    emptied_ids = []
    n_changed_total = 0
    union_start = None
    union_stop = None

    for lid, roi, result_roi, n_changed in results:
        if roi is None:
            continue
        if empty == "skip":
            target = working[roi]
            keep = (result_roi == lid) & ((target == 0) | (target == lid))
            if not keep.any():
                # Nothing left to write: leave the object exactly as it is.
                emptied_ids.append(lid)
                continue
        applied = apply_smooth_to_volume(working, roi, result_roi, lid)
        if n_changed > 0 or applied > 0:
            touched_ids.append(lid)
            n_changed_total += n_changed
            starts = tuple(s.start for s in roi)
            stops = tuple(s.stop for s in roi)
            if union_start is None:
                union_start, union_stop = list(starts), list(stops)
            else:
                union_start = [min(a, b) for a, b in zip(union_start, starts)]
                union_stop = [max(a, b) for a, b in zip(union_stop, stops)]

    union_roi = (
        tuple(slice(a, b) for a, b in zip(union_start, union_stop))
        if union_start is not None
        else None
    )
    # The voxels that actually changed, counted once, AFTER lower-id-wins
    # -- the same count as the chain path (D-L3).  Each object's own count
    # is taken before contention, so their sum over-counted wherever two
    # objects grew into the same voxels (driver: 7,338 shown vs 7,228).
    if union_roi is not None:
        n_changed_total = int(np.count_nonzero(
            working[union_roi] != snapshot[union_roi]))

    return BatchResult(
        working_volume=working,
        touched_ids=touched_ids,
        union_roi=union_roi,
        n_processed=len([r for r in results if r[1] is not None]),
        n_changed_total=n_changed_total,
        cancelled=cancelled,
        emptied_ids=emptied_ids,
        absent_ids=absent_ids,
    )


def _is_chain_recipe(recipe) -> bool:
    """A saved Refine chain (``refine_recipe.Recipe``), not a step list."""
    from .refine_recipe import Recipe

    return isinstance(recipe, Recipe)


def _run_chain_path(snapshot, working, ids, boxes, absent_ids, recipe, *,
                    cancelled, n_workers, tick, stage, cancel_event, spacing,
                    channel, prob, orphans):
    """The chain branch of :func:`run_batch_recipe`.

    Objects are computed independently on a pool (``batch_chain``), then
    written here in ascending id order with the geometry path's own fence
    and lower-id-wins: a result's write ROI is its old box united with the
    new mask's box, so ``apply_smooth_to_volume`` clears the whole old
    footprint.  An object that wrote nothing moves from ``stopped`` to
    ``emptied_ids`` when lower ids took every voxel of its kept mask.
    """
    from .batch_chain import run_chain_batch
    from .refine_recipe import METHOD_WORD

    total = len(ids)
    spacing = tuple(float(s) for s in (spacing or (1.0, 1.0, 1.0)))
    outcomes = []
    if not cancelled and ids:
        stage("refining", 0, total)
        done = [0]
        # Absent ids count as done at once, as the geometry path's
        # `_compute_one` returns for them.
        for _lid in absent_ids:
            done[0] += 1
            tick(done[0])

        def _one():
            done[0] += 1
            tick(done[0])

        outcomes, cancelled = run_chain_batch(
            snapshot, ids, boxes, tuple(recipe.steps), spacing=spacing,
            channel=channel, prob=prob, n_workers=n_workers,
            cancel_event=cancel_event, tick=_one, orphans=orphans)

    touched_ids = []
    emptied_ids = []
    stopped = {}
    n_changed_total = 0
    union_start = None
    union_stop = None
    n_processed = 0
    # A Cancel that lands after every object was collected (while the
    # summary is being assembled) still cancels: nothing is staged (D-L2).
    if not cancelled and cancel_event is not None and cancel_event.is_set():
        cancelled = True
    if not cancelled:
        stage("writing", 0, total)
        for lid, out in sorted(outcomes, key=lambda r: r[0]):
            n_processed += 1
            if out.status == "cancelled":
                continue
            if out.status == "emptied":
                emptied_ids.append(lid)
                continue
            if out.status == "stopped":
                stopped[lid] = (int(out.step), str(out.reason))
                word = METHOD_WORD.get(
                    recipe.steps[out.step - 1].method, "") if out.step else ""
                print("[BATCH] object {} stopped at step {} ({}): {}".format(
                    lid, out.step, word, out.reason))
            if out.mask is None or out.n_changed <= 0:
                continue
            mbox = tuple(slice(o, o + n) for o, n in zip(out.offset,
                                                          out.mask.shape))
            roi = tuple(slice(min(int(a.start), int(b.start)),
                              max(int(a.stop), int(b.stop)))
                        for a, b in zip(out.box, mbox))
            result_roi = np.zeros(tuple(s.stop - s.start for s in roi),
                                  dtype=working.dtype)
            inner = tuple(slice(b.start - r.start, b.stop - r.start)
                          for b, r in zip(mbox, roi))
            result_roi[inner][out.mask] = lid
            target = working[roi]
            keep = (result_roi == lid) & ((target == 0) | (target == lid))
            if not keep.any():
                emptied_ids.append(lid)
                stopped.pop(lid, None)
                continue
            applied = apply_smooth_to_volume(working, roi, result_roi, lid)
            if out.n_changed > 0 or applied > 0:
                touched_ids.append(lid)
                starts = [s.start for s in roi]
                stops = [s.stop for s in roi]
                if union_start is None:
                    union_start, union_stop = starts, stops
                else:
                    union_start = [min(a, b) for a, b in zip(union_start, starts)]
                    union_stop = [max(a, b) for a, b in zip(union_stop, stops)]
    union_roi = (
        tuple(slice(a, b) for a, b in zip(union_start, union_stop))
        if union_start is not None
        else None
    )
    # The voxels that actually changed, counted once, AFTER lower-id-wins
    # (D-L3): each outcome's own count is taken before contention, so
    # their sum over-counted (394,415 reported against 173,496 on the
    # driver's patch), and a voxel handed from one object to another
    # counts once here, not twice.
    if union_roi is not None:
        n_changed_total = int(np.count_nonzero(
            working[union_roi] != snapshot[union_roi]))
    return BatchResult(
        working_volume=working,
        touched_ids=touched_ids,
        union_roi=union_roi,
        n_processed=n_processed,
        n_changed_total=n_changed_total,
        cancelled=bool(cancelled),
        emptied_ids=sorted(emptied_ids),
        absent_ids=absent_ids,
        stopped=stopped,
    )
