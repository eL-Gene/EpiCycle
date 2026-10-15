"""The Add tab's worker functions and every sentence it shows (Add rebuild).

Pure, napari-free, Qt-free.  Two functions run on a worker thread through
``RefineWorker``:

* :func:`run_add_preview` -- one Preview of the box: the **Nuclei** preset
  (today's ``add_segment_region`` at the user's settings, byte-identical
  except ids) or the **Cells** preset (the engine's region target,
  ``reshape.region.reshape_empty_region``, with the Sato ridge OFF).  It
  mints the FINAL ids (``next_label + rank``), refuses when the label
  dtype has no room, drops objects under Smallest object, flags the edge
  and small objects, and smooths.  A Stop inside the engine RAISES
  ``ReshapeCancelled``; it is caught here and answered as
  ``status="cancelled"`` -- never an ``error`` with a blank message.
* :func:`add_box_info` -- the one box-info job: the empty share, the
  patch's dark threshold (a strided read of ONE level of each channel), the
  run-count line and the Markers Auto value.

Every user-facing sentence is a named constant here; the panel and the
orchestrator import them.  Raw exception text never reaches the page.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from .reshape.region import (
    ADD_ENGINE_NO_MARKERS,
    ADD_ENGINE_TOO_BIG,
    DARK_SAMPLE_VOXELS,
    too_big_sentence,
)
from .reshape.spec import ADD_BOX_BUDGET

__all__ = [
    "ADD_ENGINE_NO_MARKERS", "ADD_ENGINE_TOO_BIG", "ADD_BOX_BUDGET",
    "AddPreviewResult", "AddBoxInfo", "run_add_preview", "add_box_info",
    "count_phrase", "small_piece_ids", "counts_note", "s1_sentence",
    "preview_block", "dark_stride", "box_line",
]

# --------------------------------------------------------------------------- #
# presets and sources (read BY INDEX by the panel; new items APPEND)
# --------------------------------------------------------------------------- #
ADD_PRESETS = ("nuclei", "cells")
ADD_PRESET_LABELS = ("Nuclei", "Cells with a membrane stain")
ADD_PRESET_TIPS = (
    "Finds bright objects by brightness alone. Best for a nuclear stain.",
    "Grows one object from each marker until it meets a wall in the "
    "membrane channel or dark space. Measures in microns (Step 3's "
    "Spacing).",
)
#: marker sources, as the engine names them
SOURCE_RUN = "labels"
SOURCE_NUCLEI = "nuclei_masks"
SOURCE_CLICKS = "paint"

#: rows in the candidate and REVIEW lists before they scroll
ADD_LIST_ROWS = 4
#: the click-marker list's visible rows: 4 rows measured
#: 761 px in state 7f against the 759.5 px ceiling, 3 rows 739
ADD_MARKER_LIST_ROWS = 3
#: lines of the counts note before it is elided -- 1 since the
#: live-shot fix pass, which showed one line at the 312 px live width
ADD_NOTES_LINES = 1

# --------------------------------------------------------------------------- #
# the description
# --------------------------------------------------------------------------- #
ADD_ABOUT_LINE = (
    "Fills empty space inside a box with new objects — many at once. "
    "Existing objects never lose a voxel, and nothing is written until you "
    "press Accept."
)
ADD_ABOUT_BODY = (
    "<b>Nuclei</b> finds bright objects by brightness alone and suits a "
    "nuclear stain. <b>Cells with a membrane stain</b> grows one object from "
    "each marker until it meets a wall in the membrane channel or dark "
    "space; markers come from another run's objects, from nuclei in a "
    "second channel, or from your clicks, and it measures in microns, from "
    "Step 3's Spacing. New "
    "objects are written only where there is no object yet, so clear the "
    "masks you do not want first. Objects touching the box edge and very "
    "small objects start unticked."
)

# --------------------------------------------------------------------------- #
# S1
# --------------------------------------------------------------------------- #
ADD_S1_IDLE = (
    "Clear the masks you do not want, then place a box over the empty space."
)
ADD_S1_BOX = (
    "Box {z} × {y} × {x} voxels ({m} million; the limit for cells is 4 "
    "million) · {p} % of it is empty space"
)
ADD_S1_BOX_NUCLEI = (
    "Box {z} × {y} × {x} voxels ({m} million) · {p} % of it is empty space"
)
ADD_S1_BOX_CLICKS = "Box {z} × {y} × {x} voxels · {n} markers placed"
ADD_S1_BOX_CLICKS_ONE = "Box {z} × {y} × {x} voxels · 1 marker placed"
ADD_S1_MEASURING = "Measuring the box…"
ADD_S1_OVER_LIMIT = (
    "This box is {m} million voxels; the limit for cells is 4 million. "
    "Shrink it with Numbers… or pick S, M or L."
)
#: Step 3's physical units come from its QC Settings spacing (the same
#: source Reshape, Refine and the QC features read), NOT from Step 1's
#: voxel size, which does not flow into it today (live-shot fix F1, an
#: owner follow-up) -- so the sentence names where it is really set: the
#: "Settings..." button beside "Extract" in Step 3's QC Features section
#: (``fusion_reconstruction/ui.py`` ``qc_settings_btn``), row "Spacing".
#: Cells uses
#: that spacing AS IT IS and never refuses -- ``ADD_S1_NO_VOXEL_SIZE`` and
#: ``ADD_UM_OFF`` are retired; at the default (1, 1, 1) this AMBER,
#: non-blocking note sits under S1 instead (Wound: a wrong 1 um took the
#: leak from 0.002-0.017 to 0.05-0.21).
#: "Accept and refine": Accept, then Batch Refine on
#: everything added this session -- the two shipped paths in a row
ADD_ACCEPT_REFINE_BTN = "Accept and refine ▸"
ADD_ACCEPT_REFINE_TIP = (
    "Adds the ticked objects, then opens Batch Refine on every object added "
    "this session — the same as Accept followed by Refine all."
)
ADD_NOTE_DEFAULT_SPACING = (
    "Spacing is the default 1.0 µm here — if your voxels are smaller, set "
    "Spacing under QC Features → Settings… so the µm knobs mean what they "
    "say."
)
#: the same note while a box is placed (A-M2, the A1-A4 audit): the QC
#: Settings button is in the Step 3 lock set, so it is GREY while an Add box
#: holds the lock -- the note names the way that works (Cancel first)
ADD_NOTE_DEFAULT_SPACING_BOXED = (
    "Spacing is the default 1.0 µm here — Cancel the box, then set Spacing "
    "under QC Features → Settings… if your voxels are smaller."
)
#: the owner's live test (2026-09-27 20:33, D-1): with "Treat dark space as
#: outside" OFF on Cells, nothing but a membrane wall or a box face stops the
#: fill -- the amber pre-result note under S1, the spacing note's rules
ADD_NOTE_DARK_OFF_LEAK = (
    "Dark space is not treated as outside, so objects run to the box faces "
    "wherever there is no wall — tick it in Advanced if objects come out "
    "open at the top or bottom."
)
ADD_S1_PICK_CHANNEL = (
    "Pick the channel the nuclei are bright in, under Brightness."
)
ADD_S1_PICK_NUCLEI_CHANNEL = "Pick the nuclei channel under Markers."
ADD_S1_PICK_WALL_CHANNEL = "Pick the membrane channel under Walls."
ADD_S1_CHANNEL_CLASH = "The nuclei and wall channels must be different."
ADD_S1_NO_MARKERS = (
    "Place at least one marker: turn on Place markers, then click inside "
    "the box."
)
ADD_S1_CLICK_ON_KEPT = (
    "That spot belongs to object {id} — markers go in empty space."
)
ADD_S1_CLICK_OUTSIDE = "Click inside the box."
ADD_S1_CLICKS_DROPPED = (
    "{n} markers were outside the moved box and were removed."
)
ADD_S1_CLICKS_DROPPED_ONE = (
    "1 marker was outside the moved box and was removed."
)
#: A click whose centre is inside an existing marker's ball
#: (ticked or not) -- the earlier ball keeps the shared voxels, so the new
#: marker would be shadowed and every later id would shift
ADD_S1_CLICK_SAME_SPOT = (
    "That spot is already inside marker {id} — click a little further away."
)
#: the same refusal on an UNTICKED marker, whose ball is not on screen (the
#: A5 UX audit, C-L1): say where it is hiding
ADD_S1_CLICK_SAME_SPOT_UNTICKED = (
    "That spot is inside unticked marker {id} — tick it in the list to see "
    "it, or click a little further away."
)
#: Every marker unticked (Preview grey)
ADD_S1_TICK_ONE_MARKER = (
    "Tick at least one marker to cut — all {n} are unticked."
)
ADD_S1_TICK_ONE_MARKER_ONE = "Tick the marker to cut — it is unticked."
#: the A5 data-flow audit (B-M2): a re-base recolours every ball and row,
#: so S1 says why (a transient, amber) -- counted by MARKERS for its
#: singular; the shift itself is {n}
ADD_S1_REBASED = (
    "The object numbers moved up by {n} — something else took the old ones."
)
ADD_S1_REBASED_ONE = (
    "The object number moved up by {n} — something else took the old one."
)
ADD_S1_REBASED_DOWN = (
    "The object numbers moved down by {n} — the lower ones are free again."
)
ADD_S1_REBASED_DOWN_ONE = (
    "The object number moved down by {n} — the lower one is free again."
)
#: the A5 data-flow audit (B-M1): a marker whose ball sits inside a kept
#: object at Preview time makes nothing -- the counts note says so, and
#: its row is dimmed with this tooltip when the list returns on Discard
ADD_NOTE_ABSORBED = "{n} markers now sit inside kept objects and made nothing"
ADD_NOTE_ABSORBED_ONE = (
    "1 marker now sits inside a kept object and made nothing"
)
ADD_MARKER_ABSORBED_TIP = (
    "At the last Preview this marker sat inside a kept object, so it made "
    "nothing — remove it and click in empty space."
)
#: after ADD_S1_BOX_CLICKS when k < n (k >= 1 there, its own singular)
ADD_S1_TICKED_TAIL = " · {k} of {n} ticked"
ADD_S1_WORKING = "Working on the box…"
ADD_S1_BUSY_REFUSED = "Wait for the preview to finish, or press Stop."
ADD_S1_STOPPED = "Stopped. Nothing from this box was added."
ADD_S1_FAILED = (
    "The preview failed — the reason is in the log. Nothing from this box "
    "was added."
)
ADD_S1_INPUTS_CHANGED = "The box changed after this preview — press Preview again."
ADD_S1_FOUND = "{n} new objects found"
ADD_S1_FOUND_ONE = "1 new object found"
ADD_S1_UNTICKED = " · {k} left unticked ({why})"
#: the live count once the user has unticked a row that started ticked --
#: the reason no longer explains every unticked row, so none is given
ADD_S1_UNTICKED_PLAIN = " · {k} left unticked"
ADD_WHY_EDGE = "at the box edge"
ADD_WHY_SMALL = "small"
ADD_WHY_BOTH = "at the box edge or small"
ADD_S1_NOTHING_NUCLEI = "Nothing found — lower the threshold or grow the box."
ADD_S1_ALL_FILTERED = (
    "Every new object was below Smallest object or vanished when smoothed "
    "— lower them under Advanced."
)
ADD_S1_TICK_ONE = "Tick at least one object to add — all are unticked."
ADD_S1_TAKEN = (
    "Those voxels are already taken — move or grow the box and try again."
)
ADD_S1_IDS_FULL = (
    "The label volume has no free ids left for this many objects. "
    "(Nothing was changed.)"
)
ADD_S1_ADDED = "Added {n} objects · {v} voxels"
ADD_S1_ADDED_ONE = "Added 1 object · {v} voxels"
ADD_S1_SELECT_ONE = " Select one object in the list to refine it."
#: REVIEW with more than one row selected: Refine ▸ takes exactly one
ADD_S1_SELECT_JUST_ONE = " Select just one object to refine it."
ADD_S1_COMMIT_FAILED = (
    "The objects are in the volume, but recording them failed — the reason "
    "is in the log. Undo all cannot take them back."
)
ADD_S1_RUN_LOADING = "Reading that run's objects…"
#: the Markers help slot when reading a run's mask FAILED (not refused):
#: the raw error goes to the log, never to the page (B2 fix pass U-M4)
ADD_RUN_READ_FAILED = (
    "That run's objects could not be read — the reason is in the log. "
    "Pick another source."
)
ADD_S1_RUN_UNUSABLE = (
    "That run cannot be used on this patch — the reason is under Markers."
)
ADD_S1_NO_PATCH = "Load a patch before adding objects."

# --------------------------------------------------------------------------- #
# help lines, items, notes, footers, the close prompt
# --------------------------------------------------------------------------- #
ADD_HELP_BOX_EMPTY = (
    "Scroll the side views to the empty space first — the box drops where "
    "the crosshair is."
)
ADD_HELP_DRAG = (
    "Drag the box outline in a side view to move it — press on the line "
    "itself. Click a side view, then Shift + arrow keys nudge it one voxel."
)
ADD_HELP_CELLS = (
    "Markers come from nuclei, another run or your clicks; the cut follows "
    "the membrane."
)
ADD_HELP_NUCLEI = (
    "Finds bright objects by brightness alone. Best for a nuclear stain."
)
ADD_HELP_NUCLEI_MASKS = (
    "Each nucleus found in this channel becomes one marker — the same "
    "finder as the Nuclei preset at its default settings. Slivers pressed "
    "against kept objects are skipped; two nuclei that touch can still come "
    "out as one."
)
ADD_AUTO_PENDING = "Measuring the box for Auto…"
#: the Brightness card's help slot when the box landed bars but no Auto: a
#: flat crop has nothing for Otsu to split, so "pending" would never end
#: (final fix pass B2)
ADD_AUTO_NOTHING_TO_SPLIT = (
    "Nothing to separate in this box — every voxel has the same brightness."
)
#: the Brightness histogram and its Auto are measured on a strided sample of
#: at most this many voxels (final fix pass B5: Local contrast on a full
#: 4 M-voxel crop cost ~2 s per measurement and cannot be cancelled).  The
#: Preview itself always reads the full crop.
BRIGHTNESS_SAMPLE_VOXELS = 1_000_000
#: the Brightness histogram's bin count (the panel's former local default)
ADD_HIST_BINS = 32
#: the Markers help slot while the nuclei channel is blank -- Auto has
#: nothing to measure, so "pending" would stay up for ever (live-shot F9)
ADD_AUTO_NEEDS_CHANNEL = "Pick the nuclei channel first."
ADD_HELP_RUN = (
    "Each object of that run, shrunk by {um} µm, becomes one marker. "
    "Objects too small to shrink are skipped."
)
ADD_RUN_COUNT = "{n} in the box · {m} over empty space become markers."
ADD_RUN_COUNT_ONE = "{n} in the box · 1 over empty space becomes a marker."
ADD_RUN_COUNT_PENDING = "Counting the run's objects in the box…"
ADD_HELP_CLICKS = (
    "While Place markers is on, click inside the box in any side view. "
    "Each click becomes a small ball, about {d} µm across. Ctrl + click "
    "removes one."
)
ADD_AMBER_MEMBRANE_ONLY = (
    "No automatic marker works on a membrane stain alone. Click once inside "
    "each object you want. If dim cells come out empty, lower Dark below or "
    "untick Treat dark space as outside."
)
ADD_HELP_THRESHOLD = (
    "Drag the red line to the dip between background and nuclei, or type a "
    "value, then press Preview."
)
ADD_HELP_DARK = (
    "Dark areas are left empty. The fill may reach this far into them "
    "before it stops."
)
ADD_HELP_DARK_BELOW = (
    "Signal below this share of the patch's usual brightness counts as dark "
    "space. Lower it if dim cells come out empty."
)
ADD_HELP_DARK_OUTSIDE = (
    "Dark space (no signal in any channel) stops the fill, like the box "
    "edges. Untick it if dim cells come out empty."
)
ADD_DARK_OFF = (
    "Dark space is not used here: the box edges alone stop the fill, so "
    "objects next to open space may spill into it."
)
ADD_HELP_REVIEW = (
    "Refine all ▸ opens Batch Refine with all {n} as the target. Refine ▸ "
    "takes the one selected in this list to the Refine tab. + Add another "
    "starts again in the same box."
)
ADD_ITEM_RUN = "Objects from run ▸ {name}"
ADD_ITEM_RAW_SUFFIX = " (reconstruction, not curated)"
ADD_ITEM_NO_RUN = "Objects from another run (no other run of this patch)"
ADD_ITEM_NUCLEI = "Nuclei masks from a channel"
ADD_ITEM_NUCLEI_GREY = "Nuclei masks (needs a second channel)"
ADD_ITEM_CLICKS = "Click markers"

#: the walls picture refused by the display budget -- the legend slot's
#: second line, under WALLS_TOGGLE_TOO_LARGE ("... see the line below")
ADD_WALLS_TOO_LARGE = "This box is too large to show the wall picture."

#: the walls tick on the Nuclei preset (legend slot, on screen)
WALLS_TOGGLE_NO_WALLS = (
    "Nuclei builds no wall picture — the walls are a Cells idea."
)

# the counts note: (key, many, one); joined with " · "
_COUNT_PARTS = (
    # first: the one part the user must act on, so the one-line elision
    # never hides it (B-M1)
    ("n_absorbed", ADD_NOTE_ABSORBED, ADD_NOTE_ABSORBED_ONE),
    ("n_dropped_dark", "{n} markers fell in dark space and were skipped",
     "1 marker fell in dark space and was skipped"),
    ("n_dropped_thin", "{n} run objects the box edge cut too thin to use",
     "1 run object the box edge cut too thin to use"),
    ("n_nuclei_slivers",
     "{n} nuclei slivers against kept objects were skipped",
     "1 nuclei sliver against kept objects was skipped"),
    ("n_markers_over_kept", "{n} markers sat on kept objects",
     "1 marker sat on kept objects"),
    ("n_fragments_dropped",
     "{n} stray pieces cut off by kept objects were left out",
     "1 stray piece cut off by kept objects was left out"),
    ("n_below_min_size", "{n} objects were below Smallest object",
     "1 object was below Smallest object"),
    ("n_emptied_by_smooth",
     "{n} objects vanished when smoothed and were left out",
     "1 object vanished when smoothed and was left out"),
)
ADD_NOTE_COUNTS_SEP = " · "

ADD_CONFIRM_UNDO_ALL = "Take back all {n} objects added this session?"
ADD_CONFIRM_UNDO_ALL_ONE = "Take back the object added this session?"
ADD_UNDO_PARTIAL = (
    "Took back {k} of {n} added objects — a later edit is in the way of the "
    "rest."
)
ADD_UNDO_PARTIAL_ONE = (
    "Took back {k} of 1 added object — a later edit is in the way of the "
    "rest."
)
ADD_UNDO_NONE = (
    "Nothing was taken back — a later edit is in the way. Undo it first, or "
    "use the main Undo button."
)
ADD_UNDO_DONE = "Took back every object added this session."
ADD_FOOT_IDLE = "Nothing has been added yet."
ADD_FOOT_BOX = "Cancel removes the box and its markers; nothing was added."
ADD_FOOT_BOX_SESSION = (
    "Cancel removes this box and its markers; the {n} objects already added "
    "stay."
)
ADD_FOOT_BOX_SESSION_ONE = (
    "Cancel removes this box and its markers; the 1 object already added "
    "stays."
)
ADD_FOOT_CANDIDATES = (
    "Only empty space is written. Discard throws these objects away; the "
    "box and markers stay."
)
ADD_FOOT_REVIEW = (
    "Undo all takes back every object added this session, unless a later "
    "edit is in the way."
)
ADD_CLOSE_CANDIDATES = "{n} new objects are on screen and have not been added."
ADD_CLOSE_CANDIDATES_ONE = (
    "1 new object is on screen and has not been added."
)
ADD_CLOSE_BUSY = (
    "A preview is still running; nothing from this box has been added."
)
ADD_CLOSE_CLICKS = "{n} markers are placed in this box."
ADD_CLOSE_CLICKS_ONE = "1 marker is placed in this box."

#: the chevron summaries (mockup 2a-2d; B2 fix pass U-M5c) -- one elided
#: line each
ADD_SUM_MARKERS = "Markers"
ADD_SUM_MARKERS_NUCLEI = "Markers: nuclei masks"
ADD_SUM_MARKERS_NUCLEI_CH = "Markers: nuclei masks · {channel}"
ADD_SUM_MARKERS_CLICKS = "Markers: clicks · {placed}"
ADD_SUM_MARKERS_RUN = "Markers: another run"
ADD_SUM_MARKERS_RUN_N = "Markers: another run · {n} markers"
ADD_SUM_MARKERS_RUN_ONE = "Markers: another run · 1 marker"
ADD_SUM_WALLS = "Walls"
#: the ROLE the Walls summary names (mockup 2a-2d; B3.4) -- the channel's
#: layer name stays in the Channel combo
ADD_SUM_WALLS_ROLE = "membrane"
ADD_SUM_SEP = " · "
#: the click count beside Place markers ("7 placed")
ADD_CLICKS_PLACED = "{n} placed"
#: the count line and the Markers summary when k < n
ADD_CLICKS_TICKED = "{n} placed · {k} ticked"

#: the orchestrator's inline sentences, in object vocabulary
ADD_REFINING_OBJECT = "Refining object {id} on the Refine tab."
ADD_S1_NOTHING_YET = "Nothing to add yet — press Preview first."
ADD_RECENTER_NO_CROSSHAIR = (
    "The crosshair is not in this patch — move it, then press Recenter."
)
ADD_NO_ADDED_TO_REFINE = "No added objects to refine yet."
ADD_NO_ADDED_TO_SMOOTH = "No added objects to smooth yet."

ADD_AGAIN_TIP = (
    "Start again in the same box, with the same preset and marker source."
)
ADD_UNDO_ALL_TIP = "Take back every object added this session."


# --------------------------------------------------------------------------- #
# formatting
# --------------------------------------------------------------------------- #
def count_phrase(n, one, many, **kw) -> str:
    """``one`` when ``n == 1``, else ``many``; both formatted with ``n`` and
    ``kw``.  The singular partner of :func:`batch_refine.objects_word`."""
    n = int(n)
    return (one if n == 1 else many).format(n=n, **kw)


def millions(n) -> str:
    """``{m}``: millions at 1 dp, at least 0.1.  Rounded DOWN at or under
    the Cells limit and UP above it, so a box over the limit never reads
    as the limit itself ("4.0 million" for 4,000,001 -- B2 fix pass)."""
    n = float(n)
    tenths = n / 1e5
    if n > ADD_BOX_BUDGET:
        v = math.ceil(tenths - 1e-9) / 10.0
    else:
        v = math.floor(tenths + 1e-9) / 10.0
    return "{:.1f}".format(max(0.1, v))


def box_line(shape, *, preset, empty_share=None, n_clicks=None,
             n_ticked=None) -> str:
    """The ready S1 line.  ``n_ticked`` adds
    " · k of n ticked" when fewer than all are ticked."""
    z, y, x = (int(v) for v in shape)
    n = z * y * x
    if preset == "cells" and n_clicks is not None:
        line = count_phrase(n_clicks, ADD_S1_BOX_CLICKS_ONE,
                            ADD_S1_BOX_CLICKS, z=z, y=y, x=x)
        if n_ticked is not None and 0 < int(n_ticked) < int(n_clicks):
            line += ADD_S1_TICKED_TAIL.format(k=int(n_ticked),
                                              n=int(n_clicks))
        return line
    p = int(round(100.0 * float(empty_share or 0.0)))
    tpl = ADD_S1_BOX if preset == "cells" else ADD_S1_BOX_NUCLEI
    return tpl.format(z=z, y=y, x=x, m=millions(n), p=p)


def over_limit_line(shape) -> str:
    return ADD_S1_OVER_LIMIT.format(m=millions(int(np.prod([int(v) for v in shape]))))


def found_line(n, *, k_edge=0, k_small=0, k_both=None, k_plain=None) -> str:
    """``ADD_S1_FOUND`` + the unticked tail.  ``k_edge`` / ``k_small`` are
    the sizes of the UNTICKED rows at the box edge / small; an object in
    both counts once (``k_both``).  ``k_plain`` (the live unticked count
    once the user has unticked a row that started ticked) wins and drops
    the reason (B3.4)."""
    head = count_phrase(n, ADD_S1_FOUND_ONE, ADD_S1_FOUND)
    if k_plain is not None:
        k = int(k_plain)
        return head + ADD_S1_UNTICKED_PLAIN.format(k=k) if k > 0 else head
    k = int(k_edge) + int(k_small) - int(k_both or 0)
    if k <= 0:
        return head
    if k_edge and k_small:
        why = ADD_WHY_BOTH
    elif k_edge:
        why = ADD_WHY_EDGE
    else:
        why = ADD_WHY_SMALL
    return head + ADD_S1_UNTICKED.format(k=k, why=why)


def added_line(n, voxels, *, has_selection=True, n_selected=None) -> str:
    """``ADD_S1_ADDED`` + what Refine ▸ needs: nothing with exactly one row
    selected, ``ADD_S1_SELECT_ONE`` with none, ``ADD_S1_SELECT_JUST_ONE``
    with more than one (B2 fix pass U-M3).  ``has_selection`` is the old
    boolean form."""
    head = count_phrase(n, ADD_S1_ADDED_ONE, ADD_S1_ADDED, v="{:,}".format(int(voxels)))
    if n_selected is None:
        n_selected = 1 if has_selection else 0
    n_selected = int(n_selected)
    if n_selected == 1:
        return head
    # two sentences: the head ends with a period (live-shot fix F5)
    return head + "." + (ADD_S1_SELECT_ONE if n_selected == 0
                         else ADD_S1_SELECT_JUST_ONE)


def run_count_line(n_in_box, n_markers) -> str:
    tpl = ADD_RUN_COUNT_ONE if int(n_markers) == 1 else ADD_RUN_COUNT
    return tpl.format(n=int(n_in_box), m=int(n_markers))


def plain_reason(text, limit=300) -> str:
    """A refusal as the page may show it (B2 fix pass U-M4): its first
    line, every file path cut to its basename (a full path is one
    unbreakable word that word-wrap cannot fold), at most ``limit``
    characters."""
    lines = str(text or "").strip().splitlines()
    first = lines[0] if lines else ""
    out = []
    for word in first.split(" "):
        core = word.rstrip(".,;:)")
        tail = word[len(core):]
        if "/" in core or "\\" in core:
            base = re.split(r"[\\/]", core.rstrip("\\/"))[-1]
            word = (base or core) + tail
        out.append(word)
    return " ".join(out)[:int(limit)]


def click_count_text(n, n_ticked=None) -> str:
    """ "7 placed", or "7 placed · 5 ticked" when fewer are ticked (12.5)."""
    if n_ticked is not None and int(n_ticked) < int(n):
        return ADD_CLICKS_TICKED.format(n=int(n), k=int(n_ticked))
    return ADD_CLICKS_PLACED.format(n=int(n))


def tick_one_marker_line(n) -> str:
    """Every one of ``n`` markers is unticked."""
    return count_phrase(n, ADD_S1_TICK_ONE_MARKER_ONE, ADD_S1_TICK_ONE_MARKER)


def rebased_line(shift, n_markers) -> str:
    """S1's re-base transient (B-M2): ``shift`` is the id shift (its sign
    picks up / down), the singular follows the number of markers."""
    d = int(shift)
    one = int(n_markers) == 1
    if d >= 0:
        tpl = ADD_S1_REBASED_ONE if one else ADD_S1_REBASED
    else:
        tpl = ADD_S1_REBASED_DOWN_ONE if one else ADD_S1_REBASED_DOWN
    return tpl.format(n=abs(d))


def ball_value(label_id) -> int:
    """The uint8 ball layer's value for a marker that will become
    ``label_id``: ``((id - 1) % 49) + 1`` draws the SAME colour under
    napari's default 49-colour cyclic label colormap as the id itself
    does on the candidate layer and in the row swatch.
    Colours repeat every 49 ids, which is why the rows show the id."""
    return ((int(label_id) - 1) % 49) + 1


def marker_row_text(label_id, zyx) -> str:
    """A click-marker row: "Object 412  ·  z / y / x" (the delegate draws
    the part after the separator right-aligned in mono)."""
    z, y, x = (int(v) for v in zyx)
    return "Object {}  ·  {} / {} / {}".format(int(label_id), z, y, x)


def markers_summary(kind, *, channel="", n_clicks=0, n_run_markers=None,
                    n_ticked=None) -> str:
    """The Markers chevron's summary (U-M5c)."""
    if kind == SOURCE_NUCLEI:
        return (ADD_SUM_MARKERS_NUCLEI_CH.format(channel=channel) if channel
                else ADD_SUM_MARKERS_NUCLEI)
    if kind == SOURCE_CLICKS:
        return ADD_SUM_MARKERS_CLICKS.format(
            placed=click_count_text(n_clicks, n_ticked=n_ticked))
    if kind == SOURCE_RUN:
        if n_run_markers is None:
            return ADD_SUM_MARKERS_RUN
        return count_phrase(n_run_markers, ADD_SUM_MARKERS_RUN_ONE,
                            ADD_SUM_MARKERS_RUN_N)
    return ADD_SUM_MARKERS


def walls_summary(*, channel="", wall_kind="", prep=()) -> str:
    """The Walls chevron's summary: channel · wall type · the ticked
    prepare words, e.g. "Walls: membrane · Membrane sheets · even out
    depth" (U-M5c)."""
    parts = [p for p in (channel, wall_kind) if p]
    if prep:
        parts.append(", ".join(prep))
    if not parts:
        return ADD_SUM_WALLS
    return ADD_SUM_WALLS + ": " + ADD_SUM_SEP.join(parts)


def help_clicks(radius_um) -> str:
    return ADD_HELP_CLICKS.format(d="{:g}".format(round(2.0 * float(radius_um))))


def help_run(erode_um) -> str:
    return ADD_HELP_RUN.format(um="{:g}".format(float(erode_um)))


def counts_note(counts, *, source=None) -> str:
    """The grey counts note.  Zero parts are left out;
    the "markers sat on kept objects" part is OMITTED for the run-cores
    source (152-338 per box by design).  ``""`` when every count is 0."""
    parts = []
    for key, many, one in _COUNT_PARTS:
        if key == "n_markers_over_kept" and source == SOURCE_RUN:
            continue
        n = int((counts or {}).get(key, 0) or 0)
        if n > 0:
            parts.append(count_phrase(n, one, many))
    return ADD_NOTE_COUNTS_SEP.join(parts)


def small_piece_ids(sizes) -> set:
    """The widget's "small" rule, exactly: sizes sorted, the middle one
    (``svals[len // 2]``) is the median, and an object under ``0.2 x`` it is
    small.  A zero size is not counted in the median."""
    sizes = {int(k): int(v) for k, v in dict(sizes or {}).items()}
    svals = sorted(s for s in sizes.values() if s)
    median = svals[len(svals) // 2] if svals else 0
    cut = 0.2 * median if median > 0 else 0
    if not cut:
        return set()
    return {i for i, s in sizes.items() if s < cut}


def dark_stride(shape, budget=DARK_SAMPLE_VOXELS) -> int:
    """The smallest stride whose ``[::s, ::s, ::s]`` sample is at most
    ``budget`` voxels."""
    shape = [max(1, int(v)) for v in shape]
    s = 1
    while int(np.prod([math.ceil(n / s) for n in shape])) > int(budget):
        s += 1
    return s


# --------------------------------------------------------------------------- #
# S1's priority table -- first match wins
# --------------------------------------------------------------------------- #
def spacing_is_default(spacing, tol=1e-6) -> bool:
    """True when ``spacing`` is Step 3's default ``(1.0, 1.0, 1.0)`` (within
    ``tol`` per axis) -- or cannot be read as three numbers, which the
    context resolves to that same default."""
    try:
        vals = [float(v) for v in spacing]
    except Exception:
        return True
    if len(vals) != 3:
        return True
    return all(abs(v - 1.0) <= float(tol) for v in vals)


def preview_block(state) -> str | None:
    """The sentence of rows 6-13 (the states in which Preview is grey), or
    ``None``.  Row 7 (the voxel size) is retired: the spacing never
    blocks.  ``state`` keys: ``has_box``,
    ``preset``, ``box_shape``, ``box_budget``, ``channel`` / ``wall_channel`` /
    ``nuclei_channel`` (names, ``""`` = blank), ``source``, ``run_state``
    (``None`` / ``"loading"`` / ``"refused"``), ``n_clicks``,
    ``info_pending``."""
    s = dict(state or {})
    if not s.get("has_box"):
        return ADD_S1_IDLE
    cells = s.get("preset") == "cells"
    shape = s.get("box_shape")
    if cells and shape is not None:
        n = int(np.prod([int(v) for v in shape]))
        if n > int(s.get("box_budget", ADD_BOX_BUDGET)):
            return over_limit_line(shape)
    if not cells:
        if not s.get("channel"):
            return ADD_S1_PICK_CHANNEL
        return None
    if not s.get("wall_channel"):
        return ADD_S1_PICK_WALL_CHANNEL
    source = s.get("source")
    if source == SOURCE_NUCLEI:
        if not s.get("nuclei_channel"):
            return ADD_S1_PICK_NUCLEI_CHANNEL
        if s.get("nuclei_channel") == s.get("wall_channel"):
            return ADD_S1_CHANNEL_CLASH
    if source == SOURCE_RUN:
        if s.get("run_state") == "loading":
            return ADD_S1_RUN_LOADING
        if s.get("run_state") == "refused":
            return ADD_S1_RUN_UNUSABLE
    if source == SOURCE_CLICKS and int(s.get("n_clicks", 0) or 0) == 0:
        return ADD_S1_NO_MARKERS
    if (source == SOURCE_CLICKS and s.get("n_ticked") is not None
            and int(s.get("n_ticked") or 0) == 0):
        # row 12b: markers placed, none ticked
        return tick_one_marker_line(int(s.get("n_clicks", 0) or 0))
    if s.get("info_pending"):
        return ADD_S1_MEASURING
    return None


def s1_sentence(state) -> str:
    """The one S1 sentence.  Adds to :func:`preview_block`'s keys:
    ``has_patch``, ``transient`` (a sentence or ``None``), ``busy``,
    ``review`` (``(n, voxels, has_selection)`` or ``None``), ``candidates``
    (``(n, k_edge, k_small, k_both, n_ticked)`` or ``None``),
    ``empty_share``."""
    s = dict(state or {})
    if not s.get("has_patch", True):
        return ADD_S1_NO_PATCH
    if s.get("transient"):
        return str(s["transient"])
    if s.get("busy"):
        return ADD_S1_WORKING
    review = s.get("review")
    if review is not None:
        n, v, n_sel = review
        # a bool (the old form) or a count of selected rows
        return added_line(n, v, n_selected=int(n_sel))
    cand = s.get("candidates")
    if cand is not None:
        # a 6th entry (the orchestrator's live plain count) is optional
        n, k_edge, k_small, k_both, n_ticked = tuple(cand)[:5]
        k_plain = tuple(cand)[5] if len(tuple(cand)) > 5 else None
        # TICK_ONE only when the USER unticked the last row; when every
        # row STARTED unticked (all at the box edge) the count is the news
        # (mockup; live-shot fix V3)
        if int(n_ticked) == 0 and s.get("user_emptied"):
            return ADD_S1_TICK_ONE
        return found_line(n, k_edge=k_edge, k_small=k_small, k_both=k_both,
                          k_plain=k_plain)
    block = preview_block(s)
    if block is not None:
        return block
    cells = s.get("preset") == "cells"
    if not cells and s.get("empty_share") is None:
        # the empty share is measured on the box-info worker; Nuclei's
        # Preview does not wait for it (it is not row 13), S1 does
        return ADD_S1_MEASURING
    return box_line(
        s.get("box_shape") or (0, 0, 0), preset=s.get("preset"),
        empty_share=s.get("empty_share"),
        n_clicks=(int(s.get("n_clicks", 0) or 0)
                  if cells and s.get("source") == SOURCE_CLICKS else None),
        n_ticked=s.get("n_ticked"))


def s1_kind(state) -> str:
    """The S1 dot for the state s1_sentence reads (live-shot fix V7):
    "idle" no box, "warn" no patch, a transient outcome or a
    blocking reason (rows 7-12 of 4.9a, and TICK_ONE), "busy" working
    or measuring (row 13), "review" candidates or REVIEW, "ready"
    otherwise -- the dot says what the sentence MEANS."""
    s = dict(state or {})
    if not s.get("has_patch", True):
        # ADD_S1_NO_PATCH is a blocking reason: amber here as it is in
        # _start_add's direct push (A-L4, the A1-A4 audit)
        return "warn"
    if s.get("transient"):
        return "warn"
    if s.get("busy"):
        return "busy"
    if s.get("review") is not None:
        return "review"
    cand = s.get("candidates")
    if cand is not None:
        if s1_sentence(s) == ADD_S1_TICK_ONE:
            return "warn"
        return "review"
    block = preview_block(s)
    if block == ADD_S1_IDLE:
        return "idle"
    if block == ADD_S1_MEASURING:
        return "busy"
    if block is not None:
        return "warn"
    if s1_sentence(s) == ADD_S1_MEASURING:
        return "busy"
    return "ready"


# --------------------------------------------------------------------------- #
# the worker functions
# --------------------------------------------------------------------------- #
@dataclass
class AddPreviewResult:
    """One Preview's answer.  ``result`` holds the FINAL ids."""

    status: str = "ok"                 # ok / refused / cancelled
    refusal: str | None = None         # a constant of this module / region
    roi: tuple | None = None           # echoed: the ROI it was computed on
    result: np.ndarray | None = None
    ids: list = field(default_factory=list)
    sizes: dict = field(default_factory=dict)
    edge_ids: set = field(default_factory=set)
    small_ids: set = field(default_factory=set)
    flags: dict = field(default_factory=dict)
    faint_pairs: list = field(default_factory=list)
    n_dropped_dark: int = 0
    n_dropped_thin: int = 0
    n_nuclei_slivers: int = 0
    n_markers_over_kept: int = 0
    n_fragments_dropped: int = 0
    n_below_min_size: int = 0
    n_emptied_by_smooth: int = 0
    unfilled_tissue_fraction: float | None = None
    elevation: np.ndarray | None = None
    notes: list = field(default_factory=list)
    messages: list = field(default_factory=list)
    source: str | None = None
    preset: str = "nuclei"
    #: reserved ids of click markers whose ball the engine never
    #: saw (every voxel taken by an earlier ball or a kept object)
    absorbed_ids: list = field(default_factory=list)

    @property
    def n_absorbed(self) -> int:
        """The counts note's first part (B-M1)."""
        return len(self.absorbed_ids or ())

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def counts(self) -> dict:
        return {key: int(getattr(self, key, 0) or 0)
                for key, _m, _o in _COUNT_PARTS}

    def log_line(self) -> str:
        """The ``[ADD]`` log line: counts, the faint pairs and the unfilled
        share (both notes go to the log only)."""
        faint = ", ".join("{}|{} {:.3g}".format(a, b, m)
                          for a, b, m in self.faint_pairs[:8])
        more = len(self.faint_pairs) - 8
        if more > 0:
            faint += " (+{} more)".format(more)
        uf = ("none" if self.unfilled_tissue_fraction is None
              else "{:.2f}".format(self.unfilled_tissue_fraction))
        line = ("[ADD] preview {} preset={} source={} n={} counts={} "
                "faint=[{}] unfilled={}".format(
                    self.status, self.preset, self.source, len(self.ids),
                    self.counts(), faint, uf))
        if self.absorbed_ids:
            line += " absorbed_markers={}".format(list(self.absorbed_ids))
        return line


def _cancelled(cancel_event) -> bool:
    return cancel_event is not None and cancel_event.is_set()


class _Steps:
    """Emit ``progress(stage, i, n)`` as each stage STARTS; a raising
    callback is swallowed."""

    def __init__(self, progress, names):
        self._cb = progress
        self._names = tuple(names)
        self._i = 0

    def __call__(self, stage):
        self._i += 1
        if self._cb is None:
            return
        try:
            self._cb(stage, self._i, len(self._names))
        except Exception:
            pass


def _sizes_of(result):
    ids = [int(v) for v in np.unique(result) if v]
    if not ids:
        return ids, {}
    counts = np.bincount(result.ravel().astype(np.int64, copy=False))
    return ids, {i: int(counts[i]) if i < counts.size else 0 for i in ids}


def _finish(out, result, params, cancel_event, step):
    """Edge flags on the RAW result, then the smoothing; sizes, small."""
    from .add_segment import edge_flag_ids, smooth_pieces

    if params.get("clear_border", True) and result.ndim == 3:
        out.edge_ids = set(edge_flag_ids(result, int(params.get("edge_margin", 3))))
    if _cancelled(cancel_event):
        out.status = "cancelled"
        return out
    if params.get("auto_smooth", True) and result.ndim == 3:
        step("add_smooth")
        before = {int(v) for v in np.unique(result) if v}
        result = smooth_pieces(result, float(params.get("smooth_sigma", 1.5)))
        after = {int(v) for v in np.unique(result) if v}
        out.n_emptied_by_smooth = len(before - after)
        out.edge_ids &= after
    if _cancelled(cancel_event):
        out.status = "cancelled"
        return out
    out.ids, out.sizes = _sizes_of(result)
    out.small_ids = small_piece_ids(out.sizes)
    out.result = result
    return out


def _ids_fit(next_label, n, id_max) -> bool:
    if id_max is None or n <= 0:
        return True
    return int(next_label) + int(n) - 1 <= int(id_max)


def run_add_preview(*, preset, roi, existing_roi, next_label, spacing, params,
                    image_roi=None, nuclei_roi=None, wall_roi=None,
                    dark_rois=None, run_guide_roi=None, click_points=(),
                    reserved_ids=None, dark_threshold=None, id_max=None,
                    cancel_event=None, progress=None) -> AddPreviewResult:
    """One Preview of the box.  Every array is a BOX CROP the
    main thread copied; ``click_points`` are VOLUME ZYX (made box-local
    here).  ``id_max`` is the label dtype's largest id.
    ``reserved_ids``, one per click point in the same order:
    the ids the markers reserved at placement -- the k-th marker's object
    is written as ``reserved_ids[k]``, not ``next_label + rank``."""
    if reserved_ids is not None:
        reserved_ids = [int(i) for i in reserved_ids]
        if len(reserved_ids) != len(list(click_points or ())):
            raise ValueError(
                "reserved_ids has {} ids for {} click points".format(
                    len(reserved_ids), len(list(click_points or ()))))
    from .reshape.engine import ReshapeCancelled

    params = dict(params or {})
    out = AddPreviewResult(roi=roi, preset=str(preset))
    if (reserved_ids and id_max is not None
            and max(reserved_ids) > int(id_max)):
        # the dtype guard BEFORE the engine runs (B-L2): a reserved id that
        # cannot be written refuses at once, not after the fill
        out.status = "refused"
        out.refusal = ADD_S1_IDS_FULL
        return out
    try:
        if preset == "nuclei":
            return _run_nuclei(out, roi=roi, existing_roi=existing_roi,
                               next_label=next_label, spacing=spacing,
                               params=params, image_roi=image_roi,
                               id_max=id_max, cancel_event=cancel_event,
                               progress=progress)
        return _run_cells(out, roi=roi, existing_roi=existing_roi,
                          next_label=next_label, spacing=spacing,
                          params=params, nuclei_roi=nuclei_roi,
                          wall_roi=wall_roi, dark_rois=dark_rois,
                          run_guide_roi=run_guide_roi,
                          click_points=click_points,
                          reserved_ids=reserved_ids,
                          dark_threshold=dark_threshold, id_max=id_max,
                          cancel_event=cancel_event, progress=progress)
    except ReshapeCancelled:
        out.status = "cancelled"
        out.result = None
        return out


def _run_nuclei(out, *, roi, existing_roi, next_label, spacing, params,
                image_roi, id_max, cancel_event, progress):
    from .add_segment import add_segment_region

    step = _Steps(progress, ("add_box", "add_find", "add_smooth"))
    step("add_box")
    if image_roi is None:
        raise ValueError("the Nuclei preset needs image_roi")
    if _cancelled(cancel_event):
        out.status = "cancelled"
        return out
    step("add_find")
    result = add_segment_region(
        image_roi, existing_roi,
        clahe=bool(params.get("clahe", False)),
        clahe_clip=float(params.get("clahe_clip", 0.01)),
        threshold=params.get("threshold", "otsu"),
        min_size=int(params.get("min_size", 0)),
        separate=bool(params.get("separate", True)),
        min_distance=int(params.get("min_distance", 3)),
        next_label=int(next_label),
        spacing=spacing,
    )
    n = int(len(np.unique(result)) - (1 if (result == 0).any() else 0))
    if n == 0:
        out.status = "refused"
        out.refusal = ADD_S1_NOTHING_NUCLEI
        return out
    if not _ids_fit(next_label, n, id_max):
        out.status = "refused"
        out.refusal = ADD_S1_IDS_FULL
        return out
    if _cancelled(cancel_event):
        out.status = "cancelled"
        return out
    _finish(out, result, params, cancel_event, step)
    if out.status == "ok" and not out.ids:
        out.status = "refused"
        out.refusal = ADD_S1_ALL_FILTERED
        out.result = None
    return out


def _mint(result, next_label):
    """Engine ids ``1..N`` (some absent) -> ``next_label + rank`` in marker
    order.  Returns ``(minted, mapping)``."""
    present = [int(v) for v in np.unique(result) if v]
    if not present:
        return np.zeros(result.shape, np.int32), {}
    top = int(next_label) + len(present) - 1
    dtype = np.int32 if top <= np.iinfo(np.int32).max else np.int64
    lut = np.zeros(int(result.max()) + 1, dtype)
    mapping = {}
    for rank, old in enumerate(present):
        lut[old] = int(next_label) + rank
        mapping[old] = int(next_label) + rank
    return lut[result], mapping


def _reserved_mapping(seed_points, reserved_ids, existing_roi, *, radius_um,
                      spacing):
    """Engine id -> reserved id, computed the ENGINE's way:
    ``rasterise_points`` of the points, clipped to empty space, relabelled
    ascending -- so a marker whose ball vanished (an earlier ball kept every
    voxel, or a kept object covers it) is skipped instead of shifting every
    later id.  Returns ``(mapping, absorbed_reserved_ids)``."""
    from .reshape.region import rasterise_points

    existing = np.asarray(existing_roi)
    balls = rasterise_points(seed_points, existing.shape, radius_um=radius_um,
                             spacing=spacing)
    balls = np.where(existing == 0, balls, 0)
    survivors = [int(v) for v in np.unique(balls) if v]
    kept = set(survivors)
    mapping = {j: reserved_ids[placement - 1]
               for j, placement in enumerate(survivors, start=1)}
    absorbed = [reserved_ids[i] for i in range(len(reserved_ids))
                if (i + 1) not in kept]
    return mapping, absorbed


def _mint_to(result, mapping):
    """Engine ids -> the given final ids (the reserved-id twin of _mint)."""
    top = max([int(v) for v in mapping.values()] or [0])
    dtype = np.int32 if top <= np.iinfo(np.int32).max else np.int64
    size = max(int(result.max()) if result.size else 0,
               max([int(k) for k in mapping] or [0])) + 1
    lut = np.zeros(size, dtype)
    for old, new in mapping.items():
        lut[int(old)] = int(new)
    return lut[result]


def _run_cells(out, *, roi, existing_roi, next_label, spacing, params,
               nuclei_roi, wall_roi, dark_rois, run_guide_roi, click_points,
               dark_threshold, id_max, cancel_event, progress,
               reserved_ids=None):
    from .reshape import ReshapeSpec
    from .reshape.region import reshape_empty_region

    source = params.get("source", SOURCE_CLICKS)
    out.source = source
    outside = "faces+dark" if params.get("dark_outside", True) else "faces"
    budget = int(params.get("box_budget", ADD_BOX_BUDGET))
    spec = ReshapeSpec(
        target="region", seeds=source,
        walls=params.get("wall_kind", "membrane"),
        solver="watershed",
        params={
            "wall_channel": "membrane",
            "wall_preproc": params.get("wall_chain", "zcorrect") or "none",
            "ridge": False,
            "outside": outside,
            "dark_otsu_factor": float(params.get("dark_otsu_factor", 0.5)),
            "dark_clearance_um": float(params.get("dark_clearance_um", 2.0)),
            "core_erode_um": float(params.get("core_erode_um", 1.0)),
            "click_radius_um": float(params.get("click_radius_um", 2.0)),
            "box_budget": budget,
        },
    )
    starts = tuple(int(sl.start or 0) for sl in roi)
    seed_layer = None
    seed_points = None
    if source == SOURCE_NUCLEI:
        from .add_segment import add_segment_region

        if nuclei_roi is None:
            raise ValueError("the nuclei-masks source needs nuclei_roi")
        if _cancelled(cancel_event):
            out.status = "cancelled"
            return out
        if progress is not None:
            try:
                # the nuclei are this source's MARKERS: "Finding markers...",
                # not the Nuclei preset's "Finding bright
                # objects..." (live-shot fix V6)
                progress("add_markers", 0, 4)
            except Exception:
                pass
        seed_layer = add_segment_region(
            nuclei_roi, existing_roi, clahe=False, clahe_clip=0.01,
            threshold=params.get("nuclei_threshold", "otsu"),
            min_size=0, separate=True, min_distance=3, next_label=1,
            spacing=spacing)
    elif source == SOURCE_RUN:
        if run_guide_roi is None:
            raise ValueError("the run source needs run_guide_roi")
        seed_layer = run_guide_roi
    else:
        seed_points = [tuple(int(p[a]) - starts[a] for a in range(3))
                       for p in (click_points or ())]
    if _cancelled(cancel_event):
        out.status = "cancelled"
        return out
    res = reshape_empty_region(
        existing_roi, {"membrane": wall_roi}, spec, roi=roi,
        dark_channels=(dict(dark_rois or {}) if outside == "faces+dark" else None),
        seed_layer=seed_layer, seed_points=seed_points,
        dark_threshold=dark_threshold, spacing=spacing,
        cancel_event=cancel_event, progress=progress)
    out.messages = list(res.messages or [])
    for key in ("n_dropped_dark", "n_dropped_thin", "n_nuclei_slivers",
                "n_markers_over_kept", "n_fragments_dropped"):
        setattr(out, key, int(getattr(res, key, 0) or 0))
    out.unfilled_tissue_fraction = res.unfilled_tissue_fraction
    if res.status != "ok":
        out.status = "refused"
        out.refusal = (too_big_sentence(budget) if res.refusal_reason == "too-big"
                       else ADD_ENGINE_NO_MARKERS)
        return out
    out.elevation = res.elevation
    present = [int(v) for v in np.unique(res.result) if v]
    if seed_points is not None and reserved_ids is not None:
        # every click keeps the id it reserved at placement; the
        # dtype guard reads the largest reserved id
        if (id_max is not None and reserved_ids
                and max(reserved_ids) > int(id_max)):
            out.status = "refused"
            out.refusal = ADD_S1_IDS_FULL
            return out
        mapping, out.absorbed_ids = _reserved_mapping(
            seed_points, reserved_ids, existing_roi,
            radius_um=float(params.get("click_radius_um", 2.0)),
            spacing=spacing)
        if out.absorbed_ids:
            out.messages.append(
                "click markers absorbed (their ball was already taken): "
                "{}".format(out.absorbed_ids))
        result = _mint_to(res.result, mapping)
    else:
        if not _ids_fit(next_label, len(present), id_max):
            out.status = "refused"
            out.refusal = ADD_S1_IDS_FULL
            return out
        result, mapping = _mint(res.result, next_label)
    out.faint_pairs = [(mapping.get(int(a), int(a)), mapping.get(int(b), int(b)),
                        float(m)) for a, b, m in (res.faint_pairs or [])]
    min_size = int(params.get("min_size", 0) or 0)
    if min_size > 0:
        ids, sizes = _sizes_of(result)
        small = [i for i in ids if sizes[i] < min_size]
        if small:
            result = np.where(np.isin(result, small), 0, result).astype(result.dtype)
            out.n_below_min_size = len(small)
    if _cancelled(cancel_event):
        out.status = "cancelled"
        return out
    step = _Steps(progress, ("add_smooth",))
    _finish(out, result, params, cancel_event, step)
    if out.status == "ok" and not out.ids:
        out.status = "refused"
        out.refusal = ADD_S1_ALL_FILTERED
        out.result = None
    return out


@dataclass
class AddBoxInfo:
    """The box-info answer."""

    empty_share: float | None = None
    dark_threshold: dict | None = None
    run_count: tuple | None = None      # (n_in_box, n_markers)
    nuclei_auto: float | None = None
    over_limit: bool = False
    status: str = "ok"                  # ok / cancelled
    #: B4.2 -- the Nuclei preset's Brightness card, measured here and not
    #: on the main thread: ``{"counts": [...], "edges": [...]}``
    #: of the box crop in the threshold's value space, and its Otsu
    histogram: dict | None = None
    brightness_auto: float | None = None


def brightness_histogram(image_roi, *, clahe=False, clahe_clip=0.01,
                         n_bins=None, sample_voxels=BRIGHTNESS_SAMPLE_VOXELS):
    """``(histogram, auto)`` of the Nuclei preset's Brightness card: the
    box crop in the value space the threshold is compared in
    (``add_threshold_space``), binned (``{"counts", "edges"}``), and its
    Otsu -- ``None`` for a flat crop.  Pure; runs on the box-info worker
    (Add rebuild B4.2: it cost 169-928 ms per tick on the main thread).

    A crop larger than ``sample_voxels`` is measured on its
    ``[::s, ::s, ::s]`` sample (the smallest stride that fits, as the dark
    rule's ``dark_stride``) -- final fix pass B5.  ``None`` / ``0`` reads
    the whole crop."""
    from .add_segment import _resolve_threshold, add_threshold_space

    image_roi = np.asarray(image_roi)
    if (sample_voxels and image_roi.ndim > 0
            and int(image_roi.size) > int(sample_voxels)):
        s = dark_stride(image_roi.shape, int(sample_voxels))
        if s > 1:
            image_roi = image_roi[(slice(None, None, s),) * image_roi.ndim]
    proc = add_threshold_space(image_roi, clahe=bool(clahe),
                               clahe_clip=float(clahe_clip))
    proc = np.asarray(proc, dtype=np.float64)
    finite = proc[np.isfinite(proc)]
    if finite.size == 0:
        return {"counts": [], "edges": []}, None
    counts, edges = np.histogram(finite, bins=int(n_bins or ADD_HIST_BINS))
    auto = _resolve_threshold(proc, "otsu")
    return ({"counts": counts.tolist(), "edges": edges.tolist()},
            None if auto is None else float(auto))


def add_box_info(*, existing_roi, dark_sources, dark_rois, spacing, stride,
                 dark_cache, source, run_guide_roi, nuclei_roi, params,
                 roi=None, image_roi=None, cancel_event=None) -> AddBoxInfo:
    """The one box-info job.  ``dark_sources`` maps each channel name
    to ONE level's 3D array of the WHOLE patch (only ``[::stride]`` is
    read, here, on the worker); ``dark_cache`` is the cached threshold or
    ``None``.  ``existing_roi`` / ``dark_rois`` / ``run_guide_roi`` /
    ``nuclei_roi`` / ``image_roi`` are box crops the main thread copied.
    On the Nuclei preset ``image_roi`` (the Brightness channel's crop)
    yields the histogram and its Auto (B4.2)."""
    from .reshape.engine import ReshapeCancelled

    params = dict(params or {})
    info = AddBoxInfo()
    try:
        existing = np.asarray(existing_roi)
        total = int(existing.size)
        info.empty_share = (float(np.count_nonzero(existing == 0)) / total
                            if total else 0.0)
        if _cancelled(cancel_event):
            info.status = "cancelled"
            return info
        budget = int(params.get("box_budget", ADD_BOX_BUDGET))
        preset = params.get("preset", "nuclei")
        if preset != "cells" and image_roi is not None:
            info.histogram, info.brightness_auto = brightness_histogram(
                image_roi, clahe=params.get("clahe", False),
                clahe_clip=params.get("clahe_clip", 0.01))
            if _cancelled(cancel_event):
                info.status = "cancelled"
                return info
        info.over_limit = preset == "cells" and total > budget
        outside = "faces+dark" if params.get("dark_outside", True) else "faces"
        info.dark_threshold = dark_cache
        if info.dark_threshold is None and dark_sources and preset == "cells":
            from .reshape.region import patch_dark_threshold

            info.dark_threshold = patch_dark_threshold(
                dict(dark_sources), spacing=spacing, stride=max(1, int(stride)))
        if _cancelled(cancel_event):
            info.status = "cancelled"
            return info
        if (preset == "cells" and source == SOURCE_NUCLEI
                and nuclei_roi is not None):
            from .add_segment import _resolve_threshold

            thr = _resolve_threshold(np.asarray(nuclei_roi, np.float64), "otsu")
            info.nuclei_auto = None if thr is None else float(thr)
        if (preset == "cells" and source == SOURCE_RUN
                and run_guide_roi is not None and not info.over_limit
                and (outside == "faces" or info.dark_threshold is not None)):
            from .reshape import ReshapeSpec
            from .reshape.region import find_region_seeds

            if roi is None:
                roi = tuple(slice(0, int(n)) for n in existing.shape)
            spec = ReshapeSpec(
                target="region", seeds=SOURCE_RUN,
                walls=params.get("wall_kind", "membrane"),
                params={"wall_channel": "membrane", "ridge": False,
                        "outside": outside,
                        "dark_otsu_factor": float(params.get("dark_otsu_factor", 0.5)),
                        "dark_clearance_um": float(params.get("dark_clearance_um", 2.0)),
                        "core_erode_um": float(params.get("core_erode_um", 1.0)),
                        "box_budget": budget})
            rep = find_region_seeds(
                existing, {}, spec, roi=roi,
                dark_channels=(dict(dark_rois or {})
                               if outside == "faces+dark" else None),
                seed_layer=run_guide_roi, dark_threshold=info.dark_threshold,
                spacing=spacing, cancel_event=cancel_event)
            info.run_count = (int(rep.n_found), int(rep.n_in_region))
    except ReshapeCancelled:
        info.status = "cancelled"
    return info
