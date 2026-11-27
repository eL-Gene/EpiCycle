"""The Add tab's worker functions and every sentence it shows (Add rebuild).

Pure, napari-free, Qt-free.  Two functions run on a worker thread through
``RefineWorker``:

* :func:`run_add_preview` -- one Preview of the box: the **Nuclei** preset
  (today's ``add_segment_region`` at the user's settings, byte-identical
  except ids; or, with Look for nuclei ▸ Inside each cell of another run,
  ``add_derive.nuclei_in_cells`` on that run's cells) or the **Cells**
  preset (the engine's region target,
  ``reshape.region.reshape_empty_region``, with the Sato ridge OFF).  It
  mints the FINAL ids (``next_label + rank``), refuses when the label
  dtype has no room, drops objects under Smallest object, flags the edge
  and small objects, and smooths.  A Stop inside the engine RAISES
  ``ReshapeCancelled``; it is caught here and answered as
  ``status="cancelled"`` -- never an ``error`` with a blank message.
* :func:`add_box_info` -- the one box-info job: the empty share, the
  patch's dark threshold (a strided read of ONE level of each channel), the
  run-count line and the Markers Auto value.
* :func:`clear_plan` -- Clear objects' count: which objects a Clear takes
  (those mostly inside the box, or every object in the patch), how many of
  them reach outside the box, and the box around them all.  It never
  writes; the main thread writes after the user confirms.

Every user-facing sentence is a named constant here; the panel and the
orchestrator import them.  Raw exception text never reaches the page.

Nuclei ▸ At the markers I place on a WHOLE PATCH with "Update as I click"
on runs through :mod:`add_incremental` (the patch prepared once, each click
solving only the bright pieces whose markers changed); with the tick off
it runs through :func:`add_segment.segment_at_markers` as it always did.
The two agree voxel for voxel except inside a nucleus holding several
seeds, where a few voxels at a tie between equal-height seeds can go to
the other seed's object -- so turning the tick on or off can move those
few voxels.  A box always runs the one-go way.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from time import perf_counter as _perf_counter

import numpy as np

from .add_segment import ADD_MARKER_SPACING_DEFAULT  # noqa: F401
from .reshape import region as _region
from .reshape._common import sample_stride
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
    "ClearPlan", "clear_plan",
    "count_phrase", "small_piece_ids", "counts_note", "s1_sentence",
    "preview_block", "dark_stride", "box_line",
]

# --------------------------------------------------------------------------- #
# presets and sources (read BY INDEX by the panel; new items APPEND)
# --------------------------------------------------------------------------- #
ADD_PRESETS = ("nuclei", "cells")
ADD_PRESET_LABELS = ("Nuclei", "Cells with a membrane stain")
ADD_PRESET_TIPS = (
    "Finds bright objects by brightness, or the nucleus inside each cell "
    "of another run. Best for a nuclear stain.",
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
    "Objects you keep never lose a voxel, and nothing is added until you "
    "press Accept."
)
ADD_ABOUT_BODY = (
    "<b>Clear objects</b> is the one exception: it removes objects at once, "
    "and only the main Undo button brings them back. "
    "<b>Nuclei</b> finds bright objects by brightness, or the nucleus "
    "inside each cell of another run, and suits a nuclear stain. "
    "<b>Cells with a membrane stain</b> grows one object from "
    "each marker until it meets a wall in the membrane channel or dark "
    "space; markers come from another run's objects, from nuclei in a "
    "second channel, or from your clicks, and it measures in microns, from "
    "Step 3's Spacing. New "
    "objects are written only where there is no object yet, so clear the "
    "objects you do not want first. Objects touching the box edge "
    "and very "
    "small objects start unticked."
)


def voxel_limit_words(n) -> str:
    """A voxel limit in words, as the Cells size lines say theirs ("the
    limit for cells is 100 million"): whole or tenth millions as
    "1 million" / "1.5 million"; under a million, the number with
    thousands separators ("1,000")."""
    n = int(n)
    if n >= 1_000_000:
        return "{:g} million".format(round(n / 1e6, 1))
    return "{:,}".format(n)


#: On Cells, S, M and L and a new box reach at most this many voxels: their
#: SIZE, not a limit (a box made larger with Numbers... is kept, and filled
#: up to ``ADD_BOX_BUDGET``).  It keeps the presets the size they were
#: when it was the Cells limit.
ADD_CELLS_PRESET_MAX_VOXELS = 4_000_000

# --------------------------------------------------------------------------- #
# S1
# --------------------------------------------------------------------------- #
#: no box, on both presets: Whole patch is never grey (Clear objects is
#: named once a box is up, when it is on screen beside Preview)
ADD_S1_IDLE = "Place a box over the space to fill, or press Whole patch."
#: the same on Cells where Spread from each marker cannot run (no numba):
#: a Cells whole patch cannot be filled there, so it is not offered
ADD_S1_IDLE_BOX = "Place a box over the space to fill."
#: ``{k}``: the Cells limit in words (:func:`voxel_limit_words` of
#: ``ADD_BOX_BUDGET``, "100 million"), the same for a box and a whole patch
ADD_S1_BOX = (
    "Box {z} × {y} × {x} voxels ({m} million; the limit for cells is {k}) "
    "· {p} % of it is empty space"
)
ADD_S1_BOX_NUCLEI = (
    "Box {z} × {y} × {x} voxels ({m} million) · {p} % of it is empty space"
)
#: the ready line counts objects (the markers of one object count once)
ADD_S1_BOX_CLICKS = "Box {z} × {y} × {x} voxels · {n} objects marked"
ADD_S1_BOX_CLICKS_ONE = "Box {z} × {y} × {x} voxels · 1 object marked"
#: the same on a whole patch (markers are placed there too)
ADD_S1_PATCH_CLICKS = (
    "Whole patch {z} × {y} × {x} voxels · {n} objects marked")
ADD_S1_PATCH_CLICKS_ONE = (
    "Whole patch {z} × {y} × {x} voxels · 1 object marked")
ADD_S1_MEASURING = "Measuring the box…"
#: the same on a whole patch
ADD_S1_MEASURING_PATCH = "Measuring the patch…"
#: ``{k}`` as in ``ADD_S1_BOX`` (see :func:`over_limit_line`)
ADD_S1_OVER_LIMIT = (
    "This box is {m} million voxels; the limit for cells is {k}. "
    "Shrink it with Numbers… or pick S, M or L."
)
#: Step 3's physical units are the patch's own voxel size (Set spacing… on
#: the Session list, Step 1's registry or the file), with QC Settings'
#: spacing only as the fallback -- the same source Reshape, Refine and the
#: QC features read.  The sentences name Set spacing…: it keeps a value per
#: patch (1 / 1 / 1 too, which QC Settings cannot tell from its factory
#: default), and it stays live while an Add box holds the Step 3 lock (it
#: is greyed only while Predict or Reconstruct runs).  Every Add path --
#: Cells on a box or the whole patch, the Nuclei markers, Inside each cell
#: -- uses that spacing AS IT IS and never refuses (the 1.0 um isotropic
#: fallback runs) -- ``ADD_S1_NO_VOXEL_SIZE``, ``ADD_UM_OFF`` and the
#: Inside / whole-patch spacing refusals are retired; at the default
#: (1, 1, 1) this AMBER, non-blocking note sits under S1 instead (Wound: a
#: wrong 1 um took the leak from 0.002-0.017 to 0.05-0.21).
ADD_NOTE_DEFAULT_SPACING = (
    "Spacing is the default 1.0 µm here — if your voxels are smaller, set "
    "their size with Set spacing… in the Session list, so the µm knobs mean "
    "what they say."
)
#: the same note while a box is placed (A-M2, the A1-A4 audit): once it
#: named QC Settings, which the box's lock greys; Set spacing… stays live
#: with a box up, so the two notes are one sentence now
ADD_NOTE_DEFAULT_SPACING_BOXED = ADD_NOTE_DEFAULT_SPACING
#: the owner's live test (2026-09-27 20:33, D-1): with "Treat dark space as
#: outside" OFF on Cells, nothing but a membrane wall or a box face stops the
#: fill -- the amber pre-result note under S1, the spacing note's rules
ADD_NOTE_DARK_OFF_LEAK = (
    "Dark space is not treated as outside, so objects run to the box faces "
    "wherever there is no wall — tick it in Advanced if objects come out "
    "open at the top or bottom."
)
#: the same note on a whole patch (no face is a box face; only the
#: background stops a cell in open space)
ADD_NOTE_DARK_OFF_LEAK_WHOLE = (
    "Dark space is not treated as outside and there is no tissue outline, "
    "so cells grow a few µm into open background — tick it in Advanced, or "
    "pick a Tissue outline under Walls."
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
#: the same row while Place markers is already on (never ask to turn on
#: what is on)
ADD_S1_NO_MARKERS_ON = (
    "Place at least one marker: click inside the box in a side view."
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
#: Shift+click adds a seed to the current object; a seed whose bright
#: piece does not touch the object's own makes nothing (its marker stays,
#: Undo last removes it).  Said once, when the result that refused it lands
ADD_S1_NOT_CONNECTED = (
    "That spot is not touching #{id} — click it plainly for a new object, "
    "or extend from a spot that touches."
)
# --- extending an object: the current one, the rows, the hint ------------
#: the Markers card's hint line, in place of the click help, at every
#: marker count: the gestures of a click in a side view and Ctrl+Z (the
#: same words on Nuclei and on Cells; two lines at the dock's width).  The
#: current object is the bold row; Esc is in Place markers' tooltip
ADD_HELP_CLICK_KEYS = (
    "Click: a new object · Shift+click: add to the bold one · "
    "Alt+click it or its row: make it bold · Ctrl+click: remove · "
    "Ctrl+Z: undo last"
)
#: a row's seed count, for an object grown from several markers
ADD_ROW_SEEDS = "{n} seeds"
#: a row's status words (drawn as chips, before edge / small), one per
#: kind of seed that made nothing at the last result: a seed whose bright
#: piece (or cell) does not touch its object's own, a seed off the stain, a
#: seed inside a kept object.  Short, so they fit a row beside its seed
#: count and size; the long form is in the row's tooltip
ADD_ROW_NOT_CONNECTED = "apart"
ADD_ROW_DIM = "dim"
ADD_ROW_KEPT = "on kept"
ADD_ROW_STATUS_LONG = {
    ADD_ROW_NOT_CONNECTED: "not connected",
    ADD_ROW_DIM: "nothing bright here",
    ADD_ROW_KEPT: "inside a kept object",
}
#: the row tooltip's line for its status words
ADD_ROW_STATUS_TIP = "Seeds that made nothing: {words}."
#: the ball layer's value for a refused (not connected) seed: drawn hollow,
#: in its own red (the 49 object colours are 1-49)
ADD_BALL_APART_VALUE = 50
ADD_BALL_APART_RGBA = (0.95, 0.15, 0.15, 1.0)
#: Every marker unticked (Preview grey)
ADD_S1_TICK_ONE_MARKER = (
    "Tick at least one object to grow from — all {n} are unticked."
)
ADD_S1_TICK_ONE_MARKER_ONE = (
    "Tick the object to grow from — it is unticked."
)
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
#: Nuclei ▸ At the markers I place: every marker unticked (Preview grey)
ADD_S1_TICK_ONE_NUCLEUS = (
    "Tick at least one object to find its nucleus — all {n} are unticked."
)
ADD_S1_TICK_ONE_NUCLEUS_ONE = (
    "Tick the object to find its nucleus — it is unticked."
)
#: ... the Preview found nothing and at least one marker's whole ball lay
#: off the stain (below the threshold)
ADD_S1_MARKERS_NONE_ON_STAIN = (
    "No marker sits on stain above the threshold — lower Threshold, or "
    "move the markers onto the nuclei."
)
#: ... the Preview found nothing and every marker's ball lay on kept objects
ADD_S1_MARKERS_ALL_KEPT = (
    "Every marker now sits inside a kept object — remove them and click in "
    "empty space."
)
ADD_S1_MARKERS_ALL_KEPT_ONE = (
    "The marker now sits inside a kept object — remove it and click in "
    "empty space."
)
#: the counts note's part for markers whose whole ball lay off the stain
ADD_NOTE_MARKERS_DIM = (
    "{n} markers sat on stain below the threshold and made nothing"
)
ADD_NOTE_MARKERS_DIM_ONE = (
    "1 marker sat on stain below the threshold and made nothing"
)
#: the marker row's tooltip after a Preview in which it made nothing
ADD_MARKER_DIM_TIP = (
    "At the last Preview this marker sat on stain below the threshold, so "
    "it made nothing — move it onto the nucleus or lower Threshold under "
    "Brightness."
)
ADD_MARKER_SMALL_TIP = (
    "At the last Preview this marker's nucleus was below Smallest object or "
    "vanished when smoothed."
)
#: a marker whose nucleus the cut to the tissue outline emptied
ADD_MARKER_OUTLINE_TIP = (
    "At the last Preview this marker's nucleus lay outside the tissue "
    "outline and was left out."
)
#: after ADD_S1_BOX_CLICKS when k < n (k >= 1 there, its own singular)
ADD_S1_TICKED_TAIL = " · {k} of {n} ticked"
ADD_S1_WORKING = "Working on the box…"
ADD_S1_WORKING_PATCH = "Working on the patch…"
ADD_S1_BUSY_REFUSED = "Wait for the preview to finish, or press Stop."

# --- "Update as I click" (the markers run after every change) --------------
#: the quiet after a marker or a setting change before one run starts
ADD_LIVE_DEBOUNCE_MS = 150
#: the tick under Advanced, and its help line
ADD_LIVE_LABEL = "Update as I click"
ADD_HELP_LIVE_TICK = (
    "Finds the objects again after every click and setting change, so you "
    "see the result without pressing Preview. Preview still works.")
ADD_HELP_LIVE = ADD_HELP_LIVE_TICK
#: the tick beside it (Cells ▸ Click markers with "Update as I click" on;
#: off by default, never kept between napari sessions): the live
#: restrictions below switched off -- the click fill runs where its flat
#: spots would step aside, and where it cannot run at all the region is
#: filled whole a moment after the clicks settle
ADD_LIVE_ALWAYS_LABEL = "Always update as I click"
ADD_HELP_LIVE_ALWAYS = (
    "Runs the quick fill even where its flat spots could place a few tied "
    "voxels differently from Preview, and where the quick fill cannot run "
    "at all, redoes the region a moment after you stop clicking. That may "
    "take minutes on a large region or a slow Wall type; the first line "
    "says how long.")
#: the quiet before that whole fill (longer than a click's: a burst of
#: clicks makes one slow run, not several)
ADD_LIVE_SLOW_DEBOUNCE_MS = 700
#: S1 (busy) while that whole fill runs; nothing greys, Stop works, and a
#: click meanwhile queues one more run.  The time is estimated from the
#: region's voxels, its walls and Grow by (:func:`live_update_seconds`,
#: worded by :func:`live_updating_line`): under
#: ``LIVE_FEW_SECONDS`` (or with no estimate) this sentence ...
ADD_S1_LIVE_UPDATING = (
    "Updating the region — this takes a few seconds here.")
#: ... and from there the estimate, ``{t}`` "about 35 s" / "about a
#: minute" / "about 3 minutes"
ADD_S1_LIVE_UPDATING_TIME = "Updating the region — {t} here."
#: under this many seconds the slow update says "a few seconds"
LIVE_FEW_SECONDS = 10.0
#: Cells ▸ Click markers runs as you click on a box or a whole patch of at
#: most this many voxels (equality is live) -- ONE rule for any region,
#: since the box and the whole patch share the same click fill.  The click
#: fill holds its prepared region in memory (about 40 bytes a voxel, about
#: 66 while it is prepared), and its first flood -- the first click on a
#: region -- cannot be stopped.  The number is the slow-patch threshold.
#: Above it the balls appear and Preview is the way (a box keeps its plain
#: flood with closed faces: its edge cuts through tissue; only a whole
#: patch's faces are the end of the data).
ADD_LIVE_CELLS_MAX_VOXELS = 20_000_000
#: A live click the click fill cannot take (its flat spots, the flood not
#: compiled, a failure, or -- on a box -- walls or a Grow by it does not
#: redo) still runs the full fill when the region (a box or a whole patch)
#: holds at most this many voxels: a full run on a live click is allowed up
#: to this size (measured 0.5-0.6 s at 0.66 million voxels, 5.8-6.8 s at
#: 5.2 million).  Above it the click steps aside -- nothing runs, S1 says
#: to press Preview -- since that flood cannot be stopped; a flood still
#: being compiled is not a step aside there: the click waits for it
#: (:data:`LIVE_PREPARING`).
ADD_LIVE_FULL_RUN_MAX_VOXELS = 1_000_000
#: S1 row 13.5 with the tick on for Cells ▸ Click markers: one family, the
#: region its only difference.  Over ``ADD_LIVE_CELLS_MAX_VOXELS`` the
#: limit is named ({n}: the limit in words, see :func:`voxel_limit_words`
#: -- "20 million") ...
ADD_S1_LIVE_OVER_GATE_BOX = (
    "Press Preview to see the cells — this box is over {n} voxels.")
ADD_S1_LIVE_OVER_GATE_PATCH_SIZE = (
    "Press Preview to see the cells — this patch is over {n} voxels.")
#: ... and under it, a region not filled as you click for another reason:
#: on a whole patch, walls the click fill does not redo (only Membrane
#: sheets and Intensity edges are); on a box over
#: ``ADD_LIVE_FULL_RUN_MAX_VOXELS``, those walls or a Grow by other than
#: Fill to the walls; on either region over that size, one the click fill
#: could not take (its flat spots, switched off, a failure) -- the
#: console line says which.  Nuclei ▸ the markers on a whole patch
#: always are.
ADD_S1_LIVE_OVER_GATE_PATCH = (
    "Press Preview to see the cells — this patch is not updated as you "
    "click.")
ADD_S1_LIVE_STEPPED_ASIDE_BOX = (
    "Press Preview to see the cells — this box is not updated as you "
    "click.")
#: ... and the same step aside, for the click fill's flat spots, while the
#: spacing was never set: the tissue outline's smallest piece is a volume
#: in µm³, so at the 1.0 µm fallback (larger voxels than most stacks')
#: debris in the dark around the tissue can count as tissue, and its flat
#: spots stop the click fill where the true spacing would not
ADD_S1_LIVE_STEPPED_ASIDE_SPACING_BOX = (
    "Press Preview to see the cells — at the default 1.0 µm spacing this "
    "box is not updated as you click; Set spacing… in the Session list may "
    "change that.")
ADD_S1_LIVE_STEPPED_ASIDE_SPACING_PATCH = (
    "Press Preview to see the cells — at the default 1.0 µm spacing this "
    "patch is not updated as you click; Set spacing… in the Session list "
    "may change that.")
#: ... and on a box or a whole patch over ``ADD_LIVE_FULL_RUN_MAX_VOXELS``
#: that IS run as you click, while the click fill is still being compiled
#: (once per process, a few seconds after the first box): the clicks wait
#: and are drawn once it is ready -- not a step aside (Preview still works,
#: and is not named)
ADD_S1_LIVE_PREPARING = (
    "Preparing the fill — your clicks will be drawn in a moment.")
#: the grey line directly under "Update as I click" (Cells ▸ Click
#: markers) while S1 row 13.5 holds: its reason in at most 60 characters,
#: naming the control that changes it by its word on the page.  One
#: decision for both rows (:func:`live_row_reason`, read by
#: :func:`s1_sentence` and :func:`live_tick_note`).  Hidden while clicks
#: run as you click, with the tick off, and while "Always update as I
#: click" is on (it lifts every one of these).  Over the live limit
#: (``{n}`` as S1 names it) ...
ADD_LIVE_NOTE_OVER = "Not live over {n} — tick Always update as I click."
#: ... walls the quick fill does not redo (only Membrane sheets and
#: Intensity edges are) ...
ADD_LIVE_NOTE_WALLS = (
    "Not live here: Wall type — tick Always update as I click.")
#: ... Grow by on Spread from each marker ...
ADD_LIVE_NOTE_GROW = "Not live here: Grow by — tick Always update as I click."
#: ... a step aside for the quick fill's flat spots at a set spacing ...
ADD_LIVE_NOTE_FLAT = (
    "Not live here: flat spots — tick Always update as I click.")
#: ... and at the never-set spacing (S1 names Set spacing… too) ...
ADD_LIVE_NOTE_SPACING = (
    "Not live here: default spacing — Set spacing… may help.")
#: ... a quick fill that could not take the region for another reason
#: (it failed, or it is switched off; the console says which) ...
ADD_LIVE_NOTE_OFF = (
    "Not live here: the quick fill could not run — press Preview.")
#: ... and the quick fill still being prepared (the clicks wait)
ADD_LIVE_NOTE_PREPARING = "Preparing the fill…"
#: Accept pressed before the last change is drawn: it waits for it (the
#: button reads ADD_ACCEPT_WAITING_BTN meanwhile)
ADD_S1_ACCEPT_WAITING = "Accepting once the last click is drawn."
ADD_ACCEPT_WAITING_BTN = "Accepting…"
#: a marker click while Accept waits
ADD_S1_ACCEPT_WAIT_CLICK = (
    "Wait — Accept is finishing; place more markers after it.")
#: the wait ended without a drawn result (a refusal, an error, Stop, a
#: change that runs nothing, Discard, the tick off)
ADD_S1_ACCEPT_WAIT_OFF = (
    "Accept was cancelled — the result changed before it was drawn; press "
    "Accept again.")
#: Clear objects pressed while the last click is still being drawn
ADD_S1_CLEAR_WAIT_LIVE = (
    "Wait — the last click is still being drawn; Clear once it appears.")
#: a Preview while an earlier preview's thread still runs in the background
#: (a Cancel or Done could not wait for it): Stop is grey then, so it is not
#: named; the sentence goes by itself when that thread ends
ADD_S1_PARKED_REFUSED = (
    "An earlier preview is still finishing in the background. Wait for it "
    "to end, then press Preview again."
)
ADD_S1_STOPPED = "Stopped. Nothing from this box was added."
ADD_S1_STOPPED_PATCH = "Stopped. Nothing from this patch was added."
ADD_S1_FAILED = (
    "The preview failed — the reason is in the log. Nothing from this box "
    "was added."
)
ADD_S1_FAILED_PATCH = (
    "The preview failed — the reason is in the log. Nothing from this "
    "patch was added."
)
#: painted on S1 when the worker has finished, before the result is drawn
#: on the main thread (a visible moment on a whole patch)
ADD_S1_SHOWING = "Showing the result…"
#: S1 (a transient) when the main Undo (Curation) took back an Accept that
#: the review lists: those rows leave the list; with none left the review
#: closes
ADD_S1_UNDO_TOOK_BACK = (
    "The main Undo took back the last objects added here; they are off the "
    "list."
)
ADD_S1_UNDO_TOOK_BACK_ALL = (
    "The main Undo took back the objects added here, so the review is "
    "closed."
)
#: S1 (a transient) when Commit Triage or a new extraction emptied the main
#: Undo while the review was up: its Accepts can no longer be taken back,
#: so the review closes (the objects stay)
ADD_S1_UNDO_EMPTIED = (
    "Commit Triage or a new extraction made the objects added here "
    "permanent; the review is closed."
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
#: S1 when the user unticked the last row (amber)
ADD_S1_TICK_ONE = "None is ticked — tick the objects to add, or press All."
#: S1's tail when every row STARTED unticked (at the box edge / small): the
#: objects are on screen (faintly in the side views), and nothing is ticked
ADD_S1_NONE_TICKED = (
    " · none ticked ({why}) — press All, or tick the ones to add"
)
ADD_S1_NONE_TICKED_ONE = " · not ticked ({why}) — press All, or tick it"
#: how faint an unticked object is drawn in the three side views (its row
#: swatch's colour at this alpha, times the layer's opacity); the main view
#: does not draw it, since napari's 3D labels drawing stops at the first
#: visible label and a faint shell would hide the ticked objects inside it
ADD_UNTICKED_ALPHA = 0.4
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
#: Preview grey BEFORE the press: the box-info run count (found, markers)
#: left no marker.  The count runs the Preview's own seed finder, so the
#: Preview would refuse; these say why, in what is true for this box.
#: The run has no object here at all:
ADD_S1_RUN_NONE_HERE = (
    "That run has no object in this box — move the box, or pick another run."
)
ADD_S1_RUN_NONE_HERE_PATCH = (
    "That run has no object in this patch — pick another run."
)
#: ... its objects are here, and objects are kept here too: each marker was
#: dropped in dark space, over a kept object, or as too small.  Every line
#: of this family is at most 138 characters (the longest Cells line the
#: page's heights were measured with)
ADD_S1_RUN_NO_MARKERS = (
    "None of that run's objects here can be a marker: each is in dark "
    "space, on a kept object, or too small. Move the box, or Clear objects."
)
ADD_S1_RUN_NO_MARKERS_PATCH = (
    "None of that run's objects can be a marker: each is in dark space, on "
    "an object already here, or too small. Clear objects to make room."
)
#: ... nothing is kept here, so each was dropped in dark space or as too
#: small; ``{dark}`` is the dark-space tick's label (said only while it is
#: on and no tissue outline decides what is outside instead)
ADD_S1_RUN_NO_MARKERS_EMPTY = (
    "None of that run's objects here can be a marker: each is in dark "
    "space or too small. Turn off {dark} under Advanced."
)
ADD_S1_RUN_NO_MARKERS_EMPTY_PATCH = (
    "None of that run's objects can be a marker: each is in dark space or "
    "too small. Turn off {dark} under Advanced."
)
ADD_S1_RUN_NO_MARKERS_EMPTY_PLAIN = (
    "None of that run's objects here can be a marker: each is in dark "
    "space or too small. Move the box."
)
ADD_S1_RUN_NO_MARKERS_EMPTY_PLAIN_PATCH = (
    "None of that run's objects can be a marker: each is in dark space or "
    "too small."
)
#: the Cells Advanced tick that drops markers lying in dark space
ADD_DARK_OUTSIDE_LABEL = "Treat dark space as outside"
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
    "Markers come from nuclei, another run or your clicks; the new cells "
    "follow the membrane."
)
ADD_HELP_NUCLEI = (
    "Finds bright objects by brightness, or the nucleus inside each cell of "
    "another run. Best for a nuclear stain."
)
ADD_HELP_NUCLEI_MASKS = (
    "Each nucleus found in this channel becomes one marker — the same "
    "finder as the Nuclei preset at its default settings. Slivers pressed "
    "against kept objects are skipped; two nuclei that touch can still come "
    "out as one."
)
ADD_AUTO_PENDING = "Measuring the box for Auto…"
#: the Brightness card's help slot with no box: Auto is measured on a box,
#: so with none placed nothing is being measured
ADD_AUTO_NEEDS_BOX = "Auto is measured once a box is placed."
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
ADD_RUN_COUNT_PATCH_PENDING = "Counting the run's objects in the patch…"
ADD_HELP_CLICKS = (
    "While Place markers is on, click inside the box in any side view. "
    "Each click becomes a small ball, about {d} µm across. Ctrl + click "
    "removes one."
)
#: the Markers card's help on Nuclei ▸ At the markers I place, until a
#: marker is placed (never longer than the Cells help it stands in for)
ADD_HELP_NUCLEI_CLICKS = (
    "With Place markers on, click once on each nucleus. The threshold sets "
    "its edge; unclicked nuclei are left out. Ctrl + click removes one."
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
    "space. It also sets the shrink-wrap outline. Lower it if dim cells "
    "come out empty."
)
#: Dark below's tooltip on Nuclei (the shrink-wrap is all it sets there)
ADD_HELP_DARK_BELOW_NUCLEI = (
    "Signal below this share of the patch's usual brightness counts as dark "
    "space; it sets the shrink-wrap outline. Lower it if dim tissue falls "
    "outside the outline."
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

#: the counts note's Inside each cell parts (each with its singular)
ADD_NOTE_CELL_HAS_ONE = "{n} cells already held a nucleus and were skipped"
ADD_NOTE_CELL_HAS_ONE_ONE = "1 cell already held a nucleus and was skipped"
ADD_NOTE_NO_NUCLEUS = "{n} cells had no nucleus"
ADD_NOTE_NO_NUCLEUS_ONE = "1 cell had no nucleus"
ADD_NOTE_SPLIT_BORDER = (
    "{n} nuclei were split between two cells at their border"
)
ADD_NOTE_SPLIT_BORDER_ONE = (
    "1 nucleus was split between two cells at its border"
)
ADD_NOTE_OUTSIDE_CELLS = "{n} nuclei outside every cell were left out"
ADD_NOTE_OUTSIDE_CELLS_ONE = "1 nucleus outside every cell was left out"
#: the counts note's tissue-outline parts (Cells): markers mostly outside
#: the outline, and run objects lying mostly over kept objects (both
#: dropped before the growth)
ADD_NOTE_OUTSIDE_OUTLINE = (
    "{n} markers lay outside the tissue outline and were skipped"
)
ADD_NOTE_OUTSIDE_OUTLINE_ONE = (
    "1 marker lay outside the tissue outline and was skipped"
)
ADD_NOTE_RUN_SLIVERS = (
    "{n} run objects lay mostly over kept objects and were skipped"
)
ADD_NOTE_RUN_SLIVERS_ONE = (
    "1 run object lay mostly over kept objects and was skipped"
)

#: Cells ▸ Nuclei masks on a whole patch: a marker under this many cubic
#: microns is counted in the counts note (kept all the same: no floor yet)
ADD_SMALL_MARKER_UM3 = 10.0
ADD_NOTE_SMALL_MARKERS = "{n} markers were under 10 µm³"
ADD_NOTE_SMALL_MARKERS_ONE = "1 marker was under 10 µm³"

# the counts note: (key, many, one); joined with " · "
_COUNT_PARTS = (
    # first: the one part the user must act on, so the one-line elision
    # never hides it (B-M1)
    ("n_absorbed", ADD_NOTE_ABSORBED, ADD_NOTE_ABSORBED_ONE),
    # Inside each cell: what the user acts on first (a skipped cell, an
    # empty cell), then what was decided for them
    ("n_cell_has_one", ADD_NOTE_CELL_HAS_ONE, ADD_NOTE_CELL_HAS_ONE_ONE),
    ("n_no_nucleus", ADD_NOTE_NO_NUCLEUS, ADD_NOTE_NO_NUCLEUS_ONE),
    ("n_split_border", ADD_NOTE_SPLIT_BORDER, ADD_NOTE_SPLIT_BORDER_ONE),
    ("n_outside_cells", ADD_NOTE_OUTSIDE_CELLS, ADD_NOTE_OUTSIDE_CELLS_ONE),
    # Nuclei ▸ At the markers I place: markers that made nothing (the user
    # moves them or lowers the threshold)
    ("n_markers_dim", ADD_NOTE_MARKERS_DIM, ADD_NOTE_MARKERS_DIM_ONE),
    ("n_dropped_dark", "{n} markers fell in dark space and were skipped",
     "1 marker fell in dark space and was skipped"),
    ("n_outside_outline", ADD_NOTE_OUTSIDE_OUTLINE,
     ADD_NOTE_OUTSIDE_OUTLINE_ONE),
    ("n_dropped_thin", "{n} run objects the box edge cut too thin to use",
     "1 run object the box edge cut too thin to use"),
    ("n_nuclei_slivers",
     "{n} nuclei slivers against kept objects were skipped",
     "1 nuclei sliver against kept objects was skipped"),
    # Cells ▸ Nuclei masks (a box or a whole patch): the markers found
    # under ADD_SMALL_MARKER_UM3 (kept; said so the user can judge them)
    ("n_small_markers", ADD_NOTE_SMALL_MARKERS, ADD_NOTE_SMALL_MARKERS_ONE),
    ("n_slivers", ADD_NOTE_RUN_SLIVERS, ADD_NOTE_RUN_SLIVERS_ONE),
    ("n_markers_over_kept", "{n} markers sat on kept objects",
     "1 marker sat on kept objects"),
    # the engine's one count of the pieces left after the growth that hold
    # no voxel of their own marker (cut off by kept objects, the tissue
    # outline or a narrow neck alike): said as what it counts
    ("n_fragments_dropped",
     "{n} pieces not joined to their marker were left out",
     "1 piece not joined to its marker was left out"),
    ("n_below_min_size", "{n} objects were below Smallest object",
     "1 object was below Smallest object"),
    ("n_emptied_by_smooth",
     "{n} objects vanished when smoothed and were left out",
     "1 object vanished when smoothed and was left out"),
)
ADD_NOTE_COUNTS_SEP = " · "
#: the parts said otherwise on a whole patch (no box edge cuts anything)
_COUNT_PARTS_WHOLE = {
    "n_dropped_thin": (
        "{n} run objects were too small to shrink into a marker and were "
        "skipped",
        "1 run object was too small to shrink into a marker and was skipped"),
}
#: the Nuclei preset's tissue-outline part: the nuclei left out for lying
#: mostly outside it and the nuclei the cut to it emptied
ADD_NOTE_NUCLEI_OUTSIDE_OUTLINE = (
    "{n} nuclei lay outside the tissue outline and were left out"
)
ADD_NOTE_NUCLEI_OUTSIDE_OUTLINE_ONE = (
    "1 nucleus lay outside the tissue outline and was left out"
)
#: the parts said otherwise on the Nuclei preset (the same place in order)
_COUNT_PARTS_NUCLEI = {
    "n_outside_outline": (ADD_NOTE_NUCLEI_OUTSIDE_OUTLINE,
                          ADD_NOTE_NUCLEI_OUTSIDE_OUTLINE_ONE),
}

ADD_CONFIRM_UNDO_ALL = "Take back all {n} objects added this session?"
ADD_CONFIRM_UNDO_ALL_ONE = "Take back the object added this session?"
#: the confirm's second line when a Clear made after the Accepts is still
#: on the undo stack (Undo all stops at it)
ADD_CONFIRM_UNDO_ALL_BLOCKED = (
    "A Clear made after them stands in the way: the main Undo takes it back "
    "first."
)
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
#: Undo all after a Clear of this session that is still in force (Undo all
#: takes back Accepts only; a Clear is undone with the main Undo button)
ADD_UNDO_DONE_CLEARED = (
    "The objects you cleared are still cleared — the main Undo button "
    "brings them back."
)

# --------------------------------------------------------------------------- #
# Clear objects (the menu button beside Preview)
# --------------------------------------------------------------------------- #
ADD_CLEAR_BTN = "Clear objects"
ADD_CLEAR_ITEM_BOX = "Objects mostly inside this box…"
ADD_CLEAR_ITEM_PATCH = "Every object in the patch…"
ADD_CLEAR_TIP = (
    "Remove the objects that are here so new ones can be made in their "
    "place. The main Undo button brings them back once the box is closed."
)
#: the idle strip's stage line while Clear objects is on screen
ADD_STRIP_CLEAR_HINT = "Clear makes room by deleting old objects."
ADD_S1_CLEAR_COUNTING = "Counting the objects to clear…"
ADD_CLEAR_CONFIRM_TITLE = "Clear objects"
ADD_CLEAR_CONFIRM_BOX = (
    "Clear the {n} objects that sit mostly inside this box?"
)
ADD_CLEAR_CONFIRM_BOX_ONE = (
    "Clear the object that sits mostly inside this box?"
)
ADD_CLEAR_CONFIRM_PATCH = "Clear all {n} objects in this patch?"
ADD_CLEAR_CONFIRM_PATCH_ONE = "Clear the object in this patch?"
ADD_CLEAR_CONFIRM_OUTSIDE = (
    "{k} of them stick out of the box; those parts go too."
)
ADD_CLEAR_CONFIRM_OUTSIDE_ONE = (
    "1 of them sticks out of the box; that part goes too."
)
#: the confirm of ONE object (n == 1): "it", not "1 of them"
ADD_CLEAR_CONFIRM_OUTSIDE_IT = "It sticks out of the box; that part goes too."
#: the count is of active rows marked Curated or Good
ADD_CLEAR_CONFIRM_GOOD = "{g} of them are marked Curated or Good."
ADD_CLEAR_CONFIRM_GOOD_ONE = "1 of them is marked Curated or Good."
ADD_CLEAR_CONFIRM_GOOD_IT = "It is marked Curated or Good."
ADD_CLEAR_CONFIRM_UNDO = (
    "The main Undo button brings them back after you close the box, until "
    "you switch patches, run Extract or Commit Triage, or use Reset to "
    "Predicted."
)
ADD_CLEAR_CONFIRM_UNDO_IT = (
    "The main Undo button brings it back after you close the box, until you "
    "switch patches, run Extract or Commit Triage, or use Reset to "
    "Predicted."
)
ADD_CLEAR_CONFIRM_OK = "Clear"
ADD_CLEAR_CONFIRM_BACK = "Go back"
ADD_S1_CLEARING = "Clearing {n} objects…"
ADD_S1_CLEARING_ONE = "Clearing 1 object…"
ADD_S1_CLEARED = (
    "Cleared {n} objects. The main Undo button brings them back once this "
    "box is closed."
)
ADD_S1_CLEARED_ONE = (
    "Cleared 1 object. The main Undo button brings it back once this box is "
    "closed."
)
#: S1's second sentence after a box Clear that took parts outside the box
ADD_S1_CLEARED_OUTSIDE = (
    "Some stuck out of the box — grow the box, or place another, to fill "
    "their space."
)
#: the same after a Clear of ONE object
ADD_S1_CLEARED_OUTSIDE_IT = (
    "It stuck out of the box — grow the box, or place another, to fill its "
    "space."
)
ADD_S1_CLEAR_NOTHING = (
    "No object sits mostly inside this box — nothing to clear."
)
ADD_S1_CLEAR_EMPTY_PATCH = "This patch has no objects — nothing to clear."
ADD_S1_CLEAR_STALE = (
    "The box changed while counting — pick Clear objects again."
)
ADD_S1_CLEAR_FAILED = (
    "The clear failed — the reason is in the log. Nothing was changed."
)
ADD_FOOT_BOX_CLEARED = (
    "Close box removes the box; what you cleared stays cleared until you "
    "Undo."
)
#: the undo entry's text (the log's "Undid: ..." line)
ADD_CLEAR_UNDO_TEXT = "Cleared {n} objects (Add)"
ADD_CLEAR_UNDO_TEXT_ONE = "Cleared 1 object (Add)"
#: no box: what to do first (true after a Done that added objects too)
ADD_FOOT_IDLE = "Place a box to start."
#: the footer's secondary has one word per meaning: Close box (a box is up),
#: Discard (a result is on screen), Undo all (REVIEW)
ADD_FOOT_BOX = "Close box ends this box; nothing is added."
#: a box opened by "+ Add another": Close box goes back to the review (the
#: session and its Undo all are kept)
ADD_FOOT_BOX_SESSION = (
    "Close box ends this box and goes back to the {n} objects already added."
)
ADD_FOOT_BOX_SESSION_ONE = (
    "Close box ends this box and goes back to the 1 object already added."
)
ADD_FOOT_CANDIDATES = (
    "Accept keeps the ticked objects. Discard throws this result away; the "
    "box stays."
)
#: REVIEW: the primary is Done, the secondary Undo all
ADD_FOOT_DONE = (
    "Done keeps the {n} objects. Undo all takes back every Accept since the "
    "first box."
)
ADD_FOOT_DONE_ONE = (
    "Done keeps the 1 object. Undo all takes back every Accept since the "
    "first box."
)
#: REVIEW's primary (Done) -- it closes the session and keeps the objects
ADD_DONE_TIP = (
    "Close this Add session and keep the objects you added. After Done the "
    "main Undo button takes them back, one Accept at a time."
)


#: the same box after a Clear in it: Close box still goes back to the
#: review, and the Clear stays (only the main Undo brings it back)
ADD_FOOT_BOX_SESSION_CLEARED = (
    "Close box goes back to the {n} objects added; what you cleared stays "
    "cleared."
)
ADD_FOOT_BOX_SESSION_CLEARED_ONE = (
    "Close box goes back to the 1 object added; what you cleared stays "
    "cleared."
)
#: REVIEW with a Clear made after the Accepts still on the undo stack: Undo
#: all stops at it, so the line says so instead of promising every Accept
ADD_FOOT_DONE_BLOCKED = (
    "Done keeps the {n} objects. A Clear made after them stands in the way "
    "of Undo all."
)
ADD_FOOT_DONE_BLOCKED_ONE = (
    "Done keeps the 1 object. A Clear made after it stands in the way of "
    "Undo all."
)
#: REVIEW with every object added here cleared since
ADD_FOOT_DONE_NONE = (
    "Every object added here was cleared since. Done closes the review."
)


def foot_box_session(n, *, cleared=False) -> str:
    """The footer line of a box opened by "+ Add another" (``n`` objects
    added this session so far; ``cleared``: a Clear was made in this
    box)."""
    if cleared:
        return count_phrase(n, ADD_FOOT_BOX_SESSION_CLEARED_ONE,
                            ADD_FOOT_BOX_SESSION_CLEARED)
    return count_phrase(n, ADD_FOOT_BOX_SESSION_ONE, ADD_FOOT_BOX_SESSION)


def foot_review(n, *, blocked=False) -> str:
    """REVIEW's footer line under Done and Undo all (``blocked``: a Clear
    made after the Accepts stands in Undo all's way)."""
    if int(n) <= 0:
        return ADD_FOOT_DONE_NONE
    if blocked:
        return count_phrase(n, ADD_FOOT_DONE_BLOCKED_ONE,
                            ADD_FOOT_DONE_BLOCKED)
    return count_phrase(n, ADD_FOOT_DONE_ONE, ADD_FOOT_DONE)


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
#: the click count ("7 placed"): the Markers summary and the first line
ADD_CLICKS_PLACED = "{n} placed"
#: the count line and the Markers summary when k < n
ADD_CLICKS_TICKED = "{n} placed · {k} ticked"
#: the word before the click ball's radius, beside Place markers
ADD_BALL_LABEL = "Ball"

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
# Nuclei inside each cell of another run, and the whole patch (Nuclei)
# --------------------------------------------------------------------------- #
#: the Nuclei preset's "Look for nuclei" combo, directly under Channel; read
#: BY INDEX through the panel's key map (index 0 is always Anywhere)
ADD_LOOK_FOR_LABEL = "Look for nuclei"
ADD_LOOK_ANYWHERE = "Anywhere in the box"
#: the last item: one click per nucleus (a box or the whole patch)
ADD_LOOK_MARKERS = "At the markers I place"
#: the Look for nuclei combo's help line
ADD_HELP_LOOK_FOR = (
    "Where to look for nuclei: anywhere in the box by brightness, inside "
    "each cell of another run of this patch, or at the markers you place, "
    "one click per nucleus."
)
#: ``{target}`` = " · " + the run's target when it has one
ADD_LOOK_INSIDE = "Inside each cell of run ▸ {name}{target}"
ADD_LOOK_NO_RUN = (
    "Inside each cell of another run (no other run of this patch)"
)
ADD_HELP_INSIDE = (
    "Finds each nucleus by its own edge in this channel, then gives it to "
    "the cell of that run that holds most of it. Needs a nuclear stain. The "
    "threshold is not used here; nuclei outside every cell are left out."
)
#: the grey no-run item's tooltip (and the help line only if Inside is ever
#: current with no run) -- never under Anywhere
ADD_HELP_INSIDE_NO_RUN = (
    "No other run of this patch holds cells yet. To find nuclei inside your "
    "cells: in Step 3 press New Run (a nuclei model), Predict + Reconstruct "
    "this patch, open it in that run, and pick your cells run here."
)
#: the picked run's target says nuclei: the open run probably holds the
#: cells, and the nuclei belong in that other run
ADD_HELP_INSIDE_PICKED_NUCLEI = (
    "That run is marked as nuclei, so this run probably holds the cells. To "
    "find the nucleus inside each of these cells, open this patch in that "
    "nuclei run and pick this run under Look for nuclei."
)
ADD_INSIDE_COUNT = "{c} cells of that run in the box."
ADD_INSIDE_COUNT_ONE = "1 cell of that run in the box."
ADD_INSIDE_COUNT_PENDING = "Counting that run's cells in the box…"
#: the same three on the whole patch
ADD_INSIDE_COUNT_PATCH = "{c} cells of that run in the patch."
ADD_INSIDE_COUNT_PATCH_ONE = "1 cell of that run in the patch."
ADD_INSIDE_COUNT_PATCH_PENDING = "Counting that run's cells in the patch…"
#: the one amber slot under Look for nuclei (first match wins: half size,
#: then a box mostly full, then a slow whole patch)
ADD_AMBER_BOX_MOSTLY_FULL = (
    "Only {p} % of the box is empty — new nuclei go only into empty space. "
    "Clear objects first, or open the nuclei run."
)
#: the same, when the picked run is marked nuclei: the objects here are
#: then probably the curated cells, and clearing them would lose them
ADD_AMBER_BOX_MOSTLY_FULL_NUCLEI_RUN = (
    "Only {p} % of the box is empty — new nuclei go only into empty space. "
    "Open the nuclei run."
)
#: the share of empty space under which the box counts as mostly full
ADD_MOSTLY_FULL_SHARE = 0.2
#: the half-size switch is Step 3's "Downsample patches 2× on load" tick
#: box under Curate; it acts when a patch is next opened
ADD_NOTE_HALFSIZE_BOXED = (
    "Half-size session: the other run is shrunk to match, so its small or "
    "very thin objects can disappear, and new objects are saved back at "
    "full size in 2-voxel steps. For best results, press Close box, untick "
    "Downsample patches 2× on load under Curate, and open the patch again."
)
ADD_S1_INSIDE_UNUSABLE = (
    "That run cannot be used on this patch — the reason is under Look for "
    "nuclei."
)
#: the patch's brightness levels refused the Brightness channel (it is the
#: channel, not the run): a channel change clears it and measures again
ADD_S1_INSIDE_CHANNEL_UNUSABLE = (
    "This channel cannot be used to find nuclei — the reason is under Look "
    "for nuclei."
)
ADD_S1_INSIDE_NO_CELLS = (
    "That run has no cell in this box — move the box or pick another run."
)
#: the same on the whole patch (there is no box to move)
ADD_S1_INSIDE_NO_CELLS_PATCH = (
    "That run has no cell in this patch — pick another run."
)
ADD_S1_INSIDE_NOTHING = (
    "No nucleus was found inside that run's cells — check the channel, and "
    "Nucleus radius under Advanced."
)
ADD_S1_INSIDE_ALL_TAKEN = (
    "Nuclei were found, but every one sits on an object already here — "
    "Clear objects first, or open the nuclei run."
)
#: the same, when the picked run is marked nuclei (no "Clear objects": the
#: objects here are probably the curated cells)
ADD_S1_INSIDE_ALL_TAKEN_NUCLEI_RUN = (
    "Nuclei were found, but every one sits on an object already here — "
    "open the nuclei run."
)
#: S1 tails (no leading separator: the line joins them with " · ")
ADD_S1_TWO_TAIL = "{s} cells hold two nuclei"
ADD_S1_TWO_TAIL_ONE = "1 cell holds two nuclei"
ADD_S1_ADDED_TWO_TAIL = (
    "{s} cells hold two nuclei — after Save, run Extract and pick that run "
    "under Compare with to find them."
)
ADD_S1_ADDED_TWO_TAIL_ONE = (
    "1 cell holds two nuclei — after Save, run Extract and pick that run "
    "under Compare with to find it."
)
#: the two check-this chips of a candidate (and REVIEW) row; neither
#: unticks the row
ADD_CHIP_TWO = "two in a cell"
ADD_TIP_TWO = (
    "Cell {cell} of that run holds {k} nuclei here. It may be two cells "
    "drawn as one, a dividing cell, or a neighbour's nucleus poking in. "
    "Each is added on its own; untick any you do not want."
)
ADD_CHIP_CUT = "cut at edge"
ADD_TIP_CUT = (
    "The border of cell {cell} cuts off more than a quarter of this "
    "nucleus. The border may run through the nucleus — check it."
)
ADD_WALLS_INSIDE = (
    "The nuclei follow that run's cells; there is no wall picture to show."
)
ADD_HELP_NUC_RADIUS = (
    "The usual radius of a nucleus. It sets the size of the bright spots "
    "the finder starts from."
)
ADD_HELP_NUC_EDGE = (
    "Where each nucleus's edge is drawn: this share of the way from the "
    "cytoplasm up to the nucleus's own brightest point. Lower draws bigger "
    "nuclei."
)
ADD_HELP_TWO_GAP = (
    "Two nuclei in one cell stay separate only when the darkest point "
    "between them is below this share of the dimmer one; otherwise they "
    "become one nucleus."
)
ADD_SUM_ADVANCED_INSIDE = "radius {r} µm · edge {e} · gap {g}"
ADD_WHOLE_BTN = "Whole patch"
#: the box row's fourth size (the idle row's button is ``ADD_WHOLE_BTN``)
ADD_WHOLE_SIZE_BTN = "Whole"
#: the two Whole buttons' tooltip: plain help -- they are never grey; a
#: choice that cannot fill a whole patch greys Preview and S1 says why
ADD_WHOLE_TIP = "Use the whole patch as the box."
#: the strip's busy line after Stop on a region over
#: ``ADD_SLOW_PATCH_VOXELS``: a step already running (a distance map, a
#: flood) is never cut short, so Stop waits for it to end
ADD_STOP_STEP_ENDS = "Stop takes effect when this step ends."
ADD_HELP_WHOLE = (
    "The box is the whole patch, so it does not move. S, M or L make it a "
    "box again."
)
#: the ready line of a whole patch, both presets
ADD_S1_PATCH_PLAIN = (
    "Whole patch {z} × {y} × {x} voxels ({m} million) · {p} % of it is "
    "empty space"
)
#: above this many voxels the whole patch says a Preview may take minutes
ADD_SLOW_PATCH_VOXELS = 20_000_000
#: the nuclei finder alone (Cells ▸ Nuclei masks finds them and stops),
#: seconds per million voxels: 61 s on 100 million (the steps below)
FINDER_SECONDS_PER_MILLION = 0.61
#: Nuclei ▸ Anywhere in the box, seconds per million voxels for the WHOLE
#: Preview (the finder, then the edge flags, the smoothing of every object
#: and the size pass): 6.8 s on 5.2 million voxels with 358 nuclei (a
#: real patch), 12.6 s on 8.2 million with 1,243 (synthetic).  The finder
#: alone is about 0.6 s per million (on 100 million: 2.9 threshold + 37.2
#: distance map + 3.3 blur + 5.9 peaks + 10.2 flood + 1.1 size floor).
ANYWHERE_SECONDS_PER_MILLION = 1.4
#: above this many voxels a Cells whole patch filled as you click (at most
#: ``ADD_LIVE_CELLS_MAX_VOXELS``) says its first update prepares it once
#: and may take a while: that prepare was measured at 10.5 s on 5.2
#: million voxels
ADD_SLOW_LIVE_PATCH_VOXELS = 5_000_000
#: S1 (Preview grey) on a Cells whole patch with nothing picked under
#: Markers ▸ From (a run is never picked for the user there)
ADD_S1_WHOLE_PICK_RUN = (
    "Pick a run, Nuclei masks or Click markers under Markers ▸ From — a "
    "whole patch grows its cells from those markers."
)
#: ... and on a patch with no other run (the list holds its grey item,
#: the nuclei masks and Click markers)
ADD_S1_WHOLE_NO_RUN = (
    "This patch has no other run — pick Nuclei masks or Click markers "
    "under Markers ▸ From."
)
#: the amber slot on a large Cells whole patch; ``{t}`` is
#: :func:`wait_phrase` of :func:`whole_fill_minutes` (see
#: :func:`patch_slow_line`)
ADD_NOTE_PATCH_SLOW_STEPS = (
    "This patch is {m} million voxels; a Preview may take {t}, "
    "longer on a slower computer. Stop acts between steps."
)
#: ... and on a Cells box over ``ADD_SLOW_PATCH_VOXELS`` (a box follows the
#: whole patch's size rule, so a box that large is filled in one go by a
#: Preview too); ``{t}`` is the box's own figure (``patch_slow_line``
#: with its Grow by: ``BOX_FILL_SECONDS_PER_MILLION`` for the fill to the
#: walls, Spread's rate for Spread), a rough one
ADD_NOTE_BOX_SLOW_STEPS = (
    "This box is {m} million voxels; a Preview may take {t}, "
    "longer on a slower computer. Stop acts between steps."
)
#: the amber slot on a large Nuclei whole patch: that wait was never timed
#: and a Nuclei whole patch has no size ceiling, so no figure (see
#: :func:`nuclei_patch_slow_line`)
ADD_NOTE_PATCH_SLOW_PLAIN = (
    "This patch is {m} million voxels; a Preview may take several minutes "
    "or longer. Stop acts between steps."
)
#: ... and with "Update as I click" on for Nuclei ▸ At the markers I place:
#: no Preview is pressed and Stop is grey during an update; the patch is
#: read once, at the first update (no longer than the line above)
ADD_NOTE_PATCH_SLOW_LIVE = (
    "This patch is {m} million voxels; the first update reads it once and "
    "may take a while. Later clicks are quick."
)
#: ... and for Cells ▸ Click markers on a whole patch filled as you click,
#: over ``ADD_SLOW_LIVE_PATCH_VOXELS`` and while no prepared patch is kept
#: for it: the first update prepares the patch (its wall picture and its
#: fill)
ADD_NOTE_PATCH_SLOW_LIVE_CELLS = (
    "This patch is {m} million voxels; the first update prepares it once "
    "and may take a while. Later clicks are quick."
)
#: the wait in words (:func:`wait_words`): under 0.75 minutes, under 1.5
#: minutes, and from there whole minutes; ``{t}`` of the wait notes adds
#: "about" to the last two (:func:`wait_phrase`)
WAIT_UNDER_A_MINUTE = "under a minute"
WAIT_A_MINUTE = "a minute"
WAIT_MINUTES = "{n} minutes"
WAIT_ABOUT = "about {w}"
#: under 45 s the slow update's wait is said in seconds, to 5 s
WAIT_SECONDS = "{n} s"
#: painted before the whole-patch read on the main thread
ADD_S1_GATHERING = "Reading the patch…"
#: painted before Accept's write on a whole patch (a visible moment on the
#: main thread)
ADD_S1_ADDING = "Adding {n} objects…"
ADD_S1_ADDING_ONE = "Adding 1 object…"
#: painted before the main-thread waits (the app is frozen while they run;
#: the words are set and painted first)
ADD_S1_UNDOING_ALL = "Taking back {n} objects…"
ADD_S1_UNDOING_ALL_ONE = "Taking back 1 object…"
ADD_S1_REMOVING_RESULT = "Taking the result off the screen…"
ADD_S1_DRAWING_WALLS = "Drawing the wall picture…"
ADD_S1_DRAWING_MARKERS = "Drawing the markers…"
#: the amber slot on Cells with a ridge wall type: its wall picture is
#: slow to build (it grows with the patch and with fewer cores).  It is kept
#: on a whole patch and on a box at or under ``ADD_LIVE_CELLS_MAX_VOXELS``
#: (said until it is kept), and built again at every Preview of a larger
#: box.  On a whole
#: patch ``{t}`` is :func:`wait_phrase` of the whole Preview's
#: :func:`whole_fill_minutes` (the picture included), and from
#: ``ADD_RIDGE_MEMORY_NOTE_GB`` on the ``_MEMORY`` form names the memory
#: (``{g}`` GB) too (see :func:`ridges_whole_line`)
ADD_NOTE_RIDGES_WHOLE = (
    "{kind} is slow on a whole patch: a Preview takes {t} here, most of it "
    "building the wall picture. Membrane sheets is much faster."
)
ADD_NOTE_RIDGES_WHOLE_MEMORY = (
    "{kind} is slow on a whole patch: a Preview takes {t} here, most of it "
    "building the wall picture, and needs about {g} GB of memory. Membrane "
    "sheets is much faster."
)
#: the ridge note names the memory from this many GB on
ADD_RIDGE_MEMORY_NOTE_GB = 8
#: the two wall types whose wall picture is the slow one
RIDGE_WALL_KINDS = ("frangi", "meijering")
#: rough whole-patch Cells fill times, measured on a fast workstation:
#: Membrane sheets (and every wall type but the ridges) takes
#: m * (A + B * m) seconds for a patch of m million voxels ...
FILL_SECONDS_PER_MILLION = 4.6
FILL_SECONDS_GROWTH = 0.035
#: ... and a ridge wall type adds its wall picture, this many seconds per
#: million voxels, with this peak memory in GB per million voxels
RIDGE_PICTURE_SECONDS_PER_MILLION = 10.0
RIDGE_GB_PER_MILLION = 0.2
#: ... and on a box filled as you click, the wall picture of a ridge wall
#: type per million voxels: measured on Vessel ridges, one 4.2 million box
#: (its 58.6 s update less the fill's own time)
RIDGE_BOX_PICTURE_SECONDS_PER_MILLION = 12.6
#: ... the fill to the walls of a box (or of a whole patch filled the way a
#: box is) as the slow live update runs it, on a warm stage: measured
#: 5.8-6.8 s on 5.2 million voxels with Membrane sheets
LIVE_FLOOD_SECONDS_PER_MILLION = 1.3
#: ... the same fill as a cold Preview of a box runs it (the levels read,
#: the walls built, the smoothing): 12.5 s on 4.6 million and 52.0 s on
#: 23.0 million voxels (synthetic tissue, Membrane sheets, one machine)
BOX_FILL_SECONDS_PER_MILLION = 2.3
#: ... and Spread from each marker: measured 65.7 s on a 5.2 million patch
#: (one measurement, on one patch)
SPREAD_SECONDS_PER_MILLION = 12.6
#: a box over ``ADD_LIVE_CELLS_MAX_VOXELS``: its wall picture is not kept
ADD_NOTE_RIDGES_BOX = (
    "{kind} is slow on a box this large: its wall picture is built again at "
    "every Preview. Membrane sheets is much faster."
)
#: a box at or under ``ADD_LIVE_CELLS_MAX_VOXELS`` whose wall picture is
#: not kept yet: it is built once, by the next run, and kept for the box
ADD_NOTE_RIDGES_BOX_ONCE = (
    "{kind} is slow on a box this large: its wall picture is built once, "
    "then kept for this box. Membrane sheets is much faster."
)
#: a Cells box from this size on says so with a ridge wall type (its wall
#: picture: about 12.6 s per million voxels, measured on Vessel ridges, one
#: box -- about 25 s here)
ADD_RIDGE_BOX_VOXELS = 2_000_000
#: what "Look for nuclei" sends the engine (by index through the panel's map)
LOOK_ANYWHERE = "anywhere"
LOOK_INSIDE = "inside"
LOOK_MARKERS = "markers"
#: what "Separate touching objects" ticked asks of the markers (one of the
#: engine's split modes): ``"always"`` cuts every clicked bright piece at
#: its waists, so one click gives the nucleus it sits on; ``"single"`` cuts
#: only a piece holding one click.  Unticked is always ``"off"`` (each
#: clicked piece whole, as before).  Provisional: the choice between the two
#: is made by trying both.
ADD_MARKER_SPLIT_DEFAULT = "always"

# --------------------------------------------------------------------------- #
# Cells: how cells grow, the whole patch, the tissue outline
# --------------------------------------------------------------------------- #
#: Advanced ▸ Grow by, read BY INDEX through ``GROW_KEYS`` (what the engine
#: calls each rule)
ADD_GROW_LABEL = "Grow by"
ADD_GROW_FLOOD = "Fill to the walls (as before)"
ADD_GROW_SPREAD = "Spread from each marker"
#: item 1 when this install cannot run it (greyed; the reason is in the
#: Advanced help slot)
ADD_GROW_SPREAD_GREY = "Spread from each marker (not available here)"
GROW_FLOOD = "flood"
GROW_SPREAD = "compete"
GROW_KEYS = (GROW_FLOOD, GROW_SPREAD)
#: what a Cells whole patch grows by, whatever Grow by says: one fill to
#: the walls from every marker at once, with a background marker far from
#: every marker (not a Grow by item: the row is hidden on a whole patch)
GROW_PATCH = "fill_with_background"
ADD_HELP_GROW = (
    "Fill: the fill runs to the walls, the box edges and dark space, as "
    "before. Spread: each marker grows along its easiest path and stops "
    "where a neighbour got there first, at a wall, or at the growth limit. "
    "Pick Spread when the fill leaks through a gap in the walls or merges "
    "neighbours."
)
ADD_WALL_COST_LABEL = "Wall cost"
ADD_HELP_WALL_COST = (
    "How costly a bright membrane is to cross. Higher keeps cells inside "
    "their walls; too high and cells stop short. 0 ignores the walls: each "
    "cell is the space closest to its marker."
)
ADD_GROWTH_LIMIT_LABEL = "Growth limit"
ADD_HELP_GROWTH_LIMIT = (
    "How far a cell may spread: a budget, not a distance. Plain cytoplasm "
    "uses it up about 6 µm from the marker; a bright membrane costs nearly "
    "the whole limit per µm, so a cell stops in its wall. Lower it if cells "
    "grow past their walls."
)
ADD_INCLUDE_MEMBRANE = "New cells include the membrane at their edge"
ADD_HELP_INCLUDE_MEMBRANE = (
    "New cells include the strip of membrane at their edge (up to 1 µm). "
    "Untick it if your cells are drawn without their membrane."
)
#: the Advanced summary's growth part (Spread only)
ADD_SUM_ADVANCED_GROW = (
    "spreading · wall cost {w} · limit {l} · membrane included"
)
ADD_SUM_ADVANCED_GROW_NO_MEMBRANE = (
    "spreading · wall cost {w} · limit {l} · membrane left out"
)
#: ... and on a Cells whole patch (Grow by, Wall cost and Growth limit are
#: hidden there)
ADD_SUM_ADVANCED_FILL = "filled to the walls · membrane included"
ADD_SUM_ADVANCED_FILL_NO_MEMBRANE = "filled to the walls · membrane left out"
#: ... and on a whole patch filled as you click (the way a box is: the
#: membrane tick does not apply and is hidden)
ADD_SUM_ADVANCED_FILL_LIVE = "filled to the walls"
#: the helper line at the top of the Cells card under Advanced, on a whole
#: patch only: what fills it, and how to fill what it leaves empty
#: the one way a whole patch differs from a box, and its reason
ADD_HELP_WHOLE_FILL_WHY = (
    "A patch edge is the end of the data; a box edge cuts through tissue."
)
#: (worded to stay three lines at the page's width: the Advanced-open
#: whole-patch states may not grow)
ADD_HELP_WHOLE_FILL = (
    "Filled from every marker at once; far tissue stays empty (click "
    "markers there). " + ADD_HELP_WHOLE_FILL_WHY
)
#: ... in the same slot on a whole patch filled as you click (Cells ▸ Click
#: markers, "Update as I click" on, walls the click fill redoes)
ADD_HELP_WHOLE_FILL_LIVE = (
    "Filled as you click, the way a box is: tissue far from every marker "
    "fills too. To keep it empty, untick Update as I click and press "
    "Preview."
)
#: ... and on a whole patch with Grow by ▸ Spread from each marker
ADD_HELP_WHOLE_FILL_SPREAD = (
    "Spread from each marker grows every cell of the whole patch at once; "
    "each stops at a wall, where a neighbour got there first, or at its "
    "growth limit. Tissue no cell gets to stays empty."
)
#: ... and on the same patch when the click fill stepped aside from it (its
#: flat spots, a failure: S1 says to press Preview): a Preview fills it,
#: still the way a box is
ADD_HELP_WHOLE_FILL_PREVIEW = (
    "Filled on Preview, the way a box is: tissue far from every marker "
    "fills too. To keep it empty, untick Update as I click and press "
    "Preview."
)
#: ... and there with "Always update as I click" on: the region is filled
#: whole a moment after the clicks settle
ADD_HELP_WHOLE_FILL_ALWAYS = (
    "Filled a moment after your clicks (Always update as I click is on), "
    "the way a box is: tissue far from every marker fills too."
)


def whole_fill_help(live_rule, live_ok=None, spread=False,
                    always=False) -> str:
    """The helper line of a Cells whole patch under Advanced: grown by
    Spread from each marker (``spread``, Grow by's pick; never the live
    rule); the fill with a background marker (``live_rule`` False); filled
    as you click by the box rule; or by that rule on Preview only, when the
    click fill stepped aside (``live_ok`` False -- the S1 gate says the
    same) -- or, there with "Always update as I click" on (``always``), a
    moment after the clicks.  ``live_ok`` ``None`` (not said): as you
    click."""
    if spread:
        return ADD_HELP_WHOLE_FILL_SPREAD
    if not live_rule:
        return ADD_HELP_WHOLE_FILL
    if live_ok is False:
        return (ADD_HELP_WHOLE_FILL_ALWAYS if always
                else ADD_HELP_WHOLE_FILL_PREVIEW)
    return ADD_HELP_WHOLE_FILL_LIVE
#: S1 (Preview grey) on a Cells whole patch over the fill's ceiling;
#: ``{k}`` is the ceiling in words, as a box's line says its limit (see
#: :func:`whole_too_big_line`)
ADD_S1_WHOLE_TOO_BIG = (
    "This patch is {m} million voxels; a whole patch can be filled up to "
    "{k}. Press S, M or L for a box."
)
#: in the Advanced help slot under Spread with no outline and the dark tick
#: off: nothing but the growth limit stops a cell in open background
ADD_NOTE_SPREAD_OPEN = (
    "With no tissue outline and dark space not treated as outside, Spread "
    "grows each cell into open background until the growth limit stops it."
)
#: Spread from each marker cannot run in this install: the solver's
#: compiler (the numba package) is missing or cannot keep its compiled code
ADD_NO_NUMBA = (
    "Spread from each marker cannot run here: the numba package it needs "
    "is missing from this install, so Cells fills boxes only. Reinstall "
    "EpiCycle's environment to get it back."
)
#: the same when numba is there but cannot keep its compiled code (its
#: cache folder cannot be written): the engine's own sentence goes to the
#: log
ADD_NO_NUMBA_CACHE = (
    "Spread from each marker cannot run here: numba, the package it needs, "
    "cannot write its compiled code to its cache folder (this install's "
    "folder may be read-only), so Cells fills boxes only. Point the cache "
    "at a folder you can write: set the NUMBA_CACHE_DIR environment "
    "variable to it, then restart napari."
)
#: Whole patch on Cells needs the numba package (S1, Preview grey, in an
#: install where it is missing or cannot keep its compiled code)
ADD_WHOLE_NEEDS_NUMBA = (
    "Cells fills a whole patch only with the numba package, which is "
    "missing from this install. Use boxes."
)
ADD_WHOLE_NEEDS_NUMBA_CACHE = (
    "Cells fills a whole patch only with the numba package, which cannot "
    "write its cache folder here. Use boxes."
)
#: why Spread cannot run: the keys of the once-per-process answer
NUMBA_OK = "ok"
NUMBA_MISSING_KEY = "missing"
NUMBA_CACHE_KEY = "cache"
#: painted on S1 while the growth engine is set up for its first run on
#: this computer (only when that takes more than a few seconds)
ADD_S1_WARMING = (
    "Preparing the cell-growth engine for its first run on this computer "
    "— about 10 s, once."
)
#: S1 (a transient) when a Numbers... slider or S / M / L makes a Cells
#: whole patch a box again and the box had to shrink to the Cells limit
#: (``ADD_BOX_BUDGET``).  Formatted once, at import, from the constant: a
#: test that patches ``ADD_BOX_BUDGET`` does not change this sentence
ADD_S1_LEFT_WHOLE_CLAMPED = (
    "The box is no longer the whole patch; for cells it was shrunk to the "
    "{} voxel limit.".format(voxel_limit_words(ADD_BOX_BUDGET))
)
#: the confirm a Numbers... slider or S / M / L asks before it makes a
#: whole patch holding markers a box again (red, as Clear objects')
ADD_LEAVE_WHOLE_TITLE = "Leaving the whole patch"
ADD_LEAVE_WHOLE_CONFIRM = "Leaving the whole patch removes its {n} markers."
ADD_LEAVE_WHOLE_CONFIRM_ONE = "Leaving the whole patch removes its 1 marker."
ADD_LEAVE_WHOLE_OK = "Remove them"
ADD_LEAVE_WHOLE_BACK = "Go back"
#: S1 (a transient) after "Remove them": the box is up and every marker of
#: the whole patch is gone (it wins over ADD_S1_LEFT_WHOLE_CLAMPED)
ADD_S1_WHOLE_MARKERS_CLEARED = (
    "The markers placed on the whole patch were removed — place new ones in "
    "the box."
)
ADD_RUN_COUNT_PATCH = (
    "{n} in the patch · {m} over empty space become markers."
)
ADD_RUN_COUNT_PATCH_ONE = (
    "{n} in the patch · 1 over empty space becomes a marker."
)
#: Walls ▸ Tissue outline, read BY INDEX through ``OUTLINE_KEYS``
ADD_OUTLINE_LABEL = "Tissue outline"
ADD_OUTLINE_CLEARED = "Outline of the objects you cleared"
ADD_OUTLINE_CLEARED_GREY = (
    "Outline of the objects you cleared (clear some first)"
)
ADD_OUTLINE_SHRINK = "Shrink-wrap around the stain"
ADD_OUTLINE_NONE = "No outline"
OUTLINE_CLEARED = "cleared"
OUTLINE_SHRINK = "shrink"
OUTLINE_NONE = "none"
OUTLINE_KEYS = (OUTLINE_CLEARED, OUTLINE_SHRINK, OUTLINE_NONE)
ADD_HELP_OUTLINE_CLEARED = (
    "New cells stay inside the outline of the objects you cleared on this "
    "patch."
)
ADD_HELP_OUTLINE_SHRINK = (
    "New cells stay inside a smooth outline around the stained tissue. Gaps "
    "narrower than Bridge gaps (Advanced) are closed over."
)
ADD_HELP_OUTLINE_NONE = (
    "Only the box edges and dark space stop the cells."
)
ADD_SHOW_OUTLINE = "Show the tissue outline"
ADD_OUTLINE_PENDING = "Measuring the tissue outline…"
#: S1 and the Show legend while the outline's layers are added (it holds
#: the main thread for a moment; painted before it starts)
ADD_OUTLINE_DRAWING = "Drawing the tissue outline…"
ADD_HELP_SHOW_OUTLINE = "Drawn in the main view and the three side views."
#: the Show legend when the side views could not take the outline
ADD_SHOW_OUTLINE_MAIN_ONLY = (
    "Drawn in the main view; the side views could not show it (the reason "
    "is in the log)."
)
#: the Show legend while new objects are on screen, and when the outline
#: could not be measured
ADD_SHOW_OUTLINE_REFUSED = "Not drawn: the outline could not be measured."
#: S1 when the outline's worker failed (a reason other than a plain refusal)
ADD_S1_OUTLINE_FAILED = (
    "The tissue outline could not be measured — the reason is in the log. "
    "Pick another Tissue outline under Walls."
)
#: S1 (a transient) when the Clears behind "Outline of the objects you
#: cleared" were undone while it was the choice: the choice falls back
ADD_S1_OUTLINE_CLEARED_GONE = (
    "The objects you cleared are back, so there is no outline of them any "
    "more — Tissue outline is now {choice}."
)
ADD_BRIDGE_LABEL = "Bridge gaps up to"
ADD_HELP_BRIDGE = (
    "The tissue outline closes over gaps narrower than this. Keep it below "
    "the narrowest fold, lumen or wound you want left open."
)
ADD_DARK_BY_OUTLINE = (
    "The tissue outline decides what is outside, so Treat dark space as "
    "outside and Dark space reach are not used. Dark below still sets the "
    "outline."
)
#: the share of markers outside the outline above which the amber shows
ADD_OUTLINE_DROPS_SHARE = 0.05
ADD_AMBER_OUTLINE_DROPS = (
    "{p} % of the markers lie outside the tissue outline and will make "
    "nothing — raise Bridge gaps, lower Dark below, or pick another outline."
)
#: the same for the outline of the objects you cleared: Bridge gaps and
#: Dark below do not shape it, so only another outline helps
ADD_AMBER_OUTLINE_DROPS_CLEARED = (
    "{p} % of the markers lie outside the outline of the objects you "
    "cleared and will make nothing — pick another Tissue outline under "
    "Walls if they should make cells."
)
#: one channel, markers from a run: with Fill picked (on a box), and with
#: Spread picked; ``{bridge}`` is ``ADD_AMBER_ONE_CHANNEL_BRIDGE`` with the
#: shrink-wrap, else empty (each names only controls on the page)
ADD_AMBER_ONE_CHANNEL_RUN = (
    "Only one channel. If it is the nuclear stain, pick Grow by ▸ Spread "
    "from each marker under Advanced and set Wall cost to 0{bridge} — walls "
    "read from nuclei squeeze cells down to their nuclei."
)
ADD_AMBER_ONE_CHANNEL_RUN_SPREAD = (
    "Only one channel. If it is the nuclear stain, set Wall cost to "
    "0{bridge} under Advanced — walls read from nuclei squeeze cells down "
    "to their nuclei."
)
ADD_AMBER_ONE_CHANNEL_BRIDGE = " and Bridge gaps to 16 µm"
#: the walls summary's outline word
ADD_SUM_OUTLINE_SHRINK = "outline: shrink-wrap"
ADD_SUM_OUTLINE_CLEARED = "outline: cleared objects"
#: the engine's two outline refusals as the page says them (the engine's
#: own words go to the log)
ADD_S1_NO_MARKERS_OUTLINE = (
    "Every marker left lay outside the tissue outline — raise Bridge gaps "
    "or lower Dark below under Advanced, or pick another outline."
)
ADD_S1_NO_MARKERS_OUTLINE_CLEARED = (
    "Every marker left lay outside the outline of the objects you cleared "
    "— pick another Tissue outline under Walls."
)
ADD_S1_OUTLINE_PER_BOX = (
    "The tissue outline is not measured yet — wait for it, then press "
    "Preview."
)
# -- Nuclei: the Tissue outline card (an option, off by default) ----------- #
#: the help line under the Nuclei card's Tissue outline, per choice
ADD_HELP_NUC_OUTLINE_CLEARED = (
    "New nuclei are kept only inside the outline of the objects you cleared "
    "on this patch."
)
ADD_HELP_NUC_OUTLINE_SHRINK = (
    "New nuclei are kept only inside a smooth outline around the stained "
    "tissue. With a nuclear stain alone, lone nuclei can fall outside it. "
    "Gaps narrower than Bridge gaps (Advanced) are closed over."
)
ADD_HELP_NUC_OUTLINE_NONE = (
    "Nothing is left out for lying outside the tissue, as before."
)
#: the Nuclei Tissue outline combo's tooltip
ADD_HELP_NUC_OUTLINE = (
    "Where new nuclei may be kept: inside the outline of the objects you "
    "cleared, inside a shrink-wrap around the stain, or (as before) anywhere "
    "in the box."
)
#: the Nuclei card's chevron title: "Tissue outline: {choice}"
ADD_SUM_NUC_OUTLINE = "Tissue outline: {choice}"
ADD_SUM_NUC_OUTLINE_NONE = "none"
ADD_SUM_NUC_OUTLINE_SHRINK = "shrink-wrap"
ADD_SUM_NUC_OUTLINE_CLEARED = "cleared objects"
#: the Nuclei Preview's refusals when every nucleus found lay outside
ADD_S1_NUCLEI_ALL_OUTSIDE = (
    "Every nucleus found lay outside the tissue outline — raise Bridge gaps "
    "or lower Dark below under Advanced, or pick No outline."
)
ADD_S1_NUCLEI_ALL_OUTSIDE_CLEARED = (
    "Every nucleus found lay outside the outline of the objects you cleared "
    "— pick another Tissue outline."
)
#: S1 on Nuclei when the outline's worker failed (Cells' twin names Walls)
ADD_S1_OUTLINE_FAILED_NUCLEI = (
    "The tissue outline could not be measured — the reason is in the log. "
    "Pick another Tissue outline."
)
#: Bridge gaps on a patch with one channel (the outline then leaves wider
#: dim gaps inside the tissue, whichever stain it is) and with more
ADD_BRIDGE_UM = 10.0
ADD_BRIDGE_UM_ONE_CHANNEL = 16.0
#: the display skin of the outline, in microns (at least one voxel on
#: every axis: on a coarse z the caps are one slice thick)
ADD_OUTLINE_SHOW_UM = 1.0


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
    the Cells limit (``ADD_BOX_BUDGET``) and UP above it, so a box over the
    limit never reads as the limit itself ("100.0 million" for
    100,000,001 -- B2 fix pass)."""
    n = float(n)
    tenths = n / 1e5
    if n > ADD_BOX_BUDGET:
        v = math.ceil(tenths - 1e-9) / 10.0
    else:
        v = math.floor(tenths + 1e-9) / 10.0
    return "{:.1f}".format(max(0.1, v))


def box_line(shape, *, preset, empty_share=None, n_clicks=None,
             n_ticked=None, whole_patch=False, look_for=None) -> str:
    """The ready S1 line.  ``n_ticked`` adds
    " · k of n ticked" when fewer than all are ticked.  ``whole_patch``
    says "Whole patch" instead of "Box" (the same words on both
    presets).  The marker count is said on Cells and on Nuclei with
    ``look_for`` = :data:`LOOK_MARKERS`."""
    z, y, x = (int(v) for v in shape)
    n = z * y * x
    if ((preset == "cells" or look_for == LOOK_MARKERS)
            and n_clicks is not None):
        if whole_patch:
            line = count_phrase(n_clicks, ADD_S1_PATCH_CLICKS_ONE,
                                ADD_S1_PATCH_CLICKS, z=z, y=y, x=x)
        else:
            line = count_phrase(n_clicks, ADD_S1_BOX_CLICKS_ONE,
                                ADD_S1_BOX_CLICKS, z=z, y=y, x=x)
        if n_ticked is not None and 0 < int(n_ticked) < int(n_clicks):
            line += ADD_S1_TICKED_TAIL.format(k=int(n_ticked),
                                              n=int(n_clicks))
        return line
    p = int(round(100.0 * float(empty_share or 0.0)))
    if whole_patch:
        tpl = ADD_S1_PATCH_PLAIN
    elif preset == "cells":
        tpl = ADD_S1_BOX
    else:
        tpl = ADD_S1_BOX_NUCLEI
    return tpl.format(z=z, y=y, x=x, m=millions(n), p=p,
                      k=voxel_limit_words(ADD_BOX_BUDGET))


def whole_too_big_line(n) -> str:
    """``ADD_S1_WHOLE_TOO_BIG`` for a whole patch of ``n`` voxels, against
    the fill's ceiling (the engine's ``WHOLE_FILL_MAX_VOXELS``), named in
    words as :func:`over_limit_line` names a box's."""
    limit = int(_region.WHOLE_FILL_MAX_VOXELS)
    return ADD_S1_WHOLE_TOO_BIG.format(m=millions(int(n)),
                                       k=voxel_limit_words(limit))


def ridge_picture_minutes(m) -> float:
    """Rough minutes a ridge wall type takes to build the wall picture of
    a whole patch of ``m`` million voxels."""
    return max(0.0, float(m)) * RIDGE_PICTURE_SECONDS_PER_MILLION / 60.0


def ridge_memory_gb(m) -> float:
    """Rough peak memory (GB) of a ridge wall type's whole-patch Preview
    over ``m`` million voxels."""
    return max(0.0, float(m)) * RIDGE_GB_PER_MILLION


def whole_fill_minutes(m, wall_kind=None, *, picture_kept=False) -> float:
    """Rough minutes of a whole-patch Preview over ``m`` million voxels:
    the Membrane sheets figure, plus the wall picture for a ridge wall type
    (``RIDGE_WALL_KINDS``) unless that picture is kept already."""
    m = max(0.0, float(m))
    seconds = m * (FILL_SECONDS_PER_MILLION + FILL_SECONDS_GROWTH * m)
    minutes = seconds / 60.0
    if wall_kind in RIDGE_WALL_KINDS and not picture_kept:
        minutes += ridge_picture_minutes(m)
    return minutes


def wait_words(minutes) -> str:
    """The wait in words: "under a minute" under 0.75 minutes, "a minute"
    under 1.5, else whole minutes rounded half up."""
    minutes = float(minutes)
    if minutes < 0.75:
        return WAIT_UNDER_A_MINUTE
    if minutes < 1.5:
        return WAIT_A_MINUTE
    return WAIT_MINUTES.format(n=int(math.floor(minutes + 0.5)))


def wait_phrase(minutes) -> str:
    """``{t}`` of the wait notes: :func:`wait_words` with "about" in front,
    except "under a minute"."""
    words = wait_words(minutes)
    if words == WAIT_UNDER_A_MINUTE:
        return words
    return WAIT_ABOUT.format(w=words)


def live_update_seconds(n, wall_kind=None, growth=GROW_FLOOD, *,
                        picture_kept=False) -> float:
    """Rough seconds the slow update ("Always update as I click": the
    region filled whole) takes on a region of ``n`` voxels: the growth
    rule's figure (``growth`` as :func:`cells_growth` names it -- the fill
    to the walls, Spread, or a whole patch's fill with a background marker,
    whose figure is the whole Preview's), plus the wall picture of a ridge
    wall type (``RIDGE_WALL_KINDS``, ``RIDGE_BOX_PICTURE_SECONDS_PER_MILLION``)
    unless it is kept already.  A 4.2 million box with Vessel ridges walls
    was measured at 58.6 s (this gives about 58 s: "about a minute")."""
    m = max(0.0, float(n)) / 1e6
    if growth == GROW_SPREAD:
        seconds = m * SPREAD_SECONDS_PER_MILLION
    elif growth == GROW_PATCH:
        seconds = 60.0 * whole_fill_minutes(m, None)
    else:
        seconds = m * LIVE_FLOOD_SECONDS_PER_MILLION
    if wall_kind in RIDGE_WALL_KINDS and not picture_kept:
        # a whole patch's fill keeps the whole Preview's figure; a box's
        # picture was timed on a box
        seconds += (60.0 * ridge_picture_minutes(m) if growth == GROW_PATCH
                    else m * RIDGE_BOX_PICTURE_SECONDS_PER_MILLION)
    return seconds


def live_updating_line(seconds) -> str:
    """S1 while the slow update runs: ``ADD_S1_LIVE_UPDATING`` with no
    estimate or under ``LIVE_FEW_SECONDS``; else
    ``ADD_S1_LIVE_UPDATING_TIME`` with "about N s" (to 5 s) under 45 s, and
    from there :func:`wait_phrase`'s minutes."""
    if seconds is None or float(seconds) < LIVE_FEW_SECONDS:
        return ADD_S1_LIVE_UPDATING
    seconds = float(seconds)
    if seconds < 45.0:
        n = int(5 * math.floor(seconds / 5.0 + 0.5))
        t = WAIT_ABOUT.format(w=WAIT_SECONDS.format(n=n))
    else:
        t = wait_phrase(seconds / 60.0)
    return ADD_S1_LIVE_UPDATING_TIME.format(t=t)


def patch_slow_line(n, wall_kind=None, *, picture_kept=False,
                    box=False, find_nuclei=False, growth=None) -> str:
    """``ADD_NOTE_PATCH_SLOW_STEPS`` for a Cells whole patch of ``n`` voxels
    with the wall type ``wall_kind`` (``None``: the Membrane sheets
    figure); ``box``: ``ADD_NOTE_BOX_SLOW_STEPS`` for a Cells box of ``n``
    voxels.  ``growth`` (``GROW_FLOOD`` / ``GROW_SPREAD``): the figure of
    that growth rule (:func:`live_update_seconds`) instead of the whole
    patch's fill with a background marker.  ``find_nuclei``: Markers ▸
    Nuclei masks still to be found here -- the finder's figure
    (:func:`finder_minutes`) is added."""
    n = int(n)
    if growth == GROW_FLOOD:
        # a cold Preview of a box: its own rate, plus a ridge wall picture
        # still to be built
        m = n / 1e6
        seconds = m * BOX_FILL_SECONDS_PER_MILLION
        if wall_kind in RIDGE_WALL_KINDS and not picture_kept:
            seconds += m * RIDGE_BOX_PICTURE_SECONDS_PER_MILLION
        minutes = seconds / 60.0
    elif growth == GROW_SPREAD:
        minutes = live_update_seconds(n, wall_kind, growth,
                                      picture_kept=picture_kept) / 60.0
    else:
        minutes = whole_fill_minutes(n / 1e6, wall_kind,
                                     picture_kept=picture_kept)
    if find_nuclei:
        minutes += finder_minutes(n / 1e6)
    t = wait_phrase(minutes)
    tpl = ADD_NOTE_BOX_SLOW_STEPS if box else ADD_NOTE_PATCH_SLOW_STEPS
    return tpl.format(m=millions(n), t=t)


def anywhere_minutes(m) -> float:
    """Rough minutes of a Nuclei ▸ Anywhere in the box Preview over ``m``
    million voxels (the finder, then the finish: ``ANYWHERE_SECONDS_PER_MILLION``)."""
    return max(0.0, float(m)) * ANYWHERE_SECONDS_PER_MILLION / 60.0


def finder_minutes(m) -> float:
    """Rough minutes of the nuclei finder alone over ``m`` million voxels
    (Cells ▸ Nuclei masks: ``FINDER_SECONDS_PER_MILLION``)."""
    return max(0.0, float(m)) * FINDER_SECONDS_PER_MILLION / 60.0


def nuclei_patch_slow_line(n, live=False, anywhere=False) -> str:
    """``ADD_NOTE_PATCH_SLOW_PLAIN`` for a Nuclei whole patch of ``n``
    voxels (no figure for the wait); ``live`` -- the markers updating as
    you click -- ``ADD_NOTE_PATCH_SLOW_LIVE``; ``anywhere`` -- Anywhere in
    the box, whose wait was timed -- ``ADD_NOTE_PATCH_SLOW_STEPS`` with
    :func:`anywhere_minutes`."""
    if anywhere and not live:
        return ADD_NOTE_PATCH_SLOW_STEPS.format(
            m=millions(int(n)),
            t=wait_phrase(anywhere_minutes(int(n) / 1e6)))
    tpl = ADD_NOTE_PATCH_SLOW_LIVE if live else ADD_NOTE_PATCH_SLOW_PLAIN
    return tpl.format(m=millions(int(n)))


def ridges_whole_line(kind, n, wall_kind) -> str:
    """``ADD_NOTE_RIDGES_WHOLE`` for the ridge wall type ``wall_kind``
    (named ``kind`` on the page) on a whole patch of ``n`` voxels whose
    picture is still to be built: the whole Preview's wait; the
    ``_MEMORY`` form from ``ADD_RIDGE_MEMORY_NOTE_GB`` on."""
    m = int(n) / 1e6
    t = wait_phrase(whole_fill_minutes(m, wall_kind, picture_kept=False))
    gb = ridge_memory_gb(m)
    if gb >= ADD_RIDGE_MEMORY_NOTE_GB:
        return ADD_NOTE_RIDGES_WHOLE_MEMORY.format(
            kind=kind, t=t, g=int(math.floor(gb + 0.5)))
    return ADD_NOTE_RIDGES_WHOLE.format(kind=kind, t=t)


def cells_growth(params) -> str:
    """The growth rule a Cells Preview hands the engine: on a whole patch,
    Spread from each marker when Grow by says so (``GROW_SPREAD``, grown in
    tiles), else the fill to the walls with a background marker
    (``GROW_PATCH``) -- or, with ``params["box_rule"]`` (Click markers on a
    whole patch with "Update as I click" on and the fill to the walls), the
    fill to the walls with no background marker, as in a box
    (``GROW_FLOOD``); on a box the Grow by pick (the flood by default)."""
    p = params or {}
    if bool(p.get("whole_patch", False)):
        if bool(p.get("box_rule", False)):
            return GROW_FLOOD
        if p.get("growth") == GROW_SPREAD:
            return GROW_SPREAD
        return GROW_PATCH
    return p.get("growth") or GROW_FLOOD


def over_limit_line(shape) -> str:
    """``ADD_S1_OVER_LIMIT`` for a Cells box of ``shape``, against
    ``ADD_BOX_BUDGET`` (the whole patch's ceiling too, see
    :func:`whole_too_big_line`)."""
    return ADD_S1_OVER_LIMIT.format(
        m=millions(int(np.prod([int(v) for v in shape]))),
        k=voxel_limit_words(ADD_BOX_BUDGET))


def two_tail(s) -> str:
    """``ADD_S1_TWO_TAIL`` for ``s`` cells, or ``""`` for none."""
    s = int(s or 0)
    if s <= 0:
        return ""
    return count_phrase(s, ADD_S1_TWO_TAIL_ONE, ADD_S1_TWO_TAIL, s=s)


def found_line(n, *, k_edge=0, k_small=0, k_both=None, k_plain=None,
               n_two=0) -> str:
    """``ADD_S1_FOUND`` + the unticked tail.  ``k_edge`` / ``k_small`` are
    the sizes of the UNTICKED rows at the box edge / small; an object in
    both counts once (``k_both``).  ``k_plain`` (the live unticked count
    once the user has unticked a row that started ticked) wins and drops
    the reason (B3.4).  ``n_two`` (cells holding two nuclei) adds
    ``ADD_S1_TWO_TAIL`` LAST, after the unticked tail (the mockup's
    "46 new objects found · 2 left unticked (…) · 3 cells hold two
    nuclei")."""
    head = count_phrase(n, ADD_S1_FOUND_ONE, ADD_S1_FOUND)
    tail = two_tail(n_two)
    tail = (ADD_NOTE_COUNTS_SEP + tail) if tail else ""
    if k_plain is not None:
        k = int(k_plain)
        if k > 0:
            head = head + ADD_S1_UNTICKED_PLAIN.format(k=k)
        return head + tail
    k = int(k_edge) + int(k_small) - int(k_both or 0)
    if k <= 0:
        return head + tail
    if k_edge and k_small:
        why = ADD_WHY_BOTH
    elif k_edge:
        why = ADD_WHY_EDGE
    else:
        why = ADD_WHY_SMALL
    return head + ADD_S1_UNTICKED.format(k=k, why=why) + tail


def unticked_why(k_edge=0, k_small=0) -> str | None:
    """The reason rows started unticked: at the box edge, small, or both
    (``None`` for neither)."""
    if k_edge and k_small:
        return ADD_WHY_BOTH
    if k_edge:
        return ADD_WHY_EDGE
    if k_small:
        return ADD_WHY_SMALL
    return None


def found_none_ticked_line(n, why, n_two=0) -> str:
    """``ADD_S1_FOUND`` when no row is ticked because every one started
    unticked: the count, the cells holding two nuclei, then
    ``ADD_S1_NONE_TICKED`` (``_ONE`` for one object), which names All --
    the instruction ends the line, so here the two-nuclei tail comes
    before it (the reverse of :func:`found_line`)."""
    head = count_phrase(n, ADD_S1_FOUND_ONE, ADD_S1_FOUND)
    tail = two_tail(n_two)
    if tail:
        head = head + ADD_NOTE_COUNTS_SEP + tail
    return head + count_phrase(n, ADD_S1_NONE_TICKED_ONE,
                               ADD_S1_NONE_TICKED, why=why)


def _none_ticked(cand) -> bool:
    """A result is up and no row is ticked."""
    c = tuple(cand)
    return int(c[0]) > 0 and int(c[4]) == 0


def added_line(n, voxels, *, has_selection=True, n_selected=None,
               n_two=0) -> str:
    """``ADD_S1_ADDED`` + what Refine ▸ needs: nothing with exactly one row
    selected, ``ADD_S1_SELECT_ONE`` with none, ``ADD_S1_SELECT_JUST_ONE``
    with more than one (B2 fix pass U-M3).  ``has_selection`` is the old
    boolean form.  ``n_two`` (cells that got two of the added nuclei) adds
    ``ADD_S1_ADDED_TWO_TAIL``, which ends its own sentence."""
    head = count_phrase(n, ADD_S1_ADDED_ONE, ADD_S1_ADDED, v="{:,}".format(int(voxels)))
    s = int(n_two or 0)
    if s > 0:
        head = head + ADD_NOTE_COUNTS_SEP + count_phrase(
            s, ADD_S1_ADDED_TWO_TAIL_ONE, ADD_S1_ADDED_TWO_TAIL, s=s)
    if n_selected is None:
        n_selected = 1 if has_selection else 0
    n_selected = int(n_selected)
    if n_selected == 1:
        return head
    # two sentences: the head ends with a period (live-shot fix F5); the
    # two-nuclei tail already ends with one
    return head + ("" if s > 0 else ".") + (
        ADD_S1_SELECT_ONE if n_selected == 0 else ADD_S1_SELECT_JUST_ONE)


def adding_line(n) -> str:
    """``ADD_S1_ADDING`` for ``n`` objects (the singular at 1)."""
    return count_phrase(n, ADD_S1_ADDING_ONE, ADD_S1_ADDING)


def undoing_line(n) -> str:
    """``ADD_S1_UNDOING_ALL`` for ``n`` objects (the singular at 1)."""
    return count_phrase(n, ADD_S1_UNDOING_ALL_ONE, ADD_S1_UNDOING_ALL)


def inside_count_line(c, *, whole_patch=False) -> str:
    """``ADD_INSIDE_COUNT`` for ``c`` cells, ``_PENDING`` for ``None``;
    the ``_PATCH`` words on the whole patch."""
    if whole_patch:
        if c is None:
            return ADD_INSIDE_COUNT_PATCH_PENDING
        return count_phrase(int(c), ADD_INSIDE_COUNT_PATCH_ONE,
                            ADD_INSIDE_COUNT_PATCH, c=int(c))
    if c is None:
        return ADD_INSIDE_COUNT_PENDING
    return count_phrase(int(c), ADD_INSIDE_COUNT_ONE, ADD_INSIDE_COUNT,
                        c=int(c))


def look_inside_text(name, target="", *, curated=True) -> str:
    """One "Inside each cell of run ▸ ..." item: the run's name, its target
    after " · " when it has one, and the not-curated suffix."""
    target = str(target or "").strip()
    text = ADD_LOOK_INSIDE.format(
        name=str(name), target=(" · " + target) if target else "")
    if not curated:
        text += ADD_ITEM_RAW_SUFFIX
    return text


def target_is_nuclei(target) -> bool:
    """True when a run's free-text target says nuclei."""
    return "nucle" in str(target or "").strip().lower()


def candidate_chips(pid, *, edge_ids=(), small_ids=(), flags=None) -> list:
    """A row's chips in the page's order: two in a cell, cut at edge,
    edge, small."""
    pid = int(pid)
    words = set((flags or {}).get(pid, ()))
    out = []
    if "two" in words:
        out.append(ADD_CHIP_TWO)
    if "cut" in words:
        out.append(ADD_CHIP_CUT)
    if pid in {int(i) for i in edge_ids}:
        out.append("edge")
    if pid in {int(i) for i in small_ids}:
        out.append("small")
    return out


def candidate_order(ids, *, edge_ids=(), small_ids=(), flags=None) -> list:
    """The rows sorted: two in a cell, cut at edge, edge, small, then
    by id.  Without derive flags this is the old edge / small / id order."""
    edge = {int(i) for i in edge_ids}
    small = {int(i) for i in small_ids}
    flags = flags or {}

    def key(i):
        i = int(i)
        words = set(flags.get(i, ()))
        if "two" in words:
            rank = 0
        elif "cut" in words:
            rank = 1
        elif i in edge:
            rank = 2
        elif i in small:
            rank = 3
        else:
            rank = 4
        return (rank, i)

    return sorted((int(i) for i in ids), key=key)


def derive_tip(pid, *, flags=None, owner=None, two_in_cell=None) -> str:
    """The tooltip a derive chip adds to its row ("" without one)."""
    pid = int(pid)
    words = set((flags or {}).get(pid, ()))
    cell = (owner or {}).get(pid)
    tips = []
    if "two" in words and cell is not None:
        k = len((two_in_cell or {}).get(cell, ())) or 2
        tips.append(ADD_TIP_TWO.format(cell=cell, k=k))
    if "cut" in words and cell is not None:
        tips.append(ADD_TIP_CUT.format(cell=cell))
    return " ".join(tips)


def cells_with_two(ids, owner) -> int:
    """How many cells own two or more of ``ids`` (by ``owner``)."""
    per = {}
    for i in ids:
        c = (owner or {}).get(int(i))
        if c is not None:
            per[c] = per.get(c, 0) + 1
    return sum(1 for k in per.values() if k >= 2)


def one_channel_note(*, spread, shrink, whole=False) -> str:
    """``ADD_AMBER_ONE_CHANNEL_RUN`` for the page as it is: Spread already
    picked or not, Bridge gaps on the page (the shrink-wrap) or not.  A
    whole patch (``whole``) reads the same: Grow by, Wall cost and Bridge
    gaps are under Advanced there too."""
    tpl = (ADD_AMBER_ONE_CHANNEL_RUN_SPREAD if spread
           else ADD_AMBER_ONE_CHANNEL_RUN)
    return tpl.format(bridge=ADD_AMBER_ONE_CHANNEL_BRIDGE if shrink else "")


def no_numba_sentence(why) -> str:
    """Why Spread cannot run here, in the page's words."""
    return ADD_NO_NUMBA_CACHE if why == NUMBA_CACHE_KEY else ADD_NO_NUMBA


def whole_needs_numba_sentence(why) -> str:
    """Why a Cells whole patch cannot be filled in this install."""
    return (ADD_WHOLE_NEEDS_NUMBA_CACHE if why == NUMBA_CACHE_KEY
            else ADD_WHOLE_NEEDS_NUMBA)


def numba_answer():
    """``(key, engine sentence)``: whether Spread from each marker can run
    here -- ``NUMBA_OK`` / ``NUMBA_MISSING_KEY`` / ``NUMBA_CACHE_KEY`` (its
    cache folder cannot be written).  Imports numba (about a second the
    first time): call it on a worker.  When it can, the click-by-click
    Cells fill's flood is compiled here too (``incremental.warm_kernel``,
    once per process), so the first Cells click never pays it; until then
    that fill leaves the run to the full one by itself, and a live click on
    a region over ``ADD_LIVE_FULL_RUN_MAX_VOXELS`` waits
    (:data:`LIVE_PREPARING`).  However it ends, the click fill is settled
    for the process (:func:`click_fill_compiling` is False from here): a
    flood still not compiled then cannot be, and is a real fallback."""
    global _CLICK_FILL_SETTLED
    try:
        return _numba_answer()
    finally:
        _CLICK_FILL_SETTLED = True


def _numba_answer():
    from .reshape import compete

    try:
        compete._kernels()
    except ImportError as exc:
        return NUMBA_MISSING_KEY, str(exc)
    except RuntimeError as exc:
        return NUMBA_CACHE_KEY, str(exc)
    from .reshape import incremental

    if not incremental.kernel_is_warm():
        try:
            incremental.warm_kernel()
        except Exception as exc:  # noqa: BLE001 - the full run answers
            print("[ADD] click-by-click fill not compiled: {!r}".format(exc))
    return NUMBA_OK, ""


def run_count_pending_line(*, whole_patch=False) -> str:
    return (ADD_RUN_COUNT_PATCH_PENDING if whole_patch
            else ADD_RUN_COUNT_PENDING)


def run_count_line(n_in_box, n_markers, *, whole_patch=False) -> str:
    if whole_patch:
        tpl = (ADD_RUN_COUNT_PATCH_ONE if int(n_markers) == 1
               else ADD_RUN_COUNT_PATCH)
    else:
        tpl = ADD_RUN_COUNT_ONE if int(n_markers) == 1 else ADD_RUN_COUNT
    return tpl.format(n=int(n_in_box), m=int(n_markers))


def outline_drops_share(n_outside, n_left):
    """The share of the markers the tissue outline drops: ``n_outside`` of
    the ``n_outside + n_left`` that reached it, or ``None`` with none."""
    total = int(n_outside or 0) + int(n_left or 0)
    if total <= 0:
        return None
    return float(int(n_outside or 0)) / float(total)


def outline_drops_line(n_outside, n_left, mode=OUTLINE_SHRINK) -> str:
    """``ADD_AMBER_OUTLINE_DROPS`` (``_CLEARED`` for the outline of the
    objects you cleared) when more than ``ADD_OUTLINE_DROPS_SHARE`` of the
    markers lie outside the outline, else ``""``."""
    share = outline_drops_share(n_outside, n_left)
    if share is None or share <= ADD_OUTLINE_DROPS_SHARE:
        return ""
    tpl = (ADD_AMBER_OUTLINE_DROPS_CLEARED if mode == OUTLINE_CLEARED
           else ADD_AMBER_OUTLINE_DROPS)
    return tpl.format(p=int(round(100.0 * share)))


def outline_item_text(key) -> str:
    """The Tissue outline item's words for ``key``."""
    return {OUTLINE_CLEARED: ADD_OUTLINE_CLEARED,
            OUTLINE_SHRINK: ADD_OUTLINE_SHRINK}.get(key, ADD_OUTLINE_NONE)


def outline_help(key) -> str:
    """The help line under Tissue outline for ``key``."""
    return {OUTLINE_CLEARED: ADD_HELP_OUTLINE_CLEARED,
            OUTLINE_SHRINK: ADD_HELP_OUTLINE_SHRINK}.get(
                key, ADD_HELP_OUTLINE_NONE)


def nuclei_outline_help(key) -> str:
    """The help line under the Nuclei card's Tissue outline for ``key``."""
    return {OUTLINE_CLEARED: ADD_HELP_NUC_OUTLINE_CLEARED,
            OUTLINE_SHRINK: ADD_HELP_NUC_OUTLINE_SHRINK}.get(
                key, ADD_HELP_NUC_OUTLINE_NONE)


def nuclei_outline_summary(key) -> str:
    """The Nuclei Tissue outline card's chevron title, e.g. "Tissue
    outline: shrink-wrap"."""
    word = {OUTLINE_CLEARED: ADD_SUM_NUC_OUTLINE_CLEARED,
            OUTLINE_SHRINK: ADD_SUM_NUC_OUTLINE_SHRINK}.get(
                key, ADD_SUM_NUC_OUTLINE_NONE)
    return ADD_SUM_NUC_OUTLINE.format(choice=word)


def advanced_fill_summary(include_membrane, live=False) -> str:
    """The Advanced summary's growth part on a Cells whole patch; ``live``
    -- filled as you click, the way a box is (no membrane part)."""
    if live:
        return ADD_SUM_ADVANCED_FILL_LIVE
    return (ADD_SUM_ADVANCED_FILL if include_membrane
            else ADD_SUM_ADVANCED_FILL_NO_MEMBRANE)


def advanced_grow_summary(wall_cost, growth_limit, include_membrane) -> str:
    """The Advanced summary's growth part (Spread only)."""
    tpl = (ADD_SUM_ADVANCED_GROW if include_membrane
           else ADD_SUM_ADVANCED_GROW_NO_MEMBRANE)
    return tpl.format(w="{:g}".format(float(wall_cost)),
                      l="{:g}".format(float(growth_limit)))


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


def nuclei_tick_one_line(n) -> str:
    """Nuclei ▸ At the markers I place: every one of ``n`` markers is
    unticked."""
    return count_phrase(n, ADD_S1_TICK_ONE_NUCLEUS_ONE,
                        ADD_S1_TICK_ONE_NUCLEUS)


def markers_all_kept_line(n) -> str:
    """Every one of ``n`` markers sat inside a kept object at Preview."""
    return count_phrase(n, ADD_S1_MARKERS_ALL_KEPT_ONE,
                        ADD_S1_MARKERS_ALL_KEPT)


def marker_why_tip(why):
    """The marker row's tooltip for why it made nothing at the last
    Preview (``"kept"`` / ``"dim"`` / ``"small"`` / ``"outline"``), or
    ``None``."""
    return {"kept": ADD_MARKER_ABSORBED_TIP,
            "dim": ADD_MARKER_DIM_TIP,
            "small": ADD_MARKER_SMALL_TIP,
            "outline": ADD_MARKER_OUTLINE_TIP}.get(why)


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


def object_row_text(label_id, zyx, n_seeds=1) -> str:
    """A click-marker list row, one per object: the marker row
    (:func:`marker_row_text`) for an object of one marker, "Object 412  ·
    3 seeds" for an object grown from several."""
    if int(n_seeds) <= 1:
        return marker_row_text(label_id, zyx)
    return "Object {}  ·  {}".format(
        int(label_id), ADD_ROW_SEEDS.format(n=int(n_seeds)))


def candidate_row_text(label_id, size=None, n_seeds=1) -> str:
    """A result row: "Object 412  ·  1,234", with the seed count between
    for an object grown from several markers ("Object 412  ·  3 seeds  ·
    1,234"; the delegate draws the size right-aligned in mono)."""
    text = "Object {}".format(int(label_id))
    if int(n_seeds) > 1:
        text += "  ·  " + ADD_ROW_SEEDS.format(n=int(n_seeds))
    if size is not None:
        text += "  ·  {:,}".format(int(size))
    return text


def seed_status_words(whys) -> list:
    """An object's status words from what its seeds made at the last
    result (``"apart"`` / ``"dim"`` / ``"kept"`` / ``None`` per seed), each
    word once, in that order: not connected, nothing bright here, inside a
    kept object."""
    got = {str(w) for w in whys if w}
    return [word for key, word in (("apart", ADD_ROW_NOT_CONNECTED),
                                   ("dim", ADD_ROW_DIM),
                                   ("kept", ADD_ROW_KEPT)) if key in got]


def status_tip(words) -> str:
    """The row tooltip's line for its status words, each with its long
    form ("Seeds that made nothing: apart (not connected)."), or ``""``."""
    parts = ["{} ({})".format(w, ADD_ROW_STATUS_LONG[w]) for w in words
             if w in ADD_ROW_STATUS_LONG]
    if not parts:
        return ""
    return ADD_ROW_STATUS_TIP.format(words=", ".join(parts))


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


def walls_summary(*, channel="", wall_kind="", prep=(), outline="") -> str:
    """The Walls chevron's summary: channel · wall type · the ticked
    prepare words, e.g. "Walls: membrane · Membrane sheets · even out
    depth" (U-M5c); with a tissue outline chosen (``outline`` =
    ``"shrink"`` / ``"cleared"``) its word comes last."""
    parts = [p for p in (channel, wall_kind) if p]
    if prep:
        parts.append(", ".join(prep))
    word = {OUTLINE_SHRINK: ADD_SUM_OUTLINE_SHRINK,
            OUTLINE_CLEARED: ADD_SUM_OUTLINE_CLEARED}.get(outline or "")
    if word:
        parts.append(word)
    if not parts:
        return ADD_SUM_WALLS
    return ADD_SUM_WALLS + ": " + ADD_SUM_SEP.join(parts)


def help_clicks(radius_um) -> str:
    return ADD_HELP_CLICKS.format(d="{:g}".format(round(2.0 * float(radius_um))))


def help_run(erode_um) -> str:
    return ADD_HELP_RUN.format(um="{:g}".format(float(erode_um)))


def counts_note(counts, *, source=None, whole_patch=False,
                preset=None) -> str:
    """The grey counts note.  Zero parts are left out;
    the "markers sat on kept objects" part is OMITTED for the run-cores
    source (152-338 per box by design).  ``""`` when every count is 0.
    ``whole_patch``: a part that names the box edge says the patch's
    words instead.  ``preset="nuclei"``: the tissue outline's part says
    nuclei, not markers (same place in the order)."""
    parts = []
    for key, many, one in _COUNT_PARTS:
        if key == "n_markers_over_kept" and source == SOURCE_RUN:
            continue
        if whole_patch and key in _COUNT_PARTS_WHOLE:
            many, one = _COUNT_PARTS_WHOLE[key]
        if preset == "nuclei" and key in _COUNT_PARTS_NUCLEI:
            many, one = _COUNT_PARTS_NUCLEI[key]
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
    ``budget`` voxels (the tissue outline samples with the same rule)."""
    return sample_stride(shape, budget)


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
    ``info_pending``; ``run_id`` (the run picked under Markers, ``None``
    for a blank combo), ``has_other_run``, ``n_run_markers`` (the box-info
    run count's markers, ``None`` = not counted) and ``n_run_found`` (the
    run's objects it found here), ``n_outside_outline`` and ``outline`` (the
    tissue outline asked for), ``dark_outside`` (the dark-space tick),
    ``empty_share``, ``place_markers_on``."""
    s = dict(state or {})
    if not s.get("has_box"):
        # Whole patch sits beside Place box on both presets, never grey: the
        # idle line names it whatever the choices
        if (s.get("preset") == "cells"
                and s.get("numba_ok", True) is False):
            # a Cells whole patch needs Spread, which cannot run here
            return ADD_S1_IDLE_BOX
        return ADD_S1_IDLE
    cells = s.get("preset") == "cells"
    shape = s.get("box_shape")
    whole = bool(s.get("whole_patch"))
    if (not cells and whole and shape is not None
            and s.get("look_for") not in (LOOK_INSIDE, LOOK_MARKERS)):
        # Anywhere runs on a whole patch as on a box (Stop acts between its
        # steps); its peak memory (about 55 bytes a voxel) holds it to the
        # Cells limit, said as a whole patch's size
        n = int(np.prod([int(v) for v in shape]))
        if n > int(_region.WHOLE_FILL_MAX_VOXELS):
            return whole_too_big_line(n)
    if cells and shape is not None and not whole:
        n = int(np.prod([int(v) for v in shape]))
        if n > int(s.get("box_budget", ADD_BOX_BUDGET)):
            return over_limit_line(shape)
    if cells and whole:
        # a whole patch is filled to the walls in one go: the growth engine
        # must be able to run here and the patch must be under the fill's
        # ceiling (refused before anything is read).  The spacing refuses
        # nothing: at the never-set default it runs as a box does and the
        # amber note says so
        if s.get("numba_ok", True) is False:
            return whole_needs_numba_sentence(s.get("numba_why"))
        if shape is not None:
            n = int(np.prod([int(v) for v in shape]))
            if n > int(_region.WHOLE_FILL_MAX_VOXELS):
                return whole_too_big_line(n)
        source = s.get("source", SOURCE_RUN)
        if source == SOURCE_RUN and not s.get("run_id"):
            # the markers of a whole patch come from any source a box takes;
            # a blank From (a run is never picked for the user there) waits
            # for a pick
            return (ADD_S1_WHOLE_PICK_RUN if s.get("has_other_run")
                    else ADD_S1_WHOLE_NO_RUN)
    if not cells:
        if not s.get("channel"):
            return ADD_S1_PICK_CHANNEL
        if s.get("look_for") == LOOK_MARKERS:
            # one click per nucleus: a marker to place, then one ticked
            n_clicks = int(s.get("n_clicks", 0) or 0)
            if n_clicks == 0:
                return (ADD_S1_NO_MARKERS_ON if s.get("place_markers_on")
                        else ADD_S1_NO_MARKERS)
            if (s.get("n_ticked") is not None
                    and int(s.get("n_ticked") or 0) == 0):
                return nuclei_tick_one_line(n_clicks)
        if s.get("look_for") != LOOK_INSIDE:
            # a Tissue outline chosen (Anywhere or the markers): it could not
            # be measured (its plain reason), or the Preview waits for it
            if s.get("outline_error"):
                return str(s["outline_error"])
            if s.get("outline_pending"):
                return ADD_OUTLINE_PENDING
            return None
        if s.get("look_for") == LOOK_INSIDE:
            # Inside each cell: row 11 on the Nuclei preset (the other run),
            # then a refused channel and a box with no cell of that run (the
            # spacing refuses nothing: the amber note says the default)
            if s.get("run_state") == "loading":
                return ADD_S1_RUN_LOADING
            if s.get("run_state") == "refused":
                return ADD_S1_INSIDE_UNUSABLE
            if s.get("levels_error"):
                # the patch's levels refused this channel (not the run)
                return ADD_S1_INSIDE_CHANNEL_UNUSABLE
            count = s.get("inside_count")
            if count is not None and int(count) == 0:
                return (ADD_S1_INSIDE_NO_CELLS_PATCH if s.get("whole_patch")
                        else ADD_S1_INSIDE_NO_CELLS)
            if s.get("info_pending"):
                return measuring_line(s)
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
        return (ADD_S1_NO_MARKERS_ON if s.get("place_markers_on")
                else ADD_S1_NO_MARKERS)
    if (source == SOURCE_CLICKS and s.get("n_ticked") is not None
            and int(s.get("n_ticked") or 0) == 0):
        # row 12b: markers placed, none ticked
        return tick_one_marker_line(int(s.get("n_clicks", 0) or 0))
    if s.get("outline_error"):
        # the outline could not be measured (its plain reason)
        return str(s["outline_error"])
    if s.get("outline_pending"):
        # the tissue outline is measured once per patch on the box-info
        # worker; the Preview waits for it
        return ADD_OUTLINE_PENDING
    if s.get("info_pending"):
        return measuring_line(s)
    if (source == SOURCE_RUN and s.get("n_run_markers") is not None
            and int(s["n_run_markers"]) == 0):
        return run_no_markers_line(s)
    return None


def run_no_markers_line(state) -> str:
    """Why the run count left no marker, in what is true for this box (or
    patch): the tissue outline's own sentence when it dropped them; no
    object of the run here; objects kept here (dark space, over an object,
    too small); nothing kept here (dark space, too small -- naming the
    dark-space tick while it is on and no outline decides instead)."""
    s = dict(state or {})
    whole = bool(s.get("whole_patch"))
    outline = s.get("outline")
    if (int(s.get("n_outside_outline") or 0) > 0
            and outline in (OUTLINE_SHRINK, OUTLINE_CLEARED)):
        return (ADD_S1_NO_MARKERS_OUTLINE_CLEARED
                if outline == OUTLINE_CLEARED else ADD_S1_NO_MARKERS_OUTLINE)
    found = s.get("n_run_found")
    if found is not None and int(found) == 0:
        return ADD_S1_RUN_NONE_HERE_PATCH if whole else ADD_S1_RUN_NONE_HERE
    empty = s.get("empty_share")
    if empty is None or float(empty) < 1.0:
        # objects are kept here (or the share is unknown)
        return (ADD_S1_RUN_NO_MARKERS_PATCH if whole
                else ADD_S1_RUN_NO_MARKERS)
    if s.get("dark_outside") and outline in (None, OUTLINE_NONE):
        tpl = (ADD_S1_RUN_NO_MARKERS_EMPTY_PATCH if whole
               else ADD_S1_RUN_NO_MARKERS_EMPTY)
        return tpl.format(dark=ADD_DARK_OUTSIDE_LABEL)
    return (ADD_S1_RUN_NO_MARKERS_EMPTY_PLAIN_PATCH if whole
            else ADD_S1_RUN_NO_MARKERS_EMPTY_PLAIN)


def whole_measures_nothing(preset, source, look_for) -> bool:
    """True when a whole patch with these choices can be neither previewed
    nor measured: Cells with no marker source the page knows.  Every Look
    for nuclei runs on a whole patch as on a box.  The box-info gather
    reads nothing for it, and the page's Auto lines say nothing is being
    measured -- one rule for both."""
    if preset == "cells":
        return source not in (SOURCE_RUN, SOURCE_CLICKS, SOURCE_NUCLEI)
    return False


def measuring_line(state) -> str:
    """"Measuring the box…", or "…the patch…" on a whole patch."""
    return (ADD_S1_MEASURING_PATCH if (state or {}).get("whole_patch")
            else ADD_S1_MEASURING)


def s1_sentence(state) -> str:
    """The one S1 sentence.  Adds to :func:`preview_block`'s keys:
    ``has_patch``, ``transient`` (a sentence or ``None``), ``busy``,
    ``review`` (``(n, voxels, has_selection)`` or ``None``), ``candidates``
    (``(n, k_edge, k_small, k_both, n_ticked)`` or ``None``),
    ``empty_share``.  ``n_clicks`` / ``n_ticked`` count objects, not
    markers; the current object is named by its bold row and the hint
    line, never here."""
    s = dict(state or {})
    if not s.get("has_patch", True):
        return ADD_S1_NO_PATCH
    if s.get("transient"):
        return str(s["transient"])
    if s.get("busy"):
        return s.get("busy_text") or (ADD_S1_WORKING_PATCH
                                      if s.get("whole_patch")
                                      else ADD_S1_WORKING)
    if s.get("live_updating"):
        # "Always update as I click": the region filled whole after the
        # clicks settled, on a live run (never busy: nothing greys); the
        # time estimated at its launch
        return live_updating_line(s.get("live_updating_seconds"))
    review = s.get("review")
    if review is not None:
        n, v, n_sel = review
        # a bool (the old form) or a count of selected rows
        return added_line(n, v, n_selected=int(n_sel),
                          n_two=int(s.get("review_two", 0) or 0))
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
        why = unticked_why(k_edge, k_small)
        if _none_ticked(cand) and k_plain is None and why is not None:
            # every row started unticked: the count, and what to press
            return found_none_ticked_line(n, why,
                                          int(s.get("n_two", 0) or 0))
        return found_line(n, k_edge=k_edge, k_small=k_small, k_both=k_both,
                          k_plain=k_plain, n_two=int(s.get("n_two", 0) or 0))
    block = preview_block(s)
    if block is not None:
        return block
    cells = s.get("preset") == "cells"
    markers = not cells and s.get("look_for") == LOOK_MARKERS
    row = live_row_reason(s)
    if row is not None:
        whole_patch = bool(s.get("whole_patch"))
        if row == LIVE_ROW_OVER:
            return (ADD_S1_LIVE_OVER_GATE_PATCH_SIZE if whole_patch
                    else ADD_S1_LIVE_OVER_GATE_BOX).format(
                n=voxel_limit_words(ADD_LIVE_CELLS_MAX_VOXELS))
        if row == LIVE_ROW_SPACING:
            return (ADD_S1_LIVE_STEPPED_ASIDE_SPACING_PATCH if whole_patch
                    else ADD_S1_LIVE_STEPPED_ASIDE_SPACING_BOX)
        if row == LIVE_ROW_ASIDE:
            return (ADD_S1_LIVE_OVER_GATE_PATCH if whole_patch
                    else ADD_S1_LIVE_STEPPED_ASIDE_BOX)
        return ADD_S1_LIVE_PREPARING
    if not cells and not markers and s.get("empty_share") is None:
        # the empty share is measured on the box-info worker; Nuclei's
        # Preview does not wait for it (it is not row 13), S1 does.  The
        # markers' line counts the markers and names no share
        return measuring_line(s)
    clicks = (cells and s.get("source") == SOURCE_CLICKS) or markers
    n_clicks = int(s.get("n_clicks", 0) or 0) if clicks else None
    return box_line(
        s.get("box_shape") or (0, 0, 0), preset=s.get("preset"),
        empty_share=s.get("empty_share"),
        n_clicks=n_clicks, n_ticked=s.get("n_ticked"),
        whole_patch=bool(s.get("whole_patch")),
        look_for=s.get("look_for"))


#: S1 row 13.5's reasons (:func:`live_row_reason`): over the live limit;
#: stepped aside for the flat spots at the never-set spacing; stepped
#: aside for anything else (walls or a Grow by the click fill does not
#: redo, its flat spots at a set spacing, a failure); the click fill still
#: being prepared
LIVE_ROW_OVER = "over"
LIVE_ROW_SPACING = "spacing"
LIVE_ROW_ASIDE = "aside"
LIVE_ROW_PREPARING = "preparing"


def live_row_reason(state):
    """Why a Cells ▸ Click markers click is not run as you click here, as
    S1 row 13.5 and the line under the tick both say it (one decision for
    the two rows), or ``None``: the tick on, "Always update as I click"
    off (with it on only the compile wait holds -- every region up to the
    Preview limit is updated, slowly where the click fill cannot run), and one
    rule for a box and a whole patch -- a region over the live limit
    (named); a whole patch whose walls the click fill does not redo or
    that it could not take (``live_whole_ok`` False); a box over the
    full-run size the click fill does not or could not take
    (``live_box_ok`` False); a region that is run as you click but whose
    click fill is still being compiled (``fill_pending``).  A step aside
    for the flat spots at the never-set spacing (``flat_spots`` and
    ``default_spacing``) is told apart: the spacing may change it."""
    s = state or {}
    if not (s.get("live_on") and s.get("preset") == "cells"
            and s.get("source") == SOURCE_CLICKS):
        return None
    if s.get("live_always"):
        # "Always update as I click": no region steps aside, but a click
        # fill still being compiled is a wait with the tick on too
        return LIVE_ROW_PREPARING if s.get("fill_pending") else None
    shape = s.get("box_shape")
    if shape and int(np.prod([int(v) for v in shape])) > int(
            ADD_LIVE_CELLS_MAX_VOXELS):
        return LIVE_ROW_OVER
    if s.get("whole_patch"):
        aside = not s.get("live_whole_ok")
    else:
        aside = not s.get("live_box_ok", True)
    if aside:
        if s.get("flat_spots") and s.get("default_spacing"):
            return LIVE_ROW_SPACING
        return LIVE_ROW_ASIDE
    if s.get("fill_pending"):
        return LIVE_ROW_PREPARING
    return None


def _s1_outranks_live_row(s) -> bool:
    """True when S1 says a passing state instead of row 13.5: a transient,
    busy, an update running, the review, a result on screen.  (A reason
    Preview is grey -- a marker to place -- keeps the line: it says what
    the clicks will meet.)"""
    if s.get("transient") or s.get("busy") or s.get("live_updating"):
        return True
    return s.get("review") is not None or s.get("candidates") is not None


def live_tick_note(state) -> str:
    """The grey line under "Update as I click": S1 row 13.5's reason
    (:func:`live_row_reason`), short, naming the control that changes it
    (``""``: no line).  A step aside is told apart by ``live_aside``
    (``"walls"`` / ``"grow"``: what the click fill does not redo) and
    ``flat_spots``; any other is the click fill that could not run."""
    s = state or {}
    if s.get("live_always"):
        # the tick lifts every reason this line gives (the compile wait
        # S1 says on its own)
        return ""
    if _s1_outranks_live_row(s):
        # S1 says a passing state first (a transient, busy, an update, the
        # review, a result): the line under the tick waits for it to go
        return ""
    shape = s.get("box_shape")
    whole = bool(s.get("whole_patch"))
    if shape:
        n = int(np.prod([int(v) for v in shape]))
        limit = (int(_region.WHOLE_FILL_MAX_VOXELS) if whole
                 else int(s.get("box_budget", ADD_BOX_BUDGET)))
        if n > limit:
            # Preview is grey for the size (S1 says it): a line sending
            # the user to it, or to the tick, would mislead
            return ""
    if whole and s.get("numba_ok", True) is False:
        return ""               # the whole patch cannot be filled here
    row = live_row_reason(s)
    if row is None:
        return ""
    if row == LIVE_ROW_OVER:
        return ADD_LIVE_NOTE_OVER.format(
            n=voxel_limit_words(ADD_LIVE_CELLS_MAX_VOXELS))
    if row == LIVE_ROW_SPACING:
        return ADD_LIVE_NOTE_SPACING
    if row == LIVE_ROW_PREPARING:
        return ADD_LIVE_NOTE_PREPARING
    why = s.get("live_aside")
    if why == "walls":
        return ADD_LIVE_NOTE_WALLS
    if why == "grow":
        return ADD_LIVE_NOTE_GROW
    if s.get("flat_spots"):
        return ADD_LIVE_NOTE_FLAT
    return ADD_LIVE_NOTE_OFF


def measuring_stage(state, reader=None):
    """The strip's stage key while something is being measured or read --
    the box, the patch, the tissue outline, another run -- or ``None``.
    The page runs its moving bar with it only while a reader is alive or
    due.  S1's own measuring sentence names it; when S1 says something else
    (a marker to place outranks it), ``reader`` -- ``"info"`` (the box-info
    worker: the box, the patch, the outline) or ``"run"`` (a run read) --
    names what is being read, so the strip says so all the same."""
    key = {
        ADD_S1_MEASURING: "add_info",
        ADD_S1_MEASURING_PATCH: "add_info_patch",
        ADD_OUTLINE_PENDING: "add_outline",
        ADD_S1_RUN_LOADING: "add_run",
    }.get(s1_sentence(state))
    if key is not None or reader is None:
        return key
    s = dict(state or {})
    if preview_block(s) in (ADD_S1_WHOLE_PICK_RUN, ADD_S1_WHOLE_NO_RUN):
        # a whole patch these choices cannot measure: nothing is named
        return None
    if reader == "run":
        return "add_run"
    if s.get("outline_pending"):
        return "add_outline"
    return "add_info_patch" if s.get("whole_patch") else "add_info"


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
    if s.get("busy") or s.get("live_updating"):
        return "busy"
    if s.get("review") is not None:
        return "review"
    cand = s.get("candidates")
    if cand is not None:
        # no row ticked: Accept is grey and the line names All (amber)
        if s1_sentence(s) == ADD_S1_TICK_ONE or _none_ticked(cand):
            return "warn"
        return "review"
    block = preview_block(s)
    if block in (ADD_S1_IDLE, ADD_S1_IDLE_BOX):
        return "idle"
    if block in (ADD_S1_MEASURING, ADD_S1_MEASURING_PATCH,
                 ADD_OUTLINE_PENDING):
        return "busy"
    if block is not None:
        return "warn"
    if s1_sentence(s) in (ADD_S1_MEASURING, ADD_S1_MEASURING_PATCH,
                          ADD_S1_LIVE_PREPARING):
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
    #: the engine's reason code when the engine refused (Cells), unchanged
    refusal_reason: str | None = None
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
    #: Cells ▸ Nuclei masks (a box or a whole patch): markers under
    #: ``ADD_SMALL_MARKER_UM3``
    n_small_markers: int = 0
    #: ... and the nuclear seeds found there, handed back for the page to
    #: keep (``{"seeds", "n_small"}``; ``None`` when none were found here)
    nuclei_seeds: dict | None = None
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
    #: Nuclei ▸ At the markers I place: {reserved id: "dim" | "small"} for
    #: each marker that made nothing for that reason (a marker on a kept
    #: object is in ``absorbed_ids``)
    marker_why: dict = field(default_factory=dict)
    #: click markers sharing a reserved id are the seeds of ONE object:
    #: per click point, in the order handed in, why that seed added
    #: nothing to its object (``"kept"`` / ``"dim"`` / ``"apart"``) or
    #: ``None``.  Empty when no two markers share an id
    seed_why: list = field(default_factory=list)
    #: the reserved ids of the objects one of whose seeds was refused as
    #: not touching the object's own piece, in placement order of those
    #: seeds
    apart_ids: list = field(default_factory=list)
    #: the click points handed in (VOLUME z, y, x), in the order
    #: ``seed_why`` reads them, so the page can name each seed's status
    seed_zyx: list = field(default_factory=list)
    #: markers whose whole ball lay off the stain
    n_markers_dim: int = 0
    # -- Inside each cell (all keyed by the FINAL ids) -------------------- #
    #: "anywhere" / "inside" / "markers" (the Nuclei preset's Look for
    #: nuclei)
    look_for: str = LOOK_ANYWHERE
    #: the box was the whole patch (no face is a cut)
    whole_patch: bool = False
    # -- the markers on a whole patch, run as you click --------------------- #
    #: the prepared patch (``add_incremental.MarkerStage``) the run used,
    #: kept by the page for the next click
    marker_stage: object = None
    #: the answer (``add_incremental.MarkerSolution``) and what changed
    #: from the solution the run was handed -- named by a weak reference
    #: (``marker_previous_ref``, ``None`` with none), so a result on screen
    #: never keeps a second whole solution alive
    marker_solution: object = None
    marker_diff: object = None
    marker_previous_ref: object = None
    #: the run prepared the patch itself (its first run, or after a change)
    marker_prepared: bool = False
    #: {id: the other run's cell that owns it}
    owner: dict = field(default_factory=dict)
    #: {cell: [ids]} for cells holding two or more nuclei
    two_in_cell: dict = field(default_factory=dict)
    #: ids the owner's border cut by more than a quarter
    cut_at_edge: set = field(default_factory=set)
    n_cell_has_one: int = 0
    n_no_nucleus: int = 0
    n_split_border: int = 0
    n_outside_cells: int = 0
    n_all_taken: int = 0
    # -- Cells: the tissue outline, and Spread's tiles on a box ------------ #
    #: markers dropped because most of them lay outside the outline
    n_outside_outline: int = 0
    #: run objects dropped because most of them lay over kept objects
    n_slivers: int = 0
    #: the pieces a Spread filled (0 for the fill)
    n_tiles: int = 0
    #: Cells ▸ Click markers redone click by click (only where the click
    #: changed it): the faint pairs and the unfilled share are not measured
    by_click: bool = False
    #: Cells ▸ Click markers with a stage handed in: how the fill ran --
    #: "stage prepared" / "stage reused" (click by click), "full run
    #: (why)", or "stage refused (reason)"; "" when no stage was handed in.
    #: ``click_window``: the voxels the click's floods covered (the box on
    #: a full run); ``click_ms``: the run's time on the worker
    click_how: str = ""
    click_window: int = 0
    click_ms: float = 0.0
    #: Cells: the growth rule the run filled by (``cells_growth``): on a
    #: whole patch ``GROW_PATCH`` (the fill with a background marker) or
    #: ``GROW_FLOOD`` (the box rule, filled as you click); "" otherwise
    growth: str = ""

    @property
    def n_two_in_cell(self) -> int:
        """Cells holding two or more of the found nuclei (the S1 tail)."""
        return len(self.two_in_cell or {})

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
        faint = "[{}]".format(faint)
        if self.by_click:
            faint = uf = "n/a (by click)"
        line = ("[ADD] preview {} preset={} source={} n={} counts={} "
                "faint={} unfilled={}".format(
                    self.status, self.preset, self.source, len(self.ids),
                    self.counts(), faint, uf))
        if self.absorbed_ids:
            line += " absorbed_markers={}".format(list(self.absorbed_ids))
        if self.marker_why:
            line += " marker_why={}".format(dict(self.marker_why))
        if self.apart_ids:
            line += " not_connected={}".format(list(self.apart_ids))
        return line

    def click_line(self) -> str:
        """The ``[ADD] cells by click`` console line (``""`` when no stage
        was handed in)."""
        if not self.click_how:
            return ""
        return ("[ADD] cells by click: {}, window {} voxels, engine {:.0f} "
                "ms".format(self.click_how, int(self.click_window),
                            float(self.click_ms)))


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


def _finish(out, result, params, cancel_event, step, outline=None,
            after_smooth=None):
    """Edge flags on the RAW result, then the smoothing; sizes, small.

    On a whole patch every box face is the edge of the data, not a cut by
    the box, so no face flags anything (``params["whole_patch"]``); an
    ordinary box flags all six, as it always did.

    ``outline`` (bool, the box's shape): the tissue outline is a hard
    edge; the smoothing can round a cell past it, so what lies outside it
    is cleared after the smoothing.  ``None`` (no outline): nothing is
    clipped.

    ``after_smooth`` (``result -> result``): run on the smoothed result
    only (Inside each cell cuts each nucleus to its owner cell there);
    without the smoothing it is not called."""
    from .add_segment import BOX_FACES, edge_flag_ids, smooth_pieces

    out.whole_patch = bool(params.get("whole_patch", False))
    if params.get("clear_border", True) and result.ndim == 3:
        out.edge_ids = set(edge_flag_ids(
            result, int(params.get("edge_margin", 3)),
            open_faces=BOX_FACES if out.whole_patch else ()))
    if _cancelled(cancel_event):
        out.status = "cancelled"
        return out
    if params.get("auto_smooth", True) and result.ndim == 3:
        step("add_smooth")
        before = {int(v) for v in np.unique(result) if v}
        result = smooth_pieces(result, float(params.get("smooth_sigma", 1.5)),
                               cancel_event=cancel_event)
        if outline is not None:
            result = _clip_to_outline(result, outline)
        if after_smooth is not None:
            result = after_smooth(result)
        after = {int(v) for v in np.unique(result) if v}
        out.n_emptied_by_smooth = len(before - after)
        out.edge_ids &= after
    elif outline is not None and result.ndim == 3:
        result = _clip_to_outline(result, outline)
    if _cancelled(cancel_event):
        out.status = "cancelled"
        return out
    out.ids, out.sizes = _sizes_of(result)
    out.small_ids = small_piece_ids(out.sizes)
    out.result = result
    return out


def _clip_to_outline(result, outline):
    """``result`` with every voxel outside ``outline`` cleared (a copy)."""
    mask = np.asarray(outline, bool)
    if mask.shape != np.shape(result):
        raise ValueError(
            "the tissue outline is {} but the result is {}".format(
                mask.shape, np.shape(result)))
    n_out = int(np.count_nonzero(np.asarray(result)[~mask]))
    if n_out:
        print("[ADD] {} voxels past the tissue outline cleared after the "
              "smoothing".format(n_out))
    return np.where(mask, result, 0).astype(np.asarray(result).dtype,
                                            copy=False)


def _clip_to_owner(result, guide, owner, min_voxels=0):
    """``result`` with every voxel of a nucleus that lies outside its owner
    cell cleared (a copy when anything is cleared).  Every id the cut
    touched then keeps only its largest connected piece, and is left out
    under ``min_voxels`` (the recipe's own size floor) -- one nucleus, one
    piece, as the recipe makes them.  ``owner`` ``{id: cell}`` in the
    numbering of ``guide``; an id it does not name is left alone.

    Returns ``(result, {id: voxels it lost}, [ids left out])``."""
    from scipy import ndimage as ndi

    r = np.asarray(result)
    if not owner or not r.size:
        return r, {}, []
    ids = np.asarray(sorted(int(i) for i in owner), np.int64)
    cell = np.asarray([int(owner[int(i)]) for i in ids], np.int64)
    nz = np.flatnonzero(r)
    if not nz.size:
        return r, {}, []
    vals = r.reshape(-1)[nz].astype(np.int64)
    pos = np.minimum(np.searchsorted(ids, vals), ids.size - 1)
    known = ids[pos] == vals
    have = np.asarray(guide).reshape(-1)[nz].astype(np.int64)
    bad = known & (cell[pos] != have)
    if not bad.any():
        return r, {}, []
    out = np.array(r, copy=True)
    out.reshape(-1)[nz[bad]] = 0
    u, n_all = np.unique(vals, return_counts=True)
    total = dict(zip(u.tolist(), n_all.tolist()))
    touched = np.unique(vals[bad])
    # what is left of each touched id, grouped by id
    sel = ~bad & np.isin(vals, touched)
    idx = nz[sel]
    v = vals[sel]
    order = np.argsort(v, kind="stable")
    idx = idx[order]
    v = v[order]
    starts = np.r_[0, np.flatnonzero(v[1:] != v[:-1]) + 1, v.size]
    final = {}
    dropped = []
    for a, b in zip(starts[:-1], starts[1:]):
        if a == b:
            continue
        i = int(v[a])
        coords = np.unravel_index(idx[a:b], r.shape)
        sl = tuple(slice(int(c.min()), int(c.max()) + 1) for c in coords)
        sub = out[sl]
        m = sub == i
        cc, n = ndi.label(m)
        keep = m
        if n > 1:
            keep = cc == (np.argmax(np.bincount(cc.ravel())[1:]) + 1)
        k = int(keep.sum())
        if k < min_voxels:
            sub[m] = 0
            dropped.append(i)
            k = 0
        elif n > 1:
            sub[m & ~keep] = 0
        final[i] = k
    for i in touched.tolist():
        if int(i) not in final:
            final[int(i)] = 0
            dropped.append(int(i))
    lost = {i: int(total[i]) - k for i, k in final.items()}
    print("[ADD] {} voxels past their owner cell cleared after the "
          "smoothing ({} nuclei, {} left out)".format(
              int(sum(lost.values())), len(lost), len(dropped)))
    return out, lost, sorted(dropped)


def _ids_fit(next_label, n, id_max) -> bool:
    if id_max is None or n <= 0:
        return True
    return int(next_label) + int(n) - 1 <= int(id_max)


def run_add_preview(*, preset, roi, existing_roi, next_label, spacing, params,
                    image_roi=None, nuclei_roi=None, wall_roi=None,
                    dark_rois=None, run_guide_roi=None, click_points=(),
                    reserved_ids=None, dark_threshold=None, id_max=None,
                    patch_levels=None, cancel_event=None,
                    progress=None, outline_roi=None,
                    elevation_roi=None, run_guide_patch=None,
                    existing_patch=None, marker_engine=False,
                    marker_stage=None,
                    marker_previous=None, stage_holder=None,
                    stage_key=None, nuclei_seeds=None) -> AddPreviewResult:
    """One Preview of the box.  Every array is a BOX CROP the
    main thread copied, except Cells' image channels (``wall_roi``,
    ``dark_rois``, ``nuclei_roi``): those are the layers' own arrays, on a
    box as on the whole patch, cut to the box here on the worker (the
    engine slices before it reads); ``click_points`` are VOLUME ZYX (made
    box-local here).  ``id_max`` is the label dtype's largest id.
    ``reserved_ids``, one per click point in the same order:
    the ids the markers reserved at placement -- the k-th marker's object
    is written as ``reserved_ids[k]``, not ``next_label + rank``.

    On the Nuclei preset with ``params["look_for"] == "inside"`` the
    other run's cells (``run_guide_roi``) are searched with
    :func:`add_derive.nuclei_in_cells`; ``patch_levels`` is the patch's own
    brightness levels already cropped to the box's z range (a box's own
    levels differ from the patch's).  On a box, ``run_guide_patch`` (the
    other run over the whole patch, read only) and ``existing_patch`` (a
    copy of the run being edited over the whole patch) let "a cell that
    already holds a nucleus is skipped" see each cell whole: a box sees
    only its crop of a cell.  Without them the crop decides, as before.

    On the Nuclei preset with ``params["look_for"] == "markers"`` each
    click point (the ticked markers, in placement order) finds the one
    nucleus it sits on (:func:`add_segment.segment_at_markers`, with
    ``params["click_radius_um"]`` and ``spacing``; ``params["separate"]``
    with ``params["peak_spacing_um"]`` also cuts a touching group at its
    waists, see :func:`_marker_split`); its object is written as
    its reserved id (without ``reserved_ids``: ``next_label`` plus its
    place in the list).  A marker that made nothing is named in
    ``absorbed_ids`` (on a kept object) or in ``marker_why`` (``"dim"``,
    ``"small"``), on a refusal too.  Anywhere in the box ignores the click
    points.

    On the Nuclei preset (Anywhere and the markers) with
    ``params["outline"]`` ``"shrink"`` / ``"cleared"`` and an ``outline_roi``
    (the tissue outline, measured once over the whole patch, or its box
    crop), Anywhere leaves out each object lying at least
    ``OUTLINE_DROP_SHARE`` outside it and cuts the rest to it; the markers
    only cut (a marker whose object the cut empties is marked
    ``"outline"``).  With no outline chosen ``outline_roi`` is ignored.

    On the Cells preset ``outline_roi`` is the finished tissue outline
    (measured once over the whole patch, see :func:`measure_outline`) and
    ``elevation_roi`` the wall picture built once and handed in again: over
    the whole patch on a whole patch, or over the box on a box at or under
    ``ADD_LIVE_CELLS_MAX_VOXELS`` (kept from its first run; ``None``: built
    here).  ``stage_holder`` / ``stage_key`` (Cells ▸ Click markers with the
    fill to the walls): redo the fill click by click, see
    :func:`_cells_by_click`.  The growth rule is
    :func:`cells_growth`: a whole patch is filled to the walls with a
    background marker, a box grows by its Grow by pick.  Before the first
    Spread from each marker in a process -- or the first whole-patch fill,
    which needs numba whatever the membrane tick -- the growth engine is set
    up (``region.warm_up_compete``); when that takes long the progress
    callback hears ``("add_warm", 0, 0)``.  An install that cannot run it
    refuses with ``ADD_NO_NUMBA`` (a whole patch: ``ADD_WHOLE_NEEDS_NUMBA``,
    or its cache-folder twin).

    ``marker_engine`` (Nuclei ▸ the markers on a whole patch, "Update as I
    click" on): the markers run through :mod:`add_incremental` --
    ``marker_stage`` the patch prepared by an earlier run (``None``: it is
    prepared here, from ``image_roi`` and ``existing_roi``; with a stage
    ``existing_roi`` may be ``None``), ``marker_previous`` the last
    solution on it; the result carries the stage, the new solution and
    what changed."""
    if reserved_ids is not None:
        reserved_ids = [int(i) for i in reserved_ids]
        if len(reserved_ids) != len(list(click_points or ())):
            raise ValueError(
                "reserved_ids has {} ids for {} click points".format(
                    len(reserved_ids), len(list(click_points or ()))))
    from .reshape.engine import ReshapeCancelled

    params = dict(params or {})
    out = AddPreviewResult(roi=roi, preset=str(preset))
    out.seed_zyx = [tuple(int(round(float(v))) for v in pt)
                    for pt in (click_points or ())]
    if (reserved_ids and id_max is not None
            and max(reserved_ids) > int(id_max)):
        # the dtype guard BEFORE the engine runs (B-L2): a reserved id that
        # cannot be written refuses at once, not after the fill
        out.status = "refused"
        out.refusal = ADD_S1_IDS_FULL
        return out
    try:
        if preset == "nuclei" and params.get("look_for") == LOOK_INSIDE:
            return _run_nuclei_inside(
                out, existing_roi=existing_roi, next_label=next_label,
                spacing=spacing, params=params, image_roi=image_roi,
                run_guide_roi=run_guide_roi, patch_levels=patch_levels,
                id_max=id_max, cancel_event=cancel_event, progress=progress,
                run_guide_patch=run_guide_patch,
                existing_patch=existing_patch)
        if preset == "nuclei":
            return _run_nuclei(out, roi=roi, existing_roi=existing_roi,
                               next_label=next_label, spacing=spacing,
                               params=params, image_roi=image_roi,
                               id_max=id_max, cancel_event=cancel_event,
                               progress=progress, click_points=click_points,
                               reserved_ids=reserved_ids,
                               outline_roi=outline_roi,
                               marker_engine=marker_engine,
                               marker_stage=marker_stage,
                               marker_previous=marker_previous)
        growth = cells_growth(params)
        out.growth = str(growth)
        # a whole patch stays gated on numba, whatever the membrane tick
        if growth in (GROW_SPREAD, GROW_PATCH):
            why, engine_says = _warm_up(progress)
            if why != NUMBA_OK:
                out.status = "refused"
                out.refusal = (whole_needs_numba_sentence(why)
                               if growth == GROW_PATCH
                               else no_numba_sentence(why))
                out.refusal_reason = "numba-" + why
                out.messages.append("[ADD] Spread cannot run: {}".format(
                    engine_says))
                return out
            if _cancelled(cancel_event):
                out.status = "cancelled"
                return out
        t0 = _perf_counter()
        res = _run_cells(out, roi=roi, existing_roi=existing_roi,
                         next_label=next_label, spacing=spacing,
                         params=params, nuclei_roi=nuclei_roi,
                         wall_roi=wall_roi, dark_rois=dark_rois,
                         run_guide_roi=run_guide_roi,
                         click_points=click_points,
                         reserved_ids=reserved_ids,
                         dark_threshold=dark_threshold, id_max=id_max,
                         cancel_event=cancel_event, progress=progress,
                         outline_roi=outline_roi,
                         elevation_roi=elevation_roi,
                         stage_holder=stage_holder, stage_key=stage_key,
                         nuclei_seeds=nuclei_seeds)
        if res.click_how:
            res.click_ms = (_perf_counter() - t0) * 1000.0
        return res
    except ReshapeCancelled as cancel:
        out.status = "cancelled"
        out.result = None
        # the walls the cancelled run had built already (the engine hands
        # them on the exception): the page keeps them for the run that
        # follows a superseded update
        elev = getattr(cancel, "elevation", None)
        if elev is not None:
            out.elevation = elev
        return out


def _warm_up(progress):
    """Set the growth engine up before its first run (on this worker
    thread); ``progress("add_warm", 0, 0)`` only when it takes long.
    ``(key, engine sentence)``: ``NUMBA_OK``, or why this install cannot
    run it (``NUMBA_MISSING_KEY`` / ``NUMBA_CACHE_KEY``)."""
    from .reshape.region import compete_is_warm, warm_up_compete

    if compete_is_warm():
        return NUMBA_OK, ""

    def _announce():
        if progress is not None:
            progress("add_warm", 0, 0)

    try:
        warm_up_compete(announce=_announce)
    except ImportError as exc:
        return NUMBA_MISSING_KEY, str(exc)
    except RuntimeError as exc:
        return NUMBA_CACHE_KEY, str(exc)
    return NUMBA_OK, ""


def _nuclei_outline_box(params, outline_roi, roi, shape):
    """The Nuclei preset's tissue outline cut to the box (bool), or ``None``
    when none is chosen (``params["outline"]``) or none was handed in."""
    mode = params.get("outline", OUTLINE_NONE) or OUTLINE_NONE
    if mode == OUTLINE_NONE or outline_roi is None:
        return None
    from .reshape.region import _region_crop

    return np.asarray(_region_crop(outline_roi, roi, tuple(shape),
                                   what="the tissue outline"), bool)


def _nuclei_all_outside(params) -> str:
    """The refusal when every nucleus found lay outside the outline."""
    return (ADD_S1_NUCLEI_ALL_OUTSIDE_CLEARED
            if params.get("outline") == OUTLINE_CLEARED
            else ADD_S1_NUCLEI_ALL_OUTSIDE)


def _ids_present(labels) -> set:
    """The non-zero ids of ``labels`` (small ids: one bincount)."""
    a = np.asarray(labels)
    if not a.size or int(a.max()) <= 0:
        return set()
    counts = np.bincount(a.ravel().astype(np.int64, copy=False))
    return {int(i) for i in np.flatnonzero(counts) if i}


def _cut_to_outline(result, outline_box, *, drop, min_size=0, balls=None):
    """``result`` (ids ``1..K``) cut to the tissue outline, and the ids that
    lie outside it: with ``drop`` every object at least
    ``OUTLINE_DROP_SHARE`` outside is left out first (Anywhere, as Cells
    drops its markers); then every voxel outside is cleared.  Every object
    the cut trimmed then keeps ONE 6-connected piece -- the one holding the
    most of its marker's ball in ``balls`` (the markers: the k-th ball is
    ``k``), else its largest (a tie goes to the first in scan order) -- and
    is left out under ``min_size`` voxels (Smallest object): one nucleus,
    one piece, never under the floor.  An object the cut empties, or trims
    under the floor, counts with the ones left out.

    Returns ``(cut, {ids left out}, {ids the cut trimmed})``."""
    from scipy import ndimage as ndi

    from .reshape.outline import OUTLINE_DROP_SHARE, markers_outside

    before = _ids_present(result)
    if drop and before:
        gone = markers_outside(result, outline_box, OUTLINE_DROP_SHARE)
        if gone.size:
            result = np.where(np.isin(result, gone), 0, result).astype(
                result.dtype, copy=False)
    cut = np.where(outline_box, result, 0).astype(result.dtype, copy=False)
    trimmed = set()
    if before:
        top = max(before) + 1
        was = np.bincount(np.asarray(result).ravel().astype(np.int64),
                          minlength=top)
        now = np.bincount(cut.ravel().astype(np.int64), minlength=top)
        trimmed = {int(i) for i in np.flatnonzero((now < was) & (now > 0))}
    if trimmed:
        slices = ndi.find_objects(cut.astype(np.int64, copy=False),
                                  max_label=max(trimmed))
        for i in sorted(trimmed):
            sl = slices[i - 1]
            sub = cut[sl]
            m = sub == i
            cc, n = ndi.label(m)
            keep = m
            if n > 1:
                sizes = np.bincount(cc.ravel(), minlength=n + 1)
                held = np.zeros(n + 1, np.int64)
                if balls is not None:
                    held = np.bincount(cc[np.asarray(balls)[sl] == i],
                                       minlength=n + 1)
                best = max(range(1, n + 1),
                           key=lambda r: (int(held[r]), int(sizes[r]), -r))
                keep = cc == best
            if int(np.count_nonzero(keep)) < int(min_size or 0):
                sub[m] = 0
            elif n > 1:
                sub[m & ~keep] = 0
    outside = before - _ids_present(cut)
    if outside:
        print("[ADD] {} nuclei outside the tissue outline left out".format(
            len(outside)))
    return cut, outside, trimmed


def _run_nuclei(out, *, roi, existing_roi, next_label, spacing, params,
                image_roi, id_max, cancel_event, progress, click_points=(),
                reserved_ids=None, outline_roi=None, marker_engine=False,
                marker_stage=None, marker_previous=None):
    if params.get("look_for") == LOOK_MARKERS:
        return _run_nuclei_markers(
            out, roi=roi, existing_roi=existing_roi, next_label=next_label,
            spacing=spacing, params=params, image_roi=image_roi,
            id_max=id_max, cancel_event=cancel_event, progress=progress,
            click_points=click_points, reserved_ids=reserved_ids,
            outline_roi=outline_roi, marker_engine=marker_engine,
            marker_stage=marker_stage, marker_previous=marker_previous)
    from .add_segment import add_segment_region

    step = _Steps(progress, ("add_box", "add_find", "add_smooth"))
    step("add_box")
    if image_roi is None:
        raise ValueError("the Nuclei preset needs image_roi")
    if _cancelled(cancel_event):
        out.status = "cancelled"
        return out
    step("add_find")
    # a Tissue outline: the finder numbers from 1, the objects outside are
    # left out and the rest cut, then the survivors take next_label on
    # (without one the call is the one it always was)
    outline_box = _nuclei_outline_box(params, outline_roi, roi,
                                      np.shape(existing_roi))
    result = add_segment_region(
        image_roi, existing_roi,
        clahe=bool(params.get("clahe", False)),
        clahe_clip=float(params.get("clahe_clip", 0.01)),
        threshold=params.get("threshold", "otsu"),
        min_size=int(params.get("min_size", 0)),
        separate=bool(params.get("separate", True)),
        min_distance=int(params.get("min_distance", 3)),
        next_label=(1 if outline_box is not None else int(next_label)),
        spacing=spacing,
        cancel_event=cancel_event,
    )
    n = int(len(np.unique(result)) - (1 if (result == 0).any() else 0))
    if n == 0:
        out.status = "refused"
        out.refusal = ADD_S1_NOTHING_NUCLEI
        return out
    if outline_box is not None:
        if _cancelled(cancel_event):
            out.status = "cancelled"
            return out
        result, outside, _trimmed = _cut_to_outline(
            result, outline_box, drop=True,
            min_size=int(params.get("min_size", 0)))
        out.n_outside_outline = len(outside)
        n -= len(outside)
        if n == 0:
            out.status = "refused"
            out.refusal = _nuclei_all_outside(params)
            return out
        result, _mapping = _mint(result, next_label)
    if not _ids_fit(next_label, n, id_max):
        out.status = "refused"
        out.refusal = ADD_S1_IDS_FULL
        return out
    if _cancelled(cancel_event):
        out.status = "cancelled"
        return out
    _finish(out, result, params, cancel_event, step, outline=outline_box)
    if out.status == "ok" and not out.ids:
        out.status = "refused"
        out.refusal = ADD_S1_ALL_FILTERED
        out.result = None
    return out


def _marker_split(params):
    """``(split, peak_spacing_um)`` for the markers' engine call.

    "Separate touching objects" ticked (``params["separate"]``) asks for
    :data:`ADD_MARKER_SPLIT_DEFAULT` at ``params["peak_spacing_um"]`` (the
    markers' Spacing in microns, which the page always sends).  Unticked, a
    missing tick, or a ticked call without a finite Spacing in microns > 0
    runs as before: ``("off", None)``, and says so in one console line (the
    page always sends the Spacing, so the line means a caller left it out).
    A missing key falls back as every other key here does; the Spacing never
    falls back to the engine's own default, which counts voxels of the
    spacing it is handed -- twice the tissue distance in a half-size
    session."""
    if not bool(params.get("separate", False)):
        return "off", None
    raw = params.get("peak_spacing_um")
    try:
        um = float(raw)
    except (TypeError, ValueError):
        um = float("nan")
    if not math.isfinite(um) or um <= 0:
        print("[ADD-SPLIT] Separate touching objects is ticked but the "
              "markers came with no Spacing in microns ({!r}): the markers "
              "run as before (split off)".format(raw))
        return "off", None
    return ADD_MARKER_SPLIT_DEFAULT, um


def _run_nuclei_markers(out, *, roi, existing_roi, next_label, spacing,
                        params, image_roi, id_max, cancel_event, progress,
                        click_points, reserved_ids, outline_roi=None,
                        marker_engine=False, marker_stage=None,
                        marker_previous=None):
    """Look for nuclei ▸ At the markers I place: one object per click, on
    the bright piece the click sits on (see
    :func:`add_segment.segment_at_markers`), numbered by the click's
    reserved id.  The same stages, threshold, Smallest object, smoothing and
    edge flags as Anywhere; a clicked nucleus at the box edge or small
    starts unticked like any other.  A Tissue outline only CUTS here (a
    click is never dropped for lying mostly outside, as Cells keeps its
    clicked markers); a marker whose nucleus the cut empties is marked
    ``"outline"``.  With "Separate touching objects" ticked and the
    markers' Spacing in microns (``params["peak_spacing_um"]``) a click on a
    nucleus in a touching group gives that nucleus, not the group (see
    :func:`_marker_split`); otherwise each click takes its whole bright
    piece, as before.  ``marker_engine``: see :func:`_run_markers_engine`."""
    from .add_segment import segment_at_markers

    out.look_for = LOOK_MARKERS
    step = _Steps(progress, ("add_box", "add_find", "add_smooth"))
    step("add_box")
    if image_roi is None:
        raise ValueError("the Nuclei preset needs image_roi")
    points = list(click_points or ())
    if not points:
        out.status = "refused"
        out.refusal = ADD_S1_NO_MARKERS
        return out
    if reserved_ids is None:
        # no reservation: each click takes the next id in placement order
        if not _ids_fit(next_label, len(points), id_max):
            out.status = "refused"
            out.refusal = ADD_S1_IDS_FULL
            return out
        ids = [int(next_label) + i for i in range(len(points))]
    else:
        ids = [int(i) for i in reserved_ids]
    if _cancelled(cancel_event):
        out.status = "cancelled"
        return out
    starts = tuple(int(sl.start or 0) for sl in roi)
    local = [tuple(int(p[a]) - starts[a] for a in range(3)) for p in points]
    step("add_find")
    split, peak_um = _marker_split(params)
    if marker_engine:
        return _run_markers_engine(
            out, existing_roi=existing_roi, spacing=spacing, params=params,
            image_roi=image_roi, cancel_event=cancel_event, step=step,
            local=local, ids=ids, outline_roi=outline_roi, split=split,
            peak_um=peak_um, stage=marker_stage, previous=marker_previous)
    # markers sharing a reserved id are the seeds of one object (Shift+
    # click); only then is the engine asked to group them
    grouped = len(set(ids)) < len(ids)
    kw = {"groups": ids} if grouped else {}
    got = segment_at_markers(
        image_roi, existing_roi, local,
        radius_um=float(params.get("click_radius_um", 1.0)),
        spacing=spacing,
        clahe=bool(params.get("clahe", False)),
        clahe_clip=float(params.get("clahe_clip", 0.01)),
        threshold=params.get("threshold", "otsu"),
        min_size=int(params.get("min_size", 0)),
        cancel_event=cancel_event,
        split=split,
        peak_spacing_um=peak_um,
        **kw,
    )
    if grouped:
        pieces, why, seed_why = got
        out.seed_why = [seed_why.get(k) for k in range(1, len(points) + 1)]
        out.apart_ids = [ids[k - 1] for k in sorted(seed_why)
                         if seed_why[k] == "apart"]
        # from here on, one entry per object (in the order each first
        # appears), its balls drawn from its seeds that joined it
        ball_pts = [local[k - 1] for k in range(1, len(points) + 1)
                    if seed_why.get(k) != "apart"]
        ball_ids = [ids[k - 1] for k in range(1, len(points) + 1)
                    if seed_why.get(k) != "apart"]
        ids = list(dict.fromkeys(ids))
    else:
        pieces, why = got
        ball_pts, ball_ids = local, ids
    outline_box = _nuclei_outline_box(params, outline_roi, roi,
                                      np.shape(existing_roi))
    trimmed = set()
    if outline_box is not None:
        if _cancelled(cancel_event):
            out.status = "cancelled"
            return out
        why = dict(why)
        # the click balls, as the engine draws them: a nucleus the cut
        # splits keeps the piece holding its marker's ball
        from .reshape.region import rasterise_points

        balls = rasterise_points(
            ball_pts, np.shape(existing_roi),
            radius_um=float(params.get("click_radius_um", 1.0)),
            spacing=(tuple(float(v) for v in spacing)
                     if spacing is not None else (1.0, 1.0, 1.0)))
        if grouped:
            # numbered as the engine numbers the objects
            lut = np.zeros(len(ball_ids) + 1, np.int32)
            lut[1:] = [ids.index(i) + 1 for i in ball_ids]
            balls = lut[balls]
        pieces, outside, trimmed = _cut_to_outline(
            pieces, outline_box, drop=False,
            min_size=int(params.get("min_size", 0)), balls=balls)
        for k in outside:
            why[int(k)] = "outline"
        out.n_outside_outline = len(outside)
    out.absorbed_ids = [ids[k - 1] for k in sorted(why) if why[k] == "kept"]
    out.marker_why = {ids[k - 1]: w for k, w in sorted(why.items())
                      if w in ("dim", "small", "outline")}
    out.n_markers_dim = sum(1 for w in why.values() if w == "dim")
    out.n_below_min_size = sum(1 for w in why.values() if w == "small")
    if out.absorbed_ids:
        out.messages.append(
            "click markers absorbed (their ball was already taken): "
            "{}".format(out.absorbed_ids))
    if out.n_markers_dim:
        out.messages.append(
            "click markers off the stain (their ball is below the "
            "threshold): {}".format(
                [i for i, w in out.marker_why.items() if w == "dim"]))
    mapping = {k: ids[k - 1] for k in range(1, len(ids) + 1)
               if k not in why}
    if not mapping:
        out.status = "refused"
        out.refusal = _markers_refusal(out, params, len(points))
        return out
    if _cancelled(cancel_event):
        out.status = "cancelled"
        return out
    result = _mint_to(pieces, mapping)
    _finish(out, result, params, cancel_event, step, outline=outline_box)
    if out.status != "ok":
        return out
    alive = {int(i) for i in out.ids}
    for k, new in mapping.items():
        if int(new) not in alive:
            if int(k) in trimmed:
                # the cut left a sliver the smoothing emptied: the outline
                # took it, not the size
                out.marker_why[int(new)] = "outline"
                out.n_outside_outline += 1
                out.n_emptied_by_smooth = max(
                    0, int(out.n_emptied_by_smooth) - 1)
            else:
                # emptied by the smoothing
                out.marker_why[int(new)] = "small"
    if not out.ids:
        out.status = "refused"
        out.refusal = _markers_refusal(out, params, len(points))
        out.result = None
    return out


def _run_markers_engine(out, *, existing_roi, spacing, params, image_roi,
                        cancel_event, step, local, ids, outline_roi, split,
                        peak_um, stage, previous):
    """The markers on a whole patch, run as you click: the patch prepared
    once (:func:`add_incremental.prepare_markers_box`, here when ``stage``
    is ``None``) and each run solving only the bright pieces whose markers
    changed (:func:`add_incremental.solve_markers`, from ``previous``).  The
    result says what :func:`_run_nuclei_markers` says -- ids, sizes, small,
    the objects the smoothing emptied, why each marker made nothing, the
    seeds refused as apart -- and carries the stage, the solution and what
    changed.  A whole patch has no tissue outline and no box edge."""
    import weakref

    from .add_incremental import prepare_markers_box, solve_markers

    if (outline_roi is not None
            or params.get("outline", OUTLINE_NONE) not in (None,
                                                          OUTLINE_NONE)):
        raise ValueError("the markers on a whole patch have no tissue "
                         "outline")
    if not params.get("whole_patch", False):
        raise ValueError("the markers run as you click only on a whole "
                         "patch")
    out.marker_prepared = stage is None
    if stage is None:
        stage = prepare_markers_box(
            image_roi, existing_roi,
            radius_um=float(params.get("click_radius_um", 1.0)),
            spacing=spacing,
            clahe=bool(params.get("clahe", False)),
            clahe_clip=float(params.get("clahe_clip", 0.01)),
            threshold=params.get("threshold", "otsu"),
            min_size=int(params.get("min_size", 0)),
            split=split, peak_spacing_um=peak_um,
            cancel_event=cancel_event)
    sigma = (float(params.get("smooth_sigma", 1.5))
             if params.get("auto_smooth", True) else None)
    if sigma is not None:
        step("add_smooth")
    sol, diff = solve_markers(stage, local, ids, groups=ids,
                              smooth_sigma=sigma, previous=previous,
                              cancel_event=cancel_event)
    out.whole_patch = True
    out.marker_stage = stage
    out.marker_solution = sol
    out.marker_diff = diff
    out.marker_previous_ref = (None if previous is None
                               else weakref.ref(previous))
    gids = list(dict.fromkeys(ids))
    why = dict(sol.why)
    if len(gids) < len(ids):
        seed_why = dict(sol.seed_why)
        out.seed_why = [seed_why.get(k) for k in range(1, len(ids) + 1)]
        out.apart_ids = [ids[k - 1] for k in sorted(seed_why)
                         if seed_why[k] == "apart"]
    out.absorbed_ids = [gids[g - 1] for g in sorted(why) if why[g] == "kept"]
    out.marker_why = {gids[g - 1]: w for g, w in sorted(why.items())
                      if w in ("dim", "small", "outline")}
    out.n_markers_dim = sum(1 for w in why.values() if w == "dim")
    out.n_below_min_size = sum(1 for w in why.values() if w == "small")
    if out.absorbed_ids:
        out.messages.append(
            "click markers absorbed (their ball was already taken): "
            "{}".format(out.absorbed_ids))
    if out.n_markers_dim:
        out.messages.append(
            "click markers off the stain (their ball is below the "
            "threshold): {}".format(
                [i for i, w in out.marker_why.items() if w == "dim"]))
    made = [gids[g - 1] for g in range(1, len(gids) + 1) if g not in why]
    if not made:
        out.status = "refused"
        out.refusal = _markers_refusal(out, params, len(ids))
        return out
    out.edge_ids = set()
    out.ids = list(sol.ids)
    out.sizes = dict(sol.sizes)
    out.small_ids = small_piece_ids(out.sizes)
    out.n_emptied_by_smooth = len(sol.emptied_ids)
    out.result = sol.result
    alive = set(out.ids)
    for new in made:
        if int(new) not in alive:
            out.marker_why[int(new)] = "small"     # emptied by the smoothing
    if not out.ids:
        out.status = "refused"
        out.refusal = _markers_refusal(out, params, len(ids))
        out.result = None
    return out


#: the markers' refusal reasons, a tie going to the earlier one
_MARKER_REFUSAL_ORDER = ("small", "outline", "dim", "kept")


def _markers_refusal(out, params, n_points) -> str:
    """The markers' refusal when no marker made anything: the reason that
    covers the most markers (``out.marker_why``, the kept ones in
    ``out.absorbed_ids``); a tie goes in the order the size floor, the
    tissue outline, off the stain, inside a kept object."""
    n = {w: 0 for w in _MARKER_REFUSAL_ORDER}
    for w in dict(out.marker_why or {}).values():
        if w in n:
            n[w] += 1
    n["kept"] += len(out.absorbed_ids or ())
    pick = max(_MARKER_REFUSAL_ORDER,
               key=lambda w: (n[w], -_MARKER_REFUSAL_ORDER.index(w)))
    if pick == "small":
        return ADD_S1_ALL_FILTERED
    if pick == "outline":
        # what was found lay outside the tissue outline
        return _nuclei_all_outside(params)
    if pick == "dim":
        return ADD_S1_MARKERS_NONE_ON_STAIN
    return markers_all_kept_line(n_points)


def _run_nuclei_inside(out, *, existing_roi, next_label, spacing, params,
                       image_roi, run_guide_roi, patch_levels, id_max,
                       cancel_event, progress, run_guide_patch=None,
                       existing_patch=None):
    """Look for nuclei ▸ Inside each cell of another run: the nuclei-first
    recipe on the box, the run's cells as owners, the cut at each cell's
    border on.  Ids are minted with an offset only (the recipe numbers
    them 1..N by owner cell, then size).

    On a box with ``run_guide_patch`` and ``existing_patch``, the cells
    that already hold a kept object are measured over the whole patch
    (each cell of the box whole).  With the smoothing on, every nucleus is
    cut to its owner cell again after it (one piece, the recipe's size
    floor), and the "cut at edge" chip counts that cut on top of the
    recipe's own.  The second-nucleus guard stays the recipe's."""
    from .add_derive import (
        CUT_FLAG_SHARE,
        MIN_NUCLEUS_UM3,
        NUC_EDGE,
        NUC_RADIUS_UM,
        TWO_GAP,
        nuclei_in_cells,
        owning_cells,
    )

    out.look_for = LOOK_INSIDE
    if image_roi is None:
        raise ValueError("Inside each cell needs image_roi")
    if run_guide_roi is None:
        raise ValueError("Inside each cell needs run_guide_roi")
    if spacing is None:
        # every knob here is in microns; with no spacing handed in, the
        # 1.0 um isotropic fallback (what the page runs on when the
        # spacing was never set -- its amber note says so)
        spacing = (1.0, 1.0, 1.0)
    if _cancelled(cancel_event):
        out.status = "cancelled"
        return out
    try:
        owning_kw = {}
        if (run_guide_patch is not None and existing_patch is not None
                and existing_roi is not None
                and not params.get("whole_patch", False)):
            # a box sees only its crop of a cell: a kept nucleus of a cell
            # that lies outside the box must still count (the box's cells,
            # whole)
            among = np.unique(np.asarray(run_guide_roi))
            owning_kw["owning"] = owning_cells(
                run_guide_patch, existing_patch, among=among[among > 0])
        res = nuclei_in_cells(
            image_roi, run_guide_roi, existing_roi,
            tuple(float(v) for v in spacing),
            radius_um=float(params.get("nuc_radius_um", NUC_RADIUS_UM)),
            edge=float(params.get("nuc_edge", NUC_EDGE)),
            gap=float(params.get("two_gap", TWO_GAP)),
            cut_at_border=True, levels=patch_levels,
            cancel_event=cancel_event, progress=progress, **owning_kw)
    except (ValueError, TypeError) as exc:
        # the recipe's own plain-words refusals (a flat channel, a channel
        # with no bright level, a knob out of range): said as they are,
        # never as a failed Preview.  A Stop is not one of them
        # (ReshapeCancelled is answered by run_add_preview)
        out.status = "refused"
        out.refusal = plain_reason(str(exc))
        out.messages.append("[ADD] Inside each cell refused: {}".format(exc))
        return out
    out.messages.append(res.log_line())
    for key in ("n_cell_has_one", "n_no_nucleus", "n_split_border",
                "n_outside_cells", "n_all_taken"):
        setattr(out, key, int(getattr(res, key, 0) or 0))
    if res.n_cells == 0:
        out.status = "refused"
        out.refusal = (ADD_S1_INSIDE_NO_CELLS_PATCH
                       if params.get("whole_patch")
                       else ADD_S1_INSIDE_NO_CELLS)
        return out
    if res.all_taken:
        out.status = "refused"
        out.refusal = ADD_S1_INSIDE_ALL_TAKEN
        return out
    if res.n_nuclei == 0:
        out.status = "refused"
        out.refusal = ADD_S1_INSIDE_NOTHING
        return out
    if not _ids_fit(next_label, res.n_nuclei, id_max):
        out.status = "refused"
        out.refusal = ADD_S1_IDS_FULL
        return out
    result, mapping = _mint(res.labels, next_label)
    min_size = int(params.get("min_size", 0) or 0)
    if min_size > 0:
        ids, sizes = _sizes_of(result)
        small = [i for i in ids if sizes[i] < min_size]
        if small:
            result = np.where(np.isin(result, small), 0,
                              result).astype(result.dtype)
            out.n_below_min_size = len(small)
    if _cancelled(cancel_event):
        out.status = "cancelled"
        return out
    step = _Steps(None, ("add_smooth",))
    # the smoothing can round a nucleus back over its owner's border: cut
    # it there again (cells own the membrane), one piece each, the recipe's
    # size floor.  The second-nucleus guard is NOT run again: it was judged
    # once, on the recipe's own shapes (the smoothing shrinks every nucleus)
    owner_now = {int(mapping[k]): int(c) for k, c in res.owner.items()
                 if int(k) in mapping}
    floor = MIN_NUCLEUS_UM3 / float(np.prod([float(v) for v in spacing]))
    post = {"lost": {}, "out": [], "pre": {}}

    def _after_smooth(r):
        u, n = np.unique(result[result != 0], return_counts=True)
        post["pre"] = dict(zip(u.tolist(), n.tolist()))
        r, post["lost"], post["out"] = _clip_to_owner(
            r, run_guide_roi, owner_now, min_voxels=floor)
        return r

    _finish(out, result, params, cancel_event, step,
            after_smooth=_after_smooth)
    if post["out"]:
        # left out by the cut at the owner's border, not by the smoothing
        out.n_emptied_by_smooth = max(
            0, out.n_emptied_by_smooth - len(post["out"]))
        out.messages.append(
            "[ADD] {} nuclei kept too little inside their owner cell after "
            "the smoothing and were left out: {}".format(
                len(post["out"]), post["out"]))
    if out.status != "ok":
        return out
    if not out.ids:
        out.status = "refused"
        out.refusal = ADD_S1_ALL_FILTERED
        out.result = None
        return out
    alive = {int(i) for i in out.ids}
    owner = {int(mapping[k]): int(c) for k, c in res.owner.items()
             if int(mapping.get(k, 0)) in alive}
    two = {}
    for cell, ids in res.two_in_cell.items():
        kept = [int(mapping[k]) for k in ids if int(mapping.get(k, 0)) in alive]
        if len(kept) >= 2:
            two[int(cell)] = kept
    cut = {int(mapping[k]) for k in res.cut_at_edge
           if int(mapping.get(k, 0)) in alive}
    # the cut after the smoothing adds to the recipe's own cut: both are
    # shares of the nucleus as the recipe found it, before any cut
    recipe_id = {int(m): int(k) for k, m in mapping.items()}
    recipe_share = getattr(res, "cut_share", None) or {}
    for i, n_lost in post["lost"].items():
        if i not in alive or not n_lost:
            continue
        s = float(recipe_share.get(recipe_id.get(i), 0.0))
        whole = float(post["pre"].get(i, 0)) / max(1.0 - s, 1e-9)
        if whole > 0 and s + n_lost / whole > CUT_FLAG_SHARE:
            cut.add(i)
    out.owner = owner
    out.two_in_cell = two
    out.cut_at_edge = cut
    flags = {}
    for ids in two.values():
        for i in ids:
            flags.setdefault(i, []).append("two")
    for i in sorted(cut):
        flags.setdefault(i, []).append("cut")
    out.flags = {i: tuple(w) for i, w in flags.items()}
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


def _join_cell_seeds(engine, seed_points, reserved_ids, existing_roi, *,
                     radius_um, spacing):
    """Click markers sharing a reserved id grow ONE cell: every seed is its
    own engine marker, and the cells they grew are joined.  A later seed
    whose cell does not touch what its object's earlier seeds grew (one
    6-connected piece) is refused: its cell is left empty.  Returns
    ``(engine, mapping, absorbed, seed_why, apart_ids)`` -- the engine's
    labels (a copy when a cell was emptied), engine id -> reserved id,
    the reserved ids none of whose seeds reached the engine, per seed
    ``None`` / ``"kept"`` / ``"apart"``, and the ids a seed was refused
    for, in placement order."""
    from scipy import ndimage as ndi

    from .reshape.region import rasterise_points

    existing = np.asarray(existing_roi)
    balls = rasterise_points(seed_points, existing.shape, radius_um=radius_um,
                             spacing=spacing)
    balls = np.where(existing == 0, balls, 0)
    survivors = [int(v) for v in np.unique(balls) if v]
    eng_of = {placement: j for j, placement in enumerate(survivors, start=1)}
    n = len(reserved_ids)
    seed_why = [None if k in eng_of else "kept" for k in range(1, n + 1)]
    engine = np.asarray(engine)
    boxes = (ndi.find_objects(engine.astype(np.int64, copy=False),
                              max_label=len(survivors))
             if survivors and engine.size else [])
    copied = False
    apart = []
    for r in dict.fromkeys(reserved_ids):
        js = [eng_of[k] for k in range(1, n + 1)
              if reserved_ids[k - 1] == r and k in eng_of]
        js = [j for j in js if j <= len(boxes) and boxes[j - 1] is not None]
        if len(js) < 2:
            continue
        sl = tuple(slice(min(boxes[j - 1][a].start for j in js),
                         max(boxes[j - 1][a].stop for j in js))
                   for a in range(engine.ndim))
        sub = engine[sl]
        cur = sub == js[0]
        for j in js[1:]:
            m = sub == j
            _lab, pieces = ndi.label(cur | m)
            if pieces == 1:
                cur = cur | m
                continue
            if not copied:
                engine = np.array(engine, copy=True)
                sub = engine[sl]
                copied = True
            sub[m] = 0
            k = survivors[j - 1]
            seed_why[k - 1] = "apart"
            apart.append((k, r))
    mapping = {j: reserved_ids[k - 1] for k, j in eng_of.items()
               if seed_why[k - 1] != "apart"}
    reached = set(mapping.values())
    absorbed = [r for r in dict.fromkeys(reserved_ids) if r not in reached]
    return (engine, mapping, absorbed, seed_why,
            [r for _k, r in sorted(apart)])


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


def cells_spec(params):
    """The engine's ``ReshapeSpec`` for a Cells Preview with the page's
    ``params`` (the marker source, the walls, the dark rule, the outline,
    the growth rule of :func:`cells_growth`, the box budget).

    With ``params["box_rule"]`` (Click markers on a whole patch with
    "Update as I click" on) the whole patch runs as one box: the engine's
    flood, no ``whole_patch`` in the spec (the flood never takes one), and
    the whole fill's ceiling (``WHOLE_FILL_MAX_VOXELS``) as its budget.
    The finish still reads ``params["whole_patch"]`` (no face of a whole
    patch is a cut by a box), and so does the flood: a whole patch's faces
    are the edge of the data, not a cut, so they are not in the outside
    marker (``faces_outside=False``) and a cell may run to them.

    A knob the growth rule never reads is held at its default, so it
    cannot change the spec (nor the click fill's prepared box, keyed on
    it): Wall cost and Growth limit are Spread's alone, and the membrane
    tick is Spread's and the background fill's (the fill to the walls of a
    box, or of a whole patch filled the way a box is, ignores it)."""
    from .reshape import ReshapeSpec
    from .reshape.spec import DEFAULT_PARAMS

    source = params.get("source", SOURCE_CLICKS)
    outline = params.get("outline", "none") or "none"
    # a tissue outline turns the dark plumbing on whatever the tick says
    outside = ("faces+dark" if params.get("dark_outside", True)
               or outline != "none" else "faces")
    box_rule = bool(params.get("box_rule", False))
    budget = (int(_region.WHOLE_FILL_MAX_VOXELS) if box_rule
              else int(params.get("box_budget", ADD_BOX_BUDGET)))
    growth = cells_growth(params)
    wall_cost = float(params.get("wall_cost", DEFAULT_PARAMS["wall_cost"]))
    growth_limit = float(params.get("growth_limit",
                                    DEFAULT_PARAMS["growth_limit"]))
    include_membrane = bool(params.get(
        "include_membrane", DEFAULT_PARAMS["include_membrane"]))
    if growth != GROW_SPREAD:
        wall_cost = float(DEFAULT_PARAMS["wall_cost"])
        growth_limit = float(DEFAULT_PARAMS["growth_limit"])
    if growth == GROW_FLOOD:
        include_membrane = bool(DEFAULT_PARAMS["include_membrane"])
    return ReshapeSpec(
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
            "click_radius_um": float(params.get("click_radius_um", 1.0)),
            "box_budget": budget,
            "growth": growth,
            "wall_cost": wall_cost,
            "growth_limit": growth_limit,
            "include_membrane": include_membrane,
            "outline": outline,
            "bridge_um": float(params.get("bridge_um",
                                          DEFAULT_PARAMS["bridge_um"])),
            "whole_patch": (bool(params.get("whole_patch", False))
                            and not box_rule),
            "faces_outside": not (box_rule
                                  and bool(params.get("whole_patch", False))),
        },
    )


def _small_marker_count(seeds, spacing) -> int:
    """The objects of ``seeds`` under ``ADD_SMALL_MARKER_UM3`` cubic
    microns at ``spacing`` (z, y, x; ``None``: 1 µm a side)."""
    a = np.asarray(seeds)
    if not a.size or int(a.max()) <= 0:
        return 0
    voxel = (float(np.prod([float(v) for v in spacing]))
             if spacing is not None else 1.0)
    sizes = np.bincount(a.ravel().astype(np.int64, copy=False))[1:]
    sizes = sizes[sizes > 0]
    return int(np.count_nonzero(sizes * voxel < ADD_SMALL_MARKER_UM3))


def _run_cells(out, *, roi, existing_roi, next_label, spacing, params,
               nuclei_roi, wall_roi, dark_rois, run_guide_roi, click_points,
               dark_threshold, id_max, cancel_event, progress,
               reserved_ids=None, outline_roi=None, elevation_roi=None,
               stage_holder=None, stage_key=None, nuclei_seeds=None):
    """``nuclei_seeds`` -- Markers ▸ Nuclei masks on a whole patch: the
    seeds an earlier Preview found (``out.nuclei_seeds``), handed in again
    while what they were found from is unchanged; ``None``: found here.
    ``outline_roi`` -- the finished tissue outline (bool; built ONCE over
    the whole patch, or its box crop) for ``params["outline"]``, handed to
    the engine as its run argument ``outline=``; ``elevation_roi`` -- the
    wall picture built ONCE and handed in again: for the whole patch (Whole
    patch), or for a box at or under ``ADD_LIVE_CELLS_MAX_VOXELS`` (its own,
    kept from the box's first run); ``None``: built here over the box.
    ``params["live"]`` -- a run after a click, on a box or a whole patch:
    over ``ADD_LIVE_FULL_RUN_MAX_VOXELS``, where the click fill steps aside
    the run is refused (:data:`NOT_LIVE_HERE`) instead of a full flood, and
    while the click fill is still being compiled it waits
    (:data:`LIVE_PREPARING`).  An
    engine refusal comes back with
    its sentence in ``out.refusal`` and its reason code, unchanged, in
    ``out.refusal_reason``."""
    from .reshape.region import reshape_empty_region

    source = params.get("source", SOURCE_CLICKS)
    out.source = source
    spec = cells_spec(params)
    outline = spec.params["outline"]
    outside = spec.params["outside"]
    budget = spec.params["box_budget"]
    starts = tuple(int(sl.start or 0) for sl in roi)
    seed_layer = None
    seed_points = None
    if source == SOURCE_NUCLEI:
        from .add_segment import add_segment_region

        whole = bool(params.get("whole_patch", False))
        kept = (nuclei_seeds or {}).get("seeds") if whole else None
        if (kept is not None
                and tuple(np.shape(kept)) == tuple(np.shape(existing_roi))):
            # the seeds an earlier Preview of this patch found, unchanged
            seed_layer = kept
            out.n_small_markers = int(nuclei_seeds.get("n_small", 0) or 0)
        else:
            if nuclei_roi is None:
                raise ValueError("the nuclei-masks source needs nuclei_roi")
            if tuple(np.shape(nuclei_roi)) != tuple(np.shape(existing_roi)):
                # the channel's own array (a box's): cut to the box here, on
                # the worker, as the page's crop was
                nuclei_roi = np.array(nuclei_roi[tuple(roi)], np.float64,
                                      copy=True)
            if _cancelled(cancel_event):
                out.status = "cancelled"
                return out
            if progress is not None:
                try:
                    # the nuclei are this source's MARKERS: "Finding
                    # markers...", not the Nuclei preset's "Finding bright
                    # objects..." (live-shot fix V6)
                    progress("add_markers", 0, 4)
                except Exception:
                    pass
            # Stop acts between the finder's steps (ReshapeCancelled, as
            # the engine raises it)
            seed_layer = add_segment_region(
                nuclei_roi, existing_roi, clahe=False, clahe_clip=0.01,
                threshold=params.get("nuclei_threshold", "otsu"),
                min_size=0, separate=True, min_distance=3, next_label=1,
                spacing=spacing, cancel_event=cancel_event)
            # one rule for every region: no floor; the markers under
            # ADD_SMALL_MARKER_UM3 are counted on a box and a whole patch
            # alike; a whole patch's seeds are handed back for the page to
            # keep for this patch
            out.n_small_markers = _small_marker_count(seed_layer, spacing)
            if whole:
                out.nuclei_seeds = {"seeds": seed_layer,
                                    "n_small": int(out.n_small_markers)}
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
    # click markers sharing a reserved id are one cell grown from several
    # seeds: each seed is its own engine marker, and the cells are joined
    # after the run (see the loop below)
    grouped = (seed_points is not None and reserved_ids is not None
               and len(set(reserved_ids)) < len(reserved_ids))
    # a run after a click (not a Preview press), on a box or a whole
    # patch over ADD_LIVE_FULL_RUN_MAX_VOXELS: where the click fill steps
    # aside the region is not flooded whole -- that flood cannot be
    # stopped, and takes many seconds near the live limit -- the run is
    # refused instead and the page asks for Preview.  At or under it the
    # full run answers the click, as a Preview would
    live_click = (bool(params.get("live", False))
                  and int(np.prod(np.shape(existing_roi)))
                  > int(ADD_LIVE_FULL_RUN_MAX_VOXELS))
    region_word = ("whole patch" if params.get("whole_patch", False)
                   else "box")
    if (live_click and stage_holder is not None and seed_points is not None
            and reserved_ids is not None and _click_fill_compiling(spec)):
        # the click fill is still being compiled: the click waits for it
        # (nothing is remembered against the region; the page runs again
        # once the solver check has answered)
        return _live_preparing(out, cancel_event)
    clicks = None
    if (stage_holder is not None and seed_points is not None
            and reserved_ids is not None):
        bridge = dict(spec=spec, roi=roi, existing_roi=existing_roi,
                      spacing=spacing, wall_roi=wall_roi, dark_rois=dark_rois,
                      dark_threshold=dark_threshold, outline_roi=outline_roi,
                      elevation_roi=elevation_roi, cancel_event=cancel_event,
                      progress=progress, holder=stage_holder, key=stage_key,
                      allow_flat=bool(params.get("live_always", False)))
        if not grouped:
            done = _cells_by_click(
                out, params=params, seed_points=seed_points,
                reserved_ids=reserved_ids, id_max=id_max, **bridge)
            if done is not None:
                return done
            if live_click:
                return _not_live_here(out, out.click_how, cancel_event,
                                      region=region_word)
        else:
            # the seeds of one cell: every run of the loop below is one
            # step of the same stage (a re-run without the refused seeds is
            # that stage with fewer points); the join, the ids and the
            # finish are the full run's own, below
            clicks = _ClickSteps(**bridge)

    def _engine(keep, elevation):
        points = (None if seed_points is None
                  else [seed_points[i] for i in keep])
        if clicks is not None:
            res = clicks.step(points, keep)
            if res is not None:
                return res
            if live_click:
                return None             # the caller refuses (no flood)
        return reshape_empty_region(
            existing_roi, {"membrane": wall_roi}, spec, roi=roi,
            dark_channels=(dict(dark_rois or {}) if outside == "faces+dark"
                           else None),
            seed_layer=seed_layer, seed_points=points,
            dark_threshold=dark_threshold, spacing=spacing,
            cancel_event=cancel_event, progress=progress,
            elevation=elevation, outline=outline_roi)

    keep = list(range(len(seed_points or ())))
    skipped = []
    memo_now = None
    if clicks is not None:
        # the seeds the last click's run refused ("apart") are remembered
        # in the holder with the list they were judged against and the
        # stage that judged them: while clicks are only added (that list
        # is a prefix of this one, the same stage, the labels unchanged) a
        # refused seed cannot regain touch, so it is left out of the first
        # step.  Handing every refused seed in again at each click, and
        # taking it out again, put a change at each one's place, and the
        # click fill's one window spanned them all -- the whole patch,
        # seconds a click, growing with every refusal.  The rule this
        # gives is placement-time judgement: a seed refused when it was
        # placed stays refused.  A run without a kept stage judges every
        # seed at once instead, and can refuse a seed that the refused
        # one's cell stood in front of, which the click-by-click run keeps
        # (the refused seed was never there for it) -- a few voxels, in a
        # corner; recorded
        memo_now = [(tuple(int(v) for v in p), int(r))
                    for p, r in zip(seed_points, reserved_ids)]
        memo = stage_holder.get("apart")
        stage_now = stage_holder.get("stage")
        if (memo and stage_holder.get("key") == stage_key
                and stage_now is not None and memo[2] is stage_now
                and np.array_equal(existing_roi,
                                   getattr(stage_now, "existing", None))
                and memo_now[:len(memo[0])] == list(memo[0])):
            skipped = [i for i, s in enumerate(memo_now) if s in memo[1]]
            if skipped:
                keep = [i for i in keep if i not in skipped]
                print("[ADD] click markers refused last time stay out: {}"
                      .format([reserved_ids[i] for i in skipped]))
    res = _engine(keep, elevation_roi)
    if res is None:
        return _not_live_here(out, clicks.why, cancel_event,
                              region=region_word)
    # a seed whose cell does not touch its object's is refused, and the
    # engine runs again without it -- the answer is the one it would have
    # given had that seed never been placed (its neighbours keep the space
    # it took).  The walls do not read the markers: the run's own wall
    # picture is handed in again
    joined = None                   # the loop's last join, reused below
    joined_for = None
    while grouped and res.status == "ok":
        joined = _join_cell_seeds(
            res.result, [seed_points[i] for i in keep],
            [reserved_ids[i] for i in keep], existing_roi,
            radius_um=float(params.get("click_radius_um", 1.0)),
            spacing=spacing)
        joined_for = (res.result, list(keep))
        why = joined[3]
        new = [keep[j] for j, w in enumerate(why) if w == "apart"]
        if not new:
            break
        skipped += new
        keep = [i for i in keep if i not in new]
        print("[ADD] click markers not touching their cell, run again "
              "without them: {}".format([reserved_ids[i] for i in new]))
        if _cancelled(cancel_event):
            out.status = "cancelled"
            return out
        res = _engine(keep,
                      elevation_roi if elevation_roi is not None
                      else getattr(res, "elevation", None))
        if res is None:
            return _not_live_here(out, clicks.why, cancel_event,
                                  region=region_word)
    out.by_click = bool(getattr(res, "by_click", False))
    if (clicks is not None and memo_now is not None
            and stage_holder.get("stage") is not None):
        # remembered for the next click (the step cleared the holder),
        # with the stage that judged them
        stage_holder["apart"] = (tuple(memo_now),
                                 {memo_now[i] for i in skipped},
                                 stage_holder.get("stage"))
    if clicks is not None and out.by_click:
        out.click_how = clicks.how
        out.click_window = clicks.window
    elif stage_holder is not None and seed_points is not None:
        # the full run answered: say why (the bridge or the stage did)
        why = (clicks.why if clicks is not None else out.click_how) or ""
        out.click_how = (why if why.startswith("full run")
                         else "full run ({})".format(
                             why or _click_fill_off_why(spec)))
        out.click_window = int(np.prod(np.shape(existing_roi)))
    elif seed_points is not None:
        # no prepared box was handed in (Spread, walls not redone, a box
        # over the live limit, a whole patch not filled as you click): one
        # line all the same, so a slow click says why it was slow
        out.click_how = "full run ({})".format(
            _click_fill_off_why(spec, holder=False))
        out.click_window = int(np.prod(np.shape(existing_roi)))
    out.messages = list(res.messages or [])
    for key in ("n_dropped_dark", "n_dropped_thin", "n_nuclei_slivers",
                "n_markers_over_kept", "n_fragments_dropped",
                "n_outside_outline", "n_slivers", "n_tiles"):
        setattr(out, key, int(getattr(res, key, 0) or 0))
    out.unfilled_tissue_fraction = res.unfilled_tissue_fraction
    # the walls come back with a "no markers" refusal too: a whole patch's
    # page keeps them, so the retry does not build them again
    out.elevation = getattr(res, "elevation", None)
    if res.status != "ok":
        return _engine_refusal(out, res.refusal_reason, outline=outline,
                               budget=budget)
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
        engine = res.result
        if grouped:
            # markers sharing an id: one cell grown from several seeds (the
            # refused ones were left out of the last run); the loop's last
            # join judged these same inputs -- it is reused, not redone
            if (joined is not None and joined_for[0] is engine
                    and joined_for[1] == keep):
                (engine, mapping, out.absorbed_ids, why, _apart) = joined
            else:
                (engine, mapping, out.absorbed_ids, why,
                 _apart) = _join_cell_seeds(
                    engine, [seed_points[i] for i in keep],
                    [reserved_ids[i] for i in keep], existing_roi,
                    radius_um=float(params.get("click_radius_um", 1.0)),
                    spacing=spacing)
            out.seed_why = [None] * len(seed_points)
            for j, i in enumerate(keep):
                out.seed_why[i] = why[j]
            for i in skipped:
                out.seed_why[i] = "apart"
            out.apart_ids = [reserved_ids[i] for i in sorted(skipped)]
        else:
            mapping, out.absorbed_ids = _reserved_mapping(
                seed_points, reserved_ids, existing_roi,
                radius_um=float(params.get("click_radius_um", 1.0)),
                spacing=spacing)
        if out.absorbed_ids:
            out.messages.append(
                "click markers absorbed (their ball was already taken): "
                "{}".format(out.absorbed_ids))
        result = _mint_to(engine, mapping)
    else:
        if not _ids_fit(next_label, len(present), id_max):
            out.status = "refused"
            out.refusal = ADD_S1_IDS_FULL
            return out
        result, mapping = _mint(res.result, next_label)
    out.faint_pairs = [(mapping.get(int(a), int(a)), mapping.get(int(b), int(b)),
                        float(m)) for a, b, m in (res.faint_pairs or [])]
    # two seeds of one cell are no pair, and an engine id no marker kept
    # (a cell left empty) is no object
    out.faint_pairs = [p for p, (a, b, _m) in zip(
        out.faint_pairs, res.faint_pairs or [])
        if p[0] != p[1] and (not grouped or (int(a) in mapping
                                             and int(b) in mapping))]
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
    outline_box = None
    if outline_roi is not None and outline != "none":
        # the outline as the engine used it: the box's crop of the patch's
        # outline (or the box crop handed in)
        from .reshape.region import _region_crop

        outline_box = np.asarray(_region_crop(
            outline_roi, roi, tuple(np.shape(existing_roi)),
            what="the tissue outline"), bool)
    _finish(out, result, params, cancel_event, step, outline=outline_box)
    if out.status == "ok" and not out.ids:
        out.status = "refused"
        out.refusal = ADD_S1_ALL_FILTERED
        out.result = None
    return out


#: The refusal reason of a run after a click on a box or a whole patch
#: over ``ADD_LIVE_FULL_RUN_MAX_VOXELS`` where the click fill steps aside
#: (flat spots, switched off, a flood that could not be compiled, a
#: failure): no full flood runs, nothing is drawn, the markers stay, and
#: the page says to press Preview (which fills the region whole).
NOT_LIVE_HERE = "not-live-here"
#: The refusal reason of the same run while the click fill is still being
#: compiled (:func:`click_fill_compiling`): nothing runs, nothing is drawn,
#: the markers stay and nothing is remembered against the region -- the
#: page says the clicks wait (``ADD_S1_LIVE_PREPARING``) and runs again
#: once the solver check has answered.
LIVE_PREPARING = "live-preparing"


def _live_preparing(out, cancel_event):
    """``out`` refused with :data:`LIVE_PREPARING` (no sentence: the page
    says its own), or cancelled when a Stop came meanwhile.  No console
    line of the click fill (``click_how`` empty): the page prints the
    wait."""
    if _cancelled(cancel_event):
        out.status = "cancelled"
        out.result = None
        return out
    out.status = "refused"
    out.refusal = None
    out.refusal_reason = LIVE_PREPARING
    out.result = None
    out.by_click = False
    out.click_how = ""
    out.click_window = 0
    return out


def _not_live_here(out, why, cancel_event, *, region="whole patch"):
    """``out`` refused with :data:`NOT_LIVE_HERE` (no sentence: the page
    says its own gate), or cancelled when a Stop came meanwhile.  ``why``
    -- the console words of why the full run would have answered;
    ``region`` -- "box" or "whole patch", for the console."""
    if _cancelled(cancel_event):
        out.status = "cancelled"
        out.result = None
        return out
    why = str(why or "")
    if why.startswith("full run"):
        why = why[len("full run"):].strip()
    out.status = "refused"
    out.refusal = None
    out.refusal_reason = NOT_LIVE_HERE
    out.result = None
    out.by_click = False
    out.click_how = "not run as you click {}".format(why).strip()
    out.click_window = 0
    out.messages.append("[ADD] the {} is not filled as you click here: "
                        "press Preview to fill it".format(region))
    return out


def _engine_refusal(out, reason, *, outline, budget):
    """``out`` refused for the engine's ``reason``, in the page's words
    (moved out of ``_run_cells`` unchanged; the click-by-click fill says
    its refusals the same way)."""
    from .reshape.region import ENGINE_REFUSALS

    out.status = "refused"
    out.refusal_reason = reason
    if reason == "too-big":
        out.refusal = too_big_sentence(budget)
    elif reason in _PAGE_REFUSALS:
        # the page's own words (what to change); the engine's sentence
        # is kept for the log.  Bridge gaps and Dark below do not shape
        # the outline of the objects you cleared: only another outline
        out.refusal = _PAGE_REFUSALS[reason]
        if reason == "no-markers-outline" and outline == OUTLINE_CLEARED:
            out.refusal = ADD_S1_NO_MARKERS_OUTLINE_CLEARED
        out.messages.append("[ADD] engine refused: {}".format(
            ENGINE_REFUSALS.get(reason, reason)))
    else:
        out.refusal = ENGINE_REFUSALS.get(reason, ADD_ENGINE_NO_MARKERS)
    return out


#: Set once this process found it cannot run the click-by-click fill (numba
#: missing, or no folder for its compiled code): every Cells Preview then
#: takes the full run, with no sentence on the page.
_INCREMENTAL_OFF = False

#: Set once the solver check (:func:`numba_answer`) has answered in this
#: process, however it ended: until then a flood not compiled yet is being
#: compiled, after it a flood not compiled never will be.
_CLICK_FILL_SETTLED = False


def click_fill_compiling() -> bool:
    """True while the click fill's flood is still being compiled in this
    process (the solver check has not answered, the fill is not switched
    off): a cold flood is a wait, not a fallback."""
    from .reshape import incremental as INC

    return (not _INCREMENTAL_OFF and not _CLICK_FILL_SETTLED
            and not INC.kernel_is_warm())


def _click_fill_compiling(spec) -> bool:
    """:func:`click_fill_compiling`, for a ``spec`` the click fill would
    take once compiled (the fill to the walls with click markers)."""
    from .reshape import incremental as INC

    return click_fill_compiling() and INC.is_click_fill(spec)


def _click_fill_possible(spec) -> bool:
    """True when the click-by-click fill may answer for ``spec`` in this
    process (the flood compiled, not switched off, the fill to the walls
    with click markers on a wall picture it redoes)."""
    from .reshape import incremental as INC

    return (not _INCREMENTAL_OFF and INC.kernel_is_warm()
            and INC.is_click_fill(spec))


def _click_stage(*, spec, roi, existing_roi, spacing, wall_roi, dark_rois,
                 dark_threshold, outline_roi, elevation_roi, cancel_event,
                 holder, key, progress=None, allow_flat=False):
    """The holder's stage for ``key`` -- kept, or prepared again when it is
    missing or anything but the clicks changed.  ``(stage, None)``;
    ``(None, None)`` when this box keeps the full run (flat spots,
    remembered for the key); ``(None, (reason, sentence))`` for prepare's
    own refusals.  Raises what ``prepare`` raises (the caller drops the
    stage).

    ``allow_flat`` ("Always update as I click", a live run): flat spots
    never send the box to the full run -- a remembered full run is
    prepared again, and the stage is kept despite them (one console line
    says so).  Without it a stage kept despite flat spots answers nothing
    (``(None, None)``: the full run answers exactly) and stays in the
    holder for the next live click."""
    from .reshape import incremental as INC

    if (not allow_flat and holder.get("full") and holder.get("key") == key
            and holder.get("spec") == spec
            and np.array_equal(existing_roi, holder.get("existing"))):
        return None, None
    stage = holder.get("stage")
    if (stage is not None and holder.get("key") == key and stage.spec == spec
            and np.array_equal(existing_roi, stage.existing)):
        if not allow_flat and getattr(stage, "flat_override", None):
            # a Preview over a stage kept despite its flat spots: the full
            # run is the exact answer; the stage waits for the next click
            return None, None
        return stage, None
    holder.clear()
    # the strip names the wait (the first click of a box or a patch): the
    # page says "patch" on a whole patch
    _say(progress, "add_prepare")
    if elevation_roi is None:
        _say(progress, "elevation")
    stage, refusal = INC.prepare(
        existing_roi, {"membrane": wall_roi}, spec, roi=roi,
        dark_channels=(dict(dark_rois or {})
                       if spec.params["outside"] == "faces+dark" else None),
        dark_threshold=dark_threshold, spacing=spacing,
        elevation=elevation_roi, outline=outline_roi,
        cancel_event=cancel_event, allow_flat=bool(allow_flat))
    if stage is not None and getattr(stage, "flat_override", None):
        print("[ADD] click fill kept despite {} flat pairs (Always update "
              "is on)".format(int(getattr(stage, "n_flat_pairs", 0) or 0)))
    if refusal is not None and refusal[0] == INC.FULL_RUN:
        # flat spots: the full run answers, nothing is shown
        print(refusal[1])
        holder.update(full=True, key=key, spec=spec,
                      existing=np.array(existing_roi, copy=True))
        return None, None
    return stage, refusal


def _stage_how(stage, reused) -> str:
    """The console line's words for a stage that answers: "stage prepared"
    or "stage reused", and the coincidental flat pairs the flat-spot rule
    let through when there are any ("stage prepared, 9 flat pairs")."""
    how = "stage reused" if reused else "stage prepared"
    n = int(getattr(stage, "n_flat_pairs", 0) or 0)
    if n:
        how += ", {} flat pair{}".format(n, "" if n == 1 else "s")
    return how


def _click_fill_off_why(spec, *, holder=True) -> str:
    """Why the click-by-click fill does not answer for ``spec`` here (the
    console line's words); ``holder=False`` -- the page handed in no
    prepared box (it keeps none for this box)."""
    from .reshape import incremental as INC

    growth = spec.params.get("growth", "flood")
    if growth == GROW_SPREAD:
        return "Spread"
    if growth == GROW_PATCH:
        return "the whole patch's fill with a background"
    if spec.walls not in INC.CLICK_FILL_WALLS:
        return "walls not redone"
    if not INC.is_click_fill(spec):
        return "not the fill to the walls"
    if not holder:
        return "no prepared box here: over the live limit"
    if _INCREMENTAL_OFF:
        return "switched off for this session"
    if not INC.kernel_is_warm():
        return ("the flood could not be compiled" if _CLICK_FILL_SETTLED
                else "the flood is not compiled yet")
    return "the click fill cannot run"


def _say(progress, stage) -> None:
    """``progress(stage, 0, 1)``; a raising callback is swallowed."""
    if progress is None:
        return
    try:
        progress(stage, 0, 1)
    except Exception:
        pass


def _fell_back(holder, key, why) -> None:
    """The holder remembers that the click fill could not take this key
    (switched off, the flood that could not be compiled, a failure): the
    page then stops running the region as you click until the key changes.
    A flood still being compiled is never recorded
    (:func:`_click_fill_compiling`): that is a wait."""
    holder.update(key=key, fallback=str(why))


def _click_fill_failed(holder, exc) -> str:
    """A step, a finish or a prepare went wrong: the stage is dropped, and
    a numba problem switches the click fill off for this process.  Returns
    the console line's words for why the full run answered."""
    global _INCREMENTAL_OFF
    from .reshape.compete import NUMBA_CACHE_FAILED, NUMBA_MISSING
    from .reshape.engine import ReshapeCancelled

    holder.clear()
    if str(exc) in (NUMBA_MISSING, NUMBA_CACHE_FAILED):
        _INCREMENTAL_OFF = True
        print("[ADD] click-by-click fill off for this session ({}); "
              "the fill runs whole".format(exc))
        return "numba cannot run it"
    if isinstance(exc, ReshapeCancelled):
        print("[ADD] click-by-click fill stopped; its stage is dropped")
        return "stopped"
    print("[ADD] click-by-click fill failed ({!r}); this Preview "
          "runs whole and the box is prepared again".format(exc))
    return "the stage failed"


class _ClickSteps:
    """The engine run of ``_run_cells`` taken by the stage when click
    markers share a reserved id (the seeds of one cell): every seed is its
    own marker on the stage, named by its place in the full click list
    (plus one), so a run without some seeds keeps the others' work.
    :meth:`step` answers like ``reshape_empty_region`` -- the raw cells
    numbered as the engine numbers its markers (ascending among the seeds
    whose ball kept a voxel) -- or ``None``: the full run answers this
    run and every later one of this Preview."""

    def __init__(self, *, spec, roi, existing_roi, spacing, wall_roi,
                 dark_rois, dark_threshold, outline_roi, elevation_roi,
                 cancel_event, progress, holder, key, allow_flat=False):
        self._stage_args = dict(
            spec=spec, roi=roi, existing_roi=existing_roi, spacing=spacing,
            wall_roi=wall_roi, dark_rois=dark_rois,
            dark_threshold=dark_threshold, outline_roi=outline_roi,
            elevation_roi=elevation_roi, cancel_event=cancel_event,
            holder=holder, key=key, progress=progress,
            allow_flat=bool(allow_flat))
        self._spec = spec
        self._cancel = cancel_event
        self._progress = progress
        self._holder = holder
        self._key = key
        # the compile state is read before the fill's: the worker warms the
        # kernel between two reads, and a cold read followed by a warm one
        # must not be remembered as a fallback
        compiling = _click_fill_compiling(spec)
        self.off = not _click_fill_possible(spec)
        #: the console line's words: how the first step found the stage,
        #: the voxels every step flooded, why the full run answered
        self.how = ""
        self.window = 0
        self.why = ("" if not self.off
                    else "full run ({})".format(_click_fill_off_why(spec)))
        if self.off and not compiling:
            _fell_back(holder, key, self.why)

    def step(self, points, keep):
        from types import SimpleNamespace

        from .reshape.engine import ReshapeCancelled
        from .reshape.region import ENGINE_REFUSALS, REGION_STAGES

        if self.off:
            return None
        holder = self._holder
        uids = [int(i) + 1 for i in keep]
        try:
            before = holder.get("stage")
            stage, refusal = _click_stage(**self._stage_args)
            if stage is None:
                # flat spots, or prepare's own refusal: the full run
                # answers (and says the refusal its own way)
                self.off = True
                self.why = ("full run (flat spots)" if refusal is None
                            else "full run (prepare refused: {})".format(
                                refusal[0]))
                return None
            if not self.how:
                self.how = _stage_how(stage, stage is before)
            holder.clear()              # no stage while it changes
            if self._progress is not None:
                try:
                    self._progress("add_cut",
                                   REGION_STAGES.index("add_cut") + 1,
                                   len(REGION_STAGES))
                except Exception:
                    pass
            st = stage.step(points, uids, cancel_event=self._cancel)
            raw = stage.raw_result()
            holder["key"] = self._key
            holder["stage"] = stage
        except (Exception, ReshapeCancelled) as exc:
            self.why = "full run ({})".format(_click_fill_failed(holder, exc))
            self.off = True
            if not isinstance(exc, ReshapeCancelled):
                _fell_back(holder, self._key, self.why)
            return None
        self.window += int(st.voxels_flooded)
        absorbed = set(int(i) for i in st.absorbed_ids)
        lut = np.zeros(max(uids or [0]) + 1, np.int32)
        for j, u in enumerate([u for u in uids if u not in absorbed],
                              start=1):
            lut[u] = j
        messages = list(stage.messages)
        if st.status != "ok":
            messages.append(ENGINE_REFUSALS.get(st.refusal_reason,
                                                st.refusal_reason))
        return SimpleNamespace(
            status=st.status, refusal_reason=st.refusal_reason,
            result=lut[raw], messages=messages, elevation=stage.elevation,
            faint_pairs=[], unfilled_tissue_fraction=None,
            n_fragments_dropped=int(st.n_fragments_dropped), by_click=True)


def _cells_by_click(out, *, spec, roi, existing_roi, spacing, params,
                    wall_roi, dark_rois, dark_threshold, outline_roi,
                    elevation_roi, seed_points, reserved_ids, id_max,
                    cancel_event, progress, holder, key, allow_flat=False):
    """Cells ▸ Click markers ▸ Fill to the walls, redone only where the
    clicks changed it (``reshape.incremental``); ``None`` when the full run
    must answer instead -- the fill is not the fill to the walls with click
    markers, the flood is not compiled yet in this process
    (``incremental.warm_kernel``), the process switched it off, the wall
    picture is not one of ``incremental.CLICK_FILL_WALLS``, the box has
    flat spots (``incremental.flat_survey``: remembered for the key), or
    anything went wrong (the stage is then dropped and the full run answers
    in the same call: a numba problem never reaches the page as a
    sentence).

    ``holder`` -- a dict the caller keeps between Previews; this function
    keeps ``holder["stage"]`` (the prepared box: its flood, raw cells and
    smoothed shapes; never shared with the main thread) and
    ``holder["key"]``.  The stage is reused only when ``key`` equals the
    kept key, the spec is the same and ``existing_roi`` holds the same
    voxels; else it is prepared again (about one full flood of the box).
    While a step runs the holder holds no stage, so a Stop or an error
    mid-step never leaves a half-updated stage to reuse.

    ``key`` is the CALLER's and must change whenever anything but the
    clicks changes: the box wall picture's key (the box, the patch, the run
    epoch, the walls channel / kind / chain, the spacing, the half-size
    factor); the labels (``id(volume)``, the volume epoch, the labels
    layer's name); ``cells_spec(params)``; the dark threshold's value; the
    sorted dark channel names; the outline's key.  The markers and their
    reserved ids never invalidate it (that is the point), and ``min_size``,
    ``auto_smooth``, ``smooth_sigma``, ``clear_border`` and ``edge_margin``
    are the finish's arguments, read from ``params`` on every call.

    The answer equals the full run's except that the faint pairs and the
    unfilled share are not measured (empty / ``None``, the log line says
    ``n/a (by click)``); ``result`` is a copy, never the stage's array.
    ``allow_flat`` ("Always update as I click" on, a live run): the box's
    flat spots never send it to the full run (see :func:`_click_stage`);
    a tie there may then fall differently from the full run's by a few
    voxels."""
    from .reshape.engine import ReshapeCancelled
    from .reshape.region import REGION_STAGES

    compiling = _click_fill_compiling(spec)   # before the fill's state (as _ClickSteps)
    if not _click_fill_possible(spec):
        out.click_how = "full run ({})".format(_click_fill_off_why(spec))
        if not compiling:
            # a flood still being compiled is a wait, never remembered
            _fell_back(holder, key, out.click_how)
        return None
    try:
        before = holder.get("stage")
        stage, refusal = _click_stage(
            spec=spec, roi=roi, existing_roi=existing_roi, spacing=spacing,
            wall_roi=wall_roi, dark_rois=dark_rois,
            dark_threshold=dark_threshold, outline_roi=outline_roi,
            elevation_roi=elevation_roi, cancel_event=cancel_event,
            holder=holder, key=key, progress=progress,
            allow_flat=bool(allow_flat))
        if stage is None and refusal is None:
            out.click_how = "full run (flat spots)"
            return None                 # this box keeps the full run
        if refusal is not None:
            reason, sentence = refusal
            out.messages = [sentence]
            out.click_how = "stage refused ({})".format(reason)
            return _engine_refusal(out, reason,
                                   outline=spec.params["outline"],
                                   budget=spec.params["box_budget"])
        out.click_how = _stage_how(stage, stage is before)
        holder.clear()                  # no stage while it changes
        if progress is not None:
            try:
                progress("add_cut", REGION_STAGES.index("add_cut") + 1,
                         len(REGION_STAGES))
            except Exception:
                pass
        st = stage.step(seed_points, reserved_ids, cancel_event=cancel_event)
        fin = None
        if st.status == "ok":
            if params.get("auto_smooth", True):
                _Steps(progress, ("add_smooth",))("add_smooth")
            fin = stage.finish(
                min_size=int(params.get("min_size", 0) or 0),
                auto_smooth=bool(params.get("auto_smooth", True)),
                smooth_sigma=float(params.get("smooth_sigma", 1.5)),
                clear_border=bool(params.get("clear_border", True)),
                edge_margin=int(params.get("edge_margin", 3)),
                whole_patch=bool(params.get("whole_patch", False)),
                cancel_event=cancel_event)
            result = fin.result.copy()
        holder["key"] = key
        holder["stage"] = stage
    except (Exception, ReshapeCancelled) as exc:
        out.click_how = "full run ({})".format(_click_fill_failed(holder, exc))
        if not isinstance(exc, ReshapeCancelled):
            _fell_back(holder, key, out.click_how)
        return None
    out.click_window = int(st.window_voxels)
    # ---- the answer, field for field as the full run gives it ------------ #
    from .reshape.region import ENGINE_REFUSALS

    out.by_click = True
    out.messages = list(stage.messages)
    for name in ("n_dropped_dark", "n_dropped_thin", "n_nuclei_slivers",
                 "n_markers_over_kept", "n_outside_outline", "n_slivers",
                 "n_tiles"):
        setattr(out, name, 0)
    out.unfilled_tissue_fraction = None
    out.faint_pairs = []
    out.elevation = stage.elevation
    if st.status != "ok":
        out.n_fragments_dropped = 0
        out.messages.append(ENGINE_REFUSALS.get(st.refusal_reason,
                                                st.refusal_reason))
        return _engine_refusal(out, st.refusal_reason,
                               outline=spec.params["outline"],
                               budget=spec.params["box_budget"])
    out.n_fragments_dropped = int(st.n_fragments_dropped)
    if id_max is not None and reserved_ids and max(reserved_ids) > int(id_max):
        out.status = "refused"
        out.refusal = ADD_S1_IDS_FULL
        return out
    out.absorbed_ids = list(st.absorbed_ids)
    if out.absorbed_ids:
        out.messages.append(
            "click markers absorbed (their ball was already taken): "
            "{}".format(out.absorbed_ids))
    out.whole_patch = bool(params.get("whole_patch", False))
    out.edge_ids = set(fin.edge_ids)
    out.n_emptied_by_smooth = int(fin.n_emptied_by_smooth)
    out.n_below_min_size = int(fin.n_below_min_size)
    out.ids = list(fin.ids)
    out.sizes = dict(fin.sizes)
    out.small_ids = set(fin.small_ids)
    out.result = result
    if not out.ids:
        out.status = "refused"
        out.refusal = ADD_S1_ALL_FILTERED
        out.result = None
    return out


#: the engine's outline refusals, as the page says them
_PAGE_REFUSALS = {
    "no-markers-outline": ADD_S1_NO_MARKERS_OUTLINE,
    "outline-per-box": ADD_S1_OUTLINE_PER_BOX,
}


# --------------------------------------------------------------------------- #
# the tissue outline, measured once per patch on the box-info worker
# --------------------------------------------------------------------------- #
def pack_mask(mask):
    """``(shape, packed bits)`` of a boolean volume (what a Clear keeps of
    the voxels it zeroed: 1 bit per voxel)."""
    m = np.asarray(mask, bool)
    return tuple(int(n) for n in m.shape), np.packbits(m.ravel())


def unpack_mask(shape, packed):
    shape = tuple(int(n) for n in shape)
    n = int(np.prod(shape))
    return np.unpackbits(np.asarray(packed, np.uint8), count=n).reshape(
        shape).astype(bool)


def cleared_union(patch_shape, records):
    """The union of what the Clears in force zeroed: ``records`` =
    ``[(roi, shape, packed)]`` (whole-patch slices, the box's shape, its
    packed voxels).  A bool volume ``patch_shape``."""
    out = np.zeros(tuple(int(n) for n in patch_shape), bool)
    for roi, shape, packed in records or ():
        out[tuple(roi)] |= unpack_mask(shape, packed)
    return out


def measure_outline(job, *, spacing, dark_threshold=None, cancel_event=None):
    """The tissue outline of the whole patch (bool), for ``job``:

    * ``{"mode": "shrink", "channels": {name: whole-patch array},
      "factor": Dark below, "bridge_um": Bridge gaps}`` -- the shrink-wrap
      around the stain (``reshape.outline.tissue_outline``), at the patch's
      dark threshold (``dark_threshold``; ``None`` takes it from the same
      strided sample the box-info worker does);
    * ``{"mode": "cleared", "shape": patch shape, "cleared": records}`` --
      the outline of the objects the Clears in force removed
      (``reshape.outline.cleared_outline`` over :func:`cleared_union`);
    * ``{"mode": "packed", "shape": ..., "packed": ...}`` -- an outline
      measured in an earlier session, kept packed (:func:`pack_mask`):
      unpacked here.

    Pure; reads its inputs only.  A cancel raises ``ReshapeCancelled``."""
    from .reshape.outline import cleared_outline, tissue_outline

    mode = job.get("mode")
    if mode == "packed":
        return unpack_mask(job["shape"], job["packed"])
    if mode == OUTLINE_SHRINK:
        return tissue_outline(
            dict(job["channels"]), tuple(float(v) for v in spacing),
            dark_factor=float(job.get("factor", 0.5)),
            bridge_um=float(job.get("bridge_um", ADD_BRIDGE_UM)),
            dark_threshold=dark_threshold, cancel=cancel_event)
    if mode == OUTLINE_CLEARED:
        mask = cleared_union(job["shape"], job.get("cleared") or ())
        return cleared_outline(mask, tuple(float(v) for v in spacing),
                               cancel=cancel_event)
    raise ValueError("no tissue outline to measure for {!r}.".format(mode))


def outline_display_shell(outline, spacing):
    """The outline's skin for the main and side views: about ``ADD_OUTLINE_SHOW_UM``
    thick on every axis and at least one voxel (a coarse z gets one-slice
    caps), as uint8 0 / 1."""
    from .reshape.outline import boundary_shell

    shell = boundary_shell(outline, tuple(float(v) for v in spacing),
                           thickness_um=ADD_OUTLINE_SHOW_UM)
    return shell.astype(np.uint8)


@dataclass
class ClearPlan:
    """What a Clear would take.  ``ids`` ascending; ``bbox`` the slices of
    the box around every one of them, whole (whole-volume coordinates), or
    ``None`` when nothing is taken."""

    status: str = "ok"                  # ok / cancelled
    mode: str = "box"                   # box / patch
    ids: tuple = ()
    n_outside: int = 0                  # taken objects reaching outside
    bbox: tuple | None = None
    n_voxels: int = 0                   # voxels the Clear would zero
    n_in_patch: int = 0                 # objects in the whole patch

    def log_line(self) -> str:
        return ("[ADD-CLEAR] plan {} mode={} n={} outside={} voxels={} "
                "patch_objects={} bbox={}".format(
                    self.status, self.mode, len(self.ids), self.n_outside,
                    self.n_voxels, self.n_in_patch, self.bbox))


#: an object is taken by a box Clear when MORE than this share of its
#: voxels is inside the box -- and then it is taken whole
CLEAR_INSIDE_SHARE = 0.5

#: up to this largest id the count indexes its arrays by the id itself (one
#: ``np.bincount``, no sort); above it the ids are compacted first, so a
#: huge id never sizes an array
CLEAR_DIRECT_MAX_ID = 1 << 24


def clear_plan(*, labels, mode="box", box=None, cancel_event=None):
    """Which objects of ``labels`` a Clear takes.  Reads, never writes.

    ``mode="box"``: the objects with more than half of their voxels inside
    ``box`` (a tuple of three slices); each is taken WHOLE, so
    ``n_outside`` counts the taken objects that also have voxels outside
    the box.  ``mode="patch"``: every object.  Ids of 0 or below are
    background and never taken.

    With the largest id at most ``CLEAR_DIRECT_MAX_ID`` the voxels are
    counted by id directly; above it the ids are compacted (sorted unique,
    then ranks), so the counting arrays are as long as the number of
    objects, whatever the largest id is.  Both give the same plan.
    """
    if mode not in ("box", "patch"):
        raise ValueError("clear_plan: mode must be 'box' or 'patch', not "
                         "{!r}".format(mode))
    if mode == "box" and box is None:
        raise ValueError("clear_plan: a box Clear needs the box")
    out = ClearPlan(mode=mode)
    if _cancelled(cancel_event):
        out.status = "cancelled"
        return out
    lab = np.asarray(labels)
    nz = lab > 0
    vals = lab[nz]
    if vals.size == 0:
        return out
    direct = int(vals.max()) <= CLEAR_DIRECT_MAX_ID
    if direct:
        # key = the id itself; 0 is never a key (only ids above 0 are read)
        key = vals.astype(np.intp, copy=False)
        del vals
        total = np.bincount(key)
        ids = None
        n_keys = total.size - 1

        def _keys(v):
            return v.astype(np.intp, copy=False)
    else:
        ids = np.unique(vals)
        key = np.searchsorted(ids, vals) + 1        # 1..N, 0 = background
        del vals
        total = np.bincount(key, minlength=ids.size + 1)
        n_keys = ids.size

        def _keys(v):
            return np.searchsorted(ids, v) + 1
    out.n_in_patch = int(np.count_nonzero(total[1:]))
    if _cancelled(cancel_event):
        out.status = "cancelled"
        return out
    if mode == "box":
        box = tuple(box)
        crop = lab[box]
        cvals = crop[crop > 0]
        inside = np.bincount(_keys(cvals), minlength=n_keys + 1)
        del cvals
        take = inside > CLEAR_INSIDE_SHARE * total
        take[0] = False
        out.n_outside = int(np.count_nonzero(take & (inside < total)))
    else:
        take = total > 0
        take[0] = False
    if _cancelled(cancel_event):
        out.status = "cancelled"
        return out
    chosen = np.flatnonzero(take)
    if chosen.size == 0:
        return out
    out.ids = tuple(int(i) for i in (chosen if direct else ids[chosen - 1]))
    out.n_voxels = int(total[chosen].sum())
    # the box around the taken objects: one boolean volume, three
    # projections (no box per object is needed)
    if mode == "patch":
        sel = nz
    else:
        sel = np.zeros(lab.shape, bool)
        sel[nz] = take[key]
    del key
    bounds = []
    for axis in range(sel.ndim):
        other = tuple(a for a in range(sel.ndim) if a != axis)
        hit = np.flatnonzero(sel.any(axis=other))
        bounds.append(slice(int(hit[0]), int(hit[-1]) + 1))
    out.bbox = tuple(bounds)
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
    #: Inside each cell: how many cells of the other run have a voxel in
    #: the box, and the patch's own brightness levels (measured on the
    #: WHOLE session array, once per patch; ``None`` when cached already)
    inside_count: int | None = None
    patch_levels: object | None = None
    levels_error: str | None = None
    #: Cells: the tissue outline measured here (whole patch, bool) when it
    #: was asked for, its display skin when asked for, and how many markers
    #: the outline dropped from the run count (with how many were left)
    outline: np.ndarray | None = None
    outline_shell: np.ndarray | None = None
    outline_error: str | None = None
    n_outside_outline: int | None = None


def count_ids(labels) -> int:
    """How many distinct non-zero ids ``labels`` holds (one bincount when
    the ids are small, else a sort)."""
    lab = np.asarray(labels)
    if lab.size == 0:
        return 0
    lo, hi = lab.min(), lab.max()
    if lo >= 0 and int(hi) <= (1 << 24):
        counts = np.bincount(lab.ravel().astype(np.int64, copy=False))
        return int(np.count_nonzero(counts[1:]))
    vals = np.unique(lab)
    return int(np.count_nonzero(vals))


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
                 roi=None, image_roi=None, levels_sources=None,
                 cancel_event=None, outline=None, outline_job=None,
                 want_shell=False, nuclei_source=None) -> AddBoxInfo:
    """The one box-info job.  ``dark_sources`` maps each channel name
    to ONE level's 3D array of the WHOLE patch (only ``[::stride]`` is
    read, here, on the worker); ``dark_cache`` is the cached threshold or
    ``None``.  ``existing_roi`` and ``nuclei_roi`` / ``image_roi`` on a
    box are box crops the main thread copied; ``run_guide_roi`` is a view
    of the box (read only); ``dark_rois`` are the layers' own arrays, on a
    box as on the whole patch, cut to the box by the engine here on the
    worker (an array already the box's shape is taken as the crop).
    On the Nuclei preset ``image_roi`` (the Brightness channel's crop)
    yields the histogram and its Auto (B4.2).

    Inside each cell (``params["look_for"] == "inside"``): the cells of
    the other run with a voxel in the box are counted (``run_guide_roi``),
    and, when ``levels_sources`` = ``(nuclear, cells)`` -- the WHOLE
    session arrays, read only here -- the patch's brightness levels are
    measured once (the page caches them per patch).  A Cells whole patch
    (``params["whole_patch"]``) is never over the box limit: it is filled
    to the walls in one go, whatever Grow by says (:func:`cells_growth`),
    and the run count asks the engine for that same rule.  On a whole
    patch ``existing_roi`` is the live session volume itself (read only
    here, under the Add lock), not a copy.

    ``outline`` -- the tissue outline the Preview will get (the run
    argument ``outline=``), so the run count drops the same markers.
    ``outline_job`` (see :func:`measure_outline`) measures it here first,
    once per patch (the page caches ``info.outline``); ``want_shell`` adds
    its display skin.  With an outline chosen and none given or measured,
    the run count is left out (pending).  ``nuclei_source`` -- the nuclei
    channel layer's own array, cut to ``roi`` here (on the worker) for
    Markers ▸ Nuclei masks' Auto when no ``nuclei_roi`` crop is given.
    The worker never raises on a knob combination the spec refuses:
    the run count is then left out."""
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
        if preset != "cells" and params.get("look_for") == LOOK_INSIDE:
            if run_guide_roi is not None:
                info.inside_count = count_ids(run_guide_roi)
            if _cancelled(cancel_event):
                info.status = "cancelled"
                return info
            if levels_sources is not None:
                from .add_derive import patch_levels

                nuclear, cells = levels_sources
                try:
                    info.patch_levels = patch_levels(
                        nuclear, cells, tuple(float(v) for v in spacing))
                except (ValueError, TypeError) as exc:
                    # a plain-words refusal of the recipe's checks (a flat
                    # channel, no bright level): said under Look for nuclei
                    info.levels_error = str(exc)
            if _cancelled(cancel_event):
                info.status = "cancelled"
                return info
        # a Cells whole patch is filled to the walls in one go: never over
        # the box limit (the fill has its own ceiling, said by the page)
        growth = cells_growth(params)
        whole_patch = (bool(params.get("whole_patch", False))
                       and preset == "cells")
        info.over_limit = (preset == "cells" and total > budget
                           and not whole_patch)
        outline_mode = params.get("outline", "none") or "none"
        # a tissue outline turns the dark plumbing on whatever the tick
        # says, as in the Preview
        outside = ("faces+dark" if params.get("dark_outside", True)
                   or outline_mode != "none" else "faces")
        info.dark_threshold = dark_cache
        if info.dark_threshold is None and dark_sources and preset == "cells":
            from .reshape.region import patch_dark_threshold

            info.dark_threshold = patch_dark_threshold(
                dict(dark_sources), spacing=spacing, stride=max(1, int(stride)))
        if _cancelled(cancel_event):
            info.status = "cancelled"
            return info
        # the tissue outline, on either preset (the page sends a job only
        # when one is wanted; on Nuclei the dark threshold is Cells' cached
        # one when there is one, else the outline takes it from the same
        # strided sample -- one mask either way)
        if outline_job is not None:
            try:
                info.outline = measure_outline(
                    outline_job, spacing=spacing,
                    dark_threshold=info.dark_threshold,
                    cancel_event=cancel_event)
            except (ValueError, TypeError) as exc:
                # a plain-words refusal of the outline's checks: logged, and
                # the Preview names what is missing
                info.outline_error = str(exc)
            outline = info.outline
        if want_shell and outline is not None:
            info.outline_shell = outline_display_shell(outline, spacing)
        if _cancelled(cancel_event):
            info.status = "cancelled"
            return info
        if (preset == "cells" and source == SOURCE_NUCLEI
                and nuclei_roi is None and nuclei_source is not None):
            # the whole layer: cut to the box first (never a whole read);
            # on a whole patch its strided sample, as the Brightness card's
            # histogram is (a float64 copy of the patch was 0.8 GB at 100
            # million voxels); a box keeps its whole crop
            nuc = (nuclei_source if roi is None
                   else nuclei_source[tuple(roi)])
            if (params.get("whole_patch")
                    and int(np.prod(np.shape(nuc)))
                    > int(BRIGHTNESS_SAMPLE_VOXELS)):
                st = dark_stride(np.shape(nuc), int(BRIGHTNESS_SAMPLE_VOXELS))
                nuc = nuc[(slice(None, None, st),) * len(np.shape(nuc))]
            nuclei_roi = np.asarray(nuc, np.float64)
        if (preset == "cells" and source == SOURCE_NUCLEI
                and nuclei_roi is not None):
            from .add_segment import _resolve_threshold

            thr = _resolve_threshold(np.asarray(nuclei_roi, np.float64), "otsu")
            info.nuclei_auto = None if thr is None else float(thr)
        if (preset == "cells" and source == SOURCE_RUN
                and run_guide_roi is not None and not info.over_limit
                and (outside == "faces" or info.dark_threshold is not None)
                and (outline_mode == "none" or outline is not None)):
            from .reshape import ReshapeSpec
            from .reshape.region import find_region_seeds
            from .reshape.spec import DEFAULT_PARAMS

            if roi is None:
                roi = tuple(slice(0, int(n)) for n in existing.shape)
            try:
                spec = ReshapeSpec(
                    target="region", seeds=SOURCE_RUN,
                    walls=params.get("wall_kind", "membrane"),
                    params={"wall_channel": "membrane", "ridge": False,
                            "outside": outside,
                            "dark_otsu_factor": float(params.get("dark_otsu_factor", 0.5)),
                            "dark_clearance_um": float(params.get("dark_clearance_um", 2.0)),
                            "core_erode_um": float(params.get("core_erode_um", 1.0)),
                            "box_budget": budget,
                            "growth": growth,
                            "outline": outline_mode,
                            "bridge_um": float(params.get(
                                "bridge_um", DEFAULT_PARAMS["bridge_um"])),
                            "whole_patch": whole_patch})
                rep = find_region_seeds(
                    existing, {}, spec, roi=roi,
                    dark_channels=(dict(dark_rois or {})
                                   if outside == "faces+dark" else None),
                    seed_layer=run_guide_roi, dark_threshold=info.dark_threshold,
                    spacing=spacing, cancel_event=cancel_event,
                    outline=(outline if outline_mode != "none" else None))
            except ValueError:
                # a knob combination the spec or the engine refuses: the
                # Preview names it; the box-info line has no run count
                rep = None
            if rep is not None and rep.refusal_reason == "outline-per-box":
                rep = None
            if rep is not None:
                # the engine's own counts: the markers left after every drop
                # (0 when the outline dropped the last ones) and the outline's
                # share of the drop
                if outline_mode != "none":
                    info.n_outside_outline = int(rep.n_outside_outline)
                info.run_count = (int(rep.n_found), int(rep.n_in_region)
                                  if rep.ok else 0)
    except ReshapeCancelled:
        info.status = "cancelled"
    return info
