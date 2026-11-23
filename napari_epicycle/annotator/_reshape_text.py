"""Every sentence the Reshape tab shows about markers taken from labels.

One module, Qt-free and napari-free, so the panel, the orchestrator, the
tests and the guide's parity test all read the same words.  ASCII only:
these lines are also printed to a console that may be cp1252.

Placeholders are named; a test reads a template back by turning each
``{...}`` into a wildcard, so a sentence changed here and not in the guide
turns that test red.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# The marker list, after a Find on markers from another run or a mask file
# --------------------------------------------------------------------------- #
#: The word added to a row the touching rule left unticked.
MARKER_ROW_TOUCHING = "touching"

#: That row's tooltip: how much of the object the label touches.  True
#: whether the row is ticked or not, so it never goes stale on a re-tick.
MARKER_TOUCHING_TIP = (
    "It touches the object by {n:,} voxel{s}, {pct} % of this object."
)

#: The line beside Find markers, every row ticked.
MARKERS_FOUND_LINE = "Found {n} marker{s}"

#: ...some rows unticked by hand.
MARKERS_TICKED_LINE = "{kept} of {total} ticked"

#: ...the touching rule left some rows unticked, and they still are.
MARKERS_KEPT_LINE = (
    "{kept} marker{s} kept, {n} touching one{s2} left unticked"
)

#: ...the touching rule kept none of them (two or more listed).
MARKERS_NONE_KEPT = (
    "None of the {n} markers covers enough of this object - tick the ones "
    "you want, or paint one."
)

#: ...the same, when the list holds a single marker.
MARKER_NONE_KEPT_ONE = (
    "The only marker does not cover enough of this object - tick it if you "
    "want it, or paint one."
)

#: Preview made with no Find, and every label only touches the object:
#: the cut is refused with this as its reason.
MARKERS_ALL_TOUCHING = (
    "Every marker only touches this object - press Find markers and tick "
    "the ones you want, or paint one."
)

#: Preview made with no Find left some touching labels out.  Put in front
#: of the cut's own remarks, so it shows beside the cut or a refusal.
MARKERS_LEFT_OUT = (
    "{n} touching marker{s} left out - press Find markers to see them and "
    "tick any you want."
)

#: The count above the painting rows: the TICKED markers.
MARKERS_COUNT = "Markers: {n}"

# --------------------------------------------------------------------------- #
# The status line
# --------------------------------------------------------------------------- #
#: After a Find; ``ticked`` is what is ticked now.
STATUS_FOUND = "Object {id} - {n} marker{s} found, {ticked} ticked"

#: At Start, when the markers need a mask that is not loaded yet.
STATUS_READY_LOAD_MASK = (
    "Object {id} is ready. Load a mask to take the markers from, then press "
    "Preview."
)

#: At Start, when nothing more is needed before Preview.
STATUS_READY_PREVIEW = (
    "Object {id} is ready. Press Preview - the window stays alive while it "
    "thinks."
)

#: Preview pressed while the only unticked rows are the ones the touching
#: rule left unticked -- the user unticked nothing.
STATUS_TICK_A_TOUCHING = (
    "Tick at least one of the touching markers, or paint one."
)

#: Once the mask has been read.
STATUS_MASK_LOADED = (
    "Object {id} - mask loaded. Press Find markers to see the markers, or "
    "Preview to cut."
)

#: At review.  ``from_markers`` is :data:`REVIEW_FROM_MARKERS` or empty,
#: ``share`` the " N % assigned." part or empty.
REVIEW_STATUS = (
    "Object {id} - {n} pieces{from_markers}.{share} Scroll up to change the "
    "markers and Preview again."
)

#: The markers the cut was made from -- what the engine actually used.
REVIEW_FROM_MARKERS = " from {n} marker{s}"

# --------------------------------------------------------------------------- #
# The guide note under the mask picker
# --------------------------------------------------------------------------- #
#: The two ways a note is drawn: information in grey, a refusal or a
#: caution in amber.
GUIDE_NOTE_INFO = "info"
GUIDE_NOTE_WARN = "warn"

#: A file was picked with Browse.
GUIDE_NOTE_CHOSEN = "Chosen: {name}. Press Load mask."

#: The mask is being read.
GUIDE_NOTE_READING = "Reading the mask..."

#: The mask arrived.  ``notes`` are the loader's own sentences.
GUIDE_NOTE_LOADED = "{display} - {n:,} labels in the mask. {notes}"

#: Load mask on "another run" with no other run to read.
NO_RUN_NOTE = (
    "No other run of this patch has labels to trust. Choose \"Markers from "
    "a mask file\" to pick one on disk."
)

#: The run picker's only item when it is empty.
NO_RUN_ITEM = "No other run of this patch has labels to trust"


def plural(n) -> str:
    """``""`` for one, ``"s"`` otherwise."""
    return "" if int(n) == 1 else "s"


def none_kept_line(n) -> str:
    """The count line when the touching rule kept none of ``n`` markers."""
    if int(n) == 1:
        return MARKER_NONE_KEPT_ONE
    return MARKERS_NONE_KEPT.format(n=int(n))
