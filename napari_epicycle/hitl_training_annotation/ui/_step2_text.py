"""Step 2's words -- Qt-free, no controller import.

Mirrors ``fusion_reconstruction/_step3_pr_text.py`` / ``segmentation/
_step4_text.py``: every on-screen sentence of the Step 2 look lives here as a
constant, so the help guides' parity test (brick H) and the panel read ONE
source.  Brick L owns the Session and the Curation-Patches sentences; the
Curation right column, Slices, Training and the loss plot add theirs in
bricks P / A / T.

``display_round`` is the ONE helper every screen place that names a round
goes through (D16, closed by the owner 2026-10-12: the screen shows the
disk's numbering, ``round_00`` / ``lat r0``), so it returns the disk number
unchanged; a later change of base is this function.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

# -- the one round helper (D16) ---------------------------------------------


def display_round(tag_or_index: Any) -> Optional[int]:
    """The round number the screen shows for a disk round: the 0-based
    index itself (``round_00`` -> 0, ``"lateral_2"`` -> 2, ``3`` -> 3).
    ``None`` for a tag that names no round (``"-"``, ``""``, ``None``)."""
    if tag_or_index is None or isinstance(tag_or_index, bool):
        return None
    if isinstance(tag_or_index, int):
        return int(tag_or_index) if tag_or_index >= 0 else None
    text = str(tag_or_index).strip()
    if not text or text == "-":
        return None
    digits = ""
    for ch in reversed(text.split("/")[0]):
        if ch.isdigit():
            digits = ch + digits
        elif digits:
            break
    return int(digits) if digits else None


def round_tag_text(tag_or_index: Any) -> str:
    """``round_00`` for a number, the tag as it is for a tag, ``""`` for
    none (the chip's own text)."""
    n = display_round(tag_or_index)
    return "" if n is None else f"round_{n:02d}"


# -- Session ------------------------------------------------------------------
SESSION_WORKSPACE_HEADER = "Workspace"
SESSION_WORKSPACE_DESC = (
    "The project folder every step reads and writes. "
    "Shared with Steps 1, 3 and 4.")
SESSION_WORKSPACE_PLACEHOLDER = "Pick the project folder…"
SESSION_PATCH_FOLDER_HEADER = "Patch folder"
SESSION_PATCH_FOLDER_DESC = (
    "Where the patch list and the 2D annotator read patches from.")
SESSION_PATCH_FOLDER_PLACEHOLDER = "Patch folder…"
SESSION_PATCH_FOLDER_QUALIFIER = (
    "Using the patches from Project Setup · {n} patches.")
SESSION_PATCH_FOLDER_IDLE = (
    "Pick a workspace first; then Step 1's patches are used unless you "
    "Browse.")
SESSION_PATCHES_HEADER = "Patches"
SESSION_PATCHES_DESC = (
    "Patches you extracted in Step 1 (Project Setup), with where each one "
    "stands. You pick the patch to curate in Curation.")
SESSION_PATCHES_SUMMARY = (
    "{n} patches · {t} in a trained round · {f} finalized · "
    "{p} in progress · {s} not started")
SESSION_PATCHES_EMPTY_LINES = (
    "No patches yet.", "Pick a workspace, or Browse to a patch folder.")
SESSION_LOCK_LINE = (
    "Round {n} ({bucket}) is training — the workspace and the patch "
    "list are read-only until it finishes.")

# -- the rows (Session wide, Curation narrow) ------------------------------
SEP = " · "
ROW_CHANNELS = "{c} ch"
ROW_DIMS = "{z} × {y} × {x}"
#: Session line 2's tail: how far the patch has been curated (lateral)
ROW_CURATED = "{k} of {z} {bucket} slices curated"
ROW_VOXEL_NONE = "voxels"
ROW_VOXEL_TIP_NONE = (
    "No voxel size — features are measured in voxels. Set one in "
    "Step 1 (Project Setup).")
ROW_NOT_STARTED = "not started"
ROW_NO_RECORDS = "no records yet"
ROW_CHIP_LATERAL = "lat {n}"
ROW_CHIP_AXIAL = "ax {n}"
ROW_CHIP_ROUND = "round_{n:02d}"
ROW_COUNTS_TIP = "{n} {word}"

#: stored status -> (the word on screen, colour, icon)
STATUS_WORDS: Dict[str, str] = {
    "predicted": "predicted",
    "curating": "curating",
    "accepted": "accepted",
    "staged": "finalized",
    "committed": "trained",
}
STATUS_COLOURS: Dict[str, str] = {
    "predicted": "#d4a843",
    "curating": "#a78bca",
    "accepted": "#6abf69",
    "staged": "#80deea",
    "committed": "#8c9eff",
}
STATUS_ICONS: Dict[str, str] = {
    "predicted": "mdi6.layers-outline",
    "curating": "mdi6.pencil-outline",
    "accepted": "mdi6.check-circle-outline",
    "staged": "mdi6.diamond-stone",
    "committed": "mdi6.brain",
}
NOT_STARTED_COLOUR = "#777777"
#: the "All" filter chip has no status, so its word is a neutral grey
FILTER_ALL_COLOUR = "#aaaaaa"
#: the order the icon-counts line draws its slots in (the Slices order of
#: the mockup: trained first would read backwards; the counts run in the
#: order a slice travels, predicted -> trained)
STATUS_ORDER = ("predicted", "curating", "accepted", "staged", "committed")

# -- Curation: Patches (the left column's first block) ------------------------
CURATION_PATCHES_HEADER = "Patches"
CURATION_PATCHES_DESC = "Click a patch to load it. Up / Down load too."
CURATION_PATCHES_EMPTY_LINES = (
    "No patches yet.", "Pick a workspace in Session.")

# -- Run summary ----------------------------------------------------------------
RUN_SUMMARY_HEADER = "Run summary"
RUN_SUMMARY_DESC = (
    "The active run: what the next round writes and what it trains on.")
RUN_SUMMARY_RUN = "Run:"
RUN_SUMMARY_COL_LATERAL = "Lateral (XY)"
RUN_SUMMARY_COL_AXIAL = "Axial (XZ/YZ)"
RUN_SUMMARY_ROW_LAST = "Last round"
RUN_SUMMARY_ROW_NEXT = "Next round"
RUN_SUMMARY_ROW_PAIRS = "Pairs already trained on"
RUN_SUMMARY_ROW_MODEL = "Last model"
RUN_SUMMARY_MODEL = "{run} · round {n}"
RUN_SUMMARY_NONE = "—"
RUN_SUMMARY_FOOT = (
    "Switch or create runs with the run row at the top of Curation. Slice "
    "counts by status are in Curation → Curation progress; the GPU is "
    "in Diagnostics.")
RUN_SUMMARY_EMPTY_LINES = (
    "No workspace yet.",
    "Pick one above, then make a run with New Run in Curation.")
#: the card's run word when the workspace has no active run (a new
#: workspace has none until New Run makes one), an old-layout folder's single
#: run, and the three run lists Step 2 cannot use (one word per kind: a run
#: list that cannot be read, old-layout data beside a folder of runs, run
#: folders with no list)
RUN_SUMMARY_NO_RUN = "No run yet"
RUN_SUMMARY_LEGACY = "{run} (legacy layout)"
RUN_SUMMARY_UNREADABLE = "Run list unreadable"
RUN_SUMMARY_UNREADABLE_MIXED = "Old layout, update with Browse"
RUN_SUMMARY_UNREADABLE_MISSING = "Run folders without a run list"
RUN_SUMMARY_LOCKED = "Run list busy"
RUN_SUMMARY_NONE_ACTIVE = "None active"
RUN_SUMMARY_PAIRS_UNIT = "pairs"
#: the pair count is only known for the ACTIVE bucket (the controller scans
#: that bucket's rounds); the other column says so instead of showing a 0
RUN_SUMMARY_PAIRS_ELSEWHERE = "—"
RUN_SUMMARY_PAIRS_ELSEWHERE_TIP = "Shown for the active view."

#: the guides the ``?`` buttons open (brick H writes the Session one)
GUIDE_SESSION = "step2_session.md"
GUIDE_PATCHES = "step2_patches.md"

# -- the thumbnail words (the picture's hover) -------------------------------
THUMB_TIP = (
    "The middle slice of the raw patch: channel 0 magenta, channel 1 green, "
    "a third channel blue; one channel in inferno.")
THUMB_PENDING = "The picture of this patch is being drawn."
THUMB_FAILED = "No picture: {reason}"
THUMB_FAIL_UNKNOWN = "The patch file could not be read."
THUMB_FAIL_AXES = (
    "The patch's axes are not ZYX, CZYX or ZCYX, so no picture is drawn.")
#: the shared worker's four reasons speak of a "stack"; Step 2's rows say
#: "patch" (keyed by the shared constant's NAME, as Step 3's are)
THUMB_REASONS = {
    "THUMB_FAIL_MISSING": "The patch file is missing.",
    "THUMB_FAIL_UNREADABLE": "The patch file could not be read.",
    "THUMB_FAIL_AXES": THUMB_FAIL_AXES,
    "THUMB_FAIL_ENCODE": "The patch could not be drawn.",
}


# -- pure helpers over the words ------------------------------------------------

def dims_text(shape) -> str:
    """``124 × 240 × 240`` for a (z, y, x) shape, ``""`` when it
    is not three numbers."""
    try:
        z, y, x = (int(v) for v in list(shape)[-3:])
    except (TypeError, ValueError):
        return ""
    return ROW_DIMS.format(z=z, y=y, x=x)


def _num(v: Any) -> str:
    return f"{float(v):.2f}"


_SPACING_BY_SOURCE = {
    "step1": "{z} × {y} × {x} µm (from Step 1)",
    "file": "{z} × {y} × {x} µm (from the file)",
    "user": "{z} × {y} × {x} µm (set by you)",
}


def voxel_text(spacing: Optional[Dict[str, Any]]) -> str:
    """The voxel-size line of a ``patch_spacing`` entry in Step 3's own
    words (``spacing_row_text``), or ``ROW_VOXEL_NONE`` -- the amber
    "voxels" -- for none."""
    if not isinstance(spacing, dict):
        return ROW_VOXEL_NONE
    try:
        z, y, x = (float(v) for v in spacing.get("zyx"))
    except (TypeError, ValueError):
        return ROW_VOXEL_NONE
    tpl = _SPACING_BY_SOURCE.get(spacing.get("source"),
                                 _SPACING_BY_SOURCE["file"])
    return tpl.format(z=_num(z), y=_num(y), x=_num(x))


# ---------------------------------------------------------------------------
# Curation right column, run row, foot (brick P)
# ---------------------------------------------------------------------------
CUR_RUN_LABEL = "Run:"
CUR_NEW_RUN_BTN = "New Run"
CUR_MANAGE_BTN = "Manage"
CUR_MANAGE_RUNS_ACTION = "Manage runs…"
CUR_MANAGE_CAPTION = "Bring training pairs into this run"
CUR_IMPORT_PAIRS_ACTION = "Import pairs from another run…"
CUR_IMPORT_EXTERNAL_ACTION = "Import external pairs…"
#: the same two actions as a box names them ("press Import pairs again")
CUR_IMPORT_PAIRS_WORD = "Import pairs"
CUR_IMPORT_EXTERNAL_WORD = "Import external pairs"

CUR_VIEW_HEADER = "View"
CUR_VIEW_DESC = (
    "Which bucket you curate and train. XY → lateral; XZ + YZ → pooled axial.")
CUR_VIEW_LATERAL = "Lateral (XY)"
CUR_VIEW_AXIAL = "Axial (XZ/YZ)"

CUR_PROGRESS_HEADER = "Curation progress"
CUR_PROGRESS_DESC = "Slices in this run, all patches, by status."
CUR_PROGRESS_NOTE = (
    "Trained = in a round model already ({trained}). Finalized = waiting for "
    "the next Train. Next round: lateral {lat}, axial {ax}.")
# Not in the original wording list (invented by the module, owner to confirm):
CUR_PROGRESS_NOTE_NO_ROUNDS = (
    "Trained = in a round model already (no round yet). Finalized = waiting "
    "for the next Train. Next round: lateral {lat}, axial {ax}.")

CUR_MODEL_HEADER = "Model"
CUR_MODEL_DESC = "What Predict in the 2D annotator uses."
CUR_MODEL_USE_LABEL = "Use:"
CUR_MODEL_PRETRAINED_LABEL = "Pretrained:"
CUR_MODEL_ACTIVE = "Active: {text}"
CUR_MODEL_PRETRAINED_NOTE = (
    "Used only with “Cellpose pretrained”, or for a bucket with no "
    "round model yet.")
CUR_MODEL_EXTERNAL_NOTE = "Not used with an external file."
CUR_MODEL_AUTO_UPDATE = "Use the newest round model after each Train"
CUR_MODEL_CHANNEL1_LABEL = "Channel to segment"
CUR_MODEL_CHANNEL2_LABEL = "Helper channel"
CUR_MODEL_INFERENCE_TITLE = "Inference parameters"

CUR_BTN_OPEN = "Open 2D annotator"
CUR_BTN_ACCEPT = "Accept"
CUR_BTN_FINALIZE = "Finalize accepted"
CUR_BTN_DISCARD = "Discard"
CUR_BTN_REVERT = "Revert"

CUR_STAGE_READY = "Ready · {patch} · {bucket} · {sel}"
# Not in the original wording list (invented by the module, owner to confirm):
CUR_STAGE_NO_PATCH = "Ready · no patch loaded"
CUR_STAGE_NO_SLICES = "Ready · no slices in this view yet"
CUR_STAGE_LOCKED = (
    "Locked · round {n} ({bucket}) is training · cancel it in Training")
CUR_STAGE_SEL_ONE = "1 slice selected"
CUR_STAGE_SEL_MANY = "{k} slices selected"
CUR_STAGE_SEL_NONE = "no slice selected"

# The Active line the controller writes (neutral, no accent).
CUR_MODEL_ACTIVE_NAME = "{run} · {tag}"
CUR_MODEL_ACTIVE_LINE = "Active: lat: {lat}  |  ax: {ax}"
CUR_MODEL_ACTIVE_FILE = "Active: lat + ax: {name} (file)"
CUR_MODEL_ACTIVE_NONE = "Active: none (manual / SAM only)"

# The Curation lock line.  (Polish: the run row is NOT
# in the lock set, so the sentence no longer says "the run".)
CUR_LOCK_LINE = (
    "Round {n} ({bucket}) is training — slices and the model are "
    "read-only until it finishes.")

# The run picker's one item: ``Cell (whole cell · lat 2 · ax 1 · 83 pairs)``.
# The numbers are the LAST trained round's disk number per bucket (the same
# number the Run summary's "Last round" shows), a dash with no round yet.
RUN_PICKER_ROUNDS = "lat {lat} · ax {ax}"
RUN_PICKER_PAIRS = "{n} pair"
RUN_PICKER_PAIRS_MANY = "{n} pairs"
RUN_PICKER_NO_ROUND = "—"

# -- No run yet ---------------------------------------------------------------
# Step 2 keeps every slice, pair and model in a run, and a folder with no run
# gets no Step 2 file.  An action that needs a run says which of these four
# states the folder is in, then offers New Run (never for an unreadable list).
#: the same sentence the shell's banner shows (``ui_shell._workspace_facts``
#: holds the banner's copy; a test pins the two together)
MSG_STEP2_NO_RUN = "No run yet — click New Run to start training."
RUN_NEED_WORKSPACE = "Pick a workspace first."
RUN_NONE_ACTIVE = (
    "No run is active — pick a run in the Run picker, or click New Run.")
#: the Run picker's own top line when runs exist but none is active
RUN_PICKER_NONE_ACTIVE = "No run is active — pick one below, or click New Run."
RUN_LIST_UNREADABLE = (
    "Step 2 could not read this project's list of runs, so it saves nothing "
    "here. Nothing was deleted; keep the folder as it is and ask for help.")
#: the two other run lists Step 2 cannot use: old-layout data beside a folder
#: of runs with no list (Browse offers the update), and run folders with no list
RUN_LIST_MIXED = (
    "This project has old-layout Step 2 data beside a folder of runs, so Step "
    "2 saves nothing here. Press Browse and pick this same folder to update "
    "it.")
#: the run list could not be OPENED when the project was opened (a lock held
#: by another program): every write is refused until it opens again
RUN_LIST_LOCKED = (
    "The run list could not be opened just now (another program may hold "
    "it); close and reopen the project, or press Browse and pick it again.")
RUN_LIST_MISSING = (
    "This project has Step 2 run folders but no list of runs, so Step 2 saves "
    "nothing here. Nothing was deleted; keep the folder as it is and ask for "
    "help.")
#: the box an action without a run shows ({action} = the button's own word)
RUN_ASK_TITLE = "No run yet"
#: the same box when the project has runs but none is active
RUN_ASK_TITLE_PICK = "No run is active"
RUN_ASK_BODY = (
    "Step 2 keeps your slices and models in a run; make one, then press "
    "{action} again.")
RUN_ASK_BODY_PICK = (
    "Step 2 keeps your slices and models in a run; pick a run in the Run "
    "picker (your slices from before are not kept there), or click New Run "
    "to keep them, then press {action} again.")
RUN_ASK_NEW_RUN = "New Run…"
RUN_ASK_NOT_NOW = "Not now"
RUN_NEED_WORKSPACE_TITLE = "No workspace yet"
RUN_UNREADABLE_TITLE = "Step 2 cannot save in this project"
#: New Run was pressed and the run was not made (or not switched to)
NEW_RUN_FAILED_TITLE = "The run was not made"
NEW_RUN_EXISTS = (
    "A folder for the run '{handle}' is already there. Pick another name.")
NEW_RUN_FAILED = "The run could not be made: {error}."
NEW_RUN_SWITCH_FAILED = (
    "The run '{handle}' was made, but Step 2 could not switch to it: "
    "{error}. Pick it in the Run picker.")
#: a run switch that was refused or failed (the screen is left as it is)
RUN_SWITCH_FAILED_TITLE = "The run was not switched"
RUN_SWITCH_SAVE_FAILED = (
    "Step 2 could not save the current run, so the switch was cancelled: "
    "{error}; check that the workspace folder is still there.")
#: picking an existing run while none is active, with slices or a patch on
#: screen that belong to no run
RUN_SWITCH_NO_RUN_TITLE = "Switch to the run '{name}'?"
RUN_SWITCH_NO_RUN_BODY = (
    "Your slices from before a run was active are not kept; make a New Run "
    "to keep them, or switch and lose them.")
RUN_SWITCH_NO_RUN_GO = "Switch"
#: the second button of a switch cancelled by a drawing that could not be
#: saved (its cause may not go away: the run's folder was moved)
RUN_SWITCH_ANYWAY = "Switch anyway (the drawing is lost)"
#: leaving a workspace with a drawing that could not be saved
STATE_LEFT_DRAWING_LOST = (
    "A drawing that could not be saved was not kept.")
RUN_FOLDER_GONE = "The run '{name}' has no folder any more. Pick another run."
RUN_SWITCH_FAILED = "The switch to the run '{name}' failed: {error}."
#: a training round launched where the run list cannot be read
TRAIN_NO_RUN_LIST = (
    "Training could not read this project's list of runs, so nothing was "
    "trained. Your staged slices are unchanged.")
#: a slice whose image and mask could not be written into the run
PAIR_NOT_SAVED_TITLE = "The slice was not saved"
PAIR_NOT_SAVED = (
    "This slice could not be saved, so it is only on screen. Press {action} "
    "again; if it fails again, check that the workspace folder is still "
    "there.")
PAIR_NOT_SAVED_DRAWING = (
    "Your drawing could not be saved, so it is only on screen; check that "
    "the workspace folder is still there.")
#: a save of the run's state that failed (the state stays in memory)
STATE_NOT_SAVED_TITLE = "Step 2 could not save"
STATE_NOT_SAVED = (
    "Step 2 could not save its state just now; your work is still open. It "
    "will try again at your next change.")
#: the busy line while the first run of a folder is made and switched to
RUN_MAKING = "Making the run '{name}' …"
#: the busy line of any other run switch
RUN_SWITCHING = "Switching to '{name}' …"
#: a refused drawing save cancels a run switch, as a refused state save does
RUN_SWITCH_DRAWING_FAILED = (
    "Step 2 could not save your drawing, so the switch was cancelled: "
    "{error}; check that the workspace folder is still there.")
#: Train when a drawing could not be saved before the round
TRAIN_DRAWING_NOT_SAVED = (
    "Step 2 could not save your drawing, so nothing was trained: {error}; "
    "check that the workspace folder is still there.")
#: a no-run question that cannot be shown (no screen): nothing is switched
RUN_SWITCH_NO_RUN_UNASKED = (
    "The run '{name}' was not switched to: the slices on screen belong to no "
    "run and could not be asked about.")
#: Train when the save before the round was refused
TRAIN_SAVE_REFUSED_TITLE = "Training did not start"
TRAIN_SAVE_REFUSED = (
    "Step 2 could not save before training, so nothing was trained: {error}. "
    "Press Train again; if it fails again, check the workspace folder.")
#: New Run when the run list could not be written just now (a lock)
NEW_RUN_LIST_LOCKED = (
    "Another program had the run list open for a moment, so nothing was "
    "made. Press New Run again.")
#: why a drawing could not be saved, as the cancelled switch says it
DRAWING_CAUSE_FOLDER = "the run's folder could not be used"
DRAWING_CAUSE_NO_RUN = "there is no run to save it in"
DRAWING_CAUSE_OTHER = "the write failed (the console says why)"
#: leaving a workspace (Browse) whose state could not be saved
STATE_LEFT_RESCUED = (
    "Step 2 could not save the project you left, so its unsaved state was "
    "put in {path}.")
STATE_LEFT_LOST = (
    "Step 2 could not save the project you left, and the rescue copy failed "
    "too; its unsaved changes are lost.")

# The one hint line above the accordion (the shell's flow title).
FLOW_HINT = ("Select a patch → curate slices in the active view → finalize → "
             "train a round")

# -- brick A: the Slices list and Training's staged rows --
ROW_SLICE_NAME = "{view} slice {n}"
ROW_CHIP_IMPORTED = "imported"
ROW_LABELS = "{n} labels"
ROW_LABELS_ONE = "1 label"
ROW_PLANE_DIMS = "{h} × {w}"
FILTER_ALL_WORD = "All"
FILTER_CHIP_FORMAT = "{word} {n}"
# the Slices header's right-hand text: the loaded patch and the bucket
SLICES_HEADER_PATCH = "{patch} · {view}"
SLICES_VIEW_WORDS = {"lateral": "lat", "axial": "ax"}
SLICES_EMPTY_LINES = (
    "No slices in this view yet.",
    "Open the 2D annotator and Predict, or draw one.")
STAGED_EMPTY_LINES = (
    "Nothing finalized in this view yet.",)

SLICE_THUMB_TIP = (
    "The slice as you saw it, with its labels drawn over it; one channel in "
    "inferno, two or three magenta / green / blue.")
SLICE_THUMB_PENDING = "The picture of this slice is being drawn."
SLICE_THUMB_FAILED = "No picture: {reason}"
SLICE_THUMB_REASONS = {
    "THUMB_FAIL_MISSING": "The slice's image file is missing.",
    "THUMB_FAIL_UNREADABLE": "The slice's image file could not be read.",
    "THUMB_FAIL_AXES": "The patch's axes do not fit, so the slice cannot "
                       "be cut from it.",
    "THUMB_FAIL_ENCODE": "The slice could not be drawn.",
}
SLICE_THUMB_BUSY = "The slice is being saved; its picture follows."
SLICE_THUMB_NO_SOURCE = "This slice has no image file to draw."


# ---------------------------------------------------------------------------
# Training (brick T) -- moved from
# _step2_training.py at integration
# ---------------------------------------------------------------------------
TRAIN_STAGED_TITLE = "Staged for training"
TRAIN_SETUP_TITLE = "Training setup"
STAGED_DESC = (
    "Finalized {bucket} slices the next Train adds, from every patch. "
    "Each thumbnail is the slice with its labels."
)
STAGED_COUNT = (
    "{n} new · round {r} trains on {t} {bucket} pairs in all "
    "({p} from rounds 0–{q})."
)
STAGED_COUNT_FIRST = "{n} new · round {r} trains on {t} {bucket} pairs in all."
STAGED_OTHER_NONE = (
    "{Other}: nothing finalized. Switch the view in Curation to train {other}."
)
STAGED_OTHER_SOME = (
    "{Other}: {n} finalized — this Train does not use them. "
    "Switch the view in Curation to train {other}."
)
SETUP_DESC = "What the next Train writes, where it starts and how long it runs."
WRITES_LABEL = "Writes:"
WRITES_LINE = "{file} · round {r}"
WRITES_OTHER_IDLE = "{other}: not in this round"
MODE_LABEL = "Mode:"
# D15: the controller identifies the mode by these substrings, so both
# key words MUST stay ("cumulative" CTRL:968, "scratch" CTRL:1422/1440/3491).
MODE_CONTINUE = "Continue from the last round (cumulative)"
MODE_SCRATCH = "Start again from the starting model (from scratch)"
STARTING_MODEL_LABEL = "Starting model"
STARTING_FILE_LABEL = "Start from a file (optional)…"
STARTING_NOTE = (
    "Starting model and file are used for round 0, or with “from scratch”."
)
STARTING_GREY_REASON = (
    "Not used this round — the line above says where it starts."
)
EPOCHS_LABEL = "Epochs"
BATCH_LABEL = "Batch size"
DEVICE_LABEL = "Device:"
ADVANCED_TITLE = "Advanced"
TRAIN_LOCK_LINE = (
    "Round {n} ({bucket}) is training — the staged list and the setup "
    "are read-only until it finishes."
)
TRAIN_BTN_TEXT = "Train"
CANCEL_BTN_TEXT = "Cancel"
STAGE_IDLE = (
    "Ready · Train writes {name} from {n} {bucket} pairs (run {run}). "
    "Cancel works while a round trains."
)
STAGE_IDLE_BARE = "Ready · nothing staged yet"
STAGE_RUNNING = (
    "{run} · {bucket} round {r} · epoch {e} / {N} · loss {loss}"
)
STAGE_STARTING = "{run} · {bucket} round {r} · starting…"
#: the Training bar's text while a round runs (epoch / epochs)
STAGE_BAR_FORMAT = "{e} / {N}"
LOSS_TITLE = "Loss curves"
LOSS_LIVE_CAPTION = "{fam} r{r} · epoch {e} · {loss}"

# The six hint cases (D19; the controller's _compute_init_hint branches).
HINT_ROUND0_SCRATCH = "Starts from {source} — the first {bucket} round."
HINT_ROUND0_LOCKED = (
    "The run’s setup fixes where the first round starts ({display}) — "
    "the starting model and file are not used. Create a new run to change it."
)
HINT_CUMULATIVE_PRIOR = "Starts from {model} — round {n}’s {bucket} model."
HINT_CUMULATIVE_MISSING = (
    "The last round’s model is missing — starts from {base} instead."
)
HINT_SCRATCH_FILE = (
    "Starts from {file} again, every round. Earlier rounds’ pairs are still used."
)
HINT_SCRATCH_BASE = (
    "Starts from {base} again, every round. Earlier rounds’ pairs are still used."
)

# The start-from-a-file path is not a file (the status label's old warning).
STARTING_FILE_NOT_FOUND = (
    "That file was not found — the starting model is used instead.")

HINT_CASES = {
    "round0_scratch": HINT_ROUND0_SCRATCH,
    "round0_locked": HINT_ROUND0_LOCKED,
    "cumulative_prior": HINT_CUMULATIVE_PRIOR,
    "cumulative_missing": HINT_CUMULATIVE_MISSING,
    "scratch_file": HINT_SCRATCH_FILE,
    "scratch_base": HINT_SCRATCH_BASE,
}


# The Advanced disclosure's words (the engine name under a plain one)
ADV_LEARNING_RATE = "Learning rate"
ADV_WEIGHT_DECAY = "Weight decay"
ADV_MIN_OBJECTS = "Fewest objects per image"
ADV_MIN_OBJECTS_NOTE = (
    "An image with fewer objects than this is left out of training.")
ADV_CHANNEL1 = "Channel to segment"
ADV_CHANNEL2 = "Helper channel"
ADV_NORMALIZE = "Normalize"
ADV_RESCALE = "Rescale"
ADV_FLOWS = "Recompute flows"
ADV_FLOWS_NOTE = (
    "Recompute flows: work the flow targets out during training rather than "
    "once before it.")
ADV_ENGINE_NAMES = {
    "learning_rate": "learning_rate",
    "weight_decay": "weight_decay",
    "min_train_masks": "min_train_masks",
}
ADV_SUMMARY = "{lr} · {wd} · {epochs} epochs"
# The engine name beside a plain setup label (the mockup's dim mono tag)
SETUP_ENGINE_NAMES = {
    "starting_model": "pretrained_model",
    "epochs": "n_epochs",
    "batch": "batch_size",
}

# Set once every sentence the Training section shows is a constant here
# (the guide parity test's strict check runs for every guide).
STEP2_TEXT_COMPLETE = True


# The Loss header: the idle caption (the plot's ``grading_caption`` writes
# it) and the three view words on the combo beside it.
LOSS_IDLE_CAPTION = "last: {label} · train {loss} · {n} rounds"
LOSS_MODE_WORDS = ("Overlay", "Current round", "Per round")

# Words the guides quote that other sections own (checked against the page)
CUR_MODEL_SOURCE_PRETRAINED = "Cellpose pretrained"
CUR_MANAGE_BTN_TEXT = "Manage ▾"

# Diagnostics (unchanged apart from the descs and the `?`)
DIAG_SYSTEM_HEADER = "System"
DIAG_SYSTEM_DESC = "What this computer can train on."
DIAG_LOG_HEADER = "Log output"
DIAG_LOG_DESC = "Training events, checkpoint saves and pipeline messages."


def hint_text(case: str, **kw: Any) -> str:
    """The sentence for one of the six init-hint cases (``HINT_CASES``)."""
    if case not in HINT_CASES:
        raise KeyError(f"unknown hint case {case!r}")
    return HINT_CASES[case].format(**kw)


def writes_text(file_name: str, round_number: Any) -> str:
    """``Cell_lateral_round_03 · round 3`` (the disk's numbering, D16)."""
    return WRITES_LINE.format(file=file_name, r=round_number)


def _cap(word: str) -> str:
    return word[:1].upper() + word[1:]


def staged_count_text(new: int, total: int, round_number: int, bucket: str) -> str:
    """The honest count line under the staged list."""
    prior = max(0, int(total) - int(new))
    if round_number <= 0 or prior <= 0:
        return STAGED_COUNT_FIRST.format(n=new, r=round_number, t=total, bucket=bucket)
    return STAGED_COUNT.format(n=new, r=round_number, t=total, p=prior,
                               q=round_number - 1, bucket=bucket)


def staged_other_text(other: str, n_other: int) -> str:
    """The line about the bucket this Train does not use."""
    if n_other and n_other > 0:
        return STAGED_OTHER_SOME.format(Other=_cap(other), other=other, n=n_other)
    return STAGED_OTHER_NONE.format(Other=_cap(other), other=other)


def compose_training_stage_line(
    *,
    run: str = "",
    bucket: str = "lateral",
    round_number: Optional[int] = None,
    epoch: Optional[int] = None,
    epochs: Optional[int] = None,
    loss: Optional[float] = None,
    file_name: str = "",
    n_pairs: int = 0,
) -> str:
    """The Training foot's ONE stage line: running when ``epoch`` is given
    (or ``round_number`` without a file name), else the idle sentence."""
    if epoch is not None and epochs:
        return STAGE_RUNNING.format(
            run=run, bucket=bucket, r=round_number, e=epoch, N=epochs,
            loss="—" if loss is None else f"{float(loss):.2f}")
    if epoch is None and round_number is not None and not file_name:
        return STAGE_STARTING.format(run=run, bucket=bucket, r=round_number)
    if not file_name:
        return STAGE_IDLE_BARE
    return STAGE_IDLE.format(name=file_name, n=n_pairs, bucket=bucket, run=run)


def run_picker_label(display_name: str, target: str = "", lat_rounds: Any = 0,
                     ax_rounds: Any = 0, pairs: Any = 0) -> str:
    """``Cell (whole cell · lat 2 · ax 1 · 83 pairs)``: the run picker's item
    (string only; the controller hands its summary's numbers in).
    *lat_rounds* / *ax_rounds* are the COUNTS of trained rounds the summary
    carries; the label shows the last one's disk number (count - 1)."""
    def last(n: Any) -> str:
        try:
            n = int(n)
        except (TypeError, ValueError):
            n = 0
        return str(n - 1) if n > 0 else RUN_PICKER_NO_ROUND

    try:
        p = int(pairs)
    except (TypeError, ValueError):
        p = 0
    parts = []
    if target:
        parts.append(str(target))
    parts.append(RUN_PICKER_ROUNDS.format(lat=last(lat_rounds),
                                          ax=last(ax_rounds)))
    parts.append((RUN_PICKER_PAIRS if p == 1 else RUN_PICKER_PAIRS_MANY
                  ).format(n=p))
    return f"{display_name} ({SEP.join(parts)})"


def round_from_name(round_name: str) -> Optional[int]:
    """The disk round number in ``{run}_{bucket}_round_{NN}`` / ``round_NN``."""
    m = re.search(r"round_(\d+)\s*$", str(round_name or ""))
    return int(m.group(1)) if m else None
