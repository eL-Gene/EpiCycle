"""Step 4's own user-facing sentences.

Qt-free.  Every sentence Step 4 puts on screen or in its log is a named
constant here, so a test can read it and a reword happens in one place.
The sentences Step 3 and Step 4 share (the ``PIPE_*`` run sentences, the
running row words, ``RUN_ROW_DEFAULT``, ``MODEL_NONE``, ``MODEL_ACTIVE_PAIR``)
live in ``ui_shell/theme.py``'s pipeline block; this module does not
import them, so it stays Qt-free (``state_manager`` imports it).

Placeholders are the table's; ``count_phrase`` makes Step 4's own count
sentences count-aware ("1 stack", "2 stacks").
"""
from __future__ import annotations

# The settings words Steps 3 and 4 share live in ui_shell (shared by both steps);
# re-exported here under the same names.
from ..ui_shell._recon_settings_text import (  # noqa: F401
    SEG_SETTINGS_TITLE,
    SEG_START_FROM_LABEL,
    SEG_PRESET_EPICYCLE,
    SEG_PRESET_USEG3D,
    SEG_PRESET_CUSTOM,
    SEG_INFERENCE_TITLE,
    SEG_MEMBRANE_LABEL,
    SEG_NUCLEI_LABEL,
    SEG_CELL_WIDTH_LABEL,
    SEG_CELL_WIDTH_AUTO,
    SEG_CELL_WIDTH_SUFFIX,
    SEG_CELL_WIDTH_PX,
    SEG_NORMALIZE_LABEL,
    SEG_RESCALE_LABEL,
    SEG_ADVANCED_TITLE,
    SEG_METHOD_DIRECT,
    SEG_JOIN_TITLE,
    SEG_JOIN_DESC,
    SEG_SIZE_DESC,
    SEG_SIZE_OFF_NOTE,
    SEG_MAX_SIZE_NONE,
    SEG_PROB_THRESH_AUTO,
    PS_FLOWS_OFF_NOTE,
    STEP4_ADVANCED_LABELS,
    STEP4_ADVANCED_GROUPS,
    STEP4_HIDDEN_KNOBS,
    RECON_HELP_LINES,
)


# ── The frame ─────────────────────────────────────────────────────────
FLOW_TITLE = "Stacks → predict → reconstruct → measure & tidy → export"
FLOW_SUBTITLE = "Run your trained model on whole stacks, then measure the cells."
SECTION_SESSION = "Session"
SECTION_SEGMENT = "Segment"
SECTION_MEASURE = "Measure & tidy"
SECTION_DIAGNOSTICS = "Diagnostics"

# ── ① Session ───────────────────────────────────────────────────────
SESSION_WORKSPACE_TITLE = "Workspace"
SESSION_WORKSPACE_DESC = (
    "The project folder every step reads and writes. Shared with Steps 1–3.")
SESSION_WORKSPACE_PLACEHOLDER = "Select workspace directory..."
BROWSE_BTN = "Browse"
SESSION_STACKS_TITLE = "Stacks"
SESSION_STACKS_DESC = (
    "Stacks you prepared in Step 1 are listed here. Ticked stacks are the "
    "ones Segment runs on.")
STACKS_NONE = "No stacks yet."
STACKS_COUNT = "{n} stacks · {k} with voxel size · "
STACKS_IN_VOXELS = "{m} in voxels"
STACKS_EMPTY_1 = "Stacks you prepared in Step 1 appear here."
STACKS_EMPTY_2 = (
    "You have not prepared any yet — or add a folder of .tif stacks.")
ADD_FOLDER_BTN = "Add a folder…"
SET_SPACING_BTN = "Set spacing…"
SET_SPACING_NEEDS_STACK = "Set spacing needs at least one stack."
SET_SPACING_NEEDS_TICK = "Tick a stack, or use a row’s ⋯ menu."
TICKED_COUNT = "{n} ticked"
PIPELINE_TITLE = "Pipeline status"
PIPELINE_DESC = "Stack counts in this run."
#: The counts' words, in order; Failed only while above zero.
PIPELINE_COUNTS = (
    "Unprocessed · Predicted · Reconstructed · Measured · Finalized")
PIPELINE_COUNT_LABELS = (
    ("unprocessed", "Unprocessed:"),
    ("predicted", "Predicted:"),
    ("reconstructed", "Reconstructed:"),
    ("qc_extracted", "Measured:"),
    ("finalized", "Finalized:"),
)
PIPELINE_FAILED_LABEL = "Failed:"
SESSION_MODEL_POINTER = "The model is chosen in Segment, above its stack list."

# Row parts
ROW_SPACING_STEP1 = "{z} × {y} × {x} µm (from Step 1)"
ROW_SPACING_FILE = "{z} × {y} × {x} µm (from the file)"
ROW_SPACING_USER = "{z} × {y} × {x} µm (set by you)"
ROW_SPACING_NONE = "voxels"
ROW_WIDE_LINE2 = "{axes} · {c} ch · {spacing}"
ROW_WARN_CHANNELS = (
    "Channels don’t fit this model: it needs {n}, this stack has {k}.")
ROW_WARN_NO_SPACING = "No voxel size — numbers will be in voxels. Set spacing…"
ROW_WARN_NO_SPACING_SHORT = "No voxel size — runs in voxels."
ROW_WARN_SAME_NAME = "Another stack already has this name."
#: N3: a stack whose axis order could not be read cannot run
ROW_WARN_NO_AXES = "Its axis order could not be read, so it cannot run."
ROW_WARN_CHANGED_SINCE_PREDICT = "The stack file changed since it was predicted."
THUMB_PLACEHOLDER = ""
MENU_CLEAR_PREDICTIONS = "Clear 2D predictions…"
MENU_SET_SPACING = "Set spacing…"
MENU_CLEAR_PREDICTIONS_HAS_3D = "Clear 2D predictions (it has a 3D result)"
CLEAR_PRED_TITLE = "Clear 2D predictions"
CLEAR_PRED_CONFIRM = (
    "Delete the 2D predictions of {stack}? It goes back to unprocessed and "
    "needs Predict again.")
CLEAR_PRED_BTN = "Delete"
CANCEL_BTN = "Cancel"

# Set spacing…
SPACING_DIALOG_TITLE = "Set spacing"
SPACING_DIALOG_DESC = (
    "The voxel size, the Z direction and the axis order of {n} stacks.")
SPACING_DIALOG_DESC_ONE = (
    "The voxel size, the Z direction and the axis order of {stack}.")
SPACING_Z_LABEL = "Z (µm):"
SPACING_Y_LABEL = "Y (µm):"
SPACING_X_LABEL = "X (µm):"
SPACING_ZORIENT_LABEL = "Z orientation:"
SPACING_ZORIENT_ITEMS = ("Basal-down (default)", "Basal-up (inverted)")
SPACING_AXES_LABEL = "Axes:"
SPACING_AXES_AUTO = "As detected"
SPACING_OK_BTN = "OK"
SPACING_CHANGE_TITLE = "Voxel size changes"
SPACING_CHANGE_CONFIRM = (
    "Changing this makes {what} out of date for {n} stacks. They will need "
    "{action} again.")
SPACING_WHAT_MEASURE = ("the measurements", "Measure")
SPACING_WHAT_PREDICT = ("the 2D predictions", "Predict")
SPACING_WHAT_BOTH = ("the measurements and the 2D predictions",
                     "Predict and Measure")
SPACING_APPLY_BTN = "Change it"
SPACING_SAVE_FAILED = (
    "The voxel size could not be saved. The log in Diagnostics says why.")

# ── ② Segment ───────────────────────────────────────────────────────
RUN_ROW_LABEL = "Run:"
NEW_RUN_BTN = "New Run"
MANAGE_RUNS_BTN = "Manage"
RUN_ROW_AFTER_RELEASE = "Named runs come after the release."
SEG_MODEL_TITLE = "Model"
SEG_MODEL_DESC = "The model that finds the cells."
SEG_ACTIVE_LABEL = "Active:"
MODEL_CHANGE_BTN = "Change…"
SEG_MODEL_TRAINED_ON = "Trained on {n} channels: {names}."
SEG_MODEL_TRAINED_UNKNOWN = (
    "The channels this model was trained on are not recorded.")
MODEL_MISSING = (
    "The active model’s files are missing. Pick it again with Change…"
)
#: One trained channel on the model card: "{name} ({index})".
SEG_MODEL_CHANNEL = "{name} ({index})"
SEG_MODEL_ROLE_MEMBRANE = "membrane"
SEG_MODEL_ROLE_NUCLEI = "nuclei"
SEG_PICK_CHANNEL = "Pick the membrane channel."
SEG_NONE_RUNNABLE = "No ticked stack can run with this model."
#: F1: nothing ticked at all -- said before "cannot run"
SEG_NOTHING_TICKED = "Tick a stack to run."
SEG_NOTHING_TO_RECONSTRUCT = "Predict the ticked stacks first."
VIEW_NO_RESULT = "The highlighted stack has no result to show yet."
SEG_TICKED_SKIPPED = "{n} ticked · {k} cannot run"
SEG_STACKS_TITLE = "Stacks"
SEG_STACKS_DESC = (
    "Ticked stacks are the ones Predict and Reconstruct run on — the same "
    "ticks as in Session.")
ROW_SHAPE = "{Z}×{Y}×{X} · {c} ch"
ROW_SHAPE_CELLS = " · {n} cells"
ROW_MADE_BY = "made by {model}"
ROW_2D_BY = "2D by {model}"
ROW_NOT_ACTIVE_MODEL = "made by {model} — not the active model"
ROW_2D_NOT_ACTIVE = "2D by {model} — not the active model"
ROW_MADE_BY_UNKNOWN = "made before models were recorded"
ROW_UNSAVED = "Edited, not saved"
ROW_RUNNING_VIEW = " · {VIEW} {c} / {t}"
ROW_2D_STALE = "2D made with other axes: Predict again"
ROW_2D_BY_UNKNOWN = "2D made before models were recorded"
ROW_LAST_RUN_FAILED = "The last {stage} failed; the earlier result is kept."
#: ``{stage}`` on screen (stored lowercase in ``failed_stage``).
ROW_STAGE_WORDS = {"predict": "Predict", "reconstruct": "Reconstruct"}
SEG_TICKED_LINE = "{n} ticked"
SEG_TICKED_VOXELS = "{stack} runs in voxels"
SEG_TICKED_VOXELS_MANY = "{m} stacks run in voxels"
#: The two spans of the ticked line are joined by this.
SEG_TICKED_JOIN = " · "
STACKS_RUNNING_HINT = "You can still click rows and use View on finished stacks."
SEG_SETTINGS_DESC = (
    "Start from the settings your Step 3 run used, or from a standard set.")
SEG_PRESET_STEP3 = "Step 3 run “{run}”"
SEG_CHANNEL_ITEM = "{i}: {name}"
#: a channel item whose role the model records (owner 2026-09-29): the
#: channel's name and what the model uses it for, e.g. "1: DAPI (nuclei)"
SEG_CHANNEL_ITEM_ROLE = "{i}: {name} ({role})"
#: no stack's layers name the channel -- the model's role only
SEG_CHANNEL_ITEM_ROLE_ONLY = "{i}: {role} (model)"
SEG_CHANNEL_NONE = "none"
SEG_CHANNEL_NAME = "ch{i}"
PREDICT_BTN = "Predict Selected"
RECONSTRUCT_BTN = "Reconstruct Selected"
CANCEL_RUN_BTN = "Cancel"
VIEW_BTN = "View"
STEP4_SEG_IDLE = "Ready · {n} stacks ticked"
STEP4_POSTPROCESSING = "Postprocessing {stack}...  [{t}]"
RECON_PARTIAL_CACHE = (
    "{stack}: the 2D predictions are incomplete. Predict it again first."
)
RECON_STALE_CACHE = (
    "{stack}: the 2D predictions were made with other axes. Predict it again "
    "first.")
RECON_SOURCE_CHANGED = (
    "{stack}: the stack file changed since it was predicted. Predict it "
    "again first.")
RECON_OVERWRITES_TITLE = "Reconstruct again"
RECON_OVERWRITES_CONFIRM = (
    "{stack} is {status}. Reconstructing it again deletes its current cells, "
    "measurements, marks and exported 3D meshes, and starts from new cells."
)
RECON_OVERWRITES_BTN = "Reconstruct"
PREDICT_OVER_3D_TITLE = "Predict again"
PREDICT_OVER_3D_CONFIRM = (
    "{stack} already has 3D cells. Predicting again replaces its 2D "
    "outlines; the 3D cells stay until you Reconstruct.")
PREDICT_OVER_3D_BTN = "Predict"
LOG_PREDICT_STACK_DONE = "{stack}: 2D predictions, {s} slices × 3 views ({t} s)"
LOG_RECON_START = "Reconstructing {n} selected stacks"
LOG_RECON_STACK_DONE = "  Done: {n} cells, saved to labels_3d.tif"
LOG_EXPORT_EXCLUDED = " ({n} deleted/border excluded)"
LOG_EXPORT_NAN = " ({k} cols NaN-filled)"
LOG_PREDICT_START = "Predicting {n} selected stacks · model {model}"
LOG_WIDTH_IN_PIXELS = "{stack}: no voxel size; cell width taken as {px} px."
#: Log-only lines of the B1a paths (raw reasons go here, never on a label).
LOG_STACK_SKIPPED = "[predict] {stack} skipped: {why}"
#: A1-10: a stack whose axis order could not be read is refused, not guessed
LOG_NO_AXES = "its axis order could not be read"
#: A1-4: one Start from value that would not apply
LOG_PRESET_SKIPPED = "[start from] {name} kept its value: the saved one would not apply"
#: F12: a Start from that cannot be brought back in a new session
LOG_START_FROM_NOT_RESTORED = (
    "[start from] '{preset}' could not be restored; the settings are "
    "EpiCycle's defaults")
LOG_RUNS_IN_VOXELS = "[predict] {stack} runs in voxels"
LOG_NO_CACHE = "[WARN] No prediction cache for {stack} — run Predict first."
LOG_STACKS_LISTED = "[session] {n} stacks listed"
LOG_MODEL_SET = "[model] active model: {model}"
LOG_PRESET_APPLIED = "[settings] start from: {preset}"
LOG_SPACING_SET = "[spacing] {stack}: {spacing}"
LOG_FOLDER_ADDED = "[session] folder added: {folder}"
LOG_CANCEL_REQUESTED = "[run] cancel requested"
MT_PREVIEW_FAILED = (
    "{stack} could not be shown. The log in Diagnostics says why.")

# ── The lock lines ────────────────────────────────────────────────
STEP4_LOCK_PREDICT = (
    "Predicting — Cancel stops it after the current stack. The workspace, "
    "the ticks, the model, the settings, Measure & tidy and the other steps "
    "are locked.")
STEP4_LOCK_RECONSTRUCT = (
    "Reconstructing — Cancel stops it after the current stack. The "
    "workspace, the ticks, the model, the settings, Measure & tidy and the "
    "other steps are locked.")
STEP4_LOCK_MEASURE = (
    "Measuring — wait for it to finish. The stack list, the edits, Save, "
    "Finalize, Export and the other steps are locked.")
STEP4_LOCK_SMOOTH = (
    "Smoothing {stack}. Wait for it to finish. The stack list, the edits, "
    "Save, Finalize, Export and the other steps are locked.")
STEP4_LOCK_SAVE = (
    "Saving {stack}. Wait for it to finish. The stack list, the edits, Save, "
    "Finalize, Export and the other steps are locked.")
STEP4_LOCK_LOAD = (
    "Loading {stack}. Wait for it to finish. The stack list, the edits, "
    "Save, Finalize, Export and the other steps are locked.")
STEP4_LOCK_EXPORT = (
    "Exporting. Wait for it to finish. The stack list, the edits, Save, "
    "Finalize, Export and the other steps are locked.")
# One owner: the Workbench's text module (the selection card shows it).
from ._workbench_text import WB_LOCKED  # noqa: E402,F401
LOCK_WORKSPACE_REFUSED = (
    "A Step 4 job is running; the workspace was not changed."
)
#: B2 (F23 / audit D13): leaving Step 4 with unsaved work.  The
#: prompt's words are today's ("unchanged prompt"), now constants.
STEP_EXIT_TITLE = "Unsaved Changes"
STEP_EXIT_BODY = (
    "You have unsaved changes (deletions, smoothing, marks or hand edits) "
    "that will be lost if "
    "you leave without saving.\n\nSave changes now?")
STEP_EXIT_SAVE_BTN = "Save & Leave"
STEP_EXIT_LEAVE_BTN = "Leave Without Saving"
#: ③'s status line and the log when the save before leaving failed: the
#: step change is stopped (D13).
STEP_EXIT_SAVE_FAILED_TITLE = "Still in Step 4"
#: R1-4: the short form for ③'s one-line status (the notice has the rest).
STEP_EXIT_SAVE_FAILED_SHORT = "Saving {stack} failed; still in Step 4."
STEP_EXIT_SAVE_FAILED = (
    "Saving {stack} failed, so you are still in Step 4. Your edits are kept; "
    "the log in Diagnostics says why.")
#: B2 UI re-audit LOW-5: the merged-columns question of Export features
#: (stacks measured with different feature sets).
EXPORT_COLS_SHARED = "• {n} shared columns across all stacks"
EXPORT_COLS_NOT_SHARED = "• ⚠ {n} columns not shared by all stacks"
EXPORT_COLS_MISSING_FROM = "    {col} — missing from: {stacks}"
EXPORT_COLS_RULE = ("\nMerge uses intersection (shared columns only).\n"
                    "Missing values filled with NaN.")
EXPORT_COLS_SHARED_ONLY = "Shared columns only (drop stack-specific)"
EXPORT_COLS_EXPORT_BTN = "Export"
#: The "What is measured?" window's title: the button's own words (owner
#: decision, 2026-10-05; it was "QC Feature Reference").  Step 4 only.
FEATURE_REFERENCE_TITLE = "What is measured?"
#: B2 R1-10: the log line of any Step 4 action refused under the lock.
LOCK_ACTION_REFUSED = "[lock] {action} refused: a Step 4 job is running"
#: The log line when a ``⋯`` menu is refused under the lock.
LOCK_MENU_REFUSED = "A Step 4 job is running; the stack menu waits until it ends."

# ── ③ Measure & tidy (built in B1b, the words live here now) ─────────
MT_STACKS_DESC = "Click a stack to load it. Only reconstructed stacks are listed."
MT_NOT_RECONSTRUCTED = (
    "{stack} is not reconstructed yet; it appears here after Reconstruct.")
MT_NOT_RECONSTRUCTED_MANY = (
    "{m} stacks are not reconstructed yet; they appear here after "
    "Reconstruct.")
MT_STACK_PILL_UNSAVED = "Unsaved edits · measurements stale"
#: unsaved label edits the last Measure already describes (its own border
#: removal): the edits are unsaved, the measurements are current
MT_STACK_PILL_UNSAVED_FRESH = "Unsaved edits"
MT_STACK_PILL_STALE = "Measurements stale"
MT_STACK_PILL_SAVED = "Saved"
MT_STACK_PILL_MEASURING = "Measuring…"
MT_STACK_PILL_UNSAVED_MARKS = "Unsaved marks"
MT_SPACING_CHANGED = "Voxel size changed since the last Measure."
MT_STACK_LINE = "{Z}×{Y}×{X} · {n} cells · µm {z} × {y} × {x} ({source})"
MT_STACK_LINE_VOXELS = "{Z}×{Y}×{X} · {n} cells · in voxels"
MT_NOTHING_LOADED = "No stack loaded."
MT_CHIP_MEASURE = "{hh:mm} · {n} cells"
MT_CHIP_MEASURE_STALE = "stale · {hh:mm} · {n} cells"
MT_CHIP_NEVER = "not measured"
MT_CHIP_REVIEW = "{g} good · {b} bad · {u} unmarked"
MT_CHIP_TIDY = "{hh:mm} · {k} thin kept"
MT_CHIP_NOT_SMOOTHED = "not smoothed"
MT_CHIP_FIX_LATER = "after the release"
MEASURE_DESC = (
    "Measure every cell — size, shape, brightness, neighbours — in real "
    "units.")
MEASURE_BTN_FIRST = "Measure cells…"
MEASURE_BTN_AGAIN = "Measure again…"
MEASURE_BTN_RUNNING = "Measuring…"
REVIEW_DESC = "Cells you mark good or bad in the Workbench are counted here."
TIDY_DESC = (
    "Round off jagged outlines. A cell too thin to survive is kept as it "
    "was.")
FIX_DESC = "Opens the 3D curator on this stack. After the release."
MT_ADV_HINT = "smoothing in µm · Reset"
TIDY_TOPOLOGY_LABEL = "Keep each cell in one piece"
FINISH_DESC = (
    "Finalize saves this stack and marks it done, so Export takes it. It "
    "locks nothing.")
FINALIZE_NEEDS = "Save and re-measure first."
FINALIZE_NEEDS_MEASURE = "Measure this stack first."
BUSY_MEASURING = "Measuring · {pct}%"
BUSY_MEASURE_STAGE = "{stage} · cell {i} of {n} · about {t} left"
BUSY_MEASURE_STAGE_ONLY = "{stage} · about {t} left"
#: B1b audit (G1-3): the percent and the time so far give the time left
#: (``run_extraction`` names no stage).
BUSY_MEASURE_ETA = "about {t} left"
QC_BAR_IDLE = "Done"
MT_LOADED = "Loaded {stack}: {shape}, {n} labels"
MT_SMOOTHED = (
    "Smoothed {n} cells ({v} voxels changed, not saved). Save keeps them.")
MT_SMOOTHED_THIN = (
    "{k} cells were too thin to smooth and were kept as they were.")
LOG_SMOOTH_METHOD = (
    "Smoothing uses the Batch Refine method: a wider crop, one snapshot, thin "
    "cells kept. Results differ voxel for voxel from smoothing before this "
    "version."
)
MT_TIERS_KEPT = "{k} marks kept."
MT_DELETED = "Removed {n} cells ({m} left, not saved). Save keeps it."
MT_UNDONE = "Undid the last {what}."
MT_UNDO_EMPTY = "Nothing to undo."
#: B1b audit (B1-2): a hand edit after a Smooth would be lost by its Undo
MT_UNDO_AFTER_PAINT = (
    "Undo is off: cells were painted by hand after the smoothing.")
REMOVE_CONFIRM = (
    "Remove {n} bad cells from {stack}? Undo can bring them back until you "
    "Save.")
RESET_CONFIRM = (
    "Reset {stack} to the cells Reconstruct made? Your edits, measurements "
    "and exported 3D meshes for it are deleted."
)
REMEASURE_FINALIZED_CONFIRM = (
    "{stack} is finalized. Measuring again takes it out of Export until you "
    "Finalize it again."
)
LOAD_UNSAVED_TITLE = "Unsaved edits"
LOAD_UNSAVED_BODY = "{stack} has edits that are not saved."
SAVE_NO_STACK = "Load a stack to save it."
RESET_NO_BACKUP = "There is no original result to go back to for {stack}."
DIAG_LOG_DESC = "Segmentation run events and output status."

# ── ③ as built in B1b: titles, buttons and the lines left to the
#    build (each recorded in the B1b report as an addition) ────────────
MT_STACKS_TITLE = "Stacks"
MT_MEASURE_TITLE = "Measure"
MT_REVIEW_TITLE = "Review"
MT_TIDY_TITLE = "Tidy"
MT_FIX_TITLE = "Fix by hand"
MT_ADV_TITLE = "Advanced"
FINISH_TITLE = "Finish this stack"
OPEN_WORKBENCH_BTN = "Open Workbench"
WHAT_MEASURED_BTN = "What is measured?"
REMOVE_BAD_BTN = "Remove bad cells"
UNDO_BTN = "Undo"
SMOOTH_OUTLINES_BTN = "Smooth outlines"
FIX_BY_HAND_BTN = "Fix by hand in 3D…"
SAVE_BTN = "Save"
FINALIZE_BTN = "Finalize"
EXPORT_OPEN_BTN = "Export…"
RESET_BTN = "Reset"
#: the strength combo (Medium = today's σ; Light half, Strong double)
SMOOTH_STRENGTHS = ("Light", "Medium", "Strong")
SMOOTH_STRENGTH_CUSTOM = "Custom"
TIDY_SIGMA_Z_LABEL = "Up and down (σ z)"
TIDY_SIGMA_XY_LABEL = "Across (σ xy)"
SIGMA_SUFFIX_UM = " µm"
SIGMA_SUFFIX_VOX = " voxels"
MT_CHANNELS_TITLE = "Channels to measure"
MT_CHANNELS_DESC = (
    "Tick the channels whose brightness is measured; the names head the "
    "brightness columns of the table.")
MT_CHANNEL_TICK = "Measure"
MT_CHANNELS_AXES = "Axes:"
MT_CHANNELS_AXES_UNKNOWN = "(read when a stack loads)"
#: ``{source}`` of MT_STACK_LINE
MT_SOURCE_WORDS = {"step1": "from Step 1", "file": "from the file",
                   "user": "set by you"}
MT_STACK_LINE_VOXELS_HEAD = "{Z}×{Y}×{X} · {n} cells · "
MT_STACK_LINE_VOXELS_TAIL = "in voxels"
#: ``{what}`` of MT_UNDONE
UNDO_WHAT_REMOVE = "Remove bad cells"
UNDO_WHAT_SMOOTH = "Smooth outlines"
MT_MEASURED = "Measured {n} cells of {stack}."
MT_SAVED = "Saved {stack}."
MT_FINALIZED = "Finalized {stack}."
MT_RESET_DONE = "Reset {stack} to the cells Reconstruct made ({n} cells)."
MT_NO_BAD = "No cell is marked bad."
MT_MESH_RUNNING = "Exporting meshes · {i} of {n}"
REMOVE_TITLE = "Remove bad cells"
REMOVE_CONFIRM_BTN = "Remove"
RESET_TITLE = "Reset"
RESET_CONFIRM_BTN = "Reset"
REMEASURE_TITLE = "Measure again"
REMEASURE_BTN = "Measure"
SAVE_TITLE = "Save"
SAVE_CONFIRM = (
    "Write {stack}’s cells, measurements, marks and maps to disk?")
#: appended to SAVE_CONFIRM / MESH_SAVE_FIRST for a finalized stack (B1-7)
SAVE_TAKES_OUT_OF_EXPORT = (
    " This takes it out of Export until it is finalized again.")
FINALIZE_TITLE = "Finalize"
FINALIZE_CONFIRM = (
    "Save {stack} and mark it finalized, so Export takes it?")
LOAD_UNSAVED_SAVE = "Save"
LOAD_UNSAVED_DISCARD = "Discard"
LOAD_UNSAVED_STAY = "Stay"
MESH_SAVE_FIRST_TITLE = "Unsaved edits"
MESH_SAVE_FIRST = (
    "{stack} has edits that are not saved. The meshes are made from the "
    "saved cells, so it is saved first.")
MESH_SAVE_FIRST_BTN = "Save and export"
EXPORT_CLOSE_BTN = "Close"
EXPORT_MESHES_BTN = "Export 3D meshes…"
EXPORT_FILE_TITLE = "Export features"
#: owner: the default file of Export features, dated
#: and counted, so a re-export never overwrites the previous table
EXPORT_DEFAULT_NAME = "features_{date}_{n}stacks.csv"

# ── The Export window ──────────────────────────────────────────
EXPORT_WINDOW_TITLE = "Export — the end of Step 4"
EXPORT_DESC = (
    "Workspace-wide: works on every finalized stack, whichever stack is "
    "loaded.")
#: through ``say``: "{n} stacks" is read with the right number (G1-5);
#: {k} counts the stacks that CAN go (B1b audit).
EXPORT_FINALIZED_COUNT = "{k} of {n} stacks can go into Export features:"
EXPORT_BLOCK = "Finalize a stack first."
EXPORT_ROW_WHY_FINALIZE = "finalize it first"
EXPORT_ROW_WHY_MEASURE = "measure and finalize it first"
EXPORT_ROW_WHY_EDITED = "edited after Finalize: measure and finalize it again"
EXPORT_ROW_UNITS_VOXELS = "in voxels, not µm"
#: B1b audit additions (additions): the loaded stack's unsaved edits
#: (G1-2), a stack finalized before Finalize recorded its files (Q1), one
#: measured before its units were recorded (Q2), marks saved after
#: Finalize (B1-7), and a list that changed while the window was open.
EXPORT_ROW_UNSAVED = "its unsaved edits are not in the table"
EXPORT_ROW_LEGACY = "finalized before its files were recorded"
EXPORT_ROW_UNITS_UNKNOWN = (
    "measured before units were recorded: measure and finalize it again")
EXPORT_ROW_WHY_REFINALIZE = "changed after Finalize: finalize it again"
EXPORT_ROWS_CHANGED = "The list changed; check the ticks and export again."
EXPORT_NONE_TICKED = "Tick a stack to export."
EXPORT_RUNNING = "Exporting… the window’s buttons wait until it ends."
EXPORT_FEATURES_BTN = "Export features ({k} stacks)"
EXPORT_FEATURES_DONE = "Exported {k} stacks ({n} cells) to {file}"
EXPORT_FEATURES_WHAT = (
    "One table of every active cell; removed and border cells are left out, "
    "and a column names the stack and the model.")
EXPORT_LAST = "Last exported {hh:mm} · {k} stacks · {file}"
EXPORT_MESH_TARGET = "the loaded stack, {stack}"
EXPORT_MESH_NONE = "Load a stack in Measure & tidy first."
EXPORT_MESH_WHAT = (
    "Saves first if there are unsaved edits, then writes one surface file "
    "per cell to {folder}.")
EXPORT_MESH_DONE = "Exported {n} meshes to {stack}/"
#: owner export 2026-09-29: the cells that get no mesh are said
EXPORT_MESH_DONE_SKIPPED = (
    "Exported {n} meshes to {stack}/ ({k} cells under 10 voxels: no mesh)")
#: the log line naming them
LOG_MESH_SKIPPED = "[export] no mesh for {k} cells under 10 voxels: {ids}"
EXPORT_BTN_LINE = (
    "Opens the export window: features from every finalized stack, 3D meshes "
    "for the loaded stack.")

# ── The state file ──────────────────────────────────────────────────
STATE_RESTORED_FROM_BACKUP = (
    "Step 4's saved state could not be read. It was kept as {file}, and "
    "Step 4 started from its last backup."
)
STATE_STARTED_EMPTY = (
    "Step 4's saved state could not be read. It was kept as {file}, and "
    "Step 4 started with no stacks."
)
STATE_RESTORED_FROM_LEGACY = (
    "Step 4's saved state could not be read. It was kept as {file}, and "
    "Step 4 started from the older state file."
)
STATE_LEGACY_UNREADABLE = (
    "The old Step 4 state file could not be read and was left as it is."
)
STATE_LEGACY_NEWER = (
    "An older EpiCycle wrote Step 4 results after this one moved to its new "
    "state file. Those results are not shown here."
)
#: log: a COPIED workspace's paths moved onto this workspace
STATE_REROOTED = (
    "[state] this workspace was copied from {old}: {n} stored paths now "
    "point into this workspace")
#: log: a delete refused because the file is not in this workspace
DELETE_OUTSIDE_REFUSED = (
    "[WARN] not deleted, outside this workspace: {path}")
#: log: the state moved from artifacts/step4/ to the workspace root
STATE_MOVED_TO_ROOT = (
    "[state] step4_state.json now lives at the workspace root; the old copy "
    "was kept as artifacts/step4/{file}")
STATE_SAVE_BLOCKED = (
    "Step 4 cannot save: its state file could not be set aside. Close other "
    "programs using it, then reopen the workspace."
)
#: Composed by ``state_reconciled_line`` ("{n} stacks" -> count_phrase).
STATE_RECONCILED = "Updated {n} stacks to match the files on disk."
STATE_SAVE_FAILED = (
    "Step 4 could not save its state. The log in Diagnostics says why."
)
#: Adopt (read-only) found an unreadable state file: nothing was renamed.
STATE_UNREADABLE_ON_ADOPT = (
    "Step 4's saved state could not be read. It will be set aside the first "
    "time you work in Step 4."
)

# ── Failure sentences ───────────────────────────────────────────────
MT_MEASURE_FAILED = (
    "Measuring {stack} failed. The log in Diagnostics says why."
)
MT_SMOOTH_FAILED = (
    "Smoothing {stack} failed. Nothing was changed. The log in Diagnostics "
    "says why."
)
MT_LOAD_FAILED = "{stack} could not be loaded. The log in Diagnostics says why."
MT_SAVE_FAILED = (
    "Saving {stack} failed. Nothing was overwritten. The log in Diagnostics "
    "says why."
)
MT_RESET_FAILED = (
    "Resetting {stack} failed. Nothing was changed. The log in Diagnostics "
    "says why."
)
EXPORT_FAILED = "The feature export failed. The log in Diagnostics says why."
EXPORT_MESH_FAILED = "The mesh export failed. The log in Diagnostics says why."

# ── Help lines (the help lines) ──────────────────────────────
# Keyed by the widget's attribute name; the painted parts of a row are keyed
# ``row_tick`` / ``row_chip`` / ``row_eye`` (① and ②), ``mt_row_click`` (③)
# and ``export_row_tick`` (the Export window).  ``flow_thresh_sb`` and
# ``cellprob_sb`` are hidden and have no line.
STEP4_HELP_LINES = {
    "run_picker_combo": (
        "The run these results belong to. This release has one run, Default; "
        "named runs, each with its own model, come later."),
    "new_run_btn": RUN_ROW_AFTER_RELEASE,
    "manage_runs_btn": RUN_ROW_AFTER_RELEASE,
    "model_change_btn": (
        "Pick the model to run — your Step 2 and Step 3 models, Cellpose’s "
        "built-in ones, or a file."),
    "row_tick": (
        "Ticked stacks are the ones Predict Selected and Reconstruct Selected "
        "run on; the two lists share the ticks."),
    "row_chip": (
        "The model that made this stack’s result; amber when it is not the "
        "model now active, so a new Reconstruct would not match it."),
    "row_eye": "Show or hide this stack in the viewer.",
    "start_from_combo": (
        "Fill the settings below from your Step 3 run or from a standard "
        "set; you can still change any of them."),
    "chan1_combo": (
        "The channel the model reads as the cell membrane. A loaded or "
        "previewed stack shows its layer names here."),
    "chan2_combo": (
        "The channel the model reads as the nuclei, or none. A loaded or "
        "previewed stack shows its layer names here."),
    # the settings both steps share (ui_shell/_recon_settings_text)
    **RECON_HELP_LINES,
    # ② foot
    "predict_btn": (
        "Make the 2D outlines, slice by slice in all three views, for every "
        "ticked stack; the slow part, on the graphics card."),
    "reconstruct_btn": (
        "Join each ticked stack’s 2D outlines into 3D cells; needs Predict "
        "first, and uses the 2D outlines already made."),
    "cancel_btn": (
        "Stop after the stack being worked on; finished stacks are kept."),
    "view_btn": (
        "Show the highlighted stack’s result in the viewer (the highlighted "
        "row, not the ticks); works while a run is going."),
    # ③ and the Export window (their widgets are B1b's)
    "mt_row_click": (
        "Load this stack to measure and tidy it; if the loaded stack has "
        "unsaved edits you are asked first."),
    "save_tiers_btn": (
        "Write this stack’s cells and measurements to disk; after this, Undo "
        "cannot go back further."),
    "finalize_btn": FINISH_DESC,
    "export_btn": EXPORT_BTN_LINE,
    "export_row_tick": (
        "Finalized stacks that go into Export features; a stack that cannot "
        "go says why."),
    "export_features_btn": (
        "Write one table of every cell from the finalized stacks, ready for "
        "your notebooks."),
    "export_meshes_btn": (
        "Save each cell of the loaded stack as a 3D surface file for "
        "rendering or printing."),
    # ①
    "browse_workdir_btn": "Choose the project folder. Every step reads and writes it.",
    "add_folder_btn": "Add every .tif stack in a folder to the list.",
    "set_spacing_btn": (
        "Set the voxel size, the Z direction and the axis order of the "
        "ticked stacks."),
    # ③ (B1b)
    "extract_features_btn": (
        "Measure every cell of the loaded stack. The Workbench opens when it "
        "is done."),
    "open_analysis_btn": (
        "Open the Workbench to look at the cells and mark them good or bad."),
    "feature_reference_btn": "List every measurement and what it means.",
    "delete_bad_btn": (
        "Remove the cells marked bad. Undo can bring them back until you "
        "Save."),
    "undo_btn": (
        "Take back the last Remove or Smooth, until you Save. The maps are "
        "not brought back: compute them again."),
    "smooth_strength_combo": "How strongly to round off the outlines.",
    "smooth_all_btn": (
        "Smooth every cell’s outline. A cell too thin to survive is kept as "
        "it was."),
    "fix_by_hand_btn": "Fix cells by hand in 3D. This comes after the release.",
    "smooth_sigma_z_sb": "How far the smoothing reaches, up and down / across.",
    "smooth_sigma_xy_sb": "How far the smoothing reaches, up and down / across.",
    "smooth_preserve_topology_cb": (
        "Keep each cell in one piece: if smoothing would split a cell, it is "
        "reconnected."),
    "reset_original_btn": (
        "Put back the cells Reconstruct made. Your edits, measurements and "
        "exported 3D meshes are deleted."),
}


def count_phrase(n: int, noun: str) -> str:
    """``"1 stack"`` / ``"2 stacks"``: no Step 4 sentence reads "1 stacks"."""
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def seg_idle_line(n: int) -> str:
    """``STEP4_SEG_IDLE`` with the count read right: "Ready · 1 stack
    ticked", never "1 stacks" (the driver's STATE_RECONCILED finding)."""
    return STEP4_SEG_IDLE.replace("{n} stacks", count_phrase(n, "stack"))


def state_reconciled_line(n: int) -> str:
    """``STATE_RECONCILED`` with the count read right: "Updated 1 stack
    ...", never "1 stacks" (the driver saw "1 stacks were ...")."""
    return STATE_RECONCILED.replace("{n} stacks", count_phrase(n, "stack"))


def fmt_count(n) -> str:
    """A count with thousands separators."""
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return str(n)


_IRREGULAR = {"mesh": "meshes"}


def say(template: str, **values) -> str:
    """*template* filled in, every count with thousands separators, and
    every "{k} <noun>s" read with the right number ("1 cell", "2 cells",
    "1 mesh", "2 meshes"), so no Step 4 sentence reads "1 stacks"."""
    import re as _re
    out = template
    for key, val in values.items():
        if isinstance(val, bool) or not isinstance(val, int):
            continue
        for noun in ("stack", "cell", "mark", "mesh"):
            plural = _IRREGULAR.get(noun, noun + "s")
            word = noun if val == 1 else plural
            out = _re.sub(r"\{" + _re.escape(key) + r"\} " + plural + r"\b",
                          lambda _m, _w=fmt_count(val) + " " + word: _w,
                          out)
    for key in [k for k in values if ":" in k]:
        # "{hh:mm}" is not a format field (it reads as "hh" with spec "mm")
        out = out.replace("{" + key + "}", str(values.pop(key)))
    fill = {k: (fmt_count(v) if isinstance(v, int)
                and not isinstance(v, bool) else v)
            for k, v in values.items()}
    try:
        return out.format(**fill)
    except (KeyError, IndexError, ValueError):
        return out
