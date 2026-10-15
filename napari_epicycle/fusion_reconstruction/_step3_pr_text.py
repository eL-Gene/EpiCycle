"""Step 3 Predict + Reconstruct: every sentence of the settings column.

Qt-free.  The words both steps share (the plain names, the seven groups,
the shared help lines, the preset names) live in
``ui_shell/_recon_settings_text.py`` and are used through
``STEP3_SETTING_KEYS``; this module holds what is Step 3's own -- the
description, the last-run item and its note, the channel items (Cellpose
codes), the cell width's line and warning (R2b), Step 3's own
help lines and the log lines of Start from.  Every sentence that reaches the
screen from the rebuilt code is a named constant here
(``test_step3_pr_text_literals.py``); the settings guide quotes them
(``test_step3_pr_guide_parity.py``).

Nouns: settings say "cells"; row counts say "labels".
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from ..ui_shell import _recon_settings_text as R

# ── The column ────────────────────────────────────────────────────────
#: the description under the Settings header: with the run clause while
#: this run has a reconstruction, without it otherwise
S3_SETTINGS_DESC = (
    "Start from this run’s last reconstruction, or from a standard set.")
S3_SETTINGS_DESC_NO_RUN = (
    "Start from a standard set, then change what you need; any change "
    "makes it Custom.")
#: the Start from item
S3_PRESET_LAST_RUN = "This run’s last reconstruction"
#: under Start from, for a manifest older than R2c (no ``manifest_version``)
S3_LAST_RUN_NOTE = "Settings that run did not record are EpiCycle’s defaults."
#: in the note slot under Start from when the last-run item is picked after
#: its reconstruction was deleted (the item is then taken off the list)
S3_LAST_RUN_VANISHED = (
    "This run’s last reconstruction is no longer there; the settings were "
    "not changed.")
#: in the same slot when "u-Segment3D defaults" cannot be read here
#: (u-Segment3D is not importable, so its preset resolves to nothing)
S3_USEG3D_UNAVAILABLE = (
    "u-Segment3D is not installed here, so its defaults cannot be read; "
    "the settings were not changed.")
#: the grey engine names on the CLOSED Advanced header (batch size is at the
#: top level in Step 3, the owner's 2026-10-06 addition)
S3_ADVANCED_ENGINE = "method · n_cpu · u-Segment3D"
#: the cell width is in µm since R2b: Step 4's shared suffix
S3_CELL_WIDTH_SUFFIX = R.SEG_CELL_WIDTH_SUFFIX
#: under the width, for the highlighted (else the first) patch Predict
#: would run that has a voxel size: Step 4's shared line
S3_CELL_WIDTH_PX = R.SEG_CELL_WIDTH_PX
#: under it, amber, while the width is not Auto and a patch Predict would
#: run has no voxel size (that patch runs with the number as pixels)
S3_NO_VOXEL_WARNING = "No voxel size — cell width falls back to pixels."
#: B (board D): under the Even out brightness / Rescale ticks -- Cellpose's
#: two mask thresholds act only on the masks, and the 3D build computes none
S3_UNUSED_KNOBS_NOTE = (
    "Outline tolerance and cell confidence are not used by the 3D build, "
    "so they are not shown.")

#: Cellpose's channel codes, unchanged: the item's INDEX is
#: the code ``_collect_seg_config`` passes
S3_CHAN1_ITEMS = ("0: gray", "1: red", "2: green", "3: blue")
S3_CHAN2_ITEMS = ("0: none", "1: red", "2: green", "3: blue")

# ── Step 3 widget names -> the shared keys ───────────────────
#: Step 3 keeps its own widget names; this alias map feeds the labels, the
#: help lines and the presets from the shared tables.
STEP3_SETTING_KEYS = {
    "diameter_sb": "cell_width_um_sb",
    "norm_cb": "normalize_cb",
    "rescale_cb": "rescale_cb",
    "batch_sb": "batch_size_sb",
    "method_combo": "method_combo",
    "n_cpu_sb": "n_cpu_sb",
    "flow_thresh": "flow_thresh_sb",
    "cellprob_thresh": "cellprob_sb",
}
for _name in R.STEP4_ADVANCED_LABELS:
    if _name not in STEP3_SETTING_KEYS.values():
        STEP3_SETTING_KEYS[_name] = _name
del _name

#: Step 3 widget -> (plain name, engine name), for the .frm4 rows
S3_ADVANCED_LABELS = {
    s3: R.STEP4_ADVANCED_LABELS[key]
    for s3, key in STEP3_SETTING_KEYS.items()
    if key in R.STEP4_ADVANCED_LABELS
}

# ── Help lines ─────────────────────────────────────────────────────────────
#: Step 3's own lines: where Step 3 differs from Step 4
S3_OWN_HELP_LINES = {
    "start_from_combo": (
        "Fill the settings below in one go: from EpiCycle’s defaults, from "
        "u-Segment3D’s own defaults, or from the settings this run last "
        "reconstructed with. You can still change any of them; the list "
        "then says Custom."),
    "chan1_combo": (
        "The channel that shows the cell outlines, as Cellpose’s code: "
        "0 gray (all channels together), 1 red, 2 green, 3 blue."),
    "chan2_combo": (
        "An optional second channel, usually the nuclei, that helps the "
        "model tell touching cells apart; 0 means none."),
    # "diameter_sb": Step 3's own pixel line until R2b; since R2b the width
    # is in µm and the shared line (``cell_width_um_sb``) is Step 3's too
    # "rescale_cb": the prediction manifest records the resize RULE since
    # R2c, not the factor (that needs the model's trained size, R6), so
    # the shared line's "The factor used is recorded" is not true here yet
    # (R2c audit MED-1)
    "rescale_cb": (
        "Used only when no cell width is set: when ticked, the image is "
        "passed to the model at its own scale; when unticked, the model "
        "resizes it by its own trained cell size. The rule used is "
        "recorded with the result."),
}

#: Step 3 widget -> help line: the shared lines through the alias map, then
#: Step 3's own over them.  The hidden knobs have none.
S3_HELP_LINES = {
    s3: R.RECON_HELP_LINES[key]
    for s3, key in STEP3_SETTING_KEYS.items()
    if key in R.RECON_HELP_LINES
}
S3_HELP_LINES.update(S3_OWN_HELP_LINES)

# ── Log lines of Start from ────────────────────────────────────────────────
S3_LOG_PRESET_APPLIED = "[SETTINGS] Start from: {preset}."
S3_LOG_PRESET_SKIPPED = "[SETTINGS] Start from: {name} was left as it was."
#: a preset value the widget could not take as given -- Qt clamped it to
#: the range or rounded it to the decimals, or the list / combo has no such
#: item (a method, a channel code); named, never dropped silently
S3_LOG_PRESET_INEXACT = (
    "[SETTINGS] Start from: {name} could not take {wanted}; it holds {got}.")
S3_LOG_CUSTOM_NOT_RESTORED = (
    "[SETTINGS] Start from was Custom; custom values are not saved, so the "
    "settings keep the values they have now.")
S3_LOG_RUN_NO_CHOICE = (
    "[SETTINGS] This run has no Start from choice saved; the settings keep "
    "the values they have now.")
S3_LOG_LAST_RUN_VANISHED = (
    "[SETTINGS] Start from: this run’s last reconstruction is no longer "
    "there, so it was taken off the list; the settings were not changed.")
S3_LOG_USEG3D_UNAVAILABLE = (
    "[SETTINGS] Start from: u-Segment3D defaults could not be read "
    "(u-Segment3D is not importable); the settings were not changed.")
S3_LOG_LAST_RUN_GONE = (
    "[SETTINGS] Start from was this run’s last reconstruction, which is no "
    "longer there; the settings keep the values they have now.")

# ── Predict (R2b, R2c) ─────────────────────────────────────────────────────
#: the log line after a patch's arrays (and its prediction manifest) are
#: on disk -- the words it had before R2c
S3_LOG_CACHE_SAVED = "Prediction cache saved: {stem}"
#: one line per patch Predict runs with the width taken as pixels (R2b)
S3_LOG_WIDTH_IN_PIXELS = (
    "[PREDICT] {patch}: no voxel size; cell width taken as {px} pixels.")
# --- R3 ---
# The section's layout (brick R3):
# the run row, the Model and Patches cards on the left, the
# pinned foot under the settings, Session's pointer line.

# ── The run row ────────────────────────────────────────────────────────────
S3_RUN_LABEL = "Run:"
S3_NEW_RUN_BTN = "New Run"
S3_MANAGE_RUNS_BTN = "Manage"

# ── The Model card ─────────────────────────────────────────────────────────
S3_MODEL_TITLE = "Model"
S3_MODEL_DESC = (
    "Two models predict the cells in 2D: one for top-down (XY) slices, one "
    "for side (XZ, YZ) slices.")
S3_MODEL_ACTIVE_LABEL = "Active:"
S3_MODEL_CHANGE_BTN = "Change…"
#: one neutral line per bucket, elided in the middle; the
#: full name is the line's tooltip
S3_MODEL_LAT_LINE = "lat: {name}"
S3_MODEL_AX_LINE = "ax: {name}"
#: a bucket with no model yet
S3_MODEL_NO_NAME = "none"
S3_MODEL_LAT_TIP = "Top-down (XY) model: {name}"
S3_MODEL_AX_TIP = "Side (XZ, YZ) model: {name}"
#: B (board D): under the two model lines, what the active models were
#: trained on (``_step3_rows.trained_line``; the Cellpose codes read from
#: the training config beside the weights).  ``{run}`` is
#: ``S3_MODEL_TRAINED_RUN`` for a Step 2 model, else empty.
S3_MODEL_TRAINED_ON = "Trained on {n} channels: {names}{run}."
S3_MODEL_TRAINED_ONE = "Trained on 1 channel: {names}{run}."
S3_MODEL_TRAINED_RUN = " · Step 2 run “{run}”"
S3_MODEL_CHANNEL = "{name} ({index})"
S3_MODEL_CHANNEL_JOIN = ", "
S3_MODEL_ROLE_MEMBRANE = "membrane"
S3_MODEL_ROLE_NUCLEI = "nuclei"
#: Cellpose code 0 for the first channel: the image read as grayscale -- no
#: role and no index are claimed for it
S3_MODEL_ROLE_GRAY = "grayscale"
S3_MODEL_TRAINED_GRAY = "Trained on one grayscale channel{run}."
#: grey: nothing is invented for a built-in model or a missing config
S3_MODEL_TRAINED_UNKNOWN = (
    "The channels this model was trained on are not recorded.")
#: the two models disagree: BOTH sets are named; ``{runs}`` is
#: ``S3_MODEL_TRAINED_DIFFER_RUNS`` / ``_RUN`` for Step 2 models, else empty
S3_MODEL_TRAINED_DIFFER = (
    "Trained on different channels: lateral {lat} · axial {ax}.{runs}")
S3_MODEL_TRAINED_DIFFER_RUNS = " Step 2 runs “{lat}” / “{ax}”."
S3_MODEL_TRAINED_DIFFER_RUN = " Step 2 run “{run}”."

# ── The Patches card ───────────────────────────────────────────────────────
S3_PATCHES_TITLE = "Patches"
S3_PATCHES_DESC = "Ticked patches are the ones Predict and Reconstruct run on."
S3_TICK_ALL_LINK = "Tick all unprocessed"
#: the ticked lines, one per button; " · {k} protected" only when k > 0
S3_TICKED_PREDICT = "Predict: {n} ticked"
S3_TICKED_PROTECTED = " · {k} protected"
S3_TICKED_RECONSTRUCT = "Reconstruct: {m} ready of {n} ticked"

# ── The pinned foot ────────────────────────────────────────────────────────
S3_PREDICT_BTN = "Predict Selected"
S3_RECONSTRUCT_BTN = "Reconstruct Selected"
S3_CANCEL_BTN = "Cancel"
S3_VIEW_BTN = "View"
#: the foot's gate sentences (Step 4's ``_refresh_segment_gates`` in Step
#: 3's words), first that applies wins
S3_GATE_PICK_MODEL = "Pick a model first."
S3_GATE_NOTHING_TICKED = "Tick a patch to run."
S3_GATE_NONE_RUNNABLE = "Every ticked patch is protected."
#: when Predict can run: the idle line, then why Reconstruct is grey
S3_GATE_READY = "Ready · {n} ticked"
S3_GATE_NOTHING_TO_RECONSTRUCT = "Predict the ticked patches first."
S3_GATE_JOIN = " · "
#: B (board D): the idle line says what a ticked protected patch does --
#: " · 1 protected (finalized) is skipped"; the statuses are the ones the
#: protected patches have, joined by ``S3_GATE_STATUS_JOIN``
S3_GATE_SKIPPED = "{k} protected ({statuses}) is skipped"
S3_GATE_SKIPPED_PLURAL = "{k} protected ({statuses}) are skipped"
S3_GATE_STATUS_JOIN = ", "
#: the same without the statuses -- for a foot too narrow for two lines
S3_GATE_SKIPPED_SHORT = "{k} protected is skipped"
S3_GATE_SKIPPED_SHORT_PLURAL = "{k} protected are skipped"

#: a Session-list click on a predicted (or further) patch: its cached
#: predictions restored, or none on disk (the words they had before R3)
#: a Session click on a patch whose predictions are on disk (A2 fold: the
#: click only checks the files -- nothing is restored any more)
S3_STATUS_PREDICTIONS_CACHED = "Predictions cached for {patch}"
S3_STATUS_NO_CACHE = "Status: {status} (no cache)"
S3_LOG_PREDICTIONS_CACHED = "Cached predictions found: {patch}"

# ── Help lines of the run row, the cards and the foot ──────────────────────
S3_LAYOUT_HELP_LINES = {
    "run_picker_combo": (
        "The run you are working in: a model applied to the patches, with "
        "its own predictions, reconstructions and curation. Switch to work "
        "on another run without disturbing the others."),
    "new_run_button": "Make a new run with its own model and results.",
    "manage_runs_button": (
        "Rename, archive or delete the runs of this workspace."),
    "model_change_btn": (
        "Pick the two models: a refined model, a Cellpose built-in, or a "
        "model file on disk."),
    # V1: the ticks are the choice (S3_LIST_HELP / S3_TICK_ALL_HELP below)
    "predict_selected_btn": (
        "Predicts the cells in 2D on the ticked patches, one slice at a "
        "time in all three views. Accepted and finalized patches are "
        "skipped."),
    "reconstruct_selected_btn": (
        "Joins the 2D predictions of the ticked patches into 3D cells. A "
        "patch must be predicted first."),
    "pipeline_cancel_btn": (
        "Stops the run after the patch it is working on."),
    "view_reconstructed_button": (
        "Shows the highlighted patch’s 3D cells over its image in the "
        "viewer."),
}

# ── Session ────────────────────────────────────────────────────────────────
#: where the hidden Model Status block was (P6)
S3_SESSION_MODEL_POINTER = (
    "The model is chosen in Predict + Reconstruct, above its patch list.")
# --- V1 ---
# The Patches card's painted rows, the ticks, the lock line and Forget.

# ── The rows ───────────────────────────────────────────────────────────────
#: the shape line under the status word, from the patch's reconstruction
#: manifest (never a TIFF header on the main thread); nothing without one
S3_ROW_SHAPE = "{z}×{y}×{x} · {n} labels"
S3_ROW_SHAPE_NO_COUNT = "{z}×{y}×{x}"
#: under an accepted / finalized row (row 20); the sentence is its tooltip
S3_ROW_PROTECTED = "protected"
S3_ROW_PROTECTED_TIP = (
    "{status} — protected; Reset to Predicted in Curation to redo it")
#: a patch whose file is gone (row 27): the row stays, untickable
S3_PATCH_MISSING = "The patch file is missing."
#: the sign in front of a row's amber line
S3_WARN_SIGN = "⚠ "
#: a ticked patch with curation edits (row 20)
S3_CHIP_HAS_CURATION = "has curation"
#: help on the row's painted parts (the shared delegate's keys)
S3_ROW_HELP_LINES = {
    "row_tick": (
        "Tick the patches Predict and Reconstruct run on. Accepted, "
        "finalized and missing patches cannot be ticked."),
    "row_chip": (
        "This patch has curation edits; running it again replaces them."),
}
#: the list and the link, now that the ticks are the choice
S3_LIST_HELP = (
    "Tick the patches to run on; Ctrl+A ticks every patch that can be "
    "ticked. Click a row to highlight it for View.")
S3_TICK_ALL_HELP = "Ticks every patch that has not been predicted yet."
S3_LAYOUT_HELP_LINES["_patch_state_list"] = S3_LIST_HELP
S3_LAYOUT_HELP_LINES["pr_tick_all_link"] = S3_TICK_ALL_HELP
#: under the list while Predict / Reconstruct runs (the ticked lines give
#: way to it; the mockup's board F)
S3_LOCK_LIST_NOTE = "You can still click rows and use View on finished patches."

# ── Re-running patches that already have results (row 20) ─────────────────
S3_REPREDICT_TITLE = "Predict Again"
S3_REPREDICT_BODY = (
    "These ticked patches already have results. Predicting again replaces "
    "their predictions:\n\n{lines}")
S3_REPREDICT_CONFIRM = "Predict again"
#: one line of the list above
S3_RERUN_LINE = "{patch} — {status}"

# ── The lock line (row 21) ─────────────────────────────────────────────────
#: jobs, in Step 4's form
S3_LOCK_PREDICT = (
    "Predicting — Cancel stops it after the current patch. The run, the "
    "ticks, the model and the settings are locked.")
S3_LOCK_RECONSTRUCT = (
    "Reconstructing — Cancel stops it after the current patch. The run, "
    "the ticks, the model and the settings are locked.")
S3_LOCK_LOADING = "Loading a patch — wait for it to finish."
S3_LOCK_CURATION_BUSY = (
    "A curation step is running — wait for it to finish.")
S3_LOCK_TRAINING = (
    "Training a model — wait for it to finish, or stop it in Model "
    "Refinement.")
S3_LOCK_PAIRS = (
    "Extracting training pairs — wait for it to finish.")
S3_LOCK_COMMIT = "Committing the triage — wait for it to finish."
S3_LOCK_SAVE = "Saving the curation — wait for it to finish."
S3_LOCK_EXTRACT_ALL = "Measuring every patch — wait for it to finish."
S3_LOCK_QC_EXTRACT = "Measuring the cells — wait for it to finish."
#: annotator sessions, in the imperative (they grey nothing here)
S3_LOCK_REFINE = (
    "A refine step is staged in the annotator — Accept or Discard it there.")
S3_LOCK_RESHAPE = (
    "A split is open in the annotator — Accept or Cancel it there.")
S3_LOCK_ADD = (
    "An Add session is open in the annotator — Accept or Cancel it there.")
S3_LOCK_BATCH_REFINE = (
    "Batch Refine is open in the annotator — finish or close it there.")
#: flag -> sentence, in priority order (specific first, ``_curation_busy``
#: last); the first set flag wins
S3_LOCK_SENTENCES = (
    ("_prediction_running", S3_LOCK_PREDICT),
    ("_reconstruction_running", S3_LOCK_RECONSTRUCT),
    ("_refinement_running", S3_LOCK_TRAINING),
    ("_extraction_running", S3_LOCK_PAIRS),
    ("_extract_all_running", S3_LOCK_EXTRACT_ALL),
    ("_qc_extract_running", S3_LOCK_QC_EXTRACT),
    ("_commit_triage_in_flight", S3_LOCK_COMMIT),
    ("_save_in_flight", S3_LOCK_SAVE),
    ("_refine_running", S3_LOCK_REFINE),
    ("_reshape_running", S3_LOCK_RESHAPE),
    ("_add_mode_running", S3_LOCK_ADD),
    ("_batch_refine_running", S3_LOCK_BATCH_REFINE),
    ("_patch_load_in_flight", S3_LOCK_LOADING),
    ("_curation_busy", S3_LOCK_CURATION_BUSY),
)
#: the two families (a test holds every flag to one of them)
S3_LOCK_SESSION_FLAGS = frozenset({
    "_refine_running", "_reshape_running", "_add_mode_running",
    "_batch_refine_running"})
#: flags whose line shows only after a short wait (they are often brief)
S3_LOCK_DELAYED_FLAGS = frozenset({"_patch_load_in_flight",
                                   "_curation_busy"})
S3_LOCK_DELAY_MS = 300

# ── Forget this patch ───────────────────────────────────────
S3_FORGET_MENU = "Forget this patch…"
S3_FORGET_TITLE = "Forget This Patch"
S3_FORGET_BODY = (
    "{patch} ({status} in this run) is no longer on disk. Forgetting it "
    "moves this run’s files for it to the trash folder:\n\n{items}\n\n"
    "Kept: the snapshots and the training archive, and every other run’s "
    "files. Everything moved can be put back from\n{trash}")
S3_FORGET_NOTHING_ON_DISK = "(no files of this run on disk)"
S3_FORGET_ITEM = "• {what}"
S3_FORGET_CONFIRM = "Forget"
S3_FORGET_DONE = "Forgot {patch}: {n} item(s) moved to {trash}."
S3_FORGET_PARTIAL = (
    "Forgetting {patch} stopped after {n} of {total} item(s); they are in "
    "{trash} and the patch is still listed.")
S3_FORGET_REFUSED_LOCKED = "Wait for the running work to finish first."
S3_FORGET_REFUSED_UNSAVED = (
    "Save or discard this patch’s curation first.")
S3_FORGET_REFUSED_PRESENT = "The patch file is there; nothing to forget."
#: the items the confirm names, as found on disk
S3_FORGET_WHAT = {
    "recon": "the 3D labels and their manifest",
    "features": "the measured features",
    "curation": "the curation",
    "predictions": "the prediction cache",
    "pairs": "the refinement pairs",
}
#: a click on a missing patch in Curation's list
S3_CURATION_MISSING_REFUSED = S3_PATCH_MISSING
#: the log lines of Forget and of a run that met a missing file
S3_LOG_SKIPPED_MISSING = "Skipped {n} patch(es) whose file is missing"
S3_LOG_FORGOT = "Forgot {patch}: {n} item(s) moved to {trash}"
S3_LOG_FORGET_FAILED = "[ERROR] Forget {patch}: {error}"
S3_LOG_FORGET_STOPPED = "[ERROR] Forget {patch} stopped: {error}"
# --- V2 ---
# Session's and Curation's rows on the shared painter, with the pictures.

# ── The pictures (moved here from ``_step3_thumbs.py``) ────────────────────
#: the tooltip of a row's picture
S3_THUMB_TIP = ("The middle slice of the raw patch: channel 0 magenta, "
                "channel 1 green, a third channel blue; one channel grey.")
#: the tooltip while the picture is being drawn
S3_THUMB_PENDING = "The picture of this patch is being drawn."
#: the tooltip when no picture could be drawn; ``{reason}`` is a sentence
S3_THUMB_FAILED = "No picture: {reason}"
#: the reason when the worker gave none
S3_THUMB_FAIL_UNKNOWN = "The patch file could not be read."
#: the shared worker's four reasons speak of a "stack" (Step 4's word);
#: keyed by the shared constant's NAME, Step 3's rows say "patch"
S3_THUMB_REASONS = {
    "THUMB_FAIL_MISSING": "The patch file is missing.",
    "THUMB_FAIL_UNREADABLE": "The patch file could not be read.",
    "THUMB_FAIL_AXES": "The patch's axes do not fit the file.",
    "THUMB_FAIL_ENCODE": "The patch could not be drawn.",
}
#: the console line of a picture that could not be drawn
S3_LOG_THUMB_FAILED = "[STEP3-THUMB] {patch}: {reason}"

# ── Session's rows ─────────────────────────────────────────────────────────
#: between the parts of a row's second line
S3_LINE_SEP = " · "
#: the parts of Session's second line, each only when known: the axes, the
#: shape (``S3_ROW_SHAPE_NO_COUNT``), these two, and from R7 the spacing
S3_SESSION_CHANNELS = "{c} ch"
S3_SESSION_LABELS = "{n} labels"
#: Session's Workspace card (A2, board B): its title and its words
S3_SESSION_WORKSPACE_TITLE = "Workspace"
S3_SESSION_WORKSPACE_DESC = (
    "The project folder every step reads and writes. Shared with Steps 1, "
    "2 and 4.")
#: the Patches card over Session's patch list (A2, board B) and its ``?``
S3_SESSION_PATCHES_TITLE = "Patches"
S3_SESSION_PATCHES_DESC = (
    "Patches you extracted in Step 1 (Project Setup). Each row shows the "
    "patch's voxel size; you tick patches in Predict + Reconstruct.")
#: the header count line under the description -- each clause only when
#: above 0 (:func:`session_header_counts`); the last one amber
S3_SESSION_HEADER_COUNTS = (
    "{n} patches · {c} curating · {f} finalized · {v} without a voxel size")
S3_SESSION_HEADER_ONE = "1 patch"
S3_SESSION_HEADER_PATCHES = "{n} patches"
S3_SESSION_HEADER_CURATING = "{n} curating"
S3_SESSION_HEADER_FINALIZED = "{n} finalized"
S3_SESSION_HEADER_NO_VOXEL = "{n} without a voxel size"
S3_SESSION_HEADER_EMPTY = "No patches yet."
#: the empty list's lines (board B, state A)
S3_SESSION_EMPTY_LINES = (
    "No patches yet.",
    "Patches you extract in Step 1 appear here.",
    "You have not extracted any yet — or use a folder of .tif patches.")
#: under the list: the folder button and the dim sentence beside it
S3_SESSION_USE_FOLDER_BTN = "Use another folder…"
S3_SESSION_USE_FOLDER_TIP = (
    "Read the patches from another folder of .tif files instead.")
S3_SESSION_USING_DEFAULT = "Using the patches from Project Setup."
S3_SESSION_USING_FOLDER = "Using {folder}."
#: the help lines of a Session row's eye, inline button and tier counts
S3_SESSION_EYE_TIP = (
    "View this patch's reconstruction in the viewer (the same as View in "
    "Predict + Reconstruct).")
S3_SESSION_TIERS_TIP = (
    "Labels per tier in this patch's saved features: curated, good, review, "
    "bad, garbage. They change when the features on disk change.")
#: the help line of Session's patch list
S3_SESSION_LIST_HELP = (
    "The patches Step 3 reads, with a picture of each one's middle slice. "
    "Click a patch to make it the active one.")
#: the Pipeline status block (board B)
S3_PIPELINE_TITLE = "Pipeline status"
S3_PIPELINE_DESC = "Patch counts in this run."
S3_PIPELINE_COUNTERS = ("Unprocessed:", "Predicted:", "Reconstructed:",
                        "Curating:", "Accepted:", "Finalized:")
S3_PIPELINE_LAST_ACTIVITY = "Last activity:"
S3_PIPELINE_RESUME = "Resume:"
S3_PIPELINE_SEP = "·"


def session_header_counts(n: int, curating: int = 0, finalized: int = 0,
                          no_voxel: int = 0) -> tuple:
    """Session's header count line as ``(head, amber)``: "{n} patches ·
    {c} curating · {f} finalized · " and the amber "{v} without a voxel
    size", each clause only when above 0; ``("No patches yet.", "")`` for
    an empty list."""
    if n <= 0:
        return S3_SESSION_HEADER_EMPTY, ""
    parts = [S3_SESSION_HEADER_ONE if n == 1
             else S3_SESSION_HEADER_PATCHES.format(n=n)]
    if curating > 0:
        parts.append(S3_SESSION_HEADER_CURATING.format(n=curating))
    if finalized > 0:
        parts.append(S3_SESSION_HEADER_FINALIZED.format(n=finalized))
    head = S3_LINE_SEP.join(parts)
    if no_voxel > 0:
        return head + S3_LINE_SEP, S3_SESSION_HEADER_NO_VOXEL.format(
            n=no_voxel)
    return head, ""

# ── Curation's rows ────────────────────────────────────────────────────────
#: the marks on line 2 of a Curation row (today's words): its cells were
#: measured, and it was edited in this session
S3_CURATION_TAG_QC = "✔QC"
S3_CURATION_TAG_EDITED = "ed"
# --- R6 ---
# Model provenance:
# the "made by" chip on the Patches rows, the ⓘ and the details window.
# Folded here from R6a's ``_step3_provenance_text`` (deleted).

# ── The chip (on screen) ────────────────────────────────────────────
#: the chip on a Predict + Reconstruct row; ``{short}`` = the lateral
#: model's name, elided (``_model_provenance.short_name``)
MADE_BY = "made by {short}"
#: a patch with no recorded block (predicted before R6)
MADE_BY_UNKNOWN = "made by: unknown (predicted before 2026-10)"
#: the longest model name the chip carries before it is elided
SHORT_NAME_MAX = 28
#: the elision mark
ELLIPSIS = "…"
#: the chip, amber-bordered, when the recorded model is not the active one
#: (``{made_by}`` = ``MADE_BY`` filled in)
S3_MADE_BY_NOT_ACTIVE_TAIL = " — not the active model"
S3_MADE_BY_NOT_ACTIVE = "{made_by}" + S3_MADE_BY_NOT_ACTIVE_TAIL
#: help on the chip and on the ⓘ (the shared painter's keys)
S3_ROW_HELP_LINES["row_made_by"] = (
    "The model that predicted this patch. An amber border means it is not "
    "the model now active above.")
S3_ROW_HELP_LINES["row_tiers"] = S3_SESSION_TIERS_TIP     # (audit L8)
S3_ROW_HELP_LINES["row_info"] = (
    "Opens the details of the model that predicted this patch, with notes "
    "you can keep.")

# ── The kinds (kinds) ───────────────────────────────────────────────
#: one plain word per kind, shown in the details window beside the view
KIND_WORDS = {
    "pretrained": "Cellpose built-in model",
    "bootstrap": "Step 2 round model",
    "domain_specialized": "Step 2 round model, started from a refined "
                          "model or another run",
    "refined": "Step 3 refined model",
    "external": "Model file from outside EpiCycle",
    "unknown": "Unknown model",
}

# ── What ``view_record`` writes into ``extra.error`` ──────────────────────
#: the model file is gone (or the path is empty)
ERR_MODEL_MISSING = "The model file was not found."
#: a manifest sits beside the weights but cannot be read as one
ERR_MANIFEST_UNREADABLE = "The model's manifest could not be read."

# ── The details window ──────────────────────────────────────────────
PROV_WINDOW_TITLE = "Model details"
#: the window's title and first line; ``{patch}`` = the patch name
PROV_TITLE = "The model that made {patch}"
PROV_HEADER = PROV_TITLE
PROV_VIEW_LATERAL = "Lateral model (XY)"
PROV_VIEW_AXIAL = "Axial model (XZ, YZ)"
#: the section listing the block's own keys (schema, recorded_at, …)
PROV_RECORD = "Record"
PROV_NOTES = "Notes"
PROV_NOTES_HINT = "Anything worth keeping with this patch's model."
PROV_SAVE = "Save notes"
PROV_CLOSE = "Close"
#: Save greyed under the lock (row 21) -- written ON SCREEN, since Qt shows
#: no tooltip on a disabled widget
PROV_LOCKED = ("Notes cannot be saved while Predict or Reconstruct runs; "
               "they can be read.")
PROV_SAVED = "Notes saved."
#: the caller's ``on_save`` answered with no reason
PROV_SAVE_FAILED = "The notes could not be saved; see the log."
#: an empty value
PROV_EMPTY = "—"
#: a view the block does not carry
PROV_VIEW_MISSING = "Not recorded."
#: a patch with no block at all
PROV_NO_BLOCK = ("No model was recorded for this patch: it was predicted "
                 "before 2026-10.  Notes saved here are kept with it.")
#: Save refused: the window's patch is not in this workspace / run any more
PROV_SAVE_GONE = "This patch is no longer in the open run; nothing was saved."
#: the log lines of a save
S3_LOG_PROV_SAVED = "Model notes saved for {patch}"
S3_LOG_PROV_SAVE_FAILED = "[ERROR] Model notes for {patch}: {error}"

#: the words the details window takes (``texts=``); a test holds every key
#: to a constant above
DIALOG_TEXTS = {
    "header": PROV_HEADER,
    "lateral": PROV_VIEW_LATERAL,
    "axial": PROV_VIEW_AXIAL,
    "record": PROV_RECORD,
    "notes": PROV_NOTES,
    "notes_hint": PROV_NOTES_HINT,
    "save": PROV_SAVE,
    "close": PROV_CLOSE,
    "locked": PROV_LOCKED,
    "saved": PROV_SAVED,
    "save_failed": PROV_SAVE_FAILED,
    "empty": PROV_EMPTY,
    "view_missing": PROV_VIEW_MISSING,
    "no_block": PROV_NO_BLOCK,
    "kinds": KIND_WORDS,
}
# --- R7 ---
# Per-patch voxel size (brick R7):
# Session's spacing line, Set spacing… (its button, the row
# menu, the dialog's words, the confirm), the readers' log lines and QC
# Settings' fallback line.  Folded here from R7a's ``_step3_spacing_text``
# (deleted in R7w).  Step 3 may not import a Step 4 module, so these are
# Step 3's own; they match Step 4's ``ROW_SPACING_*`` / ``SPACING_*`` where
# the meaning is the same.

# ── Session's row (the spacing part of line 2) ─────────────────────────────
S3_ROW_SPACING_STEP1 = "{z} × {y} × {x} µm (from Step 1)"
S3_ROW_SPACING_FILE = "{z} × {y} × {x} µm (from the file)"
S3_ROW_SPACING_USER = "{z} × {y} × {x} µm (set by you)"
#: the amber word of a patch with no voxel size (it runs in voxels)
S3_ROW_SPACING_NONE = "voxels"
S3_ROW_SPACING_BY_SOURCE = {"step1": S3_ROW_SPACING_STEP1,
                            "file": S3_ROW_SPACING_FILE,
                            "user": S3_ROW_SPACING_USER}
#: the row's tooltip on the amber word
S3_ROW_SPACING_NONE_TIP = (
    "No voxel size — features are measured in voxels. Set spacing… gives "
    "this patch its own.")

# ── Set spacing… (Session's button and row menu) ───────────────────────────
S3_SET_SPACING_BTN = "Set spacing…"
S3_MENU_SET_SPACING = "Set spacing…"
S3_SET_SPACING_TIP = "Set the voxel size of the highlighted patches."
S3_SET_SPACING_NEEDS_PATCH = "Highlight a patch to set its voxel size."
S3_SPACING_DIALOG_TITLE = "Set spacing"
S3_SPACING_DIALOG_DESC = "The voxel size of {n} patches."
#: R7w: several patches, each named (the dialog acts on ALL highlighted
#: rows); ``{names}`` is filled before the dialog fills ``{n}``
S3_SPACING_DIALOG_DESC_NAMED = "The voxel size of {n} patches: {names}."
S3_SPACING_DIALOG_DESC_ONE = "The voxel size of {name}."
S3_SPACING_Z_LABEL = "Z (µm):"
S3_SPACING_Y_LABEL = "Y (µm):"
S3_SPACING_X_LABEL = "X (µm):"
S3_SPACING_OK_BTN = "OK"
S3_SPACING_CANCEL_BTN = "Cancel"

# ── The confirm (Step 4's ``_apply_spacing`` confirm, Step 3's nouns) ─────
S3_SPACING_CHANGE_TITLE = "Voxel size changes"
S3_SPACING_CHANGE_CONFIRM_ONE = (
    "Changing this makes the measured features of {name} out of date. "
    "Extract them again to update them.")
S3_SPACING_CHANGE_CONFIRM = (
    "Changing this makes the measured features of {n} patches out of date: "
    "{names}. Extract them again to update them.")
#: how many names the confirm lists before "and {k} more"
S3_SPACING_CONFIRM_MAX_NAMES = 5
S3_SPACING_CONFIRM_MORE = "{names} and {k} more"
S3_SPACING_APPLY_BTN = "Change it"
S3_SPACING_SAVE_FAILED = (
    "The voxel size could not be saved. The log says why.")
S3_LOG_SPACING_SET = "[spacing] {patch}: {spacing}"

# ── The hint for a Step 1 patch with no voxel size ─────────────────────────
#: Step 1 writes its patch TIFFs with no voxel tags (the R2b-2 audit), so a
#: patch's size comes from Step 1's registry or the user; shown by the R7
#: wiring where a patch from Step 1 has none
S3_SPACING_NO_TAGS_HINT = (
    "Step 1's patch files carry no voxel size of their own. Set it on the "
    "source in Step 1, or here with Set spacing….")

# ── QC Settings (the dialog's spacing row) ─────────────────────────────────
QC_SPACING_FALLBACK = (
    "Used for patches with no voxel size of their own; set one per patch in "
    "Session.")


def _number(v: Any) -> str:
    return f"{float(v):.2f}"


def spacing_row_text(spacing: Optional[Dict[str, Any]]) -> str:
    """Session's spacing part for a ``patch_spacing`` entry, or
    ``S3_ROW_SPACING_NONE`` (the amber "voxels")."""
    if not isinstance(spacing, dict):
        return S3_ROW_SPACING_NONE
    zyx = spacing.get("zyx")
    try:
        z, y, x = (float(v) for v in zyx)
    except (TypeError, ValueError):
        return S3_ROW_SPACING_NONE
    tpl = S3_ROW_SPACING_BY_SOURCE.get(spacing.get("source"),
                                       S3_ROW_SPACING_FILE)
    return tpl.format(z=_number(z), y=_number(y), x=_number(x))


# ── S1: one voxel size per patch ─────
#: the row's tail when the patch's features were measured at another size
S3_ROW_MEASURED_TAIL = "measured at {z} × {y} × {x} µm"
#: the Extract Features dialog's read-only line (the source in the row's words)
S3_EXTRACT_SPACING_SOURCE = {"step1": "from Step 1", "file": "from the file",
                             "user": "set by you"}
S3_EXTRACT_SPACING_ONE = (
    "This patch: {z} × {y} × {x} µm ({source}) — used for the next extraction")
S3_EXTRACT_SPACING_NONE = (
    "No voxel size of its own: the file's tags, then the fallback below")
S3_EXTRACT_SPACING_MANY = (
    "{n} patches: {k} with their own voxel size, {m} use the fallback below")


def _three(raw):
    try:
        vals = [float(v) for v in (raw or ())]
    except (TypeError, ValueError):
        return None
    return vals if len(vals) == 3 and all(v > 0 for v in vals) else None


def measured_differs(effective: Optional[Dict[str, Any]],
                     measured: Optional[Dict[str, Any]]) -> bool:
    """True when the patch's features were measured at a voxel size other
    than the one in FORCE now -- its own, else QC Settings' fallback (what
    ``_record_extraction_spacing`` stores).  A patch measured in voxels
    (no ``measured`` size) never differs: the record cannot say what it
    should have been (S1 audit H2).  This module is Qt- and
    package-free, so the comparison (to half a spin-box step, as
    ``_voxel_spacing.same_spacing``) is inline."""
    got = _three(measured.get("zyx")) if isinstance(measured, dict) else None
    if got is None:
        return False
    cur = _three(effective.get("zyx")) if isinstance(effective, dict) \
        else None
    if cur is None:
        return True
    return not all(abs(a - b) <= 0.5e-3 + 1e-12 for a, b in zip(cur, got))


def measured_tail(measured: Optional[Dict[str, Any]]) -> str:
    """``" · measured at a × b × c µm"`` for a measured size, else ``""``
    (the caller has decided it differs: ``measured_differs``)."""
    zyx = _three(measured.get("zyx")) if isinstance(measured, dict) else None
    if zyx is None:
        return ""
    z, y, x = (_number(v) for v in zyx)
    return S3_LINE_SEP + S3_ROW_MEASURED_TAIL.format(z=z, y=y, x=x)


def extract_spacing_lines(own_by_patch: Mapping[str, Any]
                          ) -> List[Tuple[str, bool]]:
    """The Extract Features dialog's line(s) about the voxel size the next
    extraction uses: ``[(text, warn)]``.  One patch: its size and where it
    came from, or the amber "no voxel size"; several: the counts; none
    given: nothing (the line is not drawn)."""
    items = list(own_by_patch.items())
    if not items:
        return []
    if len(items) == 1:
        own = items[0][1]
        zyx = own.get("zyx") if isinstance(own, dict) else None
        try:
            z, y, x = (_number(v) for v in zyx)
        except (TypeError, ValueError):
            return [(S3_EXTRACT_SPACING_NONE, True)]
        word = S3_EXTRACT_SPACING_SOURCE.get(own.get("source"),
                                             S3_EXTRACT_SPACING_SOURCE["file"])
        return [(S3_EXTRACT_SPACING_ONE.format(z=z, y=y, x=x, source=word),
                 False)]
    k = sum(1 for _n, own in items if isinstance(own, dict)
            and own.get("zyx") is not None)
    return [(S3_EXTRACT_SPACING_MANY.format(n=len(items), k=k,
                                            m=len(items) - k), k < len(items))]


def _listed_names(names) -> str:
    """Up to ``S3_SPACING_CONFIRM_MAX_NAMES`` names, then "and {k} more"."""
    names = [str(n) for n in names if n]
    cap = S3_SPACING_CONFIRM_MAX_NAMES
    listed = ", ".join(names[:cap])
    if len(names) > cap:
        listed = S3_SPACING_CONFIRM_MORE.format(names=listed,
                                                k=len(names) - cap)
    return listed


def step3_spacing_texts(names=None) -> Dict[str, Any]:
    """Step 3's words for ``ui_shell._set_spacing_dialog.SetSpacingDialog``
    (opened with ``show_z_orientation=False, show_axes=False``).  With
    *names* the several-patches line names them (R7w); the dialog itself
    fills ``{n}`` (the names' own braces are escaped)."""
    desc = S3_SPACING_DIALOG_DESC
    if names:
        listed = _listed_names(names).replace("{", "{{").replace("}", "}}")
        desc = S3_SPACING_DIALOG_DESC_NAMED.format(n="{n}", names=listed)
    return {
        "title": S3_SPACING_DIALOG_TITLE,
        "desc": desc,
        "desc_one": S3_SPACING_DIALOG_DESC_ONE,
        "z_label": S3_SPACING_Z_LABEL,
        "y_label": S3_SPACING_Y_LABEL,
        "x_label": S3_SPACING_X_LABEL,
        "cancel": S3_SPACING_CANCEL_BTN,
        "ok": S3_SPACING_OK_BTN,
    }


def spacing_confirm_text(stale: Iterable[str]) -> str:
    """The confirm naming the patches whose features would go stale; ``""``
    when none would (no confirm then)."""
    names = [str(n) for n in stale if n]
    if not names:
        return ""
    if len(names) == 1:
        return S3_SPACING_CHANGE_CONFIRM_ONE.format(name=names[0])
    cap = S3_SPACING_CONFIRM_MAX_NAMES
    listed = ", ".join(names[:cap])
    if len(names) > cap:
        listed = S3_SPACING_CONFIRM_MORE.format(names=listed,
                                                k=len(names) - cap)
    return S3_SPACING_CHANGE_CONFIRM.format(n=len(names), names=listed)


# ── R7w: the wiring's own words ────────────────────────────────────────────
#: beside the greyed button while Predict or Reconstruct runs (row 21's lock)
S3_SET_SPACING_LOCKED = (
    "Locked while Predict or Reconstruct runs; set it when the run ends.")
#: the console line of a voxel size a worker read from the patch file
S3_LOG_SPACING_FOUND = "[spacing] {patch}: {spacing} (from the file)"
#: the console line of one trigger's main-thread resolve
S3_LOG_SPACING_RESOLVED = (
    "[STEP3-SPACING] {why}: {n} patch(es), {changed} changed")

