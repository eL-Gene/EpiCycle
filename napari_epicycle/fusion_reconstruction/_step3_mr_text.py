"""Step 3 Model Refinement: every sentence of the rebuilt panel.

Qt-free.  Every sentence that reaches the screen from the rebuilt
``_create_model_refinement_panel`` is a named constant here; the four
Model Refinement guides quote them (``test_step3_mr_guide_parity.py``).
The controller's own sentences (the foot, the card, the lock line) are
Brick B2's; their constants are defined here too so the guides and the
panel read ONE source.  Nouns: the screen says "lateral" (the folder on
disk is still ``spatial``).
"""
from __future__ import annotations

from typing import Dict

# ── Patch Selection ──────────────────────────────────────────────────────
MR_PATCHES_TITLE = "Patch Selection"
MR_PATCHES_DESC = (
    "Finalized patches. Ticked patches are the ones Extract Training Pairs "
    "runs on; every extracted pair trains.")
#: the empty list's two painted lines
MR_EMPTY_LINES = (
    "No finalized patch yet.",
    "Finalize a patch in Curation to train on it. Accepted and curating "
    "patches are not listed.",
)
MR_LIST_HELP = (
    "Tick the patches to extract from; Ctrl+A ticks every patch. Click a "
    "row to highlight it.")
MR_ROW_TICK_HELP = (
    "Tick the patches Extract Training Pairs runs on. A patch whose file "
    "is gone cannot be ticked.")
MR_ALL_BTN = "All"
MR_NONE_BTN = "None"
MR_ALL_TIP = (
    "Tick every finalized patch in the list: the next Extract Training "
    "Pairs run reads all of them.")
MR_NONE_TIP = (
    "Take every tick off. Extract Training Pairs needs at least one "
    "ticked patch.")
MR_TICKED_COUNT = "{k} of {n} ticked"
MR_WILL_TRAIN = (
    "Will train on: {n} lateral + {m} axial pairs on disk, from {k} "
    "patches.")
MR_WILL_TRAIN_ONE = (
    "Will train on: {n} lateral + {m} axial pairs on disk, from 1 patch.")
MR_WILL_TRAIN_NOTHING = "Will train on: nothing yet."
#: the block under the line: lateral pairs under this many
FEW_PAIRS_LATERAL = 100
MR_FEW_PAIRS_WARN = (
    "⚠ Lateral pairs are few ({n}) — the lateral model may barely change.")
MR_NO_LATERAL_PAIRS = (
    "No lateral pairs on disk — the lateral model has nothing to train on.")
#: a row's chip: lateral pairs under this many
FEW_PAIRS_ROW = 20
#: two stacked chips; the lateral one is drawn amber under FEW_PAIRS_ROW
#: and the sentence below is the row tooltip
MR_ROW_PAIRS_LATERAL = "{n} lateral"
MR_ROW_PAIRS_AXIAL = "{m} axial"
MR_ROW_NOT_EXTRACTED = "no pairs yet"
MR_ROW_FEW_PAIRS_TIP = (
    "⚠ {n} lateral pairs — adds little to a lateral model.")
#: the two markers a row keeps (the amber line; the whole sentence is the
#: row's tooltip)
MR_ROW_STALE = "⏱ Stale — re-extract before training."
MR_ROW_STALE_TIP = (
    "⏱ Stale — labels_3d.tif on disk is newer than the per-patch "
    "extraction.  Re-extract before training.")
MR_ROW_MIXED = "⚠ Mixed settings — re-extract to match."
MR_ROW_MIXED_TIP = (
    "⚠ Mixed settings — this patch was extracted with different "
    "preprocessing than the current UI.  Re-extract to match.")
MR_ROW_TIP_PATCH = "Patch: {name}"
MR_ROW_TIP_STATUS = "Status: {status}"
MR_ROW_TIP_PAIRS = "Pairs: {n} lateral + {m} axial = {t}"

# ── Extraction ─────────────────────────────────────────────────────────────
MR_EXTRACT_BTN = "Extract Training Pairs"
MR_EXTRACT_TIP = (
    "Extract orthogonal 2D slices from curated/finalized 3D volumes\n"
    "to create image+mask training pairs for model fine-tuning.")
MR_CLEAR_BTN = "Clear…"
MR_CLEAR_TIP = (
    "Delete every patch's training pairs from disk.\n"
    "Use this to reset after major preprocessing changes.\n"
    "A confirmation dialog will ask you to confirm.")
MR_EXTRACT_NEEDS_TICK = "Extract needs a ticked finalized patch."
MR_EXTRACT_NO_PAIRS = (
    "{patch} has no pairs yet; Extract makes them. Last extraction: "
    "{time}.")
MR_EXTRACT_CANCEL_BTN = "Cancel Extraction"
MR_EXTRACT_STARTING = "Extracting training pairs…"
MR_EXTRACT_STALE_STARTING = "Re-extracting stale patches…"
MR_EXTRACT_NEVER = "never"
MR_EXTRACT_RUNNING = "Extracting slice {i}/{n}… {time}"
MR_EXTRACT_DONE = (
    "Training pairs ready {time} · {n} lateral + {m} axial = {t} pairs "
    "from {k} patches")
MR_EXTRACT_SETTINGS_TITLE = "Extraction Settings"
MR_PER_ORIENTATION_DEFAULT = "Lateral: defaults · Axial: defaults"
MR_CONFIGURE_BTN = "Configure per-orientation params…"
MR_CONFIGURE_TIP = (
    "Open the per-orientation extraction config dialog.\n"
    "Tune lateral (XY) and axial (XZ + YZ) parameters\n"
    "independently. Saved values persist per workspace.")
MR_SPLIT_TITLE = "Train / Validation Split"
MR_TRAIN_FRAC_LABEL = "Train fraction:"
MR_CHUNK_LABEL = "Chunk size:"
MR_SPATIAL_CB = "Keep neighbouring slices together"
MR_SPLIT_SUMMARY = "Train 80 % · validation 20 %."

# ── Fine-Tuning ──────────────────────────────────────────────────────────
MR_FT_TITLE = "Fine-Tuning"
MR_FT_DESC = "Which models to train and where they start from."
MR_MODE_LABEL = "Mode:"
MR_MODE_ITEMS = ("Fine-tune Lateral", "Fine-tune Axial", "Fine-tune Both")
MR_MODE_TIP = (
    "Which model(s) to train this run.\n\n"
    "  • Fine-tune Lateral: only the XY (lateral) model.\n"
    "  • Fine-tune Axial: only the XZ+YZ (axial) model.\n"
    "  • Fine-tune Both: lateral pass then axial pass, sequentially, one "
    "bundle.  Each pass has its own train/val split + loss curve.\n\n"
    "EpiCycle prediction needs BOTH a lateral and axial model — train "
    "both at some point to close the loop.")
MR_HINT_BOTH = (
    "Starts from {lat} (lateral) and {ax} (axial). Saves as {name} — one "
    "name for both models.")
MR_HINT_LATERAL = "Starts from {lat} (lateral). Saves as {name}."
MR_HINT_AXIAL = "Starts from {ax} (axial). Saves as {name}."
MR_HINT_START_HITL = "the HITL model"
MR_HINT_NAME_AUTO = "refined_NN"
MR_INIT_LABEL = "Init:"
MR_INIT_ITEMS = (
    "From HITL model", "From prior refinement", "From Cellpose pretrained",
    "Import model", "From scratch")
MR_INIT_TIP = "Weight initialization strategy for fine-tuning"
MR_SOURCE_LABEL = "Source:"
MR_AXIAL_LABEL = "Axial:"
MR_PRETRAINED_LABEL = "Pretrained:"
MR_IMPORT_LABEL = "Import:"
MR_BROWSE_BTN = "Browse…"
MR_SOURCE_TIP = "Base model to fine-tune from"
MR_SOURCE_AXIAL_TIP = "Axial pass base model (when Mode=Both)"
MR_PRETRAINED_TIP = "Cellpose pretrained model to initialize from"
MR_PRETRAINED_AXIAL_TIP = "Axial pass pretrained model (when Mode=Both)"
MR_IMPORT_PLACEHOLDER = "Browse for model file…"
MR_IMPORT_AXIAL_PLACEHOLDER = "Browse for axial model file…"
MR_MODEL_NAME_LABEL = "Model name:"
MR_MODEL_NAME_PLACEHOLDER = "refined_NN if empty"

# ── Training Configuration ─────────────────────────────────────────────────
MR_TC_TITLE = "Training Configuration"
MR_TC_DESC = "Use a preset or set the numbers yourself."
MR_PRESET_LABEL = "Preset:"
MR_PRESET_ITEMS = ("default", "quick", "thorough", "finetune",
                   "dense_tissue", "stratified_epithelia")
#: what the combo SHOWS; the keys above are what the controller reads
MR_PRESET_DISPLAY = {"dense_tissue": "Dense tissue",
                     "stratified_epithelia": "Stratified epithelia"}


def preset_key(text: str) -> str:
    """The preset key of a combo's shown text (a key passes through)."""
    shown = {v: k for k, v in MR_PRESET_DISPLAY.items()}
    return shown.get(str(text), str(text))
MR_PRESET_TIP = (
    "Pre-tuned training hyperparameter bundles.  Select one + click "
    "Apply to populate Epochs / Learning rate / Batch / etc. with "
    "sensible values.  Each preset's description appears below — read "
    "before applying.")
MR_APPLY_BTN = "Apply"
MR_APPLY_TIP = (
    "Overwrite the training fields below with the selected preset's "
    "values.  Existing values are lost — use Save Config first if you "
    "want to keep them.")
MR_EPOCHS_LABEL = "Epochs:"
MR_LR_LABEL = "Learning rate:"
MR_BATCH_LABEL = "Batch size:"
MR_ADVANCED_TITLE = "Advanced Parameters"
MR_SAVE_CONFIG_BTN = "Save Config"
MR_LOAD_CONFIG_BTN = "Load Config"

#: what each preset DOES -- the numbers it sets -- not where training
#: starts: that is Init's choice, never the preset's
PRESET_DESCRIPTIONS: Dict[str, str] = {
    "default": (
        "100 epochs, learning rate 0.01, batch 16. The numbers the "
        "panel opens with; a safe pick when no other preset fits."),
    "quick": (
        "100 epochs, learning rate 0.01, batch 32. A larger batch runs "
        "faster; use it to check whether new curated data helped before "
        "a long run."),
    "thorough": (
        "1000 epochs, learning rate 0.01, batch 16. A long run for a "
        "final model; it is slow, so use it sparingly."),
    "finetune": (
        "500 epochs, learning rate 0.001 (ten times lower). Small steps "
        "keep what the starting model already knows; pair it with an "
        "Init that starts from a trained model."),
    "dense_tissue": (
        "750 epochs, batch 8, the minimum-masks filter off. A smaller "
        "batch needs less memory on crowded slices, and every fragment "
        "stays in as training signal."),
    "stratified_epithelia": (
        "500 epochs, learning rate 0.01, channels green (segment) and "
        "red (auxiliary). Changes nothing if you have not moved the "
        "channel defaults."),
}

# ── Model Output ─────────────────────────────────────────────────────────
MR_OUT_TITLE = "Model Output"
MR_OUT_DESC = "The model this run just trained."
MR_OUT_EMPTY = "No model trained in this run yet."
MR_OUT_EMPTY_SESSION = (
    "Nothing trained in this session yet. The last model in this run: "
    "{name} ({when}).")
MR_CARD_TRAINED_ON = "Trained on"
MR_CARD_STARTED_FROM = "Started from"
MR_CARD_LATERAL_LOSS = "Lateral loss"
MR_CARD_AXIAL_LOSS = "Axial loss"
MR_CARD_TOOK = "Took"
MR_CARD_FOLDER = "Folder"
MR_CARD_FOLDER_EMPTY = "—"
MR_USE_BTN = "Use for Re-Prediction"
MR_USE_TIP = (
    "Set this model as the prediction source and return to Predict + "
    "Reconstruct")
MR_VIEW_MODELS_BTN = "View Refined Models…"
MR_VIEW_MODELS_TIP = (
    "Open a dialog listing every refined model in this workspace")
MR_NEW_RUN_HINT = (
    "To try it on the curated patches without touching them, make a New "
    "Run in Predict + Reconstruct and pick {name} there.")
MR_NEW_RUN_HINT_NONE = (
    "To try a model on the curated patches without touching them, make a "
    "New Run in Predict + Reconstruct and pick it there.")
MR_RUN_LINE = "Run: ● {name} · trained models are kept with this run"
MR_RUN_HEAD = "Run:"
MR_RUN_NAME = "● {name}"
MR_RUN_TAIL = " · trained models are kept with this run"

# ── The pinned foot ────────────────────────────────────────────────────────
MR_START_BTN = "Start Training"
MR_START_TIP = (
    "Begin Cellpose fine-tuning with the settings above.\n\n"
    "Reads the Mode (lateral / axial / both), the Init source and every "
    "training setting.  Mode Both trains the lateral model first, then "
    "the axial one — about twice the single-model runtime on one GPU.\n\n"
    "Each pass saves its weights, loss curves and config under the "
    "run's models folder.")
MR_CANCEL_BTN = "Cancel"
MR_CANCEL_TIP = (
    "Stop the training run.  The pass that was running leaves its "
    "weights folder behind.")
MR_LOSS_TITLE = "Loss curves"
#: the Loss curves header's suffix: the model names only (the controller
#: sets them while a run is drawn); blank until then
MR_LOSS_SUFFIX = ""
MR_LOSS_EMPTY = (
    "Loss curves appear here while it trains — train and validation, one "
    "pair per model.")
MR_STATUS_IDLE = "Ready · {which} · {t} pairs"
MR_STATUS_NOTHING = (
    "Nothing to train on yet — finalize a patch in Curation, then "
    "extract its pairs.")
MR_STATUS_BUSY = "Busy — finish the current operation before training."
MR_STATUS_NAME_TAKEN = (
    "The name {name} is taken in this run — change Model name.")
MR_STATUS_NO_SOURCE = (
    "No source for {bucket}: it would start from scratch — pick a Source "
    "or choose From scratch.")
MR_STATUS_DONE = "Complete: {name} · {which} (lateral {l}, axial {a}) {time}"
MR_STATUS_FAILED = (
    "Training failed in the {bucket} pass at epoch {e}: {first}. Details "
    "in Diagnostics.")


def will_train_text(n_lat: int, n_ax: int, k: int) -> str:
    """The "Will train on" line: nothing when no pair is on disk."""
    n_lat, n_ax, k = int(n_lat), int(n_ax), int(k)
    if n_lat + n_ax <= 0:
        return MR_WILL_TRAIN_NOTHING
    tpl = MR_WILL_TRAIN_ONE if k == 1 else MR_WILL_TRAIN
    return tpl.format(n=f"{n_lat:,}", m=f"{n_ax:,}", k=k)


def mode_hint(mode_text: str, lat: str, ax: str, name: str) -> str:
    """The ONE-sentence Mode hint (singular for a single model)."""
    text = str(mode_text or "")
    if "Both" in text:
        return MR_HINT_BOTH.format(lat=lat, ax=ax, name=name)
    if "Axial" in text:
        return MR_HINT_AXIAL.format(ax=ax, name=name)
    return MR_HINT_LATERAL.format(lat=lat, name=name)


# ── Sentences the controller used to carry inline (fix pass F8) ────────────
MR_NO_WORKDIR = "Set a working directory first."
MR_MODEL_SET_FROM_REGISTRY = "Model set from registry: {name}."
MR_MODEL_PATHS_UPDATED = "Model paths updated for re-prediction ({mode})."
MR_NO_REFINED_MODEL = "No refined model available — run fine-tuning first."


# ── The lock line (Brick B2) ───────────────────────────────────────────────
MR_LOCK_TRAINING = (
    "Training the {bucket} model ({i} of {n}) — Cancel stops it within ten "
    "epochs. The ticks, the pairs, every training setting, the run and the "
    "other steps are locked.")
MR_LOCK_EXTRACT = (
    "Extracting training pairs — Cancel stops it within a moment. The "
    "ticks and every setting are locked.")

# ── The foot and the result card, as the controller fills them (B2) ────────
#: the bar carries "{bucket} {epoch} / {n}", so the line keeps the loss,
#: the last sampled validation, the clock and the pass that comes next
MR_STATUS_RUNNING_NOVAL = "Training {bucket}… (loss {l}) {time}{next}"
MR_STATUS_RUNNING_VAL = (
    "Training {bucket}… (loss {l}, val {v}) {time}{next}")
MR_PROGRESS_FORMAT = "{bucket} %v / %m"
MR_PROGRESS_DONE = "Done"
MR_STATUS_NEXT = " · {next} next"
MR_STATUS_CANCELLING = "Cancelling training…"
MR_STATUS_CANCELLED = "Training cancelled."
MR_STATUS_LAUNCH_FAILED = (
    "Training could not start: {first}. Details in Diagnostics.")
MR_STATUS_EXTRACT_FAILED = (
    "Extraction failed: {first}. Details in Diagnostics.")
MR_STATUS_EXTRACT_CANCELLED = "Extraction cancelled."
MR_CARD_TRAINED_VALUE = (
    "{t} pairs ({n} lateral, {m} axial) from {k} {patches}")
MR_CARD_STARTED_BOTH = "{lat} (lateral), {ax} (axial)"
MR_CARD_INIT_SCRATCH = "scratch"
MR_CARD_LOSS_TRAIN_VAL = "train {t} · validation {v}"
MR_CARD_LOSS_TRAIN = "train {t}"
MR_STATUS_STARTING = "Training {bucket}… {time}{next}"
MR_STATUS_DONE_ONE = (
    "Complete: {name} · {which} ({bucket} {l}) {time}")
MR_DONE_WHICH_BOTH = "both models"
MR_DONE_WHICH_LATERAL = "lateral model"
MR_DONE_WHICH_AXIAL = "axial model"


def is_val_sampled_epoch(epoch: int) -> bool:
    """Cellpose computes validation at epoch 5 and every 10th epoch
    (``train.py``: ``iepoch == 5 or iepoch % 10 == 0``); any other epoch
    reports ``0.0`` (M11)."""
    e = int(epoch)
    return e == 5 or e % 10 == 0
