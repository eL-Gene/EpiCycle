# Fine-Tuning

The “go” button — run training on the extracted pairs with the configured settings.

"Which models to train and where they start from."

---

## Choosing what to train

- **Mode** picks the model(s): **Fine-tune Lateral**, **Fine-tune Axial** or **Fine-tune Both** (lateral first, then axial, saved together in ONE bundle).
- A one-sentence hint under Mode says what will happen: "Starts from the HITL model (lateral) and cyto3 (axial). Saves as refined_03 — one name for both models." With one model it reads "Starts from the HITL model (lateral). Saves as refined_03."
- **Init** picks where training starts: from the HITL model, from a prior refinement, from a Cellpose pretrained model, from a model file you import, or from scratch. Source / Pretrained / Import rows appear to match; with Mode = Both each has an **Axial** twin row below it, so the two models can start from different places.
- **Model name** is optional. Left empty, the bundle is named `refined_NN`: the next free two-digit number in this run's models folder (`refined_01`, `refined_02`, …). Type a name of your own to keep a domain model apart (for example `wound_membrane`). A name already taken in this run is refused.

---

## Before you click

You need both of these in place:

1. **Training pairs extracted** for at least one view (**Patch Selection** and **Extraction Settings**). The line under each row and the Will-train-on line under the list tell you what is ready.
2. **Channel mapping** configured (**Training Configuration** → Advanced Parameters). Default `[2, 1]` works for green=membrane / red=nuclei; adjust if your stains differ.

If nothing is extracted yet, **Start Training** does not run; the line under the bar says why:

"Nothing to train on yet — finalize a patch in Curation, then extract its pairs."

When it can run, that line reads like this: "Ready · lateral + axial · 4,480 pairs". Other refusals are written there too, in amber:

"Busy — finish the current operation before training."

"The name refined_03 is taken in this run — change Model name."

"No source for lateral: it would start from scratch — pick a Source or choose From scratch."

---

## What happens when you click Start Training

The progress bar starts ticking through epochs, the fields lock (an amber line at the top of the section says what is locked), and **Cancel** next to the button turns live. The button keeps the name **Start Training**; it never renames itself.

Internally:
1. All `*_img.tif` and `*_mask.tif` pairs for the selected mode are loaded into memory
2. Train / val sets are split deterministically by stem hash (so re-runs with the same data give the same split)
3. A Cellpose trainer is instantiated with the channels + init-mode + hyperparameter settings from **Training Configuration**, all read once at the click (editing a field during the run changes nothing)
4. Training runs for the configured number of epochs
5. The trained model gets saved in its bundle folder, `models/{model_name}/`, under the `refinement` folder of the current run (`{workspace}/artifacts/step3/`)

If **Mode = Both**, lateral trains first; when it finishes, axial automatically queues and trains next. The bar names the pass and the epoch (`lateral 20 / 100`), and the line under it names the pass that runs:

"Training lateral… (loss 0.52, val 0.58) [0:41 | 5:02] · axial next"

---

## Watching the loss plot

**Loss curves** (below the status line, click its title to open) plots the live run: the **train** loss as a solid line and the **validation** loss as a dashed line of the same colour. Cellpose measures validation loss only at epoch 5 and every 10th epoch after that, so the dashed line is a few points joined, not a line per epoch; with no validation split it is not drawn.

| What you see | What it means |
|---|---|
| **Both lines drop steadily** | Model is learning — let it finish |
| **Both lines flatten early** | Model has learned what it can from this data — can stop or accept |
| **Train loss drops but val loss climbs** | Starting to overfit — consider stopping, or reducing epochs next round |
| **Loss diverges or jumps wildly** | Numerical instability — cancel, lower learning rate, retry |
| **Loss stays flat from epoch 1** | Model isn't learning at all — usually wrong channel mapping |

Before a run, the plot says:

"Loss curves appear here while it trains — train and validation, one pair per model."

---

## Cancelling

Click **Cancel** next to Start Training any time; it stops the run within about ten epochs. A cancel is not a rollback: the pass that was running leaves its weights folder behind in the bundle, so delete that folder if you do not want it. Nothing is registered as a finished model.

---

## When training finishes

The line under the bar reports it: "Complete: refined_03 · lateral + axial (lateral 0.41, axial 0.38) [24:10]". If a pass failed, the line says which one and where, in red: "Training failed in the lateral pass at epoch 412: CUDA out of memory. Details in Diagnostics."

**Model Output** shows the result card — "The model this run just trained." — with what it was trained on, where it started from, the lateral and axial loss, how long it took and its folder. Before anything is trained it says "No model trained in this run yet." Two actions sit in the card:

### Use for Re-Prediction
Wires the freshly-trained model directly into Step 3's **Predict + Reconstruct** section — sets it as the lateral and axial model for the next prediction run (both, after a Both run).

After clicking this, go to **Predict + Reconstruct** and click **Predict** to re-segment with the refined model. That replaces the predictions of the patches you re-run.

### View Refined Models...
Opens the list of every refined model in this workspace, to pick one for re-prediction or as the starting point of the next round.

To try a model without touching the curated patches, make a New Run in **Predict + Reconstruct** and pick the model there; the sentence under the buttons says so: "To try it on the curated patches without touching them, make a New Run in Predict + Reconstruct and pick refined_03 there."

---

## The iteration loop

This is a feedback cycle. Each round typically improves prediction quality on similar tissue:

```
finalize curated patches
       ↓
extract training pairs (Extraction Settings)
       ↓
fine-tune model (this section)
       ↓
use refined model in Predict
       ↓
predict on more patches → curate → finalize → loop again
```

Stop iterating when predictions look “good enough out of the box” — usually 2–4 rounds for a new tissue type.

---

## Troubleshooting

### Training crashes with `CUDA out of memory`
Lower **Batch size** in Training Configuration (try 2, then 1 if needed). If still hitting it, the patches are unusually large — consider re-extracting with a higher **Min distance between slices** to thin the training set.

### Trained model is worse than the starting model
- **Too few training cells** — add more finalized patches
- **Too many epochs** — model overfit; lower **Epochs** and retrain
- **Wrong channel mapping** — double-check the channel combos match your stain layout
- **Curation issues** — if your finalized masks have systematic errors (e.g. consistent under-segmentation), the model learned them. Re-curate and retrain.

### Loss plot stays flat from epoch 1
The model isn't learning. Most common cause: channel indices don't match your data (e.g. you picked channel 2 but the image only has 2 channels). Cancel, fix channels in **Training Configuration**, restart.

### Trained model expects different image shape than what I'm running on
Shouldn't happen — refinement always trains on full-resolution data (even if you curated on downsampled patches, the save step lifts to full-res before training). If you see shape mismatch errors, something's off in the workspace state; check that the patch's `labels_path` matches the source image's original resolution.

---

## Related

- **Extraction Settings** — produced the pairs you're training on
- **Training Configuration** — the knobs that govern how training runs
- **Patch Selection** — the upstream finalized patches that fed everything
