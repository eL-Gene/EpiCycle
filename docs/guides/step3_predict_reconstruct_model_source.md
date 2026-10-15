# Model

Which Cellpose models EpiCycle uses to predict the cells in 2D on your patches, and the run they belong to.

The Model card sits at the top of the left column of **Predict + Reconstruct**. It says: "Two models predict the cells in 2D: one for top-down (XY) slices, one for side (XZ, YZ) slices." The 3D reconstruction then joins the two models' 2D predictions into 3D cells.

---

## The run row

Above the two columns, full width:

| Control | What it does |
|---|---|
| **Run** | "The run you are working in: a model applied to the patches, with its own predictions, reconstructions and curation. Switch to work on another run without disturbing the others." |
| **New Run** | "Make a new run with its own model and results." |
| **Manage** | "Rename, archive or delete the runs of this workspace." |

---

## The Model card

- **Active:** with **Change…** beside it — "Pick the two models: a refined model, a Cellpose built-in, or a model file on disk."
- Under it, one line per model, in plain grey:
  - "lat: wound-v2" — the top-down (XY, lateral) model;
  - "ax: wound-v2" — the side (XZ, YZ, axial) model.

  A third grey line says what the models were trained on, for example "Trained on 2 channels: membrane (1), nuclei (0) · Step 2 run “wound-v2”." — the channels the models were trained to read (the number in brackets is the channel's place in your image, counting from 0) and, for a Step 2 model, the run it came from.  The line wraps onto a second line when it is long, and hovering over it shows it whole.  The number in brackets is the channel's position in your image, counting from 0; a model trained on a grayscale image says so ("Trained on one grayscale channel · Step 2 run “wound-v2”.") instead of naming a membrane.  When the models were not recorded as trained on any channels — a Step 2 or refined model whose training record is missing — the line says "The channels this model was trained on are not recorded." and never guesses.  When the two models were trained on different channels, it names both sets: "Trained on different channels: lateral membrane (1), nuclei (0) · axial membrane (0). Step 2 runs “a” / “b”."  A Cellpose built-in records no channels, so with one of those active — and with no model picked, with only one of the two picked, or with a model file that is gone — the line is not shown.

  A refined model shows its own name; a Cellpose built-in shows *Pretrained:* before its name (for example *lat: Pretrained: cyto3*). A model that has not been picked yet reads "lat: none" or "ax: none". A long name is shortened in the middle, so both its beginning and its end (often the round number) stay readable; hover over the line to see the full name or path.

While either model is missing, Predict Selected stays grey and the status line under the buttons says "Pick a model first."

The model is no longer shown in **Session**: where its old block of model names used to be, Session now says "The model is chosen in Predict + Reconstruct, above its patch list."

---

## The picker (Change…)

| Choice | What it means |
|---|---|
| **Refined model** | A model you trained on your own curated patches (this plugin's **Model Refinement** section, or Step 2). |
| **Cellpose built-in** | One of Cellpose's named pretrained models (`cyto3`, `nuclei`, etc.). Good starting point if you've never trained on your tissue. |
| **Browse** | A model file somewhere on disk — a model shared by a colleague or trained elsewhere. |

The lateral and axial models can differ — for example a Cellpose built-in for one view and a trained model for the other.

---

## How to pick

**First time on a new tissue:**
- Start with `cyto3` (Cellpose's most recent general model) for both lateral and axial.
- Run a prediction, look at the result in 3D, decide if it needs refinement.

**After running Model Refinement at least once:**
- Click **Use for Re-Prediction** in the Model Refinement section — it fills in the freshly-trained models for you.
- Or open **Change…** and pick the refined model from the list.

**For published or shared models:**
- Use **Change…** → Browse, and pick the model file.
- Models trained on similar tissue (same species + organ + staining) usually transfer well.

> **Tip:** the chosen models are saved with your workspace, so the next session opens with the same models already picked.

---

## Related

- **Patches** — the patches these models run on, and the Predict / Reconstruct buttons
- **Settings** — how the models are run and how the slices are joined into cells
- **Model Refinement** — where you'd train a custom model from your curated patches
