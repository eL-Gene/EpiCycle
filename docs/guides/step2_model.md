# Model

"What Predict in the 2D annotator uses."

This block chooses which Cellpose model draws the first-guess masks when you press Predict in the annotator. It controls **prediction only**. What *training* starts from is set in Training setup (Starting model); the two are independent, so picking a pretrained model here does not make training start from it.

---

## Use

| Choice | What Predict uses |
|---|---|
| **Auto (recommended)** | A model trained in an earlier round of this run for the view you are on, if one exists; otherwise the external model file if you set one; otherwise the Cellpose pretrained model below. The best default. |
| **Round model** | Only a model trained in an earlier round of this run. If no round has been trained yet there is nothing to use, so Predict does nothing. |
| **External model file** | A `.model` file you browse to. Its path appears in a row below, shortened, with the full path on hover. It is remembered per run and training never overwrites it. |
| **Cellpose pretrained** | A model that ships with Cellpose (`cyto3`, `nuclei` and so on), picked in the Pretrained box. |

At the start of a new run you will usually use a Cellpose pretrained model. After the first Train, Auto picks up your own model by itself.

## Active

One neutral line tells you what Predict will really use right now, per bucket, for example "Active: lat: Cell · round 3 | ax: Cell · round 2". With an external file it names the file. It is a different thing from "Writes:" in Training setup, which names the file the next Train will create.

## Pretrained

The Pretrained box is used only with "Cellpose pretrained", or for a bucket that has no round model yet. When it does not apply it is greyed with its reason on screen, for example "Not used with an external file."

## Use the newest round model after each Train

"Use the newest round model after each Train"

When this box is ticked, the model Predict uses switches to the new round's model as soon as a Train finishes, but only if the new model file is really on disk. If saving failed, EpiCycle says so in the log and keeps your previous model. Most people leave it ticked. Untick it if you want to compare the new round with the old one before adopting it.

---

## Inference parameters

A closed block, with a one-line summary of the current values (for example `Balanced · membrane / nuclei · flow 0.4 · cellprob 0.0 · diameter 30 px`). Open it to change how sensitive Predict is:

* **Cell probability threshold**: how sure Cellpose must be before it calls a pixel part of a cell. Lower finds more cells (and more noise); higher finds fewer, surer ones.
* **Flow threshold**: how smooth the cell outlines must be. Lower accepts looser outlines.
* **Diameter**: the typical cell width in pixels. Estimated for you by default; set it if you know your tissue's scale.
* **Channel to segment** and **Helper channel** (for example membrane and nuclei): which of your stains Cellpose looks at. These are shown outside the closed block so you can see them at a glance.

The presets fill in the thresholds with ready-made starting points; adjust them and press Apply to make the values active for later predictions.

**Adjust or retrain?** If predictions are mostly right but a little too eager or too shy, change the thresholds. If Cellpose is missing whole kinds of cells or inventing structures that no threshold removes, curate more slices and train a new round (see Staged slices, Train and loss curves).

---

## Everything here belongs to the active run

Switching runs swaps the Use choice, the external file and the Pretrained pick for that run's own. If you load a training config that came from a different run, that run's model path is ignored (the log says so), so you never train against another run's model by accident.
