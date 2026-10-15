# Training setup

"What the next Train writes, where it starts and how long it runs."

The right-hand column of the Training section. Nothing here starts training: it only sets up the round that the **Train** button at the foot will run. The **History** button beside the header opens the record of past rounds.

---

## Writes

"Writes:" tells you the real name of the model file the next Train will create for the view you are on, together with the round number, for example `Cell_lateral_round_03`. Rounds are numbered as on disk, so a run's very first round is `round_00`. The other bucket is shown in grey as "axial: not in this round", because a Train trains one view at a time.

## Mode

Both modes train on the **same data**: every accepted pair in this run, from earlier rounds and the new one. The only difference is which weights each round **starts from**.

| Mode | What it means |
|---|---|
| "Continue from the last round (cumulative)" | Each round fine-tunes the previous round's model, so it keeps what it learned and settles faster. The recommended choice. |
| "Start again from the starting model (from scratch)" | Each round starts again from the Starting model below instead of the previous round. Useful for a clean retraining if you think the previous model drifted badly. |

Starting again does **not** mean using fewer pairs: it still trains on all of them. A single line under the box says exactly what will happen, for example "Starts from Cell_lateral_round_02 — round 2’s lateral model." in the first mode, or "Starts from cyto3 again, every round. Earlier rounds’ pairs are still used." in the second (shown in amber, because it overrides the usual chaining).

Mode is kept **per view**: lateral and axial each remember their own choice, and if one Train covers both, each starts the way you set it.

## Starting model

"Starting model" is the Cellpose model a round starts from when it is not continuing from the last round: the first round of a run, or any round in the start-again mode. The small grey word under the name is the engine's own name for the setting. It is used by training only, and is separate from the Use box in Model, which only decides what Predict uses.

* **scratch** starts from random weights (slowest to settle).
* **cyto3**, **nuclei** and the other names fine-tune that Cellpose model.
* **Start from a file (optional)…** with **Browse** lets you start from a model file of your own instead, for example another run's model. A file overrides the Starting model name; it is not the same as scratch.

When the Mode makes the name irrelevant (a normal continued round) the box is greyed and the line under Mode says why.

## Epochs, Batch size, Device

* **Epochs** is how many full passes the model makes through your curated slices. More lets it learn finer detail but, on a small set, eventually makes it memorise your slices instead of generalising. The default of 500 is a sensible start; watch the loss curves and cancel if they have flattened.
* **Batch size** is how many slices the GPU handles at once. Bigger is faster but needs more memory: if you hit an out-of-memory error, lower it.
* **Device:** GPU or CPU. CPU works but is far slower.

## Advanced

A closed block with a grey one-line summary of its values. Each setting has a plain name over the engine's own name:

* Learning rate: how boldly the model changes after each batch. The default is usually right; lower it if the loss jumps around.
* Weight decay: a mild brake that stops the model fitting your slices too tightly. The default is usually right.
* "Fewest objects per image": "An image with fewer objects than this is left out of training."
* Channel to segment and "Helper channel": which of your stains the model looks at (for example membrane and nuclei).
* Normalize and Rescale: how the image brightness and cell size are adjusted before training.
* "Recompute flows": "Recompute flows: work the flow targets out during training rather than once before it."
* **Save config…** writes the current settings to a `training_config.json` file (the picker opens in the active run's folder); **Load config…** applies a saved one. Step 2 does not write a config file by itself; your real settings live with the run and come back when you open the workspace or switch runs.

## Use the newest round model after each Train

This option lives in the Model block of Curation; see the Model guide.

## While a round is training

"Round 3 (lateral) is training — the staged list and the setup are read-only until it finishes." The column greys out. If you used start-again mode the Starting model box lights up, since it matters in that mode.

## What gets written on Train

* The new model, named after the run, the bucket and the round.
* The train loss curve saved next to the weights.
* A `model_manifest.json` recording how the round was made (settings, where it started from).
