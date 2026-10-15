# Staged slices, Train and loss curves

The Training section has a left column (what the next round will train on), a right column (how it will train, see Training setup), and a foot across both with the **Train** and **Cancel** buttons, a progress bar, one status line and the loss curves.

---

## Staged for training

"Finalized lateral slices the next Train adds, from every patch. Each thumbnail is the slice with its labels."

The list shows the slices you finalized in Curation and that no round has used yet, from every patch, for the view you are on. Each row is a picture of the finalized slice with its labels drawn on it, its name (for example `XY slice 26`) with the word finalized, the patch it belongs to, and its size, channels and number of labels. The pictures are drawn in the background; an empty tile means the picture is not ready or could not be drawn, and a hover says why.

Under the list a count tells you what the next round will really train on: how many new slices there are, which round number it is, and how many pairs in all, including the ones from earlier rounds. A Train always trains the **active view only**, so the other bucket gets its own sentence, either "Axial: nothing finalized. Switch the view in Curation to train axial." or a note that its finalized slices will not be used by this Train.

---

## Train and Cancel

* **Train** (the amber button) starts the round. It greys while a round runs.
* **Cancel** lights up only while a round runs. It asks the run to stop at the next checkpoint. Cellpose reports only every tenth epoch, so expect up to about ten epochs (longer on slow hardware) before it actually stops. Your previous round's model is restored, so a cancel never leaves you without a model.

The status line under the bar says what will happen or what is happening:

* Ready: "Ready · Train writes Cell_lateral_round_03 from 13 lateral pairs (run Cell). Cancel works while a round trains."
* Running: "Cell · lateral round 4 · epoch 412 / 1000 · loss 0.81". In a tall window the bar also reads the epoch, for example 412 / 1000; in a short one the line is cut to one row and its full text is on hover

While a round trains, an amber line is the first row of the section: "Round 3 (lateral) is training — the staged list and the setup are read-only until it finishes." The setup column greys out. The loss curves open by themselves.

## What a round does

1. EpiCycle collects the training pairs for the active bucket (lateral or axial): all earlier rounds' pairs you accepted plus the newly finalized ones.
2. Cellpose trains in the background (typically a few minutes for about fifty pairs on a GPU) while the window stays usable.
3. The loss curves update live.
4. When it finishes, the new model is saved under the run's models folder, named after the run, the bucket and the round, for example `Cell_lateral_round_03`.
5. If "Use the newest round model after each Train" is ticked in Model, Predict switches to the new model, but only when the file is really on disk.
6. Finalized slices that were actually trained on become trained. A finalized slice with no image or mask to train from is named in the log and stays finalized, so you can fix it for the next round.

Only slices you accepted are inherited from earlier rounds. Raw predictions and edits you never finished never train the model. When a round leaves slices out, the training log says how many and from which round.

Round numbers are shown as on disk: the first round of a run is `round_00`.

**Re-predicting a slice you already drew on asks first**, because a fresh prediction would replace your drawing and that cannot be undone.

## Training one view at a time

Lateral and axial are trained separately. Switch the View in Curation, finalize slices in that view, then Train. A common pattern: curate about thirty lateral slices, train lateral, switch to axial, curate about thirty axial slices of the same patches, train axial. Then both models exist for Step 3.

## If a round fails

A round can fail for one view and succeed for the other, for example when the axial run runs out of GPU memory.

* A warning names the view. You get a `[TRAIN FAILED]` line per failure in the log and a dialog naming the bucket.
* A failed view keeps its slices finalized, and its round counter does not advance. Fix the cause and train again.
* A hard failure (out of memory, a bad pair) shows a dialog with the error headline, and the full detail is in the log.

---

## Loss curves

The Loss curves block sits at the foot of the section and is **closed** while nothing is running. Its header still tells you the latest result, for example "last: lat r3 · train 0.62 · 5 rounds", and the box beside it chooses the view: "Overlay" (every round together), "Current round", or "Per round".

Open it to see the curves. The x axis is the epoch (one full pass through your curated slices) and the y axis is the loss (how far the model's masks are from yours; lower is better). The colours:

* **Lateral** rounds are amber, **axial** rounds are blue.
* Inside each colour the **newest round is the bold solid line**; older rounds are thinner and fainter (dashed, then dotted). The line of the live round is bold while a round runs.
* The view you are training is full strength; the other view's lines are faint.
* The y axis is fitted to the later part of training, so the early, very high epochs are clipped at the top.

The legend names each line by view and round, and the live one is marked. A healthy run falls and then flattens. If the loss jumps around, lower the learning rate (Training setup, Advanced); if it flattens too high, the model probably needs more curated slices. The plot starts fresh when you switch runs, since each run has its own history. No validation curve is drawn.

Drag the handle above the foot upwards to give the plot more room; the two columns above shorten and scroll.

---

## Cross-round benchmark

The Benchmarking panel is **hidden** in this version. The plan is to re-predict your accepted pairs with each earlier round's model and compare them. Until it ships, compare rounds by eye and by the loss curves.
