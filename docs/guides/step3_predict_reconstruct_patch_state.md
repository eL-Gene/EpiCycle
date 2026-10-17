# Patches

The list of patches in your workspace, with the buttons that run on them. The Patches card sits in the left column of **Predict + Reconstruct**, under the Model card; it says "Ticked patches are the ones Predict and Reconstruct run on."

Each row shows a tick box (centred on the row; a ticked box fills with amber), a small picture of the patch (80 px, with the run's cells over it once the patch is reconstructed), a status icon, the patch name, its status (how far it has progressed through the pipeline) with its axis order on the right (see [Axis order](#axis-order-per-patch) below), the patch's size, how many channels it has and, once it has been reconstructed, how many labels it has, for example "124×240×240 · 2 ch · 4,653 labels", and under that the patch's voxel size ("0.52 × 0.52 × 0.52 µm (from Step 1)", or the amber word "voxels" when it has none; set it in Session).  Hover over "voxels" and the row says why it matters: "No voxel size — cell width falls back to pixels." — a cell width given in µm is counted in pixels for that patch until it has a voxel size.

Step 3 finds your patches on its own: with no folder picked, it uses the ones
Step 1 extracted into this workspace, and says so in a faint line under the
Patch / Stack Directory box in **Session**.  The list refreshes every time you open
Step 3, so a patch you extracted, renamed or trimmed in Step 1 a moment ago is
already here — you do not have to press Browse to make it appear.

If every patch is deleted from that folder, Step 3 has nowhere to read from:
the buttons that need a patch folder say *Set a stack directory first*, and you
press **Browse** to point at one.  A folder you pick with Browse is remembered;
the automatic one is worked out fresh each time, so a project copied to another
computer still finds its patches.  Deleting a patch file is the one exception: its row stays in the list, marked missing — see [A patch whose file is gone](#a-patch-whose-file-is-gone) below.

---

## How patches flow

Each patch moves through these stages, in order:

```
unprocessed → predicted → reconstructed → curating → accepted → finalized
```

You can see the current stage of every patch on its row.

---

## What each status means

| Status | Meaning |
|---|---|
| **unprocessed** | Patch exists in the workspace but nothing has been run on it yet. |
| **predicted** | 2D Cellpose predictions exist for this patch. Ready for reconstruction. |
| **reconstructed** | 3D mask exists. Ready for human curation. |
| **curating** | Curation in progress — you've opened the 3D annotator on this patch but haven't yet hit Accept Volume. |
| **accepted** | You clicked Accept Volume in the Curation section — the curated 3D mask is locked. |
| **finalized** | You clicked Finalize Labels — this patch is now training-data-ready for Model Refinement. |

---

## Working with patches

### Choosing the patches to run
**The ticked patches are the ones Predict and Reconstruct run on.**  Tick a row's box to add it; clicking the row itself only highlights it (one row at a time), which is what **View** shows.  Hovering over the list says: "Tick the patches to run on; Ctrl+A ticks every patch that can be ticked. Click a row to highlight it for View."

The **Tick all unprocessed** link under the list: "Ticks every patch that has not been predicted yet."

Your ticks are saved with the run: they are still there after a run finishes, after you leave Step 3 and come back, and after you switch to another run and back (each run keeps its own).

Accepted and finalized patches are **protected**: their row says "protected", and hovering over it says why, for example "finalized — protected; Reset to Predicted in Curation to redo it".  A protected patch cannot be ticked; if it was ticked before it was protected, you can take the tick off.  A protected row can still be highlighted for View, and it shows the same tier counts as Session's row: a coloured dot and a number per tier (curated, good, review, bad, garbage), what the patch's saved features say; hover over a dot to read its tier.

A ticked patch that is being curated carries the chip "has curation": running it again replaces that curation.

### Which model made a patch
Once a patch has been predicted, its row carries a chip under the size line naming the model that predicted it, for example "made by cyto3" — the top-down (XY) model's name, shortened when it is long.  In a narrow list the chip stays on one line and is cut at the end with "…", so the start of the name stays readable; when it is cut, the amber "— not the active model" is left off (the amber border says it); hover over the chip to read all of it.  Hovering over it says: "The model that predicted this patch. An amber border means it is not the model now active above."  When the patch was made by another model than the one now chosen in the Model card, the chip has an amber border and reads, for example, "made by nuclei — not the active model": predicting it again would give a different result.  A patch predicted before this record existed (October 2026) reads "made by: unknown (predicted before 2026-10)".  Once the patch is reconstructed, the chip reads the record kept with its 3D labels, so predicting it again later does not change what its labels say made them.

The **ⓘ** at the right of the row (where the axis order is) opens a small window: "Opens the details of the model that predicted this patch, with notes you can keep."  Its title names the patch, for example "The model that made b.tif", and it lists everything recorded about the two models — "Lateral model (XY)" and "Axial model (XZ, YZ)": what kind of model each is (for example "Step 2 round model" or "Cellpose built-in model"), its file, its run and round, when it was trained and on what — and then a **Notes** box.  Type anything worth keeping with the patch's model and press "Save notes"; the window says "Notes saved." and the notes are written into the patch's prediction record and into the copy kept with its 3D labels.  A patch predicted before the record existed says "No model was recorded for this patch: it was predicted before 2026-10.  Notes saved here are kept with it."  While Predict or Reconstruct runs, the window can be read but not saved: "Notes cannot be saved while Predict or Reconstruct runs; they can be read."  A second click on the same ⓘ brings its window to the front; the windows close when you change the patch folder, switch runs or workspaces, or leave Step 3.

Under the list, two lines count what the buttons will run on, for example "Predict: 3 ticked" (followed by "· 1 protected" when protected patches are ticked) and "Reconstruct: 2 ready of 3 ticked" — a patch is ready to reconstruct once it has been predicted.

### Running patches that already have results
If some ticked patches already have predictions, **Predict Selected** asks first and names each one with its status, for example "b.tif — predicted"; predicting again replaces their predictions.  **Reconstruct Selected** lists, per patch, what a new reconstruction replaces.

### The buttons
The four buttons stay at the bottom of the right column, under the settings, however far you scroll the settings:

| Button | What it does |
|---|---|
| **Predict Selected** | "Predicts the cells in 2D on the ticked patches, one slice at a time in all three views. Accepted and finalized patches are skipped." |
| **Reconstruct Selected** | "Joins the 2D predictions of the ticked patches into 3D cells. A patch must be predicted first." |
| **Cancel** | "Stops a Predict after the slice it is working on, and a Reconstruct after the patch it is working on." |
| **View** | "Shows the highlighted patch’s 3D cells over its image in the viewer." |

### Why a button is grey
The line under the progress bar says why, in this order:

- "Pick a model first." — one of the two models is missing (see **Model**).
- "Tick a patch to run." — nothing is ticked.  (Earlier versions predicted every unprocessed patch when nothing was selected; Predict now runs only what you tick, and the **Tick all unprocessed** link is one click away.)
- "Every ticked patch is protected." — only accepted or finalized patches are chosen.
- When Predict can run, the line reads "Ready · 2 ticked", followed by "Predict the ticked patches first." while none of them is predicted yet, so Reconstruct stays grey.  If protected patches are still ticked, it ends, after a dot, by saying what happens to them: "1 protected (finalized) is skipped" (several read, for example, "3 protected (accepted, finalized) are skipped").  The number in "Ready · 2 ticked" is every ticked patch, as in the line under the list.  If the line is too long for the two lines under the buttons, the statuses are left out ("2 protected are skipped"), and if that still does not fit the clause moves before "Predict the ticked patches first."

While a run is going, the same line follows it (which patch, which view, the time left) and keeps its last sentence when the run ends, until you change your ticks.  A run where some patches failed ends on an amber line; one where none succeeded, on a red one.  A cancelled Predict ends on an amber line, for example "Segmentation cancelled: 1/3 patches done", and the patch it was working on stays unprocessed.

### While something is running
An amber line at the top of **Predict + Reconstruct** says what is running.  While Predict or Reconstruct runs, for example "Predicting — Cancel stops it after the current slice. The run, the ticks, the model and the settings are locked.", the run row, **Change…**, the settings and the ticks are grey; **Cancel** and **View** stay live, and under the list: "You can still click rows and use View on finished patches."  While something else is open — for example "A refine step is staged in the annotator — Accept or Discard it there." — the line tells you where to finish it, and nothing here is greyed except the buttons that would start a run, and **View**.  A short pause (a patch loading) shows its line only if it lasts more than a moment.

### Going back a stage
If a patch is at `reconstructed` and you re-run prediction, it moves back to `predicted` and the old reconstruction is invalidated. EpiCycle warns before overwriting curated work — finalized patches are protected.

---

## A patch whose file is gone

If a patch's file is deleted (in Step 1, or outside EpiCycle), its row stays where it was, dimmed, with the amber line "The patch file is missing." — in this list and in the **Curation** list, where clicking it is refused with the same sentence, because the annotator needs the image.  The row cannot be ticked.  Step 3 checks for missing files when it reads the patch folder, when you switch runs and when you come back to Step 3 (and again when Predict or Reconstruct starts).  If the file comes back, the line goes at the next check.

The row's menu (right-click it, or click the **⋯** on the row) offers "Forget this patch…".  It first asks you, naming the patch, its status in this run and what will move — the 3D labels and their manifest, the measured features, the curation, the prediction cache and the refinement pairs, as far as they are on disk.  Then it **moves** this run's files for the patch into a trash folder, `artifacts/step3/.trash/<patch>-<date-time>/`, keeping their folder layout, with a `forgotten.json` that lists every file and where it came from — nothing is deleted, and everything can be moved back.  The snapshots, the training archive and the other runs' files are kept.  The patch then leaves this list and Session's, and its small pictures are removed.

If one of the files is held open by another program (an image viewer, a virus scan, a synced folder), the move stops at that file: the files already moved stay in the trash folder, `forgotten.json` says which ones moved and which did not, the rest stay where they were, and the patch keeps its row and its record.  Close the other program and choose "Forget this patch…" again; this makes a second trash folder with its own `forgotten.json`, so the patch's files may then be split across two trash folders.

Forget is refused while anything is running ("Wait for the running work to finish first."), while the patch's file is there, and while the patch is open in Curation with unsaved edits ("Save or discard this patch’s curation first.").

---

## Axis order (per patch)

Each patch row also shows its detected **axis order** — the layout of the source image file:

| Shown | Meaning |
|---|---|
| **CZYX** | 4D stack, channels first (channel, depth, height, width). EpiCycle's own extracted stacks use this layout. |
| **ZCYX** | 4D stack, depth first (depth, channel, height, width). Common from some microscopes / exports. |
| **ZYX** | 3D single-channel stack (depth, height, width). |

EpiCycle figures this out **automatically, for each patch individually**, by reading the file's header (no need to load the whole image). It uses, in order of trust: the file's own embedded axis metadata, then a comparison against the matching label shape, then a size-based best guess.

Because detection is per patch, a folder that mixes layouts (some CZYX, some ZCYX) is handled correctly — each patch is predicted and has training pairs extracted using its own real layout. You don't have to set anything.

> **Note:** Earlier versions had a single global *Axes* dropdown among the inference settings that applied one layout to every patch. It was removed — it was misleading for mixed-layout folders, and the per-patch detection replaced it. If a patch's shown axis looks wrong, check that the source TIFF is well-formed; the displayed value is what prediction and extraction will use as their starting point (they re-confirm it per file at run time).

---

## Downsample badges

If your workspace is in downsample mode, patches that were curated at downsampled resolution show a small *DS* indicator. The on-disk TIFF is always full-res (the save step automatically upsamples back), so downstream consumers (Step 4, Model Refinement) see the canonical shape regardless.

---

## Reading the colors

Status colours follow the project's convention:

- **gray** = unprocessed
- **blue** = predicted
- **teal** = reconstructed
- **amber** = curating (in progress)
- **light green** = accepted
- **dark green** = finalized

---

## Common workflows

**First-time setup on a new dataset:**
1. Patches start as `unprocessed`
2. Click **Tick all unprocessed** (or press Ctrl+A in the list) → **Predict Selected**
3. After predictions complete, the ticks are still there → **Reconstruct Selected**
4. Patches are now `reconstructed`; switch to the **Curation** section

**Iterative refinement loop:**
1. Curate + finalize a few patches
2. Train a refined model (Model Refinement section)
3. Use the refined model to re-predict on remaining `reconstructed` patches
4. Curate those, finalize, retrain — repeat

---

## Troubleshooting

### A patch's status seems stuck
If prediction or reconstruction crashed, the patch may be left at the previous stage. Re-running on the affected patch usually resolves it. Check the Diagnostics section for error messages.

### A patch isn't appearing
Make sure the Patch Directory in the Session section points at your patch root. If the patches are nested in subfolders, EpiCycle should still discover them; if not, check that each patch has the expected TIFF naming.

---

## Related

- **Model** — the models that will be used when you click Predict Selected
- **Settings** — how they are run, in the right column above the buttons
- **Curation** — what to do once a patch is `reconstructed`
- **Model Refinement** — only operates on `finalized` patches
