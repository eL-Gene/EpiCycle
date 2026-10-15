# Segment

Segment turns whole 3D stacks into 3D cells with your trained model. It works in two passes, the same two as Step 3. **Predict Selected** runs the model on every slice of a stack, in all three views (XY, XZ and YZ), and keeps the 2D outlines. **Reconstruct Selected** joins those 2D outlines into 3D cells. You choose the model once, at the top, and every ticked stack runs with it. Each stack's row then says which model made its result. So you can always tell whether a result matches the model you have now.

The three **?** buttons in this section (Model, Stacks and Settings) open this guide.

---

## The layout

    Run: [● Default (6 stacks)]  [New Run] [Manage]
    ───────────────────────────────────────────────
     Model card            │  Settings
     Stack list            │  (Start from, Inference settings, Advanced)
     (ticked stacks)       │  ───────────────────────────
                           │  [Predict Selected] [Reconstruct Selected]
                           │  [Cancel]           [View]
                           │  progress bar · status line

The left column holds the model and the stacks. The right column holds the settings. Pinned at its foot are the four buttons, the bar and the status line. The right column scrolls on its own, so opening Advanced never pushes the buttons out of sight. You can drag the divider between the two columns.

---

## The run row

At the top, across both columns, is **Run:** with one entry, for example "● Default (6 stacks)". Beside it, **New Run** and **Manage** are grey.

Step 4 has **one run** in this release, called Default. Named runs, each with its own model, come later, as Step 3 has them. A grey button shows no tooltip, so the reason is written under the two buttons: "Named runs come after the release."

The run picker's help line: "The run these results belong to. This release has one run, Default; named runs, each with its own model, come later."

---

## Model

"The model that finds the cells."

**Active:** names the model in use, and **Change…** picks another. Change…'s help line: "Pick the model to run — your Step 2 and Step 3 models, Cellpose’s built-in ones, or a file."

**Change…** opens the same model window as Step 3. It lists:

- the refined models of **every** Step 3 run in this project, each named with its run;
- the models trained in Step 2;
- Cellpose's built-in models (cyto3 and nuclei);
- a way to browse for a model file anywhere on disk.

Pick one and press **Use**. The card, every row's chip and every row's channel warning update at once.

### What the Active line tells you

- **One name in green:** the same model file serves the lateral (XY) and the axial (XZ, YZ) views.
- **One name in blue:** a Cellpose built-in model.
- **Two names in amber,** like "lat: wound-v2 · run A | ax: wound-v2 · run A". The lateral and the axial views use two different models. That is normal for a model refined in Step 3, which has one per view.
- "(no model loaded — click Change… to pick)" in grey: no model yet. Predict is grey until you pick one.
- "The active model’s files are missing. Pick it again with Change…" in amber. The model files were deleted or moved, for example with a deleted Step 3 run. Predict is grey until you pick a model whose files exist.

Under it, a dim line names the channels the model learned, for example "Trained on 2 channels: membrane (1), nuclei (0)." If the model does not say, the line reads "The channels this model was trained on are not recorded."

---

## Stacks

"Ticked stacks are the ones Predict and Reconstruct run on — the same ticks as in Session."

These are the stacks and the ticks of Session. Tick a stack here and it is ticked there too. Predict Selected and Reconstruct Selected act on the **ticked** stacks, not on the highlighted row.

### What a row shows

- at the left, the tick (centred on the row, filled with green when ticked) and a small picture (80 px) of the stack's middle slice, with its 3D cells over it once it has them;
- line 1: the status icon in its colour, and the name;
- line 2: the status word with its dots (see *Status words*), then at the right the axis order and an eye ("Show or hide this stack in the viewer.");
- line 3, dim: the size and channels, for example 212×1024×1024 · 2 ch, plus the cell count once reconstructed;
- line 4, mono: the voxel size and where it came from, or the amber word "voxels" when the stack has none;
- then a chip naming the model that made the result, or an amber warning.

### The “made by” chips

Each row says which model made its result. The chip's help line: "The model that made this stack’s result; amber when it is not the model now active, so a new Reconstruct would not match it."

- "made by wound-v2 · run A": the 3D cells were built from this model's 2D outlines.
- "2D by wound-v2 · run A": only the 2D predictions exist so far, made by this model.
- "made by cyto3 — not the active model" (amber): a different model made the 3D cells. That is not an error. But a new Reconstruct would give different cells, so do not compare the two as one.
- "2D by cyto3 — not the active model" (amber): the same, for the 2D predictions. Predict the stack again before you reconstruct it.
- "made before models were recorded" and "2D made before models were recorded": the result is older than this version of EpiCycle, which did not record the model.
- "2D made with other axes: Predict again" (amber): you changed the stack's axis order in Set spacing… after it was predicted. Reconstruct refuses it until you predict it again.

### Warnings on a row

- "Channels don’t fit this model: it needs 2, this stack has 1." The stack cannot run with the active model. Its tick and name are grey.
- "No voxel size — runs in voxels." The stack runs, but its cell width and every later number are in voxels. See *Typical cell width* and the Session guide.
- "Another stack already has this name." Only one of the two files can be used. This one cannot be ticked.
- "The stack file changed since it was predicted." Reconstruct refuses the stack until you predict it again.
- "The last Reconstruct failed; the earlier result is kept." A run failed on a stack that already had a result. The old result is still there and still used. The reason is in Diagnostics.

### The line under the list

It counts the ticks and warns you before you press a button:

- "4 ticked", or "4 ticked · 1 cannot run" when some ticked stacks cannot run with this model. Those stay ticked and are skipped. You can untick such a stack (its tick is grey but still clickable to take it off); you cannot tick it again while it cannot run.
- then, after a dot and in amber, "Wound_patch_2.tiff runs in voxels" for the first ticked stack with no voxel size. With more than one: "2 stacks run in voxels".

While a job runs, the line reads "You can still click rows and use View on finished stacks."

---

## Settings

"Start from the settings your Step 3 run used, or from a standard set."

### Start from

**Start from:** fills every setting below in one go. Its help line: "Fill the settings below from your Step 3 run or from a standard set; you can still change any of them."

- "Step 3 run “wound-v2”": the settings the **active** Step 3 run used for its latest reconstruction. Only the settings Step 3 recorded are filled; the rest stay as they are. This item appears only when that run has reconstructed something.
- "EpiCycle defaults": EpiCycle's standard values.
- "u-Segment3D defaults (without its flow filter)": the standard values of u-Segment3D, the program that joins the slices into 3D cells. u-Segment3D's own last clean-up, a filter that drops cells whose flows disagree, is not offered here, so this item leaves it off.
- "Custom": appears by itself once you change any setting by hand after a preset.

### Inference settings (open)

These decide how the model reads each slice.

- **Membrane channel** and **Nuclei channel**: which channel shows the membranes, and which the nuclei (or none). Their help lines: "The channel the model reads as the cell membrane. A loaded or previewed stack shows its layer names here." and "The channel the model reads as the nuclei, or none. A loaded or previewed stack shows its layer names here." With no stack shown, each item is the channel's number and what the model uses it for, for example 0: nuclei (model) and 1: membrane (model); a channel the model has no use for reads 2: ch2. When you load a stack in Measure & tidy, or preview one with View or the eye, the items take that stack's channel layer names from the viewer, for example 0: DAPI (nuclei): the names always come from one stack, the loaded one first. Rename a channel layer in the viewer and the item follows; the channel you picked does not change. A model trained in Step 2 or Step 3 also picks its own membrane and nuclei channels for you.
  - **If the model did not record its channels, the membrane box starts empty.** It stays empty until you pick one: Step 4 will not guess. Predict is grey meanwhile, and the status line says "Pick the membrane channel."
- **Typical cell width**, in µm: "About how wide one cell is, in micrometres; the model looks for cells near this size." 0 shows as Auto, and the model then estimates it.
  - A dim line under it gives the width in pixels for the highlighted stack, for example "≈ 24 pixels at 0.33 µm per pixel".
  - Each stack converts the width with its own voxel size when it runs. So stacks imaged at different resolutions all look for cells of the same real size.
  - A stack with no voxel size uses the pixel number shown, and the log says so. For example: "Wound_patch_2.tiff: no voxel size; cell width taken as 24 px."
- **Even out brightness**: "Stretch each slice’s brightness before the model sees it."
- **Rescale**: "Used only when no cell width is set: when ticked, the image is passed to the model at its own scale; when unticked, the model resizes it by its own trained cell size. The factor used is recorded with the result."

  When a cell width is set, the model always resizes the image from that width, and this tick has no effect.

### Advanced (closed)

Each setting shows a plain name, with the program's own name in small grey letters under it. The grey name helps if you look a setting up in the Cellpose or u-Segment3D documentation. Hover over a setting for its help line, quoted below with the default in brackets. If you are not sure, leave Advanced alone: **Start from** gives sensible values.

**The three settings at the top**

- **Batch size (slices at once)** (128): "How many slices go to the graphics card together; lower it if the card runs out of memory."
- **How 2D becomes 3D** (Direct): "The way the three views are joined into 3D cells. Only Direct is offered."
- **CPU cores** (6): "How many processor cores the 3D build uses."

Cellpose's own mask thresholds are not shown, because the 3D build does not use them.

**Joining slices into cells.** "u-Segment3D, the engine that builds the 3D cells. Each group keeps its own help lines." Seven groups follow, each closed until you open it.

**Smooth before joining** (the same settings and defaults as Step 3):

- **Smooth the cell maps** (off): "Blur each view's cell-probability map before the views are joined, so cells do not step from slice to slice."
- **Smooth the flows** (off): "Blur each view's flows before the views are joined."
- **Blur across slices** (3.0): "How far the blur reaches across the slices of each view; higher removes stripes between slices."
- **Blur within a slice** (1.0): "How far the blur reaches within a slice; keep it low to keep detail."
- **Ignore flows below** (0.20): "Flows where the cell probability is below this are set to zero. Used only with Smooth the flows."
- **Weigh flows by cell probability** (on): "Scale the smoothed flows by how likely each voxel is inside a cell. Used only with Smooth the flows."

**Follow the flows:**

- **Use several cores** (on): "Cut the volume into pieces and follow the flows on several cores at once."
- **Step fade** (0.001): "How fast each step shrinks as voxels follow the flows; higher suits long, thin cells."
- **Steps** (25): "How many steps each voxel takes along the flows."
- **Momentum** (0.98): "How much of the previous step each step keeps (0 to 1). Rarely changed."
- **Rescale the flows at each step** (on): "Set the flows back to unit length at every step."
- **Piece depth** (128): "The depth of the pieces the volume is cut into for several cores. Used only with Use several cores."
- **Piece height** (256): "The height of those pieces. Used only with Use several cores."
- **Piece width** (256): "The width of those pieces. Used only with Use several cores."
- **Piece overlap** (0.25): "How much neighbouring pieces overlap, as a fraction. Used only with Use several cores."

**Combine the probabilities:**

- **Neighbourhood size** (25): "How many neighbouring voxels are pooled when the three views' probabilities are joined into 3D."
- **Balance between views** (0.50): "A small count added before pooling so no single view dominates; higher evens the views out more."
- **Smooth the joined mask** (0.8): "Blur the joined 3D cell mask before it is cut."
- **Threshold level** (1): "Which automatic cut level decides whether a voxel is a cell; 0 is the lowest, and it must be below Number of levels minus 1."
- **Number of levels** (3): "How many bands the automatic threshold splits the probabilities into; there is one fewer level to choose from."
- **Fixed cell threshold** (Auto): "Set the cell-or-not cut by hand, from 0 to 1. Auto uses the two levels above."
- **Lowest cell threshold** (0.00): "The automatic cut never goes below this."
- **Round the threshold down to 0.1** (on): "Round the automatic cut down to one decimal place."

**Clean the mask** (these always run, as part of the 3D build):

- **Remove specks smaller than** (10000 voxels): "Pieces of the 3D mask with fewer voxels than this are removed."
- **Fill holes** (on): "Fill holes inside the 3D mask."
- **Close gaps** (1): "Close small gaps in the mask; the number is the size of the tool in voxels."
- **Grow the mask** (1): "Grow the mask outwards by this many voxels."
- **Shrink the mask** (0): "Shrink the mask by this many voxels after the steps above."

**Combine the flows:**

- **Neighbourhood size** (25): "How many neighbouring voxels are pooled when the three views' flows are joined into 3D."
- **Balance between views** (0.50): "A small count added before pooling so no single view dominates."
- **Smooth the 2D flows first** (1.5): "Blur each view's flows before they are joined."
- **Smooth the joined flows** (0.5): "Blur the 3D flows after they are joined."

**Group into cells:**

- **Smallest cluster** (5 voxels): "Clusters of fewer voxels than this are not made into cells."
- **Density cut** (0.0): "Where points count as crowded: the average density plus this many spreads."
- **Density blur** (1.0): "How much the point density is blurred before clusters are found."

**Size limits:** "Cells outside this size range are dropped by the final clean-up after the 3D build, which runs only while “Skip the size limits” is unticked."

- **Skip the size limits** (on), at the top of the group: "When ticked (the default), the final clean-up after the 3D build does not run, so the size limits below drop nothing. Untick it to apply them." While it is ticked, an amber note sits under it: "Size limits are off while this is ticked."
- **Smallest cell** (200 voxels): "Cells with fewer voxels than this are dropped."
- **Largest cell** (no limit): "Cells with more voxels than this are dropped."
- **Largest vs typical** (10.0): "With the tick below: a cell larger than the average size plus this many spreads is dropped."
- **Drop cells far from the typical size** (on): "Drop cells much larger than the typical cell, by the rule above."

**With the defaults, the size limits drop nothing.** The final clean-up is skipped unless you untick Skip the size limits. This is also Step 3's default, so both steps build cells the same way.

---

## The buttons

### Predict Selected

"Make the 2D outlines, slice by slice in all three views, for every ticked stack; the slow part, on the graphics card."

- It runs every ticked stack that can run, one after another, in list order.
- Each stack is read with its own axis order, its own channels and its own voxel size.
- A ticked stack that cannot run is skipped, and the log names it.
- The model is recorded on each stack as it finishes. Its chip then reads 2D by … .
- On a stack that already has 3D cells, it asks first: "WholeScale_patch_1.tiff already has 3D cells. Predicting again replaces its 2D outlines; the 3D cells stay until you Reconstruct." The stack keeps its status. Its chips then show both the old 3D model and the new 2D one.

### Reconstruct Selected

"Join each ticked stack’s 2D outlines into 3D cells; needs Predict first, and uses the 2D outlines already made."

- It runs every ticked stack whose 2D predictions are complete and up to date. It never runs the model again.
- A stack without them is skipped, and the log says why. For example:
  - "Wound_patch_2.tiff: the 2D predictions are incomplete. Predict it again first."
  - "Wound_patch_2.tiff: the 2D predictions were made with other axes. Predict it again first."
  - "Wound_patch_2.tiff: the stack file changed since it was predicted. Predict it again first."
- The “made by” chip is copied from the 2D predictions, since the 3D cells are built from them.
- **Reconstructing a stack that already has 3D cells deletes its old result.** The old cells go, with their measurements, marks, tidied outlines, saved maps and exported 3D meshes folder. Nothing is kept or renamed, the same as in Step 3.
- If the stack was measured or finalized, it asks first: "WholeScale_patch_1.tiff is finalized. Reconstructing it again deletes its current cells, measurements, marks and exported 3D meshes, and starts from new cells."

### Cancel

"Stop after the stack being worked on; finished stacks are kept."

Cancel is honest about when it stops. The stack being worked on runs to its end, then the job stops. The status line says "Cancelling... will stop after current stack completes".

### View

"Show the highlighted stack’s result in the viewer (the highlighted row, not the ticks); works while a run is going."

View shows the result of the row you clicked, whatever is ticked. It stays live during a run, so you can look at finished stacks while others work. If the stack cannot be shown: "Wound_patch_2.tiff could not be shown. The log in Diagnostics says why."

### The status line

When nothing runs, it reads "Ready · 4 stacks ticked", counting only the ticked stacks that will run (with one stack, it says 1 stack, not 1 stacks). If Reconstruct is grey, the reason follows on the same line after a dot: for example, Ready · 4 stacks ticked · Predict the ticked stacks first. If View is grey, the highlighted row's own status word already says it has no result. With nothing ticked it says "Tick a stack to run." If the model is missing, the model card says why and the status line adds nothing.

During a run it uses Step 3's sentences, with “stack” in place of “patch”. A long sentence wraps onto a second line; if it is longer still, the second line is cut short and the whole sentence is in the tooltip:

- "Segmenting 3 stacks (1,120 slices)..." when Predict starts;
- "Segmenting stack 2/3: Wound_patch_2.tiff — XZ (412/768)  [6:02 | ~5:40 remaining]" per view. The running row adds · XZ 412 / 768 to its third line.
- "Reconstructing 3 stacks..." and "Reconstructing 2/3: Wound_patch_2.tiff...  [2:31]" during Reconstruct;
- "Postprocessing Wound_patch_2.tiff...  [0:42]" after the 3D build, only when the final clean-up runs;
- at the end, "Segmentation complete: 3/3 stacks  [11:44]" or "Reconstruction complete: 2/3 stacks (1 failed)  [7:38]". The (1 failed) part appears only when a stack failed, so the count is always true.

The running row reads "segmenting..." or "reconstructing..." in grey, with an hourglass. A finished stack changes its word at once.

---

## Status words

| Word | Colour | Meaning |
|---|---|---|
| **unprocessed** | grey ○ | Nothing has been run yet. |
| **predicted** | teal ● | 2D predictions exist; ready for Reconstruct. |
| **reconstructed** | amber ●● | 3D cells exist; ready for Measure & tidy. |
| "segmenting..." / "reconstructing..." | grey, hourglass | Being worked on now. |
| **measured** | purple ●● | Measured in Measure & tidy. |
| **finalized** | light cyan ●●● | Saved and marked done; Export takes it. |
| **failed** | red, crossed circle | The last run failed and there is no earlier result. |

---

## When things are grey

A grey control shows no tooltip, so every reason is written on the page:

| Control | Grey when | Where the reason is |
|---|---|---|
| **Predict Selected** | no model is chosen | the model card: "(no model loaded — click Change… to pick)" |
| **Predict Selected** | the model's files are missing | the model card, amber: "The active model’s files are missing. Pick it again with Change…" |
| **Predict Selected** | no model, or its files are missing | the model card (the status line adds nothing) |
| **Predict Selected** | nothing is ticked | the status line: "Tick a stack to run." |
| **Predict Selected** | the membrane channel is empty | the status line: "Pick the membrane channel." |
| **Predict Selected** | no ticked stack can run with this model | the status line: "No ticked stack can run with this model." |
| **Reconstruct Selected** | no ticked stack has complete, current 2D predictions | the status line, after the idle words: "Predict the ticked stacks first." |
| **View** | the highlighted stack has no result | the highlighted row's own status word (unprocessed or predicted) |
| **Cancel** | nothing runs | (nothing to cancel) |
| **New Run**, **Manage** | always, in this release | under them: "Named runs come after the release." |
| a row's tick | the stack cannot run | the row's amber line |

**While a job runs,** an amber line at the top of the section names it. The same line shows in Session and in Measure & tidy. For example: "Reconstructing — Cancel stops it after the current stack. The workspace, the ticks, the model, the settings, Measure & tidy and the other steps are locked."

- The run row, Change…, the ticks, Start from, every setting and both run buttons are grey. The **?** buttons and the Advanced disclosures stay live, so you can still read the settings and their guide.
- **Cancel and View stay live** during a Predict or a Reconstruct. You can still click rows to highlight them and use the eye.
- **During any other Step 4 job** (loading a stack, Measure, Smooth, Save, an export), Predict, Reconstruct and View are grey too, the eyes are dimmed, and the status line is blank: the amber line is the reason.
- The other steps of EpiCycle are locked too, so a step change cannot lose a Predict's results.

---

## Diagnostics

Diagnostics, the last section of the panel, has no **?** of its own.

- **System** shows the graphics card Step 4 found.
- **Log Output**, "Segmentation run events and output status.", keeps the history of each run.

The log holds the details the panel keeps short. An on-screen sentence ending “The log in Diagnostics says why.” always has its reason here. Typical lines:

- "Predicting 3 selected stacks · model lat: wound-v2 · run A | ax: wound-v2 · run A". The model part is the same text as the Active line.
- "WholeScale_patch_2.tiff: 2D predictions, 212 slices × 3 views (38 s)"
- "Wound_patch_2.tiff: no voxel size; cell width taken as 24 px."
- "Reconstructing 3 selected stacks"
- "  Done: 4,410 cells, saved to labels_3d.tif"
- "[ERROR] Prediction failed for Wound_patch_2.tiff: CUDA out of memory", in amber.
- "[ERROR] Reconstruction failed for Wound_patch_2.tiff: out of memory in gradient descent", in amber.

---

## What can go wrong

- **A stack has no voxel size.** It runs, with the cell width taken in pixels (the log says how many). Everything measured later is in voxels. For µm, use **Set spacing…** in Session first.
- **A stack's channels do not fit the model.** Its tick is grey and it is skipped. Choose a model with the right number of channels, or leave the stack out.
- **A stack failed.** With no earlier result, its row turns red **failed**, and the done line counts it, as in (1 failed). With an earlier result, that result is kept, and the row says "The last Reconstruct failed; the earlier result is kept." The reason is in Diagnostics.
- **Out of memory in Predict.** The graphics card ran out of memory (“CUDA out of memory” in the log). Lower **Batch size (slices at once)** and predict again.
- **Out of memory in Reconstruct.** The computer's own memory ran out during the 3D build. Close other programs, or lower **CPU cores**, and reconstruct again. A very large stack may need to be cut smaller in Step 1.
- **A stack was edited after it was finalized.** Segment does not change its word. But the Export window refuses it until it is measured and finalized again. See the Measure & tidy guide.
- **Reconstruct again deletes.** Reconstructing a stack that has 3D cells deletes the old cells, measurements, marks, tidied outlines and exported 3D meshes. A confirmation asks first when the stack was measured or finalized. There is no undo.
- **Measure again keeps your marks.** Measuring again in Measure & tidy carries your marks over to the same cells. Only a new Reconstruct loses them.
- **A chip is amber.** A different model made the result. Predict and Reconstruct the stack again if it should match the active model.
