# Settings

The settings Predict and Reconstruct use for every ticked patch: how the model reads each slice, and how the three views of 2D outlines are joined into 3D cells.

You can leave all of them as they are. **Start from** fills them in one go, and every setting has a help line: hover over it to read it. The help lines are quoted below, with the default in brackets.

Quoted sentences (in straight double quotes) are exactly what the page says.

---

## The description

Under the **Settings** title, one line says where the settings can come from:

- "Start from this run’s last reconstruction, or from a standard set." while this run has reconstructed at least one patch;
- "Start from a standard set, then change what you need; any change makes it Custom." while it has not.

---

## Start from

**Start from:** fills the settings below. Its help line: "Fill the settings below in one go: from EpiCycle’s defaults, from u-Segment3D’s own defaults, or from the settings this run last reconstructed with. You can still change any of them; the list then says Custom."

The list, in this order:

- "EpiCycle defaults": EpiCycle's standard values. This also resets the cell width, Even out brightness, Rescale, the batch size and the CPU cores. For Cellpose's built-in nuclei model it also sets **Remove specks smaller than** to 100 voxels: that clean-up acts on the joined 3D mask before the cells are cut, separate nuclei are each a small piece of it, and the cell value (10,000) would delete most of them. Any other model keeps 10,000. When you pick another model while the list says EpiCycle defaults, **Remove specks smaller than** follows the new model, and the list stays EpiCycle defaults. Nothing else changes, with one exception: in a run that has no channels saved yet and has not predicted, the two channel boxes follow the new model too (see **What is remembered** below).
- "u-Segment3D defaults (without its flow filter)": the standard values of u-Segment3D, the program that joins the slices into 3D cells. u-Segment3D's own last clean-up, a filter that drops cells whose flows disagree, is not offered here, so this item leaves it off.
- "This run’s last reconstruction": the settings this run used for its newest reconstruction. It is offered only once this run has reconstructed a patch. EpiCycle's defaults are filled first, then every setting that reconstruction recorded is put over them, including its two channels.
  - A reconstruction made from October 2026 on records every setting: the inference settings the model was given when the patch was predicted (cell width, batch size, Even out brightness, Rescale), the CPU cores and every u-Segment3D setting. The item then gives back exactly those settings, and no grey line shows.
  - A reconstruction made before October 2026 recorded only part of its settings. While you use one, a grey line says so: "Settings that run did not record are EpiCycle’s defaults."
- "Custom": appears by itself as soon as you change any setting by hand, either channel included. When you pick a model refined in Step 3 (with Use for re-prediction in Model Refinement, or from the model registry), the two channels it was trained with are filled in for you; that does not make the list Custom. Any other model picked later (a Step 2 model, a Cellpose built-in model or a file) leaves the two channel boxes as they are, unless the run has no channels saved yet and has not predicted: then the boxes show the channels the new model was trained on, or **Channel to segment** goes empty when the model does not say. A new run fills them from its model (see **A new run** below).

**What is remembered.** Each run remembers its own choice, its two channels and its **Remove specks smaller than**. A run made before these were remembered shows the channels its own newest record asked for: its newest prediction or reconstruction, by the time each was made (a one-channel patch's record counts too, since it keeps the channels you asked for; a record whose channel to segment was empty does not). It takes the values it shows when you first change one or first switch away from it. A run with no channels saved and no such record (the workspace's first run, or any run that has never predicted) shows the channels its model was trained on, or leaves **Channel to segment** empty, and the log says why: a view has no model, the model does not say which channels it reads, or the two views were trained on different channels. Those boxes are only shown: they are kept once you change a channel or predict. Switching to another run and back does not keep them, so until then they follow the run's model. When you open the workspace again, or switch to another run, that run's choice is filled in again, then its two channels; its **Remove specks smaller than** too when the choice is Custom (under a standard set the set's own value is used). Other hand-changed values are not saved: a run left on Custom shows Custom, and its other settings keep the values they have. A run that has never had a choice made in it (a run made before these settings were remembered) shows EpiCycle defaults when you open the workspace; when you switch to it from another run, the settings it does not remember keep the values they have and the list shows Custom.

**A new run** starts from "EpiCycle defaults" for its own model, never from the settings of the run you were in. What this paragraph says about the two channel boxes also holds for the workspace's first run and for any run that has never predicted, as long as it has no channels saved. A model trained in Step 2, or refined in Step 3, fills the two channel boxes with the channels it was trained on, when its training record says so and its two views (lateral and axial) agree; otherwise **Channel to segment** starts empty. A Cellpose built-in model does not say which channel it reads, so **Channel to segment** starts empty, showing "pick a channel" in grey, and **Helper channel** starts at 0: none. Once a patch with more than one channel is ticked, the line under the buttons says "Pick the channel to segment." and Predict Selected stays grey until you pick one. Filling the empty box is not counted as a change: the list keeps EpiCycle defaults, so **Remove specks smaller than** still follows the model. Changing a filled box later makes the list Custom, as any other change does. The New Run dialog says so under its list of built-in models: "A built-in model does not say which channel it reads: after creating the run, pick the channel to segment in Settings."

**A patch with one channel** (a ZYX stack, or a stack whose channel axis holds a single channel) is read by Cellpose as gray whatever the two boxes say, so it is predicted as 0: gray on its own. The boxes keep the run's choice, and the patch's record keeps the channels you asked for, so "This run’s last reconstruction" and Step 4 never get 0: gray from it. While such a patch is ticked, the line under the buttons ends by naming it, for example "Wound_patch_5.tiff: one channel, runs as 0: gray." (several read "2 patches: one channel, run as 0: gray."), but not after "Predict the ticked patches first.": the line says one thing at a time. While **Channel to segment** is empty, Predict runs only when every ticked patch has one channel.

---

## Inference settings (open)

These decide how the model reads each slice.

- **Channel to segment:** "The channel that shows what this model finds: the cell outlines for a cell model, the nuclei for a nuclei model. As Cellpose’s code: 0 gray (all channels averaged), 1 red, 2 green, 3 blue."
- **Helper channel:** "An optional second channel that helps a cell model tell touching cells apart, usually the nuclei; 0 means none. A nuclei model needs none. With Channel to segment on 0: gray, this box is not used."

  For a nuclei model, such as Cellpose's built-in nuclei model, put the nuclear stain in Channel to segment and leave Helper channel at 0: none. On a patch with two or more channels, 0: gray averages them into one image, so a nuclear stain gets mixed with the membrane: pick the stain's own number instead.

  The items are Cellpose's channel codes, not your layer names: 1 is the first channel of the patch, 2 the second, 3 the third.

  After Predict, the row's eye shows what the model saw, so you can check the channel before you reconstruct (see **Patches**).
- **Typical cell width:** "About how wide one cell is, in micrometres; the model looks for cells near this size." (Auto) Auto lets the model use the cell size it was trained on.

  The model works in pixels, so each patch's width is turned into pixels from that patch's own voxel size (the one Step 1 recorded for its source image, or, when Step 1 has none, the one written in the patch file). Under the box a grey line shows the result for the patch you highlighted in the list (or the first patch Predict would run), for example "≈ 24 pixels at 0.33 µm per pixel".

  A patch with no voxel size cannot be converted. While the width is not Auto and a patch Predict would run has none, an amber line says so: "No voxel size — cell width falls back to pixels." That patch is then predicted with the number you typed taken as pixels, and the log names it. Patches cut out in Step 1 usually take their voxel size from Step 1; their own files carry none.
- **Batch size (slices at once)**, with its program name batch_size in small grey letters under it: "How many slices go to the graphics card together; lower it if the card runs out of memory." (128) If Predict stops with “CUDA out of memory” in the log, lower it and predict again.
- **Even out brightness:** "Stretch each slice’s brightness before the model sees it." (on)
- **Rescale:** "Used only when no cell width is set: when ticked, the image is passed to the model at its own scale; when unticked, the model resizes it by its own trained cell size. The rule used is recorded with the result." (on)

  When a cell width is set, the model always resizes the image from that width, and this tick has no effect.

Under the two ticks a grey note says "Outline tolerance and cell confidence are not used by the 3D build, so they are not shown." — Cellpose's two mask thresholds only shape the 2D outlines, and the 3D build is made from the model's flows instead, so there is nothing to set.

**Not shown.** Cellpose's own two mask thresholds and its “compute masks” step are not on the page. The 3D build reads the model's flows and cell probabilities, never its 2D masks, so Predict no longer makes them. Older patches may still have mask files in their prediction folder; they are never read.

---

## Advanced (closed)

Each setting shows a plain name, with the program's own name in small grey letters under it. The grey name helps if you look a setting up in the Cellpose or u-Segment3D documentation. If you are not sure, leave Advanced alone: **Start from** gives sensible values.

**The two settings at the top**

- **How 2D becomes 3D** (Direct): "The way the three views are joined into 3D cells. Only Direct is offered."
- **CPU cores** (6): "How many processor cores the 3D build uses." A number close to all of your computer's cores can make napari slow while it reconstructs.

**Joining slices into cells.** "u-Segment3D, the engine that builds the 3D cells. Each group keeps its own help lines." Seven groups follow, each closed until you open it.

**Smooth before joining:**

- **Smooth the cell maps** (off): "Blur each view's cell-probability map before the views are joined, so cells do not step from slice to slice."
- **Smooth the flows** (off): "Blur each view's flows before the views are joined."
- **Blur across slices** (3.0): "How far the blur reaches across the slices of each view; higher removes stripes between slices."
- **Blur within a slice** (1.0): "How far the blur reaches within a slice; keep it low to keep detail."
- **Ignore flows below** (0.20): "Flows where the cell probability is below this are set to zero. Used only with Smooth the flows."
- **Weigh flows by cell probability** (on): "Scale the smoothed flows by how likely each voxel is inside a cell. Used only with Smooth the flows."

While Smooth the flows is off, the last two are grey, and a grey line under them says why: "The two fields above are used only with Smooth the flows."

**Follow the flows:**

- **Step fade** (0.001): "How fast each step shrinks as voxels follow the flows; higher suits long, thin cells."
- **Steps** (25): "How many steps each voxel takes along the flows."
- **Momentum** (0.98): "How much of the previous step each step keeps (0 to 1). Rarely changed."
- **Rescale the flows at each step** (on): "Set the flows back to unit length at every step."
- **Use several cores** (on): "Cut the volume into pieces and follow the flows on several cores at once."
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

- **Remove specks smaller than** (10000 voxels; 100 under EpiCycle defaults for the built-in nuclei model): "Pieces of the 3D mask with fewer voxels than this are removed."
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

- **Skip the size limits** (on), at the top of the group: "When ticked (the default), the final clean-up after the 3D build does not run, so the size limits below drop nothing. Untick it to apply them." While it is ticked, a grey note sits under it: "Size limits are off while this is ticked."
- **Smallest cell** (200 voxels): "Cells with fewer voxels than this are dropped."
- **Largest cell** (no limit): "Cells with more voxels than this are dropped."
- **Largest vs typical** (10.0): "With the tick below: a cell larger than the average size plus this many spreads is dropped."
- **Drop cells far from the typical size** (on): "Drop cells much larger than the typical cell, by the rule above."

**With the defaults, the size limits drop nothing.** The final clean-up is skipped unless you untick Skip the size limits. Step 4 builds cells the same way.

---

## Same settings in Step 4

Step 4's Segment section uses the same plain names, groups, help lines and defaults, and the same cell width in micrometres. Both steps' channel boxes count as Cellpose does (0: gray, then 1 for the first channel), so the same pick is the same channel in both. Two things differ in Step 3: the channel boxes name each code by its colour rather than by your layer names, and the batch size sits at the top of the settings here, while Step 4 keeps it under Advanced.
