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

- "EpiCycle defaults": EpiCycle's standard values. This also resets the cell width, Even out brightness, Rescale, the batch size and the CPU cores.
- "u-Segment3D defaults (without its flow filter)": the standard values of u-Segment3D, the program that joins the slices into 3D cells. u-Segment3D's own last clean-up, a filter that drops cells whose flows disagree, is not offered here, so this item leaves it off.
- "This run’s last reconstruction": the settings this run used for its newest reconstruction. It is offered only once this run has reconstructed a patch. EpiCycle's defaults are filled first, then every setting that reconstruction recorded is put over them, including its two channels.
  - A reconstruction made from October 2026 on records every setting: the inference settings the model was given when the patch was predicted (cell width, batch size, Even out brightness, Rescale), the CPU cores and every u-Segment3D setting. The item then gives back exactly those settings, and no grey line shows.
  - A reconstruction made before October 2026 recorded only part of its settings. While you use one, a grey line says so: "Settings that run did not record are EpiCycle’s defaults."
- "Custom": appears by itself as soon as you change any setting by hand, either channel included. When you pick a model, the channels it was trained with are filled in for you; that does not make the list Custom.

**What is remembered.** Each run remembers its own choice. When you open the workspace again, or switch to another run, that run's choice is filled in again. Hand-changed values are not saved: a run left on Custom shows Custom, and the settings keep the values they have. A run that has never had a choice made in it shows EpiCycle defaults when you open the workspace; when you switch to it from another run, the settings keep the values they have and the list shows Custom.

---

## Inference settings (open)

These decide how the model reads each slice.

- **Membrane channel:** "The channel that shows the cell outlines, as Cellpose’s code: 0 gray (all channels together), 1 red, 2 green, 3 blue." (2: green)
- **Nuclei channel:** "An optional second channel, usually the nuclei, that helps the model tell touching cells apart; 0 means none." (1: red)

  The items are Cellpose's channel codes, not your layer names: 1 is the first channel of the patch, 2 the second, 3 the third.
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

- **Skip the size limits** (on), at the top of the group: "When ticked (the default), the final clean-up after the 3D build does not run, so the size limits below drop nothing. Untick it to apply them." While it is ticked, a grey note sits under it: "Size limits are off while this is ticked."
- **Smallest cell** (200 voxels): "Cells with fewer voxels than this are dropped."
- **Largest cell** (no limit): "Cells with more voxels than this are dropped."
- **Largest vs typical** (10.0): "With the tick below: a cell larger than the average size plus this many spreads is dropped."
- **Drop cells far from the typical size** (on): "Drop cells much larger than the typical cell, by the rule above."

**With the defaults, the size limits drop nothing.** The final clean-up is skipped unless you untick Skip the size limits. Step 4 builds cells the same way.

---

## Same settings in Step 4

Step 4's Segment section uses the same plain names, groups, help lines and defaults, and the same cell width in micrometres. Two things differ in Step 3: the channels are Cellpose's codes rather than your layer names, and the batch size sits at the top of the settings here, while Step 4 keeps it under Advanced.
