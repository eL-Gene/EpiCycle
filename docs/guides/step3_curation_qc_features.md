# QC Features

Extract per-cell morphology + intensity features from the 3D reconstruction, run PCA / clustering / dimensionality reduction on them, and feed the results into the QC Plot for triage.

---

## The big picture

```
3D mask + raw image
        ↓
[ Extract ]  ← reads volume, computes 50+ features per cell, saves to CSV
        ↓
in-memory DataFrame (the *QC features df*)
        ↓                ↓                 ↓
   [ PCA ]         [ Cluster ]      [ UMAP / tSNE ]    ← recompute on the same df
        ↓                ↓                 ↓
    QC Plot (right column) — colour labels by any feature, lasso-select, assign tiers
```

You typically run **Extract** once per curation session (or after major edits, once you have pressed **Save Curation**), then use the **PCA / Cluster / UMAP / tSNE** recompute buttons to explore different parameter settings without re-reading the volume.

---

## The buttons

### Extract (amber, primary)
Reads the current 3D mask + raw image, computes the full feature set, saves a per-patch CSV under `artifacts/step3/qc/{patch}_features.csv` in the workspace (for a named run, under `artifacts/step3/runs/{run}/qc/`). Spawns a background worker — UI stays responsive, progress shows below.

**Save Curation first.** Extract reads the saved 3D mask from disk, so with unsaved mask edits it does nothing and the line under the plot says "Extract refused: this patch has unsaved curation edits. Save Curation first, or switch patch and choose Discard."  After a finished Extract the **Undo** buttons start over: the edits they would have taken back belong to the old table.

This is the canonical path when extraction parameters (spacing, erosion depth, channel selection, etc.) change.

### Settings...
Opens the **QC Extraction & Settings** dialog (modal). Pick voxel spacing, erosion depth, neighbor search radius, intensity channels, QC score method, clustering algorithm, and optional UMAP / t-SNE. Settings persist with your workspace. **OK** keeps your changes; **Cancel** drops them. The dialog scrolls, from top to bottom:

- **Extraction parameters** — **Spacing** (Z, Y, X; **Auto** reads it from the image layer in view), **Erosion iters**, **Neighbor radius**, **Exclude border cells from composite/PCA fitting** and **Invert Z axis (flip apical/basal)**, each explained below; then **Compare with**, only when the patch has another run with labels (see *Comparing with another run*).
- **Intensity channels** — which channels to measure brightness in (see below).
- **QC Score (qc_score)** — **Method**: *zscore* (the mean distance from typical, feature by feature) or *pca* (the distance from typical in the top PCA components); a grey line under it says which. Under **Features used in qc_score**, each feature group opens with a click and lists its features with a tick each; the group's header counts how many are ticked (*● 3 in PCA*). The ticked features feed the QC score, the PCA, UMAP and t-SNE. **Select all** ticks every feature; **Reset defaults** goes back to the standard set.
- **Clustering** — **Algorithm** (*leiden*, *hdbscan*, *kmeans* or *gmm*), with a grey line saying what it does and only the settings it uses: **Resolution** and **N neighbors** for leiden, **Min cluster size** for hdbscan, **N clusters** for kmeans and gmm.
  - **PCA dimensions** is how many PCA components clustering uses; the *pca* score uses the same number, and it sets how many PCA columns in the table are saved.
  - **Distance metric** is how distance between cells is measured. **Auto-run clustering after Extract** clusters as soon as Extract finishes.
- **Nonlinear Dimensionality Reduction** — **Compute UMAP** (with its **Min dist**) and **Compute t-SNE** (with its **Perplexity**) add 2D maps to the plot at each Extract; each costs extra time. PCA is always computed.

Each extraction parameter has one line under it saying what it does. **Erosion iters:** "How many voxel layers count as the boundary of each object when the intensity features split it into boundary and interior." **Neighbor radius:** "How far, in voxels, to look for neighbours when counting them and measuring the nearest one." **Exclude border cells from composite/PCA fitting:** "Objects touching the top or bottom of the stack are left out when the composite score and the PCA are fitted, then placed on them afterwards." **Invert Z axis:** "Tick when the apical side is at the top slice instead of the bottom, so apical/basal features read the right way round."

**The voxel spacing here is the fallback.** Each patch has its own voxel size, shown on its row in **Session** (from Step 1, from the file, or set by you with **Set spacing…**), and Extract measures a patch with that one. The spacing in this dialog is used only for a patch that has no voxel size of its own; the line under it in the dialog says so.  Above that note the dialog shows, read-only, the voxel size the next extraction will use for the patch in view (and where it came from), or in amber that the patch has no size of its own, so its file's own tags are tried first and then the fallback below; with no patch in view it counts the patches that have their own size. With no spacing anywhere, the features are measured in voxels. Changing a patch's voxel size after its features were extracted makes them out of date, and the usual out-of-date notice tells you so when you load it: run **Extract** again.

**Intensity channels.** One row per channel of the patch; tick a row to measure that channel's brightness inside every label. Channels are counted from 0, and a row also says what the channel was in the active run: *Channel 0 (channel to segment)*, *Channel 1 (helper channel)*, or just *Channel 2* for a channel the run did not use. A one-channel patch has no rows here. No role is shown when the run's channel to segment was **0: gray** (every channel averaged). Channel 0 is **1: red** in the channel boxes, Channel 1 is **2: green**, and so on. Hold the pointer over a row to see where its columns go: the columns of channel 0 start with ch0_, those of channel 1 with ch1_, and so on. The patch files carry no channel names, so the rows cannot show the names of your stains.

### Recompute row: PCA | Cluster | UMAP | tSNE
These operate on the **in-memory df** using the current Settings — no worker, no re-read of source images. Use when you tweak one knob (e.g. number of PCA components, clustering algorithm) and want to see the result quickly without a full extract.

**Workflow if you change Settings:**
1. Open Settings, change the knob (e.g. cluster algorithm: leiden → hdbscan)
2. Click OK
3. Click **Cluster** → in-memory df gets new `cluster_id` column, QC plot refreshes

For changes that require regenerating the feature columns themselves (e.g. spacing, channels), run **Extract** instead.

---

## What gets computed

The feature catalog covers (50+ features total):
- **Morphology** — volume, surface area, sphericity, compactness, etc.
- **Shape** — elongation, aspect ratio, solidity, etc.
- **Intensity** — per-channel mean / median / std (one set per ticked channel)
- **Membrane** — boundary intensity profile
- **Adjacency** — number of neighbors, distance to the nearest neighbor
- **Topology** — Euler number, skeleton branches and nodes
- **Probability** — the u-Segment3D reconstruction probability volume: how confident the model is inside each cell and at its boundary (when available)
- **Composite** — `qc_score` (the mean absolute robust z-score of 15 decorrelated features; the PCA-norm score is the alternative method)

Every feature has a tooltip in the Extract Features dialog and in the EDA tools downstream; the tooltip gives its definition, with the formula where that helps.

---

## Comparing with another run

A patch can carry more than one run: for example cells segmented from the membrane channel and nuclei segmented from DAPI. The other run can then serve as a check while you triage: for every object of the run you are curating, which objects of the other run share its voxels, and how many. A cell with no nucleus or with two, or a nucleus lying across two cells, then shows up in the QC plot, where you can colour, sort and lasso it. It is a check, not an edit: nothing in either run's labels changes.

**Picking the other run.** When the patch in view has another run with labels, **Settings...** shows a row "Compare with:" with one line under it: "Finds, for each object of this run, the objects of the chosen run it overlaps, and adds match columns and a Colour by entry named after that run." Pick the run there and press **Extract**; leave it at *(none)* to skip the comparison. When no other run of the patch has labels, the row is not shown. Each patch keeps its own pick: the next Extract of that patch, and **Extract All**, compare it with the same run until you change it, and a patch left at *(none)* stays at *(none)*. If the run a patch was compared with has been archived or deleted, the row shows *(none)*.

**The five match columns.** Extract writes five columns into the feature table. Each one means the same thing whichever run you are curating: curate cells and the matches are nuclei; curate nuclei and the matches are cells.

| Column | What it says |
|---|---|
| `match_count` | How many objects of the other run are linked to this one |
| `match_id` | The label of the linked object it shares the most voxels with (0 = none, too small, or not measured). A label number, so it is never offered as a plot axis |
| `match_overlap` | The voxels it shares with that best match, over the size of the smaller of the two (0 to 1) |
| `match_shared` | The most objects of this run that any one of its matches is linked to (2 or more: a match it shares with another object) |
| `match_minor_ratio` | The voxels shared with its second match over those shared with its first (0 to 1; empty with fewer than two matches). About 0.2 is a small fragment beside the real match; about 0.7 is an object split almost evenly between two |

**The rule.** "Two objects are linked when their shared voxels are at least 0.20 of the smaller one." The number is the same seen from either run. An object much smaller than the typical object of its run (under a tenth of the median size, counted in voxels) is never linked, on either side.

**Colouring by it.** In the QC plot, Colour by offers an entry named after the run you compared with, for example "Overlap with Nuclei" ("Overlap with the Default run" for the Default run, and "Overlap with other run" when the name is not known). It colours every object by what the other run says about it. The legend always lists the words in this order:

- "no match": no object of the other run is linked (a cell with no nucleus). Many border cells read this because their match lies outside the patch; colour by **Border cell** to tell them apart.
- "one match": exactly one, and that match is linked to nothing else. The two runs agree.
- "several matches": two or more (a cell holding two nuclei: a likely merge). Such an object may also share one of them.
- "shared match": one match, which is also linked to another object of this run (a nucleus lying across two cells: a likely split).
- "too small to judge" (dark grey): the object is under its run's size floor.
- "not measured" (light grey): the values were blanked after an edit, or the label was deleted.

Click a legend word to select its objects, as with any other colouring. The colours reach the viewer like any other Colour by. The match numbers can also be plotted on X or Y; to colour by them, use the **Overlap with** entry, so that an object that was not measured is light grey and never the low end of a colour scale.

The triage histogram offers the same entry under **Colour bars by**: each bar is then stacked by the six words, in the same colours as the plot's legend, with deleted labels in the grey ghost stack as for any other colouring.

**What it cannot tell you.** "One match means the two runs agree, not that the cell is right, and not that they are the same size: an object nested inside a much larger one still counts as one match. A merge where both runs merged, or where the second nucleus mostly lies in the neighbour, is invisible to this check. Interior membrane brightness is the signal that still catches it."

**When the values are out of date.** While you colour by the **Overlap with** entry, up to two lines can appear under the legend; the match columns' tooltips repeat them and name the other run:

- "Run 'Nuclei' has changed since these match values were measured. Extract again to refresh them." The other run's labels were saved again after this Extract.
- "Match values predate your edits. Extract again to refresh them." You painted, split, merged, smoothed, refined or deleted since that Extract, or the saved labels are newer than it. Every label you edited reads *not measured* at once, and so do new labels and labels you **Recalculate QC** in the Navigator; the values of their neighbours may have moved too, so Extract again for fresh numbers.

Neither line stops you working, and both go away after the next Extract.

**When Extract skips the comparison.** If the two runs' labels differ in shape, the two runs were made from different stack files, or the other run's file cannot be read or compared, Extract measures everything else and leaves the match columns out, saying why in the log, for example "The match columns were skipped: run 'Nuclei' was made from a different stack file (wound_2.tif, this run: wound_1.tif)." When it does compare, the log says so in one line: "Compared with run 'Nuclei': 412 links."

The match columns are **not part of the feature catalogue**: they need two runs of the same patch. They are left out of the standard PCA features, the QC score and Step 4.

---

## QC tiers

Each cell gets one of five tiers:

| Tier | Meaning |
|---|---|
| **curated** | Manually verified by you (lasso → Assign or annotator interaction) |
| **good** | Auto-assigned (lowest qc_score: the most typical cells) |
| **review** | Auto-assigned (middle of distribution) |
| **bad** | Auto-assigned (high qc_score) |
| **garbage** | Auto-assigned (highest qc_score: the most unusual cells, candidate for deletion) |

Drag the tier lines of the histogram under the QC plot (its **Histogram** view, Triage by qc_score) to adjust which qc_score values map to which tiers; **Undo assign** takes a drag back. The tier column lives in the same df as the features.

---

## Re-extract behavior

Re-running **Extract** on a patch:
- **Preserves** existing tier assignments (left-merged by label ID)
- **Drops** ghost rows (deleted / superseded labels) — they don't survive re-extract
- **Auto-saves** the merged df immediately after to disk
- **Warns** before discarding cluster_id (since clustering is computed separately)

---

## Common workflows

**Initial extraction on a freshly reconstructed patch:**
1. Open Settings → review parameters
2. Click **Extract** → wait for completion
3. Switch to the QC Plot to see the scatter

**Tuning the clustering algorithm:**
1. Open Settings → change Algorithm (leiden → kmeans)
2. OK
3. Click **Cluster** → new `cluster_id` column, plot refreshes
4. Look at the scatter colored by cluster
5. If unsatisfied, repeat with different settings

**Changing which features feed the QC score and PCA:**
1. Open Settings → under **QC Score**, click a feature group (e.g. **Topology**) and tick or untick its features
2. OK. If the patch already has features, the QC score is worked out again straight away from the new ticks (with the *pca* method it uses the PCA already in the table)
3. To refit the PCA on the new ticks, click **PCA** in the Recompute row (**Cluster**, **UMAP** and **tSNE** use the ticks too); to redo the score and the PCA together, click **Extract**. Extract always measures every feature, ticked or not, so the feature table keeps the same columns; the ticks only choose what feeds the QC score, the PCA, UMAP and t-SNE

---

## Troubleshooting

### Extract takes a long time
50+ features × thousands of cells = real work. Typical patches take 30s–5min depending on cell count. Unticking features under **QC Score** does not make it faster: Extract measures every feature either way. Unticking channels under **Intensity channels** does: their brightness is then not measured. **Compute UMAP** and **Compute t-SNE** each add time, so leave them off if you do not need them.

### *Extract features first* warning when clicking PCA / Cluster
The in-memory df doesn't exist yet (or is empty). Run **Extract** first.

### Cluster button greyed out
It is greyed only while another Step 3 job is running (an Extract, a prediction, a reconstruction, a training run and the like). Wait for that job to finish.

### Clusters gone after re-extract
Re-extract throws away `cluster_id` (the confirm box says so before it starts). Click **Cluster** after the re-extract, or tick **Auto-run clustering after Extract** in Settings to have it done for you.

---

## Related

- **QC Plot** — the visualization where you actually triage cells
- **Curate** — where you produce the curated masks that feed into extraction
- **Model Refinement** — only operates on finalized patches; QC triage helps you decide which patches to finalize
- **Feature reference** — every feature, its equation and what an unusual value means: `docs/research/QC_FEATURE_REFERENCE.md` in the repository, with a plain-language version in `docs/research/QC_FEATURES_BIOLOGIST_GUIDE.md`
