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

You typically run **Extract** once per curation session (or after major edits), then use the **PCA / Cluster / UMAP / tSNE** recompute buttons to explore different parameter settings without re-reading the volume.

---

## The buttons

### Extract (amber, primary)
Reads the current 3D mask + raw image, computes the full feature set, saves a per-patch CSV under `artifacts/qc/{patch}_features.csv`. Spawns a background worker — UI stays responsive, progress shows below.

This is the canonical path when extraction parameters (spacing, erosion depth, channel selection, etc.) change.

### Settings...
Opens the **QC Extraction Settings** dialog (modal). Pick voxel spacing, erosion depth, neighbor search radius, intensity channels, QC score method, clustering algorithm, and optional UMAP / t-SNE. Settings persist with your workspace.

**The voxel spacing here is the fallback.** Each patch has its own voxel size, shown on its row in **Session** (from Step 1, from the file, or set by you with **Set spacing…**), and Extract measures a patch with that one. The spacing in this dialog is used only for a patch that has no voxel size of its own; the line under it in the dialog says so.  Above that note the dialog shows, read-only, the voxel size the next extraction will use for the patch in view (and where it came from), or in amber that the patch has no size of its own, so its file's own tags are tried first and then the fallback below; with no patch in view it counts the patches that have their own size. With no spacing anywhere, the features are measured in voxels. Changing a patch's voxel size after its features were extracted makes them out of date, and the usual out-of-date notice tells you so when you load it: run **Extract** again.

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

**Adding a new feature category:**
1. Open Settings → tick the additional category (e.g. **Topology**)
2. OK → click **Extract** (Settings change requires full extract)
3. Wait for completion, new feature columns appear in the df

---

## Troubleshooting

### Extract takes a long time
50+ features × thousands of cells = real work. Typical patches take 30s–5min depending on cell count and which features are enabled. If too slow, deselect categories you don't need in Settings.

### *Extract features first* warning when clicking PCA / Cluster
The in-memory df doesn't exist yet (or is empty). Run **Extract** first.

### Cluster button greyed out
Settings → Clustering → set Algorithm to something non-empty. Or check that `pca_features` has at least 2 items selected.

### Features changed after re-extract but cluster_id stayed
Cluster_id is preserved through re-extract but isn't recomputed automatically. Click **Cluster** after the re-extract to refresh.

---

## Related

- **QC Plot** — the visualization where you actually triage cells
- **Curate** — where you produce the curated masks that feed into extraction
- **Model Refinement** — only operates on finalized patches; QC triage helps you decide which patches to finalize
- **Feature reference** — every feature, its equation and what an unusual value means: `docs/research/QC_FEATURE_REFERENCE.md` in the repository, with a plain-language version in `docs/research/QC_FEATURES_BIOLOGIST_GUIDE.md`
