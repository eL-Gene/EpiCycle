# Curate

Where you open the 3D annotator on a patch, edit the masks until you're satisfied, and lock the result so it can feed into Model Refinement.

---

## The workflow at a glance

```
select patch  →  Open 3D Curation Annotator  →  edit masks
                                                     ↓
                                                Save Curation
                                                     ↓
                                                Accept Volume   (locks state)
                                                     ↓
                                                Finalize Labels (commits as training data)
```

Each step is its own button. You can stop and restart at any point — the workspace remembers progress.

---

## Step 1 — Open the 3D Curation Annotator

Pick a `reconstructed` patch from the Curate Bucket list (left), then click **Open 3D Curation Annotator** (the amber-styled primary button).

A separate dock opens with the 3D multi-view annotator. You'll see:
- A main 3D viewer on the left
- Three 2D subviewers (XY / XZ / YZ) on the right
- A floating Label Navigator window for triage actions
- An Advanced Features dock with three tools on one layout: **Reshape**, **Refine** and **Add**

This is where the actual mask editing happens. Each of the dock's three tabs has its own `?` and its own guide.

> **Tip:** keep the napari main window where you can see it — the 3D rendering of the mask updates as you edit, useful for catching errors.

---

## Step 2 — Save Curation

As you edit, the **Save Curation** button lights up to indicate unsaved changes.

Click it to write the current mask state to disk. This is **non-destructive** — it just updates the on-disk TIFF + the QC features CSV.  You can save many times during a single session.

**Undo** lets you revert the last curation action (one step at a time).

---

## Step 3 — Accept Volume

When you're done editing the masks, click **Accept Volume**. This locks the curation state — no more editing until you explicitly **Reset to Predicted** (which discards your work).

Accepting transitions the patch from `curating` → `accepted`. The patch is now ready for finalization.

> **Why this gate exists:** it prevents you from accidentally finalizing a patch you're still editing. Think of it as the *I'm satisfied* checkpoint.

---

## Step 4 — Finalize Labels

Click **Finalize Labels** to commit this patch as canonical training data. Status moves from `accepted` → `finalized`.

**What finalization changes:**
- The patch becomes selectable in the **Model Refinement** section
- The masks are considered ground-truth for downstream analysis
- Status badge becomes dark green

You can still edit a finalized patch if you change your mind — use **Reset to Predicted** to start over, or open the annotator again to make tweaks (then re-Finalize).

---

## The Curate Bucket list

The list on the left shows your `reconstructed`, `curating`, `accepted` and `finalized` patches.

The row of the patch that is loaded in the viewer carries an amber bar on its left edge and a faint glow, apart from the highlight of the row you clicked.  Each row shows a picture of the patch, its name, its status on the right (the coloured dots and word), and under the name the marks *✔QC* (its cells have been measured) and *ed* (you edited it in this session) first, then its axis order, size, channels and label count, then its voxel size (the amber word voxels when it has none) and the number of labels in each tier — curated, good, review, bad, garbage — which wraps onto more lines when the column is narrow. The picture is the middle slice of the raw patch image — the same picture as on the patch's row in **Session**: grey for a patch with one channel; for two channels, channel 0 (nuclei, by EpiCycle's convention) in magenta and channel 1 (membrane) in green. It shows the image, not the labels; the status says how far the patch has come. Hover over the picture to read what it is, or why there is none.

A patch whose file is gone keeps its row, with an amber line under it; a click on it is refused (the annotator needs the image), and its **⋯** offers to forget the patch — see the Patches guide in **Predict + Reconstruct**.

Selecting a patch loads its labels into the 3D annotator if it's already open, or readies it for the next time you click Open 3D Curation Annotator.

---

## Common workflows

**Curating a fresh reconstruction:**
1. Patch is at `reconstructed`
2. Select → Open 3D Curation Annotator
3. Use the annotator's tools (split, merge, interpolate, etc.) to fix mistakes
4. Save Curation periodically
5. When satisfied → Accept Volume → Finalize Labels

**Coming back to an in-progress patch:**
1. Select the `curating` patch
2. Open 3D Curation Annotator — your previous edits are still there
3. Continue editing
4. Accept + Finalize when ready

**Discarding curation and starting over:**
1. Click **Reset to Predicted** (in the curation actions row or elsewhere)
2. Confirm in the dialog
3. The patch returns to `predicted`; you can re-run reconstruction or re-curate from scratch

---

## While an operation is running

Step 3 does one thing at a time. Whenever a longer operation is underway — loading a patch, extracting QC features, computing PCA / UMAP / t-SNE / clustering, saving, committing triage, predicting, reconstructing, extracting training pairs, or training a refined model — the Curate panel and the step buttons at the top of the window gray out and clicks are ignored. This is deliberate: it prevents you from, say, switching patches or starting a second computation in the middle of the first one and corrupting the result.

What you'll see:
- A **progress bar** under the QC plot area (below its Assign row) animates in *scanning* mode, and the **status line** directly beneath it tells you what's running (e.g. *Loading patch001 — downsampling…* or *Computing UMAP…*).
- In the QC plot area the histogram's controls and lines and the Assign row are greyed too; its **Histogram** / **Selection** tabs, the divider and **Colours in viewer** stay live.
- The panel controls and the step navigation are dimmed until the operation finishes.
- The one button that stays available is the **Cancel** button for whatever is running — the Model Refinement *Cancel*, *Cancel Extraction*, or the predict/reconstruct *Cancel* — so you can stop it if you started the wrong thing.

You can still **pan, zoom, and rotate** the 3D view the whole time — only the editing/action controls are locked, not the camera. When the operation finishes, everything re-enables on its own and the status line returns to showing the current patch.

> **Why patch loading no longer freezes:** the heavy work of reading and shrinking a large patch now happens in the background, so the window stays responsive (the progress bar moves) instead of locking up for several seconds.

---

## Related

- **QC Features** — produces the morphology features you'll see plotted in the QC Plot panel
- **QC Plot** — where you do bulk triage: the plot, the triage histogram under it, the Selection card and the Assign row
- **Model Refinement** — what to do with patches once they're finalized
