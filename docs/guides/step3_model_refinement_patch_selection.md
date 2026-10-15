# Patch Selection

Picking which finalized patches to extract training pairs from, and seeing what is already on disk to train on.

"Finalized patches. Ticked patches are the ones Extract Training Pairs runs on; every extracted pair trains."

---

## What “finalized” means

This list only shows patches whose status is **finalized**. That means you've gone through three steps in the Curation section for each patch:

1. Curated the masks (fixed mergers, splits, deletions) until you're happy
2. Clicked **Accept Volume** to lock the curation state
3. Clicked **Finalize Labels** to commit those masks as the canonical labels

Patches still in `accepted`, `curating`, or any earlier state won't appear here — the model shouldn't train on half-finished work.

If none is finalized yet, the list says:

"No finalized patch yet."

"Finalize a patch in Curation to train on it. Accepted and curating patches are not listed."

---

## What a row shows

Each row has a tick box, a small picture of the patch (the same one the Session and Curation lists show), its name and status, and two small chips about its training pairs:

| The chip says | Meaning |
|---|---|
| "no pairs yet" | No pairs on disk for this patch |
| "3 lateral" | How many lateral (XY) pairs Extract Training Pairs already wrote for it. The chip turns amber under 20: the patch will barely move the lateral model. Hover the row for the whole sentence, with the count |
| "900 axial" | How many axial (XZ / YZ) pairs it wrote for it |

Two more amber lines can appear under a row. Hover the row for the whole sentence.

| The line says | Meaning |
|---|---|
| "⏱ Stale — re-extract before training." | You curated the patch again after its pairs were extracted, so the pairs on disk are older than its labels |
| "⚠ Mixed settings — re-extract to match." | The patch was extracted with different settings than the ones in Extraction Settings now |

Extracting again overwrites the patch's existing pairs.

---

## Ticks are the extraction scope

Tick the patches **Extract Training Pairs** should run on. Ticks do not change what trains: training reads every pair on disk, from every patch that has pairs, ticked or not.

- **All** ticks every patch; **None** takes every tick off. Ctrl+A in the list ticks every patch too.
- Under the buttons a count reads like this: "2 of 5 ticked".
- Under the count, one line says what training would read: "Will train on: 543 lateral + 1,280 axial pairs on disk, from 3 patches." With nothing extracted it says "Will train on: nothing yet."
- When the lateral pairs on disk are few, an amber block follows: "⚠ Lateral pairs are few (40) — the lateral model may barely change."
- With no tick, Extract Training Pairs says why: "Extract needs a ticked finalized patch."

---

## How to pick

**For a typical refinement session:**
- Pick 2–5 finalized patches that **represent the variability** you care about — different regions, different tissue layers, different staining intensities
- Even a few hundred curated cells across 2–3 patches is enough to noticeably improve predictions on similar tissue
- More patches = better generalization, but diminishing returns past ~10 patches for most cases

---

## What if no patches show up?

You haven't finalized any patches yet. Two routes:
- Go to the **Curation** section, finish curation on a patch, click **Accept Volume**, then **Finalize Labels**
- If you have patches that should be finalized but aren't appearing, check the patch list in **Predict + Reconstruct** — the status word should read `finalized`

---

---

## Extracting training pairs

Cellpose is a 2D segmenter, but your data is 3D. Refinement trains **two separate 2D models** — one for top-down (XY / lateral) slices, one for side-on (XZ / YZ / axial) slices — and the 3D reconstruction pipeline combines their predictions.

This step **slices** each finalized patch along three orthogonal axes, picks a representative subset of those slices, and saves `image_slice + mask_slice` pairs to `artifacts/step3/refinement/pairs/{patch_stem}/(spatial|axial)/` for training to consume (inside the run's own `refinement` folder when you work in a named run). The folder name `spatial` is the lateral (XY) set; the screen says lateral.

A separate `manifest.json` in the `refinement` folder records every setting + per-patch counts, so months later you can reproduce exactly what was extracted.

---

## Extraction settings

Below the extract button, **Extraction Settings** is a collapsed block (click its title to open it). It holds a one-line summary of the current effective extraction config, the `Configure per-orientation params…` button, and the Train / Validation Split (train fraction, chunk size, keep neighbouring slices together; see the Training Configuration guide for what they do). To **change** the extraction config, click `Configure per-orientation params…` to open the modal.

Inside the dialog you get two independent columns — **Lateral (XY)** and **Axial (XZ + YZ)** — because cells genuinely look different in those two views and benefit from different tuning. Saved values persist per workspace.

---

## The knobs (each appears in both columns)

### Min labels per slice

Slices with fewer than this many distinct cells are skipped — they don't teach the model much. **Defaults:** Lateral = 3, Axial = 2 (axial cross-sections often have fewer cells per slice).

### Edge buffer

Slices within this many indices of the volume edge are excluded — labels there are often partial / clipped by the patch boundary. **Defaults:** Lateral = 5 (Z is small, ~80 slices), Axial = 20 (Y/X are typically 512+).

### Slices per patch

Maximum number of slices to extract from each patch, **per sub-view**. `0` means *all valid slices*.

When > 0, picks are sampled **uniformly across the stack** (not greedy-from-bottom), so a cap of 50 with 200 valid slices picks slices ~at indices 2, 6, 10, …, 198. The full Z range is represented.

> **⚠ The cap is per-sub-view, not per-orientation.**
>
> Axial = XZ + YZ. An axial cap of `N` yields **up to 2N slices per patch** (N from each sub-view). Lateral has one view, so N is exact.
>
> If you want *exactly 100 axial slices per patch*, set the axial cap to 50.

### Min distance

Minimum spacing (in slice indices) between two picked slices in the same view. Acts as a soft floor after uniform sampling — only matters when the cap is very high. **Defaults:** Lateral = 3, Axial = 5.

### Mask preprocessing (Remove boundaries / Relabel / Min region size / Smoothing disk)

Standard pre-training mask hygiene — separates touching cells (1 px erosion), drops fragments below the min size, smooths boundaries via per-instance morphological open + close. **Defaults:** all on, min_size = 100 (Lateral) / 50 (Axial), smoothing disk = 1.

> **For iterative refinement, consider turning Remove boundaries OFF.** Each round shaves 1 px off every label, so repeated rounds without it OFF can shrink cells across iterations. Once you have curated ground truth, your boundaries don't need the erosion-to-separate trick anymore.

---

## Worked example: 100 axial slices per patch?

You set `axial.slices_per_patch = 100` for a patch with shape `(80, 512, 512)`, `axial.min_labels=2`, `axial.edge_buffer=20`. Trace:

```
xz view (slicing along Y):
  N_total         = 512
  N_candidates    = 512 - 2×20 = 472
  N_valid         = slices where unique labels ≥ 2  (say ~80% → 378)
  After cap (100) = 100  ← uniform-stride pick from 378

yz view (slicing along X):  same math → 100
```

**Result: ~200 axial slices per patch.** Plus whatever lateral contributes (controlled separately).

If you re-run with `axial.slices_per_patch = 0`, you'd get ~378 + 378 = ~756 axial slices per patch — all valid slices retained.

---

## Re-extraction is patch-safe

Extracting again with different settings: only the **selected** patches get re-extracted. Other patches keep their existing pairs on disk. The manifest merges new counts on top.

If you change settings between runs, the unrun patches now have *stale settings* — an amber line, "⚠ Mixed settings — re-extract to match.", appears under their row, and a dialog warns you when you next try to extract.

---

## Running extraction (the button)

Tick the patches in **Patch Selection**, then click **Extract Training Pairs**. With no tick, the line under the button says "Extract needs a ticked finalized patch."

While it runs, the button becomes its own **Cancel** (it stops within a slice) and the bar and the line under it show the progress: "Extracting slice 12/40… [0:12 | 0:30]". The ticks, the list and every setting are locked until it finishes.

When extraction finishes:

- Pairs land at `{workspace}/artifacts/step3/refinement/pairs/{patch_stem}/(spatial|axial)/*_img.tif` + `*_mask.tif`
- `spatial/` = XY pairs (used to train the **lateral model**)
- `axial/` = XZ + YZ pairs (used to train the **axial model**)
- The line under the bar reads like this: "Training pairs ready [0:30] · 3,200 lateral + 1,280 axial = 4,480 pairs from 3 patches"
- Each row's line under its name now counts that patch's pairs, and the Will-train-on line under the list follows
- **Clear…** (next to the extract button) deletes every patch's pairs from disk after a confirmation; use it after a big preprocessing change

A patch that has no pairs yet says so on that line: "WholeScale_patch_3.tiff has no pairs yet; Extract makes them. Last extraction: never."

---

## Troubleshooting

### Nothing extracted

Likely cause: `min_labels_per_slice` is too high for your data, or `edge_buffer` is eating the whole stack. Check the `[PERVIEW-PRESCAN]` lines in the console — they show candidate count per view per patch.

### Progress bar doesn't match the expected number

This is fixed as of 2026-05-19: the pre-scan now uses the same per-view filters as actual extraction. If you see a mismatch, the `[PERVIEW-PRESCAN]` and `[PERVIEW-EXTRACT]` console lines for the same view should print identical valid counts.

### An amber ⏱ line under a patch: "⏱ Stale — re-extract before training."

That patch's `labels_3d.tif` on disk has been re-saved (you curated more) **since** the last extraction. The on-disk training pairs don't reflect the new edits. Re-extract that patch before training.

### An amber ⚠ line under a patch: "⚠ Mixed settings — re-extract to match."

That patch's manifest snapshot shows different extraction settings than the current UI. Training would mix preprocessing regimes — usually a bug. Re-extract that patch with current settings, or accept the mixed state via the dialog at extraction launch.

---

## Train / validation split

Inside Extraction Settings, below the per-orientation summary, the **Train / Validation Split** holds the train fraction, the chunk size and the keep-neighbouring-slices-together switch. It decides which extracted pairs train the model and which are held back to score it; the Training Configuration guide explains each control.

---

## Related

- **Training Configuration** — preset, epochs, learning rate, batch size and the Cellpose hyperparameters
- **Fine-Tuning** — what happens after the extracted pairs are ready
