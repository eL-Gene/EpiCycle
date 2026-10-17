# EpiCycle release notes

## 0.1.1 (2026-10-16)

Patch release. Installing 0.1.0 with Option A or Option B (the release zip) produced a package without Step 2 and without the logging helpers, and napari failed at launch with `No module named 'napari_epicycle.hitl_training_annotation'`. Two directories lacked an `__init__.py`, so the packaging step skipped them; installs from a clone (Option C) were not affected. 0.1.1 adds the two files and a test that catches this. No other change.

If you installed 0.1.0, remove that environment (`conda env remove -n epicycle`) and install again following the README.

## 0.1.0 (first public pre-release, 2026-10-14)

Pre-release 0.1.0. EpiCycle is actively developed; the interface, APIs and on-disk file formats may change between versions, and existing workspaces are migrated on load where a format changes.

### What EpiCycle does

- **Step 1, Data Prep:** load a 3D volume, optionally preprocess its channels, and draw and extract the ROI patches you will work on.
- **Step 2, HITL 2D:** annotate slices in the lateral (XY) and axial (XZ, YZ) views, train a Cellpose model, review its predictions, stage corrections and train again, round after round.
- **Step 3, Reconstruct and Curate 3D:** predict every slice, fuse the 2D results into 3D labels, measure more than 50 features per cell, sort cells into good, review, bad and garbage, fix cells in the three-view editor, and train a refined model.
- **Step 4, 3D Stack Segmentation:** run the finished model over whole stacks, measure, tidy and export, and explore the results in the Workbench.

## What is in this release

### Step 1: data preparation and large data

- Load a volume, with its voxel size and axis order remembered per source. For very large files, "Prepare for large-data viewing" writes a multi-resolution OME-Zarr copy (your original is never changed) so the volume can be browsed smoothly.
- A per-channel preprocessing toolkit with a recipe you can reuse.
- An interactive 3D box tool to choose regions, an inspector with three orthogonal views and a rotatable 3D view, and a review library of every extracted patch with thumbnails and locations shown in the source volume.

### Step 2: human-in-the-loop 2D training

- Annotation of slices in the lateral and axial views, with Cellpose predictions to correct.
- Training rounds that build on each other: predicted, accepted and staged slices are tracked in separate lists, with loss curves for each round.
- Several training runs per workspace; each run keeps its own training pairs, channels and trained models, and you can switch between them.

### Step 3: reconstruction, QC, curation and model refinement

- Predict all slices and fuse them into 3D labels with u-Segment3D, patch by patch.
- More than 50 shape, intensity, neighbour and surface measurements per cell, with histogram, scatter, PCA, UMAP and clustering plots, lasso selection and tier-based triage.
- A three-view editor with an Advanced Features dock offering Reshape (cut an over-merged cell into several), Refine and Add.
- Model Refinement: train a new model from your corrected cells.
- Every section has a **?** button with a plain-language guide.

### Step 4: stack segmentation and the Workbench

- Segment whole stacks with the model you finished in Step 3, then measure and tidy the result.
- The Workbench: feature plots, PCA, UMAP and t-SNE maps, clustering, and a selection card for inspecting and marking cells.
- An Export window that writes the per-cell feature table.

### Whole-tissue engine (notebook and command line)

- A separate engine for very large images that tiles, segments, reconstructs, stitches, cleans up and measures each cell, run from the notebook in `tutorials/segmentation/` (`01_run_whole_tissue_segmentation.ipynb`) or the `epicycle-scaledseg` command.
- It uses the two models you trained in Steps 2 and 3.
- `tutorials/segmentation/` holds three notebooks: run the engine, reconcile cell and nuclei segmentations, and clean them up to a consensus.
- `tutorials/analysis/` holds five notebooks: cell-shape morphospace, neighbourhood niches, figure panels for each, and a comparison of two groups of cells.
- Every feature the engine and the plugin measure is documented, with its equation, in `docs/research/` (the reference, a plain-language guide and the catalogue table).

## Known limitations

- Windows 11 is the tested platform. Linux is experimental; macOS is untested.
- An NVIDIA GPU is needed for training, prediction and the whole-tissue engine. Without one, only the curation steps run; training and prediction may not run properly or may be very slow.
- The whole-tissue engine runs from a notebook or the command line, not from a plugin panel. It has been tested on Linux GPU clusters and on one small Windows run.
- File formats and the interface may change before 1.0.
- Some advanced annotator tools are not yet available.

## Upgrading

Where a file format changes, EpiCycle migrates your workspace when you open it. Keep a backup of any workspace you care about before opening it in a new version. Releases are tagged `vX.Y.Z` on GitHub; install a specific version by using its tag in the install commands in the README.

## How to report problems

Open an issue on GitHub. Please include:

- the console output from the moment the problem happened,
- your napari and EpiCycle versions,
- the step and the button you used.

## Licence and third-party notices

EpiCycle is free software under the GNU General Public License, version 3 or later; see `LICENSE`. The licences of the libraries it uses are in `THIRD_PARTY_NOTICES.md`.

Cellpose's pretrained models are not covered by this licence: they were trained on CC-BY-NC data, and a model you fine-tune from them may carry that restriction. For commercial use, check with the Cellpose authors or train from scratch.
