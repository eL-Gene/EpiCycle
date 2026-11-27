# EpiCycle release notes

## 0.1.5 (2026-11-26)

Step 3's Add page: clicking updates the cells on any region, and two slowdowns found while curating with 0.1.4 are gone.  A new project no longer falls back to the old folder layout or to made-up values: Step 2 asks for a run before it writes, Step 1 records the voxel size it read and says when Z was filled in, and Step 3 writes what it knows and nothing else.

**Step 3, the Add page (3D annotator ▸ Advanced Features ▸ Add)**

- **One rule for every region.** A whole patch and a box follow the same rules: Cells update as you click on a whole patch as on a box (up to 20 million voxels), Preview runs up to 100 million voxels on either, and the 1 and 4 million voxel limits boxes had are gone. Nuclei ▸ Anywhere in the box and Cells with Markers from Nuclei masks run on a whole patch too, and Grow by is offered there as well.
- **No refusal for a missing voxel size.** When a patch has no voxel size recorded, the Add page runs at 1 µm a side and an amber line under the first line says so, pointing to Set spacing… in the Session list. Before, Cells on a whole patch refused to run until the spacing was set, without a word about why.
- **Always update as I click.** A new tick under Advanced, off by default and not remembered between napari sessions. On, every click is drawn: where the quick fill would step aside (flat spots in the walls, Vessel ridges or Thin-faint ridges walls, Grow by Spread), the region is filled whole a moment after you stop clicking and the first line says how long it takes. Off, the page keeps its quick rules and a short grey line under Update as I click says why a click is not live and which control changes that.
- **Clicking stays fast.** After a Shift+click whose seed landed in another cell, every later click grew slower until each took over ten seconds; and once two markers sat on the same dark flat spot, every later click anywhere re-filled the whole patch. Both are fixed.
- **Stop, Undo and closing behave.** Stop drops a click that came in during a slow update and draws one placed while it winds down; unticking Update as I click, changing the preset or the markers' source cuts a slow update short; an Undo in the review refreshes what the next Preview reuses; a Preview cut off by Close box no longer holds its inputs in memory for the rest of the session.
- **Words and figures.** The wait figures under the first line come from timed runs of the right kind; the line under the tick stands down where Preview is grey; a Nuclei whole patch shows the default-spacing note when a Tissue outline is chosen.

**A new project (Steps 1 to 3)**

- **Step 2 writes nothing until you make a run.** A new project used to be written in the old folder layout (`hitl_state.json`, `training_data/`) the moment you opened a patch, and then showed a run called "default" that did not exist. Now a new project has no run until you press New Run; until then Accept, Finalize accepted, Train, Open 2D annotator, the imports and History ask you to make one, and nothing is written. Making the first run keeps the patch you have open, what you drew and your settings. A save that fails is said in a box and the work stays on screen; on close (or when you browse to another project) what could not be saved is written to `artifacts/step2/rescue/`. A project whose run list cannot be read is never overwritten and never offered a new run. A file briefly held by another program (a backup or antivirus tool) is a "try again", never a verdict on the project.
- **Step 1 records the voxel size it read.** The size read from the image file is recorded when you select the source or extract the first patch, so Steps 3 and 4 see it; before, a size that was only shown was never saved and the patches reached Step 3 with none. When the file gives no Z step (an XY-only header, or a Fiji stack saved with no slice spacing, which ImageJ records as a depth of 1), Z is filled in and the row says so in amber in Steps 3 and 4 ("Z assumed"); hover it for why, press Enter in Step 1's Z box to confirm the value as yours, or type the real Z step. A header pixel size over 50 µm (a print resolution read as a size) is shown with a warning and never recorded; type the real values.
- **Step 3 writes what it knows.** A run with no channels saved shows its model's channels instead of a stored default; an unknown voxel size is written as unknown, not as 1 µm; the reconstruction manifest records the axes the prediction used. Features measured with no voxel size are marked "voxel units" and the out-of-date notice says so when a size arrives.
- **Step 3's words.** The "no trained 2D model" line now says you can pick a model under Model ▸ Change… (hover it for the model-file route), and Step 3's run is labelled "Step 3 run" so it is not mistaken for the Step 2 run.

**Known limits.** After a Shift+click, each later click on a large patch pays about two seconds for the join of the seeds and the smoothing of every cell; a click that lands against a large cell can re-fill most of the patch once (six to eight seconds on five million voxels). The tissue outline is measured once per patch over the whole patch, so the first box on a large patch waits for it. The Add page on a whole patch at the default spacing runs at 1 µm a side; set the real spacing in the Session list for sizes in µm.

## 0.1.4 (2026-11-23)

Install fix, from a fresh install on a new NVIDIA card, and EpiCycle reopens your last workspace.

- **Install: new NVIDIA cards (RTX 50 series) now work.** The install pulled the CUDA 11.8 build of PyTorch, which does not know these cards: PyTorch printed "... with CUDA capability sm_120 is not compatible with the current PyTorch installation" and segmentation never moved. The README's three install options and `environment.yml` now pull the CUDA 12.8 build of the same PyTorch (2.7.1), which runs these cards and the older ones alike. It needs an NVIDIA driver that supports CUDA 12.8 or newer; any current driver does.
- **An existing environment is fixed with one line.** In an Anaconda Prompt, run `conda activate epicycle`, then:

  ```
  pip install --force-reinstall --no-deps torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
  ```

  There is no need to reinstall EpiCycle or anything else.
- **Still being checked:** the GPU steps of reconstruction (u-Segment3D, through its GPU library cupy) still use the CUDA 11 runtime, and we are checking them on RTX 50 series cards.
- **When you reopen napari, EpiCycle opens the folder you used last**, the same as choosing it with Browse: Step 1 lists its sources again and the other steps pick it up when you open them, but no image is loaded until you press Load. A folder that first needs the old-layout migration or a repair is not opened by itself; it is named under the top line, for you to open with Browse.

## 0.1.3 (2026-11-24)

New ways to build objects on Step 3's Add page, and fixes found while curating with 0.1.2.

**Step 3, the Add page (3D annotator ▸ Advanced Features ▸ Add)**

- **Clear objects.** One button removes the objects mostly inside the box, or every object in the patch. A red confirm first says how many will go and how many are marked Curated or Good, and one Undo brings them all back. New objects are numbered above every number the patch has used, so a cleared number is never given to a new object.
- **Nuclei inside each cell of another run.** When one run holds good cells, Look for nuclei ▸ Inside each cell of run finds the nucleus of each. A nucleus that crosses a cell border is cut there and marked "cut at edge"; two nuclei kept in one cell are marked "two in a cell". This and the other new Add paths need the patch's voxel size: at the unset 1.0 µm default they send you to Set spacing… in the Session list.
- **The whole patch.** Whole patch is now a box size. Cells on a whole patch fill to the walls from every marker at once. A Tissue outline row (the outline of the objects you cleared, a shrink-wrap around the stain, or none) keeps new cells inside the tissue, and Show the tissue outline draws its edge in the main view and the three side views. The outline now stays on screen over a result.
- **One click, one nucleus.** With Look for nuclei ▸ At the markers I place and Separate touching objects ticked (the default), a click gives the nucleus under it, not the whole clump of touching nuclei. This works for most clicks into a clump, not all; its Spacing setting is under Advanced, and unticking brings back the earlier behaviour.
- **Markers on a whole patch**, for nuclei at the markers and for Cells ▸ Click markers. Going back from a whole patch to a box with markers placed asks first.
- **Update as I click** (under Advanced, on by default): each marker you place or remove, and each setting you change, is drawn a moment later without pressing Preview. It covers nuclei in a box and on a whole patch, cells in a box of up to a million voxels, and cells on a whole patch of up to 20 million voxels. Over those limits the page says to press Preview. Accept waits until the last click is drawn.
- **Extending an object.** A click starts a new object, shown bold in the list; Shift+click adds a seed to it; Alt+click on an object, or a click on its row, makes it the bold one; Ctrl+click removes a marker; Ctrl+Z in a side view takes back the last seed. The page lists these keys while you place click markers. A seed on a separate piece of stain makes nothing, and the page says why.
- **A faster Preview.** After the first Preview of a box, each later one draws about ten times faster.

**Elsewhere**

- **Closing napari asks about unsaved work.** Before, closing the window with unsaved Step 3 curation (tier marks, mask edits, a Clear) or unsaved Step 4 edits lost them without a word. Closing now asks the same Save / Discard / Cancel questions as switching patches, and Cancel keeps the window open. An edit still open in the annotator has to be accepted or rejected first.
- **Step 3, Delete Label with several cells isolated** now removes the cell your brush is on, when it is one of the isolated cells, instead of the highlighted row's cell. The highlighted row moves to it; Undo puts it back.
- **Help:** the guides behind the `?` buttons were checked against the panels and corrected where they had fallen behind.
- **Install:** numba is now a declared requirement (the Add page's cell growth compiles with it); `requirements-lock.txt` pins the version we test (0.66.0). An environment built from `environment.yml` also sets `VISPY_PYQT5_SHARE_CONTEXT=true`, so the annotator's docks can be floated.

**Known limits.** Cells on a whole patch update as you click only with the shrink-wrap tissue outline and Membrane sheets or Intensity edges walls; otherwise that patch is drawn when you press Preview. With two or more markers in one nucleus, the live result and a single Preview of the same markers can differ by a few voxels where the markers' regions meet.

## 0.1.2 (2026-10-17)

Patch release, from the first week of people installing EpiCycle.

- **Step 3, Cancel during Predict now stops after the current slice.** Before, it only stopped after the whole patch, so on a large patch the panel looked frozen for minutes after the click and then reported "Segmentation complete". The run now ends within one slice, the patch it was working on is left unprocessed, and the summary says how many patches finished, for example "Segmentation cancelled: 1/3 patches done". Cancel during Reconstruct still stops after the current patch.
- **Step 3, model paths on a copied workspace.** A workspace copied to another computer kept the absolute paths of its chosen lateral and axial models, which no longer existed there. Step 3 then quietly picked the newest model file it could find instead. The stored paths are now rebuilt under the new workspace when the original is gone, so the model you chose is the one that runs. A copy on the same computer, where the original still exists, keeps using the original's files.
- **Tests:** three Step 2 test files imported helpers from a file that is not part of the public repository; the helpers now live in a shared module, so the public test suite runs from a clone. The CI also builds and installs the package the way users do, so a file left out of the install cannot pass again.

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
