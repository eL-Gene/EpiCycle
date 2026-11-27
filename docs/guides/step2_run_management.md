# Runs

A **run** is one training campaign: its own training pairs, channels, segmentation style and trained models. One workspace can hold many runs. The run row sits at the top of the Curation section:

"Run:" · "New Run" · "Manage ▾" · and the `?` that opens this guide.

---

## Why have several runs?

* **Different biological targets**: train a `nuclei` model next to your general cell model in the same workspace, each with its own data and weights.
* **Different segmentation styles**: compare a strict and a permissive style on the same patches without losing either.
* **Different channels**: one run on DAPI alone, another on DAPI plus actin.
* **Starting from another run's model**: bootstrap a new run from an existing model and, if you like, copy over some curated pairs as a seed.

Each run lives on its own on disk under `artifacts/step2/runs/{run_id}/`. Switching runs swaps the whole working context: patches' counts, slice statuses, models, training history and the loss curves. The Session section's Run summary always describes the active run.

---

## The Run picker

The box after "Run:" lists the runs in this workspace with a short summary, for example `Cell (whole cell · lat 3 · ax 2 · 83 pairs)`: the run's name, what it targets, how many lateral and axial rounds it has and how many pairs it holds. Archived runs are hidden unless you ask for them.

Switching closes the annotator, loads the new run's state, reloads the patch that run was last on and reopens the annotator on it. If the new run had no patch loaded, the annotator stays closed and you pick a patch from the list.

Before switching, the drawing you were doing is saved. If it cannot be saved (the run's folder was moved, the disk is full), the switch stops and the box says why; it offers "Switch anyway (the drawing is lost)", and Cancel is the default.

The first run of a project is different: nothing closes and the patch you have open stays, in the same view and on the same slice. What you drew before the run existed and the channel, model and training settings you set go into it, so pressing Accept again after making the run writes the slice into it. A model you choose in the New Run dialog wins over the one picked in the panel. Every later switch closes, reloads and reopens as described above.

When the project has runs but none is active, for example because the active run's folder was removed, the Run picker shows "No run is active — pick one below, or click New Run." instead of showing a run as the current one. Picking one of those runs is a normal switch: it loads that run's own patch and slices. If you have slices or a drawing on screen, you are asked first (an open patch alone switches without a question): "Your slices from before a run was active are not kept; make a New Run to keep them, or switch and lose them."

### When Step 2 cannot use the list of runs

The list of runs is the file `step2_state.json` in the project folder. In three cases Step 2 saves nothing in the project, greys out New Run and says why in the Run picker (hover it for the whole sentence) and in the Run summary:

* **The file is damaged.** Nothing was deleted: Step 2 keeps a copy beside it (`step2_state.json.corrupt-<time>`) and never writes over it. Keep the folder as it is and ask for help.
* **Old-layout Step 2 data sits beside a folder of runs.** Press Browse and pick this same folder; Step 2 offers to update it.
* **Another program held the file when the project was opened** (a backup, a sync tool or a virus scanner). Close and reopen the project, or press Browse and pick it again; Step 2 also tries again by itself at your next action.

A file that another program holds for a moment while you work is not damaged: that one save is refused, said once, and tried again at your next change.

### If a save is refused at close

When Step 2 cannot save at close, or when you leave a project with Browse and its save is refused, it writes what it could not save (the slices, the settings, the round counters) to `artifacts/step2/rescue/<time>.json` and says where. Step 2 does not read this file back by itself: open the project again, check what is missing, and keep the file to recover it by hand (or ask for help). A drawing that could not be saved when you leave a project is not kept, and the box says so.

---

## New Run

Until the project has a run, the Run picker says "No run yet — click New Run to start training." Until then Step 2 saves nothing in the project, and Predict, Accept, Finalize accepted, Train, the 2D annotator, the two imports and the training history ask you to make a run first.

If New Run cannot write the list of runs because another program holds it for a moment, nothing is made; press New Run again.

Opens a dialog with four parts:

* **Identity**: a file-system-friendly name, a display name, the target, an optional description (Cellpose is the only backend for now).
* **Initialization**, pick one: **Scratch** (random weights, slowest to settle), **Run model** (another run's trained model in this workspace), **Step 3 refined** (a Step 3 refinement bundle, usually the best-polished start), **Pretrained** (a Cellpose built-in such as cyto3 or nuclei) or **Browse** (any model file).
* **Channels**: a preview of the channels in the viewer (information only).
* **Preview**: the folder tree that will be created.

Two buttons: **Create only** (makes the run but stays on the current one; not offered for the first run of a project, which is always switched to) and **Create + Activate** (makes it and switches to it; the default).

---

## Manage ▾

A small menu with the run-level actions:

* "Manage runs…" opens the run manager: edit a run's name, target and description, archive or unarchive it, or delete it. Delete is blocked while another run was started from this one. A deleted run is **recoverable**: it is moved to `artifacts/step2/.trash/{run}-{timestamp}/`, not erased, and disappears from every list at once. If you deleted the wrong one, move that folder back under `artifacts/step2/runs/`. Nothing empties `.trash/` for you.
* A grey caption, "Bring training pairs into this run", introduces the next two.
* "Import pairs from another run…" copies accepted image and mask pairs from another run into this one, for seeding a new run with samples you already curated well. You pick the source run, round and bucket, tick the pairs to copy (all by default), choose the target bucket and round, and import. The originals are untouched. Each imported pair shows an `imported` chip in the Slices list, with the source on hover. Copying across buckets (lateral pairs into axial, or the reverse) would silently train on mis-oriented data, so the dialog blocks it unless you explicitly choose to proceed anyway.
* "Import external pairs…" does the same for image and mask pairs that were made outside EpiCycle.

Imported pairs take part in training exactly like pairs you curated yourself.

While a round is training the run row is read-only: finish or cancel the round in Training first.
