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

---

## New Run

Opens a dialog with four parts:

* **Identity**: a file-system-friendly name, a display name, the target, an optional description (Cellpose is the only backend for now).
* **Initialization**, pick one: **Scratch** (random weights, slowest to settle), **Run model** (another run's trained model in this workspace), **Step 3 refined** (a Step 3 refinement bundle, usually the best-polished start), **Pretrained** (a Cellpose built-in such as cyto3 or nuclei) or **Browse** (any model file).
* **Channels**: a preview of the channels in the viewer (information only).
* **Preview**: the folder tree that will be created.

Two buttons: **Create only** (makes the run but stays on the current one) and **Create + Activate** (makes it and switches to it; the default).

---

## Manage ▾

A small menu with the run-level actions:

* "Manage runs…" opens the run manager: edit a run's name, target and description, archive or unarchive it, or delete it. Delete is blocked while another run was started from this one. A deleted run is **recoverable**: it is moved to `artifacts/step2/.trash/{run}-{timestamp}/`, not erased, and disappears from every list at once. If you deleted the wrong one, move that folder back under `artifacts/step2/runs/`. Nothing empties `.trash/` for you.
* A grey caption, "Bring training pairs into this run", introduces the next two.
* "Import pairs from another run…" copies accepted image and mask pairs from another run into this one, for seeding a new run with samples you already curated well. You pick the source run, round and bucket, tick the pairs to copy (all by default), choose the target bucket and round, and import. The originals are untouched. Each imported pair shows an `imported` chip in the Slices list, with the source on hover. Copying across buckets (lateral pairs into axial, or the reverse) would silently train on mis-oriented data, so the dialog blocks it unless you explicitly choose to proceed anyway.
* "Import external pairs…" does the same for image and mask pairs that were made outside EpiCycle.

Imported pairs take part in training exactly like pairs you curated yourself.

While a round is training the run row is read-only: finish or cancel the round in Training first.
