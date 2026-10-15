# Patches (Curation)

A **patch** is one 3D image volume (a `.tiff` stack) that came in from Step 1 (Project Setup). It is the unit you work on in this step. Session shows every patch with where it stands; this list, in Curation, is where you **pick** the patch to work on.

"Click a patch to load it. Up / Down load too."

---

## What you see in the list

Each row has:

* **A picture** of the raw patch (its middle slice, seen from above, no labels). It is the same picture whichever View you are on, and it is drawn in the background, so a new row may be blank for a moment.
* **The name**, shortened in the middle if it is long (hover for the full name and the voxel size).
* **A shape line**: axes, size in slices and pixels, and number of channels.
* **The status counts**: one icon and number per status the patch has in the View you are on (lateral or axial). Predicted is amber, curating purple, accepted green, finalized cyan, trained blue-violet. The counts follow the View switch; the picture does not.
* **The strip** along the foot of the row, the same counts as proportions.

The patch that is loaded in the viewer has an amber bar at its left edge and a faint amber glow. A patch with no slices at all in this run shows a grey "not started" in the place of the counts. A patch with slices only in the other view shows dimmed icons.

An empty list says "No patches yet." Pick a workspace in Session first.

## Loading a patch

* **Click a row** to load that patch into the napari viewer and unlock the tools beside it.
* **Up / Down** move to the neighbouring patch and load it. Loading waits a moment after the last key press, so you can hold the key down to skip across the list without loading every patch on the way.
* The loaded patch is remembered per run: switching from run A to run B reloads the patch B was last on.

Loading a patch swaps every image and label layer in the viewer, which takes a couple of seconds on a real patch. A thin strip at the foot of Curation sweeps while it happens and goes flat when the patch is ready; the window stays responsive. The label layer of a view is built the first time you look at that view, so the first switch from lateral to axial on a freshly loaded patch takes an extra moment. Nothing is lost: any masks you already have for that view are restored.

## Opening the annotator

With a patch loaded, press "Open 2D annotator" (the amber button at the foot of the right column). It opens the XY / XZ / YZ side views with the drawing tools and the slice navigator. If you press it with no patch loaded you are asked to pick one first.

## Where patches come from

Step 2 uses the patches Step 1 extracted into this workspace, and Session says so ("Using the patches from Project Setup · 15 patches."). To work from another folder, use Browse in Session's Patch folder block. See the Session guide.

## While a round is training

The Curation section shows an amber line first: "Round 3 (lateral) is training — slices and the model are read-only until it finishes." The list dims and the buttons at the foot grey out. The stage line under the foot says "Locked · round 3 (lateral) is training · cancel it in Training".

---

## What happens on disk

Selecting a patch writes nothing except a remembered last-loaded patch in the run's state file. Curating slices and training write under `artifacts/step2/runs/{run}/` in your workspace. The picture of each patch is a small cached image kept in `artifacts/step2/thumbs/patches/`; it is disposable and is redrawn whenever the patch file changes.
