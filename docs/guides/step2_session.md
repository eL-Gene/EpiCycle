# Session

The first section of Step 2. It answers three questions: which project am I in, where do my patches come from, and where does each patch stand? Nothing here changes your data. You pick the patch to curate later, in the Curation section.

---

## Workspace

"The project folder every step reads and writes. Shared with Steps 1, 3 and 4."

This is the same project folder you chose in Step 1. Pick it once and every step uses it. A new workspace starts with one run called default; you can add more runs in Curation.

## Patch folder

"Where the patch list and the 2D annotator read patches from."

You normally do not need to touch this. Step 2 uses the patches Step 1 extracted into your workspace and says so under the box:

* "Using the patches from Project Setup · 15 patches."

Until a workspace is picked the box is greyed out, with its reason on screen: "Pick a workspace first; then Step 1's patches are used unless you Browse."

**Browse** points Step 2 at another folder of patch `.tiff` files, and EpiCycle remembers that choice. **Refresh** rescans the folder, which is useful if you just extracted a patch in Step 1. If a folder you picked has gone (the project was copied to another computer, a drive was unplugged) EpiCycle falls back to the Project Setup patches and tells you so; your choice is kept and used again when the folder comes back.

---

## Patches

"Patches you extracted in Step 1 (Project Setup), with where each one stands. You pick the patch to curate in Curation."

This list is for looking, not for clicking: nothing here selects or loads a patch. Above it, one summary line counts them all, for example "15 patches · 4 in a trained round · 2 finalized · 8 in progress · 1 not started".

Each row is one patch:

* **The picture** is the middle slice of the raw patch, seen from above (XY), with no labels on it. Two channels are drawn magenta and green (a third is blue); a single channel is drawn in a warm single-colour scale. It is a static picture of the patch: it does not change when you switch the View in Curation, and the same picture is used in every run. It is drawn in the background the first time, so a row can show an empty tile for a moment; hover over it to see why if it stays empty.
* **The name** and, under it, the shape: axes, size in slices and pixels, and number of channels, then how much of the patch you have curated, for example "2 of 124 lateral slices curated".
* **The voxel size**, for example "0.52 × 0.52 × 0.52 µm (from Step 1)". When no size is known the row says "voxels" in amber, and measurements made later are in voxels rather than microns.
* **The status counts**: one small icon and number per status that this patch has, in the run you are working in. The icons and colours are the same everywhere in Step 2:

| Status | Colour and icon | Meaning |
|---|---|---|
| predicted | amber, stacked layers | Cellpose has drawn a mask; you have not looked yet |
| curating | purple, pencil | you started editing the mask |
| accepted | green, tick in a circle | you accepted the mask |
| finalized | cyan, diamond | accepted and queued for the next Train |
| trained | blue-violet, brain | already part of a round model |

A patch with no slices in this run says "no records yet" in place of the counts.

* **Chips** such as `lat 2` and `ax 5`: how many lateral and axial slices of this patch have records. A chip such as `round_00` appears on a patch that has a slice in a trained round.
* **The strip** along the bottom of the row shows the same counts as proportions of colour, grey for the rest.

Round numbers are shown as they are on disk: the very first round of a run is `round_00`.

If you have not picked a workspace yet the list says "No patches yet." and "Pick a workspace, or Browse to a patch folder."

---

## Run summary

"The active run: what the next round writes and what it trains on."

A small table for the run you are working in, one column for each bucket (Lateral (XY) and Axial (XZ/YZ)):

| Row | What it tells you |
|---|---|
| Last round | the most recent round that was trained |
| Next round | the number the next Train will write |
| Pairs already trained on | how many of the run's image and mask pairs earlier rounds have already used (the run line above counts all its pairs). It is counted for the view you are on; the other view shows a dash, and its tooltip says so |
| Last model | the model that last round produced |

A dash means nothing has been trained for that bucket yet. Under the table: "Switch or create runs with the run row at the top of Curation. Slice counts by status are in Curation → Curation progress; the GPU is in Diagnostics."

Before a workspace is picked the card says "No workspace yet." Pick a workspace above and it fills.

---

## While a round is training

An amber line appears first in the section: "Round 3 (lateral) is training — the workspace and the patch list are read-only until it finishes." The fields and the list dim, and Browse and Refresh are disabled, so nothing can move under the training run. The lock lifts when the round finishes or you cancel it in Training.
