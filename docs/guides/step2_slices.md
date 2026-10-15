# Slices

The Slices list shows the individual 2D slices of the patch you have loaded, in the View you are on (lateral = XY, axial = XZ and YZ). This is where you triage single training samples. Its header names the loaded patch and the view, for example `WholeScale_patch_4 · lat`.

---

## What a row shows

* **A picture of the actual slice with its labels drawn on it** (a coloured fill and an outline for each cell). Lateral slices are shown from above (XY); axial slices are the side-on XZ or YZ cut, which is wide and flat, so the picture is a strip rather than a square. The picture is the real image and mask saved for that slice, not a preview of the patch, so it shows exactly what training will see.
* **The slice name**, for example `XY slice 31` or `XZ slice 140`.
* **A status icon and the status word** on the right, in the status colour.
* **Chips**: `round_00` style on a slice that is in a trained round, and `imported · Nuclei r2` style on a pair copied in from another run or an external folder.

**When the pictures refresh.** A picture is drawn in the background the first time a row appears and is then kept, so scrolling and re-filtering do not redraw it. It is drawn again when the slice's mask changes. While you are painting in the annotator the pictures of slices you are still editing wait until you pause for a second or two, so they do not flicker at every stroke. If a picture cannot be drawn the tile stays empty and a hover tells you why; EpiCycle never shows a wrong picture.

---

## Status flow

```
predicted → curating → accepted → finalized → trained
```

| Status | Colour and icon | Meaning |
|---|---|---|
| **predicted** | amber, stacked layers | Cellpose has drawn a mask for this slice. Waiting for your review. |
| **curating** | purple, pencil | You started editing the mask but have not accepted it. |
| **accepted** | green, tick in a circle | You clicked Accept. The image and mask pair is saved. |
| **finalized** | cyan, diamond | You clicked Finalize accepted. The slice waits for the next Train. |
| **trained** | blue-violet, brain | The slice is already part of a round model. |

The list is ordered by how far along a slice is: trained first, then finalized, accepted, curating and predicted last.

---

## Filter chips

The row of chips above the list controls what you see. Each chip carries its count, for example All 7, Predicted 3, Curating 1, Accepted 1, Finalized 2.

* **All** shows everything.
* **Predicted / Curating / Accepted / Finalized** can be combined. Finalized covers the trained slices too.

An empty view says "No slices in this view yet." and "Open the 2D annotator and Predict, or draw one."

---

## Navigating

* **Click a row** to jump to that slice in the loaded patch. The viewer switches to its view and scrolls to its index.
* **Shift / Ctrl click** selects several rows for the buttons below.

---

## The buttons at the foot of the right column

They sit at the foot of the right-hand column, always in view, and they act on the highlighted slices:

* **Open 2D annotator** is the one amber button; it opens the side-view annotator for the loaded patch.
* **Accept** keeps the selected slices' masks and saves each pair.
* **Finalize accepted** queues the accepted slices of the current view for the next round.
* **Discard** drops the prediction for the selected slices.
* **Revert** takes selected finalized slices back to accepted. A slice that is already trained asks before it is reopened.

Below the buttons a stage line says what will happen, for example "Ready · no slices in this view yet", and a thin bar sweeps while a patch loads.

While a round is training the list dims, the buttons grey out, and the stage line says "Locked · round 3 (lateral) is training · cancel it in Training".
