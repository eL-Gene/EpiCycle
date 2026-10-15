# QC Plot

Where you look at the labels of one patch as points, pick out the ones you are unhappy with, and sort them into tiers (curated, good, review, bad, garbage). The plot sits on the right of Curation; under it is the triage histogram, and at the bottom a row that assigns the selected labels to a tier.

Every point is one label of the patch that is loaded in the list on the left. Nothing is written to disk until you press **Commit Triage** (or **Save Curation**).

---

## The line at the top

The first line says what the plot shows, for example "842 labels · 403 selected · 837 ghosts": the labels, how many are selected, and the ghosts (counted even while the **Ghosts** toggle hides them; what the points are coloured by is in **Colour by**). On its right is the **status**: what just happened. With nothing selected it reads "Lasso or click points to select." After a lasso it tells you what the viewer shows ("Viewer: 97 labels shown alone"); after an assignment it gives a receipt ("Assigned 15 labels to review." or "Undid the last assignment."). A problem is written in gold, a failure in red ("Isolation failed — toggle Isolate off and retry.").

The **Colours in viewer** tick paints the labels in the napari viewer with the plot's colours. Untick it and the viewer goes back to napari's own label colours.

Before a patch is loaded the line reads "No patch loaded.", while one loads "Loading {patch}…", and for a patch whose features were never measured "No QC features yet." The plot itself then says "Load a patch to see its QC plot." (or "No QC features for this patch yet." and "Press Extract under QC Features.").

---

## The plot

The words above the plot choose how the labels are laid out:

| Button | What it shows |
|---|---|
| **Scatter** | Any two features against each other (pick them in **X** and **Y**). |
| **PCA** | The first two principal components of the feature set: labels of similar shape sit together. |
| **UMAP** / **t-SNE** | A 2D map of the feature set. Before one has been made the plot says "No UMAP yet." with "Press UMAP under QC Features. It takes about 45 s for 3,000 labels." (and "No t-SNE yet." for t-SNE). |
| **Violin** | The spread of one feature (pick it in **Feature**), one violin per group of labels. The groups are what **Colour by** names when it is a tier, a cluster, the border labels, the topology, or QC status when the table has it; with a measured feature (qc_score after a load) the groups are the QC tiers, and the line next to **Feature** says so, for example *area by QC tier*. On Violin the viewer keeps the Colour by colours; the legend shows the groups, with the feature's colour bar under it when Colour by is a feature. Ghosts are never a group. Click a violin, or its colour in the legend, to select its labels. If no label has a value for the feature the plot says "No values for area on this patch." with "Pick another feature next to Feature." |

Three toggles follow: **Lasso** (drag a loop to select), **Density** (contour lines where labels are packed) and **Ghosts**. On **Violin** there are no points to loop around, so **Lasso** and **Density** are greyed and the status reads "Click a violin to select its group." (a colour in the legend selects a group too); a **Lasso** you left on comes back on when you leave the violin.

**Colour by** colours the points by a tier, a cluster, the border labels, or any measured feature. With a feature, **Scale** (Linear, Log or Rank) and **Colours** appear, and a bar under the plot shows the feature's range. After a patch loads the plot is coloured by qc_score, Coolwarm, Rank.

The **legend** under the plot counts the labels of each colour. Click a colour in the legend to select those labels. Changing the plot type keeps what you selected.

### Ghosts

Ghosts are labels you deleted or replaced in earlier edits, drawn as small grey crosses behind the others. They are counted ("ghosts" in the legend) but cannot be selected. The **Ghosts** toggle hides them in the plot and in the histogram.

---

## The divider

Between the plot and the area under it is a divider. Drag it to give the plot or the histogram more room; double-click it to put it back. The plot never gets smaller than 150 px, and the area under it never smaller than its histogram needs.

---

## Histogram and Selection

The area under the plot shows one of two views. The two word-tabs **Histogram** and **Selection** switch between them, and the area also switches on its own: a lasso, a legend click or a selection kept across a plot change shows **Selection**; **✕ Clear** shows **Histogram**. A single click, a selection made in the Label Navigator or the viewer, and the histogram's Gate never switch the view — they only update the card and the count on the Selection tab.

### Histogram

**Triage by** picks the feature along the bottom; **Colour bars by** splits each bar, for example by tier. With no patch the histogram reads "No QC data yet."; a feature with nothing measured reads "No values for {feature} on this patch."

On **qc_score** three lines sort the labels into tiers (a higher qc_score is worse):

- left of the green line → **good**
- between green and orange → **review**
- between orange and red → **bad**
- right of the red line → **garbage**

A line under the histogram reminds you that "Dragging a line re-sorts every label that is not curated, hand assignments included." (or "All {n} labels are curated; the lines change nothing."). Curated labels never move. **Reset lines** puts the three lines at the quartiles of qc_score on this patch. Each drag, and each Reset lines, can be taken back with **Undo assign**.

On any other feature, tick **Gate**: a blue band appears; drag it or its edges and the labels whose values fall inside it are selected, the same way a lasso selects them.

### Selection

A card shows what is selected: "15 labels selected", a bar with the mix of tiers, the count per tier, where the selection came from ("Lasso: 15 labels.", "Click: label 42.", "Legend: review, 12 labels.", "Violin: review, 12 labels.", "Gate: 402 labels with area between 1,020 and 3,400.", "From the Label Navigator or the viewer: 3 labels.", "Kept from the last plot: 15 labels.") and the totals of the whole patch ("All labels: curated 842 · good 0 · review 0 · bad 0 · garbage 0, plus 837 ghosts"). With nothing selected it says "No labels selected. Drag a lasso, click a point, click a colour in the legend, or tick Gate under Histogram (any feature but qc_score)."

---

## Assigning a tier

The row at the bottom reads **Assign {n} selected to** (**Assign selected to** with nothing selected) [tier] **Assign** **Undo assign** **✕ Clear**, and stays there in both views.

1. Select labels (lasso, click, a legend colour, or the Gate).
2. Pick the tier in the list.
3. Press **Assign**. The status says "Assigned 15 labels to review." until you select something else or change the plot. A legend selection stays on the same labels after they change tier.

**Undo assign** takes back the last tier change: an assignment, a line drag or Reset lines ("Undid the last assignment."); after a drag the lines go back too. It only works on the patch the change was made on — loading another patch forgets it. **✕ Clear** deselects everything; the tiers are not changed.

**Assign** needs a selection, **Undo assign** something to undo, and both a loaded patch. If every selected label is being recomputed, nothing is assigned and the card says "All {n} selected labels are being recomputed; none was assigned."

---

## While an operation is running

While Step 3 is busy (loading a patch, extracting, computing a map, saving…) the histogram's controls and lines, and the row at the bottom, are greyed. The two tabs, the divider and **Colours in viewer** stay live, so you can still look.

---

## Common workflows

**Bulk garbage triage:**
1. Colour by `qc_score` (the default after a load), on PCA or UMAP.
2. Lasso the obvious garbage (tiny fragments, hollow shells).
3. Pick garbage, press **Assign**.
4. **Commit Triage** (in the left column) deletes them from the volume.

**Per-cluster review:**
1. Run **Cluster** under QC Features.
2. Colour by cluster, click a cluster's colour in the legend.
3. If the cluster is uniformly bad, pick bad and press **Assign**.

**Moving the tier lines:**
1. Histogram view, Triage by qc_score.
2. Drag a line; the points recolour as you drag.
3. Not what you wanted? Press **Undo assign**.

---

## Troubleshooting

### The plot says "No QC features for this patch yet."
The patch has no measured features. Press **Extract** under QC Features.

### A lasso selects nothing
Make sure **Lasso** is on and the loop closes around some points. The count shows on the Selection tab and on the top line.

### Assign stays grey
Nothing is selected, or Step 3 is busy (look at the status line under the busy bar).

### Tier changes are lost on reload
Tier changes live in memory until you press **Commit Triage** or **Save Curation**.

---

## Related

- **QC Features** — measures the features the plot shows
- **Curate** — Save Curation keeps tier assignments
- **Commit Triage** — deletes the garbage labels and saves the tiers
