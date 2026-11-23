# Curate a patch in five moves

**1. Open the patch.** In the Curation box, click your patch in the Curate Bucket list (every patch listed there can be opened), then press **Open 3D Curation Annotator**. Under **QC Features** press **Extract** and wait for the progress bar to finish.

**2. Find the worst cells.** In the plot, colour by `qc_score`. A higher score is worse, and the Coolwarm colours run blue (low) to red (high), so red is the worst end. Press **Lasso** and drag a loop round the odd points. Pick a tier in **Assign {n} selected to** and press **Assign**. Wrong? Press **Undo assign**. You can also click a row in **Label Navigator...** and press **Mark Bad** or **Mark Garbage**; **Mark Bad** does the same as **Assign** with `bad`. In the navigator, **Score (asc)** puts the worst cells last. Dragging a tier line in the histogram re-tiers every label that is not `curated`, including ones you marked by hand.

**3. Split a merged cell.** Press **Advanced Features...** and open the **Reshape** tab. Click the merged cell's row in **Label Navigator...** and press **Start** (the cell vanishes and a Refine layer appears; that is normal). Press `2` for paint mode and paint a short stroke inside each cell in the XY, XZ or YZ side views, pressing **Add marker** before the second one. Press **Preview**, tick the pieces you want and press **Accept**. **Discard** puts the cell back untouched.

**4. Fix a boundary.** Click the cell's row in the navigator and press `2`. On two or more slices in the same view, draw the correct shape: a closed outline is filled automatically when you release the stroke, so an outline or a painted area both work. In the **Refinement** section press **Preview**. It fills only the gaps between the slices you drew, and never removes or overwrites existing voxels. Then **Accept**, or **Reject** (your strokes stay), or **Clear Edits** (back to the original shape).

**5. Save.** Press **Save Curation** in the left column.

## What saves what

- **Save Curation** writes your edits and your tiers to disk.
- **Commit Triage** deletes every label tiered `garbage`, then saves. It cannot be undone on disk. `bad` labels are never deleted.
- **Accept Volume** saves, then marks the patch `accepted`. It is not a lock: opening the patch in the annotator again sets it back to `curating`. You do not need it, or **Finalize Labels**, to keep your edits.
- **Undo** (beside **Save Curation**) reverses the last accepted edit, one step at a time. It refuses while an edit is still open. There is no Ctrl+Z in Step 3.
- The **Undo** on the Reshape tab only removes the last marker.
- **Undo assign** in the plot reverses tier assignments. Any curation edit clears it.
