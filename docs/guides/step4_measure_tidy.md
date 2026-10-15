# Measure & tidy

Measure & tidy is where a reconstructed stack becomes a table of cells you can trust. You load one stack at a time and **measure** every cell: size, shape, brightness and neighbours, in real units when the stack has a voxel size. You look at the cells in the **Workbench** and mark the bad ones. You **remove** them, and **smooth** the outlines if you want. Then you **finish** the stack: Save writes your work to disk, and Finalize also marks the stack done. When your stacks are finalized, **Export…** writes one table of every cell from all of them, ready for your notebooks.

The **?** buttons in this section (Stacks, Measure, Review, Tidy and Fix by hand) open this guide.

---

## The workflow at a glance

    click a stack   →  Measure cells…  →  Open Workbench: mark good / bad
                                                  ↓
                              Remove bad cells  ·  Smooth outlines   (Undo)
                                                  ↓
                                       Save  →  Finalize
                                                  ↓
                              Export…  (every finalized stack, one table)

---

## The layout

The section has two columns, like Segment. The **left** column is the stack list, with a card for the loaded stack at its foot. The **right** column holds four blocks, Measure · Review · Tidy · Fix by hand, and a closed **Advanced**. It scrolls on its own. Pinned under it are **Finish this stack** (Save · Finalize), the progress bar with **Export…** at its right, and a one-line status line.

---

## Stacks

"Click a stack to load it. Only reconstructed stacks are listed."

- Every stack with 3D cells on disk is listed, with its status word, size, cell count and “made by” chip (see the Segment guide). A stack whose last run failed but which kept an earlier result is listed too.
- There are no ticks here: this list is for loading. Each row has the stack's picture (80 px, with its cells over it), its size, its voxel size and a line of counts (a dot, a number and the word) for good, bad and unmarked cells from the saved table.
- **Click a row to load that stack.** Its help line: "Load this stack to measure and tidy it; if the loaded stack has unsaved edits you are asked first."
- Loading reads the stack in the background, so the panel stays responsive. Meanwhile an amber line sits at the top of the section: "Loading WholeScale_patch_2.tiff. Wait for it to finish. The stack list, the edits, Save, Finalize, Export and the other steps are locked."
- **If the loaded stack has unsaved edits or marks,** a window titled "Unsaved edits" says "WholeScale_patch_2.tiff has edits that are not saved." It offers **Save**, **Discard** or **Stay**. Stay keeps your stack loaded. If you choose Save and the save fails, the new stack is not loaded either.
- The loaded row carries a green bar on its left edge and a faint green glow, apart from the highlight of the row you clicked, so you can tell which stack is in the viewer. While it has unsaved work, it carries the amber line "Edited, not saved".
- **Painting cells by hand counts as unsaved work.** After a brush stroke on the labels in the viewer the row reads "Edited, not saved" and the card reads "Unsaved edits · measurements stale", so loading another stack, a Browse or leaving Step 4 asks first. While a stack loads, a Measure or a Smooth runs, the labels cannot be painted: a stroke then would be thrown away when the job lands.
- **Leaving Step 4 with unsaved work** asks whether to save now, with **Save & Leave** or **Leave Without Saving**. If the save fails, you stay in Step 4 and a window says so, for example: "Saving WholeScale_patch_2.tiff failed, so you are still in Step 4. Your edits are kept; the log in Diagnostics says why." The status line keeps the short form: "Saving WholeScale_patch_2.tiff failed; still in Step 4."
- Under the list, a dim line names what is missing. For example "Wound_patch_1.tiff is not reconstructed yet; it appears here after Reconstruct." or "2 stacks are not reconstructed yet; they appear here after Reconstruct."
- If a stack cannot be read: "WholeScale_patch_2.tiff could not be loaded. The log in Diagnostics says why."

### The stack card

At the foot of the left column, a small card describes the loaded stack:

- **line 1:** its name;
- **line 2:** a pill telling you the state of your work:
  - "Unsaved edits · measurements stale" (amber): cells were removed or smoothed and not saved. The measurements no longer describe them.
  - "Unsaved edits" (amber): the last Measure took the border cells out of the stack and has not been saved yet. The measurements describe the cells as they are now.
  - "Unsaved marks" (amber): only good or bad marks changed. Nothing needs measuring again.
  - "Measurements stale" (amber): the cells or the voxel size changed since the last Measure.
  - "Measuring…" (grey): a Measure is running.
  - "Saved" (grey): nothing is waiting to be saved.
- **line 3, dim:** size, cells and units. For example "212×1024×1024 · 4,616 cells · µm 1.00 × 0.33 × 0.33 (from Step 1)". Without a voxel size: "160×768×768 · 2,540 cells · in voxels", the last words amber.

If the voxel size changed in Session since the last Measure, the card also says "Voxel size changed since the last Measure."

With nothing loaded, the card reads "No stack loaded."

---

## Measure

"Measure every cell — size, shape, brightness, neighbours — in real units."

The chip at the right of the title says when the stack was last measured, and how many cells there were:

- "14:02 · 4,653 cells";
- "stale · 14:02 · 4,653 cells" in amber, when the cells or the voxel size changed since;
- "not measured".

The big button reads "Measure cells…" the first time and "Measure again…" after. Its help line: "Measure every cell of the loaded stack. The Workbench opens when it is done."

- It opens the Extract Features window, where you choose which groups of measurements to make. The measuring then runs in the background.
- Measurements use the stack's voxel size: µm, or voxels when the stack has none. The voxel size used is recorded with them. So the exported table always states the units the numbers were made in.
- Cells cut by the edge of the stack are set aside as border cells. The export leaves them out.
- **Measuring again keeps your marks.** Every good or bad mark is carried over to the same cell. Cells that no longer exist drop out, and new cells start unmarked. When it ends the status line says, for example, "Measured 4,616 cells of WholeScale_patch_2.tiff." followed by how many marks were kept, for example "1,180 marks kept."
- **Measuring a finalized stack** asks first: "WholeScale_patch_1.tiff is finalized. Measuring again takes it out of Export until you Finalize it again." Afterwards it reads **measured** again, until you finalize it.

**Open Workbench**: "Open the Workbench to look at the cells and mark them good or bad." See *The Workbench* below.

**What is measured?**: "List every measurement and what it means." It opens a window with the same title, listing each measurement with its meaning.

### While Measure runs

- An amber line sits at the top of Session, Segment and this section: "Measuring — wait for it to finish. The stack list, the edits, Save, Finalize, Export and the other steps are locked."
- The stack list cannot be clicked: loading another stack would swap it out from under the measurement.
- Everything in the right column is grey except **What is measured?** and the four **?** buttons, which still open this guide.
- If the Export window is open, its buttons are grey too, and its status line shows the same sentence as the running job (an export's own line is "Exporting… the window’s buttons wait until it ends.").
- The bar shows the progress, for example "Measuring · 62%".
- The status line gives the time left, worked out from the progress so far, for example "about 1 min left".
- A Measure cannot be cancelled in this release. It usually takes minutes.
- If it fails: "Measuring WholeScale_patch_2.tiff failed. The log in Diagnostics says why."

---

## Review

"Cells you mark good or bad in the Workbench are counted here."

The chip counts the marks, for example "1,204 good · 0 bad · 3,412 unmarked": good in green, bad in red, unmarked in grey. Step 4 has only two marks, good and bad. Every other cell is **unmarked**.

**Remove bad cells**: "Remove the cells marked bad. Undo can bring them back until you Save."

- A red confirmation asks first, for example "Remove 12 bad cells from WholeScale_patch_2.tiff? Undo can bring them back until you Save."
- Afterwards the status line says, for example, "Removed 12 cells (4,604 left, not saved). Save keeps it."
- Removing cells clears the maps in the Workbench (PCA, UMAP and t-SNE), because those were made from the old cells.

**Undo**: "Take back the last Remove or Smooth, until you Save. The maps are not brought back: compute them again."

- Undo goes back **one** step: the last Remove bad cells or Smooth outlines, whichever came last. A second edit replaces the first.
- If you painted cells by hand after a Smooth, Undo would throw your painting away, so it refuses: "Undo is off: cells were painted by hand after the smoothing."
- It then says, for example, "Undid the last Remove bad cells."
- The Undo slot is emptied by Save, Measure, Reset, loading another stack and changing the project folder.
- With nothing to undo, pressing it says "Nothing to undo."
- Marks have their own Undo, in the Workbench.

---

## Tidy

"Round off jagged outlines. A cell too thin to survive is kept as it was."

The chip says when the stack was last smoothed, and how many cells were too thin to smooth. For example "14:12 · 3 thin kept", or "not smoothed".

**Strength**: Light, Medium (the default) or Strong. Its help line: "How strongly to round off the outlines." Light is half of Medium, and Strong is double. If you change the numbers under Advanced by hand, the box reads Custom.

**Smooth outlines**: "Smooth every cell’s outline. A cell too thin to survive is kept as it was."

- It runs in the background on a copy of the cells, with the same method as Step 3's Batch Refine. The result is swapped in when it is done.
- The status line then says, for example, "Smoothed 4,616 cells (18,402 voxels changed, not saved). Save keeps them."
- **Thin cells are kept, not deleted.** A cell so thin that smoothing would make it vanish is left exactly as it was. The chip counts it.
- **Outlines differ slightly from older versions of EpiCycle.** Smoothing now works in a wider region around each cell. It also starts every cell from the same picture. The log says so on every run. A stack smoothed with an older version will not match voxel for voxel.
- While it runs, the amber line reads, for example, "Smoothing WholeScale_patch_2.tiff. Wait for it to finish. The stack list, the edits, Save, Finalize, Export and the other steps are locked."
- If it fails: "Smoothing WholeScale_patch_2.tiff failed. Nothing was changed. The log in Diagnostics says why."

---

## Fix by hand

The chip reads "after the release". The line under it reads "Opens the 3D curator on this stack. After the release." The button **Fix by hand in 3D…** is always grey in this release. Its help line: "Fix cells by hand in 3D. This comes after the release."

---

## Advanced (closed)

The hint beside it reads "smoothing in µm · Reset".

- **The two smoothing distances**: "How far the smoothing reaches, up and down / across." One is along Z (up and down), one within each slice (across).
  - They are in µm when the stack has a voxel size, and in voxels when not.
  - Medium is 3.0 voxels up and down and 1.5 across. Changing them turns the strength box to Custom.
- **"Keep each cell in one piece"** (a tick): "Keep each cell in one piece: if smoothing would split a cell, it is reconnected."
- **Channels to measure**: "Tick the channels whose brightness is measured; the names head the brightness columns of the table." One row per channel of the loaded stack, each with a tick labelled Measure and a name you can edit. Measure reads them.
- **Reset** (red): "Put back the cells Reconstruct made. Your edits, measurements and exported 3D meshes are deleted."
  - A red confirmation asks first, for example "Reset WholeScale_patch_2.tiff to the cells Reconstruct made? Your edits, measurements and exported 3D meshes for it are deleted."
  - Reset puts back the cells exactly as Reconstruct made them. The status line then says, for example, "Reset WholeScale_patch_2.tiff to the cells Reconstruct made (4,653 cells)."
  - It **deletes** everything made from the edited cells: the measurements, your good and bad marks, the neighbour table, the saved maps, the border cells, the smoothed outlines and the stack's exported 3D meshes folder.
  - The stack goes back to **reconstructed**, and the Undo slot is emptied.
  - Reset needs the copy of the original cells that Step 4 keeps the first time a stack is loaded. Without one, the status line says, for example, "There is no original result to go back to for WholeScale_patch_2.tiff."
  - If it fails: "Resetting WholeScale_patch_2.tiff failed. Nothing was changed. The log in Diagnostics says why."

---

## Finish this stack

"Finalize saves this stack and marks it done, so Export takes it. It locks nothing."

**Save**: "Write this stack’s cells and measurements to disk; after this, Undo cannot go back further."

- Save asks first: "Write WholeScale_patch_2.tiff’s cells, measurements, marks and maps to disk?" On a finalized stack it adds " This takes it out of Export until it is finalized again."
- When it is done: "Saved WholeScale_patch_2.tiff."
- Save writes to a temporary file first, then replaces the old one. A crash during a save never leaves a broken file.
- While it runs: "Saving WholeScale_patch_2.tiff. Wait for it to finish. The stack list, the edits, Save, Finalize, Export and the other steps are locked."
- If it fails: "Saving WholeScale_patch_2.tiff failed. Nothing was overwritten. The log in Diagnostics says why."

**Finalize** saves the stack and marks it **finalized**, so the Export window takes it. It asks first: "Save WholeScale_patch_2.tiff and mark it finalized, so Export takes it?" and then says "Finalized WholeScale_patch_2.tiff." You can still edit it later; see *Editing after Finalize* below.

**When Save or Finalize is grey,** the page says why:

- Save, with nothing loaded: the stack card says "Load a stack to save it."
- Save, with nothing to save: the stack card's pill says "Saved".
- Finalize, never measured: the amber line under the two buttons says "Measure this stack first."
- Finalize, measurements out of date: "Save and re-measure first." The cells or the voxel size changed since the last Measure. The measurements would not describe the cells you are finalizing.

Unsaved **marks** never grey Finalize: Finalize saves them.

### Editing after Finalize

Finalize locks nothing. You can still remove, smooth or reset a finalized stack. It keeps the word **finalized**, but its Measure chip turns stale. **The Export window then refuses it** until you measure and finalize it again, because its measurements no longer describe its cells. Step 4 knows by comparing the stack's files with the ones it saw at Finalize.

### The bar and the status line

The progress bar under Finish this stack carries Measure, Smooth, Save and both exports. After a job it reads "Done". **Export…** sits at its right. The status line under it shows the section's last message, cut to one line.

---

## The Export window

**Export…** opens it. Its help line: "Opens the export window: features from every finalized stack, 3D meshes for the loaded stack."

- Export… works whether or not a stack is loaded. It is grey only while a Step 4 job runs.
- The window stays open beside the panel while you work. **Close** hides it.

Its title is "Export — the end of Step 4". Top to bottom it reads:

- "Workspace-wide: works on every finalized stack, whichever stack is loaded."
- a count of the stacks that can go, for example "3 of 5 stacks can go into Export features:"
- a list of the reconstructed stacks, one line each: tick, status, name, and, at the right in amber, why a stack cannot go. The tick's help line: "Finalized stacks that go into Export features; a stack that cannot go says why."

In that list:

- Finalized stacks are ticked. You can untick one to leave it out of **this** export only. Next time the window opens, it is ticked again.
- "finalize it first": measured, not finalized. The row is grey and cannot be ticked.
- "measure and finalize it first": reconstructed, not measured. Grey, and cannot be ticked.
- "edited after Finalize: measure and finalize it again": its cells or voxel size changed after Finalize (see *Editing after Finalize*). Grey, and cannot be ticked.
- "in voxels, not µm": a stack with no voxel size. It still goes and stays ticked. Its rows in the table say voxels.
- "changed after Finalize: finalize it again": only its measurements file changed after Finalize (for example marks saved since). Grey until you finalize it again.
- "its unsaved edits are not in the table": the loaded stack has edits you have not saved. It goes as it was finalized.
- "finalized before its files were recorded": finalized with an older EpiCycle, and its files are unchanged since. It goes; finalize it again to be sure.
- "measured before units were recorded: measure and finalize it again": measured with an older EpiCycle, so the table cannot say its units. It goes, and its rows say unknown in **meta_units**.
- If the list changes while the window is open, it updates itself. If you press Export features just as a ticked stack stops qualifying, nothing is written and the status line says "The list changed; check the ticks and export again."

### Export features

The button reads, for example, "Export features (3 stacks)". Its help line: "Write one table of every cell from the finalized stacks, ready for your notebooks." Under it: "One table of every active cell; removed and border cells are left out, and a column names the stack and the model."

- It first opens a save window, so you choose where the table goes. It suggests the project's artifacts\step4\exports\ folder.
- The table holds each stack as it was when you finalized it. Unsaved edits of the loaded stack are not in it; a Save after Finalize takes the stack out of Export until you measure and finalize it again.
- If only some measurements exist on every stack, you are asked which ones to keep.
- The file is offered in artifacts\step4\exports\, named with the date and the number of stacks, for example features_2026-09-29_3stacks.csv, so a new export never overwrites an earlier one. You can change the name or the folder.
- When it ends, the status line says, for example, "Exported 3 stacks (11,512 cells) to features_2026-09-29_3stacks.csv".
- The window then shows the last export, for example "Last exported 15:31 · 3 stacks · features_2026-09-29_3stacks.csv". The full path is in the log, and in the status line's tooltip.
- The log also says how many removed or border cells were left out.

The table has one row per cell and one column per measurement, plus these columns:

- the stack each cell comes from;
- **meta_model**: the model that made the stack's cells (empty for a stack made before models were recorded);
- **meta_units**: um, voxels, or unknown (measured before units were recorded);
- **meta_spacing_z**, **meta_spacing_y**, **meta_spacing_x**: the voxel size the measurements were made with (empty for a stack in voxels).

The **meta** columns describe the cells; they are not measurements. Leave them out when you choose numeric columns for an analysis, such as a PCA or a UMAP in your notebooks. Stacks in µm and stacks in voxels can share one table. The **meta_units** column tells them apart: do not compare their numbers directly.

When the button is grey, the window's status line says why:

- "Finalize a stack first." (no stack is finalized);
- "Tick a stack to export." (you unticked every stack);
- "Exporting… the window’s buttons wait until it ends." (an export is running).

If it fails: "The feature export failed. The log in Diagnostics says why."

### Export 3D meshes…

"Save each cell of the loaded stack as a 3D surface file for rendering or printing."

- It works on the stack loaded in Measure & tidy, named beside the button, for example "the loaded stack, WholeScale_patch_2.tiff".
- With nothing loaded, the button is grey and the line reads "Load a stack in Measure & tidy first."
- "Saves first if there are unsaved edits, then writes one surface file per cell to artifacts\step4\meshes\WholeScale_patch_2.tiff\." The folder is named after the stack's file.
- With unsaved edits, a window says so first: "WholeScale_patch_2.tiff has edits that are not saved. The meshes are made from the saved cells, so it is saved first."
- If that save fails, no meshes are written. Meshes are never made from old cells.
- When it ends: "Exported 4,616 meshes to WholeScale_patch_2.tiff/". If it fails: "The mesh export failed. The log in Diagnostics says why."
- A cell smaller than 10 voxels is too small to have a surface and gets no mesh. The line then counts them, for example "Exported 693 meshes to Wound_patch_1.tiff/ (8 cells under 10 voxels: no mesh)", and the log in Diagnostics lists their numbers.
- Each mesh is a plain text .obj file (UTF-8, one vertex or face per line), which any 3D viewer or Python library can open.

While either export runs, the window's buttons are grey, but Close stays live. The amber line at the top of the panel's sections reads "Exporting. Wait for it to finish. The stack list, the edits, Save, Finalize, Export and the other steps are locked."

---

## The Workbench

**Open Workbench** opens the EDA Workbench, a separate dock beside the panel. It also opens by itself when a Measure finishes. There you look at the measured cells on scatter plots and on maps such as PCA, UMAP and t-SNE.

- When the Workbench is already open, or was hidden when you left Step 4, **Open Workbench** only brings it to the front: your ticked features, maps and colours stay exactly as they are, and nothing is computed again.
- If you closed the Workbench with the X on its title bar, **Open Workbench** builds it again from what was last saved for the stack (ticked features, PCA, UMAP, t-SNE). Maps and ticks changed since the last Save are not kept after that X; when something was lost, the Workbench's message line says “Reopened as last saved; maps made since the last Save were not kept.” A map that was still running when you closed it keeps running, and its progress shows again.
- When you come back to Step 4, the labels in the viewer keep the Workbench's colours as long as **Colours in viewer** is ticked.
- With **Colours in viewer** ticked, the labels in the viewer always show the same colours as the plot, for every choice of Colour by, Scale and Colours. Cells you removed are not coloured. Untick it to keep napari's own label colours.
- **Colour plot by this**, in a feature's detail panel, colours the plot (and the viewer, when **Colours in viewer** is ticked) by that feature. Press it again to go back to the colouring you had before.

- You select groups of cells with a lasso, a click, a legend colour or a feature gate.
- You **mark them good or bad** with the buttons, or with the keys G and B.
- Marks show at once in the Review chip, and in the stack card's “Unsaved marks”. **Save** keeps them.
- Marking never clears a map. You can keep marking while a UMAP or t-SNE is computed.
- The Workbench has its own Undo for marks. Its history is cleared by Remove bad cells, Save, Measure, Reset and loading another stack.
- The Workbench has no guide of its own yet; one comes with its next update.

While a Measure, a Smooth, a Save, a load or an export runs, the Workbench's mark buttons are grey. Its message line says, for example, "A Step 4 job is running (Measure). Marking and maps wait until it ends."

---

## Status words

| Word | Colour | Meaning here |
|---|---|---|
| **reconstructed** | amber ●● | 3D cells exist; not measured yet. |
| **measured** | purple ●● | Measured; can be finalized. |
| **finalized** | light cyan ●●● | Saved and marked done; Export takes it. |
| **failed** | red, crossed circle | Its last run failed. |

---

## When things are grey

A grey control shows no tooltip, so every reason is written on the page:

| Control | Grey when | Where the reason is |
|---|---|---|
| **Measure cells…** | no stack is loaded | the stack card: "No stack loaded." |
| **Open Workbench** | the stack is not measured | the Measure chip: "not measured" |
| **Remove bad cells** | no cell is marked bad | the Review chip (0 bad) |
| **Undo** | never, outside a running job | pressing it with nothing to undo says "Nothing to undo." |
| **Smooth outlines**, Strength | no stack is loaded | the stack card: "No stack loaded." |
| **Save** | no stack is loaded | the stack card: "Load a stack to save it." |
| **Save** | nothing to save | the stack card's pill: "Saved" |
| **Finalize** | not measured, or measurements out of date | the amber line under the buttons: "Measure this stack first." or "Save and re-measure first." |
| **Fix by hand in 3D…** | always, in this release | the chip "after the release" and the line under it |
| **Reset** | there is no original copy to go back to | the status line |
| **Export…** | only while a Step 4 job runs | the amber line at the top |
| **Export features** | nothing finalized / nothing ticked / an export runs | the window's status line |
| **Export 3D meshes…** | nothing loaded / an export runs | the line beside it / the window's status line |

**While a Step 4 job runs** (in Segment or here), one amber line names it. The same line sits at the top of Session, Segment and this section, and says what is locked. The other steps of EpiCycle are locked too until it ends.

---

## What can go wrong

- **A stack has no voxel size.** It is measured anyway, with every number in voxels. The stack card says in voxels, and so do the smoothing distances. In the Export window the stack says "in voxels, not µm", and its rows say voxels in **meta_units**. For µm, set the voxel size in Session (**Set spacing…**) and measure again.
- **A stack's channels do not fit the model.** It never reaches this section, because Segment cannot run it. See the Segment guide.
- **A stack failed.** If it kept an earlier result, it is still listed here with that result. Otherwise it appears once it is reconstructed. The reason is in Diagnostics.
- **A stack was edited after it was finalized.** It keeps the word finalized, but the Export window refuses it with "edited after Finalize: measure and finalize it again". Measure it again, then Finalize.
- **Reconstructing again deletes.** Reconstructing a stack in Segment deletes its cells, measurements, marks, tidied outlines and exported 3D meshes. A confirmation asks first when it was measured or finalized. Export what you need first.
- **Measuring again keeps your marks.** Marks are carried over to the same cells. Only cells that no longer exist lose theirs.
- **Undo after Save.** Save empties the Undo slot. After that, Reset is the only way back, and it goes all the way back to the cells Reconstruct made.
- **Unsaved work and a stack or workspace change.** Loading another stack or changing the project folder asks you to Save, Discard or Stay. Closing napari does not ask, so save before you quit.

**Every measured feature.** The equation of each feature and what an unusual value means are in `docs/research/QC_FEATURE_REFERENCE.md` in the repository, with a plain-language version in `docs/research/QC_FEATURES_BIOLOGIST_GUIDE.md`.
