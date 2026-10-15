# Session

Session is where Step 4 starts. It shows which project folder you are working in and lists the 3D stacks you can segment. It also holds the one fact Step 4 needs about each stack to give real units: the **voxel size**, how many micrometres one pixel covers in Z, Y and X. You tick the stacks you want to work on here. The **Segment** section below runs on exactly the same ticks. The model is not chosen here: it lives in Segment, above its stack list.

Both **?** buttons in this section (Workspace and Stacks) open this guide.

---

## The workflow at a glance

    Session (pick stacks, check voxel sizes)
       →  Segment (Predict Selected, then Reconstruct Selected)
       →  Measure & tidy (measure, mark, tidy, Save, Finalize)
       →  Export… (one table of cells from every finalized stack)

The flow hint at the top of the panel says the same in one line: "Stacks → predict → reconstruct → measure & tidy → export". Under it: "Run your trained model on whole stacks, then measure the cells."

---

## Workspace

"The project folder every step reads and writes. Shared with Steps 1–3."

The box shows the folder, and **Browse** changes it. Its help line: "Choose the project folder. Every step reads and writes it."

- Step 4 keeps a record of every stack inside the project folder: its status, the model that made it, its voxel size. A project copied to another computer keeps all of it.
- **Unsaved work first.** If the stack loaded in Measure & tidy has unsaved edits or marks, Browse asks first: **Save**, **Discard** or **Stay**. Stay cancels the Browse, and the folder is not changed. After Save or Discard, the new folder opens and the stack is unloaded.
- **While a Step 4 job runs** (a Predict, a Measure, a Save…), Browse is grey. Changing the project mid-job would send the results to the wrong place.

---

## Stacks

"Stacks you prepared in Step 1 are listed here. Ticked stacks are the ones Segment runs on."

### Where the stacks come from

- **Step 1 patches, automatically.** Every patch you extracted in Step 1 appears here on its own. You do not point Step 4 at a folder for them, and a patch is never listed twice.
- **Any other folder of stacks.** **Add a folder…** adds every .tif or .tiff file in a folder you choose. Its help line: "Add every .tif stack in a folder to the list." Added folders are remembered. Only TIFF stacks are listed.

With nothing to list, the line above the list reads "No stacks yet." The list itself says "Stacks you prepared in Step 1 appear here." and "You have not prepared any yet — or add a folder of .tif stacks."

With stacks, it counts them and those with a voxel size, for example *6 stacks · 5 with voxel size · 1 in voxels*. The last part is amber: those stacks will give numbers in voxels (see *Voxel sizes* below).

### What a row shows

Each row reads, left to right:

- **the tick,** centred on the row; a ticked box fills with green. Its help line: "Ticked stacks are the ones Predict Selected and Reconstruct Selected run on; the two lists share the ticks." A tick here is also a tick in Segment, and the other way round.
- **a picture** of the middle slice (128 px square). It is made in the background, so the panel never waits for it. Until it is ready you see a plain grey square. The picture is kept on disk, so next time it appears at once. Once a stack has 3D cells, they are drawn over the picture: each cell in the colour the viewer gives it, as a soft fill with a thin outline.
- **the name,** and at the right the **status word** in its colour (see *Status words and colours*).
- **a dim second line:** the axis order (for example ZCYX), the size (Z × Y × X), the number of channels and, once it has 3D cells, how many cells.
- **a third line, the voxel size** and its source. For example: "1.00 × 0.33 × 0.33 µm (from Step 1)", "0.80 × 0.25 × 0.25 µm (from the file)" or "0.80 × 0.25 × 0.25 µm (set by you)". With no voxel size, the amber word "voxels". To set it, use the row's **⋯** menu or right-click the row and choose Set spacing…; there is no button on the row itself.
- on a measured stack, a **line of counts**: a coloured dot, a number and the word for good, bad and unmarked cells, as the stack's saved table says. They change when the table on disk changes (a Measure, a Save, a Finalize), not while you mark cells in the Workbench.
- sometimes an **amber warning line** (see *Warnings on a row*).
- an **eye**, "Show or hide this stack in the viewer.", and a **⋯** menu.

At the right of the buttons under the list, a count says how many stacks are ticked, for example "5 ticked".

### The ⋯ menu of a row

- "Set spacing…" opens the voxel-size window for that one stack.
- "Clear 2D predictions…" deletes the stack's 2D predictions, the slow part of Segment. A red confirmation asks first: "Delete the 2D predictions of Wound_patch_2.tiff? It goes back to unprocessed and needs Predict again." The stack goes back to **unprocessed**.
  - On a stack that already has 3D cells, the item is grey. Its own text says why: "Clear 2D predictions (it has a 3D result)". A menu item's tooltip is only seen with the menu open, so the reason is in the text.

### Warnings on a row

- "Channels don’t fit this model: it needs 2, this stack has 1." The model chosen in Segment was trained on more channels than this stack has. The tick is grey: the stack cannot run with this model. Pick another model in Segment, or leave the stack out.
- The amber word "voxels" on the voxel-size line (no warning line): Step 4 found no voxel size for this stack. It still runs; see *Voxel sizes* below.
- "Another stack already has this name." Two different files share a name (or differ only by .tif and .tiff). Step 4 keeps its records by file name, so only one can be used. The other is listed but cannot be ticked. Rename one of the files.
- "The stack file changed since it was predicted." The file on disk is not the one Segment predicted, for example because you trimmed the patch in Step 1. Segment will not reconstruct it until you predict it again. Step 1 also warns you, before a Trim or a Delete, that Step 4 uses the patch.

---

## Voxel sizes

A voxel size turns pixel counts into micrometres. Everything Step 4 measures (volume, surface, width, distance to neighbours) is in µm when the stack has a voxel size. Without one, it is in voxels.

Step 4 looks for the voxel size in this order, and the row says which source it used:

1. **(set by you)**: a value you typed in Set spacing…. Step 4 never replaces it.
2. **(from Step 1)**: the voxel size you gave the source in Step 1. If Step 1's preprocessing resized the source, Step 4 uses the resized size.
3. **(from the file)**: the size written inside the TIFF file itself.
4. **voxels** (amber): none of the above. Nothing is refused: the stack runs and is measured, with every number in voxels.

A voxel size of 1 × 1 × 1 counts as none. That is what a file says when nobody recorded its size.

The voxel size you set is kept in one file at the top of the project folder, `patch_spacing.json`, which Step 3 uses too. A size set for a stack in Step 3 shows here as (set by you), and one set here shows the same way in Step 3, as long as both steps point at the same TIFF file. If the two disagree when Step 4 first meets a stack, the value in that file wins and the log in Diagnostics says so.

Step 4 looks again each time it lists the stacks. A measured stack never changes its numbers behind your back, though. A new value is shown as a change, and the stack is marked out of date until you measure it again.

### Set spacing…

**Set spacing…** opens a small window for the ticked stacks, or for one stack from its row's ⋯ menu. Its help line: "Set the voxel size, the Z direction and the axis order of the ticked stacks." The window has:

- **Z, Y and X in µm**: the voxel size;
- **Z orientation**: which way up the tissue lies in the stack (basal side down or up);
- **Axes**: the order of the dimensions in the file, for example ZCYX or CZYX. Use it when Step 4 guessed a stack's order wrong. The row's axis column shows the order in use.

OK saves the values on each stack. The row then says (set by you).

When the button is grey, the line under it says why:

- "Set spacing needs at least one stack."
- "Tick a stack, or use a row’s ⋯ menu."

**Changing a voxel size can make earlier work out of date.** A new voxel size or Z orientation makes a measured stack's measurements out of date. A new axis order makes a predicted stack's 2D predictions out of date. Before OK applies such a change, a confirmation names what goes out of date. For example: "Changing this makes the measurements out of date for 2 stacks. They will need Measure again." Or: "Changing this makes the 2D predictions out of date for 3 stacks. They will need Predict again." Afterwards:

- a measured stack shows "Voxel size changed since the last Measure." in Measure & tidy, until you measure it again;
- a predicted stack's chip in Segment reads "2D made with other axes: Predict again". Reconstruct refuses it until you do.

If the voxel size could not be saved, the section says "The voxel size could not be saved. The log in Diagnostics says why."

---

## Pipeline status

"Stack counts in this run."

**Pipeline status** is a row of counts: Unprocessed · Predicted · Reconstructed · Measured · Finalized. Each number is in its status colour. A red **Failed** count appears only while some stack has failed.

The last line of the section is a pointer: "The model is chosen in Segment, above its stack list."

---

## Status words and colours

The same words and colours appear on every row: in Session, Segment, Measure & tidy and the Export window. The first three are Step 3's words and colours, because they mean the same thing there.

| Word | Colour | Dots | Meaning |
|---|---|---|---|
| **unprocessed** | grey, open circle | ○ | Nothing has been run on the stack yet. |
| **predicted** | teal (blue-green), tick | ● | The 2D predictions exist; ready for Reconstruct. |
| **reconstructed** | amber, cube | ●● | The 3D cells exist; ready to measure. |
| "segmenting..." / "reconstructing..." | grey, hourglass | none | Being worked on right now. |
| **measured** | purple, scatter plot | ●● | Every cell has been measured. |
| **finalized** | light cyan, diamond | ●●● | Saved and marked done; Export takes it. |
| **failed** | red, crossed circle | none | The last Predict or Reconstruct failed, and there is no earlier result. |

A stack that failed but still has an earlier result keeps its old word. It gets an amber line instead (see the Segment guide). A stack an older EpiCycle marked *segmented* had a Reconstruct that never finished. Its word is read from what it still has: **measured** if it has measurements, **reconstructed** if it has a 3D result, and **failed** only if it has neither.

---

## When things are grey

- **While a Step 4 job runs,** an amber line at the top of this section names the job. The same line shows in Segment and in Measure & tidy. For example: "Predicting — Cancel stops it after the current stack. The workspace, the ticks, the model, the settings, Measure & tidy and the other steps are locked." Browse, Add a folder…, Set spacing… and the ticks are grey until it ends. The ⋯ menus and the eyes are dimmed and do not open (the eye still works during a Predict or a Reconstruct, so you can look at stacks while they run). The stack list gives up a little height while the amber line shows, so Pipeline status stays on screen. A grey control shows no tooltip, which is why the reason is written on the page.
- **Set spacing…** is grey with no stacks or no ticks. Its reason is under it (above).
- **A row's tick** is grey when the stack cannot run. The row's amber line says why. A grey row you ticked earlier keeps its tick, and you can take it off; you cannot tick a grey row.
- **Clear 2D predictions** is grey on a stack with 3D cells. The item's own text says why.

---

## Where Step 4 keeps its state

Step 4 remembers your stacks, ticks, model and results in `step4_state.json` at the root of the project folder, beside `step1_state.json`, `step2_state.json` and `step3_state.json`. A project made with an earlier build kept it in `artifacts/step4/`: the first time Step 4 opens such a project it moves the file to the root and keeps the old copy there as `step4_state.json.moved-` followed by the date and time. Backups stay in `artifacts/step4/state_backup/`.

## If Step 4's saved state cannot be read

Step 4 keeps its records in one file in the project folder, and makes a backup copy at every save. The file can be damaged, for example by a crash during a save. Step 4 then does not silently start over. It sets the damaged file aside under a new name, and tells you what it did:

- "Step 4's saved state could not be read. It was kept as step4_state.json.corrupt-20260928-141502, and Step 4 started from its last backup."
- "Step 4's saved state could not be read. It was kept as step4_state.json.corrupt-20260928-141502, and Step 4 started from the older state file."
- "Step 4's saved state could not be read. It was kept as step4_state.json.corrupt-20260928-141502, and Step 4 started with no stacks."

Sometimes the damaged file cannot even be moved, because another program has it open. Step 4 then refuses to save anything rather than overwrite it: "Step 4 cannot save: its state file could not be set aside. Close other programs using it, then reopen the workspace." A save that fails for another reason says "Step 4 could not save its state. The log in Diagnostics says why."

An older copy of EpiCycle may write Step 4 results to the same project after this version took over. You then see once: "An older EpiCycle wrote Step 4 results after this one moved to its new state file. Those results are not shown here."

---

## What can go wrong

- **A stack has no voxel size.** It runs anyway, with every number in voxels. The row says "voxels", Segment says the stack runs in voxels, and the exported table marks it. To get µm, use **Set spacing…** before you measure.
- **A stack's channels do not fit the model.** Its tick is grey and its row says so. Choose a model with the right number of channels in Segment, or leave the stack out.
- **A stack failed.** Its word turns red **failed**. If it had an earlier result, an amber line says so and keeps that result. The reason is in **Diagnostics**; see the Segment guide.
- **A stack was edited after it was finalized.** It still reads **finalized**, but the Export window refuses it until you measure and finalize it again. See the Measure & tidy guide.
- **Reconstructing a stack again deletes its old cells.** Its measurements, marks, tidied outlines and exported 3D meshes go too. A confirmation asks first when the stack was measured or finalized. See the Segment guide.
- **Measuring again keeps your marks.** Good and bad marks are carried over to the same cells. See the Measure & tidy guide.
- **The stack file changed after it was predicted,** for example trimmed in Step 1. Its row warns you, and Reconstruct refuses it until you predict it again.
