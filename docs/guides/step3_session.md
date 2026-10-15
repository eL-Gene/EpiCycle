# Session — the workspace and the patch list

**Session** has two cards and a summary.

**"Workspace"** — "The project folder every step reads and writes. Shared with Steps 1, 2 and 4."  Press **Browse** to open another project folder.

**"Patches"** — "Patches you extracted in Step 1 (Project Setup). Each row shows the patch's voxel size; you tick patches in Predict + Reconstruct."  Under that sentence one line counts the list, for example "10 patches · 3 curating · 1 finalized · 1 without a voxel size" — each part only when it is above zero, the last one in amber.  An empty list says "No patches yet." and, inside the list, "Patches you extract in Step 1 appear here." and "You have not extracted any yet — or use a folder of .tif patches."  A part that is zero is left out, so a list of four patches, two of them without a voxel size, reads 4 patches · 2 without a voxel size.

Under the list, **"Use another folder…"** points Step 3 at another folder of `.tif` patches; the faint line beside it says which folder is read: "Using the patches from Project Setup." (the patches Step 1 extracted into this workspace) or "Using {folder}." with the folder's name (hover over it for the full path).

Click a row to make that patch the active one; when its predictions are on disk the status line of **Predict + Reconstruct** says "Predictions cached for {patch}".  A click only does that — it reads no file, and it does not load the patch's features into the QC plot; that happens when you click the patch in **Curation**'s list.  You tick the patches to predict and reconstruct in **Predict + Reconstruct**, not here — these rows have no tick box.  The list's own tooltip says: "The patches Step 3 reads, with a picture of each one's middle slice. Click a patch to make it the active one."

---

## What a row shows

- **A picture of the patch** on the left (128 px square).  Once the patch is reconstructed in this run, its cells are drawn over the picture: each cell in the colour the viewer gives it (the same cell, the same colour), as a soft fill with a thin outline.  The picture itself is the same on every run; only the cells on it follow the run.  The list shows four rows at a time and scrolls past that.
- **The patch name**, and on the right its **status** — the coloured dots and word (`unprocessed`, `predicted`, `reconstructed`, `curating`, `accepted`, `finalized`).
- **A second line** with what Step 3 knows about the patch, for example `ZCYX · 81×256×256 · 2 ch · 1,650 labels`: the axis order, the size (Z × Y × X), how many channels it has, and — once it has been reconstructed — how many labels its 3D result holds.  A part that is not known is left out rather than guessed — Step 3 never opens a patch file just to fill this line.  The channel count, for example, is always shown for a one-channel patch (axes `ZYX`) and for a patch whose axes Step 3 first read in this version; a patch whose axes were read by an earlier version shows no channel count (its picture still shows grey or colour).
- **A third line with the patch's voxel size** — the size of one voxel in µm, Z × Y × X, and where it came from: "0.33 × 0.33 × 1.00 µm (from Step 1)", "0.33 × 0.33 × 1.00 µm (from the file)" or "0.33 × 0.33 × 1.00 µm (set by you)".  A patch with no voxel size shows the amber word "voxels": its measurements are in voxels, not µm — to set it, use **"Set spacing…"** on the row's **⋯** menu or when you right-click the row; there is no button on the row itself.
- **A line of tier counts** once the patch's features have been extracted and triaged: a coloured dot, the number of labels and the tier — first the labels you marked curated, then good, review, bad, garbage.  Hover over it: "Labels per tier in this patch's saved features: curated, good, review, bad, garbage. They change when the features on disk change."  The counts are what the saved features file says — they change when an extraction finishes, when you press **Commit Triage** or **Save**, when the folder is read, the run is switched or you come back to this step, not while you drag the tier lines in the plot.  The same line is on the patch's row in **Curation**'s list.  When the row is too narrow the last tiers are left out ("…").
- **An eye** on the right, on a patch that has been reconstructed: "View this patch's reconstruction in the viewer (the same as View in Predict + Reconstruct)."  If you have unsaved curation edits, Step 3 asks about them first, because viewing closes the curation.  While another job holds Step 3 (a save, an extraction, a curation step…) the eye is dimmed and a click says why on a line under the list (and on the status line of **Predict + Reconstruct**); while Predict or Reconstruct runs it works, as View does.
- **A ⋯ menu** on the right with "Set spacing…" (and, on a missing patch, "Forget this patch…").

---

## The voxel size

Every patch has its own voxel size, and everything in Step 3 that measures in µm uses it: the cell width in **Predict + Reconstruct**, the features **Extract** measures in **Curation**, and the Reshape, Refine and Add tools.  It is found in this order:

1. **The value you set** with **Set spacing…** — it always wins and is never replaced.
2. **Step 1's value** — the voxel size of the source the patch was cut from (set in Step 1's Load & Adjust).  Step 3 looks again every time the folder is read, the run is switched or you come back to this step, so a value you set in Step 1 later shows up here.
3. **The patch file's own tags** — read in the background when the patch is predicted or its features are extracted, never just to fill the row.
4. **None** — the amber "voxels".  Hover over the row to read why: "Step 1's patch files carry no voxel size of their own. Set it on the source in Step 1, or here with Set spacing…."  For such a patch, **QC Settings**' spacing (in **Curation**) is used as the fallback when there is one; else the measurements are in voxels.

The voxel size belongs to the workspace, not to a run: it is the same in every run.  It is kept in one file at the top of your project folder, `patch_spacing.json`, which Step 4 reads and writes too: a size you set here shows in Step 4 as (set by you), and a size you set there shows here the same way.  If a patch's features were extracted at another voxel size than the one it has now, its row says so after the voxel size, in amber (for example `· measured at 1.00 × 0.25 × 0.25 µm`), and so does its row in **Curation**: extract them again to bring them up to date.

### Set spacing…

Press the **Set spacing…** button on an amber row to set that patch's voxel size.  For several patches at once, highlight their rows and pick **Set spacing…** in the **⋯** menu of one of them (or right-click it).  The window names the patches it will change, asks for Z, Y and X in µm and starts from the first highlighted patch's value.  **OK** gives every highlighted patch that value, marked **(set by you)**.

If any of those patches already has extracted features measured with another voxel size, Step 3 asks first: "Changing this makes the measured features of {name} out of date. Extract them again to update them."  The features keep their old numbers until you extract them again; the patch you are curating shows the usual out-of-date notice straight away.

While Predict or Reconstruct runs, the row's button and the menu item are refused, and the line under the list says "Locked while Predict or Reconstruct runs; set it when the run ends."

---

## The picture

The picture is **the middle slice of the raw patch image** — the middle Z plane of the file in the folder, never the prediction or the labels.  The status word is what tells you how far the patch has come; the picture tells you which patch it is.

Its colours:

- **One channel** — grey.
- **Two channels** — channel 0 in **magenta** and channel 1 in **green**.  By EpiCycle's convention channel 0 holds the nuclei and channel 1 the membrane, so nuclei look magenta and cell outlines green.
- **A third channel** adds **blue**; channels after the third are not drawn.

Each channel is stretched to its own brightness range for the picture, so a dim channel still shows.  The colour follows the channel's position in the file, not what it contains.

Hover over the picture to read it in words: "The middle slice of the raw patch: channel 0 magenta, channel 1 green, a third channel blue; one channel grey."

**While it is being drawn** the row shows a dark square, and the picture's tooltip says "The picture of this patch is being drawn."  Pictures are drawn in the background, one after another — the panel stays responsive while they arrive.

**When no picture can be drawn** the dark square stays and the tooltip says why, for example "No picture: The patch file could not be read." or "No picture: The patch's axes do not fit the file."  Step 3 does not try that file again in this session unless it changes.

The same picture appears on the patch's row in **Curation**'s list.

### Where the pictures are kept

Step 3 keeps the pictures in `artifacts/step3/thumbs` inside the workspace (one small PNG per patch, shared by every run), so the next time the pictures come back quickly.  A picture is drawn again when its patch file changes, and the pictures of patches that are no longer in the folder are removed when the folder is read.  Deleting the folder is safe: the pictures are drawn again.  A workspace whose Step 3 state file is from an older version is only read, never written, until you press **Browse**; its pictures go to a temporary folder in the meantime.

---

## A patch whose file is gone

If a patch file disappears after the folder was read, its row keeps its place (until the folder is read again) with an amber line, "The patch file is missing.", and its **⋯** offers "Forget this patch…" — the same as on the Patches list of **Predict + Reconstruct** (see that guide).

---

## While Predict or Reconstruct runs

The rows dim while a prediction or reconstruction runs, like the Patches list in **Predict + Reconstruct**; you can still click them, and the eye still views.  **Set spacing…** is refused (its item in the **⋯** menu greys); nothing else in Session greys.

---

## Pipeline status

Under the list: "Pipeline status" — "Patch counts in this run." — how many patches of the active run are unprocessed, predicted, reconstructed, curating, accepted and finalized, then the last time anything changed and **Resume**, which jumps to the patch you were curating.
