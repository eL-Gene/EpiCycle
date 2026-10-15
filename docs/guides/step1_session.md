# Project Workspace

The workspace is the folder EpiCycle keeps this project in. Everything Step 1
produces — the image patches you extract and the metadata recording where each
one came from — is written inside it, and every later step reads back out of it.

Pick one workspace per project and stay in it. Steps 2 through 5 all look in the
same place, so pointing Step 1 somewhere new means the later steps will not find
your patches.

## Choosing a folder

Click **Browse**, pick a location, and optionally type a subfolder name to create
one on the spot. An empty folder is a perfectly good starting point — the
structure inside is created for you at your first export.

The status line underneath tells you what EpiCycle found:

| What it says | What it means |
|---|---|
| ✔ Existing workspace — N patch(es) | You are reopening a project you worked on before. Your patches are still there. |
| ✔ Directory exists — empty | A valid, fresh workspace. Nothing extracted yet. |
| ⚠ Directory does not exist | It will be created at your first export. Fine for a new project — but look twice if you expected to find existing work here, because it usually means a typo or a moved folder. |

## One workspace, shared by every step

You choose the workspace once — **in whichever step you happen to be in** —
and the others pick it up on their own the first time you open them, with
their path box already filled in. You never have to find the same folder five
times, and you do not have to come back to Step 1 to set it: coming back to a
project and pressing Browse in Step 3 works exactly as well.

When the workspace changes, Project Setup closes the project you are leaving:
the image you had loaded comes off the canvas along with its boxes and
overlays, and the Sources list is rebuilt from the new folder — so nothing on
screen ever belongs to a project you have moved away from. Sources you had only
browsed to, without registering them in the project, are dropped with it; the
new folder's registered sources are listed instead. If an extraction, a Prepare,
a whole-volume Process or a trim is still running, the change is declined
instead: a line under the path box asks you to finish or cancel that job first,
and Project Setup stays on its current folder until you do.

Under each step's title there is a line that tells you where that step is
working:

| Colour | What it says | What it means |
|---|---|---|
| Red | Choose a workspace folder to begin. | Nothing chosen yet. Nothing will load. |
| Green | Workspace: … (shared across steps) | This step is using the shared workspace. |
| Amber | Workspace: … (different from the shared workspace) | You pressed Browse in that step earlier, and the shared folder has since moved somewhere else. That step stays where you put it. Hover the line to see both folders. |
| Red | That workspace folder is no longer where it was. | The folder has been moved, renamed or deleted since you chose it. Press Browse and point at its new location. |

Underneath it a second line sometimes appears, about the step you are on:

| Where | What it says | What it means |
|---|---|---|
| Step 2 | No run yet — click New Run to start training. | The project has no training run yet. Nothing is wrong; make one when you are ready. |
| Step 2, Step 3 | This project uses the old folder layout. Press Browse and pick this same folder to update it. | The project was made by an older version. Opening a step only READS it — nothing on disk is changed until you press Browse and pick the same folder, which updates it properly. |
| Step 3, Step 4 | No trained 2D model in this workspace yet… | Step 2 has not finished a training round in this project. |

The line is always there, it updates the moment you pick a folder, and hovering
it shows the full path. If the panel is narrow the middle of the path is
shortened so the drive and the folder name stay readable, and you can select
the text with the mouse to copy it.

The panel on the left shows the same thing in short: the folder's name in bold
(hover for the full path), and one line about the step you are on — how many
patches, which run, which patch, how many stacks.

**Pressing Browse is the only thing that changes a step's folder** — and it
also makes that folder the shared one for every step that is still following
along. Opening a step never overrides a folder you chose there by hand: a step
you pointed somewhere else keeps its folder and turns amber, until you browse
it back (or the shared folder comes back to it).

## EpiCycle remembers your last workspace

When you reopen napari, Step 1's box is already filled with the folder you used
last. Nothing is read or loaded until you actually click into a step — the path
is a reminder, not an action. If the folder has been moved or renamed the box
starts empty again, and if you want a different project just press Browse.

## Where to put it

Somewhere you can find again, and somewhere with room. Extracted patches are
image data: one 3-channel patch of 256×256×60 voxels is roughly 24 MB, and a
working project usually holds dozens.

Two practical cautions:

- **Local disks are much faster than network drives.** Reading and writing image
  stacks over a network share can make loading and extraction feel broken when
  it is only slow.
- **Keep the workspace separate from your raw microscopy data.** The workspace is
  something EpiCycle writes into; your originals should stay untouched and backed
  up elsewhere. Step 1 never modifies your source files, and keeping them in a
  different folder makes that separation obvious.

## What EpiCycle keeps in the folder

Everything EpiCycle writes lives in two places: `step1_state.json` — the record
of your sources, boxes and settings — and an `artifacts` folder holding the
data itself.

Inside `artifacts/step1` you will find:

- `patches/` — the patches you extracted, plus a small `_metadata.json` beside
  each one.  This is the folder Steps 2 and 3 read from by default.
- `state_backup/` — one copy of the previous `step1_state.json`.  EpiCycle
  rotates it there before every save, so if the current file is ever damaged
  there is a recoverable predecessor.  You never need to touch it; it is kept
  out of the main folder so the project stays readable.
- `exports/` — snapshots you saved from the viewer.  It appears the first time
  you actually save one, not merely when you open the save dialog.

Older projects may still have a `step1_state.json.bak` sitting in the main
folder from before the backup moved.  It is simply ignored — not read, not
deleted.  You can remove it yourself if you like.

## Sources versus patches

Two different things, and the difference matters:

- A **source** is one of your original image stacks, registered with the project
  so EpiCycle remembers it. Registering records a reference — the file itself is
  never copied or altered.
- A **patch** is a smaller region you drew a box around and extracted from a
  source. Patches are the units the rest of the pipeline segments.

A source can sit in your project with **zero** patches, and that is a normal,
valid state — it just means you have not drawn any boxes in it yet. Adding a
source and extracting from it are separate actions.

## The session summary

The three counters below the workspace track where your boxes stand:

- **new** — drawn on the image, nothing extracted from them yet.
- **staged** — extracted during *this* session. Extract writes the patch
  straight into the workspace, so these are already files on disk; the counter
  is simply telling you what you have cut since you opened napari.
- **captured** — extracted in an *earlier* session and restored from the
  workspace when you reopened it. Also files on disk, and locked against edits,
  so a patch that Step 2 has already trained on cannot shift underneath it.

Both staged and captured patches are safely written; the split is only about
*when* they were cut, not about whether they have been saved.

## Leaving Step 1 and coming back

Moving to another step puts your image away: the picture disappears from the
canvas and the status line says so. Nothing is lost — your sources, recipe and
extracted patches are all still there, and the Review & Library list stays fully
usable — you simply click **Load** again when you want to see the
image.  Drawing boxes and extracting do need the image on screen, so those wait
for that Load and will tell you so.

## Coming back to a project

Reopening a workspace restores what was written to disk — your sources and the
patches you extracted. Boxes you drew but never extracted are **not** restored: if
you draw a set of boxes, extract nothing, and close napari, you will need to draw
them again. Extract before you stop for the day.

## Naming

Each source can carry a short, readable **alias**, and that alias becomes the
prefix on every patch extracted from it — a source aliased `wound_d3` yields
`wound_d3_patch_1`, `wound_d3_patch_2`, and so on. Pick something you will still
recognise in a figure caption months from now.

The original filename is always kept, so renaming is safe: it changes the label,
never the link back to your data. Renaming affects only *new* patches — ones you
already extracted keep the name they were written under.

## Moving a project between computers

You can copy a whole workspace to another machine — a laptop, a shared drive, a
new PC — and carry on where you left off.

**If you copy the image data too**, just browse to it in its new location and
load it as usual. EpiCycle recognises the file as one it already knows — it
compares the **filename**, the file size and the image dimensions, ignoring only
the folder the file sits in — and quietly re-attaches it to the source it was
already registered as. Your extracted patches stay linked to it, and there is
nothing to re-prepare.

The filename has to match because size and dimensions on their own are not
enough to tell two images apart: two stacks from the same microscope, same
objective, same settings routinely share both. If you also **renamed** the file,
use **Find the original file…** below — that asks you before it commits.

**If you copy only the workspace** and leave the originals behind, a source you
had already prepared still opens: EpiCycle shows you the prepared copy it keeps
inside the workspace, with an amber line saying it cannot find the original file.
In that state you can still draw boxes, cut patches and trim them — but
**Prepare** and **Process whole volume** are switched off, because both of them
have to read the original file.

### Find the original file…

When you know where the file went, use **Find the original file…** — the button
under Load, or the same action on a right-click of any source in the list.

- If the file you pick has the same image dimensions, it is re-attached silently
  and the amber line clears.
- If it has different dimensions, EpiCycle asks first, because using it anyway
  could show you the wrong pixels. **Cancel** leaves everything exactly as it
  was.
- If it is plainly a different image, it is refused with a short message.

### If a source file changes

EpiCycle notices when a source file is not the one it was prepared from and says
so in an amber line — it never interrupts you with a pop-up:

- **The file's contents changed** (you re-saved or edited it): EpiCycle reads the
  original file rather than the prepared copy, so what you see is current. Run
  **Prepare** again when convenient.
- **Only the file's date changed** (a plain copy, a cloud sync, a restore from
  backup — the contents are identical): EpiCycle keeps using the prepared copy.
  There is nothing to redo.
