# Preprocess

Preprocessing is where you clean up each channel's signal *before* it becomes
training data — denoising, evening out background, boosting local contrast, and
putting the image on an even voxel grid. You build a short **recipe** of steps,
and the same recipe is applied every time you extract a patch, so your training
data stays consistent.

> **How this panel works.** You build a recipe, **Preview** it on a small region
> to judge it, then **Process whole volume** to run it over the entire stack and
> see the result. Everything you set up is saved automatically. Which version a
> patch is cut from follows the **Raw / Processed** switch: with **Processed**
> showing, Extract takes the processed pixels; with **Raw** showing, it takes
> the original ones. The hint beside the Extract button always says which.

## One column per channel

Each channel of your image gets its own column, side by side. A two-channel stack
(say, a nuclear stain and a membrane marker) shows two columns; scroll sideways if
you have more. The coloured stripe and swatch at the top of each column match that
channel's display colour, so you always know which one you are editing.

Channels are **independent**: the steps you add to the nuclei column have nothing
to do with the steps in the membrane column. A nuclear stain and a membrane marker
usually want different cleanup, and this keeps them separate.

## Adding and arranging steps

Click **+ Add step** at the bottom of a column to open the menu of operations,
grouped by what they do — denoise, adjust contrast, remove background, correct for
depth or bleaching. Pick one and it is added to that column with sensible starting
values already filled in.

- **Tune a step** by typing in the small boxes on its chip (for example a
  Gaussian's `sigma`, in pixels).
- **Reorder** with the up/down arrows. Order matters — a background subtraction
  followed by contrast boosting gives a different result than the reverse.
- **Remove** a step with its ✕. The ✕ on a column header clears all of that
  channel's steps at once.

Some operations in the menu are shown greyed out with a **"— soon"** tag. Those
are planned but not built yet; they are listed so you can see what is coming.

## The shared resample step

One step behaves differently from the rest: **resample** (isotropic rescale). It
is drawn as a **blue chip**, and it appears in *every* column at the same
position, tagged *"linked · all channels."*

That is deliberate. Resampling changes the physical spacing of the voxels, and
every channel has to sit on the same grid — you cannot rescale one channel without
the others. So there is only ever **one** resample step, shared across all
channels. Add it, move it, or remove it in any column and the change is mirrored
everywhere.

Confocal stacks are often **anisotropic** — the spacing between Z slices is larger
than the pixel spacing within a slice. Resampling to an even (isotropic) grid
before segmentation usually helps the 3D model, which is why resample sits in the
middle of the typical recipe rather than being a separate button.

Click the **⚙** on the blue chip to configure it:

- **Mode** — *Isotropic* rescales everything to a single target spacing (in µm);
  *Per-axis* lets you set separate Z, Y, and X factors.
- **Method** — how new voxel values are interpolated. *Linear* is a safe default;
  *Nearest* is fastest but blocky; *Cubic* is smoother; *Area-average* is good for
  shrinking.

## A typical recipe

A common, well-tested order for a fluorescence channel is:

**Gaussian** (gentle denoise) → **resample** (to an isotropic grid) →
**background** (remove uneven illumination).

When a column reads this way you will see **"order = Fiji default"** in the footer
— a nod to the order this sequence is usually run in.

## Preview a region

Preview lets you judge a recipe on a real piece of your data before committing to
the whole stack.

1. **Place a region.** Click **▭ Place a region to preview** — a box drops onto
   your image at the current view centre, marking where the preview will run.
2. **Position and size it.** Use **⌖ Recenter** to snap it to where you're
   looking, the **S / M / L** buttons for quick sizes, and the **X / Y / Z**
   sliders to move it. The **Z-depth** slider sets how many slices it spans — the
   box starts spanning the full depth, and once you thin the depth the
   **Z-position** slider wakes up so you can slide the slab up and down the stack.
   The readouts show the slice index and range (e.g. `128 / 512`).
3. **Preview.** Click **↻ Preview this region**. The recipe runs on just that
   region and you get two things: a **before → after plot** of the intensities,
   and the **processed pixels** drawn back onto the image where the box sits.
   Your original layer for that channel is **hidden automatically** while the
   preview is up, so you are looking at the processed pixels and nothing else —
   the status line names which layer went away, and it comes back the moment you
   exit or preview again. Move the box or change a step and just click Preview
   again.
4. **Exit** with **✕ Exit preview** when you're done — it removes the box and the
   preview layers, brings your original layer back, and leaves a clean canvas.

Preview runs on a representative *region*, not the whole stack, so it stays fast —
pick a spot that is typical of your data (dense nuclei, a fold, a dim corner).

### Reading the before → after plot

Both curves are drawn as **voxels per million voxels**, not raw counts. That
matters when your recipe has a resize step: resizing changes how many voxels
there are, and on a raw count the processed curve would sit lower simply because
there are fewer of them — nothing to do with your image. Per-million keeps the
two curves comparable. **Log Y** (on by default) stretches the low, sparse part
of the curve so faint bright tails stay visible next to the tall pile-up.

The preview also uses the **same brightness window as the whole-volume run** —
the p0.1/p99.5 levels measured over your whole image, not over the little region.
That is what makes the preview a fair picture of the final result rather than a
region-specific one.

### When the preview says it can't show a step honestly

Under the green "Preview ready" line you may see one or more amber **⚠** lines.
They are not errors — the preview worked — they tell you where the region is too
small to stand in for the whole stack:

- **Depth-brightness** is measured across the slices it can see. If your region
  is 16 slices deep and the stack is 200, it fits a different correction here
  than it will on the whole volume (and on a single-slice region it does nothing
  at all, while the whole-volume run still applies it). Make the region deeper if
  you want to judge this step.
- **Contrast (CLAHE)** works in tiles. If the region is thinner than one tile in
  Z, it is treated as a single tile here and the result will not look like the
  whole volume's. A taller region gives a truer picture.
- One caveat is always shown, because it is always true: **the edges of the
  region and any resize step still differ on the whole volume.** A blur or
  background step has nothing beyond the box to work with, so the outer rim of
  the preview is not what those voxels will become — judge from the middle of the
  region, not its border.

## Process the whole volume

The teal **Process whole volume** button at the bottom runs your recipe over the
*entire* stack (every channel, full resolution) and shows you the processed
result in the viewer, so you can judge it at full scale — not just on the preview
region. Your original files are never touched; the result is held in memory.

While it runs, the progress bar shows the **current step** ("Removing background",
"Boosting contrast", …) and, for the steps that can show it, a **rough time
remaining**. The button turns into **Cancel** — press it to stop; the run stops
promptly and nothing is changed.

A couple of things worth knowing:

- The **contrast-boosting step (CLAHE)** runs on the whole image at once, so it is
  usually the slow part — the progress bar says so while it's working.
- For a very large image the 3D view is shown simplified so it stays smooth; the
  2D slices always show full detail, and a region **Preview** is always
  full-resolution for fine judgement.
- If an image is too large to fit in memory, the tool does **not** give up: an
  image you have run **Prepare for large-data viewing** on is processed by
  streaming it through in blocks, so memory stays bounded however big it is, and
  the result is written to disk instead of held in RAM.
- Streaming cannot do everything yet. **Contrast (CLAHE)**, **depth-brightness**
  and **resize** need the whole image at once, so on a large-data image the
  whole-volume run refuses a recipe containing them and says so rather than
  quietly dropping the step. Preview warns you about this as soon as you preview
  such a recipe, so you find out in seconds rather than at the end of a long run.
  If you need one of those steps on that image, load it the normal way (without
  Prepare) so it goes through the in-memory path.

## Switching between raw and processed

Once you have run **Process whole volume**, a **Raw / Processed** switch (in the
left column, under *Voxel size*) flips the viewer between your original image and
the processed result — one at a time, cleanly, with no blending. It shows the
processed result straight after processing; click **Raw** (or press the **\\**
key) to compare against the original, and **Processed** to go back.

The **Processed** side stays greyed out until there is a processed version for the
image you have loaded. Flipping is **instant** and never reloads the image, so it
does not stall even on very large stacks.

*(This switch will also decide which pixels get extracted — raw or processed — in
an upcoming update; for now it controls what you see.)*

## Trimming or deleting a patch that later steps already use

Trimming a patch changes how many Z slices it has, which renumbers every slice
in it; deleting one removes the image altogether. Either can pull the rug out
from under work you have already done in Step 2 or Step 3, so EpiCycle checks
first.

If the patch is already used — slices you curated in Step 2, or a reconstruction,
QC analysis or curation in Step 3 — the confirmation tells you what is using it
and the safe option (**Cancel**) is selected for you.

**Step 1 never changes or deletes anything in Steps 2 or 3.** If you go ahead:

- **After a trim**, the Step 2 slices you curated no longer line up with the
  patch, so you would redo the affected slices yourself.
- **After a delete**, the Step 2 and Step 3 work stays on disk but points at a
  patch that no longer exists. To clear it out you would delete that patch's run
  in Step 3 and its slices in Step 2 yourself.

If the patch is not used anywhere yet, you get the plain confirmation, exactly as
before.
