# Refining the shape of one object

This tool fixes the shape of an object that is already there: it smooths a
staircased surface, slides an outline onto the edge the image shows, fills a
hollow core, grows into a gap. It works on **one object at a time** —
whichever object is selected in the 3D view or the Label Navigator — and **no
other object ever loses a voxel**.

Nothing is written to disk here. The change lives in the session until you
press **Save curation**, and one **Undo** takes it back.

---

## The quick version

1. Select the object.
2. Choose a **Method**.
3. If the method reads a channel, pick the **Channel** — and tick any
   preparation steps you need.
4. Press **Preview**. You see the result on every slice.
5. **Accept** keeps it. **Discard** puts the object back as it was.

You can run several methods in a row on the same object — smooth it, then fill
its holes, then settle it onto the wall — and accept the lot at once. See
[Running several steps in a row](#running-several-steps-in-a-row).

---

## Which method for which problem

| What is wrong with the object | Try |
|---|---|
| A staircase from being drawn slice by slice | **Smooth** |
| The outline sits inside or outside a wall you can see in the image | **Settle** (fast), or **Active contour → Snap to edges** |
| The inside is dim and only the shell is bright, so there is no clean edge | **Active contour → Inside vs outside brightness** |
| Uniformly a little too big or too small | **Dilate/Erode** |
| An open top or bottom the reconstruction left a gap in | **Guided Grow** (needs the probability map) |
| A hollow core, or holes after interpolation | **Fill Holes** |
| One pocket of empty space inside or against the object | **Flood Fill** |
| It is two objects glued together | Not this tab — the **Reshape** tab cuts |

---

## The panel, top to bottom

- Above everything, one sentence with a chevron: *Fixes the shape of the
  object you have selected — one object at a time. No other object ever loses
  a voxel, and nothing is final until you press Accept.* Click it and a
  short explanation of the seven methods opens underneath. ("Final", not
  "written": while a result is staged it is already in the volume — Accept
  keeps it, Discard takes it out again.)
- **Status** — one line with a coloured dot: which object, what has been done
  to it so far, and how many steps are staged.
- **Object** — *Selected* shows the object this will act on and how big it is.
  There is no Start button and no dropdown: the tool follows your selection.
- **Method** — the *Method* picker, a sentence about the chosen method, and
  only that method's settings. Hover over a method in the open list to read
  what it does before you pick it. For the methods that read an image, the
  *Channel* picker and **Prepare the channel first** sit here too.
- **Preview** — pinned just above the two buttons at the bottom, so it is
  reachable however long the settings are. While Active contour or Settle
  runs, **Stop** appears beside it, and underneath are a progress bar and one
  grey line saying what the tool is doing now. **Recipes**, at the end of the
  same row, saves and runs saved chains (see *Saving a chain as a recipe*).
  The strip never changes height.
- **Result** — appears once something is staged: what changed, in the words of
  the method that did it, the steps so far, any warning, and **Show what the
  tool sees**.
- **Advanced** — the settings you rarely need, at the very bottom.
- **Accept / Discard** — the two buttons at the bottom, which never move. With
  nothing staged the second one reads **Cancel**.

---

## The seven methods

Four of the seven never look at the image: **Smooth**, **Dilate/Erode**,
**Fill Holes** and **Flood Fill** only change the shape. **Guided Grow** reads
the reconstruction's probability map, not a channel. **Active contour** and
**Settle** read the channel you pick — except when their *Wall type* is one of
the shape types or the probability map, which need no channel at all (the
channel picker then disappears).

### Smooth (signed distance)

Rounds the object off. It turns the object into a distance field — a
picture of how far each voxel is from the outline — blurs that, and cuts it
again at the halfway line. This is the one that takes the
slice-by-slice staircase out of a hand-drawn object.

- **Blur through depth** — how far the shape is blurred from slice to slice.
  Raise it to take out a staircase.
- **Blur within a slice** — how far it is blurred inside each slice. Raise it
  to round off jagged edges; lower it to keep fine detail.

Both are counted in **voxels, not microns**. The grey line under them says
what they come to on this patch — on coarse depth one voxel can be several
microns, so a small number is a long way. A big enough blur on a thin object
can remove it completely (see
[When a step removes the whole object](#when-a-step-removes-the-whole-object)).

- **Keep the object in one piece** — if the blur would cut the object apart, a
  thin core along its middle is kept to hold it together. Leave it on.
  **Either way only the biggest piece is kept afterwards** — so an object that
  was already in several pieces before you pressed Preview loses all but the
  biggest one, whatever this says.

### Active contour

Slides the outline onto what the image shows. It has two **Mode**s:

- **Snap to edges** moves the outline until it sits on a wall. It reads the
  wall picture made from the *Wall type* you choose (see
  [Wall types](#wall-types)).
- **Inside vs outside brightness** needs no visible edge. It compares a band
  just inside the outline with a band just outside, and gives each voxel in
  that band to the side it looks more like. Use it when the inside of the
  object is fainter than its shell. It reads the prepared channel directly, so
  *Wall type* is hidden in this mode — and *Local contrast* is offered.

The settings, in the order they appear:

- **Orientation** — *3D* moves the whole outline at once. *XY*, *XZ* or *YZ*
  **trace some slices in that plane and fill in the rest**, which is far
  quicker (see [Slice by slice](#slice-by-slice)).
- **Slices** (a plane orientation only) — which slices are traced: *A share of
  the slices* (the **Share**, 25 % by default), *A fixed number of slices*
  (**Slices traced**, 12 by default), or *Every slice (slowest)*.
- **Edge may move** — a hard limit, in **microns**, in both directions: no
  point of the outline ends up further than this from where it started, out
  or in. If the outline reaches it, the result block says so.
- **Steps** — how many steps the outline takes. More lets it travel further,
  up to the limit.
- **Push outward** (Snap to edges only) — pushes the outline out (+) or pulls
  it in (−) where there is no wall. The grey line under it is **live**: it
  says where the push will act, or, in amber, **"No effect at this setting"**
  when the push is not bigger than the *Edge gate* — before you press Preview.
  At 0 the outline follows the walls alone.
- **Weight inside** / **Weight outside** (Inside vs outside brightness only) —
  raise *Weight inside* to make the object **smaller** (a voxel has to look
  more like the inside to stay in); raise *Weight outside* to make it
  **bigger**. Both start at 1.

A plane orientation takes a few seconds on an ordinary object; *3D* takes ten
seconds or more, because every step moves the whole outline in every direction
at once.

#### Slice by slice

With *XY*, *XZ* or *YZ*, the tool traces the chosen slices one at a time, then
fills in the slices between them the way interpolation does, and joins the
result into one smooth 3D shape. The first and last slice the object is on are
always traced, so the ends cannot grow a cap. At most 60 slices are traced.

While it runs the bar counts: *"Plane 4 of 11 — about 1 s left"*. Afterwards
the result block says what was traced and what was filled in, because those
are not the same evidence: *"Traced 11 of 44 slices and filled in the other
33."* A traced slice that comes out **empty** is not taken as the shape — the
object's own slice stands in for it — and the line counts it apart, so the
three numbers add up to the object's slices: *"Traced 9 of 52 slices; 4 more
came out empty and were kept as they were; the other 39 were filled in."*

### Dilate/Erode

Grows the object (+) or shrinks it (−) by a fixed **Radius**, in voxels.
Growing only ever fills empty voxels, never a neighbour's. Shrinking can remove
a thin object completely. The grey line under it says what the radius comes to
in microns on this patch (in *XY* it gives only the across-a-slice number,
because XY never moves through depth).

**Orientation**: *3D* works in every direction; *XY*, *XZ* or *YZ* works
within that plane only — each slice is changed on its own.

### Guided Grow

Grows the object outward into empty voxels the reconstruction's probability
map says belong to an object.

- **Max radius** — how many voxels it may grow out from the outline, one step
  at a time. At most 15, which is how much room the tool looks at around the
  object.
- **Confidence needed** — how sure the reconstruction must have been about a
  voxel before it may be taken, from 0 to 1 across the range seen around this
  object. Lower takes more.
- **Orientation** — *3D* grows in every direction; a plane grows within that
  plane only, each slice on its own.

It needs the probability map. **On a patch without one, Guided Grow is greyed
in the method list before you ever click it**, and its own text says why:
*"Guided Grow (needs a probability map)"*. It never falls back to anything
else.

### Fill Holes

Closes cavities inside the object. **Orientation** decides what counts as a
hole: *3D* fills only a cavity sealed on every side; *XY*, *XZ* or *YZ* fills
any hole sealed within one slice — even one that is open above or below — so a
plane fills **more**, not less.

### Flood Fill

Gives the object a gap of empty space under the crosshair. Put the crosshair
in the gap first — **on or next to this object** — using the three views, then
press Preview. *3D* takes the whole gap; a plane takes only the part of it on
the crosshair's slice.

It fills a gap **in this object**, never open background, and refuses in plain
words when the click is somewhere else:

- the empty space at the crosshair does not touch the object — *"The empty
  space at the crosshair does not touch this object."*;
- it runs on past the region around the object, i.e. it is open background
  rather than a gap — *"... it is open background, not a gap in it."*;
- the crosshair is outside the region around the object altogether — *"Put
  the crosshair on or next to this object."*

**Max size** is the second check, a safety limit in voxels: a gap bigger than
this is refused too.

### Settle onto the wall signal

Moves the outline onto the nearest wall, and no further than you allow.

- **Wall type** — what counts as a wall (see [Wall types](#wall-types)).
- **Solve with** — how hard the new edge is (see
  [Solve with](#solve-with--how-the-new-edge-is-placed)).
- **Edge may move** — how far the outline may move, in **microns**, in or out.
- **Steps** — how long the diffused edge is allowed to spread. The other two
  ignore it.

Two things about Settle are worth knowing before you use it:

- **Neighbouring objects take part.** The gap between two objects is divided
  on the wall signal rather than swallowed by whichever object is settling —
  and **no other object ever loses a voxel**.
- **An edge that already touches a neighbour does not move.** Settle only
  moves edges that face background. The result block leads with the share of
  this object's border that does — *"38 % of this object's border faces
  background; the rest touches neighbours and was left alone."* — because that
  number decides whether the tool can do anything at all, and you cannot guess
  it by looking. When the outline stopped on a wall rather than at the limit,
  it says that too.
- **Use a wall type from the channel, or the probability map.** The two shape
  types (*Pinch points*, *Between the spines*) are for splitting. Their
  picture is built from the object alone — Pinch points is bright everywhere
  outside the object — so with the default *Sharp edge* the outline simply
  grows to the limit (+75 % on a round object, measured). The tool does not
  forbid it, but an amber line under *Wall type* says so while one is chosen:
  *"Shape-based walls are for splitting. For Settle, pick a channel wall type
  or the probability map, or with the default solver the outline will grow to
  the limit."*

### Solve with — how the new edge is placed

Settle floods out from three places at once — the core of this object, the
core of every neighbour, and the background — and the new edge is where they
meet on the wall picture. The three solvers differ in how that meeting is
decided:

- **Sharp edge** (the default) puts the edge on the highest wall between them.
  Fast, and the one to start with.
- **Soft edge** walks probabilities out from the same places, so the edge eases
  onto a faint wall instead of snapping to the nearest strong one. It ignores
  *Balance*.
- **Diffused edge** starts from the sharp edge, then lets the labels spread for
  *Steps* rounds, held to where they started by *Hold the outline*. It needs
  the u-Segment3D package; without it the item is greyed and the reason is
  written under the picker.

---

## Wall types

Active contour's *Snap to edges* and Settle read the same **Wall type** picker,
under the channel. Each one makes a picture that is bright where the tool
thinks a wall is. Hover over a type in the open list for a sentence about it.

- **Membrane sheets** (the default) — local contrast, blur, then a filter that
  picks out thin bright sheets.
- **Membrane sheets plus shape** — mostly the membrane sheets, with the
  object's own shape as a tie-breaker.
- **Intensity edges** — bright where the brightness changes fastest.
- **Raw brightness** — the channel after local contrast, with no ridge filter.
- **Vessel ridges** and **Thin-faint ridges** — local contrast, then one of
  two other ridge filters, for rounded walls and for weak, thin walls.
- **Pinch points (shape)** and **Between the spines** — the object's own
  shape, not the image. No channel is needed. Meant for splitting on the
  Reshape tab; under Settle they let the outline grow to the limit (see
  [Settle](#settle-onto-the-wall-signal)).
- **Probability map (from the reconstruction)** — the reconstruction's own
  map, bright where it was least sure this was one object. No channel is
  needed; the channel picker is replaced by a line saying so.

---

## One channel, prepared once

The whole tab reads **one** channel picker. It appears only while the chosen
method will actually read a channel, and is hidden otherwise — asking for a
channel a method will not read is no better than guessing one.

**Prepare the channel first** holds the same five steps the Reshape tab uses:
**Local contrast**, **Even out depth**, **Remove haze**, **Lift dim signal**
and **Despeckle**. They always run in the same order, whichever you tick, and
the chevron's title lists what is ticked.

**Local contrast** is offered — and ticked by default — only for *Inside vs
outside brightness*. Whenever the tool is looking for walls (Snap to edges,
Settle) it is hidden, and the card says *"Local contrast is not used when the
tool is looking for walls."*: the wall step applies its own, and a second one
would flatten the very contrast it looks for.

If the layer named in the picker is gone (renamed, removed) the picker goes
**blank and stays blank**, and **Preview is greyed with the reason**
(*"Pick a channel first."*). It never quietly works on the first image it can
find.

---

## How far things may move

Every **new** number on this tab is in **microns**: both *Edge may move*
settings, Active contour's *Blur* and *Smooth the join*. The numbers that were
already here — Smooth's two blurs and the Dilate/Erode radius — stay in
**voxels**, because changing them would change every result you have had so
far, and a grey line says what they come to in microns on this patch. Guided
Grow's *Max radius* is in voxels too.

If the patch has no voxel size the panel can work out, the micron read-outs
are **not shown at all** and the result block says *"Voxel size unknown"*. A
micron number worked out from a guess is worse than none.

---

## While it runs, and Stop

**Active contour and Settle run in the background**, so the window stays alive
and the 3D view still turns while they work. The grey line under the bar says
what they are doing — *"Building the wall picture..."*, *"Sliding the
outline..."*, *"Plane 4 of 11 — about 1 s left"*, *"Settling the outline onto
the wall..."*. You cannot draw in the side views until it finishes: the
object's region is copied when the run starts, so anything painted in between
would be overwritten when the answer lands. The other five methods take a
blink.

**Stop** stops, and puts the object back as it was before the first step. It
is honest about how quickly that happens:

- A plane orientation of Active contour stops **between two slices** —
  almost at once.
- A *3D* run and Settle **cannot be interrupted** once they have started. The
  bar keeps going and the line reads *"Stopping..."* until the run really
  ends; Preview stays greyed for that time, so a second run cannot start on
  top of the first. If you close the annotator during one, the window can
  pause for up to three seconds while the tool lets go of it.

Either way the object is not changed. **Stop in the middle of several steps
puts the object back to how it was before the first step**, not just before
the last one — the status line reads *"Stopped. The object is unchanged."*

---

## Running several steps in a row

**Any method can follow any other**, in any order, with any settings. After a
Preview, change the method (or the settings) and press **Preview again**: the
new step works on the staged result. The result card shows the steps so far —
*"Smooth -> Dilate/Erode -> Fill holes"* — the status line says *"3 steps
staged"*, and Accept reads **Accept (3)**. Accept writes them all; **Discard
puts the object back as it was before the first step.**

Changing the method while a result is staged does **not** throw it away:
Accept stays live for it, and Preview stays live to add the next step. If a
later step needs more room around the object than the first one set aside (a
wider *Edge may move*, say), the region simply grows — Discard still puts the
whole larger region back exactly.

Preview says no in only four cases:

- **a run is still going** — *"Still working - wait, or press Stop."*;
- **the chosen method needs a channel and none is picked** — Preview is greyed
  with the reason;
- **a different object is selected** — *"Chaining needs the same object.
  Accept or Discard first."*;
- **no object is selected at all** — Preview is greyed and the status line
  says *"Select an object first."* (with nothing staged).

A step that changes nothing, or that cannot run (Guided Grow on a patch with no
probability map, say), leaves the earlier steps staged and says so.

---

## Saving a chain as a recipe

When a chain works well on one object, keep it. With the steps staged, open
**Recipes** (the button at the end of the Preview row) and choose **Save the 3
staged steps as…**, then give it a name of up to 40 characters. Recipes are
kept in the workspace's Step 3 folder (`refine_chain_recipes.json`), for every
patch and every run. **A recipe remembers the methods and their settings — not
the object, the patch or the channel.** Batch Refine can run these recipes on
many objects at once (see *Running it on many objects*, below); its own
**Saved steps** list is a different file, and the two never mix. The name box suggests the chain itself (*"Smooth -> Dilate/Erode ->
Fill Holes"*). A name that is already taken — capital letters or not — asks
first: *"Replace the saved recipe 'tidy' (Smooth -> Settle) with these 3
steps?"* with **Replace** and **Cancel**.

To use one, select another object and pick the recipe from **Recipes**. It
presses Preview for you, one step after another, and stops with the steps
staged — **Accept** and **Discard** work as for a chain you built by hand. If
steps are already staged on that object, you are asked first: *"Add 'tidy' (3
steps) after the 2 steps already staged on object 149? Discard would then undo
all 5."* with **Add** and **Cancel**.

- **The channel is the one picked on this page.** If a step needs one and none
  is picked, the recipe is greyed in the menu with the reason, and it does not
  start.
- While it runs, **Preview, Accept and Discard are greyed** and the line under
  Preview counts the steps and names the recipe — *"2/3 · Dilate/Erode —
  'tidy'"*. The status line says *"Running a recipe (3 steps) on object
  149."* and then what each step is doing. **Stop** stays available for the
  whole run, between steps too.
- **If a step moves nothing or cannot run, the recipe stops there.** The steps
  before it stay staged, the status line keeps that step's own sentence, and
  the line under Preview says where it stopped — *"Stopped at 2/3 · Fill
  Holes — 'tidy'"*. A step that leaves nothing of the object stops the recipe too,
  with the result staged and **Delete object N** on the button, exactly as a
  step you ran by hand would.
- **Stop puts the object back as it was before the first staged step** —
  including steps that were staged before you ran the recipe.
- Settings in microns mean the same microns on every patch; settings in
  voxels mean the same voxels. The **voxel size** is how many microns one
  voxel spans along each axis — the patch's physical scale, set in QC
  Settings. A recipe saved on a patch with a voxel size does not run on one
  without (and the other way round): its micron settings would give a
  different shape. The status line says which, and what to do.
- **Flood Fill cannot go in a recipe**: it fills from where the crosshair is.
  With a Flood Fill step staged, Save is greyed and says so.
- **Load into the panel** sets the controls to the recipe's first step without
  running anything. **Delete** removes a recipe for everyone using this
  workspace, after asking (**Delete** or **Keep it**).
- Every greyed line in the menu says why in its own text — a recipe that needs
  a probability map this patch does not have, u-Segment3D that is not
  installed, a different object staged, a run already going.
- If the recipes file cannot be read, the menu says *"The recipes file could
  not be read."* and Save is refused, so the file is left exactly as it is for
  you to look at. To start over, move or rename
  `artifacts/step3/refine_chain_recipes.json` in the workspace folder.

---

## Running it on many objects

The Recipes menu runs a recipe on one object. To run it on many objects at
once — every object you lassoed in the QC plot, say — use **Batch Refine**:
the **⚙ Batch Refine…** button in the Label Navigator's side bar, or
**Refine all ▸** on the Add tab after adding cells.

1. **Pick the recipe.** At the top of the RECIPE section, choose it under
   **Refine-tab recipe** — every recipe saved on this tab is listed with its
   steps (*"tidy — Smooth -> Settle"*). The steps then appear one per line in
   this page's own words (*"2. Settle onto the wall signal — walls: Membrane
   signal; preparation: Subtract background"*). The step builder underneath
   is hidden, and **Recent**, **Saved steps** and **Save recipe…** are greyed:
   *"The steps are the Refine recipe's. Pick “none” above to build steps
   here."* A recipe that cannot be read is greyed in the list, and its text
   says so.
2. **Pick the channel**, if a step reads one. A recipe does not store its
   channel, so a **Channel** row appears. It starts on the channel picked on
   this page when that layer is in the list; otherwise it is blank, and Run
   stays greyed until you pick one.
3. **Pick the objects** (TARGET):
   - **All active (N)** — every object in the patch.
   - **Selected (N)** — the objects selected in the navigator. Opened from
     Add's **Refine all ▸**, this one reads **Added this session (N)** and is
     already ticked.
   - **Plot selection (N)** — the points the QC plot shows highlighted
     **right now**: a lasso, a click, a category. It follows the plot, and a
     click on a navigator row repaints the plot too, so one navigator click
     makes it *"Plot selection (1)"*. It works with the navigator closed.
     Opened from the navigator with points highlighted in the plot, it starts
     ticked. With nothing highlighted, the line under the targets says
     *"→ Lasso or click points in the QC plot first"*; on a patch whose QC
     features were never extracted, *"→ No QC plot for this patch yet —
     extract QC features first."*; and when the plot shows a different patch,
     *"→ The QC plot shows another patch — open this patch's plot first."*
   - **By tier** — every object in the tiers you tick.
4. **Preview on object N** runs the recipe on the one object selected in the
   viewer or the navigator (with none selected it reads *"Preview on one
   object"* and is greyed: *"Select one object in the viewer or the navigator
   to preview on."*). **Run on N objects** runs it on the whole target. Either
   way the result is staged: **Accept** keeps all of it as ONE step the main
   Undo takes back; **Reject** puts every object back exactly as it was.

While it runs, the status line says what it is doing: *"Preparing — copying
the labels and finding 40 objects…"* (the bar just moves), then *"Refining
object 3 of 40 · about 50 s left"* (the time left appears once two objects
are done and two seconds have passed), then *"Writing 40 objects into the
volume…"*. A staged result says *"Review the result, then Accept or
Reject."*

**Each object is refined exactly as this page would refine it on its own** —
the same region around it, grown the same way from step to step — so a
recipe that gives a good shape here gives the same shape in a batch. Every
object is worked out against the patch as it was before the batch started;
where two new shapes want the same voxel, the object with the lower number
keeps it.

### What a batch refuses, and why

The reason is written in amber under the recipe rows, and the buttons it
refuses are greyed. Nothing is copied and nothing starts.

- **A Settle step using Diffused edge** — *"Step 2 (Settle) uses Diffused
  edge, which needs too much memory to run on many objects at once. Save a
  copy of the recipe with Sharp edge or Soft edge on the Refine tab."*
- **An Active contour in 3D on more than 20 objects** — *"Step 1 (Active
  contour) works in 3D, about 15–25 s per object — at most 20 objects in one
  batch. Pick fewer, or save the recipe with Orientation XY."* Only Run is
  refused (the line starts *"Run: "*); **Preview** on one object still works.
- **A recipe too large for memory** — a batch may use up to 8 GB, and works on
  at most two objects at a time: *"The largest object here would need about
  … GB (2 at a time) — more than a batch may use (8.0 GB). Pick smaller
  objects, or save the recipe with a smaller Edge may move."* Ordinary
  recipes come nowhere near it; a very large *Edge may move* on a fine voxel
  size can.
- **A probability map or a channel that is not this patch's size** (a map
  made before a downsample, say) — *"The probability map does not match this
  patch's size (it may be from before a downsample), so steps that read it
  cannot run here."* / *"The channel does not match this patch's size, so
  steps that read it cannot run here. Pick another channel."*
- The same things the Recipes menu refuses: a step that needs a channel when
  none is picked, a step that needs a probability map the patch does not
  have, and a recipe saved with a voxel size on a patch without one (or the
  other way round).

### How each object ends

- **A step that moves nothing is skipped, and the next step runs.** This is
  the one place a batch differs from this page on purpose: a *Fill Holes →
  Smooth* recipe still smooths the objects that had no holes.
- **A step that cannot run stops that object, and it keeps the steps before
  it** — as on this page. The summary says *"2 objects stopped part-way: step
  2 (Active contour) could not run on them, so they keep the steps before it
  (details in the console): 8, 22"*; at step 1, *"… left unchanged: step 1
  (…) could not run on them"*. The console has each object's own reason.
- **Settle next to neighbours on every side.** Settle moves an object's edge
  into background, so an object with no background left around it has
  nothing to settle into. It counts as stopped — its earlier steps are kept —
  and the summary says so plainly: *"3 objects already touching their
  neighbours everywhere, so Settle had nothing to move; their earlier steps
  were kept: …"*
- **A step that would leave an object with no voxels keeps none of its
  steps**: *"2 objects unchanged — a step would have removed all of their
  voxels, so none of their steps were kept: 14, 57"*. A batch never deletes
  an object. (On this page the same step stages the empty result with
  **Delete object N** on the button; in a batch there is no one to ask.)
- An object that is no longer in the patch when the batch runs is listed as
  *"… no longer in the volume"*.

### Cancel, and "still finishing"

**Cancel** stops the run and writes nothing. Smooth, Dilate/Erode, Fill Holes
and Guided Grow stop between objects, and an Active contour between planes.
**Settle cannot stop halfway through an object**: *"Cancelling… a Settle step
cannot stop halfway; the objects already started finish and are thrown
away."* Until they finish, Run and Preview wait — *"2 objects from the
stopped batch are still finishing — Run is available again in a few
seconds."* — and come back by themselves. Quitting napari at that moment
also waits for them (at most one object's time each).

### Good to know

- **The count is taken when the dialog opens.** An object you paint, or bring
  back with Undo, while the dialog is open is not in the *"→ Will process …"*
  line — but **Run looks again** and includes it. One you delete meanwhile is
  still counted, and simply not run.
- **One Batch Refine at a time.** The navigator button, while the dialog is
  running or holding a staged result, just brings it to the front; Add's
  **Refine all ▸** says *"Batch Refine is already open — Accept, Reject or
  Cancel it there first."* The other refusals name themselves: *"The last
  batch is still stopping — try again in a moment."*, *"Step 3 is busy with
  another job — try again when it finishes."*, *"Accept or Discard the change
  on screen first."* From the navigator they come in a small notice; from
  **Refine all ▸**, in the Add status line.
- Running a Refine recipe in Batch Refine never changes the recipes file, and
  never changes Batch Refine's own **Recent** or **Saved steps** lists or the
  steps it pre-loads next time.

---

## When nothing moves

A Preview that changes nothing says **why**, in its own sentence — never a
generic one. For example:

- Settle, flush with its neighbours: *"Nothing moved: this object's outline
  already touches its neighbours everywhere, and Settle only moves edges that
  face background."*
- Settle, already on its wall: *"Nothing moved: the wall signal already runs
  along this object's outline."*
- Active contour, when the push is too small: *"Nothing moved. 'Push outward'
  is 0.2 and the Edge gate is 0.30 — below ±0.30 the push does nothing
  anywhere. Lower the gate on the Advanced card, or push harder."*
- Active contour, when every move went past the limit: *"… further than
  3.0 um, so all were undone. Raise 'Edge may move'."*
- Active contour, when every move was into a neighbour: *"Nothing moved: every
  move the outline tried was into a neighbouring object, and no other object
  ever loses a voxel."*
- Fill Holes: *"Nothing moved: this object has no enclosed cavity."*
- Flood Fill: one of its three sentences about where the crosshair is (see
  [Flood Fill](#flood-fill)).

---

## When a step removes the whole object

Some methods can legitimately leave nothing: a big blur on a thin remnant, an
erode on something two voxels wide. When that happens:

- the status line turns amber and says *nothing is left of it*;
- the result block turns red and spells out what Accept would do. After
  Smooth it also names **how thick the object was** and **the knob that did
  it** — *"The object is 3 slices thick."* and *"lower 'Blur through depth',
  or press Discard"*;
- the primary button reads **Delete object N** instead of Accept, and it
  **stays enabled** — deleting is sometimes exactly what you wanted;
- the object is **held in the navigator list** until you Accept or Discard, so
  it does not look as though it has already gone.

**Settle is the exception**: it refuses outright, changes nothing, and tells
you to lower *Edge may move*.

---

## Show what the tool sees

When the staged step built a wall picture — Settle, or Active contour in
*Snap to edges* — the result card offers **Show what the tool sees**. Ticking
it adds the wall picture as a layer over the object, so you can see why the
outline went where it did. The line under the tick says what bright means for
the wall type **that step used** — turning the *Wall type* picker afterwards
does not relabel a picture that has not changed.

When there is nothing to show, the tick is **greyed and the reason is written
where the legend goes** — *"This method does not use a wall picture."*, or,
after a step with no picture of its own (a Smooth after a Settle, say), *"The
last step does not use a wall picture, so there is nothing to show."*

The layer is temporary: it is never offered as a channel, it does not shift the
`a`/`s`/`d`/`f` keys, and it is removed at Accept, at Discard, on the next
Preview, on a patch switch and when the annotator closes.

---

## Advanced

At the bottom of the scroll. What you see depends on the method.

For **Settle**:

- **Hold the outline** — *Diffused edge* only: how firmly the outline is held
  where it started. 1 would hold it exactly; 0 would let it dissolve.
- **Balance** — *Sharp* and *Diffused edge* only: pulls the new outline toward
  a rounder shape. Leave it at 0 unless the result grows a spike.

For **Active contour**:

- **Edge gate** (Snap to edges) — the push only acts where there is no wall,
  and this decides how faint a wall has to be. With the push at or below the
  gate it does nothing anywhere — the line under *Push outward* says so. Lower
  it if *Push outward* seems to do nothing.
- **Blur** (Snap to edges) — how much the channel is blurred before walls are
  measured, in **microns**, so it blurs the same distance through depth as
  across. Only *Membrane sheets*, *Membrane sheets plus shape* and *Intensity
  edges* use it; the other wall types ignore it.
- **Edge sharpness** (Snap to edges) — how hard the outline brakes on a wall.
  Higher stops it dead at a faint wall; lower lets it slide past one.
- **Smooth the join** (a plane orientation) — how smoothly the traced slices
  are joined back into one 3D shape, in **microns**, with what that comes to
  in voxels on this patch beside it.

On every method: **Show the probability map** shows the reconstruction's
confidence around the selected object, in the 3D view and the three side views
— brighter is more sure — with a colour picker beside it. It is for looking
only and changes nothing. With no object selected, or on a patch with no
probability map, the tick comes straight back off and an amber line under it
says which.

---

## Accept, and undoing it

**Accept** keeps the staged shape. Nothing more is written at that moment —
the shape is already in the volume from the Preview — but it stops being
provisional: the object is released, the side views unlock, and the change
becomes one entry in the curation history. It only ever changes **this
object and empty space**, never a neighbour's voxel.

**One Undo reverses the whole chain.** However many steps were staged when you
pressed Accept, they went in as one change, so one **Undo** puts the object
back as it was before the first step.

Nothing reaches disk until **Save curation**.

While a step is running or staged, these are locked, because the staged shape
is already in the volume and has not been decided on:

- **Save curation**, **Commit Triage**, **a run switch** and **switching
  patch**;
- drawing in the side views (Discard would silently undo it);
- choosing a different object to refine (the staged one is held).

---

## Things worth knowing

- While a step is **running or staged**, **Save curation**, **a run switch**
  and **switching patch** are unavailable. The staged shape is already in the
  volume, and saving it would write a shape you have not accepted — Accept or
  Discard first.
- While a result is staged the side views are locked: anything you painted
  would be silently undone by Discard.
- Refine will not run while a Reshape session or a Batch Refine is open —
  finish that first. The status line says which.

---

## Troubleshooting

**A recipe is greyed in the Recipes menu.** Its line says why: no channel
picked, no probability map on this patch, u-Segment3D not installed, or a
voxel size that this patch has and the recipe's did not (or the other way
round). With nothing selected, or a different object staged, every recipe is
greyed and the first line of the menu says which.

**A recipe stopped early.** The line under Preview says at which step; the
status line keeps that step's own reason. The steps before it are still staged
— Accept them, or Discard.

**Preview is greyed.** Read the status line — a greyed button shows no
tooltip. Either no object is selected (*"Select an object first."*), or the
method needs a channel and none is picked (*"Pick a channel first."*).

**Active contour's push does nothing.** Read the grey line under *Push
outward* — if it says *No effect at this setting*, push harder or lower the
*Edge gate* on the Advanced card.

**Inside vs outside brightness makes the object too big (or too small).**
Raise *Weight inside* to shrink it, or *Weight outside* to grow it.

**Smooth deleted a thin object.** Discard, then lower *Blur through depth* —
the red block says how thick the object was.

**Settle did nothing.** Read the status line: it says whether the object is
flush with its neighbours or already sits on its wall. If only a small share
of the border faces background, there is little Settle is allowed to move. A
wider *Edge may move* only helps when the wall is further out than the limit —
the result says which one stopped it.

**Settle made the object much bigger.** Check *Wall type*: a shape type
(*Pinch points*, *Between the spines*) has no wall outside the object, so the
outline grows to *Edge may move*. The amber line under the picker says so.
Discard, and pick a wall type from the channel or the probability map.

**Flood Fill says the empty space does not touch this object, or is open
background.** The crosshair is on the wrong spot. Move it into the gap itself,
against the object, in the side views, and Preview again.

**Dilate +10 twice gave the same as +20.** That is right: when a later step
needs more room than the first step set aside, the region around the object
grows, so chained steps are never cut off at its edge.

**Stop did not stop at once.** A *3D* Active contour and Settle cannot be
interrupted once they have started; the strip reads *"Stopping..."* until they
end. A plane orientation stops between two slices.

**The side views will not let me paint.** A result is staged (or a run is
going). Accept or Discard first.

**Show what the tool sees is grey.** The line where the legend goes says why:
nothing is staged, the method uses no wall picture, or the last step of the
chain used none.

**The probability map will not show.** Read the amber line under its tick: no
object is selected, or this patch has no probability map.

**Guided Grow is greyed in the list.** This patch has no probability map from
the reconstruction; the item's own text says so. Dilate/Erode or Settle work
without one.

**Where did Guided Grow's "Image" and "Both" sources go?** Retired. Growing
into voxels that looked like the object's interior took about 82 % of the
region on membrane data and leaked straight out of an open top; Guided Grow
reads the probability map only.

**Where did the "Image weight" knob go?** Retired. It did nothing at all on a
patch without a probability map, which is most patches. To steer the outline
by the reconstruction's confidence now, pick *Probability map* as the *Wall
type*.

**Where did the image picker on each method go?** There is one **Channel**
picker for the whole tab now, with one **Prepare the channel first** list, and
it appears only for a method that reads a channel.
