# Adding many objects to a cleared region

This tab fills empty space with new objects — many at once. The usual way to
use it: in a region where the reconstruction went wrong, delete the masks you
do not like, put a box over the space they left, and let the tool fill that
space with new objects in one go. You look at what it found, untick anything
you do not want, and press **Accept**.

Two promises hold everywhere on this tab:

- **Existing objects never lose or gain a voxel.** New objects are written
  only where there is no object yet. That is why you clear first: space that
  still belongs to an object is never touched.
- **Nothing is written until you press Accept.** Preview only shows you what
  would be added.

Even after Accept nothing is on disk. The change lives in the session until
you press **Save curation**, and **Undo all** takes back every object this
session added.

---

## The quick version

1. Delete the masks you are unhappy with (the Label Navigator's Delete, or
   the triage tools).
2. Scroll the side views so the crosshair sits in the empty space.
3. Pick what you are adding — **Nuclei** or **Cells with a membrane stain**.
4. Press **Place box at the crosshair**. Resize it with **S**, **M**, **L**
   or **Numbers…**, move it by dragging its outline in a side view.
5. Set up the preset: for Nuclei, the channel and the threshold; for Cells,
   where the markers come from and which channel holds the membrane.
6. Press **Preview**. Every new object appears in the list and on screen.
7. Untick what you do not want and press **Accept (n)** — n is how many
   objects it will add.

---

## The two presets — What are you adding?

The **Adding** picker at the top of the second card chooses how the empty
space is filled. The picker starts on the preset you used last in this
workspace (Nuclei the very first time), and a preset only sets the starting
values of the settings — anything you have changed yourself in this session
stays as you set it.

**Nuclei** finds bright objects by brightness alone. Use it for a nuclear
stain, or anything else that is bright inside and dark around it. You pick
the channel the nuclei are bright in, set a threshold (the **Brightness**
card), and every bright blob in the box becomes an object; blobs that touch
are split apart.

**Cells with a membrane stain** grows one object from each **marker** until
it meets a wall in the membrane channel — or dark space, or the edge of the
box. Use it when the channel you have outlines the cells instead of filling
them: brightness alone would find the outlines, not the insides. It needs
two things, set up in the **Markers** and **Walls** cards: where each object
starts from, and which channel holds the membrane. Its settings are in
microns, taken from Step 3's Spacing (see *The spacing (Cells)* below).

With no box placed, the Cells preset shows only the picker and its one-line
explanation; the Markers and Walls cards appear once there is a box.

---

## The box

- **Place box at the crosshair** drops the box where the crosshair is — so
  scroll the side views to the empty space first.
- **S**, **M** and **L** resize it to about an eighth, a quarter and a half of
  the patch. On the Cells preset the box is kept to at most 4 million voxels,
  so L may come out smaller than half; on Nuclei there is no limit.
- **Numbers…** opens a small window, *The box, in numbers*, with the box's
  centre and size on each axis as sliders.
- **⌖ Recenter** moves the box so its centre is where the crosshair is now.
- **Drag the outline** in any side view to move it: press on the rectangle's
  line itself with the left mouse button and drag. A drag started inside the
  box (or anywhere off the line) pans the view as usual. The mouse pointer
  turns into a move cursor when it is over the line. The band you can grab
  is a few pixels either side of the line, and never more than a quarter of
  a small box's width — zoomed far out, the middle of a small box stays free
  for placing markers.
- **The outline is bright on the slices it covers and faint on the others —
  scroll to a bright slice to drag it.** On a faint slice you are looking at
  the box from outside its depth, and it cannot be grabbed there.
- **Shift + arrow keys** nudge the box one voxel at a time. Click the side
  view first, so the keys go to it; plain Up and Down still step through
  the labels as before, and plain Left and Right still step through the
  slices.
- The box keeps its size when you drag it against the edge of the patch: it
  slides back inside instead of shrinking.

The first line of the page tells you the box's size, for example
*"Box 40 × 125 × 125 voxels (0.6 million) · 85 % of it is empty space"* on
Nuclei. On Cells it also names the limit:
*"Box 40 × 125 × 125 voxels (0.6 million; the limit for cells is 4 million) · 85 % of it is empty space"*.
The empty share is worth a glance: a box that is mostly taken already has
little room to fill.

The box is a drawing, not a layer: it never appears in the layer list, so it
cannot be painted on or saved by accident. The 3D view shows it as a
wireframe.

---

## Nuclei — the Brightness card

- **Channel** — the channel the nuclei are bright in.
- **Local contrast** — evens out bright and dim regions of the box before
  the threshold. Off by default; turn it on for a box that is unevenly lit.
- The **histogram** shows the brightness of the box, with a red line at the
  threshold. Voxels at or above the line count as object. Drag the line to
  the dip between background and nuclei, or type a value in **Threshold**.
- **Auto** puts the threshold at the automatic split between background and
  the bright objects (Otsu's method).

The histogram and Auto are measured in the background whenever the box, the
channel or Local contrast changes, so they may take a moment to appear —
until then **Auto** is grey and the card says *"Measuring the box for
Auto…"*. While you have not touched the threshold it follows Auto after each
change; once you have dragged the line or typed a value it stays where you
put it. Turning Local contrast on or off changes the scale of the histogram,
so the line goes back to Auto.

---

## Cells — Markers

A marker is where one new object starts from: **one marker, one object**. The
**From** picker offers three sources:

- **Objects from run ▸ …** — the objects of another run of this patch (a
  curated run is listed first). Each object of that run, shrunk a little so
  that neighbours do not touch, becomes one marker. The card counts them,
  for example *"14 in the box · 9 over empty space become markers."* — only
  the part over empty space can start a new object. Until the count is in,
  the line reads *"Counting the run's objects in the box…"*. A run that is not
  curated is marked *(reconstruction, not curated)*. When no other run of
  this patch exists, the item reads *Objects from another run (no other run
  of this patch)* and is grey.
- **Nuclei masks from a channel** — nuclei are found in a second channel
  with the Nuclei preset's own finder, and each nucleus becomes one marker.
  Pick the **Nuclei channel**; the **Threshold** beside it has its own
  **Auto**, measured in the background like the Brightness one. Slivers of
  nuclei pressed against objects you kept are skipped. Two nuclei that touch
  can still come out as one marker, so one object — look before you Accept.
  In densely packed tissue a nucleus can also leave an extra small object;
  those usually carry the *small* chip.
- **Click markers** — you place the markers yourself. Turn on **Place
  markers**, then click once inside each object you want, in any side view.
  Each click becomes a small ball (about 4 µm across at the default size),
  drawn on screen. **Undo last** removes the last one, **Clear** removes
  them all, and **Ctrl + click** on a ball removes that one. The count beside
  the button says how many are placed — *"7 placed · 5 ticked"* when some
  are unticked.

**The marker list.** Every click marker gets a row under Undo last and
Clear: *Object 412* and where it sits (z / y / x), with a colour swatch. The
number is the object the marker will become, and its ball is drawn in that
object's colour — after Preview the new object has the same number and the
same colour, so a marker and the object it grew read as one. Its tooltip:
"Each marker becomes the object of its colour. Untick one to leave it out of
the next cut."

- An unticked marker's ball disappears from the views and the next Preview
  cuts without it. Unticking a marker leaves a gap in the numbering; the
  other colours do not change. Removing a marker in the middle leaves a gap
  too; Undo last, or removing the last one, gives its number back.
- Colours repeat every 49 numbers, which is why each row shows the number.
- The list shows three rows and scrolls. It is hidden while a result is on
  screen — the result list is the list then, in the same colours — and comes
  back, ticks and all, after **Discard**. So: untick an object in the result
  to leave it out of this Accept; or Discard, untick its marker and Preview
  again to cut the box without it.
- With every marker unticked, Preview is grey and the page says "Tick at
  least one marker to cut — all 3 are unticked."; with some unticked the box
  line says so, for example "Box 40 × 125 × 125 voxels · 7 markers placed ·
  5 of 7 ticked".
- The number a marker gets is fixed when you place it. If the patch's
  numbering moves on meanwhile (an object with a higher number was deleted
  elsewhere), every marker's number moves by the same amount and the colours
  follow — and the first line says so, for example "The object numbers moved
  up by 3 — something else took the old ones." (or "The object number moved
  up by 3 — something else took the old one." for a single marker). If the
  numbering moves down instead, it says "The object numbers moved down by 3
  — the lower ones are free again." ("The object number moved down by 3 —
  the lower one is free again." for one).
- A marker whose ball sat inside an object you kept at the last Preview made
  nothing. After Discard its row is dim and its tooltip says why: "At the
  last Preview this marker sat inside a kept object, so it made nothing —
  remove it and click in empty space."

Details of clicking:

- A click only counts while **Place markers** is on. It switches itself off
  when you change tab, preset or source.
- A click must be on empty space inside the box. On an existing object the
  page says *"That spot belongs to object 12 — markers go in empty space."*;
  outside the box it says *"Click inside the box."*
- A quick click on the box outline still places a marker — only a press that
  moves becomes a drag.
- A click inside a marker you already placed — ticked or not — is refused:
  "That spot is already inside marker 12 — click a little further away."
  (two markers that close would give one object).
- A double click does not zoom. Its first click places a marker; its second
  lands inside that marker and is ignored quietly — no warning, one marker.
- A click inside a marker you have unticked (its ball is not on screen) is
  refused too, and the page says where it is hiding: "That spot is inside
  unticked marker 12 — tick it in the list to see it, or click a little
  further away."
- **Ctrl + click** reaches only the balls on screen, so an unticked marker is
  removed with Undo last or Clear, or ticked again first.
- Moving the box keeps the markers that are still inside it and removes the
  others — unticked ones too — and says so.

**On a patch with only one channel** the nuclei source is grey — *Nuclei
masks (needs a second channel)* — because it needs a channel of its own.
When the run source is grey too, the card shows an amber line: *"No automatic
marker works on a membrane stain alone. Click once inside each object you
want. If dim cells come out empty, lower Dark below or untick Treat dark
space as outside."* Click markers are the way to go there.

---

## Cells — Walls

- **Channel** — the channel the membrane is bright in. It must be a
  different channel from the nuclei channel.
- **Prepare the channel first** — cleans the channel before the tool reads
  it, with four ticks: **Even out depth** (ticked by default; lifts deep, dim
  slices to match the shallow ones), **Remove haze** (takes away smooth
  background glow), **Lift dim signal** (brightens faint signal more than
  bright signal) and **Despeckle** (removes salt-and-pepper noise). They
  always run in that order. Local contrast is not used when the tool is
  looking for walls — these ticks are the only preparation the membrane
  channel gets.
- **Wall type** — what the edge between two cells looks like in the
  membrane channel. **Membrane sheets** (the default) suits an ordinary
  membrane stain; the others are **Intensity edges**, **Raw brightness**,
  **Vessel ridges** and **Thin-faint ridges**. The Reshape guide describes
  each one.

---

## Advanced

At the bottom of the page, closed by default, with a one-line summary of the
current values. Each preset shows only its own rows. You will not often
need it.

**Cells only** (in microns, so they mean what they say only once Step 3's
Spacing is set — see *The spacing (Cells)*):

- **Click ball radius** — how big each click marker is: a ball of this
  radius. 2.0 µm by default.
- **Shrink run objects** — each object of the other run is shrunk by this
  before it becomes a marker, so neighbours do not touch. Objects too small
  to survive the shrinking are skipped.
- **Treat dark space as outside** — ticked by default. Dark space (no signal
  in any channel) then stops the fill, the way the box edges do, so an object
  next to open space does not spill out into it.
- **Dark space reach** — the fill may reach this far into dark space before
  it stops.
- **Dark below** — signal below this share of the patch's usual brightness
  counts as dark space.

**If dim cells come out empty** — their inside is so faint that it reads as
dark space — lower **Dark below**, or untick **Treat dark space as outside**.
Unticked, the box edges alone stop the fill and the two Dark rows are grey:
dim cells are filled properly, but an object next to open space may spill into it,
so look at those before you Accept. The tick is remembered for this
workspace, beside the last preset. While it is off (on Cells, before a
result), an amber line under the first one says so: *"Dark space is not
treated as outside, so objects run to the box faces wherever there is no
wall — tick it in Advanced if objects come out open at the top or bottom."*
The page has room for one amber line there: when the spacing line would show
too, this one takes the place and the spacing line moves into Advanced, under
the Dark rows.

**Nuclei only:**

- **Separate touching objects** — splits objects that touch. Off gives one
  object per bright blob.
- **Spacing** — the closest two object centres may be, in voxels. Lower gives
  more, smaller objects.
- **Local contrast strength** — how strongly Local contrast evens the box
  out.

**Both presets:**

- **Untick objects at the box edge** and **Edge zone** — an object that
  reaches within this many voxels of the box edge was probably cut by the
  box, so it starts unticked. A lone object stays ticked.
- **Smallest object** — objects smaller than this many voxels are left out.
  0 keeps every one.
- **Smooth the added objects** and **Smoothing** — smooths each new object's
  surface. This happens **in the preview**, so what you see is exactly what
  Accept writes. 1.5 voxels is gentle. Smoothing can make a very small
  object vanish; those are left out and counted (see below).

---

## Preview, Stop and the stage line

**Preview** sits just above the two buttons at the bottom, so it is always in
reach. It reads **Preview again** once a result is on screen. Beside it are
**Stop**, a busy bar and one grey line saying what the tool is doing now —
*"Reading the box..."*, *"Finding markers..."*, *"Building the wall
picture..."*, *"Filling the empty space..."*, *"Smoothing the new
objects..."* (Nuclei: *"Finding bright objects..."*). The strip is always
drawn, dimmed while nothing runs, so the page never jumps.

The preview runs in the background and the napari window stays usable. A
Cells preview of a large box — near the 4-million-voxel limit — takes about
12 to 20 seconds, which is why **Stop** exists. Stop throws the run away:
*"Stopped. Nothing from this box was added."* Nothing is written.

While a preview runs every change is refused rather than half-applied —
moving the box, a click, a preset or channel change, the Advanced settings:
*"Wait for the preview to finish, or press Stop."*

---

## The result

The **Result** card lists every new object, four rows at a time (scroll for
more). Each row shows a tick box, *Object 1234*, its size in voxels, and
sometimes a chip:

- **edge** — the object reaches the box edge, so the box probably cut it. It
  starts unticked.
- **small** — much smaller than the others in this result (under a fifth of
  the typical size here). It starts unticked. *Small* is relative to this
  result, not a fixed size.

Unticking is a suggestion, not a rule: tick anything you actually want. Rows
with a chip are listed first. **All** and **None** tick or untick every row.
The first line of the page says, for example, *"12 new objects found · 3 left
unticked (at the box edge)"*.

Below the list, one grey line counts what the tool set aside, for example
*"2 markers fell in dark space and were skipped · 1 object vanished when
smoothed and was left out"*. When it is cut short, hover it to read the
whole of it; the same text is in the log. A click marker whose ball ended up
inside an object you kept makes nothing, and that is counted first: *"1
marker now sits inside a kept object and made nothing"* (*"2 markers now sit
inside kept objects and made nothing"*).

**Show what the tool sees** (Cells only) lays the wall picture the fill
followed over the box, in the main view. On Nuclei it is
grey: *"Nuclei builds no wall picture — the walls are a Cells idea."*

**Show candidates as outlines** draws the new objects as outlines in the
three side views, so the membrane under them stays visible. The 3D view
cannot show outlines — napari does not draw them in 3D — so it is unchanged.
The tick is kept when you Discard and preview again in the same box, and goes
back off after Accept, Cancel or a patch switch.

The new object ids **skip the ids of objects you deleted**: a new object never
reuses the number of one you removed, so a deleted object's old number never
comes back as something else.

---

## Accept, Discard, Cancel

The two buttons at the bottom never move, and one grey line under them
says what the second one would do — *"Nothing has been added yet."* before a
box, *"Cancel removes the box and its markers; nothing was added."* with a
box, *"Only empty space is written. Discard throws these objects away; the
box and markers stay."* with a result — though with a result on screen that
line gives its place to **Accept and refine ▸** (below), and the sentence is
on Accept's tooltip instead. What the second button says depends on where
you are:

- **Accept (n)** writes the ticked objects — only into empty space — and
  takes you to the review (below). With every row unticked it says *"Tick at
  least one object to add — all are unticked."*
- **Accept and refine ▸** (a result is on screen, under the two buttons)
  does both in one click: *"Adds the ticked objects, then opens Batch Refine
  on every object added this session — the same as Accept followed by Refine
  all."* It is grey whenever Accept is — with nothing ticked it is
  disabled, not refused. If the Accept itself is refused (the voxels already
  taken, say), Batch Refine does not open, the button goes grey, and the
  first line says why.
- **Discard** (a result is on screen) throws the result away and goes back to
  the box. **The box and your markers stay**, so you can change a setting
  and press Preview again.
- **Cancel** (no result on screen) ends the session: the box and its markers
  are removed. Nothing was added.

A result goes stale as soon as you change what it was made from — the box,
the markers, a setting. The result is then taken down and the page asks for
another Preview.

---

## After Accept — the review

The page turns into a short review of this session's work: *"Added 12 objects
· 34,567 voxels"*, and the **Added this session** list (four rows at a time,
scrolling).

- **Refine all ▸** opens Batch Refine with **every** object added in this
  session as its target (*Added this session (N)*) — pick a recipe saved on
  the Refine tab and run it on all of them. The Refine guide's *Running it on
  many objects* has the details. If Batch Refine cannot open, the first line
  here says why.
- **Refine ▸** takes **one** object — the one selected in the list — to the
  Refine tab. It is grey until exactly one row is selected.
- **+ Add another** starts again in the same box, with the same preset and
  marker source. Click markers are cleared.
- **Undo all** takes back every object added this session — every Accept,
  newest first. It asks first (*"Take back all 12 objects added this
  session?"* — **Take them back** / **Keep them**).
- **Done** closes the session and keeps everything you added.

**If you ran a batch on the new objects and accepted it**, that batch is a
later edit sitting on top of them, so **Undo all** stops at it: *"Nothing was
taken back — a later edit is in the way. Undo it first, or use the main Undo
button."* Press the main Undo to take the batch back, then **Undo all**
again. If only the newer objects could be taken back it says how many, for
example *"Took back 4 of 12 added objects — a later edit is in the way of the
rest."*

**While Batch Refine runs, or waits for its Accept / Reject,** Add refuses
to place a box, preview, accept, add another or undo: *"Batch Refine is
working on these cells — Accept or Reject it first."* Rejecting a batch puts
the objects back exactly as they were before it, so anything added in the
meantime would be lost.

---

## The spacing (Cells)

Cells measures in microns — the click balls, the shrinking of run objects,
how far the fill may reach into dark space — using Step 3's own **Spacing**,
set under QC Features → Settings… in Step 3: the **Settings…** button beside
**Extract** in the QC Features section, row **Spacing**. A voxel size entered in Step 1 does not reach it today.

If it has never been set it is the default 1.0 µm, and Cells **runs on it
anyway** — nothing is refused and nothing is grey. An amber line under the
first one says so: *"Spacing is the default 1.0 µm here — if your voxels are
smaller, set Spacing under QC Features → Settings… so the µm knobs mean what
they say."* While a box is placed that button is grey (the box holds
Step 3's lock), so the line then reads: *"Spacing is the default 1.0 µm
here — Cancel the box, then set Spacing under QC Features → Settings… if
your voxels are smaller."* On real voxels of about half a micron every micron setting then
acts at about twice its size, and the fill leaks further into open space, so
set it once. The line is not shown while a result is on screen; it comes
back when you set up the next Preview. With **Treat dark space as outside**
unticked, the dark-space line takes its place and this one is in Advanced,
under the Dark rows. Nuclei works in voxels and does not
need it.

---

## What is refused, and why

The first line of the page always says why **Preview** is grey, and says it
in amber — so does every refusal (a click that landed on an object, a box
the batch is still working on). The common reasons:

- **The box is over the limit (Cells).** *"This box is 5.2 million voxels;
  the limit for cells is 4 million. Shrink it with Numbers… or pick S, M or
  L."* Switching from Nuclei to Cells never resizes your box, so a large
  Nuclei box can arrive here.
- **A channel is missing.** *"Pick the channel the nuclei are bright in, under
  Brightness."*, *"Pick the membrane channel under Walls."*, *"Pick the nuclei
  channel under Markers."* — or *"The nuclei and wall channels must be
  different."*
- **No markers yet (click markers).** *"Place at least one marker: turn on
  Place markers, then click inside the box."*
- **Still measuring.** *"Measuring the box…"* — the background measurement
  of the box has not finished; it takes a second or two after a change.
- **Busy.** A preview is running (see above), or Batch Refine is running or
  staged.

And after a Preview:

- *"Nothing found — lower the threshold or grow the box."* (Nuclei.)
- *"No marker was left to grow from — every marker fell in dark space, over
  kept objects, or was too small to use. (Nothing was changed.)"* (Cells.)
- *"Every new object was below Smallest object or vanished when smoothed —
  lower them under Advanced."*
- *"The preview failed — the reason is in the log. Nothing from this box was
  added."*

---

## Closing the annotator

When the annotator window itself is asked to close while Add work is on
screen — a preview running, new objects not yet accepted, or click markers
placed — it asks first, in a window titled *Uncommitted Work* that names
what would be lost. **Go Back** keeps everything; **Discard** throws it away
and closes.

**Be aware:** the × on the title bar of the annotator's panel in napari does
not go through that question. It closes straight away, and the unaccepted
work — the preview and your markers — is lost without asking. Nothing
unaccepted was ever written, so the volume is unchanged; objects you already
accepted stay, because they are in the volume. Press **Accept** or **Cancel**
before you close, and nothing is lost by surprise.

While a box session is open the patch cannot be switched and the run cannot
be changed — finish or cancel first.

---

## What changed from the old Add tab

- It fills a whole region with many objects at once, instead of adding one
  missed cell. The **Cells with a membrane stain** preset is new; **Nuclei**
  is the old behaviour.
- New object ids skip the ids of deleted objects (the old tab reused them).
  The shapes Nuclei finds are the same as before.
- **Discard** no longer ends the session: it drops the result and keeps the
  box and markers. **Cancel** ends it.
- The box can be dragged by its outline and nudged with Shift + arrow keys.
- Closing the annotator asks before it drops unaccepted work (except the
  title-bar ×, above).
- *Connectivity* and the *Signal* choice are gone from Advanced.
- Cells no longer refuses when Step 3's Spacing was never set: it runs on the
  default 1.0 µm and says so in amber (see *The spacing (Cells)*).
