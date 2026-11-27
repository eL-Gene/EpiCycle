# Adding many objects to a cleared region

This tab fills empty space with new objects — many at once. The usual way to
use it: in a region where the reconstruction went wrong, clear the objects you
do not like, put a box over the space they left, and let the tool fill that
space with new objects in one go. You look at what it found, untick anything
you do not want, and press **Accept**.

Two promises hold everywhere on this tab:

- **Objects you keep never lose or gain a voxel.** New objects are written
  only where there is no object yet. That is why you clear first: space that
  still belongs to an object is never touched.
- **Nothing is added until you press Accept.** Preview only shows you what
  would be added.

**Clear objects** is the one exception: it removes objects at once. Only the
main Undo button brings them back — not Close box, Discard or Undo all — and
only until you switch patches, run **Extract** or **Commit Triage**, or use
Reset to Predicted (see *Clear objects* below).

Even after Accept nothing is on disk. The change lives in the session until
you press **Save curation**. Until you press **Done**, **Undo all** takes back
every object this session added; after Done the main Undo button takes them
back one Accept at a time.

---

## The quick version

1. Delete the few objects you are unhappy with (the Label Navigator's
   Delete).
2. Scroll the side views so the crosshair sits in the empty space.
3. Pick what you are adding — **Nuclei** or **Cells with a membrane stain**.
4. Press **Place box**, or **Whole patch** to make the box the whole patch.
   Resize the box with **S**, **M**, **L** or **Numbers…**, move it by
   dragging its outline in a side view.
5. Set up the preset, now that the box is there: for Nuclei, the channel
   and the threshold — or, to find the nucleus inside each cell of another
   run, **Look for nuclei ▸ Inside each cell of run ▸ …**, or, to pick the
   nuclei yourself, **Look for nuclei ▸ At the markers I place** and one
   click per nucleus; for Cells, where
   the markers come from (**Markers ▸ From**), which channel holds the
   membrane, and the **Tissue outline** the new cells stay inside. Every
   choice works the same on a box and on a whole patch; on Cells a whole
   patch needs a source under **From** (a run is never picked for you; a
   source picked on the box stays picked), and until then **Preview** is
   grey and the first line says so.
6. If objects are in the way, **Clear objects** (left of Preview) removes
   the ones mostly inside the box, or every object in the patch.
7. Press **Preview**. Every new object appears in the list and on screen.
   Objects at the box edge or very small start unticked: they are drawn
   faintly in the three side views, not at all in the main view. After a
   Clear the new objects often reach the box edge, so all of them can start
   unticked — press **All** to tick them.
8. Untick what you do not want and press **Accept (n)** — n is how many
   objects it will add.
9. Look over the review, then press **Done** to keep the objects — or **Undo
   all** to take every one of them back.

---

## The panel, top to bottom

The Add page has the same slots as the other tools in this dock, top to
bottom, and the two buttons at the bottom never move. What you cannot use
yet is hidden, not greyed: a setting appears once it means something.

- Above everything, a single sentence with a chevron: *Fills empty space
  inside a box with new objects — many at once.* Click it and a paragraph
  opens underneath: it starts with the one exception — **Clear objects**
  removes objects at once, and only the main Undo button brings them back —
  and then explains what each preset does.
- **Status** — one line with a coloured dot: what the box holds, what was
  found, or why **Preview** is grey. Under it there is room for one amber
  line (the spacing, dark space, a half-size session, a slow patch).
- **The box** — with no box, one row: **Place box** and **Whole patch**,
  both always live. With a box, one row of sizes: **S**, **M**, **L**,
  **Whole** (the whole patch), **Numbers…** and **⌖** (Recenter: the box's
  centre goes to the crosshair). A grey line under it says how to move the
  box.
- **What are you adding?** — the **Adding** picker and one line about the
  preset. On Nuclei, the **Brightness** card (and under it, with **Look for
  nuclei ▸ At the markers I place**, the **Markers** card), and — on a box,
  before a result, with **Anywhere in the box** or the markers — the
  **Tissue outline** card, closed and set to **No outline**; on
  Cells, once a box is placed, the **Markers** and **Walls** cards. Cards
  open one at a time.
- **Result** — appears once a Preview has found something: the list of new
  objects with their ticks, and one line counting what was set aside.
- **Advanced** — at the bottom of the page, closed, and shown once a box is
  placed. Only the rows that apply are there: **Shrink run objects** only
  with markers from another run, the edge rows only on a box (not on the
  whole patch). The size of a click marker, **Ball**, is not here: it sits
  beside **Place markers** on the Markers card.
- **Clear objects**, **Preview** and **Stop** — pinned just above the two
  buttons, so they never scroll away. **Stop** is the one way out while a
  preview runs.
- **The two buttons** and one grey line under them that says what they do.
  The first is **Accept (n)** over a result and **Done** in the review. The
  second has one word for each meaning: **Close box** while a box is up
  (and grey, with the same word, before one is), **Discard** over a result,
  **Undo all** in the review. It is grey while a preview runs.

After **Accept** the page turns into the review (see *After Accept — the
review*), which takes the place of everything between the status line and
the pinned strip.

---

## The two presets — What are you adding?

The **Adding** picker at the top of the second card chooses how the empty
space is filled. The picker starts on the preset you used last in this
workspace (Nuclei the very first time), and a preset only sets the starting
values of the settings — anything you have changed yourself in this session
stays as you set it.

**Nuclei** finds bright objects by brightness — or, with **Look for nuclei
▸ Inside each cell of run ▸ …**, the nucleus inside each cell of another
run (see *Nuclei inside each cell of another run* below), or, with **Look
for nuclei ▸ At the markers I place**, only the nuclei you click (see
*Nuclei at the markers you place*). Use it for a
nuclear stain, or anything else that is bright inside and dark around it. You pick
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
explanation; the Markers and Walls cards appear once there is a box. On
both presets **Advanced** waits for a box too.

---

## The box

- **Place box** drops the box where the crosshair is — so scroll the side
  views to the empty space first.
- **Whole patch** (beside Place box), and **Whole** on the row of sizes once
  a box is up, make the box the whole patch — see *The whole patch* below.
- **S**, **M** and **L** make each side of the box about an eighth, a quarter
  or a half of the patch's side. On the Cells preset they make a box of at
  most 4 million voxels, so L may come out smaller than half on a large
  patch; that is their size, not a limit — a box made larger with
  **Numbers…** is kept. On Nuclei they are always the full fraction.
- **Numbers…** opens a small window, *The box, in numbers*, with the box's
  centre and size on each axis as sliders.
- **⌖** (Recenter — the word is in its tooltip) moves the box so its centre
  is where the crosshair is now.
- **Drag the outline** in any side view to move it: press on the rectangle's
  line itself with the left mouse button and drag. A drag started inside the
  box (or anywhere off the line) pans the view as usual. The mouse pointer
  turns into a move cursor when it is over the line. The grab band is a few
  pixels either side of the line, and never more than a quarter of
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
*"Box 40 × 125 × 125 voxels (0.6 million; the limit for cells is 100 million) · 85 % of it is empty space"*.
The limit is the same for a box and a whole patch: Preview fills either up
to 100 million voxels, and **Update as I click** updates either up to 20
million.
The empty share is worth a glance: a box that is mostly taken already has
little room to fill.

The box is a drawing, not a layer: it never appears in the layer list, so it
cannot be painted on or saved by accident. The 3D view shows it as a
wireframe.

---

## The whole patch

**Whole patch** (beside **Place box** when no box is placed) and **Whole**
(on the row of sizes once a box is up) make the box the whole patch:
*"Use the whole patch as the box."* The two are never grey — pressing either
always makes the box the whole patch. Whether the page can fill it depends
on what is picked, and **Preview** says so:

- Every feature behaves the same on a box and on a whole patch. On
  **Nuclei** every **Look for nuclei** works there (the markers are placed
  on the whole patch as in a box); **Anywhere in the box** finds the nuclei
  over the whole patch, its **Tissue outline** card included. Its finder
  has a few long steps (the distance map, the flood) that cannot be cut
  short: **Stop** takes effect when the step running ends.
- On **Cells with a membrane stain** it takes any marker source a box
  takes — **Objects from run ▸ …**, **Nuclei masks from a channel** or
  **Click markers** (see *Cells: the whole patch* below). The Markers card
  is on screen as soon as the box is, right under the box.
- The one difference: a whole patch is filled with a background marker and
  its faces left open, because a patch edge is the end of the data and a
  box edge cuts through tissue.

With no box placed, the first line reminds you it is there:
"Place a box over the space to fill, or press Whole patch."

- The first line then reads, for example, *"Whole patch 81 × 512 × 512
  voxels (21.2 million) · 64 % of it is empty space"*.
- The whole patch does not move: *"The box is the whole patch, so it does
  not move. S, M or L make it a box again."* **⌖** is grey, a drag on the
  outline pans the view, and Shift + arrow keys do nothing.
- Only a slider in **Numbers…**, or **S**, **M** or **L**, makes it an
  ordinary box again. Picking another preset, another run or another Look
  for nuclei keeps the whole patch; when the new pick cannot fill it,
  Preview is grey and the first line says what to pick.
- **Place markers** stays as it was when the box becomes the whole patch,
  and a click anywhere in the patch places a marker, as in a box.
- Click markers you placed on the box before stay on the whole patch.
- When **S**, **M** or **L** make it a box again, the box comes back where
  it was before the whole patch: the size it had gives you exactly that box
  again, and another size is centred on it. With no box before (Whole patch
  pressed beside Place box), the new box sits at the centre of the patch.
- Leaving the whole patch (S, M, L or a slider in **Numbers…**) while
  markers are placed asks first, in a box titled *Leaving the whole patch*:
  *"Leaving the whole patch removes its 3 markers."* **Go back** keeps the
  whole patch and its markers as they were. **Remove them** makes the box
  and removes every marker, and the first line says *"The markers placed on
  the whole patch were removed — place new ones in the box."* **Place
  markers** stays as it was. With no marker placed nothing is asked.
  **Close box** removes all the markers.
- **Clear objects** offers only **Every object in the patch…** here.
- A face of the whole patch is the edge of the data, not a cut made by the
  box, so on the whole patch no object is marked **edge**, and the two edge
  rows of Advanced are hidden.
- When you press Preview, reading the whole patch takes a moment before
  the preview starts; the first line says *"Reading the patch…"* while it
  does. The busy bar then counts the steps of the run (on Cells it moves to
  and fro during the fill itself, see below). A large patch says so in
  amber before you press Preview. On Nuclei it says *"This patch is 73.0
  million voxels; a Preview may take several minutes or longer. Stop acts
  between steps."* — on **Anywhere in the box** with a rough figure
  instead: *"This patch is 73.0 million voxels; a Preview may take under
  a minute, longer on a slower computer. Stop acts between steps."* (on
  Cells see *How long it takes* below). On Nuclei **Stop** takes effect
  between the steps, and inside the long per-cell steps every 200 cells.
  On a box or a patch over 20 million voxels the grey line after a Stop
  adds *"Stop takes effect when this step ends."* while the step running
  ends.
- With **Update as I click** on, Nuclei ▸ **At the markers I place** on the
  whole patch needs no Preview: each click updates the objects (see
  **Update as I click** under Advanced below). The first update after a
  click reads the patch once, so it takes longest; the grey line says
  *"Reading the patch..."* while it does, the bar moves to and fro, nothing
  on the page goes grey and **Stop** stays grey. Later clicks reuse what
  was read and are quick. A large patch says so in amber: *"This patch is
  73.0 million voxels; the first update reads it once and may take a
  while. Later clicks are quick."* With the tick off the whole patch waits
  for Preview, as above.
- **Accept** on the whole patch also takes a moment: the first line says
  for example *"Adding 812 objects…"* while the new objects are written,
  the grey line says *"Writing the new objects..."*, and the bar steps
  through the six parts of the work. The napari window can stay busy for a
  few seconds after the review appears.

### Cells: the whole patch

On **Cells with a membrane stain** the whole patch is filled **in one
step**: every marker grows at once until the walls stop it, the way the fill
of a box runs to its walls. Neighbouring cells meet at the membrane between
them. **Grow by** applies on a whole patch as on a box (see *How cells
grow*), and the line at the top of Advanced says what fills the patch under
your pick. With **Fill to the walls (as before)** it reads *"Filled from
every marker at once; far tissue stays empty (click markers there). A patch
edge is the end of the data; a box edge cuts through tissue."* That last
sentence is the one way a whole patch differs from a box: the fill keeps
the space far from every marker empty with a background marker and runs
to the patch's faces. **Wall cost**
and **Growth limit** are hidden (the fill does not use them); **New cells
include the membrane at their edge** stays, and is honoured with **Update
as I click** off. The Advanced summary then reads *"filled to the walls ·
membrane included"* (or *"filled to the walls · membrane left out"*). With
**Spread from each marker** the whole patch grows by Spread, with its Wall
cost, Growth limit and membrane tick, and the line reads *"Spread from each
marker grows every cell of the whole patch at once; each stops at a wall,
where a neighbour got there first, or at its growth limit. Tissue no cell
gets to stays empty."* The pick is one for boxes and the whole patch: what
you pick on the whole patch is what your next box grows by. With **Update
as I click** on, Fill to the walls and Membrane sheets or
Intensity edges walls, a whole patch of up to 20 million voxels with Click
markers is filled as you click the way a box is instead (see **Update as I
click** under Advanced): the line at the top of Advanced then reads
*"Filled as you click, the way a box is: tissue far from every marker
fills too. To keep it empty, untick Update as I click and press
Preview."*, the membrane tick is hidden (that fill does not use it), and
the summary reads *"filled to the walls"*. When the quick update cannot
take such a patch (see **Update as I click**), the line says so too:
*"Filled on Preview, the way a box is: tissue far from every marker fills
too. To keep it empty, untick Update as I click and press Preview."* With
**Always update as I click** on, such a patch is filled a moment after
your clicks instead, and the line reads *"Filled a moment after your
clicks (Always update as I click is on), the way a box is: tissue far from
every marker fills too."*

**Tissue far from every marker stays empty.** Before the fill, the tool
marks as background the space that lies far from every marker: how far
depends on how close each marker's neighbours are (never less than 6 µm
from it, and further when its neighbours are far away). That is where the
background starts, not how far a cell grows: in plain cytoplasm a cell and
the background meet about half way. The background reads only where the
markers are and the voxel size — never a channel. A cell the other run
missed that sits among marked cells can still be shared out between its
neighbours, and a tall cell whose marker sits half way up can come out
short — look for cells that came out too big or too short before you
Accept. The empty space that is left is where a box with **Click markers**
fills the cells the run missed.

- On Cells the whole patch takes its markers from any source a box takes:
  another run (**Markers ▸ Objects from run ▸ …**), **Nuclei masks from a
  channel**, or the markers you click (**Click markers**). A run is never
  picked for you; a source picked on the box stays picked. So the way to
  fill a whole patch with cells is: **Cells** → **Whole patch** →
  **Markers ▸ From ▸** a source (with **Click markers**, then **Place
  markers** and a click in each cell) → **Preview** → **Accept**.
  With nothing picked, Preview is grey and the first line says "Pick a
  run, Nuclei masks or Click markers under Markers ▸ From — a whole patch
  grows its cells from those markers." On a patch with no other run the
  list holds the grey *Objects from another run (no other run of this
  patch)* too, and the first line says "This patch has no other run —
  pick Nuclei masks or Click markers under Markers ▸ From."
- With **Nuclei masks from a channel** the nuclei found over the whole
  patch are kept for the next Preview of the patch, and found again only
  when the labels, the Clears in force, the nuclei channel, its threshold,
  the voxel size or the half size change. Every nucleus found is a marker,
  as in a box; the line under the result counts the small ones on a box
  and a whole patch alike, for example *"12 markers were under 10 µm³"*,
  so you can judge them.
- With Click markers, a marker far from every other one can make a large
  cell that reaches into the empty space around it (the background starts
  further from a marker whose neighbours are far away, see *Tissue far from
  every marker stays empty* above): click one marker in each cell of the
  area you want filled, not one here and there.
- The source picked under **From** on the box stays picked on the whole
  patch, and a source picked on the whole patch stays picked on the next
  box (for the click markers themselves see *The whole patch* above).
- The first line reads, for example, *"Whole patch 81 × 512 × 512 voxels
  (21.2 million) · 64 % of it is empty space"*. The size limits are a
  box's: 100 million voxels for a Preview, 20 million to update as you
  click. While it is measured it reads *"Measuring the
  patch…"*; while a preview runs, *"Working on the patch…"* and the grey
  line *"Reading the patch..."*; a Stop says *"Stopped. Nothing from this
  patch was added."*
- **The size limit.** A whole patch can be filled up to 100 million voxels.
  Above that Preview is grey and the first line says, for example, *"This
  patch is 150.0 million voxels; a whole patch can be filled up to 100.0
  million. Press S, M or L for a box."* — Preview does not start.
- The markers' count speaks of the patch: *"14000 in the patch · 12500
  over empty space become markers."* (and *"Counting the run's objects in
  the patch…"* until it is in).
- The **Tissue outline** (under Walls) starts at *Outline of the objects
  you cleared* when a Clear is in force on this patch, else at *Shrink-wrap
  around the stain*, as on a box — unless you picked one yourself (see
  *The tissue outline*).
- At the never-set spacing the whole patch runs as a box does, on the
  default 1.0 µm, and the amber spacing line says so (see *The spacing
  (Cells)*); when another amber line (a slow patch, a half-size session)
  holds that place, the spacing line moves into Advanced.
- **Stop.** While it runs, the bar counts the steps and the grey line names
  each one. Building the wall picture (*"Building the wall picture..."*) is
  one long step: the bar moves to and fro while it runs, and **Stop** acts
  inside it. **Stop** acts between the steps, also while the grey line says
  *"Finding markers..."*, which is when the background is measured. The
  fill itself is one long step that cannot be
  stopped once it has started: the bar then moves to and fro while the grey
  line says *"Filling the empty space..."*, and a Stop pressed meanwhile
  lands when it is done — nothing is added.
- **How long it takes.** A whole-patch Preview takes longer the bigger the
  patch, and grows a little faster than the patch does. With **Membrane
  sheets** on a fast computer it took about 2 minutes on a 24 million voxel
  patch and about 13 minutes near the 100 million voxel limit; a slower
  computer takes longer. **Vessel ridges** and **Thin-faint ridges** take
  longer still (see *Wall type*). Above 20 million voxels the page says so
  first, in amber, with a rough figure for this patch and wall type:
  *"This patch is 73.0 million voxels; a Preview may take about 9 minutes,
  longer on a slower computer. Stop acts between steps."* With a ridge
  wall type the amber line is the ridge one instead (see *Wall type*),
  until its wall picture is kept; after that this line's figure leaves the
  picture out, since it is not built again. While an earlier
  preview is still finishing in the background, a new Preview waits until
  it has ended: the first line says "An earlier preview is still finishing
  in the background. Wait for it to end, then press Preview again." and
  goes by itself once it has.
- The wall picture of the whole patch is built once, at the first Preview,
  even when that Preview is refused (with **Membrane sheets** it is part of
  the times above; **Vessel ridges** and **Thin-faint ridges** add about 4
  minutes for it on a 24 million voxel patch and need two to three times
  the memory), and kept for the next Previews on the same patch and
  run, after Close box or Done too. It is built again only when the wall
  channel, its preparation, the wall type or the spacing change, or after a
  patch switch. A box of up to 20 million voxels keeps its own wall
  picture the same way, for that box (moving or resizing the box builds it
  again); a larger box keeps none: its wall picture is built at every
  Preview.
- When the worker is done, drawing the result takes a visible moment on a
  whole patch; the first line says *"Showing the result…"* meanwhile, the
  grey line *"Drawing the new objects..."*, and the bar steps from the
  drawing to the list.
- **Edge zone** and **Untick objects at the box edge** (under Advanced) are
  hidden on a whole patch: no face of it is a box edge, so nothing is
  unticked for touching one.
- A **Numbers…** slider, or **S**, **M** or **L**, makes it an ordinary box
  again (S, M and L at their own size, see *Placing the box*); on Cells a
  box over the 100 million voxel limit is shrunk to it, and then the first
  line says *"The box is no longer the whole patch; for cells it was shrunk
  to the 100 million voxel limit."*
  When markers were removed too, the first line names the markers instead
  (see *The whole patch* above).
- The whole patch needs the numba package (part of EpiCycle's
  environment). Where it is missing, Preview is grey on a Cells whole patch
  and the first line says "Cells fills a whole patch only with the numba
  package, which is missing from this install. Use boxes." When numba is
  there but cannot write its cache folder it says "Cells fills a whole
  patch only with the numba package, which cannot write its cache folder
  here. Use boxes." (see *How cells grow*).

---

## Clear objects

**Clear objects** sits at the left of **Preview**, in the strip that never
moves. It is there while a box is placed and no result is on screen — it
comes back after **Discard** — and not in the review. Click it for its two
items:

- **Objects mostly inside this box…** takes every object with **more than
  half** of its voxels inside the box. It takes such an object **whole**,
  so a part that sticks out of the box goes too; an object with half or less
  of itself inside stays as it is.
- **Every object in the patch…** takes every object in the patch.

The tool first counts what it would take — *"Counting the objects to
clear…"* — while the rest of the page waits; **Stop** drops the count and
clears nothing. Then it asks, in a red window titled *Clear objects*:
*"Clear the 12 objects that sit mostly inside this box?"* (or *"Clear all
746 objects in this patch?"*). Two more lines are added when they apply:
*"3 of them stick out of the box; those parts go too."* and *"5 of them are
marked Curated or Good."* — clearing takes good objects as readily as bad
ones, so check that number. The last line is the undo rule: *"The main Undo
button brings them back after you close the box, until you switch patches,
run Extract or Commit Triage, or use Reset to Predicted."* For a single
object the window says *"Clear the object
that sits mostly inside this box?"* and speaks of it, not them. **Go back**
clears nothing; **Clear** clears them.

After a box Clear the first line says, for example, *"Cleared 12 objects.
The main Undo button brings them back once this box is closed."*, followed,
when some objects stuck out of the box, by *"Some stuck out of the box —
grow the box, or place another, to fill their space."* If Preview cannot run
yet in this box, the line ends with the reason, as it does before a Clear.
The box stays where it is, so you can go straight on to Preview. With
nothing to take it says *"No object sits mostly inside this box — nothing
to clear."* or *"This patch has no objects — nothing to clear."*

**Undo.** One Clear is one step of the main **Undo** button in the Curation
section (next to Save), however many objects it took, and it puts back the
objects and their rows in the QC table exactly as they were. While a box is
placed every Undo is refused, so press **Close box** or **Done** first,
then Undo. **Close box**, **Discard** and **Undo all** never bring cleared
objects back. The undo steps survive **Save curation**; they are emptied when
you switch patches, run **Extract** or **Commit Triage**, or use Reset to
Predicted. The main Undo
keeps the last 20 steps, so a Clear with 20 later edits on top of it can no
longer be undone. Save keeps only one backup of the mask, always one save
behind — so after a Clear, a Save and a patch switch, the cleared objects
cannot be brought back.

**Numbers.** New objects never take the number of an object you cleared.
The numbers are kept through **Save curation** and **Extract**, and after
you close and reopen napari: the QC table keeps a deleted row for each
removed number above the highest one in use.

If the box or the volume changes between the count and your answer, nothing
is cleared: *"The box changed while counting — pick Clear objects again."*
If the write itself fails nothing is changed either: *"The clear failed —
the reason is in the log. Nothing was changed."*

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
Auto…"*. They are measured for a box or a whole patch alike (on a whole
patch from an even sample of it, so the measurement stays quick): before a
box is placed the card says *"Auto is measured once a box is placed."* A box where every
voxel has the same brightness says *"Nothing to separate in this box —
every voxel has the same brightness."*; a measurement that fails leaves
**Auto** grey and the usual help. While you have not touched the threshold it follows Auto after each
change; once you have dragged the line or typed a value it stays where you
put it. Turning Local contrast on or off changes the scale of the histogram,
so the line goes back to Auto.

**Look for nuclei**, directly under Channel, picks where to look; hover over it
to read: "Where to
look for nuclei: anywhere in the box by brightness, inside each cell of
another run of this patch, or at the markers you place, one click per
nucleus." It starts on *"Anywhere in the box"* — the brightness finder
above, exactly as it always was. The items for another run are described
next; the last item, *"At the markers I place"*, has its own section after
them. While a result is on screen the threshold's help line is hidden; the
result card is the news then.

With *"At the markers I place"* the histogram folds away: your clicks pick
the nuclei, and the threshold only sets where each one's edge is. Its help
line about the red line goes with it; an Auto line such as *"Measuring the
box for Auto…"* still shows. **Threshold**, **Auto** and **Local contrast**
stay.

---

## Nuclei inside each cell of another run

When another run of this patch holds the **cells** (curated or not), the
Nuclei preset can find **the nucleus inside each of those cells** instead
of every bright blob. Pick it under **Look for nuclei**: one item per other
run, for example *"Inside each cell of run ▸ default · cells"* — the run's
name, then its target when it has one, and *(reconstruction, not curated)*
when that run has no curated mask yet. The item is never picked for you.

What it does, as the page says: *"Finds each nucleus by its own edge in this
channel, then gives it to the cell of that run that holds most of it. Needs
a nuclear stain. The threshold is not used here; nuclei outside every cell
are left out."* So the histogram, the threshold, **Auto** and **Local
contrast** go away while it is picked, and the card counts the cells of that
run in the box: *"Counting that run's cells in the box…"*, then for example
*"312 cells of that run in the box."* (on the whole patch: *"812 cells of
that run in the patch."*). The other run is read in the
background (*"Reading that run's objects…"* on the first line meanwhile),
only ever read — never changed, and never shown as a layer. It is read
again for each new box, and dropped when you change the preset or leave
Add.

**No cells run yet?** With no other run of this patch the item is grey —
*"Inside each cell of another run (no other run of this patch)"* — and its
tooltip (hover it in the open list) says how to get one: *"No other run of
this patch holds cells yet. To find nuclei inside your cells: in Step 3
press New Run (a nuclei model), Predict + Reconstruct this patch, open it in
that run, and pick your cells run here."* The card itself stays quiet about
it. The closed list is narrow and cuts a long run name; hover it to read the
picked item in full. If the run you pick is marked as nuclei, you are
probably in the cells run already: *"That run is marked as nuclei, so this
run probably holds the cells. To find the nucleus inside each of these
cells, open this patch in that nuclei run and pick this run under Look for
nuclei."* The objects here are then probably your curated cells, so in that
case no sentence on the card suggests clearing them — for example *"Nuclei
were found, but every one sits on an object already here — open the nuclei
run."* A run counts as marked nuclei when its target (the word after the
run's name in the list) contains *nucle* (Nuclei, nucleus, nuclear).

**The usual way, for every nucleus of a patch from curated cells:** open the
patch in the nuclei run, Add → **Nuclei** → **Look for nuclei ▸ Inside each
cell of run ▸ …** (the cells run) → **Whole patch** → **Clear objects ▸
Every object in the patch…** → **Preview** → untick what looks wrong →
**Accept** → **Done** → Save → run **Extract** and pick the cells run under
**Compare with**.

How it decides, in plain words:

- Each nucleus is found by its own edge — from its own brightest point
  down towards the cytoplasm around it — not by one threshold for the whole
  box, so dim and bright nuclei both come out at their own size.
- Each nucleus goes **whole** to the cell that holds most of it. A nucleus
  less than half inside the cells is left out.
- A nucleus shared between two cells, where the second cell would otherwise
  have none, is split between them at their border.
- Each nucleus is then cut at its own cell's border. One that loses more
  than a quarter of itself to that cut is marked **cut at edge** (the
  edge of its cell).
- A cell that comes out with two nuclei keeps both only when the darkest
  point between them is dark enough (**Gap that splits two nuclei**,
  Advanced) and the second one is not tiny; otherwise they become one. Both
  are marked **two in a cell**.
- A cell with no nucleus yet is searched once more, inside that cell only.
- A cell that already holds an object you keep is skipped, and a nucleus
  that sits a quarter or more on an object you keep is left out — new
  nuclei go only into empty space, as everywhere on this tab.

New nuclei go only into empty space. When the box is mostly taken the card
warns before you Preview, for example *"Only 12 % of the box is empty — new
nuclei go only into empty space. Clear objects first, or open the nuclei
run."*

**Smooth the added objects** (Advanced) starts **off** here — the finder
already tidies each nucleus. Turn it on if you want; your choice is kept
for the session.

**Show what the tool sees** is grey here: *"The nuclei follow that run's
cells; there is no wall picture to show."*

**The spacing.** Every setting here is in microns. When the patch's spacing
was never set, Step 3 uses 1.0 µm as a stand-in and this runs on it anyway;
the amber line under the first one says so (see *The spacing (Cells)*), and
on voxels smaller than 1.0 µm every setting acts smaller than it says until
you set the real size. **Set spacing…** keeps a
voxel size for this patch — 1 / 1 / 1 too, when your voxels really are 1.0
µm — and it works while a box is placed. A real voxel size recorded in Step
1 or in the file counts as set as well. QC Features → Settings… counts
only when you change its Spacing boxes there: pressing OK with them left
at 1 / 1 / 1 (to pick features or the run to compare with) does not
count, and a 1 / 1 / 1 typed there counts only until you open another
workspace or patch folder.

**The channel.** The patch's brightness levels are measured on the
**Channel** you picked as soon as a box is up. A channel with nothing
bright in it (an empty or wrong channel) cannot be used, and the first line
says so before you Preview: *"This channel cannot be used to find nuclei —
the reason is under Look for nuclei."* The run is still fine; pick another
channel and it is measured again.

**Half size.** When the patch is open at half size, the other run is shrunk
to match before it is read, and the card says so in amber before you
Preview: *"Half-size session: the other run is shrunk to match, so its
small or very thin objects can disappear, and new objects are saved back at
full size in 2-voxel steps. For best results, press Close box, untick
Downsample patches 2× on load under Curate, and open the patch again."* Half
size is that tick box in Step 3's **Curate** section; it acts when a patch
is next opened. The micron settings still mean microns there, and
Smoothing counts voxels of the full-size image, so a new object is smoothed
about as much at half size as at full size: a little less, so objects come
out slightly larger. At half size a Smoothing of about 1.2 or below does
nothing. Smallest object, Edge zone and Anywhere's Spacing still count
voxels of the half-size image, so a Smallest object of 100 there is about
800 voxels of the full-size image.

The stage line follows the finder's steps — *"Reading the cells and the
objects already here..."*, *"Smoothing the nuclear channel..."*, *"Measuring
the patch's brightness..."*, *"Finding nuclei..."*, *"Drawing each
nucleus's edge (Stop takes effect after this step)..."*, *"Giving each
nucleus to a cell..."*, *"Splitting nuclei shared by two cells..."*,
*"Checking cells that hold two nuclei..."*, *"Looking again inside empty
cells..."*, *"Tidying the nuclei..."*, *"Leaving out what is already
here..."* — and inside the long steps it counts, for example *"Cell 200 of
760 with a nucleus · Checking cells that hold two nuclei..."*, *"Empty cell
200 of 275 · Looking again inside empty cells..."* or *"Nucleus 400 of 812 ·
Tidying the nuclei..."*. **Stop** acts between the steps and every 200
cells or nuclei; drawing the edges is one step it cannot cut short.

The brightness levels the finder compares against are measured once for
the whole patch, in the background, and used for every box — so a box sees
the same brightness levels as the whole patch. (Which cell a nucleus goes
to is still decided inside the box: a nucleus near the box's edge can go to
a cell that lies mostly outside it.)

---

## Nuclei at the markers you place

Sometimes you want a few particular nuclei, not every bright blob in the
box — or the box holds bright debris you do not want. Pick **Look for
nuclei ▸ At the markers I place** (the last item) and click once on each
nucleus you want. Only the nuclei you click come back.

- **A box or the whole patch.** On a whole patch the markers are placed
  the same way, anywhere in the patch, and the first line reads, for
  example, *"Whole patch 81 × 512 × 512 voxels · 3 objects marked"*.
- The **Markers** card opens under Brightness with the same controls as
  Cells' click markers: **Place markers**, **Ball** (the ball's radius in
  µm), **Undo last**, **Remove all**, **Ctrl + click** on a ball to remove
  it, and the marker list with its numbers, colours and ticks (see *Cells —
  Markers* for the list). The card shows, in two lines, how a click
  works, before and over a result (see *Clicking*). Click once on each
  nucleus you want: the threshold sets its edge, and nuclei you do not
  click are left out.
- **What a click finds.** The threshold decides what counts as nucleus,
  exactly as on Anywhere in the box (with **Local contrast** if it is on).
  Each click takes the bright piece it sits on. A click just beside a
  nucleus still counts, as long as its small ball touches the nucleus. A
  ball that touches two separate bright pieces takes the one under the
  click itself — or, when the click is just off the stain, the one that
  fills more of the ball. With **Separate touching objects** ticked (under
  Advanced, the default) the nucleus you click also stops where it narrows
  towards a neighbour.
- **Touching nuclei — one click, one nucleus.** With **Separate touching
  objects** ticked, one click on a nucleus in a group of touching nuclei
  gives the nucleus you clicked, not its neighbours. It does not always
  manage: about a third of single clicks in a group still come back joined
  to a neighbour, and nuclei pressed together with no pinch between them
  cannot be told apart at all — click each one of them. Unticked, the
  markers work as before: each click takes its whole bright blob, so one
  click on two touching nuclei gives one object, and you click each of
  them to split them at the narrow neck between them. Either way, two
  plain clicks in one nucleus give two objects, split unevenly (the click
  nearer its middle takes most of it): click each nucleus once, and use
  Shift + click (next) to add to one.
- **Adding to an object — Shift + click.** A plain click starts a new
  object, and that object becomes the current one. **Shift + click** adds
  another marker to the current object instead, wherever you click: the
  object grows from all its markers, with nothing cut between them. Use it
  when a nucleus came out too small or is missing a lobe. Every marker of
  the object shows its number and colour, and the list has one row for the
  object, with its seed count; ticking or unticking it ticks every marker
  of the object. To add to an earlier object, Alt + click it or click its
  row first (see *Clicking*). **Undo
  last** — or **Ctrl + Z** in the side view you last clicked in — takes back
  the last marker placed; when it was the object's only marker the object
  goes and no object is current. **Esc** leaves no object current, so the
  next Shift + click starts a new object. Both keys work in the three side
  views while Place markers is on; they never undo a paint of the curated
  labels. A Shift + click on a bright piece that does not touch the
  object's own makes nothing: the marker stays, the rest of the result is
  unchanged, and the first line says "That spot is not touching #12 —
  click it plainly for a new object, or extend from a spot that touches."
  The marker is drawn as a hollow red ball and its object's row says
  "apart". Click it plainly: the refused marker is replaced by a new
  object.
- **Thick slices.** If your slices are two or three times thicker than a
  pixel is wide and you click every nucleus of a group, unticking
  **Separate touching objects** can split them a little better. One click
  on one nucleus of a group still works better ticked.
- **What you do not click is left out** — debris, and nuclei you do not
  want.
- **A click that finds nothing.** A click whose whole ball misses the stain
  makes nothing. After the Preview its row in the list is dim and its
  tooltip says why: "At the last Preview this marker sat on stain below the
  threshold, so it made nothing — move it onto the nucleus or lower
  Threshold under Brightness." A nucleus below **Smallest object**, or one
  that vanished when smoothed, says "At the last Preview this marker's nucleus was below
  Smallest object or vanished when smoothed." While a result is on screen
  the list is hidden, so the line under the result counts them, for example
  "1 marker sat on stain below the threshold and made nothing". The marks
  stay until the next Preview, and go when you change the preset or Look
  for nuclei.
- **Checked like every new object.** A clicked nucleus at the box edge, or
  small, starts unticked, and the first line says why. Each new nucleus
  takes its marker's number and colour.
- **Advanced.** **Separate touching objects** is the markers' on / off for
  one click, one nucleus. It is one tick shared with Anywhere in the box:
  unticking it on either turns it off for both. It stays off while you
  work on: **Add another** and **Whole patch** keep it as it is. Only after
  **Close box** or **Done** does the next box you place start with it
  ticked again (and the markers' Spacing back at 5). While it is ticked
  the markers have their own **Spacing** under it (5 by default;
  Anywhere's stays 3); unticked, it is hidden, as it does nothing then.
  Spacing is how far apart the middles of two nuclei must be for them to
  count as two. It counts voxels of the full-size image across, and the
  same distance in microns between slices, so in a half-size session it
  still means the same distance in the tissue. Without a voxel size set
  for the patch, it is counted in voxels in every direction. The default
  suits most nuclei. Much higher joins close neighbours to the nucleus you
  click; much lower does not separate more — it can let a click spread
  through rough, uneven stain. The Advanced summary reads as on Anywhere:
  separate touching, then the markers' spacing; unticked, only one per
  blob. The ball is in microns, so at the default spacing the amber
  spacing line shows (see *The spacing (Cells)*).

Place markers turns itself off when you change tab, preset, source or Look
for nuclei. The markers you placed stay, with their numbers and ticks, when
you switch to Cells or to another Look for nuclei item. On another Look for
nuclei item they are hidden, and they come back when you pick the markers
again. On Cells their balls stay on screen: they are Cells' click markers
again.

---

## Clicking

With click markers (Nuclei ▸ At the markers I place, Cells ▸ Click
markers) the card always shows how a click works, in two lines, however
many markers you have placed and with a result on screen too (with
**Update as I click** on, a result is there from the first click): "Click: a new object · Shift+click: add to
the bold one · Alt+click it or its row: make it bold · Ctrl+click: remove
· Ctrl+Z: undo last". The same clicks and keys work on both, in the three
side views, while Place markers is on.

- **A click** places a marker and starts a new object, and that object
  becomes the *current* one: its row in the list is drawn in **bold**.
- **Shift + click** adds another marker (a seed) to the current object,
  wherever you click: the object grows from all its seeds. With no object
  current it starts a new object, as a click does. A seed whose bright
  piece (or cell) does not touch the object's own makes nothing: it stays
  where you put it, drawn as a hollow red ball in the main view and the
  side views, and the first line says "That spot is not touching #12 —
  click it plainly for a new object, or extend from a spot that touches."
  Click it plainly: the refused marker is replaced by a new object. Undo
  last removes it instead.
- **Alt + click** on an object — its coloured voxels in a result, or its
  ball — makes that object the current one. **A click on its row** in the
  list does the same and always works, whatever your computer does with
  Alt. Neither places a marker nor changes the result.
- **Ctrl + click** on a ball removes that marker.
- **Undo last**, or **Ctrl + Z** in a side view, takes back the last marker
  placed; when it was the object's only marker, the object goes and no
  object is current.
- **Esc** leaves no object current, so the next Shift + click starts a new
  object. It is not on the card's two lines; the tooltip of **Place
  markers** names it with the others.

**One row per object.** The list has one row for each object, not for each
marker: an object of one marker shows where it sits (z / y / x), and an
object grown from several shows how many, for example *Object 412 · 3
seeds*. The **Markers** title and the first line count objects too: two
objects made of three markers read "2 placed", and the first line reads,
for example, "Box 40 × 125 × 125 voxels · 2 objects marked". While a
result is on screen
its rows carry the same seed count, before the size.

**What each seed made.** After a result, a row names in a small gold
word the seeds of that object that made nothing, before its edge or small
mark: "apart" (not connected: a Shift + click on a bright piece, or a
cell, that does not touch the object's own), "dim" (nothing bright here: a
marker on stain below the threshold) or "on kept" (inside a kept object: a
marker on an object you kept). The row's tooltip spells the word out. A
refused seed is drawn as a hollow red ball (see the Shift + click line
above).

---

## Cells — Markers

A marker is where a new object starts from. Each marker from a run or from
the nuclei is one object; with click markers a plain click starts a new
object and **Shift + click** adds another marker to the current one (see
*Click markers* below). The **From** picker offers three sources:

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
  Each click becomes a small ball, drawn on screen. **Ball**, beside Place
  markers, is the ball's radius: 1.0 µm by default (a ball about 2 µm
  across), up to 5 µm. A new value redraws the balls already placed once
  you stop changing it. **Undo last** removes the last one, **Remove all**
  removes them all, and **Ctrl + click** on a ball removes that one. The
  **Markers** title says how many objects are placed — *"7 placed · 5
  ticked"* when some are unticked — and the first line says it too, in its
  own words. **Shift + click** adds a marker to the current object (the
  last one you started, or the one you picked), as on Nuclei: one cell
  grows from all its markers (see *Clicking*). A
  marker whose cell does not touch the rest of its object makes nothing
  and says so, as on Nuclei. With Grow by ▸ Spread from each marker,
  each marker grows its own reach, so a second marker can make the cell
  larger.

The source you pick stays picked for the next box or whole patch on this
patch, after Close box, Done or Undo all; with a run picked, that run is
read again when the box starts. A whole patch on Cells lists the same
sources as a box. It goes back to the default when you switch patch or run,
or close the annotator.

**The marker list.** Every object gets a row under Undo last and Remove
all: *Object 412* and where its marker sits (z / y / x) — or, for an object
grown from several markers, how many (*3 seeds*) — with a colour swatch. The
number is the object the marker will become, and its ball is drawn in that
object's colour — after Preview the new object has the same number and the
same colour, so a marker and the object it grew read as one. Its tooltip:
"Each object has a row in its colour, with its seed count when it grows
from several markers. Click a row to make that object current; untick one
and the next Preview grows nothing from it."

- An unticked marker's ball disappears from the views and the next Preview
  grows nothing from it. Unticking a marker leaves a gap in the numbering; the
  other colours do not change. Removing a marker in the middle leaves a gap
  too; Undo last, or removing the last one, gives its number back.
- Colours repeat every 49 numbers, which is why each row shows the number.
- The list shows three rows and scrolls. It is hidden while a result is on
  screen — the result list is the list then, in the same colours — and comes
  back, ticks and all, after **Discard**. So: untick an object in the result
  to leave it out of this Accept; or Discard, untick its marker and Preview
  again to fill the box without it.
- With every marker unticked, Preview is grey and the page says "Tick at
  least one object to grow from — all 3 are unticked."; with some unticked the box
  line says so, for example "Box 40 × 125 × 125 voxels · 7 objects marked ·
  5 of 7 ticked" (objects are counted: the markers of one object count
  once).
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
  when you change tab, preset or source; pressing Whole patch leaves it as
  it was, and on a whole patch a click places a marker as in a box.
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
  removed with Undo last or Remove all, or ticked again first.
- Moving the box keeps the markers that are still inside it and removes the
  others — unticked ones too — and says so, for example *"3 markers were
  outside the moved box and were removed."* Leaving a whole patch for a box
  asks first, and then removes every marker (see *The whole patch*).

**On a patch with only one channel** the nuclei source is grey — *Nuclei
masks (needs a second channel)* — because it needs a channel of its own.
When the run source is grey too, the card shows an amber line: *"No automatic
marker works on a membrane stain alone. Click once inside each object you
want. If dim cells come out empty, lower Dark below or untick Treat dark
space as outside."* Click markers are the way to go there. A whole patch
leaves that line off: Click markers are already picked there.

---

## Cells — Walls

- **Channel** — the channel the membrane is bright in. It must be a
  different channel from the nuclei channel. Each name in the list (here,
  and in every channel list of the page) has a small swatch of the colour
  that layer is drawn in, so you can tell the channels apart; a colour you
  change in napari shows on the swatch the next time the list is filled (a
  new box, or another tab).
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
  each one. The two ridge types build their wall picture slowly: on a fast
  computer about a minute for every 6 million voxels, on top of the rest
  of the Preview, so a whole-patch Preview took about 6 minutes in all on
  a 24 million voxel patch (about 4 of them for the wall picture). They
  need two to three times the memory of **Membrane sheets** (roughly 20 GB
  near the 100 million voxel limit). The ridge figures above 24 million
  voxels are estimates, not measured. On a whole patch, until its wall
  picture is kept, the amber line says so before you press Preview, with a
  rough figure for the whole Preview on this patch: *"Vessel ridges is slow
  on a whole patch: a Preview takes about 6 minutes here, most of it
  building the wall picture. Membrane sheets is much faster."* From about
  40 million voxels it names the memory too, for example at 96 million
  voxels: *"Thin-faint ridges is slow on a whole patch: a Preview takes
  about 29 minutes here, most of it building the wall picture, and needs
  about 19 GB of memory. Membrane sheets is much faster."* On a box
  of 2 million voxels or more it says so too. Up to 20 million voxels a box
  keeps its wall picture like the whole patch, so the line shows until the
  picture is kept: *"Vessel ridges is slow on a box this large: its wall
  picture is built once, then kept for this box. Membrane sheets is much
  faster."* Over 20 million voxels the picture is built at every Preview:
  *"Vessel ridges is slow on a box this large: its wall picture is built
  again at every Preview. Membrane sheets is much faster."*
- **Tissue outline** — where new cells may grow; see *The tissue outline*
  below. A help line under it says what the pick does (it is hidden while a
  result is on screen).
- **Show the tissue outline** (shown when an outline is picked) draws the
  outline's edge in the main view and in the three side views, and keeps it
  there while new objects are on screen, so you see what they grew to. The
  line under it says *"Drawn in the main view and the three side views."*, or
  *"Measuring the tissue outline…"* while it is measured and *"Drawing the
  tissue outline…"* while it is drawn.

---

## How cells grow

**Grow by**, under Advanced (Cells only), picks how each marker becomes a
cell, in a box and on a whole patch alike (on a whole patch, Fill to the
walls keeps tissue far from every marker empty; see *Cells: the whole
patch*).

The row's tooltip says it in short: *"Fill: the fill runs to the walls, the
box edges and dark space, as before. Spread: each marker grows along its
easiest path and stops where a neighbour got there first, at a wall, or at
the growth limit. Pick Spread when the fill leaks through a gap in the walls
or merges neighbours."*

- **Fill to the walls (as before)** — the fill used so far. It is the
  default on a box.
- **Spread from each marker** — each marker grows outwards along its
  easiest path; a voxel goes to the marker that reaches it most cheaply.
  Crossing a bright wall is costly, so a cell stops at its membrane, where
  a neighbour got there first, or when it has spent its growth limit.
  **When to pick it:** when the fill leaks through a gap in the walls into
  a neighbour's space or out of the tissue, or when two neighbours come out
  merged into one object.

With Spread picked, three more rows appear:

- **Wall cost** — *"How costly a bright membrane is to cross. Higher keeps
  cells inside their walls; too high and cells stop short. 0 ignores the
  walls: each cell is the space closest to its marker."* 40 by default (0 to
  100).
- **Growth limit** — *"How far a cell may spread: a budget, not a distance.
  Plain cytoplasm uses it up about 6 µm from the marker; a bright membrane
  costs nearly the whole limit per µm, so a cell stops in its wall. Lower it
  if cells grow past their walls."* 28 by default (2 to 40). In very dark
  cytoplasm the same limit lets a cell go further, about 10 µm.
- **New cells include the membrane at their edge** — with Spread, and on
  a whole patch too; ticked by default:
  *"New cells include the strip of membrane at their edge (up to 1 µm).
  Untick it if your cells are drawn without their membrane."* Neighbouring
  new cells then touch across the membrane; a new cell next to an object you
  kept stops one voxel short of it.

The first Spread on a computer — or the first whole patch that includes
the membrane — sets the cell-growth engine up (about 10 seconds, once);
while it does, the first line says *"Preparing the cell-growth engine for
its first run on this computer — about 10 s, once."* Where this install
cannot run it, its item reads *Spread from each marker (not available
here)* and is grey, and the Advanced help line says why. When numba is
missing: *"Spread from each marker cannot run here: the numba package it
needs is missing from this install, so Cells fills boxes only. Reinstall
EpiCycle's environment to get it back."* When numba is there but cannot keep
its compiled code (a lab install in a folder you cannot write to): *"Spread
from each marker cannot run here: numba, the package it needs, cannot write
its compiled code to its cache folder (this install's folder may be
read-only), so Cells fills boxes only. Point the cache at a folder you can
write: set the NUMBA_CACHE_DIR environment variable to it, then restart
napari."* The full reason goes to the log. The check is made once per
napari session; a Preview that finds the engine cannot be set up says the
same and is not tried again.

**Spread with no outline and dark space not treated as outside** has
nothing to stop a cell in open background but its growth limit, and the
Advanced help line says so: *"With no tissue outline and dark space not
treated as outside, Spread grows each cell into open background until the
growth limit stops it."* On a whole patch filled to the walls only the
background stops a cell there, and the amber line under the first one says
so: *"Dark space is not
treated as outside and there is no tissue outline, so cells grow a few µm
into open background — tick it in Advanced, or pick a Tissue outline under
Walls."* On a whole patch under Spread that amber line gives the Spread
sentence above instead (one line, not two that disagree), and the Advanced
help line does not repeat it.

---

## The tissue outline

The **Tissue outline** row of the Walls card says where the tissue is: new
cells grow only inside it. It removes nothing and is never written — an
object you kept outside it is left as it is. Its three items:

- **Outline of the objects you cleared** — *"New cells stay inside the
  outline of the objects you cleared on this patch."* It is the shape of
  every object the Clears still in force removed, closed over small gaps.
  Until a Clear is in force on this patch it reads *Outline of the objects
  you cleared (clear some first)* and is grey. A Clear stays in force as
  long as its objects are gone — also after the main Undo can no longer
  reach it (twenty edits later, or after Extract). Undo a Clear and its
  objects come back and leave the outline; when the last one comes back
  while it is the pick, the pick falls back to the shrink-wrap, on a box and
  a whole patch alike, and the first line says so: *"The objects you cleared
  are back, so there is no outline of them any more — Tissue outline is now
  Shrink-wrap around the stain."*
- **Shrink-wrap around the stain** — *"New cells stay inside a smooth
  outline around the stained tissue. Gaps narrower than Bridge gaps
  (Advanced) are closed over."* It is measured from every channel at once,
  with **Dark below** deciding what counts as stain. The default on a box
  and a whole patch alike, while no Clear is in force (with one, the
  outline of the objects you cleared).
- **No outline** — *"Only the box edges and dark space stop the cells."*
  Pick it to grow cells the way boxes did before the outline: your pick is
  kept for every box and the whole patch until you leave the tab (Accept
  or Close box); the next session starts from the default again, like
  every other control here.

The outline is measured **once per patch**, in the background, over the
whole patch — a box is always given its part of that one outline, so two
boxes side by side agree at their shared face. While it is measured the
first line says *"Measuring the tissue outline…"* and Preview waits. It is
kept for the next box on the same patch and run — after Close box, Done or
+ Add another too — and measured again only when Bridge gaps, Dark below or
the channels change, when the Clears in force change, or after a patch
switch. If it cannot be measured at all (the reason goes to the log), the
first line says *"The tissue outline could not be measured — the reason is
in the log. Pick another Tissue outline under Walls."*

With an outline picked, markers mostly outside it are skipped and counted
under the result: *"3 markers lay outside the tissue outline and were
skipped"*. When more than 5 % of the run's markers lie outside, an amber
line warns before you press Preview: *"12 % of the markers lie outside the
tissue outline and will make nothing — raise Bridge gaps, lower Dark below,
or pick another outline."* (a faint basal face can break the shrink-wrap).
If every marker left lies outside, the Preview is refused: *"Every marker
left lay outside the tissue outline — raise Bridge gaps or lower Dark below
under Advanced, or pick another outline."* Bridge gaps and Dark below shape
only the shrink-wrap; on the outline of the objects you cleared the two
lines say *"12 % of the markers lie outside the outline of the objects you
cleared and will make nothing — pick another Tissue outline under Walls if
they should make cells."* and *"Every marker left lay outside the outline
of the objects you cleared — pick another Tissue outline under Walls."*

**The outline decides what is outside**, so under an outline **Treat dark
space as outside** and **Dark space reach** are grey, and the Advanced help
line says *"The tissue outline decides what is outside, so Treat dark space
as outside and Dark space reach are not used. Dark below still sets the
outline."* The
amber dark-space line under the first one is not shown then.

**Show the tissue outline** draws the outline's edge — a thin skin, about a
micron thick, at least one voxel on every axis — in the main view and in
the three side views. In the main view it is a faint, see-through cyan. In
a side view the outline's edge is painted in cyan at half strength wherever
that skin crosses the slice on screen, so it shows over bright stain too:
scroll the slices and the edge follows the tissue, so you can see where the
outline was drawn before you press Preview. On a slice that runs along the
top or the bottom of the tissue (most often in the top-down view) the skin
lies flat in that slice, so the whole tissue there takes a cyan tint
instead of showing a thin edge; scroll a few slices on and the edge comes
back.

It is hidden in every view while a result is on screen, and removed from
every view when you untick it, pick No outline, Accept, Close box, switch
patches or close the annotator; the tick goes back off when the session
ends. After an Accept, **+ Add another** draws it again (the tick is kept),
and **Stop** leaves it as it was. Drawing it holds the window for a moment.
The line under the tick says what is true: *"Measuring the tissue
outline…"* until the skin is measured, *"Drawing the tissue outline…"*
while it is drawn, *"Drawn in the main view and the three side views."*
once it is, *"Drawn in the main view; the side views could not show it (the
reason is in the log)."* if the side views could not take it, and *"Not
drawn: the outline could not be measured."* when it could not be. The
outline stays drawn over a result.

**On a patch with only one channel**, picking another run's objects as
markers shows an amber line that names only what is on the page then —
with Fill picked and the shrink-wrap, for example: *"Only one channel. If it
is the nuclear stain, pick Grow by ▸ Spread from each marker under Advanced
and set Wall cost to 0 and Bridge gaps to 16 µm — walls read from nuclei
squeeze cells down to their nuclei."* With Spread already picked it says
only to set Wall cost to 0; without the shrink-wrap it does not name Bridge
gaps (that row is not on the page then). A whole patch says the same:
Grow by, Wall cost and Bridge gaps are under Advanced there too. On a
whole patch Bridge gaps already starts at 16 µm with one channel (10 µm
with two channels or more), because one channel leaves wider
dim gaps inside the tissue. Where Spread cannot run, the line is not shown.

**Half size, on a Cells box too.** When the patch is open at half size and
the markers come from another run (**Objects from run ▸ …**), the same
amber line as for Nuclei inside each cell shows before you Preview, on an
ordinary box as on a whole patch: *"Half-size session: the other run is
shrunk to match, so its small or very thin objects can disappear, and new
objects are saved back at full size in 2-voxel steps. For best results,
press Close box, untick Downsample patches 2× on load under Curate, and open
the patch again."*

**One amber line at a time.** The page has one amber slot: when one of the
lines above shows, the spacing line and the dark-space line give way — the
spacing line then sits in the Advanced help line, and the dark-space reason
is there already.

### Nuclei: the Tissue outline card

On Nuclei the tissue outline is an option, **off by default**: nuclei come
out exactly as before unless you pick one. Its card sits under the
Brightness card (and under the Markers card when you place markers) with
**Look for nuclei ▸ Anywhere in the box** or **At the markers I place**, on a
box or a whole patch, before a result. It is not offered on **Inside each
cell of …** (the cells there already keep each nucleus in), and it is
hidden while a result is on screen. With **Update as I click** on, the
markers on a whole patch with an outline picked run the one-go way at each
click (the quick way has no outline). It opens closed, its title saying the
choice — *Tissue outline: none* — and, like the other cards, opening it
closes the card that was open; on **Anywhere in the box** an open Tissue
outline card folds the histogram (close it to see the histogram again).

The card has its own **Tissue outline** row with the same three items as on
Cells, and its own **Show the tissue outline** tick: what you pick for
Nuclei does not change Cells' pick, and the other way round. Its default is
always **No outline**, on a box and a whole patch alike — it is never picked
for you, and it goes
back to that when the session ends. What each item does is in its tooltip
in the list:

- **Outline of the objects you cleared** — *"New nuclei are kept only
  inside the outline of the objects you cleared on this patch."* Grey until
  a Clear is in force, as on Cells; when the objects you cleared come back
  while it is the pick, the pick falls back to No outline and the first line
  says so.
- **Shrink-wrap around the stain** — *"New nuclei are kept only inside a
  smooth outline around the stained tissue. With a nuclear stain alone, lone
  nuclei can fall outside it. Gaps narrower than Bridge gaps (Advanced) are
  closed over."* With a membrane channel as well the stain fills the tissue
  and the outline behaves as on Cells. With a nuclear stain alone, a crowded
  field of nuclei is closed over into one piece and kept, but a nucleus on
  its own stays a small piece and is left out of the outline — so lone
  nuclei outside the crowd fall outside it.
- **No outline** — *"Nothing is left out for lying outside the
  tissue, as before."*

It is **one outline for both presets**: Nuclei and Cells ask for the same
mask, measured once per patch in the background with the same Dark below
and Bridge gaps, so a box on Cells after Nuclei (or the other way round)
does not measure it again. While it is measured the first line says
*"Measuring the tissue outline…"* and Preview waits; if it cannot be
measured, *"The tissue outline could not be measured — the reason is in the
log. Pick another Tissue outline."*

How it is used:

- **Anywhere in the box** — a nucleus with at least 70 % of it outside the
  outline is left out; the others are cut to the outline. A nucleus the cut
  splits keeps its largest piece, and one the cut leaves below **Smallest
  object** is left out and counted with the ones outside. The count under
  the result says how many: *"1 nucleus lay outside the tissue outline and
  was left out"* (or *"4 nuclei lay outside the tissue outline and were left
  out"*). If every nucleus found lay outside, the Preview is refused:
  *"Every nucleus found lay outside the tissue outline — raise Bridge gaps
  or lower Dark below under Advanced, or pick No outline."* (on the cleared
  outline: *"Every nucleus found lay outside the outline of the objects you
  cleared — pick another Tissue outline."*).
- **At the markers I place** — the outline only cuts: a nucleus you
  clicked is never left out for lying mostly outside, only trimmed to it
  (one piece: the one holding your click's ball, else the largest). A
  marker whose whole nucleus lies outside makes nothing, and so does one
  whose nucleus the cut leaves below **Smallest object**, or leaves as a
  sliver that vanishes when smoothed; its row in the
  marker list says *"At the last Preview this marker's nucleus lay outside
  the tissue outline and was left out."* and the count under the result
  counts it. When no marker makes anything, the Preview is refused with
  the reason most of the markers share.

Kept objects are never touched. **Show the tissue outline** draws the same
skin as on Cells, with the same line under the tick, from Nuclei's own
tick; the session's end unticks both. At the never-set 1 µm spacing the
amber spacing line shows with an outline picked, as on Cells.

---

## Advanced

At the bottom of the page, closed by default, with a one-line summary of the
current values. It appears once a box is placed. Each preset shows only its
own rows, and a row that cannot apply is hidden, not greyed (the summary
leaves it out too). You will not often need it.

**Cells only** (in microns, so they mean what they say only once Step 3's
Spacing is set — see *The spacing (Cells)*):

- **Grow by**, and with Spread **Wall cost**, **Growth limit** and **New
  cells include the membrane at their edge** — see *How cells grow*. On a
  whole patch a line above them says what fills the patch under the Grow by
  pick (see *Cells: the whole patch*).
- **Shrink run objects** (shown with **Objects from run ▸ …**) — each object
  of the other run is shrunk by this before it becomes a marker, so
  neighbours do not touch. Objects too small to survive the shrinking are
  skipped.
- **Treat dark space as outside** — ticked by default. Dark space (no signal
  in any channel) then stops the fill, the way the box edges do, so an object
  next to open space does not spill out into it.
- **Dark space reach** — the fill may reach this far into dark space before
  it stops.
- **Dark below** — signal below this share of the patch's usual brightness
  counts as dark space. It also sets the shrink-wrap outline. Lower it if
  dim cells come out empty. It stays live under a tissue outline (the other
  two Dark rows grey there — see *The tissue outline*).
- **Bridge gaps up to** (shown with the shrink-wrap) — *"The tissue outline
  closes over gaps narrower than this. Keep it below the narrowest fold,
  lumen or wound you want left open."* 10 µm by default, 16 µm on a patch
  with one channel (0 to 30).

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

**Nuclei only** (Look for nuclei ▸ Anywhere in the box, and At the markers
I place):

- **Separate touching objects** — "Splits objects that touch, so a marker
  takes the nucleus it sits on, not its neighbours. Off: one object per
  bright blob; at the markers, two clicks in one blob still split it." One
  tick for Anywhere and the markers: unticking it on either turns it off for
  both, until **Close box** or **Done** (Add another and Whole patch keep
  it off).
- **Spacing** — on Anywhere, the closest two object centres may be, in
  voxels. Lower gives more, smaller objects (3 by default). On the markers
  it is the markers' own box (5 by default): "How far apart the middles of
  two nuclei must be for them to count as two, in voxels of the full-size
  image across (the same distance in microns between slices). The default
  suits most nuclei; much higher joins close neighbours to the nucleus you
  click; much lower can let a click spread through rough, uneven stain."
  Without a voxel size set for the patch, it is counted in voxels in every
  direction. It shows only while Separate touching objects is ticked. Each
  keeps its own number.
- **Local contrast strength** — how strongly Local contrast evens the box
  out.

On **Look for nuclei ▸ At the markers I place** the summary reads as on
Anywhere, with the markers' Spacing: separate touching, then the spacing;
unticked, only one per blob. The ball's size, **Ball**, is on the Markers card, not
here.

**Nuclei with the shrink-wrap** (the Tissue outline card's **Shrink-wrap
around the stain**, on Anywhere or the markers) adds the two rows that shape
it, above the Nuclei rows, and the summary names Bridge gaps:

- **Dark below** — signal below this share of the patch's usual brightness
  counts as dark space; here it sets the shrink-wrap outline and nothing
  else. Lower it if dim tissue falls outside the outline. It is always live
  here, whatever the remembered **Treat dark space as outside** tick says
  (that tick and **Dark space reach** are Cells' and are not shown on
  Nuclei).
- **Bridge gaps up to** — as on Cells; 16 µm by default on a patch with one
  channel.

Both are the same settings as on Cells: the outline is one for both presets.

**Nuclei, inside each cell** (these three replace the three above; in
microns, from Step 3's Spacing):

- **Nucleus radius** — *"The usual radius of a nucleus. It sets the size of
  the bright spots the finder starts from."* 2.5 µm by default (1 to 10 µm).
- **Nucleus edge** — *"Where each nucleus's edge is drawn: this share of the
  way from the cytoplasm up to the nucleus's own brightest point. Lower
  draws bigger nuclei."* 0.25 by default.
- **Gap that splits two nuclei** — *"Two nuclei in one cell stay separate
  only when the darkest point between them is below this share of the
  dimmer one; otherwise they become one nucleus."* 0.70 by default.

**Both presets:**

- **Untick objects at the box edge** and **Edge zone** — an object that
  reaches within this many voxels of the box edge was probably cut by the
  box, so it starts unticked. A lone object stays ticked. Both are hidden on
  the whole patch, which has no box edge.
- **Smallest object** — objects smaller than this many voxels are left out.
  0 keeps every one.
- **Smooth the added objects** and **Smoothing** — smooths each new object's
  surface. This happens **in the preview**, so what you see is exactly what
  Accept writes. Smoothing is in voxels of the full-size image: 1.5 is
  gentle. In a half-size session it smooths each object about as much as
  at full size, a little less, so objects come out slightly larger; there a
  Smoothing of about 1.2 or below does nothing. Smoothing can make a very
  small object vanish; those are left out and counted (see below).
- **Update as I click** (Nuclei ▸ At the markers I place, and Cells ▸
  Click markers; on by default) —
  *"Finds the objects again after every click and setting change, so you
  see the result without pressing Preview. Preview still works."* In a box,
  each marker you place or remove is run a moment later: the objects on
  screen stay until the new ones replace them, and nothing on the page goes
  grey while it runs. A setting you change clears the result until the next
  update draws it. Preview still works with the tick on. Accept pressed in
  that moment waits for the last click — S1 reads *"Accepting once the last
  click is drawn."* and the button reads **Accepting…** — and no new marker
  can be placed until it is done (*"Wait — Accept is finishing; place more
  markers after it."*). If the result changes before it is drawn, S1 says
  *"Accept was cancelled — the result changed before it was drawn; press
  Accept again."* **Clear objects** is refused until the update is drawn:
  *"Wait — the last click is still being drawn; Clear once it appears."*
  Nuclei markers on a whole patch update as you click too: the first update
  reads the patch once and takes longest; after it, each click finds again
  only the nuclei its markers touch, and only those are redrawn. With the
  tick on, Preview on a whole patch works the same way, and pressing it
  when the result already matches your markers does nothing. Turning the
  tick on or off can move a few voxels inside a nucleus you placed several
  markers on (where two of them meet). In a Cells box of up to 20 million
  voxels the cells are found again after every click; the first update of
  a box takes longest, because it also reads the walls and prepares the
  box (on a large box it can take many seconds, and it cannot be stopped
  once it runs), and after it each click fills again only the part of the
  box it changed; a Preview pressed when the result already matches your
  markers and settings does nothing. The box's edges count as outside, as
  in a Preview (a box edge cuts through tissue). Cells ▸ Click markers on a
  whole patch of up to 20 million voxels — the same limit — is updated as
  you click too when the walls are Membrane sheets or Intensity edges. It
  is then filled the way a box is:
  every marker fills to the walls, with no background marker, and **New
  cells include the membrane at their edge** is not applied. A cell at the
  edge of the patch fills to the edge. The first update prepares the patch
  once and takes longest: the grey line says *"Preparing the patch..."*
  and then *"Building the wall picture..."* the first time, and a patch of
  more than 5 million voxels says so in amber until it is prepared (*"This
  patch is 18.0 million voxels; the first update prepares it once and may
  take a while. Later clicks are quick."*); later clicks are quick, and a
  Preview pressed when the result already matches your markers does
  nothing. With the tick off, a Preview
  fills it with a background marker instead, which keeps tissue far from
  every marker empty — the two results can differ. A box or a whole patch
  of more than 20 million voxels is not updated as you click (*"Press
  Preview to see the cells — this box is over 20 million voxels."*, or
  *"Press Preview to see the cells — this patch is over 20 million
  voxels."*), nor is a whole patch with other walls. Such a box, like such
  a patch, also says in amber how long a Preview may take, since a Preview
  fills it in one go: *"This box is 25.0 million voxels; a Preview may take
  about a minute, longer on a slower computer. Stop acts between
  steps."* (a rough figure from a timed box fill, by Grow by's pick). When the quick
  update cannot take a click — flat spots in the walls, for example — a
  box or a patch of up to a million voxels is filled whole instead (about
  half a second), and the next click updates it again. Above a million
  voxels that whole fill would take many seconds and could not be
  stopped, so the click that finds out runs nothing, and so does a box
  with walls other than Membrane sheets or Intensity edges, or with Grow
  by on Spread from each marker (*"Press Preview to see the cells — this
  box is not updated as you click."*, or *"Press Preview to see the cells
  — this patch is not updated as you click."*). The flat spots can come
  from the spacing: the tissue outline's smallest piece is a volume in
  µm³, so at the default 1.0 µm, with the spacing never set, debris in the
  dark around the tissue can count as tissue, and the region waits for
  Preview where its true spacing might not. Then S1 says so instead:
  *"Press Preview to see the cells — at the default 1.0 µm spacing this
  box is not updated as you click; Set spacing… in the Session list may
  change that."* (the same words for a patch). The balls still appear as
  you click, and Preview finds the cells; once the settings, the walls or
  the objects change, clicks update the cells again. The quick update is
  itself made ready once per session, a few seconds after the first box
  is placed: a click on a box or a patch of more than a million voxels
  that comes before then waits — S1 says *"Preparing the fill — your
  clicks will be drawn in a moment."*, the balls stay — and the cells
  appear by themselves once it is ready, with every click placed
  meanwhile. That wait is not the step aside above: no Preview is needed,
  and a Preview pressed during it fills the region as usual and keeps
  the clicks updating afterwards.
  **The reason under the tick.** While a Cells click is not updated as
  you click for any of these reasons, a short grey line directly under
  **Update as I click** says why, in the words of the control that
  changes it (the first line says the reason; this one names what lifts
  it): *"Not live over
  20 million — tick Always update as I click."*; *"Not live here: Wall type —
  tick Always update as I click."* (walls other than Membrane sheets or
  Intensity edges); *"Not live here: Grow by — tick Always update as I
  click."* (Spread from each marker); *"Not live here: flat spots — tick
  Always update as I click."*; *"Not live here: default spacing — Set
  spacing… may help."* (the flat spots at the default 1.0 µm); *"Not live
  here: the quick fill could not run — press Preview."* (it failed; the
  console says why); and *"Preparing the fill…"* while the clicks wait.
  The line is not there while your clicks are updated, with the tick off,
  or while **Always update as I click** is on.
- **Always update as I click** (Cells ▸ Click markers, shown only while
  Update as I click is on; off by default, and off again every time napari
  starts) — *"Runs the quick fill even where its flat spots could place a
  few tied voxels differently from Preview, and where the quick fill cannot
  run at all, redoes the region a moment after you stop clicking. That may
  take minutes on a large region or a slow Wall type; the first line says
  how long."* It
  switches off the limits above, at two costs. **Flat spots:** where the
  quick update would step aside for flat spots in the walls, it keeps
  running instead. Where two cells grow into the same flat spot, a few
  voxels there can go to the other cell than a Preview would give them; a
  Preview pressed with the tick on still fills the region exactly (it does
  not reuse the quick update's answer). **A slower update:** where
  the quick update cannot run at all — walls other than Membrane sheets or
  Intensity edges, Grow by on Spread from each marker, a box or a patch of
  more than 20 million voxels, or a quick update that failed — the region
  is filled whole, as a Preview fills it, about 0.7 s after your last
  click. While it runs the first line says roughly how long it takes,
  estimated from the region's size, its Wall type (Vessel ridges and
  Thin-faint ridges build a slow wall picture, counted until it is kept
  for the region) and Grow by: *"Updating the region — about a minute
  here."* (a box of 4.2 million voxels with Vessel ridges took about a
  minute), *"Updating the region — about 35 s here."*, or, under ten
  seconds, *"Updating the region — this takes a few seconds here."* The
  figure is rough, longer on a slower computer; the console prints how
  long each update really took. Nothing on the page goes grey, **Stop**
  stays live and stops it — and the result drawn for the earlier clicks
  goes with it (press Preview, or click again, to fill the region).
  A click or a setting changed while it runs makes that update stale: it
  is cancelled (a fill to the walls is cut between its steps, Spread
  between its tiles; the step in progress finishes first), and one more
  update follows for everything placed meanwhile, reusing the wall picture
  the cancelled update had built. A Preview pressed during or
  after an update on the same markers and settings runs nothing — the
  update is the Preview's own fill. None of the Press Preview lines above,
  nor the line under the tick, appear while the tick is on; a region over
  the Preview limit (100 million voxels) is not updated either way. The
  quick update being made ready (*"Preparing the fill — your clicks will
  be drawn in a moment."*) is still a short wait with the tick on: the
  region is not filled whole meanwhile. Turning the tick on or off keeps
  a box already prepared for the quick update, unless it was kept only
  because of the tick (prepared despite flat spots, or the flat spots'
  whole fill).

---

## Preview, Stop and the stage line

**Preview** sits just above the two buttons at the bottom, so it is always in
reach. It reads **Preview again** once a result is on screen. **Clear
objects** sits to its left while a box is placed (above). Beside it are
**Stop**, a busy bar and one grey line saying what the tool is doing now —
*"Reading the box..."*, *"Finding markers..."*, *"Building the wall
picture..."*, *"Filling the empty space..."* (Spread: *"Spreading from each
marker..."*), *"Smoothing the new objects..."* (Nuclei: *"Finding bright
objects..."*). The strip is always drawn, dimmed while nothing runs, so the
page never jumps. While **Clear objects** is on screen and nothing runs, the
grey line says what that button is for: *"Clear makes room by deleting old
objects."* When something runs or is being measured, its own words take
the line; the Clear line comes back when it ends.

The bar tells you what kind of wait it is:

- **Moving to and fro** — the work runs in the background and the napari
  window stays usable: a preview of a box, or the box, the patch, the
  tissue outline or another run being read (*"Measuring the box..."*,
  *"Measuring the patch..."*, *"Measuring the tissue outline..."*,
  *"Reading the run's objects..."*). While something is only being read,
  the rest of the page stays live (Preview waits for it) and **Stop** stays
  grey.
- **Still** — the page itself is working and napari is frozen for a moment,
  so nothing can move. The first line and the grey line say what it is
  doing before it starts: *"Writing the new objects..."* (Accept on a
  box: the first line says for example *"Adding 4 objects…"*), *"Drawing
  the new objects..."*, *"Clearing the objects..."*, *"Taking the objects
  back..."*, *"Taking the result off the screen..."*, *"Drawing the wall
  picture..."*, *"Drawing the markers..."* (the first marker of a box
  only).
- **Stepping** — on a whole patch the page counts the parts of a long
  frozen wait (Accept: *"Writing the new objects..."*; drawing the result;
  Clear objects), one step at a time; it never shows full while napari is
  still frozen. A Preview of the whole patch counts its steps too, while it
  runs in the background; its two long steps, the wall picture and the fill
  itself, move to and fro.

The preview runs in the background and the napari window stays usable. A
Cells preview of a 4 million voxel box took about 12 to 20 seconds when it
was measured; a larger box takes longer (up to the 100 million voxel
limit), which is why **Stop** exists. Stop throws the run away:
*"Stopped. Nothing from this box was added."* Nothing is written, and the
box stays. While a preview runs, **Stop** is the one way out: **Close box**
is grey until it has finished or stopped.

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
- **two in a cell** (inside each cell) — that cell of the other run got two
  nuclei. Hover the row: *"Cell 812 of that run holds 2 nuclei here. It may
  be two cells drawn as one, a dividing cell, or a neighbour's nucleus poking
  in. Each is added on its own; untick any you do not want."* It stays
  ticked.
- **cut at edge** (inside each cell) — *"The border of cell 812 cuts
  off more than a quarter of this nucleus. The border may run through the
  nucleus — check it."* It stays ticked.

Unticking is a suggestion, not a rule: tick anything you actually want. Rows
with a chip are listed first — **two in a cell**, then **cut at edge**,
then **edge**, then **small**. **All** and **None** tick or untick every row.
An unticked object stays on screen in the three side views, drawn faintly in
its row's colour, so you can judge it where you judge every object; the main
view shows only the ticked ones, exactly what **Accept** will write (in 3D a
faint object would hide the ticked ones behind it). Ticking changes the
colours only — nothing is computed again.
The first line of the page says, for example, *"12 new objects found · 3 left
unticked (at the box edge)"*; inside each cell it also says how many cells
got two nuclei, last: *"14 new objects found · 2 cells hold two nuclei"*,
or *"16 new objects found · 2 left unticked (at the box edge) · 2 cells hold
two nuclei"*. When nothing is ticked because every row started unticked,
the line says so and names the button: *"12 new objects found · none ticked
(at the box edge) — press All, or tick the ones to add"*, with the cells
holding two nuclei before that tail: *"16 new objects found · 2 cells hold
two nuclei · none ticked (at the box edge) — press All, or tick the ones to
add"*.

Below the list, one grey line counts what the tool set aside, for example
*"2 markers fell in dark space and were skipped · 1 object vanished when
smoothed and was left out"*. When it is cut short, hover it to read the
whole of it; the same text is in the log. A click marker whose ball ended up
inside an object you kept makes nothing, and that is counted first: *"1
marker now sits inside a kept object and made nothing"* (*"2 markers now sit
inside kept objects and made nothing"*). Inside each cell, what you may want
to act on comes next: *"3 cells already held a nucleus and were skipped · 5
cells had no nucleus · 2 nuclei were split between two cells at their
border · 4 nuclei outside every cell were left out"*. On Cells, markers
from another run that lay mostly over objects you kept are counted too:
*"4 run objects lay mostly over kept objects and were skipped"*. With
**Look for nuclei ▸ At the markers I place**, markers whose ball missed the
stain are counted after those: *"2 markers sat on stain below the threshold
and made nothing"*. After **Discard** the list comes back, and each marker
that made nothing has a dim row whose tooltip says why (see *Nuclei at the
markers you place*).

**Show what the tool sees** (Cells only) shows the wall picture the fill
followed in the main view: one slice of it, at the depth of the XY view,
moving with it as you scroll that view — so the honeycomb of walls is not
hidden behind the slices above and below it. In the 2D main view it is the
slice on screen. Drawing it takes a moment; the first line says *"Drawing
the wall picture…"* meanwhile. On Nuclei it is grey: *"Nuclei builds no
wall picture — the walls are a Cells idea."*

**Show candidates as outlines** draws the new objects as outlines in the
three side views, so the membrane under them stays visible. The 3D view
cannot show outlines — napari does not draw them in 3D — so it is unchanged.
The tick is kept when you Discard and preview again in the same box, and goes
back off after Accept, Close box or a patch switch.

The new object ids **skip the ids of objects you deleted or cleared**: a new
object never reuses the number of one you removed, so a removed object's old
number never comes back as something else.

---

## Accept, Discard, Close box

The two buttons at the bottom never move. The second has one word for each
meaning — **Close box**, **Discard** or **Undo all** — and one grey line
under the two says what they do. Reshape and Refine end a session with
**Discard**, which puts the object back as it was; Add ends a box with
**Close box** instead, because on Add **Discard** only throws a result away
and keeps the box, while Close box ends a box in which nothing was added —
so nothing is lost.

- **No box.** The first is a grey **Accept**, the second a grey **Close
  box**, and the line says "Place a box to start."
- **A box is up.** The first is **Accept** (grey until a result is on
  screen). The second is **Close box**: the box and its markers are
  removed and nothing is added — "Close box ends this box; nothing is
  added." A Clear made in this box stays — after a Clear the line says
  "Close box removes the box; what you cleared stays cleared until you
  Undo."
- **A result is on screen.** The first is **Accept (n)**: it writes the
  ticked objects — only into empty space — and takes you to the review
  (below). When you untick the last row, or press **None**, the first
  line says *"None is ticked — tick the objects to add, or press All."*
  The second is **Discard**: it throws the result away and goes back to
  the box (the first line says *"Taking the result off the screen…"*
  while it does; so does switching the preset over a result). **The box
  and your markers stay**, so you can change a setting and press Preview
  again. The line
  says "Accept keeps the ticked objects. Discard throws this result away;
  the box stays."
- **The review (after Accept).** The first is **Done**, the second **Undo
  all** — see *After Accept — the review* below.
- **While a preview runs** the second button is grey: **Stop** is the one
  way out, and it keeps the box.

A result goes stale as soon as you change what it was made from — the box,
the markers, a setting. The result is then taken down and the page asks for
another Preview.

Closing this window with its **X** only hides it: the session stays open, with
its box, markers and result, until you press Accept, or Discard and then
Close box, and **Advanced Features...** in the annotator brings the same
page back. Until then switching run in Step 3, or starting Batch Refine,
is refused with a line that says to finish the box first: press Accept,
or Discard a result and close the box, on the Add tab.

---

## After Accept — the review

The page turns into a short review of this session's work: *"Added 12 objects
· 34,567 voxels"*, and the **Added this session** list (four rows at a time,
scrolling). Rows keep their **two in a cell** and **cut at edge**
chips. When cells got two of the nuclei you added, the first line says how
to find them later: *"Added 14 objects · 2,345 voxels · 2 cells hold two
nuclei — after Save, run Extract and pick that run under Compare with to
find them."* (**Compare with** is the run picker in the Extract window.)

Under the list, three buttons share one row:

- **Refine all ▸** opens Batch Refine with **every** object added in this
  session as its target (*Added this session (N)*) — pick a recipe saved on
  the Refine tab and run it on all of them. The Refine guide's *Running it on
  many objects* has the details. If Batch Refine cannot open, the first line
  here says why.
- **Refine ▸** takes **one** object — the one selected in the list — to the
  Refine tab. It is grey until exactly one row is selected.
- **+ Add another** starts again in the same box, with the same preset and
  marker source. Click markers are cleared. It will not start while another
  edit is open (see *Busy* below); the review stays as it is. While that box
  is up the line under the buttons says, for example, "Close box ends this
  box and goes back to the 12 objects already added." — **Close box** brings
  you back to this review, with **Undo all** still on offer.

The two buttons at the bottom, with the line "Done keeps the 12 objects.
Undo all takes back every Accept since the first box.":

- **Done** closes the session and keeps everything you added. **Undo all**
  is gone after Done: from then on the main Undo button takes the objects
  back, one Accept at a time.
- **Undo all** takes back every object added this session — every Accept,
  newest first, since the first box. It asks first (*"Take back all 12
  objects added this session?"* — **Take them back** / **Keep them**), then
  the first line says *"Taking back 12 objects…"* while it works. It
  will not run while any other edit is open (a Reshape, a Refine, an
  interpolation preview or a merge dialog; Batch Refine says its own
  sentence) — accept or reject that edit first.

**The main Undo during the review.** The review is not an open edit, so the
main Undo button (Curation) still works while it is up. When it takes back
an Accept the review lists, those objects leave the list: *"The main Undo
took back the last objects added here; they are off the list."* — and when
none is left the review closes: *"The main Undo took back the objects added
here, so the review is closed."* **Commit Triage** and a new **Extract** empty
the main Undo, so the objects of an open review can no longer be taken back;
the review closes and the objects stay: *"Commit Triage or a new extraction
made the objects added here permanent; the review is closed."*

**The rows to check come first.** Rows with a **two in a cell** or **cut at
edge** chip are listed first, as in the result list, so they are in view.

**If you ran a batch on the new objects and accepted it**, that batch is a
later edit sitting on top of them, so **Undo all** stops at it: *"Nothing was
taken back — a later edit is in the way. Undo it first, or use the main Undo
button."* Press the main Undo to take the batch back, then **Undo all**
again. If only the newer objects could be taken back it says how many, for
example *"Took back 4 of 12 added objects — a later edit is in the way of the
rest."*

**Undo all takes back Accepts, never a Clear.** A Clear made in this session
is still in force afterwards, and the first line adds: *"The objects you
cleared are still cleared — the main Undo button brings them back."* Objects
you added and then cleared leave the **Added this session** list.

**While Batch Refine runs, or waits for its Accept / Reject,** Add refuses
to place a box, preview, accept, add another or undo: *"Batch Refine is
working on these cells — Accept or Reject it first."* Rejecting a batch puts
the objects back exactly as they were before it, so anything added in the
meantime would be lost.

---

## The spacing (Cells)

Cells measures in microns — the click balls, the shrinking of run objects,
how far the fill may reach into dark space — using the patch's own voxel
size when it has one (from **Set spacing…** on the Session list, from Step
1, or from the file), else Step 3's own **Spacing**, set under QC Features →
Settings… in Step 3: the **Settings…** button beside **Extract** in the QC
Features section, row **Spacing**.

If it has never been set it is the default 1.0 µm, and Cells **runs on it
anyway** — nothing is refused and nothing is grey. An amber line under the
first one says so: *"Spacing is the default 1.0 µm here — if your voxels are
smaller, set their size with Set spacing… in the Session list, so the µm
knobs mean what they say."* **Set spacing…** works while a box is placed,
and keeps 1 / 1 / 1 too when your voxels really are 1.0 µm. On real voxels of
about half a micron every micron setting then acts at about half its size —
the click balls, the gaps bridged and the reach into dark space are all
smaller than the numbers say — and the fill can run further into open space,
so set it once. The line is not shown while a result is on screen; it comes
back when you set up the next Preview. With **Treat dark space as outside**
unticked, the dark-space line takes its place and this one is in Advanced,
under the Dark rows. Nuclei ▸ Anywhere in the box works in voxels and does
not need it. **Nuclei ▸ At the markers I place** does: its **Ball** is in
microns, so it shows the same amber line before a result, and runs on the
default anyway; on a whole patch over 20 million voxels the slow-patch line
takes that place and the spacing line moves into Advanced. **Nuclei ▸
Inside each cell** and a **Cells whole patch** do the same: all of their
settings are in microns, they run on the default, and the amber spacing line
shows as on a box — in Advanced when another amber line (a slow patch, a
half-size session) holds its place. Nothing in Add refuses because the
spacing was never set.

---

## What is refused, and why

The first line of the page always says why **Preview** is grey, and says it
in amber — so does every refusal (a click that landed on an object, a box
the batch is still working on). The common reasons:

- **The box is over the limit (Cells).** *"This box is 120.0 million
  voxels; the limit for cells is 100 million. Shrink it with Numbers… or
  pick S, M or L."* Switching from Nuclei to Cells never resizes your box,
  so a large Nuclei box can arrive here.
- **A channel is missing.** *"Pick the channel the nuclei are bright in, under
  Brightness."*, *"Pick the membrane channel under Walls."*, *"Pick the nuclei
  channel under Markers."* — or *"The nuclei and wall channels must be
  different."*
- **No markers yet (click markers, and Nuclei ▸ At the markers I place).**
  *"Place at least one marker: turn on Place markers, then click inside the
  box."* — or, with Place markers already on, *"Place at least one marker:
  click inside the box in a side view."*
- **Every marker unticked (Nuclei ▸ At the markers I place).** *"Tick at
  least one object to find its nucleus — all 3 are unticked."* (*"Tick the
  object to find its nucleus — it is unticked."* for one.)
- **No marker from the run (Cells).** When the count under Markers leaves
  no marker, Preview is grey before you press it, and the first line says
  why:
  - the run has no object here: *"That run has no object in this box —
    move the box, or pick another run."* (on a whole patch: *"That run has
    no object in this patch — pick another run."*);
  - objects are already in the box: *"None of that run's objects here can
    be a marker: each is in dark space, on a kept object, or too small.
    Move the box, or Clear objects."* (on a whole patch: *"None of that
    run's objects can be a marker: each is in dark space, on an object
    already here, or too small. Clear objects to make room."*);
  - the box holds no object yet: *"None of that run's objects here can be
    a marker: each is in dark space or too small. Turn off Treat dark space
    as outside under Advanced."* (on a whole patch: *"None of that run's
    objects can be a marker: each is in dark space or too small. Turn off
    Treat dark space as outside under Advanced."*). While that tick is off,
    or a tissue outline decides what is outside, it says *"None of that
    run's objects here can be a marker: each is in dark space or too small.
    Move the box."* (on a whole patch: *"None of that run's objects can be a
    marker: each is in dark space or too small."*).

  When the tissue outline dropped them, the outline's own sentence (below)
  is said instead.
- **Still measuring.** *"Measuring the box…"* — the background measurement
  of the box has not finished; it takes a second or two after a change.
- **Busy.** A preview is running (see above), or Batch Refine is running or
  staged.
- **Another edit is open.** *"Accept or reject the open edit first."* — Add
  does not start on top of an open Reshape, Refine, interpolation preview or
  merge, because rejecting that edit later would paste its old snapshot over
  the new object. Accept or reject it, then press Place box again.
- **Inside each cell.** *"That run has no cell in this box — move the box or pick another
  run."* (on the whole patch: *"That run has no cell in this patch — pick
  another run."*); *"This channel cannot be used to find nuclei — the reason
  is under Look for nuclei."*; *"That run cannot be used on this patch — the reason is
  under Look for nuclei."* (the reason is in the card, where the help line
  was).
- **Cells on a whole patch.** Where the numba
  package is missing: *"Cells fills a whole patch only with the numba
  package, which is missing from this install. Use boxes."*; a patch over
  the size limit: *"This patch is 150.0 million voxels; a whole patch can
  be filled up to 100 million. Press S, M or L for a box."* (Nuclei ▸
  Anywhere in the box on a whole patch has the same limit and says the
  same); with nothing picked under Markers ▸ From: *"Pick a run, Nuclei
  masks or Click markers under Markers ▸ From — a whole patch grows its
  cells from those markers."*, or on a patch with no other run *"This patch
  has no other run — pick Nuclei masks or Click markers under Markers ▸
  From."*; while an earlier
  preview still finishes in the
  background: "An earlier preview is still finishing in the background.
  Wait for it to end, then press Preview again."
- **The tissue outline is still being measured.** *"Measuring the tissue
  outline…"* — Preview waits for it (see *The tissue outline*).
- **Clear objects while something runs.** While a preview runs, while the
  tool is still counting for a Clear, or while Batch Refine runs or waits,
  **Clear objects** is grey or refused like every other change. It is not
  shown over a result or in the review: Discard the result, or start
  another box, first.

And after a Preview:

- *"Nothing found — lower the threshold or grow the box."* (Nuclei.)
- *"No marker sits on stain above the threshold — lower Threshold, or
  move the markers onto the nuclei."* (the **Threshold** row of the
  Brightness card; **Auto** puts it back at the automatic split) and
  *"Every marker now sits inside a kept object — remove them and click in
  empty space."* (Nuclei ▸ At the
  markers I place; each marker's row says why it made nothing, too.)
- *"No nucleus was found inside that run's cells — check the channel, and
  Nucleus radius under Advanced."* and *"Nuclei were found, but every one
  sits on an object already here — Clear objects first, or open the nuclei
  run."* (Nuclei, inside each cell.)
- *"No marker was left to grow from — every marker fell in dark space, over
  kept objects, or was too small to use. (Nothing was changed.)"* (Cells;
  with markers from another run the count says it before the press, above.)
- *"Every marker left lay outside the tissue outline — raise Bridge gaps or
  lower Dark below under Advanced, or pick another outline."* (Cells, with
  a tissue outline; with markers from another run, before the press too.)
- *"Spread from each marker cannot run here: the numba package it needs is
  missing from this install, so Cells fills boxes only. Reinstall EpiCycle's
  environment to get it back."* (Cells, Spread; or its cache-folder twin,
  see *How cells grow*.)
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
accepted stay, because they are in the volume. Press **Accept** or **Close box**
before you close, and nothing is lost by surprise.

While a box session is open the patch cannot be switched and the run cannot
be changed — finish, or close the box, first. The same goes the other way
round: with any edit open, **Switch** and Add's own **Place box** and
**Whole patch** say *"Accept or reject the open edit first."* and do nothing.

---

## What changed from the old Add tab

- **Look for nuclei ▸ At the markers I place:** one click on a nucleus in
  a group of touching nuclei now gives that nucleus, not the whole group
  (with **Separate touching objects** ticked, under Advanced; unticked, as
  before).
- It fills a whole region with many objects at once, instead of adding one
  missed cell. The **Cells with a membrane stain** preset is new; **Nuclei**
  is the old behaviour.
- New object ids skip the ids of deleted objects (the old tab reused them).
  The shapes Nuclei finds are the same as before.
- **Clear objects** empties a region or the whole patch in one step, with
  one Undo.
- **Look for nuclei ▸ Inside each cell of run ▸ …** finds the nucleus inside
  each cell of another run of the patch, with **two in a cell** and **cut at
  edge** chips on the rows to check.
- **Whole patch** makes the box the whole patch — on Nuclei ▸ Inside each
  cell, and on Cells, which fills it to the walls in one step and leaves
  the tissue far from every marker empty. It is a box size (**Whole** on
  the row of sizes) and is never grey; when the page cannot fill a whole
  patch, Preview is grey and the first line says what to pick. (Until this
  version Cells filled a whole patch in pieces, by Spread.)
- **Grow by ▸ Spread from each marker** (Cells, on a box) grows each cell
  from its marker until a wall, a neighbour or its growth limit stops it;
  the growth limit is 28 by default now (it was 14).
- **Show what the tool sees** shows one slice of the wall picture, at the
  depth of the XY view, instead of the whole volume.
- **Tissue outline** (Cells) keeps new cells inside the outline of the
  objects you cleared, or a shrink-wrap around the stain; **Show the tissue
  outline** draws it in the main view and the three side views.
- The second button at the bottom has one word for each meaning: **Close
  box** ends the box, **Discard** throws a result away and keeps the box and
  markers, and in the review **Undo all** takes back the session while
  **Done** keeps it. As on Reshape and Refine, **Discard** throws away what
  is on offer and never anything already accepted.
- The box can be dragged by its outline and nudged with Shift + arrow keys.
- Closing the annotator asks before it drops unaccepted work (except the
  title-bar ×, above).
- *Connectivity* and the *Signal* choice are gone from Advanced.
- Nothing in Add refuses when Step 3's Spacing was never set: Cells (a box
  or the whole patch) and Nuclei ▸ Inside each cell run on the default
  1.0 µm and say so in amber (see *The spacing (Cells)*).
- On a Cells whole patch **Markers ▸ From** lists the other runs and Click
  markers, and the pick you had on the box comes back with the box; Place
  markers stays as it was. Preview greys before the
  press when a run gives no marker, the Auto lines no longer say
  *Measuring…* when nothing is measured, and the Step 3 message about an
  open Add box names Accept and Close box.
- Unticked objects no longer vanish: they stay drawn faintly in the three
  side views (the main view still shows only what Accept writes), and a
  result with nothing ticked says so and names **All**.
- The click marker's size, **Ball**, sits beside **Place markers** and is
  1.0 µm by default (it was *Click ball radius* in Advanced, 2.0 µm); the
  placed count moved to the Markers title. The channel lists show each
  layer's colour, the idle strip says what Clear objects is for, and
  leaving a whole patch with S, M or L goes back to the box from before
  it: the same size is exactly that box, with its markers; another size
  is centred on it and keeps the markers inside it.
- Every wait says what it is doing before it starts, in the first line and
  in the grey line of the strip; the bar moves while something runs in the
  background, stays still while napari is frozen for a moment, and steps
  through a long whole-patch wait. **Vessel ridges** and **Thin-faint
  ridges** warn in amber that their wall picture is slow.
