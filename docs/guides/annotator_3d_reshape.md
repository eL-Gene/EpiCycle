# Reshaping an object — the Reshape tab

Sometimes the reconstruction glues two objects together. This tool cuts one of
them apart. You pick the object, you say where the markers go and what the cut
should follow, and you see the proposal before anything changes.

Nothing is written to disk here. The change lives in the session until you
press **Save curation**, and one **Undo** takes it back.

---

## The quick version

1. Click the merged object (in the list on the right, or right-click it in the
   3D view).
2. Press **Start**. The object disappears from the main view and a painting
   layer called *Refine* appears in its place — that is normal, and the object
   comes straight back if you discard.
3. Put one marker inside each object you expect to get out.
4. Press **Preview**. You see the proposed pieces on every slice.
5. Not happy? **Change the markers and press Preview again.** You do not have
   to discard anything. This is a loop, not a one-shot.
6. Tick the pieces you want. Press **Accept**.

### The loop, and the one thing it does that surprises people

The markers stay open and editable after a preview. Untick a marker, add one,
undo one, change how the cut is made — anything in the marker list or in
**Sources** — and **the preview is put away and your markers come back on the
object.** (The marker list itself is not inside Sources: it sits with **Add
marker**, **Undo** and **Edit markers** in the session block, which is on
screen the whole time the object is held out.)

That is on purpose. The cut and the markers are drawn in the same layer, so
they can never be on screen together; if the cut stayed, the tool would have
no way left to tell a marker from a piece. What you see after touching
anything in Sources is exactly what you saw before you pressed Preview. Press
**Preview** again and the new cut appears.

The status line at the top tells you which of the two you are looking at, and
says so in words: *"Object 139 - 3 pieces from 3 markers. 94 % assigned.
Scroll up to change the markers and Preview again."*

One thing does **not** put the preview away: ticking a **piece**. That only
changes what Accept writes, and it is a redraw, never a re-cut.

---

## How the panel is laid out

Every tool in this dock has the same six slots, top to bottom, and the two
buttons at the bottom never move:

- Above all six, a single sentence with a chevron: *Cuts one merged object
  into several. Markers say where the objects are; walls say where their edges
  are.* Click it and four short paragraphs open underneath, explaining what the
  tool actually does — ordinary image processing, no learning, no prediction.
  Both go away while you are working on an object and come back when you are
  done, and the body stays shut unless you opened it.
- **Status** — one line with a coloured dot: which object, what stage, what
  happens next.
- **Target** — the object. **Start** begins a session; once it has, the same
  button reads **Change** and is greyed out, because from that moment the
  object is held out of the volume and re-targeting would leave a hole behind.

  Under it sits **Show this object as an outline**. Tick it and the object you
  are reshaping is drawn as a thin edge instead of a solid block **in the three
  side views** — which is where it matters, because the side views are where
  you paint markers, and an outlined body is what lets you see the wall signal
  you are aiming a marker at. **The 3D view is unchanged**: napari does not
  draw outlines in 3D at all, so there is nothing to see there either way.
  **No other object changes**, and nothing is written to the data — it is
  purely how this one object is drawn. The tick stays on from one object to the
  next until you turn it off.

  One thing to expect: the outline applies to *every* id on that layer, so a
  large marker is drawn as a ring too. Small markers are too thin to hollow out
  and still read as dots.
- **Sources** — one line with a chevron, reading something like
  *Sources - markers: painted · walls: shape*. Click it to open
  three sub-sections in place — **Markers**, **Walls**, **How to cut** — of
  which **only one is open at a time**, so the panel never stacks three forms
  into a dock this size. If the line is too long for the button it is
  shortened with an ellipsis; hover it to read the whole thing.
  The marker controls — **Add marker**, **Undo**, **Edit markers**, the
  distance heatmap and the marker list itself — belong to the **session**, not
  to the source. Whichever way the markers came, you can add one by hand,
  remove the last one, or untick one you do not believe. They appear when you
  press **Start** and not before: every one of them needs the object held out
  of the volume, so before that they would be a block of greyed buttons
  between you and the Preview button. The list appears with the first marker.

  The two **finders** are the part that is keyed on the source, and there is
  only ever one of them on screen: **Find from shape** for hand-painted
  markers, **Find markers** for every other source.
- **Preview** — the button, with **Stop waiting** appearing beside it while
  the tool is thinking and a thin progress bar underneath. It is **pinned**
  just above the two buttons at the bottom and does not scroll: open a
  Sources sub-section as far as you like, and Preview stays where it is.
- **Result** — appears **above** the buttons when there is something to
  decide: the piece checklist, the verdict, **Split anyway**, **Refine
  Fragments…**.  On its title row, over on the right, sits
  **Show what the tool sees** — the wall signal the cut followed, as a
  layer.  See *"Show what the tool sees"* below.
- **Accept** and the button beside it — **Cancel** when nothing is staged,
  **Discard** for as long as the object is held out of the volume. Two words,
  one meaning each.

Everything else the engine can do is behind **Advanced**, at the bottom of
the page, one click deep.

### Shape or image, in one place

The cut can follow the object's own **shape** or something it can see in an
**image channel**. You choose that once, in *Walls from*, under **Walls** —
the old "Watershed on" row asked the same question in different words and is
gone from the panel.

- The **shape** choices (*Pinch points*, *Between the spines*, *No wall*) use
  no image at all. They work when the objects have distinct lobes.
- The **image** choices (everything else) follow what is actually in the
  channel you pick. They are much better when the two objects are similar in
  shape but genuinely separated by a membrane — and only they can be
  arbitrated (see "Only cut where I can see a wall").

The full list is below.

---

## Where the markers come from — "Markers from"

Open the **Sources** line, then **Markers**. The first picker is *Markers
from*. **If you never open it, this tool behaves exactly as it always has.**
That is deliberate.

A marker is one blob per object you expect to come out. One marker, one object.

### One list, whichever way they came

However the markers arrive, they end up in **the same tickable list**, with
the same ticks and the same meaning: *Preview cuts with the ticked ones.*

Unticking a marker **lifts it off the object**. The row is dimmed and struck
through, and the marker itself disappears — from the 3D view and from all three
side views — so what you are looking at is exactly the set of markers the cut
will use. The object itself is not touched: the marker comes off it, no hole
is left behind. Tick it again and it comes back where it was (if you painted
another marker over it meanwhile, that marker keeps the overlap). Nothing is
deleted: a marker you unticked is still in the list, still has its colour, and
is one click from being back.

You can always add to what a finder found. Press **Add marker**, paint a blob
where the finder missed one, and Preview uses your marker alongside the found
ones. **Undo** removes the last marker you added.

**Hand-painted markers** — you paint the markers yourself. Paint a blob inside
the first object, press **Add marker**, paint a blob inside the second. This is
the original behaviour and it is still the default.

**Shape thickness peaks** — the tool finds the markers from the object's own
shape: the peaks of its thickness. Press **Find markers** and they are placed
on the object for you, as ordinary markers you can then untick, add to or undo.
Good when the two halves bulge apart clearly, and a last resort otherwise — it
cannot see the biology, so it can shatter a tall columnar object into a stack
of discs. It also needs a wall it can measure a shape against, so while it is
chosen the wall options *No wall* and *Between the spines* are greyed out with
the reason on them. It only ever puts markers **inside** the object: until
2026-09-20 it measured thickness across the whole search box, so on a long
branched object every peak it found was in the empty space around it and the
tool reported no markers at all.

**Markers from another run of this patch** — if you already have a nuclei
segmentation of this same patch in another run, each nucleus inside the merged
object becomes one marker. Two nuclei, two markers, two objects. You do not paint
anything.

The picker lists every run of this patch that has labels, showing the run's
name, what it holds, how far it got on this patch, and **which file is being
read** — the curated labels if that run curated this patch, otherwise the raw
reconstruction. It also says, plainly, *you are trusting these labels.* That is
the whole risk of this mode: if the nuclei mask is wrong, the split will be
wrong in the same way.

**Markers from a mask file** — the same idea, from a TIFF anywhere on disk. Press
**Browse…**, then **Load mask**.

The run picker and the **Browse…** button are not both on screen at once: you
see whichever one your choice needs. **Load mask** applies to both.

**Find them in an image channel** — the tool looks straight at a channel, with
no mask needed. This is the one to reach for when you have a nuclei stain.

### Finding them in an image channel

Two rows appear under the picker.

**Channel** — which image channel to look in. Usually the nuclei stain. If the
layer you chose is renamed or removed the box goes **blank and stays blank**,
and the tool then says "pick a channel" rather than quietly looking at some
other channel.

**Find by** — how to look. Six ways, commonest first:

- **Nucleus peaks** — blurs the channel and puts a marker at each local peak.
  Start here.
- **Prominent peaks** — only peaks that rise a set height above their
  surroundings. Use it when *Nucleus peaks* finds two markers in one nucleus:
  raise *Prominence* under Advanced until it finds one.
- **Round-blob finder** — finds blob centres by shape as well as brightness, so
  two nuclei that touch still come out separate.
- **Round-blob finder (fast)** — the same idea, cheaper and slightly coarser.
- **Bright-object centres** — thresholds the channel, then puts one marker at
  the centre of each bright object. Good for a clean, high-contrast stain.
- **Membrane-enclosed interiors** — the odd one out: it reads the **membrane**
  channel, not the nuclei one, and finds each dim space the membrane fully
  encloses. Use it when there is no nuclei stain at all. Point *Channel* at
  the membrane for this one.

### "Prepare the channel first"

Under those rows is a chip reading *Prepare the channel first: …*, naming what
it would actually do. Open it for five tick-boxes:

- **Local contrast** — evens out bright and dim regions so faint nuclei count
  too. On by default, and usually all you need.
- **Even out depth** — lifts deep, dim slices to match the shallow ones. Tick
  it when the bottom of the stack is washed out.
- **Remove haze** — takes away smooth background glow so only the nuclei stand
  out.
- **Lift dim signal** — brightens faint signal more than bright signal.
- **Despeckle** — removes salt-and-pepper noise.

They always run in the same order, whichever you tick: even out depth, remove
haze, lift dim signal, despeckle, local contrast. The chip's line tells you
what is on without opening it.

The **Walls** sub-section has the same chip for the wall channel, with one
extra tick at the top: **Same as markers**, which simply reuses the marker
channel's preparation here too. (Local contrast is always applied to the wall
anyway, so it is not repeated in that list.)

### "Find markers"

Press **Find markers** and the tool looks now, without cutting anything. The
line beside the button says how many it found; underneath, every marker is
listed with its size or its position, and its colour:

> Markers - before the cut. Decides what the cut is made from.
>   [x] Marker 1 · 1,204
>   [x] Marker 2 · 980
>   [ ] ~~Marker 3 · 61~~
> Unticked markers are lifted off the object - tick again to put them back.

Untick anything that is not an object — a speck of noise, a nucleus that
belongs to the neighbour. It comes off the object as you untick it, so the
object shows you exactly what the cut is about to use. **Preview then cuts
with the ticked markers only.**
Nothing is unticked for you: the tool found them, so it treats them all as
objects until you say otherwise. **All** and **None** tick and untick the lot.

You do not have to press **Find markers** at all. **Preview** finds them itself
and keeps them all. Finding first is how you get to *look* before you cut.

Change any marker setting — the channel, the way of finding, a preparation
tick, *Prominence*, *Min marker gap* — and the list empties with a note saying
so. That is not the tool being fussy: the markers are numbered in the order
they are found, so a changed setting renumbers them, and your ticks would then
be pointing at different markers.

While the search runs, **Sources stays open** (it would be absurd to hide the
button you just pressed) and **Stop waiting** appears beside Preview.

Markers are drawn on the **Refine** layer itself, whichever source they came
from — they are the coloured dots you can see on the object. That matters more
than it sounds: the Refine layer is the one that is mirrored into the **XY, XZ
and YZ side views**, which is where you paint and check markers, so markers
found in a mask or an image channel show up there too. There is no second
marker layer to hide or hunt for in the layer list.

You can add to them by hand at any time. **Add marker** gives you a fresh
colour to paint with, and the found markers stay exactly where they are; the
cut then starts from all of them together.

**Find from shape** is the finder for **hand-painted markers only** — it puts
a marker at each of the thickest points inside the object, which is all the
shape alone can tell you. Every other source has **Find markers** instead: one
finder per card, so there is never a second button offering to do the same job
in a different way.

### If the mask is refused

The mask has to line up with the volume you are looking at. If it does not, you
get a plain sentence saying why. The usual cause is **downsampling**: if this
session is showing the patch at half size, a full-size mask does not match.
When that happens the tool matches the mask to the session for you and shows a
note. Be aware that shrinking a mask can merge two small nuclei into one — so
after **Preview**, check the number of pieces is what you expected. If a marker
has gone missing, turn downsampling off in **QC Settings** and reload the
patch.

The mask file on disk is never modified.

---

## What the cut follows — "Walls from"

Nine choices, under **Walls**. The first is the original behaviour and the
default; the rest are the engine's, in the order the picker lists them.

- **Pinch points (shape)** — negated distance transform: a narrow neck becomes
  a ridge. No image is used. The default.
- **Membrane sheets** — local contrast, blur, then a sheet-ridge filter. The
  one to pick for a membrane stain.
- **Membrane sheets plus shape** — mostly the sheet-ridge filter above, with
  the object's shape as a tie-breaker.
- **Intensity edges** — cuts where brightness changes fastest.
- **Raw brightness** — the channel as it is, with no ridge filter.
- **Vessel ridges** — a tube-shaped ridge filter, for rounded walls.
- **Thin-faint ridges** — a ridge filter for weak, thin walls.
- **Between the spines** — cuts midway between the shape's internal spines.
  No image is used.
- **No wall (midline only)** — no wall at all: cuts halfway between the
  markers. No image is used.

Whichever you pick, the grey line under the row says what it does.

Choosing any of the six **image** kinds reveals a **channel** row: the
image channel the walls are read from. It does not have to be a membrane
stain — any channel whose bright parts sit where objects meet will do, which
is why the panel calls it the **wall channel** and every action on this page
says "wall". The word *membrane* survives in exactly two places, and both
name a method rather than a channel: the wall types **Membrane sheets** and
**Membrane sheets plus shape** (the sheet-ridge filter), and the marker
finder **Membrane-enclosed interiors** (a basin with a rim all the way
round). The three **shape** kinds need no channel and show none.

Two knobs for the wall signal live under **Advanced**, in the WALLS card:
**Wall clip** (local contrast on the wall channel — raise it if the signal
is faint) and **Wall blur** (smooths speckle before the ridge filter looks
at it). The **Sheet-ridge filter** tick there turns the signal into a thin
sheet the cut can follow; off, the cut uses brightness directly.

> **Shape thickness peaks needs a shape wall.** When the markers come from the
> object's own thickness, the marker finder reads the wall you chose — so *No
> wall* would find nothing at all and *Between the spines* would quietly be a
> different finder than its name says. Both are greyed out while that marker
> source is chosen, with the reason on them.

---

## "How to cut" — the three decisions

The third sub-section under Sources holds everything that shapes the answer
rather than tuning it: **Solve with** (below), and three rows under it.

- **Snap to the wall signal** — after the cut, each piece's edge slides onto
  the nearest edge in the wall channel. Shown only for the image wall kinds,
  because a shape wall has no edge in the image to snap to. Its **Snap
  radius** (how far an edge may slide) and **Snap strength** (lower snaps
  harder) are under Advanced. A snap that would erase a piece is skipped with
  a note.
- **Only cut where I can see a wall** — the arbiter; its own section below.
- **I expect exactly [n] objects** — tick it and say how many. If more pieces
  come out, the weakest joins are merged until the count matches, and the
  tool says how many it merged. If **fewer** come out you get a warning and
  nothing is forced: a cut that is not there is not invented to hit a number.

---

## "Only cut where I can see a wall"

This checkbox lives in **How to cut**, beside the solver it second-guesses.
It appears whenever the walls come from an **image channel** — any of the
six image kinds, not just *Membrane sheets*. It is **off** by default, and
it is hidden for the three shape kinds, because arbitrating a shape-derived
cut against its own shape proves nothing.

Turn it on and the tool checks, before writing anything, whether there really
is wall signal along the surface it is about to cut. If there is, it splits.
If there is not, it leaves the object whole and tells you:

> kept whole — no wall between the markers (possibly dividing)

That is a **successful** answer, not a failure. Two nuclei with no membrane
between them is what a dividing object looks like. Splitting it would invent an
object that does not exist.

Two things to know about it:

- The test is **relative to that object's own brightness**, not an absolute
  level. A very bright object raises its own bar, and a very dim one lowers it.
  So a verdict is a judgement about *this* object, and it does not transfer
  between datasets unchanged. **Wall required (%)** under Advanced is the
  slider: 75 % means "stronger than three-quarters of this object's own wall
  signal".
- On a badly attenuated stack, a real membrane sitting in a dim deep slice can
  read as "dividing". If you see that happening on real data, say so — the fix
  is known but has not been built, because building it without evidence would
  be guessing.

### "Split anyway"

If the tool keeps an object whole and you disagree, press **Split anyway**. It
makes the cut you asked for. The result is marked, so that months later you can
still tell a cut the membrane justified apart from a cut a person insisted on.
That mark survives re-extracting the QC features.

---

## The preview, and the piece list

**Preview** does the work on a background thread. The window stays alive — you
can rotate the volume while it thinks — and a small bar shows what stage it is
at. If you get bored, **Stop waiting** abandons it and nothing is changed.

Be aware of what that button honestly does: the calculation is checked between
stages, so a stage already running finishes in the background. Its answer is
then thrown away. Nothing is left behind either way.

While it thinks, the **Sources** line closes itself and greys out — and so does
the marker block beneath it: the list, **All** / **None**, **Add marker**,
**Undo**, **Edit markers** and the finder all go grey, because they live
outside that disclosure and closing Sources alone would leave them live on the
very set the calculation is using. Your choices are not lost: everything comes
back exactly as you left it when the answer arrives, and Sources **stays** open
once the pieces are on screen.

When it finishes you get the pieces overlaid on the volume — in the 3D view
and on every slice in all three side views — plus a list, which appears
**above** the two buttons rather than below them:

>   6 % of the object was not given to any piece. Those voxels stay with
>   object 149 when you Accept — nothing is lost.
>   Piece 3 is very small (140 voxels) — unticked for you.
>
>   [x] Piece 1  4,120 vx
>   [x] Piece 2  3,880 vx
>   [ ] Piece 3    140 vx  (small)
> unticked pieces stay part of object 149

Read the grey line under the list carefully. **Unticking a piece does not delete that tissue.**
It means "I don't want that as its own object", and those voxels go back to being
part of the original object. Very small pieces start unticked as a suggestion,
not a decision — tick one if you actually want it.

Unticking does not re-run anything. It only changes what **Accept** commits.

Under the list, in grey, the tool reports what it saw. The one worth reading
is **"Wall contrast 0.12-0.81."** — the range of the wall signal inside this
object, on the scale the cut actually floods. A healthy range is most of
0–1. A narrow one near zero means there was almost nothing in the channel to
cut on, and that a disappointing cut is a data problem rather than a settings
problem. It only appears for the six **image** wall kinds; a shape wall has
no channel, so there is nothing to report.

### The amber block above the list

One line per problem, in amber, above the pieces — so a cut with nothing
wrong with it shows no block at all, and a cut with three slivers gets three
lines. They are things worth knowing about the cut you are looking at; none
of them is an error, and none of them stops you accepting. A genuine refusal
is a separate **red** line, in the same place.

- **"1 marker did not get a piece of its own."** Two markers probably landed
  in the same object. Raise *Min marker gap* or *Marker blur*, or untick the
  spare marker and Preview again. You will not see this one when *I expect
  exactly n objects* did the merging — that absorbs marker-backed pieces on
  purpose, and the tool already says so.
- **"6 % of the object was not given to any piece."** The cut left some voxels
  over. They are not lost: they stay with the original object when you Accept,
  and the status line at the top says the same number the other way round
  ("94 % assigned"). A few percent is normal. If it is more, try *Soft cut*,
  or turn *Snap to the wall signal* off.
- **"Piece 3 is very small (410 voxels) — unticked for you."** A piece less
  than a fifth the size of the middle one. It is a suggestion, not a
  decision: tick it if it really is an object. If one marker keeps getting a
  starved sliver, raise *Balance* under Advanced.
- **"You expected 3 objects; 2 came out."** Only when you ticked *I expect
  exactly n objects*. Nothing was forced. Add another marker, or lower
  *Prominence* under Advanced.

### Two ticklists, and which is which

There are two lists in this tab and they do different jobs. **Since the loop
landed you see both at the same time**, so each one carries a permanent line
saying which it is, and the marker rows carry a gold edge the piece rows do
not:

- **"Markers - before the cut. Decides what the cut is made from."** Ticking
  there says "this is an object". Untick a marker and it is **lifted off the
  object** and the preview is put away; press **Preview** to cut again without
  it. Tick it again and it comes straight back.
- **"Pieces - after the cut. Decides what Accept writes."** Untick a piece and
  nothing is recomputed; those voxels simply stay part of the original object.

The short version: the **markers** decide what is cut, the **pieces** decide
what is kept.

If you untick *every* marker and then press Preview, the tool refuses and says
so rather than cutting with none.

### "Show what the tool sees"

The wall signal is the one thing the cut follows that you cannot see. On the
**Result** card's title row, beside the word RESULT, there is a tick:
**Show what the tool sees**. It appears once a Preview has produced something,
and it adds the wall picture the cut actually followed as an ordinary layer
called **`guide_walls`** — bright where the tool believed there was an edge,
dark where it did not. Look for it in napari's layer list if you want to turn
its contrast or its colours up.

Not every Preview makes one. If you painted your markers by hand **and** left
the wall kind at **Pinch points (shape)**, there is nothing to show — that cut
is placed from the object's own outline and follows no wall signal at all —
so the tick stays grey and a line under it says why: *"Only a wall-based cut
makes a picture. Pick a marker finder, or a wall kind other than 'Pinch points
(shape)'."* Every reason this tick is grey is written on the page like that,
rather than in a tooltip, because a tooltip on a greyed-out control is never
shown.

It is lined up with the object and with your markers, which stay drawn on the
**Refine** layer, so you can put a marker and a wall side by side and see why
the cut landed where it did. Nothing is written and nothing is changed: it is
a picture of an intermediate step.

It appears in the **main 3D view only**, not in the three side views. Those
show your image channels and your labels, and this is neither — it is a
picture the tool made, not part of your data. If the labels sit on top of it
and you cannot see it, either turn the labels layer's opacity down in napari's
own layer controls on the left, or tick **Show this object as an outline** up
in the object card so only the outline is drawn.

Under the tick, one grey line says what the brightness means, and it changes
with the wall kind — because the same greyscale means four different things:

- an **image** wall (*Membrane sheets*, *Intensity edges*, *Raw brightness*,
  *Vessel ridges*, *Thin-faint ridges*, *Membrane sheets plus shape*) —
  *"Bright = where the tool thinks an edge is."* This is evidence out of your
  data, and a picture with no bright lines in it is the honest explanation for
  a disappointing cut.
- **Pinch points (shape)** — *"The object's own shape, not the image.
  Bright = thin — the pinch the cut goes through."* Nothing in it came from a
  channel; it is the object describing itself, and it is **brightest where
  the object is thinnest**, darkest deep inside the bulk. So a picture that is
  dark nearly everywhere means there is no waist to cut at, and the answer is
  an image wall rather than a different setting.
- **Between the spines** — *"Bright = far from the object's spine — the
  valley between two merged objects. The spines are dark."* This is the same
  greyscale read the other way round: the dark lines are the two objects'
  own middles, and the bright sheet lying between them is where the cut goes.
  Two dark lines with a clear bright sheet between them is the picture you
  want; one dark line means the tool sees one object, not two.
- **No wall (midline only)** — *"Nothing to see — this cut follows no wall,
  only the midline between markers."* This picture is flat on purpose. There
  is genuinely nothing to look at, and that is the answer: the cut is placed
  halfway between the markers and follows no signal at all.

Remove the tick and the layer goes away. It also goes away by itself whenever
the preview does: change a marker, change a wall setting, press **Accept** or
**Discard**, switch patch, or close the annotator — and the tick goes down
with it, so what is on screen and what the box says can never disagree. Press
**Preview** again with the box still ticked and the picture is rebuilt from the
new cut rather than left behind from the old one.

Two things **keep** it up, deliberately. Ticking and unticking **pieces** in
the checklist does not touch it — that only changes what Accept writes. And
pressing **Refine Fragments…** keeps it up for the whole of that stage: the
picture belongs to the cut that made those fragments, so it is still the
honest answer to "why did it cut there" while you shape them. It goes when
you Accept all or Discard out of that stage, like everything else.

The layer covers only the region around the object, not the whole patch, so it
costs very little memory and lands exactly on the object either way. On an
enormous object the tick is still refused with *"This object's region is too
large to show the wall picture."* — a wall picture is a floating-point layer
and also a texture on the graphics card, so there is a ceiling (64 MB, which is
about 16 million voxels, roughly twenty-four times a typical object's region).
Nothing else changes; the cut and the piece list are unaffected, and the
console says how many bytes it was.

### "Refine Fragments…"

Under the checklist there is **Refine Fragments…**. It writes the pieces into
the volume and then steps through them **here, one at a time** — it does not
switch tabs. Each piece can be smoothed, dilated or grown on its own, the
primary button becomes **Accept all (n)**, and **Discard** still puts the
original object back whole.

It is offered after **every** Preview, however you found the markers and
whichever wall kind you cut with — hand-painted, shape thickness peaks, labels
from another run or from a file, or markers found in an image channel. (Before
2026-09-20 it appeared only after a hand-painted, shape-only Preview.)

**It follows the ticks.** Only the pieces that are still **ticked** become
fragments you can refine. An unticked piece is not deleted: its voxels stay
part of the original object, exactly as they would if you pressed **Accept**
instead — so the fragment list and the object beside it always add up to what
you started with. The button **disappears** when there is nothing to refine:
when you have unticked every piece, or when the tool decided the object is one
object after all and kept it whole. Either way the panel says so, under
**Accept** and in the line beneath the two buttons.

**While you are refining fragments the marker controls are put away.** The
markers and the *Markers from* section are switched off, because the pieces are
now in the volume and the temporary **Refine** layer the markers lived on is
gone. Press **Discard** if you want to go back and change a marker — the object comes
back whole and you can Start again.

**Accept all** records the same things a normal Accept records: which pieces
became new objects, whether the original object survived (it does, if you left
a piece unticked), and the tool's verdict — *dividing*, *low-evidence* or
*split-override* — in the object's history.

### If Preview comes back empty

- **"No markers found in this object"** — there were no markers inside the
  object. Lower *Prominence* under Advanced, change where the markers come
  from, or paint one by hand. The tool will never fall back to cutting the
  object without markers; that can shatter a tall object into a stack of discs.
- **"3 of 5 markers lie outside this object — ignored"** — markers outside the
  object are dropped and counted, so the piece count you see is honest.
- **"1 piece — boundary refined"** — one marker is legal. You get the object
  back, possibly with a tidier boundary.

---

## Accept, and undoing it

**Accept** writes the pieces into the volume. It only ever writes inside the
object you selected — a neighbouring object cannot be overwritten, in any mode,
ever.

The original object's id is retired and every piece gets a new one. That is how
this tool has always behaved; it is not new in the guided modes.

**Undo** once puts everything back — the volume, the list, and the marks.

While a split is open the rest of Step 3 is locked: you cannot save, commit
triage, switch patch or leave the step. That is stricter than the old manual
split was, and it is on purpose — halfway through a split the object is
temporarily not in the volume, and saving at that moment would write a hole to
disk.

---

## Advanced

Everything else the engine can do lives behind the **Advanced** button, in
three cards. Every knob carries a grey sentence saying what it does, so this
list is a map rather than the explanation. You will not normally need any of
it.

**MARKERS**

- **Prominence** — one number, two jobs. For the peak and interior finders
  it is how far a peak must stand out from what is around it (for the
  interiors, how deep the basin must be). For the two blob finders it is the
  detection threshold instead — and it runs the same way round, so *lower*
  finds more blobs.
- **Marker blur** — how much smoothing before peaks are found. The single
  most useful knob here: more smoothing means fewer, larger markers.
- **Min marker gap (um)** — two markers closer than this become one. In
  microns, converted using the finest voxel spacing, because the underlying
  peak finder takes one number for all three axes.
- **Largest nucleus (um)** — the blob finders only: bigger blobs are ignored.
- **Marker threshold** — *Nucleus peaks* (the default finder) and
  *Bright-object centres*. Left on **automatic** the tool picks the level
  itself.
- **Local contrast strength** — the marker channel's local contrast. Higher
  gives more contrast and lets more noise through with it.
- **Lift dim signal** (gentle / strong) and **Lift amount** — which kind of
  lift the *Lift dim signal* tick runs, and how hard it pulls. Smaller
  amounts lift faint signal more.
- **Haze size**, **Despeckle size** — the sizes the matching prepare ticks
  use. Haze size is the largest feature to KEEP, so set it bigger than a
  nucleus. Both apply to the wall chain too when you tick them there.

**WALLS**

- **Wall clip** — local contrast on the wall channel. Raise it if the wall
  signal is faint.
- **Wall blur** — smooths speckle before the ridge filter looks at it.
- **Sheet-ridge filter** — turns the wall signal into a thin sheet the cut
  can follow. Off, the cut uses brightness directly.
- **Wall required (%)** — how strong the wall must be along a proposed cut
  before *Only cut where I can see a wall* keeps it.
- **Search padding (voxels)** — how far outside the object the tool looks.
  The value is taken **when you press Start**: changing it mid-session
  does nothing until the next object, because the region everything else
  works in was already cut out.

**CUT AND SNAP**

- **Soft cut: wall resistance** — *Soft cut* only. Higher makes the cut
  stick more tightly to walls.
- **Balance** — *Sharp cut* only, 0 to 1. Above zero, pieces are pushed
  toward equal size; it is the fix when one marker keeps getting a starved
  sliver.
- **Snap radius**, **Snap strength** — how far an edge may slide when
  *Snap to the wall signal* is on, and how hard it snaps (lower is harder).
- **Absorb slivers below** and **Sliver size** — **on by default at 20
  voxels**, which is what the original prototype always did. Pieces smaller
  than that are folded into their biggest neighbour; a piece backed by its
  own marker is never absorbed.
- **Auto-smooth results** — rounds off the staircase edges a cut leaves
  behind. Two things to know, because neither is obvious: it runs at
  **Preview**, not at Accept, so what you see is what you get; and it only
  applies to a **hand-painted** cut. Any cut that came from *Markers from*
  or a wall other than *Pinch points* ignores it.
- **Auto-advance after accept** — unchanged.

The three promoted rows that used to live here — snap, the expected count and
the arbiter — are in **How to cut** now.

---

## "Solve with" — how the cut is placed

This one is not in Advanced; it sits in **Sources**, because it changes the
answer rather than tuning it.

- **Sharp cut (watershed)** — floods out from each marker and cuts on the
  highest ridge between them. The default, and the right answer most of the
  time.
- **Soft cut (random walker)** — walks probabilities out from the markers, so
  the cut eases onto a faint wall instead of snapping to the nearest strong
  one. Use it when the membrane between the two objects is weak or broken.
- **Settle onto the wall signal (label diffusion)** — starts from the sharp cut,
  then lets the labels spread through the image so the boundary slides onto
  the wall signal; the spread stops at intensity steps.

  **It is not smoothing.** The number of pieces is preserved and the result
  can look *rougher*, not smoother — it is following the image, and real
  membranes are not smooth. It needs the optional u-Segment3D package; if that
  is missing the tool uses the sharp cut and says so rather than silently
  doing something different.

---

## Troubleshooting

**A marker disappeared off the object when I unticked it.** That is the tool
working. An unticked marker is lifted off the object — in the 3D view and in
all three side views — so what you are looking at is exactly the set of
markers the next cut will use. The object underneath is left whole; you will
not see a hole where the marker was. It is not deleted either: its row is
still in the list, dimmed and struck through, and ticking it puts it back
where it was (if you painted another marker over it meanwhile, that marker
keeps the overlap). **All** and **None** do it for the whole list at once.

Nothing is written to your data either way. Unticking changes only what you
see and what the next cut uses; the object is put back whole by **Discard**,
and only **Accept** ever writes.

**My preview disappeared when I ticked a marker.** That is the tool working,
not a crash. Anything you change in **Sources** puts the preview away and
brings your markers back, because the cut and the markers are drawn in the
same layer and cannot both be on screen. Press **Preview** again to see the
new cut. Ticking a **piece** does not do this — that only changes what Accept
writes.

**The cut ignored an edge I can clearly see.** Tick **Show what the tool
sees** on the Result card and look at `guide_walls` where the cut went wrong.
If the edge is bright there, the wall signal is fine and the problem is
elsewhere — usually a missing marker on one side of it, or *Only cut where I
can see a wall* set too strictly. If the edge is **not** bright there, the tool
never had it: pick a different wall kind, a different channel, or turn on the
preparation steps under **Walls**. On a shape wall there is no channel at all —
the picture is made from the object's own outline — so an edge you can see in
the image is invisible to the cut by construction.

**The tick is grey and I did press Preview.** Read the line under it — it
says which of the three reasons it is. *"Preview first"* means no cut has been
made yet on this object. *"Only a wall-based cut makes a picture"* means you
painted your markers by hand and left the wall kind at **Pinch points (shape)**,
so there is genuinely no wall signal behind the cut; pick a marker finder, or
any other wall kind. *"Too large to show"* means the object's region is over the
memory ceiling, and the sentence below gives the detail.

**`guide_walls` is in my layer list and I did not put it there.** You ticked
**Show what the tool sees**. Remove the tick and it goes; it also goes on
Accept, Discard, a patch switch or closing the annotator. It is a picture only:
nothing reads it, it is never saved, and it is not offered as a channel in the
annotator's own combo or on the `a`/`s`/`d`/`f` keys.

**"Refine Fragments…" is not there.** Since 2026-09-20 it is offered after
every Preview, however the markers were found. It is only missing when there is
nothing to refine: you have unticked every piece, or the tool kept the object
whole (the amber block will say *possibly dividing*). Tick a piece and it comes
back.

**Markers from the last object are not carried to the next one.** Accept
clears the marker list, and only a real commit does: if Accept refuses because
another Step 3 job is running, your ticks are left exactly as they were so you
can press it again once the job finishes.

**Preview says "No markers found in this object" and there obviously are
some.** Press **Find markers**
once — the list it fills is what Preview cuts with.

**I painted markers on "Shape thickness peaks" and they did nothing.**
**Find markers** puts its markers onto the object as ordinary
markers, and anything you add with **Add marker** is used alongside them.

**"I expect exactly n objects" was still ticked on the next object.** The tick
is cleared after every Accept, on purpose — inheriting it is the one setting
that can quietly produce a *wrong* cut rather than an inconvenient one. The
number is kept, so re-arming it is one click. While it is armed, the collapsed
**Sources** line says *expect n*, so it can never be on without showing.

**"I expect exactly n objects" did nothing.** It only ever works in one
direction: the tool **merges pieces down** to your number, and never invents a
cut to reach it. If you asked for 3 and got 2, there were only ever 2 — the fix
is another marker (or a lower Prominence), not a bigger number in the box.
There are six reasons the count can come out other than you expected, and the
console tells them apart. After a Preview, look for the two `[COUNT-DIAG]`
lines (napari's log panel, or the terminal you launched from) — one for what
was asked, one for what happened:

- **Neither line is there.** Both pickers are at their first entry, so this is
  the shape-only cut, which never sees the number at all.
- **`target_count=None`.** The tick above the number is off.
- **`n_seeds=2` with `target_count=3`.** Unreachable: you never get more pieces
  than markers. Add a marker.
- **`n_before=2 target=3`.** Fewer pieces than you asked for, and nothing is
  forced. Same fix.
- **`n_before=3 target=3`.** Already there — you asked for 3 and got 3, so
  there was nothing to merge. Nothing to do.
- **`arbiter_merged=2`**, or a sliver line in the messages with `min_size=20`.
  Something merged pieces back together *before* the count stage ever ran —
  either *Only cut where I can see a wall* decided there was no wall behind a
  cut, or small pieces were folded into their neighbours. Both can be true at
  once.

**The object vanished and I can't get it back.** Press **Discard** — the button
beside Accept. The object is restored exactly as it was. It is also restored if
you switch patches, if you close the annotator, or if the tool keeps the object
whole — whichever tool you were using, hand-painted markers included. Any
markers you had lifted off are dropped along with the session, because they
belong to that object and nothing else.

**Why does the button say Cancel sometimes and Discard other times?** One
rule: **Discard** whenever the object is being held out of the volume, which is
from **Start** until you Accept or Discard. **Cancel** only when nothing is
staged and there is nothing to put back. **Stop waiting** is a different
button with a different job — it abandons a calculation and leaves the object
where it is; only Discard puts the object back.

**Accept is greyed out.** Either you have not run Preview, or another Step 3
job is running — the status line at the top names the job that is holding
things up, and so does Accept's tooltip.

**Nothing appears when I load a guide mask.** Check **Show guide mask**. The
guide has no keyboard shortcut, because it is not one of the layers you are
curating — it is read-only input and is never edited.

**The pieces look right in 3D but not on the slices.** They are the same layer;
scroll through Z and look again. If they genuinely differ, that is a bug worth
reporting.

**The amber block says some of the object was "not given to any piece" — have I
lost it?** No. Those voxels stay part of the object you started from when you
press Accept, exactly as an unticked piece does. The block says so in the same
sentence. A few percent is normal; if it is a lot, the cut is following
something thick — try *Soft cut*, or turn *Snap to the wall signal* off.

**A piece came back unticked and I want it.** Tick it. The tool unticks a
piece less than a fifth the size of the middle one as a suggestion, and says
which piece and how big it is in the amber block. Nothing is decided until
Accept.

**I asked for exactly 3 objects and got 2.** Nothing is forced upwards: the tool
will merge pieces down to a count, but it will never invent a cut to reach
one. Add another marker, or lower *Prominence* under Advanced, and Preview
again.

**Where did "Watershed on:" and "Image layer:" go?** They were retired. They
asked the same question as *Walls from* one chevron above, and had been hidden
since the panel rework; *Walls from* is where that choice lives now.
