# Quick Annotate — 2D HITL Annotator

The 2D annotator opens as a floating dock with three image views (XY / XZ / YZ) and a control panel on the right. This guide covers the **Quick Annotate** section — the controls and keyboard shortcuts you'll use most while curating slices.

---

## The three views at a glance

| View | What you see | When you draw here |
|---|---|---|
| **XY** (yellow crosshair) | Top-down slice through the volume | Lateral (top-down) cell boundaries |
| **XZ** (cyan crosshair) | Side slice along Y | Axial cell shape — top-to-bottom |
| **YZ** (magenta crosshair) | Side slice along X | Axial cell shape — the other side |

The three views all show the same patch from different angles. Drawing in one updates the others where they overlap.

---

## Keyboard shortcuts

These are the keys you'll press most often:

| Key | What it does |
|---|---|
| **`2`** | Switch to **paint** mode (default tool when curating) |
| **`1`** | Switch to **pan/zoom** mode (move around, no drawing) |
| **`X`** | Erase the label under the crosshair (atomic — one undo step) |
| **`R`** | Pick the label under the crosshair as the active label, switching to it for further refinement |
| **`A` / `S` / `D` / `F`** | Toggle visibility of image channels (mapped positionally: A = channel 1, S = channel 2, etc.) |
| **`Space` (held)** | Temporary pan while held (napari built-in) |
| **Scroll wheel** | Move through Z-slices on the focused view |

### About brush size

The brush is **locked to size 1** across every drawing surface in EpiCycle, so that every view receives exactly the same edit; curation needs boundaries, not broad strokes. A brush larger than 1 will silently snap back to 1.

A configurable brush will return once the sync layer is redesigned. Until then, draw with single-pixel precision; the model only needs accurate boundaries, not thick painted regions.

---

## Crosshair

Toggle the **Crosshair** checkbox at the top of the panel to show/hide the colored crosshairs that mark the same world position across the three views. Useful as a constant visual reference for "where am I in 3D"; turn off if you find them distracting during fine detail work.

---

## Patch selector + Switch

The **Patch** dropdown shows every patch in the active run; **Switch** loads the selected one into the annotator. Use this when you're curating across multiple patches in the same run and want to jump between them without going back to the main panel.

Switching closes the annotator briefly and reopens it on the new patch. Any unsaved drawing changes are flushed before the switch.

---

## Segment + Accept (per orientation)

Three rows below the Annotate label, one per orientation:

| Row | Buttons | What they do |
|---|---|---|
| **XY** | Segment XY · Accept XY | Run the model on the current XY slice; accept the result for training |
| **XZ** | Segment XZ · Accept XZ | Same for the XZ side view |
| **YZ** | Segment YZ · Accept YZ | Same for the YZ side view |

The active orientation (the one you're currently painting on) tints its button row in the matching crosshair color — yellow XY / cyan XZ / magenta YZ — so you always know which surface your strokes are landing on.

**Typical flow:**
1. Navigate to a slice in one view (scroll wheel or click).
2. Click **Segment** for that orientation — the model predicts cell boundaries.
3. Fix any mistakes by painting (mode `2`) or erasing (key `X`).
4. Click **Accept** for that orientation — the corrected slice is added to your training set.
5. Repeat for the other orientations if needed.

---

## Overlay

A read-only visual aid showing the model's confidence on the predicted slice:

| Mode | Shows |
|---|---|
| **Off** | No overlay |
| **Probability** | Heat map of how confident the model is at each pixel (brighter = more confident) |
| **Flow** | The model's flow-field prediction (advanced; useful for debugging stitch errors) |

**Opacity** controls how strongly the overlay tints the image (0–100%). Doesn't affect the underlying segmentation — only what you see while editing.

---

## Common gotchas

- **Drawing on the main viewer does nothing.** The main viewer is locked to pan-zoom; all painting happens on the three subviewers. This is deliberate — it prevents accidental edits at the wrong scale.
- **Pressing `2` doesn't always work.** If you're focused outside a subviewer (e.g., on a settings dialog), napari's keybindings don't fire. Click into one of the three views first.
---

## Related sections

- **Main panel curation guides** — the Step 2 main panel `?` buttons cover patch management, slices, model selection, and training setup.
