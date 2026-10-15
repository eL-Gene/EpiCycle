# 3D Curation Tools — Annotator

The 3D annotator opens as a floating dock with three image views (XY / XZ / YZ) and a control panel on the right. This guide covers the panel's top section + the keyboard shortcuts. **Refinement** workflow (interpolation Preview/Accept/Reject) has its own `?` button next to that section header.

The Label Navigator and Advanced Features dock launched from this panel have their own UI; help for those will live with them in a follow-up.

---

## The three views at a glance

| View | Crosshair color | Shows |
|---|---|---|
| **XY** | Yellow | Top-down slice |
| **XZ** | Cyan | Side slice along Y |
| **YZ** | Magenta | Side slice along X |

All three show the same patch from different angles. Drawing in one view updates the others where the strokes overlap.

---

## Keyboard shortcuts

| Key | What it does |
|---|---|
| **`2`** | Switch to **paint** mode (most common tool during curation) |
| **`1`** | Switch to **pan/zoom** mode |
| **`A` / `S` / `D` / `F`** | Toggle visibility of image channels (positional — A = channel 1, S = channel 2, etc.) |
| **`Space` (held)** | Temporary pan |
| **Left / Right arrows** | Step through Z-slices on the focused subviewer by the **Scroll Step** size |
| **Up / Down arrows** | Jump to the previous / next label in the Label Navigator |
| **Scroll wheel** | Step one slice at a time on the focused subviewer (independent of Scroll Step) |
| **Right-click** in a subviewer | Pick the label under the cursor + focus it in the Label Navigator |

### About brush size

The brush is **locked to size 1** across every drawing surface in EpiCycle, so that every view receives exactly the same edit; curation needs boundaries, not broad strokes. A brush larger than 1 will silently snap back to 1.

A configurable brush will return once the sync layer is redesigned. Until then, draw with single-pixel precision; curation only needs accurate boundaries.

---

## Patch + Scroll Step

| Control | What it does |
|---|---|
| **Patch** dropdown | Lists every patch in the current workspace |
| **Switch** | Loads the selected patch into the annotator |
| **Scroll Step** | How many slices the Left/Right arrows skip per press (1–100). Default 1. Useful for big patches — set to 5 or 10 to navigate large Z-ranges quickly. |

Switching patches closes the dock briefly and reopens it on the new patch. Any unsaved drawing changes are flushed first.

---

## Tools section (launchers)

Two buttons at the bottom of the panel — each opens a separate floating dock or dialog:

| Button | What it opens |
|---|---|
| **Label Navigator...** | The triage table — every label in this patch with thumbnails, QC tier, color, status filters. The main hub for "show me what to fix next." |
| **Advanced Features...** | Floating dock with three curation tools, each on the same layout: **Reshape** (cut one over-merged cell into several), **Refine** (reshape one cell — smooth, grow, fill holes, follow a membrane) and **Add** (build a cell the reconstruction missed). Each tab has its own `?`. |

Each of these has its own UI surface; help for them will be added in a follow-up branch.

---

## Common gotchas

- **Drawing on the main viewer does nothing.** The main 3D viewer is locked to pan/zoom; all painting happens on the three subviewers. This is deliberate — prevents accidental edits at the wrong scale.
- **Up/Down arrows navigate the Label Navigator even if its dock is closed** — as long as the navigator has been opened at least once this session. If you press Up/Down on a fresh launch and nothing happens, open Label Navigator… first.
- **Right-click sometimes doesn't fire.** It's blocked while an interpolation / split / smooth operation is actively running (the Preview button is in flight), while a merge dialog is open, or when the navigator's right-click toggle has been turned off. Once Preview is staged (Accept / Reject visible), right-click works again.
- **Scroll Step > 1 changes arrow-key behavior.** When Scroll Step is 2+, Left/Right arrows jump by that many slices. For one-slice-at-a-time, use the scroll wheel instead.

---

## Related sections

- **Refinement** (below this in the panel) — SDF interpolation workflow. Click the `?` next to the Refinement header.
- **Step 3 main panel guides** — patch management, reconstruction, QC features.
