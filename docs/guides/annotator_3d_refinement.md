# Refinement — 3D Annotator

The Refinement section is the **interpolation workflow** — fix a label's 3D shape by drawing on a few key slices and letting the algorithm fill the gaps between them. Faster than painting every slice; preserves smooth contours.

---

## When to use it

You'll reach for interpolation when:

- A label's 3D shape is wrong but the cell is in the right general region — typical case is a cell whose boundary the model got right on a couple of slices but smeared or missed on others.
- You want to redraw a label from scratch with clean contours — paint the boundary on a few representative slices, then let interpolation fill the rest.
- You're recovering a missed cell — pick a fresh label ID, paint its silhouette on 2-3 slices spanning the Z-range, interpolate.

**When NOT to use it:** small per-slice touch-ups (just paint directly), cutting one label into several (use Advanced Features → Reshape), or merging two labels (use Advanced Features → Merge).

---

## The workflow

```
Draw on 2+ slices (same label ID, same view — XY or XZ or YZ)
         │
         ▼
    [ Preview ]            ← runs SDF interpolation, shows result
         │
         ▼
   ┌─────────────┐
   │ Looks good? │
   └──┬──────┬───┘
      │      │
     Yes    No
      │      └─→  [ Reject ]  → undoes to your hand-drawn input
      │                          (your strokes are preserved; you can keep editing)
      ▼
  [ Accept ]   →  keeps the result; advances to next label
       OR
  [ Refine Further... ]
       → keeps the result + opens the Refine paint layer
         so you can touch up the interpolated shape immediately
```

Repeat until the label looks right.

---

## The buttons

| Button | When enabled | What it does |
|---|---|---|
| **Preview** (teal) | Always (assuming a label has hand-drawn strokes on ≥2 slices **in the same view** — XY or XZ or YZ) | Runs SDF contour interpolation on the strokes. The 3D shape between your slices fills in via signed-distance morphing. Existing un-drawn slices for this label are replaced by the interpolated result. |
| **Accept** (green) | After a successful Preview | Commits the interpolated result + advances the navigator to the next label (skip via Label Navigator settings). |
| **Reject** (red) | After Preview | Undoes the interpolation, restoring the volume to the state BEFORE Preview ran. Your hand-drawn strokes survive — only the algorithm's fill is removed. |
| **Refine Further...** (cyan) | After Preview | Keeps the interpolated result staged (you can still Reject from inside Refine to undo back to your hand strokes) and opens the Refine paint layer in the Advanced Features dock so you can correct any remaining issues without leaving the workflow. |
| **Clear Edits** | After you've drawn anything on a label | Undoes ALL your drawn strokes on this label and restores its original shape (the pre-edit state when you first touched it). Useful when you've started over from scratch. |

---

## Typical session

1. Open the Label Navigator (Tools section).
2. Pick a label flagged for review.
3. The navigator pans the views and isolates the label.
4. Hit `2` to enter paint mode.
5. Paint the correct boundary on the XY slice you're on. Move to a different Z (scroll wheel or Left/Right arrows) and paint again. Repeat on a third slice if the label is tall.
6. Click **Preview**.
7. Check the 3D result by toggling visibility (`A` / `S` / `D` / `F`) or scrolling through slices.
8. If good → **Accept**. If not → **Reject** and re-paint, OR click **Refine Further...** to keep what's there and touch up.

---

## SDF interpolation — what it actually does

"SDF" is signed distance function — the algorithm computes a smooth distance field from your hand-drawn boundary on each input slice, then interpolates the field between slices in Z. The output is the iso-contour of the interpolated field.

Practical implications:

- **Smooth shapes interpolate well.** A cell shape that's roughly ellipsoidal across Z will fill in convincingly with just 2-3 input slices.
- **Sharp Z-changes need more inputs.** If a cell branches or has a thin neck, draw on more slices through that region. Otherwise the interpolation smooths the branch into a blob.
- **No texture or intensity awareness.** The algorithm doesn't read the image — only your strokes. If the model's prediction was wrong and you don't re-draw, interpolation can't "fix" it from the image alone. (For image-guided refinement, use the Refine tab in Advanced Features.)

---

## Common gotchas

- **Preview fails with "need ≥ 2 annotated slices in one orientation".** The constraint is two slices in the SAME view (two XY slices, OR two XZ slices, OR two YZ slices). Mixing one XY + one XZ doesn't count. If you accidentally switched the active label between strokes, only the slices with the current label count.
- **Accept doesn't advance.** Auto-advance respects the Label Navigator's setting. Open the navigator and check the "Auto-advance" checkbox.
- **Reject restores hand strokes.** Your strokes are snapshotted before Preview runs and restored on Reject.
- **The label shape looks "blobby" after interpolation.** Either you need more input slices, or the SDF smooth is doing what it's meant to do (smoothing). If the cell has fine structure, switch to manual painting on more slices, or use Refine to touch up after Accept.

---

## Related sections

- **3D Curation Tools** (above this) — hotkeys, patch switch, scroll step, launcher overview.
- **Advanced Features dock** — Reshape (one label → many), Refine (reshape one label: smooth, grow, fill holes, follow a membrane), Add (build a label the reconstruction missed). Merge lives on the Label Navigator.
