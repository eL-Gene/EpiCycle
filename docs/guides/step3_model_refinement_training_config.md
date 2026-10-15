# Training Configuration

The parameters that control how the model actually learns from your extracted training pairs.

---

"Use a preset or set the numbers yourself."

---

## Mode (in Fine-Tuning)

Which view(s) to fine-tune — the **Mode** box at the top of Fine-Tuning:

| Choice | What it does |
|---|---|
| **Fine-tune Lateral** | Train only the XY (top-down) model |
| **Fine-tune Axial** | Train only the XZ / YZ (side-on) model |
| **Fine-tune Both** | Train lateral first, then axial automatically, saved in one bundle |

For a full refinement pass, **Both** is the typical choice. The hint under the box says in one sentence which models start from where.

---

## Preset

Pick a **Preset** and click **Apply** to fill Epochs, Learning rate, Batch size and the Advanced values in one go. The line under the box says what the preset sets — read it before applying, because Apply overwrites the fields (Save Config first to keep yours). A preset never chooses where training starts; that is Init's job.

| Preset | What it sets |
|---|---|
| default | "100 epochs, learning rate 0.01, batch 16. The numbers the panel opens with; a safe pick when no other preset fits." |
| quick | "100 epochs, learning rate 0.01, batch 32. A larger batch runs faster; use it to check whether new curated data helped before a long run." |
| thorough | "1000 epochs, learning rate 0.01, batch 16. A long run for a final model; it is slow, so use it sparingly." |
| finetune | "500 epochs, learning rate 0.001 (ten times lower). Small steps keep what the starting model already knows; pair it with an Init that starts from a trained model." |
| dense_tissue | "750 epochs, batch 8, the minimum-masks filter off. A smaller batch needs less memory on crowded slices, and every fragment stays in as training signal." |
| stratified_epithelia | "500 epochs, learning rate 0.01, channels green (segment) and red (auxiliary). Changes nothing if you have not moved the channel defaults." |

---

## Channels (in Advanced Parameters)

Two combo boxes — one for each Cellpose channel slot. Cellpose expects two channels: one for **membranes** and one for **nuclei**.

> **What to pick:** put the channel index that matches your **membrane stain** in slot 1, and your **nuclei stain** in slot 2.
>
> Indices follow Cellpose's convention: `0 = gray`, `1 = red`, `2 = green`, `3 = blue` — they refer to physical channel position in your image, not biological meaning.
>
> **Default `[2, 1]`:** green = membrane, red = nuclei. Adjust based on your imaging setup.
>
> **Single-channel data:** pick `0: gray` for slot 1 and `0: none` for slot 2.

---

## Init (in Fine-Tuning)

Where to **start** the model training — five choices:

| Choice | When to use |
|---|---|
| **From HITL model** | Start from the model your Step 2 rounds produced (the Source box picks which). **The usual first refinement.** |
| **From prior refinement** | Start from a model you refined earlier (the Source box lists them). Best for **iterative refinement** — picking up from where you left off last round. |
| **From Cellpose pretrained** | Start from one of Cellpose's general-purpose models (cyto3, nuclei, etc.). Cellpose's model already knows what cells look like; you're teaching it your specific tissue. |
| **Import model** | Start from a model file on disk; Browse to the file. |
| **From scratch** | Train from zero. Slowest, needs a lot of curated data, and rarely the right call. Starting a model from scratch by accident is refused: pick a Source, or choose From scratch on purpose. |

With Mode = Both each choice has an **Axial** row of its own, so the lateral and axial models can start from different places.

---

## Epochs

How many complete passes the model makes through your training pairs.

- **More epochs** = the model learns finer patterns from your hand-curated cells
- **Too many** = the model starts memorizing your training cells instead of generalizing to new ones
- **Default 100** works for most cases

If the training loss is still dropping at the end (visible in the live loss plot under the Start Training button), bump it up. If it flattens out early, you can stop sooner.

---

## Learning rate

How aggressively the model adjusts after each batch of slices.

- **Smaller** = safer, slower convergence
- **Larger** = faster, but training can become unstable
- **Default works for most cases.** Only lower this if training starts oscillating or diverging.

---

## Batch size

How many image slices the GPU processes at once.

- **Bigger** = faster per epoch, more GPU memory used
- **If you hit a `CUDA out of memory` error, lower this.** Common values: 1, 2, 4, 8.

---

## Train / val split

What fraction of your extracted pairs gets held out for validation — not used in training, just used to measure how well the model is generalizing.

**Default 0.8 / 0.2** (80% train, 20% val) is standard and rarely needs changing.

### Keep neighbouring slices together (recommended ON)

The split lives under **Extraction Settings** in the left column (train fraction, chunk size and the “Keep neighbouring slices together” tick).

When ON (default), the splitter groups **contiguous adjacent slices** into chunks (default size 10) and assigns entire chunks to train or val — never splits adjacent slices across sets. Why this matters:

Adjacent Z slices in your volume are *almost identical* (same cells, one voxel apart). If slice 200 lands in train and slice 201 lands in val, the model has effectively seen the val data already during training — val loss becomes meaninglessly low.

Keeping neighbouring slices together puts slices 200–209 *together* in either train or val, so the nearest train slice to a val slice is ≥ 10 indices away — far enough to be a genuinely separate prediction problem.

### Worked example: 2 patches, no cap, Fine-tune Lateral

Suppose you extracted:

- **patch_A** → 50 valid XY slices → files `xy_0005_img.tif` ... `xy_0245_img.tif` (sorted by Z)
- **patch_B** → 40 valid XY slices → files `xy_0003_img.tif` ... `xy_0210_img.tif`

`_collect_training_pairs` for Fine-tune Lateral builds one flat list, alphabetical patch order, sorted within each patch:

```
positions 0..49   patch_A's 50 XY stems (ascending Z)
positions 50..89  patch_B's 40 XY stems (ascending Z)
                  ────────────────────
                  90 total stems
```

The splitter then:

```
n_chunks    = ceil(90 / 10) = 9 chunks of 10
n_train     = round(9 × 0.8) = 7  →  RNG picks 7 chunks for train, 2 for val
```

Example RNG outcome — train chunks = {0, 1, 2, 3, 5, 7, 8}, val chunks = {4, 6}:

```
positions   chunk   set     stems
  0..9        0     train   patch_A Z=~5..70
 10..19       1     train   patch_A Z=~75..140
 20..29       2     train   patch_A Z=~145..205
 30..39       3     train   patch_A Z=~210..240
 40..49       4     val     patch_A Z=~245..285  ← last 10 of A
 ──────────────────────────  patch boundary
 50..59       5     train   patch_B Z=~3..40
 60..69       6     val     patch_B Z=~45..100
 70..79       7     train   patch_B Z=~105..170
 80..89       8     train   patch_B Z=~175..210
```

**Result:** 70 train (40 from A + 30 from B), 20 val (10 from A + 10 from B). Both patches contribute proportionally on average.

Inside chunk 4 (val), the 10 slices are *contiguous*. The nearest train slice is Z = ~240 (last of chunk 3) or Z = ~290 (first of chunk 5 → patch B, unrelated). So the val set is a genuinely held-out region of the volume.

### Patch boundaries

Chunks usually align cleanly within one patch (50 slices ÷ chunk_size 10 = 5 clean chunks). When patch sizes don't divide evenly, one chunk straddles the boundary — harmless because cross-patch slices aren't spatially correlated.

### Mode Both = two independent splits

If you pick **Mode = Both**, two separate train/val splits run sequentially:

1. **Lateral pass** — collect every patch's `spatial/`, split with its own RNG outcome, train `lateral_model`.
2. **Axial pass** — collect every patch's `axial/`, split with a *different* RNG outcome, train `axial_model`.

The val sets are unrelated between the two. A specific Z slice that contributes to the lateral val set has no determined fate in the axial split — that's expected, since axial views are independent images of the same volume.

### Keep neighbouring slices together OFF

If you turn this off, the split is pure random shuffle on the flat list — every slice independently goes to train or val. **Avoid this** unless you really want pure randomness and accept that adjacent slices will leak across sets.

---

## Model name

Optional. Left empty, the bundle is named `refined_NN` — the next free two-digit number in this run's models folder (`refined_01`, `refined_02`, …; the box says "refined_NN if empty"). Type a name of your own for a domain model (for example `wound_membrane`); it is used as typed, and a name already taken in the run is refused. With Mode = Both one name covers both models.

---

## Save Config / Load Config

- **Save Config** persists the current settings to
  `artifacts/step3/refinement/refinement_config.json` in your workspace.  There
  is one saved config per workspace, shared by every run.
- **Load Config** restores previously-saved settings.  A project that saved a
  config before this folder existed kept it loose in the workspace folder; that
  older file is still read, and is left where it is.

Useful if you've worked out a good preset for your tissue and want it to persist across sessions.

---

## Related

- **Fine-Tuning** — once these settings are configured, click Start Training to launch training
- **Patch Selection** — which patches contributed to the training pairs you're configuring for here
