# EpiCycle QC Features: A Practical Guide for Biologists

**Module:** `napari_epicycle/fusion_reconstruction/qc_features.py`
**Last updated:** 2026-10-14
**Companion files:** `QC_FEATURE_REFERENCE.md` (equations and definitions for every feature), `FEATURE_CATALOG.csv` (one row per feature: definition, rationale, units, optional inputs).

This document explains every quality-control (QC) feature that EpiCycle extracts
from 3D segmentation masks. It is written for biologists and microscopists — no
computer science background needed. Each feature includes:

- **What it measures** in plain language
- **How the mask is used** to compute it (step by step)
- **The equation** (with variables defined)
- **What abnormal values mean** biologically (merge, fragment, noise)

---

## How to Read This Document

Every feature starts from the **3D segmentation mask** — a volume where each
cell is assigned a unique integer label (1, 2, 3, ...) and the background is 0.
Think of it as a 3D coloring book where each cell is a different color.

When we say "the mask for label 5", we mean: all the voxels (3D pixels) in the
volume that have the value 5.

Some features also use an **intensity image** (the raw fluorescence, e.g.
membrane-GFP) or a **probability volume** (how confident the reconstruction
algorithm was at each voxel). These are optional and noted where relevant.

**Spacing** (voxel size in physical units like micrometers) is used throughout.
For example, if your voxels are 0.3 x 0.1 x 0.1 um, one voxel along Z covers
3x more physical distance than one voxel in XY.

---

## Table of Contents

| # | Feature | Category | Section |
|---|---------|----------|---------|
| 1-6 | area, area_filled, equivalent_diameter, extent, euler_number, fill_ratio | Core Morphology | [1](#1-core-morphology) |
| 7-10 | sphericity, aspect_ratio, bbox_volume, compactness | Bounding Box Shape | [2](#2-bounding-box-shape) |
| 10a | discrete_compactness | Discrete (Bribiesca) Compactness | [2.5](#25-discrete-compactness-bribiesca) |
| 11-19 | z_span, z/xy_elongation, major_axis_z_angle, inertia_elongation/flatness, ellipsoid_solidity, axis lengths | Orientation & Inertia | [3](#3-orientation--inertia-ellipsoid) |
| 20-23 | surface_area, true_sphericity, roughness, solidity | Surface | [4](#4-surface-features) |
| 24-27 | intensity_mean/std/min/max | Intensity | [5](#5-intensity-features) |
| 28-31 | boundary/interior intensity, ratio, z_variation | Membrane | [6](#6-boundaryinterior-membrane-features) |
| 32-35 | neighbor_count, nearest_neighbor_distance, max_contact_fraction, background_fraction | Adjacency | [7](#7-neighboradjacency-features) |
| 36-38 | skeleton_branches, skeleton_nodes, skeleton_length | Skeleton Topology | [8](#8-skeleton-topology-features) |
| 39-51 | prob_mean, prob_p10, prob_std, boundary/interior prob, entropy, gradients | Probability Consensus | [9](#9-probability-consensus-features) |
| 52 | qc_score | Composite Score | [10](#10-composite-qc-score) |
| 53-58 | apical_area, basal_area, apical_basal_area_ratio, max_cross_section_z_relative, cross_section_taper, cross_section_bulge | Z-Stratified Cross-Section | [11](#11-z-stratified-cross-section-features) |
| 59-61 | mean_neighbor_size_ratio, neighbor_size_cv, contact_weighted_neighbor_size_ratio | Neighbor Quality | [12](#12-neighbor-quality-features) |
| 62-67 | nn_count_r1, nn_count_r3, nn_birth_median, nn_birth_std, nn_accumulation_rate, nn_accumulation_knee | Birth-Radius Filtration | [13](#13-neighbor-birth-radius-filtration) |
| 68-71 | basal_neighbor_count, apical_neighbor_count, neighbor_z_turnover, neighbor_persistence_ratio | Z-Stratified Topology | [14](#14-z-stratified-neighbor-topology) |
| 72-75 | merge_signal, fragment_signal, boundary_confidence, shape_abnormality | Named Composite Signals | [15](#15-named-composite-signals) |
| 76 | is_border_cell | Border Cell Detection | [16](#16-border-cell-detection) |
| 77-79 | centroid-0, centroid-1, centroid-2 | Position | [17](#17-position-centroid-coordinates) |
| 80-89 | {channel}_intensity_mean/median/std/min/max/sum/cv/boundary/interior/ratio | Per-Channel Intensity | [18](#18-per-channel-intensity-features) |

---

## 1. Core Morphology

These features describe the most basic properties of each cell's mask. They are
always computed.

### `area` (Volume)

**What it measures:** The total volume of the cell mask — literally, how many
voxels belong to this cell, converted to physical units.

**How the mask is used:**
1. Count every voxel where `mask == label_id`
2. Multiply by voxel volume: `voxel_size_z * voxel_size_y * voxel_size_x`

**Equation:**

    V = N_voxels * s_z * s_y * s_x

where N_voxels is the count of voxels belonging to this label, and s_z, s_y, s_x
are the voxel dimensions in micrometers.

**What to look for:**
- **Merged cells** are roughly 2x the expected volume (two cells counted as one)
- **Fragments** are much smaller than typical cells
- The volume systematically *underestimates* true cell volume by ~5-15% because
  the segmentation mask stops short of the membrane (the gap between cells)

---

### `area_filled` (Hole-Filled Volume)

**What it measures:** The volume of the cell mask *after filling in any internal
holes*. If a cell mask has a hollow cavity inside it, `area_filled` counts those
cavity voxels too.

**How the mask is used:**
1. Take the cell's binary mask
2. Fill any enclosed internal cavities (background voxels completely surrounded
   by the cell mask)
3. Count the resulting voxels

**What to look for:**
- For most well-segmented cells, `area_filled == area` (no internal holes)
- If `area_filled > area`, the mask has internal cavities, which may indicate
  reconstruction artifacts

---

### `equivalent_diameter`

**What it measures:** "If this cell's volume were reshaped into a perfect sphere,
what would its diameter be?" Converts volume into a linear length scale that's
easier to compare across cells.

**Equation:**

    d_eq = (6V / pi)^(1/3)

where V is the cell volume.

**Intuition:** A cell with volume 1000 um^3 has an equivalent diameter of about
12.4 um.

---

### `extent`

**What it measures:** How much of the cell's bounding box (the tightest
rectangular box that encloses the cell) is actually filled by the cell.

**How the mask is used:**
1. Find the smallest axis-aligned 3D box that contains all of the cell's voxels
2. Compute: `cell volume / box volume`

**Equation:**

    extent = V / V_bbox

**What to look for:**
- A sphere fills about 52% of its bounding box (extent ~ 0.52)
- **Low extent** (< 0.3) indicates an irregular, L-shaped, or tentacle-like
  object — possibly a merge artifact
- **High extent** (> 0.7) indicates a compact, block-like cell

---

### `euler_number`

**What it measures:** A topological invariant that counts "how many holes and
tunnels go through the cell." Think of it like asking: "Is this shape a solid
ball, a donut, or a pretzel?"

**Equation:**

    chi = (# of connected pieces) + (# of enclosed cavities) - (# of tunnels)

**How the mask is used:** Computed directly from the 3D binary mask using the
Euler characteristic algorithm in scikit-image.

**What to look for:**
- **chi = 1** for a simple solid cell (expected for most real cells)
- **chi < 1** means tunnels — strongly suggests a merge artifact where two cell
  lumens are connected through a narrow passage
- **chi > 1** means enclosed cavities (e.g., a cell-within-a-cell artifact)
- One of the most informative single features for detecting merges

---

### `fill_ratio`

**What it measures:** Ratio of actual volume to hole-filled volume. Tells you
whether the cell mask is solid or has internal cavities.

**Equation:**

    fill_ratio = V / V_filled

**What to look for:**
- **= 1.0** means the mask is completely solid (no internal holes)
- **< 1.0** means there are internal cavities (rare in well-segmented cells)

---

## 2. Bounding Box Shape

These features use the **axis-aligned bounding box** — the smallest rectangular
3D box (aligned with the Z, Y, X axes) that completely encloses the cell.

### `sphericity` (Bounding-Box Sphericity)

**What it measures:** How closely the cell's bounding box resembles a cube (which
would contain a sphere). Compares the surface area of a volume-equivalent sphere
to the surface area of the bounding box.

**How the mask is used:**
1. Compute the equivalent sphere radius from cell volume:
   `r_eq = d_eq / 2`
2. Compute the equivalent sphere surface area:
   `S_sphere = 4 * pi * r_eq^2`
3. Compute the bounding box surface area:
   `S_bbox = 2*(bz*by + by*bx + bz*bx)`
4. Ratio: `S_sphere / S_bbox`

**Equation:**

    sphericity = (4 * pi * (d_eq/2)^2) / (2 * (bz*by + by*bx + bz*bx))

where bz, by, bx are the bounding box dimensions in physical units.

**Note:** This is a *proxy* — it uses the bounding box, not the actual surface.
For true surface-based sphericity, see `true_sphericity` in section 4.

---

### `aspect_ratio`

**What it measures:** How elongated the bounding box is. The ratio of its longest
dimension to its shortest.

**Equation:**

    aspect_ratio = max(bz, by, bx) / min(bz, by, bx)

**What to look for:**
- **~1** = roughly cubical bounding box (compact cell)
- **> 3** = very elongated — possibly two cells merged end-to-end
- Note: a flat, disc-like cell can also have a high aspect ratio

---

### `bbox_volume`

**What it measures:** The total volume of the bounding box.

**Equation:**

    V_bbox = bz * by * bx

---

### `compactness`

**What it measures:** What fraction of the bounding box is filled by the cell.

**Equation:**

    compactness = V / V_bbox

**Note:** This is mathematically identical to `extent` (both = V / V_bbox).
Both are included for compatibility; in practice you only need one.

---

## 2.5 Discrete Compactness (Bribiesca)

### `discrete_compactness`

**What it measures:** How "cube-like" the cell's voxel mask is, on a scale from 0 to 1. Counts how many of the cell's voxel faces are *internal* (touching another voxel of the same cell) vs *external* (touching background or a different cell). A perfect cube has the most internal contacts possible per voxel.

**Equation (Bribiesca, *Pattern Recognition* 41(3), 2008):**

    C_d = (n - A_v / 6) / (n - n^(2/3))

where `n` = voxel count and `A_v` = number of external (boundary) voxel faces.

**Bounded [0, 1]**:

| Shape | C_d |
|---|---|
| Single isolated voxel | 0 (max surface per voxel) |
| Voxelized sphere | ~0.85 |
| Perfect cube | 1.0 |
| Thin rod (1×1×40 voxels) | ~0.5 |

**Why it's useful biologically:** The classical sphericity measures (`true_sphericity`, `sphericity`) operate on a smoothed surface — marching cubes produces a continuous mesh that "rounds off" voxel jaggedness. `discrete_compactness` operates directly on the voxel face count, so it captures *jaggedness* and *fragmentation* that the smoothed measures miss. For QC, a healthy cell has high values for both; a bumpy/fragmented mask has lower `discrete_compactness` even if the smoothed surface looks fine. The two together give the PCA two roughly orthogonal "irregularity" axes.

**When to use it:**
- As a fragmentation/over-segmentation detector — bad masks have low `discrete_compactness` even when their `true_sphericity` is acceptable.
- In PCA / clustering when you want a shape-irregularity signal that's not derived from the marching-cubes mesh.
- It's bounded [0, 1], so the EDA workbench's "auto" transform will not log-transform it.

**Reference:** Bribiesca, E. (2008). "An easy measure of compactness for 2D and 3D shapes." *Pattern Recognition*, 41(3), 856-868.

---

## 3. Orientation & Inertia Ellipsoid

These features come from the **inertia tensor** — a 3x3 matrix that describes
how the cell's mass (voxels) is distributed in 3D space. Think of it like
asking: "If I spun this cell around each axis, how hard would it be to rotate?"

The inertia tensor has three **eigenvalues** (I0 >= I1 >= I2) and three
**eigenvectors** (the axes of rotation). The eigenvalues tell you how spread
out the mass is along each principal axis. The eigenvectors tell you which
directions those axes point.

From these, we derive an imaginary **best-fit ellipsoid** — the 3D egg shape
that best matches how the cell's mass is distributed.

### `z_span`

**What it measures:** How many micrometers the cell extends along the Z-axis
(the imaging/slicing axis).

**Equation:**

    z_span = (bbox_z_max - bbox_z_min) * s_z

where s_z is the Z voxel spacing.

**What to look for:**
- Unusually large z_span may indicate two cells stacked on top of each other
  (apical-basal merge)

---

### `z_elongation`

**What it measures:** How stretched the cell is along the Z-axis compared to its
widest lateral (XY) dimension.

**Equation:**

    z_elongation = bz / max(by, bx)

**What to look for:**
- **> 1** = taller than wide (elongated along the imaging axis)
- **High z_elongation + high z_span** = strong signal for apical-basal stacked
  merges in Cellpose 2D-to-3D reconstructions

---

### `xy_elongation`

**What it measures:** How stretched the cell is within the imaging plane
(lateral elongation).

**Equation:**

    xy_elongation = max(by, bx) / min(by, bx)

**What to look for:**
- **> 2** = elongated in the XY plane — may indicate a lateral merge (two
  cells side by side counted as one)

---

### `major_axis_z_angle`

**What it measures:** The angle (in degrees) between the cell's longest axis
and the Z-axis. "Is this cell standing upright or lying on its side?"

**How the mask is used:**
1. Compute the 3x3 inertia tensor from the cell's voxel positions
2. Eigendecompose: find the three principal axes and their eigenvalues
3. The eigenvector with the **smallest** eigenvalue points along the direction
   of greatest elongation (the "major axis")
4. Measure the angle between this axis and the Z-direction

**Equation:**

    theta = arccos(|v_major . z_hat|)

where v_major is the major axis direction (eigenvector of smallest eigenvalue),
z_hat is the unit vector along Z, and the absolute value handles the sign
ambiguity (axes have no preferred direction).

**What to look for:**
- **0 degrees** = cell elongated along Z (standing upright)
- **90 degrees** = cell elongated in the XY plane (lying flat)
- Z-aligned cells (low angle) with high z_elongation strongly suggest
  apical-basal stacked merges

---

### `inertia_elongation`

**What it measures:** How elongated the cell is overall, regardless of
orientation. Uses the ratio of the most-spread to least-spread eigenvalue.

**Equation:**

    inertia_elongation = sqrt(I0 / I2)

where I0 is the largest eigenvalue (axis with most spread = least elongated)
and I2 is the smallest (most elongated axis).

**What to look for:**
- **~1** = roughly spherical (mass equally spread in all directions)
- **>> 1** = elongated in some direction — merges in any axis show up here
- More informative than bounding-box aspect_ratio for irregularly shaped cells

---

### `inertia_flatness`

**What it measures:** Whether the cell is disc-shaped (flat) or rod-shaped
(elongated like a stick).

**Equation:**

    inertia_flatness = sqrt(I1 / I0)

where I1 is the middle eigenvalue and I0 is the largest.

**What to look for:**
- **Close to 1** = disc-like (two large dimensions, one small)
- **Close to 0** = rod-like (one large dimension, two small) = likely merge

---

### `ellipsoid_solidity`

**What it measures:** How well the cell fills its best-fit inertia ellipsoid.
"If I drew the best-fitting 3D egg around this cell, would the cell fill it up,
or would there be a lot of empty space?"

**How the mask is used:**
1. From the three eigenvalues (I0 >= I1 >= I2), compute the semi-axis lengths
   of the best-fit ellipsoid:

       a^2 = 2.5 * (I0 + I1 - I2)    (longest semi-axis)
       b^2 = 2.5 * (I0 - I1 + I2)    (middle)
       c^2 = 2.5 * (-I0 + I1 + I2)   (shortest semi-axis)

   (No division by mass needed because scikit-image returns mass-normalized
   eigenvalues — already divided by voxel count.)

2. Compute the ellipsoid volume:

       V_ellipsoid = (4/3) * pi * sqrt(a^2 * b^2 * c^2)

3. Ratio:

       ellipsoid_solidity = V / V_ellipsoid

**What to look for:**
- **> 1** = cell is larger than its inertia ellipsoid (common for round cells
  whose mass distribution implies a smaller ellipsoid)
- **<< 1** = cell is much smaller than its ellipsoid. This is the hallmark of
  **dumbbell-shaped merges**: two lobes connected by a narrow neck inflate the
  inertia ellipsoid (because the mass is spread far from center), but the actual
  cell volume is small relative to that ellipsoid

---

### `axis_major_length` / `axis_minor_length`

**What they measure:** The length of the longest and shortest axes of the
best-fit ellipsoid (in physical units).

**Equation:**

    axis_major_length = 2 * sqrt(a^2)    (= 2a, the full diameter)
    axis_minor_length = 2 * sqrt(c^2)    (= 2c)

where a^2 and c^2 are the ellipsoid semi-axis values computed above.

---

## 4. Surface Features

These features measure the actual 3D surface of the cell mask using triangle
meshes. Much more accurate than bounding-box features for detecting surface
irregularities.

### `surface_area`

**What it measures:** The true 3D surface area of the cell mask, measured by
extracting a triangle mesh from the surface.

**How the mask is used:**
1. Crop the label volume to the cell's bounding box
2. Create a binary mask (1 = this cell, 0 = everything else)
3. Pad with one voxel of background on all sides (so the algorithm can find the
   complete surface)
4. Run the **marching cubes** algorithm at the 0.5 isosurface — this generates a
   triangle mesh that wraps the cell's surface
5. Sum the area of all triangles

**Equation:**

    S = sum of triangle areas (from marching cubes mesh, respecting voxel spacing)

**What to look for:**
- Large surface area relative to volume suggests an irregular, convoluted shape
- Compare with `equivalent_diameter` to assess if the surface area is larger
  than expected for a smooth cell

---

### `true_sphericity` (Wadell Sphericity)

**What it measures:** How closely the cell's actual surface area matches that of
a perfect sphere with the same volume. This is the gold-standard sphericity
metric — it uses the real surface, not just the bounding box.

**Equation:**

    psi = (pi^(1/3) * (6V)^(2/3)) / S

This is equivalent to:

    psi = S_equivalent_sphere / S_actual

**Range:** 0 to 1, where 1 = perfect sphere.

**What to look for:**
- **> 0.8** = very round cell
- **< 0.5** = highly irregular surface — merge artifacts with convoluted
  boundaries score low here
- More sensitive than bounding-box sphericity to surface wrinkles and dumbbell
  shapes

---

### `roughness`

**What it measures:** How "wrinkled" the cell surface is compared to a smooth
sphere. The inverse of true_sphericity.

**Equation:**

    roughness = S / S_equivalent_sphere
             = S / (4 * pi * r_eq^2)

where:

    r_eq = (3V / (4*pi))^(1/3)

**Range:** >= 1.0, where 1.0 = perfectly smooth sphere.

**What to look for:**
- **~1** = smooth surface
- **> 2** = very rough/irregular surface, likely a merge artifact
- Note: this is exactly 1/true_sphericity, so they carry the same information.
  Both are provided for convenience.

---

### `solidity`

**What it measures:** How much of the cell's convex hull (the tightest convex
wrapper) is actually filled by the cell. "If I shrink-wrapped the cell with a
taut elastic sheet, how much of the interior would be cell vs. empty space?"

**How the mask is used:**
1. Crop the label volume to the cell's bounding box
2. Create a binary mask, then erode it by one voxel
3. Subtract: the **surface voxels** are the mask minus the eroded mask
4. Convert surface voxel coordinates to physical units (multiply by spacing)
5. Subsample to ~1500 points
6. Compute the 3D convex hull of those points using scipy
7. Ratio: `cell volume / convex hull volume`

**Equation:**

    solidity = V / V_convex_hull

**What to look for:**
- **> 0.8** = compact cell that fills its convex hull well
- **< 0.6** = strong merge signal. Dumbbell-shaped merges (two lobes connected
  by a narrow neck) have a huge convex hull but a relatively small actual volume
  because of the concave "waist" between the lobes

---

## 5. Intensity Features

These require the **raw fluorescence image** (e.g., membrane-GFP channel) to be
loaded alongside the segmentation mask. They measure the fluorescence signal
*inside* each cell's mask.

### `intensity_mean`

**What it measures:** The average fluorescence brightness inside the cell mask.

**How the mask is used:**
1. Select all voxels where `labels == label_id`
2. Read the fluorescence intensity at those voxel positions
3. Compute the mean

**Equation:**

    I_mean = (1/N) * sum of I(v) for all voxels v in the cell mask

**What to look for (membrane-GFP):**
- **Low** = correct segmentation. The cell interior is cytoplasm (GFP-negative),
  so correctly segmented cells should be dark inside
- **High** = possible merge. When two cells are incorrectly merged, the bright
  membrane between them falls *inside* the merged mask, raising the average
  intensity

---

### `intensity_std`

**What it measures:** How variable the fluorescence intensity is across the cell.

**Equation:**

    I_std = sqrt((1/N) * sum of (I(v) - I_mean)^2)

**What to look for:**
- Very high std in large labels = heterogeneous interior = merge signal
  (bright membrane band + dark cytoplasm regions within the same label)

---

### `intensity_min` / `intensity_max`

**What they measure:** The darkest and brightest voxel values within the cell mask.

---

## 6. Boundary/Interior Membrane Features

These are specifically designed for **membrane-fluorescence** segmentation QC
(e.g., membrane-GFP, E-cadherin staining). The idea: correctly segmented cells
should have bright boundaries (where the membrane sits) and dark interiors
(cytoplasm). Merged cells have bright signal *inside* because the internal
membrane between the two original cells is now captured within the mask.

### How the boundary and interior are defined

1. Start with the cell's binary mask
2. **Erode** the mask by 2 voxels (shrink it inward by ~2 voxels on all sides)
3. The eroded mask = **interior**
4. The original mask minus the eroded mask = **boundary shell** (~2 voxels thick)

```
    Original mask:          Eroded (interior):      Boundary shell:
    +-----------+           +-------+               +-----------+
    |xxxxxxxxxxx|           |       |               |xxxxxxxxxxx|
    |xxxxxxxxxxx|           | xxxxx |               |xx       xx|
    |xxxxxxxxxxx|           | xxxxx |               |xx       xx|
    |xxxxxxxxxxx|           | xxxxx |               |xx       xx|
    |xxxxxxxxxxx|           |       |               |xxxxxxxxxxx|
    +-----------+           +-------+               +-----------+
```

For tiny cells where erosion eliminates the entire interior, the full mask mean
is used as a fallback.

---

### `boundary_intensity_mean`

**What it measures:** Average fluorescence in the outer ring (boundary shell).

**Equation:**

    I_bnd = (1/|boundary|) * sum of I(v) for v in boundary

---

### `interior_intensity_mean`

**What it measures:** Average fluorescence in the eroded inner core.

**Equation:**

    I_int = (1/|interior|) * sum of I(v) for v in interior

---

### `boundary_interior_ratio`

**What it measures:** The ratio of boundary brightness to interior brightness.
This is often the **single most discriminative feature** for detecting merges
in membrane-fluorescence data.

**Equation:**

    R = I_bnd / I_int

**What to look for:**
- **R >> 1** (typically 2-10x) = correct segmentation. Bright membrane at the
  boundary, dark cytoplasm inside. This is what you want.
- **R ~ 1** = merged cells. The internal membrane raises the interior intensity
  to match the boundary.
- **R < 1** = unusual (boundary darker than interior) — possible reconstruction
  artifact or background noise label

**Why it works despite the gap:** Even though the segmentation mask stops short
of the actual membrane (there's a ~2-voxel gap between labels), the GFP signal
from the membrane **bleeds inward** due to the microscope's point spread function
(PSF). The boundary shell of the mask still captures significant membrane signal.

---

### `z_intensity_variation`

**What it measures:** How much the average intensity varies from slice to slice
as you move through the cell along Z.

**How the mask is used:**
1. For each Z-slice that the cell occupies:
   a. Select voxels in that slice belonging to this cell
   b. Compute the mean intensity for that slice
2. Compute the standard deviation across all per-slice means

**Equation:**

    sigma_z = std({mean intensity of slice z : z in [z_min, z_max]})

**What to look for:**
- **High sigma_z** = there's a bright band at some Z-level inside the cell.
  This is the hallmark of **apical-basal merges**: two cells stacked vertically
  with a bright membrane plane separating them. One Z-slice will be very bright
  (the membrane plane) while others are dark (cytoplasm).

---

## 7. Neighbor/Adjacency Features

These features describe how each cell relates to its neighbors. They use
**graduated dilation** — a technique specifically designed for reconstructions
where cells don't quite touch (there are 1-5 voxel gaps between them).

### How graduated dilation works

Instead of a single fixed dilation, the cell mask is expanded outward one voxel
at a time, from radius 1 to a maximum (default 5):

```
    Radius 0 (original):   Radius 1:              Radius 2:
    +---+                   +-----+                +-------+
    | A |                   |  A  |                |   A   |
    +---+                   +-----+                +-------+
                  gap                    gap
    +---+                   +-----+                +-------+
    | B |                   |  B  |                |   B   |
    +---+                   +-----+                +-------+

    At radius 1: A and B still separated by gap
    At radius 3: dilation rings from A and B overlap — neighbors detected!
```

At each radius, new voxels are added as a "ring". The algorithm checks what
labels (or background) those ring voxels overlap with. This way:
- Touching cells are found at radius 1
- Cells with a 3-voxel gap are found at radius 3
- Truly isolated fragments are never found (distance = max_radius + 1)

---

### `neighbor_count`

**What it measures:** How many distinct other cells are nearby.

**Equation:**

    N_neighbors = count of unique non-background labels found across all
                  dilation rings (excluding self)

**What to look for:**
- Cells in packed epithelial tissue typically have **4-8 neighbors**
- **0 neighbors** = isolated fragment (suspicious)
- **Very high** neighbor count may indicate a large merged region touching many
  cells

---

### `nearest_neighbor_distance`

**What it measures:** The approximate gap width (in voxels) between this cell
and its closest neighbor. "How many voxels of background separate me from the
nearest other cell?"

**Equation:**

    d_nn = the dilation radius at which the first non-background neighbor
           was discovered

**What to look for:**
- **1** = touching (no gap)
- **2-4** = typical for non-touching reconstructions
- **max_radius + 1** (default: 6) = truly isolated, no neighbors found within
  search range

---

### `max_contact_fraction`

**What it measures:** Of all the "contact shell" voxels around this cell (found
across all dilation rings), what fraction belongs to the single most-contacted
neighbor?

**Equation:**

    f_max = (contact voxels with most-contacted neighbor) / (total shell voxels)

**What to look for:**
- **> 0.5** = more than half the contact shell faces one neighbor. This pair may
  be an over-segmentation split (two fragments of the same cell that should be
  merged)

---

### `background_fraction`

**What it measures:** What fraction of the contact shell faces background
(empty space) rather than other cells.

**Equation:**

    f_bg = (background voxels in all rings) / (total shell voxels)

**What to look for:**
- **High** = surface/border cell or isolated artifact
- In non-touching segmentations, all cells have somewhat elevated
  background_fraction (because of the gaps), but the relative ranking still
  identifies the most isolated labels

---

## 8. Skeleton Topology Features

These features analyze the **medial axis** (skeleton) of the cell — a thin,
stick-figure-like representation of the cell's core shape. Like the spine inside
a body, the skeleton captures the cell's topology without its bulk.

### How the skeleton is extracted

1. Crop the label volume to the cell's bounding box
2. Pad with 1 voxel of background (so the distance transform measures distance
   to the true boundary, not the array edge)
3. Compute the **Euclidean Distance Transform (EDT)**: at each voxel inside the
   cell, how far is it from the nearest background voxel?
4. The skeleton = **local maxima** of the distance transform (the ridgeline of
   the distance mountain). Only voxels with distance > 1 are kept.
5. Branch points = skeleton voxels with more than 2 skeleton neighbors (in
   26-connectivity, i.e., including diagonals)

```
    Cell mask cross-section:     Distance transform:     Skeleton:
    +-------------------+        +-------------------+   +-------------------+
    |xxxxxxxxxxxxxxxxxxx|        |1 1 1 1 1 1 1 1 1 1|   |                   |
    |xxxxxxxxxxxxxxxxxxx|        |1 2 2 2 2 2 2 2 2 1|   |                   |
    |xxxxxxxxxxxxxxxxxxx|        |1 2 3 3 3 3 3 3 2 1|   |     * * * * *     |
    |xxxxxxxxxxxxxxxxxxx|        |1 2 2 2 2 2 2 2 2 1|   |                   |
    |xxxxxxxxxxxxxxxxxxx|        |1 1 1 1 1 1 1 1 1 1|   |                   |
    +-------------------+        +-------------------+   +-------------------+
```

---

### `skeleton_branches`

**What it measures:** Number of skeleton branches. A simple, rounded cell has
~1 branch. A dumbbell-shaped merge (two lobes connected by a neck) has >= 2.

**Equation:**

    branches = (number of branch points) + 1

A branch point is a skeleton voxel with more than 2 skeleton neighbors.

**What to look for:**
- **1** = simple cell shape
- **>= 2** = complex shape with branching — may indicate a merge

---

### `skeleton_nodes`

**What it measures:** Number of branch points (junctions) in the skeleton.

**Equation:**

    nodes = count of skeleton voxels where
            (number of skeleton neighbors in 3x3x3 cube) > 2

---

### `skeleton_length`

**What it measures:** Total number of voxels in the skeleton — a proxy for the
cell's linear extent or complexity.

**What to look for:**
- Longer skeletons = more elongated or complex shapes
- Very long relative to cell volume = likely elongated merge

---

## 9. Probability Consensus Features

These features use the **reconstruction probability volume** — a 3D map produced
by u-Segment3D during the reconstruction step. At each voxel, the probability
represents how confidently the algorithm believes a cell is present, based on
agreement between predictions from three orthogonal views (XY, XZ, YZ):

- **Probability ~1.0** = all three views agree: "there's definitely a cell here"
- **Probability ~0.5** = views disagree (maximum uncertainty)
- **Probability ~0.0** = all views agree: "this is background"

The boundary shell and interior are defined by erosion, exactly as in the
membrane intensity features (section 6).

### Group 1: Confidence Statistics

#### `prob_mean`

**What it measures:** Average reconstruction confidence across the cell mask.

**Equation:**

    prob_mean = (1/|M|) * sum of P(v) for v in mask

**What to look for:**
- **Low** = the reconstruction was uncertain about this cell — few views
  detected it
- **High** = strong multi-view agreement

---

#### `prob_p10` (10th Percentile)

**What it measures:** The probability value below which the worst 10% of the
cell's voxels fall. More sensitive to localized problems than the mean.

**Equation:**

    prob_p10 = 10th percentile of {P(v) : v in mask}

**What to look for:**
- **Low prob_p10** even when prob_mean is decent = one region of the cell is
  poorly supported (e.g., a "bad slice" in the Z-stack where one view failed)

---

#### `prob_std`

**What it measures:** How variable the probability is across the cell.

**Equation:**

    prob_std = standard deviation of {P(v) : v in mask}

**What to look for:**
- **High** = mixed confidence (some regions well-supported, others not)
- **Low** = consistent confidence throughout (good or bad, but uniform)

---

### Group 2: Boundary vs Interior Probability

#### `prob_boundary_mean` / `prob_interior_mean`

**What they measure:** Average probability in the boundary shell and eroded
interior, respectively.

---

#### `prob_boundary_drop`

**What it measures:** The difference between interior and boundary probability.

**Equation:**

    delta = prob_interior_mean - prob_boundary_mean

**What to look for:**
- **Positive** = confident core, fuzzy edges (normal for cells at tissue edges)
- **~0** = uniform probability (good agreement everywhere)
- **Negative** = edges more confident than core (unusual — possible artifact)

---

### Group 3: Soft Volume vs Hard Mask

#### `prob_weighted_volume` (V_soft)

**What it measures:** Total probability mass in and around the cell. Sum of all
probability values within the cell's bounding box plus a 3-voxel padding.

**Equation:**

    V_soft = sum of P(v) for all voxels v in (padded bounding box)

Note: this sums over ALL voxels in the padded region, not just mask voxels.
It captures probability mass in the surrounding area too.

---

#### `prob_soft_hard_ratio`

**What it measures:** Ratio of the soft volume (probability mass) to the hard
mask volume (voxel count). Detects mismatches between what the probability field
says and what the segmentation mask says.

**Equation:**

    R = V_soft / |M|

**What to look for:**
- **R << 1** = **Over-segmentation signal**. The mask extends into a region
  where the reconstruction was not confident. The label claims territory the
  probability map doesn't support.
- **R >> 1** = **Under-segmentation signal**. There's significant probability
  mass around the cell that the mask doesn't capture. The reconstruction
  detected more cell presence than the label covers.
- **R ~ 1** = good agreement between probability field and mask boundary

---

### Group 4: Uncertainty / Entropy

Uses **voxel-wise binary entropy** — a measure of how uncertain the
reconstruction was at each voxel:

    H(v) = -P(v) * ln(P(v)) - (1-P(v)) * ln(1-P(v))

- H = 0 when P = 0 or P = 1 (completely certain)
- H = ln(2) ~ 0.693 when P = 0.5 (maximum uncertainty — views disagree equally)

#### `prob_boundary_entropy`

**What it measures:** Mean uncertainty at the cell boundary.

**Equation:**

    U_bnd = (1/|boundary|) * sum of H(v) for v in boundary

**What to look for:**
- **High** = the reconstruction was uncertain about where to place the cell
  boundary. Most direct measure of boundary reliability.

---

#### `prob_mask_entropy`

**What it measures:** Mean uncertainty across the entire cell mask.

**Equation:**

    U_mask = (1/|M|) * sum of H(v) for v in mask

---

### Group 5: Edge Sharpness

Uses the **gradient magnitude** of the probability field — how rapidly the
probability changes at each voxel:

    |grad P(v)| = sqrt((dP/dz)^2 + (dP/dy)^2 + (dP/dx)^2)

#### `prob_edge_gradient`

**What it measures:** Average gradient magnitude at the cell boundary. "How
sharply does the probability drop off at the cell edge?"

**Equation:**

    G_bnd = (1/|boundary|) * sum of |grad P(v)| for v in boundary

**What to look for:**
- **High** = sharp probability drop at boundary (confident, well-defined edge)
- **Low** = gradual, diffuse boundary (uncertain edge placement)

---

#### `prob_edge_gradient_std`

**What it measures:** How uniform the edge sharpness is around the cell.

**Equation:**

    sigma_G = std({|grad P(v)| : v in boundary})

**What to look for:**
- **Low** = consistently sharp (or consistently diffuse) boundary
- **High** = patchy — some boundary regions are sharp, others are uncertain

---

## 10. Composite QC Score

### `qc_score`

**What it measures:** A single number summarizing how "unusual" each cell is
relative to the rest of the population. Higher = more outlier-like. This is not
a pass/fail score — it's a ranking tool that puts the most suspicious cells at
the top of the review queue.

**How it's computed (z-score method, default):**

1. Select a curated subset of 15 decorrelated features covering
   morphology, shape, surface, adjacency, topology and probability
2. **Log-transform** right-skewed features (like area, surface_area) using
   `log(1 + x)` — this prevents a few huge cells from dominating the analysis
3. **Robust standardize** each feature using median and MAD (median absolute
   deviation) instead of mean/std — this makes the normalization resistant to
   the very outliers we're trying to detect
4. **Winsorize** to +/-10 robust z-scores (clip extreme values to prevent any
   single feature from dominating)
5. Compute the **mean absolute z-score** across all features

**Equation:**

    For each feature j:
        z_ij = (x_ij - median_j) / (MAD_j * 1.4826)
        z_ij = clip(z_ij, -10, 10)

    qc_score_i = (1/F) * sum of |z_ij| for all features j

where MAD_j is the median absolute deviation of feature j, and the 1.4826
constant makes MAD comparable to standard deviation for normally distributed
data.

**Why robust standardization matters:** Regular mean/std is heavily influenced
by outliers — the very cells you want to detect would inflate the standard
deviation, making themselves look less unusual. Median/MAD is immune to this:
outliers remain visibly extreme.

**Alternative (PCA method):** Projects the standardized features onto their top
principal components (the axes of maximum variance; 5 by default) and uses the
cell's distance from the average cell in that space, so outliers score high.
More mathematically sophisticated but less interpretable.

**Key property:** The score is **population-relative**. If all cells in a dataset
are flat, "flat" is normal and won't be flagged. Only cells that are unusual
*compared to their neighbors* score high.

---

### Tier assignment

After computing `qc_score`, cells are assigned to quality tiers based on three
user-adjustable thresholds:

| Tier | Condition | Meaning |
|------|-----------|---------|
| **good** | score <= t1 | Looks normal — no review needed |
| **review** | t1 < score <= t2 | Somewhat unusual — worth a look |
| **bad** | t2 < score <= t3 | Clearly abnormal — likely needs correction |
| **garbage** | score > t3 | Extreme outlier — probably junk |
| **curated** | (manual) | Manually verified by the user — never overwritten |

Manually curated labels are always preserved — threshold re-assignment never
overwrites a "curated" tier.

---

## 11. Z-Stratified Cross-Section Features

**Source:** `compute_cross_section_features()` in `qc_features.py`
**Requires:** 3D label volume only (no intensity image needed)

These features capture **epithelial cell polarity** by analyzing how the
cross-section area changes along the cell's Z-extent. In stratified epithelia,
cells have distinct shapes depending on their layer: basal cells are often
pyramidal (wide base, narrow top), while superficial cells are flat.

**Convention:** Lower Z = basal, upper Z = apical. If your stack is inverted,
use the "Z orientation" toggle before extraction.

### `basal_area` / `apical_area`

**What it measures:** The average cross-section area in the bottom 20% (basal)
or top 20% (apical) of the cell's Z-range.

**How it works:**
1. Find the cell's Z extent: z_min to z_max
2. Divide into basal zone (bottom 20%) and apical zone (top 20%)
3. For each Z-slice in each zone, count the number of mask voxels
4. Average the per-slice areas within each zone

**Units:** Voxels (or physical area if spacing is set).

**What abnormal values mean:**
- Very large basal_area + small apical_area = pyramidal cell (normal in folds/crypts)
- Very large apical_area + small basal_area = inverted pyramid (unusual, possible delamination)
- Both very small = fragment or debris

### `apical_basal_area_ratio`

**What it measures:** The ratio of apical to basal cross-section area.

**Equation:** `apical_basal_area_ratio = apical_area / basal_area`

| Value | Cell Shape | Biological Meaning |
|-------|-----------|-------------------|
| < 0.5 | Pyramidal (wide base, narrow top) | Normal in folds, crypts, apical constriction |
| ~ 1.0 | Columnar (uniform width) | Normal in flat monolayers |
| > 2.0 | Inverted pyramid (wide top, narrow base) | Unusual — possible artifact or delaminating cell |

**Why it matters:** This is the single most direct measure of apical-basal
polarity in 3D. Nothing else in the feature set captures this shape distinction.
In PCA, it provides a unique polarity axis independent of size.

### `max_cross_section_z_relative`

**What it measures:** The Z-position of the cell's widest cross-section,
normalized to [0, 1] within the cell's Z-range.

**Equation:** `max_cross_section_z_relative = (z_of_max_area - z_min) / z_range`

| Value | Where the cell is widest |
|-------|-------------------------|
| 0.0 | Widest at the bottom (basal) |
| 0.5 | Widest in the middle (barrel or vase shape) |
| 1.0 | Widest at the top (apical) |

### `cross_section_taper`

**What it measures:** The slope of a linear fit of cross-section area versus
normalized Z. Captures the dominant trend: is the cell getting wider or
narrower from bottom to top?

**Equation:** Linear regression slope of `area_per_slice ~ z_normalized`

- **Negative slope** = tapering upward (pyramidal)
- **Positive slope** = expanding upward (inverted)
- **~0** = columnar (uniform width)

### `cross_section_bulge`

**What it measures:** How much the cell's cross-section deviates from a
simple linear taper. High bulge means the cell has a vase, barrel, or
dumbbell shape.

**Equation:** `bulge = RMS(residuals_of_linear_fit) / mean(area_per_slice)`

- **Low bulge + negative taper** = clean pyramid
- **Low bulge + zero taper** = clean column
- **High bulge** = vase, barrel, or dumbbell (strong merge signal — two lobes with a narrow waist)

---

## 12. Neighbor Quality Features

**Source:** `compute_neighbor_features()` in `qc_features.py`
**Requires:** 3D label volume + neighbor map from graduated dilation

These are **cell-contextual features** — they compare a cell's size
to its neighbors. A cell might look perfectly normal by its own morphology,
but if it's 3x larger than everything around it, it's probably a merge.

### `mean_neighbor_size_ratio`

**What it measures:** This cell's volume divided by the average volume of its
neighbors.

**Equation:** `mean_neighbor_size_ratio = V_cell / mean(V_neighbors)`

| Value | Meaning |
|-------|---------|
| ~ 1.0 | Similar size to neighbors (normal) |
| > 2.0 | Much larger than neighbors (possible merge) |
| < 0.5 | Much smaller than neighbors (possible fragment) |

**Why it matters:** This catches cases where a cell's own morphology looks fine
but it doesn't fit its local context. The intrinsic features (area,
aspect_ratio, etc.) can't detect this.

### `neighbor_size_cv`

**What it measures:** Coefficient of variation (std/mean) of neighbor volumes.

**What it tells you:** High CV means this cell sits at a boundary between
different-sized populations — for example, between a well-segmented region and
a region full of merged blobs.

### `contact_weighted_neighbor_size_ratio`

**What it measures:** Like `mean_neighbor_size_ratio`, but weighted by how much
boundary surface is shared with each neighbor. Neighbors with more contact
contribute more to the ratio.

**Why it matters:** Emphasizes the neighbors that are actually touching, not
distant ones discovered at higher dilation radii.

---

## 13. Neighbor Birth-Radius Filtration

**Source:** `compute_neighbor_features()` in `qc_features.py`
**Requires:** 3D label volume

Instead of evaluating the neighborhood at a single fixed radius, these features
record the full **filtration** — which neighbors appear at each step of the
graduated dilation. Think of it like slowly inflating each cell: at radius 1,
only directly touching neighbors are found; at radius 5, neighbors separated
by a 5-voxel gap are included.

### `nn_count_r1`

**What it measures:** Number of neighbors found at radius 1 (direct contact —
masks physically touching).

**Interpretation:** The "hard" neighbor count. In a tissue with 1-2 voxel gaps
between cells, this counts only cells whose masks actually share a boundary.

### `nn_count_r3`

**What it measures:** Number of neighbors found within radius 3.

**Interpretation:** Includes near-neighbors separated by typical 1-2 voxel
gaps from the segmentation pipeline. This is usually the most biologically
meaningful neighbor count.

### `nn_birth_median`

**What it measures:** The median "birth radius" across all discovered neighbors.
A neighbor's birth radius is the dilation step at which it was first found.

**Why it matters:** More robust than `nearest_neighbor_distance` (which is just
the minimum). Tells you the *typical* gap width to neighbors, not just the
closest one.

### `nn_birth_std`

**What it measures:** Standard deviation of birth radii across all neighbors.

| Value | Meaning |
|-------|---------|
| Low | All neighbors at similar distance (uniform packing) |
| High | Mix of touching and distant neighbors (heterogeneous geometry, e.g., one side packed, other side facing a gap or tissue edge) |

### `nn_accumulation_rate`

**What it measures:** Slope of a linear fit to the cumulative neighbor count
versus radius curve.

- **Steep slope** = dense tissue, neighbors appearing rapidly at low radii
- **Flat slope** = sparse region, neighbors trickling in slowly
- Captures the "density" of the local neighborhood in a single number

### `nn_accumulation_knee`

**What it measures:** The radius at which 80% of total neighbors have been found.

| Value | Meaning |
|-------|---------|
| Early (r=2-3) | Tight packing — most neighbors very close |
| Late (r=8+) | Scattered arrangement — neighbors spread across distances |

---

## 14. Z-Stratified Neighbor Topology

**Source:** `compute_neighbor_features()` in `qc_features.py`
**Requires:** 3D label volume + contact Z-coordinates per neighbor

These features track **which neighbors are present at different Z-levels**
within the cell. This captures the 3D arrangement of the local neighborhood
along the apical-basal axis — information that no other feature provides.

**Minimum Z-span guard:** For cells spanning fewer than 5 Z-slices, these
features are set to 0 (unreliable when "thirds" become 1-2 slices).

### `basal_neighbor_count` / `apical_neighbor_count`

**What it measures:** Count of distinct neighbors contacting the cell in the
bottom third (basal) or top third (apical) of its Z-range.

### `neighbor_z_turnover`

**What it measures:** What fraction of neighbors change from bottom to top.

**Equation:** `turnover = 1 - |basal_set AND apical_set| / |basal_set OR apical_set|`

| Value | Meaning |
|-------|---------|
| 0 | Same neighbors at top and bottom (columnar packing, neighbors span full height) |
| ~0.5 | Partial rearrangement (some persistent, some changing) |
| 1 | Completely different neighbors at top vs bottom (multilayer, tissue fold, or complex 3D stacking) |

**Why it matters:** This is the single most informative scalar for characterizing
**3D tissue organization** at the single-cell level. In a simple columnar
epithelium, turnover is low. In a stratified or folded tissue, turnover is high.

### `neighbor_persistence_ratio`

**What it measures:** Fraction of ALL neighbors that span the full cell height
(present in both the top and bottom thirds).

**Equation:** `persistence_ratio = |basal_set AND apical_set| / total_neighbors`

- **High** = stable columnar packing (neighbors are side-by-side the whole way)
- **Low** = transient contacts (neighbors only share part of the cell height)

---

## 15. Named Composite Signals

**Source:** `compute_named_composites()` in `qc_features.py`
**Requires:** Feature DataFrame (all primitives must be computed first)

Targeted outlier scores for specific failure modes. Instead of one generic
`qc_score`, these give biologists **actionable categories**: "this cell looks
like a merge" vs "this cell looks like a fragment."

All composites use the same robust z-score methodology as `qc_score`
(median/MAD standardization, winsorized to +/-10), but with handpicked inputs
and directional weights.

### `merge_signal`

**What it measures:** How "merge-like" is this cell?

**Inputs (weighted):**
- `boundary_interior_ratio` (inverted — low ratio = merge)
- `inertia_elongation` (high = merge)
- `skeleton_branches` (high = merge)
- `euler_number` (inverted — low euler = merge)
- `cross_section_bulge` (high = dumbbell merge)
- `mean_neighbor_size_ratio` (high = larger than neighbors)

**Interpretation:** Higher = more suspicious. A cell scoring high on merge_signal
has low membrane contrast, elongated shape, many skeleton branches, and is larger
than its neighbors — all hallmarks of two cells fused together.

### `fragment_signal`

**What it measures:** How "fragment-like" is this cell?

**Inputs (weighted):**
- `area` (inverted — small = fragment)
- `neighbor_count` (inverted — isolated = fragment)
- `background_fraction` (high = isolated)
- `mean_neighbor_size_ratio` (inverted — smaller than neighbors)

**Interpretation:** Higher = more suspicious. A cell scoring high is tiny,
isolated, surrounded by background, and much smaller than its neighbors.

### `boundary_confidence`

**What it measures:** How confident is the boundary segmentation?

**Inputs:** `prob_edge_gradient` (+), `1 - prob_boundary_entropy` (+),
`prob_boundary_mean` (+)

**Note:** Requires probability features (from u-Segment3D probability volume).
Set to NaN when probability features are not available (e.g., after smooth
re-extraction).

**Interpretation:** Higher = MORE confident (not inverted). Cells with low
boundary_confidence have uncertain, blurry, or ambiguous boundaries.

### `shape_abnormality`

**What it measures:** Population-relative shape outlier score.

**Inputs:** `apical_basal_area_ratio` (+), `cross_section_taper` (+),
`inertia_elongation` (+), `true_sphericity` (inverted)

**Interpretation:** Higher = more unusual shape for this tissue. Catches cells
whose shapes are unusual for their population context.

---

## 16. Border Cell Detection

**Source:** `compute_border_cells()` in `qc_features.py`

### `is_border_cell`

**What it measures:** Whether the cell's mask touches any face of the 3D
volume (Z=0, Z=max, Y=0, Y=max, X=0, X=max).

**Type:** Binary (0 or 1)

| Value | Meaning |
|-------|---------|
| 0 | Interior cell — fully contained within the volume |
| 1 | Border cell — touches a volume boundary |

**Why it matters:** Border cells have **truncated morphology** — their real
shape extends beyond the imaged volume, so all size, shape, and surface
features are systematically biased. EpiCycle excludes border cells from
PCA fitting (they are projected as supplementary points) and can optionally
remove them during re-extraction.

---

## 17. Position (Centroid Coordinates)

**Source:** `regionprops` via `extract_features()` in `qc_features.py`

### `centroid-0` / `centroid-1` / `centroid-2`

**What they measure:** The center of mass of the cell's mask, in voxel
coordinates.

| Feature | Axis | Biological Meaning |
|---------|------|-------------------|
| `centroid-0` | Z | Depth in the tissue (basal-apical position) |
| `centroid-1` | Y | Lateral position |
| `centroid-2` | X | Lateral position |

**Units:** Voxels (not physical units, even when spacing is set — these are
coordinate positions, not distances).

**Why they matter:**
- **Color-by-Z** reveals tissue stratification in PCA/UMAP plots
- **Gating on Z** selects cells from a specific tissue layer
- **Spatial autocorrelation** — if neighboring cells in the scatter plot are
  also spatial neighbors, your features are capturing real tissue structure

**PCA note:** Centroids are excluded from PCA by default (unchecked in the
EDA Workbench). Position is not a morphological feature — including it in
PCA would make PC1 "where in the tissue" rather than "what shape."

**Orientation vectors and inertia eigenvalues:** A results table also carries
`major_axis_z/y/x`, `mid_axis_z/y/x` and `minor_axis_z/y/x` (the direction of
each of the cell's three principal axes, as unit vectors in Z, Y, X; the "major"
axis points along the most elongated direction) and `inertia_eigval_major`,
`inertia_eigval_mid`, `inertia_eigval_minor` (the three inertia-tensor
eigenvalues). They describe orientation rather than quality, and are not used in
the composite score. `major_axis_z_angle` (section 3) is derived from the major
axis.

---

## 18. Per-Channel Intensity Features

**Source:** `compute_channel_intensity()` in `qc_features.py`
**Requires:** Multi-channel source stack + channel configuration in Step 4 UI

When your source image has multiple fluorescence channels (e.g., membrane-GFP,
nuclear-DAPI, a signaling marker), EpiCycle extracts intensity features for
each channel separately. You name the channels in the "Intensity Channels" UI
(e.g., "Membrane0", "Nuclei"), and the features are prefixed with that name.

### Features per channel (11 per channel)

For a channel named `Nuclei`, the features are:

| Feature | What it measures |
|---------|-----------------|
| `Nuclei_intensity_mean` | Average brightness of this channel inside the cell |
| `Nuclei_intensity_median` | Median brightness (robust to outlier pixels) |
| `Nuclei_intensity_std` | Variation in brightness (heterogeneity) |
| `Nuclei_intensity_min` | Darkest voxel in the mask for this channel |
| `Nuclei_intensity_max` | Brightest voxel in the mask |
| `Nuclei_intensity_sum` | Total integrated fluorescence (proxy for protein amount) |
| `Nuclei_intensity_cv` | Coefficient of variation (std/mean) — normalized variability, comparable across brightness levels |
| `Nuclei_boundary_intensity_mean` | Mean brightness at the 2-voxel boundary shell |
| `Nuclei_interior_intensity_mean` | Mean brightness in the eroded interior |
| `Nuclei_boundary_interior_ratio` | boundary/interior. >1 = signal at membrane. ~1 = cytoplasmic. <1 = nuclear/interior enriched. |
| `Nuclei_z_intensity_variation` | Standard deviation of the per-Z-slice mean brightness: how much the signal changes along the cell's height |

**Biological examples:**
- **Membrane marker** (E-cadherin, ZO-1): Expect `boundary_interior_ratio > 1`.
  Low ratio on a membrane marker = suspicious segmentation (boundary doesn't
  align with real membrane).
- **Nuclear marker** (DAPI, H2B): Expect `boundary_interior_ratio < 1`.
  The signal should be concentrated in the interior.
- **Cytoplasmic marker** (actin, tubulin): Expect ratio ~1 (uniform).
- **`intensity_sum`**: Scales with cell size — larger cells have more total
  fluorescence. Use `intensity_mean` for size-independent comparisons.
- **`intensity_cv`**: High CV suggests punctate or heterogeneous staining
  within the cell.

**PCA note:** Per-channel features are NOT included in the default PCA feature
set but can be checked in the EDA Workbench. Consider using `intensity_mean`
and `boundary_interior_ratio` per channel as the most informative pair.

---

## 19. Non-Touching Segmentations: Why Features Still Work

In EpiCycle's 3D reconstructions, there is typically a **1-5 voxel gap** of
background between neighboring cells (from Cellpose flow-field convergence,
Z-fusion margins, and membrane-width exclusion). Here's why each feature
category still works:

| Category | Impact of Gap | Why It Still Works |
|----------|--------------|-------------------|
| Volume (area) | Underestimates by ~5-15% | Systematic bias, affects all cells equally; merges still 2x |
| Bounding box | Unaffected | Box determined by label extent, not touching |
| Surface | Slightly smoothed | Mask surface is smoother than true membrane; still detects merge distortion |
| Solidity | Works well | Convex hull reflects mask shape; dumbbell concavity preserved |
| Intensity | Works well | Membrane GFP signal bleeds ~2-3 voxels inward due to PSF; mask edge still captures it |
| Boundary/interior | Robust | PSF blurring ensures boundary shell captures membrane signal; ratio discriminates merges |
| Neighbors | Designed for gaps | Graduated dilation (r=1..5) bridges typical gaps; nearest_neighbor_distance reads out gap width |
| Skeleton | Unaffected | EDT operates within each label independently |
| Probability | Independent of gaps | Probability field reflects multi-view agreement, not mask geometry |
| Composite score | Self-adjusting | Population-relative normalization cancels systematic biases |
