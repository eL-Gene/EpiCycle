# EpiCycle QC Feature Reference

**Module:** `napari_epicycle/fusion_reconstruction/qc_features.py`
**Last updated:** 2026-10-14
**Companion files:** `FEATURE_CATALOG.csv` (one row per feature: definition, rationale, units, optional inputs), `QC_FEATURES_BIOLOGIST_GUIDE.md` (plain-language guide with interpretation).

This document provides a complete reference for all QC features extracted from
3D reconstructed segmentation volumes in EpiCycle. Designed for membrane-GFP
fluorescence microscopy of cell segmentation, these features enable interactive
outlier detection, merge/fragment identification, and population-level quality
assessment.

## How to read the feature catalogue

`FEATURE_CATALOG.csv` has one row per measurement and three flag columns:

- `in_composite_qc` = 1: the feature is one of the 15 that make up the QC score (listed in section 2.10).
- `in_morphospace` = 1: the feature is one of the 31 shape measurements of the paper's morphospace analysis (the default list of `tutorials/analysis/01_cell_shape_morphospace.ipynb`).
- `in_named_composite`: the named signal(s) the feature feeds, for example `shape_abnormality` or `merge_signal`; the sign says whether a high value pushes the signal up (+) or down (-).

A feature with the flags empty or 0 is still measured and shown in the plugin; it is just not part of the QC score or the morphospace.

---

## 1. Feature Summary

| # | Feature | Category | Source | Cost |
|---|---------|----------|--------|------|
| 1 | `area` | Morphology | regionprops | Trivial |
| 2 | `area_filled` | Morphology | regionprops | Trivial |
| 3 | `equivalent_diameter` | Morphology | regionprops | Trivial |
| 4 | `extent` | Morphology | regionprops | Trivial |
| 5 | `euler_number` | Topology | regionprops | Trivial |
| 6 | `sphericity` | Shape | Derived (bbox) | Trivial |
| 7 | `aspect_ratio` | Shape | Derived (bbox) | Trivial |
| 8 | `bbox_volume` | Shape | Derived (bbox) | Trivial |
| 9 | `compactness` | Shape | Derived (area/bbox_vol) | Trivial |
| 10 | `z_span` | Orientation | Derived (bbox) | Trivial |
| 11 | `z_elongation` | Orientation | Derived (bbox) | Trivial |
| 12 | `xy_elongation` | Orientation | Derived (bbox) | Trivial |
| 13 | `fill_ratio` | Morphology | Derived (area/area_filled) | Trivial |
| 14 | `ellipsoid_solidity` | Shape | Derived (inertia eigenvalues) | Trivial |
| 15 | `major_axis_z_angle` | Orientation | Derived (inertia eigenvectors) | Trivial |
| 16 | `inertia_elongation` | Shape | Derived (eigenval ratio) | Trivial |
| 17 | `inertia_flatness` | Shape | Derived (eigenval ratio) | Trivial |
| 18 | `axis_major_length` | Shape | regionprops | Trivial |
| 19 | `axis_minor_length` | Shape | regionprops | Trivial |
| 19a | `discrete_compactness` | Shape | Voxel face counting (Bribiesca) | Cheap |
| 20 | `surface_area` | Surface | Marching cubes | Moderate |
| 21 | `true_sphericity` | Surface | Derived (volume + SA) | Moderate |
| 22 | `roughness` | Surface | Derived (SA / equiv sphere SA) | Moderate |
| 23 | `solidity` | Surface | Subsampled convex hull | Moderate |
| 24 | `intensity_mean` | Intensity | regionprops | Cheap* |
| 25 | `intensity_std` | Intensity | regionprops | Cheap* |
| 26 | `intensity_min` | Intensity | regionprops | Cheap* |
| 27 | `intensity_max` | Intensity | regionprops | Cheap* |
| 28 | `boundary_intensity_mean` | Membrane | Per-label erosion | Moderate* |
| 29 | `interior_intensity_mean` | Membrane | Per-label erosion | Moderate* |
| 30 | `boundary_interior_ratio` | Membrane | Derived (boundary/interior) | Moderate* |
| 31 | `z_intensity_variation` | Membrane | Per-Z-slice std | Moderate* |
| 32 | `neighbor_count` | Adjacency | Graduated dilation | Moderate |
| 33 | `nearest_neighbor_distance` | Adjacency | Graduated dilation | Moderate |
| 34 | `max_contact_fraction` | Adjacency | Graduated dilation | Moderate |
| 35 | `background_fraction` | Adjacency | Graduated dilation | Moderate |
| 36 | `skeleton_branches` | Topology | EDT medial axis | Moderate |
| 37 | `skeleton_nodes` | Topology | EDT medial axis | Moderate |
| 38 | `skeleton_length` | Topology | EDT medial axis | Moderate |
| 39 | `qc_score` | Composite | Z-score or PCA | Trivial |
| 40 | `prob_mean` | Probability | Per-label mask mean | Moderate† |
| 41 | `prob_p10` | Probability | 10th percentile | Moderate† |
| 42 | `prob_std` | Probability | Std deviation | Moderate† |
| 43 | `prob_boundary_mean` | Probability | Boundary shell mean | Moderate† |
| 44 | `prob_interior_mean` | Probability | Eroded interior mean | Moderate† |
| 45 | `prob_boundary_drop` | Probability | Interior − boundary | Moderate† |
| 46 | `prob_weighted_volume` | Probability | Soft volume (prob sum over ROI) | Moderate† |
| 47 | `prob_soft_hard_ratio` | Probability | V_soft / V_hard | Moderate† |
| 48 | `prob_boundary_entropy` | Probability | Mean binary entropy at boundary | Moderate† |
| 49 | `prob_mask_entropy` | Probability | Mean binary entropy over mask | Moderate† |
| 50 | `prob_edge_gradient` | Probability | Mean |∇P| at boundary | Moderate† |
| 51 | `prob_edge_gradient_std` | Probability | Std |∇P| at boundary | Moderate† |
| 52 | `cross_section_taper`, `cross_section_bulge`, `max_cross_section_z_relative`, `basal_area`, `apical_area`, `apical_basal_area_ratio` | Cross-section | Per-Z-slice area profile A(z) | Cheap |
| 53 | `nn_count_r1`, `nn_count_r3`, `nn_birth_median`, `nn_birth_std`, `nn_accumulation_rate`, `nn_accumulation_knee` | Adjacency (filtration) | Graduated dilation, per-neighbor first-contact radius | Moderate |
| 54 | `mean_neighbor_size_ratio`, `neighbor_size_cv`, `contact_weighted_neighbor_size_ratio` | Adjacency (size) | Neighbor volumes | Moderate |
| 55 | `basal_neighbor_count`, `apical_neighbor_count`, `neighbor_z_turnover`, `neighbor_persistence_ratio` | Adjacency (Z-stratified) | Contacts per basal/apical third | Moderate |
| 56 | `<channel>_intensity_mean/median/std/min/max/sum/cv`, `<channel>_boundary_intensity_mean`, `<channel>_interior_intensity_mean`, `<channel>_boundary_interior_ratio`, `<channel>_z_intensity_variation` | Per-channel intensity | One set per named image channel | Cheap-Moderate* |
| 57 | `merge_signal`, `fragment_signal`, `boundary_confidence`, `shape_abnormality` | Named composites | Directional robust z-scores | Trivial |
| 58 | `centroid-0/1/2`, `major_axis_{z,y,x}`, `mid_axis_{z,y,x}`, `minor_axis_{z,y,x}`, `inertia_eigval_{major,mid,minor}` | Position / orientation vectors | regionprops, inertia tensor | Trivial |

\* Requires `intensity_image` (source fluorescence stack). When not provided, these features are omitted.

† Requires probability volume (`{patch_stem}_probabilities.npy` from u-Segment3D reconstruction). When not available, these features are omitted.

---

## 2. Feature Descriptions and Equations

### 2.1 Core Morphology (always present)

**`area`** — Voxel count (= volume in 3D). With `spacing = (s_z, s_y, s_x)`, reports physical volume:

$$V = N_\text{voxels} \cdot s_z \cdot s_y \cdot s_x$$

The most basic feature; merged cells are ~2× expected size.

**`area_filled`** — Volume with internal holes filled. Differs from `area` when the label has internal cavities (rare in well-segmented cells).

**`equivalent_diameter`** — Diameter of a sphere with the same volume:

$$d_\text{eq} = \left(\frac{6V}{\pi}\right)^{1/3}$$

Normalizes volume into a linear scale for easier comparison.

**`extent`** — Fraction of the bounding box occupied by the label:

$$\text{extent} = \frac{V}{V_\text{bbox}}$$

Low values indicate irregular or L-shaped objects.

**`euler_number`** — Topological invariant:

$$\chi = \text{objects} + \text{cavities} - \text{tunnels}$$

$\chi = 1$ for a simple solid (expected for cells). $\chi < 1$ indicates tunnels (strongly suggestive of merge artifacts where two cell lumens connect). One of the most informative single features.

**`fill_ratio`** — Ratio of actual volume to hole-filled volume:

$$\text{fill\_ratio} = \frac{V}{V_\text{filled}}$$

Equal to 1.0 for solid objects; < 1 indicates internal holes.

### 2.2 Bounding Box Shape

**`sphericity`** — Ratio of a volume-equivalent sphere's surface area to the bounding box surface area:

$$\text{sphericity} = \frac{S_\text{sphere}}{S_\text{bbox}}$$

where:

$$S_\text{sphere} = 4\pi r_\text{eq}^2 = 4\pi \left(\frac{d_\text{eq}}{2}\right)^2$$

$$S_\text{bbox} = 2(b_z \cdot b_y + b_y \cdot b_x + b_z \cdot b_x)$$

and $b_z, b_y, b_x$ are the bounding box dimensions. This is a proxy for how sphere-like the object is relative to its bounding box.

**`aspect_ratio`** — Ratio of the longest to shortest bounding box dimension:

$$\text{aspect\_ratio} = \frac{\max(b_z, b_y, b_x)}{\min(b_z, b_y, b_x)}$$

High values indicate elongated objects. Merged cells often have elevated aspect ratios.

**`bbox_volume`** — Volume of the axis-aligned bounding box:

$$V_\text{bbox} = b_z \cdot b_y \cdot b_x$$

**`compactness`** — Fraction of the bounding box filled by the label (same as extent):

$$\text{compactness} = \frac{V}{V_\text{bbox}}$$

### 2.3 Orientation and Elongation

**`z_span`** — Extent of the label along the Z-axis (imaging axis):

$$z_\text{span} = b_z$$

With physical spacing, $z_\text{span} = n_\text{slices} \cdot s_z$.

**`z_elongation`** — Elongation along the imaging axis relative to the lateral extent:

$$\text{z\_elongation} = \frac{b_z}{\max(b_y, b_x)}$$

High values indicate objects elongated along the imaging axis. In Cellpose 2D-to-3D stitching, apical-basal stacked merges show high z_elongation.

**`xy_elongation`** — Lateral elongation (within the imaging plane):

$$\text{xy\_elongation} = \frac{\max(b_y, b_x)}{\min(b_y, b_x)}$$

High values indicate laterally elongated objects (lateral merge signal).

**`major_axis_z_angle`** — Angle between the object's major inertia axis and the Z-axis, in degrees. The 3×3 inertia tensor $\mathbf{J}$ is computed from the label voxels and decomposed via eigendecomposition:

$$\mathbf{J} \mathbf{v}_i = \lambda_i \mathbf{v}_i$$

where $\lambda_0 \leq \lambda_1 \leq \lambda_2$ (sorted ascending by `numpy.linalg.eigh`). The eigenvector $\mathbf{v}_0$ corresponding to the *smallest* eigenvalue points along the axis of greatest elongation (the "major axis"). The Z-angle is:

$$\theta = \arccos\left(|\mathbf{v}_0 \cdot \hat{\mathbf{z}}|\right) = \arccos\left(|v_{0,z}|\right)$$

$0°$ = aligned with Z, $90°$ = in the XY plane. Z-aligned cells with high z_elongation strongly suggest apical-basal stacked merges.

**`inertia_elongation`** — Direction-independent elongation from the inertia tensor eigenvalues. In our convention (regionprops returns eigenvalues sorted descending: $I_0 \geq I_1 \geq I_2$):

$$\text{inertia\_elongation} = \sqrt{\frac{I_0}{I_2}}$$

The largest eigenvalue corresponds to the axis of *least* elongation (most mass spread), while the smallest corresponds to the axis of *most* elongation. This ratio catches merges in any axis. More informative than bbox-based aspect_ratio for irregular shapes.

**`inertia_flatness`** — Distinguishes disc-like vs rod-like shapes:

$$\text{inertia\_flatness} = \sqrt{\frac{I_1}{I_0}}$$

Range [0, 1]. High flatness ($\approx 1$) = disc-like. Low flatness = rod-like = likely merge.

### 2.4 Ellipsoid Solidity

**`ellipsoid_solidity`** — Ratio of the label's actual volume to the volume of its best-fit inertia ellipsoid:

$$\text{ellipsoid\_solidity} = \frac{V}{V_\text{ellipsoid}}$$

The ellipsoid semi-axes $(a, b, c)$ are derived from the principal moments of inertia $(I_0, I_1, I_2)$ of a solid body with mass $M$ (= volume in voxels):

$$a^2 = \frac{5}{2M}(-I_0 + I_1 + I_2)$$

$$b^2 = \frac{5}{2M}(I_0 - I_1 + I_2)$$

$$c^2 = \frac{5}{2M}(I_0 + I_1 - I_2)$$

$$V_\text{ellipsoid} = \frac{4}{3}\pi \sqrt{a^2 \cdot b^2 \cdot c^2}$$

Values $> 1$ indicate the label overflows its inertia ellipsoid (common for round cells). Values $\ll 1$ indicate the label is much smaller than its ellipsoid — the hallmark of dumbbell-shaped merges where the inertia ellipsoid is inflated by the two lobes.

**`axis_major_length`** / **`axis_minor_length`** — Lengths of the major and minor axes of the best-fit ellipsoid, from regionprops inertia tensor computation.

### 2.4a Discrete Compactness — Bribiesca (from `compute_discrete_compactness`)

**`discrete_compactness`** — Voxel-based shape compactness measure from Bribiesca (Pattern Recognition 41(3), 2008). Counts the number of internal voxel-voxel face contacts and normalizes against the per-label minimum and maximum so the value lies in $[0, 1]$:

$$C_d = \frac{n - A_v / 6}{n - n^{2/3}}$$

where $n$ is the voxel count and $A_v$ is the number of external (boundary) voxel faces. Reference values:

| Shape | $C_d$ |
|---|---|
| Single isolated voxel (max surface per voxel) | $0$ |
| Voxelized sphere | $\approx 0.85$ |
| Perfect cube | $1$ |

Bounded $[0, 1]$, so the EDA workbench's "auto" transform never log-transforms it.

**Why it complements `true_sphericity`**: `true_sphericity` is derived from the marching-cubes surface area (continuous-surface measure). `discrete_compactness` operates directly on the voxel face count (discrete-surface measure). For elongated and irregular cells the two can diverge — e.g. a thin rod has low discrete compactness (~0.5) but moderate true_sphericity if its surface mesh smooths out. Having both gives the PCA two roughly orthogonal "irregularity" axes.

### 2.5 Surface Features (from `compute_surface_features`)

**`surface_area`** — True surface area computed via the marching cubes algorithm on per-label bounding box crops (padded with 1 voxel of background to ensure a closed isosurface). Respects voxel spacing:

$$S = \sum_{i}^{N_\text{faces}} \text{area}(\text{triangle}_i)$$

where the triangulated mesh is extracted at the 0.5 isosurface by `skimage.measure.marching_cubes`.

**`true_sphericity`** — The Wadell sphericity, measuring how closely the object's surface area approaches that of a volume-equivalent sphere:

$$\Psi = \frac{\pi^{1/3}(6V)^{2/3}}{S}$$

Range $(0, 1]$, where $1$ = perfect sphere. This is equivalent to:

$$\Psi = \frac{S_\text{equiv. sphere}}{S_\text{actual}}$$

More sensitive than bbox-based sphericity to surface irregularities. Merged cells with convoluted surfaces score lower.

**`roughness`** — Ratio of actual surface area to the surface area of a volume-equivalent sphere:

$$\text{roughness} = \frac{S}{S_\text{equiv. sphere}} = \frac{S}{4\pi r_\text{eq}^2}$$

where:

$$r_\text{eq} = \left(\frac{3V}{4\pi}\right)^{1/3}$$

Always $\geq 1$. High roughness indicates irregular, convoluted surfaces typical of merge artifacts.

**`solidity`** — Ratio of volume to convex hull volume:

$$\text{solidity} = \frac{V}{V_\text{convex hull}}$$

Computed via a fast subsampled approach: extract surface voxels (mask $\setminus$ eroded mask), subsample to ~1500 points in physical coordinates, compute `ConvexHull`. Values $< 0.6$ strongly suggest merged cells (dumbbell concavity where the neck between two lobes creates a large convex hull).

### 2.6 Intensity Features (from regionprops, requires `intensity_image`)

**`intensity_mean`** — Mean fluorescence intensity within the label:

$$\bar{I} = \frac{1}{N}\sum_{v \in \text{label}} I(v)$$

For membrane-GFP: should be LOW for correctly segmented cells (GFP-negative cytoplasm). HIGH mean intensity suggests merged cells with internal membrane signal bleeding into the interior.

**`intensity_std`** — Standard deviation of intensity within the label:

$$\sigma_I = \sqrt{\frac{1}{N}\sum_{v \in \text{label}} (I(v) - \bar{I})^2}$$

Very high $\sigma_I$ in large labels = heterogeneous interior = merge signal.

**`intensity_min`** / **`intensity_max`** — Extremes of the intensity distribution within the label.

### 2.7 Boundary/Interior Membrane Features (from `compute_boundary_interior_features`)

These features are specifically designed for membrane-fluorescence segmentation QC. The label mask is decomposed into a boundary shell and an interior region via binary erosion (default $k = 2$ iterations):

$$\text{interior} = \text{erode}^k(\text{mask})$$

$$\text{boundary} = \text{mask} \setminus \text{interior}$$

**`boundary_intensity_mean`** — Mean intensity in the boundary shell:

$$\bar{I}_\text{bnd} = \frac{1}{|\text{boundary}|}\sum_{v \in \text{boundary}} I(v)$$

**`interior_intensity_mean`** — Mean intensity in the eroded interior:

$$\bar{I}_\text{int} = \frac{1}{|\text{interior}|}\sum_{v \in \text{interior}} I(v)$$

For tiny labels where erosion eliminates the entire interior, falls back to the full mask mean.

**`boundary_interior_ratio`** — The single most discriminative feature for membrane-GFP segmentation:

$$R_\text{b/i} = \frac{\bar{I}_\text{bnd}}{\bar{I}_\text{int}}$$

- **Correct cells:** $R_\text{b/i} \gg 1$ (bright membrane boundary, dark cytoplasm), typically 2–10×
- **Merged cells:** $R_\text{b/i} \to 1$ (internal membrane raises interior intensity)
- **Fragments/noise:** $R_\text{b/i} \approx 1$ or $< 1$ (no clear membrane boundary)

**`z_intensity_variation`** — Standard deviation of per-Z-slice mean intensities within the label:

$$\sigma_{z} = \text{std}\left(\left\{\frac{1}{|M_z|}\sum_{v \in M_z} I(v) \;\middle|\; z \in [z_0, z_1]\right\}\right)$$

where $M_z$ is the set of label voxels in slice $z$. High $\sigma_z$ indicates a bright membrane band at some Z-plane — the hallmark of apical-basal merges where two cells are stacked vertically and a membrane plane separates them.

### 2.8 Neighbor/Adjacency Features (from `compute_neighbor_features`)

Uses **graduated dilation** to detect neighbors across gaps of any width, making it robust to non-touching segmentations. Rather than a single fixed dilation, the mask is incrementally dilated from radius 1 to `max_radius` (default 5), discovering new neighbors at each step:

$$\text{ring}_r = \text{dilate}^r(\text{mask}) \setminus \text{dilate}^{r-1}(\text{mask})$$

At each radius $r$, the label identities in $\text{ring}_r$ are examined. Neighbors are accumulated across all radii, with the contact shell being the union of all rings:

$$\text{shell} = \bigcup_{r=1}^{R} \text{ring}_r$$

This approach adapts to any gap width — it will find touching neighbors at $r = 1$ and neighbors separated by a 4-voxel gap at $r = 4$.

**`neighbor_count`** — Number of distinct non-background labels found across all dilation rings:

$$N_\text{neighbors} = \left|\bigcup_{r=1}^{R} \{l \in \text{ring}_r : l \neq 0, l \neq \text{self}\}\right|$$

Cells in tissue typically have 4–8 neighbors. $N = 0$ = isolated fragment. Very high $N$ = possibly large merged region.

**`nearest_neighbor_distance`** — The dilation radius at which the first non-background neighbor is discovered:

$$d_\text{nn} = \min\{r : \exists\, l \in \text{ring}_r,\; l \neq 0,\; l \neq \text{self}\}$$

This is an approximate gap width in voxels. $d_\text{nn} = 1$ means touching labels. $d_\text{nn} = 3$–$5$ is typical for non-touching reconstructions. $d_\text{nn} = R + 1$ (max_radius + 1) means no neighbor was found within the search radius — the label is truly isolated.

**`max_contact_fraction`** — Fraction of the total contact shell shared with the single most-contacted neighbor:

$$f_\text{max} = \frac{\max_l \sum_{r=1}^{R} |\{v \in \text{ring}_r : v = l\}|}{|\text{shell}|}$$

If $f_\text{max} > 0.5$, the pair may be an over-segmentation split (two fragments of the same cell that should be one).

**`background_fraction`** — Fraction of the total contact shell facing background (label 0):

$$f_\text{bg} = \frac{\sum_{r=1}^{R}|\{v \in \text{ring}_r : v = 0\}|}{|\text{shell}|}$$

High values = surface/border cell or isolated artifact. Note: in non-touching segmentations, `background_fraction` will be elevated for all cells due to the inter-cell gaps, but the **relative** ranking still identifies the most isolated labels.

### 2.9 Skeleton Topology Features (from `compute_skeleton_features`)

Uses distance-transform-based medial axis extraction. The skeleton is extracted as:

$$\text{DT}(v) = \text{distance\_transform\_edt}(\text{mask}_\text{padded})$$

$$\text{skeleton} = \{v : \text{DT}(v) = \max_{u \in N_{26}(v)} \text{DT}(u)\} \cap \{\text{DT}(v) > 1\}$$

where $N_{26}(v)$ is the 26-connected neighborhood (3×3×3 cube). The `DT > 1` threshold eliminates single-voxel protrusions. The mask is padded with 1 voxel of background before the EDT so that the distance transform measures distance to the true label boundary rather than to the array edge.

**`skeleton_branches`** — Number of skeleton branches:

$$B = N_\text{branch\_points} + 1$$

A single round cell yields $B \approx 1$. A dumbbell merge yields $B \geq 2$.

**`skeleton_nodes`** — Number of branch points, defined as skeleton voxels with more than 2 skeleton neighbors in 26-connectivity:

$$N_\text{branch\_points} = |\{v \in \text{skeleton} : |\{u \in N_{26}(v) : u \in \text{skeleton}\}| > 2\}|$$

Junction points indicate complex topology.

**`skeleton_length`** — Total number of medial axis voxels:

$$L = |\text{skeleton}|$$

Longer skeletons indicate more elongated or complex shapes.

### 2.10 Composite Score (from `compute_composite_score`)

**`qc_score`** — Population-relative outlier score over a curated set of 15 decorrelated features (any available subset is used): `area`, `extent`, `equivalent_diameter`, `aspect_ratio`, `true_sphericity`, `solidity`, `major_axis_z_angle`, `inertia_elongation`, `surface_area`, `neighbor_count`, `nearest_neighbor_distance`, `skeleton_branches`, `prob_mean`, `prob_boundary_drop`, `prob_boundary_entropy`. Each feature is first robust-standardized: right-skewed unbounded features get `log1p`, then the median and 1.4826 × MAD give centre and scale, and the result is winsorized at ±10. Two methods:

**Z-score method (default):** Mean absolute robust z-score across the selected features:

$$\text{qc\_score}_i = \frac{1}{K}\sum_{k=1}^{K} \left|\tilde z_{ik}\right|, \qquad \tilde z_{ik} = \frac{x_{ik} - \text{median}_k}{1.4826\,\text{MAD}_k}$$

where $K$ is the number of available selected features. Simple, interpretable, always non-negative. Higher = more unusual relative to the population.

**PCA method:** L2 norm of the cell's coordinates in the top $n_\text{pcs}$ principal components (default 5) of the robust-standardized matrix, shifted so the minimum is zero:

$$\mathbf{U}, \mathbf{S}, \mathbf{V}^T = \text{SVD}(\tilde{\mathbf{Z}}), \qquad \text{qc\_score}_i = \left\lVert (\tilde{\mathbf{Z}}\mathbf{V})_{i,1:n_\text{pcs}} \right\rVert_2$$

This is the geometric distance from the average cell in PCA space. Optionally, cells touching the volume border can be left out of the fit (they still receive a score).

The composite score is population-relative — if all cells in a dataset are flat, "flat" is not flagged as an outlier. Only cells unusual relative to their neighbors score high.

### 2.11 Probability Consensus Features (from `compute_probability_features`)

These features leverage the **u-Segment3D multi-view reconstruction probability volume** — a variance-weighted combination of Cellpose 2D probability maps from three orthogonal views (XY, XZ, YZ). Saved as `{patch_stem}_probabilities.npy` (float16, Z×Y×X, range [0,1]). High probability = all three views agree on cell presence; low probability = views disagree.

The boundary shell and interior are defined identically to the intensity boundary/interior features:

$$\text{interior} = \text{erode}(\text{mask})$$

$$\text{boundary} = \text{mask} \setminus \text{interior}$$

For tiny labels where erosion eliminates the entire mask, features fall back to the full mask mean.

#### Group 1: Confidence Statistics

**`prob_mean`** — Mean reconstruction probability inside the label mask:

$$S_\text{in} = \frac{1}{|M|}\sum_{v \in M} P(v)$$

Overall model confidence. Low values indicate the reconstruction was uncertain about this label's existence — the label may occupy a region where few views detected a cell.

**`prob_p10`** — 10th percentile of probability within the mask:

$$P_{10} = \text{percentile}_{10}\{P(v) : v \in M\}$$

Robust low-confidence detector. Catches labels where most voxels are high-confidence but a subregion has very low probability (e.g., one "bad slice" in the z-stack). More sensitive to localized problems than `prob_mean`.

**`prob_std`** — Standard deviation of probability within the mask:

$$\sigma_P = \text{std}\{P(v) : v \in M\}$$

Confidence uniformity. High $\sigma_P$ = mixed confidence within the label, suggesting parts of the cell are well-reconstructed but other parts are unreliable. Low $\sigma_P$ = consistent confidence (whether high or low).

#### Group 2: Boundary vs Interior Spatial Pattern

**`prob_boundary_mean`** — Mean probability in the boundary shell:

$$S_\text{bnd} = \frac{1}{|B|}\sum_{v \in B} P(v)$$

Edge confidence. Low values mean the reconstruction was uncertain at the cell's periphery, suggesting the boundary placement may be inaccurate.

**`prob_interior_mean`** — Mean probability in the eroded interior core:

$$S_\text{int} = \frac{1}{|C|}\sum_{v \in C} P(v)$$

Core confidence. The interior should generally have higher probability than the boundary.

**`prob_boundary_drop`** — Difference between interior and boundary probability:

$$\Delta = S_\text{int} - S_\text{bnd}$$

Large positive $\Delta$ = confident core but fuzzy edges (common for real cells at tissue boundaries). $\Delta \approx 0$ = uniform probability throughout. Negative $\Delta$ = edges more confident than core, which is unusual and may indicate a reconstruction artifact.

#### Group 3: Probability Mass vs Mask Size

**`prob_weighted_volume`** — Sum of probability values over a padded bounding box ROI:

$$V_\text{soft} = \sum_{v \in \Omega} P(v)$$

where $\Omega$ is the label's bounding box expanded by `bbox_padding` (default 3) voxels in each direction. This is the "soft volume" — how much total probability mass exists in and around the label, regardless of the hard mask boundary.

**`prob_soft_hard_ratio`** — Ratio of soft volume to hard (mask) volume:

$$R = \frac{V_\text{soft}}{|M|}$$

This ratio detects mismatches between the probability field and the segmentation mask:

- $R \ll 1$: **Over-segmentation signal** — the mask extends into a low-probability zone where the reconstruction was not confident. The label claims territory that the probability map does not support.
- $R \gg 1$: **Under-segmentation signal** — there is significant probability mass around the label that is not captured by the mask. The reconstruction detected more cell presence than the label covers.
- $R \approx 1$: Good agreement between the probability field and the mask boundary.

#### Group 4: Uncertainty / Entropy

Uses voxel-wise binary (Bernoulli) entropy, which quantifies how uncertain the reconstruction is at each voxel:

$$H(v) = -P(v)\ln P(v) - (1 - P(v))\ln(1 - P(v))$$

$H(v) = 0$ when $P = 0$ or $P = 1$ (completely certain). $H(v)$ peaks at $\ln(2) \approx 0.693$ when $P = 0.5$ (maximum uncertainty — the views disagree equally).

**`prob_boundary_entropy`** — Mean boundary uncertainty:

$$U_\text{bnd} = \frac{1}{|B|}\sum_{v \in B} H(v)$$

High values mean the reconstruction was uncertain about where to place the cell boundary. This is the most direct measure of boundary reliability.

**`prob_mask_entropy`** — Mean uncertainty across the entire mask:

$$U_\text{mask} = \frac{1}{|M|}\sum_{v \in M} H(v)$$

Overall voxel-wise uncertainty. High values throughout the mask suggest the label is poorly supported by multi-view consensus.

#### Group 5: Edge Sharpness

Uses the gradient magnitude of the probability field:

$$|\nabla P(v)| = \sqrt{\left(\frac{\partial P}{\partial z}\right)^2 + \left(\frac{\partial P}{\partial y}\right)^2 + \left(\frac{\partial P}{\partial x}\right)^2}$$

**`prob_edge_gradient`** — Mean gradient magnitude at boundary voxels:

$$G_\text{bnd} = \frac{1}{|B|}\sum_{v \in B} |\nabla P(v)|$$

A sharp probability drop at the boundary ($G_\text{bnd}$ high) indicates the reconstruction had a confident, well-defined cell edge. A gradual transition ($G_\text{bnd}$ low) indicates an uncertain, diffuse boundary.

**`prob_edge_gradient_std`** — Standard deviation of gradient magnitude at boundary voxels:

$$\sigma_G = \text{std}\{|\nabla P(v)| : v \in B\}$$

Measures uniformity of edge sharpness around the cell. Low $\sigma_G$ = consistently sharp (or consistently diffuse) boundary. High $\sigma_G$ = patchy edge confidence — some boundary regions are sharp while others are uncertain.

### 2.12 Cross-Section Profile Features (from `compute_cross_section_features`)

Let $A(z)$ be the number of mask voxels in Z-slice $z$, for $z$ from $z_\min$ to $z_\max$ of the label. Z is normalized to $[0, 1]$ across the cell's own Z-range (the sign is flipped for volumes imaged apex-first).

**`basal_area`** / **`apical_area`** — Mean $A(z)$ over the bottom 20% / top 20% of the cell's Z-range (voxels²). A wide base suggests a pyramidal or basal cell; a wide top suggests a squamous or apical cell.

**`apical_basal_area_ratio`** — $\text{apical\_area} / \text{basal\_area}$. Below 0.5 is pyramidal, about 1 columnar, above 2 inverted.

**`max_cross_section_z_relative`** — $(\arg\max_z A(z) - z_\min)/(z_\max - z_\min)$: position of the widest slice (0 = base, 1 = apex).

**`cross_section_taper`** — Ordinary least squares slope of $A(z)$ against normalized $z$ (voxels² per unit fraction); signed tapering rate along Z.

**`cross_section_bulge`** — Root-mean-square residual of the linear fit of $A(z)$ versus $z$, divided by the mean area. Dimensionless; high for barrel, vase or dumbbell profiles.

### 2.13 Per-Channel Intensity Features (from `compute_channel_intensity`)

For each user-named image channel, the following are computed over the label's mask voxels and prefixed with the channel name (`<channel>_`): `intensity_mean`, `intensity_median`, `intensity_std`, `intensity_min`, `intensity_max`, `intensity_sum` (integrated signal, scales with volume), `intensity_cv` ($= \text{std}/\text{mean}$, 0 if the mean is 0), plus the boundary/interior set of section 2.7 (`boundary_intensity_mean`, `interior_intensity_mean`, `boundary_interior_ratio`, `z_intensity_variation`). A second channel gives an independent signal check; for example the mean of a nucleus channel plotted against `area` separates real cells (nucleus present) from spurious, merged or fragment masks (nucleus signal absent or weak).

### 2.14 Adjacency Extensions (from `compute_neighbor_features`)

These build on the graduated-dilation scheme of section 2.8. For each neighbor $L'$ of label $L$, $r^*(L, L')$ is the smallest dilation radius at which $L'$ is first reached (1 to `max_radius`).

**Filtration features:** `nn_count_r1` = number of neighbors with $r^* = 1$ (directly touching); `nn_count_r3` = number with $r^* \le 3$; `nn_birth_median`, `nn_birth_std` = median and standard deviation of $r^*$ over neighbors (voxels); `nn_accumulation_rate` = OLS slope of the cumulative neighbor count against $r$ for $r = 1..5$ (neighbors per voxel); `nn_accumulation_knee` = smallest $r$ at which the cumulative count reaches 80% of the total.

**Size-context features:** with $V$ the label volume, `mean_neighbor_size_ratio` $= V_L / \text{mean}(V_{L'})$; `neighbor_size_cv` $= \text{std}(V_{L'})/\text{mean}(V_{L'})$; `contact_weighted_neighbor_size_ratio` $= V_L / \left(\sum c_{LL'}V_{L'} / \sum c_{LL'}\right)$ with $c_{LL'}$ the contact voxel count.

**Z-stratified features:** `basal_neighbor_count` / `apical_neighbor_count` = neighbors contacted in the basal / apical third of the cell's Z-range; `neighbor_z_turnover` $= 1 - |B \cap A| / |B \cup A|$ (1 = complete change of neighbor identity from base to apex); `neighbor_persistence_ratio` $= |B \cap A| / N_\text{neighbors}$, where $B$ and $A$ are the basal and apical neighbor sets.

### 2.15 Named Composite Scores (from `compute_named_composites`)

Directional composites of robust z-scores $\tilde z_k$ (median/MAD, clipped at ±10). For a signed weight $w_k = \pm 1$, a contributor adds $\max(\tilde z_k \cdot w_k, 0)$, and the score is the mean over the contributors that are available.

| Score | Contributors (weight) | High value flags |
|---|---|---|
| `merge_signal` | `boundary_interior_ratio` (-1), `inertia_elongation` (+1), `skeleton_branches` (+1), `euler_number` (-1), `cross_section_bulge` (+1), `mean_neighbor_size_ratio` (+1) | Merge-like cells: elongated, branched, low membrane contrast |
| `fragment_signal` | `area` (-1), `neighbor_count` (-1), `background_fraction` (+1), `mean_neighbor_size_ratio` (-1) | Fragment-like cells: small, isolated, exposed to background |
| `shape_abnormality` | `apical_basal_area_ratio` (+1), `cross_section_taper` (+1), `inertia_elongation` (+1), `true_sphericity` (-1) | Unusual shape relative to the population |

`boundary_confidence` $= \big(\tilde z(\text{prob\_edge\_gradient}) - \tilde z(\text{prob\_boundary\_entropy}) + \tilde z(\text{prob\_boundary\_mean})\big)/3$ (requires the probability volume). High = sharp edges, low uncertainty and high boundary probability, i.e. a confident boundary.

### 2.16 Position and Orientation Vectors

These columns are descriptive and are not part of the composite score.

- `centroid-0`, `centroid-1`, `centroid-2`: center of mass in voxel coordinates (Z, Y, X), not scaled by spacing.
- `major_axis_{z,y,x}`, `mid_axis_{z,y,x}`, `minor_axis_{z,y,x}`: unit direction vectors (ZYX components) of the three principal axes of the inertia tensor. The "major" axis is the eigenvector of the smallest inertia eigenvalue, that is the most elongated direction; `major_axis_z_angle` (section 2.3) is derived from it.
- `inertia_eigval_major`, `inertia_eigval_mid`, `inertia_eigval_minor`: the three inertia-tensor eigenvalues as returned by regionprops (descending order), kept for downstream ratios.

---

## 3. Practical Considerations: Non-Touching Segmentations

### 3.1 The Gap Problem

In the EpiCycle pipeline, the QC features operate on **3D reconstructed volumes** — the product of fusing dense 2D Cellpose predictions across Z-slices and stitching them into coherent 3D labels. In these reconstructions, adjacent label masks **do not extend to the cell boundaries**. There is a gap of 1–5 voxels of background (label 0) between neighboring cells, even in tightly packed epithelial tissue. These gaps arise from multiple stages:

1. **2D Cellpose predictions** use flow fields that converge toward cell centers, naturally leaving a margin at the membrane boundaries where the GFP signal is strongest.
2. **Z-slice fusion** via IoU-based label matching can shrink labels at Z-transitions where slice-to-slice alignment is imperfect.
3. **3D reconstruction** inherits and compounds these per-slice margins into a 3D gap network.
4. The membrane itself (1–2 voxels wide in the fluorescence image) sits in the gap, not inside any label.

This is clearly visible in the reconstruction overlays: each colored label mask is surrounded by a ring of black (background), with the green membrane-GFP signal sitting in those gaps.

### 3.2 How Each Feature Category Handles Gaps

**Morphology features (`area`, `extent`, `compactness`, etc.)** — These measure the label mask directly. Since masks are slightly smaller than the true cell boundaries, `area` consistently **underestimates** true cell volume by ~5–15% (depending on cell size and gap width). This is a systematic bias that affects all labels equally, so **relative comparisons remain valid** — a merged cell is still ~2× the volume of a normal cell. The underestimate is proportionally larger for small cells (where the gap is a larger fraction of the total volume).

**Bounding box features (`aspect_ratio`, `z_elongation`, etc.)** — Bounding boxes are determined by the label extent and are unaffected by gaps. These features work normally.

**Surface features (`surface_area`, `true_sphericity`, `roughness`)** — Marching cubes extracts the isosurface of the label mask. Since the reconstructed mask boundaries are relatively smooth (inheriting Cellpose's rounded per-slice predictions), surface features reflect the **mask shape** rather than the true cell membrane. This means:
- `roughness` may be **lower** than the actual cell membrane roughness
- `true_sphericity` may be **higher** (mask is smoother than reality)
- These features are still useful for detecting merges (which distort the mask shape), but less sensitive to fine-grained membrane topology

**Solidity** — The convex hull comparison works well with non-touching labels. A correctly reconstructed cell will have high solidity ($\approx 0.8$–$1.0$). A dumbbell merge will still show low solidity ($< 0.6$) because the concave neck is a property of the mask shape, not the gaps.

**Intensity features (`intensity_mean`, `intensity_std`)** — Computed over voxels **inside the label mask only**. Since the bright membrane sits in the gap (outside the mask), a correctly segmented cell's label mask covers mostly dark cytoplasm. This is exactly why `intensity_mean` works: correct cells have low interior intensity, and merges have elevated intensity from the internal membrane that falls *inside* the merged mask.

**Boundary/interior features (`boundary_interior_ratio`)** — This is where the gap geometry is most relevant. The "boundary shell" is defined by binary erosion of the label mask — it captures the **outer ring of the mask**, not the actual cell membrane. In non-touching segmentations:

- The boundary shell of the mask is 1–2 voxels *inward* from the actual membrane
- However, because membrane-GFP signal diffuses slightly and the PSF blurs signal across 2–3 voxels, the **boundary shell still captures significant membrane signal**
- The boundary/interior ratio remains discriminative: $R_\text{b/i} \approx 2$–$10$ for correct cells (GFP signal bleeds into the mask edge) vs $R_\text{b/i} \to 1$ for merges (internal membrane raises interior intensity)
- The absolute values of $\bar{I}_\text{bnd}$ will be lower than if the mask perfectly covered the membrane, but the **ratio** is robust

**Neighbor features (`neighbor_count`, `nearest_neighbor_distance`, `max_contact_fraction`)** — These use **graduated dilation** (incremental dilation from radius 1 to `max_radius`, default 5) specifically designed to handle non-touching labels. The algorithm dilates outward one voxel at a time, recording when each neighbor is first encountered:

- A gap of 3 voxels between two cells is bridged at dilation step 3 — the neighbor is detected and `nearest_neighbor_distance = 3`
- The `max_radius` parameter (configurable, default 5) controls how far the search extends
- `nearest_neighbor_distance` directly encodes the gap width, providing an interpretable isolation measure
- `background_fraction` will still be elevated relative to touching segmentations since the gap voxels are background, but the graduated approach ensures neighbors are not missed
- `max_contact_fraction` accumulates contact across all dilation rings, so it captures the full extent of the shared boundary

**Skeleton features (`skeleton_branches`, `skeleton_length`)** — Unaffected by gaps. The EDT-based medial axis operates on each label independently, and the gap between labels does not influence the distance transform within a label (the padding ensures the EDT measures distance to the label's own boundary).

**Probability consensus features (`prob_mean`, `prob_boundary_drop`, etc.)** — These operate on the reconstruction probability volume, which is independent of the gap structure. The probability field reflects the multi-view agreement at each voxel, not the mask boundaries. The boundary/interior decomposition uses the same binary erosion as the intensity features, so it shares the same 1–2 voxel inward offset from the actual membrane. The key insight is that probability features measure a fundamentally different quantity than intensity features: they capture **how confident the reconstruction algorithm was** about cell presence, not the fluorescence signal. `prob_soft_hard_ratio` is particularly informative: it compares the probability mass in a padded ROI against the hard mask volume, detecting cases where the mask boundary doesn't agree with the probability field (over-segmentation vs under-segmentation).

**Composite score (`qc_score`)** — Since the z-score method normalizes each feature by the population median and MAD, systematic biases from non-touching labels (like elevated `background_fraction` for all cells) are automatically accounted for. A cell is only flagged as an outlier if it is unusual *relative to the population*, so if all cells have high `background_fraction`, that feature contributes little to the score.

### 3.3 Recommendations

1. **Graduated dilation bridges gaps** — the default `max_radius=5` covers typical reconstruction gaps of 2–5 voxels. Increase it for very sparse segmentations or decrease it if you only care about direct-contact neighbors.
2. **`nearest_neighbor_distance`** is a direct readout of gap width. Use it to characterize the segmentation density: tight clusters have $d_\text{nn} = 1$–$2$, loose reconstructions have $d_\text{nn} = 4$–$5$, and isolated fragments have $d_\text{nn} = R + 1$.
3. **Boundary/interior ratio is robust** despite the gap, thanks to PSF blurring and GFP signal diffusion. It remains the single best discriminator for merge detection in membrane-GFP data.
4. **Volume underestimation** is systematic and proportional — it does not affect the ability to detect merges (which are ~2× volume outliers).
5. **Surface features** reflect the smooth reconstructed mask rather than the true membrane. They are effective for merge/fragment detection but not for measuring true membrane roughness.
6. **The composite `qc_score`** handles all of these biases gracefully through population-relative normalization.

---

## 4. Failure Mode Detection Summary

| Failure Mode | Key Discriminative Features |
|---|---|
| **Merged cells (under-segmentation)** | $R_\text{b/i} \downarrow$, `area` $\uparrow$ (~2×), `ellipsoid_solidity` $\downarrow$, `skeleton_branches` $\uparrow$, $\chi < 1$, `inertia_elongation` $\uparrow$, `prob_soft_hard_ratio` $\uparrow$ (probability mass exceeds mask) |
| **Fragments (over-segmentation)** | `area` $\downarrow$, $f_\text{max} \uparrow$ (>0.5), $N_\text{neighbors}$ = 0 or 1, $d_\text{nn} \uparrow$, $f_\text{bg} \uparrow$, `prob_soft_hard_ratio` $\downarrow$ (mask in low-prob zone) |
| **Apical-basal stacked merge** | `z_elongation` $\uparrow$, $\theta \to 0°$, $\sigma_z \uparrow$ |
| **Lateral merge** | `xy_elongation` $\uparrow$, $\theta \to 90°$ |
| **Topological defects** | $\chi \neq 1$, `fill_ratio` $< 1$ |
| **Surface artifacts** | `roughness` $\uparrow$, $\Psi \downarrow$ |
| **Uncertain reconstruction** | `prob_mean` $\downarrow$, `prob_p10` $\downarrow$, `prob_mask_entropy` $\uparrow$ |
| **Fuzzy/uncertain boundaries** | `prob_boundary_drop` $\uparrow$ (confident core, fuzzy edges), `prob_boundary_entropy` $\uparrow$, `prob_edge_gradient` $\downarrow$ (gradual transition), `prob_edge_gradient_std` $\uparrow$ (patchy sharpness) |
| **Multi-view disagreement** | `prob_std` $\uparrow$ (mixed confidence), `prob_p10` $\downarrow$ (localized low-confidence subregion) |

---

## 5. References

1. CellProfiler MeasureObjectSizeShape — ~60 shape features per object
2. CellSegmentationEvaluator (Murphy Lab) — 14 reference-free metrics, PCA composite
3. MorphoNet 2.0 (eLife 2025) — per-object outlier detection for 3D curation
4. u-Segment3D (Nature Methods 2025) — skeleton branching + elongation features
5. Ge et al. (arXiv 2025) — Geo-Wasserstein divergence for oversegmentation correction

See `docs/research/QC_FEATURES_BIOLOGIST_GUIDE.md` for a plain-language guide and `docs/research/FEATURE_CATALOG.csv` for the machine-readable catalogue.
