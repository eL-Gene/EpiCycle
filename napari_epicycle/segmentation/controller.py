from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
from qtpy.QtCore import QObject, QThread, Qt, Signal
from qtpy.QtWidgets import QFileDialog, QMessageBox

from packages.epicycle_domain import StackSegmentationJobSpec
from packages.epicycle_io import FileArtifactRegistry
from packages.epicycle_reconstruction.logic import ReconstructionPipelineService
from packages.epicycle_training.config import SegmentationConfig

from .io import StackSegmentationRunStore
from .logic import StackSegmentationService
from . import _step4_models as S4M
from . import _step4_rows as S4R
from . import _step4_text as S4T
from ..ui_shell import theme as TH
from ._step4_lock import ALL_FLAGS as _STEP4_ALL_FLAGS
from ._step4_lock import is_locked as _step4_is_locked
from ._step4_lock import ALL_FLAGS as STEP4_BLOCK_FLAGS_ALL

#: F15: every Step 4 layer name starts with this, and every Step 4
#: layer is registered under the "step4" context by its ACTUAL name.
STEP4_LAYER_PREFIX = "step4 \u00b7 "

#: Job threads still running at napari's close, parked so Python never
#: deletes a QThread that runs (shutdown()).
_LIVE_STEP4_THREADS: set = set()


def _disconnect_all_signals(obj) -> None:
    """Disconnect every slot from every signal *obj*'s class declares
    (shutdown(), K2).  Dropping the connection drops a call already queued
    through it."""
    from qtpy.QtCore import QMetaMethod
    try:
        meta = obj.metaObject()
    except RuntimeError:
        return
    for i in range(meta.methodOffset(), meta.methodCount()):
        m = meta.method(i)
        if m.methodType() != QMetaMethod.MethodType.Signal:
            continue
        name = bytes(m.name()).decode()
        sig = getattr(obj, name, None)
        if sig is None:
            continue
        try:
            sig.disconnect()
        except (TypeError, RuntimeError):
            pass
from .state_manager import StackStateManager


def T_WB():
    """The Workbench's sentence table."""
    from . import _workbench_text
    return _workbench_text


def _category_word(column, value) -> str:
    """The word CAT_PICKED names a legend category by."""
    if str(column) == "cluster_id":
        try:
            v = int(float(value))
        except (TypeError, ValueError):
            return str(value)
        return "noise" if v == -1 else f"G{v + 1}"
    return str(value)

_FEATURE_REFERENCE_HTML = """\
<html>
<head><style>
body { font-family: 'Segoe UI', sans-serif; color: #e0e0e0; background: #1a1a2e;
       margin: 16px; line-height: 1.5; }
h1 { color: #6abf69; font-size: 20px; border-bottom: 2px solid #6abf69;
     padding-bottom: 6px; margin-top: 0; }
h2 { color: #8bc34a; font-size: 15px; margin-top: 20px; margin-bottom: 4px;
     border-bottom: 1px solid rgba(139,195,74,0.3); padding-bottom: 3px; }
h3 { color: #a78bca; font-size: 13px; margin-top: 14px; margin-bottom: 2px; }
table { border-collapse: collapse; width: 100%; margin: 6px 0 10px 0; }
th { text-align: left; padding: 5px 8px; background: #2d2d4e; color: #6abf69;
     font-size: 12px; border-bottom: 1px solid #444; }
td { padding: 4px 8px; font-size: 12px; border-bottom: 1px solid #2a2a3e; }
tr:nth-child(even) td { background: rgba(255,255,255,0.02); }
code { font-family: 'Consolas', monospace; color: #80deea; background: #2d2d4e;
       padding: 1px 4px; border-radius: 2px; font-size: 11px; }
.note { color: #ff9800; font-size: 11px; font-style: italic; margin: 4px 0; }
.eq { color: #80deea; font-family: 'Consolas', monospace; font-size: 12px;
      background: #1e1e3e; padding: 4px 8px; border-radius: 3px; display: block;
      margin: 4px 0; }
</style></head>
<body>

<h1>QC Feature Reference</h1>
<p style="color:#999; font-size:12px;">
~50 features extracted per label. All measurements respect voxel spacing when configured.
Features marked with &#x26A0; are skipped during post-smooth re-extraction.
</p>

<h2>Core Morphology (18 features)</h2>
<p style="color:#aaa; font-size:12px;">
Basic shape measurements from <code>skimage.measure.regionprops</code>.
Always computed for every label.
</p>
<table>
<tr><th>Feature</th><th>What it measures</th><th>Why it matters</th></tr>
<tr><td><code>area</code></td><td>Total volume of the cell mask (voxels or &micro;m&sup3;)</td><td>Cell size &mdash; abnormally large may be merged, tiny may be fragments</td></tr>
<tr><td><code>area_filled</code></td><td>Volume with all internal holes filled</td><td>Compare with <code>area</code> to detect hollow cells</td></tr>
<tr><td><code>equivalent_diameter</code></td><td>Diameter of a sphere with the same volume</td><td>Size-normalized comparison across shapes</td></tr>
<tr><td><code>extent</code></td><td>Fraction of bounding box filled by the cell</td><td>Low extent = sparse/elongated, high = compact/cubic</td></tr>
<tr><td><code>euler_number</code></td><td>Topological invariant: 1 = solid, &lt;1 = has holes or tunnels</td><td>Detects donut-shaped artifacts or merged cells with gaps</td></tr>
<tr><td><code>sphericity</code></td><td>How sphere-like the bounding box proportions are</td>
    <td>Cells far from spherical may be elongated or merged</td></tr>
<tr><td><code>aspect_ratio</code></td><td>Longest / shortest bounding box edge</td><td>High = very elongated</td></tr>
<tr><td><code>bbox_volume</code></td><td>Volume of the axis-aligned bounding box</td><td>Context for compactness</td></tr>
<tr><td><code>compactness</code></td>
    <td><span class="eq">area / bbox_volume</span></td>
    <td>1.0 = fills entire bounding box (cube). Lower = more irregular</td></tr>
<tr><td><code>z_span</code></td><td>Extent of the cell along Z</td><td>Anisotropy indicator &mdash; very short Z = flat, very tall = columnar</td></tr>
<tr><td><code>z_elongation</code></td>
    <td><span class="eq">z_span / max(Y_span, X_span)</span></td>
    <td>&gt;1 = vertically elongated, &lt;1 = flat/lateral</td></tr>
<tr><td><code>xy_elongation</code></td>
    <td><span class="eq">max(Y, X) / min(Y, X)</span></td>
    <td>&gt;1 = elliptical cross-section, 1.0 = circular</td></tr>
<tr><td><code>fill_ratio</code></td>
    <td><span class="eq">area / area_filled</span></td>
    <td>1.0 = no internal holes. &lt;1 = has cavities</td></tr>
<tr><td><code>ellipsoid_solidity</code></td><td>Volume / volume of best-fit ellipsoid (from inertia tensor)</td>
    <td>&gt;1 = label extends beyond ellipsoid, &lt;1 = label is smaller</td></tr>
<tr><td><code>major_axis_z_angle</code></td><td>Angle between major inertia axis and Z (degrees)</td>
    <td>0&deg; = vertical cell, 90&deg; = horizontal</td></tr>
<tr><td><code>inertia_elongation</code></td>
    <td><span class="eq">I&#x2080; / I&#x2082;</span> (largest / smallest eigenvalue)</td>
    <td>Higher = more elongated along one axis</td></tr>
<tr><td><code>inertia_flatness</code></td>
    <td><span class="eq">I&#x2081; / I&#x2082;</span> (middle / smallest eigenvalue)</td>
    <td>Higher = more disc-like vs rod-like</td></tr>
</table>

<h2>Surface Features (4 features)</h2>
<p style="color:#aaa; font-size:12px;">
Computed via marching cubes triangle mesh (<code>skimage.measure.marching_cubes</code>).
</p>
<table>
<tr><th>Feature</th><th>What it measures</th><th>Equation / Detail</th></tr>
<tr><td><code>surface_area</code></td><td>Total triangle mesh surface area</td><td>Respects voxel spacing. Sensitive to surface roughness.</td></tr>
<tr><td><code>true_sphericity</code></td><td>How close to a perfect sphere</td>
    <td><span class="eq">&Psi; = 36&pi; &middot; V&sup2; / SA&sup3;</span>1.0 = perfect sphere. Lower = more irregular.</td></tr>
<tr><td><code>roughness</code></td><td>Surface irregularity</td>
    <td><span class="eq">SA / convex_hull_SA</span>&gt;1 = surface has bumps/indentations beyond convex hull.</td></tr>
<tr><td><code>solidity</code></td><td>How convex the shape is</td>
    <td><span class="eq">V / convex_hull_V</span>1.0 = perfectly convex. Lower = concavities/lobes.</td></tr>
</table>

<h2>Intensity Features (4 features)</h2>
<p style="color:#aaa; font-size:12px;">
From the segmentation channel image. Only computed when an intensity image is available.
</p>
<table>
<tr><th>Feature</th><th>Description</th></tr>
<tr><td><code>intensity_mean</code></td><td>Average pixel brightness inside the cell mask</td></tr>
<tr><td><code>intensity_std</code></td><td>Variation in brightness (high = heterogeneous staining)</td></tr>
<tr><td><code>intensity_min</code></td><td>Darkest voxel in the mask</td></tr>
<tr><td><code>intensity_max</code></td><td>Brightest voxel in the mask</td></tr>
</table>

<h2>Boundary / Interior Features (4 features)</h2>
<p style="color:#aaa; font-size:12px;">
The cell mask is eroded (2 iterations) to separate a boundary shell from the interior.
Useful for membrane-stained samples where correctly segmented cells have bright boundaries.
</p>
<table>
<tr><th>Feature</th><th>What it measures</th><th>Interpretation</th></tr>
<tr><td><code>boundary_intensity_mean</code></td><td>Mean brightness in the 2-voxel outer shell</td>
    <td>Should be high for membrane-stained cells</td></tr>
<tr><td><code>interior_intensity_mean</code></td><td>Mean brightness in the eroded core</td>
    <td>Should be low for membrane-stained cells</td></tr>
<tr><td><code>boundary_interior_ratio</code></td>
    <td><span class="eq">boundary_mean / interior_mean</span></td>
    <td>&gt;1 = bright membrane, dark interior (good). ~1 = merged cells share internal membrane signal.</td></tr>
<tr><td><code>z_intensity_variation</code></td><td>Std of per-Z-slice mean intensities</td>
    <td>High = inconsistent staining or Z-artifact</td></tr>
</table>

<h2>Probability Consensus Features (12 features) &#x26A0;</h2>
<p style="color:#aaa; font-size:12px;">
From the multi-view probability volume produced by u-Segment3D.
<span class="note">&#x26A0; Skipped during post-smooth re-extraction (probability volume no longer aligns with smoothed boundaries).</span>
</p>
<table>
<tr><th>Group</th><th>Features</th><th>What they measure</th></tr>
<tr><td>Confidence</td><td><code>prob_mean</code>, <code>prob_p10</code>, <code>prob_std</code></td>
    <td>Overall segmentation confidence. Low prob_p10 = weak evidence at cell edges.</td></tr>
<tr><td>Spatial Pattern</td><td><code>prob_boundary_mean</code>, <code>prob_interior_mean</code>, <code>prob_boundary_drop</code></td>
    <td>How probability drops at boundaries. Large drop = well-defined edges.</td></tr>
<tr><td>Mass</td><td><code>prob_weighted_volume</code>, <code>prob_soft_hard_ratio</code></td>
    <td>Soft volume vs hard mask size. Low ratio = mask extends beyond confident region.</td></tr>
<tr><td>Uncertainty</td><td><code>prob_boundary_entropy</code>, <code>prob_mask_entropy</code></td>
    <td>Shannon entropy of probability values. High = uncertain segmentation.</td></tr>
<tr><td>Edge Sharpness</td><td><code>prob_edge_gradient</code>, <code>prob_edge_gradient_std</code></td>
    <td>Gradient magnitude at boundaries. High = sharp edges (good).</td></tr>
</table>

<h2>Neighbor / Adjacency Features (4 features)</h2>
<p style="color:#aaa; font-size:12px;">
Computed via graduated dilation to detect touching labels.
</p>
<table>
<tr><th>Feature</th><th>Description</th></tr>
<tr><td><code>neighbor_count</code></td><td>Number of distinct labels touching this cell</td></tr>
<tr><td><code>nearest_neighbor_distance</code></td><td>EDT-based distance to closest neighbor (voxels)</td></tr>
<tr><td><code>max_contact_fraction</code></td><td>Largest shared surface area / total surface. High = one dominant neighbor.</td></tr>
<tr><td><code>background_fraction</code></td><td>Fraction of surface touching background (0). High = isolated cell.</td></tr>
</table>

<h2>Skeleton Topology (3 features)</h2>
<p style="color:#aaa; font-size:12px;">
Medial-axis skeleton via <code>skimage.morphology.skeletonize</code>.
</p>
<table>
<tr><th>Feature</th><th>Description</th></tr>
<tr><td><code>skeleton_branches</code></td><td>Number of branches. High = complex/lobulated shape.</td></tr>
<tr><td><code>skeleton_nodes</code></td><td>Number of branch points (junctions).</td></tr>
<tr><td><code>skeleton_length</code></td><td>Total skeleton path length (voxels).</td></tr>
</table>

<h2>Per-Channel Intensity (10 features &times; N channels)</h2>
<p style="color:#aaa; font-size:12px;">
Step 4 only. For each checked channel in the Intensity Channels UI, these features are
extracted using the user-assigned channel name as the column prefix.
</p>
<table>
<tr><th>Feature</th><th>What it measures</th></tr>
<tr><td><code>{name}_intensity_mean</code></td><td>Average brightness from this channel inside the cell</td></tr>
<tr><td><code>{name}_intensity_median</code></td><td>Median brightness (robust to outlier pixels)</td></tr>
<tr><td><code>{name}_intensity_std</code></td><td>Variation in brightness</td></tr>
<tr><td><code>{name}_intensity_min/max</code></td><td>Intensity range</td></tr>
<tr><td><code>{name}_intensity_sum</code></td><td>Total integrated fluorescence (proxy for protein amount)</td></tr>
<tr><td><code>{name}_intensity_cv</code></td>
    <td><span class="eq">std / mean</span>Normalized variability. Comparable across different brightness levels.</td></tr>
<tr><td><code>{name}_boundary_intensity_mean</code></td><td>Mean brightness at the 2-voxel boundary shell</td></tr>
<tr><td><code>{name}_interior_intensity_mean</code></td><td>Mean brightness in the eroded interior</td></tr>
<tr><td><code>{name}_boundary_interior_ratio</code></td>
    <td><span class="eq">boundary / interior</span>
    &gt;1 = signal concentrated at membrane (e.g. membrane protein).
    ~1 = cytoplasmic/uniform. &lt;1 = nuclear/interior enriched.</td></tr>
</table>

<h2>Composite Score (1 feature)</h2>
<p style="color:#aaa; font-size:12px;">
Aggregated outlier score across all numeric features.
</p>
<table>
<tr><th>Feature</th><th>Equation</th><th>Interpretation</th></tr>
<tr><td><code>qc_score</code></td>
    <td><span class="eq">qc_score = (1/N) &Sigma; |z&#x1D62;|</span>
    where <code>z&#x1D62;</code> = median-based robust z-score per feature,
    winsorized to &plusmn;10&sigma;.</td>
    <td>Higher = more unusual cell. Good cells cluster near 0.
    Use histogram threshold lines to set triage cutoffs.</td></tr>
</table>

<h1 style="margin-top:30px; border-bottom-color:#a78bca;">
Preprocessing &amp; Scoring Pipeline
</h1>
<p style="color:#999; font-size:12px;">
Applied before PCA, UMAP, clustering, and composite scoring.
This pipeline ensures that all features contribute proportionally
regardless of scale, distribution, or outliers.
</p>

<h2>Step 1: Log-Transform (Skewness Correction)</h2>
<p style="color:#aaa; font-size:12px;">
Right-skewed features are <code>log1p</code>-transformed to compress
heavy tails. Without this, a few extremely large cells dominate
PCA variance and hide biological variation.
</p>

<h3>Curated features (always transformed)</h3>
<p style="color:#aaa; font-size:12px;">
These 18 features are known to be right-skewed in cell segmentation data.
They are log-transformed regardless of measured skewness in the current dataset.
</p>
<table>
<tr><th>Category</th><th>Features</th><th>Why log-transform</th></tr>
<tr><td>Size</td>
    <td><code>area</code>, <code>area_filled</code>, <code>equivalent_diameter</code>,
        <code>bbox_volume</code>, <code>z_span</code>, <code>surface_area</code>,
        <code>axis_major_length</code>, <code>axis_minor_length</code></td>
    <td>Cell volumes follow log-normal distributions in tissue.
        Epithelial cells range from small basal to large superficial &mdash;
        multiplicative growth produces log-normal size distributions.</td></tr>
<tr><td>Intensity</td>
    <td><code>intensity_mean</code>, <code>intensity_std</code>,
        <code>intensity_min</code>, <code>intensity_max</code>,
        <code>boundary_intensity_mean</code>, <code>interior_intensity_mean</code></td>
    <td>Fluorescence is inherently multiplicative (photon counting &times;
        staining efficiency &times; exposure). Raw intensities are
        log-normal by physics.</td></tr>
<tr><td>Distance</td>
    <td><code>nearest_neighbor_distance</code>, <code>skeleton_length</code></td>
    <td>Right-skewed by geometry &mdash; isolated cells or cells with
        long processes create heavy right tails.</td></tr>
<tr><td>Probability</td>
    <td><code>prob_weighted_volume</code>, <code>prob_soft_hard_ratio</code></td>
    <td><code>prob_weighted_volume</code>: proportional to cell volume (log-normal).
        <code>prob_soft_hard_ratio</code>: V_soft/|M|, unbounded above 1
        (right-skewed in under-segmented cells).</td></tr>
</table>

<h3>Auto-detected features (adaptive)</h3>
<p style="color:#aaa; font-size:12px;">
Additional features with |skewness| &gt; 1.0 (Fisher's definition) are
also log-transformed. This catches per-channel intensity features
(e.g., <code>nuclei_intensity_sum</code>) and any other new features
not in the curated list. The threshold of 1.0 follows the
<a style="color:#80deea;">pycytominer</a> convention.
</p>

<h3>Excluded from log-transform</h3>
<p style="color:#aaa; font-size:12px;">
Bounded [0, 1] features are never log-transformed &mdash;
<code>log1p</code> would compress their already-small range unhelpfully.
</p>
<table>
<tr><th>Feature</th><th>Range</th><th>Why excluded</th></tr>
<tr><td><code>extent</code>, <code>compactness</code>, <code>fill_ratio</code></td>
    <td>[0, 1]</td><td>Volume ratios &mdash; 1.0 = perfectly fills reference shape</td></tr>
<tr><td><code>solidity</code>, <code>true_sphericity</code>, <code>sphericity</code>,
    <code>ellipsoid_solidity</code></td>
    <td>[0, 1]</td><td>Shape measures &mdash; 1.0 = perfectly convex/spherical</td></tr>
<tr><td><code>prob_mean</code>, <code>prob_p10</code>, <code>prob_boundary_mean</code>,
    <code>prob_interior_mean</code></td>
    <td>[0, 1]</td><td>Probability values are bounded by definition</td></tr>
<tr><td><code>background_fraction</code>, <code>max_contact_fraction</code></td>
    <td>[0, 1]</td><td>Surface fraction ratios</td></tr>
</table>

<h2>Step 2: Robust Standardization</h2>
<p style="color:#aaa; font-size:12px;">
Each feature is centered and scaled using <strong>median / MAD</strong>
(median absolute deviation) instead of mean / std.
</p>
<table>
<tr><th>Operation</th><th>Equation</th><th>Why</th></tr>
<tr><td>Center</td>
    <td><span class="eq">x&#x0302; = x - median(x)</span></td>
    <td>Median is robust to outliers (unlike mean)</td></tr>
<tr><td>Scale</td>
    <td><span class="eq">z = x&#x0302; / (MAD &times; 1.4826)</span></td>
    <td>MAD &times; 1.4826 &asymp; &sigma; for normal data.
        Unlike std, MAD is not inflated by outliers,
        so outliers retain visibly extreme z-scores.</td></tr>
</table>

<h2>Step 3: Winsorization</h2>
<p style="color:#aaa; font-size:12px;">
Z-scores are clipped to &plusmn;10. Without this, a single extreme
outlier can get z=500+ (since MAD is insensitive to outliers,
their z-scores are not self-dampened) and hijack an entire
principal component.
</p>
<span class="eq">z_final = clip(z, -10, +10)</span>

<h2>Step 4: PCA Feature Selection</h2>
<p style="color:#aaa; font-size:12px;">
PCA and composite scoring use a curated subset of 18 features
selected for statistical independence (one representative per
correlated group). Configurable via QC Settings dialog.
</p>
<table>
<tr><th>Category</th><th>Features (18 total)</th></tr>
<tr><td>Morphology</td>
    <td><code>area</code>, <code>extent</code>, <code>equivalent_diameter</code></td></tr>
<tr><td>Shape</td>
    <td><code>aspect_ratio</code>, <code>compactness</code>,
        <code>true_sphericity</code>, <code>solidity</code></td></tr>
<tr><td>Orientation</td>
    <td><code>major_axis_z_angle</code>, <code>inertia_elongation</code></td></tr>
<tr><td>Surface</td>
    <td><code>surface_area</code></td></tr>
<tr><td>Intensity</td>
    <td><code>intensity_mean</code>, <code>boundary_interior_ratio</code></td></tr>
<tr><td>Topology</td>
    <td><code>neighbor_count</code>, <code>nearest_neighbor_distance</code>,
        <code>skeleton_branches</code></td></tr>
<tr><td>Probability</td>
    <td><code>prob_mean</code>, <code>prob_boundary_drop</code>,
        <code>prob_boundary_entropy</code></td></tr>
</table>
<p style="color:#999; font-size:11px;">
<em>Note: Per-channel intensity features are not included in the default
PCA set but can be added via the QC Settings dialog.</em>
</p>

<h2>Z-Stratified Cross-Section (6 features)</h2>
<p style="color:#aaa; font-size:12px;">
Epithelial polarity features. Analyzes how cross-section area changes along the
cell's Z-extent. Captures pyramidal, columnar, and inverted cell shapes.
</p>
<table>
<tr><th>Feature</th><th>What it measures</th><th>Interpretation</th></tr>
<tr><td><code>basal_area</code></td><td>Mean area in bottom 20% of Z-range</td>
    <td>Large = wide base</td></tr>
<tr><td><code>apical_area</code></td><td>Mean area in top 20% of Z-range</td>
    <td>Large = wide top</td></tr>
<tr><td><code>apical_basal_area_ratio</code></td>
    <td><span class="eq">apical / basal</span></td>
    <td>&lt;0.5 = pyramidal (wide base). ~1.0 = columnar. &gt;2.0 = inverted.</td></tr>
<tr><td><code>max_cross_section_z_relative</code></td>
    <td>Z of widest slice, normalized [0,1]</td>
    <td>0 = widest at bottom. 0.5 = barrel/vase. 1 = widest at top.</td></tr>
<tr><td><code>cross_section_taper</code></td>
    <td><span class="eq">slope(area vs Z)</span></td>
    <td>Negative = tapering upward (pyramid). Positive = expanding. ~0 = columnar.</td></tr>
<tr><td><code>cross_section_bulge</code></td>
    <td><span class="eq">RMS(residual) / mean(area)</span></td>
    <td>Low = clean taper/column. High = vase, barrel, or dumbbell (merge signal).</td></tr>
</table>

<h2>Neighbor Birth-Radius Filtration (6 features)</h2>
<p style="color:#aaa; font-size:12px;">
Instead of evaluating neighbors at a single fixed radius, records the full
filtration &mdash; which neighbors appear at each dilation step.
Eliminates the "what radius?" question.
</p>
<table>
<tr><th>Feature</th><th>What it measures</th><th>Interpretation</th></tr>
<tr><td><code>nn_count_r1</code></td><td>Neighbors at radius 1 (direct contact)</td>
    <td>The "hard" neighbor count &mdash; masks physically touching.</td></tr>
<tr><td><code>nn_count_r3</code></td><td>Neighbors within radius 3</td>
    <td>Includes near-neighbors across typical 1-2 voxel gaps.</td></tr>
<tr><td><code>nn_birth_median</code></td><td>Median birth radius across all neighbors</td>
    <td>More robust than nearest_neighbor_distance (which is just the minimum).
    Typical gap width to neighbors.</td></tr>
<tr><td><code>nn_birth_std</code></td><td>Std of birth radii</td>
    <td>Low = uniform packing. High = heterogeneous (some touching, some distant).</td></tr>
<tr><td><code>nn_accumulation_rate</code></td>
    <td><span class="eq">slope(cumulative count vs radius)</span></td>
    <td>Steep = dense tissue. Flat = sparse region.</td></tr>
<tr><td><code>nn_accumulation_knee</code></td><td>Radius at 80% of total neighbors</td>
    <td>Early (r=1-2) = tight packing. Late (r=5+) = scattered.</td></tr>
</table>

<h2>Neighbor Quality (3 features)</h2>
<p style="color:#aaa; font-size:12px;">
First contextual features &mdash; compare a cell's size to its neighbors.
Catches cells that look fine individually but don't fit their local context.
</p>
<table>
<tr><th>Feature</th><th>What it measures</th><th>Interpretation</th></tr>
<tr><td><code>mean_neighbor_size_ratio</code></td>
    <td><span class="eq">cell volume / mean(neighbor volumes)</span></td>
    <td>~1.0 = similar size (normal). &gt;2.0 = much larger (merge). &lt;0.5 = much smaller (fragment).</td></tr>
<tr><td><code>neighbor_size_cv</code></td>
    <td><span class="eq">std(neighbor volumes) / mean(neighbor volumes)</span></td>
    <td>High = cell at boundary between different-sized populations.</td></tr>
<tr><td><code>contact_weighted_neighbor_size_ratio</code></td>
    <td>Like mean ratio but weighted by contact area</td>
    <td>Emphasizes neighbors that share more boundary surface.</td></tr>
</table>

<h2>Named Composite Signals (4 features)</h2>
<p style="color:#aaa; font-size:12px;">
Targeted outlier scores for specific failure modes. Built from weighted robust
z-scores of existing primitives. Give biologists actionable categories.
</p>
<table>
<tr><th>Feature</th><th>Inputs</th><th>Interpretation</th></tr>
<tr><td><code>merge_signal</code></td>
    <td>boundary_interior_ratio (&minus;), inertia_elongation (+),
    skeleton_branches (+), euler_number (&minus;),
    cross_section_bulge (+), mean_neighbor_size_ratio (+)</td>
    <td>"How merge-like is this cell?" High = suspicious.</td></tr>
<tr><td><code>fragment_signal</code></td>
    <td>area (&minus;), neighbor_count (&minus;),
    background_fraction (+), mean_neighbor_size_ratio (&minus;)</td>
    <td>"How fragment-like?" High = suspicious.</td></tr>
<tr><td><code>boundary_confidence</code></td>
    <td>prob_edge_gradient (+), 1&minus;prob_boundary_entropy (+),
    prob_boundary_mean (+)</td>
    <td>"How confident is the boundary?" High = good.
    &#x26A0; Requires probability features.</td></tr>
<tr><td><code>shape_abnormality</code></td>
    <td>apical_basal_area_ratio (+), cross_section_taper (+),
    inertia_elongation (+), true_sphericity (&minus;)</td>
    <td>"How unusual is the shape?" Population-relative outlier score.</td></tr>
</table>
<p style="color:#999; font-size:11px;">
<em>(+) = high value contributes to signal. (&minus;) = inverted (low value contributes).
All use median/MAD robust z-scores, winsorized to &plusmn;10.</em>
</p>

<h2>Z-Stratified Neighbor Topology (4 features)</h2>
<p style="color:#aaa; font-size:12px;">
Tracks which neighbors are present at different Z-levels within the cell.
Captures 3D neighborhood arrangement along the apical-basal axis.
Minimum Z-span guard: cells &lt; 5 Z-slices get 0 (unreliable).
</p>
<table>
<tr><th>Feature</th><th>What it measures</th><th>Interpretation</th></tr>
<tr><td><code>basal_neighbor_count</code></td>
    <td>Distinct neighbors in bottom third of Z-range</td>
    <td>High = many basal contacts</td></tr>
<tr><td><code>apical_neighbor_count</code></td>
    <td>Distinct neighbors in top third of Z-range</td>
    <td>High = many apical contacts</td></tr>
<tr><td><code>neighbor_z_turnover</code></td>
    <td><span class="eq">1 &minus; |basal &cap; apical| / |basal &cup; apical|</span></td>
    <td>0 = same neighbors top and bottom (columnar packing).
    1 = completely different (multilayer, fold).
    <strong>The most informative single scalar for 3D tissue organization.</strong></td></tr>
<tr><td><code>neighbor_persistence_ratio</code></td>
    <td><span class="eq">|basal &cap; apical| / total neighbors</span></td>
    <td>High = stable columnar packing (neighbors span full height).
    Low = transient contacts.</td></tr>
</table>

<h2>Border Cell Detection (1 feature)</h2>
<p style="color:#aaa; font-size:12px;">
Flags labels touching any volume face (Z=0, Z=max, Y=0, Y=max, X=0, X=max).
Border cells have truncated morphology and are excluded from PCA fitting.
</p>
<table>
<tr><th>Feature</th><th>Type</th><th>Interpretation</th></tr>
<tr><td><code>is_border_cell</code></td><td>Binary (0/1)</td>
    <td>1 = touches volume boundary. Excluded from PCA/standardization
    fitting but projected as supplementary points.</td></tr>
</table>

<h2>Position / Centroid (3 features)</h2>
<p style="color:#aaa; font-size:12px;">
Center of mass of the cell mask, in voxel coordinates. Useful for spatial
analysis and verifying Z-orientation. Excluded from PCA by default.
</p>
<table>
<tr><th>Feature</th><th>Axis</th><th>Use</th></tr>
<tr><td><code>centroid-0</code></td><td>Z</td>
    <td>Depth in tissue (basal-apical position). Color-by-Z reveals stratification.</td></tr>
<tr><td><code>centroid-1</code></td><td>Y</td>
    <td>Lateral position.</td></tr>
<tr><td><code>centroid-2</code></td><td>X</td>
    <td>Lateral position.</td></tr>
</table>

</body></html>
"""

# Active status label style — green accent matching Step 4 "Apply" theme
_ACTIVE_STATUS_STYLE = (
    "color: #6abf69; font-size: 11px; "
    "font-family: 'Consolas', 'Cascadia Mono', 'Courier New', monospace;"
)
_IDLE_STATUS_STYLE = (
    "color: #aaaaaa; font-size: 11px; "
    "font-family: 'Consolas', 'Cascadia Mono', 'Courier New', monospace;"
)


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------


def _num_cells(result: dict):
    """The final cell count of a ``run_real`` result, when it has one."""
    inner = result.get("result") if isinstance(result, dict) else None
    stats = getattr(inner, "stats", None) or {}
    val = stats.get("num_cells_final") if isinstance(stats, dict) else None
    try:
        return int(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _measure_pending(final: str, token: int) -> str:
    """Where a measure job writes a file before its finish accepts it."""
    return f"{final}.{int(token)}.pending"


def _tier_map_of(df) -> dict:
    """``{label: tier}`` for the marked rows of *df* (empty tiers skipped)."""
    if df is None or "qc_tier" not in df.columns or "label" not in df.columns:
        return {}
    marked = df[df["qc_tier"].fillna("").astype(str) != ""]
    return dict(zip(marked["label"].astype(int), marked["qc_tier"]))


def _apply_tier_map(df, tiers: dict) -> int:
    """Write ``qc_tier`` from *tiers* by label id; how many were kept."""
    if not tiers or "label" not in df.columns:
        return 0
    mapped = df["label"].astype(int).map(tiers)
    df["qc_tier"] = mapped.fillna("").astype(object)
    return int(mapped.notna().sum())


def _carry_tiers(prev_df, new_df) -> int:
    """Copy ``qc_tier`` from *prev_df* onto *new_df* rows with the same
    label id (F18).  Returns how many marks were kept.  Pure."""
    if (prev_df is None or new_df is None or "qc_tier" not in prev_df.columns
            or "label" not in prev_df.columns
            or "label" not in new_df.columns):
        return 0
    marked = prev_df[prev_df["qc_tier"].fillna("").astype(str) != ""]
    if marked.empty:
        return 0
    tiers = dict(zip(marked["label"].astype(int), marked["qc_tier"]))
    mapped = new_df["label"].astype(int).map(tiers)
    new_df["qc_tier"] = mapped.fillna("").astype(object)
    return int(mapped.notna().sum())


def _thread_running(thread) -> bool:
    try:
        return bool(thread.isRunning())
    except RuntimeError:      # wrapped C++ object deleted
        return False


def _launch_copies(stack_configs: list) -> list:
    """Shallow copies of the launch configs, with the model snapshot and
    the cache meta deep-copied: what the worker carries is what was true
    at LAUNCH, whatever the widgets or the caller's dicts do after."""
    import copy as _copy
    out = []
    for cfg in stack_configs:
        cfg = dict(cfg)
        for key in ("model_snapshot", "made_with", "cache_meta"):
            if key in cfg:
                cfg[key] = _copy.deepcopy(cfg[key])
        out.append(cfg)
    return out


class _StackSegmentationWorker(QObject):
    """Run real Cellpose + u-Segment3D on selected stacks."""

    stack_started = Signal(str, int, int)    # stack_name, index, total
    stack_completed = Signal(str, dict)      # stack_name, result_dict
    all_completed = Signal()
    slice_progress = Signal(str, str, int, int)  # stack, view, current, total
    log = Signal(str)
    error = Signal(str, str)                 # stack_name, error_msg

    def __init__(
        self,
        stack_configs: list,
        pipeline_service: ReconstructionPipelineService,
    ):
        super().__init__()
        self._configs = _launch_copies(stack_configs)
        self._service = pipeline_service
        self._cancel_requested = False

    def request_cancel(self):
        self._cancel_requested = True

    def run(self):
        # I-13: the terminal signal ``all_completed`` (the only link to the
        # thread's quit) is emitted in a ``finally``, whatever happens.
        try:
            self._run_all()
        except BaseException as exc:      # noqa: BLE001
            self.log.emit(f"  ERROR: {exc}")
        finally:
            self.all_completed.emit()

    def _run_all(self):
        from contextlib import redirect_stdout, redirect_stderr
        from io import StringIO

        total = len(self._configs)
        for i, cfg in enumerate(self._configs):
            if self._cancel_requested:
                self.log.emit("Cancelled by user.")
                break
            stack_name = cfg["stack_name"]
            self.stack_started.emit(stack_name, i, total)
            self.log.emit(
                f"[{i + 1}/{total}] Processing {stack_name}..."
            )
            try:
                def _progress_cb(stage_or_view, current, total_slices,
                                 _sn=stack_name):
                    self.slice_progress.emit(
                        _sn, str(stage_or_view), current, total_slices,
                    )

                buf = StringIO()
                with redirect_stdout(buf), redirect_stderr(buf):
                    # Step 4 outputs to artifacts/step4/runs/
                    output_base = str(
                        Path(cfg["working_dir"]) / "artifacts" / "step4" / "runs"
                    )
                    # Load prediction cache from disk if specified
                    cached_views = cfg.get("cached_views")
                    cache_dir = cfg.get("prediction_cache_dir")
                    cache_files = cfg.get("cache_files")
                    if cached_views is None and cache_files is not None:
                        # The files the launch found (F7, G1-11).
                        views = {}
                        for vn, (fp, pp, mp) in cache_files.items():
                            if fp and pp and Path(fp).exists() and Path(pp).exists():
                                flows = np.load(str(fp)).astype(np.float32)
                                probs = np.load(str(pp)).astype(np.float32)
                                masks = (
                                    np.load(str(mp)).astype(np.int32)
                                    if mp and Path(mp).exists() else None
                                )
                                views[vn] = (flows, probs, masks)
                        if len(views) == 3:
                            cached_views = views
                            self.log.emit(
                                f"  Loaded prediction cache for {cfg['stack_name']}"
                            )
                    elif cached_views is None and cache_dir:
                        stem = Path(cfg["stack_name"]).stem
                        cache_path = Path(cache_dir)
                        views = {}
                        for vn in ("xy", "xz", "yz"):
                            fp = cache_path / f"{stem}_{vn}_flows.npy"
                            pp = cache_path / f"{stem}_{vn}_probs.npy"
                            mp = cache_path / f"{stem}_{vn}_masks.npy"
                            if fp.exists() and pp.exists():
                                flows = np.load(str(fp)).astype(np.float32)
                                probs = np.load(str(pp)).astype(np.float32)
                                masks = (
                                    np.load(str(mp)).astype(np.int32)
                                    if mp.exists() else None
                                )
                                views[vn] = (flows, probs, masks)
                        if len(views) == 3:
                            cached_views = views
                            self.log.emit(
                                f"  Loaded prediction cache for {cfg['stack_name']}"
                            )

                    # TODO(re-audit L-b): run_real names its run folder to
                    # the SECOND (packages/epicycle_reconstruction/logic.py,
                    # run_id) and takes no run_id, so two stacks started in
                    # one second would share runs/{run_id}/labels_3d.tif.
                    # packages/ is not changed in B0; the delete side keeps
                    # a folder another stack uses (audit D4).
                    if cfg.get("require_cache") and cached_views is None:
                        # Never re-predict inside Reconstruct with default
                        # channels (G1-11, assessment #12).
                        raise RuntimeError(
                            "the 2D predictions are incomplete; "
                            "Predict it again first")
                    result = self._service.run_real(
                        working_dir=cfg["working_dir"],
                        source_stack_path=cfg["source_stack_path"],
                        lateral_model_path=cfg["lateral_model_path"],
                        axial_model_path=cfg["axial_model_path"],
                        seg_config=cfg.get("seg_config"),
                        recon_config=cfg.get("recon_config"),
                        axes=cfg.get("axes", "ZCYX"),
                        cached_views=cached_views,
                        progress_cb=_progress_cb,
                        output_base=output_base,
                    )
                captured = buf.getvalue().strip()
                if captured:
                    for line in captured.split("\n"):
                        self.log.emit(f"  {line}")
                self.log.emit(
                    f"  Done: {result.get('num_cells_final', '?')} cells, "
                    f"saved to {Path(result.get('labels_path', '')).name}"
                )
                # Include source_stack_path in result (not in run_real return)
                result["source_stack_path"] = cfg["source_stack_path"]
                result["made_with"] = cfg.get("made_with")
                result["num_cells_final"] = result.get(
                    "num_cells_final", _num_cells(result))
                self.stack_completed.emit(stack_name, result)
            except Exception as exc:
                self.log.emit(f"  ERROR: {exc}")
                self.error.emit(stack_name, str(exc))


class _StackPredictionWorker(QObject):
    """Run dense 2D prediction only (no reconstruction). Saves cache."""

    stack_started = Signal(str, int, int)
    stack_completed = Signal(str, dict)    # stack_name, {"views": {...}}
    all_completed = Signal()
    slice_progress = Signal(str, str, int, int)
    log = Signal(str)
    error = Signal(str, str)

    def __init__(self, stack_configs: list):
        super().__init__()
        self._configs = _launch_copies(stack_configs)
        self._cancel_requested = False

    def request_cancel(self):
        self._cancel_requested = True

    def run(self):
        # I-13: ``all_completed`` in a ``finally``.  The import used to sit
        # outside any try, so an ImportError emitted nothing and the thread
        # never quit.
        try:
            self._run_all()
        except BaseException as exc:      # noqa: BLE001
            self.log.emit(f"  ERROR: {exc}")
        finally:
            self.all_completed.emit()

    def _run_all(self):
        try:
            from packages.epicycle_reconstruction.logic import segment_views
        except Exception as exc:      # noqa: BLE001
            msg = f"could not load the prediction code: {exc}"
            self.log.emit(f"  ERROR: {msg}")
            for cfg in self._configs:
                self.error.emit(cfg["stack_name"], msg)
            return

        total = len(self._configs)
        sha1s = {}
        for i, cfg in enumerate(self._configs):
            if self._cancel_requested:
                self.log.emit("Cancelled by user.")
                break
            stack_name = cfg["stack_name"]
            self.stack_started.emit(stack_name, i, total)
            self.log.emit(f"[{i + 1}/{total}] Predicting {stack_name}...")
            # v2.6: the model sha1s and the library versions, here
            # on the worker (a sha1 of a large model file is not a UI job)
            try:
                from ._step4_models import finish_run_snapshot
                finish_run_snapshot(cfg.get("model_snapshot"), sha1s)
            except Exception as exc:      # noqa: BLE001
                self.log.emit(f"  snapshot: {exc}")
            try:
                import tifffile
                image = tifffile.imread(cfg["source_stack_path"])

                def _progress_cb(stage, current, total_slices,
                                 _sn=stack_name):
                    self.slice_progress.emit(
                        _sn, str(stage), current, total_slices,
                    )

                views = segment_views(
                    image,
                    cfg["lateral_model_path"],
                    cfg["axial_model_path"],
                    cfg.get("seg_config"),
                    axes=cfg.get("axes", "ZCYX"),
                    progress_cb=_progress_cb,
                )
                self.log.emit(f"  Prediction complete for {stack_name}")
                self.stack_completed.emit(stack_name, {
                    "views": views,
                    "source_stack_path": cfg["source_stack_path"],
                    "model_snapshot": cfg.get("model_snapshot"),
                    "cache_meta": cfg.get("cache_meta"),
                })
            except Exception as exc:
                self.log.emit(f"  ERROR: {exc}")
                self.error.emit(stack_name, str(exc))


class _StackQCFeatureWorker(QObject):
    """Extract QC features for a single stack in a background thread."""

    finished = Signal(object, str, int)   # (DataFrame, stack_name, token)
    error = Signal(str, str, int)         # (message, stack_name, token)
    csv_saved = Signal(str, str)     # (stack_name, csv_path)
    progress = Signal(int)           # percent 0-100
    log = Signal(str)

    def __init__(
        self,
        labels_path: str,
        stack_name: str,
        csv_path: str,
        *,
        spacing=None,
        source_stack_path: str = "",
        seg_channel: int = 0,
        prob_path: str = "",
        channel_configs: Optional[list] = None,
        skip_probability: bool = False,
        labels_array: Optional[np.ndarray] = None,
        exclude_border: bool = False,
        invert_z: bool = False,
        token: int = 0,
        tier_map: Optional[dict] = None,
        tier_csv: str = "",
    ):
        super().__init__()
        # The SAVED marks at launch (audit D2, re-audit M1): read from the
        # saved CSV and written into the CSV this worker saves.  A
        # ``tier_map`` given directly wins (tests).
        self._tier_map = dict(tier_map or {})
        self._tier_csv = tier_csv
        # Stamped at launch and carried in the terminal signals (F1): the
        # finish is a bound method, so ``sender()`` is not needed.
        self._token = int(token)
        self._emitted = False
        self._labels_path = labels_path
        self._stack_name = stack_name
        self._csv_path = csv_path
        self._spacing = spacing
        self._source_stack_path = source_stack_path
        self._seg_channel = seg_channel
        self._prob_path = prob_path
        self._channel_configs = channel_configs or []
        self._skip_probability = skip_probability
        self._labels_array = labels_array
        self._exclude_border = exclude_border
        self._invert_z = invert_z

    def run(self):
        # I-13: one terminal signal, always -- a ``finally`` backstops any
        # exit that emitted neither ``finished`` nor ``error``.
        try:
            self._run_extraction()
        except BaseException as exc:      # noqa: BLE001
            if not self._emitted:
                self._emitted = True
                self.error.emit(f"{exc}", self._stack_name, self._token)
        finally:
            if not self._emitted:
                self._emitted = True
                self.error.emit("the measure stopped without a result",
                                self._stack_name, self._token)

    def _launch_tiers(self) -> dict:
        """The launch-time marks: the given map, else the saved CSV's."""
        if self._tier_map:
            return self._tier_map
        if self._tier_csv and Path(self._tier_csv).exists():
            try:
                import pandas as _pd
                old = _pd.read_csv(self._tier_csv)
                if {"label", "qc_tier"} <= set(old.columns):
                    return _tier_map_of(old)
            except Exception as exc:      # noqa: BLE001
                self.log.emit(f"[WARN] saved marks unreadable: {exc}")
        return {}

    def _run_extraction(self):
        try:
            import tifffile
            from .feature_registry import build_registry, run_extraction
            from napari_epicycle.fusion_reconstruction.qc_features import (
                extract_single_channel, save_patch_csv,
            )

            if self._labels_array is not None:
                self.log.emit("Using in-memory labels volume...")
                labels = self._labels_array
            else:
                self.log.emit(f"Loading labels from {Path(self._labels_path).name}...")
                labels = tifffile.imread(self._labels_path)
            n_labels = len(np.unique(labels)) - 1
            self.log.emit(
                f"Volume: {labels.shape[0]}x{labels.shape[1]}x{labels.shape[2]} "
                f"({labels.size / 1e6:.1f}M voxels, {n_labels} labels)"
            )
            self.progress.emit(5)

            # Build available inputs
            inputs = {"labels": labels}
            if self._spacing is not None:
                inputs["spacing"] = self._spacing
            if self._exclude_border:
                inputs["exclude_border"] = True
            if self._invert_z:
                inputs["invert_z"] = True

            # Load intensity image for segmentation channel
            self.log.emit(
                f"[debug] source_stack_path={self._source_stack_path!r}, "
                f"exists={Path(self._source_stack_path).exists() if self._source_stack_path else False}"
            )
            self.log.emit(
                f"[debug] channel_configs={self._channel_configs}"
            )
            if self._source_stack_path and Path(self._source_stack_path).exists():
                self.log.emit("Loading source stack for intensity features...")
                raw_stack = tifffile.imread(self._source_stack_path)
                self.log.emit(
                    f"[debug] raw_stack shape={raw_stack.shape}, "
                    f"labels shape={labels.shape}"
                )
                # NOTE: intensity_image is NOT set here — intensity features
                # are computed per-channel via compute_channel_intensity.
                # raw_stack is provided for per-channel extraction.
                if self._channel_configs:
                    inputs["raw_stack"] = raw_stack
                    inputs["seg_channel"] = self._seg_channel
                    self.log.emit(
                        f"[debug] raw_stack added to inputs for "
                        f"{len(self._channel_configs)} channel(s)"
                    )
                else:
                    self.log.emit(
                        "[info] No channels selected — skipping intensity features"
                    )
            else:
                self.log.emit(
                    "[WARN] Source stack not found — skipping intensity features"
                )
            self.progress.emit(10)

            # Load probability volume
            if (
                not self._skip_probability
                and self._prob_path
                and Path(self._prob_path).exists()
            ):
                self.log.emit("Loading probability volume...")
                inputs["prob_volume"] = np.load(self._prob_path)

            # Build registry with channel configs
            registry = build_registry(
                checked_channels=self._channel_configs or None
            )
            self.log.emit(
                f"[debug] Registry: {len(registry)} groups, "
                f"inputs: {list(inputs.keys())}"
            )

            # Run extraction
            self.log.emit("Running feature extraction...")
            df = run_extraction(
                registry, inputs,
                post_smooth=self._skip_probability,
                progress_cb=lambda pct: self.progress.emit(
                    10 + int(pct * 0.85)
                ),
            )

            # Free large arrays
            inputs.pop("prob_volume", None)
            inputs.pop("raw_stack", None)

            self.progress.emit(95)
            intensity_cols = [c for c in df.columns if "intensity" in c.lower()]
            self.log.emit(
                f"[debug] {len(df.columns)} columns total, "
                f"{len(intensity_cols)} intensity columns: {intensity_cols[:10]}"
            )
            _apply_tier_map(df, self._launch_tiers())
            # The worker writes PENDING files only (audit D2 / D3): the
            # guarded finish on the main thread moves them into place, so a
            # stale or failed measure never overwrites the saved CSV.
            pending_csv = _measure_pending(self._csv_path, self._token)
            save_patch_csv(df, pending_csv)
            self.csv_saved.emit(self._stack_name, pending_csv)

            # Save adjacency list if available (from neighbor computation)
            adj_rows = df.attrs.get("_adjacency_list")
            if adj_rows:
                adj_path = Path(_measure_pending(str(
                    Path(self._csv_path).parent
                    / f"{self._stack_name}_adjacency.csv"), self._token))
                import pandas as _pd
                adj_df = _pd.DataFrame(adj_rows)
                adj_df.to_csv(adj_path, index=False)
                self.log.emit(
                    f"Adjacency list: {len(adj_df)} edges → "
                    f"{adj_path.name}"
                )

            self.log.emit(
                f"Done — {len(df)} labels extracted, saved to "
                f"{Path(self._csv_path).name}"
            )
            self.progress.emit(100)
            self._emitted = True
            self.finished.emit(df, self._stack_name, self._token)
        except Exception as exc:
            import traceback
            self._emitted = True
            self.error.emit(f"{exc}\n{traceback.format_exc()}",
                            self._stack_name, self._token)


class _SmoothBatchWorker(QObject):
    """Smooth All through Batch Refine's executor (F2).

    ``run_batch_recipe`` works on its OWN copy of the volume it is given
    (the input is never written), crops each object with ``2 + ceil(4σ)``
    per axis, computes every object from the ORIGINAL snapshot, writes in
    ascending id order with lower-id-wins, and leaves an object whose
    result would be empty exactly as it was (``empty="skip"``, R1c).  The
    finish handler swaps ``result.working_volume`` in on the main thread;
    no worker ever writes an array a napari layer displays (I-2).
    """

    progress = Signal(int)
    finished = Signal(object, str, int)   # (BatchResult, stack, token)
    error = Signal(str, str, int)         # (message, stack, token)
    log = Signal(str)

    def __init__(
        self,
        labels: np.ndarray,
        sigma_z: float = 3.0,
        sigma_xy: float = 1.5,
        preserve_topology: bool = True,
        *,
        stack_name: str = "",
        token: int = 0,
    ):
        super().__init__()
        import threading
        self._labels = labels
        self._sigma_z = float(sigma_z)
        self._sigma_xy = float(sigma_xy)
        self._preserve = bool(preserve_topology)
        self._stack_name = stack_name
        self._token = int(token)
        self._cancel = threading.Event()
        self._emitted = False

    def request_cancel(self):
        self._cancel.set()

    def run(self):
        try:
            self._run_smooth()
        except BaseException as exc:      # noqa: BLE001
            if not self._emitted:
                import traceback
                self._emitted = True
                self.error.emit(f"{exc}\n{traceback.format_exc()}",
                                self._stack_name, self._token)
        finally:
            if not self._emitted:
                self._emitted = True
                self.error.emit("the smoothing stopped without a result",
                                self._stack_name, self._token)

    def _run_smooth(self):
        from packages.epicycle_annotator.batch_refine import (
            RefineStep, run_batch_recipe,
        )

        ids = np.unique(self._labels)
        ids = [int(i) for i in ids if i > 0]
        self.log.emit(
            f"Smoothing {len(ids)} labels (\u03c3_z={self._sigma_z}, "
            f"\u03c3_xy={self._sigma_xy})...")
        recipe = [RefineStep("sdf", {
            "sigma_z": self._sigma_z, "sigma_xy": self._sigma_xy,
            "preserve_topology": self._preserve,
        })]

        def _progress(done, total):
            self.progress.emit(int(done * 100 / max(total, 1)))

        result = run_batch_recipe(
            self._labels, ids, recipe, empty="skip",
            cancel_event=self._cancel, progress_cb=_progress,
        )
        self.log.emit(
            f"Smoothed {len(result.touched_ids)} labels, "
            f"{result.n_changed_total} voxels changed; "
            f"{len(result.emptied_ids)} thin labels kept as they were.")
        self._emitted = True
        self.finished.emit(result, self._stack_name, self._token)


class _StackLoadWorker(QObject):
    """Read one stack for ③ off the main thread.

    It reads the labels TIFF, counts the cells, backs up the first-load
    original with an atomic copy, and reads the raw stack ONCE (today's
    second read in ``_auto_populate_channels`` is gone).  It never writes
    the record (grill 1 G1-22): the backup path it made travels in the
    result, and the finish handler (main thread, token and stack checked)
    writes ``original_labels_path``.  One terminal signal, always (I-13).
    """

    finished = Signal(object, str, int)   # (result dict, stack, token)
    error = Signal(str, str, int)         # (message, stack, token)

    def __init__(self, *, stack_name: str, labels_path: str,
                 source_path: str = "", backup_path: str = "",
                 token: int = 0, kind: str = "load"):
        super().__init__()
        self._stack_name = stack_name
        self._labels_path = labels_path
        self._source_path = source_path
        self._backup_path = backup_path
        self._token = int(token)
        self._kind = kind
        self._cancel = False
        self._emitted = False

    def request_cancel(self):
        self._cancel = True

    def run(self):
        try:
            result = self._load()
            if not self._emitted:
                self._emitted = True
                self.finished.emit(result, self._stack_name, self._token)
        except BaseException as exc:      # noqa: BLE001
            if not self._emitted:
                import traceback
                self._emitted = True
                self.error.emit(f"{exc}\n{traceback.format_exc()}",
                                self._stack_name, self._token)
        finally:
            if not self._emitted:
                self._emitted = True
                self.error.emit("the load stopped without a result",
                                self._stack_name, self._token)

    def _check(self):
        if self._cancel:
            raise RuntimeError("the load was cancelled")

    def _load(self) -> dict:
        import shutil
        import tifffile

        if self._kind == "restore":
            # BUG A (2026-10-05): the loaded stack's image channels only;
            # the labels stay the in-memory array (unsaved edits kept)
            print(f"[STEP4-LOAD] worker: re-reading the image of "
                  f"{self._stack_name}")
            raw = None
            if self._source_path and Path(self._source_path).exists():
                raw = SegmentStackController._smart_load_image(
                    tifffile.imread(self._source_path))
            self._check()
            return {"labels": None, "n_cells": None, "backup": "",
                    "raw": raw, "source_path": self._source_path,
                    "kind": "restore"}
        print(f"[STEP4-LOAD] worker: reading {self._stack_name}")
        labels = tifffile.imread(self._labels_path)
        self._check()
        ids = np.unique(labels)
        n_cells = int(ids.size - (1 if ids.size and ids[0] == 0 else 0))
        backup = ""
        bp = self._backup_path
        if bp and not Path(bp).exists():
            tmp = bp + ".tmp"
            try:
                shutil.copy2(self._labels_path, tmp)
                os.replace(tmp, bp)
                backup = bp
            except OSError as exc:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                print(f"[STEP4-LOAD] backup not made: {exc}")
        self._check()
        raw = None
        if self._source_path and Path(self._source_path).exists():
            raw = SegmentStackController._smart_load_image(
                tifffile.imread(self._source_path))
        self._check()
        print(f"[STEP4-LOAD] worker: {self._stack_name} read, "
              f"{n_cells} cells")
        return {"labels": labels, "n_cells": n_cells, "backup": backup,
                "raw": raw, "source_path": self._source_path,
                "kind": self._kind}


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class SegmentStackController:
    def __init__(
        self,
        widget,
        viewer,
        *,
        artifact_registry: Optional[FileArtifactRegistry] = None,
        run_store: Optional[StackSegmentationRunStore] = None,
        service: Optional[StackSegmentationService] = None,
    ):
        self.ui = widget
        self.viewer = viewer
        self.working_dir = ""
        self.artifact_registry = artifact_registry or FileArtifactRegistry()
        self.run_store = run_store or StackSegmentationRunStore()
        self.service = service or StackSegmentationService()
        self.state_manager = StackStateManager()

        self._init_runtime_state()

        self._connect_signals()
        self._update_qc_button_states()

    def _init_runtime_state(self):
        """Every runtime attribute, in one place (tests build a bare
        instance with ``__new__`` and call this)."""
        # K6: Step 4 is not the active step until the shell enters it
        self._step4_active = False
        self._qc_features_df = None
        self._labels_volume = None
        self._current_stack_name = None
        self._selected_label_ids = []
        self._isolated_label_ids = set()
        # Colour sync (2026-10-05): the last colours put on the viewer's
        # labels (C6 skips a recolour with the same ones), how many real
        # recolours ran, and Colour by before "Colour plot by this" (C3).
        self._viewer_colour_digest = None
        self._viewer_colour_layer = None
        self._viewer_recolours = 0
        self._colour_by_before_pick = None
        # Unsaved work (F8): marks and label edits are separate flags;
        # ``_has_unsaved_changes`` is their OR (a property).
        self._tiers_dirty = False
        self._labels_dirty = False
        # B3b: the Workbench's mark history, one entry per Mark / Unmark:
        # (label ids, their tiers before), popped by Undo.
        self._mark_history = []
        # B3b: the one Workbench map job.
        self._eda_thread = None
        self._eda_worker = None
        self._eda_launch = None
        self._eda_stopping = False
        self._eda_abandoned = False
        self._eda_serial = 0
        self._last_pca_key = None
        self._last_pca_residual = False
        # Measured seconds per cell of the last run of each map kind
        # (audit X1-5: BUSY's estimate after the first run).
        self._eda_rates = {}
        self._eda_warmed = set()
        # Edit timestamps held until Save / Finalize (G1-9).
        self._pending_edits = {}
        # One-level Undo slot (F2 fills it with the pre-smooth array).
        self._undo_slot = None
        # Bumped by every change to _labels_volume, so a worker that read an
        # older volume cannot land its result over a newer edit.
        self._labels_epoch = 0
        self._pending_border_removal = False
        self._last_pca_projected = None  # full N-component PCA for UMAP/t-SNE/clustering input
        self._analysis_widget = None
        self._qc_thread = None
        self._qc_worker = None
        self._seg_thread = None
        self._seg_worker = None
        self._smooth_thread = None
        self._smooth_worker = None
        # ③ load, the loaded stack's cell count, the mesh export
        self._load_thread = None
        self._load_worker = None
        self._load_token = None
        self._load_launch = None
        self._n_cells = 0
        self._mesh_proc = None
        self._mesh_timer = None
        self._mesh_job = None
        self._cached_eda_state = None  # loaded EDA cache for deferred injection
        # The Step 4 lock flags: all False at construction.
        for flag in _STEP4_ALL_FLAGS:
            setattr(self, flag, False)
        # True after a read-only adopt until the user acts here (L12).
        self._adopt_only = False
        # One launch token for every Step 4 worker (F1, F2, F4).
        self._step4_token = 0
        self._measure_token = None
        self._smooth_token = None
        # B1a: the listing, the ticks' rows, the active model and the run.
        self._sources = []
        self._stack_folders = []
        self._transient_meta = {}
        self._rows = []
        self._running = {}
        self._run_kind = None
        self._run_total = 0
        self._run_done = 0
        self._run_failures = 0
        self._seg_index = 0
        self._seg_start_time = 0.0
        self._active_model = None
        self._trained_codes = None
        self._applying_preset = False
        self._step3_preset = None
        self._shown_px = None
        self._last_done_line = ""
        self._tick_save_pending = False
        self._thumb_thread = None
        self._thumb_worker = None
        self._thumb_cache = None
        self._thumb_requested = {}

    # ------------------------------------------------------------------
    # Workers: flags, tokens and the thread-finished backstop
    # ------------------------------------------------------------------

    def _next_token(self) -> int:
        self._step4_token = int(getattr(self, "_step4_token", 0)) + 1
        return self._step4_token

    def _set_step4_flag(self, flag: str, value: bool) -> None:
        setattr(self, flag, bool(value))
        print(f"[STEP4-LOCK] {flag}={bool(value)}")
        self._refresh_step4_lock()

    def _refresh_step4_lock(self) -> bool:
        """Called on every flag change: the lock set greys (① and ②
        here, ③ in ``_refresh_mt_panel``, the Export window with it, the
        Workbench in ``_refresh_workbench_lock``), the lock lines carry the
        job's sentence, and the step nav takes the ``"step4"`` owner while
        Step 4 is the active step."""
        locked = _step4_is_locked(self)
        run = bool(getattr(self, "_predict_running", False)
                   or getattr(self, "_reconstruct_running", False))
        setter = getattr(self.ui, "set_surface_locked", None)
        self._lock_labels_layer()
        if callable(setter):
            try:
                setter(locked, eye_live=run)
            except Exception as exc:      # noqa: BLE001
                print(f"[STEP4-LOCK] surface not locked: {exc}")
        try:
            self._refresh_segment_gates(keep_status=True)
        except Exception as exc:      # noqa: BLE001
            print(f"[STEP4-LOCK] gates not refreshed: {exc}")
        try:
            self._refresh_mt_panel()
        except Exception as exc:      # noqa: BLE001
            print(f"[STEP4-LOCK] (3) not refreshed: {exc}")
        self._refresh_workbench_lock()       # B3b: WB_LOCKED (B2 keeps it)
        self._sync_step4_nav_lock()
        return _step4_is_locked(self)

    #: K4: the jobs that read or replace the labels volume, so a paint
    #: during them would make the result stale.
    _VOLUME_JOB_FLAGS = ("_smooth_running", "_measure_running",
                         "_load_in_flight")

    def _lock_labels_layer(self) -> None:
        """K4: the stack labels layer is not editable while a volume job
        runs (Step 3's "drawing locked at launch"); restored when it ends,
        only if Step 4 locked it."""
        busy = any(getattr(self, f, False) for f in self._VOLUME_JOB_FLAGS)
        try:
            lyr = self._find_labels_layer()
        except Exception:      # noqa: BLE001
            lyr = None
        if lyr is None:
            self._labels_edit_locked = False
            return
        if busy and not getattr(self, "_labels_edit_locked", False):
            if not getattr(lyr, "editable", True):
                return            # L4: not editable already -- not ours
            try:
                self._labels_edit_mode = getattr(lyr, "mode", None)  # L2
                lyr.editable = False
                self._labels_edit_locked = True
                print("[STEP4-LOCK] labels layer locked for the job")
            except Exception as exc:      # noqa: BLE001
                print(f"[STEP4-LOCK] labels layer not locked: {exc}")
        elif not busy and getattr(self, "_labels_edit_locked", False):
            try:
                lyr.editable = True
                mode = getattr(self, "_labels_edit_mode", None)
                if mode is not None:
                    lyr.mode = mode           # L2: editable=False forced pan
            except Exception as exc:      # noqa: BLE001
                print(f"[STEP4-LOCK] labels layer not unlocked: {exc}")
            self._labels_edit_locked = False
            self._labels_edit_mode = None
            print("[STEP4-LOCK] labels layer editable again")

    def _sync_step4_nav_lock(self) -> None:
        """Hold the shell's step-nav lock under the ``"step4"`` owner while
        a block flag is set AND Step 4 is the active step; release only the
        owner Step 4 took (I-8, grill G1-4).  A Workbench map
        (``_eda_running``) is not a block flag, so it never locks the nav."""
        want = bool(_step4_is_locked(self)
                    and getattr(self, "_step4_active", False))
        held = bool(getattr(self, "_step4_nav_held", False))
        if want == held:
            return
        shell = getattr(self, "_workflow_shell", None)
        if shell is None or not hasattr(shell, "lock_step_nav"):
            return
        try:
            shell.lock_step_nav(want, owner="step4")
        except Exception as exc:      # noqa: BLE001
            print(f"[STEP4-LOCK] lock_step_nav failed: {exc}")
            return
        self._step4_nav_held = want
        print(f"[STEP4-LOCK] step nav {'taken' if want else 'released'} "
              f"(owner step4; {self._running_flag_names() or 'no job'})")

    def _start_step4_thread(self, worker, flag: str, *, quit_on, attr,
                            cleanup=None):
        """Move *worker* to a new QThread and start it with *flag* raised.

        Every signal in *quit_on* quits the thread; the flag is cleared by
        the thread's ``finished`` (the backstop), never only by a result
        handler, so a stale early return or a silent worker can never hold
        the lock (grill G1-5).  ``attr`` names the controller slot
        holding the thread (``_seg`` / ``_qc`` / ``_smooth``).
        """
        from functools import partial

        old = getattr(self, f"{attr}_thread", None)
        if old is not None:
            if _thread_running(old):
                # Audit L1: a second launch must never replace a running
                # thread -- its cleanup would then release the new job's
                # lock, and the replaced QThread could be collected while
                # it still runs.
                print(f"[STEP4-LOCK] {flag} refused: {attr} thread "
                      f"still running")
                try:
                    self.ui.log_output.append(
                        "[lock] refused: the previous job is still running")
                except Exception:      # noqa: BLE001
                    pass
                return None
            self._retire_thread(old)

        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        for name in quit_on:
            getattr(worker, name).connect(thread.quit)
        thread.finished.connect(
            partial(self._on_step4_thread_finished, flag, attr, thread,
                    cleanup))
        setattr(self, f"{attr}_thread", thread)
        setattr(self, f"{attr}_worker", worker)
        self._set_step4_flag(flag, True)
        thread.start()
        return thread

    def _retire_thread(self, thread):
        """Keep a finished (or finishing) QThread referenced until it has
        fully stopped, so Python never deletes it while it runs."""
        retired = getattr(self, "_retired_threads", None)
        if retired is None:
            retired = self._retired_threads = []
        retired[:] = [t for t in retired if _thread_running(t)]
        if thread is not None and thread not in retired:
            retired.append(thread)

    def _on_step4_thread_finished(self, flag, attr, thread, cleanup):
        """The backstop: clear the flag, run the job's cleanup, retire the
        thread object.  Only the job that OWNS the slot releases the lock
        (audit L1): a finish from a thread that is no longer the slot's
        never touches the live job's flag."""
        current = getattr(self, f"{attr}_thread", None)
        self._retire_thread(thread)
        if current not in (thread, None):
            print(f"[STEP4-LOCK] finish of a replaced {attr} thread: "
                  f"{flag} kept")
            return
        setattr(self, f"{attr}_thread", None)
        setattr(self, f"{attr}_worker", None)
        if cleanup is not None:
            try:
                cleanup()
            except Exception as exc:      # noqa: BLE001
                print(f"[STEP4-LOCK] cleanup after {flag} failed: {exc}")
        self._set_step4_flag(flag, False)
        try:
            self._update_qc_button_states()          # audit L7
        except Exception as exc:      # noqa: BLE001
            print(f"[STEP4-LOCK] button refresh failed: {exc}")

    def _refuse_while_locked(self, what: str) -> bool:
        """True (and a log line) when a Step 4 job is running."""
        if not _step4_is_locked(self):
            return False
        print(f"[STEP4-LOCK] {what} refused while "
              f"{self._running_flag_names()}")
        try:
            self.ui.log_output.append(
                S4T.LOCK_ACTION_REFUSED.format(action=what))
        except Exception:      # noqa: BLE001
            pass
        return True

    def _running_flag_names(self):
        from ._step4_lock import running_flags
        return ", ".join(running_flags(self))

    @property
    def _has_unsaved_changes(self) -> bool:
        return bool(getattr(self, "_tiers_dirty", False)
                    or getattr(self, "_labels_dirty", False))

    @_has_unsaved_changes.setter
    def _has_unsaved_changes(self, value) -> None:
        # Legacy writers: True means a label edit, False means "all saved".
        if value:
            self._labels_dirty = True
        else:
            # Saved or discarded: the held edit timestamps and the Undo slot
            # belong to the edits that are gone (audit D1).
            self._labels_dirty = False
            self._tiers_dirty = False
            self._pending_edits = {}
            self._undo_slot = None
            # A Save or a load ends the mark history.
            self._mark_history = []
            self._wb_sync_mark_state()

    @property
    def _pw(self):
        """Return the QCPlotWidget from the EDA dock, or None."""
        aw = self._analysis_widget
        if aw is not None and hasattr(aw, "qc_plot_widget"):
            return aw.qc_plot_widget
        return None

    @property
    def _sync_cb(self):
        """Return the sync_viewer_colors checkbox from the EDA dock."""
        aw = self._analysis_widget
        if aw is not None and hasattr(aw, "sync_viewer_colors_cb"):
            return aw.sync_viewer_colors_cb
        return None

    def _connect_signals(self):
        # ① Session
        self.ui.browse_workdir_btn.clicked.connect(self._browse_workdir)
        self.ui.add_folder_btn.clicked.connect(self._add_folder)
        self.ui.set_spacing_btn.clicked.connect(self._on_set_spacing_clicked)
        self.ui.ticks_changed.connect(self._on_ticks_changed)
        self.ui.stack_eye_clicked.connect(self._on_stack_eye)
        self.ui.stack_menu_requested.connect(self._on_stack_menu)

        # ② Segment — model, settings, stack selection + execution
        self.ui.model_change_btn.clicked.connect(self._on_model_change_clicked)
        self.ui.start_from_combo.activated.connect(self._on_start_from_chosen)
        self.ui.stacks_list.itemSelectionChanged.connect(
            self._on_stack_selected
        )
        self.ui.chan1_combo.currentIndexChanged.connect(
            self._on_channels_changed)
        self.ui.chan2_combo.currentIndexChanged.connect(
            self._on_channels_changed)
        self.ui.cell_width_um_sb.valueChanged.connect(
            self._on_cell_width_changed)
        self.ui.predict_btn.clicked.connect(self._run_prediction)
        self.ui.reconstruct_btn.clicked.connect(self._run_reconstruction)
        self.ui.cancel_btn.clicked.connect(self._cancel_run)
        self.ui.view_btn.clicked.connect(self._on_view_clicked)
        self._connect_setting_edits()

        # ③ Measure & tidy: a row click loads
        self.ui.mt_load_requested.connect(self._request_load)
        self.ui.open_analysis_btn.clicked.connect(self._on_workbench_button)
        self.ui.feature_reference_btn.clicked.connect(
            self._show_feature_reference
        )
        self.ui.extract_features_btn.clicked.connect(self._extract_features)
        self.ui.delete_bad_btn.clicked.connect(self._delete_bad_labels)
        self.ui.undo_btn.clicked.connect(self._undo_last_edit)
        self.ui.smooth_all_btn.clicked.connect(self._smooth_all)
        # fix_by_hand_btn: always grey, no handler
        self.ui.save_tiers_btn.clicked.connect(self._on_save_clicked)
        self.ui.finalize_btn.clicked.connect(self._on_finalize_clicked)
        self.ui.reset_original_btn.clicked.connect(self._reset_to_original)

        # The Export window
        self.ui.export_btn.clicked.connect(self._open_export_window)
        self.ui.export_dialog.features_requested.connect(
            self._export_all_stacks)
        self.ui.export_dialog.meshes_requested.connect(self._export_meshes)

        # napari's close: stop the thumbnail worker (shutdown())
        try:
            from qtpy.QtWidgets import QApplication
            app = QApplication.instance()
            if app is not None and not getattr(self, "_quit_wired", False):
                app.aboutToQuit.connect(self.shutdown)     # A1-6: once
                self._quit_wired = True
        except Exception as exc:      # noqa: BLE001
            print(f"[STEP4-UI] aboutToQuit not wired: {exc}")
        # the panel's destruction stops every worker too (Step 3's hook)
        try:
            destroyed = getattr(self.ui, "destroyed", None)
            if destroyed is not None and not getattr(
                    self, "_destroyed_wired", False):
                destroyed.connect(self._on_ui_destroyed)
                self._destroyed_wired = True
        except Exception as exc:      # noqa: BLE001
            print(f"[STEP4-UI] destroyed not wired: {exc}")
        self._refresh_segment_gates()

    # ------------------------------------------------------------------
    # Session handlers
    # ------------------------------------------------------------------

    def _browse_workdir(self):
        if _step4_is_locked(self):
            self.ui.log_output.append(S4T.LOCK_WORKSPACE_REFUSED)
            return
        path = QFileDialog.getExistingDirectory(
            None, "Select Workspace Folder")
        if not path:
            return
        changing = not self._same_workspace(self.working_dir or "", path)
        if changing:
            # F22: unsaved work asks Save / Discard / Stay first; Stay
            # (or a failed Save) keeps this workspace and this stack.
            if not self._settle_unsaved("a workspace change"):
                self.ui.log_output.append(
                    "[workspace] kept: the loaded stack has unsaved work")
                return
            if self._current_stack_name is not None:
                self._unload_stack("the workspace changed")
        self.working_dir = path
        self.ui.workdir_line.setText(path)
        self.state_manager.load(path)
        self._adopt_only = False
        self._report_state_messages(show=True)
        self._reconciled_on_browse = self._reconcile_state(write=False,
                                                           show=True)
        self._restore_session_from_state()
        self._refresh_stacks()
        if getattr(self.state_manager, "loaded_from", "") in (
                "legacy", "backup") \
                or getattr(self, "_reconciled_on_browse", 0):
            # A Browse is the user choosing this workspace: write the
            # migrated (or recovered) state now, so step4_state.json exists.  The legacy file itself is never written.
            self._persist_state()
        # An explicit Browse is the ONE thing that publishes a workspace to
        # the shell.
        self._publish_workspace_to_shell(path)

    # ------------------------------------------------------------------
    # Shared workspace
    # ------------------------------------------------------------------

    def adopt_workspace(self, path: str, previous: str = "") -> None:
        """Quietly take the shell's shared workspace — READ-ONLY.

        Called by the shell when the user enters this step, or when the
        shared workspace moves because the user browsed in another step
        (symmetric rule).  It fills the path box, loads the state and
        lists whatever stacks the saved session points at, so the step is
        usable without a second Browse — but it writes NOTHING: no
        `stack_segmentation_state.json`, and no stack rows registered into
        the state manager (invariant I12).  The moment the user actually
        chooses this step with Browse, the full Browse path runs and
        registers everything as before.

        `previous` is the workspace this step INHERITED before: adoption
        happens when the box is empty or still holds exactly that path, so a
        folder the user browsed to HERE is never overwritten (invariant I2).
        A re-adopt clears the stale listing first, so the step never shows
        the previous workspace's stacks under the new path.
        """
        if not path:
            return
        if _step4_is_locked(self):
            # a running job keeps its workspace (a Browse mid-run used
            # to point stack_completed at the other workspace's state).
            print(f"[STEP4-LOCK] adopt of {path} refused: a job is running")
            try:
                self.ui.log_output.append(S4T.LOCK_WORKSPACE_REFUSED)
            except Exception:      # noqa: BLE001
                pass
            return
        current = (self.working_dir or "").strip()
        replacing = False
        if current:
            if not previous or not self._same_workspace(current, previous):
                return
            if self._same_workspace(current, path):
                return
            replacing = True
            # F22: unsaved work asks Save / Discard / Stay first.
            if not self._settle_unsaved("a workspace change"):
                self.ui.log_output.append(
                    "[workspace] kept: the loaded stack has unsaved work")
                return
            if self._current_stack_name is not None:
                self._unload_stack("the workspace changed")
        self.working_dir = path
        self._suppress_publish = True
        try:
            self.ui.workdir_line.setText(path)
        finally:
            self._suppress_publish = False
        try:
            # READ-ONLY: an unreadable state file is reported, never renamed;
            # the first user action that saves re-loads in write mode.
            self.state_manager.load(path, write=False)
            self._report_state_messages(show=False)
            # In memory only: the adopt path writes nothing.
            self._reconcile_state(write=False)
            self._adopt_only = True
            if replacing:
                # Drop the previous workspace's listing before the new one is
                # read, so a workspace with no saved session shows an empty
                # list rather than the old workspace's stacks.
                self._sources = []
                self._stack_folders = []
                try:
                    self.ui.refresh_stack_lists([], ticked=())
                except Exception:      # noqa: BLE001
                    pass
            self._restore_session_from_state(persist=False)
        except Exception as exc:      # noqa: BLE001
            print(f"[WORKSPACE] Step 4 adopt could not read {path}: {exc}")

    @staticmethod
    def _same_workspace(a: str, b: str) -> bool:
        """Path comparison for the shared-workspace handshake (case + slash)."""
        if not a or not b:
            return False
        try:
            return (os.path.normcase(os.path.normpath(a))
                    == os.path.normcase(os.path.normpath(b)))
        except Exception:      # noqa: BLE001
            return a == b

    def _publish_workspace_to_shell(self, path: str) -> None:
        """Tell the shell about a workspace the USER chose here."""
        if getattr(self, "_suppress_publish", False):
            return
        shell = getattr(self, "_workflow_shell", None)
        setter = getattr(shell, "set_workspace", None)
        if not callable(setter):
            return
        try:
            setter(path, source_step="apply_3d")
        except Exception as exc:      # noqa: BLE001
            print(f"[WORKSPACE] publish from Step 4 failed: {exc}")

    # ------------------------------------------------------------------
    # The state file
    # ------------------------------------------------------------------

    def _report_state_messages(self, show: bool) -> None:
        """Log the last load's notices; show them too after a Browse."""
        sm = self.state_manager
        notices = sm.take_notices() if hasattr(sm, "take_notices") else []
        logs = sm.take_logs() if hasattr(sm, "take_logs") else []
        log = getattr(getattr(self, "ui", None), "log_output", None)
        for line in list(logs) + list(notices):
            print(f"[STEP4-STATE] {line}")
            if log is not None:
                try:
                    log.append(f"[state] {line}")
                except Exception:      # noqa: BLE001
                    pass
        if show and notices:
            try:
                from ..ui_shell.theme import themed_notice
                themed_notice(self.ui, "Step 4 state", "\n\n".join(notices))
            except Exception as exc:      # noqa: BLE001
                print(f"[STEP4-STATE] notice not shown: {exc}")

    def _reconcile_state(self, *, write: bool, show: bool = False) -> int:
        """After EVERY load: records whose files are gone are brought in
        line with the disk, and an orphaned run is adopted again (live
        driver, 2026-09-29).  Nothing is deleted.  ``write`` saves the
        result (never on the read-only adopt path)."""
        if not self.working_dir:
            return 0
        manifests = []
        store = getattr(self, "run_store", None)
        loader = getattr(store, "load_manifest", None)
        try:
            paths = store.list_manifests(self.working_dir) if store else []
        except Exception as exc:      # noqa: BLE001
            print(f"[STEP4-STATE] manifests unreadable: {exc}")
            paths = []
        for p in paths:
            if loader is None:
                break
            try:
                manifests.append(loader(p))
            except Exception as exc:      # noqa: BLE001
                print(f"[STEP4-STATE] manifest {p} unreadable: {exc}")
        try:
            changes = self.state_manager.reconcile_with_disk(
                cache_dir=self._prediction_cache_dir(), manifests=manifests)
        except Exception as exc:      # noqa: BLE001
            print(f"[STEP4-STATE] reconcile failed: {exc}")
            return 0
        log = getattr(getattr(self, "ui", None), "log_output", None)
        for name, what in changes.items():
            line = f"[STEP4-STATE] reconciled {name}: {what}"
            print(line)
            if log is not None:
                log.append(line)
        if changes:
            notice = S4T.state_reconciled_line(len(changes))
            if log is not None:
                log.append(f"[state] {notice}")
            if show:
                try:
                    from ..ui_shell.theme import themed_notice
                    themed_notice(self.ui, "Step 4 state", notice)
                except Exception as exc:      # noqa: BLE001
                    print(f"[STEP4-STATE] notice not shown: {exc}")
            if write:
                self._persist_state()
        return len(changes)

    def _ensure_writable_state(self) -> None:
        """After a read-only adopt that saw an unreadable state file, the
        first user action that saves re-loads in write mode, which sets the
        file aside.  A no-op otherwise."""
        self._adopt_only = False
        sm = self.state_manager
        if getattr(sm, "needs_write_load", False) and self.working_dir:
            print("[STEP4-STATE] write-mode re-load after adopt")
            sm.load(self.working_dir)
            self._report_state_messages(show=True)
            self._reconcile_state(write=True, show=True)

    def _persist_state(self) -> bool:
        """Save the state file; a failure is logged, never raised into Qt.

        It does NOT re-load after an adopt: a re-load here would drop the
        change being saved.  User actions call ``_ensure_writable_state``
        before they change anything; until then a save refuses.
        """
        try:
            ok = bool(self.state_manager.save())
        except Exception as exc:      # noqa: BLE001
            ok = False
            print(f"[STEP4-STATE] save failed: {exc}")
            try:
                self.ui.log_output.append(f"[state] save failed: {exc}")
            except Exception:      # noqa: BLE001
                pass
        if not ok and self.working_dir:
            blocked = getattr(self.state_manager, "save_blocked", False)
            line = S4T.STATE_SAVE_BLOCKED if blocked else S4T.STATE_SAVE_FAILED
            try:
                self.ui.log_output.append(f"[state] {line}")
            except Exception:      # noqa: BLE001
                pass
        return ok

    def _restore_session_from_state(self, persist: bool = True):
        """Restore the schema-2 session: the added folders, the
        ticks, the active model and Start from; then list the stacks.

        `persist=False` is the shared-workspace ADOPT path: the stacks are
        listed for the user but nothing is registered into the state manager.
        """
        session = self.state_manager.get_session_metadata() or {}
        folders = [f for f in (session.get("stack_folders") or []) if f]
        legacy = session.get("input_stacks_dir", "")
        if legacy and legacy not in folders:
            folders.insert(0, legacy)
        self._stack_folders = [f for f in folders if Path(f).is_dir()]
        snap = self.state_manager.get_active_model()
        if snap is None:
            # A schema-1 session B0 kept as it was: Import / Pretrained
            # (mode 0 "Auto from Workspace" is gone).
            mode = session.get("model_source")
            lat = session.get("lateral_model_path", "")
            ax = session.get("axial_model_path", "")
            if mode == 1 and lat:
                snap = S4M.model_snapshot(lat, ax or lat, kind="external",
                                          workspace=self.working_dir)
        self._set_active_model(snap, save=False)
        self._restore_start_from(session.get("start_from"))
        self._rescan_stacks(persist=persist)
        if snap is not None:
            self.ui.log_output.append(S4T.LOG_MODEL_SET.format(
                model=S4M.model_short_label(snap)))

    def _save_session_metadata(self):
        """Save the schema-2 session."""
        self._ensure_writable_state()
        self.state_manager.set_session_metadata(
            stack_folders=list(self._stack_folders),
            start_from=self._start_from_key(),
        )
        self.state_manager.set_ticked(self.ui.ticked_names())
        self.state_manager.set_active_model(self._active_model)
        self._persist_state()

    def _add_folder(self):
        """Add a folder…: every ``.tif`` / ``.tiff`` of a folder is
        listed; the folder is remembered in ``session.stack_folders``."""
        if self._refuse_while_locked("Add a folder"):
            return
        path = QFileDialog.getExistingDirectory(
            None, S4T.ADD_FOLDER_BTN
        )
        if not path:
            return
        self._ensure_writable_state()
        self._scan_stacks_directory(path)
        self.ui.log_output.append(S4T.LOG_FOLDER_ADDED.format(folder=path))
        self._save_session_metadata()

    def _scan_stacks_directory(self, stacks_dir: str, persist: bool = True):
        """Add *stacks_dir* to the listed folders and list the stacks.

        `persist=False` (the shared-workspace adopt path) lists the files and
        stops: nothing is registered and no TIFF is opened.
        """
        if not stacks_dir or not Path(stacks_dir).is_dir():
            return
        folders = getattr(self, "_stack_folders", None)
        if folders is None:
            folders = self._stack_folders = []
        if not any(self._same_workspace(stacks_dir, f) for f in folders):
            folders.append(stacks_dir)
        self._rescan_stacks(persist=persist)

    # ------------------------------------------------------------------
    # The workspace's one voxel size per patch (S1): patch_spacing.json
    # ------------------------------------------------------------------

    def _spacing_file_key(self, path) -> str:
        from ..data_preparation import _voxel_spacing as VS
        return VS.patch_spacing_key(path, self.working_dir) if path else ""

    def _record_with_file_spacing(self, name, record, path, *, write):
        """*record*, its ``spacing`` replaced by the file's ``user`` entry
        for this stack when there is one (Step 3 or an earlier Step 4 run
        set it; a differing file entry wins, logged ``[STEP4-SPACING]``).
        With *write*, the record's own ``user`` spacing is copied into the
        file when the file has no ``user`` entry (the first load after the
        file existed).  The adopt passes ``write=False``: it only reads."""
        from ..data_preparation import _voxel_spacing as VS
        key = self._spacing_file_key(path)
        if not key or not self.working_dir:
            return record
        try:
            # the read-only adopt never renames a corrupt file (audit M3)
            entries = VS.read_patch_spacing_file(
                self.working_dir, quarantine=bool(write))
            entry = VS.find_patch_spacing(entries, key)
        except Exception as exc:      # noqa: BLE001 -- a decoration
            print(f"[STEP4-SPACING] {name}: file not read: {exc!r}")
            return record
        own = record.get("spacing") if isinstance(record, dict) else None
        own_user = (isinstance(own, dict) and own.get("source") == "user"
                    and VS._positive(own.get("zyx") or ()) is not None)
        if entry is not None and entry["source"] == "user":
            file_sp = {"zyx": list(entry["zyx"]), "source": "user"}
            if own_user and not S4M.same_spacing(own, file_sp):
                line = (f"[STEP4-SPACING] {name}: the file's "
                        f"{file_sp['zyx']} wins over the record's "
                        f"{own.get('zyx')}")
                print(line)
                try:
                    self.ui.log_output.append(line)
                except Exception:      # noqa: BLE001 -- a log, never a crash
                    pass
            out = dict(record)
            out["spacing"] = file_sp
            return out
        if own_user and write:
            self._write_spacing_file(name, path, own)
        return record

    def _write_spacing_file(self, name, path, spacing) -> bool:
        """Write a resolved spacing to the shared file (a ``user`` value is
        never replaced by ``step1`` / ``file``).  ``False`` when it could
        not be written.  Never raises."""
        if not isinstance(spacing, dict) or not self.working_dir:
            return True
        key = self._spacing_file_key(path)
        if not key:
            return True
        try:
            from ..data_preparation import _voxel_spacing as VS
            VS.write_patch_spacing_entry(
                self.working_dir, key, spacing.get("zyx"),
                spacing.get("source"))
            return True
        except Exception as exc:      # noqa: BLE001 -- the record still holds it
            print(f"[STEP4-SPACING] {name}: file not written: {exc!r}")
            return False

    def _reread_spacing_from_file(self) -> bool:
        """Put the shared file's ``user`` sizes onto the records (Step 3 may
        have set one while Step 4 was away; audit H3).  Reads only: no TIFF
        is opened and nothing is written.  ``True`` when a record changed."""
        if not self.working_dir or getattr(self, "_adopt_only", False):
            return False
        changed = False
        for name in list(self.state_manager.get_all_stack_statuses()):
            rec = self.state_manager.get_record(name)
            path = rec.get("source_stack_path")
            if not path:
                continue
            new = self._record_with_file_spacing(name, rec, path,
                                                 write=False)
            if new is rec or S4M.same_spacing(rec.get("spacing"),
                                              new.get("spacing")):
                continue
            self.state_manager.update_record(name, spacing=new["spacing"])
            changed = True
        return changed

    def _resolve_spacing_fields(self, name, record, path, step1_state):
        """``spacing`` and ``origin`` for a scanned stack.

        A user value is never overwritten; a changed value on a measured
        stack is said in the log (``measured_spacing`` keeps what Measure
        used, G1-2)."""
        original = record               # M8: `old` is what the record held
        record = self._record_with_file_spacing(name, record, path,
                                                write=True)
        spacing = S4M.resolve_stack_spacing(
            record, step1_state, path, self.working_dir)
        self._write_spacing_file(name, path, spacing)
        in_step1 = False
        if step1_state and self.working_dir:
            try:
                from napari_epicycle.data_preparation._state import (
                    voxel_size_for_patch_path,
                )
                rel = os.path.relpath(path, self.working_dir)
                in_step1 = (not rel.startswith("..")
                            and voxel_size_for_patch_path(
                                step1_state, rel) is not None)
            except (OSError, ValueError):
                in_step1 = False
        fields = {"spacing": spacing,
                  "origin": "step1" if in_step1 else "folder"}
        old = original.get("spacing")
        if not S4M.same_spacing(old, spacing):
            print(f"[STEP4-SPACING] {name}: {old} -> {spacing}")
            measured = record.get("measured_spacing")
            if measured is not None and not S4M.same_spacing(measured,
                                                             spacing):
                self.ui.log_output.append(
                    f"[spacing] {name}: {S4T.MT_SPACING_CHANGED}")
        return fields

    def _rescan_stacks(self, persist: bool = True):
        """List every stack: Step 1's patch registry, then the added
        folders.  With *persist*, register each one and
        fill its record (origin, shape, axes, channels, spacing); an adopt
        (``persist=False``) registers nothing and opens no TIFF."""
        step1_state = S4M.read_step1_state(self.working_dir)
        sources = S4R.list_stack_sources(
            self.working_dir, step1_state,
            getattr(self, "_stack_folders", []) or [])
        self._sources = sources
        # An adopt registers nothing and writes nothing (I12); the rows
        # still show what can be known without a write: Step 1's voxel
        # size (no TIFF tag read) and the axes, shape and dtype from
        # each TIFF's HEADER, read on the main thread (about 4.6 ms a
        # stack on the test workspace).
        self._transient_meta = {}
        if not persist:
            for e in sources:
                if e["conflict"]:
                    continue
                meta = {"source_stack_path": e["path"], "origin": e["origin"],
                        "spacing": S4M.resolve_stack_spacing(
                            self._record_with_file_spacing(
                                e["name"],
                                self.state_manager.get_record(e["name"]),
                                e["path"], write=False),
                            step1_state, e["path"], self.working_dir,
                            read_file=False)}
                info = S4M.resolve_stack_axes(e["path"])
                if info is not None:
                    meta.update(stack_shape=info["shape"], axes=info["axes"],
                                n_channels=info["n_channels"],
                                axes_how=info.get("axes_how"),
                                dtype=info.get("dtype"))
                self._transient_meta[e["name"]] = meta
        if persist and sources:
            names = [e["name"] for e in sources if not e["conflict"]]
            self.state_manager.register_stacks(names)
            for e in sources:
                if e["conflict"]:
                    continue
                self._fill_record(e, step1_state)
        self.ui.log_output.append(
            S4T.LOG_STACKS_LISTED.format(n=len(sources)))
        self._thumbs_pending = True       # one worker, started just below
        self._refresh_stacks()
        self._start_thumbs()

    def _fill_record(self, entry, step1_state):
        """The per-stack metadata of a listed stack: a TIFF
        header is opened only when the shape or axes are not on the record,
        or the file is a different one."""
        name = entry["name"]
        path = entry["path"]
        record = self.state_manager.get_record(name)
        fields = {"source_stack_path": path}
        same_file = S4R.norm_path(record.get("source_stack_path") or "") \
            == S4R.norm_path(path)
        if not same_file or not record.get("axes") or \
                not record.get("stack_shape") or not record.get("dtype"):
            # N5: an older record gains axes_how / dtype from its header
            info = S4M.resolve_stack_axes(path)
            if info is not None:
                fields.update(stack_shape=info["shape"], axes=info["axes"],
                              n_channels=info["n_channels"],
                              axes_how=info.get("axes_how"),
                              dtype=info.get("dtype"))
        fields.update(self._resolve_spacing_fields(
            name, record, path, step1_state))
        fields["origin"] = entry["origin"]
        self.state_manager.set_stack_status(
            name, self.state_manager.get_stack_status(name), **fields)





    # ------------------------------------------------------------------
    # Run handlers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Prediction cache (adapted from Step 3)
    # ------------------------------------------------------------------

    def _prediction_cache_dir(self) -> Path:
        return Path(self.working_dir) / "artifacts" / "step4" / "prediction_cache"

    def _persist_prediction_cache(self, views: dict, stack_name: str,
                                  meta: Optional[dict] = None):
        """Save prediction flows/probs/masks to disk (F7, F24).

        Keyed by the FILE NAME.  The old meta is deleted FIRST and the new
        ``{name}_cache_meta.json`` written LAST, so a half-written cache has
        no meta and Reconstruct refuses it.
        """
        import json as _json

        cache_dir = self._prediction_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        meta_path = S4M.cache_meta_path(cache_dir, stack_name)
        try:
            meta_path.unlink()
        except FileNotFoundError:
            pass
        # Audit D11: all nine name-keyed arrays go first, so a view this
        # Predict does not write cannot survive from the last one.
        for v in S4M.VIEWS:
            for k in S4M.CACHE_KINDS:
                try:
                    S4M.cache_array_path(cache_dir, stack_name, v, k).unlink()
                except FileNotFoundError:
                    pass
        for view_name, view_data in views.items():
            flows, probs = view_data[0], view_data[1]
            masks = view_data[2] if len(view_data) > 2 else None
            np.save(
                S4M.cache_array_path(cache_dir, stack_name, view_name, "flows"),
                flows.astype(np.float16),
            )
            np.save(
                S4M.cache_array_path(cache_dir, stack_name, view_name, "probs"),
                probs.astype(np.float16),
            )
            if masks is not None:
                np.save(
                    S4M.cache_array_path(cache_dir, stack_name, view_name,
                                         "masks"),
                    masks.astype(np.int32),
                )
        payload = dict(meta or {})
        payload["stack"] = stack_name
        payload["views"] = sorted(views)
        tmp = meta_path.with_name(meta_path.name + ".tmp")
        tmp.write_text(_json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(str(tmp), str(meta_path))
        self.ui.log_output.append(f"[predict] Cache saved: {stack_name}")

    def _try_restore_prediction_cache(self, stack_name: str):
        """Load cached predictions. Returns views dict or None."""
        cache_dir = self._prediction_cache_dir()
        stem = Path(stack_name).stem
        views = {}
        for view_name in ("xy", "xz", "yz"):
            flows_path = cache_dir / f"{stem}_{view_name}_flows.npy"
            probs_path = cache_dir / f"{stem}_{view_name}_probs.npy"
            masks_path = cache_dir / f"{stem}_{view_name}_masks.npy"
            if flows_path.exists() and probs_path.exists():
                flows = np.load(str(flows_path)).astype(np.float32)
                probs = np.load(str(probs_path)).astype(np.float32)
                masks = (
                    np.load(str(masks_path)).astype(np.int32)
                    if masks_path.exists() else None
                )
                views[view_name] = (flows, probs, masks)
        if len(views) == 3:
            return views
        return None

    def _clear_prediction_cache_for(self, stack_name: str):
        """Remove cached prediction files for a stack, by EXACT names (F7).

        Today's ``{stem}_*`` glob deleted ``stack_2``'s cache with
        ``stack``'s.  The name-keyed files and the meta go; a legacy
        stem-keyed cache goes only when no other stack shares the stem.
        """
        cache_dir = self._prediction_cache_dir()
        if not cache_dir.is_dir():
            return
        targets = [S4M.cache_meta_path(cache_dir, stack_name)]
        for v in S4M.VIEWS:
            for k in S4M.CACHE_KINDS:
                targets.append(S4M.cache_array_path(cache_dir, stack_name, v, k))
        stem = Path(stack_name).stem
        others = [n for n in self.state_manager.get_all_stack_statuses()
                  if n != stack_name and Path(n).stem == stem]
        if not others:
            for v in S4M.VIEWS:
                for k in S4M.CACHE_KINDS:
                    targets.append(
                        S4M.legacy_cache_array_path(cache_dir, stack_name, v, k))
        count = 0
        for f in targets:
            try:
                f.unlink()
                count += 1
            except FileNotFoundError:
                pass
            except OSError as exc:
                self.ui.log_output.append(f"[WARN] {f.name}: {exc}")
        if count:
            self.ui.log_output.append(
                f"[predict] Cleared cache for {stack_name} ({count} files)"
            )

    def _on_stack_selected(self):
        """The highlighted ② row drives View and the cell-width pixel line; a click no longer previews the stack."""
        self._update_cell_width_line()
        self._refresh_segment_gates()

    def _highlighted_row(self):
        lst = getattr(self.ui, "stacks_list", None)
        items = lst.selectedItems() if lst is not None else []
        if not items:
            return None
        return items[0].data(Qt.UserRole + 2) or None

    def _on_stack_eye(self, name: str):
        """The eye: show or hide this stack in the viewer.
        Live under Predict / Reconstruct only: it reads a TIFF on
        the main thread and adds layers."""
        if not name or self.viewer is None:
            return
        if _step4_is_locked(self) and not (
                getattr(self, "_predict_running", False)
                or getattr(self, "_reconstruct_running", False)):
            print(f"[STEP4-LOCK] eye of {name} refused while "
                  f"{self._running_flag_names()}")
            return
        short = Path(name).stem
        if len(short) > 30:
            short = short[:27] + "..."
        shown = [layer for layer in list(self.viewer.layers)
                 if self._PREVIEW_TAG in layer.name
                 and layer.name.startswith(
                     STEP4_LAYER_PREFIX + short + " ")]
        if getattr(self, "_preview_stack", None) == name:
            # the preview's own layers, even after the user renamed one
            live = list(self.viewer.layers)
            for lyr in getattr(self, "_preview_layers", None) or []:
                if any(lyr is x for x in live) and not any(
                        lyr is y for y in shown):
                    shown.append(lyr)
        if shown:
            for layer in shown:
                try:
                    layer.visible = False
                    self.viewer.layers.remove(layer)
                except Exception:      # noqa: BLE001
                    pass
            self._refresh_channel_combos()   # ② names: the preview went
            return
        self._preview_selected_stack(name, self.state_manager.get_record(name))

    _PREVIEW_TAG = "[preview]"

    @staticmethod
    def _smart_load_image(raw: np.ndarray) -> dict:
        """Auto-infer channel layout from array shape.

        Handles 3D (ZYX), 4D single-channel (squeeze), and 4D multi-channel.

        Returns dict:
            data       : np.ndarray — 3D or 4D array ready for viewer
            n_channels : int — number of channels (1 for 3D/squeezed)
            axes       : str — "ZYX", "CZYX", or "ZCYX"
            ch_axis    : int or None — channel axis for 4D, None for 3D
        """
        if raw.ndim == 3:
            return {"data": raw, "n_channels": 1, "axes": "ZYX", "ch_axis": None}

        if raw.ndim == 4:
            # Detect channel axis: the small dimension (<=10) that's smaller
            # than its neighbor
            if raw.shape[0] <= 10 and raw.shape[0] < raw.shape[1]:
                ch_axis, axes = 0, "CZYX"
                n_ch = raw.shape[0]
            elif raw.shape[1] <= 10 and raw.shape[1] < raw.shape[0]:
                ch_axis, axes = 1, "ZCYX"
                n_ch = raw.shape[1]
            else:
                # Ambiguous — default CZYX
                ch_axis, axes = 0, "CZYX"
                n_ch = raw.shape[0]

            # Single-channel 4D → squeeze to 3D
            if n_ch == 1:
                return {
                    "data": raw.squeeze(axis=ch_axis),
                    "n_channels": 1,
                    "axes": "ZYX",
                    "ch_axis": None,
                }

            return {
                "data": raw, "n_channels": n_ch,
                "axes": axes, "ch_axis": ch_axis,
            }

        # 2D or 5D+ — return as-is
        return {"data": raw, "n_channels": 1, "axes": "", "ch_axis": None}

    def _preview_selected_stack(self, stack_name: str, record: dict):
        """Load the selected stack into the viewer for inspection.

        Safe multi-channel handling: uses channel_axis to split correctly.
        Removes previous preview layers before adding new ones.
        Layer names include the stack name for identification.
        """
        source_path = record.get("source_stack_path", "")
        if not source_path or not Path(source_path).exists():
            return

        if self.viewer is None:
            return

        # Remove previous preview layers (Step 4's own, F15)
        to_remove = [
            layer for layer in list(self.viewer.layers)
            if self._PREVIEW_TAG in layer.name
            and layer.name.startswith(STEP4_LAYER_PREFIX)
        ]
        for layer in to_remove:
            try:
                layer.visible = False
                self.viewer.layers.remove(layer)
            except Exception:
                pass

        # Short display name: strip extension, truncate if long
        short_name = Path(stack_name).stem
        if len(short_name) > 30:
            short_name = short_name[:27] + "..."

        try:
            import tifffile
            raw = tifffile.imread(source_path)
        except Exception as exc:
            self.ui.log_output.append(
                f"[preview] Failed to load {stack_name}: {exc}"
            )
            return

        try:
            info = self._smart_load_image(raw)
            data = info["data"]
            if info["ch_axis"] is not None:
                # Multi-channel 4D
                n_ch = info["n_channels"]
                ch_names = [
                    f"{STEP4_LAYER_PREFIX}{short_name} ch{i} "
                    f"{self._PREVIEW_TAG}"
                    for i in range(n_ch)
                ]
                added = self._track_step4_layers(self.viewer.add_image(
                    data, name=ch_names, channel_axis=info["ch_axis"],
                ))
                self._tag_channel_layers(added)
                self._preview_layers = list(added)
                self._preview_stack = stack_name
            else:
                # 3D (or squeezed single-channel 4D)
                added = self._track_step4_layers(self.viewer.add_image(
                    data, name=f"{STEP4_LAYER_PREFIX}{short_name} "
                               f"{self._PREVIEW_TAG}",
                ))
                self._tag_channel_layers([added])
                self._preview_layers = [added]
                self._preview_stack = stack_name
        except Exception as exc:
            self.ui.log_output.append(
                f"[preview] Error displaying {stack_name}: {exc}"
            )
        self._refresh_channel_combos()      # ② names follow the preview

    def _refresh_stacks(self):
        """Rebuild every row record and push the three lists, ①'s summary,
        the run row, the ticked line and the gates."""
        statuses = self.state_manager.get_all_stack_statuses()
        # Also scan for manifests not yet in state
        if self.working_dir and not getattr(self, "_adopt_only", False):
            for manifest_path in self.run_store.list_manifests(self.working_dir):
                payload = self.run_store.load_manifest(manifest_path)
                source = payload.get("source_stack_path", "")
                stack_name = Path(source).name if source else payload.get("run_id", "")
                if stack_name and stack_name not in statuses:
                    self.state_manager.register_stacks([stack_name])
                    extra = {}
                    if isinstance(payload.get("made_with"), dict):
                        extra["made_with"] = payload["made_with"]   # N6
                    self.state_manager.set_stack_status(
                        stack_name, "reconstructed",
                        labels_path=payload.get("labels_path", ""),
                        source_stack_path=source, **extra,
                    )
            statuses = self.state_manager.get_all_stack_statuses()
        rows = self._build_rows()
        self._rows = rows
        stored = self.state_manager.get_ticked()
        # a greyed row shows its stored tick, so it can be
        # taken off (it is never tickable while greyed)
        listed = {r["name"] for r in rows}
        ticked = [n for n in stored if n in listed]
        self.ui.refresh_stack_lists(rows, ticked=ticked)
        self._push_summaries(rows, stored)
        self._refresh_channel_combos()
        self._refresh_segment_gates()
        self._refresh_mt_panel()
        self._restart_thumbs_if_labels_changed()
        self._restart_tier_counts_if_csv_changed()

    def _build_rows(self):
        """The row records: every listed source, then any record not
        listed (a stack known from a manifest or an older folder)."""
        entries = list(getattr(self, "_sources", []) or [])
        listed = {e["name"] for e in entries if not e["conflict"]}
        for name in self.state_manager.get_all_stack_statuses():
            if Path(name).suffix.lower() not in S4R.STACK_SUFFIXES:
                continue          # F12: an old .npy record is not shown
            if name not in listed:
                rec = self.state_manager.get_record(name)
                entries.append({"name": name,
                                "path": rec.get("source_stack_path") or "",
                                "origin": rec.get("origin") or "folder",
                                "conflict": False})
                listed.add(name)
        ticked = set(self.state_manager.get_ticked())
        running = getattr(self, "_running", {}) or {}
        snap = self._active_model
        needs = self._channels_needed()
        cache_dir = self._prediction_cache_dir() if self.working_dir else None
        rows = []
        for e in entries:
            rec = {} if e["conflict"] else self.state_manager.get_record(
                e["name"])
            extra = (getattr(self, "_transient_meta", {}) or {}).get(
                e["name"])
            if extra:
                rec = dict(extra, **{k: v for k, v in rec.items()
                                     if v is not None})
            complete = False
            if cache_dir is not None and not e["conflict"]:
                complete = bool(S4M.find_prediction_cache(
                    cache_dir, e["name"])["complete"])
            rows.append(S4R.build_row(
                e, rec, active_model=snap, needs=needs,
                ticked=e["name"] in ticked,
                running=running.get(e["name"]),
                cache_complete=complete))
        return rows

    def _push_summaries(self, rows, ticked):
        main, amber = S4R.session_summary(rows)
        counts = S4R.status_counts(rows)
        if not rows:
            note = S4T.SET_SPACING_NEEDS_STACK
        elif not ticked:
            note = S4T.SET_SPACING_NEEDS_TICK
        else:
            note = ""
        self.ui.show_session_summary(main, amber, counts, len(ticked), note)
        self.ui.show_run_row(len(rows))
        if getattr(self, "_running", None):
            self.ui.show_ticked_line(S4T.STACKS_RUNNING_HINT, "")
        else:
            dim, amb = S4R.ticked_line(rows, ticked)
            self.ui.show_ticked_line(dim, amb)

    def _on_ticks_changed(self, names):
        """A tick in ① or ②: ``session.ticked`` is written and saved
        one event-loop turn later (debounced)."""
        if getattr(self, "_adopt_only", False):
            # The first tick after a read-only adopt is a user action that
            # saves: load in write mode, register, then keep the tick.
            self._ensure_writable_state()
            self.state_manager.set_ticked(self._with_untickable(names))
            self._rescan_stacks(persist=True)
            self._persist_state()
            return
        names = self._with_untickable(names)
        self.state_manager.set_ticked(names)
        rows = [dict(r, ticked=r["name"] in set(names))
                for r in getattr(self, "_rows", [])]
        self._rows = rows
        self._push_summaries(rows, list(names))
        self._refresh_channel_combos()
        self._refresh_segment_gates()
        if not getattr(self, "_tick_save_pending", False):
            self._tick_save_pending = True
            from qtpy.QtCore import QTimer
            QTimer.singleShot(0, self._save_ticks)

    def _with_untickable(self, names):
        """The ticks the lists report.  A stack that cannot run keeps its
        stored tick (F2) and SHOWS it, so SEG_TICKED_SKIPPED counts it and
        the user can take it off: its
        checked box is reported like any other, and an untick drops it.
        Only a stored tick of a stack that is not listed is kept as is."""
        listed = {r["name"] for r in getattr(self, "_rows", []) or []}
        keep = [n for n in self.state_manager.get_ticked()
                if n not in listed and n not in names]
        return list(names) + keep

    def _save_ticks(self):
        self._tick_save_pending = False
        self._persist_state()

    def _get_model_paths(self):
        """Lateral + axial model paths from the hidden state widgets.  Mode 0 "Auto from Workspace" is gone: it read a
        ``model_pack.json`` no bundle writes (package B1)."""
        mode = self.ui.model_source_combo.currentIndex()
        if mode == 2:  # Pretrained
            name = self.ui.pretrained_combo.currentText()
            return name, name  # Cellpose uses pretrained name as path
        return (
            self.ui.lateral_model_path.text().strip(),
            self.ui.axial_model_path.text().strip(),
        )

    def _model_snapshot(self):
        """The active model's snapshot: ``session.active_model``
        when it names the files the hidden widgets hold, else one built
        from the widgets.  Taken at LAUNCH and carried to the finish."""
        try:
            lateral, axial = self._get_model_paths()
        except Exception as exc:      # noqa: BLE001
            print(f"[STEP4-MODEL] model paths unreadable: {exc}")
            return None
        active = getattr(self, "_active_model", None)
        if active and str(active.get("lateral")) == lateral and \
                str(active.get("axial")) == (axial or lateral):
            import copy as _copy
            return _copy.deepcopy(active)
        mode = self.ui.model_source_combo.currentIndex()
        kind = "pretrained" if mode == 2 else "external"
        return S4M.model_snapshot(lateral, axial, kind=kind,
                                  workspace=self.working_dir)

    def _model_files_missing(self, snapshot) -> bool:
        """A model given by file whose files are not there (G1-12)."""
        return S4M.model_files_missing(snapshot)

    # ------------------------------------------------------------------
    # The model card and Change…
    # ------------------------------------------------------------------

    def _set_active_model(self, snapshot, *, save: bool = True):
        """Make *snapshot* the active model: fill the hidden state widgets,
        refresh the card, re-derive every row's chip and channel warning."""
        self._active_model = snapshot
        ui = self.ui
        if snapshot is None:
            ui.lateral_model_path.setText("")
            ui.axial_model_path.setText("")
            ui.model_source_combo.setCurrentIndex(1)
        elif snapshot.get("kind") == "pretrained":
            ui.model_source_combo.setCurrentIndex(2)
            idx = ui.pretrained_combo.findText(str(snapshot.get("lateral")))
            if idx < 0:
                ui.pretrained_combo.addItem(str(snapshot.get("lateral")))
                idx = ui.pretrained_combo.count() - 1
            ui.pretrained_combo.setCurrentIndex(idx)
            ui.lateral_model_path.setText(str(snapshot.get("lateral")))
            ui.axial_model_path.setText(str(snapshot.get("axial")))
        else:
            ui.model_source_combo.setCurrentIndex(1)
            ui.lateral_model_path.setText(str(snapshot.get("lateral") or ""))
            ui.axial_model_path.setText(str(snapshot.get("axial") or ""))
        self._trained_codes = self._read_trained(snapshot)
        self._refresh_model_card()
        idx = S4M.trained_channel_indices(self._trained_codes)
        if idx is not None and hasattr(ui, "set_channel_choice"):
            self._refresh_channel_combos()
            ui.set_channel_choice(ui.chan1_combo, idx[0])
            ui.set_channel_choice(ui.chan2_combo, idx[1])
        if save:
            self.state_manager.set_active_model(snapshot)
            self._persist_state()
            self._refresh_stacks()

    @staticmethod
    def _read_trained(snapshot):
        if not snapshot or snapshot.get("kind") == "pretrained":
            return None
        return S4M.read_trained_channels(snapshot.get("lateral"))

    def _refresh_model_card(self):
        snap = self._active_model
        text, colour = S4M.model_indicator(snap)
        missing = S4M.model_files_missing(snap)
        trained = (S4T.MODEL_MISSING if missing
                   else S4M.trained_channels_text(
                       getattr(self, "_trained_codes", None)))
        if hasattr(self.ui, "show_model_card"):
            self.ui.show_model_card(text, colour, trained, missing=missing)

    def _on_model_change_clicked(self):
        """Change…: Step 3's picker with Step 4's rows."""
        if self._refuse_while_locked("Change model"):
            return
        if not self.working_dir:
            return
        from ..fusion_reconstruction.refinement_registry_dialog import (
            RefinementRegistryDialog,
        )
        from ..fusion_reconstruction._run_management import (
            list_cross_run_refined_models,
        )
        rows = S4M.step4_model_rows(list_cross_run_refined_models(
            self.working_dir, trained_only=True))
        try:
            from ..hitl_training_annotation._run_management import (
                list_run_model_sources,
            )
            step2 = [s for s in list_run_model_sources(self.working_dir)
                     if s.get("bucket")]
        except Exception as exc:      # noqa: BLE001
            print(f"[STEP4-MODEL] Step 2 models not listed: {exc}")
            step2 = []
        print(f"[STEP4-MODEL] picker: {len(rows)} refined, "
              f"{len(step2)} Step 2")
        RefinementRegistryDialog.show_dialog(
            parent=self.ui, rows=rows, on_use=self._on_model_use,
            on_open=self._on_model_open, on_init=None,
            step2_models=step2, show_init=False)

    def _on_model_use(self, row):
        """Use in the picker: the row's model becomes
        ``session.active_model``."""
        self._ensure_writable_state()
        snap = S4M.snapshot_from_row(row, self._active_model,
                                     self.working_dir)
        if snap is None:
            print(f"[STEP4-MODEL] row gave no model: {row.get('_kind')}")
            return
        self.ui.log_output.append(S4T.LOG_MODEL_SET.format(
            model=S4M.model_short_label(snap)))
        self._set_active_model(snap)

    def _on_model_open(self, row):
        """Open folder in the picker: the bundle (or weights) folder."""
        folder = row.get("bundle_dir") or ""
        if not folder and row.get("weights_path"):
            folder = str(Path(row["weights_path"]).parent)
        if folder and Path(folder).is_dir():
            try:
                os.startfile(folder)  # noqa: S606 -- Windows open-folder
            except Exception as exc:      # noqa: BLE001
                print(f"[STEP4-MODEL] open folder failed: {exc}")

    def _collect_seg_config(self, record=None, px=None) -> SegmentationConfig:
        """SegmentationConfig for one stack.

        Channel dropdowns use 0-based array indices.  Cellpose expects
        its own convention: 0=gray, 1=first channel, 2=second, 3=third.
        Convert: array_index → cellpose_code = array_index + 1.
        Nuclear "none" at combo index 0 maps to Cellpose 0.  The diameter
        is the stack's cell width in pixels (``px``), or Auto.
        """
        ui = self.ui
        chan1_arr = max(ui.chan1_combo.currentIndex(), 0)
        chan1_cp = chan1_arr + 1  # 0→1, 1→2, 2→3
        chan2_idx = max(ui.chan2_combo.currentIndex(), 0)
        chan2_cp = chan2_idx  # 0→0(none), 1→1(ch0), 2→2(ch1), 3→3(ch2)
        return SegmentationConfig(
            channels=[chan1_cp, chan2_cp],
            normalize=ui.normalize_cb.isChecked(),
            # F25: Cellpose's rescale is a resize FACTOR; False became 0 and
            # crashed.  Off -> None (the model's own trained cell size);
            # on -> today's True (factor 1.0).
            rescale=True if ui.rescale_cb.isChecked() else None,
            cellprob_threshold=ui.cellprob_sb.value(),
            flow_threshold=ui.flow_thresh_sb.value(),
            diameter=None if not px else float(px),
            batch_size=ui.batch_size_sb.value(),
            compute_masks=False,        # D-d
        )

    def _stack_px(self, record):
        """``(px or None, in_voxels)``: the cell width in pixels from the
        stack's own spacing, or the pixel number shown."""
        um = self.ui.cell_width_um_sb.value()
        if not um:
            return None, False
        px = S4M.cell_width_px(um, record.get("spacing"))
        if px is not None:
            return px, False
        shown = getattr(self, "_shown_px", None)
        return (shown if shown else max(1, int(round(um)))), True

    def _collect_recon_config(self):
        """Build ReconstructionConfig from Run panel UI widgets."""
        from packages.epicycle_reconstruction.config import ReconstructionConfig
        ui = self.ui
        prob_thresh_val = ui.ccp_prob_thresh_sb.value()
        prob_thresh = None if prob_thresh_val < 0 else prob_thresh_val
        # R0: the item's data "direct" is persisted, never its
        # text "Direct"; an item without data falls back to its text, lower
        method = (ui.method_combo.currentData()
                  or ui.method_combo.currentText().lower())
        return ReconstructionConfig(
            method=method,
            skip_postprocess=ui.skip_postprocess_cb.isChecked(),
            n_cpu=ui.n_cpu_sb.value(),
            pre_smooth_probs=ui.ps_smooth_probs_cb.isChecked(),
            pre_smooth_flows=ui.ps_smooth_flows_cb.isChecked(),
            pre_smooth_sigma_perp=ui.ps_sigma_perp_sb.value(),
            pre_smooth_sigma_inplane=ui.ps_sigma_inplane_sb.value(),
            flow_mask_threshold=ui.ps_mask_thresh_sb.value(),
            flow_soft_weighting=ui.ps_soft_weight_cb.isChecked(),
            combine_cell_gradients={
                "ksize": ui.ccg_ksize_sb.value(),
                "alpha": ui.ccg_alpha_sb.value(),
                "smooth_sigma": ui.ccg_smooth_sigma_sb.value(),
                "post_sigma": ui.ccg_post_sigma_sb.value(),
            },
            combine_cell_probs={
                "ksize": ui.ccp_ksize_sb.value(),
                "alpha": ui.ccp_alpha_sb.value(),
                "smooth_sigma": ui.ccp_smooth_sigma_sb.value(),
                "threshold_level": ui.ccp_threshold_level_sb.value(),
                "threshold_n_levels": ui.ccp_threshold_n_levels_sb.value(),
                "prob_thresh": prob_thresh,
                "min_prob_thresh": ui.ccp_min_prob_thresh_sb.value(),
                "apply_one_d_p_thresh": ui.ccp_apply_one_d_p_cb.isChecked(),
            },
            gradient_descent={
                "gradient_decay": ui.gd_decay_sb.value(),
                "n_iter": ui.gd_iters_sb.value(),
                "momenta": ui.gd_momenta_sb.value(),
                "do_mp": ui.do_mp_cb.isChecked(),
                "renorm_gradient": ui.gd_renorm_cb.isChecked(),
                "tile_shape": [
                    ui.gd_tile_z_sb.value(),
                    ui.gd_tile_y_sb.value(),
                    ui.gd_tile_x_sb.value(),
                ],
                "tile_aspect": [1, 2, 2],
                "tile_overlap_ratio": ui.gd_tile_overlap_sb.value(),
            },
            postprocess_binary={
                "binary_closing": ui.post_closing_sb.value(),
                "binary_dilation": ui.post_dilation_sb.value(),
                "binary_fill_holes": ui.post_fill_cb.isChecked(),
                "remove_small_objects": ui.post_small_obj_sb.value(),
                "extra_erode": ui.post_extra_erode_sb.value(),
            },
            connected_component={
                "min_area": ui.cc_min_area_sb.value(),
                "thresh_factor": ui.cc_thresh_factor_sb.value(),
                "smooth_sigma": ui.cc_smooth_sigma_sb.value(),
            },
            size_filters={
                "min_size": ui.sf_min_size_sb.value(),
                "max_size": None if ui.sf_max_size_sb.value() < 0 else ui.sf_max_size_sb.value(),
                "max_size_factor": ui.sf_max_size_factor_sb.value(),
                "do_stats_filter": ui.sf_do_stats_cb.isChecked(),
            },
        )

    def _run_prediction(self):
        """Predict Selected: every ticked stack that can run, each
        with its own axes, channels and cell width; the model snapshot is
        taken now and carried."""
        if self._refuse_while_locked("Predict"):
            return
        if not self.working_dir:
            return
        self._ensure_writable_state()
        if getattr(self, "_adopt_only", False) or not self._rows:
            self._rescan_stacks(persist=True)
        snapshot = self._model_snapshot()
        if snapshot is None:
            self._refresh_segment_gates()
            return
        if self._model_files_missing(snapshot):
            self.ui.log_output.append(f"[predict] {S4T.MODEL_MISSING}")
            self._refresh_model_card()
            self._refresh_segment_gates()
            return
        c1, c2 = self.ui.chosen_channels()
        if c1 < 0:
            self.ui.show_seg_status(S4T.SEG_PICK_CHANNEL)
            return
        lateral_path, axial_path = self._get_model_paths()
        ticked = set(self._stored_ticked())
        rows = [r for r in self._rows if r["name"] in ticked]
        need = max(c1, c2 if c2 is not None else 0) + 1
        run_rows = []
        for r in rows:
            if not r["can_run"]:
                self.ui.log_output.append(S4T.LOG_STACK_SKIPPED.format(
                    stack=r["name"], why=S4T.ROW_WARN_SAME_NAME
                    if r["conflict"] else S4T.SEG_NONE_RUNNABLE))
                continue
            if not r.get("axes"):
                # A1-10: never guess an axis order for a stack
                self.ui.log_output.append(S4T.LOG_STACK_SKIPPED.format(
                    stack=r["name"], why=S4T.LOG_NO_AXES))
                continue
            if r["n_channels"] is not None and r["n_channels"] < need:
                self.ui.log_output.append(S4T.LOG_STACK_SKIPPED.format(
                    stack=r["name"], why=S4T.ROW_WARN_CHANNELS.format(
                        n=need, k=r["n_channels"])))
                continue
            run_rows.append(r)
        if not run_rows:
            self.ui.show_seg_status(S4T.SEG_NONE_RUNNABLE)
            return
        over_3d = [r["name"] for r in run_rows if r["has_3d"]]
        if over_3d:
            from ..ui_shell.theme import themed_confirm
            text = "\n\n".join(S4T.PREDICT_OVER_3D_CONFIRM.format(stack=n)
                               for n in over_3d)
            if not themed_confirm(self.ui, S4T.PREDICT_OVER_3D_TITLE, text,
                                  confirm_text=S4T.PREDICT_OVER_3D_BTN,
                                  cancel_text=S4T.CANCEL_BTN,
                                  accent="green"):
                return

        stack_configs = []
        total_slices = 0
        recon_now = self._collect_recon_config()
        for r in run_rows:
            record = self.state_manager.get_record(r["name"])
            px, in_voxels = self._stack_px(record)
            seg_config = self._collect_seg_config(record, px)
            axes = S4M.stack_axes(record)
            if record.get("spacing") is None:
                self.ui.log_output.append(
                    S4T.LOG_RUNS_IN_VOXELS.format(stack=r["name"]))
            if in_voxels and px:
                self.ui.log_output.append(S4T.LOG_WIDTH_IN_PIXELS.format(
                    stack=r["name"], px=px))
            zyx = r.get("zyx") or (0, 0, 0)
            total_slices += int(sum(zyx)) or 300
            source = record.get("source_stack_path") or r["path"]
            run_snap = S4M.run_snapshot(
                snapshot, record=record, source_path=source,
                seg_config=seg_config, recon_config=recon_now,
                cell_width_um=self.ui.cell_width_um_sb.value(),
                diameter_px=px)
            stack_configs.append({
                "stack_name": r["name"],
                "source_stack_path": source,
                "lateral_model_path": lateral_path,
                "axial_model_path": axial_path,
                "seg_config": seg_config,
                "axes": axes,
                "model_snapshot": run_snap,
                "cache_meta": {
                    "model": run_snap,
                    "channels": list(getattr(seg_config, "channels", [])
                                     or []),
                    "axes": axes,
                    "diameter_px": getattr(seg_config, "diameter", None),
                    "px_in_voxels": bool(in_voxels and px),
                    "rescale_factor": run_snap["seg"]["rescale_factor"],
                },
            })
        self.ui.log_output.append(S4T.LOG_PREDICT_START.format(
            n=len(stack_configs), model=S4M.model_indicator(snapshot)[0]))
        self._begin_run("predict", stack_configs, total_slices)
        self.ui.run_progress.setRange(0, max(total_slices, 1))
        self.ui.run_progress.setValue(0)
        self.ui.show_seg_status(TH.PIPE_SEGMENTING_START.format(
            n=len(stack_configs), units=TH.pipe_units("stack"),
            slices=total_slices), active=True)
        if self._launch_prediction_worker(stack_configs) is None:
            self._restore_run_buttons()               # re-audit L-d

    def _begin_run(self, kind, stack_configs, total_slices=0):
        import time as _time
        self._run_kind = kind
        self._run_total = len(stack_configs)
        self._run_done = 0
        self._run_failures = 0
        self._running = {}
        self._seg_total_slices = total_slices
        self._seg_slices_done = 0
        self._seg_last_view = None
        self._seg_last_view_total = 0
        self._seg_last_stack = None
        self._seg_index = 0
        self._seg_start_time = _time.monotonic()
        self.ui.predict_btn.setEnabled(False)
        self.ui.reconstruct_btn.setEnabled(False)
        self.ui.cancel_btn.setEnabled(True)
        self.ui.run_progress.setFormat("%p%")

    def _launch_prediction_worker(self, stack_configs):
        worker = _StackPredictionWorker(stack_configs)
        worker.log.connect(self.ui.log_output.append)
        worker.stack_started.connect(self._on_stack_started)
        worker.slice_progress.connect(self._on_seg_slice_progress)
        worker.stack_completed.connect(self._on_prediction_completed)
        worker.error.connect(self._on_predict_error)
        worker.all_completed.connect(self._on_all_stacks_completed)
        return self._start_step4_thread(
            worker, "_predict_running", quit_on=("all_completed",),
            attr="_seg", cleanup=self._restore_run_buttons)

    def _on_prediction_completed(self, stack_name, result):
        """Save prediction cache and update the record (F7, G1-13, G1-20).

        ``predicted_with`` is the snapshot the worker carried from launch;
        a stack that already has a 3D result keeps its status (a Predict
        again must not drop a finalized stack out of Export).
        """
        views = result.get("views")
        snapshot = result.get("model_snapshot")
        meta = dict(result.get("cache_meta") or {})
        if snapshot is not None:
            # N1: the worker finished the snapshot (sha1s, versions, the real
            # factor) on ITS copy; the cache meta must carry that one
            meta["model"] = snapshot
            seg = snapshot.get("seg") if isinstance(snapshot, dict) else None
            if isinstance(seg, dict):
                meta["rescale_factor"] = seg.get("rescale_factor")
        if views:
            self._persist_prediction_cache(views, stack_name, meta)
        source = result.get("source_stack_path", "")
        status = self.state_manager.get_stack_status(stack_name)
        keep = status in ("reconstructed", "qc_extracted", "finalized")
        fields = dict(
            source_stack_path=source,
            predicted_with=snapshot,
            source_fingerprint=S4M.file_fingerprint(source) if source else None,
        )
        if meta.get("axes"):
            fields["predicted_axes"] = meta["axes"]
        if keep:
            print(f"[STEP4-PREDICT] {stack_name}: kept status {status}")
            self.state_manager.update_record(stack_name, **fields)
        else:
            self.state_manager.set_stack_status(stack_name, "predicted",
                                                **fields)
        self.state_manager.clear_stack_keys(stack_name, ["failed_stage",
                                                         "failed_at"])
        self._run_done = getattr(self, "_run_done", 0) + 1
        getattr(self, "_running", {}).pop(stack_name, None)
        self._persist_state()
        self._refresh_stacks()

    def _run_reconstruction(self):
        """Reconstruct Selected: every ticked stack with a complete,
        current 2D cache; the worker never re-predicts."""
        if self._refuse_while_locked("Reconstruct"):
            return
        if not self.working_dir:
            return
        self._ensure_writable_state()

        recon_config = self._collect_recon_config()
        lateral_path, axial_path = self._get_model_paths()

        cache_dir = str(self._prediction_cache_dir())
        ticked = set(self.ui.ticked_names())
        stack_configs = []
        for r in self._rows:
            stack_name = r["name"]
            if stack_name not in ticked or not r["can_run"]:
                continue
            record = self.state_manager.get_record(stack_name)
            source_path = record.get("source_stack_path", "")

            # A complete cache: all three views, and its meta for a
            # name-keyed cache (F7, F24).  Read at LAUNCH and carried.
            found = S4M.find_prediction_cache(
                self._prediction_cache_dir(), stack_name)
            if found["kind"] is None:
                self.ui.log_output.append(
                    S4T.LOG_NO_CACHE.format(stack=stack_name))
                continue
            if not found["complete"]:
                self.ui.log_output.append(
                    "[WARN] " + S4T.RECON_PARTIAL_CACHE.format(
                        stack=stack_name))
                continue
            if r.get("cache_stale"):
                self.ui.log_output.append(
                    "[WARN] " + S4T.RECON_STALE_CACHE.format(stack=stack_name))
                continue
            if r.get("changed_since_predict"):
                self.ui.log_output.append(
                    "[WARN] " + S4T.RECON_SOURCE_CHANGED.format(
                        stack=stack_name))
                continue
            if not S4M.stack_axes(record):
                self.ui.log_output.append(S4T.LOG_STACK_SKIPPED.format(
                    stack=stack_name, why=S4T.LOG_NO_AXES))
                continue
            made_with = (found["meta"] or {}).get("model")
            if isinstance(made_with, dict):
                # v2.6: the recon config of THIS Reconstruct
                made_with = dict(made_with,
                                 recon=S4M._plain(recon_config))

            stack_configs.append({
                "stack_name": stack_name,
                "working_dir": self.working_dir,
                "source_stack_path": source_path,
                "lateral_model_path": lateral_path,
                "axial_model_path": axial_path,
                "recon_config": recon_config,
                "prediction_cache_dir": cache_dir,
                "cache_files": found["files"],
                "require_cache": True,
                "made_with": made_with,
                "axes": S4M.stack_axes(record),
            })

        if not stack_configs:
            self.ui.show_seg_status(S4T.SEG_NOTHING_TO_RECONSTRUCT)
            return

        overwrites = [
            (cfg["stack_name"],
             self.state_manager.get_stack_status(cfg["stack_name"]))
            for cfg in stack_configs
            if self.state_manager.get_stack_status(cfg["stack_name"])
            in ("qc_extracted", "finalized")
        ]
        if overwrites:
            from ..ui_shell.theme import STACK_STATUS_WORDS, themed_confirm
            text = "\n\n".join(
                S4T.RECON_OVERWRITES_CONFIRM.format(
                    stack=n, status=STACK_STATUS_WORDS.get(s, s))
                for n, s in overwrites)
            if not themed_confirm(self.ui, S4T.RECON_OVERWRITES_TITLE, text,
                                  confirm_text=S4T.RECON_OVERWRITES_BTN,
                                  cancel_text=S4T.CANCEL_BTN, accent="green"):
                return

        self.ui.log_output.append(
            S4T.LOG_RECON_START.format(n=len(stack_configs)))
        self._begin_run("reconstruct", stack_configs)
        self.ui.run_progress.setRange(0, 0)  # scanning mode (reconstruction)
        self.ui.show_seg_status(TH.PIPE_RECONSTRUCTING_START.format(
            n=len(stack_configs), units=TH.pipe_units("stack")), active=True)

        if self._launch_reconstruction_worker(stack_configs) is None:
            self._restore_run_buttons()               # re-audit L-d

    def _launch_reconstruction_worker(self, stack_configs, pipeline=None):
        # The cache files are carried in the configs; the worker never
        # re-predicts (require_cache, G1-11).
        worker = _StackSegmentationWorker(
            stack_configs, pipeline or ReconstructionPipelineService())
        worker.log.connect(self.ui.log_output.append)
        worker.slice_progress.connect(self._on_seg_slice_progress)
        worker.stack_started.connect(self._on_stack_started)
        worker.stack_completed.connect(self._on_stack_completed)
        worker.error.connect(self._on_recon_error)
        worker.all_completed.connect(self._on_all_stacks_completed)
        return self._start_step4_thread(
            worker, "_reconstruct_running", quit_on=("all_completed",),
            attr="_seg", cleanup=self._restore_run_buttons)

    def _clear_predictions(self, name: str):
        """⋯ Clear 2D predictions…: a red confirm, then the
        cache goes and a predicted (or failed-at-Predict) stack goes back to
        unprocessed."""
        if self._refuse_while_locked("Clear 2D predictions"):
            return
        record = self.state_manager.get_record(name)
        if record.get("status") in S4R.HAS_3D:
            return
        from ..ui_shell.theme import themed_confirm
        if not themed_confirm(self.ui, S4T.CLEAR_PRED_TITLE,
                              S4T.CLEAR_PRED_CONFIRM.format(stack=name),
                              confirm_text=S4T.CLEAR_PRED_BTN,
                              cancel_text=S4T.CANCEL_BTN, accent="red"):
            return
        self._ensure_writable_state()
        self._clear_prediction_cache_for(name)
        status = record.get("status")
        if status == "predicted" or (status in ("failed", "segmented")
                                     and record.get("failed_stage")
                                     in (None, "predict")):
            self.state_manager.set_stack_status(name, "unprocessed")
        self.state_manager.clear_stack_keys(
            name, ["predicted_with", "source_fingerprint", "predicted_axes",
                   "failed_stage", "failed_at"])
        self._persist_state()
        self._refresh_stacks()

    def _on_stack_menu(self, name: str, pos):
        """A ① row's ``⋯`` menu, built on click, parented to the
        list; a greyed item carries its reason in its TEXT.  Under the lock
        it does not open (①'s lock line is the reason)."""
        if _step4_is_locked(self):
            print(f"[STEP4-LOCK] menu of {name} refused while "
                  f"{self._running_flag_names()}")
            self.ui.log_output.append(S4T.LOCK_MENU_REFUSED)
            return
        menu, clear, spacing = self._build_stack_menu(name)
        chosen = menu.exec_(pos) if hasattr(menu, "exec_") else menu.exec(pos)
        menu.deleteLater()
        if chosen is clear:
            self._clear_predictions(name)
        elif chosen is spacing:
            self._open_set_spacing([name])

    def _build_stack_menu(self, name: str):
        """``(menu, clear_action, spacing_action)`` for one row."""
        from qtpy.QtWidgets import QMenu
        record = self.state_manager.get_record(name)
        menu = QMenu(self.ui.session_stacks_list)
        has_3d = record.get("status") in S4R.HAS_3D
        clear = menu.addAction(S4T.MENU_CLEAR_PREDICTIONS_HAS_3D if has_3d
                               else S4T.MENU_CLEAR_PREDICTIONS)
        clear.setEnabled(not has_3d)
        spacing = menu.addAction(S4T.MENU_SET_SPACING)
        return menu, clear, spacing

    # ------------------------------------------------------------------
    # Set spacing…
    # ------------------------------------------------------------------

    def _on_set_spacing_clicked(self):
        names = [n for n in self.ui.ticked_names() if n]
        if names:
            self._open_set_spacing(names)

    def _open_set_spacing(self, names):
        if self._refuse_while_locked("Set spacing"):
            return
        if not names:
            return
        from ._set_spacing_dialog import SetSpacingDialog
        first = self.state_manager.get_record(names[0])
        dlg = SetSpacingDialog(self.ui, names=names, record=first)
        try:
            if not dlg.exec_():
                return
            vals = dlg.values()
        finally:
            dlg.deleteLater()                       # A1-5
        self._apply_spacing(names, vals)

    def _apply_spacing(self, names, vals, *, confirm=True) -> bool:
        """Write the user's spacing, Z orientation and axes onto each
        record, after a confirm naming what goes stale (5)."""
        new_spacing = {"zyx": [float(v) for v in vals["zyx"]],
                       "source": "user"}
        measure_stale, predict_stale = [], []
        for n in names:
            rec = self.state_manager.get_record(n)
            if rec.get("status") in ("qc_extracted", "finalized"):
                if (not S4M.same_spacing(rec.get("measured_spacing")
                                         or rec.get("spacing"), new_spacing)
                        or (rec.get("z_orientation") or "basal_down")
                        != vals["z_orientation"]):
                    measure_stale.append(n)
            new_axes = vals["axes_override"] or rec.get("axes") or ""
            old_axes = S4M.stack_axes(rec)
            if new_axes != old_axes and (
                    rec.get("predicted_axes") or rec.get("status")
                    == "predicted"):
                predict_stale.append(n)
        if confirm and (measure_stale or predict_stale):
            if measure_stale and predict_stale:
                what, action = S4T.SPACING_WHAT_BOTH
            elif measure_stale:
                what, action = S4T.SPACING_WHAT_MEASURE
            else:
                what, action = S4T.SPACING_WHAT_PREDICT
            n = len(set(measure_stale) | set(predict_stale))
            from ..ui_shell.theme import themed_confirm
            if not themed_confirm(
                    self.ui, S4T.SPACING_CHANGE_TITLE,
                    S4T.SPACING_CHANGE_CONFIRM.format(what=what, n=n,
                                                      action=action),
                    confirm_text=S4T.SPACING_APPLY_BTN,
                    cancel_text=S4T.CANCEL_BTN, accent="green"):
                return False
        self._ensure_writable_state()
        for n in names:
            # the shared file FIRST (audit M2): a record that was updated
            # while the file kept the old size would be put back by the
            # next scan
            if not self._write_spacing_file(
                    n, self.state_manager.get_record(n).get(
                        "source_stack_path"), new_spacing):
                self.ui.log_output.append(
                    f"[spacing] {S4T.SPACING_SAVE_FAILED}")
                self._refresh_stacks()
                return False
            self.state_manager.update_record(
                n, spacing=dict(new_spacing),
                z_orientation=vals["z_orientation"],
                axes_override=vals["axes_override"])
            self.ui.log_output.append(S4T.LOG_SPACING_SET.format(
                stack=n, spacing=S4R.spacing_text(new_spacing)))
        if not self._persist_state():
            self.ui.log_output.append(f"[spacing] {S4T.SPACING_SAVE_FAILED}")
        self._refresh_stacks()
        return True

    def _on_stack_started(self, stack_name, index, total):
        """A stack starts: its row reads the running word; the status
        line walks Step 3's sentences.  No "segmented" write."""
        import time as _time
        self._seg_index = index
        stage = getattr(self, "_run_kind", None) or "predict"
        self._running = getattr(self, "_running", {}) or {}
        self._running[stack_name] = (stage, "")
        if stage == "reconstruct":
            elapsed = _time.monotonic() - self._seg_start_time
            eta = ""
            if index > 0:
                eta = self._eta_text(elapsed, index / max(total, 1))
            self.ui.show_seg_status(TH.PIPE_RECONSTRUCTING_ITEM.format(
                i=index + 1, n=total, name=stack_name,
                time=TH.pipe_time(self._mmss(elapsed), eta)), active=True)
        self._refresh_stacks()

    @staticmethod
    def _mmss(seconds) -> str:
        m, s = divmod(int(seconds), 60)
        return f"{m}:{s:02d}"

    def _eta_text(self, elapsed, progress) -> str:
        if progress <= 0.01:
            return ""
        remaining = elapsed / progress * (1 - progress)
        return f"~{self._mmss(remaining)} remaining"

    def _on_seg_slice_progress(self, stack_name, view, current, total_slices):
        """Slice progress: Step 3's per-view sentence with
        unit "stack"; the reconstruction stages go indeterminate."""
        import time as _time

        elapsed = _time.monotonic() - self._seg_start_time
        if view == "reconstruct" or view == "postprocess":
            if self.ui.run_progress.maximum() != 0:
                if self._seg_last_view is not None:
                    self._seg_slices_done += self._seg_last_view_total
                    self._seg_last_view = None
                self.ui.run_progress.setRange(0, 0)  # scanning mode
            if view == "postprocess":
                text = S4T.STEP4_POSTPROCESSING.format(
                    stack=stack_name, t=self._mmss(elapsed))
            else:
                text = TH.PIPE_RECONSTRUCTING_ITEM.format(
                    i=getattr(self, "_seg_index", 0) + 1,
                    n=max(getattr(self, "_run_total", 1), 1),
                    name=stack_name,
                    time=TH.pipe_time(self._mmss(elapsed), ""))
            self.ui.show_seg_status(text, active=True)
            return

        if self.ui.run_progress.maximum() == 0:
            self.ui.run_progress.setRange(
                0, max(self._seg_total_slices, 1)
            )
        if (self._seg_last_view is not None
                and (view != self._seg_last_view
                     or stack_name != self._seg_last_stack)):
            self._seg_slices_done += self._seg_last_view_total
        view_changed = view != self._seg_last_view
        self._seg_last_view = view
        self._seg_last_view_total = total_slices
        self._seg_last_stack = stack_name

        cumulative = self._seg_slices_done + current
        self.ui.run_progress.setValue(min(cumulative, self._seg_total_slices))
        progress = cumulative / max(self._seg_total_slices, 1)
        eta = self._eta_text(elapsed, progress) if cumulative > 10 else ""
        self.ui.show_seg_status(TH.PIPE_SEGMENTING_VIEW.format(
            unit="stack", i=getattr(self, "_seg_index", 0) + 1,
            n=max(getattr(self, "_run_total", 1), 1), name=stack_name,
            VIEW=str(view).upper(), c=current, t=total_slices,
            time=TH.pipe_time(self._mmss(elapsed), eta)), active=True)
        running = getattr(self, "_running", {})
        if stack_name in running and (view_changed or current % 5 == 0
                                      or current == total_slices):
            running[stack_name] = (running[stack_name][0],
                                   S4T.ROW_RUNNING_VIEW.format(
                                       VIEW=str(view).upper(), c=current,
                                       t=total_slices))
            row = next((r for r in self._rows if r["name"] == stack_name),
                       None)
            if row is not None:
                entry = {"name": row["name"], "path": row["path"],
                         "origin": row["origin"], "conflict": row["conflict"]}
                new = S4R.build_row(
                    entry, self.state_manager.get_record(stack_name),
                    active_model=self._active_model,
                    needs=self._channels_needed(), ticked=row["ticked"],
                    running=running[stack_name],
                    cache_complete=row["cache_complete"])
                self.ui.update_row(stack_name, new)

    def _clear_workbench_caches(self):
        """Drop the plot's map caches (a Reset or a new result)."""
        self._workbench_cells_changed("reset or unload")
        pw = self._pw
        if pw is not None:
            for attr in ("_pca_cache", "_umap_cache", "_tsne_cache",
                         "_pc_projected"):
                try:
                    setattr(pw, attr, None)
                except Exception:      # noqa: BLE001
                    pass

    def _derived_file_paths(self, stack_name, record):
        """Every file derived from a stack's 3D result (F5, F19)."""
        paths = []
        csv = record.get("qc_features_csv") or ""
        if not csv and self.working_dir:
            csv = str(Path(self.working_dir) / "artifacts" / "step4" / "qc"
                      / f"{stack_name}_features.csv")
        if csv:
            c = Path(csv)
            paths += [c, c.parent / f"{stack_name}_adjacency.csv",
                      c.parent / (c.stem + "_raw.csv")]
        if self.working_dir:
            for ext in (".npz", ".json"):
                paths.append(self._eda_dir() / f"{stack_name}_eda{ext}")
        for key in ("border_cells_path", "smoothed_cache_path"):
            if record.get(key):
                paths.append(Path(record[key]))
        labels = record.get("labels_path") or ""
        if labels:
            paths.append(Path(labels).parent / "labels_3d_smoothed.tif")
        return paths

    def _delete_derived_files(self, stack_name, record):
        """Delete (never rename) a stack's derived files; returns names."""
        import shutil
        deleted = []
        if self.working_dir:
            meshes = (Path(self.working_dir) / "artifacts" / "step4" / "meshes"
                      / stack_name)
            if meshes.is_dir():           # audit D9: no stale meshes either
                try:
                    shutil.rmtree(meshes)
                    deleted.append(f"meshes/{stack_name}/")
                except OSError as exc:
                    self.ui.log_output.append(
                        f"[WARN] could not delete {meshes}: {exc}")
        for p in self._derived_file_paths(stack_name, record):
            if not self._inside_workspace(p):
                # a record from a copied workspace can still point into
                # the source: never delete there (live driver 2026-09-29)
                self.ui.log_output.append(
                    S4T.DELETE_OUTSIDE_REFUSED.format(path=p))
                print(f"[STEP4-DELETE] refused outside the workspace: {p}")
                continue
            try:
                p.unlink()
                deleted.append(p.name)
            except FileNotFoundError:
                pass
            except OSError as exc:
                self.ui.log_output.append(f"[WARN] could not delete {p}: {exc}")
        if deleted:
            self.ui.log_output.append(
                f"[reset] {stack_name}: deleted {', '.join(deleted)}")
        print(f"[STEP4-DELETE] {stack_name}: {deleted}")
        return deleted

    def _inside_workspace(self, path) -> bool:
        """Is *path* inside the current workspace?  (Every Step 4 delete
        asks first.)"""
        if not self.working_dir:
            return False
        try:
            ws = os.path.normcase(str(Path(self.working_dir).resolve()))
            p = os.path.normcase(str(Path(path).resolve()))
        except OSError:
            return False
        return p == ws or p.startswith(ws + os.sep)

    def _delete_old_run_folder(self, old_labels, new_labels,
                               stack_name=None):
        """Delete the previous result's run folder (Step 3's
        ``_cleanup_old_reconstruction_folders``), only inside
        ``artifacts/step4/runs/`` and never the new run's folder."""
        import shutil
        if not old_labels or not self.working_dir:
            return
        old_dir = Path(old_labels).parent
        runs = Path(self.working_dir) / "artifacts" / "step4" / "runs"
        try:
            inside = os.path.normcase(str(old_dir.resolve().parent)) == \
                os.path.normcase(str(runs.resolve()))
            same = (new_labels and os.path.normcase(str(old_dir.resolve()))
                    == os.path.normcase(str(Path(new_labels).parent.resolve())))
        except OSError:
            return
        if not inside or same or not old_dir.is_dir():
            print(f"[STEP4-DELETE] old run folder kept: {old_dir} "
                  f"(inside={inside}, same={same})")
            return
        # Audit D4: two runs started in the same second share a folder
        # name (logic.run_real, run_id to the second).  Keep a folder any
        # other stack's files still live in.
        sharers = self._stacks_using_folder(old_dir, exclude=stack_name)
        if sharers:
            self.ui.log_output.append(
                f"[reconstruct] kept {old_dir.name}/: it also holds "
                f"{', '.join(sharers)}")
            print(f"[STEP4-DELETE] {old_dir} shared with {sharers}; kept")
            return
        try:
            shutil.rmtree(old_dir)
            self.ui.log_output.append(
                f"[reconstruct] deleted the previous result {old_dir.name}/")
        except OSError as exc:
            self.ui.log_output.append(
                f"[WARN] could not delete {old_dir}: {exc}")

    def _stacks_using_folder(self, folder, exclude=None):
        """Other stacks whose recorded files sit inside *folder*."""
        keys = ("labels_path", "prob_path", "original_labels_path",
                "border_cells_path", "smoothed_cache_path")
        try:
            root = os.path.normcase(str(Path(folder).resolve()))
        except OSError:
            return []
        out = []
        for name in self.state_manager.get_all_stack_statuses():
            if name == exclude:
                continue
            rec = self.state_manager.get_record(name)
            for key in keys:
                p = rec.get(key)
                if not p:
                    continue
                try:
                    parent = os.path.normcase(str(Path(p).resolve().parent))
                except OSError:
                    continue
                if parent == root:
                    out.append(name)
                    break
        return out

    def _unload_stack(self, reason: str):
        """Forget the loaded stack (its files are gone)."""
        print(f"[STEP4-UNLOAD] {self._current_stack_name}: {reason}")
        if self._has_unsaved_changes:
            # Audit L6 (B2 closes the door with a confirm first).
            self.ui.log_output.append(
                f"[unload] {self._current_stack_name}: unsaved edits and "
                f"marks were dropped ({reason})")
        self._pending_border_removal = False                  # audit D14
        try:
            self._clear_stack_layers()
        except Exception as exc:      # noqa: BLE001
            print(f"[STEP4-UNLOAD] layers: {exc}")
        self._labels_volume = None
        self._labels_epoch += 1
        self._qc_features_df = None
        self._last_pca_projected = None
        self._cached_eda_state = None
        self._current_stack_name = None
        self._sync_loaded_stack_marker()
        self._has_unsaved_changes = False
        self._pending_edits = {}
        self._undo_slot = None
        self._clear_workbench_caches()
        self._update_qc_button_states()

    def _on_stack_completed(self, stack_name, result):
        old = self.state_manager.get_record(stack_name)
        labels_path = result.get("labels_path", "")
        source_path = result.get("source_stack_path", "")
        new_ok = bool(labels_path) and Path(labels_path).is_file()
        if not new_ok:
            # Re-audit M2 / D15: a result whose labels are not on disk is a
            # failure; the old record and files are left as they are.
            print(f"[STEP4-DELETE] {stack_name}: new labels missing "
                  f"({labels_path!r}); old result kept")
            self._on_stack_error(stack_name, "reconstruct",
                                 f"the new labels file is missing: "
                                 f"{labels_path or '(none)'}")
            return
        # Derive probability path
        prob_path = ""
        if labels_path:
            p = Path(labels_path).parent / f"{Path(source_path).stem}_probabilities.npy"
            if p.exists():
                prob_path = str(p)
        self.state_manager.set_stack_status(
            stack_name, "reconstructed",
            labels_path=labels_path,
            source_stack_path=source_path,
            prob_path=prob_path,
            made_with=result.get("made_with"),
            n_cells=result.get("num_cells_final"),
        )
        self.state_manager.clear_stack_keys(stack_name,
                                            ["failed_stage", "failed_at"])
        self._write_manifest_made_with(result.get("manifest_path"),
                                       result.get("made_with"))
        old_labels = old.get("labels_path") or ""
        if old_labels and \
                os.path.normcase(os.path.normpath(old_labels)) != \
                os.path.normcase(os.path.normpath(labels_path)):
            # A new 3D result replaces the old one: every measure key off
            # the record, every derived file and the old run folder deleted
            # (F19, I-16).
            self.state_manager.reset_to_reconstructed(stack_name)
            self.state_manager.clear_stack_keys(stack_name,
                                                ["original_labels_path"])
            self._delete_derived_files(stack_name, old)
            self._delete_old_run_folder(old_labels, labels_path,
                                        stack_name)
            if self._current_stack_name == stack_name:
                self._unload_stack("reconstructed again")
        self._run_done = getattr(self, "_run_done", 0) + 1
        getattr(self, "_running", {}).pop(stack_name, None)
        self._persist_state()
        self._refresh_stacks()

    def _write_manifest_made_with(self, manifest_path, made_with):
        """v2.6: the run's manifest carries ``made_with`` too, so a
        workspace re-adopted from its manifests keeps the model."""
        if not manifest_path or not made_with:
            return
        import json as _json
        try:
            p = Path(manifest_path)
            data = _json.loads(p.read_text(encoding="utf-8"))
            data["made_with"] = made_with
            tmp = p.with_name(p.name + ".tmp")
            tmp.write_text(_json.dumps(data, indent=2, default=str),
                           encoding="utf-8")
            os.replace(str(tmp), str(p))
        except Exception as exc:      # noqa: BLE001
            print(f"[STEP4-MANIFEST] made_with not written: {exc}")

    def _on_all_stacks_completed(self):
        """The done line: Step 3's sentence with the success
        count, the failure count and the elapsed time."""
        import time as _time
        self.ui.cancel_btn.setEnabled(False)
        if self.ui.run_progress.maximum() == 0:
            self.ui.run_progress.setRange(0, 100)
        self.ui.run_progress.setValue(self.ui.run_progress.maximum())
        self.ui.run_progress.setFormat("")     # F10: no text when idle
        elapsed = _time.monotonic() - getattr(self, "_seg_start_time",
                                              _time.monotonic())
        kind = getattr(self, "_run_kind", None) or "predict"
        total = int(getattr(self, "_run_total", 0))
        failures = int(getattr(self, "_run_failures", 0))
        done = int(getattr(self, "_run_done", 0))
        head = (TH.PIPE_RECONSTRUCTION_DONE if kind == "reconstruct"
                else TH.PIPE_SEGMENTATION_DONE)
        summary = head.format(s=done, n=total, units=TH.pipe_units("stack"))
        if failures:
            summary += TH.PIPE_FAILED_SUFFIX.format(k=failures)
        summary += TH.PIPE_TIME_SUFFIX.format(t=self._mmss(elapsed))
        self.ui.log_output.append(summary)
        self._running = {}
        self._run_kind = None
        self._refresh_stacks()
        self.ui.show_seg_status(summary)
        self._last_done_line = summary

    def _cleanup_seg_thread(self):
        self._seg_thread = None
        self._seg_worker = None

    def _restore_run_buttons(self):
        """Backstop at thread end: the run buttons follow their gates."""
        self._running = {}
        self._run_kind = None
        btn = getattr(self.ui, "cancel_btn", None)
        if btn is not None:
            btn.setEnabled(False)
        self._refresh_segment_gates(keep_status=True)

    def _on_predict_error(self, stack_name, msg):
        self._on_stack_error(stack_name, "predict", msg)

    def _on_recon_error(self, stack_name, msg):
        self._on_stack_error(stack_name, "reconstruct", msg)

    def _on_stack_error(self, stack_name, stage, msg):
        """A per-stack failure: record it; demote to ``failed``
        only when nothing later exists.  Raw text goes to the log only."""
        text = (TH.PIPE_ERROR_PREDICT if stage == "predict"
                else TH.PIPE_ERROR_RECON)
        self.ui.log_output.append(text.format(name=stack_name, msg=msg))
        self._run_failures = getattr(self, "_run_failures", 0) + 1
        getattr(self, "_running", {}).pop(stack_name, None)
        if not stack_name:
            return
        from datetime import datetime, timezone
        status = self.state_manager.get_stack_status(stack_name)
        fields = dict(failed_stage=stage,
                      failed_at=datetime.now(timezone.utc).isoformat())
        has_3d = status in ("reconstructed", "qc_extracted", "finalized")
        kept_cache = (stage == "predict" and status == "predicted"
                      and S4M.find_prediction_cache(
                          self._prediction_cache_dir(),
                          stack_name)["complete"])
        if has_3d or kept_cache:
            print(f"[STEP4-FAIL] {stack_name}: {stage} failed; "
                  f"{status} kept")
            self.state_manager.update_record(stack_name, **fields)
        else:
            print(f"[STEP4-FAIL] {stack_name}: {stage} failed -> failed")
            self.state_manager.set_stack_status(stack_name, "failed",
                                                **fields)
        self._persist_state()
        try:
            self._refresh_stacks()
        except Exception as exc:      # noqa: BLE001
            print(f"[STEP4-FAIL] refresh failed: {exc}")

    def _cancel_run(self):
        if self._seg_worker is not None:
            self._seg_worker.request_cancel()
        self.ui.show_seg_status(TH.PIPE_CANCELLING.format(unit="stack"))
        self.ui.log_output.append(S4T.LOG_CANCEL_REQUESTED)
        self.ui.cancel_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # Start from
    # ------------------------------------------------------------------

    def _refresh_start_from_items(self):
        """The Step 3 item is offered only when a Step 3 manifest exists."""
        combo = self.ui.start_from_combo
        found = S4M.newest_step3_manifest(self.working_dir)
        self._step3_preset = found
        cur = self._start_from_key()
        combo.blockSignals(True)
        try:
            combo.clear()
            if found:
                combo.addItem(S4T.SEG_PRESET_STEP3.format(run=found["run"]),
                              "step3")
            combo.addItem(S4T.SEG_PRESET_EPICYCLE, "epicycle")
            combo.addItem(S4T.SEG_PRESET_USEG3D, "useg3d")
            if cur == "custom":
                combo.addItem(S4T.SEG_PRESET_CUSTOM, "custom")
            idx = combo.findData(cur)
            combo.setCurrentIndex(idx if idx >= 0 else combo.findData(
                "epicycle"))
        finally:
            combo.blockSignals(False)

    def _start_from_key(self) -> str:
        combo = getattr(self.ui, "start_from_combo", None)
        if combo is None or combo.currentIndex() < 0:
            return "epicycle"
        return combo.currentData() or "epicycle"

    def _restore_start_from(self, key):
        self._refresh_start_from_items()
        if key in ("step3", "useg3d") and self.ui.start_from_combo.findData(
                key) >= 0:
            self._apply_start_from(key, save=False)
        else:
            # "custom" values were not saved across sessions: the widgets
            # hold EpiCycle's defaults, and the log says so (F12).
            if key in ("custom", "step3"):
                self.ui.log_output.append(
                    S4T.LOG_START_FROM_NOT_RESTORED.format(preset=key))
            self._select_start_from("epicycle")

    def _select_start_from(self, key):
        combo = self.ui.start_from_combo
        idx = combo.findData(key)
        if idx >= 0:
            combo.blockSignals(True)
            combo.setCurrentIndex(idx)
            combo.blockSignals(False)

    def _on_start_from_chosen(self, index):
        key = self.ui.start_from_combo.itemData(index)
        if key and key != "custom":
            self._apply_start_from(key)

    def _preset_values(self, key):
        if key == "epicycle":
            return dict(S4M.STEP4_PRESET_EPICYCLE)
        if key == "useg3d":
            return S4M.useg3d_preset()
        if key == "step3" and self._step3_preset:
            return S4M.step3_recon_preset(self._step3_preset["recon_config"])
        return {}

    def _apply_start_from(self, key, *, save: bool = True):
        """Write a preset's real values into the widgets."""
        values = self._preset_values(key)
        self._applying_preset = True
        try:
            for name, value in values.items():
                w = getattr(self.ui, name, None)
                if w is None:
                    continue
                try:
                    if hasattr(w, "setChecked") and not hasattr(w,
                                                                "setValue"):
                        w.setChecked(bool(value))
                    elif hasattr(w, "decimals"):
                        w.setValue(float(value))
                    elif hasattr(w, "setValue"):
                        w.setValue(int(round(float(value))))   # QSpinBox
                except Exception as exc:      # noqa: BLE001 -- A1-4
                    print(f"[STEP4-UI] start from {key}: {name}={value!r} "
                          f"skipped ({exc})")
                    self.ui.log_output.append(S4T.LOG_PRESET_SKIPPED.format(
                        name=name))
            if key == "step3" and self._step3_preset:
                self._apply_step3_channels(self._step3_preset.get(
                    "channels"))
        finally:
            self._applying_preset = False
        combo = self.ui.start_from_combo
        ci = combo.findData("custom")
        if ci >= 0:
            combo.removeItem(ci)
        self._select_start_from(key)
        print(f"[STEP4-UI] start from {key}: {len(values)} values")
        self.ui.log_output.append(S4T.LOG_PRESET_APPLIED.format(
            preset=combo.currentText()))
        if save and self.working_dir and not getattr(self, "_adopt_only",
                                                     False):
            self.state_manager.set_session_metadata(start_from=key)
            self._persist_state()

    def _apply_step3_channels(self, codes):
        """The Step 3 preset also fills the channels (its
        manifest's Cellpose codes); it never touches the model card."""
        idx = S4M.trained_channel_indices(codes) if codes else None
        if idx is None or not hasattr(self.ui, "set_channel_choice"):
            return
        self._refresh_channel_combos()
        self.ui.set_channel_choice(self.ui.chan1_combo, idx[0])
        self.ui.set_channel_choice(self.ui.chan2_combo, idx[1])

    def _connect_setting_edits(self):
        """A hand edit after a preset flips Start from to "Custom"."""
        names = set(S4M.STEP4_PRESET_EPICYCLE) | set(
            S4M.USEG3D_WIDGET_KEYS)
        for name in sorted(names):
            w = getattr(self.ui, name, None)
            if w is None:
                continue
            sig = getattr(w, "valueChanged", None) or getattr(w, "toggled",
                                                             None)
            if sig is not None:
                sig.connect(self._on_setting_edited)

    def _on_setting_edited(self, *_):
        if getattr(self, "_applying_preset", False):
            return
        combo = self.ui.start_from_combo
        if combo.currentData() == "custom":
            return
        if combo.findData("custom") < 0:
            combo.addItem(S4T.SEG_PRESET_CUSTOM, "custom")
        self._select_start_from("custom")
        if self.working_dir and not getattr(self, "_adopt_only", False):
            self.state_manager.set_session_metadata(start_from="custom")

    # ------------------------------------------------------------------
    # Channels, cell width, gates
    # ------------------------------------------------------------------

    def _channels_needed(self):
        """How many channels a stack needs: the model's trained channels
        when known, else the chosen combos; None while blank."""
        need = S4M.channels_needed(getattr(self, "_trained_codes", None))
        if need is not None:
            return need
        if not hasattr(self.ui, "chosen_channels"):
            return None
        c1, c2 = self.ui.chosen_channels()
        if c1 < 0:
            return None
        return max(c1, c2 if c2 is not None else 0) + 1

    def _refresh_channel_combos(self):
        """Items for 0 .. the largest channel count among the ticked stacks
        (all listed stacks when none is ticked); names from the loaded or
        previewed stack's channel layers, roles from the model's trained
        channels."""
        if not hasattr(self.ui, "update_channel_combos"):
            return
        rows = getattr(self, "_rows", []) or []
        ticked = set(self.ui.ticked_names()) if hasattr(
            self.ui, "ticked_names") else set()
        pool = [r for r in rows if r["name"] in ticked] or rows
        counts = [r["n_channels"] for r in pool if r.get("n_channels")]
        n = max(counts) if counts else 1
        names = self._channel_display_names(n)
        roles = {}
        idx = S4M.trained_channel_indices(getattr(self, "_trained_codes",
                                                  None))
        if idx is not None:
            mem, nuc = idx
            if mem < n:
                roles[mem] = S4T.SEG_MODEL_ROLE_MEMBRANE
            if nuc is not None and nuc < n:
                roles[nuc] = S4T.SEG_MODEL_ROLE_NUCLEI
        self.ui.update_channel_combos(n, channel_names=names, roles=roles)

    def _tag_channel_layers(self, added) -> None:
        """Mark each channel layer of the loaded stack with its channel
        index, and refresh ②'s channel names when the user renames one in
        the viewer (owner 2026-09-29).  The pick is never changed: the
        combos restore the remembered choice."""
        layers = list(added) if isinstance(added, (list, tuple)) else [added]
        for i, lyr in enumerate(layers):
            try:
                lyr._epicycle_channel_index = i
                lyr.events.name.connect(self._on_channel_layer_renamed)
            except Exception as exc:      # noqa: BLE001
                print(f"[STEP4-UI] channel layer {i} not tagged: {exc}")

    def _on_channel_layer_renamed(self, event=None):
        print("[STEP4-UI] a channel layer was renamed; (2) names refresh")
        try:
            self._refresh_channel_combos()
        except Exception as exc:      # noqa: BLE001
            print(f"[STEP4-UI] channel names not refreshed: {exc}")

    def _channel_display_names(self, n: int) -> list:
        """The names ② shows (owner live test 2026-10-05): the
        channel layers' names of ONE stack -- the one loaded in ③ if its
        channel layers are in the viewer, else the one previewed with View
        or the eye -- or ``None`` per channel when no stack's layers name
        it (the item then reads the model's role, or "ch{i}").  The model's
        training-time names are never used: they belong to another
        dataset."""
        names = [None] * n
        if self.viewer is None:
            return names
        live = list(self.viewer.layers)

        def channel_layers(layers):
            out = {}
            for lyr in layers or []:
                if not any(lyr is x for x in live):
                    continue
                i = getattr(lyr, "_epicycle_channel_index", None)
                if i is not None and 0 <= int(i) < n:
                    out[int(i)] = lyr
            return out

        found = channel_layers(getattr(self, "_stack_layers", None))
        if not found:
            found = channel_layers(getattr(self, "_preview_layers", None))
        for i, lyr in found.items():
            name = str(lyr.name)
            if name.startswith(STEP4_LAYER_PREFIX):
                name = name[len(STEP4_LAYER_PREFIX):]
            tag = " " + self._PREVIEW_TAG
            if name.endswith(tag):
                name = name[:-len(tag)]
            names[i] = name.strip() or None
        return names

    def _on_channels_changed(self, *_):
        if getattr(self, "_trained_codes", None) is None:
            # With unknown trained channels the chosen combos drive the
            # row warnings.
            self._refresh_stacks()
        else:
            self._refresh_segment_gates()

    def _on_cell_width_changed(self, *_):
        self._update_cell_width_line()

    def _update_cell_width_line(self):
        """``SEG_CELL_WIDTH_PX`` from the highlighted row's Y/X spacing, or
        the first ticked row with a spacing."""
        um = self.ui.cell_width_um_sb.value()
        row = self._highlighted_row()
        if not row or not row.get("spacing"):
            ticked = set(self.ui.ticked_names())
            row = next((r for r in getattr(self, "_rows", [])
                        if r["name"] in ticked and r.get("spacing")), row)
        spacing = row.get("spacing") if row else None
        px = S4M.cell_width_px(um, spacing)
        self._shown_px = px
        if px is None:
            self.ui.show_cell_width_px("")
            return
        self.ui.show_cell_width_px(S4T.SEG_CELL_WIDTH_PX.format(
            px=px, um=f"{S4M.xy_um(spacing):.2f}"))

    def _refresh_segment_gates(self, keep_status: bool = False):
        """Every grey state of ② and its on-screen reason.

        A missing or absent model: the model card says why and the status
        line says nothing more.  Otherwise the first of SEG_NOTHING_TICKED,
        SEG_NONE_RUNNABLE and SEG_PICK_CHANNEL; when Predict can run, the
        idle line "Ready · {n} stacks ticked", followed by
        SEG_NOTHING_TO_RECONSTRUCT while Reconstruct is grey and by
        VIEW_NO_RESULT while a highlighted row has no result."""
        ui = self.ui
        if not hasattr(ui, "show_seg_status"):
            return
        running = bool(getattr(self, "_predict_running", False)
                       or getattr(self, "_reconstruct_running", False))
        rows = getattr(self, "_rows", []) or []
        ticked = set(self._stored_ticked())
        snap = getattr(self, "_active_model", None) or self._model_snapshot()
        missing = S4M.model_files_missing(snap)
        c1, _c2 = ui.chosen_channels()
        runnable = [r for r in rows if r["name"] in ticked and r["can_run"]]
        n_run = len(runnable)
        model_reason = snap is None or missing       # the card says why
        predict_reason = None
        if model_reason:
            predict_reason = ""
        elif not ticked:
            predict_reason = S4T.SEG_NOTHING_TICKED
        elif not runnable:
            predict_reason = S4T.SEG_NONE_RUNNABLE
        elif c1 < 0:
            predict_reason = S4T.SEG_PICK_CHANNEL
        recon_ok = any(S4R.recon_ready(r) for r in runnable)
        high = self._highlighted_row()
        view_ok = bool(high and high.get("has_result"))
        if running or _step4_is_locked(self):
            # R1-1: ANY Step 4 job greys Predict and Reconstruct; View
            # stays live only under Predict / Reconstruct (it reads a TIFF
            # on the main thread and adds layers).  The lock line is the
            # reason, so the status line never reads "Ready" under a lock.
            ui.predict_btn.setEnabled(False)
            ui.reconstruct_btn.setEnabled(False)
            ui.view_btn.setEnabled(view_ok and running)
            ui.set_lock_lines(self._lock_line_text())
            if not running:
                ui.show_seg_status("")
            return
        ui.set_lock_lines(self._lock_line_text())
        ui.predict_btn.setEnabled(predict_reason is None)
        ui.reconstruct_btn.setEnabled(recon_ok)
        ui.view_btn.setEnabled(view_ok)
        if keep_status and getattr(self, "_last_done_line", ""):
            return
        self._last_done_line = ""
        if predict_reason is not None:
            text = predict_reason
        else:
            # the stacks that WILL run; View's reason is the highlighted
            # row's own status word, so it is not repeated here
            parts = [S4T.seg_idle_line(n_run)]
            if not recon_ok:
                parts.append(S4T.SEG_NOTHING_TO_RECONSTRUCT)
            text = S4T.SEG_TICKED_JOIN.join(parts)
        ui.show_seg_status(text)
        print(f"[STEP4-UI] gates predict={predict_reason is None} "
              f"recon={recon_ok} view={view_ok} status={text!r}")

    def _stored_ticked(self):
        """``session.ticked``: the ticks, INCLUDING stacks that cannot run
        (their box is greyed, so the lists cannot report them; F2)."""
        try:
            stored = list(self.state_manager.get_ticked())
        except Exception:      # noqa: BLE001
            return list(self.ui.ticked_names())
        rows = getattr(self, "_rows", None)
        if rows:
            listed = {r["name"] for r in rows}
            stored = [n for n in stored if n in listed]      # N4
        return stored

    def _lock_line_text(self) -> str:
        """The running job's lock sentence; B2 greys the rest
        of the lock set."""
        if getattr(self, "_predict_running", False):
            return S4T.STEP4_LOCK_PREDICT
        if getattr(self, "_reconstruct_running", False):
            return S4T.STEP4_LOCK_RECONSTRUCT
        if getattr(self, "_measure_running", False):
            return S4T.STEP4_LOCK_MEASURE
        stack = self._current_stack_name or ""
        if getattr(self, "_smooth_running", False):
            return S4T.STEP4_LOCK_SMOOTH.format(stack=stack)
        if getattr(self, "_save_in_flight", False):
            return S4T.STEP4_LOCK_SAVE.format(stack=stack)
        if getattr(self, "_load_in_flight", False) or \
                getattr(self, "_preview_in_flight", False):
            launch = getattr(self, "_load_launch", None) or {}
            return S4T.STEP4_LOCK_LOAD.format(
                stack=launch.get("stack") or stack)
        if getattr(self, "_export_running", False) or \
                getattr(self, "_mesh_running", False):
            return S4T.STEP4_LOCK_EXPORT
        return ""

    def _on_view_clicked(self):
        """View: the HIGHLIGHTED row's result in the viewer."""
        row = self._highlighted_row()
        if not row or not row.get("has_result"):
            self.ui.show_seg_status(S4T.VIEW_NO_RESULT)
            return
        name = row["name"]
        record = self.state_manager.get_record(name)
        if self.viewer is None:
            return
        try:
            import tifffile
            labels = tifffile.imread(record["labels_path"])
            short = Path(name).stem
            self._preview_selected_stack(name, record)
            self._track_step4_layers(self.viewer.add_labels(
                labels, name=f"{STEP4_LAYER_PREFIX}{short} labels "
                             f"{self._PREVIEW_TAG}"))
        except Exception as exc:      # noqa: BLE001
            self.ui.log_output.append(f"[view] {name}: {exc}")
            self.ui.show_seg_status(S4T.MT_PREVIEW_FAILED.format(stack=name))

    # ------------------------------------------------------------------
    # Session thumbnails (I-20)
    # ------------------------------------------------------------------

    def _sync_loaded_stack_marker(self) -> None:
        """(3) marks the row of the stack loaded in the viewer
        (``_current_stack_name``; none after an unload) -- A3 fold.  A
        decoration: a UI without the setter is skipped."""
        setter = getattr(getattr(self, "ui", None), "set_loaded_stack", None)
        if callable(setter):
            setter(getattr(self, "_current_stack_name", None))

    def _thumb_cache_dir(self) -> Path:
        """``artifacts/step4/thumbs``.  After a read-only adopt the
        workspace is not written (I12): the PNGs go to a per-user temp
        folder instead, and the in-memory cache carries the pictures over
        to the first write-mode refresh."""
        if getattr(self, "_adopt_only", False):
            import hashlib
            import tempfile
            key = hashlib.sha1(os.path.normcase(os.path.abspath(
                self.working_dir)).encode("utf-8")).hexdigest()[:12]
            return Path(tempfile.gettempdir()) / "epicycle_step4_thumbs" / key
        return Path(self.working_dir) / "artifacts" / "step4" / "thumbs"

    # ------------------------------------------------------------------
    # A3: the bucket counts on the measured rows
    # ------------------------------------------------------------------

    def _tier_csv_stamps(self):
        """``{stack name: (csv path, mtime_ns)}`` of every row with a saved
        features CSV on disk (one ``os.stat`` each, main thread)."""
        out = {}
        for r in getattr(self, "_rows", []) or []:
            csv = r.get("features_csv") or ""
            if not csv or not r.get("name"):
                continue
            try:
                out[r["name"]] = (csv, os.stat(csv).st_mtime_ns)
            except OSError:
                continue
        return out

    def _restart_tier_counts_if_csv_changed(self):
        """Count again when a saved CSV is new or newer (Measure, Save,
        Finalize, Reset, a Browse) -- what is ON DISK, not the marks in the
        plot.  The count runs on a worker (the Step 3 counter, given
        Step 4's three buckets)."""
        if not hasattr(self.ui, "set_tier_counts"):
            return
        stamps = self._tier_csv_stamps()
        key = (self.working_dir, tuple(sorted(stamps.items())))
        if key == getattr(self, "_tier_counts_key", None):
            return
        self._tier_counts_key = key
        from napari_epicycle.fusion_reconstruction import (
            _step3_tier_counts as TC,
        )
        from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
            WB_TIER_BUCKETS, wb_tier_bucket,
        )
        self._tier_counts_token = (
            getattr(self, "_tier_counts_token", 0) or 0) + 1
        requests = [(n, c) for n, (c, _m) in stamps.items()]
        if not requests:
            self._on_tier_counts_counted((self._tier_counts_token, None), {})
            TC.stop_tier_counts(getattr(self, "_tier_counts_state", None),
                                wait_ms=0)
            return
        self._tier_counts_state = TC.start_tier_counts(
            requests, (self._tier_counts_token, self.working_dir),
            self._on_tier_counts_counted,
            state=getattr(self, "_tier_counts_state", None),
            order=WB_TIER_BUCKETS, bucket_of=wb_tier_bucket)
        print(f"[STEP4-UI] counts: {len(requests)} stack(s)")

    def _on_tier_counts_counted(self, stamp, result):
        """A batch's counts on the main thread: dropped when the workspace
        or the batch moved on; else put on the rows (the whole map
        replaced)."""
        try:
            token, ws = stamp
        except (TypeError, ValueError):
            return
        if token != getattr(self, "_tier_counts_token", None) or (
                ws is not None and ws != self.working_dir):
            print("[STEP4-UI] counts: stale batch dropped")
            return
        from napari_epicycle.fusion_reconstruction import (
            _step3_tier_counts as TC,
        )
        from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
            WB_TIER_BUCKETS, _WB_BUCKET_RGB,
        )
        setter = getattr(self.ui, "set_tier_counts", None)
        if callable(setter):                 # a lightweight test UI has none
            setter({n: TC.tier_items(c, _WB_BUCKET_RGB, order=WB_TIER_BUCKETS)
                    for n, c in (result or {}).items()}, replace=True)

    @staticmethod
    def _thumb_labels_for(row):
        """The labels file whose mid-Z overlay a stack's picture carries
        (A3, mirroring Step 3's ``_s3_thumb_labels_for``): the record's
        ``labels_path`` once the stack has a 3D result, else ``None``.
        Names the path only; the caller does the one ``os.stat``."""
        path = (row or {}).get("labels_path") or ""
        return path or None

    def _thumb_label_stamps(self):
        """``{normalised stack path: labels mtime_ns}`` of every listed row
        with a labels file on disk -- what the drawn pictures depend on."""
        out = {}
        for r in getattr(self, "_rows", []) or []:
            lab = self._thumb_labels_for(r)
            if not lab or not r.get("path"):
                continue
            try:
                out[S4R.norm_path(r["path"])] = os.stat(lab).st_mtime_ns
            except OSError:
                continue
        return out

    def _restart_thumbs_if_labels_changed(self):
        """A Reconstruct / Reset / Measure / Save moved a labels file: the
        overlay on that picture is out of date, so draw the pictures again
        (the cache keys the labels' mtime, so only that stack is redrawn)."""
        if getattr(self, "_thumbs_pending", False):
            return          # the caller starts them right after (audit L4)
        if self._thumb_label_stamps() != getattr(
                self, "_thumb_labels_seen", {}):
            self._start_thumbs()

    def _start_thumbs(self):
        """One new worker per Session refresh; rows keep a placeholder
        until their ``ready`` arrives.  The main thread only stats files."""
        if not self.working_dir or not hasattr(self.ui, "set_row_thumbnail"):
            return
        from ._stack_thumbs import (
            THUMB_WIDE_PX, ThumbCache, ThumbRequest, has_labels_part,
            start_thumb_worker,
        )
        self._thumbs_pending = False
        if getattr(self, "_thumb_cache", None) is None:
            self._thumb_cache = ThumbCache()
        requests = []
        self._thumb_requested = {}
        self._thumb_labels_seen = self._thumb_label_stamps()
        for r in getattr(self, "_rows", []) or []:
            path = r.get("path")
            if not path or r.get("conflict"):
                continue
            try:
                st = os.stat(path)
            except OSError:
                continue
            lab_path = self._thumb_labels_for(r) or ""
            lab_ns = self._thumb_labels_seen.get(S4R.norm_path(path), 0)
            if not lab_ns:
                lab_path = ""
            pm = self._thumb_cache.get(path, st.st_mtime_ns, st.st_size,
                                       lab_ns)
            if pm is not None:
                self.ui.set_row_thumbnail(path, pm)
                continue
            axes = r.get("axes") or ""
            if not axes:
                continue
            key = S4R.norm_path(path)
            self._thumb_requested[key] = (st.st_mtime_ns, st.st_size, lab_ns)
            requests.append(ThumbRequest(r["name"], path, axes,
                                         st.st_mtime_ns, st.st_size,
                                         lab_path, lab_ns))
        if not requests:
            return
        old = getattr(self, "_thumb_worker", None)
        if old is not None:
            try:
                old.cancel()
            except RuntimeError:
                pass
        from functools import partial
        thread, worker = start_thumb_worker(
            None, requests, str(self._thumb_cache_dir()), THUMB_WIDE_PX,
            on_ready=partial(self._on_thumb_ready_gen,
                             dict(self._thumb_requested)),
            on_failed=self._on_thumb_failed,
            on_finished=self._on_thumb_finished,
            on_error=self._on_thumb_error)
        self._thumb_thread, self._thumb_worker = thread, worker
        print(f"[STEP4-UI] thumbs: {len(requests)} requested")

    def _on_thumb_ready_gen(self, requested, name, stack_path, png_path,
                            png_bytes):
        """A1-7: a picture is cached only under the stamp (mtime, size) its
        own worker was asked for, and only while that stamp is still the
        listed one: an older worker's picture never lands under a newer
        stamp."""
        key = S4R.norm_path(stack_path)
        meta = requested.get(key)
        if meta is None or getattr(self, "_thumb_requested", {}).get(
                key) != meta:
            print(f"[STEP4-UI] thumb {name}: stale picture dropped")
            return
        self._on_thumb_ready(name, stack_path, png_path, png_bytes)

    def _on_thumb_ready(self, name, stack_path, png_path, png_bytes):
        key = S4R.norm_path(stack_path)
        meta = getattr(self, "_thumb_requested", {}).get(key)
        if meta is None:
            return            # a stack no longer listed: dropped
        # a raw FALLBACK (the labels were unreadable) is named without the
        # labels part: cache it under the raw key so the next trigger asks
        # again
        extra = meta[2] if len(meta) > 2 else 0
        from ._stack_thumbs import has_labels_part
        if extra and png_path and not has_labels_part(png_path):
            extra = 0
        pm = self._thumb_cache.load(stack_path, meta[0], meta[1],
                                    png_path=png_path or None,
                                    png_bytes=png_bytes, extra=extra)
        if pm is not None:
            self.ui.set_row_thumbnail(stack_path, pm)

    def _on_thumb_failed(self, name, stack_path, reason):
        print(f"[STEP4-UI] thumb {name}: {reason}")

    def _on_thumb_error(self, message):
        print(f"[STEP4-UI] thumbs: {message}")

    def _on_thumb_finished(self):
        pass

    def _stop_eda_for_shutdown(self):
        """napari's close: a running map's slots are disconnected FIRST, so
        no handler runs after; then quit, wait 3 s, park (audit E1-2)."""
        thread = getattr(self, "_eda_thread", None)
        worker = getattr(self, "_eda_worker", None)
        if thread is None:
            return
        if worker is not None:
            for sig, slot in ((worker.finished, self._on_eda_finished),
                              (worker.error, self._on_eda_error)):
                try:
                    sig.disconnect(slot)
                except (TypeError, RuntimeError):
                    pass
        try:
            thread.finished.disconnect()
        except (TypeError, RuntimeError):
            pass
        from ._eda_worker import stop_eda_thread
        stop_eda_thread(thread, worker)
        self._eda_thread = None
        self._eda_worker = None
        self._eda_launch = None
        self._eda_running = False

    #: The job threads ``shutdown()`` stops (``{attr}_thread`` /
    #: ``{attr}_worker``): Predict / Reconstruct, Measure, Smooth, Load.
    _SHUTDOWN_JOB_ATTRS = ("_seg", "_qc", "_smooth", "_load")

    def _on_ui_destroyed(self, *_args) -> None:
        """The panel is being destroyed (Step 3's hook)."""
        self.shutdown()

    def _stop_job_thread_for_shutdown(self, attr: str) -> None:
        """One job thread at close: ask the worker to stop,
        silence it and its thread (no handler of this controller runs after
        this -- the backstop included), quit, wait at most 3 s; a thread
        still running (a Cellpose predict) is parked in
        ``_LIVE_STEP4_THREADS``, never destroyed."""
        thread = getattr(self, f"{attr}_thread", None)
        worker = getattr(self, f"{attr}_worker", None)
        setattr(self, f"{attr}_thread", None)
        setattr(self, f"{attr}_worker", None)
        if thread is None:
            return
        if worker is not None:
            cancel = getattr(worker, "request_cancel", None)
            if callable(cancel):
                try:
                    cancel()
                except Exception as exc:      # noqa: BLE001
                    print(f"[STEP4-LOCK] shutdown: {attr} cancel failed: "
                          f"{exc}")
            # K2: DISCONNECT every signal of the worker (as
            # _stop_eda_for_shutdown does), so a call it queued just
            # before the close is dropped too -- blockSignals alone stops
            # only new emits, not ones already posted to the loop.
            _disconnect_all_signals(worker)
            try:
                worker.blockSignals(True)
            except RuntimeError:
                pass
        try:
            try:
                thread.finished.disconnect()
            except (TypeError, RuntimeError):
                pass
            thread.blockSignals(True)       # the finished backstop too
            thread.quit()
            stopped = bool(thread.wait(3000)) if thread.isRunning() \
                else True
        except RuntimeError:
            stopped = True
        if not stopped:
            _LIVE_STEP4_THREADS.add((thread, worker))
            print(f"[STEP4-LOCK] shutdown: {attr} thread still running; "
                  f"parked")
        else:
            print(f"[STEP4-LOCK] shutdown: {attr} thread stopped")
        self._retire_thread(thread)

    def shutdown(self):
        """napari's close: stop a Workbench map, the thumbnail
        worker, every job thread -- Predict / Reconstruct, Measure,
        Smooth, Load -- (at most 3 s each; a thread still running is
        parked, never destroyed) and the mesh subprocess.  Idempotent:
        ``aboutToQuit`` and ``ui.destroyed`` both call it."""
        print("[STEP4-LOCK] shutdown")
        self._stop_eda_for_shutdown()
        from ._stack_thumbs import stop_thumb_worker
        thread = getattr(self, "_thumb_thread", None)
        if thread is not None:
            stop_thumb_worker(thread, getattr(self, "_thumb_worker", None))
        self._thumb_thread = None
        self._thumb_worker = None
        from napari_epicycle.fusion_reconstruction._step3_tier_counts import (
            stop_tier_counts,
        )
        self._tier_counts_token = None            # a late count is stale
        stop_tier_counts(getattr(self, "_tier_counts_state", None))
        self._load_token = None                   # a late finish is stale
        self._measure_token = None                # K2: and every other
        self._smooth_token = None
        for attr in self._SHUTDOWN_JOB_ATTRS:
            self._stop_job_thread_for_shutdown(attr)
        # The flags go down with their jobs; nothing is refreshed on a
        # closing panel, but the nav owner Step 4 took is given back.
        for flag in STEP4_BLOCK_FLAGS_ALL:
            if getattr(self, flag, False):
                setattr(self, flag, False)
        try:
            self._sync_step4_nav_lock()
        except Exception as exc:      # noqa: BLE001
            print(f"[STEP4-LOCK] shutdown: nav not released: {exc}")
        timer = getattr(self, "_mesh_timer", None)
        if timer is not None:
            try:
                timer.stop()
            except RuntimeError:
                pass
        proc = getattr(self, "_mesh_proc", None)
        if proc is not None:
            try:
                proc.terminate()
            except OSError:
                pass
            self._mesh_proc = None



    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_source_path(self, stack_name: str, labels_path: str) -> str:
        """Resolve source stack path from the listed stacks (fallback)."""
        for e in getattr(self, "_sources", []) or []:
            if e["name"] == stack_name and Path(e["path"]).exists():
                return e["path"]
        stem = stack_name.replace(".tiff", "").replace(".tif", "")
        for e in getattr(self, "_sources", []) or []:
            if stem and stem in Path(e["path"]).stem:
                return e["path"]
        return ""

    def _update_qc_button_states(self):
        """③'s gating and on-screen reasons."""
        self._refresh_mt_panel()

    # ------------------------------------------------------------------
    # ③ Measure & tidy: the panel's state
    # ------------------------------------------------------------------

    _MT_JOB_FLAGS = ("_measure_running", "_smooth_running",
                     "_save_in_flight", "_load_in_flight", "_export_running",
                     "_mesh_running")

    def _mt_busy(self) -> bool:
        """A ③ job runs (the list rows cannot be clicked, D3)."""
        return any(getattr(self, f, False) for f in self._MT_JOB_FLAGS)

    @staticmethod
    def _about(seconds: float) -> str:
        """``{t}`` of ``BUSY_MEASURE_ETA``: "40 s", "1 min", "3 min"."""
        seconds = max(float(seconds), 0.0)
        if seconds < 60:
            return f"{max(int(round(seconds / 5.0)) * 5, 5)} s"
        return f"{int(round(seconds / 60.0))} min"

    @staticmethod
    def _hhmm(ts) -> str:
        from datetime import datetime
        try:
            return datetime.fromisoformat(str(ts)).astimezone().strftime(
                "%H:%M")
        except (TypeError, ValueError):
            return ""

    def _loaded_record(self) -> dict:
        name = self._current_stack_name
        if not name or self._labels_volume is None:
            return {}
        return self.state_manager.get_record(name)

    def _is_measured(self, rec) -> bool:
        return self._qc_features_df is not None

    def _measure_stale(self, rec) -> bool:
        """The measurements no longer describe the cells: unsaved label
        edits, labels saved after the last Measure, or another voxel size
        (``not same_spacing``)."""
        if not rec or not self._is_measured(rec):
            return False
        # Unsaved label edits make the measurements stale only when they
        # came AFTER the Measure: a Measure's own border removal zeroes
        # cells in the volume it measured (driver 2026-09-29) -- still an
        # unsaved edit, not a stale measurement.  Every edit bumps
        # _labels_epoch; the Measure records the epoch it describes.
        if self._labels_dirty and self._labels_epoch != getattr(
                self, "_measured_epoch", None):
            return True
        changed, measured = rec.get("labels_changed_at"), rec.get(
            "measured_at")
        if changed and measured and str(changed) > str(measured):
            return True
        return self._spacing_changed(rec)

    @staticmethod
    def _spacing_changed(rec) -> bool:
        if "measured_spacing" not in rec or not rec.get("measured_at"):
            return False
        return not S4M.same_spacing(rec.get("spacing"),
                                    rec.get("measured_spacing"))

    def _review_counts(self):
        df = self._qc_features_df
        if df is None:
            return None
        active = df
        if "label_status" in df.columns:
            active = df[df["label_status"] == "active"]
        if "qc_tier" not in active.columns:
            return (0, 0, len(active))
        # The Workbench's ONE bucket rule (qc_plot_widget.wb_tier_bucket):
        # garbage = bad; empty / review / curated = unmarked.
        from .analysis_widget import tier_counts
        counts = tier_counts(active["qc_tier"].tolist())
        return (counts["good"], counts["bad"], counts["unmarked"])

    def _tidy_fields(self, rec):
        pending = self._held_edit_fields(self._current_stack_name or "")
        at = pending.get("smoothed_at", rec.get("smoothed_at"))
        k = pending.get("n_thin_kept", rec.get("n_thin_kept"))
        return at, k

    def _stack_line(self, rec):
        """``MT_STACK_LINE`` / its voxels form (the amber tail apart)."""
        shape = getattr(self._labels_volume, "shape", ()) or ()
        zyx = [int(v) for v in shape[-3:]] if len(shape) >= 3 else [0, 0, 0]
        n = S4T.fmt_count(getattr(self, "_n_cells", 0) or 0)
        sp = rec.get("spacing") if isinstance(rec.get("spacing"),
                                              dict) else None
        vals = None
        if sp and sp.get("zyx"):
            try:
                vals = [float(v) for v in sp["zyx"]]
            except (TypeError, ValueError):
                vals = None
        if vals:
            source = S4T.MT_SOURCE_WORDS.get(sp.get("source"),
                                             S4T.MT_SOURCE_WORDS["file"])
            return S4T.MT_STACK_LINE.format(
                Z=zyx[0], Y=zyx[1], X=zyx[2], n=n, z=f"{vals[0]:.2f}",
                y=f"{vals[1]:.2f}", x=f"{vals[2]:.2f}", source=source), ""
        return (S4T.MT_STACK_LINE_VOXELS_HEAD.format(
            Z=zyx[0], Y=zyx[1], X=zyx[2], n=n),
            S4T.MT_STACK_LINE_VOXELS_TAIL)

    def _not_reconstructed_line(self) -> str:
        rows = [r for r in getattr(self, "_rows", []) or []
                if not r.get("measure_listed")]
        if not rows:
            return ""
        if len(rows) == 1:
            return S4T.MT_NOT_RECONSTRUCTED.format(stack=rows[0]["name"])
        return S4T.MT_NOT_RECONSTRUCTED_MANY.format(
            m=S4T.fmt_count(len(rows)))

    def _refresh_mt_panel(self):
        """Every ③ control's state and its on-screen reason;
        the stack card, the three chips, the Finalize line, the Measure
        button's word, "Edited, not saved", the Export window's mesh
        line."""
        ui = self.ui
        if not hasattr(ui, "show_stack_card"):
            return
        rec = self._loaded_record()
        name = self._current_stack_name if rec or (
            self._labels_volume is not None) else None
        loaded = self._labels_volume is not None and bool(name)
        locked = _step4_is_locked(self)
        busy = self._mt_busy()
        measuring = bool(getattr(self, "_measure_running", False))
        measured = loaded and self._is_measured(rec)
        stale = loaded and self._measure_stale(rec)
        dirty = loaded and self._has_unsaved_changes

        # the stack card
        if loaded:
            if measuring:
                pill, grey = S4T.MT_STACK_PILL_MEASURING, True
            elif self._labels_dirty:
                pill = (S4T.MT_STACK_PILL_UNSAVED if stale or not measured
                        else S4T.MT_STACK_PILL_UNSAVED_FRESH)
                grey = False
            elif self._tiers_dirty:
                pill, grey = S4T.MT_STACK_PILL_UNSAVED_MARKS, False
            elif stale:
                pill, grey = S4T.MT_STACK_PILL_STALE, False
            else:
                pill, grey = S4T.MT_STACK_PILL_SAVED, True
            line, amber = self._stack_line(rec)
            note = (S4T.MT_SPACING_CHANGED
                    if measured and self._spacing_changed(rec) else "")
            ui.show_stack_card(name, pill, pill_grey=grey, line=line,
                               line_amber=amber, note=note)
        else:
            ui.show_stack_card()

        # the chips
        if loaded:
            if measured:
                n = rec.get("measured_cells")
                if n is None and self._qc_features_df is not None:
                    n = len(self._qc_features_df)
                hhmm = self._hhmm(rec.get("measured_at")) or "--:--"
                warn = stale and not measuring
                tpl = S4T.MT_CHIP_MEASURE_STALE if warn else \
                    S4T.MT_CHIP_MEASURE
                measure = S4T.say(tpl, n=int(n or 0), **{"hh:mm": hhmm})
            else:
                warn, measure = False, S4T.MT_CHIP_NEVER
            at, k = self._tidy_fields(rec)
            tidy = (S4T.say(S4T.MT_CHIP_TIDY, k=int(k or 0),
                            **{"hh:mm": self._hhmm(at)})
                    if at else S4T.MT_CHIP_NOT_SMOOTHED)
            ui.show_mt_chips(measure=measure, measure_warn=warn,
                             review=(self._review_counts() if measured
                                     else S4T.MT_CHIP_NEVER), tidy=tidy)
        else:
            ui.show_mt_chips()

        # the buttons; under a job every ③ control is grey except
        # "What is measured?" (D3), and the lock line is the reason
        counts = self._review_counts() or (0, 0, 0)
        has_backup = False
        if loaded:
            orig = rec.get("original_labels_path") or ""
            has_backup = bool(orig) and Path(orig).exists()
        ui.set_measure_button_text(
            S4T.MEASURE_BTN_RUNNING if measuring
            else S4T.MEASURE_BTN_AGAIN if measured
            else S4T.MEASURE_BTN_FIRST)
        live = not locked
        states = {
            "extract_features_btn": loaded and live,
            "open_analysis_btn": measured and live,
            "feature_reference_btn": True,
            "delete_bad_btn": loaded and counts[1] > 0 and live,
            "undo_btn": live,
            "smooth_strength_combo": loaded and live,
            "smooth_all_btn": loaded and live,
            "smooth_sigma_z_sb": loaded and live,
            "smooth_sigma_xy_sb": loaded and live,
            "smooth_preserve_topology_cb": loaded and live,
            "fix_by_hand_btn": False,
            "reset_original_btn": loaded and has_backup and live,
            "save_tiers_btn": loaded and dirty and live,
            "finalize_btn": measured and not stale and live,
            "export_btn": live,
        }
        for attr, on in states.items():
            w = getattr(ui, attr, None)
            if w is not None:
                w.setEnabled(bool(on))
        lst = getattr(ui, "qc_stacks_list", None)
        if lst is not None and hasattr(lst, "setEnabled"):
            lst.setEnabled(not busy)

        # the foot's one amber sentence (hidden under a job: the lock line)
        note = ""
        if loaded and not locked:
            if not measured:
                note = S4T.FINALIZE_NEEDS_MEASURE
            elif stale:
                note = S4T.FINALIZE_NEEDS
        ui.show_finalize_note(note)
        dim = getattr(ui, "set_mt_titles_dimmed", None)
        if callable(dim):
            dim(locked)                                    # G1-10
        ui.show_not_reconstructed(self._not_reconstructed_line())
        ui.set_mt_unsaved(name if dirty else None)
        ui.select_mt_row(name if loaded else None)
        dlg = getattr(ui, "export_dialog", None)
        if dlg is not None:
            dlg.set_mesh_target(name if loaded else None,
                                self._mesh_folder_text(name))
            try:
                if dlg.isVisible():
                    self._refresh_export_window(keep_ticks=True)
            except Exception as exc:      # noqa: BLE001
                print(f"[STEP4-UI] export window not refreshed: {exc}")
        print(f"[STEP4-UI] (3) {name or '-'} measured={measured} stale={stale}"
              f" dirty={dirty} busy={busy} note={note!r}")

    @staticmethod
    def _mesh_folder_text(name) -> str:
        """``{folder}`` of EXPORT_MESH_WHAT: relative to the workspace."""
        return "artifacts/step4/meshes/" + (f"{name}/" if name else "")

    def _track_step4_layers(self, added, *, stack=False):
        """F15: register what an ``add_*`` returned (one layer or a list)
        under the "step4" context by each layer's ACTUAL name (napari
        renames on a collision); *stack* layers are also the loaded
        stack's, which ``_clear_stack_layers`` removes."""
        layers = list(added) if isinstance(added, (list, tuple)) else [added]
        llm = getattr(self, "layer_lifecycle_manager", None)
        mine = getattr(self, "_stack_layers", None)
        if mine is None:
            mine = self._stack_layers = []
        for layer in layers:
            if layer is None:
                continue
            if llm is not None:
                try:
                    llm.register_existing(layer.name, "step4")
                except Exception as exc:      # noqa: BLE001
                    print(f"[STEP4-LAYERS] not registered: {exc}")
            if stack:
                mine.append(layer)
        return added

    def _add_stack_image_layers(self, info):
        """The loaded stack's image layer(s), named, tracked and (per
        channel) tagged with their index -- from a load or a restore."""
        if info["ch_axis"] is not None:
            n_ch = info["n_channels"]
            ch_names = []
            for i in range(n_ch):
                name = ""
                if i < len(self.ui._channel_rows):
                    name = self.ui._channel_rows[i][1].text().strip()
                ch_names.append(STEP4_LAYER_PREFIX + (
                    name or S4T.SEG_CHANNEL_NAME.format(i=i)))
            added = self._track_step4_layers(self.viewer.add_image(
                info["data"], name=ch_names,
                channel_axis=info["ch_axis"]), stack=True)
            self._tag_channel_layers(added)
        else:
            self._track_step4_layers(self.viewer.add_image(
                info["data"], name=STEP4_LAYER_PREFIX + "image"), stack=True)

    def _live_stack_layers(self):
        if self.viewer is None:
            return []
        live = list(self.viewer.layers)
        return [lyr for lyr in (getattr(self, "_stack_layers", None) or [])
                if any(lyr is x for x in live)]

    def _stack_layers_missing(self) -> bool:
        """Is any layer of the loaded stack missing from the viewer (its
        labels, or its image when it had one)?"""
        if self.viewer is None or self._labels_volume is None:
            return False
        if self._find_labels_layer() is None:
            return True
        lab = getattr(self, "_stack_labels_layer", None)
        images = [x for x in self._live_stack_layers() if x is not lab]
        return bool(getattr(self, "_stack_has_image", False)) and not images

    def _restore_stack_layers(self, why: str) -> None:
        """BUG A (owner 2026-10-05): bring back the loaded stack's layers.
        The labels layer from memory (the in-memory, possibly edited array:
        no unsaved-work prompt, ``_labels_dirty`` unchanged); the image
        channels re-read on the ``_StackLoadWorker`` (kind "restore"), with
        ``_load_in_flight`` raised and the usual stale-token guard."""
        stack = self._current_stack_name
        if self.viewer is None or self._labels_volume is None or not stack:
            return
        self._stack_layers = self._live_stack_layers()
        if self._find_labels_layer() is None:
            self._add_stack_labels_layer()
            print(f"[STEP4-LAYERS] {stack} labels re-added ({why})")
            self._reapply_viewer_colours(why)               # W4
        lab = getattr(self, "_stack_labels_layer", None)
        images = [x for x in self._stack_layers if x is not lab]
        if not getattr(self, "_stack_has_image", False) or images:
            self._refresh_channel_combos()
            return
        if _step4_is_locked(self):
            print(f"[STEP4-LOCK] image restore of {stack} refused while "
                  f"{self._running_flag_names()}")
            return
        record = self.state_manager.get_record(stack)
        labels_path = record.get("labels_path", "")
        source = record.get("source_stack_path", "")
        if not source or not Path(source).exists():
            source = self._resolve_source_path(stack, labels_path)
        if not source:
            print(f"[STEP4-LAYERS] {stack}: no source to re-read")
            return
        token = self._next_token()
        worker = _StackLoadWorker(stack_name=stack, labels_path=labels_path,
                                  source_path=source, token=token,
                                  kind="restore")
        worker.finished.connect(self._on_load_worker_finished)
        worker.error.connect(self._on_load_worker_error)
        self._load_launch = {"token": token, "stack": stack,
                             "workspace": self.working_dir, "backup": ""}
        thread = self._start_step4_thread(
            worker, "_load_in_flight", quit_on=("finished", "error"),
            attr="_load", cleanup=self._after_load_thread)
        if thread is None:
            return
        self._load_token = token
        self.ui.show_qc_bar(busy=True, text="")
        print(f"[STEP4-LAYERS] {stack}: image re-read launched ({why}) "
              f"token={token}")

    def _apply_restored(self, stack_name, result):
        """A restore's finish (main thread): the image layers only; the
        labels, the df, the record and the dirty flags are not touched."""
        if stack_name != self._current_stack_name or self.viewer is None:
            return
        info = result.get("raw")
        if info is None:
            self.ui.log_output.append(
                f"[load] {stack_name}: its image could not be read again")
            return
        lab = getattr(self, "_stack_labels_layer", None)
        if [x for x in self._live_stack_layers() if x is not lab]:
            return                          # already back (a second click)
        self._add_stack_image_layers(info)
        # the labels drawn over the image, as after a load
        if lab is not None and lab in self.viewer.layers:
            try:
                self.viewer.layers.move(self.viewer.layers.index(lab),
                                        len(self.viewer.layers))
            except Exception as exc:      # noqa: BLE001
                print(f"[STEP4-LAYERS] labels not moved on top: {exc}")
        self._refresh_channel_combos()
        # W4's colours went on with the labels in _restore_stack_layers; an
        # image re-read does not touch them (audit LOW-4: one pass only).
        print(f"[STEP4-LAYERS] {stack_name}: image layers restored")

    def _reapply_viewer_colours(self, why: str) -> None:
        """W4 (owner live test 2026-10-05): a labels layer re-added from
        memory carries napari's default colormap; with "Colours in viewer"
        ticked it gets the Workbench's colours back -- the same call the
        tick's own handler makes."""
        cb = self._sync_cb
        try:
            on = cb is not None and cb.isChecked()
        except RuntimeError:                  # the Workbench is gone
            return
        if not on or self._find_labels_layer() is None:
            return
        try:
            self._on_sync_toggled(True)
            print(f"[STEP4-LAYERS] viewer colours re-applied ({why})")
        except Exception as exc:      # noqa: BLE001
            print(f"[STEP4-LAYERS] viewer colours not re-applied: {exc}")

    def _add_stack_labels_layer(self):
        """The loaded stack's labels layer (F15 name + registration; audit
        L9's edit hooks).  Also used on re-entry, from memory."""
        lyr = self.viewer.add_labels(self._labels_volume,
                                     name=STEP4_LAYER_PREFIX + "labels")
        self._track_step4_layers(lyr, stack=True)
        self._stack_labels_layer = lyr
        # Audit L9: a hand edit on the layer is a volume change.
        emitter = getattr(lyr.events, "data", None)
        if emitter is not None:
            emitter.connect(self._on_labels_layer_edited)
        emitter = getattr(lyr.events, "paint", None)
        if emitter is not None:
            emitter.connect(self._on_labels_painted)
        lyr.opacity = 1.0
        try:
            lyr.iso_gradient_mode = "smooth"
        except (AttributeError, Exception):
            pass
        return lyr

    def _clear_stack_layers(self):
        """Remove the loaded stack's image and labels layers -- only the
        layers Step 4 added (F15), hidden before removal."""
        if self.viewer is None:
            return
        mine = list(getattr(self, "_stack_layers", None) or [])
        self._stack_layers = []
        self._stack_labels_layer = None
        llm = getattr(self, "layer_lifecycle_manager", None)
        for layer in mine:
            if layer not in self.viewer.layers:
                continue
            name = layer.name
            try:
                layer.visible = False
                self.viewer.layers.remove(layer)
            except Exception as exc:      # noqa: BLE001
                print(f"[STEP4-LAYERS] {name} not removed: {exc}")
            if llm is not None:
                try:
                    llm._unregister(name)
                except Exception:      # noqa: BLE001
                    pass

    def _get_seg_channel(self) -> int:
        return self.ui.chan1_combo.currentIndex()

    def _get_channel_configs(self) -> list:
        return self.ui.get_checked_channels()

    def _get_spacing(self):
        """The loaded stack's voxel size ``(z, y, x)`` from its record, or
        None for a stack with no spacing (in voxels)."""
        return self._spacing_for(self._current_stack_name)

    def _spacing_for(self, stack_name):
        if not stack_name:
            return None
        sp = self.state_manager.get_record(stack_name).get("spacing")
        zyx = sp.get("zyx") if isinstance(sp, dict) else None
        try:
            vals = tuple(float(v) for v in zyx) if zyx else None
        except (TypeError, ValueError):
            return None
        if not vals or len(vals) != 3:
            return None
        # B1b audit B1-1: a user's 1/1/1 is a spacing.  Returning None for
        # it recorded measured_spacing=None against spacing 1/1/1, so the
        # stack read stale for ever (Finalize grey, Export refused).
        return vals

    def _set_default_plot_view(self):
        """Set PCA plot, qc_score color-by, Coolwarm colormap."""
        pw = self._pw
        if pw is None:
            return
        try:
            if hasattr(pw, "plot_type_combo"):
                idx = pw.plot_type_combo.findText("PCA")
                if idx >= 0:
                    pw.plot_type_combo.setCurrentIndex(idx)
            if hasattr(pw, "_cmap_combo"):
                idx = pw._cmap_combo.findText("Coolwarm")
                if idx >= 0:
                    pw._cmap_combo.setCurrentIndex(idx)
            # C3 (2026-10-05): the colour is set LAST and through the same
            # path as a user's pick -- the combo, then one redraw -- so the
            # plot and the viewer never keep the old colouring under a new
            # name (setCurrentIndex alone redraws nothing).
            if (hasattr(pw, "_color_combo") and self._qc_features_df is not None
                    and "qc_score" in self._qc_features_df.columns):
                if hasattr(pw, "set_colour_by"):
                    pw.set_colour_by("qc_score")
                else:
                    idx = pw._color_combo.findText("qc_score")
                    if idx >= 0:
                        pw._color_combo.setCurrentIndex(idx)
        except Exception:
            pass

    def _update_tier_summary(self):
        if self._qc_features_df is None:
            return
        df = self._qc_features_df
        active = df[df.get("label_status", "active") == "active"] if "label_status" in df.columns else df
        if "qc_tier" not in active.columns:
            # No mark yet: the Workbench's totals still count every cell
            # unmarked (B3b shots).
            aw = self._analysis_widget
            if aw is not None and hasattr(aw, "tier_summary_label"):
                from .analysis_widget import totals_line
                aw.tier_summary_label.setText(totals_line([""] * len(active)))
            self._refresh_mt_panel()      # ③'s Review chip, pill and Save
            return
        # The Workbench card's totals line, in Step 4's words
        # (TOTALS_STEP4, "unmarked").
        aw = self._analysis_widget
        if aw is not None and hasattr(aw, "tier_summary_label"):
            from .analysis_widget import totals_line
            aw.tier_summary_label.setText(totals_line(active["qc_tier"]))
        self._refresh_mt_panel()          # ③'s Review chip, pill and Save

    # ------------------------------------------------------------------
    # QC Analysis — Load
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # ③ Load: a row click, on a worker
    # ------------------------------------------------------------------

    def _load_reconstruction(self):
        """Load ③'s highlighted row (a click calls ``_request_load``)."""
        if self._refuse_while_locked("Load"):
            return
        lst = getattr(self.ui, "qc_stacks_list", None)
        items = lst.selectedItems() if lst is not None else []
        if not items:
            return
        self._request_load(items[0].data(Qt.UserRole))

    def _ask_unsaved(self, stack: str) -> str:
        """Save / Discard / Stay: ``"save"``,
        ``"discard"`` or ``"stay"``."""
        from ..ui_shell import theme as th
        choice = th.themed_three_choice(
            self.ui, S4T.LOAD_UNSAVED_TITLE,
            S4T.LOAD_UNSAVED_BODY.format(stack=stack),
            primary_text=S4T.LOAD_UNSAVED_SAVE,
            secondary_text=S4T.LOAD_UNSAVED_DISCARD,
            cancel_text=S4T.LOAD_UNSAVED_STAY, accent="green")
        return {"primary": "save", "secondary": "discard"}.get(choice, "stay")

    def _settle_unsaved(self, what: str) -> bool:
        """Before a load or a workspace change: ask Save / Discard / Stay
        when the loaded stack has unsaved work.  True = go on (saved, or
        discarded and unloaded); False = stay (or a failed Save, F23)."""
        stack = self._current_stack_name
        if not (self._has_unsaved_changes and stack):
            return True
        choice = self._ask_unsaved(stack)
        print(f"[STEP4-LOAD] unsaved {stack} before {what}: {choice}")
        if choice == "stay":
            return False
        if choice == "save":
            return bool(self._save_tier_assignments(confirm=False))
        self._unload_stack(f"discarded before {what}")
        return True

    def _reselect_loaded(self):
        sel = getattr(self.ui, "select_mt_row", None)
        if callable(sel):
            sel(self._current_stack_name
                if self._labels_volume is not None else None)

    def _request_load(self, stack_name):
        """A ③ row click: ask about unsaved work, then read the
        stack on a ``_StackLoadWorker`` with ``_load_in_flight`` raised."""
        if not stack_name:
            return
        if self._refuse_while_locked("Load"):
            self._reselect_loaded()
            return
        if (stack_name == self._current_stack_name
                and self._labels_volume is not None):
            if self._stack_layers_missing():
                self._restore_stack_layers("a click on its row")
            else:
                print(f"[STEP4-LOAD] {stack_name} is already loaded")
            return
        if not self._settle_unsaved("a load"):
            self._reselect_loaded()
            return
        self._ensure_writable_state()
        record = self.state_manager.get_record(stack_name)
        labels_path = record.get("labels_path", "")
        if not labels_path or not Path(labels_path).exists():
            self.ui.log_output.append(
                f"[load] labels not found for {stack_name}: "
                f"{labels_path or '(none)'}")
            self.ui.show_mt_status(S4T.MT_LOAD_FAILED.format(
                stack=stack_name))
            self._reselect_loaded()
            return
        source = record.get("source_stack_path", "")
        if not source or not Path(source).exists():
            source = self._resolve_source_path(stack_name, labels_path)
        backup = str(Path(labels_path).parent
                     / (Path(labels_path).stem + "_original.tif"))
        # B3b, at REQUEST time: a running map stops and the card
        # clears before the previous stack's df is invalidated
        self._workbench_cells_changed("load")
        token = self._next_token()
        worker = _StackLoadWorker(stack_name=stack_name,
                                  labels_path=labels_path,
                                  source_path=source or "",
                                  backup_path=backup, token=token)
        worker.finished.connect(self._on_load_worker_finished)
        worker.error.connect(self._on_load_worker_error)
        self._load_launch = {"token": token, "stack": stack_name,
                             "workspace": self.working_dir,
                             "backup": backup}
        thread = self._start_step4_thread(
            worker, "_load_in_flight", quit_on=("finished", "error"),
            attr="_load", cleanup=self._after_load_thread)
        if thread is None:
            self._reselect_loaded()
            return
        self._load_token = token
        self.ui.show_qc_bar(busy=True, text="")
        print(f"[STEP4-LOAD] {stack_name}: launched token={token}")
        self._refresh_mt_panel()

    def _load_is_stale(self, stack_name, token) -> bool:
        launch = getattr(self, "_load_launch", None) or {}
        return (token != getattr(self, "_load_token", None)
                or stack_name != launch.get("stack")
                or launch.get("workspace") != self.working_dir)

    def _on_load_worker_finished(self, result, stack_name, token):
        if self._load_is_stale(stack_name, token):
            print(f"[STEP4-LOAD] stale load of {stack_name} discarded "
                  f"(token={token})")
            self.ui.log_output.append(
                f"[load] {stack_name}: the result arrived after the "
                f"workspace or the request changed; nothing was loaded")
            return
        try:
            if (result or {}).get("kind") == "restore":
                self._apply_restored(stack_name, result)
            else:
                self._apply_loaded(stack_name, result)
        except Exception as exc:      # noqa: BLE001
            import traceback
            self.ui.log_output.append(
                f"[ERROR] load {stack_name}: {exc}\n{traceback.format_exc()}")
            self.ui.show_mt_status(S4T.MT_LOAD_FAILED.format(
                stack=stack_name))

    def _on_load_worker_error(self, msg, stack_name, token):
        self.ui.log_output.append(f"[ERROR] load {stack_name}: {msg}")
        if self._load_is_stale(stack_name, token):
            return
        self.ui.show_mt_status(S4T.MT_LOAD_FAILED.format(stack=stack_name))
        self._reselect_loaded()

    def _after_load_thread(self):
        """Backstop at thread end: the bar is never left spinning."""
        self._load_token = None
        self.ui.show_qc_bar(busy=False)

    def _forget_loaded_state(self):
        """Everything that belonged to the previously loaded stack."""
        self._qc_features_df = None
        self._last_pca_projected = None
        self._cached_eda_state = None
        self._has_unsaved_changes = False
        self._pending_edits = {}
        self._undo_slot = None
        self._pending_border_removal = False
        self._pending_pca_features = None
        self._pending_pca_mode = None
        self._pending_pca_preset_name = None
        self._last_pca_feature_selection = None
        self._last_pca_preset_name = None
        self._last_pca_mode = None
        self._colour_by_before_pick = None
        self._clear_workbench_caches()

    def _apply_loaded(self, stack_name, result):
        """The load's finish (main thread): the record's backup path, the
        layers, the CSV and the EDA cache as today, the stack card."""
        record = self.state_manager.get_record(stack_name)
        backup = result.get("backup") or ""
        launch_backup = (getattr(self, "_load_launch", None)
                         or {}).get("backup", "")
        if backup:
            self.ui.log_output.append(
                f"[load] Backed up original mask → {Path(backup).name}")
        elif launch_backup and Path(launch_backup).exists():
            backup = launch_backup
        fields = {}
        if backup and record.get("original_labels_path") != backup:
            fields["original_labels_path"] = backup
        source = result.get("source_path") or ""
        if source and record.get("source_stack_path") != source:
            fields["source_stack_path"] = source
        if fields:
            self.state_manager.update_record(stack_name, **fields)
            self._persist_state()

        self._forget_loaded_state()
        self._labels_volume = result["labels"]
        self._labels_epoch += 1
        self._current_stack_name = stack_name
        self._sync_loaded_stack_marker()
        self._n_cells = int(result.get("n_cells") or 0)
        self.state_manager.active_stack = stack_name
        info = result.get("raw")

        self._stack_has_image = info is not None
        if self.viewer is not None:
            self._clear_stack_layers()
            if info is not None:
                self._add_stack_image_layers(info)
            self._add_stack_labels_layer()
        if info is not None:
            n_ch, axes = info["n_channels"], info["axes"]
            self.ui.set_channel_rows(max(int(n_ch), 1), axes_label=axes)
        self._refresh_channel_combos()      # ②'s names follow the viewer

        shape = tuple(int(v) for v in getattr(self._labels_volume, "shape",
                                                ()))
        self.ui.log_output.append(S4T.MT_LOADED.format(
            stack=stack_name, shape=shape,
            n=S4T.fmt_count(self._n_cells)))
        print(f"[STEP4-LOAD] {stack_name}: loaded {shape}, "
              f"{self._n_cells} cells")

        csv_path = record.get("qc_features_csv", "")
        if csv_path and Path(csv_path).exists():
            from napari_epicycle.fusion_reconstruction.qc_features import (
                load_patch_csv,
            )
            self._qc_features_df = load_patch_csv(csv_path)
            self._last_pca_projected = None
            self._restore_eda_state()
            self._open_analysis_dock()
            self._set_default_plot_view()
            self._update_tier_summary()
        sp = (self.state_manager.get_record(stack_name).get("spacing")
              or {})
        self.ui.set_sigma_spacing(sp.get("zyx") if isinstance(sp, dict)
                                  else None)
        self.ui.show_qc_bar(value=0, text="")
        self.ui.show_mt_status("")
        orig = self.state_manager.get_record(stack_name).get(
            "original_labels_path") or ""
        if not orig or not Path(orig).exists():
            self.ui.show_mt_status(S4T.RESET_NO_BACKUP.format(
                stack=stack_name))
        self._refresh_mt_panel()

    # ------------------------------------------------------------------
    # EDA State Persistence
    # ------------------------------------------------------------------

    def _eda_dir(self) -> Path:
        return Path(self.working_dir) / "artifacts" / "step4" / "eda"

    def _save_eda_state(self):
        """Persist EDA analysis state (PCA, UMAP, t-SNE) to disk."""
        if not self.working_dir or not self._current_stack_name:
            self.ui.log_output.append("[eda-save] Skip: no working_dir or stack")
            return
        if self._qc_features_df is None:
            self.ui.log_output.append("[eda-save] Skip: no features DataFrame")
            return

        from napari_epicycle.fusion_reconstruction.qc_features import (
            save_eda_state,
        )

        # Gather PCA result from analysis widget
        pca_result = None
        if self._analysis_widget and hasattr(self._analysis_widget, "_pca_result"):
            pca_result = self._analysis_widget._pca_result

        # Gather plot widget caches (PCA scatter, UMAP, t-SNE)
        pw = self._pw
        pca_scatter_cache = getattr(pw, "_pca_cache", None) if pw else None
        umap_cache = getattr(pw, "_umap_cache", None) if pw else None
        tsne_cache = getattr(pw, "_tsne_cache", None) if pw else None

        self.ui.log_output.append(
            f"[eda-save] PCA={'yes' if pca_result else 'no'}, "
            f"UMAP={'yes' if umap_cache is not None else 'no'}, "
            f"t-SNE={'yes' if tsne_cache is not None else 'no'}, "
            f"pw={'yes' if pw else 'no'}"
        )

        if pca_result is None and umap_cache is None and tsne_cache is None:
            self.ui.log_output.append("[eda-save] Skip: nothing to save")
            return

        # Gather UI state from analysis widget
        ui_state = None
        if self._analysis_widget is not None:
            aw = self._analysis_widget
            is_robust = "Robust" in aw._scaling_combo.currentText()
            ui_state = {
                "selected_features": aw.get_selected_features(),
                "transforms": dict(getattr(aw, "_transforms", {})),
                "preset_name": getattr(self, "_last_pca_preset_name", None),
                "pca_mode": getattr(self, "_last_pca_mode", None),
                "scaling": "qc" if is_robust else "eda",
                "winsorize_enabled": aw._winsorize_cb.isChecked(),
                "winsorize_value": aw._winsorize_spin.value(),
                # Spec: additive, older files load without it.
                "cluster_seed": aw._cluster_seed_spin.value(),
            }

        try:
            save_eda_state(
                self._eda_dir(),
                self._current_stack_name,
                pca_result=pca_result,
                pca_scatter_cache=pca_scatter_cache,
                umap_cache=umap_cache,
                tsne_cache=tsne_cache,
                df=self._qc_features_df,
                ui_state=ui_state,
            )
            self.ui.log_output.append(
                f"[eda] Saved EDA state for {self._current_stack_name}"
            )
        except Exception as exc:
            self.ui.log_output.append(
                f"[WARN] Failed to save EDA state: {exc}"
            )

    def _restore_eda_state(self):
        """Try to restore cached EDA state from disk."""
        if not self.working_dir or not self._current_stack_name:
            self._cached_eda_state = None
            self.ui.log_output.append("[eda-restore] Skip: no working_dir or stack")
            return
        if self._qc_features_df is None:
            self._cached_eda_state = None
            self.ui.log_output.append("[eda-restore] Skip: no features DataFrame")
            return

        from napari_epicycle.fusion_reconstruction.qc_features import (
            load_eda_state,
        )

        eda_dir = self._eda_dir()
        npz_path = eda_dir / f"{self._current_stack_name}_eda.npz"
        json_path = eda_dir / f"{self._current_stack_name}_eda.json"
        self.ui.log_output.append(
            f"[eda-restore] Looking for: {npz_path.name} "
            f"(exists={npz_path.exists()}), {json_path.name} "
            f"(exists={json_path.exists()})"
        )

        try:
            result = load_eda_state(
                eda_dir,
                self._current_stack_name,
                df=self._qc_features_df,
            )
        except Exception as exc:
            self.ui.log_output.append(
                f"[eda-restore] Exception: {exc}"
            )
            result = None

        if result is not None:
            pca = result.get("pca_result")
            umap = result.get("umap_cache")
            tsne = result.get("tsne_cache")
            ui = result.get("ui_state")
            self.ui.log_output.append(
                f"[eda-restore] Loaded: PCA={'yes' if pca else 'no'}, "
                f"UMAP={'yes' if umap is not None else 'no'}, "
                f"t-SNE={'yes' if tsne is not None else 'no'}, "
                f"UI state={'yes' if ui else 'no'}"
            )
            if pca is not None:
                self._last_pca_projected = pca["projected"]
                self._last_pca_residual = "regress_cols_used" in pca
                self._last_pca_key = None
        else:
            self.ui.log_output.append(
                "[eda-restore] No cached state (missing, stale, or corrupt)"
            )
        self._cached_eda_state = result

    def _inject_or_compute_eda(self):
        """Inject cached EDA state into widgets, or auto-compute PCA."""
        self.ui.log_output.append(
            f"[eda-inject] cached_state={'yes' if self._cached_eda_state else 'no'}, "
            f"pw={'yes' if self._pw else 'no'}"
        )
        pw = self._pw
        if self._cached_eda_state is not None and pw is not None:
            eda = self._cached_eda_state
            pca_r = None
            if eda["pca_result"] is not None:
                pca_r = eda["pca_result"]
                self._analysis_widget.show_pca_result(pca_r)
                # Set PC coloring options from restored PCA
                proj = pca_r["projected"]
                pw._pc_projected = proj
                pw.set_pc_color_options(proj.shape[1])
            # Restore plot widget's own scatter PCA cache (validate row count)
            n_rows = len(self._qc_features_df) if self._qc_features_df is not None else 0
            scatter = eda.get("pca_scatter_cache")
            if scatter is not None and scatter.shape[0] == n_rows:
                pw._pca_cache = scatter
            elif pca_r is not None and pca_r["projected"].shape[0] == n_rows:
                # Fallback for old saves: derive from analysis PCA
                pw._pca_cache = np.column_stack([
                    pca_r["projected"][:, 0],
                    pca_r["projected"][:, 1],
                ])
            if eda["umap_cache"] is not None:
                umap = eda["umap_cache"]
                if umap.shape[0] == n_rows:
                    pw._umap_cache = umap
            if eda["tsne_cache"] is not None:
                tsne = eda["tsne_cache"]
                if tsne.shape[0] == n_rows:
                    pw._tsne_cache = tsne
            # Ensure Cluster appears in color-by if present
            if (self._qc_features_df is not None
                    and "cluster_id" in self._qc_features_df.columns):
                if hasattr(pw, "_color_combo"):
                    if pw._color_combo.findText("Cluster") < 0:
                        pw._color_combo.addItem("Cluster")

            # Restore UI state (feature selection, transforms, preset)
            ui_state = eda.get("ui_state")
            if ui_state and self._analysis_widget is not None:
                aw = self._analysis_widget
                saved_features = ui_state.get("selected_features")
                if saved_features:
                    aw.restore_selected_features(saved_features)
                    # Set plot widget features for any future recomputation
                    pw._pca_features = saved_features
                saved_transforms = ui_state.get("transforms")
                if saved_transforms:
                    aw.restore_transforms(saved_transforms)
                # Restore scaling controls
                scaling = ui_state.get("scaling")
                if scaling:
                    idx = 1 if scaling == "qc" else 0
                    aw._scaling_combo.blockSignals(True)
                    aw._scaling_combo.setCurrentIndex(idx)
                    aw._scaling_combo.blockSignals(False)
                win_enabled = ui_state.get("winsorize_enabled")
                if win_enabled is not None:
                    aw._winsorize_cb.blockSignals(True)
                    aw._winsorize_cb.setChecked(win_enabled)
                    aw._winsorize_cb.blockSignals(False)
                win_val = ui_state.get("winsorize_value")
                if win_val is not None:
                    aw._winsorize_spin.blockSignals(True)
                    aw._winsorize_spin.setValue(win_val)
                    aw._winsorize_spin.blockSignals(False)
                seed = ui_state.get("cluster_seed")
                if seed is not None and hasattr(aw, "set_cluster_seed"):
                    aw.set_cluster_seed(seed)
                # The PCA line is show_pca_result's PCA_DONE (B3b: no
                # literal reaches a Workbench status label).
                preset_name = ui_state.get("preset_name")
                pca_mode = ui_state.get("pca_mode")
                self._last_pca_preset_name = preset_name
                self._last_pca_mode = pca_mode

            pw._redraw()
            self._cached_eda_state = None
            self.ui.log_output.append(
                f"[eda] Restored PCA/UMAP/t-SNE from cache"
                + (f" + UI state" if ui_state else "")
            )
        elif self._analysis_widget is not None and self._qc_features_df is not None:
            # No cache — compute PCA using dialog selection or default
            from napari_epicycle.fusion_reconstruction.qc_features import (
                PCA_FEATURES, compute_pca_full,
            )

            # Use dialog selection if available, else fall back to QC preset
            pca_features = getattr(self, "_pending_pca_features", None)
            pca_mode = getattr(self, "_pending_pca_mode", None)
            preset_name = getattr(self, "_pending_pca_preset_name", None)

            # Clear pending state
            self._pending_pca_features = None
            self._pending_pca_mode = None
            self._pending_pca_preset_name = None

            # Audit MED-1 (2026-10-05): with no dialog pending, the PCA runs
            # on what _populate_analysis_features ticked -- the remembered
            # Measure selection -- never on the defaults beside other ticks
            # (a Workbench rebuilt after napari's X with nothing saved yet,
            # or the live dock re-populated).
            if pca_features is None:
                remembered = getattr(self, "_last_pca_feature_selection",
                                     None)
                if remembered is not None:
                    pca_features = list(remembered)
                    pca_mode = getattr(self, "_last_pca_mode", None)
                    preset_name = getattr(self, "_last_pca_preset_name",
                                          None)

            # Skip auto-PCA if user chose "None" preset
            if pca_mode is None and pca_features is not None and len(pca_features) == 0:
                self.ui.log_output.append("[eda] PCA skipped (None preset)")
                return

            # Fall back to EpiCycle QC defaults
            if not pca_features:
                pca_features = PCA_FEATURES
                pca_mode = "qc"
                preset_name = "EpiCycle QC"

            try:
                result = compute_pca_full(
                    self._qc_features_df,
                    n_components=10,
                    mode=pca_mode or "qc",
                    exclude_border=True,
                    features=pca_features,
                )
                self._last_pca_projected = result["projected"]
                self._analysis_widget.show_pca_result(result)

                # Set PC color options on the plot widget
                if self._pw is not None:
                    self._pw._pc_projected = result["projected"]
                    self._pw.set_pc_color_options(result["projected"].shape[1])

                # Persist for later save
                self._last_pca_preset_name = preset_name
                self._last_pca_mode = pca_mode

                mode_label = "robust" if pca_mode == "qc" else "standard"
                self._last_pca_key = None
                self._last_pca_residual = False
                used = set(result['feature_names'])
                requested = set(pca_features)
                missing = sorted(requested - used)
                self.ui.log_output.append(
                    f"[eda] Initial PCA ({preset_name}): 10 PCs, "
                    f"{len(result['feature_names'])}/{len(pca_features)} features ({mode_label})"
                    + (f" — missing: {missing}" if missing else "")
                )
            except Exception as exc:
                self.ui.log_output.append(
                    f"[WARN] PCA failed: {exc}"
                )

    def _invalidate_eda_cache_on_disk(self):
        """Delete on-disk EDA cache when features change."""
        if not self.working_dir or not self._current_stack_name:
            return
        eda_dir = self._eda_dir()
        for ext in (".npz", ".json"):
            path = eda_dir / f"{self._current_stack_name}_eda{ext}"
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # QC Analysis — Feature Extraction
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Analysis Controls dock
    # ------------------------------------------------------------------

    _EDA_DOCK_NAME = "EDA Workbench"

    def _find_eda_dock(self):
        """Find the existing EDA Workbench QDockWidget by title."""
        from qtpy.QtWidgets import QDockWidget

        qt_main = getattr(
            getattr(getattr(self, "viewer", None), "window", None), "_qt_window", None
        )
        if qt_main is None:
            return None
        try:
            from ..annotator.launcher import _dock_is_empty
        except Exception:      # noqa: BLE001
            def _dock_is_empty(_d):
                return False
        try:
            for dock in qt_main.findChildren(QDockWidget):
                if str(dock.windowTitle()) == self._EDA_DOCK_NAME:
                    if _dock_is_empty(dock):
                        continue          # left by napari's X
                    return dock
        except Exception:
            pass
        return None

    def _open_analysis_dock(self):
        """Launch or focus the EDA Workbench as a docked napari widget."""
        from .analysis_widget import AnalysisControlsWidget

        if not getattr(self, "_step4_active", False):
            print("[STEP4-LOCK] Workbench refused: Step 4 is not active")
            return
        # an empty "EDA Workbench" dock left by napari's X is purged
        # before the Workbench is shown again.
        if self.viewer is not None:
            try:
                from ..annotator.launcher import _purge_empty_docks
                _purge_empty_docks(self.viewer, self._EDA_DOCK_NAME)
            except Exception as exc:      # noqa: BLE001
                print(f"[STEP4-UI] empty Workbench dock not purged: {exc}")
        # If already open, just raise the existing dock
        existing_dock = self._find_eda_dock()
        if existing_dock is not None:
            try:
                existing_dock.show()
                existing_dock.raise_()
                # Re-populate in case data changed
                self._analysis_widget = (
                    existing_dock.widget()
                    if hasattr(existing_dock, "widget")
                    else self._analysis_widget
                )
                self._populate_analysis_features()
                # Ensure QCPlotWidget has data too
                if self._qc_features_df is not None and self._pw is not None:
                    pw = self._pw
                    n_rows = len(self._qc_features_df)
                    has_cache = bool(
                        self._cached_eda_state
                        and (self._cached_eda_state.get("pca_scatter_cache") is not None
                             or self._cached_eda_state.get("pca_result"))
                    )
                    # Pre-inject cached PCA BEFORE set_features redraws
                    # Validate row count to prevent stale cache from wrong stack
                    if has_cache:
                        scatter = self._cached_eda_state.get("pca_scatter_cache")
                        if scatter is not None and scatter.shape[0] == n_rows:
                            pw._pca_cache = scatter
                        elif self._cached_eda_state.get("pca_result"):
                            pca_r = self._cached_eda_state["pca_result"]
                            proj = pca_r["projected"]
                            if proj.shape[0] == n_rows:
                                pw._pca_cache = np.column_stack([
                                    proj[:, 0], proj[:, 1],
                                ])
                            else:
                                has_cache = False
                        else:
                            has_cache = False
                    pw.set_features(
                        self._qc_features_df, preserve_caches=has_cache,
                    )
                    self._set_default_plot_view()
                    self._inject_or_compute_eda()
                self._update_tier_summary()           # W5
                return
            except RuntimeError:
                pass

        self._analysis_widget = AnalysisControlsWidget()
        self._analysis_widget.setMinimumWidth(500)

        # Populate features from DataFrame + QCPlotWidget
        if self._qc_features_df is not None:
            self._populate_analysis_features()

        # Connect analysis signals
        self._analysis_widget.compute_pca_requested.connect(
            self._on_compute_pca
        )
        self._analysis_widget.compute_clusters_requested.connect(
            self._on_compute_clusters
        )
        self._analysis_widget.compute_umap_requested.connect(
            self._on_compute_umap
        )
        self._analysis_widget.compute_tsne_requested.connect(
            self._on_compute_tsne
        )
        self._analysis_widget.labels_gated.connect(
            self._on_eda_gate_selection
        )
        self._analysis_widget.color_by_feature.connect(
            self._on_eda_color_by_feature
        )

        # Connect triage signals (Mark Good/Bad now live in EDA)
        self._analysis_widget.mark_good_requested.connect(self._mark_good)
        self._analysis_widget.mark_bad_requested.connect(self._mark_bad)
        # B3b
        self._analysis_widget.unmark_requested.connect(self._unmark)
        self._analysis_widget.undo_mark_requested.connect(self._undo_mark)
        self._analysis_widget.stop_map_requested.connect(self._on_stop_map)

        # Connect sync viewer colors
        self._analysis_widget.sync_viewer_toggled.connect(
            self._on_sync_toggled
        )

        # Connect QCPlotWidget signals (lasso, color, category)
        pw = self._analysis_widget.qc_plot_widget
        pw.labels_selected.connect(self._on_lasso_selection)
        pw.viewer_color_requested.connect(self._on_viewer_color_requested)
        pw.category_selected.connect(self._on_category_selected)

        # Populate QCPlotWidget with feature data
        if self._qc_features_df is not None:
            has_cache = bool(
                self._cached_eda_state
                and self._cached_eda_state.get("pca_result")
            )
            # Pre-inject cached PCA BEFORE set_features redraws,
            # so the internal PCA doesn't overwrite the saved one
            if has_cache:
                pca_r = self._cached_eda_state["pca_result"]
                pw._pca_cache = np.column_stack([
                    pca_r["projected"][:, 0],
                    pca_r["projected"][:, 1],
                ])
            pw.set_features(
                self._qc_features_df, preserve_caches=has_cache,
            )
            self._set_default_plot_view()
            self._inject_or_compute_eda()

        # Dock via napari — proper Qt parenting, tabbable
        window = getattr(self.viewer, "window", None)
        if window is not None:
            # W1 (owner live test 2026-10-05): no napari stretch.  napari
            # appends addStretch(next(counter)) to a docked widget's root
            # layout, the counter module-wide: a Workbench built after any
            # other stretched dock (or after napari's X) got a stretch >= 1
            # that took half the dock from the content area (stretch 1).
            window.add_dock_widget(
                self._analysis_widget,
                name=self._EDA_DOCK_NAME,
                area="right",
                add_vertical_stretch=False,
            )
            # Tabify with the main EpiCycle panel
            dock = self._find_eda_dock()
            if dock is not None:
                self._tabify_eda_dock(dock)
        else:
            # Fallback: floating window (testing / headless)
            self._analysis_widget.setWindowFlags(
                Qt.Window | Qt.WindowStaysOnTopHint
            )
            self._analysis_widget.resize(900, 700)
            self._analysis_widget.show()
        self._update_tier_summary()                   # W5: every build
        # Audit LOW-3: the session's state, not the widget's defaults --
        # Undo while marks have a history, the lock line, a running map.
        self._wb_sync_mark_state()
        self._refresh_workbench_lock()
        self._wb_push_running_map()

    def _wb_push_running_map(self):
        """A map still running when this Workbench was built (napari's X
        during a UMAP): its busy strip, Stop and BUSY_ONE, as at launch."""
        aw = self._analysis_widget
        if (not getattr(self, "_eda_running", False) or aw is None
                or not hasattr(aw, "set_busy_strip")):
            return
        launch = self._eda_launch or {}
        kind = launch.get("kind")
        word = self._eda_word(kind)
        if getattr(self, "_eda_stopping", False) or getattr(
                self, "_eda_abandoned", False):
            aw.set_busy_strip(T_WB().STOPPING.format(map=word))
            aw.busy_stop_btn.setEnabled(False)
        else:
            warmed = getattr(self, "_eda_warmed", None) or set()
            rate = (getattr(self, "_eda_rates", None) or {}).get(kind)
            aw.set_busy_strip(T_WB().busy_text(
                kind, int(launch.get("n") or 0), rate,
                warmed=kind in warmed))
            aw.busy_stop_btn.setEnabled(True)
            pw = self._pw
            if kind in ("umap", "tsne") and hasattr(pw, "set_map_busy"):
                pw.set_map_busy(word)
        aw.set_map_running(word)
        print(f"[EDA] {word} still running: shown on the rebuilt Workbench")

    def _on_workbench_button(self):
        """③'s Workbench button (W3, owner live test 2026-10-05): it SHOWS
        the Workbench and never resets it.

        A live Workbench (docked, or hidden by a step exit) is shown and
        raised -- no ``set_features``, no PCA, no job.  Only when there is
        none (napari's X took it) is one built, and then it is filled from
        the loaded stack's SAVED EDA state, the way a load fills it.  The
        other doors keep their own rules: a load fills from the new stack's
        saved state, a Measure's finish starts from the new table.
        """
        if not getattr(self, "_step4_active", False):
            print("[STEP4-LOCK] Workbench refused: Step 4 is not active")
            return
        aw = self._analysis_widget
        alive = aw is not None
        if alive:
            try:
                from qtpy import sip
                alive = not sip.isdeleted(aw)
            except Exception:      # noqa: BLE001
                pass
        dock = self._find_eda_dock()
        try:
            if alive and dock is not None and dock.widget() is aw:
                dock.show()
                dock.raise_()
                self._update_tier_summary()
                print("[STEP4-UI] Workbench shown (button): nothing "
                      "recomputed")
                return
            if (alive and getattr(self.viewer, "window", None) is None
                    and aw.isWindow()):
                aw.show()                     # the headless floating window
                aw.raise_()
                self._update_tier_summary()
                return
        except RuntimeError:
            pass
        old = self._wb_snapshot(aw) if alive else None
        if self._qc_features_df is not None and self._current_stack_name:
            self._restore_eda_state()
        lost = (self._qc_features_df is not None
                and self._wb_lost_since_save(old, self._cached_eda_state))
        print("[STEP4-UI] Workbench built (button): filled from the saved "
              f"state (unsaved maps or ticks lost: {lost})")
        self._open_analysis_dock()
        if lost:
            # Audit LOW-2: napari's X dropped work the user never saved.
            self._card_message(T_WB().WB_REBUILT_FROM_SAVED, "warn")

    @staticmethod
    def _wb_snapshot(aw):
        """What a closing Workbench held (its ticks, its PCA's features,
        its UMAP / t-SNE), or None when it cannot be read."""
        try:
            pw = aw.qc_plot_widget
            pca = getattr(aw, "_pca_result", None)
            return {"ticks": sorted(aw.get_selected_features()),
                    "pca": (sorted(pca["feature_names"]) if pca else None),
                    "umap": getattr(pw, "_umap_cache", None),
                    "tsne": getattr(pw, "_tsne_cache", None)}
        except Exception:      # noqa: BLE001
            return None

    @staticmethod
    def _wb_lost_since_save(old, saved) -> bool:
        """Did the closed Workbench hold something the saved EDA state
        does not?  Unknown (``old`` None) counts as lost."""
        if old is None:
            return True
        saved = saved or {}

        def differs(mine, theirs):
            if mine is None:
                return False
            if theirs is None:
                return True
            mine, theirs = np.asarray(mine), np.asarray(theirs)
            return mine.shape != theirs.shape or not np.allclose(
                mine, theirs, equal_nan=True)

        if differs(old["umap"], saved.get("umap_cache")):
            return True
        if differs(old["tsne"], saved.get("tsne_cache")):
            return True
        if not saved:
            return False         # nothing saved: the PCA follows the ticks
        ui = saved.get("ui_state") or {}
        ticks = ui.get("selected_features")
        if ticks and sorted(ticks) != old["ticks"]:
            return True
        pca = saved.get("pca_result")
        theirs = sorted(pca["feature_names"]) if pca else None
        return old["pca"] is not None and old["pca"] != theirs

    def _tabify_eda_dock(self, eda_dock):
        """Tab the EDA dock alongside the main EpiCycle dock widget."""
        from qtpy.QtWidgets import QDockWidget

        qt_main = getattr(
            getattr(self.viewer, "window", None), "_qt_window", None
        )
        if qt_main is None:
            return
        # Find the main EpiCycle plugin dock to tabify with
        try:
            for dock in qt_main.findChildren(QDockWidget):
                title = str(dock.windowTitle() or "")
                if title == self._EDA_DOCK_NAME:
                    continue
                if "epicycle" in title.lower() or "EpiCycle" in title:
                    qt_main.tabifyDockWidget(dock, eda_dock)
                    eda_dock.show()
                    eda_dock.raise_()
                    return
        except Exception:
            pass
        # No host found — just show it
        eda_dock.show()
        eda_dock.raise_()

    def _populate_analysis_features(self):
        """Populate the analysis widget with current DataFrame features."""
        if self._analysis_widget is None or self._qc_features_df is None:
            return
        # Use persistent selection (survives _inject_or_compute_eda clearing)
        pca_override = getattr(self, "_last_pca_feature_selection", None)
        # Fall back to cached UI state if no dialog-based override
        if pca_override is None and self._cached_eda_state is not None:
            ui_state = self._cached_eda_state.get("ui_state")
            if ui_state:
                pca_override = ui_state.get("selected_features")
            elif self._cached_eda_state.get("pca_result"):
                # v2 fallback: use PCA feature_names
                pca_override = self._cached_eda_state["pca_result"].get(
                    "feature_names"
                )
        self._analysis_widget.set_feature_data(
            self._qc_features_df, exclude_border=True,
            pca_features_override=pca_override,
        )

    # ------------------------------------------------------------------
    # Workbench maps on a worker
    # ------------------------------------------------------------------
    #
    # PCA, UMAP, t-SNE and the grouping run on ONE ``_EdaWorker`` at a
    # time (``_eda_running``, which is NOT a nav-lock flag): the heavy
    # bodies live in ``_eda_worker.py`` and work on copies.  Marks stay
    # live meanwhile and never invalidate the running map: its token is
    # the stack, the ticked features and the map's parameters -- no df
    # counter (I-17).  Stop is a STATE (``_eda_stopping``): the fit runs
    # to its end, its answer is dropped, the last map is kept.

    @staticmethod
    def _pca_axes(params):
        """The two PCs to draw (never the same one twice)."""
        n_components = params.get("n_components", 10)
        pc_x = params.get("pc_x", 0)
        pc_y = params.get("pc_y", 1)
        if pc_x == pc_y:
            pc_y = min(pc_x + 1, n_components - 1)
            if pc_y == pc_x:
                pc_x = max(pc_y - 1, 0)
        return pc_x, pc_y

    @staticmethod
    def _params_key(params):
        """A hashable form of a map's parameters (the PC axes are a view,
        not a parameter)."""
        def freeze(v):
            if isinstance(v, dict):
                return tuple(sorted((k, freeze(x)) for k, x in v.items()))
            if isinstance(v, (list, tuple)):
                return tuple(freeze(x) for x in v)
            return v
        return freeze({k: v for k, v in params.items()
                       if k not in ("pc_x", "pc_y")})

    def _on_compute_pca(self, params):
        """Compute PCA (standard or residual) -- or, when only the PC axes
        changed since the last PCA, just draw the new axes."""
        if self._qc_features_df is None:
            return
        if len(params.get("features") or []) < 2:
            self._wb_map_error(T_WB().PCA_TOO_FEW)
            return
        if (self._params_key(params) == self._last_pca_key
                and self._last_pca_projected is not None):
            self._show_pca_axes(*self._pca_axes(params))
            return
        self._launch_eda("pca", params)

    def _on_compute_umap(self, params):
        if self._qc_features_df is None:
            return
        self._launch_eda("umap", params)

    def _on_compute_tsne(self, params):
        if self._qc_features_df is None:
            return
        self._launch_eda("tsne", params)

    def _on_compute_clusters(self, params):
        """Find groups: CLUSTER_NEEDS_UMAP when grouping from
        a UMAP that does not exist; the seed rides in ``params``."""
        if self._qc_features_df is None:
            return
        if len(self._qc_features_df) < 2:
            self._wb_groups_result(T_WB().MAP_TOO_FEW_CELLS.format(
                map=T_WB().MAP_WORDS["cluster"]), "warn")
            return
        if (params.get("input_space") == "UMAP"
                and getattr(self._pw, "_umap_cache", None) is None):
            self._wb_groups_result(T_WB().CLUSTER_NEEDS_UMAP, "warn")
            return
        self._launch_eda("cluster", params)

    def _show_pca_axes(self, pc_x, pc_y):
        """Draw two other PCs of the stored PCA (main thread, cheap)."""
        projected = self._last_pca_projected
        pw = self._pw
        if projected is None or pw is None:
            return
        n = projected.shape[1]
        pw._pca_cache = np.column_stack([
            projected[:, pc_x] if pc_x < n else np.zeros(len(projected)),
            projected[:, pc_y] if pc_y < n else np.zeros(len(projected)),
        ])
        pw._redraw()

    def _eda_word(self, kind=None):
        launch = self._eda_launch or {}
        return T_WB().MAP_WORDS.get(kind or launch.get("kind"), "")

    def _launch_eda(self, kind: str, params: dict):
        """Start one map job on a worker, or say why not."""
        aw = self._analysis_widget
        if not getattr(self, "_step4_active", False):
            print(f"[EDA] {kind} refused: Step 4 is not the active step")
            return
        if self._refuse_while_locked(f"Compute {kind}"):
            self._refresh_workbench_lock()
            return
        if getattr(self, "_eda_running", False):
            word = self._eda_word()
            print(f"[EDA] {kind} refused: {word} is still running")
            if aw is not None and hasattr(aw, "set_map_running"):
                aw.set_map_running(word)
            return
        df = self._qc_features_df
        from ._eda_worker import PCA_INPUTS, EdaWorker
        word = self._eda_word(kind)
        uses_pca = (params.get("input_space", "PCA") in PCA_INPUTS
                    and self._last_pca_projected is not None)
        uses_umap = kind == "cluster" and params.get("input_space") == "UMAP"
        if (kind != "pca" and not uses_pca and not uses_umap
                and not params.get("features")):
            # Audit E1-4: never start a worker that can only refuse.
            self._wb_map_error(T_WB().MAP_NEEDS_FEATURES.format(map=word),
                               kind=kind)
            return
        inputs = {"df": df.copy(deep=True)}
        if self._last_pca_projected is not None:
            inputs["pca_projected"] = np.array(self._last_pca_projected,
                                               copy=True)
            inputs["pca_residual"] = bool(
                getattr(self, "_last_pca_residual", False))
        umap = getattr(self._pw, "_umap_cache", None)
        if kind == "cluster" and umap is not None:
            inputs["umap"] = np.array(umap, copy=True)
        params = dict(params)
        self._eda_serial = int(getattr(self, "_eda_serial", 0)) + 1
        token = (self._current_stack_name,
                 tuple(params.get("features") or ()),
                 self._params_key(params), self._eda_serial)
        worker = EdaWorker(kind, params, inputs, token)
        worker.finished.connect(self._on_eda_finished)
        worker.error.connect(self._on_eda_error)
        import time as _time
        pw = self._pw
        self._eda_launch = {"kind": kind, "token": token, "params": params,
                            "stack": self._current_stack_name,
                            "n": int(len(df)), "t0": _time.monotonic(),
                            "plot": (pw.plot_type_combo.currentText()
                                     if hasattr(pw, "plot_type_combo")
                                     else None)}
        self._eda_stopping = False
        self._eda_abandoned = False
        thread = self._start_step4_thread(
            worker, "_eda_running", quit_on=("finished", "error"),
            attr="_eda", cleanup=self._on_eda_thread_done)
        if thread is None:
            self._eda_launch = None
            return
        print(f"[EDA] {word} started on a worker ({len(df)} cells)")
        if aw is not None and hasattr(aw, "set_busy_strip"):
            warmed = getattr(self, "_eda_warmed", None)
            if warmed is None:
                warmed = self._eda_warmed = set()
            rate = getattr(self, "_eda_rates", {}).get(kind)
            self._eda_launch["warmup"] = (
                kind == "umap" and rate is None and kind not in warmed)
            aw.set_busy_strip(T_WB().busy_text(
                kind, len(df), rate, warmed=kind in warmed))
            aw.busy_stop_btn.setEnabled(True)
            aw.set_map_running(word)
            aw.show_map_error("")
        pw = self._pw
        if kind in ("umap", "tsne") and hasattr(pw, "set_map_busy"):
            pw.set_map_busy(word)

    def _on_stop_map(self):
        """■ Stop: a STATE, never a pump.  The fit cannot be
        interrupted, so it runs to its end; its answer is dropped and a
        new map waits for the thread (BUSY_ONE)."""
        if not getattr(self, "_eda_running", False) or self._eda_stopping:
            return
        self._eda_stopping = True
        word = self._eda_word()
        print(f"[EDA] Stop pressed: {word} will be dropped when it ends")
        aw = self._analysis_widget
        if aw is not None and hasattr(aw, "set_busy_strip"):
            aw.set_busy_strip(T_WB().STOPPING.format(map=word))
            aw.busy_stop_btn.setEnabled(False)
        pw = self._pw
        if hasattr(pw, "set_map_busy"):
            pw.set_map_busy(None)          # the last map (or its empty text)

    def _abandon_eda(self, reason: str):
        """A change of the cells stops a running map:
        LOG_MAP_ABANDONED, and its answer will be dropped."""
        if not getattr(self, "_eda_running", False):
            return
        if self._eda_abandoned:
            return
        # Even after Stop (audit E1-3): no STOPPED lands once the cells
        # changed.
        self._eda_abandoned = True
        word = self._eda_word()
        line = T_WB().LOG_MAP_ABANDONED.format(map=word)
        print(f"[EDA] {line} ({reason})")
        try:
            self.ui.log_output.append(f"[analysis] {line}")
        except Exception:      # noqa: BLE001
            pass
        aw = self._analysis_widget
        if aw is not None and hasattr(aw, "set_busy_strip"):
            aw.set_busy_strip(T_WB().STOPPING.format(map=word))
            aw.busy_stop_btn.setEnabled(False)
        pw = self._pw
        if hasattr(pw, "set_map_busy"):
            pw.set_map_busy(None)

    def _eda_outcome_belongs(self, info) -> bool:
        """The finish's guard (main thread): the launch it answers is the
        live one, on the stack still loaded, and nobody stopped it."""
        launch = self._eda_launch
        if launch is None or info.get("token") != launch.get("token"):
            print("[EDA] a stale answer was dropped (another launch)")
            return False
        if self._eda_stopping or self._eda_abandoned:
            print(f"[EDA] {self._eda_word()} dropped: it was stopped")
            return False
        if (launch.get("stack") != self._current_stack_name
                or self._qc_features_df is None):
            print("[EDA] a stale answer was dropped (the stack changed)")
            return False
        return True

    def _on_eda_finished(self, outcome):
        if not self._eda_outcome_belongs(outcome):
            return
        kind = outcome["kind"]
        try:
            {"pca": self._apply_pca_outcome,
             "umap": self._apply_embedding_outcome,
             "tsne": self._apply_embedding_outcome,
             "cluster": self._apply_cluster_outcome}[kind](outcome)
        except Exception as exc:      # noqa: BLE001
            print(f"[EDA] applying {kind} failed: {exc}")
            self.ui.log_output.append(f"[ERROR] {kind}: {exc}")
            self._wb_map_error(T_WB().MAP_FAILED.format(
                map=self._eda_word(kind)), kind=kind)
            return
        if outcome.get("log"):
            self.ui.log_output.append(outcome["log"])

    def _on_eda_error(self, info):
        if not self._eda_outcome_belongs(info):
            return
        kind = info.get("kind")
        word = self._eda_word(kind)
        if self._eda_launch is not None:
            self._eda_launch["failed"] = True
        print(f"[EDA] {word} failed: {info.get('message')}")
        self.ui.log_output.append(
            f"[ERROR] {word}: {info.get('message')}")
        sentence = (T_WB().MAP_TOO_FEW_CELLS if info.get("too_few")
                    else T_WB().MAP_FAILED).format(map=word)
        self._wb_map_error(sentence, kind=kind)

    def _on_eda_thread_done(self):
        """The thread's ``finished`` (the backstop): the busy strip goes,
        Compute / Cluster come back, and a stopped map says STOPPED."""
        launch = self._eda_launch or {}
        word = self._eda_word()
        stopped = self._eda_stopping and not self._eda_abandoned
        if launch.get("warmup") and not launch.get("failed"):
            # The compile is not a rate (re-audit MED-1): remember only
            # that the session is warm.
            warmed = getattr(self, "_eda_warmed", None)
            if warmed is None:
                warmed = self._eda_warmed = set()
            warmed.add(launch["kind"])
        elif launch.get("t0") is not None and not launch.get("failed"):
            import time as _time
            n = max(1, int(launch.get("n") or 1))
            rates = getattr(self, "_eda_rates", None)
            if rates is None:
                rates = self._eda_rates = {}
            rates[launch["kind"]] = (_time.monotonic() - launch["t0"]) / n
        aw = self._analysis_widget
        if aw is not None and hasattr(aw, "set_busy_strip"):
            aw.set_busy_strip(None)
            aw.set_map_running(None)
            if stopped:
                sentence = T_WB().STOPPED.format(map=word)
                if launch.get("kind") == "cluster":
                    aw.show_groups_result(sentence, "warn")
                else:
                    aw.show_map_result(sentence)
                # On screen where the user pressed Stop (audit X1-3).
                if hasattr(aw, "flash_busy_strip"):
                    aw.flash_busy_strip(sentence)
        pw = self._pw
        if hasattr(pw, "set_map_busy"):
            pw.set_map_busy(None)
        self._eda_stopping = False
        self._eda_abandoned = False
        self._eda_launch = None

    def _wb_map_error(self, text, *, kind=None):
        aw = self._analysis_widget
        if kind == "cluster":
            self._wb_groups_result(text, "warn")
        elif aw is not None and hasattr(aw, "show_map_error"):
            aw.show_map_error(text)

    def _wb_groups_result(self, text, kind="ok"):
        aw = self._analysis_widget
        if aw is not None and hasattr(aw, "show_groups_result"):
            aw.show_groups_result(text, kind)

    def _pca_still_matches(self, params) -> bool:
        """The ticked features are still the ones the PCA was made from
        (a residual PCA leaves its regress-out features out)."""
        aw = self._analysis_widget
        if aw is None or not hasattr(aw, "get_selected_features"):
            return True
        now = set(aw.get_selected_features())
        made = set(params.get("features") or [])
        left_out = set(params.get("regress_out") or [])
        return made <= now and (now - made) <= left_out

    def _may_switch_plot(self) -> bool:
        """A landing map takes the plot only if the user is still on the
        plot they launched from and has nothing selected (audit X1-2);
        otherwise MAP_DONE / PCA_DONE offer it."""
        pw = self._pw
        launch = self._eda_launch or {}
        if pw is None or not hasattr(pw, "plot_type_combo"):
            return False
        if pw.plot_type_combo.currentText() != launch.get("plot"):
            return False
        return not self._selected_label_ids

    def _apply_pca_outcome(self, outcome):
        launch = self._eda_launch
        params = launch["params"]
        result = outcome["result"]
        projected = result["projected"]
        if len(projected) != len(self._qc_features_df):
            raise ValueError("the PCA's rows are not the table's")
        self._last_pca_projected = projected
        self._last_pca_residual = "regress_cols_used" in result
        self._last_pca_key = self._params_key(params)
        pc_x, pc_y = self._pca_axes(params)
        pw = self._pw
        if pw is not None:
            n = projected.shape[1]
            pw._pca_cache = np.column_stack([
                projected[:, pc_x] if pc_x < n else np.zeros(len(projected)),
                projected[:, pc_y] if pc_y < n else np.zeros(len(projected)),
            ])
            pw._pc_projected = projected
            pw.set_pc_color_options(n)
            if self._may_switch_plot():
                idx = pw.plot_type_combo.findText("PCA")
                if idx >= 0:
                    pw.plot_type_combo.setCurrentIndex(idx)
            pw._redraw()
        aw = self._analysis_widget
        if aw is not None:
            aw.show_pca_result(result, pc_x=pc_x, pc_y=pc_y)
            # A new PCA marks the UMAP and t-SNE as made from the previous
            # one -- and does not clear them.
            old = pw is not None and (
                getattr(pw, "_umap_cache", None) is not None
                or getattr(pw, "_tsne_cache", None) is not None)
            if hasattr(aw, "set_maps_old"):
                aw.set_maps_old(bool(old))
            # A feature, scaling, clip or Rescale change made while the
            # PCA ran (audit E1-1): the landed map is already stale.
            now = (aw.current_pca_params()
                   if hasattr(aw, "current_pca_params") else None)
            if (not self._pca_still_matches(params) or now is None
                    or self._params_key(now) != self._params_key(params)):
                aw._mark_embeddings_stale()
        log = outcome.get("log", "")
        transforms = params.get("transforms")
        if transforms:
            outcome["log"] = log + self._format_transform_log(transforms)

    def _apply_embedding_outcome(self, outcome):
        kind = outcome["kind"]
        word = self._eda_word(kind)
        projected = outcome["projected"]
        if len(projected) != len(self._qc_features_df):
            raise ValueError("the map's rows are not the table's")
        pw = self._pw
        if pw is not None:
            setattr(pw, "_umap_cache" if kind == "umap" else "_tsne_cache",
                    projected)
            if hasattr(pw, "set_map_input"):
                pw.set_map_input(word, outcome.get("input", ""))
            took = self._may_switch_plot()
            if took:
                idx = pw.plot_type_combo.findText(word)
                if idx >= 0:
                    pw.plot_type_combo.setCurrentIndex(idx)
            pw._redraw()
            on_it = pw.plot_type_combo.currentText() == word
        else:
            on_it = True
        aw = self._analysis_widget
        if aw is not None and hasattr(aw, "show_map_result"):
            aw.show_map_result(
                (T_WB().MAP_DONE if on_it else T_WB().MAP_DONE_PICK).format(
                    map=word, input=outcome.get("input", "")))
            aw.set_maps_old(False)

    def _apply_cluster_outcome(self, outcome):
        labels = outcome["labels"]
        df = self._qc_features_df
        if len(labels) != len(df):
            raise ValueError("the groups' rows are not the table's")
        df["cluster_id"] = labels
        pw = self._pw
        if pw is not None:
            # Only a column was added: no set_features (which would drop
            # the maps).
            pw._df = df
            if hasattr(pw, "_color_combo"):
                if pw._color_combo.findText("Cluster") < 0:
                    pw._color_combo.addItem("Cluster")
                pw._color_combo.setCurrentText("Cluster")
            pw._redraw()
        if outcome.get("algorithm") == "hdbscan":
            line = T_WB().CLUSTER_DONE_HDB.format(
                k=outcome["n_groups"], noise=f"{outcome['noise']:,}")
        else:
            line = T_WB().CLUSTER_DONE.format(
                k=outcome["n_groups"],
                method=T_WB().CLUSTER_METHOD_WORDS.get(
                    outcome.get("algorithm"), outcome.get("algorithm")),
                seed=outcome.get("seed"))
        self._wb_groups_result(line, "ok")
        aw = self._analysis_widget
        if aw is not None and hasattr(aw, "refresh_groups_legend"):
            aw.refresh_groups_legend()

    # ------------------------------------------------------------------
    # Workbench on a load, a Reset, a Remove, a Measure
    # ------------------------------------------------------------------

    def _workbench_cells_changed(self, reason: str):
        """The cells themselves changed: stop a running map, clear the
        card and the mark history, drop the map results (the plot's own
        caches go with the caller's ``set_features``); the seed stays."""
        aw = getattr(self, "_analysis_widget", None)
        if aw is not None and hasattr(aw, "set_busy_strip"):
            aw.set_busy_strip(None)          # a STOPPED flash goes (LOW-1)
        self._abandon_eda(reason)
        self._mark_history = []
        self._selected_label_ids = []
        self._last_pca_key = None
        self._card_message("")
        self._refresh_selection_card()
        aw = getattr(self, "_analysis_widget", None)
        if aw is not None and hasattr(aw, "clear_map_results"):
            aw.clear_map_results()

    @staticmethod
    def _format_transform_log(transforms: dict) -> str:
        """Format per-feature transforms as a compact log string."""
        by_method = {}
        for feat, method in transforms.items():
            by_method.setdefault(method, []).append(feat)
        parts = []
        for method, feats in sorted(by_method.items()):
            parts.append(f"{method}({len(feats)}): {', '.join(feats)}")
        return "\n    transforms: " + " | ".join(parts)

    def _show_feature_reference(self):
        """Show a styled dialog listing all QC features."""
        from qtpy.QtWidgets import (
            QDialog, QVBoxLayout, QDialogButtonBox,
        )
        from qtpy.QtCore import QUrl
        try:
            from qtpy.QtWidgets import QTextBrowser
        except ImportError:
            from qtpy.QtWidgets import QTextEdit as QTextBrowser

        dlg = QDialog()
        dlg.setWindowTitle(S4T.FEATURE_REFERENCE_TITLE)
        dlg.resize(760, 620)
        dlg.setStyleSheet("background: #1a1a2e; color: #e0e0e0;")
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(0, 0, 0, 8)

        browser = QTextBrowser()
        browser.setOpenExternalLinks(False)
        browser.setStyleSheet(
            "QTextBrowser { border: none; background: #1a1a2e; }"
        )
        browser.setHtml(_FEATURE_REFERENCE_HTML)
        lay.addWidget(browser)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.setStyleSheet(
            "QPushButton { background: #2d2d4e; color: #e0e0e0; "
            "border: 1px solid #444; padding: 6px 20px; border-radius: 3px; }"
            "QPushButton:hover { background: #3d3d5e; }"
        )
        buttons.rejected.connect(dlg.close)
        lay.addWidget(buttons)
        dlg.exec_()

    def _extract_features(self):
        """Show Extract Features dialog with full feature transparency."""
        if self._refuse_while_locked("Measure"):
            return
        if self._labels_volume is None:
            self.ui.show_mt_status(S4T.MT_NOTHING_LOADED)
            return

        from qtpy.QtWidgets import QDialog

        from napari_epicycle.fusion_reconstruction.qc_features import PCA_PRESETS
        from napari_epicycle.segmentation.analysis_widget import (
            _FEATURE_DESCRIPTIONS,
        )
        from napari_epicycle.segmentation.extract_dialog import (
            ExtractFeaturesDialog,
        )
        from napari_epicycle.segmentation.feature_registry import (
            get_feature_manifest,
        )

        invert = (self.state_manager.get_record(self._current_stack_name)
                  .get("z_orientation") == "basal_up")

        # Owner: measuring a finalized stack again takes it out
        # of Export until it is finalized again -- ask first.
        if self.state_manager.get_stack_status(
                self._current_stack_name) == "finalized":
            from ..ui_shell.theme import themed_confirm
            if not themed_confirm(
                self.ui, S4T.REMEASURE_TITLE,
                S4T.REMEASURE_FINALIZED_CONFIRM.format(
                    stack=self._current_stack_name),
                confirm_text=S4T.REMEASURE_BTN, cancel_text=S4T.CANCEL_BTN,
                accent="green",
            ):
                return
        self._ensure_writable_state()

        # Check for saved border cells
        record = self.state_manager.get_record(self._current_stack_name)
        # F16: smoothed since the last Measure (held or saved)
        smoothed_at, _k = self._tidy_fields(record)
        measured_at = record.get("measured_at")
        was_smoothed = smoothed_at is not None and (
            measured_at is None or str(smoothed_at) > str(measured_at))
        border_path = record.get("border_cells_path", "")
        has_saved_border = bool(border_path and Path(border_path).exists())

        n_total = int(getattr(self, "_n_cells", 0) or 0) or (
            len(np.unique(self._labels_volume)) - 1)

        # Build channel configs from UI
        channel_configs = [
            {"index": idx, "prefix": name}
            for idx, name in self._get_channel_configs()
        ]

        manifest = get_feature_manifest(checked_channels=channel_configs or None)

        dlg = ExtractFeaturesDialog(
            self.ui,
            feature_manifest=manifest,
            n_labels=n_total,
            was_smoothed=was_smoothed,
            has_saved_border_cells=has_saved_border,
            pca_presets=PCA_PRESETS,
            descriptions=_FEATURE_DESCRIPTIONS,
        )

        if dlg.exec_() != QDialog.Accepted:
            return

        # Store PCA selection for post-extraction auto-PCA
        self._pending_pca_features = dlg.selected_pca_features()
        self._pending_pca_mode = dlg.pca_mode()
        self._pending_pca_preset_name = dlg.selected_preset_name()
        # Persist selection for EDA table pre-check (survives _inject clearing)
        self._last_pca_feature_selection = list(self._pending_pca_features)

        # Border removal happens AFTER extraction so neighbor features
        # of interior cells are computed with border cells still present
        self._pending_border_removal = dlg.remove_border()

        self._run_feature_extraction(
            skip_probability=dlg.skip_probability(),
            exclude_border=not dlg.remove_border(),
            invert_z=invert,
        )

        # TODO: batch extraction with queue (sequential for now)

    def _run_feature_extraction(self, *, skip_probability: bool,
                                exclude_border: bool = False,
                                invert_z: bool = False):
        if self._labels_volume is None or self._current_stack_name is None:
            self.ui.show_mt_status(S4T.MT_NOTHING_LOADED)
            return

        record = self.state_manager.get_record(self._current_stack_name)
        labels_path = record.get("labels_path", "")
        source_stack_path = record.get("source_stack_path", "")
        prob_path = record.get("prob_path", "")

        # Fallback: resolve source path from stacks directory if missing
        if not source_stack_path or not Path(source_stack_path).exists():
            source_stack_path = self._resolve_source_path(
                self._current_stack_name, labels_path
            )

        csv_dir = Path(self.working_dir) / "artifacts" / "step4" / "qc"
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = str(
            csv_dir / f"{self._current_stack_name}_features.csv"
        )

        # Gather channel configs from UI
        channel_configs = [
            {"index": idx, "prefix": name}
            for idx, name in self._get_channel_configs()
        ]

        # Temp-merge border cells for neighbor computation if previously removed
        # The worker gets its OWN copy (audit L2): the volume on screen can
        # change while it runs.
        labels_for_extraction = self._labels_volume.copy()
        border_cells_path = record.get("border_cells_path", "")
        if border_cells_path and Path(border_cells_path).exists():
            import tifffile
            try:
                border_vol = tifffile.imread(border_cells_path)
                if border_vol.shape == self._labels_volume.shape:
                    border_mask = border_vol > 0
                    labels_for_extraction[border_mask] = border_vol[border_mask]
                    n_restored = len(np.unique(border_vol[border_mask]))
                    self.ui.log_output.append(
                        f"[extract] Temporarily restored {n_restored} "
                        f"border cells for neighbor computation"
                    )
                    # Auto-remove border rows after extraction
                    self._pending_border_removal = True
                del border_vol  # free memory
            except Exception as exc:
                self.ui.log_output.append(
                    f"[WARN] Could not load border cells: {exc}"
                )

        self.ui.show_qc_bar(value=0, text=S4T.BUSY_MEASURING.format(pct=0))
        self.ui.show_mt_status("")                  # G1-3: no old sentence
        import time as _time
        self._measure_t0 = _time.monotonic()

        # Stamp the job (F1): token + stack travel in its signals; what
        # Measure used is written on the record at a CURRENT finish only.
        token = self._next_token()
        spacing = self._get_spacing()
        measured = ({"zyx": [float(v) for v in spacing], "source": "user"}
                    if spacing else None)
        rec_sp = record.get("spacing")
        if measured and S4M.same_spacing(measured, rec_sp):
            # Audit D6: the same numbers as the record keep its source.
            measured["source"] = rec_sp.get("source", "user")
        adj_path = str(Path(csv_path).parent
                       / f"{self._current_stack_name}_adjacency.csv")
        pendings = getattr(self, "_measure_pendings", None)
        if pendings is None:
            pendings = self._measure_pendings = {}
        pendings[token] = [(_measure_pending(csv_path, token), csv_path),
                           (_measure_pending(adj_path, token), adj_path)]
        self._measure_launch = {
            "token": token,
            "epoch": self._labels_epoch,
            "stack": self._current_stack_name,
            "spacing": measured,
            "z_orientation": "basal_up" if invert_z else "basal_down",
        }
        worker = _StackQCFeatureWorker(
            labels_path=labels_path,
            stack_name=self._current_stack_name,
            csv_path=csv_path,
            spacing=self._get_spacing(),
            source_stack_path=source_stack_path,
            seg_channel=self._get_seg_channel(),
            prob_path=prob_path,
            channel_configs=channel_configs,
            skip_probability=skip_probability,
            labels_array=labels_for_extraction,
            exclude_border=exclude_border,
            invert_z=invert_z,
            token=token,
            # Re-audit M1: the CSV Measure writes carries the SAVED marks
            # only; marks made since the last Save stay in memory (carried
            # by the finish, _tiers_dirty) so a Discard still drops them.
            tier_map=None,
            tier_csv=record.get("qc_features_csv", ""),
        )
        if self._launch_measure_worker(worker, token) is None:
            self._restore_measure_widgets()           # re-audit L-d

    def _on_measure_progress(self, pct):
        """``run_extraction``'s percent: the bar reads
        ``BUSY_MEASURING``.  It reports no stage names, so the status line
        keeps its last sentence (``BUSY_MEASURE_STAGE*`` stay unused)."""
        self.ui.show_qc_bar(value=pct,
                            text=S4T.BUSY_MEASURING.format(pct=int(pct)))
        t0 = getattr(self, "_measure_t0", None)
        if t0 is not None and 5 <= int(pct) < 100:
            import time as _time
            elapsed = _time.monotonic() - t0
            left = elapsed * (100 - int(pct)) / max(int(pct), 1)
            self.ui.show_mt_status(S4T.BUSY_MEASURE_ETA.format(
                t=self._about(left)))

    def _launch_measure_worker(self, worker, token):
        worker.progress.connect(self._on_measure_progress)
        worker.log.connect(self.ui.log_output.append)
        worker.finished.connect(self._on_extract_worker_finished)
        worker.error.connect(self._on_extract_worker_error)
        worker.csv_saved.connect(self._on_csv_saved)
        thread = self._start_step4_thread(
            worker, "_measure_running", quit_on=("finished", "error"),
            attr="_qc", cleanup=self._restore_measure_widgets)
        if thread is not None:
            # Only a job that really started owns the live token (L1).
            self._measure_token = token
        return thread

    def _measure_is_stale(self, stack_name, token) -> bool:
        launch = getattr(self, "_measure_launch", None) or {}
        return (token != getattr(self, "_measure_token", None)
                or stack_name != self._current_stack_name
                or launch.get("epoch") != self._labels_epoch)

    def _drop_measure_pending(self, token):
        """Delete a measure job's pending files (stale, failed, orphaned)."""
        for pending, _final in (getattr(self, "_measure_pendings", {})
                                or {}).pop(token, []):
            try:
                os.unlink(pending)
            except FileNotFoundError:
                pass
            except OSError as exc:
                print(f"[STEP4-MEASURE] could not delete {pending}: {exc}")

    def _commit_measure_files(self, stack_name, token) -> bool:
        """Move a CURRENT measure's pending files into place and point the
        record at the CSV (audit D2 / D3)."""
        pairs = (getattr(self, "_measure_pendings", {}) or {}).pop(token, [])
        csv_final = None
        for pending, final in pairs:
            if not Path(pending).exists():
                continue
            try:
                os.replace(pending, final)
            except OSError as exc:
                self.ui.log_output.append(
                    f"[ERROR] could not keep {Path(final).name}: {exc}")
                for p, _f in pairs:
                    try:
                        os.unlink(p)
                    except OSError:
                        pass
                return False
            if final.endswith("_features.csv"):
                csv_final = final
        if csv_final is not None:
            self.state_manager.update_record(stack_name,
                                             qc_features_csv=csv_final)
        return True

    def _on_extract_worker_finished(self, df, stack_name, token):
        """F1: a finish for another job or another stack changes nothing."""
        if self._measure_is_stale(stack_name, token):
            self.ui.log_output.append(
                f"[STEP4] measure result for {stack_name} discarded; "
                f"its saved CSV is unchanged")
            print(f"[STEP4-MEASURE] stale finish token={token} "
                  f"live={getattr(self, '_measure_token', None)}")
            self._drop_measure_pending(token)
            self._pending_border_removal = False          # audit D14
            return
        if not self._commit_measure_files(stack_name, token):
            self._on_extract_error("the measured files could not be kept",
                                   stack_name)
            return
        self._on_extract_finished(df, stack_name)

    def _on_extract_worker_error(self, msg, stack_name, token):
        self._drop_measure_pending(token)
        if self._measure_is_stale(stack_name, token):
            self.ui.log_output.append(
                f"[STEP4] measure of {stack_name} failed after "
                f"{self._current_stack_name} was loaded: {msg}")
            return
        self._on_extract_error(msg, stack_name)

    def _restore_measure_widgets(self):
        """Backstop at thread end: Measure is never left grey, and a job
        that ended without a finish leaves no pending file behind."""
        for token in list((getattr(self, "_measure_pendings", {}) or {})):
            self._drop_measure_pending(token)
        self.ui.show_qc_bar(busy=False)

    def _on_extract_finished(self, df, stack_name):
        # Post-extraction border cell removal (if requested via Re-Extract dialog)
        if getattr(self, "_pending_border_removal", False) and "is_border_cell" in df.columns:
            self._pending_border_removal = False
            border_ids = df.loc[df["is_border_cell"] == 1, "label"].values
            if len(border_ids) > 0:
                # Remove border rows from DataFrame
                n_before = len(df)
                df = df[df["is_border_cell"] != 1].reset_index(drop=True)
                n_removed = n_before - len(df)
                # Save border cells as separate volume before removing
                border_set = set(int(lid) for lid in border_ids)
                record = self.state_manager.get_record(stack_name)
                labels_path = record.get("labels_path", "")
                if labels_path:
                    border_volume = np.zeros_like(self._labels_volume)
                    border_mask = np.isin(self._labels_volume, list(border_set))
                    border_volume[border_mask] = self._labels_volume[border_mask]
                    border_path = str(
                        Path(labels_path).parent / "border_cells.tif"
                    )
                    import tifffile
                    tifffile.imwrite(border_path, border_volume)
                    self.state_manager.set_stack_status(
                        stack_name,
                        self.state_manager.get_stack_status(stack_name),
                        border_cells_path=border_path,
                    )
                    self.ui.log_output.append(
                        f"[re-extract] Border cells saved: {Path(border_path).name}"
                    )

                # Remove from volume
                mask = np.isin(self._labels_volume, list(border_set))
                self._labels_volume[mask] = 0
                self._labels_epoch += 1
                self._hold_edit_time()
                self._has_unsaved_changes = True
                # Update viewer
                lyr = self._find_labels_layer()
                if lyr is not None:
                    lyr.data = self._labels_volume
                    lyr.refresh()
                self.ui.log_output.append(
                    f"[re-extract] Removed {n_removed} border cells "
                    f"from DataFrame + volume (post-extraction)"
                )
                # Re-save CSV without border cells
                record = self.state_manager.get_record(stack_name)
                csv_path = record.get("qc_features_csv", "")
                if csv_path:
                    from napari_epicycle.fusion_reconstruction.qc_features import (
                        save_patch_csv,
                    )
                    save_patch_csv(df, csv_path)

        # Ensure label_status column exists from first extraction
        if "label_status" not in df.columns:
            df["label_status"] = "active"

        # F18: marks carried over from the previous df by label id (Step
        # 3's left-merge on re-extraction); gone cells drop, new cells are
        # unmarked.  The measured CSV holds the SAVED marks only; marks
        # made since the last Save stay in memory, unsaved (re-audit M1).
        kept = _carry_tiers(self._qc_features_df, df)
        if not kept and "qc_tier" in df.columns:
            kept = int((df["qc_tier"].fillna("").astype(str) != "").sum())
        if kept:
            self._tiers_dirty = True
            self.ui.log_output.append(
                "[measure] " + S4T.MT_TIERS_KEPT.format(k=kept))

        self._workbench_cells_changed("measure")      # B3b
        self._qc_features_df = df
        self._last_pca_projected = None  # invalidate — row count changed
        self._invalidate_eda_cache_on_disk()
        self._cached_eda_state = None  # clear in-memory cache so fresh PCA uses dialog features

        # Auto-open EDA Workbench so the plot widget is available
        self._open_analysis_dock()

        if self._pw is not None:
            self._pw.set_features(df)
        self._set_default_plot_view()
        # a Measure clears the Undo slot
        self._undo_slot = None
        self.ui.show_qc_bar(value=100, text=S4T.QC_BAR_IDLE)
        done = S4T.say(S4T.MT_MEASURED, n=int(len(df)), stack=stack_name)
        if kept:
            done += " " + S4T.say(S4T.MT_TIERS_KEPT, k=int(kept))
        self.ui.show_mt_status(done)
        from datetime import datetime, timezone
        launch = getattr(self, "_measure_launch", None) or {}
        fields = {"measured_at": datetime.now(timezone.utc).isoformat(),
                  "measured_cells": int(len(df))}
        if launch.get("stack") == stack_name:
            fields["measured_spacing"] = launch.get("spacing")
            fields["measured_z_orientation"] = launch.get("z_orientation")
        self.state_manager.set_stack_status(stack_name, "qc_extracted",
                                            **fields)
        # the volume this Measure describes (after its border removal)
        self._measured_epoch = self._labels_epoch
        # Measuring a finalized stack again de-finalizes it.
        self.state_manager.clear_stack_keys(
            stack_name, ["finalized_at", "finalized_fingerprint"])
        self._persist_state()
        self._update_tier_summary()
        self._update_qc_button_states()
        self._populate_analysis_features()
        self._refresh_stacks()

    def _on_extract_error(self, msg, stack_name=None):
        self.ui.show_qc_bar(value=0, text="")
        self.ui.show_mt_status(S4T.MT_MEASURE_FAILED.format(
            stack=stack_name or self._current_stack_name or ""))
        self.ui.log_output.append(f"[ERROR] {msg}")
        self._refresh_mt_panel()

    def _on_csv_saved(self, stack_name, csv_path):
        # Audit D3: the record is written only by the guarded finish, which
        # moves this pending file into place.  Log only.
        print(f"[STEP4-MEASURE] {stack_name}: pending CSV {csv_path}")

    def _cleanup_qc_thread(self):
        self._qc_thread = None
        self._qc_worker = None

    # ------------------------------------------------------------------
    # QC Analysis — Lasso Selection
    # ------------------------------------------------------------------

    def _on_lasso_selection(self, label_ids):
        """The plot's selection (lasso, a click, a legend colour, Clear).

        The card's message line: LASSO_DONE after a lasso; a legend pick
        keeps the CAT_PICKED ``_on_category_selected`` wrote just before;
        a click or a Clear empties it."""
        self._selected_label_ids = [int(lid) for lid in label_ids]
        pw = self._pw
        lasso = bool(getattr(pw, "_selected_indices", None))
        picked = bool(getattr(pw, "_selected_categories", None))
        if lasso and self._selected_label_ids:
            m = len(self._selected_label_ids)
            self._card_message(T_WB().LASSO_DONE_ONE if m == 1 else
                               T_WB().LASSO_DONE.format(m=f"{m:,}"), "ok")
        elif not picked:
            self._card_message("")
        self._refresh_selection_card()
        self._highlight_selected_in_viewer()

    def _find_labels_layer(self):
        """Find the stack labels layer in the viewer."""
        if self.viewer is None:
            return None
        # F15: the layer Step 4 added (its actual name may differ)
        lyr = getattr(self, "_stack_labels_layer", None)
        if lyr is not None and lyr in self.viewer.layers:
            return lyr
        return None

    def _on_category_selected(self, column, value):
        """Handle legend click in QC plot — select all labels in category."""
        if self._qc_features_df is None:
            return
        df = self._qc_features_df
        if column == "qc_tier":
            # The Workbench's three buckets, the plot's own rule (E1-5):
            # "bad" takes garbage, "unmarked" takes empty / review / ...
            from .analysis_widget import tier_bucket
            if column in df.columns:
                mask = (df[column].map(tier_bucket) == str(value)).to_numpy()
            else:
                mask = np.full(len(df), str(value) == "unmarked")
        elif column not in df.columns:
            return
        else:
            # Find all labels matching this category value
            mask = (df[column].astype(str) == str(value)).to_numpy()
        if "label_status" in df.columns:
            # A removed cell is in no pick (re-audit MED-2).
            mask = mask & (df["label_status"].fillna("active").astype(str)
                           == "active").to_numpy()
        label_ids = self._qc_features_df.loc[mask, "label"].values.tolist()
        # Treat as a lasso selection
        self._selected_label_ids = [int(lid) for lid in label_ids]
        self._card_message(T_WB().CAT_PICKED.format(
            cat=_category_word(column, value),
            m=f"{len(self._selected_label_ids):,}"), "ok")
        self._refresh_selection_card()
        self._highlight_selected_in_viewer()

    def _highlight_selected_in_viewer(self):
        """Isolate lasso-selected labels in the viewer.

        Matches Step 3's behavior:
        - Lasso works regardless of sync checkbox
        - Sync ON + lasso: selected labels get feature colors, rest hidden
        - Sync OFF + lasso: selected labels get distinct colors, rest hidden
        - Empty selection: restore previous color state
        """
        layer = self._find_labels_layer()
        if layer is None:
            return

        try:
            from napari.utils.colormaps import DirectLabelColormap
        except ImportError:
            return

        # CLEAR selection → restore colors
        if not self._selected_label_ids:
            self._clear_isolation(layer)
            return

        # ISOLATE selected labels
        self._isolated_label_ids = set(self._selected_label_ids)
        sync_on = self._sync_cb.isChecked() if self._sync_cb else False

        if sync_on and self._qc_features_df is not None:
            # Use current color-by feature colors for selected labels
            pw = self._pw
            if hasattr(pw, "_color_combo") and pw._color_combo.count() > 0:
                color_by = pw._color_combo.currentText()
                cmap_name = (pw._cmap_combo.currentText()
                             if hasattr(pw, "_cmap_combo") else "Coolwarm")
                norm = (pw._norm_combo.currentText()
                        if hasattr(pw, "_norm_combo") else "Rank")
                self._apply_feature_colors(layer, color_by, cmap_name, norm)
            else:
                self._show_distinct_colors(layer)
        else:
            # Distinct napari colors for selected, rest hidden
            self._show_distinct_colors(layer)

    def _show_distinct_colors(self, layer):
        """Show isolated labels with napari's default distinct colors."""
        try:
            from napari.utils.colormaps import DirectLabelColormap, label_colormap
        except ImportError:
            return

        hidden = np.array([0, 0, 0, 0], dtype=np.float32)
        default_cmap = label_colormap(49, seed=0.5)

        color_dict = {None: hidden, 0: hidden}
        if self._isolated_label_ids:
            all_ids = np.array(sorted(self._isolated_label_ids), dtype=np.intp)
            all_colors = default_cmap.map(all_ids).astype(np.float32)
            for i, lid in enumerate(all_ids):
                color_dict[int(lid)] = all_colors[i]

        layer.colormap = DirectLabelColormap(color_dict=color_dict)
        self._viewer_colour_digest = None        # C6: not the plot's colours

    def _on_eda_gate_selection(self, label_ids):
        """Handle histogram gate selection from EDA Workbench.

        Highlights in both the viewer AND the QC scatter/PCA plot.
        """
        # Highlight in viewer (the card keeps the GATE_LINE the Workbench
        # wrote)
        self._selected_label_ids = [int(lid) for lid in label_ids]
        self._refresh_selection_card()
        self._highlight_selected_in_viewer()
        # Highlight in QC plot
        pw = self._pw
        if hasattr(pw, "highlight_labels"):
            pw.highlight_labels(label_ids)

    def _on_eda_color_by_feature(self, feature: str):
        """"Colour plot by this" (the feature detail panel).

        Pressed: the plot's Colour by becomes ``feature`` through the same
        path as a user's pick (``set_colour_by``: the combo, then a redraw),
        so the plot, its legend and -- under the "Colours in viewer" tick
        only -- the viewer follow (C3, 2026-10-05).  Unpressed (``""``):
        Colour by goes back to what it was before the press, the same way.
        """
        pw = self._pw
        if pw is None or not hasattr(pw, "set_colour_by"):
            return
        if not feature:
            prev = getattr(self, "_colour_by_before_pick", None)
            self._colour_by_before_pick = None
            if prev and pw._color_combo.currentText() != prev:
                pw.set_colour_by(prev)
            return
        if self._qc_features_df is None:
            return
        if feature not in self._qc_features_df.columns:
            return
        if getattr(self, "_colour_by_before_pick", None) is None:
            self._colour_by_before_pick = pw._color_combo.currentText()
        if not pw.set_colour_by(feature):
            print(f"[WB-COLOUR] {feature!r} is not offered in Colour by")

    #: Categorical Colour-by words -> their column (hover properties).
    _CAT_FEATURE_TO_COL = {
        "QC tier": "qc_tier",
        "Cluster": "cluster_id",
        "Border cell": "is_border_cell",
        "Topology": "euler_number",
        "QC status": "qc_status",
    }

    def _apply_feature_colors(self, layer, feature, cmap_name, norm_mode):
        """The viewer shows what the plot draws (owner question 2026-10-05).

        The ids and RGBA come from ``QCPlotWidget.viewer_colours()`` -- the
        plot's own colour rule for every option: the Workbench's three tier
        buckets (C1), the plot's fallback grey for a missing column (C2),
        its normalisation of NaN and constant values (C4), its noise alpha,
        removed rows left out (C5).  ``feature`` / ``cmap_name`` /
        ``norm_mode`` are the plot's current choices (every caller reads
        them off the plot); only ``feature`` is used, for the hover.
        Isolation-aware.  C6: skipped when the colours and the layer are
        the ones applied last.
        """
        try:
            from napari.utils.colormaps import DirectLabelColormap
        except ImportError:
            return
        df = self._qc_features_df
        pw = self._pw
        if (df is None or pw is None or not hasattr(pw, "viewer_colours")
                or getattr(pw, "_df", None) is None):
            return
        try:
            ids, rgba = pw.viewer_colours()
        except Exception as exc:      # noqa: BLE001
            print(f"[WB-COLOUR] viewer colours not read: {exc}")
            return
        ids = np.asarray(ids, dtype=np.int64)
        rgba = np.nan_to_num(np.asarray(rgba, dtype=np.float32))
        iso = self._isolated_label_ids if self._isolated_label_ids else None
        if iso is not None:
            keep = np.isin(ids, np.fromiter(iso, dtype=np.int64))
            ids, rgba = ids[keep], rgba[keep]

        import hashlib
        import weakref
        digest = hashlib.blake2b(ids.tobytes() + rgba.tobytes(),
                                 digest_size=16).digest()
        ref = getattr(self, "_viewer_colour_layer", None)
        if (digest == getattr(self, "_viewer_colour_digest", None)
                and ref is not None and ref() is layer):
            return                               # C6: nothing changed

        hidden = np.array([0, 0, 0, 0], dtype=np.float32)
        color_dict = {None: hidden, 0: hidden}
        for i, lid in enumerate(ids):
            color_dict[int(lid)] = rgba[i]
        layer.colormap = DirectLabelColormap(color_dict=color_dict)
        self._viewer_colour_digest = digest
        try:
            self._viewer_colour_layer = weakref.ref(layer)
        except TypeError:
            self._viewer_colour_layer = (lambda _l=layer: _l)
        self._viewer_recolours = int(
            getattr(self, "_viewer_recolours", 0)) + 1
        self._set_hover_properties(layer, feature, df, ids)

    def _set_hover_properties(self, layer, feature, df, ids):
        """The labels layer's hover: the Colour-by value and qc_score."""
        if ids.size == 0 or "label" not in df.columns:
            return
        rows = df.set_index("label", drop=False)
        rows = rows[~rows.index.duplicated(keep="last")]
        rows = rows.reindex(ids)
        max_label = int(ids.max())
        props = {}
        col = self._CAT_FEATURE_TO_COL.get(feature, feature)
        if col in rows.columns:
            vals = rows[col].to_numpy()
            if feature in self._CAT_FEATURE_TO_COL:
                arr = np.empty(max_label + 1, dtype=object)
                arr[:] = ""
                arr[ids] = ["" if v is None else str(v) for v in vals]
            else:
                arr = np.zeros(max_label + 1)
                arr[ids] = np.round(vals.astype(np.float64), decimals=2)
            props[col] = arr
        if col != "qc_score" and "qc_score" in rows.columns:
            qc = np.zeros(max_label + 1)
            qc[ids] = np.round(rows["qc_score"].to_numpy(np.float64), 2)
            props["qc_score"] = qc
        if props:
            try:
                layer.properties = props
            except Exception as exc:      # noqa: BLE001
                print(f"[WB-COLOUR] hover properties not set: {exc}")

    def _clear_isolation(self, layer):
        """Restore colors after clearing lasso selection."""
        self._isolated_label_ids = set()
        sync_on = self._sync_cb.isChecked() if self._sync_cb else False

        if sync_on and self._qc_features_df is not None:
            # Re-apply feature colors to ALL labels (no isolation filter)
            pw = self._pw
            if hasattr(pw, "_color_combo") and pw._color_combo.count() > 0:
                color_by = pw._color_combo.currentText()
                cmap_name = (pw._cmap_combo.currentText()
                             if hasattr(pw, "_cmap_combo") else "Coolwarm")
                norm = (pw._norm_combo.currentText()
                        if hasattr(pw, "_norm_combo") else "Rank")
                self._apply_feature_colors(layer, color_by, cmap_name, norm)
                return
        # Fallback: default napari colors
        cmap = self._get_default_label_colormap()
        if cmap is not None:
            layer.colormap = cmap
            self._viewer_colour_digest = None

    def _on_viewer_color_requested(self, feature, cmap_name, norm_mode):
        """Apply feature-based colormap when color-by selection changes."""
        if not (self._sync_cb and self._sync_cb.isChecked()):
            return
        layer = self._find_labels_layer()
        if layer is None or self._qc_features_df is None:
            return
        self._apply_feature_colors(layer, feature, cmap_name, norm_mode)

    def _on_sync_toggled(self, checked):
        """Handle sync viewer colors checkbox toggle."""
        layer = self._find_labels_layer()
        if layer is None:
            return

        if checked:
            # Apply feature colors (isolation-aware)
            if self._isolated_label_ids:
                self._highlight_selected_in_viewer()
            elif self._qc_features_df is not None:
                pw = self._pw
                if pw is not None and hasattr(pw, "_color_combo") and pw._color_combo.count() > 0:
                    color_by = pw._color_combo.currentText()
                    cmap_name = (pw._cmap_combo.currentText()
                                 if hasattr(pw, "_cmap_combo") else "Coolwarm")
                    norm = (pw._norm_combo.currentText()
                            if hasattr(pw, "_norm_combo") else "Rank")
                    self._apply_feature_colors(layer, color_by, cmap_name, norm)
        else:
            if self._isolated_label_ids:
                # Isolation active — switch to distinct colors
                self._show_distinct_colors(layer)
            else:
                # Reset to default
                cmap = self._get_default_label_colormap()
                if cmap is not None:
                    layer.colormap = cmap
                    self._viewer_colour_digest = None

    def _get_default_label_colormap(self):
        """Return napari's default label colormap."""
        try:
            from napari.utils.colormaps import label_colormap
            return label_colormap(49, seed=0.5)
        except ImportError:
            return None

    # ------------------------------------------------------------------
    # QC Analysis — Triage
    # ------------------------------------------------------------------

    def _mark_good(self):
        self._assign_tier("good")

    def _mark_bad(self):
        self._assign_tier("bad")

    def _unmark(self):
        """Unmark: the selection back to unassigned."""
        self._assign_tier("")

    def _mark_refused(self, what: str) -> bool:
        """Every mark slot checks the lock itself: a key fires
        whatever the buttons show.  Outside Step 4 a mark is refused too
        (K7; WB_STEP_INACTIVE is on the card)."""
        if not getattr(self, "_step4_active", False):
            print(f"[STEP4-LOCK] {what} refused: Step 4 is not active")
            aw = getattr(self, "_analysis_widget", None)
            if aw is not None and hasattr(aw, "set_step_inactive"):
                aw.set_step_inactive(True)
            return True
        if not self._refuse_while_locked(what):
            return False
        self._refresh_workbench_lock()
        return True

    def _assign_tier(self, tier: str):
        """Mark good / Mark bad / Unmark (``tier == ""``).

        F8 / I-7 / I-17: marks are unsaved work of their own (_tiers_dirty,
        not _labels_dirty), and a mark never clears PCA / UMAP / t-SNE --
        nor a running map, whose token holds no df counter.  Every mark
        pushes (ids, previous tiers) for Undo.
        """
        if self._mark_refused("Mark"):
            return
        if self._qc_features_df is None:
            return
        if not self._selected_label_ids:
            self._card_message(T_WB().MARK_NEEDS_SEL, "warn")
            return
        df = self._qc_features_df
        mask = df["label"].isin(self._selected_label_ids)
        if "qc_tier" not in df.columns:
            df["qc_tier"] = ""
        labels = [int(v) for v in df.loc[mask, "label"].tolist()]
        before = list(df.loc[mask, "qc_tier"].tolist())
        history = getattr(self, "_mark_history", None)
        if history is None:
            history = self._mark_history = []
        history.append((labels, before))
        df.loc[mask, "qc_tier"] = tier
        self._tiers_dirty = True
        n = int(mask.sum())
        print(f"[STEP4-MARK] {n} labels -> {tier or 'unassigned'}")
        self._refresh_after_marks()
        if tier:
            self._card_message(
                T_WB().MARKED_ONE.format(tier=tier) if n == 1 else
                T_WB().MARKED.format(m=f"{n:,}", tier=tier), "ok")
        else:
            self._card_message(T_WB().UNMARKED_ONE if n == 1 else
                               T_WB().UNMARKED.format(m=f"{n:,}"), "ok")

    def _undo_mark(self):
        """Undo: the last Mark or Unmark taken back."""
        if self._mark_refused("Undo"):
            return
        if self._qc_features_df is None:
            return
        history = getattr(self, "_mark_history", None) or []
        if not history:
            self._card_message(T_WB().UNDO_EMPTY, "warn")
            self._wb_sync_mark_state()
            return
        labels, before = history.pop()
        df = self._qc_features_df
        back = dict(zip(labels, before))
        mask = df["label"].isin(list(back))
        if "qc_tier" not in df.columns:
            df["qc_tier"] = ""
        df.loc[mask, "qc_tier"] = [back.get(int(v), "")
                                   for v in df.loc[mask, "label"]]
        self._tiers_dirty = True
        print(f"[STEP4-MARK] undo: {len(labels)} labels restored")
        self._refresh_after_marks()
        self._card_message(
            T_WB().UNDONE_ONE if len(labels) == 1 else
            T_WB().UNDONE.format(m=f"{len(labels):,}"), "ok")

    def _refresh_after_marks(self):
        """Redraw with the maps AND the selection kept, then the card, the
        totals and the panel's buttons."""
        pw = self._pw
        if pw is not None:
            if getattr(pw, "_layout_mode", "classic") == "workbench":
                pw.refresh_marks(self._qc_features_df)
            else:
                pw.set_features(self._qc_features_df, preserve_caches=True)
        self._update_tier_summary()
        self._refresh_selection_card()
        try:
            self._update_qc_button_states()
        except Exception as exc:      # noqa: BLE001
            print(f"[STEP4-MARK] button refresh failed: {exc}")

    # ------------------------------------------------------------------
    # Workbench: the selection card and its lock line
    # ------------------------------------------------------------------

    def _card_message(self, text: str, kind: str = ""):
        aw = getattr(self, "_analysis_widget", None)
        if aw is not None and hasattr(aw, "show_card_message"):
            aw.show_card_message(text, kind)

    def _wb_sync_mark_state(self):
        aw = getattr(self, "_analysis_widget", None)
        if aw is not None and hasattr(aw, "set_mark_state"):
            aw.set_mark_state(
                has_selection=bool(getattr(self, "_selected_label_ids", [])),
                can_undo=bool(getattr(self, "_mark_history", [])))

    def _refresh_selection_card(self):
        """The card's line 1, mix bar and chips from the selection."""
        aw = getattr(self, "_analysis_widget", None)
        if aw is None or not hasattr(aw, "show_selection"):
            return
        from .analysis_widget import selection_card_parts
        ids = list(self._selected_label_ids or [])
        df = self._qc_features_df
        tiers = []
        if ids and df is not None:
            rows = df.loc[df["label"].isin(ids)]
            tiers = (rows["qc_tier"].tolist() if "qc_tier" in rows.columns
                     else [""] * len(rows))
        aw.show_selection(**selection_card_parts(
            tiers, first_id=ids[0] if ids else None))
        self._wb_sync_mark_state()

    def _refresh_workbench_lock(self):
        """WB_LOCKED while any Step 4 block flag is set."""
        aw = getattr(self, "_analysis_widget", None)
        if aw is None or not hasattr(aw, "set_workbench_lock"):
            return
        from ._step4_lock import running_flags
        flags = running_flags(self)
        job = (T_WB().LOCK_JOB_WORDS.get(flags[0], flags[0])
               if flags else None)
        aw.set_workbench_lock(job)

    def _delete_bad_labels(self):
        """Remove bad cells (Review): a red confirm, the voxels zeroed,
        the removed voxels kept in the one-level Undo slot."""
        if self._refuse_while_locked("Remove bad cells"):
            return
        if self._qc_features_df is None or self._labels_volume is None:
            return
        df = self._qc_features_df
        if "qc_tier" not in df.columns:
            self.ui.show_mt_status(S4T.MT_NO_BAD)
            return
        is_active = (df["label_status"] == "active"
                     if "label_status" in df.columns else True)
        bad_mask = df["qc_tier"].isin(["bad", "garbage"]) & is_active
        bad_labels = df.loc[bad_mask, "label"].values
        if len(bad_labels) == 0:
            self.ui.show_mt_status(S4T.MT_NO_BAD)
            return
        stack = self._current_stack_name
        from ..ui_shell import theme as th
        if not th.themed_confirm(
            self.ui, S4T.REMOVE_TITLE,
            S4T.say(S4T.REMOVE_CONFIRM, n=int(len(bad_labels)), stack=stack),
            confirm_text=S4T.REMOVE_CONFIRM_BTN,
            cancel_text=S4T.CANCEL_BTN, accent="red",
        ):
            return
        if not self._same_stack_after_modal(stack, "Remove"):
            return

        # Auto-backup raw CSV before first deletion
        if ("label_status" not in df.columns
                or (df["label_status"] == "active").all()):
            record = self.state_manager.get_record(stack)
            csv_path = record.get("qc_features_csv", "")
            if csv_path and Path(csv_path).exists():
                raw_path = Path(csv_path).parent / (
                    Path(csv_path).stem + "_raw.csv")
                if not raw_path.exists():
                    import shutil
                    try:
                        shutil.copy2(csv_path, raw_path)
                        self.ui.log_output.append(
                            f"[triage] Backed up raw features → "
                            f"{raw_path.name}")
                    except Exception as exc:      # noqa: BLE001
                        self.ui.log_output.append(
                            f"[WARN] Could not backup raw CSV: {exc}")

        bad_df = df.loc[bad_mask]
        summary = [f"Deleted {len(bad_labels)} labels"]
        if "area" in bad_df.columns:
            summary.append(f"avg area={bad_df['area'].mean():.0f}")
        if "qc_score" in bad_df.columns:
            summary.append(f"avg qc_score={bad_df['qc_score'].mean():.2f}")
        self.ui.log_output.append(f"[triage] {', '.join(summary)}")

        bad_set = [int(lid) for lid in bad_labels]
        mask = np.isin(self._labels_volume, bad_set)
        flat = np.flatnonzero(mask)
        old_ids = self._labels_volume.reshape(-1)[flat].copy()
        pending = getattr(self, "_pending_edits", None) or {}
        prev_changed = pending.get("labels_changed_at")
        prev_status = (df.loc[bad_mask, "label_status"].copy()
                       if "label_status" in df.columns else None)
        self._labels_volume[mask] = 0
        self._labels_epoch += 1
        self._hold_edit_time()
        if "label_status" not in self._qc_features_df.columns:
            self._qc_features_df["label_status"] = "active"
        self._qc_features_df.loc[bad_mask, "label_status"] = "deleted"
        self._undo_slot = {
            "kind": "remove", "stack": stack, "flat": flat,
            "ids": old_ids, "rows": bad_mask.copy(),
            "prev_status": prev_status,
            "labels_changed_at": prev_changed,
            "labels_dirty": self._labels_dirty,
            "epoch": self._labels_epoch,
        }
        self._n_cells = max(int(getattr(self, "_n_cells", 0) or 0)
                            - len(bad_set), 0)

        lyr = self._find_labels_layer()
        if lyr is not None:
            lyr.data = self._labels_volume
            lyr.refresh()

        self._workbench_cells_changed("remove bad cells")  # B3b
        self._last_pca_projected = None  # invalidate -- rows deleted
        self._invalidate_eda_cache_on_disk()
        if self._pw is not None:
            self._pw.set_features(self._qc_features_df)
        self._labels_dirty = True
        self._update_tier_summary()
        n_left = (int((self._qc_features_df["label_status"] == "active")
                      .sum()))
        self.ui.show_mt_status(S4T.say(S4T.MT_DELETED, n=len(bad_set),
                                       m=n_left))
        print(f"[STEP4-UNDO] slot: remove {len(bad_set)} cells "
              f"({flat.size} voxels)")
        self._refresh_mt_panel()

    def _undo_last_edit(self):
        """Undo: the last Remove bad cells or Smooth outlines,
        whichever came last; one level.  The maps are not brought back."""
        if self._refuse_while_locked("Undo"):
            return
        slot = getattr(self, "_undo_slot", None)
        if (not slot or self._labels_volume is None
                or slot.get("stack") != self._current_stack_name):
            self._undo_slot = None
            self.ui.show_mt_status(S4T.MT_UNDO_EMPTY)
            return
        pending = getattr(self, "_pending_edits", None)
        if pending is None:
            pending = self._pending_edits = {}
        if (slot["kind"] == "smooth" and slot.get("paints", 0)
                != int(getattr(self, "_paint_count", 0))):
            # B1-2: the old array would drop the hand edits made since
            self._undo_slot = None
            self.ui.show_mt_status(S4T.MT_UNDO_AFTER_PAINT)
            print("[STEP4-UNDO] refused: painted after the smooth")
            self._refresh_mt_panel()
            return
        exact = True
        if slot["kind"] == "remove":
            self._workbench_cells_changed("undo remove")  # B3b
            vol = self._labels_volume.reshape(-1)
            flat, ids = slot["flat"], slot["ids"]
            free = vol[flat] == 0        # safe write: never over a new id
            exact = bool(free.all())
            vol[flat[free]] = ids[free]
            df = self._qc_features_df
            if df is not None and "label_status" in df.columns:
                rows = slot["rows"].reindex(df.index, fill_value=False)
                prev = slot.get("prev_status")
                df.loc[rows, "label_status"] = (
                    prev.reindex(df.index[rows]).fillna("active").values
                    if prev is not None else "active")
            self._n_cells = int(getattr(self, "_n_cells", 0) or 0) + len(
                np.unique(ids))
            if self._pw is not None and df is not None:
                self._pw.set_features(df)
            what = S4T.UNDO_WHAT_REMOVE
        else:
            self._labels_volume = slot["volume"]
            for key in ("smoothed_at", "n_thin_kept"):
                if slot.get(key) is None:
                    pending.pop(key, None)
                else:
                    pending[key] = slot[key]
            if self._qc_features_df is not None and \
                    "feature_state" in self._qc_features_df.columns:
                self._qc_features_df["feature_state"] = slot.get(
                    "feature_state", "fresh")
            what = S4T.UNDO_WHAT_SMOOTH
        self._undo_slot = None
        self._labels_epoch += 1
        if exact:
            # B1-6: the volume is what it was before the edit: its held
            # edit time goes back too
            if slot.get("labels_changed_at") is None:
                pending.pop("labels_changed_at", None)
            else:
                pending["labels_changed_at"] = slot["labels_changed_at"]
                pending["_stack"] = self._current_stack_name
        else:
            self._hold_edit_time()
        self._labels_dirty = True
        lyr = self._find_labels_layer()
        if lyr is not None:
            lyr.data = self._labels_volume
            lyr.refresh()
        self._update_tier_summary()
        self.ui.show_mt_status(S4T.MT_UNDONE.format(what=what))
        print(f"[STEP4-UNDO] undid {slot['kind']}")
        self._refresh_mt_panel()

    def _on_save_clicked(self, *_args):
        """The Save button (``clicked`` passes ``checked``: never let it
        reach ``confirm``)."""
        self._save_tier_assignments(confirm=True)

    def _on_finalize_clicked(self, *_args):
        self._finalize()

    def _write_stack_files(self, stack_name, record, tag):
        """Write the loaded labels and CSV for *stack_name*, all or nothing
        as far as the files go (F11): both are written to temp files in
        their own folders first, then replaced.  A failure unlinks the
        temp files, so nothing is overwritten, and raises.  Returns the
        summary parts for the status line."""
        import tifffile
        from napari_epicycle.fusion_reconstruction.qc_features import (
            save_patch_csv,
        )

        saved = []
        pending = []          # (tmp, final)
        moved = []            # (final, prev) -- finals already replaced
        try:
            labels_path = record.get("labels_path", "")
            if self._labels_volume is not None and labels_path:
                tmp = labels_path + ".tmp"
                pending.append((tmp, labels_path))
                tifffile.imwrite(tmp, self._labels_volume)
            csv_path = record.get("qc_features_csv", "")
            if self._qc_features_df is not None and csv_path:
                tmp_csv = csv_path + ".tmp"
                pending.append((tmp_csv, csv_path))
                save_patch_csv(self._qc_features_df, tmp_csv)
            # Audit L4: each old file is renamed aside first (cheap), so a
            # failure on the second replace puts the first one back.
            for tmp, final in pending:
                prev = final + ".prev"
                had_old = os.path.exists(final)
                if had_old:
                    os.replace(final, prev)
                moved.append((final, prev if had_old else None))
                os.replace(tmp, final)
        except BaseException:
            for final, prev in reversed(moved):
                try:
                    if prev is not None:
                        os.replace(prev, final)
                    elif os.path.exists(final):
                        os.unlink(final)
                except OSError as exc:
                    print(f"[STEP4-SAVE] could not restore {final}: {exc}")
                    self.ui.log_output.append(
                        f"[ERROR] could not put back {final} -- the "
                        f"previous file is kept as {prev}: {exc}")
            for tmp, _final in pending:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
            raise
        for _final, prev in moved:
            if prev is not None:
                try:
                    os.unlink(prev)
                except OSError:
                    pass
        self._last_write = {"labels": any(not f.endswith(".csv")
                                          for _t, f in pending),
                            "csv": any(f.endswith(".csv")
                                       for _t, f in pending)}
        for _tmp, final in pending:
            if final.endswith(".csv"):
                df = self._qc_features_df
                n_active = (int((df["label_status"] == "active").sum())
                            if "label_status" in df.columns else len(df))
                saved.append(f"CSV ({n_active} labels, {len(df.columns)} cols)")
            else:
                size_mb = Path(final).stat().st_size / 1e6
                saved.append(f"volume ({size_mb:.1f} MB)")
            self.ui.log_output.append(f"[{tag}] written: {final}")
        return saved

    def _held_edit_fields(self, stack_name):
        """The edit timestamps held since the last Save (G1-9) -- for
        *stack_name* only (audit D1: never another stack's edits)."""
        pending = getattr(self, "_pending_edits", None) or {}
        if pending.get("_stack") != stack_name:
            return {}
        return {k: v for k, v in pending.items()
                if k in ("labels_changed_at", "smoothed_at", "n_thin_kept")}

    def _same_stack_after_modal(self, stack, what) -> bool:
        """Audit L3: a modal runs the event loop, so the stack can change
        or a job start under it.  Re-check before acting."""
        if self._current_stack_name != stack or _step4_is_locked(self):
            self.ui.log_output.append(
                f"[{what.lower()}] {stack} changed while the question was "
                f"open; nothing was written")
            print(f"[STEP4-SAVE] {what} after modal refused: "
                  f"{stack} -> {self._current_stack_name}")
            return False
        return True

    def _unwritten_edits(self):
        """A reason when a dirty part of the stack had nowhere to go."""
        wrote = getattr(self, "_last_write", None) or {}
        if self._labels_dirty and not wrote.get("labels"):
            return "the labels have no file to write to"
        if self._tiers_dirty and not wrote.get("csv"):
            return "the marks have no features CSV to write to"
        return None

    def _save_failed(self, stack_name, exc) -> bool:
        self.ui.log_output.append(f"[ERROR] save {stack_name}: {exc}")
        print(f"[STEP4-SAVE] {stack_name}: FAILED {exc}")
        self.ui.show_mt_status(S4T.MT_SAVE_FAILED.format(stack=stack_name))
        return False

    def _save_tier_assignments(self, confirm: bool = True) -> bool:
        """Save: persist volume + CSV + EDA analysis state to disk.

        Returns whether it saved (F23).  A caller that saves first asks
        once itself, passes ``confirm=False`` and STOPS on False.  On any
        failure the stack stays dirty and nothing is overwritten.
        """
        if self._current_stack_name is None:
            return False
        if self._refuse_while_locked("Save"):
            return False
        stack = self._current_stack_name

        if confirm:
            from ..ui_shell.theme import themed_confirm
            body = S4T.SAVE_CONFIRM.format(stack=stack)
            if self.state_manager.get_stack_status(stack) == "finalized":
                body += S4T.SAVE_TAKES_OUT_OF_EXPORT
            if not themed_confirm(
                self.ui, S4T.SAVE_TITLE, body,
                confirm_text=S4T.SAVE_BTN, cancel_text=S4T.CANCEL_BTN,
                accent="green",
            ):
                return False
            if not self._same_stack_after_modal(stack, "Save"):
                return False

        self._ensure_writable_state()
        if getattr(self.state_manager, "save_blocked", False):
            self.ui.log_output.append(f"[state] {S4T.STATE_SAVE_BLOCKED}")
            return self._save_failed(stack, "the state file is blocked")

        record = self.state_manager.get_record(stack)
        self._set_step4_flag("_save_in_flight", True)
        try:
            try:
                saved = self._write_stack_files(stack, record, "save")
            except Exception as exc:      # noqa: BLE001
                return self._save_failed(stack, exc)
            missing = self._unwritten_edits()
            if missing:
                # Audit L5: never report a save that wrote nothing.
                return self._save_failed(stack, missing)

            # Save EDA state (PCA, UMAP, t-SNE caches)
            self._save_eda_state()
            eda_parts = []
            if self._analysis_widget and hasattr(self._analysis_widget, "_pca_result"):
                if self._analysis_widget._pca_result is not None:
                    eda_parts.append("PCA")
            pw = self._pw
            if pw is not None:
                if getattr(pw, "_umap_cache", None) is not None:
                    eda_parts.append("UMAP")
                if getattr(pw, "_tsne_cache", None) is not None:
                    eda_parts.append("t-SNE")
            if eda_parts:
                saved.append(f"EDA ({', '.join(eda_parts)})")

            held = self._held_edit_fields(stack)
            if held:
                self.state_manager.update_record(stack, **held)
            if not self._persist_state():
                self.state_manager.restore_record(stack, record)   # D5
                return self._save_failed(stack, "the state file was not saved")
        finally:
            self._set_step4_flag("_save_in_flight", False)

        self._has_unsaved_changes = False
        self._pending_edits = {}
        self._undo_slot = None
        self.ui.log_output.append(
            f"[save] {stack}: {' + '.join(saved) or 'nothing written'}")
        self.ui.show_mt_status(S4T.MT_SAVED.format(stack=stack))
        self._refresh_mt_panel()
        # a saved CSV / labels file: the bucket counts and the overlay on
        # the pictures follow the disk (audit M1)
        self._restart_tier_counts_if_csv_changed()
        self._restart_thumbs_if_labels_changed()
        return True

    # ------------------------------------------------------------------
    # QC Analysis — Smooth
    # ------------------------------------------------------------------

    def _smooth_all(self):
        if self._refuse_while_locked("Smooth All"):
            return
        if self._labels_volume is None:
            self.ui.show_mt_status(S4T.MT_NOTHING_LOADED)
            return
        sigma_z, sigma_xy = self.ui.smooth_sigmas_voxels()
        preserve = self.ui.smooth_preserve_topology_cb.isChecked()
        self.ui.log_output.append(f"[smooth] {S4T.LOG_SMOOTH_METHOD}")
        self.ui.show_qc_bar(value=0, text="")

        token = self._next_token()
        self._smooth_launch = {
            "token": token, "stack": self._current_stack_name,
            "volume": self._labels_volume, "epoch": self._labels_epoch,
        }
        # L1: the worker gets its own copy, taken HERE on the main thread
        # (Ctrl+Z writes the layer's data without a paint event, so the
        # live array could change under a worker-side copy)
        worker = _SmoothBatchWorker(
            self._labels_volume.copy(),
            sigma_z=sigma_z, sigma_xy=sigma_xy,
            preserve_topology=preserve,
            stack_name=self._current_stack_name or "",
            token=token,
        )
        if self._launch_smooth_worker(worker) is not None:
            # Only a job that really started owns the live token.
            self._smooth_token = token
        self._refresh_mt_panel()

    def _launch_smooth_worker(self, worker):
        worker.progress.connect(self._on_smooth_progress)
        worker.log.connect(self.ui.log_output.append)
        worker.finished.connect(self._on_smooth_worker_finished)
        worker.error.connect(self._on_smooth_worker_error)
        return self._start_step4_thread(
            worker, "_smooth_running", quit_on=("finished", "error"),
            attr="_smooth", cleanup=self._restore_smooth_widgets)

    def _on_smooth_progress(self, pct):
        self.ui.show_qc_bar(value=pct, percent=True)

    def _smooth_is_stale(self, stack_name, token) -> bool:
        """I-1: another job, another stack, or a volume edited since the
        launch -- the result was computed from something no longer shown."""
        launch = getattr(self, "_smooth_launch", None) or {}
        return (token != getattr(self, "_smooth_token", None)
                or stack_name != self._current_stack_name
                or launch.get("volume") is not self._labels_volume
                or launch.get("epoch") != self._labels_epoch)

    def _on_smooth_worker_finished(self, result, stack_name, token):
        if self._smooth_is_stale(stack_name, token):
            self.ui.log_output.append(
                f"[STEP4] smooth result for {stack_name} discarded: the "
                f"labels on screen changed while it ran")
            print(f"[STEP4-SMOOTH] stale finish token={token}")
            return
        if getattr(result, "cancelled", False):
            self.ui.log_output.append("[smooth] cancelled; nothing changed")
            return
        self._on_smooth_finished(result)

    def _on_labels_layer_edited(self, event=None):
        self._labels_epoch = int(getattr(self, "_labels_epoch", 0)) + 1

    def _on_labels_painted(self, event=None):
        """A hand edit on the layer (napari's ``paint``): the volume moved,
        and a Smooth's Undo would throw the edit away (B1b audit B1-2).
        B2: a paint is unsaved work -- ``_labels_dirty`` is set, the edit
        time held until Save (G1-9), and ③ shows "Edited, not saved", so a
        load, a Browse or leaving Step 4 asks first."""
        self._on_labels_layer_edited(event)
        self._paint_count = int(getattr(self, "_paint_count", 0)) + 1
        if self._labels_volume is None or not self._current_stack_name:
            return
        was = self._labels_dirty
        self._labels_dirty = True
        self._hold_edit_time()
        # refresh ③ on the first unsaved paint, and on the first paint after
        # a Measure (its measurements turn stale now)
        first_after_measure = (getattr(self, "_measured_epoch", None)
                               == self._labels_epoch - 1)
        if not was or first_after_measure:
            print(f"[STEP4-LOCK] paint on {self._current_stack_name}: "
                  f"unsaved label edits")
            try:
                self._refresh_mt_panel()
            except Exception as exc:      # noqa: BLE001
                print(f"[STEP4-UI] (3) not refreshed after a paint: {exc}")

    def _hold_edit_time(self):
        """A label edit's ``labels_changed_at``, held until Save (G1-9)."""
        from datetime import datetime, timezone
        pending = getattr(self, "_pending_edits", None)
        if pending is None:
            pending = self._pending_edits = {}
        pending["labels_changed_at"] = datetime.now(timezone.utc).isoformat()
        pending["_stack"] = self._current_stack_name

    def _on_smooth_finished(self, result):
        """Swap the smoothed copy in (main thread), keep the old array as
        the Undo slot, hold the edit timestamps until Save (F2, G1-9)."""
        from datetime import datetime, timezone

        old = self._labels_volume
        now = datetime.now(timezone.utc).isoformat()
        pending = getattr(self, "_pending_edits", None)
        if pending is None:
            pending = self._pending_edits = {}
        df = self._qc_features_df
        self._undo_slot = {
            "kind": "smooth", "stack": self._current_stack_name,
            "volume": old,
            "paints": int(getattr(self, "_paint_count", 0)),
            "labels_changed_at": pending.get("labels_changed_at"),
            "smoothed_at": pending.get("smoothed_at"),
            "n_thin_kept": pending.get("n_thin_kept"),
            "feature_state": (df["feature_state"].iloc[0]
                              if df is not None and len(df)
                              and "feature_state" in df.columns
                              else "fresh"),
        }
        self._labels_volume = result.working_volume
        self._labels_epoch += 1
        n_thin = len(getattr(result, "emptied_ids", []) or [])
        pending.update(smoothed_at=now, n_thin_kept=n_thin,
                       labels_changed_at=now,
                       _stack=self._current_stack_name)
        self._labels_dirty = True

        self.ui.show_qc_bar(value=100, text=S4T.QC_BAR_IDLE)
        self.ui.show_mt_status(S4T.say(
            S4T.MT_SMOOTHED, n=len(result.touched_ids),
            v=int(result.n_changed_total)))
        if n_thin:
            self.ui.log_output.append(
                "[smooth] " + S4T.say(S4T.MT_SMOOTHED_THIN, k=n_thin))
        lyr = self._find_labels_layer()
        if lyr is not None:
            lyr.data = self._labels_volume
            lyr.refresh()
        # labels_3d_smoothed.tif is no longer written (F2).
        if self._qc_features_df is not None:
            self._qc_features_df["feature_state"] = "stale"
        print("[STEP4-UNDO] slot: smooth")
        self._refresh_mt_panel()

    def _on_smooth_worker_error(self, msg, stack_name, token):
        self.ui.log_output.append(f"[ERROR] {msg}")
        if self._smooth_is_stale(stack_name, token):
            return
        self.ui.show_qc_bar(value=0, text="")
        self.ui.show_mt_status(S4T.MT_SMOOTH_FAILED.format(stack=stack_name))

    def _restore_smooth_widgets(self):
        """Backstop at thread end: the launch record lets go of the volume
        it compared against (L8); the panel re-reads its gates."""
        launch = getattr(self, "_smooth_launch", None)
        if launch:
            launch["volume"] = None
        self.ui.show_qc_bar(busy=False)

    def _cleanup_smooth_thread(self):
        self._smooth_thread = None
        self._smooth_worker = None

    # ------------------------------------------------------------------
    # QC Analysis — Export
    # ------------------------------------------------------------------

    def _finalize(self) -> bool:
        """Finalize: save volume + CSV to disk, mark as finalized, and
        record the files' fingerprint (F20) so a later edit is caught."""
        if self._current_stack_name is None:
            return False
        if self._refuse_while_locked("Finalize"):
            return False
        stack = self._current_stack_name

        from ..ui_shell.theme import themed_confirm
        if not themed_confirm(
            self.ui, S4T.FINALIZE_TITLE,
            S4T.FINALIZE_CONFIRM.format(stack=stack),
            confirm_text=S4T.FINALIZE_BTN, cancel_text=S4T.CANCEL_BTN,
            accent="green",
        ):
            return False
        if not self._same_stack_after_modal(stack, "Finalize"):
            return False

        self._ensure_writable_state()
        if getattr(self.state_manager, "save_blocked", False):
            self.ui.log_output.append(f"[state] {S4T.STATE_SAVE_BLOCKED}")
            return self._save_failed(stack, "the state file is blocked")

        from datetime import datetime, timezone
        record = self.state_manager.get_record(stack)
        self._set_step4_flag("_save_in_flight", True)
        try:
            try:
                saved = self._write_stack_files(stack, record, "finalize")
            except Exception as exc:      # noqa: BLE001
                return self._save_failed(stack, exc)
            missing = self._unwritten_edits()
            if missing:
                return self._save_failed(stack, missing)

            # Save EDA state (PCA, UMAP, t-SNE caches)
            self._save_eda_state()

            fields = dict(self._held_edit_fields(stack))
            fields["finalized_at"] = datetime.now(timezone.utc).isoformat()
            fields["finalized_fingerprint"] = S4M.finalized_fingerprint(
                record.get("labels_path", ""),
                record.get("qc_features_csv", ""))
            self.state_manager.set_stack_status(stack, "finalized", **fields)
            if not self._persist_state():
                self.state_manager.restore_record(stack, record)   # D5
                return self._save_failed(stack, "the state file was not saved")
        finally:
            self._set_step4_flag("_save_in_flight", False)

        self._has_unsaved_changes = False
        self._pending_edits = {}
        self._undo_slot = None
        self._refresh_stacks()
        self.ui.log_output.append(
            f"[finalize] {stack}: {' + '.join(saved) or 'nothing written'}")
        self.ui.show_mt_status(S4T.MT_FINALIZED.format(stack=stack))
        self._refresh_mt_panel()
        return True

    # ------------------------------------------------------------------
    # The Export window
    # ------------------------------------------------------------------

    def _export_rows(self):
        """Every reconstructed stack's Export row."""
        records = {}
        for name in self.state_manager.get_all_stack_statuses():
            if Path(name).suffix.lower() not in S4R.STACK_SUFFIXES:
                continue
            records[name] = self.state_manager.get_record(name)
        unsaved = (self._current_stack_name
                   if self._labels_volume is not None
                   and self._has_unsaved_changes else None)
        rows = S4R.export_rows(records, unsaved=unsaved)
        for r in rows:
            if r.get("legacy_finalized"):
                self.ui.log_output.append(
                    f"[export] {r['name']} was finalized before Finalize "
                    f"recorded its files; it is taken as it is")
        return rows

    def _refresh_export_window(self, keep_ticks=False):
        dlg = getattr(self.ui, "export_dialog", None)
        if dlg is None:
            return
        rows = self._export_rows()
        dlg.set_rows(rows, keep_unticked=keep_ticks)
        loaded = (self._current_stack_name
                  if self._labels_volume is not None else None)
        dlg.set_mesh_target(loaded, self._mesh_folder_text(loaded))
        exporting = bool(getattr(self, "_export_running", False)
                         or getattr(self, "_mesh_running", False))
        locked = _step4_is_locked(self)
        # B2: any Step 4 job greys the window's buttons; an export's
        # own line is EXPORT_RUNNING, any other job's is its lock sentence.
        dlg.set_busy(locked, reason="" if exporting or not locked
                     else self._lock_line_text())
        last = self.state_manager.get_session_metadata().get("last_export")
        if isinstance(last, dict) and last.get("path"):
            dlg.show_last(S4T.say(S4T.EXPORT_LAST,
                                  k=int(last.get("n_stacks") or 0),
                                  file=Path(last["path"]).name,
                                  **{"hh:mm": self._hhmm(last.get("at"))}))
        else:
            dlg.show_last("")
        print(f"[STEP4-UI] export window: {len(rows)} rows, "
              f"{sum(1 for r in rows if r['can_go'])} can go")

    def _open_export_window(self):
        """Export…: workspace-wide, live with nothing loaded."""
        if self._refuse_while_locked("Export"):
            return
        dlg = getattr(self.ui, "export_dialog", None)
        if dlg is None:
            return
        self._refresh_export_window()
        dlg.show()
        dlg.raise_()

    def _export_all_stacks(self):
        """Export features over the TICKED finalized stacks."""
        if self._refuse_while_locked("Export features"):
            return
        if not self.working_dir:
            return
        dlg = getattr(self.ui, "export_dialog", None)
        rows = {r["name"]: r for r in self._export_rows()}
        ticked = (dlg.ticked_names() if dlg is not None
                  else [n for n, r in rows.items() if r["can_go"]])
        names = [n for n in ticked if rows.get(n, {}).get("can_go")]
        if len(names) != len(ticked):
            # B1-3: a ticked stack can no longer go (edited, re-measured
            # since the window was filled): never drop it silently
            self._refresh_export_window(keep_ticks=True)
            self._export_status(S4T.EXPORT_ROWS_CHANGED)
            return
        if not names:
            self._export_status(S4T.EXPORT_NONE_TICKED
                                if rows and any(r["can_go"]
                                                for r in rows.values())
                                else S4T.EXPORT_BLOCK)
            return
        if self._current_stack_name in names and self._has_unsaved_changes:
            # the table holds each stack as it was finalized;
            # a Save now would take this one out of Export (F20).
            self.ui.log_output.append(
                f"[export] {self._current_stack_name}: its unsaved edits are "
                f"not in the table (it goes as it was finalized)")

        import pandas as pd
        from qtpy.QtWidgets import (
            QCheckBox, QDialog, QDialogButtonBox, QLabel, QVBoxLayout,
        )
        from napari_epicycle.fusion_reconstruction.qc_features import (
            merge_patch_csvs, save_patch_csv,
        )
        csv_paths, stack_names = [], []
        for name in names:
            csv = self.state_manager.get_record(name).get("qc_features_csv")
            csv_paths.append(csv)
            stack_names.append(name)
        per_stack_cols = {}
        for csv_path, sname in zip(csv_paths, stack_names):
            per_stack_cols[sname] = set(pd.read_csv(csv_path,
                                                    nrows=0).columns)
        all_cols = list(per_stack_cols.values())
        shared_cols = set.intersection(*all_cols) if all_cols else set()
        union_cols = set.union(*all_cols) if all_cols else set()
        missing_cols = union_cols - shared_cols

        shared_only = True
        if missing_cols:
            # The column-intersection question, kept
            qd = QDialog(self.ui)
            qd.setWindowTitle(S4T.EXPORT_FILE_TITLE)
            qd.setMinimumWidth(450)
            qd.setStyleSheet(
                "QDialog { background: #2b2b2b; border: 1px solid "
                "rgba(255,255,255,0.12); border-radius: 6px; } QLabel { "
                "color: #dddddd; font-size: 11px; }")
            lay = QVBoxLayout(qd)
            lay.setContentsMargins(20, 16, 20, 16)
            lay.setSpacing(8)
            parts = [S4T.EXPORT_COLS_SHARED.format(n=len(shared_cols)),
                     S4T.EXPORT_COLS_NOT_SHARED.format(
                         n=len(missing_cols))]
            for col in sorted(missing_cols):
                missing_from = [s for s, c in per_stack_cols.items()
                                if col not in c]
                if len(missing_from) <= 2:
                    parts.append(S4T.EXPORT_COLS_MISSING_FROM.format(
                        col=col, stacks=", ".join(missing_from)))
            parts.append(S4T.EXPORT_COLS_RULE)
            info = QLabel("\n".join(parts))
            info.setWordWrap(True)
            lay.addWidget(info)
            shared_only_cb = QCheckBox(S4T.EXPORT_COLS_SHARED_ONLY)
            shared_only_cb.setChecked(True)
            lay.addWidget(shared_only_cb)
            btns = QDialogButtonBox()
            btns.addButton(S4T.CANCEL_BTN, QDialogButtonBox.RejectRole)
            btns.addButton(S4T.EXPORT_COLS_EXPORT_BTN,
                           QDialogButtonBox.AcceptRole)
            btns.accepted.connect(qd.accept)
            btns.rejected.connect(qd.reject)
            lay.addWidget(btns)
            if qd.exec_() != QDialog.Accepted:
                return
            shared_only = shared_only_cb.isChecked()

        from datetime import date as _date
        default_name = str(
            Path(self.working_dir) / "artifacts" / "step4" / "exports"
            / S4T.EXPORT_DEFAULT_NAME.format(date=_date.today().isoformat(),
                                             n=len(names)))
        path, _ = QFileDialog.getSaveFileName(
            None, S4T.EXPORT_FILE_TITLE, default_name, "CSV Files (*.csv)")
        if not path:
            return

        self._set_step4_flag("_export_running", True)
        self._export_busy(True)
        try:
            # B1-8: every frame tagged with its record name (via
            # merge_patch_csvs one file at a time), never a CSV stem
            combined = S4R.merge_stack_csvs(dict(zip(stack_names,
                                                     csv_paths)))
            if shared_only and missing_cols:
                drop = [c for c in missing_cols if c in combined.columns]
                if drop:
                    combined.drop(columns=drop, inplace=True)
                    self.ui.log_output.append(
                        f"[export] Dropped {len(drop)} non-shared columns")
            n_total = len(combined)
            if "label_status" in combined.columns:
                combined = combined[combined["label_status"] == "active"
                                    ].reset_index(drop=True)
            if "is_border_cell" in combined.columns:
                combined = combined[combined["is_border_cell"] != 1
                                    ].reset_index(drop=True)
            n_filtered = n_total - len(combined)
            combined = S4R.add_meta_columns(
                combined, {n: self.state_manager.get_record(n)
                           for n in names})
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            save_patch_csv(combined, path)
        except Exception as exc:      # noqa: BLE001
            self.ui.log_output.append(f"[ERROR] export: {exc}")
            self._export_status(S4T.EXPORT_FAILED)
            return
        finally:
            self._set_step4_flag("_export_running", False)
            self._export_busy(False)
        from datetime import datetime, timezone
        fname = Path(path).name
        done = S4T.say(S4T.EXPORT_FEATURES_DONE, k=len(names),
                       n=int(len(combined)), file=fname)
        log = done
        if n_filtered > 0:
            log += S4T.LOG_EXPORT_EXCLUDED.format(n=n_filtered)
        if missing_cols:
            log += S4T.LOG_EXPORT_NAN.format(k=len(missing_cols))
        self.ui.log_output.append(f"[export] {log} ({path})")
        now = datetime.now(timezone.utc).isoformat()
        try:
            self._ensure_writable_state()                    # B1-4
            self.state_manager.set_session_metadata(last_export={
                "at": now, "n_stacks": len(names), "path": path})
            for n in names:
                self.state_manager.update_record(n, exported_at=now)
            self._persist_state()
        except Exception as exc:      # noqa: BLE001
            print(f"[STEP4-EXPORT] last export not recorded: {exc}")
        self._refresh_export_window()
        self._export_status(done, tip=path)
        self.ui.show_qc_bar(value=100, text=S4T.QC_BAR_IDLE)

    def _export_status(self, text, tip=None):
        dlg = getattr(self.ui, "export_dialog", None)
        if dlg is not None:
            dlg.show_status(text, tip)

    def _export_busy(self, busy):
        dlg = getattr(self.ui, "export_dialog", None)
        if dlg is not None:
            dlg.set_busy(busy)
            if busy:
                dlg.show_status(S4T.EXPORT_RUNNING)
        self.ui.show_qc_bar(busy=True if busy else False,
                            text="" if busy else None)

    # ------------------------------------------------------------------
    # 3D Mesh Export
    # ------------------------------------------------------------------

    def _export_meshes(self):
        """Export each cell of the LOADED stack as an OBJ (subprocess).
        Saves first when there are unsaved edits; a failed Save stops it,
        so meshes are never made from old cells (F23)."""
        if self._refuse_while_locked("Export meshes"):
            return
        if self._labels_volume is None or self._current_stack_name is None:
            self._export_status(S4T.EXPORT_MESH_NONE)
            return
        stack = self._current_stack_name
        if self._has_unsaved_changes:
            from ..ui_shell import theme as th
            body = S4T.MESH_SAVE_FIRST.format(stack=stack)
            if self.state_manager.get_stack_status(stack) == "finalized":
                body += S4T.SAVE_TAKES_OUT_OF_EXPORT
            if not th.themed_confirm(
                self.ui, S4T.MESH_SAVE_FIRST_TITLE, body,
                confirm_text=S4T.MESH_SAVE_FIRST_BTN,
                cancel_text=S4T.CANCEL_BTN, accent="green",
            ):
                return
            if not self._save_tier_assignments(confirm=False):
                self._export_status(S4T.EXPORT_MESH_FAILED)
                return
        record = self.state_manager.get_record(stack)
        labels_path = record.get("labels_path", "")
        if not labels_path or not Path(labels_path).exists():
            self.ui.log_output.append(
                f"[ERROR] mesh export: labels file not found "
                f"({labels_path or '(none)'})")
            self._export_status(S4T.EXPORT_MESH_FAILED)
            return
        mesh_dir = (Path(self.working_dir) / "artifacts" / "step4" / "meshes"
                    / stack)
        spacing = self._get_spacing()
        n_labels = int(getattr(self, "_n_cells", 0) or 0) or (
            len(np.unique(self._labels_volume)) - 1)
        mesh_dir.mkdir(parents=True, exist_ok=True)
        sp_tuple = (f"({spacing[0]}, {spacing[1]}, {spacing[2]})"
                    if spacing else "None")
        script = f"""
import sys
sys.path.insert(0, {str(Path.cwd())!r})
import tifffile
labels = tifffile.imread({labels_path!r})
from napari_epicycle.fusion_reconstruction.qc_features import export_all_cell_meshes
manifest = export_all_cell_meshes(
    labels,
    {str(mesh_dir)!r},
    {stack!r},
    spacing={sp_tuple},
)
del labels
import gc; gc.collect()
print(f"DONE:{{len(manifest)}}")
_skipped = manifest.attrs.get("skipped_labels", [])
print(f"SKIPPED:{{len(_skipped)}}")
print("SKIPPED_IDS:" + ",".join(str(i) for i in _skipped))
"""
        script_path = mesh_dir / "_mesh_export.py"
        script_path.write_text(script, encoding="utf-8")
        import subprocess as sp
        import sys
        self._mesh_proc = sp.Popen([sys.executable, str(script_path)],
                                   stdout=sp.PIPE, stderr=sp.PIPE, text=True)
        self._mesh_job = {"dir": mesh_dir, "n": n_labels,
                          "script": script_path, "stack": stack}
        self._set_step4_flag("_mesh_running", True)
        self._export_busy(True)
        self.ui.show_qc_bar(value=0, percent=True)
        from qtpy.QtCore import QTimer
        self._mesh_timer = QTimer()
        self._mesh_timer.setInterval(1000)
        self._mesh_timer.timeout.connect(self._on_mesh_poll)
        self._mesh_timer.start()
        print(f"[STEP4-EXPORT] meshes of {stack}: {n_labels} cells")

    def _on_mesh_poll(self):
        """The poll's timer slot: a bound method holding its arguments on
        ``self``, never a lambda."""
        job = getattr(self, "_mesh_job", None) or {}
        self._poll_mesh_export(job.get("dir"), job.get("n", 0),
                               job.get("script"))

    def _poll_mesh_export(self, mesh_dir, n_labels, script_path):
        """Check if the mesh subprocess has finished."""
        proc = getattr(self, "_mesh_proc", None)
        timer = getattr(self, "_mesh_timer", None)
        if proc is None:
            if timer is not None:
                timer.stop()
            return
        ret = proc.poll()
        if ret is None:
            n_done = len(list(Path(mesh_dir).glob("cell_*.obj")))
            self.ui.show_qc_bar(value=int(n_done * 100 / max(n_labels, 1)),
                                percent=True)
            self._export_status(S4T.say(S4T.MT_MESH_RUNNING, i=n_done,
                                        n=n_labels))
            return
        if timer is not None:
            timer.stop()
        self._mesh_proc = None
        try:
            self._finish_mesh_export(proc, ret, mesh_dir, n_labels,
                                     script_path)
        finally:
            # B1-5: whatever the ending does, the lock is released
            if getattr(self, "_mesh_running", False):
                self._set_step4_flag("_mesh_running", False)
            try:
                self._export_busy(False)
            except Exception as exc:      # noqa: BLE001
                print(f"[STEP4-EXPORT] busy not cleared: {exc}")

    def _finish_mesh_export(self, proc, ret, mesh_dir, n_labels,
                            script_path):
        stdout = proc.stdout.read() if proc.stdout else ""
        stderr = proc.stderr.read() if proc.stderr else ""
        try:
            Path(script_path).unlink()
        except (OSError, TypeError):
            pass
        self._set_step4_flag("_mesh_running", False)
        self._export_busy(False)
        # Refresh ③ (and with it the Export window) BEFORE the ending's
        # sentence: the window's refresh writes its own grey reason
        # (EXPORT_BLOCK) and used to overwrite the done line (driver
        # 2026-09-29).
        self._refresh_mt_panel()
        stack = (getattr(self, "_mesh_job", None) or {}).get("stack", "")
        if ret == 0 and "DONE:" in stdout:
            fields = {}
            for line in stdout.splitlines():
                key, sep, value = line.partition(":")
                if sep and key in ("DONE", "SKIPPED", "SKIPPED_IDS"):
                    fields[key] = value.strip()
            n_exported = int(fields.get("DONE") or 0)
            k = int(fields.get("SKIPPED") or 0)
            if k:
                self._export_status(S4T.say(
                    S4T.EXPORT_MESH_DONE_SKIPPED, n=n_exported,
                    stack=stack, k=k), tip=str(mesh_dir))
                self.ui.log_output.append(S4T.LOG_MESH_SKIPPED.format(
                    k=k, ids=fields.get("SKIPPED_IDS", "")))
            else:
                self._export_status(S4T.say(S4T.EXPORT_MESH_DONE,
                                            n=n_exported, stack=stack),
                                    tip=str(mesh_dir))
            self.ui.log_output.append(
                f"[export] {n_exported} cell meshes → {mesh_dir}")
            self.ui.log_output.append(
                f"[export] Manifest: {Path(mesh_dir) / 'mesh_manifest.csv'}")
            self.ui.show_qc_bar(value=100, text=S4T.QC_BAR_IDLE)
        else:
            self._export_status(S4T.EXPORT_MESH_FAILED)
            self.ui.log_output.append(
                f"[ERROR] Mesh export subprocess (rc={ret}): {stderr[:500]}")
            self.ui.show_qc_bar(value=0, text="")

    # ------------------------------------------------------------------
    # Reset to Original
    # ------------------------------------------------------------------

    def _reset_to_original(self):
        """Restore the original reconstruction mask (undo all edits)."""
        if not self._current_stack_name:
            return
        record = self.state_manager.get_record(self._current_stack_name)
        original_path = record.get("original_labels_path", "")
        if not original_path or not Path(original_path).exists():
            self.ui.show_mt_status(S4T.RESET_NO_BACKUP.format(
                stack=self._current_stack_name))
            return

        if _step4_is_locked(self):
            self.ui.log_output.append(
                "[reset] refused: a Step 4 job is running")
            return
        self._ensure_writable_state()

        from ..ui_shell.theme import themed_confirm
        if not themed_confirm(
            self.ui, S4T.RESET_TITLE,
            S4T.RESET_CONFIRM.format(stack=self._current_stack_name),
            confirm_text=S4T.RESET_CONFIRM_BTN,
            cancel_text=S4T.CANCEL_BTN, accent="red",
        ):
            return

        import shutil
        import tifffile
        stack = self._current_stack_name
        labels_path = record.get("labels_path", "")
        # Atomic: copy the original next to the labels, then replace (F5).
        tmp = labels_path + ".tmp"
        try:
            shutil.copy2(original_path, tmp)
            os.replace(tmp, labels_path)
            volume = tifffile.imread(labels_path)
        except Exception as exc:      # noqa: BLE001
            try:
                os.unlink(tmp)
            except OSError:
                pass
            self.ui.log_output.append(f"[ERROR] reset {stack}: {exc}")
            self.ui.show_mt_status(S4T.MT_RESET_FAILED.format(stack=stack))
            return
        self._labels_volume = volume
        self._labels_epoch += 1
        self._qc_features_df = None
        self._last_pca_projected = None
        self._cached_eda_state = None
        self._has_unsaved_changes = False
        self._pending_edits = {}
        self._undo_slot = None
        self._pending_border_removal = False                  # audit D14
        self._clear_workbench_caches()

        # Every measure key off the record, every derived file deleted --
        # never renamed or kept (F5, I-5).
        self.state_manager.reset_to_reconstructed(stack)
        self._delete_derived_files(stack, record)
        self._persist_state()

        # Update viewer
        lyr = self._find_labels_layer()
        if lyr is not None:
            lyr.data = self._labels_volume
            lyr.refresh()

        ids = np.unique(self._labels_volume)
        self._n_cells = int(ids.size - (1 if ids.size and ids[0] == 0
                                        else 0))
        self.ui.show_mt_status(S4T.say(S4T.MT_RESET_DONE, stack=stack,
                                       n=self._n_cells))
        self.ui.log_output.append(
            f"[reset] Restored original mask from {Path(original_path).name}"
        )
        self._refresh_stacks()
        self._refresh_mt_panel()

    # ------------------------------------------------------------------
    # Step lifecycle hooks
    # ------------------------------------------------------------------

    def on_step_exit(self):
        """Called when user navigates away from Step 4.  Returns False to
        STOP the step change (F23 / audit D13: the save before leaving
        failed); the shell honours an explicit False only."""
        if _step4_is_locked(self):
            # The nav lock makes this unreachable; a programmatic change
            # while a job runs is refused the same way.
            print(f"[STEP4-LOCK] step exit refused while "
                  f"{self._running_flag_names()}")
            return False
        if self._has_unsaved_changes:
            from ..ui_shell.theme import themed_confirm
            if themed_confirm(
                self.ui,
                S4T.STEP_EXIT_TITLE,
                S4T.STEP_EXIT_BODY,
                confirm_text=S4T.STEP_EXIT_SAVE_BTN,
                cancel_text=S4T.STEP_EXIT_LEAVE_BTN,
                accent="green",
            ):
                if not self._save_tier_assignments(confirm=False):
                    stack = self._current_stack_name or ""
                    msg = S4T.STEP_EXIT_SAVE_FAILED.format(stack=stack)
                    self.ui.log_output.append("[save] " + msg)
                    self.ui.show_mt_status(
                        S4T.STEP_EXIT_SAVE_FAILED_SHORT.format(stack=stack),
                        msg)
                    try:
                        from ..ui_shell import theme as _TH
                        _TH.themed_notice(self.ui,
                                          S4T.STEP_EXIT_SAVE_FAILED_TITLE,
                                          msg)
                    except Exception as exc:      # noqa: BLE001
                        print(f"[STEP4-LOCK] exit notice failed: {exc}")
                    acc = getattr(self.ui, "accordion", None)
                    if acc is not None and hasattr(acc, "setCurrentIndex"):
                        acc.setCurrentIndex(2)          # ③, where it shows
                    print("[STEP4-LOCK] step exit stopped: the save "
                          "before leaving failed")
                    return False
        self._step4_active = False
        self._sync_step4_nav_lock()
        aw = getattr(self, "_analysis_widget", None)
        if aw is not None and hasattr(aw, "set_step_inactive"):
            try:
                aw.set_step_inactive(True)                  # K7
            except RuntimeError:
                pass
        if self.state_manager and not getattr(self, "_adopt_only", False):
            # An exit after a read-only adopt writes nothing (L12).
            self._persist_state()
        llm = getattr(self, "layer_lifecycle_manager", None)
        if llm is not None:
            llm.clear_step("step4")
        dlg = getattr(self.ui, "export_dialog", None)
        if dlg is not None:
            dlg.hide()                             # hidden, not destroyed
        dock = self._find_eda_dock()
        if dock is not None:
            try:
                dock.hide()                        # hidden, not destroyed
                print("[STEP4-LOCK] step exit: Workbench dock hidden")
            except RuntimeError:
                pass
        return None

    def on_step_enter(self):
        """Called when user navigates to Step 4 (``_step4_active``;
        the Workbench comes back only when the user opens it)."""
        self._step4_active = True
        # L3: forget layers removed while away (and the data they hold)
        if self.viewer is not None:
            live = list(self.viewer.layers)
            self._stack_layers = [
                lyr for lyr in (getattr(self, "_stack_layers", None) or [])
                if any(lyr is x for x in live)]
        # F15: leaving Step 4 cleared its layers; a stack still loaded in
        # memory comes back as its labels layer
        if (self.viewer is not None and self._labels_volume is not None
                and self._current_stack_name
                and self._stack_layers_missing()):
            try:
                self._restore_stack_layers("re-entry")
            except Exception as exc:      # noqa: BLE001
                print(f"[STEP4-LAYERS] restore on re-entry failed: {exc}")
        # S1 (audit H3): a size set in Step 3 while we were away
        try:
            if self._reread_spacing_from_file():
                self._refresh_stacks()
        except Exception as exc:      # noqa: BLE001
            print(f"[STEP4-SPACING] re-read on enter failed: {exc}")
        aw = getattr(self, "_analysis_widget", None)
        if aw is not None and hasattr(aw, "set_step_inactive"):
            try:
                aw.set_step_inactive(False)                 # K7
            except RuntimeError:
                pass
        try:
            self._refresh_step4_lock()
        except Exception as exc:      # noqa: BLE001
            print(f"[STEP4-LOCK] refresh on enter failed: {exc}")
