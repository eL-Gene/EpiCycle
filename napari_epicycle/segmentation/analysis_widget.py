"""EDA Workbench — dockable widget for interactive feature analysis.

Phase 1: Dockable widget + dark theme + feature table with checkboxes,
grouped by category. Foundation for sparklines, transforms, and
dimensionality reduction in later phases.

Phase 2: Sparkline histograms, skewness indicators, normality tests
in the feature table.  Uses feature_stats.py for pure computation.

Step 4 revamp, brick B3a: one 578 px column -- the
"what you see" line, the pinned plot part (plot, legend, selection card,
busy strip), a draggable divider, then the tabs Features · Map · Groups;
at a wide dock the feature panel moves to a 420 px left column.  Every
sentence and help line lives in ``_workbench_text.py``.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from qtpy.QtCore import QEvent, Qt, QTimer, Signal
from qtpy.QtGui import QBrush, QColor, QFont, QKeySequence, QPainter, QPixmap
from qtpy.QtWidgets import (
    QAbstractItemView,
    QAbstractScrollArea,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QShortcut,
    QSplitterHandle,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from napari_epicycle.segmentation import _workbench_text as T
from napari_epicycle.segmentation._stack_row_delegate import (
    CHECK_DISABLED_STYLE,
)

H = T.WORKBENCH_HELP_LINES


# ---------------------------------------------------------------------------
# Feature category definitions
# ---------------------------------------------------------------------------

# Map feature names to their category for grouped display
_FEATURE_CATEGORIES = {
    # Core morphology
    "area": "Morphology",
    "area_filled": "Morphology",
    "equivalent_diameter": "Morphology",
    "extent": "Morphology",
    "euler_number": "Morphology",
    "bbox_volume": "Morphology",
    "compactness": "Morphology",
    "fill_ratio": "Morphology",
    # Shape / orientation
    "sphericity": "Shape",
    "aspect_ratio": "Shape",
    "z_span": "Shape",
    "z_elongation": "Shape",
    "xy_elongation": "Shape",
    "ellipsoid_solidity": "Shape",
    "discrete_compactness": "Shape",
    "major_axis_z_angle": "Shape",
    "inertia_elongation": "Shape",
    "inertia_flatness": "Shape",
    "axis_major_length": "Shape",
    "axis_minor_length": "Shape",
    # Orientation (principal axis directions + eigenvalues)
    "major_axis_z": "Orientation",
    "major_axis_y": "Orientation",
    "major_axis_x": "Orientation",
    "mid_axis_z": "Orientation",
    "mid_axis_y": "Orientation",
    "mid_axis_x": "Orientation",
    "minor_axis_z": "Orientation",
    "minor_axis_y": "Orientation",
    "minor_axis_x": "Orientation",
    "inertia_eigval_major": "Orientation",
    "inertia_eigval_mid": "Orientation",
    "inertia_eigval_minor": "Orientation",
    # Surface
    "surface_area": "Surface",
    "true_sphericity": "Surface",
    "roughness": "Surface",
    "solidity": "Surface",
    # NOTE: Intensity and Membrane categories removed from Step 4.
    # These features are now computed per-channel via compute_channel_intensity
    # (11 features each). Dynamic channel categorization in _rebuild_table()
    # handles them as "Channel: {name}" groups.
    # Probability
    "prob_mean": "Probability",
    "prob_p10": "Probability",
    "prob_std": "Probability",
    "prob_boundary_mean": "Probability",
    "prob_interior_mean": "Probability",
    "prob_boundary_drop": "Probability",
    "prob_weighted_volume": "Probability",
    "prob_soft_hard_ratio": "Probability",
    "prob_boundary_entropy": "Probability",
    "prob_mask_entropy": "Probability",
    "prob_edge_gradient": "Probability",
    "prob_edge_gradient_std": "Probability",
    # Neighbor / adjacency
    "neighbor_count": "Neighbors",
    "nearest_neighbor_distance": "Neighbors",
    "max_contact_fraction": "Neighbors",
    "background_fraction": "Neighbors",
    "nn_count_r1": "Neighbors",
    "nn_count_r3": "Neighbors",
    "nn_birth_median": "Neighbors",
    "nn_birth_std": "Neighbors",
    "nn_accumulation_rate": "Neighbors",
    "nn_accumulation_knee": "Neighbors",
    "mean_neighbor_size_ratio": "Neighbors",
    "neighbor_size_cv": "Neighbors",
    "contact_weighted_neighbor_size_ratio": "Neighbors",
    # Z-stratified topology
    "basal_neighbor_count": "Topology",
    "apical_neighbor_count": "Topology",
    "neighbor_z_turnover": "Topology",
    "neighbor_persistence_ratio": "Topology",
    # Skeleton
    "skeleton_branches": "Skeleton",
    "skeleton_nodes": "Skeleton",
    "skeleton_length": "Skeleton",
    # Cross-section (epithelial)
    "apical_area": "Cross-Section",
    "basal_area": "Cross-Section",
    "apical_basal_area_ratio": "Cross-Section",
    "max_cross_section_z_relative": "Cross-Section",
    "cross_section_taper": "Cross-Section",
    "cross_section_bulge": "Cross-Section",
    # Named composites
    "merge_signal": "Composites",
    "fragment_signal": "Composites",
    "boundary_confidence": "Composites",
    "shape_abnormality": "Composites",
    "qc_score": "Composites",
    # Position (centroid coordinates)
    "centroid-0": "Position",
    "centroid-1": "Position",
    "centroid-2": "Position",
}

_CATEGORY_ORDER = [
    "Morphology", "Shape", "Orientation", "Surface",
    "Probability", "Neighbors", "Topology", "Skeleton",
    "Cross-Section", "Composites", "Position",
]

# One-line descriptions for feature name tooltips (from Feature Reference)
_FEATURE_DESCRIPTIONS = {
    # Core morphology
    "area": "Total volume of the cell mask (voxels or µm³). Abnormally large → merge, tiny → fragment.",
    "area_filled": "Volume with all internal holes filled. Compare with area to detect hollow cells.",
    "equivalent_diameter": "Diameter of a sphere with the same volume. Size-normalized comparison.",
    "extent": "Fraction of bounding box filled. Low = sparse/elongated, high = compact.",
    "euler_number": "Topological invariant: 1 = solid, <1 = holes or tunnels. Detects donut artifacts.",
    "bbox_volume": "Volume of the axis-aligned bounding box.",
    "compactness": "area / bbox_volume. 1.0 = fills entire box. Lower = more irregular.",
    "fill_ratio": "area / area_filled. 1.0 = no internal holes. <1 = has cavities.",
    # Shape / orientation
    "sphericity": "How sphere-like the bounding box proportions are.",
    "aspect_ratio": "Longest / shortest bounding box edge. High = very elongated.",
    "z_span": "Extent along Z. Very short = flat, very tall = columnar.",
    "z_elongation": "z_span / max(Y, X). >1 = vertically elongated, <1 = flat.",
    "xy_elongation": "max(Y, X) / min(Y, X). >1 = elliptical cross-section, 1.0 = circular.",
    "ellipsoid_solidity": "Volume / best-fit ellipsoid volume. >1 = extends beyond, <1 = smaller.",
    "major_axis_z_angle": "Angle between major inertia axis and Z (degrees). 0° = vertical, 90° = horizontal.",
    "inertia_elongation": "I₀/I₂ (largest/smallest eigenvalue). Higher = more elongated.",
    "inertia_flatness": "I₁/I₂ (middle/smallest eigenvalue). Higher = more disc-like.",
    "axis_major_length": "Length of the longest principal axis.",
    "axis_minor_length": "Length of the shortest principal axis.",
    # Orientation
    "major_axis_z": "Major principal axis Z-component. Defines cell orientation direction.",
    "major_axis_y": "Major principal axis Y-component.",
    "major_axis_x": "Major principal axis X-component.",
    "mid_axis_z": "Middle principal axis Z-component.",
    "mid_axis_y": "Middle principal axis Y-component.",
    "mid_axis_x": "Middle principal axis X-component.",
    "minor_axis_z": "Minor principal axis Z-component.",
    "minor_axis_y": "Minor principal axis Y-component.",
    "minor_axis_x": "Minor principal axis X-component.",
    "inertia_eigval_major": "Largest inertia eigenvalue. Proportional to spread along major axis.",
    "inertia_eigval_mid": "Middle inertia eigenvalue.",
    "inertia_eigval_minor": "Smallest inertia eigenvalue. Low = elongated along that axis.",
    # Surface
    "surface_area": "Triangle mesh surface area (marching cubes). Sensitive to roughness.",
    "true_sphericity": "Ψ = 36π·V²/SA³. 1.0 = perfect sphere. Lower = irregular.",
    "roughness": "SA / convex_hull_SA. >1 = bumps/indentations beyond convex hull.",
    "solidity": "V / convex_hull_V. 1.0 = perfectly convex. Lower = concavities/lobes.",
    "discrete_compactness": "Bribiesca C_d = (n - A_v/6) / (n - n^(2/3)). 1.0 = perfect cube; voxel-face based, orthogonal to true_sphericity.",
    # NOTE: Per-seg-channel intensity descriptions removed.
    # Channel-specific descriptions are generated dynamically by
    # _get_channel_description() below.
    # Probability
    "prob_mean": "Overall segmentation confidence. Low = uncertain.",
    "prob_p10": "10th percentile probability. Low = weak evidence at cell edges.",
    "prob_std": "Variation in probability values within the mask.",
    "prob_boundary_mean": "Mean probability at boundaries.",
    "prob_interior_mean": "Mean probability in interior.",
    "prob_boundary_drop": "Probability drop at boundaries. Large = well-defined edges.",
    "prob_weighted_volume": "Soft volume. Low ratio vs mask = extends beyond confident region.",
    "prob_soft_hard_ratio": "Soft/hard volume ratio. Low = mask extends beyond confident region.",
    "prob_boundary_entropy": "Shannon entropy at boundaries. High = uncertain segmentation.",
    "prob_mask_entropy": "Shannon entropy of mask probabilities. High = uncertain.",
    "prob_edge_gradient": "Gradient magnitude at boundaries. High = sharp edges (good).",
    "prob_edge_gradient_std": "Variation in edge sharpness. High = inconsistent boundaries.",
    # Neighbors
    "neighbor_count": "Number of distinct labels touching this cell.",
    "nearest_neighbor_distance": "EDT distance to closest neighbor (voxels).",
    "max_contact_fraction": "Largest shared surface / total surface. High = one dominant neighbor.",
    "background_fraction": "Surface touching background. High = isolated cell.",
    "nn_count_r1": "Neighbors at radius 1 (direct contact). The 'hard' neighbor count.",
    "nn_count_r3": "Neighbors within radius 3. Includes near-neighbors across 1-2 voxel gaps.",
    "nn_birth_median": "Median birth radius. Typical gap width to neighbors (more robust than min).",
    "nn_birth_std": "Std of birth radii. Low = uniform packing, high = heterogeneous.",
    "nn_accumulation_rate": "Slope of cumulative neighbors vs radius. Steep = dense tissue.",
    "nn_accumulation_knee": "Radius at 80% of total neighbors. Early = tight, late = scattered.",
    "mean_neighbor_size_ratio": "Cell volume / mean neighbor volume. >2 = merge, <0.5 = fragment.",
    "neighbor_size_cv": "CV of neighbor volumes. High = boundary between different-sized populations.",
    "contact_weighted_neighbor_size_ratio": "Size ratio weighted by contact area. Emphasizes true neighbors.",
    # Topology
    "basal_neighbor_count": "Distinct neighbors in bottom third of Z-range.",
    "apical_neighbor_count": "Distinct neighbors in top third of Z-range.",
    "neighbor_z_turnover": "1 − |basal∩apical|/|basal∪apical|. 0 = same neighbors, 1 = completely different.",
    "neighbor_persistence_ratio": "|basal∩apical| / total. High = stable columnar packing.",
    # Skeleton
    "skeleton_branches": "Number of branches. High = complex/lobulated shape.",
    "skeleton_nodes": "Number of branch points (junctions).",
    "skeleton_length": "Total skeleton path length (voxels).",
    # Cross-section
    "apical_area": "Mean cross-section area in top 20% of Z-range.",
    "basal_area": "Mean cross-section area in bottom 20% of Z-range.",
    "apical_basal_area_ratio": "apical/basal. <0.5 = pyramidal, ~1 = columnar, >2 = inverted.",
    "max_cross_section_z_relative": "Z of widest slice [0,1]. 0 = bottom, 0.5 = barrel, 1 = top.",
    "cross_section_taper": "Slope of area vs Z. Negative = pyramid, positive = expanding, ~0 = columnar.",
    "cross_section_bulge": "RMS residual / mean area. High = vase, barrel, or dumbbell (merge signal).",
    # Composites
    "merge_signal": "Weighted composite: how merge-like is this cell? High = suspicious.",
    "fragment_signal": "Weighted composite: how fragment-like? High = suspicious.",
    "boundary_confidence": "Probability-based boundary confidence. High = good. Requires prob features.",
    "shape_abnormality": "Population-relative shape outlier score. High = unusual shape.",
    "qc_score": "Mean |robust z-score| across all features. Higher = more unusual cell.",
    # Position
    "centroid-0": "Cell centroid Z coordinate (voxels). Spatial position along Z axis.",
    "centroid-1": "Cell centroid Y coordinate (voxels). Spatial position along Y axis.",
    "centroid-2": "Cell centroid X coordinate (voxels). Spatial position along X axis.",
}

# Dynamic channel feature descriptions
_CHANNEL_STAT_DESCRIPTIONS = {
    "_intensity_mean": "Average pixel brightness inside the cell mask.",
    "_intensity_median": "Median pixel brightness inside the cell mask.",
    "_intensity_std": "Variation in brightness. High = heterogeneous staining.",
    "_intensity_min": "Darkest voxel in the mask.",
    "_intensity_max": "Brightest voxel in the mask.",
    "_intensity_sum": "Total brightness across all voxels in the mask.",
    "_intensity_cv": "Coefficient of variation (std/mean). Normalized variability.",
    "_boundary_intensity_mean": "Mean brightness in the 2-voxel outer shell.",
    "_interior_intensity_mean": "Mean brightness in the eroded core.",
    "_boundary_interior_ratio": "boundary/interior. >1 = bright membrane. ~1 = possible merge.",
    "_z_intensity_variation": "Std of per-Z-slice means. High = inconsistent staining or Z-artifact.",
}


def _get_feature_description(feat_name: str) -> str:
    """Return tooltip description, including dynamic channel features."""
    if feat_name in _FEATURE_DESCRIPTIONS:
        return _FEATURE_DESCRIPTIONS[feat_name]
    # Check channel feature suffixes
    for suffix, desc in _CHANNEL_STAT_DESCRIPTIONS.items():
        if feat_name.endswith(suffix):
            return desc
    return ""


# Import canonical suffix list from feature_registry (single source of truth).
# Sort longest-first so "_boundary_intensity_mean" matches before "_intensity_mean".
from napari_epicycle.segmentation.feature_registry import (
    _CHANNEL_SUFFIXES as _CHANNEL_SUFFIXES_RAW,
)
_CHANNEL_SUFFIXES = tuple(sorted(_CHANNEL_SUFFIXES_RAW, key=len, reverse=True))


def _extract_channel_prefix(feat_name: str) -> str | None:
    """Extract channel prefix from a per-channel feature name.

    Returns the prefix (e.g. ``"membrane"`` from ``"membrane_intensity_mean"``,
    or ``"membrane"`` from ``"membrane_boundary_intensity_mean"``), or None
    if the feature doesn't match any known channel suffix.
    """
    for suffix in _CHANNEL_SUFFIXES:
        if feat_name.endswith(suffix):
            prefix = feat_name[: -len(suffix)]
            if prefix:  # non-empty prefix
                return prefix
    return None


# Styling constants -- the family's neutral dock skin:
# the navy panes (#1e1e2e / #333348) go; green only on controls.
_BG = "#23242b"
_BG_CONTENT = "#23242b"
_TEXT = "#e0e0e0"
_TEXT_DIM = "#999999"
_ACCENT = "#6abf69"
_ACCENT_LIGHT = "#8bc34a"
_HEADER_BG = "#2a2c33"
_ROW_ALT = "rgba(255, 255, 255, 0.02)"
_SELECTED_BG = "rgba(255, 255, 255, 0.08)"
_AMBER = "#d4a843"
#: The bad tier's own orange (qc_plot_widget._TIER_COLORS["bad"]), for
#: Mark bad instead of today's red.
_BAD_ORANGE = "#FF7828"
_GOOD_GREEN = "#50C878"
#: The mockup's grey for unassigned (its plot colour is B3b's).
_UNASSIGNED_GREY = "#8b909a"

# Sparkline dimensions (a 46 x 16 sparkline)
_SPARK_W = 46
_SPARK_H = 16

# --- The Workbench layout ---------------------------------
#: The dock widths the layout is built for: tabbed beside the plugin, and
#: split out with the features in a left column.
WORKBENCH_NARROW_W = 578
WORKBENCH_WIDE_W = 1000
#: The left feature column at the wide width.
WORKBENCH_LEFT_COL_W = 420
#: Resize hysteresis (Appendix B R6): the feature panel moves
#: to the left column at WIDE_ON or wider, back into its tab below
#: WIDE_OFF.  The earlier 1020 / 980 put the 1000 px layout itself INSIDE the
#: band (a 1000 px dock stayed narrow when it grew from 578), so the band
#: sits below 1000 instead: 420 + 578 = 998 px is where the column fits.
WORKBENCH_WIDE_ON_PX = 1000
WORKBENCH_WIDE_OFF_PX = 960
#: The divider's own height (9 px, green on hover).
WORKBENCH_HANDLE_PX = 9
#: The plot height a divider reset gives (the mockup's default plot): 270 px
#: in the 578 px layout, 340 px split out (the mockup's ``sizePlot``).
WORKBENCH_PLOT_DEFAULT_PX = 270
WORKBENCH_PLOT_DEFAULT_WIDE_PX = 340
#: The least the tabs keep: the first reset waits until they can have it,
#: a reset never takes it, and the plot's kept height (a chrome line that
#: appears grows the top pane) gives way at it.
WORKBENCH_TABS_MIN_PX = 160
#: The pinned part's height WITHOUT the plot, at 578 px
#: (WORKBENCH_PINNED_PX): toolbar, the axes row wrapped for a feature colour,
#: legend, the selection card with every chip and the busy strip on.  The
#: mockup measures 331.8 px there (headless Chrome, 603.8 px pinned less its
#: 272 px plot, B3a mockup check); + 5 %.
WORKBENCH_PINNED_PX = 348


# ---------------------------------------------------------------------------
# Sparkline renderer (QPixmap)
# ---------------------------------------------------------------------------

def render_sparkline(
    counts: np.ndarray,
    max_count: int,
    *,
    bar_color: str = _ACCENT,
    bg_color: str = _BG_CONTENT,
    width: int = _SPARK_W,
    height: int = _SPARK_H,
) -> QPixmap:
    """Render a mini histogram as a QPixmap.

    Parameters
    ----------
    counts : array of int
        Bin counts from compute_histogram.
    max_count : int
        Maximum bin count (for normalization across features).
    bar_color : str
        Hex color for the bars.
    bg_color : str
        Hex color for the background.
    width, height : int
        Pixel dimensions of the output pixmap.

    Returns
    -------
    QPixmap
    """
    pixmap = QPixmap(width, height)
    pixmap.fill(QColor(bg_color))

    if max_count == 0 or len(counts) == 0:
        return pixmap

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, False)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QBrush(QColor(bar_color)))

    n_bins = len(counts)
    bar_w = max(1, width // n_bins)
    padding = 1  # 1px gap between bars

    for i, c in enumerate(counts):
        if c == 0:
            continue
        bar_h = max(1, int((c / max_count) * (height - 2)))
        x = i * bar_w
        y = height - bar_h - 1
        painter.drawRect(x, y, max(1, bar_w - padding), bar_h)

    painter.end()
    return pixmap


def skew_look(value: float) -> tuple:
    """The Skew column's arrow, number and colour.

    Amber when |s| > 1, green below 0.5, grey otherwise; the arrow points
    up-right for a long tail of large values, down-right for small ones.
    """
    if value is None or not np.isfinite(value):
        return "—", _TEXT_DIM
    a = abs(value)
    colour = _AMBER if a > 1.0 else (_ACCENT if a < 0.5 else "#bbbbbb")
    arrow = "↗" if value > 0.5 else ("↘" if value < -0.5 else "→")
    return f"{arrow} {value:.1f}", colour


def loading_tint(value: float) -> QColor:
    """A loading cell's tint: green pushes up the component, red down
    (the mockup's scale, alpha grows with |loading|)."""
    a = min(1.0, abs(float(value)) * 1.6) * 0.6
    if value > 0:
        return QColor(80, 200, 120, int(round(a * 255)))
    return QColor(240, 60, 60, int(round(a * 255)))


#: The words ``QCPlotWidget(layout="workbench")`` shows, passed in so the
#: Step 3 plot module never imports Step 4's sentence table (W1-10).
WORKBENCH_PLOT_TEXTS = {
    "help": T.WORKBENCH_HELP_LINES,
    "UMAP_EMPTY": T.UMAP_EMPTY,
    "TSNE_EMPTY": T.TSNE_EMPTY,
    "MAP_BUSY_EMPTY": T.MAP_BUSY_EMPTY,
    "VIOLIN_GROUPS": T.VIOLIN_GROUPS,
    "MAP_AXES": T.MAP_AXES,
    # B3b re-audit LOW-3: the legend's Step 4 words
    "UNMARKED_WORD": T.UNMARKED_WORD,
    "GROUP_WORD": T.GROUP_WORD,
    "NOISE_WORD": T.NOISE_WORD,
}

#: The Rescale combo's words (the mockup's XF_LABEL); the item DATA stays
#: the transform key every reader uses.
RESCALE_WORDS = {
    "raw": "none", "log": "log", "sqrt": "square root", "rank": "rank",
    "zscore": "z-score", "winsorize": "clip ends", "auto": "auto",
}


# ---------------------------------------------------------------------------
# The selection card's parts -- pure, Qt-free
# ---------------------------------------------------------------------------

#: Step 4's three tier buckets, in the card's order, with their colours:
#: the plot's good / bad (``qc_plot_widget._TIER_COLORS``) and the
#: Workbench's grey unassigned.  "garbage" counts as bad, anything else
#: (empty, NaN, a Step 3 "review") as unmarked -- the rule
#: ``_update_tier_summary`` has always used.
TIER_BUCKETS = (
    ("good", "#50C878"),
    ("bad", "#FF7828"),
    ("unmarked", "#8b909a"),
)


def tier_bucket(tier) -> str:
    """The Step 4 bucket of one ``qc_tier`` value -- the plot's own rule
    (``qc_plot_widget.wb_tier_bucket``), so the card, the legend and the
    drawn highlight always agree (audit E1-5)."""
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
        wb_tier_bucket,
    )
    return wb_tier_bucket(tier)


def tier_counts(tiers) -> dict:
    """``{"good": g, "bad": b, "unmarked": u}`` over ``tiers``."""
    out = {name: 0 for name, _ in TIER_BUCKETS}
    for t in tiers:
        out[tier_bucket(t)] += 1
    return out


def selection_card_parts(tiers, first_id=None) -> dict:
    """What the selection card shows for a selection whose tiers are
    ``tiers`` (one per selected cell): line 1, the mix bar's shares and
    the chips.  ``first_id`` names the one cell of SEL_ONE."""
    tiers = list(tiers)
    n = len(tiers)
    if n == 0:
        return {"count_text": T.SEL_NONE, "mix": [], "chips_html": ""}
    if n == 1:
        count_text = T.SEL_ONE.format(id=first_id)
    else:
        count_text = T.SEL_COUNT.format(m=f"{n:,}")
    counts = tier_counts(tiers)
    mix = [(colour, counts[name]) for name, colour in TIER_BUCKETS
           if counts[name]]
    chips = [T.CHIP_HTML.format(colour=colour, tier=name,
                                n=f"{counts[name]:,}")
             for name, colour in TIER_BUCKETS]
    return {"count_text": count_text, "mix": mix,
            "chips_html": T.CHIP_SEP.join(chips)}


def totals_line(tiers) -> str:
    """TOTALS_STEP4 over every active cell's tier."""
    c = tier_counts(tiers)
    return T.TOTALS_STEP4.format(g=f"{c['good']:,}", b=f"{c['bad']:,}",
                                 u=f"{c['unmarked']:,}")


# ---------------------------------------------------------------------------
# Small widgets of the Workbench layout
# ---------------------------------------------------------------------------

class _AutoHideLabel(QLabel):
    """A label that is on screen only while it has text (the card's totals
    line, which the controller writes: no empty band at the card's foot).
    ``text_set`` lets the Workbench keep the plot's height."""

    text_set = Signal()

    def setText(self, text):  # noqa: N802
        super().setText(text)
        self.setVisible(bool(text))
        self.text_set.emit()


class _KeyButton(QPushButton):
    """A QPushButton sized by its child labels (its own text is empty), so
    the word and the key letter are never cut once the theme is applied."""

    def sizeHint(self):  # noqa: N802
        hint = super().sizeHint()
        lay = self.layout()
        if lay is not None:
            inner = lay.sizeHint()
            hint.setWidth(max(hint.width(), inner.width()))
            hint.setHeight(max(hint.height(), inner.height(), 28))
        return hint

    def minimumSizeHint(self):  # noqa: N802
        return self.sizeHint()


def _key_button(text: str, key: str, parent, fg: str) -> QPushButton:
    """A button whose key letter is drawn dimmed after its word (mockup
    "Mark good  G").  The word and the letter are child labels that let the
    mouse through, so the button clicks as any button does."""
    btn = _KeyButton("", parent)
    lay = QHBoxLayout(btn)
    lay.setContentsMargins(12, 0, 12, 0)
    lay.setSpacing(6)
    word = QLabel(text, btn)
    word.setStyleSheet(f"font-weight: bold; font-size: 11px;"
                       f" background: transparent; color: {fg};")
    letter = QLabel(key, btn)
    letter.setStyleSheet("font-size: 11px; background: transparent;"
                         " color: rgba(255, 255, 255, 0.55);"
                         " font-family: Consolas, monospace;")
    for lab in (word, letter):
        lab.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        lay.addWidget(lab)
    btn._epicycle_word = word
    btn._epicycle_key = letter
    btn._epicycle_fg = fg
    return btn


def _key_button_enable(btn, enabled: bool):
    """Enable a key button AND dim its child labels: they carry their own
    colours, so a disabled one otherwise still reads as live (B3b shots)."""
    btn.setEnabled(enabled)
    fg = getattr(btn, "_epicycle_fg", "#dddddd") if enabled else "#6f6f6f"
    key = ("rgba(255, 255, 255, 0.55)" if enabled
           else "rgba(255, 255, 255, 0.25)")
    btn._epicycle_word.setStyleSheet(
        f"font-weight: bold; font-size: 11px; background: transparent;"
        f" color: {fg};")
    btn._epicycle_key.setStyleSheet(
        f"font-size: 11px; background: transparent; color: {key};"
        " font-family: Consolas, monospace;")


class _WorkbenchHandle(QSplitterHandle):
    """The divider between the plot part and the tabs.

    9 px, a grip line that turns green on hover; a double click resets the
    two panes (``reset_requested``).
    """

    reset_requested = Signal()

    def __init__(self, orientation, parent):
        super().__init__(orientation, parent)
        self.setAttribute(Qt.WA_Hover, True)
        self.setMouseTracking(True)
        self._hover = False

    def enterEvent(self, event):  # noqa: N802
        self._hover = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):  # noqa: N802
        self._hover = False
        self.update()
        super().leaveEvent(event)

    def mouseDoubleClickEvent(self, event):  # noqa: N802
        self.reset_requested.emit()
        event.accept()

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(255, 255, 255, 8))
        p.fillRect(0, 0, w, 1, QColor("#555555"))
        grip = QColor(_ACCENT) if self._hover else QColor(255, 255, 255, 72)
        p.fillRect(w // 2 - 18, 4, 36, 2, grip)
        p.end()


class _WorkbenchSplitter(QSplitter):
    """A vertical QSplitter whose handle is a ``_WorkbenchHandle``."""

    reset_requested = Signal()

    def createHandle(self):  # noqa: N802
        handle = _WorkbenchHandle(self.orientation(), self)
        handle.reset_requested.connect(self.reset_requested)
        return handle


class _TierMixBar(QWidget):
    """The selection card's tier-mix bar: one coloured share per tier."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._shares: list = []
        self.setFixedHeight(10)
        self.setMinimumWidth(60)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_counts(self, counts: list):
        """``counts`` = ``[(colour, n), ...]``; an empty list clears it."""
        self._shares = [(c, int(n)) for c, n in counts if int(n) > 0]
        self.update()

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(255, 255, 255, 15))
        total = sum(n for _, n in self._shares)
        x = 0.0
        for colour, n in self._shares:
            if total <= 0:
                break
            dw = w * n / total
            p.fillRect(int(round(x)), 0, int(round(x + dw)) - int(round(x)), h,
                       QColor(colour))
            x += dw
        p.end()


class _LoadingsHeader(QHeaderView):
    """The loadings table's column header.

    Each PC column draws a green variance bar (its share of the variation
    over PC1's share, at most 28 px), then ``PC{i}``, then ``{%}``; the
    bars together replace the scree plot.
    """

    BAR_MAX_PX = 28

    def __init__(self, parent=None):
        super().__init__(Qt.Horizontal, parent)
        self._shares: list = []
        self.setSectionsClickable(False)
        self.setHighlightSections(False)
        self.setFixedHeight(self.BAR_MAX_PX + 34)

    def set_shares(self, shares):
        self._shares = [float(s) for s in shares]
        self.viewport().update()

    def bar_height(self, pc_index: int) -> int:
        """The bar of PC ``pc_index`` (0-based), in px."""
        if not self._shares or self._shares[0] <= 0:
            return 1
        return max(1, int(round(self._shares[pc_index] / self._shares[0]
                                * self.BAR_MAX_PX)))

    def paintSection(self, painter, rect, logical):  # noqa: N802
        painter.save()
        painter.fillRect(rect, QColor(_HEADER_BG))
        painter.setPen(QColor(255, 255, 255, 30))
        painter.drawLine(rect.bottomLeft(), rect.bottomRight())
        font = QFont(painter.font())
        font.setPixelSize(10)
        painter.setFont(font)
        if logical == 0:
            painter.setPen(QColor("#aaaaaa"))
            painter.drawText(rect.adjusted(4, 0, -2, -4),
                             int(Qt.AlignLeft | Qt.AlignBottom), "Feature")
        else:
            i = logical - 1
            if i < len(self._shares):
                bh = self.bar_height(i)
                bw = max(4, int(rect.width() * 0.7))
                bx = rect.x() + (rect.width() - bw) // 2
                by = rect.y() + 2 + self.BAR_MAX_PX - bh
                painter.fillRect(bx, by, bw, bh, QColor(_ACCENT))
                painter.setPen(QColor("#aaaaaa"))
                text_top = rect.y() + self.BAR_MAX_PX + 4
                painter.drawText(rect.x(), text_top, rect.width(), 14,
                                 int(Qt.AlignHCenter | Qt.AlignTop),
                                 f"PC{i + 1}")
                painter.setPen(QColor("#999999"))
                painter.drawText(rect.x(), text_top + 14, rect.width(), 14,
                                 int(Qt.AlignHCenter | Qt.AlignTop),
                                 f"{self._shares[i] * 100:.0f}%")
        painter.restore()


def _heading(title: str, desc: str, parent) -> QLabel:
    """A form section heading: bold title, dim description on the line."""
    lab = QLabel(parent)
    lab.setTextFormat(Qt.RichText)
    lab.setWordWrap(True)
    lab.setText(
        f'<span style="color:#cccccc; font-size:12px; font-weight:bold;">'
        f'{title}</span>&nbsp;&nbsp;'
        f'<span style="color:#888888; font-size:10px;">{desc}</span>')
    return lab


class AnalysisControlsWidget(QWidget):
    """EDA Workbench — dockable widget for feature analysis.

    Layout: the "what you see" line; a vertical
    splitter with the pinned part (plot, legend, selection card, busy
    strip) above and the tabs Features · Map · Groups below.  At a wide dock
    (> 1020 px) the one feature panel moves into a 420 px left column and
    the Features tab is hidden; back below 980 px.
    """

    # Signals emitted to the controller
    compute_pca_requested = Signal(dict)
    compute_clusters_requested = Signal(dict)
    compute_umap_requested = Signal(dict)
    compute_tsne_requested = Signal(dict)
    labels_gated = Signal(list)         # label IDs from histogram range gate
    color_by_feature = Signal(str)      # feature name for label colormap sync
    mark_good_requested = Signal()
    mark_bad_requested = Signal()
    #: New: the controller connects these in B3b.
    unmark_requested = Signal()
    undo_mark_requested = Signal()
    #: New: Stop on the busy strip; the controller connects it
    #: in B3b.
    stop_map_requested = Signal()
    sync_viewer_toggled = Signal(bool)  # sync viewer colors checkbox

    # Column indices
    COL_CHECK = 0
    COL_NAME = 1
    COL_SPARK = 2
    COL_SKEW = 3
    COL_NORM = 4       # the "Bell" column since, kept by index
    COL_XFRM = 5

    #: The labels whose text is a SENTENCE of ``_workbench_text`` (or a
    #: number): ``test_workbench_sentences.py`` forbids a literal or an
    #: f-string reaching any of them from this module.
    STATUS_LABELS = (
        "status_label", "groups_result_label", "qc_selection_label",
        "tier_summary_label", "sel_msg_label", "busy_label", "see_label",
        "selection_label", "_detail_norm_label", "_pca_done_label",
        "_pca_stale_label", "_maps_old_label", "_map_error_label",
        "_seed_off_label", "_gate_status", "sel_chips_label",
        "_pca_excluded_label", "_map_why_label", "_groups_why_label",
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(T.WB_WINDOW_TITLE)
        self._feature_names: list = []
        self._selected_features: set = set()
        self._category_rows: dict = {}  # category -> list of table row indices
        self._category_header_rows: dict = {}  # category -> header row index
        self._profiles: Dict[str, dict] = {}  # feature -> profile dict
        self._bell: Dict[str, tuple] = {}  # feature -> (skew, excess kurtosis)
        self._raw_data: Dict[str, np.ndarray] = {}  # feature -> raw values
        self._transforms: Dict[str, str] = {}  # feature -> current method
        self._source_df = None  # for correlation computation
        self._wide = False
        self._pca_result = None
        self._embeddings_stale = False
        self._status_base_text = ""
        self._spark_hist: Dict[str, dict] = {}  # feature -> rescaled histogram
        self._collapsed: set = set()  # folded categories (W1-3)
        self._pca_excluded_n = 0  # regress-out features left out of the PCA
        # Divider state: reset once, then keep the plot's height.
        self._divider_was_reset = False
        self._divider_user_moved = False
        self._last_chrome: Optional[int] = None
        self._pending_width = 0
        # B3b: what the mark buttons and keys may do.
        self._has_selection = False
        self._can_undo = False
        self._locked_job: Optional[str] = None
        self._step_inactive = False       # K7: shown outside Step 4
        self._map_running: Optional[str] = None
        self._build_ui()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_ui(self):
        from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
            QCPlotWidget,
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.setStyleSheet(f"""
            QWidget {{ background: {_BG}; color: {_TEXT}; }}
            QLabel {{ color: {_TEXT}; font-size: 11px; background: transparent; }}
            QToolTip {{ background: #2b2b2b; color: #dddddd;
                        border: 1px solid rgba(255,255,255,0.15); }}
        """ + CHECK_DISABLED_STYLE)   # a greyed tick dims too
        self._styles = self._make_styles()

        # ── 1. the "what you see" line + Colours in viewer ──
        see_row = QWidget(self)
        see_row.setObjectName("wbSeeRow")
        see_row.setStyleSheet(
            "#wbSeeRow { border-bottom: 1px solid #555555; }")
        see_lay = QHBoxLayout(see_row)
        see_lay.setContentsMargins(10, 4, 10, 6)
        see_lay.setSpacing(10)
        self.see_label = QLabel(see_row)
        self.see_label.setTextFormat(Qt.RichText)
        self.see_label.setWordWrap(True)
        self.see_label.setStyleSheet("color: #cccccc; font-size: 12px;")
        self.see_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        see_lay.addWidget(self.see_label, 1)
        self.sync_viewer_colors_cb = QCheckBox(T.COLOURS_IN_VIEWER, see_row)
        self.sync_viewer_colors_cb.setChecked(True)
        self.sync_viewer_colors_cb.setStyleSheet(
            "QCheckBox { color: #dddddd; font-size: 12px; }")
        self.sync_viewer_colors_cb.toggled.connect(self._on_sync_viewer_toggled)
        see_lay.addWidget(self.sync_viewer_colors_cb)
        root.addWidget(see_row)

        # ── the main area: [left column (wide only)] | splitter ──
        main = QWidget(self)
        main_lay = QHBoxLayout(main)
        main_lay.setContentsMargins(0, 0, 0, 0)
        main_lay.setSpacing(0)
        self._main_lay = main_lay

        self._left_scroll = QScrollArea(main)
        self._left_scroll.setObjectName("wbLeftColumn")
        self._left_scroll.setWidgetResizable(True)
        self._left_scroll.setFrameShape(QFrame.NoFrame)
        self._left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._left_scroll.setFixedWidth(WORKBENCH_LEFT_COL_W)
        self._left_scroll.setStyleSheet(
            "#wbLeftColumn { border-right: 1px solid #555555; }")
        self._left_scroll.hide()
        main_lay.addWidget(self._left_scroll)

        self._splitter = _WorkbenchSplitter(Qt.Vertical, main)
        self._splitter.setChildrenCollapsible(False)
        self._splitter.setHandleWidth(WORKBENCH_HANDLE_PX)
        self._splitter.reset_requested.connect(self.reset_divider)
        main_lay.addWidget(self._splitter, 1)
        root.addWidget(main, 1)

        # ── 2. the pinned part (top pane) ──
        pinned = QWidget(self._splitter)
        pinned.setObjectName("wbPinned")
        pin_lay = QVBoxLayout(pinned)
        pin_lay.setContentsMargins(8, 6, 8, 6)
        pin_lay.setSpacing(6)
        self._pinned = pinned

        self.qc_plot_widget = QCPlotWidget(pinned, layout="workbench",
                                           texts=WORKBENCH_PLOT_TEXTS)
        pin_lay.addWidget(self.qc_plot_widget, 1)
        self.qc_plot_widget.stats_changed.connect(self.refresh_see_line)
        self.qc_plot_widget._color_combo.currentTextChanged.connect(
            self._on_plot_colour_changed)
        self.qc_plot_widget.plot_type_combo.currentIndexChanged.connect(
            self._on_plot_type_changed)

        self._build_selection_card(pinned)
        pin_lay.addWidget(self._sel_card)
        self._build_busy_strip(pinned)
        pin_lay.addWidget(self.busy_strip)
        # When the plot is at its 560 px clamp, the spare room goes here,
        # not into the card's labels.
        pin_lay.addStretch(0)

        # ── 4. the tabs (bottom pane) ──
        self._tabs = QTabWidget(self._splitter)
        self._tabs.setStyleSheet(self._styles["tabs"])
        self._build_feature_panel()
        self._features_scroll = self._tab_scroll()
        self._features_scroll.setWidget(self._feature_panel)
        self._tabs.addTab(self._features_scroll, T.TAB_FEATURES)
        self._map_scroll = self._tab_scroll()
        self._map_scroll.setWidget(self._build_map_tab())
        self._tabs.addTab(self._map_scroll, T.TAB_MAP)
        self._groups_scroll = self._tab_scroll()
        self._groups_scroll.setWidget(self._build_groups_tab())
        self._tabs.addTab(self._groups_scroll, T.TAB_GROUPS)

        self._splitter.addWidget(pinned)
        self._splitter.addWidget(self._tabs)
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        for i in range(self._splitter.count()):
            handle = self._splitter.handle(i)
            if handle is not None:
                handle.setToolTip(H["Divider"])

        # The PC pickers ride the plot's axes row.
        self.qc_plot_widget.add_axes_widget(self._pc_axes_field)

        self._apply_help_lines()
        self._on_plot_type_changed()
        self.refresh_see_line()
        self._on_cluster_method_changed()
        self._splitter.splitterMoved.connect(self._on_splitter_moved)
        # One coalescing timer keeps the plot's height after the chrome
        # changed; another applies a width change one loop turn later.
        self._keep_timer = QTimer(self)
        self._keep_timer.setSingleShot(True)
        self._keep_timer.setInterval(0)
        self._keep_timer.timeout.connect(self.keep_plot_height)
        self._width_timer = QTimer(self)
        self._width_timer.setSingleShot(True)
        self._width_timer.setInterval(0)
        self._width_timer.timeout.connect(self._apply_pending_width)
        # W2 (owner live test 2026-10-05): the top pane's minimum is the
        # pinned chrome AT ITS WIDTH plus the plot's floor, recomputed one
        # loop turn after a width change (and on every chrome change, in
        # keep_plot_height).  The non-collapsible splitter trusts a child's
        # minimum, and the FlowLayouts' own minimum is one row: with the
        # axes row wrapped to two, the divider's top stop squeezed the plot
        # and drew the legend over its x-axis.
        self._min_timer = QTimer(self)
        self._min_timer.setSingleShot(True)
        self._min_timer.setInterval(0)
        self._min_timer.timeout.connect(self._update_pinned_minimum)
        self._pinned_w = -1
        self._pinned.installEventFilter(self)
        self._install_keys()
        self.qc_plot_widget._select_btn.toggled.connect(
            self._on_lasso_toggled)
        self._refresh_action_enables()

    def _make_styles(self) -> dict:
        combo = f"""
            QComboBox {{
                background: rgba(255, 255, 255, 0.04); color: #dddddd;
                border: 1px solid rgba(255, 255, 255, 0.10);
                border-radius: 3px; padding: 2px 6px; font-size: 11px;
                min-height: 20px;
            }}
            QComboBox:disabled {{ color: #666666; }}
            QComboBox QAbstractItemView {{
                background: #2b2d35; color: #dddddd;
                selection-background-color: rgba(106, 191, 105, 0.25);
            }}
        """
        spin = """
            QSpinBox, QDoubleSpinBox {
                background: rgba(255, 255, 255, 0.04); color: #dddddd;
                border: 1px solid rgba(255, 255, 255, 0.10);
                border-radius: 3px; padding: 2px 4px; font-size: 11px;
                min-height: 20px;
            }
            QSpinBox:disabled, QDoubleSpinBox:disabled {
                color: #666666; background: rgba(255, 255, 255, 0.02);
            }
        """
        small_btn = """
            QPushButton {
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.10);
                border-radius: 4px; color: #cccccc;
                padding: 4px 10px; font-size: 10px;
            }
            QPushButton:hover { background: rgba(255, 255, 255, 0.10); }
            QPushButton:checked {
                background: rgba(106, 191, 105, 0.25);
                border-color: rgba(106, 191, 105, 0.50); color: #ffffff;
            }
            QPushButton:disabled { color: #666666; }
        """
        primary = """
            QPushButton {
                background: rgba(106, 191, 105, 0.25);
                border: 1px solid rgba(106, 191, 105, 0.40);
                border-radius: 4px; color: #eeeeee; font-weight: bold;
                padding: 6px 12px; font-size: 11px;
            }
            QPushButton:hover { background: rgba(106, 191, 105, 0.35); }
            QPushButton:disabled {
                background: rgba(106, 191, 105, 0.08);
                border: 1px solid rgba(106, 191, 105, 0.15); color: #777777;
            }
        """
        tabs = f"""
            QTabWidget::pane {{
                border: none; border-top: 1px solid rgba(255, 255, 255, 0.08);
                background: {_BG_CONTENT};
            }}
            QTabBar::tab {{
                background: rgba(255, 255, 255, 0.03); color: #aaaaaa;
                border: 1px solid rgba(255, 255, 255, 0.08); border-bottom: none;
                border-top-left-radius: 4px; border-top-right-radius: 4px;
                padding: 5px 14px; font-size: 12px; margin-right: 2px;
            }}
            QTabBar::tab:selected {{
                background: rgba(255, 255, 255, 0.06); color: #eeeeee;
                border-color: rgba(255, 255, 255, 0.14);
                border-bottom: 2px solid {_ACCENT};
            }}
        """
        return {"combo": combo, "spin": spin, "small_btn": small_btn,
                "primary": primary, "tabs": tabs}

    def _tab_scroll(self) -> QScrollArea:
        sc = QScrollArea(self._tabs)
        sc.setWidgetResizable(True)
        sc.setFrameShape(QFrame.NoFrame)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        return sc

    def _combo(self, parent, items=(), data=None) -> QComboBox:
        c = QComboBox(parent)
        for i, text in enumerate(items):
            c.addItem(text, data[i] if data else None)
        c.setStyleSheet(self._styles["combo"])
        c.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        c.setMinimumContentsLength(8)
        return c

    def _spin(self, cls, parent, lo, hi, value, *, step=None, decimals=None):
        s = cls(parent)
        s.setRange(lo, hi)
        if decimals is not None:
            s.setDecimals(decimals)
        if step is not None:
            s.setSingleStep(step)
        s.setValue(value)
        s.setStyleSheet(self._styles["spin"])
        s.setMinimumWidth(self._readable_spin_width(s))
        return s

    @staticmethod
    def _readable_spin_width(box) -> int:
        """A spin box's readable floor: an explicit minimum width that fits its
        widest value, so a shrinking form never squeezes the number out of sight."""
        fm = box.fontMetrics()
        texts = [str(box.maximum()), str(box.minimum())]
        widest = max(fm.horizontalAdvance(t) for t in texts)
        return widest + 44

    def _form(self, parent) -> QGridLayout:
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(5)
        grid.setColumnStretch(0, 44)
        grid.setColumnStretch(1, 56)
        return grid

    def _form_label(self, text, parent, *, extra: str = "") -> QLabel:
        lab = QLabel(parent)
        lab.setTextFormat(Qt.RichText)
        lab.setWordWrap(True)
        if extra:
            lab.setText(
                f'<span style="color:#dddddd; font-size:12px;">{text}</span>'
                f'&nbsp;<span style="color:#777777; font-size:10px; '
                f'font-family:Consolas, monospace;">{extra}</span>')
        else:
            lab.setText(f'<span style="color:#dddddd; font-size:12px;">'
                        f'{text}</span>')
        lab.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        return lab

    # ── the selection card (its behaviour is B3b's) ──

    def _build_selection_card(self, parent):
        card = QFrame(parent)
        card.setObjectName("wbSelCard")
        self._sel_card = card
        self._set_card_style(False)
        card.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(5)

        l1 = QHBoxLayout()
        l1.setSpacing(10)
        self.qc_selection_label = QLabel(T.SEL_NONE, card)
        self.qc_selection_label.setWordWrap(True)
        self.qc_selection_label.setStyleSheet(
            "color: #aaaaaa; font-size: 11px;")
        self.qc_selection_label.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Preferred)
        l1.addWidget(self.qc_selection_label, 1)
        self.sel_mix_bar = _TierMixBar(card)
        self.sel_mix_bar.hide()
        l1.addWidget(self.sel_mix_bar, 1)
        self.clear_selection_btn = self.qc_plot_widget.take_clear_button()
        l1.addWidget(self.clear_selection_btn)
        lay.addLayout(l1)

        self.sel_chips_label = QLabel("", card)
        self.sel_chips_label.setTextFormat(Qt.RichText)
        self.sel_chips_label.setWordWrap(True)
        self.sel_chips_label.setStyleSheet("color: #bbbbbb; font-size: 11px;")
        self.sel_chips_label.hide()
        lay.addWidget(self.sel_chips_label)

        acts = QHBoxLayout()
        acts.setSpacing(6)
        self.mark_good_btn = _key_button("Mark good", "G", card, "#e6f7ec")
        self.mark_good_btn.setStyleSheet(self._tier_btn_style(80, 200, 120,
                                                              "#e6f7ec"))
        self.mark_good_btn.clicked.connect(self.mark_good_requested)
        self.mark_bad_btn = _key_button("Mark bad", "B", card, "#fbe3d4")
        self.mark_bad_btn.setStyleSheet(self._tier_btn_style(255, 120, 40,
                                                             "#fbe3d4"))
        self.mark_bad_btn.clicked.connect(self.mark_bad_requested)
        self.unmark_btn = QPushButton("Unmark", card)
        self.unmark_btn.setStyleSheet(self._styles["small_btn"])
        self.unmark_btn.clicked.connect(self.unmark_requested)
        self.undo_mark_btn = QPushButton("↶ Undo", card)
        self.undo_mark_btn.setStyleSheet(self._styles["small_btn"])
        self.undo_mark_btn.clicked.connect(self.undo_mark_requested)
        for b in (self.mark_good_btn, self.mark_bad_btn, self.unmark_btn,
                  self.undo_mark_btn):
            acts.addWidget(b)
        acts.addStretch(1)
        lay.addLayout(acts)

        self.sel_msg_label = QLabel("", card)
        self.sel_msg_label.setWordWrap(True)
        self.sel_msg_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        self.sel_msg_label.hide()
        lay.addWidget(self.sel_msg_label)

        self.tier_summary_label = _AutoHideLabel("", card)
        self.tier_summary_label.setWordWrap(True)
        self.tier_summary_label.setStyleSheet("color: #999999; font-size: 11px;")
        self.tier_summary_label.hide()
        self.tier_summary_label.text_set.connect(self._schedule_keep_plot)
        lay.addWidget(self.tier_summary_label)

    @staticmethod
    def _tier_btn_style(r, g, b, fg) -> str:
        return f"""
            QPushButton {{
                background: rgba({r}, {g}, {b}, 0.20);
                border: 1px solid rgba({r}, {g}, {b}, 0.45);
                border-radius: 4px; color: {fg}; font-weight: bold;
                padding: 5px 12px; font-size: 11px;
            }}
            QPushButton:hover {{ background: rgba({r}, {g}, {b}, 0.30); }}
            QPushButton:disabled {{ color: #777777;
                background: rgba({r}, {g}, {b}, 0.06); }}
        """

    def _set_card_style(self, has_selection: bool):
        edge = _ACCENT if has_selection else "rgba(255, 255, 255, 0.15)"
        self._sel_card.setStyleSheet(f"""
            #wbSelCard {{
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.10);
                border-left: 3px solid {edge}; border-radius: 4px;
            }}
        """)

    def show_selection(self, *, count_text: str, mix: list, chips_html: str):
        """View-only (B3b decides what to show): line 1, the tier-mix bar
        and the chips of the selection card.  ``mix`` = ``[(colour, n)]``;
        an empty ``mix`` means nothing is selected."""
        has = bool(mix)
        self.qc_selection_label.setText(count_text)
        self.qc_selection_label.setStyleSheet(
            "color: #eeeeee; font-size: 13px; font-weight: bold;" if has
            else "color: #aaaaaa; font-size: 11px;")
        self.sel_mix_bar.set_counts(mix)
        self.sel_mix_bar.setVisible(has)
        self.sel_chips_label.setText(chips_html)
        self.sel_chips_label.setVisible(has and bool(chips_html))
        self._set_card_style(has)
        self._schedule_keep_plot()

    def show_card_message(self, text: str, kind: str = ""):
        """The selection card's message line (``kind``: "", "ok", "warn").
        While a Step 4 job locks the card (R1-2) the line keeps WB_LOCKED:
        a lasso, a pick or a gate never replaces the reason the Mark
        buttons are grey."""
        locked = getattr(self, "_locked_job", None)
        if getattr(self, "_step_inactive", False):
            text, kind = T.WB_STEP_INACTIVE, "warn"
        elif locked:
            text, kind = T.WB_LOCKED.format(job=locked), "warn"
        colour = {"ok": _ACCENT, "warn": _AMBER}.get(kind, "#aaaaaa")
        self.sel_msg_label.setStyleSheet(f"color: {colour}; font-size: 11px;")
        self.sel_msg_label.setText(text)
        self.sel_msg_label.setVisible(bool(text))
        self._schedule_keep_plot()

    # ── what the mark buttons, the keys and the maps may do (B3b) ──

    def set_mark_state(self, *, has_selection: bool, can_undo: bool):
        """The controller's word on the card's buttons: Mark
        good / Mark bad / Unmark need a selection, Undo a history."""
        self._has_selection = bool(has_selection)
        self._can_undo = bool(can_undo)
        self._refresh_action_enables()

    def set_workbench_lock(self, job: Optional[str]):
        """While a Step 4 job runs (``job`` = its word) marking
        and maps wait: the buttons are grey and WB_LOCKED is on the card's
        message line.  ``None`` lifts it.  B2's lock drives this."""
        was = self._locked_job
        self._locked_job = job or None
        if self._locked_job:
            self.show_card_message(T.WB_LOCKED.format(job=self._locked_job),
                                   "warn")
        elif was:
            self.show_card_message("")
        self._refresh_action_enables()

    def set_step_inactive(self, inactive: bool):
        """K7: outside Step 4 the Workbench's Mark / Unmark / Undo /
        Clear are grey and WB_STEP_INACTIVE is on the card's line."""
        was = self._step_inactive
        self._step_inactive = bool(inactive)
        if self._step_inactive:
            self.show_card_message(T.WB_STEP_INACTIVE, "warn")
        elif was:
            try:
                sel = self.qc_plot_widget.selected_label_ids()
            except Exception:      # noqa: BLE001
                sel = None
            self.clear_selection_btn.setEnabled(bool(sel))
            if self._locked_job:
                self.show_card_message(
                    T.WB_LOCKED.format(job=self._locked_job), "warn")
            else:
                self.show_card_message("")
        self._refresh_action_enables()

    def set_map_running(self, map_word: Optional[str]):
        """A map is computing (``map_word``) or none is: only Compute and
        Cluster go grey, each with BUSY_ONE beside it.  Marks
        stay live (I-17)."""
        self._map_running = map_word or None
        why = (T.BUSY_ONE.format(map=self._map_running)
               if self._map_running else "")
        for label in (self._map_why_label, self._groups_why_label):
            label.setText(why)
            label.setVisible(bool(why))
        self._refresh_action_enables()

    # ── the Map and Groups result lines ──

    def show_map_result(self, text: str):
        """The Map tab's result line (MAP_DONE, STOPPED)."""
        self.status_label.setText(text)

    def show_map_error(self, text: str):
        """The Map tab's amber line (PCA_TOO_FEW, MAP_FAILED)."""
        self._map_error_label.setText(text)
        self._map_error_label.setVisible(bool(text))

    def show_groups_result(self, text: str, kind: str = "ok"):
        """The Groups tab's result line: CLUSTER_DONE / CLUSTER_DONE_HDB
        in green, CLUSTER_NEEDS_UMAP / STOPPED / a failure in amber."""
        colour = _AMBER if kind == "warn" else _ACCENT
        self.groups_result_label.setStyleSheet(
            f"color: {colour}; font-size: 11px;")
        self.groups_result_label.setText(text)
        self.groups_result_label.setVisible(bool(text))

    def set_maps_old(self, old: bool):
        """MAPS_OLD (amber): the UMAP / t-SNE predate the current PCA."""
        self._maps_old_label.setVisible(bool(old))

    def set_cluster_seed(self, value):
        """The saved seed back into the Seed box (EDA ``ui_state``)."""
        try:
            self._cluster_seed_spin.setValue(int(value))
        except (TypeError, ValueError):
            pass

    def refresh_groups_legend(self):
        """The clickable Groups legend: one swatch per group, the plot's
        own colours, each click selecting that group."""
        pw = self.qc_plot_widget
        df = getattr(pw, "_df", None)
        if df is None or "cluster_id" not in df.columns:
            self._groups_legend_label.setText("")
            self.groups_legend.hide()
            return
        _col, cmap = pw._get_categorical_info("Cluster")
        counts = df["cluster_id"].value_counts(dropna=True)
        items = []
        for cid, (r, g, b) in (cmap or {}).items():
            word = (T.NOISE_WORD if cid == -1
                    else T.GROUP_WORD.format(n=int(cid) + 1))
            n = int(counts.get(cid, 0))
            items.append(T.GROUP_SWATCH_HTML.format(
                cid=cid, r=int(r), g=int(g), b=int(b), word=word,
                n=f"{n:,}"))
        self._groups_legend_label.setText(T.CHIP_SEP.join(items))
        self.groups_legend.setVisible(bool(items))

    def _on_group_link(self, link: str):
        if not link.startswith("group://"):
            return
        self.qc_plot_widget._toggle_category_selection(
            "cluster_id", link[len("group://"):])

    def clear_map_results(self):
        """Spec: a load, a Reset, a Remove or a Measure drops the
        map results -- the loadings table, the result lines, MAPS_OLD /
        PCA_STALE, the Groups legend.  The seed box keeps its value."""
        self._pca_result = None
        self._embeddings_stale = False
        self.loadings_table.clear()
        self.loadings_table.setRowCount(0)
        self.loadings_table.setColumnCount(0)
        self._pca_results_panel.setVisible(False)
        self._pca_done_label.setText("")
        self._pca_excluded_label.setText("")
        self._pca_excluded_label.hide()
        for label in (self._pca_stale_label, self._maps_old_label):
            label.hide()
        self.show_map_error("")
        self.show_map_result("")
        self.show_groups_result("")
        self._groups_legend_label.setText("")
        self.groups_legend.hide()
        for combo in (self._umap_input_combo, self._tsne_input_combo,
                      self._cluster_input_combo):
            combo.blockSignals(True)
            combo.setItemText(0, "PCA components")
            combo.setItemData(0, "PCA")
            combo.blockSignals(False)

    def _refresh_action_enables(self):
        locked = bool(self._locked_job) or self._step_inactive
        can_mark = self._has_selection and not locked
        if self._step_inactive:
            # K7: grey outside Step 4 (the plot's own selection rule
            # enables it otherwise; see set_step_inactive)
            self.clear_selection_btn.setEnabled(False)
        _key_button_enable(self.mark_good_btn, can_mark)
        _key_button_enable(self.mark_bad_btn, can_mark)
        self.unmark_btn.setEnabled(can_mark)
        self.undo_mark_btn.setEnabled(self._can_undo and not locked)
        can_map = not locked and not self._map_running
        for btn in (self.compute_pca_btn, self.compute_umap_btn,
                    self.compute_tsne_btn, self.compute_clusters_btn):
            btn.setEnabled(can_map)

    def _install_keys(self):
        """G / B / Ctrl+Z / Esc on this dock only.  A
        QShortcut fires whatever the buttons show, so every slot checks
        its button's enable state and does nothing while it is grey."""
        self._shortcuts = []
        for seq, slot in (("G", self._on_key_good), ("B", self._on_key_bad),
                          ("Ctrl+Z", self._on_key_undo),
                          ("Esc", self._on_key_escape)):
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(slot)
            self._shortcuts.append(sc)

    @staticmethod
    def _press_if_enabled(btn) -> bool:
        if btn is None or not btn.isEnabled():
            return False
        btn.click()
        return True

    def _key_refused(self, reason: str):
        """A key whose button is grey (audit X1-4): under the lock it does
        nothing (WB_LOCKED is on screen); otherwise the reason goes to the
        message line and nothing else happens."""
        if self._locked_job or self._step_inactive:
            return
        self.show_card_message(reason, "warn")

    def _on_key_good(self):
        if not self._press_if_enabled(self.mark_good_btn):
            self._key_refused(T.MARK_NEEDS_SEL)

    def _on_key_bad(self):
        if not self._press_if_enabled(self.mark_bad_btn):
            self._key_refused(T.MARK_NEEDS_SEL)

    def _on_key_undo(self):
        if not self._press_if_enabled(self.undo_mark_btn):
            self._key_refused(T.UNDO_EMPTY)

    def _on_key_escape(self):
        btn = self.clear_selection_btn
        if self._step_inactive:
            return
        if btn is not None and btn.isEnabled() and btn.isVisibleTo(self):
            btn.click()

    def _on_lasso_toggled(self, checked: bool):
        """LASSO_VIOLIN on the card when the lasso is turned on over a
        violin."""
        if (checked and self.qc_plot_widget.plot_type_combo.currentText()
                == "Violin"):
            self.show_card_message(T.LASSO_VIOLIN, "warn")

    # ── the busy strip (its behaviour is B3b's) ──

    def _build_busy_strip(self, parent):
        strip = QFrame(parent)
        strip.setObjectName("wbBusy")
        strip.setStyleSheet("""
            #wbBusy {
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.10); border-radius: 4px;
            }
        """)
        strip.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        lay = QVBoxLayout(strip)
        lay.setContentsMargins(8, 5, 8, 5)
        lay.setSpacing(4)
        r = QHBoxLayout()
        r.setSpacing(8)
        self.busy_label = QLabel("", strip)
        self.busy_label.setWordWrap(True)
        self.busy_label.setStyleSheet("color: #dddddd; font-size: 11px;")
        self.busy_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        r.addWidget(self.busy_label, 1)
        self.busy_stop_btn = QPushButton("■ Stop", strip)
        self.busy_stop_btn.setStyleSheet(self._styles["small_btn"])
        self.busy_stop_btn.clicked.connect(self.stop_map_requested)
        r.addWidget(self.busy_stop_btn)
        lay.addLayout(r)
        self.busy_bar = QProgressBar(strip)
        self.busy_bar.setFixedHeight(14)
        self.busy_bar.setTextVisible(False)
        self.busy_bar.setStyleSheet(f"""
            QProgressBar {{ background: rgba(255, 255, 255, 0.06);
                            border: none; border-radius: 3px; }}
            QProgressBar::chunk {{ background: {_ACCENT}; border-radius: 3px; }}
        """)
        lay.addWidget(self.busy_bar)
        self.busy_strip = strip
        strip.hide()

    def set_busy_strip(self, text: Optional[str], *, maximum: int = 0,
                       value: int = 0):
        """View-only: show the busy strip with ``text`` (``None`` hides it).
        ``maximum=0`` is a marquee bar."""
        timer = getattr(self, "_flash_timer", None)
        if timer is not None:
            timer.stop()
        if text is None:
            self.busy_strip.hide()
            self.busy_label.setText("")
        else:
            self.busy_label.setText(text)
            self.busy_bar.setRange(0, maximum)
            if maximum:
                self.busy_bar.setValue(value)
            self.busy_bar.show()
            self.busy_stop_btn.show()
            self.busy_strip.show()
        self._schedule_keep_plot()

    #: How long STOPPED stays on the strip (audit X1-3).
    FLASH_MS = 3000

    def flash_busy_strip(self, text: str, ms: Optional[int] = None):
        """Keep the strip up with ``text`` (STOPPED) for a moment, no bar,
        no Stop, then hide it.  A timer, never a pump; a new
        ``set_busy_strip`` cancels the hide."""
        self.set_busy_strip(text)
        self.busy_bar.hide()
        self.busy_stop_btn.hide()
        timer = getattr(self, "_flash_timer", None)
        if timer is None:
            # Parented, single-shot, stopped by any set_busy_strip (LOW-1).
            timer = self._flash_timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._end_flash)
        timer.start(int(self.FLASH_MS if ms is None else ms))

    def _end_flash(self):
        self.set_busy_strip(None)

    # ── the feature panel (one instance: Features tab, or the left column) ──

    def _build_feature_panel(self):
        panel = QWidget()
        panel.setObjectName("wbFeaturePanel")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)
        self._feature_panel = panel

        title = QLabel(T.FEATURES_TITLE, panel)
        title.setStyleSheet(
            "color: #cccccc; font-size: 12px; font-weight: bold;")
        lay.addWidget(title)
        desc = QLabel(T.FEATURES_DESC, panel)
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {_TEXT_DIM}; font-size: 10px;")
        lay.addWidget(desc)

        self.feature_table = QTableWidget(panel)
        self.feature_table.setColumnCount(6)
        self.feature_table.setHorizontalHeaderLabels(list(T.FEATURE_COLUMNS))
        self.feature_table.horizontalHeaderItem(self.COL_NAME).setTextAlignment(
            Qt.AlignLeft | Qt.AlignVCenter)
        hdr = self.feature_table.horizontalHeader()
        hdr.setMinimumSectionSize(18)
        hdr.setSectionResizeMode(self.COL_CHECK, QHeaderView.Fixed)
        hdr.resizeSection(self.COL_CHECK, 26)
        hdr.setSectionResizeMode(self.COL_NAME, QHeaderView.Stretch)
        hdr.setSectionResizeMode(self.COL_SPARK, QHeaderView.Fixed)
        hdr.resizeSection(self.COL_SPARK, _SPARK_W + 8)
        hdr.setSectionResizeMode(self.COL_SKEW, QHeaderView.Fixed)
        hdr.resizeSection(self.COL_SKEW, 54)
        hdr.setSectionResizeMode(self.COL_NORM, QHeaderView.Fixed)
        hdr.resizeSection(self.COL_NORM, 34)
        hdr.setSectionResizeMode(self.COL_XFRM, QHeaderView.Fixed)
        hdr.resizeSection(self.COL_XFRM, 84)
        self.feature_table.verticalHeader().setVisible(False)
        self.feature_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.feature_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.feature_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.feature_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.feature_table.setAlternatingRowColors(False)
        self.feature_table.setWordWrap(True)
        self.feature_table.setMinimumHeight(160)
        self.feature_table.setStyleSheet(f"""
            QTableWidget {{
                background: rgba(255, 255, 255, 0.02);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 3px;
                gridline-color: rgba(255, 255, 255, 0.04);
                font-size: 11px; color: #cccccc;
            }}
            QTableWidget::item {{
                padding: 1px 4px;
                border-bottom: 1px solid rgba(255, 255, 255, 0.04);
            }}
            QTableWidget::item:selected {{
                background: {_SELECTED_BG}; color: #eeeeee;
            }}
            QHeaderView::section {{
                background: {_HEADER_BG};
                color: #aaaaaa;
                border: none;
                border-bottom: 1px solid rgba(255, 255, 255, 0.12);
                padding: 3px 4px;
                font-size: 10px;
            }}
        """)
        self.feature_table.cellClicked.connect(self._on_cell_clicked)
        self.feature_table.cellClicked.connect(self._on_row_clicked)
        self.feature_table.itemChanged.connect(self._on_checkbox_changed)
        lay.addWidget(self.feature_table, 1)

        # The foot wraps (mockup ``.ffoot``) so the panel fits the 420 px
        # left column too.
        from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
            FlowLayout,
        )

        foot = FlowLayout(h_spacing=6, v_spacing=4)
        self.select_all_btn = QPushButton("All", panel)
        self.select_all_btn.clicked.connect(self._select_all)
        self.deselect_all_btn = QPushButton("None", panel)
        self.deselect_all_btn.clicked.connect(self._deselect_all)
        self.reset_defaults_btn = QPushButton("Defaults", panel)
        self.reset_defaults_btn.clicked.connect(self._reset_defaults)
        self.auto_transform_btn = QPushButton("Auto rescale", panel)
        self.auto_transform_btn.clicked.connect(self._auto_transform_all)
        for btn in (self.select_all_btn, self.deselect_all_btn,
                    self.reset_defaults_btn, self.auto_transform_btn):
            btn.setStyleSheet(self._styles["small_btn"])
            foot.addWidget(btn)
        self.selection_label = QLabel("", panel)
        self.selection_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        foot.addWidget(self.selection_label)
        lay.addLayout(foot)

        # Feature Detail Panel
        self._detail_panel = self._build_detail_panel()
        lay.addWidget(self._detail_panel)
        self._detail_panel.setVisible(False)

    def _build_detail_panel(self) -> QWidget:
        """Build the collapsible feature detail panel under the table."""
        import pyqtgraph as pg

        panel = QFrame(self._feature_panel)
        panel.setObjectName("wbDetail")
        panel.setStyleSheet("""
            #wbDetail {
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.10); border-radius: 4px;
            }
        """)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        # Header: feature name · Colour plot by this · Gate · close
        header_row = QHBoxLayout()
        self._detail_title = QLabel("", panel)
        self._detail_title.setWordWrap(True)
        self._detail_title.setStyleSheet(
            f"font-weight: bold; font-size: 13px; color: {_ACCENT};")
        self._detail_title.setSizePolicy(QSizePolicy.Ignored,
                                         QSizePolicy.Preferred)
        header_row.addWidget(self._detail_title, 1)
        # A checkable button (mockup), same toggled / isChecked API.
        self._sync_colors_cb = QPushButton("Colour plot by this", panel)
        self._sync_colors_cb.setCheckable(True)
        self._sync_colors_cb.setStyleSheet(self._styles["small_btn"])
        self._sync_colors_cb.toggled.connect(self._on_sync_colors_toggled)
        header_row.addWidget(self._sync_colors_cb)
        self._gate_btn = QPushButton("Gate", panel)
        self._gate_btn.setCheckable(True)
        self._gate_btn.setStyleSheet(self._styles["small_btn"])
        self._gate_btn.toggled.connect(self._on_gate_toggled)
        header_row.addWidget(self._gate_btn)
        close_btn = QPushButton("✕", panel)
        close_btn.setFixedSize(22, 22)
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; border: none;
                color: {_TEXT_DIM}; font-size: 12px;
            }}
            QPushButton:hover {{ color: {_TEXT}; }}
        """)
        close_btn.clicked.connect(self._close_detail_panel)
        header_row.addWidget(close_btn)
        lay.addLayout(header_row)

        # Description
        self._detail_desc = QLabel("", panel)
        self._detail_desc.setWordWrap(True)
        self._detail_desc.setStyleSheet("color: #bbbbbb; font-size: 11px;")
        lay.addWidget(self._detail_desc)

        # Histogram (pyqtgraph)
        self._detail_plot = pg.PlotWidget(background="#1a1b22")
        self._detail_plot.setFixedHeight(110)
        self._detail_plot.getPlotItem().hideAxis("left")
        self._detail_plot.getPlotItem().getAxis("bottom").setPen(
            pg.mkPen(color="#666666", width=1)
        )
        self._detail_plot.getPlotItem().getAxis("bottom").setTextPen(
            pg.mkPen(color="#999999")
        )
        self._detail_plot.getPlotItem().getAxis("bottom").setStyle(
            tickFont=QFont("Consolas", 8)
        )
        lay.addWidget(self._detail_plot)

        # Gate region (draggable range selector)
        self._gate_region = pg.LinearRegionItem(
            values=[0, 1],
            brush=pg.mkBrush(106, 191, 105, 40),
            pen=pg.mkPen(color=_ACCENT, width=1),
            movable=True,
        )
        self._gate_region.setVisible(False)
        self._detail_plot.addItem(self._gate_region, ignoreBounds=True)
        self._gate_region.sigRegionChangeFinished.connect(
            self._on_gate_changed
        )
        # The gate's result goes to the selection card's message line
        # (GATE_LINE); this line is kept empty and hidden.
        self._gate_status = QLabel("", panel)
        self._gate_status.hide()

        # Stats line (wraps)
        self._detail_stats_label = QLabel("", panel)
        self._detail_stats_label.setWordWrap(True)
        self._detail_stats_label.setStyleSheet(
            "color: #bbbbbb; font-size: 11px; "
            "font-family: 'Consolas', monospace;"
        )
        lay.addWidget(self._detail_stats_label)

        # Bell line + the Shapiro-Wilk caveat (BELL_YES / BELL_NO / SHAPIRO)
        self._detail_norm_label = QLabel("", panel)
        self._detail_norm_label.setWordWrap(True)
        self._detail_norm_label.setStyleSheet(
            f"color: {_TEXT_DIM}; font-size: 11px;")
        lay.addWidget(self._detail_norm_label)

        # Correlations (kept empty: skipped for performance, as before)
        self._detail_corr_label = QLabel("", panel)
        self._detail_corr_label.setWordWrap(True)
        self._detail_corr_label.setStyleSheet(
            f"color: {_TEXT_DIM}; font-size: 10px;")
        self._detail_corr_label.hide()
        lay.addWidget(self._detail_corr_label)

        return panel

    # ── the Map tab ──

    def _build_map_tab(self) -> QWidget:
        from napari_epicycle.fusion_reconstruction.qc_features import (
            RESIDUAL_PCA_REGRESS_OUT,
        )

        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        # PCA results FIRST, so they never clip
        self._pca_results_panel = self._build_pca_results_panel(page)
        lay.addWidget(self._pca_results_panel)
        self._pca_results_panel.setVisible(False)

        # PCA
        lay.addWidget(_heading(T.MAP_PCA_TITLE, T.MAP_PCA_DESC, page))
        grid = self._form(page)
        self._pca_method_combo = self._combo(page, ["PCA", "Residual PCA"])
        self._pca_method_combo.currentTextChanged.connect(
            self._on_pca_method_changed)
        self._lab_method_map = self._form_label("Method", page)
        grid.addWidget(self._lab_method_map, 0, 0)
        grid.addWidget(self._pca_method_combo, 0, 1)
        self._n_components_spin = self._spin(QSpinBox, page, 2, 200, 10)
        self._lab_components = self._form_label("Components", page)
        grid.addWidget(self._lab_components, 1, 0)
        grid.addWidget(self._n_components_spin, 1, 1)
        # Items keep "Robust": the controller reads that word (S4c:2531).
        self._scaling_combo = self._combo(
            page, ["Standard (mean and spread)", "Robust (median and MAD)"])
        self._scaling_combo.currentIndexChanged.connect(
            self._mark_embeddings_stale)
        self._lab_scaling = self._form_label("Scaling", page)
        grid.addWidget(self._lab_scaling, 2, 0)
        grid.addWidget(self._scaling_combo, 2, 1)
        clip = QHBoxLayout()
        clip.setSpacing(6)
        self._winsorize_cb = QCheckBox("at ±", page)
        self._winsorize_cb.setStyleSheet(
            "QCheckBox { color: #bbbbbb; font-size: 11px; }")
        self._winsorize_cb.stateChanged.connect(self._mark_embeddings_stale)
        clip.addWidget(self._winsorize_cb)
        self._winsorize_spin = self._spin(QDoubleSpinBox, page, 1.0, 50.0, 10.0,
                                          step=1.0, decimals=1)
        self._winsorize_spin.valueChanged.connect(self._mark_embeddings_stale)
        clip.addWidget(self._winsorize_spin)
        clip.addStretch(1)
        self._lab_clip = self._form_label("Clip extremes", page,
                                          extra="winsorize")
        grid.addWidget(self._lab_clip, 3, 0)
        grid.addLayout(clip, 3, 1)
        lay.addLayout(grid)

        # Regress out (Residual PCA only)
        self._regress_out_widget = QWidget(page)
        rlay = QGridLayout(self._regress_out_widget)
        rlay.setContentsMargins(0, 0, 0, 0)
        rlay.setHorizontalSpacing(8)
        rlay.setVerticalSpacing(4)
        rlay.addWidget(self._form_label("Remove the effect of",
                                        self._regress_out_widget,
                                        extra="regress out"), 0, 0, 1, 2)
        self._regress_checkboxes: dict = {}
        for i, feat in enumerate(RESIDUAL_PCA_REGRESS_OUT):
            cb = QCheckBox(feat, self._regress_out_widget)
            cb.setChecked(True)
            cb.setStyleSheet("QCheckBox { color: #cccccc; font-size: 11px; }")
            rlay.addWidget(cb, 1 + i // 2, i % 2)
            self._regress_checkboxes[feat] = cb
        lay.addWidget(self._regress_out_widget)
        self._regress_out_widget.setVisible(False)

        # UMAP
        lay.addWidget(_heading(T.MAP_UMAP_TITLE, T.MAP_UMAP_DESC, page))
        grid = self._form(page)
        self._umap_neighbors_spin = self._spin(QSpinBox, page, 2, 200, 15)
        self._lab_umap_nb = self._form_label("Neighbours", page)
        grid.addWidget(self._lab_umap_nb, 0, 0)
        grid.addWidget(self._umap_neighbors_spin, 0, 1)
        self._umap_min_dist_spin = self._spin(QDoubleSpinBox, page, 0.0, 1.0,
                                              0.1, step=0.05, decimals=2)
        self._lab_umap_md = self._form_label("Spacing", page, extra="min_dist")
        grid.addWidget(self._lab_umap_md, 1, 0)
        grid.addWidget(self._umap_min_dist_spin, 1, 1)
        # The DATA of each item is today's text, which the controller reads
        # as input_space; the words on screen are the mockup's.
        self._umap_input_combo = self._combo(
            page, ["PCA components", "Features"], ["PCA", "Features"])
        self._lab_umap_in = self._form_label("Start from", page)
        grid.addWidget(self._lab_umap_in, 2, 0)
        grid.addWidget(self._umap_input_combo, 2, 1)
        self._umap_metric_combo = self._combo(
            page, ["euclidean", "cosine", "correlation"])
        self._lab_umap_metric = self._form_label("Distance", page)
        grid.addWidget(self._lab_umap_metric, 3, 0)
        grid.addWidget(self._umap_metric_combo, 3, 1)
        lay.addLayout(grid)

        # t-SNE
        lay.addWidget(_heading(T.MAP_TSNE_TITLE, T.MAP_TSNE_DESC, page))
        grid = self._form(page)
        self._tsne_perplexity_spin = self._spin(QSpinBox, page, 5, 100, 30)
        self._lab_tsne_perp = self._form_label("Perplexity", page)
        grid.addWidget(self._lab_tsne_perp, 0, 0)
        grid.addWidget(self._tsne_perplexity_spin, 0, 1)
        self._tsne_input_combo = self._combo(
            page, ["PCA components", "Features"], ["PCA", "Features"])
        self._lab_tsne_in = self._form_label("Start from", page)
        grid.addWidget(self._lab_tsne_in, 1, 0)
        grid.addWidget(self._tsne_input_combo, 1, 1)
        lay.addLayout(grid)

        # Compute buttons
        row = QHBoxLayout()
        row.setSpacing(6)
        self.compute_pca_btn = QPushButton(T.COMPUTE_PCA_BTN, page)
        self.compute_pca_btn.clicked.connect(self._on_compute_pca)
        self.compute_umap_btn = QPushButton("Compute UMAP", page)
        self.compute_umap_btn.clicked.connect(self._on_compute_umap)
        self.compute_tsne_btn = QPushButton("Compute t-SNE", page)
        self.compute_tsne_btn.clicked.connect(self._on_compute_tsne)
        for b in (self.compute_pca_btn, self.compute_umap_btn,
                  self.compute_tsne_btn):
            b.setStyleSheet(self._styles["primary"])
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            b.setMinimumWidth(0)
            row.addWidget(b)
        lay.addLayout(row)

        # BUSY_ONE beside the greyed buttons while a map runs (B3b).
        self._map_why_label = QLabel("", page)
        self._map_why_label.setWordWrap(True)
        self._map_why_label.setStyleSheet(
            f"color: {_AMBER}; font-size: 10px;")
        self._map_why_label.hide()
        lay.addWidget(self._map_why_label)

        # Result lines (the old status line under the tabs lives here now)
        self._map_error_label = QLabel("", page)
        self._map_error_label.setWordWrap(True)
        self._map_error_label.setStyleSheet(
            f"color: {_AMBER}; font-size: 10px;")
        self._map_error_label.hide()
        lay.addWidget(self._map_error_label)
        self._pca_stale_label = QLabel(T.PCA_STALE, page)
        self._pca_stale_label.setWordWrap(True)
        self._pca_stale_label.setStyleSheet(
            f"color: {_AMBER}; font-size: 11px;")
        self._pca_stale_label.hide()
        lay.addWidget(self._pca_stale_label)
        self._maps_old_label = QLabel(T.MAPS_OLD, page)
        self._maps_old_label.setWordWrap(True)
        self._maps_old_label.setStyleSheet(
            f"color: {_AMBER}; font-size: 11px;")
        self._maps_old_label.hide()
        lay.addWidget(self._maps_old_label)
        self.status_label = QLabel("", page)
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"color: {_ACCENT}; font-size: 11px;")
        lay.addWidget(self.status_label)
        lay.addStretch(1)

        # PC pickers for the PCA plot: they ride the plot's axes row.
        self._pc_axes_field = QWidget(self)
        f = QHBoxLayout(self._pc_axes_field)
        f.setContentsMargins(0, 0, 0, 0)
        f.setSpacing(6)
        self._pc_x_label = QLabel("X", self._pc_axes_field)
        self._pc_y_label = QLabel("Y", self._pc_axes_field)
        self._pc_x_combo = self._combo(self._pc_axes_field)
        self._pc_y_combo = self._combo(self._pc_axes_field)
        for lab in (self._pc_x_label, self._pc_y_label):
            lab.setStyleSheet("color: #bbbbbb; font-size: 12px;")
        for combo in (self._pc_x_combo, self._pc_y_combo):
            combo.setMinimumContentsLength(5)
        self._populate_pc_combos(10)
        self._pc_x_combo.setCurrentIndex(0)
        self._pc_y_combo.setCurrentIndex(1)
        self._pc_x_combo.currentIndexChanged.connect(self._on_pc_axis_changed)
        self._pc_y_combo.currentIndexChanged.connect(self._on_pc_axis_changed)
        f.addWidget(self._pc_x_label)
        f.addWidget(self._pc_x_combo)
        f.addSpacing(4)
        f.addWidget(self._pc_y_label)
        f.addWidget(self._pc_y_combo)
        return page

    def _build_pca_results_panel(self, parent) -> QWidget:
        """The PCA results card: one named table of every ticked
        feature against every component, the variation share drawn as a
        bar in each column's header."""
        panel = QFrame(parent)
        panel.setObjectName("wbPcaResults")
        panel.setStyleSheet("""
            #wbPcaResults {
                background: rgba(255, 255, 255, 0.02);
                border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 4px;
            }
        """)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)
        header = QLabel(T.PCA_RESULTS_TITLE, panel)
        header.setStyleSheet(
            f"font-weight: bold; font-size: 12px; color: {_ACCENT};")
        lay.addWidget(header)
        self._pca_done_label = QLabel("", panel)
        self._pca_done_label.setWordWrap(True)
        self._pca_done_label.setStyleSheet(
            f"color: {_TEXT_DIM}; font-size: 10px;")
        lay.addWidget(self._pca_done_label)
        self._pca_excluded_label = QLabel("", panel)
        self._pca_excluded_label.setWordWrap(True)
        self._pca_excluded_label.setStyleSheet(
            f"color: {_TEXT_DIM}; font-size: 10px;")
        self._pca_excluded_label.hide()
        lay.addWidget(self._pca_excluded_label)

        self.loadings_table = QTableWidget(panel)
        self._loadings_header = _LoadingsHeader(self.loadings_table)
        self.loadings_table.setHorizontalHeader(self._loadings_header)
        self.loadings_table.verticalHeader().setVisible(False)
        self.loadings_table.verticalHeader().setDefaultSectionSize(18)
        self.loadings_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.loadings_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.loadings_table.setFocusPolicy(Qt.NoFocus)
        self.loadings_table.setShowGrid(False)
        self.loadings_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.loadings_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.loadings_table.setSizeAdjustPolicy(
            QAbstractScrollArea.AdjustToContents)
        self.loadings_table.setStyleSheet("""
            QTableWidget { background: transparent; border: none;
                           font-size: 10px; color: #eeeeee; }
            QTableWidget::item { padding: 0px 3px; }
        """)
        lay.addWidget(self.loadings_table)

        self._pca_caption_label = QLabel(T.PCA_TABLE_CAPTION, panel)
        self._pca_caption_label.setWordWrap(True)
        self._pca_caption_label.setStyleSheet(
            f"color: {_TEXT_DIM}; font-size: 10px;")
        lay.addWidget(self._pca_caption_label)
        return panel

    # ── the Groups tab (seeds are B3b's) ──

    def _build_groups_tab(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)
        lay.addWidget(_heading(T.GROUPS_TITLE, T.GROUPS_DESC, page))

        grid = self._form(page)
        # DATA = today's algorithm keys, which the controller reads.
        self._cluster_algo_combo = self._combo(
            page, ["Leiden", "HDBSCAN", "K-means", "GMM (mixture)"],
            ["leiden", "hdbscan", "kmeans", "gmm"])
        self._cluster_algo_combo.currentIndexChanged.connect(
            self._on_cluster_method_changed)
        self._lab_method_groups = self._form_label("Method", page)
        grid.addWidget(self._lab_method_groups, 0, 0)
        grid.addWidget(self._cluster_algo_combo, 0, 1)

        self._cluster_resolution_spin = self._spin(
            QDoubleSpinBox, page, 0.1, 5.0, 1.0, step=0.1, decimals=1)
        self._lab_resolution = self._form_label("Resolution", page)
        grid.addWidget(self._lab_resolution, 1, 0)
        grid.addWidget(self._cluster_resolution_spin, 1, 1)
        self._cluster_knn_spin = self._spin(QSpinBox, page, 2, 100, 15)
        self._lab_knn = self._form_label("Neighbours", page, extra="kNN")
        grid.addWidget(self._lab_knn, 2, 0)
        grid.addWidget(self._cluster_knn_spin, 2, 1)
        # Today's one "k/min_size" spin, split into two visible rows over
        # the same value: _cluster_k_spin stays the one read.
        self._cluster_min_size_spin = self._spin(QSpinBox, page, 2, 50, 5)
        self._lab_min_size = self._form_label("Smallest group", page,
                                              extra="min_size")
        grid.addWidget(self._lab_min_size, 3, 0)
        grid.addWidget(self._cluster_min_size_spin, 3, 1)
        self._cluster_k_spin = self._spin(QSpinBox, page, 2, 50, 5)
        self._lab_k = self._form_label("Number of groups", page, extra="k")
        grid.addWidget(self._lab_k, 4, 0)
        grid.addWidget(self._cluster_k_spin, 4, 1)
        self._cluster_k_spin.valueChanged.connect(self._on_k_spin_changed)
        self._cluster_min_size_spin.valueChanged.connect(
            self._on_min_size_spin_changed)

        self._cluster_input_combo = self._combo(
            page, ["PCA components", "UMAP", "Features"],
            ["PCA", "UMAP", "Features"])
        self._lab_cluster_in = self._form_label("Start from", page)
        grid.addWidget(self._lab_cluster_in, 5, 0)
        grid.addWidget(self._cluster_input_combo, 5, 1)
        self._cluster_metric_combo = self._combo(
            page, ["euclidean", "cosine", "correlation"])
        self._lab_cluster_metric = self._form_label("Distance", page)
        grid.addWidget(self._lab_cluster_metric, 6, 0)
        grid.addWidget(self._cluster_metric_combo, 6, 1)

        seed_box = QWidget(page)
        seed_lay = QHBoxLayout(seed_box)
        seed_lay.setContentsMargins(0, 0, 0, 0)
        seed_lay.setSpacing(6)
        self._cluster_seed_spin = self._spin(QSpinBox, seed_box, 0, 99999, 42)
        seed_lay.addWidget(self._cluster_seed_spin)
        self._seed_off_label = QLabel(T.SEED_OFF, seed_box)
        self._seed_off_label.setWordWrap(True)
        self._seed_off_label.setStyleSheet(
            f"color: {_AMBER}; font-size: 10px;")
        self._seed_off_label.setSizePolicy(QSizePolicy.Ignored,
                                           QSizePolicy.Preferred)
        seed_lay.addWidget(self._seed_off_label, 1)
        self._lab_seed = self._form_label("Seed", page)
        grid.addWidget(self._lab_seed, 7, 0)
        grid.addWidget(seed_box, 7, 1)
        lay.addLayout(grid)

        self.compute_clusters_btn = QPushButton("Cluster", page)
        self.compute_clusters_btn.setStyleSheet(self._styles["primary"])
        self.compute_clusters_btn.clicked.connect(self._on_compute_clusters)
        lay.addWidget(self.compute_clusters_btn)
        self._groups_why_label = QLabel("", page)
        self._groups_why_label.setWordWrap(True)
        self._groups_why_label.setStyleSheet(
            f"color: {_AMBER}; font-size: 10px;")
        self._groups_why_label.hide()
        lay.addWidget(self._groups_why_label)

        self.groups_result_label = QLabel("", page)
        self.groups_result_label.setWordWrap(True)
        self.groups_result_label.setStyleSheet(
            f"color: {_ACCENT}; font-size: 11px;")
        self.groups_result_label.hide()
        lay.addWidget(self.groups_result_label)

        # The clickable Groups legend (filled by B3b after a Cluster).
        self.groups_legend = QFrame(page)
        self.groups_legend.setObjectName("wbGroupsLegend")
        glay = QVBoxLayout(self.groups_legend)
        glay.setContentsMargins(0, 0, 0, 0)
        glay.setSpacing(2)
        hint = QLabel(T.GROUPS_LEGEND_HINT, self.groups_legend)
        hint.setStyleSheet("color: #888888; font-size: 10px;")
        glay.addWidget(hint)
        self._groups_legend_label = QLabel("", self.groups_legend)
        self._groups_legend_label.setTextFormat(Qt.RichText)
        self._groups_legend_label.setWordWrap(True)
        self._groups_legend_label.setStyleSheet("font-size: 11px;")
        self._groups_legend_label.setTextInteractionFlags(
            Qt.LinksAccessibleByMouse)
        self._groups_legend_label.linkActivated.connect(self._on_group_link)
        glay.addWidget(self._groups_legend_label)
        self.groups_legend.hide()
        lay.addWidget(self.groups_legend)
        lay.addStretch(1)
        return page

    def _on_cluster_method_changed(self, *_):
        """Show only the knobs the chosen method uses."""
        key = self._cluster_algo_combo.currentData()
        rows = {
            "leiden": (True, True, False, False),
            "hdbscan": (False, False, True, False),
            "kmeans": (False, False, False, True),
            "gmm": (False, False, False, True),
        }.get(key, (True, True, False, False))
        for (lab, box), show in zip(
                ((self._lab_resolution, self._cluster_resolution_spin),
                 (self._lab_knn, self._cluster_knn_spin),
                 (self._lab_min_size, self._cluster_min_size_spin),
                 (self._lab_k, self._cluster_k_spin)), rows):
            lab.setVisible(show)
            box.setVisible(show)
        hdb = key == "hdbscan"
        self._cluster_seed_spin.setEnabled(not hdb)
        self._seed_off_label.setVisible(hdb)

    def _on_k_spin_changed(self, value):
        self._cluster_min_size_spin.blockSignals(True)
        self._cluster_min_size_spin.setValue(value)
        self._cluster_min_size_spin.blockSignals(False)

    def _on_min_size_spin_changed(self, value):
        self._cluster_k_spin.blockSignals(True)
        self._cluster_k_spin.setValue(value)
        self._cluster_k_spin.blockSignals(False)

    @staticmethod
    def _input_value(combo: QComboBox) -> str:
        """The input_space the controller reads: the item's data (today's
        text), falling back to the text."""
        data = combo.currentData()
        return data if isinstance(data, str) and data else combo.currentText()

    # ── help lines (the mockup's H table) ──

    def _apply_help_lines(self):
        pairs = (
            ("Colours in viewer", (self.sync_viewer_colors_cb,)),
            ("Mark good", (self.mark_good_btn,)),
            ("Mark bad", (self.mark_bad_btn,)),
            ("Unmark", (self.unmark_btn,)),
            ("Undo", (self.undo_mark_btn,)),
            ("Clear", (self.clear_selection_btn,)),
            ("Auto rescale", (self.auto_transform_btn,)),
            ("Defaults", (self.reset_defaults_btn,)),
            ("Gate", (self._gate_btn,)),
            ("Colour plot by this", (self._sync_colors_cb,)),
            ("Method (Map)", (self._lab_method_map, self._pca_method_combo)),
            ("Components", (self._lab_components, self._n_components_spin)),
            ("Scaling", (self._lab_scaling, self._scaling_combo)),
            ("Clip extremes", (self._lab_clip, self._winsorize_cb,
                               self._winsorize_spin)),
            ("Neighbours", (self._lab_umap_nb, self._umap_neighbors_spin,
                            self._lab_knn, self._cluster_knn_spin)),
            ("Spacing", (self._lab_umap_md, self._umap_min_dist_spin)),
            ("Start from", (self._lab_umap_in, self._umap_input_combo,
                            self._lab_tsne_in, self._tsne_input_combo,
                            self._lab_cluster_in, self._cluster_input_combo)),
            ("Distance", (self._lab_umap_metric, self._umap_metric_combo,
                          self._lab_cluster_metric, self._cluster_metric_combo)),
            ("Perplexity", (self._lab_tsne_perp, self._tsne_perplexity_spin)),
            ("Method (Groups)", (self._lab_method_groups,
                                 self._cluster_algo_combo)),
            ("Resolution", (self._lab_resolution,
                            self._cluster_resolution_spin)),
            ("Number of groups", (self._lab_k, self._cluster_k_spin)),
            ("Smallest group", (self._lab_min_size,
                                self._cluster_min_size_spin)),
            ("Seed", (self._lab_seed, self._cluster_seed_spin)),
        )
        for key, widgets in pairs:
            for w in widgets:
                w.setToolTip(H[key])
        # The header tooltips of the feature table (Shape · Skew · Bell ·
        # Rescale).
        for col, key in ((self.COL_SPARK, "Shape"), (self.COL_SKEW, "Skew"),
                         (self.COL_NORM, "Bell"), (self.COL_XFRM, "Rescale")):
            item = self.feature_table.horizontalHeaderItem(col)
            if item is not None:
                item.setToolTip(H[key])
        self._sync_viewer_tooltip(self.sync_viewer_colors_cb.isChecked())

    def _sync_viewer_tooltip(self, checked: bool):
        """The tick's tooltip: its help line, then SYNC_ON / SYNC_OFF."""
        self.sync_viewer_colors_cb.setToolTip("\n".join(
            (H["Colours in viewer"], T.SYNC_ON if checked else T.SYNC_OFF)))

    def _on_sync_viewer_toggled(self, checked: bool):
        self._sync_viewer_tooltip(checked)
        self.sync_viewer_toggled.emit(checked)

    # ------------------------------------------------------------------
    # The "what you see" line, the plot type and the divider
    # ------------------------------------------------------------------

    def refresh_see_line(self, *_):
        """Recount the "what you see" line (SEE_LINE / SEE_LINE_NONE)."""
        pw = self.qc_plot_widget
        df = getattr(pw, "_df", None)
        if df is None or len(df) == 0:
            self.see_label.setText("")
            return
        ghost = pw._get_ghost_mask()
        n = int(len(df) - (int(ghost.sum()) if ghost is not None else 0))
        try:
            m = len(pw.selected_label_ids())
        except Exception:  # noqa: BLE001 -- a count, never a crash
            m = 0
        colour = pw._color_combo.currentText() or "QC status"
        sentence = T.SEE_LINE if m else T.SEE_LINE_NONE
        self.see_label.setText(sentence.format(
            n=f"{n:,}", m=f"{m:,}", c=colour))

    def _on_plot_colour_changed(self, *_):
        self.refresh_see_line()
        self._schedule_keep_plot()   # Scale / Colours may wrap a new row

    def _on_plot_type_changed(self, *_):
        """The PC pickers show on the PCA plot only; the map notes follow
        the Start-from choices."""
        is_pca = self.qc_plot_widget.plot_type_combo.currentText() == "PCA"
        self._pc_axes_field.setVisible(is_pca)
        self.qc_plot_widget.axes_flow.invalidate()
        self._schedule_keep_plot()

    def plot_default_px(self) -> int:
        """The plot height a reset gives: 270 px, or 340 px split out."""
        return (WORKBENCH_PLOT_DEFAULT_WIDE_PX if self._wide
                else WORKBENCH_PLOT_DEFAULT_PX)

    def reset_divider(self):
        """Double-click on the divider: the plot back to its default height,
        the rest to the tabs."""
        total = sum(self._splitter.sizes())
        if total <= 0:
            return
        chrome = self.pinned_chrome_height()
        top = chrome + self.plot_default_px()
        top = max(self._pinned.minimumSizeHint().height(),
                  min(top, total - WORKBENCH_TABS_MIN_PX))
        self._splitter.setSizes([top, max(0, total - top)])
        self._divider_was_reset = True
        self._divider_user_moved = False
        self._last_chrome = chrome
        print(f"[WB-LAYOUT] divider reset -> sizes={self._splitter.sizes()} "
              f"(pinned chrome {chrome} px + plot {self.plot_default_px()} px)")

    def _on_splitter_moved(self, *_):
        """The user dragged the divider: a width change no longer resets it.
        The plot's new height is the one kept from now on."""
        self._divider_user_moved = True
        self._last_chrome = self.pinned_chrome_height()

    def _schedule_keep_plot(self, *_):
        timer = getattr(self, "_keep_timer", None)
        if timer is not None and not timer.isActive():
            timer.start()

    def keep_plot_height(self):
        """The plot keeps its height when a chrome line appears or goes
        (the busy strip, a wrapped axes row, the card's message): the top
        pane grows or shrinks by the change and the tabs give way, never
        below WORKBENCH_TABS_MIN_PX (V1-2)."""
        self._update_pinned_minimum()
        if not self._divider_was_reset or self._last_chrome is None:
            return
        chrome = self.pinned_chrome_height()
        delta = chrome - self._last_chrome
        if delta == 0:
            return
        top, bottom = self._splitter.sizes()
        if delta > 0:
            give = min(delta, max(0, bottom - WORKBENCH_TABS_MIN_PX))
        else:
            give = delta
        if give:
            self._splitter.setSizes([top + give, bottom - give])
        self._last_chrome = chrome
        print(f"[WB-LAYOUT] keep plot: chrome {delta:+d} px, top pane "
              f"{give:+d} px -> sizes={self._splitter.sizes()}")

    def eventFilter(self, obj, event):  # noqa: N802
        if (obj is getattr(self, "_pinned", None)
                and event.type() == QEvent.Resize
                and event.size().width() != self._pinned_w):
            self._pinned_w = event.size().width()
            timer = getattr(self, "_min_timer", None)
            if timer is not None and not timer.isActive():
                timer.start()
        return super().eventFilter(obj, event)

    def pinned_minimum_px(self, width: Optional[int] = None) -> int:
        """W2: the top pane's honest minimum at ``width`` -- the pinned
        chrome there (a wrapped axes row counts two rows) plus the plot's
        150 px floor."""
        from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
            WB_PLOT_MIN_H,
        )
        return self.pinned_chrome_height(width) + WB_PLOT_MIN_H

    def _update_pinned_minimum(self):
        """Hold the top pane to ``pinned_minimum_px`` at its own width
        (an explicit minimum outranks the layout's one-row estimate)."""
        pinned = getattr(self, "_pinned", None)
        if pinned is None or not pinned.isVisible() or pinned.width() <= 0:
            return
        need = self.pinned_minimum_px(pinned.width())
        if need != pinned.minimumHeight():
            pinned.setMinimumHeight(need)
            print(f"[WB-LAYOUT] top pane minimum {need} px at "
                  f"{pinned.width()} px")

    def pinned_chrome_height(self, width: Optional[int] = None) -> int:
        """The pinned part's height WITHOUT the plot, at ``width`` (default:
        the pane's own width): toolbar, axes row, legend, the selection
        card and the busy strip when shown, with their spacing.

        Summed from the real layouts' height-for-width, item by item, with
        the plot counted as zero -- nothing is resized to measure it (a
        measurement that resized the plot would flicker on every resize).
        ``tests/segmentation/test_workbench_layout.py`` checks the sum
        against a measurement that pins the plot instead.
        """
        pw = self.qc_plot_widget
        w = width if width is not None else self._pinned.width()
        return self._layout_height_without(self._pinned.layout(), w,
                                           pw._plot_widget)

    @staticmethod
    def _layout_holds(lay, widget) -> bool:
        for i in range(lay.count()):
            item = lay.itemAt(i)
            wd = item.widget()
            if wd is not None and (wd is widget or wd.isAncestorOf(widget)):
                return True
            sub = item.layout()
            if sub is not None and AnalysisControlsWidget._layout_holds(
                    sub, widget):
                return True
        return False

    @staticmethod
    def _layout_height_without(lay, width: int, skip) -> int:
        """``lay``'s height for ``width``, with the widget ``skip`` (and
        only it) counted as 0 px; hidden widgets and empty items drop out,
        as QBoxLayout drops them."""
        m = lay.contentsMargins()
        inner = max(1, width - m.left() - m.right())
        total = m.top() + m.bottom()
        n = 0
        for i in range(lay.count()):
            item = lay.itemAt(i)
            wd = item.widget()
            if wd is not None and wd.isHidden():
                continue
            if item.isEmpty() and item.layout() is None:
                continue
            n += 1
            if wd is skip:
                continue
            # Walk down only along the path that holds the plot; every
            # other item is asked for its own height-for-width.
            sub = None
            if wd is not None and wd.isAncestorOf(skip):
                sub = wd.layout()
            elif wd is None and item.layout() is not None:
                if AnalysisControlsWidget._layout_holds(item.layout(), skip):
                    sub = item.layout()
            if sub is not None:
                total += AnalysisControlsWidget._layout_height_without(
                    sub, inner, skip)
            elif item.hasHeightForWidth():
                total += item.heightForWidth(inner)
            else:
                total += item.sizeHint().height()
        spacing = max(0, lay.spacing())
        return int(total + spacing * max(0, n - 1))

    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        self._first_divider_reset()

    def _first_divider_reset(self):
        """The first time the dock is tall enough: the plot at its default
        height, the rest to the tabs (a QSplitter would otherwise give the
        tabs only their minimum).  A dock first shown small (tabbed and not
        yet raised) waits for a resize that makes room."""
        if self._divider_was_reset or not self.isVisible():
            return
        need = (self.pinned_chrome_height() + WORKBENCH_PLOT_DEFAULT_PX
                + WORKBENCH_TABS_MIN_PX)
        if sum(self._splitter.sizes()) >= need:
            self.reset_divider()

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._first_divider_reset()
        # The re-parent runs one loop turn later, never inside Qt's resize
        # dispatch; a burst of resizes applies only the last width.
        self._pending_width = event.size().width()
        timer = getattr(self, "_width_timer", None)
        if timer is not None and not timer.isActive():
            timer.start()

    def _apply_pending_width(self):
        w = self._pending_width or self.width()
        if not self._wide and w >= WORKBENCH_WIDE_ON_PX:
            self._set_wide(True, w)
        elif self._wide and w < WORKBENCH_WIDE_OFF_PX:
            self._set_wide(False, w)

    def _set_wide(self, wide: bool, width: int = 0):
        """Move the ONE feature panel between its tab and the left column
        (Appendix B R6): a reparent, never a rebuild."""
        if wide == self._wide:
            return
        self._wide = wide
        panel = self._feature_panel
        if wide:
            self._features_scroll.takeWidget()
            self._left_scroll.setWidget(panel)
            self._left_scroll.show()
            if self._tabs.currentIndex() == 0:
                self._tabs.setCurrentIndex(1)
            self._tabs.setTabVisible(0, False)
        else:
            self._left_scroll.takeWidget()
            self._features_scroll.setWidget(panel)
            self._left_scroll.hide()
            self._tabs.setTabVisible(0, True)
        panel.show()
        if self._divider_was_reset and not self._divider_user_moved:
            self.reset_divider()   # 270 px narrow, 340 px split out
        print(f"[WB-LAYOUT] width={width} -> "
              f"{'wide: features in the left column' if wide else 'narrow: features in their tab'}")

    def is_wide(self) -> bool:
        return self._wide

    def _on_row_clicked(self, row: int, col: int):
        """Show the detail panel for the clicked feature."""
        cb_item = self.feature_table.item(row, self.COL_CHECK)
        if not cb_item:
            return
        feat = cb_item.data(Qt.UserRole)
        if not feat or feat not in self._raw_data:
            return
        self._show_feature_detail(feat)

    def _show_feature_detail(self, feat: str):
        """Populate and show the detail panel for a feature."""
        import pyqtgraph as pg

        from napari_epicycle.segmentation.feature_stats import (
            apply_transform,
            compute_correlations,
            compute_detail_stats,
            compute_normality,
        )

        self._detail_panel.setVisible(True)

        # Clear any existing gate selection before switching features
        if self._gate_btn.isChecked():
            self.labels_gated.emit([])
        self._gate_status.setText("")
        self._current_detail_feature = feat

        # Title + description
        desc = _get_feature_description(feat)
        self._detail_title.setText(feat)
        self._detail_desc.setText(desc if desc else T.WB_NO_DESCRIPTION)

        # Get raw + transformed data
        raw = self._raw_data[feat]
        method = self._transforms.get(feat, "auto")
        transformed = apply_transform(raw, method, feature_name=feat)
        self._current_detail_values = transformed

        # Histogram
        self._detail_plot.clear()
        clean = transformed[np.isfinite(transformed)]
        if len(clean) > 1:
            n_bins = min(50, max(15, len(clean) // 5))
            counts, edges = np.histogram(clean, bins=n_bins)
            is_selected = feat in self._selected_features
            color = _ACCENT if is_selected else "#888888"
            for i in range(len(counts)):
                if counts[i] == 0:
                    continue
                bar = pg.BarGraphItem(
                    x0=[edges[i]], x1=[edges[i + 1]], height=[counts[i]],
                    brush=pg.mkBrush(color),
                    pen=pg.mkPen(None),
                )
                self._detail_plot.addItem(bar)

            # Re-add gate region and position at center 50%
            # Block signal during programmatic setRegion to avoid
            # spurious re-gate when switching features
            self._detail_plot.addItem(self._gate_region, ignoreBounds=True)
            self._gate_region.blockSignals(True)
            q25, q75 = np.percentile(clean, [25, 75])
            self._gate_region.setRegion([q25, q75])
            self._gate_region.blockSignals(False)
            self._gate_region.setVisible(self._gate_btn.isChecked())
        else:
            self._gate_region.setVisible(False)

        # Sync colors if checkbox is on
        if self._sync_colors_cb.isChecked():
            self.color_by_feature.emit(feat)

        # Reset gate status
        self._gate_status.setText("")

        # Summary stats
        stats = compute_detail_stats(transformed)
        xfrm_label = f" ({method})" if method != "raw" else ""
        self._detail_stats_label.setText(T.WB_DETAIL_STATS.format(
            xfrm=xfrm_label, **stats))

        # Bell (owner: |skew| < 0.5 and |excess kurtosis| < 1),
        # then the Shapiro-Wilk p-value with the sentence saying why the
        # Bell column does not use it (BELL_YES / BELL_NO / SHAPIRO).
        self._detail_norm_label.setText(
            self._bell_sentence(feat, stats, compute_normality(transformed)))

        # Correlations skipped for performance (100+ features × scipy = slow).
        # The distribution histogram + stats above are sufficient for EDA.
        self._detail_corr_label.setText("")

    def _bell_sentence(self, feat: str, stats: dict, norm: dict) -> str:
        """The detail panel's Bell line and, for a Shapiro-Wilk test, the
        caveat under it."""
        from napari_epicycle.segmentation.feature_stats import (
            resolve_auto_method,
        )

        skew, kurt = stats.get("skew"), stats.get("kurtosis")
        if skew is None or kurt is None or not (
                np.isfinite(skew) and np.isfinite(kurt)):
            return ""
        s, k = f"{skew:.2f}", f"{kurt:.2f}"
        if T.is_bell(skew, kurt):
            line = T.BELL_YES.format(s=s, k=k)
        else:
            raw = self._raw_data.get(feat)
            picks = (resolve_auto_method(raw, feat) if raw is not None
                     else "raw")
            line = T.BELL_NO.format(s=s, k=k, x=picks)
        if norm.get("test") == "shapiro":
            p = norm.get("p_value")
            p_txt = (f"{p:.1e}" if p is not None and p < 1e-4
                     else f"{p:.4f}" if p is not None else "—")
            line = "\n".join((line, T.SHAPIRO.format(p=p_txt)))
        return line

    # ------------------------------------------------------------------
    # Gating + color sync (Phase 4.5)
    # ------------------------------------------------------------------

    def _close_detail_panel(self):
        """Close the detail panel and clean up gate selection."""
        self._detail_panel.setVisible(False)
        if self._gate_btn.isChecked():
            self._gate_btn.setChecked(False)  # triggers _on_gate_toggled → clears
        self._gate_status.setText("")
        self.show_card_message("")  # the GATE_LINE goes with the gate
        self.labels_gated.emit([])  # clear any residual gate highlight

    def _on_gate_toggled(self, checked: bool):
        """Show/hide the gate region on the histogram."""
        self._gate_region.setVisible(checked)
        if not checked:
            self._gate_status.setText("")
            self.show_card_message("")  # the GATE_LINE goes with the gate
            self.labels_gated.emit([])  # clear gate selection

    def _on_gate_changed(self):
        """Handle gate region drag — find cells in range and emit."""
        feat = getattr(self, "_current_detail_feature", None)
        values = getattr(self, "_current_detail_values", None)
        if feat is None or values is None:
            return
        if self._source_df is None or "label" not in self._source_df.columns:
            return

        lo, hi = self._gate_region.getRegion()
        # Find labels within the gate range
        finite_mask = np.isfinite(values)
        in_range = finite_mask & (values >= lo) & (values <= hi)
        labels_in = self._source_df.loc[in_range, "label"].tolist()
        label_ids = [int(lid) for lid in labels_in]

        # GATE_LINE goes to the selection card's message line.
        self.show_card_message(T.GATE_LINE.format(
            m=f"{len(label_ids):,}", feat=feat, a=f"{lo:.2f}", b=f"{hi:.2f}"),
            "ok")
        self.labels_gated.emit(label_ids)

    def _on_sync_colors_toggled(self, checked: bool):
        """Toggle feature color sync to viewer."""
        feat = getattr(self, "_current_detail_feature", None)
        if checked and feat:
            self.color_by_feature.emit(feat)
        elif not checked:
            # Empty string signals "revert to default colors"
            self.color_by_feature.emit("")

    # ------------------------------------------------------------------
    # Feature table population
    # ------------------------------------------------------------------

    def set_feature_data(
        self,
        df: "pd.DataFrame",
        *,
        exclude_border: bool = True,
        pca_features_override: list | None = None,
    ):
        """Populate the feature table with stats from a DataFrame.

        This is the Phase 2 entry point.  Computes sparklines, skewness,
        and normality for each numeric feature and populates the table.

        Parameters
        ----------
        df : DataFrame
            Full QC features DataFrame (may contain ``is_border_cell``).
        exclude_border : bool
            If True, exclude border cells from all computations.
        pca_features_override : list, optional
            If provided, use this list for the initial PCA feature selection
            instead of the default ``PCA_FEATURES``.  Comes from the Extract
            Features dialog.
        """
        from napari_epicycle.fusion_reconstruction.qc_features import PCA_FEATURES
        from napari_epicycle.segmentation.feature_stats import profile_dataframe

        _exclude_meta = {
            "label", "is_border_cell", "qc_tier",
            "label_status", "feature_state", "patch_name",
            "stack_name", "euler_category", "cluster_id",
        }
        numeric_cols = [
            c for c in df.select_dtypes(include="number").columns
            if c not in _exclude_meta
            and not c.startswith("bbox-")
            and not c.startswith("inertia_tensor")
        ]

        self._feature_names = list(numeric_cols)
        pca_source = pca_features_override if pca_features_override else PCA_FEATURES
        self._selected_features = set(
            f for f in pca_source if f in numeric_cols
        )

        # Store raw data for transform recomputation + correlations
        work = df
        if exclude_border and "is_border_cell" in df.columns:
            work = df[~df["is_border_cell"].astype(bool)]
        self._source_df = work  # for correlation computation
        self._raw_data = {
            col: work[col].to_numpy() for col in numeric_cols
            if col in work.columns
        }

        # Initialize all transforms to "auto"
        self._transforms = {col: "auto" for col in numeric_cols}

        self._profiles = profile_dataframe(
            df, numeric_cols, exclude_border=exclude_border,
        )
        # Bell: skew and excess kurtosis of the same values the
        # profile's skew was computed from.
        self._bell = {
            col: self._skew_kurt(self._raw_data[col])
            for col in numeric_cols if col in self._raw_data
        }
        # Shape (V1-3, the mockup's applyXf): the histogram AFTER the
        # feature's rescale, as the frozen help line says.
        self._spark_hist = {
            col: self._rescaled_histogram(col)
            for col in numeric_cols if col in self._raw_data
        }
        self._rebuild_table()

    def _rescaled_histogram(self, feat: str) -> dict:
        """The Shape column's histogram: the values after the feature's
        current rescale (``auto`` resolved the way PCA resolves it)."""
        from napari_epicycle.segmentation.feature_stats import (
            apply_transform,
            compute_histogram,
        )

        raw = self._raw_data[feat]
        method = self._transforms.get(feat, "auto")
        return compute_histogram(apply_transform(raw, method,
                                                 feature_name=feat))

    @staticmethod
    def _skew_kurt(values) -> tuple:
        """``(skew, excess kurtosis)`` of the finite values, NaN when too few."""
        from scipy.stats import kurtosis as _kurtosis
        from scipy.stats import skew as _skew

        clean = np.asarray(values, dtype=np.float64)
        clean = clean[np.isfinite(clean)]
        if len(clean) < 4 or np.ptp(clean) == 0:
            return (float("nan"), float("nan"))
        return (float(_skew(clean, bias=False)),
                float(_kurtosis(clean, bias=False)))

    def set_available_features(self, feature_names: list):
        """Populate the feature table (Phase 1 fallback — no stats)."""
        from napari_epicycle.fusion_reconstruction.qc_features import PCA_FEATURES

        self._feature_names = list(feature_names)
        self._selected_features = set(
            f for f in PCA_FEATURES if f in feature_names
        )
        self._profiles = {}
        self._rebuild_table()

    def _rebuild_table(self):
        """Rebuild the table with grouped features + sparklines/stats."""
        self.feature_table.setRowCount(0)
        self.feature_table.clearSpans()
        self.feature_table.blockSignals(True)
        self._category_rows = {}
        self._category_header_rows = {}

        has_profiles = bool(self._profiles)

        # Group features by category
        groups = {}
        uncategorized = []
        channel_groups = {}  # "Membrane channel" → [features]
        for feat in self._feature_names:
            cat = _FEATURE_CATEGORIES.get(feat)
            if cat:
                groups.setdefault(cat, []).append(feat)
            else:
                # Check if it's a channel feature by matching known suffixes
                ch_prefix = _extract_channel_prefix(feat)
                if ch_prefix is not None:
                    ch_cat = T.CHANNEL_GROUP.format(
                        channel=ch_prefix.replace('_', ' ').capitalize())
                    channel_groups.setdefault(ch_cat, []).append(feat)
                else:
                    uncategorized.append(feat)

        row = 0
        for cat in _CATEGORY_ORDER:
            if cat not in groups:
                continue
            feats = groups[cat]
            self._category_rows[cat] = []

            # Category header row
            row = self._insert_header_row(row, cat, len(feats))
            row += 1

            # Feature rows
            for feat in feats:
                self._insert_feature_row(row, feat, has_profiles)
                self._category_rows.setdefault(cat, []).append(row)
                row += 1

        # Channel intensity groups (after core categories)
        for ch_cat in sorted(channel_groups.keys()):
            feats = channel_groups[ch_cat]
            self._category_rows[ch_cat] = []
            row = self._insert_header_row(row, ch_cat, len(feats))
            row += 1
            for feat in feats:
                self._insert_feature_row(row, feat, has_profiles)
                self._category_rows.setdefault(ch_cat, []).append(row)
                row += 1

        # Truly uncategorized (anything else)
        if uncategorized:
            self._category_rows["Other"] = []
            row = self._insert_header_row(row, "Other", len(uncategorized))
            row += 1
            for feat in uncategorized:
                self._insert_feature_row(row, feat, has_profiles)
                self._category_rows["Other"].append(row)
                row += 1

        self.feature_table.blockSignals(False)
        # itemChanged is connected ONCE, in _build_feature_panel (it used
        # to be connected again on every rebuild).
        self._update_selection_label()

    def _insert_header_row(self, row: int, category: str, count: int) -> int:
        """Insert a category header row ("▾ {cat}  {on} of {n} used") spanning the table.  Returns the same row index."""
        self.feature_table.insertRow(row)
        header_item = QTableWidgetItem("")
        header_item.setFlags(Qt.ItemIsEnabled)
        header_item.setData(Qt.UserRole, f"__category__{category}")
        header_font = QFont()
        header_font.setBold(True)
        header_font.setPixelSize(11)
        header_item.setFont(header_font)
        header_item.setForeground(QBrush(QColor("#cccccc")))
        header_item.setBackground(QBrush(QColor(255, 255, 255, 10)))
        # The same item sits under COL_NAME too, which the header-click
        # handler reads; the span shows the COL_CHECK one.
        self.feature_table.setItem(row, self.COL_CHECK, header_item)
        self.feature_table.setItem(row, self.COL_NAME, header_item.clone())
        for col in (self.COL_SPARK, self.COL_SKEW, self.COL_NORM,
                    self.COL_XFRM):
            empty = QTableWidgetItem("")
            empty.setFlags(Qt.ItemIsEnabled)
            self.feature_table.setItem(row, col, empty)
        header_item.setToolTip(T.CATEGORY_HEADER_TIP)
        self.feature_table.item(row, self.COL_NAME).setToolTip(
            T.CATEGORY_HEADER_TIP)
        self.feature_table.setSpan(row, 0, 1, self.feature_table.columnCount())
        self._category_header_rows[category] = (row, count)
        self.feature_table.setRowHeight(row, 24)
        return row

    def _refresh_category_headers(self):
        """Rewrite every category header's "{on} of {n} used"."""
        was = self.feature_table.blockSignals(True)
        try:
            self._refresh_category_headers_now()
        finally:
            self.feature_table.blockSignals(was)

    def _refresh_category_headers_now(self):
        for category, (row, count) in self._category_header_rows.items():
            item = self.feature_table.item(row, self.COL_CHECK)
            if item is None:
                continue
            on = 0
            for r in self._category_rows.get(category, []):
                cb = self.feature_table.item(r, self.COL_CHECK)
                if cb is not None and cb.data(Qt.UserRole) in \
                        self._selected_features:
                    on += 1
            fmt = (T.CATEGORY_HEADER_SHUT if category in self._collapsed
                   else T.CATEGORY_HEADER)
            item.setText(fmt.format(cat=category, on=on, n=count))
            for r in self._category_rows.get(category, []):
                self.feature_table.setRowHidden(
                    r, category in self._collapsed)

    def _insert_feature_row(self, row: int, feat: str, has_profiles: bool):
        """Insert a single feature row with checkbox, name, and stats."""
        self.feature_table.insertRow(row)
        is_selected = feat in self._selected_features

        # Col 0: Checkbox
        cb_item = QTableWidgetItem()
        cb_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        cb_item.setCheckState(Qt.Checked if is_selected else Qt.Unchecked)
        cb_item.setData(Qt.UserRole, feat)
        self.feature_table.setItem(row, self.COL_CHECK, cb_item)

        # Col 1: Feature name
        name_item = QTableWidgetItem(f"  {feat}")
        name_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
        name_item.setForeground(QBrush(QColor(_TEXT)))
        desc = _get_feature_description(feat)
        tooltip = f"{feat}\n{desc}" if desc else feat
        name_item.setToolTip(tooltip)
        self.feature_table.setItem(row, self.COL_NAME, name_item)

        if has_profiles and feat in self._profiles:
            profile = self._profiles[feat]
            self._set_sparkline_cell(row, profile, is_selected)
            self._set_skewness_cell(row, profile)
            self._set_normality_cell(row, profile)
            self._set_transform_cell(row, feat)
            self.feature_table.setRowHeight(row, 26)
        else:
            # Phase 1 fallback: empty stats columns
            for col in (self.COL_SPARK, self.COL_SKEW, self.COL_NORM,
                        self.COL_XFRM):
                empty = QTableWidgetItem("")
                empty.setFlags(Qt.ItemIsEnabled)
                self.feature_table.setItem(row, col, empty)
            self.feature_table.setRowHeight(row, 24)

    def _set_sparkline_cell(self, row: int, profile: dict, selected: bool):
        """Render and place a sparkline pixmap in the table: the rescaled
        values, green whether the feature is ticked or not (mockup)."""
        cb = self.feature_table.item(row, self.COL_CHECK)
        feat = cb.data(Qt.UserRole) if cb is not None else None
        hist = self._spark_hist.get(feat) or profile["histogram"]
        color = _ACCENT
        pixmap = render_sparkline(
            hist["counts"], hist["max_count"], bar_color=color,
        )
        label = QLabel()
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("background: transparent; padding: 0;")
        label.setToolTip(H["Shape"])
        self.feature_table.setCellWidget(row, self.COL_SPARK, label)

    def _set_skewness_cell(self, row: int, profile: dict):
        """Set skewness text with color and arrow."""
        sk = profile["skewness"]
        val = sk["value"]
        text, color = skew_look(val)

        item = QTableWidgetItem(text)
        item.setFlags(Qt.ItemIsEnabled)
        item.setForeground(QBrush(QColor(color)))
        item.setTextAlignment(Qt.AlignCenter)
        item.setToolTip(T.WB_SKEW_TIP.format(v=val) if not np.isnan(val)
                        else T.WB_SKEW_TIP_NONE)
        self.feature_table.setItem(row, self.COL_SKEW, item)

    def _set_normality_cell(self, row: int, profile: dict):
        """The Bell column: a filled dot when
        |skew| < 0.5 and |excess kurtosis| < 1, hollow otherwise.  The
        Shapiro-Wilk p-value stays in the tooltip and the detail panel."""
        norm = profile["normality"]
        cb = self.feature_table.item(row, self.COL_CHECK)
        feat = cb.data(Qt.UserRole) if cb is not None else None
        skew, kurt = self._bell.get(feat, (float("nan"), float("nan")))
        bell = T.is_bell(skew, kurt)
        item = QTableWidgetItem("●" if bell else "○")
        item.setFlags(Qt.ItemIsEnabled)
        item.setForeground(QBrush(QColor(_ACCENT if bell else "#666666")))
        item.setTextAlignment(Qt.AlignCenter)

        p = norm["p_value"]
        if np.isnan(p):
            tooltip = T.WB_NORM_TIP_NONE
        else:
            test_name = (T.WB_NORM_TEST_SHAPIRO if norm["test"] == "shapiro"
                         else T.WB_NORM_TEST_DAGOSTINO)
            tooltip = (T.WB_NORM_TIP_NORMAL if norm["normal"]
                       else T.WB_NORM_TIP_NON).format(test=test_name, p=p)
        item.setToolTip(tooltip)
        self.feature_table.setItem(row, self.COL_NORM, item)

    def _set_transform_cell(self, row: int, feat: str):
        """Place a transform combo box in the table."""
        from napari_epicycle.segmentation.feature_stats import (
            TRANSFORM_METHODS,
            resolve_auto_method,
        )

        # The mockup's words; the DATA is the transform key.  "auto" says
        # what it picked ("auto · log").
        combo = QComboBox()
        raw = self._raw_data.get(feat)
        picked = resolve_auto_method(raw, feat) if raw is not None else None
        for method in TRANSFORM_METHODS:
            word = RESCALE_WORDS.get(method, method)
            if method == "auto" and picked:
                word = f"auto · {RESCALE_WORDS.get(picked, picked)}"
            combo.addItem(word, method)
        current = self._transforms.get(feat, "auto")
        idx = combo.findData(current)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        combo.setToolTip(H["Rescale"])
        combo.setStyleSheet(f"""
            QComboBox {{
                background: rgba(255, 255, 255, 0.04);
                color: {_TEXT};
                border: 1px solid rgba(255, 255, 255, 0.10);
                border-radius: 3px;
                padding: 0px 2px;
                font-size: 10px;
            }}
            QComboBox::drop-down {{
                border: none;
                width: 14px;
            }}
            QComboBox QAbstractItemView {{
                background: {_BG};
                color: {_TEXT};
                selection-background-color: rgba(106, 191, 105, 0.25);
                font-size: 10px;
            }}
        """)
        combo.setProperty("feature_name", feat)
        combo.setProperty("table_row", row)
        combo.currentIndexChanged.connect(self._on_transform_combo_changed)
        self.feature_table.setCellWidget(row, self.COL_XFRM, combo)

    def _on_transform_combo_changed(self, _index):
        """A Rescale combo changed: its row and feature ride on the combo."""
        combo = self.sender()
        if not isinstance(combo, QComboBox):
            return
        feat = combo.property("feature_name")
        row = combo.property("table_row")
        method = combo.currentData()
        if feat and method:
            self._on_transform_changed(feat, int(row), method)

    def _on_transform_changed(self, feat: str, row: int, method: str):
        """Handle transform dropdown change — recompute stats for this row."""
        self._transforms[feat] = method
        self._recompute_row(feat, row)
        self._mark_embeddings_stale()

    def _recompute_row(self, feat: str, row: int):
        """Recompute sparkline/skew/norm for a feature after transform change."""
        from napari_epicycle.segmentation.feature_stats import (
            apply_transform,
            compute_feature_profile,
        )

        raw = self._raw_data.get(feat)
        if raw is None:
            return

        method = self._transforms.get(feat, "auto")
        transformed = apply_transform(raw, method, feature_name=feat)
        profile = compute_feature_profile(transformed)
        self._profiles[feat] = profile
        self._bell[feat] = self._skew_kurt(transformed)
        self._spark_hist[feat] = profile["histogram"]

        is_selected = feat in self._selected_features
        self._set_sparkline_cell(row, profile, is_selected)
        self._set_skewness_cell(row, profile)
        self._set_normality_cell(row, profile)

    def _auto_transform_all(self):
        """Apply skewness-based auto-transform rules to all features."""
        from napari_epicycle.segmentation.feature_stats import resolve_auto_method

        for row in range(self.feature_table.rowCount()):
            cb_item = self.feature_table.item(row, self.COL_CHECK)
            if not cb_item or not (cb_item.flags() & Qt.ItemIsUserCheckable):
                continue
            feat = cb_item.data(Qt.UserRole)
            if not feat or feat not in self._raw_data:
                continue

            resolved = resolve_auto_method(self._raw_data[feat], feat)
            self._transforms[feat] = resolved

            # Update combo box
            combo = self.feature_table.cellWidget(row, self.COL_XFRM)
            if isinstance(combo, QComboBox):
                combo.blockSignals(True)
                idx = combo.findData(resolved)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
                combo.blockSignals(False)

            # Recompute stats
            self._recompute_row(feat, row)

        # The Rescale column shows each feature's new rescale; the old
        # "Auto-transform: n → log" status line is gone with the status
        # line itself.
        self._mark_embeddings_stale()

    def _on_cell_clicked(self, row: int, col: int):
        """A click on a category header folds or unfolds it (the caret, as
        the mockup promises); Ctrl+click ticks or unticks every feature in
        it (W1-3)."""
        from qtpy.QtWidgets import QApplication

        name_item = self.feature_table.item(row, self.COL_NAME)
        if name_item is None:
            return
        data = name_item.data(Qt.UserRole)
        if not isinstance(data, str) or not data.startswith("__category__"):
            return
        category = data[len("__category__"):]
        if category not in self._category_rows:
            return
        ctrl = bool(QApplication.keyboardModifiers() & Qt.ControlModifier)
        self.on_category_header_clicked(category, ctrl=ctrl)

    def on_category_header_clicked(self, category: str, *, ctrl: bool):
        """Fold / unfold ``category``, or with ``ctrl`` tick / untick all."""
        if category not in self._category_rows:
            return
        if not ctrl:
            if category in self._collapsed:
                self._collapsed.discard(category)
            else:
                self._collapsed.add(category)
            self._refresh_category_headers()
            return
        self._toggle_category_ticks(category)

    def _toggle_category_ticks(self, category: str):
        # Check current state: if all are selected, deselect all; else select all
        feat_rows = self._category_rows[category]
        feats_in_cat = []
        for r in feat_rows:
            cb = self.feature_table.item(r, self.COL_CHECK)
            if cb and cb.flags() & Qt.ItemIsUserCheckable:
                feat = cb.data(Qt.UserRole)
                if feat:
                    feats_in_cat.append((r, feat))

        all_selected = all(f in self._selected_features for _, f in feats_in_cat)
        new_state = Qt.Unchecked if all_selected else Qt.Checked

        self.feature_table.blockSignals(True)
        for r, feat in feats_in_cat:
            cb = self.feature_table.item(r, self.COL_CHECK)
            cb.setCheckState(new_state)
            if new_state == Qt.Checked:
                self._selected_features.add(feat)
            else:
                self._selected_features.discard(feat)
            if feat in self._profiles:
                self._set_sparkline_cell(
                    r, self._profiles[feat], new_state == Qt.Checked,
                )
        self.feature_table.blockSignals(False)
        self._update_selection_label()
        self._mark_embeddings_stale()

    def _mark_embeddings_stale(self, *_):
        """PCA_STALE (amber, Map tab) once a PCA exists and the feature
        list, a rescale or the scaling changed."""
        if self._pca_result is None:
            return  # no embeddings computed yet — nothing to mark stale
        if not self._embeddings_stale:
            self._embeddings_stale = True
            self._pca_stale_label.show()

    def _on_checkbox_changed(self, item):
        """Track checkbox state changes."""
        if item.column() != self.COL_CHECK:
            return
        feat = item.data(Qt.UserRole)
        if feat is None or str(feat).startswith("__category__"):
            return
        was_selected = feat in self._selected_features
        if item.checkState() == Qt.Checked:
            self._selected_features.add(feat)
        else:
            self._selected_features.discard(feat)

        is_selected = feat in self._selected_features
        if was_selected != is_selected and feat in self._profiles:
            # Update sparkline color
            row = item.row()
            self._set_sparkline_cell(row, self._profiles[feat], is_selected)

        self._update_selection_label()
        self._mark_embeddings_stale()

    def _update_selection_label(self):
        n_sel = len(self._selected_features)
        n_total = len(self._feature_names)
        self.selection_label.setText(T.FEAT_COUNT.format(k=n_sel, n=n_total))
        self._refresh_category_headers()
        # Update PCs spinner max to match number of selected features
        if n_sel > 0:
            self._n_components_spin.setMaximum(n_sel)
            if self._n_components_spin.value() > n_sel:
                self._n_components_spin.setValue(min(10, n_sel))

    # ------------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------------

    def _select_all(self):
        self.feature_table.blockSignals(True)
        for row in range(self.feature_table.rowCount()):
            item = self.feature_table.item(row, self.COL_CHECK)
            if item and item.flags() & Qt.ItemIsUserCheckable:
                item.setCheckState(Qt.Checked)
                feat = item.data(Qt.UserRole)
                if feat:
                    self._selected_features.add(feat)
                    if feat in self._profiles:
                        self._set_sparkline_cell(row, self._profiles[feat], True)
        self.feature_table.blockSignals(False)
        self._update_selection_label()

    def _deselect_all(self):
        self.feature_table.blockSignals(True)
        for row in range(self.feature_table.rowCount()):
            item = self.feature_table.item(row, self.COL_CHECK)
            if item and item.flags() & Qt.ItemIsUserCheckable:
                item.setCheckState(Qt.Unchecked)
                feat = item.data(Qt.UserRole)
                if feat and feat in self._profiles:
                    self._set_sparkline_cell(row, self._profiles[feat], False)
        self.feature_table.blockSignals(False)
        self._selected_features.clear()
        self._update_selection_label()

    def _reset_defaults(self):
        from napari_epicycle.fusion_reconstruction.qc_features import PCA_FEATURES
        self._selected_features = set(
            f for f in PCA_FEATURES if f in self._feature_names
        )
        self.feature_table.blockSignals(True)
        for row in range(self.feature_table.rowCount()):
            item = self.feature_table.item(row, self.COL_CHECK)
            if item and item.flags() & Qt.ItemIsUserCheckable:
                feat = item.data(Qt.UserRole)
                is_sel = feat in self._selected_features
                item.setCheckState(Qt.Checked if is_sel else Qt.Unchecked)
                if feat and feat in self._profiles:
                    self._set_sparkline_cell(row, self._profiles[feat], is_sel)
        self.feature_table.blockSignals(False)
        self._update_selection_label()

    def get_selected_features(self) -> list:
        """Return the list of selected feature names."""
        return [f for f in self._feature_names if f in self._selected_features]

    def restore_selected_features(self, features: list):
        """Restore feature checkbox state from a saved list.

        Features not present in the current table are silently ignored.
        """
        valid = set(f for f in features if f in self._feature_names)
        self._selected_features = valid
        self.feature_table.blockSignals(True)
        for row in range(self.feature_table.rowCount()):
            item = self.feature_table.item(row, self.COL_CHECK)
            if item and item.flags() & Qt.ItemIsUserCheckable:
                feat = item.data(Qt.UserRole)
                is_sel = feat in self._selected_features
                item.setCheckState(Qt.Checked if is_sel else Qt.Unchecked)
                if feat and feat in self._profiles:
                    self._set_sparkline_cell(row, self._profiles[feat], is_sel)
        self.feature_table.blockSignals(False)
        self._update_selection_label()

    def restore_transforms(self, transforms: dict):
        """Restore per-feature transform choices from a saved dict.

        Only applies transforms for features present in the current table.
        Unknown features or invalid methods are silently ignored.
        """
        valid_methods = {"auto", "raw", "log", "rank"}
        for feat, method in transforms.items():
            if feat in self._transforms and method in valid_methods:
                self._transforms[feat] = method

        # Update combo box widgets in the table
        for row in range(self.feature_table.rowCount()):
            combo = self.feature_table.cellWidget(row, self.COL_XFRM)
            if combo is not None:
                feat = combo.property("feature_name")
                if feat and feat in transforms:
                    method = transforms[feat]
                    if method in valid_methods:
                        combo.blockSignals(True)
                        idx = combo.findData(method)
                        if idx >= 0:
                            combo.setCurrentIndex(idx)
                        combo.blockSignals(False)

    # ------------------------------------------------------------------
    # PCA method selection (Phase 5)
    # ------------------------------------------------------------------

    def _populate_pc_combos(self, n: int):
        """Populate the PC axis selector combos with PC1..PCn."""
        for idx, combo in enumerate((self._pc_x_combo, self._pc_y_combo)):
            combo.blockSignals(True)
            prev = combo.currentIndex()
            combo.clear()
            for i in range(n):
                combo.addItem(f"PC{i + 1}")
            if prev >= 0 and prev < n:
                combo.setCurrentIndex(prev)
            else:
                # Default: X=PC1 (0), Y=PC2 (1)
                combo.setCurrentIndex(min(idx, n - 1))
            combo.blockSignals(False)

    def show_pca_result(self, result: dict, pc_x: int = 0, pc_y: int = 1):
        """Display PCA results: the named loadings table.

        Called by the controller after PCA computation.  Every ticked
        feature is a row (no 20-feature cap), sorted by |PC1|; every PC is
        a column whose header carries its variance bar.
        """
        self._pca_result = result
        self._embeddings_stale = False
        self._pca_stale_label.hide()
        self._map_error_label.hide()
        self._pca_results_panel.setVisible(True)

        # The first "Start from" item names the PCA kind now stored; its
        # DATA is "PCA" or "Residual PCA", and since B3b the controller
        # reads both as the stored projection (``_eda_worker.PCA_INPUTS``).
        is_residual = "regress_cols_used" in result
        pca_label = "Residual PCA" if is_residual else "PCA"
        for combo in (self._umap_input_combo, self._tsne_input_combo,
                      self._cluster_input_combo):
            combo.blockSignals(True)
            prev = combo.currentIndex()
            combo.setItemText(0, f"{pca_label} components")
            combo.setItemData(0, pca_label)
            combo.setCurrentIndex(prev)
            combo.blockSignals(False)

        loadings = np.asarray(result["loadings"])   # (n_components, n_features)
        var_ratio = np.asarray(result["explained_variance_ratio"])
        feat_names = list(result["feature_names"])
        n_pcs = len(var_ratio)

        # Update PC axis combos
        self._populate_pc_combos(n_pcs)
        self._pc_x_combo.blockSignals(True)
        self._pc_y_combo.blockSignals(True)
        self._pc_x_combo.setCurrentIndex(min(pc_x, n_pcs - 1))
        self._pc_y_combo.setCurrentIndex(min(pc_y, n_pcs - 1))
        self._pc_x_combo.blockSignals(False)
        self._pc_y_combo.blockSignals(False)

        v12 = float(var_ratio[:2].sum()) * 100.0 if n_pcs else 0.0
        self._pca_done_label.setText(
            T.PCA_DONE.format(f=len(feat_names), v=f"{v12:.0f}"))
        excluded = self._pca_excluded_n if is_residual else 0
        self._pca_excluded_label.setText(
            T.PCA_EXCLUDED.format(k=excluded) if excluded else "")
        self._pca_excluded_label.setVisible(bool(excluded))
        self._fill_loadings_table(loadings, var_ratio, feat_names)

    def _fill_loadings_table(self, loadings, var_ratio, feat_names):
        """Every feature x every component, sorted by |PC1|."""
        table = self.loadings_table
        n_pcs = int(len(var_ratio))
        n_feat = int(loadings.shape[1]) if loadings.ndim == 2 else 0
        order = (list(np.argsort(-np.abs(loadings[0]), kind="stable"))
                 if n_feat and n_pcs else list(range(n_feat)))
        table.clear()
        table.setRowCount(n_feat)
        table.setColumnCount(1 + n_pcs)
        self._loadings_header.set_shares(var_ratio)
        mono = QFont("Consolas")
        mono.setPixelSize(10)
        for r, j in enumerate(order):
            name = QTableWidgetItem(str(feat_names[j]))
            name.setForeground(QBrush(QColor("#cccccc")))
            name.setToolTip(_get_feature_description(str(feat_names[j]))
                            or str(feat_names[j]))
            table.setItem(r, 0, name)
            for i in range(n_pcs):
                v = float(loadings[i, j])
                cell = QTableWidgetItem(f"{v:+.2f}")
                cell.setTextAlignment(Qt.AlignCenter)
                cell.setFont(mono)
                cell.setBackground(QBrush(loading_tint(v)))
                cell.setForeground(QBrush(QColor("#eeeeee")))
                table.setItem(r, 1 + i, cell)
        hdr = table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Interactive)
        table.resizeColumnToContents(0)
        table.setColumnWidth(0, min(170, max(118, table.columnWidth(0))))
        for i in range(n_pcs):
            hdr.setSectionResizeMode(1 + i, QHeaderView.Fixed)
            table.setColumnWidth(1 + i, 44)
        rows_h = table.verticalHeader().defaultSectionSize() * n_feat
        scroll_h = table.horizontalScrollBar().sizeHint().height()
        table.setFixedHeight(hdr.height() + rows_h + scroll_h
                             + 2 * table.frameWidth() + 2)

    def loadings_rows(self) -> list:
        """The loadings table's feature names, top to bottom (for tests and
        the shots driver)."""
        return [self.loadings_table.item(r, 0).text()
                for r in range(self.loadings_table.rowCount())]

    def _on_pca_method_changed(self, method: str):
        is_residual = method == "Residual PCA"
        self._regress_out_widget.setVisible(is_residual)
        self.compute_pca_btn.setText(
            T.COMPUTE_RESIDUAL_PCA_BTN if is_residual else T.COMPUTE_PCA_BTN
        )

    def get_regress_out_features(self) -> list:
        """Return the list of checked regress-out features."""
        return [
            feat for feat, cb in self._regress_checkboxes.items()
            if cb.isChecked()
        ]

    # ------------------------------------------------------------------
    # Compute signals
    # ------------------------------------------------------------------

    def current_pca_params(self) -> Optional[dict]:
        """The params Compute PCA would send now (``None`` with fewer than
        two features) -- for the landing's staleness check (audit E1-1)."""
        n = self._pca_excluded_n
        try:
            return self._pca_params()
        finally:
            self._pca_excluded_n = n

    def _on_compute_pca(self):
        params = self._pca_params()
        if params is None:
            self._map_error_label.setText(T.PCA_TOO_FEW)
            self._map_error_label.show()
            return
        self._map_error_label.hide()
        self.compute_pca_requested.emit(params)

    def _pca_params(self) -> Optional[dict]:
        method = self._pca_method_combo.currentText()
        self._pca_excluded_n = 0
        selected = self.get_selected_features()
        # Scaling controls
        is_robust = "Robust" in self._scaling_combo.currentText()
        scaling_mode = "qc" if is_robust else "eda"
        winsorize = (
            self._winsorize_spin.value() if self._winsorize_cb.isChecked()
            else None
        )

        if method == "Residual PCA":
            regress_out = self.get_regress_out_features()
            # Auto-exclude regress-out features from PCA input (the old
            # "Note: n regress-out features auto-excluded" status line went
            # with the status line; PCA_DONE names the feature count).
            regress_set = set(regress_out)
            before = len(selected)
            selected = [f for f in selected if f not in regress_set]
            self._pca_excluded_n = before - len(selected)

        if len(selected) < 2:
            return None

        # Pass user's per-feature transform choices so PCA respects them
        transforms = {f: self._transforms.get(f, "auto") for f in selected}
        params = {
            "features": selected,
            "transforms": transforms,
            "n_components": self._n_components_spin.value(),
            "pc_x": self._pc_x_combo.currentIndex(),
            "pc_y": self._pc_y_combo.currentIndex(),
            "scaling": scaling_mode,
            "winsorize": winsorize,
        }
        if method == "Residual PCA":
            params["method"] = "residual"
            params["regress_out"] = regress_out
        else:
            params["mode"] = scaling_mode
        return params

    def _on_pc_axis_changed(self):
        """Re-emit PCA request when user changes which PCs to display."""
        if self._pca_result is None:
            return
        # Re-send with stored result to update scatter plot axes
        self._on_compute_pca()

    def _on_compute_clusters(self):
        algo = self._input_value(self._cluster_algo_combo)
        k = self._cluster_k_spin.value()
        selected = self.get_selected_features()
        transforms = {f: self._transforms.get(f, "auto") for f in selected}
        is_robust = "Robust" in self._scaling_combo.currentText()
        winsorize = (
            self._winsorize_spin.value() if self._winsorize_cb.isChecked()
            else None
        )
        self.compute_clusters_requested.emit({
            "algorithm": algo,
            "features": selected,
            "transforms": transforms,
            "scaling": "qc" if is_robust else "eda",
            "winsorize": winsorize,
            "resolution": self._cluster_resolution_spin.value(),
            "n_neighbors": self._cluster_knn_spin.value(),
            "n_clusters": k,
            "n_components": self._n_components_spin.value(),
            "min_cluster_size": k,
            "input_space": self._input_value(self._cluster_input_combo),
            "metric": self._cluster_metric_combo.currentText(),
            # Spec: Leiden / K-means / GMM take it; HDBSCAN ignores it.
            "seed": self._cluster_seed_spin.value(),
        })

    def _on_compute_umap(self):
        selected = self.get_selected_features()
        transforms = {f: self._transforms.get(f, "auto") for f in selected}
        is_robust = "Robust" in self._scaling_combo.currentText()
        winsorize = (
            self._winsorize_spin.value() if self._winsorize_cb.isChecked()
            else None
        )
        self.compute_umap_requested.emit({
            "features": selected,
            "transforms": transforms,
            "scaling": "qc" if is_robust else "eda",
            "winsorize": winsorize,
            "n_neighbors": self._umap_neighbors_spin.value(),
            "min_dist": self._umap_min_dist_spin.value(),
            "input_space": self._input_value(self._umap_input_combo),
            "pca_dims": self._n_components_spin.value(),
            "metric": self._umap_metric_combo.currentText(),
        })

    def _on_compute_tsne(self):
        selected = self.get_selected_features()
        transforms = {f: self._transforms.get(f, "auto") for f in selected}
        is_robust = "Robust" in self._scaling_combo.currentText()
        winsorize = (
            self._winsorize_spin.value() if self._winsorize_cb.isChecked()
            else None
        )
        self.compute_tsne_requested.emit({
            "features": selected,
            "transforms": transforms,
            "scaling": "qc" if is_robust else "eda",
            "winsorize": winsorize,
            "perplexity": self._tsne_perplexity_spin.value(),
            "input_space": self._input_value(self._tsne_input_combo),
            "pca_dims": self._n_components_spin.value(),
        })
