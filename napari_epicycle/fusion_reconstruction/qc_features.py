"""Pure-Python QC feature extraction from 3D segmentation masks.

Used by the Reconstruct + Curate 3D section to detect bad segmentation
masks via interactive plots, threshold filtering, and per-patch CSV export.

No Qt dependency — this module is imported by both the controller and tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# Fast properties — includes inertia tensor for orientation/ellipsoid features.
# NOTE: axis_major_length / axis_minor_length are intentionally EXCLUDED because
# skimage computes them via Python's math.sqrt() which raises ValueError on
# degenerate labels with negative eigenvalues.  We compute them safely below
# using np.sqrt(np.clip(...)).
_FAST_PROPERTIES = (
    "label",
    "area",
    "area_filled",
    "equivalent_diameter",
    "extent",
    "euler_number",
    "centroid",
    "bbox",
    "inertia_tensor_eigvals",
    "inertia_tensor",
)

# Intensity properties — cheap but require intensity_image
_INTENSITY_PROPERTIES = (
    "intensity_mean",
    "intensity_std",
    "intensity_min",
    "intensity_max",
)

# Legacy expensive properties — solidity now computed via fast ROI approach
_EXPENSIVE_PROPERTIES = (
    "solidity",
)

# Columns in the output DataFrame (after derivation)
FEATURE_COLUMNS = [
    "label",
    "area",
    "area_filled",
    "equivalent_diameter",
    "extent",
    "euler_number",
    "centroid-0",
    "centroid-1",
    "centroid-2",
    "bbox-0",
    "bbox-1",
    "bbox-2",
    "bbox-3",
    "bbox-4",
    "bbox-5",
    "sphericity",
    "aspect_ratio",
    "bbox_volume",
    "compactness",
    "z_span",
    "z_elongation",
    "xy_elongation",
    "fill_ratio",
    "ellipsoid_solidity",
    "major_axis_z_angle",
    "inertia_elongation",
    "inertia_flatness",
    "axis_major_length",
    "axis_minor_length",
    # Added by compute_surface_features (optional):
    # "surface_area", "true_sphericity", "roughness", "solidity"
    # Added when intensity_image provided:
    # "intensity_mean", "intensity_std", "intensity_min", "intensity_max"
    # Added by compute_boundary_interior_features:
    # "boundary_intensity_mean", "interior_intensity_mean",
    # "boundary_interior_ratio", "z_intensity_variation"
    # Added by compute_probability_features (when probability volume provided):
    # "prob_mean", "prob_p10", "prob_std",
    # "prob_boundary_mean", "prob_interior_mean", "prob_boundary_drop",
    # "prob_weighted_volume", "prob_soft_hard_ratio",
    # "prob_boundary_entropy", "prob_mask_entropy",
    # "prob_edge_gradient", "prob_edge_gradient_std"
    # Added by compute_neighbor_features:
    # "neighbor_count", "nearest_neighbor_distance",
    # "max_contact_fraction", "background_fraction"
    # Added by compute_skeleton_features:
    # "skeleton_branches", "skeleton_nodes", "skeleton_length"
    # Added by compute_composite_score:
    # "qc_score"
]

# Numeric features suitable for plotting / PCA (excludes label, centroid, bbox)
# Plot widget auto-filters to columns that actually exist in the DataFrame.
NUMERIC_FEATURES = [
    # Core morphology (always present)
    "area",
    "area_filled",
    "equivalent_diameter",
    "extent",
    "euler_number",
    "sphericity",
    "aspect_ratio",
    "bbox_volume",
    "compactness",
    # Orientation & shape (always present)
    "z_span",
    "z_elongation",
    "xy_elongation",
    "fill_ratio",
    "ellipsoid_solidity",
    "major_axis_z_angle",
    "inertia_elongation",
    "inertia_flatness",
    "axis_major_length",
    "axis_minor_length",
    # Surface features (from compute_surface_features)
    "surface_area",
    "true_sphericity",
    "roughness",
    "solidity",
    # Intensity (only when intensity_image provided)
    "intensity_mean",
    "intensity_std",
    "intensity_min",
    "intensity_max",
    # Custom membrane intensity (only when intensity_image provided)
    "boundary_intensity_mean",
    "interior_intensity_mean",
    "boundary_interior_ratio",
    "z_intensity_variation",
    # Probability consensus (when probability volume provided)
    "prob_mean",
    "prob_p10",
    "prob_std",
    "prob_boundary_mean",
    "prob_interior_mean",
    "prob_boundary_drop",
    "prob_weighted_volume",
    "prob_soft_hard_ratio",
    "prob_boundary_entropy",
    "prob_mask_entropy",
    "prob_edge_gradient",
    "prob_edge_gradient_std",
    # Neighbor/adjacency
    "neighbor_count",
    "nearest_neighbor_distance",
    "max_contact_fraction",
    "background_fraction",
    # Skeleton topology
    "skeleton_branches",
    "skeleton_nodes",
    "skeleton_length",
    # Composite
    "qc_score",
]

# Bounded [0,1] features — NEVER log-transform these regardless of skewness.
# log1p compresses their already-small range unhelpfully.
BOUNDED_FEATURES = frozenset([
    "compactness", "extent", "fill_ratio", "solidity", "true_sphericity",
    "sphericity", "ellipsoid_solidity",
    # Bribiesca discrete compactness: 0 = single isolated voxel,
    # 1 = perfect cube, sphere ~= 0.85. Bounded [0, 1].
    "discrete_compactness",
    # Probability features bounded [0,1]
    "prob_mean", "prob_p10", "prob_boundary_mean", "prob_interior_mean",
    "prob_boundary_entropy", "prob_mask_entropy",
    # NOTE: prob_soft_hard_ratio is NOT bounded — V_soft/|M| can exceed 1
    # when probability mass extends beyond the mask. Left to auto-detect.
    # Surface fraction ratios
    "background_fraction", "max_contact_fraction",
    # Cross-section (bounded [0,1])
    "max_cross_section_z_relative",
    # Z-stratified neighbor topology (bounded [0,1])
    "neighbor_z_turnover", "neighbor_persistence_ratio",
])

# Skewness threshold for auto-detection. Features with |skew| above this
# get log1p transform before PCA/composite scoring.
# 1.0 is standard in cytometry (pycytominer convention).
_SKEW_THRESHOLD = 1.0

# Legacy alias — kept for backward compatibility with tests that import it.
# No longer used internally; auto-detection replaces it.
LOG_TRANSFORM_FEATURES = frozenset([
    "area", "area_filled", "equivalent_diameter", "bbox_volume",
    "z_span", "surface_area", "axis_major_length", "axis_minor_length",
    "nearest_neighbor_distance", "skeleton_length",
    "intensity_mean", "intensity_std", "intensity_min", "intensity_max",
    "boundary_intensity_mean", "interior_intensity_mean",
    "prob_weighted_volume",
    "prob_soft_hard_ratio",
])


def _detect_skewed_columns(
    X: np.ndarray,
    cols: List[str],
    threshold: float = _SKEW_THRESHOLD,
) -> List[int]:
    """Return column indices that are right-skewed and unbounded.

    Skewness is computed via Fisher's definition (scipy.stats.skew).
    Bounded [0,1] features and near-constant columns are excluded.
    Only right-skewed features (positive skewness) are flagged, since
    ``log1p`` only corrects right-skew — applying it to left-skewed
    data would make the distribution worse.

    Note: ``boundary_interior_ratio`` and ``_intensity_cv`` are
    unbounded (can exceed 1.0) and are intentionally NOT in
    BOUNDED_FEATURES — auto-detect handles them correctly.

    Args:
        X: Feature matrix (n_samples, n_features).
        cols: Column names.
        threshold: Skewness above which to log-transform.

    Returns:
        List of column indices to log1p-transform.
    """
    from scipy.stats import skew as _skew

    skew_indices = []
    for j, col in enumerate(cols):
        if col in BOUNDED_FEATURES:
            continue
        col_data = X[:, j]
        finite = col_data[np.isfinite(col_data)]
        if finite.size < 3:
            continue
        # Skip near-constant columns (MAD ≈ 0)
        mad = np.median(np.abs(finite - np.median(finite)))
        if mad < 1e-12:
            continue
        # Only flag right-skewed (positive skewness) — log1p doesn't fix left-skew
        s = _skew(finite)
        if s > threshold:
            skew_indices.append(j)
    return skew_indices


def _robust_standardize(
    X: np.ndarray,
    cols: List[str],
    exclude_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Curated + auto-detect log-transform, then robust-standardize.

    Two-stage skewness handling:

    1a. Always ``log1p`` the curated ``LOG_TRANSFORM_FEATURES`` set
        (preserves validated PCA behavior for known features).
    1b. Auto-detect additional columns with |skew| > 1.0 (catches
        per-channel intensity and other new features).
    2. Robust z-score using ``median / MAD`` (median absolute deviation).
    3. Winsorize to ±10 robust z-scores.

    This preserves Step 3's PCA behavior while adapting to new features.

    Args:
        X: Feature matrix (n_samples, n_features), modified in-place.
        cols: Column names corresponding to X's columns (same length).
        exclude_mask: Optional boolean array (n_samples,). True = exclude
            from fitting (median/MAD computation) but still transform.
            Used to exclude border cells from PCA fitting while still
            projecting them into the same space.

    Returns:
        Standardized matrix (same shape).
    """
    # Step 0: replace NaN/Inf with column median (prevents NaN poisoning PCA)
    for j in range(X.shape[1]):
        col_data = X[:, j]
        bad = ~np.isfinite(col_data)
        if bad.any():
            finite_vals = col_data[~bad]
            fill = np.median(finite_vals) if finite_vals.size > 0 else 0.0
            col_data[bad] = fill

    # Step 1: log1p transform (applied to ALL rows, including excluded).
    curated_indices = {
        j for j, col in enumerate(cols) if col in LOG_TRANSFORM_FEATURES
    }
    # Auto-detect on fitting rows only (exclude_mask aware)
    if exclude_mask is not None and exclude_mask.any():
        fit_rows = ~exclude_mask
        auto_indices = set(_detect_skewed_columns(X[fit_rows], cols))
    else:
        auto_indices = set(_detect_skewed_columns(X, cols))
    transform_indices = curated_indices | auto_indices
    for j in transform_indices:
        X[:, j] = np.log1p(np.clip(X[:, j], 0, None))

    # Step 2: robust standardize — median / MAD
    # Fit on non-excluded rows only; transform ALL rows.
    if exclude_mask is not None and exclude_mask.any():
        fit_X = X[~exclude_mask]
    else:
        fit_X = X
    med = np.median(fit_X, axis=0)
    mad = np.median(np.abs(fit_X - med), axis=0)
    scale = mad * 1.4826
    scale[scale < 1e-12] = 1.0

    # Step 3: winsorize — clip to ±10 robust z-scores.
    return np.clip((X - med) / scale, -10, 10)


# Curated feature subset for PCA / composite score.
# Selection criteria:
#   1. One representative per correlated group (no multicollinearity)
#   2. Exclude derived/circular features (qc_score is computed FROM features)
#   3. Prefer continuous over discrete, keep discrete with unique signal
#   4. 15 roughly independent features covering morphology, shape, surface,
#      and topology dimensions.
PCA_FEATURES = [
    # --- Primary morphology (1 representative each) ---
    "area",                      # volume — primary size measure
    "extent",                    # fill fraction of bounding box
    "equivalent_diameter",       # alternative size scale
    # --- Shape (1 per correlated group) ---
    "aspect_ratio",              # bbox elongation (represents z/xy_elongation)
    # NOTE: compactness excluded — numerically identical to extent (both = V/V_bbox).
    # Kept in extract_features output for backward-compat with existing CSVs.
    "true_sphericity",           # gold-standard shape (replaces bbox sphericity)
    "solidity",                  # vol/convex_hull (replaces fill_ratio, ellipsoid_solidity)
    # --- Orientation ---
    "major_axis_z_angle",        # tilt relative to Z axis
    "inertia_elongation",        # eigenvalue-based (replaces axis lengths, inertia_flatness)
    # --- Surface ---
    "surface_area",              # independent from volume-based features
    # NOTE: roughness excluded — it's 1/true_sphericity (perfect inverse correlation)
    # NOTE: boundary_interior_ratio excluded — use-case dependent (requires membrane
    # channel as seg channel). Users can add per-channel intensity features via the
    # Extract dialog.
    # --- Topology ---
    "neighbor_count",            # adjacency (discrete but unique signal)
    "nearest_neighbor_distance", # gap width
    "skeleton_branches",         # complexity (discrete but unique signal)
    # --- Probability consensus (when available) ---
    "prob_mean",                 # overall confidence (represents prob_p10/std)
    "prob_boundary_drop",        # spatial pattern (unique signal)
    "prob_boundary_entropy",     # boundary uncertainty (unique signal)
]

# Named PCA presets for the Extract Features dialog.
# Each preset defines which features go into the initial auto-PCA and how
# they are standardized.  Users can customize from any preset as a starting
# point; the dialog tracks divergence as "Custom (based on …)".
PCA_PRESETS = {
    "EpiCycle QC": {
        "features": PCA_FEATURES,
        "mode": "qc",  # robust: median/MAD, winsorize
        "description": (
            "15 curated features \u2022 auto log\u2081\u209a for skewed features "
            "(|skew| > 1) \u2022 robust z-score (median / MAD) \u2022 "
            "winsorize \u00b110 \u2022 border cells excluded from scaling"
        ),
    },
    "Standard": {
        "features": None,  # all numeric — resolved at runtime
        "mode": "eda",  # standard: mean/std
        "description": (
            "All numeric features \u2022 auto log\u2081\u209a for known "
            "size/intensity features \u2022 standard z-score (mean / std)"
        ),
    },
    "None": {
        "features": [],
        "mode": None,
        "description": "Skip auto-PCA \u2014 extract features only, explore in EDA Workbench",
    },
}


def extract_features(
    labels_3d: np.ndarray,
    intensity_image: Optional[np.ndarray] = None,
    spacing: Optional[Tuple[float, ...]] = None,
) -> pd.DataFrame:
    """Extract morphological features from a 3D label volume.

    Computes ~20 features per label including shape, orientation, ellipsoid
    solidity, and optionally intensity statistics. Additional features
    (surface, boundary/interior, neighbor, skeleton, composite) are added
    by calling the standalone ``compute_*`` functions after this.

    Args:
        labels_3d: 3D integer label array (Z, Y, X). Background is 0.
        intensity_image: Optional 3D intensity image (same shape). When
            provided, intensity_mean/std/min/max are extracted via regionprops.
        spacing: Voxel spacing as (Z, Y, X) tuple in physical units (e.g. µm).
            When provided, area/volume, equivalent_diameter, centroids, and
            bounding boxes are reported in physical units. None = voxel units.

    Returns:
        DataFrame with one row per label.
    """
    from skimage.measure import regionprops_table

    if labels_3d.ndim != 3:
        raise ValueError(f"Expected 3D labels, got shape {labels_3d.shape}")

    if not np.any(labels_3d > 0):
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    props_list = list(_FAST_PROPERTIES)
    if intensity_image is not None:
        props_list += list(_INTENSITY_PROPERTIES)

    rp_kwargs: dict = dict(properties=tuple(props_list))
    if intensity_image is not None:
        rp_kwargs["intensity_image"] = intensity_image
    if spacing is not None:
        rp_kwargs["spacing"] = tuple(spacing)

    props = regionprops_table(
        labels_3d.astype(np.int32),
        **rp_kwargs,
    )
    df = pd.DataFrame(props)

    # Centroid → GRID (voxel-index) coords.  regionprops(spacing=) scales
    # area/volume AND the centroid into physical units, but the centroid is
    # ONLY a navigation pointer — it is excluded from PCA / qc_score, and the
    # EDA position plots already document it as "(voxels)".  Divide the spacing
    # back out so the stored centroid stays in voxel-index coords, independent
    # of voxel size — matching the navigator's ÷factor reads and the edit
    # write-backs (which store ×factor full-res voxel).  Area/volume stay
    # PHYSICAL; bbox-* columns are already raw voxel indices (only the local
    # bbox temporaries below get spacing), so centroid and bbox now share one
    # grid convention.  extract_single_label_features inherits this via its
    # extract_features(spacing=) call.  Guard finite-and-positive: the QC dialog
    # clamps spacing to [0.001, 100], but the auto-detect-from-layer-scale path
    # could supply 0 / NaN / inf — never produce a non-finite centroid the
    # navigator's int() would crash on.
    if spacing is not None:
        for _ax, _col in enumerate(("centroid-0", "centroid-1", "centroid-2")):
            if _col in df.columns:
                _s = float(spacing[_ax])
                if np.isfinite(_s) and _s > 0:
                    df[_col] = df[_col] / _s

    # ------------------------------------------------------------------
    # Derived features (all vectorized, very fast)
    # ------------------------------------------------------------------
    vol = df["area"].values.astype(np.float64)
    equiv_d = df["equivalent_diameter"].values.astype(np.float64)
    equiv_r = equiv_d / 2.0
    sphere_sa = 4.0 * np.pi * equiv_r ** 2

    bz = (df["bbox-3"] - df["bbox-0"]).values.astype(np.float64).clip(min=1)
    by = (df["bbox-4"] - df["bbox-1"]).values.astype(np.float64).clip(min=1)
    bx = (df["bbox-5"] - df["bbox-2"]).values.astype(np.float64).clip(min=1)
    # bbox coords are always in pixel indices — convert to physical units
    if spacing is not None:
        bz = bz * float(spacing[0])
        by = by * float(spacing[1])
        bx = bx * float(spacing[2])
    bbox_sa = 2.0 * (bz * by + by * bx + bz * bx)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["sphericity"] = np.where(bbox_sa > 0, sphere_sa / bbox_sa, 0.0)

    bbox_dims = np.stack([bz, by, bx], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["aspect_ratio"] = np.where(
            bbox_dims.min(axis=1) > 0,
            bbox_dims.max(axis=1) / bbox_dims.min(axis=1),
            0.0,
        )

    bbox_vol = bz * by * bx
    df["bbox_volume"] = bbox_vol

    with np.errstate(divide="ignore", invalid="ignore"):
        df["compactness"] = np.where(bbox_vol > 0, vol / bbox_vol, 0.0)

    # Z-span and elongation
    df["z_span"] = bz
    with np.errstate(divide="ignore", invalid="ignore"):
        xy_max = np.maximum(by, bx)
        xy_min = np.minimum(by, bx).clip(min=1)
        df["z_elongation"] = np.where(xy_max > 0, bz / xy_max, 0.0)
        df["xy_elongation"] = np.where(xy_min > 0, xy_max / xy_min, 0.0)

    # Fill ratio (area / area_filled; < 1 means internal holes)
    area_filled = df["area_filled"].values.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["fill_ratio"] = np.where(area_filled > 0, vol / area_filled, 1.0)

    # ------------------------------------------------------------------
    # Ellipsoid solidity from inertia tensor eigenvalues
    # ------------------------------------------------------------------
    eig_cols = [f"inertia_tensor_eigvals-{i}" for i in range(3)]
    if all(c in df.columns for c in eig_cols):
        # regionprops_table returns eigenvalues sorted DESCENDING: I0 >= I1 >= I2
        I0 = df[eig_cols[0]].values.astype(np.float64)
        I1 = df[eig_cols[1]].values.astype(np.float64)
        I2 = df[eig_cols[2]].values.astype(np.float64)

        # Semi-axis squared from eigenvalues of the inertia tensor.
        # skimage's eigenvalues are mass-normalized (divided by pixel count),
        # so NO division by M is needed here. For a uniform solid ellipsoid
        # with semi-axes a >= b >= c and descending eigenvalues I0 >= I1 >= I2:
        #   I0 = (a² + b²)/5,  I1 = (a² + c²)/5,  I2 = (b² + c²)/5
        # Solving:
        #   a² = 5/2 * (I0 + I1 - I2)   (largest semi-axis)
        #   b² = 5/2 * (I0 - I1 + I2)   (middle)
        #   c² = 5/2 * (-I0 + I1 + I2)  (smallest semi-axis)
        a_sq = np.clip(2.5 * (I0 + I1 - I2), 0, None)
        b_sq = np.clip(2.5 * (I0 - I1 + I2), 0, None)
        c_sq = np.clip(2.5 * (-I0 + I1 + I2), 0, None)

        ellipsoid_vol = (4.0 / 3.0) * np.pi * np.sqrt(a_sq * b_sq * c_sq)
        with np.errstate(divide="ignore", invalid="ignore"):
            df["ellipsoid_solidity"] = np.where(
                ellipsoid_vol > 0, vol / ellipsoid_vol, 0.0,
            )

        # Inertia-based elongation and flatness (direction-independent)
        with np.errstate(divide="ignore", invalid="ignore"):
            df["inertia_elongation"] = np.where(
                I2 > 0, np.sqrt(np.clip(I0 / I2, 0, None)), 0.0,
            )
            df["inertia_flatness"] = np.where(
                I0 > 0, np.sqrt(np.clip(I1 / I0, 0, None)), 0.0,
            )

        # Axis lengths from combined eigenvalues (not single-eigenvalue).
        # Major axis (longest) = 2a, Minor axis (shortest) = 2c.
        df["axis_major_length"] = 2.0 * np.sqrt(a_sq)
        df["axis_minor_length"] = 2.0 * np.sqrt(c_sq)
    else:
        df["ellipsoid_solidity"] = 0.0
        df["inertia_elongation"] = 0.0
        df["inertia_flatness"] = 0.0
        df["axis_major_length"] = 0.0
        df["axis_minor_length"] = 0.0

    # ------------------------------------------------------------------
    # Major axis Z-angle + orientation vectors from inertia tensor
    # ------------------------------------------------------------------
    tensor_cols = [f"inertia_tensor-{i}-{j}" for i in range(3) for j in range(3)]
    if all(c in df.columns for c in tensor_cols):
        n = len(df)
        tensor_data = df[tensor_cols].values.reshape(n, 3, 3).astype(np.float64)
        angles = np.zeros(n, dtype=np.float64)
        # Principal axis direction vectors (ZYX components)
        major_dir = np.zeros((n, 3), dtype=np.float64)
        mid_dir = np.zeros((n, 3), dtype=np.float64)
        minor_dir = np.zeros((n, 3), dtype=np.float64)
        for idx in range(n):
            try:
                eigvals, eigvecs = np.linalg.eigh(tensor_data[idx])
                # eigh sorts ascending; smallest eigenvalue = most elongated axis
                major_axis = eigvecs[:, 0]
                major_dir[idx] = major_axis
                mid_dir[idx] = eigvecs[:, 1]
                minor_dir[idx] = eigvecs[:, 2]
                # Z is the first dimension in ZYX ordering
                cos_angle = abs(major_axis[0])
                angles[idx] = np.degrees(np.arccos(np.clip(cos_angle, 0.0, 1.0)))
            except np.linalg.LinAlgError:
                angles[idx] = 0.0
        df["major_axis_z_angle"] = angles
        # Keep orientation vectors (principal axes in ZYX)
        for i, name in enumerate(["z", "y", "x"]):
            df[f"major_axis_{name}"] = major_dir[:, i]
            df[f"mid_axis_{name}"] = mid_dir[:, i]
            df[f"minor_axis_{name}"] = minor_dir[:, i]
    else:
        df["major_axis_z_angle"] = 0.0
        for prefix in ("major_axis", "mid_axis", "minor_axis"):
            for name in ("z", "y", "x"):
                df[f"{prefix}_{name}"] = 0.0

    # ------------------------------------------------------------------
    # Keep eigenvalues as named columns, drop raw tensor
    # ------------------------------------------------------------------
    eig_cols_raw = [f"inertia_tensor_eigvals-{i}" for i in range(3)]
    if all(c in df.columns for c in eig_cols_raw):
        df["inertia_eigval_major"] = df[eig_cols_raw[0]]
        df["inertia_eigval_mid"] = df[eig_cols_raw[1]]
        df["inertia_eigval_minor"] = df[eig_cols_raw[2]]

    drop_cols = [c for c in df.columns
                 if c.startswith("inertia_tensor_eigvals-")
                 or c.startswith("inertia_tensor-")]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)

    # Staleness tracking — freshly extracted features are "fresh".
    df["feature_state"] = "fresh"

    return df


def enrich_expensive_features(
    labels_3d: np.ndarray,
    df: pd.DataFrame,
    label_ids: Optional[Sequence[int]] = None,
    spacing: Optional[Tuple[float, ...]] = None,
) -> pd.DataFrame:
    """Add solidity and axis lengths for selected labels (slow — uses convex hull).

    Run this AFTER filtering out small artifacts to avoid qhull errors on
    degenerate labels and to limit computation to labels that matter.

    Args:
        labels_3d: Same 3D label volume used for extract_features.
        df: DataFrame from extract_features (modified in-place).
        label_ids: Subset of labels to compute for. None = all labels in df.
        spacing: Voxel spacing as (Z, Y, X) tuple. None = voxel units.

    Returns:
        The same DataFrame with solidity, axis_major_length, axis_minor_length added.
    """
    import warnings
    from skimage.measure import regionprops_table

    if label_ids is None:
        label_ids = df["label"].values.tolist()

    # Filter volume to only requested labels (speeds up regionprops)
    mask = np.isin(labels_3d, label_ids)
    filtered = np.where(mask, labels_3d, 0).astype(np.int32)

    rp_kwargs = dict(properties=("label",) + _EXPENSIVE_PROPERTIES)
    if spacing is not None:
        rp_kwargs["spacing"] = tuple(spacing)

    # Suppress qhull warnings for any remaining edge cases
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*convex hull.*")
        warnings.filterwarnings("ignore", message=".*QH.*")
        try:
            props = regionprops_table(
                filtered,
                **rp_kwargs,
            )
        except (ValueError, Exception):
            # If it still fails, fill solidity with zeros
            # (axis lengths are already computed from eigenvalues in extract_features)
            df["solidity"] = 0.0
            return df

    expensive_df = pd.DataFrame(props)

    # Merge solidity into the main DataFrame.
    # axis_major_length and axis_minor_length are computed from eigenvalues
    # in extract_features() — do NOT overwrite them here.
    if "solidity" in expensive_df.columns:
        mapping = dict(zip(expensive_df["label"], expensive_df["solidity"]))
        df["solidity"] = df["label"].map(mapping).fillna(0.0)

    return df


def compute_surface_features(
    labels_3d: np.ndarray,
    df: pd.DataFrame,
    spacing: Optional[Tuple[float, ...]] = None,
) -> pd.DataFrame:
    """Compute surface area, true sphericity, roughness, and fast solidity.

    Uses marching cubes for surface area and subsampled surface hull for
    solidity. Operates on per-label bounding box crops for efficiency.

    Args:
        labels_3d: 3D integer label array (Z, Y, X). Background is 0.
        df: DataFrame from extract_features (modified in-place).
        spacing: Voxel spacing as (Z, Y, X) tuple. None = voxel units.

    Returns:
        The same DataFrame with surface_area, true_sphericity, roughness,
        solidity columns added.
    """
    from scipy.ndimage import binary_erosion, find_objects
    from skimage.measure import marching_cubes, mesh_surface_area

    n = len(df)
    if n == 0:
        for col in ("surface_area", "true_sphericity", "roughness", "solidity"):
            df[col] = pd.Series(dtype=np.float64)
        return df

    sa_arr = np.zeros(n, dtype=np.float64)
    true_sph_arr = np.zeros(n, dtype=np.float64)
    rough_arr = np.zeros(n, dtype=np.float64)
    sol_arr = np.zeros(n, dtype=np.float64)

    sp = tuple(spacing) if spacing is not None else (1.0, 1.0, 1.0)

    slices = find_objects(labels_3d.astype(np.int32))
    label_ids = df["label"].values
    vol = df["area"].values.astype(np.float64)

    for idx, label_id in enumerate(label_ids):
        if label_id < 1 or label_id > len(slices):
            continue
        obj_slice = slices[label_id - 1]
        if obj_slice is None:
            continue

        crop = (labels_3d[obj_slice] == label_id).astype(np.uint8)

        # Pad with 1 voxel of background so marching cubes can find the surface
        crop = np.pad(crop, 1, mode="constant", constant_values=0)

        # Surface area via marching cubes
        try:
            verts, faces, _, _ = marching_cubes(crop, level=0.5, spacing=sp)
            sa = mesh_surface_area(verts, faces)
        except (RuntimeError, ValueError):
            sa = 0.0
        sa_arr[idx] = sa

        # True sphericity: (pi^(1/3) * (6V)^(2/3)) / SA
        v = vol[idx]
        if sa > 0 and v > 0:
            true_sph_arr[idx] = (
                (np.pi ** (1.0 / 3.0)) * ((6.0 * v) ** (2.0 / 3.0)) / sa
            )

        # Roughness: SA / SA_equivalent_sphere
        if v > 0:
            equiv_r = ((3.0 * v) / (4.0 * np.pi)) ** (1.0 / 3.0)
            equiv_sa = 4.0 * np.pi * equiv_r ** 2
            if equiv_sa > 0:
                rough_arr[idx] = sa / equiv_sa

        # Fast solidity via subsampled surface hull
        try:
            eroded = binary_erosion(crop)
            surface_voxels = crop.astype(bool) & ~eroded
            coords = np.argwhere(surface_voxels)
            if len(coords) < 4:
                # Too few surface voxels — use volume ratio fallback
                sol_arr[idx] = 1.0
                continue

            # Scale to physical coordinates
            coords_phys = coords.astype(np.float64) * np.array(sp)

            # Subsample to ~1500 points for speed
            max_pts = 1500
            if len(coords_phys) > max_pts:
                rng = np.random.RandomState(42)
                idx_sub = rng.choice(len(coords_phys), max_pts, replace=False)
                coords_phys = coords_phys[idx_sub]

            from scipy.spatial import ConvexHull
            hull = ConvexHull(coords_phys)
            if hull.volume > 0:
                sol_arr[idx] = v / hull.volume
            else:
                sol_arr[idx] = 1.0
        except Exception:
            sol_arr[idx] = 0.0

    df["surface_area"] = sa_arr
    df["true_sphericity"] = true_sph_arr
    df["roughness"] = rough_arr
    df["solidity"] = sol_arr
    return df


def _iter_label_boundary_interior(
    labels_3d: np.ndarray,
    label_ids: np.ndarray,
    erosion_iters: int = 2,
):
    """Yield (idx, mask_crop, boundary, interior, obj_slice) per label.

    Shared between compute_boundary_interior_features and
    compute_channel_intensity.  Uses find_objects + binary_erosion per-label
    crop.  Skips labels that are out of range or have no slice.
    """
    from scipy.ndimage import binary_erosion, find_objects

    slices = find_objects(labels_3d.astype(np.int32))
    for idx, label_id in enumerate(label_ids):
        if label_id < 1 or label_id > len(slices):
            continue
        obj_slice = slices[label_id - 1]
        if obj_slice is None:
            continue
        mask_crop = labels_3d[obj_slice] == label_id
        interior = binary_erosion(mask_crop, iterations=erosion_iters)
        boundary = mask_crop & ~interior
        yield idx, mask_crop, boundary, interior, obj_slice


def compute_boundary_interior_features(
    labels_3d: np.ndarray,
    intensity_image: np.ndarray,
    df: pd.DataFrame,
    erosion_iters: int = 2,
) -> pd.DataFrame:
    """Compute boundary vs interior intensity features for membrane QC.

    For membrane-GFP segmentation, correctly segmented cells have bright
    boundaries (membrane) and dark interiors. Merged cells show elevated
    interior intensity from the internal membrane.

    Args:
        labels_3d: 3D integer label array (Z, Y, X). Background is 0.
        intensity_image: 3D intensity image (same shape as labels_3d).
        df: DataFrame from extract_features (modified in-place).
        erosion_iters: Number of binary erosion iterations to define the
            interior. Default 2 creates a ~2-voxel boundary shell.

    Returns:
        The same DataFrame with boundary_intensity_mean, interior_intensity_mean,
        boundary_interior_ratio, and z_intensity_variation columns added.
    """
    n = len(df)
    cols = (
        "boundary_intensity_mean", "interior_intensity_mean",
        "boundary_interior_ratio", "z_intensity_variation",
    )
    if n == 0:
        for col in cols:
            df[col] = pd.Series(dtype=np.float64)
        return df

    bnd_mean = np.zeros(n, dtype=np.float64)
    int_mean = np.zeros(n, dtype=np.float64)
    ratio = np.zeros(n, dtype=np.float64)
    z_var = np.zeros(n, dtype=np.float64)

    label_ids = df["label"].values
    for idx, mask_crop, boundary, interior, obj_slice in _iter_label_boundary_interior(
        labels_3d, label_ids, erosion_iters
    ):
        int_crop = intensity_image[obj_slice]

        if boundary.any():
            bnd_mean[idx] = float(int_crop[boundary].mean())
        if interior.any():
            int_mean[idx] = float(int_crop[interior].mean())
        else:
            int_mean[idx] = float(int_crop[mask_crop].mean()) if mask_crop.any() else 0.0

        denom = int_mean[idx] if int_mean[idx] > 0 else 1e-6
        ratio[idx] = bnd_mean[idx] / denom

        z_start = obj_slice[0].start
        z_stop = obj_slice[0].stop
        slice_means = []
        for z in range(z_stop - z_start):
            z_mask = mask_crop[z]
            if z_mask.any():
                slice_means.append(float(int_crop[z][z_mask].mean()))
        if len(slice_means) > 1:
            z_var[idx] = float(np.std(slice_means))

    df["boundary_intensity_mean"] = bnd_mean
    df["interior_intensity_mean"] = int_mean
    df["boundary_interior_ratio"] = ratio
    df["z_intensity_variation"] = z_var
    return df


def compute_channel_intensity(
    labels_3d: np.ndarray,
    df: pd.DataFrame,
    channel_image: np.ndarray,
    prefix: str,
    erosion_iters: int = 2,
) -> pd.DataFrame:
    """Per-channel intensity features for each label.

    Adds 11 columns with the given prefix:
        {prefix}_intensity_mean, _median, _std, _min, _max, _sum, _cv
        {prefix}_boundary_intensity_mean, _interior_intensity_mean
        {prefix}_boundary_interior_ratio, _z_intensity_variation

    Args:
        labels_3d: 3D integer label array (Z, Y, X).
        df: DataFrame from extract_features (modified in-place).
        channel_image: 3D intensity image for this channel (same shape).
        prefix: Column name prefix (e.g. ``"nuclei"``, ``"gfp"``).
        erosion_iters: Erosion iterations for boundary/interior split.

    Returns:
        The same DataFrame with 11 new columns.
    """
    n = len(df)
    suffixes = (
        "_intensity_mean", "_intensity_median", "_intensity_std",
        "_intensity_min", "_intensity_max", "_intensity_sum",
        "_intensity_cv",
        "_boundary_intensity_mean", "_interior_intensity_mean",
        "_boundary_interior_ratio", "_z_intensity_variation",
    )
    if n == 0:
        for s in suffixes:
            df[f"{prefix}{s}"] = pd.Series(dtype=np.float64)
        return df

    i_mean = np.zeros(n, dtype=np.float64)
    i_median = np.zeros(n, dtype=np.float64)
    i_std = np.zeros(n, dtype=np.float64)
    i_min = np.zeros(n, dtype=np.float64)
    i_max = np.zeros(n, dtype=np.float64)
    i_sum = np.zeros(n, dtype=np.float64)
    i_cv = np.zeros(n, dtype=np.float64)
    b_mean = np.zeros(n, dtype=np.float64)
    t_mean = np.zeros(n, dtype=np.float64)
    bt_ratio = np.zeros(n, dtype=np.float64)
    z_var = np.zeros(n, dtype=np.float64)

    label_ids = df["label"].values
    for idx, mask_crop, boundary, interior, obj_slice in _iter_label_boundary_interior(
        labels_3d, label_ids, erosion_iters
    ):
        ch_crop = channel_image[obj_slice]
        vals = ch_crop[mask_crop]
        if vals.size == 0:
            continue

        i_mean[idx] = float(vals.mean())
        i_median[idx] = float(np.median(vals))
        i_std[idx] = float(vals.std())
        i_min[idx] = float(vals.min())
        i_max[idx] = float(vals.max())
        i_sum[idx] = float(vals.sum())
        i_cv[idx] = float(i_std[idx] / i_mean[idx]) if i_mean[idx] > 0 else 0.0

        if boundary.any():
            b_mean[idx] = float(ch_crop[boundary].mean())
        if interior.any():
            t_mean[idx] = float(ch_crop[interior].mean())
        else:
            t_mean[idx] = i_mean[idx]

        denom = t_mean[idx] if t_mean[idx] > 0 else 1e-6
        bt_ratio[idx] = b_mean[idx] / denom

        # Z-intensity variation: std of per-Z-slice means
        z_start = obj_slice[0].start
        z_stop = obj_slice[0].stop
        slice_means = []
        for z in range(z_stop - z_start):
            z_mask = mask_crop[z]
            if z_mask.any():
                slice_means.append(float(ch_crop[z][z_mask].mean()))
        if len(slice_means) > 1:
            z_var[idx] = float(np.std(slice_means))

    df[f"{prefix}_intensity_mean"] = i_mean
    df[f"{prefix}_intensity_median"] = i_median
    df[f"{prefix}_intensity_std"] = i_std
    df[f"{prefix}_intensity_min"] = i_min
    df[f"{prefix}_intensity_max"] = i_max
    df[f"{prefix}_intensity_sum"] = i_sum
    df[f"{prefix}_intensity_cv"] = i_cv
    df[f"{prefix}_boundary_intensity_mean"] = b_mean
    df[f"{prefix}_interior_intensity_mean"] = t_mean
    df[f"{prefix}_boundary_interior_ratio"] = bt_ratio
    df[f"{prefix}_z_intensity_variation"] = z_var
    return df


# ---------------------------------------------------------------------------
# Probability consensus features
# ---------------------------------------------------------------------------


def _expand_slice(
    obj_slice: tuple,
    padding: int,
    vol_shape: tuple,
) -> tuple:
    """Expand a ``find_objects`` slice tuple by *padding* voxels, clamped to volume bounds."""
    expanded = []
    for s, dim_size in zip(obj_slice, vol_shape):
        start = max(0, s.start - padding)
        stop = min(dim_size, s.stop + padding)
        expanded.append(slice(start, stop))
    return tuple(expanded)


def _binary_entropy(p: np.ndarray) -> np.ndarray:
    """Voxel-wise binary (Bernoulli) entropy: H = -p log p - (1-p) log(1-p).

    Clamps to avoid log(0).  Returns array same shape as *p*.
    """
    eps = 1e-12
    pc = np.clip(p, eps, 1.0 - eps)
    return -(pc * np.log(pc) + (1.0 - pc) * np.log(1.0 - pc))


def compute_probability_features(
    labels_3d: np.ndarray,
    df: pd.DataFrame,
    prob_3d: np.ndarray,
    bbox_padding: int = 3,
) -> pd.DataFrame:
    """Compute per-label statistics from the 3D probability volume.

    The probability volume is the multi-view consensus map produced by
    u-Segment3D during reconstruction (variance-weighted combination of XY,
    XZ, YZ Cellpose probabilities).

    Features extracted (12 total, 5 groups):

    **Confidence statistics:**
        prob_mean, prob_p10, prob_std

    **Boundary vs interior spatial pattern:**
        prob_boundary_mean, prob_interior_mean, prob_boundary_drop

    **Probability mass vs mask size:**
        prob_weighted_volume (V_soft over padded bbox ROI),
        prob_soft_hard_ratio (V_soft / |M|)

    **Uncertainty / entropy (voxel-wise binary entropy):**
        prob_boundary_entropy, prob_mask_entropy

    **Edge sharpness (gradient magnitude at boundary):**
        prob_edge_gradient, prob_edge_gradient_std

    Parameters
    ----------
    labels_3d : np.ndarray
        3D integer label array (Z, Y, X). Background is 0.
    df : pd.DataFrame
        DataFrame from ``extract_features`` (modified in-place).
    prob_3d : np.ndarray
        3D float probability volume, same shape as *labels_3d*,
        range [0, 1]. Typically loaded from ``*_probabilities.npy``.
    bbox_padding : int
        Voxels to pad around each label's bounding box for the soft
        volume ratio computation (default 3).

    Returns
    -------
    pd.DataFrame
        The same DataFrame with 12 probability feature columns added.
    """
    from scipy.ndimage import binary_erosion, find_objects

    _PROB_COLS = (
        "prob_mean", "prob_p10", "prob_std",
        "prob_boundary_mean", "prob_interior_mean", "prob_boundary_drop",
        "prob_weighted_volume", "prob_soft_hard_ratio",
        "prob_boundary_entropy", "prob_mask_entropy",
        "prob_edge_gradient", "prob_edge_gradient_std",
    )

    n = len(df)
    if n == 0:
        for col in _PROB_COLS:
            df[col] = pd.Series(dtype=np.float64)
        return df

    # Pre-allocate output arrays
    out = {col: np.zeros(n, dtype=np.float64) for col in _PROB_COLS}

    # Cast float16 → float32 once (float16 CPU arithmetic is very slow)
    prob = prob_3d.astype(np.float32) if prob_3d.dtype == np.float16 else prob_3d.astype(np.float32)

    # Suppress expected numpy warnings from 0/1 probability regions
    with np.errstate(divide="ignore", invalid="ignore"):
        # Pre-compute gradient magnitude for edge sharpness (one-time, full volume)
        gx, gy, gz = np.gradient(prob)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
        del gx, gy, gz

        # Pre-compute voxel-wise binary entropy (one-time, full volume)
        H_vol = _binary_entropy(prob)

    slices = find_objects(labels_3d.astype(np.int32))
    label_ids = df["label"].values

    for idx, label_id in enumerate(label_ids):
        lid = int(label_id)
        if lid < 1 or lid > len(slices):
            continue
        obj_slice = slices[lid - 1]
        if obj_slice is None:
            continue

        mask_crop = labels_3d[obj_slice] == lid
        prob_crop = prob[obj_slice]
        vals = prob_crop[mask_crop]
        if vals.size == 0:
            continue

        # --- Group 1: confidence statistics ---
        out["prob_mean"][idx] = float(vals.mean())
        out["prob_p10"][idx] = float(np.percentile(vals, 10))
        out["prob_std"][idx] = float(vals.std())

        # --- Group 2: boundary vs interior ---
        eroded = binary_erosion(mask_crop)
        boundary = mask_crop & ~eroded
        interior = eroded

        if boundary.any():
            bnd_vals = prob_crop[boundary]
            out["prob_boundary_mean"][idx] = float(bnd_vals.mean())
        else:
            # Tiny label: erosion eliminates mask — use full mask
            out["prob_boundary_mean"][idx] = out["prob_mean"][idx]

        if interior.any():
            int_vals = prob_crop[interior]
            out["prob_interior_mean"][idx] = float(int_vals.mean())
        else:
            out["prob_interior_mean"][idx] = out["prob_mean"][idx]

        out["prob_boundary_drop"][idx] = (
            out["prob_interior_mean"][idx] - out["prob_boundary_mean"][idx]
        )

        # --- Group 3: probability mass vs mask size ---
        padded = _expand_slice(obj_slice, bbox_padding, prob.shape)
        roi_prob = prob[padded]
        v_soft = float(roi_prob.sum())
        v_hard = int(mask_crop.sum())
        out["prob_weighted_volume"][idx] = v_soft
        out["prob_soft_hard_ratio"][idx] = v_soft / max(v_hard, 1)

        # --- Group 4: voxel-wise binary entropy ---
        H_crop = H_vol[obj_slice]
        out["prob_mask_entropy"][idx] = float(H_crop[mask_crop].mean())
        if boundary.any():
            out["prob_boundary_entropy"][idx] = float(H_crop[boundary].mean())
        else:
            out["prob_boundary_entropy"][idx] = out["prob_mask_entropy"][idx]

        # --- Group 5: edge gradient ---
        grad_crop = grad_mag[obj_slice]
        if boundary.any():
            bnd_grad = grad_crop[boundary]
            out["prob_edge_gradient"][idx] = float(bnd_grad.mean())
            out["prob_edge_gradient_std"][idx] = float(bnd_grad.std())

    # Assign columns to DataFrame
    for col in _PROB_COLS:
        df[col] = out[col]

    return df


def compute_discrete_compactness(
    labels_3d: np.ndarray,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Bribiesca's discrete compactness per label.

    Reference: Bribiesca, E. (2008) "An easy measure of compactness for 2D
    and 3D shapes." Pattern Recognition 41(2): 543-554.

    Formula:
        C_d = (n - A_v / 6) / (n - n^(2/3))
    where:
        n   = voxel count of the object (the cell)
        A_v = total external face count (voxel faces NOT shared with another
              voxel of the same label)

    Bounded [0, 1]. 1 = perfect cube. Sphere ≈ 0.85. Thin rod = low value.

    This is genuinely orthogonal to ``true_sphericity`` (which is computed
    from a marching-cubes smooth surface). Discrete compactness uses
    voxel face counts directly, so it captures voxel-scale shape complexity
    that smooth-surface measures wash out — useful as a soft segmentation
    quality signal AND as a real shape descriptor.

    NOT a replacement for the legacy ``compactness`` column (which is
    misnamed and equals ``extent`` = V / V_bbox). The legacy column is
    kept for backward compatibility with existing CSVs.

    Args:
        labels_3d: 3D integer label array (Z, Y, X). Background is 0.
        df: DataFrame from extract_features (modified in-place).

    Returns:
        The same DataFrame with the ``discrete_compactness`` column added.
    """
    from scipy.ndimage import find_objects

    n = len(df)
    if n == 0:
        df["discrete_compactness"] = pd.Series(dtype=np.float64)
        return df

    cd = np.full(n, np.nan, dtype=np.float64)
    slices = find_objects(labels_3d.astype(np.int32))
    label_ids = df["label"].values

    for idx, label_id in enumerate(label_ids):
        if label_id < 1 or label_id > len(slices) or slices[label_id - 1] is None:
            continue
        zs, ys, xs = slices[label_id - 1]
        crop = labels_3d[zs, ys, xs] == label_id
        n_vox = int(crop.sum())
        if n_vox <= 1:
            cd[idx] = 0.0
            continue
        # Pad with False so faces at the crop boundary count as external
        padded = np.pad(crop, 1, mode="constant", constant_values=False)
        center = padded[1:-1, 1:-1, 1:-1]
        # External face count = faces of a True voxel where the neighbor is False
        a_v = 0
        a_v += int((center & ~padded[:-2, 1:-1, 1:-1]).sum())   # -Z neighbor
        a_v += int((center & ~padded[2:, 1:-1, 1:-1]).sum())    # +Z
        a_v += int((center & ~padded[1:-1, :-2, 1:-1]).sum())   # -Y
        a_v += int((center & ~padded[1:-1, 2:, 1:-1]).sum())    # +Y
        a_v += int((center & ~padded[1:-1, 1:-1, :-2]).sum())   # -X
        a_v += int((center & ~padded[1:-1, 1:-1, 2:]).sum())    # +X

        denom = n_vox - n_vox ** (2.0 / 3)
        if denom <= 0:
            cd[idx] = 0.0
        else:
            cd[idx] = (n_vox - a_v / 6.0) / denom

    df["discrete_compactness"] = cd
    return df


def compute_neighbor_features(
    labels_3d: np.ndarray,
    df: pd.DataFrame,
    max_radius: int = 5,
) -> pd.DataFrame:
    """Compute neighbor/adjacency features for each label via graduated dilation.

    Uses incremental dilation from radius 1 to *max_radius*, discovering
    neighbors at each step.  This adapts to any gap width between labels —
    critical for non-touching segmentations (e.g. 3D reconstructions from
    Cellpose) where inter-cell gaps of 2-5 voxels are typical.

    The dilation radius at which the first neighbor is discovered is recorded
    as ``nearest_neighbor_distance`` — an approximate gap width in voxels.

    Args:
        labels_3d: 3D integer label array (Z, Y, X). Background is 0.
        df: DataFrame from extract_features (modified in-place).
        max_radius: Maximum dilation radius (in voxels) to search for
            neighbors. Default 5 covers typical reconstruction gaps.

    Returns:
        The same DataFrame with 10 columns added:
        neighbor_count, nearest_neighbor_distance, max_contact_fraction,
        background_fraction (original 4), plus birth-radius filtration
        features: nn_count_r1, nn_count_r3, nn_birth_median, nn_birth_std,
        nn_accumulation_rate, nn_accumulation_knee (6 new).
    """
    from scipy.ndimage import binary_dilation, find_objects

    n = len(df)
    cols = (
        "neighbor_count", "nearest_neighbor_distance",
        "max_contact_fraction", "background_fraction",
    )
    filtration_cols = (
        "nn_count_r1", "nn_count_r3", "nn_birth_median",
        "nn_birth_std", "nn_accumulation_rate", "nn_accumulation_knee",
    )
    quality_cols = (
        "mean_neighbor_size_ratio", "neighbor_size_cv",
        "contact_weighted_neighbor_size_ratio",
    )
    topology_cols = (
        "basal_neighbor_count", "apical_neighbor_count",
        "neighbor_z_turnover", "neighbor_persistence_ratio",
    )
    if n == 0:
        for col in cols + filtration_cols + quality_cols + topology_cols:
            df[col] = pd.Series(dtype=np.float64)
        return df

    n_count = np.zeros(n, dtype=np.float64)
    nn_dist = np.full(n, float(max_radius + 1), dtype=np.float64)
    max_cf = np.zeros(n, dtype=np.float64)
    bg_frac = np.zeros(n, dtype=np.float64)
    # Birth-radius filtration features (Group E)
    nn_count_r1 = np.zeros(n, dtype=np.float64)
    nn_count_r3 = np.zeros(n, dtype=np.float64)
    nn_birth_median = np.zeros(n, dtype=np.float64)
    nn_birth_std = np.zeros(n, dtype=np.float64)
    nn_accum_rate = np.zeros(n, dtype=np.float64)
    nn_accum_knee = np.zeros(n, dtype=np.float64)
    # Neighbor quality features (Group B) — store maps during dilation
    mean_nb_size_ratio = np.zeros(n, dtype=np.float64)
    nb_size_cv = np.zeros(n, dtype=np.float64)
    contact_wt_ratio = np.zeros(n, dtype=np.float64)
    per_label_neighbor_maps: list = [None] * n   # {neighbor_id: contact_count}
    per_label_z_contacts: list = [None] * n     # {neighbor_id: set of Z indices}

    slices = find_objects(labels_3d.astype(np.int32))
    label_ids = df["label"].values

    for idx, label_id in enumerate(label_ids):
        if label_id < 1 or label_id > len(slices):
            continue
        obj_slice = slices[label_id - 1]
        if obj_slice is None:
            continue

        # Expand the crop by max_radius in each direction
        expanded = tuple(
            slice(max(s.start - max_radius, 0),
                  min(s.stop + max_radius, dim))
            for s, dim in zip(obj_slice, labels_3d.shape)
        )
        label_crop = labels_3d[expanded]
        mask_crop = label_crop == label_id

        # Graduated dilation: track which neighbors appear at each radius
        all_neighbors: dict = {}  # neighbor_id -> total contact voxels
        first_radius: dict = {}   # neighbor_id -> radius first seen
        neighbor_z_contacts: dict = {}  # neighbor_id -> set of Z indices (Group F)
        total_bg = 0
        total_shell = 0
        prev_dilated = mask_crop.copy()

        # Z offset for mapping crop coords back to cell's local Z range
        cell_z_start = expanded[0].start

        for r in range(1, max_radius + 1):
            dilated = binary_dilation(prev_dilated)
            # New ring = voxels added by this dilation step only
            new_ring = dilated & ~prev_dilated
            ring_ids = label_crop[new_ring]

            if len(ring_ids) > 0:
                # Get Z coordinates of new ring voxels for Group F
                ring_z_coords = np.where(new_ring)[0]  # Z indices in crop space
                unique, counts = np.unique(ring_ids, return_counts=True)

                # Per-neighbor Z tracking needs per-voxel iteration
                # only for non-background, non-self neighbors
                for uid, cnt in zip(unique, counts):
                    uid_int = int(uid)
                    if uid_int == int(label_id):
                        continue  # skip self
                    if uid_int == 0:
                        total_bg += int(cnt)
                    else:
                        all_neighbors[uid_int] = (
                            all_neighbors.get(uid_int, 0) + int(cnt)
                        )
                        if uid_int not in first_radius:
                            first_radius[uid_int] = r
                        # Track Z coordinates of contact with this neighbor
                        if uid_int not in neighbor_z_contacts:
                            neighbor_z_contacts[uid_int] = set()
                        nb_mask = ring_ids == uid_int
                        nb_z = ring_z_coords[nb_mask]
                        neighbor_z_contacts[uid_int].update(
                            int(z) + cell_z_start for z in nb_z
                        )
                    total_shell += int(cnt)

            prev_dilated = dilated

        # Store neighbor map for Group B + Z contacts for Group F
        per_label_neighbor_maps[idx] = dict(all_neighbors)
        per_label_z_contacts[idx] = dict(neighbor_z_contacts)

        # Populate results
        n_count[idx] = float(len(all_neighbors))

        if first_radius:
            nn_dist[idx] = float(min(first_radius.values()))
        elif n_count[idx] == 0:
            nn_dist[idx] = float(max_radius + 1)

        if total_shell > 0:
            bg_frac[idx] = float(total_bg) / total_shell
            if all_neighbors:
                max_contact = max(all_neighbors.values())
                max_cf[idx] = float(max_contact) / total_shell

        # Birth-radius filtration features
        if first_radius:
            birth_radii = np.array(list(first_radius.values()), dtype=np.float64)
            nn_count_r1[idx] = float(np.sum(birth_radii == 1))
            nn_count_r3[idx] = float(np.sum(birth_radii <= 3))
            nn_birth_median[idx] = float(np.median(birth_radii))
            nn_birth_std[idx] = float(np.std(birth_radii)) if len(birth_radii) > 1 else 0.0

            # Accumulation curve: cumulative neighbor count vs radius
            total_n = len(birth_radii)
            cumulative = np.array([
                float(np.sum(birth_radii <= r))
                for r in range(1, max_radius + 1)
            ])
            # Accumulation rate: slope of linear fit
            radii = np.arange(1, max_radius + 1, dtype=np.float64)
            if len(radii) >= 2:
                coeffs = np.polyfit(radii, cumulative, 1)
                nn_accum_rate[idx] = coeffs[0]

            # Accumulation knee: radius at which 80% of neighbors found
            threshold_80 = 0.8 * total_n
            knee_indices = np.where(cumulative >= threshold_80)[0]
            if len(knee_indices) > 0:
                nn_accum_knee[idx] = float(knee_indices[0] + 1)  # +1 for 1-based radius
            else:
                nn_accum_knee[idx] = float(max_radius)

    # Neighbor quality features (Group B) — compute from stored maps
    if "area" in df.columns:
        area_lookup = dict(zip(
            df["label"].values.astype(int),
            df["area"].values.astype(np.float64),
        ))
        eps = 1e-6

        for idx, label_id in enumerate(label_ids):
            nb_map = per_label_neighbor_maps[idx]
            if not nb_map:
                continue
            my_vol = area_lookup.get(int(label_id), 0.0)
            if my_vol < eps:
                continue

            nb_ids = list(nb_map.keys())
            nb_vols = np.array([
                area_lookup.get(nid, 0.0) for nid in nb_ids
            ], dtype=np.float64)

            # Filter out neighbors not in the DataFrame (already deleted etc)
            valid = nb_vols > 0
            if not valid.any():
                continue
            nb_vols_valid = nb_vols[valid]
            nb_contacts_valid = np.array([
                nb_map[nb_ids[i]] for i in range(len(nb_ids)) if valid[i]
            ], dtype=np.float64)

            # mean_neighbor_size_ratio: my vol / mean neighbor vol
            mean_nb_vol = nb_vols_valid.mean()
            mean_nb_size_ratio[idx] = my_vol / max(mean_nb_vol, eps)

            # neighbor_size_cv: CV of neighbor volumes
            if len(nb_vols_valid) > 1:
                nb_size_cv[idx] = float(
                    nb_vols_valid.std() / max(nb_vols_valid.mean(), eps)
                )

            # contact_weighted_neighbor_size_ratio
            total_contact = nb_contacts_valid.sum()
            if total_contact > 0:
                weighted_mean = (nb_contacts_valid * nb_vols_valid).sum() / total_contact
                contact_wt_ratio[idx] = my_vol / max(weighted_mean, eps)

    # Z-Stratified Neighbor Topology (Group F)
    basal_nb_count = np.zeros(n, dtype=np.float64)
    apical_nb_count = np.zeros(n, dtype=np.float64)
    nb_z_turnover = np.zeros(n, dtype=np.float64)
    nb_persistence = np.zeros(n, dtype=np.float64)

    for idx, label_id in enumerate(label_ids):
        if label_id < 1 or label_id > len(slices):
            continue
        obj_slice = slices[label_id - 1]
        if obj_slice is None:
            continue
        z_contacts = per_label_z_contacts[idx]
        if not z_contacts:
            continue

        # Cell's Z range
        cell_z_min = obj_slice[0].start
        cell_z_max = obj_slice[0].stop - 1
        cell_z_span = cell_z_max - cell_z_min

        # Need at least 5 Z-slices for meaningful stratification
        if cell_z_span < 4:
            continue

        third = max(cell_z_span / 3.0, 1.0)
        basal_z_end = cell_z_min + third
        apical_z_start = cell_z_max - third

        basal_set = set()
        apical_set = set()

        for nb_id, z_set in z_contacts.items():
            in_basal = any(z <= basal_z_end for z in z_set)
            in_apical = any(z >= apical_z_start for z in z_set)
            if in_basal:
                basal_set.add(nb_id)
            if in_apical:
                apical_set.add(nb_id)

        basal_nb_count[idx] = float(len(basal_set))
        apical_nb_count[idx] = float(len(apical_set))

        # Turnover: fraction that change from bottom to top
        union = basal_set | apical_set
        if union:
            intersection = basal_set & apical_set
            nb_z_turnover[idx] = 1.0 - len(intersection) / len(union)

        # Persistence: fraction spanning full height (in both zones)
        total_neighbors = len(z_contacts)
        if total_neighbors > 0:
            persistent = basal_set & apical_set
            nb_persistence[idx] = len(persistent) / total_neighbors

    df["neighbor_count"] = n_count
    df["nearest_neighbor_distance"] = nn_dist
    df["max_contact_fraction"] = max_cf
    df["background_fraction"] = bg_frac
    df["nn_count_r1"] = nn_count_r1
    df["nn_count_r3"] = nn_count_r3
    df["nn_birth_median"] = nn_birth_median
    df["nn_birth_std"] = nn_birth_std
    df["nn_accumulation_rate"] = nn_accum_rate
    df["nn_accumulation_knee"] = nn_accum_knee
    df["mean_neighbor_size_ratio"] = mean_nb_size_ratio
    df["neighbor_size_cv"] = nb_size_cv
    df["contact_weighted_neighbor_size_ratio"] = contact_wt_ratio
    df["basal_neighbor_count"] = basal_nb_count
    df["apical_neighbor_count"] = apical_nb_count
    df["neighbor_z_turnover"] = nb_z_turnover
    df["neighbor_persistence_ratio"] = nb_persistence

    # Store adjacency data as DataFrame attribute for optional export.
    # This is NOT persisted in the CSV — must be saved separately.
    adjacency_rows = []
    for idx in range(n):
        label_id = int(label_ids[idx])
        nb_map = per_label_neighbor_maps[idx]
        if nb_map:
            for nb_id, contact in nb_map.items():
                if label_id < nb_id:  # avoid duplicate edges
                    adjacency_rows.append({
                        "label_a": label_id,
                        "label_b": nb_id,
                        "contact_voxels": contact,
                    })
    df.attrs["_adjacency_list"] = adjacency_rows

    return df


def compute_skeleton_features(
    labels_3d: np.ndarray,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute skeleton topology features for each label.

    Uses distance-transform-based medial axis extraction (robust in 3D,
    unlike skimage.morphology.skeletonize which can produce empty results).
    Branch points and total length indicate shape complexity — merged
    cells (dumbbell shapes) have more branches than single cells.

    Args:
        labels_3d: 3D integer label array (Z, Y, X). Background is 0.
        df: DataFrame from extract_features (modified in-place).

    Returns:
        The same DataFrame with skeleton_branches, skeleton_nodes,
        skeleton_length columns added.
    """
    from scipy.ndimage import (
        convolve,
        distance_transform_edt,
        find_objects,
        maximum_filter,
    )

    n = len(df)
    cols = ("skeleton_branches", "skeleton_nodes", "skeleton_length")
    if n == 0:
        for col in cols:
            df[col] = pd.Series(dtype=np.float64)
        return df

    branches = np.zeros(n, dtype=np.float64)
    nodes = np.zeros(n, dtype=np.float64)
    length = np.zeros(n, dtype=np.float64)

    # 26-connectivity structure for 3D
    struct_26 = np.ones((3, 3, 3), dtype=bool)

    slices = find_objects(labels_3d.astype(np.int32))
    label_ids = df["label"].values

    for idx, label_id in enumerate(label_ids):
        if label_id < 1 or label_id > len(slices):
            continue
        obj_slice = slices[label_id - 1]
        if obj_slice is None:
            continue

        mask_crop = (labels_3d[obj_slice] == label_id)

        # Pad with 1 voxel of background so EDT measures distance to true boundary
        mask_padded = np.pad(mask_crop, 1, mode="constant", constant_values=False)

        # EDT-based medial axis: local maxima of distance transform
        dt_padded = distance_transform_edt(mask_padded)
        # Strip padding back to original crop dimensions
        dt = dt_padded[1:-1, 1:-1, 1:-1]
        skel = (
            (dt == maximum_filter(dt, size=3)) & mask_crop & (dt > 1)
        ).astype(np.uint8)

        skel_voxels = int(skel.sum())
        length[idx] = float(skel_voxels)

        if skel_voxels < 3:
            branches[idx] = 1.0 if skel_voxels > 0 else 0.0
            continue

        # Count branch points: skeleton voxels with >2 neighbors in 26-connectivity
        neighbor_count_vol = convolve(skel, struct_26, mode="constant", cval=0)
        # Subtract the center voxel itself from the count
        neighbor_count_vol = neighbor_count_vol - skel
        # Branch points: skeleton voxels with >2 skeleton neighbors
        branch_pts = skel & (neighbor_count_vol > 2)
        n_nodes = int(branch_pts.sum())
        nodes[idx] = float(n_nodes)
        branches[idx] = float(n_nodes + 1)

    df["skeleton_branches"] = branches
    df["skeleton_nodes"] = nodes
    df["skeleton_length"] = length
    return df


def compute_cross_section_features(
    labels_3d: np.ndarray,
    df: pd.DataFrame,
    invert_z: bool = False,
) -> pd.DataFrame:
    """Z-stratified cross-section features for epithelial cell polarity.

    Analyzes how the cross-section area changes along each cell's Z-extent
    to capture apical-basal polarity. Cheap to compute — iterates Z-slices,
    counts voxels, fits a line.

    Args:
        invert_z: If True, treat high Z as basal and low Z as apical
            (for inverted stacks). Default False = low Z is basal.

    Features added (6):
        ``apical_area`` — mean cross-section in top 20% of cell Z-range
        ``basal_area`` — mean cross-section in bottom 20% of cell Z-range
        ``apical_basal_area_ratio`` — apical / basal (< 0.5 = pyramidal,
            ~1.0 = columnar, > 2.0 = inverted)
        ``max_cross_section_z_relative`` — Z-position of widest slice,
            normalized to [0, 1] (0 = bottom, 1 = top)
        ``cross_section_taper`` — slope of linear fit of area vs Z
            (negative = tapering upward, positive = expanding)
        ``cross_section_bulge`` — RMS residual of linear fit, normalized
            by mean area (high = vase/barrel/dumbbell shape)

    Args:
        labels_3d: 3D integer label array (Z, Y, X).
        df: DataFrame from extract_features (modified in-place).

    Returns:
        The same DataFrame with 6 new columns.
    """
    from scipy.ndimage import find_objects

    n = len(df)
    cols = (
        "apical_area", "basal_area", "apical_basal_area_ratio",
        "max_cross_section_z_relative", "cross_section_taper",
        "cross_section_bulge",
    )
    if n == 0:
        for col in cols:
            df[col] = pd.Series(dtype=np.float64)
        return df

    apical = np.zeros(n, dtype=np.float64)
    basal = np.zeros(n, dtype=np.float64)
    ratio = np.zeros(n, dtype=np.float64)
    max_z_rel = np.zeros(n, dtype=np.float64)
    taper = np.zeros(n, dtype=np.float64)
    bulge = np.zeros(n, dtype=np.float64)

    slices = find_objects(labels_3d.astype(np.int32))
    label_ids = df["label"].values

    for idx, label_id in enumerate(label_ids):
        if label_id < 1 or label_id > len(slices):
            continue
        obj_slice = slices[label_id - 1]
        if obj_slice is None:
            continue

        crop = labels_3d[obj_slice]
        mask = crop == label_id
        z_size = mask.shape[0]

        if z_size < 2:
            # Single slice — can't compute Z-stratified features
            total_area = float(mask.sum())
            apical[idx] = total_area
            basal[idx] = total_area
            ratio[idx] = 1.0
            max_z_rel[idx] = 0.5
            continue

        # Per-Z-slice areas
        areas = np.array([
            float(mask[z].sum()) for z in range(z_size)
        ])

        # Filter to slices where the label is present
        present = areas > 0
        if not present.any():
            continue

        # Basal = bottom 20%, Apical = top 20%
        n_slices = int(present.sum())
        margin = max(int(n_slices * 0.2), 1)

        # Find first and last present Z indices
        z_indices = np.where(present)[0]
        z_min, z_max = z_indices[0], z_indices[-1]
        z_range = z_max - z_min

        if z_range == 0:
            # All on one slice
            apical[idx] = areas[z_min]
            basal[idx] = areas[z_min]
            ratio[idx] = 1.0
            max_z_rel[idx] = 0.5
            continue

        basal_end = z_min + max(int(z_range * 0.2), 1)
        apical_start = z_max - max(int(z_range * 0.2), 1)

        if invert_z:
            # Inverted: high Z = basal, low Z = apical
            basal_areas = areas[apical_start:z_max + 1]
            apical_areas = areas[z_min:basal_end + 1]
        else:
            # Default: low Z = basal, high Z = apical
            basal_areas = areas[z_min:basal_end + 1]
            apical_areas = areas[apical_start:z_max + 1]

        basal_mean = float(basal_areas[basal_areas > 0].mean()) if (basal_areas > 0).any() else 0.0
        apical_mean = float(apical_areas[apical_areas > 0].mean()) if (apical_areas > 0).any() else 0.0

        basal[idx] = basal_mean
        apical[idx] = apical_mean
        eps = 1e-6
        ratio[idx] = apical_mean / max(basal_mean, eps)

        # Max cross-section Z position (relative)
        max_z = int(np.argmax(areas))
        max_z_rel[idx] = (max_z - z_min) / max(z_range, 1)

        # Linear fit: area vs normalized Z
        present_areas = areas[z_indices]
        z_norm = (z_indices - z_min).astype(np.float64) / max(z_range, 1)

        if len(z_norm) >= 2:
            # Least squares: area = slope * z_norm + intercept
            coeffs = np.polyfit(z_norm, present_areas, 1)
            slope = coeffs[0]
            taper[idx] = -slope if invert_z else slope

            # Bulge: RMS residual normalized by mean area
            fitted = np.polyval(coeffs, z_norm)
            residuals = present_areas - fitted
            rms = float(np.sqrt(np.mean(residuals ** 2)))
            mean_area = float(present_areas.mean())
            bulge[idx] = rms / max(mean_area, eps)

    df["apical_area"] = apical
    df["basal_area"] = basal
    df["apical_basal_area_ratio"] = ratio
    df["max_cross_section_z_relative"] = max_z_rel
    df["cross_section_taper"] = taper
    df["cross_section_bulge"] = bulge
    return df


def compute_border_cells(
    labels_3d: np.ndarray,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Detect labels touching the volume boundary.

    Adds an ``is_border_cell`` column (1 = touches boundary, 0 = interior).
    Border cells have truncated morphology and should be flagged or excluded
    from downstream analysis.

    A label is a border cell if any of its voxels lie on the volume faces:
    Z=0, Z=max, Y=0, Y=max, X=0, X=max.

    Args:
        labels_3d: 3D integer label array (Z, Y, X).
        df: DataFrame from extract_features (modified in-place).

    Returns:
        The same DataFrame with ``is_border_cell`` column added.
    """
    n = len(df)
    if n == 0:
        df["is_border_cell"] = pd.Series(dtype=np.int32)
        return df

    label_ids = df["label"].values
    is_border = np.zeros(n, dtype=np.int32)

    # Collect all label IDs that appear on any boundary face
    Z, Y, X = labels_3d.shape
    boundary_labels = set()
    # Z faces
    boundary_labels.update(np.unique(labels_3d[0, :, :]))
    boundary_labels.update(np.unique(labels_3d[Z - 1, :, :]))
    # Y faces
    boundary_labels.update(np.unique(labels_3d[:, 0, :]))
    boundary_labels.update(np.unique(labels_3d[:, Y - 1, :]))
    # X faces
    boundary_labels.update(np.unique(labels_3d[:, :, 0]))
    boundary_labels.update(np.unique(labels_3d[:, :, X - 1]))
    boundary_labels.discard(0)  # background is not a cell

    for idx, lid in enumerate(label_ids):
        if lid in boundary_labels:
            is_border[idx] = 1

    df["is_border_cell"] = is_border
    return df


def compute_named_composites(df: pd.DataFrame) -> pd.DataFrame:
    """Compute named composite signals for specific failure modes.

    Built from existing feature primitives using weighted robust z-scores.
    Gives biologists actionable categories rather than one opaque qc_score.

    Features added:
        ``merge_signal`` — how merge-like is this cell? High = suspicious.
        ``fragment_signal`` — how fragment-like? High = suspicious.
        ``boundary_confidence`` — how confident is the boundary? High = good.
            (Only computed when probability features are available.)
        ``shape_abnormality`` — how unusual is the shape? High = outlier.

    Args:
        df: DataFrame with prerequisite feature columns (modified in-place).

    Returns:
        The same DataFrame with composite columns added.
    """
    n = len(df)
    if n == 0:
        for col in ("merge_signal", "fragment_signal",
                     "boundary_confidence", "shape_abnormality"):
            df[col] = pd.Series(dtype=np.float64)
        return df

    def _robust_zscore(values):
        """Median/MAD z-score for a single column."""
        v = values.copy().astype(np.float64)
        finite = np.isfinite(v)
        if not finite.any() or finite.sum() < 2:
            return np.zeros_like(v)
        med = np.median(v[finite])
        mad = np.median(np.abs(v[finite] - med))
        scale = mad * 1.4826
        if scale < 1e-12:
            return np.zeros_like(v)
        z = (v - med) / scale
        return np.clip(z, -10, 10)

    def _weighted_signal(feature_specs):
        """Directional composite: only scores in the suspicious direction.

        feature_specs: list of (column_name, weight) tuples.
        weight > 0: high value = suspicious (positive z counts).
        weight < 0: low value = suspicious (negative z counts).

        Uses max(z * sign(weight), 0) so that only the suspicious
        direction contributes.  A cell with boundary_interior_ratio far
        BELOW the median (merge-like) scores high for weight=-1, but a
        cell far ABOVE the median (healthy) scores zero.
        """
        available = [
            (col, w) for col, w in feature_specs if col in df.columns
        ]
        if not available:
            return np.full(n, np.nan, dtype=np.float64)
        z_sum = np.zeros(n, dtype=np.float64)
        count = 0
        for col, weight in available:
            z = _robust_zscore(df[col].values)
            # Directional: only the suspicious side contributes
            z_sum += np.maximum(z * np.sign(weight), 0.0)
            count += 1
        return z_sum / max(count, 1)

    # Resolve boundary_interior_ratio: Step 3 uses the unprefixed version,
    # Step 4 uses per-channel prefixed versions.  Find the best available.
    bir_col = "boundary_interior_ratio"
    if bir_col not in df.columns:
        # Try to find any per-channel version
        bir_candidates = [
            c for c in df.columns if c.endswith("_boundary_interior_ratio")
        ]
        if bir_candidates:
            bir_col = bir_candidates[0]  # use first channel found

    # Log which features are available for each composite
    import logging
    _log = logging.getLogger(__name__)
    _merge_feats = [
        bir_col, "inertia_elongation", "skeleton_branches",
        "euler_number", "cross_section_bulge", "mean_neighbor_size_ratio",
    ]
    _frag_feats = ["area", "neighbor_count", "background_fraction",
                    "mean_neighbor_size_ratio"]
    _shape_feats = ["apical_basal_area_ratio", "cross_section_taper",
                     "inertia_elongation", "true_sphericity"]
    _log.info(
        "Named composites: merge=%d/%d, fragment=%d/%d, shape=%d/%d feats available"
        " (bir_col=%s)",
        sum(1 for f in _merge_feats if f in df.columns), len(_merge_feats),
        sum(1 for f in _frag_feats if f in df.columns), len(_frag_feats),
        sum(1 for f in _shape_feats if f in df.columns), len(_shape_feats),
        bir_col,
    )

    # merge_signal: high elongation, branches, bulge, size ratio;
    # low boundary_interior_ratio and euler_number
    df["merge_signal"] = _weighted_signal([
        (bir_col, -1),
        ("inertia_elongation", +1),
        ("skeleton_branches", +1),
        ("euler_number", -1),
        ("cross_section_bulge", +1),
        ("mean_neighbor_size_ratio", +1),
    ])

    # fragment_signal: small area, few neighbors, high background,
    # small relative to neighbors
    df["fragment_signal"] = _weighted_signal([
        ("area", -1),
        ("neighbor_count", -1),
        ("background_fraction", +1),
        ("mean_neighbor_size_ratio", -1),
    ])

    # boundary_confidence: requires probability features
    prob_cols = ["prob_edge_gradient", "prob_boundary_entropy",
                 "prob_boundary_mean"]
    if all(col in df.columns for col in prob_cols):
        # Higher = more confident. NOT inverted.
        edge_z = _robust_zscore(df["prob_edge_gradient"].values)
        entropy_z = _robust_zscore(df["prob_boundary_entropy"].values)
        boundary_z = _robust_zscore(df["prob_boundary_mean"].values)
        # Sharp edges + low entropy + high boundary prob = confident
        df["boundary_confidence"] = (edge_z - entropy_z + boundary_z) / 3.0
    else:
        df["boundary_confidence"] = np.full(n, np.nan, dtype=np.float64)

    # shape_abnormality: unusual shape relative to population
    df["shape_abnormality"] = _weighted_signal([
        ("apical_basal_area_ratio", +1),
        ("cross_section_taper", +1),
        ("inertia_elongation", +1),
        ("true_sphericity", -1),
    ])

    return df


def compute_composite_score(
    df: pd.DataFrame,
    method: str = "zscore",
    features: Optional[List[str]] = None,
    exclude_border: bool = False,
    n_pcs: int = 5,
) -> pd.DataFrame:
    """Compute a composite QC outlier score for each label.

    Args:
        df: Feature DataFrame (modified in-place).
        method: ``"zscore"`` — mean absolute z-score across selected features
            (simple, interpretable). ``"pca"`` — L2 norm in the top-``n_pcs``
            PCA component space (geometric distance from the average cell;
            user-flagged 2026-05-04 — pre-fix this used PC1 only and ignored
            the 'PCA dimensions' setting).  Higher = more outlier-like.
        features: Optional list of feature column names to include.
            Defaults to :data:`PCA_FEATURES` (curated, 15 decorrelated features).
        exclude_border: If True and ``is_border_cell`` column exists, exclude
            border cells from median/MAD fitting (they get scores but don't
            influence the axes). Default False for initial extraction.
        n_pcs: Number of PCA components to integrate over when
            ``method="pca"``.  Capped to ``min(len(features), n_samples-1)``.
            Ignored for ``method="zscore"``.  Default 5 matches Step 3's
            'PCA dimensions' setting.

    Returns:
        The same DataFrame with ``qc_score`` column added.
    """
    src = features if features is not None else PCA_FEATURES
    cols = [c for c in src if c in df.columns and c != "qc_score"]
    if not cols or len(df) < 2:
        df["qc_score"] = 0.0
        return df

    if method == "pca":
        # Phase E (2026-05-04): if persisted PC1..PCN columns exist,
        # take the L2 norm directly — they were computed via the same
        # SVD pipeline (compute_pca_columns → compute_pca_full mode='qc')
        # this branch's fresh fit would do, with exclude_border honored
        # at write time.  Mathematical identity preserved.  Fall back
        # to the inline SVD when columns are absent (Step 4 CSVs,
        # pre-Phase-E Step 3 CSVs).
        n_use_request = max(1, int(n_pcs))
        pc_matrix = _get_pc_matrix(df, n_use_request)
        if pc_matrix is not None:
            score = np.linalg.norm(pc_matrix, axis=1)
            score = score - score.min()
            df["qc_score"] = score
            return df

    X = df[cols].values.astype(np.float64)
    exclude = None
    if exclude_border and "is_border_cell" in df.columns:
        exclude = df["is_border_cell"].values.astype(bool)
    X_std = _robust_standardize(X, cols, exclude_mask=exclude)

    if method == "zscore":
        z = np.abs(X_std)
        df["qc_score"] = z.mean(axis=1)
    elif method == "pca":
        # Cap n_pcs to what's mathematically possible
        max_components = min(X_std.shape[0] - 1, X_std.shape[1])
        n_use = max(1, min(int(n_pcs), max_components))
        # Fit SVD on non-border rows, project all rows
        if exclude is not None and exclude.any():
            fit_X = X_std[~exclude]
            if fit_X.shape[0] < 2:
                df["qc_score"] = 0.0
                return df
            _U, S, Vt = np.linalg.svd(fit_X, full_matrices=False)
            # Project ALL rows onto the fitted PC axes (variance-scaled
            # coordinates).  Vt[i] is the i-th PC unit vector.
            pc_scores = X_std @ Vt[:n_use].T  # shape (n_samples, n_use)
            # Apply singular-value scaling so high-variance PCs
            # contribute more (matches the "U[:,i] * S[i]" semantics
            # used in the non-exclude branch).
            pc_scores = pc_scores * (S[:n_use] / S[:1])
        else:
            U, S, _Vt = np.linalg.svd(X_std, full_matrices=False)
            pc_scores = U[:, :n_use] * S[:n_use]  # (n_samples, n_use)
        # L2 norm = geometric distance from the average cell across
        # the top n_pcs PC axes.  Higher = more outlier-like.  This
        # replaces the pre-fix "PC1 only" behavior which ignored
        # PC2..PCn even when n_cluster_components > 1.
        score = np.linalg.norm(pc_scores, axis=1)
        # Shift to non-negative (already non-negative by construction
        # since L2 norm ≥ 0, but explicit for clarity / parity with
        # the old PC1-shift behavior).
        score = score - score.min()
        df["qc_score"] = score
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'zscore' or 'pca'.")

    return df


# ---------------------------------------------------------------------------
# Phase E (2026-05-04) — canonical PCA persistence
#
# ``compute_pca_columns`` writes ``PC1..PCN`` columns into the per-patch
# DataFrame at extract time so that downstream paths (clustering,
# composite score, plot widget, UMAP/t-SNE pre-reduction) can all read
# from the same coordinate system instead of independently re-fitting
# their own SVDs.  Previously Step 3 had three independent PCA fits
# that happened to be mathematically identical only because they
# shared preprocessing — Phase E removes the drift risk and the
# duplicated compute.
#
# Step 4's worker does not call ``compute_pca_columns``, and Step 4's
# CSVs do not contain PC columns.  All readers below check for
# presence first and fall back to a fresh SVD fit when columns are
# absent — Step 4 inherits zero behaviour change.
#
# Mathematical identity: ``compute_pca_columns`` writes the same
# ``U[:, :n] * S[:n]`` quantity that the existing fresh-fit branches
# compute internally, so the L2-norm in ``compute_composite_score
# (method='pca')`` and the projected coords used by ``compute_umap`` /
# ``compute_tsne`` for ``pca_dims`` pre-reduction produce identical
# results when reading from columns vs fitting on the fly.
# ---------------------------------------------------------------------------


def compute_pca_columns(
    df: pd.DataFrame,
    n_components: int = 5,
    features: Optional[List[str]] = None,
    exclude_border: bool = False,
) -> pd.DataFrame:
    """Fit PCA once and write ``PC1..PC{n_components}`` columns to ``df``.

    Phase E (2026-05-04): canonical PCA persistence so clustering,
    qc_score, and the workbench plot all read from the same column
    set instead of independently re-fitting.

    Uses :func:`compute_pca_full` in QC mode, which applies the same
    robust standardization (median/MAD) the existing
    :func:`_pca_project` and :func:`compute_composite_score` paths
    use internally.  When ``exclude_border=True``, the SVD fits on
    non-border rows; border rows are projected as supplementary
    points using the fitted Vt.

    Returns the same df for chainability.  Modifies in place.
    """
    result = compute_pca_full(
        df,
        n_components=n_components,
        features=features,
        exclude_border=exclude_border,
        mode="qc",
    )
    projected = result["projected"]
    n_written = projected.shape[1]
    for i in range(n_written):
        df[f"PC{i + 1}"] = projected[:, i]
    return df


def _get_pc_matrix(
    df: pd.DataFrame,
    n_components: int,
) -> Optional[np.ndarray]:
    """Return persisted ``PC1..PC{n_components}`` as a contiguous matrix.

    Phase E helper used by :func:`_pca_project`,
    :func:`compute_composite_score`, :func:`compute_umap`, and
    :func:`compute_tsne` to read the canonical PC columns when they
    exist.  Returns ``None`` if ``df`` has fewer than ``n_components``
    PC columns — callers fall back to a fresh fit in that case.
    """
    if df is None or df.empty or n_components < 1:
        return None
    needed = [f"PC{i + 1}" for i in range(n_components)]
    if not all(c in df.columns for c in needed):
        return None
    return df[needed].values.astype(np.float64)


def _pca_project(
    df: pd.DataFrame,
    features: Optional[List[str]],
    n_components: int,
) -> Tuple[np.ndarray, List[str]]:
    """Shared PCA projection for clustering algorithms.

    Returns (X_pca, cols) or raises ValueError if insufficient data.

    Phase E (2026-05-04): if persisted ``PC1..PCN`` columns exist
    (written by :func:`compute_pca_columns` during Step 3 extract),
    read them directly instead of re-fitting.  Falls back to fresh
    SVD when columns are absent — preserves Step 4 behaviour and
    pre-Phase-E CSV compatibility unchanged.
    """
    src = features if features is not None else PCA_FEATURES
    cols = [c for c in src if c in df.columns and c != "qc_score"]
    if not cols or len(df) < 2:
        raise ValueError("Insufficient features or samples")

    # Try persisted PCs first.
    pc_matrix = _get_pc_matrix(df, n_components)
    if pc_matrix is not None:
        return pc_matrix, cols

    # Fresh fit (legacy path / Step 4 / pre-Phase-E CSVs).
    X = df[cols].values.astype(np.float64)
    X_std = _robust_standardize(X.copy(), cols)

    k = min(n_components, X_std.shape[0], X_std.shape[1])
    U, S, Vt = np.linalg.svd(X_std, full_matrices=False)
    return U[:, :k] * S[:k], cols


# Supported clustering algorithms
CLUSTER_ALGORITHMS = ["leiden", "hdbscan", "kmeans", "gmm"]


def compute_clusters(
    df: pd.DataFrame,
    algorithm: str = "leiden",
    features: Optional[List[str]] = None,
    n_components: int = 5,
    *,
    # HDBSCAN params
    min_cluster_size: int = 15,
    # K-Means / GMM params
    n_clusters: int = 5,
    # Leiden params
    resolution: float = 1.0,
    n_neighbors: int = 30,
    metric: str = "euclidean",
) -> pd.DataFrame:
    """Cluster labels in PCA-reduced space using the specified algorithm.

    All algorithms follow the same pipeline:
    robust-standardize → PCA (top *n_components*) → cluster.

    Adds a ``cluster_id`` column (int, -1 = noise for HDBSCAN).

    Supported algorithms:
        - ``"leiden"``: Graph-based community detection via kNN graph.
          No noise points — every label is assigned. Resolution controls
          granularity. Standard in single-cell analysis (scanpy).
        - ``"hdbscan"``: Density-based; discovers clusters automatically.
          Produces noise points (-1). Good for arbitrary-shaped clusters.
        - ``"kmeans"``: Centroid-based; requires *n_clusters*. Fast, works
          well when PCA clusters are roughly spherical.
        - ``"gmm"``: Gaussian mixture model; like K-Means but allows
          elliptical clusters and soft assignment.

    Args:
        df: Feature DataFrame (modified in-place).
        algorithm: One of ``"leiden"``, ``"hdbscan"``, ``"kmeans"``, ``"gmm"``.
        features: Feature columns for PCA. Defaults to :data:`PCA_FEATURES`.
        n_components: Number of PCA components to cluster on (default 5).
        min_cluster_size: HDBSCAN minimum cluster size.
        n_clusters: Number of clusters for K-Means / GMM.
        resolution: Leiden resolution parameter (higher = more clusters).
        n_neighbors: Number of nearest neighbors for Leiden kNN graph.

    Returns:
        The same DataFrame with ``cluster_id`` column added.
    """
    try:
        X_pca, cols = _pca_project(df, features, n_components)
    except ValueError:
        df["cluster_id"] = -1
        return df

    if algorithm == "leiden":
        try:
            df["cluster_id"] = _cluster_leiden(
                X_pca, resolution, n_neighbors, metric=metric,
            )
        except ImportError:
            import warnings
            warnings.warn(
                "leidenalg/igraph not installed — falling back to HDBSCAN. "
                "Install with: pip install leidenalg python-igraph",
                stacklevel=2,
            )
            df["cluster_id"] = _cluster_hdbscan(X_pca, min_cluster_size)
    elif algorithm == "hdbscan":
        if len(df) < min_cluster_size:
            df["cluster_id"] = -1
            return df
        df["cluster_id"] = _cluster_hdbscan(X_pca, min_cluster_size)
    elif algorithm == "kmeans":
        df["cluster_id"] = _cluster_kmeans(X_pca, n_clusters)
    elif algorithm == "gmm":
        df["cluster_id"] = _cluster_gmm(X_pca, n_clusters)
    else:
        raise ValueError(
            f"Unknown algorithm {algorithm!r}. "
            f"Use one of: {CLUSTER_ALGORITHMS}"
        )
    return df


def _cluster_hdbscan(X: np.ndarray, min_cluster_size: int) -> np.ndarray:
    from sklearn.cluster import HDBSCAN
    hdb = HDBSCAN(min_cluster_size=min_cluster_size, n_jobs=1)
    return hdb.fit_predict(X)


def _cluster_kmeans(X: np.ndarray, n_clusters: int, *,
                    seed: int = 42) -> np.ndarray:
    """K-means.  ``seed`` is the random start (Step 4's Seed box); Step 3 passes none and gets 42, as before."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=int(seed))
    return km.fit_predict(X)


def _cluster_gmm(X: np.ndarray, n_clusters: int, *,
                 seed: int = 42) -> np.ndarray:
    """Gaussian mixture.  ``seed`` as for :func:`_cluster_kmeans`."""
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=n_clusters, random_state=int(seed))
    return gmm.fit_predict(X)


def _cluster_leiden(
    X: np.ndarray, resolution: float, n_neighbors: int,
    metric: str = "euclidean", *, seed: "int | None" = None,
) -> np.ndarray:
    """Leiden community detection on a kNN graph.

    ``seed`` is passed to ``leidenalg.find_partition`` when not
    ``None``, so the same seed gives the same groups; Step 3 passes none
    and is unchanged (Leiden then has no fixed start).
    """
    import igraph as ig
    import leidenalg
    from sklearn.neighbors import NearestNeighbors

    n_samples = X.shape[0]
    k = min(n_neighbors, n_samples - 1)
    if k < 2:
        return np.zeros(n_samples, dtype=np.intp)

    # Build kNN graph
    nn = NearestNeighbors(n_neighbors=k, metric=metric)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    # Build igraph from kNN edges (undirected, no self-loops)
    edges = []
    weights = []
    for i in range(n_samples):
        for j_idx in range(k):
            j = indices[i, j_idx]
            if j != i:
                edges.append((i, j))
                # Weight = 1 / (1 + distance) so closer = stronger
                weights.append(1.0 / (1.0 + distances[i, j_idx]))

    g = ig.Graph(n=n_samples, edges=edges, directed=False)
    g.es["weight"] = weights
    # Simplify: merge duplicate edges, keep max weight
    g.simplify(combine_edges="max")

    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        **({} if seed is None else {"seed": int(seed)}),
    )
    return np.array(partition.membership, dtype=np.intp)


# Keep backward-compatible alias
def compute_hdbscan_clusters(
    df: pd.DataFrame,
    min_cluster_size: int = 15,
    features: Optional[List[str]] = None,
    n_components: int = 5,
) -> pd.DataFrame:
    """Backward-compatible wrapper — calls :func:`compute_clusters` with HDBSCAN."""
    return compute_clusters(
        df, algorithm="hdbscan", features=features,
        n_components=n_components, min_cluster_size=min_cluster_size,
    )


def assign_qc_tiers(
    df: pd.DataFrame,
    thresholds: Tuple[float, float, float],
) -> pd.DataFrame:
    """Assign ``qc_tier`` column based on 3 threshold values on ``qc_score``.

    Tiers (higher score = worse):
    - ``'good'``: score <= t1
    - ``'review'``: t1 < score <= t2
    - ``'bad'``: t2 < score <= t3
    - ``'garbage'``: score > t3

    Args:
        df: Feature DataFrame (modified in-place).
        thresholds: Tuple of 3 threshold values (will be sorted).

    Returns:
        The same DataFrame with ``qc_tier`` column added.
    """
    t1, t2, t3 = sorted(thresholds)
    if "qc_score" in df.columns:
        score = df["qc_score"].values.astype(np.float64)
    else:
        score = np.zeros(len(df))
    tiers = np.where(
        score <= t1, "good",
        np.where(
            score <= t2, "review",
            np.where(score <= t3, "bad", "garbage"),
        ),
    )
    # Preserve any existing "curated" tier assignments — curated labels are
    # manually verified and must never be overwritten by threshold re-assignment.
    if "qc_tier" in df.columns:
        curated_mask = df["qc_tier"].values == "curated"
        tiers = np.where(curated_mask, "curated", tiers)
    df["qc_tier"] = tiers
    return df


def extract_single_label_features(
    labels_3d: np.ndarray,
    label_id: int,
    intensity_image: Optional[np.ndarray] = None,
    prob_3d: Optional[np.ndarray] = None,
    spacing: Optional[Tuple[float, ...]] = None,
    bbox_padding: int = 3,
) -> pd.DataFrame:
    """Extract QC features for a single label from a full 3D volume.

    Crops the volume to the label's bounding box (+ padding), runs the full
    feature extraction pipeline on the isolated crop, then adjusts centroids
    and bounding-box coordinates back to full-volume space.

    Neighbor features are set to NaN (meaningless in an isolated crop).
    ``compute_composite_score()`` is **not** called — z-score on a single row
    is meaningless.  The caller should compute ``qc_score`` using population
    statistics from the full DataFrame after inserting this row.

    Args:
        labels_3d: Full 3D integer label array (Z, Y, X). Background is 0.
        label_id: The label to extract features for.
        intensity_image: Optional full 3D intensity image (same shape).
        prob_3d: Optional full 3D probability volume (same shape).
        spacing: Voxel spacing as (Z, Y, X) tuple. None = voxel units.
        bbox_padding: Padding (voxels) around the label bounding box.

    Returns:
        1-row DataFrame with features, or empty DataFrame if *label_id* is
        not found in the volume.

    Raises:
        ValueError: If *labels_3d* is not 3D.
    """
    if labels_3d.ndim != 3:
        raise ValueError(f"Expected 3D labels, got shape {labels_3d.shape}")

    # --- Locate label and crop -------------------------------------------------
    mask = labels_3d == label_id
    if not np.any(mask):
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    coords = np.argwhere(mask)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1  # exclusive upper bound

    # Apply padding, clamp to volume bounds
    shape = np.array(labels_3d.shape)
    crop_min = np.clip(mins - bbox_padding, 0, shape)
    crop_max = np.clip(maxs + bbox_padding, 0, shape)

    slices = tuple(slice(lo, hi) for lo, hi in zip(crop_min, crop_max))

    # Build isolated crop — only keep the target label
    crop_labels = np.zeros_like(labels_3d[slices])
    crop_mask = labels_3d[slices] == label_id
    crop_labels[crop_mask] = label_id

    crop_intensity = intensity_image[slices] if intensity_image is not None else None
    crop_prob = prob_3d[slices] if prob_3d is not None else None

    # --- Run feature pipeline on crop ------------------------------------------
    df = extract_features(crop_labels, intensity_image=crop_intensity, spacing=spacing)
    if len(df) == 0:
        return df

    # Surface features (surface_area, true_sphericity, roughness, solidity)
    compute_surface_features(crop_labels, df, spacing=spacing)

    # Skeleton features (skeleton_branches, skeleton_length, etc.)
    compute_skeleton_features(crop_labels, df)

    # Boundary/interior features (if intensity available)
    if crop_intensity is not None:
        compute_boundary_interior_features(crop_labels, crop_intensity, df)

    # Probability features (if probability volume available)
    if crop_prob is not None:
        compute_probability_features(crop_labels, df, crop_prob, bbox_padding=0)

    # --- Adjust coordinates back to full-volume space --------------------------
    offset_z, offset_y, offset_x = crop_min
    for col, off in [
        ("centroid-0", offset_z), ("centroid-1", offset_y), ("centroid-2", offset_x),
    ]:
        if col in df.columns:
            df[col] = df[col] + off

    # Adjust bounding box columns (bbox-0/1/2 = min z/y/x, bbox-3/4/5 = max z/y/x)
    for col, off in [
        ("bbox-0", offset_z), ("bbox-1", offset_y), ("bbox-2", offset_x),
        ("bbox-3", offset_z), ("bbox-4", offset_y), ("bbox-5", offset_x),
    ]:
        if col in df.columns:
            df[col] = df[col] + off

    # --- Neighbor features → NaN (meaningless in isolated crop) ----------------
    neighbor_cols = [c for c in df.columns if "neighbor" in c.lower()]
    for col in neighbor_cols:
        df[col] = np.nan

    return df


def apply_threshold(
    df: pd.DataFrame,
    feature: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> pd.DataFrame:
    """Add a 'qc_status' column based on feature thresholds.

    Labels with the feature value within [min_val, max_val] get 'pass',
    others get 'flagged'. None means no bound on that side.

    Args:
        df: DataFrame from extract_features (modified in-place).
        feature: Column name to threshold.
        min_val: Lower bound (inclusive). None = no lower bound.
        max_val: Upper bound (inclusive). None = no upper bound.

    Returns:
        The same DataFrame with 'qc_status' column updated.
    """
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not in DataFrame columns: {list(df.columns)}")

    values = df[feature].values
    mask = np.ones(len(values), dtype=bool)
    if min_val is not None:
        mask &= values >= min_val
    if max_val is not None:
        mask &= values <= max_val

    df["qc_status"] = np.where(mask, "pass", "flagged")
    return df


def active_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return only active (non-ghost) rows from a QC DataFrame.

    Ghost rows have ``label_status`` of ``"deleted"`` or ``"superseded"``.
    If the column doesn't exist, all rows are considered active.
    """
    if "label_status" not in df.columns:
        return df
    return df[df["label_status"] == "active"]


#: ``os.replace`` onto a file another reader holds open raises
#: PermissionError on Windows (a count worker, a notebook, an antivirus
#: scan); the replace is retried this many times, this far apart (A2 fold)
REPLACE_RETRY_TRIES = 10
REPLACE_RETRY_DELAY_S = 0.05


def replace_with_retry(src, dst) -> None:
    """``os.replace(src, dst)``, retried on PermissionError up to
    :data:`REPLACE_RETRY_TRIES` times :data:`REPLACE_RETRY_DELAY_S` apart,
    then raised."""
    import os
    import time

    tries = max(int(REPLACE_RETRY_TRIES), 1)
    for attempt in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == tries - 1:
                raise
            time.sleep(REPLACE_RETRY_DELAY_S)


def save_patch_csv(df: pd.DataFrame, output_path: Union[str, Path]) -> None:
    """Save feature DataFrame to CSV atomically (temp + os.replace).

    Writes to a temporary file in the same directory, then atomically
    replaces the target.  If the write fails mid-way, the original file
    (if any) is preserved and no partial CSV is left on disk.
    """
    import os
    import tempfile

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        suffix=".csv", dir=str(output_path.parent)
    )
    os.close(fd)
    try:
        # UTF-8 on every platform (pandas' default too; explicit, so a
        # notebook's plain pd.read_csv always reads it)
        df.to_csv(tmp_path, index=False, encoding="utf-8")
        replace_with_retry(tmp_path, str(output_path))
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def save_adjacency_csv(
    df: pd.DataFrame, adjacency_path: Union[str, Path],
) -> bool:
    """Write ``df.attrs["_adjacency_list"]`` to a CSV.

    Mirrors Step 4's adjacency-write pattern
    (``segmentation/controller.py:823-827``) for parity.  No-op (returns
    ``False``) if the ``_adjacency_list`` attribute is missing or
    empty — the QC extraction may have skipped adjacency computation
    for performance reasons, in which case there's nothing to write.

    Returns:
        True if a file was written, False if no adjacency data was
        available.

    
    """
    adj_rows = df.attrs.get("_adjacency_list")
    if not adj_rows:
        return False
    adj_path = Path(adjacency_path)
    adj_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(adj_rows).to_csv(str(adj_path), index=False)
    return True


def save_features_raw_backup(
    features_csv_path: Union[str, Path],
    raw_backup_path: Union[str, Path],
) -> bool:
    """Copy a freshly-written features CSV to a raw-backup sibling.

    The raw backup is an **immutable snapshot of the pre-curation
    extraction**.  Subsequent post-curation re-extractions must NOT
    overwrite it — that's how we keep the "what did the model
    originally predict?" answer available for the classifier
    training-data workflow (see memory
    ``project-classifier-training-data-vision``).

    Returns:
        True if a backup file was written this call, False if the
        backup already exists with non-zero size, or the source CSV
        is missing.

    Idempotent: safe to call multiple times.  After the first
    successful write, subsequent calls are no-ops.

    **Empty / zero-byte backup is treated as missing** — defensive
    against a prior aborted copy that left an empty stub.  Detected
    via stat().st_size and re-attempted.

    
    """
    import shutil

    raw_path = Path(raw_backup_path)
    # Treat zero-byte raw backup as missing (aborted prior write).
    if raw_path.exists() and raw_path.stat().st_size > 0:
        return False  # already backed up — never overwrite
    src_path = Path(features_csv_path)
    if not src_path.exists():
        return False  # nothing to back up yet (no canonical CSV)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src_path), str(raw_path))
    return True


def persist_qc_artifacts(
    df: pd.DataFrame,
    features_csv_path: Union[str, Path],
    *,
    patch_status: str = "unknown",
) -> dict:
    """Top-level helper called right after :func:`save_patch_csv`.

    Persists the net-new QC artifacts the Step 3 restructure scoped:

      * ``{patch}_features_raw.csv``  — immutable pre-curation snapshot
      * ``{patch}_adjacency_raw.csv``  — pre-curation neighbor graph
      * ``{patch}_adjacency_curated.csv`` — post-curation neighbor graph

    Args:
        df: Features DataFrame with optional ``df.attrs["_adjacency_list"]``.
        features_csv_path: Path of the just-written canonical features CSV.
        patch_status: Patch status from ``state_manager.get_patch_status``.
            One of: ``"reconstructed"``, ``"curating"``, ``"accepted"``,
            ``"finalized"``, ``"unknown"``.  This gates the raw-backup
            write so we don't capture POST-curation data into the "raw"
            file — that would silently violate the
            "raw = pre-curation" contract.  When status
            is unknown (legacy callers / tests), the old behavior
            (existence-only check) is preserved.

    Decision matrix (revised 2026-05-20 after user-reported overwrite bug):

    ============================  ==========================  =====================
    patch_status                  raw_features.csv exists?    Action
    ============================  ==========================  =====================
    reconstructed                 no                          INITIAL: write raw
                                                              backup + raw adj.
    reconstructed                 yes                         RE-EXTRACT pre-cur:
                                                              LEAVE BOTH raw files
                                                              alone.  Adjacency
                                                              not re-written.
                                                              (Raw files are the
                                                              first-extraction
                                                              immutable snapshot —
                                                              parity with
                                                              features_raw.csv's
                                                              idempotency.)
    curating/accepted/finalized   no                          POST-CUR FIRST:
                                                              SKIP raw backup
                                                              (would be wrong-
                                                              data), write
                                                              curated adj.
    curating/accepted/finalized   yes                         RE-EXTRACT post-cur:
                                                              leave raw alone,
                                                              write curated adj.
    unknown                       (existence-only fallback)   Legacy behavior.
    ============================  ==========================  =====================

    Returns:
        ``{
            "raw_backup_written": bool,
            "adjacency_path": str | None,
            "adjacency_written": bool,
            "is_initial": bool,
            "is_pre_curation": bool,
            "raw_skipped_post_curation": bool,
         }``

    Side effect: when transitioning into the INITIAL state (writing
    raw_backup + raw_adj), any STALE ``{patch}_adjacency_curated.csv``
    sibling is deleted.  Otherwise a re-init scenario (e.g., user
    manually deleted features_raw.csv) would leave the stale curated
    adjacency on disk alongside a brand-new raw adjacency — internally
    inconsistent.
    """
    features_csv = Path(features_csv_path)
    parent = features_csv.parent
    stem = features_csv.name
    if stem.endswith("_features.csv"):
        patch_stem = stem[: -len("_features.csv")]
    else:
        # Defensive: caller passed a non-standard CSV name.  Use the
        # filename stem as the patch prefix; the adjacency siblings
        # still cluster under that prefix.
        patch_stem = features_csv.stem

    raw_features_path = parent / f"{patch_stem}_features_raw.csv"
    raw_adj_path = parent / f"{patch_stem}_adjacency_raw.csv"
    curated_adj_path = parent / f"{patch_stem}_adjacency_curated.csv"

    # Treat zero-byte raw backup as missing (parity with
    # save_features_raw_backup's recovery rule).
    raw_exists = (
        raw_features_path.exists()
        and raw_features_path.stat().st_size > 0
    )

    is_pre_curation = patch_status in ("reconstructed", "unknown")
    is_initial = (not raw_exists) and is_pre_curation
    raw_skipped_post_curation = (
        (not raw_exists) and (not is_pre_curation)
    )

    result = {
        "raw_backup_written": False,
        "adjacency_path": None,
        "adjacency_written": False,
        "is_initial": is_initial,
        "is_pre_curation": is_pre_curation,
        "raw_skipped_post_curation": raw_skipped_post_curation,
    }

    # Track whether we'll write adjacency at all this call (sometimes
    # the right answer is "skip both raw and curated").
    write_adjacency = True
    adj_path = None

    if is_initial:
        result["raw_backup_written"] = save_features_raw_backup(
            features_csv, raw_features_path,
        )
        adj_path = raw_adj_path
        # Drop any stale curated adjacency — re-entering INITIAL
        # state would otherwise leave a contradictory pair on disk.
        if curated_adj_path.exists():
            try:
                curated_adj_path.unlink()
            except OSError:
                pass
    elif is_pre_curation:
        # Re-extract while still pre-curation (raw backup already
        # exists from a prior initial extraction).
        # 2026-05-20 fix (user-reported): the raw adjacency is the
        # PAIR for the immutable raw features backup.  Both files
        # represent the first-extraction snapshot.  Overwriting
        # adjacency_raw.csv while leaving features_raw.csv alone
        # would silently desynchronize the pair (features_raw rows
        # describe cell A,B,C... while adjacency_raw edges describe
        # the post-re-extract cells).  Instead: skip adjacency write
        # entirely on pre-curation re-extract.  The first
        # extraction's adjacency_raw.csv stays as the canonical
        # pre-curation neighbor graph.
        write_adjacency = False
        adj_path = raw_adj_path  # recorded for the return dict
    else:
        # Post-curation extraction (whether first or re-extract).
        # Skip raw backup entirely; adjacency goes to curated.
        adj_path = curated_adj_path

    result["adjacency_path"] = str(adj_path) if adj_path else None
    if write_adjacency and adj_path is not None:
        result["adjacency_written"] = save_adjacency_csv(df, adj_path)
    return result


def load_patch_csv(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Load feature DataFrame from CSV.

    Ensures the ``feature_state`` and ``label_status`` columns exist
    (backward compat for CSVs created before these columns were added).
    """
    df = pd.read_csv(str(csv_path))
    if "feature_state" not in df.columns:
        df["feature_state"] = "fresh"
    if "label_status" not in df.columns:
        df["label_status"] = "active"
    return df


def merge_patch_csvs(csv_paths: Sequence[Union[str, Path]]) -> pd.DataFrame:
    """Concatenate multiple patch CSVs, adding a 'patch_name' column.

    The patch name is derived from the CSV filename (stem without _features).
    Applies the same ``feature_state`` backward-compat fill as
    :func:`load_patch_csv` so old CSVs work in population mode.
    """
    frames = []
    for p in csv_paths:
        p = Path(p)
        df = pd.read_csv(str(p))
        # Backward compat: ensure feature_state and label_status exist
        if "feature_state" not in df.columns:
            df["feature_state"] = "fresh"
        if "label_status" not in df.columns:
            df["label_status"] = "active"
        stem = p.stem
        if stem.endswith("_features"):
            stem = stem[: -len("_features")]
        df["patch_name"] = stem
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# 3D Mesh Export (marching cubes → OBJ)
# ---------------------------------------------------------------------------


def export_cell_mesh_obj(
    labels_3d: np.ndarray,
    label_id: int,
    output_path: Union[str, Path],
    spacing: Optional[Tuple[float, float, float]] = None,
    obj_slice: Optional[tuple] = None,
) -> dict:
    """Export a single cell as an OBJ mesh via marching cubes.

    Parameters
    ----------
    labels_3d : ndarray
        3D label volume.
    label_id : int
        Label ID to extract.
    output_path : path
        Output .obj file path.
    spacing : tuple, optional
        (Z, Y, X) voxel spacing. Applied to vertex coordinates.
    obj_slice : tuple of slices, optional
        Precomputed bounding box from scipy.ndimage.find_objects.
        If None, computed internally (slower for batch calls).

    Returns
    -------
    dict with keys: n_vertices, n_faces, centroid_z, centroid_y, centroid_x.
    Returns None if the label has insufficient volume for meshing.
    """
    from skimage.measure import marching_cubes

    # Get bounding box
    if obj_slice is None:
        from scipy.ndimage import find_objects
        slices = find_objects(labels_3d)
        if label_id < 1 or label_id > len(slices) or slices[label_id - 1] is None:
            return None
        obj_slice = slices[label_id - 1]

    # Crop to bounding box + 1-voxel padding for clean mesh edges
    z0, z1 = obj_slice[0].start, obj_slice[0].stop
    y0, y1 = obj_slice[1].start, obj_slice[1].stop
    x0, x1 = obj_slice[2].start, obj_slice[2].stop
    pad = 1
    z0p = max(z0 - pad, 0)
    y0p = max(y0 - pad, 0)
    x0p = max(x0 - pad, 0)
    z1p = min(z1 + pad, labels_3d.shape[0])
    y1p = min(y1 + pad, labels_3d.shape[1])
    x1p = min(x1 + pad, labels_3d.shape[2])

    # Build mask only on the small crop (not the full volume)
    crop = (labels_3d[z0p:z1p, y0p:y1p, x0p:x1p] == label_id).astype(
        np.float32
    )
    if crop.sum() < 10:
        return None

    try:
        sp = spacing if spacing else (1.0, 1.0, 1.0)
        verts, faces, _, _ = marching_cubes(crop, level=0.5, spacing=sp)
    except Exception:
        return None

    # Offset vertices to global coordinates
    offset = np.array([z0p, y0p, x0p], dtype=np.float64)
    if spacing:
        offset = offset * np.array(spacing)
    verts = verts + offset

    # Centroid in physical coordinates
    centroid = verts.mean(axis=0)
    n_verts = len(verts)
    n_faces = len(faces)

    # Write OBJ (simple text format — vertices + faces)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # UTF-8 with "\n" and an ASCII header (owner export 2026-09-29: the
    # platform default wrote a cp1252 em-dash and "\r\n" into every .obj)
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# EpiCycle cell mesh - label {label_id}\n")
        f.write(f"# vertices: {n_verts}, faces: {n_faces}\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            # OBJ faces are 1-indexed
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    # Explicit cleanup — free mesh arrays before next iteration
    result = {
        "n_vertices": n_verts,
        "n_faces": n_faces,
        "centroid_z": float(centroid[0]),
        "centroid_y": float(centroid[1]),
        "centroid_x": float(centroid[2]),
    }
    del verts, faces, crop, centroid
    return result


def export_all_cell_meshes(
    labels_3d: np.ndarray,
    output_dir: Union[str, Path],
    stack_name: str,
    spacing: Optional[Tuple[float, float, float]] = None,
    label_ids: Optional[List[int]] = None,
    progress_cb=None,
) -> pd.DataFrame:
    """Export all cells as individual OBJ meshes + manifest CSV.

    Parameters
    ----------
    labels_3d : ndarray
        3D label volume.
    output_dir : path
        Directory for mesh files and manifest.
    stack_name : str
        Stack identifier for the manifest.
    spacing : tuple, optional
        (Z, Y, X) voxel spacing.
    label_ids : list, optional
        Specific labels to export. If None, exports all non-zero labels.
    progress_cb : callable, optional
        Called with (current, total) for progress tracking.

    Returns
    -------
    DataFrame — mesh manifest with columns:
        label, stack_name, mesh_path, n_vertices, n_faces,
        centroid_z, centroid_y, centroid_x
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if label_ids is None:
        label_ids = sorted(int(lid) for lid in np.unique(labels_3d) if lid != 0)

    # Compute bounding boxes once for all labels
    from scipy.ndimage import find_objects
    all_slices = find_objects(labels_3d)

    rows = []
    skipped = []
    n_total = len(label_ids)
    for i, lid in enumerate(label_ids):
        obj_path = output_dir / f"cell_{lid}.obj"
        # Pass precomputed slice to avoid re-scanning the full volume
        s = all_slices[lid - 1] if 0 < lid <= len(all_slices) else None
        result = export_cell_mesh_obj(
            labels_3d, lid, obj_path, spacing=spacing, obj_slice=s,
        )
        if result is not None:
            rows.append({
                "label": lid,
                "stack_name": stack_name,
                "mesh_path": str(obj_path),
                "n_vertices": result["n_vertices"],
                "n_faces": result["n_faces"],
                "centroid_z": result["centroid_z"],
                "centroid_y": result["centroid_y"],
                "centroid_x": result["centroid_x"],
            })
        else:
            skipped.append(int(lid))       # under 10 voxels, or no surface
        if progress_cb:
            progress_cb(i + 1, n_total)
        # Periodic GC to prevent memory accumulation
        if (i + 1) % 100 == 0:
            import gc
            gc.collect()

    # Final cleanup
    del all_slices
    import gc
    gc.collect()

    manifest = pd.DataFrame(rows)
    manifest_path = output_dir / "mesh_manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8")
    # The cells that got no mesh, so the caller can say so (Step 4's
    # EXPORT_MESH_DONE_SKIPPED); the manifest's columns are unchanged.
    manifest.attrs["skipped_labels"] = skipped
    return manifest


def detect_axes_order(shape: Tuple[int, ...]) -> str:
    """Auto-detect axes order for a 4D stack based on dimension sizes.

    Heuristic: the channel axis is the smallest dimension (typically 1-5
    channels vs hundreds of Z/Y/X pixels).

    - shape[0] smallest → CZYX (channel-first, most common in microscopy)
    - shape[1] smallest → ZCYX (Z-first)
    - 3D → ZYX (no channel axis)

    Args:
        shape: Array shape tuple (3D or 4D).

    Returns:
        Detected axes string: ``"CZYX"``, ``"ZCYX"``, or ``"ZYX"``.
    """
    if len(shape) == 3:
        return "ZYX"
    if len(shape) != 4:
        return "CZYX"  # fallback

    # Find the axis with the smallest size — that's likely the channel axis
    dim0, dim1 = shape[0], shape[1]

    if dim0 <= dim1 and dim0 <= shape[2]:
        return "CZYX"  # first axis is smallest → channel-first
    elif dim1 <= dim0 and dim1 <= shape[2]:
        return "ZCYX"  # second axis is smallest → Z-first, then channel
    else:
        return "CZYX"  # fallback to most common


def resolve_patch_axes(
    image: np.ndarray,
    *,
    labels: Optional[np.ndarray] = None,
    image_path: Optional[Union[str, Path]] = None,
    fallback: str = "CZYX",
) -> str:
    """Resolve the axis order of ONE patch's source stack.

    The Step 3 predict + extract paths historically took a SINGLE global
    ``axes`` value (the UI ``axes_combo``) and applied it to every patch.
    In a folder whose patches have different C/Z orderings (some CZYX, some
    ZCYX), that one value is wrong for whichever patches don't match it —
    surfacing as an IndexError ("failed to extract") on extract, or a
    silently mis-predicted stack on predict.  This helper resolves the layout
    per patch from the most authoritative signal available.

    Priority (most → least authoritative):
      1. TIFF metadata: if ``image_path`` given, read
         ``tifffile.TiffFile(path).series[0].axes``; use it ONLY if it is
         exactly ``'ZCYX'`` or ``'CZYX'`` (upper-cased).
      2. Labels-shape match (deterministic): if ``labels`` given and ``image``
         is 4D, ``arr[0].shape == labels.shape`` (ZYX) → ``'CZYX'``;
         ``arr[:, 0].shape == labels.shape`` (ZYX) → ``'ZCYX'``.  Robust to a
         thin-Z stack (Z <= C) that the size heuristic gets wrong.
      3. Size heuristic: ``detect_axes_order(arr.shape)`` if it returns
         ``'ZCYX'`` / ``'CZYX'``.
      4. ``fallback`` (normalized to ``'ZCYX'`` / ``'CZYX'``; if ``fallback``
         itself isn't one of those, use ``'CZYX'`` — EpiCycle's default
         layout for its own extracted stacks).

    Returns ``'ZYX'`` for 3D input (no C/Z ambiguity).  Every step is wrapped
    so a failure falls through to the next signal — this function never
    raises.

    Args:
        image: The patch's source stack (3D ZYX or 4D ZCYX/CZYX).
        labels: Optional 3D ZYX labels volume for the deterministic match.
        image_path: Optional path to the source TIFF, for the metadata signal.
        fallback: Last-resort axes string (the global combo value).

    Returns:
        One of ``'ZYX'`` (3D), ``'ZCYX'``, or ``'CZYX'``.
    """
    # 3D input has no C/Z ambiguity — callers ignore axes for 3D.
    try:
        if getattr(image, "ndim", None) == 3:
            return "ZYX"
    except Exception:
        pass

    # 1) TIFF metadata (most authoritative when present + recognizable).
    if image_path is not None:
        try:
            import tifffile

            with tifffile.TiffFile(str(image_path)) as tf:
                meta_axes = getattr(tf.series[0], "axes", "") or ""
            meta_axes = meta_axes.upper()
            if meta_axes in ("ZCYX", "CZYX"):
                return meta_axes
        except Exception:
            pass

    # 2) Deterministic labels-shape match (only meaningful for 4D images).
    try:
        if getattr(image, "ndim", None) == 4 and labels is not None:
            lbl_shape = tuple(labels.shape)  # (Z, Y, X)
            if tuple(image[0].shape) == lbl_shape:
                return "CZYX"  # (C, Z, Y, X): channel-0 on axis 0 → ZYX
            if tuple(image[:, 0].shape) == lbl_shape:
                return "ZCYX"  # (Z, C, Y, X): channel-0 on axis 1 → ZYX
    except Exception:
        pass

    # 3) Size heuristic.
    try:
        heur = detect_axes_order(tuple(image.shape))
        if heur in ("ZCYX", "CZYX"):
            return heur
    except Exception:
        pass

    # 4) Fallback (normalized).
    try:
        fb = str(fallback).upper()
        if fb in ("ZCYX", "CZYX"):
            return fb
    except Exception:
        pass
    return "CZYX"


def extract_single_channel(
    src: np.ndarray,
    labels_shape: tuple,
    log=None,
    seg_channel: int = 0,
) -> Optional[np.ndarray]:
    """Extract a single 3D channel from a source stack, matching labels_shape.

    Handles CZYX, ZCYX, and plain ZYX layouts.  Pure numpy — no Qt dependency.

    Args:
        src: Raw source stack (3D or 4D).
        labels_shape: Expected ZYX shape of the labels volume.
        log: Optional signal emitter (must have ``.emit(str)``).
        seg_channel: Which channel to extract (0-based array index).

    Returns:
        3D numpy array or None on mismatch.
    """
    if src.ndim == 3:
        if src.shape == labels_shape:
            return src
        if log:
            log.emit(f"[WARN] 3D intensity shape {src.shape} != labels {labels_shape}")
        return None

    if src.ndim == 4:
        arr_idx = max(0, seg_channel)

        # Try CZYX (channel axis 0)
        if src[0].shape == labels_shape:
            c = min(arr_idx, src.shape[0] - 1)
            if log:
                log.emit(
                    f"Source stack: CZYX {src.shape}, channel {c} "
                    f"(seg_channel={seg_channel})"
                )
            return src[c]
        # Try ZCYX (channel axis 1)
        if src[:, 0].shape == labels_shape:
            c = min(arr_idx, src.shape[1] - 1)
            if log:
                log.emit(
                    f"Source stack: ZCYX {src.shape}, channel {c} "
                    f"(seg_channel={seg_channel})"
                )
            return src[:, c]
        # Generic: find channel axis
        for ax in range(src.ndim):
            if np.take(src, 0, axis=ax).shape == labels_shape:
                c = min(arr_idx, src.shape[ax] - 1)
                if log:
                    log.emit(f"Source stack: axis={ax} {src.shape}, channel {c}")
                return np.take(src, c, axis=ax)

    if log:
        log.emit(
            f"[WARN] Cannot extract 3D channel from shape {src.shape} "
            f"(labels: {labels_shape})"
        )
    return None


def spacing_label(spacing: Optional[Tuple[float, ...]] = None) -> str:
    """Format spacing for display.

    Returns e.g. ``'(0.3, 0.1, 0.1) µm'`` or ``'voxel units'``.
    """
    if spacing is None:
        return "voxel units"
    return f"({spacing[0]:.3g}, {spacing[1]:.3g}, {spacing[2]:.3g}) µm"


def compute_umap(
    df: pd.DataFrame,
    n_components: int = 2,
    features: Optional[List[str]] = None,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    pca_dims: Optional[int] = None,
    metric: str = "euclidean",
) -> np.ndarray:
    """UMAP non-linear embedding using the same feature set as PCA.

    Falls back to PCA if ``umap-learn`` is not installed.

    Args:
        df: Feature DataFrame (from extract_features or merge_patch_csvs).
        n_components: Number of embedding dimensions.
        features: Optional list of feature column names to include.
            Defaults to :data:`PCA_FEATURES`.
        n_neighbors: UMAP n_neighbors parameter (locality).
        min_dist: UMAP min_dist parameter (compactness).
        pca_dims: If not None, first reduce to this many PCA components
            before running UMAP. Builds the k-NN graph in PCA space
            (denoised, variance-ranked) instead of the full ambient
            feature space.  Phase E: prefers persisted ``PC1..PC{pca_dims}``
            columns from the df when present, falls back to inline SVD.
        metric: Distance metric for the kNN graph (Phase E 2026-05-04).
            Defaults to ``"euclidean"`` (matches pre-Phase-E behaviour).
            Common values: ``"euclidean"``, ``"manhattan"``, ``"cosine"``,
            ``"correlation"``.  Forwarded to ``umap.UMAP(metric=...)``.

    Returns:
        Projected coordinates as np.ndarray of shape (n_samples, n_components).
    """
    import warnings

    try:
        import umap  # noqa: F811
    except ImportError:
        warnings.warn(
            "umap-learn not installed; falling back to PCA. "
            "Install with: pip install umap-learn",
            stacklevel=2,
        )
        return compute_pca(df, n_components, features)

    src = features if features is not None else PCA_FEATURES
    cols = [c for c in src if c in df.columns]
    if not cols:
        return np.zeros((len(df), n_components))

    X = df[cols].values.astype(np.float64)
    n_samples, n_features = X.shape

    if n_samples < 2 or n_features < 1:
        return np.zeros((n_samples, n_components))

    # Same robust standardization as PCA
    X_std = _robust_standardize(X, cols)

    # Optional PCA pre-reduction: build k-NN graph in PC space.
    # Phase E (2026-05-04): prefer persisted PC columns when available
    # so UMAP runs in the same coordinate system as clustering and
    # qc_score.  Falls back to inline SVD when PC columns absent.
    if pca_dims is not None and pca_dims > 0:
        pc_matrix = _get_pc_matrix(df, pca_dims)
        if pc_matrix is not None:
            X_std = pc_matrix
        else:
            k_pca = min(pca_dims, n_samples, n_features)
            U, S, _Vt = np.linalg.svd(X_std, full_matrices=False)
            X_std = U[:, :k_pca] * S[:k_pca]

    k = min(n_neighbors, n_samples - 1)
    if k < 2:
        return compute_pca(df, n_components, features)

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=k,
        min_dist=min_dist,
        metric=metric,
        random_state=42,
    )
    projected = reducer.fit_transform(X_std)

    if projected.shape[1] < n_components:
        pad = np.zeros((n_samples, n_components - projected.shape[1]))
        projected = np.hstack([projected, pad])

    return projected


def compute_tsne(
    df: pd.DataFrame,
    n_components: int = 2,
    features: Optional[List[str]] = None,
    perplexity: float = 30.0,
    pca_dims: Optional[int] = None,
    metric: str = "euclidean",
) -> np.ndarray:
    """t-SNE non-linear embedding using the same feature set as PCA.

    Mirrors :func:`compute_umap`'s pipeline: select features → robust
    standardize → optional PCA pre-reduction → embed.  Falls back to a
    zeros array when sklearn is not importable or the sample count is
    too small for the requested perplexity.

    Args:
        df: Feature DataFrame (from :func:`extract_features`).
        n_components: Number of embedding dimensions (default 2).
        features: Optional feature column whitelist.  Defaults to
            :data:`PCA_FEATURES`.
        perplexity: t-SNE perplexity.  Capped at ``n_samples - 1``.
        pca_dims: If not None, pre-reduce to this many PCA components
            before t-SNE so t-SNE operates in the same denoised PC
            space as Leiden / UMAP.  Phase E: prefers persisted
            ``PC1..PC{pca_dims}`` columns from the df when present,
            falls back to inline SVD.
        metric: Distance metric for t-SNE neighbor search (Phase E
            2026-05-04).  Defaults to ``"euclidean"``.  Forwarded to
            ``sklearn.manifold.TSNE(metric=...)``.

    Returns:
        Projected coordinates as np.ndarray of shape
        ``(n_samples, n_components)``.
    """
    import warnings

    try:
        from sklearn.manifold import TSNE
    except ImportError:
        warnings.warn(
            "scikit-learn not installed; falling back to PCA. "
            "Install with: pip install scikit-learn",
            stacklevel=2,
        )
        return compute_pca(df, n_components, features)

    src = features if features is not None else PCA_FEATURES
    cols = [c for c in src if c in df.columns]
    if not cols:
        return np.zeros((len(df), n_components))

    X = df[cols].values.astype(np.float64)
    n_samples, n_features = X.shape

    if n_samples < 2 or n_features < 1:
        return np.zeros((n_samples, n_components))

    X_std = _robust_standardize(X, cols)

    # Optional PCA pre-reduction so the embedding shares space with
    # Leiden clustering and UMAP (matches scanpy convention).
    # Phase E (2026-05-04): prefer persisted PC columns over inline SVD.
    if pca_dims is not None and pca_dims > 0:
        pc_matrix = _get_pc_matrix(df, pca_dims)
        if pc_matrix is not None:
            X_std = pc_matrix
        else:
            k_pca = min(pca_dims, n_samples, n_features)
            U, S, _Vt = np.linalg.svd(X_std, full_matrices=False)
            X_std = U[:, :k_pca] * S[:k_pca]

    # t-SNE requires perplexity < n_samples.  Cap defensively.
    perp = min(float(perplexity), n_samples - 1)
    if perp < 2:
        return compute_pca(df, n_components, features)

    tsne = TSNE(
        n_components=n_components,
        perplexity=perp,
        random_state=42,
        init="pca",
        metric=metric,
    )
    projected = tsne.fit_transform(X_std)

    if projected.shape[1] < n_components:
        pad = np.zeros((n_samples, n_components - projected.shape[1]))
        projected = np.hstack([projected, pad])

    return projected


def compute_umap_embedding(
    df: pd.DataFrame,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    features: Optional[List[str]] = None,
    pca_dims: Optional[int] = None,
    metric: str = "euclidean",
) -> pd.DataFrame:
    """Compute 2D UMAP and write ``UMAP_1`` / ``UMAP_2`` columns to ``df``.

    Convenience wrapper around :func:`compute_umap` that persists the
    embedding back into the feature frame so it round-trips through
    :func:`save_patch_csv` and the scatter plot can render it later
    without re-computing.

    The columns are added in-place and the same DataFrame is returned
    for chainability.
    """
    coords = compute_umap(
        df,
        n_components=2,
        features=features,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        pca_dims=pca_dims,
        metric=metric,
    )
    df["UMAP_1"] = coords[:, 0]
    df["UMAP_2"] = coords[:, 1]
    return df


def compute_tsne_embedding(
    df: pd.DataFrame,
    *,
    perplexity: float = 30.0,
    features: Optional[List[str]] = None,
    pca_dims: Optional[int] = None,
    metric: str = "euclidean",
) -> pd.DataFrame:
    """Compute 2D t-SNE and write ``tSNE_1`` / ``tSNE_2`` columns to ``df``.

    Sibling of :func:`compute_umap_embedding`.  Same persist-to-CSV
    contract.
    """
    coords = compute_tsne(
        df,
        n_components=2,
        features=features,
        perplexity=perplexity,
        pca_dims=pca_dims,
        metric=metric,
    )
    df["tSNE_1"] = coords[:, 0]
    df["tSNE_2"] = coords[:, 1]
    return df


def compute_pca(
    df: pd.DataFrame,
    n_components: int = 2,
    features: Optional[List[str]] = None,
    exclude_border: bool = False,
    mode: str = "qc",
) -> np.ndarray:
    """PCA projection (backward-compatible wrapper).

    Returns only projected coordinates.  For loadings + explained variance,
    use :func:`compute_pca_full`.
    """
    result = compute_pca_full(
        df, n_components=n_components, features=features,
        exclude_border=exclude_border, mode=mode,
    )
    return result["projected"]


def compute_pca_full(
    df: pd.DataFrame,
    n_components: int = 10,
    features: Optional[List[str]] = None,
    exclude_border: bool = False,
    mode: str = "qc",
    transforms: Optional[dict] = None,
    winsorize: Optional[float] = None,
) -> dict:
    """PCA projection with loadings and explained variance.

    Args:
        df: Feature DataFrame.
        n_components: Number of principal components (default 10).
        features: Feature columns.  Defaults to PCA_FEATURES.
        exclude_border: Exclude border cells from fitting.
        mode: "qc" (robust) or "eda" (standard).
        transforms: Per-feature transform dict from the EDA table
            (e.g. ``{"area": "log", "extent": "raw"}``).  When provided,
            these are applied instead of the hardcoded LOG_TRANSFORM_FEATURES.
            Only used in ``"eda"`` mode; ``"qc"`` mode uses its own
            validated pipeline.

    Returns:
        dict with keys:
            projected : np.ndarray (n_samples, n_components)
            loadings  : np.ndarray (n_components, n_features) — Vt matrix
            explained_variance_ratio : np.ndarray (n_components,)
            feature_names : list of str — column names used
    """
    src = features if features is not None else PCA_FEATURES
    cols = [c for c in src if c in df.columns]
    n_samples = len(df)

    empty = {
        "projected": np.zeros((n_samples, n_components)),
        "loadings": np.zeros((n_components, max(len(cols), 1))),
        "explained_variance_ratio": np.zeros(n_components),
        "feature_names": cols,
    }
    if not cols or n_samples < 2:
        return empty

    X = df[cols].values.astype(np.float64)
    n_features = X.shape[1]

    if n_features < 1:
        return empty

    # Step 1: Apply transforms
    if transforms:
        # User-specified per-feature transforms from EDA table
        from napari_epicycle.segmentation.feature_stats import apply_transform
        for j, col in enumerate(cols):
            method = transforms.get(col, "auto")
            X[:, j] = apply_transform(X[:, j], method, feature_name=col)
        # Clean NaN/Inf after transforms
        for j in range(X.shape[1]):
            bad = ~np.isfinite(X[:, j])
            if bad.any():
                X[bad, j] = np.nanmedian(X[~bad, j]) if (~bad).any() else 0.0

    # Step 2: Exclude border cells from fitting (always)
    exclude = None
    if "is_border_cell" in df.columns:
        exclude = df["is_border_cell"].values.astype(bool)
    elif exclude_border:
        exclude = None  # no column to exclude from

    if mode == "eda":
        if not transforms:
            # No user transforms — apply hardcoded log for known skewed features
            for j, col in enumerate(cols):
                if col in LOG_TRANSFORM_FEATURES:
                    X[:, j] = np.log1p(np.clip(X[:, j], 0, None))
            for j in range(X.shape[1]):
                bad = ~np.isfinite(X[:, j])
                if bad.any():
                    X[bad, j] = np.nanmedian(X[~bad, j]) if (~bad).any() else 0.0
        # Standard z-score (fit on non-border rows)
        fit_mask = ~exclude if exclude is not None and exclude.any() else None
        fit_X = X[fit_mask] if fit_mask is not None else X
        mean = fit_X.mean(axis=0)
        std = fit_X.std(axis=0)
        std[std < 1e-12] = 1.0
        X_std = (X - mean) / std
        # Optional winsorization
        if winsorize is not None:
            X_std = np.clip(X_std, -winsorize, winsorize)
    else:
        if not transforms:
            # QC mode without user transforms — use full _robust_standardize
            # (includes its own log + skewness detection + MAD + winsorize ±10)
            X_std = _robust_standardize(X, cols, exclude_mask=exclude)
        else:
            # QC mode WITH user transforms — transforms already applied above,
            # just do robust z-score (median/MAD) + winsorize
            fit_mask = ~exclude if exclude is not None and exclude.any() else None
            fit_X = X[fit_mask] if fit_mask is not None else X
            for j in range(X.shape[1]):
                med = np.median(fit_X[:, j])
                mad = np.median(np.abs(fit_X[:, j] - med))
                scale = mad * 1.4826
                if scale < 1e-12:
                    X[:, j] = 0.0
                else:
                    X[:, j] = (X[:, j] - med) / scale
            X_std = X
            # Winsorize only if explicitly requested
            if winsorize is not None:
                X_std = np.clip(X_std, -winsorize, winsorize)

    # SVD — fit on non-excluded rows
    if exclude is not None and exclude.any():
        fit_X = X_std[~exclude]
        if fit_X.shape[0] < 2:
            fit_X = X_std
    else:
        fit_X = X_std

    k = min(n_components, fit_X.shape[0], n_features)
    U, S, Vt = np.linalg.svd(fit_X, full_matrices=False)

    # Explained variance ratio
    var_explained = (S[:k] ** 2) / max(np.sum(S ** 2), 1e-12)

    # Project ALL rows
    projected = X_std @ Vt[:k].T

    # Pad if fewer components than requested
    if k < n_components:
        projected = np.hstack([
            projected, np.zeros((n_samples, n_components - k))
        ])
        var_explained = np.concatenate([
            var_explained, np.zeros(n_components - k)
        ])
        Vt_out = np.vstack([
            Vt[:k], np.zeros((n_components - k, n_features))
        ])
    else:
        Vt_out = Vt[:k]

    return {
        "projected": projected,
        "loadings": Vt_out,
        "explained_variance_ratio": var_explained[:n_components],
        "feature_names": cols,
    }


# Default size features to regress out in Residual PCA
RESIDUAL_PCA_REGRESS_OUT = [
    "area", "equivalent_diameter", "surface_area",
    "bbox_volume", "axis_major_length", "axis_minor_length",
]


def compute_residual_pca(
    df: pd.DataFrame,
    n_components: int = 2,
    features: Optional[List[str]] = None,
    regress_out: Optional[List[str]] = None,
    exclude_border: bool = True,
    transforms: Optional[dict] = None,
    scaling: str = "eda",
    winsorize: Optional[float] = None,
) -> tuple:
    """Residual PCA — regress out confounders, then PCA on residuals.

    Removes the linear effect of ``regress_out`` features (typically size)
    from all analysis features before computing PCA.  This reveals shape
    and polarity variation hidden behind the dominant size gradient.

    Args:
        df: Feature DataFrame.
        n_components: Number of principal components.
        features: Feature columns to analyze.  Defaults to PCA_FEATURES.
        regress_out: Features to regress out.  Defaults to
            RESIDUAL_PCA_REGRESS_OUT (size features).
        exclude_border: Exclude border cells from fitting (projected as
            supplementary points).

    Returns:
        dict with keys:
            projected : np.ndarray (n_samples, n_components)
            loadings  : np.ndarray (n_components, n_features)
            explained_variance_ratio : np.ndarray (n_components,)
            feature_names : list of str — analyze columns used
            regress_cols_used : list of str
    """
    if regress_out is None:
        regress_out = list(RESIDUAL_PCA_REGRESS_OUT)

    src = features if features is not None else PCA_FEATURES
    analyze_cols = [c for c in src if c in df.columns and c not in regress_out]
    regress_cols = [c for c in regress_out if c in df.columns]

    n_samples = len(df)
    n_feat = max(len(analyze_cols), 1)
    empty = {
        "projected": np.zeros((n_samples, n_components)),
        "loadings": np.zeros((n_components, n_feat)),
        "explained_variance_ratio": np.zeros(n_components),
        "feature_names": analyze_cols,
        "regress_cols_used": regress_cols,
    }

    if not analyze_cols:
        return empty
    if not regress_cols:
        result = compute_pca_full(df, n_components=n_components, features=src,
                                  exclude_border=exclude_border, mode=scaling,
                                  transforms=transforms, winsorize=winsorize)
        result["regress_cols_used"] = regress_cols
        return result

    X_analyze = df[analyze_cols].values.astype(np.float64)
    X_regress = df[regress_cols].values.astype(np.float64)

    if n_samples < 2 or X_analyze.shape[1] < 1:
        return empty

    # Clean NaN/Inf
    for arr in (X_analyze, X_regress):
        for j in range(arr.shape[1]):
            bad = ~np.isfinite(arr[:, j])
            if bad.any():
                med = np.nanmedian(arr[~bad, j]) if (~bad).any() else 0.0
                arr[bad, j] = med

    # Apply per-feature transforms (user-specified or hardcoded fallback)
    if transforms:
        from napari_epicycle.segmentation.feature_stats import apply_transform
        for j, col in enumerate(analyze_cols):
            method = transforms.get(col, "auto")
            X_analyze[:, j] = apply_transform(
                X_analyze[:, j], method, feature_name=col,
            )
        for j, col in enumerate(regress_cols):
            method = transforms.get(col, "auto")
            X_regress[:, j] = apply_transform(
                X_regress[:, j], method, feature_name=col,
            )
    else:
        for j, col in enumerate(analyze_cols):
            if col in LOG_TRANSFORM_FEATURES:
                X_analyze[:, j] = np.log1p(np.clip(X_analyze[:, j], 0, None))
        for j, col in enumerate(regress_cols):
            if col in LOG_TRANSFORM_FEATURES:
                X_regress[:, j] = np.log1p(np.clip(X_regress[:, j], 0, None))

    # Determine fit mask (exclude border cells from fitting)
    exclude = None
    if exclude_border and "is_border_cell" in df.columns:
        exclude = df["is_border_cell"].values.astype(bool)

    if exclude is not None and exclude.any():
        fit_mask = ~exclude
        if fit_mask.sum() < 2:
            fit_mask = np.ones(n_samples, dtype=bool)
    else:
        fit_mask = np.ones(n_samples, dtype=bool)

    # Standardize both sets (using user's scaling choice)
    def _scale(X):
        fit_X = X[fit_mask]
        if scaling == "qc":
            # Robust: median/MAD
            for j in range(X.shape[1]):
                med = np.median(fit_X[:, j])
                mad = np.median(np.abs(fit_X[:, j] - med))
                s = mad * 1.4826
                if s < 1e-12:
                    X[:, j] = 0.0
                else:
                    X[:, j] = (X[:, j] - med) / s
        else:
            # Standard: mean/std
            mean = fit_X.mean(axis=0)
            std = fit_X.std(axis=0)
            std[std < 1e-12] = 1.0
            X[:] = (X - mean) / std
        if winsorize is not None:
            X[:] = np.clip(X, -winsorize, winsorize)
        return X

    X_analyze = _scale(X_analyze)
    X_regress = _scale(X_regress)

    # Regress out: fit on non-excluded, apply to ALL rows
    # X_analyze = X_regress @ beta + residuals
    X_reg_fit = X_regress[fit_mask]
    X_ana_fit = X_analyze[fit_mask]
    beta, _, _, _ = np.linalg.lstsq(X_reg_fit, X_ana_fit, rcond=None)
    residuals = X_analyze - X_regress @ beta

    # SVD on residuals (fit on non-excluded)
    fit_residuals = residuals[fit_mask]
    k = min(n_components, fit_residuals.shape[0], fit_residuals.shape[1])
    if k < 1:
        return empty

    U, S, Vt = np.linalg.svd(fit_residuals, full_matrices=False)

    # Explained variance ratio
    var_explained = (S[:k] ** 2) / max(np.sum(S ** 2), 1e-12)

    # Project ALL rows (including border cells)
    projected = residuals @ Vt[:k].T

    # Pad if fewer components than requested
    if k < n_components:
        projected = np.hstack([
            projected, np.zeros((n_samples, n_components - k))
        ])
        var_explained = np.concatenate([
            var_explained, np.zeros(n_components - k)
        ])
        Vt_out = np.vstack([
            Vt[:k], np.zeros((n_components - k, len(analyze_cols)))
        ])
    else:
        Vt_out = Vt[:k]

    return {
        "projected": projected,
        "loadings": Vt_out,
        "explained_variance_ratio": var_explained[:n_components],
        "feature_names": analyze_cols,
        "regress_cols_used": regress_cols,
    }


# ---------------------------------------------------------------------------
# EDA state persistence (PCA, UMAP, t-SNE, clustering params)
# ---------------------------------------------------------------------------

_EDA_SCHEMA_VERSION = 3  # v3: adds ui_state (selected features, transforms, preset)


def _label_hash(df: pd.DataFrame) -> str:
    """Deterministic hash of sorted label IDs for staleness detection.

    Uses MD5 (not Python's hash()) because hash() is randomized across
    Python sessions via PYTHONHASHSEED.
    """
    import hashlib
    labels = np.sort(df["label"].values.astype(np.int64))
    return hashlib.md5(labels.tobytes()).hexdigest()


def save_eda_state(
    eda_dir: Union[str, Path],
    stack_name: str,
    *,
    pca_result: Optional[dict] = None,
    pca_scatter_cache: Optional[np.ndarray] = None,
    umap_cache: Optional[np.ndarray] = None,
    tsne_cache: Optional[np.ndarray] = None,
    df: Optional[pd.DataFrame] = None,
    params: Optional[dict] = None,
    ui_state: Optional[dict] = None,
) -> None:
    """Save EDA analysis state (PCA, UMAP, t-SNE) to disk.

    Creates ``{eda_dir}/{stack_name}_eda.npz`` (arrays) and
    ``{eda_dir}/{stack_name}_eda.json`` (metadata).  Uses atomic
    writes to prevent corruption from interrupted saves.

    Parameters
    ----------
    eda_dir : path
        Directory for EDA state files (created if needed).
    stack_name : str
        Stack identifier.
    pca_result : dict, optional
        PCA result dict with keys: projected, loadings,
        explained_variance_ratio, feature_names, regress_cols_used.
    umap_cache : ndarray, optional
        2D UMAP projection (N, 2).
    tsne_cache : ndarray, optional
        2D t-SNE projection (N, 2).
    df : DataFrame, optional
        Current features DataFrame (used for staleness hash).
    params : dict, optional
        Analysis parameters to persist (n_components, method, etc.).
    ui_state : dict, optional
        EDA UI state to persist.  Keys: ``selected_features`` (list),
        ``transforms`` (dict), ``preset_name`` (str), ``pca_mode`` (str).
    """
    import json
    import os
    import tempfile

    eda_dir = Path(eda_dir)
    eda_dir.mkdir(parents=True, exist_ok=True)

    npz_path = eda_dir / f"{stack_name}_eda.npz"
    json_path = eda_dir / f"{stack_name}_eda.json"

    # Build arrays dict
    arrays = {}
    if pca_result is not None:
        arrays["pca_projected"] = pca_result["projected"]
        arrays["pca_loadings"] = pca_result["loadings"]
        arrays["pca_var_ratio"] = pca_result["explained_variance_ratio"]
    if pca_scatter_cache is not None:
        arrays["pca_scatter_2d"] = pca_scatter_cache
    if umap_cache is not None:
        arrays["umap_2d"] = umap_cache
    if tsne_cache is not None:
        arrays["tsne_2d"] = tsne_cache

    if not arrays:
        return  # nothing to save

    # Atomic write NPZ
    fd, tmp_npz = tempfile.mkstemp(suffix=".npz", dir=str(eda_dir))
    os.close(fd)
    try:
        np.savez_compressed(tmp_npz, **arrays)
        os.replace(tmp_npz, str(npz_path))
    except Exception:
        if Path(tmp_npz).exists():
            os.unlink(tmp_npz)
        raise

    # Build metadata
    meta = {
        "schema_version": _EDA_SCHEMA_VERSION,
        "n_rows": len(df) if df is not None else 0,
        "label_hash": _label_hash(df) if df is not None else 0,
        "has_pca": pca_result is not None,
        "has_umap": umap_cache is not None,
        "has_tsne": tsne_cache is not None,
    }
    if pca_result is not None:
        meta["feature_names"] = list(pca_result.get("feature_names", []))
        meta["n_components"] = int(pca_result["projected"].shape[1])
        regress = pca_result.get("regress_cols_used")
        if regress:
            meta["pca_method"] = "residual"
            meta["regress_cols_used"] = list(regress)
        else:
            meta["pca_method"] = "standard"
    if params:
        meta["params"] = params
    if ui_state:
        meta["ui_state"] = ui_state

    # Atomic write JSON
    fd, tmp_json = tempfile.mkstemp(suffix=".json", dir=str(eda_dir))
    os.close(fd)
    try:
        with open(tmp_json, "w") as f:
            json.dump(meta, f, indent=2)
        os.replace(tmp_json, str(json_path))
    except Exception:
        if Path(tmp_json).exists():
            os.unlink(tmp_json)
        raise


def load_eda_state(
    eda_dir: Union[str, Path],
    stack_name: str,
    df: Optional[pd.DataFrame] = None,
) -> Optional[dict]:
    """Load EDA analysis state from disk.

    Returns None if files are missing, corrupt, or stale (row count /
    label hash mismatch vs ``df``).

    Parameters
    ----------
    eda_dir : path
        Directory containing EDA state files.
    stack_name : str
        Stack identifier.
    df : DataFrame, optional
        Current features DataFrame for staleness check.

    Returns
    -------
    dict or None
        Keys: pca_result (dict or None), umap_cache (ndarray or None),
        tsne_cache (ndarray or None), params (dict), meta (dict).
    """
    import json

    eda_dir = Path(eda_dir)
    npz_path = eda_dir / f"{stack_name}_eda.npz"
    json_path = eda_dir / f"{stack_name}_eda.json"

    if not npz_path.exists() or not json_path.exists():
        return None

    # Load metadata
    try:
        with open(json_path) as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    if meta.get("schema_version") not in (2, _EDA_SCHEMA_VERSION):
        return None

    # Staleness check
    if df is not None:
        if meta.get("n_rows", -1) != len(df):
            return None
        if meta.get("label_hash", -1) != _label_hash(df):
            return None

    # Load arrays
    try:
        data = np.load(str(npz_path))
    except Exception:
        return None

    result = {
        "pca_result": None,
        "pca_scatter_cache": data["pca_scatter_2d"] if "pca_scatter_2d" in data else None,
        "umap_cache": None,
        "tsne_cache": None,
        "params": meta.get("params", {}),
        "meta": meta,
    }

    if meta.get("has_pca") and "pca_projected" in data:
        result["pca_result"] = {
            "projected": data["pca_projected"],
            "loadings": data["pca_loadings"],
            "explained_variance_ratio": data["pca_var_ratio"],
            "feature_names": meta.get("feature_names", []),
            "regress_cols_used": meta.get("regress_cols_used"),
        }

    if meta.get("has_umap") and "umap_2d" in data:
        result["umap_cache"] = data["umap_2d"]

    if meta.get("has_tsne") and "tsne_2d" in data:
        result["tsne_cache"] = data["tsne_2d"]

    result["ui_state"] = meta.get("ui_state")  # None for v2 files

    return result
