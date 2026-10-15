"""Pure-Python per-cell QC / morphometric feature extraction (Wave 1).

Lifted **verbatim** (parity guardrail — same names, same math) from the EpiCycle
monorepo ``napari_epicycle/fusion_reconstruction/qc_features.py`` so that the
engine's whole-tissue feature table joins the plugin's per-patch table without a
later reconcile (Track-1 Wave 1).

Wave-1 scope (labels-only morphology + cheap crowding):

* :func:`extract_features` — regionprops-derived shape / orientation / ellipsoid.
* :func:`compute_surface_features` — marching-cubes surface area + true sphericity
  + roughness + fast convex-hull solidity.
* :func:`compute_discrete_compactness` — Bribiesca voxel-face compactness.
* :func:`compute_skeleton_features` — EDT-medial-axis branches / nodes / length.
* :func:`compute_cross_section_features` — Z-stratified apical/basal profile.
* :func:`compute_border_cells` — touches-volume-boundary flag.
* :func:`compute_centroid_spacing` — **new for the engine**: cKDTree centroid
  crowding (the one genuinely-new Wave-1 addition the plan calls for).

These are pure (no Qt / napari / engine-stage imports) so they unit-test trivially
and are called per-cell on bbox crops by :class:`~epicycle.core.feature_stage.FeatureStage`.

Deliberate divergences from the monorepo source (documented for the parity
spot-check):

* the misnamed ``compactness`` column (== ``extent`` == V / V_bbox) is **dropped**
  — :func:`compute_discrete_compactness` is the real shape measure.
* the Step-3 curation ``feature_state = "fresh"`` staleness flag is **dropped** —
  it is a plugin-UI concept with no meaning in a batch engine run.

Wave 2 (the gap-aware contact graph) and Wave 3 (tissue-SDF) are NOT here yet — see
the plan. The contact-graph's ``nearest_neighbor_distance`` is a *surface-gap width*;
this module's centroid crowding uses ``nn_centroid_distance`` to avoid colliding
with that reserved name.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

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

# Columns in the output DataFrame from extract_features (after derivation).
# NOTE: the legacy "compactness" column is DROPPED (it equalled "extent");
# discrete_compactness (a separate function) is the real shape measure.
# This is the MINIMAL (empty-result) schema — a populated extract_features result
# ALSO carries the 9 orientation-vector columns (major/mid/minor_axis_{z,y,x}) and
# the 3 inertia_eigval_* columns (matching the monorepo's populated-vs-empty
# asymmetry verbatim). Don't treat this list as the full populated column set.
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
]


def extract_features(
    labels_3d: np.ndarray,
    intensity_image: Optional[np.ndarray] = None,
    spacing: Optional[Tuple[float, ...]] = None,
) -> pd.DataFrame:
    """Extract morphological features from a 3D label volume.

    Computes ~20 features per label including shape, orientation, ellipsoid
    solidity, and optionally intensity statistics. Additional features
    (surface, skeleton, cross-section, border, centroid-spacing) are added
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

    # NOTE: the legacy "compactness" column (vol / bbox_vol == extent) is
    # intentionally NOT computed — discrete_compactness is the real measure.

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


def _object_edt(
    mask: np.ndarray,
    spacing: Optional[Tuple[float, ...]] = None,
) -> np.ndarray:
    """Per-object interior Euclidean distance transform (M1 / primitive P1).

    The distance from every interior voxel of ``mask`` to the nearest BACKGROUND
    voxel — i.e. how deep inside the object each voxel sits (a local thickness /
    radius field). Background voxels are 0; the per-object maximum is the radius
    of the deepest core point.

    Implementation detail (the parity guardrail for the skeleton refactor): the
    mask is padded with one voxel of background on every side BEFORE the EDT so
    that voxels touching the crop face still measure distance to a true boundary
    (rather than treating the crop edge as interior), then the padding is
    stripped — exactly the ``np.pad(..., 1) -> distance_transform_edt -> strip``
    idiom :func:`compute_skeleton_features` used as a transient. Refactoring that
    transient through this helper keeps the skeleton output byte-identical when
    called with ``spacing=None`` (the skeleton's historical isotropic behaviour).

    Args:
        mask: boolean (or 0/1) 3D array — a single isolated object crop.
        spacing: voxel spacing ``(Z, Y, X)`` for an anisotropy-aware transform
            (``sampling=spacing``). ``None`` = isotropic (voxel units), which is
            what the skeleton uses.

    Returns:
        float64 array the same shape as ``mask`` — the interior EDT (0 outside
        the object). Distances are in physical units when ``spacing`` is given,
        voxels otherwise.
    """
    from scipy.ndimage import distance_transform_edt

    m = np.asarray(mask).astype(bool)
    padded = np.pad(m, 1, mode="constant", constant_values=False)
    if spacing is not None:
        dt_padded = distance_transform_edt(padded, sampling=tuple(spacing))
    else:
        dt_padded = distance_transform_edt(padded)
    return np.asarray(dt_padded[1:-1, 1:-1, 1:-1], dtype=np.float64)


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

        # EDT-based medial axis: local maxima of the (isotropic) interior
        # distance transform. ``_object_edt`` does the pad-with-background ->
        # distance_transform_edt -> strip idiom this code used inline; with
        # spacing=None it is byte-identical to the historical transient.
        dt = _object_edt(mask_crop, spacing=None)
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


# Cross-section A(z) peak-count prominence floor (C8): a candidate peak must rise
# at least this fraction of the profile's area RANGE above its valleys to count as
# a lobe, so voxel-jitter wobbles don't inflate cross_section_peak_count.
_CROSS_SECTION_PEAK_PROMINENCE_FRAC = 0.05


def compute_cross_section_profile_stats(
    labels_3d: np.ndarray,
    df: pd.DataFrame,
    invert_z: bool = False,
) -> pd.DataFrame:
    """Cross-section profile *distribution* statistics (M0).

    A sibling of :func:`compute_cross_section_features`. That function captures
    apical/basal proxies + a linear taper/bulge fit; THIS one summarizes the
    *shape of the per-Z area profile* ``A(z)`` itself — its spread, asymmetry,
    waist position, and multi-modality. Cheap, labels-only, no new primitive.

    Like the sibling, ``A(z)`` is the cross-section area on each Z-slice where
    the cell is present (voxel count × the in-plane physical area when
    ``spacing`` was used upstream — these stats are *relative* (range/CV/skew/
    normalized position/peak count), so the unit cancels and we count voxels
    here, matching the sibling's voxel-count profile).

    Features added (6):
        ``min_cross_section_area`` — ``min_z A(z)`` (thinnest slice; a small
            value flags a neck / pinch / hourglass).
        ``cross_section_area_range`` — ``max_z A(z) - min_z A(z)``.
        ``cross_section_area_cv`` — ``std(A) / max(mean(A), eps)`` (profile
            unevenness; 0 for a uniform cylinder).
        ``cross_section_area_skew`` — Fisher skewness of ``A(z)`` over its
            present slices (area biased toward one end vs. balanced).
        ``cross_section_waist_depth`` — normalized Z position of the thinnest
            slice, ``0 = apical .. 1 = basal`` (honours ``invert_z``: the apical
            end is high Z by default, low Z when inverted — so 0 always means
            the apical end).
        ``cross_section_peak_count`` — number of local maxima in the smoothed
            profile (>1 ⇒ lobes / merged objects / bilobed shape).

    **Wave I cross-section REMAINDER** (the feature catalogue; cheap, same ``A(z)`` profile —
    no new primitive, additive only):
        ``cross_section_area_kurtosis`` — Fisher (excess) kurtosis of ``A(z)``
            over its present slices: whether the area sits in a tight bulge
            (high, leptokurtic) vs a broad / multi-modal profile (low). 0 for a
            uniform profile.
        ``min_cross_section_z_relative`` — normalized Z position of the THINNEST
            slice in ``[0, 1]`` over the present Z-extent (0 = the low-Z end,
            1 = the high-Z end). The RAW argmin position (no apical/basal sign
            flip — that's what ``cross_section_waist_depth`` carries). Where the
            waist / neck sits along the stack.
        ``cross_section_constriction_index`` — ``(max A - min A) / max(mean A,
            eps)`` ∈ [0, ∞): the strength of the strongest constriction relative
            to the typical area. ~0 for a uniform cylinder, large for an
            hourglass / dumbbell with a pronounced neck.
        ``cross_section_n_lobes`` — number of distinct LOBES = the prominence-
            gated interior peak count of ``A(z)`` (a synonym surface for
            ``cross_section_peak_count`` under the catalogue's name; >1 ⇒ a
            multi-lobed / merged mask). Equal to ``cross_section_peak_count``.

    Degenerate shapes (single present slice, empty crop) return NaN for the
    distribution-shape stats rather than crashing; ``peak_count`` is 1 for a
    single present slice and ``min_cross_section_area`` is that slice's area.

    Args:
        labels_3d: 3D integer label array (Z, Y, X). Background is 0.
        df: DataFrame from :func:`extract_features` (modified in-place).
        invert_z: If True, treat high Z as basal and low Z as apical (matches
            the sibling's convention) — only affects ``cross_section_waist_depth``.

    Returns:
        The same DataFrame with the 6 new columns.
    """
    from scipy.ndimage import find_objects, gaussian_filter1d
    from scipy.signal import find_peaks
    from scipy.stats import skew as _skew, kurtosis as _kurtosis

    n = len(df)
    cols = (
        "min_cross_section_area", "cross_section_area_range",
        "cross_section_area_cv", "cross_section_area_skew",
        "cross_section_waist_depth", "cross_section_peak_count",
        # Wave I cross-section remainder.
        "cross_section_area_kurtosis", "min_cross_section_z_relative",
        "cross_section_constriction_index", "cross_section_n_lobes",
        # Wave N cross-section remainder — basal-flare proxy, peak/trough
        # prominence, and per-slice mask moments (centroid drift, eccentricity
        # profile + taper, orientation twist).
        "basal_flare_index",
        "cross_section_peak_prominence", "cross_section_trough_prominence",
        "slice_centroid_drift_mean", "slice_centroid_drift_max",
        "slice_eccentricity_mean", "slice_eccentricity_taper",
        "slice_orientation_twist",
    )
    if n == 0:
        for col in cols:
            df[col] = pd.Series(dtype=np.float64)
        return df

    min_area = np.full(n, np.nan, dtype=np.float64)
    area_range = np.full(n, np.nan, dtype=np.float64)
    area_cv = np.full(n, np.nan, dtype=np.float64)
    area_skew = np.full(n, np.nan, dtype=np.float64)
    waist_depth = np.full(n, np.nan, dtype=np.float64)
    peak_count = np.full(n, np.nan, dtype=np.float64)
    # Wave I cross-section remainder.
    area_kurt = np.full(n, np.nan, dtype=np.float64)
    min_z_rel = np.full(n, np.nan, dtype=np.float64)
    constriction = np.full(n, np.nan, dtype=np.float64)
    n_lobes = np.full(n, np.nan, dtype=np.float64)
    # Wave N cross-section remainder.
    basal_flare = np.full(n, np.nan, dtype=np.float64)
    peak_prom = np.full(n, np.nan, dtype=np.float64)
    trough_prom = np.full(n, np.nan, dtype=np.float64)
    cdrift_mean = np.full(n, np.nan, dtype=np.float64)
    cdrift_max = np.full(n, np.nan, dtype=np.float64)
    ecc_mean = np.full(n, np.nan, dtype=np.float64)
    ecc_taper = np.full(n, np.nan, dtype=np.float64)
    orient_twist = np.full(n, np.nan, dtype=np.float64)

    slices = find_objects(labels_3d.astype(np.int32))
    label_ids = df["label"].values
    eps = 1e-6

    for idx, label_id in enumerate(label_ids):
        if label_id < 1 or label_id > len(slices):
            continue
        obj_slice = slices[label_id - 1]
        if obj_slice is None:
            continue

        crop = labels_3d[obj_slice]
        mask = crop == label_id
        z_size = mask.shape[0]

        # Per-Z areas restricted to slices where the cell exists.
        areas_full = np.array([float(mask[z].sum()) for z in range(z_size)])
        present = areas_full > 0
        if not present.any():
            continue
        z_indices = np.where(present)[0]
        a = areas_full[z_indices]  # present-only profile

        # min + range are always defined when >=1 present slice.
        min_area[idx] = float(a.min())
        area_range[idx] = float(a.max() - a.min())

        z_min, z_max = z_indices[0], z_indices[-1]
        z_range = z_max - z_min

        if a.size < 2 or z_range == 0:
            # Single present slice: profile-shape stats are undefined.
            area_cv[idx] = 0.0
            area_skew[idx] = 0.0
            waist_depth[idx] = 0.5
            peak_count[idx] = 1.0
            # Wave I: a single-slice profile has no constriction / no kurtosis;
            # the thinnest slice IS the only slice (mid-position 0.5); 1 lobe.
            area_kurt[idx] = 0.0
            min_z_rel[idx] = 0.5
            constriction[idx] = 0.0
            n_lobes[idx] = 1.0
            # Wave N: a single-slice profile has no flare/peak/drift/twist
            # structure; report the documented neutral degenerate values.
            basal_flare[idx] = 1.0      # basal == mean (single slice)
            peak_prom[idx] = 0.0
            trough_prom[idx] = 0.0
            cdrift_mean[idx] = 0.0
            cdrift_max[idx] = 0.0
            # Eccentricity of the single slice is still defined (one slice).
            ecc_mean[idx] = _slice_mask_eccentricity(mask[z_indices[0]])
            ecc_taper[idx] = 0.0
            orient_twist[idx] = 0.0
            continue

        mean_a = float(a.mean())
        std_a = float(a.std())  # population std (ddof=0), matches sibling bulge RMS
        area_cv[idx] = std_a / max(mean_a, eps)

        # Fisher skewness; constant profile (std≈0) → 0, not NaN.
        if std_a > eps:
            area_skew[idx] = float(_skew(a))
        else:
            area_skew[idx] = 0.0

        # Wave I — Fisher (excess) kurtosis of A(z); constant profile -> 0.
        if std_a > eps and a.size >= 4:
            area_kurt[idx] = float(_kurtosis(a, fisher=True))
        else:
            area_kurt[idx] = 0.0

        # Wave I — constriction index = (max - min) / mean. Strength of the
        # strongest constriction relative to the typical area (0 = uniform,
        # large = a pronounced neck / hourglass).
        constriction[idx] = (float(a.max()) - float(a.min())) / max(mean_a, eps)

        # Waist = thinnest slice. Position normalized to [0,1] over the present
        # Z-extent, then mapped so 0=apical, 1=basal.
        argmin_local = int(np.argmin(a))
        z_at_min = z_indices[argmin_local]
        frac_from_low = (z_at_min - z_min) / max(z_range, 1)
        # Wave I — RAW thinnest-slice position (0 = low-Z end, 1 = high-Z end),
        # NO apical/basal sign flip (waist_depth carries the signed version).
        min_z_rel[idx] = frac_from_low
        # Default: high Z = apical -> apical end is frac 1.0 -> invert to get
        # 0=apical. Inverted: low Z = apical -> apical end is frac 0.0 already.
        waist_depth[idx] = frac_from_low if invert_z else (1.0 - frac_from_low)

        # Peak count on the smoothed present-only profile. A bulge at either
        # END of the cell (e.g. a dumbbell with both lobes at the extremes) is
        # a real peak, but find_peaks ignores boundary maxima — so pad both
        # ends with a value below the profile minimum, turning a boundary
        # plateau into a detectable interior peak. plateau_size=1 counts a flat
        # lobe top as a single peak.
        if a.size >= 3:
            smoothed = gaussian_filter1d(a, sigma=1.0, mode="nearest")
            pad_val = float(smoothed.min()) - 1.0
            padded = np.concatenate(([pad_val], smoothed, [pad_val]))
            # PROMINENCE FLOOR (C8): without it, find_peaks counts every tiny
            # voxelization wobble in A(z) as a separate lobe, inflating the count
            # on a smooth cell. Require each peak to rise at least a small fraction
            # of the profile's own area RANGE above its surrounding valleys, so
            # only genuine lobes (real bulges) are counted. The range-relative
            # floor self-scales to the cell, and degrades to 0 (no floor) for a
            # flat profile (range 0) so a single-bump cell still reports >= 1.
            prof_range = float(smoothed.max() - smoothed.min())
            prom = _CROSS_SECTION_PEAK_PROMINENCE_FRAC * prof_range
            peaks, _ = find_peaks(
                padded, plateau_size=1,
                prominence=(prom if prom > 0 else None),
            )
            # A monotone or single-bump profile yields exactly one peak after
            # padding; clamp to >=1 so a present cell is never "zero peaks".
            peak_count[idx] = float(max(len(peaks), 1))
        else:
            peak_count[idx] = 1.0
        # Wave I — n_lobes is the same prominence-gated lobe count under the
        # feature catalogue's name; mirror peak_count exactly (one source of truth).
        n_lobes[idx] = peak_count[idx]

        # --- Wave N cross-section remainder -------------------------
        # basal_flare_index = basal_area / mean_area: how expanded the
        # BASAL end is relative to the cell average (>1 = a flared base). The
        # basal end is the low-Z end by default, the high-Z end when invert_z
        # (matches compute_cross_section_features' apical/basal convention).
        # Use the present-only profile `a` (a[0] = lowest present slice).
        basal_area_val = float(a[-1] if invert_z else a[0])
        basal_flare[idx] = basal_area_val / max(mean_a, eps)

        # peak/trough PROMINENCE on the smoothed profile. The largest INTERIOR
        # peak prominence + the largest interior trough (valley) prominence — a
        # deep interior valley flanked by tall lobes is the dumbbell/neck
        # signature. Range-normalized so they are dimensionless + cell-scaled.
        # NOTE: measured on the UN-padded `smoothed` profile (NOT the end-padded
        # one peak_count uses) — on the un-padded profile any interior extremum's
        # scipy prominence is at most the profile range, so the range-normalized
        # value stays in [0, 1] (bounded). The end-padding peak_count uses would
        # inflate the global max's prominence by the +1 pad floor (> 1). Boundary
        # extrema have no prominence here, which is correct: a "peak/trough
        # prominence" is an INTERIOR feature.
        if a.size >= 3:
            prof_range_n = float(smoothed.max() - smoothed.min())
            if prof_range_n > eps:
                pk, pk_props = find_peaks(smoothed, prominence=0.0)
                if pk.size > 0:
                    peak_prom[idx] = min(float(
                        pk_props["prominences"].max()) / prof_range_n, 1.0)
                else:
                    peak_prom[idx] = 0.0
                tr, tr_props = find_peaks(-smoothed, prominence=0.0)
                if tr.size > 0:
                    trough_prom[idx] = min(float(
                        tr_props["prominences"].max()) / prof_range_n, 1.0)
                else:
                    trough_prom[idx] = 0.0
            else:
                peak_prom[idx] = 0.0
                trough_prom[idx] = 0.0
        else:
            peak_prom[idx] = 0.0
            trough_prom[idx] = 0.0

        # --- per-slice mask MOMENTS (needs_primitives) ----------------
        # For each present slice compute the 2D mask centroid + the in-plane
        # second-moment ellipse (eccentricity + orientation). Summaries:
        #   slice_centroid_drift_* : how far the per-slice centroid wanders from
        #       the cell's overall lateral centre, normalized by the equivalent
        #       lateral radius (a leaning / banana cell drifts; a straight one
        #       does not).
        #   slice_eccentricity_mean / _taper : mean in-plane eccentricity over
        #       slices + its linear trend vs normalized Z (does the slice get
        #       rounder/flatter from base to apex?).
        #   slice_orientation_twist : total absolute change in the in-plane
        #       major-axis ANGLE between consecutive slices (a twisted cell
        #       accumulates angle; a straight prism stays ~0). In [0, pi]·(n-1)
        #       worst case; reported in radians.
        cen_y = np.full(z_indices.size, np.nan)
        cen_x = np.full(z_indices.size, np.nan)
        eccs = np.full(z_indices.size, np.nan)
        angs = np.full(z_indices.size, np.nan)
        for k, z in enumerate(z_indices):
            sl = mask[z]
            cy, cx, ecc, ang = _slice_mask_moments(sl)
            cen_y[k] = cy
            cen_x[k] = cx
            eccs[k] = ecc
            angs[k] = ang
        # centroid drift: distance of each slice centroid from the mean slice
        # centroid, normalized by the equivalent lateral radius of the mean
        # slice (so it is size-invariant). Use only finite slices.
        fin_c = np.isfinite(cen_y) & np.isfinite(cen_x)
        if fin_c.sum() >= 2:
            cy0 = float(np.mean(cen_y[fin_c]))
            cx0 = float(np.mean(cen_x[fin_c]))
            dr = np.sqrt((cen_y[fin_c] - cy0) ** 2 + (cen_x[fin_c] - cx0) ** 2)
            r_equiv_xy = float(np.sqrt(mean_a / np.pi))
            denom = max(r_equiv_xy, eps)
            cdrift_mean[idx] = float(dr.mean()) / denom
            cdrift_max[idx] = float(dr.max()) / denom
        else:
            cdrift_mean[idx] = 0.0
            cdrift_max[idx] = 0.0
        # eccentricity mean + taper (slope vs normalized z over present slices).
        fin_e = np.isfinite(eccs)
        if fin_e.sum() >= 1:
            ecc_mean[idx] = float(np.mean(eccs[fin_e]))
        if fin_e.sum() >= 2:
            zt = (z_indices[fin_e] - z_indices[0]).astype(np.float64) \
                / max(z_range, 1)
            if np.ptp(zt) > eps:
                ecc_taper[idx] = float(np.polyfit(zt, eccs[fin_e], 1)[0])
            else:
                ecc_taper[idx] = 0.0
        else:
            ecc_taper[idx] = 0.0
        # orientation twist: sum of absolute consecutive angle deltas (each
        # wrapped into [-pi/2, pi/2] since the axis orientation is modulo pi).
        fin_a = np.isfinite(angs)
        ang_seq = angs[fin_a]
        if ang_seq.size >= 2:
            d = np.diff(ang_seq)
            d = (d + np.pi / 2.0) % np.pi - np.pi / 2.0   # wrap to [-pi/2, pi/2]
            orient_twist[idx] = float(np.sum(np.abs(d)))
        else:
            orient_twist[idx] = 0.0

    df["min_cross_section_area"] = min_area
    df["cross_section_area_range"] = area_range
    df["cross_section_area_cv"] = area_cv
    df["cross_section_area_skew"] = area_skew
    df["cross_section_waist_depth"] = waist_depth
    df["cross_section_peak_count"] = peak_count
    # Wave I cross-section remainder.
    df["cross_section_area_kurtosis"] = area_kurt
    df["min_cross_section_z_relative"] = min_z_rel
    df["cross_section_constriction_index"] = constriction
    df["cross_section_n_lobes"] = n_lobes
    # Wave N cross-section remainder.
    df["basal_flare_index"] = basal_flare
    df["cross_section_peak_prominence"] = peak_prom
    df["cross_section_trough_prominence"] = trough_prom
    df["slice_centroid_drift_mean"] = cdrift_mean
    df["slice_centroid_drift_max"] = cdrift_max
    df["slice_eccentricity_mean"] = ecc_mean
    df["slice_eccentricity_taper"] = ecc_taper
    df["slice_orientation_twist"] = orient_twist
    return df


def _slice_mask_moments(sl: np.ndarray):
    """In-plane second-moment ellipse of a 2D boolean slice mask.

    Returns ``(centroid_y, centroid_x, eccentricity, orientation)`` from the
    pixel coordinate covariance of the True pixels:

    * ``centroid_y`` / ``centroid_x`` — the mask's pixel centroid (NaN if empty).
    * ``eccentricity`` — ``sqrt(1 - lambda_min / lambda_max)`` of the 2x2
      coordinate covariance ∈ [0, 1] (0 = a round disc, → 1 = an infinitely thin
      streak). NaN for < 2 pixels (covariance undefined).
    * ``orientation`` — the angle (radians, in ``[-pi/2, pi/2]``) of the major
      eigenvector of the covariance, i.e. the in-plane long-axis direction.
      NaN for < 2 pixels or a degenerate (isotropic) covariance.

    Pure / allocation-light: used by the per-slice moment summaries
    (centroid drift, eccentricity profile, orientation twist).

    ISOTROPY ASSUMPTION: the covariance is computed in PIXEL-INDEX coordinates, so the
    eccentricity + orientation assume IN-PLANE ISOTROPY (sp_y == sp_x — the EpiCycle
    data model: the XY plane is square). For anisotropic in-plane pixels the ellipse
    would need axis-scaled coordinates; this helper does not apply that scaling.
    """
    coords = np.argwhere(np.asarray(sl, dtype=bool))
    if coords.shape[0] == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    cy = float(coords[:, 0].mean())
    cx = float(coords[:, 1].mean())
    if coords.shape[0] < 2:
        return (cy, cx, np.nan, np.nan)
    cov = np.cov(coords.T.astype(np.float64))      # 2x2 (rows = [y, x])
    # Eigen-decomposition of the symmetric covariance.
    evals, evecs = np.linalg.eigh(cov)             # ascending eigenvalues
    lam_min = float(evals[0])
    lam_max = float(evals[1])
    if lam_max <= 1e-12:
        return (cy, cx, 0.0, np.nan)
    ecc = float(np.sqrt(max(0.0, 1.0 - lam_min / lam_max)))
    # Major-axis eigenvector (largest eigenvalue = last column).
    vmaj = evecs[:, 1]                              # (vy, vx)
    ang = float(np.arctan2(vmaj[0], vmaj[1]))       # angle of (y, x) major axis
    # Fold into [-pi/2, pi/2] (an axis is modulo pi).
    ang = (ang + np.pi / 2.0) % np.pi - np.pi / 2.0
    return (cy, cx, ecc, ang)


def _slice_mask_eccentricity(sl: np.ndarray) -> float:
    """The in-plane eccentricity of a single 2D slice mask (helper for the
    degenerate single-slice path); NaN when undefined."""
    return _slice_mask_moments(sl)[2]


def compute_triaxial(
    labels_3d: np.ndarray,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Three-axis (triaxial) shape descriptors (M0).

    "Ready — just expose the mid axis." Reads the EXISTING inertia-derived
    ellipsoid columns that :func:`extract_features` already emitted
    (``inertia_eigval_major/mid/minor`` + ``axis_major_length`` /
    ``axis_minor_length``) and elevates the intermediate axis + two readable
    rod-vs-disc scalars to first-class columns. ``labels_3d`` is accepted for a
    uniform ``compute_*(labels, df)`` signature but is unused (this is a pure
    table-derivation pass — the eigenvalues already encode the geometry).

    Axis lengths are derived the same way as :func:`extract_features`'
    ``axis_major_length`` / ``axis_minor_length``: from the mass-normalized
    inertia eigenvalues ``I0>=I1>=I2`` of a uniform solid ellipsoid with
    semi-axes ``a>=b>=c``::

        a² = 5/2 (I0 + I1 - I2),  b² = 5/2 (I0 - I1 + I2),  c² = 5/2 (-I0 + I1 + I2)

    with ``axis_*_length = 2·sqrt(·)`` (full axis lengths a>=b>=c -> A>=B>=C).

    Features added (3):
        ``axis_mid_length`` — ``2·sqrt(b²)``, the intermediate principal axis
            length (same unit as ``axis_major_length`` / ``axis_minor_length``).
        ``triaxiality`` — ``(A² - B²) / max(A² - C², eps)`` ∈ [0, 1]: ~0 =
            oblate/disc-like (B≈A), ~1 = prolate/rod-like (B≈C). The standard
            ellipsoid triaxiality on the ordered axis lengths.
        ``disc_vs_rod_score`` — signed oblate-vs-prolate score in [-1, 1] from
            the eigenvalue-derived axis ratios:
            ``(B/A) - (C/B)``. >0 ⇒ disc/pancake (broad mid axis, thin minor);
            <0 ⇒ rod/cigar (mid collapses toward minor while major dominates);
            ~0 ⇒ near-spherical or evenly triaxial.

    **Wave I axis-ratio REMAINDER** (the feature catalogue; pure derivations from the SAME
    ordered axis lengths A>=B>=C — no new primitive, additive only):
        ``axis_mid_to_major_ratio`` — ``B / max(A, eps)`` ∈ (0, 1] (bounded):
            how broad the mid axis is relative to the long axis (~1 = a flat
            plate / oblate disc, small = a thin rod where B collapses toward C).
        ``axis_minor_to_mid_ratio`` — ``C / max(B, eps)`` ∈ (0, 1] (bounded):
            how thick the minor axis is relative to the mid axis (~1 = a rod
            with circular cross-section, small = a flat sheet).
        ``shape_anisotropy_3d`` — ``(A - C) / max(A + C, eps)`` ∈ [0, 1]
            (bounded): the Westin-style fractional anisotropy of the three axis
            lengths. 0 = isotropic (sphere, A≈C), ~1 = strongly anisotropic
            (a long thin rod or thin sheet).

    Degenerate labels (missing eigenvalue columns, or A²≈C² so triaxiality is
    undefined) yield NaN rather than crashing.

    Args:
        labels_3d: accepted for signature uniformity; unused.
        df: DataFrame from :func:`extract_features` (modified in-place). Must
            carry the ``inertia_eigval_*`` columns; if absent every new column
            is NaN.

    Returns:
        The same DataFrame with the 6 new columns (3 original + 3 Wave I).
    """
    n = len(df)
    cols = (
        "axis_mid_length", "triaxiality", "disc_vs_rod_score",
        # Wave I axis-ratio remainder.
        "axis_mid_to_major_ratio", "axis_minor_to_mid_ratio",
        "shape_anisotropy_3d",
    )
    if n == 0:
        for col in cols:
            df[col] = pd.Series(dtype=np.float64)
        return df

    eig_cols = ["inertia_eigval_major", "inertia_eigval_mid", "inertia_eigval_minor"]
    if not all(c in df.columns for c in eig_cols):
        for col in cols:
            df[col] = np.nan
        return df

    I0 = df[eig_cols[0]].values.astype(np.float64)   # largest
    I1 = df[eig_cols[1]].values.astype(np.float64)   # middle
    I2 = df[eig_cols[2]].values.astype(np.float64)   # smallest

    # Semi-axis squared a²>=b²>=c² (identical derivation to extract_features).
    a_sq = np.clip(2.5 * (I0 + I1 - I2), 0, None)
    b_sq = np.clip(2.5 * (I0 - I1 + I2), 0, None)
    c_sq = np.clip(2.5 * (-I0 + I1 + I2), 0, None)

    # Full axis lengths A>=B>=C (A,C already emitted by extract_features).
    df["axis_mid_length"] = 2.0 * np.sqrt(b_sq)

    # Work in full-axis-length² (== 4·semi²) — ratios are scale-free so the ×4
    # cancels; this keeps the formula readable in terms of A,B,C.
    A2 = 4.0 * a_sq
    B2 = 4.0 * b_sq
    C2 = 4.0 * c_sq
    A = np.sqrt(A2)
    B = np.sqrt(B2)
    C = np.sqrt(C2)
    eps = 1e-12

    denom = A2 - C2
    with np.errstate(divide="ignore", invalid="ignore"):
        triax = np.where(denom > eps, (A2 - B2) / denom, np.nan)
    df["triaxiality"] = triax

    # disc_vs_rod_score = (B/A) - (C/B): >0 disc-like, <0 rod-like.
    with np.errstate(divide="ignore", invalid="ignore"):
        ba = np.where(A > eps, B / A, np.nan)
        cb = np.where(B > eps, C / B, np.nan)
    df["disc_vs_rod_score"] = ba - cb

    # --- Wave I axis-ratio remainder — pure ratios of A>=B>=C --------
    # ba/cb already computed above (B/A and C/B); reuse them.
    df["axis_mid_to_major_ratio"] = ba
    df["axis_minor_to_mid_ratio"] = cb
    with np.errstate(divide="ignore", invalid="ignore"):
        denom_ac = A + C
        df["shape_anisotropy_3d"] = np.where(
            denom_ac > eps, (A - C) / denom_ac, np.nan)
    return df


def compute_edt_features(
    labels_3d: np.ndarray,
    df: pd.DataFrame,
    spacing: Optional[Tuple[float, ...]] = None,
) -> pd.DataFrame:
    """Interior-EDT thickness-field summaries (M1 / the feature catalogue; C21).

    Summarizes the per-object Euclidean distance transform (:func:`_object_edt`)
    — the "how deep inside the cell is each voxel" field. The maximum is the
    radius of the deepest interior point (a thickness proxy that does NOT depend
    on the principal-axis fit), and the distribution of EDT values separates a
    cell with a thick uniform core from one that is thin/sheet-like or has a
    large core plus a thin periphery.

    Features added (5):
        ``edt_max`` — ``max`` interior distance: the radius of the deepest core
            point (large for a thick blob, small for a thin sheet/sheet-cell).
        ``edt_mean`` — mean interior distance over the object's voxels.
        ``edt_p90`` — 90th percentile (a robust "typical thick-core depth",
            less spiky than the single max).
        ``edt_cv`` — ``std/mean`` of the interior distances (thickness
            unevenness; ~0 for a shape of uniform thickness).
        ``edt_skew`` — Fisher skewness of the interior distances (a large core
            plus a thin shell skews the field; a uniform slab does not).

    **Wave I medial-axis REMAINDER** (the feature catalogue; the interior EDT IS the medial
    radius field, so these are cheap derivations of the same field):
        ``medial_radius_mean`` — the mean interior medial radius. EQUAL to
            ``edt_mean`` (the interior EDT mean is exactly the mean local radius
            of the inscribed-ball medial-axis field) — emitted under the catalogue's
            medial name so that name is recognized. Spacing unit.
        ``medial_radius_cv`` — ``std / mean`` of the interior medial radii.
            EQUAL to ``edt_cv``; the medial-radius unevenness (a tube of
            constant thickness ~0, a bulged / tapered medial field higher).
            Dimensionless.
        ``edt_core_fraction`` — fraction of the object's interior voxels whose
            EDT is at least HALF the object's ``edt_max`` ∈ [0, 1] (bounded):
            the "thick-core" coverage. ~1 for a uniformly thick blob, small for
            a thin-shell / single-deep-core shape (one fat point with a thin
            periphery). Genuinely new (NOT a rename of edt_cv).

    Units: ``edt_max/mean/p90`` + ``medial_radius_mean`` inherit the spacing
    unit (physical when ``spacing`` is given, voxels otherwise); ``edt_cv``,
    ``edt_skew``, ``medial_radius_cv`` and ``edt_core_fraction`` are
    dimensionless.

    Degenerate shapes (empty crop, or a single-voxel object whose interior EDT
    is all-zero) yield NaN for the distribution-shape stats rather than crashing.

    Args:
        labels_3d: 3D integer label array (Z, Y, X). Background is 0.
        df: DataFrame from :func:`extract_features` (modified in-place).
        spacing: voxel spacing ``(Z, Y, X)`` for an anisotropy-aware EDT. None =
            voxel units.

    Returns:
        The same DataFrame with the 5 new columns.
    """
    from scipy.ndimage import find_objects
    from scipy.stats import skew as _skew

    n = len(df)
    cols = (
        "edt_max", "edt_mean", "edt_p90", "edt_cv", "edt_skew",
        # Wave I medial-axis remainder.
        "medial_radius_mean", "medial_radius_cv", "edt_core_fraction",
        # Wave N medial-axis remainder.
        "medial_radius_range",
    )
    if n == 0:
        for col in cols:
            df[col] = pd.Series(dtype=np.float64)
        return df

    edt_max = np.full(n, np.nan, dtype=np.float64)
    edt_mean = np.full(n, np.nan, dtype=np.float64)
    edt_p90 = np.full(n, np.nan, dtype=np.float64)
    edt_cv = np.full(n, np.nan, dtype=np.float64)
    edt_skew = np.full(n, np.nan, dtype=np.float64)
    # Wave I medial-axis remainder.
    core_frac = np.full(n, np.nan, dtype=np.float64)
    # Wave N medial-axis remainder.
    med_range = np.full(n, np.nan, dtype=np.float64)

    slices = find_objects(labels_3d.astype(np.int32))
    label_ids = df["label"].values
    eps = 1e-12

    for idx, label_id in enumerate(label_ids):
        if label_id < 1 or label_id > len(slices):
            continue
        obj_slice = slices[label_id - 1]
        if obj_slice is None:
            continue

        mask_crop = labels_3d[obj_slice] == label_id
        if not mask_crop.any():
            continue

        dt = _object_edt(mask_crop, spacing=spacing)
        interior = dt[mask_crop]            # distances over the object's voxels
        interior = interior[interior > 0]   # drop the 1-voxel-thick shell zeros
        if interior.size == 0:
            # Object is a 1-voxel-thick sheet/curve: every voxel touches the
            # boundary so the interior EDT is all-zero. max thickness is 0; the
            # distribution-shape stats are undefined.
            edt_max[idx] = 0.0
            edt_mean[idx] = 0.0
            edt_p90[idx] = 0.0
            edt_cv[idx] = 0.0
            edt_skew[idx] = 0.0
            # Wave I — a 1-voxel-thick sheet has no thick core.
            core_frac[idx] = 0.0
            # Wave N — a collapsed interior has no medial-radius spread.
            med_range[idx] = 0.0
            continue

        mx = float(interior.max())
        mean = float(interior.mean())
        std = float(interior.std())  # population std (ddof=0)
        edt_max[idx] = mx
        edt_mean[idx] = mean
        edt_p90[idx] = float(np.percentile(interior, 90))
        edt_cv[idx] = std / max(mean, eps)
        # Fisher skewness; a constant field (std≈0) -> 0, not NaN.
        if std > eps and interior.size >= 3:
            edt_skew[idx] = float(_skew(interior))
        else:
            edt_skew[idx] = 0.0
        # Wave I — thick-core fraction: share of interior voxels with EDT >= half
        # the max (uniform thickness -> ~1, single-deep-core -> small).
        core_frac[idx] = float(np.mean(interior >= 0.5 * mx)) if mx > eps else 0.0
        # Wave N — medial radius RANGE = max - min of the interior medial radius
        # field (carries the spacing unit). A tube of constant thickness -> ~0;
        # a bulged / tapered cell has a wide medial radius spread.
        med_range[idx] = mx - float(interior.min())

    df["edt_max"] = edt_max
    df["edt_mean"] = edt_mean
    df["edt_p90"] = edt_p90
    df["edt_cv"] = edt_cv
    df["edt_skew"] = edt_skew
    # Wave I medial-axis remainder. The interior-EDT mean/cv ARE the
    # medial radius mean/cv — alias them under the catalogue's medial name so it's
    # recognized (one source of truth, no recompute). edt_core_fraction is new.
    df["medial_radius_mean"] = edt_mean
    df["medial_radius_cv"] = edt_cv
    df["edt_core_fraction"] = core_frac
    # Wave N medial-axis remainder.
    df["medial_radius_range"] = med_range
    return df


def compute_geodesic_features(
    labels_3d: np.ndarray,
    df: pd.DataFrame,
    spacing: Optional[Tuple[float, ...]] = None,
) -> pd.DataFrame:
    """Interior geodesic length + tortuosity (M1 / the feature catalogue; C21).

    The geodesic distance is the length of the shortest path that stays INSIDE
    the cell mask, anchored at the EDT-deepest interior point (the most "central"
    voxel). ``geodesic_max`` is the farthest any interior voxel sits from that
    anchor along an in-mask path — for a straight rod this ~equals the Euclidean
    span, but for a bent / C-shaped cell the in-mask path must follow the bend,
    so it exceeds the straight-line distance. ``tortuosity`` makes that explicit:

        ``tortuosity = geodesic_max / max(straight_line_distance, eps)``

    where ``straight_line_distance`` is the Euclidean distance from the anchor to
    the geodesic-farthest voxel. ~1.0 for a straight cell; > 1.0 the more bent /
    winding it is. (the feature catalogue defines tortuosity from a geodesic length over an
    endpoint distance; this uses the EDT-anchored geodesic, which is robust on
    blobby masks where a skeleton main-path is fragile — see the module's
    skeleton note.)

    Implementation: a single-source spacing-aware geodesic transform inside the
    mask via :class:`skimage.graph.MCP_Geometric` (anisotropy via ``sampling``),
    seeded at the EDT-argmax voxel. The geodesic-farthest finite-cost voxel is
    the path endpoint; the anchor->endpoint Euclidean distance (also
    spacing-aware) is the straight-line denominator.

    Features added (2):
        ``geodesic_max`` — longest interior geodesic distance from the EDT-
            deepest anchor (spacing unit: physical when ``spacing`` given).
        ``tortuosity`` — geodesic_max / straight-line distance (dimensionless,
            >= 1 up to numerical noise). Degenerate shapes (single voxel, zero
            straight-line distance) -> NaN.

    Args:
        labels_3d: 3D integer label array (Z, Y, X). Background is 0.
        df: DataFrame from :func:`extract_features` (modified in-place).
        spacing: voxel spacing ``(Z, Y, X)``. None = voxel units (isotropic).

    Returns:
        The same DataFrame with the 2 new columns.
    """
    from scipy.ndimage import find_objects

    n = len(df)
    cols = (
        "geodesic_max", "tortuosity",
        # Wave N skeleton main-path / medial-axis geometry.
        "skeleton_main_path_length", "skeleton_main_path_tortuosity",
        "endpoint_distance_ratio", "medial_axis_bend_energy",
        "medial_branch_count",
    )
    if n == 0:
        for col in cols:
            df[col] = pd.Series(dtype=np.float64)
        return df

    geo_max = np.full(n, np.nan, dtype=np.float64)
    tort = np.full(n, np.nan, dtype=np.float64)
    # Wave N skeleton main-path / medial-axis geometry.
    sk_path_len = np.full(n, np.nan, dtype=np.float64)
    sk_path_tort = np.full(n, np.nan, dtype=np.float64)
    endpoint_ratio = np.full(n, np.nan, dtype=np.float64)
    bend_energy_m = np.full(n, np.nan, dtype=np.float64)
    branch_count = np.full(n, np.nan, dtype=np.float64)

    slices = find_objects(labels_3d.astype(np.int32))
    label_ids = df["label"].values
    sp = tuple(float(s) for s in spacing) if spacing is not None else (1.0, 1.0, 1.0)
    eps = 1e-9

    for idx, label_id in enumerate(label_ids):
        if label_id < 1 or label_id > len(slices):
            continue
        obj_slice = slices[label_id - 1]
        if obj_slice is None:
            continue

        mask_crop = labels_3d[obj_slice] == label_id
        n_vox = int(mask_crop.sum())
        if n_vox < 2:
            # A single (or empty) voxel has no internal path -> undefined.
            continue

        # --- Wave N skeleton main-path / medial-axis geometry ------
        # Independent of the EDT-anchored geodesic below (computed first so a
        # later MCP `continue` does not skip it). Robust per-cell skeleton
        # main-path geometry: skeletonize, build the skeleton-voxel graph, take
        # the two most-separated skeleton endpoints, and trace the in-skeleton
        # shortest path between them (the cell's medial "spine").
        try:
            (sk_len, sk_t, ep_ratio, sk_bend, sk_branch) = \
                _skeleton_main_path_geometry(mask_crop, sp)
            sk_path_len[idx] = sk_len
            sk_path_tort[idx] = sk_t
            endpoint_ratio[idx] = ep_ratio
            bend_energy_m[idx] = sk_bend
            branch_count[idx] = sk_branch
        except Exception:
            # Skeleton failure on a pathological mask -> leave NaN.
            pass

        # Anchor at the deepest interior point (EDT argmax). _object_edt is
        # spacing-aware so the anchor is consistent with the geodesic metric.
        dt = _object_edt(mask_crop, spacing=spacing)
        anchor = np.unravel_index(int(np.argmax(dt)), dt.shape)

        try:
            from skimage.graph import MCP_Geometric

            # Cost 1 inside the mask, +inf outside -> paths stay interior.
            costs = np.where(mask_crop, 1.0, np.inf).astype(np.float64)
            mcp = MCP_Geometric(costs, sampling=sp)
            cumulative, _ = mcp.find_costs([list(anchor)])
            geo = np.asarray(cumulative, dtype=np.float64)
            geo[~mask_crop] = np.inf
            finite = np.isfinite(geo)
            if not finite.any():
                continue
            far_flat = int(np.argmax(np.where(finite, geo, -np.inf)))
            endpoint = np.unravel_index(far_flat, geo.shape)
            g_len = float(geo[endpoint])
        except Exception:
            # MCP failure on a pathological mask -> leave NaN.
            continue

        # Straight-line (Euclidean) anchor->endpoint distance, spacing-aware.
        delta = (np.asarray(endpoint, dtype=np.float64)
                 - np.asarray(anchor, dtype=np.float64)) * np.asarray(sp)
        straight = float(np.sqrt(np.sum(delta ** 2)))

        geo_max[idx] = g_len
        if straight > eps:
            # Guard tiny numerical undershoot (geodesic must be >= straight).
            tort[idx] = max(g_len / straight, 1.0)
        # else: anchor == endpoint (no extent) -> tortuosity undefined (NaN).

    df["geodesic_max"] = geo_max
    df["tortuosity"] = tort
    # Wave N skeleton main-path / medial-axis geometry.
    df["skeleton_main_path_length"] = sk_path_len
    df["skeleton_main_path_tortuosity"] = sk_path_tort
    df["endpoint_distance_ratio"] = endpoint_ratio
    df["medial_axis_bend_energy"] = bend_energy_m
    df["medial_branch_count"] = branch_count
    return df


# 26-neighbour offsets (excluding the centre) for skeleton-voxel graph building.
_SKEL_NEIGHBOR_OFFSETS = np.array(
    [(dz, dy, dx)
     for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
     if not (dz == 0 and dy == 0 and dx == 0)],
    dtype=np.int64,
)


def _skeleton_main_path_geometry(mask_crop: np.ndarray, sp) -> tuple:
    """Skeleton main-path + medial-axis geometry for one crop mask.

    Skeletonizes ``mask_crop`` (skimage 3D skeleton), builds the 26-connected
    skeleton-voxel graph (physical edge weights via ``sp``), and returns a
    5-tuple summarizing the cell's medial "spine":

    * ``skeleton_main_path_length`` — the longest skeleton geodesic: the
      in-skeleton shortest-path length between the two most-separated skeleton
      voxels (physical, spacing-aware). The medial-axis length of the cell.
    * ``skeleton_main_path_tortuosity`` — that path length / the straight-line
      Euclidean distance between its two endpoints (>= 1; ~1 for a straight
      spine, larger for a bent / winding one).
    * ``endpoint_distance_ratio`` — Euclidean endpoint distance / the path
      length ∈ (0, 1] (the inverse-tortuosity straightness ratio: 1 = straight,
      → 0 = highly coiled).
    * ``medial_axis_bend_energy`` — mean squared discrete curvature along the
      main path (a second-difference proxy, normalized by the path length to be
      size-invariant): 0 for a straight spine, large for a sharply bent one.
    * ``medial_branch_count`` — number of skeleton branch nodes (voxels with
      >= 3 skeleton neighbours): 0 for a simple unbranched spine, > 0 for a
      forked / Y-shaped medial axis (a merge / complex-shape cue).

    Returns ``(nan, nan, nan, nan, nan)`` when the skeleton is too small to
    define a path (< 2 skeleton voxels) — except ``medial_branch_count`` which
    is still 0 for a present-but-trivial skeleton.
    """
    from skimage.morphology import skeletonize as _skeletonize
    from skimage.graph import MCP_Geometric

    nan5 = (np.nan, np.nan, np.nan, np.nan, np.nan)
    if not mask_crop.any():
        return nan5
    skel = _skeletonize(mask_crop)
    sk_coords = np.argwhere(skel)
    n_sk = sk_coords.shape[0]
    if n_sk < 1:
        return nan5

    # --- branch count: skeleton voxels with >= 3 skeleton neighbours --------
    # Count 26-connected skeleton neighbours per skeleton voxel.
    occ = set(map(tuple, sk_coords.tolist()))
    branch = 0
    deg_map = {}
    for c in sk_coords:
        cz, cy, cx = int(c[0]), int(c[1]), int(c[2])
        d = 0
        for off in _SKEL_NEIGHBOR_OFFSETS:
            if (cz + int(off[0]), cy + int(off[1]), cx + int(off[2])) in occ:
                d += 1
        deg_map[(cz, cy, cx)] = d
        if d >= 3:
            branch += 1

    if n_sk < 2:
        # A single skeleton voxel: no path, but the (zero) branch count is real.
        return (np.nan, np.nan, np.nan, np.nan, float(branch))

    # --- main path: two most-separated skeleton voxels, in-skeleton geodesic -
    sp_arr = np.asarray(sp, dtype=np.float64)
    sampling = tuple(float(v) for v in sp_arr)   # MCP_Geometric needs a tuple
    # Pick the first endpoint as the skeleton voxel farthest (in-skeleton) from
    # an arbitrary seed, then the second as the farthest from that one (the
    # standard 2-pass graph-diameter heuristic). Costs: 1 on the skeleton, +inf
    # off it, so MCP_Geometric paths stay ON the skeleton.
    costs = np.where(skel, 1.0, np.inf).astype(np.float64)

    def _farthest_from(seed):
        mcp = MCP_Geometric(costs, sampling=sampling)
        cum, _ = mcp.find_costs([list(seed)])
        cum = np.asarray(cum, dtype=np.float64)
        cum[~skel] = -np.inf
        flat = int(np.argmax(cum))
        far = np.unravel_index(flat, cum.shape)
        return far, float(cum[far])

    seed = tuple(int(v) for v in sk_coords[0])
    ep1, _ = _farthest_from(seed)
    ep2, path_len = _farthest_from(ep1)
    if not np.isfinite(path_len) or path_len <= 0:
        return (np.nan, np.nan, np.nan, np.nan, float(branch))

    # Straight-line endpoint distance (spacing-aware).
    delta = (np.asarray(ep2, dtype=np.float64)
             - np.asarray(ep1, dtype=np.float64)) * sp_arr
    straight = float(np.sqrt(np.sum(delta ** 2)))

    main_len = float(path_len)
    if straight > 1e-9:
        path_tort = max(main_len / straight, 1.0)
        ep_ratio = min(straight / main_len, 1.0)
    else:
        path_tort = np.nan
        ep_ratio = np.nan

    # --- bend energy: trace the main path + measure discrete curvature -------
    # Re-run MCP from ep1 with traceback to recover the actual path voxels to
    # ep2, then compute the mean squared second difference of the (physical)
    # path coordinates, normalized by the path length so it is size-invariant.
    bend = np.nan
    try:
        mcp2 = MCP_Geometric(costs, sampling=sampling)
        mcp2.find_costs([list(ep1)])
        path = mcp2.traceback(list(ep2))            # list of (z, y, x)
        pts = np.asarray(path, dtype=np.float64) * sp_arr[None, :]
        if pts.shape[0] >= 3:
            second = pts[2:] - 2.0 * pts[1:-1] + pts[:-2]   # discrete curvature
            # Normalize by (path_len / n_segments)^2 ~ the squared step so the
            # measure is scale-free; use mean squared curvature / length.
            bend = float(np.mean(np.sum(second ** 2, axis=1))) / max(main_len, 1e-9)
        else:
            bend = 0.0
    except Exception:
        bend = np.nan

    return (main_len, path_tort, ep_ratio, bend, float(branch))


def compute_ma_cross_section_features(
    labels_3d: np.ndarray,
    df: pd.DataFrame,
    spacing: Optional[Tuple[float, ...]] = None,
    n_slabs: int = 20,
) -> pd.DataFrame:
    """Major-axis cross-section profile (M1 / the feature catalogue; C2, primitive P11).

    Tilt-robust analogues of the Z-stratified :func:`compute_cross_section_features`
    / :func:`compute_cross_section_profile_stats`. Instead of slicing along the
    GLOBAL Z axis (oblique for a tilted cell), this projects every object voxel
    onto the cell's PRINCIPAL (major inertia) axis and bins by that projection —
    so the slabs are perpendicular to the cell's OWN long axis and the profile is
    stable under whole-cell rotation / Z-tilt.

    The major-axis direction is the ``major_axis_{z,y,x}`` vector
    :func:`extract_features` already emitted (the most-elongated principal axis);
    the centroid (``centroid-{0,1,2}``) is the projection origin. For each voxel
    ``v`` the signed projection ``s = (v - centroid) · major_dir`` (in physical
    units when ``spacing`` is given) is min-max normalized to ``t ∈ [0, 1]`` and
    binned into ``n_slabs`` slabs; the per-slab voxel count (a cross-sectional
    area proxy, perpendicular to the major axis) is the profile ``A(t)``.

    Features added (5):
        ``ma_min_cross_section_area`` — ``min_t A(t)`` over occupied slabs (a
            small value flags a neck / pinch along the cell's own long axis).
        ``ma_max_cross_section_area`` — ``max_t A(t)`` (the widest slab).
        ``ma_cross_section_range`` — ``max - min`` of the profile.
        ``ma_cross_section_taper`` — slope of a linear fit of ``A(t)`` vs ``t``
            (negative = tapering toward t=1, positive = expanding). Sign is in
            the (arbitrary) major-axis direction; magnitude is what matters.
        ``ma_cross_section_waist_depth`` — normalized ``t`` position of the
            thinnest occupied slab, in ``[0, 1]`` (where the neck sits along the
            cell's own long axis).
        ``ma_cross_section_cv`` (Wave I, remainder) — ``std(A(t)) /
            max(mean(A(t)), eps)`` over the occupied slabs: how uneven the
            tilt-robust major-axis profile is (the cell-intrinsic analogue of
            ``cross_section_area_cv``). ~0 for a uniform rod, large for a
            necked / tapered / lobed cell. Dimensionless.

    Areas are voxel counts (matching the Z-based siblings' voxel-count profile);
    the range/taper/waist/cv summaries are relative, so spacing only affects the
    projection binning, not the units.

    Degenerate shapes (missing axis/centroid columns, single occupied slab,
    empty crop) yield NaN for the undefined summaries rather than crashing.

    Args:
        labels_3d: 3D integer label array (Z, Y, X). Background is 0.
        df: DataFrame from :func:`extract_features` (modified in-place). Must
            carry ``major_axis_{z,y,x}`` + ``centroid-{0,1,2}``; if absent every
            new column is NaN.
        spacing: voxel spacing ``(Z, Y, X)`` for the (physical) projection. None
            = voxel units.
        n_slabs: number of slabs perpendicular to the major axis (default 20).

    Returns:
        The same DataFrame with the 5 new columns.
    """
    from scipy.ndimage import find_objects, gaussian_filter1d
    from scipy.signal import find_peaks
    from scipy.stats import skew as _skew

    n = len(df)
    cols = (
        "ma_min_cross_section_area", "ma_max_cross_section_area",
        "ma_cross_section_range", "ma_cross_section_taper",
        "ma_cross_section_waist_depth",
        # Wave I major-axis remainder.
        "ma_cross_section_cv",
        # Wave N major-axis remainder — widest-slab position, residual
        # bulge, peak count, profile skew.
        "ma_max_cross_section_axis_relative", "ma_cross_section_bulge",
        "ma_peak_count", "ma_cross_section_skew",
    )
    if n == 0:
        for col in cols:
            df[col] = pd.Series(dtype=np.float64)
        return df

    axis_cols = ["major_axis_z", "major_axis_y", "major_axis_x"]
    cen_cols = ["centroid-0", "centroid-1", "centroid-2"]
    if not all(c in df.columns for c in axis_cols + cen_cols):
        for col in cols:
            df[col] = np.nan
        return df

    ma_min = np.full(n, np.nan, dtype=np.float64)
    ma_max = np.full(n, np.nan, dtype=np.float64)
    ma_range = np.full(n, np.nan, dtype=np.float64)
    ma_taper = np.full(n, np.nan, dtype=np.float64)
    ma_waist = np.full(n, np.nan, dtype=np.float64)
    ma_cv = np.full(n, np.nan, dtype=np.float64)   # Wave I
    # Wave N major-axis remainder.
    ma_max_rel = np.full(n, np.nan, dtype=np.float64)
    ma_bulge = np.full(n, np.nan, dtype=np.float64)
    ma_peak_count = np.full(n, np.nan, dtype=np.float64)
    ma_skew = np.full(n, np.nan, dtype=np.float64)

    slices = find_objects(labels_3d.astype(np.int32))
    label_ids = df["label"].values
    sp = np.asarray(spacing, dtype=np.float64) if spacing is not None else np.ones(3)
    n_slabs = max(2, int(n_slabs))

    axis_vals = df[axis_cols].values.astype(np.float64)
    cen_vals = df[cen_cols].values.astype(np.float64)

    for idx, label_id in enumerate(label_ids):
        if label_id < 1 or label_id > len(slices):
            continue
        obj_slice = slices[label_id - 1]
        if obj_slice is None:
            continue

        mask_crop = labels_3d[obj_slice] == label_id
        coords = np.argwhere(mask_crop)
        if coords.shape[0] < 2:
            continue

        # The major-axis direction (ZYX). A degenerate (all-zero) axis -> skip.
        direction = axis_vals[idx]
        norm = float(np.sqrt(np.sum(direction ** 2)))
        if norm <= 1e-12:
            continue
        direction = direction / norm

        # Project each voxel onto the major axis, in physical units. Use the
        # crop-local centroid (global centroid offset cancels in the projection
        # of a translation-invariant displacement, but staying crop-local avoids
        # depending on whether centroid-* were offset to global yet).
        local_centroid = coords.mean(axis=0)
        disp = (coords - local_centroid) * sp           # (N, 3) physical
        s = disp @ direction                            # (N,) signed projection

        s_min, s_max = float(s.min()), float(s.max())
        s_range = s_max - s_min
        if s_range <= 1e-12:
            continue

        t = (s - s_min) / s_range                       # normalized [0,1]
        # Bin into n_slabs; clip the t==1 voxels into the last bin.
        bins = np.clip((t * n_slabs).astype(np.int64), 0, n_slabs - 1)
        counts = np.bincount(bins, minlength=n_slabs).astype(np.float64)

        occupied = counts > 0
        if occupied.sum() < 1:
            continue
        occ_counts = counts[occupied]
        # Slab centers in [0,1] for the occupied slabs (for the linear fit + waist).
        slab_centers = (np.arange(n_slabs, dtype=np.float64) + 0.5) / n_slabs
        occ_centers = slab_centers[occupied]

        ma_min[idx] = float(occ_counts.min())
        ma_max[idx] = float(occ_counts.max())
        ma_range[idx] = float(occ_counts.max() - occ_counts.min())

        # Wave I — CV of the occupied-slab profile (std/mean). Constant profile
        # (single slab or zero spread) -> 0.0, not NaN.
        mean_c = float(occ_counts.mean())
        if occ_counts.size >= 2 and mean_c > 1e-12:
            ma_cv[idx] = float(occ_counts.std()) / mean_c
        else:
            ma_cv[idx] = 0.0

        # Waist = thinnest occupied slab position.
        ma_waist[idx] = float(occ_centers[int(np.argmin(occ_counts))])

        # Linear taper fit over occupied slabs (needs >=2 distinct positions).
        if occ_centers.size >= 2 and np.ptp(occ_centers) > 1e-12:
            slope = float(np.polyfit(occ_centers, occ_counts, 1)[0])
            ma_taper[idx] = slope
        else:
            ma_taper[idx] = 0.0

        # --- Wave N major-axis remainder ---------------------------
        # widest-slab position (argmax-t): where the cell bulges along its own
        # long axis (0 = one end, 1 = the other; arbitrary major-axis sign).
        ma_max_rel[idx] = float(occ_centers[int(np.argmax(occ_counts))])
        # residual bulge: RMS of the occupied profile about its linear taper
        # fit, normalized by the mean area (the tilt-robust analogue of the
        # global-Z cross_section_bulge — 0 for a clean cone/cylinder, large for
        # a barreled / necked profile). Needs a fit (>=2 distinct positions).
        if occ_centers.size >= 2 and np.ptp(occ_centers) > 1e-12:
            coeffs_b = np.polyfit(occ_centers, occ_counts, 1)
            fitted_b = np.polyval(coeffs_b, occ_centers)
            rms_b = float(np.sqrt(np.mean((occ_counts - fitted_b) ** 2)))
            ma_bulge[idx] = rms_b / max(mean_c, 1e-12)
        else:
            ma_bulge[idx] = 0.0
        # peak count of the (end-padded, smoothed) major-axis profile — lobes
        # along the cell's own long axis. Same prominence-floor + end-padding
        # convention as the global-Z compute_cross_section_profile_stats so a
        # smooth rod -> 1, a dumbbell -> 2.
        if occ_counts.size >= 3:
            smoothed_m = gaussian_filter1d(occ_counts, sigma=1.0, mode="nearest")
            pad_val_m = float(smoothed_m.min()) - 1.0
            padded_m = np.concatenate(([pad_val_m], smoothed_m, [pad_val_m]))
            prof_range_m = float(smoothed_m.max() - smoothed_m.min())
            prom_m = _CROSS_SECTION_PEAK_PROMINENCE_FRAC * prof_range_m
            peaks_m, _ = find_peaks(
                padded_m, plateau_size=1,
                prominence=(prom_m if prom_m > 0 else None),
            )
            ma_peak_count[idx] = float(max(len(peaks_m), 1))
        else:
            ma_peak_count[idx] = 1.0
        # skewness of the occupied-slab profile (whether area is biased toward
        # one end of the cell's own axis). Constant profile -> 0.
        if occ_counts.size >= 2 and float(occ_counts.std()) > 1e-12:
            ma_skew[idx] = float(_skew(occ_counts))
        else:
            ma_skew[idx] = 0.0

    df["ma_min_cross_section_area"] = ma_min
    df["ma_max_cross_section_area"] = ma_max
    df["ma_cross_section_range"] = ma_range
    df["ma_cross_section_taper"] = ma_taper
    df["ma_cross_section_waist_depth"] = ma_waist
    df["ma_cross_section_cv"] = ma_cv   # Wave I
    # Wave N major-axis remainder.
    df["ma_max_cross_section_axis_relative"] = ma_max_rel
    df["ma_cross_section_bulge"] = ma_bulge
    df["ma_peak_count"] = ma_peak_count
    df["ma_cross_section_skew"] = ma_skew
    return df


# ---------------------------------------------------------------------------
# M5 — Shell decomposition + surface curvature / concavity / symmetry.
#
# NEW features (NOT a monorepo port) built from the EpiCycle feature catalogue
# (Radial/Shell + Curvature/Concavity/Symmetry). Crop-local per-object readers run in the SAME per-cell
# pass as M0/M1 (labels-only; ``compute_shell_features`` reuses the M1
# ``_object_edt`` field, ``compute_curvature_features`` reuses the same
# marching-cubes mesh ``compute_surface_features`` builds — output of the
# latter is UNCHANGED).
# ---------------------------------------------------------------------------


# Equal-VOLUME shell thresholds on the radial position r = d / max(d): the values
# that split a SOLID SPHERE into three equal-volume shells (F(r) = 1 - (1-r)^3).
# A sphere -> ~1/3 in each shell; sheets/cores deviate from 1/3 (B4).
_SHELL_R1 = 1.0 - (2.0 / 3.0) ** (1.0 / 3.0)   # ≈ 0.1262 (outer | mid boundary)
_SHELL_R2 = 1.0 - (1.0 / 3.0) ** (1.0 / 3.0)   # ≈ 0.3066 (mid | inner boundary)


def compute_shell_features(
    labels_3d: np.ndarray,
    df: pd.DataFrame,
    spacing: Optional[Tuple[float, ...]] = None,
) -> pd.DataFrame:
    """Radial / shell decomposition per cell (M5 / the feature catalogue; C8, primitive P7).

    Partitions each cell into concentric shells using the interior Euclidean
    distance transform (:func:`_object_edt`, the M1 thickness field). Every
    object voxel's interior distance ``d`` is normalized by the object's maximum
    interior distance to a **radial position** ``r = d / max(d) ∈ [0, 1]`` —
    ``r → 1`` at the deepest core point, ``r → 0`` at the surface. The object is
    split into three EQUAL-VOLUME shells by ``r`` using the thresholds that split a
    SOLID SPHERE into equal-volume thirds (``F(r) = 1 - (1-r)^3``; equal-WIDTH
    thirds would be a geometric trap — the volume-near-the-surface bias makes the
    outer band hold ~70 % for ANY compact shape, even a sphere)::

        outer  : r <= R1 ≈ 0.126     (the surface rind)
        mid    : R1 < r <= R2 ≈ 0.307
        inner  : r > R2              (the core)

    A SPHERE therefore yields ~1/3 in each shell; a thin SHEET piles voxels into
    the outer band (``frac_outer`` > 1/3) and a thick-cored blob fattens the inner
    band — so the fractions now MEASURE the radial profile instead of reporting a
    geometric constant.

    Geometry-only for now (the feature catalogue's intensity/probability-per-shell variants
    need the M2 sampler and are deferred). The shell partition is rotation- and
    translation-invariant (it is a function of the object's own distance field),
    so ``frame="intrinsic"``.

    Features added (6):
        ``shell_volume_frac_inner`` / ``_mid`` / ``_outer`` — fraction of the
            object's voxels in each shell ∈ [0, 1]; the three SUM to 1 (bounded).
            A compact blob has a fat core (large inner frac); a thin sheet is
            almost entirely outer rind.
        ``core_surface_ratio`` — ``inner_voxels / outer_voxels`` (≥ 0, unbounded):
            a high value flags a thick solid core relative to its surface rind.
            ``inf`` is avoided — a zero outer count yields NaN (degenerate).
        ``mean_radial_position`` — mean ``r`` over the object's voxels ∈ [0, 1]
            (bounded): high for a thick blob (most voxels deep), low for a
            sheet (most voxels near the surface).
        ``radial_position_std`` — std of ``r`` over the object's voxels ∈ [0,
            ~0.5] (treated unbounded): the spread of the radial profile.

    Spacing-aware: the EDT is computed with ``spacing`` so the radial partition
    respects voxel anisotropy. The fractions / mean / std are dimensionless.

    Degenerate shapes (empty crop, or a 1-voxel-thick object whose interior EDT
    has a single distinct positive value so ``max(d) == 0`` over the >0 voxels)
    yield NaN for the undefined summaries rather than crashing.

    Args:
        labels_3d: 3D integer label array (Z, Y, X). Background is 0.
        df: DataFrame from :func:`extract_features` (modified in-place).
        spacing: voxel spacing ``(Z, Y, X)`` for an anisotropy-aware EDT. None =
            voxel units.

    Returns:
        The same DataFrame with the 6 new columns.
    """
    from scipy.ndimage import find_objects

    n = len(df)
    cols = (
        "shell_volume_frac_inner", "shell_volume_frac_mid",
        "shell_volume_frac_outer", "core_surface_ratio",
        "mean_radial_position", "radial_position_std",
        # Wave N radial/shell remainder — per-band compactness +
        # per-band anisotropy/eccentricity + core offset + volume ratios.
        "shell_compactness_inner", "shell_compactness_mid",
        "shell_compactness_outer",
        "shell_eccentricity_inner", "shell_eccentricity_mid",
        "shell_eccentricity_outer",
        "core_offset",
        "shell_volume_ratio_inner_outer", "shell_volume_ratio_mid_outer",
    )
    if n == 0:
        for col in cols:
            df[col] = pd.Series(dtype=np.float64)
        return df

    frac_inner = np.full(n, np.nan, dtype=np.float64)
    frac_mid = np.full(n, np.nan, dtype=np.float64)
    frac_outer = np.full(n, np.nan, dtype=np.float64)
    cs_ratio = np.full(n, np.nan, dtype=np.float64)
    mean_r = np.full(n, np.nan, dtype=np.float64)
    std_r = np.full(n, np.nan, dtype=np.float64)
    # Wave N radial/shell remainder.
    comp_inner = np.full(n, np.nan, dtype=np.float64)
    comp_mid = np.full(n, np.nan, dtype=np.float64)
    comp_outer = np.full(n, np.nan, dtype=np.float64)
    ecc_inner = np.full(n, np.nan, dtype=np.float64)
    ecc_mid = np.full(n, np.nan, dtype=np.float64)
    ecc_outer = np.full(n, np.nan, dtype=np.float64)
    core_off = np.full(n, np.nan, dtype=np.float64)
    vr_inner_outer = np.full(n, np.nan, dtype=np.float64)
    vr_mid_outer = np.full(n, np.nan, dtype=np.float64)

    slices = find_objects(labels_3d.astype(np.int32))
    label_ids = df["label"].values
    eps = 1e-12

    for idx, label_id in enumerate(label_ids):
        if label_id < 1 or label_id > len(slices):
            continue
        obj_slice = slices[label_id - 1]
        if obj_slice is None:
            continue

        mask_crop = labels_3d[obj_slice] == label_id
        if not mask_crop.any():
            continue

        dt = _object_edt(mask_crop, spacing=spacing)
        d = dt[mask_crop]                       # interior distance over the voxels
        n_vox = float(d.size)
        if n_vox <= 0:
            continue

        d_max = float(d.max())
        if d_max <= eps:
            # Every voxel sits on the boundary (a 1-voxel-thick sheet/curve):
            # the radial field collapses to all-zero. The whole object is the
            # outer rind; there is no core. Report that explicitly.
            frac_outer[idx] = 1.0
            frac_mid[idx] = 0.0
            frac_inner[idx] = 0.0
            cs_ratio[idx] = 0.0                 # inner / outer = 0 / 1
            mean_r[idx] = 0.0
            std_r[idx] = 0.0
            # Wave N — the whole object IS the outer rind; the inner/mid bands
            # are empty so their shape is undefined (NaN). The outer band's
            # shape IS the whole-object shape; report it. Volume ratios are 0
            # (no inner/mid volume); core offset is undefined (no core).
            comp_outer[idx] = _voxel_band_compactness(mask_crop)
            ecc_outer[idx] = _band_anisotropy(mask_crop, spacing)
            vr_inner_outer[idx] = 0.0
            vr_mid_outer[idx] = 0.0
            continue

        r = d / d_max                           # radial position in [0, 1]
        # EQUAL-VOLUME shell thresholds (B4). Splitting r ∈ [0,1] into equal-WIDTH
        # thirds is a geometric trap: for ANY compact shape the volume near the
        # surface dominates (a spherical shell's volume ∝ ρ²), so the outer band
        # holds ~70% even for a perfect sphere — the fractions can't separate
        # core-rich from sheet-like. Instead we use the thresholds that split a
        # SOLID SPHERE into equal thirds by volume, so a sphere → ~1/3,1/3,1/3 and
        # departures from that measure the actual radial profile (a sheet piles
        # voxels into the outer band → frac_outer > 1/3; a thick core → inner up).
        # For a sphere the cumulative volume fraction at radial position r is
        # F(r) = 1 - (1-r)^3, so the equal-thirds thresholds are:
        #   F=1/3 -> r = 1 - (2/3)^(1/3) ≈ 0.1262
        #   F=2/3 -> r = 1 - (1/3)^(1/3) ≈ 0.3066
        outer_n = float(np.count_nonzero(r <= _SHELL_R1))
        mid_n = float(np.count_nonzero((r > _SHELL_R1) & (r <= _SHELL_R2)))
        inner_n = float(np.count_nonzero(r > _SHELL_R2))

        frac_outer[idx] = outer_n / n_vox
        frac_mid[idx] = mid_n / n_vox
        frac_inner[idx] = inner_n / n_vox
        # Core/surface volume ratio; a zero outer rind is degenerate -> NaN.
        cs_ratio[idx] = (inner_n / outer_n) if outer_n > 0 else np.nan
        mean_r[idx] = float(r.mean())
        std_r[idx] = float(r.std())             # population std (ddof=0)

        # --- Wave N radial/shell remainder --------------------------
        # Build the 3D band masks (same equal-volume thresholds) so the per-band
        # SHAPE descriptors can read the actual voxel set of each shell. r_field
        # holds r on the object's voxels, 0 elsewhere; band masks are mask_crop
        # AND the radial slab.
        r_field = np.zeros_like(dt, dtype=np.float64)
        r_field[mask_crop] = r
        band_outer = mask_crop & (r_field <= _SHELL_R1)
        band_mid = mask_crop & (r_field > _SHELL_R1) & (r_field <= _SHELL_R2)
        band_inner = mask_crop & (r_field > _SHELL_R2)

        # per-band Bribiesca discrete compactness (1 = cube/blob, low = stringy).
        comp_inner[idx] = _voxel_band_compactness(band_inner)
        comp_mid[idx] = _voxel_band_compactness(band_mid)
        comp_outer[idx] = _voxel_band_compactness(band_outer)
        # per-band 3D anisotropy/eccentricity sqrt(1 - lam_min/lam_max) of the
        # band's voxel-coordinate covariance ∈ [0, 1] (0 = isotropic blob,
        # → 1 = a thin streak / sheet).
        ecc_inner[idx] = _band_anisotropy(band_inner, spacing)
        ecc_mid[idx] = _band_anisotropy(band_mid, spacing)
        ecc_outer[idx] = _band_anisotropy(band_outer, spacing)
        # core_offset: distance between the INNER-core band centroid and the
        # whole-object centroid, normalized by the object's equivalent radius
        # (so it is size-invariant). ~0 = a concentric core, large = the core is
        # pushed to one side (an asymmetric / merged mask). NaN when the core is
        # empty. Spacing-aware (physical centroids).
        if int(band_inner.sum()) >= 1:
            sp_arr_s = (np.asarray(spacing, dtype=np.float64)
                        if spacing is not None else np.ones(3))
            obj_coords = np.argwhere(mask_crop).astype(np.float64) * sp_arr_s
            core_coords = np.argwhere(band_inner).astype(np.float64) * sp_arr_s
            obj_cen = obj_coords.mean(axis=0)
            core_cen = core_coords.mean(axis=0)
            off = float(np.sqrt(np.sum((core_cen - obj_cen) ** 2)))
            phys_vol = float(n_vox) * float(
                sp_arr_s[0] * sp_arr_s[1] * sp_arr_s[2])
            r_eq = (3.0 * phys_vol / (4.0 * np.pi)) ** (1.0 / 3.0) if phys_vol > 0 else 0.0
            core_off[idx] = off / max(r_eq, eps)
        # volume ratios (inner/outer, mid/outer): a fat-core vs thin-rind cue.
        vr_inner_outer[idx] = (inner_n / outer_n) if outer_n > 0 else np.nan
        vr_mid_outer[idx] = (mid_n / outer_n) if outer_n > 0 else np.nan

    df["shell_volume_frac_inner"] = frac_inner
    df["shell_volume_frac_mid"] = frac_mid
    df["shell_volume_frac_outer"] = frac_outer
    df["core_surface_ratio"] = cs_ratio
    df["mean_radial_position"] = mean_r
    df["radial_position_std"] = std_r
    # Wave N radial/shell remainder.
    df["shell_compactness_inner"] = comp_inner
    df["shell_compactness_mid"] = comp_mid
    df["shell_compactness_outer"] = comp_outer
    df["shell_eccentricity_inner"] = ecc_inner
    df["shell_eccentricity_mid"] = ecc_mid
    df["shell_eccentricity_outer"] = ecc_outer
    df["core_offset"] = core_off
    df["shell_volume_ratio_inner_outer"] = vr_inner_outer
    df["shell_volume_ratio_mid_outer"] = vr_mid_outer
    return df


def _external_face_count(mask: np.ndarray) -> int:
    """Number of exposed (external) voxel faces of a boolean ``mask`` (6-neighbour).

    A True voxel contributes a face for every one of its 6 orthogonal neighbours
    that is NOT also True — i.e. the discrete surface-area-in-faces of the voxel set
    (the ``A_v`` of Bribiesca's discrete compactness). Works for ANY voxel set,
    including a hollow/non-convex shell band, so it is the robust surface primitive
    for the multi-scale shell shape templates (a marching-cubes mesh of a thin rind
    is fragile). Pads with one voxel of background so faces on the crop boundary
    count as external.
    """
    m = np.asarray(mask).astype(bool)
    if not m.any():
        return 0
    padded = np.pad(m, 1, mode="constant", constant_values=False)
    c = padded[1:-1, 1:-1, 1:-1]
    a_v = 0
    a_v += int((c & ~padded[:-2, 1:-1, 1:-1]).sum())   # -Z
    a_v += int((c & ~padded[2:, 1:-1, 1:-1]).sum())    # +Z
    a_v += int((c & ~padded[1:-1, :-2, 1:-1]).sum())   # -Y
    a_v += int((c & ~padded[1:-1, 2:, 1:-1]).sum())    # +Y
    a_v += int((c & ~padded[1:-1, 1:-1, :-2]).sum())   # -X
    a_v += int((c & ~padded[1:-1, 1:-1, 2:]).sum())    # +X
    return a_v


def _voxel_band_compactness(mask: np.ndarray) -> float:
    """Bribiesca discrete compactness of a voxel band (1 = cube-like, low = filamentary).

    ``C_d = (n - A_v / 6) / (n - n^(2/3))`` (the SAME measure as
    :func:`compute_discrete_compactness`, the un-normalised Bribiesca form), where
    ``n`` is the band's voxel count and ``A_v`` its external face count
    (:func:`_external_face_count`). Bounded ~[0, 1] (a sphere ≈ 0.85, a cube = 1, a
    thin sheet / filament -> 0). Returns NaN for a band too small to define
    (``n <= 1`` or a degenerate denominator). Reused for the per-shell-band shape
    of the multi-scale templates.
    """
    n_vox = int(np.asarray(mask).astype(bool).sum())
    if n_vox <= 1:
        return np.nan
    a_v = _external_face_count(mask)
    denom = n_vox - n_vox ** (2.0 / 3.0)
    if denom <= 0:
        return np.nan
    cd = (n_vox - a_v / 6.0) / denom
    # mathematically in [0,1] (A_v <= 6n so the numerator <= n_vox; a single isolated
    # voxel -> 0); clamp the tiny floating-point overshoot for small compact bands so
    # the "bounded" manifest contract holds exactly.
    return float(min(max(cd, 0.0), 1.0))


def _band_anisotropy(mask: np.ndarray, spacing=None) -> float:
    """3D anisotropy/eccentricity of a voxel band's coordinate covariance ∈ [0, 1].

    ``sqrt(1 - lambda_min / lambda_max)`` of the (spacing-aware) 3x3 covariance of
    the band's True voxel coordinates: 0 = an isotropic blob (all eigenvalues
    equal), → 1 = a thin streak / sheet (a near-zero smallest eigenvalue). NaN for
    a band too small (< 4 voxels) or a degenerate (zero-spread) covariance.
    Reused for the per-shell-band eccentricity of the radial/shell remainder.
    """
    coords = np.argwhere(np.asarray(mask, dtype=bool))
    if coords.shape[0] < 4:
        return np.nan
    pts = coords.astype(np.float64)
    if spacing is not None:
        pts = pts * np.asarray(spacing, dtype=np.float64)[None, :]
    cov = np.cov(pts.T)
    if not np.all(np.isfinite(cov)):
        return np.nan
    evals = np.linalg.eigvalsh(cov)              # ascending
    lam_min = float(evals[0])
    lam_max = float(evals[-1])
    if lam_max <= 1e-12:
        return np.nan
    return float(np.sqrt(max(0.0, 1.0 - lam_min / lam_max)))


# Per-cell SHAPE multi-scale template column set. Intentionally MODEST —
# a small curated set across the M5 EDT shells, NOT a feature×partition explosion
# (the catalogue warns the multi-scale pattern is "a multiplier ... apply selectively,
# otherwise redundancy will explode"). EXTENSIBLE: more shell-restricted shape
# stats can be added here, but each new column multiplies the table, so add with
# intent.
MULTISCALE_SHAPE_COLUMNS = (
    "shell_sphericity_inner", "shell_sphericity_outer",
    "shell_surface_ratio", "shell_compactness_ratio",
    # Wave N multi-scale remainder — the mid-band shape (completes
    # inner/mid/outer) + the axis-quartile volume-fraction template
    # (<feature>_axis_q1 / _axis_q4 applied to volume fraction along the major
    # axis) + the axis-quartile area-ratio (end-to-end taper).
    "shell_sphericity_mid",
    "shape_axis_q1_fraction", "shape_axis_q4_fraction",
    "shape_axis_q1_q4_ratio",
)


def compute_multiscale_shape(
    labels_3d: np.ndarray,
    df: pd.DataFrame,
    spacing: Optional[Tuple[float, ...]] = None,
) -> pd.DataFrame:
    """Multi-scale (zone-restricted) SHAPE templates across the M5 EDT shells
    (the C17 multi-scale template pattern, applied to SHAPE).

    The pattern is a MULTIPLIER — apply an existing scalar across region
    partitions (``<feature>_inner`` / ``<feature>_outer`` …). Done literally it
    explodes the column count, so this is a DELIBERATELY MINIMAL, curated instance:
    a SHAPE template across the inner / outer radial shells the M5
    :func:`compute_shell_features` partition already defines (the EDT radial
    position ``r = d / max(d)``; inner ``r > R2 ≈ 0.307`` = the core band, outer
    ``r <= R1 ≈ 0.126`` = the surface rind). The shell INTENSITY multi-scale was
    already done in Wave F; this is the SHAPE multi-scale.

    Same per-cell crop pass as M0 / M1 / M5 (labels-only, crop-local), rotation- and
    translation-invariant (a function of the object's own distance field), so
    ``frame="intrinsic"``.

    Features added (4):
        ``shell_sphericity_inner`` — the discrete (Bribiesca) compactness of the
            INNER core band (the voxels with ``r > R2``) ∈ ~[0, 1]: a compact,
            blob-like core reads high, a stringy / fragmented core low. NaN when the
            core band is too small to define (a thin sheet has no core).
        ``shell_sphericity_outer`` — the same compactness of the OUTER rind band
            (``r <= R1``) ∈ ~[0, 1]: a smooth even rind reads higher than a ragged
            one. NaN when the rind is degenerate.
        ``shell_surface_ratio`` — the fraction of the cell's voxels that are TRUE
            SKIN voxels (the 1-voxel boundary shell that touches background, from a
            single erosion) ∈ [0, 1]: the surface-to-bulk ratio. ~1 for a thin sheet
            (almost every voxel is on the skin), lower for a thick blob (a fat
            interior dilutes the skin). Bounded. (Distinct from M5's
            ``shell_volume_frac_outer``, which is the THICK equal-volume outer band's
            fraction; this is the literal 1-voxel skin — the SHAPE surface-vs-bulk
            signal of the multi-scale template.)
        ``shell_compactness_ratio`` — ``shell_sphericity_inner /
            max(shell_sphericity_outer, eps)``: > 1 = the core is more compact than
            the rind (a cohesive blob with a ragged surface), < 1 = a smooth shell
            around a fragmented interior. Unbounded >= 0; NaN when either band is
            undefined.

    All NaN-guarded: an empty crop, or a band that collapses (a 1-voxel-thick sheet
    with no core, a tiny cell) yields NaN for the affected columns rather than
    crashing. Always produced (additive; the labels-only output is otherwise
    unchanged) — there is no Wave-C gate (this is intrinsic, not signed-frame).

    Args:
        labels_3d: 3D integer label array (Z, Y, X). Background is 0.
        df: DataFrame from :func:`extract_features` (modified in-place).
        spacing: voxel spacing ``(Z, Y, X)`` for an anisotropy-aware shell EDT
            (the SAME partition as :func:`compute_shell_features`). ``None`` = voxel
            units. NOTE: the external-FACE counts are voxel-face counts (the
            compactness + the surface ratio are dimensionless), so spacing only
            affects the radial PARTITION, not the face tallies.

    Returns:
        The same DataFrame with the 4 new columns added.
    """
    from scipy.ndimage import find_objects

    n = len(df)
    if n == 0:
        for c in MULTISCALE_SHAPE_COLUMNS:
            df[c] = pd.Series(dtype=np.float64)
        return df

    sph_inner = np.full(n, np.nan, dtype=np.float64)
    sph_outer = np.full(n, np.nan, dtype=np.float64)
    surf_ratio = np.full(n, np.nan, dtype=np.float64)
    comp_ratio = np.full(n, np.nan, dtype=np.float64)
    # Wave N multi-scale remainder.
    sph_mid = np.full(n, np.nan, dtype=np.float64)
    q1_frac = np.full(n, np.nan, dtype=np.float64)
    q4_frac = np.full(n, np.nan, dtype=np.float64)
    q1_q4_ratio = np.full(n, np.nan, dtype=np.float64)

    slices = find_objects(labels_3d.astype(np.int32))
    label_ids = df["label"].values
    eps = 1e-12

    # Wave N — the axis-quartile template needs the major-axis direction +
    # centroid extract_features emitted; absent -> those columns stay NaN.
    major_cols = ["major_axis_z", "major_axis_y", "major_axis_x"]
    have_major = all(c in df.columns for c in major_cols)
    major_vals = df[major_cols].values.astype(np.float64) if have_major else None
    sp_q = np.asarray(spacing, dtype=np.float64) if spacing is not None else np.ones(3)

    for idx in range(n):
        label_id = int(label_ids[idx])
        if label_id < 1 or label_id > len(slices) or slices[label_id - 1] is None:
            continue
        mask_crop = labels_3d[slices[label_id - 1]] == label_id
        if not mask_crop.any():
            continue

        dt = _object_edt(mask_crop, spacing=spacing)
        d = dt[mask_crop]
        d_max = float(d.max()) if d.size else 0.0
        if d_max > eps:
            # radial-position field on the SAME thresholds as M5 (equal-volume shells).
            r_field = np.zeros_like(dt, dtype=np.float64)
            r_field[mask_crop] = dt[mask_crop] / d_max
            inner_band = mask_crop & (r_field > _SHELL_R2)        # the core
            mid_band = mask_crop & (r_field > _SHELL_R1) & (r_field <= _SHELL_R2)
            outer_band = mask_crop & (r_field <= _SHELL_R1)       # the surface rind

            si = _voxel_band_compactness(inner_band)
            sm = _voxel_band_compactness(mid_band)
            so = _voxel_band_compactness(outer_band)
            sph_inner[idx] = si
            sph_mid[idx] = sm
            sph_outer[idx] = so
            if np.isfinite(si) and np.isfinite(so):
                comp_ratio[idx] = si / max(so, eps)

            # surface-to-bulk ratio = the 1-voxel SKIN fraction (true boundary
            # voxels that touch background, via a single 6-connectivity erosion)
            # / total. A thin sheet -> nearly all skin (~1); a fat blob -> a
            # smaller skin fraction.
            from scipy.ndimage import binary_erosion
            n_vox = int(mask_crop.sum())
            if n_vox > 0:
                eroded = binary_erosion(mask_crop)     # interior (non-skin) voxels
                skin = int((mask_crop & ~eroded).sum())
                surf_ratio[idx] = skin / n_vox
        # else: a 1-voxel-thick sheet has no radial structure — the shell-shape
        # columns stay NaN, but the axis-quartile template below is still defined.

        # --- axis-quartile volume-fraction template ( <feature>_axis_q1/q4)
        # Project the object's voxels onto the major axis, normalize to t ∈ [0,1],
        # and report the fraction of voxels in the FIRST (t < 0.25) and LAST
        # (t >= 0.75) axial quartile. A symmetric spindle -> q1 ≈ q4; a cone /
        # vase -> imbalanced. q1_q4_ratio = q1 / q4 (>1 = fatter near t=0).
        if have_major:
            direction = major_vals[idx]
            norm = float(np.sqrt(np.sum(direction ** 2)))
            coords = np.argwhere(mask_crop)
            if norm > 1e-12 and coords.shape[0] >= 4:
                direction = direction / norm
                disp = (coords - coords.mean(axis=0)) * sp_q
                s = disp @ direction
                s_min, s_max = float(s.min()), float(s.max())
                s_range = s_max - s_min
                if s_range > 1e-12:
                    t = (s - s_min) / s_range
                    ntot = float(t.size)
                    f1 = float(np.count_nonzero(t < 0.25)) / ntot
                    f4 = float(np.count_nonzero(t >= 0.75)) / ntot
                    q1_frac[idx] = f1
                    q4_frac[idx] = f4
                    q1_q4_ratio[idx] = (f1 / f4) if f4 > eps else np.nan

    df["shell_sphericity_inner"] = sph_inner
    df["shell_sphericity_outer"] = sph_outer
    df["shell_surface_ratio"] = surf_ratio
    df["shell_compactness_ratio"] = comp_ratio
    # Wave N multi-scale remainder.
    df["shell_sphericity_mid"] = sph_mid
    df["shape_axis_q1_fraction"] = q1_frac
    df["shape_axis_q4_fraction"] = q4_frac
    df["shape_axis_q1_q4_ratio"] = q1_q4_ratio
    return df


def _object_surface_mesh(crop_mask: np.ndarray, sp: Tuple[float, float, float]):
    """Marching-cubes surface mesh of a single-object crop (verts, faces) or None.

    Mirrors EXACTLY the mesh :func:`compute_surface_features` builds (pad the
    object with 1 voxel of background, ``marching_cubes(level=0.5, spacing=sp)``)
    so the curvature family reads the SAME surface — keeping the two families
    consistent without recomputing the surface area (``compute_surface_features``
    output is untouched). Returns ``(verts, faces)`` in physical units, or
    ``None`` when the mesh is degenerate / undefined (tiny object, flat patch).
    """
    from skimage.measure import marching_cubes

    crop = np.pad(crop_mask.astype(np.uint8), 1, mode="constant", constant_values=0)
    try:
        verts, faces, _, _ = marching_cubes(crop, level=0.5, spacing=sp)
    except (RuntimeError, ValueError):
        return None
    if verts.shape[0] < 4 or faces.shape[0] < 4:
        return None
    return verts, faces


def _discrete_mean_curvature(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Per-vertex discrete mean curvature via the umbrella (uniform Laplacian) proxy.

    A robust, cotangent-free estimator: the discrete Laplace operator with
    uniform weights ``L(v_i) = mean_j(v_j) - v_i`` (``j`` = the 1-ring neighbours)
    approximates ``-H · n`` (mean curvature times the inward normal) up to a
    scale. We take the SIGNED magnitude ``H_i = (L · n_i)`` where ``n_i`` is the
    area-weighted vertex normal (outward by marching-cubes winding) — positive on
    convex bumps, negative in concave pits — and return one value per vertex.

    Cotangent / circumradius weights are deliberately avoided: they divide by
    triangle areas and blow up on the sliver triangles marching-cubes routinely
    produces, giving unstable spikes. The uniform-weight Laplacian is bounded and
    stable on exactly those meshes, which is what we want for a distribution
    summary (mean / std / |mean|) rather than an exact pointwise curvature.

    Returns a ``(n_verts,)`` float array; vertices with no incident faces get 0.

    NOTE: the returned ``H`` is a normal DISPLACEMENT — it carries a LENGTH unit and
    scales LINEARLY with the object's physical size, so it is NOT comparable across
    cells as-is. The caller (:func:`compute_curvature_features`) DIVIDES it by the
    object's equivalent-sphere radius (also a length) to make the summarized
    curvature dimensionless + size-invariant (B5).
    """
    nv = verts.shape[0]
    # --- area-weighted vertex normals (outward) -----------------------------
    tri = verts[faces]                                  # (F, 3, 3)
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])  # face normals*2A
    vnorm = np.zeros((nv, 3), dtype=np.float64)
    for k in range(3):
        np.add.at(vnorm, faces[:, k], fn)               # area-weighted accumulate
    ln = np.linalg.norm(vnorm, axis=1, keepdims=True)
    vnorm = np.divide(vnorm, ln, out=np.zeros_like(vnorm), where=ln > 1e-12)

    # --- uniform 1-ring centroid (umbrella) Laplacian -----------------------
    nbr_sum = np.zeros((nv, 3), dtype=np.float64)
    nbr_cnt = np.zeros(nv, dtype=np.float64)
    # each undirected mesh edge (a,b) contributes b to a's ring and vice versa.
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    a = edges[:, 0]
    b = edges[:, 1]
    np.add.at(nbr_sum, a, verts[b])
    np.add.at(nbr_sum, b, verts[a])
    np.add.at(nbr_cnt, a, 1.0)
    np.add.at(nbr_cnt, b, 1.0)
    valid = nbr_cnt > 0
    centroid = np.zeros_like(verts)
    centroid[valid] = nbr_sum[valid] / nbr_cnt[valid, None]
    lap = np.zeros_like(verts)
    lap[valid] = centroid[valid] - verts[valid]         # umbrella vector

    # signed mean curvature proxy = projection of the Laplacian on the normal.
    H = np.einsum("ij,ij->i", lap, vnorm)
    H[~valid] = 0.0
    return H


def _discrete_gaussian_curvature(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Per-vertex discrete Gaussian curvature via the angle-deficit estimator (B-WaveB).

    The standard discrete Gauss curvature on a triangle mesh: the *angle deficit*
    at each vertex normalized by its barycentric (1/3-of-incident-face) area::

        K_i = (2π − Σ_f θ_{i,f}) / (A_i / 3)

    where ``θ_{i,f}`` is the interior angle of incident triangle ``f`` at vertex
    ``i`` and ``A_i`` is the summed area of the incident triangles. By the
    Gauss–Bonnet theorem ``Σ_i K_i A_i / 3 = 2π·χ``, so this is the principled
    discrete Gaussian curvature (positive on convex caps, negative at saddles, ≈0
    on cylinders/planes).

    Like :func:`_discrete_mean_curvature`, this carries a LENGTH⁻² unit (it scales
    as ``1/size²``), so the caller normalizes by the equivalent-sphere radius
    SQUARED to make the summary dimensionless + size-invariant (a sphere of any
    size then scores the same ``K·R² ≈ 1``). Degenerate/sliver triangles
    (zero-area incident faces) contribute a 0-angle / 0-area term and are skipped
    in the area accumulation; a vertex with no valid area gets K=0.

    Returns a ``(n_verts,)`` float array.
    """
    nv = verts.shape[0]
    tri = verts[faces]                                   # (F, 3, 3)
    # The three edge vectors of every triangle and their lengths.
    e0 = tri[:, 1] - tri[:, 0]
    e1 = tri[:, 2] - tri[:, 1]
    e2 = tri[:, 0] - tri[:, 2]
    # Triangle area (½‖e0 × (−e2)‖ == ½‖(v1−v0)×(v2−v0)‖).
    cross = np.cross(e0, -e2)
    area = 0.5 * np.linalg.norm(cross, axis=1)           # (F,)

    # Interior angle at each of the three corners (angle between the two edges
    # leaving that corner). Corner 0: edges (v1−v0)=e0 and (v2−v0)=−e2.
    def _angle(u, v):
        un = np.linalg.norm(u, axis=1)
        vn = np.linalg.norm(v, axis=1)
        denom = un * vn
        cos = np.einsum("ij,ij->i", u, v)
        with np.errstate(divide="ignore", invalid="ignore"):
            cos = np.where(denom > 1e-20, cos / denom, 0.0)
        return np.arccos(np.clip(cos, -1.0, 1.0))

    ang0 = _angle(e0, -e2)                                # at v0
    ang1 = _angle(e1, -e0)                                # at v1
    ang2 = _angle(e2, -e1)                                # at v2

    angle_sum = np.full(nv, 2.0 * np.pi, dtype=np.float64)
    area_sum = np.zeros(nv, dtype=np.float64)
    for col, ang in ((0, ang0), (1, ang1), (2, ang2)):
        vi = faces[:, col]
        np.subtract.at(angle_sum, vi, ang)               # 2π − Σθ
        np.add.at(area_sum, vi, area)                    # incident-triangle area

    K = np.zeros(nv, dtype=np.float64)
    # Barycentric vertex area = A_i / 3; K = deficit / barycentric area.
    bary = area_sum / 3.0
    valid = bary > 1e-20
    K[valid] = angle_sum[valid] / bary[valid]
    return K


# neck_count tuning. A "neck" is a strong cross-sectional GIRDLE — a constriction
# between two larger lobes (the dumbbell/merge signature). Mesh-curvature-based
# detection is unreliable on staircased marching-cubes meshes (the angle-deficit
# Gaussian curvature is dominated by voxelization noise — a smooth sphere shows
# spurious saddles), so the neck count instead reads the robust CROSS-SECTION
# AREA PROFILE along the object's longest axis: a neck is a pronounced interior
# VALLEY (local minimum) flanked by larger area on both sides. This is the same
# voxel-profile primitive the cross-section family uses, and it is stable.
#   _NECK_DEPTH_FRAC : the valley must dip at least this fraction below the
#       SMALLER of its two flanking peaks to count (a shallow wobble is ignored).
_NECK_SMOOTH_ITERS = 5                # curvature-field smoothing (bending/skew)
_NECK_DEPTH_FRAC = 0.30               # valley depth vs the lower flanking peak


def _count_profile_necks(mask_crop: np.ndarray) -> int:
    """Count strong cross-sectional girdles (necks) from the area profile.

    Slices the object along its LONGEST bounding-box axis, builds the per-slice
    cross-section AREA profile ``A(t)``, smooths it, and counts interior local
    MINIMA whose depth below the SMALLER of the two flanking maxima is at least
    :data:`_NECK_DEPTH_FRAC` of that flanking peak. Each such valley is a pinch
    between two larger lobes — a dumbbell yields 1, a smooth ball or a straight
    rod yields 0 (no interior valley with real depth).

    Robust to staircased marching-cubes geometry (it reads the voxel profile, not
    the mesh) and to short objects (``< 5`` slices along the long axis → 0).
    """
    m = np.asarray(mask_crop).astype(bool)
    if not m.any():
        return 0
    # Profile along the LONGEST bbox axis (the dumbbell's lobe-to-lobe direction).
    long_axis = int(np.argmax(m.shape))
    other = tuple(ax for ax in range(3) if ax != long_axis)
    areas = m.sum(axis=other).astype(np.float64)          # A(t) along long_axis
    present = areas > 0
    if present.sum() < 5:
        return 0
    z_idx = np.where(present)[0]
    a = areas[z_idx[0]:z_idx[-1] + 1]                     # contiguous present span
    if a.size < 5:
        return 0
    # Smooth to kill voxel-jitter wobbles before peak/valley finding.
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    a = gaussian_filter1d(a, sigma=1.0, mode="nearest")

    # Interior local minima = peaks of the negated profile (boundaries excluded —
    # a true neck is flanked by larger area on BOTH sides, so it can't be an end).
    valleys, _ = find_peaks(-a)
    if valleys.size == 0:
        return 0
    n = 0
    for v in valleys:
        left_max = float(a[:v].max()) if v > 0 else 0.0
        right_max = float(a[v + 1:].max()) if v + 1 < a.size else 0.0
        flank = min(left_max, right_max)
        if flank <= 0:
            continue
        depth = (flank - float(a[v])) / flank
        if depth >= _NECK_DEPTH_FRAC:
            n += 1
    return n


def _smooth_over_mesh(values: np.ndarray, faces: np.ndarray, iters: int) -> np.ndarray:
    """Average a per-vertex scalar field over the mesh 1-ring for ``iters`` passes.

    A uniform (umbrella) Laplacian SMOOTH of the scalar: each pass replaces a
    vertex value with the mean of itself + its edge neighbours. Kills scattered
    marching-cubes noise while preserving spatially-coherent structure (a pinch
    ring stays negative; isolated noise specks regress to the mean).
    """
    nv = values.shape[0]
    if nv == 0 or faces.shape[0] < 1 or iters <= 0:
        return values
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    a = edges[:, 0]
    b = edges[:, 1]
    deg = np.zeros(nv, dtype=np.float64)
    np.add.at(deg, a, 1.0)
    np.add.at(deg, b, 1.0)
    out = values.astype(np.float64).copy()
    for _ in range(int(iters)):
        nbr_sum = np.zeros(nv, dtype=np.float64)
        np.add.at(nbr_sum, a, out[b])
        np.add.at(nbr_sum, b, out[a])
        with np.errstate(invalid="ignore", divide="ignore"):
            avg = np.where(deg > 0, (nbr_sum + out) / (deg + 1.0), out)
        out = avg
    return out


# Number of azimuthal sectors for the radial-symmetry-about-the-major-axis score.
_RADIAL_SYMMETRY_NSECTORS = 12


def _reflection_iou(mask_crop: np.ndarray, normal_zyx: np.ndarray, sp) -> float:
    """IoU of a crop mask with its reflection across the centroid plane ⊥ ``normal``.

    Reflects every object voxel across the plane through the (physical-frame)
    centroid with the given unit ``normal`` (ZYX), in SPACING-AWARE physical
    coordinates (so anisotropic data mirrors the right geometry), rounds back to
    voxel indices, and returns ``IoU(mask, reflected)`` ∈ [0, 1]. NaN for an empty
    mask. Shared by the bilateral (normal = mid axis) + apical/basal (normal =
    major axis) symmetry scores.
    """
    vox = np.argwhere(np.asarray(mask_crop, dtype=bool)).astype(np.float64)
    if vox.shape[0] < 1:
        return np.nan
    nrm = float(np.sqrt(np.sum(np.asarray(normal_zyx) ** 2)))
    if nrm <= 1e-12:
        return np.nan
    axis = np.asarray(normal_zyx, dtype=np.float64) / nrm
    sp_arr = np.asarray(sp, dtype=np.float64)
    pts = vox * sp_arr[None, :]
    centroid = pts.mean(axis=0)
    d = (pts - centroid) @ axis
    refl = pts - 2.0 * d[:, None] * axis[None, :]
    refl_idx = np.rint(refl / sp_arr[None, :]).astype(np.int64)
    shape = np.array(mask_crop.shape)
    inb = np.all((refl_idx >= 0) & (refl_idx < shape), axis=1)
    refl_vol = np.zeros_like(mask_crop, dtype=bool)
    ri = refl_idx[inb]
    if ri.shape[0] > 0:
        refl_vol[ri[:, 0], ri[:, 1], ri[:, 2]] = True
    inter = float(np.count_nonzero(mask_crop & refl_vol))
    union = float(np.count_nonzero(mask_crop | refl_vol))
    return (inter / union) if union > 0 else np.nan


def _radial_symmetry_about_axis(mask_crop: np.ndarray, axis_zyx: np.ndarray, sp) -> float:
    """Rotational-symmetry score about the major axis ∈ [0, 1].

    Projects each object voxel onto the plane ⊥ the (unit) major ``axis``, bins by
    azimuth into :data:`_RADIAL_SYMMETRY_NSECTORS` equal angular sectors, and
    returns ``1 / (1 + cv)`` of the per-sector voxel counts — a body of revolution
    (cylinder, sphere, cone) fills every sector evenly (cv → 0, score → 1); a flat
    slab / wedge concentrates voxels in a few sectors (high cv, low score).
    Spacing-aware (azimuth is measured in physical coordinates). NaN for a
    degenerate object (< nsectors voxels or no perpendicular spread).
    """
    vox = np.argwhere(np.asarray(mask_crop, dtype=bool)).astype(np.float64)
    nsec = _RADIAL_SYMMETRY_NSECTORS
    if vox.shape[0] < nsec:
        return np.nan
    nrm = float(np.sqrt(np.sum(np.asarray(axis_zyx) ** 2)))
    if nrm <= 1e-12:
        return np.nan
    axis = np.asarray(axis_zyx, dtype=np.float64) / nrm
    sp_arr = np.asarray(sp, dtype=np.float64)
    pts = vox * sp_arr[None, :]
    pts = pts - pts.mean(axis=0)
    # Two orthonormal in-plane basis vectors perpendicular to `axis`.
    tmp = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = tmp - np.dot(tmp, axis) * axis
    un = float(np.sqrt(np.sum(u ** 2)))
    if un <= 1e-12:
        return np.nan
    u = u / un
    v = np.cross(axis, u)
    pu = pts @ u
    pv = pts @ v
    rho = np.sqrt(pu ** 2 + pv ** 2)
    if float(rho.max()) <= 1e-9:
        return np.nan
    theta = np.arctan2(pv, pu)                       # [-pi, pi]
    bins = np.clip(((theta + np.pi) / (2.0 * np.pi) * nsec).astype(np.int64),
                   0, nsec - 1)
    counts = np.bincount(bins, minlength=nsec).astype(np.float64)
    mean_c = float(counts.mean())
    if mean_c <= 1e-12:
        return np.nan
    cv = float(counts.std()) / mean_c
    return 1.0 / (1.0 + cv)


def compute_curvature_features(
    labels_3d: np.ndarray,
    df: pd.DataFrame,
    spacing: Optional[Tuple[float, ...]] = None,
) -> pd.DataFrame:
    """Surface curvature / concavity / symmetry per cell (M5 / the feature catalogue; C13).

    Three families derived from the object's surface mesh + voxel mask. All are
    intrinsic shape descriptors (``frame="intrinsic"``).

    **Curvature** (from the marching-cubes mesh — the SAME mesh
    :func:`compute_surface_features` builds, via :func:`_object_surface_mesh`):
    per-vertex discrete mean curvature ``H`` is estimated with the robust
    umbrella-Laplacian proxy (:func:`_discrete_mean_curvature`; cotangent-free,
    NaN/spike-free on marching-cubes slivers), then summarized.
        ``mean_curvature_mean`` — mean signed ``H`` over the surface vertices
            (≈ 0 for a sphere-like rind; biased negative/positive by net
            concavity/convexity). SCALE-INVARIANT: the raw per-vertex umbrella
            curvature (a length) is DIVIDED by the object's equivalent-sphere radius,
            so two geometrically-similar cells of different size score the same.
            Dimensionless; unbounded.
        ``mean_curvature_std`` — std of the size-normalized ``H``: how variable
            the surface bending is (high for lobed / wrinkled / pinched merges).
            Dimensionless; unbounded.
        ``abs_curvature_mean`` — mean ``|H|`` of the size-normalized curvature:
            total absolute bending magnitude irrespective of sign (a smooth sphere
            is low; a bumpy/pinched mask is high). Dimensionless; unbounded.

    **Concavity** (convex hull minus object): the object's solid voxel mask is
    compared to its filled convex hull. The hull-minus-object voxels are the
    concave "pockets" (necks, dumbbell waists, bites).
        ``convexity`` — VOLUMETRIC convexity ``object_volume / hull_volume`` ∈
            (0, 1] (bounded): 1 = perfectly convex (the object fills its hull),
            < 1 = concave (a dumbbell / L-shape / bitten mask leaves hull volume
            empty). A volume ratio (not a surface-area ratio) is used because the
            surface-area ratio is NON-monotone for elongated concave shapes — a
            dumbbell's hull is a long capsule whose surface area can EXCEED the
            dumbbell's, so the surface ratio fails to flag it; the volume ratio is
            monotone and the standard shape-analysis "convexity". (This parallels
            the existing ``solidity`` column, which subsamples surface voxels for a
            fast hull; ``convexity`` here is the exact filled-hull volume ratio.)
        ``concavity_count`` — number of DISTINCT concave pockets = 26-connected
            components of (hull ∧ ¬object), excluding negligible 1–2 voxel
            specks. A solid convex blob → 0; a dumbbell → ≥ 1. Count, unbounded.
        ``max_concavity_depth`` — the deepest pocket: ``max`` over the
            hull-minus-object voxels of their EDT to the nearest object/outside
            voxel (how far the bite reaches in). 0 for a convex object; carries
            the spacing unit. Unbounded.

    **Symmetry** (voxel mask reflection): the mask is reflected across its
    BILATERAL mid-plane — the plane through the centroid CONTAINING the major
    (longest) inertia axis, with normal = the MID inertia axis — and the
    reflection is compared to the original by IoU. (Reflecting across this plane
    tests left/right mirror symmetry about the long axis; using the major axis as
    the plane NORMAL instead would test the two ENDS, which an L-shape passes.)
        ``bilateral_symmetry_score`` — ``IoU(mask, reflected_mask)`` ∈ [0, 1]
            (bounded): 1 = perfectly bilaterally symmetric (sphere, straight rod);
            an L-shape / hook scores low. Reads the ``mid_axis_*`` + ``centroid-*``
            columns :func:`extract_features` emits; NaN if absent or degenerate.

    **Curvature REMAINDER** (Wave B — the mesh-curvature columns the M5
    pass left out, derived from the SAME marching-cubes mesh):
        ``gaussian_curvature_mean`` — mean per-vertex Gaussian curvature ``K``
            (angle-deficit / :func:`_discrete_gaussian_curvature`), SCALE-INVARIANT
            via division by the equivalent-sphere radius SQUARED (``K`` carries a
            length⁻² unit). ≈ +1 for a sphere-like rind, ~0 on a cylinder/saddle
            average. Dimensionless; unbounded.
        ``gaussian_curvature_std`` — std of the size-normalized ``K``: surface
            saddle/cap heterogeneity (high on lobed / pinched merges).
            Dimensionless; unbounded.
        ``bending_energy`` — the (size-normalized) Willmore-style ∫H² bending
            energy ``mean(Hn²)`` over the surface vertices (Hn = the SAME size-
            normalized mean curvature the existing columns use), so it is
            SCALE-INVARIANT: a smooth sphere is low, a bumpy/wrinkled mask high.
            Dimensionless; unbounded.
        ``mean_curvature_skew`` — Fisher skewness of the signed size-normalized
            mean curvature ``Hn``: a one-sided bias toward convex bumps vs concave
            pits (a dumbbell waist skews negative). Dimensionless; unbounded.
        ``neck_count`` — number of strong cross-sectional "girdles": pronounced
            interior VALLEYS in the area profile ``A(t)`` along the object's
            longest axis (each flanked by larger lobes), a merge/pinch signal.
            Read from the robust voxel profile (:func:`_count_profile_necks`),
            NOT the noisy mesh curvature — a smooth ball / straight rod → 0, a
            dumbbell → 1. Count, unbounded.

    **Curvature REMAINDER — Wave I** (the last two mesh-curvature columns,
    from the SAME per-vertex curvature fields above; cheap derivations):
        ``curvature_negative_fraction`` — fraction of surface vertices with
            NEGATIVE (concave) size-normalized mean curvature ``Hn`` ∈ [0, 1]
            (bounded). ~0 for a smooth convex blob / sphere (the rind is
            outward-curving), elevated where the surface dips inward (necks,
            saddles, dumbbell waists, bitten masks). A direct concavity-coverage
            QC cue.
        ``gaussian_curvature_skew`` — Fisher skewness of the size-normalized
            Gaussian curvature ``Kn`` over the surface vertices: a one-sided bias
            toward elliptic caps (K>0) vs hyperbolic saddles (K<0). 0 for a
            symmetric distribution; dimensionless, unbounded.

    Spacing-aware: the mesh + EDT respect ``spacing``; curvature carries the
    inverse-length unit, ``max_concavity_depth`` the length unit, and the
    fractions/scores are dimensionless.

    Degenerate shapes (empty crop, tiny mesh, QhullError on a flat/collinear
    point set, missing axis columns) yield NaN for the affected summaries rather
    than crashing.

    Args:
        labels_3d: 3D integer label array (Z, Y, X). Background is 0.
        df: DataFrame from :func:`extract_features` (modified in-place). The
            symmetry score additionally needs ``mid_axis_{z,y,x}``; absent → NaN
            symmetry only.
        spacing: voxel spacing ``(Z, Y, X)``. None = voxel units.

    Returns:
        The same DataFrame with the 6 M5 columns + the 5 Wave-B remainder columns.
    """
    from scipy.ndimage import find_objects, label as cc_label, distance_transform_edt

    n = len(df)
    cols = (
        "mean_curvature_mean", "mean_curvature_std", "abs_curvature_mean",
        "convexity", "concavity_count", "max_concavity_depth",
        "bilateral_symmetry_score",
        # Wave B curvature remainder.
        "gaussian_curvature_mean", "gaussian_curvature_std", "bending_energy",
        "mean_curvature_skew", "neck_count",
        # Wave I curvature remainder.
        "curvature_negative_fraction", "gaussian_curvature_skew",
        # Wave N curvature/symmetry remainder.
        "radial_symmetry_score", "apical_basal_symmetry_score",
    )
    if n == 0:
        for col in cols:
            df[col] = pd.Series(dtype=np.float64)
        return df

    curv_mean = np.full(n, np.nan, dtype=np.float64)
    curv_std = np.full(n, np.nan, dtype=np.float64)
    abs_curv = np.full(n, np.nan, dtype=np.float64)
    convexity = np.full(n, np.nan, dtype=np.float64)
    conc_count = np.full(n, np.nan, dtype=np.float64)
    conc_depth = np.full(n, np.nan, dtype=np.float64)
    sym_score = np.full(n, np.nan, dtype=np.float64)
    # Wave B curvature remainder.
    gauss_mean = np.full(n, np.nan, dtype=np.float64)
    gauss_std = np.full(n, np.nan, dtype=np.float64)
    bend_energy = np.full(n, np.nan, dtype=np.float64)
    curv_skew = np.full(n, np.nan, dtype=np.float64)
    neck_cnt = np.full(n, np.nan, dtype=np.float64)
    # Wave I curvature remainder.
    curv_neg_frac = np.full(n, np.nan, dtype=np.float64)
    gauss_skew = np.full(n, np.nan, dtype=np.float64)
    # Wave N curvature/symmetry remainder.
    radial_sym = np.full(n, np.nan, dtype=np.float64)
    ab_sym = np.full(n, np.nan, dtype=np.float64)

    sp = tuple(float(s) for s in spacing) if spacing is not None else (1.0, 1.0, 1.0)
    slices = find_objects(labels_3d.astype(np.int32))
    label_ids = df["label"].values
    struct_26 = np.ones((3, 3, 3), dtype=bool)

    # The bilateral mid-plane NORMAL is the MID inertia axis (the plane then
    # contains the major axis). The centroid is taken crop-local per object.
    axis_cols = ["mid_axis_z", "mid_axis_y", "mid_axis_x"]
    have_axis = all(c in df.columns for c in axis_cols)
    axis_vals = df[axis_cols].values.astype(np.float64) if have_axis else None
    # Wave N — the MAJOR inertia axis powers the radial (about-the-long-axis) +
    # the apical/basal (across-the-long-axis) symmetry scores.
    major_cols = ["major_axis_z", "major_axis_y", "major_axis_x"]
    have_major = all(c in df.columns for c in major_cols)
    major_vals = df[major_cols].values.astype(np.float64) if have_major else None

    for idx, label_id in enumerate(label_ids):
        if label_id < 1 or label_id > len(slices):
            continue
        obj_slice = slices[label_id - 1]
        if obj_slice is None:
            continue

        mask_crop = labels_3d[obj_slice] == label_id
        if not mask_crop.any():
            continue

        # ----- curvature (object surface mesh) -----------------------------
        # Same marching-cubes mesh compute_surface_features builds (output of
        # that function is untouched — this only READS an equivalent mesh).
        mesh = _object_surface_mesh(mask_crop, sp)
        if mesh is not None:
            verts, faces = mesh
            H = _discrete_mean_curvature(verts, faces)
            if H.size >= 1 and np.all(np.isfinite(H)):
                # SCALE-INVARIANCE (B5): the raw umbrella-Laplacian H is a normal
                # DISPLACEMENT and carries a LENGTH unit — it scales LINEARLY with
                # the object's physical size (verified: doubling the mesh spacing
                # doubles raw H), so two geometrically-similar cells of different
                # size get DIFFERENT raw H and the value is NOT comparable across
                # cells. DIVIDING by the object's equivalent-sphere RADIUS (also a
                # length) yields a DIMENSIONLESS, size-invariant curvature: H / R is
                # constant across sizes (verified exactly under spacing scaling).
                # R is from the physical volume (voxel count * voxel volume), so
                # spacing is respected. A degenerate zero-volume object leaves the
                # summaries NaN.
                obj_vox = float(np.count_nonzero(mask_crop))
                phys_vol = obj_vox * float(sp[0] * sp[1] * sp[2])
                r_equiv = (3.0 * phys_vol / (4.0 * np.pi)) ** (1.0 / 3.0) if phys_vol > 0 else 0.0
                if r_equiv > 0:
                    Hn = H / r_equiv
                    curv_mean[idx] = float(Hn.mean())
                    curv_std[idx] = float(Hn.std()) if Hn.size >= 2 else 0.0
                    abs_curv[idx] = float(np.abs(Hn).mean())
                    # Wave I — fraction of vertices with concave (negative) mean
                    # curvature: a direct concavity-coverage cue (~0 for a smooth
                    # convex rind, elevated at necks / saddles / bites).
                    curv_neg_frac[idx] = float(np.mean(Hn < 0.0))

                    # --- Wave B curvature remainder ---------------
                    # All reuse the SAME mesh + the SAME size-normalization
                    # convention as the existing columns, so they're directly
                    # comparable across cells of different size.
                    #
                    # The raw per-vertex umbrella curvature is NOISY on a
                    # marching-cubes mesh, so the derived shape measures
                    # (bending / skew / neck) read a SMOOTHED curvature field
                    # (a few mesh-Laplacian passes) — averaging out the
                    # scattered sliver-triangle noise while genuine, spatially-
                    # coherent bending (a bumpy or pinched mask) survives. (The
                    # existing mean/std/abs columns keep the RAW field unchanged.)
                    Hsm = _smooth_over_mesh(Hn, faces, _NECK_SMOOTH_ITERS)
                    # bending_energy = mean(Hsm²): the size-normalized Willmore
                    # ∫H² energy proxy (a smooth sphere is low, a wrinkled/
                    # pinched mask high). Hsm is already dimensionless.
                    bend_energy[idx] = float(np.mean(Hsm ** 2))
                    # mean_curvature_skew: Fisher skewness of the signed smoothed
                    # curvature — a one-sided bias toward bumps vs pits (a
                    # dumbbell waist skews the distribution). Degenerate -> 0.
                    sd = float(Hsm.std())
                    if Hsm.size >= 3 and sd > 1e-9:
                        from scipy.stats import skew as _skew
                        curv_skew[idx] = float(_skew(Hsm))
                    else:
                        curv_skew[idx] = 0.0
                    # neck_count: strong cross-sectional GIRDLES = pinch valleys
                    # in the area profile along the long axis (a merge/pinch
                    # signal). Read from the robust voxel profile (NOT the noisy
                    # mesh curvature): a dumbbell -> 1, a smooth ball / rod -> 0.
                    neck_cnt[idx] = float(_count_profile_necks(mask_crop))

                    # Gaussian curvature K (angle deficit) — carries length⁻²,
                    # so normalize by R² to make K·R² dimensionless + size-
                    # invariant (a sphere of any size -> ≈ 1).
                    K = _discrete_gaussian_curvature(verts, faces)
                    if K.size >= 1 and np.all(np.isfinite(K)):
                        Kn = K * (r_equiv ** 2)
                        gauss_mean[idx] = float(Kn.mean())
                        gauss_std[idx] = float(Kn.std()) if Kn.size >= 2 else 0.0
                        # Wave I — Fisher skewness of the Gaussian curvature
                        # field (cap-vs-saddle asymmetry). Degenerate -> 0.
                        ksd = float(Kn.std())
                        if Kn.size >= 3 and ksd > 1e-9:
                            from scipy.stats import skew as _skew
                            gauss_skew[idx] = float(_skew(Kn))
                        else:
                            gauss_skew[idx] = 0.0

        # ----- concavity (convex hull minus object) ------------------------
        # Fill the convex hull on the (padded) crop voxel grid so the
        # hull-minus-object pockets are well defined.
        padded = np.pad(mask_crop, 1, mode="constant", constant_values=False)
        coords = np.argwhere(padded)
        if coords.shape[0] >= 4:
            try:
                from scipy.spatial import ConvexHull, Delaunay
                hull = ConvexHull(coords.astype(np.float64))
                tri = Delaunay(coords[hull.vertices].astype(np.float64))
                grid = np.argwhere(np.ones_like(padded))  # all voxel centers
                inside = tri.find_simplex(grid.astype(np.float64)) >= 0
                hull_mask = inside.reshape(padded.shape)
                # Concave pockets = hull voxels not in the object.
                pockets = hull_mask & (~padded)
                if pockets.any():
                    lab_p, n_p = cc_label(pockets, structure=struct_26)
                    if n_p > 0:
                        sizes = np.bincount(lab_p.ravel())[1:]
                        # Drop negligible 1-2 voxel surface-staircase specks.
                        conc_count[idx] = float(int(np.count_nonzero(sizes >= 3)))
                    else:
                        conc_count[idx] = 0.0
                    # Max concavity DEPTH: EDT of the pocket voxels to the
                    # nearest non-pocket voxel (object or outside-hull).
                    dt = distance_transform_edt(pockets, sampling=sp)
                    conc_depth[idx] = float(dt.max())
                else:
                    conc_count[idx] = 0.0
                    conc_depth[idx] = 0.0
                # convexity = VOLUMETRIC object_vol / hull_vol (monotone + bounded;
                # the surface-area ratio is non-monotone for elongated concave
                # shapes — see the docstring). Both are voxel counts on the same
                # grid, so spacing cancels.
                hull_vox = float(np.count_nonzero(hull_mask))
                obj_vox = float(np.count_nonzero(padded))
                if hull_vox > 0:
                    convexity[idx] = float(min(1.0, obj_vox / hull_vox))
            except Exception:
                # QhullError / degenerate (collinear/coplanar) point set.
                pass

        # ----- bilateral symmetry (reflect across the bilateral mid-plane) -
        # The plane through the centroid CONTAINING the major axis, with normal
        # = the MID inertia axis. Reflecting across it tests left/right mirror
        # symmetry about the long axis (an L-shape / hook fails); using the major
        # axis as the normal instead would test the two ends, which an L passes.
        if have_axis:
            direction = axis_vals[idx]                       # mid_axis_* (the normal)
            norm = float(np.sqrt(np.sum(direction ** 2)))
            vox = np.argwhere(mask_crop).astype(np.float64)  # crop-local VOXEL coords
            if norm > 1e-12 and vox.shape[0] >= 1:
                axis = direction / norm                      # ZYX unit plane-normal
                # The ``mid_axis_*`` normal lives in the PHYSICAL (spacing-aware)
                # inertia frame (extract_features ran regionprops with spacing), so
                # the reflection must happen in physical coordinates — reflecting
                # voxel indices across a physical-frame plane mirrors the wrong
                # geometry on anisotropic data. Convert voxel -> physical, reflect,
                # convert back -> voxel, then round to the nearest voxel index.
                sp_arr = np.asarray(sp, dtype=np.float64)    # (Z, Y, X), 1s if None
                pts = vox * sp_arr[None, :]                  # physical coords
                local_centroid = pts.mean(axis=0)
                # Reflect each point across the plane through the centroid with
                # this normal: p' = p - 2((p-c)·n) n.
                d = (pts - local_centroid) @ axis            # signed distance
                refl = pts - 2.0 * d[:, None] * axis[None, :]
                refl_vox = refl / sp_arr[None, :]            # back to voxel metric
                refl_idx = np.rint(refl_vox).astype(np.int64)
                shape = np.array(mask_crop.shape)
                inb = np.all((refl_idx >= 0) & (refl_idx < shape), axis=1)
                refl_vol = np.zeros_like(mask_crop, dtype=bool)
                ri = refl_idx[inb]
                if ri.shape[0] > 0:
                    refl_vol[ri[:, 0], ri[:, 1], ri[:, 2]] = True
                inter = float(np.count_nonzero(mask_crop & refl_vol))
                union = float(np.count_nonzero(mask_crop | refl_vol))
                sym_score[idx] = (inter / union) if union > 0 else np.nan

        # ----- Wave N apical/basal + radial symmetry ---------------
        if have_major:
            mdir = major_vals[idx]
            mnorm = float(np.sqrt(np.sum(mdir ** 2)))
            if mnorm > 1e-12:
                maj = mdir / mnorm                       # ZYX unit major axis
                # apical_basal_symmetry_score: reflect across the plane whose
                # NORMAL is the MAJOR axis (mirror the two ENDS — apical vs basal
                # halves). A barrel / symmetric spindle scores high; a cone /
                # vase (one end fatter) scores low.
                ab_sym[idx] = _reflection_iou(mask_crop, maj, sp)
                # radial_symmetry_score: how rotationally symmetric the cell is
                # ABOUT its major axis. Bin the object's voxels by azimuth around
                # the major axis (in the perpendicular plane) and measure how
                # uniform the per-sector voxel counts are: 1/(1+cv). A body of
                # revolution (cylinder, sphere, cone) -> ~1; a flat slab / wedge
                # off-axis -> lower.
                radial_sym[idx] = _radial_symmetry_about_axis(mask_crop, maj, sp)

    df["mean_curvature_mean"] = curv_mean
    df["mean_curvature_std"] = curv_std
    df["abs_curvature_mean"] = abs_curv
    df["convexity"] = convexity
    df["concavity_count"] = conc_count
    df["max_concavity_depth"] = conc_depth
    df["bilateral_symmetry_score"] = sym_score
    # Wave B curvature remainder.
    df["gaussian_curvature_mean"] = gauss_mean
    df["gaussian_curvature_std"] = gauss_std
    df["bending_energy"] = bend_energy
    df["mean_curvature_skew"] = curv_skew
    df["neck_count"] = neck_cnt
    # Wave I curvature remainder.
    df["curvature_negative_fraction"] = curv_neg_frac
    df["gaussian_curvature_skew"] = gauss_skew
    # Wave N curvature/symmetry remainder.
    df["radial_symmetry_score"] = radial_sym
    df["apical_basal_symmetry_score"] = ab_sym
    return df


# Tilt angle (deg) beyond which the image-Z cross-section / apical-basal features
# are treated as fully unreliable — at this tilt the image-Z slices cut the cell
# obliquely. 0° = upright (fully reliable), 90° = the long axis lies in-plane.
_TILT_RELIABILITY_MAX_ANGLE = 45.0


def compute_reliability_meta(df: pd.DataFrame) -> pd.DataFrame:
    """Reliability / confidence META-features from EXISTING columns (Wave B).

    These do NOT describe morphology — they estimate whether OTHER features are
    trustworthy on a given cell, so the EDA layer can down-weight or filter
    unreliable rows. A pure TABLE pass (no volume, no mesh): it reads columns
    :func:`extract_features` / :func:`compute_border_cells` already emitted and
    derives four trust scores. ``role="qc"`` for all four.

    Features added (4, all dimensionless):
        ``tilt_reliability`` — how trustworthy the image-Z features (apical/basal,
            cross-section profile, z_elongation) are given the cell's tilt. From
            ``major_axis_z_angle`` (degrees, 0 = upright): ``1 −
            angle/_TILT_RELIABILITY_MAX_ANGLE`` clamped to [0, 1] (1 = upright →
            image-Z slices are true cross-sections; 0 at/over the cutoff tilt →
            the slices cut the cell obliquely so the Z features are unreliable).
        ``cross_section_reliability`` — whether the Z cross-section *profile*
            stats should be trusted. The same tilt basis as ``tilt_reliability``
            ADDITIONALLY down-weighted by a short Z-span (a cell only a few slices
            tall has a poorly-resolved A(z) profile regardless of tilt): the tilt
            score × a ``z_span`` ramp that reaches 1 at ``z_span >= 3``. Falls back
            to the tilt score alone when ``z_span`` is absent.
        ``border_reliability`` — 0 when the cell touches the volume face
            (``is_border_cell`` truthy → likely truncated → measurements
            unreliable), 1 otherwise. 1 when ``is_border_cell`` is absent (no
            border info → assume interior, the legacy-workspace default).
        ``feature_confidence_score`` — a combined [0, 1] trust score = the MEAN of
            the available component reliabilities (tilt, cross-section, border, and
            — when present — ``prob_mean`` as a segmentation-support term). A
            single "trust this cell's measurements?" indicator; degrades
            gracefully (averages only the components that exist).
        ``neighbor_topology_reliability`` (Mop-up O) — whether the
            Z-stratified neighbour topology (basal/apical counts, turnover) is
            trustworthy: a ``z_span`` ramp (reaches 1 at z_span >= 3, so a cell too
            short to HAVE a basal vs apical end is down-weighted) × a contact-support
            ramp (>= ~3 neighbours = 1, when ``neighbor_count`` is present). Falls
            back to the z_span ramp alone on a labels-only table. [0, 1].
        ``probability_support_score`` (Mop-up O) — whether the segmentation is
            well supported by the confidence map: the mean of the available
            prob-support terms (``prob_mean``, ``prob_boundary_mean``, and a
            low-boundary-entropy crispness term, each in [0, 1]). NaN when no prob
            columns exist (a labels-only table has no segmentation confidence).

    Degrades gracefully: any missing input column drops that term (its
    reliability defaults to the neutral/optimistic value) rather than raising —
    so a labels-only table still gets sensible reliabilities, and a richer table
    gets a better-supported ``feature_confidence_score``.

    Args:
        df: a per-cell DataFrame (modified in-place). Reads, when present:
            ``major_axis_z_angle``, ``z_span``, ``is_border_cell``, ``prob_mean``.

    Returns:
        The same DataFrame with the 4 new columns.
    """
    n = len(df)
    cols = (
        "tilt_reliability", "cross_section_reliability",
        "border_reliability", "feature_confidence_score",
        # Mop-up O remainder.
        "neighbor_topology_reliability", "probability_support_score",
    )
    if n == 0:
        for col in cols:
            df[col] = pd.Series(dtype=np.float64)
        return df

    # --- tilt_reliability: from major_axis_z_angle (deg) ---------------------
    if "major_axis_z_angle" in df.columns:
        angle = df["major_axis_z_angle"].values.astype(np.float64)
        with np.errstate(invalid="ignore"):
            tilt = 1.0 - np.abs(angle) / _TILT_RELIABILITY_MAX_ANGLE
        tilt = np.clip(tilt, 0.0, 1.0)
        # A NaN angle -> neutral (unknown tilt): assume reliable, don't poison.
        tilt = np.where(np.isfinite(tilt), tilt, 1.0)
    else:
        tilt = np.ones(n, dtype=np.float64)

    # --- cross_section_reliability: tilt down-weighted by short Z-span -------
    if "z_span" in df.columns:
        zspan = df["z_span"].values.astype(np.float64)
        # Ramp to 1 by z_span >= 3 slices; a 1-slice cell has no real profile.
        zramp = np.clip((zspan - 1.0) / 2.0, 0.0, 1.0)
        zramp = np.where(np.isfinite(zramp), zramp, 1.0)
        cross_rel = tilt * zramp
    else:
        cross_rel = tilt.copy()

    # --- border_reliability: 0 if the cell touches a volume face -------------
    if "is_border_cell" in df.columns:
        is_border = df["is_border_cell"].values
        border_rel = np.where(
            np.asarray(is_border, dtype=bool), 0.0, 1.0,
        ).astype(np.float64)
    else:
        border_rel = np.ones(n, dtype=np.float64)

    # --- feature_confidence_score: mean of the AVAILABLE component scores ----
    # Always include tilt + cross-section + border (they degrade to optimistic
    # defaults above); include prob_mean as a segmentation-support term only when
    # present. NaN-safe element-wise mean over the stacked component columns.
    components = [tilt, cross_rel, border_rel]
    if "prob_mean" in df.columns:
        prob = df["prob_mean"].values.astype(np.float64)
        prob = np.clip(prob, 0.0, 1.0)
        prob = np.where(np.isfinite(prob), prob, np.nan)
        components.append(prob)
    stacked = np.vstack(components)                     # (k, n)
    with np.errstate(invalid="ignore"):
        conf = np.nanmean(stacked, axis=0)
    # An all-NaN column (shouldn't happen — tilt/border are always finite) -> NaN.
    conf = np.where(np.isfinite(conf), conf, np.nan)

    # --- Mop-up O — neighbor_topology_reliability ----------------------
    # The Z-stratified neighbour topology (basal/apical counts, turnover) is only
    # trustworthy when the cell spans enough Z slices to HAVE a basal vs apical end
    # AND it has measurable contact support. the feature catalogue: "function of Z-span and
    # contact support". Build it as the product of a z_span ramp (reaches 1 at
    # z_span >= 3, like cross_section_reliability) and a contact-support ramp (a
    # cell with >= 1 neighbour and non-trivial contact is supported). Degrades to
    # the z_span ramp alone when no neighbour columns exist (labels-only table).
    if "z_span" in df.columns:
        zs = df["z_span"].values.astype(np.float64)
        ztopo = np.clip((zs - 1.0) / 2.0, 0.0, 1.0)
        ztopo = np.where(np.isfinite(ztopo), ztopo, 1.0)
    else:
        ztopo = np.ones(n, dtype=np.float64)
    if "neighbor_count" in df.columns:
        nc = df["neighbor_count"].values.astype(np.float64)
        # support saturates by ~3 neighbours (a well-embedded interior cell).
        supp = np.clip(nc / 3.0, 0.0, 1.0)
        supp = np.where(np.isfinite(supp), supp, 0.0)
        neigh_topo_rel = ztopo * supp
    else:
        neigh_topo_rel = ztopo

    # --- Mop-up O — probability_support_score --------------------------
    # the feature catalogue: "summary of prob_mean, prob_boundary_mean, and low entropy" — is
    # the segmentation well supported by the confidence map? Build it as the mean
    # of the AVAILABLE prob-support terms: prob_mean (high = confident interior),
    # prob_boundary_mean (high = a confident boundary), and (1 − normalized boundary
    # entropy) (low entropy = a crisp decision). Each clipped to [0,1]; NaN when no
    # prob columns exist (a labels-only table has no segmentation confidence).
    prob_terms = []
    for col in ("prob_mean", "prob_boundary_mean"):
        if col in df.columns:
            v = np.clip(df[col].values.astype(np.float64), 0.0, 1.0)
            prob_terms.append(np.where(np.isfinite(v), v, np.nan))
    if "prob_boundary_entropy" in df.columns:
        ent = df["prob_boundary_entropy"].values.astype(np.float64)
        # binary entropy is in nats, max ln2; map low entropy -> high support.
        crispness = np.clip(1.0 - ent / float(np.log(2.0)), 0.0, 1.0)
        prob_terms.append(np.where(np.isfinite(crispness), crispness, np.nan))
    if prob_terms:
        with np.errstate(invalid="ignore"):
            prob_support = np.nanmean(np.vstack(prob_terms), axis=0)
        prob_support = np.where(np.isfinite(prob_support), prob_support, np.nan)
    else:
        prob_support = np.full(n, np.nan, dtype=np.float64)

    df["tilt_reliability"] = tilt
    df["cross_section_reliability"] = cross_rel
    df["border_reliability"] = border_rel
    df["feature_confidence_score"] = conf
    df["neighbor_topology_reliability"] = neigh_topo_rel
    df["probability_support_score"] = prob_support
    return df


# ---------------------------------------------------------------------------
# M2 — Intensity / probability sampler features (B1, B2, B3).
#
# Ported VERBATIM (parity guardrail — same names, same math) from the EpiCycle
# monorepo ``napari_epicycle/fusion_reconstruction/qc_features.py``:
#   * compute_boundary_interior_features  (+ shared _iter_label_boundary_interior)
#   * compute_channel_intensity
#   * compute_probability_features        (+ _expand_slice, _binary_entropy)
#
# These are crop-local per-object readers: given a single-label mask crop and an
# intensity/probability crop at the SAME bbox, they emit boundary/interior + per-
# channel intensity + probability QC columns. The FeatureStage / reconcile worker
# samples the raw channel crop (via reconcile.raw_roi.read_global_roi) and the
# per-tile probability crop (via read_prob_roi, apply_sigmoid=True) at the cell's
# bbox and calls these.
#
# ⚠ ENGINE PROB CAVEAT: the engine's probability is a PER-TILE Cellpose logit
# (``<tile>_probabilities.npy``, float16), NOT the monorepo's multi-view GLOBAL
# CONSENSUS map [0,1]. Apply sigmoid (read_prob_roi(..., apply_sigmoid=True))
# before passing to compute_probability_features; the resulting prob_* columns
# describe the per-tile foreground confidence, not a cross-view consensus.
# ---------------------------------------------------------------------------


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
# Channel TEXTURE (Wave E) — the feature catalogue
# ---------------------------------------------------------------------------

# Per-channel texture suffixes — emitted by compute_channel_texture as
# ``<prefix>{suffix}``. Split into the robust 3D-native block + the
# 2D-per-slice GLCM/Haralick block. NOTE: ``_texture_cv`` is DELIBERATELY
# omitted — it would duplicate the existing ``<prefix>_intensity_cv`` from
# compute_channel_intensity (the task asked to skip it if already present).
_CHANNEL_TEXTURE_3D_SUFFIXES = (
    "_texture_variance", "_texture_entropy", "_texture_mad",
    "_texture_granularity", "_texture_range_robust",
)
_CHANNEL_TEXTURE_GLCM_SUFFIXES = (
    "_glcm_contrast", "_glcm_homogeneity", "_glcm_correlation",
    "_glcm_energy", "_glcm_entropy", "_glcm_dissimilarity",
)
_CHANNEL_TEXTURE_SUFFIXES = (
    _CHANNEL_TEXTURE_3D_SUFFIXES + _CHANNEL_TEXTURE_GLCM_SUFFIXES
)

# Texture config (module-level so the values are documented + testable).
_TEXTURE_HIST_BINS = 32          # in-mask intensity histogram bins (entropy)
_TEXTURE_GRANULARITY_RADII = (1, 2, 4)   # multiscale opening radii (voxels)
_GLCM_LEVELS = 16                # quantization levels for the GLCM
_GLCM_MIN_MASK_VOXELS = 27       # min in-mask voxels to attempt any GLCM
_GLCM_MIN_SLICE_VOXELS = 9       # min in-mask voxels on a Z-slice to use it


def compute_channel_texture(
    channel_image: np.ndarray,
    single_mask: np.ndarray,
    prefix: str,
    spacing: Optional[Tuple[float, ...]] = None,
) -> dict:
    """Per-channel TEXTURE features for ONE cell (Wave E).

    A SIBLING of :func:`compute_channel_intensity`, called on the SAME raw-channel
    crop + single-label mask the M2 intensity features already read. Texture is the
    robust-to-haze/saturation signal that separates **punctate vs diffuse vs hazy**
    expression (HCR), and captures **nuclear chromatin condensation** (cell-cycle
    state). Unlike the summary-stat intensity path, these describe the *spatial
    organization* of the signal inside the cell.

    Unlike the table-mutating ``compute_channel_intensity`` (which takes the full
    ``labels_3d`` + ``df`` and indexes ``find_objects``), this works directly on a
    pre-isolated ``(channel_image, single_mask)`` crop and RETURNS a flat
    ``{column: value}`` dict — that's the natural shape for the per-cell
    :func:`epicycle.core.feature_stage._sample_m2_features` path, where the cell is
    already a single-label crop. (The caller writes the dict into the row.)

    Columns emitted (11, all ``<prefix>``-prefixed):

    **3D-native (robust, no GLCM) — 5:**
        ``<prefix>_texture_variance`` — variance of the in-mask intensities (overall
            spatial unevenness; ~0 for a uniform fill).
        ``<prefix>_texture_entropy`` — Shannon entropy (nats) of the in-mask
            intensity histogram (``_TEXTURE_HIST_BINS`` bins on the robust p2..p98
            range). High = many distinct intensity levels (patchy / multi-level);
            ~0 = single level (uniform). Saturation-robust via the robust range.
        ``<prefix>_texture_mad`` — median absolute deviation of the in-mask
            intensities about their median (a robust spread, outlier-insensitive).
        ``<prefix>_texture_granularity`` — multiscale morphological-opening residual
            (à la CellProfiler "Granularity"): the fraction of total in-mask signal
            removed by greyscale openings at increasing radii
            (``_TEXTURE_GRANULARITY_RADII``). HIGH for a punctate signal (small
            bright spots are erased by the opening), LOW for a smooth/diffuse one.
            The single strongest punctate-vs-diffuse discriminator here. NOTE: the
            opening uses an ISOTROPIC (voxel-index, ball) structuring element, so the
            granularity SCALE is in voxels — directly COMPARABLE across cells only on
            NEAR-ISOTROPIC data (sp_z ≈ sp_y ≈ sp_x); on strongly anisotropic voxels
            the same physical feature reads at a different scale along Z.
        ``<prefix>_texture_range_robust`` — p98 − p2 of the in-mask intensities (a
            saturation/clip-robust dynamic range).

    **GLCM / Haralick (2D-per-slice, averaged over Z) — 6:**
        ``<prefix>_glcm_contrast``, ``_glcm_homogeneity``, ``_glcm_correlation``,
        ``_glcm_energy``, ``_glcm_entropy``, ``_glcm_dissimilarity``.

        The in-mask intensities are quantized to ``_GLCM_LEVELS`` levels over the
        cell's own min..max; on each Z-slice that has >= ``_GLCM_MIN_SLICE_VOXELS``
        in-mask voxels a 2D :func:`skimage.feature.graycomatrix` is built
        (``distances=[1]``, 4 in-plane angles 0/45/90/135°, ``symmetric``,
        ``normed``), the :func:`skimage.feature.graycoprops` (+ a manual GLCM
        entropy) are read, and the result is AVERAGED over slices × angles.

        NOTE: skimage has NO 3D GLCM — this is a 2D-per-slice GLCM averaged over the
        Z slices that intersect the mask. The Z anisotropy is therefore handled by
        the per-slice decomposition (each slice is an isotropic in-plane texture);
        ``spacing`` is accepted for signature uniformity + future weighting but the
        per-slice GLCM itself is index-based (a 1-voxel co-occurrence step in-plane).

    NaN guard: a cell with too few in-mask voxels (``< _GLCM_MIN_MASK_VOXELS``) or no
    qualifying Z-slice gets NaN for the GLCM block (the 3D-native block is still
    defined down to a 1-voxel mask: variance/entropy/mad/granularity/range collapse
    to 0 for a single value rather than NaN). An all-NaN crop (a failed tile read)
    yields NaN for every texture column — schema-stable.

    Args:
        channel_image: 3D intensity crop (Z, Y, X) for this channel, aligned
            voxel-for-voxel with ``single_mask``.
        single_mask: 3D boolean (or 0/1) crop — the isolated cell.
        prefix: column-name prefix (e.g. ``"nuc"``, ``"mem"``), matching the
            intensity prefix for the same channel.
        spacing: voxel spacing ``(Z, Y, X)``; accepted for signature uniformity
            (the GLCM is index-based + the native stats are intensity-only).

    Returns:
        ``{f"{prefix}{suffix}": float}`` for the 11 texture columns.
    """
    suffixes = _CHANNEL_TEXTURE_SUFFIXES
    out = {f"{prefix}{s}": np.nan for s in suffixes}

    mask = np.asarray(single_mask).astype(bool)
    ch = np.asarray(channel_image, dtype=np.float64)
    if mask.shape != ch.shape or not mask.any():
        return out

    vals = ch[mask]
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        # All-NaN crop (failed read): leave every column NaN — schema stable.
        return out

    # ----- 3D-native robust block -----------------------------------------
    out[f"{prefix}_texture_variance"] = float(np.var(finite))
    med = float(np.median(finite))
    out[f"{prefix}_texture_mad"] = float(np.median(np.abs(finite - med)))

    p2, p98 = np.percentile(finite, [2, 98])
    out[f"{prefix}_texture_range_robust"] = float(p98 - p2)

    # Shannon entropy (nats) of the in-mask histogram over the ROBUST range, so a
    # few saturated/clipped voxels don't dominate the bin WIDTHS. The values are
    # CLIPPED into [p2, p98] before binning (rather than dropped), so sparse-but-
    # bright puncta still populate the top bin and lift the entropy — a punctate
    # signal reads HIGH while a few clipped voxels don't skew the bin edges.
    if p98 > p2:
        clipped = np.clip(finite, p2, p98)
        hist, _ = np.histogram(clipped, bins=_TEXTURE_HIST_BINS, range=(p2, p98))
        total = hist.sum()
        if total > 0:
            p = hist.astype(np.float64) / total
            nz = p[p > 0]
            out[f"{prefix}_texture_entropy"] = float(-(nz * np.log(nz)).sum())
        else:
            out[f"{prefix}_texture_entropy"] = 0.0
    else:
        # Single intensity level over the robust range -> zero entropy.
        out[f"{prefix}_texture_entropy"] = 0.0

    out[f"{prefix}_texture_granularity"] = _texture_granularity(ch, mask)

    # ----- GLCM / Haralick block (2D-per-slice, averaged over Z) -----------
    glcm = _glcm_haralick_per_slice(ch, mask)
    if glcm is not None:
        for key, val in glcm.items():
            out[f"{prefix}{key}"] = val

    return out


def _texture_granularity(ch: np.ndarray, mask: np.ndarray) -> float:
    """Multiscale morphological-opening granularity (à la CellProfiler).

    Fraction of the total in-mask signal removed by greyscale openings at the
    increasing radii in ``_TEXTURE_GRANULARITY_RADII``. A punctate signal (small
    bright spots) is largely erased by the openings -> high fraction; a smooth /
    diffuse signal survives -> low fraction. Background (outside the mask) is set
    to 0 BEFORE the opening so spots near the cell edge aren't propped up by
    neighbouring signal, then the residual is measured inside the mask only.
    """
    from scipy.ndimage import grey_opening

    img = np.where(mask, ch, 0.0).astype(np.float64)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    base_sum = float(img[mask].sum())
    if base_sum <= 0:
        return 0.0

    removed = 0.0
    for r in _TEXTURE_GRANULARITY_RADII:
        size = 2 * r + 1
        opened = grey_opening(img, size=(size, size, size))
        residual = float((img[mask] - opened[mask]).clip(min=0).sum())
        removed += residual
    # Average residual fraction across the scales (bounded ~[0, 1]).
    return float(removed / (base_sum * len(_TEXTURE_GRANULARITY_RADII)))


def _glcm_haralick_per_slice(ch: np.ndarray, mask: np.ndarray) -> Optional[dict]:
    """2D-per-slice GLCM / Haralick props averaged over the Z slices intersecting *mask*.

    Returns ``{"_glcm_contrast": ..., ...}`` (keys WITHOUT the channel prefix) or
    ``None`` when the cell has too few in-mask voxels / no qualifying slice (the
    caller leaves those columns NaN). skimage has no 3D GLCM — anisotropy is handled
    by the per-slice decomposition; each slice's in-plane texture is isotropic.
    """
    from skimage.feature import graycomatrix, graycoprops

    finite_mask = mask & np.isfinite(ch)
    n_vox = int(finite_mask.sum())
    if n_vox < _GLCM_MIN_MASK_VOXELS:
        return None

    vals = ch[finite_mask]
    vmin = float(vals.min())
    vmax = float(vals.max())
    if vmax <= vmin:
        # Flat signal: GLCM is degenerate (single grey level). Report the
        # well-defined limits — perfectly homogeneous, no contrast — rather than
        # NaN, so a uniform-fill cell is distinguishable from a too-small one.
        return {
            "_glcm_contrast": 0.0, "_glcm_homogeneity": 1.0,
            "_glcm_correlation": 1.0, "_glcm_energy": 1.0,
            "_glcm_entropy": 0.0, "_glcm_dissimilarity": 0.0,
        }

    levels = _GLCM_LEVELS
    scale = (levels - 1) / (vmax - vmin)
    angles = [0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0]

    contrasts, homogeneities, correlations = [], [], []
    energies, entropies, dissimilarities = [], [], []

    for z in range(ch.shape[0]):
        slice_mask = finite_mask[z]
        if int(slice_mask.sum()) < _GLCM_MIN_SLICE_VOXELS:
            continue
        # Quantize the slice to [0, levels-1]; out-of-mask voxels -> 0 (they are
        # excluded from co-occurrence below via the bounded-box restriction is not
        # possible, so background sits at level 0 — acceptable: it adds a constant
        # low-level surround, and the per-slice contrast/homogeneity are dominated
        # by the in-mask structure which is what we want).
        q = np.zeros(ch.shape[1:], dtype=np.uint8)
        sv = ch[z][slice_mask]
        q[slice_mask] = np.clip(
            ((sv - vmin) * scale).round(), 0, levels - 1
        ).astype(np.uint8)

        glcm = graycomatrix(
            q, distances=[1], angles=angles, levels=levels,
            symmetric=True, normed=True,
        )
        contrasts.append(float(graycoprops(glcm, "contrast").mean()))
        homogeneities.append(float(graycoprops(glcm, "homogeneity").mean()))
        correlations.append(float(graycoprops(glcm, "correlation").mean()))
        energies.append(float(graycoprops(glcm, "energy").mean()))
        dissimilarities.append(float(graycoprops(glcm, "dissimilarity").mean()))
        # GLCM entropy (skimage has no graycoprop for it): -sum p log p over the
        # normalized co-occurrence matrix, averaged across angles.
        ent_per_angle = []
        for a in range(glcm.shape[3]):
            mat = glcm[:, :, 0, a]
            nz = mat[mat > 0]
            ent_per_angle.append(float(-(nz * np.log(nz)).sum()))
        entropies.append(float(np.mean(ent_per_angle)))

    if not contrasts:
        return None

    return {
        "_glcm_contrast": float(np.mean(contrasts)),
        "_glcm_homogeneity": float(np.mean(homogeneities)),
        "_glcm_correlation": float(np.mean(correlations)),
        "_glcm_energy": float(np.mean(energies)),
        "_glcm_entropy": float(np.mean(entropies)),
        "_glcm_dissimilarity": float(np.mean(dissimilarities)),
    }


# ---------------------------------------------------------------------------
# Probability features
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


def _region_intensity_entropy(values: np.ndarray) -> float:
    """Shannon entropy (nats) of an intensity sample over a robust p2..p98 range.

    The SAME saturation-robust histogram-entropy used by
    :func:`compute_channel_texture` (``_TEXTURE_HIST_BINS`` bins, values clipped
    into the robust range before binning), factored out so the per-COMPARTMENT
    texture entropy (Mop-up O) and the whole-cell texture entropy share one
    definition. ``values`` = finite in-region intensities. Returns 0.0 for a
    single-level region, NaN for an empty one.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    p2, p98 = np.percentile(v, [2, 98])
    if p98 <= p2:
        return 0.0
    clipped = np.clip(v, p2, p98)
    hist, _ = np.histogram(clipped, bins=_TEXTURE_HIST_BINS, range=(p2, p98))
    total = hist.sum()
    if total <= 0:
        return 0.0
    pr = hist.astype(np.float64) / total
    nz = pr[pr > 0]
    return float(-(nz * np.log(nz)).sum())


def compute_probability_features(
    labels_3d: np.ndarray,
    df: pd.DataFrame,
    prob_3d: np.ndarray,
    bbox_padding: int = 3,
) -> pd.DataFrame:
    """Compute per-label statistics from the 3D probability volume.

    NB (engine divergence): in the monorepo ``prob_3d`` is the multi-view CONSENSUS
    map [0,1] produced by u-Segment3D during reconstruction. In the SCALED-SEG
    engine the probability is a PER-TILE Cellpose LOGIT (``<tile>_probabilities.npy``,
    float16) — NOT a global consensus. Pass it through a sigmoid first
    (``reconcile.raw_roi.read_prob_roi(..., apply_sigmoid=True)``) so the values are
    in [0,1]; the resulting prob_* columns then describe per-tile foreground
    confidence rather than a cross-view consensus. The math below is byte-identical
    to the monorepo regardless — only the meaning of the input differs.

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


# ---------------------------------------------------------------------------
# Wave F — probability-geometry interaction +
#          channel / expression localization
# ---------------------------------------------------------------------------
#
# Both families combine the raw/prob SIGNAL (the M2 sampler crops) with GEOMETRY
# (the M5 interior-EDT radial shells + the Wave C outward normal). They run on a
# SINGLE pre-isolated crop + mask (the per-cell sampler shape), so — like
# compute_channel_texture — they take ``(signal_crop, single_mask)`` and RETURN a
# flat ``{column: value}`` dict (the caller writes it into the single-row df).
# The nuclear/cytoplasm compartment part needs the NUCLEUS mask too, so it lives
# in the paired cell-nucleus measurements, not here.


# --- Wave F column templates (single-volume) -------------------------------
# prob-geometry interaction (un-prefixed `prob_*`).
_PROB_GEOMETRY_COLS = (
    "prob_area_normalized",
    "prob_shell_inner_mean", "prob_shell_outer_mean", "prob_shell_gradient",
    "prob_surface_uncertainty",
    # Mop-up O remainder — surface-area-normalized confidence + the
    # boundary-drop-per-area (the geometry-conditioned boundary confidence).
    "prob_surface_normalized", "prob_boundary_drop_per_area",
    # P1a SIGNED remainder — the apical-vs-basal confidence difference, signed
    # by the Wave C outward normal (NaN unless a trusted outward normal + centroid
    # are supplied — the default-off contract, like the channel apical/basal means).
    "prob_apical_basal_gradient",
)

# channel localization SHELL + APICAL/BASAL suffixes (per `<prefix>`).
# Shell + radial slope are single-volume (no nucleus / no outward normal);
# apical/basal need the Wave C outward normal on the row (else NaN).
_CHANNEL_LOCALIZATION_SHELL_SUFFIXES = (
    "_shell_inner_mean", "_shell_outer_mean", "_shell_gradient",
    "_radial_intensity_slope",
    # Mop-up O remainder — single-volume SHELL/radial expansion:
    #   3-band radial-intensity profile (inner core / mid / outer rind) +
    #   membrane (outer-rind) enrichment over the whole-cell mean + the explicit
    #   outer/inner ratio + the [0,1] fraction-above-threshold positivity.
    "_shell_mid_mean", "_radial_bin_inner_mean", "_radial_bin_mid_mean",
    "_radial_bin_outer_mean",
    "_membrane_enrichment", "_outer_to_inner_ratio", "_fraction_above_threshold",
)
_CHANNEL_LOCALIZATION_APICALBASAL_SUFFIXES = (
    "_apical_intensity_mean", "_basal_intensity_mean",
    "_apical_basal_intensity_ratio",
    # Mop-up O remainder — the LATERAL (equatorial side-band) intensity, the
    # third member of the apical/basal/lateral split (needs the Wave C outward
    # normal, like the apical/basal means).
    "_lateral_intensity_mean",
)
_CHANNEL_LOCALIZATION_SUFFIXES = (
    _CHANNEL_LOCALIZATION_SHELL_SUFFIXES
    + _CHANNEL_LOCALIZATION_APICALBASAL_SUFFIXES
)

# nuclear/cytoplasm compartment suffixes (per `<prefix>`; PAIRED path).
_CHANNEL_COMPARTMENT_SUFFIXES = (
    "_nuclear_mean", "_cytoplasm_mean", "_nuclear_cytoplasm_ratio",
    "_perinuclear_enrichment",
    # Mop-up O compartment remainder (PAIRED path):
    #   the nuclear SIGNAL fraction (`nuclear_fraction_<channel>`) + the
    #   compartment-restricted TEXTURE entropy on each compartment
    #   (nuclear/cytoplasmic_<channel>_texture_entropy) + an expected-compartment
    #   mismatch score (`marker_compartment_mismatch_score`).
    "_nuclear_signal_fraction",
    "_nuclear_texture_entropy", "_cytoplasmic_texture_entropy",
    "_compartment_mismatch_score",
)

# Min in-mask voxels for the radial slope linear fit to be defined (needs >= 2
# distinct radial positions; a tiny / 1-voxel-thick crop -> NaN slope).
_RADIAL_SLOPE_MIN_VOXELS = 8


def _edt_radial_position(
    single_mask: np.ndarray,
    spacing: Optional[Tuple[float, ...]] = None,
):
    """Interior-EDT radial position ``r = d / max(d) ∈ [0,1]`` over the mask voxels.

    Reuses the M5 / :func:`_object_edt` interior distance field (the SAME field
    :func:`compute_shell_features` partitions): every in-mask voxel's distance to
    the nearest background voxel, normalized by the object's max interior distance,
    so ``r → 1`` at the deepest core point and ``r → 0`` on the surface rind.

    Returns ``(r_field, mask_bool)`` where ``r_field`` is the float radial position
    over the WHOLE crop (0 outside the mask) and ``mask_bool`` the boolean mask, or
    ``(None, None)`` when the mask is empty or collapses to a single radial level
    (a 1-voxel-thick sheet whose ``max(d) == 0`` — no inner/outer split possible).
    """
    m = np.asarray(single_mask).astype(bool)
    if not m.any():
        return None, None
    dt = _object_edt(m, spacing=spacing)
    d_max = float(dt[m].max())
    if d_max <= 1e-12:
        return None, None
    r = np.zeros(m.shape, dtype=np.float64)
    r[m] = dt[m] / d_max
    return r, m


def compute_prob_geometry(
    single_mask: np.ndarray,
    prob_crop: np.ndarray,
    spacing: Optional[Tuple[float, ...]] = None,
    prob_weighted_volume: Optional[float] = None,
    surface_area: Optional[float] = None,
    prob_boundary_drop: Optional[float] = None,
    outward: Optional[Tuple[float, float, float]] = None,
    centroid: Optional[Tuple[float, float, float]] = None,
) -> dict:
    """Probability-geometry interaction features for ONE cell (Wave F).

    Combines the per-tile probability crop (the M2 sampler signal) with the cell's
    interior-EDT radial GEOMETRY (the M5 shell partition) to distinguish "weird but
    CONFIDENT" from "weird because UNCERTAIN". A SIBLING of
    :func:`compute_probability_features`, called on the SAME ``(prob_crop,
    single_mask)`` the M2 prob features read; RETURNS a flat ``{column: value}``
    dict (the per-cell sampler shape, like :func:`compute_channel_texture`).

    The prob crop is expected to already be in [0, 1] (the engine prob is a per-tile
    LOGIT — pass it through ``reconcile.raw_roi.read_prob_roi(..., apply_sigmoid=
    True)`` first, exactly as :func:`compute_probability_features` documents).

    Columns emitted (5, un-prefixed ``prob_*``):
        ``prob_area_normalized`` — ``prob_weighted_volume / geometric_volume`` where
            geometric_volume is the in-mask voxel count (physical when ``spacing``
            given). HOW MUCH of the mask is high-confidence: ~1 = a confidently
            filled mask, < 1 = the mask is larger than the confident core (soft /
            over-segmented), > 1 = confidence spills past the hard mask. Unbounded
            (>= 0). When ``prob_weighted_volume`` is not supplied it is the sum of
            the in-mask prob (a self-contained estimate).
        ``prob_shell_inner_mean`` / ``prob_shell_outer_mean`` — mean prob in the
            INNER (core) vs OUTER (surface rind) interior-EDT shells (the SAME
            equal-volume tertile partition as :func:`compute_shell_features`, reused
            via :data:`_SHELL_R1` / :data:`_SHELL_R2`). Both [0, 1].
        ``prob_shell_gradient`` — ``outer_mean − inner_mean`` (signed, [-1, 1]):
            NEGATIVE = a low-confidence outer rind around a confident core (the
            well-segmented case), POSITIVE = a confident rind around an uncertain
            core (an unusual / possibly hollow or merged mask).
        ``prob_surface_uncertainty`` — mean voxel-wise binary entropy
            (:func:`_binary_entropy`, in nats, [0, ln 2]) over the OUTER (boundary)
            shell: high = the boundary confidence sits near 0.5 (an ambiguous edge),
            ~0 = a crisp confident-or-not boundary.
        ``prob_surface_normalized`` (Mop-up O) — ``prob_weighted_volume /
            surface_area``: the soft (confidence-weighted) volume per unit cell
            surface. A thickness-like confidence measure that, unlike
            ``prob_area_normalized`` (per VOLUME), is conditioned on the cell's
            SURFACE. Needs the ``surface_area`` of this cell (passed in); NaN
            otherwise. Unbounded >= 0.
        ``prob_boundary_drop_per_area`` (Mop-up O) — ``prob_boundary_drop /
            surface_area``: how much the confidence falls off at the boundary,
            NORMALIZED by the cell's surface area (a small absolute drop on a large
            surface is less alarming than the same drop on a small one). Needs the
            cell's ``prob_boundary_drop`` + ``surface_area`` (both passed in); NaN
            otherwise. Signed (the drop can be negative), unbounded.
        ``prob_apical_basal_gradient`` (P1a, SIGNED) — the apical-vs-basal
            confidence difference: ``mean(prob over the APICAL third) − mean(prob over
            the BASAL third)`` ∈ [-1, 1], split by the SIGNED projection ``s = (voxel −
            centroid)·outward`` (apical = the top tertile of ``s``, toward the surface;
            basal = the bottom tertile; the equatorial band is EXCLUDED so the contrast
            is concentrated where apical/basal is most distinct + robust to small
            errors in the normal). +ve = the model is MORE confident apically, −ve =
            more confident basally (a basal-side or apical-truncation ambiguity). The
            SIGN is anchored to the SIGNED outward normal (its only signed input), so it
            is robust. Needs a TRUSTED ``outward`` normal + a ``centroid`` (passed in);
            NaN otherwise (the default-off contract — the per-cell pool runs BEFORE the
            global outward-normal pass, exactly like the channel apical/basal means).
            NaN'd (not 0) when either tertile is below a small voxel floor / the normal
            is ~in-plane. NB: despite the "gradient" catalogue name this is a normalized
            mean DIFFERENCE in [-1,1], not a per-length spatial derivative.

    NaN guard: an empty mask, a shape mismatch, an all-NaN prob crop (failed read),
    or a degenerate single-radial-level mask (a 1-voxel-thick sheet) -> every column
    NaN (schema-stable). A mask with no inner shell voxels (a thin object whose EDT
    has no core) NaNs the two shell means + the gradient but still defines
    ``prob_area_normalized`` + ``prob_surface_uncertainty``.

    Args:
        single_mask: 3D boolean (or 0/1) crop — the isolated cell.
        prob_crop: 3D probability crop (Z, Y, X) in [0, 1], aligned voxel-for-voxel
            with ``single_mask``.
        spacing: voxel spacing ``(Z, Y, X)`` for the anisotropy-aware EDT + the
            physical geometric volume. None = voxel units.
        prob_weighted_volume: OPTIONAL precomputed soft volume (the
            ``prob_weighted_volume`` :func:`compute_probability_features` already
            emitted for this cell) — reused for ``prob_area_normalized`` so the two
            families agree. When None it is the in-mask prob sum.
        surface_area: OPTIONAL cell surface area (the marching-cubes surface family)
            for the Mop-up O surface-conditioned columns; NaN them when absent.
        prob_boundary_drop: OPTIONAL cell ``prob_boundary_drop`` for the
            per-area boundary-drop; NaN it when absent.
        outward: OPTIONAL per-cell OUTWARD unit normal ``(z, y, x)`` (the Wave C
            ``tissue_outward_*`` columns) — the SIGN basis for
            ``prob_apical_basal_gradient``. When None that column is NaN (default-off).
        centroid: OPTIONAL cell centroid ``(z, y, x)`` in the CROP's own coordinate
            frame, PHYSICAL when ``spacing`` is given (so the apical/basal split is in
            the same frame as the physical outward normal). When None it is computed
            from the mask. Only used with ``outward``.

    Returns:
        ``{column: float}`` for the prob-geometry columns (every key present,
        NaN where undefined).
    """
    out = {c: np.nan for c in _PROB_GEOMETRY_COLS}

    mask = np.asarray(single_mask).astype(bool)
    pr = np.asarray(prob_crop, dtype=np.float64)
    if mask.shape != pr.shape or not mask.any():
        return out

    in_vals = pr[mask]
    finite = in_vals[np.isfinite(in_vals)]
    if finite.size == 0:
        return out  # all-NaN prob crop (failed read) -> schema-stable NaN

    sp = tuple(spacing) if spacing is not None else None
    vox_vol = float(np.prod(sp)) if sp is not None else 1.0
    geom_vol = float(mask.sum()) * vox_vol

    # prob_area_normalized = soft volume / geometric volume.
    if prob_weighted_volume is not None and np.isfinite(prob_weighted_volume):
        soft_vol = float(prob_weighted_volume)
    else:
        soft_vol = float(finite.sum()) * vox_vol  # in-mask prob mass (physical)
    if geom_vol > 0:
        out["prob_area_normalized"] = soft_vol / geom_vol

    # --- inner / outer EDT shells (reuse the M5 equal-volume tertile partition) ---
    r, _ = _edt_radial_position(mask, spacing=sp)
    if r is not None:
        outer = mask & (r <= _SHELL_R1)        # surface rind (the boundary shell)
        inner = mask & (r > _SHELL_R2)         # core
        # mean prob per shell (NaN-safe over finite prob voxels only)
        for key, shell in (("prob_shell_inner_mean", inner),
                           ("prob_shell_outer_mean", outer)):
            if shell.any():
                sv = pr[shell]
                sv = sv[np.isfinite(sv)]
                if sv.size:
                    out[key] = float(sv.mean())
        if (np.isfinite(out["prob_shell_inner_mean"])
                and np.isfinite(out["prob_shell_outer_mean"])):
            out["prob_shell_gradient"] = (
                out["prob_shell_outer_mean"] - out["prob_shell_inner_mean"])
        # surface uncertainty = mean binary entropy over the OUTER (boundary) shell.
        if outer.any():
            ov = pr[outer]
            ov = ov[np.isfinite(ov)]
            if ov.size:
                with np.errstate(divide="ignore", invalid="ignore"):
                    out["prob_surface_uncertainty"] = float(
                        _binary_entropy(np.clip(ov, 0.0, 1.0)).mean())

    # --- Mop-up O surface-conditioned confidence ---------------------
    # prob_surface_normalized = soft (confidence-weighted) volume per unit surface.
    sa = (float(surface_area) if surface_area is not None
          and np.isfinite(surface_area) else np.nan)
    if np.isfinite(sa) and sa > 1e-12:
        # `soft_vol` is the prob_area_normalized numerator computed above (physical
        # when spacing set). Recompute it from the same source for clarity/safety.
        if prob_weighted_volume is not None and np.isfinite(prob_weighted_volume):
            sv = float(prob_weighted_volume)
        else:
            sv = float(finite.sum()) * vox_vol
        out["prob_surface_normalized"] = sv / sa
        # prob_boundary_drop_per_area = the cell's boundary confidence drop per unit
        # surface (the geometry-conditioned boundary-confidence).
        if prob_boundary_drop is not None and np.isfinite(prob_boundary_drop):
            out["prob_boundary_drop_per_area"] = float(prob_boundary_drop) / sa

    # --- P1a SIGNED apical/basal confidence difference ---------------
    # mean(prob | apical third) - mean(prob | basal third), split by the SIGNED
    # projection onto the outward normal (apical = +outward). A cone (tertile) split:
    # the equatorial band is excluded so the contrast is concentrated where
    # apical/basal is most distinct + robust to small normal errors. Needs a trusted
    # outward normal + a centroid; NaN otherwise (default-off, like the channel
    # apical/basal means). The sign is anchored to the SIGNED outward normal.
    if outward is not None:
        ow = np.asarray(outward, dtype=np.float64)
        if ow.shape == (3,) and np.all(np.isfinite(ow)):
            ownorm = float(np.linalg.norm(ow))
            if ownorm > 1e-9:
                ow = ow / ownorm
                zz, yy, xx = np.where(mask)
                coords = np.stack([zz, yy, xx], axis=1).astype(np.float64)
                if sp is not None:
                    coords *= np.asarray(sp, dtype=np.float64)
                if centroid is not None and np.all(np.isfinite(centroid)):
                    cen = np.asarray(centroid, dtype=np.float64)
                else:
                    cen = coords.mean(axis=0)
                s = (coords - cen) @ ow            # +ve apical, -ve basal
                pv = pr[mask]
                ok = np.isfinite(pv) & np.isfinite(s)
                s, pv = s[ok], pv[ok]
                # tertile (cone) split + a small per-band voxel floor: NaN (not 0) for
                # a near-in-plane normal (s ~ constant) or a too-thin band.
                if s.size >= 6 and float(s.std()) > 1e-9:
                    lo = float(np.percentile(s, 100.0 / 3.0))
                    hi = float(np.percentile(s, 200.0 / 3.0))
                    apical = pv[s >= hi]
                    basal = pv[s <= lo]
                    if apical.size >= 2 and basal.size >= 2:
                        out["prob_apical_basal_gradient"] = float(
                            apical.mean() - basal.mean())

    return out


def compute_channel_localization(
    channel_image: np.ndarray,
    single_mask: np.ndarray,
    prefix: str,
    spacing: Optional[Tuple[float, ...]] = None,
    outward: Optional[Tuple[float, float, float]] = None,
    centroid: Optional[Tuple[float, float, float]] = None,
) -> dict:
    """Per-channel SINGLE-VOLUME localization features for ONE cell (Wave F).

    The compartment/geometry-aware expansion of the M2 per-channel intensity
    summaries: WHERE the signal sits, not just how strong it is. A SIBLING of
    :func:`compute_channel_intensity` / :func:`compute_channel_texture`, called on
    the SAME raw-channel crop + single-label mask; RETURNS a flat ``{column:
    value}`` dict (the per-cell sampler shape).

    Two blocks, both ``<prefix>``-prefixed:

    **Compartmental via SHELLS (always; single-volume, no nucleus needed) — 4:**
        ``<prefix>_shell_inner_mean`` / ``<prefix>_shell_outer_mean`` — mean
            intensity in the INNER (core) vs OUTER (surface rind) interior-EDT shells
            (the SAME M5 equal-volume tertile partition, :data:`_SHELL_R1` /
            :data:`_SHELL_R2`). A MEMBRANE-localized channel piles signal in the
            outer rind (``shell_outer_mean`` ≫ ``shell_inner_mean``); a CORE / cytosol
            channel is the opposite.
        ``<prefix>_shell_gradient`` — ``outer_mean − inner_mean`` (signed):
            POSITIVE = membrane / cortex-localized, NEGATIVE = core-localized.
        ``<prefix>_radial_intensity_slope`` — the slope of a least-squares linear fit
            of in-mask intensity vs the normalized EDT radial position
            ``r = d/max(d) ∈ [0,1]`` (1 = core, 0 = surface). NEGATIVE slope =
            brighter at the surface (membrane-localized), POSITIVE = brighter in the
            core. A continuous companion to the discrete shell gradient.

    **Apical / basal (only if a trusted outward normal + centroid are given) — 3:**
        ``<prefix>_apical_intensity_mean`` / ``<prefix>_basal_intensity_mean`` —
            mean intensity in the OUTWARD (apical, toward the tissue surface) vs
            INWARD (basal, toward the interior) half of the cell, split by the plane
            through the centroid perpendicular to the Wave C ``outward`` normal:
            a voxel is apical when ``(voxel − centroid) · outward > 0``.
        ``<prefix>_apical_basal_intensity_ratio`` — ``apical_mean / basal_mean``
            (> 1 = apically enriched, < 1 = basally enriched). NaN when either half
            is empty or the basal mean is ~0.

    When ``outward`` / ``centroid`` are absent (the FeatureStage per-cell pool runs
    BEFORE the Wave C global outward-normal pass, so they are NOT available there)
    the apical/basal block stays NaN — the shell block is always defined.

    NaN guard: empty mask / shape mismatch / all-NaN crop -> every column NaN
    (schema-stable). A degenerate single-radial-level mask (1-voxel-thick sheet)
    NaNs the shell + slope block.

    Args:
        channel_image: 3D intensity crop (Z, Y, X), aligned with ``single_mask``.
        single_mask: 3D boolean (or 0/1) crop — the isolated cell.
        prefix: column-name prefix (matches the intensity / texture prefix for the
            same channel, e.g. ``"mem"``, ``"nuc"``).
        spacing: voxel spacing ``(Z, Y, X)`` for the anisotropy-aware EDT + the
            apical/basal projection geometry. None = voxel units.
        outward: OPTIONAL per-cell OUTWARD unit normal ``(z, y, x)`` (the Wave C
            ``tissue_outward_*`` columns). When None the apical/basal block is NaN.
        centroid: OPTIONAL cell centroid ``(z, y, x)`` in the CROP's own coordinate
            frame, PHYSICAL when ``spacing`` is given (so it matches the physical
            voxel coordinates the apical/basal split uses). When None it is computed
            from the mask. Only used with ``outward``.

    Mop-up O remainder (single-volume): the 3-band radial-intensity profile
    (``_radial_bin_inner/mid/outer_mean`` + the ``_shell_mid_mean``), the explicit
    ``_outer_to_inner_ratio``, the ``_membrane_enrichment`` (outer-rind over the
    whole-cell mean), the ``_fraction_above_threshold`` positivity, and — in the
    apical/basal block — the ``_lateral_intensity_mean`` (the equatorial side-band).

    Returns:
        ``{f"{prefix}{suffix}": float}`` for the 15 localization columns.
    """
    out = {f"{prefix}{s}": np.nan for s in _CHANNEL_LOCALIZATION_SUFFIXES}

    mask = np.asarray(single_mask).astype(bool)
    ch = np.asarray(channel_image, dtype=np.float64)
    if mask.shape != ch.shape or not mask.any():
        return out
    if not np.isfinite(ch[mask]).any():
        return out  # all-NaN crop -> schema-stable NaN

    sp = tuple(spacing) if spacing is not None else None

    # whole-cell mean over finite in-mask voxels (denominator for the enrichment
    # + the basis for the positivity threshold).
    all_iv = ch[mask]
    all_fin = all_iv[np.isfinite(all_iv)]
    cell_mean = float(all_fin.mean()) if all_fin.size else np.nan

    # --- SHELL block (always; single-volume) -------------------------------
    r, _ = _edt_radial_position(mask, spacing=sp)
    if r is not None:
        outer = mask & (r <= _SHELL_R1)        # surface rind
        mid = mask & (r > _SHELL_R1) & (r <= _SHELL_R2)   # middle band
        inner = mask & (r > _SHELL_R2)         # core
        inner_mean = np.nan
        outer_mean = np.nan
        mid_mean = np.nan
        for key, shell in (("inner", inner), ("outer", outer), ("mid", mid)):
            if shell.any():
                sv = ch[shell]
                sv = sv[np.isfinite(sv)]
                if sv.size:
                    val = float(sv.mean())
                    out[f"{prefix}_shell_{key}_mean"] = val
                    if key == "inner":
                        inner_mean = val
                    elif key == "outer":
                        outer_mean = val
                    else:
                        mid_mean = val
        if np.isfinite(inner_mean) and np.isfinite(outer_mean):
            out[f"{prefix}_shell_gradient"] = outer_mean - inner_mean
            # explicit outer/inner ratio (`outer_to_inner_<channel>_ratio`):
            # the membrane-vs-core localization as a ratio (> 1 = membrane-biased).
            if abs(inner_mean) > 1e-9:
                out[f"{prefix}_outer_to_inner_ratio"] = outer_mean / inner_mean

        # 3-band radial profile (inner/mid/outer expansion — the named
        # `radial intensity bins`). Same equal-volume tertile partition as the
        # shell means; carried as distinct `_radial_bin_*` columns so the named
        # profile triple is explicit (the `_shell_*_mean` columns are the
        # membrane-vs-core synonyms).
        out[f"{prefix}_radial_bin_inner_mean"] = inner_mean
        out[f"{prefix}_radial_bin_mid_mean"] = mid_mean
        out[f"{prefix}_radial_bin_outer_mean"] = outer_mean

        # membrane enrichment (`membrane_to_cytoplasmic_<channel>_ratio`
        # proxy): the outer-rind (membrane) mean over the whole-cell mean. > 1 =
        # the signal piles on the cell boundary. We use the whole-cell mean as the
        # cytoplasm proxy in the single-volume path (no nucleus mask here — the
        # true nuclear/cytoplasm split is the PAIRED compartment family).
        if np.isfinite(outer_mean) and np.isfinite(cell_mean) and abs(cell_mean) > 1e-9:
            out[f"{prefix}_membrane_enrichment"] = outer_mean / cell_mean

        # radial intensity slope: least-squares fit of intensity vs r over the
        # in-mask voxels (needs >= 2 distinct radial levels + enough voxels).
        rv = r[mask]
        iv = ch[mask]
        ok = np.isfinite(rv) & np.isfinite(iv)
        rv, iv = rv[ok], iv[ok]
        if rv.size >= _RADIAL_SLOPE_MIN_VOXELS and float(rv.std()) > 1e-9:
            # slope of intensity (y) on r (x) — polyfit deg 1, [slope, intercept].
            slope = float(np.polyfit(rv, iv, 1)[0])
            out[f"{prefix}_radial_intensity_slope"] = slope

    # fraction of the cell that is confidently positive for the marker
    # (`<channel>_fraction_above_threshold`): the share of in-mask
    # voxels ABOVE the per-cell mean (a robust, parameter-free positivity cut —
    # the SAME mean-threshold convention compute_channel_colocalization uses). ~0
    # for a few bright spots in a dark cell, ~0.5 for a balanced gradient. [0,1].
    if all_fin.size and np.isfinite(cell_mean):
        out[f"{prefix}_fraction_above_threshold"] = float(
            np.mean(all_fin > cell_mean))

    # --- APICAL / BASAL block (only with a trusted outward normal) ----------
    if outward is not None:
        ow = np.asarray(outward, dtype=np.float64)
        if ow.shape == (3,) and np.all(np.isfinite(ow)):
            nrm = float(np.linalg.norm(ow))
            if nrm > 1e-9:
                ow = ow / nrm
                # physical voxel coordinates (ZYX): index grid × spacing so the
                # dot with the (physical) outward normal is geometrically correct.
                zz, yy, xx = np.where(mask)
                coords = np.stack([zz, yy, xx], axis=1).astype(np.float64)
                if sp is not None:
                    coords *= np.asarray(sp, dtype=np.float64)
                if centroid is not None and np.all(np.isfinite(centroid)):
                    cen = np.asarray(centroid, dtype=np.float64)
                else:
                    cen = coords.mean(axis=0)
                rel = coords - cen
                proj = rel @ ow                    # + = apical (outward), - = basal
                iv = ch[mask]
                fin = np.isfinite(iv)
                apical = (proj > 0) & fin
                basal = (proj <= 0) & fin
                a_mean = float(iv[apical].mean()) if apical.any() else np.nan
                b_mean = float(iv[basal].mean()) if basal.any() else np.nan
                out[f"{prefix}_apical_intensity_mean"] = a_mean
                out[f"{prefix}_basal_intensity_mean"] = b_mean
                if (np.isfinite(a_mean) and np.isfinite(b_mean)
                        and abs(b_mean) > 1e-9):
                    out[f"{prefix}_apical_basal_intensity_ratio"] = a_mean / b_mean
                # LATERAL (equatorial side-band) mean (`lateral_<channel>_mean`):
                # the same apical/basal/lateral cone split compute_directional_contact
                # uses — a voxel is LATERAL when the |cosine| of its direction-from-
                # centroid against the outward normal is < 0.5 (the perpendicular side
                # band, neither apical nor basal). Mean intensity over those voxels.
                rnorm = np.linalg.norm(rel, axis=1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    cosang = np.where(rnorm > 1e-9, proj / rnorm, 0.0)
                lateral = (np.abs(cosang) < 0.5) & fin
                out[f"{prefix}_lateral_intensity_mean"] = (
                    float(iv[lateral].mean()) if lateral.any() else np.nan)

    return out


def compute_channel_compartments(
    channel_crop: np.ndarray,
    cell_mask: np.ndarray,
    nucleus_mask: np.ndarray,
    prefix: str,
    spacing: Optional[Tuple[float, ...]] = None,
    perinuclear_dilation: int = 2,
) -> dict:
    """Per-channel NUCLEAR vs CYTOPLASM localization for ONE cell (Wave F).

    The compartment-aware expansion that needs BOTH the cell mask AND its nucleus
    mask, so it runs in the reconcile PAIRED path
    (the paired cell-nucleus measurements) — given the
    raw-channel crop sampled at the cell bbox + the (in-cell) nucleus mask. RETURNS
    a flat ``{column: value}`` dict (one cell).

    Columns emitted (4, ``<prefix>``-prefixed):
        ``<prefix>_nuclear_mean`` — mean intensity over the IN-CELL nucleus voxels.
        ``<prefix>_cytoplasm_mean`` — mean intensity over the cell-minus-nucleus
            (cytoplasm) voxels.
        ``<prefix>_nuclear_cytoplasm_ratio`` — ``nuclear_mean / cytoplasm_mean``
            (> 1 = nuclear-enriched, < 1 = cytoplasm-enriched). NaN when the
            cytoplasm mean is ~0 / empty.
        ``<prefix>_perinuclear_enrichment`` — ``perinuclear_mean / cytoplasm_mean``
            where the PERINUCLEAR rim is the (in-cell) nucleus dilated by
            ``perinuclear_dilation`` voxels MINUS the nucleus itself (a shell hugging
            the nuclear envelope, clipped to the cytoplasm). > 1 = the signal hugs
            the nucleus (perinuclear / ER / Golgi-like), ~1 = uniform cytoplasm.
            NaN when either region is empty / the cytoplasm mean is ~0.

    NaN guard: a missing region (no nucleus inside the cell, no cytoplasm) NaNs the
    affected columns; an all-NaN crop / shape mismatch NaNs everything (schema-
    stable).

    Args:
        channel_crop: 3D intensity crop (Z, Y, X) at the cell bbox, aligned with
            ``cell_mask`` / ``nucleus_mask``.
        cell_mask: 3D boolean cell mask (the cell crop).
        nucleus_mask: 3D boolean IN-CELL nucleus mask (same crop frame). The caller
            passes the nucleus piece INSIDE the cell (consensus pairing).
        prefix: column-name prefix (matches the channel's intensity prefix).
        spacing: accepted for signature uniformity (the means are intensity-only;
            the dilation is index-based). None = voxel units.
        perinuclear_dilation: binary-dilation iterations for the perinuclear rim
            (default 2 voxels).

    Mop-up O remainder (4 more): ``<prefix>_nuclear_signal_fraction``
    (the share of the cell's total signal MASS in the nucleus), the per-compartment
    ``<prefix>_nuclear_texture_entropy`` / ``<prefix>_cytoplasmic_texture_entropy``
    (robust histogram entropy WITHIN each compartment), and
    ``<prefix>_compartment_mismatch_score`` (the symmetric nuclear-vs-cytoplasm mean
    disagreement ∈ [0,1] — a marker-compartment QC stand-in).

    Returns:
        ``{f"{prefix}{suffix}": float}`` for the 8 compartment columns.
    """
    from scipy.ndimage import binary_dilation

    out = {f"{prefix}{s}": np.nan for s in _CHANNEL_COMPARTMENT_SUFFIXES}

    cell = np.asarray(cell_mask).astype(bool)
    nuc = np.asarray(nucleus_mask).astype(bool)
    ch = np.asarray(channel_crop, dtype=np.float64)
    if ch.shape != cell.shape or cell.shape != nuc.shape or not cell.any():
        return out
    if not np.isfinite(ch[cell]).any():
        return out  # all-NaN crop -> schema-stable NaN

    nuc_in = nuc & cell                      # nucleus restricted to the cell
    cyto = cell & ~nuc_in                    # cytoplasm = cell minus nucleus

    def _mean(region):
        if not region.any():
            return np.nan
        v = ch[region]
        v = v[np.isfinite(v)]
        return float(v.mean()) if v.size else np.nan

    nuc_mean = _mean(nuc_in)
    cyto_mean = _mean(cyto)
    out[f"{prefix}_nuclear_mean"] = nuc_mean
    out[f"{prefix}_cytoplasm_mean"] = cyto_mean
    if np.isfinite(nuc_mean) and np.isfinite(cyto_mean) and abs(cyto_mean) > 1e-9:
        out[f"{prefix}_nuclear_cytoplasm_ratio"] = nuc_mean / cyto_mean

    # perinuclear rim = (dilated nucleus − nucleus) ∩ cytoplasm.
    if nuc_in.any() and cyto.any() and np.isfinite(cyto_mean) and abs(cyto_mean) > 1e-9:
        dil = binary_dilation(nuc_in, iterations=int(perinuclear_dilation))
        rim = dil & ~nuc_in & cyto
        rim_mean = _mean(rim)
        if np.isfinite(rim_mean):
            out[f"{prefix}_perinuclear_enrichment"] = rim_mean / cyto_mean

    # --- Mop-up O compartment remainder ------------------------
    # nuclear SIGNAL fraction (`nuclear_fraction_<channel>`): the share of
    # the cell's TOTAL (non-negative) signal mass that sits in the nucleus —
    # `Σ_nuc I / Σ_cell I`. Distinct from the nuclear/cytoplasm MEAN ratio: this
    # is the fraction of expression localized to the nucleus regardless of the
    # compartment volumes. [0,1].
    cell_vals = ch[cell]
    cell_fin = cell_vals[np.isfinite(cell_vals)]
    if cell_fin.size:
        base = float(cell_fin.min())
        cell_pos = float(np.clip(cell_fin - base, 0.0, None).sum())
        if cell_pos > 1e-12 and nuc_in.any():
            nv = ch[nuc_in]
            nv = nv[np.isfinite(nv)]
            if nv.size:
                nuc_pos = float(np.clip(nv - base, 0.0, None).sum())
                out[f"{prefix}_nuclear_signal_fraction"] = nuc_pos / cell_pos

    # compartment-restricted TEXTURE entropy (nuclear/cytoplasmic_<channel>_
    # texture_entropy): the SAME robust histogram entropy as the whole-cell
    # texture, restricted to each compartment — is the signal patchy WITHIN the
    # nucleus / cytoplasm? (e.g. condensed chromatin vs diffuse). NaN for an empty
    # compartment.
    if nuc_in.any():
        out[f"{prefix}_nuclear_texture_entropy"] = _region_intensity_entropy(ch[nuc_in])
    if cyto.any():
        out[f"{prefix}_cytoplasmic_texture_entropy"] = _region_intensity_entropy(ch[cyto])

    # expected-compartment mismatch score (`marker_compartment_mismatch_score`):
    # a parameter-free disagreement signal between the two compartments' means,
    # `|nuclear_mean − cytoplasm_mean| / (|nuclear_mean| + |cytoplasm_mean|)` ∈
    # [0,1]: 0 = the marker is evenly present in both compartments, → 1 = it is
    # sharply restricted to ONE (strongly compartmentalized — or, read as QC,
    # absent from the compartment it was expected in). A symmetric, unsupervised
    # stand-in for the named expected-compartment QC flags (which need a per-marker
    # expectation table the engine does not carry).
    if (np.isfinite(nuc_mean) and np.isfinite(cyto_mean)):
        denom = abs(nuc_mean) + abs(cyto_mean)
        if denom > 1e-12:
            out[f"{prefix}_compartment_mismatch_score"] = abs(nuc_mean - cyto_mean) / denom

    return out


# P1b — marker-EXPECTATION QC suffixes. Emitted by
# compute_marker_expectation_flags as ``<prefix>{suffix}`` ONLY when the caller
# supplies a per-marker expected compartment. Recovered dynamically at manifest time
# (the prefix + expectation are a runtime choice), like the per-channel families.
_MARKER_EXPECTATION_SUFFIXES = (
    "_in_expected_compartment_fraction",
    "_missing_expected_signal_flag",
    "_compartment_concordance",
)
# the three compartments a marker can be expected in.
_MARKER_EXPECTED_COMPARTMENTS = ("nuclear", "cytoplasmic", "membrane")
# below this in-expected-compartment fraction (with real total signal present) the
# marker is "missing from where it was expected" -> the QC flag fires.
_MARKER_MISSING_FRACTION_FLOOR = 0.25


def compute_marker_expectation_flags(
    channel_crop: np.ndarray,
    cell_mask: np.ndarray,
    nucleus_mask: np.ndarray,
    prefix: str,
    expected: str,
    spacing: Optional[Tuple[float, ...]] = None,
    membrane_dilation: int = 1,
) -> dict:
    """Expected-compartment QC flags for ONE marker on ONE cell (P1b).

    This is BIOLOGY CONFIG, not geometry: the caller declares where a marker
    *should* localise (``"nuclear"`` / ``"cytoplasmic"`` / ``"membrane"``) via a
    ``marker_expectations`` table, and this function checks whether the OBSERVED
    signal actually sits there. The geometric compartments are the SAME ones the
    Wave-F compartment family uses (the in-cell nucleus, the cytoplasm = cell minus
    nucleus, and the membrane = outer surface rind), so a marker's expected
    localisation is scored against a consistent partition. It runs in the reconcile
    PAIRED path (needs the cell + nucleus masks). RETURNS a flat ``{column: value}``
    dict (one cell, one marker).

    Columns emitted (3, ``<prefix>``-prefixed):
        ``<prefix>_in_expected_compartment_fraction`` — the share of the cell's TOTAL
            (background-subtracted, non-negative) signal MASS that sits in the
            EXPECTED compartment. ~1 = the marker is exactly where it should be, ~0 =
            it is somewhere else. [0, 1]. NaN when the cell has no signal mass or the
            expected compartment is empty. CAVEAT: this is a MASS fraction, so it is
            VOLUME-BIASED toward the LARGER compartment — a marker that is correctly
            cytoplasmic scores naturally high simply because the cytoplasm is the
            biggest compartment, while a correctly NUCLEAR marker is penalised by the
            nucleus's small volume. Prefer ``_compartment_concordance`` (the per-mean
            dominant-compartment check, volume-independent) as the PRIMARY mis-
            localisation flag; read the mass fraction as a soft secondary signal.
        ``<prefix>_missing_expected_signal_flag`` — 1 when that fraction is
            implausibly low (< :data:`_MARKER_MISSING_FRACTION_FLOOR`) AND the cell
            DOES carry signal (so it is a genuine mis-localisation / dropout, not an
            empty cell). 0 otherwise. The actionable QC flag. NaN when undefined (no
            signal / empty compartment).
        ``<prefix>_compartment_concordance`` — 1.0 when the OBSERVED dominant
            compartment (the one with the highest mean intensity, among
            nuclear/cytoplasmic/membrane) EQUALS the expected one, else 0.0. A
            categorical agreement check independent of the mass fraction. NaN when no
            compartment resolves.

    When ``expected`` is not one of ``"nuclear"`` / ``"cytoplasmic"`` / ``"membrane"``
    the function returns all-NaN (a typo in the config is not silently scored).

    Args:
        channel_crop: 3D intensity crop (Z, Y, X) at the cell bbox.
        cell_mask: 3D boolean cell mask (the cell crop).
        nucleus_mask: 3D boolean IN-CELL nucleus mask (same crop frame).
        prefix: column-name prefix (matches the channel's intensity prefix).
        expected: the marker's expected compartment — ``"nuclear"`` /
            ``"cytoplasmic"`` / ``"membrane"``.
        spacing: accepted for signature uniformity (means are intensity-only). None.
        membrane_dilation: NOT used to grow — the membrane = the outer surface rind
            (the cell's boundary shell), built by a 1-step binary erosion of the
            cell. Kept for API symmetry / future tuning.

    Returns:
        ``{f"{prefix}{suffix}": value}`` for the 3 expected-compartment columns.
    """
    out = {f"{prefix}{s}": np.nan for s in _MARKER_EXPECTATION_SUFFIXES}
    if expected not in _MARKER_EXPECTED_COMPARTMENTS:
        return out

    cell = np.asarray(cell_mask).astype(bool)
    nuc = np.asarray(nucleus_mask).astype(bool)
    ch = np.asarray(channel_crop, dtype=np.float64)
    if ch.shape != cell.shape or cell.shape != nuc.shape or not cell.any():
        return out
    if not np.isfinite(ch[cell]).any():
        return out

    nuc_in = nuc & cell
    cyto = cell & ~nuc_in
    # membrane = the cell's outer surface rind (cell minus its 1-step erosion). The
    # SAME boundary convention compute_channel_localization's outer shell targets.
    membrane = cell & ~_binary_erosion_local(cell)

    compartments = {
        "nuclear": nuc_in,
        "cytoplasmic": cyto,
        "membrane": membrane,
    }

    def _mean(region):
        if not region.any():
            return np.nan
        v = ch[region]
        v = v[np.isfinite(v)]
        return float(v.mean()) if v.size else np.nan

    # --- mass fraction in the expected compartment --------------------------
    cell_vals = ch[cell]
    cell_fin = cell_vals[np.isfinite(cell_vals)]
    exp_region = compartments[expected]
    if cell_fin.size and exp_region.any():
        base = float(cell_fin.min())                    # background-subtract baseline
        cell_pos = float(np.clip(cell_fin - base, 0.0, None).sum())
        if cell_pos > 1e-12:
            ev = ch[exp_region]
            ev = ev[np.isfinite(ev)]
            exp_pos = float(np.clip(ev - base, 0.0, None).sum()) if ev.size else 0.0
            frac = exp_pos / cell_pos
            out[f"{prefix}_in_expected_compartment_fraction"] = frac
            # missing-expected flag: low fraction AND the cell carries real signal
            # (a non-trivial positive mass) -> a genuine dropout / mis-localisation.
            out[f"{prefix}_missing_expected_signal_flag"] = float(
                frac < _MARKER_MISSING_FRACTION_FLOOR)

    # --- categorical concordance (observed dominant == expected) ------------
    means = {k: _mean(v) for k, v in compartments.items()}
    finite_means = {k: m for k, m in means.items() if np.isfinite(m)}
    if finite_means:
        observed = max(finite_means, key=finite_means.get)
        out[f"{prefix}_compartment_concordance"] = float(observed == expected)

    return out


# P1b — at-nucleus / perinuclear local-expression-context suffixes.
# Emitted by compute_channel_local_context as ``<prefix>{suffix}`` in the PAIRED
# path (needs the nucleus mask). Recovered dynamically at manifest time (the prefix
# is a runtime choice), like the compartment family.
_CHANNEL_LOCAL_CONTEXT_SUFFIXES = (
    "_at_nucleus_mean",
    "_perinuclear_shell_mean",
    "_perinuclear_to_nuclear_ratio",
)


def compute_channel_local_context(
    channel_crop: np.ndarray,
    cell_mask: np.ndarray,
    nucleus_mask: np.ndarray,
    prefix: str,
    spacing: Optional[Tuple[float, ...]] = None,
    perinuclear_dilation: int = 2,
) -> dict:
    """At-nucleus + perinuclear-shell expression context for ONE cell (P1b).
    The local (sub-cellular) expression-context expansion that needs
    BOTH the cell mask AND its nucleus mask, so it runs in the reconcile PAIRED path
    alongside :func:`compute_channel_compartments`. RETURNS a flat ``{column: value}``
    dict (one cell).

    Distinct from :func:`compute_channel_compartments`: that gives the WHOLE-nucleus
    vs WHOLE-cytoplasm means + a perinuclear ENRICHMENT (perinuclear/cytoplasm). This
    gives the explicit AT-NUCLEUS mean + the perinuclear SHELL mean + their ratio —
    the named ``<channel>_at_nucleus_mean`` / ``<channel>_perinuclear_shell_mean``
    rows the catalogue lists separately.

    Columns emitted (3, ``<prefix>``-prefixed):
        ``<prefix>_at_nucleus_mean`` — mean intensity INSIDE the in-cell nucleus mask
            (the raw nuclear signal level; the same region as ``_nuclear_mean`` but
            named for the at-nucleus localization row). NaN with no in-cell nucleus.
        ``<prefix>_perinuclear_shell_mean`` — mean intensity in the PERINUCLEAR shell
            = the in-cell nucleus dilated by ``perinuclear_dilation`` voxels MINUS the
            nucleus, clipped to the CELL (a rim hugging the nuclear envelope, in the
            cytoplasm). NaN when the shell is empty.
        ``<prefix>_perinuclear_to_nuclear_ratio`` — ``perinuclear_shell_mean /
            at_nucleus_mean`` (> 1 = the signal ramps UP just outside the nucleus —
            ER / Golgi / perinuclear; < 1 = it is brighter inside the nucleus). NaN
            when the nuclear mean is ~0 or either region is empty.

    NaN guard: a missing region / shape mismatch / all-NaN crop NaNs the affected
    columns (schema-stable).

    Args:
        channel_crop: 3D intensity crop (Z, Y, X) at the cell bbox.
        cell_mask: 3D boolean cell mask (the cell crop).
        nucleus_mask: 3D boolean IN-CELL nucleus mask (same crop frame).
        prefix: column-name prefix (matches the channel's intensity prefix).
        spacing: accepted for signature uniformity (means are intensity-only; the
            dilation is index-based). None = voxel units.
        perinuclear_dilation: binary-dilation iterations for the perinuclear rim
            (default 2 voxels, matching :func:`compute_channel_compartments`).

    Returns:
        ``{f"{prefix}{suffix}": float}`` for the 3 local-context columns.
    """
    from scipy.ndimage import binary_dilation

    out = {f"{prefix}{s}": np.nan for s in _CHANNEL_LOCAL_CONTEXT_SUFFIXES}

    cell = np.asarray(cell_mask).astype(bool)
    nuc = np.asarray(nucleus_mask).astype(bool)
    ch = np.asarray(channel_crop, dtype=np.float64)
    if ch.shape != cell.shape or cell.shape != nuc.shape or not cell.any():
        return out
    if not np.isfinite(ch[cell]).any():
        return out

    nuc_in = nuc & cell
    if not nuc_in.any():
        return out  # no in-cell nucleus -> all local context undefined

    def _mean(region):
        if not region.any():
            return np.nan
        v = ch[region]
        v = v[np.isfinite(v)]
        return float(v.mean()) if v.size else np.nan

    nuc_mean = _mean(nuc_in)
    out[f"{prefix}_at_nucleus_mean"] = nuc_mean

    # perinuclear shell = (dilated nucleus − nucleus) ∩ cell (clipped to the cell so
    # the rim stays inside the cytoplasm, not bleeding into a neighbour).
    dil = binary_dilation(nuc_in, iterations=int(perinuclear_dilation))
    rim = dil & ~nuc_in & cell
    rim_mean = _mean(rim)
    out[f"{prefix}_perinuclear_shell_mean"] = rim_mean

    if np.isfinite(rim_mean) and np.isfinite(nuc_mean) and abs(nuc_mean) > 1e-9:
        out[f"{prefix}_perinuclear_to_nuclear_ratio"] = rim_mean / nuc_mean

    return out


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

    NOTE (engine): when this is called on a per-cell bbox CROP rather than the
    full tissue volume, "border" means "touches the crop face", which is not
    the same as "touches the tissue boundary". The FeatureStage therefore
    computes the global border flag from the provenance ``touches_border``
    field, not from per-cell crops. This function is kept for parity and for
    whole-volume callers.

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


def compute_centroid_spacing(
    df: pd.DataFrame,
    k_neighbors: int = 6,
    density_radius: Optional[float] = None,
) -> pd.DataFrame:
    """Centroid-based neighbor spacing / crowding via a KD-tree (Wave-1, new).

    The one genuinely-new Wave-1 feature the plan calls for (the monorepo has
    no cKDTree crowding — only the dilation-based contact graph). Computed
    GLOBALLY on the assembled per-cell table (needs every cell's centroid), so
    the FeatureStage runs this ONCE after the per-cell bbox pass, not per crop.

    Distinct from the Wave-2 gap-aware contact graph: this measures
    centroid-to-centroid geometry, NOT surface gaps. Column names are
    deliberately prefixed (``nn_centroid_distance``) so they never collide with
    the contact graph's ``nearest_neighbor_distance`` (a surface-gap width).

    Columns added:
        ``nn_centroid_distance`` — distance to the nearest OTHER centroid.
        ``knn_mean_distance`` — mean distance to the ``k_neighbors`` nearest.
        ``neighbor_count_in_radius`` — number of OTHER centroids within
            ``density_radius``.
        ``local_density`` — ``neighbor_count_in_radius`` per unit volume
            (count / (4/3 · π · r³)).

    Distances inherit the unit of the centroid columns: physical (µm) when
    ``extract_features`` was called with ``spacing``, voxels otherwise.

    Args:
        df: assembled per-cell table with ``centroid-0/1/2`` (ZYX) columns.
        k_neighbors: K for the mean-KNN distance (default 6 ≈ first 3D shell).
        density_radius: radius for the count/density features (same unit as
            the centroids). None → auto = 3 × median(nn_centroid_distance) of
            this population (self-scaling).

    Returns:
        The same DataFrame with the 4 columns added.
    """
    from scipy.spatial import cKDTree

    n = len(df)
    out_cols = (
        "nn_centroid_distance", "knn_mean_distance",
        "neighbor_count_in_radius", "local_density",
    )
    if n == 0:
        for c in out_cols:
            df[c] = pd.Series(dtype=np.float64)
        return df

    cen_cols = ["centroid-0", "centroid-1", "centroid-2"]
    if not all(c in df.columns for c in cen_cols):
        # Centroids unavailable — genuinely undefined → NaN (plan).
        for c in out_cols:
            df[c] = np.nan
        return df

    pts = df[cen_cols].values.astype(np.float64)

    nn_dist = np.full(n, np.nan, dtype=np.float64)
    knn_mean = np.full(n, np.nan, dtype=np.float64)
    nbr_count = np.zeros(n, dtype=np.float64)
    density = np.zeros(n, dtype=np.float64)

    if n == 1:
        # A lone cell has no neighbors — distances undefined, counts zero.
        df["nn_centroid_distance"] = nn_dist
        df["knn_mean_distance"] = knn_mean
        df["neighbor_count_in_radius"] = nbr_count
        df["local_density"] = density
        return df

    tree = cKDTree(pts)
    # Query k+1 because the first hit is the point itself (distance 0).
    k = min(k_neighbors + 1, n)
    dists, _ = tree.query(pts, k=k)
    if dists.ndim == 1:  # k == 1 edge case → 1-D result
        dists = dists[:, None]
    # dists[:, 0] is self (0); neighbors start at column 1.
    nn_dist = dists[:, 1]
    if k > 2:
        knn_mean = dists[:, 1:].mean(axis=1)
    else:
        knn_mean = dists[:, 1]

    # Density radius: auto = 3 × median nearest-neighbor distance.
    r = density_radius
    if r is None:
        finite = nn_dist[np.isfinite(nn_dist)]
        med_nn = float(np.median(finite)) if finite.size else 0.0
        if med_nn <= 0:
            # Degenerate (>= half the centroids coincide, so the median NN is 0)
            # — fall back to the median of the POSITIVE nearest-neighbour
            # distances so the radius features aren't silently all-zero. Can't
            # occur with distinct labels from a real segmentation.
            positive = finite[finite > 0]
            med_nn = float(np.median(positive)) if positive.size else 0.0
        r = 3.0 * med_nn
    if r and r > 0:
        # count points within r (includes the self-hit at distance 0).
        counts = tree.query_ball_point(pts, r, return_length=True)
        nbr_count = np.clip(np.asarray(counts, dtype=np.float64) - 1.0, 0, None)
        sphere_vol = (4.0 / 3.0) * np.pi * (r ** 3)
        if sphere_vol > 0:
            density = nbr_count / sphere_vol

    df["nn_centroid_distance"] = nn_dist
    df["knn_mean_distance"] = knn_mean
    df["neighbor_count_in_radius"] = nbr_count
    df["local_density"] = density
    return df


def compute_neighbor_features(
    labels_3d: np.ndarray,
    df: pd.DataFrame,
    max_radius: int = 5,
    slices: Optional[list] = None,
    area_lookup: Optional[dict] = None,
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
        slices: optional precomputed ``find_objects`` bbox list (indexed by
            ``label_id - 1``). When given, the internal full-volume
            ``find_objects`` scan is skipped — this is the S1 scaling hook
            (scaling stage S1): the caller passes the bboxes it
            already computed and a memmap ``labels_3d`` so each cell reads only
            its halo crop off disk instead of materialising the whole volume.
            ``None`` → compute internally (the original, byte-identical default).

    Returns:
        The same DataFrame with 10 columns added:
        neighbor_count, nearest_neighbor_distance, max_contact_fraction,
        background_fraction (original 4), plus birth-radius filtration
        features: nn_count_r1, nn_count_r3, nn_birth_median, nn_birth_std,
        nn_accumulation_rate, nn_accumulation_knee (6 new).

    NOTE (engine port): this is a GLOBAL pass — the graduated dilation works
    in VOXEL units (no ``spacing``; the gap width / radii are voxel counts) on
    the FULL label volume, because adjacency is inherently global (a per-cell
    bbox crop cannot see its neighbours). The engine runs it ONCE on the
    assembled per-cell table, like :func:`compute_centroid_spacing`'s cKDTree
    crowding pass. A chunked/halo adjacency scheme that avoids loading the
    whole volume is a FUTURE scale optimisation — do not reinvent it; this
    proven monorepo implementation is ported verbatim.
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

    # S1 scaling hook: reuse the caller's precomputed bboxes (off the memmap)
    # instead of a redundant full-volume find_objects + int32 copy. None => the
    # original byte-identical internal scan.
    if slices is None:
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

    # Neighbor quality features (Group B) — compute from stored maps.
    # ``area_lookup`` injection (S1.5 chunk-parallel): the neighbour SIZE ratios
    # need every NEIGHBOUR's area, but a neighbour may live in another worker's
    # row-chunk (absent from this chunk's df). The chunk driver passes the FULL
    # population ``{label: area}`` so the ratios stay byte-identical to serial.
    # ``None`` (the default / serial path) -> build it from this df, exactly as
    # before (byte-identical).
    if area_lookup is not None or "area" in df.columns:
        if area_lookup is None:
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


def write_edges_csv(df: pd.DataFrame, path, node_ids=None) -> bool:
    """Persist the cell contact graph (``df.attrs["_adjacency_list"]``) to ``edges.csv``.

    Writes one row per undirected cell-cell contact discovered by
    :func:`compute_neighbor_features` (the graduated-dilation adjacency graph),
    with columns ``label_a, label_b, contact_voxels`` (``label_a < label_b``,
    already deduplicated upstream at the ``_adjacency_list`` build). The node set
    of the graph is the cell feature table itself (``label`` + ``centroid-0/1/2``),
    so ``edges.csv`` + the feature table together load as a napari-render-ready 3D
    spatial graph (Points at the centroids + weighted edges). This is the
    persisted form of the in-memory ``df.attrs["_adjacency_list"]`` (which the
    docstring at the build site warns is "NOT persisted in the CSV — must be saved
    separately").

    No-op (returns ``False``) when the ``_adjacency_list`` attribute is missing or
    empty — i.e. the adjacency pass was not run
    (``extract_label_features(adjacency=False)``) or the volume had no touching
    cells. Mirrors the plugin's :func:`save_adjacency_csv`
    (Step 4's adjacency write); the engine names the artifact ``edges.csv``.

    ``contact_voxels`` is a contact-EXTENT WEIGHT, not a planar area: it is the
    count of graduated-dilation ring voxels that landed on the neighbour, so for two
    touching cells it is ≈ (interface area) × ``max_radius`` — a monotone proxy for
    contact strength, not a physical area. A per-edge PHYSICAL contact area
    (``contact_area_um2``) is deliberately NOT emitted: under anisotropic spacing
    there is no single honest voxel→area factor for a dilation-ring count. The
    per-CELL aggregate contact area lives in the feature table
    (``compute_contact_geometry``); a principled per-EDGE physical area is a future
    extension (it needs the contact-patch geometry resolved per edge).

    Args:
        df: the assembled per-cell feature table carrying
            ``df.attrs["_adjacency_list"]`` (set by
            :func:`compute_neighbor_features` during the ``adjacency=True`` pass).
        path: destination CSV path (conventionally ``<run>/edges.csv``).
        node_ids: optional iterable of the label ids that are actually NODES in the
            persisted feature table. When given, edges with an endpoint NOT in this
            set are DROPPED so the saved graph has no dangling node (every endpoint
            renders at a real centroid). This matters when the table was filtered
            (e.g. ``FeatureStage(min_voxels > 1)``) but the graph saw the whole
            volume: a dropped sub-floor cell can be a graph neighbour, so its edges
            would reference a node absent from ``cell_features.csv``. ``None`` (the
            default; used by the no-filter ``extract_label_features`` path, which
            keeps every cell) writes every edge. NOTE: ``neighbor_count`` in the
            table still counts the dropped physical neighbours — it is the true
            count; ``edges.csv`` is the renderable subgraph.

    Returns:
        ``True`` if a file was written, ``False`` if there was no adjacency data
        (or every edge was filtered out by ``node_ids``).
    """
    from pathlib import Path

    adj_rows = df.attrs.get("_adjacency_list")
    if not adj_rows:
        return False
    if node_ids is not None:
        valid = {int(x) for x in node_ids}
        adj_rows = [r for r in adj_rows
                    if int(r["label_a"]) in valid and int(r["label_b"]) in valid]
        if not adj_rows:
            return False                 # every edge referenced a dropped node
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Fixed column order (drops any stray keys; deterministic schema for the
    # napari graph reader + the eventual mirror-port).
    edges = pd.DataFrame(adj_rows, columns=["label_a", "label_b", "contact_voxels"])
    edges.to_csv(str(out), index=False)
    return True


# --------------------------------------------------------------------------- #
# S1.5 — chunk-parallel graph family (Phase B). The three VOLUME-reading graph  #
# passes (neighbour features / contact geometry / axis-neighbour persistence)   #
# are embarrassingly parallel PER CELL: each cell reads only its own halo crop  #
# off the label volume + its own df row, and writes its columns POSITIONALLY.   #
# So we split ``df`` into CONTIGUOUS row-chunks, run the EXISTING (byte-identical)#
# function on each chunk in a worker (the worker memmaps the labels ONCE), then  #
# merge the per-chunk new columns back at the chunk's row positions. The graph   #
# is GLOBAL (every cell can see every neighbour) — a chunk only limits WHICH     #
# cells are COMPUTED, not which neighbours they may look up — so the full        #
# ``_adjacency_list`` is passed to the contact-geometry + axis passes, and the   #
# neighbour pass concatenates its per-chunk edge lists IN CHUNK ORDER so the     #
# edge order matches the serial append order (truly identical). The worker funcs #
# + the initializer are MODULE-LEVEL (picklable under Windows ``spawn``).        #
# --------------------------------------------------------------------------- #

# Per-process source label memmap handle + the SHARED pass payload for the graph
# workers (set by the initializer; module-level so they are picklable under spawn).
# H2: the bbox ``slices``, the FULL ``_adjacency_list`` (contact-geometry + axis
# passes), and the FULL population ``{label: area}`` (neighbour pass) are
# VOLUME-SCALE objects identical for every chunk in a pass — they are stashed here
# ONCE per worker (one deserialization per process) instead of being pickled into
# EVERY per-chunk task tuple (which broadcast them ``n_workers`` times and spiked
# startup RAM + dispatch time). Mirrors how ``_init_feature_worker`` stashes the
# shared M2 config in module globals.
_GRAPH_WORKER_SRC: Optional[np.ndarray] = None
_GRAPH_WORKER_SLICES: Optional[list] = None
_GRAPH_WORKER_ADJACENCY: Optional[list] = None
_GRAPH_WORKER_AREA_LOOKUP: Optional[dict] = None


def _init_graph_worker(
    labels_path: str,
    slices: Optional[list],
    adjacency: Optional[list] = None,
    area_lookup: Optional[dict] = None,
) -> None:
    """Pool initializer: open the source label memmap + stash the shared pass
    payload ONCE per worker process.

    Mirrors :func:`epicycle.core.feature_stage._init_feature_worker`. The bbox
    ``slices`` (precomputed ``find_objects`` indexed by ``label_id - 1``), the FULL
    ``adjacency`` edge list (read by the contact-geometry + axis passes — the graph
    is global), and the FULL population ``area_lookup`` (read by the neighbour pass
    for the SIZE-ratio columns — a neighbour's area may live in another chunk) are
    stashed in module globals so each chunk task carries only its per-chunk row data,
    not these multi-megabyte / volume-scale objects per task (H2: one
    deserialization per process, not one per task). The opened array is a
    ``np.memmap`` so each per-cell crop streams just its halo off disk (peak private
    RAM tracks crop x workers, NOT volume size — the whole point of S1).
    """
    global _GRAPH_WORKER_SRC, _GRAPH_WORKER_SLICES
    global _GRAPH_WORKER_ADJACENCY, _GRAPH_WORKER_AREA_LOOKUP
    _GRAPH_WORKER_SRC = np.load(labels_path, mmap_mode="r")
    _GRAPH_WORKER_SLICES = slices
    _GRAPH_WORKER_ADJACENCY = adjacency
    _GRAPH_WORKER_AREA_LOOKUP = area_lookup


# The three parallelisable VOLUME functions, keyed by name so the worker (which
# re-imports this module under spawn) can resolve the callable without pickling
# a function object. Each is called as ``fn(labels, df_chunk, ...)`` with the
# SAME signature it has in the serial path.
_GRAPH_PASS_FNS = {
    "compute_neighbor_features": "compute_neighbor_features",
    "compute_contact_geometry": "compute_contact_geometry",
    "compute_axis_neighbor_persistence": "compute_axis_neighbor_persistence",
}


def _graph_chunk_worker(task):
    """Run ONE graph pass on ONE contiguous row-chunk of the table.

    ``task = (fn_name, df_chunk, spacing, max_radius, chunk_pos)`` where
    ``df_chunk`` is a fresh DataFrame holding just this chunk's rows (its index is
    reset 0..len-1 so the function's positional writes land at chunk positions) and
    ``chunk_pos`` is the chunk's row-position array (returned untouched so the
    driver can pair a result with its rows even under ``imap_unordered``-style
    reordering, and to report the cell range on failure). The worker resolves
    ``fn_name`` to the module-level callable, runs it on the chunk against the
    per-process memmap + the SHARED ``slices`` / ``adjacency`` / ``area_lookup``
    (stashed in module globals by :func:`_init_graph_worker` — NOT pickled per task;
    H2), and returns the chunk's NEW columns + (for the neighbour pass) the chunk's
    ``_adjacency_list``.

    H1 — FAILURE ISOLATION (one bad cell must NEVER abort a multi-hour run). The
    whole-chunk body is wrapped in try/except; on ANY exception (e.g. a scipy
    ConvexHull crash on a degenerate cell) it returns a ``("CHUNK-ERROR", chunk_pos,
    repr(e))`` sentinel instead of propagating through ``pool.map`` and killing the
    pass. The driver then LEAVES this chunk's new columns as NaN (the per-cell NaN
    convention) + logs a warning. LIMITATION: the existing per-cell functions
    compute a WHOLE chunk at once (they iterate ``df.label`` internally), so the
    finest grain we can catch here is the chunk — one bad cell NaNs its whole chunk
    (a chunk is ``ceil(n / n_workers)`` cells). The serial path adds a per-cell
    guard separately. Returns ``(new_cols_dict, adj_rows_or_None, chunk_pos)`` on
    success.
    """
    import packages.epicycle_scaledseg.core.qc_features as _qf

    fn_name, df_chunk, spacing, max_radius, chunk_pos = task
    try:
        fn = getattr(_qf, fn_name)
        labels = _qf._GRAPH_WORKER_SRC
        slices = _qf._GRAPH_WORKER_SLICES
        adjacency = _qf._GRAPH_WORKER_ADJACENCY
        area_lookup = _qf._GRAPH_WORKER_AREA_LOOKUP

        before = set(df_chunk.columns)
        if fn_name == "compute_neighbor_features":
            fn(labels, df_chunk, max_radius=max_radius, slices=slices,
               area_lookup=area_lookup)
            adj = df_chunk.attrs.get("_adjacency_list")
        else:
            fn(labels, df_chunk, adjacency=adjacency, spacing=spacing,
               max_radius=max_radius, slices=slices)
            adj = None
        new_cols = {c: df_chunk[c].to_numpy() for c in df_chunk.columns
                    if c not in before}
        return new_cols, adj, chunk_pos
    except Exception as e:  # pragma: no cover - exercised via failure-injection test
        return ("CHUNK-ERROR", chunk_pos, repr(e))


def _graph_pass_parallel(
    fn_name: str,
    labels_path: str,
    df: pd.DataFrame,
    n_workers: int,
    *,
    slices: Optional[list],
    spacing: Optional[Tuple[float, ...]],
    max_radius: int,
    adjacency: Optional[list] = None,
) -> Optional[list]:
    """Chunk-parallel driver for one of the three VOLUME graph passes.

    Splits ``df`` into ``n_workers`` CONTIGUOUS row-chunks (``np.array_split`` over
    the row positions — preserves row order), runs ``fn_name`` on each chunk in a
    ``mp.Pool`` (each worker memmaps ``labels_path`` once), and writes the per-chunk
    NEW columns back into ``df`` at the chunk's row positions. Result is row-for-row
    identical to running ``fn_name`` serially on the whole df.

    For ``compute_neighbor_features`` the per-chunk ``_adjacency_list`` lists are
    concatenated IN CHUNK ORDER and returned (the caller sets
    ``df.attrs["_adjacency_list"]``); the serial path appends edges in df-row order,
    so chunk-order concat reproduces that order exactly. For the contact-geometry +
    axis passes the FULL ``adjacency`` edge list is broadcast to every chunk (the
    graph is global) and ``None`` is returned.

    Modifies ``df`` IN PLACE (adds the pass's columns). Returns the merged
    ``_adjacency_list`` (neighbour pass) or ``None`` (the other two).
    """
    import logging
    import multiprocessing as mp

    n = len(df)
    # Contiguous row-position chunks (drop empties when n < n_workers).
    parts = [p for p in np.array_split(np.arange(n), n_workers) if p.size > 0]

    # Full population {label: area} — the neighbour pass needs every NEIGHBOUR's
    # area for the SIZE-ratio columns, and a neighbour may live in another chunk.
    # Built ONCE here and stashed in the worker initargs (H2) so each worker holds
    # ONE copy for the whole pass — NOT pickled into every per-chunk task. Only
    # meaningful for the neighbour pass (the others ignore it).
    full_area_lookup = None
    if fn_name == "compute_neighbor_features" and "area" in df.columns:
        full_area_lookup = dict(zip(
            df["label"].values.astype(int),
            df["area"].values.astype(np.float64),
        ))

    tasks = []
    for pos in parts:
        chunk = df.iloc[pos].reset_index(drop=True).copy()
        # Carry the upstream graph attrs the contact/axis passes may read from
        # df.attrs (they read the FULL adjacency explicitly via the worker global,
        # so the attr is not strictly needed — but keep the chunk's attrs clean so
        # a stale list can't leak in).
        chunk.attrs = {}
        # H2: the per-task tuple carries ONLY the per-chunk data (df_chunk + its
        # row positions) + the cheap scalars. The volume-scale ``adjacency`` +
        # ``area_lookup`` go into the worker initargs below (one copy per process).
        tasks.append((fn_name, chunk, spacing, max_radius, pos))

    merged_adj: Optional[list] = [] if fn_name == "compute_neighbor_features" else None
    with mp.Pool(len(parts), initializer=_init_graph_worker,
                 initargs=(labels_path, slices, adjacency, full_area_lookup)) as pool:
        results = pool.map(_graph_chunk_worker, tasks)

    # H1: split successes from failed-chunk sentinels. A failed chunk leaves its
    # rows' new columns as NaN (the per-cell NaN convention) + logs a warning with
    # the cell range + the exception — it does NOT abort the run.
    ok_results = []          # list of (new_cols, adj, chunk_pos)
    failed_positions: list = []
    for res in results:
        if isinstance(res, tuple) and len(res) == 3 and res[0] == "CHUNK-ERROR":
            _, chunk_pos, err = res
            failed_positions.append(chunk_pos)
            rng = (f"rows {int(chunk_pos[0])}..{int(chunk_pos[-1])}"
                   if len(chunk_pos) else "rows <empty>")
            try:
                lbls = df["label"].to_numpy()[chunk_pos]
                cells = (f"cells {int(lbls.min())}..{int(lbls.max())}"
                         if lbls.size else "cells <empty>")
            except Exception:
                cells = "cells <unknown>"
            logging.getLogger(__name__).warning(
                "_graph_pass_parallel(%s): chunk failed (%s; %s) — leaving its "
                "columns NaN and continuing: %s", fn_name, rng, cells, err)
        else:
            ok_results.append(res)

    # Write each chunk's new columns back at its row positions. Build full-length
    # arrays so the parent df keeps its identity + attrs + existing columns. Failed
    # chunks contribute nothing, so their rows stay None -> NaN after the cast.
    new_col_names: list = []
    for new_cols, _adj, _pos in ok_results:
        for c in new_cols:
            if c not in new_col_names:
                new_col_names.append(c)
    full: dict = {}
    for c in new_col_names:
        # object dtype container so any per-pass dtype (float64 / NaN) round-trips
        # exactly; cast below per-column from the first chunk that has it.
        full[c] = [None] * n
    for new_cols, adj, pos in ok_results:
        for c, arr in new_cols.items():
            for j, rowpos in enumerate(pos):
                full[c][rowpos] = arr[j]
        if merged_adj is not None and adj:
            merged_adj.extend(adj)
    for c in new_col_names:
        # Reconstruct the column as a numpy array (float64 to match the serial
        # passes, which build np.float64 accumulators for every graph column).
        # None -> NaN gives failed-chunk rows the per-cell NaN convention.
        df[c] = np.asarray(full[c], dtype=np.float64)

    return merged_adj


def run_adjacency_graph_passes(
    df: pd.DataFrame,
    graph_src: np.ndarray,
    *,
    slices: Optional[list] = None,
    spacing: Optional[Tuple[float, ...]] = None,
    max_radius: int = 5,
    n_workers: Optional[int] = None,
    labels_path: Optional[str] = None,
    feature_set: str = "full",
) -> pd.DataFrame:
    """Run the full adjacency / contact-graph GLOBAL-pass sequence on an assembled
    per-cell table, IN PLACE, and set ``df.attrs["_adjacency_list"]``.

    This is the ONE canonical ordering of the graph family, SHARED by
    :func:`epicycle.reconcile.features.extract_label_features` (``adjacency=True``)
    and :class:`epicycle.core.feature_stage.FeatureStage` (``adjacency=True``) so
    both production paths emit byte-identical graph columns (and the eventual
    monorepo mirror-port has a single reference). The graph IS global (every cell
    sees every neighbour) but is computed from per-cell HALO CROPS off
    ``graph_src`` — a ``np.memmap`` (crops stream off disk; peak private RAM tracks
    crop x workers, not volume size) or a resident array. ``slices`` = precomputed
    ``find_objects`` bboxes indexed by ``label_id - 1`` (None → each function
    computes its own — the original behaviour); pass the bboxes you already have
    (per-cell task list / provenance) to skip the redundant full-volume scans.

    The sequence (order matters — later passes read columns/``_adjacency_list`` the
    earlier ones write):
      1. neighbour graph (sets ``_adjacency_list``)   2. contact-patch geometry
      3. (unsigned) tissue frame   4. graph topology   5. axis-neighbour persistence
      6. local-comparative   7. channel neighbour context   8. graph motifs
      9. named composites.

    REQUIRES ``df["label"]`` to be integer-typed (the graph functions index
    ``slices`` with it). No-op on an empty frame. Does NOT run the truly-global
    tissue-depth / territory / lumen / qc_score passes — those are separate
    (they need the whole resident/downsampled volume, not the graph).

    PARALLELISM (S1.5 / Phase B). ``n_workers in (None, 1)`` runs the EXISTING
    SERIAL path UNCHANGED (the byte-identical default — do not perturb it).
    ``n_workers > 1`` runs the three VOLUME passes (neighbour features / contact
    geometry / axis-neighbour persistence) CHUNK-PARALLEL — ``df`` is split into
    contiguous row-chunks, each chunk runs the SAME per-cell function in a worker
    that memmaps ``labels_path`` once, and the chunk results are merged back at the
    chunk's row positions. Output is byte-identical to serial (same per-cell math,
    same column values, same ``_adjacency_list`` order). The three SERIAL TABLE
    passes (tissue frame / graph topology / local-comparative / channel context /
    graph motifs / named composites) stay serial — they are cheap table ops, not
    per-cell volume loops. The parallel path needs the label volume as a FILE PATH
    (``labels_path``) so each worker opens its OWN memmap; when ``labels_path`` is
    None (only an in-memory array is available) it FALLS BACK TO SERIAL with a
    logged note rather than pickling the multi-GB array to every worker.

    ``feature_set="epicycle"`` runs ONLY the two passes in the EpiCycle catalogue
    (:func:`compute_neighbor_features`, :func:`compute_named_composites`); the engine-only passes
    (contact geometry, tissue frame, graph topology, axis persistence, local-comparative, channel
    neighbour context, graph motifs) are skipped, in both the serial and the parallel branch.
    ``"full"`` (default) is the sequence above, unchanged.
    """
    if feature_set not in ("epicycle", "full"):
        raise ValueError(f"feature_set must be 'epicycle' or 'full', got {feature_set!r}")
    if df.empty:
        return df
    sp = tuple(spacing) if spacing is not None else None
    only_catalogue = feature_set == "epicycle"

    # Decide serial vs chunk-parallel. Serial is the byte-identical default and the
    # only path when no FILE PATH is available for the workers' per-process memmap.
    use_parallel = (n_workers is not None and int(n_workers) > 1)
    if use_parallel and labels_path is None:
        import logging
        logging.getLogger(__name__).info(
            "run_adjacency_graph_passes: n_workers=%s requested but no labels_path "
            "(in-memory array source) — falling back to SERIAL (the array is not "
            "pickled per worker).", n_workers)
        use_parallel = False

    if not use_parallel:
        # ---- SERIAL (unchanged byte-identical default) ----
        compute_neighbor_features(graph_src, df, max_radius=max_radius, slices=slices)
        if only_catalogue:
            compute_named_composites(df)
            return df
        compute_contact_geometry(
            graph_src, df, spacing=sp, max_radius=max_radius, slices=slices)
        compute_tissue_frame_features(df, spacing=sp)
        compute_graph_topology(df)
        compute_axis_neighbor_persistence(
            graph_src, df, spacing=sp, max_radius=max_radius, slices=slices)
        compute_local_comparative(df)
        compute_channel_neighbor_context(df)
        compute_graph_motifs(df)
        compute_named_composites(df)
        return df

    # ---- CHUNK-PARALLEL (S1.5) — the three VOLUME passes are parallelised; the
    # table-only passes stay serial. Order is EXACTLY the serial order so the
    # byte-identical contract holds. ----
    nw = max(2, int(n_workers))
    # 1. neighbour graph (builds _adjacency_list — concatenated in chunk order).
    adj = _graph_pass_parallel(
        "compute_neighbor_features", labels_path, df, nw,
        slices=slices, spacing=sp, max_radius=max_radius)
    df.attrs["_adjacency_list"] = adj
    if only_catalogue:
        compute_named_composites(df)
        return df
    # 2. contact-patch geometry (reads the FULL _adjacency_list per chunk).
    _graph_pass_parallel(
        "compute_contact_geometry", labels_path, df, nw,
        slices=slices, spacing=sp, max_radius=max_radius, adjacency=adj)
    # 3. tissue frame  4. graph topology — serial table passes (unchanged).
    compute_tissue_frame_features(df, spacing=sp)
    compute_graph_topology(df)
    # 5. axis-neighbour persistence (reads the FULL _adjacency_list per chunk).
    _graph_pass_parallel(
        "compute_axis_neighbor_persistence", labels_path, df, nw,
        slices=slices, spacing=sp, max_radius=max_radius, adjacency=adj)
    # 6-9. serial table passes (unchanged).
    compute_local_comparative(df)
    compute_channel_neighbor_context(df)
    compute_graph_motifs(df)
    compute_named_composites(df)
    return df


def _contact_voxels_per_neighbor(
    labels_3d: np.ndarray,
    label_id: int,
    obj_slice: tuple,
    neighbor_ids: set,
    max_radius: int = 5,
) -> dict:
    """Recover the contact voxel coordinates of *label_id* against each neighbour.

    M3's :func:`compute_neighbor_features` records contact *counts* + Z-index sets
    but NOT contact voxel coordinates — the contact-patch GEOMETRY (M4 / P3) needs
    the coordinates. Rather than thread coords through that proven (byte-identical)
    pass, we recompute the contact shells here from the label volume using the SAME
    GRADUATED DILATION as M3 (radius 1..``max_radius``): each dilation step reveals a
    new ring, and every ring voxel that lands on a graph neighbour is attributed to
    that neighbour at the radius where it FIRST appears. This makes the contact
    geometry consistent with the M3 ``_adjacency_list`` neighbour set — a
    gapped-but-adjacent neighbour (the common case for reconstructed segmentations,
    discovered by M3 at radius > 1) now contributes contact voxels here too, instead
    of being silently dropped by a single radius-1 shell (which reported 0/NaN
    while the graph said connected).

    Only the *neighbour_ids* discovered by the M3 adjacency graph are kept, so the
    contact geometry stays consistent with the graph (the graduated ring may touch a
    label the graph did NOT list only if it appears beyond ``max_radius`` of the M3
    pass — restricting to the graph's set also drops self/background ids cleanly).
    A neighbour is recorded at the FIRST radius it is met; deeper rings of the same
    neighbour are NOT re-accumulated (the first interface is the contact patch).

    Args:
        labels_3d: the FULL 3D integer label volume (Z, Y, X).
        label_id: the cell whose contacts we want.
        obj_slice: the cell's ``find_objects`` bbox slice into ``labels_3d``.
        neighbor_ids: the set of neighbour label ids from the M3 graph for this cell.
        max_radius: graduated-dilation search radius in voxels — MUST match the M3
            ``compute_neighbor_features`` ``max_radius`` so the contact set matches
            the adjacency set. Default 5 (the M3 default).

    Returns:
        ``{neighbor_id: (M, 3) float64 array of GLOBAL (Z,Y,X) contact-voxel coords}``.
        Empty when the cell has no graph neighbours.
    """
    from scipy.ndimage import binary_dilation

    if not neighbor_ids:
        return {}

    max_radius = max(1, int(max_radius))

    # Expand the crop by max_radius so the full graduated-dilation accumulation
    # stays inside the crop (clamped to the volume bounds) — matching M3.
    expanded = tuple(
        slice(max(s.start - max_radius, 0), min(s.stop + max_radius, dim))
        for s, dim in zip(obj_slice, labels_3d.shape)
    )
    crop = labels_3d[expanded]
    mask = crop == label_id
    if not mask.any():
        return {}

    offset = np.array([expanded[0].start, expanded[1].start, expanded[2].start],
                      dtype=np.float64)

    # Graduated dilation: walk out radius 1..max_radius, and for each graph
    # neighbour collect the ring voxels at the FIRST radius it is met (its contact
    # interface). The default 1-connectivity structure (6-neighbourhood) mirrors
    # the M3 binary_dilation, so the radii align voxel-for-voxel with the graph.
    out: dict = {}
    found: set = set()
    prev_dilated = mask
    for _r in range(1, max_radius + 1):
        if len(found) >= len(neighbor_ids):
            break
        dilated = binary_dilation(prev_dilated)
        ring = dilated & ~prev_dilated
        if ring.any():
            ring_ids = crop[ring]
            ring_coords = np.argwhere(ring)        # crop-local (Z, Y, X)
            for nb in neighbor_ids:
                if nb in found:
                    continue
                sel = ring_ids == nb
                if sel.any():
                    out[int(nb)] = ring_coords[sel].astype(np.float64) + offset
                    found.add(nb)
        prev_dilated = dilated
    return out


def _gini(values: np.ndarray) -> float:
    """Gini coefficient of non-negative values (0 = perfectly even, →1 = one dominates).

    Used for ``contact_balance_index = 1 - Gini`` (so high = balanced/even patches).
    Returns 0.0 for a single value (a lone patch is trivially "even").
    """
    v = np.sort(np.asarray(values, dtype=np.float64))
    nn = v.size
    if nn < 2:
        return 0.0
    s = v.sum()
    if s <= 0:
        return 0.0
    # Mean absolute difference formulation: G = (2 * Σ i*v_i) / (n * Σ v) - (n+1)/n.
    idx = np.arange(1, nn + 1, dtype=np.float64)
    return float((2.0 * np.sum(idx * v)) / (nn * s) - (nn + 1.0) / nn)


# M4 built the first 6 contact-geometry columns; Mop-up M (the feature catalogue
# remainder) adds the 9 per-edge contact-patch shape + centroid-offset + side-
# asymmetry + per-neighbour-patch-count columns, completing the family.
# Module constant so the exact-set family test + the feature_stage registry stay
# in lockstep as the family grows.
CONTACT_GEOMETRY_COLUMNS = (
    # M4 (original 6).
    "contact_area_total", "contact_area_max", "largest_contact_fraction",
    "contact_count", "multi_patch_neighbor_fraction", "contact_balance_index",
    # Mop-up M — per-edge contact-patch geometry.
    "contact_area_mean", "contact_area_std",
    "contact_patch_eccentricity_mean", "contact_patch_eccentricity_max",
    "contact_patch_compactness_mean",
    "contact_centroid_offset_mean", "contact_centroid_offset_max",
    "contact_side_asymmetry", "contact_patch_count_mean",
)


def _patch_eccentricity(coords_phys: np.ndarray) -> float:
    """IN-PLANE eccentricity of a contact PATCH's voxel cloud (Mop-up M).

    A cell–cell contact patch is inherently a thin ~2D sheet, so its SMALLEST
    covariance eigenvalue (the sheet thickness) is ~0 for almost every patch — a
    largest-vs-smallest eccentricity would saturate at ~1 for all planar contacts
    and carry no signal. We instead measure the IN-PLANE footprint shape: the two
    LARGEST eigenvalues of the (mean-centred) coordinate covariance (the patch's
    in-plane extent), returning ``sqrt(1 - λ_mid / λ_max)`` ∈ [0, 1): 0 = a round
    in-plane blob, → 1 = a long thin streak along the interface (the
    "compact blob vs long thin streak" cue). Returns 0.0 for a degenerate
    (< 3 voxel / single-point / zero in-plane spread) patch.
    """
    eps = 1e-12
    if coords_phys.shape[0] < 3:
        return 0.0
    c = coords_phys - coords_phys.mean(axis=0)
    cov = c.T @ c / coords_phys.shape[0]
    try:
        ev = np.linalg.eigvalsh(cov)            # ascending
    except np.linalg.LinAlgError:
        return 0.0
    ev = np.clip(ev, 0.0, None)
    lam_max = float(ev[-1]); lam_mid = float(ev[-2])   # the two IN-PLANE extents
    if lam_max <= eps:
        return 0.0
    return float(np.sqrt(max(0.0, 1.0 - lam_mid / lam_max)))


def compute_contact_geometry(
    labels_3d: np.ndarray,
    df: pd.DataFrame,
    adjacency: Optional[list] = None,
    spacing: Optional[Tuple[float, ...]] = None,
    max_radius: int = 5,
    slices: Optional[list] = None,
) -> pd.DataFrame:
    """Contact-patch GEOMETRY per cell (M4 / the feature catalogue; C3, primitive P3).

    M3 gives contact *counts*; this gives contact *geometry*. For each cell we
    recover its contact voxels against each graph neighbour using the SAME
    graduated dilation as M3 (radius 1..``max_radius``; :func:`_contact_voxels_per_neighbor`),
    split each neighbour's contact into connected PATCHES (a neighbour can touch the
    cell in >1 disconnected place — wrapping / folds / irregular merges), and
    summarise the patch-size distribution. Using the graduated rule (not a single
    radius-1 shell) keeps the contact set CONSISTENT with the M3 ``_adjacency_list``:
    a gapped-but-adjacent neighbour (discovered by M3 at radius > 1, common for
    reconstructed segmentations) now contributes contact voxels here too.

    This is a GLOBAL pass (a per-cell bbox crop can't see its neighbours) — run
    ONCE on the full label volume right after :func:`compute_neighbor_features`,
    inside the ``adjacency=True`` pass, so the ``_adjacency_list`` graph is
    available. ``adjacency`` defaults to ``df.attrs["_adjacency_list"]``.

    Patch areas are voxel COUNTS by default; when ``spacing`` is given each patch's
    voxel count is scaled by the in-plane physical area ``mean(spacing) ** 2`` proxy
    is NOT used — areas stay voxel counts (the ratios/fractions/balance below are
    scale-free, and an exact surface area would need a mesh, which is M5). Spacing
    is therefore accepted for signature uniformity but does not change the (relative)
    outputs; kept so a future signed/physical variant can use it.

    Features added (15 — M4 built the first 6; Mop-up M added the 9
    per-edge contact-patch shape + centroid-offset + side-asymmetry + per-
    neighbour-patch-count columns, completing the family):
        ``contact_area_total`` — total contact voxels over all patches (Σ |c_ij|).
        ``contact_area_max`` — voxels in the single largest contact patch.
        ``largest_contact_fraction`` — ``contact_area_max / contact_area_total``
            ∈ [0, 1]: ~1 means one neighbour/patch dominates the cell's contacts
            (a merge-prone footprint); low means contacts are spread out. Bounded.
        ``contact_count`` — number of DISTINCT contact patches across all
            neighbours (a neighbour touching in 2 disconnected places counts 2).
        ``multi_patch_neighbor_fraction`` — fraction of NEIGHBOURS whose contact
            splits into >1 disconnected patch ∈ [0, 1] (wrapping / fold cue). Bounded.
        ``contact_balance_index`` — ``1 - Gini`` of the per-patch area distribution
            ∈ [0, 1]: 1 = perfectly even patches, →0 = one patch dominates
            (evenness of contact-area allocation). Bounded.
        ``contact_area_mean`` (Mop-up M) — ``mean_j |c_ij|`` the typical PER-EDGE
            (per-neighbour) contact size: total contact voxels / number of
            contacting neighbours. Unbounded >= 0 voxel count.
        ``contact_area_std`` (Mop-up M) — ``std_j |c_ij|`` whether the per-edge
            contact sizes are all similar (0) or very uneven (high). Unbounded >= 0.
        ``contact_patch_eccentricity_mean`` / ``_max`` (Mop-up M) — the mean / max
            over patches of the IN-PLANE eccentricity ``sqrt(1 - λ_mid/λ_max)`` of
            each patch's coordinate covariance ∈ [0, 1) (the two LARGEST eigenvalues
            = the patch's in-plane footprint; the smallest is the trivial sheet
            thickness, excluded): 0 = round in-plane blob contacts, → 1 = long thin
            streak contacts along the interface (a slipped / sliver merge). Bounded.
        ``contact_patch_compactness_mean`` (Mop-up M) — mean over patches of
            ``λ_mid / λ_max`` (== ``1 − eccentricity²``) of each patch's in-plane
            covariance ∈ [0, 1]: the complement cue (1 = round/compact in-plane,
            → 0 = elongated). Bounded.
        ``contact_centroid_offset_mean`` / ``_max`` (Mop-up M) — mean / max
            distance from the CELL centroid to each PER-EDGE contact centroid,
            normalised by the cell's equivalent radius so it is scale-free: ~0 =
            contacts sit near the cell middle, large = contacts pushed to one side
            / the cell's rim. NaN when the cell centroid is unavailable. Unbounded
            >= 0 (normalised distance).
        ``contact_side_asymmetry`` (Mop-up M) — ``||Σ_j w_j u_j|| / Σ_j w_j`` ∈
            [0, 1], where ``u_j`` is the unit direction from the cell centroid to
            edge j's contact centroid and ``w_j = |c_ij|``: 0 =
            contacts balanced all around the cell, → 1 = one-sided exposure (all
            contacts on the same side). NaN when the cell centroid is unavailable.
            Bounded.
        ``contact_patch_count_mean`` (Mop-up M) — the mean number of disconnected
            PATCHES per contacting neighbour (>= 1): 1 = every neighbour touches in
            one place, > 1 = wrapping / repeated separated contacts. Unbounded >= 1.

    Cells with no graph neighbours get ``contact_area_total = contact_area_max =
    contact_count = 0``; the fraction/balance + Mop-up-M summaries are NaN
    (undefined with no contacts). Degenerate (no ``label`` column, empty df)
    yields the empty schema.

    Args:
        labels_3d: the FULL 3D integer label volume (Z, Y, X). Background is 0.
        df: assembled per-cell table (modified in-place). Indexed by ``label``;
            should already carry the M3 adjacency columns (run after
            :func:`compute_neighbor_features`). ``centroid-0/1/2`` enable the
            Mop-up-M centroid-offset + side-asymmetry columns (else those three
            are NaN — the per-patch shape columns are origin-free + still emitted).
        adjacency: the edge list (``[{"label_a","label_b","contact_voxels"}, ...]``)
            from M3. ``None`` → read ``df.attrs["_adjacency_list"]``.
        spacing: voxel spacing ``(Z, Y, X)`` -> the per-edge centroid offsets +
            patch covariance are computed in PHYSICAL units (the offset is then
            normalised by the cell's physical equivalent radius, staying scale-
            free). ``None`` = voxel units. (The voxel-count areas / fractions /
            balance / Gini stay scale-free regardless, as before.)
        max_radius: graduated-dilation search radius in voxels for recovering each
            cell's contact voxels — MUST match the M3 ``compute_neighbor_features``
            ``max_radius`` so the contact set matches the adjacency set. Default 5.

    Returns:
        The same DataFrame with the 15 new columns.
    """
    from scipy.ndimage import find_objects, label as cc_label

    n = len(df)
    cols = CONTACT_GEOMETRY_COLUMNS
    if n == 0:
        for c in cols:
            df[c] = pd.Series(dtype=np.float64)
        return df

    if "label" not in df.columns:
        for c in cols:
            df[c] = np.nan
        return df

    if adjacency is None:
        adjacency = df.attrs.get("_adjacency_list")
    # Build per-cell neighbour-id sets from the (undirected) edge list.
    nb_sets: dict = {}
    if adjacency:
        for e in adjacency:
            a = int(e["label_a"]); b = int(e["label_b"])
            nb_sets.setdefault(a, set()).add(b)
            nb_sets.setdefault(b, set()).add(a)

    # cell centroid (PHYSICAL when spacing was used at extract) -> the offset /
    # side-asymmetry origin. Absent -> those three columns stay NaN.
    cen_cols = ["centroid-0", "centroid-1", "centroid-2"]
    have_cen = all(c in df.columns for c in cen_cols)
    centroids = df[cen_cols].values.astype(np.float64) if have_cen else None
    sp_vec = (np.asarray(spacing, dtype=np.float64)
              if spacing is not None else np.ones(3, dtype=np.float64))
    eps = 1e-12

    area_total = np.zeros(n, dtype=np.float64)
    area_max = np.zeros(n, dtype=np.float64)
    largest_frac = np.full(n, np.nan, dtype=np.float64)
    patch_count = np.zeros(n, dtype=np.float64)
    multi_patch_frac = np.full(n, np.nan, dtype=np.float64)
    balance = np.full(n, np.nan, dtype=np.float64)
    # Mop-up M accumulators.
    area_mean = np.full(n, np.nan, dtype=np.float64)
    area_std = np.full(n, np.nan, dtype=np.float64)
    ecc_mean = np.full(n, np.nan, dtype=np.float64)
    ecc_max = np.full(n, np.nan, dtype=np.float64)
    compact_mean = np.full(n, np.nan, dtype=np.float64)
    centroid_off_mean = np.full(n, np.nan, dtype=np.float64)
    centroid_off_max = np.full(n, np.nan, dtype=np.float64)
    side_asym = np.full(n, np.nan, dtype=np.float64)
    patch_count_mean = np.full(n, np.nan, dtype=np.float64)

    # S1 scaling hook (see compute_neighbor_features): reuse the caller's bboxes
    # off the memmap; None => the original byte-identical internal scan.
    if slices is None:
        slices = find_objects(labels_3d.astype(np.int32))
    label_ids = df["label"].values
    # 26-connectivity so a patch isn't fragmented by diagonal voxel gaps.
    struct_26 = np.ones((3, 3, 3), dtype=bool)

    for idx, label_id in enumerate(label_ids):
        lid = int(label_id)
        if lid < 1 or lid > len(slices):
            continue
        obj_slice = slices[lid - 1]
        if obj_slice is None:
            continue

        neighbor_ids = nb_sets.get(lid, set())
        if not neighbor_ids:
            # No graph neighbours -> zero contacts; fractions/balance undefined.
            continue

        # TODO S1.5b: collapse the 3x dilation — this contact-shell recompute is
        # repeated here + in compute_axis_neighbor_persistence + the directional
        # passes; compute once per cell + cache. (Parallelisation is S1.5; this is
        # the separate internal refactor.)
        contacts = _contact_voxels_per_neighbor(
            labels_3d, lid, obj_slice, neighbor_ids, max_radius=max_radius,
        )
        if not contacts:
            # No contact voxels even after the full graduated dilation (should be
            # rare now that gapped neighbours are reached) -> zero contact area,
            # undefined fractions.
            continue

        patch_sizes: list = []   # one entry per connected patch (across neighbours)
        n_neighbors_with_contact = 0
        n_multi_patch = 0
        # Mop-up M per-cell collectors.
        per_edge_areas: list = []      # total voxels per CONTACTING NEIGHBOUR
        per_edge_patch_n: list = []    # # patches per CONTACTING NEIGHBOUR
        patch_eccs: list = []          # eccentricity per PATCH
        patch_compacts: list = []      # compactness per PATCH
        edge_centroids: list = []      # physical centroid per CONTACTING NEIGHBOUR
        edge_weights: list = []        # |c_ij| weight per CONTACTING NEIGHBOUR

        for _nb, coords in contacts.items():
            # Connected-component split of this neighbour's contact voxels into
            # disconnected PATCHES. Build a tight local mask of just these coords.
            if coords.shape[0] == 0:
                continue
            ci = coords.astype(np.int64)
            mn = ci.min(axis=0)
            local = ci - mn
            shape = local.max(axis=0) + 1
            patch_mask = np.zeros(tuple(int(s) for s in shape), dtype=bool)
            patch_mask[local[:, 0], local[:, 1], local[:, 2]] = True
            cc, n_patches = cc_label(patch_mask, structure=struct_26)
            if n_patches < 1:
                continue
            n_neighbors_with_contact += 1
            if n_patches > 1:
                n_multi_patch += 1
            sizes = np.bincount(cc.ravel())[1:]   # drop background bin 0
            patch_sizes.extend(float(s) for s in sizes)

            # --- Mop-up M per-edge / per-patch geometry ----------------------
            coords_phys = coords.astype(np.float64) * sp_vec   # physical ZYX
            per_edge_areas.append(float(coords.shape[0]))
            per_edge_patch_n.append(int(n_patches))
            # per-PATCH shape: split coords back into patches via the labelled crop.
            patch_labels = cc[local[:, 0], local[:, 1], local[:, 2]]
            for pl in range(1, n_patches + 1):
                sel = patch_labels == pl
                if not sel.any():
                    continue
                pc = coords_phys[sel]
                ecc = _patch_eccentricity(pc)
                patch_eccs.append(ecc)
                # compactness = λ_min/λ_max = 1 - ecc² (the complement cue).
                patch_compacts.append(float(max(0.0, 1.0 - ecc * ecc)))
            # per-EDGE contact centroid (physical) + its weight for side asymmetry.
            edge_centroids.append(coords_phys.mean(axis=0))
            edge_weights.append(float(coords.shape[0]))

        if not patch_sizes:
            continue

        sizes_arr = np.asarray(patch_sizes, dtype=np.float64)
        total = float(sizes_arr.sum())
        mx = float(sizes_arr.max())
        area_total[idx] = total
        area_max[idx] = mx
        patch_count[idx] = float(sizes_arr.size)
        if total > 0:
            largest_frac[idx] = mx / total
            balance[idx] = 1.0 - _gini(sizes_arr)
        if n_neighbors_with_contact > 0:
            multi_patch_frac[idx] = n_multi_patch / n_neighbors_with_contact

        # --- Mop-up M summaries ---------------------------------------------
        if per_edge_areas:
            ea = np.asarray(per_edge_areas, dtype=np.float64)
            area_mean[idx] = float(ea.mean())
            area_std[idx] = float(ea.std())
            patch_count_mean[idx] = float(np.mean(per_edge_patch_n))
        if patch_eccs:
            pe = np.asarray(patch_eccs, dtype=np.float64)
            ecc_mean[idx] = float(pe.mean())
            ecc_max[idx] = float(pe.max())
            compact_mean[idx] = float(np.mean(patch_compacts))
        if have_cen and edge_centroids:
            cell_c = centroids[idx]
            ec = np.asarray(edge_centroids, dtype=np.float64)   # (E, 3) physical
            offs = ec - cell_c[None, :]
            dists = np.linalg.norm(offs, axis=1)
            # equivalent radius from the cell's voxel count (physical) so the
            # offset is scale-free; fall back to the raw distance when area absent.
            r_eq = np.nan
            if "area" in df.columns:
                a = float(df["area"].values[idx])
                vox_vol = float(np.prod(sp_vec))
                if np.isfinite(a) and a > 0 and vox_vol > 0:
                    r_eq = float((3.0 * a / (4.0 * np.pi)) ** (1.0 / 3.0))
            denom = r_eq if (np.isfinite(r_eq) and r_eq > eps) else 1.0
            centroid_off_mean[idx] = float(dists.mean() / denom)
            centroid_off_max[idx] = float(dists.max() / denom)
            # side asymmetry: weighted resultant of the unit directions.
            w = np.asarray(edge_weights, dtype=np.float64)
            dn = np.linalg.norm(offs, axis=1)
            ok = dn > eps
            if ok.any() and w[ok].sum() > eps:
                units = offs[ok] / dn[ok][:, None]
                resultant = (w[ok][:, None] * units).sum(axis=0)
                side_asym[idx] = float(
                    np.linalg.norm(resultant) / w[ok].sum())

    df["contact_area_total"] = area_total
    df["contact_area_max"] = area_max
    df["largest_contact_fraction"] = largest_frac
    df["contact_count"] = patch_count
    df["multi_patch_neighbor_fraction"] = multi_patch_frac
    df["contact_balance_index"] = balance
    df["contact_area_mean"] = area_mean
    df["contact_area_std"] = area_std
    df["contact_patch_eccentricity_mean"] = ecc_mean
    df["contact_patch_eccentricity_max"] = ecc_max
    df["contact_patch_compactness_mean"] = compact_mean
    df["contact_centroid_offset_mean"] = centroid_off_mean
    df["contact_centroid_offset_max"] = centroid_off_max
    df["contact_side_asymmetry"] = side_asym
    df["contact_patch_count_mean"] = patch_count_mean
    return df


# Tissue-frame plane-fit rank-deficiency floor (C9): if the neighbour-offset
# covariance's middle/largest eigenvalue ratio falls below this, the offsets are
# essentially colinear and the fitted normal is arbitrary -> NaN the frame.
_RANK_DEFICIENT_RATIO = 1e-3

# P1a — SIGNED local tissue-frame columns. These ride on the
# Wave C per-cell OUTWARD normal (``tissue_outward_*``) to ORIENT the cell's own
# principal-axis frame against the tissue's apical (= +outward) / basal
# (= -outward) direction. Emitted by :func:`compute_tissue_frame_features` ONLY
# when (a) the ``tissue_outward_*`` columns are present (Wave C ran -> the SIGN
# basis exists) AND, for the mass-moment members, ``full_labels`` is supplied (the
# per-cell voxel coords are needed). Absent entirely (columns not created) when
# Wave C did not run -> the default-off, byte-identical contract.
#
# HONEST FORMULATION (this goes to external review):
#   * apical_basal_axis_alignment is the |cos| MAGNITUDE (columnarity) — the cell's
#     own major-axis eigenvector sign is ARBITRARY, so a "signed cos" of it would be
#     orientation-arbitrary; the POLARITY is carried instead by cell_tilt_polarity
#     (a third-moment of mass along the SIGNED outward axis, no eigenvector sign).
#   * minor_axis_in_tangent_plane_fraction / tangent_plane_anisotropy are SQUARED /
#     eigenvalue summaries -> sign-invariant (immune to the eigenvector ambiguity).
#   * tangential_asymmetry_magnitude is ROTATION-INVARIANT (the magnitude of the
#     SURFACE first-moment in the tangent plane). There is NO canonical in-plane
#     front/back or left/right axis from a single cell + its local normal — those
#     catalogue rows (front_back_asymmetry / left_right_asymmetry /
#     minor_axis_vs_local_tangent_angle) are DEFERRED (need a global tissue
#     reference axis); we ship the rotation-invariant magnitude instead.
# AXIS-only members need only the per-cell principal axes (+ the outward normal),
# so they are defined even without ``full_labels``.
_SIGNED_TISSUE_FRAME_AXIS_COLUMNS = (
    "apical_basal_axis_alignment",          # |cos(u_major, n)| in [0,1] (columnarity)
    "minor_axis_in_tangent_plane_fraction", # 1 - (u_minor·n)^2 in [0,1]
)
# MASS-MOMENT members read each cell's VOXELS (require ``full_labels``): the in-plane
# voxel-covariance anisotropy + the signed projection moments + the surface bulge.
_SIGNED_TISSUE_FRAME_MASS_COLUMNS = (
    "tangent_plane_anisotropy",             # in-plane (λ1-λ2)/(λ1+λ2) of the voxel cov
    "cell_tilt_polarity",                   # skew of (voxel-centroid)·n (signed)
    "apical_vs_basal_elongation",           # signed half-extent asymmetry in [-1,1]
    "tangential_asymmetry_magnitude",       # rotation-invariant surface in-plane bulge
)
SIGNED_TISSUE_FRAME_COLUMNS = (
    _SIGNED_TISSUE_FRAME_AXIS_COLUMNS + _SIGNED_TISSUE_FRAME_MASS_COLUMNS)


def compute_tissue_frame_features(
    df: pd.DataFrame,
    adjacency: Optional[list] = None,
    spacing: Optional[Tuple[float, ...]] = None,
    full_labels: Optional[np.ndarray] = None,
    slices: Optional[list] = None,
) -> pd.DataFrame:
    """Local tissue-frame + orientation coherence per cell (M4 / the feature catalogue;
    C4, primitive P6). **UNSIGNED**.

    For each cell we estimate an UNSIGNED local tissue NORMAL from the spread of
    its neighbours' centroids, then measure how the cell + its neighbourhood sit
    relative to that frame.

    Tissue-normal estimator:
    collect the offset vectors ``v_j = centroid_j - centroid_i`` to the cell's
    graph neighbours, form the (unweighted) covariance ``C = Σ_j v_j v_j^T``, and
    take the eigenvector of its SMALLEST eigenvalue as the local normal ``n_i``.
    Rationale: in an epithelial sheet the neighbours lie mostly IN-PLANE, so the
    direction of LEAST spread is perpendicular to the sheet — i.e. the tissue
    normal. We use UNSIGNED alignments (``|cos|``) throughout because an
    eigenvector's sign is arbitrary and we have no apical/basal reference; a signed
    apical/basal variant is deferred to a later milestone.

    Features added (4):
        ``neighbor_ring_planarity`` — ``1 - λ_min / (λ0+λ1+λ2)`` of the neighbour
            offset covariance ∈ [0, 1]: high when the neighbours are coplanar (a
            flat ring, λ_min ≈ 0), low when they fill 3D isotropically. Bounded.
        ``local_normal_alignment`` — ``|cos(angle(major_axis_i, n_i))|`` ∈ [0, 1]:
            how aligned the cell's OWN long (major inertia) axis is with the
            estimated local tissue normal (1 = the cell stands perpendicular to
            the sheet — columnar; 0 = lies in-plane). Needs the ``major_axis_*``
            columns. Bounded.
        ``neighbor_axis_coherence`` — consistency of the neighbours' orientation:
            ``mean_j (u_i · u_j)^2`` over the neighbours' MAJOR axes when available
            (``mean (u_i·u_j)^2`` ∈ [0,1], 1 = all parallel); FALLS BACK to the
            coherence of the neighbour OFFSET directions when neighbour major axes
            are unavailable (``mean over pairs (d_a·d_b)^2`` of the unit offset
            vectors) so the column is always defined for ≥3 neighbours. Bounded.
        ``neighbor_ring_anisotropy`` — in-plane elongation of the neighbour ring:
            ``1 - λ_mid / λ_max`` of the TWO in-plane covariance eigenvalues
            ∈ [0, 1]: 0 = a round ring, →1 = neighbours strung along a line / arc.
            Bounded.

    Mop-up O orientation-coherence remainder (8 more — all NaN for
    < 3 neighbours, same guard):
        ``contact_weighted_axis_coherence`` — ``Σ_j w_j (u_i·u_j)^2 / Σ_j w_j`` with
            the per-edge contact weight ``w_j = |c_ij|``: the
            neighbour-axis coherence emphasising the neighbours the cell touches
            most. [0, 1]. NaN without neighbour major axes.
        ``neighbor_axis_dispersion`` — ``var_j (u_i·u_j)^2`` across neighbours:
            0 = the neighbours' alignment with the cell is uniform,
            larger = mixed. >= 0. NaN with < 2 axis-bearing neighbours.
        ``major_axis_vs_local_normal_angle`` — ``arccos(|u_i·n_i|)`` in DEGREES:
            the angle (the explicit form of ``local_normal_alignment``'s
            ``|cos|``) between the cell's long axis and the local tissue normal. [0,
            90]. NaN without the cell's major axis.
        ``neighbor_polar_dispersion`` — std (radians) of the neighbours' out-of-plane
            TILT angles (their angle out of the local ring plane): 0 = a flat ring,
            larger = a 3D / out-of-plane neighbourhood. >= 0.
        ``neighbor_azimuth_entropy`` — normalised Shannon entropy of the neighbour
            AZIMUTH histogram in the local in-plane frame: 1 = evenly
            surrounded, → 0 = neighbours clustered in one azimuthal wedge. [0, 1].
        ``local_packing_order`` — a HEURISTIC coordination-regularity proxy (NOT the
            true ψ₆ bond-orientational / hexatic order parameter): a Gaussian on the
            in-plane coordination's distance from the ideal 6, TIMES the azimuthal
            evenness (``neighbor_azimuth_entropy``). 1 = a well-ordered ~hexagonal
            ring, → 0 = under/over-coordinated or one-sided. [0, 1]. The ideal-6
            baseline assumes a CONFLUENT IN-PLANE SHEET; on a non-confluent or
            non-planar neighbourhood the "6" reference is only a heuristic. (Column
            name kept for continuity — it is a proxy, not the physicists' ψ₆.)
        ``coordination_anisotropy`` — ``neighbor_ring_anisotropy`` weighted by
            ``min(k / 6, 1)`` (higher-coordination rings carry more weight, a
            3-neighbour ring's anisotropy is noisier). 0 = round + well-coordinated,
            → 1 = strung-out. [0, 1].
        ``frame_stability`` — how well-determined the local frame is:
            ``planarity × (1 − λ_min/λ_mid)`` — a clean flat ring AND a clear normal
            direction (a large smallest-to-middle eigenvalue gap). [0, 1].

    P1a SIGNED local-frame remainder — emitted ONLY when the
    Wave C ``tissue_outward_*`` columns are present in ``df`` (the apical/basal SIGN
    basis exists); ABSENT (columns not created) otherwise, so the default-off path
    (this function called in the adjacency pass, BEFORE Wave C) is byte-identical.
    The axis members need only the cell's principal axes (``major/mid/minor_axis_*``)
    + the outward normal; the mass-moment members additionally need ``full_labels``
    (per-cell voxel coords). Each is NaN per-cell when its outward normal is
    untrusted (low ``tissue_outward_confidence``) or its prerequisite is degenerate.
        ``apical_basal_axis_alignment`` — ``|cos(u_major, n)|`` ∈ [0, 1]: how
            COLUMNAR the cell is (its long axis aligned with the apical/basal tissue
            normal). MAGNITUDE only — the major-axis eigenvector SIGN is arbitrary,
            so a "signed cos" would be orientation-arbitrary; the polarity lives in
            ``cell_tilt_polarity``. Bounded.
        ``minor_axis_in_tangent_plane_fraction`` — ``1 − (u_minor·n)^2`` ∈ [0, 1]:
            how much the cell's SHORT axis lies in the tangent plane (1) vs along the
            normal (0). Squared -> sign-invariant. Bounded.
        ``tangent_plane_anisotropy`` — in-plane elongation ``(λ1−λ2)/(λ1+λ2)`` ∈
            [0, 1] of the cell's voxel covariance projected onto the tangent plane
            (the two largest eigenvalues of the 2×2 in-plane covariance). 0 = a
            tangentially round cell, → 1 = a tangentially elongated one. Sign-
            invariant. Bounded. NaN without ``full_labels``.
        ``cell_tilt_polarity`` — the standardized THIRD moment (skew) of the SIGNED
            projection ``s = (voxel − centroid)·n`` over the cell's voxels: +ve = the
            cell's mass tail leans APICAL (toward the surface), −ve = BASAL. No
            eigenvector sign enters (the sign comes from the SIGNED ``n``), so it is
            an honest polarity. Unbounded (signed). NaN without ``full_labels``.
        ``apical_vs_basal_elongation`` — ``(σ_apical − σ_basal) / (σ_apical +
            σ_basal)`` ∈ [-1, 1] where ``σ`` is the std of ``s`` within the apical
            (``s>0``) / basal (``s<0``) half: +ve = the apical half spreads further
            along the normal. Signed. NaN without ``full_labels`` / an empty half.
        ``tangential_asymmetry_magnitude`` — the ROTATION-INVARIANT magnitude of the
            cell's SURFACE-voxel first moment IN the tangent plane, normalised by the
            cell's in-plane surface radius: ~0 = the surface bulges evenly all around
            tangentially, large = a one-sided tangential bulge. We use the SURFACE
            (boundary) voxels, NOT all voxels (the mass first-moment about the
            centroid is identically zero), and the MAGNITUDE (NOT a signed
            front/back or left/right), because a single cell + its local normal
            carries NO canonical in-plane axis. ∈ [0, ~1). NaN without ``full_labels``.

    Deliberately DEFERRED — NEEDS A GLOBAL TISSUE REFERENCE AXIS (an embryo AP axis
    / a propagated tangent frame): ``front_back_asymmetry``, ``left_right_asymmetry``,
    ``minor_axis_vs_local_tangent_angle``. A single cell + its local normal defines
    only the normal direction and the tangent PLANE — there is no rotation-equivariant
    way to pick a SIGNED in-plane axis from one cell's local data alone (the same
    arbitrariness as an eigenvector's sign). ``tangential_asymmetry_magnitude`` is the
    rotation-invariant surrogate we ship instead; the signed L/R rows stay NOT.

    All frame features are NaN-guarded: a cell with **< 3 neighbours** (the plane
    fit is undetermined) gets NaN for every UNSIGNED column. ``local_normal_alignment``
    is additionally NaN when the ``major_axis_*`` columns are absent or the cell's
    own axis is degenerate (zero vector).

    Args:
        df: assembled per-cell table (modified in-place). Needs ``label`` +
            ``centroid-0/1/2``; ``major_axis_{z,y,x}`` enable the alignment +
            the preferred neighbour-axis coherence. Run after
            :func:`compute_neighbor_features` so the graph exists. The SIGNED P1a
            columns additionally read the Wave C ``tissue_outward_*`` columns (the
            sign basis) + ``minor_axis_{z,y,x}`` and are emitted only when those
            outward columns exist.
        adjacency: the M3 edge list; ``None`` → ``df.attrs["_adjacency_list"]``.
        spacing: accepted for signature uniformity but NOT applied to the UNSIGNED
            frame — the ``centroid-*`` columns are already physical when
            ``extract_features`` ran with spacing (regionprops scales the centroid;
            the FeatureStage offsets the crop origin by ``origin × spacing``), so the
            neighbour offsets are already in physical units. Re-scaling there would
            double-scale the covariance + put the normal in a different frame than
            the (physical) ``major_axis_*`` it is compared against. It IS used for the
            SIGNED P1a mass-moment members (the voxel coordinates from ``full_labels``
            are scaled by spacing so the projection onto the physical-frame normal is
            geometrically correct).
        full_labels: OPTIONAL full 3D label volume — supplied in the Wave C
            ``tissue_depth=True`` re-call so the SIGNED mass-moment columns
            (``cell_tilt_polarity`` / ``apical_vs_basal_elongation`` /
            ``tangential_asymmetry_magnitude``) can read each cell's voxels. When
            None, those three are NaN (the axis-only signed columns + all the unsigned
            columns are unaffected).
        slices: OPTIONAL precomputed ``find_objects`` bbox list (S1 scaling hook) —
            indexed by ``label_id - 1``, supplied by the caller to skip a redundant
            full-volume ``find_objects(full_labels.astype(int32))`` (whose ``.astype``
            on a memmap forces a FULL RESIDENT copy -> OOM at scale). None => the
            original byte-identical internal scan. Only used when ``full_labels`` is
            supplied; the per-cell crops are then lazy on a memmap.

    Returns:
        The same DataFrame with the 12 unsigned columns (+ the 6 SIGNED P1a columns
        when the Wave C ``tissue_outward_*`` columns are present).
    """
    n = len(df)
    cols = (
        "neighbor_ring_planarity", "local_normal_alignment",
        "neighbor_axis_coherence", "neighbor_ring_anisotropy",
        # Mop-up O orientation-coherence remainder.
        "contact_weighted_axis_coherence", "neighbor_axis_dispersion",
        "major_axis_vs_local_normal_angle",
        "neighbor_polar_dispersion", "neighbor_azimuth_entropy",
        "local_packing_order", "coordination_anisotropy", "frame_stability",
    )
    if n == 0:
        for c in cols:
            df[c] = pd.Series(dtype=np.float64)
        # still emit the (typed-empty) SIGNED columns when Wave C ran.
        _compute_signed_tissue_frame(df, full_labels=full_labels, spacing=spacing)
        return df

    cen_cols = ["centroid-0", "centroid-1", "centroid-2"]
    if "label" not in df.columns or not all(c in df.columns for c in cen_cols):
        for c in cols:
            df[c] = np.nan
        # the SIGNED columns don't need centroids (the axis ones read major/minor
        # axes, the mass ones read voxels) -> still emit them when Wave C ran.
        _compute_signed_tissue_frame(df, full_labels=full_labels, spacing=spacing)
        return df

    if adjacency is None:
        adjacency = df.attrs.get("_adjacency_list")
    nb_sets: dict = {}
    # Per-edge contact weights (|c_ij|) for the contact-weighted axis coherence
    #. Keyed by the unordered (cell, neighbour) -> contact_voxels; weight 1.0
    # when the edge carries no contact_voxels field.
    edge_w: dict = {}
    if adjacency:
        for e in adjacency:
            a = int(e["label_a"]); b = int(e["label_b"])
            nb_sets.setdefault(a, set()).add(b)
            nb_sets.setdefault(b, set()).add(a)
            w = e.get("contact_voxels")
            try:
                w = float(w)
            except (TypeError, ValueError):
                w = 1.0
            if not np.isfinite(w) or w <= 0:
                w = 1.0
            edge_w[(a, b)] = w
            edge_w[(b, a)] = w

    label_ids = df["label"].values.astype(np.int64)
    centroids = df[cen_cols].values.astype(np.float64)
    cen_by_label = {int(label_ids[i]): centroids[i] for i in range(n)}

    axis_cols = ["major_axis_z", "major_axis_y", "major_axis_x"]
    have_axes = all(c in df.columns for c in axis_cols)
    if have_axes:
        axes = df[axis_cols].values.astype(np.float64)
        axis_by_label = {int(label_ids[i]): axes[i] for i in range(n)}
    else:
        axes = None
        axis_by_label = {}

    # NOTE: ``spacing`` is accepted for signature uniformity but is NOT applied —
    # the ``centroid-*`` columns are already physical (see the offset comment
    # below), so scaling them again would double-scale. Kept in the signature so
    # callers don't have to special-case this pass.
    _ = spacing
    eps = 1e-12

    planarity = np.full(n, np.nan, dtype=np.float64)
    normal_align = np.full(n, np.nan, dtype=np.float64)
    axis_coh = np.full(n, np.nan, dtype=np.float64)
    ring_aniso = np.full(n, np.nan, dtype=np.float64)
    # Mop-up O remainder arrays.
    cw_axis_coh = np.full(n, np.nan, dtype=np.float64)
    axis_disp = np.full(n, np.nan, dtype=np.float64)
    maj_vs_normal_ang = np.full(n, np.nan, dtype=np.float64)
    polar_disp = np.full(n, np.nan, dtype=np.float64)
    azimuth_ent = np.full(n, np.nan, dtype=np.float64)
    packing_order = np.full(n, np.nan, dtype=np.float64)
    coord_aniso = np.full(n, np.nan, dtype=np.float64)
    frame_stab = np.full(n, np.nan, dtype=np.float64)

    # ideal in-plane coordination for hexatic packing in a 2D sheet (6 neighbours).
    _IDEAL_COORD = 6.0
    _AZ_NSECTORS = 8

    def _unit(v):
        nrm = float(np.sqrt(np.sum(v * v)))
        return v / nrm if nrm > eps else None

    for idx in range(n):
        lid = int(label_ids[idx])
        neighbor_ids = [nb for nb in nb_sets.get(lid, set()) if nb in cen_by_label]
        if len(neighbor_ids) < 3:
            # Plane fit undetermined with < 3 neighbours -> all NaN (guard).
            continue

        ci = cen_by_label[lid]
        # Offset vectors to neighbours. The ``centroid-*`` columns are ALREADY in
        # physical units when ``extract_features`` ran with ``spacing`` (regionprops
        # scales the centroid, and the FeatureStage offsets the crop origin by
        # ``origin × spacing`` too), so DO NOT re-scale by ``sp`` here — doing so
        # double-scales the covariance, tilting the estimated normal on anisotropic
        # voxels and putting it in a different frame than the (physical-frame)
        # ``major_axis_*`` it is later compared against. The difference of two
        # physical centroids is already a physical offset.
        offsets = np.array(
            [(cen_by_label[nb] - ci) for nb in neighbor_ids],
            dtype=np.float64,
        )
        # Covariance of the offsets (about the cell, NOT mean-centred — we want the
        # spread of the ring as seen FROM the cell). C = Σ v v^T.
        C = offsets.T @ offsets
        try:
            evals, evecs = np.linalg.eigh(C)   # ascending
        except np.linalg.LinAlgError:
            continue
        evals = np.clip(evals, 0.0, None)
        tot = float(evals.sum())
        if tot <= eps:
            continue

        lam_min, lam_mid, lam_max = float(evals[0]), float(evals[1]), float(evals[2])
        # Colinear-neighbour guard (C9): when the neighbour centroids are ~colinear
        # the covariance is rank-deficient (the middle eigenvalue collapses toward
        # the smallest), so the PLANE — and hence the estimated normal — is
        # arbitrary. Reporting a confident high-planarity / normal-alignment for an
        # ambiguous fit is misleading; NaN the whole frame for this cell instead.
        # The test compares the MIDDLE eigenvalue to the LARGEST (the in-plane
        # extent); a tiny ratio means the offsets span essentially one direction.
        if lam_max <= eps or (lam_mid / lam_max) < _RANK_DEFICIENT_RATIO:
            continue
        # Planarity: small smallest-eigenvalue share => neighbours are coplanar.
        planarity[idx] = 1.0 - lam_min / tot
        # In-plane anisotropy from the two LARGER eigenvalues (the in-plane axes).
        if lam_max > eps:
            ring_aniso[idx] = 1.0 - lam_mid / lam_max
        else:
            ring_aniso[idx] = 0.0

        # Local normal = eigenvector of the smallest eigenvalue (least spread).
        normal = _unit(evecs[:, 0])

        # local_normal_alignment: |cos| between the cell's major axis and n_i.
        if normal is not None and have_axes:
            u_i = _unit(axis_by_label.get(lid, np.zeros(3)))
            if u_i is not None:
                normal_align[idx] = float(abs(np.dot(u_i, normal)))

        # neighbour axis coherence: prefer the neighbours' MAJOR axes; else fall
        # back to the coherence of the unit OFFSET directions.
        nb_axes = []
        if have_axes:
            for nb in neighbor_ids:
                ua = _unit(axis_by_label.get(nb, np.zeros(3)))
                if ua is not None:
                    nb_axes.append(ua)
        if have_axes and len(nb_axes) >= 1:
            u_i = _unit(axis_by_label.get(lid, np.zeros(3)))
            if u_i is not None:
                coh = np.mean([float(np.dot(u_i, ua)) ** 2 for ua in nb_axes])
                axis_coh[idx] = float(coh)
        if np.isnan(axis_coh[idx]):
            # Fallback: pairwise coherence of the unit offset directions.
            dirs = [d for d in (_unit(o) for o in offsets) if d is not None]
            if len(dirs) >= 2:
                vals = []
                for a in range(len(dirs)):
                    for b in range(a + 1, len(dirs)):
                        vals.append(float(np.dot(dirs[a], dirs[b])) ** 2)
                if vals:
                    axis_coh[idx] = float(np.mean(vals))

        # --- Mop-up O orientation-coherence remainder -----------
        u_i = _unit(axis_by_label.get(lid, np.zeros(3))) if have_axes else None

        # per-neighbour (u_i·u_j)^2 list (for the dispersion + the contact-weighted
        # coherence). Mirror the preferred axis basis the coherence above uses.
        coh_vals = []          # (u_i·u_j)^2 per neighbour with a usable major axis
        coh_weights = []       # the |c_ij| contact weight for that neighbour
        if u_i is not None:
            for nb in neighbor_ids:
                ua = _unit(axis_by_label.get(nb, np.zeros(3)))
                if ua is not None:
                    coh_vals.append(float(np.dot(u_i, ua)) ** 2)
                    coh_weights.append(edge_w.get((lid, nb), 1.0))
        # neighbor_axis_dispersion: variance of (u_i·u_j)^2 across neighbours
        #.
        if len(coh_vals) >= 2:
            axis_disp[idx] = float(np.var(coh_vals))
        # contact_weighted_axis_coherence: the same coherence weighted by |c_ij|
        #.
        if coh_vals:
            wsum = float(np.sum(coh_weights))
            if wsum > eps:
                cw_axis_coh[idx] = float(
                    np.dot(coh_vals, coh_weights) / wsum)

        # major_axis_vs_local_normal_angle: arccos(|u_i·n_i|) in DEGREES (catalogue
        #) — the explicit angle behind local_normal_alignment's |cos|.
        if normal is not None and u_i is not None:
            c = float(abs(np.dot(u_i, normal)))
            maj_vs_normal_ang[idx] = float(np.degrees(np.arccos(np.clip(c, 0.0, 1.0))))

        # In-plane frame for the polar/azimuth dispersion: the two LARGER-eigenvalue
        # eigenvectors span the neighbour ring's plane; ``normal`` is ⊥ to it. For
        # each neighbour offset, the POLAR angle is its tilt out of the plane (angle
        # with the plane), the AZIMUTH is its direction WITHIN the plane.
        if normal is not None:
            e1 = _unit(evecs[:, 2])     # in-plane major
            e2 = _unit(evecs[:, 1])     # in-plane minor
            if e1 is not None and e2 is not None:
                polar = []              # |angle out of plane| per neighbour, radians
                az = []                 # azimuth in [0, 2π) per neighbour
                for o in offsets:
                    d = _unit(o)
                    if d is None:
                        continue
                    out_comp = float(abs(np.dot(d, normal)))      # |sin(polar)|
                    polar.append(float(np.arcsin(np.clip(out_comp, 0.0, 1.0))))
                    p = float(np.dot(d, e1))
                    q = float(np.dot(d, e2))
                    az.append(float(np.arctan2(q, p)) % (2.0 * np.pi))
                # neighbor_polar_dispersion: std of the out-of-plane tilt angles
                # (radians) — 0 = a perfectly flat ring, larger = neighbours tilt
                # in/out of the local plane (a 3D / disordered neighbourhood).
                if len(polar) >= 2:
                    polar_disp[idx] = float(np.std(polar))
                # neighbor_azimuth_entropy: normalised Shannon entropy of the
                # azimuth histogram — 1 = neighbours evenly all around
                # (surround-like), → 0 = clustered in one azimuthal wedge.
                if az:
                    hist, _ = np.histogram(
                        az, bins=_AZ_NSECTORS, range=(0.0, 2.0 * np.pi))
                    tot_h = hist.sum()
                    if tot_h > 0:
                        pmf = hist.astype(np.float64) / tot_h
                        nz = pmf[pmf > 0]
                        H = -(nz * np.log(nz)).sum()
                        azimuth_ent[idx] = float(H / np.log(_AZ_NSECTORS))

        # local_packing_order: a hexatic-like order parameter. Combine (a) how close
        # the in-plane coordination is to the ideal 6 (a Gaussian falloff in the
        # neighbour count) with (b) the azimuthal evenness (the azimuth entropy).
        # 1 = a well-ordered ~hexagonal in-plane ring, → 0 = under/over-coordinated
        # or one-sided. NaN if the azimuth entropy is undefined.
        if np.isfinite(azimuth_ent[idx]):
            k = float(len(neighbor_ids))
            coord_term = float(np.exp(-((k - _IDEAL_COORD) ** 2) / (2.0 * _IDEAL_COORD)))
            packing_order[idx] = float(coord_term * azimuth_ent[idx])

        # coordination_anisotropy: the in-plane neighbour-ring anisotropy WEIGHTED
        # toward higher-coordination cells (a 3-neighbour ring's anisotropy is
        # noisier than a 6+-neighbour ring's). ring_aniso × min(k/IDEAL, 1). 0 = a
        # round, well-coordinated ring; → 1 = a strung-out / lopsided ring.
        if np.isfinite(ring_aniso[idx]):
            k = float(len(neighbor_ids))
            coord_aniso[idx] = float(ring_aniso[idx] * min(k / _IDEAL_COORD, 1.0))

        # frame_stability: how well-determined the local frame is — high planarity
        # (a clean flat ring) AND a clear normal direction (a large gap between the
        # smallest and the middle eigenvalue). planarity × (1 − lam_min/lam_mid).
        if lam_mid > eps:
            sep = float(np.clip(1.0 - lam_min / lam_mid, 0.0, 1.0))
            frame_stab[idx] = float(planarity[idx] * sep)

    df["neighbor_ring_planarity"] = planarity
    df["local_normal_alignment"] = normal_align
    df["neighbor_axis_coherence"] = axis_coh
    df["neighbor_ring_anisotropy"] = ring_aniso
    df["contact_weighted_axis_coherence"] = cw_axis_coh
    df["neighbor_axis_dispersion"] = axis_disp
    df["major_axis_vs_local_normal_angle"] = maj_vs_normal_ang
    df["neighbor_polar_dispersion"] = polar_disp
    df["neighbor_azimuth_entropy"] = azimuth_ent
    df["local_packing_order"] = packing_order
    df["coordination_anisotropy"] = coord_aniso
    df["frame_stability"] = frame_stab

    # --- P1a SIGNED local-frame remainder ---------------------
    # Only when Wave C ran (the ``tissue_outward_*`` columns exist) -> the
    # apical/basal SIGN basis is available. Absent (no columns created) otherwise
    # -> default-off byte-identical (the adjacency-pass call, BEFORE Wave C, never
    # sees these columns). The mass-moment members additionally need ``full_labels``.
    _compute_signed_tissue_frame(df, full_labels=full_labels, spacing=spacing,
                                 slices=slices)
    return df


def _compute_signed_tissue_frame(
    df: pd.DataFrame,
    full_labels: Optional[np.ndarray] = None,
    spacing: Optional[Tuple[float, ...]] = None,
    slices: Optional[list] = None,
) -> None:
    """SIGN the cell's principal-axis frame against the Wave C outward normal.
    Writes the six :data:`SIGNED_TISSUE_FRAME_COLUMNS` IN PLACE.

    Gated on the ``tissue_outward_*`` columns being present (Wave C ran): SKIPS
    (creates NO columns) otherwise so the default-off path is byte-identical. The
    three AXIS columns (:data:`_SIGNED_TISSUE_FRAME_AXIS_COLUMNS`) read only the
    per-cell principal axes; the three MASS-MOMENT columns
    (:data:`_SIGNED_TISSUE_FRAME_MASS_COLUMNS`) read each cell's voxels and are NaN
    when ``full_labels`` is None. Per-cell NaN when the outward normal is untrusted
    (low ``tissue_outward_confidence``) or the prerequisite axis/voxel set degenerate.

    See :func:`compute_tissue_frame_features` for the honest-formulation rationale
    (magnitude-only alignment; rotation-invariant tangential asymmetry; the deferred
    signed L/R rows).

    SIGN CAVEAT: the apical/basal sign = the Wave C outward normal = "toward the
    NEAREST tissue surface", which only equals true APICAL for a CURVED / CLOSED
    tissue or a basal-cell-free FOV. The SIGNED columns here
    (``cell_tilt_polarity`` / ``apical_vs_basal_elongation``) flip meaning when that
    assumption fails, so they carry ``sign_trusted=False`` in the manifest — treat as
    low-trust unless the tissue geometry guarantees the sign.
    """
    outward, trusted = _cell_outward_vectors(df)
    if outward is None:
        # Wave C did not run -> no sign basis. SKIP (leave columns absent).
        return

    n = len(df)
    for c in SIGNED_TISSUE_FRAME_COLUMNS:
        df[c] = (pd.Series(dtype=np.float64) if n == 0
                 else np.full(n, np.nan, dtype=np.float64))
    if n == 0:
        return

    eps = 1e-12

    major_cols = ["major_axis_z", "major_axis_y", "major_axis_x"]
    minor_cols = ["minor_axis_z", "minor_axis_y", "minor_axis_x"]
    have_major = all(c in df.columns for c in major_cols)
    have_minor = all(c in df.columns for c in minor_cols)
    majors = df[major_cols].values.astype(np.float64) if have_major else None
    minors = df[minor_cols].values.astype(np.float64) if have_minor else None

    ab_align = np.full(n, np.nan, dtype=np.float64)
    minor_tan = np.full(n, np.nan, dtype=np.float64)
    tan_aniso = np.full(n, np.nan, dtype=np.float64)
    tilt_pol = np.full(n, np.nan, dtype=np.float64)
    apbas_elong = np.full(n, np.nan, dtype=np.float64)
    tan_asym = np.full(n, np.nan, dtype=np.float64)

    def _unit(v):
        nrm = float(np.sqrt(np.sum(v * v)))
        return v / nrm if nrm > eps else None

    # --- mass-moment prerequisites (only when full_labels supplied) ----------
    objs = None
    label_ids = None
    sp_vec = None
    if full_labels is not None and "label" in df.columns:
        # S1 scaling hook: reuse the caller's precomputed bboxes (off the memmap)
        # instead of a redundant full-volume find_objects + int32 copy (the
        # ``.astype`` on a memmap forces a FULL RESIDENT copy -> OOM at scale).
        # None => the original byte-identical internal scan. The per-cell crops
        # below (``np.asarray(full_labels)[sl]``) are lazy on a memmap, so with
        # ``slices`` supplied the full volume is never materialised.
        if slices is None:
            from scipy.ndimage import find_objects
            objs = find_objects(np.asarray(full_labels).astype(np.int32))
        else:
            objs = slices
        label_ids = df["label"].values.astype(np.int64)
        sp_vec = (np.asarray(spacing, dtype=np.float64) if spacing is not None
                  else np.ones(3, dtype=np.float64))

    for idx in range(n):
        if not bool(trusted[idx]):
            continue
        nrm = _unit(outward[idx])
        if nrm is None:
            continue

        # --- AXIS-ONLY signed columns (need only the principal axes) ----------
        if majors is not None:
            u_maj = _unit(majors[idx])
            if u_maj is not None:
                # |cos| MAGNITUDE — the eigenvector sign is arbitrary, so the SIGN
                # would be orientation-arbitrary (polarity is in cell_tilt_polarity).
                ab_align[idx] = float(abs(np.dot(u_maj, nrm)))
        if minors is not None:
            u_min = _unit(minors[idx])
            if u_min is not None:
                # 1 - (u_minor·n)^2 = sin^2 of the minor-axis/normal angle. Squared
                # -> sign-invariant. 1 = the short axis lies in the tangent plane.
                c_mn = float(np.dot(u_min, nrm))
                minor_tan[idx] = float(np.clip(1.0 - c_mn * c_mn, 0.0, 1.0))

        # --- MASS-MOMENT signed columns (need the cell's voxels) --------------
        if objs is None:
            continue
        lid = int(label_ids[idx])
        if not (1 <= lid <= len(objs)) or objs[lid - 1] is None:
            continue
        sl = objs[lid - 1]
        cell_mask = np.asarray(full_labels)[sl] == lid
        if not cell_mask.any():
            continue
        zz, yy, xx = np.where(cell_mask)
        # physical voxel coords (offset by the crop origin, then × spacing) so the
        # projection onto the physical-frame outward normal is geometrically correct.
        origin = np.array([sl[0].start, sl[1].start, sl[2].start], dtype=np.float64)
        coords = (np.stack([zz, yy, xx], axis=1).astype(np.float64) + origin) * sp_vec
        cen = coords.mean(axis=0)
        rel = coords - cen
        s = rel @ nrm          # signed projection onto the outward (apical+) normal

        # cell_tilt_polarity: standardized 3rd moment (skew) of s. +ve = mass tail
        # leans apical, -ve basal. Sign from the SIGNED n, not an eigenvector.
        sd = float(s.std())
        if s.size >= 3 and sd > eps:
            tilt_pol[idx] = float(np.mean(((s - s.mean()) / sd) ** 3))

        # apical_vs_basal_elongation: (σ_ap - σ_ba)/(σ_ap + σ_ba) over the two halves
        # (std, NOT max -> robust to boundary outliers). NaN if a half is ~empty.
        ap = s[s > 0]
        ba = s[s < 0]
        if ap.size >= 2 and ba.size >= 2:
            sa = float(ap.std()); sb = float(ba.std())
            denom = sa + sb
            if denom > eps:
                apbas_elong[idx] = float((sa - sb) / denom)

        # tangent_plane_anisotropy: in-plane (λ1-λ2)/(λ1+λ2) of the cell's voxel
        # covariance projected onto the tangent plane (an orthonormal {t1,t2} basis
        # -> the 2×2 in-plane covariance, eigendecomposed). Sign-invariant.
        if rel.shape[0] >= 3:
            t1, t2 = _tangent_basis(nrm, eps)
            if t1 is not None and t2 is not None:
                p1 = rel @ t1
                p2 = rel @ t2
                C2 = np.cov(np.stack([p1, p2], axis=0))
                try:
                    ev = np.linalg.eigvalsh(C2)      # ascending, length 2
                except np.linalg.LinAlgError:
                    ev = None
                if ev is not None:
                    l_hi = float(max(ev[-1], 0.0)); l_lo = float(max(ev[0], 0.0))
                    if (l_hi + l_lo) > eps:
                        tan_aniso[idx] = float((l_hi - l_lo) / (l_hi + l_lo))

        # tangential_asymmetry_magnitude: the ROTATION-INVARIANT magnitude of the
        # SURFACE first-moment in the tangent plane (the in-plane mass first-moment
        # about the centroid is identically zero, so use the boundary voxels). A
        # one-sided tangential bulge -> large; tangentially even -> ~0. Normalised by
        # the surface in-plane RMS radius so it is dimensionless. Sign-FREE (no
        # canonical in-plane axis; we ship the magnitude, not a signed L/R).
        if rel.shape[0] >= 3:
            t1, t2 = _tangent_basis(nrm, eps)
            if t1 is not None and t2 is not None:
                surf = _boundary_voxel_mask(cell_mask)
                if surf is not None and surf.any():
                    szz, syy, sxx = np.where(surf)
                    scoords = ((np.stack([szz, syy, sxx], axis=1).astype(np.float64)
                                + origin) * sp_vec)
                    srel = scoords - cen          # about the MASS centroid (not surface mean)
                    sp1 = srel @ t1
                    sp2 = srel @ t2
                    mom = np.array([float(sp1.mean()), float(sp2.mean())])
                    rms = float(np.sqrt(np.mean(sp1 * sp1 + sp2 * sp2)))
                    if rms > eps:
                        tan_asym[idx] = float(np.linalg.norm(mom) / rms)

    df["apical_basal_axis_alignment"] = ab_align
    df["minor_axis_in_tangent_plane_fraction"] = minor_tan
    df["tangent_plane_anisotropy"] = tan_aniso
    df["cell_tilt_polarity"] = tilt_pol
    df["apical_vs_basal_elongation"] = apbas_elong
    df["tangential_asymmetry_magnitude"] = tan_asym


def _tangent_basis(normal: np.ndarray, eps: float = 1e-12):
    """An orthonormal {t1, t2} spanning the plane ⊥ ``normal`` (a unit ZYX vector).
    Returns ``(t1, t2)`` unit vectors, or ``(None, None)`` if degenerate."""
    n = np.asarray(normal, dtype=np.float64)
    nn = float(np.linalg.norm(n))
    if nn <= eps:
        return None, None
    n = n / nn
    # seed = the world axis least parallel to n (avoid a near-zero cross product).
    seed = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = seed - np.dot(seed, n) * n
    t1n = float(np.linalg.norm(t1))
    if t1n <= eps:
        return None, None
    t1 = t1 / t1n
    t2 = np.cross(n, t1)
    t2n = float(np.linalg.norm(t2))
    if t2n <= eps:
        return None, None
    return t1, t2 / t2n


def _boundary_voxel_mask(cell_mask: np.ndarray) -> Optional[np.ndarray]:
    """Surface (boundary) voxels of a binary cell crop: in-mask voxels with at least
    one 6-connected background neighbour (binary erosion XOR). ``None`` if degenerate."""
    from scipy.ndimage import binary_erosion
    m = np.asarray(cell_mask, dtype=bool)
    if not m.any():
        return None
    eroded = binary_erosion(m)          # default 6-connectivity structuring element
    surf = m & ~eroded
    return surf if surf.any() else m     # tiny/thin object -> the whole mask is surface


# --------------------------------------------------------------------------- #
# Wave A — adjacency-graph-derived families. All four ride on the M3
# ``df.attrs["_adjacency_list"]`` edge list (+ the per-cell centroid /
# major-axis / size columns) and run in the SAME global ``adjacency=True`` pass
# AFTER :func:`compute_neighbor_features` (so the graph exists) and BEFORE
# :func:`compute_named_composites` (which only reads existing columns):
#
#   C5  compute_graph_topology
#   C6  compute_axis_neighbor_persistence
#   C12 compute_local_comparative
#   C16 compute_graph_motifs
#
# Shared helper :func:`_adjacency_to_graph` builds a networkx-free adjacency
# dict + degree map ONCE from the edge list; each function then walks it per
# cell. Every column is NaN-guarded for degenerate cells (degree < the minimum
# the statistic needs). Default off -> output unchanged when ``adjacency=False``.
# --------------------------------------------------------------------------- #

def _adjacency_to_graph(adjacency: Optional[list]) -> Tuple[dict, dict]:
    """Build a networkx-free undirected adjacency dict + degree map from the M3
    edge list ``[{"label_a","label_b","contact_voxels"}, ...]``.

    Returns ``(adj, deg)`` where ``adj[label] = set(neighbour labels)`` and
    ``deg[label] = len(adj[label])``. Self-loops (``a == b``) are dropped; the
    edge list is de-duplicated by the set semantics. Empty/None -> ``({}, {})``.
    """
    adj: dict = {}
    if adjacency:
        for e in adjacency:
            a = int(e["label_a"]); b = int(e["label_b"])
            if a == b:
                continue
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
    deg = {k: len(v) for k, v in adj.items()}
    return adj, deg


# Wave A built the first 4 graph-topology columns; Mop-up M (the feature catalogue
# remainder) adds the 6 neighbour-degree-spread + ego-component + second-ring-ratio
# columns, completing the family. Module constant so the exact-set family test
# + the feature_stage registry stay in lockstep as the family grows.
GRAPH_TOPOLOGY_COLUMNS = (
    # Wave A (original 4).
    "ego_density", "ego_clustering_coefficient",
    "second_ring_count", "mean_neighbor_degree",
    # Mop-up M — neighbour-degree spread + ego component count + second-ring ratios.
    "neighbor_degree_std", "neighbor_degree_max",
    "ego_component_count",
    "first_to_second_ring_ratio", "second_ring_size_ratio",
    "second_ring_degree_mean",
)


def compute_graph_topology(
    df: pd.DataFrame,
    adjacency: Optional[list] = None,
) -> pd.DataFrame:
    """Neighbourhood graph topology per cell.

    Treats each cell + its neighbours as a local graph (not a bag of scalars)
    and derives ego-subgraph + second-ring statistics from the M3 adjacency
    graph. A GLOBAL pass — run ONCE inside the ``adjacency=True`` pass AFTER
    :func:`compute_neighbor_features` so ``df.attrs["_adjacency_list"]`` exists.

    Features added (10 — Wave A built the first 4; Mop-up M added the
    6 neighbour-degree-spread + ego-component + second-ring-ratio columns,
    completing the family):
        ``ego_density`` — realized edges AMONG the cell's neighbours divided by
            the possible edges ``C(k, 2)`` (k = degree) ∈ [0, 1]: 1 = the
            neighbours form a tight clique, 0 = they are mutually unconnected (a
            star). NaN for k < 2 (no pair of neighbours). Bounded.
        ``ego_clustering_coefficient`` — the cell's local clustering coefficient
            ``2 * e_among_neighbours / (k * (k - 1))`` ∈ [0, 1]. The standard
            graph-theoretic form of the same quantity as ``ego_density`` (it
            equals it by construction; kept as a separate named column per the
            the feature catalogue list). NaN for k < 2. Bounded.
        ``second_ring_count`` — number of DISTINCT neighbours-of-neighbours that
            are NOT in the cell's first ring and are NOT the cell itself: how far
            the local neighbourhood expands beyond direct contacts. 0 (not NaN)
            for an isolated/leaf cell. Unbounded count.
        ``mean_neighbor_degree`` — the mean DEGREE of the cell's neighbours
            (are the surrounding cells themselves densely connected?). NaN for an
            isolated cell (no neighbours to average). Unbounded.
        ``neighbor_degree_std`` (Mop-up M) — the SD of the cell's neighbours'
            degrees: 0 = the surrounding cells are all equally connected, high =
            a mix of hub + leaf neighbours (a disrupted neighbourhood). NaN for
            an isolated cell (no neighbours). Unbounded >= 0.
        ``neighbor_degree_max`` (Mop-up M) — the MAX degree among the cell's
            neighbours (is the cell next to a hub?). NaN for an isolated cell.
            Unbounded count.
        ``ego_component_count`` (Mop-up M) — the number of CONNECTED COMPONENTS
            of the induced subgraph on the cell's neighbour SET (edges among
            neighbours only): 1 = the neighbours form one connected ring; > 1 =
            the neighbourhood is split into disconnected clusters (a fold / gap
            cue). NaN for an isolated cell; 0 components is impossible for k>=1
            (each neighbour is at least its own singleton -> >= 1). Unbounded count.
        ``first_to_second_ring_ratio`` (Mop-up M) — ``k / max(second_ring, 1)``:
            how the neighbourhood expands from the first to the second ring
            (> 1 = the first ring is wider than the second; < 1 = the
            neighbourhood fans OUT). NaN for an isolated cell. Unbounded >= 0.
        ``second_ring_size_ratio`` (Mop-up M) — ``second_ring / max(k, 1)``: the
            reciprocal framing (the second ring relative to the first). NaN for
            an isolated cell. Unbounded >= 0.
        ``second_ring_degree_mean`` (Mop-up M) — the mean DEGREE of the cells in
            the second ring (how connected the surrounding shell is). NaN when
            the second ring is empty (a leaf / isolated cell). Unbounded.

    Args:
        df: assembled per-cell table (modified in-place). Needs ``label``; run
            after :func:`compute_neighbor_features` so the graph exists.
        adjacency: the M3 edge list; ``None`` → ``df.attrs["_adjacency_list"]``.

    Returns:
        The same DataFrame with the 10 new columns.
    """
    n = len(df)
    cols = GRAPH_TOPOLOGY_COLUMNS
    if n == 0:
        for c in cols:
            df[c] = pd.Series(dtype=np.float64)
        return df
    if "label" not in df.columns:
        for c in cols:
            df[c] = np.nan
        return df

    if adjacency is None:
        adjacency = df.attrs.get("_adjacency_list")
    adj, deg = _adjacency_to_graph(adjacency)

    label_ids = df["label"].values.astype(np.int64)

    ego_density = np.full(n, np.nan, dtype=np.float64)
    ego_clust = np.full(n, np.nan, dtype=np.float64)
    second_ring = np.zeros(n, dtype=np.float64)
    mean_nb_deg = np.full(n, np.nan, dtype=np.float64)
    # Mop-up M accumulators.
    nb_deg_std = np.full(n, np.nan, dtype=np.float64)
    nb_deg_max = np.full(n, np.nan, dtype=np.float64)
    ego_comp = np.full(n, np.nan, dtype=np.float64)
    first_to_second = np.full(n, np.nan, dtype=np.float64)
    second_size_ratio = np.full(n, np.nan, dtype=np.float64)
    second_deg_mean = np.full(n, np.nan, dtype=np.float64)

    for idx in range(n):
        lid = int(label_ids[idx])
        neigh = adj.get(lid, set())
        k = len(neigh)
        if k == 0:
            # isolated cell: no neighbours -> no second ring, no neighbour degree
            second_ring[idx] = 0.0
            continue

        # neighbour degree distribution (always defined for k >= 1).
        nb_degs_all = np.array([deg.get(nb, 0) for nb in neigh], dtype=np.float64)
        mean_nb_deg[idx] = float(nb_degs_all.mean())
        nb_deg_std[idx] = float(nb_degs_all.std())   # population SD; 0 for k==1
        nb_deg_max[idx] = float(nb_degs_all.max())

        # second ring = neighbours-of-neighbours minus first ring minus self.
        ring2: set = set()
        for nb in neigh:
            ring2 |= adj.get(nb, set())
        ring2 -= neigh
        ring2.discard(lid)
        second_ring[idx] = float(len(ring2))

        # first/second ring ratios (k >= 1 here). max(.,1) floors the denominator
        # so a leaf cell (empty second ring) yields a finite ratio rather than inf.
        first_to_second[idx] = float(k) / max(float(len(ring2)), 1.0)
        second_size_ratio[idx] = float(len(ring2)) / max(float(k), 1.0)
        if ring2:
            second_deg_mean[idx] = float(
                np.mean([deg.get(r, 0) for r in ring2]))

        # ego component count: connected components of the induced subgraph on the
        # neighbour SET (only edges BETWEEN two neighbours count). Union-find over
        # the neighbour list; each neighbour starts in its own component, merged
        # when it shares an edge with another neighbour. >= 1 for k >= 1.
        neigh_list = list(neigh)
        nidx = {nb: i for i, nb in enumerate(neigh_list)}
        parent = list(range(k))

        def _find(x, _parent=parent):
            while _parent[x] != x:
                _parent[x] = _parent[_parent[x]]
                x = _parent[x]
            return x

        e_among = 0
        for a in range(k):
            na = adj.get(neigh_list[a], set())
            for b in range(a + 1, k):
                if neigh_list[b] in na:
                    e_among += 1
                    ra, rb = _find(a), _find(b)
                    if ra != rb:
                        parent[ra] = rb
        ego_comp[idx] = float(len({_find(i) for i in range(k)}))

        # ego density / clustering coefficient need >= 2 neighbours (a pair).
        if k >= 2:
            possible = k * (k - 1) / 2.0
            ego_density[idx] = e_among / possible
            ego_clust[idx] = (2.0 * e_among) / (k * (k - 1))

    df["ego_density"] = ego_density
    df["ego_clustering_coefficient"] = ego_clust
    df["second_ring_count"] = second_ring
    df["mean_neighbor_degree"] = mean_nb_deg
    df["neighbor_degree_std"] = nb_deg_std
    df["neighbor_degree_max"] = nb_deg_max
    df["ego_component_count"] = ego_comp
    df["first_to_second_ring_ratio"] = first_to_second
    df["second_ring_size_ratio"] = second_size_ratio
    df["second_ring_degree_mean"] = second_deg_mean
    return df


# Wave A built the first 4; Wave J adds the 14
# per-contact axis-span stats + events + bin-turnover, completing the major-axis
# analogue of the image-Z neighbour-topology family. Module constant so the test
# + the feature_stage family registry stay exact-set in sync as the family grows.
AXIS_NEIGHBOR_PERSISTENCE_COLUMNS = (
    # Wave A (original 4).
    "axis_neighbor_turnover", "axis_neighbor_persistence_ratio",
    "axis_basal_neighbor_count", "axis_apical_neighbor_count",
    # Wave J — per-contact axis-span summary stats.
    "axis_neighbor_persistence_mean", "axis_neighbor_persistence_max",
    "axis_neighbor_persistence_cv",
    "axis_neighbor_birth_median", "axis_neighbor_death_median",
    "axis_neighbor_span_cv",
    # Wave J — birth/death exchange events along the axis.
    "axis_neighbor_events", "axis_neighbor_exchange_rate",
    # Wave J — disconnected-interval (re-entry) structure.
    "axis_neighbor_multispan_fraction", "axis_neighbor_span_count_mean",
    "axis_neighbor_reentry_count",
    # Wave J — coarse axial-bin counts + adjacent-bin turnover.
    "mid_axis_neighbor_count",
    "basal_to_mid_turnover", "mid_to_apical_turnover",
)

# Number of bins used to detect disconnected contact intervals along the
# normalised axis (re-entry / multispan). 12 ~= one bin per ~8 % of the cell
# length: fine enough to catch a real gap, coarse enough that a single noisy
# missing voxel does not fracture a continuous span.
_AXIS_SPAN_NBINS = 12


def compute_axis_neighbor_persistence(
    labels_3d: np.ndarray,
    df: pd.DataFrame,
    adjacency: Optional[list] = None,
    spacing: Optional[Tuple[float, ...]] = None,
    max_radius: int = 5,
    slices: Optional[list] = None,
) -> pd.DataFrame:
    """Axis-based neighbour persistence per cell (the feature catalogue; C6, primitive
    P11+P2). The cell's-OWN-major-axis analogue of the image-Z
    ``neighbor_z_turnover`` / ``neighbor_persistence_ratio`` (B5).

    For each cell we recover its contact voxels against every graph neighbour
    (the SAME graduated dilation as M3 / :func:`_contact_voxels_per_neighbor`,
    so the contact set is consistent with ``_adjacency_list``), project each
    contact voxel onto the cell's OWN principal major axis, and normalise to a
    coordinate ``t ∈ [0, 1]`` along the axis (0 = the basal end, 1 = the apical
    end, by the unsigned eigenvector convention — see below). Per neighbour ``j``
    we get ``birth_j = min(t)``, ``death_j = max(t)``, ``persistence_j =
    death_j - birth_j``, then summarise + bin the neighbours along the axis.

    A GLOBAL pass (a per-cell bbox crop can't see its neighbours) — run ONCE on
    the full label volume inside the ``adjacency=True`` pass right after
    :func:`compute_neighbor_features` (so the graph + the per-cell
    ``major_axis_*`` columns exist). ``adjacency`` defaults to
    ``df.attrs["_adjacency_list"]``.

    AXIS SIGN: an inertia eigenvector's sign is arbitrary, so the basal/apical
    ASSIGNMENT is only defined up to a flip. We orient the axis deterministically
    so its Z-component is non-negative (``+major`` if ``major_z >= 0`` else
    ``-major``), so "basal" = low-Z end and "apical" = high-Z end — matching the
    image-Z B5 convention as closely as an intrinsic axis allows. The COUNTS
    (``axis_basal_neighbor_count`` / ``axis_apical_neighbor_count``) are reported
    UNSIGNED as the two ends; turnover is symmetric so the sign does not affect it.

    Features added (18 — Wave A built the first 4; Wave J added the
    14 remaining per-contact axis-span + event + bin-turnover columns, completing
    the major-axis analogue of the image-Z neighbour-topology family):
        ``axis_neighbor_persistence_mean`` / ``_max`` / ``_cv`` (Wave J) — the
            mean / max / coefficient-of-variation of the per-neighbour persistence
            ``d_j − b_j`` (fraction of the cell length a neighbour stays in
            contact). Bounded persistence in [0, 1]; cv unbounded >= 0.
        ``axis_neighbor_birth_median`` / ``axis_neighbor_death_median`` (Wave J) —
            median of the per-neighbour birth (first contact ``t``) / death (last
            contact ``t``) along the oriented axis ∈ [0, 1].
        ``axis_neighbor_span_cv`` (Wave J) — cv of the per-neighbour span LENGTH
            (== ``axis_neighbor_persistence_cv``; both summarise the persistence
            spread, kept distinct to match the catalogue naming).
        ``axis_neighbor_events`` / ``axis_neighbor_exchange_rate`` (Wave J) — the
            total birth+death event count (``2 × Σ intervals``) and that count per
            unit physical axis length (exchange rate). Unbounded >= 0.
        ``axis_neighbor_multispan_fraction`` / ``axis_neighbor_span_count_mean`` /
            ``axis_neighbor_reentry_count`` (Wave J) — the fraction of neighbours
            whose contact splits into > 1 disconnected axial interval, the mean
            interval count per neighbour (>= 1), and the total re-entry count
            (``Σ intervals − 1``). Detect wrapping / broken / topologically-complex
            contacts vs simple single side spans.
        ``mid_axis_neighbor_count`` (Wave J) — number of distinct neighbours
            reaching the MIDDLE axial third (completes the basal/mid/apical bins).
        ``basal_to_mid_turnover`` / ``mid_to_apical_turnover`` (Wave J) — Jaccard
            turnover of the neighbour SET between the basal↔mid and mid↔apical
            thirds ∈ [0, 1] (whether exchange happens early vs late along the cell).
        ``axis_neighbor_turnover`` — neighbour-identity turnover between the
            low-``t`` (basal third) and high-``t`` (apical third) ends:
            ``1 - |basal ∩ apical| / |basal ∪ apical|`` ∈ [0, 1]. 1 = the
            neighbour set completely changes along the cell; 0 = the same
            neighbours span both ends. NaN when no neighbour reaches either end.
            Bounded.
        ``axis_neighbor_persistence_ratio`` — fraction of neighbours whose
            ``persistence_j > 0.7`` (span > 70 % of the cell length) ∈ [0, 1]:
            high = stable side-by-side packing along the axis. NaN with no
            neighbours. Bounded.
        ``axis_basal_neighbor_count`` — number of distinct neighbours contacting
            the BASAL third of the axis (low ``t``). Unbounded count.
        ``axis_apical_neighbor_count`` — number of distinct neighbours contacting
            the APICAL third of the axis (high ``t``). Unbounded count.

    Cells with no graph neighbours, a degenerate (zero) major axis, or no
    recoverable contact voxels get NaN turnover / persistence_ratio and 0 basal /
    apical counts.

    Args:
        labels_3d: the FULL 3D integer label volume (Z, Y, X). Background 0.
        df: assembled per-cell table (modified in-place). Needs ``label`` +
            ``major_axis_{z,y,x}`` (the projection axis) + ``centroid-0/1/2``
            (a presence guard that the table came from a real ``extract_features``
            run; the projection itself is origin-free — see the loop). Run after
            :func:`compute_neighbor_features`.
        adjacency: the M3 edge list; ``None`` → ``df.attrs["_adjacency_list"]``.
        spacing: voxel spacing ``(Z, Y, X)``. When given, the contact-voxel
            coordinates are scaled to PHYSICAL units before projecting onto the
            axis so the axis coordinate is in the same (physical) frame as the
            ``major_axis_*`` direction. ``None`` = voxel units.
        max_radius: graduated-dilation search radius in voxels — MUST match the
            M3 ``compute_neighbor_features`` ``max_radius``. Default 5.

    Returns:
        The same DataFrame with the 4 new columns.
    """
    from scipy.ndimage import find_objects

    n = len(df)
    cols = AXIS_NEIGHBOR_PERSISTENCE_COLUMNS
    if n == 0:
        for c in cols:
            df[c] = pd.Series(dtype=np.float64)
        return df

    cen_cols = ["centroid-0", "centroid-1", "centroid-2"]
    axis_cols = ["major_axis_z", "major_axis_y", "major_axis_x"]
    need = ["label", *cen_cols, *axis_cols]
    if not all(c in df.columns for c in need):
        for c in cols:
            df[c] = np.nan
        return df

    if adjacency is None:
        adjacency = df.attrs.get("_adjacency_list")
    nb_sets: dict = {}
    if adjacency:
        for e in adjacency:
            a = int(e["label_a"]); b = int(e["label_b"])
            if a == b:
                continue
            nb_sets.setdefault(a, set()).add(b)
            nb_sets.setdefault(b, set()).add(a)

    label_ids = df["label"].values.astype(np.int64)
    axes = df[axis_cols].values.astype(np.float64)
    axis_by_label = {int(label_ids[i]): axes[i] for i in range(n)}

    sp = np.asarray(spacing, dtype=np.float64) if spacing is not None else None
    eps = 1e-12
    persistence_tau = 0.7   # the feature catalogue example threshold

    # S1 scaling hook (see compute_neighbor_features): reuse the caller's bboxes
    # off the memmap; None => the original byte-identical internal scan.
    if slices is None:
        slices = find_objects(labels_3d.astype(np.int32))

    turnover = np.full(n, np.nan, dtype=np.float64)
    persist_ratio = np.full(n, np.nan, dtype=np.float64)
    basal_count = np.zeros(n, dtype=np.float64)
    apical_count = np.zeros(n, dtype=np.float64)
    # Wave J accumulators (NaN where the cell has no usable neighbour signal).
    persist_mean = np.full(n, np.nan, dtype=np.float64)
    persist_max = np.full(n, np.nan, dtype=np.float64)
    persist_cv = np.full(n, np.nan, dtype=np.float64)
    birth_median = np.full(n, np.nan, dtype=np.float64)
    death_median = np.full(n, np.nan, dtype=np.float64)
    span_cv = np.full(n, np.nan, dtype=np.float64)
    events = np.full(n, np.nan, dtype=np.float64)
    exchange_rate = np.full(n, np.nan, dtype=np.float64)
    multispan_frac = np.full(n, np.nan, dtype=np.float64)
    span_count_mean = np.full(n, np.nan, dtype=np.float64)
    reentry_count = np.full(n, np.nan, dtype=np.float64)
    mid_count = np.zeros(n, dtype=np.float64)
    basal_to_mid = np.full(n, np.nan, dtype=np.float64)
    mid_to_apical = np.full(n, np.nan, dtype=np.float64)

    nbins = _AXIS_SPAN_NBINS

    for idx in range(n):
        lid = int(label_ids[idx])
        neighbor_ids = nb_sets.get(lid, set())
        if not neighbor_ids:
            continue
        if lid < 1 or lid > len(slices):
            continue
        obj_slice = slices[lid - 1]
        if obj_slice is None:
            continue

        # Orient the cell's major axis so +Z (deterministic basal->apical sign).
        major = axis_by_label.get(lid)
        if major is None:
            continue
        major = np.asarray(major, dtype=np.float64)
        nrm = float(np.linalg.norm(major))
        if nrm <= eps:
            continue              # degenerate / round cell -> axis is noise
        major = major / nrm
        if major[0] < 0:          # Z-component sign convention
            major = -major

        # TODO S1.5b: collapse the 3x dilation — same contact-shell recompute as
        # compute_contact_geometry + the directional passes; cache per cell.
        contacts = _contact_voxels_per_neighbor(
            labels_3d, lid, obj_slice, neighbor_ids, max_radius=max_radius,
        )
        if not contacts:
            continue

        # Project every contact voxel onto the oriented major axis, collect
        # per-neighbour t-arrays, then normalise t to [0, 1] across the CELL'S OWN
        # AXIAL EXTENT (the cell length — NOT the contact-t envelope:
        # normalising to the contacts would stretch a one-end neighbour cluster
        # across the whole [0,1] axis and destroy the basal/apical asymmetry). The
        # axis is a UNIT vector and the normalisation below subtracts the cell's
        # ``cell_t_lo``, so the projection ORIGIN cancels out — we therefore project
        # the voxel coords directly (no centroid origin needed), which AVOIDS a frame
        # mismatch: the df ``centroid-*`` columns are PHYSICAL when spacing was used
        # at extract, whereas ``coords`` are VOXEL — subtracting them would mix
        # frames. The only thing that must match the (physical-frame) ``major`` axis
        # is the SCALE, so we scale the voxel coords by ``sp`` first when spacing is
        # set. The cell's t-range is read from its OWN mask voxels (same scaling).
        cell_vox = np.argwhere(labels_3d[obj_slice] == lid).astype(np.float64)
        if cell_vox.shape[0] == 0:
            continue
        cell_off = np.array([s.start for s in obj_slice], dtype=np.float64)
        cell_glob = cell_vox + cell_off
        cell_pts = cell_glob * sp if sp is not None else cell_glob
        cell_t = cell_pts @ major
        t_lo = float(cell_t.min()); t_hi = float(cell_t.max())

        # Capture the FULL per-neighbour projected-t array (Wave J needs the whole
        # set of contact-t values to detect disconnected spans / re-entry, not just
        # the min/max envelope Wave A used).
        nb_t_full: dict = {}
        for nb, coords in contacts.items():
            if coords.shape[0] == 0:
                continue
            pts = coords * sp if sp is not None else coords   # voxel -> physical
            t = pts @ major                         # scalar projection (M,)
            nb_t_full[int(nb)] = t
        if not nb_t_full:
            continue

        span = t_hi - t_lo
        if span <= eps:
            # all contacts at one axial point -> no persistence/turnover signal
            # but the two ends collapse: count every neighbour as both ends.
            ids = set(nb_t_full.keys())
            k0 = float(len(ids))
            basal_count[idx] = k0
            apical_count[idx] = k0
            mid_count[idx] = k0
            turnover[idx] = 0.0
            persist_ratio[idx] = 0.0
            # Degenerate single-point span: every neighbour persists for 0 length,
            # one single span each, no births/deaths beyond presence.
            persist_mean[idx] = 0.0
            persist_max[idx] = 0.0
            persist_cv[idx] = 0.0
            birth_median[idx] = 0.0
            death_median[idx] = 0.0
            span_cv[idx] = 0.0
            events[idx] = 2.0 * k0          # one birth + one death per neighbour
            exchange_rate[idx] = 0.0        # axis length ~0 -> no spatial rate
            multispan_frac[idx] = 0.0
            span_count_mean[idx] = 1.0
            reentry_count[idx] = 0.0
            basal_to_mid[idx] = 0.0
            mid_to_apical[idx] = 0.0
            continue

        # normalise each neighbour's contact-t to [0, 1] along the cell's contact
        # span, then derive per-neighbour birth/death/persistence + span structure.
        basal_set: set = set()
        mid_set: set = set()
        apical_set: set = set()
        n_persist = 0
        persistences: list = []
        births: list = []
        deaths: list = []
        span_counts: list = []           # # disconnected intervals per neighbour
        total_reentries = 0              # Σ (intervals − 1) over neighbours
        n_multispan = 0
        for nb, t_arr in nb_t_full.items():
            # normalised contact-t along the CELL's axial extent; contacts sit on
            # the dilated outer shell so they can land just past the cell ends ->
            # clamp to [0, 1] so birth/death/persistence stay the bounded fractions
            # the manifest advertises. Bin classification uses the clamped value too.
            tn = np.clip((t_arr - t_lo) / span, 0.0, 1.0)
            b = float(tn.min())
            d = float(tn.max())
            births.append(b)
            deaths.append(d)
            persistences.append(d - b)
            if (d - b) > persistence_tau:
                n_persist += 1
            # basal third = t < 1/3, mid third = 1/3..2/3, apical third = t > 2/3
            # (a neighbour that REACHES a third counts for that end).
            if b < (1.0 / 3.0):
                basal_set.add(nb)
            if d > (2.0 / 3.0):
                apical_set.add(nb)
            if (b < (2.0 / 3.0)) and (d > (1.0 / 3.0)):   # overlaps the mid third
                mid_set.add(nb)
            # disconnected-interval count via an occupancy histogram of this
            # neighbour's normalised-t values: count runs of occupied bins.
            occ = np.zeros(nbins, dtype=bool)
            bins = np.clip((tn * nbins).astype(np.int64), 0, nbins - 1)
            occ[bins] = True
            # a "run" = a maximal block of consecutive occupied bins; the number of
            # runs is the number of rising edges in the occupancy signal.
            runs = int(np.count_nonzero(occ & ~np.concatenate(([False], occ[:-1]))))
            runs = max(runs, 1)
            span_counts.append(runs)
            if runs >= 2:
                n_multispan += 1
            total_reentries += (runs - 1)

        k = len(nb_t_full)
        basal_count[idx] = float(len(basal_set))
        mid_count[idx] = float(len(mid_set))
        apical_count[idx] = float(len(apical_set))
        persist_ratio[idx] = n_persist / k

        # --- Wave J — per-contact axis-span summary stats --------------------
        pers = np.asarray(persistences, dtype=np.float64)
        bir = np.asarray(births, dtype=np.float64)
        dea = np.asarray(deaths, dtype=np.float64)
        pmean = float(pers.mean())
        persist_mean[idx] = pmean
        persist_max[idx] = float(pers.max())
        persist_cv[idx] = float(pers.std() / pmean) if pmean > eps else 0.0
        birth_median[idx] = float(np.median(bir))
        death_median[idx] = float(np.median(dea))
        # span_cv = cv of the per-neighbour persistence LENGTH (the "span" = d − b).
        span_cv[idx] = persist_cv[idx]

        # --- Wave J — exchange events along the axis -------------------------
        # one birth + one death per contiguous neighbour span; multispan neighbours
        # contribute an extra birth+death per re-entry interval.
        total_intervals = int(np.sum(span_counts))
        events[idx] = float(2 * total_intervals)
        # physical axis length sampled by the contacts (span is in the same
        # physical-projection units as `major` when spacing was applied).
        exchange_rate[idx] = float(2 * total_intervals) / max(span, eps)

        # --- Wave J — disconnected-interval (re-entry) structure -------------
        multispan_frac[idx] = float(n_multispan) / k
        span_count_mean[idx] = float(np.mean(span_counts))
        reentry_count[idx] = float(total_reentries)

        # --- Wave A turnover (basal vs apical ends) -------------------------
        union = basal_set | apical_set
        if union:
            inter = basal_set & apical_set
            turnover[idx] = 1.0 - len(inter) / len(union)

        # --- Wave J — adjacent-bin turnover (basal↔mid, mid↔apical) ---------
        bm_union = basal_set | mid_set
        if bm_union:
            basal_to_mid[idx] = 1.0 - len(basal_set & mid_set) / len(bm_union)
        ma_union = mid_set | apical_set
        if ma_union:
            mid_to_apical[idx] = 1.0 - len(mid_set & apical_set) / len(ma_union)

    df["axis_neighbor_turnover"] = turnover
    df["axis_neighbor_persistence_ratio"] = persist_ratio
    df["axis_basal_neighbor_count"] = basal_count
    df["axis_apical_neighbor_count"] = apical_count
    # Wave J columns.
    df["axis_neighbor_persistence_mean"] = persist_mean
    df["axis_neighbor_persistence_max"] = persist_max
    df["axis_neighbor_persistence_cv"] = persist_cv
    df["axis_neighbor_birth_median"] = birth_median
    df["axis_neighbor_death_median"] = death_median
    df["axis_neighbor_span_cv"] = span_cv
    df["axis_neighbor_events"] = events
    df["axis_neighbor_exchange_rate"] = exchange_rate
    df["axis_neighbor_multispan_fraction"] = multispan_frac
    df["axis_neighbor_span_count_mean"] = span_count_mean
    df["axis_neighbor_reentry_count"] = reentry_count
    df["mid_axis_neighbor_count"] = mid_count
    df["basal_to_mid_turnover"] = basal_to_mid
    df["mid_to_apical_turnover"] = mid_to_apical
    return df


# Curated base-feature set for the local-comparative templates (the feature catalogue note:
# "Should be selective ... not blindly for every column"). Eight size/shape columns
# that span volume / surface / sphericity / elongation / triaxial / curvature; each
# gets the FIVE neighbour-normalised variants below. Columns absent from a given
# table are skipped. Mop-up M extended the original 5 by 3 (axis_minor_length,
# solidity, surface_area_to_volume — broaden the size/shape coverage per the prompt).
LOCAL_COMPARATIVE_FEATURES = (
    "area", "surface_area", "sphericity", "aspect_ratio", "axis_major_length",
    # Mop-up M — broaden the base set by 3 more size/shape scalars.
    "axis_minor_length", "solidity", "extent",
)

# The five templated neighbour-normalised suffixes a base feature gets. Wave A
# emitted the first two; Mop-up M added the median-ratio,
# difference, and contact-weighted-mean variants.
_LOCAL_COMPARATIVE_SUFFIXES = (
    "_vs_neighbor_mean_ratio", "_local_zscore",
    "_vs_neighbor_median_ratio", "_minus_neighbor_mean",
    "_vs_contact_weighted_neighbor_mean",
)


def compute_local_comparative(
    df: pd.DataFrame,
    features: Optional[Tuple[str, ...]] = None,
    adjacency: Optional[list] = None,
) -> pd.DataFrame:
    """Local (neighbourhood-normalised) comparative features (the feature catalogue;
    C12, primitive P2). Recasts a curated set of per-cell size/shape columns
    relative to the cell's GRAPH neighbours, decoupling locally-real anomalies
    from global drift ("many QC mistakes are locally weird before globally weird").

    A GLOBAL pass — run inside the ``adjacency=True`` pass after
    :func:`compute_neighbor_features` so the graph exists; reads only existing
    columns (degrades per-feature when a base column is absent).

    For each base feature ``f`` in ``features`` that is present in ``df``, adds
    FIVE templated columns (Wave A built the first two; Mop-up M added the
    last three):
        ``<f>_vs_neighbor_mean_ratio`` — ``f_i / max(mean_{j∈N(i)} f_j, eps)``:
            >1 = the cell is larger/rougher/rounder than its neighbours' average,
            <1 = smaller. NaN for an isolated cell (no neighbours). Unbounded >=0.
        ``<f>_local_zscore`` — robust z of ``f_i`` against its neighbours:
            ``(f_i - median_j f_j) / (1.4826 * MAD_j + eps)`` — how many robust
            SDs the cell sits from its local neighbourhood (sign kept: + = above
            its neighbours). NaN for < 2 neighbours (MAD undefined). Unbounded.
        ``<f>_vs_neighbor_median_ratio`` (Mop-up M) — ``f_i / max(median_j f_j,
            eps)``: the median-robust analogue of the mean ratio (less sensitive to
            a single large/small neighbour). NaN for an isolated cell. Unbounded >=0.
        ``<f>_minus_neighbor_mean`` (Mop-up M) — ``f_i − mean_j f_j``: the signed
            DIFFERENCE from the neighbour average (kept in the feature's own units,
            so absolute deviations are comparable across cells of similar local
            scale). NaN for an isolated cell. Unbounded (signed).
        ``<f>_vs_contact_weighted_neighbor_mean`` (Mop-up M) — ``f_i /
            max(Σ_j w_j f_j / Σ_j w_j, eps)``, the neighbour mean WEIGHTED by the
            shared contact area ``w_j = contact_voxels_{ij}`` (the feature catalogue: emphasise
            the neighbours the cell actually shares the most interface with). Falls
            back to the unweighted mean ratio when no edge carries a contact weight.
            NaN for an isolated cell. Unbounded >= 0.

    Only NEIGHBOUR values (not the cell itself) enter the mean / median / MAD /
    difference, so the comparison is genuinely "this cell vs its surround".
    Neighbour values that are NaN are dropped before aggregating.

    Args:
        df: assembled per-cell table (modified in-place). Needs ``label`` + the
            base feature columns. Run after :func:`compute_neighbor_features`.
        features: the base columns to template; ``None`` →
            :data:`LOCAL_COMPARATIVE_FEATURES`. Columns absent from ``df`` are
            silently skipped (their five output columns are NOT created).
        adjacency: the M3 edge list; ``None`` → ``df.attrs["_adjacency_list"]``.
            Its ``contact_voxels`` weights drive the contact-weighted variant.

    Returns:
        The same DataFrame with up to ``5 * len(features present)`` new columns.
    """
    feats = tuple(features) if features is not None else LOCAL_COMPARATIVE_FEATURES
    n = len(df)
    present = [f for f in feats if f in df.columns] if "label" in df.columns else []

    if n == 0 or not present:
        # Emit the schema for the PRESENT base features even on an empty table so
        # downstream readers see a stable column set (none when no base present).
        for f in present:
            for suf in _LOCAL_COMPARATIVE_SUFFIXES:
                df[f"{f}{suf}"] = pd.Series(dtype=np.float64)
        return df

    if adjacency is None:
        adjacency = df.attrs.get("_adjacency_list")
    adj, _ = _adjacency_to_graph(adjacency)
    # per-cell {neighbour: contact_voxels weight} for the contact-weighted variant.
    weights: dict = {}
    if adjacency:
        for e in adjacency:
            a = int(e["label_a"]); b = int(e["label_b"])
            if a == b:
                continue
            w = float(e.get("contact_voxels", 0.0) or 0.0)
            weights.setdefault(a, {})[b] = w
            weights.setdefault(b, {})[a] = w

    label_ids = df["label"].values.astype(np.int64)
    eps = 1e-12
    val_by_label = {
        f: {int(label_ids[i]): float(df[f].values[i]) for i in range(n)}
        for f in present
    }

    for f in present:
        vmap = val_by_label[f]
        ratio = np.full(n, np.nan, dtype=np.float64)
        zscore = np.full(n, np.nan, dtype=np.float64)
        median_ratio = np.full(n, np.nan, dtype=np.float64)
        minus_mean = np.full(n, np.nan, dtype=np.float64)
        cw_ratio = np.full(n, np.nan, dtype=np.float64)
        for idx in range(n):
            lid = int(label_ids[idx])
            mine = vmap.get(lid, np.nan)
            if not np.isfinite(mine):
                continue
            nbs = list(adj.get(lid, set()))
            nb_vals_raw = np.array([vmap.get(nb, np.nan) for nb in nbs],
                                   dtype=np.float64)
            finite_mask = np.isfinite(nb_vals_raw)
            nb_vals = nb_vals_raw[finite_mask]
            if nb_vals.size >= 1:
                mean_nb = float(nb_vals.mean())
                ratio[idx] = mine / max(mean_nb, eps)
                med_nb = float(np.median(nb_vals))
                median_ratio[idx] = mine / max(med_nb, eps)
                minus_mean[idx] = mine - mean_nb
                # contact-weighted neighbour mean (weights from the edge list).
                wmap = weights.get(lid, {})
                w_arr = np.array([wmap.get(nb, 0.0) for nb in nbs],
                                 dtype=np.float64)[finite_mask]
                wsum = float(w_arr.sum())
                if wsum > eps:
                    cw_mean = float((w_arr * nb_vals).sum() / wsum)
                    cw_ratio[idx] = mine / max(cw_mean, eps)
                else:
                    # no contact weights -> degrade to the unweighted mean ratio.
                    cw_ratio[idx] = ratio[idx]
            if nb_vals.size >= 2:
                med = float(np.median(nb_vals))
                mad = float(np.median(np.abs(nb_vals - med)))
                scale = 1.4826 * mad
                if scale > eps:
                    zscore[idx] = (mine - med) / scale
                else:
                    # neighbours identical -> 0 spread; the cell is at/away from
                    # them by an unmeasurable amount. 0 when it equals them, else
                    # leave NaN (an infinite z is not informative).
                    zscore[idx] = 0.0 if abs(mine - med) <= eps else np.nan
        df[f"{f}_vs_neighbor_mean_ratio"] = ratio
        df[f"{f}_local_zscore"] = zscore
        df[f"{f}_vs_neighbor_median_ratio"] = median_ratio
        df[f"{f}_minus_neighbor_mean"] = minus_mean
        df[f"{f}_vs_contact_weighted_neighbor_mean"] = cw_ratio
    return df


# P1b — per-channel NEIGHBOUR-relative expression context ( "vs neighbor" row).
# The CHANNEL sibling of compute_local_comparative: a cell's channel mean relative to
# its GRAPH neighbours' channel means — local expression context (is this cell a
# bright outlier in a dim neighbourhood?). The suffix is appended to the channel's
# intensity-mean column (``<prefix>_intensity_mean`` -> ``<prefix>_vs_neighbor_mean_
# ratio``). Recovered dynamically at manifest time (the prefix is a runtime choice).
_CHANNEL_NEIGHBOR_CONTEXT_SUFFIX = "_vs_neighbor_mean_ratio"


def compute_channel_neighbor_context(
    df: pd.DataFrame,
    channel_prefixes: Optional[Tuple[str, ...]] = None,
    adjacency: Optional[list] = None,
) -> pd.DataFrame:
    """Per-channel neighbour-relative expression context (the feature catalogue "vs neighbor";
    P1b, primitive P2). For each sampled channel, recasts the cell's mean intensity
    relative to its GRAPH neighbours' mean intensities — the local-expression-
    context row the SHAPE templates do NOT cover (those are shape features).

    A GLOBAL pass — run inside the ``adjacency=True`` pass after
    :func:`compute_neighbor_features` (so the graph exists) AND after the per-cell
    channel means are assembled (the ``<prefix>_intensity_mean`` columns). Reads only
    existing columns; degrades per-channel when a channel's mean column is absent.

    For each channel ``prefix`` whose ``<prefix>_intensity_mean`` column is present,
    adds ONE column::

        <prefix>_vs_neighbor_mean_ratio = mean_i / max(median_{j∈N(i)} mean_j, eps)

    — the cell's channel mean over the MEDIAN of its neighbours' channel means
    (median, not mean, for robustness to a single bright neighbour). > 1 = the cell
    expresses MORE of the marker than its surround, < 1 = less. NaN for an isolated
    cell (no neighbours) or a NaN cell mean. Unbounded >= 0. Only NEIGHBOUR values
    enter the median, so it is genuinely "this cell vs its surround".

    Args:
        df: assembled per-cell table (modified in-place). Needs ``label`` + the
            ``<prefix>_intensity_mean`` channel columns. Run after
            :func:`compute_neighbor_features`.
        channel_prefixes: the channel prefixes to template; ``None`` -> auto-detect
            every ``<prefix>_intensity_mean`` column in ``df`` (strip the suffix).
            A prefix with no mean column is silently skipped.
        adjacency: the M3 edge list; ``None`` -> ``df.attrs["_adjacency_list"]``.

    Returns:
        The same DataFrame with up to ``len(prefixes present)`` new columns. No
        channel means present -> no columns added (default-off byte-identical).
    """
    n = len(df)
    if n == 0 or "label" not in df.columns:
        return df

    # auto-detect channel prefixes from the present ``<prefix>_intensity_mean`` cols.
    if channel_prefixes is None:
        suffix = "_intensity_mean"
        prefixes = tuple(
            c[: -len(suffix)] for c in df.columns if c.endswith(suffix))
    else:
        prefixes = tuple(str(p) for p in channel_prefixes)
    present = [p for p in prefixes if f"{p}_intensity_mean" in df.columns]
    if not present:
        return df

    if adjacency is None:
        adjacency = df.attrs.get("_adjacency_list")
    adj, _ = _adjacency_to_graph(adjacency)

    label_ids = df["label"].values.astype(np.int64)
    eps = 1e-12
    for p in present:
        col = f"{p}_intensity_mean"
        vmap = {int(label_ids[i]): float(df[col].values[i]) for i in range(n)}
        ratio = np.full(n, np.nan, dtype=np.float64)
        for idx in range(n):
            lid = int(label_ids[idx])
            mine = vmap.get(lid, np.nan)
            if not np.isfinite(mine):
                continue
            nbs = list(adj.get(lid, set()))
            nb_raw = np.array([vmap.get(nb, np.nan) for nb in nbs],
                              dtype=np.float64)
            nb_vals = nb_raw[np.isfinite(nb_raw)]
            if nb_vals.size >= 1:
                med_nb = float(np.median(nb_vals))
                ratio[idx] = mine / max(med_nb, eps)
        df[f"{p}{_CHANNEL_NEIGHBOR_CONTEXT_SUFFIX}"] = ratio
    return df


# Wave A built the first 4 motif columns; Mop-up M adds the higher-
# order motifs (squares + cycle fraction), the node-attribute homophily pair, and
# the "research" ego-graph CENTRALITY trio (betweenness / pagerank /
# eigenvector) computed on the FIRST-RING induced subgraph (cheap, networkx-free).
# Module constant so the exact-set family test + feature_stage registry stay in
# lockstep. Centrality node attributes for homophily are read from the df columns
# (`area` + a shape column) when present; absent -> the two homophily columns NaN.
GRAPH_MOTIF_COLUMNS = (
    # Wave A (original 4).
    "ego_triangle_count", "neighbor_count_cv",
    "degree_centrality", "neighbor_degree_assortativity",
    # Mop-up M — higher-order motifs.
    "ego_square_count", "ego_cycle_fraction",
    # Mop-up M — node-attribute homophily (size + shape similarity of neighbours).
    "neighbor_homophily_area", "neighbor_homophily_shape",
    # Mop-up M — ego-graph centrality (research tier; computed locally).
    "ego_betweenness", "ego_pagerank", "ego_eigenvector_centrality",
)

# The base shape column used for ``neighbor_homophily_shape`` (first present wins).
# Sphericity is the most universally-present shape scalar; aspect_ratio is the
# fallback. Both are intrinsic, comparable across cells.
_HOMOPHILY_SHAPE_FEATURES = ("sphericity", "aspect_ratio")


def _ego_centrality(lid: int, neigh: set, adj: dict):
    """Betweenness / PageRank / eigenvector centrality of the CELL within its OWN
    ego graph (the node ``lid`` + its first-ring neighbours + the edges among
    them), computed without networkx (Mop-up M / the feature catalogue research tier).

    The ego graph is tiny (a cell + its <= ~tens of neighbours), so the exact
    O(V·E) Brandes betweenness + power-iteration PageRank / eigenvector are cheap.
    Returns ``(betweenness, pagerank, eigenvector)`` for the ego node, all in
    [0, 1] (normalised). ``(nan, nan, nan)`` for an isolated cell (no ego graph).
    """
    eps = 1e-12
    nodes = [lid, *sorted(neigh)]
    nv = len(nodes)
    if nv < 2:
        return np.nan, np.nan, np.nan
    nodeset = set(nodes)
    nidx = {nd: i for i, nd in enumerate(nodes)}
    # ego adjacency: keep only edges whose BOTH endpoints are in the ego graph.
    ego_adj = {nd: [m for m in adj.get(nd, set()) if m in nodeset] for nd in nodes}

    # --- Brandes betweenness (unweighted, undirected) -----------------------
    bet = np.zeros(nv, dtype=np.float64)
    for s in range(nv):
        s_nd = nodes[s]
        stack = []
        pred = {nd: [] for nd in nodes}
        sigma = {nd: 0.0 for nd in nodes}
        sigma[s_nd] = 1.0
        dist = {nd: -1 for nd in nodes}
        dist[s_nd] = 0
        queue = [s_nd]
        while queue:
            v = queue.pop(0)
            stack.append(v)
            for w in ego_adj[v]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)
        delta = {nd: 0.0 for nd in nodes}
        while stack:
            w = stack.pop()
            for v in pred[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s_nd:
                bet[nidx[w]] += delta[w]
    # undirected -> each shortest path counted twice; normalise by (nv-1)(nv-2).
    norm = (nv - 1) * (nv - 2)
    bet_node = (bet[nidx[lid]] / norm) if norm > 0 else 0.0

    # --- PageRank (power iteration, damping 0.85) ----------------------------
    damp = 0.85
    pr = np.full(nv, 1.0 / nv, dtype=np.float64)
    out_deg = np.array([max(len(ego_adj[nd]), 0) for nd in nodes], dtype=np.float64)
    for _ in range(100):
        new_pr = np.full(nv, (1.0 - damp) / nv, dtype=np.float64)
        dangling = 0.0
        for i, nd in enumerate(nodes):
            if out_deg[i] <= 0:
                dangling += pr[i]
                continue
            share = damp * pr[i] / out_deg[i]
            for m in ego_adj[nd]:
                new_pr[nidx[m]] += share
        # redistribute dangling mass uniformly.
        new_pr += damp * dangling / nv
        if np.abs(new_pr - pr).sum() < 1e-9:
            pr = new_pr
            break
        pr = new_pr
    s = pr.sum()
    pr_node = float(pr[nidx[lid]] / s) if s > eps else np.nan

    # --- eigenvector centrality (power iteration on the adjacency) -----------
    ev = np.ones(nv, dtype=np.float64)
    for _ in range(100):
        new_ev = np.zeros(nv, dtype=np.float64)
        for i, nd in enumerate(nodes):
            for m in ego_adj[nd]:
                new_ev[i] += ev[nidx[m]]
        nrm = float(np.linalg.norm(new_ev))
        if nrm <= eps:
            break
        new_ev = new_ev / nrm
        if np.abs(new_ev - ev).sum() < 1e-9:
            ev = new_ev
            break
        ev = new_ev
    ev_sum = ev.sum()
    ev_node = float(ev[nidx[lid]] / ev_sum) if ev_sum > eps else np.nan

    return float(bet_node), pr_node, ev_node


def compute_graph_motifs(
    df: pd.DataFrame,
    adjacency: Optional[list] = None,
    shape_feature: Optional[str] = None,
) -> pd.DataFrame:
    """Graph motif / local-network features per cell (the feature catalogue; C16, primitive
    P2). Mines small graph patterns + degree relationships around each cell that
    scalar neighbour counts miss (abnormal packing motifs at folds / merges).

    A GLOBAL pass — run inside the ``adjacency=True`` pass after
    :func:`compute_neighbor_features` so ``df.attrs["_adjacency_list"]`` exists.

    Features added (11 — Wave A built the first 4; Mop-up M added the
    higher-order motifs + homophily + the ego-graph centrality trio, completing
    the family):
        ``ego_triangle_count`` — number of TRIANGLES through the cell = edges
            among its neighbours (each neighbour-pair edge closes a triangle with
            the cell). 0 (not NaN) for any cell (a leaf/isolated cell has none).
            Unbounded count.
        ``neighbor_count_cv`` — coefficient of variation of the cell's
            neighbours' DEGREES (``std / max(mean, eps)``): 0 = the neighbours are
            all equally connected (regular packing), high = a mix of high- and
            low-degree neighbours (a disrupted motif). NaN for < 2 neighbours
            (CV undefined). Unbounded >=0.
        ``degree_centrality`` — the cell's degree divided by the GLOBAL maximum
            degree over all cells ∈ [0, 1]: 1 = the most-connected cell in the
            volume. 0 for an isolated cell. Bounded.
        ``neighbor_degree_assortativity`` — the per-cell assortativity proxy:
            the Pearson correlation between the cell's degree (a constant for the
            cell) and its neighbours' degrees is undefined, so we use the SIGNED
            deviation form ``mean_{j∈N(i)} (deg_i - <deg>)(deg_j - <deg>) /
            (var(deg) + eps)`` — positive when the cell + its neighbours are on
            the SAME side of the global mean degree (assortative: high-degree
            cells neighbour high-degree cells), negative for disassortative.
            ``<deg>`` / ``var(deg)`` are over ALL cells. NaN for an isolated cell.
            Unbounded (≈ [-1, 1] in practice).
        ``ego_square_count`` (Mop-up M) — number of 4-cycles (SQUARES) through the
            cell: pairs of neighbours ``(a, b)`` that share a COMMON neighbour
            other than the cell itself (the cell–a–x–b–cell quadrilateral). A
            regular-lattice packing cue. 0 (not NaN) for any cell. Unbounded count.
        ``ego_cycle_fraction`` (Mop-up M) — fraction of the cell's neighbour PAIRS
            that participate in a triangle OR a square through the cell ∈ [0, 1]:
            high = the neighbourhood is richly cross-linked (closed packing),
            low = a star/tree-like neighbourhood. NaN for < 2 neighbours. Bounded.
        ``neighbor_homophily_area`` (Mop-up M) — ``1 / (1 + cv)`` of the cell's
            neighbours' ``area`` values ∈ (0, 1]: 1 = the neighbours are all the
            same size as each other (homophilous), → 0 = a mix of large + small
            neighbours (heterophilous, a merge/fragment cue). NaN for < 2
            neighbours or when ``area`` is absent. Bounded.
        ``neighbor_homophily_shape`` (Mop-up M) — same ``1 / (1 + cv)`` over the
            neighbours' SHAPE scalar (``sphericity``, else ``aspect_ratio``):
            shape similarity of the surrounding cells. NaN for < 2 neighbours or
            when no shape column is present. Bounded.
        ``ego_betweenness`` (Mop-up M; research tier) — the cell's
            betweenness centrality WITHIN its own ego graph (cell + first-ring
            neighbours + their mutual edges), normalised to [0, 1]: high = the
            cell is a bridge whose neighbours are otherwise poorly connected
            (a fold / boundary cue). NaN for an isolated cell. Bounded.
        ``ego_pagerank`` (Mop-up M; research tier) — the cell's PageRank
            within its ego graph (damping 0.85), normalised to sum-1 ∈ [0, 1].
            NaN for an isolated cell. Bounded.
        ``ego_eigenvector_centrality`` (Mop-up M; research tier) — the
            cell's eigenvector centrality within its ego graph (power iteration),
            normalised to sum-1 ∈ [0, 1]. NaN for an isolated cell. Bounded.

    Args:
        df: assembled per-cell table (modified in-place). Needs ``label``; run
            after :func:`compute_neighbor_features`. ``area`` enables
            ``neighbor_homophily_area``; a shape column (``shape_feature`` or the
            first present of :data:`_HOMOPHILY_SHAPE_FEATURES`) enables
            ``neighbor_homophily_shape`` (else those two columns are NaN).
        adjacency: the M3 edge list; ``None`` → ``df.attrs["_adjacency_list"]``.
        shape_feature: the df column to use for shape homophily; ``None`` → the
            first present of :data:`_HOMOPHILY_SHAPE_FEATURES`.

    Returns:
        The same DataFrame with the 11 new columns.
    """
    n = len(df)
    cols = GRAPH_MOTIF_COLUMNS
    if n == 0:
        for c in cols:
            df[c] = pd.Series(dtype=np.float64)
        return df
    if "label" not in df.columns:
        for c in cols:
            df[c] = np.nan
        return df

    if adjacency is None:
        adjacency = df.attrs.get("_adjacency_list")
    adj, deg = _adjacency_to_graph(adjacency)

    label_ids = df["label"].values.astype(np.int64)
    eps = 1e-12

    # node-attribute maps for homophily (size + shape). Absent column -> None.
    area_by_label = (
        {int(label_ids[i]): float(df["area"].values[i]) for i in range(n)}
        if "area" in df.columns else None)
    shape_col = shape_feature
    if shape_col is None:
        for cand in _HOMOPHILY_SHAPE_FEATURES:
            if cand in df.columns:
                shape_col = cand
                break
    shape_by_label = (
        {int(label_ids[i]): float(df[shape_col].values[i]) for i in range(n)}
        if (shape_col is not None and shape_col in df.columns) else None)

    # global degree statistics (over the cells PRESENT in the table; a cell with
    # no edges has degree 0). Used by degree_centrality + assortativity.
    deg_vec = np.array([deg.get(int(label_ids[i]), 0) for i in range(n)],
                       dtype=np.float64)
    max_deg = float(deg_vec.max()) if n else 0.0
    mean_deg = float(deg_vec.mean()) if n else 0.0
    var_deg = float(deg_vec.var()) if n else 0.0

    triangle_count = np.zeros(n, dtype=np.float64)
    nb_cv = np.full(n, np.nan, dtype=np.float64)
    deg_cent = np.zeros(n, dtype=np.float64)
    assort = np.full(n, np.nan, dtype=np.float64)
    # Mop-up M accumulators.
    square_count = np.zeros(n, dtype=np.float64)
    cycle_frac = np.full(n, np.nan, dtype=np.float64)
    homo_area = np.full(n, np.nan, dtype=np.float64)
    homo_shape = np.full(n, np.nan, dtype=np.float64)
    bet_c = np.full(n, np.nan, dtype=np.float64)
    pr_c = np.full(n, np.nan, dtype=np.float64)
    ev_c = np.full(n, np.nan, dtype=np.float64)

    def _homophily(vals):
        """1 / (1 + cv) of a value array (>= 2 finite values), else NaN."""
        v = np.asarray(vals, dtype=np.float64)
        v = v[np.isfinite(v)]
        if v.size < 2:
            return np.nan
        m = float(v.mean())
        if abs(m) <= eps:
            return np.nan
        cv = float(v.std() / abs(m))
        return 1.0 / (1.0 + cv)

    for idx in range(n):
        lid = int(label_ids[idx])
        neigh = adj.get(lid, set())
        k = len(neigh)
        d_i = float(deg.get(lid, 0))

        # degree centrality (defined for all; 0 when the volume has no edges).
        deg_cent[idx] = (d_i / max_deg) if max_deg > 0 else 0.0

        if k == 0:
            continue

        # triangles + squares through the cell over the neighbour pairs.
        neigh_list = list(neigh)
        e_among = 0
        n_square = 0
        n_pairs_in_cycle = 0
        for a in range(len(neigh_list)):
            na = adj.get(neigh_list[a], set())
            for b in range(a + 1, len(neigh_list)):
                nb_node = neigh_list[b]
                in_tri = nb_node in na
                if in_tri:
                    e_among += 1
                # square: a and b share a common neighbour OTHER than the cell.
                common = (na & adj.get(nb_node, set())) - {lid}
                in_sq = len(common) > 0
                if in_sq:
                    n_square += 1
                if in_tri or in_sq:
                    n_pairs_in_cycle += 1
        triangle_count[idx] = float(e_among)
        square_count[idx] = float(n_square)
        if k >= 2:
            total_pairs = k * (k - 1) / 2.0
            cycle_frac[idx] = n_pairs_in_cycle / total_pairs

        nb_degs = np.array([deg.get(nb, 0) for nb in neigh], dtype=np.float64)
        if k >= 2:
            m = float(nb_degs.mean())
            nb_cv[idx] = float(nb_degs.std() / max(m, eps))

        # per-cell assortativity proxy (signed co-deviation, normalised by the
        # global degree variance). NaN if the global degrees are all identical.
        if var_deg > eps:
            co = np.mean((d_i - mean_deg) * (nb_degs - mean_deg))
            assort[idx] = float(co / (var_deg + eps))

        # node-attribute homophily of the neighbour SET (>= 2 neighbours).
        if k >= 2:
            if area_by_label is not None:
                homo_area[idx] = _homophily(
                    [area_by_label.get(nb, np.nan) for nb in neigh])
            if shape_by_label is not None:
                homo_shape[idx] = _homophily(
                    [shape_by_label.get(nb, np.nan) for nb in neigh])

        # ego-graph centrality trio (research tier; on the first-ring subgraph).
        b_, p_, e_ = _ego_centrality(lid, neigh, adj)
        bet_c[idx] = b_
        pr_c[idx] = p_
        ev_c[idx] = e_

    df["ego_triangle_count"] = triangle_count
    df["neighbor_count_cv"] = nb_cv
    df["degree_centrality"] = deg_cent
    df["neighbor_degree_assortativity"] = assort
    df["ego_square_count"] = square_count
    df["ego_cycle_fraction"] = cycle_frac
    df["neighbor_homophily_area"] = homo_area
    df["neighbor_homophily_shape"] = homo_shape
    df["ego_betweenness"] = bet_c
    df["ego_pagerank"] = pr_c
    df["ego_eigenvector_centrality"] = ev_c
    return df


# Wave C — tissue-depth + spatial-context columns. The "in/out" tissue-EDT
# family (C-W) + the spatial-context family + the per-cell OUTWARD normal
# (the sign basis later waves use for the apical/basal frame). Carried as a
# module constant so the FeatureStage family registry + the byte-identical
# default-off contract can enumerate them.
TISSUE_DEPTH_COLUMNS = (
    # tissue-EDT "in/out" field, per cell
    "distance_to_tissue_surface", "tissue_depth_mean", "tissue_depth_max",
    "normalized_tissue_depth", "at_fov_border",
    # spatial context (within the tissue bbox)
    "centroid_z_relative", "centroid_y_relative", "centroid_x_relative",
    "distance_to_tissue_centroid", "radial_tissue_position",
    # Wave I remainder — min distance from the centroid to a FOV face
    "distance_to_boundary_face_min",
    # Mop-up O remainder — the named per-cell LOCAL tissue depth (depth in
    # cell-radius units, distinct from the globally-normalized normalized_tissue_depth)
    "local_tissue_depth",
    # per-cell OUTWARD normal (the sign basis)
    "tissue_outward_z", "tissue_outward_y", "tissue_outward_x",
    "tissue_outward_confidence",
)


def compute_tissue_depth(
    full_labels: np.ndarray,
    df: pd.DataFrame,
    spacing: Optional[Tuple[float, ...]] = None,
    downsample: int = 1,
) -> pd.DataFrame:
    """Whole-tissue depth ("in vs out") + spatial context + a per-cell OUTWARD
    normal.

    A GLOBAL pass — it needs the WHOLE tissue at once (a per-cell bbox crop cannot
    see the tissue surface), so it runs ONCE on the full label volume inside the
    ``adjacency=True`` block of :func:`extract_label_features` (reusing the already-
    loaded ``full_labels`` — the volume is NOT read twice), exactly like
    :func:`compute_centroid_spacing` / :func:`compute_neighbor_features`.

    The primitive is a single 3D Euclidean distance transform of the WHOLE-TISSUE
    mask (``full_labels > 0``): for every interior voxel, the distance to the
    nearest tissue surface (the nearest background voxel). This "in/out" field is
    0 at the tissue surface and grows toward the core — a depth coordinate for the
    whole sample. Each cell then summarises the field over its own voxels, plus a
    handful of cheap volume-relative position covariates, plus the field's
    local gradient direction at the cell (the OUTWARD normal).

    GEOMETRY CAVEAT (read before interpreting these biologically):
        The tissue-EDT measures distance to the NEAREST tissue surface, whatever
        that surface is. This is a clean apical/basal DEPTH PROXY only for a
        CURVED / CLOSED tissue — an organoid, a gland, a budding or cystic
        epithelium — where "the nearest surface" really is the tissue's outer
        boundary and depth ≈ how far in from it you are. For a FLAT SLAB the
        "nearest surface" conflates the top face, the bottom face, AND the cut
        lateral edges of the imaged volume, so a mid-slab cell is "deep" and a cell
        near ANY of those faces is "shallow" regardless of true apical/basal
        identity. There, read ``normalized_tissue_depth`` as a THICKNESS-POSITION
        (how far through the slab, not signed apical→basal) and EXCLUDE
        ``at_fov_border`` cells (their nearest "surface" is a cut edge of the field
        of view, so their depth + outward normal are field-of-view artefacts, not
        tissue geometry). The outward normal inherits the same caveat: it points to
        the nearest surface, which is the true tissue normal only where that
        surface is the real tissue boundary.

    Features added (16):

      tissue-EDT "in/out" field (per cell, over its voxels):
        ``distance_to_tissue_surface`` — the MIN tissue-EDT over the cell's voxels
            = how close the cell gets to ANY tissue surface. Small = a surface
            (apical/basal-adjacent) cell; large = a buried core cell. Physical
            units when ``spacing`` given. Unbounded >= 0.
        ``tissue_depth_mean`` — the MEAN tissue-EDT over the cell (its typical
            depth). Unbounded >= 0.
        ``tissue_depth_max`` — the MAX tissue-EDT over the cell (its DEEPEST
            point). Unbounded >= 0.
        ``normalized_tissue_depth`` — ``tissue_depth_mean / max(tissue-EDT over the
            whole tissue)`` ∈ [0, 1] roughly: ~0 = at a surface, ~1 = the tissue
            core. Bounded (self-scaled by the global max depth).
        ``at_fov_border`` — 1.0 when the cell touches the volume bbox edge (any of
            the 6 faces of the array), else 0.0. Such a cell's nearest "surface"
            may be a CUT EDGE of the field of view, so its depth + outward normal
            are unreliable (exclude it from depth analysis). Bounded {0, 1}.

 spatial context (cell position WITHIN the tissue bbox):
        ``centroid_z_relative`` / ``centroid_y_relative`` / ``centroid_x_relative``
            — the cell centroid's position along each axis, normalised to [0, 1]
            within the TISSUE bounding box (0 = the low-index face of the occupied
            tissue, 1 = the high-index face). NaN if the tissue spans a single
            voxel on that axis. Bounded.
        ``distance_to_tissue_centroid`` — physical distance from the cell centroid
            to the GLOBAL tissue centroid (the mean of all foreground voxels).
            Unbounded >= 0.
        ``radial_tissue_position`` — ``distance_to_tissue_centroid`` divided by the
            MAX such distance over all cells ∈ [0, 1]: ~0 = a central cell, ~1 = an
            edge cell. Bounded (self-scaled).
        ``distance_to_boundary_face_min`` (Wave I) — the MIN physical
            distance from the cell centroid to any of the 6 faces of the imaged
            volume (field of view). Small = a cell near the edge of the imaged
            field (its phenotype may be truncation-affected); large = a cell well
            inside the FOV. Distinct from ``distance_to_tissue_surface`` (which is
            distance to the nearest TISSUE surface, not the FOV box). Physical
            units when ``spacing`` given. Unbounded >= 0; NaN without centroids.
        ``local_tissue_depth`` (Mop-up O) — the cell's mean tissue-EDT depth
            in CELL-RADIUS units: ``tissue_depth_mean / r_equiv`` where ``r_equiv``
            is the cell's equivalent-sphere radius. A scale-free depth (how many of
            its own radii deep the cell sits) — distinct from
            ``normalized_tissue_depth`` (normalized by the GLOBAL max depth) and
            from the raw ``tissue_depth_mean`` (physical units). Unbounded >= 0;
            only needs the cell mask (defined even without centroids).

      per-cell OUTWARD normal (the sign basis for later apical/basal frames):
        SIGN CAVEAT (read before treating ANY downstream apical/basal column as
            apical): this outward normal points toward the NEAREST tissue surface,
            which only EQUALS true APICAL for a CURVED / CLOSED tissue (organoid /
            gland / cyst) OR a basal-cell-free field of view. For a flat or
            multi-stratum FOV a basal-stratum cell's nearest surface is the BASAL
            face, so the "apical" sign flips. Every column derived from this sign
            (the signed apical/basal frame, directional contact, surface partition,
            ma_apical/basal, apical-junction, the signed nuclear offset/distances,
            the prob/puncta apical-basal gradients) is therefore tagged
            ``sign_trusted=False`` in the manifest — treat them as magnitude-only /
            low-trust unless the tissue geometry guarantees the sign (the feature catalogue
            rule #2).
        ``tissue_outward_z`` / ``tissue_outward_y`` / ``tissue_outward_x`` — a UNIT
            vector (ZYX) pointing OUTWARD (toward the nearest tissue surface) at
            the cell centroid. It is the NEGATIVE normalised gradient of the
            tissue-EDT (the EDT increases inward, so ``-∇EDT`` points out). NaN
            where the gradient is ~zero (the flat tissue core — direction
            ambiguous).
        ``tissue_outward_confidence`` — ``|∇(tissue-EDT)|`` at the centroid,
            divided by its EXPECTED value 1.0 (a clean EDT has unit gradient where
            it is well-defined), clipped to [0, 1]. ~1 near a surface (a sharp,
            well-defined outward direction), ~0 deep in the flat core (the gradient
            collapses → the direction is meaningless). Use it to down-weight /
            mask the outward normal. Bounded.

    All NaN-/empty-guarded: an empty table, a tissue with no foreground, a
    degenerate single-voxel cell, or a missing centroid column degrade to NaN/0 for
    the affected columns rather than raising.

    Args:
        full_labels: the FULL 3D integer label volume (Z, Y, X). Background 0.
            Reused from the ``adjacency=True`` block (loaded once); NOT re-read.
        df: assembled per-cell table (modified in-place). Needs ``label`` to index
            cells via ``find_objects``; ``centroid-0/1/2`` enable the spatial-
            context distances + the outward-normal sampling location (NaN'd when
            absent).
        spacing: voxel spacing ``(Z, Y, X)`` in physical units. Threaded to the
            distance transform (``sampling=spacing``) AND to ``np.gradient`` so the
            EDT field + its gradient + all distances are in PHYSICAL units, in the
            same frame as the ``centroid-*`` columns (which ``extract_features``
            already scales by spacing). ``None`` = voxel units.
        downsample: integer field-downsample factor ``N`` (S2/S3 scaling). ``N == 1``
            (default) runs the EXACT full-resolution code below (byte-identical). For
            ``N > 1`` the EDT field + its gradient run on a foreground DOWNSAMPLED by
            ``N`` (built streaming off ``full_labels`` if it is a memmap, so the full
            volume is never resident), and each cell is sampled at its centroid mapped
            onto the coarse grid. Physical units stay EXACT (the coarse EDT uses
            ``sampling = spacing * N``); the per-cell summaries (min/mean/max depth)
            collapse to the single coarse value at the cell's coarse centroid, and the
            outward normal's DIRECTION is downsample-invariant but its ACCURACY
            degrades on curved surfaces (validate ``N`` with
            ``scripts/bench_field_downsample.py``). An APPROXIMATION for ``N > 1``.

    Returns:
        The same DataFrame with the 16 new columns added.
    """
    downsample = int(downsample)
    if downsample > 1:
        return _compute_tissue_depth_downsampled(
            full_labels, df, spacing=spacing, downsample=downsample)

    from scipy.ndimage import distance_transform_edt, find_objects

    n = len(df)
    if n == 0:
        for c in TISSUE_DEPTH_COLUMNS:
            df[c] = pd.Series(dtype=np.float64)
        return df
    if "label" not in df.columns:
        for c in TISSUE_DEPTH_COLUMNS:
            df[c] = np.nan
        return df

    lab = np.asarray(full_labels)
    mask = lab > 0
    # spacing as a length-3 sampling vector (physical) — None = isotropic voxel.
    sp = (tuple(float(s) for s in spacing) if spacing is not None else None)
    sampling = sp if sp is not None else (1.0, 1.0, 1.0)

    label_ids = df["label"].values.astype(np.int64)

    # --- spatial context prerequisites (centroids + tissue extent) -----------
    cen_cols = ["centroid-0", "centroid-1", "centroid-2"]
    have_cen = all(c in df.columns for c in cen_cols)
    centroids = (df[cen_cols].values.astype(np.float64) if have_cen else None)

    # init every output to its empty-default.
    distance_to_surface = np.full(n, np.nan, dtype=np.float64)
    depth_mean = np.full(n, np.nan, dtype=np.float64)
    depth_max = np.full(n, np.nan, dtype=np.float64)
    norm_depth = np.full(n, np.nan, dtype=np.float64)
    at_border = np.zeros(n, dtype=np.float64)
    cz_rel = np.full(n, np.nan, dtype=np.float64)
    cy_rel = np.full(n, np.nan, dtype=np.float64)
    cx_rel = np.full(n, np.nan, dtype=np.float64)
    dist_centroid = np.full(n, np.nan, dtype=np.float64)
    radial_pos = np.full(n, np.nan, dtype=np.float64)
    dist_face_min = np.full(n, np.nan, dtype=np.float64)   # Wave I
    local_depth = np.full(n, np.nan, dtype=np.float64)     # Mop-up O
    out_z = np.full(n, np.nan, dtype=np.float64)
    out_y = np.full(n, np.nan, dtype=np.float64)
    out_x = np.full(n, np.nan, dtype=np.float64)
    out_conf = np.full(n, np.nan, dtype=np.float64)

    if not mask.any():
        # No tissue at all — everything undefined.
        df["distance_to_tissue_surface"] = distance_to_surface
        df["tissue_depth_mean"] = depth_mean
        df["tissue_depth_max"] = depth_max
        df["normalized_tissue_depth"] = norm_depth
        df["at_fov_border"] = at_border
        df["centroid_z_relative"] = cz_rel
        df["centroid_y_relative"] = cy_rel
        df["centroid_x_relative"] = cx_rel
        df["distance_to_tissue_centroid"] = dist_centroid
        df["radial_tissue_position"] = radial_pos
        df["distance_to_boundary_face_min"] = dist_face_min
        df["local_tissue_depth"] = local_depth
        df["tissue_outward_z"] = out_z
        df["tissue_outward_y"] = out_y
        df["tissue_outward_x"] = out_x
        df["tissue_outward_confidence"] = out_conf
        return df

    # --- the "in/out" field: 3D EDT of the WHOLE-TISSUE mask -----------------
    # distance from each foreground voxel to the nearest background (surface).
    edt = distance_transform_edt(mask, sampling=sampling).astype(np.float64)
    global_max_depth = float(edt.max())

    # the EDT gradient (∂EDT/∂z, ∂y, ∂x) in PHYSICAL units (sampling spacing).
    # -∇EDT points OUTWARD (toward the surface) because EDT grows inward. Computed
    # ONCE for the whole volume; sampled per cell at its centroid voxel below.
    gz, gy, gx = np.gradient(edt, *sampling)

    # tissue bounding box (for the relative-position normalisation) +
    # global tissue centroid (mean of all foreground voxels, physical).
    fg = np.array(np.nonzero(mask))            # (3, n_fg) voxel coords (z, y, x)
    bb_min = fg.min(axis=1).astype(np.float64)
    bb_max = fg.max(axis=1).astype(np.float64)
    bb_span = bb_max - bb_min                  # voxel span per axis
    sp_vec = np.asarray(sampling, dtype=np.float64)
    # global tissue centroid in PHYSICAL coords (mean voxel × spacing) to match the
    # physical ``centroid-*`` columns.
    tissue_centroid_phys = fg.mean(axis=1).astype(np.float64) * sp_vec

    vol_shape = np.asarray(lab.shape, dtype=np.int64)
    objs = find_objects(lab.astype(np.int32))
    n_slots = len(objs)

    eps = 1e-12

    for idx in range(n):
        lid = int(label_ids[idx])
        if not (1 <= lid <= n_slots) or objs[lid - 1] is None:
            continue
        sl = objs[lid - 1]
        cell_mask = lab[sl] == lid
        if not cell_mask.any():
            continue

        # tissue-EDT over THIS cell's voxels (crop of the global field).
        cell_edt = edt[sl][cell_mask]
        distance_to_surface[idx] = float(cell_edt.min())
        cm = float(cell_edt.mean())
        depth_mean[idx] = cm
        depth_max[idx] = float(cell_edt.max())
        if global_max_depth > eps:
            norm_depth[idx] = float(min(cm / global_max_depth, 1.0))

        # Mop-up O — local_tissue_depth: the cell's mean tissue-EDT depth
        # expressed in CELL-RADIUS units (how many of its own equivalent radii
        # deep the cell sits), so it is scale-free across cells of different sizes
        # — distinct from normalized_tissue_depth (normalized by the GLOBAL max
        # depth). equiv radius R = (3 V / 4π)^(1/3), V = the cell's physical volume.
        cell_phys_vol = float(cell_mask.sum()) * float(np.prod(sp_vec))
        if cell_phys_vol > 0:
            r_equiv = (3.0 * cell_phys_vol / (4.0 * np.pi)) ** (1.0 / 3.0)
            if r_equiv > eps:
                local_depth[idx] = cm / r_equiv

        # at_fov_border: the cell's bbox touches any of the 6 array faces.
        starts = np.array([sl[0].start, sl[1].start, sl[2].start], dtype=np.int64)
        stops = np.array([sl[0].stop, sl[1].stop, sl[2].stop], dtype=np.int64)
        if bool((starts <= 0).any() or (stops >= vol_shape).any()):
            at_border[idx] = 1.0

        # --- spatial context (needs the centroid columns) --------------
        if have_cen:
            # centroid-* are PHYSICAL when extract_features ran with spacing;
            # convert to a VOXEL coordinate (÷ spacing) for the bbox-relative
            # normalisation (bb_min/span are voxel coords) and for the outward-
            # normal voxel index.
            cen_phys = centroids[idx]
            cen_vox = cen_phys / np.where(sp_vec > 0, sp_vec, 1.0)
            rel = np.full(3, np.nan, dtype=np.float64)
            for ax in range(3):
                if bb_span[ax] > 0:
                    rel[ax] = float(
                        np.clip((cen_vox[ax] - bb_min[ax]) / bb_span[ax], 0.0, 1.0))
            cz_rel[idx], cy_rel[idx], cx_rel[idx] = rel[0], rel[1], rel[2]

            # physical distance to the global tissue centroid.
            dist_centroid[idx] = float(
                np.linalg.norm(cen_phys - tissue_centroid_phys))

            # Wave I — min physical distance from the centroid to a FOV face.
            # For each axis the nearer face is min(coord, (extent-1) - coord);
            # scale to physical and take the min over the 3 axes. The far face
            # of a single-voxel axis is at index 0, so the distance is 0 there.
            face_dists = []
            for ax in range(3):
                lo = float(cen_vox[ax])
                hi = float(vol_shape[ax] - 1) - float(cen_vox[ax])
                face_dists.append(max(min(lo, hi), 0.0) * float(sp_vec[ax]))
            dist_face_min[idx] = float(min(face_dists))

            # --- outward normal: sample -∇EDT at the centroid voxel ----------
            vz = int(np.clip(round(cen_vox[0]), 0, vol_shape[0] - 1))
            vy = int(np.clip(round(cen_vox[1]), 0, vol_shape[1] - 1))
            vx = int(np.clip(round(cen_vox[2]), 0, vol_shape[2] - 1))
            grad = np.array([gz[vz, vy, vx], gy[vz, vy, vx], gx[vz, vy, vx]],
                            dtype=np.float64)
            gmag = float(np.linalg.norm(grad))
            # confidence = |∇EDT| / expected(1.0), clipped to [0,1]. A clean EDT
            # has ~unit gradient where well-defined; it collapses in the flat core.
            out_conf[idx] = float(np.clip(gmag, 0.0, 1.0))
            if gmag > eps:
                outward = -grad / gmag        # -∇EDT points toward the surface
                out_z[idx], out_y[idx], out_x[idx] = (
                    float(outward[0]), float(outward[1]), float(outward[2]))

    # radial_tissue_position: self-scale distance_to_tissue_centroid by its max.
    finite_dc = dist_centroid[np.isfinite(dist_centroid)]
    if finite_dc.size:
        max_dc = float(finite_dc.max())
        if max_dc > eps:
            radial_pos = dist_centroid / max_dc

    df["distance_to_tissue_surface"] = distance_to_surface
    df["tissue_depth_mean"] = depth_mean
    df["tissue_depth_max"] = depth_max
    df["normalized_tissue_depth"] = norm_depth
    df["at_fov_border"] = at_border
    df["centroid_z_relative"] = cz_rel
    df["centroid_y_relative"] = cy_rel
    df["centroid_x_relative"] = cx_rel
    df["distance_to_tissue_centroid"] = dist_centroid
    df["radial_tissue_position"] = radial_pos
    df["distance_to_boundary_face_min"] = dist_face_min   # Wave I
    df["local_tissue_depth"] = local_depth                # Mop-up O
    df["tissue_outward_z"] = out_z
    df["tissue_outward_y"] = out_y
    df["tissue_outward_x"] = out_x
    df["tissue_outward_confidence"] = out_conf
    return df


def _compute_tissue_depth_downsampled(
    full_labels,
    df: pd.DataFrame,
    spacing: Optional[Tuple[float, ...]] = None,
    downsample: int = 2,
) -> pd.DataFrame:
    """S2/S3 DOWNSAMPLED tissue-depth — the ``downsample > 1`` branch of
    :func:`compute_tissue_depth`. NOT byte-identical to ``N == 1`` (an APPROXIMATION).

    The EDT "in/out" field + its gradient (the outward normal) run on a foreground
    DOWNSAMPLED by ``N`` — built STREAMING off ``full_labels`` (a memmap is never
    loaded resident; see :func:`epicycle.core.field_downsample.coarse_foreground`).
    Physical units are EXACT because the coarse EDT uses ``sampling = spacing * N`` (a
    coarse voxel spans ``N`` full-res voxels per axis), so no extra ``×N`` scaling is
    needed. Each cell is sampled at its centroid mapped onto the coarse grid:

    * The per-cell tissue-EDT min/mean/max all COLLAPSE to the single coarse-grid value
      at the cell's coarse centroid (a coarse voxel is the whole resolution available),
      so ``distance_to_tissue_surface`` / ``tissue_depth_mean`` / ``tissue_depth_max``
      are equal at ``N > 1`` (the field has no sub-coarse-voxel structure to vary over).
    * ``normalized_tissue_depth`` divides that coarse mean by the coarse global max.
    * ``at_fov_border`` + ``local_tissue_depth`` are computed EXACTLY from the per-cell
      table's full-res ``bbox-*`` + physical ``area`` columns (NOT the coarse grid) —
      they need no field, so they keep full precision.
    * The spatial-context covariates (relative position, distance to the tissue
      centroid, radial position, distance to a FOV face) use the COARSE tissue bbox /
      centroid scaled back to physical via ``coarse_spacing``; the FOV-face distance
      uses the EXACT full-res volume shape.
    * The outward normal samples ``-∇(coarse EDT)`` at the coarse centroid voxel — a
      UNIT vector whose direction is downsample-invariant but whose accuracy degrades
      on curvature (the audit's flagged risk; measured by the bench script).

    Output columns + dtypes match the ``N == 1`` path exactly.
    """
    from scipy.ndimage import distance_transform_edt
    from packages.epicycle_scaledseg.core.field_downsample import coarse_foreground

    N = int(downsample)
    n = len(df)
    if n == 0:
        for c in TISSUE_DEPTH_COLUMNS:
            df[c] = pd.Series(dtype=np.float64)
        return df
    if "label" not in df.columns:
        for c in TISSUE_DEPTH_COLUMNS:
            df[c] = np.nan
        return df

    sp = (tuple(float(s) for s in spacing) if spacing is not None else None)
    sp_vec = np.asarray(sp, dtype=np.float64) if sp is not None else np.ones(3)
    # coarse sampling: a coarse voxel spans N full-res voxels per axis -> the EDT in
    # PHYSICAL units comes out directly with sampling = spacing * N (None -> N per axis).
    coarse_sampling = tuple(float(s) * N for s in (sp if sp is not None else (1.0, 1.0, 1.0)))
    cs_vec = np.asarray(coarse_sampling, dtype=np.float64)

    # full-res volume shape (for at_fov_border + the FOV-face distance) — read off the
    # array/memmap shape WITHOUT materialising it.
    vol_shape = np.asarray(full_labels.shape, dtype=np.int64)

    # init every output to its empty-default (same as the N==1 path).
    distance_to_surface = np.full(n, np.nan, dtype=np.float64)
    depth_mean = np.full(n, np.nan, dtype=np.float64)
    depth_max = np.full(n, np.nan, dtype=np.float64)
    norm_depth = np.full(n, np.nan, dtype=np.float64)
    at_border = np.zeros(n, dtype=np.float64)
    cz_rel = np.full(n, np.nan, dtype=np.float64)
    cy_rel = np.full(n, np.nan, dtype=np.float64)
    cx_rel = np.full(n, np.nan, dtype=np.float64)
    dist_centroid = np.full(n, np.nan, dtype=np.float64)
    radial_pos = np.full(n, np.nan, dtype=np.float64)
    dist_face_min = np.full(n, np.nan, dtype=np.float64)
    local_depth = np.full(n, np.nan, dtype=np.float64)
    out_z = np.full(n, np.nan, dtype=np.float64)
    out_y = np.full(n, np.nan, dtype=np.float64)
    out_x = np.full(n, np.nan, dtype=np.float64)
    out_conf = np.full(n, np.nan, dtype=np.float64)

    # --- the COARSE "in/out" field (streamed; full volume never resident) ----------
    cfg = coarse_foreground(full_labels, N)             # bool, ceil-div shape
    eps = 1e-12

    # at_fov_border + local_tissue_depth: EXACT from the per-cell table (no field).
    have_bbox = all(f"bbox-{k}" in df.columns for k in range(6))
    have_area = "area" in df.columns
    if have_bbox:
        bb = df[[f"bbox-{k}" for k in range(6)]].values.astype(np.int64)
        starts = bb[:, 0:3]
        stops = bb[:, 3:6]
        at_border[(starts <= 0).any(axis=1) | (stops >= vol_shape[None, :]).any(axis=1)] = 1.0

    if not cfg.any():
        # No tissue at all — only the table-derived columns are defined.
        df["distance_to_tissue_surface"] = distance_to_surface
        df["tissue_depth_mean"] = depth_mean
        df["tissue_depth_max"] = depth_max
        df["normalized_tissue_depth"] = norm_depth
        df["at_fov_border"] = at_border
        df["centroid_z_relative"] = cz_rel
        df["centroid_y_relative"] = cy_rel
        df["centroid_x_relative"] = cx_rel
        df["distance_to_tissue_centroid"] = dist_centroid
        df["radial_tissue_position"] = radial_pos
        df["distance_to_boundary_face_min"] = dist_face_min
        df["local_tissue_depth"] = local_depth
        df["tissue_outward_z"] = out_z
        df["tissue_outward_y"] = out_y
        df["tissue_outward_x"] = out_x
        df["tissue_outward_confidence"] = out_conf
        return df

    edt = distance_transform_edt(cfg, sampling=coarse_sampling).astype(np.float64)
    global_max_depth = float(edt.max())
    gz, gy, gx = np.gradient(edt, *coarse_sampling)
    cshape = np.asarray(cfg.shape, dtype=np.int64)

    # coarse tissue bbox (voxel coords on the COARSE grid) + global tissue centroid
    # (mean of coarse foreground voxels) scaled to PHYSICAL via coarse_sampling.
    cfg_nz = np.array(np.nonzero(cfg))                  # (3, n_fg_coarse)
    cbb_min = cfg_nz.min(axis=1).astype(np.float64)
    cbb_max = cfg_nz.max(axis=1).astype(np.float64)
    cbb_span = cbb_max - cbb_min
    tissue_centroid_phys = cfg_nz.mean(axis=1).astype(np.float64) * cs_vec

    # --- per-cell prerequisites: centroids (-> coarse index) -----------------------
    cen_cols = ["centroid-0", "centroid-1", "centroid-2"]
    have_cen = all(c in df.columns for c in cen_cols)
    centroids = (df[cen_cols].values.astype(np.float64) if have_cen else None)

    # local_tissue_depth needs the cell's PHYSICAL volume -> from `area` (exact). It
    # also needs tissue_depth_mean (the coarse field value at the cell), so it is
    # filled inside the per-cell loop once the depth is sampled.
    area_phys = (df["area"].values.astype(np.float64) if have_area else None)

    for idx in range(n):
        if centroids is None:
            continue
        cen_phys = centroids[idx]
        # physical centroid -> full-res voxel -> coarse index (floor div by N), clamped.
        cen_vox = cen_phys / np.where(sp_vec > 0, sp_vec, 1.0)
        ci = np.floor(cen_vox / N).astype(np.int64)
        ci = np.clip(ci, 0, cshape - 1)
        cz_i, cy_i, cx_i = int(ci[0]), int(ci[1]), int(ci[2])

        # tissue-EDT at the cell's coarse centroid voxel: the only resolution we have
        # -> min/mean/max all collapse to this single coarse value.
        d = float(edt[cz_i, cy_i, cx_i])
        distance_to_surface[idx] = d
        depth_mean[idx] = d
        depth_max[idx] = d
        if global_max_depth > eps:
            norm_depth[idx] = float(min(d / global_max_depth, 1.0))

        # local_tissue_depth: mean depth / equivalent-sphere radius (EXACT cell vol).
        if area_phys is not None:
            cell_phys_vol = float(area_phys[idx])
            if cell_phys_vol > 0:
                r_equiv = (3.0 * cell_phys_vol / (4.0 * np.pi)) ** (1.0 / 3.0)
                if r_equiv > eps:
                    local_depth[idx] = d / r_equiv

        # --- spatial context (coarse bbox / centroid) -----------------------
        rel = np.full(3, np.nan, dtype=np.float64)
        for ax in range(3):
            if cbb_span[ax] > 0:
                rel[ax] = float(
                    np.clip((ci[ax] - cbb_min[ax]) / cbb_span[ax], 0.0, 1.0))
        cz_rel[idx], cy_rel[idx], cx_rel[idx] = rel[0], rel[1], rel[2]

        dist_centroid[idx] = float(np.linalg.norm(cen_phys - tissue_centroid_phys))

        # FOV-face distance: EXACT full-res volume shape (the FOV box is full-res).
        face_dists = []
        for ax in range(3):
            lo = float(cen_vox[ax])
            hi = float(vol_shape[ax] - 1) - float(cen_vox[ax])
            face_dists.append(max(min(lo, hi), 0.0) * float(sp_vec[ax]))
        dist_face_min[idx] = float(min(face_dists))

        # --- outward normal: -∇(coarse EDT) at the coarse centroid voxel ----------
        grad = np.array([gz[cz_i, cy_i, cx_i], gy[cz_i, cy_i, cx_i], gx[cz_i, cy_i, cx_i]],
                        dtype=np.float64)
        gmag = float(np.linalg.norm(grad))
        out_conf[idx] = float(np.clip(gmag, 0.0, 1.0))
        if gmag > eps:
            outward = -grad / gmag
            out_z[idx], out_y[idx], out_x[idx] = (
                float(outward[0]), float(outward[1]), float(outward[2]))

    # radial_tissue_position: self-scale distance_to_tissue_centroid by its max.
    finite_dc = dist_centroid[np.isfinite(dist_centroid)]
    if finite_dc.size:
        max_dc = float(finite_dc.max())
        if max_dc > eps:
            radial_pos = dist_centroid / max_dc

    df["distance_to_tissue_surface"] = distance_to_surface
    df["tissue_depth_mean"] = depth_mean
    df["tissue_depth_max"] = depth_max
    df["normalized_tissue_depth"] = norm_depth
    df["at_fov_border"] = at_border
    df["centroid_z_relative"] = cz_rel
    df["centroid_y_relative"] = cy_rel
    df["centroid_x_relative"] = cx_rel
    df["distance_to_tissue_centroid"] = dist_centroid
    df["radial_tissue_position"] = radial_pos
    df["distance_to_boundary_face_min"] = dist_face_min
    df["local_tissue_depth"] = local_depth
    df["tissue_outward_z"] = out_z
    df["tissue_outward_y"] = out_y
    df["tissue_outward_x"] = out_x
    df["tissue_outward_confidence"] = out_conf
    return df


# P1b — lumen-cavity primitive. A lumen is
# an ENCLOSED background cavity surrounded by tissue — the inner surface of a hollow
# organoid / gland / cyst — NOT the surrounding medium the tissue floats in. Module
# constant so the exact-set family test + the feature_stage registry stay in lockstep.
LUMEN_COLUMNS = (
    "distance_to_lumen", "lumen_facing_fraction",
)


def compute_lumen_features(
    full_labels: np.ndarray,
    df: pd.DataFrame,
    spacing: Optional[Tuple[float, ...]] = None,
    downsample: int = 1,
) -> pd.DataFrame:
    """Per-cell distance to the nearest ENCLOSED background cavity (a LUMEN) +
    the share of the cell surface that faces it (the feature catalogue ``distance_to_lumen``
    /; P1b — a NEW-input GLOBAL primitive, default-off byte-identical).

    A GLOBAL pass — it needs the WHOLE tissue at once to tell an INTERNAL cavity
    (a lumen) from the EXTERNAL medium, so it runs ONCE on the full label volume in
    the ``lumen=True`` block of :func:`epicycle.reconcile.features.extract_label_features`
    (reusing the already-loaded ``full_labels`` — the volume is NOT read twice),
    exactly like :func:`compute_tissue_depth` / :func:`compute_neighbor_features`.

    WHAT IS A LUMEN (the honest definition):
        The whole-tissue foreground is ``full_labels > 0``. ``binary_fill_holes``
        on that mask fills every ENCLOSED background pocket but leaves the
        surrounding (border-connected) medium empty. So::

            cavities = binary_fill_holes(foreground) AND NOT foreground

        is the set of all enclosed background voxels. A cavity component that
        TOUCHES the volume border is the external medium leaking in through the
        field-of-view edge (or a tissue that simply isn't closed in this crop), NOT
        a real lumen — those are DROPPED (a ``clear_border`` on the cavity labels).
        What remains is the true interior lumen(s).

    HONEST CAVEAT (read before interpreting): MANY explants / flat slabs / open
        epithelia have NO enclosed cavity at all -> the lumen mask is EMPTY and
        EVERY ``distance_to_lumen`` is **NaN** (NOT 0, NOT a huge sentinel — there
        is genuinely no lumen to measure a distance to). That all-NaN column is the
        CORRECT answer for a tissue without a lumen, and downstream code must treat
        NaN as "no lumen in this volume", not as a missing measurement. The cavity
        detection is also only as good as the segmentation closure: a lumen with a
        gap in its tissue wall connects to the medium and is (correctly) not
        detected as enclosed.

    Features added (2):
        ``distance_to_lumen`` — the MIN physical EDT from the cell's voxels to the
            nearest LUMEN voxel: how close the cell gets to an internal cavity (0
            for a cell whose surface lines the lumen, larger for a buried cell).
            Physical units when ``spacing`` is given. NaN when the volume has NO
            lumen (the common case — see the caveat). Unbounded >= 0.
        ``lumen_facing_fraction`` — the fraction of the cell's SURFACE voxels (the
            cell's 6-connected boundary shell) whose nearest background neighbour is
            a lumen voxel rather than the external medium / another cell — i.e. how
            much of the cell's exposed membrane lines the lumen. A cheap apical-
            footprint proxy for a cell at the lumen wall. [0, 1]. NaN when the
            volume has no lumen.

    Args:
        full_labels: the FULL 3D integer label volume (Z, Y, X). Background 0.
        df: the assembled per-cell table (modified in-place). Needs ``label``.
        spacing: voxel spacing ``(Z, Y, X)`` in physical units -> the distance
            transform runs in PHYSICAL units. ``None`` = voxel units.

        downsample: integer field-downsample factor ``N`` (S2/S3 scaling). ``N == 1``
            (default) runs the EXACT full-resolution code below (byte-identical). For
            ``N > 1`` the cavity detection (``binary_fill_holes`` -> ``clear_border``)
            + the lumen distance field run on a foreground DOWNSAMPLED by ``N`` (built
            streaming off ``full_labels`` if it is a memmap, so the full volume is
            never resident); ``distance_to_lumen`` is sampled at the cell's coarse
            centroid (physical-exact via ``sampling = spacing * N``) and
            ``lumen_facing_fraction`` is approximated on the COARSE grid. An
            APPROXIMATION for ``N > 1``.

    Returns:
        The same DataFrame with the 2 columns added (all-NaN when there is no
        enclosed cavity in the volume).
    """
    downsample = int(downsample)
    if downsample > 1:
        return _compute_lumen_features_downsampled(
            full_labels, df, spacing=spacing, downsample=downsample)

    from scipy.ndimage import (
        distance_transform_edt, find_objects, binary_fill_holes, label as cc_label,
    )

    n = len(df)
    if n == 0:
        for c in LUMEN_COLUMNS:
            df[c] = pd.Series(dtype=np.float64)
        return df
    if "label" not in df.columns:
        for c in LUMEN_COLUMNS:
            df[c] = np.nan
        return df

    dist_lumen = np.full(n, np.nan, dtype=np.float64)
    facing_frac = np.full(n, np.nan, dtype=np.float64)

    lab = np.asarray(full_labels)
    fg = lab > 0
    sp = (tuple(float(s) for s in spacing) if spacing is not None else None)
    sampling = sp if sp is not None else (1.0, 1.0, 1.0)

    if not fg.any():
        df["distance_to_lumen"] = dist_lumen
        df["lumen_facing_fraction"] = facing_frac
        return df

    # --- the lumen mask: enclosed background cavities, minus border-touching ----
    filled = binary_fill_holes(fg)
    cavities = filled & ~fg                  # all enclosed background pockets
    if not cavities.any():
        # No enclosed cavity at all (a solid tissue / an open slab). The CORRECT
        # answer is all-NaN — there is no lumen to measure a distance to.
        df["distance_to_lumen"] = dist_lumen
        df["lumen_facing_fraction"] = facing_frac
        return df

    # DROP any cavity component that touches the volume border — that is the
    # external medium leaking in through the FOV edge (or an unclosed tissue), NOT
    # a real internal lumen. (binary_fill_holes already excludes the
    # border-connected medium, but a partially-imaged cavity that opens onto the
    # FOV edge can survive the fill; clear it here for safety.)
    cav_lab, n_cav = cc_label(cavities)
    if n_cav == 0:
        df["distance_to_lumen"] = dist_lumen
        df["lumen_facing_fraction"] = facing_frac
        return df
    border_ids = set()
    for face in (cav_lab[0], cav_lab[-1],
                 cav_lab[:, 0], cav_lab[:, -1],
                 cav_lab[:, :, 0], cav_lab[:, :, -1]):
        border_ids.update(np.unique(face).tolist())
    border_ids.discard(0)
    lumen = cavities.copy()
    if border_ids:
        lumen &= ~np.isin(cav_lab, list(border_ids))
    if not lumen.any():
        # Every enclosed pocket touched the border -> no true interior lumen.
        df["distance_to_lumen"] = dist_lumen
        df["lumen_facing_fraction"] = facing_frac
        return df

    # --- distance field to the nearest lumen voxel ------------------------------
    # EDT of the COMPLEMENT of the lumen: at every non-lumen voxel = distance to the
    # nearest lumen voxel (physical via sampling). 0 inside the lumen.
    dist_to_lumen_field = distance_transform_edt(~lumen, sampling=sampling).astype(np.float64)

    label_ids = df["label"].values.astype(np.int64)
    objs = find_objects(lab.astype(np.int32))
    n_slots = len(objs)

    # background set used to find each cell's surface-facing neighbour. A cell
    # SURFACE voxel touches background; that background voxel is either lumen or
    # not — lumen_facing_fraction = (surface voxels whose nearest bg is lumen) /
    # (surface voxels). We approximate "nearest bg is lumen" by: the surface voxel
    # is at distance ~1 step from a lumen voxel (its 6-neighbourhood contains lumen).
    from scipy.ndimage import binary_dilation
    # lumen dilated by one 6-connected step = the voxels DIRECTLY adjacent to lumen.
    struct6 = np.zeros((3, 3, 3), dtype=bool)
    struct6[1, 1, 1] = True
    struct6[0, 1, 1] = struct6[2, 1, 1] = True
    struct6[1, 0, 1] = struct6[1, 2, 1] = True
    struct6[1, 1, 0] = struct6[1, 1, 2] = True
    lumen_adjacent = binary_dilation(lumen, structure=struct6) & ~lumen

    for idx in range(n):
        lid = int(label_ids[idx])
        if not (1 <= lid <= n_slots) or objs[lid - 1] is None:
            continue
        sl = objs[lid - 1]
        cell_mask = lab[sl] == lid
        if not cell_mask.any():
            continue
        # distance_to_lumen = MIN of the lumen-distance field over the cell voxels.
        cell_field = dist_to_lumen_field[sl][cell_mask]
        dist_lumen[idx] = float(cell_field.min())

        # lumen_facing_fraction: of the cell's SURFACE voxels (cell voxels that
        # border background), how many sit DIRECTLY against a lumen voxel.
        cell_surface = cell_mask & ~_binary_erosion_local(cell_mask)
        n_surf = int(cell_surface.sum())
        if n_surf > 0:
            facing = cell_surface & lumen_adjacent[sl]
            facing_frac[idx] = float(facing.sum()) / float(n_surf)

    df["distance_to_lumen"] = dist_lumen
    df["lumen_facing_fraction"] = facing_frac
    return df


def _compute_lumen_features_downsampled(
    full_labels,
    df: pd.DataFrame,
    spacing: Optional[Tuple[float, ...]] = None,
    downsample: int = 2,
) -> pd.DataFrame:
    """S2/S3 DOWNSAMPLED lumen features — the ``downsample > 1`` branch
    of :func:`compute_lumen_features`. NOT byte-identical (an APPROXIMATION).

    The cavity detection (``binary_fill_holes`` -> ``clear_border`` on the COARSE
    foreground) + the lumen-distance EDT all run at downsample ``N``; the coarse
    foreground is built STREAMING off ``full_labels`` (a memmap is never resident).
    ``distance_to_lumen`` is sampled at the cell's coarse centroid voxel — physical-
    exact via ``sampling = spacing * N`` (no extra scaling).

    APPROXIMATION NOTES:
    * Cavity detection is coarser: a lumen narrower than ``N`` voxels (or a thin
      tissue wall that downsampling bridges shut) can be missed or merged — at large
      ``N`` the lumen geometry is only resolved to ``N``-voxel blocks.
    * ``lumen_facing_fraction`` is approximated on the COARSE grid: the fraction of the
      cell's COARSE surface voxels (the coarse foreground block(s) the cell maps into,
      minus their 6-erosion) that are directly adjacent to a lumen voxel. With a cell
      occupying only a handful of coarse voxels this is a crude footprint estimate;
      treat it as a low-resolution lumen-lining flag, not a precise area share.
    """
    from scipy.ndimage import (
        distance_transform_edt, binary_fill_holes, label as cc_label,
        binary_dilation, binary_erosion,
    )
    from packages.epicycle_scaledseg.core.field_downsample import coarse_foreground

    N = int(downsample)
    n = len(df)
    if n == 0:
        for c in LUMEN_COLUMNS:
            df[c] = pd.Series(dtype=np.float64)
        return df
    if "label" not in df.columns:
        for c in LUMEN_COLUMNS:
            df[c] = np.nan
        return df

    dist_lumen = np.full(n, np.nan, dtype=np.float64)
    facing_frac = np.full(n, np.nan, dtype=np.float64)

    sp = (tuple(float(s) for s in spacing) if spacing is not None else None)
    sp_vec = np.asarray(sp, dtype=np.float64) if sp is not None else np.ones(3)
    coarse_sampling = tuple(float(s) * N for s in (sp if sp is not None else (1.0, 1.0, 1.0)))

    cfg = coarse_foreground(full_labels, N)            # bool coarse foreground
    if not cfg.any():
        df["distance_to_lumen"] = dist_lumen
        df["lumen_facing_fraction"] = facing_frac
        return df

    # --- coarse lumen mask: enclosed cavities minus border-touching ---------------
    filled = binary_fill_holes(cfg)
    cavities = filled & ~cfg
    if not cavities.any():
        df["distance_to_lumen"] = dist_lumen
        df["lumen_facing_fraction"] = facing_frac
        return df
    cav_lab, n_cav = cc_label(cavities)
    if n_cav == 0:
        df["distance_to_lumen"] = dist_lumen
        df["lumen_facing_fraction"] = facing_frac
        return df
    border_ids = set()
    for face in (cav_lab[0], cav_lab[-1],
                 cav_lab[:, 0], cav_lab[:, -1],
                 cav_lab[:, :, 0], cav_lab[:, :, -1]):
        border_ids.update(np.unique(face).tolist())
    border_ids.discard(0)
    lumen = cavities.copy()
    if border_ids:
        lumen &= ~np.isin(cav_lab, list(border_ids))
    if not lumen.any():
        df["distance_to_lumen"] = dist_lumen
        df["lumen_facing_fraction"] = facing_frac
        return df

    # distance field to the nearest lumen voxel (physical via coarse_sampling).
    dist_to_lumen_field = distance_transform_edt(~lumen, sampling=coarse_sampling).astype(np.float64)

    # voxels directly adjacent to lumen (6-connected) — the coarse lumen lining.
    struct6 = np.zeros((3, 3, 3), dtype=bool)
    struct6[1, 1, 1] = True
    struct6[0, 1, 1] = struct6[2, 1, 1] = True
    struct6[1, 0, 1] = struct6[1, 2, 1] = True
    struct6[1, 1, 0] = struct6[1, 1, 2] = True
    lumen_adjacent = binary_dilation(lumen, structure=struct6) & ~lumen

    cshape = np.asarray(cfg.shape, dtype=np.int64)
    cen_cols = ["centroid-0", "centroid-1", "centroid-2"]
    have_cen = all(c in df.columns for c in cen_cols)
    centroids = (df[cen_cols].values.astype(np.float64) if have_cen else None)
    label_ids = df["label"].values.astype(np.int64)

    # the cell's coarse footprint is the foreground blocks it occupies — recover it by
    # mapping each cell's full-res bbox (df bbox-* cols) into the coarse grid, masked
    # to the coarse foreground (so the footprint is the cell's coarse tissue voxels).
    have_bbox = all(f"bbox-{k}" in df.columns for k in range(6))
    bb = (df[[f"bbox-{k}" for k in range(6)]].values.astype(np.int64) if have_bbox else None)

    for idx in range(n):
        if centroids is None:
            continue
        cen_phys = centroids[idx]
        cen_vox = cen_phys / np.where(sp_vec > 0, sp_vec, 1.0)
        ci = np.clip(np.floor(cen_vox / N).astype(np.int64), 0, cshape - 1)
        cz_i, cy_i, cx_i = int(ci[0]), int(ci[1]), int(ci[2])
        dist_lumen[idx] = float(dist_to_lumen_field[cz_i, cy_i, cx_i])

        # lumen_facing_fraction on the COARSE grid: the cell's coarse footprint (its
        # full-res bbox mapped to coarse blocks, intersected with the coarse fg) ->
        # its surface = footprint minus a 6-erosion -> fraction adjacent to lumen.
        if bb is not None:
            z0, y0, x0 = bb[idx, 0] // N, bb[idx, 1] // N, bb[idx, 2] // N
            z1 = min((bb[idx, 3] - 1) // N + 1, cshape[0])
            y1 = min((bb[idx, 4] - 1) // N + 1, cshape[1])
            x1 = min((bb[idx, 5] - 1) // N + 1, cshape[2])
            sl = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
            foot = cfg[sl]
            if foot.any():
                surface = foot & ~binary_erosion(foot, structure=struct6, border_value=0)
                n_surf = int(surface.sum())
                if n_surf > 0:
                    facing = surface & lumen_adjacent[sl]
                    facing_frac[idx] = float(facing.sum()) / float(n_surf)

    df["distance_to_lumen"] = dist_lumen
    df["lumen_facing_fraction"] = facing_frac
    return df


def _binary_erosion_local(mask: np.ndarray) -> np.ndarray:
    """One-step 6-connected binary erosion of a boolean crop (lumen surface helper).

    A tiny local wrapper so the lumen surface shell = ``mask & ~erosion(mask)``
    picks the voxels that border background within the crop. 6-connected (face
    neighbours only) to match the lumen-adjacency dilation.
    """
    from scipy.ndimage import binary_erosion
    struct6 = np.zeros((3, 3, 3), dtype=bool)
    struct6[1, 1, 1] = True
    struct6[0, 1, 1] = struct6[2, 1, 1] = True
    struct6[1, 0, 1] = struct6[1, 2, 1] = True
    struct6[1, 1, 0] = struct6[1, 1, 2] = True
    return binary_erosion(mask, structure=struct6, border_value=0)


# Wave D — signed-frame families. They all consume the per-cell OUTWARD normal
# (the ``tissue_outward_*`` columns Wave C / :func:`compute_tissue_depth` writes)
# to split a direction-bearing quantity into APICAL (aligned with outward) /
# BASAL (anti-aligned) / LATERAL (perpendicular) parts. A direction's cosine with
# the outward vector is classed apical when ``cos >= +COS_THRESH``, basal when
# ``cos <= -COS_THRESH``, lateral otherwise (the equatorial band). Gated on the
# outward normal existing (Wave C ran) AND the per-cell ``tissue_outward_confidence``
# being trustworthy (a flat-core cell has an ambiguous outward direction -> NaN).
_APICAL_BASAL_COS_THRESH = 0.5      # cone half-angle 60° for apical/basal; rest lateral
_OUTWARD_CONF_FLOOR = 0.2           # below this the outward direction is too noisy -> NaN

# Wave D built the first 4 directional-contact columns; Mop-up M
# adds the 5 contact-direction-cloud columns (dominant axis alignment, hemisphere
# balance, direction entropy, sector count, dominant neighbour fraction),
# completing the family. Module constant so the exact-set family test +
# feature_stage registry stay in lockstep.
DIRECTIONAL_CONTACT_COLUMNS = (
    # Wave D (original 4).
    "apical_contact_fraction", "basal_contact_fraction",
    "lateral_contact_fraction", "contact_polarity_bias",
    # Mop-up M — directional contact-cloud structure.
    "dominant_contact_axis", "contact_hemisphere_balance",
    "contact_direction_entropy", "contact_sector_count",
    "dominant_contact_neighbor_fraction",
)

# Number of azimuthal sectors (around the outward axis) the contact directions are
# binned into for ``contact_sector_count`` / ``contact_direction_entropy``. 8 ~=
# 45° wedges: coarse enough to be robust to a few stray voxels, fine enough to
# distinguish a one-sided contact band from an all-around ring.
_CONTACT_AZIMUTH_NSECTORS = 8

SURFACE_PARTITION_COLUMNS = (
    "apical_surface_area", "basal_surface_area", "lateral_surface_area",
    "apical_surface_fraction", "basal_surface_fraction", "lateral_surface_fraction",
    "apical_basal_surface_ratio",
    # Wave I cross ratios.
    "lateral_to_apical_surface_ratio", "lateral_to_basal_surface_ratio",
    # Wave J REMAINDER — apical/basal footprint contour metrics (the cell's
    # top-/bottom-facing outline projected onto the plane ⊥ the outward normal).
    "apical_perimeter", "apical_circularity", "apical_aspect_ratio",
    "apical_compactness",
    "basal_perimeter", "basal_circularity", "basal_aspect_ratio",
    "basal_compactness",
    # Wave J REMAINDER — per-partition surface CURVATURE (size-normalised mean
    # curvature aggregated over each patch's vertices) + per-partition ROUGHNESS
    # (the patch's non-planarity = area-weighted RMS face-normal deviation).
    "apical_curvature_mean", "apical_curvature_std",
    "basal_curvature_mean", "lateral_curvature_mean",
    "apical_surface_roughness", "basal_surface_roughness",
)


def _cell_outward_vectors(df: pd.DataFrame):
    """The per-cell OUTWARD unit normal + a trust mask from the Wave C columns.

    Returns ``(outward, trusted)`` where ``outward`` is an ``(n, 3)`` float array of
    the ``tissue_outward_(z,y,x)`` ZYX components (NaN rows where Wave C left them
    NaN) and ``trusted`` is an ``(n,)`` bool: the outward vector is finite, has unit
    norm (>0), AND its ``tissue_outward_confidence`` clears :data:`_OUTWARD_CONF_FLOOR`.
    ``(None, None)`` when the outward columns are absent (Wave C did not run) so the
    caller can SKIP and leave its columns absent (the default-off contract).
    """
    cols = ("tissue_outward_z", "tissue_outward_y", "tissue_outward_x")
    if not all(c in df.columns for c in cols):
        return None, None
    outward = df[list(cols)].values.astype(np.float64)
    finite = np.isfinite(outward).all(axis=1)
    norms = np.linalg.norm(np.where(np.isfinite(outward), outward, 0.0), axis=1)
    conf = (df["tissue_outward_confidence"].values.astype(np.float64)
            if "tissue_outward_confidence" in df.columns
            else np.ones(len(df), dtype=np.float64))
    trusted = finite & (norms > 1e-9) & np.isfinite(conf) & (conf >= _OUTWARD_CONF_FLOOR)
    return outward, trusted


def compute_directional_contact(
    full_labels: np.ndarray,
    df: pd.DataFrame,
    spacing: Optional[Tuple[float, ...]] = None,
    adjacency: Optional[list] = None,
    max_radius: int = 5,
    slices: Optional[list] = None,
) -> pd.DataFrame:
    """Directional decomposition of a cell's CONTACTS in its apical/basal/lateral
    frame.

    M4 (:func:`compute_contact_geometry`) says HOW MUCH contact a cell has; this
    says WHERE around the cell those contacts live, in the SIGNED apical/basal
    frame the Wave C outward normal defines. For every contact voxel we take the
    direction from the cell centroid to that voxel and class it by its cosine with
    the cell's ``tissue_outward`` vector: APICAL (toward the surface,
    ``cos >= +0.5``), BASAL (into the tissue, ``cos <= -0.5``), or LATERAL (the
    equatorial side band). The per-direction fractions of the cell's total contact
    voxels are emitted, plus the apical−basal polarity bias.

    A GLOBAL pass (a per-cell bbox crop can't see its neighbours) — run ONCE on the
    full label volume AFTER :func:`compute_tissue_depth` (it needs the
    ``tissue_outward_*`` columns) inside the ``tissue_depth=True`` block. Reuses the
    SAME graduated-dilation contact recovery as M4
    (:func:`_contact_voxels_per_neighbor`, radius 1..``max_radius``) so the contact
    set is consistent with the M3 adjacency graph. ``adjacency`` defaults to
    ``df.attrs["_adjacency_list"]``.

    Features added (9 — Wave D built the first 4; Mop-up M added the 5
    contact-direction-cloud columns, completing the family):
        ``apical_contact_fraction`` — fraction of the cell's contact voxels whose
            direction-from-centroid is APICAL (aligned with the outward normal)
            ∈ [0, 1]. Bounded.
        ``basal_contact_fraction`` — same, BASAL (anti-aligned, into the tissue)
            ∈ [0, 1]. Bounded.
        ``lateral_contact_fraction`` — same, LATERAL (the perpendicular side band)
            ∈ [0, 1]. Bounded. The three fractions SUM TO 1 for a cell with
            contacts + a trusted outward normal.
        ``contact_polarity_bias`` — ``apical_contact_fraction −
            basal_contact_fraction`` ∈ [-1, 1]: +1 = contacts pile on the apical
            side, −1 = on the basal side, 0 = balanced. Bounded (signed).
        ``dominant_contact_axis`` (Mop-up M) — ``|cos|`` between the cell's overall
            mean contact direction (the unit resultant of the per-voxel contact
            directions) and the outward normal ∈ [0, 1]: 1 = the contacts pile
            along the apical/basal AXIS, 0 = the dominant contact direction is
            lateral / equatorial. Bounded.
        ``contact_hemisphere_balance`` (Mop-up M) — ``1 − |apical_frac −
            basal_frac| / max(apical_frac + basal_frac, eps)`` over the AXIAL
            (non-lateral) contacts ∈ [0, 1]: 1 = the axial contacts are split
            evenly between the apical + basal hemispheres, → 0 = all on one
            hemisphere. Bounded. NaN when there are no axial contacts.
        ``contact_direction_entropy`` (Mop-up M) — the Shannon entropy (nats) of
            the contact-voxel AZIMUTH histogram around the outward axis
            (:data:`_CONTACT_AZIMUTH_NSECTORS` sectors), normalised to [0, 1] by
            ``log(nsectors)``: 1 = contacts spread evenly all around the cell, → 0
            = contacts concentrated in one azimuthal wedge (a one-sided band).
            Bounded.
        ``contact_sector_count`` (Mop-up M) — the number of OCCUPIED azimuthal
            sectors (of :data:`_CONTACT_AZIMUTH_NSECTORS`) that hold at least a
            small share of the contact voxels: a coarse "how many sides does the
            cell touch on" count. Unbounded (<= nsectors) count.
        ``dominant_contact_neighbor_fraction`` (Mop-up M) — the fraction of the
            cell's total contact voxels held by its SINGLE largest-contact
            neighbour ∈ [0, 1]: ~1 = one neighbour dominates (a merge-prone
            interface), low = contacts spread across many neighbours. Bounded.

    NaN (all nine) when the cell has NO contacts OR its outward normal is
    untrusted (low ``tissue_outward_confidence`` / NaN — the flat tissue core,
    where the apical/basal direction is meaningless). Absent entirely (columns not
    created) when Wave C did not run (no ``tissue_outward_*`` columns) — the
    default-off contract.

    Args:
        full_labels: the FULL 3D integer label volume (Z, Y, X). Background 0.
        df: assembled per-cell table (modified in-place). Needs ``label`` (to index
            cells), ``centroid-0/1/2`` (the ray origin), and the Wave C
            ``tissue_outward_*`` columns (the sign basis).
        spacing: voxel spacing ``(Z, Y, X)`` -> the contact-direction vectors are
            computed in PHYSICAL units (same frame as the centroids). ``None`` =
            voxel units.
        adjacency: the M3 edge list; ``None`` -> ``df.attrs["_adjacency_list"]``.
        max_radius: graduated-dilation radius (voxels) — MUST match the M3 /
            M4 ``max_radius`` so the contact set is consistent. Default 5.
        slices: OPTIONAL precomputed ``find_objects`` bbox list (S1 scaling hook) —
            indexed by ``label_id - 1``, supplied by the caller to skip a redundant
            full-volume ``find_objects(full_labels.astype(int32))`` (whose ``.astype``
            on a memmap forces a FULL RESIDENT copy -> OOM at scale). None => the
            original byte-identical internal scan; the per-cell contact crops are
            lazy on a memmap.

    Returns:
        The same DataFrame, with the 9 columns added — or unchanged when Wave C
        did not run (no outward normal to sign the frame).
    """
    from scipy.ndimage import find_objects

    n = len(df)
    if n == 0:
        for c in DIRECTIONAL_CONTACT_COLUMNS:
            df[c] = pd.Series(dtype=np.float64)
        return df
    if "label" not in df.columns:
        for c in DIRECTIONAL_CONTACT_COLUMNS:
            df[c] = np.nan
        return df

    outward, trusted = _cell_outward_vectors(df)
    if outward is None:
        # Wave C did not run -> no sign basis. SKIP (leave the columns absent) per
        # the default-off contract.
        return df

    cen_cols = ["centroid-0", "centroid-1", "centroid-2"]
    if not all(c in df.columns for c in cen_cols):
        for c in DIRECTIONAL_CONTACT_COLUMNS:
            df[c] = np.nan
        return df
    centroids = df[cen_cols].values.astype(np.float64)   # PHYSICAL when spacing set

    if adjacency is None:
        adjacency = df.attrs.get("_adjacency_list")
    nb_sets: dict = {}
    if adjacency:
        for e in adjacency:
            a = int(e["label_a"]); b = int(e["label_b"])
            nb_sets.setdefault(a, set()).add(b)
            nb_sets.setdefault(b, set()).add(a)

    sp = tuple(float(s) for s in spacing) if spacing is not None else (1.0, 1.0, 1.0)
    sp_vec = np.asarray(sp, dtype=np.float64)

    apical_frac = np.full(n, np.nan, dtype=np.float64)
    basal_frac = np.full(n, np.nan, dtype=np.float64)
    lateral_frac = np.full(n, np.nan, dtype=np.float64)
    polarity = np.full(n, np.nan, dtype=np.float64)
    # Mop-up M accumulators.
    dom_axis = np.full(n, np.nan, dtype=np.float64)
    hemi_balance = np.full(n, np.nan, dtype=np.float64)
    dir_entropy = np.full(n, np.nan, dtype=np.float64)
    sector_count = np.full(n, np.nan, dtype=np.float64)
    dom_nb_frac = np.full(n, np.nan, dtype=np.float64)

    # S1 scaling hook: reuse the caller's precomputed bboxes (off the memmap)
    # instead of a redundant full-volume find_objects + int32 copy (the ``.astype``
    # on a memmap forces a FULL RESIDENT copy -> OOM at scale). None => the original
    # byte-identical internal scan. The per-cell contact recovery
    # (``_contact_voxels_per_neighbor``) crops ``full_labels`` lazily, so with
    # ``slices`` supplied the full volume is never materialised.
    if slices is None:
        slices = find_objects(full_labels.astype(np.int32))
    label_ids = df["label"].values
    cos_t = _APICAL_BASAL_COS_THRESH
    eps = 1e-12
    nsect = _CONTACT_AZIMUTH_NSECTORS

    for idx in range(n):
        lid = int(label_ids[idx])
        if lid < 1 or lid > len(slices) or slices[lid - 1] is None:
            continue
        if not trusted[idx]:
            continue                          # untrusted outward normal -> NaN
        neighbor_ids = nb_sets.get(lid, set())
        if not neighbor_ids:
            continue                          # no contacts -> NaN

        contacts = _contact_voxels_per_neighbor(
            full_labels, lid, slices[lid - 1], neighbor_ids, max_radius=max_radius,
        )
        if not contacts:
            continue

        # stack every neighbour's contact voxels into one (M, 3) array (GLOBAL ZYX
        # voxel coords) -> physical coords (× spacing) for the centroid frame.
        all_coords = [c for c in contacts.values() if c.shape[0] > 0]
        if not all_coords:
            continue
        coords = np.vstack(all_coords).astype(np.float64)   # voxel ZYX
        coords_phys = coords * sp_vec                        # physical ZYX

        # per-neighbour contact size (for the dominant-neighbour fraction).
        nb_sizes = np.array([c.shape[0] for c in contacts.values() if c.shape[0] > 0],
                            dtype=np.float64)

        # direction from the cell centroid to each contact voxel (physical), unit.
        vecs = coords_phys - centroids[idx][None, :]
        vn = np.linalg.norm(vecs, axis=1)
        ok = vn > 1e-9
        if not ok.any():
            continue
        units = vecs[ok] / vn[ok][:, None]
        ow = outward[idx]                                    # already unit (trusted)
        cos = units @ ow                                     # (M_ok,) in [-1, 1]

        m = cos.shape[0]
        n_ap = int(np.count_nonzero(cos >= cos_t))
        n_ba = int(np.count_nonzero(cos <= -cos_t))
        n_lat = m - n_ap - n_ba
        af = n_ap / m
        bf = n_ba / m
        lf = n_lat / m
        apical_frac[idx] = af
        basal_frac[idx] = bf
        lateral_frac[idx] = lf
        polarity[idx] = af - bf

        # --- Mop-up M — contact-direction-cloud structure -------------------
        # dominant contact axis: |cos| of the MEAN contact direction with outward.
        mean_dir = units.mean(axis=0)
        md_n = float(np.linalg.norm(mean_dir))
        if md_n > eps:
            dom_axis[idx] = float(abs(np.dot(mean_dir / md_n, ow)))
        else:
            dom_axis[idx] = 0.0
        # hemisphere balance over the AXIAL (apical+basal) contacts only.
        ax_tot = n_ap + n_ba
        if ax_tot > 0:
            hemi_balance[idx] = 1.0 - abs(n_ap - n_ba) / float(ax_tot)
        # azimuth histogram around the outward axis: build an orthonormal (u, v)
        # basis of the plane ⊥ outward, project each contact direction, take the
        # angle, bin into nsect sectors -> entropy + occupied-sector count.
        seed = (np.array([1.0, 0.0, 0.0]) if abs(ow[0]) < 0.9
                else np.array([0.0, 1.0, 0.0]))
        u = seed - ow * float(np.dot(seed, ow))
        un = float(np.linalg.norm(u))
        if un > eps:
            u = u / un
            v = np.cross(ow, u)
            au = units @ u
            av = units @ v
            ang = np.arctan2(av, au)                          # (-pi, pi]
            bins = np.clip(((ang + np.pi) / (2.0 * np.pi) * nsect).astype(np.int64),
                           0, nsect - 1)
            hist = np.bincount(bins, minlength=nsect).astype(np.float64)
            p = hist / max(hist.sum(), eps)
            nz = p[p > eps]
            ent = float(-(nz * np.log(nz)).sum())
            dir_entropy[idx] = ent / np.log(nsect) if nsect > 1 else 0.0
            # a sector is "occupied" if it holds at least ~half the uniform share.
            occ_thresh = 0.5 / nsect
            sector_count[idx] = float(int(np.count_nonzero(p >= occ_thresh)))
        # dominant neighbour fraction: largest single neighbour's contact share.
        tot_nb = float(nb_sizes.sum())
        if tot_nb > eps:
            dom_nb_frac[idx] = float(nb_sizes.max() / tot_nb)

    df["apical_contact_fraction"] = apical_frac
    df["basal_contact_fraction"] = basal_frac
    df["lateral_contact_fraction"] = lateral_frac
    df["contact_polarity_bias"] = polarity
    df["dominant_contact_axis"] = dom_axis
    df["contact_hemisphere_balance"] = hemi_balance
    df["contact_direction_entropy"] = dir_entropy
    df["contact_sector_count"] = sector_count
    df["dominant_contact_neighbor_fraction"] = dom_nb_frac
    return df


def _footprint_metrics(verts_xyz: np.ndarray, outward: np.ndarray):
    """2D footprint shape of a surface patch projected onto the plane ⊥ ``outward``.

    Wave J helper. Given the 3D physical vertices belonging to a surface
    PARTITION (the apical or basal face set) and the cell's unit outward normal,
    project them into the 2D plane perpendicular to ``outward`` (the footprint /
    "top-down" outline), take the 2D convex hull, and return the footprint shape:

        ``(perimeter, circularity, aspect_ratio, compactness)``

    where ``circularity = compactness = 4πA / max(P², eps)`` (the isoperimetric
    ratio: 1 for a perfect disc, → 0 for a jagged / elongated outline) and
    ``aspect_ratio = √(λ_major / λ_minor)`` of the 2D point covariance (>= 1, the
    footprint elongation). Returns ``(nan, nan, nan, nan)`` for a degenerate patch
    (< 3 distinct planar points / collinear / zero-area hull).

    Physical units follow the vertices (µm when the mesh used physical spacing).
    """
    eps = 1e-12
    if verts_xyz.shape[0] < 3:
        return np.nan, np.nan, np.nan, np.nan
    ow = np.asarray(outward, dtype=np.float64)
    nrm = float(np.linalg.norm(ow))
    if nrm <= eps:
        return np.nan, np.nan, np.nan, np.nan
    ow = ow / nrm
    # Build an orthonormal basis (u, v) of the plane ⊥ outward.
    seed = np.array([1.0, 0.0, 0.0]) if abs(ow[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = seed - ow * float(np.dot(seed, ow))
    un = float(np.linalg.norm(u))
    if un <= eps:
        return np.nan, np.nan, np.nan, np.nan
    u = u / un
    v = np.cross(ow, u)
    pts2d = np.column_stack([verts_xyz @ u, verts_xyz @ v])   # (M, 2)
    # de-duplicate near-identical projected points so the hull is well posed.
    uniq = np.unique(np.round(pts2d, 6), axis=0)
    if uniq.shape[0] < 3:
        return np.nan, np.nan, np.nan, np.nan

    perimeter = circ = aspect = compact = np.nan
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(uniq)
        area2d = float(hull.volume)            # 2D hull "volume" == area
        perimeter = float(hull.area)           # 2D hull "area" == perimeter
        if perimeter > eps and area2d > eps:
            circ = 4.0 * np.pi * area2d / (perimeter ** 2)
            circ = float(min(circ, 1.0))       # clamp tiny numerical overshoot
            compact = circ                     # footprint isoperimetric compactness
    except Exception:
        return np.nan, np.nan, np.nan, np.nan

    # aspect ratio from the 2D point covariance (independent of the hull).
    cov = np.cov(uniq.T)
    if np.all(np.isfinite(cov)):
        evals = np.linalg.eigvalsh(cov)
        evals = np.clip(evals, 0.0, None)
        if evals[-1] > eps and evals[0] > eps:
            aspect = float(np.sqrt(evals[-1] / evals[0]))
        elif evals[-1] > eps:
            aspect = np.inf                    # degenerate line -> infinite aspect
    return perimeter, circ, aspect, compact


def compute_surface_partition(
    full_labels: np.ndarray,
    df: pd.DataFrame,
    spacing: Optional[Tuple[float, ...]] = None,
    slices: Optional[list] = None,
) -> pd.DataFrame:
    """Partition each cell's SURFACE MESH into apical / basal / lateral patches in
    its outward frame + summarise each patch's area.

    Distinct from the cross-section family (which measures interior AREA on image-Z
    slices): this measures the BOUNDARY itself. For each cell we marching-cubes the
    bbox crop (the SAME mesh :func:`_object_surface_mesh` / :func:`compute_surface_features`
    build), compute each triangular face's OUTWARD normal (the cross product of two
    edges, oriented away from the cell interior using the face centroid vs the cell
    centroid), and class each face APICAL / BASAL / LATERAL by the cosine of its
    normal with the cell's Wave C ``tissue_outward`` vector. Each patch's physical
    area is the sum of its faces' triangle areas.

    A per-cell mesh pass, but it READS the Wave C ``tissue_outward_*`` columns, so —
    like :func:`compute_directional_contact` — it is wired into the
    ``tissue_depth=True`` global block AFTER :func:`compute_tissue_depth`. It does
    NOT need the adjacency graph.

    Features added (7):
        ``apical_surface_area`` / ``basal_surface_area`` / ``lateral_surface_area``
            — physical area (µm² when ``spacing`` given) of the apical / basal /
            lateral surface patch. Unbounded >= 0.
        ``apical_surface_fraction`` / ``basal_surface_fraction`` /
            ``lateral_surface_fraction`` — each patch's area / the cell's TOTAL mesh
            area ∈ [0, 1] (the three SUM TO 1). Bounded.
        ``apical_basal_surface_ratio`` — ``apical_surface_area /
            max(basal_surface_area, eps)``: > 1 = more apical-facing membrane than
            basal-facing. Unbounded >= 0.
        ``lateral_to_apical_surface_ratio`` / ``lateral_to_basal_surface_ratio``
            (Wave I) — ``lateral_surface_area / max(apical_surface_area,
            eps)`` and ``/ max(basal_surface_area, eps)``: how side-wall-dominated
            the cell is relative to its top-facing / bottom-facing cap. Large =
            mostly lateral membrane (columnar), small = a flat cap-dominated cell.
            Pure derivations of the three patch areas. Unbounded >= 0.

    Wave J REMAINDER (14 more — the footprint contour + per-partition shape):
        ``apical_perimeter`` / ``apical_circularity`` / ``apical_aspect_ratio`` /
            ``apical_compactness`` (+ the ``basal_*`` four) — the apical (resp.
            basal) FACE SET projected onto the plane ⊥ the outward normal (the
            cell's top-down / bottom-up footprint), then its 2D convex-hull
            perimeter, isoperimetric circularity ``4πA/P²`` ∈ [0, 1] (== compactness,
            1 = a round cap, → 0 = jagged / elongated), and covariance aspect ratio
            (>= 1, footprint elongation). Physical when ``spacing`` given. NaN for a
            degenerate / collinear footprint (e.g. a cell with no apical patch).
        ``apical_curvature_mean`` / ``apical_curvature_std`` /
            ``basal_curvature_mean`` / ``lateral_curvature_mean`` — the
            size-normalised discrete mean curvature (``H / R_equiv``, the SAME B5
            convention as :func:`compute_curvature_features`) averaged over the
            vertices of each partition's faces: whether the top / bottom / side
            surface is flat (~0), dome-like (> 0), or pit-like (< 0). Dimensionless.
        ``apical_surface_roughness`` / ``basal_surface_roughness`` — each patch's
            NON-PLANARITY: the area-weighted RMS deviation (radians) of its face
            normals from the patch mean normal (0 = a perfectly flat cap; larger =
            a bumpy / curved patch). Dimensionless.

    NaN (the area / ratio columns) when the mesh is degenerate (a tiny cell) OR the
    outward normal is untrusted; the Wave J footprint/curvature/roughness columns are
    NaN additionally when the relevant partition is EMPTY (no apical/basal faces).
    Original-9-NaN note (kept): all area/fraction/ratio columns are NaN when the
    mesh is degenerate (a tiny cell) OR the outward normal is
    untrusted (low confidence / NaN). Absent entirely when Wave C did not run (no
    ``tissue_outward_*``) — the default-off contract.

    Args:
        full_labels: the FULL 3D integer label volume (Z, Y, X). Background 0.
        df: assembled per-cell table (modified in-place). Needs ``label`` and the
            Wave C ``tissue_outward_*`` columns.
        spacing: voxel spacing ``(Z, Y, X)`` -> the mesh (and so the patch areas)
            are in PHYSICAL units. ``None`` = voxel units.
        slices: OPTIONAL precomputed ``find_objects`` bbox list (S1 scaling hook) —
            indexed by ``label_id - 1``, supplied by the caller to skip a redundant
            full-volume ``find_objects(full_labels.astype(int32))`` (whose ``.astype``
            on a memmap forces a FULL RESIDENT copy -> OOM at scale). None => the
            original byte-identical internal scan; the per-cell mesh crops are lazy
            on a memmap.

    Returns:
        The same DataFrame with the 7 new columns — or unchanged when Wave C did
        not run.
    """
    from scipy.ndimage import find_objects

    n = len(df)
    if n == 0:
        for c in SURFACE_PARTITION_COLUMNS:
            df[c] = pd.Series(dtype=np.float64)
        return df
    if "label" not in df.columns:
        for c in SURFACE_PARTITION_COLUMNS:
            df[c] = np.nan
        return df

    outward, trusted = _cell_outward_vectors(df)
    if outward is None:
        return df                              # Wave C did not run -> SKIP

    sp = tuple(float(s) for s in spacing) if spacing is not None else (1.0, 1.0, 1.0)

    apical_a = np.full(n, np.nan, dtype=np.float64)
    basal_a = np.full(n, np.nan, dtype=np.float64)
    lateral_a = np.full(n, np.nan, dtype=np.float64)
    apical_f = np.full(n, np.nan, dtype=np.float64)
    basal_f = np.full(n, np.nan, dtype=np.float64)
    lateral_f = np.full(n, np.nan, dtype=np.float64)
    ab_ratio = np.full(n, np.nan, dtype=np.float64)
    la_ratio = np.full(n, np.nan, dtype=np.float64)   # Wave I lateral/apical
    lb_ratio = np.full(n, np.nan, dtype=np.float64)   # Wave I lateral/basal
    # Wave J REMAINDER accumulators.
    ap_perim = np.full(n, np.nan, dtype=np.float64)
    ap_circ = np.full(n, np.nan, dtype=np.float64)
    ap_aspect = np.full(n, np.nan, dtype=np.float64)
    ap_compact = np.full(n, np.nan, dtype=np.float64)
    ba_perim = np.full(n, np.nan, dtype=np.float64)
    ba_circ = np.full(n, np.nan, dtype=np.float64)
    ba_aspect = np.full(n, np.nan, dtype=np.float64)
    ba_compact = np.full(n, np.nan, dtype=np.float64)
    ap_curv_mean = np.full(n, np.nan, dtype=np.float64)
    ap_curv_std = np.full(n, np.nan, dtype=np.float64)
    ba_curv_mean = np.full(n, np.nan, dtype=np.float64)
    lat_curv_mean = np.full(n, np.nan, dtype=np.float64)
    ap_rough = np.full(n, np.nan, dtype=np.float64)
    ba_rough = np.full(n, np.nan, dtype=np.float64)

    # S1 scaling hook: reuse the caller's precomputed bboxes (off the memmap)
    # instead of a redundant full-volume find_objects + int32 copy (the ``.astype``
    # on a memmap forces a FULL RESIDENT copy -> OOM at scale). None => the original
    # byte-identical internal scan. The per-cell crops (``full_labels[slices[...]]``)
    # are lazy on a memmap, so with ``slices`` supplied the full volume is never
    # materialised.
    if slices is None:
        slices = find_objects(full_labels.astype(np.int32))
    label_ids = df["label"].values
    cos_t = _APICAL_BASAL_COS_THRESH
    eps = 1e-12

    for idx in range(n):
        lid = int(label_ids[idx])
        if lid < 1 or lid > len(slices) or slices[lid - 1] is None:
            continue
        if not trusted[idx]:
            continue
        crop_mask = full_labels[slices[lid - 1]] == lid
        if not crop_mask.any():
            continue

        mesh = _object_surface_mesh(crop_mask, sp)
        if mesh is None:
            continue
        verts, faces = mesh
        tri = verts[faces]                                  # (F, 3, 3) physical ZYX

        # face area + (un-normalised) normal = 0.5 * |e0 x e1|; n = e0 x e1.
        e0 = tri[:, 1] - tri[:, 0]
        e1 = tri[:, 2] - tri[:, 0]
        cross = np.cross(e0, e1)                            # (F, 3), ‖‖ = 2*area
        face_area = 0.5 * np.linalg.norm(cross, axis=1)     # (F,)
        valid = face_area > eps
        if not valid.any():
            continue

        face_centroid = tri.mean(axis=1)                    # (F, 3) physical ZYX
        # ORIENT each normal OUTWARD: marching_cubes winding is consistent but we
        # don't rely on it — point the normal away from the MESH centroid (the cell
        # interior proxy). Flip any normal whose dot with (face_centroid - mesh
        # centroid) is negative.
        mesh_centroid = verts.mean(axis=0)
        out_ref = face_centroid - mesh_centroid[None, :]
        nrm = cross.copy()
        flip = np.einsum("ij,ij->i", nrm, out_ref) < 0.0
        nrm[flip] *= -1.0
        nn = np.linalg.norm(nrm, axis=1)
        ok = valid & (nn > eps)
        if not ok.any():
            continue
        unit_n = np.zeros_like(nrm)
        unit_n[ok] = nrm[ok] / nn[ok][:, None]

        ow = outward[idx]                                   # unit (trusted)
        cos = unit_n @ ow                                   # (F,)
        a = face_area
        ap_mask = ok & (cos >= cos_t)
        ba_mask = ok & (cos <= -cos_t)
        lat_mask = ok & ~ap_mask & ~ba_mask

        ap = float(a[ap_mask].sum())
        ba = float(a[ba_mask].sum())
        lat = float(a[lat_mask].sum())
        total = ap + ba + lat
        apical_a[idx] = ap
        basal_a[idx] = ba
        lateral_a[idx] = lat
        if total > eps:
            apical_f[idx] = ap / total
            basal_f[idx] = ba / total
            lateral_f[idx] = lat / total
        ab_ratio[idx] = ap / max(ba, eps)
        # Wave I cross ratios (pure derivations of the patch areas).
        la_ratio[idx] = lat / max(ap, eps)
        lb_ratio[idx] = lat / max(ba, eps)

        # --- Wave J REMAINDER — footprint contour + per-partition shape ---
        # Footprint outline of the apical / basal face set, projected onto the
        # plane ⊥ the outward normal (the cell's top-down / bottom-up outline).
        ow_vec = ow
        if ap_mask.any():
            ap_verts = verts[faces[ap_mask]].reshape(-1, 3)
            p, c2, ar, cm = _footprint_metrics(ap_verts, ow_vec)
            ap_perim[idx] = p; ap_circ[idx] = c2
            ap_aspect[idx] = ar; ap_compact[idx] = cm
        if ba_mask.any():
            ba_verts = verts[faces[ba_mask]].reshape(-1, 3)
            p, c2, ar, cm = _footprint_metrics(ba_verts, ow_vec)
            ba_perim[idx] = p; ba_circ[idx] = c2
            ba_aspect[idx] = ar; ba_compact[idx] = cm

        # Per-partition CURVATURE — the size-normalised discrete mean curvature
        # (same B5 convention as compute_curvature_features: H / R_equiv), then
        # average over the VERTICES of each partition's faces. R_equiv from the
        # cell's voxel volume so the curvature is dimensionless + size-invariant.
        obj_vox = float(np.count_nonzero(crop_mask))
        phys_vol = obj_vox * float(sp[0] * sp[1] * sp[2])
        r_equiv = (3.0 * phys_vol / (4.0 * np.pi)) ** (1.0 / 3.0) if phys_vol > 0 else 0.0
        if r_equiv > eps:
            H = _discrete_mean_curvature(verts, faces)
            if H.size == verts.shape[0] and np.all(np.isfinite(H)):
                Hn = H / r_equiv
                # which vertices belong to each partition (a vertex is in the
                # partition if ANY of its faces is in that partition).
                ap_v = np.unique(faces[ap_mask].ravel()) if ap_mask.any() else np.array([], int)
                ba_v = np.unique(faces[ba_mask].ravel()) if ba_mask.any() else np.array([], int)
                lat_v = np.unique(faces[lat_mask].ravel()) if lat_mask.any() else np.array([], int)
                if ap_v.size:
                    ap_curv_mean[idx] = float(Hn[ap_v].mean())
                    ap_curv_std[idx] = float(Hn[ap_v].std()) if ap_v.size >= 2 else 0.0
                if ba_v.size:
                    ba_curv_mean[idx] = float(Hn[ba_v].mean())
                if lat_v.size:
                    lat_curv_mean[idx] = float(Hn[lat_v].mean())

        # Per-partition ROUGHNESS — the patch's NON-PLANARITY: the area-weighted
        # RMS deviation (in radians) of each face normal from the patch's mean
        # (area-weighted) normal. 0 = a perfectly flat cap; larger = a bumpy /
        # curved patch. Dimensionless (an angle), so directly comparable.
        def _patch_roughness(mask):
            if not mask.any():
                return np.nan
            w = a[mask]
            wsum = float(w.sum())
            if wsum <= eps:
                return np.nan
            nv = unit_n[mask]
            mean_n = (nv * w[:, None]).sum(axis=0)
            mn = float(np.linalg.norm(mean_n))
            if mn <= eps:
                return np.nan
            mean_n = mean_n / mn
            d = np.clip(nv @ mean_n, -1.0, 1.0)
            ang = np.arccos(d)                          # per-face deviation (rad)
            return float(np.sqrt((w * ang ** 2).sum() / wsum))
        ap_rough[idx] = _patch_roughness(ap_mask)
        ba_rough[idx] = _patch_roughness(ba_mask)

    df["apical_surface_area"] = apical_a
    df["basal_surface_area"] = basal_a
    df["lateral_surface_area"] = lateral_a
    df["apical_surface_fraction"] = apical_f
    df["basal_surface_fraction"] = basal_f
    df["lateral_surface_fraction"] = lateral_f
    df["apical_basal_surface_ratio"] = ab_ratio
    df["lateral_to_apical_surface_ratio"] = la_ratio   # Wave I
    df["lateral_to_basal_surface_ratio"] = lb_ratio    # Wave I
    # Wave J REMAINDER — footprint contour + per-partition curvature/roughness.
    df["apical_perimeter"] = ap_perim
    df["apical_circularity"] = ap_circ
    df["apical_aspect_ratio"] = ap_aspect
    df["apical_compactness"] = ap_compact
    df["basal_perimeter"] = ba_perim
    df["basal_circularity"] = ba_circ
    df["basal_aspect_ratio"] = ba_aspect
    df["basal_compactness"] = ba_compact
    df["apical_curvature_mean"] = ap_curv_mean
    df["apical_curvature_std"] = ap_curv_std
    df["basal_curvature_mean"] = ba_curv_mean
    df["lateral_curvature_mean"] = lat_curv_mean
    df["apical_surface_roughness"] = ap_rough
    df["basal_surface_roughness"] = ba_rough
    return df


# --------------------------------------------------------------------------- #
# Wave H — signed-frame REMAINDERS that consume the Wave C outward normal.
#
# Two GLOBAL passes (they read the ``tissue_outward_*`` columns, so they ride the
# ``tissue_depth=True`` block AFTER compute_tissue_depth, exactly like the Wave D
# directional-contact / surface-partition passes):
#
#   * ma_apical_basal (remainder) — orients the M1 major-axis cross-section
#     profile by the outward normal and splits it apical-third vs basal-third, so
#     the tilt-robust profile becomes a SIGNED apical/basal taper (M1 only emitted
#     the unsigned min/max/range).
#   * apical_junction — the classic epithelial polygon topology read off the
#     cell's LATERAL contacts (the apical-junctional belt): how many distinct lateral
#     neighbours (polygon side count, ~6 in ordered tissue), the total lateral
#     contact perimeter, and the irregularity (cv) of the lateral patch sizes.
#
# Default-off byte-identical: both SKIP (leave their columns absent) when Wave C did
# not run (no outward normal), mirroring the Wave D contract.
# --------------------------------------------------------------------------- #

MA_APICAL_BASAL_COLUMNS = (
    "ma_apical_area", "ma_basal_area", "ma_apical_basal_ratio",
    "ma_apical_taper", "ma_basal_taper",
)

APICAL_JUNCTION_COLUMNS = (
    "apical_polygon_sides", "apical_junction_length",
    "apical_polygon_irregularity",
    # Wave J REMAINDER — the lateral-belt vertex/edge stats + the explicit
    # apical FOOTPRINT polygon (the cell's apical-cap outline) shape descriptors.
    "apical_junction_vertex_count", "apical_junction_edge_length_cv",
    "apical_vertex_angle_cv",
    "apical_polygon_area", "apical_polygon_perimeter",
    "apical_shape_index", "apical_polygon_convexity",
)


def _apical_footprint_polygon(coords_apical: np.ndarray, outward: np.ndarray,
                              pixel_area: float):
    """Apical FOOTPRINT polygon metrics from the cell's apical-cap voxel cloud.

    Wave J helper. Projects the cell's apical-third voxel coordinates
    (``coords_apical``, physical ZYX) onto the plane ⊥ the unit outward normal, takes
    the 2D convex hull (the apical outline polygon), and returns::

        (vertex_count, area, perimeter, shape_index, convexity, angle_cv)

    where ``shape_index = perimeter / sqrt(area)`` (the vertex-model order parameter:
    ~3.72 for a regular hexagon, ~3.54 for a circle, larger for elongated / jagged
    outlines), ``convexity = hull_area / point_cloud_extent_area`` clamped to [0, 1]
    (1 = a convex apical face; here the hull IS the outline so we report the hull's
    own solidity proxy via the alpha-free area-vs-bbox ratio → see below), and
    ``angle_cv`` is the coefficient of variation of the hull's interior vertex angles
    (0 = a regular polygon). Returns all-NaN for a degenerate (< 3 distinct planar
    points / collinear / zero-area) footprint.

    ``pixel_area`` (physical area of one projected pixel) is unused for the hull
    metrics (the hull works in continuous projected coordinates) but kept in the
    signature for callers that want a rasterised area; the returned area is the
    continuous hull area in the same physical units as ``coords_apical``.
    """
    eps = 1e-12
    nan6 = (np.nan,) * 6
    if coords_apical.shape[0] < 3:
        return nan6
    ow = np.asarray(outward, dtype=np.float64)
    nrm = float(np.linalg.norm(ow))
    if nrm <= eps:
        return nan6
    ow = ow / nrm
    seed = np.array([1.0, 0.0, 0.0]) if abs(ow[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = seed - ow * float(np.dot(seed, ow))
    un = float(np.linalg.norm(u))
    if un <= eps:
        return nan6
    u = u / un
    v = np.cross(ow, u)
    pts2d = np.column_stack([coords_apical @ u, coords_apical @ v])
    uniq = np.unique(np.round(pts2d, 6), axis=0)
    if uniq.shape[0] < 3:
        return nan6
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(uniq)
    except Exception:
        return nan6
    area = float(hull.volume)            # 2D hull area
    perim = float(hull.area)             # 2D hull perimeter
    if area <= eps or perim <= eps:
        return nan6
    hverts = uniq[hull.vertices]         # ordered CCW hull polygon vertices
    shape_index = perim / np.sqrt(area)
    # convexity = hull area / area of the point cloud's oriented bounding extent
    # (a planar solidity proxy: 1 for a footprint that fills its hull-aligned span,
    # smaller for a sparse / spiky cloud). Bounded to (0, 1].
    span = uniq.max(axis=0) - uniq.min(axis=0)
    bbox_area = float(span[0] * span[1])
    convexity = float(min(1.0, area / bbox_area)) if bbox_area > eps else np.nan

    # SIMPLIFY the hull to its real CORNERS: a rasterised polygon's hull has many
    # near-collinear staircase vertices along each straight edge, which inflates the
    # vertex count (a hexagon footprint -> ~11 hull points). Keep only vertices whose
    # turning angle deviates from straight by more than a tolerance (~12°), so a
    # hexagon reads ~6 corners + a clean per-corner angle distribution.
    m = hverts.shape[0]
    corner_angles = []
    for i in range(m):
        p_prev = hverts[(i - 1) % m]
        p_cur = hverts[i]
        p_next = hverts[(i + 1) % m]
        a1 = p_prev - p_cur
        a2 = p_next - p_cur
        n1 = float(np.linalg.norm(a1)); n2 = float(np.linalg.norm(a2))
        if n1 <= eps or n2 <= eps:
            continue
        cosang = float(np.clip(np.dot(a1, a2) / (n1 * n2), -1.0, 1.0))
        interior = np.arccos(cosang)               # interior angle at this vertex
        # a "corner" turns by more than ~12° away from straight (π interior).
        if (np.pi - interior) > np.deg2rad(12.0):
            corner_angles.append(interior)
    vcount = float(len(corner_angles)) if corner_angles else float(m)
    angle_cv = np.nan
    if len(corner_angles) >= 2:
        av = np.asarray(corner_angles, dtype=np.float64)
        amean = float(av.mean())
        angle_cv = float(av.std() / amean) if amean > eps else np.nan
    return vcount, area, perim, shape_index, convexity, angle_cv


def compute_ma_apical_basal(
    full_labels: np.ndarray,
    df: pd.DataFrame,
    spacing: Optional[Tuple[float, ...]] = None,
    n_slabs: int = 20,
    slices: Optional[list] = None,
) -> pd.DataFrame:
    """SIGNED apical/basal split of the major-axis cross-section profile (the feature catalogue
    remainder; the ``ma_apical_area`` / ``ma_basal_area`` candidates; C2 / P11).

    M1 (:func:`compute_ma_cross_section_features`) builds a tilt-robust cross-section
    profile ``A(t)`` perpendicular to each cell's OWN major axis (rotation-invariant),
    but its summaries (min / max / range / waist) are UNSIGNED — the major-axis
    direction is sign-arbitrary, so M1 cannot say which end is apical. This pass
    ORIENTS the major axis by the Wave C ``tissue_outward`` normal — the +end of the
    axis is the one whose projection onto the outward normal is positive (toward the
    tissue surface = APICAL), the −end basal — and reports the cross-section area in
    the APICAL third of the profile vs the BASAL third, plus each third's taper.

    A GLOBAL pass (it needs the per-cell outward normal Wave C writes) — run ONCE on
    the full label volume AFTER :func:`compute_tissue_depth` inside the
    ``tissue_depth=True`` block, like the Wave D passes. It does NOT need the
    adjacency graph (it is a per-cell profile, only the SIGN comes from the tissue
    frame). It reuses the SAME major-axis slab logic as M1.

    Features added (5):
        ``ma_apical_area`` — mean ``A(t)`` over the APICAL third of the oriented
            profile (the third nearest the apical/+outward end), a voxel-count
            cross-section area. Unbounded >= 0.
        ``ma_basal_area`` — mean ``A(t)`` over the BASAL third (the −outward end).
            Unbounded >= 0.
        ``ma_apical_basal_ratio`` — ``ma_apical_area / max(ma_basal_area, eps)``:
            > 1 = the cell is WIDER at the apical end (a wedge flaring apically),
            < 1 = wider basally. Unbounded >= 0.
        ``ma_apical_taper`` — slope of a linear fit of ``A(t)`` vs ``t`` over the
            APICAL third, SIGN-ORIENTED so +taper = the cross-section GROWS toward
            the apical surface (flare), −taper = narrows toward apical. (The image
            taper magnitude is the same as M1's; the sign is now meaningful.)
        ``ma_basal_taper`` — same over the BASAL third, +taper = grows toward the
            basal interior.

    NaN (all 5) when the cell has no major axis / centroid columns, a degenerate
    (single-slab) profile, OR its outward normal is untrusted (low
    ``tissue_outward_confidence`` / NaN — the flat tissue core, where the
    apical/basal direction is meaningless). Absent entirely (columns not created)
    when Wave C did not run (no ``tissue_outward_*`` columns) — the default-off
    contract.

    Args:
        full_labels: the FULL 3D integer label volume (Z, Y, X). Background 0.
        df: assembled per-cell table (modified in-place). Needs ``label``,
            ``major_axis_{z,y,x}`` + ``centroid-{0,1,2}`` (the M1 inputs), and the
            Wave C ``tissue_outward_*`` columns (the sign basis).
        spacing: voxel spacing ``(Z, Y, X)`` -> the projection binning is physical
            (same as M1). ``None`` = voxel units.
        n_slabs: number of slabs perpendicular to the major axis (default 20, the
            M1 default).
        slices: OPTIONAL precomputed ``find_objects`` bbox list (S1 scaling hook) —
            indexed by ``label_id - 1``, supplied by the caller to skip a redundant
            full-volume ``find_objects(full_labels.astype(int32))`` (whose ``.astype``
            on a memmap forces a FULL RESIDENT copy -> OOM at scale). None => the
            original byte-identical internal scan; the per-cell crops are lazy on a
            memmap.

    Returns:
        The same DataFrame with the 5 columns added — or unchanged when Wave C did
        not run (no outward normal to sign the frame).
    """
    from scipy.ndimage import find_objects

    n = len(df)
    if n == 0:
        for col in MA_APICAL_BASAL_COLUMNS:
            df[col] = pd.Series(dtype=np.float64)
        return df

    axis_cols = ["major_axis_z", "major_axis_y", "major_axis_x"]
    cen_cols = ["centroid-0", "centroid-1", "centroid-2"]
    ow_cols = ["tissue_outward_z", "tissue_outward_y", "tissue_outward_x"]
    if not all(c in df.columns for c in ow_cols):
        # Wave C did not run -> no sign basis. SKIP (default-off contract).
        return df
    if "label" not in df.columns or not all(c in df.columns for c in axis_cols + cen_cols):
        for col in MA_APICAL_BASAL_COLUMNS:
            df[col] = np.nan
        return df

    outward, trusted = _cell_outward_vectors(df)

    ma_ap = np.full(n, np.nan, dtype=np.float64)
    ma_ba = np.full(n, np.nan, dtype=np.float64)
    ma_ratio = np.full(n, np.nan, dtype=np.float64)
    ma_ap_taper = np.full(n, np.nan, dtype=np.float64)
    ma_ba_taper = np.full(n, np.nan, dtype=np.float64)

    # S1 scaling hook: reuse the caller's precomputed bboxes (off the memmap)
    # instead of a redundant full-volume find_objects + int32 copy (the ``.astype``
    # on a memmap forces a FULL RESIDENT copy -> OOM at scale). None => the original
    # byte-identical internal scan. The per-cell crops (``full_labels[obj_slice]``)
    # are lazy on a memmap, so with ``slices`` supplied the full volume is never
    # materialised.
    if slices is None:
        slices = find_objects(full_labels.astype(np.int32))
    label_ids = df["label"].values
    sp = np.asarray(spacing, dtype=np.float64) if spacing is not None else np.ones(3)
    n_slabs = max(3, int(n_slabs))         # need >=3 slabs to carve thirds
    axis_vals = df[axis_cols].values.astype(np.float64)
    eps = 1e-12

    for idx in range(n):
        label_id = int(label_ids[idx])
        if label_id < 1 or label_id > len(slices) or slices[label_id - 1] is None:
            continue
        if not trusted[idx]:
            continue                       # untrusted outward normal -> NaN

        obj_slice = slices[label_id - 1]
        mask_crop = full_labels[obj_slice] == label_id
        coords = np.argwhere(mask_crop)
        if coords.shape[0] < 2:
            continue

        direction = axis_vals[idx]
        norm = float(np.sqrt(np.sum(direction ** 2)))
        if norm <= eps:
            continue
        direction = direction / norm

        # ORIENT the (sign-arbitrary) major axis by the outward normal: if the axis
        # points AWAY from outward, flip it so +t is the apical (toward-surface) end.
        ow = outward[idx]                  # unit ZYX (trusted)
        if float(np.dot(direction, ow)) < 0.0:
            direction = -direction

        # Project each voxel onto the oriented major axis (physical), crop-local
        # centroid (translation cancels) — SAME as M1.
        local_centroid = coords.mean(axis=0)
        disp = (coords - local_centroid) * sp
        s = disp @ direction
        s_min, s_max = float(s.min()), float(s.max())
        s_range = s_max - s_min
        if s_range <= eps:
            continue

        t = (s - s_min) / s_range          # 0 = basal end, 1 = apical end (oriented)
        bins = np.clip((t * n_slabs).astype(np.int64), 0, n_slabs - 1)
        counts = np.bincount(bins, minlength=n_slabs).astype(np.float64)
        slab_centers = (np.arange(n_slabs, dtype=np.float64) + 0.5) / n_slabs

        third = n_slabs // 3
        basal_idx = np.arange(0, third)                       # t near 0 = basal end
        apical_idx = np.arange(n_slabs - third, n_slabs)      # t near 1 = apical end

        ba_counts = counts[basal_idx]
        ap_counts = counts[apical_idx]
        ba_occ = ba_counts[ba_counts > 0]
        ap_occ = ap_counts[ap_counts > 0]
        ma_ap[idx] = float(ap_occ.mean()) if ap_occ.size else 0.0
        ma_ba[idx] = float(ba_occ.mean()) if ba_occ.size else 0.0
        ma_ratio[idx] = ma_ap[idx] / max(ma_ba[idx], eps)

        # signed taper within each third (slope of A vs t over OCCUPIED slabs).
        for sub_idx, out_arr in ((apical_idx, ma_ap_taper), (basal_idx, ma_ba_taper)):
            sub_counts = counts[sub_idx]
            occ = sub_counts > 0
            if occ.sum() >= 2:
                cx = slab_centers[sub_idx][occ]
                if np.ptp(cx) > eps:
                    out_arr[idx] = float(np.polyfit(cx, sub_counts[occ], 1)[0])
                else:
                    out_arr[idx] = 0.0
            else:
                out_arr[idx] = 0.0

    df["ma_apical_area"] = ma_ap
    df["ma_basal_area"] = ma_ba
    df["ma_apical_basal_ratio"] = ma_ratio
    df["ma_apical_taper"] = ma_ap_taper
    df["ma_basal_taper"] = ma_ba_taper
    return df


def compute_apical_junction(
    full_labels: np.ndarray,
    df: pd.DataFrame,
    spacing: Optional[Tuple[float, ...]] = None,
    adjacency: Optional[list] = None,
    max_radius: int = 5,
    slices: Optional[list] = None,
) -> pd.DataFrame:
    """Apical junctional / epithelial-polygon topology from a cell's LATERAL contacts.

    In an ordered epithelium each cell's APICAL-JUNCTIONAL belt is the ring of
    lateral cell–cell contacts perpendicular to the apical/basal axis — the classic
    epithelial polygon (a hexagon on average). We read it off the SIGNED frame: of a
    cell's graph neighbours, we keep only the LATERAL ones (their contact direction
    is perpendicular to the Wave C ``tissue_outward`` normal — the equatorial side
    band, ``|cos| < _APICAL_BASAL_COS_THRESH``), and summarise that lateral belt.

    A GLOBAL pass (it needs the adjacency graph + the contact voxels + the per-cell
    outward normal) — run ONCE on the full label volume AFTER
    :func:`compute_tissue_depth` AND the adjacency pass, inside the
    ``tissue_depth=True`` block. Reuses the SAME graduated-dilation contact recovery
    as M4 / Wave D (:func:`_contact_voxels_per_neighbor`) so the contact set is
    consistent with the M3 adjacency graph. ``adjacency`` defaults to
    ``df.attrs["_adjacency_list"]``.

    Features added (3):
        ``apical_polygon_sides`` — the number of DISTINCT lateral neighbours = the
            epithelial polygon side count (~6 in well-ordered tissue; a count >= 0).
            A merged cell (few large lateral faces) reads LOW; an over-segmented
            fragment reads atypically. Unbounded >= 0 (treated as a count).
        ``apical_junction_length`` — the total lateral contact "perimeter" =
            ``Σ over lateral neighbours sqrt(contact_voxels)`` × the mean physical
            voxel pitch (a perimeter PROXY: a planar contact patch of area ``a`` has
            a perimeter ``~ sqrt(a)``; summing over the lateral neighbours gives the
            length of the junctional belt). Physical (µm) when ``spacing`` given.
            Unbounded >= 0.
        ``apical_polygon_irregularity`` — the coefficient of variation (std/mean) of
            the lateral contact PATCH SIZES (voxel counts). 0 = every junctional face
            is the same size (a regular polygon); high = a few dominant faces (an
            irregular / merged junction). Unbounded >= 0; NaN for < 2 lateral
            neighbours (no variation to measure).

    Wave J REMAINDER (7 more):
        ``apical_junction_vertex_count`` — the number of distinct lateral-belt
            junction vertices (== the lateral-neighbour count, the polygon vertex
            count; kept as a distinct catalogue name from ``apical_polygon_sides``).
        ``apical_junction_edge_length_cv`` — the cv of consecutive edge lengths of
            the junction polygon (the belt vertices ordered by footprint azimuth);
            0 = a regular polygon, high = uneven junction spacing. NaN for < 3 belt
            vertices (no polygon).
        ``apical_vertex_angle_cv`` — the cv of the junction polygon's interior
            vertex angles (0 = regular). Falls back to the apical FOOTPRINT polygon's
            angle cv when the belt has < 3 vertices but a cap polygon exists.
        ``apical_polygon_area`` / ``apical_polygon_perimeter`` — the area + perimeter
            of the explicit apical FOOTPRINT polygon (the cell's apical-third voxels
            collapsed onto the plane ⊥ outward, convex-hulled). Physical (µm² / µm)
            when ``spacing`` given. Unbounded >= 0.
        ``apical_shape_index`` — ``apical_polygon_perimeter / sqrt(apical_polygon_
            area)`` — the vertex-model order parameter (~3.72 for a regular hexagon,
            ~3.54 for a circle; the classic epithelial fluid/solid index). Unbounded
            >= ~3.54.
        ``apical_polygon_convexity`` — the apical footprint's planar solidity proxy
            (hull area / footprint bounding-extent area) ∈ (0, 1]; 1 = a compact
            convex cap. NaN for a degenerate footprint.

    NaN (the 3 belt columns) when the cell has NO lateral contacts OR its outward
    normal is
    untrusted (low ``tissue_outward_confidence`` / NaN). ``apical_polygon_sides`` /
    ``apical_junction_length`` are 0 for a cell WITH a trusted normal but no lateral
    neighbours (a genuine apical/basal-only contact set); ``apical_polygon_irregularity``
    needs >= 2 lateral faces. Absent entirely when Wave C did not run.

    Args:
        full_labels: the FULL 3D integer label volume (Z, Y, X). Background 0.
        df: assembled per-cell table (modified in-place). Needs ``label``,
            ``centroid-{0,1,2}`` (the ray origin for the contact direction), and the
            Wave C ``tissue_outward_*`` columns (the lateral test).
        spacing: voxel spacing ``(Z, Y, X)`` -> the contact directions + the junction
            length are physical. ``None`` = voxel units.
        adjacency: the M3 edge list; ``None`` -> ``df.attrs["_adjacency_list"]``.
        max_radius: graduated-dilation radius (voxels) — MUST match the M3 /
            M4 / Wave D ``max_radius`` so the contact set is consistent. Default 5.
        slices: OPTIONAL precomputed ``find_objects`` bbox list (S1 scaling hook) —
            indexed by ``label_id - 1``, supplied by the caller to skip a redundant
            full-volume ``find_objects(full_labels.astype(int32))`` (whose ``.astype``
            on a memmap forces a FULL RESIDENT copy -> OOM at scale). None => the
            original byte-identical internal scan; the per-cell contact crops are
            lazy on a memmap.

    Returns:
        The same DataFrame with the 3 columns added — or unchanged when Wave C did
        not run.
    """
    from scipy.ndimage import find_objects

    n = len(df)
    if n == 0:
        for c in APICAL_JUNCTION_COLUMNS:
            df[c] = pd.Series(dtype=np.float64)
        return df

    ow_cols = ["tissue_outward_z", "tissue_outward_y", "tissue_outward_x"]
    if not all(c in df.columns for c in ow_cols):
        return df                          # Wave C did not run -> SKIP (default-off)
    cen_cols = ["centroid-0", "centroid-1", "centroid-2"]
    if "label" not in df.columns or not all(c in df.columns for c in cen_cols):
        for c in APICAL_JUNCTION_COLUMNS:
            df[c] = np.nan
        return df

    outward, trusted = _cell_outward_vectors(df)
    centroids = df[cen_cols].values.astype(np.float64)   # PHYSICAL when spacing set

    if adjacency is None:
        adjacency = df.attrs.get("_adjacency_list")
    nb_sets: dict = {}
    if adjacency:
        for e in adjacency:
            a = int(e["label_a"]); b = int(e["label_b"])
            nb_sets.setdefault(a, set()).add(b)
            nb_sets.setdefault(b, set()).add(a)

    sp = tuple(float(s) for s in spacing) if spacing is not None else (1.0, 1.0, 1.0)
    sp_vec = np.asarray(sp, dtype=np.float64)
    mean_pitch = float(np.mean(sp_vec))                  # physical voxel pitch proxy

    sides = np.full(n, np.nan, dtype=np.float64)
    length = np.full(n, np.nan, dtype=np.float64)
    irreg = np.full(n, np.nan, dtype=np.float64)
    # Wave J REMAINDER accumulators.
    vcount = np.full(n, np.nan, dtype=np.float64)
    edge_cv = np.full(n, np.nan, dtype=np.float64)
    angle_cv = np.full(n, np.nan, dtype=np.float64)
    poly_area = np.full(n, np.nan, dtype=np.float64)
    poly_perim = np.full(n, np.nan, dtype=np.float64)
    shape_index = np.full(n, np.nan, dtype=np.float64)
    convexity = np.full(n, np.nan, dtype=np.float64)

    # S1 scaling hook: reuse the caller's precomputed bboxes (off the memmap)
    # instead of a redundant full-volume find_objects + int32 copy (the ``.astype``
    # on a memmap forces a FULL RESIDENT copy -> OOM at scale). None => the original
    # byte-identical internal scan. The per-cell contact recovery
    # (``_contact_voxels_per_neighbor``) crops ``full_labels`` lazily, so with
    # ``slices`` supplied the full volume is never materialised.
    if slices is None:
        slices = find_objects(full_labels.astype(np.int32))
    label_ids = df["label"].values
    cos_t = _APICAL_BASAL_COS_THRESH
    eps = 1e-12
    pixel_area = float(sp_vec[1] * sp_vec[2])      # nominal projected pixel area

    for idx in range(n):
        lid = int(label_ids[idx])
        if lid < 1 or lid > len(slices) or slices[lid - 1] is None:
            continue
        if not trusted[idx]:
            continue                       # untrusted normal -> NaN
        neighbor_ids = nb_sets.get(lid, set())
        if not neighbor_ids:
            continue                       # no contacts -> NaN

        contacts = _contact_voxels_per_neighbor(
            full_labels, lid, slices[lid - 1], neighbor_ids, max_radius=max_radius,
        )
        if not contacts:
            continue

        ow = outward[idx]                  # unit (trusted)
        lateral_sizes = []                 # voxel count of each LATERAL contact patch
        lateral_vert2d = []                # 2D footprint-plane centroid of each belt vertex
        # orthonormal plane basis (u, v) ⊥ outward, for ordering the junction belt.
        seed = np.array([1.0, 0.0, 0.0]) if abs(ow[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = seed - ow * float(np.dot(seed, ow))
        un = float(np.linalg.norm(u))
        u = u / un if un > eps else seed
        v = np.cross(ow, u)
        cell_cen = centroids[idx]
        for nb, coords in contacts.items():
            if coords.shape[0] == 0:
                continue
            # mean direction from the cell centroid to this neighbour's contact patch
            # (physical), classed lateral when |cos with outward| < threshold.
            coords_phys = coords * sp_vec
            patch_cen = coords_phys.mean(axis=0)
            mean_vec = patch_cen - cell_cen
            mv_norm = float(np.linalg.norm(mean_vec))
            if mv_norm <= 1e-9:
                continue
            cos = float(np.dot(mean_vec / mv_norm, ow))
            if abs(cos) < cos_t:           # LATERAL belt
                lateral_sizes.append(int(coords.shape[0]))
                rel = patch_cen - cell_cen
                lateral_vert2d.append((float(rel @ u), float(rel @ v)))

        # a cell with a trusted normal but no lateral neighbours: a real 0-side belt.
        k = len(lateral_sizes)
        sides[idx] = float(k)
        vcount[idx] = float(k)             # Wave J — junction vertex count (= side count)
        if k == 0:
            length[idx] = 0.0
            continue
        sizes = np.asarray(lateral_sizes, dtype=np.float64)
        # perimeter proxy: a planar patch of area `a` voxels has perimeter ~ sqrt(a);
        # sum over the lateral faces, scaled to physical length by the voxel pitch.
        length[idx] = float(np.sqrt(sizes).sum() * mean_pitch)
        if k >= 2:
            mean_sz = float(sizes.mean())
            irreg[idx] = float(sizes.std() / mean_sz) if mean_sz > 0 else np.nan

        # --- Wave J — junction belt edge-length + vertex-angle CV --------
        # Order the lateral-neighbour "vertices" by their azimuth in the footprint
        # plane (the apical-junctional polygon's natural cyclic order), then measure
        # the consecutive edge lengths + interior angles of that polygon.
        if k >= 3:
            vp = np.asarray(lateral_vert2d, dtype=np.float64)
            ang = np.arctan2(vp[:, 1], vp[:, 0])
            order = np.argsort(ang)
            vp = vp[order]
            m = vp.shape[0]
            edges = np.linalg.norm(vp - np.roll(vp, -1, axis=0), axis=1)
            emean = float(edges.mean())
            if emean > eps:
                edge_cv[idx] = float(edges.std() / emean)
            iangles = []
            for i in range(m):
                a1 = vp[(i - 1) % m] - vp[i]
                a2 = vp[(i + 1) % m] - vp[i]
                n1 = float(np.linalg.norm(a1)); n2 = float(np.linalg.norm(a2))
                if n1 <= eps or n2 <= eps:
                    continue
                ca = float(np.clip(np.dot(a1, a2) / (n1 * n2), -1.0, 1.0))
                iangles.append(np.arccos(ca))
            if len(iangles) >= 2:
                ia = np.asarray(iangles)
                iam = float(ia.mean())
                angle_cv[idx] = float(ia.std() / iam) if iam > eps else np.nan

        # --- Wave J — explicit apical FOOTPRINT polygon -----------------
        # The cell's APICAL-third voxels (toward the outward normal) collapsed onto
        # the footprint plane -> convex-hull outline -> area / perimeter / vertex /
        # shape-index / convexity. Uses the same signed apical frame as Wave D.
        crop = full_labels[slices[lid - 1]] == lid
        if crop.any():
            off = np.array([s.start for s in slices[lid - 1]], dtype=np.float64)
            vox = np.argwhere(crop).astype(np.float64) + off          # global voxel ZYX
            phys = vox * sp_vec                                       # physical ZYX
            rel = phys - cell_cen
            t = rel @ ow                                              # signed axial coord
            tmax = float(t.max()); tmin = float(t.min())
            tspan = tmax - tmin
            if tspan > eps:
                apical_sel = t >= (tmin + (2.0 / 3.0) * tspan)        # apical third
                cap = phys[apical_sel]
                if cap.shape[0] >= 3:
                    vc, ar, pe, si, cx, acv = _apical_footprint_polygon(
                        cap, ow, pixel_area)
                    poly_area[idx] = ar
                    poly_perim[idx] = pe
                    shape_index[idx] = si
                    convexity[idx] = cx
                    # prefer the FOOTPRINT vertex-angle cv when the belt was too
                    # small to define one (k < 3) but the cap polygon exists.
                    if not np.isfinite(angle_cv[idx]) and np.isfinite(acv):
                        angle_cv[idx] = acv

    df["apical_polygon_sides"] = sides
    df["apical_junction_length"] = length
    df["apical_polygon_irregularity"] = irreg
    # Wave J REMAINDER.
    df["apical_junction_vertex_count"] = vcount
    df["apical_junction_edge_length_cv"] = edge_cv
    df["apical_vertex_angle_cv"] = angle_cv
    df["apical_polygon_area"] = poly_area
    df["apical_polygon_perimeter"] = poly_perim
    df["apical_shape_index"] = shape_index
    df["apical_polygon_convexity"] = convexity
    return df


# --------------------------------------------------------------------------- #
# Wave G — watershed territory / crowding (the C7 family; primitive P8).
# A GLOBAL pass that expands every cell into the surrounding BACKGROUND up to the
# midline with its neighbours (a Voronoi-ish "territory" capped by a max distance
# from the cell). The territory volume relative to the cell volume is the local
# breathing-room / crowding signal; the territory's own geometry (anisotropy /
# solidity) describes the shape of the space the cell occupies.
# --------------------------------------------------------------------------- #

# Wave G built the first 4 territory columns; Mop-up M (the feature catalogue
# remainder) adds the 4 watershed-adjacency + nearest-gap + territory-z-bias +
# territory-surface-area columns, completing the family. Module constant so
# the exact-set family test + feature_stage registry stay in lockstep.
WATERSHED_TERRITORY_COLUMNS = (
    # Wave G (original 4).
    "ws_territory_volume", "territory_to_cell_ratio",
    "territory_anisotropy", "territory_solidity",
    # Mop-up M — watershed adjacency + gap + territory z-bias + surface area.
    "ws_neighbor_count", "ws_nearest_gap",
    "territory_z_bias", "territory_surface_area",
)

# Default expansion cap for the watershed territory, in VOXELS along each axis
# from a cell's surface (the EDT-of-background flood is masked to this distance so
# a cell at a tissue edge does NOT grow its territory into infinite open
# background). Read it as "how far a cell is allowed to claim space" — a few cell
# radii is the sensible default; data-dependent, so it is exposed as a kwarg.
_WS_TERRITORY_MAX_DIST_VOXELS = 15.0


def compute_watershed_territory(
    full_labels: np.ndarray,
    df: pd.DataFrame,
    spacing: Optional[Tuple[float, ...]] = None,
    max_distance_voxels: float = _WS_TERRITORY_MAX_DIST_VOXELS,
    downsample: int = 1,
) -> pd.DataFrame:
    """Per-cell VORONOI-ish TERRITORY volume + crowding + territory shape
    (the C7 watershed-territory family; primitive P8).

    A GLOBAL pass — it needs the WHOLE label volume at once (a per-cell bbox crop
    cannot see where its neighbours' territories meet), so it runs ONCE on the full
    label volume inside the ``territory=True`` block of
    :func:`epicycle.reconcile.features.extract_label_features` (reusing the already-
    loaded ``full_labels`` — the volume is NOT read twice), exactly like
    :func:`compute_tissue_depth` / :func:`compute_neighbor_features`.

    The primitive is a single seeded watershed that floods the BACKGROUND outward
    from every cell: ``markers = full_labels`` and the flood surface is the EDT of
    the background (distance from each background voxel to the nearest cell), so the
    watershed assigns each background voxel to its nearest cell and the basins meet
    at the MIDLINE between neighbouring cells — a discrete Voronoi tessellation of
    the inter-cell space. The flood is CAPPED at ``max_distance_voxels`` from any
    cell (the watershed mask = ``bg_edt <= cap``) so a cell at a tissue edge does
    NOT claim infinite open background; its territory is bounded by the cap there.
    Each cell's TERRITORY is its own voxels PLUS the background voxels the watershed
    assigned to it.

    FRAME NOTE: the background EDT that defines the watershed surface (and hence the
    midline placement + the territory geometry) is computed in the VOXEL FRAME
    (index distance, no ``sampling=spacing``); the territory is therefore a
    VOXEL-FRAME tessellation. Physical units are applied AFTER the fact only to the
    REPORTED measures (``ws_territory_volume`` / ``territory_surface_area`` /
    ``ws_nearest_gap``), and the dimensionless RATIOS (``territory_to_cell_ratio`` /
    ``territory_anisotropy`` / ``territory_solidity``) are robust to spacing. On
    strongly anisotropic voxels the midline sits at the voxel-frame (not physical)
    equidistant surface — interpret the absolute territory SHAPE accordingly.

    EXPANSION CAP (read before interpreting): the cap is a HARD ceiling on how far a
    cell may grow into open background. Between two crowded neighbours the basins
    meet at the midline WELL INSIDE the cap, so the cap is irrelevant (the cells
    box each other in) and ``territory_to_cell_ratio`` ~ 1. In open space a lone
    cell's territory grows out to the cap sphere and the ratio is large. So the cap
    sets the SATURATION point of the crowding signal: an isolated cell reads
    "uncrowded" up to the cap, not "infinitely uncrowded". The default
    (``_WS_TERRITORY_MAX_DIST_VOXELS`` = 15 voxels) is a few cell radii; tune it to
    your cell scale (it is in VOXELS, NOT physical units, so it does not depend on
    ``spacing``).

    Features added (8 — Wave G built the first 4; Mop-up M added the 4
    watershed-adjacency + gap + territory-z-bias + surface-area columns,
    completing the family):
        ``ws_territory_volume`` — the PHYSICAL volume (µm³ when ``spacing`` given,
            voxels otherwise) of the cell's territory = cell voxels + the background
            voxels the watershed gave it. Unbounded >= the cell volume.
        ``territory_to_cell_ratio`` — ``ws_territory_volume / cell_volume`` >= 1:
            the LOCAL BREATHING ROOM / inverse crowding. ~1 = tightly packed (no
            room to grow, neighbours on every side); large = isolated / sparse (room
            out to the cap). Unbounded >= 1.
        ``territory_anisotropy`` — the elongation of the territory from its voxel
            inertia tensor (``sqrt(largest / smallest eigenvalue)``, the same
            ``inertia_elongation`` measure :func:`extract_features` uses). 1 =
            isotropic blob; > 1 = the cell's free space is stretched along one
            direction (a cell wedged in a row has an elongated territory). NaN for a
            degenerate (1-voxel-line / point) territory whose smallest inertia
            eigenvalue is ~0 (the elongation is undefined, not infinite).
        ``territory_solidity`` — the territory's ``volume / convex_hull_volume`` ∈
            (0, 1]: how convex the claimed space is. < 1 = a concave / fingered
            territory (the cell's free space wraps around neighbours).
        ``ws_neighbor_count`` (Mop-up M) — the number of DISTINCT OTHER territories
            that share a face with this cell's territory: the watershed-tessellation
            neighbour count (an adjacency count derived from the SPACE tessellation,
            not the graduated-dilation contact graph — complementary to
            ``neighbor_count``; in open / sparse regions a cell can have a small
            ws-neighbour count even when its dilated contact count is 0). 0 (not NaN)
            for a fully isolated territory. Unbounded count.
        ``ws_nearest_gap`` (Mop-up M) — the PHYSICAL distance to the nearest cell
            across the territory boundary = ``2 × min background-EDT over the
            territory's boundary with another territory`` × the mean voxel pitch
            (the two basins meet at the midline, so the half-gap is the EDT value
            there). Small = a tightly-abutting neighbour; large = a wide gap to the
            nearest cell. NaN for an isolated territory (no shared boundary).
            Unbounded >= 0.
        ``territory_z_bias`` (Mop-up M) — the SIGNED offset of the territory
            centroid from the CELL centroid along Z, normalised by the territory's
            Z half-extent ∈ [-1, 1]: 0 = the free space is symmetric in Z about the
            cell, + = the territory extends preferentially to high Z (the cell sits
            at the low-Z side of its space), − = the reverse. NaN without a cell
            centroid or a degenerate Z extent. Bounded (signed).
        ``territory_surface_area`` (Mop-up M) — the PHYSICAL surface area (µm² when
            ``spacing`` given) of the territory boundary, from an exposed-face count
            (each of the 6 axis-faces of a territory voxel not shared with another
            territory voxel contributes its physical face area). Unbounded >= 0.

    All NaN-guarded: an empty table, a volume with no cells, a degenerate
    single-voxel territory, or a failed hull/inertia decomposition degrade to NaN
    for the affected columns rather than raising.

    Args:
        full_labels: the FULL 3D integer label volume (Z, Y, X). Background 0.
            Reused from the global block (loaded once); NOT re-read.
        df: assembled per-cell table (modified in-place). Needs ``label`` to index
            cells; the per-cell territory is keyed by that label id.
            ``centroid-0/1/2`` enable ``territory_z_bias`` (else NaN).
        spacing: voxel spacing ``(Z, Y, X)`` in physical units -> the territory +
            cell VOLUMES (and the ratio's numerator/denominator) are physical;
            anisotropy + solidity are spacing-aware via ``regionprops(spacing=...)``;
            ``ws_nearest_gap`` + ``territory_surface_area`` are scaled to physical
            units. ``None`` = voxel units. NOTE: ``max_distance_voxels`` is always in
            VOXELS (the background EDT cap is index-based), independent of spacing.
        max_distance_voxels: the territory expansion cap in VOXELS (see EXPANSION
            CAP above). Default :data:`_WS_TERRITORY_MAX_DIST_VOXELS`.
        downsample: integer field-downsample factor ``N`` (S2/S3 scaling). ``N == 1``
            (default) runs the EXACT full-resolution code below (byte-identical). For
            ``N > 1`` the seeds + the background-EDT watershed run on a COARSE label
            volume (downsampled by ``N``: per-block CENTRE voxel's label — built
            streaming off ``full_labels`` if it is a memmap) with the cap scaled to
            ``max_distance_voxels / N`` coarse voxels. The territory VOLUME is scaled
            back: a coarse voxel = ``N³`` full-res voxels, so ``ws_territory_volume``
            and ``territory_surface_area`` are multiplied by ``N³`` / ``N²`` (or the
            coarse physical voxel volume / face area when ``spacing`` is given);
            ``ws_nearest_gap`` is scaled by ``N``. The dimensionless ratios
            (``territory_to_cell_ratio`` etc.) use the cell volume in the SAME coarse-
            voxel-equivalent frame so they stay consistent. An APPROXIMATION for
            ``N > 1``.

    Returns:
        The same DataFrame with the 8 new columns added.
    """
    downsample = int(downsample)
    if downsample > 1:
        return _compute_watershed_territory_downsampled(
            full_labels, df, spacing=spacing,
            max_distance_voxels=max_distance_voxels, downsample=downsample)

    from scipy.ndimage import distance_transform_edt, find_objects
    from skimage.measure import regionprops
    from skimage.segmentation import watershed

    n = len(df)
    if n == 0:
        for c in WATERSHED_TERRITORY_COLUMNS:
            df[c] = pd.Series(dtype=np.float64)
        return df
    if "label" not in df.columns:
        for c in WATERSHED_TERRITORY_COLUMNS:
            df[c] = np.nan
        return df

    lab = np.asarray(full_labels)
    sp = tuple(float(s) for s in spacing) if spacing is not None else None
    sp_vec = np.asarray(sp, dtype=np.float64) if sp is not None else np.ones(3)
    vox_vol = float(np.prod(sp)) if sp is not None else 1.0
    mean_pitch = float(np.mean(sp_vec))
    # physical area of the face perpendicular to each axis (Z, Y, X): for axis a,
    # the face area is the product of the OTHER two spacings.
    face_area = np.array([sp_vec[1] * sp_vec[2],
                          sp_vec[0] * sp_vec[2],
                          sp_vec[0] * sp_vec[1]], dtype=np.float64)
    eps = 1e-12

    ws_vol = np.full(n, np.nan, dtype=np.float64)
    to_cell = np.full(n, np.nan, dtype=np.float64)
    aniso = np.full(n, np.nan, dtype=np.float64)
    solidity = np.full(n, np.nan, dtype=np.float64)
    # Mop-up M accumulators.
    ws_nb_count = np.zeros(n, dtype=np.float64)
    ws_gap = np.full(n, np.nan, dtype=np.float64)
    terr_z_bias = np.full(n, np.nan, dtype=np.float64)
    terr_surf = np.full(n, np.nan, dtype=np.float64)

    cen_cols = ["centroid-0", "centroid-1", "centroid-2"]
    have_cen = all(c in df.columns for c in cen_cols)
    centroids = df[cen_cols].values.astype(np.float64) if have_cen else None

    mask_fg = lab > 0
    if not mask_fg.any():
        for c in WATERSHED_TERRITORY_COLUMNS:
            df[c] = np.nan
        return df

    # --- the territory tessellation -----------------------------------------
    # bg_edt = distance from each VOXEL to the nearest cell (0 inside cells). The
    # flood surface for the watershed is bg_edt itself (low near cells, high in open
    # background): markers=cells, mask=(bg_edt<=cap), so each background voxel within
    # the cap is assigned to its nearest cell and basins meet at the inter-cell
    # midline. Index-based EDT (sampling=None) so the VOXEL cap is exact regardless
    # of physical spacing.
    bg_edt = distance_transform_edt(~mask_fg)
    flood_mask = bg_edt <= float(max_distance_voxels)
    territory = watershed(bg_edt, markers=lab.astype(np.int32), mask=flood_mask)

    # per-territory bbox (1-indexed by label) so we crop tight, like the other
    # global passes.
    objs = find_objects(territory.astype(np.int32))
    n_slots = len(objs)
    label_ids = df["label"].values

    for idx in range(n):
        lid = int(label_ids[idx])
        if not (1 <= lid <= n_slots) or objs[lid - 1] is None:
            continue
        sl = objs[lid - 1]
        # pad the crop by 1 voxel (clamped) so the territory's outer faces + the
        # neighbouring-territory ids are visible for the surface / adjacency / gap.
        pad_sl = tuple(
            slice(max(s.start - 1, 0), min(s.stop + 1, dim))
            for s, dim in zip(sl, territory.shape))
        terr_crop = territory[pad_sl]
        bg_crop = bg_edt[pad_sl]
        terr_mask = terr_crop == lid
        terr_vox = int(terr_mask.sum())
        if terr_vox <= 0:
            continue
        cell_vox = int((lab[pad_sl] == lid).sum())   # the cell (subset of terr)
        ws_vol[idx] = terr_vox * vox_vol
        if cell_vox > 0:
            to_cell[idx] = terr_vox / cell_vox

        # territory shape: inertia elongation (anisotropy) + convex-hull solidity,
        # on the single-object territory crop (spacing-aware). regionprops needs >= a
        # few voxels for the hull / inertia; guard each independently.
        single = terr_mask.astype(np.uint8)
        kw = {"spacing": sp} if sp is not None else {}
        try:
            props = regionprops(single, **kw)
        except Exception:
            props = []
        if props:
            p = props[0]
            try:
                ev = np.linalg.eigvalsh(np.asarray(p.inertia_tensor, dtype=np.float64))
                ev = np.clip(ev, 0.0, None)
                if ev[-1] > eps and ev[0] > eps:
                    aniso[idx] = float(np.sqrt(ev[-1] / ev[0]))
                # else: the smallest inertia eigenvalue is ~0 (a degenerate 1-voxel
                # line / point territory) -> the anisotropy is undefined; leave NaN.
                # (Dividing by ``eps`` here produced a ~1e6 sentinel SPIKE — the only
                # /eps divide-by-tiny in the module; every sibling floors the
                # denominator or NaNs the degenerate case.)
            except (np.linalg.LinAlgError, ValueError):
                pass
            try:
                hull_vol = float(p.area_convex)        # physical when spacing set
                terr_phys = terr_vox * vox_vol
                if hull_vol > eps:
                    solidity[idx] = float(min(terr_phys / hull_vol, 1.0))
            except (ValueError, AttributeError):
                pass

        # --- Mop-up M — exposed-face surface + ws-adjacency + nearest gap ----
        # For each of the 3 axes, a territory voxel face is EXPOSED when the
        # neighbour along that axis is NOT this territory. Count exposed faces
        # (weighted by the physical face area) for the surface; record which OTHER
        # territory id sits across each exposed face (for adjacency) + the bg_edt at
        # the shared-boundary voxel (for the nearest gap). 0-padding (constant) at
        # the crop edge makes the outer territory faces exposed against id 0.
        nb_ids: set = set()
        surf = 0.0
        min_gap_edt = np.inf
        for axis in range(3):
            # shift the territory crop by +1 and -1 along this axis; an exposed face
            # is where the shifted id differs from lid.
            for shift in (+1, -1):
                shifted = np.roll(terr_crop, shift, axis=axis)
                # zero out the wrapped plane so the roll does not see the far edge.
                sl_idx = [slice(None)] * 3
                sl_idx[axis] = (0 if shift == +1 else -1)
                shifted = shifted.copy()
                shifted[tuple(sl_idx)] = 0
                exposed = terr_mask & (shifted != lid)
                surf += float(exposed.sum()) * face_area[axis]
                if exposed.any():
                    other = shifted[exposed]
                    other_nonzero = other[other > 0]
                    nb_ids.update(int(o) for o in np.unique(other_nonzero))
                    # nearest gap: bg_edt of the OTHER-territory voxel at the
                    # boundary (the midline EDT value -> half the inter-cell gap).
                    shifted_edt = np.roll(bg_crop, shift, axis=axis)
                    shifted_edt = shifted_edt.copy()
                    shifted_edt[tuple(sl_idx)] = np.inf
                    boundary_other = exposed & (shifted > 0)
                    if boundary_other.any():
                        g = float(shifted_edt[boundary_other].min())
                        if g < min_gap_edt:
                            min_gap_edt = g
        nb_ids.discard(lid)
        ws_nb_count[idx] = float(len(nb_ids))
        terr_surf[idx] = surf
        if np.isfinite(min_gap_edt):
            # 2 × the midline half-gap, scaled to physical units (the EDT is in
            # voxels; the gap is an isotropic-distance proxy × the mean pitch).
            ws_gap[idx] = 2.0 * min_gap_edt * mean_pitch

        # territory z-bias: signed Z offset of the territory centroid from the cell
        # centroid, normalised by the territory's Z half-extent. Work in the CROP
        # frame; add the crop origin so the territory centroid is in global coords.
        if have_cen:
            tcoords = np.argwhere(terr_mask).astype(np.float64)
            if tcoords.shape[0] > 0:
                origin = np.array([s.start for s in pad_sl], dtype=np.float64)
                tz = (tcoords[:, 0] + origin[0])          # global Z (voxel)
                terr_cz = float(tz.mean())
                cell_cz_vox = centroids[idx][0] / sp_vec[0]   # back to voxel Z
                z_half = (float(tz.max()) - float(tz.min())) / 2.0
                if z_half > eps and np.isfinite(cell_cz_vox):
                    terr_z_bias[idx] = float(
                        np.clip((terr_cz - cell_cz_vox) / z_half, -1.0, 1.0))

    df["ws_territory_volume"] = ws_vol
    df["territory_to_cell_ratio"] = to_cell
    df["territory_anisotropy"] = aniso
    df["territory_solidity"] = solidity
    df["ws_neighbor_count"] = ws_nb_count
    df["ws_nearest_gap"] = ws_gap
    df["territory_z_bias"] = terr_z_bias
    df["territory_surface_area"] = terr_surf
    return df


def _compute_watershed_territory_downsampled(
    full_labels,
    df: pd.DataFrame,
    spacing: Optional[Tuple[float, ...]] = None,
    max_distance_voxels: float = _WS_TERRITORY_MAX_DIST_VOXELS,
    downsample: int = 2,
) -> pd.DataFrame:
    """S2/S3 DOWNSAMPLED watershed territory — the ``downsample > 1``
    branch of :func:`compute_watershed_territory`. NOT byte-identical (an
    APPROXIMATION).

    The seeds + the background-EDT watershed run on a COARSE label volume (the
    per-block CENTRE voxel's label — built streaming off ``full_labels`` if it is a
    memmap; see :func:`epicycle.core.field_downsample.coarse_labels`). The expansion
    cap is scaled to ``max_distance_voxels / N`` COARSE voxels (a coarse voxel spans
    ``N`` full-res voxels), so the territory grows the same PHYSICAL distance as the
    ``N == 1`` cap.

    UNIT SCALING — the key correctness step. Run the SAME per-cell measurement code as
    ``N == 1`` but on the coarse grid with ``coarse_sampling = spacing * N``:
    * ``ws_territory_volume`` = coarse-territory-voxels × ``prod(coarse_sampling)``.
      With ``spacing`` this is the PHYSICAL territory volume directly (a coarse voxel
      = ``prod(spacing)·N³``); WITHOUT spacing it is ``coarse_vox × N³`` = the full-res-
      VOXEL-equivalent volume. Same frame as the ``N == 1`` output.
    * ``territory_surface_area`` uses the coarse face area (= ``N²`` without spacing,
      physical with) -> the same physical / full-res-voxel-equivalent frame.
    * ``ws_nearest_gap`` uses the coarse mean pitch (= ``N`` without spacing, physical
      with) -> same frame. SUB-N LIMITATION (L1): two cells OR-pooled into adjacency
      CLOSER than ``N`` voxels have a coarse midline half-gap of EXACTLY 0 — the gap
      is UNRESOLVED at this resolution, not physically touching — so ``ws_nearest_gap``
      is emitted as NaN ("unknown at this downsample") rather than 0 (which would read
      as "touching", a false strong signal). The smallest gap this path can RESOLVE is
      ~``N`` voxels (one coarse pitch).
    * ``territory_to_cell_ratio`` divides coarse territory voxels by coarse CELL voxels
      (both from the coarse labels) -> dimensionless + consistent. NOTE the cell volume
      is the COARSE cell volume here (not the full-res ``area``), so the two sides of
      the ratio are in the same coarse-voxel frame.
    * ``territory_anisotropy`` / ``territory_solidity`` / ``territory_z_bias`` are
      dimensionless / self-normalised -> spacing-robust either way.

    APPROXIMATION: a cell smaller than ~``N`` voxels can vanish from the coarse seed
    volume (its centre-voxel block was background) -> its territory columns are NaN
    (no coarse seed). The midline placement is resolved only to ``N``-voxel blocks.
    """
    from scipy.ndimage import distance_transform_edt, find_objects
    from skimage.measure import regionprops
    from skimage.segmentation import watershed
    from packages.epicycle_scaledseg.core.field_downsample import coarse_labels

    N = int(downsample)
    n = len(df)
    if n == 0:
        for c in WATERSHED_TERRITORY_COLUMNS:
            df[c] = pd.Series(dtype=np.float64)
        return df
    if "label" not in df.columns:
        for c in WATERSHED_TERRITORY_COLUMNS:
            df[c] = np.nan
        return df

    sp = tuple(float(s) for s in spacing) if spacing is not None else None
    # coarse_sampling is ALWAYS defined (spacing*N, or (N,N,N) when None) so all the
    # physical-unit scaling below comes out in the right frame (physical with spacing;
    # full-res-voxel-equivalent without).
    coarse_sampling = tuple(float(s) * N for s in (sp if sp is not None else (1.0, 1.0, 1.0)))
    cs = coarse_sampling
    cs_vec = np.asarray(cs, dtype=np.float64)
    vox_vol = float(np.prod(cs))                      # physical / N³ coarse-voxel volume
    mean_pitch = float(np.mean(cs_vec))               # physical / N coarse pitch
    face_area = np.array([cs_vec[1] * cs_vec[2],
                          cs_vec[0] * cs_vec[2],
                          cs_vec[0] * cs_vec[1]], dtype=np.float64)
    eps = 1e-12

    ws_vol = np.full(n, np.nan, dtype=np.float64)
    to_cell = np.full(n, np.nan, dtype=np.float64)
    aniso = np.full(n, np.nan, dtype=np.float64)
    solidity = np.full(n, np.nan, dtype=np.float64)
    ws_nb_count = np.zeros(n, dtype=np.float64)
    ws_gap = np.full(n, np.nan, dtype=np.float64)
    terr_z_bias = np.full(n, np.nan, dtype=np.float64)
    terr_surf = np.full(n, np.nan, dtype=np.float64)

    cen_cols = ["centroid-0", "centroid-1", "centroid-2"]
    have_cen = all(c in df.columns for c in cen_cols)
    centroids = df[cen_cols].values.astype(np.float64) if have_cen else None

    # COARSE label volume (seeds) — streamed off the memmap, full volume never resident.
    lab = coarse_labels(full_labels, N)
    mask_fg = lab > 0
    if not mask_fg.any():
        for c in WATERSHED_TERRITORY_COLUMNS:
            df[c] = np.nan
        return df

    bg_edt = distance_transform_edt(~mask_fg)         # COARSE-voxel index distance
    flood_mask = bg_edt <= float(max_distance_voxels) / float(N)   # cap -> coarse voxels
    territory = watershed(bg_edt, markers=lab.astype(np.int32), mask=flood_mask)

    objs = find_objects(territory.astype(np.int32))
    n_slots = len(objs)
    label_ids = df["label"].values

    for idx in range(n):
        lid = int(label_ids[idx])
        if not (1 <= lid <= n_slots) or objs[lid - 1] is None:
            continue
        sl = objs[lid - 1]
        pad_sl = tuple(
            slice(max(s.start - 1, 0), min(s.stop + 1, dim))
            for s, dim in zip(sl, territory.shape))
        terr_crop = territory[pad_sl]
        bg_crop = bg_edt[pad_sl]
        terr_mask = terr_crop == lid
        terr_vox = int(terr_mask.sum())
        if terr_vox <= 0:
            continue
        cell_vox = int((lab[pad_sl] == lid).sum())    # COARSE cell voxel count
        ws_vol[idx] = terr_vox * vox_vol
        if cell_vox > 0:
            to_cell[idx] = terr_vox / cell_vox

        single = terr_mask.astype(np.uint8)
        kw = {"spacing": cs}
        try:
            props = regionprops(single, **kw)
        except Exception:
            props = []
        if props:
            p = props[0]
            try:
                ev = np.linalg.eigvalsh(np.asarray(p.inertia_tensor, dtype=np.float64))
                ev = np.clip(ev, 0.0, None)
                if ev[-1] > eps and ev[0] > eps:
                    aniso[idx] = float(np.sqrt(ev[-1] / ev[0]))
            except (np.linalg.LinAlgError, ValueError):
                pass
            try:
                hull_vol = float(p.area_convex)
                terr_phys = terr_vox * vox_vol
                if hull_vol > eps:
                    solidity[idx] = float(min(terr_phys / hull_vol, 1.0))
            except (ValueError, AttributeError):
                pass

        nb_ids: set = set()
        surf = 0.0
        min_gap_edt = np.inf
        for axis in range(3):
            for shift in (+1, -1):
                shifted = np.roll(terr_crop, shift, axis=axis)
                sl_idx = [slice(None)] * 3
                sl_idx[axis] = (0 if shift == +1 else -1)
                shifted = shifted.copy()
                shifted[tuple(sl_idx)] = 0
                exposed = terr_mask & (shifted != lid)
                surf += float(exposed.sum()) * face_area[axis]
                if exposed.any():
                    other = shifted[exposed]
                    other_nonzero = other[other > 0]
                    nb_ids.update(int(o) for o in np.unique(other_nonzero))
                    shifted_edt = np.roll(bg_crop, shift, axis=axis)
                    shifted_edt = shifted_edt.copy()
                    shifted_edt[tuple(sl_idx)] = np.inf
                    boundary_other = exposed & (shifted > 0)
                    if boundary_other.any():
                        g = float(shifted_edt[boundary_other].min())
                        if g < min_gap_edt:
                            min_gap_edt = g
        nb_ids.discard(lid)
        ws_nb_count[idx] = float(len(nb_ids))
        terr_surf[idx] = surf
        # L1 — a coarse midline half-gap of EXACTLY 0 means the two cells were
        # OR-pooled into adjacency CLOSER than N voxels: the gap is UNRESOLVED at
        # this resolution, NOT physically touching. Emitting 0 would read as
        # "touching" (a false strong signal); emit NaN so a sub-N gap is correctly
        # "unknown at this downsample" instead. (ws_gap stays NaN — its init value.)
        if np.isfinite(min_gap_edt) and min_gap_edt > 0.0:
            # 2 × the coarse midline half-gap × the coarse mean pitch -> physical /
            # full-res-voxel-equivalent (the EDT is in COARSE voxels; mean_pitch
            # already carries the ×N coarse scale, so no extra ×N here).
            ws_gap[idx] = 2.0 * min_gap_edt * mean_pitch

        if have_cen:
            tcoords = np.argwhere(terr_mask).astype(np.float64)
            if tcoords.shape[0] > 0:
                origin = np.array([s.start for s in pad_sl], dtype=np.float64)
                tz = (tcoords[:, 0] + origin[0])       # global COARSE Z (voxel)
                terr_cz = float(tz.mean())
                # cell centroid physical Z -> full-res voxel Z -> coarse Z (÷N).
                cell_cz_full_vox = centroids[idx][0] / (sp[0] if sp is not None else 1.0)
                cell_cz_vox = cell_cz_full_vox / N
                z_half = (float(tz.max()) - float(tz.min())) / 2.0
                if z_half > eps and np.isfinite(cell_cz_vox):
                    terr_z_bias[idx] = float(
                        np.clip((terr_cz - cell_cz_vox) / z_half, -1.0, 1.0))

    df["ws_territory_volume"] = ws_vol
    df["territory_to_cell_ratio"] = to_cell
    df["territory_anisotropy"] = aniso
    df["territory_solidity"] = solidity
    df["ws_neighbor_count"] = ws_nb_count
    df["ws_nearest_gap"] = ws_gap
    df["territory_z_bias"] = terr_z_bias
    df["territory_surface_area"] = terr_surf
    return df


# --------------------------------------------------------------------------- #
# Wave G — HCR puncta / transcript-spatial (the C24 family; primitive P10).
# Detects DISCRETE SPOTS (puncta) in a masked channel crop with blob_log — the
# spot-COUNTING (transcript) complement to the Wave-E continuous TEXTURE signal.
# A per-channel dict-return helper (like compute_channel_texture /
# compute_channel_localization), wired into the per-cell M2 sampler.
# --------------------------------------------------------------------------- #

_CHANNEL_PUNCTA_SUFFIXES = (
    "_puncta_count", "_puncta_density", "_puncta_intensity_mean",
    "_puncta_mean_volume", "_puncta_nn_distance", "_punctate_fraction",
    # Wave K — puncta SPATIAL / distribution remainder. Core-vs-rind
    # localization, clustered-vs-dispersed structure, A/B polarity, brightness
    # spread, and absolute spread. All NaN when < the stated spot count.
    "_puncta_radial_position_mean", "_puncta_clustering_index",
    "_puncta_polarity", "_puncta_intensity_cv", "_puncta_dispersion",
)

# blob_log scale range (the sigma of the LoG kernel, in voxels) the spot detector
# searches. DATA-DEPENDENT: it must bracket the radius of a real punctum for THIS
# channel + resolution. The defaults bracket ~1.5–6 voxel-wide spots (the typical
# HCR dot at confocal resolution); tune ``min_sigma`` / ``max_sigma`` to your spot
# size. blob_log reports each blob's radius as ``sigma * sqrt(ndim)``.
_PUNCTA_MIN_SIGMA = 1.0
_PUNCTA_MAX_SIGMA = 4.0
_PUNCTA_NUM_SIGMA = 5
# Detection threshold as a FRACTION of the background-subtracted dynamic range
# (blob_log ``threshold_rel``-style, applied to the normalised image). Higher =
# fewer, brighter spots. Data-dependent; the default is a moderate ~10% cut.
_PUNCTA_THRESHOLD_REL = 0.1
# Min in-mask voxels for puncta detection to even be attempted (a tiny crop has no
# room for a resolvable spot) -> count 0 / NaN spot stats below this.
_PUNCTA_MIN_MASK_VOXELS = 27


def compute_channel_puncta(
    channel_image: np.ndarray,
    single_mask: np.ndarray,
    prefix: str,
    spacing: Optional[Tuple[float, ...]] = None,
    outward: Optional[Tuple[float, float, float]] = None,
    centroid: Optional[Tuple[float, float, float]] = None,
) -> dict:
    """Per-channel HCR PUNCTA / transcript-spatial features for ONE cell (Wave G +
    Wave K; primitive P10).

    A SIBLING of :func:`compute_channel_texture` / :func:`compute_channel_intensity`,
    called on the SAME raw-channel crop + single-label mask the M2 intensity / texture
    features read. Where the texture path measures the CONTINUOUS spatial organisation
    of the signal (punctate vs diffuse vs hazy), this path COUNTS DISCRETE SPOTS — the
    transcript-counting (HCR dot) signal. For HCR data the spot COUNT + spatial
    arrangement are often more biologically meaningful than the mean intensity.

    Detection: the in-mask intensities are background-subtracted (the in-mask
    minimum) and normalised to [0, 1] over their robust range, then
    :func:`skimage.feature.blob_log` (Laplacian-of-Gaussian) finds blobs over the
    scale range ``[_PUNCTA_MIN_SIGMA, _PUNCTA_MAX_SIGMA]`` (in voxels). Only blobs
    whose CENTRE lands inside ``single_mask`` are kept (a spot detected in the
    bbox but in a neighbouring cell's territory is discarded). blob_log reports each
    blob's radius as ``sigma * sqrt(3)``; the spot volume is the sphere of that radius
    (physical when ``spacing`` given).

    blob_log is the spot-COUNTING (transcript) signal — complementary to the Wave-E
    ``<prefix>_texture_granularity`` (a continuous punctate-vs-diffuse score). Use
    the texture granularity when you want a smooth scalar; use the puncta count when
    you want an actual transcript NUMBER.

    ISOTROPY CAVEAT: the LoG SCALE search is in VOXELS with an ISOTROPIC kernel
    (``min_sigma``/``max_sigma`` are voxel scalars, not per-axis), so the detector's
    notion of a "spot of radius σ" is only directly COMPARABLE across cells / datasets
    on NEAR-ISOTROPIC data (sp_z ≈ sp_y ≈ sp_x). On strongly anisotropic voxels a
    physical spot subtends fewer voxels along Z, so the same spot may fall outside the
    voxel-frame scale window — re-tune ``min_sigma``/``max_sigma`` (or resample to
    isotropic) for anisotropic stacks. (Same caveat as the Wave-E granularity
    structuring element.)

    Columns emitted (6, all ``<prefix>``-prefixed):
        ``<prefix>_puncta_count`` — number of detected spots inside the mask = the
            transcript count. 0 for a smooth / diffuse cell (no resolvable spots).
        ``<prefix>_puncta_density`` — ``puncta_count / cell_volume`` (spots per µm³
            when ``spacing`` given, per voxel otherwise) — the spot CONCENTRATION,
            size-normalised. 0 when no spots.
        ``<prefix>_puncta_intensity_mean`` — the mean RAW peak intensity at the
            detected spot centres (how bright the dots are). NaN when no spots.
        ``<prefix>_puncta_mean_volume`` — the mean physical volume of a detected
            spot (from its blob_log radius). NaN when no spots.
        ``<prefix>_puncta_nn_distance`` — the mean nearest-neighbour distance
            between spot centres (physical when ``spacing`` given) = the spot
            CLUSTERING signal (small = clustered puncta, large = dispersed). NaN
            when < 2 spots (no neighbour pair).
        ``<prefix>_punctate_fraction`` — the fraction of the cell's TOTAL
            (background-subtracted) channel signal that falls inside the detected
            spots ∈ [0, 1]: HIGH = the signal IS the puncta (punctate expression),
            LOW = the spots carry little of the total signal (diffuse / hazy
            expression). 0 when no spots.

    Wave K — SPATIAL / distribution remainder (5 more ``<prefix>``-prefixed):
        ``<prefix>_puncta_radial_position_mean`` — mean normalized interior-EDT
            radius (``r = d/max(d) ∈ [0,1]``, 1 = deep core, 0 = surface rind) of the
            spot centres. LOW = spots hug the membrane / surface (cortical), HIGH =
            spots sit in the core. NaN when < 1 spot or the mask collapses to a single
            radial level.
        ``<prefix>_puncta_clustering_index`` — the Clark–Evans-style ratio of the
            OBSERVED mean nearest-neighbour distance to the EXPECTED mean-NN under a
            uniform (CSR) point process at the same density in the cell volume:
            ``R = nn_observed / nn_expected``, ``nn_expected = 0.554 · (V/n)^(1/3)``
            (the 3D CSR constant). ``R < 1`` = clustered, ``R ≈ 1`` = random, ``R > 1``
            = dispersed / regular. NaN when < 2 spots.
        ``<prefix>_puncta_polarity`` — the signed apical−basal spot-count difference
            as a fraction of the total, split by the plane through the cell centroid
            perpendicular to the Wave C ``outward`` normal (a spot is apical when
            ``((centre − centroid) · spacing) · outward > 0`` — the voxel-frame
            displacement is scaled to PHYSICAL units before the dot with the physical
            outward normal): ``(n_apical − n_basal) / n`` ∈
            [-1, 1]. > 0 = apically biased, < 0 = basally biased. NaN when no trusted
            ``outward`` normal is supplied (the FeatureStage per-cell pool runs BEFORE
            the global outward-normal pass) or < 1 spot.
        ``<prefix>_puncta_intensity_cv`` — the coefficient of variation
            (``std/mean``) of the RAW peak intensities at the spot centres: how
            uneven the spot brightnesses are. NaN when < 2 spots or the mean peak is
            ~0.
        ``<prefix>_puncta_dispersion`` — the RMS spread of the spot centres about
            their own mean (the root-mean-square physical distance to the spot
            centroid) divided by the cell's equivalent radius
            ``(3V/4π)^(1/3)`` — how broadly the spots fill the cell, normalised for
            cell size. ~0 = all spots in one place, large = spread across the whole
            cell. NaN when < 2 spots.

    NaN / 0 guard: a cell with no detectable spots gets ``puncta_count = 0``,
    ``puncta_density = 0``, ``punctate_fraction = 0`` and NaN for the spot-only
    stats (intensity / volume / nn_distance — undefined with zero spots). A tiny
    mask (< ``_PUNCTA_MIN_MASK_VOXELS``), a shape mismatch, or an all-NaN crop
    (failed tile read) yields ``count/density/fraction = 0`` + NaN spot stats —
    schema-stable (every cell carries all 6 columns).

    Args:
        channel_image: 3D intensity crop (Z, Y, X) for this channel, aligned
            voxel-for-voxel with ``single_mask``.
        single_mask: 3D boolean (or 0/1) crop — the isolated cell.
        prefix: column-name prefix (e.g. ``"hcr"``, ``"gene1"``), matching the
            intensity prefix for the same channel.
        spacing: voxel spacing ``(Z, Y, X)`` -> the density (per physical volume),
            spot volumes, nearest-neighbour distances, clustering index, and
            dispersion are in physical units. The blob_log SCALE search is in VOXELS
            (index-based). ``None`` = voxel units.
        outward: OPTIONAL per-cell OUTWARD unit normal ``(z, y, x)`` (the Wave C
            ``tissue_outward_*`` columns) for the Wave-K ``_puncta_polarity`` split.
            When None the polarity column stays NaN (the FeatureStage per-cell pool
            runs BEFORE the global outward-normal pass).
        centroid: OPTIONAL cell centroid ``(z, y, x)`` in the CROP's own VOXEL frame
            for the polarity split. When None it is computed from the mask. Only used
            with ``outward``.

    Returns:
        ``{f"{prefix}{suffix}": float}`` for the 11 puncta columns (6 Wave G + 5
        Wave K spatial).
    """
    from skimage.feature import blob_log

    out = {f"{prefix}{s}": np.nan for s in _CHANNEL_PUNCTA_SUFFIXES}

    mask = np.asarray(single_mask).astype(bool)
    ch = np.asarray(channel_image, dtype=np.float64)
    if mask.shape != ch.shape or not mask.any():
        return out

    in_vals = ch[mask]
    finite = in_vals[np.isfinite(in_vals)]
    if finite.size == 0:
        # all-NaN crop (failed read) -> schema-stable: 0 count, NaN spot stats.
        out[f"{prefix}_puncta_count"] = 0.0
        out[f"{prefix}_puncta_density"] = 0.0
        out[f"{prefix}_punctate_fraction"] = 0.0
        return out

    sp = tuple(float(s) for s in spacing) if spacing is not None else None
    vox_vol = float(np.prod(sp)) if sp is not None else 1.0
    cell_vox = int(mask.sum())
    cell_phys_vol = cell_vox * vox_vol

    # background-subtracted signal over the WHOLE crop, zeroed outside the mask, so
    # blob_log only ever sees in-mask structure (a bright neighbour in the bbox can't
    # seed a spurious blob). The background is the in-mask MINIMUM (a robust floor).
    bg = float(finite.min())
    sig = np.zeros_like(ch, dtype=np.float64)
    sig_in = np.clip(ch[mask] - bg, 0.0, None)
    sig_in = np.where(np.isfinite(sig_in), sig_in, 0.0)
    sig[mask] = sig_in
    total_signal = float(sig[mask].sum())

    # too small / no dynamic range -> no resolvable spot. 0 count, NaN spot stats.
    if cell_vox < _PUNCTA_MIN_MASK_VOXELS or total_signal <= 0.0:
        out[f"{prefix}_puncta_count"] = 0.0
        out[f"{prefix}_puncta_density"] = 0.0
        out[f"{prefix}_punctate_fraction"] = 0.0
        return out

    # normalise to [0,1] over the signal's own max so threshold_rel is comparable
    # across cells (blob_log's absolute `threshold` then reads as a fraction of the
    # cell's own dynamic range).
    smax = float(sig.max())
    norm = sig / smax if smax > 0 else sig

    try:
        blobs = blob_log(
            norm,
            min_sigma=_PUNCTA_MIN_SIGMA,
            max_sigma=_PUNCTA_MAX_SIGMA,
            num_sigma=_PUNCTA_NUM_SIGMA,
            threshold=_PUNCTA_THRESHOLD_REL,
        )
    except Exception:
        blobs = np.empty((0, 4), dtype=np.float64)

    # keep only blobs whose CENTRE voxel is inside the mask.
    kept = []
    for b in blobs:
        zc, yc, xc = int(round(b[0])), int(round(b[1])), int(round(b[2]))
        if (0 <= zc < mask.shape[0] and 0 <= yc < mask.shape[1]
                and 0 <= xc < mask.shape[2] and mask[zc, yc, xc]):
            kept.append(b)

    n_spots = len(kept)
    out[f"{prefix}_puncta_count"] = float(n_spots)
    out[f"{prefix}_puncta_density"] = (
        n_spots / cell_phys_vol if cell_phys_vol > 0 else 0.0)

    if n_spots == 0:
        out[f"{prefix}_punctate_fraction"] = 0.0
        return out

    kept = np.asarray(kept, dtype=np.float64)
    centres_vox = kept[:, :3]                       # (k, 3) ZYX voxel centres
    sigmas = kept[:, 3]                             # per-blob sigma (voxels)

    # peak intensity at each spot centre (RAW channel, not normalised/bg-subtracted).
    peak_vals = []
    spot_signal = 0.0
    for c, sigma in zip(centres_vox, sigmas):
        zc, yc, xc = (int(round(c[0])), int(round(c[1])), int(round(c[2])))
        pv = ch[zc, yc, xc]
        if np.isfinite(pv):
            peak_vals.append(float(pv))
        # the spot's signal contribution = the background-subtracted signal summed
        # over the spot's radius ball (intersected with the mask) -> punctate fraction.
        radius_vox = float(sigma) * np.sqrt(3.0)    # blob_log radius convention
        rv = max(int(np.ceil(radius_vox)), 1)
        z0 = max(zc - rv, 0); z1 = min(zc + rv + 1, mask.shape[0])
        y0 = max(yc - rv, 0); y1 = min(yc + rv + 1, mask.shape[1])
        x0 = max(xc - rv, 0); x1 = min(xc + rv + 1, mask.shape[2])
        sub = sig[z0:z1, y0:y1, x0:x1]
        sub_mask = mask[z0:z1, y0:y1, x0:x1]
        zz, yy, xx = np.indices(sub.shape)
        d2 = ((zz + z0 - zc) ** 2 + (yy + y0 - yc) ** 2 + (xx + x0 - xc) ** 2)
        ball = (d2 <= radius_vox ** 2) & sub_mask
        spot_signal += float(sub[ball].sum())
    if peak_vals:
        out[f"{prefix}_puncta_intensity_mean"] = float(np.mean(peak_vals))

    # mean spot volume from the blob_log radius (physical when spacing given). Use a
    # GEOMETRIC-mean voxel size as the isotropic scale so an anisotropic spacing
    # still maps the (isotropic, voxel-defined) blob radius to a physical volume.
    iso_scale = (vox_vol ** (1.0 / 3.0)) if vox_vol > 0 else 1.0
    radii_phys = sigmas * np.sqrt(3.0) * iso_scale
    spot_vols = (4.0 / 3.0) * np.pi * np.power(radii_phys, 3.0)
    out[f"{prefix}_puncta_mean_volume"] = float(np.mean(spot_vols))

    # physical spot centres (ZYX × spacing) — shared by the NN, clustering-index,
    # and dispersion computations below.
    centres_phys = centres_vox * (np.asarray(sp, dtype=np.float64)
                                  if sp is not None else 1.0)

    # mean nearest-neighbour distance between spot centres (physical). Needs >= 2.
    nn_observed = np.nan
    if n_spots >= 2:
        # pairwise distances; NN = the min off-diagonal per row.
        diff = centres_phys[:, None, :] - centres_phys[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))
        np.fill_diagonal(dist, np.inf)
        nn = dist.min(axis=1)
        nn_observed = float(np.mean(nn))
        out[f"{prefix}_puncta_nn_distance"] = nn_observed

    # punctate fraction: spot signal / total in-mask signal, clamped to [0,1] (a
    # spot ball can clip a neighbour's signal, so cap at the total). HIGH = the
    # signal IS the puncta; LOW = diffuse.
    if total_signal > 0:
        out[f"{prefix}_punctate_fraction"] = float(
            min(spot_signal / total_signal, 1.0))
    else:
        out[f"{prefix}_punctate_fraction"] = 0.0

    # ----------------------------------------------------------------------- #
    # Wave K — puncta SPATIAL / distribution remainder.
    # ----------------------------------------------------------------------- #
    # radial position: mean normalized interior-EDT radius of the spot CENTRES
    # (1 = core, 0 = surface). Reuses the M5 / Wave-F radial field; NaN if the
    # mask collapses to a single radial level (a 1-voxel sheet).
    r_field, _ = _edt_radial_position(mask, spacing=sp)
    if r_field is not None and n_spots >= 1:
        rvals = []
        for c in centres_vox:
            zc, yc, xc = (int(round(c[0])), int(round(c[1])), int(round(c[2])))
            if (0 <= zc < mask.shape[0] and 0 <= yc < mask.shape[1]
                    and 0 <= xc < mask.shape[2] and mask[zc, yc, xc]):
                rvals.append(float(r_field[zc, yc, xc]))
        if rvals:
            out[f"{prefix}_puncta_radial_position_mean"] = float(np.mean(rvals))

    # clustering index (Clark–Evans, 3D): observed mean-NN / expected-under-CSR.
    # expected NN for n points uniform in volume V is 0.554·(V/n)^(1/3); R<1 =
    # clustered, ~1 = random, >1 = dispersed. Needs >= 2 spots + a positive density.
    if n_spots >= 2 and np.isfinite(nn_observed) and cell_phys_vol > 0:
        nn_expected = 0.554 * (cell_phys_vol / n_spots) ** (1.0 / 3.0)
        if nn_expected > 1e-12:
            out[f"{prefix}_puncta_clustering_index"] = nn_observed / nn_expected

    # polarity: signed apical−basal spot-count balance along the OUTWARD normal,
    # split by the plane through the cell centroid. Needs a trusted outward normal
    # (NaN in the per-cell pool, filled when the global outward normal is available).
    if outward is not None and n_spots >= 1:
        ow = np.asarray(outward, dtype=np.float64)
        if ow.shape == (3,) and np.all(np.isfinite(ow)):
            nrm = float(np.linalg.norm(ow))
            if nrm > 1e-9:
                ow = ow / nrm
                if centroid is not None:
                    cen = np.asarray(centroid, dtype=np.float64)
                else:
                    cen = centres_vox.mean(axis=0)  # fallback: spot centroid
                    idx = np.argwhere(mask)
                    if idx.size:
                        cen = idx.mean(axis=0)       # prefer the mask centroid
                # ``centres_vox`` + ``cen`` are in the crop's VOXEL frame, but ``ow``
                # is a PHYSICAL-frame outward normal (Wave C). Scale the displacement
                # by the per-axis spacing FIRST so the dot is geometrically correct
                # under anisotropic spacing (mirrors compute_channel_localization /
                # compute_prob_geometry / compute_axis_neighbor_persistence).
                sp_vec = (np.asarray(sp, dtype=np.float64)
                          if sp is not None else np.ones(3, dtype=np.float64))
                proj = ((centres_vox - cen[None, :]) * sp_vec[None, :]) @ ow
                n_ap = int(np.sum(proj > 0))
                n_ba = int(np.sum(proj < 0))
                out[f"{prefix}_puncta_polarity"] = (n_ap - n_ba) / float(n_spots)

    # intensity cv: spread of the RAW peak intensities at the spot centres.
    if len(peak_vals) >= 2:
        pv = np.asarray(peak_vals, dtype=np.float64)
        mu = float(pv.mean())
        if abs(mu) > 1e-12:
            out[f"{prefix}_puncta_intensity_cv"] = float(pv.std()) / mu

    # dispersion: RMS spread of spot centres about their own mean (physical),
    # normalised by the cell's equivalent radius (3V/4π)^(1/3). Needs >= 2 spots.
    if n_spots >= 2:
        spot_centroid = centres_phys.mean(axis=0)
        rms = float(np.sqrt(
            np.mean(((centres_phys - spot_centroid[None, :]) ** 2).sum(axis=1))))
        equiv_radius = (3.0 * cell_phys_vol / (4.0 * np.pi)) ** (1.0 / 3.0)
        if equiv_radius > 1e-12:
            out[f"{prefix}_puncta_dispersion"] = rms / equiv_radius

    return out


# Wave K — channel-PAIR CO-LOCALIZATION suffixes. Templated per
# unordered channel pair as ``<a>_<b>{suffix}``. The classic microscopy
# co-localization battery (Pearson + the two Manders overlap coefficients + the
# Manders overlap coefficient + Li's ICQ + the co-positive voxel fraction) measured
# over the in-mask voxels common to both channels.
_CHANNEL_COLOC_SUFFIXES = (
    "_pearson", "_manders_m1", "_manders_m2", "_overlap_coefficient",
    "_icq", "_coloc_fraction",
    # Mop-up O co-localization remainder:
    #   the A:B mean ratio (`<chA>_to_<chB>_mean_ratio`) + the region-
    #   RESTRICTED co-localization: boundary-shell Pearson (`_boundary_corr`)
    #   + the interior-vs-boundary A:B ratio DELTA (`_interior_ratio_delta`).
    "_mean_ratio", "_boundary_corr", "_interior_ratio_delta",
)

# Min in-mask voxels for a co-localization estimate to be attempted (a tiny crop is
# a degenerate correlation) -> all-NaN below this.
_COLOC_MIN_MASK_VOXELS = 8


def compute_channel_colocalization(
    crop_a: np.ndarray,
    crop_b: np.ndarray,
    single_mask: np.ndarray,
    prefix_a: str,
    prefix_b: str,
) -> dict:
    """Channel-PAIR co-localization features for ONE cell (Wave K;
    primitive P5×2ch).

    The two-channel companion to the single-channel intensity / texture / puncta
    families: given TWO raw-channel crops for the SAME cell, it measures how the two
    markers co-vary and overlap inside the cell. Called on the SAME single-label mask
    the M2 intensity features read; RETURNS a flat ``{column: value}`` dict (one cell).

    Columns emitted (6, templated ``<a>_<b>``-prefixed, ``a`` / ``b`` = the channel
    prefixes):

        ``<a>_<b>_pearson`` — Pearson correlation of the two channels' intensities
            over the in-mask voxels: +1 = the markers rise and fall together, 0 =
            uncorrelated, −1 = anti-correlated. NaN when either channel is flat
            (zero variance) in the mask.
        ``<a>_<b>_manders_m1`` — Manders M1: the fraction of channel-A signal that
            sits in voxels where channel B is ABOVE its threshold
            (``Σ A·[B>tB] / Σ A``) ∈ [0, 1]. "How much of A overlaps B."
        ``<a>_<b>_manders_m2`` — Manders M2: the symmetric fraction of channel-B
            signal in voxels where channel A is above its threshold
            (``Σ B·[A>tA] / Σ B``) ∈ [0, 1]. "How much of B overlaps A."
        ``<a>_<b>_overlap_coefficient`` — Manders OVERLAP coefficient
            ``Σ A·B / sqrt(Σ A² · Σ B²)`` ∈ [0, 1] — co-localization insensitive to
            the absolute intensity scale of either channel (unlike Pearson it does NOT
            subtract the means).
        ``<a>_<b>_icq`` — Li's Intensity Correlation Quotient: the fraction of in-mask
            voxels where ``(A − mean A)·(B − mean B) > 0`` minus 0.5, rescaled to
            [-0.5, 0.5]. > 0 = dependent staining (co-localized), ~0 = random, < 0 =
            segregated / exclusion.
        ``<a>_<b>_coloc_fraction`` — the fraction of in-mask voxels co-positive in
            BOTH channels (each above its own threshold) ∈ [0, 1]: a simple
            object-overlap measure independent of intensity weighting.

    Thresholds: each channel's threshold is its in-mask MEAN over the finite voxels
    (a robust, parameter-free per-cell cut — Costes-style auto thresholds would need a
    regression per cell, overkill for a per-cell crop). The Manders / overlap /
    coloc-fraction use only the NON-NEGATIVE (background-subtracted by the in-mask
    minimum) signal so a dim baseline can't dominate the overlap sums.

    NaN guard: an empty / mismatched mask, a tiny mask
    (``< _COLOC_MIN_MASK_VOXELS``), or an all-NaN crop on EITHER channel (a failed
    tile read) -> every column NaN (schema-stable). A flat channel NaNs the Pearson +
    ICQ (their variance / sign structure is undefined) but the Manders / overlap /
    coloc-fraction are still defined on the signal magnitudes.

    Args:
        crop_a / crop_b: 3D intensity crops (Z, Y, X) for the two channels, both
            aligned voxel-for-voxel with ``single_mask``.
        single_mask: 3D boolean (or 0/1) crop — the isolated cell.
        prefix_a / prefix_b: the two channels' column prefixes (e.g. ``"nuc"`` /
            ``"mem"``). The output columns are ``<prefix_a>_<prefix_b>{suffix}``.

    Mop-up O remainder (3 more): ``<a>_<b>_mean_ratio`` (whole-cell A:B
    mean ratio), ``<a>_<b>_boundary_corr`` (Pearson restricted to the outer-rind /
    membrane shell), and ``<a>_<b>_interior_ratio_delta`` (the interior-minus-
    boundary shift in the A:B mean ratio) — the region-RESTRICTED co-localization.

    Returns:
        ``{f"{prefix_a}_{prefix_b}{suffix}": float}`` for the 9 co-localization
        columns (every key present, NaN where undefined).
    """
    pair = f"{prefix_a}_{prefix_b}"
    out = {f"{pair}{s}": np.nan for s in _CHANNEL_COLOC_SUFFIXES}

    mask = np.asarray(single_mask).astype(bool)
    a = np.asarray(crop_a, dtype=np.float64)
    b = np.asarray(crop_b, dtype=np.float64)
    if mask.shape != a.shape or mask.shape != b.shape or not mask.any():
        return out

    av = a[mask]
    bv = b[mask]
    ok = np.isfinite(av) & np.isfinite(bv)
    av, bv = av[ok], bv[ok]
    if av.size < _COLOC_MIN_MASK_VOXELS:
        return out  # too few common finite voxels / failed read -> schema-stable NaN

    a_std = float(av.std())
    b_std = float(bv.std())
    a_mean = float(av.mean())
    b_mean = float(bv.mean())

    # --- Pearson (mean-subtracted correlation) -----------------------------
    if a_std > 1e-12 and b_std > 1e-12:
        cov = float(np.mean((av - a_mean) * (bv - b_mean)))
        out[f"{pair}_pearson"] = cov / (a_std * b_std)

    # --- Li's ICQ (sign of the joint deviation) ----------------------------
    if a_std > 1e-12 and b_std > 1e-12:
        prod = (av - a_mean) * (bv - b_mean)
        out[f"{pair}_icq"] = float(np.mean(prod > 0.0)) - 0.5

    # --- Manders / overlap / coloc fraction on the NON-NEGATIVE signal ------
    a_pos = np.clip(av - float(av.min()), 0.0, None)
    b_pos = np.clip(bv - float(bv.min()), 0.0, None)
    # per-channel thresholds = the (positive) in-mask MEAN.
    ta = float(a_pos.mean())
    tb = float(b_pos.mean())
    a_above = a_pos > ta
    b_above = b_pos > tb

    sum_a = float(a_pos.sum())
    sum_b = float(b_pos.sum())
    if sum_a > 1e-12:
        out[f"{pair}_manders_m1"] = float(a_pos[b_above].sum()) / sum_a
    if sum_b > 1e-12:
        out[f"{pair}_manders_m2"] = float(b_pos[a_above].sum()) / sum_b

    denom = float(np.sqrt((a_pos ** 2).sum() * (b_pos ** 2).sum()))
    if denom > 1e-12:
        out[f"{pair}_overlap_coefficient"] = float((a_pos * b_pos).sum()) / denom

    out[f"{pair}_coloc_fraction"] = float(np.mean(a_above & b_above))

    # --- Mop-up O co-localization remainder ---------------------
    # A:B mean ratio (`<chA>_to_<chB>_mean_ratio`): the whole-cell mean of A
    # over the whole-cell mean of B (> 1 = A brighter than B in this cell). Uses
    # the raw in-mask means (not the non-negative shift) so it is a true intensity
    # ratio. NaN when B's mean is ~0.
    if abs(b_mean) > 1e-9:
        out[f"{pair}_mean_ratio"] = a_mean / b_mean

    # region-RESTRICTED co-localization (boundary_corr / interior_ratio_delta).
    # Split the in-mask voxels into a BOUNDARY (outer-rind) shell and an INTERIOR
    # (core) shell via the SAME M5 interior-EDT radial partition the localization
    # family uses, then:
    #   `_boundary_corr` = Pearson of A vs B restricted to the boundary shell — do
    #       the two markers co-vary specifically at the cell membrane?
    #   `_interior_ratio_delta` = (interior A:B mean ratio) − (boundary A:B mean
    #       ratio) — does the A-vs-B balance SHIFT from the membrane to the core?
    # Both NaN-guard a missing shell / a flat channel / a ~0 denominator.
    r, _ = _edt_radial_position(mask, spacing=None)
    if r is not None:
        rmask = r[mask]                              # aligned with av/bv pre-`ok`
        # rebuild the in-mask finite selection in the SAME order as av/bv.
        full_a = a[mask]
        full_b = b[mask]
        full_ok = np.isfinite(full_a) & np.isfinite(full_b)
        ra = rmask[full_ok]
        aa = full_a[full_ok]
        bb = full_b[full_ok]
        boundary = ra <= _SHELL_R1                   # outer (membrane) rind
        interior = ra > _SHELL_R2                    # core
        # boundary Pearson.
        if int(boundary.sum()) >= _COLOC_MIN_MASK_VOXELS:
            ab, bb_ = aa[boundary], bb[boundary]
            sa, sb = float(ab.std()), float(bb_.std())
            if sa > 1e-12 and sb > 1e-12:
                cov_b = float(np.mean((ab - ab.mean()) * (bb_ - bb_.mean())))
                out[f"{pair}_boundary_corr"] = cov_b / (sa * sb)
        # interior-vs-boundary A:B mean-ratio delta.
        if boundary.any() and interior.any():
            def _ratio(am, bm):
                bmn = float(bm.mean())
                return (float(am.mean()) / bmn) if abs(bmn) > 1e-9 else np.nan
            r_int = _ratio(aa[interior], bb[interior])
            r_bnd = _ratio(aa[boundary], bb[boundary])
            if np.isfinite(r_int) and np.isfinite(r_bnd):
                out[f"{pair}_interior_ratio_delta"] = r_int - r_bnd

    return out


# Wave K — nuclear / cytoplasm PUNCTA FRACTION suffixes (PAIRED path; needs
# the nucleus mask). Templated per channel as ``<prefix>{suffix}``.
_CHANNEL_PUNCTA_COMPARTMENT_SUFFIXES = (
    "_nuclear_puncta_fraction", "_cytoplasm_puncta_fraction",
)


def _detect_in_mask_puncta(channel_image: np.ndarray, mask: np.ndarray):
    """Detect HCR spots inside ``mask`` — the SAME blob_log pipeline as
    :func:`compute_channel_puncta` (background-subtract to the in-mask minimum, zero
    outside the mask, normalise to the in-mask max, ``blob_log`` over the module spot
    scale range, keep only blobs whose CENTRE voxel is in the mask).

    Returns the ``(k, 4)`` ZYX-σ kept-blob array (empty ``(0, 4)`` when no spot is
    resolvable / the crop is too small / all-NaN). Shared by the compartment puncta
    fraction so the spot set matches the per-channel ``compute_channel_puncta`` count.
    """
    from skimage.feature import blob_log

    m = np.asarray(mask).astype(bool)
    ch = np.asarray(channel_image, dtype=np.float64)
    if m.shape != ch.shape or not m.any():
        return np.empty((0, 4), dtype=np.float64)
    in_vals = ch[m]
    finite = in_vals[np.isfinite(in_vals)]
    if finite.size == 0 or int(m.sum()) < _PUNCTA_MIN_MASK_VOXELS:
        return np.empty((0, 4), dtype=np.float64)
    bg = float(finite.min())
    sig = np.zeros_like(ch, dtype=np.float64)
    sig_in = np.clip(ch[m] - bg, 0.0, None)
    sig[m] = np.where(np.isfinite(sig_in), sig_in, 0.0)
    if float(sig[m].sum()) <= 0.0:
        return np.empty((0, 4), dtype=np.float64)
    smax = float(sig.max())
    norm = sig / smax if smax > 0 else sig
    try:
        blobs = blob_log(norm, min_sigma=_PUNCTA_MIN_SIGMA, max_sigma=_PUNCTA_MAX_SIGMA,
                         num_sigma=_PUNCTA_NUM_SIGMA, threshold=_PUNCTA_THRESHOLD_REL)
    except Exception:
        return np.empty((0, 4), dtype=np.float64)
    kept = []
    for b in blobs:
        zc, yc, xc = int(round(b[0])), int(round(b[1])), int(round(b[2]))
        if (0 <= zc < m.shape[0] and 0 <= yc < m.shape[1]
                and 0 <= xc < m.shape[2] and m[zc, yc, xc]):
            kept.append(b)
    return np.asarray(kept, dtype=np.float64) if kept else np.empty((0, 4), dtype=np.float64)


def compute_channel_puncta_compartments(
    channel_crop: np.ndarray,
    cell_mask: np.ndarray,
    nucleus_mask: np.ndarray,
    prefix: str,
    spacing: Optional[Tuple[float, ...]] = None,
) -> dict:
    """Nuclear / cytoplasm PUNCTA-FRACTION features for ONE cell (Wave K).

    The compartment split of the Wave-G per-channel puncta: of the HCR spots detected
    inside the cell, what fraction sit in the NUCLEUS vs the CYTOPLASM. Needs BOTH the
    cell mask AND its (in-cell) nucleus mask, so it runs in the reconcile PAIRED path
    (the paired cell-nucleus measurements), NOT the
    single-volume per-cell pool. RETURNS a flat ``{column: value}`` dict (one cell).

    The spots are detected with the SAME :func:`compute_channel_puncta` pipeline
    (:func:`_detect_in_mask_puncta`, ``blob_log`` over the whole cell mask) so the two
    families agree on the spot set; each spot's CENTRE voxel is then classified as
    nuclear (inside the in-cell nucleus) or cytoplasmic (cell minus nucleus).

    Columns emitted (2, ``<prefix>``-prefixed):
        ``<prefix>_nuclear_puncta_fraction`` — fraction of the cell's detected spots
            whose centre lands in the nucleus ∈ [0, 1].
        ``<prefix>_cytoplasm_puncta_fraction`` — fraction whose centre lands in the
            cytoplasm (cell − nucleus) ∈ [0, 1]. The two sum to 1 (every in-cell spot
            is in exactly one compartment).

    NaN guard: a missing region / shape mismatch / all-NaN crop NaNs everything; a
    cell with ZERO detected spots gets 0.0 for both fractions (a defined "no nuclear /
    no cytoplasmic puncta" answer, not NaN) — schema-stable.

    Args:
        channel_crop: 3D intensity crop (Z, Y, X) at the cell bbox, aligned with
            ``cell_mask`` / ``nucleus_mask``.
        cell_mask: 3D boolean cell mask (the cell crop).
        nucleus_mask: 3D boolean IN-CELL nucleus mask (same crop frame).
        prefix: column-name prefix (matches the channel's intensity prefix).
        spacing: accepted for signature uniformity (the fractions are count-only; the
            blob scale search is index-based). None = voxel units.

    Returns:
        ``{f"{prefix}{suffix}": float}`` for the 2 compartment-fraction columns.
    """
    out = {f"{prefix}{s}": np.nan for s in _CHANNEL_PUNCTA_COMPARTMENT_SUFFIXES}

    cell = np.asarray(cell_mask).astype(bool)
    nuc = np.asarray(nucleus_mask).astype(bool)
    ch = np.asarray(channel_crop, dtype=np.float64)
    if ch.shape != cell.shape or cell.shape != nuc.shape or not cell.any():
        return out
    if not np.isfinite(ch[cell]).any():
        return out  # all-NaN crop -> schema-stable NaN

    kept = _detect_in_mask_puncta(ch, cell)
    n_spots = len(kept)
    if n_spots == 0:
        # no resolvable spots -> a defined zero in both compartments.
        out[f"{prefix}_nuclear_puncta_fraction"] = 0.0
        out[f"{prefix}_cytoplasm_puncta_fraction"] = 0.0
        return out

    nuc_in = nuc & cell
    n_nuclear = 0
    for b in kept:
        zc, yc, xc = int(round(b[0])), int(round(b[1])), int(round(b[2]))
        if nuc_in[zc, yc, xc]:
            n_nuclear += 1
    frac_nuc = n_nuclear / float(n_spots)
    out[f"{prefix}_nuclear_puncta_fraction"] = frac_nuc
    out[f"{prefix}_cytoplasm_puncta_fraction"] = 1.0 - frac_nuc
    return out


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

    NOTE (engine port): reads only EXISTING columns (degrades gracefully per
    column when an input is absent), so it is safe to run as the last step of
    the global adjacency pass right after :func:`compute_neighbor_features`.
    ``boundary_confidence`` needs the M2 ``prob_*`` columns; without them it is
    emitted as NaN.
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


# =========================================================================== #
# M7 — composite qc_score (global table pass scoring layer).
#
# Ported VERBATIM from the EpiCycle monorepo
# ``napari_epicycle/fusion_reconstruction/qc_features.py`` (the curated EDA
# transform rules + robust standardization + composite/PCA scoring) so the
# engine's whole-tissue ``qc_score`` joins the plugin's per-patch ``qc_score``
# without a later reconcile. Unlike the per-cell crop functions above, these
# operate on the WHOLE assembled table — they standardize each feature across
# the population, so they run as an OPT-IN GLOBAL pass at the very end (like
# ``compute_centroid_spacing``), AFTER neighbours / composites have populated
# the columns ``compute_composite_score`` reads.
#
# LEAKAGE WARNING (read before training a classifier): ``qc_score`` and the
# named composites (``merge_signal`` / ``fragment_signal`` / ``shape_abnormality``
# / ``boundary_confidence``) are DERIVED from the other feature columns. If a
# downstream ``status`` / tier / good-vs-bad LABEL is itself derived from
# ``qc_score`` or those composite signals, then ``qc_score`` (and any composite
# it aggregates) MUST be held OUT of the feature set of a classifier trained on
# that label — including them is circular (the model would learn its own target).
# =========================================================================== #

# Bounded [0,1] features — NEVER log-transform these regardless of skewness.
# log1p compresses their already-small range unhelpfully. (Also documents which
# extracted columns are mathematically bounded, complementing the manifest's
# per-column ``bounded`` flag.)
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
    # SIGNED bounded score columns ([-1, 1]). These MUST be listed: the auto-skew
    # path's log1p does ``log1p(np.clip(X, 0, None))``, which would ZERO the
    # negative half of a signed feature if it were flagged skewed. Listing them
    # here keeps the negative (basal / inner / low-Z) end intact.
    "prob_apical_basal_gradient",   # apical − basal prob  (P1a)
    "prob_shell_gradient",          # outer − inner prob shell
    "prob_boundary_drop",           # interior − boundary prob
    "contact_polarity_bias",        # apical − basal contact fraction
    "apical_vs_basal_elongation",   # (σ_apical − σ_basal)/(σ_apical + σ_basal)
    "territory_z_bias",             # signed Z offset of territory vs cell centroid
    "disc_vs_rod_score",            # (B/A) − (C/B): +disc / −rod
])

# Skewness threshold for auto-detection. Features with |skew| above this
# get log1p transform before PCA/composite scoring.
# 1.0 is standard in cytometry (pycytominer convention).
_SKEW_THRESHOLD = 1.0

# Curated log-transform set — always log1p these known size/intensity features
# before PCA/composite scoring (preserves the monorepo's validated PCA
# behaviour). Auto-detection (|skew| > 1) catches the rest. Also documents which
# columns are right-skewed-by-default, complementing the manifest.
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
    "true_sphericity",           # gold-standard shape (replaces bbox sphericity)
    "solidity",                  # vol/convex_hull (replaces fill_ratio, ellipsoid_solidity)
    # --- Orientation ---
    "major_axis_z_angle",        # tilt relative to Z axis
    "inertia_elongation",        # eigenvalue-based (replaces axis lengths, inertia_flatness)
    # --- Surface ---
    "surface_area",              # independent from volume-based features
    # --- Topology ---
    "neighbor_count",            # adjacency (discrete but unique signal)
    "nearest_neighbor_distance", # gap width
    "skeleton_branches",         # complexity (discrete but unique signal)
    # --- Probability consensus (when available) ---
    "prob_mean",                 # overall confidence (represents prob_p10/std)
    "prob_boundary_drop",        # spatial pattern (unique signal)
    "prob_boundary_entropy",     # boundary uncertainty (unique signal)
]


def compute_composite_score(
    df: pd.DataFrame,
    method: str = "zscore",
    features: Optional[List[str]] = None,
    exclude_border: bool = False,
    n_pcs: int = 5,
) -> pd.DataFrame:
    """Compute a composite QC outlier score for each label (M7; global pass).

    Ported byte-identical from the EpiCycle monorepo. A whole-table pass: it
    robust-standardizes the curated feature set ACROSS the population, so it
    must run AFTER the per-cell + global passes have populated those columns.

    LEAKAGE: ``qc_score`` is derived from the other feature columns (and the
    named composites it can read). If a downstream label / tier is itself
    derived from ``qc_score`` or those composites, hold them OUT of a classifier
    trained on that label — including them is circular (see the M7 header note).

    Args:
        df: Feature DataFrame (modified in-place).
        method: ``"zscore"`` — mean absolute z-score across selected features
            (simple, interpretable). ``"pca"`` — L2 norm in the top-``n_pcs``
            PCA component space (geometric distance from the average cell).
            Higher = more outlier-like.
        features: Optional list of feature column names to include.
            Defaults to :data:`PCA_FEATURES` (curated, ~16 independent features).
        exclude_border: If True and ``is_border_cell`` column exists, exclude
            border cells from median/MAD fitting (they get scores but don't
            influence the axes). Default False for initial extraction.
        n_pcs: Number of PCA components to integrate over when
            ``method="pca"``.  Capped to ``min(len(features), n_samples-1)``.
            Ignored for ``method="zscore"``.  Default 5.

    Returns:
        The same DataFrame with ``qc_score`` column added.
    """
    src = features if features is not None else PCA_FEATURES
    cols = [c for c in src if c in df.columns and c != "qc_score"]
    if not cols or len(df) < 2:
        df["qc_score"] = 0.0
        return df

    if method == "pca":
        # If persisted PC1..PCN columns exist (compute_pca_columns), take the
        # L2 norm directly — they were computed via the same SVD pipeline this
        # branch's fresh fit would do. Fall back to the inline SVD when absent.
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
            # Project ALL rows onto the fitted PC axes (variance-scaled coords).
            pc_scores = X_std @ Vt[:n_use].T  # shape (n_samples, n_use)
            # Apply singular-value scaling so high-variance PCs contribute more
            # (matches the "U[:,i] * S[i]" semantics of the non-exclude branch).
            pc_scores = pc_scores * (S[:n_use] / S[:1])
        else:
            U, S, _Vt = np.linalg.svd(X_std, full_matrices=False)
            pc_scores = U[:, :n_use] * S[:n_use]  # (n_samples, n_use)
        # L2 norm = geometric distance from the average cell across the top
        # n_pcs PC axes.  Higher = more outlier-like.
        score = np.linalg.norm(pc_scores, axis=1)
        # Shift to non-negative (L2 norm >= 0 already; explicit for parity).
        score = score - score.min()
        df["qc_score"] = score
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'zscore' or 'pca'.")

    return df


def _get_pc_matrix(
    df: pd.DataFrame,
    n_components: int,
) -> Optional[np.ndarray]:
    """Return persisted ``PC1..PC{n_components}`` as a contiguous matrix.

    Helper used by :func:`compute_composite_score` to read canonical PC columns
    when they exist (written by :func:`compute_pca_columns`).  Returns ``None``
    if ``df`` has fewer than ``n_components`` PC columns — callers fall back to a
    fresh fit in that case.
    """
    if df is None or df.empty or n_components < 1:
        return None
    needed = [f"PC{i + 1}" for i in range(n_components)]
    if not all(c in df.columns for c in needed):
        return None
    return df[needed].values.astype(np.float64)


def compute_pca_full(
    df: pd.DataFrame,
    n_components: int = 10,
    features: Optional[List[str]] = None,
    exclude_border: bool = False,
    mode: str = "qc",
    winsorize: Optional[float] = None,
) -> dict:
    """PCA projection with loadings and explained variance (M7 helper).

    Ported from the monorepo. The monorepo's ``transforms`` (per-feature EDA
    transform dict) branch is OMITTED here — it imports a plugin-UI module
    (``feature_stats.apply_transform``) and is never reached by the engine's
    opt-in QC-mode pass. The QC-default path (``mode="qc"`` + no transforms),
    which :func:`compute_pca_columns` uses, is byte-identical to the monorepo.

    Args:
        df: Feature DataFrame.
        n_components: Number of principal components (default 10).
        features: Feature columns.  Defaults to :data:`PCA_FEATURES`.
        exclude_border: Exclude border cells from fitting.
        mode: ``"qc"`` (robust median/MAD via :func:`_robust_standardize`) or
            ``"eda"`` (standard mean/std + curated log1p).
        winsorize: Optional clip in ``"eda"`` mode (``"qc"`` mode self-winsorizes
            to ±10 inside :func:`_robust_standardize`).

    Returns:
        dict with keys ``projected`` (n_samples, n_components),
        ``loadings`` (n_components, n_features) — the Vt matrix,
        ``explained_variance_ratio`` (n_components,), ``feature_names`` (list).
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

    # Exclude border cells from fitting (always, when the column exists).
    exclude = None
    if "is_border_cell" in df.columns:
        exclude = df["is_border_cell"].values.astype(bool)

    if mode == "eda":
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
        if winsorize is not None:
            X_std = np.clip(X_std, -winsorize, winsorize)
    else:
        # QC mode — full _robust_standardize (its own log + skew detection +
        # MAD + winsorize ±10).
        X_std = _robust_standardize(X, cols, exclude_mask=exclude)

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


def compute_pca_columns(
    df: pd.DataFrame,
    n_components: int = 5,
    features: Optional[List[str]] = None,
    exclude_border: bool = False,
) -> pd.DataFrame:
    """Fit PCA once and write ``PC1..PC{n_components}`` columns to ``df`` (M7).

    Canonical PCA persistence so the composite qc_score (method="pca"), and any
    downstream clustering, can read from the same coordinate system instead of
    independently re-fitting.  Uses :func:`compute_pca_full` in QC mode (the same
    robust median/MAD standardization the zscore path uses).  When
    ``exclude_border=True`` the SVD fits on non-border rows; all rows are
    projected with the fitted Vt.

    Returns the same df for chainability.  Modifies in place. Writes fewer than
    ``n_components`` PC columns when the data rank is lower (degenerate input).
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


__all__ = [
    "FEATURE_COLUMNS",
    "extract_features",
    "compute_surface_features",
    "compute_discrete_compactness",
    "compute_skeleton_features",
    "compute_cross_section_features",
    "compute_cross_section_profile_stats",
    "compute_triaxial",
    "_object_edt",
    "compute_edt_features",
    "compute_geodesic_features",
    "compute_ma_cross_section_features",
    "compute_shell_features",
    # Wave H — multi-scale (zone-restricted) SHAPE templates (per-cell)
    "compute_multiscale_shape",
    "MULTISCALE_SHAPE_COLUMNS",
    "compute_curvature_features",
    # Wave B — reliability / confidence meta (table pass)
    "compute_reliability_meta",
    "compute_boundary_interior_features",
    "compute_channel_intensity",
    # Wave E — per-channel texture (3D-native + 2D-per-slice GLCM/Haralick)
    "compute_channel_texture",
    "compute_probability_features",
    # Wave F — probability-geometry interaction + channel localization
    "compute_prob_geometry",
    "compute_channel_localization",
    "compute_channel_compartments",
    # P1b — marker-expectation QC flags (paired path, biology config)
    "compute_marker_expectation_flags",
    # P1b — at-nucleus / perinuclear / neighbour local expression context
    "compute_channel_local_context",
    "compute_channel_neighbor_context",
    "compute_border_cells",
    "compute_centroid_spacing",
    "compute_neighbor_features",
    "write_edges_csv",
    "run_adjacency_graph_passes",
    "compute_contact_geometry",
    "CONTACT_GEOMETRY_COLUMNS",
    "compute_tissue_frame_features",
    # Wave A — adjacency-graph-derived families
    "compute_graph_topology",
    "GRAPH_TOPOLOGY_COLUMNS",
    "compute_axis_neighbor_persistence",
    "compute_local_comparative",
    "LOCAL_COMPARATIVE_FEATURES",
    "compute_graph_motifs",
    "GRAPH_MOTIF_COLUMNS",
    # Wave C — tissue-depth ("in/out") + spatial context + outward normal
    "compute_tissue_depth",
    "TISSUE_DEPTH_COLUMNS",
    # P1b — lumen-cavity primitive (distance_to_lumen + lumen_facing_fraction)
    "compute_lumen_features",
    "LUMEN_COLUMNS",
    # Wave D — signed-frame families (directional contact + surface partition)
    "compute_directional_contact",
    "DIRECTIONAL_CONTACT_COLUMNS",
    "compute_surface_partition",
    "SURFACE_PARTITION_COLUMNS",
    # Wave H — signed-frame remainders (major-axis apical/basal + apical junction)
    "compute_ma_apical_basal",
    "MA_APICAL_BASAL_COLUMNS",
    "compute_apical_junction",
    "APICAL_JUNCTION_COLUMNS",
    # Wave G — watershed territory / crowding + HCR puncta
    "compute_watershed_territory",
    "WATERSHED_TERRITORY_COLUMNS",
    "compute_channel_puncta",
    # Wave K — channel-pair co-localization + nuclear/cytoplasm
    # puncta fraction (paired path)
    "compute_channel_colocalization",
    "compute_channel_puncta_compartments",
    "compute_named_composites",
    # M7 — composite qc_score (global table pass)
    "BOUNDED_FEATURES",
    "LOG_TRANSFORM_FEATURES",
    "PCA_FEATURES",
    "compute_composite_score",
    "compute_pca_columns",
    "compute_pca_full",
]
