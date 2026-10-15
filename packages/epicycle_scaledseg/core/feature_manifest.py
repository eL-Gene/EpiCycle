"""Per-feature MANIFEST — the discipline that makes "extract everything" safe.

A plain-dict tag table for the extracted feature columns. The EDA layer reads
these tags to size-normalize, log/logit-transform, regress out depth, and prune
collinear blocs **per dataset** — pruning/decorrelation/depth-control live in
the analysis layer, never in the engine.

Each entry tags one column:

* ``units`` — ``"voxel"`` (or µm when ``spacing`` was used; unit-bearing),
  ``"um"`` (always physical), ``"ratio"`` (dimensionless), ``"count"``,
  ``"angle"``, or ``"score"`` (a bounded/derived composite).
* ``bounded`` — ``True`` when the value has FINITE MATHEMATICAL BOUNDS, so the
  EDA auto-skew rule skips log1p on it (mirrors the monorepo ``BOUNDED_FEATURES``
  convention). NOTE: ``bounded=True`` means "has finite bounds → EDA skips
  log1p", NOT "∈ [0,1]" — it also covers [-1,1] signed scores (correlations,
  signed differences) and [0,90°] angles. Downstream code must NOT assume a
  bounded column is non-negative or in the unit interval; read ``units`` + the
  column's own docstring for the actual range.
* ``role`` — ``"biology"`` (a morphology / phenotype descriptor), ``"qc"`` (a
  segmentation-quality / reliability signal), or ``"covariate"`` (a DEPTH /
  POSITION / RAW-INTENSITY confound that should be REGRESSED OUT / held out of
  the morphospace, never stamped as phenotype — e.g. tissue depth, centroid
  position, distance-to-surface/centroid, raw absolute channel intensity).
  Cross-section + triaxial M0 features are biology; they double as soft QC but
  their primary role is shape. The SELF-NORMALIZING intensity ratios
  (``<ch>_intensity_cv`` / ``<ch>_boundary_interior_ratio``) stay biology — only
  the RAW absolute intensity (mean/median/min/max/sum/boundary/interior/z-var) is
  a covariate (it tracks staining/illumination, not phenotype).
* ``sign_trusted`` — OPTIONAL; present + ``False`` on the SIGNED apical/basal
  tissue-frame columns. The apical/basal SIGN comes from the Wave C outward
  normal = "toward the NEAREST tissue surface", which only equals true APICAL for
  a CURVED / CLOSED tissue or a basal-cell-free FOV. When the
  sign is wrong these columns' VALUE flips meaning (apical↔basal), so a downstream
  reader should treat them as low-trust / magnitude-only unless the tissue
  geometry guarantees the sign. Absent ⇒ not a signed apical/basal column.
* ``size_correlated`` — OPTIONAL; present + ``True`` on the columns that are ~co-
  linear with overall cell size (volumes / areas / lengths / eigenvalues /
  thickness). An EDA layer can collapse them to ONE representative before PCA so
  the first PC is not just "size". Orthogonal to ``redundancy_group`` (which
  blocs by SEMANTIC family, not by the size manifold).
* ``redundancy_group`` + ``size_correlated`` are EDA-CONSUMED CONTRACTS — the
  ENGINE does not prune / decorrelate / drop anything (it always emits every
  column). A sensible EDA default is "one REPRESENTATIVE per redundancy_group,
  size-collapsed via ``size_correlated``" rather than "all role=biology columns"
  (the latter is dominated by size-PC1).
* ``frame`` — ``"intrinsic"`` (object-only), ``"image"`` (uses image Z / global
  axes), or ``"tissue"`` (uses a local / whole-tissue frame). M0 cross-section
  profile stats are global-Z proxies (``image``); triaxial descriptors are
  ``intrinsic`` (principal-axis derived, rotation-invariant). CAUTION — the
  ``apical_*`` / ``basal_*`` NAME is reused across THREE frames: the image-Z
  cross-section family (``apical_area`` / ``basal_area``, ``frame="image"``), the
  cell's-own-axis profile (``ma_apical_*``, an INTRINSIC notion oriented by the
  tissue normal), and the signed tissue frame (``apical_surface_area`` etc.,
  ``frame="tissue"``). DISAMBIGUATE such a column by its ``frame`` tag, NEVER by
  the name alone (the catalogue semantic-drift caution).
* ``redundancy_group`` — a coarse bloc label so the EDA layer can decorrelate
  within a group without hand-listing every column.

NOTE (scope): the BASELINE M0/M1/M5-era intrinsic morphology columns emitted by
the LABELS-ONLY per-cell pass (``reconcile.features.extract_label_features`` /
the ``FeatureStage`` driver) are ALL tagged — the size manifold (area / surface
/ axis lengths / equivalent_diameter), the sphericity family, elongation /
aspect ratios, the orientation basis (axis components + inertia eigenvalues),
the topology / skeleton / roughness QC proxies, and the cross-section apical /
basal columns — plus M0/M1/M5 + the later waves below.  ``tests/
test_manifest_completeness.py`` asserts EVERY labels-only-emitted feature column
(excluding the ``label`` / ``centroid-*`` / ``bbox-*`` coordinate metadata) has
a tag, so this can't silently reopen.  Opt-in / dynamic columns (per-channel,
co-localization, local-comparative, the adjacency / tissue-depth / qc_score
global passes) are covered by their own wave tests via the ``*_tags`` helper
templates (they can't be statically enumerated — the prefix / base-feature set
is a runtime choice).

ENTRY POINTS differ by design: ``extract_label_features`` emits the per-cell
intrinsic + (when enabled) the global-pass columns; the ``FeatureStage`` DRIVER
additionally emits the ``crowding`` (cKDTree neighbour-spacing) + ``provenance``
(seam / IoU / tile bookkeeping) families, which the bare ``extract_label_features``
does NOT produce — so the two produce different column sets (the completeness
test is scoped to the ``extract_label_features`` labels-only set).

``get_manifest()`` returns whatever is currently registered; a column absent
from the manifest is simply untagged, not an error.
"""

from __future__ import annotations

from typing import Dict


# DONE(manifest backfill, 2026-06-21): the baseline Wave-1 intrinsic morphology
# columns (area / sphericity / surface_area / axis basis / skeleton_* /
# apical_area / cross_section_* / ...) are now tagged in the BASELINE block at
# the top of FEATURE_MANIFEST below, enforced by tests/test_manifest_completeness.py.
# The crowding (nn_*) family is emitted by the FeatureStage DRIVER, not by
# extract_label_features, so it is tagged separately (see that test's scope note).

FEATURE_MANIFEST: Dict[str, dict] = {
    # ================================================================= #
    # BASELINE — M0/M1/M5-era intrinsic morphology.
    # The original Wave-1 columns emitted by the LABELS-ONLY per-cell pass
    # (``reconcile.features.extract_label_features`` / ``FeatureStage``).
    # Backfilled 2026-06-21 (the TODO at the top of this file); covered by
    # ``tests/test_manifest_completeness.py`` so the gap cannot silently
    # reopen.  units="voxel" = the spacing-bearing convention (voxel counts,
    # or µm/µm²/µm³ when ``spacing`` was given — see the module docstring;
    # the engine uses ONE "voxel" token for every spatial measure, see the
    # M1 edt_* / contact_area_* entries).  POSITIONAL columns (``centroid-*``
    # / ``bbox-*``) are coordinate metadata (like ``label`` / ``cell_id``),
    # NOT features, so they are deliberately un-tagged + excluded from the
    # completeness test.
    # ----------------------------------------------------------------- #
    # Size manifold — volumes / areas / lengths.  All scale with spacing
    # (units="voxel"), unbounded, shape biology, object-only (intrinsic).
    # ----------------------------------------------------------------- #
    "area": {  # volume V (voxel count, or µm³ with spacing) — the primary size measure
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "size_manifold", "size_correlated": True,
    },
    "area_filled": {  # hole-filled volume
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "size_manifold", "size_correlated": True,
    },
    "bbox_volume": {  # bounding-box volume bz*by*bx
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "size_manifold", "size_correlated": True,
    },
    "equivalent_diameter": {  # diameter of the equal-volume sphere (length)
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "size_manifold", "size_correlated": True,
    },
    "z_span": {  # bbox extent along Z (length)
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "size_manifold", "size_correlated": True,
    },
    "axis_major_length": {  # 2a — longest ellipsoid axis (length)
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "size_manifold", "size_correlated": True,
    },
    "axis_minor_length": {  # 2c — shortest ellipsoid axis (length)
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "size_manifold", "size_correlated": True,
    },
    "surface_area": {  # mesh surface area (area)
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "size_manifold", "size_correlated": True,
    },
    # ----------------------------------------------------------------- #
    # Sphericity / compactness family — dimensionless shape ratios, all
    # mathematically bounded [0,1] (mirrors the monorepo BOUNDED_FEATURES:
    # sphericity / true_sphericity / solidity / extent / fill_ratio /
    # ellipsoid_solidity / discrete_compactness).  Shape biology, intrinsic.
    # ----------------------------------------------------------------- #
    "sphericity": {  # sphere_SA / bbox_SA — bbox-referenced roundness [0,1]
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "sphericity_family",
    },
    "true_sphericity": {  # (36πV²)^(1/3) / SA — gold-standard sphericity [0,1]
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "sphericity_family",
    },
    "solidity": {  # V / V_convex_hull [0,1]
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "sphericity_family",
    },
    "extent": {  # V / V_bbox [0,1]
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "sphericity_family",
    },
    "fill_ratio": {  # V / V_filled [0,1] (<1 = internal holes)
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "sphericity_family",
    },
    "ellipsoid_solidity": {  # V / V_fitted_ellipsoid [0,1]
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "sphericity_family",
    },
    "discrete_compactness": {  # Bribiesca compactness [0,1] (1 = cube)
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "sphericity_family",
    },
    # ----------------------------------------------------------------- #
    # Elongation / aspect ratios — dimensionless.  aspect_ratio /
    # xy_elongation / z_elongation / inertia_elongation are unbounded >=0
    # (bbox / inertia axis ratios); inertia_flatness = sqrt(I1/I0) ∈ [0,1]
    # is bounded.  Shape biology, intrinsic.
    # ----------------------------------------------------------------- #
    "aspect_ratio": {  # max(bbox dims) / min(bbox dims) — >=1, unbounded
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "elongation",
    },
    "z_elongation": {  # bz / max(by, bx) — unbounded
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "elongation",
    },
    "xy_elongation": {  # max(by, bx) / min(by, bx) — >=1, unbounded
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "elongation",
    },
    "inertia_elongation": {  # sqrt(I0/I2) — >=1, unbounded
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "elongation",
    },
    "inertia_flatness": {  # sqrt(I1/I0) ∈ [0,1] — bounded
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "elongation",
    },
    # ----------------------------------------------------------------- #
    # Orientation — the principal-axis basis (the feature catalogue Base Orientation/
    # Inertia).  major_axis_z_angle is the tilt from Z in DEGREES, bounded
    # [0,90] (mirrors the canonical nuclear alias nuclear_major_axis_z_angle).
    # The {major,mid,minor}_axis_{z,y,x} components are unit-vector direction
    # cosines ∈ [-1,1] (units="ratio", bounded).  The inertia eigenvalues are
    # mass-normalized second moments (spacing²-scaled, units="voxel",
    # unbounded).  Shape biology, intrinsic.
    # ----------------------------------------------------------------- #
    "major_axis_z_angle": {  # tilt of the major axis from Z, degrees [0,90]
        "units": "degrees", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "orientation",
    },
    "major_axis_z": {  # direction cosine ∈ [-1,1]
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "orientation",
    },
    "major_axis_y": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "orientation",
    },
    "major_axis_x": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "orientation",
    },
    "mid_axis_z": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "orientation",
    },
    "mid_axis_y": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "orientation",
    },
    "mid_axis_x": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "orientation",
    },
    "minor_axis_z": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "orientation",
    },
    "minor_axis_y": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "orientation",
    },
    "minor_axis_x": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "orientation",
    },
    "inertia_eigval_major": {  # mass-normalized 2nd moment (spacing²), unbounded
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "orientation", "size_correlated": True,
    },
    "inertia_eigval_mid": {
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "orientation", "size_correlated": True,
    },
    "inertia_eigval_minor": {
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "orientation", "size_correlated": True,
    },
    # ----------------------------------------------------------------- #
    # Topology / surface-quality proxies — role="qc" (segmentation-quality /
    # complexity signals, NOT phenotype; matches how the catalogue + monorepo
    # treat euler_number / skeleton_* / roughness).  euler_number +
    # skeleton_branches/nodes are integer counts; skeleton_length is a length
    # (units="voxel"); roughness = 1/true_sphericity (>=1 unbounded ratio).
    # ----------------------------------------------------------------- #
    "euler_number": {  # connected-components minus holes (topology integer)
        "units": "count", "bounded": False, "role": "qc",
        "frame": "intrinsic", "redundancy_group": "topology",
    },
    "skeleton_branches": {  # skeleton branch count
        "units": "count", "bounded": False, "role": "qc",
        "frame": "intrinsic", "redundancy_group": "skeleton",
    },
    "skeleton_nodes": {  # skeleton node count
        "units": "count", "bounded": False, "role": "qc",
        "frame": "intrinsic", "redundancy_group": "skeleton",
    },
    "skeleton_length": {  # total skeleton path length
        "units": "voxel", "bounded": False, "role": "qc",
        "frame": "intrinsic", "redundancy_group": "skeleton",
    },
    "roughness": {  # 1 / true_sphericity — >=1 surface-irregularity ratio
        "units": "ratio", "bounded": False, "role": "qc",
        "frame": "intrinsic", "redundancy_group": "sphericity_family",
    },
    # ----------------------------------------------------------------- #
    # Cross-section apical/basal — global-Z proxies
    # (frame="image", like the M0 cross_section_profile family).
    # apical_area/basal_area are areas (units="voxel"); apical_basal_area_ratio
    # is an unbounded >=0 ratio; max_cross_section_z_relative is a [0,1]
    # position (bounded, mirrors the monorepo BOUNDED_FEATURES);
    # cross_section_taper is a SIGNED slope (units="score"); cross_section_bulge
    # is a normalized RMS residual (unbounded >=0 ratio).  Shape biology.
    # ----------------------------------------------------------------- #
    "apical_area": {  # mean cross-section over the apical 20% of Z (area)
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "image", "redundancy_group": "cross_section_apical_basal",
    },
    "basal_area": {  # mean cross-section over the basal 20% of Z (area)
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "image", "redundancy_group": "cross_section_apical_basal",
    },
    "apical_basal_area_ratio": {  # apical / basal — unbounded >=0 ratio
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "image", "redundancy_group": "cross_section_apical_basal",
    },
    "max_cross_section_z_relative": {  # Z-position of widest slice [0,1]
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "image", "redundancy_group": "cross_section_apical_basal",
    },
    "cross_section_taper": {  # signed slope of area vs Z
        "units": "score", "bounded": False, "role": "biology",
        "frame": "image", "redundancy_group": "cross_section_apical_basal",
    },
    "cross_section_bulge": {  # normalized RMS residual of the area-vs-Z fit
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "image", "redundancy_group": "cross_section_apical_basal",
    },
    # ----------------------------------------------------------------- #
    # M0 — Cross-section profile distribution stats.
    # Global-Z proxies (frame="image"): they summarize the per-Z area
    # profile A(z). Primarily shape biology; double as merge/neck QC.
    # ----------------------------------------------------------------- #
    "min_cross_section_area": {
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "image", "redundancy_group": "cross_section_profile",
    },
    "cross_section_area_range": {
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "image", "redundancy_group": "cross_section_profile",
    },
    "cross_section_area_cv": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "image", "redundancy_group": "cross_section_profile",
    },
    "cross_section_area_skew": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "image", "redundancy_group": "cross_section_profile",
    },
    "cross_section_waist_depth": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "image", "redundancy_group": "cross_section_profile",
    },
    "cross_section_peak_count": {
        "units": "count", "bounded": False, "role": "qc",
        "frame": "image", "redundancy_group": "cross_section_profile",
    },
    # Wave I — cross-section REMAINDER (the feature catalogue; same A(z) profile, frame="image").
    # kurtosis/constriction are dimensionless (unbounded); min_z_relative is a [0,1]
    # position (bounded); n_lobes is an unbounded count (a synonym of peak_count).
    "cross_section_area_kurtosis": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "image", "redundancy_group": "cross_section_profile",
    },
    "min_cross_section_z_relative": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "image", "redundancy_group": "cross_section_profile",
    },
    "cross_section_constriction_index": {
        "units": "ratio", "bounded": False, "role": "qc",
        "frame": "image", "redundancy_group": "cross_section_profile",
    },
    "cross_section_n_lobes": {
        "units": "count", "bounded": False, "role": "qc",
        "frame": "image", "redundancy_group": "cross_section_profile",
    },
    # Wave N — cross-section REMAINDER. basal_flare_index is an
    # unbounded >=0 ratio (basal / mean); the peak/trough prominences are
    # range-normalized [0,1] (bounded); the slice-moment summaries: centroid
    # drift mean/max are unbounded >=0 (size-normalized), eccentricity mean is a
    # [0,1] in-plane eccentricity (bounded), eccentricity taper is a signed slope
    # (unbounded), orientation twist is an unbounded >=0 radian accumulation.
    # frame="image" (per-Z slice moments).
    "basal_flare_index": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "image", "redundancy_group": "cross_section_profile",
    },
    "cross_section_peak_prominence": {
        "units": "ratio", "bounded": True, "role": "qc",
        "frame": "image", "redundancy_group": "cross_section_profile",
    },
    "cross_section_trough_prominence": {
        "units": "ratio", "bounded": True, "role": "qc",
        "frame": "image", "redundancy_group": "cross_section_profile",
    },
    "slice_centroid_drift_mean": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "image", "redundancy_group": "slice_moments",
    },
    "slice_centroid_drift_max": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "image", "redundancy_group": "slice_moments",
    },
    "slice_eccentricity_mean": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "image", "redundancy_group": "slice_moments",
    },
    "slice_eccentricity_taper": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "image", "redundancy_group": "slice_moments",
    },
    "slice_orientation_twist": {
        "units": "radians", "bounded": False, "role": "biology",
        "frame": "image", "redundancy_group": "slice_moments",
    },
    # ----------------------------------------------------------------- #
    # M0 — Triaxial / three-axis descriptors.
    # Principal-axis derived -> rotation-invariant (frame="intrinsic").
    # ----------------------------------------------------------------- #
    "axis_mid_length": {
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "triaxial_axes", "size_correlated": True,
    },
    "triaxiality": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "triaxial_shape",
    },
    "disc_vs_rod_score": {
        "units": "score", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "triaxial_shape",
    },
    # Wave I — axis-ratio REMAINDER: pure ratios of the ordered
    # axis lengths A>=B>=C. mid/major + minor/mid are (0,1] (bounded);
    # shape_anisotropy_3d (Westin FA of A,C) is [0,1] (bounded). frame="intrinsic".
    "axis_mid_to_major_ratio": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "triaxial_shape",
    },
    "axis_minor_to_mid_ratio": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "triaxial_shape",
    },
    "shape_anisotropy_3d": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "triaxial_shape",
    },
    # ----------------------------------------------------------------- #
    # M1 — Interior-EDT thickness field.
    # The distance-transform field is object-only -> rotation-invariant
    # (frame="intrinsic"). edt_max/mean/p90 carry the spacing unit;
    # edt_cv/skew are dimensionless. Thickness biology; doubles as a
    # soft sheet-vs-blob QC signal.
    # ----------------------------------------------------------------- #
    "edt_max": {
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "edt_thickness",
    },
    "edt_mean": {
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "edt_thickness", "size_correlated": True,
    },
    "edt_p90": {
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "edt_thickness", "size_correlated": True,
    },
    "edt_cv": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "edt_distribution",
    },
    "edt_skew": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "edt_distribution",
    },
    # Wave I — medial-axis REMAINDER. The interior-EDT IS the medial
    # radius field: medial_radius_mean ALIASES edt_mean (spacing unit), medial_radius_cv
    # ALIASES edt_cv (dimensionless); edt_core_fraction is a genuinely-new [0,1]
    # thick-core coverage (bounded). frame="intrinsic".
    "medial_radius_mean": {
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "edt_thickness", "size_correlated": True,
    },
    "medial_radius_cv": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "edt_distribution",
    },
    "edt_core_fraction": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "edt_distribution",
    },
    # Wave N — medial-radius RANGE: max - min of the interior
    # medial radius field. Carries the spacing unit (unbounded). frame="intrinsic".
    "medial_radius_range": {
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "edt_thickness",
    },
    # ----------------------------------------------------------------- #
    # M1 — Geodesic / tortuosity.
    # In-mask path geometry; object-only (frame="intrinsic"). geodesic_max
    # carries the spacing unit; tortuosity is a >=1 dimensionless ratio
    # (NOT bounded above). Bend/winding shape biology + a merged-cell QC cue.
    # ----------------------------------------------------------------- #
    "geodesic_max": {
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "geodesic", "size_correlated": True,
    },
    "tortuosity": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "geodesic",
    },
    # Wave N — skeleton main-path / medial-axis geometry. The
    # cell's medial "spine": skeletonize, take the two most-separated skeleton
    # voxels, trace the in-skeleton geodesic. main_path_length carries the
    # spacing unit (unbounded); main_path_tortuosity is a >=1 ratio (unbounded);
    # endpoint_distance_ratio is a (0,1] straightness ratio (bounded);
    # medial_axis_bend_energy is a length-normalized squared-curvature score
    # (unbounded >=0); medial_branch_count is an unbounded count. frame="intrinsic"
    # (object-only medial axis). branch_count doubles as a fork/merge QC cue.
    "skeleton_main_path_length": {
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "geodesic", "size_correlated": True,
    },
    "skeleton_main_path_tortuosity": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "geodesic",
    },
    "endpoint_distance_ratio": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "geodesic",
    },
    "medial_axis_bend_energy": {
        "units": "score", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "geodesic",
    },
    "medial_branch_count": {
        "units": "count", "bounded": False, "role": "qc",
        "frame": "intrinsic", "redundancy_group": "geodesic",
    },
    # ----------------------------------------------------------------- #
    # M1 — Major-axis cross-section profile.
    # Tilt-robust analogues of the global-Z cross-section family: slabs are
    # perpendicular to the cell's OWN principal axis -> rotation-invariant
    # (frame="intrinsic", unlike the image-Z cross_section_profile group).
    # Areas are voxel counts; waist position is a [0,1] ratio.
    # ----------------------------------------------------------------- #
    "ma_min_cross_section_area": {
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "ma_cross_section_profile",
    },
    "ma_max_cross_section_area": {
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "ma_cross_section_profile",
    },
    "ma_cross_section_range": {
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "ma_cross_section_profile",
    },
    "ma_cross_section_taper": {
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "ma_cross_section_profile",
    },
    "ma_cross_section_waist_depth": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "ma_cross_section_profile",
    },
    # Wave I — major-axis profile CV; dimensionless, unbounded.
    "ma_cross_section_cv": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "ma_cross_section_profile",
    },
    # Wave N — major-axis profile REMAINDER. max_axis_relative is a
    # [0,1] widest-slab position (bounded); bulge is an unbounded >=0 residual
    # ratio; peak_count an unbounded count; skew a dimensionless distribution skew
    # (unbounded). frame="intrinsic" (the cell's own axis).
    "ma_max_cross_section_axis_relative": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "ma_cross_section_profile",
    },
    "ma_cross_section_bulge": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "ma_cross_section_profile",
    },
    "ma_peak_count": {
        "units": "count", "bounded": False, "role": "qc",
        "frame": "intrinsic", "redundancy_group": "ma_cross_section_profile",
    },
    "ma_cross_section_skew": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "ma_cross_section_profile",
    },
    # ----------------------------------------------------------------- #
    # M5 — Radial / shell decomposition.
    # Concentric shells from the M1 interior-EDT radial position r∈[0,1]
    # (1=core, 0=surface). Object-only distance field -> rotation-invariant
    # (frame="intrinsic"). The three volume fractions are [0,1] + sum to 1
    # (bounded); core_surface_ratio is an unbounded >=0 ratio; the radial
    # mean is [0,1] (bounded); its std is a small unbounded spread. Shape
    # biology (thick-core vs thin-rind), doubles as a soft sheet-vs-blob QC.
    # ----------------------------------------------------------------- #
    "shell_volume_frac_inner": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "shell",
    },
    "shell_volume_frac_mid": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "shell",
    },
    "shell_volume_frac_outer": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "shell",
    },
    "core_surface_ratio": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "shell",
    },
    "mean_radial_position": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "shell",
    },
    "radial_position_std": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "shell",
    },
    # Wave N — radial/shell REMAINDER. Per-band Bribiesca discrete
    # compactness (~[0,1], bounded) + per-band 3D anisotropy/eccentricity ([0,1],
    # bounded) + core_offset (unbounded >=0 ratio, the core's off-centre distance /
    # equiv-radius) + two volume ratios (unbounded >=0). frame="intrinsic" (the
    # object's own distance-field shells). Shape biology (core-vs-rind shape +
    # asymmetry); the per-band compactness doubles as a soft fragmentation cue.
    "shell_compactness_inner": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "shell",
    },
    "shell_compactness_mid": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "shell",
    },
    "shell_compactness_outer": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "shell",
    },
    "shell_eccentricity_inner": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "shell",
    },
    "shell_eccentricity_mid": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "shell",
    },
    "shell_eccentricity_outer": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "shell",
    },
    "core_offset": {
        "units": "ratio", "bounded": False, "role": "qc",
        "frame": "intrinsic", "redundancy_group": "shell",
    },
    "shell_volume_ratio_inner_outer": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "shell",
    },
    "shell_volume_ratio_mid_outer": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "shell",
    },
    # ----------------------------------------------------------------- #
    # Wave H — MULTI-SCALE (zone-restricted) SHAPE templates.
    # A per-cell crop pass (qc_features.compute_multiscale_shape) applying a MODEST
    # curated SHAPE template across the M5 EDT radial shells: discrete (Bribiesca)
    # compactness of the inner core / outer rind bands + the 1-voxel SKIN (surface-
    # to-bulk) fraction + the inner/outer compactness ratio. Object-only distance
    # field -> frame="intrinsic". The two band compactnesses are ~[0,1] (Bribiesca,
    # bounded); shell_surface_ratio is a [0,1] skin-voxel fraction (bounded);
    # shell_compactness_ratio is an unbounded >=0 ratio. role="biology" (core-vs-rind
    # shape; doubles as a soft fragmentation QC cue). NaN for a band too small to
    # define.
    # ----------------------------------------------------------------- #
    "shell_sphericity_inner": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "multiscale_shape",
    },
    "shell_sphericity_outer": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "multiscale_shape",
    },
    "shell_surface_ratio": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "multiscale_shape",
    },
    "shell_compactness_ratio": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "multiscale_shape",
    },
    # Wave N — multi-scale REMAINDER. shell_sphericity_mid is the
    # mid-band Bribiesca compactness (~[0,1], bounded; completes inner/mid/outer);
    # the axis-quartile template emits the first/last axial-quartile volume
    # fractions ([0,1], bounded) + their ratio (unbounded >=0). frame="intrinsic".
    "shell_sphericity_mid": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "multiscale_shape",
    },
    "shape_axis_q1_fraction": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "multiscale_shape",
    },
    "shape_axis_q4_fraction": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "multiscale_shape",
    },
    "shape_axis_q1_q4_ratio": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "multiscale_shape",
    },
    # ----------------------------------------------------------------- #
    # M5 — Surface curvature / concavity / symmetry.
    # From the SAME marching-cubes mesh compute_surface_features builds + the
    # convex hull + the major-axis mid-plane reflection. Object-only ->
    # rotation-invariant (frame="intrinsic"). Curvature stats carry an
    # inverse-length unit (unbounded); convexity is a (0,1] ratio (bounded);
    # concavity_count is an unbounded count; max_concavity_depth a length
    # (unbounded); bilateral_symmetry_score is a [0,1] IoU (bounded). Shape
    # biology (smooth single cell vs lobed/pinched), doubles as a merge QC cue.
    # ----------------------------------------------------------------- #
    "mean_curvature_mean": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "curvature",
    },
    "mean_curvature_std": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "curvature",
    },
    "abs_curvature_mean": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "curvature",
    },
    "convexity": {
        "units": "ratio", "bounded": True, "role": "qc",
        "frame": "intrinsic", "redundancy_group": "concavity",
    },
    "concavity_count": {
        "units": "count", "bounded": False, "role": "qc",
        "frame": "intrinsic", "redundancy_group": "concavity",
    },
    "max_concavity_depth": {
        "units": "voxel", "bounded": False, "role": "qc",
        "frame": "intrinsic", "redundancy_group": "concavity",
    },
    "bilateral_symmetry_score": {
        "units": "score", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "symmetry",
    },
    # ----------------------------------------------------------------- #
    # Wave B — Surface curvature REMAINDER.
    # Gaussian curvature (angle-deficit) + bending energy (∫H²) + mean-
    # curvature skew + neck count. All from the SAME marching-cubes mesh
    # compute_curvature_features already builds, scale-normalized like the M5
    # curvature columns -> object-only, rotation-invariant (frame="intrinsic").
    # The curvature stats / bending energy carry an (inverse-length)ⁿ unit that
    # is divided out (dimensionless, unbounded); neck_count is an unbounded
    # count. Shape biology (smooth vs lobed/pinched); neck_count doubles as a
    # merge/pinch QC cue.
    # ----------------------------------------------------------------- #
    "gaussian_curvature_mean": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "curvature",
    },
    "gaussian_curvature_std": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "curvature",
    },
    "bending_energy": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "curvature",
    },
    "mean_curvature_skew": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "curvature",
    },
    "neck_count": {
        "units": "count", "bounded": False, "role": "qc",
        "frame": "intrinsic", "redundancy_group": "curvature",
    },
    # Wave I — curvature REMAINDER.
    # curvature_negative_fraction is a [0,1] concave-coverage fraction (bounded);
    # gaussian_curvature_skew is a dimensionless distribution skew (unbounded).
    "curvature_negative_fraction": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "curvature",
    },
    "gaussian_curvature_skew": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "curvature",
    },
    # Wave N — symmetry REMAINDER. radial_symmetry_score = 1/(1+cv)
    # of the per-azimuthal-sector voxel counts about the major axis (body-of-
    # revolution score); apical_basal_symmetry_score = IoU of the mask with its
    # reflection across the plane ⊥ the MAJOR axis (the two ENDS). Both [0,1]
    # (bounded). frame="intrinsic". Read the major_axis_* columns; NaN if absent /
    # degenerate. role="biology" (shape symmetry).
    "radial_symmetry_score": {
        "units": "score", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "symmetry",
    },
    "apical_basal_symmetry_score": {
        "units": "score", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "symmetry",
    },
    # ----------------------------------------------------------------- #
    # Wave B — Reliability / confidence META-features. NOT
    # morphology: per-cell TRUST scores derived from existing columns
    # (compute_reliability_meta), so the EDA layer can down-weight / filter
    # unreliable rows. frame="intrinsic" (a table derivation, no new spatial
    # frame); role="qc" (the whole point is reliability, not biology). All four
    # are dimensionless trust scores in [0,1] -> bounded.
    # ----------------------------------------------------------------- #
    "tilt_reliability": {
        "units": "score", "bounded": True, "role": "qc",
        "frame": "intrinsic", "redundancy_group": "reliability",
    },
    "cross_section_reliability": {
        "units": "score", "bounded": True, "role": "qc",
        "frame": "intrinsic", "redundancy_group": "reliability",
    },
    "border_reliability": {
        "units": "score", "bounded": True, "role": "qc",
        "frame": "intrinsic", "redundancy_group": "reliability",
    },
    "feature_confidence_score": {
        "units": "score", "bounded": True, "role": "qc",
        "frame": "intrinsic", "redundancy_group": "reliability",
    },
    # Mop-up O remainder — neighbour-topology reliability (z_span × contact
    # support) + probability-support score (mean of prob_mean / prob_boundary_mean /
    # low-entropy crispness). Both [0,1] trust scores -> bounded; role="qc".
    "neighbor_topology_reliability": {
        "units": "score", "bounded": True, "role": "qc",
        "frame": "intrinsic", "redundancy_group": "reliability",
    },
    "probability_support_score": {
        "units": "score", "bounded": True, "role": "qc",
        "frame": "intrinsic", "redundancy_group": "reliability",
    },
    # ----------------------------------------------------------------- #
    # M2 — Boundary/interior intensity (B1). NOTE: the UN-PREFIXED B1 columns
    # (boundary_intensity_mean / interior_intensity_mean / boundary_interior_ratio /
    # z_intensity_variation) are produced by compute_boundary_interior_features,
    # which is NEVER wired into the per-cell flow — the B1 signal arrives via the
    # PREFIXED compute_channel_intensity (e.g. mem_boundary_interior_ratio), tagged
    # by channel_intensity_tags(<prefix>). The un-prefixed manifest entries were
    # therefore dead (never matched a produced column) and have been removed; if a
    # caller ever wires the un-prefixed B1, add them back (or just rely on the
    # prefixed tags). See feature_stage._FEATURE_FAMILIES for the matching note.
    # ----------------------------------------------------------------- #
    # M2 — Probability family (B3). Per-tile Cellpose confidence (LOGIT ->
    # sigmoid -> [0,1]); prob_mean/p10/std/boundary/interior are bounded [0,1];
    # entropy is bounded [0, ln2]; prob_boundary_drop is a SIGNED difference
    # (prob_interior_mean - prob_boundary_mean), bounded [-1, 1] (NOT a [0,1]
    # score — it goes negative when the boundary is more confident than the
    # interior, the well-segmented case); soft_hard_ratio / weighted_volume /
    # gradients are unbounded. frame="image"; role="qc" (a reliability signal).
    # NOTE the engine prob is a per-tile foreground confidence, NOT a multi-view
    # consensus.
    # ----------------------------------------------------------------- #
    "prob_mean": {
        "units": "ratio", "bounded": True, "role": "qc",
        "frame": "image", "redundancy_group": "probability",
    },
    "prob_p10": {
        "units": "ratio", "bounded": True, "role": "qc",
        "frame": "image", "redundancy_group": "probability",
    },
    "prob_std": {
        "units": "ratio", "bounded": True, "role": "qc",
        "frame": "image", "redundancy_group": "probability",
    },
    "prob_boundary_mean": {
        "units": "ratio", "bounded": True, "role": "qc",
        "frame": "image", "redundancy_group": "probability",
    },
    "prob_interior_mean": {
        "units": "ratio", "bounded": True, "role": "qc",
        "frame": "image", "redundancy_group": "probability",
    },
    # prob_boundary_drop = prob_interior_mean - prob_boundary_mean: a SIGNED
    # difference bounded [-1, 1] (NOT a [0,1] score). Bounded=True (it IS bounded),
    # but downstream code must not assume non-negativity.
    "prob_boundary_drop": {
        "units": "score", "bounded": True, "role": "qc",
        "frame": "image", "redundancy_group": "probability",
    },
    "prob_weighted_volume": {
        "units": "count", "bounded": False, "role": "qc",
        "frame": "image", "redundancy_group": "probability",
    },
    "prob_soft_hard_ratio": {
        "units": "ratio", "bounded": False, "role": "qc",
        "frame": "image", "redundancy_group": "probability",
    },
    "prob_boundary_entropy": {
        "units": "score", "bounded": True, "role": "qc",
        "frame": "image", "redundancy_group": "probability",
    },
    "prob_mask_entropy": {
        "units": "score", "bounded": True, "role": "qc",
        "frame": "image", "redundancy_group": "probability",
    },
    "prob_edge_gradient": {
        "units": "ratio", "bounded": False, "role": "qc",
        "frame": "image", "redundancy_group": "probability",
    },
    "prob_edge_gradient_std": {
        "units": "ratio", "bounded": False, "role": "qc",
        "frame": "image", "redundancy_group": "probability",
    },
    # ----------------------------------------------------------------- #
    # Wave F — probability-GEOMETRY interaction. A SINGLE-
    # volume pass (qc_features.compute_prob_geometry) combining the M2 prob crop
    # with the M5 interior-EDT radial shells: how much of the mask is confident +
    # the inner/outer-shell prob means/gradient + the boundary-shell uncertainty.
    # Present ONLY when prob is sampled (rides the M2 prob block). frame="image"
    # (the prob crop is image-frame); role="qc" (a confidence-vs-geometry signal).
    # prob_area_normalized is an unbounded >=0 ratio; the two shell means are
    # [0,1]; prob_shell_gradient is a SIGNED [-1,1] difference (NOT a [0,1] score —
    # negative is the well-segmented low-confidence-rind case, like
    # prob_boundary_drop); prob_surface_uncertainty is binary entropy in [0, ln2].
    # ----------------------------------------------------------------- #
    "prob_area_normalized": {
        "units": "ratio", "bounded": False, "role": "qc",
        "frame": "image", "redundancy_group": "probability_geometry",
    },
    "prob_shell_inner_mean": {
        "units": "ratio", "bounded": True, "role": "qc",
        "frame": "image", "redundancy_group": "probability_geometry",
    },
    "prob_shell_outer_mean": {
        "units": "ratio", "bounded": True, "role": "qc",
        "frame": "image", "redundancy_group": "probability_geometry",
    },
    "prob_shell_gradient": {
        "units": "score", "bounded": True, "role": "qc",
        "frame": "image", "redundancy_group": "probability_geometry",
    },
    "prob_surface_uncertainty": {
        "units": "score", "bounded": True, "role": "qc",
        "frame": "image", "redundancy_group": "probability_geometry",
    },
    # Mop-up O remainder — surface-area-normalized soft volume (unbounded
    # >=0) + the boundary-drop-per-area (signed, unbounded — like prob_boundary_
    # drop, can be negative). role="qc"; frame="image" like the family.
    "prob_surface_normalized": {
        "units": "ratio", "bounded": False, "role": "qc",
        "frame": "image", "redundancy_group": "probability_geometry",
    },
    "prob_boundary_drop_per_area": {
        "units": "score", "bounded": False, "role": "qc",
        "frame": "image", "redundancy_group": "probability_geometry",
    },
    # P1a SIGNED remainder — apical-vs-basal confidence difference, signed by
    # the Wave C outward normal. A normalized mean DIFFERENCE in [-1,1] (bounded,
    # SIGNED — not a [0,1] score): +ve = more confident apically. frame="tissue" (it
    # is signed by the tissue outward normal, unlike the unsigned image-frame prob
    # geometry above); role="qc" (a confidence-vs-orientation signal). NaN unless a
    # trusted outward normal + centroid are supplied.
    "prob_apical_basal_gradient": {
        "units": "score", "bounded": True, "role": "qc",
        "frame": "tissue", "redundancy_group": "probability_geometry", "sign_trusted": False,
    },
    # ----------------------------------------------------------------- #
    # M3 — Neighbour / adjacency graph (B4). A GLOBAL
    # graduated-dilation contact graph (voxel-radius gap-aware, NOT a
    # centroid-spacing crowding — distinct from compute_centroid_spacing's
    # nn_centroid_distance). frame="image" (uses the global label volume +
    # image Z for the birth radii / contact); role mixes biology (neighbour
    # count = packing) and qc (a too-few/too-many neighbour count flags
    # fragments/merges). Contact fractions are [0,1]; radii/counts unbounded.
    # ----------------------------------------------------------------- #
    "neighbor_count": {
        "units": "count", "bounded": False, "role": "biology",
        "frame": "image", "redundancy_group": "adjacency",
    },
    "nearest_neighbor_distance": {
        "units": "voxel", "bounded": False, "role": "qc",
        "frame": "image", "redundancy_group": "adjacency",
    },
    "max_contact_fraction": {
        "units": "ratio", "bounded": True, "role": "qc",
        "frame": "image", "redundancy_group": "adjacency",
    },
    "background_fraction": {
        "units": "ratio", "bounded": True, "role": "qc",
        "frame": "image", "redundancy_group": "adjacency",
    },
    "nn_count_r1": {
        "units": "count", "bounded": False, "role": "biology",
        "frame": "image", "redundancy_group": "adjacency_filtration",
    },
    "nn_count_r3": {
        "units": "count", "bounded": False, "role": "biology",
        "frame": "image", "redundancy_group": "adjacency_filtration",
    },
    "nn_birth_median": {
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "image", "redundancy_group": "adjacency_filtration",
    },
    "nn_birth_std": {
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "image", "redundancy_group": "adjacency_filtration",
    },
    "nn_accumulation_rate": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "image", "redundancy_group": "adjacency_filtration",
    },
    "nn_accumulation_knee": {
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "image", "redundancy_group": "adjacency_filtration",
    },
    "mean_neighbor_size_ratio": {
        "units": "ratio", "bounded": False, "role": "qc",
        "frame": "image", "redundancy_group": "adjacency_quality",
    },
    "neighbor_size_cv": {
        "units": "ratio", "bounded": False, "role": "qc",
        "frame": "image", "redundancy_group": "adjacency_quality",
    },
    "contact_weighted_neighbor_size_ratio": {
        "units": "ratio", "bounded": False, "role": "qc",
        "frame": "image", "redundancy_group": "adjacency_quality",
    },
    # ----------------------------------------------------------------- #
    # M3 — Z-stratified neighbour topology (B5 epithelial
    # polarity). basal/apical neighbour sets across the cell's Z-range -> a
    # turnover/persistence signal for whether neighbours change basal->apical
    # (a tissue-architecture cue). frame="image" (uses global image Z to
    # stratify); biology. Counts unbounded; turnover/persistence are [0,1].
    # ----------------------------------------------------------------- #
    "basal_neighbor_count": {
        "units": "count", "bounded": False, "role": "biology",
        "frame": "image", "redundancy_group": "adjacency_topology",
    },
    "apical_neighbor_count": {
        "units": "count", "bounded": False, "role": "biology",
        "frame": "image", "redundancy_group": "adjacency_topology",
    },
    "neighbor_z_turnover": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "image", "redundancy_group": "adjacency_topology",
    },
    "neighbor_persistence_ratio": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "image", "redundancy_group": "adjacency_topology",
    },
    # ----------------------------------------------------------------- #
    # Crowding — centroid-spacing (cKDTree) features. ALWAYS produced (the
    # FeatureStage runs compute_centroid_spacing once on the assembled table, and
    # extract_label_features runs it too), unlike the OPT-IN adjacency/contact
    # families. DISTINCT from the M3 graduated-dilation adjacency (this is
    # centroid-to-centroid geometry, NOT surface gaps — nn_centroid_distance vs
    # nearest_neighbor_distance), so they share NO redundancy group with it.
    # frame="image" (uses the global centroid cloud); role="biology" (packing /
    # local density is a tissue-architecture phenotype, doubles as a soft QC). The
    # distances inherit the centroid unit (µm when spacing was used, else voxel);
    # the count is unbounded; local_density is count-per-volume (a ratio).
    # ----------------------------------------------------------------- #
    "nn_centroid_distance": {
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "image", "redundancy_group": "crowding",
    },
    "knn_mean_distance": {
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "image", "redundancy_group": "crowding",
    },
    "neighbor_count_in_radius": {
        "units": "count", "bounded": False, "role": "biology",
        "frame": "image", "redundancy_group": "crowding",
    },
    "local_density": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "image", "redundancy_group": "crowding",
    },
    # ----------------------------------------------------------------- #
    # M3 — Named composites (B6 scoring layer). Directional
    # robust-z aggregates over existing columns -> actionable QC categories.
    # frame="intrinsic" (derived from the table, no new spatial frame);
    # role="qc". merge/fragment/shape_abnormality are >=0 one-sided scores;
    # boundary_confidence is a signed z-aggregate (NaN without prob_* cols).
    # ----------------------------------------------------------------- #
    "merge_signal": {
        "units": "score", "bounded": False, "role": "qc",
        "frame": "intrinsic", "redundancy_group": "named_composites",
    },
    "fragment_signal": {
        "units": "score", "bounded": False, "role": "qc",
        "frame": "intrinsic", "redundancy_group": "named_composites",
    },
    "boundary_confidence": {
        "units": "score", "bounded": False, "role": "qc",
        "frame": "intrinsic", "redundancy_group": "named_composites",
    },
    "shape_abnormality": {
        "units": "score", "bounded": False, "role": "qc",
        "frame": "intrinsic", "redundancy_group": "named_composites",
    },
    # ----------------------------------------------------------------- #
    # M4 — Contact-patch geometry (C3; primitive P3).
    # The GEOMETRY of each cell's contact patches (vs M3's contact counts):
    # recovered from the radius-1 contact shell, split into connected patches.
    # frame="image" (uses the global label volume + image-frame contact voxels);
    # role mixes biology (packing footprint) + qc (a dominant single contact /
    # multi-patch wrapping flags merges). Areas are voxel COUNTS (unbounded);
    # the fraction / multi-patch / balance summaries are [0,1] (bounded).
    # ----------------------------------------------------------------- #
    "contact_area_total": {
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "image", "redundancy_group": "contact_geometry",
    },
    "contact_area_max": {
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "image", "redundancy_group": "contact_geometry",
    },
    "largest_contact_fraction": {
        "units": "ratio", "bounded": True, "role": "qc",
        "frame": "image", "redundancy_group": "contact_geometry",
    },
    "contact_count": {
        "units": "count", "bounded": False, "role": "biology",
        "frame": "image", "redundancy_group": "contact_geometry",
    },
    "multi_patch_neighbor_fraction": {
        "units": "ratio", "bounded": True, "role": "qc",
        "frame": "image", "redundancy_group": "contact_geometry",
    },
    "contact_balance_index": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "image", "redundancy_group": "contact_geometry",
    },
    # Mop-up M — contact-patch-geometry REMAINDER (per-edge area mean/std,
    # per-patch eccentricity/compactness, per-edge centroid offset + side asymmetry,
    # per-neighbour patch-count mean). Same frame/group. area_mean/std are voxel
    # counts (unbounded); the eccentricity/compactness + side asymmetry are [0,1]
    # (bounded); the normalised centroid offsets + patch-count mean are unbounded >=0.
    "contact_area_mean": {
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "image", "redundancy_group": "contact_geometry",
    },
    "contact_area_std": {
        "units": "voxel", "bounded": False, "role": "qc",
        "frame": "image", "redundancy_group": "contact_geometry",
    },
    "contact_patch_eccentricity_mean": {
        "units": "ratio", "bounded": True, "role": "qc",
        "frame": "image", "redundancy_group": "contact_geometry",
    },
    "contact_patch_eccentricity_max": {
        "units": "ratio", "bounded": True, "role": "qc",
        "frame": "image", "redundancy_group": "contact_geometry",
    },
    "contact_patch_compactness_mean": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "image", "redundancy_group": "contact_geometry",
    },
    "contact_centroid_offset_mean": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "image", "redundancy_group": "contact_geometry",
    },
    "contact_centroid_offset_max": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "image", "redundancy_group": "contact_geometry",
    },
    "contact_side_asymmetry": {
        "units": "ratio", "bounded": True, "role": "qc",
        "frame": "image", "redundancy_group": "contact_geometry",
    },
    "contact_patch_count_mean": {
        "units": "count", "bounded": False, "role": "qc",
        "frame": "image", "redundancy_group": "contact_geometry",
    },
    # ----------------------------------------------------------------- #
    # M4 — Local tissue frame + orientation coherence (C4;
    # primitive P6). UNSIGNED. The
    # local tissue NORMAL is the least-spread direction of the neighbour-
    # centroid offsets; the columns measure how the cell + its neighbourhood
    # sit relative to that frame. frame="tissue" (the defining new frame of
    # this milestone); biology (tissue architecture / packing coherence). All
    # are |cos| / eigenvalue-share summaries in [0,1] -> bounded. NaN for
    # < 3 neighbours (plane fit undetermined).
    # ----------------------------------------------------------------- #
    "neighbor_ring_planarity": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "tissue", "redundancy_group": "tissue_frame",
    },
    "local_normal_alignment": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "tissue", "redundancy_group": "tissue_frame",
    },
    "neighbor_axis_coherence": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "tissue", "redundancy_group": "tissue_frame",
    },
    "neighbor_ring_anisotropy": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "tissue", "redundancy_group": "tissue_frame",
    },
    # Mop-up O orientation-coherence remainder. All UNSIGNED local-frame
    # coherence descriptors -> role="biology", frame="tissue", group "tissue_frame".
    # The coherence / packing / anisotropy / stability scores + the azimuth entropy
    # are [0,1] (bounded); the axis dispersion (a variance) + the polar dispersion
    # (a std, radians) are unbounded >=0; the major-vs-normal angle is in [0,90]
    # degrees (a bounded angle).
    "contact_weighted_axis_coherence": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "tissue", "redundancy_group": "tissue_frame",
    },
    "neighbor_axis_dispersion": {
        "units": "score", "bounded": False, "role": "biology",
        "frame": "tissue", "redundancy_group": "tissue_frame",
    },
    "major_axis_vs_local_normal_angle": {
        "units": "degrees", "bounded": True, "role": "biology",
        "frame": "tissue", "redundancy_group": "tissue_frame",
    },
    "neighbor_polar_dispersion": {
        "units": "radians", "bounded": False, "role": "biology",
        "frame": "tissue", "redundancy_group": "tissue_frame",
    },
    "neighbor_azimuth_entropy": {
        "units": "score", "bounded": True, "role": "biology",
        "frame": "tissue", "redundancy_group": "tissue_frame",
    },
    "local_packing_order": {
        "units": "score", "bounded": True, "role": "biology",
        "frame": "tissue", "redundancy_group": "tissue_frame",
    },
    "coordination_anisotropy": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "tissue", "redundancy_group": "tissue_frame",
    },
    "frame_stability": {
        "units": "score", "bounded": True, "role": "qc",
        "frame": "tissue", "redundancy_group": "tissue_frame",
    },
    # P1a SIGNED local-frame remainder — the cell's principal-axis frame ORIENTED
    # against the Wave C outward normal (apical = +outward). role="biology" (apical/
    # basal architecture / polarity), frame="tissue" (the signed tissue frame).
    #   apical_basal_axis_alignment = |cos(u_major, n)| in [0,1] (columnarity MAGNITUDE
    #     — the major-axis eigenvector sign is arbitrary, so signed cos would be
    #     orientation-arbitrary; the polarity lives in cell_tilt_polarity). bounded.
    #   minor_axis_in_tangent_plane_fraction = 1 - (u_minor·n)^2 in [0,1] (squared ->
    #     sign-invariant). bounded.
    #   tangent_plane_anisotropy = in-plane (λ1-λ2)/(λ1+λ2) in [0,1]. bounded.
    #   cell_tilt_polarity = skew of (voxel-centroid)·n (SIGNED, no eigenvector sign).
    #     dimensionless, unbounded.
    #   apical_vs_basal_elongation = (σ_ap-σ_ba)/(σ_ap+σ_ba) in [-1,1] (SIGNED ratio).
    #     bounded.
    #   tangential_asymmetry_magnitude = rotation-invariant surface in-plane first-
    #     moment / in-plane radius (>=0, ~[0,1) — NO canonical L/R axis, so a MAGNITUDE
    #     not a signed L/R). unbounded >=0.
    "apical_basal_axis_alignment": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "tissue", "redundancy_group": "tissue_frame",
    },
    "minor_axis_in_tangent_plane_fraction": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "tissue", "redundancy_group": "tissue_frame",
    },
    "tangent_plane_anisotropy": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "tissue", "redundancy_group": "tissue_frame",
    },
    "cell_tilt_polarity": {
        "units": "score", "bounded": False, "role": "biology",
        "frame": "tissue", "redundancy_group": "tissue_frame", "sign_trusted": False,
    },
    "apical_vs_basal_elongation": {
        "units": "score", "bounded": True, "role": "biology",
        "frame": "tissue", "redundancy_group": "tissue_frame", "sign_trusted": False,
    },
    "tangential_asymmetry_magnitude": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "tissue", "redundancy_group": "tissue_frame",
    },
    # ----------------------------------------------------------------- #
    # Wave A — Neighbourhood graph topology (C5; primitive
    # P2). Ego-subgraph + second-ring statistics from the M3 adjacency graph
    # (df.attrs["_adjacency_list"]). frame="image" (uses the global label
    # adjacency, the same frame as the M3 graph it derives from); role mixes
    # biology (local packing topology) + qc (clique/star structure flags folds /
    # gaps). ego_density / ego_clustering_coefficient are [0,1] (bounded);
    # second_ring_count / mean_neighbor_degree are unbounded counts/degrees.
    # ----------------------------------------------------------------- #
    "ego_density": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "image", "redundancy_group": "graph_topology",
    },
    "ego_clustering_coefficient": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "image", "redundancy_group": "graph_topology",
    },
    "second_ring_count": {
        "units": "count", "bounded": False, "role": "biology",
        "frame": "image", "redundancy_group": "graph_topology",
    },
    "mean_neighbor_degree": {
        "units": "count", "bounded": False, "role": "biology",
        "frame": "image", "redundancy_group": "graph_topology",
    },
    # Mop-up M — graph-topology REMAINDER (neighbour-degree spread + ego
    # component count + second-ring ratios). Same frame/group as the Wave-A four.
    # neighbor_degree_std is an unbounded >=0 spread; neighbor_degree_max /
    # ego_component_count / second_ring_degree_mean are unbounded counts/degrees;
    # the two ring RATIOS are unbounded >=0.
    "neighbor_degree_std": {
        "units": "count", "bounded": False, "role": "qc",
        "frame": "image", "redundancy_group": "graph_topology",
    },
    "neighbor_degree_max": {
        "units": "count", "bounded": False, "role": "biology",
        "frame": "image", "redundancy_group": "graph_topology",
    },
    "ego_component_count": {
        "units": "count", "bounded": False, "role": "qc",
        "frame": "image", "redundancy_group": "graph_topology",
    },
    "first_to_second_ring_ratio": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "image", "redundancy_group": "graph_topology",
    },
    "second_ring_size_ratio": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "image", "redundancy_group": "graph_topology",
    },
    "second_ring_degree_mean": {
        "units": "count", "bounded": False, "role": "biology",
        "frame": "image", "redundancy_group": "graph_topology",
    },
    # ----------------------------------------------------------------- #
    # Wave A — Axis-based neighbour persistence (C6; primitive
    # P11+P2). The cell's-OWN-major-axis analogue of the image-Z B5 turnover /
    # persistence: contact voxels projected onto the principal axis, binned into
    # basal/apical ends. frame="intrinsic" (the principal axis is rotation-
    # invariant, unlike the image-Z B5 family); biology (axial packing /
    # exchange). turnover / persistence_ratio are [0,1] (bounded); the two end
    # counts are unbounded.
    # ----------------------------------------------------------------- #
    "axis_neighbor_turnover": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "axis_neighbor_persistence",
    },
    "axis_neighbor_persistence_ratio": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "axis_neighbor_persistence",
    },
    "axis_basal_neighbor_count": {
        "units": "count", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "axis_neighbor_persistence",
    },
    "axis_apical_neighbor_count": {
        "units": "count", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "axis_neighbor_persistence",
    },
    # ----------------------------------------------------------------- #
    # Wave J — Axis-based neighbour persistence REMAINDER (C6).
    # The per-contact axis-SPAN summary stats + birth/death exchange events + the
    # disconnected-interval (re-entry) structure + the mid axial bin + adjacent-bin
    # turnover, completing the major-axis analogue of the image-Z neighbour topology.
    # frame="intrinsic" (the principal axis). The persistence mean/max + birth/death
    # medians are normalised-t fractions in [0,1] (bounded); the cv's / counts /
    # events / exchange-rate / span-count / re-entry are unbounded >=0; the bin
    # turnovers are [0,1] Jaccard (bounded). role="biology" (axial packing/exchange).
    # ----------------------------------------------------------------- #
    "axis_neighbor_persistence_mean": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "axis_neighbor_persistence",
    },
    "axis_neighbor_persistence_max": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "axis_neighbor_persistence",
    },
    "axis_neighbor_persistence_cv": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "axis_neighbor_persistence",
    },
    "axis_neighbor_birth_median": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "axis_neighbor_persistence",
    },
    "axis_neighbor_death_median": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "axis_neighbor_persistence",
    },
    "axis_neighbor_span_cv": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "axis_neighbor_persistence",
    },
    "axis_neighbor_events": {
        "units": "count", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "axis_neighbor_persistence",
    },
    "axis_neighbor_exchange_rate": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "axis_neighbor_persistence",
    },
    "axis_neighbor_multispan_fraction": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "axis_neighbor_persistence",
    },
    "axis_neighbor_span_count_mean": {
        "units": "count", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "axis_neighbor_persistence",
    },
    "axis_neighbor_reentry_count": {
        "units": "count", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "axis_neighbor_persistence",
    },
    "mid_axis_neighbor_count": {
        "units": "count", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "axis_neighbor_persistence",
    },
    "basal_to_mid_turnover": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "axis_neighbor_persistence",
    },
    "mid_to_apical_turnover": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "axis_neighbor_persistence",
    },
    # ----------------------------------------------------------------- #
    # Wave A — Graph motif / local network (C16; primitive P2).
    # Small graph patterns + degree relationships around the cell. frame="image"
    # (global adjacency frame); role mixes biology (packing motif) + qc (degree
    # disorder flags merges/folds). ego_triangle_count is an unbounded count;
    # neighbor_count_cv is an unbounded >=0 ratio; degree_centrality is [0,1]
    # (bounded); neighbor_degree_assortativity is a signed ~[-1,1] score (NOT
    # strictly bounded -> bounded=False, downstream must not assume [0,1]).
    # ----------------------------------------------------------------- #
    "ego_triangle_count": {
        "units": "count", "bounded": False, "role": "biology",
        "frame": "image", "redundancy_group": "graph_motif",
    },
    "neighbor_count_cv": {
        "units": "ratio", "bounded": False, "role": "qc",
        "frame": "image", "redundancy_group": "graph_motif",
    },
    "degree_centrality": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "image", "redundancy_group": "graph_motif",
    },
    "neighbor_degree_assortativity": {
        "units": "score", "bounded": False, "role": "biology",
        "frame": "image", "redundancy_group": "graph_motif",
    },
    # Mop-up M — graph-motif REMAINDER. Higher-order motifs (square count is
    # an unbounded count; cycle fraction is a [0,1] ratio); node-attribute homophily
    # (1/(1+cv) in (0,1], bounded); the ego-graph CENTRALITY trio (research tier,
    # computed on the first-ring subgraph) — all normalised to [0,1] (bounded).
    "ego_square_count": {
        "units": "count", "bounded": False, "role": "biology",
        "frame": "image", "redundancy_group": "graph_motif",
    },
    "ego_cycle_fraction": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "image", "redundancy_group": "graph_motif",
    },
    "neighbor_homophily_area": {
        "units": "ratio", "bounded": True, "role": "qc",
        "frame": "image", "redundancy_group": "graph_motif",
    },
    "neighbor_homophily_shape": {
        "units": "ratio", "bounded": True, "role": "qc",
        "frame": "image", "redundancy_group": "graph_motif",
    },
    "ego_betweenness": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "image", "redundancy_group": "graph_motif",
    },
    "ego_pagerank": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "image", "redundancy_group": "graph_motif",
    },
    "ego_eigenvector_centrality": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "image", "redundancy_group": "graph_motif",
    },
    # ----------------------------------------------------------------- #
    # M6 — Paired nucleus<->cell features (the feature catalogue;
    # primitive P9). These need BOTH the cell mask AND its nucleus mask
    # together, so they're computed in the reconcile pairing path
    # (the paired cell-nucleus measurements) from the
    # aligned consensus masks, NOT the single-volume per-cell pool. Count/flag
    # columns are the strongest merge/QC signals (role="qc"); the offsets /
    # ratios are biology (nuclear polarity / N:C). frame="intrinsic" (object +
    # its paired nucleus, no external tissue frame). nuclei_per_cell is an
    # unbounded count; the two flags are {0,1}; containment / axis_alignment are
    # bounded [0,1]; the ratio / offsets / volumes / thickness are unbounded
    # (>=0); offset_norm is an unbounded >=0 ratio.
    # ----------------------------------------------------------------- #
    "nuclei_per_cell": {
        "units": "count", "bounded": False, "role": "qc",
        "frame": "intrinsic", "redundancy_group": "nucleus_pairing",
    },
    "missing_nucleus_flag": {
        "units": "ratio", "bounded": True, "role": "qc",
        "frame": "intrinsic", "redundancy_group": "nucleus_pairing",
    },
    "multinucleate_flag": {
        "units": "ratio", "bounded": True, "role": "qc",
        "frame": "intrinsic", "redundancy_group": "nucleus_pairing",
    },
    "nucleus_containment": {
        "units": "ratio", "bounded": True, "role": "qc",
        "frame": "intrinsic", "redundancy_group": "nucleus_pairing",
    },
    "nucleus_cell_vol_ratio": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "nucleus_cell_ratio",
    },
    "cytoplasm_volume": {
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "nucleus_cell_ratio",
    },
    "nucleus_cell_centroid_offset": {
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "nucleus_position",
    },
    "nucleus_cell_offset_norm": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "nucleus_position",
    },
    "cytoplasm_thickness_mean": {
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "cytoplasm_thickness",
    },
    "cytoplasm_thickness_min": {
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "cytoplasm_thickness",
    },
    "nucleus_cell_axis_alignment": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "nucleus_orientation",
    },
    # Wave D — SIGNED apical/basal nuclear offset (the feature catalogue
    # `nuclear_offset_apical_basal`). Computed in the reconcile pairing path
    # (the paired cell-nucleus measurements) by projecting the (nucleus − cell)
    # centroid displacement onto the cell's Wave C OUTWARD normal, so it needs the
    # tissue frame -> frame="tissue". Signed (+apical / −basal); the raw offset
    # carries the spacing unit (unbounded), the _norm is the same / equiv-radius
    # (unbounded ratio, sign preserved). role="biology" (nuclear A/B polarity).
    "nucleus_apical_basal_offset": {
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "tissue", "redundancy_group": "nucleus_position", "sign_trusted": False,
    },
    "nucleus_apical_basal_offset_norm": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "tissue", "redundancy_group": "nucleus_position", "sign_trusted": False,
    },
    # ----------------------------------------------------------------- #
    # Wave B — single-nucleus SHAPE + CROSS-SECTION (the feature catalogue;
    # primitive P9). Computed in the reconcile pairing path
    # (the paired cell-nucleus measurements) by reusing the
    # engine's per-cell shape functions on the single in-cell nucleus crop —
    # gated to a SINGLE-nucleus cell (NaN otherwise). frame="intrinsic"
    # (nucleus-only shape). sphericity / solidity / invagination /
    # waist_depth / hourglass / bilobed are [0,1] (bounded); elongation +
    # min_cross_section_area are unbounded (>=0). role="biology" (nuclear
    # morphology); the hourglass/bilobed scores double as a mitotic/pinched-
    # nucleus QC cue.
    # ----------------------------------------------------------------- #
    "nuclear_sphericity": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "nucleus_shape",
    },
    "nuclear_solidity": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "nucleus_shape",
    },
    "nuclear_elongation": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "nucleus_shape",
    },
    "nuclear_invagination": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "nucleus_shape",
    },
    "nuclear_min_cross_section_area": {
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "nucleus_cross_section",
    },
    "nuclear_cross_section_waist_depth": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "nucleus_cross_section",
    },
    "nuclear_hourglass_score": {
        "units": "score", "bounded": True, "role": "qc",
        "frame": "intrinsic", "redundancy_group": "nucleus_cross_section",
    },
    "nuclear_bilobed_score": {
        "units": "score", "bounded": True, "role": "qc",
        "frame": "intrinsic", "redundancy_group": "nucleus_cross_section",
    },
    # ----------------------------------------------------------------- #
    # Wave I — CATALOGUE NUCLEUS NAMING RECONCILE. The canonical
    # ``nuclear_*`` names emitted as ALIASES of the re-prefixed ``nuc_*`` nucleus
    # morphology columns (the paired cell-nucleus measurements),
    # so the catalogue names are recognized. NO new data — each points at its ``nuc_*``
    # source (single-nucleus rows populated, others NaN). frame="intrinsic"
    # (nucleus-only shape). volume/surface_area/axis lengths carry the spacing unit
    # (unbounded >=0); extent + aspect_ratio + roughness are dimensionless (unbounded).
    # role="biology". NOTE: nuclear_sphericity/solidity/elongation/invagination are
    # NOT aliased here (already emitted by the nuclear-shape measurements, above).
    # ----------------------------------------------------------------- #
    # FIX 10 — units="voxel" matches the CELL convention (``area`` / ``surface_area``
    # / ``nuc_area`` / ``nuc_surface_area`` are all "voxel" = the engine's single
    # spacing-bearing token: voxel counts, or µm/µm²/µm³ when ``spacing`` was given).
    # These nucleus replicas previously tagged the SAME spacing-bearing measure as
    # "um3"/"um", inconsistently with their ``nuc_*`` sources + the cell side.
    "nuclear_volume": {
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "nucleus_shape", "size_correlated": True,
    },
    "nuclear_extent": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "nucleus_shape",
    },
    "nuclear_surface_area": {
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "nucleus_shape", "size_correlated": True,
    },
    "nuclear_roughness": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "nucleus_shape",
    },
    "nuclear_aspect_ratio": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "nucleus_shape",
    },
    "nuclear_axis_major_length": {
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "nucleus_shape", "size_correlated": True,
    },
    "nuclear_axis_minor_length": {
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "nucleus_shape", "size_correlated": True,
    },
    # ----------------------------------------------------------------- #
    # Wave L — the NUCLEUS DEEP TAIL (the feature catalogue
    # nucleus replicas; primitive P9). Two groups computed in the reconcile
    # pairing path:
    #   (a) single-nucleus INTRINSIC shape / cross-section / 3-axis REMAINDER
    #       (the paired cell-nucleus measurements) — the full
    #       nuclear morphology / profile / triaxial tail the catalogue lists beyond
    #       the Wave-B curated subset, reused from the engine's per-cell shape
    #       functions on the in-cell nucleus crop. frame="intrinsic" (nucleus-only
    #       shape); single-nucleus-gated (NaN otherwise).
    #   (b) PAIRED position / orientation / N:C-ratio / cytoplasm / dual-QC tail
    #       (the paired cell-nucleus measurements) — needs
    #       BOTH masks (+ optionally the Wave C outward normal for the apical/basal
    #       distances). Position/orientation/cross-axis-ratio columns are single-
    #       nucleus-gated; the ratio / cytoplasm / dual-QC flags + within-cell
    #       multinuclear stats are AGGREGATES over the cell's nuclei (defined for a
    #       multinucleate merge). Dual-QC flags / mismatch / confidence are
    #       role="qc"; the geometry / ratios are role="biology". The apical/basal
    #       distances need the tissue outward normal -> frame="tissue".
    # ----------------------------------------------------------------- #
    # (a) intrinsic shape remainder (single-nucleus). nuclear_euler_number is role=qc
    # (FIX 8): a nuclear Euler number != 1 is a SEGMENTATION ARTEFACT (holes / split
    # fragments), not phenotype — matching the cell ``euler_number`` (also role=qc).
    "nuclear_euler_number": {
        "units": "count", "bounded": False, "role": "qc",
        "frame": "intrinsic", "redundancy_group": "nucleus_shape",
    },
    "nuclear_equivalent_diameter": {
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "nucleus_shape", "size_correlated": True,
    },
    "nuclear_z_elongation": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "nucleus_shape",
    },
    "nuclear_major_axis_z_angle": {
        "units": "degrees", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "nucleus_orientation",
    },
    "nuclear_eccentricity": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "nucleus_shape",
    },
    "nuclear_axis_mid_length": {
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "nucleus_shape", "size_correlated": True,
    },
    "nuclear_triaxiality": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "nucleus_shape",
    },
    "nuclear_min_cross_section_axis_relative": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "nucleus_cross_section",
    },
    "nuclear_cross_section_range": {
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "nucleus_cross_section",
    },
    "nuclear_cross_section_cv": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "nucleus_cross_section",
    },
    "nuclear_cross_section_taper": {
        "units": "score", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "nucleus_cross_section",
    },
    "nuclear_cross_section_bulge": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "nucleus_cross_section",
    },
    "nuclear_cross_section_peak_count": {
        "units": "count", "bounded": False, "role": "qc",
        "frame": "intrinsic", "redundancy_group": "nucleus_cross_section",
    },
    "nuclear_cross_section_trough_prominence": {
        "units": "ratio", "bounded": True, "role": "qc",
        "frame": "intrinsic", "redundancy_group": "nucleus_cross_section",
    },
    "nuclear_waist_axis_relative": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "nucleus_cross_section",
    },
    "nuclear_vase_like_score": {
        "units": "score", "bounded": True, "role": "qc",
        "frame": "intrinsic", "redundancy_group": "nucleus_cross_section",
    },
    # (b) paired position (single-nucleus)
    "nucleus_to_cell_centroid_xy_distance": {
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "nucleus_position",
    },
    "nucleus_to_cell_centroid_axis_distance": {
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "nucleus_position",
    },
    "nucleus_position_radial": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "nucleus_position",
    },
    "nucleus_centrality": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "nucleus_position",
    },
    "nucleus_to_lateral_boundary_distance": {
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "nucleus_position",
    },
    "nucleus_to_apical_distance": {
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "tissue", "redundancy_group": "nucleus_position", "sign_trusted": False,
    },
    "nucleus_to_basal_distance": {
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "tissue", "redundancy_group": "nucleus_position", "sign_trusted": False,
    },
    # (b) paired orientation (single-nucleus)
    "nuclear_tilt_relative_to_cell": {
        "units": "degrees", "bounded": True, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "nucleus_orientation",
    },
    # (b) within-cell multinuclear stats (aggregate)
    "nuclear_size_cv_within_cell": {
        "units": "ratio", "bounded": False, "role": "qc",
        "frame": "intrinsic", "redundancy_group": "nucleus_pairing",
    },
    "largest_nucleus_fraction": {
        "units": "ratio", "bounded": True, "role": "qc",
        "frame": "intrinsic", "redundancy_group": "nucleus_pairing",
    },
    # (b) ratio / cytoplasm (aggregate)
    "nuclear_to_cytoplasm_volume_ratio": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "nucleus_cell_ratio",
    },
    "nuclear_to_cell_surface_ratio": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "nucleus_cell_ratio",
    },
    "cytoplasm_surface_area": {
        "units": "um", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "cytoplasm_thickness",
    },
    "cytoplasm_thickness_cv": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "cytoplasm_thickness",
    },
    "nucleus_outside_cell_fraction": {
        "units": "ratio", "bounded": True, "role": "qc",
        "frame": "intrinsic", "redundancy_group": "nucleus_pairing",
    },
    # (b) dual-QC flags + composite mismatch / confidence
    "nucleus_larger_than_cell_flag": {
        "units": "ratio", "bounded": True, "role": "qc",
        "frame": "intrinsic", "redundancy_group": "nucleus_pairing",
    },
    "cytoplasm_without_nucleus_flag": {
        "units": "ratio", "bounded": True, "role": "qc",
        "frame": "intrinsic", "redundancy_group": "nucleus_pairing",
    },
    "nucleus_cell_size_mismatch_score": {
        "units": "score", "bounded": True, "role": "qc",
        "frame": "intrinsic", "redundancy_group": "nucleus_pairing",
    },
    "nucleus_cell_shape_mismatch_score": {
        "units": "score", "bounded": True, "role": "qc",
        "frame": "intrinsic", "redundancy_group": "nucleus_pairing",
    },
    "nucleus_cell_pair_confidence": {
        "units": "score", "bounded": True, "role": "qc",
        "frame": "intrinsic", "redundancy_group": "nucleus_pairing",
    },
    # (b) nucleus-to-cell cross-axis ratios (single-nucleus)
    "nucleus_to_cell_major_ratio": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "nucleus_cell_ratio",
    },
    "nucleus_to_cell_mid_ratio": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "nucleus_cell_ratio",
    },
    "nucleus_to_cell_minor_ratio": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "intrinsic", "redundancy_group": "nucleus_cell_ratio",
    },
    # ----------------------------------------------------------------- #
    # Wave C — tissue-depth "in/out" field (the feature catalogue; the C-W
    # tissue-depth primitive). A GLOBAL pass (extract_label_features(
    # tissue_depth=True) -> qc_features.compute_tissue_depth): a single 3D EDT
    # of the WHOLE-TISSUE mask gives each cell its distance-from-surface depth.
    # frame="tissue" (a whole-tissue coordinate, the defining new frame of this
    # wave); role mixes biology (depth is an apical/basal-position phenotype in
    # curved tissue) + qc (at_fov_border flags depth-unreliable cells). The depth
    # distances carry the spacing unit (unbounded >=0); normalized_tissue_depth is
    # [0,1] (bounded, self-scaled by the global max depth); at_fov_border is {0,1}.
    # GEOMETRY CAVEAT (compute_tissue_depth): a clean apical/basal proxy only for
    # CURVED/CLOSED tissue.
    # ----------------------------------------------------------------- #
    "distance_to_tissue_surface": {
        "units": "voxel", "bounded": False, "role": "covariate",
        "frame": "tissue", "redundancy_group": "tissue_depth",
    },
    "tissue_depth_mean": {
        "units": "voxel", "bounded": False, "role": "covariate",
        "frame": "tissue", "redundancy_group": "tissue_depth",
    },
    "tissue_depth_max": {
        "units": "voxel", "bounded": False, "role": "covariate",
        "frame": "tissue", "redundancy_group": "tissue_depth",
    },
    "normalized_tissue_depth": {
        "units": "ratio", "bounded": True, "role": "covariate",
        "frame": "tissue", "redundancy_group": "tissue_depth",
    },
    "at_fov_border": {
        "units": "ratio", "bounded": True, "role": "qc",
        "frame": "tissue", "redundancy_group": "tissue_depth",
    },
    # ----------------------------------------------------------------- #
    # Wave C — spatial tissue context. Cheap volume-relative position
    # covariates (NOT pure QC features — useful covariates for morphospace
    # interpretation note). frame="tissue" (position within the
    # tissue bbox / relative to the tissue centroid); role="biology" (a zone /
    # position covariate). The three relative positions + radial_tissue_position
    # are [0,1] (bounded); distance_to_tissue_centroid carries the spacing unit
    # (unbounded >=0).
    # ----------------------------------------------------------------- #
    "centroid_z_relative": {
        "units": "ratio", "bounded": True, "role": "covariate",
        "frame": "tissue", "redundancy_group": "spatial_context",
    },
    "centroid_y_relative": {
        "units": "ratio", "bounded": True, "role": "covariate",
        "frame": "tissue", "redundancy_group": "spatial_context",
    },
    "centroid_x_relative": {
        "units": "ratio", "bounded": True, "role": "covariate",
        "frame": "tissue", "redundancy_group": "spatial_context",
    },
    "distance_to_tissue_centroid": {
        "units": "voxel", "bounded": False, "role": "covariate",
        "frame": "tissue", "redundancy_group": "spatial_context",
    },
    "radial_tissue_position": {
        "units": "ratio", "bounded": True, "role": "covariate",
        "frame": "tissue", "redundancy_group": "spatial_context",
    },
    # Wave I — remainder: min physical distance from the centroid to a FOV
    # face (truncation proximity). frame="tissue", spacing unit (unbounded >=0);
    # role="qc" (a truncation/edge-reliability covariate).
    "distance_to_boundary_face_min": {
        "units": "voxel", "bounded": False, "role": "qc",
        "frame": "tissue", "redundancy_group": "spatial_context",
    },
    # Mop-up O — remainder: the named per-cell LOCAL tissue depth (mean
    # tissue-EDT in cell-radius units). frame="tissue"; a dimensionless depth ratio
    # (unbounded >=0, NOT globally normalized -> not [0,1]); role="biology".
    "local_tissue_depth": {
        "units": "ratio", "bounded": False, "role": "covariate",
        "frame": "tissue", "redundancy_group": "spatial_context",
    },
    # ----------------------------------------------------------------- #
    # P1b — lumen-cavity (distance_to_lumen). distance_to_lumen is the
    # min physical EDT from the cell to the nearest ENCLOSED background cavity (a
    # lumen — NOT the external medium); lumen_facing_fraction is the share of the
    # cell surface lining a lumen ([0,1]). frame="tissue" (a whole-tissue cavity
    # model). role="biology" (an apical-footprint / depth covariate). distance is a
    # spacing-unit (unbounded >=0); the fraction is [0,1] (bounded). BOTH all-NaN
    # for a tissue with no enclosed lumen (the correct answer, not a failure).
    # ----------------------------------------------------------------- #
    "distance_to_lumen": {
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "tissue", "redundancy_group": "lumen",
    },
    "lumen_facing_fraction": {
        "units": "ratio", "bounded": True, "role": "covariate",
        "frame": "tissue", "redundancy_group": "lumen",
    },
    # ----------------------------------------------------------------- #
    # Wave C — per-cell OUTWARD normal (the sign basis later waves use to sign
    # the apical/basal frame). The negative normalised tissue-EDT gradient at the
    # cell centroid points OUTWARD (toward the nearest surface). frame="tissue"
    # (a whole-tissue direction); role="biology" (an orientation basis). The three
    # components are a UNIT vector in [-1,1] (bounded, but signed — downstream must
    # not assume non-negative); tissue_outward_confidence is the gradient magnitude
    # clipped to [0,1] (bounded), ~0 in the flat core where the direction is
    # ambiguous.
    # ----------------------------------------------------------------- #
    "tissue_outward_z": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "tissue", "redundancy_group": "tissue_outward_normal",
    },
    "tissue_outward_y": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "tissue", "redundancy_group": "tissue_outward_normal",
    },
    "tissue_outward_x": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "tissue", "redundancy_group": "tissue_outward_normal",
    },
    "tissue_outward_confidence": {
        "units": "score", "bounded": True, "role": "qc",
        "frame": "tissue", "redundancy_group": "tissue_outward_normal",
    },
    # ----------------------------------------------------------------- #
    # Wave D — DIRECTIONAL CONTACT. A GLOBAL pass
    # (extract_label_features(tissue_depth=True) -> qc_features.
    # compute_directional_contact) that splits a cell's contact voxels into
    # apical / basal / lateral by the Wave C outward normal. frame="tissue" (the
    # signed apical/basal frame, the defining new frame of this wave). The three
    # fractions are [0,1] (bounded, sum to 1); contact_polarity_bias is the signed
    # apical−basal difference in [-1,1] (bounded). role="biology" (a polarity /
    # tissue-fold phenotype). NaN for a cell with no contacts / untrusted normal.
    # ----------------------------------------------------------------- #
    "apical_contact_fraction": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "tissue", "redundancy_group": "directional_contact", "sign_trusted": False,
    },
    "basal_contact_fraction": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "tissue", "redundancy_group": "directional_contact", "sign_trusted": False,
    },
    "lateral_contact_fraction": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "tissue", "redundancy_group": "directional_contact",
    },
    "contact_polarity_bias": {
        "units": "score", "bounded": True, "role": "biology",
        "frame": "tissue", "redundancy_group": "directional_contact", "sign_trusted": False,
    },
    # Mop-up M — directional-contact REMAINDER (contact-direction-cloud
    # structure). Same frame/group. dominant_contact_axis / hemisphere_balance /
    # direction_entropy are [0,1] (bounded); sector_count is an unbounded count
    # (<= the sector total); dominant_contact_neighbor_fraction is a [0,1] ratio.
    "dominant_contact_axis": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "tissue", "redundancy_group": "directional_contact",
    },
    "contact_hemisphere_balance": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "tissue", "redundancy_group": "directional_contact",
    },
    "contact_direction_entropy": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "tissue", "redundancy_group": "directional_contact",
    },
    "contact_sector_count": {
        "units": "count", "bounded": False, "role": "biology",
        "frame": "tissue", "redundancy_group": "directional_contact",
    },
    "dominant_contact_neighbor_fraction": {
        "units": "ratio", "bounded": True, "role": "qc",
        "frame": "tissue", "redundancy_group": "directional_contact",
    },
    # ----------------------------------------------------------------- #
    # Wave D — APICAL/BASAL/LATERAL SURFACE PARTITION.
    # A per-cell mesh pass that reads the Wave C outward normal (so it rides the
    # tissue_depth=True global block -> qc_features.compute_surface_partition):
    # marching_cubes the cell, class each face apical/basal/lateral by its normal's
    # dot with the outward vector, sum each patch's physical area. frame="tissue".
    # The three areas carry the spacing² (area) unit (unbounded >=0); the three
    # fractions are [0,1] (bounded, sum to 1); apical_basal_surface_ratio is an
    # unbounded >=0 ratio. role="biology" (apical/basal membrane biology). NaN for a
    # degenerate mesh / untrusted normal.
    # ----------------------------------------------------------------- #
    "apical_surface_area": {
        "units": "um", "bounded": False, "role": "biology",
        "frame": "tissue", "redundancy_group": "surface_partition", "sign_trusted": False,
    },
    "basal_surface_area": {
        "units": "um", "bounded": False, "role": "biology",
        "frame": "tissue", "redundancy_group": "surface_partition", "sign_trusted": False,
    },
    "lateral_surface_area": {
        "units": "um", "bounded": False, "role": "biology",
        "frame": "tissue", "redundancy_group": "surface_partition",
    },
    "apical_surface_fraction": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "tissue", "redundancy_group": "surface_partition", "sign_trusted": False,
    },
    "basal_surface_fraction": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "tissue", "redundancy_group": "surface_partition", "sign_trusted": False,
    },
    "lateral_surface_fraction": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "tissue", "redundancy_group": "surface_partition",
    },
    "apical_basal_surface_ratio": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "tissue", "redundancy_group": "surface_partition", "sign_trusted": False,
    },
    # Wave I — cross ratios (lateral vs apical/basal cap). Pure derivations
    # of the three patch areas; unbounded >=0 ratios. frame="tissue".
    "lateral_to_apical_surface_ratio": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "tissue", "redundancy_group": "surface_partition", "sign_trusted": False,
    },
    "lateral_to_basal_surface_ratio": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "tissue", "redundancy_group": "surface_partition", "sign_trusted": False,
    },
    # ----------------------------------------------------------------- #
    # Wave J — SURFACE PARTITION REMAINDER. The apical/basal
    # FOOTPRINT contour (the cell's top-/bottom-facing outline projected ⊥ the Wave
    # C outward normal -> 2D convex hull): perimeter (um, unbounded), circularity /
    # compactness (the isoperimetric 4πA/P² ratio in [0,1], bounded), aspect_ratio
    # (covariance elongation >=1, unbounded). Plus per-partition CURVATURE (the
    # size-normalised mean curvature averaged over each patch's vertices — signed,
    # dimensionless, unbounded) + per-partition ROUGHNESS (the area-weighted RMS
    # face-normal deviation in radians; >=0, unbounded angle). frame="tissue".
    # role="biology" (apical/basal membrane shape). NaN for an empty partition /
    # degenerate mesh / untrusted normal.
    # ----------------------------------------------------------------- #
    "apical_perimeter": {
        "units": "um", "bounded": False, "role": "biology",
        "frame": "tissue", "redundancy_group": "surface_partition", "sign_trusted": False,
    },
    "apical_circularity": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "tissue", "redundancy_group": "surface_partition", "sign_trusted": False,
    },
    "apical_aspect_ratio": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "tissue", "redundancy_group": "surface_partition", "sign_trusted": False,
    },
    "apical_compactness": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "tissue", "redundancy_group": "surface_partition", "sign_trusted": False,
    },
    "basal_perimeter": {
        "units": "um", "bounded": False, "role": "biology",
        "frame": "tissue", "redundancy_group": "surface_partition", "sign_trusted": False,
    },
    "basal_circularity": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "tissue", "redundancy_group": "surface_partition", "sign_trusted": False,
    },
    "basal_aspect_ratio": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "tissue", "redundancy_group": "surface_partition", "sign_trusted": False,
    },
    "basal_compactness": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "tissue", "redundancy_group": "surface_partition", "sign_trusted": False,
    },
    "apical_curvature_mean": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "tissue", "redundancy_group": "surface_partition", "sign_trusted": False,
    },
    "apical_curvature_std": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "tissue", "redundancy_group": "surface_partition", "sign_trusted": False,
    },
    "basal_curvature_mean": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "tissue", "redundancy_group": "surface_partition", "sign_trusted": False,
    },
    "lateral_curvature_mean": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "tissue", "redundancy_group": "surface_partition",
    },
    "apical_surface_roughness": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "tissue", "redundancy_group": "surface_partition", "sign_trusted": False,
    },
    "basal_surface_roughness": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "tissue", "redundancy_group": "surface_partition", "sign_trusted": False,
    },
    # ----------------------------------------------------------------- #
    # Wave H — SIGNED apical/basal split of the major-axis profile (the feature catalogue
    # remainder; the ma_apical/basal candidates). A GLOBAL pass (extract_label_
    # features(tissue_depth=True) -> qc_features.compute_ma_apical_basal): orients the
    # M1 major-axis cross-section profile by the Wave C outward normal and reports the
    # cross-section area in the apical third vs the basal third + each third's signed
    # taper. frame="tissue" (the signed apical/basal frame). The areas are voxel-count
    # cross-sections (unbounded >=0, voxel unit); the ratio is an unbounded >=0 ratio;
    # the tapers are signed voxel-area slopes (unbounded). role="biology" (an
    # apically/basally flared wedge phenotype). NaN for a degenerate profile / untrusted
    # normal.
    # ----------------------------------------------------------------- #
    "ma_apical_area": {
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "tissue", "redundancy_group": "ma_apical_basal", "sign_trusted": False,
    },
    "ma_basal_area": {
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "tissue", "redundancy_group": "ma_apical_basal", "sign_trusted": False,
    },
    "ma_apical_basal_ratio": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "tissue", "redundancy_group": "ma_apical_basal", "sign_trusted": False,
    },
    "ma_apical_taper": {
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "tissue", "redundancy_group": "ma_apical_basal", "sign_trusted": False,
    },
    "ma_basal_taper": {
        "units": "voxel", "bounded": False, "role": "biology",
        "frame": "tissue", "redundancy_group": "ma_apical_basal", "sign_trusted": False,
    },
    # ----------------------------------------------------------------- #
    # Wave H — APICAL JUNCTIONAL / EPITHELIAL POLYGON topology. A
    # GLOBAL pass (extract_label_features(tissue_depth=True) -> qc_features.
    # compute_apical_junction): the cell's LATERAL contacts (perpendicular to the Wave
    # C outward normal — the apical-junctional belt) summarised as the classic
    # epithelial polygon. frame="tissue". apical_polygon_sides is the distinct lateral
    # neighbour count (~6 in ordered tissue; a count, unbounded >=0); apical_junction_
    # length is the total lateral contact perimeter PROXY (physical length, unbounded
    # >=0); apical_polygon_irregularity is the cv of lateral patch sizes (unbounded
    # >=0). role="biology" (epithelial topology / tissue mechanics; doubles as a merge
    # QC cue — a merged cell reads few large lateral faces). NaN for no lateral contacts
    # / untrusted normal.
    # ----------------------------------------------------------------- #
    "apical_polygon_sides": {
        "units": "count", "bounded": False, "role": "biology",
        "frame": "tissue", "redundancy_group": "apical_junction", "sign_trusted": False,
    },
    "apical_junction_length": {
        "units": "um", "bounded": False, "role": "biology",
        "frame": "tissue", "redundancy_group": "apical_junction", "sign_trusted": False,
    },
    "apical_polygon_irregularity": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "tissue", "redundancy_group": "apical_junction", "sign_trusted": False,
    },
    # ----------------------------------------------------------------- #
    # Wave J — APICAL JUNCTIONAL / POLYGON REMAINDER. The junction-belt
    # vertex/edge stats (vertex_count == lateral-neighbour count; edge_length_cv +
    # vertex_angle_cv of the belt polygon ordered by footprint azimuth) + the explicit
    # apical FOOTPRINT polygon (the cell's apical-cap voxels collapsed ⊥ outward ->
    # convex hull): area (um², unbounded) + perimeter (um, unbounded) + shape_index
    # (perimeter/√area — the vertex-model fluid/solid order parameter, unbounded >=
    # ~3.54) + convexity (planar solidity in (0,1], bounded). vertex_count is a count;
    # the two cv's are unbounded ratios. frame="tissue". role="biology" (epithelial
    # polygon topology / tissue mechanics). NaN for too-few vertices / degenerate
    # footprint / untrusted normal.
    # ----------------------------------------------------------------- #
    "apical_junction_vertex_count": {
        "units": "count", "bounded": False, "role": "biology",
        "frame": "tissue", "redundancy_group": "apical_junction", "sign_trusted": False,
    },
    "apical_junction_edge_length_cv": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "tissue", "redundancy_group": "apical_junction", "sign_trusted": False,
    },
    "apical_vertex_angle_cv": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "tissue", "redundancy_group": "apical_junction", "sign_trusted": False,
    },
    "apical_polygon_area": {
        "units": "um2", "bounded": False, "role": "biology",
        "frame": "tissue", "redundancy_group": "apical_junction", "sign_trusted": False,
    },
    "apical_polygon_perimeter": {
        "units": "um", "bounded": False, "role": "biology",
        "frame": "tissue", "redundancy_group": "apical_junction", "sign_trusted": False,
    },
    "apical_shape_index": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "tissue", "redundancy_group": "apical_junction", "sign_trusted": False,
    },
    "apical_polygon_convexity": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "tissue", "redundancy_group": "apical_junction", "sign_trusted": False,
    },
    # ----------------------------------------------------------------- #
    # Wave G — watershed territory / crowding (C7). A GLOBAL pass
    # (epicycle.reconcile.features.extract_label_features(territory=True) ->
    # epicycle.core.qc_features.compute_watershed_territory): a background-EDT
    # seeded watershed expands each cell into the inter-cell space up to the
    # midline with its neighbours (capped at a max distance), giving each cell a
    # Voronoi-ish TERRITORY. ws_territory_volume carries the spacing volume unit
    # (unbounded); territory_to_cell_ratio is an unbounded >=1 crowding ratio
    # (~1 = packed, large = sparse); territory_anisotropy is an unbounded >=1
    # elongation; territory_solidity is a [0,1] convexity. frame="tissue" (the
    # inter-cell space tessellation); role="biology" (a local crowding /
    # breathing-room covariate). NaN for a degenerate territory.
    # ----------------------------------------------------------------- #
    "ws_territory_volume": {
        "units": "um3", "bounded": False, "role": "biology",
        "frame": "tissue", "redundancy_group": "watershed_territory",
    },
    "territory_to_cell_ratio": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "tissue", "redundancy_group": "watershed_territory",
    },
    "territory_anisotropy": {
        "units": "ratio", "bounded": False, "role": "biology",
        "frame": "tissue", "redundancy_group": "watershed_territory",
    },
    "territory_solidity": {
        "units": "ratio", "bounded": True, "role": "biology",
        "frame": "tissue", "redundancy_group": "watershed_territory",
    },
    # Mop-up M — watershed-territory REMAINDER. ws_neighbor_count is an
    # unbounded count; ws_nearest_gap + territory_surface_area carry physical
    # length / area units (unbounded >=0); territory_z_bias is a signed [-1,1]
    # (bounded). Same frame/group as the Wave-G four.
    "ws_neighbor_count": {
        "units": "count", "bounded": False, "role": "biology",
        "frame": "tissue", "redundancy_group": "watershed_territory",
    },
    "ws_nearest_gap": {
        "units": "um", "bounded": False, "role": "biology",
        "frame": "tissue", "redundancy_group": "watershed_territory",
    },
    "territory_z_bias": {
        "units": "score", "bounded": True, "role": "biology",
        "frame": "tissue", "redundancy_group": "watershed_territory",
    },
    "territory_surface_area": {
        "units": "um2", "bounded": False, "role": "biology",
        "frame": "tissue", "redundancy_group": "watershed_territory",
    },
    # ----------------------------------------------------------------- #
    # M7 — composite qc_score (B7 scoring layer). A GLOBAL
    # table pass (epicycle.reconcile.features.extract_label_features(
    # qc_score=True) -> epicycle.core.qc_features.compute_composite_score):
    # robust-standardize the curated PCA_FEATURES subset ACROSS the
    # population, then take the mean |z| (or the PC-space L2 norm) per label
    # -> a per-label outlier score (higher = more outlier-like). frame=
    # "intrinsic" (derived from the table, no new spatial frame); role="qc".
    # An unbounded >=0 score (NOT a [0,1] ratio). LEAKAGE: qc_score is DERIVED
    # from the other feature columns, so if a downstream status/tier label is
    # itself derived from qc_score it must be held OUT of a classifier trained
    # on that label (circular). The optional PC1..PCN columns
    # (compute_pca_columns) are dynamic + count-suffixed, so they are not
    # statically enumerated here.
    # ----------------------------------------------------------------- #
    "qc_score": {
        "units": "score", "bounded": False, "role": "qc",
        "frame": "intrinsic", "redundancy_group": "composite_score",
    },
}


# Per-channel (B2) intensity columns are <prefix>-prefixed and dynamic (the
# prefix is a runtime choice — e.g. "nuc", "mem"), so they can't be statically
# enumerated. ``channel_intensity_tags(prefix)`` returns the manifest tags for
# the 11 ``<prefix>{suffix}`` columns compute_channel_intensity emits, so the EDA
# layer can tag a feature table whatever channels were sampled.
# The RAW ABSOLUTE intensity columns (mean/median/min/max/sum + boundary/interior
# means + the z-variation) are role="covariate": they track STAINING / ILLUMINATION
# / exposure, not phenotype, so a morphospace must regress them out / hold them out
# rather than stamping them as biology (the survey's raw-intensity confound). The
# SELF-NORMALIZING ratios (``_intensity_cv``, ``_boundary_interior_ratio``) divide
# the absolute scale out, so they ARE phenotype -> role="biology". ``_intensity_std``
# is an absolute spread (intensity unit) but is kept biology here (NOT in the audit's
# covariate re-tag list — see FIX 6).
_CHANNEL_INTENSITY_TAGS = {
    "_intensity_mean":   {"units": "intensity", "bounded": False, "role": "covariate"},
    "_intensity_median": {"units": "intensity", "bounded": False, "role": "covariate"},
    "_intensity_std":    {"units": "intensity", "bounded": False, "role": "biology"},
    "_intensity_min":    {"units": "intensity", "bounded": False, "role": "covariate"},
    "_intensity_max":    {"units": "intensity", "bounded": False, "role": "covariate"},
    "_intensity_sum":    {"units": "intensity", "bounded": False, "role": "covariate"},
    "_intensity_cv":     {"units": "ratio",     "bounded": False, "role": "biology"},
    "_boundary_intensity_mean": {"units": "intensity", "bounded": False, "role": "covariate"},
    "_interior_intensity_mean": {"units": "intensity", "bounded": False, "role": "covariate"},
    "_boundary_interior_ratio": {"units": "ratio",     "bounded": False, "role": "biology"},
    "_z_intensity_variation":   {"units": "intensity", "bounded": False, "role": "covariate"},
}


def channel_intensity_tags(prefix: str) -> Dict[str, dict]:
    """Manifest tags for the per-channel (B2) ``<prefix>{suffix}`` intensity columns.

    Returns ``{column_name: tag_dict}`` for the 11 columns
    :func:`epicycle.core.qc_features.compute_channel_intensity` emits for a given
    ``prefix`` (e.g. ``"nuc"``). All carry frame ``"image"`` + redundancy group
    ``channel_intensity_<prefix>``. The RAW absolute intensity columns
    (mean/median/min/max/sum + boundary/interior means + z-variation) are role
    ``"covariate"`` (a staining/illumination confound to regress out, NOT phenotype);
    the SELF-NORMALIZING ratios (``_intensity_cv`` / ``_boundary_interior_ratio``) +
    ``_intensity_std`` are role ``"biology"`` (see the table above for the per-suffix
    role).
    """
    group = f"channel_intensity_{prefix}"
    out: Dict[str, dict] = {}
    for suffix, base in _CHANNEL_INTENSITY_TAGS.items():
        out[f"{prefix}{suffix}"] = {
            "units": base["units"], "bounded": base["bounded"],
            "role": base["role"], "frame": "image", "redundancy_group": group,
        }
    return out


# Wave E — per-channel TEXTURE columns are <prefix>-prefixed + dynamic (the prefix
# is a runtime channel choice), so they can't be statically enumerated. Mirrors
# _CHANNEL_INTENSITY_TAGS / channel_intensity_tags. Texture is the spatial-
# organization signal (punctate vs diffuse vs hazy HCR; nuclear chromatin
# condensation) on the SAME raw-channel crop the intensity features read.
#
# 3D-native block (5): variance/mad/range_robust carry an intensity unit
# (variance is intensity², still "intensity" tag — unit-bearing/unbounded);
# entropy is in nats (a >=0 "score", unbounded above); granularity is a [0,1]
# residual fraction (bounded ratio).
# GLCM/Haralick block (6): contrast/dissimilarity/entropy are unbounded >=0;
# homogeneity/energy are [0,1] (bounded); correlation is signed [-1,1] (bounded,
# but downstream must not assume non-negative).
_CHANNEL_TEXTURE_TAGS = {
    "_texture_variance":    {"units": "intensity", "bounded": False},
    "_texture_entropy":     {"units": "score",     "bounded": False},
    "_texture_mad":         {"units": "intensity", "bounded": False},
    "_texture_granularity": {"units": "ratio",     "bounded": True},
    "_texture_range_robust": {"units": "intensity", "bounded": False},
    "_glcm_contrast":       {"units": "ratio",     "bounded": False},
    "_glcm_homogeneity":    {"units": "ratio",     "bounded": True},
    "_glcm_correlation":    {"units": "score",     "bounded": True},
    "_glcm_energy":         {"units": "ratio",     "bounded": True},
    "_glcm_entropy":        {"units": "score",     "bounded": False},
    "_glcm_dissimilarity":  {"units": "ratio",     "bounded": False},
}


def channel_texture_tags(prefix: str) -> Dict[str, dict]:
    """Manifest tags for the per-channel (Wave E) ``<prefix>{suffix}`` texture columns.

    Returns ``{column_name: tag_dict}`` for the 11 columns
    :func:`epicycle.core.qc_features.compute_channel_texture` emits for a given
    ``prefix`` (e.g. ``"nuc"``). All are signal-organization descriptors -> role
    ``"biology"``, frame ``"image"``, redundancy group ``channel_texture_<prefix>``.
    """
    group = f"channel_texture_{prefix}"
    out: Dict[str, dict] = {}
    for suffix, base in _CHANNEL_TEXTURE_TAGS.items():
        out[f"{prefix}{suffix}"] = {
            "units": base["units"], "bounded": base["bounded"],
            "role": "biology", "frame": "image", "redundancy_group": group,
        }
    return out


# Wave F — per-channel LOCALIZATION columns are <prefix>-prefixed +
# dynamic (the prefix is a runtime channel choice), so they can't be statically
# enumerated. Mirrors _CHANNEL_TEXTURE_TAGS / channel_texture_tags. Localization is
# the geometry-conditioned signal (membrane-vs-core shell split, radial slope,
# apical/basal split) — WHERE the signal sits, not just how strong it is.
#
# Shell block (4): the two shell means carry an intensity unit (unbounded); the
# shell gradient is a SIGNED intensity difference (unbounded, NOT non-negative —
# downstream must not assume >=0); the radial slope is an intensity-per-unit-radius
# signed slope (unbounded).
# Apical/basal block (3): the two half means carry an intensity unit (unbounded);
# the apical/basal ratio is an unbounded >=0 ratio.
# frame="image" for the SHELL/radial columns (object-intrinsic radial geometry, but
# tagged "image" like the texture/intensity siblings on the same channel); the
# apical/basal columns use the Wave C OUTWARD normal (the tissue frame) -> "tissue".
_CHANNEL_LOCALIZATION_TAGS = {
    "_shell_inner_mean":     {"units": "intensity", "bounded": False, "frame": "intrinsic"},
    "_shell_outer_mean":     {"units": "intensity", "bounded": False, "frame": "intrinsic"},
    "_shell_gradient":       {"units": "intensity", "bounded": False, "frame": "intrinsic"},
    "_radial_intensity_slope": {"units": "intensity", "bounded": False, "frame": "intrinsic"},
    # Mop-up O single-volume remainder — mid-band + 3-band radial profile +
    # membrane enrichment + outer/inner ratio + fraction-above-threshold. The means
    # carry an intensity unit (unbounded); the enrichment / outer-inner ratios are
    # unbounded >=0; fraction_above_threshold is a [0,1] positivity (bounded).
    "_shell_mid_mean":        {"units": "intensity", "bounded": False, "frame": "intrinsic"},
    "_radial_bin_inner_mean": {"units": "intensity", "bounded": False, "frame": "intrinsic"},
    "_radial_bin_mid_mean":   {"units": "intensity", "bounded": False, "frame": "intrinsic"},
    "_radial_bin_outer_mean": {"units": "intensity", "bounded": False, "frame": "intrinsic"},
    "_membrane_enrichment":   {"units": "ratio", "bounded": False, "frame": "intrinsic"},
    "_outer_to_inner_ratio":  {"units": "ratio", "bounded": False, "frame": "intrinsic"},
    "_fraction_above_threshold": {"units": "ratio", "bounded": True, "frame": "intrinsic"},
    "_apical_intensity_mean": {"units": "intensity", "bounded": False, "frame": "tissue"},
    "_basal_intensity_mean":  {"units": "intensity", "bounded": False, "frame": "tissue"},
    "_apical_basal_intensity_ratio": {"units": "ratio", "bounded": False, "frame": "tissue"},
    # Mop-up O — lateral (equatorial side-band) intensity, tissue frame.
    "_lateral_intensity_mean": {"units": "intensity", "bounded": False, "frame": "tissue"},
}


def channel_localization_tags(prefix: str) -> Dict[str, dict]:
    """Manifest tags for the per-channel (Wave F) ``<prefix>{suffix}``
    localization columns.

    Returns ``{column_name: tag_dict}`` for the 7 columns
    :func:`epicycle.core.qc_features.compute_channel_localization` emits for a given
    ``prefix`` (e.g. ``"mem"``). All are expression-localization descriptors -> role
    ``"biology"``, redundancy group ``channel_localization_<prefix>``. The SHELL /
    radial columns are object-intrinsic (frame ``"intrinsic"``); the apical/basal
    columns use the Wave C outward (tissue) frame (frame ``"tissue"``).
    """
    group = f"channel_localization_{prefix}"
    out: Dict[str, dict] = {}
    for suffix, base in _CHANNEL_LOCALIZATION_TAGS.items():
        out[f"{prefix}{suffix}"] = {
            "units": base["units"], "bounded": base["bounded"],
            "role": "biology", "frame": base["frame"], "redundancy_group": group,
        }
    return out


# Wave F — per-channel COMPARTMENT (nuclear-vs-cytoplasm) columns are
# <prefix>-prefixed + dynamic; computed in the reconcile PAIRED path
# (compute_channel_compartments, needs the cell + nucleus masks). The two means
# carry an intensity unit (unbounded); the nuclear/cytoplasm + perinuclear ratios
# are unbounded >=0. frame="intrinsic" (object + its paired nucleus, no external
# tissue frame, like the M6 paired nucleus family); role="biology".
_CHANNEL_COMPARTMENT_TAGS = {
    "_nuclear_mean":            {"units": "intensity", "bounded": False},
    "_cytoplasm_mean":          {"units": "intensity", "bounded": False},
    "_nuclear_cytoplasm_ratio": {"units": "ratio",     "bounded": False},
    "_perinuclear_enrichment":  {"units": "ratio",     "bounded": False},
    # Mop-up O compartment remainder — nuclear signal fraction ([0,1]
    # bounded), per-compartment texture entropy (>=0 nats, unbounded), compartment
    # mismatch score ([0,1] bounded).
    "_nuclear_signal_fraction":     {"units": "ratio", "bounded": True},
    "_nuclear_texture_entropy":     {"units": "score", "bounded": False},
    "_cytoplasmic_texture_entropy": {"units": "score", "bounded": False},
    "_compartment_mismatch_score":  {"units": "score", "bounded": True},
}


def channel_compartment_tags(prefix: str) -> Dict[str, dict]:
    """Manifest tags for the per-channel (Wave F) ``<prefix>{suffix}``
    nuclear-vs-cytoplasm compartment columns.

    Returns ``{column_name: tag_dict}`` for the 4 columns
    :func:`epicycle.core.qc_features.compute_channel_compartments` emits for a given
    ``prefix`` (e.g. ``"mem"``). Compartment-restricted expression descriptors ->
    role ``"biology"``, frame ``"intrinsic"`` (object + its paired nucleus),
    redundancy group ``channel_compartment_<prefix>``.
    """
    group = f"channel_compartment_{prefix}"
    out: Dict[str, dict] = {}
    for suffix, base in _CHANNEL_COMPARTMENT_TAGS.items():
        out[f"{prefix}{suffix}"] = {
            "units": base["units"], "bounded": base["bounded"],
            "role": "biology", "frame": "intrinsic", "redundancy_group": group,
        }
    return out


# P1b — marker-EXPECTATION QC columns are <prefix>-prefixed + dynamic;
# computed in the reconcile PAIRED path (compute_marker_expectation_flags, needs the
# cell + nucleus masks + a per-marker expected compartment). These are BIOLOGY
# CONFIG QC, not geometry -> role="qc" (per the prompt: "marker flags role=qc").
# frame="intrinsic" (object + its paired nucleus, like the compartment family). The
# in-expected fraction + concordance are [0,1] (bounded); the missing-signal flag is
# {0,1} (bounded).
_CHANNEL_MARKER_EXPECTATION_TAGS = {
    "_in_expected_compartment_fraction": {"units": "ratio", "bounded": True},
    "_missing_expected_signal_flag":     {"units": "ratio", "bounded": True},
    "_compartment_concordance":          {"units": "ratio", "bounded": True},
}


def channel_marker_expectation_tags(prefix: str) -> Dict[str, dict]:
    """Manifest tags for the per-marker (P1b) ``<prefix>{suffix}``
    expected-compartment QC columns.

    Returns ``{column_name: tag_dict}`` for the 3 columns
    :func:`epicycle.core.qc_features.compute_marker_expectation_flags` emits for a
    given marker ``prefix`` (e.g. ``"sox2"``) once a ``marker_expectations`` entry
    declares its expected compartment. These are BIOLOGY-CONFIG QC flags (not
    geometry) -> role ``"qc"``, frame ``"intrinsic"`` (object + its paired nucleus),
    redundancy group ``channel_marker_expectation_<prefix>``.
    """
    group = f"channel_marker_expectation_{prefix}"
    out: Dict[str, dict] = {}
    for suffix, base in _CHANNEL_MARKER_EXPECTATION_TAGS.items():
        out[f"{prefix}{suffix}"] = {
            "units": base["units"], "bounded": base["bounded"],
            "role": "qc", "frame": "intrinsic", "redundancy_group": group,
        }
    return out


# P1b — at-nucleus / perinuclear / neighbour local-expression-context
# columns are <prefix>-prefixed + dynamic. The at-nucleus + perinuclear-shell means
# live in the reconcile PAIRED path (compute_channel_local_context, needs the nucleus
# mask, frame="intrinsic"); the vs-neighbour ratio lives in the adjacency global pass
# (compute_channel_neighbor_context, frame="local"). All are expression-localization
# descriptors -> role="biology". The two MEANS carry an intensity unit (unbounded);
# the two RATIOS are unbounded >=0.
_CHANNEL_LOCAL_CONTEXT_TAGS = {
    "_at_nucleus_mean":              {"units": "intensity", "bounded": False, "frame": "intrinsic"},
    "_perinuclear_shell_mean":       {"units": "intensity", "bounded": False, "frame": "intrinsic"},
    "_perinuclear_to_nuclear_ratio": {"units": "ratio",     "bounded": False, "frame": "intrinsic"},
    # M3 NOTE: the "_vs_neighbor_mean_ratio" suffix is ALSO templated by
    # _LOCAL_COMPARATIVE_SUFFIXES (local_comparative_tags). No collision: this
    # one is keyed by a CHANNEL prefix (e.g. "mem_vs_neighbor_mean_ratio"),
    # the other by a base FEATURE prefix (e.g. "area_vs_neighbor_mean_ratio").
    "_vs_neighbor_mean_ratio":       {"units": "ratio",     "bounded": False, "frame": "local"},
}


def channel_local_context_tags(prefix: str) -> Dict[str, dict]:
    """Manifest tags for the per-channel (P1b) ``<prefix>{suffix}``
    at-nucleus / perinuclear / neighbour local-expression-context columns.

    Returns ``{column_name: tag_dict}`` for the 4 columns the P1b local-context pair
    emits for a given channel ``prefix``: the at-nucleus mean + perinuclear-shell mean
    + their ratio (:func:`epicycle.core.qc_features.compute_channel_local_context`,
    PAIRED path, frame ``"intrinsic"``) and the vs-neighbour ratio
    (:func:`epicycle.core.qc_features.compute_channel_neighbor_context`, adjacency
    global pass, frame ``"local"``). Expression-localization descriptors -> role
    ``"biology"``, redundancy group ``channel_local_context_<prefix>``.
    """
    group = f"channel_local_context_{prefix}"
    out: Dict[str, dict] = {}
    for suffix, base in _CHANNEL_LOCAL_CONTEXT_TAGS.items():
        out[f"{prefix}{suffix}"] = {
            "units": base["units"], "bounded": base["bounded"],
            "role": "biology", "frame": base["frame"], "redundancy_group": group,
        }
    return out


# Wave K — nuclear / cytoplasm PUNCTA FRACTION columns are <prefix>-prefixed
# + dynamic; computed in the reconcile PAIRED path
# (compute_channel_puncta_compartments, needs the cell + nucleus masks). Both are
# [0,1] fractions summing to 1 (bounded ratios). frame="intrinsic" (object + its
# paired nucleus, like the M6 paired family); role="biology".
_CHANNEL_PUNCTA_COMPARTMENT_TAGS = {
    "_nuclear_puncta_fraction":   {"units": "ratio", "bounded": True},
    "_cytoplasm_puncta_fraction": {"units": "ratio", "bounded": True},
}


def channel_puncta_compartment_tags(prefix: str) -> Dict[str, dict]:
    """Manifest tags for the per-channel (Wave K) ``<prefix>{suffix}``
    nuclear/cytoplasm puncta-fraction columns.

    Returns ``{column_name: tag_dict}`` for the 2 columns
    :func:`epicycle.core.qc_features.compute_channel_puncta_compartments` emits for a
    given ``prefix``. Compartment-restricted transcript-fraction descriptors -> role
    ``"biology"``, frame ``"intrinsic"`` (object + its paired nucleus), redundancy
    group ``channel_puncta_compartment_<prefix>``.
    """
    group = f"channel_puncta_compartment_{prefix}"
    out: Dict[str, dict] = {}
    for suffix, base in _CHANNEL_PUNCTA_COMPARTMENT_TAGS.items():
        out[f"{prefix}{suffix}"] = {
            "units": base["units"], "bounded": base["bounded"],
            "role": "biology", "frame": "intrinsic", "redundancy_group": group,
        }
    return out


# Wave L — single-nucleus CHROMATIN texture (nuclear_chromatin_*) columns are
# <prefix>-prefixed + dynamic (the chromatin/DAPI channel is a runtime choice), so
# they can't be statically enumerated. Computed in the reconcile PAIRED path
# (the paired cell-nucleus measurements, needs the nucleus mask).
# Both are nuclear chromatin-organization descriptors -> role="biology",
# frame="intrinsic" (the nucleus + its in-nucleus channel field). cv is a >=0
# robust dispersion ratio (unbounded); entropy is a >=0 histogram entropy
# (unbounded).
_NUCLEUS_CHROMATIN_TAGS = {
    "_chromatin_cv":      {"units": "ratio", "bounded": False},
    "_chromatin_entropy": {"units": "score", "bounded": False},
}


def nucleus_chromatin_tags(prefix: str) -> Dict[str, dict]:
    """Manifest tags for the per-channel (Wave L) ``<prefix>_chromatin_*``
    single-nucleus chromatin-texture columns.

    Returns ``{column_name: tag_dict}`` for the 2 columns
    the paired cell-nucleus measurements emits for a given
    chromatin/DAPI channel ``prefix``. Nuclear chromatin-organization descriptors ->
    role ``"biology"``, frame ``"intrinsic"`` (the nucleus + its in-nucleus channel
    field), redundancy group ``nucleus_chromatin_<prefix>``.
    """
    group = f"nucleus_chromatin_{prefix}"
    out: Dict[str, dict] = {}
    for suffix, base in _NUCLEUS_CHROMATIN_TAGS.items():
        out[f"{prefix}{suffix}"] = {
            "units": base["units"], "bounded": base["bounded"],
            "role": "biology", "frame": "intrinsic", "redundancy_group": group,
        }
    return out


# Wave G — per-channel PUNCTA columns are <prefix>-prefixed + dynamic (the
# prefix is a runtime channel choice), so they can't be statically enumerated.
# Mirrors _CHANNEL_TEXTURE_TAGS / channel_texture_tags. Puncta is the discrete
# spot-COUNTING (transcript) signal (blob_log) — the complement to the Wave-E
# continuous texture on the same channel.
#
# count is a >=0 score (unbounded); density carries a per-volume unit (unbounded);
# intensity_mean carries an intensity unit (unbounded); mean_volume carries a
# volume unit (unbounded); nn_distance carries a length unit (unbounded);
# punctate_fraction is a [0,1] fraction (bounded ratio).
#
# Wave K SPATIAL remainder: radial_position_mean is a [0,1] core-vs-rind
# position (bounded); clustering_index is the Clark–Evans ratio (>=0, unbounded —
# <1 clustered / >1 dispersed); polarity is a signed [-1,1] apical−basal balance
# (bounded, NOT non-negative); intensity_cv is a >=0 ratio (unbounded); dispersion
# is a >=0 spread / equiv-radius ratio (unbounded).
_CHANNEL_PUNCTA_TAGS = {
    "_puncta_count":          {"units": "score",     "bounded": False},
    "_puncta_density":        {"units": "ratio",     "bounded": False},
    "_puncta_intensity_mean": {"units": "intensity", "bounded": False},
    "_puncta_mean_volume":    {"units": "um3",       "bounded": False},
    "_puncta_nn_distance":    {"units": "um",        "bounded": False},
    "_punctate_fraction":     {"units": "ratio",     "bounded": True},
    "_puncta_radial_position_mean": {"units": "ratio", "bounded": True},
    "_puncta_clustering_index":     {"units": "ratio", "bounded": False},
    # _puncta_polarity is a SIGNED apical−basal transcript balance signed by the Wave
    # C outward normal -> sign_trusted=False (its sign flips with the apical/basal
    # frame, like the static signed apical/basal columns; see FIX 7).
    "_puncta_polarity":             {"units": "score", "bounded": True, "sign_trusted": False},
    "_puncta_intensity_cv":         {"units": "ratio", "bounded": False},
    "_puncta_dispersion":           {"units": "ratio", "bounded": False},
}


def channel_puncta_tags(prefix: str) -> Dict[str, dict]:
    """Manifest tags for the per-channel (Wave G) ``<prefix>{suffix}``
    HCR-puncta columns.

    Returns ``{column_name: tag_dict}`` for the 6 columns
    :func:`epicycle.core.qc_features.compute_channel_puncta` emits for a given
    ``prefix`` (e.g. ``"hcr"``). Transcript spot-counting descriptors -> role
    ``"biology"``, frame ``"image"`` (the spots live in the channel image), redundancy
    group ``channel_puncta_<prefix>``. The SIGNED ``<prefix>_puncta_polarity`` column
    additionally carries ``sign_trusted=False`` (its apical−basal sign comes from the
    Wave C outward normal — see FIX 7 / the ``sign_trusted`` note in the module header).
    """
    group = f"channel_puncta_{prefix}"
    out: Dict[str, dict] = {}
    for suffix, base in _CHANNEL_PUNCTA_TAGS.items():
        tag = {
            "units": base["units"], "bounded": base["bounded"],
            "role": "biology", "frame": "image", "redundancy_group": group,
        }
        if base.get("sign_trusted") is False:
            tag["sign_trusted"] = False
        out[f"{prefix}{suffix}"] = tag
    return out


# Wave K — channel-PAIR CO-LOCALIZATION columns are templated per
# UNORDERED channel pair as ``<a>_<b>{suffix}`` (the channel set + pairing is a
# runtime choice), so they can't be statically enumerated. Mirrors the per-channel
# *_tags helpers. The classic microscopy co-localization battery measured over the
# in-mask voxels common to both channels.
#
# pearson is a signed [-1,1] correlation (bounded, NOT non-negative); manders_m1 /
# manders_m2 / overlap_coefficient / coloc_fraction are [0,1] fractions (bounded);
# icq is Li's signed [-0.5,0.5] quotient (bounded, NOT non-negative). All
# dimensionless ratios/scores. frame="image" (the channel images); role="biology".
_CHANNEL_COLOC_TAGS = {
    "_pearson":             {"units": "score", "bounded": True},
    "_manders_m1":          {"units": "ratio", "bounded": True},
    "_manders_m2":          {"units": "ratio", "bounded": True},
    "_overlap_coefficient": {"units": "ratio", "bounded": True},
    "_icq":                 {"units": "score", "bounded": True},
    "_coloc_fraction":      {"units": "ratio", "bounded": True},
    # Mop-up O co-localization remainder — mean_ratio is an unbounded
    # >=0 intensity ratio; boundary_corr is a signed [-1,1] Pearson (bounded, NOT
    # non-negative); interior_ratio_delta is a signed unbounded ratio difference.
    "_mean_ratio":            {"units": "ratio", "bounded": False},
    "_boundary_corr":         {"units": "score", "bounded": True},
    "_interior_ratio_delta":  {"units": "score", "bounded": False},
}


def channel_coloc_tags(prefix_a: str, prefix_b: str) -> Dict[str, dict]:
    """Manifest tags for the per-channel-PAIR (Wave K)
    ``<a>_<b>{suffix}`` co-localization columns.

    Returns ``{column_name: tag_dict}`` for the 6 columns
    :func:`epicycle.core.qc_features.compute_channel_colocalization` emits for a given
    channel pair (e.g. ``"nuc"`` / ``"mem"``). Two-channel co-localization descriptors
    -> role ``"biology"``, frame ``"image"`` (the channel images), redundancy group
    ``channel_coloc_<a>_<b>``.
    """
    pair = f"{prefix_a}_{prefix_b}"
    group = f"channel_coloc_{pair}"
    out: Dict[str, dict] = {}
    for suffix, base in _CHANNEL_COLOC_TAGS.items():
        out[f"{pair}{suffix}"] = {
            "units": base["units"], "bounded": base["bounded"],
            "role": "biology", "frame": "image", "redundancy_group": group,
        }
    return out


# Wave A — local comparative (C12) columns are TEMPLATED per base feature
# (``<feature>_vs_neighbor_mean_ratio`` + ``<feature>_local_zscore``), so — like
# the per-channel intensity columns — they can't be statically enumerated (the
# base-feature set is a runtime choice; default
# qc_features.LOCAL_COMPARATIVE_FEATURES). ``local_comparative_tags(features)``
# returns the manifest tags for the 2 columns the C12 pass emits per base feature.
_LOCAL_COMPARATIVE_SUFFIXES = {
    # M3 NOTE: "_vs_neighbor_mean_ratio" is ALSO templated by
    # _CHANNEL_LOCAL_CONTEXT_TAGS (channel_local_context_tags). No collision:
    # here it is keyed by a base FEATURE prefix (e.g. "area_vs_neighbor_mean_ratio");
    # there by a CHANNEL prefix (e.g. "mem_vs_neighbor_mean_ratio").
    "_vs_neighbor_mean_ratio": {"units": "ratio", "bounded": False},
    "_local_zscore":          {"units": "score", "bounded": False},
    # Mop-up M — remainder: median ratio + signed difference + contact-
    # weighted-neighbour-mean ratio. The median ratio + contact-weighted ratio are
    # unbounded >=0 ratios; the difference is a signed unbounded score (its units
    # follow the base feature, generalised here to "score").
    "_vs_neighbor_median_ratio":          {"units": "ratio", "bounded": False},
    "_minus_neighbor_mean":               {"units": "score", "bounded": False},
    "_vs_contact_weighted_neighbor_mean": {"units": "ratio", "bounded": False},
}


def local_comparative_tags(features) -> Dict[str, dict]:
    """Manifest tags for the Wave A local-comparative (C12) templated columns.

    Returns ``{column_name: tag_dict}`` for the 2 ``<feature>{suffix}`` columns
    :func:`epicycle.core.qc_features.compute_local_comparative` emits per base
    feature in ``features`` (e.g. ``["area", "sphericity", ...]``). Both are
    neighbourhood-normalised QC/biology descriptors -> role ``"biology"``, frame
    ``"image"`` (the graph frame), redundancy group ``local_comparative_<feature>``.
    The ratio is an unbounded >=0; the local z-score is a signed unbounded score.
    """
    out: Dict[str, dict] = {}
    for feat in features:
        feat = str(feat)
        group = f"local_comparative_{feat}"
        for suffix, base in _LOCAL_COMPARATIVE_SUFFIXES.items():
            out[f"{feat}{suffix}"] = {
                "units": base["units"], "bounded": base["bounded"],
                "role": "biology", "frame": "image", "redundancy_group": group,
            }
    return out


def get_manifest() -> Dict[str, dict]:
    """Return a shallow copy of the feature manifest (name -> tag dict).

    A copy so callers can't mutate the module-level table in place. Entries are
    themselves shallow-copied dicts of plain scalars.
    """
    return {name: dict(tags) for name, tags in FEATURE_MANIFEST.items()}


__all__ = ["FEATURE_MANIFEST", "get_manifest", "channel_intensity_tags",
           "channel_texture_tags", "local_comparative_tags",
           "channel_localization_tags", "channel_compartment_tags",
           "channel_puncta_tags", "channel_coloc_tags",
           "channel_puncta_compartment_tags", "nucleus_chromatin_tags",
           "channel_marker_expectation_tags", "channel_local_context_tags"]
