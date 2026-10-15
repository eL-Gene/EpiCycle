"""Regression tests for the 5-eye CODE-CORRECTNESS audit fixes (2026-06-21).

Each test pins ONE confirmed correctness bug surfaced by the audit so it cannot
silently recur:

FIX 1 — ``compute_channel_puncta`` puncta_polarity FRAME MISMATCH (lens 1).
    The apical/basal spot split dotted VOXEL-frame displacements against the
    PHYSICAL-frame outward normal. Under anisotropic spacing the sign of a spot's
    apical/basal classification could flip. The fix scales the displacement by the
    per-axis spacing first (matching every sibling: localization / prob_geometry /
    axis_neighbor_persistence).

FIX 2 — ``territory_anisotropy`` /eps SENTINEL SPIKE (lens 2).
    A degenerate (1-voxel-line) territory whose smallest inertia eigenvalue is ~0
    divided by ``eps`` and produced a ~1e6 spike. The fix returns NaN for that
    degenerate case (a planar territory, which has a well-defined non-zero smallest
    eigenvalue, stays finite).

FIX 3 — SIGNED bounded ``score`` columns missing from ``BOUNDED_FEATURES`` (lens 4).
    ``_robust_standardize``'s auto-skew path does ``log1p(np.clip(X, 0, None))``,
    which ZEROES the negative half of a SIGNED feature. The signed bounded score
    columns must be in ``BOUNDED_FEATURES`` so they are never auto-log-transformed.

FIX 4 — eigenvector / eigenvalue columns in NO family (lens 5).
    The 12 ``{major,mid,minor}_axis_{x,y,z}`` + ``inertia_eigval_{major,mid,minor}``
    columns are emitted on every labels-only path but belonged to no
    ``_FEATURE_FAMILIES`` entry. The fix adds them to "morphology".
"""

import numpy as np
import pytest

from packages.epicycle_scaledseg.core import qc_features as qf
from packages.epicycle_scaledseg.core.feature_stage import _FEATURE_FAMILIES


# --------------------------------------------------------------------------- #
# FIX 1 — puncta_polarity frame mismatch
# --------------------------------------------------------------------------- #

def _two_spot_scene():
    """A channel crop with two bright LoG-detectable blobs at the audit's
    reproducing voxel positions, inside a generous mask."""
    shape = (28, 20, 20)
    mask = np.zeros(shape, dtype=bool)
    mask[2:26, 2:18, 2:18] = True
    ch = np.zeros(shape, dtype=np.float64)
    spots = [(20, 6, 10), (20, 14, 10)]
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    for (z, y, x) in spots:
        ch += 5.0 * np.exp(-(((zz - z) ** 2 + (yy - y) ** 2 + (xx - x) ** 2)
                             / (2.0 * 1.0 ** 2)))
    ch += 0.01
    return ch, mask


def test_fix1_puncta_polarity_uses_physical_frame_under_anisotropy():
    """The audit's exact reproducing case: two spots at the same physical z differing
    in y, anisotropic spacing (3,1,1), an outward normal mostly +y with a small +z
    component. With spacing applied, BOTH spots project positive (apical) -> polarity
    == +1, matching the hand-computed PHYSICAL answer. Without the fix the voxel-frame
    dot put one spot on the basal side -> polarity 0 (the contrast asserted below)."""
    ch, mask = _two_spot_scene()
    n = np.array([0.30, 0.95, 0.0], dtype=np.float64)
    n = n / np.linalg.norm(n)
    cen_vox = (14.5, 9.5, 9.5)

    out = qf.compute_channel_puncta(
        ch, mask, "g", spacing=(3.0, 1.0, 1.0),
        outward=tuple(n), centroid=cen_vox)
    assert out["g_puncta_count"] == 2.0, out["g_puncta_count"]
    # both apical in the PHYSICAL frame -> +1.
    assert out["g_puncta_polarity"] == pytest.approx(1.0), out["g_puncta_polarity"]


def test_fix1_puncta_polarity_voxel_frame_would_disagree():
    """The discriminating contrast: in the VOXEL frame (spacing=None) the SAME normal
    + spots split apical/basal (polarity 0). The anisotropic-spacing result (+1) MUST
    differ — this proves the fix actually changed the physical-vs-voxel behaviour and
    the test isn't vacuously passing."""
    ch, mask = _two_spot_scene()
    n = np.array([0.30, 0.95, 0.0], dtype=np.float64)
    n = n / np.linalg.norm(n)
    cen_vox = (14.5, 9.5, 9.5)

    out_phys = qf.compute_channel_puncta(
        ch, mask, "g", spacing=(3.0, 1.0, 1.0),
        outward=tuple(n), centroid=cen_vox)
    out_vox = qf.compute_channel_puncta(
        ch, mask, "g", spacing=None,
        outward=tuple(n), centroid=cen_vox)
    assert out_vox["g_puncta_count"] == 2.0
    # voxel frame splits the two spots -> 0; physical frame -> +1. They DIFFER.
    assert out_vox["g_puncta_polarity"] == pytest.approx(0.0)
    assert out_phys["g_puncta_polarity"] != pytest.approx(out_vox["g_puncta_polarity"])


# --------------------------------------------------------------------------- #
# FIX 2 — territory_anisotropy /eps sentinel spike
# --------------------------------------------------------------------------- #

def test_fix2_territory_anisotropy_nan_for_degenerate_line():
    """A 1-voxel-thick LINE cell -> its (cap-0) watershed territory is the line
    itself, whose smallest inertia eigenvalue is ~0. The OLD code divided by ``eps``
    and produced a ~1e6 sentinel SPIKE; the fix returns NaN (anisotropy undefined for
    a degenerate line). Asserts it is NOT a finite huge value."""
    lab = np.zeros((20, 20, 40), dtype=np.int32)
    lab[10, 10, 5:9] = 1                       # a short 1-voxel-thick rod
    df = qf.extract_features(lab)
    qf.compute_watershed_territory(lab, df, max_distance_voxels=0)
    val = df.set_index("label").loc[1, "territory_anisotropy"]
    assert np.isnan(val), val
    # explicitly: not the old ~1e6 sentinel.
    assert not (np.isfinite(val) and abs(val) > 1e4)


def test_fix2_planar_territory_stays_finite():
    """Don't over-NaN: a 1-voxel-thick PLANE territory has a well-defined non-zero
    smallest in-plane eigenvalue, so its anisotropy is FINITE (and modest), not NaN.
    A compact cube territory is likewise finite ~1."""
    lab = np.zeros((20, 30, 30), dtype=np.int32)
    lab[10, 8:22, 8:22] = 1                    # a flat square plane (2D extent)
    df = qf.extract_features(lab)
    qf.compute_watershed_territory(lab, df, max_distance_voxels=0)
    plane = df.set_index("label").loc[1, "territory_anisotropy"]
    assert np.isfinite(plane), plane
    assert plane < 1e3

    lab2 = np.zeros((24, 24, 24), dtype=np.int32)
    lab2[8:16, 8:16, 8:16] = 1                 # a compact cube
    df2 = qf.extract_features(lab2)
    qf.compute_watershed_territory(lab2, df2, max_distance_voxels=0)
    cube = df2.set_index("label").loc[1, "territory_anisotropy"]
    assert np.isfinite(cube) and cube == pytest.approx(1.0, abs=0.2), cube


# --------------------------------------------------------------------------- #
# FIX 3 — signed bounded score columns in BOUNDED_FEATURES
# --------------------------------------------------------------------------- #

# Signed (range includes negatives), mathematically-bounded score columns. Each is
# an emitted bounded column whose negative half would be destroyed by the auto-skew
# log1p(clip(X, 0, None)).
_SIGNED_BOUNDED = [
    "prob_apical_basal_gradient",
    "prob_shell_gradient",
    "prob_boundary_drop",
    "contact_polarity_bias",
    "apical_vs_basal_elongation",
    "territory_z_bias",
    "disc_vs_rod_score",
]


@pytest.mark.parametrize("col", _SIGNED_BOUNDED)
def test_fix3_signed_bounded_in_BOUNDED_FEATURES(col):
    """Each signed bounded score column is listed in BOUNDED_FEATURES."""
    assert col in qf.BOUNDED_FEATURES, f"{col} missing from BOUNDED_FEATURES"


def test_fix3_signed_bounded_columns_exist_and_are_bounded():
    """Each added name is a REAL emitted column tagged bounded:True in the manifest
    (so we're not protecting a phantom name)."""
    from packages.epicycle_scaledseg.core import feature_manifest as fm
    man = fm.get_manifest()
    for col in _SIGNED_BOUNDED:
        assert col in man, f"{col} not in manifest"
        assert man[col]["bounded"] is True, f"{col} not tagged bounded"


def test_fix3_skew_detection_skips_signed_bounded():
    """A synthetic heavily-RIGHT-SKEWED, SIGNED frame (values in [-1, 1] including
    negatives, |skew| > 1) -> ``_detect_skewed_columns`` selects NONE of the signed
    bounded columns for log transform. Without the BOUNDED_FEATURES membership they
    would be flagged + their negative half clipped to 0."""
    rng = np.random.default_rng(0)
    n = 200
    cols = list(_SIGNED_BOUNDED)
    X = np.empty((n, len(cols)), dtype=np.float64)
    for j in range(len(cols)):
        # right-skewed in [-1, 1] -> includes negatives, |skew| > 1.
        X[:, j] = rng.beta(0.4, 5.0, n) * 2.0 - 1.0

    # sanity: the synthetic columns ARE skewed past the threshold (so the bug is live
    # if they weren't bounded).
    from scipy.stats import skew as _skew
    assert all(_skew(X[:, j]) > qf._SKEW_THRESHOLD for j in range(len(cols)))
    # ... and they include negatives (genuinely signed).
    assert (X < 0).any()

    flagged = qf._detect_skewed_columns(X, cols)
    assert flagged == [], (
        "signed bounded columns were flagged for log transform: "
        + ", ".join(cols[j] for j in flagged))


def test_fix3_robust_standardize_preserves_signed_half():
    """End-to-end: a signed bounded column survives ``_robust_standardize`` with its
    negative-side ordering INTACT (the smallest input maps to the smallest output).
    log1p(clip(X, 0, None)) would have collapsed every negative value to log1p(0)==0,
    tying them and destroying the order."""
    rng = np.random.default_rng(1)
    n = 200
    cols = ["disc_vs_rod_score"]                       # one representative signed col
    X = (rng.beta(0.4, 5.0, n) * 2.0 - 1.0).reshape(-1, 1)
    arg_min_in = int(np.argmin(X[:, 0]))
    Xs = qf._robust_standardize(X.copy(), cols)
    # the most-negative input is still the most-negative (smallest) standardized value.
    assert int(np.argmin(Xs[:, 0])) == arg_min_in
    # the negative inputs did NOT all collapse to a single tied value.
    neg = X[:, 0] < 0
    assert np.unique(np.round(Xs[neg, 0], 9)).size > 1


# --------------------------------------------------------------------------- #
# FIX 4 — every labels-only emitted column is in exactly one family
# --------------------------------------------------------------------------- #

# the 12 orientation / eigenvalue columns the fix adds.
_EIGEN_COLS = [
    "major_axis_x", "major_axis_y", "major_axis_z",
    "mid_axis_x", "mid_axis_y", "mid_axis_z",
    "minor_axis_x", "minor_axis_y", "minor_axis_z",
    "inertia_eigval_major", "inertia_eigval_mid", "inertia_eigval_minor",
]


def _labels_only_feature_columns():
    """The labels-only emitted FEATURE columns (exclude ``label`` + positional
    ``centroid-*`` / ``bbox-*``), from the base ``extract_features`` path."""
    lab = np.zeros((24, 24, 24), dtype=np.int32)
    lab[3:10, 3:10, 3:16] = 1
    lab[12:20, 12:20, 12:20] = 2
    df = qf.extract_features(lab)
    return [c for c in df.columns
            if c != "label"
            and not c.startswith("centroid-")
            and not c.startswith("bbox-")]


@pytest.mark.parametrize("col", _EIGEN_COLS)
def test_fix4_eigen_columns_in_morphology_family(col):
    """Each of the 12 orientation / eigenvalue columns is now in the morphology
    family (where ``major_axis_z_angle`` lives)."""
    assert col in _FEATURE_FAMILIES["morphology"], col


def test_families_cover_all_emitted_columns():
    """Every labels-only emitted FEATURE column is in EXACTLY ONE _FEATURE_FAMILIES
    entry — no orphans (the bug) and no double-listing."""
    from collections import Counter
    member_counts = Counter()
    for cols in _FEATURE_FAMILIES.values():
        for c in cols:
            member_counts[c] += 1

    emitted = _labels_only_feature_columns()
    orphans = [c for c in emitted if member_counts[c] == 0]
    duplicates = [c for c in emitted if member_counts[c] > 1]
    assert not orphans, f"emitted columns in NO family: {orphans}"
    assert not duplicates, f"emitted columns in MULTIPLE families: {duplicates}"


def test_no_family_member_listed_twice_globally():
    """Defensive: no column appears in two different families anywhere in the
    registry (not just the labels-only emit subset)."""
    from collections import Counter
    member_counts = Counter()
    for cols in _FEATURE_FAMILIES.values():
        for c in cols:
            member_counts[c] += 1
    dups = {c: n for c, n in member_counts.items() if n > 1}
    assert not dups, f"columns listed in multiple families: {dups}"
