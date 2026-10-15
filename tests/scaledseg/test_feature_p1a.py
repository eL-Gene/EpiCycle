"""P1a feature tests — the SIGNED local tissue frame (atlas ) + the SIGNED
probability-geometry remainder.

These ride on the Wave C per-cell OUTWARD normal (``tissue_outward_*``) to ORIENT a
direction-bearing quantity against the tissue's apical (= +outward) / basal
(= -outward) direction. They EXTEND the existing ``compute_tissue_frame_features``
/ ``compute_prob_geometry`` functions, gated on the outward columns being present:
default-off byte-identical (the columns are ABSENT — frame — or NaN — prob — without
a trusted outward normal).

NEW features (NOT a monorepo port) built from the atlas equations. Pure-function
known-geometry tests (no GPU / Cellpose); synthetic arrangements whose answers we
know by construction:

* a cell elongated ALONG the outward normal -> ``apical_basal_axis_alignment`` ≈ 1
  (columnar); a cell elongated PERPENDICULAR -> ≈ 0.
* a "club" cell with a long thin BASAL stalk + a fat APICAL head -> ``cell_tilt_
  polarity`` and ``apical_vs_basal_elongation`` are SIGNED and FLIP when the outward
  normal flips; ``apical_basal_axis_alignment`` (a |cos| MAGNITUDE) is unchanged.
* a one-sided ("shaved") ball -> ``tangential_asymmetry_magnitude`` > a symmetric
  ball (≈ 0); the metric is rotation-invariant (NO canonical L/R axis).
* a prob field higher APICALLY -> ``prob_apical_basal_gradient`` > 0; flipping the
  outward normal flips its sign; no outward normal -> NaN (default-off).
* the manifest tags every new column + the ``_FEATURE_FAMILIES`` registry lockstep.

HONESTY NOTE (this goes to external review): the SIGNED L/R rows the atlas lists
(``front_back_asymmetry`` / ``left_right_asymmetry`` /
``minor_axis_vs_local_tangent_angle``) are DEFERRED — a single cell + its local
normal carries NO canonical in-plane axis. We ship the ROTATION-INVARIANT
``tangential_asymmetry_magnitude`` instead. These tests assert the magnitude
behaviour, NOT a signed L/R.
"""

import numpy as np
import pandas as pd

from packages.epicycle_scaledseg.core import qc_features as qf
from packages.epicycle_scaledseg.core import feature_manifest as fm
from packages.epicycle_scaledseg.core.qc_features import (
    compute_tissue_frame_features, compute_prob_geometry,
    SIGNED_TISSUE_FRAME_COLUMNS,
)


# --------------------------------------------------------------------------- #
# column groups
# --------------------------------------------------------------------------- #
_SIGNED_FRAME = list(SIGNED_TISSUE_FRAME_COLUMNS)
_SIGNED_FRAME_AXIS = [
    "apical_basal_axis_alignment", "minor_axis_in_tangent_plane_fraction",
]
_SIGNED_FRAME_MASS = [
    "tangent_plane_anisotropy", "cell_tilt_polarity", "apical_vs_basal_elongation",
    "tangential_asymmetry_magnitude",
]
_PROB_SIGNED = "prob_apical_basal_gradient"
# the atlas rows we DELIBERATELY do NOT emit (need a global tissue reference axis).
_DEFERRED_LR = (
    "front_back_asymmetry", "left_right_asymmetry",
    "minor_axis_vs_local_tangent_angle",
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _with_outward(df, outward=(0.0, 0.0, 1.0), conf=1.0):
    df = df.copy()
    df["tissue_outward_z"] = outward[0]
    df["tissue_outward_y"] = outward[1]
    df["tissue_outward_x"] = outward[2]
    df["tissue_outward_confidence"] = conf
    return df


def _rod_along(axis, shape=(24, 24, 40)):
    """A rod elongated along ``axis`` ('x' or 'y'), centred in the volume."""
    Z, Y, X = shape
    lab = np.zeros(shape, np.int32)
    if axis == "x":
        lab[Z // 2 - 2:Z // 2 + 2, Y // 2 - 2:Y // 2 + 2, 5:X - 5] = 1
    else:  # 'y'
        lab[Z // 2 - 2:Z // 2 + 2, 5:Y - 5, X // 2 - 2:X // 2 + 2] = 1
    return lab


def _club_cell():
    """A long thin BASAL stalk (-X) + a fat APICAL head (+X). Used with outward=+X
    so the basal half spreads further along the normal."""
    Z, Y, X = 24, 24, 44
    lab = np.zeros((Z, Y, X), np.int32)
    lab[11:13, 11:13, 4:22] = 1      # thin basal stalk (-X side)
    lab[8:16, 8:16, 22:30] = 1       # fat apical head (+X side)
    return lab


def _ball(shape=(24, 40, 40), c=(12, 20, 20), R=8, shave_y=None):
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    m = ((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) <= R * R
    if shave_y is not None:
        m = m & (yy <= shave_y)         # cut off the +Y cap -> a one-sided shape
    lab = np.zeros(shape, np.int32)
    lab[m] = 1
    return lab


def _frame_row(lab, outward=(0.0, 0.0, 1.0), conf=1.0, spacing=None):
    df = qf.extract_features(lab, spacing=spacing).sort_values("label").reset_index(drop=True)
    df = _with_outward(df, outward=outward, conf=conf)
    sp = tuple(spacing) if spacing is not None else None
    compute_tissue_frame_features(df, spacing=sp, full_labels=lab)
    return df.set_index("label").loc[1]


# =========================================================================== #
# 1) axis-only signed columns — alignment with the outward normal
# =========================================================================== #
def test_alignment_high_when_elongated_along_normal():
    """A cell elongated ALONG the outward normal (+X) -> apical_basal_axis_alignment
    ≈ 1 (columnar)."""
    r = _frame_row(_rod_along("x"), outward=(0.0, 0.0, 1.0))
    assert r["apical_basal_axis_alignment"] > 0.95


def test_alignment_low_when_elongated_perpendicular():
    """A cell elongated PERPENDICULAR to the normal (long Y, outward +X) -> ≈ 0."""
    r = _frame_row(_rod_along("y"), outward=(0.0, 0.0, 1.0))
    assert r["apical_basal_axis_alignment"] < 0.05


def test_alignment_is_magnitude_unsigned():
    """apical_basal_axis_alignment is a |cos| MAGNITUDE — it is invariant to FLIPPING
    the outward normal (and to the arbitrary major-axis eigenvector sign)."""
    lab = _rod_along("x")
    r_pos = _frame_row(lab, outward=(0.0, 0.0, 1.0))
    r_neg = _frame_row(lab, outward=(0.0, 0.0, -1.0))
    assert r_pos["apical_basal_axis_alignment"] == \
        pytest_approx(r_neg["apical_basal_axis_alignment"])


def test_axis_signed_columns_bounded():
    r = _frame_row(_rod_along("x"))
    for c in ("apical_basal_axis_alignment", "minor_axis_in_tangent_plane_fraction",
              "tangent_plane_anisotropy"):
        v = r[c]
        if np.isfinite(v):
            assert -1e-9 <= v <= 1.0 + 1e-9, (c, v)


# =========================================================================== #
# 2) mass-moment signed columns — SIGNED + flip with the outward direction
# =========================================================================== #
def test_tilt_polarity_and_elongation_signed_club_cell():
    """The club cell (long basal stalk, fat apical head; outward +X): the basal half
    spreads further along the normal -> apical_vs_basal_elongation < 0; the mass tail
    leans basal -> cell_tilt_polarity < 0."""
    r = _frame_row(_club_cell(), outward=(0.0, 0.0, 1.0))
    assert r["apical_vs_basal_elongation"] < 0.0
    assert r["cell_tilt_polarity"] < 0.0


def test_mass_moment_signs_flip_with_outward_direction():
    """Flipping the outward normal flips the SIGN of the signed mass-moment columns
    (they ARE projections onto that signed vector) but NOT the magnitude alignment."""
    lab = _club_cell()
    r_pos = _frame_row(lab, outward=(0.0, 0.0, 1.0))
    r_neg = _frame_row(lab, outward=(0.0, 0.0, -1.0))
    assert r_pos["cell_tilt_polarity"] == pytest_approx(-r_neg["cell_tilt_polarity"])
    assert r_pos["apical_vs_basal_elongation"] == \
        pytest_approx(-r_neg["apical_vs_basal_elongation"])
    # the magnitude alignment is unchanged by the flip.
    assert r_pos["apical_basal_axis_alignment"] == \
        pytest_approx(r_neg["apical_basal_axis_alignment"])


def test_elongation_bounded_in_minus1_1():
    r = _frame_row(_club_cell())
    v = r["apical_vs_basal_elongation"]
    if np.isfinite(v):
        assert -1.0 - 1e-9 <= v <= 1.0 + 1e-9


# =========================================================================== #
# 3) tangential asymmetry — rotation-invariant, asymmetric > symmetric
# =========================================================================== #
def test_tangential_asymmetry_higher_for_one_sided_shape():
    """A one-sided ("shaved on +Y") ball has a tangentially lopsided SURFACE about its
    mass centroid -> tangential_asymmetry_magnitude > a symmetric ball (≈ 0)."""
    r_shaved = _frame_row(_ball(shave_y=22), outward=(0.0, 0.0, 1.0))
    r_sym = _frame_row(_ball(), outward=(0.0, 0.0, 1.0))
    assert r_shaved["tangential_asymmetry_magnitude"] > \
        r_sym["tangential_asymmetry_magnitude"] + 0.02
    assert r_sym["tangential_asymmetry_magnitude"] < 0.02       # symmetric ~0
    assert r_shaved["tangential_asymmetry_magnitude"] >= 0.0    # a magnitude


def _ball_shaved_z(shape=(24, 40, 40), c=(12, 20, 20), R=8, shave_z=14):
    """A ball with its +Z cap cut off (a tangent-plane direction for outward=+X)."""
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    m = ((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) <= R * R
    m = m & (zz <= shave_z)
    lab = np.zeros(shape, np.int32)
    lab[m] = 1
    return lab


def test_tangential_asymmetry_rotation_invariant_in_plane():
    """The metric is the MAGNITUDE of the in-plane surface moment, so it is invariant
    to ROTATING the cell within the tangent plane (no canonical L/R axis). A shaved
    ball cut on +Y vs cut on +Z (both tangential to outward=+X) give the same value."""
    r_y = _frame_row(_ball(shave_y=22), outward=(0.0, 0.0, 1.0))
    r_z = _frame_row(_ball_shaved_z(shave_z=14), outward=(0.0, 0.0, 1.0))
    assert abs(r_y["tangential_asymmetry_magnitude"]
               - r_z["tangential_asymmetry_magnitude"]) < 0.02


# =========================================================================== #
# 4) default-off contract (frame) + NaN guards
# =========================================================================== #
def test_signed_frame_absent_without_waveC():
    """No tissue_outward_* columns (Wave C did not run) -> the SIGNED columns are not
    created (default-off byte-identical)."""
    lab = _rod_along("x")
    df = qf.extract_features(lab).sort_values("label").reset_index(drop=True)
    compute_tissue_frame_features(df, full_labels=lab)        # no outward columns
    for c in _SIGNED_FRAME:
        assert c not in df.columns, c


def test_signed_frame_unsigned_columns_byte_identical():
    """Adding the outward columns + re-running adds ONLY the signed columns; the
    UNSIGNED frame columns are byte-identical."""
    Z, Y, X = 20, 20, 20
    # a small 3x3-ish sheet so the unsigned plane fit is defined for the centre cell.
    lab = np.zeros((Z, Y, X), np.int32)
    nid = 1
    for yi in range(2, 17, 5):
        for xi in range(2, 17, 5):
            lab[8:12, yi:yi + 4, xi:xi + 4] = nid
            nid += 1
    df_base = qf.extract_features(lab).sort_values("label").reset_index(drop=True)
    qf.compute_neighbor_features(lab, df_base, max_radius=3)
    df_signed = df_base.copy()
    df_signed = _with_outward(df_signed)
    compute_tissue_frame_features(df_base, adjacency=df_base.attrs.get("_adjacency_list"))
    compute_tissue_frame_features(
        df_signed, adjacency=df_signed.attrs.get("_adjacency_list"), full_labels=lab)
    unsigned = [c for c in df_base.columns if c not in df_signed.columns
                or c not in _SIGNED_FRAME]
    shared = [c for c in df_base.columns if c in df_signed.columns and c not in _SIGNED_FRAME]
    pd.testing.assert_frame_equal(
        df_base[shared].reset_index(drop=True),
        df_signed[shared].reset_index(drop=True),
        check_dtype=False,
    )
    for c in _SIGNED_FRAME:
        assert c in df_signed.columns and c not in df_base.columns, c


def test_signed_frame_low_confidence_is_nan():
    """A cell with a deliberately ZEROED outward confidence -> NaN signed columns."""
    r = _frame_row(_club_cell(), outward=(0.0, 0.0, 1.0), conf=0.0)
    for c in _SIGNED_FRAME:
        assert np.isnan(r[c]), c


def test_mass_columns_nan_without_full_labels():
    """Without full_labels the AXIS signed columns are defined but the MASS-MOMENT
    ones are NaN (no voxels to read)."""
    lab = _rod_along("x")
    df = qf.extract_features(lab).sort_values("label").reset_index(drop=True)
    df = _with_outward(df)
    compute_tissue_frame_features(df)            # no full_labels
    r = df.set_index("label").loc[1]
    for c in _SIGNED_FRAME_AXIS:
        assert np.isfinite(r[c]), c
    for c in _SIGNED_FRAME_MASS:
        assert np.isnan(r[c]), c


def test_empty_df_typed_signed_columns():
    df = pd.DataFrame({"label": pd.Series([], dtype=np.int64),
                       "centroid-0": pd.Series([], dtype=np.float64),
                       "centroid-1": pd.Series([], dtype=np.float64),
                       "centroid-2": pd.Series([], dtype=np.float64),
                       "tissue_outward_z": pd.Series([], dtype=np.float64),
                       "tissue_outward_y": pd.Series([], dtype=np.float64),
                       "tissue_outward_x": pd.Series([], dtype=np.float64),
                       "tissue_outward_confidence": pd.Series([], dtype=np.float64)})
    compute_tissue_frame_features(df, full_labels=np.zeros((4, 4, 4), np.int32))
    for c in _SIGNED_FRAME:
        assert c in df.columns, c


# =========================================================================== #
# 5) prob signed — apical-vs-basal confidence difference
# =========================================================================== #
def _rod_mask_and_prob(grad="apical"):
    """A rod along X + a prob field that ramps brighter toward +X (apical) or -X."""
    Z, Y, X = 8, 8, 30
    mask = np.zeros((Z, Y, X), dtype=bool)
    mask[2:6, 2:6, 2:28] = True
    xx = np.arange(X)
    ramp = 0.3 + 0.6 * xx / X            # rises with +X
    if grad == "basal":
        ramp = ramp[::-1]
    prob = np.zeros((Z, Y, X), dtype=np.float64)
    prob[:] = ramp[None, None, :]
    prob[~mask] = 0.0
    return mask, prob


def test_prob_apical_basal_gradient_positive_when_apical_brighter():
    """prob higher APICALLY (+X), outward = +X -> prob_apical_basal_gradient > 0."""
    mask, prob = _rod_mask_and_prob("apical")
    out = compute_prob_geometry(mask, prob, outward=(0.0, 0.0, 1.0))
    assert out[_PROB_SIGNED] > 0.0


def test_prob_apical_basal_gradient_negative_when_basal_brighter():
    mask, prob = _rod_mask_and_prob("basal")
    out = compute_prob_geometry(mask, prob, outward=(0.0, 0.0, 1.0))
    assert out[_PROB_SIGNED] < 0.0


def test_prob_apical_basal_gradient_sign_flips_with_outward():
    mask, prob = _rod_mask_and_prob("apical")
    pos = compute_prob_geometry(mask, prob, outward=(0.0, 0.0, 1.0))
    neg = compute_prob_geometry(mask, prob, outward=(0.0, 0.0, -1.0))
    assert pos[_PROB_SIGNED] == pytest_approx(-neg[_PROB_SIGNED])


def test_prob_apical_basal_gradient_bounded():
    mask, prob = _rod_mask_and_prob("apical")
    out = compute_prob_geometry(mask, prob, outward=(0.0, 0.0, 1.0))
    assert -1.0 - 1e-9 <= out[_PROB_SIGNED] <= 1.0 + 1e-9


def test_prob_apical_basal_gradient_nan_without_outward():
    """No outward normal -> NaN (default-off; the column is always a present key)."""
    mask, prob = _rod_mask_and_prob("apical")
    out = compute_prob_geometry(mask, prob)
    assert _PROB_SIGNED in out
    assert np.isnan(out[_PROB_SIGNED])


def test_prob_apical_basal_gradient_nan_on_empty_mask():
    out = compute_prob_geometry(np.zeros((6, 6, 6), bool), np.zeros((6, 6, 6)),
                                outward=(0.0, 0.0, 1.0))
    assert np.isnan(out[_PROB_SIGNED])


# =========================================================================== #
# 6) manifest + family registry + honesty (deferred L/R)
# =========================================================================== #
def test_manifest_tags_all_p1a_columns():
    man = fm.get_manifest()
    for c in (*_SIGNED_FRAME, _PROB_SIGNED):
        assert c in man, f"{c} missing from FEATURE_MANIFEST"
        tags = man[c]
        for key in ("units", "bounded", "role", "frame", "redundancy_group"):
            assert key in tags, (c, key)
        assert tags["frame"] == "tissue", c               # all signed-frame
        assert tags["role"] in ("biology", "qc", "covariate")
        assert isinstance(tags["bounded"], bool)


def test_manifest_p1a_bounded_flags():
    man = fm.get_manifest()
    for c in ("apical_basal_axis_alignment", "minor_axis_in_tangent_plane_fraction",
              "tangent_plane_anisotropy", "apical_vs_basal_elongation",
              "prob_apical_basal_gradient"):
        assert man[c]["bounded"] is True, c
    for c in ("cell_tilt_polarity", "tangential_asymmetry_magnitude"):
        assert man[c]["bounded"] is False, c


def test_feature_stage_families_registry_includes_p1a():
    from packages.epicycle_scaledseg.core.feature_stage import _FEATURE_FAMILIES
    # the signed frame columns are folded into the tissue_frame family.
    assert set(_SIGNED_FRAME).issubset(set(_FEATURE_FAMILIES["tissue_frame"]))
    # the signed prob column is folded into probability_geometry.
    assert _PROB_SIGNED in _FEATURE_FAMILIES["probability_geometry"]


def test_deferred_lr_columns_not_emitted():
    """HONESTY: the signed L/R atlas rows are DEFERRED (need a global tissue reference
    axis) — they are NOT emitted and NOT NaN-stubbed as if done."""
    r = _frame_row(_club_cell())
    for c in _DEFERRED_LR:
        assert c not in r.index, c
    man = fm.get_manifest()
    for c in _DEFERRED_LR:
        assert c not in man, c


# --------------------------------------------------------------------------- #
# tiny local approx (avoid importing pytest.approx at module top for clarity)
# --------------------------------------------------------------------------- #
import pytest  # noqa: E402


def pytest_approx(v, rel=1e-6, abs_=1e-9):
    return pytest.approx(v, rel=rel, abs=abs_)
