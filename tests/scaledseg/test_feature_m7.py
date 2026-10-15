"""M7 feature tests — composite qc_score (B7; global table pass).

Pure-function known-shape tests (no tile readers / GPU). Builds a small synthetic
feature table with the columns ``compute_composite_score`` reads, exercises the
ported helpers + the scoring function directly, then the OPT-IN global pass wired
into ``extract_label_features``.

Asserts:

* The ported transform RULES behave: a bounded feature is NOT in the auto-skew
  set (so it's never log1p'd), a skewed positive feature IS detected; the curated
  constants are populated.
* :func:`compute_composite_score` (zscore) emits a finite ``qc_score`` and orders
  it sensibly — a clearly-abnormal row scores worse than the population.
* The ``pca`` method also runs, is finite, and flags the same abnormal row.
* :func:`compute_pca_columns` writes PC columns, and qc_score(method="pca") reads
  them (the persisted-PC fast path) giving the same shift-to-zero-min L2 score.
* Degenerate inputs (no usable columns, <2 rows) -> qc_score == 0.0, no crash.
* The ``qc_score=False`` default leaves the table byte-identical (additive-opt-in).
* The opt-in pass: ``extract_label_features(adjacency=True, qc_score=True)`` adds
  exactly the ``qc_score`` column on top of the adjacency output; with
  ``adjacency=False, qc_score=True`` it still scores (labels-only subset) + warns.
* The manifest tags ``qc_score`` and the family registry lists it.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from packages.epicycle_scaledseg.core import qc_features as qf
from packages.epicycle_scaledseg.core import feature_manifest as fm


# --------------------------------------------------------------------------- #
# synthetic feature table
# --------------------------------------------------------------------------- #

def _synthetic_table(n_normal: int = 30, seed: int = 0) -> pd.DataFrame:
    """A population of ``n_normal`` tightly-clustered 'normal' cells PLUS one
    clearly-abnormal cell (the last row), with the columns qc_score reads.

    The abnormal row is an outlier on every curated feature simultaneously
    (huge area / elongation / many branches / low sphericity & confidence), so
    it must score worse than every normal row regardless of method.
    """
    rng = np.random.RandomState(seed)
    cols = qf.PCA_FEATURES

    # baseline "typical" value per curated feature (rough, biologically plausible)
    base = {
        "area": 1000.0, "extent": 0.55, "equivalent_diameter": 12.0,
        "aspect_ratio": 1.4, "true_sphericity": 0.85, "solidity": 0.92,
        "major_axis_z_angle": 30.0, "inertia_elongation": 1.6,
        "surface_area": 600.0, "neighbor_count": 6.0,
        "nearest_neighbor_distance": 2.0, "skeleton_branches": 1.0,
        "prob_mean": 0.9, "prob_boundary_drop": 0.2,
        "prob_boundary_entropy": 0.15,
    }
    data = {}
    for c in cols:
        mu = base[c]
        # small heteroscedastic jitter so MAD is non-zero but the cloud is tight
        data[c] = rng.normal(mu, abs(mu) * 0.05 + 1e-3, size=n_normal)
    df = pd.DataFrame(data)

    # The abnormal row: a merged-cell-like outlier on every axis at once.
    abnormal = {
        "area": 20000.0, "extent": 0.15, "equivalent_diameter": 40.0,
        "aspect_ratio": 6.0, "true_sphericity": 0.25, "solidity": 0.40,
        "major_axis_z_angle": 85.0, "inertia_elongation": 9.0,
        "surface_area": 9000.0, "neighbor_count": 0.0,
        "nearest_neighbor_distance": 15.0, "skeleton_branches": 7.0,
        "prob_mean": 0.4, "prob_boundary_drop": 0.9,
        "prob_boundary_entropy": 0.65,
    }
    df = pd.concat([df, pd.DataFrame([abnormal])], ignore_index=True)
    df.insert(0, "label", np.arange(1, len(df) + 1))
    return df


_ABNORMAL_IDX = 30  # row index of the abnormal cell when n_normal=30


# --------------------------------------------------------------------------- #
# ported transform rules (constants + helpers)
# --------------------------------------------------------------------------- #

def test_curated_constants_populated():
    assert isinstance(qf.PCA_FEATURES, list) and len(qf.PCA_FEATURES) >= 15
    assert "area" in qf.PCA_FEATURES and "true_sphericity" in qf.PCA_FEATURES
    # bounded set carries the canonical [0,1] features; log set the size ones
    for c in ("true_sphericity", "solidity", "extent", "prob_mean"):
        assert c in qf.BOUNDED_FEATURES, c
    for c in ("area", "surface_area", "skeleton_length"):
        assert c in qf.LOG_TRANSFORM_FEATURES, c


def test_bounded_feature_not_auto_log_transformed():
    """A bounded feature (in BOUNDED_FEATURES) is excluded from auto-skew, even
    when its sample distribution is skewed -> never log1p'd."""
    n = 50
    cols = ["true_sphericity", "area"]
    # true_sphericity: a right-skewed cloud inside [0,1]; area: a skewed positive.
    sph = np.clip(np.random.RandomState(1).beta(1.2, 6.0, size=n), 0, 1)  # skewed in [0,1]
    area = np.random.RandomState(2).exponential(1000.0, size=n)           # skewed positive
    X = np.column_stack([sph, area])
    idx = qf._detect_skewed_columns(X, cols)
    # area (unbounded, skewed) is flagged; true_sphericity (bounded) is NOT.
    assert 1 in idx                 # area
    assert 0 not in idx             # true_sphericity excluded because bounded


def test_skewed_positive_is_detected():
    n = 100
    cols = ["surface_area"]
    x = np.random.RandomState(3).exponential(500.0, size=n)  # strongly right-skewed
    idx = qf._detect_skewed_columns(np.column_stack([x]), cols)
    assert idx == [0]


def test_robust_standardize_centers_and_clips():
    df = _synthetic_table()
    cols = [c for c in qf.PCA_FEATURES if c in df.columns]
    X = df[cols].values.astype(np.float64)
    Xs = qf._robust_standardize(X.copy(), cols)
    assert Xs.shape == X.shape
    assert np.all(np.isfinite(Xs))
    # winsorized to +/-10 robust z
    assert Xs.min() >= -10.0 - 1e-9 and Xs.max() <= 10.0 + 1e-9
    # the normal population is centered near 0 (median ~ 0 per column)
    normal = Xs[:_ABNORMAL_IDX]
    assert np.all(np.abs(np.median(normal, axis=0)) < 1.0)


# --------------------------------------------------------------------------- #
# compute_composite_score
# --------------------------------------------------------------------------- #

def test_qc_score_zscore_finite_and_orders_abnormal_worst():
    df = _synthetic_table()
    out = qf.compute_composite_score(df.copy(), method="zscore")
    assert "qc_score" in out.columns
    s = out["qc_score"].values
    assert np.all(np.isfinite(s))
    assert np.all(s >= 0.0)                       # mean |z| is non-negative
    # the abnormal cell is the single worst (highest) score by a clear margin.
    assert int(np.argmax(s)) == _ABNORMAL_IDX
    assert s[_ABNORMAL_IDX] > np.max(s[:_ABNORMAL_IDX]) * 1.5


def test_qc_score_pca_finite_and_flags_abnormal():
    df = _synthetic_table()
    out = qf.compute_composite_score(df.copy(), method="pca", n_pcs=5)
    s = out["qc_score"].values
    assert np.all(np.isfinite(s))
    assert np.all(s >= 0.0)                       # shifted to min 0
    assert s.min() == pytest.approx(0.0)
    # PC-space L2 distance: the abnormal row is the farthest from the cloud.
    assert int(np.argmax(s)) == _ABNORMAL_IDX


def test_qc_score_pca_reads_persisted_pc_columns():
    """compute_pca_columns writes PC1..PCN; compute_composite_score(method='pca')
    then reads them (the fast path) -> the same shift-to-zero-min L2 score as the
    inline SVD would give for the same projection."""
    df = _synthetic_table()
    qf.compute_pca_columns(df, n_components=5)
    pc_cols = [c for c in df.columns if c.startswith("PC")]
    assert pc_cols, "compute_pca_columns wrote no PC columns"
    out = qf.compute_composite_score(df.copy(), method="pca", n_pcs=5)
    s = out["qc_score"].values
    # identity check: score == L2 norm of the persisted PCs, shifted to min 0.
    expected = np.linalg.norm(df[[f"PC{i+1}" for i in range(5)]].values, axis=1)
    expected = expected - expected.min()
    assert np.allclose(s, expected)
    assert int(np.argmax(s)) == _ABNORMAL_IDX


def test_qc_score_unknown_method_raises():
    df = _synthetic_table()
    with pytest.raises(ValueError):
        qf.compute_composite_score(df, method="bogus")


def test_qc_score_degenerate_inputs_score_zero():
    # <2 rows -> 0.0
    one = pd.DataFrame({"label": [1], "area": [100.0]})
    qf.compute_composite_score(one, method="zscore")
    assert one["qc_score"].tolist() == [0.0]
    # no usable curated columns -> 0.0
    nocols = pd.DataFrame({"label": [1, 2, 3], "not_a_feature": [1.0, 2.0, 3.0]})
    qf.compute_composite_score(nocols, method="zscore")
    assert nocols["qc_score"].tolist() == [0.0, 0.0, 0.0]


# --------------------------------------------------------------------------- #
# manifest + families registry
# --------------------------------------------------------------------------- #

def test_manifest_tags_qc_score():
    man = fm.get_manifest()
    assert "qc_score" in man
    tags = man["qc_score"]
    for key in ("units", "bounded", "role", "frame", "redundancy_group"):
        assert key in tags
    assert tags["role"] == "qc"
    assert tags["frame"] == "intrinsic"
    assert tags["bounded"] is False           # an unbounded >=0 outlier score
    assert tags["redundancy_group"] == "composite_score"


def test_feature_stage_families_registry_includes_qc_score():
    from packages.epicycle_scaledseg.core.feature_stage import _FEATURE_FAMILIES
    assert _FEATURE_FAMILIES["composite_score"] == ["qc_score"]


# --------------------------------------------------------------------------- #
# opt-in global pass wired into extract_label_features
# --------------------------------------------------------------------------- #

def _toy_labels():
    """Cells 1 & 2 touch at x==8; cell 3 is isolated (gap >> max_radius=5)."""
    vol = np.zeros((8, 16, 36), np.int32)
    vol[1:6, 2:8, 2:8] = 1
    vol[1:6, 2:8, 8:14] = 2          # touches cell 1 at x==8
    vol[1:6, 2:8, 28:34] = 3         # isolated
    return vol


def test_extract_label_features_qc_score_off_byte_identical(tmp_path):
    """``extract_label_features(qc_score=False)`` is byte-identical to a call
    WITHOUT the new param — the M7 additive-optional contract (both on the
    labels-only path and on the adjacency path)."""
    from packages.epicycle_scaledseg.reconcile.features import extract_label_features

    vol = _toy_labels()
    p = tmp_path / "labels.npy"
    np.save(p, vol)

    # labels-only path
    a = extract_label_features(str(p), n_workers=1)
    b = extract_label_features(str(p), n_workers=1, qc_score=False)
    assert list(a.columns) == list(b.columns)
    pd.testing.assert_frame_equal(a, b)
    assert "qc_score" not in a.columns

    # adjacency path
    c = extract_label_features(str(p), n_workers=1, adjacency=True)
    d = extract_label_features(str(p), n_workers=1, adjacency=True, qc_score=False)
    assert list(c.columns) == list(d.columns)
    pd.testing.assert_frame_equal(c, d)
    assert "qc_score" not in c.columns


def test_extract_label_features_qc_score_on_adds_only_qc_score(tmp_path):
    """``adjacency=True, qc_score=True`` adds EXACTLY the qc_score column on top of
    the adjacency output; all other columns/values are unchanged."""
    from packages.epicycle_scaledseg.reconcile.features import extract_label_features

    vol = _toy_labels()
    p = tmp_path / "labels.npy"
    np.save(p, vol)

    base = extract_label_features(str(p), n_workers=1, adjacency=True, max_radius=5)
    scored = extract_label_features(
        str(p), n_workers=1, adjacency=True, max_radius=5, qc_score=True)

    assert "qc_score" not in base.columns
    assert "qc_score" in scored.columns
    # exactly one new column
    assert set(scored.columns) - set(base.columns) == {"qc_score"}
    # the pre-existing columns are byte-identical (qc_score only APPENDS).
    shared = [c for c in base.columns if c in scored.columns]
    pd.testing.assert_frame_equal(
        scored[shared].reset_index(drop=True),
        base[shared].reset_index(drop=True),
    )
    assert np.all(np.isfinite(scored["qc_score"].values))
    assert np.all(scored["qc_score"].values >= 0.0)


def test_extract_label_features_qc_score_without_adjacency_warns(tmp_path, caplog):
    """``qc_score=True, adjacency=False`` still scores on the labels-only subset
    (no neighbour columns) and logs a one-line warning rather than blocking."""
    from packages.epicycle_scaledseg.reconcile.features import extract_label_features

    vol = _toy_labels()
    p = tmp_path / "labels.npy"
    np.save(p, vol)

    with caplog.at_level(logging.WARNING):
        df = extract_label_features(str(p), n_workers=1, qc_score=True)
    assert "qc_score" in df.columns
    assert np.all(np.isfinite(df["qc_score"].values))
    # neighbour columns were NOT produced (adjacency off)
    assert "neighbor_count" not in df.columns
    # the warning fired
    assert any("qc_score requested with adjacency=False" in r.message
               for r in caplog.records)
