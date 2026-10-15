"""Manifest-completeness regression — every labels-only feature column is tagged.

The FEATURE_MANIFEST (``epicycle.core.feature_manifest``) tags each extracted
feature column with ``{units, bounded, role, frame, redundancy_group}`` so the
EDA layer can size-normalize / log-transform / decorrelate per dataset. The
baseline M0/M1/M5-era intrinsic morphology columns were historically NOT
backfilled (a TODO in feature_manifest.py); this test pins the gap CLOSED so it
cannot silently reopen as new baseline columns are added.

SCOPE: the **labels-only / default** path of
:func:`epicycle.reconcile.features.extract_label_features` — i.e. the per-cell
intrinsic morphology with NO optional inputs (no channels / prob / adjacency /
tissue_depth / lumen / territory / qc_score). That is the column set every run
produces unconditionally.

* ``label`` / ``cell_id`` (ids) and ``centroid-*`` / ``bbox-*`` (raw regionprops
  COORDINATE metadata) are NOT features and are excluded.
* OPT-IN / DYNAMIC columns (per-channel intensity/texture/localization/puncta,
  channel-pair co-localization, local-comparative, and the adjacency /
  tissue-depth / qc_score GLOBAL passes) are tagged via the ``*_tags`` helper
  TEMPLATES, not static keys — their prefix / base-feature set is a runtime
  choice, so they can't be statically enumerated. Each is covered by its own
  wave test (``test_feature_waveA``, ``test_feature_m2``, …). This test
  deliberately exercises only the always-on labels-only set to stay robust.
"""

import warnings

import numpy as np
import pytest

from packages.epicycle_scaledseg.core import feature_manifest as fm
from packages.epicycle_scaledseg.reconcile.features import extract_label_features


# Ids + raw regionprops positional/index metadata — coordinate systems, NOT
# phenotype/QC features (treated like ``label`` / ``cell_id`` per the manifest
# header). Deliberately un-tagged + excluded from the completeness assertion.
_ID_LIKE = {"label", "cell_id"}
_POSITIONAL = {f"centroid-{i}" for i in range(3)} | {f"bbox-{i}" for i in range(6)}
_EXCLUDED = _ID_LIKE | _POSITIONAL


def _multi_cell_volume():
    """A 3-cell volume exercising every labels-only family.

    A prism, a sphere, and a rod — large enough that surface mesh / skeleton /
    cross-section / curvature / shell passes all compute finite values.
    """
    lab = np.zeros((40, 60, 60), dtype=np.int32)
    lab[5:30, 5:20, 5:20] = 1                                   # prism
    zz, yy, xx = np.ogrid[:40, :60, :60]
    lab[(zz - 20) ** 2 + (yy - 40) ** 2 + (xx - 40) ** 2 <= 8 ** 2] = 2   # sphere
    lab[10:35, 30:34, 5:30] = 3                                 # rod
    return lab


def _labels_only_columns():
    lab = _multi_cell_volume()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                        # pandas fragmentation noise
        df = extract_label_features(
            lab, include_surface=True, progress=False, n_workers=1,
        )
    return list(df.columns)


def _untagged_feature_columns(columns):
    """Feature columns (excl. ids/positional) with NO static manifest tag."""
    man = fm.get_manifest()
    return [c for c in columns if c not in _EXCLUDED and c not in man]


# --------------------------------------------------------------------------- #
# the regression: zero untagged baseline columns
# --------------------------------------------------------------------------- #

def test_every_labels_only_feature_is_tagged():
    """No labels-only feature column may lack a manifest tag (the gap stays closed)."""
    cols = _labels_only_columns()
    untagged = _untagged_feature_columns(cols)
    assert untagged == [], (
        f"{len(untagged)} labels-only feature column(s) missing a FEATURE_MANIFEST "
        f"tag — add them to feature_manifest.py: {untagged}"
    )


def test_labels_only_set_is_non_trivial():
    """Guard the guard: the labels-only run actually emits the baseline columns.

    A silent change that drops the extraction (empty df) would make the
    completeness test vacuously pass; assert a representative sample is present.
    """
    cols = set(_labels_only_columns())
    for c in ("area", "sphericity", "surface_area", "skeleton_branches",
              "apical_area", "axis_major_length", "euler_number",
              "major_axis_z", "inertia_eigval_major"):
        assert c in cols, c


# --------------------------------------------------------------------------- #
# the tags themselves are well-formed for the baseline columns
# --------------------------------------------------------------------------- #

_VALID_UNITS = {"voxel", "um", "ratio", "count", "score", "intensity",
                "angle", "degrees", "radians"}


def test_baseline_tags_are_well_formed():
    """Every tagged labels-only column carries a complete, valid tag dict."""
    man = fm.get_manifest()
    cols = [c for c in _labels_only_columns() if c not in _EXCLUDED]
    for c in cols:
        assert c in man, c
        tags = man[c]
        for key in ("units", "bounded", "role", "frame", "redundancy_group"):
            assert key in tags, (c, key)
        assert tags["units"] in _VALID_UNITS, (c, tags["units"])
        assert isinstance(tags["bounded"], bool), c
        assert tags["role"] in ("biology", "qc", "covariate"), (c, tags["role"])
        assert tags["frame"] in ("intrinsic", "image", "tissue", "local"), (c, tags["frame"])


def test_baseline_bounded_flags_match_feature_math():
    """Spot-check bounded flags against the monorepo BOUNDED_FEATURES convention."""
    man = fm.get_manifest()
    # Bounded [0,1] shape ratios (mirror BOUNDED_FEATURES).
    for c in ("sphericity", "true_sphericity", "solidity", "extent",
              "fill_ratio", "ellipsoid_solidity", "discrete_compactness",
              "inertia_flatness", "max_cross_section_z_relative"):
        assert man[c]["bounded"] is True, c
    # Unbounded size / ratios / counts.
    for c in ("area", "surface_area", "axis_major_length", "aspect_ratio",
              "roughness", "euler_number", "skeleton_branches",
              "apical_basal_area_ratio", "inertia_elongation"):
        assert man[c]["bounded"] is False, c


def test_topology_quality_proxies_are_qc_role():
    """euler_number / skeleton_* / roughness are QC proxies, not phenotype biology."""
    man = fm.get_manifest()
    for c in ("euler_number", "skeleton_branches", "skeleton_nodes",
              "skeleton_length", "roughness"):
        assert man[c]["role"] == "qc", c
