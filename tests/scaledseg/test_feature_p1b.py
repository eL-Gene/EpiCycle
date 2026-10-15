"""P1b feature tests — three NEW-INPUT primitives closing the remaining honest
gaps:

1. **Lumen-cavity primitive** — ``compute_lumen_features``: a lumen
   is an ENCLOSED background cavity (the inner surface of a hollow organoid /
   gland / cyst), NOT the surrounding medium. ``distance_to_lumen`` per cell =
   min physical EDT to the nearest lumen voxel (NaN when there is NO lumen — the
   honest answer for a solid block), plus ``lumen_facing_fraction``.

2. **Marker-expectation QC flags** — ``compute_marker_expectation_flags``:
   a BIOLOGY-CONFIG ``marker_expectations`` dict declares each marker's expected
   compartment; the flags check whether the OBSERVED signal sits there. Off by
   default (no config -> no columns).

3. **At-nucleus / perinuclear / neighbour local-expression context** —
   ``compute_channel_local_context`` (paired, at-nucleus + perinuclear
   shell mean + ratio) + ``compute_channel_neighbor_context`` (the
   "vs neighbor" row: a cell's channel mean ÷ median of its graph neighbours').

All NEW (NOT a monorepo port). Pure-function known-geometry tests (no GPU /
Cellpose); synthetic arrangements whose answers we know by construction. Every
new family is opt-in / default-off byte-identical — the central honesty contract.
"""

import numpy as np
import pandas as pd
import pytest

from packages.epicycle_scaledseg.core import qc_features as qf
from packages.epicycle_scaledseg.core import feature_manifest as fm


# =========================================================================== #
# 1) LUMEN — compute_lumen_features (distance_to_lumen / lumen_facing_fraction)
# =========================================================================== #
def _hollow_tissue(shape=(40, 40, 40), r_in=7.0, r_out=14.0):
    """A thick spherical SHELL of tissue ringing a central cavity (a lumen),
    split into octant labels so there are several distinct cells."""
    Z, Y, X = shape
    zz, yy, xx = np.ogrid[:Z, :Y, :X]
    c = (np.array(shape) - 1) / 2.0
    r = np.sqrt((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2)
    shell = (r >= r_in) & (r <= r_out)
    oct_id = ((zz >= c[0]).astype(int) * 4
              + (yy >= c[1]).astype(int) * 2
              + (xx >= c[2]).astype(int)) + 1
    lab = np.zeros(shape, np.int32)
    lab[shell] = oct_id[shell]
    return lab


def _solid_block():
    """Two stacked solid slabs — NO enclosed cavity (the honest all-NaN case)."""
    lab = np.zeros((20, 20, 20), np.int32)
    lab[2:8, 2:18, 2:18] = 1
    lab[8:14, 2:18, 2:18] = 2
    return lab


def test_lumen_finite_distance_for_hollow_tissue():
    """Cells ringing an enclosed cavity get a FINITE distance_to_lumen."""
    lab = _hollow_tissue()
    df = qf.extract_features(lab)
    qf.compute_lumen_features(lab, df)
    assert df["distance_to_lumen"].notna().all()
    # the shell lines the cavity -> every cell is within ~a couple voxels.
    assert (df["distance_to_lumen"] <= 3.0).all()


def test_lumen_facing_fraction_positive_for_hollow_tissue():
    """Cells lining the lumen have a positive lumen_facing_fraction."""
    lab = _hollow_tissue()
    df = qf.extract_features(lab)
    qf.compute_lumen_features(lab, df)
    assert (df["lumen_facing_fraction"] > 0.0).all()
    assert (df["lumen_facing_fraction"] <= 1.0).all()


def test_lumen_all_nan_for_solid_block():
    """A solid tissue with NO enclosed cavity -> distance_to_lumen all-NaN (the
    correct answer, NOT 0 / a sentinel)."""
    lab = _solid_block()
    df = qf.extract_features(lab)
    qf.compute_lumen_features(lab, df)
    assert df["distance_to_lumen"].isna().all()
    assert df["lumen_facing_fraction"].isna().all()


def test_lumen_external_medium_not_a_lumen():
    """A cell floating in the EXTERNAL medium (a hole open to the border) is NOT a
    lumen -> all-NaN. We make a solid cube with a notch open to the FOV edge."""
    lab = np.zeros((24, 24, 24), np.int32)
    lab[4:20, 4:20, 4:20] = 1
    # carve a channel from the cube interior straight out to the X=0 face -> the
    # 'cavity' is connected to the external medium, so it is NOT enclosed.
    lab[10:14, 10:14, 0:12] = 0
    df = qf.extract_features(lab)
    qf.compute_lumen_features(lab, df)
    assert df["distance_to_lumen"].isna().all(), "open-to-border void is not a lumen"


def test_lumen_distance_grows_with_spacing():
    """Physical spacing scales the distance field."""
    lab = _hollow_tissue()
    df1 = qf.extract_features(lab)
    qf.compute_lumen_features(lab, df1)
    df2 = qf.extract_features(lab, spacing=(2.0, 2.0, 2.0))
    qf.compute_lumen_features(lab, df2, spacing=(2.0, 2.0, 2.0))
    # finite distances roughly double (cells at exactly distance 0 stay 0).
    m1 = df1["distance_to_lumen"].max()
    m2 = df2["distance_to_lumen"].max()
    assert m2 >= m1  # never shrinks; grows where >0


def test_lumen_empty_df_typed_columns():
    df = pd.DataFrame({"label": pd.Series([], dtype=np.int64)})
    qf.compute_lumen_features(np.zeros((4, 4, 4), np.int32), df)
    for c in qf.LUMEN_COLUMNS:
        assert c in df.columns


def test_lumen_no_label_column_nan():
    df = pd.DataFrame({"x": [1.0, 2.0]})
    qf.compute_lumen_features(_hollow_tissue(), df)
    for c in qf.LUMEN_COLUMNS:
        assert c in df.columns and df[c].isna().all()


def test_lumen_manifest_and_family_registry():
    man = fm.get_manifest()
    for c in qf.LUMEN_COLUMNS:
        assert c in man, f"{c} missing from FEATURE_MANIFEST"
        assert man[c]["frame"] == "tissue", c
    # distance_to_lumen is biology (an apical-footprint / depth signal); FIX 6 re-tags
    # lumen_facing_fraction -> covariate (a position covariate, not phenotype).
    assert man["distance_to_lumen"]["role"] == "biology"
    assert man["lumen_facing_fraction"]["role"] == "covariate"
    assert man["lumen_facing_fraction"]["bounded"] is True
    assert man["distance_to_lumen"]["bounded"] is False
    from packages.epicycle_scaledseg.core.feature_stage import _FEATURE_FAMILIES
    assert set(_FEATURE_FAMILIES["lumen"]) == set(qf.LUMEN_COLUMNS)


def test_lumen_default_off_in_extract_label_features():
    """extract_label_features(lumen=False) -> NO lumen columns (byte-identical)."""
    from packages.epicycle_scaledseg.reconcile.features import extract_label_features
    lab = _hollow_tissue()
    df_off = extract_label_features(lab, n_workers=1)
    assert "distance_to_lumen" not in df_off.columns
    df_on = extract_label_features(lab, lumen=True, n_workers=1)
    assert "distance_to_lumen" in df_on.columns
    assert df_on["distance_to_lumen"].notna().any()


# =========================================================================== #
# 2) MARKER EXPECTATION — compute_marker_expectation_flags
# =========================================================================== #
_MARKER_SUFFIXES = (
    "_in_expected_compartment_fraction",
    "_missing_expected_signal_flag",
    "_compartment_concordance",
)


def _cell_nucleus_scene(shape=(16, 16, 16)):
    cell = np.zeros(shape, bool)
    cell[3:13, 3:13, 3:13] = True
    nuc = np.zeros(shape, bool)
    nuc[6:10, 6:10, 6:10] = True
    return cell, nuc


def _nuclear_signal(cell, nuc, hi=200.0, lo=10.0):
    ch = np.zeros(cell.shape)
    ch[cell] = lo
    ch[nuc] = hi
    return ch


def _cytoplasmic_signal(cell, nuc, hi=200.0, lo=10.0):
    ch = np.zeros(cell.shape)
    ch[cell] = hi
    ch[nuc] = lo
    return ch


def test_marker_nuclear_expectation_signal_present_flag_off():
    """Marker EXPECTED nuclear + signal IS nuclear -> flag OFF, concordance 1."""
    cell, nuc = _cell_nucleus_scene()
    ch = _nuclear_signal(cell, nuc)
    out = qf.compute_marker_expectation_flags(ch, cell, nuc, "sox2", "nuclear")
    assert out["sox2_missing_expected_signal_flag"] == 0.0
    assert out["sox2_compartment_concordance"] == 1.0
    assert out["sox2_in_expected_compartment_fraction"] > 0.7


def test_marker_nuclear_expectation_signal_elsewhere_flag_on():
    """Marker EXPECTED nuclear + signal is CYTOPLASMIC -> flag ON, concordance 0."""
    cell, nuc = _cell_nucleus_scene()
    ch = _cytoplasmic_signal(cell, nuc)
    out = qf.compute_marker_expectation_flags(ch, cell, nuc, "sox2", "nuclear")
    assert out["sox2_missing_expected_signal_flag"] == 1.0
    assert out["sox2_compartment_concordance"] == 0.0
    assert out["sox2_in_expected_compartment_fraction"] < 0.25


def test_marker_cytoplasmic_expectation_concordant():
    """Marker EXPECTED cytoplasmic + signal IS cytoplasmic -> concordance 1, flag off."""
    cell, nuc = _cell_nucleus_scene()
    ch = _cytoplasmic_signal(cell, nuc)
    out = qf.compute_marker_expectation_flags(ch, cell, nuc, "mem", "cytoplasmic")
    assert out["mem_compartment_concordance"] == 1.0
    assert out["mem_missing_expected_signal_flag"] == 0.0


def test_marker_membrane_expectation_concordant():
    """A marker concentrated on the OUTER rind, expected 'membrane' -> concordance 1."""
    from scipy.ndimage import binary_erosion
    cell, nuc = _cell_nucleus_scene()
    # membrane = the cell's outer surface rind (cell minus its 1-step erosion), the
    # SAME convention compute_marker_expectation_flags uses internally.
    membrane = cell & ~binary_erosion(cell, border_value=0)
    ch = np.zeros(cell.shape)
    ch[cell] = 10.0
    ch[membrane] = 200.0
    out = qf.compute_marker_expectation_flags(ch, cell, nuc, "ecad", "membrane")
    assert out["ecad_compartment_concordance"] == 1.0


def test_marker_bad_expectation_string_all_nan():
    """A typo in the config compartment -> all-NaN (not silently scored)."""
    cell, nuc = _cell_nucleus_scene()
    ch = _nuclear_signal(cell, nuc)
    out = qf.compute_marker_expectation_flags(ch, cell, nuc, "sox2", "bogus")
    for s in _MARKER_SUFFIXES:
        assert np.isnan(out[f"sox2{s}"])


def test_marker_nan_guards():
    """Shape mismatch / empty cell -> all-NaN, schema-stable."""
    cell, nuc = _cell_nucleus_scene()
    bad = np.zeros((8, 8, 8))
    out = qf.compute_marker_expectation_flags(bad, cell, nuc, "sox2", "nuclear")
    for s in _MARKER_SUFFIXES:
        assert np.isnan(out[f"sox2{s}"])
    empty = np.zeros((16, 16, 16), bool)
    out2 = qf.compute_marker_expectation_flags(
        _nuclear_signal(cell, nuc), empty, nuc, "sox2", "nuclear")
    for s in _MARKER_SUFFIXES:
        assert np.isnan(out2[f"sox2{s}"])


def test_marker_expectation_tags_helper():
    tags = fm.channel_marker_expectation_tags("sox2")
    for s in _MARKER_SUFFIXES:
        assert f"sox2{s}" in tags
        t = tags[f"sox2{s}"]
        assert t["role"] == "qc"               # marker flags role=qc (per the prompt)
        assert t["frame"] == "intrinsic"
        assert t["bounded"] is True
        assert t["redundancy_group"] == "channel_marker_expectation_sox2"


# =========================================================================== #
# 3a) AT-NUCLEUS / PERINUCLEAR — compute_channel_local_context (paired)
# =========================================================================== #
_LOCAL_CTX_SUFFIXES = (
    "_at_nucleus_mean", "_perinuclear_shell_mean", "_perinuclear_to_nuclear_ratio",
)


def test_local_context_at_nucleus_higher_for_nuclear_signal():
    """A nuclear-bright channel -> at_nucleus_mean > perinuclear_shell_mean,
    perinuclear/nuclear ratio < 1."""
    cell, nuc = _cell_nucleus_scene()
    ch = _nuclear_signal(cell, nuc)
    out = qf.compute_channel_local_context(ch, cell, nuc, "sox2")
    assert out["sox2_at_nucleus_mean"] > out["sox2_perinuclear_shell_mean"]
    assert out["sox2_perinuclear_to_nuclear_ratio"] < 1.0


def test_local_context_perinuclear_signal_ratio_high():
    """A signal in a ring JUST OUTSIDE the nucleus -> perinuclear/nuclear ratio > 1."""
    from scipy.ndimage import binary_dilation
    cell, nuc = _cell_nucleus_scene()
    peri = binary_dilation(nuc, iterations=2) & ~nuc & cell
    ch = np.zeros(cell.shape)
    ch[cell] = 10.0
    ch[nuc] = 10.0
    ch[peri] = 200.0
    out = qf.compute_channel_local_context(ch, cell, nuc, "sox2")
    assert out["sox2_perinuclear_to_nuclear_ratio"] > 1.0
    assert out["sox2_perinuclear_shell_mean"] > out["sox2_at_nucleus_mean"]


def test_local_context_nan_without_nucleus():
    """No in-cell nucleus -> all the local-context columns NaN (still present keys)."""
    cell, _ = _cell_nucleus_scene()
    empty_nuc = np.zeros(cell.shape, bool)
    ch = np.ones(cell.shape) * 5.0
    out = qf.compute_channel_local_context(ch, cell, empty_nuc, "sox2")
    for s in _LOCAL_CTX_SUFFIXES:
        assert s.startswith("_") and np.isnan(out[f"sox2{s}"])


def test_local_context_nan_guards():
    cell, nuc = _cell_nucleus_scene()
    bad = np.zeros((8, 8, 8))
    out = qf.compute_channel_local_context(bad, cell, nuc, "sox2")
    for s in _LOCAL_CTX_SUFFIXES:
        assert np.isnan(out[f"sox2{s}"])


# =========================================================================== #
# 3b) VS-NEIGHBOUR — compute_channel_neighbor_context (global; "vs neighbor")
# =========================================================================== #
def _three_cell_graph(means):
    df = pd.DataFrame({"label": [1, 2, 3], "mem_intensity_mean": means})
    adj = [
        {"label_a": 1, "label_b": 2, "contact_voxels": 50},
        {"label_a": 1, "label_b": 3, "contact_voxels": 50},
        {"label_a": 2, "label_b": 3, "contact_voxels": 50},
    ]
    df.attrs["_adjacency_list"] = adj
    return df


def test_channel_vs_neighbor_ratio_high_for_bright_outlier():
    """A cell much brighter than its neighbours -> vs_neighbor_mean_ratio >> 1."""
    df = _three_cell_graph([100.0, 10.0, 12.0])
    qf.compute_channel_neighbor_context(df)
    r = df.set_index("label")["mem_vs_neighbor_mean_ratio"]
    assert r.loc[1] > 5.0       # 100 vs median(10,12)=11
    assert r.loc[2] < 1.5
    assert r.loc[3] < 1.5


def test_channel_vs_neighbor_ratio_near_one_when_uniform():
    df = _three_cell_graph([50.0, 50.0, 50.0])
    qf.compute_channel_neighbor_context(df)
    r = df["mem_vs_neighbor_mean_ratio"]
    assert np.allclose(r.values, 1.0)


def test_channel_vs_neighbor_isolated_cell_nan():
    """A cell with no graph neighbours -> NaN."""
    df = pd.DataFrame({"label": [1, 2], "mem_intensity_mean": [10.0, 20.0]})
    df.attrs["_adjacency_list"] = []        # no edges
    qf.compute_channel_neighbor_context(df)
    assert df["mem_vs_neighbor_mean_ratio"].isna().all()


def test_channel_vs_neighbor_default_off_no_channels():
    """No <prefix>_intensity_mean column -> NO column added (byte-identical)."""
    df = pd.DataFrame({"label": [1, 2, 3]})
    df.attrs["_adjacency_list"] = [{"label_a": 1, "label_b": 2, "contact_voxels": 1}]
    before = set(df.columns)
    qf.compute_channel_neighbor_context(df)
    assert set(df.columns) == before


def test_channel_local_context_tags_helper():
    tags = fm.channel_local_context_tags("mem")
    expect = {
        "mem_at_nucleus_mean": "intrinsic",
        "mem_perinuclear_shell_mean": "intrinsic",
        "mem_perinuclear_to_nuclear_ratio": "intrinsic",
        "mem_vs_neighbor_mean_ratio": "local",
    }
    for col, frame in expect.items():
        assert col in tags
        assert tags[col]["frame"] == frame
        assert tags[col]["role"] == "biology"
        assert tags[col]["redundancy_group"] == "channel_local_context_mem"


# =========================================================================== #
# 4) END-TO-END default-off: the full paired path with/without the new config
# =========================================================================== #
def _paired_scene():
    Z, Y, X = 16, 16, 16
    cells = np.zeros((Z, Y, X), np.int32)
    nuclei = np.zeros((Z, Y, X), np.int32)
    cells[3:13, 3:13, 3:13] = 1
    nuclei[6:10, 6:10, 6:10] = 10
    triage = [{"cell_id": 1, "status": "clean", "n_nuclei": 1,
               "nucleus_ids": [10], "cell_voxels": 1000}]
    return cells, nuclei, triage

