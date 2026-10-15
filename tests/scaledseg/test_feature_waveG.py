"""Wave G feature tests — watershed territory / crowding (atlas ) + HCR puncta
/ transcript spatial (atlas ).

Pure-function known-shape tests (no tile readers / GPU).

* :func:`qc_features.compute_watershed_territory` — GLOBAL pass; a
  background-EDT seeded watershed expands every cell into the inter-cell space up to
  the midline with its neighbours (capped at a max distance), giving each cell a
  Voronoi-ish TERRITORY. We assert: two cells with empty space between them get a
  territory LARGER than the cell + a sensible ``territory_to_cell_ratio`` > 1; a
  tightly-packed cell (neighbours on every side) gets ratio ~1.

* :func:`qc_features.compute_channel_puncta` — per-channel dict-return; a
  blob_log spot detector on the masked channel crop. We assert: a crop with N bright
  spots -> ``puncta_count`` ≈ N + high ``punctate_fraction``; a smooth / diffuse crop
  -> count 0 + low fraction; spot clustering is reflected in ``puncta_nn_distance``.

Plus: NaN / 0 guards on every path; the manifest tags every new column (static
territory entries + the per-channel ``channel_puncta_tags`` helper); the
FeatureStage families registry + per-channel recovery; the FeatureStage wiring emits
the per-channel puncta columns when channels are sampled (labels-only stays
byte-identical) + the ``extract_label_features(territory=True)`` global pass emits the
territory columns (default-off byte-identical).
"""

import numpy as np
import pandas as pd
import pytest

from packages.epicycle_scaledseg.core import qc_features as qf
from packages.epicycle_scaledseg.core import feature_manifest as fm


# --------------------------------------------------------------------------- #
# column templates
# --------------------------------------------------------------------------- #
# Wave G built the first 4 territory columns; Mop-up M (remainder) extended
# the family. The exact-set family test reads the module constant so it tracks the
# growth (the 4 Wave-G columns asserted explicitly elsewhere stay a subset).
_TERRITORY_COLS = qf.WATERSHED_TERRITORY_COLUMNS
_TERRITORY_COLS_WAVEG = (
    "ws_territory_volume", "territory_to_cell_ratio",
    "territory_anisotropy", "territory_solidity",
)
_PUNCTA_SUFFIXES = (
    "_puncta_count", "_puncta_density", "_puncta_intensity_mean",
    "_puncta_mean_volume", "_puncta_nn_distance", "_punctate_fraction",
    # Wave K extended compute_channel_puncta with the spatial/distribution remainder
    #: radial position / clustering index / polarity / intensity cv /
    # dispersion. They ride the SAME dict, so the column-count assertions cover them.
    "_puncta_radial_position_mean", "_puncta_clustering_index",
    "_puncta_polarity", "_puncta_intensity_cv", "_puncta_dispersion",
)


# --------------------------------------------------------------------------- #
# — compute_watershed_territory
# --------------------------------------------------------------------------- #
def _two_separated_cells(shape=(20, 20, 40)):
    """Two equal cubes (id 1, id 2) with a wide empty gap between them, well inside
    the bounded volume."""
    lab = np.zeros(shape, np.int32)
    lab[6:14, 6:14, 4:12] = 1
    lab[6:14, 6:14, 28:36] = 2
    return lab


def _packed_grid(shape=(12, 12, 12), block=4):
    """A volume FILLED edge-to-edge with cube cells (every cell hemmed in on all
    in-plane sides by a neighbour -> no free space to expand into)."""
    lab = np.zeros(shape, np.int32)
    nid = 1
    for y in range(0, shape[1], block):
        for x in range(0, shape[2], block):
            lab[:, y:y + block, x:x + block] = nid
            nid += 1
    return lab


def test_territory_columns_present():
    lab = _two_separated_cells()
    df = pd.DataFrame({"label": [1, 2]})
    qf.compute_watershed_territory(lab, df)
    for c in _TERRITORY_COLS:
        assert c in df.columns, c


def test_territory_larger_than_cell_for_separated_cells():
    """Two cells with empty space between them: each territory > its cell volume and
    territory_to_cell_ratio > 1 (room to expand into the gap)."""
    lab = _two_separated_cells()
    df = pd.DataFrame({"label": [1, 2]})
    qf.compute_watershed_territory(lab, df, max_distance_voxels=15.0)
    cell_vol = float((lab == 1).sum())
    for idx in (0, 1):
        assert df.loc[idx, "ws_territory_volume"] > cell_vol
        assert df.loc[idx, "territory_to_cell_ratio"] > 1.0
    # both cells are equal + symmetric -> equal territories.
    assert df.loc[0, "ws_territory_volume"] == pytest.approx(
        df.loc[1, "ws_territory_volume"])


def test_territory_ratio_near_one_for_packed_cells():
    """A tightly-packed grid (neighbours on every in-plane side) -> each cell can't
    expand -> territory_to_cell_ratio ~ 1."""
    lab = _packed_grid()
    ids = sorted(int(i) for i in np.unique(lab) if i > 0)
    df = pd.DataFrame({"label": ids})
    qf.compute_watershed_territory(lab, df)
    # every cell is boxed in by neighbours -> ratio exactly ~1.
    assert df["territory_to_cell_ratio"].max() == pytest.approx(1.0, abs=1e-6)


def test_territory_ratio_orders_sparse_above_packed():
    """The separated (sparse) cell has a much larger territory ratio than a packed
    cell — the crowding signal orders correctly."""
    sparse = _two_separated_cells()
    df_s = pd.DataFrame({"label": [1, 2]})
    qf.compute_watershed_territory(sparse, df_s, max_distance_voxels=15.0)

    packed = _packed_grid()
    ids = sorted(int(i) for i in np.unique(packed) if i > 0)
    df_p = pd.DataFrame({"label": ids})
    qf.compute_watershed_territory(packed, df_p)

    assert df_s["territory_to_cell_ratio"].min() > df_p["territory_to_cell_ratio"].max()


def test_territory_cap_bounds_isolated_cell():
    """A LARGER cap lets an isolated cell claim MORE background -> a larger
    territory. Confirms the documented expansion cap actually bounds the flood."""
    # one lone cell in a big empty volume.
    lab = np.zeros((30, 30, 30), np.int32)
    lab[13:17, 13:17, 13:17] = 1
    df_small = pd.DataFrame({"label": [1]})
    qf.compute_watershed_territory(lab, df_small, max_distance_voxels=4.0)
    df_big = pd.DataFrame({"label": [1]})
    qf.compute_watershed_territory(lab, df_big, max_distance_voxels=10.0)
    assert df_big.loc[0, "ws_territory_volume"] > df_small.loc[0, "ws_territory_volume"]


def test_territory_anisotropy_and_solidity_bounds():
    """A compact cube territory is ~isotropic (anisotropy ~1) and convex
    (solidity ~1)."""
    lab = _two_separated_cells()
    df = pd.DataFrame({"label": [1, 2]})
    qf.compute_watershed_territory(lab, df, max_distance_voxels=15.0)
    assert df.loc[0, "territory_anisotropy"] >= 1.0 - 1e-6
    assert 0.0 < df.loc[0, "territory_solidity"] <= 1.0 + 1e-9


def test_territory_spacing_scales_volume():
    """Physical spacing scales ws_territory_volume by the voxel volume; the ratio
    (dimensionless) is unchanged."""
    lab = _two_separated_cells()
    df_v = pd.DataFrame({"label": [1, 2]})
    qf.compute_watershed_territory(lab, df_v, spacing=None, max_distance_voxels=15.0)
    df_p = pd.DataFrame({"label": [1, 2]})
    qf.compute_watershed_territory(lab, df_p, spacing=(2.0, 1.0, 1.0),
                                   max_distance_voxels=15.0)
    assert df_p.loc[0, "ws_territory_volume"] == pytest.approx(
        2.0 * df_v.loc[0, "ws_territory_volume"])
    # ratio is dimensionless -> identical.
    assert df_p.loc[0, "territory_to_cell_ratio"] == pytest.approx(
        df_v.loc[0, "territory_to_cell_ratio"])


def test_territory_nan_guards():
    # empty table
    df = pd.DataFrame({"label": []})
    qf.compute_watershed_territory(np.zeros((6, 6, 6), np.int32), df)
    for c in _TERRITORY_COLS:
        assert c in df.columns
    # no label column
    df2 = pd.DataFrame({"x": [1]})
    qf.compute_watershed_territory(np.zeros((6, 6, 6), np.int32), df2)
    for c in _TERRITORY_COLS:
        assert c in df2.columns
        assert np.isnan(df2.loc[0, c])
    # no foreground at all
    df3 = pd.DataFrame({"label": [1]})
    qf.compute_watershed_territory(np.zeros((6, 6, 6), np.int32), df3)
    for c in _TERRITORY_COLS:
        assert np.isnan(df3.loc[0, c])
    # a label not present in the volume -> NaN row (graceful skip).
    lab = _two_separated_cells()
    df4 = pd.DataFrame({"label": [1, 2, 99]})
    qf.compute_watershed_territory(lab, df4, max_distance_voxels=15.0)
    assert np.isnan(df4.loc[2, "ws_territory_volume"])
    assert np.isfinite(df4.loc[0, "ws_territory_volume"])


# --------------------------------------------------------------------------- #
# — compute_channel_puncta
# --------------------------------------------------------------------------- #
def _cell_mask(shape=(24, 24, 24), margin=2):
    m = np.zeros(shape, bool)
    m[margin:shape[0] - margin, margin:shape[1] - margin,
      margin:shape[2] - margin] = True
    return m


def _spot_image(mask, centres, base=5.0, peak=200.0, half=1):
    """A masked channel crop with a bright 3-voxel cube at each centre on a dim
    background (a synthetic punctate signal)."""
    img = np.full(mask.shape, base, dtype=np.float64)
    for (z, y, x) in centres:
        img[z - half:z + half + 1, y - half:y + half + 1, x - half:x + half + 1] = peak
    return np.where(mask, img, 0.0)


def test_puncta_columns_present_with_prefix():
    mask = _cell_mask()
    out = qf.compute_channel_puncta(_spot_image(mask, [(12, 12, 12)]), mask, "hcr")
    for s in _PUNCTA_SUFFIXES:
        assert f"hcr{s}" in out, f"hcr{s} missing"
    assert len(out) == len(_PUNCTA_SUFFIXES)


def test_returns_dict_not_dataframe():
    mask = _cell_mask()
    out = qf.compute_channel_puncta(_spot_image(mask, [(12, 12, 12)]), mask, "c")
    assert isinstance(out, dict)


def test_n_spots_counted():
    """N bright spots -> puncta_count ≈ N + a HIGH punctate_fraction (the signal IS
    the spots) + a finite peak intensity."""
    mask = _cell_mask()
    centres = [(6, 6, 6), (6, 16, 16), (16, 6, 16), (16, 16, 6), (12, 12, 12)]
    out = qf.compute_channel_puncta(_spot_image(mask, centres), mask, "hcr")
    assert out["hcr_puncta_count"] == pytest.approx(len(centres), abs=1)
    assert out["hcr_punctate_fraction"] > 0.3
    assert out["hcr_puncta_intensity_mean"] == pytest.approx(200.0, rel=0.2)
    assert np.isfinite(out["hcr_puncta_mean_volume"])
    assert out["hcr_puncta_density"] > 0.0


def test_diffuse_no_spots_low_fraction():
    """A smooth / uniform fill -> no resolvable spots -> count 0, density 0,
    punctate_fraction 0, NaN spot-only stats."""
    mask = _cell_mask()
    diffuse = np.where(mask, 100.0, 0.0).astype(np.float64)
    out = qf.compute_channel_puncta(diffuse, mask, "hcr")
    assert out["hcr_puncta_count"] == 0.0
    assert out["hcr_puncta_density"] == 0.0
    assert out["hcr_punctate_fraction"] == 0.0
    assert np.isnan(out["hcr_puncta_intensity_mean"])
    assert np.isnan(out["hcr_puncta_mean_volume"])
    assert np.isnan(out["hcr_puncta_nn_distance"])


def test_spot_clustering_reflected_in_nn_distance():
    """Clustered spots -> a SMALL mean nearest-neighbour distance; dispersed spots ->
    a LARGE one. The nn_distance carries the clustering signal."""
    mask = _cell_mask((30, 30, 30))
    clustered = [(12, 12, 12), (12, 12, 16), (12, 16, 12), (16, 12, 12)]
    dispersed = [(6, 6, 6), (6, 24, 24), (24, 6, 24), (24, 24, 6)]
    oc = qf.compute_channel_puncta(_spot_image(mask, clustered), mask, "g")
    od = qf.compute_channel_puncta(_spot_image(mask, dispersed), mask, "g")
    assert oc["g_puncta_count"] == pytest.approx(4, abs=1)
    assert od["g_puncta_count"] == pytest.approx(4, abs=1)
    assert oc["g_puncta_nn_distance"] < od["g_puncta_nn_distance"]


def test_puncta_nn_distance_nan_for_single_spot():
    """A single spot has no neighbour -> nn_distance NaN, but count/intensity/volume
    are still defined."""
    mask = _cell_mask()
    out = qf.compute_channel_puncta(_spot_image(mask, [(12, 12, 12)]), mask, "g")
    assert out["g_puncta_count"] == pytest.approx(1, abs=1)
    assert np.isnan(out["g_puncta_nn_distance"])
    assert np.isfinite(out["g_puncta_intensity_mean"])


def test_puncta_spacing_scales_nn_distance():
    """Physical spacing scales the nearest-neighbour distance by the spacing along
    the separating axis."""
    mask = _cell_mask((30, 30, 30))
    # two spots separated along Z only (so anisotropic Z-spacing scales their NN).
    centres = [(8, 15, 15), (20, 15, 15)]
    o_v = qf.compute_channel_puncta(_spot_image(mask, centres), mask, "g")
    o_p = qf.compute_channel_puncta(_spot_image(mask, centres), mask, "g",
                                    spacing=(2.0, 1.0, 1.0))
    assert o_p["g_puncta_nn_distance"] == pytest.approx(
        2.0 * o_v["g_puncta_nn_distance"], rel=0.05)


def test_punctate_high_for_spots_low_for_diffuse():
    """The punctate_fraction discriminates a punctate signal (HIGH) from a
    diffuse one (LOW) on the same total signal budget."""
    mask = _cell_mask()
    spots = _spot_image(mask, [(8, 8, 8), (8, 16, 16), (16, 8, 16), (16, 16, 8)])
    diffuse = np.where(mask, 100.0, 0.0).astype(np.float64)
    fr_spots = qf.compute_channel_puncta(spots, mask, "x")["x_punctate_fraction"]
    fr_diffuse = qf.compute_channel_puncta(diffuse, mask, "x")["x_punctate_fraction"]
    assert fr_spots > fr_diffuse


def test_puncta_outside_mask_discarded():
    """A bright blob OUTSIDE the cell mask (in the bbox but a neighbour's territory)
    must NOT be counted — only in-mask spot centres count."""
    shape = (24, 24, 24)
    mask = np.zeros(shape, bool)
    mask[2:12, 2:22, 2:22] = True            # cell occupies only the low-Z half
    img = np.full(shape, 5.0)
    img[7, 12, 12] = 200.0                    # in-mask spot
    img[7 + 14, 12, 12] = 200.0               # spot in the high-Z half (OUTSIDE mask)
    # zero outside the mask so only the in-mask spot can seed.
    img = np.where(mask, img, np.where(np.arange(shape[0])[:, None, None] >= 12,
                                       200.0 * (img == 200.0), 0.0))
    out = qf.compute_channel_puncta(img, mask, "g")
    # exactly the one in-mask spot is counted.
    assert out["g_puncta_count"] == pytest.approx(1, abs=1)


def test_puncta_nan_guards():
    mask = _cell_mask()
    # empty mask -> all NaN (schema-stable).
    empty = np.zeros((8, 8, 8), bool)
    out = qf.compute_channel_puncta(np.zeros((8, 8, 8)), empty, "c")
    for s in _PUNCTA_SUFFIXES:
        assert np.isnan(out[f"c{s}"])
    # shape mismatch -> all NaN.
    out = qf.compute_channel_puncta(np.zeros((8, 8, 8)), mask, "c")
    for s in _PUNCTA_SUFFIXES:
        assert np.isnan(out[f"c{s}"])
    # all-NaN crop (failed read) -> count/density/fraction 0, spot stats NaN.
    out = qf.compute_channel_puncta(np.full(mask.shape, np.nan), mask, "c")
    assert out["c_puncta_count"] == 0.0
    assert out["c_puncta_density"] == 0.0
    assert out["c_punctate_fraction"] == 0.0
    assert np.isnan(out["c_puncta_intensity_mean"])
    # a tiny mask (< min voxels) -> 0 count, NaN spot stats.
    tiny = np.zeros((8, 8, 8), bool)
    tiny[3:5, 3:5, 3] = True                  # 4 voxels < _PUNCTA_MIN_MASK_VOXELS
    out = qf.compute_channel_puncta(np.where(tiny, 100.0, 0.0).astype(float), tiny, "c")
    assert out["c_puncta_count"] == 0.0
    assert np.isnan(out["c_puncta_nn_distance"])


def test_distinct_prefixes_distinct_columns():
    mask = _cell_mask()
    img = _spot_image(mask, [(12, 12, 12)])
    a = qf.compute_channel_puncta(img, mask, "nuc")
    b = qf.compute_channel_puncta(img, mask, "mem")
    assert all(k.startswith("nuc") for k in a)
    assert all(k.startswith("mem") for k in b)
    assert a["nuc_puncta_count"] == b["mem_puncta_count"]


# --------------------------------------------------------------------------- #
# manifest tags
# --------------------------------------------------------------------------- #
def test_territory_in_static_manifest():
    man = fm.get_manifest()
    for c in _TERRITORY_COLS:
        assert c in man, c
        t = man[c]
        for key in ("units", "bounded", "role", "frame", "redundancy_group"):
            assert key in t, (c, key)
        assert t["role"] == "biology"
        assert t["frame"] == "tissue"
        assert t["redundancy_group"] == "watershed_territory"
    assert man["territory_solidity"]["bounded"] is True
    assert man["ws_territory_volume"]["bounded"] is False


def test_channel_puncta_tags_helper():
    tags = fm.channel_puncta_tags("hcr")
    for s in _PUNCTA_SUFFIXES:
        assert f"hcr{s}" in tags, s
        t = tags[f"hcr{s}"]
        for key in ("units", "bounded", "role", "frame", "redundancy_group"):
            assert key in t, (s, key)
        assert t["role"] == "biology"
        assert t["frame"] == "image"
        assert t["redundancy_group"] == "channel_puncta_hcr"
    assert tags["hcr_punctate_fraction"]["bounded"] is True
    assert tags["hcr_puncta_count"]["bounded"] is False


def test_waveg_in_module_all():
    for fn in ("compute_watershed_territory", "compute_channel_puncta",
               "WATERSHED_TERRITORY_COLUMNS"):
        assert fn in qf.__all__, fn
    assert "channel_puncta_tags" in fm.__all__


# --------------------------------------------------------------------------- #
# FeatureStage families registry + per-channel recovery
# --------------------------------------------------------------------------- #
def test_watershed_territory_family_registered():
    from packages.epicycle_scaledseg.core.feature_stage import _FEATURE_FAMILIES
    assert set(_FEATURE_FAMILIES["watershed_territory"]) == set(_TERRITORY_COLS)


def test_per_channel_puncta_families_recovered_from_columns():
    from packages.epicycle_scaledseg.core.feature_stage import _per_channel_puncta_families
    cols = (
        [f"nuc{s}" for s in _PUNCTA_SUFFIXES]
        + [f"hcr{s}" for s in _PUNCTA_SUFFIXES]
        + ["area", "nuc_intensity_mean"]
    )
    fams = _per_channel_puncta_families(cols, [(0, "nuc"), (1, "hcr")])
    assert set(fams["channel_puncta_nuc"]) == {f"nuc{s}" for s in _PUNCTA_SUFFIXES}
    assert set(fams["channel_puncta_hcr"]) == {f"hcr{s}" for s in _PUNCTA_SUFFIXES}


def test_per_channel_puncta_families_empty_when_no_channels():
    from packages.epicycle_scaledseg.core.feature_stage import _per_channel_puncta_families
    assert _per_channel_puncta_families(["area", "nuc_puncta_count"], None) == {}


# --------------------------------------------------------------------------- #
# FeatureStage wiring: puncta appear ONLY when channels are sampled.
# --------------------------------------------------------------------------- #
def _solid_cube_label(shape=(20, 24, 24), lo=2, hi=18):
    lab = np.zeros(shape, dtype=np.int32)
    lab[lo:hi, lo:hi, lo:hi] = 1
    return lab


def test_extract_one_labels_only_emits_no_puncta(tmp_path):
    """The per-cell worker with NO sampler config emits NO Wave-G puncta columns
    (default-off byte-identical)."""
    from packages.epicycle_scaledseg.core import feature_stage as fs

    lab = _solid_cube_label()
    p = tmp_path / "labels.npy"
    np.save(p, lab)
    fs._init_feature_worker(str(p))
    try:
        task = (1, (2, 18, 2, 18, 2, 18), None, False, False)
        row = fs._extract_one(task)
    finally:
        fs._WORKER_SRC = None
    assert isinstance(row, dict), row
    assert not any("_puncta_" in k or "_punctate_fraction" in k for k in row)


def test_extract_one_with_channel_emits_puncta(tmp_path, monkeypatch):
    """With a channel sampler config, _extract_one emits the per-channel puncta
    columns (a bright spotty channel -> a positive count)."""
    from packages.epicycle_scaledseg.core import feature_stage as fs
    from packages.epicycle_scaledseg.reconcile import raw_roi

    lab = _solid_cube_label()
    p = tmp_path / "labels.npy"
    np.save(p, lab)

    full_mask = lab == 1
    full_ch = np.full(lab.shape, 5.0)
    for (z, y, x) in [(6, 6, 6), (6, 14, 14), (14, 6, 14), (14, 14, 6)]:
        full_ch[z - 1:z + 2, y - 1:y + 2, x - 1:x + 2] = 200.0
    full_ch = np.where(full_mask, full_ch, 0.0)

    def _fake_read_global_roi(tiles_dir, bbox, channel, halo=0, halo_z=None):
        z0, z1, y0, y1, x0, x1 = bbox
        return full_ch[z0:z1, y0:y1, x0:x1]

    monkeypatch.setattr(raw_roi, "read_global_roi", _fake_read_global_roi)

    fs._init_feature_worker(str(p), tiles_dir="FAKE", recon_dir=None,
                            channels=[(0, "hcr")], prob=False)
    try:
        task = (1, (2, 18, 2, 18, 2, 18), None, False, False)
        row = fs._extract_one(task)
    finally:
        fs._WORKER_SRC = None
        fs._WORKER_TILES_DIR = None
        fs._WORKER_RECON_DIR = None
        fs._WORKER_CHANNELS = None
        fs._WORKER_PROB = False
    assert isinstance(row, dict), row
    for s in _PUNCTA_SUFFIXES:
        assert f"hcr{s}" in row, f"hcr{s}"
    assert row["hcr_puncta_count"] >= 1.0
    # intensity columns still present (puncta rides alongside texture/localization).
    assert "hcr_intensity_mean" in row


# --------------------------------------------------------------------------- #
# extract_label_features(territory=True) global pass: territory columns appear
# only with territory=True; labels-only stays byte-identical.
# --------------------------------------------------------------------------- #
def test_extract_label_features_territory_off_no_columns():
    """The default (territory=False) extract emits NO territory columns."""
    from packages.epicycle_scaledseg.reconcile.features import extract_label_features
    lab = _two_separated_cells()
    df = extract_label_features(lab, include_surface=False, n_workers=1)
    assert not df.empty
    assert not any(c in df.columns for c in _TERRITORY_COLS)


def test_extract_label_features_territory_on_emits_columns():
    """territory=True runs the global watershed-territory pass and merges the 4
    columns onto the per-cell table."""
    from packages.epicycle_scaledseg.reconcile.features import extract_label_features
    lab = _two_separated_cells()
    df = extract_label_features(
        lab, include_surface=False, territory=True, n_workers=1
    ).set_index("label")
    for c in _TERRITORY_COLS:
        assert c in df.columns, c
    # both separated cells got room to grow -> ratio > 1.
    assert df.loc[1, "territory_to_cell_ratio"] > 1.0
    assert df.loc[2, "territory_to_cell_ratio"] > 1.0
