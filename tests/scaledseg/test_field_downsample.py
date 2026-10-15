"""Phase C / S2-S3 — DOWNSAMPLED field passes (``compute_tissue_depth`` /
``compute_watershed_territory`` / ``compute_lumen_features`` with ``downsample=N``)
+ the streaming coarse-foreground helper.

The gate (most important): ``downsample=1`` MUST be BYTE-IDENTICAL to the current
code (the same call without the arg). For ``N > 1`` the field is computed on a
foreground downsampled by ``N`` (an APPROXIMATION) — the tests assert it runs,
produces the SAME columns, stays finite where ``N == 1`` is, and correlates with
the ``N == 1`` reference on a curved/blobby multi-cell volume.

Plus: the streaming coarse foreground matches the resident max-pool exactly, and the
FeatureStage memmap path (``field_downsample=2``) does not crash + produces the
columns (and uses a memmap, not a resident copy).
"""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from packages.epicycle_scaledseg.core.qc_features import (
    extract_features,
    compute_tissue_depth, TISSUE_DEPTH_COLUMNS,
    compute_watershed_territory, WATERSHED_TERRITORY_COLUMNS,
    compute_lumen_features, LUMEN_COLUMNS,
)
from packages.epicycle_scaledseg.core.assembly import downsample_binary_maxpool
from packages.epicycle_scaledseg.core.field_downsample import coarse_foreground, coarse_labels


_SP = (2.0, 0.5, 0.5)


# --------------------------------------------------------------------------- #
# synthetic volumes
# --------------------------------------------------------------------------- #
def _ball(R, c, shape):
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    return ((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) <= R * R


def _sphere_of_cells(shape=(48, 48, 48), R=22, cube=8):
    """A solid ball partitioned into cube cells — a curved tissue with a genuine
    inside/outside surface (the case the downsampled NORMAL is hardest on)."""
    centre = (shape[0] // 2, shape[1] // 2, shape[2] // 2)
    ball = _ball(R, centre, shape)
    lab = np.zeros(shape, np.int32)
    lid = 1
    for z0 in range(0, shape[0], cube):
        for y0 in range(0, shape[1], cube):
            for x0 in range(0, shape[2], cube):
                blk = np.zeros(shape, bool)
                blk[z0:z0 + cube, y0:y0 + cube, x0:x0 + cube] = True
                cell = blk & ball
                if cell.sum() >= cube ** 3 // 4:
                    lab[cell] = lid
                    lid += 1
    return lab


def _hollow_ball(shape=(48, 48, 48), R_out=22, R_in=10, cube=8):
    """A shell of cells around an enclosed cavity (a lumen) — for the lumen pass."""
    centre = (shape[0] // 2, shape[1] // 2, shape[2] // 2)
    shell = _ball(R_out, centre, shape) & ~_ball(R_in, centre, shape)
    lab = np.zeros(shape, np.int32)
    lid = 1
    for z0 in range(0, shape[0], cube):
        for y0 in range(0, shape[1], cube):
            for x0 in range(0, shape[2], cube):
                blk = np.zeros(shape, bool)
                blk[z0:z0 + cube, y0:y0 + cube, x0:x0 + cube] = True
                cell = blk & shell
                if cell.sum() >= 12:
                    lab[cell] = lid
                    lid += 1
    return lab


def _ref_df(labels):
    """The per-cell morphology table the field passes consume (label / centroid-* /
    bbox-* / area), built the SAME way as a real run."""
    df = extract_features(labels, spacing=_SP)
    df["label"] = df["label"].astype(np.int64)
    return df


# --------------------------------------------------------------------------- #
# 1) N=1 BYTE-IDENTICAL (the gate)
# --------------------------------------------------------------------------- #
def test_tissue_depth_n1_byte_identical():
    lab = _sphere_of_cells()
    a = compute_tissue_depth(lab, _ref_df(lab), spacing=_SP, downsample=1)
    b = compute_tissue_depth(lab, _ref_df(lab), spacing=_SP)        # default arg
    assert_frame_equal(a, b)


def test_watershed_territory_n1_byte_identical():
    lab = _sphere_of_cells()
    a = compute_watershed_territory(lab, _ref_df(lab), spacing=_SP, downsample=1)
    b = compute_watershed_territory(lab, _ref_df(lab), spacing=_SP)
    assert_frame_equal(a, b)


def test_lumen_n1_byte_identical():
    lab = _hollow_ball()
    a = compute_lumen_features(lab, _ref_df(lab), spacing=_SP, downsample=1)
    b = compute_lumen_features(lab, _ref_df(lab), spacing=_SP)
    assert_frame_equal(a, b)


def test_tissue_depth_n1_byte_identical_no_spacing():
    lab = _sphere_of_cells()
    a = compute_tissue_depth(lab, _ref_df(lab), downsample=1)
    b = compute_tissue_depth(lab, _ref_df(lab))
    assert_frame_equal(a, b)


# --------------------------------------------------------------------------- #
# 1b) extract_label_features + FeatureStage N=1 byte-identical
# --------------------------------------------------------------------------- #
def test_extract_label_features_field_downsample1_identical():
    from packages.epicycle_scaledseg.reconcile.features import extract_label_features
    lab = _sphere_of_cells()
    a = extract_label_features(
        lab, spacing=_SP, tissue_depth=True, territory=True, lumen=True,
        field_downsample=1, n_workers=1)
    b = extract_label_features(
        lab, spacing=_SP, tissue_depth=True, territory=True, lumen=True,
        n_workers=1)
    assert_frame_equal(a.sort_index(axis=1), b.sort_index(axis=1))


def _write_prov(path, vol):
    cols = ["dense_id", "seam_status", "n_tiles", "min_match_iou", "voxel_count",
            "z_min", "z_max", "y_min", "y_max", "x_min", "x_max",
            "touches_border", "tiles", "constituents"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for lbl in range(1, int(vol.max()) + 1):
            m = vol == lbl
            if not m.any():
                continue
            zs, ys, xs = np.nonzero(m)
            w.writerow({
                "dense_id": lbl, "seam_status": "clean", "n_tiles": 1,
                "min_match_iou": "", "voxel_count": int(m.sum()),
                "z_min": int(zs.min()), "z_max": int(zs.max()),
                "y_min": int(ys.min()), "y_max": int(ys.max()),
                "x_min": int(xs.min()), "x_max": int(xs.max()),
                "touches_border": 0, "tiles": "t0", "constituents": "",
            })


def _stage_fixture(tmp, vol):
    stitch = tmp / "stage4_full_stitch"
    stitch.mkdir(parents=True, exist_ok=True)
    np.save(str(stitch / "global_labels_clean.npy"), vol)
    _write_prov(stitch / "label_provenance_clean.csv", vol)
    return stitch


def _run_stage(tmp, vol, **kw):
    from packages.epicycle_scaledseg.core.feature_stage import FeatureStage
    stitch = _stage_fixture(tmp, vol)
    st = FeatureStage(input_dir=str(stitch), voxel_size_um=_SP, include_surface=False,
                      n_workers=1, **kw)
    st.run([Path("x/tiles/t0")], n_gpus=0)
    return pd.read_csv(stitch / "cell_features.csv")


def test_feature_stage_field_downsample1_identical(tmp_path):
    vol = _sphere_of_cells()
    a = _run_stage(tmp_path / "a", vol, tissue_depth=True, territory=True, lumen=True,
                   field_downsample=1)
    b = _run_stage(tmp_path / "b", vol, tissue_depth=True, territory=True, lumen=True)
    # Same columns + same values (sorted by label; ignore column order).
    a = a.sort_values("label").reset_index(drop=True)
    b = b.sort_values("label").reset_index(drop=True)
    assert_frame_equal(a.sort_index(axis=1), b.sort_index(axis=1))


# --------------------------------------------------------------------------- #
# 2) N>1 runs + sane
# --------------------------------------------------------------------------- #
def _finite_mask(s):
    return np.isfinite(s.values.astype(float))


def test_tissue_depth_downsample2_sane():
    lab = _sphere_of_cells()
    ref = compute_tissue_depth(lab, _ref_df(lab), spacing=_SP)
    ds = compute_tissue_depth(lab, _ref_df(lab), spacing=_SP, downsample=2)

    # same columns
    for c in TISSUE_DEPTH_COLUMNS:
        assert c in ds.columns

    # normalized_tissue_depth in [0,1] where finite
    nd = ds["normalized_tissue_depth"].dropna().values
    assert nd.size > 0
    assert ((nd >= 0.0) & (nd <= 1.0)).all()

    # outward normal is a unit vector where defined
    ov = ds[["tissue_outward_z", "tissue_outward_y", "tissue_outward_x"]].values
    fin = np.isfinite(ov).all(axis=1)
    assert fin.any()
    norms = np.linalg.norm(ov[fin], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)

    # finite where the N=1 reference is finite (for the field-derived depth)
    rf = _finite_mask(ref["tissue_depth_mean"])
    df_ = _finite_mask(ds["tissue_depth_mean"])
    assert (df_[rf]).all()

    # depth correlates with the N=1 reference
    m = _finite_mask(ref["tissue_depth_mean"]) & _finite_mask(ds["tissue_depth_mean"])
    r = np.corrcoef(ref["tissue_depth_mean"].values[m],
                    ds["tissue_depth_mean"].values[m])[0, 1]
    assert r > 0.9, f"tissue_depth_mean corr={r:.3f}"


def test_tissue_depth_downsample2_at_border_and_local_exact():
    """at_fov_border + local_tissue_depth are computed from the per-cell table, so
    they should be IDENTICAL between N=1 and N=2 (no field involved)."""
    lab = _sphere_of_cells()
    ref = compute_tissue_depth(lab, _ref_df(lab), spacing=_SP)
    ds = compute_tissue_depth(lab, _ref_df(lab), spacing=_SP, downsample=2)
    np.testing.assert_array_equal(
        ref["at_fov_border"].values, ds["at_fov_border"].values)
    # local_tissue_depth = depth_mean / r_equiv; depth_mean differs (coarse) but
    # r_equiv (from exact area) is the same -> just assert all finite + positive.
    ld = ds["local_tissue_depth"].dropna().values
    assert ld.size > 0 and (ld >= 0).all()


def test_watershed_territory_downsample2_sane():
    lab = _sphere_of_cells()
    ref = compute_watershed_territory(lab, _ref_df(lab), spacing=_SP)
    ds = compute_watershed_territory(lab, _ref_df(lab), spacing=_SP, downsample=2)
    for c in WATERSHED_TERRITORY_COLUMNS:
        assert c in ds.columns
    # territory_to_cell_ratio >= ~1 where finite
    tr = ds["territory_to_cell_ratio"].dropna().values
    assert tr.size > 0
    assert (tr >= 0.9).all()       # >= 1 modulo coarse rounding
    # rank-correlation of territory_to_cell_ratio with the N=1 reference
    from scipy.stats import spearmanr
    m = _finite_mask(ref["territory_to_cell_ratio"]) & _finite_mask(ds["territory_to_cell_ratio"])
    if m.sum() >= 3:
        rho = spearmanr(ref["territory_to_cell_ratio"].values[m],
                        ds["territory_to_cell_ratio"].values[m]).correlation
        assert rho > 0.3, f"territory rank-corr={rho:.3f}"


def test_ws_nearest_gap_subN_is_nan_not_zero():
    """L1 — two cells separated by a gap SMALLER than the downsample factor get
    OR-pooled into coarse adjacency: their coarse midline half-gap is exactly 0, so
    ``ws_nearest_gap`` is UNRESOLVED at this resolution. It must be emitted as NaN
    ("unknown at this downsample"), NOT 0 (which would read as "physically touching"
    — a false strong signal). Two 6-wide cubes with a 1-voxel gap, downsampled by N=4
    (gap << N) -> the two cells merge in the coarse grid -> midline EDT 0."""
    shape = (16, 24, 24)
    lab = np.zeros(shape, np.int32)
    # 1-voxel gap on Y between cell 1 (y 4..10) and cell 2 (y 11..17), well below N=4.
    lab[4:12, 4:10, 8:16] = 1
    lab[4:12, 11:17, 8:16] = 2
    df = _ref_df(lab)
    ds = compute_watershed_territory(lab, df, spacing=_SP, downsample=4)
    assert "ws_nearest_gap" in ds.columns
    gaps = ds["ws_nearest_gap"].to_numpy()
    # The sub-N gap is unresolved -> NaN for these touching-in-coarse cells, and in
    # particular NEVER an exact 0 (the pre-L1 value).
    assert not np.any(gaps == 0.0), \
        "sub-N gap must be NaN (unresolved), never 0 (touching)"


def test_lumen_downsample2_sane():
    lab = _hollow_ball()
    ref = compute_lumen_features(lab, _ref_df(lab), spacing=_SP)
    ds = compute_lumen_features(lab, _ref_df(lab), spacing=_SP, downsample=2)
    for c in LUMEN_COLUMNS:
        assert c in ds.columns
    # a hollow ball HAS a lumen -> some finite distances at both N
    assert _finite_mask(ref["distance_to_lumen"]).any()
    assert _finite_mask(ds["distance_to_lumen"]).any()
    # facing fraction in [0,1]
    ff = ds["lumen_facing_fraction"].dropna().values
    if ff.size:
        assert ((ff >= 0.0) & (ff <= 1.0)).all()


# --------------------------------------------------------------------------- #
# 3) streaming coarse foreground == resident maxpool
# --------------------------------------------------------------------------- #
def test_coarse_foreground_streaming_matches_resident(tmp_path):
    lab = _sphere_of_cells()
    p = tmp_path / "lab.npy"
    np.save(str(p), lab)
    mm = np.load(str(p), mmap_mode="r")
    assert isinstance(mm, np.memmap)

    streamed = coarse_foreground(mm, 2)
    resident = downsample_binary_maxpool(lab > 0, 2)
    # downsample_binary_maxpool drops the partial last block (floor-div); coarse_
    # foreground keeps it (ceil-div). On a divisible shape they MUST match exactly;
    # compare the common (floor-div) region to be robust to the tail.
    z2, y2, x2 = resident.shape
    np.testing.assert_array_equal(streamed[:z2, :y2, :x2], resident)


def test_coarse_foreground_divisible_exact():
    """On a fully-divisible shape, streamed == resident maxpool everywhere."""
    lab = _sphere_of_cells(shape=(48, 48, 48))     # divisible by 2,3,4
    for n in (2, 3, 4):
        streamed = coarse_foreground(lab, n)        # resident-array path
        resident = downsample_binary_maxpool(lab > 0, n)
        assert streamed.shape == resident.shape
        np.testing.assert_array_equal(streamed, resident)


def test_coarse_foreground_non_divisible_clamps():
    """A non-divisible shape -> ceil-div coarse shape, partial last block kept."""
    lab = np.zeros((10, 10, 10), np.int32)
    lab[8:10, 8:10, 8:10] = 1                        # only the LAST (partial) block
    cf = coarse_foreground(lab, 3)                   # ceil(10/3)=4 per axis
    assert cf.shape == (4, 4, 4)
    assert cf[3, 3, 3]                               # the tail block is foreground


def _brute_coarse_fg(lab, n):
    """Reference coarse foreground via an explicit per-block any() (slow but obviously
    correct), ceil-div shape with the partial last block clamped."""
    z, y, x = lab.shape
    cz, cy, cx = -(-z // n), -(-y // n), -(-x // n)
    out = np.zeros((cz, cy, cx), bool)
    fg = lab > 0
    for kz in range(cz):
        for ky in range(cy):
            for kx in range(cx):
                out[kz, ky, kx] = fg[kz * n:min(kz * n + n, z),
                                     ky * n:min(ky * n + n, y),
                                     kx * n:min(kx * n + n, x)].any()
    return out


@pytest.mark.parametrize("n", [2, 3, 4])
def test_coarse_foreground_streaming_non_divisible(tmp_path, n):
    """Streaming memmap path == brute-force reference on a NON-divisible shape (so all
    three of _maxpool's tails — interior, x-tail, y-tail/corner — are exercised)."""
    rng = np.random.default_rng(0)
    lab = (rng.random((11, 13, 17)) > 0.5).astype(np.int32)   # all axes non-divisible
    p = tmp_path / f"lab{n}.npy"
    np.save(str(p), lab)
    mm = np.load(str(p), mmap_mode="r")
    streamed = coarse_foreground(mm, n)
    ref = _brute_coarse_fg(lab, n)
    assert streamed.shape == ref.shape
    np.testing.assert_array_equal(streamed, ref)
    # the resident path must also match the same reference.
    np.testing.assert_array_equal(coarse_foreground(lab, n), ref)


def test_coarse_labels_majority_nonzero():
    lab = np.zeros((6, 6, 6), np.int32)
    lab[2:4, 2:4, 2:4] = 7                           # a block around the centre
    cl = coarse_labels(lab, 2)
    assert cl.shape == (3, 3, 3)
    # block (1,1,1) covers full-res [2:4, 2:4, 2:4] = all label 7 -> majority-of-
    # nonzero seed is 7. (H1: majority-of-nonzero replaces the old block-CENTRE seed.)
    assert cl[1, 1, 1] == 7


def test_coarse_labels_off_centre_cell_retained():
    """H1: a cell that does NOT occupy its block centre still gets a coarse seed
    (the old block-CENTRE seed dropped it, NaN-ing its territory columns)."""
    lab = np.zeros((4, 4, 4), np.int32)
    # block (0,0,0) = full-res [0:2,0:2,0:2]; its CENTRE voxel is (1,1,1)=off=2//2.
    # Put the cell ANYWHERE in the block EXCEPT the centre voxel -> the old
    # centre-seed gave this block 0; majority-of-nonzero gives it the cell id.
    lab[0, 0, 0] = 9                                  # corner, not the centre (1,1,1)
    cl = coarse_labels(lab, 2)
    assert cl.shape == (2, 2, 2)
    assert cl[0, 0, 0] == 9, "off-centre cell must still seed its block (H1 fix)"


def test_coarse_labels_dropout_warning(caplog):
    """A sub-N cell entirely overlapped by a bigger neighbour in every block it
    touches gets no seed -> coarse_labels logs a WARNING (no longer silent)."""
    import logging
    lab = np.zeros((2, 2, 2), np.int32)
    lab[:] = 1                                        # big cell fills the whole block
    lab[0, 0, 0] = 2                                  # cell 2 is a 1-voxel minority
    with caplog.at_level(logging.WARNING, logger="epicycle.features"):
        cl = coarse_labels(lab, 2)
    # the single coarse voxel goes to the majority (cell 1); cell 2 is dropped.
    assert cl[0, 0, 0] == 1
    assert any("no coarse seed" in r.message.lower() or "NO coarse seed" in r.message
               for r in caplog.records), "expected a residual-dropout WARNING"


# --------------------------------------------------------------------------- #
# 4) FeatureStage memmap path (field_downsample=2)
# --------------------------------------------------------------------------- #
def test_feature_stage_downsample2_uses_memmap(tmp_path, monkeypatch):
    """field_downsample=2 must produce the field columns AND must NOT materialise the
    full-res volume — neither via ``np.array`` (the resident field branch) nor via a
    ``find_objects(full_labels.astype(int32))`` RESIDENT int32 copy inside the 5 SIGNED
    families (the C1 OOM the old test missed: it only spied ``fs.np.array``, which the
    N=2 branch never calls).

    Strategy (the cleanest direct check per the C1 audit): spy on the
    ``scipy.ndimage.find_objects`` the SIGNED families import, recording the SIZE +
    RESIDENCY of every input. A full-volume-sized input is fine ONLY if it is a memmap
    (``find_objects`` STREAMS a memmap); a full-volume-sized RESIDENT ``ndarray`` is the
    ``.astype(int32)`` materialisation we are forbidding. PRE-FIX: the 5 signed families
    each materialise one -> the assertion FAILS. POST-FIX: they take ``slices=`` and
    skip it -> 0 resident full-volume scans.
    """
    import scipy.ndimage as ndi

    # A fixture whose FULL volume is >> the largest per-cell bbox: a sphere of small
    # cube cells in a big box. The biggest cell bbox is ~cube³; the volume is the box.
    vol = _sphere_of_cells(shape=(48, 48, 48), R=22, cube=6)
    full_size = vol.size

    # largest per-cell bbox size, to set a "full-volume-sized" threshold well above any
    # legitimate per-cell crop (>= 50x headroom asked for by the audit).
    _real_find_objects = ndi.find_objects
    objs = _real_find_objects(vol)
    max_bbox = max(
        int(np.prod([s.stop - s.start for s in sl])) for sl in objs if sl is not None)
    assert full_size >= 50 * max_bbox, (
        f"fixture too small: full={full_size} vs 50*max_bbox={50 * max_bbox}")
    big_threshold = full_size // 2          # anything >= half the volume is "full-sized"

    # Spy on scipy.ndimage.find_objects (the SIGNED families import it as
    # ``from scipy.ndimage import find_objects`` at CALL time, so patching the module
    # attribute is what they pick up). Record (size, is_memmap) per call.
    calls = []

    def _spy_find_objects(arr, *args, **kw):
        try:
            calls.append((int(getattr(arr, "size", 0)), isinstance(arr, np.memmap)))
        except Exception:
            calls.append((0, False))
        return _real_find_objects(arr, *args, **kw)

    monkeypatch.setattr(ndi, "find_objects", _spy_find_objects)

    df = _run_stage(tmp_path, vol, tissue_depth=True, territory=True, lumen=True,
                    field_downsample=2)
    # columns present (the families actually ran)
    for c in ("distance_to_tissue_surface", "ws_territory_volume", "distance_to_lumen"):
        assert c in df.columns

    # NO full-volume-sized RESIDENT find_objects input (a memmap is allowed — it
    # streams). PRE-FIX the 5 signed families each .astype(int32) a full resident copy.
    resident_full = [sz for sz, is_mm in calls if sz >= big_threshold and not is_mm]
    assert not resident_full, (
        f"field_downsample=2 ran find_objects on {len(resident_full)} full-volume "
        f"RESIDENT array(s) (sizes {resident_full}) — the C1 .astype(int32) "
        "materialisation. Expected 0 (slices threaded; memmap streamed).")
