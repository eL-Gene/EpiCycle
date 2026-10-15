"""FeatureStage Phase-A opt-in TRULY-GLOBAL field passes
(``tissue_depth`` / ``territory`` / ``lumen`` / ``qc_score``): they run on a
RESIDENT copy of the global label volume — the SAME sequence as
``reconcile.extract_label_features`` with the matching flags — so the columns are
byte-identical.

Asserts: the parity gate (every shared field/qc_score column matches
``extract_label_features`` exactly); each flag OFF by default (no signature
columns); the manifest records the flags; and tissue_depth / qc_score run WITHOUT
adjacency (degraded, never crash)."""

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from packages.epicycle_scaledseg.core.feature_stage import FeatureStage

_PROV_COLS = [
    "dense_id", "seam_status", "n_tiles", "min_match_iou", "voxel_count",
    "z_min", "z_max", "y_min", "y_max", "x_min", "x_max",
    "touches_border", "tiles", "constituents",
]

# Signature columns that prove each global family actually landed in the table.
_TISSUE_DEPTH_SIG = "distance_to_tissue_surface"
_TERRITORY_SIG = "ws_territory_volume"
_LUMEN_SIG = "distance_to_lumen"
_QC_SCORE_SIG = "qc_score"


def _touch_volume():
    """A small SHEET of touching cells (a 2x3 grid of blocks in a Z-slab) so the
    tissue-depth + signed apical/basal families have real geometric signal, plus an
    isolated cell. Cells tile the Y-X plane edge-to-edge -> a multi-cell epithelial
    patch with a genuine inside/outside surface for the tissue-EDT field.

    Layout (Z = 2..9, an 8-voxel-thick slab embedded in a 14-deep volume so the
    slab has tissue surface above + below it):
        cols x in [4,12), [12,20), [20,28) ; rows y in [4,12), [12,20)
        -> labels 1..6 ; label 7 isolated far away.
    """
    vol = np.zeros((14, 28, 36), np.int32)
    lbl = 1
    for y0 in (4, 12):
        for x0 in (4, 12, 20):
            vol[2:10, y0:y0 + 8, x0:x0 + 8] = lbl
            lbl += 1
    # isolated cell, well separated from the sheet
    vol[2:10, 20:26, 28:34] = 7
    return vol


def _write_prov(path, vol):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_PROV_COLS)
        w.writeheader()
        for lbl in range(1, int(vol.max()) + 1):
            mask = vol == lbl
            if not mask.any():
                continue
            zs, ys, xs = np.nonzero(mask)
            w.writerow({
                "dense_id": lbl, "seam_status": "clean", "n_tiles": 1,
                "min_match_iou": "", "voxel_count": int(mask.sum()),
                "z_min": int(zs.min()), "z_max": int(zs.max()),
                "y_min": int(ys.min()), "y_max": int(ys.max()),
                "x_min": int(xs.min()), "x_max": int(xs.max()),
                "touches_border": 0, "tiles": "t0", "constituents": "",
            })


def _fixture(tmp: Path):
    stitch = tmp / "stage4_full_stitch"
    stitch.mkdir(parents=True, exist_ok=True)
    vol = _touch_volume()
    np.save(str(stitch / "global_labels_clean.npy"), vol)
    _write_prov(stitch / "label_provenance_clean.csv", vol)
    return stitch, vol


def _td():
    return [Path("x/tiles/t0")]


_SP = (2.0, 0.5, 0.5)


def test_global_passes_parity_with_extract_label_features(tmp_path):
    """THE GATE: every shared column the FeatureStage global passes emit must be
    byte-identical to reconcile.extract_label_features with the SAME flags — both
    run the SAME compute_* sequence on the SAME resident volume."""
    from packages.epicycle_scaledseg.reconcile.features import extract_label_features

    stitch, vol = _fixture(tmp_path)
    FeatureStage(
        input_dir=str(stitch), n_workers=1,
        adjacency=True, tissue_depth=True, lumen=True, territory=True,
        qc_score=True, voxel_size_um=_SP,
    ).run(_td(), n_gpus=0)
    cf = (pd.read_csv(stitch / "cell_features.csv")
          .sort_values("label").reset_index(drop=True))

    p = tmp_path / "lab.npy"
    np.save(p, vol)
    ref = (extract_label_features(
        str(p), n_workers=1, adjacency=True, tissue_depth=True, lumen=True,
        territory=True, qc_score=True, spacing=_SP, max_radius=5)
        .sort_values("label").reset_index(drop=True))

    # Same set of rows (labels) on both paths.
    np.testing.assert_array_equal(
        cf["label"].to_numpy(), ref["label"].to_numpy())

    # The signature columns of each opt-in global family MUST be in the shared set
    # (proves the families really ran on the FeatureStage path AND match reconcile).
    shared = [c for c in cf.columns if c in ref.columns]
    for sig in (_TISSUE_DEPTH_SIG, _TERRITORY_SIG, _LUMEN_SIG, _QC_SCORE_SIG):
        assert sig in shared, f"{sig} missing from the shared FeatureStage/reconcile columns"

    # Compare EVERY shared column byte-identically (numeric via assert_allclose;
    # object/string columns via direct equality, NaN-tolerant).
    mismatches = []
    for c in shared:
        a = cf[c].to_numpy()
        b = ref[c].to_numpy()
        if np.issubdtype(cf[c].dtype, np.number) and np.issubdtype(ref[c].dtype, np.number):
            try:
                np.testing.assert_allclose(
                    a.astype(float), b.astype(float), equal_nan=True)
            except AssertionError:
                mismatches.append(c)
        else:
            # object/string columns (e.g. seam_status) — compare elementwise,
            # treating NaN==NaN.
            for x, y in zip(a, b):
                if x != y and not (pd.isna(x) and pd.isna(y)):
                    mismatches.append(c)
                    break
    assert not mismatches, f"columns differ FeatureStage vs reconcile: {mismatches}"


def test_global_flags_off_by_default(tmp_path):
    """adjacency ALONE (no field flags) -> none of the global-field signature
    columns appear (each flag is independently default-off)."""
    stitch, _ = _fixture(tmp_path)
    FeatureStage(input_dir=str(stitch), n_workers=1, adjacency=True).run(_td(), n_gpus=0)
    cf = pd.read_csv(stitch / "cell_features.csv")
    for sig in (_TISSUE_DEPTH_SIG, _TERRITORY_SIG, _LUMEN_SIG, _QC_SCORE_SIG):
        assert sig not in cf.columns, f"{sig} present with the flag OFF"
    # And the fully-default stage (no flags at all) likewise omits them.
    stitch2, _ = _fixture(tmp_path / "b")
    FeatureStage(input_dir=str(stitch2), n_workers=1).run(_td(), n_gpus=0)
    cf2 = pd.read_csv(stitch2 / "cell_features.csv")
    for sig in (_TISSUE_DEPTH_SIG, _TERRITORY_SIG, _LUMEN_SIG, _QC_SCORE_SIG):
        assert sig not in cf2.columns


def test_manifest_records_global_flags(tmp_path):
    stitch, _ = _fixture(tmp_path)
    FeatureStage(
        input_dir=str(stitch), n_workers=1,
        adjacency=True, tissue_depth=True, lumen=True, territory=True,
        qc_score=True, voxel_size_um=_SP,
    ).run(_td(), n_gpus=0)
    man = json.loads((stitch / "feature_manifest.json").read_text())
    assert man["status"] == "complete"
    assert man["tissue_depth"] is True
    assert man["territory"] is True
    assert man["lumen"] is True
    assert man["qc_score"] is True
    assert man["field_downsample"] == 1
    # The params block carries them too.
    assert man["params"]["tissue_depth"] is True
    assert man["params"]["qc_score"] is True
    # The dynamic family breakdown picked up the global families.
    assert man["families"].get("tissue_depth")
    assert man["families"].get("watershed_territory")
    assert man["families"].get("lumen")
    assert man["families"].get("composite_score")


def test_manifest_records_flags_off(tmp_path):
    stitch, _ = _fixture(tmp_path)
    FeatureStage(input_dir=str(stitch), n_workers=1).run(_td(), n_gpus=0)
    man = json.loads((stitch / "feature_manifest.json").read_text())
    assert man["tissue_depth"] is False
    assert man["territory"] is False
    assert man["lumen"] is False
    assert man["qc_score"] is False
    assert man["field_downsample"] == 1


def test_tissue_depth_without_adjacency(tmp_path):
    """tissue_depth WITHOUT adjacency runs (the contact-dependent members —
    directional_contact / apical_junction — NaN out, but the EDT-field depth
    columns are real and it does NOT crash). And matches reconcile in that mode."""
    from packages.epicycle_scaledseg.reconcile.features import extract_label_features

    stitch, vol = _fixture(tmp_path)
    FeatureStage(
        input_dir=str(stitch), n_workers=1,
        tissue_depth=True, voxel_size_um=_SP,
    ).run(_td(), n_gpus=0)
    cf = (pd.read_csv(stitch / "cell_features.csv")
          .sort_values("label").reset_index(drop=True))
    # The EDT-field depth columns are present + finite (the field is real even with
    # no graph).
    assert _TISSUE_DEPTH_SIG in cf.columns
    assert np.isfinite(cf[_TISSUE_DEPTH_SIG].to_numpy()).any()
    # adjacency was off -> no contact graph columns.
    assert "neighbor_count" not in cf.columns

    p = tmp_path / "lab.npy"
    np.save(p, vol)
    ref = (extract_label_features(
        str(p), n_workers=1, tissue_depth=True, spacing=_SP)
        .sort_values("label").reset_index(drop=True))
    shared = [c for c in cf.columns if c in ref.columns]
    assert _TISSUE_DEPTH_SIG in shared
    for c in shared:
        if np.issubdtype(cf[c].dtype, np.number) and np.issubdtype(ref[c].dtype, np.number):
            np.testing.assert_allclose(
                cf[c].to_numpy(dtype=float), ref[c].to_numpy(dtype=float),
                equal_nan=True, err_msg=f"{c} differs (tissue_depth, no adjacency)")


def test_qc_score_without_adjacency(tmp_path):
    """qc_score WITHOUT adjacency scores on the labels-only subset (the neighbour
    columns are absent) — it runs + adds the column, never crashes."""
    stitch, _ = _fixture(tmp_path)
    FeatureStage(
        input_dir=str(stitch), n_workers=1,
        qc_score=True, voxel_size_um=_SP,
    ).run(_td(), n_gpus=0)
    cf = pd.read_csv(stitch / "cell_features.csv")
    assert _QC_SCORE_SIG in cf.columns
    # No graph columns (adjacency off), but the score still computed on a subset.
    assert "neighbor_count" not in cf.columns
    assert cf[_QC_SCORE_SIG].notna().any()
