"""Tests for the cleanup feature/triage table (epicycle.reconcile.features + return_triage)."""

import numpy as np
import pandas as pd
import pytest

from packages.epicycle_scaledseg.reconcile.cleanup import cleanup_volume
from packages.epicycle_scaledseg.reconcile.features import extract_label_features, assemble_cell_table, assemble_nucleus_table


def _toy():
    """3 cells: 1 clean, 1 merged (2 nuclei), 1 strung-victim; + an orphan nucleus."""
    Z, Y, X = 6, 12, 30
    cells = np.zeros((Z, Y, X), np.int32)
    cells[1:5, 1:11, 1:9] = 1            # clean
    cells[1:5, 1:11, 9:17] = 2           # abuts 1 -> strung victim
    cells[1:5, 1:11, 17:25] = 3          # merged (2 nuclei)
    nuclei = np.zeros((Z, Y, X), np.int32)
    nuclei[2:4, 3:8, 3:7] = 10           # in cell 1
    nuclei[2:4, 3:8, 7:12] = 20          # straddles 1|2, mostly in 2
    nuclei[2:4, 3:8, 18:20] = 30         # in cell 3
    nuclei[2:4, 3:8, 22:24] = 40         # also in cell 3 (merge)
    nuclei[2:4, 3:8, 26:28] = 50         # orphan
    return cells, nuclei


def test_return_triage_matches_report():
    cells, nuclei = _toy()
    oc, on, rep, tri = cleanup_volume(cells, nuclei, smooth=False, recover=False, return_triage=True)
    by_status = {}
    for r in tri:
        by_status.setdefault(r["status"], 0)
        by_status[r["status"]] += 1
    # the per-cell triage must reproduce the aggregate report exactly (for the surviving cells)
    for k in ("clean", "merged_cell", "no_nucleus_strung_victim"):
        assert by_status.get(k, 0) == rep.get(k, 0), (k, by_status, rep)
    # the merged cell row records both nuclei; the clean cell exactly one
    merged = next(r for r in tri if r["status"] == "merged_cell")
    assert merged["n_nuclei"] == 2 and set(merged["nucleus_ids"]) == {30, 40}
    clean = next(r for r in tri if r["status"] == "clean")
    assert clean["n_nuclei"] == 1 and clean["cell_voxels"] > 0


def test_extract_label_features_per_label(tmp_path):
    vol = np.zeros((8, 16, 16), np.int32)
    vol[1:5, 1:6, 1:6] = 1
    vol[1:5, 9:15, 9:15] = 2
    p = tmp_path / "labels.npy"; np.save(p, vol)
    df = extract_label_features(str(p), n_workers=1)        # serial (deterministic in test)
    assert set(df["label"]) == {1, 2}
    assert (df["area"] > 0).all()
    assert "sphericity" in df.columns


def test_extract_label_features_parallel_matches_serial(tmp_path):
    rng = np.random.RandomState(0)
    vol = np.zeros((10, 30, 30), np.int32)
    for i in range(1, 6):
        z, y, x = rng.randint(0, 6), rng.randint(0, 24), rng.randint(0, 24)
        vol[z:z + 3, y:y + 5, x:x + 5] = i
    p = tmp_path / "labels.npy"; np.save(p, vol)
    s = extract_label_features(str(p), n_workers=1).set_index("label")
    q = extract_label_features(str(p), n_workers=2).set_index("label")
    assert list(s.index) == list(q.index)
    assert np.allclose(s["area"].values, q["area"].values)


def test_assemble_cell_table_joins_status_and_nucleus_aggregates(tmp_path):
    cells, nuclei = _toy()
    oc, on, rep, tri = cleanup_volume(cells, nuclei, smooth=False, recover=False, return_triage=True)
    cp = tmp_path / "c.npy"; np.save(cp, oc)
    npth = tmp_path / "n.npy"; np.save(npth, on)
    cell_feats = extract_label_features(str(cp), n_workers=1)
    nuc_feats = extract_label_features(str(npth), n_workers=1)

    table = assemble_cell_table(tri, cell_feats, nuc_feats)
    assert "status" in table and "n_nuclei" in table and "nucleus_cell_frac" in table
    # the merged cell (id 3) has 2 nuclei + summed nucleus volume, ambiguous single-nucleus shape
    row3 = table.set_index("label").loc[3]
    assert row3["status"] == "merged_cell" and row3["n_nuclei"] == 2
    assert row3["nucleus_voxels"] > 0 and np.isnan(row3["nuc_sphericity"])
    # the clean cell (id 1) has a defined single-nucleus shape + a sensible fraction in (0,1]
    row1 = table.set_index("label").loc[1]
    assert row1["status"] == "clean" and row1["n_nuclei"] == 1
    assert not np.isnan(row1["nuc_sphericity"])
    assert 0 < row1["nucleus_cell_frac"] <= 1


def test_assemble_cell_table_keeps_cells_that_failed_extraction():
    # a cell present in triage but MISSING from cell_feats (extraction failed) must NOT vanish —
    # its status is the whole point of the worklist (audit finding D).
    triage = [
        {"cell_id": 1, "status": "clean", "n_nuclei": 1, "nucleus_ids": [10], "cell_voxels": 100},
        {"cell_id": 2, "status": "merged_cell", "n_nuclei": 2, "nucleus_ids": [20, 21], "cell_voxels": 200},
    ]
    cell_feats = pd.DataFrame({"label": [1], "area": [100.0], "sphericity": [0.8]})  # cell 2 absent
    nuc_feats = pd.DataFrame({"label": [10, 20], "area": [30.0, 40.0], "sphericity": [0.9, 0.9],
                              "aspect_ratio": [1.1, 1.2]})                            # nucleus 21 absent
    table = assemble_cell_table(triage, cell_feats, nuc_feats).set_index("label")
    assert 2 in table.index                                  # cell 2 survived the join
    assert table.loc[2, "status"] == "merged_cell"
    assert not bool(table.loc[2, "features_extracted"])      # but its morphology is NaN/flagged
    # cell 2 counted 2 nuclei but only 1 (id 20) has features -> flagged incomplete (audit finding G)
    assert not bool(table.loc[2, "nucleus_feats_complete"])
    assert bool(table.loc[1, "nucleus_feats_complete"])


def test_assemble_nucleus_table_tags_owning_cell(tmp_path):
    cells, nuclei = _toy()
    oc, on, rep, tri = cleanup_volume(cells, nuclei, smooth=False, recover=False, return_triage=True)
    npth = tmp_path / "n.npy"; np.save(npth, on)
    nuc_feats = extract_label_features(str(npth), n_workers=1)
    ntab = assemble_nucleus_table(nuc_feats, tri)
    m = ntab.set_index("label")["cell_id"]
    assert int(m.loc[10]) == 1            # nucleus 10 -> cell 1
    assert int(m.loc[30]) == 3 and int(m.loc[40]) == 3   # both merged nuclei -> cell 3
    assert 50 not in m.index              # orphan nucleus was dropped, never in the consensus


def test_optional_names_are_attribute_errors_when_the_module_is_absent(monkeypatch):
    import packages.epicycle_scaledseg.reconcile as reconcile
    monkeypatch.setitem(reconcile._LAZY, "apply_edits", "no_such_module_here")
    monkeypatch.delitem(reconcile.__dict__, "apply_edits", raising=False)
    assert hasattr(reconcile, "apply_edits") is False
    with pytest.raises(AttributeError, match="not part of this release"):
        reconcile.apply_edits
    with pytest.raises(AttributeError):
        reconcile.not_a_name_at_all
    # `import *` only offers names that exist
    ns = {}
    exec("from packages.epicycle_scaledseg.reconcile import *", ns)
    assert "cleanup_volume" in ns and "apply_edits" not in ns
