"""A re-extract keeps the tiers already saved in the patch's features CSV."""

import pandas as pd

from napari_epicycle.fusion_reconstruction._qc_extract_core import run_full_qc_extraction
from napari_epicycle.fusion_reconstruction.qc_features import load_patch_csv

from tests.fusion_reconstruction.test_qc_extract_subprocess import _write_workspace


def _seed(cfg, rows):
    pd.DataFrame(rows).to_csv(cfg["csv_path"], index=False)


def _run(cfg):
    run_full_qc_extraction(**cfg)
    return load_patch_csv(cfg["csv_path"])


def test_tiers_survive_and_removed_label_is_dropped(tmp_path):
    cfg = _write_workspace(tmp_path)
    _seed(cfg, [
        {"label": 1, "qc_tier": "good", "label_status": "active"},
        {"label": 2, "qc_tier": "bad", "label_status": "active"},
        {"label": 3, "qc_tier": "garbage", "label_status": "superseded"},
        {"label": 99, "qc_tier": "review", "label_status": "active"},
    ])
    df = _run(cfg)
    by = dict(zip(df["label"], df["qc_tier"]))
    assert by[1] == "good" and by[2] == "bad"
    assert pd.isna(by[3])                     # a superseded row never lends its tier
    assert 99 not in set(df["label"])         # a label that no longer exists is gone
    assert len(df) == df["label"].nunique()   # one row per label
    assert (df["label_status"] == "active").all()


def test_raw_backup_comes_from_the_fresh_frame(tmp_path):
    cfg = _write_workspace(tmp_path)
    _seed(cfg, [{"label": 1, "qc_tier": "good", "label_status": "active"}])
    _run(cfg)
    raw = pd.read_csv(tmp_path / "patch001_features_raw.csv")
    assert "qc_tier" not in raw.columns


def test_csv_without_tiers_adds_no_column(tmp_path):
    cfg = _write_workspace(tmp_path)
    _seed(cfg, [{"label": 1, "area": 5.0}, {"label": 2, "area": 6.0}])
    assert "qc_tier" not in _run(cfg).columns


def test_all_empty_tier_column_adds_no_tiers(tmp_path):
    cfg = _write_workspace(tmp_path)
    _seed(cfg, [{"label": 1, "qc_tier": None}, {"label": 2, "qc_tier": None}])
    assert "qc_tier" not in _run(cfg).columns


def test_new_labels_are_active(tmp_path):
    cfg = _write_workspace(tmp_path)
    _seed(cfg, [{"label": 1, "qc_tier": "good", "label_status": "active"}])
    df = _run(cfg)
    new = df[df["label"] != 1]
    assert len(new) > 0
    assert (new["label_status"] == "active").all()
    assert new["qc_tier"].isna().all()


def test_features_csv_is_written_once_and_holds_the_tiers(tmp_path, monkeypatch):
    from napari_epicycle.fusion_reconstruction import qc_features as core

    cfg = _write_workspace(tmp_path)
    _seed(cfg, [{"label": 1, "qc_tier": "good", "label_status": "active"}])
    real = core.save_patch_csv
    seen = []

    def spy(df, path):
        seen.append((str(path), "qc_tier" in df.columns))
        return real(df, path)

    monkeypatch.setattr(core, "save_patch_csv", spy)
    df = _run(cfg)
    writes = [has for p, has in seen if p == str(cfg["csv_path"])]
    assert writes == [True]
    assert dict(zip(df["label"], df["qc_tier"]))[1] == "good"


def test_raw_backup_has_no_tier_column_when_tiers_are_carried(tmp_path):
    cfg = _write_workspace(tmp_path)
    _seed(cfg, [{"label": 1, "qc_tier": "good", "label_status": "active"}])
    df = _run(cfg)
    assert "qc_tier" in df.columns
    raw = pd.read_csv(tmp_path / "patch001_features_raw.csv")
    assert "qc_tier" not in raw.columns
    assert len(raw) == len(df)


def test_a_failed_tier_merge_fails_the_run_and_keeps_the_old_file(tmp_path, monkeypatch):
    import pytest

    from napari_epicycle.fusion_reconstruction import _qc_extract_core as core

    cfg = _write_workspace(tmp_path)
    _seed(cfg, [{"label": 1, "qc_tier": "good", "label_status": "active"}])
    before = open(cfg["csv_path"], "rb").read()
    monkeypatch.setattr(core, "_read_saved_tiers", lambda p: "not a table")
    with pytest.raises(Exception):
        run_full_qc_extraction(**cfg)
    assert open(cfg["csv_path"], "rb").read() == before
