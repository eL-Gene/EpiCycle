"""Tests for epicycle.utils.io module."""

import json
import numpy as np
import pytest
from pathlib import Path

from packages.epicycle_scaledseg.utils.io import (
    base_name,
    ensure_dir,
    now_iso,
    update_tile_config,
    load_array,
    pick_array_path,
)


def test_base_name():
    assert base_name("/data/tiles/tile_0_0.tif") == "tile_0_0"
    assert base_name("something.npy") == "something"
    assert base_name("/a/b/c/file.with.dots.tif") == "file"


def test_ensure_dir(tmp_path):
    d = tmp_path / "a" / "b" / "c"
    assert not d.exists()
    ensure_dir(d)
    assert d.is_dir()
    # idempotent
    ensure_dir(d)
    assert d.is_dir()


def test_now_iso():
    ts = now_iso()
    assert "T" in ts
    assert len(ts) == 19  # e.g. 2026-01-31T12:00:00


def test_update_tile_config_creates_new(tmp_path):
    cfg = tmp_path / "config.json"
    update_tile_config(cfg, {"key": "value"})
    data = json.loads(cfg.read_text())
    assert data == {"key": "value"}


def test_update_tile_config_merges_dicts(tmp_path):
    cfg = tmp_path / "config.json"
    update_tile_config(cfg, {"models": {"lateral": "a.model"}})
    update_tile_config(cfg, {"models": {"axial": "b.model"}})
    data = json.loads(cfg.read_text())
    assert data["models"] == {"lateral": "a.model", "axial": "b.model"}


def test_update_tile_config_overwrites_non_dict(tmp_path):
    cfg = tmp_path / "config.json"
    update_tile_config(cfg, {"version": 1})
    update_tile_config(cfg, {"version": 2})
    data = json.loads(cfg.read_text())
    assert data["version"] == 2


def test_load_array_npy(tmp_path):
    arr = np.random.rand(3, 4).astype(np.float64)
    p = tmp_path / "test.npy"
    np.save(p, arr)
    loaded = load_array(p)
    assert loaded.dtype == np.float32
    np.testing.assert_allclose(loaded, arr, atol=1e-6)


def test_load_array_npz_with_arr_key(tmp_path):
    arr = np.random.rand(2, 5).astype(np.float32)
    p = tmp_path / "test.npz"
    np.savez_compressed(p, arr=arr)
    loaded = load_array(p)
    np.testing.assert_array_equal(loaded, arr)


def test_load_array_npz_single_key(tmp_path):
    arr = np.ones((3, 3), dtype=np.float32)
    p = tmp_path / "test.npz"
    np.savez_compressed(p, data=arr)
    loaded = load_array(p)
    np.testing.assert_array_equal(loaded, arr)


def test_load_array_npz_multiple_keys_raises(tmp_path):
    p = tmp_path / "test.npz"
    np.savez_compressed(p, a=np.zeros(3), b=np.ones(3))
    with pytest.raises(ValueError, match="Unexpected npz"):
        load_array(p)


def test_pick_array_path_npz(tmp_path):
    (tmp_path / "tile_xy_flows.npz").touch()
    assert pick_array_path(tmp_path, "tile_xy_flows").suffix == ".npz"


def test_pick_array_path_npy(tmp_path):
    (tmp_path / "tile_xy_flows.npy").touch()
    assert pick_array_path(tmp_path, "tile_xy_flows").suffix == ".npy"


def test_pick_array_path_prefers_npz(tmp_path):
    (tmp_path / "tile_xy_flows.npz").touch()
    (tmp_path / "tile_xy_flows.npy").touch()
    assert pick_array_path(tmp_path, "tile_xy_flows").suffix == ".npz"


def test_pick_array_path_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        pick_array_path(tmp_path, "nonexistent")
