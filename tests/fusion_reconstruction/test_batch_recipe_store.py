"""Tests for the persisted Batch-Refine recipe store (Qt-free, pure).

Covers ``save_named_recipe`` / ``load_named_recipes`` / ``delete_named_recipe``
round-trips, atomic-write durability, collision overwrite, bad-JSON tolerance,
and the ``_paths.batch_refine_recipes_path`` workspace resolution.
"""

from __future__ import annotations

import json

import pytest

from napari_epicycle.fusion_reconstruction import _batch_recipe_store as store
from napari_epicycle.fusion_reconstruction import _paths


_RECIPE_A = [
    {"method": "sdf", "params": {"sigma_z": 2.0, "sigma_xy": 1.5, "preserve_topology": True}},
    {"method": "fill_holes", "params": {"axis": "3D"}},
]
_RECIPE_B = [
    {"method": "dilate_erode", "params": {"radius": 2, "axis": "XY"}},
]


def _store_path(tmp_path):
    return _paths.batch_refine_recipes_path(tmp_path)


# --------------------------------------------------------------------------- #
# Path resolution
# --------------------------------------------------------------------------- #
def test_recipes_path_under_step3(tmp_path):
    p = _paths.batch_refine_recipes_path(tmp_path)
    assert p.name == "batch_refine_recipes.json"
    assert "step3" in str(p)
    # Should be under artifacts/step3/.
    assert p.parent == _paths.step3_root(tmp_path)


# --------------------------------------------------------------------------- #
# load on a fresh / missing workspace
# --------------------------------------------------------------------------- #
def test_load_missing_returns_empty(tmp_path):
    assert store.load_named_recipes(_store_path(tmp_path)) == {}


# --------------------------------------------------------------------------- #
# save -> load round trip
# --------------------------------------------------------------------------- #
def test_save_then_load_round_trip(tmp_path):
    path = _store_path(tmp_path)
    store.save_named_recipe(path, "Smooth + Fill", _RECIPE_A)
    loaded = store.load_named_recipes(path)
    assert loaded == {"Smooth + Fill": _RECIPE_A}


def test_save_creates_parent_dir(tmp_path):
    path = _store_path(tmp_path)
    assert not path.parent.exists()
    store.save_named_recipe(path, "R", _RECIPE_A)
    assert path.exists()


def test_save_two_recipes(tmp_path):
    path = _store_path(tmp_path)
    store.save_named_recipe(path, "A", _RECIPE_A)
    store.save_named_recipe(path, "B", _RECIPE_B)
    loaded = store.load_named_recipes(path)
    assert set(loaded.keys()) == {"A", "B"}
    assert loaded["A"] == _RECIPE_A
    assert loaded["B"] == _RECIPE_B


def test_save_collision_overwrites(tmp_path):
    path = _store_path(tmp_path)
    store.save_named_recipe(path, "X", _RECIPE_A)
    store.save_named_recipe(path, "X", _RECIPE_B)
    loaded = store.load_named_recipes(path)
    assert loaded == {"X": _RECIPE_B}


def test_save_returns_updated_store(tmp_path):
    path = _store_path(tmp_path)
    out = store.save_named_recipe(path, "A", _RECIPE_A)
    assert out == {"A": _RECIPE_A}


def test_save_strips_name_whitespace(tmp_path):
    path = _store_path(tmp_path)
    store.save_named_recipe(path, "  Trim Me  ", _RECIPE_A)
    loaded = store.load_named_recipes(path)
    assert "Trim Me" in loaded


def test_save_rejects_empty_name(tmp_path):
    path = _store_path(tmp_path)
    with pytest.raises(ValueError):
        store.save_named_recipe(path, "   ", _RECIPE_A)


def test_save_rejects_non_list_recipe(tmp_path):
    path = _store_path(tmp_path)
    with pytest.raises(ValueError):
        store.save_named_recipe(path, "A", {"not": "a list"})


# --------------------------------------------------------------------------- #
# delete
# --------------------------------------------------------------------------- #
def test_delete_round_trip(tmp_path):
    path = _store_path(tmp_path)
    store.save_named_recipe(path, "A", _RECIPE_A)
    store.save_named_recipe(path, "B", _RECIPE_B)
    store.delete_named_recipe(path, "A")
    loaded = store.load_named_recipes(path)
    assert set(loaded.keys()) == {"B"}


def test_delete_missing_is_noop(tmp_path):
    path = _store_path(tmp_path)
    store.save_named_recipe(path, "A", _RECIPE_A)
    out = store.delete_named_recipe(path, "ghost")
    assert out == {"A": _RECIPE_A}


def test_delete_last_leaves_empty_object(tmp_path):
    path = _store_path(tmp_path)
    store.save_named_recipe(path, "A", _RECIPE_A)
    store.delete_named_recipe(path, "A")
    # File still exists, but is an empty object.
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == {}
    assert store.load_named_recipes(path) == {}


# --------------------------------------------------------------------------- #
# bad-JSON tolerance
# --------------------------------------------------------------------------- #
def test_load_corrupt_json_returns_empty(tmp_path):
    path = _store_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json", encoding="utf-8")
    assert store.load_named_recipes(path) == {}


def test_load_empty_file_returns_empty(tmp_path):
    path = _store_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    assert store.load_named_recipes(path) == {}


def test_load_non_object_json_returns_empty(tmp_path):
    path = _store_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[1, 2, 3]", encoding="utf-8")
    assert store.load_named_recipes(path) == {}


def test_load_drops_non_list_entries(tmp_path):
    path = _store_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"good": _RECIPE_A, "bad": "not a list", "alsobad": 42}),
        encoding="utf-8",
    )
    loaded = store.load_named_recipes(path)
    assert loaded == {"good": _RECIPE_A}


def test_save_over_corrupt_store_recovers(tmp_path):
    """A save against a corrupt store overwrites it cleanly (no crash)."""
    path = _store_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("garbage", encoding="utf-8")
    store.save_named_recipe(path, "A", _RECIPE_A)
    assert store.load_named_recipes(path) == {"A": _RECIPE_A}


def test_save_leaves_no_tmp_files(tmp_path):
    path = _store_path(tmp_path)
    store.save_named_recipe(path, "A", _RECIPE_A)
    leftover = [p for p in path.parent.iterdir() if p.name.startswith(".batch_recipes_")]
    assert leftover == []
