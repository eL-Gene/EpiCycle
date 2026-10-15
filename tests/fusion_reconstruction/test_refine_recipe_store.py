"""The Refine chain store.

The Batch store gains ``value_type=`` (keyword-only, default ``list``) and a
checked load; the Refine file lives at ``refine_chain_recipes_path``.
Invariant 9: Batch is untouched, a Refine delete keeps every other Refine
recipe, and a default-type load of the Refine file returns ``{}``.
"""

from __future__ import annotations

import json

import pytest

from napari_epicycle.fusion_reconstruction import _batch_recipe_store as store
from napari_epicycle.fusion_reconstruction import _migrate_step3_runs as m
from napari_epicycle.fusion_reconstruction import _paths


def _rec(n=1):
    return {"kind": "refine_chain", "version": 1, "created": "",
            "spacing_known": False, "spacing": [],
            "steps": [{"method": "fill_holes", "params": {"axis": "3D"}}] * n}


@pytest.fixture
def path(tmp_path):
    return _paths.refine_chain_recipes_path(tmp_path)


def test_the_path(tmp_path):
    p = _paths.refine_chain_recipes_path(tmp_path)
    assert p.name == "refine_chain_recipes.json"
    assert p.parent == _paths.step3_root(tmp_path)
    assert p.parent == _paths.batch_refine_recipes_path(tmp_path).parent
    assert "refine_chain_recipes_path" in _paths.__all__


def test_dict_save_load_delete(path):
    store.save_named_recipe(path, "tidy", _rec(3), value_type=dict)
    store.save_named_recipe(path, "other", _rec(1), value_type=dict)
    got = store.load_named_recipes(path, value_type=dict)
    assert set(got) == {"tidy", "other"}
    assert got["tidy"] == _rec(3)
    store.delete_named_recipe(path, "tidy", value_type=dict)
    got = store.load_named_recipes(path, value_type=dict)
    assert set(got) == {"other"}                   # the others are kept


def test_a_dict_save_refuses_a_list_and_vice_versa(path, tmp_path):
    with pytest.raises(ValueError):
        store.save_named_recipe(path, "x", [1], value_type=dict)
    with pytest.raises(ValueError):
        store.save_named_recipe(_paths.batch_refine_recipes_path(tmp_path),
                                "x", {"a": 1})


def test_value_type_is_keyword_only():
    import inspect

    for fn in (store.load_named_recipes, store.save_named_recipe,
               store.delete_named_recipe, store.load_named_recipes_checked):
        p = inspect.signature(fn).parameters["value_type"]
        assert p.kind is inspect.Parameter.KEYWORD_ONLY
    assert (inspect.signature(store.load_named_recipes)
            .parameters["value_type"].default is list)
    assert (inspect.signature(store.load_named_recipes_checked)
            .parameters["value_type"].default is inspect.Parameter.empty)


def test_WHY_the_keyword_exists(path):
    """A default-type save on the Refine file loads nothing and writes back
    only its own entry -- every recipe in the file is gone.  A default-type
    delete finds nothing and is a no-op."""
    store.save_named_recipe(path, "tidy", _rec(3), value_type=dict)
    store.save_named_recipe(path, "other", _rec(1), value_type=dict)
    assert store.load_named_recipes(path) == {}          # default type
    before = path.read_bytes()
    store.delete_named_recipe(path, "tidy")               # default: no-op
    assert path.read_bytes() == before
    store.save_named_recipe(path, "batchy", [{"method": "sdf"}])
    assert set(json.loads(path.read_text("utf-8"))) == {"batchy"}


def test_the_batch_default_is_unchanged(tmp_path):
    p = _paths.batch_refine_recipes_path(tmp_path)
    store.save_named_recipe(p, "A", [{"method": "sdf", "params": {}}])
    p.write_text(json.dumps({"A": [{"method": "sdf"}], "junk": {"x": 1}}),
                 encoding="utf-8")
    assert store.load_named_recipes(p) == {"A": [{"method": "sdf"}]}
    store.delete_named_recipe(p, "A")
    assert json.loads(p.read_text("utf-8")) == {}


class TestChecked:
    def test_missing_is_no_problem(self, path):
        assert store.load_named_recipes_checked(path, value_type=dict) == (
            {}, None)

    def test_empty_file_is_no_problem(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("   ", encoding="utf-8")
        assert store.load_named_recipes_checked(path, value_type=dict) == (
            {}, None)

    @pytest.mark.parametrize("text", ["{{{", "[1, 2]", "3", '"x"'])
    def test_unreadable_is_a_problem(self, path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        got, problem = store.load_named_recipes_checked(path, value_type=dict)
        assert got == {}
        assert problem == "The recipes file could not be read."
        # ...and the default load still tolerates it
        assert store.load_named_recipes(path, value_type=dict) == {}

    def test_a_wrong_type_entry_is_kept(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"tidy": _rec(1), "old": [1, 2]}),
                        encoding="utf-8")
        got, problem = store.load_named_recipes_checked(path, value_type=dict)
        assert problem is None
        assert got == {"tidy": _rec(1), "old": [1, 2]}
        # A dict-typed save of another name keeps it; a dict-typed delete
        # of the good one keeps it; and it can be deleted by its own name
        # (the menu offers it under Delete).
        store.save_named_recipe(path, "new", _rec(2), value_type=dict)
        store.delete_named_recipe(path, "tidy", value_type=dict)
        on_disk = json.loads(path.read_text("utf-8"))
        assert on_disk == {"old": [1, 2], "new": _rec(2)}
        store.delete_named_recipe(path, "old", value_type=dict)
        assert json.loads(path.read_text("utf-8")) == {"new": _rec(2)}


def test_unified_layout_conversion_leaves_the_file_at_the_root(tmp_path):
    from tests.fusion_reconstruction.test_migrate_step3_runs import (
        _seed_flat_workspace_with_siblings,
    )

    _seed_flat_workspace_with_siblings(tmp_path)
    p = _paths.refine_chain_recipes_path(tmp_path)
    store.save_named_recipe(p, "tidy", _rec(2), value_type=dict)
    before = p.read_bytes()
    plan = m.plan_migration(tmp_path)
    assert all(mv.src != p for mv in plan.moves)
    backup = m.create_backup(tmp_path)
    m.execute_migration(plan, backup)
    assert p.exists() and p.read_bytes() == before
    assert p.parent == _paths.step3_root(tmp_path)


def test_the_migration_comments_name_the_file():
    import inspect

    src = inspect.getsource(m)
    assert src.count("refine_chain_recipes.json") == 2
