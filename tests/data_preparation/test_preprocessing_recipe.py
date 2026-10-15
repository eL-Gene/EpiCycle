"""Brick 1a tests — the pure Step-1 preprocessing recipe data-model.

Covers the preprocessing op set: flatten (default →
``[gaussian, isotropic, background]``); the shared ``boundary_index``
re-segments every channel identically; to_json/from_json round-trip (incl. a
real ``json`` round-trip); ``normalize_recipe`` drops unknown + unimplemented
keys, clamps channel count + boundary, coerces params, and never raises on
garbage; ``default_recipe(2)`` is valid.

Pure — no napari / Qt.
"""

from __future__ import annotations

import json

import pytest

from packages.epicycle_preprocessing.recipe import (
    GeometryStep,
    IntensityStep,
    Recipe,
    default_geometry_config,
    default_recipe,
    flatten_for_channel,
    from_json,
    normalize_recipe,
    to_json,
)


def _keys(steps):
    return [s.key for s in steps]


# --------------------------------------------------------------------------- #
# flatten
# --------------------------------------------------------------------------- #

def test_flatten_gaussian_background_with_boundary_one():
    """The locked default order: [gaussian, background] + boundary 1 →
    [gaussian, isotropic, background]."""
    recipe = Recipe(
        channels=[[IntensityStep("gaussian"), IntensityStep("background")]],
        geometry=GeometryStep(key="isotropic", boundary_index=1),
    )
    assert _keys(flatten_for_channel(recipe, 0)) == [
        "gaussian", "isotropic", "background"
    ]


def test_flatten_default_recipe():
    recipe = default_recipe(1)
    assert _keys(flatten_for_channel(recipe, 0)) == ["gaussian", "isotropic"]


def test_flatten_no_geometry_is_just_the_ops():
    recipe = Recipe(
        channels=[[IntensityStep("gaussian"), IntensityStep("clahe")]],
        geometry=None,
    )
    assert _keys(flatten_for_channel(recipe, 0)) == ["gaussian", "clahe"]


def test_flatten_boundary_zero_puts_geometry_first():
    recipe = Recipe(
        channels=[[IntensityStep("gaussian")]],
        geometry=GeometryStep(key="isotropic", boundary_index=0),
    )
    assert _keys(flatten_for_channel(recipe, 0)) == ["isotropic", "gaussian"]


def test_flatten_boundary_beyond_len_clamps_to_end():
    recipe = Recipe(
        channels=[[IntensityStep("gaussian")]],
        geometry=GeometryStep(key="isotropic", boundary_index=99),
    )
    assert _keys(flatten_for_channel(recipe, 0)) == ["gaussian", "isotropic"]


def test_boundary_resegments_all_channels_identically():
    """Moving the shared geometry chip re-segments EVERY channel at the same
    relative index."""
    ch = lambda: [
        IntensityStep("gaussian"), IntensityStep("clahe"), IntensityStep("background")
    ]
    for boundary in range(0, 4):
        recipe = Recipe(
            channels=[ch(), ch()],
            geometry=GeometryStep(key="isotropic", boundary_index=boundary),
        )
        flat0 = _keys(flatten_for_channel(recipe, 0))
        flat1 = _keys(flatten_for_channel(recipe, 1))
        assert flat0 == flat1, f"channels diverged at boundary {boundary}"
        # isotropic must sit at the shared boundary index.
        assert flat0.index("isotropic") == boundary


# --------------------------------------------------------------------------- #
# (de)serialization
# --------------------------------------------------------------------------- #

def _rich_recipe():
    return Recipe(
        channels=[
            [IntensityStep("gaussian", {"sigma": 2.0}),
             IntensityStep("background", {"radius": 12})],
            [IntensityStep("clahe", {"clip": 0.02, "kernel": 32})],
        ],
        geometry=GeometryStep(
            key="isotropic",
            boundary_index=1,
            config={
                "mode": "aniso",
                "target_um": 0.5,
                "factors": (2.0, 1.0, 1.0),
                "method": "cubic",
            },
        ),
    )


def test_to_from_json_round_trip():
    recipe = _rich_recipe()
    assert from_json(to_json(recipe)) == recipe


def test_json_dumps_round_trip_preserves_recipe():
    """A real JSON serialize/deserialize (the persistence path) round-trips,
    including geometry config + per-channel lists (factors tuple survives)."""
    recipe = _rich_recipe()
    restored = from_json(json.loads(json.dumps(to_json(recipe))))
    assert restored == recipe
    assert restored.geometry.config["factors"] == (2.0, 1.0, 1.0)


def test_to_json_is_json_safe():
    payload = to_json(_rich_recipe())
    # No tuples survive into the serialized form.
    assert isinstance(payload["geometry"]["config"]["factors"], list)
    json.dumps(payload)  # must not raise


def test_geometry_none_round_trip():
    recipe = Recipe(channels=[[IntensityStep("gaussian")]], geometry=None)
    assert from_json(to_json(recipe)) == recipe


# --------------------------------------------------------------------------- #
# normalize_recipe
# --------------------------------------------------------------------------- #

def test_normalize_drops_unknown_keys():
    recipe = Recipe(
        channels=[[IntensityStep("gaussian"), IntensityStep("totally_bogus")]],
        geometry=None,
    )
    out = normalize_recipe(recipe, n_channels=1)
    assert _keys(out.channels[0]) == ["gaussian"]


def test_normalize_drops_unimplemented_keys():
    # `gamma` is a real registry op but implemented=False → dropped.
    recipe = Recipe(
        channels=[[IntensityStep("gaussian"), IntensityStep("gamma")]],
        geometry=None,
    )
    out = normalize_recipe(recipe, n_channels=1)
    assert _keys(out.channels[0]) == ["gaussian"]


def test_normalize_clamps_channel_count_down():
    recipe = Recipe(
        channels=[[IntensityStep("gaussian")],
                  [IntensityStep("clahe")],
                  [IntensityStep("background")]],
        geometry=None,
    )
    out = normalize_recipe(recipe, n_channels=2)
    assert len(out.channels) == 2
    assert _keys(out.channels[0]) == ["gaussian"]
    assert _keys(out.channels[1]) == ["clahe"]


def test_normalize_pads_channel_count_up():
    recipe = Recipe(channels=[[IntensityStep("gaussian")]], geometry=None)
    out = normalize_recipe(recipe, n_channels=3)
    assert len(out.channels) == 3
    assert out.channels[1] == [] and out.channels[2] == []


def test_normalize_clamps_boundary_index():
    recipe = Recipe(
        channels=[[IntensityStep("gaussian")]],
        geometry=GeometryStep(key="isotropic", boundary_index=99),
    )
    out = normalize_recipe(recipe, n_channels=1)
    # max channel length is 1 → boundary clamps to 1.
    assert out.geometry.boundary_index == 1


def test_normalize_clamps_negative_boundary_to_zero():
    recipe = Recipe(
        channels=[[IntensityStep("gaussian")]],
        geometry=GeometryStep(key="isotropic", boundary_index=-5),
    )
    out = normalize_recipe(recipe, n_channels=1)
    assert out.geometry.boundary_index == 0


def test_normalize_coerces_param_types():
    recipe = Recipe(
        channels=[[
            IntensityStep("gaussian", {"sigma": "2.5"}),      # str → float
            IntensityStep("clahe", {"clip": 1, "kernel": 32.0}),  # int→float, float→int
        ]],
        geometry=None,
    )
    out = normalize_recipe(recipe, n_channels=1)
    g, c = out.channels[0]
    assert isinstance(g.params["sigma"], float) and g.params["sigma"] == 2.5
    assert isinstance(c.params["clip"], float) and c.params["clip"] == 1.0
    assert isinstance(c.params["kernel"], int) and c.params["kernel"] == 32


def test_normalize_drops_unknown_params_and_fills_defaults():
    recipe = Recipe(
        channels=[[IntensityStep("gaussian", {"bogus": 1})]],
        geometry=None,
    )
    out = normalize_recipe(recipe, n_channels=1)
    params = out.channels[0][0].params
    assert set(params) == {"sigma"}          # unknown "bogus" dropped
    assert params["sigma"] == 1.5            # missing → registry default


def test_normalize_choice_out_of_range_falls_back_to_default():
    recipe = Recipe(
        channels=[[IntensityStep("bleach", {"model": "linear"})]],  # not a choice
        geometry=None,
    )
    out = normalize_recipe(recipe, n_channels=1)
    assert out.channels[0][0].params["model"] == "exp"


def test_normalize_drops_geometry_with_bad_key():
    recipe = Recipe(
        channels=[[IntensityStep("gaussian")]],
        geometry=GeometryStep(key="not_a_geometry_op", boundary_index=0),
    )
    out = normalize_recipe(recipe, n_channels=1)
    assert out.geometry is None


def test_normalize_coerces_geometry_config():
    recipe = Recipe(
        channels=[[IntensityStep("gaussian")]],
        geometry=GeometryStep(
            key="isotropic",
            boundary_index=1,
            config={"mode": "garbage", "target_um": "0.5",
                    "factors": [2, 1, 1], "method": "cubic"},
        ),
    )
    out = normalize_recipe(recipe, n_channels=1)
    cfg = out.geometry.config
    assert cfg["mode"] == "iso"                     # invalid mode → iso
    assert cfg["target_um"] == 0.5                  # "0.5" → 0.5
    assert cfg["factors"] == (2.0, 1.0, 1.0)        # list → tuple of floats
    assert cfg["method"] == "cubic"


def test_normalize_accepts_dict_input():
    recipe_dict = to_json(_rich_recipe())
    out = normalize_recipe(recipe_dict, n_channels=2)
    assert isinstance(out, Recipe)
    assert len(out.channels) == 2
    assert _keys(out.channels[0]) == ["gaussian", "background"]


@pytest.mark.parametrize("garbage", [
    None,
    "not a recipe",
    12345,
    [],
    {"channels": "not a list"},
    {"channels": [["bad", None, {"no": "key"}]], "geometry": "nope"},
    {"channels": [[{"key": "gaussian", "params": "not a dict"}]]},
])
def test_normalize_never_raises_on_garbage(garbage):
    out = normalize_recipe(garbage, n_channels=2)
    assert isinstance(out, Recipe)
    assert len(out.channels) == 2  # always clamped to n_channels


def test_normalize_garbage_yields_zero_channels_when_requested():
    out = normalize_recipe(None, n_channels=0)
    assert out.channels == [] and out.geometry is None


# --------------------------------------------------------------------------- #
# default_recipe / seeds
# --------------------------------------------------------------------------- #

def test_default_recipe_two_channels_is_valid():
    recipe = default_recipe(2)
    assert len(recipe.channels) == 2
    for channel in recipe.channels:
        assert _keys(channel) == ["gaussian"]
        assert channel[0].params == {"sigma": 1.5}
    assert recipe.geometry is not None
    assert recipe.geometry.key == "isotropic"
    assert recipe.geometry.boundary_index == 1
    # normalize is a no-op on the seed (already clean).
    assert normalize_recipe(recipe, n_channels=2) == recipe


def test_default_recipe_survives_json_round_trip():
    recipe = default_recipe(3)
    assert from_json(json.loads(json.dumps(to_json(recipe)))) == recipe


def test_default_geometry_config_shape():
    cfg = default_geometry_config()
    assert cfg["mode"] == "iso"
    assert cfg["factors"] == (1.0, 1.0, 1.0)
    assert cfg["method"] == "linear"
