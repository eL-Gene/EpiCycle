"""Tests for batch-refine recipe serialization (Qt-free, pure).

Covers ``RefineStep.to_dict`` / ``from_dict`` + the module helpers
``recipe_to_json`` / ``recipe_from_json`` — round-trip fidelity across all
three methods, validation-on-load, and clear errors on bad input.
"""

from __future__ import annotations

import pytest

from packages.epicycle_annotator.batch_refine import (
    RefineStep,
    recipe_from_json,
    recipe_to_json,
)


# --------------------------------------------------------------------------- #
# RefineStep.to_dict / from_dict
# --------------------------------------------------------------------------- #
def test_step_to_dict_shape():
    step = RefineStep(method="sdf", params={"sigma_z": 2.0, "sigma_xy": 1.5})
    d = step.to_dict()
    assert d == {"method": "sdf", "params": {"sigma_z": 2.0, "sigma_xy": 1.5}}


def test_step_to_dict_copies_params():
    """The serialized params dict must not alias the live step's params."""
    params = {"axis": "XY", "radius": 2}
    step = RefineStep(method="dilate_erode", params=params)
    d = step.to_dict()
    d["params"]["radius"] = 99
    assert step.params["radius"] == 2  # unchanged


def test_step_from_dict_validates_and_builds():
    d = {"method": "fill_holes", "params": {"axis": "3D"}}
    step = RefineStep.from_dict(d)
    assert isinstance(step, RefineStep)
    assert step.method == "fill_holes"
    assert step.params == {"axis": "3D"}


def test_step_from_dict_missing_params_defaults_empty():
    step = RefineStep.from_dict({"method": "sdf"})
    assert step.method == "sdf"
    assert step.params == {}


@pytest.mark.parametrize(
    "step",
    [
        RefineStep(method="sdf", params={"sigma_z": 3.0, "sigma_xy": 1.5, "preserve_topology": False}),
        RefineStep(method="dilate_erode", params={"radius": -3, "axis": "XZ"}),
        RefineStep(method="fill_holes", params={"axis": "YZ"}),
    ],
)
def test_step_round_trip(step):
    again = RefineStep.from_dict(step.to_dict())
    assert again.method == step.method
    assert again.params == step.params


def test_step_from_dict_rejects_unknown_method():
    with pytest.raises(ValueError):
        RefineStep.from_dict({"method": "bogus", "params": {}})


def test_step_from_dict_rejects_out_of_range_param():
    with pytest.raises(ValueError):
        RefineStep.from_dict({"method": "sdf", "params": {"sigma_z": 99.0}})


def test_step_from_dict_rejects_non_dict():
    with pytest.raises(ValueError):
        RefineStep.from_dict(["not", "a", "dict"])


def test_step_from_dict_rejects_missing_method():
    with pytest.raises(ValueError):
        RefineStep.from_dict({"params": {}})


def test_step_from_dict_rejects_non_dict_params():
    with pytest.raises(ValueError):
        RefineStep.from_dict({"method": "sdf", "params": [1, 2, 3]})


# --------------------------------------------------------------------------- #
# recipe_to_json / recipe_from_json
# --------------------------------------------------------------------------- #
def test_recipe_round_trip_all_methods():
    recipe = [
        RefineStep(method="sdf", params={"sigma_z": 2.5, "sigma_xy": 1.0, "preserve_topology": True}),
        RefineStep(method="dilate_erode", params={"radius": 2, "axis": "3D"}),
        RefineStep(method="fill_holes", params={"axis": "XY"}),
    ]
    data = recipe_to_json(recipe)
    assert isinstance(data, list)
    assert all(isinstance(d, dict) for d in data)
    again = recipe_from_json(data)
    assert [s.method for s in again] == ["sdf", "dilate_erode", "fill_holes"]
    assert [s.params for s in again] == [s.params for s in recipe]


def test_recipe_to_json_is_json_safe():
    import json

    recipe = [RefineStep(method="sdf", params={"sigma_z": 1.5, "sigma_xy": 1.5})]
    data = recipe_to_json(recipe)
    # Must survive a real json dump/load round trip.
    restored = json.loads(json.dumps(data))
    again = recipe_from_json(restored)
    assert again[0].params["sigma_z"] == pytest.approx(1.5)


def test_recipe_from_json_empty_list():
    assert recipe_from_json([]) == []


def test_recipe_from_json_rejects_non_list():
    with pytest.raises(ValueError):
        recipe_from_json({"method": "sdf"})


def test_recipe_from_json_raises_on_bad_step():
    with pytest.raises(ValueError):
        recipe_from_json([{"method": "sdf"}, {"method": "nope"}])
