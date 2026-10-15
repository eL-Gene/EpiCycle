"""``ReshapeSpec`` validation + serialisation.

The whole point of the dataclass is that it RAISES on an unknown value.  The
sandbox's ``_alias`` dict and ``seeds_auto`` fallthrough turned a typo into a
silently different algorithm; these tests pin the opposite behaviour.

Pure pytest -- no Qt, no napari.
"""

from __future__ import annotations

import json

import pytest

from packages.epicycle_annotator.reshape import (
    DEFAULT_PARAMS,
    IMAGE_DERIVED_WALLS,
    REFINES,
    SEEDS,
    SOLVERS,
    TARGETS,
    WALLS,
    ReshapeSpec,
)


def _spec(**kw):
    base = dict(target="label", seeds="labels", walls="membrane")
    base.update(kw)
    return ReshapeSpec(**base)


# --------------------------------------------------------------------------- #
# enum validation -- every slot, no fallbacks
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("field,bad", [
    ("target", "nonsense"),
    ("seeds", "seeds_auto"),          # the sandbox name must NOT be accepted
    ("walls", "cadherin"),
    ("solver", "magic"),
])
def test_unknown_slot_value_raises_naming_field_and_options(field, bad):
    with pytest.raises(ValueError) as exc:
        _spec(**{field: bad})
    message = str(exc.value)
    assert field in message
    assert bad in message
    # the error must list what IS allowed
    allowed = {"target": TARGETS, "seeds": SEEDS, "walls": WALLS,
               "solver": SOLVERS}[field]
    assert allowed[0] in message


def test_every_declared_enum_value_constructs():
    for seeds in SEEDS:
        # `nuclei_masks` exists only for the Add region target (Add rebuild
        # ); every other marker source still builds on `label`.
        _spec(seeds=seeds,
              target="region" if seeds == "nuclei_masks" else "label")
    for walls in WALLS:
        _spec(walls=walls, arbiter=walls in IMAGE_DERIVED_WALLS)
    for solver in SOLVERS:
        _spec(solver=solver)
    for target in TARGETS:
        _spec(target=target)


def test_unknown_refine_stage_raises():
    with pytest.raises(ValueError, match="refine"):
        _spec(refine=("guided", "polish"))


def test_duplicate_refine_stage_raises():
    with pytest.raises(ValueError, match="duplicate"):
        _spec(refine=("guided", "guided"))


def test_refine_normalised_to_tuple():
    assert _spec(refine=list(REFINES)).refine == tuple(REFINES)


# --------------------------------------------------------------------------- #
# params -- a knob that reaches nothing must not exist
# --------------------------------------------------------------------------- #
def test_unknown_param_key_raises():
    with pytest.raises(ValueError, match="unknown key"):
        _spec(params={"connectivity": 2})


def test_missing_params_filled_from_defaults():
    for arbiter in (False, True):
        spec = _spec(arbiter=arbiter)
        for key, value in DEFAULT_PARAMS.items():
            assert spec.params[key] == value, key


def test_get_rejects_unknown_key():
    with pytest.raises(KeyError):
        _spec().get("connectivity")


def test_with_params_revalidates():
    spec = _spec()
    assert spec.with_params(clip=0.05).params["clip"] == 0.05
    with pytest.raises(ValueError):
        spec.with_params(bogus=1)


def test_spec_is_frozen():
    spec = _spec()
    with pytest.raises(Exception):
        spec.target = "region"


# --------------------------------------------------------------------------- #
# the arbiter's two hard preconditions
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("walls", ["distance", "skeleton", "none"])
def test_arbiter_refused_on_shape_derived_walls(walls):
    """Arbitrating a shape-derived cut against its own shape is circular."""
    with pytest.raises(ValueError) as exc:
        _spec(walls=walls, arbiter=True)
    assert "arbiter" in str(exc.value)
    assert walls in str(exc.value)


@pytest.mark.parametrize("walls", sorted(IMAGE_DERIVED_WALLS))
def test_arbiter_allowed_on_image_derived_walls(walls):
    assert _spec(walls=walls, arbiter=True).arbiter_available is True


@pytest.mark.parametrize("walls", ["distance", "skeleton", "none"])
def test_shape_derived_walls_work_with_arbiter_off(walls):
    spec = _spec(walls=walls, arbiter=False)
    assert spec.arbiter is False
    assert spec.arbiter_available is False


@pytest.mark.parametrize("given", ["none", "", "gamma", "zcorrect+denoise"])
@pytest.mark.parametrize("arbiter", [False, True])
def test_the_wall_chain_is_left_exactly_as_the_caller_wrote_it(given, arbiter):
    """An earlier draft FORCED ``zdepth_correct`` into the wall chain whenever
    the arbiter was on.  That rule was **dropped** on 2026-09-08:
    ``zdepth_correct`` matches every z-slice's mean to the brightest slice's,
    and a membrane occupying one z-plane IS the brightest slice -- so the
    mandate absorbed exactly the evidence the arbiter exists to weigh, and a
    real z-stacked merge came back "dividing".

    The chain is now whatever the caller asked for, arbiter or not."""
    spec = _spec(arbiter=arbiter, params={"wall_preproc": given})
    assert spec.params["wall_preproc"] == given


def test_the_arbiter_is_off_by_default():
    """A cut the user asked for is not second-guessed unless they opt in."""
    assert ReshapeSpec(target="label", seeds="labels", walls="membrane").arbiter is False
    assert ReshapeSpec.from_dict(
        {"target": "label", "seeds": "labels", "walls": "membrane"}
    ).arbiter is False


def test_zdepth_correct_is_still_available_as_an_ordinary_step():
    """Dropped as a mandate, kept as an option -- exactly as in the original
    recipe."""
    spec = _spec(arbiter=True, params={"wall_preproc": "zcorrect+gamma"})
    assert set(spec.params["wall_preproc"].split("+")) == {"zcorrect", "gamma"}


# --------------------------------------------------------------------------- #
# serialisability -- presets, derived-run `parameters`, batch recipes
# --------------------------------------------------------------------------- #
def test_json_round_trip_is_lossless():
    spec = _spec(seeds="hmax", walls="frangi", solver="random_walker",
                 refine=("guided", "merge_small"),
                 params={"seed_channel": "nuclei", "wall_channel": "cadherin",
                         "h": 0.3, "min_size": 40})
    again = ReshapeSpec.from_json(spec.to_json())
    assert again == spec
    assert again.to_dict() == spec.to_dict()
    json.loads(spec.to_json())          # genuinely JSON, not a repr


def test_from_dict_rejects_unknown_fields():
    data = _spec().to_dict()
    data["preset"] = "P1"
    with pytest.raises(ValueError, match="unknown field"):
        ReshapeSpec.from_dict(data)


def test_from_dict_revalidates_enums():
    data = _spec().to_dict()
    data["seeds"] = "seeds_auto"
    with pytest.raises(ValueError):
        ReshapeSpec.from_dict(data)


# --------------------------------------------------------------------------- #
# the Add region target
# --------------------------------------------------------------------------- #
from packages.epicycle_annotator.reshape.spec import (  # noqa: E402
    ADD_BOX_BUDGET,
    ADD_WALL_KINDS,
    ReshapeResult,
)

_REGION_KEYS = {
    "outside": "faces+dark",
    "dark_otsu_factor": 0.5,
    "dark_clearance_um": 2.0,
    "core_erode_um": 1.0,
    "click_radius_um": 2.0,
    "box_budget": 4_000_000,
}


def test_the_add_box_budget_is_four_million_voxels():
    assert ADD_BOX_BUDGET == 4_000_000


@pytest.mark.parametrize("key,value", sorted(_REGION_KEYS.items()))
def test_the_six_region_keys_join_the_defaults(key, value):
    assert DEFAULT_PARAMS[key] == value
    assert _spec().params[key] == value


def test_the_ridge_default_is_unchanged_add_passes_it_off_itself():
    """Decision 17: Add passes ``ridge=False``; Reshape keeps the ridge."""
    assert DEFAULT_PARAMS["ridge"] is True


def test_nuclei_masks_is_appended_last():
    """Item order is a wire format: the new source goes at the END."""
    assert SEEDS[-1] == "nuclei_masks"
    assert SEEDS[:-1] == ("paint", "peaks", "hmax", "hmin", "blobs_log",
                          "blobs_dog", "threshcc", "labels", "distance")


@pytest.mark.parametrize("outside", ["faces+dark", "faces"])
def test_both_outside_values_construct(outside):
    spec = _spec(target="region", params={"outside": outside})
    assert spec.params["outside"] == outside


@pytest.mark.parametrize("bad", ["dark", "", None, "faces+Dark"])
def test_an_unknown_outside_raises(bad):
    with pytest.raises(ValueError, match="outside"):
        _spec(target="region", params={"outside": bad})


@pytest.mark.parametrize("key,lo,hi", [
    ("dark_otsu_factor", 0.05, 1.0),
    ("dark_clearance_um", 0.0, 10.0),
    ("core_erode_um", 0.0, 5.0),
    ("click_radius_um", 0.5, 5.0),
])
def test_region_knob_ranges(key, lo, hi):
    assert _spec(target="region", params={key: lo}).params[key] == lo
    assert _spec(target="region", params={key: hi}).params[key] == hi
    for bad in (lo - 0.01, hi + 0.01):
        with pytest.raises(ValueError, match=key):
            _spec(target="region", params={key: bad})


def test_the_box_budget_must_be_at_least_one():
    assert _spec(target="region", params={"box_budget": 1}).params["box_budget"] == 1
    with pytest.raises(ValueError, match="box_budget"):
        _spec(target="region", params={"box_budget": 0})


def test_nuclei_masks_is_refused_on_the_label_target():
    with pytest.raises(ValueError, match="nuclei_masks"):
        _spec(target="label", seeds="nuclei_masks")
    with pytest.raises(ValueError, match="nuclei_masks"):
        _spec(target="layer_label", seeds="nuclei_masks")
    assert _spec(target="region", seeds="nuclei_masks").seeds == "nuclei_masks"


def test_add_wall_kinds_are_the_image_walls_without_hybrid():
    assert ADD_WALL_KINDS == IMAGE_DERIVED_WALLS - {"hybrid"}


def test_hybrid_is_refused_on_the_region_target():
    with pytest.raises(ValueError, match="hybrid"):
        _spec(target="region", walls="hybrid")
    # still fine on the label target
    assert _spec(target="label", walls="hybrid").walls == "hybrid"


@pytest.mark.parametrize("walls", sorted(set(WALLS) - ADD_WALL_KINDS))
def test_a_region_spec_needs_an_add_wall_kind(walls):
    with pytest.raises(ValueError, match="walls"):
        _spec(target="region", walls=walls)


@pytest.mark.parametrize("walls", sorted(ADD_WALL_KINDS))
def test_every_add_wall_kind_constructs_on_region(walls):
    assert _spec(target="region", walls=walls).walls == walls


def test_the_new_result_fields_default_so_old_constructions_hold():
    r = ReshapeResult()
    assert r.n_dropped_dark == 0
    assert r.n_dropped_thin == 0
    assert r.n_nuclei_slivers == 0
    assert r.n_markers_over_kept == 0
    assert r.n_fragments_dropped == 0
    assert r.unfilled_tissue_fraction is None
    assert r.faint_pairs == []
    assert r.dark_threshold is None


def test_faint_pairs_is_not_shared_between_instances():
    a, b = ReshapeResult(), ReshapeResult()
    a.faint_pairs.append((1, 2, 0.1))
    assert b.faint_pairs == []


def test_a_region_spec_round_trips_through_json():
    spec = _spec(target="region", seeds="nuclei_masks",
                 params={"outside": "faces", "dark_otsu_factor": 0.35,
                         "ridge": False})
    assert ReshapeSpec.from_json(spec.to_json()) == spec


# --------------------------------------------------------------------------- #
# the region target refuses the slots it does not read (B1 fix pass, L-A3)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("solver", [s for s in SOLVERS if s != "watershed"])
def test_a_region_spec_refuses_a_solver_other_than_watershed(solver):
    with pytest.raises(ValueError, match="solver"):
        _spec(target="region", solver=solver)
    assert _spec(target="label", solver=solver).solver == solver


@pytest.mark.parametrize("stage", REFINES)
def test_a_region_spec_refuses_refine(stage):
    with pytest.raises(ValueError, match="refine"):
        _spec(target="region", refine=(stage,))
    assert _spec(target="label", refine=(stage,)).refine == (stage,)


def test_a_region_spec_refuses_the_arbiter():
    with pytest.raises(ValueError, match="arbiter"):
        _spec(target="region", arbiter=True)
    assert _spec(target="label", arbiter=True).arbiter is True


@pytest.mark.parametrize("seeds", [s for s in SEEDS
                                   if s not in ("labels", "nuclei_masks", "paint")])
def test_a_region_spec_refuses_a_marker_source_it_has_not_got(seeds):
    with pytest.raises(ValueError, match="region"):
        _spec(target="region", seeds=seeds)


def test_the_region_marker_sources_construct():
    from packages.epicycle_annotator.reshape.spec import REGION_SEEDS

    assert REGION_SEEDS == ("labels", "nuclei_masks", "paint")
    for seeds in REGION_SEEDS:
        assert _spec(target="region", seeds=seeds).seeds == seeds
