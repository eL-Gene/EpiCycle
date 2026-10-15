"""``reshape_region`` / ``apply_reshape_to_volume`` -- the R0 engine.

Covers the R0 test list: a two-cell blob with a wall splits; zero
seeds refuses with a message and no ``seeds_auto`` fallback; seeds outside the
region are dropped and counted; the safe-write never touches a neighbour;
``merge_small_pieces`` never absorbs a protected seed-backed cell; a guided
refine that would erase a cell keeps the unrefined split; u-Segment3D absent
degrades with a message; anisotropic spacing changes the sigmas; the pre-flight
size guard; determinism; and ``cancel_event`` between stages.

Pure pytest -- no Qt, no napari.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from packages.epicycle_annotator.reshape import (
    DEFAULT_SIZE_BUDGET,
    MessageSink,
    ReshapeSpec,
    apply_reshape_to_volume,
    axis_sigma,
    build_elevation,
    guided_filter_refine,
    merge_small_pieces,
    merge_to_count,
    physical_to_voxel_scalar,
    refine_result_guided,
    reshape_region,
    seeds_distance_maxima,
    seeds_from_labels,
    solve,
)
from packages.epicycle_annotator.reshape import optional_deps, postprocess, solvers

from .reshape_fixtures import solid_cell, two_columns

ISO = (1.0, 1.0, 1.0)
ANISO = (5.0, 1.0, 1.0)


def _spec(**kw):
    base = dict(target="label", seeds="labels", walls="intensity",
                params={"wall_channel": "mem", "clip": 0.0}, arbiter=False)
    params = dict(base["params"])
    params.update(kw.pop("params", {}))
    base.update(kw)
    base["params"] = params
    return ReshapeSpec(**base)


def _run(spec=None, contrast=0.4, *, spacing=ISO, seed_layer=None, **kw):
    labels, guide, mem, region = two_columns(contrast)
    spec = spec if spec is not None else _spec()
    layer = guide if seed_layer is None else seed_layer
    return reshape_region(labels, {"mem": mem}, spec, target_id=7,
                          seed_layer=layer, spacing=spacing, **kw), labels, region


# --------------------------------------------------------------------------- #
# 1. the happy path
# --------------------------------------------------------------------------- #
def test_two_cell_blob_with_a_wall_splits_into_two():
    out, _labels, region = _run()
    assert out.status == "ok"
    assert out.ok
    assert out.n_pieces == 2
    assert out.n_seeds == 2
    assert out.result.shape == region[out.roi].shape
    assert len(out.piece_sizes) == 2
    assert out.coverage == pytest.approx(1.0, abs=0.01)


def test_the_partition_stays_inside_the_target_footprint():
    out, labels, _ = _run()
    region = (labels[out.roi] == 7)
    assert not (out.result[~region]).any(), "the solve leaked outside the target"


def test_result_reports_the_seed_source_in_words():
    out, _, _ = _run()
    assert "trusted mask" in out.seed_source


def test_one_seed_is_legal_and_reports_one_piece():
    labels, guide, mem, _ = two_columns(0.4)
    guide[guide == 2] = 0                       # keep a single seed
    out = reshape_region(labels, {"mem": mem}, _spec(), target_id=7,
                         seed_layer=guide, spacing=ISO)
    assert out.status == "ok"
    assert out.n_seeds == 1
    assert out.n_pieces == 1
    assert any("1 piece" in m for m in out.messages)


def test_keep_seed_ids_cherry_picks():
    out, _, _ = _run(keep_seed_ids=[1])
    assert out.n_seeds == 1
    assert out.n_pieces == 1


def test_intermediates_are_opt_in():
    out, _, _ = _run()
    assert out.intermediates == {}
    out2, _, _ = _run(keep_intermediates=True)
    assert set(out2.intermediates) == {"clahe", "ridge", "elevation_raw"}
    assert out2.intermediates["elevation_raw"] is not None


# --------------------------------------------------------------------------- #
# 2. refusals -- an answer, never a fallback
# --------------------------------------------------------------------------- #
def test_zero_seeds_refuses_with_a_message_and_no_unseeded_watershed():
    labels, _guide, mem, _ = two_columns(0.4)
    empty = np.zeros_like(labels)
    out = reshape_region(labels, {"mem": mem}, _spec(), target_id=7,
                         seed_layer=empty, spacing=ISO)
    assert out.status == "refused"
    assert out.refusal_reason == "no-seeds"
    assert out.result is None, "a refusal must not carry a partition"
    assert out.n_pieces == 0
    # M7: the engine speaks the panel's vocabulary now -- "marker",
    # not "seed".  The BEHAVIOUR this test is about (a refusal, and no
    # unseeded watershed behind it) is unchanged.
    assert any("no markers" in m.lower() for m in out.messages)
    assert any("paint one" in m.lower() for m in out.messages)


def test_zero_seeds_does_not_silently_route_to_distance_maxima(monkeypatch):
    """The sandbox's `seeds_auto` fallthrough must be structurally absent."""
    called = []
    import packages.epicycle_annotator.reshape.engine as engine_mod

    def _spy(*a, **k):
        called.append(1)
        return np.zeros((1, 1, 1), np.int32)

    monkeypatch.setattr(engine_mod.seed_ops, "seeds_distance_maxima", _spy)
    labels, _guide, mem, _ = two_columns(0.4)
    out = reshape_region(labels, {"mem": mem}, _spec(), target_id=7,
                         seed_layer=np.zeros_like(labels), spacing=ISO)
    assert out.status == "refused"
    assert called == [], "a failed seed source fell through to distance maxima"


def test_a_missing_target_label_refuses():
    labels, guide, mem, _ = two_columns(0.4)
    out = reshape_region(labels, {"mem": mem}, _spec(), target_id=1234,
                         seed_layer=guide, spacing=ISO)
    assert out.status == "refused"
    assert out.refusal_reason == "no-target"
    assert "1234" in " ".join(out.messages)


def test_target_id_is_required():
    labels, guide, mem, _ = two_columns(0.4)
    with pytest.raises(ValueError, match="target_id"):
        reshape_region(labels, {"mem": mem}, _spec(), seed_layer=guide)


def test_the_layer_label_shape_is_explicitly_not_implemented():
    """Narrowed by the Add rebuild: ``target="region"`` is now
    implemented (``reshape/region.py``) and dispatched BEFORE this
    refusal; ``layer_label`` still raises."""
    labels, guide, mem, _ = two_columns(0.4)
    spec = ReshapeSpec(target="layer_label", seeds="labels", walls="intensity",
                       params={"wall_channel": "mem"}, arbiter=False)
    with pytest.raises(NotImplementedError, match="R0"):
        reshape_region(labels, {"mem": mem}, spec, target_id=7, seed_layer=guide)


def test_a_non_spec_object_is_rejected():
    labels, guide, mem, _ = two_columns(0.4)
    with pytest.raises(TypeError, match="ReshapeSpec"):
        reshape_region(labels, {"mem": mem}, {"target": "label"}, target_id=7,
                       seed_layer=guide)


# --------------------------------------------------------------------------- #
# 3. seeds outside the target are dropped AND counted
# --------------------------------------------------------------------------- #
def test_seeds_outside_the_target_are_dropped_and_counted():
    labels, guide, mem, _ = two_columns(0.4)
    # inside the padded ROI, outside the cell footprint
    guide[:, 4, 7] = 11
    guide[:, 4, 9] = 12
    guide[:, 4, 11] = 13
    out = reshape_region(labels, {"mem": mem}, _spec(), target_id=7,
                         seed_layer=guide, spacing=ISO)
    assert out.n_seeds_found == 5
    assert out.n_seeds == 2
    assert out.n_seeds_dropped == 3
    assert any("3 of 5 markers lie outside" in m for m in out.messages)


def test_a_seed_outside_the_target_never_pulls_the_partition_outward():
    labels, guide, mem, _ = two_columns(0.4)
    guide[:, 4, 7] = 11
    out = reshape_region(labels, {"mem": mem}, _spec(), target_id=7,
                         seed_layer=guide, spacing=ISO)
    region = labels[out.roi] == 7
    assert not out.result[~region].any()


def test_a_missing_seed_layer_raises_for_the_layer_backed_kinds():
    labels, _guide, mem, _ = two_columns(0.4)
    with pytest.raises(ValueError, match="seed_layer"):
        reshape_region(labels, {"mem": mem}, _spec(), target_id=7, spacing=ISO)


# --------------------------------------------------------------------------- #
# 4. channels -- named, validated, 3D only
# --------------------------------------------------------------------------- #
def test_an_unknown_wall_channel_raises_naming_what_is_available():
    labels, guide, mem, _ = two_columns(0.4)
    spec = _spec(params={"wall_channel": "cadherin"})
    with pytest.raises(ValueError) as exc:
        reshape_region(labels, {"mem": mem}, spec, target_id=7, seed_layer=guide)
    assert "cadherin" in str(exc.value)
    assert "mem" in str(exc.value)


def test_a_missing_channel_name_raises():
    labels, guide, mem, _ = two_columns(0.4)
    spec = _spec(params={"wall_channel": None})
    with pytest.raises(ValueError, match="wall_channel"):
        reshape_region(labels, {"mem": mem}, spec, target_id=7, seed_layer=guide)


def test_a_4d_image_is_rejected_rather_than_silently_sliced():
    """The recorded intensity-split bug class: a 4D image handed to a 3D op."""
    labels, guide, mem, _ = two_columns(0.4)
    stacked = np.stack([mem, mem], axis=0)
    with pytest.raises(ValueError, match="3D"):
        reshape_region(labels, {"mem": stacked}, _spec(), target_id=7,
                       seed_layer=guide)


def test_geometry_only_walls_need_no_image_at_all():
    labels, guide, _mem, _ = two_columns(0.4)
    spec = ReshapeSpec(target="label", seeds="labels", walls="distance",
                       arbiter=False)
    out = reshape_region(labels, {}, spec, target_id=7, seed_layer=guide,
                         spacing=ISO)
    assert out.status == "ok"
    assert out.n_pieces == 2


# --------------------------------------------------------------------------- #
# 5. the pre-flight size guard
# --------------------------------------------------------------------------- #
def test_an_over_budget_job_is_refused_up_front_with_the_numbers():
    spec = _spec(params={"size_budget": 10})
    out, _, _ = _run(spec)
    assert out.status == "refused"
    assert out.refusal_reason == "too-big"
    text = " ".join(out.messages)
    assert "10" in text and "markers" in text


def test_the_default_budget_lets_a_normal_cell_through():
    out, _, _ = _run()
    assert out.status == "ok"
    assert DEFAULT_SIZE_BUDGET == 2 ** 27


def test_a_zero_budget_disables_the_guard():
    spec = _spec(params={"size_budget": 0})
    out, _, _ = _run(spec)
    assert out.status == "ok"


def test_the_guard_runs_before_the_solver(monkeypatch):
    import packages.epicycle_annotator.reshape.engine as engine_mod

    def _boom(*a, **k):
        raise AssertionError("the solver ran despite an over-budget job")

    monkeypatch.setattr(engine_mod, "solve", _boom)
    out, _, _ = _run(_spec(params={"size_budget": 10}))
    assert out.status == "refused"


# --------------------------------------------------------------------------- #
# 6. the safe write-back
# --------------------------------------------------------------------------- #
def test_apply_never_overwrites_a_neighbour_label():
    volume = solid_cell()
    before_neighbour = int((volume == 99).sum())
    roi = (slice(None), slice(0, 40), slice(0, 32))
    result = np.zeros(volume[roi].shape, np.int32)
    result[:, 8:14, 6:26] = 1
    result[:, 14:20, 6:26] = 2
    result[:, 24:34, 6:26] = 3          # deliberately ON TOP of the neighbour
    written = apply_reshape_to_volume(volume, roi, result,
                                      allowed_ids={7}, id_map={1: 7, 2: 200, 3: 300})
    assert int((volume == 99).sum()) == before_neighbour
    assert 300 not in set(np.unique(volume)), "a neighbour was overwritten"
    assert written == int((volume == 7).sum()) + int((volume == 200).sum())


def test_apply_writes_only_where_allowed_ids_were():
    volume = solid_cell()
    roi = (slice(None), slice(0, 40), slice(0, 32))
    result = np.ones(volume[roi].shape, np.int32)     # claim the WHOLE roi
    apply_reshape_to_volume(volume, roi, result, allowed_ids={7}, id_map={1: 55})
    assert set(np.unique(volume)) == {0, 55, 99}
    assert int((volume == 55).sum()) == 12 * 12 * 20


def test_apply_with_background_allowed_ids_is_the_add_shape():
    volume = solid_cell()
    roi = (slice(None), slice(0, 40), slice(0, 32))
    result = np.ones(volume[roi].shape, np.int32)
    apply_reshape_to_volume(volume, roi, result, allowed_ids={0}, id_map={1: 55})
    assert int((volume == 7).sum()) == 12 * 12 * 20, "the target was overwritten"
    assert int((volume == 99).sum()) == 12 * 10 * 20


def test_an_unticked_piece_mapped_back_to_the_target_keeps_the_voxel_count():
    """Unticking piece 2 means 'not a separate cell', never 'delete tissue'."""
    volume = solid_cell()
    total_before = int((volume == 7).sum())
    roi = (slice(None), slice(0, 40), slice(0, 32))
    result = np.zeros(volume[roi].shape, np.int32)
    result[:, 8:14, 6:26] = 1
    result[:, 14:20, 6:26] = 2
    apply_reshape_to_volume(volume, roi, result, allowed_ids={7},
                            id_map={1: 7, 2: 7})       # piece 2 unticked
    assert int((volume == 7).sum()) == total_before


def test_apply_requires_every_piece_to_be_mapped():
    volume = solid_cell()
    roi = (slice(None), slice(0, 40), slice(0, 32))
    result = np.zeros(volume[roi].shape, np.int32)
    result[:, 8:14, 6:26] = 1
    result[:, 14:20, 6:26] = 2
    with pytest.raises(ValueError, match="id_map is missing"):
        apply_reshape_to_volume(volume, roi, result, allowed_ids={7}, id_map={1: 7})


def test_apply_rejects_a_shape_mismatch():
    volume = solid_cell()
    roi = (slice(None), slice(0, 10), slice(0, 10))
    with pytest.raises(ValueError, match="shape"):
        apply_reshape_to_volume(volume, roi, np.zeros((3, 3, 3), np.int32),
                                allowed_ids={7}, id_map={})


def test_apply_returns_zero_when_nothing_is_writable():
    volume = solid_cell()
    roi = (slice(None), slice(24, 34), slice(6, 26))    # the neighbour only
    result = np.ones(volume[roi].shape, np.int32)
    assert apply_reshape_to_volume(volume, roi, result, allowed_ids={7},
                                   id_map={1: 5}) == 0
    assert set(np.unique(volume)) == {0, 7, 99}


def test_an_end_to_end_split_commits_two_new_ids_without_touching_anything_else():
    labels, guide, mem, _ = two_columns(0.4)
    labels[:, 34:38, 6:26] = 99        # a neighbour beside the target
    out = reshape_region(labels, {"mem": mem}, _spec(), target_id=7,
                         seed_layer=guide, spacing=ISO)
    assert out.n_pieces == 2
    volume = labels.copy()
    ids = sorted(set(int(v) for v in np.unique(out.result)) - {0})
    written = apply_reshape_to_volume(volume, out.roi, out.result,
                                      allowed_ids={7},
                                      id_map={ids[0]: 201, ids[1]: 202})
    assert written > 0
    assert int((volume == 99).sum()) == int((labels == 99).sum())
    assert int((volume == 201).sum()) + int((volume == 202).sum()) == \
        int((labels == 7).sum())


# --------------------------------------------------------------------------- #
# 7. refine stages
# --------------------------------------------------------------------------- #
def test_merge_small_pieces_absorbs_a_sliver():
    result = np.zeros((4, 12, 6), np.int32)
    result[:, 1:9, 1:5] = 1
    result[:, 9:10, 1:5] = 2                  # a 1-voxel-thick sliver
    merged = merge_small_pieces(result, min_size=100)
    assert set(np.unique(merged)) == {0, 1}


def test_merge_small_pieces_never_absorbs_a_protected_seed_backed_cell():
    result = np.zeros((4, 12, 6), np.int32)
    result[:, 1:9, 1:5] = 1
    result[:, 9:10, 1:5] = 2
    merged = merge_small_pieces(result, min_size=100, protect={2})
    assert set(np.unique(merged)) == {0, 1, 2}
    assert np.array_equal(merged, result)


def test_merge_small_pieces_is_a_no_op_at_zero_min_size():
    result = np.zeros((4, 8, 6), np.int32)
    result[:, 1:4, 1:5] = 1
    result[:, 4:7, 1:5] = 2
    assert np.array_equal(merge_small_pieces(result, 0), result)


def test_merge_to_count_reduces_but_never_invents_pieces():
    result = np.zeros((4, 14, 6), np.int32)
    result[:, 1:5, 1:5] = 1
    result[:, 5:9, 1:5] = 2
    result[:, 9:13, 1:5] = 3
    assert len(set(np.unique(merge_to_count(result, 2))) - {0}) == 2
    assert len(set(np.unique(merge_to_count(result, 9))) - {0}) == 3


def test_merge_small_never_absorbs_a_seed_backed_cell_end_to_end():
    """Every piece the engine produces is seed-backed, so `protect=seed_labels`
    means even an absurd min_size cannot collapse the split. A poor split is
    flagged by the quality fields, never silently merged away."""
    spec = _spec(refine=("merge_small",), params={"min_size": 10 ** 9})
    out, _, _ = _run(spec)
    assert out.n_pieces == 2


def test_merge_to_count_is_reported_in_words():
    spec = _spec(refine=("merge_to_count",), params={"target_count": 1})
    out, _, _ = _run(spec)
    assert out.n_pieces == 1
    assert any("Merged 2 pieces down to the expected 1" in m for m in out.messages)


def test_too_few_pieces_produces_the_warning_message():
    spec = _spec(refine=("merge_to_count",), params={"target_count": 5})
    out, _, _ = _run(spec)
    assert any("Wanted 5 pieces but only got" in m for m in out.messages)


def test_guided_refine_that_would_erase_a_cell_keeps_the_unrefined_split(monkeypatch):
    """The safety guard ported verbatim from _refine_result_guided."""
    result = np.zeros((4, 12, 6), np.int32)
    result[:, 1:9, 1:5] = 1
    result[:, 9:11, 1:5] = 2
    region = result > 0

    def _piece_one_swallows_piece_two(mask, guide, radius=4, eps=0.01, sink=None):
        mask = np.asarray(mask, bool)
        if (result[mask] == 1).any():
            return region.copy()                # piece 1 expands over everything
        return np.zeros_like(mask)              # piece 2 erodes to nothing

    sink = MessageSink()
    monkeypatch.setattr(postprocess, "guided_filter_refine",
                        _piece_one_swallows_piece_two)
    refined = refine_result_guided(result, region, np.zeros(result.shape, np.float32),
                                   sink=sink)
    assert np.array_equal(refined, result)
    assert any("would have erased an object" in m for m in sink.messages)


# --------------------------------------------------------------------------- #
# 8. optional dependency -- degrade with a message, never an import-time failure
# --------------------------------------------------------------------------- #
def _break_usegment3d(monkeypatch, module, name):
    def _raise():
        raise ImportError(
            optional_deps.USEGMENT3D_MISSING.format(what="X", fallback="Y")
        )
    monkeypatch.setattr(module, name, _raise)


def test_guided_refine_degrades_with_a_message_when_usegment3d_is_absent(monkeypatch):
    _break_usegment3d(monkeypatch, postprocess, "import_guidedfilter")
    sink = MessageSink()
    mask = np.zeros((3, 4, 4), bool)
    mask[1, 1:3, 1:3] = True
    out = guided_filter_refine(mask, np.zeros((3, 4, 4), np.float32), sink=sink)
    assert np.array_equal(out, mask), "the mask must come back unchanged"
    assert any("u-Segment3D is not installed" in m for m in sink.messages)


def test_label_diffusion_falls_back_to_watershed_with_a_message(monkeypatch):
    _break_usegment3d(monkeypatch, solvers, "import_diffuse_labels3D")
    labels, guide, mem, _ = two_columns(0.4)
    spec = _spec(solver="label_diffusion")
    out = reshape_region(labels, {"mem": mem}, spec, target_id=7,
                         seed_layer=guide, spacing=ISO)
    assert out.status == "ok"
    assert out.n_pieces == 2
    assert any("u-Segment3D is not installed" in m for m in out.messages)


def test_the_optional_import_message_names_the_package_and_the_fallback():
    message = optional_deps.USEGMENT3D_MISSING.format(
        what="the label-diffusion solver", fallback="watershed was used instead")
    assert "segment3D" in message
    assert "watershed" in message


def test_have_usegment3d_returns_a_bool():
    assert isinstance(optional_deps.have_usegment3d(), bool)


# --------------------------------------------------------------------------- #
# 9. the random-walker solver -- spacing, and the forbidden multigrid mode
# --------------------------------------------------------------------------- #
def test_random_walker_receives_spacing(monkeypatch):
    seen = {}
    import skimage.segmentation as segmentation

    real = segmentation.random_walker

    def _spy(data, labels_, **kw):
        seen.update(kw)
        return real(data, labels_, **kw)

    monkeypatch.setattr(segmentation, "random_walker", _spy)
    labels, guide, mem, _ = two_columns(0.4)
    out = reshape_region(labels, {"mem": mem}, _spec(solver="random_walker"),
                         target_id=7, seed_layer=guide, spacing=ANISO)
    assert out.status == "ok"
    assert seen.get("spacing") == ANISO


def test_random_walker_never_uses_the_multigrid_mode(monkeypatch):
    """`pyamg` is not installed; cg_mg would raise deep inside the worker."""
    seen = {}
    import skimage.segmentation as segmentation

    real = segmentation.random_walker

    def _spy(data, labels_, **kw):
        seen.update(kw)
        return real(data, labels_, **kw)

    monkeypatch.setattr(segmentation, "random_walker", _spy)
    labels, guide, mem, _ = two_columns(0.4)
    reshape_region(labels, {"mem": mem}, _spec(solver="random_walker"),
                   target_id=7, seed_layer=guide, spacing=ISO)
    assert seen.get("mode") == "cg_j"
    assert seen.get("mode") not in ("cg_mg",)


def test_random_walker_with_one_seed_falls_back_to_watershed_with_a_message():
    labels, guide, mem, _ = two_columns(0.4)
    guide[guide == 2] = 0
    out = reshape_region(labels, {"mem": mem}, _spec(solver="random_walker"),
                         target_id=7, seed_layer=guide, spacing=ISO)
    assert out.status == "ok"
    assert any("at least 2 markers" in m for m in out.messages)


def test_solve_rejects_nothing_and_returns_a_partition():
    region = np.zeros((4, 10, 6), bool)
    region[:, 1:9, 1:5] = True
    seeds = np.zeros(region.shape, np.int32)
    seeds[2, 2, 2] = 1
    seeds[2, 7, 2] = 2
    out = solve(np.zeros(region.shape, np.float32), seeds, region)
    assert set(np.unique(out)) == {0, 1, 2}


# --------------------------------------------------------------------------- #
# 10. anisotropy reaches the sigmas
# --------------------------------------------------------------------------- #
def test_axis_sigma_shrinks_the_coarse_axis():
    assert axis_sigma(2.0, ISO) == (2.0, 2.0, 2.0)
    z, y, x = axis_sigma(2.0, ANISO)
    assert z == pytest.approx(0.4)
    assert (y, x) == (2.0, 2.0)


def test_physical_min_distance_converts_against_the_finest_axis():
    assert physical_to_voxel_scalar(5.0, ISO) == 5
    assert physical_to_voxel_scalar(5.0, (5.0, 0.5, 0.5)) == 10
    assert physical_to_voxel_scalar(0.01, ISO) == 1, "never below one voxel"


def test_anisotropic_spacing_changes_the_elevation():
    _labels, _guide, mem, region = two_columns(0.4)
    iso, _, _ = build_elevation("membrane", mem, region, spacing=ISO)
    aniso, _, _ = build_elevation("membrane", mem, region, spacing=ANISO)
    assert not np.allclose(iso, aniso), "spacing did not reach the wall builder"


def test_distance_and_skeleton_walls_use_spacing():
    region = np.zeros((14, 20, 20), bool)
    region[3:11, 4:16, 4:16] = True        # inset from every border
    for kind in ("distance", "skeleton"):
        iso, _, _ = build_elevation(kind, None, region, spacing=ISO)
        aniso, _, _ = build_elevation(kind, None, region, spacing=ANISO)
        assert not np.allclose(iso, aniso), kind


def test_none_walls_are_flat_and_distinct_from_distance_walls():
    _labels, _guide, _mem, region = two_columns(0.0)
    flat, _, _ = build_elevation("none", None, region, spacing=ISO)
    dist, _, _ = build_elevation("distance", None, region, spacing=ISO)
    assert flat.max() == flat.min() == 0.0
    assert dist.max() > dist.min(), "no value silently aliases another"


# --------------------------------------------------------------------------- #
# 11. determinism
# --------------------------------------------------------------------------- #
def test_the_same_inputs_give_a_bit_identical_partition():
    first, _, _ = _run()
    second, _, _ = _run()
    assert np.array_equal(first.result, second.result)
    assert np.array_equal(first.seeds, second.seeds)
    assert first.piece_sizes == second.piece_sizes
    assert first.messages == second.messages


def test_the_fixture_rng_is_seeded():
    a = two_columns(0.2, seed=3)[2]
    b = two_columns(0.2, seed=3)[2]
    assert np.array_equal(a, b)


def test_seed_ids_are_stable_between_a_preview_and_a_commit():
    """Detection is deterministic, so Preview and the commit see the SAME ids --
    which is what makes `keep_seed_ids` cherry-picking safe."""
    preview, _, _ = _run()
    commit, _, _ = _run(keep_seed_ids=sorted(
        set(int(v) for v in np.unique(preview.seeds)) - {0}))
    assert np.array_equal(preview.result, commit.result)


def test_distance_maxima_are_deterministic():
    _labels, _guide, mem, _ = two_columns(0.4)
    assert np.array_equal(seeds_distance_maxima(mem), seeds_distance_maxima(mem))


# --------------------------------------------------------------------------- #
# 11b. the distance detector must look INSIDE the object
# --------------------------------------------------------------------------- #
def _branched(shape=(3, 24, 24), w=2):
    """An L-shaped object that fills 19 % of its own bounding box.

    The shape the owner measured the bug on: `_build_seeds`'s `distance`
    branch feeds `seeds_distance_maxima` the WALL elevation, and that
    detector was the only one of the six that did not take `region=` --
    so it normalised and took its h-maxima over the WHOLE crop.  The
    background is one big blob and the object is two thin arms, so
    every regional maximum of the distance transform sat OUTSIDE the
    object and the engine answered `found=11 in_region=0` at every
    prominence (object 139 of CELLS-figure/patch_6, 2026-09-20).
    """
    region = np.zeros(shape, bool)
    region[:, 2:2 + w, 2:22] = True
    region[:, 2:22, 2:2 + w] = True
    from scipy import ndimage as _ndi
    mem = np.zeros(shape, np.float32)
    mem[_ndi.binary_dilation(region, iterations=1) & ~region] = 1.0
    labels = np.zeros(shape, np.int32)
    labels[region] = 7
    return labels, region, mem


def test_a_branched_object_finds_NOTHING_without_the_region():
    """The bug, stated as an arrangement rather than as a screenshot."""
    _labels, region, mem = _branched()
    seeds = seeds_distance_maxima(mem, ISO, 0.15)
    found = set(int(v) for v in np.unique(seeds)) - {0}
    inside = set(int(v) for v in np.unique(np.where(region, seeds, 0))) - {0}
    assert found, "the arrangement no longer reproduces the finding"
    assert not inside, (
        "every marker used to land in the background; if this now passes "
        "the fixture has drifted and the test below proves nothing"
    )


def test_the_region_puts_the_maxima_inside_the_object():
    _labels, region, mem = _branched()
    seeds = seeds_distance_maxima(mem, ISO, 0.15, region=region)
    inside = set(int(v) for v in np.unique(np.where(region, seeds, 0))) - {0}
    assert inside, "the marker source is unusable on the shape you most "                    "want to cut"


def test_no_marker_is_left_outside_the_region():
    """Its five siblings all end with `np.where(region, seeds, 0)`."""
    _labels, region, mem = _branched()
    seeds = seeds_distance_maxima(mem, ISO, 0.15, region=region)
    assert int((seeds[~region] != 0).sum()) == 0


def test_without_a_region_it_is_BYTE_IDENTICAL_to_before():
    """The DEFAULT argument is inert: passing nothing and passing
    ``region=None`` give the same array (audit L1).

    This is not the parity guard against the pre-change code -- it
    cannot be, since both calls run the new function.  That job belongs
    to `test_reshape_golden.py`, which compares against stored fixtures
    and was re-run with ``EPICYCLE_STRICT_GOLDEN=1``.  What this one
    rules out is a default that quietly does something, which is the
    way a "purely additive argument" usually stops being one."""
    _labels, _guide, mem, _ = two_columns(0.4)
    a = seeds_distance_maxima(mem, ISO, 0.15)
    b = seeds_distance_maxima(mem, ISO, 0.15, region=None)
    assert np.array_equal(a, b)


def test_the_engine_hands_the_region_to_the_distance_detector(monkeypatch):
    import packages.epicycle_annotator.reshape.engine as engine_mod

    seen = {}

    def _spy(mem, spacing=(1, 1, 1), h=0.15, region=None):
        seen["region"] = region
        out = np.zeros(mem.shape, np.int32)
        out[region] = 1
        return out

    monkeypatch.setattr(engine_mod.seed_ops, "seeds_distance_maxima", _spy)
    labels, region, mem = _branched()
    spec = _spec(seeds="distance", walls="membrane",
                 params={"wall_channel": "mem"})
    reshape_region(labels, {"mem": mem}, spec, target_id=7, spacing=ISO)
    assert seen.get("region") is not None, (
        "the detector was called over the whole crop again"
    )
    assert seen["region"].shape == region.shape


def test_find_seeds_reports_no_marker_outside_the_object():
    """What the panel reads: `found` and `in_region` now agree, so the
    "n of m markers lie outside this object" line stops firing for a
    source that cannot produce one."""
    from packages.epicycle_annotator.reshape import find_seeds

    labels, _region, mem = _branched()
    spec = _spec(seeds="distance", walls="membrane",
                 params={"wall_channel": "mem"})
    report = find_seeds(labels, {"mem": mem}, spec, target_id=7, spacing=ISO)
    assert report.n_found == report.n_in_region, (
        "found={} in_region={}".format(report.n_found, report.n_in_region)
    )


# --------------------------------------------------------------------------- #
# 12. cancellation -- "abandon the result", polled between stages
# --------------------------------------------------------------------------- #
def test_a_cancel_set_up_front_stops_before_any_work():
    event = threading.Event()
    event.set()
    out, _, _ = _run(cancel_event=event)
    assert out.status == "cancelled"
    assert out.result is None
    assert any("Stopped waiting" in m for m in out.messages)


def test_a_cancel_set_mid_run_is_honoured_between_stages(monkeypatch):
    event = threading.Event()
    import packages.epicycle_annotator.reshape.engine as engine_mod

    real_solve = engine_mod.solve

    def _solve_then_cancel(*a, **k):
        out = real_solve(*a, **k)
        event.set()
        return out

    monkeypatch.setattr(engine_mod, "solve", _solve_then_cancel)
    out, _, _ = _run(cancel_event=event)
    assert out.status == "cancelled"


def test_no_cancel_event_runs_to_completion():
    out, _, _ = _run(cancel_event=None)
    assert out.status == "ok"


# --------------------------------------------------------------------------- #
# 13. messages / the log callback (the carved-out prints)
# --------------------------------------------------------------------------- #
def test_the_log_callback_mirrors_every_message():
    seen = []
    labels, guide, mem, _ = two_columns(0.4)
    guide[:, 4, 7] = 11
    out = reshape_region(labels, {"mem": mem}, _spec(), target_id=7,
                         seed_layer=guide, spacing=ISO, log=seen.append)
    assert seen == out.messages
    assert seen, "at least the dropped-seed report should have been emitted"


def test_a_broken_log_callback_never_breaks_the_run():
    def _boom(_text):
        raise RuntimeError("the console is on fire")

    labels, guide, mem, _ = two_columns(0.4)
    guide[:, 4, 7] = 11
    out = reshape_region(labels, {"mem": mem}, _spec(), target_id=7,
                         seed_layer=guide, spacing=ISO, log=_boom)
    assert out.status == "ok"
    assert out.messages


def test_the_engine_never_prints(capsys):
    _run()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_message_sink_collects_and_counts():
    sink = MessageSink()
    sink.emit("one")
    sink.extend(["two", "three"])
    assert list(sink) == ["one", "two", "three"]
    assert len(sink) == 3


# --------------------------------------------------------------------------- #
# 14. the seed helpers, directly
# --------------------------------------------------------------------------- #
def test_seeds_from_labels_masks_to_the_region():
    guide = np.zeros((3, 6, 6), np.int32)
    guide[1, 1, 1] = 4
    guide[1, 4, 4] = 5
    region = np.zeros((3, 6, 6), bool)
    region[1, 0:3, 0:3] = True
    seeds = seeds_from_labels(guide, region=region)
    assert set(np.unique(seeds)) == {0, 4}


def test_seeds_from_labels_preserves_the_guide_ids():
    guide = np.zeros((2, 4, 4), np.int32)
    guide[0, 1, 1] = 77
    assert set(np.unique(seeds_from_labels(guide))) == {0, 77}


@pytest.mark.parametrize("seeds_kind,channel_needed", [
    ("peaks", True), ("hmax", True), ("hmin", True),
    ("blobs_log", True), ("blobs_dog", True), ("threshcc", True),
    ("distance", False),
])
def test_every_seed_kind_runs_end_to_end(seeds_kind, channel_needed):
    labels, _guide, mem, _ = two_columns(0.4)
    params = {"wall_channel": "mem", "clip": 0.0, "h": 0.05}
    if channel_needed:
        params["seed_channel"] = "mem"
    spec = ReshapeSpec(target="label", seeds=seeds_kind, walls="intensity",
                       params=params, arbiter=False)
    out = reshape_region(labels, {"mem": mem}, spec, target_id=7, spacing=ISO)
    assert out.status in ("ok", "refused")
    if out.status == "ok":
        assert out.n_pieces >= 1
        assert out.seed_source


def test_an_image_seed_kind_without_a_channel_raises():
    labels, _guide, mem, _ = two_columns(0.4)
    spec = ReshapeSpec(target="label", seeds="hmax", walls="intensity",
                       params={"wall_channel": "mem"}, arbiter=False)
    with pytest.raises(ValueError, match="seed_channel"):
        reshape_region(labels, {"mem": mem}, spec, target_id=7, spacing=ISO)


# --------------------------------------------------------------------------- #
# 15. the ROI
# --------------------------------------------------------------------------- #
def test_the_roi_is_the_padded_bounding_box_of_the_target():
    out, labels, _ = _run()
    crop = labels[out.roi]
    assert (crop == 7).any()
    assert crop.shape[1] < labels.shape[1], "the ROI should be tighter than the volume"


def test_a_caller_supplied_roi_is_honoured():
    labels, guide, mem, _ = two_columns(0.4)
    roi = (slice(0, 12), slice(0, 40), slice(0, 32))
    out = reshape_region(labels, {"mem": mem}, _spec(), target_id=7,
                         seed_layer=guide, spacing=ISO, roi=roi)
    assert out.roi == roi
    assert out.result.shape == labels[roi].shape


def test_an_roi_sized_seed_layer_is_accepted_as_well_as_a_full_volume_one():
    labels, guide, mem, _ = two_columns(0.4)
    full = reshape_region(labels, {"mem": mem}, _spec(), target_id=7,
                          seed_layer=guide, spacing=ISO)
    cropped = reshape_region(labels, {"mem": mem}, _spec(), target_id=7,
                             seed_layer=guide[full.roi], spacing=ISO)
    assert np.array_equal(full.result, cropped.result)


def test_random_walker_partition_actually_differs_at_anisotropic_spacing():
    """A behavioural companion to the kwarg spy above, and to the golden test
    that proves the same thing against the membrane sandbox's library.

    The golden module SKIPS when the membrane worktree is absent, so on a fresh
    checkout a dropped `spacing=` would go completely unnoticed.  This test
    needs nothing but the repo.  It uses the Z-stacked fixture deliberately:
    the two-column fixture is extruded uniformly in Z, so reweighting the Z
    axis cannot move a cut lying in a Y plane."""
    from .reshape_fixtures import stacked_in_z

    labels, guide, mem, region = stacked_in_z(0.4)
    elevation, _c, _r = build_elevation("membrane", mem, region, spacing=ISO)
    seeds = np.where(region, guide, 0)
    iso = solve(elevation, seeds, region, "random_walker", spacing=ISO)
    aniso = solve(elevation, seeds, region, "random_walker", spacing=ANISO)
    assert not np.array_equal(iso, aniso), (
        "spacing did not reach random_walker -- a soft cut is biased along the "
        "coarse axis without it"
    )


# --------------------------------------------------------------------------- #
# Brick 2b -- the wall contrast range
#
# Two of the sandbox's eight diagnostic prints were summary lines; one of
# them said how much contrast the elevation actually had.  Without it,
# "the cut went nowhere" and "there was nothing to cut on" look the same
# on screen.  One string, computed from an array the engine already holds.
# --------------------------------------------------------------------------- #
class TestTheWallContrastRangeIsReported:
    def test_a_run_says_what_contrast_the_wall_had(self):
        out, _labels, _region = _run()
        line = [m for m in out.messages if m.startswith("Wall contrast ")]
        assert line, out.messages
        assert len(line) == 1

    def test_the_numbers_are_the_elevation_inside_the_cell(self):
        out, _labels, _region = _run()
        line = [m for m in out.messages if m.startswith("Wall contrast ")][0]
        lo_hi = line.split("Wall contrast ", 1)[1].rstrip(".")
        lo, hi = (float(v) for v in lo_hi.split("-"))
        elev = np.asarray(out.elevation)[out.region]
        assert lo == pytest.approx(round(float(elev.min()), 2), abs=0.01)
        assert hi == pytest.approx(round(float(elev.max()), 2), abs=0.01)
        assert lo <= hi

    def test_a_SHAPE_wall_says_nothing_at_all(self):
        """The line answers "was there anything in the channel to cut
        on".  A shape wall has no channel, so its contrast is a fact
        about the distance transform -- noise in the one place the
        result slot has room for a sentence."""
        from packages.epicycle_annotator.reshape import GEOMETRY_ONLY_WALLS

        labels, guide, mem, _ = two_columns(0.4)
        for kind in sorted(GEOMETRY_ONLY_WALLS):
            out = reshape_region(
                labels, {"mem": mem},
                _spec(walls=kind, params={"wall_channel": None}),
                target_id=7, seed_layer=guide, spacing=ISO)
            assert not [m for m in out.messages
                        if m.startswith("Wall contrast ")], (kind,
                                                             out.messages)

    def test_every_IMAGE_wall_says_it(self):
        from packages.epicycle_annotator.reshape import IMAGE_DERIVED_WALLS

        labels, guide, mem, _ = two_columns(0.4)
        for kind in sorted(IMAGE_DERIVED_WALLS):
            out = reshape_region(
                labels, {"mem": mem},
                _spec(walls=kind, params={"wall_channel": "mem"}),
                target_id=7, seed_layer=guide, spacing=ISO)
            assert [m for m in out.messages
                    if m.startswith("Wall contrast ")], (kind, out.messages)

    def test_it_is_cp1252_safe(self):
        out, _labels, _region = _run()
        line = [m for m in out.messages if m.startswith("Wall contrast ")]
        line[0].encode("cp1252")

    def test_a_refusal_before_the_wall_does_not_claim_a_contrast(self):
        labels, guide, mem, _ = two_columns(0.4)
        out = reshape_region(labels, {"mem": mem}, _spec(), target_id=9999,
                             seed_layer=guide, spacing=ISO)
        assert out.status == "refused"
        assert not [m for m in out.messages if m.startswith("Wall contrast ")]


# --------------------------------------------------------------------------- #
# Brick 2b fix pass -- `merged_to_count` as a FLAG (M2)
#
# The quality gate suppresses its "a marker got no piece" rule when the
# merge-to-count stage fired, because that stage absorbs marker-backed
# pieces on purpose.  The panel was deciding that by searching the
# messages for "Merged " -- a substring of user-facing prose, which is
# exactly the string this fix pass is rewording elsewhere.
# --------------------------------------------------------------------------- #
class TestTheEngineSaysWhetherItMergedToACount:
    def test_a_plain_run_did_not(self):
        out, _labels, _region = _run()
        assert out.merged_to_count is False

    def test_a_run_that_merged_says_so(self):
        labels, guide, mem, _ = two_columns(0.4)
        out = reshape_region(
            labels, {"mem": mem},
            _spec(refine=("merge_to_count",), params={"target_count": 1}),
            target_id=7, seed_layer=guide, spacing=ISO)
        assert out.n_pieces == 1
        assert out.merged_to_count is True
        assert any("Merged 2 pieces down to the expected 1" in m
                   for m in out.messages)

    def test_asking_for_a_count_it_ALREADY_has_is_not_a_merge(self):
        """The flag means "the stage actually merged something", which is
        the only case rule (a) has to be silent for."""
        labels, guide, mem, _ = two_columns(0.4)
        out = reshape_region(
            labels, {"mem": mem},
            _spec(refine=("merge_to_count",), params={"target_count": 2}),
            target_id=7, seed_layer=guide, spacing=ISO)
        assert out.n_pieces == 2
        assert out.merged_to_count is False

    def test_the_field_defaults_to_False_on_a_bare_result(self):
        from packages.epicycle_annotator.reshape import ReshapeResult

        assert ReshapeResult().merged_to_count is False

    def test_a_refusal_carries_the_field_too(self):
        labels, guide, mem, _ = two_columns(0.4)
        out = reshape_region(labels, {"mem": mem}, _spec(), target_id=9999,
                             seed_layer=guide, spacing=ISO)
        assert out.status == "refused"
        assert out.merged_to_count is False


# --------------------------------------------------------------------------- #
# Brick 2b fix pass -- the engine speaks the panel's language (M7)
#
# Every message here is shown VERBATIM in the panel's result slot, under
# labels that say "marker" and "wall".  A user who reads "no membrane
# between the seeds" under a row called "Only cut where I can see a wall"
# is reading two vocabularies for one thing -- and on a non-membrane wall
# channel the sentence is simply wrong.
#
# The algorithms are untouched: `test_reshape_golden` compares numbers,
# and these assertions only look at strings.
# --------------------------------------------------------------------------- #
def _message_strings():
    """Every string literal that is EMITTED as a message.

    Walked with `ast`, not grepped: a regex over quotes also picks up
    docstrings and identifiers, and the point of this sweep is the
    sentences a user actually reads.  The calls that reach a user are
    `sink.emit(...)`, `result.add_message(...)` and the engine's own
    `_refuse(reason, message)`; a `.format()` on any of them is followed
    through to the literal underneath.
    """
    import ast

    from packages.epicycle_annotator.reshape import (
        arbiter, engine, postprocess, solvers,
    )

    EMITTERS = {"emit", "add_message", "_refuse"}
    out = []

    def literals(node, mod):
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                out.append((mod, sub.value))

    for mod in (arbiter, engine, postprocess, solvers):
        name = mod.__name__.rsplit(".", 1)[-1]
        tree = ast.parse(open(mod.__file__, encoding="utf-8").read())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            called = (fn.attr if isinstance(fn, ast.Attribute)
                      else getattr(fn, "id", ""))
            if called not in EMITTERS:
                continue
            for arg in node.args:
                literals(arg, name)
    return out


class TestTheEngineSpeaksThePanelsLanguage:
    #: The one emitted literal that is an IDENTIFIER: `_refuse`'s first
    #: argument is a reason CODE the orchestrator branches on, never
    #: shown.  It is filtered by shape -- a code has no spaces.
    def _prose(self):
        return [(mod, text) for mod, text in _message_strings()
                if " " in text]

    def test_no_message_says_membrane(self):
        bad = [(m, t) for m, t in self._prose() if "membrane" in t.lower()]
        assert not bad, bad

    def test_no_message_says_seed(self):
        """"Seed" is the engine's word for it; the panel has said
        "marker" since Brick 1, and the two lists on screen are the
        marker list and the piece list."""
        bad = [(m, t) for m, t in self._prose() if "seed" in t.lower()]
        assert not bad, bad

    def test_the_sentences_a_user_actually_reads_are_the_new_ones(self):
        """Behavioural, not textual: drive the arbiter to a "dividing"
        verdict and read what it says."""
        from .reshape_fixtures import two_columns

        labels, guide, mem, _ = two_columns(0.0)
        out = reshape_region(
            labels, {"mem": mem},
            _spec(walls="membrane", arbiter=True,
                  params={"wall_channel": "mem", "clip": 0.01}),
            target_id=7, seed_layer=guide, spacing=ISO)
        text = " ".join(out.messages).lower()
        assert "membrane" not in text, out.messages
        assert "seed" not in text, out.messages

    def test_a_zero_marker_refusal_reads_in_the_panels_words(self):
        labels, guide, mem, _ = two_columns(0.4)
        out = reshape_region(labels, {"mem": mem}, _spec(), target_id=7,
                             seed_layer=np.zeros_like(guide), spacing=ISO)
        assert out.status == "refused"
        joined = " ".join(out.messages)
        assert "marker" in joined.lower()
        assert "seed" not in joined.lower()

    def test_the_refusal_REASON_codes_are_untouched(self):
        """They are identifiers the orchestrator branches on, not prose.
        Renaming one would be an API change wearing a wording change."""
        labels, guide, mem, _ = two_columns(0.4)
        out = reshape_region(labels, {"mem": mem}, _spec(), target_id=7,
                             seed_layer=np.zeros_like(guide), spacing=ISO)
        assert out.refusal_reason == "no-seeds"

    def test_every_message_is_cp1252_safe(self):
        for _mod, text in self._prose():
            text.encode("cp1252")


# --------------------------------------------------------------------------- #
# 11c. the shape finder on a hollow object -- audit M7 (2026-09-20)
# --------------------------------------------------------------------------- #
class TestTheShapeFinderRefusesAHollowObject:
    """A silent 68-piece cut is worse than a refusal.

    `seeds_distance_maxima` gained `region=` on 2026-09-20 so it would
    look inside the object instead of at the whole search box.  On every
    SOLID shape measured that is strictly better (cube, cross, dumbbell,
    L, slabs: 1-3 markers).  On a hollow or very thin one it is not:
    every point of a shell is an equal local maximum of the interior
    distance transform, so the finder returns one marker per surface
    bump -- and the same number at every `h`, because there is no
    thickness variation left for `h` to bite on.

    Measured on a closed spherical shell, r 12-14 in a 40-cube (6.9 %
    fill): 68 markers with `walls="skeleton"`, 98 with
    `walls="distance"`, `status=ok`, one piece each.  The user asked to
    split one object in two and got it shattered, with nothing on screen
    saying why.

    So the engine refuses above a cap, and says what to do instead.
    """

    def _run(self, labels, *, walls="skeleton", h=0.15, seeds="distance"):
        spec = _spec(seeds=seeds, walls=walls, params={"h": h})
        return reshape_region(labels, {}, spec, target_id=7, spacing=ISO)

    def test_a_shell_is_refused_rather_than_shattered(self):
        from .reshape_fixtures import hollow_shell

        out = self._run(hollow_shell())
        assert out.status == "refused", (
            "{} markers, {} pieces".format(out.n_seeds, out.n_pieces))

    def test_the_refusal_says_what_it_found_and_what_to_do(self):
        from .reshape_fixtures import hollow_shell

        out = self._run(hollow_shell())
        text = " ".join(out.messages).lower()
        assert "hollow" in text or "thin" in text, out.messages
        assert "image channel" in text, out.messages

    def test_it_names_the_number(self):
        from .reshape_fixtures import hollow_shell

        out = self._run(hollow_shell())
        assert str(out.n_seeds) in " ".join(out.messages), out.messages

    def test_nothing_was_cut(self):
        from .reshape_fixtures import hollow_shell

        out = self._run(hollow_shell())
        assert out.n_pieces == 0
        assert out.result is None or not np.any(out.result)

    @pytest.mark.parametrize("walls", ["skeleton", "distance"])
    def test_both_shape_wall_kinds_are_refused(self, walls):
        from .reshape_fixtures import hollow_shell

        assert self._run(hollow_shell(), walls=walls).status == "refused"

    @pytest.mark.parametrize("h", [0.05, 0.15, 0.5])
    def test_no_h_rescues_it_which_is_the_point(self, h):
        from .reshape_fixtures import hollow_shell

        assert self._run(hollow_shell(), h=h).status == "refused"

    def test_a_DUMBBELL_is_untouched_by_the_cap(self):
        """The shape the finder is for: two lobes, two markers."""
        from .reshape_fixtures import dumbbell

        out = self._run(dumbbell())
        assert out.status == "ok", out.messages
        assert out.n_seeds <= 4, out.n_seeds

    def test_a_SOLID_two_column_blob_is_untouched(self):
        out, _labels, _region = _run()
        assert out.status == "ok"
        assert out.n_pieces == 2

    def test_the_cap_only_applies_to_the_SHAPE_finder(self):
        """A mask or an image detector may legitimately hand over many
        markers -- that is the user's own label image talking, not a
        geometry artefact."""
        labels, guide, mem, _ = two_columns(0.4)
        spec = _spec(seeds="labels", walls="intensity",
                     params={"wall_channel": "mem", "clip": 0.0})
        out = reshape_region(labels, {"mem": mem}, spec, target_id=7,
                             seed_layer=guide, spacing=ISO)
        assert out.status == "ok"

    def test_the_cap_is_a_named_constant_with_a_reason(self):
        from packages.epicycle_annotator.reshape import engine as E

        assert isinstance(E.SHAPE_SEED_CAP, int)
        assert E.SHAPE_SEED_CAP == 32
        assert E.__doc__ is not None

    def test_the_refusal_carries_a_reason_code(self):
        from .reshape_fixtures import hollow_shell

        out = self._run(hollow_shell())
        assert out.refusal_reason, out.refusal_reason
