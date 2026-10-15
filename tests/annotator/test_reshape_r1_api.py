"""R1-facing API surface on the R0 engine (added 2026-09-08).

These four additions exist so R1's orchestration never has to reach into the
engine and change it: a progress callback, per-piece stats in VOLUME
coordinates, elevation reuse between Preview and Accept, and flag remapping
onto the final label ids.

Pure pytest -- no Qt, no napari.
"""

from __future__ import annotations

import numpy as np
import pytest

from packages.epicycle_annotator.reshape import (
    ReshapeSpec,
    apply_reshape_to_volume,
    remap_flags,
    reshape_region,
)

from .reshape_fixtures import solid_cell, two_columns

ISO = (1.0, 1.0, 1.0)


def _spec(**kw):
    params = {"wall_channel": "mem", "clip": 0.0}
    params.update(kw.pop("params", {}))
    base = dict(target="label", seeds="labels", walls="intensity",
                params=params, arbiter=False)
    base.update(kw)
    base["params"] = params
    return ReshapeSpec(**base)


def _run(**kw):
    labels, guide, mem, _ = two_columns(0.4)
    return reshape_region(labels, {"mem": mem}, _spec(), target_id=7,
                          seed_layer=guide, spacing=ISO, **kw), labels


# --------------------------------------------------------------------------- #
# (a) progress callback
# --------------------------------------------------------------------------- #
def test_progress_reports_named_stages_in_order():
    seen = []
    out, _ = _run(progress=lambda stage, i, n: seen.append((stage, i, n)))
    assert out.status == "ok"
    assert seen, "no progress was reported"
    stages = [s for s, _i, _n in seen]
    assert stages == sorted(set(stages), key=stages.index), "stages repeated"
    assert "walls" in stages and "seeds" in stages and "solve" in stages
    indices = [i for _s, i, _n in seen]
    assert indices == sorted(indices), "the step index went backwards"
    totals = {n for _s, _i, n in seen}
    assert len(totals) == 1, "the step total changed mid-run"
    assert max(indices) <= seen[0][2]


def test_progress_is_optional():
    out, _ = _run()
    assert out.status == "ok"


def test_a_broken_progress_callback_never_breaks_the_run():
    def _boom(_stage, _i, _n):
        raise RuntimeError("the progress bar exploded")

    out, _ = _run(progress=_boom)
    assert out.status == "ok"
    assert out.n_pieces == 2


def test_progress_reports_on_a_refusal_too():
    labels, _guide, mem, _ = two_columns(0.4)
    seen = []
    out = reshape_region(labels, {"mem": mem}, _spec(), target_id=7,
                         seed_layer=np.zeros_like(labels), spacing=ISO,
                         progress=lambda s, i, n: seen.append(s))
    assert out.status == "refused"
    assert seen, "a refusal reported no progress at all"


# --------------------------------------------------------------------------- #
# (b) per-piece stats, in VOLUME coordinates
# --------------------------------------------------------------------------- #
def test_piece_stats_are_reported_per_piece():
    out, _ = _run()
    assert len(out.piece_stats) == out.n_pieces
    for row in out.piece_stats:
        assert set(row) == {"id", "voxels", "centroid_zyx", "bbox"}
        assert row["voxels"] > 0
        assert len(row["centroid_zyx"]) == 3
        assert len(row["bbox"]) == 6


def test_piece_stats_voxels_match_piece_sizes():
    out, _ = _run()
    assert sorted((r["voxels"] for r in out.piece_stats), reverse=True) == \
        out.piece_sizes


def test_piece_stats_are_offset_into_volume_coordinates():
    """The engine works in an ROI crop; R1 writes df rows in VOLUME
    coordinates.  The offset is applied here, once, not by every caller."""
    out, _labels = _run()
    z0, y0, x0 = (s.start for s in out.roi)
    for row in out.piece_stats:
        local = np.argwhere(out.result == row["id"])
        expected = local.mean(axis=0) + np.array([z0, y0, x0])
        assert row["centroid_zyx"] == pytest.approx(tuple(expected))
        zmin, zmax, ymin, ymax, xmin, xmax = row["bbox"]
        assert zmin == local[:, 0].min() + z0
        assert zmax == local[:, 0].max() + z0
        assert ymin == local[:, 1].min() + y0
        assert ymax == local[:, 1].max() + y0
        assert xmin == local[:, 2].min() + x0
        assert xmax == local[:, 2].max() + x0


def test_a_piece_centroid_lands_inside_the_volume():
    out, labels = _run()
    for row in out.piece_stats:
        for value, extent in zip(row["centroid_zyx"], labels.shape):
            assert 0 <= value < extent


def test_piece_stats_are_empty_on_a_refusal():
    labels, _guide, mem, _ = two_columns(0.4)
    out = reshape_region(labels, {"mem": mem}, _spec(), target_id=7,
                         seed_layer=np.zeros_like(labels), spacing=ISO)
    assert out.status == "refused"
    assert out.piece_stats == []


# --------------------------------------------------------------------------- #
# (c) elevation reuse between Preview and Accept
# --------------------------------------------------------------------------- #
def test_a_supplied_elevation_is_used_instead_of_rebuilding():
    preview, _labels = _run()
    labels, guide, mem, _ = two_columns(0.4)
    accept = reshape_region(labels, {"mem": mem}, _spec(), target_id=7,
                            seed_layer=guide, spacing=ISO, roi=preview.roi,
                            elevation=preview.elevation)
    assert accept.status == "ok"
    assert np.array_equal(accept.result, preview.result)
    assert np.array_equal(accept.elevation, preview.elevation)


def test_a_supplied_elevation_skips_the_wall_build(monkeypatch):
    import packages.epicycle_annotator.reshape.engine as engine_mod

    preview, _ = _run()

    def _boom(*a, **k):
        raise AssertionError("build_elevation ran despite a supplied elevation")

    monkeypatch.setattr(engine_mod, "build_elevation", _boom)
    labels, guide, mem, _ = two_columns(0.4)
    out = reshape_region(labels, {"mem": mem}, _spec(), target_id=7,
                         seed_layer=guide, spacing=ISO, roi=preview.roi,
                         elevation=preview.elevation)
    assert out.status == "ok"


def test_a_supplied_elevation_needs_no_image_channel():
    """Reuse must work when the caller no longer holds the wall image."""
    preview, _ = _run()
    labels, guide, _mem, _ = two_columns(0.4)
    out = reshape_region(labels, {}, _spec(), target_id=7, seed_layer=guide,
                         spacing=ISO, roi=preview.roi,
                         elevation=preview.elevation)
    assert out.status == "ok"
    assert out.n_pieces == preview.n_pieces


def test_a_mismatched_supplied_elevation_is_rejected():
    preview, _ = _run()
    labels, guide, mem, _ = two_columns(0.4)
    with pytest.raises(ValueError, match="elevation"):
        reshape_region(labels, {"mem": mem}, _spec(), target_id=7,
                       seed_layer=guide, spacing=ISO, roi=preview.roi,
                       elevation=np.zeros((3, 3, 3), np.float32))


# --------------------------------------------------------------------------- #
# (d) flags remapped onto the FINAL label ids
# --------------------------------------------------------------------------- #
def test_remap_flags_translates_interim_ids_to_final_ids():
    assert remap_flags({1: "dividing", 2: "low-evidence"},
                       {1: 201, 2: 202}) == {201: "dividing",
                                             202: "low-evidence"}


def test_remap_flags_ignores_pieces_that_were_not_written():
    assert remap_flags({1: "dividing", 7: "low-evidence"}, {1: 201}) == \
        {201: "dividing"}


def test_remap_flags_on_an_unticked_piece_folded_into_the_target():
    """Two interim pieces can map to the SAME final id when one is unticked; a
    flag on either must land on that id, not be lost."""
    assert remap_flags({2: "dividing"}, {1: 7, 2: 7}) == {7: "dividing"}


def test_remap_flags_handles_the_empty_cases():
    assert remap_flags({}, {1: 5}) == {}
    assert remap_flags({1: "dividing"}, {}) == {}


def test_apply_reshape_to_volume_returns_written_count_and_final_flags():
    volume = solid_cell()
    roi = (slice(None), slice(0, 40), slice(0, 32))
    result = np.zeros(volume[roi].shape, np.int32)
    result[:, 8:14, 6:26] = 1
    result[:, 14:20, 6:26] = 2
    written, final_flags = apply_reshape_to_volume(
        volume, roi, result, allowed_ids={7}, id_map={1: 201, 2: 202},
        flags={1: "dividing"}, return_flags=True,
    )
    assert written == 12 * 12 * 20
    assert final_flags == {201: "dividing"}


def test_apply_reshape_to_volume_stays_backward_compatible():
    """Without ``return_flags`` it still returns a bare int, as R0 shipped."""
    volume = solid_cell()
    roi = (slice(None), slice(0, 40), slice(0, 32))
    result = np.zeros(volume[roi].shape, np.int32)
    result[:, 8:14, 6:26] = 1
    written = apply_reshape_to_volume(volume, roi, result, allowed_ids={7},
                                      id_map={1: 201})
    assert isinstance(written, int)
