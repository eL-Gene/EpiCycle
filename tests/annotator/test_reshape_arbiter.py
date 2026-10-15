"""The arbiter rule -- the membrane decides whether a cut is real.

    Two seeds inside one target:
      - a WALL between them  -> two cells wrongly merged  -> split.
      - NO wall between them -> one dividing cell -> keep whole, flag it.

is explicit that *"a test that only proves 'wall -> split, no wall ->
merge' on one clean fixture proves nothing about the threshold, which is the
only part that can be wrong."*  So the mandatory fixture here is a **sweep**
over wall contrast x SNR x spacing ``(1,1,1)`` and ``(5,1,1)``, asserting:

* the decision flips from "split" to "dividing" at a stated contrast, and that
  contrast is recorded as the tool's sensitivity floor;
* an anisotropic case gives the SAME decision as the isotropic one (this is
  what catches a voxel-unit interface dilation);
* a BRIGHT-cell case with a real wall still splits (this is what catches the
  scale-relative inversion: a bright cell raises its own bar).

Pure pytest -- no Qt, no napari.
"""

from __future__ import annotations

import numpy as np
import pytest

from packages.epicycle_annotator.reshape import (
    MessageSink,
    apply_preproc,
    build_elevation,
    zdepth_correct,
    ReshapeSpec,
    adjacent_pairs,
    arbitrate,
    interface_mask,
    physical_ball,
    reshape_region,
)

from .reshape_fixtures import NOISE, stacked_in_z, two_columns

ISO = (1.0, 1.0, 1.0)
ANISO = (5.0, 1.0, 1.0)

#: The recorded sensitivity floor of the tool on this fixture.  Below this wall
#: contrast the arbiter reports "dividing"; at or above it, it splits.  If a
#: change to the engine moves this number, that is a real change in sensitivity
#: and it must be a deliberate one.
#:
#: **Re-measured 2026-09-08** after the mandatory ``zdepth_correct`` was dropped:
#: 0.12 -> 0.10, i.e. the tool got slightly MORE sensitive once the wall was no
#: longer being flattened before it was weighed.  With NOISE=0.05 that is a
#: 2.0-sigma separation, and the SNR is the number that transfers to real data,
#: not the grey level (see the fixture docstring).
SENSITIVITY_FLOOR_CONTRAST = 0.10
BELOW_FLOOR_CONTRAST = 0.08


def _run(contrast, *, spacing=ISO, interior=0.5, noise=NOISE, arbiter=True,
         percentile=75.0, z_attenuate=False, seed=0):
    labels, guide, mem, _region = two_columns(
        contrast, interior=interior, noise=noise, seed=seed,
        z_attenuate=z_attenuate,
    )
    spec = ReshapeSpec(
        target="label", seeds="labels", walls="intensity",
        params={"wall_channel": "mem", "clip": 0.0,
                "arbiter_percentile": percentile},
        arbiter=arbiter,
    )
    return reshape_region(labels, {"mem": mem}, spec, target_id=7,
                          seed_layer=guide, spacing=spacing)


# --------------------------------------------------------------------------- #
# 1. the core rule
# --------------------------------------------------------------------------- #
def test_wall_between_two_seeds_splits():
    out = _run(0.4)
    assert out.status == "ok"
    assert out.n_pieces == 2
    assert out.arbiter.verdict == "split"
    assert out.arbiter.pairs[0].decision == "split"


def test_no_wall_between_two_seeds_is_kept_whole_and_flagged_dividing():
    out = _run(0.0)
    assert out.status == "ok", "a 'dividing' verdict is a SUCCESSFUL run"
    assert out.n_pieces == 1
    assert out.arbiter.verdict == "dividing"
    assert out.arbiter.pairs[0].decision == "merge"
    assert set(out.flags.values()) == {"dividing"}
    assert any("dividing" in m.lower() for m in out.messages)


def test_without_the_arbiter_the_same_no_wall_case_over_splits():
    """This is the failure the arbiter exists to prevent: a count-only tool
    splits every dividing cell in the volume."""
    out = _run(0.0, arbiter=False)
    assert out.n_pieces == 2
    assert out.arbiter is None


# --------------------------------------------------------------------------- #
# 2. the mandated sweep: contrast x SNR x spacing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spacing", [ISO, ANISO], ids=["iso", "aniso_5x"])
@pytest.mark.parametrize("contrast,expected", [
    (0.00, "dividing"),
    (0.02, "dividing"),
    (0.05, "dividing"),
    (BELOW_FLOOR_CONTRAST, "dividing"),
    (SENSITIVITY_FLOOR_CONTRAST, "split"),
    (0.15, "split"),
    (0.20, "split"),
    (0.40, "split"),
])
def test_decision_sweep_over_contrast_and_spacing(spacing, contrast, expected):
    out = _run(contrast, spacing=spacing)
    assert out.arbiter.verdict == expected, (
        "contrast={} spacing={} gave {} (threshold {:.3f}, interface median "
        "{:.3f})".format(contrast, spacing, out.arbiter.verdict,
                         out.arbiter.threshold,
                         out.arbiter.pairs[0].interface_median)
    )


def test_the_decision_flips_at_the_stated_contrast():
    """The sensitivity floor is a recorded property of the tool, not folklore."""
    assert _run(BELOW_FLOOR_CONTRAST).arbiter.verdict == "dividing"
    assert _run(SENSITIVITY_FLOOR_CONTRAST).arbiter.verdict == "split"


@pytest.mark.parametrize("contrast", [0.0, 0.05, 0.10, 0.4])
def test_anisotropic_spacing_gives_the_same_decision_as_isotropic(contrast):
    """Same physical contrast -> same decision. A voxel-unit interface band
    would make the anisotropic case disagree."""
    iso = _run(contrast, spacing=ISO)
    aniso = _run(contrast, spacing=ANISO)
    assert iso.arbiter.verdict == aniso.arbiter.verdict
    assert iso.n_pieces == aniso.n_pieces


@pytest.mark.parametrize("noise,contrast,expected", [
    (0.02, 0.10, "split"),        # high SNR: a modest wall is still evidence
    (0.20, 0.10, "dividing"),     # low SNR: the same wall is lost in the noise
])
def test_snr_moves_the_decision(noise, contrast, expected):
    assert _run(contrast, noise=noise).arbiter.verdict == expected


# --------------------------------------------------------------------------- #
# 3. the scale-relative inversion -- the bright-cell case
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("interior", [0.5, 3.0, 10.0])
def test_a_bright_cell_with_a_real_wall_still_splits(interior):
    """`percentile(elevation inside target, 75)` is computed over the target's
    OWN voxels, so a bright cell raises its own bar.  A real wall must still
    clear it."""
    out = _run(0.4 * interior / 0.5, interior=interior)
    assert out.n_pieces == 2
    assert out.arbiter.verdict == "split"


def test_a_bright_cell_without_a_wall_is_still_kept_whole():
    out = _run(0.0, interior=3.0)
    assert out.n_pieces == 1
    assert out.arbiter.verdict == "dividing"


# --------------------------------------------------------------------------- #
# 4. depth attenuation -- why zdepth_correct is mandatory
# --------------------------------------------------------------------------- #
def test_depth_attenuation_can_cost_a_split_and_that_is_the_accepted_tradeoff():
    """The KNOWN limit of the arbiter as shipped, recorded rather than hidden.

    On a strongly depth-attenuated stack a real wall in a dim deep slice can
    fall below the target's own percentile and come back "dividing".  Forcing
    `zdepth_correct` was the earlier answer and it was DROPPED (2026-09-08): it
    cured this case but broke a worse one, flattening any membrane lying in a
    single z-plane.  A smooth robust depth ramp is the intended refinement IF
    real data shows deep cells being misjudged; until then the user's lever is
    to add `zcorrect` to the wall chain themselves, and the engine cautions
    them when they do."""
    assert _run(0.0, z_attenuate=True).arbiter.verdict == "dividing"
    assert _run(0.4, z_attenuate=True).arbiter.verdict == "dividing", (
        "if this becomes 'split', the depth-attenuation limit is gone and this "
        "test should be retired"
    )
    # the same wall, unattenuated, splits -- so attenuation is the cause
    assert _run(0.4).arbiter.verdict == "split"


def test_the_wall_chain_is_not_touched_by_turning_the_arbiter_on():
    spec = ReshapeSpec(target="label", seeds="labels", walls="intensity",
                       params={"wall_channel": "mem"}, arbiter=True)
    assert spec.params["wall_preproc"] == "none"


def test_combining_zcorrect_with_the_arbiter_emits_a_caution():
    """Not forbidden -- the user may genuinely need it on attenuated data --
    but they are told what it costs."""
    labels, guide, mem, _ = two_columns(0.4)
    spec = ReshapeSpec(target="label", seeds="labels", walls="intensity",
                       params={"wall_channel": "mem", "clip": 0.0,
                               "wall_preproc": "zcorrect"},
                       arbiter=True)
    out = reshape_region(labels, {"mem": mem}, spec, target_id=7,
                         seed_layer=guide, spacing=ISO)
    assert any("Depth correction is on" in m for m in out.messages)
    assert any("single z-slice may be flattened" in m for m in out.messages)


def test_no_caution_when_zcorrect_is_not_requested():
    out = _run(0.4)
    assert not any("Depth correction" in m for m in out.messages)


def test_no_caution_when_the_arbiter_is_off():
    labels, guide, mem, _ = two_columns(0.4)
    spec = ReshapeSpec(target="label", seeds="labels", walls="intensity",
                       params={"wall_channel": "mem", "clip": 0.0,
                               "wall_preproc": "zcorrect"},
                       arbiter=False)
    out = reshape_region(labels, {"mem": mem}, spec, target_id=7,
                         seed_layer=guide, spacing=ISO)
    assert not any("Depth correction" in m for m in out.messages)


# --------------------------------------------------------------------------- #
# 5. the percentile knob
# --------------------------------------------------------------------------- #
def test_a_higher_percentile_demands_more_membrane():
    """"How much membrane do you require?" -- raising q must never make the tool
    split MORE readily."""
    marginal = SENSITIVITY_FLOOR_CONTRAST
    assert _run(marginal, percentile=50.0).arbiter.verdict == "split"
    assert _run(marginal, percentile=99.0).arbiter.verdict == "dividing"


def test_low_evidence_flag_marks_a_marginal_but_kept_cut():
    out = _run(SENSITIVITY_FLOOR_CONTRAST)
    assert out.arbiter.verdict == "split"
    assert set(out.flags.values()) == {"low-evidence"}


def test_a_strong_wall_carries_no_flag():
    assert _run(0.4).flags == {}


# --------------------------------------------------------------------------- #
# 6. the interface band is built in PHYSICAL units
# --------------------------------------------------------------------------- #
def test_physical_ball_is_isotropic_in_physical_space():
    iso = physical_ball(1.0, ISO)
    assert iso.shape == (3, 3, 3)
    assert iso[1, 1, 1]
    assert iso[0, 1, 1] and iso[1, 0, 1] and iso[1, 1, 0]


def test_physical_ball_always_reaches_its_six_face_neighbours():
    """The band is sized in PHYSICAL units, but it must never be EMPTY along an
    axis the two pieces genuinely meet across.

    The `dist2 <= r2` test alone defeats the `half = max(1, ...)` floor whenever
    `spacing[axis] > radius`: at (5,1,1) with a 1 um radius the z-neighbour sits
    5 um away and drops out, giving the element zero z-extent.  Two pieces
    stacked in z then never appear to touch and a real interface goes
    unmeasured -- in exactly the anisotropic regime this tool exists for."""
    for spacing in (ISO, ANISO, (10.0, 1.0, 1.0)):
        ball = physical_ball(1.0, spacing)
        centre = tuple(s // 2 for s in ball.shape)
        assert ball[centre]
        for axis in range(3):
            for step in (-1, 1):
                neighbour = list(centre)
                neighbour[axis] += step
                assert ball[tuple(neighbour)], (
                    "spacing {} lost its face neighbour on axis {}".format(
                        spacing, axis)
                )


def test_physical_ball_is_still_sized_in_physical_units():
    """The face-neighbour floor must not turn the band into a voxel-unit blob:
    beyond the 6 faces, reach is still governed by physical distance."""
    iso = physical_ball(3.0, ISO)
    aniso = physical_ball(3.0, ANISO)
    assert iso.sum() > aniso.sum(), (
        "a coarse Z axis must still buy less Z reach for the same radius"
    )
    # a 3 um radius buys 3 voxels of reach in Y/X but only 1 in a 5 um Z axis
    assert iso.shape == (7, 7, 7)
    assert aniso.shape == (3, 7, 7)


def test_physical_ball_grows_with_the_radius():
    small = physical_ball(1.0, ISO)
    large = physical_ball(3.0, ISO)
    assert large.sum() > small.sum()
    assert large.shape[0] > small.shape[0]


def test_interface_mask_is_symmetric_between_the_two_pieces():
    result = np.zeros((6, 12, 10), np.int32)
    result[:, 1:6, 1:9] = 1
    result[:, 6:11, 1:9] = 2
    band = interface_mask(result, 1, 2, spacing=ISO, radius=1.0)
    assert (band & (result == 1)).any()
    assert (band & (result == 2)).any()
    assert not (band & (result == 0)).any()


def test_adjacent_pairs_ignores_pieces_that_do_not_touch():
    result = np.zeros((6, 20, 10), np.int32)
    result[:, 1:5, 1:9] = 1
    result[:, 5:9, 1:9] = 2
    result[:, 15:19, 1:9] = 3          # far away
    assert adjacent_pairs(result, spacing=ISO, radius=1.0) == [(1, 2)]


# --------------------------------------------------------------------------- #
# 7. an interface lying ALONG the coarse axis -- a KNOWN GAP
# --------------------------------------------------------------------------- #
def _z_stacked_verdict(contrast, spacing):
    labels, guide, mem, _ = stacked_in_z(contrast)
    spec = ReshapeSpec(target="label", seeds="labels", walls="intensity",
                       params={"wall_channel": "mem", "clip": 0.0},
                       arbiter=True)
    out = reshape_region(labels, {"mem": mem}, spec, target_id=7,
                         seed_layer=guide, spacing=spacing)
    return out.arbiter.verdict


def test_a_z_normal_interface_decides_identically_at_both_spacings():
    """FIXED (2026-09-08). The two pieces here meet across Z ONLY, so they are
    invisible to an interface band with no z-extent.  Before `physical_ball`
    was floored at its 6 face neighbours, (5,1,1) found no pair at all and
    `arbitrate` still answered "split" -- indistinguishable from an evidenced
    cut, in exactly the anisotropic regime this tool is built for."""
    for contrast in (0.0, 0.4):
        iso = _z_stacked_verdict(contrast, ISO)
        aniso = _z_stacked_verdict(contrast, ANISO)
        assert iso == aniso, (
            "contrast {} decided {} isotropically but {} at 5x".format(
                contrast, iso, aniso)
        )


def test_a_z_normal_interface_is_actually_measured_at_both_spacings():
    """The band must be a real surface (~800 voxels here), not a handful of
    ragged in-plane contacts."""
    for spacing in (ISO, ANISO):
        labels, guide, mem, _ = stacked_in_z(0.4)
        spec = ReshapeSpec(target="label", seeds="labels", walls="intensity",
                           params={"wall_channel": "mem", "clip": 0.0},
                           arbiter=True)
        out = reshape_region(labels, {"mem": mem}, spec, target_id=7,
                             seed_layer=guide, spacing=spacing)
        assert out.arbiter.pairs, "no interface found at spacing {}".format(spacing)
        assert out.arbiter.pairs[0].interface_voxels > 500


def test_two_pieces_with_no_measurable_interface_are_not_reported_as_split():
    """`verdict="split"` must mean "the wall signal justified this cut", never
    "there was nothing to measure".  Two pieces that do not touch at all get
    `"unarbitrated"` and a message saying so."""
    result = np.zeros((6, 24, 10), np.int32)
    result[:, 1:5, 1:9] = 1
    result[:, 18:22, 1:9] = 2          # nowhere near piece 1
    region = result > 0
    elevation = np.zeros(result.shape, np.float32)
    elevation[region] = 0.5
    sink = MessageSink()
    merged, report, flags = arbitrate(result, elevation, region, spacing=ISO,
                                      sink=sink)
    assert report.verdict == "unarbitrated"
    assert report.pairs == []
    assert flags == {}
    assert np.array_equal(merged, result), "an unarbitrated split is left alone"
    assert any("no wall evidence" in m for m in sink.messages)


def test_a_wall_lying_in_a_z_plane_is_detected():
    """FIXED by dropping the mandatory `zdepth_correct` (2026-09-08).

    This was a strict xfail for the whole of R0.  The cause was never the
    arbiter's rule: it was the mandate in front of it.  `zdepth_correct` matches
    every z-slice's mean to the brightest slice's, and a membrane occupying one
    z-plane IS the brightest slice -- so it became the reference, the wall was
    absorbed into the baseline, the interior was lifted to meet it, and the
    corrected 75th percentile inside the target saturated at 1.000 while the
    real wall's band read 0.976.

    Measured and rejected along the way: reading the band off the uncorrected
    elevation while keeping the corrected threshold (8/12 scenarios against
    11/12 -- it compares a median from one normalised distribution against a
    percentile of another).  Dropping the mandate outright fixes it and costs
    only the depth-attenuation case above, which is now a recorded limit."""
    for spacing in (ISO, ANISO):
        assert _z_stacked_verdict(0.4, spacing) == "split"
        assert _z_stacked_verdict(0.0, spacing) == "dividing"


def test_a_z_plane_wall_survives_when_the_arbiter_is_off():
    """The solver itself has no trouble with a z-oriented wall -- the gap above
    is in the arbiter's mandated preprocessing, not in the partition."""
    labels, guide, mem, _ = stacked_in_z(0.4)
    spec = ReshapeSpec(target="label", seeds="labels", walls="intensity",
                       params={"wall_channel": "mem", "clip": 0.0},
                       arbiter=False)
    out = reshape_region(labels, {"mem": mem}, spec, target_id=7,
                         seed_layer=guide, spacing=ISO)
    assert out.n_pieces == 2


def test_zdepth_correct_absorbs_a_z_plane_wall():
    """Documents the HAZARD a user takes on when they choose `zcorrect`
    themselves alongside the arbiter -- the case the engine cautions about.
    Kept after the mandate was dropped, so the mechanism stays visible and any
    future depth correction can be judged against it.

    `zdepth_correct` matches every z-slice's mean to the brightest slice's.  A
    membrane occupying one z-plane IS a bright slice, so it becomes the
    reference: the wall is absorbed into the baseline and every other slice is
    lifted to meet it.  A smooth, robust depth ramp cannot do this, which is
    why that is the candidate replacement."""
    _labels, _guide, mem, region = stacked_in_z(0.4)
    per_slice = lambda vol: [float(vol[z][region[z]].mean())
                             for z in range(vol.shape[0]) if region[z].any()]
    raw = per_slice(mem)
    corrected = per_slice(zdepth_correct(mem))
    assert max(raw) - min(raw) > 0.3, "the wall should be obvious beforehand"
    assert max(corrected) - min(corrected) < 0.05, "and all but gone afterwards"


def test_the_corrected_percentile_saturates_on_a_z_plane_wall():
    """The consequence, if a user does opt into `zcorrect` here: the threshold
    the arbiter compares against reaches the top of the normalised range, so
    nothing can clear it.  This is what the caution message is about."""
    _labels, _guide, mem, region = stacked_in_z(0.4)
    corrected, _c, _r = build_elevation(
        "intensity", apply_preproc(mem, "zcorrect"), region, spacing=ISO, clip=0.0)
    assert float(np.percentile(corrected[region], 75.0)) == pytest.approx(1.0, abs=1e-3)


# --------------------------------------------------------------------------- #
# 8. arbitrate() directly -- the pure seam
# --------------------------------------------------------------------------- #
def _two_block_partition():
    result = np.zeros((6, 12, 10), np.int32)
    result[:, 1:6, 1:9] = 1
    result[:, 6:11, 1:9] = 2
    region = result > 0
    return result, region


def test_arbitrate_merges_the_pair_when_the_interface_is_flat():
    result, region = _two_block_partition()
    elevation = np.zeros(result.shape, np.float32)
    elevation[region] = np.linspace(0, 1, int(region.sum()), dtype=np.float32)
    merged, report, flags = arbitrate(result, elevation, region, spacing=ISO,
                                      percentile=99.0)
    assert report.verdict == "dividing"
    assert set(np.unique(merged)) == {0, 1}
    assert flags == {1: "dividing"}


def test_arbitrate_keeps_the_pair_when_the_interface_is_the_ridge():
    result, region = _two_block_partition()
    elevation = np.zeros(result.shape, np.float32)
    elevation[:, 5:7, 1:9] = 1.0
    merged, report, flags = arbitrate(result, elevation, region, spacing=ISO,
                                      percentile=75.0)
    assert report.verdict == "split"
    assert set(np.unique(merged)) == {0, 1, 2}
    assert np.array_equal(merged, result)


def test_arbitrate_with_one_piece_reports_no_pairs():
    result = np.zeros((4, 6, 6), np.int32)
    result[:, 1:5, 1:5] = 1
    region = result > 0
    merged, report, flags = arbitrate(result, np.zeros(result.shape, np.float32),
                                      region, spacing=ISO)
    assert report.verdict == "no-pairs"
    assert flags == {}
    assert np.array_equal(merged, result)


def test_arbitrate_merges_a_chain_into_one_survivor():
    result = np.zeros((4, 14, 6), np.int32)
    result[:, 1:5, 1:5] = 1
    result[:, 5:9, 1:5] = 2
    result[:, 9:13, 1:5] = 3
    region = result > 0
    elevation = np.zeros(result.shape, np.float32)
    elevation[region] = np.linspace(0, 1, int(region.sum()), dtype=np.float32)
    merged, report, flags = arbitrate(result, elevation, region, spacing=ISO,
                                      percentile=99.0)
    survivors = set(int(v) for v in np.unique(merged)) - {0}
    assert survivors == {1}, "a merged chain keeps the smallest id"
    assert report.verdict == "dividing"
    assert flags == {1: "dividing"}


def test_arbitrate_report_serialises():
    result, region = _two_block_partition()
    elevation = np.zeros(result.shape, np.float32)
    elevation[:, 5:7, 1:9] = 1.0
    _merged, report, _flags = arbitrate(result, elevation, region, spacing=ISO)
    data = report.to_dict()
    assert data["enabled"] is True
    assert data["pairs"][0]["decision"] == "split"
    assert "interface_median" in data["pairs"][0]


# --------------------------------------------------------------------------- #
# 9. the default percentile is CALIBRATED, not incidental
# --------------------------------------------------------------------------- #
def test_the_default_arbiter_percentile_is_75():
    """Pinned in one place so a drift is a deliberate edit, not a silent one."""
    from packages.epicycle_annotator.reshape import DEFAULT_PARAMS

    assert DEFAULT_PARAMS["arbiter_percentile"] == 75.0
    assert DEFAULT_PARAMS["arbiter_low_evidence_percentile"] == 90.0


def test_the_default_percentile_sits_on_a_real_boundary():
    """A calibration test, not a tautology: this fixture DECIDES DIFFERENTLY at
    q=50 and at q=75, so the default is load-bearing.

    Without this, `arbiter_percentile` could drift to 5 or to 99 and the rest of
    the suite would stay green -- every other test either uses a wall far above
    any plausible threshold or none at all."""
    marginal = SENSITIVITY_FLOOR_CONTRAST
    assert _run(marginal, percentile=50.0).arbiter.verdict == "split"
    assert _run(marginal, percentile=75.0).arbiter.verdict == "split"
    assert _run(marginal, percentile=99.0).arbiter.verdict == "dividing"

    # and the default really is the q used when the caller says nothing
    labels, guide, mem, _ = two_columns(marginal)
    spec = ReshapeSpec(target="label", seeds="labels", walls="intensity",
                       params={"wall_channel": "mem", "clip": 0.0}, arbiter=True)
    assert spec.params["arbiter_percentile"] == 75.0
    default_run = reshape_region(labels, {"mem": mem}, spec, target_id=7,
                                 seed_layer=guide, spacing=ISO)
    assert default_run.arbiter.verdict == _run(marginal, percentile=75.0).arbiter.verdict
    assert default_run.arbiter.percentile == 75.0


@pytest.mark.parametrize("q,expected", [(5.0, "split"), (99.9, "dividing")])
def test_an_extreme_percentile_moves_the_verdict(q, expected):
    """The knob has to actually reach the decision at both ends."""
    assert _run(SENSITIVITY_FLOOR_CONTRAST,
                percentile=q).arbiter.verdict == expected


def test_a_median_exactly_on_the_threshold_counts_as_evidence():
    """The comparison is `median >= threshold`, so a tie is a SPLIT. Pinned
    because flipping it to `>` would silently make every borderline cut merge."""
    result = np.zeros((6, 12, 10), np.int32)
    result[:, 1:6, 1:9] = 1
    result[:, 6:11, 1:9] = 2
    region = result > 0
    elevation = np.zeros(result.shape, np.float32)
    elevation[region] = 0.5           # every voxel identical -> median == pctile
    _merged, report, _flags = arbitrate(result, elevation, region, spacing=ISO,
                                        percentile=75.0)
    pair = report.pairs[0]
    assert pair.interface_median == pytest.approx(pair.threshold)
    assert pair.decision == "split"
    assert report.verdict == "split"
