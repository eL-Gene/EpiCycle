"""The merged Active contour, the pure function.

``refine.contour_refine`` replaces the two 2024 contours with ONE method in
two modes.  What is pinned here is what the 2026-09-21 audit found broken:

* the balloon was a NO-OP over its whole documented range, because skimage
  gates it on ``gimage > threshold / |balloon|`` and the automatic threshold
  sat at ~0.974 -- so the gate is now passed EXPLICITLY, and a push that is
  not bigger than it acts nowhere and SAYS so (``push_acts_below``);
* the limit was one-sided -- it capped growth and never restrained collapse;
* the brightness mode deleted the object on membrane data, because the
  interior and the background are one intensity population -- so the two
  populations are measured in a BAND around the outline;
* neighbours are a fence.

Pure: numpy / scipy / skimage only, no Qt.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from packages.epicycle_annotator.refine import (
    CONTOUR_MODES,
    ContourResult,
    contour_refine,
    morphacwe_refine,
    push_acts_below,
)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _sphere(shape=(32, 32, 32), radius=8, label=3, centre=None):
    centre = centre or tuple(s // 2 for s in shape)
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    r = np.sqrt((zz - centre[0]) ** 2 + (yy - centre[1]) ** 2
                + (xx - centre[2]) ** 2)
    vol = np.zeros(shape, np.int32)
    vol[r <= radius] = label
    return vol


def _flat(vol):
    """A wall picture with no wall anywhere: gimage == 1 everywhere, so
    the image-attachment force is zero and only the push can move the
    outline -- which isolates the push."""
    return np.zeros(vol.shape, np.float32)


def _open_top_cell():
    """The audit's failure case: a membrane box with NO wall on top, a dark
    interior (0.08) that looks like the background (0.07), a bright rim
    (0.62) -- and a neighbour."""
    rng = np.random.default_rng(0)
    Z, Y, X = 24, 48, 48
    zz, yy, xx = np.mgrid[:Z, :Y, :X]
    img = np.full((Z, Y, X), 0.07) + 0.02 * rng.random((Z, Y, X))
    cell = (zz >= 4) & (zz < 20) & (yy >= 12) & (yy < 36) & (xx >= 12) & (xx < 36)
    inner = (zz >= 6) & (zz < 20) & (yy >= 14) & (yy < 34) & (xx >= 14) & (xx < 34)
    img[cell & ~inner] = 0.62
    img[inner] = 0.08
    vol = np.zeros((Z, Y, X), np.int32)
    vol[cell] = 5
    vol[4:20, 12:36, 36:44] = 6
    return vol, img


# --------------------------------------------------------------------------- #
# the gate -- the knob that did nothing now says when it does nothing
# --------------------------------------------------------------------------- #
class TestThePushGate:
    def test_a_full_push_acts_below_0_15_at_the_default_gate(self):
        """Point 2's worked number."""
        assert push_acts_below(1.0, 0.30, 100.0) == pytest.approx(0.1528, abs=1e-3)

    @pytest.mark.parametrize("b", [0.0, 0.1, 0.2, 0.3, -0.3])
    def test_a_push_not_bigger_than_the_gate_acts_nowhere(self, b):
        assert push_acts_below(b, 0.30) is None

    def test_just_above_the_gate_it_acts_a_little(self):
        v = push_acts_below(0.31, 0.30)
        assert v is not None and 0.0 < v < 0.02

    def test_a_gate_of_zero_lets_it_act_everywhere(self):
        assert push_acts_below(0.2, 0.0) == float("inf")

    def test_it_is_monotone_in_both_knobs(self):
        assert push_acts_below(0.9, 0.3) > push_acts_below(0.6, 0.3)
        assert push_acts_below(0.9, 0.2) > push_acts_below(0.9, 0.3)
        assert push_acts_below(0.9, 0.3, 50.0) > push_acts_below(0.9, 0.3, 200.0)

    def test_below_the_gate_the_contour_does_not_run_and_says_why(self):
        vol = _sphere()
        res = contour_refine(vol, 3, mode="snap", elevation=_flat(vol),
                             balloon=0.2, gate=0.30)
        assert res[1] == 0
        assert res.details["reason"] == "push_below_gate"
        assert np.array_equal(res[0] == 3, vol == 3)
        assert "0.30" in res[2] and "0.20" in res[2]

    def test_lower_the_gate_and_the_same_push_moves_it(self):
        vol = _sphere()
        res = contour_refine(vol, 3, mode="snap", elevation=_flat(vol),
                             balloon=0.2, gate=0.10, iterations=20)
        assert res.details["reason"] is None
        assert res[1] > 0

    def test_the_gate_is_passed_explicitly_not_auto(self):
        """On a FLAT wall picture gimage == 1 everywhere, so skimage's
        'auto' threshold (its 40th percentile) is 1.0 and gate/|push| >= 1:
        with 'auto' a push of 0.5 would do nothing.  It grows here."""
        vol = _sphere(radius=6)
        res = contour_refine(vol, 3, mode="snap", elevation=_flat(vol),
                             balloon=0.5, gate=0.30, iterations=10,
                             limit_um=20)
        assert int((res[0] == 3).sum()) > int((vol == 3).sum())


class TestTheBalloonSign:
    def test_a_positive_push_grows_and_a_negative_one_shrinks(self):
        vol = _sphere(radius=9)
        n0 = int((vol == 3).sum())
        grow = contour_refine(vol, 3, mode="snap", elevation=_flat(vol),
                              balloon=1.0, iterations=3, limit_um=20)
        # One step: the push erodes a voxel a step AND the two tidy passes
        # shave the curvature, so a few steps take a sphere to nothing.
        shrink = contour_refine(vol, 3, mode="snap", elevation=_flat(vol),
                                balloon=-1.0, iterations=1, limit_um=20)
        assert int((grow[0] == 3).sum()) > n0
        assert 0 < int((shrink[0] == 3).sum()) < n0


# --------------------------------------------------------------------------- #
# the two-sided limit, in microns
# --------------------------------------------------------------------------- #
class TestTheTwoSidedLimit:
    def _dist(self, mask, spacing):
        from scipy.ndimage import distance_transform_edt

        return (distance_transform_edt(~mask, sampling=spacing),
                distance_transform_edt(mask, sampling=spacing))

    def test_growth_stops_at_the_limit(self):
        vol = _sphere(radius=6)
        res = contour_refine(vol, 3, mode="snap", elevation=_flat(vol),
                             balloon=1.0, iterations=40, limit_um=2.0)
        d_out, _ = self._dist(vol == 3, (1, 1, 1))
        grew = (res[0] == 3) & (vol != 3)
        assert grew.any()
        assert float(d_out[grew].max()) <= 2.0 + 1e-6
        assert res.details["hit_limit"] is True

    def test_COLLAPSE_stops_at_the_limit_too(self):
        """The half that did not exist: a shrinking contour is held."""
        vol = _sphere(radius=9)
        res = contour_refine(vol, 3, mode="snap", elevation=_flat(vol),
                             balloon=-1.0, iterations=40, limit_um=2.0)
        _, d_in = self._dist(vol == 3, (1, 1, 1))
        shrank = (vol == 3) & (res[0] != 3)
        assert shrank.any()
        assert float(d_in[shrank].max()) <= 2.0 + 1e-6
        # the core deeper than the limit is untouched
        assert np.all(res[0][d_in > 2.0] == 3)

    def test_the_limit_is_physical_on_anisotropic_data(self):
        """3 um on (3, 1, 1): one voxel through depth, three across."""
        vol = _sphere(shape=(24, 40, 40), radius=6)
        res = contour_refine(vol, 3, mode="snap", elevation=_flat(vol),
                             spacing=(3.0, 1.0, 1.0), balloon=1.0,
                             iterations=40, limit_um=3.0)
        grew = (res[0] == 3) & (vol != 3)
        z_orig = np.where((vol == 3).any(axis=(1, 2)))[0]
        z_new = np.where((res[0] == 3).any(axis=(1, 2)))[0]
        assert z_new.min() >= z_orig.min() - 1
        assert z_new.max() <= z_orig.max() + 1
        x_orig = np.where((vol == 3).any(axis=(0, 1)))[0]
        x_new = np.where((res[0] == 3).any(axis=(0, 1)))[0]
        assert x_new.max() - x_orig.max() in (2, 3)
        assert grew.any()

    def test_nothing_moved_at_the_limit_says_so(self):
        """Every point of the outline already at the limit -> reason
        'limit', which is state 9's second contour sentence."""
        vol = _sphere(radius=6)
        res = contour_refine(vol, 3, mode="snap", elevation=_flat(vol),
                             balloon=1.0, iterations=10, limit_um=0.5)
        assert res[1] == 0
        assert res.details["reason"] == "limit"


# --------------------------------------------------------------------------- #
# neighbours are a fence
# --------------------------------------------------------------------------- #
class TestNeighboursAreAFence:
    @pytest.mark.parametrize("mode", ["snap", "brightness"])
    def test_no_neighbour_voxel_is_ever_taken(self, mode):
        vol = _sphere(radius=7)
        vol[:, :, 24:] = 9          # a neighbour flush against one side
        img = np.where(vol == 3, 0.9, 0.1)
        res = contour_refine(vol, 3, mode=mode, elevation=_flat(vol),
                             image_roi=img, balloon=1.0, iterations=20,
                             limit_um=20)
        assert not np.any((res[0] == 3) & (vol == 9))
        # the result carries this object and background only
        assert set(np.unique(res[0])) <= {0, 3}


# --------------------------------------------------------------------------- #
# the brightness mode's band -- the fix for the deletion
# --------------------------------------------------------------------------- #
class TestTheBrightnessBand:
    def test_UNBANDED_it_deletes_the_open_top_cell(self):
        """The audit's failure, reproduced -- this is what the band is for.
        If this ever stops being 0 the synthetic case no longer shows the
        bug and the next test proves nothing."""
        vol, img = _open_top_cell()
        res = morphacwe_refine(vol, 5, img, distance_clamp=0)
        assert int((res[0] == 5).sum()) == 0

    def test_the_band_is_a_SHELL_not_the_whole_region(self):
        """Audit C-H1: `(d_out <= b) | (d_in <= b)` is True everywhere --
        d_out is 0 inside and d_in is 0 outside -- so the "band" was the
        whole crop and the brightness mode still compared whole interior
        with whole exterior."""
        from packages.epicycle_annotator.refine import _band_around

        m = np.zeros((40, 40, 40), bool)
        m[10:30, 10:30, 10:30] = True
        band = _band_around(m, 2.0, (1, 1, 1))
        assert band.mean() < 0.5
        assert not band[20, 20, 20], "the deep interior is in the band"
        assert not band[0, 0, 0], "the far background is in the band"
        assert band[10, 20, 20] and band[9, 20, 20]

    def test_the_band_alone_keeps_the_open_top_cell(self):
        """With the limit out of the way (huge), only the BAND can stop the
        collapse -- the old whole-region "band" let it go."""
        vol, img = _open_top_cell()
        n0 = int((vol == 5).sum())
        res = contour_refine(vol, 5, mode="brightness", image_roi=img,
                             limit_um=1000.0, band_um=3.0)
        n1 = int((res[0] == 5).sum())
        assert n1 > 0.70 * n0, "{} -> {}".format(n0, n1)

    @pytest.mark.parametrize("spacing", [(1, 1, 1), (3, 1, 1)])
    def test_BANDED_it_does_not(self, spacing):
        vol, img = _open_top_cell()
        n0 = int((vol == 5).sum())
        res = contour_refine(vol, 5, mode="brightness", image_roi=img,
                             spacing=spacing, limit_um=3.0)
        n1 = int((res[0] == 5).sum())
        assert n1 > 0.70 * n0, "{} -> {}".format(n0, n1)

    def test_the_band_defaults_to_the_limit(self):
        """A band narrower than the limit must give a DIFFERENT answer --
        otherwise this proves nothing (with the whole-region "band" every
        band_um was the same)."""
        vol, img = _open_top_cell()
        a = contour_refine(vol, 5, mode="brightness", image_roi=img,
                           limit_um=3.0)
        b = contour_refine(vol, 5, mode="brightness", image_roi=img,
                           limit_um=3.0, band_um=3.0)
        c = contour_refine(vol, 5, mode="brightness", image_roi=img,
                           limit_um=3.0, band_um=1.0)
        assert np.array_equal(a[0], b[0])
        assert not np.array_equal(a[0], c[0])

    def test_nothing_outside_the_band_ever_changes(self):
        """'Outside the band' = further than band_um from the outline on
        EITHER side -- with a huge limit, so the limit cannot be what
        holds it."""
        from scipy.ndimage import distance_transform_edt

        vol, img = _open_top_cell()
        res = contour_refine(vol, 5, mode="brightness", image_roi=img,
                             limit_um=1000.0, band_um=2.0)
        m = vol == 5
        far = ((~m & (distance_transform_edt(~m) > 2.0))
               | (m & (distance_transform_edt(m) > 2.0)))
        assert far.any()
        assert np.array_equal((res[0] == 5)[far], m[far])


# --------------------------------------------------------------------------- #
# 3D vs a plane orientation, on a cylinder
# --------------------------------------------------------------------------- #
class TestOrientationParity:
    def test_3D_and_XY_agree_in_the_middle_of_a_cylinder(self):
        """A cylinder along z, segmented 3 voxels too thin, inside a ring
        of wall: both orientations must move the outline out onto the
        ring, and agree on the middle slice."""
        from packages.epicycle_annotator.reshape.walls import build_elevation

        Z, Y, X = 30, 40, 40
        zz, yy, xx = np.mgrid[:Z, :Y, :X]
        r = np.sqrt((yy - 20) ** 2 + (xx - 20) ** 2)
        img = np.exp(-((r - 10) ** 2) / 2.0).astype(np.float32)
        vol = np.zeros((Z, Y, X), np.int32)
        vol[(r < 7) & (zz >= 4) & (zz < 26)] = 3
        elev, _c, _r = build_elevation("membrane", img, vol == 3)
        kw = dict(mode="snap", elevation=elev, balloon=0.0, limit_um=5.0,
                  iterations=60)
        a = contour_refine(vol, 3, orientation="3D", **kw)
        b = contour_refine(vol, 3, orientation="XY", sampling="every", **kw)
        ma, mb = (a[0] == 3)[15], (b[0] == 3)[15]
        dice = 2 * np.count_nonzero(ma & mb) / (ma.sum() + mb.sum())
        assert ma.sum() > (vol == 3)[15].sum()
        assert mb.sum() > (vol == 3)[15].sum()
        assert dice > 0.9, dice


# --------------------------------------------------------------------------- #
# the contract
# --------------------------------------------------------------------------- #
class TestTheContract:
    def test_it_unpacks_as_the_shared_three(self):
        vol = _sphere()
        res = contour_refine(vol, 3, mode="snap", elevation=_flat(vol),
                             balloon=1.0, iterations=3)
        assert isinstance(res, ContourResult) and isinstance(res, tuple)
        result_roi, n_mod, msg = res
        assert result_roi.shape == vol.shape and isinstance(msg, str)
        assert n_mod == int(np.count_nonzero((result_roi == 3) != (vol == 3)))

    def test_mode_by_index_is_the_panel_wire_format(self):
        assert CONTOUR_MODES == ("snap", "brightness")
        vol = _sphere()
        a = contour_refine(vol, 3, mode=0, elevation=_flat(vol), balloon=0.2)
        assert a.details["mode"] == "snap"

    def test_a_missing_object_is_an_answer(self):
        vol = _sphere()
        res = contour_refine(vol, 99, mode="snap", elevation=_flat(vol))
        assert res[1] == 0 and res.details["reason"] == "missing"

    def test_an_oversized_region_is_refused_before_any_work(self):
        vol = _sphere()
        res = contour_refine(vol, 3, mode="snap", elevation=_flat(vol),
                             size_budget=100)
        assert res.details["reason"] == "too_large" and res[1] == 0

    def test_snap_without_a_wall_picture_raises(self):
        with pytest.raises(ValueError):
            contour_refine(_sphere(), 3, mode="snap")

    def test_brightness_without_a_channel_raises(self):
        with pytest.raises(ValueError):
            contour_refine(_sphere(), 3, mode="brightness")

    def test_the_input_is_not_written(self):
        vol = _sphere()
        before = vol.copy()
        contour_refine(vol, 3, mode="snap", elevation=_flat(vol),
                       balloon=1.0, iterations=5)
        assert np.array_equal(vol, before)

    def test_a_3D_run_reports_its_time_for_the_cost_table(self):
        vol = _sphere()
        res = contour_refine(vol, 3, mode="snap", elevation=_flat(vol),
                             balloon=1.0, iterations=3)
        assert len(res.details["plane_seconds"]) == 1

    def test_a_set_cancel_event_does_not_stop_a_3D_run_half_way(self):
        """3D is ONE call we do not own -- it cannot be interrupted, and
        the orchestrator's Stop throws the answer away instead."""
        vol = _sphere()
        ev = threading.Event()
        ev.set()
        res = contour_refine(vol, 3, mode="snap", elevation=_flat(vol),
                             balloon=1.0, iterations=3, cancel_event=ev)
        assert res.details["reason"] is None


# --------------------------------------------------------------------------- #
# B2 audit: the fence has its own reason; the wall picture is built with it
# --------------------------------------------------------------------------- #
class TestTheFenceHasItsOwnReason:
    def test_every_move_into_a_neighbour_says_fence(self):
        """UX-M3: an object boxed in on every side by a neighbour, pushed
        out -- every move is into the neighbour, the fence takes it all."""
        vol = np.full((20, 20, 20), 9, np.int32)
        vol[6:14, 6:14, 6:14] = 3
        res = contour_refine(vol, 3, mode="snap", elevation=_flat(vol),
                             balloon=1.0, iterations=3, limit_um=20)
        assert res[1] == 0
        assert res.details["reason"] == "fence"


class TestContourWithWalls:
    """C-M3: the prepare + wall-picture step moved onto the worker."""

    def _ring(self):
        Z, Y, X = 20, 40, 40
        zz, yy, xx = np.mgrid[:Z, :Y, :X]
        r = np.sqrt((yy - 20) ** 2 + (xx - 20) ** 2)
        img = np.exp(-((r - 10) ** 2) / 2.0).astype(np.float32)
        vol = np.zeros((Z, Y, X), np.int32)
        vol[(r < 7) & (zz >= 4) & (zz < 16)] = 3
        return vol, img

    def test_it_builds_the_picture_and_hands_it_back(self):
        from packages.epicycle_annotator.refine import contour_with_walls

        vol, img = self._ring()
        calls = []
        res = contour_with_walls(
            vol, 3, mode=0, wall_kind="membrane", channel_roi=img,
            prep_chain="none", blur_um=2.0, spacing=(1, 1, 1),
            limit_um=5.0, iterations=30,
            progress=lambda *a: calls.append(a))
        elev = res.details["elevation"]
        assert elev is not None and elev.shape == vol.shape
        assert calls[0] == ("elevation", 0, 0)
        assert res[1] > 0

    def test_it_matches_the_main_thread_recipe(self):
        """Byte-for-byte the elevation the old main-thread path built."""
        from packages.epicycle_annotator.refine import contour_with_walls
        from packages.epicycle_annotator.reshape.preproc import apply_preproc
        from packages.epicycle_annotator.reshape.walls import build_elevation

        vol, img = self._ring()
        res = contour_with_walls(
            vol, 3, mode="snap", wall_kind="membrane", channel_roi=img,
            prep_chain="bgsub", blur_um=3.0, spacing=(3, 1, 1),
            iterations=10)
        want, _c, _r = build_elevation(
            "membrane", apply_preproc(img, "bgsub"), vol == 3,
            spacing=(3.0, 1.0, 1.0), smooth=3.0)
        assert np.array_equal(res.details["elevation"], want)

    def test_brightness_builds_no_picture(self):
        from packages.epicycle_annotator.refine import contour_with_walls

        vol, img = self._ring()
        calls = []
        res = contour_with_walls(
            vol, 3, mode="brightness", channel_roi=img, prep_chain="none",
            iterations=10, progress=lambda *a: calls.append(a))
        assert res.details["elevation"] is None
        assert calls[0] == ("prepare", 0, 0)

    def test_the_shape_and_probability_kinds_need_no_channel(self):
        from packages.epicycle_annotator.refine import contour_with_walls

        vol, _img = self._ring()
        a = contour_with_walls(vol, 3, mode=0, wall_kind="distance",
                               iterations=5)
        assert a.details["elevation"] is not None
        b = contour_with_walls(vol, 3, mode=0, wall_kind="probability",
                               prob_roi=(vol == 3).astype(np.float32),
                               iterations=5)
        assert b.details["elevation"] is not None
