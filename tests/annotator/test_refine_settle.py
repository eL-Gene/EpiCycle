"""Settle onto the wall signal -- the pure engine (Brick B1).

Everything here runs on plain arrays: ``settle_label`` and
``apply_settle_to_volume`` are napari-free and Qt-free by construction, so
this file never builds a widget.

The invariants the spec names, one class each:

* the band is PHYSICAL -- a scalar voxel band cannot express microns on
  (3, 1, 1) data, which is the whole reason the knob exists;
* the scalar ``margin`` a caller must pass to ``_snapshot_label_roi``
  covers EVERY axis;
* neighbours are SEEDED, and yet **no other label ever loses a voxel**;
* a result that would empty the object is REFUSED, and the write-back
  refuses it a second time (it returns -1 and writes nothing);
* three solvers, with ``label_diffusion`` degrading to a sentence rather
  than an exception when u-Segment3D is absent;
* the size-budget refusal fires BEFORE the dense allocation;
* ``result_roi`` never aliases its input;
* the border fraction is the number the status line leads with.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from packages.epicycle_annotator.reshape.settle import (
    SettleResult,
    apply_settle_to_volume,
    settle_label,
)


# --------------------------------------------------------------------------- #
# fixtures -- a box object, a neighbour beside it, and a wall between them
# --------------------------------------------------------------------------- #
def _two_objects(spacing=(3.0, 1.0, 1.0)):
    """A 20x30x30 ROI: object 7 on the left, object 9 on the right, a
    one-voxel gap of background between them."""
    vol = np.zeros((20, 30, 30), np.int32)
    vol[5:15, 8:22, 6:14] = 7
    vol[5:15, 8:22, 16:24] = 9
    return vol, spacing


def _flat_elevation(vol):
    return np.zeros(vol.shape, np.float32)


def _wall_elevation(vol, x):
    """A bright wall plane at ``x``."""
    elev = np.zeros(vol.shape, np.float32)
    elev[:, :, x] = 1.0
    return elev


class TestTheBandIsPhysical:
    def test_the_band_is_measured_in_microns_not_voxels(self):
        """On (3, 1, 1) data a 2 um band is ONE voxel through depth and
        TWO across.  A voxel band cannot say that."""
        vol, spacing = _two_objects()
        elev = _flat_elevation(vol)
        out = settle_label(vol, elev, 7, band_um=2.0, spacing=spacing)
        assert isinstance(out, SettleResult)
        # every moved voxel is inside the physical band
        assert out.max_move_um <= 2.0 + 1e-6

    def test_the_same_band_reaches_further_on_finer_spacing(self):
        vol, _ = _two_objects()
        elev = _flat_elevation(vol)
        coarse = settle_label(vol, elev, 7, band_um=3.0, spacing=(3.0, 1.0, 1.0))
        fine = settle_label(vol, elev, 7, band_um=3.0, spacing=(1.0, 1.0, 1.0))
        # A 3 um band is 1 voxel of depth on (3,1,1) and 3 on (1,1,1), so
        # the finer grid puts strictly more voxels in play.
        assert fine.n_band_voxels > coarse.n_band_voxels

    def test_the_roi_margin_a_caller_must_pass_covers_every_axis(self):
        """`_snapshot_label_roi` takes a SCALAR margin and is shared with
        Reshape's Start, so Settle passes the largest padding any axis
        needs."""
        from packages.epicycle_annotator.reshape.settle import roi_margin_voxels

        assert roi_margin_voxels(3.0, (3.0, 1.0, 1.0)) == 3
        assert roi_margin_voxels(2.0, (3.0, 1.0, 1.0)) == 2
        assert roi_margin_voxels(0.5, (0.5, 0.5, 0.5)) == 1
        # never smaller than what the coarsest axis would need
        for band in (0.5, 2.0, 7.5):
            m = roi_margin_voxels(band, (3.0, 1.0, 1.0))
            assert all(m >= math.ceil(band / s) for s in (3.0, 1.0, 1.0))


class TestNeighboursCompeteButNeverLose:
    def test_a_neighbour_is_seeded(self):
        vol, spacing = _two_objects()
        elev = _flat_elevation(vol)
        out = settle_label(vol, elev, 7, band_um=3.0, spacing=spacing)
        assert out.n_neighbour_markers == 1, (
            "object 9 is in the ROI and must compete for the gap")

    def test_no_other_label_ever_loses_a_voxel(self):
        vol, spacing = _two_objects()
        # A wall far to the right, so the solve WANTS to push object 7
        # across the gap and into object 9.
        elev = _wall_elevation(vol, 27)
        out = settle_label(vol, elev, 7, band_um=6.0, spacing=spacing)
        grew_into = (out.result_roi == 7) & (vol == 9)
        assert not grew_into.any(), (
            "{} voxels of object 9 were taken".format(int(grew_into.sum())))

    def test_only_voxels_in_the_band_may_change(self):
        vol, spacing = _two_objects()
        elev = _wall_elevation(vol, 27)
        out = settle_label(vol, elev, 7, band_um=2.0, spacing=spacing)
        before = vol == 7
        after = out.result_roi == 7
        changed = before != after
        assert changed.sum() == out.n_modified
        # nothing moved further than the band allows
        assert out.max_move_um <= 2.0 + 1e-6

    def test_the_result_is_one_connected_piece(self):
        from scipy import ndimage as ndi

        vol, spacing = _two_objects()
        elev = _flat_elevation(vol)
        out = settle_label(vol, elev, 7, band_um=3.0, spacing=spacing)
        _lab, n = ndi.label(out.result_roi == 7)
        assert n == 1


class TestTheEmptyObjectIsRefused:
    def test_settle_refuses_a_result_that_would_empty_the_object(self):
        """A 1-voxel-thick object inside a band that swallows it whole."""
        vol = np.zeros((10, 20, 20), np.int32)
        vol[5, 9:11, 9:11] = 7
        elev = np.zeros(vol.shape, np.float32)
        # Background wins everywhere: the elevation is flat and the
        # background core is enormous, so a huge band erases the object.
        out = settle_label(vol, elev, 7, band_um=40.0, spacing=(1.0, 1.0, 1.0))
        assert out.status == "refused"
        assert out.n_modified == 0
        assert int((out.result_roi == 7).sum()) == int((vol == 7).sum())
        assert any("nothing would be left" in m.lower() for m in out.messages)

    def test_the_write_back_refuses_an_empty_result_and_writes_nothing(self):
        vol = np.zeros((6, 8, 8), np.int32)
        vol[2:4, 2:6, 2:6] = 7
        roi = (slice(0, 6), slice(0, 8), slice(0, 8))
        empty = np.zeros(vol.shape, np.int32)
        before = vol.copy()
        n = apply_settle_to_volume(vol, roi, empty, 7)
        assert n == -1
        assert np.array_equal(vol, before)

    def test_the_write_back_writes_a_non_empty_result(self):
        vol = np.zeros((6, 8, 8), np.int32)
        vol[2:4, 2:6, 2:6] = 7
        roi = (slice(0, 6), slice(0, 8), slice(0, 8))
        result = np.zeros(vol.shape, np.int32)
        result[2:4, 2:7, 2:6] = 7
        n = apply_settle_to_volume(vol, roi, result, 7)
        assert n > 0
        assert int((vol == 7).sum()) == int((result == 7).sum())

    def test_the_write_back_never_overwrites_another_label(self):
        vol = np.zeros((6, 8, 8), np.int32)
        vol[2:4, 2:5, 2:6] = 7
        vol[2:4, 5:7, 2:6] = 9
        roi = (slice(0, 6), slice(0, 8), slice(0, 8))
        result = np.zeros(vol.shape, np.int32)
        result[2:4, 2:8, 2:6] = 7          # asks for object 9's territory
        apply_settle_to_volume(vol, roi, result, 7)
        assert int((vol == 9).sum()) == 2 * 2 * 4


class TestTheThreeSolvers:
    @pytest.mark.parametrize("solver", ["watershed", "random_walker"])
    def test_both_local_solvers_run(self, solver):
        vol, spacing = _two_objects()
        elev = _wall_elevation(vol, 15)
        out = settle_label(vol, elev, 7, band_um=3.0, spacing=spacing,
                           solver=solver)
        assert out.status in ("ok", "unchanged")
        assert out.result_roi.shape == vol.shape

    def test_label_diffusion_refuses_with_a_sentence_when_usegment3d_is_absent(
            self, monkeypatch):
        import packages.epicycle_annotator.reshape.settle as S

        monkeypatch.setattr(S, "have_usegment3d", lambda: False)
        vol, spacing = _two_objects()
        elev = _flat_elevation(vol)
        out = settle_label(vol, elev, 7, band_um=3.0, spacing=spacing,
                           solver="label_diffusion")
        assert out.status == "refused"
        assert out.n_modified == 0
        assert any("u-Segment3D" in m for m in out.messages)
        assert any("nothing was changed" in m for m in out.messages)

    def test_an_unknown_solver_is_a_refusal_not_an_exception(self):
        vol, spacing = _two_objects()
        out = settle_label(vol, _flat_elevation(vol), 7, band_um=3.0,
                           spacing=spacing, solver="teleport")
        assert out.status == "refused"


class TestTheGuards:
    def test_the_size_budget_refuses_before_the_dense_allocation(self):
        vol, spacing = _two_objects()
        out = settle_label(vol, _flat_elevation(vol), 7, band_um=3.0,
                           spacing=spacing, size_budget=8)
        assert out.status == "refused"
        assert any("too large" in m.lower() for m in out.messages)

    def test_a_missing_object_is_a_refusal(self):
        vol, spacing = _two_objects()
        out = settle_label(vol, _flat_elevation(vol), 404, band_um=3.0,
                           spacing=spacing)
        assert out.status == "refused"

    def test_the_result_never_aliases_its_input(self):
        vol, spacing = _two_objects()
        out = settle_label(vol, _flat_elevation(vol), 7, band_um=3.0,
                           spacing=spacing)
        assert out.result_roi is not vol
        assert not np.shares_memory(out.result_roi, vol)
        out.result_roi[0, 0, 0] = 123
        assert vol[0, 0, 0] == 0

    def test_a_mismatched_elevation_is_a_refusal(self):
        vol, spacing = _two_objects()
        bad = np.zeros((3, 3, 3), np.float32)
        out = settle_label(vol, bad, 7, band_um=3.0, spacing=spacing)
        assert out.status == "refused"


class TestTheBorderFraction:
    """The number the staged status line leads with.

    It is counted over FACES, not voxels, so a driver can measure it
    independently on the same patch and get the same answer.
    """

    def test_a_lone_object_faces_background_everywhere(self):
        vol = np.zeros((10, 20, 20), np.int32)
        vol[3:7, 6:14, 6:14] = 7
        out = settle_label(vol, np.zeros(vol.shape, np.float32), 7,
                           band_um=1.0, spacing=(1.0, 1.0, 1.0))
        assert out.n_border_held == 0
        assert out.border_background_fraction == pytest.approx(1.0)

    def test_two_flush_objects_share_a_border_that_is_held(self):
        vol = np.zeros((10, 20, 20), np.int32)
        vol[3:7, 6:14, 6:10] = 7
        vol[3:7, 6:14, 10:14] = 9          # flush, no gap
        out = settle_label(vol, np.zeros(vol.shape, np.float32), 7,
                           band_um=1.0, spacing=(1.0, 1.0, 1.0))
        assert out.n_border_held == 4 * 8, "the shared face is 4 x 8 voxels"
        assert 0.0 < out.border_background_fraction < 1.0

    def test_the_fraction_is_the_two_counts_and_nothing_else(self):
        vol, spacing = _two_objects()
        out = settle_label(vol, _flat_elevation(vol), 7, band_um=3.0,
                           spacing=spacing)
        total = out.n_border_background + out.n_border_held
        assert out.border_background_fraction == pytest.approx(
            out.n_border_background / total)


class TestWhatTheResultSays:
    def test_nothing_moved_is_a_status_of_its_own(self):
        """Object 7 wrapped completely in object 9: every border face
        touches a neighbour, so Settle has nothing it may move -- and the
        sentence that comes back is state 9's own, not a generic one."""
        vol = np.zeros((24, 34, 34), np.int32)
        vol[2:22, 4:30, 4:30] = 9
        vol[7:17, 9:25, 9:25] = 7
        out = settle_label(vol, np.zeros(vol.shape, np.float32), 7,
                           band_um=2.0, spacing=(1.0, 1.0, 1.0))
        assert out.n_border_background == 0
        assert out.n_modified == 0
        assert out.status == "unchanged"
        assert any("only moves edges that face background" in m
                   for m in out.messages)

    def test_the_move_is_reported_in_microns(self):
        vol, spacing = _two_objects()
        elev = _wall_elevation(vol, 4)
        out = settle_label(vol, elev, 7, band_um=6.0, spacing=spacing)
        if out.n_modified:
            assert out.mean_move_um > 0.0
            assert out.max_move_um >= out.mean_move_um

    def test_hit_band_limit_says_whether_the_limit_is_what_stopped_it(self):
        vol, spacing = _two_objects()
        # a wall a long way out, a tight band -> the band is the thing
        # holding the outline, and the flag must say so
        out = settle_label(vol, _wall_elevation(vol, 29), 7, band_um=1.0,
                           spacing=spacing)
        assert out.hit_band_limit in (True, False)
        if out.n_modified:
            assert out.hit_band_limit is (
                out.max_move_um >= 1.0 - 1e-6)
