"""The serial-2D Active contour and its SDF rebuild.

A plane orientation traces a SAMPLE of slices and fills the rest in with the
annotator's own ``interpolate_slices_sdf`` -- the same interpolation Accept
already uses -- so what is pinned here is the sampler (the ends always in,
an even spread, the floor and the cap), the progress it reports, the Stop
between planes, the rebuild, and the public ``interpolation.get_slice``
alias the planes are read through.

Pure: numpy / scipy / skimage only, no Qt.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from packages.epicycle_annotator import interpolation as interp_mod
from packages.epicycle_annotator.refine import (
    SLICE_MAX_COUNT,
    SLICE_MIN_COUNT,
    SLICE_SAMPLINGS,
    contour_refine,
    sample_slice_indices,
)


def _cylinder(z0=2, z1=46, shape=(48, 40, 40), radius=7, label=4):
    """A cylinder along z over ``z0..z1-1`` -- 44 slices by default, the
    spec's worked example."""
    zz, yy, xx = np.mgrid[:shape[0], :shape[1], :shape[2]]
    r = np.sqrt((yy - shape[1] // 2) ** 2 + (xx - shape[2] // 2) ** 2)
    vol = np.zeros(shape, np.int32)
    vol[(r <= radius) & (zz >= z0) & (zz < z1)] = label
    return vol


def _flat(vol):
    return np.zeros(vol.shape, np.float32)


# --------------------------------------------------------------------------- #
# the sampler
# --------------------------------------------------------------------------- #
class TestTheSampler:
    def test_the_constants(self):
        assert SLICE_MIN_COUNT == 5
        assert SLICE_MAX_COUNT == 60
        assert SLICE_SAMPLINGS == ("share", "count", "every")

    def test_a_quarter_of_44_is_11(self):
        """The spec's worked example: "Traced 11 of 44"."""
        idx, capped = sample_slice_indices(0, 43, sampling="share",
                                           share_pct=25)
        assert len(idx) == 11 and capped is False

    @pytest.mark.parametrize("sampling,kw", [
        ("share", dict(share_pct=25)), ("share", dict(share_pct=5)),
        ("count", dict(n_slices=3)), ("count", dict(n_slices=12)),
        ("every", {}),
    ])
    def test_the_first_and_last_are_always_included(self, sampling, kw):
        idx, _c = sample_slice_indices(7, 90, sampling=sampling, **kw)
        assert idx[0] == 7 and idx[-1] == 90

    def test_the_spread_is_even(self):
        idx, _c = sample_slice_indices(0, 43, sampling="share", share_pct=25)
        gaps = np.diff(idx)
        assert gaps.max() - gaps.min() <= 1

    def test_a_share_is_floored_at_five(self):
        idx, _c = sample_slice_indices(0, 19, sampling="share", share_pct=5)
        assert len(idx) == SLICE_MIN_COUNT

    def test_every_sampling_is_capped_at_sixty_and_says_so(self):
        for sampling, kw in (("every", {}), ("share", dict(share_pct=100)),
                             ("count", dict(n_slices=500))):
            idx, capped = sample_slice_indices(0, 311, sampling=sampling, **kw)
            assert len(idx) == SLICE_MAX_COUNT, sampling
            assert capped is True, sampling

    def test_the_cap_does_not_claim_to_bite_on_a_short_object(self):
        idx, capped = sample_slice_indices(0, 9, sampling="every")
        assert idx == list(range(10)) and capped is False

    def test_a_count_larger_than_the_object_is_every_slice(self):
        idx, _c = sample_slice_indices(3, 8, sampling="count", n_slices=12)
        assert idx == [3, 4, 5, 6, 7, 8]

    def test_by_index_is_the_panel_wire_format(self):
        a = sample_slice_indices(0, 43, sampling=0, share_pct=25)
        b = sample_slice_indices(0, 43, sampling="share", share_pct=25)
        assert a == b

    def test_a_one_slice_object(self):
        idx, _c = sample_slice_indices(5, 5, sampling="share", share_pct=25)
        assert idx == [5]


# --------------------------------------------------------------------------- #
# get_slice -- the public alias
# --------------------------------------------------------------------------- #
class TestGetSlice:
    @pytest.mark.parametrize("orientation", ["XY", "XZ", "YZ"])
    def test_the_alias_is_the_private_helper(self, orientation):
        vol = np.arange(4 * 5 * 6).reshape(4, 5, 6)
        a = interp_mod.get_slice(vol, orientation, 2)
        b = interp_mod._get_slice(vol, orientation, 2)
        assert np.array_equal(a, b)

    def test_it_is_a_view(self):
        vol = np.zeros((4, 5, 6), np.int32)
        interp_mod.get_slice(vol, "XZ", 1)[:] = 7
        assert np.all(vol[:, 1, :] == 7)

    def test_an_unknown_orientation_is_None(self):
        assert interp_mod.get_slice(np.zeros((2, 2, 2)), "3D", 0) is None


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
class TestTheSerialRun:
    def _run(self, vol, **kw):
        base = dict(mode="snap", elevation=_flat(vol), balloon=1.0,
                    iterations=3, limit_um=2.0, orientation="XY")
        base.update(kw)
        return contour_refine(vol, 4, **base)

    def test_it_says_what_was_traced_and_what_was_inferred(self):
        res = self._run(_cylinder())
        assert res.details["traced"] == 11 and res.details["total"] == 44
        assert res.details["traced_text"] == (
            "Traced 11 of 44 slices and filled in the other 33.")

    def test_the_cap_is_said_when_it_bites(self):
        vol = _cylinder(z0=0, z1=70, shape=(70, 24, 24), radius=5)
        res = self._run(vol, sampling="every")
        assert res.details["capped"] is True
        assert res.details["traced_text"] == (
            "Traced 60 of 70 slices (the maximum); the rest were filled in.")

    def test_progress_is_counted_plane_by_plane_then_the_rebuild(self):
        calls = []
        self._run(_cylinder(), progress=lambda s, i, n: calls.append((s, i, n)))
        planes = [c for c in calls if c[0] == "contour"]
        assert planes == [("contour", i, 11) for i in range(1, 12)]
        assert calls[-1][0] == "rebuild"

    def test_a_3D_run_reports_an_uncountable_stage(self):
        calls = []
        self._run(_cylinder(), orientation="3D",
                  progress=lambda s, i, n: calls.append((s, i, n)))
        assert calls == [("contour", 0, 0)]

    def test_stop_between_planes_leaves_the_object_unchanged(self):
        vol = _cylinder()
        ev = threading.Event()

        def _progress(stage, i, n):
            if stage == "contour" and i == 3:
                ev.set()

        res = self._run(vol, progress=_progress, cancel_event=ev)
        assert res.details["reason"] == "stopped"
        assert res[1] == 0
        assert np.array_equal(res[0] == 4, vol == 4)
        # it stopped BETWEEN planes: plane 3 ran, plane 4 never started
        assert res.details["traced"] == 3

    def test_the_rebuild_reuses_interpolate_slices_sdf(self, monkeypatch):
        calls = []
        real = interp_mod.interpolate_slices_sdf

        def spy(a, b, n):
            calls.append(n)
            return real(a, b, n)

        monkeypatch.setattr(interp_mod, "interpolate_slices_sdf", spy)
        res = self._run(_cylinder())
        # 11 traced planes -> 10 gaps, each 3 or 4 planes wide, 33 in all
        assert len(calls) == 10 and sum(calls) == 33
        assert res[1] > 0

    def test_the_rebuilt_shape_is_ONE_object_not_a_stack_of_slices(self):
        from scipy.ndimage import label as nd_label

        res = self._run(_cylinder())
        _lab, n = nd_label(res[0] == 4)
        assert n == 1

    def test_the_rebuild_does_not_grow_a_cap_past_the_ends(self):
        vol = _cylinder()
        res = self._run(vol, balloon=1.0, limit_um=5.0)
        z = np.where((res[0] == 4).any(axis=(1, 2)))[0]
        z0 = np.where((vol == 4).any(axis=(1, 2)))[0]
        assert z.min() >= z0.min() and z.max() <= z0.max()

    @pytest.mark.parametrize("orientation,axis", [("XZ", 1), ("YZ", 2)])
    def test_the_other_two_planes_trace_along_their_own_axis(self, orientation,
                                                             axis):
        vol = _cylinder()
        res = self._run(vol, orientation=orientation)
        extent = int((vol == 4).any(
            axis=tuple(i for i in range(3) if i != axis)).sum())
        assert res.details["total"] == extent
        assert res[1] > 0

    def test_the_limit_holds_on_the_rebuilt_shape(self):
        from scipy.ndimage import distance_transform_edt

        vol = _cylinder()
        res = self._run(vol, balloon=1.0, iterations=20, limit_um=2.0)
        d_out = distance_transform_edt(vol != 4)
        grew = (res[0] == 4) & (vol != 4)
        assert grew.any()
        assert float(d_out[grew].max()) <= 2.0 + 1e-6

    def test_the_fence_holds_on_the_rebuilt_shape(self):
        vol = _cylinder()
        vol[:, :, 27:] = 9
        res = self._run(vol, balloon=1.0, iterations=20, limit_um=10.0)
        assert not np.any((res[0] == 4) & (vol == 9))

    def test_every_plane_is_timed_for_the_cost_table(self):
        res = self._run(_cylinder())
        assert len(res.details["plane_seconds"]) == 11
        assert res.details["rebuild_seconds"] >= 0.0


class TestTheAuditFixes:
    def test_an_emptied_plane_is_not_counted_and_is_kept(self):
        """C-L2: the first slice is ONE voxel; a full pull-in empties it.
        It must not be counted as traced, and the object keeps it."""
        vol = _cylinder()
        vol[2] = 0
        vol[2, 20, 20] = 4                       # a one-voxel first slice
        res = contour_refine(vol, 4, mode="snap", elevation=_flat(vol),
                             balloon=-1.0, iterations=3, limit_um=5.0,
                             orientation="XY")
        assert res.details["dropped"] >= 1
        assert res.details["traced"] == 11 - res.details["dropped"]
        text = res.details["traced_text"]
        d = res.details["dropped"]
        # B3 audit LOW: the three parts add up (traced + kept + filled).
        assert (
            "; 1 more came out empty and was kept as it was; " if d == 1
            else "; {} more came out empty and were kept as they were; "
                 .format(d)) in text, text
        assert res[0][2, 20, 20] == 4

    def test_the_rebuild_blur_in_microns_follows_the_stacking_spacing(self):
        """C-L3: 3 um through depth is ONE voxel at (3, 1, 1)."""
        vol = _cylinder()
        a = contour_refine(vol, 4, mode="snap", elevation=_flat(vol),
                           balloon=1.0, iterations=2, orientation="XY",
                           spacing=(3.0, 1.0, 1.0), rebuild_um=3.0)
        assert a.details["rebuild_sigma_voxels"] == pytest.approx(1.0)
        b = contour_refine(vol, 4, mode="snap", elevation=_flat(vol),
                           balloon=1.0, iterations=2, orientation="XZ",
                           spacing=(3.0, 1.0, 1.0), rebuild_um=3.0)
        assert b.details["rebuild_sigma_voxels"] == pytest.approx(3.0)

    def test_anisotropic_serial_2D_holds_the_limit_in_microns(self):
        """At (3, 1, 1) an XZ plane is 3 um per voxel vertically: a 3 um
        limit lets the outline grow ONE slice through depth, three across."""
        from scipy.ndimage import distance_transform_edt

        vol = _cylinder()
        sp = (3.0, 1.0, 1.0)
        res = contour_refine(vol, 4, mode="snap", elevation=_flat(vol),
                             balloon=1.0, iterations=20, limit_um=3.0,
                             orientation="XZ", sampling="every", spacing=sp)
        grew = (res[0] == 4) & (vol != 4)
        assert grew.any()
        d = distance_transform_edt(vol != 4, sampling=sp)
        assert float(d[grew].max()) <= 3.0 + 1e-6
        z = np.where((res[0] == 4).any(axis=(1, 2)))[0]
        z0 = np.where((vol == 4).any(axis=(1, 2)))[0]
        assert z.min() >= z0.min() - 1 and z.max() <= z0.max() + 1
