"""``refine_recipe.batch_blocker`` -- the batch's pure pre-flight.

``step_blocker`` passes through with the batch's own sentences; then the
diffusion solver, the 3D contour cap (Run-only vs both), the probability
map's and the channel's size, and the memory budget.  Napari-free.
"""

from __future__ import annotations

import pytest

from packages.epicycle_annotator import batch_chain as bc
from packages.epicycle_annotator import refine_recipe as rr
from tests.annotator.test_batch_chain_dispatch import (
    contour,
    guided,
    sdf,
    settle,
)

SHAPE = (60, 200, 200)
BOX = (slice(20, 30), slice(50, 70), slice(50, 70))


def rec(*steps, spacing_known=False):
    return rr.Recipe(steps=tuple(steps), spacing_known=spacing_known)


def block(recipe, **kw):
    base = dict(n_objects=1, channel="membrane",
                availability={"spacing_known": False, "prob": True},
                bboxes=[BOX], volume_shape=SHAPE, spacing=(1.0, 1.0, 1.0),
                prob_shape=SHAPE, labels_shape=SHAPE, workers=4)
    base.update(kw)
    return rr.batch_blocker(recipe, **base)


class TestPassThrough:
    def test_a_runnable_recipe_is_not_refused(self):
        assert block(rec(sdf(), settle())) is None

    def test_channel_missing(self):
        b = block(rec(sdf(), settle()), channel="")
        assert b[:2] == ("channel", 2)
        assert b.text == rr.BATCH_NEEDS_CHANNEL.format(k=2, word="Settle")

    def test_no_probability_map(self):
        b = block(rec(guided()), availability={"spacing_known": False,
                                               "prob": False})
        assert b[:2] == ("prob", 1)
        assert b.text == rr.BATCH_NEEDS_PROB.format(k=1, word="Guided Grow")

    def test_spacing_both_ways(self):
        b = block(rec(sdf(), spacing_known=True))
        assert b[:2] == ("spacing_known", 0)
        assert b.text == rr.BATCH_SPACING_KNOWN
        b = block(rec(sdf()), availability={"spacing_known": True})
        assert b[:2] == ("spacing_unknown", 0)
        assert b.text == rr.BATCH_SPACING_UNKNOWN
        assert "on the Refine tab" in rr.BATCH_SPACING_KNOWN
        assert "on the Refine tab" in rr.BATCH_SPACING_UNKNOWN


class TestBatchOnly:
    def test_label_diffusion_is_refused(self):
        b = block(rec(sdf(), settle(solver="label_diffusion")))
        assert b[:2] == ("batch_diffusion", 2)
        assert b.text == rr.BATCH_NO_DIFFUSION.format(k=2)

    def test_3d_contour_cap(self):
        r = rec(sdf(), contour(axis="3D"))
        assert rr.BATCH_CONTOUR_3D_MAX == 20
        assert block(r, n_objects=20, bboxes=[BOX] * 20) is None
        b = block(r, n_objects=21, bboxes=[BOX] * 21)
        assert b[:2] == ("batch_contour_3d", 2)
        lo, hi = rr.CONTOUR_3D_SECONDS
        assert (lo, hi) == (15, 25)
        assert b.text == rr.BATCH_CONTOUR_3D.format(k=2, lo=lo, hi=hi,
                                                    cap=20)
        assert "15–25 s" in b.text
        # Preview is n = 1: never refused by the cap (Run-only)
        assert block(r, n_objects=1) is None

    def test_xy_contour_is_uncapped(self):
        assert block(rec(contour(axis="XY")), n_objects=500,
                     bboxes=[BOX] * 500) is None

    def test_prob_shape(self):
        b = block(rec(guided()), prob_shape=(30, 100, 100))
        assert b[:2] == ("batch_prob_shape", 1)
        assert b.text == rr.BATCH_PROB_SHAPE
        # a recipe that does not read it does not care
        assert block(rec(sdf()), prob_shape=(30, 100, 100)) is None

    def test_channel_shape(self):
        b = block(rec(settle()), channel_shape=(30, 100, 100))
        assert b[:2] == ("batch_channel_shape", 1)
        assert b.text == rr.BATCH_CHANNEL_SHAPE

    def test_too_large(self):
        # a 30 um contour limit at 0.1 um spacing: 302 voxels of margin on
        # every axis -- Z clamps to the whole volume
        big = (300, 500, 500)
        r = rec(contour(axis="XY", limit=30.0))
        b = block(r, volume_shape=big, labels_shape=big, prob_shape=big,
                  spacing=(0.1, 0.1, 0.1), n_objects=8, bboxes=[BOX] * 8)
        assert b[:2] == ("batch_too_large", 0)
        vox = 300 * (70 + 302) * (70 + 302)
        need = bc.estimate_chain_bytes(vox) * 4
        assert need > bc.BATCH_CHAIN_RAM_BUDGET
        assert b.text == rr.BATCH_TOO_LARGE.format(
            gb=rr._gb(need), w=4, budget=rr._gb(bc.BATCH_CHAIN_RAM_BUDGET))
        # one object at a time halves nothing here, but the same recipe on a
        # small object at normal spacing runs
        assert block(r, n_objects=8, bboxes=[BOX] * 8) is None

    def test_the_budget_numbers(self):
        assert bc.BATCH_CHAIN_RAM_BUDGET == 8 * 1024 ** 3
        assert bc.CHAIN_BYTES_PER_VOXEL == 64
        # a 73 Mvox patch whole: ~4.4 GiB per object at 64 B / voxel
        assert bc.estimate_chain_bytes(317 * 480 * 480) == 317 * 480 * 480 * 64

    def test_preview_uses_only_its_own_box(self):
        huge = (slice(0, 60), slice(0, 200), slice(0, 200))
        r = rec(contour(axis="XY", limit=30.0))
        kw = dict(spacing=(0.1, 0.1, 0.1))
        # Run over a huge target list refuses; Preview on a small box alone
        # is evaluated on that box
        run_b = block(r, n_objects=4, bboxes=[huge] * 4,
                      volume_shape=(300, 800, 800),
                      labels_shape=(300, 800, 800),
                      prob_shape=(300, 800, 800), **kw)
        prev_b = block(r, n_objects=1, bboxes=[],
                       volume_shape=(300, 800, 800),
                       labels_shape=(300, 800, 800),
                       prob_shape=(300, 800, 800), **kw)
        assert run_b is not None and run_b.key == "batch_too_large"
        assert prev_b is None


@pytest.mark.parametrize("key", ["batch_diffusion", "batch_contour_3d",
                                 "batch_prob_shape", "batch_too_large"])
def test_every_sentence_is_a_named_constant(key):
    name = {"batch_diffusion": "BATCH_NO_DIFFUSION",
            "batch_contour_3d": "BATCH_CONTOUR_3D",
            "batch_prob_shape": "BATCH_PROB_SHAPE",
            "batch_too_large": "BATCH_TOO_LARGE"}[key]
    assert isinstance(getattr(rr, name), str)
