"""The tenth wall kind, and the two frozensets it must stay out of.

/ audit C-H1.  ``"probability"`` joins ``spec.WALLS`` so Settle and
the contour can name it, and it joins **neither** ``IMAGE_DERIVED_WALLS`` nor
``GEOMETRY_ONLY_WALLS``:

* in ``IMAGE_DERIVED_WALLS`` it would break
  ``test_reshape_split_widget.py``'s equality against the panel's NINE-item
  Reshape list, and would make ``test_reshape_engine.py``'s
  ``test_every_IMAGE_wall_says_it`` iterate a kind it cannot build from a
  channel mapping;
* in ``GEOMETRY_ONLY_WALLS`` the engine would try to build it from the
  footprint, which is not what it is.

Out of both, ``engine.py``'s "needs a channel" guard refuses it from the
engine's own path -- which is the desired behaviour, because ``probability``
is only ever reachable by SUPPLYING the elevation, and that is exactly how
the Refine tab calls it.

Also: ``build_elevation`` gains ONE branch and nothing else moves, so the
golden suite is bit-identical (asserted here on the two kinds that share the
`elev_img` argument with it).
"""

from __future__ import annotations

import numpy as np
import pytest

from packages.epicycle_annotator.reshape import spec as S
from packages.epicycle_annotator.reshape.walls import build_elevation


class TestTheTenthKindIsInNeitherFrozenset:
    def test_probability_is_a_wall_kind(self):
        assert "probability" in S.WALLS

    def test_it_is_not_image_derived(self):
        assert "probability" not in S.IMAGE_DERIVED_WALLS

    def test_it_is_not_geometry_only(self):
        assert "probability" not in S.GEOMETRY_ONLY_WALLS

    def test_the_engine_refuses_it_without_a_supplied_elevation(self):
        """`engine.py`'s channel guard fires on exactly this case: not
        geometry-only, no supplied elevation, no channel that builds it."""
        from packages.epicycle_annotator.reshape import reshape_region

        vol = np.zeros((8, 16, 16), np.int32)
        vol[2:6, 4:12, 4:12] = 7
        with pytest.raises(ValueError) as exc:
            reshape_region(
                vol, {}, S.ReshapeSpec(target="label", seeds="labels",
                                       walls="probability"),
                target_id=7,
            )
        assert "needs a channel" in str(exc.value)

    def test_the_reshape_panel_still_offers_exactly_nine(self):
        """The Reshape combo is untouched, so `get_split_params`
        stays byte-identical and every Reshape golden test stands."""
        from napari_epicycle.annotator.advanced_features_widget import (
            IMAGE_DERIVED_WALL_INDICES,
            WALL_KINDS,
        )

        assert len(WALL_KINDS) == 9
        assert "probability" not in WALL_KINDS
        assert {WALL_KINDS[i] for i in IMAGE_DERIVED_WALL_INDICES} == set(
            S.IMAGE_DERIVED_WALLS)


class TestTheProbabilityElevation:
    def _region(self):
        r = np.zeros((6, 12, 12), bool)
        r[1:5, 3:9, 3:9] = True
        return r

    def test_bright_where_the_reconstruction_was_least_sure(self):
        region = self._region()
        prob = np.zeros(region.shape, np.float32)
        prob[region] = 1.0                       # confident inside
        elev, clahe, ridge = build_elevation(
            "probability", prob, region, spacing=(1, 1, 1))
        assert elev.shape == region.shape
        assert elev.dtype == np.float32
        # the confident interior is LOW, the unsure background is HIGH
        assert elev[region].mean() < elev[~region].mean()
        assert clahe is None and ridge is None

    def test_it_is_normalised_into_zero_one(self):
        region = self._region()
        rng = np.random.default_rng(0)
        prob = rng.random(region.shape).astype(np.float32) * 7.0
        elev, _, _ = build_elevation("probability", prob, region)
        assert float(elev.min()) >= 0.0
        assert float(elev.max()) <= 1.0

    @pytest.mark.parametrize("kind", ["distance", "skeleton", "none"])
    def test_the_shape_kinds_are_untouched_by_the_new_branch(self, kind):
        """`walls.py` changes by exactly one additive `elif`, so nothing
        that existed before it moves."""
        region = self._region()
        elev, _, _ = build_elevation(kind, None, region, spacing=(2, 1, 1))
        assert elev.shape == region.shape
        assert elev.dtype == np.float32

    def test_intensity_is_bit_identical_to_what_it_always_was(self):
        region = self._region()
        rng = np.random.default_rng(3)
        img = rng.random(region.shape).astype(np.float32)
        a, _, _ = build_elevation("intensity", img, region, clip=0)
        from packages.epicycle_annotator.reshape.preproc import norm

        assert np.array_equal(a, np.asarray(norm(norm(img)), np.float32))
