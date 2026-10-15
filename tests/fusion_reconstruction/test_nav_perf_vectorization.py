"""Tests for the [NAV-PERF] vectorization refactors (fix/step3-extract-perf).

These guard the behavior-preserving vectorization of:
  * ``FusionReconstructionController._build_flagged_label_items``
    (controller.py — was a per-row ``df.iterrows()`` loop)
  * the flagged-list build inside ``_update_flagged_labels_list``
    (controller.py — same per-row pattern)
  * ``curation_label_navigator._batch_swatch_colors`` (navigator-side
    batched colormap map that replaces per-row ``swatch_color_for_label``)

The refactors MUST be byte-for-byte output-preserving — only faster.  We
assert known values on small fixtures so a regression that drifts the
output (wrong order, wrong score handling, wrong centroid scaling, wrong
color) fails here.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Minimal controller harness binding the real vectorized methods
# ---------------------------------------------------------------------------

from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController as _Ctrl,
)


class _Harness:
    """Just enough state for the vectorized builder methods."""

    def __init__(self, df=None, factor=1):
        self._qc_features_df = df
        self._flagged_centroids = {}
        self._qc_view_patch_name = "patch_a"
        self._fixed_factor = factor

    # _build_flagged_label_items + _update_flagged_labels_list call this.
    def _current_downsample_factor(self, patch_name: str = "") -> int:
        return self._fixed_factor


# Bind the real (vectorized) method as an unbound function.
_Harness._build_flagged_label_items = _Ctrl._build_flagged_label_items


def _make_df(rows):
    """rows: list of dicts with at least label / centroid / area / tier."""
    return pd.DataFrame(rows)


def _full_row(lid, *, area, c0, c1, c2, tier="review", score=None,
              feature_state="fresh", status="active"):
    return {
        "label": lid,
        "area": area,
        "centroid-0": c0,
        "centroid-1": c1,
        "centroid-2": c2,
        "qc_tier": tier,
        "qc_score": score if score is not None else float("nan"),
        "feature_state": feature_state,
        "label_status": status,
    }


# ---------------------------------------------------------------------------
# _build_flagged_label_items — vectorized output equals known values
# ---------------------------------------------------------------------------

class TestBuildFlaggedLabelItemsValues:

    def test_known_values_factor_1(self):
        h = _Harness(factor=1)
        h._qc_features_df = _make_df([
            _full_row(5, area=50, c0=10.7, c1=20.2, c2=30.9,
                      tier="bad", score=2.5, feature_state="fresh"),
            _full_row(8, area=80, c0=4.0, c1=6.0, c2=8.0,
                      tier="good", score=None, feature_state="stale"),
        ])
        items = h._build_flagged_label_items()
        assert len(items) == 2

        # Order preserved (DataFrame order).
        assert items[0]["label_id"] == 5
        assert items[1]["label_id"] == 8

        # Exact key set.
        assert set(items[0].keys()) == {
            "label_id", "tier", "centroid", "area", "qc_score",
            "feature_state",
        }

        # Centroid: int(float(.)) truncates toward zero (10.7 -> 10).
        assert items[0]["centroid"] == (10, 20, 30)
        assert items[0]["area"] == 50
        assert items[0]["tier"] == "bad"
        assert items[0]["qc_score"] == 2.5
        assert items[0]["feature_state"] == "fresh"

        # NaN score -> None (em-dash placeholder downstream).
        assert items[1]["qc_score"] is None
        assert items[1]["feature_state"] == "stale"
        assert items[1]["centroid"] == (4, 6, 8)

    def test_centroid_scaled_by_factor(self):
        h = _Harness(factor=2)
        h._qc_features_df = _make_df([
            _full_row(1, area=10, c0=10.0, c1=20.0, c2=30.0, tier="review"),
        ])
        items = h._build_flagged_label_items()
        # int(float(10.0)) // 2 == 5, etc.
        assert items[0]["centroid"] == (5, 10, 15)

    def test_excludes_ghost_rows(self):
        h = _Harness(factor=1)
        h._qc_features_df = _make_df([
            _full_row(1, area=10, c0=1, c1=1, c2=1, status="active"),
            _full_row(2, area=20, c0=2, c1=2, c2=2, status="deleted"),
        ])
        items = h._build_flagged_label_items()
        assert [it["label_id"] for it in items] == [1]

    def test_none_df_returns_empty(self):
        h = _Harness(factor=1)
        h._qc_features_df = None
        assert h._build_flagged_label_items() == []

    def test_missing_qc_tier_column_returns_empty(self):
        h = _Harness(factor=1)
        h._qc_features_df = _make_df([
            {"label": 1, "area": 10, "label_status": "active"},
        ])
        assert h._build_flagged_label_items() == []

    def test_missing_centroid_columns_default_zero(self):
        h = _Harness(factor=1)
        h._qc_features_df = _make_df([
            {"label": 7, "area": 5, "qc_tier": "review",
             "label_status": "active"},
        ])
        items = h._build_flagged_label_items()
        assert items[0]["centroid"] == (0, 0, 0)
        # Missing feature_state column -> "" (FIX 2 default).
        assert items[0]["feature_state"] == ""
        # Missing qc_score column -> None.
        assert items[0]["qc_score"] is None


# ---------------------------------------------------------------------------
# Reference re-implementation of the OLD per-row loop to prove equivalence
# ---------------------------------------------------------------------------

def _reference_build_items(df, factor):
    """The pre-refactor per-row algorithm, kept here as an oracle."""
    from napari_epicycle.fusion_reconstruction.qc_features import active_rows
    if df is None:
        return []
    d = active_rows(df)
    if "qc_tier" not in d.columns:
        return []
    out = []
    for _, row in d.iterrows():
        label_id = int(row["label"])
        tier = str(row.get("qc_tier", "review"))
        z = int(float(row.get("centroid-0", 0)))
        y = int(float(row.get("centroid-1", 0)))
        x = int(float(row.get("centroid-2", 0)))
        if factor > 1:
            z //= factor
            y //= factor
            x //= factor
        area = int(row.get("area", 0))
        score = None
        if "qc_score" in d.columns:
            raw = row.get("qc_score")
            try:
                sf = float(raw)
                if not math.isnan(sf):
                    score = sf
            except (TypeError, ValueError):
                score = None
        feature_state = str(row.get("feature_state", ""))
        out.append({
            "label_id": label_id, "tier": tier, "centroid": (z, y, x),
            "area": area, "qc_score": score, "feature_state": feature_state,
        })
    return out


class TestBuildFlaggedLabelItemsMatchesReference:

    @pytest.mark.parametrize("factor", [1, 2, 3])
    def test_matches_reference_on_mixed_df(self, factor):
        rows = [
            _full_row(11, area=111, c0=10.9, c1=0.0, c2=99.4,
                      tier="garbage", score=1.25, feature_state="fresh"),
            _full_row(2, area=2, c0=3.3, c1=3.3, c2=3.3,
                      tier="", score=None, feature_state="stale"),
            _full_row(40, area=400, c0=80.0, c1=60.0, c2=40.0,
                      tier="curated", score=0.0, feature_state="fresh"),
            _full_row(9, area=9, c0=1, c1=1, c2=1,
                      tier="review", score=float("nan"), status="deleted"),
        ]
        df = _make_df(rows)
        h = _Harness(factor=factor)
        h._qc_features_df = df
        got = h._build_flagged_label_items()
        expected = _reference_build_items(df, factor)
        assert got == expected


# ---------------------------------------------------------------------------
# _batch_swatch_colors — byte-for-byte equal to per-row swatch_color_for_label
# ---------------------------------------------------------------------------

from napari_epicycle.annotator.curation_label_navigator import (
    _batch_swatch_colors,
)
from napari_epicycle.ui_shell.theme import swatch_color_for_label


class _FakeColormap:
    """Maps each id to a deterministic RGBA; id 0 has alpha 0 (hidden)."""

    def __init__(self, hide_ids=()):
        self._hide = set(hide_ids)

    def map(self, ids):
        ids = np.asarray(ids).reshape(-1)
        out = np.zeros((ids.shape[0], 4), dtype=float)
        for i, lid in enumerate(ids):
            lid = int(lid)
            if lid in self._hide:
                out[i] = (0.5, 0.5, 0.5, 0.0)  # alpha 0 -> fallback path
            else:
                # Deterministic, distinct per id, in [0, 1].
                out[i] = (
                    (lid % 7) / 7.0,
                    (lid % 5) / 5.0,
                    (lid % 3) / 3.0,
                    1.0,
                )
        return out


class _FakeLayer:
    def __init__(self, cmap):
        self.colormap = cmap


class TestBatchSwatchColors:

    def test_matches_per_id_helper(self):
        layer = _FakeLayer(_FakeColormap())
        ids = [1, 2, 3, 17, 42, 100]
        batched = _batch_swatch_colors(layer, ids)
        for lid in ids:
            ref = swatch_color_for_label(layer, lid)
            got = batched[lid]
            assert (got.red(), got.green(), got.blue()) == (
                ref.red(), ref.green(), ref.blue()
            ), f"color drift for id={lid}"

    def test_none_layer_all_fallback(self):
        ids = [1, 2, 3]
        batched = _batch_swatch_colors(None, ids)
        for lid in ids:
            c = batched[lid]
            assert (c.red(), c.green(), c.blue()) == (204, 204, 204)

    def test_empty_ids(self):
        layer = _FakeLayer(_FakeColormap())
        assert _batch_swatch_colors(layer, []) == {}

    def test_alpha_zero_delegates_to_per_id(self):
        # id 5 is hidden (alpha 0) -> must match per-id helper exactly,
        # which uses the napari default-label-colormap fallback.
        layer = _FakeLayer(_FakeColormap(hide_ids=(5,)))
        ids = [4, 5, 6]
        batched = _batch_swatch_colors(layer, ids)
        for lid in ids:
            ref = swatch_color_for_label(layer, lid)
            got = batched[lid]
            assert (got.red(), got.green(), got.blue()) == (
                ref.red(), ref.green(), ref.blue()
            ), f"color drift for hidden/visible id={lid}"

    def test_map_raises_falls_back_per_id(self):
        class _RaisingCmap:
            def map(self, ids):
                raise RuntimeError("boom")
        layer = _FakeLayer(_RaisingCmap())
        ids = [1, 2]
        batched = _batch_swatch_colors(layer, ids)
        # Per-id helper also hits the raise -> neutral fallback for each.
        for lid in ids:
            c = batched[lid]
            assert (c.red(), c.green(), c.blue()) == (204, 204, 204)

    def test_many_hidden_ids_batched_match_per_id(self):
        # The fix batches the hidden-label path through the cached default
        # colormap instead of ~N per-id calls.  Verify a LARGE hidden set
        # (the isolation case — most rows hidden) is still byte-identical to
        # the per-id helper for every id.
        hide = tuple(range(1, 40))
        layer = _FakeLayer(_FakeColormap(hide_ids=hide))
        ids = list(range(1, 45))  # 39 hidden + 5 visible
        batched = _batch_swatch_colors(layer, ids)
        for lid in ids:
            ref = swatch_color_for_label(layer, lid)
            got = batched[lid]
            assert (got.red(), got.green(), got.blue()) == (
                ref.red(), ref.green(), ref.blue()
            ), f"color drift for id={lid} (hidden={lid in hide})"


class TestDefaultLabelColormapCache:
    def test_default_label_colormap_is_cached(self):
        # The fix caches label_colormap(49, seed=0.5) module-level (it was
        # rebuilt per hidden-label call → ~2.1s during isolation).  Repeated
        # calls must return the SAME object (cache hit), and the args are
        # constant so the cache can never go stale.
        from napari_epicycle.ui_shell.theme import default_label_colormap
        a = default_label_colormap()
        b = default_label_colormap()
        assert a is b, "default_label_colormap must return the cached instance"
