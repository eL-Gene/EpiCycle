"""Tests for Step 4 feature extraction registry pattern."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_labels(n_labels=3, shape=(16, 24, 24)):
    """Small 3D label volume with n_labels distinct cubic regions."""
    labels = np.zeros(shape, dtype=np.int32)
    cube = min(shape[0] // 3, shape[1] // n_labels - 2, shape[2] // 3)
    cube = max(cube, 4)
    step = shape[1] // n_labels
    for i in range(n_labels):
        y0 = i * step + 1
        labels[2:2 + cube, y0:y0 + cube, 2:2 + cube] = i + 1
    return labels


class TestFeatureGroupDataclass:
    """Verify FeatureGroup structure."""

    def test_has_required_fields(self):
        from napari_epicycle.segmentation.feature_registry import FeatureGroup
        fg = FeatureGroup(
            name="test",
            fn=lambda: None,
            requires=frozenset({"labels"}),
        )
        assert fg.name == "test"
        assert fg.skip_after_smooth is False

    def test_skip_after_smooth_default_false(self):
        from napari_epicycle.segmentation.feature_registry import FeatureGroup
        fg = FeatureGroup(name="x", fn=lambda: None, requires=frozenset())
        assert fg.skip_after_smooth is False

    def test_skip_after_smooth_can_be_true(self):
        from napari_epicycle.segmentation.feature_registry import FeatureGroup
        fg = FeatureGroup(
            name="prob", fn=lambda: None,
            requires=frozenset(), skip_after_smooth=True,
        )
        assert fg.skip_after_smooth is True


class TestDefaultRegistry:
    """Verify the built-in FEATURE_REGISTRY."""

    def test_default_registry_has_expected_groups(self):
        from napari_epicycle.segmentation.feature_registry import FEATURE_REGISTRY
        names = [g.name for g in FEATURE_REGISTRY]
        assert "core_morphology" in names
        assert "surface" in names
        # boundary_interior removed — intensity features are per-channel now
        assert "boundary_interior" not in names
        assert "probability" in names
        assert "neighbor" in names
        assert "skeleton" in names
        assert "border_cells" in names
        assert "composite" in names

    def test_probability_group_is_skip_after_smooth(self):
        from napari_epicycle.segmentation.feature_registry import FEATURE_REGISTRY
        prob = [g for g in FEATURE_REGISTRY if g.name == "probability"][0]
        assert prob.skip_after_smooth is True

    def test_non_probability_groups_are_not_skipped(self):
        from napari_epicycle.segmentation.feature_registry import FEATURE_REGISTRY
        for g in FEATURE_REGISTRY:
            if g.name != "probability":
                assert g.skip_after_smooth is False, f"{g.name} should not skip"


class TestBuildRegistry:
    """Verify dynamic registry construction with channel configs."""

    def test_no_channels_returns_base_registry(self):
        from napari_epicycle.segmentation.feature_registry import (
            FEATURE_REGISTRY, build_registry,
        )
        reg = build_registry()
        assert len(reg) == len(FEATURE_REGISTRY)

    def test_channels_add_entries(self):
        from napari_epicycle.segmentation.feature_registry import (
            FEATURE_REGISTRY, build_registry,
        )
        channels = [{"index": 0, "prefix": "nuclei"}, {"index": 1, "prefix": "gfp"}]
        reg = build_registry(checked_channels=channels)
        assert len(reg) == len(FEATURE_REGISTRY) + 2

    def test_channel_entries_have_correct_names(self):
        from napari_epicycle.segmentation.feature_registry import build_registry
        channels = [{"index": 0, "prefix": "dapi"}]
        reg = build_registry(checked_channels=channels)
        ch_groups = [g for g in reg if g.name.startswith("intensity_")]
        assert len(ch_groups) == 1
        assert ch_groups[0].name == "intensity_dapi"

    def test_channel_entries_not_skipped_after_smooth(self):
        from napari_epicycle.segmentation.feature_registry import build_registry
        channels = [{"index": 0, "prefix": "ch0"}]
        reg = build_registry(checked_channels=channels)
        ch_group = [g for g in reg if g.name == "intensity_ch0"][0]
        assert ch_group.skip_after_smooth is False


class TestRunExtraction:
    """Verify the extraction loop logic."""

    def test_skips_missing_inputs(self):
        from napari_epicycle.segmentation.feature_registry import (
            FeatureGroup, run_extraction,
        )
        called = []
        def fake_fn(**kwargs):
            called.append("called")
            return pd.DataFrame({"label": [1]})

        registry = [
            FeatureGroup(
                name="needs_prob",
                fn=fake_fn,
                requires=frozenset({"labels", "prob_volume"}),
            ),
        ]
        # Only provide labels — prob_volume missing
        result = run_extraction(registry, {"labels": np.zeros((5, 5, 5))})
        assert len(called) == 0  # should have been skipped

    def test_skips_probability_post_smooth(self):
        from napari_epicycle.segmentation.feature_registry import (
            FeatureGroup, run_extraction,
        )
        called = []
        def fake_prob(**kwargs):
            called.append("prob")
            return kwargs.get("df", pd.DataFrame())

        def fake_morph(**kwargs):
            called.append("morph")
            return pd.DataFrame({"label": [1], "area": [100]})

        registry = [
            FeatureGroup(name="morph", fn=fake_morph, requires=frozenset({"labels"})),
            FeatureGroup(
                name="prob", fn=fake_prob,
                requires=frozenset({"labels"}),
                skip_after_smooth=True,
            ),
        ]
        run_extraction(
            registry,
            {"labels": np.zeros((5, 5, 5))},
            post_smooth=True,
        )
        assert "morph" in called
        assert "prob" not in called

    def test_runs_all_when_inputs_available(self):
        from napari_epicycle.segmentation.feature_registry import (
            FeatureGroup, run_extraction,
        )
        called = []
        def track(name):
            def fn(**kwargs):
                called.append(name)
                return kwargs.get("df", pd.DataFrame({"label": [1]}))
            return fn

        registry = [
            FeatureGroup(name="a", fn=track("a"), requires=frozenset({"labels"})),
            FeatureGroup(name="b", fn=track("b"), requires=frozenset({"labels"})),
        ]
        run_extraction(registry, {"labels": np.zeros((5, 5, 5))})
        assert called == ["a", "b"]

    def test_progress_callback_called(self):
        from napari_epicycle.segmentation.feature_registry import (
            FeatureGroup, run_extraction,
        )
        progress_values = []
        def fake_fn(**kwargs):
            return kwargs.get("df", pd.DataFrame({"label": [1]}))

        registry = [
            FeatureGroup(name="a", fn=fake_fn, requires=frozenset({"labels"})),
            FeatureGroup(name="b", fn=fake_fn, requires=frozenset({"labels"})),
        ]
        run_extraction(
            registry,
            {"labels": np.zeros((5, 5, 5))},
            progress_cb=lambda v: progress_values.append(v),
        )
        assert len(progress_values) >= 2
        assert progress_values[-1] == 100

    def test_post_smooth_false_includes_probability(self):
        from napari_epicycle.segmentation.feature_registry import (
            FeatureGroup, run_extraction,
        )
        called = []
        def fake_fn(**kwargs):
            called.append("prob")
            return kwargs.get("df", pd.DataFrame({"label": [1]}))

        registry = [
            FeatureGroup(
                name="prob", fn=fake_fn,
                requires=frozenset({"labels"}),
                skip_after_smooth=True,
            ),
        ]
        run_extraction(registry, {"labels": np.zeros((5, 5, 5))}, post_smooth=False)
        assert "prob" in called
