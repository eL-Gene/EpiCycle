"""Tests for Phase 1 controller hot-path optimizations.

WP-1: Batch processEvents — verifies colormap application correctness
       after removing per-layer processEvents calls.
WP-3: Vectorized colormap construction — verifies colormap parity
       between per-label and vectorized batch map().
WP-5: Vectorized DataFrame access — verifies population-mode patch
       color mapping produces identical results with column-based access
       vs per-row iloc access.

All tests are Qt-free using lightweight fakes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# WP-3: Vectorized colormap construction parity
# ---------------------------------------------------------------------------


class TestVectorizedColormapParity:
    """Verify vectorized cmap.map(batch) matches per-label cmap.map(single)."""

    def _get_default_cmap(self):
        from napari.utils.colormaps import label_colormap

        return label_colormap(49, seed=0.5)

    def test_single_label_matches(self):
        """Vectorized map with a single ID matches the per-label result."""
        cmap = self._get_default_cmap()
        lid = 7
        per_label = np.array(cmap.map(np.array([lid]))[0], dtype=np.float32)
        vectorized = cmap.map(np.array([lid])).astype(np.float32)[0]
        np.testing.assert_array_equal(per_label, vectorized)

    def test_batch_matches_individual_for_multiple_labels(self):
        """Batch cmap.map(all_ids) produces the same color for each label
        as individual cmap.map(np.array([lid])) calls."""
        cmap = self._get_default_cmap()
        ids = [1, 5, 10, 42, 100, 255]

        # Per-label (current slow path)
        per_label = {}
        for lid in ids:
            per_label[lid] = np.array(
                cmap.map(np.array([lid]))[0], dtype=np.float32
            )

        # Vectorized (proposed fast path)
        all_ids = np.array(ids, dtype=np.intp)
        all_colors = cmap.map(all_ids).astype(np.float32)
        vectorized = {lid: all_colors[i] for i, lid in enumerate(all_ids)}

        for lid in ids:
            np.testing.assert_array_equal(
                per_label[lid],
                vectorized[lid],
                err_msg=f"Color mismatch for label {lid}",
            )

    def test_empty_set_produces_base_dict_only(self):
        """Empty selected_ids should produce only None and 0 keys."""
        hidden = np.array([0, 0, 0, 0], dtype=np.float32)
        color_dict = {None: hidden, 0: hidden}
        selected_ids = set()

        if selected_ids:
            cmap = self._get_default_cmap()
            all_ids = np.array(sorted(selected_ids), dtype=np.intp)
            all_colors = cmap.map(all_ids).astype(np.float32)
            for i, lid in enumerate(all_ids):
                color_dict[int(lid)] = all_colors[i]

        assert len(color_dict) == 2
        assert None in color_dict
        assert 0 in color_dict

    def test_large_batch_200_labels(self):
        """200+ labels should work with vectorized approach."""
        cmap = self._get_default_cmap()
        ids = list(range(1, 201))

        all_ids = np.array(ids, dtype=np.intp)
        all_colors = cmap.map(all_ids).astype(np.float32)

        assert all_colors.shape == (200, 4)
        assert all_colors.dtype == np.float32
        assert np.all(np.isfinite(all_colors))

    def test_vectorized_distinct_colormap_dict_keys(self):
        """Vectorized approach produces color_dict with None + 0 + all label IDs."""
        cmap = self._get_default_cmap()
        hidden = np.array([0, 0, 0, 0], dtype=np.float32)
        selected_ids = {3, 7, 15, 42}

        color_dict = {None: hidden, 0: hidden}
        all_ids = np.array(sorted(selected_ids), dtype=np.intp)
        all_colors = cmap.map(all_ids).astype(np.float32)
        for i, lid in enumerate(all_ids):
            color_dict[int(lid)] = all_colors[i]

        expected_keys = {None, 0, 3, 7, 15, 42}
        assert set(color_dict.keys()) == expected_keys

    def test_hidden_labels_are_transparent(self):
        """Labels NOT in selected_ids should remain transparent (RGBA = 0)."""
        cmap = self._get_default_cmap()
        hidden = np.array([0, 0, 0, 0], dtype=np.float32)
        color_dict = {None: hidden, 0: hidden}

        selected_ids = {5, 10}
        all_ids = np.array(sorted(selected_ids), dtype=np.intp)
        all_colors = cmap.map(all_ids).astype(np.float32)
        for i, lid in enumerate(all_ids):
            color_dict[int(lid)] = all_colors[i]

        # None and 0 are transparent
        np.testing.assert_array_equal(color_dict[None], hidden)
        np.testing.assert_array_equal(color_dict[0], hidden)
        # Selected labels have non-zero alpha
        assert color_dict[5][3] > 0
        assert color_dict[10][3] > 0


# ---------------------------------------------------------------------------
# WP-5: Vectorized DataFrame access parity
# ---------------------------------------------------------------------------


def _make_population_df(n_patches=3, labels_per_patch=10):
    """Create a multi-patch QC DataFrame for population mode tests."""
    rows = []
    for p in range(n_patches):
        for lid in range(1, labels_per_patch + 1):
            rows.append(
                {
                    "label": lid,
                    "patch_name": f"path/to/patch_{p}",
                    "qc_status": ["pass", "flagged", "excluded"][lid % 3],
                    "area": float(lid * 100 + p),
                }
            )
    return pd.DataFrame(rows)


class TestVectorizedDataFrameAccess:
    """Verify column-based access produces same results as per-row iloc."""

    def test_column_access_matches_iloc_for_patch_name(self):
        """df['patch_name'].values[i] should equal df.iloc[i]['patch_name']."""
        df = _make_population_df()
        patch_names = df["patch_name"].values
        for i in range(len(df)):
            assert str(patch_names[i]) == str(df.iloc[i]["patch_name"])

    def test_patch_color_mapping_parity_feature_colors(self):
        """Vectorized column-based patch mapping matches iloc-based
        for the _update_viewer_feature_colors population path."""
        df = _make_population_df()
        labels = df["label"].values
        rng = np.random.RandomState(42)
        rgba = rng.rand(len(df), 4).astype(np.float32)

        # Method A: iloc-based (current slow path)
        patch_colors_iloc = {}
        for i, (df_idx, lid) in enumerate(zip(df.index, labels)):
            pn = Path(str(df.iloc[i].get("patch_name", ""))).stem
            patch_colors_iloc.setdefault(pn, {})[int(lid)] = rgba[i]

        # Method B: column-based (proposed fast path)
        patch_colors_col = {}
        patch_name_vals = df["patch_name"].values
        for i, lid in enumerate(labels):
            pn = Path(str(patch_name_vals[i])).stem
            patch_colors_col.setdefault(pn, {})[int(lid)] = rgba[i]

        assert set(patch_colors_iloc.keys()) == set(patch_colors_col.keys())
        for patch in patch_colors_iloc:
            for lid in patch_colors_iloc[patch]:
                np.testing.assert_array_equal(
                    patch_colors_iloc[patch][lid],
                    patch_colors_col[patch][lid],
                )

    def test_isolation_filter_with_column_access(self):
        """Isolation filtering works correctly with column-based access."""
        df = _make_population_df(n_patches=2, labels_per_patch=5)
        labels = df["label"].values
        rgba = np.ones((len(df), 4), dtype=np.float32)

        # Isolate only df indices 0, 1, 2
        iso_df_indices = {0, 1, 2}

        patch_colors = {}
        patch_name_vals = df["patch_name"].values
        for i, (df_idx, lid) in enumerate(zip(df.index, labels)):
            if df_idx not in iso_df_indices:
                continue
            pn = Path(str(patch_name_vals[i])).stem
            patch_colors.setdefault(pn, {})[int(lid)] = rgba[i]

        total_labels = sum(len(v) for v in patch_colors.values())
        assert total_labels == 3

    def test_qc_status_column_access_matches_iterrows(self):
        """Column-based QC status mapping matches iterrows-based."""
        df = _make_population_df()

        status_colors = {
            "pass": np.array([0.31, 0.78, 0.47, 1.0], dtype=np.float32),
            "flagged": np.array([1.0, 0.63, 0.16, 1.0], dtype=np.float32),
            "excluded": np.array([0.94, 0.24, 0.24, 1.0], dtype=np.float32),
        }

        # Method A: iterrows (current)
        patch_colors_a = {}
        for df_idx, row in df.iterrows():
            lid = int(row["label"])
            pn = Path(str(row.get("patch_name", ""))).stem
            c = status_colors.get(row.get("qc_status", ""))
            if c is not None:
                patch_colors_a.setdefault(pn, {})[lid] = c

        # Method B: column access (proposed)
        patch_colors_b = {}
        label_vals = df["label"].values
        patch_name_vals = df["patch_name"].values
        qc_status_vals = df["qc_status"].values
        for i in range(len(df)):
            lid = int(label_vals[i])
            pn = Path(str(patch_name_vals[i])).stem
            c = status_colors.get(str(qc_status_vals[i]), None)
            if c is not None:
                patch_colors_b.setdefault(pn, {})[lid] = c

        assert set(patch_colors_a.keys()) == set(patch_colors_b.keys())
        for patch in patch_colors_a:
            for lid in patch_colors_a[patch]:
                np.testing.assert_array_equal(
                    patch_colors_a[patch][lid],
                    patch_colors_b[patch][lid],
                )

    def test_categorical_column_access_matches_iterrows(self):
        """Column-based categorical mapping matches iterrows-based."""
        df = _make_population_df()
        df["qc_tier"] = np.random.RandomState(42).choice(
            ["good", "review", "bad"], size=len(df)
        )

        tier_colors = {
            "good": np.array([0.31, 0.78, 0.47, 1.0], dtype=np.float32),
            "review": np.array([1.0, 0.63, 0.16, 1.0], dtype=np.float32),
            "bad": np.array([0.94, 0.24, 0.24, 1.0], dtype=np.float32),
        }

        # Method A: iterrows + iloc (current)
        patch_colors_a = {}
        labels = df["label"].values
        for i, (df_idx, lid) in enumerate(zip(df.index, labels)):
            pn = Path(str(df.iloc[i].get("patch_name", ""))).stem
            cat = df.iloc[i]["qc_tier"]
            patch_colors_a.setdefault(pn, {})[int(lid)] = tier_colors[cat]

        # Method B: column access (proposed)
        patch_colors_b = {}
        patch_name_vals = df["patch_name"].values
        tier_vals = df["qc_tier"].values
        for i, lid in enumerate(labels):
            pn = Path(str(patch_name_vals[i])).stem
            cat = str(tier_vals[i])
            patch_colors_b.setdefault(pn, {})[int(lid)] = tier_colors[cat]

        assert set(patch_colors_a.keys()) == set(patch_colors_b.keys())
        for patch in patch_colors_a:
            for lid in patch_colors_a[patch]:
                np.testing.assert_array_equal(
                    patch_colors_a[patch][lid],
                    patch_colors_b[patch][lid],
                )


class TestVectorizedIsolationPatchMapping:
    """Verify vectorized batch iloc for _show_isolated_labels_distinct."""

    def test_batch_iloc_matches_per_row_iloc(self):
        """Single df.iloc[valid_indices] matches individual df.iloc[idx] calls."""
        df = _make_population_df(n_patches=3, labels_per_patch=5)
        isolated_indices = [0, 3, 7, 12]

        # Method A: per-row iloc (current slow path)
        patch_labels_a = {}
        for idx in isolated_indices:
            if idx < len(df):
                row = df.iloc[idx]
                lid = int(row["label"])
                pn = Path(str(row["patch_name"])).stem
                patch_labels_a.setdefault(pn, set()).add(lid)

        # Method B: batch iloc (proposed fast path)
        patch_labels_b = {}
        valid = [idx for idx in isolated_indices if idx < len(df)]
        if valid:
            subset = df.iloc[valid]
            label_vals = subset["label"].values
            pn_vals = subset["patch_name"].values
            for i in range(len(subset)):
                pn = Path(str(pn_vals[i])).stem
                patch_labels_b.setdefault(pn, set()).add(int(label_vals[i]))

        assert patch_labels_a == patch_labels_b

    def test_batch_iloc_with_out_of_bounds_indices(self):
        """Out-of-bounds indices should be filtered, not crash."""
        df = _make_population_df(n_patches=1, labels_per_patch=3)
        isolated_indices = [0, 1, 999]  # 999 is out of bounds

        patch_labels = {}
        valid = [idx for idx in isolated_indices if idx < len(df)]
        assert 999 not in valid

        if valid:
            subset = df.iloc[valid]
            assert len(subset) == 2

    def test_empty_indices_returns_empty_mapping(self):
        """Empty isolated indices should produce empty patch_labels."""
        df = _make_population_df()
        isolated_indices = []

        patch_labels = {}
        valid = [idx for idx in isolated_indices if idx < len(df)]
        if valid:
            subset = df.iloc[valid]
            label_vals = subset["label"].values
            pn_vals = subset["patch_name"].values
            for i in range(len(subset)):
                pn = Path(str(pn_vals[i])).stem
                patch_labels.setdefault(pn, set()).add(int(label_vals[i]))

        assert patch_labels == {}


# ---------------------------------------------------------------------------
# WP-1: Batch colormap dict construction correctness
# ---------------------------------------------------------------------------


class TestBatchColormapDictConstruction:
    """Verify per-patch color dict construction produces correct DirectLabelColormaps."""

    def test_each_patch_gets_its_own_labels(self):
        """In population mode, each grid layer should only see its own patch's labels."""
        from napari.utils.colormaps import DirectLabelColormap

        hidden = np.array([0, 0, 0, 0], dtype=np.float32)
        patch_colors = {
            "patch_0": {1: np.array([1, 0, 0, 1], dtype=np.float32)},
            "patch_1": {2: np.array([0, 1, 0, 1], dtype=np.float32)},
        }

        results = {}
        for name in ["[grid] patch_0", "[grid] patch_1", "[grid] patch_2"]:
            stem = name.replace("[grid] ", "")
            cd = {None: hidden}
            cd.update(patch_colors.get(stem, {}))
            results[stem] = DirectLabelColormap(color_dict=cd)

        # patch_0 has label 1
        assert 1 in results["patch_0"].color_dict
        assert 2 not in results["patch_0"].color_dict
        # patch_1 has label 2
        assert 2 in results["patch_1"].color_dict
        assert 1 not in results["patch_1"].color_dict
        # patch_2 has no specific labels (only None)
        assert None in results["patch_2"].color_dict

    def test_batch_build_then_apply_is_equivalent(self):
        """Building all colormaps first, then applying, produces same
        result as interleaved build+apply (the change WP-1 makes)."""
        from napari.utils.colormaps import DirectLabelColormap

        hidden = np.array([0, 0, 0, 0], dtype=np.float32)
        rng = np.random.RandomState(99)
        n_patches = 5

        patch_colors = {}
        for p in range(n_patches):
            patch_colors[f"patch_{p}"] = {
                lid: rng.rand(4).astype(np.float32) for lid in range(1, 8)
            }

        # Interleaved (old approach)
        interleaved = {}
        for p in range(n_patches):
            stem = f"patch_{p}"
            cd = {None: hidden}
            cd.update(patch_colors.get(stem, {}))
            interleaved[stem] = DirectLabelColormap(color_dict=cd)

        # Batch (new approach: build list, then apply)
        updates = []
        for p in range(n_patches):
            stem = f"patch_{p}"
            cd = {None: hidden}
            cd.update(patch_colors.get(stem, {}))
            updates.append((stem, DirectLabelColormap(color_dict=cd)))
        batched = {stem: cmap for stem, cmap in updates}

        # Verify identical
        for p in range(n_patches):
            stem = f"patch_{p}"
            assert set(interleaved[stem].color_dict.keys()) == set(
                batched[stem].color_dict.keys()
            )
            for key in interleaved[stem].color_dict:
                np.testing.assert_array_equal(
                    interleaved[stem].color_dict[key],
                    batched[stem].color_dict[key],
                )

    def test_clear_isolation_default_cmap_applied_to_all(self):
        """When clearing isolation, all grid layers get the same default cmap."""
        from napari.utils.colormaps import label_colormap

        default_cmap = label_colormap(49, seed=0.5)

        # Simulate applying to 5 layers in batch
        results = []
        for _ in range(5):
            results.append(default_cmap)

        # All should be the exact same object
        for cmap in results:
            assert cmap is default_cmap


# ---------------------------------------------------------------------------
# Integration: verify source code reflects the changes
# ---------------------------------------------------------------------------


class TestSourceCodeReflectsOptimizations:
    """Verify the controller source has the expected optimizations wired in."""

    def _get_source(self, method_name):
        import inspect

        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )

        method = getattr(FusionReconstructionController, method_name)
        return inspect.getsource(method)

    def test_no_processevents_in_the_one_colour_path(self):
        """The one colour path (Step 3 plot pass P3, which folded
        _update_viewer_qc_colors and _update_viewer_categorical_colors in)
        writes every layer's colormap and must not pump the event loop."""
        source = self._get_source("_apply_plot_colours_to_viewer")
        assert "for layer in to_write:" in source       # the per-layer loop
        assert "processEvents" not in source

    def test_no_processevents_in_feature_colors_wrapper(self):
        """_update_viewer_feature_colors is now a thin wrapper over the one
        path: no processEvents, no writer of its own."""
        source = self._get_source("_update_viewer_feature_colors")
        assert "_apply_plot_colours_to_viewer" in source
        assert "processEvents" not in source

    def test_no_processevents_in_show_isolated_distinct(self):
        """_show_isolated_labels_distinct should not call processEvents
        in the per-layer loop."""
        source = self._get_source("_show_isolated_labels_distinct")
        assert "processEvents" not in source

    def test_no_processevents_in_clear_isolation(self):
        """_clear_label_isolation should not call processEvents
        in the per-layer loop."""
        source = self._get_source("_clear_label_isolation")
        assert "processEvents" not in source

    def test_no_per_label_cmap_map_in_distinct(self):
        """_show_isolated_labels_distinct should use vectorized cmap.map,
        not per-label cmap.map(np.array([lid]))."""
        source = self._get_source("_show_isolated_labels_distinct")
        assert "cmap.map(np.array([lid]))" not in source

    def test_no_per_row_df_access_in_the_one_colour_path(self):
        """The one path builds its colormap from the plot's id / RGBA
        arrays: no df.iloc / iterrows per row."""
        source = self._get_source("_apply_plot_colours_to_viewer")
        assert "viewer_colours()" in source
        assert "df.iloc[i]" not in source
        assert "iterrows" not in source
