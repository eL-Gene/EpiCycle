"""Tests for Phase 3: 3D curation render optimization (WP-6).

Verifies:
- Vectorized colormap construction in _apply_multi_isolation_colormap
- Sync suppression during annotator isolation colormap operations
- No per-label cmap.map calls in the annotator isolation path

All tests are Qt-free using lightweight fakes.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# WP-6a: Vectorized colormap construction in annotator
# ---------------------------------------------------------------------------


class TestAnnotatorVectorizedColormap:
    """_apply_multi_isolation_colormap should use vectorized cmap.map."""

    def test_source_code_no_per_label_map(self):
        """_apply_multi_isolation_colormap should not use per-label
        cmap.map(np.array([lid])) calls."""
        import inspect

        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        source = inspect.getsource(
            Curation3DAnnotator._apply_multi_isolation_colormap
        )
        assert "cmap.map(np.array([lid]))" not in source
        assert "default_cmap.map(np.array([lid]))" not in source

    def test_vectorized_parity_in_annotator_context(self):
        """Vectorized approach produces same color dict as per-label approach
        for the annotator's isolation colormap build."""
        from napari.utils.colormaps import label_colormap

        default_cmap = label_colormap(49, seed=0.5)
        hidden = np.array([0, 0, 0, 0], dtype=np.float32)
        label_ids = {3, 7, 15, 42, 100}

        # Per-label (old approach)
        cd_old = {None: hidden, 0: hidden}
        for lid in label_ids:
            cd_old[int(lid)] = np.array(
                default_cmap.map(np.array([lid]))[0], dtype=np.float32
            )

        # Vectorized (new approach)
        cd_new = {None: hidden, 0: hidden}
        all_ids = np.array(sorted(label_ids), dtype=np.intp)
        all_colors = default_cmap.map(all_ids).astype(np.float32)
        for i, lid in enumerate(all_ids):
            cd_new[int(lid)] = all_colors[i]

        assert set(cd_old.keys()) == set(cd_new.keys())
        for key in cd_old:
            np.testing.assert_array_equal(cd_old[key], cd_new[key])


# ---------------------------------------------------------------------------
# WP-6b: Sync suppression during annotator isolation
# ---------------------------------------------------------------------------


class TestAnnotatorSyncSuppression:
    """Annotator should suppress sync cascade during isolation colormap ops."""

    def test_apply_multi_isolation_uses_sync_suppression(self):
        """_apply_multi_isolation_colormap should suppress sync to prevent
        double subviewer updates (sync cascade + explicit propagation)."""
        import inspect

        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        source = inspect.getsource(
            Curation3DAnnotator._apply_multi_isolation_colormap
        )
        assert "suppress_property_sync" in source or "_suppress" in source.lower()

    def test_isolation_restore_uses_sync_suppression(self):
        """The restore path in _on_isolation_changed should suppress sync
        when restoring the colormap + propagating to subviewers."""
        import inspect

        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        source = inspect.getsource(
            Curation3DAnnotator._on_isolation_changed
        )
        # Should have sync suppression somewhere in the method
        assert "suppress_property_sync" in source or "_suppress" in source.lower()


# ---------------------------------------------------------------------------
# WP-6c: Frozenset cache still works
# ---------------------------------------------------------------------------


class TestIsolationCacheIntact:
    """The frozenset cache optimization should still be present."""

    def test_frozenset_cache_in_source(self):
        """_apply_multi_isolation_colormap should still have frozenset cache."""
        import inspect

        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        source = inspect.getsource(
            Curation3DAnnotator._apply_multi_isolation_colormap
        )
        assert "frozenset" in source
        assert "_isolation_label_set" in source
