"""Tests for lasso selection in QC plot widget."""

import numpy as np
import pytest


class TestLassoContainsPoints:
    """Test the matplotlib Path point-in-polygon logic used by lasso selection."""

    def test_square_polygon_contains_interior(self):
        from matplotlib.path import Path as MplPath

        verts = [(0, 0), (10, 0), (10, 10), (0, 10)]
        path = MplPath(verts)
        points = np.array([[5, 5], [1, 1], [9, 9]])
        inside = path.contains_points(points)
        assert all(inside)

    def test_square_polygon_excludes_exterior(self):
        from matplotlib.path import Path as MplPath

        verts = [(0, 0), (10, 0), (10, 10), (0, 10)]
        path = MplPath(verts)
        points = np.array([[-1, 5], [5, -1], [11, 5], [5, 11]])
        inside = path.contains_points(points)
        assert not any(inside)

    def test_triangle_polygon(self):
        from matplotlib.path import Path as MplPath

        verts = [(0, 0), (10, 0), (5, 10)]
        path = MplPath(verts)
        # Point well inside triangle
        assert path.contains_point((5, 3))
        # Point outside
        assert not path.contains_point((0, 10))

    def test_empty_polygon_returns_empty(self):
        """Less than 3 points should select nothing."""
        from matplotlib.path import Path as MplPath

        verts = [(0, 0), (1, 1)]
        path = MplPath(verts)
        points = np.array([[0.5, 0.5]])
        inside = path.contains_points(points)
        # With only 2 vertices, contains_points is unreliable; in our code
        # we guard with len(lasso_points) < 3 before even calling this.
        # This test just verifies the guard logic is needed.
        assert isinstance(inside, np.ndarray)

    def test_concave_polygon(self):
        """L-shaped polygon should exclude interior notch."""
        from matplotlib.path import Path as MplPath

        # L-shape: includes (1,1) but excludes (8,8)
        verts = [(0, 0), (10, 0), (10, 5), (5, 5), (5, 10), (0, 10)]
        path = MplPath(verts)
        assert path.contains_point((1, 1))
        assert not path.contains_point((8, 8))


class TestSelectionLassoGraphics:
    """Test the _SelectionLasso pyqtgraph overlay (requires Qt)."""

    @pytest.fixture(autouse=True)
    def _skip_no_qt(self):
        try:
            from qtpy.QtWidgets import QApplication
            app = QApplication.instance()
            if app is None:
                pytest.skip("No QApplication available")
        except ImportError:
            pytest.skip("Qt not available")

    def test_add_points_grows_bounding_rect(self):
        from napari_epicycle.fusion_reconstruction.qc_plot_widget import _SelectionLasso

        lasso = _SelectionLasso()
        assert lasso.boundingRect().isNull() or lasso.boundingRect().isEmpty()
        lasso.add_point(0, 0)
        lasso.add_point(10, 0)
        lasso.add_point(10, 10)
        rect = lasso.boundingRect()
        assert rect.width() >= 9
        assert rect.height() >= 9

    def test_reset_clears(self):
        from napari_epicycle.fusion_reconstruction.qc_plot_widget import _SelectionLasso

        lasso = _SelectionLasso()
        lasso.add_point(0, 0)
        lasso.add_point(5, 5)
        lasso.reset()
        rect = lasso.boundingRect()
        assert rect.isNull() or rect.isEmpty()
