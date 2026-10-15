"""Tests for Z-stratified cross-section features (Group A)."""

from __future__ import annotations

import numpy as np
import pytest

from napari_epicycle.fusion_reconstruction.qc_features import extract_features


class TestComputeCrossSectionFeatures:
    """Tests for compute_cross_section_features."""

    def test_adds_expected_columns(self):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_cross_section_features,
        )
        # Cylinder: uniform cross-section along Z
        labels = np.zeros((20, 30, 30), dtype=np.int32)
        labels[3:17, 10:20, 10:20] = 1
        df = extract_features(labels)
        compute_cross_section_features(labels, df)
        expected = [
            "apical_area", "basal_area", "apical_basal_area_ratio",
            "max_cross_section_z_relative", "cross_section_taper",
            "cross_section_bulge",
        ]
        for col in expected:
            assert col in df.columns, f"Missing column: {col}"

    def test_cylinder_has_ratio_near_one(self):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_cross_section_features,
        )
        # Uniform cylinder: apical ≈ basal
        labels = np.zeros((30, 30, 30), dtype=np.int32)
        labels[5:25, 10:20, 10:20] = 1
        df = extract_features(labels)
        compute_cross_section_features(labels, df)
        ratio = df.iloc[0]["apical_basal_area_ratio"]
        assert 0.8 < ratio < 1.2, f"Cylinder ratio should be ~1.0, got {ratio}"

    def test_cylinder_has_low_taper(self):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_cross_section_features,
        )
        labels = np.zeros((30, 30, 30), dtype=np.int32)
        labels[5:25, 10:20, 10:20] = 1
        df = extract_features(labels)
        compute_cross_section_features(labels, df)
        taper = df.iloc[0]["cross_section_taper"]
        assert abs(taper) < 5.0, f"Cylinder taper should be ~0, got {taper}"

    def test_cylinder_has_low_bulge(self):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_cross_section_features,
        )
        labels = np.zeros((30, 30, 30), dtype=np.int32)
        labels[5:25, 10:20, 10:20] = 1
        df = extract_features(labels)
        compute_cross_section_features(labels, df)
        bulge = df.iloc[0]["cross_section_bulge"]
        assert bulge < 0.1, f"Cylinder bulge should be ~0, got {bulge}"

    def test_cone_has_low_ratio(self):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_cross_section_features,
        )
        # Cone: wide base (low Z), narrow top (high Z)
        labels = np.zeros((30, 40, 40), dtype=np.int32)
        for z in range(5, 25):
            # Radius decreases linearly with Z
            radius = int(15 - (z - 5) * 0.6)
            radius = max(radius, 2)
            cy, cx = 20, 20
            for y in range(cy - radius, cy + radius):
                for x in range(cx - radius, cx + radius):
                    if (y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2:
                        labels[z, y, x] = 1
        df = extract_features(labels)
        compute_cross_section_features(labels, df)
        ratio = df.iloc[0]["apical_basal_area_ratio"]
        assert ratio < 0.5, f"Cone ratio should be < 0.5, got {ratio}"

    def test_cone_has_negative_taper(self):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_cross_section_features,
        )
        labels = np.zeros((30, 40, 40), dtype=np.int32)
        for z in range(5, 25):
            radius = int(15 - (z - 5) * 0.6)
            radius = max(radius, 2)
            cy, cx = 20, 20
            for y in range(cy - radius, cy + radius):
                for x in range(cx - radius, cx + radius):
                    if (y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2:
                        labels[z, y, x] = 1
        df = extract_features(labels)
        compute_cross_section_features(labels, df)
        taper = df.iloc[0]["cross_section_taper"]
        assert taper < -5, f"Cone taper should be negative, got {taper}"

    def test_sphere_has_high_bulge(self):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_cross_section_features,
        )
        # Sphere: widest in middle, narrow at top and bottom
        labels = np.zeros((30, 30, 30), dtype=np.int32)
        cy, cx, cz = 15, 15, 15
        r = 10
        for z in range(30):
            for y in range(30):
                for x in range(30):
                    if (z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2 <= r ** 2:
                        labels[z, y, x] = 1
        df = extract_features(labels)
        compute_cross_section_features(labels, df)
        bulge = df.iloc[0]["cross_section_bulge"]
        assert bulge > 0.05, f"Sphere bulge should be > 0, got {bulge}"

    def test_max_cross_section_z_relative_sphere_is_middle(self):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_cross_section_features,
        )
        labels = np.zeros((30, 30, 30), dtype=np.int32)
        cy, cx, cz = 15, 15, 15
        r = 10
        for z in range(30):
            for y in range(30):
                for x in range(30):
                    if (z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2 <= r ** 2:
                        labels[z, y, x] = 1
        df = extract_features(labels)
        compute_cross_section_features(labels, df)
        z_rel = df.iloc[0]["max_cross_section_z_relative"]
        assert 0.3 < z_rel < 0.7, f"Sphere max cross-section should be ~0.5, got {z_rel}"

    def test_tiny_label_no_crash(self):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_cross_section_features,
        )
        labels = np.zeros((10, 10, 10), dtype=np.int32)
        labels[4:6, 4:6, 4:6] = 1  # 2x2x2
        df = extract_features(labels)
        compute_cross_section_features(labels, df)
        assert np.all(np.isfinite(df["apical_basal_area_ratio"].values))

    def test_empty_volume(self):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_cross_section_features,
        )
        labels = np.zeros((10, 10, 10), dtype=np.int32)
        df = extract_features(labels)
        compute_cross_section_features(labels, df)
        assert "apical_area" in df.columns

    def test_multiple_labels(self):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_cross_section_features,
        )
        labels = np.zeros((20, 30, 30), dtype=np.int32)
        labels[3:17, 3:10, 3:10] = 1   # cylinder
        labels[3:17, 15:25, 15:25] = 2  # another cylinder
        df = extract_features(labels)
        compute_cross_section_features(labels, df)
        assert len(df) == 2
        for col in ("apical_area", "basal_area", "cross_section_taper"):
            assert np.all(np.isfinite(df[col].values))

    def test_values_are_finite(self):
        from napari_epicycle.fusion_reconstruction.qc_features import (
            compute_cross_section_features,
        )
        labels = np.zeros((20, 30, 30), dtype=np.int32)
        labels[2:18, 5:25, 5:25] = 1
        df = extract_features(labels)
        compute_cross_section_features(labels, df)
        for col in ("apical_area", "basal_area", "apical_basal_area_ratio",
                     "max_cross_section_z_relative", "cross_section_taper",
                     "cross_section_bulge"):
            assert np.all(np.isfinite(df[col].values)), f"Non-finite in {col}"
