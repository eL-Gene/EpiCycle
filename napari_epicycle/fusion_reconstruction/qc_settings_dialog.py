"""QC Feature Extraction Settings dialog.

Provides a popup for configuring voxel spacing, boundary erosion depth,
neighbor search radius, and composite score method.  Accessed via the
"Settings..." button in the Curation panel.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from napari_epicycle.fusion_reconstruction.qc_features import (
    NUMERIC_FEATURES,
    PCA_FEATURES,
)


class QCSettingsDialog(QDialog):
    """Modal dialog for QC feature extraction parameters."""

    def __init__(self, parent=None, *, settings: Optional[dict] = None):
        super().__init__(parent)
        self.setWindowTitle("QC Feature Extraction Settings")
        self.setMinimumWidth(340)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # --- Voxel Spacing ---
        spacing_group = QGroupBox("Voxel Spacing")
        spacing_lay = QVBoxLayout(spacing_group)

        sp_row = QHBoxLayout()
        sp_row.setSpacing(6)
        self.spacing_z_sb = QDoubleSpinBox()
        self.spacing_y_sb = QDoubleSpinBox()
        self.spacing_x_sb = QDoubleSpinBox()
        for sb, axis in [
            (self.spacing_z_sb, "Z"),
            (self.spacing_y_sb, "Y"),
            (self.spacing_x_sb, "X"),
        ]:
            sb.setRange(0.001, 100.0)
            sb.setDecimals(3)
            sb.setValue(1.0)
            sb.setSingleStep(0.01)
            sb.setMaximumWidth(80)
            sb.setToolTip(
                f"{axis} voxel size in physical units (e.g. \u00b5m). "
                "Changing spacing after QC features have been extracted means "
                "the navigator's auto-pan + physical-unit features reference "
                "the previous spacing \u2014 re-extract Features to refresh."
            )
            sp_row.addWidget(QLabel(f"{axis}:"))
            sp_row.addWidget(sb)

        self.spacing_auto_btn = QPushButton("Auto-detect")
        self.spacing_auto_btn.setToolTip(
            "Auto-detect spacing from image layer scale"
        )
        sp_row.addWidget(self.spacing_auto_btn)
        sp_row.addStretch(1)

        spacing_lay.addLayout(sp_row)
        self.spacing_units_label = QLabel("voxel units")
        self.spacing_units_label.setStyleSheet("color: #888; font-size: 9px;")
        spacing_lay.addWidget(self.spacing_units_label)
        layout.addWidget(spacing_group)

        # --- Boundary Analysis ---
        boundary_group = QGroupBox("Boundary Analysis")
        boundary_lay = QVBoxLayout(boundary_group)
        boundary_desc = QLabel(
            "Defines a shell around each label via binary erosion.\n"
            "Used to compare boundary vs interior intensity (membrane QC)."
        )
        boundary_desc.setStyleSheet("color: #888; font-size: 9px;")
        boundary_desc.setWordWrap(True)
        boundary_lay.addWidget(boundary_desc)
        boundary_form = QFormLayout()
        boundary_lay.addLayout(boundary_form)
        self.erosion_iters_sb = QSpinBox()
        self.erosion_iters_sb.setRange(1, 10)
        self.erosion_iters_sb.setValue(2)
        self.erosion_iters_sb.setToolTip(
            "Number of binary erosion iterations to define the boundary shell.\n"
            "Higher = thicker boundary, thinner interior."
        )
        boundary_form.addRow("Erosion depth (voxels):", self.erosion_iters_sb)
        layout.addWidget(boundary_group)

        # --- Neighbor Detection ---
        neighbor_group = QGroupBox("Neighbor Detection")
        neighbor_lay = QVBoxLayout(neighbor_group)
        neighbor_desc = QLabel(
            "Graduated dilation to find adjacent labels and gap widths.\n"
            "Computes neighbor count, nearest distance, and contact fractions."
        )
        neighbor_desc.setStyleSheet("color: #888; font-size: 9px;")
        neighbor_desc.setWordWrap(True)
        neighbor_lay.addWidget(neighbor_desc)
        neighbor_form = QFormLayout()
        neighbor_lay.addLayout(neighbor_form)
        self.max_radius_sb = QSpinBox()
        self.max_radius_sb.setRange(1, 20)
        self.max_radius_sb.setValue(5)
        self.max_radius_sb.setToolTip(
            "Maximum dilation radius (voxels) for graduated neighbor search.\n"
            "Higher values bridge wider gaps between non-touching labels.\n"
            "Default 5 covers typical reconstruction gaps of 2\u20135 voxels."
        )
        neighbor_form.addRow("Max search radius (voxels):", self.max_radius_sb)
        layout.addWidget(neighbor_group)

        # --- Composite Score ---
        composite_group = QGroupBox("Composite Score")
        composite_lay = QVBoxLayout(composite_group)
        composite_desc = QLabel(
            "Aggregates features into a single outlier score per label.\n"
            "zscore: mean |z| across features. pca: PC1 oriented for outliers."
        )
        composite_desc.setStyleSheet("color: #888; font-size: 9px;")
        composite_desc.setWordWrap(True)
        composite_lay.addWidget(composite_desc)
        composite_form = QFormLayout()
        composite_lay.addLayout(composite_form)
        self.composite_method_combo = QComboBox()
        self.composite_method_combo.addItems(["zscore", "pca"])
        self.composite_method_combo.setToolTip(
            "zscore: mean absolute z-score across all features (simple, interpretable).\n"
            "pca: PCA PC1 oriented so outliers score high."
        )
        composite_form.addRow("Method:", self.composite_method_combo)

        layout.addWidget(composite_group)

        # --- Clustering ---
        cluster_group = QGroupBox("Clustering")
        cluster_lay = QVBoxLayout(cluster_group)
        cluster_desc = QLabel(
            "Clustering in PCA-reduced space.\n"
            "Standardize → PCA → cluster on top N components."
        )
        cluster_desc.setStyleSheet("color: #888; font-size: 9px;")
        cluster_desc.setWordWrap(True)
        cluster_lay.addWidget(cluster_desc)
        cluster_form = QFormLayout()
        cluster_lay.addLayout(cluster_form)

        # Algorithm selector
        self.cluster_algorithm_combo = QComboBox()
        self.cluster_algorithm_combo.addItems(["leiden", "hdbscan", "kmeans", "gmm"])
        self.cluster_algorithm_combo.setToolTip(
            "leiden: graph-based community detection (scanpy standard, no noise).\n"
            "hdbscan: density-based, auto-discovers clusters (may produce noise).\n"
            "kmeans: centroid-based, specify k (fast, spherical clusters).\n"
            "gmm: Gaussian mixture, specify k (elliptical clusters)."
        )
        cluster_form.addRow("Algorithm:", self.cluster_algorithm_combo)

        # Shared: PCA components
        self.n_cluster_components_sb = QSpinBox()
        self.n_cluster_components_sb.setRange(2, 15)
        self.n_cluster_components_sb.setValue(5)
        self.n_cluster_components_sb.setToolTip(
            "Number of PCA components to cluster on.\n"
            "More = more variance captured but noisier.\n"
            "Default 5 captures majority of variance while denoising."
        )
        cluster_form.addRow("PCA components:", self.n_cluster_components_sb)

        # HDBSCAN: min_cluster_size
        self._hdbscan_label = QLabel("Min cluster size:")
        self.min_cluster_size_sb = QSpinBox()
        self.min_cluster_size_sb.setRange(5, 100)
        self.min_cluster_size_sb.setValue(15)
        self.min_cluster_size_sb.setToolTip(
            "Minimum cluster size for HDBSCAN.\n"
            "Smaller values find more clusters; larger values merge small groups."
        )
        cluster_form.addRow(self._hdbscan_label, self.min_cluster_size_sb)

        # K-Means / GMM: n_clusters
        self._n_clusters_label = QLabel("Number of clusters:")
        self.n_clusters_sb = QSpinBox()
        self.n_clusters_sb.setRange(2, 30)
        self.n_clusters_sb.setValue(5)
        self.n_clusters_sb.setToolTip(
            "Number of clusters for K-Means or GMM."
        )
        cluster_form.addRow(self._n_clusters_label, self.n_clusters_sb)

        # Leiden: resolution
        self._resolution_label = QLabel("Resolution:")
        self.leiden_resolution_sb = QDoubleSpinBox()
        self.leiden_resolution_sb.setRange(0.01, 10.0)
        self.leiden_resolution_sb.setValue(1.0)
        self.leiden_resolution_sb.setSingleStep(0.1)
        self.leiden_resolution_sb.setDecimals(2)
        self.leiden_resolution_sb.setToolTip(
            "Leiden resolution parameter.\n"
            "Higher = more clusters, lower = fewer clusters.\n"
            "Default 1.0 is a good starting point."
        )
        cluster_form.addRow(self._resolution_label, self.leiden_resolution_sb)

        # Leiden: n_neighbors
        self._n_neighbors_label = QLabel("kNN neighbors:")
        self.leiden_n_neighbors_sb = QSpinBox()
        self.leiden_n_neighbors_sb.setRange(5, 50)
        self.leiden_n_neighbors_sb.setValue(15)
        self.leiden_n_neighbors_sb.setToolTip(
            "Number of nearest neighbors for kNN graph.\n"
            "Higher = smoother graph, lower = more local structure."
        )
        cluster_form.addRow(self._n_neighbors_label, self.leiden_n_neighbors_sb)

        # Show/hide params based on algorithm
        self.cluster_algorithm_combo.currentTextChanged.connect(
            self._on_cluster_algorithm_changed
        )
        self._on_cluster_algorithm_changed(
            self.cluster_algorithm_combo.currentText()
        )

        layout.addWidget(cluster_group)

        # --- PCA / Composite Feature Selection ---
        pca_group = QGroupBox("PCA / Composite Features")
        pca_lay = QVBoxLayout(pca_group)
        pca_lay.setSpacing(4)

        desc = QLabel(
            "Select features included in PCA and composite score.\n"
            "Defaults are curated for statistical independence."
        )
        desc.setStyleSheet("color: #888; font-size: 9px;")
        desc.setWordWrap(True)
        pca_lay.addWidget(desc)

        # Scrollable grid of checkboxes
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(180)
        scroll_widget = QWidget()
        grid = QGridLayout(scroll_widget)
        grid.setSpacing(2)

        # Exclude qc_score — it's circular
        self._feature_checkboxes: dict[str, QCheckBox] = {}
        available = [f for f in NUMERIC_FEATURES if f != "qc_score"]
        n_cols_grid = 2
        for i, feat in enumerate(available):
            cb = QCheckBox(feat)
            cb.setChecked(feat in PCA_FEATURES)
            cb.setToolTip(feat)
            self._feature_checkboxes[feat] = cb
            grid.addWidget(cb, i // n_cols_grid, i % n_cols_grid)

        scroll.setWidget(scroll_widget)
        pca_lay.addWidget(scroll)

        # Select All / Reset Defaults buttons
        btn_row = QHBoxLayout()
        self._select_all_btn = QPushButton("Select All")
        self._select_all_btn.clicked.connect(self._select_all_features)
        self._reset_defaults_btn = QPushButton("Reset Defaults")
        self._reset_defaults_btn.clicked.connect(self._reset_default_features)
        btn_row.addWidget(self._select_all_btn)
        btn_row.addWidget(self._reset_defaults_btn)
        btn_row.addStretch(1)
        pca_lay.addLayout(btn_row)

        layout.addWidget(pca_group)

        # --- OK / Cancel ---
        btn_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        # Apply stored settings if provided
        if settings:
            self._apply_settings(settings)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_settings(self) -> dict:
        """Return current dialog values as a dict."""
        z = self.spacing_z_sb.value()
        y = self.spacing_y_sb.value()
        x = self.spacing_x_sb.value()
        spacing = None if (z == 1.0 and y == 1.0 and x == 1.0) else (z, y, x)
        return {
            "spacing": spacing,
            "spacing_values": (z, y, x),
            "erosion_iters": self.erosion_iters_sb.value(),
            "max_radius": self.max_radius_sb.value(),
            "composite_method": self.composite_method_combo.currentText(),
            "cluster_algorithm": self.cluster_algorithm_combo.currentText(),
            "min_cluster_size": self.min_cluster_size_sb.value(),
            "n_clusters": self.n_clusters_sb.value(),
            "leiden_resolution": self.leiden_resolution_sb.value(),
            "leiden_n_neighbors": self.leiden_n_neighbors_sb.value(),
            "n_cluster_components": self.n_cluster_components_sb.value(),
            "pca_features": self.get_selected_features(),
        }

    def get_selected_features(self) -> List[str]:
        """Return the list of checked PCA feature names."""
        return [
            feat for feat, cb in self._feature_checkboxes.items()
            if cb.isChecked()
        ]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _apply_settings(self, settings: dict) -> None:
        vals = settings.get("spacing_values")
        if vals and len(vals) == 3:
            self.spacing_z_sb.setValue(vals[0])
            self.spacing_y_sb.setValue(vals[1])
            self.spacing_x_sb.setValue(vals[2])
        if "erosion_iters" in settings:
            self.erosion_iters_sb.setValue(settings["erosion_iters"])
        if "max_radius" in settings:
            self.max_radius_sb.setValue(settings["max_radius"])
        if "composite_method" in settings:
            idx = self.composite_method_combo.findText(settings["composite_method"])
            if idx >= 0:
                self.composite_method_combo.setCurrentIndex(idx)
        if "cluster_algorithm" in settings:
            idx = self.cluster_algorithm_combo.findText(settings["cluster_algorithm"])
            if idx >= 0:
                self.cluster_algorithm_combo.setCurrentIndex(idx)
        if "min_cluster_size" in settings:
            self.min_cluster_size_sb.setValue(settings["min_cluster_size"])
        if "n_clusters" in settings:
            self.n_clusters_sb.setValue(settings["n_clusters"])
        if "leiden_resolution" in settings:
            self.leiden_resolution_sb.setValue(settings["leiden_resolution"])
        if "leiden_n_neighbors" in settings:
            self.leiden_n_neighbors_sb.setValue(settings["leiden_n_neighbors"])
        if "n_cluster_components" in settings:
            self.n_cluster_components_sb.setValue(settings["n_cluster_components"])
        if "pca_features" in settings:
            selected = set(settings["pca_features"])
            for feat, cb in self._feature_checkboxes.items():
                cb.setChecked(feat in selected)

    def _select_all_features(self) -> None:
        """Check all feature checkboxes."""
        for cb in self._feature_checkboxes.values():
            cb.setChecked(True)

    def _reset_default_features(self) -> None:
        """Restore default PCA_FEATURES selection."""
        for feat, cb in self._feature_checkboxes.items():
            cb.setChecked(feat in PCA_FEATURES)

    def _on_cluster_algorithm_changed(self, algorithm: str) -> None:
        """Show/hide clustering parameters based on selected algorithm."""
        is_hdbscan = algorithm == "hdbscan"
        is_kmeans_gmm = algorithm in ("kmeans", "gmm")
        is_leiden = algorithm == "leiden"

        self._hdbscan_label.setVisible(is_hdbscan)
        self.min_cluster_size_sb.setVisible(is_hdbscan)

        self._n_clusters_label.setVisible(is_kmeans_gmm)
        self.n_clusters_sb.setVisible(is_kmeans_gmm)

        self._resolution_label.setVisible(is_leiden)
        self.leiden_resolution_sb.setVisible(is_leiden)
        self._n_neighbors_label.setVisible(is_leiden)
        self.leiden_n_neighbors_sb.setVisible(is_leiden)
