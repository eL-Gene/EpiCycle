"""Step 3 QC Extraction & Settings dialog.

Replacement for the older ``QCSettingsDialog`` (`qc_settings_dialog.py`).
Phase A scope: full settings panel + collapsible feature-group selection
for the PCA composite, plus the three new knobs needed by the upcoming
feature-registry migration (``exclude_border``, ``invert_z``,
``channel_configs``).

Reuses the collapsible feature-group widget pattern from
``napari_epicycle/segmentation/extract_dialog.py`` (forked, not
imported, because Step 3 needs additional settings the Step 4 dialog
doesn't expose: composite method, clustering algorithm, per-algorithm
params, voxel spacing, erosion depth, neighbor radius).

Sparkline rendering per feature row is deferred to Phase B.

Output contract: ``get_settings()`` returns a superset of the keys the
old QCSettingsDialog produced — controller's
``self._qc_settings.get(key, default)`` calls keep working unchanged.
New keys: ``exclude_border``, ``invert_z``, ``channel_configs``.
"""

from __future__ import annotations

from typing import Optional

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from napari_epicycle.fusion_reconstruction._step3_pr_text import (
    QC_SPACING_FALLBACK,
)
from napari_epicycle.fusion_reconstruction.qc_features import (
    PCA_FEATURES,
)


# ---------------------------------------------------------------------------
# Styling constants (mirror Step 4's extract_dialog for visual continuity)
# ---------------------------------------------------------------------------

_BG = "#2b2b2b"
_BG_CONTENT = "#1e1e2e"
_TEXT = "#e0e0e0"
_TEXT_DIM = "#999999"
_TEXT_DIMMER = "#777777"
_ACCENT = "#d4a843"  # Step 3 amber (matches PRIMARY_BTN_STYLE)
from ._step3_rows import WARN_COLOUR as _WARN  # the rows' warn colour
_ACCENT_BG = "rgba(212,168,67,0.20)"
_ACCENT_BORDER = "rgba(212,168,67,0.40)"
_HEADER_BG = "#333348"
_GROUP_BG = "rgba(255,255,255,0.03)"
_GROUP_HOVER = "rgba(255,255,255,0.06)"
_SECTION_BG = "rgba(255,255,255,0.025)"


# ---------------------------------------------------------------------------
# Dynamic description text shown under method/algorithm dropdowns.
#
# These short strings tell the user what each option actually does (and
# which params it consumes) without making them hover for the tooltip.
# Phase A.3 (2026-05-04).
# ---------------------------------------------------------------------------

_COMPOSITE_METHOD_DESCRIPTIONS: dict[str, str] = {
    "zscore": (
        "Mean absolute z-score across selected features. Simple "
        "and interpretable; weights every feature equally."
    ),
    "pca": (
        "L2 distance in the top-N PC space (N = 'PCA dimensions' "
        "in Clustering). Geometric outlier score; high-variance "
        "features contribute more."
    ),
}

_CLUSTER_ALGO_DESCRIPTIONS: dict[str, str] = {
    "leiden": (
        "Graph-based community detection on a KNN graph. "
        "Modularity-based; handles arbitrary cluster shapes. "
        "Uses N neighbors + Resolution."
    ),
    "hdbscan": (
        "Density-based clustering. Discovers arbitrary-shape "
        "clusters; low-density points are labeled as noise. "
        "Uses Min cluster size."
    ),
    "kmeans": (
        "Partitions cells into K clusters by minimizing "
        "within-cluster variance. Best for roughly spherical "
        "clusters. Uses N clusters."
    ),
    "gmm": (
        "Fits K Gaussian mixtures with soft probabilistic "
        "assignments. Tolerates elliptical clusters better than "
        "K-Means. Uses N clusters."
    ),
}


# ---------------------------------------------------------------------------
# Collapsible feature-group widget (forked from extract_dialog._FeatureGroupWidget)
# ---------------------------------------------------------------------------

class _FeatureGroupWidget(QWidget):
    """Collapsible feature group with per-feature PCA checkboxes.

    Click the header to expand/collapse the feature list.  Each
    checkbox toggles whether that feature is included in the
    Step 3 PCA composite (``qc_score`` driver).
    """

    selection_changed = Signal()

    def __init__(
        self,
        category: str,
        features: list[str],
        pca_set: set[str],
        descriptions: dict[str, str],
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._category = category
        self._features = features
        self._expanded = False
        self._checkboxes: dict[str, QCheckBox] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # --- Header (clickable) ---
        self._header = QWidget(self)
        self._header.setCursor(Qt.PointingHandCursor)
        self._header.setStyleSheet(
            f"QWidget {{ background: {_GROUP_BG}; border-radius: 3px; }}"
            f"QWidget:hover {{ background: {_GROUP_HOVER}; }}"
        )
        header_lay = QHBoxLayout(self._header)
        header_lay.setContentsMargins(8, 5, 8, 5)
        header_lay.setSpacing(6)

        self._arrow = QLabel("▸", self._header)  # ▸
        self._arrow.setFixedWidth(12)
        self._arrow.setStyleSheet(f"color: {_TEXT_DIM}; font-size: 11px;")
        header_lay.addWidget(self._arrow)

        self._name_label = QLabel(
            f"{category} ({len(features)})", self._header,
        )
        self._name_label.setStyleSheet(
            f"color: {_TEXT}; font-size: 12px; font-weight: bold;"
        )
        header_lay.addWidget(self._name_label)
        header_lay.addStretch()

        self._pca_label = QLabel(self._header)
        self._pca_label.setStyleSheet("font-size: 11px;")
        header_lay.addWidget(self._pca_label)

        self._header.mousePressEvent = self._toggle_expand
        layout.addWidget(self._header)

        # --- Feature list (hidden until expanded) ---
        self._feature_container = QWidget(self)
        self._feature_container.setVisible(False)
        feat_lay = QVBoxLayout(self._feature_container)
        feat_lay.setContentsMargins(28, 2, 8, 4)
        feat_lay.setSpacing(1)

        for feat in features:
            cb = QCheckBox(feat, self._feature_container)
            cb.setChecked(feat in pca_set)
            tooltip = descriptions.get(feat, "")
            if tooltip:
                cb.setToolTip(tooltip)
            cb.setStyleSheet(self._cb_style(feat in pca_set))
            cb.stateChanged.connect(self._on_cb_changed)
            feat_lay.addWidget(cb)
            self._checkboxes[feat] = cb

        layout.addWidget(self._feature_container)
        self._refresh_pca_label()

    # --- Public API ---

    def checked_features(self) -> list[str]:
        return [f for f, cb in self._checkboxes.items() if cb.isChecked()]

    def set_pca_features(self, pca_set: set[str]) -> None:
        for feat, cb in self._checkboxes.items():
            cb.blockSignals(True)
            cb.setChecked(feat in pca_set)
            cb.setStyleSheet(self._cb_style(feat in pca_set))
            cb.blockSignals(False)
        self._refresh_pca_label()

    @property
    def category(self) -> str:
        return self._category

    # --- Internals ---

    def _toggle_expand(self, _event=None) -> None:
        self._expanded = not self._expanded
        self._feature_container.setVisible(self._expanded)
        self._arrow.setText("▾" if self._expanded else "▸")  # ▾ / ▸

    def _on_cb_changed(self) -> None:
        sender = self.sender()
        if sender is not None:
            sender.setStyleSheet(self._cb_style(sender.isChecked()))
        self._refresh_pca_label()
        self.selection_changed.emit()

    def _refresh_pca_label(self) -> None:
        n = sum(1 for cb in self._checkboxes.values() if cb.isChecked())
        if n > 0:
            self._pca_label.setText(f"● {n} in PCA")
            self._pca_label.setStyleSheet(
                f"color: {_ACCENT}; font-size: 11px;",
            )
        else:
            self._pca_label.setText("○ 0 in PCA")
            self._pca_label.setStyleSheet(
                f"color: {_TEXT_DIMMER}; font-size: 11px;",
            )

    @staticmethod
    def _cb_style(checked: bool) -> str:
        color = _TEXT if checked else _TEXT_DIM
        return f"QCheckBox {{ color: {color}; font-size: 11px; }}"


# ---------------------------------------------------------------------------
# Main dialog
# ---------------------------------------------------------------------------

class QCExtractionDialog(QDialog):
    """Step 3 QC extraction settings + feature selection dialog.

    Phase A: full settings panel + collapsible feature-group selection.
    Sparkline rendering deferred to Phase B.

    Parameters
    ----------
    parent : QWidget
        Parent widget (Step 3 main UI).
    settings : dict, optional
        Pre-populated settings dict (typically the controller's
        ``self._qc_settings``).
    feature_manifest : dict[str, list[str]], optional
        ``{category: [feature_names]}`` from
        ``feature_registry.get_feature_manifest()``.  When None, the
        feature-selection section falls back to a flat checkbox grid
        of ``PCA_FEATURES`` (legacy behavior).
    descriptions : dict[str, str], optional
        Feature name → tooltip description mapping (from
        ``analysis_widget._FEATURE_DESCRIPTIONS``).
    available_channels : list[dict], optional
        Auto-detected channels: ``[{"index": int, "prefix": str}, ...]``.
        When empty/None, the channel section is hidden — Step 3 falls
        back to using the manifest's seg_channel only.
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        settings: Optional[dict] = None,
        feature_manifest: Optional[dict[str, list[str]]] = None,
        descriptions: Optional[dict[str, str]] = None,
        available_channels: Optional[list[dict]] = None,
        patch_spacing_lines: Optional[list] = None,
    ):
        super().__init__(parent)
        self._patch_spacing_lines = list(patch_spacing_lines or [])
        self.setWindowTitle("QC Extraction & Settings")
        self.setMinimumWidth(480)
        # Allow vertical resize.  Pre-fix used a hardcoded
        # setMaximumHeight(720) which prevented the user from making
        # the dialog taller — content below the cap was unreachable
        # without scrolling on every section.  Now: set a generous
        # minimum + initial size, no maximum.
        self.setMinimumHeight(520)
        try:
            screen_h = self.screen().geometry().height()
            self.resize(self.minimumWidth(), int(screen_h * 0.75))
        except Exception:
            self.resize(480, 700)
        # Enable the corner size grip so the resize affordance is
        # visible (some platforms don't draw resize handles on the
        # window frame for modal dialogs).
        self.setSizeGripEnabled(True)
        self.setStyleSheet(
            f"QDialog {{ background: {_BG}; border: 1px solid "
            f"rgba(255,255,255,0.12); border-radius: 6px; }}"
            f"QLabel {{ color: {_TEXT}; font-size: 11px; }}"
            f"QGroupBox {{ color: {_TEXT}; font-size: 11px; "
            f"font-weight: bold; border: 1px solid "
            f"rgba(255,255,255,0.10); border-radius: 4px; "
            f"margin-top: 8px; padding-top: 6px; }}"
            f"QGroupBox::title {{ subcontrol-origin: margin; "
            f"left: 8px; padding: 0 4px; }}"
            f"QSpinBox, QDoubleSpinBox, QComboBox {{ "
            f"background: {_BG_CONTENT}; color: {_TEXT}; "
            f"border: 1px solid rgba(255,255,255,0.12); "
            f"border-radius: 3px; padding: 2px 6px; font-size: 11px; }}"
            f"QComboBox QAbstractItemView {{ background: {_BG}; "
            f"color: {_TEXT}; selection-background-color: {_HEADER_BG}; }}"
        )

        self._descriptions = descriptions or {}
        self._available_channels = available_channels or []
        self._group_widgets: dict[str, _FeatureGroupWidget] = {}
        self._channel_checkboxes: dict[int, QCheckBox] = {}

        main_lay = QVBoxLayout(self)
        main_lay.setContentsMargins(16, 12, 16, 12)
        main_lay.setSpacing(8)

        # --- Title ---
        title = QLabel("QC Extraction & Settings")
        title.setStyleSheet(
            f"color: #eeeeee; font-size: 14px; font-weight: bold;",
        )
        main_lay.addWidget(title)

        # --- Sections (top-down) ---
        # Restructured 2026-05-04 per user UX direction: split the
        # original "Composite & clustering" section into separate
        # "QC Score" and "Clustering" sections, and fold the feature
        # selection panel into "QC Score" so the "what feeds qc_score"
        # decision sits next to the "how qc_score is computed" choice.
        self._build_extraction_section(main_lay)
        if self._available_channels:
            self._build_channels_section(main_lay)
        self._build_qc_score_section(main_lay, feature_manifest)
        self._build_clustering_section(main_lay)
        self._build_dimred_section(main_lay)

        # --- OK / Cancel ---
        btn_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
        )
        btn_box.button(QDialogButtonBox.Ok).setStyleSheet(
            f"QPushButton {{ background: {_ACCENT_BG}; color: {_TEXT}; "
            f"border: 1px solid {_ACCENT_BORDER}; border-radius: 3px; "
            f"padding: 4px 14px; font-weight: bold; }}"
            f"QPushButton:hover {{ background: {_ACCENT}; color: black; }}"
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        main_lay.addWidget(btn_box)

        # Apply incoming settings
        if settings:
            self._apply_settings(settings)

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def set_patch_spacing_lines(self, lines) -> None:
        """The read-only voxel-size line: ``[(text, warn)]`` (the first is
        drawn); empty hides it."""
        self._patch_spacing_lines = list(lines or [])
        label = self.spacing_patch_label
        if not self._patch_spacing_lines:
            label.setText("")
            label.setVisible(False)
            label.setHidden(True)
            return
        text, warn = self._patch_spacing_lines[0]
        colour = _WARN if warn else _TEXT_DIM
        label.setStyleSheet(f"color: {colour}; font-size: 11px;")
        label.setText(str(text))
        # the field column sits in a group box inside a scroll area, which
        # does not hand a wrapped label its height-for-width: reserve the
        # lines the text needs at a conservative column width (audit L6)
        from qtpy.QtCore import QRect
        wrap = label.fontMetrics().boundingRect(
            QRect(0, 0, 230, 10000), int(Qt.TextWordWrap), str(text))
        label.setMinimumHeight(wrap.height() + 2)
        label.setHidden(False)

    def _build_extraction_section(self, parent_lay: QVBoxLayout) -> None:
        from qtpy.QtWidgets import QFormLayout, QGroupBox

        group = QGroupBox("Extraction parameters")
        lay = QFormLayout(group)
        lay.setContentsMargins(8, 12, 8, 8)
        lay.setSpacing(4)

        # Voxel spacing — 3 spinboxes in a row + Auto button
        sp_row = QHBoxLayout()
        sp_row.setSpacing(4)
        self.spacing_z_sb = QDoubleSpinBox()
        self.spacing_y_sb = QDoubleSpinBox()
        self.spacing_x_sb = QDoubleSpinBox()
        for sb, axis in (
            (self.spacing_z_sb, "Z"),
            (self.spacing_y_sb, "Y"),
            (self.spacing_x_sb, "X"),
        ):
            sb.setRange(0.001, 100.0)
            sb.setDecimals(3)
            sb.setValue(1.0)
            sb.setSingleStep(0.01)
            sb.setMaximumWidth(72)
            sb.setToolTip(
                f"{axis} voxel size (physical units, e.g. µm).\n"
                "Leave at 1.0 to compute features in voxel units.\n\n"
                "Changing spacing after QC features have been extracted "
                "means the navigator's auto-pan + physical-unit features "
                "(area, volume, etc.) reference the previous spacing — "
                "re-extract Features to refresh."
            )
            sp_row.addWidget(QLabel(f"{axis}:"))
            sp_row.addWidget(sb)
        self.spacing_auto_btn = QPushButton("Auto")
        self.spacing_auto_btn.setToolTip(
            "Auto-detect voxel spacing from the active image layer's scale",
        )
        self.spacing_auto_btn.setMaximumWidth(60)
        sp_row.addWidget(self.spacing_auto_btn)
        sp_row.addStretch(1)
        lay.addRow("Spacing:", _row_widget(sp_row))
        # S1 (4.3a): what the next extraction will actually use for the
        # patch(es) -- read-only; the numbers above are only the fallback
        self.spacing_patch_label = QLabel("")
        self.spacing_patch_label.setWordWrap(True)
        lay.addRow("", self.spacing_patch_label)
        self.set_patch_spacing_lines(self._patch_spacing_lines)
        # R7a: this value is now only the fallback for patches with
        # no voxel size of their own
        self.spacing_fallback_label = QLabel(QC_SPACING_FALLBACK)
        self.spacing_fallback_label.setWordWrap(True)
        self.spacing_fallback_label.setStyleSheet(
            f"color: {_TEXT_DIM}; font-size: 11px;")
        lay.addRow("", self.spacing_fallback_label)

        # Erosion depth + neighbor radius
        self.erosion_iters_sb = QSpinBox()
        self.erosion_iters_sb.setRange(0, 10)
        self.erosion_iters_sb.setValue(2)
        self.erosion_iters_sb.setMaximumWidth(72)
        self.erosion_iters_sb.setToolTip(
            "Erosion iterations for boundary/interior intensity split",
        )
        lay.addRow("Erosion iters:", self.erosion_iters_sb)

        self.max_radius_sb = QSpinBox()
        self.max_radius_sb.setRange(1, 50)
        self.max_radius_sb.setValue(5)
        self.max_radius_sb.setMaximumWidth(72)
        self.max_radius_sb.setToolTip(
            "Max neighbor search radius in voxels (graduated dilation)",
        )
        lay.addRow("Neighbor radius:", self.max_radius_sb)

        # New: exclude_border + invert_z
        self.exclude_border_cb = QCheckBox(
            "Exclude border cells from composite/PCA fitting",
        )
        self.exclude_border_cb.setToolTip(
            "Cells touching the volume boundary (z=0 or z=max) can\n"
            "skew composite scores.  When enabled, fit qc_score on\n"
            "non-border cells only and project border cells as\n"
            "supplementary points."
        )
        lay.addRow("", self.exclude_border_cb)

        self.invert_z_cb = QCheckBox("Invert Z axis (flip apical/basal)")
        self.invert_z_cb.setToolTip(
            "If your dataset has apical at z=max instead of z=0,\n"
            "enable this to compute cross-section features\n"
            "(apical_basal_area_ratio etc.) with the right orientation.",
        )
        lay.addRow("", self.invert_z_cb)

        parent_lay.addWidget(group)

    def _build_qc_score_section(
        self,
        parent_lay: QVBoxLayout,
        feature_manifest: Optional[dict[str, list[str]]],
    ) -> None:
        """**QC Score** section — composite method + features used.

        Replaces the bottom "Features in qc_score (PCA composite)"
        section: feature selection lives here next to the method
        choice, since the two decisions are inseparable ("how is
        qc_score computed, on which features?").
        """
        from qtpy.QtWidgets import QFormLayout, QGroupBox

        group = QGroupBox("QC Score (qc_score)")
        outer = QVBoxLayout(group)
        outer.setContentsMargins(8, 12, 8, 8)
        outer.setSpacing(6)

        # --- Method row ---
        form = QFormLayout()
        form.setSpacing(4)

        self.composite_method_combo = QComboBox()
        self.composite_method_combo.addItems(["zscore", "pca"])
        self.composite_method_combo.setCurrentText("pca")
        self.composite_method_combo.setMaximumWidth(140)
        self.composite_method_combo.setToolTip(
            "How qc_score is computed from the selected features:\n"
            "  zscore — mean absolute z-score across selected features.\n"
            "  pca    — L2 norm in the top-N PCA component space\n"
            "           (N = 'PCA dimensions' setting in Clustering)."
        )
        form.addRow("Method:", self.composite_method_combo)

        # Dynamic description that swaps on method change so the user
        # sees what the dropdown actually does without hovering for a
        # tooltip.  Style mirrors the Intensity-channels hint label.
        self._qc_method_desc = QLabel()
        self._qc_method_desc.setWordWrap(True)
        self._qc_method_desc.setStyleSheet(
            f"color: {_TEXT_DIM}; font-size: 10px; "
            "padding: 2px 0 2px 0;"
        )
        form.addRow(self._qc_method_desc)
        self.composite_method_combo.currentTextChanged.connect(
            self._on_composite_method_changed,
        )
        self._on_composite_method_changed(
            self.composite_method_combo.currentText(),
        )

        outer.addLayout(form)

        # --- Feature selection (collapsible groups) ---
        feat_label = QLabel("Features used in qc_score:")
        feat_label.setStyleSheet(
            f"color: {_TEXT}; font-size: 11px; font-weight: bold;"
            " margin-top: 4px;"
        )
        outer.addWidget(feat_label)

        # Selection counter + Select All / Reset Defaults
        ctrl_row = QHBoxLayout()
        self._selection_label = QLabel()
        self._selection_label.setStyleSheet(
            f"color: {_ACCENT}; font-size: 11px; font-weight: bold;",
        )
        ctrl_row.addWidget(self._selection_label)
        ctrl_row.addStretch(1)
        self._select_all_btn = QPushButton("Select all")
        self._select_all_btn.setMaximumWidth(90)
        self._select_all_btn.clicked.connect(self._select_all_features)
        self._reset_defaults_btn = QPushButton("Reset defaults")
        self._reset_defaults_btn.setMaximumWidth(110)
        self._reset_defaults_btn.clicked.connect(
            self._reset_default_features,
        )
        ctrl_row.addWidget(self._select_all_btn)
        ctrl_row.addWidget(self._reset_defaults_btn)
        outer.addLayout(ctrl_row)

        # Scrollable groups
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setMinimumHeight(180)

        scroll_widget = QWidget()
        scroll_lay = QVBoxLayout(scroll_widget)
        scroll_lay.setContentsMargins(0, 0, 0, 0)
        scroll_lay.setSpacing(2)

        pca_set = set(PCA_FEATURES)
        if feature_manifest:
            for category, features in feature_manifest.items():
                gw = _FeatureGroupWidget(
                    category, features, pca_set, self._descriptions,
                )
                gw.selection_changed.connect(self._refresh_selection_count)
                scroll_lay.addWidget(gw)
                self._group_widgets[category] = gw
        else:
            for feat in PCA_FEATURES:
                cb = QCheckBox(feat, scroll_widget)
                cb.setChecked(True)
                cb.stateChanged.connect(self._refresh_selection_count)
                scroll_lay.addWidget(cb)
                self._group_widgets[feat] = _FlatCBWrapper(cb, feat)

        scroll_lay.addStretch(1)
        scroll.setWidget(scroll_widget)
        outer.addWidget(scroll, 1)

        parent_lay.addWidget(group, 1)
        self._refresh_selection_count()

    def _build_clustering_section(self, parent_lay: QVBoxLayout) -> None:
        """**Clustering** section — algorithm choice + per-algo params.

        Now separated from QC Score.  Adds the "PCA dimensions" setting
        (renamed from "PCA dims for clustering" to reflect that it
        also drives the qc_score PC count when method=pca) and an
        Auto-run checkbox for triggering clustering immediately after
        Extract.
        """
        from qtpy.QtWidgets import QFormLayout, QGroupBox

        group = QGroupBox("Clustering")
        outer = QVBoxLayout(group)
        outer.setContentsMargins(8, 12, 8, 8)
        outer.setSpacing(4)

        form = QFormLayout()
        form.setSpacing(4)

        self.cluster_algorithm_combo = QComboBox()
        self.cluster_algorithm_combo.addItems(
            ["leiden", "hdbscan", "kmeans", "gmm"],
        )
        self.cluster_algorithm_combo.setMaximumWidth(140)
        self.cluster_algorithm_combo.currentTextChanged.connect(
            self._on_cluster_algorithm_changed,
        )
        form.addRow("Algorithm:", self.cluster_algorithm_combo)

        # Dynamic description — same pattern as the QC Score method
        # description above.  Updated by _on_cluster_algorithm_changed.
        self._cluster_algo_desc = QLabel()
        self._cluster_algo_desc.setWordWrap(True)
        self._cluster_algo_desc.setStyleSheet(
            f"color: {_TEXT_DIM}; font-size: 10px; "
            "padding: 2px 0 2px 0;"
        )
        form.addRow(self._cluster_algo_desc)

        # Per-algorithm params (visibility toggled on selection)
        self._resolution_label = QLabel("Resolution:")
        self.leiden_resolution_sb = QDoubleSpinBox()
        self.leiden_resolution_sb.setRange(0.1, 10.0)
        self.leiden_resolution_sb.setSingleStep(0.1)
        self.leiden_resolution_sb.setValue(1.0)
        self.leiden_resolution_sb.setMaximumWidth(72)
        form.addRow(self._resolution_label, self.leiden_resolution_sb)

        self._n_neighbors_label = QLabel("N neighbors:")
        self.leiden_n_neighbors_sb = QSpinBox()
        self.leiden_n_neighbors_sb.setRange(2, 200)
        self.leiden_n_neighbors_sb.setValue(15)
        self.leiden_n_neighbors_sb.setMaximumWidth(72)
        form.addRow(self._n_neighbors_label, self.leiden_n_neighbors_sb)

        self._hdbscan_label = QLabel("Min cluster size:")
        self.min_cluster_size_sb = QSpinBox()
        self.min_cluster_size_sb.setRange(2, 1000)
        self.min_cluster_size_sb.setValue(10)
        self.min_cluster_size_sb.setMaximumWidth(72)
        form.addRow(self._hdbscan_label, self.min_cluster_size_sb)

        self._n_clusters_label = QLabel("N clusters:")
        self.n_clusters_sb = QSpinBox()
        self.n_clusters_sb.setRange(2, 50)
        self.n_clusters_sb.setValue(5)
        self.n_clusters_sb.setMaximumWidth(72)
        form.addRow(self._n_clusters_label, self.n_clusters_sb)

        # Renamed from "PCA dims for clustering" — this also drives
        # the qc_score PC count when method=pca (Phase A.2 fix).
        self.n_cluster_components_sb = QSpinBox()
        self.n_cluster_components_sb.setRange(2, 30)
        self.n_cluster_components_sb.setValue(5)
        self.n_cluster_components_sb.setMaximumWidth(72)
        self.n_cluster_components_sb.setToolTip(
            "Top-N PCA components used by:\n"
            "  - clustering algorithm (input dimensionality), and\n"
            "  - qc_score when method=pca (L2 norm over top-N PCs).\n"
            "  Phase E (2026-05-04): also the number of PC1..PCN\n"
            "  columns persisted to the CSV at extract time."
        )
        form.addRow("PCA dimensions:", self.n_cluster_components_sb)

        # Phase E (2026-05-04) — distance metric for the kNN graph.
        # Shared with UMAP / t-SNE since both run on the same kNN
        # graph as Leiden (matches scanpy convention).
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(
            ["euclidean", "manhattan", "cosine", "correlation"],
        )
        self.metric_combo.setMaximumWidth(140)
        self.metric_combo.setToolTip(
            "Distance metric for the kNN graph used by Leiden,\n"
            "UMAP, and t-SNE.\n"
            "  euclidean   — Default; standard L2 distance.\n"
            "  manhattan   — L1 distance; more robust to outliers.\n"
            "  cosine      — Angle-based; ignores magnitude.\n"
            "  correlation — Pearson; useful when feature scales\n"
            "                vary widely."
        )
        form.addRow("Distance metric:", self.metric_combo)

        outer.addLayout(form)

        # Auto-run checkbox — triggers clustering immediately after
        # Extract Features completes.  Uses the params above.
        self.auto_cluster_cb = QCheckBox("Auto-run clustering after Extract")
        self.auto_cluster_cb.setToolTip(
            "When checked, clustering runs automatically with the\n"
            "algorithm and parameters above as soon as Extract\n"
            "Features completes.  Otherwise clustering is triggered\n"
            "explicitly via the Cluster button (currently hidden)."
        )
        outer.addWidget(self.auto_cluster_cb)

        parent_lay.addWidget(group)

        # Initial visibility for per-algorithm rows
        self._on_cluster_algorithm_changed(
            self.cluster_algorithm_combo.currentText(),
        )

    def _build_dimred_section(self, parent_lay: QVBoxLayout) -> None:
        """**Nonlinear Dimensionality Reduction** section.

        Adds opt-in computation of UMAP and t-SNE during Extract.
        When checked, the worker calls
        :func:`compute_umap_embedding` / :func:`compute_tsne_embedding`
        and the resulting columns (``UMAP_1``/``UMAP_2``,
        ``tSNE_1``/``tSNE_2``) are written to the per-patch CSV.

        Both methods read ``n_neighbors`` from the Clustering section
        (shared with Leiden) and ``PCA dimensions`` from the same place
        for the PCA pre-reduction.  Method-specific knobs live here:
        ``min_dist`` for UMAP, ``perplexity`` for t-SNE.  Defaults
        OFF so the extraction loop stays fast unless the user
        explicitly opts in.
        """
        from qtpy.QtWidgets import QFormLayout, QGroupBox

        group = QGroupBox("Nonlinear Dimensionality Reduction")
        outer = QVBoxLayout(group)
        outer.setContentsMargins(8, 12, 8, 8)
        outer.setSpacing(4)

        hint = QLabel(
            "Optional 2D embeddings for the scatter plot.  Both "
            "share the 'N neighbors' (UMAP) and 'PCA dimensions' "
            "settings from Clustering, so clusters and embedding "
            "agree visually.  PCA is always computed.",
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {_TEXT_DIM}; font-size: 10px;")
        outer.addWidget(hint)

        # --- UMAP row ---
        self.compute_umap_cb = QCheckBox("Compute UMAP")
        self.compute_umap_cb.setToolTip(
            "Compute 2D UMAP during Extract Features and persist as\n"
            "UMAP_1 / UMAP_2 columns in the per-patch CSV.\n"
            "Reads N neighbors from the Clustering section (shared\n"
            "with Leiden) so clusters and the embedding use the same\n"
            "k-NN graph.  Adds ~10-30s per patch on typical sizes."
        )
        outer.addWidget(self.compute_umap_cb)

        umap_form = QFormLayout()
        umap_form.setSpacing(4)
        umap_form.setContentsMargins(20, 0, 0, 0)
        self.umap_min_dist_sb = QDoubleSpinBox()
        self.umap_min_dist_sb.setRange(0.0, 1.0)
        self.umap_min_dist_sb.setSingleStep(0.05)
        self.umap_min_dist_sb.setDecimals(2)
        self.umap_min_dist_sb.setValue(0.10)
        self.umap_min_dist_sb.setMaximumWidth(72)
        self.umap_min_dist_sb.setToolTip(
            "UMAP min_dist parameter.  Lower = tighter local structure;\n"
            "higher = more uniformly distributed points.  Default 0.10."
        )
        umap_form.addRow("Min dist:", self.umap_min_dist_sb)
        outer.addLayout(umap_form)

        # --- t-SNE row ---
        self.compute_tsne_cb = QCheckBox("Compute t-SNE")
        self.compute_tsne_cb.setToolTip(
            "Compute 2D t-SNE during Extract Features and persist as\n"
            "tSNE_1 / tSNE_2 columns in the per-patch CSV.\n"
            "Slower than UMAP (~30-60s per patch on typical sizes);\n"
            "preserves local structure best."
        )
        outer.addWidget(self.compute_tsne_cb)

        tsne_form = QFormLayout()
        tsne_form.setSpacing(4)
        tsne_form.setContentsMargins(20, 0, 0, 0)
        self.tsne_perplexity_sb = QDoubleSpinBox()
        self.tsne_perplexity_sb.setRange(2.0, 200.0)
        self.tsne_perplexity_sb.setSingleStep(1.0)
        self.tsne_perplexity_sb.setDecimals(1)
        self.tsne_perplexity_sb.setValue(30.0)
        self.tsne_perplexity_sb.setMaximumWidth(72)
        self.tsne_perplexity_sb.setToolTip(
            "t-SNE perplexity.  Roughly the number of effective near\n"
            "neighbors per point.  Capped at n_samples-1 internally.\n"
            "Typical values 5-50; default 30."
        )
        tsne_form.addRow("Perplexity:", self.tsne_perplexity_sb)
        outer.addLayout(tsne_form)

        # When UMAP is toggled, refresh n_neighbors visibility — UMAP
        # consumes the same widget as Leiden, so the row should show
        # whenever either consumer is active.
        self.compute_umap_cb.toggled.connect(
            lambda _checked: self._on_cluster_algorithm_changed(
                self.cluster_algorithm_combo.currentText(),
            ),
        )

        parent_lay.addWidget(group)

    def _build_channels_section(self, parent_lay: QVBoxLayout) -> None:
        from qtpy.QtWidgets import QGroupBox

        group = QGroupBox("Intensity channels")
        lay = QVBoxLayout(group)
        lay.setContentsMargins(8, 12, 8, 8)
        lay.setSpacing(4)
        hint = QLabel(
            "Per-channel intensity features (mean, median, std, "
            "boundary-vs-interior split) for each channel checked below.\n"
            "Uncheck all to skip per-channel intensity extraction.",
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {_TEXT_DIM}; font-size: 10px;")
        lay.addWidget(hint)

        for ch in self._available_channels:
            idx = ch.get("index")
            prefix = ch.get("prefix", f"ch{idx}")
            cb = QCheckBox(f"  Channel {idx} — {prefix}")
            cb.setChecked(False)  # opt-in per channel
            lay.addWidget(cb)
            self._channel_checkboxes[idx] = cb

        parent_lay.addWidget(group)

    # _build_feature_selection_section removed 2026-05-04 — its
    # contents (collapsible feature groups + select-all/reset)
    # were folded into _build_qc_score_section above so the
    # "which features feed qc_score" decision sits next to the
    # "how qc_score is computed" choice.

    # ------------------------------------------------------------------
    # Public API — output dict matches QCSettingsDialog plus new keys
    # ------------------------------------------------------------------

    def get_settings(self) -> dict:
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
            "cluster_algorithm":
                self.cluster_algorithm_combo.currentText(),
            "min_cluster_size": self.min_cluster_size_sb.value(),
            "n_clusters": self.n_clusters_sb.value(),
            "leiden_resolution": self.leiden_resolution_sb.value(),
            "leiden_n_neighbors": self.leiden_n_neighbors_sb.value(),
            "n_cluster_components":
                self.n_cluster_components_sb.value(),
            "metric": self.metric_combo.currentText(),
            "pca_features": self.get_selected_features(),
            "auto_cluster": self.auto_cluster_cb.isChecked(),
            # Phase A new keys (registry migration prep):
            "exclude_border": self.exclude_border_cb.isChecked(),
            "invert_z": self.invert_z_cb.isChecked(),
            "channel_configs": self.get_channel_configs(),
            # Phase A.4 keys (Nonlinear DimRed):
            "compute_umap": self.compute_umap_cb.isChecked(),
            "umap_min_dist": self.umap_min_dist_sb.value(),
            "compute_tsne": self.compute_tsne_cb.isChecked(),
            "tsne_perplexity": self.tsne_perplexity_sb.value(),
        }

    def get_selected_features(self) -> list[str]:
        out: list[str] = []
        for gw in self._group_widgets.values():
            out.extend(gw.checked_features())
        return out

    def get_channel_configs(self) -> list[dict]:
        configs: list[dict] = []
        for ch in self._available_channels:
            idx = ch.get("index")
            cb = self._channel_checkboxes.get(idx)
            if cb is not None and cb.isChecked():
                configs.append(
                    {"index": idx, "prefix": ch.get("prefix", f"ch{idx}")},
                )
        return configs

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_cluster_algorithm_changed(self, algorithm: str) -> None:
        is_hdbscan = algorithm == "hdbscan"
        is_kmeans_gmm = algorithm in ("kmeans", "gmm")
        is_leiden = algorithm == "leiden"

        self._hdbscan_label.setVisible(is_hdbscan)
        self.min_cluster_size_sb.setVisible(is_hdbscan)
        self._n_clusters_label.setVisible(is_kmeans_gmm)
        self.n_clusters_sb.setVisible(is_kmeans_gmm)
        self._resolution_label.setVisible(is_leiden)
        self.leiden_resolution_sb.setVisible(is_leiden)

        # n_neighbors is shared with UMAP (Phase A.4) — show whenever
        # Leiden OR UMAP-compute is active, since both read this value.
        umap_on = (
            getattr(self, "compute_umap_cb", None) is not None
            and self.compute_umap_cb.isChecked()
        )
        n_neighbors_visible = is_leiden or umap_on
        self._n_neighbors_label.setVisible(n_neighbors_visible)
        self.leiden_n_neighbors_sb.setVisible(n_neighbors_visible)

        # Update the description label so the user sees what each
        # algorithm does + which params it consumes.
        desc = _CLUSTER_ALGO_DESCRIPTIONS.get(algorithm, "")
        if hasattr(self, "_cluster_algo_desc"):
            self._cluster_algo_desc.setText(desc)

    def _on_composite_method_changed(self, method: str) -> None:
        """Swap the QC Score description label as the method changes."""
        text = _COMPOSITE_METHOD_DESCRIPTIONS.get(method, "")
        if hasattr(self, "_qc_method_desc"):
            self._qc_method_desc.setText(text)

    def _select_all_features(self) -> None:
        for gw in self._group_widgets.values():
            feats = (
                set(gw._checkboxes.keys())
                if hasattr(gw, "_checkboxes") else set()
            )
            gw.set_pca_features(feats)
        self._refresh_selection_count()

    def _reset_default_features(self) -> None:
        defaults = set(PCA_FEATURES)
        for gw in self._group_widgets.values():
            gw.set_pca_features(defaults)
        self._refresh_selection_count()

    def _refresh_selection_count(self) -> None:
        n = sum(
            len(gw.checked_features())
            for gw in self._group_widgets.values()
        )
        self._selection_label.setText(f"{n} features selected for qc_score")

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
            idx = self.composite_method_combo.findText(
                settings["composite_method"],
            )
            if idx >= 0:
                self.composite_method_combo.setCurrentIndex(idx)
        if "cluster_algorithm" in settings:
            idx = self.cluster_algorithm_combo.findText(
                settings["cluster_algorithm"],
            )
            if idx >= 0:
                self.cluster_algorithm_combo.setCurrentIndex(idx)
        if "min_cluster_size" in settings:
            self.min_cluster_size_sb.setValue(
                settings["min_cluster_size"],
            )
        if "n_clusters" in settings:
            self.n_clusters_sb.setValue(settings["n_clusters"])
        if "leiden_resolution" in settings:
            self.leiden_resolution_sb.setValue(
                settings["leiden_resolution"],
            )
        if "leiden_n_neighbors" in settings:
            self.leiden_n_neighbors_sb.setValue(
                settings["leiden_n_neighbors"],
            )
        if "n_cluster_components" in settings:
            self.n_cluster_components_sb.setValue(
                settings["n_cluster_components"],
            )
        if "metric" in settings:
            idx = self.metric_combo.findText(settings["metric"])
            if idx >= 0:
                self.metric_combo.setCurrentIndex(idx)
        if "pca_features" in settings:
            sel = set(settings["pca_features"])
            for gw in self._group_widgets.values():
                gw.set_pca_features(sel)
            self._refresh_selection_count()
        if "exclude_border" in settings:
            self.exclude_border_cb.setChecked(
                bool(settings["exclude_border"]),
            )
        if "invert_z" in settings:
            self.invert_z_cb.setChecked(bool(settings["invert_z"]))
        if "channel_configs" in settings:
            checked = {c.get("index") for c in settings["channel_configs"]}
            for idx, cb in self._channel_checkboxes.items():
                cb.setChecked(idx in checked)
        if "auto_cluster" in settings:
            self.auto_cluster_cb.setChecked(bool(settings["auto_cluster"]))
        # Phase A.4: nonlinear DimRed knobs
        if "compute_umap" in settings:
            self.compute_umap_cb.setChecked(bool(settings["compute_umap"]))
        if "umap_min_dist" in settings:
            self.umap_min_dist_sb.setValue(float(settings["umap_min_dist"]))
        if "compute_tsne" in settings:
            self.compute_tsne_cb.setChecked(bool(settings["compute_tsne"]))
        if "tsne_perplexity" in settings:
            self.tsne_perplexity_sb.setValue(
                float(settings["tsne_perplexity"]),
            )
        # n_neighbors visibility may have changed if compute_umap came in
        # as True — rerun the algorithm-changed handler to refresh.
        self._on_cluster_algorithm_changed(
            self.cluster_algorithm_combo.currentText(),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FlatCBWrapper:
    """Adapter so the legacy fallback path matches _FeatureGroupWidget's API."""

    def __init__(self, cb: QCheckBox, feat: str) -> None:
        self._cb = cb
        self._feat = feat
        self._checkboxes = {feat: cb}

    def checked_features(self) -> list[str]:
        return [self._feat] if self._cb.isChecked() else []

    def set_pca_features(self, pca_set: set[str]) -> None:
        self._cb.blockSignals(True)
        self._cb.setChecked(self._feat in pca_set)
        self._cb.blockSignals(False)

    @property
    def category(self) -> str:
        return self._feat


def _row_widget(row_layout: QHBoxLayout) -> QWidget:
    """Wrap a layout in a QWidget so QFormLayout.addRow accepts it."""
    w = QWidget()
    w.setLayout(row_layout)
    return w
