"""Extract Features dialog with full feature transparency and PCA preset selection.

Shows all feature groups that will be computed, lets the user choose a PCA
preset (or customize per-feature), and carries forward the existing options
(skip probability, remove border cells).  Progressive disclosure: groups are
collapsed by default, expand on click to show individual features.
"""

from __future__ import annotations

from typing import Optional

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------------------
# Styling constants (match analysis_widget / existing dialog)
# ---------------------------------------------------------------------------

_BG = "#2b2b2b"
_BG_CONTENT = "#1e1e2e"
_TEXT = "#e0e0e0"
_TEXT_DIM = "#999999"
_TEXT_DIMMER = "#777777"
_ACCENT = "#6abf69"
_ACCENT_BG = "rgba(106,191,105,0.25)"
_ACCENT_BORDER = "rgba(106,191,105,0.40)"
_CANCEL_BG = "rgba(255,255,255,0.05)"
_CANCEL_BORDER = "rgba(255,255,255,0.10)"
_HEADER_BG = "#333348"
_GROUP_BG = "rgba(255,255,255,0.03)"
_GROUP_HOVER = "rgba(255,255,255,0.06)"
_SEPARATOR = "rgba(255,255,255,0.08)"
_WARNING = "#ff9800"
_INFO = "#80deea"


# ---------------------------------------------------------------------------
# Collapsible feature group widget
# ---------------------------------------------------------------------------

class _FeatureGroupWidget(QWidget):
    """A single collapsible feature group with per-feature PCA checkboxes."""

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

        # --- Header row (clickable) ---
        self._header = QWidget(self)
        self._header.setCursor(Qt.PointingHandCursor)
        self._header.setStyleSheet(
            f"QWidget {{ background: {_GROUP_BG}; border-radius: 3px; }}"
            f"QWidget:hover {{ background: {_GROUP_HOVER}; }}"
        )
        header_lay = QHBoxLayout(self._header)
        header_lay.setContentsMargins(8, 5, 8, 5)
        header_lay.setSpacing(6)

        self._arrow = QLabel("\u25b8", self._header)  # ▸
        self._arrow.setFixedWidth(12)
        self._arrow.setStyleSheet(f"color: {_TEXT_DIM}; font-size: 11px;")
        header_lay.addWidget(self._arrow)

        self._name_label = QLabel(
            f"{category} ({len(features)})", self._header
        )
        self._name_label.setStyleSheet(
            f"color: {_TEXT}; font-size: 12px; font-weight: bold;"
        )
        header_lay.addWidget(self._name_label)

        header_lay.addStretch()

        self._pca_label = QLabel(self._header)
        self._pca_label.setStyleSheet(f"font-size: 11px;")
        header_lay.addWidget(self._pca_label)

        # Click handler on whole header
        self._header.mousePressEvent = self._toggle_expand

        layout.addWidget(self._header)

        # --- Feature list (hidden by default) ---
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
            cb.setStyleSheet(self._checkbox_style(feat in pca_set))
            cb.stateChanged.connect(self._on_checkbox_changed)
            feat_lay.addWidget(cb)
            self._checkboxes[feat] = cb

        layout.addWidget(self._feature_container)
        self._update_pca_label()

    # --- Public API ---

    def checked_features(self) -> list[str]:
        """Return list of features currently checked for PCA."""
        return [f for f, cb in self._checkboxes.items() if cb.isChecked()]

    def set_pca_features(self, pca_set: set[str]):
        """Update checkboxes to match a new PCA feature set."""
        for feat, cb in self._checkboxes.items():
            cb.blockSignals(True)
            cb.setChecked(feat in pca_set)
            cb.setStyleSheet(self._checkbox_style(feat in pca_set))
            cb.blockSignals(False)
        self._update_pca_label()

    @property
    def category(self) -> str:
        return self._category

    # --- Internals ---

    def _toggle_expand(self, event=None):
        self._expanded = not self._expanded
        self._feature_container.setVisible(self._expanded)
        self._arrow.setText("\u25be" if self._expanded else "\u25b8")  # ▾ / ▸

    def _on_checkbox_changed(self):
        sender = self.sender()
        if sender:
            feat = sender.text()
            sender.setStyleSheet(self._checkbox_style(sender.isChecked()))
        self._update_pca_label()
        self.selection_changed.emit()

    def _update_pca_label(self):
        n_checked = sum(1 for cb in self._checkboxes.values() if cb.isChecked())
        if n_checked > 0:
            self._pca_label.setText(f"\u25cf {n_checked} in PCA")
            self._pca_label.setStyleSheet(f"color: {_ACCENT}; font-size: 11px;")
        else:
            self._pca_label.setText(f"\u25cb 0 in PCA")
            self._pca_label.setStyleSheet(f"color: {_TEXT_DIMMER}; font-size: 11px;")

    @staticmethod
    def _checkbox_style(checked: bool) -> str:
        color = _TEXT if checked else _TEXT_DIM
        return f"QCheckBox {{ color: {color}; font-size: 11px; }}"


# ---------------------------------------------------------------------------
# Main dialog
# ---------------------------------------------------------------------------

class ExtractFeaturesDialog(QDialog):
    """Feature extraction dialog with full transparency and PCA preset selection.

    Parameters
    ----------
    parent : QWidget
        Parent widget (usually the Step 4 UI panel).
    feature_manifest : dict[str, list[str]]
        ``{category: [feature_names]}`` from ``get_feature_manifest()``.
    n_labels : int
        Number of labels in the current volume.
    was_smoothed : bool
        Whether the current stack was smoothed (triggers defaults).
    has_saved_border_cells : bool
        Whether border cells were previously removed and saved.
    pca_presets : dict
        ``PCA_PRESETS`` from ``qc_features.py``.
    descriptions : dict[str, str]
        Feature name → tooltip description mapping.
    """

    def __init__(
        self,
        parent: QWidget,
        *,
        feature_manifest: dict[str, list[str]],
        n_labels: int,
        was_smoothed: bool = False,
        has_saved_border_cells: bool = False,
        pca_presets: dict | None = None,
        descriptions: dict[str, str] | None = None,
    ):
        super().__init__(parent)
        self._manifest = feature_manifest
        self._pca_presets = pca_presets or {}
        self._descriptions = descriptions or {}
        self._group_widgets: dict[str, _FeatureGroupWidget] = {}
        self._base_preset_name: str | None = None

        self.setWindowTitle("Extract Features")
        self.setMinimumWidth(480)
        self.setMaximumHeight(int(self.screen().geometry().height() * 0.7))
        self.setStyleSheet(
            f"QDialog {{ background: {_BG}; "
            f"border: 1px solid rgba(255,255,255,0.12); border-radius: 6px; }}"
            f"QLabel {{ color: {_TEXT}; font-size: 12px; }}"
        )

        main_lay = QVBoxLayout(self)
        main_lay.setContentsMargins(20, 16, 20, 16)
        main_lay.setSpacing(10)

        # --- Title ---
        total_features = sum(len(f) for f in feature_manifest.values())
        n_groups = len(feature_manifest)

        title = QLabel("Extract Features")
        title.setStyleSheet(
            f"font-weight: bold; font-size: 14px; color: #eeeeee;"
        )
        main_lay.addWidget(title)

        desc = QLabel(
            f"Computing {total_features} features across {n_groups} groups "
            f"for {n_labels} labels.\n"
            f"All features will be extracted. Select which to include in "
            f"the initial PCA."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: #cccccc; font-size: 11px;")
        main_lay.addWidget(desc)

        # --- PCA Preset Dropdown ---
        preset_row = QHBoxLayout()
        preset_row.setSpacing(8)
        preset_label = QLabel("PCA Preset:")
        preset_label.setStyleSheet(
            f"color: {_TEXT}; font-size: 12px; font-weight: bold;"
        )
        preset_row.addWidget(preset_label)

        self._preset_combo = QComboBox(self)
        for name, info in self._pca_presets.items():
            n_feat = (
                len(info["features"]) if info["features"] else
                ("all" if info["features"] is None else "0")
            )
            self._preset_combo.addItem(f"{name} ({n_feat} features)", name)
        self._preset_combo.setStyleSheet(
            f"QComboBox {{ background: {_BG_CONTENT}; color: {_TEXT}; "
            f"border: 1px solid rgba(255,255,255,0.12); border-radius: 3px; "
            f"padding: 4px 8px; font-size: 11px; }}"
            f"QComboBox::drop-down {{ border: none; }}"
            f"QComboBox QAbstractItemView {{ background: {_BG}; color: {_TEXT}; "
            f"selection-background-color: {_HEADER_BG}; font-size: 11px; }}"
        )
        self._preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        preset_row.addWidget(self._preset_combo, 1)
        main_lay.addLayout(preset_row)

        # Preset description
        self._preset_desc = QLabel()
        self._preset_desc.setWordWrap(True)
        self._preset_desc.setStyleSheet(
            f"color: {_TEXT_DIM}; font-size: 10px; font-style: italic;"
        )
        main_lay.addWidget(self._preset_desc)

        # --- Selection counter ---
        self._selection_label = QLabel()
        self._selection_label.setStyleSheet(
            f"color: {_ACCENT}; font-size: 11px; font-weight: bold;"
        )
        main_lay.addWidget(self._selection_label)

        # --- Scrollable feature groups ---
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(
            f"QScrollArea {{ background: transparent; border: none; }}"
            f"QScrollBar:vertical {{ background: {_BG}; width: 8px; "
            f"border-radius: 4px; }}"
            f"QScrollBar::handle:vertical {{ background: rgba(255,255,255,0.12); "
            f"min-height: 20px; border-radius: 4px; }}"
            f"QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical "
            f"{{ height: 0px; }}"
        )

        scroll_content = QWidget()
        scroll_lay = QVBoxLayout(scroll_content)
        scroll_lay.setContentsMargins(0, 0, 0, 0)
        scroll_lay.setSpacing(3)

        # Build group widgets
        all_features_flat = set()
        channel_separator_added = False

        for cat, feats in feature_manifest.items():
            # Add separator before channel groups
            if cat.startswith("Channel:") and not channel_separator_added:
                sep = self._make_separator("Intensity Channels")
                scroll_lay.addWidget(sep)
                channel_separator_added = True

            all_features_flat.update(feats)
            group_w = _FeatureGroupWidget(
                cat, feats,
                pca_set=set(),  # will be set by preset
                descriptions=self._descriptions,
                parent=scroll_content,
            )
            group_w.selection_changed.connect(self._on_selection_changed)
            scroll_lay.addWidget(group_w)
            self._group_widgets[cat] = group_w

        scroll_lay.addStretch()
        scroll.setWidget(scroll_content)
        main_lay.addWidget(scroll, 1)

        self._all_features = all_features_flat

        # --- Options section ---
        options_sep = self._make_separator("Options")
        main_lay.addWidget(options_sep)

        self._skip_prob_cb = QCheckBox(
            "Skip probability features (recommended after smoothing)"
        )
        self._skip_prob_cb.setChecked(was_smoothed)
        self._skip_prob_cb.setStyleSheet(f"color: {_TEXT}; font-size: 11px;")
        self._skip_prob_cb.stateChanged.connect(self._on_skip_prob_changed)
        main_lay.addWidget(self._skip_prob_cb)

        self._remove_border_cb = QCheckBox(
            f"Remove border cells from final results"
        )
        self._remove_border_cb.setChecked(was_smoothed)
        self._remove_border_cb.setStyleSheet(f"color: {_TEXT}; font-size: 11px;")
        main_lay.addWidget(self._remove_border_cb)

        # Conditional notes
        if was_smoothed:
            note = QLabel(
                "Smooth detected \u2014 probability features are misaligned "
                "with the smoothed volume and should be skipped."
            )
            note.setWordWrap(True)
            note.setStyleSheet(
                f"color: {_WARNING}; font-size: 10px; font-style: italic;"
            )
            main_lay.addWidget(note)

        if has_saved_border_cells:
            border_note = QLabel(
                "Border cells will be temporarily restored for neighbor "
                "computation, then removed from the final DataFrame."
            )
            border_note.setWordWrap(True)
            border_note.setStyleSheet(
                f"color: {_INFO}; font-size: 10px; font-style: italic;"
            )
            main_lay.addWidget(border_note)

        # --- Buttons ---
        btn_row = QDialogButtonBox()
        cancel_btn = btn_row.addButton("Cancel", QDialogButtonBox.RejectRole)
        cancel_btn.setStyleSheet(
            f"QPushButton {{ background: {_CANCEL_BG}; "
            f"border: 1px solid {_CANCEL_BORDER}; border-radius: 4px; "
            f"color: #bbbbbb; padding: 6px 18px; min-width: 80px; }}"
        )
        extract_btn = btn_row.addButton(
            "Extract Features", QDialogButtonBox.AcceptRole
        )
        extract_btn.setStyleSheet(
            f"QPushButton {{ background: {_ACCENT_BG}; "
            f"border: 1px solid {_ACCENT_BORDER}; border-radius: 4px; "
            f"color: #eeeeee; font-weight: bold; padding: 6px 18px; "
            f"min-width: 100px; }}"
        )
        btn_row.accepted.connect(self.accept)
        btn_row.rejected.connect(self.reject)
        main_lay.addWidget(btn_row)

        # --- Apply initial preset ---
        self._apply_preset_by_index(0)

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def selected_pca_features(self) -> list[str]:
        """Return list of feature names currently checked for PCA."""
        result = []
        for gw in self._group_widgets.values():
            result.extend(gw.checked_features())
        return result

    def selected_preset_name(self) -> str:
        """Return current preset name, or 'Custom (based on ...)'."""
        if self._is_selection_matching_preset():
            return self._base_preset_name
        if self._base_preset_name:
            return f"Custom (based on {self._base_preset_name})"
        return "Custom"

    def skip_probability(self) -> bool:
        return self._skip_prob_cb.isChecked()

    def remove_border(self) -> bool:
        return self._remove_border_cb.isChecked()

    def pca_mode(self) -> str | None:
        """Return the PCA mode for the selected preset ('qc', 'eda', or None)."""
        idx = self._preset_combo.currentIndex()
        base_name = self._preset_combo.itemData(idx)
        preset = self._pca_presets.get(base_name, {})
        return preset.get("mode")

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    def _on_preset_changed(self, index: int):
        self._apply_preset_by_index(index)

    def _apply_preset_by_index(self, index: int):
        """Apply a named preset: check/uncheck features accordingly."""
        base_name = self._preset_combo.itemData(index)
        self._base_preset_name = base_name
        preset = self._pca_presets.get(base_name, {})

        features = preset.get("features")
        desc = preset.get("description", "")
        self._preset_desc.setText(desc)

        # Resolve feature set
        if features is None:
            # "Standard" — all numeric features (excluding position, orientation vectors)
            pca_set = set(self._all_features)
            # Exclude position and raw orientation components
            _exclude = {
                "centroid-0", "centroid-1", "centroid-2",
                "major_axis_z", "major_axis_y", "major_axis_x",
                "mid_axis_z", "mid_axis_y", "mid_axis_x",
                "minor_axis_z", "minor_axis_y", "minor_axis_x",
                "inertia_eigval_major", "inertia_eigval_mid",
                "inertia_eigval_minor",
            }
            pca_set -= _exclude
        elif features:
            pca_set = set(features)
        else:
            pca_set = set()

        for gw in self._group_widgets.values():
            gw.set_pca_features(pca_set)

        self._update_selection_count()

    def _on_selection_changed(self):
        """Called when any individual feature checkbox changes."""
        self._update_selection_count()
        self._update_preset_label()

    def _update_selection_count(self):
        n = len(self.selected_pca_features())
        self._selection_label.setText(f"{n} features selected for PCA")

    def _is_selection_matching_preset(self) -> bool:
        """Check if current selection matches the active preset exactly."""
        preset = self._pca_presets.get(self._base_preset_name, {})
        preset_features = preset.get("features")
        current = set(self.selected_pca_features())

        if preset_features is None:
            # "Standard" — all numeric minus exclusions
            _exclude = {
                "centroid-0", "centroid-1", "centroid-2",
                "major_axis_z", "major_axis_y", "major_axis_x",
                "mid_axis_z", "mid_axis_y", "mid_axis_x",
                "minor_axis_z", "minor_axis_y", "minor_axis_x",
                "inertia_eigval_major", "inertia_eigval_mid",
                "inertia_eigval_minor",
            }
            expected = self._all_features - _exclude
            return current == expected
        else:
            # Named preset — compare against its feature list
            # Only compare features that actually exist in the manifest
            expected = set(preset_features) & self._all_features
            return current == expected

    def _update_preset_label(self):
        """Update the preset description to show Custom if diverged."""
        if self._is_selection_matching_preset():
            preset = self._pca_presets.get(self._base_preset_name, {})
            self._preset_desc.setText(preset.get("description", ""))
        else:
            n = len(self.selected_pca_features())
            self._preset_desc.setText(
                f"Custom selection based on {self._base_preset_name} "
                f"({n} features)"
            )

    def _on_skip_prob_changed(self, state):
        """Dim the Probability group when skip is checked."""
        prob_group = self._group_widgets.get("Probability")
        if prob_group:
            prob_group.setEnabled(state != Qt.Checked)
            prob_group.setStyleSheet(
                f"color: {_TEXT_DIMMER};" if state == Qt.Checked else ""
            )

    @staticmethod
    def _make_separator(text: str) -> QWidget:
        """Create a styled separator label with a line."""
        container = QWidget()
        lay = QHBoxLayout(container)
        lay.setContentsMargins(0, 6, 0, 2)
        lay.setSpacing(8)

        line_left = QFrame()
        line_left.setFrameShape(QFrame.HLine)
        line_left.setStyleSheet(f"color: {_SEPARATOR};")
        line_left.setFixedHeight(1)
        lay.addWidget(line_left, 1)

        label = QLabel(text)
        label.setStyleSheet(
            f"color: {_TEXT_DIM}; font-size: 10px; font-weight: bold; "
            f"letter-spacing: 1px;"
        )
        lay.addWidget(label)

        line_right = QFrame()
        line_right.setFrameShape(QFrame.HLine)
        line_right.setStyleSheet(f"color: {_SEPARATOR};")
        line_right.setFixedHeight(1)
        lay.addWidget(line_right, 1)

        return container
