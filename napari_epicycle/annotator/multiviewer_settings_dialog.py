"""Multiviewer Settings dialog for EpiCycle multiview annotator.

Non-modal floating dock that houses display settings for the multiview
annotator: pan/zoom sync, crosshair visibility, style, and color.

Extracted from the Advanced Features widget (K13.3) so that configuration
controls live separately from segmentation tools.
"""

from __future__ import annotations

from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QVBoxLayout,
    QWidget,
)


class MultiviewerSettingsWidget(QWidget):
    """Lightweight widget for multiviewer display options.

    Intended to be docked as a small floating panel via the launcher.
    Exposes the same public attribute names as the former
    ``_build_viewer_options_section()`` in AdvancedFeaturesWidget so that
    future wiring code is unchanged.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.sync_pan_zoom_cb = QCheckBox("Sync pan/zoom across views")
        layout.addWidget(self.sync_pan_zoom_cb)

        self.show_crosshairs_cb = QCheckBox("Show crosshairs")
        self.show_crosshairs_cb.setChecked(True)
        layout.addWidget(self.show_crosshairs_cb)

        form = QFormLayout()
        form.setVerticalSpacing(2)
        self.crosshair_style_combo = QComboBox()
        self.crosshair_style_combo.addItems(["Solid", "Dashed", "Dotted", "None"])
        self.crosshair_color_combo = QComboBox()
        self.crosshair_color_combo.addItems(
            ["Auto", "White", "Yellow", "Cyan", "Magenta"]
        )
        form.addRow("Crosshair style:", self.crosshair_style_combo)
        form.addRow("Crosshair color:", self.crosshair_color_combo)
        layout.addLayout(form)

        layout.addStretch(1)
