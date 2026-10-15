"""Per-orientation extraction configuration dialog.

Modal popup that lets the user configure lateral (XY) and axial (XZ + YZ)
extraction parameters INDEPENDENTLY.  Reads / writes the
``refinement_extraction_config`` field in state_manager so the values
persist across sessions and inform every subsequent Extract Training
Pairs run.

Shipped 2026-05-19 as part of feature/refinement-extraction-redesign
sub-commit (2).  Replaces the legacy "Views: All / Lateral / Axial"
combo with a richer per-orientation knob layout — extraction always
populates both spatial/ and axial/ subdirs, and the dialog controls
HOW each orientation is processed.

Usage:

    initial = state_manager.get_refinement_extraction_config()
    scalar_defaults = controller._read_extraction_settings()
    cfg = ExtractionConfigDialog.show_dialog(
        parent, initial_config=initial,
        scalar_defaults=scalar_defaults,
    )
    if cfg is not None:
        state_manager.set_refinement_extraction_config(
            lateral=cfg["lateral"], axial=cfg["axial"],
        )

Returns None when the user clicks Cancel.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


# 2026-05-19 sub-commit 6: style tokens copied verbatim from
# qc_extraction_dialog.py so the two settings-style dialogs render
# IDENTICALLY (dark bg, subtle group borders, darker spinbox fills,
# amber primary button).  Kept local to the dialog file rather than
# importing the qc_extraction_dialog's private constants — this
# dialog should own its style copy, not couple to another dialog.
_BG = "#2b2b2b"
_BG_CONTENT = "#1e1e2e"
_TEXT = "#e0e0e0"
_TEXT_DIM = "#999999"
_ACCENT = "#d4a843"
_ACCENT_BG = "rgba(212,168,67,0.20)"
_ACCENT_BORDER = "rgba(212,168,67,0.40)"
_HEADER_BG = "#333348"

_DIALOG_STYLESHEET = (
    # Dialog window itself — dark bg + subtle border + rounded.
    f"QDialog {{ background: {_BG}; border: 1px solid "
    f"rgba(255,255,255,0.12); border-radius: 6px; }}"
    f"QLabel {{ color: {_TEXT}; font-size: 11px; }}"
    f"QSpinBox, QDoubleSpinBox, QComboBox {{ "
    f"background: {_BG_CONTENT}; color: {_TEXT}; "
    f"border: 1px solid rgba(255,255,255,0.12); "
    f"border-radius: 3px; padding: 2px 6px; font-size: 11px; }}"
    f"QCheckBox {{ color: {_TEXT}; font-size: 11px; }}"
)


# Inline styles for the per-column structure (2026-05-19 sub-commit 8).
# No QGroupBox at any level — just a bold amber column header, a
# dim section header, content rows, a subtle horizontal line, then
# the next section header + content.
_COLUMN_HEADER_STYLE = (
    f"color: {_ACCENT}; font-size: 13px; font-weight: bold; "
    f"padding-bottom: 4px;"
)
_SECTION_HEADER_STYLE = (
    f"color: {_TEXT_DIM}; font-size: 10px; font-weight: bold; "
    f"text-transform: uppercase; letter-spacing: 0.5px; "
    f"padding-top: 4px; padding-bottom: 2px;"
)
_HLINE_STYLE = (
    "background-color: rgba(255,255,255,0.10); border: none;"
)

_BTN_CONFIRM_STYLE = (
    f"QPushButton {{ background: {_ACCENT_BG}; color: {_TEXT}; "
    f"border: 1px solid {_ACCENT_BORDER}; border-radius: 3px; "
    f"padding: 4px 14px; font-weight: bold; }}"
    f"QPushButton:hover {{ background: {_ACCENT}; color: black; }}"
)

_BTN_CANCEL_STYLE = (
    f"QPushButton {{ background: rgba(255,255,255,0.05); "
    f"color: {_TEXT_DIM}; border: 1px solid rgba(255,255,255,0.10); "
    f"border-radius: 3px; padding: 4px 14px; }}"
    f"QPushButton:hover {{ background: rgba(255,255,255,0.10); "
    f"color: {_TEXT}; }}"
)


# Keys that the dialog reads/writes per orientation, in display order.
# Must match the 8 keys consumed by extract_training_pairs_from_3d
# (and persisted by state_manager).
_SAMPLING_KEYS = (
    "min_labels_per_slice",
    "edge_buffer",
    "slices_per_patch",
    "min_distance",
)
_CLEANING_KEYS = (
    "remove_boundaries",
    "relabel",
    "min_size",
    "smoothing_disk_size",
    "expand_labels_distance",
)

_LABELS = {
    "min_labels_per_slice": "Min labels per slice",
    "edge_buffer": "Edge buffer",
    "slices_per_patch": "Slices per patch (0 = all)",
    "min_distance": "Min distance",
    "remove_boundaries": "Remove boundaries",
    "relabel": "Relabel components",
    "min_size": "Min region size (px)",
    "smoothing_disk_size": "Smoothing disk size",
    "expand_labels_distance": "Expand labels (px)",
}

_TOOLTIPS = {
    "min_labels_per_slice": (
        "Skip 2D slices that contain fewer than this many unique cell "
        "labels.\nKeeps the training set focused on slices with rich "
        "annotation.\nDefault: 3."
    ),
    "edge_buffer": (
        "Skip slices within this many voxels of the volume edge along "
        "the relevant axis.\nAvoids partial cells clipped by the patch "
        "boundary.\nDefault: 50."
    ),
    "slices_per_patch": (
        "Maximum slices to extract per patch (per view).  0 means "
        "extract every valid slice.\nWhen > 0, picks are sampled "
        "uniformly across the full stack — never clustered at one "
        "end.\nDefault: 0 (no cap).\n"
        "Note: the cap is PER-SUB-VIEW.  Axial = XZ + YZ, so axial "
        "cap of N yields up to 2N slices per patch (N from each "
        "sub-view).  Lateral has one view, so N is exact."
    ),
    "min_distance": (
        "Minimum spacing (in slices) between two extracted slices in "
        "the same view.\nApplied as a soft floor after uniform "
        "sampling — only matters if the cap is very high.\nDefault: 5."
    ),
    "remove_boundaries": (
        "Erode 1 px from the perimeter of every label to separate "
        "touching cells.\nIMPORTANT: this is mostly useful for first-"
        "round extractions from raw Cellpose output.\nFor iterative "
        "refinement, consider turning this OFF — repeated boundary "
        "removal will shrink cells across rounds."
    ),
    "relabel": (
        "Reassign connected-component IDs after boundary removal so "
        "split fragments get unique labels.\nUsually paired with "
        "remove_boundaries=True.  Off by itself is rarely useful."
    ),
    "min_size": (
        "Drop any cell region smaller than this many pixels in 2D.\n"
        "Removes tiny fragments created by boundary erosion + cell "
        "edges clipped by the slice plane.\nDefault: 100."
    ),
    "smoothing_disk_size": (
        "Morphological opening + closing per-cell with a disk of this "
        "radius (in pixels).\n0 disables smoothing.  Default: 1 "
        "(3×3 kernel — cleans NN-upsample stair-steps from downsampled "
        "curation).\nLarger values will erode thin features."
    ),
    "expand_labels_distance": (
        "Expand each label outward by this many pixels AFTER the "
        "smoothing step.\n0 disables expansion (legacy behavior).\n"
        "Uses skimage.segmentation.expand_labels — labels stop "
        "growing when they meet another label, so cells don't merge.\n"
        "\nWHY: \"Remove boundaries\" erodes 1 px off every cell to "
        "separate touching cells.  Trained models then learn under-"
        "segmentation as correct, and masks shrink across refinement "
        "rounds.  Setting this to 1 adds that 1 px back AFTER "
        "relabeling — cells stay separated but recover original size.\n"
        "\nRecommended values:\n"
        "  • 1 if remove_boundaries=True and smoothing_disk_size=0 "
        "(exactly compensates the 1 px erosion).\n"
        "  • 2 if remove_boundaries=True and smoothing_disk_size≥1 "
        "(also compensates for smoothing-induced shrinkage of a few "
        "more pixels).\n"
        "  • 3+ to make cells larger than the original annotation "
        "(grows into background, never into neighbors)."
    ),
}


# Hard-coded ranges (mirrors the spinbox configs in fusion_reconstruction/ui.py).
_RANGES = {
    "min_labels_per_slice": (0, 50),
    "edge_buffer": (0, 200),
    "slices_per_patch": (0, 500),
    "min_distance": (1, 50),
    "min_size": (0, 10000),
    "smoothing_disk_size": (0, 10),
    "expand_labels_distance": (0, 10),
}


# Differentiated per-orientation defaults (2026-05-19 sub-commit 4).
# Lateral cells (XY top-down) and axial cells (XZ + YZ cross-section)
# have genuinely different geometries in 2D — these defaults reflect
# that.  Tune by editing here; the controller imports both constants
# and uses them as the fallback when state_manager has no overrides.
#
# Lateral: XY slices sliced along Z.  Cell appearance ~ circular blobs.
#   - edge_buffer SMALLER because Z is typically small (~80 slices)
#   - min_distance SMALLER because Z is dense in real µm
#   - min_size LARGER because XY area of a cell is bigger
LATERAL_DEFAULT_CONFIG = {
    "min_labels_per_slice": 3,
    "edge_buffer": 5,
    "slices_per_patch": 0,        # 0 = all valid slices
    "min_distance": 3,
    "remove_boundaries": True,
    "relabel": True,
    "min_size": 100,
    "smoothing_disk_size": 1,
    # 2026-05-21: default 0 (legacy behavior).  User opts in via
    # the per-orientation extraction dialog.  Setting to 1 cancels
    # out remove_boundaries=True's 1 px erosion bias — see
    # `clean_mask_slice` docstring for the why.
    "expand_labels_distance": 0,
}

# Axial: XZ + YZ slices sliced along Y or X.  Cell appearance ~ tall
# narrow cross-sections.
#   - edge_buffer LARGER because Y/X are typically big (~1000 px)
#   - min_distance LARGER because in-plane pixels are independent
#   - min_size SMALLER because cross-section area is smaller
#   - min_labels_per_slice LOWER because fewer cells per cross-section
AXIAL_DEFAULT_CONFIG = {
    "min_labels_per_slice": 2,
    "edge_buffer": 20,
    "slices_per_patch": 0,
    "min_distance": 5,
    "remove_boundaries": True,
    "relabel": True,
    "min_size": 50,
    "smoothing_disk_size": 1,
    "expand_labels_distance": 0,  # see lateral comment
}


def get_default_config_for(orientation: str) -> dict:
    """Return a fresh copy of the default config for the orientation.

    Used by both the dialog (for the Reset button + initial population)
    and the controller (as fallback when state_manager has no override).
    """
    if orientation == "lateral":
        return dict(LATERAL_DEFAULT_CONFIG)
    if orientation == "axial":
        return dict(AXIAL_DEFAULT_CONFIG)
    raise ValueError(
        f"Invalid orientation {orientation!r}; expected 'lateral' or 'axial'."
    )


class ExtractionConfigDialog(QDialog):
    """Modal dialog for per-orientation extraction parameter configuration.

    Two columns (Lateral / Axial) × two group boxes (Slice Sampling /
    Mask Preprocessing) × 4 controls each = 16 widgets per column.
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        initial_config: Optional[Dict[str, Dict[str, Any]]] = None,
        scalar_defaults: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Configure per-orientation extraction")
        self.setMinimumWidth(720)
        self.setSizeGripEnabled(True)

        # 2026-05-19 sub-commit 6: visual match to the Curation
        # Settings dialog (qc_extraction_dialog.py) — single
        # combined stylesheet, flat group hierarchy (no outer
        # column-wrapping QGroupBox).
        self.setStyleSheet(_DIALOG_STYLESHEET)

        self._scalar_defaults = dict(scalar_defaults or {})
        self._initial_config = dict(initial_config or {})

        # ── Widgets are stored as nested dicts:
        #    self._widgets[orientation][key] = QSpinBox / QCheckBox
        self._widgets: Dict[str, Dict[str, QWidget]] = {
            "lateral": {},
            "axial": {},
        }

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 12, 16, 12)
        outer.setSpacing(10)

        # ── Title (matches QC Settings dialog pattern) ───────────────
        title = QLabel("Per-Orientation Extraction Configuration")
        title.setStyleSheet(
            "color: #eeeeee; font-size: 14px; font-weight: bold;"
        )
        outer.addWidget(title)

        # ── Top description ──────────────────────────────────────────
        desc = QLabel(
            "Lateral (XY top-down) and axial (XZ + YZ cross-section) "
            "cells often look different — and benefit from different "
            "extraction tuning.  Set per-orientation overrides here; "
            "Reset to defaults restores the orientation-specific "
            "defaults.  Settings persist per workspace."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {_TEXT_DIM}; font-size: 10px;")
        outer.addWidget(desc)

        # ── Two columns: Lateral | Axial ─────────────────────────────
        #
        # 2026-05-19 sub-commit 8: no QGroupBox wrappers at all.
        # Each column = amber column header + section header +
        # content + subtle horizontal line + section header +
        # content.  Mirrors the "subtle horizontal line between
        # sections" pattern of the Curation Settings dialog.
        grid = QGridLayout()
        grid.setHorizontalSpacing(24)
        grid.setVerticalSpacing(10)

        grid.addWidget(self._build_orientation_column("lateral"), 0, 0)
        grid.addWidget(self._build_orientation_column("axial"), 0, 1)
        outer.addLayout(grid)

        # ── Button row ───────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self._reset_btn = QPushButton("Reset to defaults")
        self._reset_btn.setStyleSheet(_BTN_CANCEL_STYLE)
        self._reset_btn.setToolTip(
            "Restore every field in both columns to the orientation-"
            "specific default.  Does not save until you click Save."
        )
        self._reset_btn.clicked.connect(self._on_reset)
        btn_row.addWidget(self._reset_btn)

        btn_row.addStretch(1)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setStyleSheet(_BTN_CANCEL_STYLE)
        self._cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self._cancel_btn)

        self._save_btn = QPushButton("Save")
        self._save_btn.setStyleSheet(_BTN_CONFIRM_STYLE)
        self._save_btn.setDefault(True)
        self._save_btn.clicked.connect(self.accept)
        btn_row.addWidget(self._save_btn)

        outer.addLayout(btn_row)

        # Populate widget values from initial_config + scalar_defaults.
        self._populate_widgets_from_initial_state()

    # ----------------------------------------------------------------
    # Column builders
    # ----------------------------------------------------------------

    def _build_orientation_column(self, orientation: str) -> QWidget:
        """Build a single orientation's column.

        2026-05-19 sub-commit 8: no QGroupBox at any level.  Layout
        is:
          1. Bold amber column header label ("Lateral (XY)" or
             "Axial (XZ + YZ)")
          2. Dim "SLICE SAMPLING" section header
          3. Form rows for the 4 slice-sampling keys
          4. Subtle horizontal line separator
          5. Dim "MASK PREPROCESSING" section header
          6. Form rows for the 4 cleaning keys

        Matches the "subtle horizontal line between sections"
        pattern the user pointed to in the Curation Settings dialog.
        """
        container = QWidget()
        col_lay = QVBoxLayout(container)
        col_lay.setContentsMargins(0, 0, 0, 0)
        col_lay.setSpacing(4)

        # 1. Bold amber column header
        if orientation == "lateral":
            header_text = "Lateral (XY)"
        else:
            header_text = "Axial (XZ + YZ)"
        col_header = QLabel(header_text)
        col_header.setStyleSheet(_COLUMN_HEADER_STYLE)
        col_lay.addWidget(col_header)

        # 2. Slice Sampling section header + form
        sampling_header = QLabel("Slice Sampling")
        sampling_header.setStyleSheet(_SECTION_HEADER_STYLE)
        col_lay.addWidget(sampling_header)

        sampling_form = QFormLayout()
        sampling_form.setLabelAlignment(Qt.AlignRight)
        sampling_form.setContentsMargins(8, 2, 0, 4)
        sampling_form.setSpacing(4)
        for key in _SAMPLING_KEYS:
            widget = self._make_spinbox(key)
            self._widgets[orientation][key] = widget
            sampling_form.addRow(_LABELS[key] + ":", widget)
        col_lay.addLayout(sampling_form)

        # 3. Subtle horizontal line separator
        hline = QFrame()
        hline.setFrameShape(QFrame.HLine)
        hline.setFrameShadow(QFrame.Plain)
        hline.setFixedHeight(1)
        hline.setStyleSheet(_HLINE_STYLE)
        col_lay.addWidget(hline)

        # 4. Mask Preprocessing section header + form
        cleaning_header = QLabel("Mask Preprocessing")
        cleaning_header.setStyleSheet(_SECTION_HEADER_STYLE)
        col_lay.addWidget(cleaning_header)

        cleaning_form = QFormLayout()
        cleaning_form.setLabelAlignment(Qt.AlignRight)
        cleaning_form.setContentsMargins(8, 2, 0, 4)
        cleaning_form.setSpacing(4)
        for key in _CLEANING_KEYS:
            if key in ("remove_boundaries", "relabel"):
                widget = self._make_checkbox(key)
            else:
                widget = self._make_spinbox(key)
            self._widgets[orientation][key] = widget
            cleaning_form.addRow(_LABELS[key] + ":", widget)
        col_lay.addLayout(cleaning_form)

        col_lay.addStretch(1)
        return container

    def _make_spinbox(self, key: str) -> QSpinBox:
        sb = QSpinBox()
        lo, hi = _RANGES.get(key, (0, 100000))
        sb.setRange(lo, hi)
        if key in _TOOLTIPS:
            sb.setToolTip(_TOOLTIPS[key])
        return sb

    def _make_checkbox(self, key: str) -> QCheckBox:
        cb = QCheckBox()
        if key in _TOOLTIPS:
            cb.setToolTip(_TOOLTIPS[key])
        return cb

    # ----------------------------------------------------------------
    # State population
    # ----------------------------------------------------------------

    def _populate_widgets_from_initial_state(self) -> None:
        """Fill each widget from initial_config[orientation][key] if
        present, else the orientation-specific default.

        2026-05-19 sub-commit 4: defaults are now ORIENTATION-SPECIFIC
        (LATERAL_DEFAULT_CONFIG vs AXIAL_DEFAULT_CONFIG) — the legacy
        scalar_defaults pass-through is gone since the main panel no
        longer has scalar widgets to read from.  scalar_defaults param
        is kept for backwards compatibility with older callers but no
        longer drives the precedence.
        """
        for orientation in ("lateral", "axial"):
            orient_cfg = self._initial_config.get(orientation, {}) or {}
            orientation_defaults = get_default_config_for(orientation)
            for key in _SAMPLING_KEYS + _CLEANING_KEYS:
                # Precedence: initial_config orientation override →
                # orientation-specific default.
                if key in orient_cfg:
                    value = orient_cfg[key]
                else:
                    value = orientation_defaults[key]
                self._set_widget_value(orientation, key, value)

    def _set_widget_value(
        self, orientation: str, key: str, value: Any,
    ) -> None:
        widget = self._widgets[orientation][key]
        if isinstance(widget, QCheckBox):
            widget.setChecked(bool(value))
        else:
            widget.setValue(int(value))

    def _get_widget_value(self, orientation: str, key: str) -> Any:
        widget = self._widgets[orientation][key]
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        return widget.value()

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def get_config(self) -> Dict[str, Dict[str, Any]]:
        """Read current widget state into the {lateral, axial} dict."""
        return {
            orientation: {
                key: self._get_widget_value(orientation, key)
                for key in _SAMPLING_KEYS + _CLEANING_KEYS
            }
            for orientation in ("lateral", "axial")
        }

    def _on_reset(self) -> None:
        """Clear all overrides — repopulate from scalar_defaults only."""
        self._initial_config = {}  # forget per-orientation overrides
        self._populate_widgets_from_initial_state()

    # ----------------------------------------------------------------
    # Convenience launcher
    # ----------------------------------------------------------------

    @classmethod
    def show_dialog(
        cls,
        parent: Optional[QWidget],
        *,
        initial_config: Optional[Dict[str, Dict[str, Any]]] = None,
        scalar_defaults: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """Open the dialog modally and return the saved config or None.

        Returns ``{"lateral": {...}, "axial": {...}}`` if the user
        clicked Save, or ``None`` if they clicked Cancel.
        """
        dlg = cls(
            parent=parent,
            initial_config=initial_config,
            scalar_defaults=scalar_defaults,
        )
        if dlg.exec_() != QDialog.Accepted:
            print("[PERVIEW-DLG] user cancelled dialog — no save")
            return None
        cfg = dlg.get_config()
        print(
            f"[PERVIEW-DLG] user saved dialog: "
            f"lateral={cfg.get('lateral')} | axial={cfg.get('axial')}"
        )
        return cfg
