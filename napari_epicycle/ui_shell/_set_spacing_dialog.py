"""Set spacing… -- the voxel-size dialog Steps 3 and 4 share.

Lifted from Step 4's ``segmentation/_set_spacing_dialog.py`` in brick R7a.
Step 4's module is now a thin subclass that passes its own words (from
``_step4_text``) with both rows shown; its widget tree is held byte for byte
by ``tests/segmentation/golden/step4_spacing_dialog.json``, captured from the
untouched code before the lift.

Modal and themed (``_DIALOG_STYLE``).  It only collects the values; the
caller writes them and runs its own "what goes stale" confirm.

* ``texts`` -- every word on the dialog (``ui_shell`` may not import a step
  package, so a step's words arrive as a mapping; the keys are
  ``SPACING_TEXT_KEYS``).  ``desc_one`` is formatted with ``stack=`` and
  ``name=`` (the first name), ``desc`` with ``n=``.
* ``show_z_orientation`` / ``show_axes`` -- build the Z orientation and the
  axes rows (Step 4: both; Step 3 passes ``False, False``: it resolves axes
  per patch already and has no Z orientation).  A row not shown is never
  built: its attribute is ``None`` and ``values()`` leaves its key out.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from . import theme as th

Z_ORIENTATIONS = ("basal_down", "basal_up")
AXES_CHOICES = ("", "CZYX", "ZCYX", "ZYX")

#: the keys ``texts`` must carry; ``zorient_label`` / ``zorient_items`` and
#: ``axes_label`` / ``axes_auto`` are read only when their row is shown
SPACING_TEXT_KEYS = ("title", "desc", "desc_one", "z_label", "y_label",
                     "x_label", "zorient_label", "zorient_items",
                     "axes_label", "axes_auto", "cancel", "ok")
_ROW_KEYS = {"zorient_label": "show_z_orientation",
             "zorient_items": "show_z_orientation",
             "axes_label": "show_axes", "axes_auto": "show_axes"}


class SetSpacingDialog(QDialog):
    """Z / Y / X in µm and, when shown, the Z orientation and the axes
    override."""

    def __init__(self, parent=None, *, names=(), record: Optional[Dict] = None,
                 texts: Mapping[str, Any],
                 show_z_orientation: bool = True,
                 show_axes: bool = True, accent: str = "green"):
        super().__init__(parent)
        shown = {"show_z_orientation": bool(show_z_orientation),
                 "show_axes": bool(show_axes)}
        missing = [k for k in SPACING_TEXT_KEYS
                   if k not in texts and shown.get(_ROW_KEYS.get(k), True)]
        if missing:
            raise ValueError(f"SetSpacingDialog texts lack {missing}")
        record = record or {}
        names = list(names)
        self.setWindowTitle(texts["title"])
        self.setStyleSheet(th._DIALOG_STYLE + th.INPUT_STYLE)
        self.setMinimumWidth(360)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 16, 20, 16)
        lay.setSpacing(10)
        self.desc_label = QLabel(
            texts["desc_one"].format(stack=names[0], name=names[0])
            if len(names) == 1 else texts["desc"].format(n=len(names)))
        self.desc_label.setWordWrap(True)
        self.desc_label.setStyleSheet(th.FLOW_SUBTITLE_STYLE)
        lay.addWidget(self.desc_label)

        form = QFormLayout()
        form.setSpacing(6)
        zyx = None
        sp = record.get("spacing")
        if isinstance(sp, dict):
            zyx = sp.get("zyx")
        zyx = list(zyx) if zyx else [1.0, 1.0, 1.0]
        self.spacing_z_sb = self._spin(zyx[0])
        self.spacing_y_sb = self._spin(zyx[1])
        self.spacing_x_sb = self._spin(zyx[2])
        form.addRow(texts["z_label"], self.spacing_z_sb)
        form.addRow(texts["y_label"], self.spacing_y_sb)
        form.addRow(texts["x_label"], self.spacing_x_sb)
        self.z_orientation_combo = None
        if shown["show_z_orientation"]:
            self.z_orientation_combo = QComboBox()
            self.z_orientation_combo.addItems(list(texts["zorient_items"]))
            zo = record.get("z_orientation") or "basal_down"
            self.z_orientation_combo.setCurrentIndex(
                Z_ORIENTATIONS.index(zo) if zo in Z_ORIENTATIONS else 0)
            form.addRow(texts["zorient_label"], self.z_orientation_combo)
        self.axes_combo = None
        if shown["show_axes"]:
            self.axes_combo = QComboBox()
            self.axes_combo.addItems([texts["axes_auto"]]
                                     + list(AXES_CHOICES[1:]))
            ov = record.get("axes_override") or ""
            self.axes_combo.setCurrentIndex(
                AXES_CHOICES.index(ov) if ov in AXES_CHOICES else 0)
            form.addRow(texts["axes_label"], self.axes_combo)
        lay.addLayout(form)

        row = QHBoxLayout()
        row.addStretch(1)
        self.cancel_btn = QPushButton(texts["cancel"])
        self.cancel_btn.setStyleSheet(th._DIALOG_BTN_CANCEL)
        self.cancel_btn.clicked.connect(self.reject)
        self.ok_btn = QPushButton(texts["ok"])
        # the OK button is the only accent-coloured part of the dialog
        # (A3 fold): Step 4's green by default, Step 3 passes "amber"
        self.ok_btn.setStyleSheet(th._DIALOG_BTN_CONFIRM if accent == "amber"
                                  else th._DIALOG_BTN_CONFIRM_GREEN)
        self.ok_btn.setDefault(True)
        self.ok_btn.clicked.connect(self.accept)
        row.addWidget(self.cancel_btn)
        row.addWidget(self.ok_btn)
        lay.addLayout(row)

    @staticmethod
    def _spin(value) -> QDoubleSpinBox:
        sb = QDoubleSpinBox()
        sb.setRange(0.001, 100.0)
        sb.setDecimals(3)
        sb.setSingleStep(0.1)
        try:
            sb.setValue(float(value))
        except (TypeError, ValueError):
            sb.setValue(1.0)
        return sb

    def values(self) -> Dict[str, Any]:
        """``{"zyx"}`` plus ``"z_orientation"`` / ``"axes_override"`` for
        the rows that are shown, as chosen."""
        out: Dict[str, Any] = {
            "zyx": [self.spacing_z_sb.value(), self.spacing_y_sb.value(),
                    self.spacing_x_sb.value()],
        }
        if self.z_orientation_combo is not None:
            out["z_orientation"] = Z_ORIENTATIONS[
                max(self.z_orientation_combo.currentIndex(), 0)]
        if self.axes_combo is not None:
            out["axes_override"] = AXES_CHOICES[
                max(self.axes_combo.currentIndex(), 0)] or None
        return out
