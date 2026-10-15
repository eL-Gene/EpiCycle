"""Set spacing…: the voxel size, the Z orientation
and the axis order of the ticked stacks, or of one row's ``⋯`` menu.

Since R7a the dialog lives in
``ui_shell/_set_spacing_dialog.py``, shared with Step 3; this module is a
thin subclass that passes Step 4's words from ``_step4_text`` with both the
Z orientation and the axes rows shown.  Its widget tree is held byte for
byte by ``tests/segmentation/golden/step4_spacing_dialog.json`` (captured
from the code before the lift).

Modal and themed (``_DIALOG_STYLE``).  It only collects the values; the
controller writes ``spacing = {"zyx": [...], "source": "user"}``,
``z_orientation`` and ``axes_override`` onto each record, after the
"what goes stale" confirm (``SPACING_CHANGE_CONFIRM``).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from ..ui_shell import _set_spacing_dialog as _shared
from ..ui_shell._set_spacing_dialog import (  # noqa: F401  (re-exported)
    AXES_CHOICES,
    Z_ORIENTATIONS,
)
from . import _step4_text as T


def step4_spacing_texts() -> Dict[str, Any]:
    """Step 4's words for the shared dialog, read from ``_step4_text`` at
    call time (``ui_shell`` may not import a step package)."""
    return {
        "title": T.SPACING_DIALOG_TITLE,
        "desc": T.SPACING_DIALOG_DESC,
        "desc_one": T.SPACING_DIALOG_DESC_ONE,
        "z_label": T.SPACING_Z_LABEL,
        "y_label": T.SPACING_Y_LABEL,
        "x_label": T.SPACING_X_LABEL,
        "zorient_label": T.SPACING_ZORIENT_LABEL,
        "zorient_items": T.SPACING_ZORIENT_ITEMS,
        "axes_label": T.SPACING_AXES_LABEL,
        "axes_auto": T.SPACING_AXES_AUTO,
        "cancel": T.CANCEL_BTN,
        "ok": T.SPACING_OK_BTN,
    }


class SetSpacingDialog(_shared.SetSpacingDialog):
    """Z / Y / X in µm, the Z orientation and the axes override."""

    def __init__(self, parent=None, *, names=(), record: Optional[Dict] = None):
        super().__init__(parent, names=names, record=record,
                         texts=step4_spacing_texts(),
                         show_z_orientation=True, show_axes=True)
