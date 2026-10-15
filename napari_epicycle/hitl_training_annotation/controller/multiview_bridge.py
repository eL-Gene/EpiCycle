"""Optional bridge to launch napari-multiview-annotator from EpiCycle HITL."""

from __future__ import annotations

import importlib
from typing import Tuple


def launch_multiview_annotator(
    viewer,
    *,
    dock_name: str = "EpiCycle Multiview Annotator",
) -> Tuple[bool, str]:
    """Launch the multiview annotator as a dock widget when available.

    Returns `(ok, message)`.
    """
    if viewer is None:
        return False, "Viewer is not available."
    window = getattr(viewer, "window", None)
    if window is None:
        return False, "Viewer window is not available."

    try:
        mod = importlib.import_module("napari_multiview_annotator._widget")
        widget_cls = getattr(mod, "MultiviewAnnotator")
    except Exception:
        return (
            False,
            "napari-multiview-annotator is not importable in this environment. "
            "Install it in the same env to enable this bridge.",
        )

    existing = getattr(window, "_dock_widgets", None)
    if isinstance(existing, dict):
        for key, dock in existing.items():
            if str(key) == dock_name:
                try:
                    if hasattr(dock, "show"):
                        dock.show()
                    if hasattr(dock, "raise_"):
                        dock.raise_()
                except Exception:
                    pass
                return True, "Multiview annotator is already open."

    try:
        widget = widget_cls(viewer)
        window.add_dock_widget(widget, name=dock_name, area="right")
        return True, "Opened multiview annotator."
    except Exception as exc:
        return False, f"Failed to open multiview annotator: {exc}"

