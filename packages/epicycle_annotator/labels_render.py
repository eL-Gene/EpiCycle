"""Labels render-state repair for napari >= 0.7 (hidden-layer placeholder hazard).

Background
-----------------------------------------------------------------------
Since napari 0.7 (PR #8254) a *hidden* Labels layer no longer re-slices.
When the dims change while the layer is hidden, napari installs a tiny
EMPTY placeholder slice whose ``raw`` dtype is the layer's *texture* dtype
(uint8 / uint16), not the data dtype.  If any colormap-affecting event then
reaches vispy while the layer is still hidden (``events.colormap``,
``show_selected_label``, ``selected_label`` with show-selected on), vispy's
``_on_colormap_change`` reads that placeholder dtype, decides the data is a
small integer type, and builds the *auto* label shader
(``LabelVispyColormap``) for what is really a ``DirectLabelColormap`` over
uint32 data.  The later re-show re-slices the real data but never rebuilds
the shader, so the volume renders a random subset of labels in wrong
colours until something forces a node reset (a 2D<->3D toggle, or a
texture-dtype change).

EpiCycle's isolation paths hide the labels layers, recolour them, navigate
(which moves the Z step and pokes ``show_selected_label``) and re-show them
one event-loop tick later — exactly this sequence.  The helpers here are
called right after each re-show: they compare the vispy node's colormap
class with the layer's colormap and rebuild only when they disagree.  Cost
is zero in the healthy case and ~1-4 ms in the poisoned case.

Pure-napari reproduction: ``scripts/diag/napari_hidden_layer_colormap_poison_probe.py``.
"""
from __future__ import annotations

from typing import Optional

__all__ = [
    "labels_colormap_node_is_stale",
    "rebuild_stale_labels_colormap",
    "heal_labels_render",
]


def _vispy_layer_for(layer, qt_viewer):
    """The vispy visual wrapping *layer* on *qt_viewer* (or None)."""
    if qt_viewer is None or layer is None:
        return None
    try:
        return qt_viewer.canvas.layer_to_visual[layer]
    except Exception:  # noqa: BLE001 — missing canvas / not yet attached
        return None


def labels_colormap_node_is_stale(layer, qt_viewer) -> Optional[bool]:
    """Return True if the vispy colormap node for *layer* disagrees with the
    layer's colormap, False if consistent, None if it cannot be determined
    (no vispy handle, layer hidden, placeholder slice still installed).

    Only the *direct* colormap case is checked: a ``DirectLabelColormap``
    over data wider than 16 bits must be drawn by ``DirectLabelVispyColormap``.
    The auto (cyclic) colormap is drawn by ``LabelVispyColormap`` on every
    path, so it cannot be told apart from the poisoned state by class alone
    (and EpiCycle never navigates a hidden layer that carries one).
    """
    try:
        from napari.layers import Labels
        from napari.utils.colormaps import DirectLabelColormap
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(layer, Labels):
        return None
    if not getattr(layer, "visible", True):
        return None  # hidden: the placeholder may be installed — do not touch
    try:
        sl = layer._slice
        if getattr(sl, "empty", False):
            return None  # placeholder still installed (async slicing in flight)
        raw_dtype = sl.image.raw.dtype
    except Exception:  # noqa: BLE001
        return None
    vl = _vispy_layer_for(layer, qt_viewer)
    if vl is None:
        return None
    node = getattr(vl, "node", None)
    node_cmap = getattr(node, "cmap", None)
    if node_cmap is None:
        return None
    if not isinstance(layer.colormap, DirectLabelColormap):
        return False
    if getattr(raw_dtype, "itemsize", 0) <= 2:
        return False  # small ints legitimately use the auto shader path
    return type(node_cmap).__name__ != "DirectLabelVispyColormap"


def rebuild_stale_labels_colormap(layer, qt_viewer) -> bool:
    """Rebuild the vispy colormap node for *layer* if it is stale.

    Returns True when a rebuild was performed.  Never raises.
    """
    stale = labels_colormap_node_is_stale(layer, qt_viewer)
    if not stale:
        return False
    vl = _vispy_layer_for(layer, qt_viewer)
    try:
        if vl is not None and hasattr(vl, "_on_colormap_change"):
            vl._on_colormap_change()  # rebuild against the REAL slice now
        else:
            layer.events.colormap()  # public fallback (~3 ms)
        return True
    except Exception:  # noqa: BLE001
        return False


def heal_labels_render(layers_and_viewers, *, tag: str = "") -> int:
    """Repair every ``(layer, qt_viewer)`` pair that is stale; return the
    number rebuilt.  Prints one ``[RENDER-HEAL]`` line when anything was
    rebuilt so a poisoned session is visible in the console.
    """
    n = 0
    names = []
    for layer, qt_viewer in layers_and_viewers:
        try:
            if rebuild_stale_labels_colormap(layer, qt_viewer):
                n += 1
                names.append(str(getattr(layer, "name", "?")))
        except Exception:  # noqa: BLE001
            pass
    if n:
        print(
            f"[RENDER-HEAL] {tag or 'reshow'}: rebuilt stale colormap node "
            f"on {n} layer(s): {names}"
        )
    return n
