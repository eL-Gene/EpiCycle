"""The Add orchestrator on REAL napari layers, offscreen.

``add_harness.Harness`` stubs every visual part of the page.  The candidate's
colours need the real thing: this rig keeps the harness's borrowed
orchestrator but gives it

* real ``ViewerModel`` objects -- the main viewer and the three side views;
* a real main ``Labels`` layer holding the volume;
* the real ``SubviewerManager`` and ``LabelSyncManager`` (so the side copies
  are made by ``add_labels_to_subviewers`` and every property change goes
  through the real ``connect_label_sync`` / ``sync_label_properties``);
* the real ``_render_add_candidate`` / ``_remove_add_candidate`` /
  ``_heal_label_render`` / channel-key toggle of the annotator.

Not real: the Qt viewers (offscreen there is no GL context, so no canvas can
be built; each side view gets a plain marker object that stands in for its
``QtViewer``), the drawing handler (a stand-in that returns a no-op drag
callback) and the Add panel (``ListAdv``: the harness's fake panel that also
fills its tick list the way the real list does -- rows at the box edge or
small start unticked -- and reports the fill to the orchestrator).
"""

from __future__ import annotations

import numpy as np

from napari_epicycle.annotator import curation_3d_widget as CW
from napari_epicycle.annotator import native_multiview_widget as NMW

from .add_harness import FakeAdv, Harness

C = CW.Curation3DAnnotator

#: the annotator's own methods this rig runs for real (the harness stubs
#: some of them)
_REAL = ("_render_add_candidate", "_remove_add_candidate",
         "_heal_label_render", "_add_main_label_layer",
         "_add_toggle_targets", "_add_set_candidate_colours",
         "_on_add_candidate_props", "_add_ensure_cmap_warm")
_REAL_BASE = ("_setup_channel_keybindings", "_teardown_channel_keybindings",
              "_all_subviewer_label_layers", "_orientation_for_vm")


class QtStandIn:
    """Stands in for a side view's ``QtViewer`` (no GL offscreen)."""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"QtStandIn({self.name})"


class _Drawing:
    def create_mouse_drag_callback(self, *_a, **_k):
        def _noop(_layer, _event):
            return None
        return _noop


class ListAdv(FakeAdv):
    """The fake panel, plus the candidate list's fill: the rows at the box
    edge or small start unticked, then the check callback fires once (as
    the real list does inside ``update_add_ui_state``)."""

    def __init__(self, owner):
        super().__init__()
        self._owner = owner

    def update_add_ui_state(self, **kw):
        super().update_add_ui_state(**kw)
        if "candidate_ids" in kw:
            ids = [int(i) for i in kw.get("candidate_ids") or []]
            off = (set(kw.get("candidate_edge_ids") or ())
                   | set(kw.get("candidate_small_ids") or ()))
            self.checked = [i for i in ids if i not in off]
            if ids:
                self._owner._sync_add_candidate_overlay()

    # what the All / None buttons and one tick do: change the ticks, then
    # fire the check callback
    def tick(self, ids):
        self.checked = [int(i) for i in ids]
        self._owner._sync_add_candidate_overlay()


class LayerRig(Harness):
    """The borrowed orchestrator on real napari layers (see the module)."""

    def __init__(self, vol, images=None, spacing=(1.0, 1.0, 1.0),
                 ndisplay=3):
        from napari.components import ViewerModel
        from napari.layers import Labels

        from packages.epicycle_annotator.subviewers import SubviewerManager
        from packages.epicycle_annotator.sync import LabelSyncManager

        super().__init__(vol, images=images, spacing=spacing)
        self.adv = ListAdv(self)
        self.viewer = ViewerModel()
        self.viewer.dims.ndisplay = ndisplay
        self.main_layer = Labels(vol, name="L")
        self.viewer.layers.append(self.main_layer)
        self.viewer_model_xy = ViewerModel()
        self.viewer_model_xz = ViewerModel()
        self.viewer_model_yz = ViewerModel()
        self.qt_viewer_xy = QtStandIn("XY")
        self.qt_viewer_xz = QtStandIn("XZ")
        self.qt_viewer_yz = QtStandIn("YZ")
        vms = (self.viewer_model_xy, self.viewer_model_xz,
               self.viewer_model_yz)
        qts = (self.qt_viewer_xy, self.qt_viewer_xz, self.qt_viewer_yz)
        self.subviewer_manager = SubviewerManager(*vms, *qts)
        self.sync_manager = LabelSyncManager(self.viewer, *vms, *qts)
        self.sync_manager.register_shared_data("L", vol)
        self.drawing_handler = _Drawing()
        self._active_labels_names = ["L"]
        self._active_image_layer_names = []
        self._channel_keybindings = []

    def _get_toggle_targets_for_name(self, name):
        """The labels key's targets while Add runs: the main labels layer
        and the Add artefacts (the annotator's own hook calls ``super()``,
        which a borrowed method cannot, so its Add branch is copied)."""
        out = [self.main_layer]
        out += [t for t in self._add_toggle_targets() if t not in out]
        return out

    def side_copies(self, name="_epicycle_add_candidate"):
        return [self.subviewer_manager.get_subviewer_layer(name, o)
                for o in ("XY", "XZ", "YZ")]


for _name in _REAL:
    if hasattr(C, _name):
        setattr(LayerRig, _name, getattr(C, _name))
LayerRig._CHANNEL_KEYS = NMW.NativeMultiviewAnnotator._CHANNEL_KEYS
LayerRig._ADD_CANDIDATE_PROP_EVENTS = getattr(
    C, "_ADD_CANDIDATE_PROP_EVENTS", ())
for _name in _REAL_BASE:
    setattr(LayerRig, _name, getattr(NMW.NativeMultiviewAnnotator, _name))
# every Add method the real annotator has now (the harness borrowed the set
# that existed when it was imported; a method added since is borrowed here)
for _name in dir(C):
    if (_name.startswith(("_add", "_on_add")) and _name not in
            LayerRig.__dict__ and not hasattr(Harness, _name)):
        _fn = getattr(C, _name)
        if callable(_fn):
            setattr(LayerRig, _name, _fn)


def blocks(shape, n, *, dtype=np.int32, first=1, block=None):
    """``n`` labelled cubes on a grid inside *shape* (ids ``first`` ...)."""
    out = np.zeros(shape, dtype)
    if block is None:
        vol = float(np.prod(shape)) / max(n, 1)
        block = max(2, int(round(vol ** (1.0 / 3.0))))
    k = 0
    for z in range(0, shape[0] - 1, block):
        for y in range(0, shape[1] - 1, block):
            for x in range(0, shape[2] - 1, block):
                if k >= n:
                    return out
                out[z:z + block - 1, y:y + block - 1, x:x + block - 1] = (
                    first + k)
                k += 1
    return out


def result_for(roi, result, *, edge=(), small=(), whole_patch=False,
               preset="cells"):
    """An ``AddPreviewResult`` for a hand-made *result* on *roi*."""
    from packages.epicycle_annotator import add_preview as AP

    ids = sorted(int(v) for v in np.unique(result) if v)
    sizes = {i: int(np.count_nonzero(result == i)) for i in ids} \
        if len(ids) < 64 else {i: 1 for i in ids}
    return AP.AddPreviewResult(
        roi=tuple(roi), result=result, ids=ids, sizes=sizes,
        edge_ids=set(int(i) for i in edge),
        small_ids=set(int(i) for i in small), preset=preset,
        source="labels" if preset == "cells" else None,
        whole_patch=bool(whole_patch))
