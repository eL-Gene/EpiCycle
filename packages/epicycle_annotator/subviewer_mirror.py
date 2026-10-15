"""Subviewer mirror manager for the native EpiCycle annotator.

Propagates main viewer layer add/remove/rename events to the three
internal subviewer panels (XY/XZ/YZ).  This is NOT the global
LayerLifecycleManager — that lives in ``napari_epicycle.ui_shell``.
"""

from __future__ import annotations

from napari.layers import Image, Labels


class SubviewerMirrorManager:
    """Coordinate layer add/remove/rename sync for native subviewers."""

    def __init__(
        self,
        main_viewer,
        subviewer_manager,
        sync_manager,
        drawing_handler,
        is_managed_layer=None,
        right_click_callback_factory=None,
    ) -> None:
        self.main_viewer = main_viewer
        self.subviewer_manager = subviewer_manager
        self.sync_manager = sync_manager
        self.drawing_handler = drawing_handler
        self.is_managed_layer = is_managed_layer
        # Optional factory that returns a napari mouse callback to be
        # appended to each subviewer label layer's mouse_drag_callbacks.
        # Called once per add_labels_to_subviewers invocation — letting
        # the annotator return a closure bound to its own state without
        # the mirror manager having to know the annotator's API.
        self.right_click_callback_factory = right_click_callback_factory
        self._listeners_connected = False
        # When True, on_layer_removed() and on_layer_added() skip subviewer
        # changes.  Set by the annotator during controlled patch switches so
        # that existing subviewer layers survive for the FAST reload path.
        self._suppress_events = False
        # Track per-layer rename callbacks for symmetric teardown.
        # Maps layer id -> (layer_weakref_or_ref, callback)
        self._rename_cbs: dict = {}

    def _is_managed(self, layer) -> bool:
        if layer is None:
            return False
        if self.is_managed_layer is None:
            return True
        try:
            return bool(self.is_managed_layer(layer))
        except Exception:
            return False

    def setup_layer_listeners(self) -> None:
        if self._listeners_connected:
            return
        self.main_viewer.layers.events.inserted.connect(self.on_layer_added)
        self.main_viewer.layers.events.removed.connect(self.on_layer_removed)
        self.main_viewer.layers.selection.events.active.connect(self.connect_label_mode_sync)
        self.main_viewer.layers.selection.events.active.connect(self.sync_manager.sync_active_layer)
        self._listeners_connected = True

    def disconnect_layer_listeners(self) -> None:
        """Disconnect listeners registered by ``setup_layer_listeners``.

        This must be called during annotator teardown. Otherwise each
        close/relaunch cycle leaves stale global listeners on the main viewer,
        causing repeated callbacks, lag, and cross-talk.
        """
        if not self._listeners_connected:
            return
        try:
            self.main_viewer.layers.events.inserted.disconnect(self.on_layer_added)
        except (ValueError, AttributeError):
            pass
        try:
            self.main_viewer.layers.events.removed.disconnect(self.on_layer_removed)
        except (ValueError, AttributeError):
            pass
        try:
            self.main_viewer.layers.selection.events.active.disconnect(
                self.connect_label_mode_sync
            )
        except (ValueError, AttributeError):
            pass
        try:
            self.main_viewer.layers.selection.events.active.disconnect(
                self.sync_manager.sync_active_layer
            )
        except (ValueError, AttributeError):
            pass

        # --- Per-layer rename callbacks ---
        # on_layer_added() connects lambdas to layer.events.name.
        # Without disconnecting here, they accumulate across close/relaunch.
        for layer_id, (layer, cb) in list(self._rename_cbs.items()):
            try:
                layer.events.name.disconnect(cb)
            except (ValueError, AttributeError):
                pass
        self._rename_cbs.clear()

        self._listeners_connected = False

    def on_layer_added(self, event) -> None:
        layer = getattr(event, "value", None)
        if layer is None or not self._is_managed(layer):
            return
        if not self._listeners_connected:
            return
        if self._suppress_events:
            return
        if isinstance(layer, Image) and getattr(layer.data, "ndim", 0) == 3:
            self.subviewer_manager.add_image_to_subviewers(layer)
            self.subviewer_manager.sync_layer_visibility(layer)
            self.subviewer_manager.sync_image_style(layer)
            old_name = layer.name
            cb = lambda _e, old=old_name: self.on_layer_renamed(old, layer, is_label=False)
            layer.events.name.connect(cb)
            self._rename_cbs[id(layer)] = (layer, cb)
        elif isinstance(layer, Labels) and getattr(layer.data, "ndim", 0) == 3:
            rc_cb = None
            if self.right_click_callback_factory is not None:
                try:
                    rc_cb = self.right_click_callback_factory()
                except Exception:
                    rc_cb = None
            self.subviewer_manager.add_labels_to_subviewers(
                layer,
                sync_manager=self.sync_manager,
                drawing_handler=self.drawing_handler,
                right_click_callback=rc_cb,
            )
            self.subviewer_manager.sync_layer_visibility(layer)
            old_name = layer.name
            cb = lambda _e, old=old_name: self.on_layer_renamed(old, layer, is_label=True)
            layer.events.name.connect(cb)
            self._rename_cbs[id(layer)] = (layer, cb)

    def on_layer_renamed(self, old_name: str, main_layer, is_label: bool) -> None:
        new_name = str(getattr(main_layer, "name", old_name))
        self.subviewer_manager.rename_layer_in_subviewers(old_name, new_name)
        if is_label:
            self.sync_manager.rename_layer(old_name, new_name)

    def on_layer_removed(self, event) -> None:
        layer = getattr(event, "value", None)
        if layer is None:
            return
        # Guard: if disconnect_layer_listeners already ran, this is a
        # stale queued event firing after teardown.  The subviewers are
        # dead — touching them crashes vispy.  Bail immediately.
        if not self._listeners_connected:
            return
        # Clean up per-layer rename callback before removing.
        entry = self._rename_cbs.pop(id(layer), None)
        if entry is not None:
            try:
                entry[0].events.name.disconnect(entry[1])
            except (ValueError, AttributeError):
                pass
        # During controlled patch switches, skip subviewer cleanup so
        # existing subviewer layers survive for the FAST reload path.
        if self._suppress_events:
            return
        self.subviewer_manager.remove_layer_from_subviewers(layer.name)
        if isinstance(layer, Labels):
            self.sync_manager.unregister_layer(layer.name)

    def connect_label_mode_sync(self, event) -> None:
        active = getattr(event, "value", None)
        if isinstance(active, Labels):
            self.sync_manager.connect_label_sync(active)
