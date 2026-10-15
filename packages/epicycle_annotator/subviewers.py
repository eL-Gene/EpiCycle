"""Subviewer layer management for native EpiCycle annotator."""

from __future__ import annotations

from napari.layers import Image, Labels

from .orientation import transpose_data
from .sync import lock_brush_size_to_1


def install_button_aware_paint(label_layer) -> None:
    """Replace napari's PAINT mouse callback with a button-aware wrapper
    that only paints on left-click (``event.button == 1``).

    Why this exists:
    napari's ``draw`` callback
    (``napari.layers.labels._labels_mouse_bindings.draw``) paints on
    every mouse press regardless of button — see
    ``layer._draw(...)`` BEFORE the first ``yield``.  In EpiCycle's
    subviewer workflow we use right-click for navigator-focus, and
    without intervention each right-click leaves a single-voxel paint
    dot of ``selected_label``.

    Why patch at the source (not after the fact):
    Removing ``draw`` from ``layer.mouse_drag_callbacks`` inside the
    right-click handler is too late — napari iterates the callback
    list in order and ``draw`` is appended FIRST (at
    ``mode = "paint"`` set-time, before the right-click handler is
    added).  By the time the right-click handler runs, the dot is
    already painted.  Patching ``_drag_modes[Mode.PAINT]`` with a
    wrapper means napari always pulls the wrapper when paint mode is
    set, and the wrapper itself decides whether to paint.

    Per-layer scope (CRITICAL):
    ``Labels._drag_modes`` is a ``ClassVar`` dict — mutating it on the
    class poisons EVERY Labels layer in the application, including
    the main viewer's labels.  We instance-shadow the dict
    (``layer._drag_modes = dict(cls._drag_modes)``) before patching so
    only this subviewer label sees the wrapper.

    Live ``mouse_drag_callbacks`` swap:
    If PAINT is the current mode at install time, the original
    ``draw`` cb is already in ``mouse_drag_callbacks`` (napari's mode
    setter put it there).  We find it by identity and swap it with
    the wrapper so the FIRST press hits the wrapper, not the
    original.

    Idempotent:
    Guarded by ``_epicycle_paint_wrapper_installed`` flag.  Safe to
    call multiple times on the same layer (e.g., across patch-switch
    reload).
    """
    if getattr(label_layer, "_epicycle_paint_wrapper_installed", False):
        return
    try:
        from napari.layers.labels._labels_constants import Mode
    except ImportError:
        return
    cls = type(label_layer)
    class_drag_modes = getattr(cls, "_drag_modes", None)
    if not isinstance(class_drag_modes, dict):
        return
    original_paint = class_drag_modes.get(Mode.PAINT)
    if original_paint is None:
        return

    def _paint_left_only(layer, event):
        """Button-aware wrapper around napari's PAINT draw callback.

        Left-button (1) presses delegate to the original generator
        (yielding through for napari's drag protocol).  Right-button
        (2) and other buttons no-op silently — this is what prevents
        right-click paint dots on subviewer labels.
        """
        if getattr(event, "button", 1) == 1:
            yield from original_paint(layer, event)
        # else: silently skip — right-click or other button

    # Instance-shadow _drag_modes to avoid mutating the ClassVar dict
    # (which would affect every Labels layer in the application).
    label_layer._drag_modes = dict(class_drag_modes)
    label_layer._drag_modes[Mode.PAINT] = _paint_left_only

    # If PAINT is the active mode, napari has already added the
    # original draw cb to mouse_drag_callbacks.  Swap it in place
    # so the FIRST mouse press hits our wrapper.
    cbs = getattr(label_layer, "mouse_drag_callbacks", None)
    if cbs is not None:
        try:
            for i, cb in enumerate(list(cbs)):
                if cb is original_paint:
                    cbs[i] = _paint_left_only
                    break
        except (AttributeError, TypeError):
            pass

    label_layer._epicycle_paint_wrapper_installed = True
    label_layer._epicycle_original_paint = original_paint
    label_layer._epicycle_paint_wrapper = _paint_left_only


def _disable_brush_stroke_overlay(label_layer) -> bool:
    """Turn napari 0.8's labels brush-stroke overlay OFF for this layer.

    Returns True when the overlay model was found and disabled.

    The overlay model is created in ``Labels.__init__``
    (``_overlays['brush_stroke']``), so this is safe to call before any vispy
    visual exists.
    """
    try:
        overlay = getattr(label_layer, "_overlays", {}).get("brush_stroke")
    except Exception:
        return False
    if overlay is None:
        return False
    try:
        overlay.enabled = False
    except Exception:
        return False
    return True


def disable_encircle_fill(label_layer) -> bool:
    """Contain napari 0.8's right-click encircle-and-fill on a subviewer label.

    napari 0.8 gives every Labels layer a brush-stroke overlay whose press
    handler starts an encircle-and-fill stroke on the RIGHT button, painting
    the subviewer layer's data directly.  EpiCycle's subviewer -> main sync is
    drag-generator-only (``DrawingHandler``), so that fill is visible until the
    next reslice and is then silently discarded -- the user loses the work with
    no warning.

    We disable it through the overlay MODEL and never touch
    ``mouse_drag_callbacks``.  Two reasons:

    * ``VispyLabelsBrushStrokeOverlay._on_mouse_press`` short-circuits on
      ``overlay.enabled`` before it can start a stroke
      (``_only_when_overlay_enabled``), so ``enabled = False`` blocks the
      gesture completely; and
    * ``VispyLabelsBrushStrokeOverlay.close()`` does an UNGUARDED
      ``layer.mouse_drag_callbacks.remove(self._on_mouse_press)``, so removing
      that callback ourselves makes every later layer removal raise
      ``ValueError: list.remove(x): x not in list`` out of
      ``viewer.layers.remove(...)`` -- a crash in BOTH annotators
      (``_add_or_replace_labels``, ``remove_layer_from_subviewers``, and
      ``clear()``, which would abort mid-loop leaving orphan layers).

    ``Labels._mode_setter_helper`` re-enables the overlay on every switch into
    PAINT, BEFORE ``events.mode`` is emitted, so we re-assert on that event.

    EpiCycle's own callbacks (the DrawingHandler drag callback, the right-click
    navigator-focus handler), ``preserve_labels`` and
    ``install_button_aware_paint`` are all untouched.  Shared with Step 3,
    whose subviewers have the same one-way sync.

    Idempotent; safe on a layer with no ``events`` / ``_overlays``.
    """
    disabled = _disable_brush_stroke_overlay(label_layer)
    if not getattr(label_layer, "_epicycle_encircle_fill_disabled", False):
        try:
            label_layer.events.mode.connect(
                lambda *_a, **_k: _disable_brush_stroke_overlay(label_layer)
            )
        except Exception:
            pass
        try:
            label_layer._epicycle_encircle_fill_disabled = True
        except Exception:
            pass
    return disabled


class SubviewerManager:
    """Add/remove image and label layers across XY/XZ/YZ viewer models."""

    def __init__(
        self,
        viewer_model_xy,
        viewer_model_xz,
        viewer_model_yz,
        qt_viewer_xy,
        qt_viewer_xz,
        qt_viewer_yz,
    ) -> None:
        self.viewer_model_xy = viewer_model_xy
        self.viewer_model_xz = viewer_model_xz
        self.viewer_model_yz = viewer_model_yz
        self.qt_viewer_xy = qt_viewer_xy
        self.qt_viewer_xz = qt_viewer_xz
        self.qt_viewer_yz = qt_viewer_yz

        # Handler references for clean disconnection on close/relaunch.
        self._vis_handlers: dict = {}    # id(layer) → (event_emitter, handler)
        self._style_handlers: dict = {}  # id(layer) → [(event_emitter, handler), ...]

    def _find_layer(self, vm, name: str):
        for layer in vm.layers:
            if layer.name == name:
                return layer
        return None

    def _add_or_replace_image(self, vm, name: str, data, **kwargs):
        existing = self._find_layer(vm, name)
        if existing is not None:
            # Update data in-place if shape matches — avoids full layer teardown.
            try:
                if existing.data.shape == data.shape:
                    existing.data = data
                    for k, v in kwargs.items():
                        if k in ("colormap", "contrast_limits", "opacity", "blending", "gamma", "rendering"):
                            try:
                                setattr(existing, k, v)
                            except Exception:
                                pass
                    return
            except Exception:
                pass
            vm.layers.remove(existing)
        vm.add_image(data, name=name, **kwargs)

    def _add_or_replace_labels(self, vm, name: str, data):
        existing = self._find_layer(vm, name)
        if existing is not None:
            # Update data in-place if shape matches — avoids full layer teardown.
            try:
                if existing.data.shape == data.shape:
                    existing.data = data
                    return existing
            except Exception:
                pass
            vm.layers.remove(existing)
        return vm.add_labels(data, name=name)

    def add_image_to_subviewers(self, layer: Image) -> None:
        data_xy = transpose_data(layer.data, "XY")
        data_xz = transpose_data(layer.data, "XZ")
        data_yz = transpose_data(layer.data, "YZ")
        is_rgb = bool(getattr(layer, "rgb", False))
        if not is_rgb:
            try:
                arr = layer.data
                is_rgb = getattr(arr, "ndim", 0) == 4 and int(arr.shape[-1]) in {3, 4}
            except Exception:
                is_rgb = False

        style = {
            "opacity": layer.opacity,
            "blending": layer.blending,
        }
        if is_rgb:
            style["rgb"] = True
        else:
            style["colormap"] = (
                layer.colormap.name if hasattr(layer.colormap, "name") else layer.colormap
            )
            style["contrast_limits"] = layer.contrast_limits
            style["rendering"] = "mip"
        self._add_or_replace_image(
            self.viewer_model_xy,
            f"{layer.name}_XY",
            data_xy,
            **style,
        )
        self._add_or_replace_image(
            self.viewer_model_xz,
            f"{layer.name}_XZ",
            data_xz,
            **style,
        )
        self._add_or_replace_image(
            self.viewer_model_yz,
            f"{layer.name}_YZ",
            data_yz,
            **style,
        )

    def add_labels_to_subviewers(
        self,
        label_layer: Labels,
        sync_manager,
        drawing_handler,
        right_click_callback=None,
    ) -> None:
        """Add transposed subviewer Labels layers + wire drawing drag
        callback + optional right-click callback.

        Parameters
        ----------
        right_click_callback : callable(layer, event) or None
            Optional napari mouse callback.  Appended to each
            subviewer's ``mouse_drag_callbacks`` AFTER the drawing
            callback.  The callback is expected to short-circuit on
            anything but right-button (``event.button == 2``) events
            so it doesn't interfere with painting (left-button drag).
            Idempotent — guarded by
            ``_epicycle_right_click_connected`` per-layer.
        """
        name = label_layer.name
        sync_manager.register_shared_data(name, label_layer.data)
        sync_manager.connect_label_sync(label_layer)

        l_xy = self._add_or_replace_labels(
            self.viewer_model_xy,
            f"{name}_XY",
            transpose_data(label_layer.data, "XY"),
        )
        l_xz = self._add_or_replace_labels(
            self.viewer_model_xz,
            f"{name}_XZ",
            transpose_data(label_layer.data, "XZ"),
        )
        l_yz = self._add_or_replace_labels(
            self.viewer_model_yz,
            f"{name}_YZ",
            transpose_data(label_layer.data, "YZ"),
        )

        # Clip drawing to existing labels, but only for
        # layers that opt in.  Work-labels default ON; the split Refine layer sets
        # _epicycle_clip_to_existing=False so seeds can paint OVER the target.
        _clip = bool(getattr(label_layer, "_epicycle_clip_to_existing", True))
        for lbl in [l_xy, l_xz, l_yz]:
            lbl.editable = True
            # brush_size locked at 1 — see lock_brush_size_to_1 docstring.
            lock_brush_size_to_1(lbl)
            # preserve_labels (napari built-in) makes the brush fill ONLY
            # background and the eraser remove ONLY the selected label — so
            # neither overwrites a neighbour.  Gated by the per-layer clip flag,
            # which is also read by the enclosed-fill mask in drawing.py.
            lbl.preserve_labels = _clip
            lbl._epicycle_clip_to_existing = _clip
            lbl.selected_label = int(getattr(label_layer, "selected_label", 1))
            lbl.mode = "paint"
            # Install button-aware paint wrapper AFTER mode is set so
            # the live mouse_drag_callbacks (which napari just populated
            # with the original draw cb) gets the wrapper swapped in.
            # Prevents right-click from painting single-voxel dots —
            # see install_button_aware_paint docstring for rationale.
            install_button_aware_paint(lbl)
            # napari 0.8 right-click encircle-and-fill would paint straight
            # into this layer's data and never reach the main volume — see
            # disable_encircle_fill (overlay model only; the callback list is
            # deliberately left alone, napari's own close() unlinks it).
            disable_encircle_fill(lbl)
            # Propagate colormap from source label (preserves DirectLabelColormap
            # from QC coloring so subviewers match the main viewer).
            try:
                lbl.colormap = label_layer.colormap
            except Exception:
                pass
            # Only append drag callback once per label layer.
            if not getattr(lbl, "_epicycle_drag_connected", False):
                drag_callback = drawing_handler.create_mouse_drag_callback(
                    lbl.name[-2:], self._viewer_model_for_orientation(lbl.name[-2:])
                )
                lbl.mouse_drag_callbacks.append(drag_callback)
                lbl._epicycle_drag_connected = True
            # Right-click → navigator-focus (subviewer-only feature,
            # 2026-05-10).  Idempotent guard prevents N callbacks if
            # the layer is re-passed through this method (e.g.,
            # patch-switch reload).
            if right_click_callback is not None and not getattr(
                lbl, "_epicycle_right_click_connected", False
            ):
                lbl.mouse_drag_callbacks.append(right_click_callback)
                lbl._epicycle_right_click_connected = True

    def sync_layer_visibility(self, layer) -> None:
        # Guard: only connect once per layer.
        if getattr(layer, "_epicycle_vis_synced", False):
            return
        layer._epicycle_vis_synced = True

        def update_visibility(_event):
            for orient, vm in zip(
                ["XY", "XZ", "YZ"],
                [self.viewer_model_xy, self.viewer_model_xz, self.viewer_model_yz],
            ):
                target = self._find_layer(vm, f"{layer.name}_{orient}")
                if target is not None:
                    target.visible = bool(getattr(layer, "visible", True))

        layer.events.visible.connect(update_visibility)
        self._vis_handlers[id(layer)] = (layer.events.visible, update_visibility)

    def disconnect_layer_visibility(self, layer) -> None:
        """Disconnect the visibility handler and reset the guard flag."""
        layer_id = id(layer)
        entry = self._vis_handlers.pop(layer_id, None)
        if entry is not None:
            event_emitter, handler = entry
            try:
                event_emitter.disconnect(handler)
            except (ValueError, AttributeError):
                pass
        layer._epicycle_vis_synced = False

    def sync_image_style(self, layer: Image) -> None:
        # Guard: only connect once per layer.
        if getattr(layer, "_epicycle_style_synced", False):
            return
        layer._epicycle_style_synced = True

        def update_style(_event):
            for orient, vm in zip(
                ["XY", "XZ", "YZ"],
                [self.viewer_model_xy, self.viewer_model_xz, self.viewer_model_yz],
            ):
                target = self._find_layer(vm, f"{layer.name}_{orient}")
                if not isinstance(target, Image):
                    continue
                target.colormap = layer.colormap
                target.contrast_limits = layer.contrast_limits
                target.opacity = layer.opacity
                target.gamma = layer.gamma

        handlers = []
        for event_attr in ("colormap", "contrast_limits", "opacity", "gamma"):
            emitter = getattr(layer.events, event_attr)
            emitter.connect(update_style)
            handlers.append((emitter, update_style))
        self._style_handlers[id(layer)] = handlers

    def disconnect_image_style(self, layer) -> None:
        """Disconnect image style handlers and reset the guard flag."""
        layer_id = id(layer)
        handlers = self._style_handlers.pop(layer_id, [])
        for event_emitter, handler in handlers:
            try:
                event_emitter.disconnect(handler)
            except (ValueError, AttributeError):
                pass
        layer._epicycle_style_synced = False

    def disconnect_all(self) -> None:
        """Flush ALL tracked handler dicts.

        Called as a belt-and-suspenders after per-layer disconnect loops.
        Disconnects any remaining event-emitter-based handlers and clears
        the dicts so no stale entries survive into the next cycle.
        """
        for layer_id, entry in list(self._vis_handlers.items()):
            if entry is not None:
                event_emitter, handler = entry
                try:
                    event_emitter.disconnect(handler)
                except (ValueError, AttributeError):
                    pass
        self._vis_handlers.clear()

        for layer_id, handlers in list(self._style_handlers.items()):
            for event_emitter, handler in handlers:
                try:
                    event_emitter.disconnect(handler)
                except (ValueError, AttributeError):
                    pass
        self._style_handlers.clear()

    def rename_layer_in_subviewers(self, old_name: str, new_name: str) -> None:
        for orient, vm in zip(
            ["XY", "XZ", "YZ"],
            [self.viewer_model_xy, self.viewer_model_xz, self.viewer_model_yz],
        ):
            src = self._find_layer(vm, f"{old_name}_{orient}")
            if src is not None:
                src.name = f"{new_name}_{orient}"

    def remove_layer_from_subviewers(self, layer_name: str) -> None:
        expected = {f"{layer_name}_{o}" for o in ("XY", "XZ", "YZ")}
        for vm in [self.viewer_model_xy, self.viewer_model_xz, self.viewer_model_yz]:
            for layer in list(vm.layers):
                if layer.name in expected or layer.name == layer_name:
                    # Hide before remove — disconnect the vispy visual from the
                    # scene graph before freeing it (prevents freed-texture
                    # access violations).
                    try:
                        layer.visible = False
                    except Exception:
                        pass
                    vm.layers.remove(layer)

    def get_subviewer_layer(self, base_name: str, orientation: str):
        vm = self._viewer_model_for_orientation(orientation)
        return self._find_layer(vm, f"{base_name}_{str(orientation).upper()}")

    def _viewer_model_for_orientation(self, orientation: str):
        orient = str(orientation).upper()
        if orient == "XY":
            return self.viewer_model_xy
        if orient == "XZ":
            return self.viewer_model_xz
        return self.viewer_model_yz

    def try_fast_reload(self, image_proxies, labels_proxies):
        """Rename + data-swap existing subviewer layers instead of teardown/rebuild.

        Avoids GPU texture allocation by reusing existing layers when shapes
        match.  Falls back gracefully when counts or shapes differ.

        Returns ``(success, old_label_base_names)`` where *old_label_base_names*
        is the list of label base names that were replaced (needed so the
        caller can update the sync manager).
        """
        vms = [self.viewer_model_xy, self.viewer_model_xz, self.viewer_model_yz]
        orients = ["XY", "XZ", "YZ"]

        # --- pre-flight checks ---
        for vm in vms:
            if len(vm.layers) == 0:
                print(f"[FAST-RELOAD] FAIL: {vm} has 0 layers (first load)")
                return False, []

        n_img = len(image_proxies)
        n_lbl = len(labels_proxies)
        for vm in vms:
            existing_images = [l for l in vm.layers if isinstance(l, Image)]
            existing_labels = [l for l in vm.layers if isinstance(l, Labels)]
            if len(existing_images) != n_img or len(existing_labels) != n_lbl:
                print(
                    f"[FAST-RELOAD] FAIL: layer count mismatch — "
                    f"existing img={len(existing_images)} lbl={len(existing_labels)}, "
                    f"new img={n_img} lbl={n_lbl}"
                )
                return False, []

        # --- shape checks ---
        for vm, orient in zip(vms, orients):
            imgs = [l for l in vm.layers if isinstance(l, Image)]
            lbls = [l for l in vm.layers if isinstance(l, Labels)]
            for i, proxy in enumerate(image_proxies):
                new_data = transpose_data(proxy.data, orient)
                if imgs[i].data.shape != new_data.shape:
                    print(
                        f"[FAST-RELOAD] FAIL: image shape mismatch {orient} — "
                        f"existing={imgs[i].data.shape} new={new_data.shape}"
                    )
                    return False, []
            for i, proxy in enumerate(labels_proxies):
                new_data = transpose_data(proxy.data, orient)
                if lbls[i].data.shape != new_data.shape:
                    print(
                        f"[FAST-RELOAD] FAIL: labels shape mismatch {orient} — "
                        f"existing={lbls[i].data.shape} new={new_data.shape}"
                    )
                    return False, []

        # --- fast reload: rename + data swap ---
        old_label_names = []
        is_rgb_cache = {}
        for idx, proxy in enumerate(image_proxies):
            is_rgb = bool(getattr(proxy, "rgb", False))
            if not is_rgb:
                try:
                    arr = proxy.data
                    is_rgb = getattr(arr, "ndim", 0) == 4 and int(arr.shape[-1]) in {3, 4}
                except Exception:
                    is_rgb = False
            is_rgb_cache[idx] = is_rgb

        for vm, orient in zip(vms, orients):
            imgs = [l for l in vm.layers if isinstance(l, Image)]
            for i, proxy in enumerate(image_proxies):
                old_layer = imgs[i]
                old_layer.data = transpose_data(proxy.data, orient)
                old_layer.name = f"{proxy.name}_{orient}"
                # Update style props.
                try:
                    old_layer.opacity = proxy.opacity
                    old_layer.blending = proxy.blending
                except Exception:
                    pass
                # Propagate visibility from the new (main-viewer) proxy.
                # Without this, the in-place rename keeps whatever
                # visibility the previous patch's subviewer image had
                # (e.g. False if user toggled it off with `D`),
                # leaving the new patch's image hidden until manual
                # toggle.  SLOW path doesn't have this issue because
                # it tears down + re-adds (default visible=True).
                try:
                    old_layer.visible = bool(getattr(proxy, "visible", True))
                except Exception:
                    pass
                if not is_rgb_cache.get(i, False):
                    try:
                        cmap = proxy.colormap
                        if hasattr(cmap, "name"):
                            cmap = cmap.name
                        old_layer.colormap = cmap
                        old_layer.contrast_limits = proxy.contrast_limits
                    except Exception:
                        pass

            lbls = [l for l in vm.layers if isinstance(l, Labels)]
            for i, proxy in enumerate(labels_proxies):
                old_layer = lbls[i]
                # Track old base name (only once per label, not per orient).
                if orient == "XY":
                    base = old_layer.name
                    if base.endswith("_XY"):
                        base = base[:-3]
                    if base not in old_label_names:
                        old_label_names.append(base)
                old_layer.data = transpose_data(proxy.data, orient)
                old_layer.name = f"{proxy.name}_{orient}"
                # Update editing props.  brush_size locked at 1 — the
                # lock handler installed on slow-path creation persists
                # through fast reload, but call lock_brush_size_to_1
                # again defensively (idempotent guard short-circuits).
                lock_brush_size_to_1(old_layer)
                # Re-assert the encircle-and-fill block: the fast path reuses
                # the existing layer, and napari re-enables the overlay on
                # mode changes.  Idempotent.
                disable_encircle_fill(old_layer)
                # Clip drawing to existing labels — keep across fast reload too,
                # honouring the source layer's opt-in flag.
                _clip = bool(getattr(proxy, "_epicycle_clip_to_existing", True))
                old_layer.preserve_labels = _clip
                old_layer._epicycle_clip_to_existing = _clip
                old_layer.selected_label = int(getattr(proxy, "selected_label", 1))
                # Propagate visibility — same reasoning as image layers
                # above.  Keeps FAST and SLOW reload paths symmetric.
                try:
                    old_layer.visible = bool(getattr(proxy, "visible", True))
                except Exception:
                    pass
                # Propagate colormap (preserves DirectLabelColormap from QC).
                try:
                    old_layer.colormap = proxy.colormap
                except Exception:
                    pass

        return True, old_label_names

    def clear(self) -> None:
        import traceback as _tb
        n = sum(len(vm.layers) for vm in [self.viewer_model_xy, self.viewer_model_xz, self.viewer_model_yz])
        if n > 0:
            print(f"[SUBVIEWER-CLEAR] clearing {n} subviewer layers — caller stack:")
            _tb.print_stack(limit=10)
        for vm in [self.viewer_model_xy, self.viewer_model_xz, self.viewer_model_yz]:
            for layer in list(vm.layers):
                vm.layers.remove(layer)
